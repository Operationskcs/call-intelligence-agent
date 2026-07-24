"""Step 1: legacy polling helpers and idempotency checks.

Webhook-triggered processing is now the primary ingestion path. The polling
helpers remain available for manual reprocessing and the existing scheduler job
until webhook production validation is complete.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any

from app.adapters.telephony.ringcentral import RingCentralAdapter
from app.config import get_settings
from app.models.call_event import CallEvent, CallSource
from app.storage.audit import PROCESSED_CALL_IDS_QUERY, is_processed, processed_call_ids

logger = logging.getLogger(__name__)

_PHONEBURNER_QUERY = """
SELECT *
FROM `phoneburner_logs.call_events`
WHERE connected = @connected
  AND duration >= @min_duration_seconds
  AND recording_gcs_uri IS NOT NULL
  AND end_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @lookback_hours HOUR)
ORDER BY end_time ASC
"""
_MAX_UNPROCESSED_BATCH_SIZE = 200
_RINGCENTRAL_CONNECTED_RESULT = "Call connected"
_RINGCENTRAL_MIN_CALL_DURATION_SECONDS = 15


async def poll_new_calls() -> list[CallEvent]:
    """Legacy: poll call sources for recent connected calls and return unprocessed events."""

    phoneburner_events, ringcentral_events = await asyncio.gather(
        _poll_phoneburner(),
        _poll_ringcentral_api(),
    )

    events = [*phoneburner_events, *ringcentral_events]
    events = _dedupe_events_by_call_id(events)
    unprocessed_events = await _filter_unprocessed(events)
    return _cap_unprocessed_batch(unprocessed_events)


async def poll_legacy_phoneburner_calls() -> list[CallEvent]:
    """Legacy: poll only PhoneBurner calls for manual Intake reprocessing."""

    events = await _poll_phoneburner()
    events = _dedupe_events_by_call_id(events)
    unprocessed_events = await _filter_unprocessed(events)
    return _cap_unprocessed_batch(unprocessed_events)


async def check_idempotency(call_id: str) -> bool:
    """Query call_audit_log and return True when call_id was already processed."""

    return await is_processed(call_id)


async def _poll_phoneburner() -> list[CallEvent]:
    """Poll PhoneBurner calls from BigQuery."""

    settings = get_settings()
    rows = await _query_phoneburner_rows(
        bq_project=settings.gcp.bq_project,
        min_duration_seconds=settings.pipeline.min_call_duration_seconds,
        lookback_hours=settings.pipeline.phoneburner_lookback_hours,
    )
    return [_phoneburner_row_to_event(row) for row in rows]


async def _poll_ringcentral_api() -> list[CallEvent]:
    """Poll RingCentral call logs through the API for recent MedHub recordings."""

    settings = get_settings()
    adapter = RingCentralAdapter()
    date_from = _isoformat_utc(
        datetime.now(UTC) - timedelta(minutes=settings.pipeline.ringcentral_lookback_minutes)
    )
    records = await adapter.list_call_log(
        date_from=date_from,
        recording_type="All",
        per_page=100,
    )
    return _ringcentral_records_to_events(records)


async def _query_phoneburner_rows(
    *,
    bq_project: str,
    min_duration_seconds: int,
    lookback_hours: int,
) -> list[dict[str, Any]]:
    """Query all eligible PhoneBurner connected call rows."""

    return await _query_rows(
        bq_project=bq_project,
        query=_PHONEBURNER_QUERY,
        query_parameters=[
            _scalar_query_parameter("connected", "BOOL", True),
            _scalar_query_parameter("min_duration_seconds", "INT64", min_duration_seconds),
            _scalar_query_parameter("lookback_hours", "INT64", lookback_hours),
        ],
    )


async def _query_rows(
    *,
    bq_project: str,
    query: str,
    query_parameters: Sequence[Any],
) -> list[dict[str, Any]]:
    """Run a parameterized BigQuery query in a thread."""

    if not bq_project:
        raise RuntimeError("BQ_PROJECT must be configured for BigQuery polling.")

    logger.info(
        "BigQuery candidate call polling SQL:\n%s",
        query.strip(),
    )
    return await asyncio.to_thread(
        _query_rows_sync,
        bq_project,
        query,
        query_parameters,
    )


def _query_rows_sync(
    bq_project: str,
    query: str,
    query_parameters: Sequence[Any],
) -> list[dict[str, Any]]:
    """Run a synchronous BigQuery query and return JSON-friendly rows."""

    from google.cloud import bigquery

    client = bigquery.Client(project=bq_project)
    job_config = bigquery.QueryJobConfig(query_parameters=list(query_parameters))
    rows = client.query(query, job_config=job_config).result()
    return [_row_to_dict(row) for row in rows]


def _scalar_query_parameter(name: str, type_: str, value: Any) -> Any:
    """Create a BigQuery scalar query parameter without importing BigQuery at module load."""

    from google.cloud import bigquery

    return bigquery.ScalarQueryParameter(name, type_, value)


async def _filter_unprocessed(events: list[CallEvent]) -> list[CallEvent]:
    """Remove calls that already have successful audit-log rows."""

    call_ids = [event.call_id for event in events]
    logger.info(
        "Cloud SQL call_audit_log batch idempotency SQL used to exclude "
        "already-processed call IDs:\n%s\nCriterion: call_id is processed when "
        "processed_at IS NOT NULL AND error_message IS NULL. Checking %d "
        "candidate call_id(s) in one PostgreSQL array query.",
        PROCESSED_CALL_IDS_QUERY.strip(),
        len(call_ids),
    )
    already_processed_call_ids = await processed_call_ids(call_ids)
    return [event for event in events if event.call_id not in already_processed_call_ids]


def _dedupe_events_by_call_id(events: list[CallEvent]) -> list[CallEvent]:
    """Keep the first event for each call_id in one polling batch."""

    seen_call_ids: set[str] = set()
    deduped_events: list[CallEvent] = []
    duplicate_count = 0

    for event in events:
        if event.call_id in seen_call_ids:
            duplicate_count += 1
            continue

        seen_call_ids.add(event.call_id)
        deduped_events.append(event)

    if duplicate_count:
        logger.warning(
            "Dropped %d duplicate call_id event(s) from polling batch before "
            "idempotency filtering.",
            duplicate_count,
        )

    return deduped_events


def _cap_unprocessed_batch(events: list[CallEvent]) -> list[CallEvent]:
    """Cap large unprocessed batches after Cloud SQL idempotency checks."""

    if len(events) <= _MAX_UNPROCESSED_BATCH_SIZE:
        return events

    logger.warning(
        "Unprocessed call count %d exceeds batch cap %d; processing the first %d "
        "event(s) this run and leaving the rest for later runs.",
        len(events),
        _MAX_UNPROCESSED_BATCH_SIZE,
        _MAX_UNPROCESSED_BATCH_SIZE,
    )
    return events[:_MAX_UNPROCESSED_BATCH_SIZE]


def _phoneburner_row_to_event(row: dict[str, Any]) -> CallEvent:
    """Map a PhoneBurner BigQuery row to the normalized CallEvent model."""

    raw_payload = _with_matching_hints(row)
    agent_name = _join_name(
        _nested_str(raw_payload, "agent", "first_name"),
        _nested_str(raw_payload, "agent", "last_name"),
    )

    return CallEvent(
        call_id=str(_required_value(raw_payload, "call_id")),
        source=CallSource.PHONEBURNER,
        workspace="intake",
        phone_from=_nested_str(raw_payload, "contact", "phone"),
        phone_to=_str_value(raw_payload.get("endpoint")),
        duration_sec=_int_value(raw_payload.get("duration")),
        agent_id=agent_name or None,
        gcs_audio_uri=_str_value(raw_payload.get("recording_gcs_uri")) or None,
        raw_payload=raw_payload,
    )


def _ringcentral_record_to_event(record: dict[str, Any]) -> CallEvent:
    """Map a RingCentral API call-log record to the normalized CallEvent model."""

    from_party = _dict_value(record.get("from"))
    to_party = _dict_value(record.get("to"))
    recording = _dict_value(record.get("recording"))
    from_name = _str_value(from_party.get("name"))
    to_name = _str_value(to_party.get("name"))
    patient_phone_primary = _str_value(to_party.get("phoneNumber"))
    patient_phone_fallback = _str_value(from_party.get("phoneNumber"))

    return CallEvent(
        call_id=_str_value(_required_value(record, "id")),
        source=CallSource.RINGCENTRAL,
        workspace="medhub",
        phone_from=patient_phone_primary,
        phone_to=patient_phone_fallback,
        patient_phone_primary=patient_phone_primary or None,
        patient_phone_fallback=patient_phone_fallback or None,
        duration_sec=_int_value(record.get("duration")),
        agent_id=from_name or None,
        gcs_audio_uri=None,
        raw_payload={
            "patient_phone_primary": patient_phone_primary,
            "patient_phone_fallback": patient_phone_fallback,
            "recording_content_uri": _str_value(recording.get("contentUri")),
            "recording_id": _str_value(recording.get("id")),
            "from_name": from_name,
            "to_name": to_name,
            "direction": _str_value(record.get("direction")),
        },
    )


def _ringcentral_records_to_events(records: Sequence[dict[str, Any]]) -> list[CallEvent]:
    """Map eligible RingCentral records to events and debug-log skipped records."""

    events: list[CallEvent] = []
    for record in records:
        if _is_ringcentral_connected_record(
            record,
            min_duration_seconds=_RINGCENTRAL_MIN_CALL_DURATION_SECONDS,
        ):
            events.append(_ringcentral_record_to_event(record))
            continue

        logger.debug(
            "Skipping RingCentral call. call_id=%s result=%s duration=%s",
            _str_value(record.get("id")),
            _str_value(record.get("result")),
            _int_value(record.get("duration")),
        )

    return events


def _is_ringcentral_connected_record(
    record: dict[str, Any],
    *,
    min_duration_seconds: int,
) -> bool:
    """Return True when a RingCentral record is eligible for processing."""

    recording = _dict_value(record.get("recording"))
    return (
        _str_value(record.get("result")) == _RINGCENTRAL_CONNECTED_RESULT
        and _int_value(record.get("duration")) >= min_duration_seconds
        and bool(_str_value(recording.get("contentUri")))
    )


def _with_matching_hints(row: dict[str, Any]) -> dict[str, Any]:
    """Add normalized fallback matching fields while preserving the full BigQuery row."""

    raw_payload = dict(row)
    contact_name = _join_name(
        _nested_str(raw_payload, "contact", "first_name"),
        _nested_str(raw_payload, "contact", "last_name"),
    )
    if contact_name and not raw_payload.get("name"):
        raw_payload["name"] = contact_name

    contact_email = _nested_str(raw_payload, "contact", "primary_email")
    if contact_email and not raw_payload.get("email"):
        raw_payload["email"] = contact_email

    return raw_payload


def _row_to_dict(row: Any) -> dict[str, Any]:
    """Convert a BigQuery Row into a JSON-friendly dictionary."""

    items = row.items() if hasattr(row, "items") else dict(row).items()
    return {str(key): _plain_value(value) for key, value in items}


def _plain_value(value: Any) -> Any:
    """Convert BigQuery values into JSON-friendly Python values."""

    if isinstance(value, Mapping):
        return {str(key): _plain_value(item) for key, item in value.items()}
    if hasattr(value, "items"):
        return {str(key): _plain_value(item) for key, item in value.items()}
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Decimal):
        if value.is_finite() and value == value.to_integral_value():
            return int(value)
        return float(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain_value(item) for item in value]
    return value


def _nested_str(payload: dict[str, Any], *keys: str) -> str:
    """Read a nested field as a string."""

    value: Any = payload
    for key in keys:
        if not isinstance(value, Mapping):
            return ""
        value = value.get(key)
    return _str_value(value)


def _dict_value(value: Any) -> dict[str, Any]:
    """Return a mapping payload as a plain dictionary."""

    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    return {}


def _required_value(payload: dict[str, Any], key: str) -> Any:
    """Read a required BigQuery field."""

    value = payload.get(key)
    if value is None or value == "":
        raise ValueError(f"BigQuery row is missing required field {key!r}.")
    return value


def _join_name(first_name: str, last_name: str) -> str:
    """Join nullable first and last name parts."""

    return " ".join(part for part in (first_name.strip(), last_name.strip()) if part)


def _str_value(value: Any) -> str:
    """Convert nullable BigQuery values into strings."""

    if value is None:
        return ""
    return str(value).strip()


def _int_value(value: Any) -> int:
    """Convert BigQuery numeric values into integers."""

    if isinstance(value, Decimal):
        return int(value)
    if value is None or value == "":
        return 0
    return int(value)


def _isoformat_utc(value: datetime) -> str:
    """Format a datetime for RingCentral API date filters."""

    return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
