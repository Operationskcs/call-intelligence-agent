"""RingCentral webhook receiver for MedHub call completion events."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Request, Response, status

from app.adapters.telephony.ringcentral import RingCentralAdapter
from app.config import get_settings
from app.models.call_event import CallEvent, CallSource
from app.storage import audit
from app.worker import pipeline
from app.worker.steps.s1_ingest import _isoformat_utc, _ringcentral_records_to_events

logger = logging.getLogger(__name__)

router = APIRouter()

_COMPLETED_RECORDING_STATUS = "Completed"
_CONNECTED_RESULT = "Call connected"
_DISCONNECTED_STATUS_CODE = "Disconnected"
_MIN_DURATION_SECONDS = 15


@router.post(
    "/webhook/ringcentral",
    status_code=status.HTTP_200_OK,
    response_model=None,
)
async def receive_ringcentral_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    validation_token: Annotated[str | None, Header(alias="Validation-Token")] = None,
) -> Response | dict[str, Any]:
    """Accept RingCentral validation challenges and call-completion notifications."""

    if validation_token:
        return Response(
            status_code=status.HTTP_200_OK,
            headers={"Validation-Token": validation_token},
        )

    payload = await _json_object(request)
    return await handle_ringcentral_notification(payload, background_tasks)


async def handle_ringcentral_notification(
    payload: dict[str, Any],
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    """Parse, validate, dedupe, and schedule pipeline processing for a notification."""

    event = parse_call_event(payload)
    if event is None:
        disconnected_result = await _handle_disconnected_telephony_session(
            payload,
            background_tasks,
        )
        if disconnected_result is not None:
            return disconnected_result
        return {"status": "skipped"}

    return await _schedule_single_event(event, background_tasks)


async def _schedule_single_event(
    event: CallEvent,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    """Schedule one event unless it has already reached a terminal audit state."""

    if await audit.is_processed(event.call_id):
        return {"status": "duplicate", "call_id": event.call_id}

    background_tasks.add_task(pipeline.run, event)
    return {"status": "accepted", "call_id": event.call_id}


async def handle_ringcentral_webhook(
    payload: dict[str, Any],
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    """Handle an already-authenticated RingCentral notification payload."""

    return await handle_ringcentral_notification(payload, background_tasks)


async def _handle_disconnected_telephony_session(
    payload: dict[str, Any],
    background_tasks: BackgroundTasks,
) -> dict[str, Any] | None:
    """Fetch Call Log records when a telephony session reports a final disconnect."""

    record = _notification_record(payload)
    if _DISCONNECTED_STATUS_CODE not in _party_status_code_set(record):
        return None

    telephony_session_id = _str_value(record.get("telephonySessionId"))
    if not telephony_session_id:
        logger.warning("RingCentral disconnected webhook missing telephonySessionId.")
        return {"status": "skipped", "reason": "missing_telephony_session_id"}

    try:
        events = await _call_log_events_for_telephony_session(telephony_session_id)
    except Exception:
        logger.exception(
            "RingCentral call-log fetch failed for disconnected webhook. "
            "has_telephony_session_id=%s",
            bool(telephony_session_id),
        )
        return {"status": "skipped", "reason": "call_log_fetch_failed"}

    if not events:
        return {"status": "skipped", "reason": "recording_not_ready_yet"}

    accepted_call_ids: list[str] = []
    duplicate_call_ids: list[str] = []
    for event in events:
        result = await _schedule_single_event(event, background_tasks)
        if result["status"] == "accepted":
            accepted_call_ids.append(event.call_id)
        elif result["status"] == "duplicate":
            duplicate_call_ids.append(event.call_id)

    if accepted_call_ids:
        return {
            "status": "accepted",
            "accepted": len(accepted_call_ids),
            "duplicates": len(duplicate_call_ids),
            "call_ids": accepted_call_ids,
        }

    return {
        "status": "duplicate",
        "duplicates": len(duplicate_call_ids),
        "call_ids": duplicate_call_ids,
    }


async def _call_log_events_for_telephony_session(
    telephony_session_id: str,
) -> list[CallEvent]:
    """Fetch and normalize eligible Call Log records for one telephony session."""

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
    matching_records = [
        record
        for record in records
        if _str_value(record.get("telephonySessionId")) == telephony_session_id
    ]
    return _ringcentral_records_to_events(matching_records)


def parse_call_event(payload: dict[str, Any]) -> CallEvent | None:
    """Map an eligible RingCentral push notification payload into a CallEvent."""

    record = _notification_record(payload)
    recording = _dict_value(record.get("recording"))

    if _str_value(recording.get("status")) != _COMPLETED_RECORDING_STATUS:
        _log_ringcentral_skip("recording_not_completed", payload, record, recording)
        return None

    result = _str_value(record.get("result"))
    duration_sec = _int_value(record.get("duration"))
    if result != _CONNECTED_RESULT:
        _log_ringcentral_skip(
            "result_not_connected",
            payload,
            record,
            recording,
            result,
            duration_sec,
        )
        return None

    if duration_sec < _MIN_DURATION_SECONDS:
        _log_ringcentral_skip(
            "duration_below_threshold",
            payload,
            record,
            recording,
            result,
            duration_sec,
        )
        return None

    call_id = _call_id(record)
    if not call_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing RingCentral call_id",
        )

    recording_content_uri = _str_value(recording.get("contentUri"))
    if not recording_content_uri:
        _log_ringcentral_skip(
            "missing_recording_content_uri",
            payload,
            record,
            recording,
            result,
            duration_sec,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing RingCentral recording.contentUri",
        )

    from_party = _dict_value(record.get("from"))
    to_party = _dict_value(record.get("to"))
    from_phone = _str_value(from_party.get("phoneNumber"))
    to_phone = _str_value(to_party.get("phoneNumber"))
    from_name = _str_value(from_party.get("name"))
    to_name = _str_value(to_party.get("name"))

    raw_payload = dict(payload)
    raw_payload.update(
        {
            "call_id": call_id,
            "result": result,
            "duration": duration_sec,
            "from_phone_number": from_phone,
            "to_phone_number": to_phone,
            "patient_phone_primary": to_phone,
            "patient_phone_fallback": from_phone,
            "recording_content_uri": recording_content_uri,
            "recording_id": _str_value(recording.get("id")),
            "recording_status": _COMPLETED_RECORDING_STATUS,
            "from_name": from_name,
            "to_name": to_name,
            "direction": _str_value(record.get("direction")),
        }
    )

    return CallEvent(
        call_id=call_id,
        source=CallSource.RINGCENTRAL,
        workspace="medhub",
        phone_from=to_phone,
        phone_to=from_phone,
        patient_phone_primary=to_phone or None,
        patient_phone_fallback=from_phone or None,
        duration_sec=duration_sec,
        agent_id=from_name or None,
        agent_name=from_name or None,
        gcs_audio_uri=None,
        raw_payload=raw_payload,
    )


def _log_ringcentral_skip(
    reason: str,
    payload: Mapping[str, Any],
    record: Mapping[str, Any],
    recording: Mapping[str, Any],
    result: str = "",
    duration_sec: int = 0,
) -> None:
    """Log a PII-safe RingCentral skip summary for temporary diagnostics."""

    logger.debug(
        "RingCentral webhook skipped. reason=%s has_body=%s has_recording=%s "
        "recording_status=%s result=%s duration_sec=%s has_recording_content_uri=%s "
        "has_parties=%s party_status_codes=%s has_telephony_session_id=%s",
        reason,
        isinstance(payload.get("body"), Mapping),
        bool(recording),
        _str_value(recording.get("status")),
        result or _str_value(record.get("result")),
        duration_sec or _int_value(record.get("duration")),
        bool(_str_value(recording.get("contentUri"))),
        isinstance(record.get("parties"), list),
        _party_status_codes(record),
        bool(
            _str_value(record.get("telephonySessionId"))
            or _str_value(record.get("sessionId"))
        ),
    )


async def _json_object(request: Request) -> dict[str, Any]:
    """Parse the request body and require a JSON object."""

    payload = await request.json()
    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Expected JSON object",
        )
    return payload


def _party_status_codes(record: Mapping[str, Any]) -> str:
    """Return unique RingCentral party status codes without logging party details."""

    return ",".join(sorted(_party_status_code_set(record)))


def _party_status_code_set(record: Mapping[str, Any]) -> set[str]:
    """Return unique RingCentral party status codes without party details."""

    parties = record.get("parties")
    if not isinstance(parties, list):
        return set()

    codes: set[str] = set()
    for party in parties:
        party_record = _dict_value(party)
        status_record = _dict_value(party_record.get("status"))
        code = _str_value(status_record.get("code"))
        if code:
            codes.add(code)
    return codes


def _notification_record(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the call-log shaped object from a RingCentral notification payload."""

    body = payload.get("body")
    if isinstance(body, Mapping):
        return {str(key): value for key, value in body.items()}
    return payload


def _call_id(record: Mapping[str, Any]) -> str:
    """Extract the most stable RingCentral call identifier available."""

    return (
        _str_value(record.get("call_id"))
        or _str_value(record.get("callId"))
        or _str_value(record.get("id"))
        or _str_value(record.get("telephonySessionId"))
        or _str_value(record.get("sessionId"))
    )


def _dict_value(value: Any) -> dict[str, Any]:
    """Return a mapping payload as a plain dictionary."""

    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    return {}


def _str_value(value: Any) -> str:
    """Convert nullable vendor values into trimmed strings."""

    if value is None:
        return ""
    return str(value).strip()


def _int_value(value: Any) -> int:
    """Convert nullable vendor duration values into integer seconds."""

    if value is None or value == "":
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
