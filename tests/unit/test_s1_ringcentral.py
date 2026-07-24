"""Tests for source polling normalization."""

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from app.models.call_event import CallEvent, CallSource
from app.worker.steps import s1_ingest
from app.worker.steps.s1_ingest import (
    _MAX_UNPROCESSED_BATCH_SIZE,
    _PHONEBURNER_QUERY,
    _RINGCENTRAL_MIN_CALL_DURATION_SECONDS,
    _dedupe_events_by_call_id,
    _cap_unprocessed_batch,
    _filter_unprocessed,
    _is_ringcentral_connected_record,
    _ringcentral_record_to_event,
    _ringcentral_records_to_events,
)


def test_ringcentral_record_maps_patient_to_phone_from() -> None:
    """MedHub outbound calls use the RingCentral `to` party as the patient phone."""

    event = _ringcentral_record_to_event(
        {
            "id": "rc-123",
            "result": "Call connected",
            "duration": 91,
            "direction": "Outbound",
            "from": {"phoneNumber": "+17865550100", "name": "MedHub Agent"},
            "to": {"phoneNumber": "+13055551234", "name": "Patient Name"},
            "recording": {
                "id": "rec-123",
                "contentUri": "https://platform.ringcentral.com/restapi/recording/rec-123",
            },
        }
    )

    assert event.call_id == "rc-123"
    assert event.source is CallSource.RINGCENTRAL
    assert event.workspace == "medhub"
    assert event.phone_from == "+13055551234"
    assert event.phone_to == "+17865550100"
    assert event.patient_phone_primary == "+13055551234"
    assert event.patient_phone_fallback == "+17865550100"
    assert event.agent_id == "MedHub Agent"
    assert event.gcs_audio_uri is None
    assert event.raw_payload["patient_phone_primary"] == "+13055551234"
    assert event.raw_payload["patient_phone_fallback"] == "+17865550100"
    assert event.raw_payload["recording_content_uri"].endswith("/rec-123")
    assert event.raw_payload["recording_id"] == "rec-123"
    assert event.raw_payload["from_name"] == "MedHub Agent"
    assert event.raw_payload["to_name"] == "Patient Name"


def test_ringcentral_filter_requires_connected_duration_and_recording() -> None:
    """Only connected calls meeting duration and recording requirements are processed."""

    record = {
        "result": "Call connected",
        "duration": 15,
        "recording": {"contentUri": "https://example.com/audio.mp3"},
    }

    assert _is_ringcentral_connected_record(
        record,
        min_duration_seconds=_RINGCENTRAL_MIN_CALL_DURATION_SECONDS,
    )
    assert not _is_ringcentral_connected_record(
        {**record, "result": "No Answer"},
        min_duration_seconds=_RINGCENTRAL_MIN_CALL_DURATION_SECONDS,
    )
    assert not _is_ringcentral_connected_record(
        {**record, "duration": 14},
        min_duration_seconds=_RINGCENTRAL_MIN_CALL_DURATION_SECONDS,
    )
    assert not _is_ringcentral_connected_record(
        {**record, "recording": None},
        min_duration_seconds=_RINGCENTRAL_MIN_CALL_DURATION_SECONDS,
    )


def test_ringcentral_records_to_events_logs_skipped_calls(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Skipped RingCentral call-log records should be visible at debug level."""

    records = [
        {
            "id": "connected-call",
            "result": "Call connected",
            "duration": 15,
            "direction": "Outbound",
            "from": {"phoneNumber": "+17865550100"},
            "to": {"phoneNumber": "+13055551234"},
            "recording": {"contentUri": "https://example.com/audio.mp3"},
        },
        {
            "id": "voicemail-call",
            "result": "Voicemail",
            "duration": 60,
            "recording": {"contentUri": "https://example.com/audio.mp3"},
        },
        {
            "id": "short-call",
            "result": "Call connected",
            "duration": 14,
            "recording": {"contentUri": "https://example.com/audio.mp3"},
        },
    ]

    with caplog.at_level("DEBUG"):
        events = _ringcentral_records_to_events(records)

    assert [event.call_id for event in events] == ["connected-call"]
    assert "call_id=voicemail-call result=Voicemail duration=60" in caplog.text
    assert "call_id=short-call result=Call connected duration=14" in caplog.text


async def test_ringcentral_polling_uses_ringcentral_lookback_minutes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RingCentral polling should use its dedicated one-hour lookback setting."""

    captured: dict[str, object] = {}

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz: object = None) -> "FixedDateTime":
            _ = tz
            return cls(2026, 6, 25, 12, 0, tzinfo=UTC)

    class FakeRingCentralAdapter:
        async def list_call_log(
            self,
            *,
            date_from: str,
            recording_type: str,
            per_page: int,
        ) -> list[dict[str, object]]:
            captured["date_from"] = date_from
            captured["recording_type"] = recording_type
            captured["per_page"] = per_page
            return []

    monkeypatch.setattr(s1_ingest, "datetime", FixedDateTime)
    monkeypatch.setattr(s1_ingest, "RingCentralAdapter", FakeRingCentralAdapter)
    monkeypatch.setattr(
        s1_ingest,
        "get_settings",
        lambda: SimpleNamespace(pipeline=SimpleNamespace(ringcentral_lookback_minutes=60)),
    )

    events = await s1_ingest._poll_ringcentral_api()

    assert events == []
    assert captured == {
        "date_from": "2026-06-25T11:00:00Z",
        "recording_type": "All",
        "per_page": 100,
    }


def test_phoneburner_bigquery_polling_has_lookback_window() -> None:
    """PhoneBurner polling should only consider recent calls."""

    assert "connected = @connected" in _PHONEBURNER_QUERY
    assert "duration >= @min_duration_seconds" in _PHONEBURNER_QUERY
    assert "recording_gcs_uri IS NOT NULL" in _PHONEBURNER_QUERY
    assert (
        "end_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @lookback_hours HOUR)"
        in _PHONEBURNER_QUERY
    )
    assert "ORDER BY end_time ASC" in _PHONEBURNER_QUERY


async def test_poll_phoneburner_uses_phoneburner_lookback_hours(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PhoneBurner polling should pass its configured BigQuery lookback parameter."""

    captured: dict[str, object] = {}

    async def fake_query_phoneburner_rows(
        *,
        bq_project: str,
        min_duration_seconds: int,
        lookback_hours: int,
    ) -> list[dict[str, object]]:
        captured["bq_project"] = bq_project
        captured["min_duration_seconds"] = min_duration_seconds
        captured["lookback_hours"] = lookback_hours
        return []

    monkeypatch.setattr(s1_ingest, "_query_phoneburner_rows", fake_query_phoneburner_rows)
    monkeypatch.setattr(
        s1_ingest,
        "get_settings",
        lambda: SimpleNamespace(
            gcp=SimpleNamespace(bq_project="keep-calm-database"),
            pipeline=SimpleNamespace(
                min_call_duration_seconds=30,
                phoneburner_lookback_hours=24,
            ),
        ),
    )

    events = await s1_ingest._poll_phoneburner()

    assert events == []
    assert captured == {
        "bq_project": "keep-calm-database",
        "min_duration_seconds": 30,
        "lookback_hours": 24,
    }


def _call_event(index: int) -> CallEvent:
    """Build a minimal PhoneBurner event for polling tests."""

    return CallEvent(
        call_id=f"call-{index}",
        source=CallSource.PHONEBURNER,
        workspace="intake",
        phone_from="+15550000001",
        phone_to="+15550000002",
        duration_sec=60,
        agent_id=None,
        gcs_audio_uri="gs://bucket/audio.mp3",
        raw_payload={},
    )


def test_unprocessed_batch_cap_warns_and_limits_events(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Large unprocessed batches should be capped after Cloud SQL checks."""

    events = [_call_event(index) for index in range(_MAX_UNPROCESSED_BATCH_SIZE + 1)]

    with caplog.at_level("WARNING"):
        capped = _cap_unprocessed_batch(events)

    assert len(capped) == _MAX_UNPROCESSED_BATCH_SIZE
    assert capped[0].call_id == "call-0"
    assert capped[-1].call_id == f"call-{_MAX_UNPROCESSED_BATCH_SIZE - 1}"
    assert "exceeds batch cap" in caplog.text


def test_dedupe_events_by_call_id_keeps_first_event(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Duplicate call IDs in one source batch should only be processed once."""

    first = _call_event(1)
    duplicate = _call_event(1).model_copy(update={"phone_to": "+15550000999"})
    second = _call_event(2)

    with caplog.at_level("WARNING"):
        deduped = _dedupe_events_by_call_id([first, duplicate, second])

    assert deduped == [first, second]
    assert "Dropped 1 duplicate call_id" in caplog.text


async def test_filter_unprocessed_uses_batch_processed_call_id_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Idempotency filtering should use one batch lookup for all candidate IDs."""

    captured_call_ids: list[str] = []

    async def fake_processed_call_ids(call_ids: list[str]) -> set[str]:
        captured_call_ids.extend(call_ids)
        return {"call-1"}

    monkeypatch.setattr(s1_ingest, "processed_call_ids", fake_processed_call_ids)

    events = [_call_event(index) for index in range(3)]
    unprocessed = await _filter_unprocessed(events)

    assert captured_call_ids == ["call-0", "call-1", "call-2"]
    assert [event.call_id for event in unprocessed] == ["call-0", "call-2"]


async def test_poll_new_calls_filters_idempotency_before_capping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 200-call cap should apply after Cloud SQL idempotency checks."""

    captured_filter_count = 0

    async def fake_poll_phoneburner() -> list[CallEvent]:
        return [_call_event(index) for index in range(_MAX_UNPROCESSED_BATCH_SIZE + 1)]

    async def fake_poll_ringcentral_api() -> list[CallEvent]:
        return []

    async def fake_filter_unprocessed(events: list[CallEvent]) -> list[CallEvent]:
        nonlocal captured_filter_count
        captured_filter_count = len(events)
        return events

    monkeypatch.setattr(s1_ingest, "_poll_phoneburner", fake_poll_phoneburner)
    monkeypatch.setattr(s1_ingest, "_poll_ringcentral_api", fake_poll_ringcentral_api)
    monkeypatch.setattr(s1_ingest, "_filter_unprocessed", fake_filter_unprocessed)

    events = await s1_ingest.poll_new_calls()

    assert captured_filter_count == _MAX_UNPROCESSED_BATCH_SIZE + 1
    assert len(events) == _MAX_UNPROCESSED_BATCH_SIZE


async def test_poll_new_calls_deduplicates_before_idempotency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Duplicate source call IDs should not reach expensive pipeline steps."""

    captured_call_ids: list[str] = []

    async def fake_poll_phoneburner() -> list[CallEvent]:
        return [_call_event(1), _call_event(1), _call_event(2)]

    async def fake_poll_ringcentral_api() -> list[CallEvent]:
        return []

    async def fake_filter_unprocessed(events: list[CallEvent]) -> list[CallEvent]:
        captured_call_ids.extend(event.call_id for event in events)
        return events

    monkeypatch.setattr(s1_ingest, "_poll_phoneburner", fake_poll_phoneburner)
    monkeypatch.setattr(s1_ingest, "_poll_ringcentral_api", fake_poll_ringcentral_api)
    monkeypatch.setattr(s1_ingest, "_filter_unprocessed", fake_filter_unprocessed)

    events = await s1_ingest.poll_new_calls()

    assert captured_call_ids == ["call-1", "call-2"]
    assert [event.call_id for event in events] == ["call-1", "call-2"]
