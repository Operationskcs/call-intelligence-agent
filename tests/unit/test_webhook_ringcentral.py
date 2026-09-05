"""Tests for the RingCentral webhook handler."""

from types import SimpleNamespace
from typing import Any, cast

import pytest
from fastapi import BackgroundTasks, FastAPI
from fastapi.testclient import TestClient

from app.models.call_event import CallEvent, CallSource
from app.webhook import ringcentral
from app.worker.steps.s1_ingest import _ringcentral_record_to_event


def test_validation_token_handshake_echoes_header() -> None:
    """RingCentral subscription validation receives the token in a response header."""

    app = FastAPI()
    app.include_router(ringcentral.router)
    client = TestClient(app)

    response = client.post(
        "/webhook/ringcentral",
        headers={"Validation-Token": "validation-token-123"},
    )

    assert response.status_code == 200
    assert response.headers["Validation-Token"] == "validation-token-123"
    assert response.content == b""


def test_parse_completed_connected_recording_payload() -> None:
    """Eligible RingCentral notifications become MedHub RingCentral CallEvents."""

    event = ringcentral.parse_call_event(_payload())

    assert event is not None
    assert event.call_id == "rc-call-123"
    assert event.source is CallSource.RINGCENTRAL
    assert event.workspace == "medhub"
    assert event.phone_from == "+13055551234"
    assert event.phone_to == "+17865550100"
    assert event.patient_phone_primary == "+13055551234"
    assert event.patient_phone_fallback == "+17865550100"
    assert event.duration_sec == 61
    assert event.agent_id == "MedHub Agent"
    assert event.agent_name == "MedHub Agent"
    assert event.gcs_audio_uri is None
    assert event.raw_payload["recording_content_uri"] == "https://platform.ringcentral.test/rec-1"
    assert event.raw_payload["from_phone_number"] == "+17865550100"
    assert event.raw_payload["to_phone_number"] == "+13055551234"
    assert event.raw_payload["result"] == "Call connected"


@pytest.mark.parametrize(
    ("overrides", "recording_overrides"),
    [
        ({"result": "No Answer"}, {}),
        ({"duration": 14}, {}),
        ({}, {"status": "InProgress"}),
    ],
)
def test_parse_skips_ineligible_notifications(
    overrides: dict[str, Any],
    recording_overrides: dict[str, Any],
) -> None:
    """Disconnected, too-short, or incomplete recordings are ignored."""

    payload = _payload(overrides=overrides, recording_overrides=recording_overrides)

    assert ringcentral.parse_call_event(payload) is None


async def test_handler_skips_duplicate_before_scheduling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Already processed calls should not enqueue pipeline work."""

    background_tasks = BackgroundTasks()

    async def fake_is_processed(call_id: str) -> bool:
        assert call_id == "rc-call-123"
        return True

    monkeypatch.setattr("app.webhook.ringcentral.audit.is_processed", fake_is_processed)

    result = await ringcentral.handle_ringcentral_notification(_payload(), background_tasks)

    assert result == {"status": "duplicate", "call_id": "rc-call-123"}
    assert background_tasks.tasks == []


async def test_handler_schedules_pipeline_in_background(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Eligible new calls are scheduled without awaiting pipeline.run."""

    background_tasks = BackgroundTasks()
    ran_events: list[CallEvent] = []

    async def fake_is_processed(call_id: str) -> bool:
        assert call_id == "rc-call-123"
        return False

    async def fake_run(event: CallEvent) -> None:
        ran_events.append(event)

    monkeypatch.setattr("app.webhook.ringcentral.audit.is_processed", fake_is_processed)
    monkeypatch.setattr("app.webhook.ringcentral.pipeline.run", fake_run)

    result = await ringcentral.handle_ringcentral_notification(_payload(), background_tasks)

    assert result == {"status": "accepted", "call_id": "rc-call-123"}
    assert ran_events == []
    assert len(background_tasks.tasks) == 1
    task = background_tasks.tasks[0]
    assert task.func is fake_run
    scheduled_event = cast(CallEvent, task.args[0])
    assert scheduled_event.call_id == "rc-call-123"


async def test_handler_skips_invalid_payload_without_idempotency_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Filtered notifications should return 200-style payloads and avoid scheduling."""

    background_tasks = BackgroundTasks()

    async def fail_is_processed(call_id: str) -> bool:
        raise AssertionError(f"idempotency should not be checked for {call_id}")

    monkeypatch.setattr("app.webhook.ringcentral.audit.is_processed", fail_is_processed)

    result = await ringcentral.handle_ringcentral_notification(
        _payload(overrides={"duration": 10}),
        background_tasks,
    )

    assert result == {"status": "skipped"}
    assert background_tasks.tasks == []


@pytest.mark.parametrize("status_code", ["Setup", "Proceeding", "Answered", "Voicemail", "Gone"])
async def test_telephony_session_fetch_only_runs_for_disconnected(
    monkeypatch: pytest.MonkeyPatch,
    status_code: str,
) -> None:
    """Non-terminal telephony session states should not call the Call Log API."""

    background_tasks = BackgroundTasks()

    class FailRingCentralAdapter:
        async def list_call_log(self, **kwargs: object) -> list[dict[str, object]]:
            raise AssertionError(f"Unexpected Call Log fetch: {kwargs}")

    async def fail_is_processed(call_id: str) -> bool:
        raise AssertionError(f"idempotency should not be checked for {call_id}")

    monkeypatch.setattr(ringcentral, "RingCentralAdapter", FailRingCentralAdapter)
    monkeypatch.setattr(ringcentral.audit, "is_processed", fail_is_processed)

    result = await ringcentral.handle_ringcentral_notification(
        _telephony_session_payload(status_code=status_code),
        background_tasks,
    )

    assert result == {"status": "skipped"}
    assert background_tasks.tasks == []


async def test_disconnected_fetches_call_log_and_schedules_polling_shaped_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Disconnected sessions should fetch Call Log and schedule events using record.id."""

    background_tasks = BackgroundTasks()
    captured: dict[str, object] = {}
    call_log_record = _call_log_record()
    ran_events: list[CallEvent] = []

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
            return [call_log_record]

    async def fake_is_processed(call_id: str) -> bool:
        assert call_id == "call-log-123"
        return False

    async def fake_run(event: CallEvent) -> None:
        ran_events.append(event)

    monkeypatch.setattr(ringcentral, "RingCentralAdapter", FakeRingCentralAdapter)
    monkeypatch.setattr(
        ringcentral,
        "get_settings",
        lambda: SimpleNamespace(pipeline=SimpleNamespace(ringcentral_lookback_minutes=5)),
    )
    monkeypatch.setattr(ringcentral.audit, "is_processed", fake_is_processed)
    monkeypatch.setattr(ringcentral.pipeline, "run", fake_run)

    result = await ringcentral.handle_ringcentral_notification(
        _telephony_session_payload(),
        background_tasks,
    )

    assert result == {
        "status": "accepted",
        "accepted": 1,
        "duplicates": 0,
        "call_ids": ["call-log-123"],
    }
    assert captured["recording_type"] == "All"
    assert captured["per_page"] == 100
    assert str(captured["date_from"]).endswith("Z")
    assert ran_events == []
    assert len(background_tasks.tasks) == 1
    task = background_tasks.tasks[0]
    assert task.func is fake_run

    scheduled_event = cast(CallEvent, task.args[0])
    polled_event = _ringcentral_record_to_event(call_log_record)
    assert scheduled_event.call_id == polled_event.call_id == "call-log-123"
    assert scheduled_event.raw_payload["recording_content_uri"].endswith("/rec-123/content")


async def test_disconnected_returns_skipped_when_recording_is_not_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A matching Call Log record without contentUri should leave polling as fallback."""

    background_tasks = BackgroundTasks()

    class FakeRingCentralAdapter:
        async def list_call_log(self, **kwargs: object) -> list[dict[str, object]]:
            _ = kwargs
            return [_call_log_record(recording={})]

    async def fail_is_processed(call_id: str) -> bool:
        raise AssertionError(f"idempotency should not be checked for {call_id}")

    monkeypatch.setattr(ringcentral, "RingCentralAdapter", FakeRingCentralAdapter)
    monkeypatch.setattr(
        ringcentral,
        "get_settings",
        lambda: SimpleNamespace(pipeline=SimpleNamespace(ringcentral_lookback_minutes=5)),
    )
    monkeypatch.setattr(ringcentral.audit, "is_processed", fail_is_processed)

    result = await ringcentral.handle_ringcentral_notification(
        _telephony_session_payload(),
        background_tasks,
    )

    assert result == {"status": "skipped", "reason": "recording_not_ready_yet"}
    assert background_tasks.tasks == []


async def test_disconnected_schedules_multiple_matching_call_log_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Multiple eligible records for one telephony session should all be scheduled."""

    background_tasks = BackgroundTasks()
    records = [
        _call_log_record(call_id="call-log-1", recording_id="rec-1"),
        _call_log_record(call_id="call-log-2", recording_id="rec-2"),
        _call_log_record(call_id="other-session", telephony_session_id="other-session"),
    ]

    class FakeRingCentralAdapter:
        async def list_call_log(self, **kwargs: object) -> list[dict[str, object]]:
            _ = kwargs
            return records

    async def fake_is_processed(call_id: str) -> bool:
        _ = call_id
        return False

    async def fake_run(event: CallEvent) -> None:
        _ = event

    monkeypatch.setattr(ringcentral, "RingCentralAdapter", FakeRingCentralAdapter)
    monkeypatch.setattr(
        ringcentral,
        "get_settings",
        lambda: SimpleNamespace(pipeline=SimpleNamespace(ringcentral_lookback_minutes=5)),
    )
    monkeypatch.setattr(ringcentral.audit, "is_processed", fake_is_processed)
    monkeypatch.setattr(ringcentral.pipeline, "run", fake_run)

    result = await ringcentral.handle_ringcentral_notification(
        _telephony_session_payload(),
        background_tasks,
    )

    assert result == {
        "status": "accepted",
        "accepted": 2,
        "duplicates": 0,
        "call_ids": ["call-log-1", "call-log-2"],
    }
    assert [cast(CallEvent, task.args[0]).call_id for task in background_tasks.tasks] == [
        "call-log-1",
        "call-log-2",
    ]


async def test_disconnected_call_log_fetch_error_is_logged_and_skipped(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """RingCentral API errors should not make the webhook request fail."""

    background_tasks = BackgroundTasks()

    class FailingRingCentralAdapter:
        async def list_call_log(self, **kwargs: object) -> list[dict[str, object]]:
            _ = kwargs
            raise RuntimeError("429 Too Many Requests")

    monkeypatch.setattr(ringcentral, "RingCentralAdapter", FailingRingCentralAdapter)
    monkeypatch.setattr(
        ringcentral,
        "get_settings",
        lambda: SimpleNamespace(pipeline=SimpleNamespace(ringcentral_lookback_minutes=5)),
    )

    with caplog.at_level("ERROR"):
        result = await ringcentral.handle_ringcentral_notification(
            _telephony_session_payload(),
            background_tasks,
        )

    assert result == {"status": "skipped", "reason": "call_log_fetch_failed"}
    assert background_tasks.tasks == []
    assert "RingCentral call-log fetch failed for disconnected webhook" in caplog.text


def _payload(
    *,
    overrides: dict[str, Any] | None = None,
    recording_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    recording = {
        "id": "recording-1",
        "status": "Completed",
        "contentUri": "https://platform.ringcentral.test/rec-1",
    }
    if recording_overrides:
        recording.update(recording_overrides)

    body: dict[str, Any] = {
        "id": "rc-call-123",
        "result": "Call connected",
        "duration": 61,
        "direction": "Outbound",
        "from": {"phoneNumber": "+17865550100", "name": "MedHub Agent"},
        "to": {"phoneNumber": "+13055551234", "name": "Patient Name"},
        "recording": recording,
    }
    if overrides:
        body.update(overrides)

    return {
        "event": "/restapi/v1.0/account/~/telephony/sessions",
        "body": body,
    }


def _telephony_session_payload(
    *,
    status_code: str = "Disconnected",
    telephony_session_id: str = "session-123",
) -> dict[str, Any]:
    return {
        "event": "/restapi/v1.0/account/~/telephony/sessions",
        "body": {
            "telephonySessionId": telephony_session_id,
            "parties": [
                {
                    "id": "party-1",
                    "status": {"code": status_code},
                }
            ],
        },
    }


def _call_log_record(
    *,
    call_id: str = "call-log-123",
    telephony_session_id: str = "session-123",
    recording_id: str = "rec-123",
    recording: dict[str, object] | None = None,
) -> dict[str, object]:
    if recording is None:
        recording = {
            "id": recording_id,
            "contentUri": (
                "https://media.ringcentral.com/restapi/v1.0/account/123/"
                f"recording/{recording_id}/content"
            ),
        }

    return {
        "id": call_id,
        "telephonySessionId": telephony_session_id,
        "result": "Call connected",
        "duration": 61,
        "direction": "Outbound",
        "from": {"phoneNumber": "+17865550100", "name": "MedHub Agent"},
        "to": {"phoneNumber": "+13055551234", "name": "Patient Name"},
        "recording": recording,
    }
