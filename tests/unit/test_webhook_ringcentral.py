"""Tests for the RingCentral webhook handler."""

from typing import Any, cast

import pytest
from fastapi import BackgroundTasks, FastAPI
from fastapi.testclient import TestClient

from app.models.call_event import CallEvent, CallSource
from app.webhook import ringcentral


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
