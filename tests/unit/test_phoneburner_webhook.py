"""Tests for the PhoneBurner webhook parser and handler."""

from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI

from app.models.call_event import CallEvent, CallSource
from app.webhook import phoneburner
from app.worker import pipeline
from app.worker.steps import s1_ingest


def _payload(**overrides: object) -> dict[str, object]:
    """Build a representative PhoneBurner webhook payload."""

    payload: dict[str, object] = {
        "call_id": "pb-call-123",
        "recording_gcs_uri": "gs://pb-dispositions-call-recordings/pb-call-123.mp3",
        "phone_from": "+15550000001",
        "phone_to": "+15550000002",
        "duration": 90,
        "connected": True,
        "end_time": "2026-07-23T14:30:00Z",
        "from": {"name": "PhoneBurner Agent"},
    }
    payload.update(overrides)
    return payload


def _app() -> FastAPI:
    """Create a minimal test app for the PhoneBurner router."""

    app = FastAPI()
    app.include_router(phoneburner.router)
    return app


async def _post_phoneburner(
    payload: dict[str, object] | list[object],
    *,
    token: str | None = "secret",
) -> httpx.Response:
    """POST a PhoneBurner payload through the ASGI app."""

    headers = {"X-Webhook-Token": token} if token is not None else {}
    transport = httpx.ASGITransport(app=_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post("/webhook/phoneburner", json=payload, headers=headers)


def test_parse_phoneburner_payload_extracts_required_fields() -> None:
    """The parser should normalize the fields used by the Intake pipeline."""

    parsed_payload = phoneburner.parse_phoneburner_payload(
        _payload(duration="91", connected="true")
    )

    assert parsed_payload.call_id == "pb-call-123"
    assert parsed_payload.recording_gcs_uri == (
        "gs://pb-dispositions-call-recordings/pb-call-123.mp3"
    )
    assert parsed_payload.phone_from == "+15550000001"
    assert parsed_payload.phone_to == "+15550000002"
    assert parsed_payload.duration == 91
    assert parsed_payload.connected is True
    assert parsed_payload.end_time == "2026-07-23T14:30:00Z"
    assert parsed_payload.agent_name == "PhoneBurner Agent"


def test_build_call_event_uses_intake_phoneburner_fields() -> None:
    """Parsed webhook fields should map into the existing CallEvent contract."""

    raw_payload = _payload()
    event = phoneburner.build_call_event(
        phoneburner.parse_phoneburner_payload(raw_payload),
        raw_payload,
    )

    assert event == CallEvent(
        call_id="pb-call-123",
        source=CallSource.PHONEBURNER,
        workspace="intake",
        phone_from="+15550000001",
        phone_to="+15550000002",
        duration_sec=90,
        agent_id="PhoneBurner Agent",
        agent_name="PhoneBurner Agent",
        gcs_audio_uri="gs://pb-dispositions-call-recordings/pb-call-123.mp3",
        raw_payload=raw_payload,
    )


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        (_payload(connected=False), "not_connected"),
        (_payload(duration=29), "duration_below_minimum"),
    ],
)
async def test_receive_phoneburner_webhook_skips_ineligible_calls(
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, object],
    reason: str,
) -> None:
    """Ineligible calls should return a clear skip response without pipeline work."""

    monkeypatch.setenv("WEBHOOK_SECRET_TOKEN", "secret")

    async def fail_check_idempotency(call_id: str) -> bool:
        raise AssertionError(f"Unexpected idempotency check for {call_id}")

    async def fail_pipeline_run(event: CallEvent) -> None:
        raise AssertionError(f"Unexpected pipeline run for {event.call_id}")

    monkeypatch.setattr(s1_ingest, "check_idempotency", fail_check_idempotency)
    monkeypatch.setattr(pipeline, "run", fail_pipeline_run)

    response = await _post_phoneburner(payload)

    assert response.status_code == 200
    assert response.json() == {
        "status": "skipped",
        "call_id": "pb-call-123",
        "reason": reason,
    }


async def test_receive_phoneburner_webhook_returns_already_processed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Previously processed calls should not be handed to the pipeline again."""

    monkeypatch.setenv("WEBHOOK_SECRET_TOKEN", "secret")

    async def fake_check_idempotency(call_id: str) -> bool:
        assert call_id == "pb-call-123"
        return True

    async def fail_pipeline_run(event: CallEvent) -> None:
        raise AssertionError(f"Unexpected pipeline run for {event.call_id}")

    monkeypatch.setattr(s1_ingest, "check_idempotency", fake_check_idempotency)
    monkeypatch.setattr(pipeline, "run", fail_pipeline_run)

    response = await _post_phoneburner(_payload())

    assert response.status_code == 200
    assert response.json() == {"status": "already_processed", "call_id": "pb-call-123"}


async def test_receive_phoneburner_webhook_runs_pipeline_for_accepted_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Eligible unprocessed calls should be converted to CallEvent and processed."""

    monkeypatch.setenv("WEBHOOK_SECRET_TOKEN", "secret")
    processed_events: list[CallEvent] = []

    async def fake_check_idempotency(call_id: str) -> bool:
        assert call_id == "pb-call-123"
        return False

    async def fake_pipeline_run(event: CallEvent) -> None:
        processed_events.append(event)

    monkeypatch.setattr(s1_ingest, "check_idempotency", fake_check_idempotency)
    monkeypatch.setattr(pipeline, "run", fake_pipeline_run)

    response = await _post_phoneburner(_payload())

    assert response.status_code == 200
    assert response.json() == {"status": "accepted", "call_id": "pb-call-123"}
    assert len(processed_events) == 1
    event = processed_events[0]
    assert event.call_id == "pb-call-123"
    assert event.source is CallSource.PHONEBURNER
    assert event.workspace == "intake"
    assert event.phone_from == "+15550000001"
    assert event.phone_to == "+15550000002"
    assert event.duration_sec == 90
    assert event.agent_name == "PhoneBurner Agent"
    assert event.gcs_audio_uri == "gs://pb-dispositions-call-recordings/pb-call-123.mp3"
    assert event.raw_payload["end_time"] == "2026-07-23T14:30:00Z"


async def test_receive_phoneburner_webhook_rejects_invalid_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PhoneBurner webhooks must present the shared webhook token."""

    monkeypatch.setenv("WEBHOOK_SECRET_TOKEN", "secret")

    response = await _post_phoneburner(_payload(), token="wrong")

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid webhook token"}


async def test_receive_phoneburner_webhook_requires_json_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The handler should reject JSON arrays before parsing call fields."""

    monkeypatch.setenv("WEBHOOK_SECRET_TOKEN", "secret")

    response = await _post_phoneburner([])

    assert response.status_code == 400
    assert response.json() == {"detail": "Expected JSON object"}


def test_parse_phoneburner_payload_requires_call_id() -> None:
    """A webhook without call_id cannot be correlated with the audit log."""

    with pytest.raises(Exception) as exc_info:
        phoneburner.parse_phoneburner_payload(_payload(call_id=""))

    assert getattr(exc_info.value, "status_code") == 400


def test_phoneburner_skip_reason_accepts_connected_long_calls() -> None:
    """Eligible calls should not receive a skip reason."""

    parsed_payload = phoneburner.parse_phoneburner_payload(_payload())

    assert phoneburner.phoneburner_skip_reason(parsed_payload) is None
