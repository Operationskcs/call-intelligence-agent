"""Tests for the call-quality trigger client."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace

import httpx
import pytest

from app.models.call_event import CallEvent, CallSource
from app.models.match_result import MatchMethod, MatchResult
from app.models.note import ExtractedNote
import app.services.call_quality_trigger as call_quality_trigger


def test_audience_from_url_uses_cloud_run_origin() -> None:
    """The OIDC audience should be the service origin, not the endpoint path."""

    assert (
        call_quality_trigger._audience_from_url(
            "https://call-quality-trigger-znesczxkka-uc.a.run.app/"
            "integrations/call-intelligence/trigger"
        )
        == "https://call-quality-trigger-znesczxkka-uc.a.run.app"
    )


def test_payload_omits_agent_name_and_truncates_phone_fields() -> None:
    """The trigger payload should match the downstream schema."""

    payload = call_quality_trigger._payload(
        event=_event().model_copy(
            update={
                "phone_from": "+155500000011234567890123456789999",
                "phone_to": "+155500000022234567890123456789999",
            }
        ),
        match=_match(),
        note=_note(),
        transcript="[Agent]: hello",
        processed_at=datetime(2026, 7, 31, 12, 30, tzinfo=UTC),
    )

    assert payload == {
        "call_id": "call-123",
        "lead_id": "lead-123",
        "workspace": "intake",
        "transcript": "[Agent]: hello",
        "summary": "Lead needs help.",
        "disposition": "Interested",
        "duration_sec": 90,
        "phone_from": "+1555000000112345678901234567899",
        "phone_to": "+1555000000222345678901234567899",
        "created_at": "2026-07-31T12:30:00+00:00",
    }
    assert "agent_name" not in payload
    assert len(str(payload["phone_from"])) == 32
    assert len(str(payload["phone_to"])) == 32


async def test_notify_dispatches_trigger_in_background_and_logs_success(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The trigger POST should run in the background after dispatch."""

    captured: dict[str, object] = {}
    post_started = asyncio.Event()
    post_release = asyncio.Event()

    def fake_fetch_id_token(audience: str) -> str:
        captured["audience"] = audience
        return "signed-token"

    async def fake_post_trigger(
        trigger_url: str,
        token: str,
        payload: dict[str, object],
    ) -> httpx.Response:
        captured["url"] = trigger_url
        captured["token"] = token
        captured["json"] = payload
        post_started.set()
        await post_release.wait()
        return httpx.Response(status_code=202)

    monkeypatch.setattr(call_quality_trigger, "_fetch_id_token", fake_fetch_id_token)
    monkeypatch.setattr(call_quality_trigger, "_post_trigger", fake_post_trigger)
    monkeypatch.setattr(
        call_quality_trigger,
        "get_settings",
        lambda: SimpleNamespace(
            pipeline=SimpleNamespace(
                call_quality_trigger_url=(
                    "https://call-quality-trigger-znesczxkka-uc.a.run.app/"
                    "integrations/call-intelligence/trigger"
                )
            )
        ),
    )

    with caplog.at_level("INFO"):
        await call_quality_trigger.notify_call_quality_trigger(
            event=_event(),
            match=_match(),
            note=_note(),
            transcript="[Agent]: hello",
            processed_at=datetime(2026, 7, 31, 12, 30, tzinfo=UTC),
        )

        assert "Call quality trigger dispatched. call_id=call-123" in caplog.text
        assert "Call quality trigger succeeded. call_id=call-123" not in caplog.text

        await asyncio.wait_for(post_started.wait(), timeout=1.0)
        post_release.set()
        await _wait_for_log(
            caplog,
            "Call quality trigger succeeded. call_id=call-123 status_code=202",
        )

    assert captured["audience"] == "https://call-quality-trigger-znesczxkka-uc.a.run.app"
    assert captured["token"] == "signed-token"
    assert captured["json"] == {
        "call_id": "call-123",
        "lead_id": "lead-123",
        "workspace": "intake",
        "transcript": "[Agent]: hello",
        "summary": "Lead needs help.",
        "disposition": "Interested",
        "duration_sec": 90,
        "phone_from": "+15550000001",
        "phone_to": "+15550000002",
        "created_at": "2026-07-31T12:30:00+00:00",
    }


async def test_notify_warning_on_endpoint_failure(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Endpoint failures should be warnings, not raised pipeline errors."""

    captured: dict[str, object] = {}

    def fake_fetch_id_token(audience: str) -> str:
        captured["audience"] = audience
        return "signed-token"

    async def fake_post_trigger(
        trigger_url: str,
        token: str,
        payload: dict[str, object],
    ) -> httpx.Response:
        captured["url"] = trigger_url
        captured["token"] = token
        captured["json"] = payload
        return httpx.Response(status_code=503, text="unavailable")

    monkeypatch.setattr(call_quality_trigger, "_fetch_id_token", fake_fetch_id_token)
    monkeypatch.setattr(call_quality_trigger, "_post_trigger", fake_post_trigger)
    monkeypatch.setattr(
        call_quality_trigger,
        "get_settings",
        lambda: SimpleNamespace(
            pipeline=SimpleNamespace(
                call_quality_trigger_url=(
                    "https://call-quality-trigger-znesczxkka-uc.a.run.app/"
                    "integrations/call-intelligence/trigger"
                )
            )
        ),
    )

    with caplog.at_level("INFO"):
        await call_quality_trigger.notify_call_quality_trigger(
            event=_event(),
            match=_match(),
            note=_note(),
            transcript="[Agent]: hello",
            processed_at=datetime(2026, 7, 31, 12, 30, tzinfo=UTC),
        )
        await _wait_for_log(
            caplog,
            "Call quality trigger failed. call_id=call-123 error=status_code=503",
        )

    assert captured["audience"] == "https://call-quality-trigger-znesczxkka-uc.a.run.app"
    assert captured["token"] == "signed-token"
    assert (
        "Call quality trigger failed. call_id=call-123 "
        "error=status_code=503 response=unavailable"
    ) in caplog.text


async def _wait_for_log(
    caplog: pytest.LogCaptureFixture,
    expected: str,
) -> None:
    for _ in range(100):
        if expected in caplog.text:
            return
        await asyncio.sleep(0.01)
    pytest.fail(f"Expected log was not emitted: {expected}")


def _event() -> CallEvent:
    return CallEvent(
        call_id="call-123",
        source=CallSource.PHONEBURNER,
        workspace="intake",
        phone_from="+15550000001",
        phone_to="+15550000002",
        duration_sec=90,
        agent_id="agent-123",
        agent_name="PhoneBurner Agent",
        gcs_audio_uri="gs://bucket/audio.mp3",
        raw_payload={},
    )


def _match() -> MatchResult:
    return MatchResult(
        crm_record_id="lead-123",
        workspace="intake",
        confidence=1.0,
        method=MatchMethod.PHONE,
        requires_review=False,
        agent_name="PhoneBurner Agent",
    )


def _note() -> ExtractedNote:
    return ExtractedNote(
        summary="Lead needs help.",
        disposition="Interested",
        next_steps="Follow up.",
        callback_date=None,
        sentiment="positive",
        objections=None,
        pii_detected=False,
        confidence=0.95,
    )
