"""Integration tests for the PhoneBurner webhook."""

from __future__ import annotations

import httpx
import pytest

from app.config import get_settings
from app.models.call_event import CallEvent, CallSource
from app.models.match_result import MatchMethod, MatchResult
from app.models.note import ExtractedNote
from app.worker import main, pipeline
from app.worker.steps import s1_ingest, s2_fetch, s3_transcribe, s4_extract, s5_match
from app.worker.steps import s6_route, s7_write, s8_audit


async def test_phoneburner_webhook_runs_pipeline_with_mocked_steps(
    monkeypatch: pytest.MonkeyPatch,
    mock_extracted_note: ExtractedNote,
) -> None:
    """A simulated PhoneBurner webhook should drive the existing s2-s8 pipeline."""

    monkeypatch.setenv("WEBHOOK_SECRET_TOKEN", "secret")
    get_settings.cache_clear()
    calls: list[str] = []
    audit_rows: list[dict[str, object]] = []

    async def fake_check_idempotency(call_id: str) -> bool:
        calls.append(f"s1:{call_id}")
        return False

    async def fake_try_reserve_call_id(call_id: str, source: str) -> bool:
        _ = call_id, source
        return True

    async def fake_fetch_recording(event: CallEvent) -> CallEvent:
        calls.append(f"s2:{event.call_id}")
        assert event.source is CallSource.PHONEBURNER
        assert event.workspace == "intake"
        assert event.gcs_audio_uri == "gs://pb-dispositions-call-recordings/pb-call-123.mp3"
        return event

    async def fake_transcribe(event: CallEvent) -> str:
        calls.append(f"s3:{event.call_id}")
        return "[Agent]: hello\n[Lead]: I need help with my case"

    async def fake_extract(transcript: str, workspace: str) -> ExtractedNote:
        calls.append(f"s4:{workspace}")
        assert "I need help" in transcript
        return mock_extracted_note

    async def fake_match_lead(event: CallEvent, crm_clients: object) -> MatchResult:
        calls.append(f"s5:{event.call_id}")
        assert crm_clients == {"intake": object}
        return MatchResult(
            crm_record_id="lead-123",
            workspace="intake",
            confidence=0.99,
            method=MatchMethod.PHONE,
            requires_review=False,
        )

    def fake_route(match_result: MatchResult, event: CallEvent) -> None:
        calls.append(f"s6:{event.call_id}:{match_result.crm_record_id}")

    async def fake_write_note(
        match_result: MatchResult,
        note: ExtractedNote,
        event: CallEvent,
        transcript: str,
    ) -> None:
        calls.append(f"s7:{event.call_id}:{match_result.crm_record_id}")
        assert note is mock_extracted_note
        assert transcript.startswith("[Agent]")

    async def fake_log_result(
        event: CallEvent,
        match_result: MatchResult,
        note: ExtractedNote | None,
        error: str | None,
    ) -> None:
        calls.append(f"s8:{event.call_id}")
        audit_rows.append(
            {
                "call_id": event.call_id,
                "crm_record_id": match_result.crm_record_id,
                "note": note,
                "error": error,
            }
        )

    monkeypatch.setattr(s1_ingest, "check_idempotency", fake_check_idempotency)
    monkeypatch.setattr(pipeline.audit, "try_reserve_call_id", fake_try_reserve_call_id)
    monkeypatch.setattr(s2_fetch, "fetch_recording", fake_fetch_recording)
    monkeypatch.setattr(s3_transcribe, "transcribe", fake_transcribe)
    monkeypatch.setattr(s4_extract, "extract", fake_extract)
    monkeypatch.setattr(s5_match, "match_lead", fake_match_lead)
    monkeypatch.setattr(s6_route, "route", fake_route)
    monkeypatch.setattr(s7_write, "write_note", fake_write_note)
    monkeypatch.setattr(s8_audit, "log_result", fake_log_result)
    monkeypatch.setattr(pipeline, "get_crm_clients", lambda: {"intake": object})

    transport = httpx.ASGITransport(app=main.create_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/webhook/phoneburner",
            headers={"X-Webhook-Token": "secret"},
            json={
                "call_id": "pb-call-123",
                "recording_gcs_uri": "gs://pb-dispositions-call-recordings/pb-call-123.mp3",
                "phone_from": "+15550000001",
                "phone_to": "+15550000002",
                "duration": 90,
                "connected": True,
                "end_time": "2026-07-23T14:30:00Z",
            },
        )

    get_settings.cache_clear()

    assert response.status_code == 200
    assert response.json() == {"status": "accepted", "call_id": "pb-call-123"}
    assert calls == [
        "s1:pb-call-123",
        "s1:pb-call-123",
        "s2:pb-call-123",
        "s3:pb-call-123",
        "s4:intake",
        "s5:pb-call-123",
        "s6:pb-call-123:lead-123",
        "s7:pb-call-123:lead-123",
        "s8:pb-call-123",
    ]
    assert audit_rows == [
        {
            "call_id": "pb-call-123",
            "crm_record_id": "lead-123",
            "note": mock_extracted_note,
            "error": None,
        }
    ]
