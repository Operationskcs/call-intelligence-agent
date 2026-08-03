"""Tests for pipeline terminal-state handling."""

from datetime import UTC, datetime

import pytest

from app.models.call_event import CallEvent
from app.models.call_event import CallSource
from app.models.match_result import MatchMethod, MatchResult
from app.models.note import ExtractedNote
from app.worker import pipeline


async def test_manual_review_is_audited_without_error(
    monkeypatch: pytest.MonkeyPatch,
    mock_call_event: CallEvent,
    mock_extracted_note: ExtractedNote,
) -> None:
    """Manual-review calls should be processed for idempotency, not error-looped."""

    captured_results: list[dict[str, object]] = []

    async def fake_check_idempotency(call_id: str) -> bool:
        _ = call_id
        return False

    async def fake_fetch_recording(event: CallEvent) -> CallEvent:
        return event

    async def fake_transcribe(event: CallEvent) -> str:
        _ = event
        return "[Agent]: hello\n[Lead]: hello"

    async def fake_extract(transcript: str, workspace: str) -> ExtractedNote:
        _ = transcript, workspace
        return mock_extracted_note

    async def fake_match_lead(
        event: CallEvent,
        crm_clients: object,
    ) -> MatchResult:
        _ = event, crm_clients
        return MatchResult(
            crm_record_id="lead-123",
            workspace="intake",
            confidence=0.7,
            method=MatchMethod.NAME,
            requires_review=True,
        )

    async def fake_log_result(
        event: CallEvent,
        match: MatchResult,
        note: ExtractedNote | None,
        error: str | None,
    ) -> None:
        captured_results.append(
            {
                "call_id": event.call_id,
                "requires_review": match.requires_review,
                "note": note,
                "error": error,
            }
        )

    monkeypatch.setattr(pipeline.s1_ingest, "check_idempotency", fake_check_idempotency)
    monkeypatch.setattr(pipeline.s2_fetch, "fetch_recording", fake_fetch_recording)
    monkeypatch.setattr(pipeline.s3_transcribe, "transcribe", fake_transcribe)
    monkeypatch.setattr(pipeline.s4_extract, "extract", fake_extract)
    monkeypatch.setattr(pipeline.s5_match, "match_lead", fake_match_lead)
    monkeypatch.setattr(pipeline.s8_audit, "log_result", fake_log_result)
    monkeypatch.setattr(
        pipeline,
        "notify_call_quality_trigger",
        lambda **kwargs: pytest.fail(f"Unexpected call quality trigger: {kwargs}"),
    )
    monkeypatch.setattr(pipeline, "get_crm_clients", lambda: {})

    with pytest.raises(pipeline.ManualReviewRequiredError):
        await pipeline.run(mock_call_event)

    assert captured_results == [
        {
            "call_id": "test-call-123",
            "requires_review": True,
            "note": mock_extracted_note,
            "error": None,
        }
    ]


async def test_successful_crm_write_notifies_call_quality_trigger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-review calls should notify the call-quality trigger after CRM write and audit."""

    event = _call_event()
    note = _extracted_note()
    transcript = "[Agent]: hello\n[Lead]: hello"
    processed_at = datetime(2026, 7, 31, 12, 30, tzinfo=UTC)
    calls: list[str] = []
    notifications: list[dict[str, object]] = []

    async def fake_check_idempotency(call_id: str) -> bool:
        calls.append(f"s1:{call_id}")
        return False

    async def fake_fetch_recording(call_event: CallEvent) -> CallEvent:
        calls.append(f"s2:{call_event.call_id}")
        return call_event

    async def fake_transcribe(call_event: CallEvent) -> str:
        calls.append(f"s3:{call_event.call_id}")
        return transcript

    async def fake_extract(call_transcript: str, workspace: str) -> ExtractedNote:
        calls.append(f"s4:{workspace}")
        assert call_transcript == transcript
        return note

    async def fake_match_lead(
        call_event: CallEvent,
        crm_clients: object,
    ) -> MatchResult:
        calls.append(f"s5:{call_event.call_id}")
        _ = crm_clients
        return MatchResult(
            crm_record_id="lead-123",
            workspace="intake",
            confidence=1.0,
            method=MatchMethod.PHONE,
            requires_review=False,
            agent_name=call_event.agent_name,
        )

    def fake_route(match_result: MatchResult, call_event: CallEvent) -> None:
        calls.append(f"s6:{call_event.call_id}:{match_result.crm_record_id}")

    async def fake_write_note(
        match_result: MatchResult,
        extracted_note: ExtractedNote,
        call_event: CallEvent,
        call_transcript: str,
    ) -> None:
        calls.append(f"s7:{call_event.call_id}:{match_result.crm_record_id}")
        assert extracted_note is note
        assert call_transcript == transcript

    async def fake_log_result(
        call_event: CallEvent,
        match_result: MatchResult,
        extracted_note: ExtractedNote | None,
        error: str | None,
    ) -> datetime:
        calls.append(f"s8:{call_event.call_id}")
        assert match_result.requires_review is False
        assert extracted_note is note
        assert error is None
        return processed_at

    async def fake_notify_call_quality_trigger(
        *,
        event: CallEvent,
        match: MatchResult,
        note: ExtractedNote,
        transcript: str,
        processed_at: datetime,
    ) -> None:
        calls.append(f"quality:{event.call_id}")
        notifications.append(
            {
                "call_id": event.call_id,
                "lead_id": match.crm_record_id,
                "agent_name": event.agent_name,
                "summary": note.summary,
                "transcript": transcript,
                "processed_at": processed_at,
            }
        )

    monkeypatch.setattr(pipeline.s1_ingest, "check_idempotency", fake_check_idempotency)
    monkeypatch.setattr(pipeline.s2_fetch, "fetch_recording", fake_fetch_recording)
    monkeypatch.setattr(pipeline.s3_transcribe, "transcribe", fake_transcribe)
    monkeypatch.setattr(pipeline.s4_extract, "extract", fake_extract)
    monkeypatch.setattr(pipeline.s5_match, "match_lead", fake_match_lead)
    monkeypatch.setattr(pipeline.s6_route, "route", fake_route)
    monkeypatch.setattr(pipeline.s7_write, "write_note", fake_write_note)
    monkeypatch.setattr(pipeline.s8_audit, "log_result", fake_log_result)
    monkeypatch.setattr(
        pipeline,
        "notify_call_quality_trigger",
        fake_notify_call_quality_trigger,
    )
    monkeypatch.setattr(pipeline, "get_crm_clients", lambda: {})

    await pipeline.run(event)

    assert calls == [
        "s1:pb-call-123",
        "s2:pb-call-123",
        "s3:pb-call-123",
        "s4:intake",
        "s5:pb-call-123",
        "s6:pb-call-123:lead-123",
        "s7:pb-call-123:lead-123",
        "s8:pb-call-123",
        "quality:pb-call-123",
    ]
    assert notifications == [
        {
            "call_id": "pb-call-123",
            "lead_id": "lead-123",
            "agent_name": "PhoneBurner Agent",
            "summary": "Lead needs help.",
            "transcript": transcript,
            "processed_at": processed_at,
        }
    ]


async def test_successful_non_intake_crm_write_skips_call_quality_trigger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-Intake calls should not notify the Intake-only call-quality trigger."""

    event = _call_event(workspace="medhub")
    note = _extracted_note()
    transcript = "[Agent]: hello\n[Lead]: hello"
    processed_at = datetime(2026, 7, 31, 12, 30, tzinfo=UTC)
    calls: list[str] = []

    async def fake_check_idempotency(call_id: str) -> bool:
        calls.append(f"s1:{call_id}")
        return False

    async def fake_fetch_recording(call_event: CallEvent) -> CallEvent:
        calls.append(f"s2:{call_event.call_id}")
        return call_event

    async def fake_transcribe(call_event: CallEvent) -> str:
        calls.append(f"s3:{call_event.call_id}")
        return transcript

    async def fake_extract(call_transcript: str, workspace: str) -> ExtractedNote:
        calls.append(f"s4:{workspace}")
        assert call_transcript == transcript
        return note

    async def fake_match_lead(
        call_event: CallEvent,
        crm_clients: object,
    ) -> MatchResult:
        calls.append(f"s5:{call_event.call_id}")
        _ = crm_clients
        return MatchResult(
            crm_record_id="lead-123",
            workspace="medhub",
            confidence=1.0,
            method=MatchMethod.PHONE,
            requires_review=False,
            agent_name=call_event.agent_name,
        )

    def fake_route(match_result: MatchResult, call_event: CallEvent) -> None:
        calls.append(f"s6:{call_event.call_id}:{match_result.crm_record_id}")

    async def fake_write_note(
        match_result: MatchResult,
        extracted_note: ExtractedNote,
        call_event: CallEvent,
        call_transcript: str,
    ) -> None:
        calls.append(f"s7:{call_event.call_id}:{match_result.crm_record_id}")
        assert extracted_note is note
        assert call_transcript == transcript

    async def fake_log_result(
        call_event: CallEvent,
        match_result: MatchResult,
        extracted_note: ExtractedNote | None,
        error: str | None,
    ) -> datetime:
        calls.append(f"s8:{call_event.call_id}")
        assert match_result.requires_review is False
        assert extracted_note is note
        assert error is None
        return processed_at

    monkeypatch.setattr(pipeline.s1_ingest, "check_idempotency", fake_check_idempotency)
    monkeypatch.setattr(pipeline.s2_fetch, "fetch_recording", fake_fetch_recording)
    monkeypatch.setattr(pipeline.s3_transcribe, "transcribe", fake_transcribe)
    monkeypatch.setattr(pipeline.s4_extract, "extract", fake_extract)
    monkeypatch.setattr(pipeline.s5_match, "match_lead", fake_match_lead)
    monkeypatch.setattr(pipeline.s6_route, "route", fake_route)
    monkeypatch.setattr(pipeline.s7_write, "write_note", fake_write_note)
    monkeypatch.setattr(pipeline.s8_audit, "log_result", fake_log_result)
    monkeypatch.setattr(
        pipeline,
        "notify_call_quality_trigger",
        lambda **kwargs: pytest.fail(f"Unexpected call quality trigger: {kwargs}"),
    )
    monkeypatch.setattr(pipeline, "get_crm_clients", lambda: {})

    await pipeline.run(event)

    assert calls == [
        "s1:pb-call-123",
        "s2:pb-call-123",
        "s3:pb-call-123",
        "s4:medhub",
        "s5:pb-call-123",
        "s6:pb-call-123:lead-123",
        "s7:pb-call-123:lead-123",
        "s8:pb-call-123",
    ]


def _call_event(workspace: str = "intake") -> CallEvent:
    """Build a representative successful Intake event."""

    return CallEvent(
        call_id="pb-call-123",
        source=CallSource.PHONEBURNER,
        workspace=workspace,
        phone_from="+15550000001",
        phone_to="+15550000002",
        duration_sec=90,
        agent_id="agent-123",
        agent_name="PhoneBurner Agent",
        gcs_audio_uri="gs://bucket/audio.mp3",
        raw_payload={},
    )


def _extracted_note() -> ExtractedNote:
    """Build a minimal extracted note."""

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
