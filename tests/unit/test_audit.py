"""Tests for audit log row construction."""

from pathlib import Path
from typing import Any

from app.models.call_event import CallEvent, CallSource
from app.models.match_result import MatchMethod, MatchResult
import app.storage.audit as audit
from app.storage.audit import (
    AUDIT_STATUS_FAILED,
    AUDIT_STATUS_SUCCEEDED,
    DEEPGRAM_BAD_REQUEST_ERROR_MESSAGE,
    DEEPGRAM_NO_TRANSCRIPT_ERROR_MESSAGE,
    PROCESSING_RECLAIM_TTL_SECONDS,
)
from app.worker.steps import s8_audit


async def test_try_reserve_call_id_claims_unseen_call(monkeypatch: Any) -> None:
    """The reservation should create processing rows and only reclaim expired crashes."""

    captured: dict[str, Any] = {}

    class FakeConnection:
        async def fetchrow(
            self,
            query: str,
            call_id: str,
            source: str,
            reclaim_after_seconds: int,
        ) -> dict[str, str]:
            captured["query"] = query
            captured["call_id"] = call_id
            captured["source"] = source
            captured["reclaim_after_seconds"] = reclaim_after_seconds
            return {"call_id": call_id}

        async def close(self) -> None:
            captured["closed"] = True

    async def fake_connect() -> FakeConnection:
        return FakeConnection()

    monkeypatch.setattr(audit, "_connect", fake_connect)

    reserved = await audit.try_reserve_call_id("pb-call-123", "phoneburner")

    assert reserved is True
    assert (
        "INSERT INTO call_audit_log "
        "(call_id, source, status, reserved_at, updated_at)"
    ) in captured["query"]
    assert "VALUES ($1, $2, 'processing', NOW(), NOW())" in captured["query"]
    assert "WHERE call_audit_log.status = 'processing'" in captured["query"]
    assert "processed_at IS NULL" not in captured["query"]
    assert captured["call_id"] == "pb-call-123"
    assert captured["source"] == "phoneburner"
    assert captured["reclaim_after_seconds"] == PROCESSING_RECLAIM_TTL_SECONDS
    assert captured["closed"] is True


async def test_try_reserve_call_id_returns_false_for_active_claim(
    monkeypatch: Any,
) -> None:
    """A concurrent active processing row should not be reopened."""

    captured: dict[str, Any] = {}

    class FakeConnection:
        async def fetchrow(
            self,
            query: str,
            call_id: str,
            source: str,
            reclaim_after_seconds: int,
        ) -> None:
            captured["query"] = query
            captured["call_id"] = call_id
            captured["source"] = source
            captured["reclaim_after_seconds"] = reclaim_after_seconds
            return None

        async def close(self) -> None:
            captured["closed"] = True

    async def fake_connect() -> FakeConnection:
        return FakeConnection()

    monkeypatch.setattr(audit, "_connect", fake_connect)

    reserved = await audit.try_reserve_call_id("pb-call-123", "phoneburner")

    assert reserved is False
    assert "WHERE call_audit_log.status = 'processing'" in captured["query"]
    assert captured["reclaim_after_seconds"] == PROCESSING_RECLAIM_TTL_SECONDS
    assert captured["closed"] is True


async def test_log_result_includes_matched_on_phone(monkeypatch: Any) -> None:
    """Audit rows should record which MedHub patient-phone candidate matched."""

    captured: dict[str, Any] = {}

    async def fake_upsert_call_log(row: dict[str, Any]) -> None:
        captured.update(row)

    monkeypatch.setattr(s8_audit, "upsert_call_log", fake_upsert_call_log)
    event = CallEvent(
        call_id="rc-123",
        source=CallSource.RINGCENTRAL,
        workspace="medhub",
        phone_from="+13055551234",
        phone_to="+17865550100",
        patient_phone_primary="+13055551234",
        patient_phone_fallback="+17865550100",
        duration_sec=60,
        agent_id="MedHub Agent",
        gcs_audio_uri="gs://bucket/audio.mp3",
        raw_payload={},
    )
    match = MatchResult(
        crm_record_id="lead-123",
        workspace="medhub",
        confidence=1.0,
        method=MatchMethod.PHONE,
        requires_review=False,
        matched_on_phone="fallback",
    )

    await s8_audit.log_result(event, match, note=None, error=None)

    assert captured["matched_on_phone"] == "fallback"
    assert captured["status"] == AUDIT_STATUS_SUCCEEDED
    assert captured["processed_at"] is not None
    assert captured["updated_at"] == captured["processed_at"]


async def test_manual_review_log_result_is_processed_without_note(monkeypatch: Any) -> None:
    """Manual-review rows should satisfy idempotency without claiming a note write."""

    captured: dict[str, Any] = {}

    async def fake_upsert_call_log(row: dict[str, Any]) -> None:
        captured.update(row)

    monkeypatch.setattr(s8_audit, "upsert_call_log", fake_upsert_call_log)
    event = CallEvent(
        call_id="review-call",
        source=CallSource.PHONEBURNER,
        workspace="intake",
        phone_from="+15550000001",
        phone_to="+15550000002",
        duration_sec=60,
        agent_id=None,
        gcs_audio_uri="gs://bucket/audio.mp3",
        raw_payload={},
    )
    match = MatchResult(
        crm_record_id="lead-123",
        workspace="intake",
        confidence=0.7,
        method=MatchMethod.NAME,
        requires_review=True,
    )

    await s8_audit.log_result(event, match, note=None, error=None)

    assert captured["review_required"] is True
    assert captured["error_message"] is None
    assert captured["processed_at"] is not None
    assert captured["status"] == AUDIT_STATUS_SUCCEEDED
    assert captured["updated_at"] == captured["processed_at"]
    assert captured["note_created"] is False


async def test_deepgram_no_transcript_error_is_processed_terminal_state(
    monkeypatch: Any,
) -> None:
    """No-transcript audio errors should not be retried forever."""

    captured: dict[str, Any] = {}

    async def fake_upsert_call_log(row: dict[str, Any]) -> None:
        captured.update(row)

    monkeypatch.setattr(s8_audit, "upsert_call_log", fake_upsert_call_log)
    event = CallEvent(
        call_id="silent-call",
        source=CallSource.PHONEBURNER,
        workspace="intake",
        phone_from="+15550000001",
        phone_to="+15550000002",
        duration_sec=60,
        agent_id=None,
        gcs_audio_uri="gs://bucket/audio.mp3",
        raw_payload={},
    )
    match = MatchResult(
        crm_record_id=None,
        workspace="intake",
        confidence=0.0,
        method=MatchMethod.UNMATCHED,
        requires_review=False,
    )

    await s8_audit.log_result(
        event,
        match,
        note=None,
        error=DEEPGRAM_NO_TRANSCRIPT_ERROR_MESSAGE,
    )

    assert captured["review_required"] is False
    assert captured["error_message"] == DEEPGRAM_NO_TRANSCRIPT_ERROR_MESSAGE
    assert captured["processed_at"] is not None
    assert captured["status"] == AUDIT_STATUS_FAILED
    assert captured["updated_at"] == captured["processed_at"]
    assert captured["note_created"] is False


async def test_deepgram_bad_request_error_is_processed_terminal_state(
    monkeypatch: Any,
) -> None:
    """Corrupt or unsupported audio should not be retried forever."""

    captured: dict[str, Any] = {}

    async def fake_upsert_call_log(row: dict[str, Any]) -> None:
        captured.update(row)

    monkeypatch.setattr(s8_audit, "upsert_call_log", fake_upsert_call_log)
    event = CallEvent(
        call_id="corrupt-call",
        source=CallSource.PHONEBURNER,
        workspace="intake",
        phone_from="+15550000001",
        phone_to="+15550000002",
        duration_sec=60,
        agent_id=None,
        gcs_audio_uri="gs://bucket/audio.mp3",
        raw_payload={},
    )
    match = MatchResult(
        crm_record_id=None,
        workspace="intake",
        confidence=0.0,
        method=MatchMethod.UNMATCHED,
        requires_review=False,
    )

    await s8_audit.log_result(
        event,
        match,
        note=None,
        error=DEEPGRAM_BAD_REQUEST_ERROR_MESSAGE,
    )

    assert captured["review_required"] is False
    assert captured["error_message"] == DEEPGRAM_BAD_REQUEST_ERROR_MESSAGE
    assert captured["processed_at"] is not None
    assert captured["status"] == AUDIT_STATUS_FAILED
    assert captured["updated_at"] == captured["processed_at"]
    assert captured["note_created"] is False


async def test_non_terminal_exception_log_result_is_failed_terminal_state(
    monkeypatch: Any,
) -> None:
    """Handled pipeline errors should close the audit row as failed."""

    captured: dict[str, Any] = {}

    async def fake_upsert_call_log(row: dict[str, Any]) -> None:
        captured.update(row)

    monkeypatch.setattr(s8_audit, "upsert_call_log", fake_upsert_call_log)
    event = CallEvent(
        call_id="forbidden-call",
        source=CallSource.RINGCENTRAL,
        workspace="medhub",
        phone_from="+15550000001",
        phone_to="+15550000002",
        duration_sec=60,
        agent_id=None,
        gcs_audio_uri="gs://bucket/audio.mp3",
        raw_payload={},
    )
    match = MatchResult(
        crm_record_id=None,
        workspace="medhub",
        confidence=0.0,
        method=MatchMethod.UNMATCHED,
        requires_review=True,
    )

    await s8_audit.log_result(event, match, note=None, error="403 Forbidden")

    assert captured["status"] == AUDIT_STATUS_FAILED
    assert captured["error_message"] == "403 Forbidden"
    assert captured["processed_at"] is not None
    assert captured["updated_at"] == captured["processed_at"]


def test_call_audit_status_migration_keeps_legacy_unprocessed_out_of_reclaim() -> None:
    """Legacy NULL processed_at rows must not become TTL-reclaimable."""

    sql = Path("migrations/003_call_audit_status.sql").read_text()

    assert "WHEN processed_at IS NULL THEN 'legacy_unprocessed'" in sql
    assert "WHERE status = 'processing'" in audit.RECLAIMABLE_PROCESSING_CALL_IDS_QUERY
    assert "processed_at IS NULL" not in audit.RECLAIMABLE_PROCESSING_CALL_IDS_QUERY
