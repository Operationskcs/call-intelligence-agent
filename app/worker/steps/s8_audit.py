"""Step 8: audit result logging."""

from datetime import UTC, datetime

from app.models.call_event import CallEvent
from app.models.match_result import MatchResult
from app.models.note import ExtractedNote
from app.storage.audit import AUDIT_STATUS_FAILED, AUDIT_STATUS_SUCCEEDED, upsert_call_log


async def log_result(
    event: CallEvent,
    match: MatchResult,
    note: ExtractedNote | None,
    error: str | None,
) -> datetime | None:
    """Upsert pipeline result details into call_audit_log."""

    _ = note
    processed_at = datetime.now(UTC)
    status = AUDIT_STATUS_SUCCEEDED if error is None else AUDIT_STATUS_FAILED
    await upsert_call_log(
        {
            "call_id": event.call_id,
            "source": event.source.value,
            "status": status,
            "workspace": match.workspace or event.workspace,
            "crm_record_id": match.crm_record_id,
            "phone_from": event.phone_from,
            "phone_to": event.phone_to,
            "duration_sec": event.duration_sec,
            "gcs_audio_uri": event.gcs_audio_uri,
            "gcs_transcript_uri": None,
            "match_confidence": match.confidence,
            "match_method": match.method.value,
            "matched_on_phone": match.matched_on_phone,
            "note_created": (
                error is None
                and bool(match.crm_record_id)
                and not match.requires_review
            ),
            "review_required": match.requires_review,
            "error_message": error,
            "processed_at": processed_at,
            "updated_at": processed_at,
        }
    )
    return processed_at
