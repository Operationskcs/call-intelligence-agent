"""Step 8: audit result logging."""

from datetime import UTC, datetime

from app.models.call_event import CallEvent
from app.models.match_result import MatchResult
from app.models.note import ExtractedNote
from app.storage.audit import TERMINAL_PROCESSED_ERROR_MESSAGES, upsert_call_log


async def log_result(
    event: CallEvent,
    match: MatchResult,
    note: ExtractedNote | None,
    error: str | None,
) -> datetime | None:
    """Upsert pipeline result details into call_audit_log."""

    _ = note
    is_terminal_state = error is None or error in TERMINAL_PROCESSED_ERROR_MESSAGES
    processed_at = datetime.now(UTC) if is_terminal_state else None
    await upsert_call_log(
        {
            "call_id": event.call_id,
            "source": event.source.value,
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
        }
    )
    return processed_at
