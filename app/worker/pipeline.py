"""Eight-step call intelligence pipeline orchestration."""

import logging

from app.adapters.crm.factory import get_crm_clients
from app.models.call_event import CallEvent
from app.models.match_result import MatchMethod, MatchResult
from app.models.note import ExtractedNote
from app.worker.steps import s1_ingest, s2_fetch, s3_transcribe, s4_extract, s5_match
from app.worker.steps import s6_route, s7_write, s8_audit

logger = logging.getLogger(__name__)

__all__ = [
    "MANUAL_REVIEW_ERROR_MESSAGE",
    "ManualReviewRequiredError",
    "get_crm_clients",
    "run",
    "s1_ingest",
    "s2_fetch",
    "s3_transcribe",
    "s4_extract",
    "s5_match",
    "s6_route",
    "s7_write",
    "s8_audit",
]

MANUAL_REVIEW_ERROR_MESSAGE = "Lead match requires manual review; CRM write skipped."


class ManualReviewRequiredError(ValueError):
    """Raised when a call was routed to manual review instead of CRM write."""


async def run(event: CallEvent) -> None:
    """Run the call pipeline from idempotency check through audit logging."""

    match_result = MatchResult(
        crm_record_id=None,
        workspace=event.workspace,
        confidence=0.0,
        method=MatchMethod.UNMATCHED,
        requires_review=True,
    )
    note: ExtractedNote | None = None

    try:
        if await s1_ingest.check_idempotency(event.call_id):
            logger.info("Skipping already processed call_id=%s", event.call_id)
            return

        event = await s2_fetch.fetch_recording(event)
        transcript = await s3_transcribe.transcribe(event)
        note = await s4_extract.extract(transcript, event.workspace)
        match_result = await s5_match.match_lead(event, get_crm_clients())

        if match_result.requires_review:
            await s8_audit.log_result(event, match_result, note, error=None)
            raise ManualReviewRequiredError(MANUAL_REVIEW_ERROR_MESSAGE)

        s6_route.route(match_result, event)
        await s7_write.write_note(match_result, note, event, transcript)
        await s8_audit.log_result(event, match_result, note, error=None)
    except ManualReviewRequiredError:
        raise
    except Exception as exc:
        await s8_audit.log_result(event, match_result, note, error=str(exc))
        raise
