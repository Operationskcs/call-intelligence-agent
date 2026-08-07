"""Notify the downstream call-quality trigger after successful CRM writes."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from urllib.parse import urlsplit

import httpx
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token

from app.config import get_settings
from app.models.call_event import CallEvent
from app.models.match_result import MatchResult
from app.models.note import ExtractedNote

logger = logging.getLogger(__name__)

_REQUEST_TIMEOUT_SECONDS = 30.0
_MAX_PHONE_LENGTH = 32
_BACKGROUND_TASKS: set[asyncio.Task[None]] = set()


async def notify_call_quality_trigger(
    *,
    event: CallEvent,
    match: MatchResult,
    note: ExtractedNote,
    transcript: str,
    processed_at: datetime,
) -> None:
    """Schedule a successful CRM write notification without blocking the pipeline."""

    try:
        trigger_url = get_settings().pipeline.call_quality_trigger_url.strip()
        if not trigger_url:
            logger.warning(
                "CALL_QUALITY_TRIGGER_URL is not configured; skipping call quality trigger."
            )
            return

        audience = _audience_from_url(trigger_url)
        payload = _payload(
            event=event,
            match=match,
            note=note,
            transcript=transcript,
            processed_at=processed_at,
        )

        task = asyncio.create_task(
            _run_call_quality_trigger(
                event.call_id,
                trigger_url,
                audience,
                payload,
            )
        )
        _BACKGROUND_TASKS.add(task)
        task.add_done_callback(_BACKGROUND_TASKS.discard)

        logger.info("Call quality trigger dispatched. call_id=%s", event.call_id)
    except Exception as exc:
        logger.warning(
            "Call quality trigger failed. call_id=%s error=%s",
            event.call_id,
            _error_text(exc),
            exc_info=True,
        )


async def _run_call_quality_trigger(
    call_id: str,
    trigger_url: str,
    audience: str,
    payload: dict[str, object],
) -> None:
    """Send the trigger notification and own all background-task logging."""

    try:
        token = await asyncio.to_thread(_fetch_id_token, audience)
        response = await _post_trigger(trigger_url, token, payload)

        if response.is_success:
            logger.info(
                "Call quality trigger succeeded. call_id=%s status_code=%s",
                call_id,
                response.status_code,
            )
            return

        logger.warning(
            "Call quality trigger failed. call_id=%s error=%s",
            call_id,
            f"status_code={response.status_code} response={response.text[:500]}",
        )
    except Exception as exc:
        logger.warning(
            "Call quality trigger failed. call_id=%s error=%s",
            call_id,
            _error_text(exc),
            exc_info=True,
        )


def _payload(
    *,
    event: CallEvent,
    match: MatchResult,
    note: ExtractedNote,
    transcript: str,
    processed_at: datetime,
) -> dict[str, object]:
    """Build the request body expected by the trigger integration."""

    return {
        "call_id": event.call_id,
        "lead_id": match.crm_record_id,
        "workspace": match.workspace or event.workspace,
        "transcript": transcript,
        "summary": note.summary,
        "disposition": note.disposition,
        "duration_sec": event.duration_sec,
        "phone_from": _truncate_phone(event.phone_from),
        "phone_to": _truncate_phone(event.phone_to),
        "created_at": processed_at.isoformat(),
    }


def _truncate_phone(phone: str) -> str:
    """Limit phone fields to the downstream schema length."""

    return phone[:_MAX_PHONE_LENGTH]


def _fetch_id_token(audience: str) -> str:
    """Generate a Google-signed OIDC ID token for the configured audience."""

    request = google_requests.Request()
    token = id_token.fetch_id_token(request, audience)  # type: ignore[no-untyped-call]
    return str(token)


def _error_text(error: Exception) -> str:
    """Return a useful one-line error for structured trigger logs."""

    return str(error) or error.__class__.__name__


async def _post_trigger(
    trigger_url: str,
    token: str,
    payload: dict[str, object],
) -> httpx.Response:
    """POST the trigger request once without retries."""

    async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT_SECONDS) as client:
        return await client.post(
            trigger_url,
            headers={"Authorization": f"Bearer {token}"},
            json=payload,
        )


def _audience_from_url(trigger_url: str) -> str:
    """Use the Cloud Run service origin as the OIDC token audience."""

    parsed = urlsplit(trigger_url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("CALL_QUALITY_TRIGGER_URL must be an absolute URL.")
    return f"{parsed.scheme}://{parsed.netloc}"
