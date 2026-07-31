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

_REQUEST_TIMEOUT_SECONDS = 10.0


async def notify_call_quality_trigger(
    *,
    event: CallEvent,
    match: MatchResult,
    note: ExtractedNote,
    transcript: str,
    processed_at: datetime,
) -> None:
    """POST a successful CRM write to the call-quality trigger endpoint."""

    try:
        trigger_url = get_settings().pipeline.call_quality_trigger_url.strip()
        if not trigger_url:
            logger.warning(
                "CALL_QUALITY_TRIGGER_URL is not configured; skipping call quality trigger."
            )
            return

        audience = _audience_from_url(trigger_url)
        token = await asyncio.to_thread(_fetch_id_token, audience)
        payload = _payload(
            event=event,
            match=match,
            note=note,
            transcript=transcript,
            processed_at=processed_at,
        )

        response = await _post_trigger(trigger_url, token, payload)

        if response.is_success:
            logger.info(
                "Call quality trigger notified. call_id=%s lead_id=%s",
                event.call_id,
                match.crm_record_id,
            )
            return

        logger.warning(
            "Call quality trigger failed. call_id=%s status_code=%s response=%s",
            event.call_id,
            response.status_code,
            response.text[:500],
        )
    except Exception:
        logger.warning(
            "Call quality trigger failed. call_id=%s",
            event.call_id,
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
        "phone_from": event.phone_from,
        "phone_to": event.phone_to,
        "agent_name": event.agent_name or match.agent_name or event.agent_id,
        "created_at": processed_at.isoformat(),
    }


def _fetch_id_token(audience: str) -> str:
    """Generate a Google-signed OIDC ID token for the configured audience."""

    request = google_requests.Request()
    token = id_token.fetch_id_token(request, audience)  # type: ignore[no-untyped-call]
    return str(token)


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
