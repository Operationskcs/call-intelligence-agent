"""PhoneBurner webhook handler for Intake calls."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Request, status

from app.models.call_event import CallEvent, CallSource
from app.worker import pipeline
from app.worker.steps import s1_ingest

router = APIRouter()

_MIN_DURATION_SECONDS = 30
_WEBHOOK_SECRET_TOKEN_ENV = "WEBHOOK_SECRET_TOKEN"


@dataclass(frozen=True)
class PhoneBurnerCallPayload:
    """Normalized fields extracted from a PhoneBurner webhook payload."""

    call_id: str
    recording_gcs_uri: str | None
    phone_from: str
    phone_to: str
    duration: int
    connected: bool
    end_time: str | None


@router.post("/webhook/phoneburner", status_code=status.HTTP_200_OK)
async def receive_phoneburner_webhook(
    request: Request,
    x_webhook_token: str | None = Header(default=None),
) -> dict[str, str]:
    """Accept a PhoneBurner call-completed webhook and run the call pipeline."""

    _require_valid_webhook_token(x_webhook_token)

    payload = await _json_object(request)
    return await handle_phoneburner_webhook(payload)


async def handle_phoneburner_webhook(payload: dict[str, Any]) -> dict[str, str]:
    """Handle an already-authenticated PhoneBurner JSON webhook payload."""

    parsed_payload = parse_phoneburner_payload(payload)

    skip_reason = phoneburner_skip_reason(parsed_payload)
    if skip_reason is not None:
        return {
            "status": "skipped",
            "call_id": parsed_payload.call_id,
            "reason": skip_reason,
        }

    if await s1_ingest.check_idempotency(parsed_payload.call_id):
        return {"status": "already_processed", "call_id": parsed_payload.call_id}

    event = build_call_event(parsed_payload, payload)
    await pipeline.run(event)
    return {"status": "accepted", "call_id": event.call_id}


def parse_phoneburner_payload(payload: Mapping[str, Any]) -> PhoneBurnerCallPayload:
    """Extract PhoneBurner call fields from the incoming JSON payload."""

    call_id = _str_value(payload.get("call_id"))
    if not call_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing call_id")

    return PhoneBurnerCallPayload(
        call_id=call_id,
        recording_gcs_uri=_optional_str(payload.get("recording_gcs_uri")),
        phone_from=_str_value(payload.get("phone_from")),
        phone_to=_str_value(payload.get("phone_to")),
        duration=_int_value(payload.get("duration")),
        connected=_bool_value(payload.get("connected")),
        end_time=_optional_str(payload.get("end_time")),
    )


def phoneburner_skip_reason(parsed_payload: PhoneBurnerCallPayload) -> str | None:
    """Return a clear skip reason for webhook events that should not be processed."""

    if not parsed_payload.connected:
        return "not_connected"
    if parsed_payload.duration < _MIN_DURATION_SECONDS:
        return "duration_below_minimum"
    return None


def build_call_event(
    parsed_payload: PhoneBurnerCallPayload,
    raw_payload: Mapping[str, Any],
) -> CallEvent:
    """Build the normalized CallEvent consumed by the existing pipeline."""

    return CallEvent(
        call_id=parsed_payload.call_id,
        source=CallSource.PHONEBURNER,
        workspace="intake",
        phone_from=parsed_payload.phone_from,
        phone_to=parsed_payload.phone_to,
        duration_sec=parsed_payload.duration,
        agent_id=None,
        gcs_audio_uri=parsed_payload.recording_gcs_uri,
        raw_payload=dict(raw_payload),
    )


async def _json_object(request: Request) -> dict[str, Any]:
    """Parse the request body and require a JSON object."""

    payload = await request.json()
    if not isinstance(payload, dict):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Expected JSON object")
    return payload


def _require_valid_webhook_token(token: str | None) -> None:
    """Require the shared webhook token from the X-Webhook-Token header."""

    expected_token = os.getenv(_WEBHOOK_SECRET_TOKEN_ENV)
    if not expected_token or token != expected_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid webhook token")


def _bool_value(value: Any) -> bool:
    """Convert vendor truthy values into a strict bool."""

    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value == 1
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return False


def _int_value(value: Any) -> int:
    """Convert vendor duration values into seconds."""

    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _str_value(value: Any) -> str:
    """Convert required-ish vendor values into strings without inventing data."""

    if value is None:
        return ""
    return str(value)


def _optional_str(value: Any) -> str | None:
    """Convert optional vendor values into non-empty strings."""

    text = _str_value(value)
    return text or None
