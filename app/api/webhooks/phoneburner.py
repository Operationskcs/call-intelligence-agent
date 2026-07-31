"""PhoneBurner webhook receiver."""

from typing import Any

from fastapi import APIRouter, Header, HTTPException, Request, status

from app.models.call_event import CallEvent, CallSource

router = APIRouter()


@router.post("/webhook/phoneburner", status_code=status.HTTP_200_OK)
async def receive_phoneburner_webhook(
    request: Request,
    x_phoneburner_signature: str | None = Header(default=None),
) -> dict[str, str]:
    """Accept a PhoneBurner call completion webhook and enqueue pipeline work."""

    body = await request.body()
    if not _validate_hmac_signature(body, x_phoneburner_signature):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid signature")

    payload = await _json_object(request)
    event = _parse_call_event(payload)

    if await _is_duplicate_call(event.call_id):
        return {"status": "duplicate", "call_id": event.call_id}

    await _publish_call_completed(event)
    return {"status": "accepted", "call_id": event.call_id}


def _validate_hmac_signature(payload: bytes, signature: str | None) -> bool:
    """TODO: validate PhoneBurner HMAC once the header format is confirmed."""

    _ = payload, signature
    return True


async def _json_object(request: Request) -> dict[str, Any]:
    """Parse the request body and require a JSON object."""

    payload = await request.json()
    if not isinstance(payload, dict):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Expected JSON object")
    return payload


def _parse_call_event(payload: dict[str, Any]) -> CallEvent:
    """Map the pending PhoneBurner webhook schema into the internal event model."""

    call_id = str(payload.get("call_id") or payload.get("callId") or payload.get("id") or "")
    if not call_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing call_id")

    return CallEvent(
        call_id=call_id,
        source=CallSource.PHONEBURNER,
        workspace="intake",
        phone_from=str(payload.get("phone_from") or payload.get("from") or ""),
        phone_to=str(payload.get("phone_to") or payload.get("to") or ""),
        duration_sec=_int_value(payload.get("duration_sec") or payload.get("duration") or 0),
        agent_id=_optional_str(payload.get("agent_id") or payload.get("agentId")),
        agent_name=_phoneburner_agent_name(payload),
        gcs_audio_uri=_optional_str(payload.get("gcs_audio_uri") or payload.get("recording_gcs_uri")),
        raw_payload=payload,
    )


async def _is_duplicate_call(call_id: str) -> bool:
    """TODO: query call_audit_log for idempotency before publishing to Pub/Sub."""

    _ = call_id
    return False


async def _publish_call_completed(event: CallEvent) -> None:
    """TODO: publish serialized CallEvent to GCP Pub/Sub topic call.completed."""

    _ = event


def _int_value(value: Any) -> int:
    """Convert vendor duration values into seconds."""

    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _optional_str(value: Any) -> str | None:
    """Convert optional vendor values into strings without inventing data."""

    if value is None:
        return None
    text = str(value)
    return text or None


def _phoneburner_agent_name(payload: dict[str, Any]) -> str | None:
    """Read the PhoneBurner agent name from the call's from party."""

    from_party = payload.get("from")
    if isinstance(from_party, dict):
        name = _optional_str(from_party.get("name"))
        if name:
            return name

    return _optional_str(payload.get("from_name") or payload.get("agent_name"))
