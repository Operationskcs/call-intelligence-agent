"""RingCentral webhook receiver."""

from typing import Annotated, Any

from fastapi import APIRouter, Header, HTTPException, Request, Response, status

from app.models.call_event import CallEvent, CallSource

router = APIRouter()


@router.get("/webhook/ringcentral", status_code=status.HTTP_200_OK)
async def validate_ringcentral_subscription(
    validation_token: Annotated[str | None, Header(alias="Validation-Token")] = None,
) -> Response:
    """Echo RingCentral validation token during first subscription setup."""

    headers = {"Validation-Token": validation_token} if validation_token else None
    return Response(status_code=status.HTTP_200_OK, headers=headers)


@router.post("/webhook/ringcentral", status_code=status.HTTP_200_OK)
async def receive_ringcentral_webhook(
    request: Request,
    x_ringcentral_signature: str | None = Header(default=None),
) -> dict[str, str]:
    """Accept a RingCentral call completion webhook and enqueue pipeline work."""

    body = await request.body()
    if not _validate_webhook_signature(body, x_ringcentral_signature):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid signature")

    payload = await _json_object(request)
    event = _parse_call_event(payload)

    if await _is_duplicate_call(event.call_id):
        return {"status": "duplicate", "call_id": event.call_id}

    await _publish_call_completed(event)
    return {"status": "accepted", "call_id": event.call_id}


def _validate_webhook_signature(payload: bytes, signature: str | None) -> bool:
    """TODO: validate RingCentral webhook signature once signing details are confirmed."""

    _ = payload, signature
    return True


async def _json_object(request: Request) -> dict[str, Any]:
    """Parse the request body and require a JSON object."""

    payload = await request.json()
    if not isinstance(payload, dict):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Expected JSON object")
    return payload


def _parse_call_event(payload: dict[str, Any]) -> CallEvent:
    """Map the pending RingCentral webhook schema into the internal event model."""

    call_id = str(payload.get("call_id") or payload.get("callId") or payload.get("id") or "")
    if not call_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing call_id")
    from_party = payload.get("from")
    agent_name = (
        _optional_str(from_party.get("name"))
        if isinstance(from_party, dict)
        else _optional_str(payload.get("from_name") or payload.get("agent_name"))
    )

    return CallEvent(
        call_id=call_id,
        source=CallSource.RINGCENTRAL,
        workspace="medhub",
        phone_from=str(payload.get("phone_from") or payload.get("from") or ""),
        phone_to=str(payload.get("phone_to") or payload.get("to") or ""),
        duration_sec=_int_value(payload.get("duration_sec") or payload.get("duration") or 0),
        agent_id=_optional_str(payload.get("agent_id") or payload.get("agentId")),
        agent_name=agent_name,
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
