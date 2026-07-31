"""RingCentral webhook receiver for MedHub call completion events."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Annotated, Any

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Request, Response, status

from app.models.call_event import CallEvent, CallSource
from app.storage import audit
from app.worker import pipeline

logger = logging.getLogger(__name__)

router = APIRouter()

_COMPLETED_RECORDING_STATUS = "Completed"
_CONNECTED_RESULT = "Call connected"
_MIN_DURATION_SECONDS = 15


@router.post(
    "/webhook/ringcentral",
    status_code=status.HTTP_200_OK,
    response_model=None,
)
async def receive_ringcentral_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    validation_token: Annotated[str | None, Header(alias="Validation-Token")] = None,
) -> Response | dict[str, str]:
    """Accept RingCentral validation challenges and call-completion notifications."""

    if validation_token:
        return Response(
            status_code=status.HTTP_200_OK,
            headers={"Validation-Token": validation_token},
        )

    payload = await _json_object(request)
    return await handle_ringcentral_notification(payload, background_tasks)


async def handle_ringcentral_notification(
    payload: dict[str, Any],
    background_tasks: BackgroundTasks,
) -> dict[str, str]:
    """Parse, validate, dedupe, and schedule pipeline processing for a notification."""

    event = parse_call_event(payload)
    if event is None:
        return {"status": "skipped"}

    if await audit.is_processed(event.call_id):
        return {"status": "duplicate", "call_id": event.call_id}

    background_tasks.add_task(pipeline.run, event)
    return {"status": "accepted", "call_id": event.call_id}


async def handle_ringcentral_webhook(
    payload: dict[str, Any],
    background_tasks: BackgroundTasks,
) -> dict[str, str]:
    """Handle an already-authenticated RingCentral notification payload."""

    return await handle_ringcentral_notification(payload, background_tasks)


def parse_call_event(payload: dict[str, Any]) -> CallEvent | None:
    """Map an eligible RingCentral push notification payload into a CallEvent."""

    record = _notification_record(payload)
    recording = _dict_value(record.get("recording"))

    if _str_value(recording.get("status")) != _COMPLETED_RECORDING_STATUS:
        logger.debug(
            "Skipping RingCentral notification without completed recording. call_id=%s",
            _call_id(record),
        )
        return None

    result = _str_value(record.get("result"))
    duration_sec = _int_value(record.get("duration"))
    if result != _CONNECTED_RESULT or duration_sec < _MIN_DURATION_SECONDS:
        logger.debug(
            "Skipping RingCentral notification. call_id=%s result=%s duration=%s",
            _call_id(record),
            result,
            duration_sec,
        )
        return None

    call_id = _call_id(record)
    if not call_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing RingCentral call_id",
        )

    recording_content_uri = _str_value(recording.get("contentUri"))
    if not recording_content_uri:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing RingCentral recording.contentUri",
        )

    from_party = _dict_value(record.get("from"))
    to_party = _dict_value(record.get("to"))
    from_phone = _str_value(from_party.get("phoneNumber"))
    to_phone = _str_value(to_party.get("phoneNumber"))
    from_name = _str_value(from_party.get("name"))
    to_name = _str_value(to_party.get("name"))

    raw_payload = dict(payload)
    raw_payload.update(
        {
            "call_id": call_id,
            "result": result,
            "duration": duration_sec,
            "from_phone_number": from_phone,
            "to_phone_number": to_phone,
            "patient_phone_primary": to_phone,
            "patient_phone_fallback": from_phone,
            "recording_content_uri": recording_content_uri,
            "recording_id": _str_value(recording.get("id")),
            "recording_status": _COMPLETED_RECORDING_STATUS,
            "from_name": from_name,
            "to_name": to_name,
            "direction": _str_value(record.get("direction")),
        }
    )

    return CallEvent(
        call_id=call_id,
        source=CallSource.RINGCENTRAL,
        workspace="medhub",
        phone_from=to_phone,
        phone_to=from_phone,
        patient_phone_primary=to_phone or None,
        patient_phone_fallback=from_phone or None,
        duration_sec=duration_sec,
        agent_id=from_name or None,
        agent_name=from_name or None,
        gcs_audio_uri=None,
        raw_payload=raw_payload,
    )


async def _json_object(request: Request) -> dict[str, Any]:
    """Parse the request body and require a JSON object."""

    payload = await request.json()
    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Expected JSON object",
        )
    return payload


def _notification_record(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the call-log shaped object from a RingCentral notification payload."""

    body = payload.get("body")
    if isinstance(body, Mapping):
        return {str(key): value for key, value in body.items()}
    return payload


def _call_id(record: Mapping[str, Any]) -> str:
    """Extract the most stable RingCentral call identifier available."""

    return (
        _str_value(record.get("call_id"))
        or _str_value(record.get("callId"))
        or _str_value(record.get("id"))
        or _str_value(record.get("telephonySessionId"))
        or _str_value(record.get("sessionId"))
    )


def _dict_value(value: Any) -> dict[str, Any]:
    """Return a mapping payload as a plain dictionary."""

    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    return {}


def _str_value(value: Any) -> str:
    """Convert nullable vendor values into trimmed strings."""

    if value is None:
        return ""
    return str(value).strip()


def _int_value(value: Any) -> int:
    """Convert nullable vendor duration values into integer seconds."""

    if value is None or value == "":
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
