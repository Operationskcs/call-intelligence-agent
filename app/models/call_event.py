"""Call event model emitted by webhook receivers."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class CallSource(str, Enum):
    """Supported telephony event sources."""

    PHONEBURNER = "phoneburner"
    RINGCENTRAL = "ringcentral"


class CallEvent(BaseModel):
    """Normalized call completion event for pipeline processing."""

    call_id: str
    source: CallSource
    workspace: str
    phone_from: str
    phone_to: str
    patient_phone_primary: str | None = None
    patient_phone_fallback: str | None = None
    duration_sec: int = Field(ge=0)
    agent_id: str | None
    agent_name: str | None = None
    gcs_audio_uri: str | None
    raw_payload: dict[str, Any]
