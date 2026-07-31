"""CRM lead matching result models."""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class MatchMethod(str, Enum):
    """Supported CRM matching strategies."""

    PHONE = "phone"
    NAME = "name"
    DOB = "dob"
    EMAIL = "email"
    MANUAL = "manual"
    UNMATCHED = "unmatched"


class MatchResult(BaseModel):
    """Outcome of trying to match a call to a CRM record."""

    crm_record_id: str | None
    workspace: str | None
    confidence: float = Field(ge=0.0, le=1.0)
    method: MatchMethod
    requires_review: bool
    matched_on_phone: Literal["primary", "fallback", "none"] = "none"
    agent_name: str | None = None
