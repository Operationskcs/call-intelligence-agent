"""Lead matching orchestration."""

from datetime import date, datetime
from typing import Any

from app.adapters.crm.base import CRMClient
from app.config import get_settings
from app.models.call_event import CallEvent
from app.models.match_result import MatchMethod, MatchResult


async def match(event: CallEvent, crm_clients: dict[str, CRMClient]) -> MatchResult:
    """Match a call event to a CRM record using a conservative fallback chain."""

    client = crm_clients.get(event.workspace)
    if client is None:
        return _unmatched(event.workspace, agent_name=event.agent_name)

    for phone in (event.phone_from, event.phone_to):
        if not phone:
            continue
        record = await client.find_record_by_phone(phone)
        if record:
            return _matched(
                record,
                event.workspace,
                confidence=1.0,
                method=MatchMethod.PHONE,
                agent_name=event.agent_name,
            )

    name = _name_from_payload(event.raw_payload)
    if name:
        candidates = await client.find_record_by_name(name)
        if candidates:
            return _matched(
                candidates[0],
                event.workspace,
                confidence=0.7,
                method=MatchMethod.NAME,
                agent_name=event.agent_name,
            )

    email = _email_from_payload(event.raw_payload)
    if email:
        record = await client.find_record_by_email(email)
        if record:
            return _matched(
                record,
                event.workspace,
                confidence=0.9,
                method=MatchMethod.EMAIL,
                agent_name=event.agent_name,
            )

    birth_date = _birth_date_from_payload(event.raw_payload)
    if birth_date:
        candidates = await client.find_records_by_birth_date(birth_date)
        if candidates:
            return _matched(
                candidates[0],
                event.workspace,
                confidence=0.7,
                method=MatchMethod.DOB,
                agent_name=event.agent_name,
            )

    return _unmatched(event.workspace, agent_name=event.agent_name)


def _matched(
    record: dict[str, Any],
    workspace: str,
    confidence: float,
    method: MatchMethod,
    agent_name: str | None,
) -> MatchResult:
    """Build a MatchResult from a CRM record dictionary."""

    record_id = record.get("id") or record.get("record_id") or record.get("crm_record_id")
    return MatchResult(
        crm_record_id=str(record_id) if record_id is not None else None,
        workspace=workspace,
        confidence=confidence,
        method=method,
        requires_review=confidence < get_settings().pipeline.match_confidence_threshold,
        agent_name=agent_name,
    )


def _unmatched(workspace: str | None, *, agent_name: str | None = None) -> MatchResult:
    """Build an unmatched review-required result."""

    return MatchResult(
        crm_record_id=None,
        workspace=workspace,
        confidence=0.4,
        method=MatchMethod.UNMATCHED,
        requires_review=True,
        agent_name=agent_name,
    )


def _name_from_payload(payload: dict[str, Any]) -> str | None:
    """Extract a candidate lead name from known webhook payload fields."""

    direct_value = payload.get("name") or payload.get("full_name") or payload.get("lead_name")
    if direct_value:
        return str(direct_value)

    contact = payload.get("contact")
    if isinstance(contact, dict):
        contact_name = contact.get("name") or contact.get("full_name")
        if contact_name:
            return str(contact_name)

    return None


def _email_from_payload(payload: dict[str, Any]) -> str | None:
    """Extract a candidate lead email from known webhook payload fields."""

    direct_value = payload.get("email") or payload.get("primary_email") or payload.get("primaryEmail")
    if direct_value:
        return str(direct_value)

    contact = payload.get("contact")
    if isinstance(contact, dict):
        contact_email = contact.get("email") or contact.get("primary_email") or contact.get("primaryEmail")
        if contact_email:
            return str(contact_email)

    return None


def _birth_date_from_payload(payload: dict[str, Any]) -> date | None:
    """Extract a candidate Lead birthDate from known webhook payload fields."""

    direct_value = payload.get("birth_date") or payload.get("birthDate") or payload.get("dob")
    parsed = _parse_date(direct_value)
    if parsed is not None:
        return parsed

    contact = payload.get("contact")
    if isinstance(contact, dict):
        return _parse_date(contact.get("birth_date") or contact.get("birthDate") or contact.get("dob"))

    return None


def _parse_date(value: Any) -> date | None:
    """Parse date-only or ISO datetime values from webhook metadata."""

    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if not isinstance(value, str) or not value.strip():
        return None

    text = value.strip()
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None
