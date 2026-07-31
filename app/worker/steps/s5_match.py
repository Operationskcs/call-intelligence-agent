"""Step 5: lead matching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

from app.adapters.crm.base import CRMClient
from app.config import get_settings
from app.matching.review_queue import send_to_review
from app.models.call_event import CallEvent, CallSource
from app.models.match_result import MatchMethod, MatchResult


@dataclass(frozen=True)
class _PhoneParts:
    """Normalized phone parts for CRM matching."""

    calling_code: str
    number: str

    @property
    def e164(self) -> str:
        """Return the normalized E.164-style phone value accepted by CRM clients."""

        return f"{self.calling_code}{self.number}"


async def match_lead(
    event: CallEvent,
    crm_clients: dict[str, CRMClient],
) -> MatchResult:
    """Match a call to a workspace Lead and send low-confidence matches to review."""

    client = crm_clients.get(event.workspace)
    if client is None:
        return await _review_required(event, "No CRM client configured for workspace.")

    phone_match = (
        await _match_medhub_by_phone(event, client)
        if event.source is CallSource.RINGCENTRAL and event.workspace == "medhub"
        else await _match_by_phone(event, client)
    )
    if phone_match is not None:
        return phone_match

    name_match = await _match_by_name(event, client)
    if name_match is not None:
        await send_to_review(event, "Name-only CRM match requires human confirmation.")
        return name_match

    email_match = await _match_by_email(event, client)
    if email_match is not None:
        return email_match

    return await _review_required(event, "No CRM match found.")


async def _match_by_phone(event: CallEvent, client: CRMClient) -> MatchResult | None:
    """Try a high-confidence phone match."""

    if not event.phone_from:
        return None

    phone = _normalize_phone(event.phone_from)
    record = await client.find_record_by_phone(phone.e164)
    if not record:
        return None

    return _match_result(
        event=event,
        record=record,
        confidence=1.0,
        method=MatchMethod.PHONE,
        matched_on_phone="primary",
    )


async def _match_medhub_by_phone(event: CallEvent, client: CRMClient) -> MatchResult | None:
    """Try MedHub patient phone candidates in RingCentral primary/fallback order."""

    primary_phone, fallback_phone = _medhub_patient_phone_candidates(event)
    if not primary_phone and not fallback_phone:
        return None

    record = await cast(Any, client).find_record_by_phone(primary_phone, fallback_phone)
    if not record:
        return None

    return _match_result(
        event=event,
        record=record,
        confidence=1.0,
        method=MatchMethod.PHONE,
        matched_on_phone=_matched_on_phone(record, default="primary"),
    )


async def _match_by_name(event: CallEvent, client: CRMClient) -> MatchResult | None:
    """Try a low-confidence name match."""

    name = _name_from_payload(event)
    if not name:
        return None

    candidates = await client.find_record_by_name(name)
    if not candidates:
        return None

    return _match_result(
        event=event,
        record=candidates[0],
        confidence=0.7,
        method=MatchMethod.NAME,
    )


async def _match_by_email(event: CallEvent, client: CRMClient) -> MatchResult | None:
    """Try a high-confidence email match when the source provides one."""

    email = _email_from_payload(event)
    if not email:
        return None

    record = await client.find_record_by_email(email)
    if not record:
        return None

    return _match_result(
        event=event,
        record=record,
        confidence=0.85,
        method=MatchMethod.EMAIL,
    )


def _match_result(
    *,
    event: CallEvent,
    record: dict[str, Any],
    confidence: float,
    method: MatchMethod,
    matched_on_phone: Literal["primary", "fallback", "none"] = "none",
) -> MatchResult:
    """Build a MatchResult and apply the configured confidence threshold."""

    crm_record_id = record.get("id") or record.get("record_id") or record.get("crm_record_id")
    threshold = get_settings().pipeline.match_confidence_threshold
    return MatchResult(
        crm_record_id=str(crm_record_id) if crm_record_id else None,
        workspace=event.workspace,
        confidence=confidence,
        method=method,
        requires_review=confidence < threshold,
        matched_on_phone=matched_on_phone,
        agent_name=event.agent_name,
    )


async def _review_required(event: CallEvent, reason: str) -> MatchResult:
    """Send an unmatched call to review and return an unmatched MatchResult."""

    await send_to_review(event, reason)
    return MatchResult(
        crm_record_id=None,
        workspace=event.workspace,
        confidence=0.0,
        method=MatchMethod.UNMATCHED,
        requires_review=True,
        agent_name=event.agent_name,
    )


def _normalize_phone(phone: str) -> _PhoneParts:
    """Strip non-digits and split into calling code plus national number."""

    digits = "".join(character for character in phone if character.isdigit())
    if len(digits) >= 10:
        number = digits[-10:]
        prefix = digits[:-10]
        calling_code = "+1" if not prefix or prefix == "1" else f"+{prefix}"
        return _PhoneParts(calling_code=calling_code, number=number)

    raise ValueError(f"Cannot normalize phone number {phone!r} for CRM matching.")


def _medhub_patient_phone_candidates(event: CallEvent) -> tuple[str, str | None]:
    """Return RingCentral patient-phone candidates for MedHub matching."""

    primary = (
        event.patient_phone_primary
        or _str_value(event.raw_payload.get("patient_phone_primary"))
        or event.phone_from
    )
    fallback = (
        event.patient_phone_fallback
        or _str_value(event.raw_payload.get("patient_phone_fallback"))
        or event.phone_to
    )
    return primary, fallback or None


def _matched_on_phone(
    record: dict[str, Any],
    *,
    default: Literal["primary", "fallback", "none"],
) -> Literal["primary", "fallback", "none"]:
    """Read the internal phone-candidate label from a MedHub CRM match."""

    label = record.get("_matched_on_phone")
    if label in {"primary", "fallback"}:
        return cast(Literal["primary", "fallback"], label)
    return default


def _name_from_payload(event: CallEvent) -> str | None:
    """Extract source-specific fallback name from raw_payload."""

    if event.source is CallSource.PHONEBURNER:
        return _join_name(
            _nested_str(event.raw_payload, "contact", "first_name"),
            _nested_str(event.raw_payload, "contact", "last_name"),
        )

    if event.source is CallSource.RINGCENTRAL:
        to_name = _str_value(event.raw_payload.get("to_name"))
        parts = to_name.split()
        if not parts:
            return None
        return _join_name(parts[0], " ".join(parts[1:]))

    return None


def _email_from_payload(event: CallEvent) -> str | None:
    """Extract source-specific fallback email from raw_payload."""

    if event.source is not CallSource.PHONEBURNER:
        return None

    email = _nested_str(event.raw_payload, "contact", "primary_email")
    return email or None


def _nested_str(payload: dict[str, Any], *keys: str) -> str:
    """Read a nested value from raw_payload as a string."""

    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return ""
        value = value.get(key)
    return _str_value(value)


def _join_name(first_name: str, last_name: str) -> str | None:
    """Join first and last name parts."""

    name = " ".join(part for part in (first_name.strip(), last_name.strip()) if part)
    return name or None


def _str_value(value: Any) -> str:
    """Convert nullable raw payload values into strings."""

    if value is None:
        return ""
    return str(value).strip()
