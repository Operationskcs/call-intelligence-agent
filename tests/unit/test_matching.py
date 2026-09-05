"""Tests for lead matching behavior."""

from typing import Any

from app.models.call_event import CallEvent, CallSource
from app.models.match_result import MatchMethod
from app.worker.steps.s5_match import _name_from_payload, match_lead


class _MedHubCRMClient:
    """Minimal CRM fake for MedHub phone matching tests."""

    def __init__(self) -> None:
        self.phone_calls: list[tuple[str, str | None]] = []

    async def find_record_by_phone(
        self,
        phone: str,
        fallback_phone: str | None = None,
    ) -> dict[str, Any] | None:
        """Capture primary and fallback phones and return a fallback match."""

        self.phone_calls.append((phone, fallback_phone))
        return {"id": "lead-123", "_matched_on_phone": "fallback"}

    async def find_record_by_name(self, name: str) -> list[dict[str, Any]]:
        """Unused name fallback."""

        _ = name
        return []

    async def find_record_by_email(self, email: str) -> dict[str, Any] | None:
        """Unused email fallback."""

        _ = email
        return None


class _IntakeCRMClient:
    """Minimal CRM fake for Intake phone matching tests."""

    def __init__(self) -> None:
        self.phone_calls: list[str] = []

    async def find_record_by_phone(self, phone: str) -> dict[str, Any] | None:
        """Capture the phone used for lookup and return a fake Lead."""

        self.phone_calls.append(phone)
        return {"id": "lead-intake-123"}

    async def find_record_by_name(self, name: str) -> list[dict[str, Any]]:
        """Unused name fallback."""

        _ = name
        return []

    async def find_record_by_email(self, email: str) -> dict[str, Any] | None:
        """Unused email fallback."""

        _ = email
        return None


def test_matching_placeholder() -> None:
    """TODO: test confidence scoring and phone-to-name fallback chain."""

    assert True


def test_ringcentral_name_fallback_uses_patient_name_not_agent_name() -> None:
    """RingCentral from_name is the agent, not the MedHub Lead."""

    event = CallEvent(
        call_id="rc-123",
        source=CallSource.RINGCENTRAL,
        workspace="medhub",
        phone_from="+13055551234",
        phone_to="+17865550100",
        duration_sec=60,
        agent_id="MedHub Agent",
        gcs_audio_uri=None,
        raw_payload={
            "from_name": "MedHub Agent",
            "to_name": "Patient Name",
        },
    )

    assert _name_from_payload(event) == "Patient Name"


async def test_intake_phone_matching_uses_phone_to_for_outbound_lead() -> None:
    """Intake outbound calls should match the Lead by the called phone number."""

    crm_client = _IntakeCRMClient()
    event = CallEvent(
        call_id="pb-123",
        source=CallSource.PHONEBURNER,
        workspace="intake",
        phone_from="+17604105255",
        phone_to="+14157143441",
        duration_sec=60,
        agent_id="Intake Agent",
        gcs_audio_uri=None,
        raw_payload={},
    )

    result = await match_lead(event, {"intake": crm_client})  # type: ignore[dict-item]

    assert crm_client.phone_calls == ["+14157143441"]
    assert result.crm_record_id == "lead-intake-123"
    assert result.method is MatchMethod.PHONE
    assert result.requires_review is False
    assert result.matched_on_phone == "primary"


async def test_medhub_phone_matching_passes_primary_and_fallback_candidates() -> None:
    """MedHub phone matching should pass both RingCentral patient-phone candidates."""

    crm_client = _MedHubCRMClient()
    event = CallEvent(
        call_id="rc-123",
        source=CallSource.RINGCENTRAL,
        workspace="medhub",
        phone_from="+13055551234",
        phone_to="+17865550100",
        patient_phone_primary="+13055551234",
        patient_phone_fallback="+17865550100",
        duration_sec=60,
        agent_id="MedHub Agent",
        gcs_audio_uri=None,
        raw_payload={},
    )

    result = await match_lead(event, {"medhub": crm_client})  # type: ignore[dict-item]

    assert crm_client.phone_calls == [("+13055551234", "+17865550100")]
    assert result.crm_record_id == "lead-123"
    assert result.matched_on_phone == "fallback"
