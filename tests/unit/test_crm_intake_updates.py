"""Tests for Intake CRM Lead update guardrails."""

from typing import Any

from app.adapters.crm.intake import (
    IntakeCRMClient,
    _build_lead_update_payload,
    _format_note_markdown,
    _phone_call_name,
    _phone_call_to_phone,
)
from app.models.call_event import CallEvent, CallSource
from app.models.note import IntakeCallNote


class _CapturingIntakeClient(IntakeCRMClient):
    """Capture GraphQL payloads without making network calls."""

    def __init__(self) -> None:
        super().__init__(base_url="https://crm.example.test", api_token="test-token")
        self.variables: dict[str, Any] | None = None

    async def gql_request(
        self,
        query: str,
        variables: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Capture variables and return a fake PhoneCall id."""

        _ = query
        self.variables = variables
        return {"createPhoneCall": {"id": "phone-call-123"}}


class _CapturingIntakeLeadLookupClient(IntakeCRMClient):
    """Capture Intake Lead lookup GraphQL requests without making network calls."""

    def __init__(self) -> None:
        super().__init__(base_url="https://crm.example.test", api_token="test-token")
        self.query: str | None = None
        self.variables: dict[str, Any] | None = None

    async def gql_request(
        self,
        query: str,
        variables: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Capture the lookup query and return a fake Lead."""

        self.query = query
        self.variables = variables
        return {"leads": {"edges": [{"node": {"id": "lead-123"}}]}}


async def test_intake_phone_lookup_strips_us_country_code_prefix() -> None:
    """Intake stores Lead phone numbers without the +1 country-code prefix."""

    client = _CapturingIntakeLeadLookupClient()

    record = await client.find_record_by_phone("+14157143441")

    assert record == {"id": "lead-123"}
    assert client.variables == {"phoneNumber": "4157143441"}
    assert client.query is not None
    assert "primaryPhoneNumber" in client.query
    assert "primaryPhoneCallingCode" not in client.query


def test_lead_update_payload_preserves_existing_manual_fields() -> None:
    """Operations-owned Lead fields must never be written by call intelligence."""

    payload = _build_lead_update_payload(
        current_fields={
            "id": "lead-123",
            "summary": "Operations team entered this summary.",
            "lastContactAttemptAt": "2026-06-18T12:00:00Z",
            "contactAttemptCount": 4,
        },
        requested_fields={
            "summary": "AI-generated summary.",
            "lastContactAttemptAt": "2026-06-18T13:00:00Z",
            "contactAttemptCount": 1,
        },
    )

    assert payload == {"contactAttemptCount": 5}


def test_lead_update_payload_never_writes_empty_operations_fields() -> None:
    """Even empty operations-owned Lead fields are left untouched."""

    payload = _build_lead_update_payload(
        current_fields={
            "id": "lead-123",
            "summary": " ",
            "lastContactAttemptAt": None,
            "contactAttemptCount": None,
        },
        requested_fields={
            "summary": "AI-generated summary.",
            "lastContactAttemptAt": "2026-06-18T13:00:00Z",
            "contactAttemptCount": 1,
        },
    )

    assert payload == {"contactAttemptCount": 1}


def test_lead_update_payload_is_empty_without_contact_attempt_count() -> None:
    """Skip updateLead entirely when no allowed Lead field remains."""

    payload = _build_lead_update_payload(
        current_fields={
            "id": "lead-123",
            "summary": None,
            "lastContactAttemptAt": None,
        },
        requested_fields={
            "summary": "AI-generated summary.",
            "lastContactAttemptAt": "2026-06-18T13:00:00Z",
        },
    )

    assert payload == {}


def test_note_markdown_includes_full_transcript_and_audio_recording() -> None:
    """The CRM Note body should embed transcript text, not a transcript link."""

    note = IntakeCallNote(
        summary="Detailed call summary.",
        disposition="Interested",
        next_steps=None,
        callback_date=None,
        sentiment="neutral",
        objections="Already signed settlement papers.",
        pii_detected=False,
        confidence=0.75,
        injury_details="Rear-end collision, vehicle total loss.",
        case_type="Motor Vehicle Accident",
    )
    transcription = "[Agent]: hola\n[Lead]: quiero revisar mi accidente"

    markdown = _format_note_markdown(
        note,
        transcription,
        "gs://bucket/audio.mp3",
    )

    assert "## GCS Transcript" not in markdown
    assert "## Transcript\n\n[Agent]: hola\n[Lead]: quiero revisar mi accidente" in markdown
    assert "## Audio Recording\n\ngs://bucket/audio.mp3" in markdown
    assert "## Next Steps\n\nNone" in markdown
    assert [
        line
        for line in markdown.splitlines()
        if line.startswith("## ")
    ] == [
        "## Call Summary",
        "## Transcript",
        "## Disposition",
        "## Next Steps",
        "## Injury Details",
        "## Objections",
        "## Audio Recording",
    ]


def test_phoneburner_phone_call_to_uses_outbound_lead_phone() -> None:
    """PhoneBurner outbound PhoneCall.to should be the called Lead phone."""

    event = CallEvent(
        call_id="pb-123",
        source=CallSource.PHONEBURNER,
        workspace="intake",
        phone_from="+17604105255",
        phone_to="+14157143441",
        duration_sec=90,
        agent_id="Agent Name",
        gcs_audio_uri="gs://bucket/audio.mp3",
        raw_payload={},
    )

    assert _phone_call_to_phone(event) == "+14157143441"


async def test_create_phone_call_sets_call_name() -> None:
    """Twenty PhoneCall records should carry a stable callName."""

    client = _CapturingIntakeClient()

    phone_call_id = await client.create_phone_call(
        remote_call_id="pb-123",
        transcription="[Agent]: hello",
        direction="OUTBOUND",
        from_phone="+13055551234",
        to_phone="+13055551234",
        lead_id="lead-123",
        record_url="gs://bucket/audio.mp3",
        record_updated=True,
        call_name=_phone_call_name("pb-123"),
        agent_name="Agent Name",
    )

    assert phone_call_id == "phone-call-123"
    assert client.variables is not None
    data = client.variables["data"]
    assert data["callName"] == "Call Intelligence - pb-123"
    assert data["to"]["phoneNumber"] == "+13055551234"
