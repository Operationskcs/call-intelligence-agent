"""Tests for MedHub CRM workspace behavior."""

from typing import Any

from app.adapters.crm.medhub import MedHubCRMClient
from app.models.call_event import CallEvent, CallSource
from app.models.note import MedHubCallNote


class _CapturingMedHubClient(MedHubCRMClient):
    """Capture GraphQL requests without making network calls."""

    def __init__(self, responses: list[dict[str, Any]] | None = None) -> None:
        super().__init__(base_url="https://crm.example.test", api_token="test-token")
        self.query: str | None = None
        self.variables: dict[str, Any] | None = None
        self.variables_list: list[dict[str, Any] | None] = []
        self.responses = responses or [{"leads": {"edges": [{"node": {"id": "lead-123"}}]}}]

    async def gql_request(
        self,
        query: str,
        variables: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Capture the query and return one fake Lead."""

        self.query = query
        self.variables = variables
        self.variables_list.append(variables)
        if self.responses:
            return self.responses.pop(0)
        return {"leads": {"edges": []}}


async def test_medhub_phone_lookup_uses_phone_number_without_calling_code() -> None:
    """MedHub stores empty primaryPhoneCallingCode, so phone lookup ignores it."""

    client = _CapturingMedHubClient()

    record = await client.find_record_by_phone("+14807536161")

    assert record == {"id": "lead-123", "_matched_on_phone": "primary"}
    assert client.variables == {"phoneNumber": "4807536161"}
    assert client.query is not None
    assert "primaryPhoneNumber" in client.query
    assert "primaryPhoneCallingCode" not in client.query


async def test_medhub_phone_lookup_falls_back_to_from_phone_number() -> None:
    """MedHub retries with fallback patient phone when primary has no Lead."""

    client = _CapturingMedHubClient(
        responses=[
            {"leads": {"edges": []}},
            {"leads": {"edges": [{"node": {"id": "lead-456"}}]}},
        ]
    )

    record = await client.find_record_by_phone("+14807536161", "+16025550199")

    assert record == {"id": "lead-456", "_matched_on_phone": "fallback"}
    assert client.variables_list == [
        {"phoneNumber": "4807536161"},
        {"phoneNumber": "6025550199"},
    ]


async def test_medhub_note_markdown_puts_transcript_after_summary() -> None:
    """MedHub call notes should use the shared CRM section order."""

    client = _CapturingMedHubClient(
        responses=[
            {"createPhoneCall": {"id": "phone-call-123"}},
            {"createNote": {"id": "note-123"}},
            {"createNoteTarget": {"id": "note-target-123"}},
            {"leads": {"edges": [{"node": {"id": "lead-123", "contactAttemptCount": 0}}]}},
            {"updateLead": {"id": "lead-123"}},
        ]
    )
    event = CallEvent(
        call_id="rc-123",
        source=CallSource.RINGCENTRAL,
        workspace="medhub",
        phone_from="+17865550100",
        phone_to="+14807536161",
        duration_sec=120,
        agent_id="MedHub Agent",
        gcs_audio_uri="gs://bucket/medhub-audio.mp3",
        raw_payload={},
    )
    note = MedHubCallNote(
        summary="Patient asked about follow-up care.",
        disposition="Interested",
        next_steps="Schedule consult.",
        callback_date=None,
        sentiment="neutral",
        objections="Needs to confirm insurance.",
        pii_detected=False,
        confidence=0.84,
        patient_complaints="Neck pain after a collision.",
        procedures_mentioned=None,
    )

    await client.write_call_note(
        lead_id="lead-123",
        event=event,
        note=note,
        transcription="[Agent]: hello\n[Patient]: I need help",
        transcript_uri="gs://bucket/transcript.txt",
    )

    create_note_variables = client.variables_list[1]
    assert create_note_variables is not None
    markdown = create_note_variables["data"]["bodyV2"]["markdown"]
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
    assert "## Transcript\n\n[Agent]: hello\n[Patient]: I need help" in markdown
    assert "## Audio Recording\n\ngs://bucket/medhub-audio.mp3" in markdown
