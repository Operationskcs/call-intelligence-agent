"""Twenty CRM client for the Intake workspace."""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from typing import Any

from app.adapters.crm.base import CallWriteResult, CRMClient, TwentyCRMError, TwentyGraphQLClient
from app.models.call_event import CallEvent, CallSource
from app.models.note import ExtractedNote

_LEAD_NODE_FIELDS = """
id
name { firstName lastName }
emails { primaryEmail }
phones { primaryPhoneCallingCode primaryPhoneNumber }
birthDate
contactAttemptCount
"""

_CUSTOM_FIELD_PENDING_MESSAGE = (
    "callDisposition and nextFollowUpAt are custom fields pending DATA_MODEL approval "
    "from the workspace admin."
)


class IntakeCRMClient(TwentyGraphQLClient, CRMClient):
    """Workspace-scoped Twenty GraphQL client for Intake Leads."""

    async def find_record_by_phone(self, phone: str) -> dict[str, Any] | None:
        """Find an Intake Lead by phones.primaryPhoneCallingCode and primaryPhoneNumber."""

        normalized = self.normalize_phone(phone)
        query = f"""
          query FindLeadByPhone($callingCode: String!, $phoneNumber: String!) {{
            leads(filter: {{
              phones: {{
                primaryPhoneCallingCode: {{ eq: $callingCode }}
                primaryPhoneNumber: {{ eq: $phoneNumber }}
              }}
            }}) {{
              edges {{ node {{ {_LEAD_NODE_FIELDS} }} }}
            }}
          }}
        """
        data = await self.gql_request(
            query,
            {
                "callingCode": normalized.calling_code,
                "phoneNumber": normalized.number,
            },
        )
        return self.first_edge_node(data.get("leads"))

    async def find_record_by_name(self, name: str) -> list[dict[str, Any]]:
        """Find Intake Lead candidates by exact firstName and lastName."""

        first_name, last_name = _split_name(name)
        if not first_name:
            return []

        last_name_filter = (
            "{ name: { lastName: { eq: $lastName } } }"
            if last_name
            else ""
        )
        variable_definition = "$firstName: String!, $lastName: String!" if last_name else "$firstName: String!"
        query = f"""
          query FindLeadsByName({variable_definition}) {{
            leads(filter: {{
              and: [
                {{ name: {{ firstName: {{ eq: $firstName }} }} }}
                {last_name_filter}
              ]
            }}) {{
              edges {{ node {{ {_LEAD_NODE_FIELDS} }} }}
            }}
          }}
        """
        variables = {"firstName": first_name}
        if last_name:
            variables["lastName"] = last_name

        data = await self.gql_request(query, variables)
        return self.edge_nodes(data.get("leads"))

    async def find_record_by_email(self, email: str) -> dict[str, Any] | None:
        """Find an Intake Lead by emails.primaryEmail."""

        if not email.strip():
            return None

        query = f"""
          query FindLeadByEmail($email: String!) {{
            leads(filter: {{ emails: {{ primaryEmail: {{ eq: $email }} }} }}) {{
              edges {{ node {{ {_LEAD_NODE_FIELDS} }} }}
            }}
          }}
        """
        data = await self.gql_request(query, {"email": email.strip()})
        return self.first_edge_node(data.get("leads"))

    async def find_records_by_birth_date(self, birth_date: date) -> list[dict[str, Any]]:
        """Find Intake Lead candidates by a full-day DATE_TIME birthDate range."""

        day_start = datetime.combine(birth_date, time.min, tzinfo=UTC)
        next_day = day_start + timedelta(days=1)
        query = f"""
          query FindLeadsByBirthDate($start: DateTime!, $end: DateTime!) {{
            leads(filter: {{ birthDate: {{ gte: $start, lt: $end }} }}) {{
              edges {{ node {{ {_LEAD_NODE_FIELDS} }} }}
            }}
          }}
        """
        data = await self.gql_request(
            query,
            {
                "start": _isoformat_utc(day_start),
                "end": _isoformat_utc(next_day),
            },
        )
        return self.edge_nodes(data.get("leads"))

    async def write_call_note(
        self,
        *,
        lead_id: str,
        event: CallEvent,
        note: ExtractedNote,
        transcription: str,
        transcript_uri: str | None,
    ) -> CallWriteResult:
        """Create PhoneCall, Note, combined NoteTarget, and update native Lead fields."""

        phone_call_id = await self.create_phone_call(
            remote_call_id=event.call_id,
            transcription=transcription,
            direction=_call_direction(event),
            from_phone=event.phone_from,
            to_phone=_phone_call_to_phone(event),
            lead_id=lead_id,
            record_url=event.gcs_audio_uri,
            record_updated=True,
            call_name=_phone_call_name(event.call_id),
            agent_name=event.agent_name or event.agent_id,
        )
        note_id = await self.create_note_only(
            title=f"Call Intelligence - {event.call_id}",
            markdown=_format_note_markdown(note, transcription, event.gcs_audio_uri),
        )
        note_target_id = await self.create_note_target(
            note_id=note_id,
            phone_call_id=phone_call_id,
            lead_id=lead_id,
        )
        await self.update_fields(
            lead_id,
            {
                "contactAttemptCount": 1,
            },
        )
        return CallWriteResult(
            phone_call_id=phone_call_id,
            note_id=note_id,
            note_target_id=note_target_id,
        )

    async def create_phone_call(
        self,
        *,
        remote_call_id: str,
        transcription: str,
        direction: str,
        from_phone: str,
        to_phone: str,
        lead_id: str,
        record_url: str | None,
        record_updated: bool,
        call_name: str,
        agent_name: str | None = None,
    ) -> str:
        """Create a Twenty PhoneCall linked to a Lead."""

        mutation = """
          mutation CreatePhoneCall($data: PhoneCallCreateInput!) {
            createPhoneCall(data: $data) { id }
          }
        """
        variables = {
            "data": {
                "remoteCallId": remote_call_id,
                "callName": call_name,
                "transcription": transcription,
                "direction": direction,
                "from": _call_contact(from_phone, agent_name),
                "to": _call_contact(to_phone, None),
                "leadId": lead_id,
                "recordUrl": record_url,
                "recordUpdated": record_updated,
            }
        }
        data = await self.gql_request(mutation, variables)
        phone_call = data.get("createPhoneCall")
        phone_call_id = phone_call.get("id") if isinstance(phone_call, dict) else None
        if not phone_call_id:
            raise TwentyCRMError("createPhoneCall did not return an id.")
        return str(phone_call_id)

    async def create_note(self, record_id: str, content: str, transcript_uri: str) -> str:
        """Create a note object for legacy callers.

        New call-processing code should use write_call_note because it creates
        PhoneCall and combined NoteTarget links using the confirmed schema.
        """

        _ = record_id
        return await self.create_note_only(
            title="Call Intelligence Note",
            markdown=(
                f"## Call Summary\n\n{content.strip()}\n\n"
                f"## Transcript\n\n{transcript_uri or 'No transcript text available.'}\n\n"
                "## Disposition\n\nNone\n\n"
                "## Next Steps\n\nNone\n\n"
                "## Injury Details\n\nNone\n\n"
                "## Objections\n\nNone\n\n"
                "## Audio Recording\n\nNo recording link available."
            ),
        )

    async def create_note_only(self, title: str, markdown: str) -> str:
        """Create a Twenty Note with bodyV2.markdown."""

        mutation = """
          mutation CreateNote($data: NoteCreateInput!) {
            createNote(data: $data) { id }
          }
        """
        data = await self.gql_request(
            mutation,
            {
                "data": {
                    "title": title,
                    "bodyV2": {"markdown": markdown.strip()},
                }
            },
        )
        note = data.get("createNote")
        note_id = note.get("id") if isinstance(note, dict) else None
        if not note_id:
            raise TwentyCRMError("createNote did not return an id.")
        return str(note_id)

    async def create_note_target(self, *, note_id: str, phone_call_id: str, lead_id: str) -> str:
        """Create one combined NoteTarget linked to noteId, phoneCallId, and leadId."""

        mutation = """
          mutation CreateNoteTarget($data: NoteTargetCreateInput!) {
            createNoteTarget(data: $data) {
              id
              noteId
              phoneCallId
              leadId
            }
          }
        """
        data = await self.gql_request(
            mutation,
            {
                "data": {
                    "noteId": note_id,
                    "phoneCallId": phone_call_id,
                    "leadId": lead_id,
                }
            },
        )
        note_target = data.get("createNoteTarget")
        note_target_id = note_target.get("id") if isinstance(note_target, dict) else None
        if not note_target_id:
            raise TwentyCRMError("createNoteTarget did not return an id.")
        return str(note_target_id)

    async def update_fields(self, record_id: str, fields: dict[str, Any]) -> None:
        """Update approved native Lead fields.

        The agent must never write operations-owned Lead fields such as summary
        or lastContactAttemptAt. Extracted summaries belong only in Note bodyV2.
        """

        blocked_fields = {"callDisposition", "nextFollowUpAt"} & set(fields)
        if blocked_fields:
            # TODO: custom fields pending DATA_MODEL approval from workspace admin
            raise NotImplementedError(_CUSTOM_FIELD_PENDING_MESSAGE)

        current_fields = await self.get_lead_update_state(record_id)
        update_data = _build_lead_update_payload(
            current_fields=current_fields,
            requested_fields=fields,
        )
        if not update_data:
            return

        await self._send_update_lead(record_id, update_data)

    async def _send_update_lead(self, record_id: str, update_data: dict[str, Any]) -> None:
        """Send an updateLead mutation after overwrite guards have filtered payload."""

        mutation = """
          mutation UpdateLead($id: UUID!, $data: LeadUpdateInput!) {
            updateLead(id: $id, data: $data) { id }
          }
        """
        data = await self.gql_request(
            mutation,
            {
                "id": record_id,
                "data": update_data,
            },
        )
        updated_lead = data.get("updateLead")
        if not isinstance(updated_lead, dict) or not updated_lead.get("id"):
            raise TwentyCRMError("updateLead did not return an id.")

    async def get_contact_attempt_count(self, lead_id: str) -> int:
        """Read the current contactAttemptCount for a Lead by id."""

        lead = await self.get_lead_update_state(lead_id)
        return _coerce_contact_attempt_count(lead.get("contactAttemptCount"), lead_id)

    async def get_lead_update_state(self, lead_id: str) -> dict[str, Any]:
        """Read current Lead fields that must be protected from overwrite."""

        query = """
          query GetLeadUpdateState($id: UUID!) {
            leads(filter: { id: { eq: $id } }) {
              edges {
                node {
                  id
                  contactAttemptCount
                }
              }
            }
          }
        """
        data = await self.gql_request(query, {"id": lead_id})
        lead = self.first_edge_node(data.get("leads"))
        if lead is None:
            raise TwentyCRMError(f"Lead {lead_id!r} was not found.")
        return lead

    async def update_call_disposition(self, record_id: str, call_disposition: str) -> None:
        """Update callDisposition once the workspace data model is approved."""

        _ = record_id, call_disposition
        # TODO: custom fields pending DATA_MODEL approval from workspace admin
        raise NotImplementedError(_CUSTOM_FIELD_PENDING_MESSAGE)

    async def update_next_follow_up_at(self, record_id: str, next_follow_up_at: datetime) -> None:
        """Update nextFollowUpAt once the workspace data model is approved."""

        _ = record_id, next_follow_up_at
        # TODO: custom fields pending DATA_MODEL approval from workspace admin
        raise NotImplementedError(_CUSTOM_FIELD_PENDING_MESSAGE)


def _split_name(name: str) -> tuple[str, str]:
    """Split a full name into Twenty full-name subfields."""

    parts = name.strip().split()
    if not parts:
        return "", ""
    return parts[0], " ".join(parts[1:])


def _format_note_markdown(
    note: ExtractedNote,
    transcription: str,
    gcs_audio_uri: str | None,
) -> str:
    """Build the Note bodyV2 markdown required for call notes."""

    injury_details = getattr(note, "injury_details", None) or getattr(
        note,
        "patient_complaints",
        None,
    )
    return (
        f"## Call Summary\n\n{_markdown_value(note.summary)}\n\n"
        f"## Transcript\n\n{_markdown_value(transcription)}\n\n"
        f"## Disposition\n\n{_markdown_value(note.disposition)}\n\n"
        f"## Next Steps\n\n{_markdown_value(note.next_steps)}\n\n"
        f"## Injury Details\n\n{_markdown_value(injury_details)}\n\n"
        f"## Objections\n\n{_markdown_value(note.objections)}\n\n"
        f"## Audio Recording\n\n{_markdown_value(gcs_audio_uri, fallback='No recording link available.')}"
    )


def _markdown_value(value: Any, fallback: str = "None") -> str:
    """Render optional extracted fields for markdown."""

    if value is None:
        return fallback
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else fallback
    return str(value)


def _build_lead_update_payload(
    *,
    current_fields: dict[str, Any],
    requested_fields: dict[str, Any],
) -> dict[str, Any]:
    """Build an updateLead payload for fields the agent may modify."""

    update_data: dict[str, Any] = {}

    if "contactAttemptCount" in requested_fields:
        update_data["contactAttemptCount"] = _coerce_contact_attempt_count(
            current_fields.get("contactAttemptCount"),
            str(current_fields.get("id") or "unknown"),
        ) + 1

    return update_data


def _coerce_contact_attempt_count(raw_count: Any, lead_id: str) -> int:
    """Coerce Twenty contactAttemptCount to an integer, treating null as zero."""

    if raw_count is None:
        return 0

    try:
        return int(raw_count)
    except (TypeError, ValueError) as exc:
        raise TwentyCRMError(
            f"Lead {lead_id!r} returned invalid contactAttemptCount: {raw_count!r}"
        ) from exc


def _call_contact(phone: str, name: str | None) -> dict[str, str | None]:
    """Build Twenty CALL_CONTACT metadata."""

    return {"name": name, "phoneNumber": phone}


def _phone_call_to_phone(event: CallEvent) -> str:
    """Return the lead-facing phone number for Twenty PhoneCall.to."""

    if event.source is CallSource.PHONEBURNER:
        return event.phone_from
    return event.phone_to


def _phone_call_name(call_id: str) -> str:
    """Return the stable Twenty callName for a call."""

    return f"Call Intelligence - {call_id}"


def _call_direction(event: CallEvent) -> str:
    """Return a Twenty PhoneCall direction value."""

    raw_direction = (
        event.raw_payload.get("direction")
        or event.raw_payload.get("call_direction")
        or event.raw_payload.get("callDirection")
    )
    if isinstance(raw_direction, str):
        normalized = raw_direction.strip().upper()
        if normalized in {"INBOUND", "OUTBOUND"}:
            return normalized

    return "OUTBOUND" if event.source is CallSource.PHONEBURNER else "INBOUND"


def _call_timestamp(event: CallEvent) -> str:
    """Extract the call timestamp from webhook metadata, falling back to now."""

    for key in (
        "call_timestamp",
        "callTimestamp",
        "timestamp",
        "started_at",
        "startedAt",
        "start_time",
        "startTime",
        "completed_at",
        "completedAt",
    ):
        value = event.raw_payload.get(key)
        parsed = _parse_datetime(value)
        if parsed is not None:
            return _isoformat_utc(parsed)
    return _isoformat_utc(datetime.now(UTC))


def _parse_datetime(value: Any) -> datetime | None:
    """Parse common webhook timestamp shapes."""

    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        seconds = value / 1000 if value > 10_000_000_000 else value
        return datetime.fromtimestamp(seconds, UTC)
    if isinstance(value, str) and value.strip():
        normalized = value.strip().replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    return None


def _coerce_datetime(value: Any) -> str:
    """Coerce an update value into Twenty's DATE_TIME string shape."""

    parsed = _parse_datetime(value)
    if parsed is None:
        raise ValueError(f"Expected datetime-compatible value, got {value!r}")
    return _isoformat_utc(parsed)


def _isoformat_utc(value: datetime) -> str:
    """Return an ISO 8601 UTC timestamp."""

    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
