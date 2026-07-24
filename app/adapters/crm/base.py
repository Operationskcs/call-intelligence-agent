"""Base CRM protocols and Twenty GraphQL helpers."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import date
from typing import Any, Protocol

import httpx

from app.models.call_event import CallEvent
from app.models.note import ExtractedNote

logger = logging.getLogger(__name__)

__all__ = [
    "CRMClient",
    "CallWriteResult",
    "NormalizedPhone",
    "TwentyCRMError",
    "TwentyGraphQLClient",
    "asyncio",
    "httpx",
    "is_twenty_rate_limit_error",
]

_RATE_LIMIT_RETRY_COUNT = 3
_RATE_LIMIT_RETRY_DELAY_SECONDS = 60


class TwentyCRMError(RuntimeError):
    """Raised when Twenty CRM rejects or cannot complete a request."""

    def __init__(self, message: str, *, code: str | None = None) -> None:
        """Store the GraphQL extension code when Twenty returns one."""

        super().__init__(message)
        self.code = code


def is_twenty_rate_limit_error(error: TwentyCRMError) -> bool:
    """Return True when a Twenty error matches the token/rate-limit shape."""

    message = str(error)
    message_lower = message.lower()
    code = error.code or ""
    return "limit reached" in message_lower or (
        code == "BAD_USER_INPUT"
        and ("token" in message_lower or "rate" in message_lower)
    )


@dataclass(frozen=True)
class NormalizedPhone:
    """Twenty CRM phone query parts."""

    calling_code: str
    number: str


@dataclass(frozen=True)
class CallWriteResult:
    """Identifiers created by the call-note write flow."""

    phone_call_id: str
    note_id: str
    note_target_id: str


class CRMClient(Protocol):
    """Interface required for workspace-scoped Twenty CRM clients."""

    async def find_record_by_phone(self, phone: str) -> dict[str, Any] | None:
        """Find a Lead by normalized primary phone."""
        ...

    async def find_record_by_name(self, name: str) -> list[dict[str, Any]]:
        """Find candidate Leads by firstName and lastName."""
        ...

    async def find_record_by_email(self, email: str) -> dict[str, Any] | None:
        """Find a Lead by emails.primaryEmail."""
        ...

    async def find_records_by_birth_date(self, birth_date: date) -> list[dict[str, Any]]:
        """Find candidate Leads whose birthDate falls within the UTC day."""
        ...

    async def write_call_note(
        self,
        *,
        lead_id: str,
        event: CallEvent,
        note: ExtractedNote,
        transcription: str,
        transcript_uri: str | None,
    ) -> CallWriteResult:
        """Create PhoneCall, Note, NoteTarget, and update native Lead fields."""
        ...

    async def create_note(self, record_id: str, content: str, transcript_uri: str) -> str:
        """Create a note object.

        This legacy interface remains for older callers. New call-processing
        code should use write_call_note so PhoneCall and Lead links are created
        in the confirmed order.
        """
        ...

    async def update_fields(self, record_id: str, fields: dict[str, Any]) -> None:
        """Update native Lead fields derived from extraction output."""
        ...


class TwentyGraphQLClient:
    """Small Twenty GraphQL client matching the Fireflies integration pattern."""

    def __init__(self, base_url: str, api_token: str, timeout_seconds: float = 30.0) -> None:
        """Store workspace-scoped Twenty endpoint and bearer token."""

        self.base_url = base_url.rstrip("/")
        self.api_token = api_token
        self.timeout_seconds = timeout_seconds

    async def gql_request(
        self,
        query: str,
        variables: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """POST a GraphQL operation to `/graphql` with bearer auth and retries."""

        if not self.base_url:
            raise TwentyCRMError("Twenty CRM base_url is required.")
        if not self.api_token:
            raise TwentyCRMError("Twenty CRM api_token is required.")

        for attempt in range(_RATE_LIMIT_RETRY_COUNT + 1):
            try:
                return await self._gql_request_once(query, variables)
            except TwentyCRMError as exc:
                if (
                    not is_twenty_rate_limit_error(exc)
                    or attempt >= _RATE_LIMIT_RETRY_COUNT
                ):
                    raise

                logger.warning(
                    "Twenty CRM rate limit reached; retrying GraphQL request "
                    "in %d seconds. attempt=%d/%d",
                    _RATE_LIMIT_RETRY_DELAY_SECONDS,
                    attempt + 1,
                    _RATE_LIMIT_RETRY_COUNT,
                )
                await asyncio.sleep(_RATE_LIMIT_RETRY_DELAY_SECONDS)

        raise TwentyCRMError("Twenty CRM GraphQL retry loop exited unexpectedly.")

    async def _gql_request_once(
        self,
        query: str,
        variables: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """POST one GraphQL operation attempt to `/graphql` with bearer auth."""

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(
                f"{self.base_url}/graphql",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_token}",
                },
                json={"query": query, "variables": variables or {}},
            )

        if not response.is_success:
            raise TwentyCRMError(
                f"GraphQL request failed with status {response.status_code}: {response.text}"
            )

        payload = response.json()
        errors = payload.get("errors")
        if isinstance(errors, list) and errors:
            first_error = errors[0] if isinstance(errors[0], dict) else {}
            message = first_error.get("message") or "GraphQL error"
            extensions = first_error.get("extensions") or {}
            code = extensions.get("code") if isinstance(extensions, dict) else None
            suffix = f" (Code: {code})" if code else ""
            raise TwentyCRMError(f"{message}{suffix}", code=code)

        data = payload.get("data")
        if not isinstance(data, dict):
            raise TwentyCRMError("GraphQL response did not include a data object.")
        return data

    async def metadata_request(self) -> dict[str, Any]:
        """GET `/metadata` for schema inspection only."""

        if not self.base_url:
            raise TwentyCRMError("Twenty CRM base_url is required.")
        if not self.api_token:
            raise TwentyCRMError("Twenty CRM api_token is required.")

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.get(
                f"{self.base_url}/metadata",
                headers={"Authorization": f"Bearer {self.api_token}"},
            )

        if not response.is_success:
            raise TwentyCRMError(
                f"Metadata request failed with status {response.status_code}: {response.text}"
            )

        payload = response.json()
        if not isinstance(payload, dict):
            raise TwentyCRMError("Metadata response did not include a JSON object.")
        return payload

    @staticmethod
    def normalize_phone(phone: str) -> NormalizedPhone:
        """Normalize phone input for Lead.phones GraphQL filters.

        Example: `+13055551234` becomes calling code `+1` and number `3055551234`.
        """

        digits = "".join(character for character in phone if character.isdigit())
        if len(digits) == 11 and digits.startswith("1"):
            return NormalizedPhone(calling_code="+1", number=digits[1:])
        if len(digits) == 10:
            return NormalizedPhone(calling_code="+1", number=digits)
        if phone.strip().startswith("+") and len(digits) > 10:
            return NormalizedPhone(calling_code=f"+{digits[:-10]}", number=digits[-10:])
        raise ValueError(f"Cannot normalize phone number {phone!r} for Twenty CRM lookup.")

    @staticmethod
    def first_edge_node(connection: Any) -> dict[str, Any] | None:
        """Return the first node from a Twenty GraphQL connection."""

        if not isinstance(connection, dict):
            return None
        edges = connection.get("edges")
        if not isinstance(edges, list) or not edges:
            return None
        first_edge = edges[0]
        if not isinstance(first_edge, dict):
            return None
        node = first_edge.get("node")
        return node if isinstance(node, dict) else None

    @staticmethod
    def edge_nodes(connection: Any) -> list[dict[str, Any]]:
        """Return all nodes from a Twenty GraphQL connection."""

        if not isinstance(connection, dict):
            return []
        edges = connection.get("edges")
        if not isinstance(edges, list):
            return []
        return [
            edge["node"]
            for edge in edges
            if isinstance(edge, dict) and isinstance(edge.get("node"), dict)
        ]
