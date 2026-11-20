"""Context evidence is bilingual, identity-scoped and never complementary."""

import pytest

from app.models.call_event import CallEvent
from app.services.phoneburner_roles import resolve_phoneburner_roles


@pytest.mark.parametrize(
    "text,expected",
    [
        ("This is Alex with the legal department.", "Agent"),
        ("I’m reaching out on behalf of the law office.", "Agent"),
        ("Le estoy llamando de la oficina de abogados.", "Agent"),
        ("Mi nombre es Ana de parte de la firma de abogados.", "Agent"),
        ("Si, me llevaron al hospital.", "Lead"),
        ("The other driver hit my car.", "Lead"),
        ("I got rear-ended.", "Lead"),
        ("If the other driver hit my car, what happens?", "Unknown"),
        ("Si me llevaron al hospital, es un ejemplo.", "Unknown"),
        ("You mentioned: this is Alex with the legal department.", "Unknown"),
        ("My brother works for a law office.", "Unknown"),
        ("This is a client from the law office.", "Unknown"),
        ("Soy paciente de la firma de abogados.", "Unknown"),
    ],
)
def test_scoped_evidence(mock_call_event: CallEvent, text: str, expected: str) -> None:
    assert resolve_phoneburner_roles(f"[Speaker 4]: {text}", mock_call_event).roles == {
        "Speaker 4": expected
    }


@pytest.mark.parametrize(
    "first,second,expected",
    [
        ("Do you already have an attorney?", "Were you injured?", "Agent"),
        ("¿Tiene usted un abogado?", "¿En qué fecha ocurrió el accidente?", "Agent"),
        ("Were you injured?", "Did you file a police report?", "Unknown"),
        ("Do you already have an attorney?", "Do you already have an attorney?", "Unknown"),
        ('She said "Do you already have an attorney?"', "Were you injured?", "Unknown"),
        ("Are you currently represented? Were you hurt?", "Hello.", "Unknown"),
    ],
)
def test_discovery_requires_distinct_topics_and_turns(
    mock_call_event: CallEvent,
    first: str,
    second: str,
    expected: str,
) -> None:
    tx = f"[Speaker 0]: {first}\n[Speaker 1]: Okay.\n[Speaker 0]: {second}"
    assert resolve_phoneburner_roles(tx, mock_call_event).roles == {
        "Speaker 0": expected,
        "Speaker 1": "Unknown",
    }


def test_conflicting_first_person_account_is_not_overrideable(mock_call_event: CallEvent) -> None:
    tx = (
        "[Speaker 0]: Do you have a lawyer?\n[Speaker 0]: Were you injured?\n"
        "[Speaker 0]: The other driver hit my car."
    )
    r = resolve_phoneburner_roles(tx, mock_call_event)
    assert r.roles["Speaker 0"] == "Unknown" and r.reasons["Speaker 0"] == "conflicting_evidence"
