import pytest

from app.models.call_event import CallEvent
from app.services.phoneburner_roles import resolve_phoneburner_roles


@pytest.mark.parametrize(("text", "expected"), [
    ("Mucho gusto, le habla Álex del equipo legal.", "Agent"),
    ("Con gusto te habla Alex del departamento jurídico.", "Agent"),
    ("¿Te habla Alex del equipo legal?", "Unknown"),
    ("Ella dijo que te habla Alex del equipo legal.", "Unknown"),
    ("No te habla Alex del equipo legal.", "Unknown"),
    ("Cuando te habla Alex del equipo legal, escucha.", "Unknown"),
    ("Con gusto te habla Ana del equipo legal.", "Unknown"),
    ("Mucho gusto, le habla Alex del centro comercial.", "Unknown"),
    ("Me dijo: «te habla Alex del equipo legal».", "Unknown"),
    ("Mucho gusto, le habla Alex del equipo legal. My car was hit.", "Unknown"),
])
def test_affiliation_scope(mock_call_event: CallEvent, text: str, expected: str) -> None:
    event = mock_call_event.model_copy(update={"agent_name": "Alex Rivera", "raw_payload": {}})
    raw = f"[Channel 1 Speaker 3]: {text}\n[Channel 1 Speaker 2]: Hola."
    r = resolve_phoneburner_roles(raw, event)
    assert r.roles["Channel 1 Speaker 3"] == expected
    assert r.roles["Channel 1 Speaker 2"] == "Unknown"
