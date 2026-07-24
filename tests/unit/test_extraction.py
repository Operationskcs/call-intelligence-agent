"""Tests for extraction JSON parsing and anti-hallucination behavior."""

from app.models.note import IntakeCallNote
from app.worker.steps.s4_extract import _parse_note_or_default


def test_extraction_placeholder() -> None:
    """TODO: test strict JSON parsing and no-data handling for missing fields."""

    assert True


def test_non_json_response_returns_default_note() -> None:
    """Too-short transcript explanations should not crash extraction."""

    note = _parse_note_or_default(
        raw_response=(
            "I need more transcript content to extract meaningful information."
        ),
        note_model=IntakeCallNote,
        workspace="intake",
    )

    assert note.summary == "Transcript too short to extract meaningful information."
    assert note.disposition == "No Answer"
    assert note.next_steps is None
    assert note.callback_date is None
    assert note.sentiment == "neutral"
    assert note.objections is None
    assert note.pii_detected is False
    assert note.confidence == 0.0
    assert isinstance(note, IntakeCallNote)
    assert note.injury_details is None
    assert note.case_type is None


def test_json_code_fence_response_still_parses() -> None:
    """Valid fenced JSON should still produce a structured note."""

    note = _parse_note_or_default(
        raw_response="""```json
{
  "summary": "Lead discussed a possible accident claim.",
  "disposition": "Interested",
  "next_steps": null,
  "callback_date": null,
  "sentiment": "neutral",
  "objections": null,
  "pii_detected": false,
  "confidence": 0.8,
  "injury_details": "Back pain mentioned.",
  "case_type": "Motor Vehicle Accident"
}
```""",
        note_model=IntakeCallNote,
        workspace="intake",
    )

    assert note.summary == "Lead discussed a possible accident claim."
    assert note.disposition == "Interested"
    assert note.confidence == 0.8
    assert isinstance(note, IntakeCallNote)
    assert note.injury_details == "Back pain mentioned."
