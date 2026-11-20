"""Synthetic transcriber-to-trigger regression cases: no network or customer data."""

from datetime import UTC, datetime
from typing import Any

import pytest

from app.adapters.stt.deepgram import _format_multichannel_transcript
from app.models.call_event import CallEvent
from app.models.match_result import MatchResult
from app.models.note import IntakeCallNote
from app.services.call_quality_trigger import _payload
from app.services.phoneburner_roles import resolve_phoneburner_roles


def test_mono_word_speakers_survive_through_trigger(
    mock_call_event: CallEvent, mock_match_result: MatchResult, mock_extracted_note: IntakeCallNote
) -> None:
    words = [
        {"speaker": speaker, "start": start + i * .15, "punctuated_word": word}
        for speaker, start, phrase in ((0, 1, "Me chocaron."), (1, 3, "Le llamo de la firma."))
        for i, word in enumerate(phrase.split())
    ]
    raw = _format_multichannel_transcript({"results": {
        "channels": [{}], "utterances": [
            {"start": 1, "speaker": 0, "words": words,
             "transcript": "Me chocaron. Le llamo de la firma."}
        ],
    }})
    resolved = resolve_phoneburner_roles(raw, mock_call_event, {"0": "Agent"})
    payload = _payload(event=mock_call_event, match=mock_match_result, note=mock_extracted_note,
                       transcript=resolved.transcript, processed_at=datetime.now(UTC))
    assert payload["transcript"] == "00:01 [Lead]: Me chocaron.\n00:03 [Agent]: Le llamo de la firma."
    assert payload["role_attribution"] == "phoneburner-evidence-v2"


@pytest.mark.parametrize("channels", [1, 2])
def test_uncertain_word_alignment_never_attributes_quote(
    channels: int, mock_call_event: CallEvent
) -> None:
    u: dict[str, Any] = {"start": 2, "speaker": 0, "channel": 0,
                         "transcript": "My car was hit. Please hold!",
                         "words": [{"speaker": 0, "word": "hit"}, {"speaker": 1, "word": "hold"}]}
    raw = _format_multichannel_transcript(
        {"results": {"channels": [{} for _ in range(channels)], "utterances": [u]}}
    )
    resolved = resolve_phoneburner_roles(raw, mock_call_event, {"0": "Agent"})
    assert resolved.transcript == "00:02 [Unknown]: My car was hit. Please hold!"


@pytest.mark.parametrize("quote", [
    "They are calling from the law firm.",
    "She said I was driving.",
    "For example, my car was hit.",
    "My attorney will review your case.",
    "Mi abogado revisará su caso.",
    "Si me chocaron, ¿qué debo hacer?",
    "Ella estaba manejando.",
    "Él contó: «me chocaron».",
])
def test_other_people_or_examples_are_not_role_evidence(quote: str, mock_call_event: CallEvent) -> None:
    result = resolve_phoneburner_roles(f"[Speaker 0]: {quote}", mock_call_event)
    assert result.roles["Speaker 0"] == "Unknown"


def test_intro_does_not_assign_a_whole_unseparated_channel(mock_call_event: CallEvent) -> None:
    raw = _format_multichannel_transcript({"results": {"channels": [{"utterances": [
        {"transcript": "I am calling from the law firm."}, {"transcript": "That was rude."}
    ]}, {}]}})
    result = resolve_phoneburner_roles(raw, mock_call_event)
    assert set(result.roles.values()) == {"Unknown"}
    assert "[Agent]" not in result.transcript


def test_separate_sentences_do_not_cancel_an_intro(mock_call_event: CallEvent) -> None:
    result = resolve_phoneburner_roles(
        "[Speaker 2]: I am calling from the law firm. You said my car was hit.", mock_call_event
    )
    assert result.roles == {"Speaker 2": "Agent"}


def test_preserve_crlf_and_do_not_trust_imported_roles(mock_call_event: CallEvent) -> None:
    raw = "00:00 [Agent]: Hello.\r\n00:03 [Speaker 2]: Sí, me chocaron.\r\n"
    assert resolve_phoneburner_roles(raw, mock_call_event).transcript == (
        "00:00 [Unknown]: Hello.\r\n00:03 [Lead]: Sí, me chocaron.\r\n"
    )
