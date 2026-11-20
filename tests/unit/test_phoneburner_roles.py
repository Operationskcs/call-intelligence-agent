"""Regression cases for the PhoneBurner evidence boundary; no customer data."""

import pytest

from app.models.call_event import CallEvent
from app.services.phoneburner_roles import resolve_phoneburner_roles


def test_channel_order_does_not_assign_roles(mock_call_event: CallEvent) -> None:
    result = resolve_phoneburner_roles(
        "00:00 [Channel 0]: I was driving when my car was hit.\n"
        "00:04 [Channel 1]: I am calling from the law firm.",
        mock_call_event,
    )
    assert result.roles == {"Channel 0": "Lead", "Channel 1": "Agent"}


@pytest.mark.parametrize("identity", ["Speaker 0", "Speaker 1", "Channel 0", "Channel 1"])
def test_greeting_alone_remains_unknown(identity: str, mock_call_event: CallEvent) -> None:
    result = resolve_phoneburner_roles(f"00:00 [{identity}]: Hello.", mock_call_event)
    assert result.roles == {identity: "Unknown"}
    assert result.transcript == f"00:00 [Unknown ({identity})]: Hello."


def test_no_complementary_role_invention(mock_call_event: CallEvent) -> None:
    result = resolve_phoneburner_roles(
        "00:00 [Channel 0]: I am calling from the law firm.\n"
        "00:03 [Channel 1]: Hello.\n00:04 [Channel 2]: Joining.",
        mock_call_event,
    )
    assert result.roles == {"Channel 0": "Agent", "Channel 1": "Unknown", "Channel 2": "Unknown"}


def test_confirmed_channel_map_is_not_overridden_silently(mock_call_event: CallEvent) -> None:
    result = resolve_phoneburner_roles(
        "00:00 [Channel 0]: My car was hit.\n00:03 [Channel 1]: I am calling from the law firm.",
        mock_call_event,
        {"0": "Agent", "1": "Lead"},
    )
    assert set(result.roles.values()) == {"Unknown"}
    assert set(result.reasons.values()) == {"conflicting_evidence"}


def test_verified_map_preserves_single_active_lead_channel(mock_call_event: CallEvent) -> None:
    result = resolve_phoneburner_roles(
        "00:00 [Channel 1]: Hello.",
        mock_call_event,
        {"0": "Agent", "1": "Lead"},
    )
    assert result.transcript == "00:00 [Lead]: Hello."


def test_mono_diarized_speakers_remain_separate(mock_call_event: CallEvent) -> None:
    result = resolve_phoneburner_roles(
        "00:00 [Channel 0 Speaker 0]: My car was hit.\n"
        "00:03 [Channel 0 Speaker 1]: I am calling from the law firm.",
        mock_call_event,
    )
    assert result.roles == {"Channel 0 Speaker 0": "Lead", "Channel 0 Speaker 1": "Agent"}


def test_mixed_channel_invalidates_single_party_map(mock_call_event: CallEvent) -> None:
    result = resolve_phoneburner_roles(
        "00:00 [Channel 0 Speaker 0]: Hello.\n00:03 [Channel 0 Speaker 1]: Hi.",
        mock_call_event,
        {"0": "Agent"},
    )
    assert set(result.roles.values()) == {"Unknown"}


def test_metadata_self_identification_and_exact_text_preservation(
    mock_call_event: CallEvent,
) -> None:
    event = mock_call_event.model_copy(update={"agent_name": "Alex Rivera"})
    transcript = "00:01 [Speaker 7]: Hola, soy Alex Rivera.  ¿Cómo está?"
    result = resolve_phoneburner_roles("00:01 [Speaker 7]: Hola, soy Alex Rivera.", event)
    assert result.roles == {"Speaker 7": "Agent"}
    assert resolve_phoneburner_roles(transcript, event).transcript == (
        "00:01 [Agent]: Hola, soy Alex Rivera.  ¿Cómo está?"
    )


@pytest.mark.parametrize(
    "text",
    [
        "I have a question.",
        "I need to check.",
        "I was going to call.",
        "You said my car was hit.",
        "Usted dijo me duele mucho.",
        "Do you have my insurance?",
        "¿Me dijo mi accidente?",
        "Shut up.",
        "puta",
        "No estoy interesado.",
        'The recording says "I am calling from the law firm."',
        "Soy cliente de la firma de abogados.",
    ],
)
def test_generic_or_reported_phrases_do_not_assign_roles(
    text: str,
    mock_call_event: CallEvent,
) -> None:
    result = resolve_phoneburner_roles(f"00:00 [Speaker 0]: {text}", mock_call_event)
    assert result.roles == {"Speaker 0": "Unknown"}


def test_speaker_numbers_are_not_channel_numbers(mock_call_event: CallEvent) -> None:
    event = mock_call_event.model_copy(update={"raw_payload": {"speaker_roles": {"0": "agent"}}})
    result = resolve_phoneburner_roles("00:00 [Channel 0]: Hello.", event)
    assert result.roles == {"Channel 0": "Unknown"}


def test_conflicting_metadata_does_not_fall_back(mock_call_event: CallEvent) -> None:
    event = mock_call_event.model_copy(
        update={
            "raw_payload": {
                "speaker_roles": {"0": "agent"},
                "lead_speaker": "0",
            }
        }
    )
    result = resolve_phoneburner_roles(
        "00:00 [Speaker 0]: I am calling from the law firm.",
        event,
    )
    assert result.roles == {"Speaker 0": "Unknown"}


def test_same_identity_contradictions_remain_unknown(mock_call_event: CallEvent) -> None:
    result = resolve_phoneburner_roles(
        "00:00 [Speaker 0]: I am calling from the law firm.\n00:03 [Speaker 0]: My car was hit.",
        mock_call_event,
    )
    assert result.roles == {"Speaker 0": "Unknown"}


def test_no_transcript_content_or_names_in_decision_logs(
    mock_call_event: CallEvent,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("INFO"):
        resolve_phoneburner_roles("00:00 [Channel 0]: Private phrase.", mock_call_event)
    assert "Private phrase" not in caplog.text
    assert "unknown=1" in caplog.text
