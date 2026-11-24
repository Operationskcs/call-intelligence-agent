"""Deepgram speech-to-text adapter."""

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from deepgram import AsyncDeepgramClient
from deepgram.core.api_error import ApiError

from app.adapters.stt.base import STTAdapter
from app.adapters.stt.audio_hints import Hint, recognition_hints
from app.services.timed_conduct import Proposal, propose
from app.storage.gcs import download_bytes


class DeepgramCreditsExhaustedError(RuntimeError):
    """Raised when Deepgram rejects transcription because credits are exhausted."""


_WORD_TURN_GAP_SECONDS = 1.25


@dataclass(frozen=True)
class _MultichannelTurn:
    """One timestamped transcript turn from a Deepgram audio channel."""

    channel_index: int
    start: float | None
    text: str
    speaker_index: int | None = None
    unattributed: bool = False


class DeepgramAdapter(STTAdapter):
    """Deepgram Python SDK adapter using the nova-2-phonecall model."""

    supports_diarization = True
    supports_language_detection = True

    def __init__(self, api_key: str) -> None:
        """Initialize the adapter with a Deepgram API key."""

        self.api_key = api_key
        self.intake_hints: tuple[Hint, ...] = ()
        self.intake_proposal: Proposal | None = None

    async def transcribe(self, gcs_uri: str, *, workspace: str | None = None) -> str:
        """Download GCS audio bytes and transcribe them with Deepgram."""

        self.intake_hints = ()
        self.intake_proposal = None
        if not self.api_key:
            raise RuntimeError("DEEPGRAM_API_KEY must be configured before transcription.")

        audio_bytes = await download_bytes(gcs_uri)
        client = AsyncDeepgramClient(api_key=self.api_key)
        transcription_options = _transcription_options_for_workspace(workspace)
        try:
            response = await client.listen.v1.media.transcribe_file(
                request=audio_bytes,
                **transcription_options,
            )
        except ApiError as exc:
            if _is_deepgram_credits_exhausted_error(exc):
                raise DeepgramCreditsExhaustedError(
                    "Deepgram credits exhausted: ASR_PAYMENT_REQUIRED"
                ) from exc
            raise

        response_json = _response_to_dict(response)
        if _uses_multichannel_transcription(workspace):
            self.intake_hints = recognition_hints(response_json)
            raw = _format_multichannel_transcript(response_json)
            self.intake_proposal = propose(raw, response_json)
            return raw
        return _format_deepgram_transcript(response_json)


def _transcription_options_for_workspace(workspace: str | None) -> dict[str, bool | str]:
    """Return Deepgram request options for the given CRM workspace."""

    options: dict[str, bool | str] = {
        "model": "nova-2-phonecall",
        "detect_language": True,
        "utterances": True,
    }
    if _uses_multichannel_transcription(workspace):
        options.update(
            {
                "multichannel": True,
                "diarize": True,
                "punctuate": True,
            }
        )
        return options

    options["diarize"] = True
    return options


def _uses_multichannel_transcription(workspace: str | None) -> bool:
    """Request physical channels for Intake; layout is not proof of participant roles."""

    return isinstance(workspace, str) and workspace.strip().lower() == "intake"


def _is_deepgram_credits_exhausted_error(error: ApiError) -> bool:
    """Return True for Deepgram 402 ASR payment-required responses."""

    if error.status_code != 402:
        return False

    body = error.body
    if isinstance(body, Mapping):
        err_code = body.get("err_code")
        if isinstance(err_code, str) and err_code == "ASR_PAYMENT_REQUIRED":
            return True

        err_msg = body.get("err_msg")
        if isinstance(err_msg, str) and "not enough credits" in err_msg.lower():
            return True

    headers = error.headers or {}
    dg_error = headers.get("dg-error")
    return isinstance(dg_error, str) and "not enough credits" in dg_error.lower()


def _response_to_dict(response: Any) -> dict[str, Any]:
    """Convert a Deepgram SDK response model into a plain dictionary."""

    if isinstance(response, dict):
        return response
    model_dump = getattr(response, "model_dump", None)
    if callable(model_dump):
        return cast(dict[str, Any], model_dump(mode="json"))
    legacy_dict = getattr(response, "dict", None)
    if callable(legacy_dict):
        return cast(dict[str, Any], legacy_dict())
    raise TypeError("Unsupported Deepgram response type.")


def _format_deepgram_transcript(response_json: dict[str, Any]) -> str:
    """Format Deepgram diarization output into speaker-labeled transcript text."""

    utterance_lines = _format_utterances(response_json)
    if utterance_lines:
        return "\n".join(utterance_lines)

    word_lines = _format_words(response_json)
    if word_lines:
        return "\n".join(word_lines)

    transcript = _first_transcript(response_json)
    if transcript:
        return transcript

    raise ValueError("Deepgram response did not include a transcript.")


def _format_multichannel_transcript(response_json: dict[str, Any]) -> str:
    """Format multichannel output without assuming participant roles."""

    turns = _multichannel_turns(response_json)
    if turns:
        mono = len(response_json["results"]["channels"]) == 1
        return "\n".join(_format_multichannel_turn_lines(turns, mono=mono))

    raise ValueError("Deepgram response did not include a transcript.")


def _multichannel_turns(response_json: dict[str, Any]) -> list[_MultichannelTurn]:
    """Return all transcript turns from a Deepgram multichannel response."""

    channels = response_json.get("results", {}).get("channels")
    if not isinstance(channels, list) or not channels:
        return []

    root_utterance_turns = _root_multichannel_utterance_turns(response_json)
    turns: list[_MultichannelTurn] = list(root_utterance_turns)
    represented_channels = {turn.channel_index for turn in root_utterance_turns}
    for channel_index, channel in enumerate(channels):
        if not isinstance(channel, dict) or channel_index in represented_channels:
            continue
        turns.extend(_channel_turns(channel, channel_index))

    return sorted(turns, key=_multichannel_turn_sort_key)


def _root_multichannel_utterance_turns(
    response_json: dict[str, Any],
) -> list[_MultichannelTurn]:
    """Return root-level Deepgram utterances when they include channel metadata."""

    utterances = response_json.get("results", {}).get("utterances")
    if not isinstance(utterances, list):
        return []

    turns: list[_MultichannelTurn] = []
    for utterance in utterances:
        if not isinstance(utterance, dict):
            continue
        channels = response_json["results"]["channels"]
        raw_channel = utterance.get("channel")
        channel_index = 0 if raw_channel is None and len(channels) == 1 else _channel_index(
            raw_channel
        )
        if channel_index is None or channel_index >= len(channels):
            continue
        transcript = _clean_transcript(utterance.get("transcript"))
        if transcript is None:
            continue
        turns.extend(_utterance_turns(channel_index, utterance, transcript))

    return turns


def _channel_turns(channel: dict[str, Any], channel_index: int) -> list[_MultichannelTurn]:
    """Return transcript turns from one Deepgram channel."""

    utterance_turns = _channel_utterance_turns(channel, channel_index)
    if utterance_turns:
        return utterance_turns

    word_turns = _channel_word_turns(channel, channel_index)
    if word_turns:
        return word_turns

    transcript = _channel_transcript(channel)
    if transcript is None:
        return []
    return [_MultichannelTurn(channel_index=channel_index, start=None, text=transcript)]


def _channel_utterance_turns(
    channel: dict[str, Any],
    channel_index: int,
) -> list[_MultichannelTurn]:
    """Return utterance-level turns for one channel when Deepgram provides them."""

    turns: list[_MultichannelTurn] = []
    for utterance in _channel_utterance_items(channel):
        transcript = _clean_transcript(utterance.get("transcript"))
        if transcript is None:
            continue
        turns.extend(_utterance_turns(channel_index, utterance, transcript))
    return turns


def _utterance_turns(
    channel: int, utterance: dict[str, Any], transcript: str
) -> list[_MultichannelTurn]:
    """Do not attribute a multi-voice utterance to its summary speaker."""
    start = _seconds(utterance.get("start"))
    summary = _channel_index(utterance.get("speaker"))
    words = utterance.get("words")
    if isinstance(words, list):
        tokens = [(w, _word_text(w)) for w in words if isinstance(w, dict)]
        speakers = {_channel_index(w.get("speaker")) for w, token in tokens if token}
        if speakers - {None}:
            rebuilt = " ".join(token for _, token in tokens if token)
            if len(tokens) != len(words) or not all(token for _, token in tokens):
                return [_MultichannelTurn(channel, start, transcript, unattributed=True)]
            if len(speakers) == 1 and summary is not None and summary not in speakers:
                return [_MultichannelTurn(channel, start, transcript, unattributed=True)]
            if rebuilt == transcript:
                return _group_channel_words([w for w, _ in tokens], channel, start)
            # Formatting may differ. A label for only part of the wording
            # cannot vouch for an entire quote, even if all labels agree.
            lexemes = re.findall(r"\w+", transcript.casefold())
            if (not lexemes or lexemes != re.findall(r"\w+", rebuilt.casefold())
                    or len(speakers) != 1
                    or (summary is not None and summary not in speakers)):
                return [_MultichannelTurn(channel, start, transcript, unattributed=True)]
            summary = next(iter(speakers))
    return [_MultichannelTurn(channel, start, transcript, summary)]


def _channel_utterance_items(channel: dict[str, Any]) -> list[dict[str, Any]]:
    """Return channel-scoped utterance dictionaries from known Deepgram shapes."""

    direct_utterances = channel.get("utterances")
    if isinstance(direct_utterances, list):
        return [utterance for utterance in direct_utterances if isinstance(utterance, dict)]

    alternative = _first_channel_alternative(channel)
    if alternative is None:
        return []

    alternative_utterances = alternative.get("utterances")
    if not isinstance(alternative_utterances, list):
        return []
    return [utterance for utterance in alternative_utterances if isinstance(utterance, dict)]


def _channel_word_turns(
    channel: dict[str, Any],
    channel_index: int,
) -> list[_MultichannelTurn]:
    """Group channel words into rough turns when explicit utterances are absent."""

    return _group_channel_words(_channel_words(channel), channel_index)


def _group_channel_words(
    words: list[dict[str, Any]], channel_index: int, initial_start: float | None = None
) -> list[_MultichannelTurn]:

    turns: list[_MultichannelTurn] = []
    current_words: list[str] = []
    current_start: float | None = initial_start
    last_end: float | None = None
    current_speaker: int | None = None
    has_speaker_ids = any(_channel_index(w.get("speaker")) is not None for w in words)

    for word in words:
        text = _word_text(word)
        if text is None:
            continue

        start = _seconds(word.get("start"))
        end = _seconds(word.get("end"))
        speaker = _channel_index(word.get("speaker"))
        if current_words and (
            speaker != current_speaker
            or (
                start is not None
                and last_end is not None
                and start - last_end > _WORD_TURN_GAP_SECONDS
            )
        ):
            turns.append(
                _MultichannelTurn(
                    channel_index=channel_index,
                    start=current_start,
                    text=" ".join(current_words),
                    speaker_index=current_speaker,
                    unattributed=has_speaker_ids and current_speaker is None,
                )
            )
            current_words = []
            current_start = None

        if current_start is None:
            current_start = start
        current_speaker = speaker
        current_words.append(text)
        if end is not None:
            last_end = end
        elif start is not None:
            last_end = start

    if current_words:
        turns.append(
            _MultichannelTurn(
                channel_index=channel_index,
                start=current_start,
                text=" ".join(current_words),
                speaker_index=current_speaker,
                unattributed=has_speaker_ids and current_speaker is None,
            )
        )

    return turns


def _channel_words(channel: dict[str, Any]) -> list[dict[str, Any]]:
    """Return word dictionaries from one Deepgram channel alternative."""

    alternative = _first_channel_alternative(channel)
    if alternative is None:
        return []
    words = alternative.get("words")
    if not isinstance(words, list):
        return []
    return [word for word in words if isinstance(word, dict)]


def _channel_transcript(channel: dict[str, Any]) -> str | None:
    """Return a plain transcript from one Deepgram channel alternative."""

    alternative = _first_channel_alternative(channel)
    if alternative is None:
        return None
    return _clean_transcript(alternative.get("transcript"))


def _first_channel_alternative(channel: dict[str, Any]) -> dict[str, Any] | None:
    """Return the first Deepgram alternative for one channel."""

    alternatives = channel.get("alternatives")
    if not isinstance(alternatives, list) or not alternatives:
        return None
    first_alternative = alternatives[0]
    return first_alternative if isinstance(first_alternative, dict) else None


def _format_multichannel_turn_lines(turns: list[_MultichannelTurn], *, mono: bool) -> list[str]:
    """Preserve acoustic identities; business roles require independent evidence."""

    lines: list[str] = []
    active = {t.channel_index for t in turns}
    for turn in turns:
        label = f"Channel {turn.channel_index}"
        if turn.unattributed:
            label = "Unknown"
        elif mono:
            label = _speaker_label(turn.speaker_index)
        elif turn.speaker_index is not None:
            label += f" Speaker {turn.speaker_index}"
        elif len(active) == 1:
            label = f"Unseparated {label}"
        speaker_turn = f"[{label}]: {turn.text}"
        timestamp = _format_timestamp(turn.start)
        lines.append(f"{timestamp} {speaker_turn}" if timestamp else speaker_turn)
    return lines


def _multichannel_turn_sort_key(turn: _MultichannelTurn) -> tuple[int, float, int]:
    """Sort timestamped turns first, then untimed fallback turns by channel."""

    if turn.start is None:
        return (1, 0.0, turn.channel_index)
    return (0, turn.start, turn.channel_index)


def _channel_index(raw_channel: Any) -> int | None:
    """Coerce Deepgram channel metadata into a zero-based channel index."""

    if isinstance(raw_channel, bool):
        return None
    if isinstance(raw_channel, int) and raw_channel >= 0:
        return raw_channel
    if isinstance(raw_channel, str) and raw_channel.isdigit():
        return int(raw_channel)
    if isinstance(raw_channel, list) and len(raw_channel) == 1:
        return _channel_index(raw_channel[0])
    return None


def _clean_transcript(raw_transcript: Any) -> str | None:
    """Return stripped transcript text when present."""

    if not isinstance(raw_transcript, str):
        return None
    transcript = raw_transcript.strip()
    return transcript or None


def _seconds(raw_seconds: Any) -> float | None:
    """Coerce Deepgram second offsets into a valid non-negative float."""

    if isinstance(raw_seconds, bool):
        return None
    if isinstance(raw_seconds, int | float):
        seconds = float(raw_seconds)
    elif isinstance(raw_seconds, str):
        try:
            seconds = float(raw_seconds)
        except ValueError:
            return None
    else:
        return None
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


def _format_utterances(response_json: dict[str, Any]) -> list[str]:
    """Format Deepgram utterances when available."""

    utterances = response_json.get("results", {}).get("utterances")
    if not isinstance(utterances, list):
        return []

    lines: list[str] = []
    for utterance in utterances:
        if not isinstance(utterance, dict):
            continue
        transcript = utterance.get("transcript")
        if not isinstance(transcript, str) or not transcript.strip():
            continue
        speaker_turn = f"[{_speaker_label(utterance.get('speaker'))}]: {transcript.strip()}"
        timestamp = _format_timestamp(utterance.get("start"))
        lines.append(f"{timestamp} {speaker_turn}" if timestamp else speaker_turn)
    return lines


def _format_words(response_json: dict[str, Any]) -> list[str]:
    """Group consecutive diarized words by speaker."""

    words = _first_words(response_json)
    if not words:
        return []

    lines: list[str] = []
    current_speaker: Any | None = None
    current_words: list[str] = []

    for word in words:
        if not isinstance(word, dict):
            continue
        text = _word_text(word)
        if not text:
            continue

        speaker = word.get("speaker")
        if current_words and speaker != current_speaker:
            lines.append(f"[{_speaker_label(current_speaker)}]: {' '.join(current_words)}")
            current_words = []

        current_speaker = speaker
        current_words.append(text)

    if current_words:
        lines.append(f"[{_speaker_label(current_speaker)}]: {' '.join(current_words)}")

    return lines


def _first_words(response_json: dict[str, Any]) -> list[dict[str, Any]]:
    """Return words from the first Deepgram alternative."""

    alternative = _first_alternative(response_json)
    if alternative is None:
        return []
    words = alternative.get("words")
    if not isinstance(words, list):
        return []
    return [word for word in words if isinstance(word, dict)]


def _first_transcript(response_json: dict[str, Any]) -> str | None:
    """Return the first plain transcript when diarized segments are unavailable."""

    alternative = _first_alternative(response_json)
    if alternative is None:
        return None
    transcript = alternative.get("transcript")
    return transcript if isinstance(transcript, str) and transcript else None


def _first_alternative(response_json: dict[str, Any]) -> dict[str, Any] | None:
    """Return the first channel alternative from a Deepgram response."""

    channels = response_json.get("results", {}).get("channels")
    if not isinstance(channels, list) or not channels:
        return None
    first_channel = channels[0]
    if not isinstance(first_channel, dict):
        return None
    alternatives = first_channel.get("alternatives")
    if not isinstance(alternatives, list) or not alternatives:
        return None
    first_alternative = alternatives[0]
    return first_alternative if isinstance(first_alternative, dict) else None


def _word_text(word: dict[str, Any]) -> str | None:
    """Return the most readable Deepgram word token."""

    text = word.get("punctuated_word") or word.get("word")
    if not isinstance(text, str) or not text.strip():
        return None
    return text.strip()


def _speaker_label(speaker: Any) -> str:
    """Convert Deepgram speaker identifiers into raw diarization labels."""

    if speaker is None:
        return "Unknown"
    return f"Speaker {speaker}"


def _format_timestamp(raw_seconds: Any) -> str | None:
    """Format Deepgram utterance start seconds as MM:SS."""

    if not isinstance(raw_seconds, int | float):
        return None
    if raw_seconds < 0:
        return None

    total_seconds = int(raw_seconds)
    minutes, seconds = divmod(total_seconds, 60)
    return f"{minutes:02d}:{seconds:02d}"
