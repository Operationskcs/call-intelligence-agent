"""Tests for Deepgram adapter error handling."""

from types import SimpleNamespace

import pytest
from deepgram.core.api_error import ApiError

from app.adapters.stt import deepgram
from app.adapters.stt.deepgram import DeepgramAdapter, DeepgramCreditsExhaustedError


async def test_transcribe_maps_deepgram_402_to_credits_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deepgram payment-required responses should pause worker scheduling upstream."""

    async def fake_download_bytes(uri: str) -> bytes:
        assert uri == "gs://bucket/audio.mp3"
        return b"audio"

    async def fake_transcribe_file(**kwargs: object) -> object:
        assert kwargs["request"] == b"audio"
        raise ApiError(
            status_code=402,
            body={
                "err_code": "ASR_PAYMENT_REQUIRED",
                "err_msg": "Project does not have enough credits for an ASR request.",
            },
        )

    class FakeDeepgramClient:
        def __init__(self, api_key: str) -> None:
            assert api_key == "api-key"
            self.listen = SimpleNamespace(
                v1=SimpleNamespace(media=SimpleNamespace(transcribe_file=fake_transcribe_file))
            )

    monkeypatch.setattr(deepgram, "download_bytes", fake_download_bytes)
    monkeypatch.setattr(deepgram, "AsyncDeepgramClient", FakeDeepgramClient)

    with pytest.raises(DeepgramCreditsExhaustedError):
        await DeepgramAdapter(api_key="api-key").transcribe("gs://bucket/audio.mp3")


async def test_transcribe_propagates_other_deepgram_api_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only ASR payment-required responses should be remapped."""

    async def fake_download_bytes(uri: str) -> bytes:
        return b"audio"

    async def fake_transcribe_file(**kwargs: object) -> object:
        raise ApiError(status_code=500, body={"err_msg": "temporary failure"})

    class FakeDeepgramClient:
        def __init__(self, api_key: str) -> None:
            self.listen = SimpleNamespace(
                v1=SimpleNamespace(media=SimpleNamespace(transcribe_file=fake_transcribe_file))
            )

    monkeypatch.setattr(deepgram, "download_bytes", fake_download_bytes)
    monkeypatch.setattr(deepgram, "AsyncDeepgramClient", FakeDeepgramClient)

    with pytest.raises(ApiError):
        await DeepgramAdapter(api_key="api-key").transcribe("gs://bucket/audio.mp3")


async def test_transcribe_uses_multichannel_options_for_intake(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Intake PhoneBurner recordings should use channel separation instead of diarization."""

    captured_kwargs: dict[str, object] = {}

    async def fake_download_bytes(uri: str) -> bytes:
        assert uri == "gs://bucket/audio.mp3"
        return b"audio"

    async def fake_transcribe_file(**kwargs: object) -> object:
        captured_kwargs.update(kwargs)
        return {
            "results": {
                "channels": [
                    {
                        "utterances": [
                            {
                                "start": 0.2,
                                "transcript": "Hola, Alejandro, estas?",
                            }
                        ]
                    },
                    {
                        "utterances": [
                            {
                                "start": 2.0,
                                "transcript": "Alo? Bien, buenos dias.",
                            }
                        ]
                    },
                ]
            }
        }

    class FakeDeepgramClient:
        def __init__(self, api_key: str) -> None:
            assert api_key == "api-key"
            self.listen = SimpleNamespace(
                v1=SimpleNamespace(media=SimpleNamespace(transcribe_file=fake_transcribe_file))
            )

    monkeypatch.setattr(deepgram, "download_bytes", fake_download_bytes)
    monkeypatch.setattr(deepgram, "AsyncDeepgramClient", FakeDeepgramClient)

    transcript = await DeepgramAdapter(api_key="api-key").transcribe(
        "gs://bucket/audio.mp3",
        workspace="intake",
    )

    assert captured_kwargs == {
        "request": b"audio",
        "model": "nova-2-phonecall",
        "detect_language": True,
        "utterances": True,
        "multichannel": True,
        "diarize": False,
        "punctuate": True,
    }
    assert transcript == (
        "00:00 [Agent]: Hola, Alejandro, estas?\n"
        "00:02 [Lead]: Alo? Bien, buenos dias."
    )


async def test_transcribe_keeps_diarization_options_for_medhub(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-Intake calls should keep the existing Deepgram diarization path."""

    captured_kwargs: dict[str, object] = {}

    async def fake_download_bytes(uri: str) -> bytes:
        assert uri == "gs://bucket/audio.mp3"
        return b"audio"

    async def fake_transcribe_file(**kwargs: object) -> object:
        captured_kwargs.update(kwargs)
        return {
            "results": {
                "utterances": [
                    {"speaker": 0, "start": 0.0, "transcript": "calling from the clinic"},
                    {"speaker": 1, "start": 2.0, "transcript": "hello"},
                ]
            }
        }

    class FakeDeepgramClient:
        def __init__(self, api_key: str) -> None:
            assert api_key == "api-key"
            self.listen = SimpleNamespace(
                v1=SimpleNamespace(media=SimpleNamespace(transcribe_file=fake_transcribe_file))
            )

    monkeypatch.setattr(deepgram, "download_bytes", fake_download_bytes)
    monkeypatch.setattr(deepgram, "AsyncDeepgramClient", FakeDeepgramClient)

    transcript = await DeepgramAdapter(api_key="api-key").transcribe(
        "gs://bucket/audio.mp3",
        workspace="medhub",
    )

    assert captured_kwargs == {
        "request": b"audio",
        "model": "nova-2-phonecall",
        "detect_language": True,
        "utterances": True,
        "diarize": True,
    }
    assert transcript == "00:00 [Speaker 0]: calling from the clinic\n00:02 [Speaker 1]: hello"


def test_format_multichannel_transcript_interleaves_two_active_channels() -> None:
    """Channel-scoped utterances should be sorted together by start timestamp."""

    transcript = deepgram._format_multichannel_transcript(
        {
            "results": {
                "channels": [
                    {
                        "utterances": [
                            {"start": 3.4, "transcript": "Hola, Alejandro, estas?"},
                            {"start": 8.1, "transcript": "El motivo de mi llamada es..."},
                        ]
                    },
                    {
                        "utterances": [
                            {"start": 5.2, "transcript": "Alo? Bien, buenos dias."},
                        ]
                    },
                ]
            }
        }
    )

    assert transcript == (
        "00:03 [Agent]: Hola, Alejandro, estas?\n"
        "00:05 [Lead]: Alo? Bien, buenos dias.\n"
        "00:08 [Agent]: El motivo de mi llamada es..."
    )


def test_format_multichannel_transcript_labels_single_active_channel_as_agent() -> None:
    """One-sided Intake calls should be treated as Agent-only outreach."""

    transcript = deepgram._format_multichannel_transcript(
        {
            "results": {
                "channels": [
                    {
                        "utterances": [
                            {"start": 1.0, "transcript": "Hola, te llamo de la firma."},
                            {"start": 4.0, "transcript": "Solo queria dar seguimiento."},
                        ]
                    },
                    {"utterances": []},
                ]
            }
        }
    )

    assert transcript == (
        "00:01 [Agent]: Hola, te llamo de la firma.\n"
        "00:04 [Agent]: Solo queria dar seguimiento."
    )


def test_format_multichannel_transcript_groups_channel_words_when_needed() -> None:
    """Timed channel words can still produce interleaved turns without utterances."""

    transcript = deepgram._format_multichannel_transcript(
        {
            "results": {
                "channels": [
                    {
                        "alternatives": [
                            {
                                "words": [
                                    {
                                        "start": 0.0,
                                        "end": 0.3,
                                        "punctuated_word": "Hola,",
                                    },
                                    {
                                        "start": 0.4,
                                        "end": 0.8,
                                        "punctuated_word": "Alejandro.",
                                    },
                                    {
                                        "start": 3.0,
                                        "end": 3.4,
                                        "punctuated_word": "Seguimos.",
                                    },
                                ]
                            }
                        ]
                    },
                    {
                        "alternatives": [
                            {
                                "words": [
                                    {"start": 1.4, "end": 1.8, "punctuated_word": "Alo?"},
                                ]
                            }
                        ]
                    },
                ]
            }
        }
    )

    assert transcript == (
        "00:00 [Agent]: Hola, Alejandro.\n"
        "00:01 [Lead]: Alo?\n"
        "00:03 [Agent]: Seguimos."
    )


def test_format_deepgram_transcript_prefixes_utterances_with_start_timestamps() -> None:
    """Utterance-level diarization should include readable start timestamps."""

    transcript = deepgram._format_deepgram_transcript(
        {
            "results": {
                "utterances": [
                    {"speaker": 0, "start": 5.8, "transcript": "hello"},
                    {"speaker": 1, "start": 72, "transcript": "hi there"},
                ]
            }
        }
    )

    assert transcript == "00:05 [Speaker 0]: hello\n01:12 [Speaker 1]: hi there"


def test_format_deepgram_transcript_preserves_utterance_speaker_ids() -> None:
    """Utterance-level diarization should not assign participant roles."""

    transcript = deepgram._format_deepgram_transcript(
        {
            "results": {
                "utterances": [
                    {"speaker": 1, "start": 0, "transcript": "hello from intake"},
                    {"speaker": 0, "start": 3, "transcript": "hello"},
                    {"speaker": 2, "start": 5, "transcript": "joining now"},
                ]
            }
        }
    )

    assert transcript == (
        "00:00 [Speaker 1]: hello from intake\n"
        "00:03 [Speaker 0]: hello\n"
        "00:05 [Speaker 2]: joining now"
    )


def test_format_deepgram_transcript_preserves_single_speaker_id() -> None:
    """Single-speaker calls should keep the raw diarization speaker id."""

    transcript = deepgram._format_deepgram_transcript(
        {
            "results": {
                "utterances": [
                    {"speaker": 1, "start": 0, "transcript": "hello"},
                    {"speaker": 1, "start": 4, "transcript": "following up"},
                ]
            }
        }
    )

    assert transcript == "00:00 [Speaker 1]: hello\n00:04 [Speaker 1]: following up"
