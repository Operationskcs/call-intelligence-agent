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
