"""Tests for speech-to-text worker dispatch."""

from types import SimpleNamespace

import pytest

from app.models.call_event import CallEvent, CallSource
from app.worker.steps import s3_transcribe


async def test_transcribe_passes_workspace_to_stt_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Workspace context should reach STT adapters for provider-specific behavior."""

    captured: dict[str, object] = {}

    class FakeDeepgramAdapter:
        def __init__(self, api_key: str) -> None:
            assert api_key == "api-key"

        async def transcribe(self, gcs_uri: str, *, workspace: str | None = None) -> str:
            captured["gcs_uri"] = gcs_uri
            captured["workspace"] = workspace
            return "00:00 [Agent]: hola"

    event = CallEvent(
        call_id="call-123",
        source=CallSource.PHONEBURNER,
        workspace="intake",
        phone_from="+15550000001",
        phone_to="+15550000002",
        duration_sec=60,
        agent_id="agent-123",
        agent_name="Agent Name",
        gcs_audio_uri="gs://bucket/audio.mp3",
        raw_payload={},
    )

    monkeypatch.setattr(
        s3_transcribe,
        "get_settings",
        lambda: SimpleNamespace(
            stt=SimpleNamespace(
                provider="deepgram", deepgram_api_key="api-key", phoneburner_channel_roles={},
                phoneburner_semantic_roles=False,
            )
        ),
    )
    monkeypatch.setattr(s3_transcribe, "DeepgramAdapter", FakeDeepgramAdapter)
    monkeypatch.setattr(
        s3_transcribe,
        "resolve_speaker_roles",
        lambda transcript, call_event: transcript,
    )

    transcript = await s3_transcribe.transcribe(event)

    assert transcript == "00:00 [Unknown]: hola"
    assert captured == {
        "gcs_uri": "gs://bucket/audio.mp3",
        "workspace": "intake",
    }


async def test_phoneburner_raw_channels_reach_evidence_resolver(
    monkeypatch: pytest.MonkeyPatch,
    mock_call_event: CallEvent,
) -> None:
    class FakeDeepgramAdapter:
        def __init__(self, api_key: str) -> None:
            pass

        async def transcribe(self, uri: str, *, workspace: str | None = None) -> str:
            return (
                "00:00 [Channel 0 Speaker 1]: My car was hit.\n"
                "00:02 [Channel 0 Speaker 0]: I am calling from the law firm."
            )

    monkeypatch.setattr(s3_transcribe, "DeepgramAdapter", FakeDeepgramAdapter)
    monkeypatch.setattr(
        s3_transcribe,
        "get_settings",
        lambda: SimpleNamespace(
            stt=SimpleNamespace(
                provider="deepgram", deepgram_api_key="test", phoneburner_channel_roles={},
                phoneburner_semantic_roles=False,
            ),
        ),
    )
    assert await s3_transcribe.transcribe(mock_call_event) == (
        "00:00 [Lead]: My car was hit.\n00:02 [Agent]: I am calling from the law firm."
    )
