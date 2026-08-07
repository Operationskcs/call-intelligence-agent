"""Step 3: speech-to-text transcription."""

from app.adapters.stt.base import STTAdapter
from app.adapters.stt.deepgram import DeepgramAdapter
from app.adapters.stt.google_stt import GoogleSTTAdapter
from app.adapters.stt.passthrough import PassthroughAdapter
from app.config import get_settings
from app.models.call_event import CallEvent
from app.services.speaker_role import resolve_speaker_roles


async def transcribe(event: CallEvent) -> str:
    """Dispatch transcription to the configured STT adapter."""

    if not event.gcs_audio_uri:
        raise ValueError("gcs_audio_uri is required before transcription.")

    settings = get_settings()
    if settings.stt.provider == "deepgram":
        adapter: STTAdapter = DeepgramAdapter(api_key=settings.stt.deepgram_api_key)
    elif settings.stt.provider == "google":
        adapter = GoogleSTTAdapter()
    elif settings.stt.provider == "passthrough":
        adapter = PassthroughAdapter()
    else:
        raise ValueError(f"Unsupported STT_PROVIDER: {settings.stt.provider}")

    raw_transcript = await adapter.transcribe(event.gcs_audio_uri)
    return resolve_speaker_roles(raw_transcript, event)
