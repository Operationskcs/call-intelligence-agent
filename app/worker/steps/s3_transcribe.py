"""Step 3: speech-to-text transcription."""

from app.adapters.stt.base import STTAdapter
from app.adapters.stt.deepgram import DeepgramAdapter
from app.adapters.stt.google_stt import GoogleSTTAdapter
from app.adapters.stt.passthrough import PassthroughAdapter
from app.config import get_settings
from app.models.call_event import CallEvent, CallSource
from app.services.phoneburner_roles import resolve_phoneburner_roles
from app.services.phoneburner_semantic import vertex_refine
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

    raw_transcript = await adapter.transcribe(event.gcs_audio_uri, workspace=event.workspace)
    if event.source is CallSource.PHONEBURNER and event.workspace == "intake":
        resolved = resolve_phoneburner_roles(
            raw_transcript,
            event,
            settings.stt.phoneburner_channel_roles,
        )
        if settings.stt.phoneburner_semantic_roles:
            return await vertex_refine(raw_transcript, resolved, settings.gcp.project_id,
                                       settings.gcp.vertex_region)
        return resolved.transcript
    return resolve_speaker_roles(raw_transcript, event)
