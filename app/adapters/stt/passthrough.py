"""Passthrough transcript adapter."""

from app.adapters.stt.base import STTAdapter
from app.storage.gcs import download_bytes, file_exists


class PassthroughAdapter(STTAdapter):
    """# Use when RingCentral already returns a transcript - skips STT entirely."""

    supports_diarization = False
    supports_language_detection = False

    async def transcribe(self, gcs_uri: str, *, workspace: str | None = None) -> str:
        """Fetch a pre-existing `.txt` transcript beside the audio object in GCS."""

        _ = workspace
        transcript_uri = _transcript_uri_for_audio(gcs_uri)
        if not await file_exists(transcript_uri):
            raise FileNotFoundError(f"Transcript not found at {transcript_uri}")

        return (await download_bytes(transcript_uri)).decode("utf-8")


def _transcript_uri_for_audio(gcs_uri: str) -> str:
    """Return the matching GCS transcript URI for an audio URI."""

    stem, separator, _extension = gcs_uri.rpartition(".")
    if separator:
        return f"{stem}.txt"
    return f"{gcs_uri}.txt"
