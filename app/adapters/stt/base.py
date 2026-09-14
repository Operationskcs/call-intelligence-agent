"""Base protocol for speech-to-text adapters."""

from typing import Protocol


class STTAdapter(Protocol):
    """Interface required for speech-to-text providers."""

    supports_diarization: bool
    supports_language_detection: bool

    async def transcribe(self, gcs_uri: str, *, workspace: str | None = None) -> str:
        """Transcribe an audio file and return plain text with speaker labels."""
        ...
