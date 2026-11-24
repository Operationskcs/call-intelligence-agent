"""Step 3: speech-to-text transcription."""

import logging

from app.adapters.stt.base import STTAdapter
from app.adapters.stt.deepgram import DeepgramAdapter
from app.adapters.stt.google_stt import GoogleSTTAdapter
from app.adapters.stt.passthrough import PassthroughAdapter
from app.config import get_settings
from app.models.call_event import CallEvent, CallSource
from app.services.conduct_audio import suspect, uncertain_windows
from app.services.timed_conduct import Proposal
from app.services.conduct_recovery import recover
from app.services.offline_conduct import inspect_uncertainty, verified_leads, verify
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
        resolved = await verified_leads(raw_transcript, resolved, event.gcs_audio_uri)
        lead_checks = [r for r in resolved.reasons.values() if r.startswith("first_person_audio_")]
        if lead_checks:
            logging.getLogger(__name__).info("lead_original_audio accepted=%d abstained=%d",
                                             lead_checks.count("first_person_audio_confirmed"),
                                             lead_checks.count("first_person_audio_inconclusive"))
        final = resolved.transcript
        if (settings.stt.phoneburner_semantic_roles
                and "first_person_audio_inconclusive" not in resolved.reasons.values()):
            final = await vertex_refine(raw_transcript, resolved, settings.gcp.project_id,
                                       settings.gcp.vertex_region)
        hints = uncertain_windows(final, getattr(adapter, "intake_hints", ()))
        scan = "not_needed"
        if hints:
            try:
                scan = await inspect_uncertainty(raw_transcript, final, event.gcs_audio_uri, hints)
            except Exception as exc:
                scan = "unavailable"
                logging.getLogger(__name__).warning("context_audio unavailable: %s",
                                                    type(exc).__name__)
            logging.getLogger(__name__).info("context_audio windows=%d outcome=%s", len(hints), scan)
        if scan == "needs_review" or (scan == "unavailable" and any(
                h.reason == "non_agent_cue" for h in hints)):
            event._conduct_audio = "needs_review"
        elif suspect(final):
            event._conduct_audio = "needs_review"
            try:
                event._conduct_audio = await verify(raw_transcript, final, event)
            except Exception as exc:
                logging.getLogger(__name__).warning("conduct_audio unavailable: %s",
                                                    type(exc).__name__)
        if (scan != "unavailable" and not (event._conduct_audio or "").startswith("dual-asr-v1:")
                and (hints or event._conduct_audio)):
            try:
                corrected = await recover(raw_transcript, final, event, hints,
                                          unresolved=scan == "needs_review")
            except Exception as exc:
                corrected = None
                logging.getLogger(__name__).warning("conduct_recovery unavailable: %s",
                                                    type(exc).__name__)
            if corrected:
                final, event._conduct_audio = corrected
        proposal = getattr(adapter, "intake_proposal", None)
        if (isinstance(proposal, Proposal) and scan in {"not_needed", "consistent"}
                and not (event._conduct_audio or "").startswith("dual-asr-v1:")):
            revised = proposal.render(raw_transcript, business=False)
            expected = proposal.render(final, business=True)
            if revised and expected and resolve_phoneburner_roles(revised, event).transcript == expected:
                try:
                    # A second listen must corroborate the intro and the whole
                    # original sentence, including its framing and competing voice.
                    proof = await verify(revised, expected, event, parent=proposal)
                    if proof.startswith(f"dual-asr-v1:{proposal.index + proposal.clause}:"):
                        final, event._conduct_audio = expected, proof
                except Exception as exc:
                    logging.getLogger(__name__).warning("timed_conduct unavailable: %s",
                                                        type(exc).__name__)
        return final
    return resolve_speaker_roles(raw_transcript, event)
