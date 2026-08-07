"""Tests for resolving raw diarized speakers into participant roles."""

import pytest

from app.models.call_event import CallEvent, CallSource
from app.services.speaker_role import resolve_speaker_roles


def test_resolve_speaker_roles_uses_explicit_speaker_metadata() -> None:
    """Speaker metadata, when present, should beat first-speaker order."""

    event = _event(raw_payload={"speaker_roles": {"1": "agent", "0": "lead"}})
    transcript = "00:00 [Speaker 0]: hello\n00:02 [Speaker 1]: calling from the legal team"

    assert resolve_speaker_roles(transcript, event) == (
        "00:00 [Lead]: hello\n00:02 [Agent]: calling from the legal team"
    )


def test_resolve_speaker_roles_uses_agent_name_self_identification() -> None:
    """CallEvent agent metadata can resolve a speaker who introduces themself."""

    event = _event(agent_name="Alex Rivera", raw_payload={})
    transcript = (
        "00:00 [Speaker 0]: I was in an accident yesterday\n"
        "00:03 [Speaker 1]: this is Alex Rivera calling from the clinic"
    )

    assert resolve_speaker_roles(transcript, event) == (
        "00:00 [Lead]: I was in an accident yesterday\n"
        "00:03 [Agent]: this is Alex Rivera calling from the clinic"
    )


def test_resolve_speaker_roles_falls_back_and_logs_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """No reliable metadata or semantic cues should use the legacy first-speaker fallback."""

    event = _event(call_id="fallback-call", raw_payload={})
    transcript = "00:00 [Speaker 1]: hello\n00:02 [Speaker 0]: hi"

    with caplog.at_level("WARNING"):
        resolved = resolve_speaker_roles(transcript, event)

    assert resolved == "00:00 [Agent]: hello\n00:02 [Lead]: hi"
    assert (
        "speaker role resolved via fallback heuristic, no reliable metadata "
        "for call_id=fallback-call"
    ) in caplog.text


def test_resolve_speaker_roles_ignores_conflicting_metadata(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Conflicting metadata should not override stronger speaker-content evidence."""

    event = _event(
        raw_payload={
            "agent_speaker": "Speaker 0",
            "lead_speaker": "Speaker 0",
        }
    )
    transcript = (
        "00:00 [Speaker 0]: my insurance and my doctor have the accident paperwork\n"
        "00:04 [Speaker 1]: te llamo de parte de la firma de abogados"
    )

    with caplog.at_level("WARNING"):
        resolved = resolve_speaker_roles(transcript, event)

    assert resolved == (
        "00:00 [Lead]: my insurance and my doctor have the accident paperwork\n"
        "00:04 [Agent]: te llamo de parte de la firma de abogados"
    )
    assert "conflicting speaker role metadata ignored" in caplog.text


@pytest.mark.parametrize(
    ("call_id", "source", "workspace"),
    [
        ("3023088051", CallSource.PHONEBURNER, "intake"),
        ("3030755019", CallSource.PHONEBURNER, "intake"),
        ("3022841471", CallSource.PHONEBURNER, "intake"),
        ("3022469143", CallSource.PHONEBURNER, "intake"),
        ("AK3FYvSvHqmSbI1A", CallSource.RINGCENTRAL, "medhub"),
        ("ALBxmgYZCli2Bo1A", CallSource.RINGCENTRAL, "medhub"),
    ],
)
def test_resolve_speaker_roles_regression_redacted_confirmed_swaps(
    call_id: str,
    source: CallSource,
    workspace: str,
) -> None:
    """Redacted excerpts model confirmed swapped CRM transcripts without storing PII."""

    event = _event(call_id=call_id, source=source, workspace=workspace)
    transcript = (
        "00:00 [Speaker 0]: I was in an accident and my insurance sent paperwork\n"
        "00:03 [Speaker 1]: calling from the legal team to schedule an appointment"
    )

    assert resolve_speaker_roles(transcript, event) == (
        "00:00 [Lead]: I was in an accident and my insurance sent paperwork\n"
        "00:03 [Agent]: calling from the legal team to schedule an appointment"
    )


def _event(
    *,
    call_id: str = "call-123",
    source: CallSource = CallSource.PHONEBURNER,
    workspace: str = "intake",
    agent_name: str | None = "Agent Name",
    raw_payload: dict[str, object] | None = None,
) -> CallEvent:
    return CallEvent(
        call_id=call_id,
        source=source,
        workspace=workspace,
        phone_from="+15550000001",
        phone_to="+15550000002",
        patient_phone_primary="+15550000001" if source is CallSource.RINGCENTRAL else None,
        patient_phone_fallback="+15550000002" if source is CallSource.RINGCENTRAL else None,
        duration_sec=90,
        agent_id=agent_name,
        agent_name=agent_name,
        gcs_audio_uri="gs://bucket/audio.mp3",
        raw_payload=raw_payload or {},
    )
