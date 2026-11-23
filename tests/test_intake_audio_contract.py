from unittest.mock import AsyncMock, patch

import pytest

from app.models.call_event import CallEvent
from app.services.conduct_audio import probe_plan, suspect
from app.services import offline_conduct as offline
from app.services.phoneburner_roles import resolve_phoneburner_roles


def event():
    return CallEvent(call_id="fixture", workspace="intake", source="phoneburner",
                     agent_id=None, phone_from="", phone_to="", duration_sec=100,
                     raw_payload={}, gcs_audio_uri="gs://fixture/recording")


def dialogue(answer="I was stopped at a red light.", abuse="You are a fucking idiot."):
    return ("00:02 [Channel 1 Speaker 3]: I am calling from the law firm.\n"
            "00:05 [Channel 1 Speaker 3]: What happened?\n"
            f"00:08 [Channel 0 Speaker 0]: {answer}\n"
            f"00:14 [Channel 1 Speaker 3]: {abuse}")


def resolve(text):
    return resolve_phoneburner_roles(text, event()).transcript


@pytest.mark.parametrize("text", ["Yes.", "My husband was in a car accident.",
                                  "He said I was in a car accident.", "I was not the driver.",
                                  "I was in a car accident?", "Not my accident."])
def test_no_inferred_lead(text):
    assert "[Lead]" not in resolve(dialogue(text))


@pytest.mark.parametrize("spanish", [False, True])
def test_own_response_requires_direct_agent_question(spanish):
    raw = dialogue()
    if spanish:
        raw = raw.replace("What happened?", "¿Cómo fue su accidente?").replace(
            "I was stopped at a red light.", "Pues yo iba manejando.")
    assert "[Lead]" in resolve(raw)
    assert "[Lead]" not in resolve(raw.replace("00:08", "01:28"))
    assert "[Lead]" not in resolve(raw.replace("Channel 1 Speaker 3]: What happened?",
                                                "Channel 0 Speaker 0]: What happened?")
                                    .replace("¿Cómo fue su accidente?", "Buenas tardes."))


async def test_intro_and_direct_phrase_on_same_acoustic_track():
    raw = dialogue()
    assert suspect(resolve(raw))
    with patch.object(offline, "download_bytes", AsyncMock(return_value=b"fixture")), \
         patch.object(offline, "_listen", return_value=True) as check:
        assert (await offline.verify(raw, resolve(raw), event())).startswith("dual-asr-v1:3:")
        assert [c.args[1:3] for c in check.call_args_list] == [(1, 2), (1, 14)]


@pytest.mark.parametrize("checks", [[False], [True, False]])
async def test_both_acoustic_checks_mandatory(checks):
    raw = dialogue()
    with patch.object(offline, "download_bytes", AsyncMock(return_value=b"fixture")), \
         patch.object(offline, "_listen", side_effect=checks):
        assert await offline.verify(raw, resolve(raw), event()) == "needs_review"


def test_separate_first_person_facts_cannot_cross_identities():
    raw = ("00:00 [Channel 0 Speaker 1]: My insurance paid the deductible.\n"
           "00:09 [Channel 0 Speaker 1]: The driver who hit me left.")
    assert "[Lead]" in resolve(raw)
    assert "[Lead]" not in resolve(raw.replace("Speaker 1]: My", "Speaker 2]: My"))
    assert "[Lead]" not in resolve(raw.replace("The driver", "She said the driver"))


def test_operator_repeating_a_callers_prompt_is_not_a_verified_insult():
    raw = dialogue().replace("00:14", "00:12 [Channel 0 Speaker 0]: Repeat after me.\n00:14")
    assert probe_plan(raw, resolve(raw), event()) is None


@pytest.mark.parametrize("old,new", [
    ("fucking idiot", "friendly caller"), ("Speaker 3]: You", "Speaker 8]: You"),
    ("Channel 1 Speaker 3]: You", "Channel 0 Speaker 0]: You"),
    ("You are", "He said you are"), ("calling from the law firm", "here"),
    ("Speaker 3]: You", "Speaker 3]: Let me transfer you. You"),
])
def test_mismatched_words_time_voice_or_context_not_evidence(old, new):
    raw = dialogue().replace(old, new)
    assert probe_plan(raw, resolve(raw), event()) is None


async def test_secondary_unavailable_does_not_lose_primary_transcript(monkeypatch):
    from app.worker.steps import s3_transcribe as step
    monkeypatch.setenv("STT_PROVIDER", "deepgram")
    step.get_settings.cache_clear()
    raw = dialogue()
    with patch.object(step.DeepgramAdapter, "transcribe", AsyncMock(return_value=raw)), \
         patch.object(offline, "download_bytes", AsyncMock(return_value=b"fixture")), \
         patch.object(offline, "_listen", return_value=True), \
         patch.object(step, "verify", AsyncMock(side_effect=TimeoutError)):
        e = event()
        assert await step.transcribe(e) == resolve(raw)
        assert e._conduct_audio == "needs_review"


@pytest.mark.parametrize("audible", [True, False])
async def test_new_lead_fails_closed_without_original_audio(audible):
    raw = dialogue()
    resolved = resolve_phoneburner_roles(raw, event())
    with patch.object(offline, "download_bytes", AsyncMock(return_value=b"fixture")), \
         patch.object(offline, "_listen", return_value=audible):
        checked = await offline.verified_leads(raw, resolved, "gs://fixture/audio")
    assert ("[Lead]" in checked.transcript) == audible
    assert "[Agent]" in checked.transcript
    assert checked.transcript.split("]: ")[-1] == resolved.transcript.split("]: ")[-1]
