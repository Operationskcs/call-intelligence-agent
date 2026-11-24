"""Two-track discovery must never be mistaken for an audio-backed conduct verdict."""
import asyncio
import threading
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock, patch

import pytest

from app.adapters.stt.audio_hints import Hint, recognition_hints
from app.services import offline_conduct as offline
from app.services.conduct_audio import directed, probe_plan, uncertain_windows
from app.worker.steps import s3_transcribe as step
from test_intake_audio_contract import dialogue, event, resolve


def test_provider_signal_selection_without_profanity():
    def w(t, speaker=0, confidence=.99):
        return dict(start=t, end=t+.5, confidence=confidence, speaker=speaker)
    result = recognition_hints({"results": {"channels": [
        {"alternatives": [{"words": [w(1), w(2, 1), w(2.5), w(20, confidence=.42)]}]},
        {"alternatives": [{"words": [w(35)]}]}]}})
    assert result == (Hint(1, 2, "voice_switch"), Hint(2, 20, "unclear_word"))
    channels = [{"alternatives": [{"words": [w(4)]}]}]*2
    assert recognition_hints({"results": {"channels": channels}}) == (Hint(2, 4, "overlap"),)
    assert recognition_hints({"results": {"channels": [{}, {}]}}) == ()


@pytest.mark.parametrize("text", [
    "Mire, usted es un idiota, déjeme terminar.", "Listen. You are a fucking idiot. Let me finish.",
    "You are a fucking idiot.", "¡Oiga, cállate la puta boca!",
    "I have your file here. You are an idiot. We have two minutes left.",
    "¡Cállese, idiota!", "Cállate, hijueputa.",
    "Tengo el expediente. Usted es un idiota. Ahora necesito la fecha.",
])
def test_supported_full_turn_qualifies_for_audio_verification(text):
    assert directed(text)
    raw = dialogue(abuse=text)
    assert probe_plan(raw, resolve(raw), event()) is not None


@pytest.mark.parametrize("text", [
    "El otro conductor me dijo: usted es un idiota.", "You are NOT a fucking idiot.",
    "Did she say you are a fucking idiot?", "Suppose you are a fucking idiot.",
    "Mire, «usted es un idiota», así me dijeron.", "The words were: you are a fucking idiot.",
    "Cállate.", "La última perra hijueputa.", "Me dijo: cállate, hijueputa.",
    "¿Cállese, idiota?", "Yo nunca diría cállate, imbécil.",
    "Let me read this aloud. You are an idiot.", "Voy a leer el mensaje. Usted es un idiota.",
])
def test_reported_hypothetical_and_negated_are_not_misconduct(text):
    assert not directed(text)
    assert probe_plan(dialogue(abuse=text), resolve(dialogue(abuse=text)), event()) is None


def test_standalone_imperative_can_be_checked_but_not_on_the_lead_track():
    utterance = "Go fuck yourself!"
    assert directed(utterance)
    raw = dialogue(abuse=utterance)
    assert probe_plan(raw, resolve(raw), event())[:2] == (1, 2)
    other = dialogue(answer="I was stopped at a red light. " + utterance, abuse="Thank you.")
    assert probe_plan(other, resolve(other), event()) is None
    reported = raw.replace("I was stopped at a red light.", "I was stopped at a red light. He said:")
    assert probe_plan(reported, resolve(reported), event()) is None


@pytest.mark.parametrize("text", [
    "He told me to go fuck yourself.", "Did he say go fuck yourself?",
    "Don't go fuck yourself.", '"Go fuck yourself!"', "I heard this. Go fuck yourself.",
    "If I say go fuck yourself, is that what you heard?", "Go fuck yourself?"])
def test_imperative_embedded_in_a_longer_or_ambiguous_turn_never_qualifies(text):
    assert not directed(text)


def seg(t, text, confidence=.95):
    ws = text.split()
    return NS(start=t, end=t+.7, text=text, avg_logprob=-.2, no_speech_prob=.03,
              words=[NS(word=w, probability=confidence, start=t+i*.01, end=t+(i+1)*.01)
                     for i,w in enumerate(ws)])


def test_literal_match_spans_only_contiguous_high_quality_whole_segments():
    target = offline.words("Mire, usted es un idiota, déjeme terminar.").split()
    a, b = seg(4,"Mire, usted es un idiota,"), seg(4.8,"déjeme terminar.")
    assert offline._literal_bounds([a,b],target)
    assert not offline._literal_bounds([a], target)
    assert not offline._literal_bounds([a, seg(8,b.text)], target)
    assert not offline._literal_bounds([a, seg(4.8,b.text,.40)], target)
    assert not offline._literal_bounds([seg(3,"El otro conductor me dijo:"),a,b],target)
    assert not offline._literal_bounds([a,b,seg(6,"Eso dijo el conductor.")],target)
    assert not offline._literal_bounds([a,seg(4.8,"no me deje terminar.")],target)


def test_clean_preamble_cannot_hide_echo_on_the_critical_word():
    np=pytest.importorskip("numpy")
    samples=np.zeros((2, 16000*7),dtype=np.float32)
    samples[1,16000*4:16000*6]=.08
    phrase="listen you are an idiot"
    part=seg(4,phrase)
    for i,w in enumerate(part.words):
        w.start,w.end=4+i*.2,4+(i+1)*.2
    with patch.object(offline,"_wave",return_value=samples), \
         patch.object(offline,"_recognize",return_value=[part]):
        assert offline._listen(b"fixture",1,14,phrase,True)
        samples[0,int(4.8*16000):5*16000]=.08
        assert not offline._listen(b"fixture",1,14,phrase,True)


def test_blind_bilingual_scan_reads_both_original_tracks_and_ignores_outside_words():
    np = pytest.importorskip("numpy")
    samples = np.full((2,16000*13), .01, dtype=np.float32)
    # Overlap/echo need to be heard for discovery, not treated as separated proof.
    with patch.object(offline, "_wave", return_value=samples) as wav, \
         patch.object(offline, "_recognize", side_effect=[
             [seg(0,"idiota"), seg(4,"Eres un idiota.")], [seg(5,"You are an idiot.")],
         ]) as recog:
        hits = offline._scan(b"fixture", Hint(1,24,"overlap"))
    assert wav.call_args.args[1:] == (20,33)
    assert recog.call_count == 2
    assert hits == {(0,24,"idiota","eres un idiota"),(1,25,"idiot","you are an idiot")}


@pytest.mark.parametrize("changes", ["omitted", "wrong_track", "negated_primary", "mono", "unknown"])
async def test_independent_discovery_alone_cannot_attest(changes):
    raw = dialogue(abuse="Let me finish." if changes == "omitted" else "You are not an idiot."
                   if changes == "negated_primary" else "You are an idiot.")
    final = resolve(raw)
    if changes == "unknown":
        final = final.replace("[Agent]", "[Unknown]")
    ch = 0 if changes == "wrong_track" else -1 if changes == "mono" else 1
    with patch.object(offline,"download_bytes",AsyncMock(return_value=b"fixture")), \
         patch.object(offline,"_scan",return_value={(ch,14,"idiot","you are an idiot")}):
        assert await offline.inspect_uncertainty(raw,final,"synthetic",(Hint(2,14,"unclear_word"),)) == "needs_review"


@pytest.mark.parametrize("heard", [True,False])
async def test_lead_cue_requires_independent_lead_anchor_to_dismiss(heard):
    raw = dialogue().replace("Channel 1 Speaker 3]: You", "Channel 0 Speaker 0]: You")
    final = resolve(raw)
    assert uncertain_windows(final, ()) == (Hint(0,14,"non_agent_cue"),)
    with patch.object(offline,"download_bytes",AsyncMock(return_value=b"fixture")), \
         patch.object(offline,"_scan",return_value={(0,14,"idiot","you are a fucking idiot")}), \
         patch.object(offline,"_listen",return_value=heard):
        r = await offline.inspect_uncertainty(raw, final, "synthetic", uncertain_windows(final,()))
    assert r == ("consistent" if heard else "needs_review")


async def test_worker_uses_unclear_words_without_overwriting_source_or_issuing_proof(monkeypatch):
    e=event()
    raw=dialogue(answer="My car was hit.", abuse="Let me finish.")
    adapter=NS(transcribe=AsyncMock(return_value=raw),intake_hints=(Hint(2,14,"unclear_word"),))
    monkeypatch.setattr(step,"DeepgramAdapter",Mock(return_value=adapter))
    monkeypatch.setattr(step,"get_settings",lambda:NS(stt=NS(provider="deepgram",deepgram_api_key="fixture",
                        phoneburner_channel_roles={},phoneburner_semantic_roles=False)))
    with patch.object(step,"inspect_uncertainty",AsyncMock(return_value="needs_review")) as inspect, \
         patch.object(step,"verify",AsyncMock()) as certify:
        assert await step.transcribe(e) == resolve(raw)
        inspect.assert_awaited_once()
        certify.assert_not_awaited()
    assert e._conduct_audio == "needs_review"


async def test_advisory_deadline_does_not_release_busy_decoder_early(monkeypatch):
    gate=threading.Event()
    def slow(*args):
        gate.wait(2)
        return set()
    slot=threading.BoundedSemaphore(1)
    monkeypatch.setattr(offline,"_SLOT",slot)
    monkeypatch.setattr(offline,"_SCAN_TIMEOUT_SECONDS",.03)
    with patch.object(offline,"download_bytes",AsyncMock(return_value=b"fixture")), \
         patch.object(offline,"_scan",side_effect=slow) as scan:
        try:
            assert await offline.inspect_uncertainty("","","fixture",(Hint(2,4,"overlap"),
                                                       Hint(2,15,"overlap"))) == "unavailable"
            assert not slot.acquire(blocking=False)
        finally:
            gate.set()
        for _ in range(100):
            await asyncio.sleep(.01)
            if slot.acquire(blocking=False):
                slot.release()
                break
        else:
            pytest.fail("Decoder slot was not returned")
        assert scan.call_count == 1
