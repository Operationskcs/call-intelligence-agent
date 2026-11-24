from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock

import pytest

from app.services.conduct_audio import directed
from app.adapters.stt.audio_hints import Hint
from app.services import offline_conduct as audio
from app.services.timed_conduct import propose
from app.worker.steps import s3_transcribe as step
from test_intake_audio_contract import dialogue, event, resolve


def source(text="I need to help you but you are an idiot and please let me finish."):
    raw = dialogue(abuse=text)
    ws = [
        dict(
            speaker=3,
            start=14 + i * 0.6,
            end=14 + i * 0.6 + 0.45,
            word=w.strip(".,").lower(),
            punctuated_word=w,
        )
        for i, w in enumerate(text.split())
    ]
    return raw, {"results": {"channels": [{}, {"alternatives": [{"words": ws}]}]}}


@pytest.mark.parametrize(
    "turn",
    [
        "I need to help you but you are an idiot and please let me finish.",
        "Necesito entender lo que pasó pero cállese, idiota y déjeme terminar.",
    ],
)
def test_contextual_clause_at_real_word_time(turn):
    raw, data = source(turn)
    p = propose(raw, data)
    assert directed(turn) and p and p.hint.at > 14 and p.clause == 1
    assert " ".join(s for _, s in p.pieces) == turn
    assert p.render(resolve(raw), business=True)


def test_long_original_stays_intact_until_bounded_quote_can_be_verified():
    turn = (
        "I need to understand what happened and I am trying to review your case "
        "and I want to finish your intake and I need you to listen to me "
        "and I am trying to help you but you are an idiot and please let me finish."
    )
    assert len(turn) > 180
    raw, data = source(turn)
    plan = propose(raw, data)
    assert plan and plan.hint.at > 14 + 12
    assert " ".join(s for _, s in plan.pieces) == turn


@pytest.mark.parametrize(
    "turn",
    [
        "I was quoting my supervisor, you are an idiot and please let me finish.",
        "She was like. You are an idiot.",
        "Me dice. Usted es un idiota.",
        "I need to help you but he thinks you are an idiot.",
        "I need to help you but you are not an idiot.",
        "Quiero ayudarle pero ¿usted es un idiota?",
    ],
)
def test_no_context_free_extraction(turn):
    raw, data = source(turn)
    assert not directed(turn) and propose(raw, data) is None


@pytest.mark.parametrize(
    "kind", ["missing_time", "wrong_speaker", "edited_word", "overlap", "lead"]
)
def test_original_metadata_or_role_conflicts(kind):
    raw, data = source()
    ws = data["results"]["channels"][1]["alternatives"][0]["words"]
    if kind == "lead":
        p = propose(raw, data)
        assert p and p.render(resolve(raw).replace("[Agent]", "[Lead]"), business=True) is None
        return
    if kind == "missing_time":
        del ws[5]["start"]
    if kind == "wrong_speaker":
        ws[5]["speaker"] = 2
    if kind == "edited_word":
        ws[5]["punctuated_word"] = "not"
    if kind == "overlap":
        raw += "\n00:16 [Channel 0 Speaker 0]: No."
    assert propose(raw, data) is None


@pytest.mark.parametrize(
    "outcome,valid",
    [("consistent", True), ("consistent", False), ("needs_review", True), ("unavailable", True)],
)
async def test_only_exact_verified_proposal_can_replace_turn(monkeypatch, outcome, valid):
    raw, data = source()
    p = propose(raw, data)
    assert p
    e = event()
    adapter = NS(
        transcribe=AsyncMock(return_value=raw),
        intake_hints=(Hint(2, 14, "unclear_word"),),
        intake_proposal=p,
    )
    monkeypatch.setattr(step, "DeepgramAdapter", Mock(return_value=adapter))
    monkeypatch.setattr(
        step,
        "get_settings",
        lambda: NS(
            stt=NS(
                provider="deepgram",
                deepgram_api_key="test",
                phoneburner_channel_roles={},
                phoneburner_semantic_roles=False,
            )
        ),
    )
    monkeypatch.setattr(step, "recover", AsyncMock(return_value=None))
    monkeypatch.setattr(step, "verified_leads", AsyncMock(side_effect=lambda _, r, uri: r))
    inspect = AsyncMock(return_value=outcome)
    monkeypatch.setattr(step, "inspect_uncertainty", inspect)
    monkeypatch.setattr(
        step,
        "verify",
        AsyncMock(
            side_effect=[
                "needs_review",
                f"dual-asr-v1:{p.index + p.clause}:hash" if valid else "needs_review",
            ]
        ),
    )
    result = await step.transcribe(e)
    inspect.assert_awaited_once()
    assert result == (
        p.render(resolve(raw), business=True) if valid and outcome == "consistent" else resolve(raw)
    )


@pytest.mark.parametrize(
    "change",
    [
        "none",
        "lead",
        "echo",
        "negation",
        "truncation",
        "low_confidence",
        "wrong_time",
        "inaudible",
        "same_cue_other_voice",
    ],
)
def test_complete_acoustic_parent_not_just_a_substring(monkeypatch, change):
    import numpy as np
    from test_contextual_audio import seg

    raw, data = source()
    p = propose(raw, data)
    assert p
    ws = data["results"]["channels"][1]["alternatives"][0]["words"]
    s = seg(2, p.original)
    for w, x in zip(s.words, ws, strict=True):
        w.start, w.end = x["start"] - 12, x["end"] - 12
    s.end = s.words[-1].end
    x = np.zeros((2, 16 * 16000), dtype=np.float32)
    x[0 if change == "lead" else 1] = 0.1
    if change == "echo":
        x[0] = 0.09
    if change == "inaudible":
        x.fill(0)
    if change == "negation":
        s.words[6].word = "not"
    if change == "truncation":
        s.words = s.words[:-2]
    if change == "low_confidence":
        s.words[7].probability = 0.22
    if change == "wrong_time":
        for w in s.words:
            w.start += 3
            w.end += 3
    monkeypatch.setattr(audio, "_wave", lambda *args: x)
    monkeypatch.setattr(
        audio,
        "_recognize",
        Mock(side_effect=[[seg(2, "idiot")] if change == "same_cue_other_voice" else [], [s]]),
    )
    assert audio._context_listen(b"synthetic", 1, p) is (change == "none")
