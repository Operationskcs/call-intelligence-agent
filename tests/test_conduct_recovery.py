import hashlib
import threading
from unittest.mock import AsyncMock, patch

import pytest

from app.adapters.stt.audio_hints import Hint
from app.services import conduct_recovery as r
from app.worker.steps import s3_transcribe as step
from test_intake_audio_contract import dialogue, event, resolve
from test_contextual_audio import seg


def spoken(text):
    out = seg(4, text)
    for i, w in enumerate(out.words):
        w.start, w.end = 4 + i * 0.18, 4 + (i + 1) * 0.18
    out.end = out.words[-1].end
    return out


@pytest.mark.parametrize(
    "txt,hints,enabled",
    [
        ("You are in idiot.", (), True),
        ("You are an ideal.", (Hint(2, 14, "unclear_word"),), True),
        ("No, you are an idiot.", (), False),
        ("He is an idiot.", (), False),
        ("She said: you are an idiot.", (), False),
        ("You are an idiot?", (), False),
        ("You are in idiot.", (Hint(2, 5, "unclear_word"),), False),
    ],
)
def test_no_unsafe_or_unrelated_target(txt, hints, enabled):
    raw = dialogue(abuse=txt)
    assert bool(r.target(raw, resolve(raw), hints)) is enabled


def test_changed_role_or_competing_voice_never_recoverable():
    raw = dialogue(abuse="You are in idiot.")
    assert not r.target(raw, resolve(raw).replace("[Agent]", "[Lead]"), ())
    raw += "\n00:20 [Channel 1 Speaker 4]: Hi."
    assert not r.target(raw, resolve(raw), ())


@pytest.mark.parametrize(
    "other,echo,accepted",
    [
        ("You are an idiot.", False, True),
        ("You are a bitch.", False, False),
        ("You are an idiot.", True, False),
        ("You are not an idiot.", False, False),
    ],
)
def test_two_different_recognizers_must_agree_on_isolated_words(other, echo, accepted):
    import numpy as np

    x = np.zeros((2, 16 * 16000), dtype=np.float32)
    x[1] = 0.15
    if echo:
        x[0] = 0.14
    t = r.Target(3, 14, 1, 21, "You are in idiot.")
    with (
        patch.object(r.audio, "_wave", return_value=x),
        patch.object(r.audio, "_recognize", return_value=[spoken("You are an idiot.")]),
        patch.object(r, "_second", return_value=[spoken(other)]),
    ):
        assert bool(r.listen(b"audio", t, threading.Event())) is accepted


def test_nearby_or_truncated_disclaimer_does_not_count():
    import numpy as np

    x = np.zeros((2, 16 * 16000), dtype=np.float32)
    x[1] = 0.15
    a = seg(4, "You are an idiot.")
    t = r.Target(3, 14, 1, 20, "You are in idiot.")
    assert not r._vote([a, seg(6, "That was someone reading.")], x, 10, t)
    assert not r._vote([seg(2, "He says:"), a], x, 10, t)


def phrase_part(at, text):
    out = spoken(text)
    for w in out.words:
        w.start += at - 4
        w.end += at - 4
    out.start = out.words[0].start
    out.end = out.words[-1].end
    return out


@pytest.mark.parametrize("head,tail", [("You are", "an idiot."), ("Usted es", "un idiota.")])
def test_two_models_can_segment_same_directed_phrase_differently(head, tail):
    import numpy as np

    x = np.zeros((2, 16 * 16000), dtype=np.float32)
    x[1] = 0.008
    t = r.Target(3, 14, 1, 21, head + " in " + tail)
    with (
        patch.object(r.audio, "_wave", return_value=x),
        patch.object(
            r.audio, "_recognize", return_value=[phrase_part(4, head), phrase_part(4.4, tail)]
        ) as first,
        patch.object(r, "_second", return_value=[phrase_part(4, head + " " + tail)]) as second,
    ):
        assert r.listen(b"audio", t, threading.Event()) == head + " " + tail
    assert first.call_count == second.call_count == 1
    assert first.call_args.args[0].max() > x[1].max()
    assert float(x[1].max()) == pytest.approx(0.008)


@pytest.mark.parametrize(
    "failure",
    [
        "pause",
        "punctuation",
        "echo",
        "interjection",
        "low_confidence",
        "attribution",
        "denial",
        "truncated",
        "repeat",
    ],
)
def test_no_splicing_or_removing_contradictory_evidence(failure):
    import numpy as np

    x = np.zeros((2, 16 * 16000), dtype=np.float32)
    x[1] = 0.09
    parts = [
        phrase_part(4, "You are." if failure == "punctuation" else "You are"),
        phrase_part(5.5 if failure == "pause" else 4.6, "an idiot."),
    ]
    if failure == "echo":
        x[0, 4 * 16000 : 5 * 16000] = 0.09
    if failure == "interjection":
        x[0, int(4.38 * 16000) : int(4.56 * 16000)] = 0.09
    if failure == "low_confidence":
        parts[1].words[0].probability = 0.4
    if failure == "attribution":
        parts.insert(0, phrase_part(0.4, "He said:"))
    if failure == "denial":
        parts.insert(1, phrase_part(4.37, "not"))
    if failure == "repeat":
        parts.append(phrase_part(6, "You are an idiot."))
    t = r.Target(3, 14, 1, 14.7 if failure == "truncated" else 21, "You are in idiot.")
    assert not r._vote(parts, x, 10, t)


def test_gain_does_not_amplify_noise_or_shared_audio_or_clip_loud_samples():
    import numpy as np

    t = r.Target(0, 4, 0, 9, "")
    for level, other in [(0.0, 0.0), (0.001, 0.001), (0.1, 0.0)]:
        x = np.zeros((2, 16 * 16000), dtype=np.float32)
        x[0] = level
        x[1] = other
        assert np.array_equal(x[0], r.recognition_track(x, 0, t))
    x[0] = 0.004
    x[0, 1] = 1
    assert np.array_equal(x[0], r.recognition_track(x, 0, t))


@pytest.mark.parametrize("audible_intro", [True, False])
async def test_recovery_binds_one_corrected_turn_to_exact_payload(audible_intro):
    raw = dialogue(abuse="You are in idiot.").replace("\n", "\r\n")
    e = event()
    with (
        patch.object(r.audio, "download_bytes", AsyncMock(return_value=b"audio")),
        patch.object(r, "listen", return_value="You are an idiot."),
        patch.object(r.audio, "_listen", return_value=audible_intro),
    ):
        result = await r.recover(raw, resolve(raw), e, ())
    if not audible_intro:
        assert result is None
    else:
        shown, proof = result
        assert shown == resolve(raw).replace("in idiot", "an idiot")
        assert (
            proof
            == "dual-asr-v1:3:" + hashlib.sha256((e.call_id + "\0" + shown).encode()).hexdigest()
        )


async def test_worker_returns_only_corroborated_repair():
    from types import SimpleNamespace as NS

    e = event()
    raw = dialogue(abuse="You are in idiot.")
    adapter = NS(transcribe=AsyncMock(return_value=raw), intake_hints=())
    with (
        patch.object(step, "DeepgramAdapter", return_value=adapter),
        patch.object(
            step,
            "get_settings",
            return_value=NS(
                stt=NS(
                    provider="deepgram",
                    deepgram_api_key="fixture",
                    phoneburner_channel_roles={},
                    phoneburner_semantic_roles=False,
                )
            ),
        ),
        patch.object(step, "verified_leads", AsyncMock(side_effect=lambda raw, r, uri: r)),
        patch.object(step, "verify", AsyncMock(return_value="needs_review")),
        patch.object(r.audio, "download_bytes", AsyncMock(return_value=b"audio")),
        patch.object(r, "listen", return_value="You are an idiot."),
        patch.object(r.audio, "_listen", return_value=True),
    ):
        assert await step.transcribe(e) == resolve(raw).replace("in idiot", "an idiot")
    assert e._conduct_audio.startswith("dual-asr-v1:3:")


@pytest.mark.parametrize(
    "txt,expected",
    [
        ("Fuck you.", True),
        ("Mire, usted es una hija de puta.", True),
        ("Eres un hijueputa.", True),
        ("Me dijo: eres un hijueputa.", False),
        ("La última perra y que puta.", False),
        ("What did he mean by fuck you?", False),
    ],
)
def test_unambiguous_address_only(txt, expected):
    from app.services.conduct_audio import directed

    assert directed(txt) is expected


def test_language_uses_conversation_not_expected_quote():
    assert r.language("usted tuvo un accidente gracias por llamar a nuestros abogados") == "es"
    assert r.language("I am calling from the firm about your accident") == "en"
    assert r.language("ok") is None


def test_resolved_unrelated_hint_does_not_hide_intaker_target():
    raw = dialogue(abuse="You are in idiot.")
    hints = (Hint(2, 4, "unclear_word"),)
    assert r.target(raw, resolve(raw), hints) is None
    assert r.target(raw, resolve(raw), hints, unresolved=False) is not None


def test_same_checkpoint_at_two_paths_does_not_count_twice(monkeypatch):
    from types import SimpleNamespace

    for name, key in [("a", "CONDUCT_ASR_PATH"), ("b", "CONDUCT_SECOND_ASR_PATH")]:
        monkeypatch.setenv(key, name)
    r._second_model.cache_clear()
    with (
        patch.object(r.Path, "stat", return_value=SimpleNamespace(st_size=123)),
        pytest.raises(ValueError, match="Independent acoustic checkpoint"),
    ):
        r._second_model()
