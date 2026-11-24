"""Recover one misheard Intake turn, never from a single ASR vote or a role guess."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
import os
from pathlib import Path
import re
import threading
from typing import Any

from app.adapters.stt.audio_hints import Hint
from app.models.call_event import CallEvent
from app.services import offline_conduct as audio
from app.services.conduct_audio import _CONTEXT, _CUE, _NARRATING, bind, directed, probe_plan
from app.services.phoneburner_roles import _TURN, _normalize, resolve_phoneburner_roles

_NEG = re.compile(r"\b(?:no|not|never|nunca|jamas|don't|dont|isn't|wasn't|wasnt)\b")
_THIRD = re.compile(
    r"\b(?:he|she|they|him|her|ella|ellos|ellas|el conductor|"
    r"el otro|that driver|the driver|someone|alguien)\b"
)
_GAP = re.compile(r"\[?(?:inaudible|unintelligible|ininteligible)\]?\.?")
_ADDRESS = re.compile(r"\b(?:you(?:'re| are)|usted es|tu eres|eres|callate|shut)\b")


def language(text: str) -> str | None:
    """Only a language hint (never words/prompts); short clips often mix up EN and ES."""
    terms = set(audio.words(text).split())
    es = len(
        terms
        & {
            "usted",
            "llamo",
            "accidente",
            "abogados",
            "tengo",
            "para",
            "porque",
            "gracias",
            "como",
            "buenas",
        }
    )
    en = len(
        terms
        & {
            "your",
            "the",
            "accident",
            "calling",
            "attorney",
            "thanks",
            "from",
            "with",
            "what",
            "happened",
        }
    )
    if es >= 4 and en <= 1:
        return "es"
    if en >= 4 and es <= 1:
        return "en"
    return None


def unsafe(text: str) -> bool:
    t = _normalize(text)
    return bool(
        _CONTEXT.search(t)
        or _NARRATING.search(t)
        or _NEG.search(t)
        or any(c in t for c in '?¿"“”«»()')
    )


@dataclass(frozen=True)
class Target:
    index: int
    at: int
    channel: int
    until: int
    original: str
    lang: str | None = None


def target(
    raw: str, final: str, hints: tuple[Hint, ...], *, unresolved: bool = True
) -> Target | None:
    rows = [_TURN.fullmatch(line) for line in raw.splitlines()]
    labels = [_TURN.fullmatch(line) for line in final.splitlines()]
    if not rows or len(rows) != len(labels):
        return None
    times = []
    matches = []
    identities: set[tuple[str, str, str]] = set()
    for idx, (m, role) in enumerate(zip(rows, labels, strict=True)):
        if (
            not m
            or not role
            or not m["prefix"].strip()
            or m["prefix"] != role["prefix"]
            or (m["text"] != role["text"] or role["label"] not in {"Agent", "Lead"})
        ):
            return None
        at = 0
        for n in m["prefix"].split(":"):
            at = 60 * at + int(n)
        times.append(at)
        channel = re.fullmatch(r"Channel ([01]) Speaker \d+", m["label"])
        if not channel:
            return None
        identities.add((channel[1], m["label"], role["label"]))
        txt = m["text"]
        gap = _GAP.fullmatch(_normalize(txt))
        cue = _CUE.search(_normalize(txt))
        if (cue or gap) and (role["label"] != "Agent" or directed(txt) or unsafe(txt)):
            return None
        if (
            role["label"] == "Agent"
            and not unsafe(txt)
            and not directed(txt)
            and (
                cue
                or gap
                or (
                    _ADDRESS.search(_normalize(txt))
                    and any(h.reason == "unclear_word" and abs(h.at - at) <= 2 for h in hints)
                )
            )
        ):
            matches.append((idx, at, int(channel[1]), txt))
    if (
        len(matches) != 1
        or times != sorted(times)
        or len(identities) != 2
        or {x[0] for x in identities} != {"0", "1"}
        or {x[2] for x in identities} != {"Agent", "Lead"}
    ):
        return None
    idx, at, ch, txt = matches[0]
    if times.count(at) != 1 or not 0 < len(txt) <= 180:
        return None
    if _THIRD.search(_normalize(txt)) or any(
        m and unsafe(m["text"]) for j, m in enumerate(rows) if j != idx and abs(times[j] - at) <= 5
    ):
        return None
    # Unrelated uncertainty must not be overridden by recovering one good turn.
    if unresolved and any(
        abs(h.at - at) > 2 or h.reason in {"non_agent_cue", "unknown_voice"} for h in hints
    ):
        return None
    until = min(at + 9, times[idx + 1] if idx + 1 < len(times) else at + 9)
    return Target(idx, at, ch, until, txt, language(raw)) if until > at else None


@lru_cache(maxsize=1)
def _second_model() -> Any:
    path = os.getenv("CONDUCT_SECOND_ASR_PATH", "/opt/conduct-asr-base")
    a = Path(path) / "model.bin"
    b = Path(os.getenv("CONDUCT_ASR_PATH", "/opt/conduct-asr")) / "model.bin"
    # Pinned small/base files have different sizes. Reject even a copied checkpoint
    # at a different path, rather than treating two decodes as independent votes.
    if a.stat().st_size == b.stat().st_size:
        raise ValueError("Independent acoustic checkpoint required")
    from faster_whisper import WhisperModel

    return WhisperModel(
        path, device="cpu", compute_type="int8", cpu_threads=1, local_files_only=True
    )


def _second(track: Any, lang: str | None = None) -> list[Any]:
    import numpy as np

    segs, _ = _second_model().transcribe(
        np.ascontiguousarray(track),
        task="transcribe",
        language=lang,
        beam_size=4,
        temperature=0.0,
        condition_on_previous_text=False,
        max_new_tokens=64,
        word_timestamps=True,
    )
    return list(segs)


def _vote(segs: list[Any], samples: Any, start: int, t: Target) -> tuple[str, float, float] | None:
    import numpy as np

    votes = []
    for seg in segs:
        text = seg.text.strip()
        if not text or not directed(text) or unsafe(text):
            continue
        bounds = audio._literal_bounds(segs, audio.words(text).split())
        if not bounds or abs(start + bounds[0][0] - t.at) > 1.5 or start + bounds[-1][1] > t.until:
            continue
        if any(
            unsafe(s.text)
            for s in segs
            if s.end >= bounds[0][0] - 3 and s.start <= bounds[-1][1] + 3
        ):
            return None
        # Do not crop a qualifying/contradicting clause out of the same timed turn.
        if any(
            start + w.start >= t.at - 0.15
            and start + w.end <= t.until
            and (w.end < bounds[0][0] - 0.05 or w.start > bounds[-1][1] + 0.05)
            for other in segs
            for w in (other.words or [])
        ):
            return None
        for begin, end in bounds:
            clip = samples[:, int(begin * 16000) : int(end * 16000)]
            if clip.shape[1] < 160:
                return None
            power = np.square(clip.astype(np.float64)).mean(axis=1)
            if not (power[t.channel] > 1e-7 and power[t.channel] > 12 * power[1 - t.channel]):
                return None
        votes.append((text, bounds[0][0], bounds[-1][1]))
    return votes[0] if len(votes) == 1 else None


def listen(data: bytes, t: Target, stop: threading.Event) -> str | None:
    start = max(0, t.at - 4)
    samples = audio._wave(data, start, t.until + 3)
    if samples is None or not audio._has_isolated_speech(samples, t.channel) or stop.is_set():
        return None
    a = _vote(audio._recognize(samples[t.channel], t.lang), samples, start, t)
    if not a or stop.is_set():
        return None
    b = _vote(_second(samples[t.channel], t.lang), samples, start, t)
    if (
        not b
        or audio.words(a[0]) != audio.words(b[0])
        or abs(a[1] - b[1]) > 0.65
        or abs(a[2] - b[2]) > 0.65
    ):
        return None
    if (
        not _GAP.fullmatch(_normalize(t.original))
        and SequenceMatcher(
            None, audio.words(t.original).split(), audio.words(a[0]).split(), autojunk=False
        ).ratio()
        < 0.6
    ):
        return None
    return a[0]


async def recover(
    raw: str, final: str, event: CallEvent, hints: tuple[Hint, ...], *, unresolved: bool = True
) -> tuple[str, str] | None:
    t = target(raw, final, hints, unresolved=unresolved)
    if not t or not event.gcs_audio_uri or not audio._SLOT.acquire(blocking=False):
        return None
    stop = threading.Event()

    async def check() -> tuple[str, str] | None:
        data = await audio.download_bytes(event.gcs_audio_uri or "", max_bytes=32_000_000)
        if len(data) > 32_000_000 or stop.is_set():
            return None
        text = await asyncio.to_thread(listen, data, t, stop)
        if not text or stop.is_set():
            return None
        lines = raw.splitlines(keepends=True)
        old = lines[t.index]
        m = _TURN.fullmatch(old.rstrip("\r\n"))
        if not m:
            return None
        lines[t.index] = old[: m.start("text")] + text + old[m.end("text") :]
        fixed = "".join(lines)
        shown = resolve_phoneburner_roles(fixed, event).transcript
        original = final.splitlines(keepends=True)
        old = original[t.index]
        label = _TURN.fullmatch(old.rstrip("\r\n"))
        if not label:
            return None
        original[t.index] = old[: label.start("text")] + text + old[label.end("text") :]
        plan = probe_plan(fixed, shown, event)
        if (
            not plan
            or shown != "".join(original)
            or plan[0] != t.channel
            or plan[3] != t.at
            or not await asyncio.to_thread(audio._listen, data, plan[0], plan[1], plan[2], False)
        ):
            return None
        proof = bind(fixed, shown, event)
        return (shown, proof) if proof.startswith(f"dual-asr-v1:{t.index}:") else None

    task = asyncio.create_task(check())

    def release(done: asyncio.Task[tuple[str, str] | None]) -> None:
        audio._SLOT.release()
        if not done.cancelled():
            done.exception()

    task.add_done_callback(release)
    try:
        return await asyncio.wait_for(asyncio.shield(task), timeout=audio._SCAN_TIMEOUT_SECONDS)
    except TimeoutError:
        return None
    finally:
        stop.set()
