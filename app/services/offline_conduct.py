"""Independent, unprompted recognition inside the worker; no external media upload."""
from __future__ import annotations

import asyncio
from functools import lru_cache
import io
import os
import re
import threading
from typing import Any

from app.models.call_event import CallEvent
from app.services.conduct_audio import bind, probe_plan
from app.services.phoneburner_roles import (
    RoleResolution, _normalize, _NONPERSONAL, _OWN_ANSWER, _REPORTED, _TURN, _VICTIM_ACCOUNT,
)
from app.storage.gcs import download_bytes

_SLOT = threading.BoundedSemaphore(1)


@lru_cache(maxsize=1)
def _model() -> Any:
    from faster_whisper import WhisperModel
    return WhisperModel(os.getenv("CONDUCT_ASR_PATH", "/opt/conduct-asr"), device="cpu",
                        compute_type="int8", cpu_threads=1, local_files_only=True)


def words(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", _normalize(text).replace("’", "'")))


def _listen(data: bytes, channel: int, at: int, phrase: str, exact: bool) -> bool:
    import av
    import numpy as np
    # Decode a bounded window only. No conversion of mono or multivoice
    # recordings into fabricated stereo identities.
    with av.open(io.BytesIO(data), mode="r") as c:
        st = c.streams.audio[0]
        if st.codec_context.channels != 2 or channel not in (0, 1):
            return False
        start = max(0, at - 1)
        c.seek(int(start * av.time_base), backward=True)
        resampler = av.AudioResampler(format="fltp", layout="stereo", rate=16000)
        chunks = []
        for frame in c.decode(st):
            if frame.time is None:
                return False
            if frame.time >= at + 12:
                break
            for f in resampler.resample(frame):
                if f.time is None:
                    return False
                arr = f.to_ndarray()
                lo = max(0, round((start - f.time) * 16000))
                hi = min(arr.shape[1], round((at + 12 - f.time) * 16000))
                if hi > lo:
                    chunks.append(arr[:, lo:hi])
        if not chunks:
            return False
        samples = np.concatenate(chunks, axis=1)
    segs, _ = _model().transcribe(np.ascontiguousarray(samples[channel]), beam_size=4,
                                  condition_on_previous_text=False, word_timestamps=True)
    target = words(phrase).split()
    for seg in segs:
        if seg.avg_logprob < -.8 or seg.no_speech_prob > .55:
            continue
        wordlist = [(w, words(w.word)) for w in seg.words or []]
        for idx in range(len(wordlist) - len(target) + 1):
            group = wordlist[idx:idx+len(target)]
            if [v for _, v in group] != target or any(w.probability < .60 for w, _ in group):
                continue
            if exact and (idx != 0 or len(group) != len(wordlist)):
                continue
            begin, end = group[0][0].start, group[-1][0].end
            if abs(start + begin - at) > (2 if exact else 5):
                continue
            clip = samples[:, int(begin*16000):int(end*16000)]
            if clip.shape[1] < 1600:
                continue
            power = np.square(clip.astype(np.float64)).mean(axis=1)
            if power[channel] > 1e-7 and power[channel] > 12 * power[1-channel]:
                return True
    return False


async def verify(primary: str, final: str, event: CallEvent) -> str:
    plan = probe_plan(primary, final, event)
    if not plan or not event.gcs_audio_uri or not _SLOT.acquire(blocking=False):
        return "needs_review"
    try:
        data = await download_bytes(event.gcs_audio_uri, max_bytes=32_000_000)
        if len(data) > 32_000_000:
            return "needs_review"
        ch, intro_at, intro, at, phrase = plan
        if (await asyncio.to_thread(_listen, data, ch, intro_at, intro, False)
                and await asyncio.to_thread(_listen, data, ch, at, phrase, True)):
            return bind(primary, final, event)
        return "needs_review"
    finally:
        _SLOT.release()


async def verified_leads(primary: str, result: RoleResolution, uri: str) -> RoleResolution:
    """New Lead rules cannot promote an identity without an audible first-person anchor."""
    novel = {i for i, why in result.reasons.items() if why in {
        "first_person_intake_response", "first_person_case_account"}}
    if not novel:
        return result
    confirmed: set[str] = set()
    if _SLOT.acquire(blocking=False):
        try:
            data = await download_bytes(uri, max_bytes=32_000_000)
            if len(data) <= 32_000_000:
                for identity in sorted(novel)[:2]:
                    ch = re.fullmatch(r"Channel ([01])(?: Speaker \d+)?", identity)
                    if not ch:
                        continue
                    attempts = 0
                    for line in primary.splitlines():
                        m = _TURN.fullmatch(line)
                        if not m or m["label"] != identity or not m["prefix"].strip():
                            continue
                        tx = _normalize(m["text"])
                        cue = _VICTIM_ACCOUNT.search(tx) or _OWN_ANSWER.search(tx)
                        if (not cue or _NONPERSONAL.search(tx) or _REPORTED.search(tx)
                                or re.search(r"\b(?:no|not|never|nunca)\b", tx)
                                or any(c in tx for c in '?¿"“”«»')):
                            continue
                        at = 0
                        for n in m["prefix"].strip().split(":"):
                            at = at * 60 + int(n)
                        if await asyncio.to_thread(_listen, data, int(ch[1]), at, cue[0], False):
                            confirmed.add(identity)
                            break
                        attempts += 1
                        if attempts == 2:
                            break
        except Exception:
            # Missing/corrupt audio is not evidence. Preserve the primary words.
            pass
        finally:
            _SLOT.release()
    lost = novel - confirmed
    roles, reasons = dict(result.roles), dict(result.reasons)
    for identity in novel:
        if identity in lost:
            roles[identity], reasons[identity] = "Unknown", "first_person_audio_inconclusive"
        else:
            reasons[identity] = "first_person_audio_confirmed"
    lines = []
    for raw, rendered in zip(primary.splitlines(keepends=True),
                             result.transcript.splitlines(keepends=True), strict=True):
        m = _TURN.match(raw)
        if m and m["label"] in lost:
            lo, hi = m.span("label")
            rendered = raw[:lo] + f"Unknown ({m['label']})" + raw[hi:]
        lines.append(rendered)
    return RoleResolution("".join(lines), roles, reasons)
