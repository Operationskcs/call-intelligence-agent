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
from app.adapters.stt.audio_hints import Hint
from app.services.conduct_audio import _CONTEXT, _CUE, bind, probe_plan
from app.services.timed_conduct import Proposal
from app.services.phoneburner_roles import (
    RoleResolution, _LEAD, _normalize, _NONPERSONAL, _OWN_ANSWER, _PERSONAL_EVENT, _REPORTED,
    _TURN, _VICTIM_ACCOUNT,
)
from app.storage.gcs import download_bytes as download_bytes

_SLOT = threading.BoundedSemaphore(1)
_SCAN_TIMEOUT_SECONDS = 60.0


@lru_cache(maxsize=1)
def _model() -> Any:
    from faster_whisper import WhisperModel
    return WhisperModel(os.getenv("CONDUCT_ASR_PATH", "/opt/conduct-asr"), device="cpu",
                        compute_type="int8", cpu_threads=1, local_files_only=True)


def words(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", _normalize(text).replace("’", "'")))


def _has_isolated_speech(samples: Any, channel: int) -> bool:
    """Silence and mirrored tracks cannot corroborate a speaker; skip ASR entirely."""
    import numpy as np
    # 100ms frames. A single noisy sample or one active click is not speech.
    n = samples.shape[1] // 1600
    if n < 2:
        return False
    powers = np.square(samples[:, :n*1600].reshape(2, n, 1600)).mean(axis=2)
    valid = (powers[channel] > 1e-7) & (powers[channel] > 12 * powers[1-channel])
    return bool(np.any(valid[1:] & valid[:-1]))


def _wave(data: bytes, start: int, stop: int, *, allow_mono: bool = False) -> Any:
    import av
    import numpy as np
    # Decode a bounded window only. No conversion of mono or multivoice
    # recordings into fabricated stereo identities.
    with av.open(io.BytesIO(data), mode="r") as c:
        st = c.streams.audio[0]
        channels = st.codec_context.channels
        if channels != 2 and not (allow_mono and channels == 1):
            return None
        c.seek(int(start * av.time_base), backward=True)
        resampler = av.AudioResampler(format="fltp", layout="stereo" if channels == 2 else "mono",
                                      rate=16000)
        samples = np.zeros((channels, (stop-start)*16000), dtype=np.float32)
        written = 0
        for frame in c.decode(st):
            if frame.time is None:
                return None
            if frame.time >= stop:
                break
            for f in resampler.resample(frame):
                if f.time is None:
                    return None
                arr = f.to_ndarray()
                lo = max(0, round((start - f.time) * 16000))
                hi = min(arr.shape[1], round((stop - f.time) * 16000))
                if hi > lo:
                    pos = max(0, round((f.time-start)*16000) + lo)
                    n = min(hi-lo, samples.shape[1]-pos)
                    if n > 0:
                        samples[:, pos:pos+n] = arr[:, lo:lo+n]
                        written = max(written, pos+n)
        if not written:
            return None
        return samples[:, :written]


def _recognize(samples: Any, language: str | None = None) -> Any:
    import numpy as np
    segs, _ = _model().transcribe(np.ascontiguousarray(samples), beam_size=4,
                                  task="transcribe", language=language, temperature=0.0, max_new_tokens=64,
                                  condition_on_previous_text=False, word_timestamps=True)
    return list(segs)


def _scan(data: bytes, hint: Hint, stop: threading.Event | None = None
          ) -> set[tuple[int, int, str, str]] | None:
    """Discovery only. Listen to both tracks (or a single unassigned mono track).

    Context stays in memory; never save the alternative text or use it as a
    coaching quote. Unlike proof, discovery must not discard overlapping voices.
    """
    import numpy as np
    start = max(0, hint.at - 4)
    samples = _wave(data, start, hint.at + 9, allow_mono=True)
    if samples is None:
        return None
    hits = set()
    for ch, track in enumerate(samples):
        if stop is not None and stop.is_set():
            return None
        # Adjacent context can contain an earlier caller's answer. Decode that
        # context only if THIS track also has signal in the interval at issue.
        core = track[max(0,hint.at-2-start)*16000:(hint.at+7-start)*16000]
        frames = len(core)//1600
        if not frames:
            continue
        energy = np.square(core[:frames*1600].reshape(frames,1600)).mean(axis=1)
        if np.count_nonzero(energy > 1e-7) < 2:
            continue
        for seg in _recognize(track):
            if seg.avg_logprob < -.8 or seg.no_speech_prob > .55 or not seg.words:
                continue
            # Only cues near the selected instant count; other words supplied
            # context to ASR. Never concatenate across an uncertain/missing word.
            nearby = [w for w in seg.words if hint.at-2 <= start+w.start <= hint.at+6]
            for idx, w in enumerate(nearby):
                if w.probability < .60:
                    continue
                suffix = []
                for z in nearby[idx:idx+6]:
                    if z.probability < .60:
                        break
                    suffix.append(words(z.word))
                match = _CUE.match(" ".join(suffix))
                if match:
                    hits.add((ch if len(samples) == 2 else -1, int(start+w.start), match[0],
                              words(seg.text)))
    return hits


def _consistent(raw: str, final: str, ch: int, at: int, cue: str, passage: str) -> str | None:
    """A changed/ambiguous voice or novel wording is not proof; keep it for review."""
    turns: list[tuple[int, str, str]] = []
    for line, labeled in zip(raw.splitlines(), final.splitlines(), strict=True):
        m, label = _TURN.fullmatch(line), _TURN.fullmatch(labeled)
        if not m or not label or not m["prefix"].strip():
            continue
        track = re.fullmatch(r"Channel (\d+)(?: Speaker \d+)?", m["label"])
        if track and int(track[1]) == ch:
            t = 0
            for n in m["prefix"].strip().split(":"):
                t = t * 60 + int(n)
            turns.append((t, label["label"], words(m["text"])))
    nearby = [(role, text) for i,(t,role,text) in enumerate(turns)
              if t-1 <= at <= t+10 and (i+1 == len(turns) or at < turns[i+1][0])]
    if len(nearby) != 1 or nearby[0][0] not in {"Agent", "Lead"}:
        return None
    role, original = nearby[0]
    if (" "+words(cue)+" " not in " "+original+" " or passage != original):
        return None
    return role


def _lead_anchor(raw: str, final: str, channel: int) -> tuple[int, str] | None:
    anchors = []
    labels: set[str] = set()
    for line, rendered in zip(raw.splitlines(), final.splitlines(), strict=True):
        m, f = _TURN.fullmatch(line), _TURN.fullmatch(rendered)
        if not m or not f or not m["label"].startswith(f"Channel {channel} "):
            continue
        labels.add(m["label"])
        if f["label"] != "Lead":
            return None
        text = _normalize(m["text"])
        if (not m["prefix"].strip() or _REPORTED.search(text) or _NONPERSONAL.search(text)
                or re.search(r"\b(?:no|not|never|nunca)\b", text)
                or any(c in text for c in '?¿"“”«»')):
            continue
        match = _PERSONAL_EVENT.search(text) or _OWN_ANSWER.search(text) or _LEAD.search(text)
        if match and len(words(match[0]).split()) >= 3:
            at = 0
            for part in m["prefix"].strip().split(":"):
                at = at*60 + int(part)
            anchors.append((at, match[0]))
    return anchors[0] if len(labels) == 1 and anchors else None


async def inspect_uncertainty(raw: str, final: str, uri: str, hints: tuple[Hint, ...]) -> str:
    if not _SLOT.acquire(blocking=False):
        return "unavailable"
    # A timed-out to_thread is still running. Keep its slot until it really
    # returns; do not launch further windows/voices after the request deadline.
    stop = threading.Event()
    task = asyncio.create_task(_inspect_selected(raw, final, uri, hints, stop))
    def finished(done: asyncio.Task[str]) -> None:
        _SLOT.release()
        if not done.cancelled():
            done.exception()
    task.add_done_callback(finished)
    try:
        return await asyncio.wait_for(asyncio.shield(task), timeout=_SCAN_TIMEOUT_SECONDS)
    except TimeoutError:
        return "unavailable"
    finally:
        stop.set()


async def _inspect_selected(raw: str, final: str, uri: str, hints: tuple[Hint, ...],
                            stop: threading.Event) -> str:
    data = await download_bytes(uri, max_bytes=32_000_000)
    unavailable = False
    confirmed_lead: set[int] = set()
    for hint in hints[:2]:
        if stop.is_set():
            return "unavailable"
        hits = await asyncio.to_thread(_scan, data, hint, stop)
        if hits is None:
            unavailable = True
        elif hint.reason == "non_agent_cue" and not hits:
            return "needs_review"
        if hits:
            for ch, at, cue, passage in hits:
                role = _consistent(raw, final, ch, at, cue, passage)
                if role is None:
                    return "needs_review"
                if role == "Lead" and ch not in confirmed_lead:
                    if stop.is_set():
                        return "unavailable"
                    anchor = _lead_anchor(raw, final, ch)
                    if not anchor or not await asyncio.to_thread(
                        _listen, data, ch, anchor[0], anchor[1], False
                    ):
                        return "needs_review"
                    confirmed_lead.add(ch)
    return "unavailable" if unavailable else "consistent"


def _literal_bounds(segs: list[Any], target: list[str]) -> list[tuple[float, float]] | None:
    """Match the entire attested turn across contiguous ASR segments, never a loose substring."""
    for i, seg in enumerate(segs):
        if i and seg.start - segs[i-1].end < 5 and _CONTEXT.search(_normalize(segs[i-1].text)):
            continue
        heard: list[str] = []
        bounds: list[tuple[float, float]] = []
        for j in range(i, len(segs)):
            part = segs[j]
            if (part.avg_logprob < -.8 or part.no_speech_prob > .55 or not part.words
                    or any(c in part.text for c in '?¿"“”«»')
                    or (j > i and not 0 <= part.start - segs[j-1].end <= 1)):
                break
            if any(w.probability < .60 for w in part.words):
                break
            for w in part.words:
                heard.extend(words(w.word).split())
                bounds.append((w.start, w.end))
            if heard != target[:len(heard)]:
                break
            if heard == target:
                if (j+1 < len(segs) and segs[j+1].start - part.end < 4
                        and _CONTEXT.search(_normalize(segs[j+1].text))):
                    break
                if (any(end <= begin for begin,end in bounds) or any(
                    not -.05 <= second[0]-one[1] <= 1 for one,second in zip(bounds,bounds[1:])
                )):
                    break
                return bounds
    return None


def _listen(data: bytes, channel: int, at: int, phrase: str, exact: bool) -> bool:
    import numpy as np
    start = max(0, at - (4 if exact else 1))
    samples = _wave(data, start, at + 12)
    if samples is None or channel not in (0, 1):
        return False
    if not _has_isolated_speech(samples, channel):
        return False
    # Twelve seconds of speech do not warrant repeated temperature fallback
    # decoding on a Cloud Run CPU. Never relax the matching/evidence thresholds.
    segs = _recognize(samples[channel])
    target = words(phrase).split()
    if exact:
        bounds = _literal_bounds(segs, target)
        if not bounds or abs(start + bounds[0][0] - at) > 2:
            return False
        # A long, clean preamble must not mask cross-talk on the insult itself.
        for begin, end in bounds:
            clip = samples[:, int(begin*16000):int(end*16000)]
            if clip.shape[1] < 160:
                return False
            power = np.square(clip.astype(np.float64)).mean(axis=1)
            if not (power[channel] > 1e-7 and power[channel] > 12 * power[1-channel]):
                return False
        return True
    for seg in segs:
        if seg.avg_logprob < -.8 or seg.no_speech_prob > .55:
            continue
        wordlist = [(w, words(w.word)) for w in seg.words or []]
        for idx in range(len(wordlist) - len(target) + 1):
            group = wordlist[idx:idx+len(target)]
            if [v for _, v in group] != target or any(w.probability < .60 for w, _ in group):
                continue
            begin, end = group[0][0].start, group[-1][0].end
            if abs(start + begin - at) > 5:
                continue
            clip = samples[:, int(begin*16000):int(end*16000)]
            if clip.shape[1] < 1600:
                continue
            power = np.square(clip.astype(np.float64)).mean(axis=1)
            if power[channel] > 1e-7 and power[channel] > 12 * power[1-channel]:
                return True
    return False


def _context_listen(data: bytes, ch: int, p: Proposal) -> bool:
    """Independently hear the entire parent (not a cutout or a primed transcript).

    Punctuation can put a short direct clause inside one long audio segment.
    Keep word-by-word acoustic proof and cross-talk veto on the clause itself.
    """
    import numpy as np
    import math
    start = max(0, p.pieces[0][0] - 2)
    end = math.ceil(p.end) + 2
    # One native Whisper context. Never trim the parent to make a fragment match.
    if end - start > 30:
        return False
    x = _wave(data, start, end)
    if x is None or x.shape[0] != 2 or not _has_isolated_speech(x,ch):
        return False
    other = _recognize(x[1-ch])
    if any(_CUE.search(_normalize(seg.text)) for seg in other):
        return False
    segments = _recognize(x[ch])
    parent = words(p.original).split()
    bounds = _literal_bounds(segments,parent)
    if not bounds or len(bounds) != len(parent) or abs(start+bounds[0][0]-p.pieces[0][0])>2:
        return False
    low = sum(len(words(text).split()) for _,text in p.pieces[:p.clause])
    selected = bounds[low:low+len(words(p.pieces[p.clause][1]).split())]
    if not selected or abs(start+selected[0][0]-p.hint.at)>2:
        return False
    for begin,finish in selected:
        clip=x[:,int(begin*16000):int(finish*16000)]
        if clip.shape[1]<160:
            return False
        power=np.square(clip.astype(np.float64)).mean(axis=1)
        if not (power[ch]>1e-7 and power[ch]>12*power[1-ch]):
            return False
    return True


async def verify(primary: str, final: str, event: CallEvent, *, parent: Proposal | None = None) -> str:
    plan = probe_plan(primary, final, event)
    if not plan or not event.gcs_audio_uri or not _SLOT.acquire(blocking=False):
        return "needs_review"
    try:
        data = await download_bytes(event.gcs_audio_uri, max_bytes=32_000_000)
        if len(data) > 32_000_000:
            return "needs_review"
        ch, intro_at, intro, at, phrase = plan
        if not await asyncio.to_thread(_listen, data, ch, intro_at, intro, False):
            return "needs_review"
        if parent is not None:
            if at != parent.hint.at or words(phrase) != words(parent.pieces[parent.clause][1]):
                return "needs_review"
            heard = await asyncio.to_thread(_context_listen,data,ch,parent)
        else:
            heard = await asyncio.to_thread(_listen,data,ch,at,phrase,True)
        if heard:
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
