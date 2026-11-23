"""Corroborate a narrow class of severe, directly addressed conduct on original audio.

Provider speaker indexes may change between passes. Require each pass to identify
the employee independently and place the same direct utterance on the same track.
"""
import hashlib
import re

from app.models.call_event import CallEvent
from app.services.phoneburner_roles import (
    _AGENT, _PROFESSIONAL_INTRO, _TURN, _normalize, resolve_phoneburner_roles,
)

_CUE = re.compile(r"\b(?:puta|puto|perra|pendej[oa]|idiot[ae]?|imbecil|bitch|asshole|"
                  r"motherfucker|fuck(?:ing)?|te voy a (?:matar|golpear)|i will hurt you)\b")
_DIRECT = re.compile(r"(?:you(?: are|'re) (?:a |an )?(?:fucking )?(?:idiot|bitch|asshole)|"
                     r"shut (?:the fuck up|your fucking mouth)|(?:tu )?eres (?:un |una )?"
                     r"(?:(?:put[oa]|maldit[oa]) )?(?:idiota|imbecil|perra|pendej[oa])|"
                     r"callate la puta boca)")
_CONTEXT = re.compile(r"\b(?:said|says|say|told|quote|called me|repeat|for example|"
                      r"dij[oe]|decia|me llamo|citando|por ejemplo|repite|diga|dime|transfer\w*|"
                      r"transfier\w*|interprete|interpreter|handing you over)\b")


def suspect(text: str, *, include_lead: bool = False) -> bool:
    return any((include_lead or m["label"] != "Lead") and _CUE.search(_normalize(m["text"]))
               for line in text.splitlines() if (m := _TURN.fullmatch(line)))


def _candidate(raw: str, event: CallEvent) -> tuple[int, str, int, str] | None:
    # Neither a configured channel nor a provider-supplied speaker mapping
    # counts as a second recognition of the speaker in this proof.
    result = resolve_phoneburner_roles(raw, event.model_copy(update={"raw_payload": {}}))
    if len(result.roles) != 2 or set(result.roles.values()) != {"Agent", "Lead"}:
        return None
    tracks: dict[str, str] = {}
    for identity, role in result.roles.items():
        track = re.fullmatch(r"Channel (\d+) Speaker \d+", identity)
        if not track or track[1] in tracks:
            return None
        tracks[track[1]] = role
    agent = next(i for i, r in result.roles.items() if r == "Agent")
    last_agent = previous = ""
    hits: list[tuple[int, str, int, str]] = []
    for index, line in enumerate(raw.splitlines()):
        m = _TURN.fullmatch(line)
        if not m or m["label"] not in result.roles or not m["prefix"].strip():
            return None
        words = _normalize(m["text"])
        if re.search(r"\b(?:transfer\w*|transfier\w*|interpreter|interprete)\b", words):
            return None
        if _CUE.search(words):
            # A competing/quoted use anywhere makes automatic attribution unsafe.
            statement = words.strip(" .!,¡")
            if (m["label"] != agent or not _DIRECT.fullmatch(statement)
                    or _CONTEXT.search(last_agent) or _CONTEXT.search(previous)
                    or any(c in last_agent + previous for c in '"“”«»')):
                return None
            seconds = 0
            for part in m["prefix"].strip().split(":"):
                seconds = seconds * 60 + int(part)
            hits.append((index, agent.split()[1], seconds, statement))
        if m["label"] == agent:
            last_agent = words
        previous = words
    return hits[0] if len(hits) == 1 else None


def probe_plan(primary: str, final: str, event: CallEvent) -> tuple[int, int, str, int, str] | None:
    a = _candidate(primary, event)
    if not a or final != resolve_phoneburner_roles(primary, event).transcript:
        return None
    for line in primary.splitlines()[:a[0]]:
        m = _TURN.fullmatch(line)
        if not m or not m["label"].startswith("Channel "+a[1]+" Speaker "):
            continue
        txt = _normalize(m["text"])
        anchor = _AGENT.search(txt) or _PROFESSIONAL_INTRO.search(txt)
        if anchor and not _CONTEXT.search(txt) and not any(c in txt for c in '?¿"“”«»'):
            intro_at = 0
            for n in m["prefix"].strip().split(":"):
                intro_at = intro_at * 60 + int(n)
            if intro_at <= 90:
                return int(a[1]), intro_at, anchor[0], a[2], a[3]
    return None


def bind(primary: str, final: str, event: CallEvent) -> str:
    a = _candidate(primary, event)
    if not a:
        return "needs_review"
    digest = hashlib.sha256(f"{event.call_id}\0{final}".encode()).hexdigest()
    return f"dual-asr-v1:{a[0]}:{digest}"
