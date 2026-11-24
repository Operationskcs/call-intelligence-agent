"""Corroborate a narrow class of severe, directly addressed conduct on original audio.

Provider speaker indexes may change between passes. Require each pass to identify
the employee independently and place the same direct utterance on the same track.
"""
import hashlib
import re

from app.models.call_event import CallEvent
from app.adapters.stt.audio_hints import Hint, select
from app.services.phoneburner_roles import (
    _AGENT, _PROFESSIONAL_INTRO, _TURN, _normalize, resolve_phoneburner_roles,
)

_CUE = re.compile(r"\b(?:puta|puto|hijueputa|perra|pendej[oa]|idiot[ae]?|imbecil|bitch|asshole|"
                  r"motherfucker|fuck(?:ing)?|te voy a (?:matar|golpear)|i will hurt you)\b")
_DIRECT = re.compile(r"(?:you(?: are|'re) (?:a |an )?(?:fucking )?(?:idiot|bitch|asshole)|"
                     r"shut (?:the fuck up|your fucking mouth)|(?:(?:tu )?eres|usted es) (?:un |una )?"
                     r"(?:(?:put[oa]|maldit[oa]) )?(?:idiota|imbecil|perra|pendej[oa])|"
                     r"callate la puta boca|fuck you|"
                     r"(?:(?:tu )?eres|usted es) (?:un |una )?(?:hijueputa|hij[oa] de puta))")
_CONTEXT = re.compile(r"\b(?:said|says|say|told|quote[ds]?|called me|repeat|for example|"
                      r"what i heard|hypothetically|what if|pretend|suppose|would never|"
                      r"dij[oe]|dijeron|decia|me llamo|citando|por ejemplo|repite|diga|dime|"
                      r"supongamos|hipoteticamente|imaginese|transfer\w*|"
                      r"transfier\w*|interprete|interpreter|handing you over)\b")


def suspect(text: str, *, include_lead: bool = False) -> bool:
    return any((include_lead or m["label"] != "Lead") and _CUE.search(_normalize(m["text"]))
               for line in text.splitlines() if (m := _TURN.fullmatch(line)))


def uncertain_windows(final: str, provider: tuple[Hint, ...]) -> tuple[Hint, ...]:
    hints = list(provider)
    for line in final.splitlines():
        m = _TURN.fullmatch(line)
        if not m or not m["prefix"].strip():
            continue
        txt = _normalize(m["text"])
        at = 0
        for n in m["prefix"].strip().split(":"):
            at = at * 60 + int(n)
        if m["label"] != "Agent" and _CUE.search(txt):
            hints.append(Hint(0, at, "non_agent_cue"))
        elif re.search(r"\b(?:inaudible|unintelligible|ininteligible)\b", txt):
            hints.append(Hint(1, at, "inaudible"))
        elif m["label"].startswith("Unknown") and len(txt.split()) >= 3:
            hints.append(Hint(3, at, "unknown_voice"))
    return select(hints)


# Allow common conversational framing, NOT arbitrary words before/after a slur.
# The proof sent downstream covers the WHOLE original turn, including this framing.
_SEP = r"[\s,.!¡;:—–-]+"
_INTRO = r"(?:look|listen(?: to me)?|sir|ma'am|honestly|mire|mira|oiga|escuche|señor[a]?|senor[a]?)"
_END = r"(?:let me finish|listen to me|please listen|dejeme terminar|dejame terminar|escucheme)"
_ADDRESSED = re.compile(r"(?:(?:" + _INTRO + r")" + _SEP + r"){0,2}(?:"
                        + _DIRECT.pattern + r")(?:" + _SEP + _END + r")?")
_NARRATING = re.compile(
    r"\b(?:read(?:ing)? (?:aloud|out loud|this|the message)|the words|his words|her words|"
    r"shout(?:ed)?|yell(?:ed)?|wrote|written|message (?:was|reads)|called (?:me|you)|"
    r"example|imagine|if|hypothetical|means|translat\w*|never|would(?:n't| not)|"
    r"leer\w*|leyendo|grit\w*|escrib\w*|sus palabras|el mensaje|un ejemplo|"
    r"imagin\w*|supong\w*|significa|traduc\w*|diria|nunca|jamas|me llamaron|me insult\w*)\b"
)


def directed(text: str) -> bool:
    clean = _normalize(text)
    if (len(text) > 180 or any(c in clean for c in '?¿"“”«»()')
            or _CONTEXT.search(clean) or _NARRATING.search(clean)):
        return False
    # A direct imperative, but also common as reported speech. Only a complete
    # standalone turn may reach the existing speaker and acoustic checks.
    if re.fullmatch(
        r"go fuck yourself|(?:callate|callese)[ ,]+(?:hijueputa|idiota|imbecil|pendej[oa]|perra)",
        clean.strip(" .!,¡"),
    ):
        return True
    if _ADDRESSED.fullmatch(clean.strip(" .!,¡")):
        return True
    # Allow other complete statements in the turn without treating arbitrary
    # words inside a clause ("I would never call you ...") as direct address.
    clauses = [s.strip(" ,¡") for s in re.split(r"[.!;]+", clean) if s.strip(" ,¡")]
    risky = [s for s in clauses if _CUE.search(s)]
    return len(risky) == 1 and bool(_ADDRESSED.fullmatch(risky[0]))


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
            if (m["label"] != agent or not directed(m["text"])
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
    if len(hits) != 1:
        return None
    following = raw.splitlines()[hits[0][0]+1:hits[0][0]+2]
    if following and (later := _TURN.fullmatch(following[0])) and _CONTEXT.search(
        _normalize(later["text"])
    ):
        return None
    return hits[0]


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
