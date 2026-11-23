"""Evidence-bound PhoneBurner roles, independent of channel or speaking order.

This resolver deliberately abstains rather than fabricating a complete mapping.
Acoustic identity is retained in Unknown labels for downstream investigation.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from app.models.call_event import CallEvent

logger = logging.getLogger(__name__)
Role = Literal["Agent", "Lead"]
VERSION = "phoneburner-evidence-v2"
POLICY_REVISION = "first-person-intake-dialogue-v3"
_TURN = re.compile(
    r"^(?P<prefix>(?:\d+:\d{2}(?::\d{2})?\s+)?)"
    r"\[(?P<label>[^\]]+)]:\s*(?P<text>.*)$"
)
_IDENTITY = re.compile(r"^(?:Speaker \d+|(?:Unseparated )?Channel \d+(?: Speaker \d+)?)$", re.I)
_CHANNEL = re.compile(r"^(?:Unseparated )?Channel (\d+)(?: Speaker \d+)?$", re.I)
# Explicit first-person professional introductions, not generic topic words.
_AGENT = re.compile(
    r"\b(?:i am calling from|i'm calling from)\s+(?:the\s+)?"
    r"(?:law firm|legal team|intake|accident heroes|ayuda latina)\b"
    r"|\b(?:te|le|lo|la) llamo\s+(?:de parte de\s+|de\s+|del\s+)"
    r"(?:la\s+|el\s+)?(?:firma|equipo legal|abogados|ayuda latina|accident heroes)\b"
    r"|\b(?:te|le) llama\s+\w+(?: \w+){0,2}\s+de parte de\s+"
    r"(?:la\s+|el\s+)?(?:firma|equipo legal|abogados)\b",
    re.IGNORECASE,
)
_LEAD = re.compile(
    r"\b(?:my accident|my injuries|my car was hit|i was (?:hit|injured|driving)|"
    r"i have (?:back|neck|shoulder) pain|"
    r"mi accidente|me chocaron|yo estaba manejando|me duele|"
    r"tengo dolor|"
    r"fui al hospital|estoy en terapia|voy a terapia)\b",
    re.IGNORECASE,
)
_REPORTED = re.compile(
    r"\b(?:you said|you told me|you mentioned|the caller said|the lead said|"
    r"(?:he|she|they) said|(?:he|she) told me|if|suppose|for example|"
    r"usted dijo|me dijo|me dijiste|dijiste|usted me comento|como me comentaste|"
    r"ella dijo|el dijo|por ejemplo|supongamos|en caso de que)\b",
    re.IGNORECASE,
)

# Business context must belong to a first-person introduction. "My lawyer",
# an organization name by itself or a question about a firm is not enough.
_FIRM = (
    r"(?:law (?:firm|office)|legal (?:team|department)|intake (?:team|department)|"
    r"american compensation|accident heroes|ayuda latina|"
    r"firma(?: de abogados)?|bufete|oficina de abogados|equipo legal)"
)
_PROFESSIONAL_INTRO = re.compile(
    r"\b(?:i(?: am|'m) (?:calling |reaching out )(?:to you )?(?:from|on behalf of)|"
    r"(?:this is|my name is) [\w'-]+(?: [\w'-]+){0,2} (?:calling from|with|from)|"
    r"(?:te |le |lo |la )?(?:estoy llamando|llamo|me comunico) (?:desde|de parte de|de)|"
    r"(?:le habla|te habla|mi nombre es) [\w'-]+(?: [\w'-]+){0,2} "
    r"(?:de parte de|de)) (?:the |our |la |el |nuestra )?" + _FIRM + r"\b"
)
_NON_EMPLOYEE = re.compile(r"\b(?:client[ea]?|paciente|patient|customer|family member|"
                           r"spouse|espos[oa]|familiar)\b")
_AFFILIATED_INTRO = re.compile(
    r"\b(?:te|le|les) (?:habla|saluda) ([a-z]+(?: [a-z]+){0,3}?)"
    r"(?:,)? del (?:equipo|departamento) (?:legal|juridico)\b"
)
_INDIRECT_INTRO = re.compile(
    r"\b(?:no|nunca|cuando|segun|dice[n]?|dijo|dijeron|decia|mensaje|grabacion|ejemplo)\b"
)

# Direct outreach on behalf of a legal organization PLUS a commitment to assist
# the listener with their matter. Neither a firm name nor "I can help" suffices.
_OUTREACH = re.compile(
    r"\b(?:(?:(?:te|le|lo|la) )?(?:llamo|(?:estoy|estamos) llamando|me comunico)"
    r" (?:de parte (?:del?|de la)|desde|del?|en nombre (?:del?|de la))"
    r"|(?:soy|mi nombre es|(?:te|le) habla) [a-z'-]+(?: [a-z'-]+){0,2} "
    r"(?:de parte (?:del?|de la)|del?|de la)"
    r"|(?:i(?:'m| am) (?:calling|reaching out)|we(?:'re| are) calling|this is [a-z'-]+)"
    r" (?:from|with|on behalf of)) (?:the |our |la |nuestra |el )?" + _FIRM + r"\b"
)
_CASE_HELP = re.compile(
    r"\b(?:para (?:ayudarte|ayudarle|ayudarlo|ayudarla|asistirte|asistirle|guiarte|guiarle)"
    r"|(?:voy a|estoy aqui para) (?:ayudarte|ayudarle|asistirte|asistirle|guiarte|guiarle)"
    r"|(?:soy|sere) (?:la persona |el asesor |la asesora )?encargad[oa] de "
    r"(?:ayudarte|ayudarle|asistirte|asistirle|guiarte|guiarle)"
    r"|(?:soy|sere) la persona que (?:te|le) (?:va a ayudar|ayudara))"
    r" (?:con|en|durante|a traves de) (?:el|la|tu|su|todo el) "
    r"(?:accidente|caso|reclamo|reclamacion|proceso)\b"
    r"|\b(?:to (?:help|assist|guide) you|i(?:'ll| will) (?:be (?:helping|assisting)|help|assist) you"
    r"|i(?:'m| am) (?:the (?:person|specialist) (?:assigned|here) to (?:help|assist) you)"
    r"|i(?:'ll| will) be the (?:person|specialist) (?:helping|assisting) you)"
    r" (?:with|through) (?:your|the|this) (?:accident|case|claim|intake|process)\b"
)
_OWNERSHIP = re.compile(
    r"\b(?:(?:soy|sere) (?:la persona |el asesor |la asesora )?encargad[oa] de"
    r"|i(?:'m| am) the (?:person|specialist) (?:assigned|here) to"
    r"|(?:soy|sere) la persona que (?:te|le) (?:va a ayudar|ayudara)"
    r"|i(?:'ll| will) be (?:the (?:person|specialist) )?(?:helping|assisting) you)\b"
)
_FIRM_OUTREACH_PLURAL = re.compile(
    r"\b(?:te|le) (?:estamos llamando|llamamos) (?:de parte|en nombre) "
    r"de (?:los abogados|la firma de abogados)\b"
)
_ASSIST_ACCIDENT = re.compile(
    r"\b(?:estamos (?:aqui para|interesad[oa]s en)|queremos) "
    r"(?:ayudarte|ayudarle|asistirte|asistirle) (?:en el caso de|con) "
    r"(?:tu|su) accidente\b"
)
_TEAM_IDENTITY = re.compile(
    r"^(?:(?:si|claro|bueno|mire|ah|ok|okay|yes|well)[ ,.]+)*"
    r"(?:nosotros somos (?:los abogados|el equipo legal)|"
    r"we(?: are|'re) (?:the legal team|the attorneys))\b"
)
_EXTERNAL_TEAM = re.compile(
    r"\b(?:otra (?:parte|firma|oficina)|parte contraria|aseguradora|other (?:party|firm)|"
    r"opposing|insurance company|insurance carrier)\b"
)
_DIRECT_CASE_OFFER = re.compile(
    r"\b(?:estamos (?:aqui para|interesados en)|queremos) "
    r"(?:ayudarte|ayudarle) (?:en (?:el )?caso de|con) (?:tu|su) accidente\b"
    r"|\bwe(?: are|'re) here to help you with your (?:accident|injury) (?:claim|case)\b"
)
# PhoneBurner sometimes receives a Portuguese introduction. Recognizing the
# agent here does not change consumer language routing or intake eligibility.
_PT_OUTREACH = re.compile(
    r"\b(?:a gente (?:esta|ta) te ligando|estamos (?:te |lhe )?ligando|"
    r"estou (?:te |lhe )?ligando) (?:de|da|em nome da) "
    r"(?:(?:la |l[' ]?)?(?:ayuda|ajuda) latina|equipe juridica|equipe de advogados)\b"
)
_PT_OFFER = re.compile(
    r"\bpara (?:te |lhe )?ajudar (?:voce |voce a lidar )?com (?:o )?seu acidente\b"
)
_PT_INDIRECT = re.compile(
    r"\b(?:nao|nem|nunca|se|quando|disse|disseram|falou|mensagem|gravacao|"
    r"exemplo|repetir|repita|citando|segundo|imagine)\b"
)
# Strip only a *separate*, trailing courtesy question. Do not turn a question
# containing the outreach words into a statement. Do not strip reporting or
# quotation guards from the original turn.
_COURTESY_TAIL = re.compile(
    r"(?:[.!;]\s*¿?\s*|,\s*¿\s*|\s+¿\s*)"
    r"(?:(?:como\s*[, ]+)?como (?:estas?|esta usted|te (?:encuentras|va)|"
    r"se (?:siente|encuentra)|amanecio)(?: (?:hoy|el dia de hoy))?|"
    r"me (?:escucha|escuchas)|how are you(?: (?:doing|today|doing today))?)\s*\?\s*$"
)
_FOLLOWUP_QUESTIONS = re.compile(
    r"[.!]\s+(?:(?:¿como\b|how (?:are|have) you\b)[^.!?¿\n]{0,55}\?\s*){1,2}$"
)
_OUR_FIRM = re.compile(r"\b(?:nuestr[oa] (?:firma|bufete|equipo legal)|our (?:law firm|legal team))\b")
_UNSAFE_INTRO = re.compile(
    r"\b(?:no|not|never|don't|didn't|wasn't|voicemail|automated|recorded|repeat|repite|"
    r"grabado|grabada|contestador|segun|dijo|dijeron|dice|dijiste|told|said|tells|"
    r"asked|quoted|pretend|imagina|supongamos|suppose|if|si (?:te|le|yo|fuera)|"
    r"me explicaron|me dijeron|she explained|he explained|when you say|you said|"
    r"cuando dices|me repitio|mi (?:abogado|hermano|hermana)|my lawyer)\b"
)


def _guided_intros(transcript: str) -> set[str]:
    """Use only opening evidence, then follow that acoustic identity in the resolver."""
    found: set[str] = set()
    window: list[tuple[int, str]] = []
    previous = ""
    previous_at = -1
    quote_lead_in = False
    # A caller's answer can separate two complete declarations from one
    # acoustic identity. Never splice partial phrases across the interruption.
    anchors: dict[str, dict[str, int]] = {}
    for line in transcript.splitlines():
        m = _TURN.fullmatch(line)
        if not m or not m["prefix"].strip() or not _IDENTITY.fullmatch(m["label"]):
            window.clear()
            previous = ""
            anchors.clear()
            continue
        elapsed = 0
        for part in m["prefix"].split(":"):
            elapsed = elapsed * 60 + int(part)
        label, sentence = m["label"], _normalize(m["text"])
        contiguous = label == previous and 0 <= elapsed - previous_at <= 15
        skip_continuation = contiguous and quote_lead_in
        quote_lead_in = bool(re.search(
            r"\b(?:dijo|dijeron|decia|disse|disseram|falou|said|told me|the message was)"
            r"\s*[:,.]?\s*$", sentence
        ))
        if not contiguous:
            window.clear()
        previous, previous_at = label, elapsed
        statement = _COURTESY_TAIL.sub("", _FOLLOWUP_QUESTIONS.sub("", sentence))
        if (skip_continuation or elapsed > 90 or re.search(r"\bllamó\b", m["text"], re.I)
                or _REPORTED.search(sentence) or _UNSAFE_INTRO.search(sentence)
                or _NON_EMPLOYEE.search(sentence) or any(c in sentence for c in '"“”«»')
                or any(c in statement for c in "¿?")):
            window.clear()
            anchors.pop(label, None)
            continue
        marks = anchors.setdefault(label, {})
        if _TEAM_IDENTITY.search(statement) and not _EXTERNAL_TEAM.search(statement):
            marks["team"] = elapsed
        if _DIRECT_CASE_OFFER.search(statement):
            marks["offer"] = elapsed
        if ("team" in marks and "offer" in marks
                and abs(marks["team"] - marks["offer"]) <= 40):
            found.add(label)
        window = [(at, text) for at, text in window[-2:] if elapsed - at <= 25]
        window.append((elapsed, statement))
        speech = " ".join(re.sub(r"[^a-z0-9' ]+", " ", text) for _, text in window)
        speech = " ".join(speech.split())
        pt_intro = (_PT_OUTREACH.search(speech) and _PT_OFFER.search(speech)
                    and not _PT_INDIRECT.search(speech))
        if pt_intro or ((_CASE_HELP.search(speech) or _ASSIST_ACCIDENT.search(speech))
                and (_OUTREACH.search(speech) or _FIRM_OUTREACH_PLURAL.search(speech)
                or (_OWNERSHIP.search(speech) and _OUR_FIRM.search(speech)))):
            found.add(label)
    # Echo or repeated introductions on different identities cannot establish
    # which voice originated the speech. Leave other independent rules intact.
    return found if len(found) == 1 else set()


def _affiliated_staff_intro(text: str, names: set[str]) -> bool:
    """Require staff metadata, a direct self-introduction and an explicit affiliation."""
    m = _AFFILIATED_INTRO.search(text)
    if not m or _INDIRECT_INTRO.search(text) or _NON_EMPLOYEE.search(text):
        return False
    prefix = text[:m.start()].strip()
    if prefix and not re.search(r"(?:[,!:;]|\b(?:mucho|con|un) gusto)[ ,!]*$", prefix):
        return False
    spoken = m[1].split()
    return len(spoken[0]) >= 3 and any(
        name.split()[:len(spoken)] == spoken for name in names
    )
_PERSONAL_EVENT = re.compile(
    r"\b(?:i (?:got|was) rear[ -]ended|(?:another driver|the other (?:driver|car)) "
    r"(?:hit|rear[ -]ended) (?:me|my car)|i was taken to (?:the |a )?hospital|"
    r"my (?:back|neck|shoulder) (?:hurts|started hurting)|me llevaron al hospital|"
    r"(?:el otro (?:carro|conductor)|otro carro) me (?:choco|pego))\b"
)
_DISCOVERY = {
    "injury": re.compile(r"^(?:were you (?:hurt|injured)|did you (?:sustain|suffer) "
                         r"(?:any )?injur|(?:sufrio|tuvo) (?:usted )?(?:alguna )?lesion)"),
    "representation": re.compile(r"^(?:do you (?:currently |already )?have (?:an? )?"
                                 r"(?:attorney|lawyer)|are you (?:currently )?represented|"
                                 r"(?:cuenta con|tiene) (?:usted )?(?:un |algun )?abogado)\b"),
    "police": re.compile(r"^(?:did you (?:file|make) a police report|do you have (?:a|the) "
                         r"police report|(?:tiene|tienes) (?:el|un) reporte "
                         r"(?:policial|de (?:la )?policia))\b"),
    "incident_date": re.compile(r"^(?:when (?:was (?:your|the) accident|did (?:your|the) "
                                r"accident (?:happen|occur))|(?:cuando|en que fecha|que dia) "
                                r"(?:fue|ocurrio|sucedio) (?:el|su|tu) accidente)\b"),
}


@dataclass(frozen=True)
class RoleResolution:
    """Non-PII decision metadata plus the transcript with source text untouched."""

    transcript: str
    roles: dict[str, str]
    reasons: dict[str, str]
    version: str = VERSION


# A response is evidence only when its subject is the respondent. A bare
# confirmation, greeting, date, injury adjective or contact name is not enough.
_INTAKE_QUESTION = re.compile(
    r"\b(?:what happened|how did (?:the|your) (?:accident|crash) happen|"
    r"were you (?:driving|injured|hurt|involved)|did you (?:receive|seek|get) "
    r"(?:any )?(?:medical|treatment)|que (?:te |le )?(?:paso|sucedio)|"
    r"como (?:fue|ocurrio|paso) (?:el|tu|su) accidente|"
    r"(?:tuviste|tuvo|sufriste|sufrio) (?:un |alguna? )?(?:accidente|lesion)|"
    r"(?:recibiste|recibio) (?:atencion|tratamiento))\b"
)
_OWN_ANSWER = re.compile(
    r"^(?:(?:si|yes|well|bueno|pues|mire|actually|so|yo)[, ]+)*"
    r"(?:i (?:was (?:in|involved in) (?:a|an|the) (?:car |traffic )?(?:accident|crash)|"
    r"had (?:a|an) (?:car |traffic )?(?:accident|crash)|was (?:rear[ -]ended|the driver)|"
    r"went to (?:urgent care|the (?:emergency room|doctor|hospital))|"
    r"was stopped at (?:a|the) (?:red light|intersection))|"
    r"(?:tuve|sufri) un accidente|me accidente|me (?:pegaron|impactaron) por (?:detras|atras)|"
    r"(?:iba|estaba) (?:conduciendo|manejando)|estaba (?:parad[oa]|detenid[oa]) en "
    r"(?:el|un) semaforo|(?:fui|acudi) (?:a urgencias|al medico))\b"
)
_NONPERSONAL = re.compile(
    r"\b(?:wrong number|someone else|not me|my (?:example|client|son|daughter|husband|wife)|"
    r"no fui yo|no era yo|mi (?:cliente|espos[oa]|hij[oa])|hypothetical|imagine|"
    r"pretend|repeating|repitiendo|repita|supongamos|imagine que)\b"
)

# The participant can identify themselves before the operator asks a scripted
# question. Look for a collision involving the speaker personally, not just
# references to a car, a relative's treatment, or another driver's fault.
_VICTIM_ACCOUNT = re.compile(
    r"\b(?:la (?:persona|senora|muchacha) que me (?:choco|atropello)|"
    r"el (?:conductor|senor|muchacho) que me (?:choco|atropello)|"
    r"(?:el otro (?:conductor|carro)|otro (?:carro|vehiculo)) "
    r"(?:me (?:choco|impacto)|choco mi (?:carro|vehiculo))|"
    r"(?:the (?:person|driver|lady|guy) (?:who|that) (?:hit|rear[ -]ended) me)|"
    r"another (?:driver|vehicle) (?:hit|struck) my (?:vehicle|car))\b"
)
_OWN_DETAILS = re.compile(
    r"\b(?:mi (?:carro|vehiculo|seguro|aseguradora|abogado)|yo (?:iba|estaba)|"
    r"my (?:car|vehicle|insurance|lawyer|attorney)|i (?:went|had to|was driving))\b"
)


def _own_case_account(turns: list[str]) -> bool:
    safe = []
    for text in turns:
        clean = _normalize(text)
        if (_REPORTED.search(clean) or _NONPERSONAL.search(clean)
                or re.search(r"\b(?:no|not|never|si (?:me|yo)|nunca)\b", clean)
                or any(ch in clean for ch in '¿?"“”«»')):
            safe.append("")
        else:
            safe.append(clean)
    # Both statements must belong to this identity; no other speaker supplies
    # evidence. Require distinct turns close in this speaker's own narrative.
    for i, text in enumerate(safe):
        if (_VICTIM_ACCOUNT.search(text) or _OWN_ANSWER.search(text)) and any(
            _OWN_DETAILS.search(safe[j]) for j in range(max(0, i-3), min(len(safe), i+4))
            if j != i
        ):
            return True
    return False


def _corroborated_answers(transcript: str, roles: dict[str, str]) -> set[str]:
    if len(roles) != 2 or list(roles.values()).count("Agent") != 1:
        return set()
    found: set[str] = set()
    prompt_at: int | None = None
    for line in transcript.splitlines():
        m = _TURN.fullmatch(line)
        if not m or not m["prefix"].strip() or m["label"] not in roles:
            prompt_at = None
            continue
        elapsed = 0
        for part in m["prefix"].strip().split(":"):
            elapsed = elapsed * 60 + int(part)
        txt = _normalize(m["text"])
        unsafe = (_REPORTED.search(txt) or _NONPERSONAL.search(txt)
                  or any(ch in txt for ch in '"“”«»'))
        if roles[m["label"]] == "Agent":
            prompt_at = elapsed if not unsafe and "?" in txt and _INTAKE_QUESTION.search(txt) else None
        else:
            if (prompt_at is not None and 0 <= elapsed - prompt_at <= 30
                    and not unsafe and "?" not in txt and _OWN_ANSWER.search(txt)):
                found.add(m["label"])
            prompt_at = None
    return found


def resolve_phoneburner_roles(
    transcript: str,
    event: CallEvent,
    channel_roles: Mapping[str, Role] | None = None,
) -> RoleResolution:
    """Resolve only observed identities; never infer roles from their numeric IDs."""

    texts: dict[str, list[str]] = {}
    for line in transcript.splitlines():
        match = _TURN.match(line)
        if match and _IDENTITY.fullmatch(match["label"]):
            texts.setdefault(match["label"], []).append(match["text"])

    roles: dict[str, str] = {}
    reasons: dict[str, str] = {}
    configured = channel_roles or {}
    channel_identities: dict[str, list[str]] = {}
    for identity in texts:
        channel = _CHANNEL.fullmatch(identity)
        if channel:
            channel_identities.setdefault(channel[1], []).append(identity)

    metadata, metadata_conflicts = _explicit_speaker_roles(event, set(texts))
    agent_names = _agent_names(event)
    guided_intros = _guided_intros(transcript)
    for identity, turns in texts.items():
        evidence: set[Role] = {"Agent"} if identity in guided_intros else set()
        discovery: dict[str, set[int]] = {}
        for turn_index, text in enumerate(turns):
            # Remove quotations before examining identity evidence. Questions in
            # a later sentence must not erase an otherwise explicit introduction.
            unquoted = re.sub(r'"[^"\n]*"|“[^”\n]*”|«[^»\n]*»', "", text)
            for sentence in re.split(r"(?<=[.!?])\s+|(?=¿)", unquoted):
                clean = _normalize(sentence)
                if _REPORTED.search(clean) or re.search(
                    r"\bsi\b", re.sub(r"^\s*si\s*,", "", sentence, flags=re.I), re.I
                ):
                    continue
                if "?" in clean or "¿" in sentence:
                    question = clean.strip("¿? ,.")
                    for topic, pattern in _DISCOVERY.items():
                        if pattern.search(question):
                            discovery.setdefault(topic, set()).add(turn_index)
                    continue
                if (_AGENT.search(clean) or (_PROFESSIONAL_INTRO.search(clean)
                                             and not _NON_EMPLOYEE.search(clean))
                        or _named_introduction(clean, agent_names)
                        or _affiliated_staff_intro(clean, agent_names)):
                    evidence.add("Agent")
                anchor = _LEAD.search(clean) or _PERSONAL_EVENT.search(clean)
                if (anchor and not _NONPERSONAL.search(clean) and not re.search(
                    r"\b(?:not|never|no|nunca)\b.{0,22}$", clean[:anchor.start()],
                )):
                    evidence.add("Lead")

        # Require legal/intake context, two different topics and separate turns.
        # Repeating a question, even many times, does not establish a role.
        if ("representation" in discovery and len(discovery) >= 2
                and len(set.union(*discovery.values())) >= 2):
            evidence.add("Agent")

        channel = _CHANNEL.fullmatch(identity)
        mapped = configured.get(channel[1]) if channel else None
        mixed_channel = bool(channel and len(channel_identities[channel[1]]) > 1)
        metadata_role = metadata.get(identity)
        new_account = "Lead" not in evidence and _own_case_account(turns)
        if new_account:
            evidence.add("Lead")
        candidates = set(evidence)
        if mapped and not mixed_channel:
            candidates.add(mapped)
        if metadata_role:
            candidates.add(metadata_role)

        if identity in metadata_conflicts or (mapped and mixed_channel) or len(candidates) > 1:
            roles[identity], reasons[identity] = "Unknown", "conflicting_evidence"
        elif identity.startswith("Unseparated ") and not (mapped or metadata_role):
            roles[identity], reasons[identity] = "Unknown", "no_speaker_separation"
        elif candidates:
            roles[identity] = next(iter(candidates))
            reasons[identity] = (
                "verified_channel_map"
                if mapped
                else "explicit_speaker_metadata"
                if metadata_role
                else "first_person_case_account"
                if new_account
                else "identity_evidence"
            )
        else:
            roles[identity], reasons[identity] = "Unknown", "insufficient_evidence"

    for identity in _corroborated_answers(transcript, roles):
        if reasons[identity] == "insufficient_evidence":
            roles[identity], reasons[identity] = "Lead", "first_person_intake_response"

    # Two different acoustic identities claiming Agent may be transfers or a
    # diarization split. Do not turn either into Lead to force a two-party map.
    result: list[str] = []
    unknown_turns = 0
    labeled_turns = 0
    for line in transcript.splitlines(keepends=True):
        match = _TURN.match(line)
        if match:
            identity = match["label"]
            if identity in roles:
                role = roles[identity]
                label = f"Unknown ({identity})" if role == "Unknown" else role
            else:
                # Already-labelled passthrough text has no verified acoustic identity.
                # Never pass along an inherited business role as new attribution.
                role = label = "Unknown"
            unknown_turns += role == "Unknown"
            labeled_turns += 1
            # Replace the label only: preserve punctuation, quotes and wording.
            start, end = match.span("label")
            line = line[:start] + label + line[end:]
        result.append(line)

    logger.info(
        "speaker_roles version=%s policy=%s call_id=%s resolved=%d unknown=%d conflicts=%d "
        "unknown_turns=%d labeled_turns=%d",
        VERSION,
        POLICY_REVISION,
        event.call_id,
        sum(role != "Unknown" for role in roles.values()),
        sum(role == "Unknown" for role in roles.values()),
        sum(reason == "conflicting_evidence" for reason in reasons.values()),
        unknown_turns,
        labeled_turns,
    )
    return RoleResolution("".join(result), roles, reasons)


def _explicit_speaker_roles(
    event: CallEvent,
    identities: set[str],
) -> tuple[dict[str, Role], set[str]]:
    """Speaker metadata never implicitly maps a diarized speaker to a channel."""

    assignments: dict[str, set[Role]] = {}
    for key in ("speaker_roles", "speakerRoles", "speaker_role_map", "speakerRoleMap"):
        mapping = event.raw_payload.get(key)
        if not isinstance(mapping, Mapping):
            continue
        for raw_identity, role in mapping.items():
            identity = str(raw_identity).strip()
            if identity.isdigit():
                identity = f"Speaker {identity}"
            normalized = str(role).strip().lower()
            if identity in identities and normalized in {"agent", "lead"}:
                assignments.setdefault(identity, set()).add(
                    "Agent" if normalized == "agent" else "Lead"
                )
    for key, role in (
        ("agent_speaker", "Agent"),
        ("agentSpeaker", "Agent"),
        ("lead_speaker", "Lead"),
        ("leadSpeaker", "Lead"),
    ):
        raw_identity = event.raw_payload.get(key)
        if raw_identity is None:
            continue
        identity = str(raw_identity).strip()
        if identity.isdigit():
            identity = f"Speaker {identity}"
        if identity in identities:
            assignments.setdefault(identity, set()).add("Agent" if role == "Agent" else "Lead")
    return (
        {identity: next(iter(roles)) for identity, roles in assignments.items() if len(roles) == 1},
        {identity for identity, roles in assignments.items() if len(roles) > 1},
    )


def _agent_names(event: CallEvent) -> set[str]:
    # IDs, phone numbers and direction are not evidence of who uttered a turn.
    values = [
        event.agent_name,
        event.raw_payload.get("agent_name"),
        event.raw_payload.get("agentName"),
        event.raw_payload.get("from_name"),
    ]
    from_party = event.raw_payload.get("from")
    if isinstance(from_party, Mapping):
        values.append(from_party.get("name"))
    return {
        _normalize(value)
        for value in values
        if isinstance(value, str)
        and len(value.strip()) >= 3
        and all(part.isalpha() for part in value.split())
    }


def _named_introduction(text: str, names: set[str]) -> bool:
    return any(
        re.search(
            rf"^(?:hello[,.]? |hi[,.]? |hola[,.]? )?"
            rf"(?:this is|my name is|soy|mi nombre es|le habla|te habla)\s+{re.escape(name)}"
            rf"(?:[,.]|$|\s+(?:calling|from|with|de|del|y)\b)",
            text,
        )
        for name in names
    )


def _normalize(text: str) -> str:
    return " ".join(
        "".join(
            char
            for char in unicodedata.normalize("NFKD", text.lower().replace("’", "'"))
            if not unicodedata.combining(char)
        ).split()
    )
