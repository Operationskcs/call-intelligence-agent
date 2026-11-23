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
POLICY_REVISION = "context-2026-11-20"
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
    for identity, turns in texts.items():
        evidence: set[Role] = set()
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
                if _LEAD.search(clean) or _PERSONAL_EVENT.search(clean):
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
                else "identity_evidence"
            )
        else:
            roles[identity], reasons[identity] = "Unknown", "insufficient_evidence"

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
