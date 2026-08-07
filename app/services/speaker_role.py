"""Resolve raw diarized speakers into call participant roles."""

from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from app.models.call_event import CallEvent

logger = logging.getLogger(__name__)

Role = Literal["Agent", "Lead"]

_TURN_RE = re.compile(
    r"^(?P<prefix>(?:\d{2}:\d{2}\s+)?)\[(?P<label>[^\]]+)]:\s*(?P<text>.*)$"
)
_SPEAKER_RE = re.compile(r"^speaker\s+(?P<number>\d+)$", re.IGNORECASE)

_EXPLICIT_ROLE_MAP_KEYS = ("speaker_roles", "speakerRoles", "speaker_role_map", "speakerRoleMap")
_EXPLICIT_AGENT_KEYS = ("agent_speaker", "agentSpeaker", "agent_speaker_id", "agentSpeakerId")
_EXPLICIT_LEAD_KEYS = (
    "lead_speaker",
    "leadSpeaker",
    "lead_speaker_id",
    "leadSpeakerId",
    "patient_speaker",
    "patientSpeaker",
    "patient_speaker_id",
    "patientSpeakerId",
)

_AGENT_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(te|le)\s+llam(o|ó|aba)\b",
        r"\bde parte\b",
        r"firma de abogados",
        r"equipo legal",
        r"ayuda latina",
        r"legal team",
        r"law firm",
        r"following up",
        r"calling from",
        r"this is .* calling",
        r"c[oó]mo te encuentras",
        r"c[oó]mo est[aá]s",
        r"schedule",
        r"appointment",
        r"doctor'?s office",
        r"clinic",
        r"medhub",
    )
)
_LEAD_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bme choc",
        r"\bmi carro\b",
        r"\bmi accidente\b",
        r"\byo iba\b",
        r"manejando",
        r"\bdolor\b",
        r"hospital",
        r"terapia",
        r"doctor",
        r"aseguranza",
        r"seguro",
        r"\bfirm[eé]\b",
        r"\babogado\b",
        r"no estoy interesado",
        r"no me interesa",
        r"\bme lastim",
        r"my pain",
        r"my appointment",
        r"my doctor",
        r"my insurance",
        r"\bi was\b",
        r"\bi have\b",
        r"\bi need\b",
    )
)


@dataclass(frozen=True)
class _Turn:
    raw: str
    prefix: str
    speaker: str
    text: str


def resolve_speaker_roles(transcript: str, event: CallEvent) -> str:
    """Return transcript text with raw Speaker labels mapped to Agent/Lead roles."""

    turns = _parse_turns(transcript)
    speakers = _ordered_speakers(turns)
    if not speakers:
        return transcript

    role_map = _metadata_role_map(event, turns, speakers)
    if role_map is None:
        role_map = _semantic_role_map(turns, speakers)

    if role_map is None:
        role_map = _fallback_role_map(speakers)
        logger.warning(
            "speaker role resolved via fallback heuristic, no reliable metadata for call_id=%s",
            event.call_id,
        )

    return _replace_speaker_labels(transcript, role_map)


def _metadata_role_map(
    event: CallEvent,
    turns: list[_Turn],
    speakers: list[str],
) -> dict[str, Role] | None:
    role_map = _explicit_role_map(event.raw_payload, speakers)
    if role_map is not None:
        return role_map
    return _agent_self_identification_role_map(event, turns, speakers)


def _parse_turns(transcript: str) -> list[_Turn]:
    turns: list[_Turn] = []
    for raw_line in transcript.splitlines():
        match = _TURN_RE.match(raw_line)
        if match is None:
            continue
        speaker = match.group("label").strip()
        if not _is_raw_speaker_label(speaker):
            continue
        turns.append(
            _Turn(
                raw=raw_line,
                prefix=match.group("prefix"),
                speaker=speaker,
                text=match.group("text").strip(),
            )
        )
    return turns


def _ordered_speakers(turns: list[_Turn]) -> list[str]:
    speakers: list[str] = []
    seen: set[str] = set()
    for turn in turns:
        if turn.speaker in seen:
            continue
        speakers.append(turn.speaker)
        seen.add(turn.speaker)
    return speakers


def _explicit_role_map(raw_payload: Mapping[str, Any], speakers: list[str]) -> dict[str, Role] | None:
    role_map: dict[str, Role] = {}
    assignments: dict[str, set[Role]] = {}

    for key in _EXPLICIT_ROLE_MAP_KEYS:
        value = raw_payload.get(key)
        if not isinstance(value, Mapping):
            continue
        for raw_speaker, raw_role in value.items():
            speaker = _speaker_label_from_value(raw_speaker, speakers)
            role = _role_from_value(raw_role)
            if speaker and role:
                assignments.setdefault(speaker, set()).add(role)
                role_map[speaker] = role

    for key in _EXPLICIT_AGENT_KEYS:
        speaker = _speaker_label_from_value(raw_payload.get(key), speakers)
        if speaker:
            assignments.setdefault(speaker, set()).add("Agent")
            role_map[speaker] = "Agent"

    for key in _EXPLICIT_LEAD_KEYS:
        speaker = _speaker_label_from_value(raw_payload.get(key), speakers)
        if speaker:
            assignments.setdefault(speaker, set()).add("Lead")
            role_map[speaker] = "Lead"

    if not role_map:
        return None

    if _has_conflicting_roles(role_map, assignments):
        logger.warning("conflicting speaker role metadata ignored")
        return None

    return _complete_role_map(role_map, speakers)


def _agent_self_identification_role_map(
    event: CallEvent,
    turns: list[_Turn],
    speakers: list[str],
) -> dict[str, Role] | None:
    agent_names = _candidate_agent_names(event)
    if not agent_names:
        return None

    candidate_speakers = {
        turn.speaker
        for turn in turns
        if _speaker_self_identifies_as_agent(turn.text, agent_names)
    }
    if len(candidate_speakers) != 1:
        return None

    return _complete_role_map({candidate_speakers.pop(): "Agent"}, speakers)


def _candidate_agent_names(event: CallEvent) -> tuple[str, ...]:
    raw_payload = event.raw_payload
    values = [
        event.agent_name,
        event.agent_id,
        raw_payload.get("agent_name"),
        raw_payload.get("agentName"),
        raw_payload.get("from_name"),
        raw_payload.get("fromName"),
        _nested_value(raw_payload, "from", "name"),
        _join_name(
            _nested_value(raw_payload, "agent", "first_name"),
            _nested_value(raw_payload, "agent", "last_name"),
        ),
    ]
    names: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        normalized = _normalize_name(value)
        if normalized is None or normalized in seen:
            continue
        names.append(normalized)
        seen.add(normalized)
    return tuple(names)


def _speaker_self_identifies_as_agent(text: str, agent_names: tuple[str, ...]) -> bool:
    normalized_text = _normalize_for_text_match(text)
    for agent_name in agent_names:
        escaped_name = re.escape(agent_name)
        if re.search(rf"\b(this is|my name is|soy|habla)\s+{escaped_name}\b", normalized_text):
            return True
        if re.search(rf"\b{escaped_name}\s+(calling|from|with)\b", normalized_text):
            return True
    return False


def _semantic_role_map(turns: list[_Turn], speakers: list[str]) -> dict[str, Role] | None:
    if len(speakers) < 2:
        return None

    scores = _speaker_scores(turns)
    agent_candidate = _best_speaker(scores, role="agent")
    lead_candidate = _best_speaker(scores, role="lead")

    if agent_candidate and lead_candidate and agent_candidate != lead_candidate:
        return _complete_role_map({agent_candidate: "Agent", lead_candidate: "Lead"}, speakers)

    if agent_candidate:
        return _complete_role_map({agent_candidate: "Agent"}, speakers)

    if lead_candidate:
        return _complete_role_map({lead_candidate: "Lead"}, speakers)

    return None


def _speaker_scores(turns: list[_Turn]) -> dict[str, Counter[str]]:
    scores: dict[str, Counter[str]] = {}
    for turn in turns:
        speaker_scores = scores.setdefault(turn.speaker, Counter())
        speaker_scores["agent"] += _pattern_hits(turn.text, _AGENT_PATTERNS)
        speaker_scores["lead"] += _pattern_hits(turn.text, _LEAD_PATTERNS)
    return scores


def _pattern_hits(text: str, patterns: tuple[re.Pattern[str], ...]) -> int:
    return sum(1 for pattern in patterns if pattern.search(text))


def _best_speaker(scores: dict[str, Counter[str]], *, role: Literal["agent", "lead"]) -> str | None:
    opposite = "lead" if role == "agent" else "agent"
    ranked = sorted(
        (
            (speaker_scores[role] - speaker_scores[opposite], speaker_scores[role], speaker)
            for speaker, speaker_scores in scores.items()
        ),
        reverse=True,
    )
    if not ranked:
        return None

    margin, hits, speaker = ranked[0]
    if hits <= 0 or margin <= 0:
        return None
    if len(ranked) > 1 and margin == ranked[1][0] and hits == ranked[1][1]:
        return None
    return speaker


def _fallback_role_map(speakers: list[str]) -> dict[str, Role]:
    role_map: dict[str, Role] = {speaker: "Lead" for speaker in speakers}
    role_map[speakers[0]] = "Agent"
    return role_map


def _complete_role_map(role_map: dict[str, Role], speakers: list[str]) -> dict[str, Role]:
    completed = dict(role_map)
    assigned_agents = [speaker for speaker, role in completed.items() if role == "Agent"]
    assigned_leads = [speaker for speaker, role in completed.items() if role == "Lead"]

    if assigned_agents and not assigned_leads:
        for speaker in speakers:
            completed.setdefault(speaker, "Lead")
    elif assigned_leads and not assigned_agents and len(speakers) == 2:
        for speaker in speakers:
            completed.setdefault(speaker, "Agent")
    else:
        for speaker in speakers:
            completed.setdefault(speaker, "Lead")

    return completed


def _has_conflicting_roles(role_map: dict[str, Role], assignments: dict[str, set[Role]]) -> bool:
    if any(len(roles) > 1 for roles in assignments.values()):
        return True
    return len([speaker for speaker, role in role_map.items() if role == "Agent"]) > 1


def _speaker_label_from_value(value: Any, speakers: list[str]) -> str | None:
    if isinstance(value, int):
        value = str(value)
    if not isinstance(value, str):
        return None

    normalized = value.strip()
    if not normalized:
        return None

    if normalized in speakers:
        return normalized

    if normalized.isdigit():
        candidate = f"Speaker {normalized}"
        return candidate if candidate in speakers else None

    match = _SPEAKER_RE.match(normalized)
    if match:
        candidate = f"Speaker {match.group('number')}"
        return candidate if candidate in speakers else None

    return None


def _role_from_value(value: Any) -> Role | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if normalized == "agent":
        return "Agent"
    if normalized in {"lead", "patient", "caller", "customer"}:
        return "Lead"
    return None


def _nested_value(payload: Mapping[str, Any], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _join_name(first_name: Any, last_name: Any) -> str | None:
    parts = [
        _normalize_name(value)
        for value in (first_name, last_name)
        if isinstance(value, str)
    ]
    joined = " ".join(part for part in parts if part)
    return joined or None


def _normalize_name(value: str) -> str | None:
    normalized = _normalize_for_text_match(value)
    if len(normalized) < 3:
        return None
    if not re.search(r"[a-z]", normalized):
        return None
    return normalized


def _normalize_for_text_match(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _is_raw_speaker_label(label: str) -> bool:
    return bool(_SPEAKER_RE.match(label.strip()))


def _replace_speaker_labels(transcript: str, role_map: dict[str, Role]) -> str:
    lines: list[str] = []
    for raw_line in transcript.splitlines():
        match = _TURN_RE.match(raw_line)
        if match is None:
            lines.append(raw_line)
            continue
        speaker = match.group("label").strip()
        role = role_map.get(speaker)
        if role is None:
            lines.append(raw_line)
            continue
        lines.append(f"{match.group('prefix')}[{role}]: {match.group('text')}")
    return "\n".join(lines)
