"""Ephemeral, word-timed proposal. Never change the published transcript without audio proof."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from app.adapters.stt.audio_hints import Hint, number
from app.services.conduct_audio import embedded_clause
from app.services.phoneburner_roles import _TURN, _normalize


@dataclass(frozen=True)
class Proposal:
    index: int
    original: str
    pieces: tuple[tuple[int, str], ...]
    clause: int
    end: float

    @property
    def hint(self) -> Hint:
        return Hint(0, self.pieces[self.clause][0], "timed_conduct_clause")

    def render(self, transcript: str, *, business: bool) -> str | None:
        lines = transcript.splitlines()
        if self.index >= len(lines):
            return None
        m = _TURN.fullmatch(lines[self.index])
        if not m or m["text"] != self.original or (business and m["label"] != "Agent"):
            return None
        lines[self.index : self.index + 1] = [
            f"{at // 60:02}:{at % 60:02} [{m['label']}]: {text}" for at, text in self.pieces
        ]
        return "\n".join(lines)


def _seconds(prefix: str) -> int:
    t = 0
    for p in prefix.split(":"):
        t = 60 * t + int(p)
    return t


def propose(raw: str, response: dict[str, Any]) -> Proposal | None:
    channels = response.get("results", {}).get("channels")
    if not isinstance(channels, list) or len(channels) != 2:
        return None
    turns = [_TURN.fullmatch(line) for line in raw.splitlines()]
    candidates: list[Proposal] = []
    for i, m in enumerate(turns):
        if not m or not m["prefix"].strip():
            return None
        clause = embedded_clause(m["text"])
        role = re.fullmatch(r"Channel ([01]) Speaker (\d+)", m["label"])
        if not clause or not role:
            continue
        ch = channels[int(role[1])]
        alts = ch.get("alternatives") if isinstance(ch, dict) else None
        if not isinstance(alts, list) or not alts or not isinstance(alts[0], dict):
            continue
        ws = alts[0].get("words")
        if not isinstance(ws, list):
            continue
        tokens = m["text"].split()
        at = _seconds(m["prefix"])
        # Never infer time by proportion of words or use a channel number as a business role.
        matches = []
        for pos, w in enumerate(ws):
            if not isinstance(w, dict) or not number(w.get("start")) or int(w["start"]) != at:
                continue
            span = ws[pos : pos + len(tokens)]
            if (
                len(span) != len(tokens)
                or any(
                    not isinstance(x, dict)
                    or x.get("speaker") != int(role[2])
                    or (x.get("punctuated_word") or x.get("word")) != tok
                    or not number(x.get("start"))
                    or not number(x.get("end"))
                    or x["start"] >= x["end"]
                    for x, tok in zip(span, tokens, strict=True)
                )
                or any(a["end"] > b["start"] for a, b in zip(span, span[1:]))
            ):
                continue
            matches.append(span)
        if len(matches) != 1:
            continue
        span = matches[0]
        target = clause.split()
        # Consume entire word tokens: never manufacture a timestamp inside a provider word.
        offsets = [
            j
            for j in range(len(tokens) - len(target) + 1)
            if _normalize(" ".join(tokens[j : j + len(target)])).strip(" .,;!:¡") == clause
        ]
        if len(offsets) != 1:
            continue
        left = offsets[0]
        right = left + len(target)
        pieces = tuple(
            (int(span[a]["start"]), " ".join(tokens[a:b]))
            for a, b in [(0, left), (left, right), (right, len(span))]
            if a < b
        )
        if len(set(t for t, _ in pieces)) != len(pieces):
            continue
        nxt = turns[i + 1] if i + 1 < len(turns) else None
        if nxt and (not nxt["prefix"].strip() or _seconds(nxt["prefix"]) < span[-1]["end"]):
            continue
        candidates.append(Proposal(i, m["text"], pieces, int(left > 0), span[-1]["end"]))
    return candidates[0] if len(candidates) == 1 else None
