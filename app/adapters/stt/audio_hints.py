"""Ephemeral timestamps for an independent Intake listen; no words or identities retained."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


@dataclass(frozen=True, order=True)
class Hint:
    priority: int
    at: int
    reason: str


def select(hints: list[Hint], limit: int = 2) -> tuple[Hint, ...]:
    selected: list[Hint] = []
    for h in sorted(hints):
        if all(abs(h.at - old.at) >= 8 for old in selected):
            selected.append(h)
            if len(selected) == limit:
                break
    return tuple(selected)


def number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def recognition_hints(response: dict[str, Any]) -> tuple[Hint, ...]:
    channels = response.get("results", {}).get("channels")
    if not isinstance(channels, list) or len(channels) not in (1, 2):
        return ()
    hints: list[Hint] = []
    tracks: list[list[tuple[float, float]]] = []
    for ch in channels:
        tracks.append([])
        alts = ch.get("alternatives") if isinstance(ch, dict) else None
        if not isinstance(alts, list) or not alts or not isinstance(alts[0], dict):
            continue
        previous: list[tuple[int, float]] = []
        for w in alts[0].get("words") or []:
            if not isinstance(w, dict) or not number(w.get("start")) or not number(w.get("end")):
                continue
            start, end = w["start"], w["end"]
            if end <= start:
                continue
            tracks[-1].append((start, end))
            if number(w.get("confidence")) and w["confidence"] < .65:
                hints.append(Hint(2, int(start), "unclear_word"))
            if number(w.get("speaker_confidence")) and w["speaker_confidence"] < .65:
                hints.append(Hint(1, int(start), "uncertain_voice"))
            speaker = w.get("speaker")
            if type(speaker) is int:
                if (len(previous) == 2 and previous[0][0] == speaker != previous[1][0]
                        and 0 <= start - previous[0][1] <= 2):
                    hints.append(Hint(1, int(previous[1][1]), "voice_switch"))
                previous = (previous + [(speaker, start)])[-2:]
    if len(tracks) == 2:
        left, right = (sorted(track) for track in tracks)
        i = j = 0
        while i < len(left) and j < len(right):
            start = max(left[i][0], right[j][0])
            if min(left[i][1], right[j][1]) - start >= .18:
                hints.append(Hint(2, int(start), "overlap"))
            if left[i][1] <= right[j][1]:
                i += 1
            else:
                j += 1
    return select(hints)
