import json
from unittest.mock import AsyncMock

import pytest

from app.models.call_event import CallEvent
from app.services.phoneburner_roles import resolve_phoneburner_roles
from app.services.phoneburner_semantic import refine

RAW = (
    "00:01 [Speaker 7]: Our team handles new client applications.\n"
    "00:03 [Speaker 9]: My vehicle spun around at the intersection.\n"
    "00:05 [Speaker 7]: What date did this collision occur on?\n"
    "00:07 [Speaker 9]: I received a brace at urgent care.\n"
)


def reply(body: str) -> dict:
    rows = sorted(
        json.loads(body)["turns"],
        key=lambda r: (
            0
            if r["speech"].startswith("Our team")
            else 1
            if "vehicle" in r["speech"]
            else 2
            if r["speech"].startswith("What date")
            else 3
        ),
    )
    return {
        "tracks": [
            {
                "id": rows[i]["track"],
                "role": role,
                "evidence": [
                    {"turn": rows[j]["id"], "kind": kind, "quote": rows[j]["speech"]}
                    for j, kind in zip((i, i + 2), kinds, strict=True)
                ],
            }
            for i, role, kinds in [
                (0, "Agent", ["firm_process", "intake_discovery"]),
                (1, "Lead", ["own_accident", "own_injury_or_care"]),
            ]
        ]
    }


@pytest.mark.parametrize(
    "fault",
    [
        "none",
        "bad_quote",
        "wrong_track",
        "repeated_turn",
        "disagree",
        "bad_json",
        "timeout",
        "one_track",
        "unknown",
        "reported",
        "twice",
    ],
)
async def test_semantic_boundary(mock_call_event: CallEvent, fault: str) -> None:
    raw = RAW.replace("My vehicle", "You said my vehicle") if fault == "reported" else RAW
    base = resolve_phoneburner_roles(raw, mock_call_event)
    assert set(base.roles.values()) == {"Unknown"}
    bodies = []

    async def ask(body: str) -> str:
        bodies.append(body)
        assert "Speaker 7" not in body and "Speaker 9" not in body
        res = reply(body)
        row = res["tracks"][1]
        if fault == "timeout":
            raise TimeoutError()
        if fault == "bad_quote":
            row["evidence"][0]["quote"] = "This quote was never said"
        if fault == "wrong_track":
            row["evidence"][0]["turn"] = 0
        if fault == "repeated_turn":
            row["evidence"][1]["turn"] = 1
            row["evidence"][1]["quote"] = row["evidence"][0]["quote"]
        if fault == "twice":
            row["evidence"][1]["kind"] = row["evidence"][0]["kind"]
        if fault == "one_track":
            res["tracks"].pop()
        if fault == "unknown" or (fault == "disagree" and len(bodies) == 2):
            row["role"] = "Unknown"
        return "not json" if fault == "bad_json" else json.dumps(res)

    result = await refine(raw, base, ask)
    if fault == "none":
        assert result == raw.replace("[Speaker 7]", "[Agent]").replace("[Speaker 9]", "[Lead]")
        assert len(bodies) == 2 and bodies[0] != bodies[1]
        assert (
            json.loads(bodies[0])["turns"][0]["speech"]
            != json.loads(bodies[1])["turns"][0]["speech"]
        )
    else:
        assert result == base.transcript


@pytest.mark.parametrize(
    "raw",
    [
        "[Speaker 0]: Hi.",
        "[Speaker 0]: Hello.\n[Speaker 1]: Hi there.",
        "[Channel 0]: I am calling from the law firm.\n[Channel 0]: My car was hit.\n[Channel 1]: Hi.",
        "[Unseparated Channel 0]: Hello.\n[Speaker 3]: Okay.",
        "[Speaker 4]: Hello.\n[Speaker 5]: Hi.\n[Speaker 6]: Bye.",
        "[Speaker 4]: Hello.\n[Speaker 5]: Hi.\nUnattributable speech.",
    ],
)
async def test_noneligible_inputs_do_not_invoke_model(mock_call_event: CallEvent, raw: str) -> None:
    base = resolve_phoneburner_roles(raw, mock_call_event)
    ask = AsyncMock()
    assert await refine(raw, base, ask) == base.transcript
    ask.assert_not_called()
