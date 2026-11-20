import json
from unittest.mock import AsyncMock

import pytest

from app.adapters.stt.deepgram import _format_multichannel_transcript
from app.models.call_event import CallEvent
from app.services.phoneburner_roles import resolve_phoneburner_roles
from app.services.phoneburner_semantic import _read_vote, refine

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
@pytest.mark.parametrize("fenced", [False, True])
async def test_semantic_boundary(mock_call_event: CallEvent, fault: str, fenced: bool) -> None:
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
        text = "not json" if fault == "bad_json" else json.dumps(res)
        return f"```json\n{text}\n```" if fenced else text

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


@pytest.mark.parametrize("header", ["```", "```json", "```JSON"])
def test_single_fence_crlf(header: str) -> None:
    assert _read_vote(f' \r\n{header}\r\n{{"tracks":[]}}\r\n```\r\n') == {"tracks": []}


@pytest.mark.parametrize("text", [
    'Here is the JSON:\n```json\n{}\n```', '```json\n{}\n```\nDone.',
    '```json\n{}\n```\n```json\n{}\n```', '```json\n{}', '```python\n{}\n```',
    '{} {}', '```json\n{} {}\n```', '```json\n[]\n```', '{"tracks":[{"id":"A","id":"B"}]}',
    '{"tracks":[],"score":NaN}',
])
def test_ambiguous_or_non_json_vote_rejected(text: str) -> None:
    with pytest.raises(ValueError):
        _read_vote(text)


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


@pytest.mark.parametrize(("handoff", "new_segment"), [
    ("One moment please.", "I have the paperwork right here."),  # diarizer split
    ("Let me put my daughter on the phone.", "I am her daughter."),  # third party
    ("Le paso a mi hija.", "Soy su hija, mi madre está aquí."),  # third party, ES
])
async def test_same_channel_return_is_not_proof_of_a_single_voice(
    mock_call_event: CallEvent, handoff: str, new_segment: str
) -> None:
    # Identical channel/speaker sequence and timing in every case. Even a return to
    # the original speaker after a short segment does not establish who said it.
    observations = [
        (0, 0, "I am calling from the law firm."),
        (1, 0, "My car was hit."),
        (1, 0, handoff),
        (1, 1, new_segment),
        (1, 0, "Thanks, I am back."),
    ]
    raw = _format_multichannel_transcript({"results": {
        "channels": [{}, {}],
        "utterances": [
            {"channel": ch, "speaker": sp, "start": 4*i, "transcript": text}
            for i, (ch, sp, text) in enumerate(observations)
        ],
    }})
    base = resolve_phoneburner_roles(raw, mock_call_event)
    assert base.roles == {
        "Channel 0 Speaker 0": "Agent",
        "Channel 1 Speaker 0": "Lead",
        "Channel 1 Speaker 1": "Unknown",
    }
    ask = AsyncMock()
    assert await refine(raw, base, ask) == base.transcript
    assert f"[Unknown (Channel 1 Speaker 1)]: {new_segment}" in base.transcript
    ask.assert_not_called()


async def test_two_parseable_but_opposing_votes_abstain(mock_call_event: CallEvent) -> None:
    # Both speakers describe personal and intake experience: shape-valid votes
    # can disagree even when every quoted excerpt is exact and on the right track.
    raw = (
        "[Speaker 1]: Our office reviews new applications.\n"
        "[Speaker 2]: Our office reviews incoming applications.\n"
        "[Speaker 1]: I completed my insurance claim yesterday.\n"
        "[Speaker 2]: I completed my insurance claim today.\n"
        "[Speaker 1]: What date did this collision occur on?\n"
        "[Speaker 2]: What date did this collision take place?\n"
        "[Speaker 1]: My vehicle spun around twice.\n"
        "[Speaker 2]: My vehicle spun around once."
    )
    base = resolve_phoneburner_roles(raw, mock_call_event)
    assert set(base.roles.values()) == {"Unknown"}
    calls = []

    async def ask(body: str) -> str:
        prompt = json.loads(body)
        calls.append(prompt["order"])
        # A and X are *opposite* original tracks under the two alias mappings.
        agent, lead = prompt["tracks"]
        rows = []
        for track, role, specs in [
            (agent, "Agent", [("Our office", "firm_process"), ("What date", "intake_discovery")]),
            (lead, "Lead", [("I completed", "own_claim"), ("My vehicle", "own_accident")]),
        ]:
            evidence = []
            for prefix, kind in specs:
                turn = next(t for t in prompt["turns"]
                            if t["track"] == track and t["speech"].startswith(prefix))
                evidence.append({"turn": turn["id"], "kind": kind, "quote": turn["speech"]})
            rows.append({"id": track, "role": role, "evidence": evidence})
        return "```json\n" + json.dumps({"tracks": rows}) + "\n```"

    assert await refine(raw, base, ask) == base.transcript
    assert calls == ["chronological", "grouped by speaker"]
