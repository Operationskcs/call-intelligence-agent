# PhoneBurner speaker attribution

**Scope:** PhoneBurner / Intake only. This change assigns the role of a voice; it does
not repair misheard words, decide whether conduct breached an SOP, or send historical
calls again. The CRM and the Call Quality POST receive the same transcript string.

## Why the change is needed

In upstream commit `7032f84`, the Deepgram formatter labeled channel 0 Agent,
channel 1 Lead, and **any sole active channel Agent**. Once the formatter wrote a
business role, the generic resolver did not reconsider it. The generic resolver also
had a first-speaker fallback for diarized speaker labels. Neither convention
establishes who spoke.

## Decision policy

- Keep the physical channel and the voice on that channel distinct. For mono audio,
  a diarized voice can supply an identity; a stereo mapping cannot. An undiarized
  mono recording stays Unknown.
- Use an independently verified channel mapping, explicit speaker metadata, or
  corroborating first-person language. Do not identify the other party by elimination.
  An insult alone, a greeting or a quoted example does not identify its speaker.
- During the first 90 seconds of Intake calls, direct outreach on behalf of a
  legal organization together with a commitment to assist the listener with
  their case can establish the Intaker. The two cues may span up to three
  consecutive fragments on the same acoustic identity (at most 25 seconds).
  For example: “Te llamo de parte del equipo legal. Para ayudarte con tu
  accidente.” The role follows that identity, not that entire channel. No
  role is inherited by someone who only answers, or by a later new speaker.
  Repeated introductions on different identities, contradictory evidence,
  and unseparated speech do not qualify under this rule.
- A separate courtesy question after a completed introduction does not erase
  that introduction. The question itself supplies no role evidence. A firm
  calling on behalf of its lawyers may also explain its assistance in the
  next consecutive turn from the same acoustic identity.
- Separate a mixed utterance by its word speakers when the words reproduce the
  transcript. If speaker evidence conflicts and the quote cannot be split reliably,
  keep the original quote as Unknown. Keep the recording timestamp where available.
- Unknown is a valid result. Do not count it as a correct assignment or as a
  failure by an intaker. Log counts of unknown and total labeled turns, not speech.

## Offline acceptance before deployment

1. Build an AI-only reference from channel-separated audio on a held-out ES/EN
   sample: reversed stereo, mono, a silent channel, overlap, transfers, Lead-first
   openings, and misconduct. Keep the CRM labels and alert judgments out of
   attribution. Uncertain and conflicting evidence must abstain. Report agreement
   with this reference as AI agreement, not human-verified accuracy. Intake has
   no role-validation task.
2. Run both revisions on the same inputs without CRM writes or Slack. Measure
   **role precision** (correct assignments / assigned turns), **coverage** (assigned /
   reference-identifiable turns), and the number of calls with any Lead→Agent error.
   Report counts by language and layout. Check flagged conduct excerpts separately.
   A higher Unknown rate alone is not evidence of improved performance.
3. Require zero wrong-speaker attributions on the known critical excerpts. Compare
   precision and coverage with the baseline and agree on an acceptable coverage floor
   with Intake before release. A passing synthetic suite does not establish field accuracy.
4. Check the receiving Call Quality version with three payload fixtures: Lead first,
   one known and one unknown voice, and two unknown voices. Explicit roles must stay
   assigned; Unknown must never become an operator for coaching. The consumer source
   inspected at `9270be5` can infer an operator from two labels; do not release on
   the assumption that the POST alone protects that boundary.
   The companion receiver patch recognizes the optional field
   `role_attribution: phoneburner-evidence-v2` emitted by this producer. Deploy
   the compatible receiver **first** (old receivers reject extra fields); roll
   back the producer first. Incomplete attribution is a technical abstention,
   not a coaching alert or an Intake task.
5. Roll out only after these gates pass. Compare the aggregate Unknown-turn rate
   before and after rollout alongside disagreements with independent audio evidence. Do not
   replay the reported incident into the live alert workflow.

## Optional semantic fallback

Leave `PHONEBURNER_SEMANTIC_ROLES=false` until a blind holdout passes. When
enabled, only two separated identities with insufficient evidence can invoke
the existing project/region's Vertex model. A physical/metadata conflict,
third speaker, or unseparated input is not eligible. The model sees at most
44 bounded text turns with new aliases and common contact strings masked; no
audio, call ID, CRM role labels or contact metadata is sent. This is still
sensitive speech content, not anonymized data. Require two consistent votes
with different aliases, two distinct verbatim evidence turns and kinds for
each participant, plus compatibility with already-grounded roles. A timeout,
malformed result, uncertainty or disagreement leaves the original abstention.
Disable the flag to roll back this pass without changing deterministic rules.

### Vote envelope and multi-voice recordings

The vote reader accepts one JSON object, either bare or within one standalone
Markdown code fence (unlabelled or labelled `json`). It does not extract JSON
from surrounding prose, repair truncated objects, or select among multiple
objects or duplicate keys. Removing the fence is a transport-format change:
both votes must still independently pass the same exact-quote, turn, track
and evidence-kind checks and agree with each other and any established role.

If an eligible vote is rejected, the service emits one diagnostic such as
`semantic_vote_rejected vote=1 reason=unverifiable_quote`. The number is the
failed vote (1 or 2); the reason is the first failed check, not a manager
judgment. Model abstention, malformed JSON, non-independent evidence, role
conflict, disagreement, timeout and provider errors are distinguished. This
event contains no transcript, quotation, model reply, call ID or contact data.
Ineligible calls do not vote or emit a vote-rejection event.

A diarization ID identifies a segment group, not necessarily a person. Even
on one physical channel an A–B–A sequence could be one person split into two
IDs or a second person briefly speaking on the same phone. Turn length,
absence of overlap, similar words and the existing CRM role are not proof.
A handoff followed by a distinct introduction is a useful place to inspect
the recording, not an instruction to join or relabel tracks. To adjudicate a
case, align the disputed turns with independent, time-bounded participant
or isolated-source evidence; a call-level contact or participant count is
not enough. Without that evidence, preserve the tracks and any unresolved
role. The two-party semantic fallback must not run on a three-ID transcript.
