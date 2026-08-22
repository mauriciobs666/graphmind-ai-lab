# Delegate mid-run escalation — Coordination

> **Status:** active · **Owner:** `teco` · **Tracks:** — (—)

## Goal
Design and implement the requirements at `claude/docs/requirements/mid-run-escalation.md`
(Status: Ready for design, confirmed 2026-08-21, FR-1..FR-5 / AC-1..AC-5). Read that document in
full — do not rely on this coordination doc's paraphrase. In short: let a `teco`-coordinated
delegate stop mid-run on a genuinely high-stakes, undecided fork and return an open question as
its result, instead of guessing or only flagging it after the fact; `teco` relays the question to
the stakeholder and, once answered, resumes the *same* delegate via `SendMessage` (addressed by
its recorded `agentId`) rather than re-briefing cold. Routine ambiguity is unaffected — delegates
keep making the call themselves, same as today.

This is agent/prompt engineering (it edits `claude/teco/teco.md`'s brief-writing step and the
coordination-ledger conventions `teco` uses) — routes to `cobb`, not `architect`, per
`claude/AGENTS.md`'s "Agent/subagent/skill/prompt/hook engineering → cobb" routing and the
`agent-permission-friction` precedent (`claude/docs/plans/agent-permission-friction.md`, also a
`cobb`-authored plan touching only agent prompts/hooks). The requirements doc's own "the
architect's design" phrase in its Out-of-scope section is read as generic ("the next stage's
call, not this document's"), not a literal routing instruction — it names no specific agent
elsewhere and predates any routing decision.

## Scope notes for the design step
- **FR-1 (scope of escalation)** is the crux of the whole feature and needs a concrete, usable
  test in the design — not just "high-stakes fork," but something a delegate can actually apply
  mid-run: examples of qualifying vs. non-qualifying forks, phrased so it doesn't become "ask
  about everything." AC-1's language ("would change scope, touch something irreversible, or waste
  substantial downstream work if guessed wrong") is the requirements-level bar; the design step
  should turn that into concrete brief-template wording.
- **FR-3 (resume, not restart)** — mechanism is already proven (K-007, K-013,
  `claude/teco/kaizen/history.md`); the design step is choosing how `teco` represents "paused,
  waiting on an answer" in a coordination ledger row (new `Status` value or reuse of an existing
  one — the requirements doc explicitly leaves this open, FR-3/Out-of-scope) and how the relay +
  resume steps get written into `teco.md`'s process (§4 "Track what's in flight" is the most
  likely home, alongside the existing similar clauses on stale/superseded/misrouted results).
- **FR-4 (non-blocking)** — needs to be explicit that a paused unit doesn't stall dispatch of
  other independent, already-dispatched units — this is closely related to (and should stay
  consistent with) `teco.md`'s existing "never state or predict a pending delegate's result" and
  "a unit superseded while in flight" clauses in the same numbered list.
- **FR-5 (team-wide)** — the design should make explicit that this doesn't require touching any
  delegate's own tool grants (`SendMessage` or otherwise) — it's entirely a `teco`-side process
  change, since `teco` performs the resume, not the delegate.
- **Out of scope, restated:** no `SendMessage` grant changes for any specialist; no numeric cap on
  stop-and-ask round trips (left to `teco`'s judgment); no change to the existing
  "state reasoning and make the call" default for routine ambiguity.

## Ledger

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| U1 | `cobb` | `a18ec535eadd875b6` | delivered | `claude/docs/plans/mid-run-escalation.md` | `analyst` → — | 120546 tok / 12 tools |
| U2 | `analyst` | `a9d9383e694ee3645` | accepted | `claude/docs/reviews/mid-run-escalation.md` | self → approve with suggestions | 108124 tok / 27 tools |
| U3 | `cobb` | `a18ec535eadd875b6` | delivered | `claude/teco/teco.md`, `claude/README.md`, `claude/teco/kaizen/history.md`, plan-doc Finding-1 fix | `analyst` (diff-scoped) → — | 161063 tok / 18 tools |
| U4 | `analyst` | `a3cf78153918b0e4d` | accepted | diff-scoped re-check, `docs/reviews/mid-run-escalation.md` Pass 2 | self → approve with suggestions | 114037 tok / 15 tools |

## Close-out

Accepted 2026-08-22. Both gates passed with no blocker: Pass 1 (plan) and Pass 2 (diff-scoped,
fresh `analyst` instance, independent re-derivation of every AC) both verdict **approve with
suggestions**. Pass 2 confirmed all three Pass-1 minor/nit findings correctly folded in, `grep`-
confirmed Finding 1's false provenance claim fully removed, ran `claude/scripts/audit-team.sh`
(full PASS, no regression) as independent corroborating evidence, and confirmed scope discipline
via a whole-repo `git status`/`git diff --stat` — only `teco.md`, `README.md`,
`kaizen/history.md`, and this feature's own three doc-tree artifacts changed.

Two residual items, both non-blocking:
- Pass 2's own new Finding 1 (cosmetic): step 2's coordination-doc-trigger paragraph uses `paused`
  before the `Status` enum formally defines it later in the same step. Consistent with the
  document's existing forward-reference style elsewhere (`teco` verified this framing); left as-is,
  not worth a third round for a cosmetic ordering nit.
- Pass 2's evidentiary-gap note: `cobb`'s claim that plan §2.2 never carried the false-provenance
  claim (only §1.1 did) can't be independently confirmed since the plan doc was never committed
  (no "before" state to diff). Practically moot — the current file reads clean either way, which is
  what the gate actually checks.

`teco` committed the deliverables by explicit path after this acceptance (see repo git log for
this slug). Family archival flips (`requirements/`, `plans/`, `reviews/`, this coordination doc)
applied in the same close, per the standard milestone-close convention.

## U3 outcome — teco-verified before dispatching U4
- `teco.md` diff spot-checked directly (`git diff`): matches the plan's §2 with all four review
  findings folded in — `paused` status + full 7-column example row (Finding 4), Subagent-awareness
  clause scoped to skip mechanical dispatches (Finding 3), step-4 recognize/relay bullet spells out
  the two-hop `SendMessage` chain for teco-as-subagent (Finding 2).
- Plan-doc Finding 1 fix confirmed: `grep` shows no remaining "grounded in this repo... history" /
  "agent-permission-friction investigation" claim anywhere in `mid-run-escalation.md`; §1.1 now
  reads "illustrative example." `cobb`'s claim that §2.2 never had the false claim in the first
  place (only §1.1 did) checked out on inspection — no fabricated §2.2 edit needed.
- `README.md`/`kaizen/history.md` diffs read cleanly; the history entry states only what was
  actually verified this pass (a read-through against AC-1..AC-5, no hook/hook-frontmatter file
  touched) and explicitly flags a live dry-run as still a follow-up, not fabricated as done.
- Everything left uncommitted per U3's brief — `teco` commits after the U4 gate.

## Review outcome (U2, `analyst`)
**Verdict: approve with suggestions.** No blocker. One major finding (Finding 1): the plan's
worked qualifying example (NULL-backfill fork) is introduced with a false "grounded in this
repo's real history / agent-permission-friction investigation" provenance claim — `analyst`
grepped and found no such event; `agent-permission-friction` was pure hook/permission-guard
engineering, no migrations. The example itself is fine as a hypothetical; only the provenance
sentence in §1.1/§2.2 needs dropping. Three minor/nit suggestions: (2) the "teco-as-subagent"
resume path is actually a two-hop `SendMessage` chain the plan's §5 doesn't spell out; (3) the new
brief clause is unconditionally "every brief" unlike the adjacent Model-routing bullet's
mechanical/judgment scoping; (4) the `paused`-row example doesn't show a full 7-column row and
doesn't note it repurposes a normally path-typed `Deliverable` column. All four are folded into
U3 (implementation) rather than requiring a second design/review round.

## Notes
- No CPG applies — this is agent-prompt design work, not a review of CPG-bearing source.
- Once U1 is accepted, this ledger grows an implementation unit (likely `cobb` again, editing
  `claude/teco/teco.md` + `claude/teco/kaizen/history.md` + `claude/README.md` if the catalog
  description needs updating) and its own `analyst` diff-scoped re-gate, per the standard
  plan-gate + implementation-gate pattern.
