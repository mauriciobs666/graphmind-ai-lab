# Delegate mid-run escalation — Design

> **Status:** active · **Owner:** `cobb` · **Tracks:** — (—)

**Component:** `claude/` · **Owner altitude:** cobb (design only, per `claude/AGENTS.md`'s routing
convention — this is agent/prompt engineering, not a codebase change) · **Reviewer:** `analyst`
(gate before implementation)
**Upstream:** `claude/docs/requirements/mid-run-escalation.md` (Status: Ready for design,
confirmed 2026-08-21)
**Goal:** let a `teco`-coordinated delegate stop mid-run on a genuinely high-stakes, undecided
fork and return an open question as its result (FR-1); `teco` relays it to the stakeholder (FR-2)
and, once answered, resumes the *same* delegate by its ledger-recorded `agentId` via `SendMessage`
(FR-3) rather than re-briefing a fresh delegate cold — without stalling the rest of an in-flight
coordination (FR-4), and available to every delegate `teco` coordinates regardless of that
delegate's own tool grants (FR-5).

This is a **design for implementation**, not the implementation. No file is edited in this unit —
every diff below is specified precisely enough to apply as-is. All changes are confined to
`claude/teco/teco.md` plus two catalog/context docs; **no other agent's file, no hook, and no
tool-grant frontmatter changes** (see §6, Scope).

---

## 1. The six open design questions, decided

### 1.1 FR-1's operational test — concrete brief wording, not just the AC-1 sentence

AC-1's test ("would change scope, touch something irreversible, or waste substantial downstream
work if guessed wrong") is stakeholder-approved language but not, on its own, something a delegate
mid-task can apply — it needs to arrive as *brief text with worked examples*, the way the existing
"mutation-test the green-on-arrival tests" and "state only figures you directly observed" clauses
in `teco.md` step 3 already do (prose the coordinator folds into every brief, not prose only teco
itself reads). §2.2 below gives the exact clause, including one qualifying and one non-qualifying
illustrative example (a hypothetical NOT-NULL-backfill decision, and a reversible naming choice)
so the line between "stop and ask" and "state reasoning and decide" is legible without becoming
"ask about everything."

### 1.2 Ledger representation — a new `Status` value, not a reused one + note column

**Decision: add `paused` to the ledger's `Status` enum** (`queued · in-flight · delivered · gated
· accepted · abandoned` → **`+ paused`**). The requirements doc explicitly leaves this open
("Any new ledger-status vocabulary... is the architect's design, not this document's" — Out of
scope) and explicitly uses "pause"/"paused" as the operative word throughout (FR-4's title, the
Intent, the user stories) — a same-named `Status` value is the least-surprising choice, not an
invented vocabulary.

**Rejected alternative: reuse `in-flight` plus a note.** `in-flight` currently means "dispatched,
the agent is actively working" — a coordinator scanning the ledger to answer "what's actually
moving right now" (exactly the scenario `teco.md` step 1 describes: reconciling a resumed
coordination's `in-flight` rows against `git log` and the working tree before doing anything else)
would have to also read a free-text note on every `in-flight` row to tell "actively running" from
"stalled on a human," which defeats the ledger's own stated purpose ("the ledger, not your context
window, is where a unit's state lives"). A distinct value is one grep-able word that means one
thing, and it lets step 4's own prose refer to "a `paused` unit" unambiguously without qualification.

**No new column.** The ledger table today is `Unit | Owner | Agent id | Status | Deliverable |
Gate → verdict | Cost`. Adding a `Question` column would touch every existing row/example
(`teco.md` step 2's own sample row, `claude/README.md`'s description of the ledger's five fields)
for a value that only exists on `paused` rows. Instead: **for a `paused` row, the existing
`Deliverable` column carries the question text and the relay date** (there is no deliverable yet
to put there) — e.g. `⏸ "does the migration backfill NULLs to 0 or error the row?" — relayed
2026-08-22`. This is the minimum-blast-radius option: it changes one enum and one usage note, not
the table's schema or every place that schema is described.

### 1.3 Where in `teco.md` — step 3 (the rule change) and step 4 (the relay/resume mechanics)

Confirmed by reading `teco.md` in full before drafting (not paraphrased): the "cannot ask
questions mid-run" line is step 3's **Subagent-awareness** bullet (`teco.md:71`, exact text: *"remind
each delegate it cannot ask questions mid-run: blockers, questions, and approval requests come
back as its deliverable"*) — that's where FR-1's scope carve-out belongs, since it's the
delegate-facing instruction. The relay-and-resume mechanics (FR-2/FR-3/FR-4) belong in step 4,
**Track what's in flight**, which already carries the closest-shaped existing clauses: "never
state or predict a pending delegate's result," the stale-placeholder-result check, the
transient-failure re-dispatch, the superseded-unit `abandoned` rule, and the misrouting-signal
rule. The new bullets are written to match that list's existing register (bold lead phrase,
mechanism, then rationale/cross-reference) — see §2.3 for the exact text.

### 1.4 FR-4's non-blocking guarantee — made explicit, not implied

Stated as its own bullet in step 4 (§2.3): a `paused` unit stalls only itself and any unit that
*structurally* depends on its output — every other independent, already-dispatched unit keeps
being dispatched and progressed exactly as if the paused unit weren't there. This also states the
requirements doc's own silence explicitly: **no cap, no deadline, no auto-escalation** — a unit
can sit `paused` indefinitely if the stakeholder hasn't answered, which is a deliberate absence
(Out of scope), not a gap this design fills in.

### 1.5 FR-5's team-wide scope — confirmed by fact-check, zero grant changes needed

Fact-check (`grep -n "^tools:" claude/*/[a-z]*.md`, run this session): of the 13 agents, exactly
**6** carry an explicit `tools:` allow-list in frontmatter — `analyst`, `architect`,
`data-scientist`, `security-expert`, `tico`, and `teco`. Of those six, only **`teco`'s** list
includes `SendMessage`; the other five (`analyst`, `architect`, `data-scientist`,
`security-expert`, `tico`) do not carry it. The remaining **7** agents (`coder`, `tdd-engineer`,
`frontend-engineer`, `devops`, `graph-dba`, `qa-engineer`, `cobb`) have no `tools:` line at all —
full default tool access, which may include `SendMessage` implicitly. This exactly matches the
requirements doc's own list of the four specialists it names as not carrying `SendMessage`
(`analyst`, `architect`, `data-scientist`, `security-expert` — Problem & current state) plus
`tico`, which is moot here since `tico` is not a `teco` delegation target at all (`teco.md`
routing table: "not a delegation target").

This confirms the mechanism needs **zero** tool-grant changes: the delegate never calls
`SendMessage` itself in this design — it only ever stops its run and returns the question as its
*result* (an ordinary subagent completion, no new tool involved); `teco` is the sole `SendMessage`
caller, exactly as today. FR-5's team-wide scope falls out of that mechanism for free, which is
also why the requirements doc frames it as settled rather than open (Problem & current state,
final paragraph) — this fact-check is a verification of that framing, not a new decision.

### 1.6 Explicit non-goals (restated for this plan's own scope)

See §6.

---

## 2. The `teco.md` diff

Three edits, all inside `claude/teco/teco.md`. Quoted current text is verified against the file as
read this session (line numbers as of that read).

### 2.1 Step 2 — ledger `Status` enum and coordination-doc trigger

Current (`teco.md:68`):

> `Status` is one of `queued` · `in-flight` · `delivered` · `gated` · `accepted` · `abandoned`.

Replace with:

> `Status` is one of `queued` · `in-flight` · `delivered` · `gated` · `accepted` · `abandoned` ·
> **`paused`** (added by `mid-run-escalation` — a unit whose delegate stopped mid-run with an open
> question, now relayed to the stakeholder and awaiting an answer; see step 4). For a `paused`
> row, the **Deliverable** column carries the open question and the date it was relayed instead of
> a deliverable path (none exists yet) — e.g. `⏸ "does the migration backfill NULLs to 0 or error
> the row?" — relayed 2026-08-22`.

Current (`teco.md:62`):

> **Open a coordination doc** — `<component>/docs/plans/<slug>-coordination.md`, co-located with
> the architect's plan — whenever the work has **three or more units, or any unit carries a review
> gate**; below that threshold, hold the plan in your report.

Replace with:

> **Open a coordination doc** — `<component>/docs/plans/<slug>-coordination.md`, co-located with
> the architect's plan — whenever the work has **three or more units, any unit carries a review
> gate, or any unit escalates via the stop-and-ask path (step 4)**; below that threshold, hold the
> plan in your report. The stop-and-ask trigger is necessarily **reactive**, not known at
> decomposition time: if a delegate returns an open question under a coordination that hasn't
> crossed the other two thresholds and has no coordination doc yet, open one now — backfilling a
> ledger row for every unit dispatched so far — before marking anything `paused`. A paused unit's
> `agentId` and open question must survive a compaction or an interrupted session, and only the
> ledger, not your context window, guarantees that.

### 2.2 Step 3 — Subagent-awareness bullet

Current (`teco.md:71`):

> - **Subagent-awareness** — remind each delegate it cannot ask questions mid-run: blockers,
>   questions, and approval requests come back as its deliverable.

Replace with:

> - **Subagent-awareness** — remind each delegate that, for **routine ambiguity**, it still can't
>   ask questions mid-run: it states its reasoning and makes the call itself, same as always —
>   blockers, questions, and approval requests otherwise come back as its deliverable. **One
>   narrow exception (`mid-run-escalation`):** a **high-stakes fork** — a decision that, if guessed
>   wrong, would change scope, touch something irreversible, or waste substantial downstream work
>   — may be stopped on mid-run instead, with the open question returned as the unit's *result*
>   rather than folded into a guess or held for the final report. Fold this into every brief close
>   to verbatim:
>
>   > If, mid-run, you hit a decision where guessing wrong would change scope, touch something
>   > irreversible (e.g. a schema change already applied against real data, a destructive
>   > migration, a public interface once other code depends on it), or waste substantial downstream
>   > work if reversed — stop here and return the specific question as your result, instead of
>   > guessing or only noting it in a final report. Say what you've done so far, the fork, your
>   > options, and your own recommendation if you have one — you'll be resumed with the answer to
>   > continue. Qualifying example: a migration script must decide whether backfilling a NULL
>   > column defaults it to 0 or errors the row — wrong on production data is expensive to unwind.
>   > Non-qualifying example: which of two equally-idiomatic names to use for a new helper function
>   > — reversible by a rename, no scope or data impact; make the call yourself and say why, same
>   > as any other routine judgment call.
>
>   This is a **narrow, additional** path, not a replacement for stating reasoning and deciding —
>   treat genuine uncertainty about which side of the line a fork falls on as routine (decide,
>   state why) rather than defaulting to asking; over-firing this defeats its purpose.

### 2.3 Step 4 — new bullets (append to the existing list, after the misrouting-signal bullet)

Insert, as new bullets at the end of step 4's list (after `teco.md:89`):

> - **A delegate's result that is an open question, not a deliverable, is a `paused` unit, not a
>   `delivered` one.** Recognize it by shape: it states a specific fork, the delegate's own
>   recommendation if it has one, and asks to be resumed — not a completed deliverable with an
>   unrelated caveat attached. Mark the unit `paused` in the ledger (question + relay date in the
>   Deliverable column, per step 2) and relay the question to the stakeholder — first-order, via
>   `AskUserQuestion` exactly as any other decision point (options plus your recommendation); as a
>   subagent, the harness withholds that tool the same way it does everywhere else in this document
>   (see "Pause vs. proceed") — return the question in your own report instead, which nests the
>   same stop-and-ask shape one level up rather than requiring new machinery. **Never answer it
>   yourself or silently decide on the delegate's behalf** — same principle as never stating or
>   predicting a pending delegate's result, above.
> - **Once the stakeholder answers, resume the *same* delegate** — `SendMessage` addressed by the
>   `agentId` recorded in its ledger row at dispatch, folding the answer into the message, never a
>   fresh `Agent` call re-briefing it cold. Flip the unit back to `in-flight` on send; it becomes
>   `delivered` when the (now-unblocked) actual deliverable arrives, same as any other unit. If the
>   `SendMessage` fails to resolve, the existing fallback in step 5 applies here too — attempt the
>   send first and treat a resolution failure, not surprise, as the signal to fall back to a fresh
>   `Agent` call.
> - **A `paused` unit stalls only itself.** Every other independent, already-dispatched unit in the
>   same coordination keeps moving — don't hold dispatch of unrelated units, and don't treat the
>   paused unit as blocking anything except a unit that structurally depends on its output. If the
>   stakeholder simply hasn't answered yet, the unit sits `paused` indefinitely — there's no cap,
>   deadline, or auto-escalation; that's a deliberate absence, not an oversight, and matches every
>   other in-run judgment call left to you.
> - **No fixed cap on stop-and-ask round trips per unit** — including for a unit that pauses more
>   than once. Use judgment the same as anywhere else in this document: if a unit keeps re-pausing
>   on what looks like the same underlying disagreement, that pattern itself is worth surfacing to
>   the stakeholder as the actual problem, rather than relaying yet another narrow question.

No other section of `teco.md` changes. In particular: **no hook file, no frontmatter `hooks:`
block, and no `tools:` grant changes anywhere** (§1.5, §6) — this is a pure prompt-text diff.

---

## 3. Sequencing (for the implementation unit — not performed in this pass)

1. Apply the three edits in §2 to `claude/teco/teco.md` (a single-file diff — no dependency
   ordering between them, but step 3's clause should land before step 4's, since step 4's bullets
   presuppose the reader already knows what a "stop-and-ask" result looks like).
2. Update `claude/README.md`'s `teco` catalog entry (§4) in the same change.
3. Confirm `claude/AGENTS.md` needs **no** edit — its "Hook machinery" and tool-roster sections
   describe hooks and frontmatter grants, neither of which this feature touches; note this
   explicitly in the implementation unit's done-condition so a reviewer doesn't go looking for a
   missing hook change.
4. Manual verification pass — no automated suite covers prompt text, so verification is a
   read-through against each AC (§7) plus, ideally, one live dry run: a small `teco`-coordinated
   task where a delegate is briefed to intentionally hit a qualifying fork, to confirm the ledger
   row, relay, and `SendMessage` resume all work end-to-end before trusting the mechanism on real
   work. This mirrors how `SendMessage` itself was only trusted after `K-013`'s live exercise
   (requirements doc, Intent).
5. `claude/teco/kaizen/history.md` — dated entry: what changed (the three `teco.md` edits, the new
   `paused` ledger status) and why (`mid-run-escalation` FR-1..FR-5). **Not performed in this
   design pass** — flagged here for the implementation unit, per this plan's own brief.
6. Hand to `analyst` for review before the `teco.md` edit lands (standard gate, same as any
   `cobb`-authored change to a live agent prompt).

---

## 4. Doc-impact list

| Doc | Change | When |
|---|---|---|
| `claude/teco/teco.md` | The three edits in §2 | Implementation unit |
| `claude/README.md` | `teco` catalog entry: add one clause describing the new capability user-facingly. Insertion point — right after the existing sentence ending *"...the `agentId` is what `SendMessage` addresses to continue a delegate instead of respawning it cold."* Proposed text: *"A delegate facing a genuinely high-stakes, undecided fork (would change scope, touch something irreversible, or waste substantial downstream work if guessed wrong) can stop mid-run and return the question as its result instead of guessing — teco relays it to the stakeholder and resumes the same delegate via `SendMessage` once answered, marking the unit `paused` in the ledger meanwhile; routine ambiguity is unaffected, still resolved by the delegate itself (`docs/plans/mid-run-escalation.md`)."* | Implementation unit |
| `claude/AGENTS.md` | **No change.** No hook, no tool-grant change — the "Hook machinery" and roster sections stay accurate as-is. Stated here explicitly so the implementation unit's reviewer doesn't treat its absence as a missed update. | — |
| `claude/teco/kaizen/history.md` | Dated entry once implemented (see §3 step 5) | Implementation unit |
| `claude/docs/requirements/mid-run-escalation.md` | No change — this design doesn't revise the requirements | — |

---

## 5. Risks and edge cases

- **A paused unit that never gets answered.** Explicitly acceptable per the requirements doc's own
  Out-of-scope (no cap) and this design's §2.3 bullet — the coordination doc simply carries a
  `paused` row indefinitely. Not a defect to fix; a reviewer should not flag this as a gap.
- **`teco` itself running as a delegated subagent.** The requirements doc's user stories implicitly
  assume a first-order (`claude --agent teco`) `teco` that can hold a live back-and-forth with the
  stakeholder via `AskUserQuestion`. When `teco` itself is dispatched as a subagent (rare, but the
  existing "Pause vs. proceed" section already accounts for this mode), it inherits the same
  no-mid-run-questions constraint any other delegate has — so a paused inner delegate's question
  necessarily surfaces as **`teco`'s own** stop-and-ask result to whatever dispatched it, one level
  up. §2.3's first new bullet states this directly (the subagent branch: "return the question in
  your own report instead"). This is a structurally consistent extension of the same mechanism,
  not a special case needing new machinery — flagged here so the analyst review doesn't treat it as
  an unaddressed gap.
- **Misclassifying a normal deliverable-with-a-caveat as a pause.** A delegate's final report can
  legitimately raise open questions as *follow-ups* without stopping mid-run (that's today's
  existing "flag it in the final report" path, unaffected by this feature). §2.3's first bullet
  gives a shape test (asks to be resumed, no completed deliverable) precisely to keep these
  distinguishable; genuine ambiguity here is a judgment call for `teco`, same register as the rest
  of step 4.
- **Over-firing (every delegate starts pinging back on everything).** The requirements doc names
  this directly as a user story `teco` itself holds ("a clear line... so the loop doesn't turn into
  every delegate pinging back on everything it isn't 100% sure of"). Mitigated by the worked
  qualifying/non-qualifying examples in §2.2's brief clause, not by a mechanical cap (deliberately
  out of scope) — this is a prompt-quality risk, not something a rule can fully close; worth
  watching via `kaizen_team` entries if it happens in practice, per the standing learning-capture
  convention.
- **`SendMessage` resume failing to resolve the `agentId`.** Not a new risk — `teco.md` step 5
  already has this exact fallback ("attempt the `SendMessage` first and treat an addressing error
  as the non-resolution signal... fall back to a fresh `Agent` call only when the id no longer
  resolves"). §2.3's second bullet cross-references it rather than duplicating the logic.

---

## 6. Scope (restated non-goals)

- **No `SendMessage` tool-grant changes for any agent** — confirmed unnecessary by the §1.5
  fact-check; the mechanism only ever has `teco` call `SendMessage`, exactly as today.
- **No numeric cap on stop-and-ask round trips per unit** — left to `teco`'s judgment per the
  requirements doc's Out-of-scope; this design does not invent one (§2.3, §5).
- **No change to the existing "state reasoning and make the call" default** for routine ambiguity —
  §2.2's replacement bullet states this is a narrow *addition*, not a replacement, in its own text.
- **Standalone (non-`teco`-coordinated) agent runs are unaffected** — there is no `teco` in the
  loop to relay a question when a specialist runs directly (e.g. `claude --agent architect`); this
  design touches only `teco.md`, so nothing changes for a directly-run specialist by construction.
- **No hook or frontmatter tool-grant changes anywhere** — this is a pure prompt-text diff to one
  file plus two catalog/context docs (§4).

---

## 7. Acceptance-criteria mapping

- **AC-1 (FR-1, scope):** §2.2's replacement Subagent-awareness bullet gives the delegate-facing
  operational test plus one qualifying and one non-qualifying worked example; the "routine
  ambiguity... same as always" clause preserves the unchanged default path in the same bullet.
- **AC-2 (FR-2, relay):** §2.3's first new bullet — `teco` marks the unit `paused` and relays to
  the stakeholder (via `AskUserQuestion` first-order, or its own report as a subagent) rather than
  answering or deciding itself.
- **AC-3 (FR-3, resume not restart):** §2.3's second new bullet — `SendMessage` addressed by the
  ledger-recorded `agentId`, answer folded in, never a fresh `Agent` call.
- **AC-4 (FR-4, non-blocking):** §2.3's third new bullet, explicit — a `paused` unit stalls only
  itself and its structural dependents; every other independent unit keeps moving.
- **AC-5 (FR-5, team-wide):** §1.5's fact-check confirms the mechanism needs no delegate-side
  `SendMessage` grant, so the path is available identically to all 13 agents `teco` coordinates,
  independent of each one's own tool grants — nothing in §2's diff is agent-specific.

---

## 8. Residual open question (not resolved by the requirements doc, flagged rather than guessed)

None found that block implementation. The one genuine judgment call this design makes on the
requirements doc's behalf — §1.2's ledger-vocabulary choice and §2.1's reactive coordination-doc
trigger — is exactly the class of "implementation detail" the requirements doc's Out-of-scope
section assigns to this design step, not an unresolved requirement. If the `analyst` review
disagrees with either choice, both are isolated (one enum value; one added trigger clause) and
cheap to revise without touching the rest of the diff.
