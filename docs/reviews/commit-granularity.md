# Commit granularity — review

> **Status:** active · **Owner:** `analyst` · **Tracks:** — (M-)

## Scope & verdict

Reviewed: commit `320f682` ("feat(agent-standards): enforce no-footer via settings deny; batch
docs-only chain commits", 2026-09-13), **restricted to the docs-only-chain commit-batching change
only** — `git show 320f682 -- claude/AGENTS.md claude/teco/teco.md claude/tico/tico.md`. The
bundled no-attribution-footer/`settings.json` change is out of scope by instruction and not
assessed here. Baseline: `docs/requirements/commit-granularity.md` (FR-1…FR-4, AC-1…AC-5).
Confirmed via `git log --oneline 320f682..HEAD -- claude/AGENTS.md claude/teco/teco.md
claude/tico/tico.md` that no later commit has touched any of the three files, so the current
working-tree text (except one unrelated uncommitted hunk in `claude/AGENTS.md`, see Appendix) is
exactly the text under review. Also read `claude/cobb/kaizen/history.md`'s 2026-09-13 "Docs-only
coordination chains…" entry and the linked `K-033` backlog item for `cobb`'s own account and
verified its factual/quantitative claims independently rather than taking them at face value.

**Verdict: needs changes.** The requirements' FR/AC list is faithfully and correctly worded into
`claude/AGENTS.md` and into `teco.md`'s two edited passages, and the two self-corrections `cobb`
logged (narrative bleed into `claude/AGENTS.md`, a stray "(unchanged)" in `teco.md`) both verifiably
landed clean. But the edit left one real, evidenced contradiction in `tico.md` between its
unmodified Mode-1 commit rule and the newly-batched chain rule, for the exact document type
(requirements) FR-1's own example chain opens with — that's the blocker. A second, lower-severity
tension in `teco.md`'s pre-existing recovery guidance, and a materially inaccurate self-reported
word-count claim in `cobb`'s own kaizen tracking, round out the findings.

CPG: not applicable — this is a prompt/documentation-text review with no code-level component;
the change under review is exclusively Markdown prose in agent definition files.

## Findings

### Blocker — `tico.md`'s Mode-1 "Commit at document boundaries" rule contradicts the newly-batched docs-only-chain rule for the requirements document itself

`claude/tico/tico.md:68` (unmodified by this diff except for one unrelated exception clause) still
states: "Commit that document's file only when you're actually leaving it — switching to a
different document, stepping away from Mode 1 into Mode 2/3, or **the interview/session
closing**." `claude/tico/tico.md:58` (edited by this diff) now says the opposite for a
docs-only coordination: "Hold every verified deliverable uncommitted as the chain proceeds…
A chain paused mid-way (e.g. awaiting a review) **across a session boundary leaves the tree
dirty**… expected, not a sign anything was missed."

The "Coordinating a docs-only chain" section's own worked example (`tico.md:54`) is "**your own
requirements interview** → `architect`'s plan → `analyst`'s review of that plan → a revision
round" — i.e., the Mode-1-authored requirements document is explicitly unit 1 of the chain FR-1
batches. Nothing in either passage states which rule wins when they collide, and before this diff
they didn't collide (the old Integrating bullet was itself "commit each verified unit's
deliverable… once confirmed" — i.e., immediate, same direction as Mode-1's session-close trigger).
This diff flipped the chain rule to batched-at-terminal-state but left Mode-1's own document rule,
including its explicit session-close trigger, untouched — so a session that closes after just the
Mode-1 interview, before `architect`/`analyst` have run, now has two directly conflicting
instructions for the same file: commit it now (Mode 1) vs. leave the tree dirty (Coordinating).
Following the unmodified Mode-1 rule literally violates AC-2 for this scenario.

**Suggested fix:** add the same kind of carve-out already added for the review-consult exception
at `tico.md:68` — e.g. "except when this document is a unit of an active docs-only coordination
(see 'Coordinating a docs-only chain') — then it holds uncommitted with the rest of the chain
until the chain's terminal state, not at this document's own boundary." Land it in the same pass
that touches the Integrating bullet, since both bullets govern the same file.

### Major — `teco.md`'s pre-existing concurrent-churn recovery rule is not reconciled with the new docs-only batching rule

`claude/teco/teco.md:110` (pre-existing, untouched by this diff): "A delegate's edits can vanish
mid-run with no platform failure… Recovery is the same discipline as a platform kill… and
**commit by explicit path promptly once concurrent churn is visible** in `git status`, rather than
leaving verified work uncommitted." This bullet predates and was not touched by the new
docs-only-chain batching rule (`teco.md:131`, "hold every verified deliverable uncommitted
instead… commit them all together… once the whole chain reaches its terminal state"). If the
at-risk, just-recovered unit belongs to a docs-only chain, the two bullets now point opposite
ways — commit promptly to protect against loss, vs. hold until terminal state — and neither states
which governs. This is exactly the kind of race scenario `teco.md` elsewhere (the three-way
baseline/`HEAD`/worktree diff, `teco.md:155`) treats as a real, recurring operational hazard, not
a hypothetical.

**Suggested fix:** add one clause to `teco.md:110` (or a cross-reference from `teco.md:131`)
stating that the concurrent-churn-recovery commit is an explicit exception to docs-only batching —
protecting already-verified work from a second loss outranks the batching preference.

### Major — `cobb`'s own K-033 word-count verification claim materially understates the actual growth this commit produced

`claude/cobb/kaizen/plan.md` K-033 states: "`claude/AGENTS.md` moved 3,041→3,074 words (+33 net…)
`teco.md` and `tico.md` also grew (~450, ~250 words)". Measured directly against git
(`git show 320f682^:<path> | wc -w` vs. `git show 320f682:<path> | wc -w`):

| File | Claimed before→after (delta) | Actual before→after (delta) |
|---|---|---|
| `claude/AGENTS.md` | 3,041→3,074 (+33) | 2,783→3,074 (**+291**) |
| `claude/teco/teco.md` | ~450 | 10,765→10,899 (**+134**) |
| `claude/tico/tico.md` | ~250 | 5,623→5,809 (**+186**, closer) |

`claude/AGENTS.md`'s real net growth from this commit is ~9× the claimed figure, and it now sits
at 3,074 words — 574 over root `AGENTS.md`'s own ~2,500-word smell line, not the near-negligible
overage K-033 implies. `teco.md`'s claimed growth is ~3× the actual. This matters because K-033
sets the item's priority ("low… folds into whichever of K-030/K-031/K-032 executes first") based
on these numbers — the real figures argue for reassessing that priority, particularly for
`claude/AGENTS.md`, sooner rather than folding it into a later cleanup. Separately, the same
history entry's "confirmed… line-length budget still clean (`awk length>700`, zero hits)" claim
for `teco.md` doesn't hold literally (33 lines in `teco.md`, including the two lines this diff
edited, exceed 700 chars) — though root `AGENTS.md`'s 700-char bar is textually scoped to
`git ls-files '*AGENTS.md'` only, so `teco.md` was never actually in scope for that check and this
is a citation mix-up rather than a real violation.

**Suggested fix:** correct K-033's numbers in `claude/cobb/kaizen/plan.md` against the figures
above and re-assess its priority; drop or scope the `teco.md` line-length claim to what the
700-char bar actually covers.

### Minor — "terminal state" isn't mapped to the ledger's actual `Status` vocabulary

Both `claude/AGENTS.md`'s new paragraph and `tico.md`/`teco.md`'s edited bullets use "terminal
state (accepted / gated-closed / handed off)" — carried faithfully from the requirements doc's own
wording (AC-1) — but the ledger's actual `Status` enum (`teco.md:85`: `queued · in-flight ·
delivered · gated · accepted · abandoned · paused`) has no `closed` value, and `gated` is
documented there as a pre-`accepted` intermediate state, not necessarily final. An implementer
reading only the prompt text has to infer which concrete row value(s) count as "terminal" for
triggering the batched commit. This ambiguity originates in the requirements doc, not in `cobb`'s
wording choices, so it's a minor/nit rather than a defect in this diff specifically.

**Suggested fix:** one clause in `teco.md`/`tico.md` tying "terminal state" to the concrete
`Status` values it means (e.g., "a unit's row reads `accepted`, or `gated` with no further
revision round scheduled").

## What's solid

- FR-1/AC-1, FR-2/AC-4, FR-3/AC-2, FR-4, and AC-3 are all stated in `claude/AGENTS.md`'s new
  paragraph in wording that tracks the requirements doc closely, without contradiction elsewhere
  in that file (`awk`-verified: no leftover "per round"/"per gate round" phrasing survives
  anywhere in the three files after this edit).
- AC-5 (no retroactive touching of already-made commits) is correctly not implied anywhere in the
  new text — the rule is stated as forward-only, matching the requirements doc's decision log.
- `teco.md`'s "Commit what you verified" bullet and "The grant" bullet are internally consistent
  with each other on the code-implementation/docs-only branch (both use the same two-way split,
  same terminology).
- Both self-corrections `cobb`'s kaizen entry claims making mid-flight verifiably landed in the
  committed text: the no-footer paragraph cites `claude/cobb/kaizen/history.md, 2026-09-13` in one
  clause rather than restating the incident narrative (root `AGENTS.md`'s "history is not context"
  rule, respected), and no "(unchanged)" parenthetical or other diff-review commentary survives in
  `teco.md`'s edited bullets.
- `claude/AGENTS.md` itself has zero lines over 700 chars (verified directly), so the one
  line-length claim that rule actually covers holds.

## Open questions

- Does the `tico`/`teco` team want the Blocker finding's fix folded into a fast follow-up commit,
  or held until the next time either file is touched for an unrelated reason? Given it's a
  documentation/prompt-only fix with no code-level component, either is safe; I have no basis to
  prefer one over the other.

## Pass 2 — 2026-09-16

**Re-checked by executing against the current tree** (not re-argued): `git diff HEAD --
claude/tico/tico.md claude/teco/teco.md` for the two still-uncommitted fixes, `git diff 320f682
HEAD -- claude/AGENTS.md` for the fix that landed swept into an unrelated commit, `git diff HEAD
-- claude/cobb/kaizen/plan.md` for the K-033 correction, and an independent third re-run of
`git show 320f682^:<path> | wc -w` vs. `320f682:<path>` for all three files (matches both my Pass
1 numbers and the coordinator's independent third run exactly). Also re-ran the "no leftover
per-round phrasing" and "zero lines >700 in `claude/AGENTS.md`" checks against the post-fix text —
both still clean.

**Verdict: approve with suggestions.** All four Pass-1 findings are fixed and independently
confirmed against source. Checking the fixes against each other (not just each against its own
finding) surfaces two new, low-stakes items — no new blocker or major.

**Disposition of Pass-1 findings:**
- Blocker (`tico.md` Mode-1/batching contradiction) — **fixed.** `claude/tico/tico.md:68` now
  reads "except when this document is a unit of an active docs-only coordination you are running…
  this document's own boundary is not a commit trigger while the chain is open." Confirmed present
  in the uncommitted working-tree diff; resolves the contradiction with `tico.md:58` for exactly
  the scenario the finding named (session closing mid-chain, before `architect`/`analyst` run).
- Major (`teco.md` concurrent-churn vs. batching) — **fixed.** `claude/teco/teco.md`'s
  concurrent-churn bullet now states "this promptness is an explicit exception to the docs-only-
  chain batching rule… protecting it outranks the batching preference. The rest of the chain still
  batches normally." Confirmed present in the uncommitted working-tree diff; resolves the
  unaddressed tension identified in Pass 1.
- Major (K-033 word-count claim) — **fixed.** `claude/cobb/kaizen/plan.md` K-033 now states the
  same figures I re-verified in Pass 1 (`claude/AGENTS.md` +291, `teco.md` +134, `tico.md` +186)
  and I independently re-ran `wc -w` against both git refs a third time, matching exactly; priority
  raised low→medium with a stated rationale. Also corrected the `teco.md` line-length citation
  mix-up I'd noted only as a parenthetical aside — now explicit in the plan entry.
- Minor (terminal-state mapping) — **fixed.** `claude/AGENTS.md`'s docs-only-batching paragraph
  gained the mapping clause ("accepted, or gated with no further revision round scheduled…");
  confirmed present in `git diff 320f682 HEAD -- claude/AGENTS.md` even though it landed via a
  concurrent, unrelated commit (`702704f`) rather than this session's own — verified the content
  itself is correct and complete regardless of which commit message carries it.

**New — Minor: `tico.md:68`'s new exception clause has an ambiguous scope reach.** The fix inserts
"except when this document is a unit of an active docs-only coordination…" directly after "or
whenever the stakeholder asks you to commit, which is always a valid trigger wherever you are in
the work" and before the sentence's end. A literal parse could read the exception as also
suspending the stakeholder's explicit ask-to-commit override while a chain is open — which no FR/AC
calls for (FR-3/AC-2 only bar an *involuntary* interim commit, never an explicit human request).
The clause's own closing words ("this document's own **boundary** is not a commit trigger") argue
for the narrower, correct reading — it scopes to the boundary-trigger list, not the separate
stakeholder-ask trigger — but the sentence is dense enough that this took a second, careful read to
resolve. **Suggested fix:** split the stakeholder-ask sentence out on its own (an explicit "this
override still applies even during an open coordination" clause) rather than leaving the exception
positioned to plausibly swallow it.

**New — Minor: `teco.md`'s Guardrails "Grant" bullet still states an unqualified single-commit
rule for docs-only chains, now stale relative to the concurrent-churn exception.** `teco.md:155`
("The grant") reads "For a purely docs-only chain, the opposite: every constituent unit's files
land together in the chain's single terminal commit" with no mention of the just-added exception
in the step-5 "Commit what you verified" bullet it cross-references. An implementer who consults
only the Guardrails/authority-scope bullet (which is where "what may I actually commit" questions
naturally land) would not learn that a mid-chain emergency commit for one at-risk unit is
permitted. Low stakes because the operative timing instruction (step 5) already states the
exception clearly and the two bullets are meant to be read together — but it's a real completeness
gap the concurrent-churn fix didn't fully close. **Suggested fix:** one clause in `teco.md:155`
cross-referencing the concurrent-churn exception, mirroring how it already cross-references step 5
for the rest of the rule.

**Open question raised by the coordinator — was declining a `docs/HISTORY.md` entry the right
call?** Checked: repo-root `docs/HISTORY.md` states its own scope in its header ("Change History —
CPG code-graph component… Joern → FalkorDB"), and root `AGENTS.md`'s module-documentation
convention routes agent-prompt changes to each agent's own `kaizen/history.md`, not a
component-level `docs/HISTORY.md` (`claude/` has no such file — by design, per that convention:
"Modules do not use `kaizen/` dirs — that's for agent folders only"). This delivery is a
`claude/`-scoped prompt/policy change with no CPG relevance. **Cobb's call was correct** — there is
no applicable `docs/HISTORY.md` for this delivery to enter, and `claude/cobb/kaizen/history.md`'s
2026-09-13/2026-09-16 entries already carry its complete record.

## Appendix — unrelated uncommitted hunk noted during grounding

`claude/AGENTS.md` carries one uncommitted, unrelated hunk at the time of this review (adding a
`frontend-quirks.md` knowledge-base mention to the `frontend-engineer` roster line, per the
session's git status). It does not touch the "Git-commit authority" section and was excluded from
every measurement and quote above (all git commands used explicit `320f682`/`320f682^` refs, not
the working tree, except the line-length check on `claude/AGENTS.md` which is unaffected since the
uncommitted hunk adds no line near the 700-char threshold).
