# Live specs — agent/prompt-safety gate on `analyst.md` (L3a)

> **Status:** active · **Owner:** `security-expert` · **Tracks:** —

## Pass 1

**Scope.** Lens 2 (agent/prompt-safety) plus a correctness gate on unit L3a's uncommitted edit to
`claude/analyst/analyst.md` and `claude/analyst/kaizen/history.md` (`git diff -- claude/analyst/`),
authored by `cobb`. Baseline: `docs/plans/live-specs.md` §3.1, §3.3, §3.4 step 4, §4 L3a row;
`docs/requirements/live-specs.md` FR-6, FR-9, AC-6; root `REVIEW.md` (read and applied);
`docs/reviews/live-specs.md` Pass 6 duplicate list and Pass 7 P7-1/P7-2; root `AGENTS.md`
feature-close bullet (l.165–173). No component `REVIEW.md` applies (the work touches `claude/`
only). Static review; no exploitation.

**Verdict: approve with suggestions.** The pointer satisfies FR-6/AC-6 for `analyst`, every
replacement maps to a real root `REVIEW.md` line, and nothing in the permission surface moved. Two
replacements are slightly narrower than the sentence they replaced (S1, S2); both are cheap to
restore and neither is a blocker.

### Checks run

- **Permissions** — verified. Every diff hunk starts at l.42 or later; the frontmatter (l.1–12:
  `tools:`, `permissionMode`, `hooks:` → `guard-review-doc-writes.sh`) is byte-identical to
  `HEAD`. `git diff --quiet HEAD -- claude/analyst/hooks .claude` reports no change. No tool grant,
  hook or write guard is widened.
- **Pointer** — verified. l.45 (step 2): "Before any review or gate verdict, read and apply root
  `REVIEW.md` and the `REVIEW.md` of every component the reviewed work touches." This is FR-6's
  wording (read *and apply*, root *and* touched components). The Guardrails line l.84 also invokes
  the evidence bar unconditionally, so RCA mode, which has no verdict, still gets it.
- **Six edits against P7-1** — traced line by line (table below). l.85 (the `git grep` bound) is
  correctly left unchanged, as P7-1 marks it partial.
- **Kaizen** — verified. The new entry sits directly under "Most recent first.", above the
  2026-10-29 entry. Its claims check out: `wc -w` 2760 → 2765 reproduces, and the description of
  each edit matches the diff.

| # | Old sentence | Root `REVIEW.md` line now cited | Behavior kept? |
|---|---|---|---|
| 1 | (new) step-2 pointer | whole file | Added, FR-6 wording |
| 2 | growth-stops "run its check, say whether any growth is new scope or rework" | *Always check* 1 (**Growth-stops check.**) | Yes, and wider: it adds "report rework growth, restated history and restated facts". `## Pass N` and Ready-for-design clauses stay. |
| 3 | guard placement "can never redden" | *Always check* 4 ("A check that cannot fail … is a finding") | Yes. The inserted clause hurts readability (S4). |
| 4 | "Distinguish what you verified … from what you infer" | *Evidence bar* 1 ("each finding") | **Narrower in scope** (S1) |
| 5 | "A fix you suggest is a claim, not a nit — run it" | *Evidence bar* 1 ("A fix you suggest is a claim: run it, and check its reach") | Yes, and wider (it adds reach). The local sentence is no longer imperative (S3). |
| 6 | "acceptance-check command is a claim too — run it verbatim before approving" | *Always check* 5, last sentence | **Imperative became conditional, and the cited line is in a sweep context** (S2) |

## Findings

**S1 (minor) — The evidence-bar pointer narrows "distinguish verified from inferred" from every
claim to findings only.** Verified by comparing the texts. The old l.84 lead said "Distinguish
what you verified (ran, traced, checked against docs) from what you infer". That covers every
claim in the review. Root `REVIEW.md` *Evidence bar* says "State whether each finding was
verified … or inferred". Step 3's "Every finding you report should survive … — say which" is also
finding-scoped. So the verdict's basis, *What's solid*, and *Checks run* claims (e.g. "byte-identical",
"suite green") no longer carry the duty explicitly. This is the class of claim an approve verdict
rests on. Replace the l.84 lead with:

```markdown
- **Evidence over vibes.** Root `REVIEW.md`'s evidence bar applies, and to every claim you make — the verdict's basis, *What's solid*, checks run — not only to findings; beyond it, never report a suite as green without running it, never claim a bug you didn't trace to a concrete path.
```

**S2 (minor) — The acceptance-command trap became conditional and points at a sweep-scoped
line.** Verified by reading. The old l.88 said "A plan's prescribed acceptance-check command is a
claim too — run it verbatim before approving". The new text says "When you run … verbatim (root
`REVIEW.md`), watch for …". The imperative now lives only in `REVIEW.md` *Always check* 5. That
bullet opens "A 'no X remains' check is an unfiltered sweep", so a reader can take its "Run a
prescribed check command verbatim" as limited to sweeps. The old line covered any acceptance
command, e.g. a `pytest` invocation or a script. The "before approving" timing is also gone. If
a later prune or rewording touches that sweep bullet, `analyst` loses the rule with no local
trace. Replace l.88 with:

```markdown
  - A plan's prescribed acceptance-check command is a claim too — run it verbatim, from the working directory it states, before approving (root `REVIEW.md`, *Always check*). Watch for a pasted doc-template placeholder (`kaizen_<agent>`, `cpg_<component>`): against the repo's expanded keys (`cpg_falkorchat`) it matches nothing, reporting a clean sweep while the real occurrences go untouched.
```

**S3 (nit) — The suggested-fix trap's new sentence reads as a description, not an instruction.**
"Running a suggested fix (root `REVIEW.md`'s evidence bar) covers a regex/glob/pattern …" can be
parsed as "running a fix is sufficient coverage". The intended reading is "the duty to run a fix
covers a regex and a design". Replace l.86's first sentence with:

```markdown
  - The evidence bar's "a fix you suggest is a claim: run it" (root `REVIEW.md`) covers a regex/glob/pattern and equally a *design* you propose.
```

**S4 (nit) — The inserted clause in the guard-placement sentence detaches "that" from its
referent.** In "…so it can never redden, which root `REVIEW.md` reports as a finding: that is
deleting the guard while keeping its name", "that" now reads as referring to the reporting.
Replace with:

```markdown
…so it can never redden: that is deleting the guard while keeping its name, and root `REVIEW.md` reports a check that cannot fail as a finding.
```

**S5 (advisory, indirection — not blocking) — `REVIEW.md` can steer `analyst` through
additions that narrow without being a prune or a *Do not report* line. The prompt also has no
precedence rule.** Inferred from plan §3.4 step 4 and root `AGENTS.md` l.170–171. Nothing was
exploited. The plan gives a closed definition of narrowing: "a prune or a *Do not report*
addition". Root `AGENTS.md` leaves "narrowing" undefined, and `teco` applies changes verbatim
without a gate. Ungated additions can silence or downgrade a finding class from the other three
headings:

- a severity cap under *What counts as serious*, e.g. "X is at most a nit";
- a scope limiter under *Always check*, e.g. "check Y only in plan reviews";
- an evidence relaxation under *Evidence bar*, e.g. "a doc-only change needs no run".

Every line still has to come from a review proposal with a stated source, and `teco` checks both
ways. That gives provenance and an audit trail, which is the main control. It is a sound one
against accidental drift. It is weaker against a poisoned proposal, e.g. a reviewer whose
reviewed artifact carried injected text. Separately, `analyst.md` says "read and apply" but does
not say which wins on conflict. A `REVIEW.md` line that contradicts a Guardrails line is
therefore ambiguous. Two suggestions, both for `cobb` to weigh:

1. Define narrowing by effect, in root `AGENTS.md`'s feature-close bullet, item (4). Add after
   "`cobb` gates a narrowing one": "(any change that removes, caps the severity of, limits the
   scope of, or relaxes the evidence for a check, under any heading)".
2. Add one precedence sentence to `analyst.md` step 2, after the pointer: "`REVIEW.md` adds to
   this prompt and never relaxes it: where a line would silence, downgrade or exempt a check this
   prompt requires, follow the prompt and report the conflict as a finding." The same sentence
   fits each L3a–L3c pointer.

## What's solid

- Every pointer names its target file, and the growth-stops pointer resolves to the bold
  **Growth-stops check.** lead that P7-2 asked for. Agent-specific remainders stay, as P7-1
  directs: the green-suite caveat, the placeholder example, the untracked-count bound, the
  `## Pass N`/Ready-for-design clauses and guard placement.
- Two replacements (2, 5) are strictly wider than what they replaced, because the `REVIEW.md`
  lines add duties (rework/restated-fact reporting, reach check).
- The permission surface is untouched, and the edit stays inside the prompt body.
- Net growth is +5 words for a new cross-file duty, so the edit is a net consolidation.

## Open questions

- For `cobb`: is the effect-based definition of narrowing (S5.1) wanted team-wide? If so, it
  belongs to L2c's text, which is already committed, so it would be a follow-up edit rather than
  part of L3a.

No `### Proposed REVIEW.md lines`: S1–S4 are prompt wording, and S5 is a coordination rule, not
reviewer behavior.
