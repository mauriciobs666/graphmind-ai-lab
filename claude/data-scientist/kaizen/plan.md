# Kaizen — Improvement Plan: data-scientist

> Forward-looking backlog for the `data-scientist` agent.
> Status: 🔵 proposed · 🟡 in-progress · ✅ done (then moved to history.md) · ⚪ rejected/deferred
> Last reviewed: 2026-09-07

## Active

| ID | Added | Priority | Status | Summary |
|------|------------|----------|--------|---------|
| K-001 | 2026-07-09 | med | 🔵 | First-run shakedown: a real method note + a real methodology review |
| K-002 | 2026-07-09 | low | 🔵 | Perishable model/embedding landscape reference (skill or resource file) |

### K-001 — First-run shakedown: a real method note + a real methodology review
- **Status:** 🔵 proposed
- **Priority:** medium
- **Rationale:** The prompt is untested against a live run. Likely weak spots: whether the method note stays at method altitude (vs. drifting into the architect's sequencing), whether every recommendation actually ships with an evaluation design, and whether the `-ml.md` naming + hook behave as intended in both doc homes (`docs/plans/`, `docs/reviews/`).
- **Proposed change:** Delegate (a) a real method question from this lab — e.g. an embedding/chunking strategy or retrieval-eval design for `falkor-chat`'s GraphRAG layer — and (b) a methodology review of an existing plan with ML content; assess deliverables against the prompt's own structures; fold findings back.

### K-002 — Perishable model/embedding landscape reference
- **Status:** 🔵 proposed
- **Priority:** low
- **Rationale:** Model/embedding capabilities and pricing are perishable; the prompt rightly forbids quoting them from memory, but repeated WebFetch verification is wasteful. A dated, `Verified:`-stamped resource file (pattern: `graph-dba/falkordb-quirks.md`) or skill could cache the current landscape.
- **Proposed change:** If model-selection questions recur, add `data-scientist/model-landscape.md` (dated entries, re-verify stamps) and point the prompt at it — kept out of the always-on prompt.

## Parking lot / ideas

- **`lm-studio-model-notes.md`'s provenance section now carries two topics (noted 2026-09-07,
  U13 distillation).** The "A live-run report's provenance … verify live" section's habit (2) has
  grown from "grep for a pinned `temperature`" into a full determinism-vs-comparability rule, so
  the section is now *provenance drift* **and** *sampling design* under one heading. Minor — the
  two are genuinely linked through the same config file, and the fold was deliberately chosen over
  a fifth parallel section. Split into its own "a pin is not determinism" section if a third
  sampling-design fact lands there.

- **Judged and kept, do not re-litigate (2026-08-24, C5 lint).** Two restatements will read as
  class-7 duplicates to a future dedup sweep; both are keeps.
  - **The model-perishability rule, in "Model selection" and again in the "No fabricated numbers"
    guardrail.** Two decision points: the first fires at **model-selection time**, the second at
    **claim-writing time** — and a deliverable can carry a capability claim with no model-selection
    step anywhere in sight. *(A weaker rationale was recorded during C5's inventory — that a
    capability claim isn't a number, so the guardrail's first sentence doesn't cover it. That
    explains why the guardrail's second sentence exists, but not why it isn't redundant against the
    fuller statement in "Model selection". The two-decision-points reading is the one that holds;
    corrected here so a later unit doesn't cut the clause after finding the stated reason weak.)*
  - **"You do not implement" in the opening paragraph vs. the `Write`/`Edit` guardrail.** Persona-
    setting vs. enforced scope plus the routing target (`coder`/`tdd-engineer`, `graph-dba`).
- **`this lab` is not a provenance habit in this file, three of four times (2026-08-24, C5 lint).**
  C5 cut two "this lab" attributives from the LLM-as-judge bullet as class-6 provenance. Four
  instances remain and every one is a keep — the test is *the lab cited as a rule's **authority***,
  not the phrase itself. The close call is "This lab's established convention for a small-n pass/fail
  bound is the **Wilson score interval**": that names the lab as the rule's **scope**, not its
  authority, and the rule ("stay consistent with the lab's convention") is unstateable without it.
  Its parenthetical "(not Clopper-Pearson or the naive rule-of-three)" is a live **anti-trigger** —
  an agent would plausibly reach for the exact interval absent it — so it is class 1, not class 6.
- **LLM-as-judge bullet is getting dense (noted 2026-08-21, team certification §7 lint fold-in).**
  The "Evaluation engineering" section's LLM-as-judge bullet now carries three distinct rules in
  one paragraph: general validity caveats, class-conditional-rate gating for a biased judge, and
  the judge-collapses-onto-agent-under-test caveat-splitting rule (added this session). Minor —
  still thematically coherent (all LLM-as-judge validity) and each sentence is self-contained, so
  not fixed now. Revisit (split into two bullets: general validity + judge-collapse) if a fourth
  rule lands in the same paragraph.
- **The agent owns two recurring `Status: archived` flips it isn't told about yet (noted 2026-07-27).** Root `AGENTS.md`'s routing table makes `data-scientist` the performer for `plans/<slug>-ml.md` and `reviews/<slug>-ml.md` at milestone close, on `teco`'s coordination; today that reaches the agent only through the closing unit's brief. One prompt line if closes start leaving `-ml` documents `active`.
- Revisit the advisory-only shape if the lab starts wanting evals *executed* rather than designed — either grant hands-on eval-execution powers (graph-dba-style) or define a standing data-scientist→qa-engineer handoff for eval execution (2026-07-09, creation decision: user chose advisory).
- A worked example of a good method note (once one exists) linked from the prompt, if note quality proves inconsistent.
