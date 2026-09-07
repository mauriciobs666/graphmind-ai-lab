# Kaizen — Improvement Plan: data-scientist

> Forward-looking backlog for the `data-scientist` agent.
> Status: 🔵 proposed · 🟡 in-progress · ✅ done (then moved to history.md) · ⚪ rejected/deferred
> Last reviewed: 2026-09-07

## Active

| ID | Added | Priority | Status | Summary |
|------|------------|----------|--------|---------|
| K-001 | 2026-07-09 | med | 🔵 | First-run shakedown: a real method note + a real methodology review |
| K-002 | 2026-07-09 | low | 🔵 | Perishable model/embedding landscape reference (skill or resource file) |
| K-003 | 2026-09-07 | low | 🔵 | Route the eager-provider-resolution trap to `falkor-chat`'s provider-config manual (owner: `tico`) |
| K-004 | 2026-09-07 | med | 🔵 | Correct a committed arithmetic error: the pinned `_Z_95` is **two** ULPs from `inv_cdf(0.975)`, not one (owner: `teco` to route) |

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

### K-003 — Route the eager-provider-resolution trap to its owning document
- **Status:** 🔵 proposed
- **Priority:** low
- **Origin:** kaizen entry `e1a6c4d2-8b3f-4b1a-9c7e-3f2a6d9b1c4e` (2026-08-31), kept open in the
  U14 distillation pass — see `history.md`, 2026-09-07, chunk B.
- **Rationale:** `ModelGateway.__init__` → `_build_providers` resolves the `{env:}`/`{file:}`
  substitution for **every** declared provider, not the one a caller actually dispatches to, so a
  harness pointed at `falkor-chat/config/opencode.example.json` dies on the example file's unused
  `openai` provider unless `OPENAI_API_KEY` is set to a placeholder first. Four independent scripts
  have hit it. The mechanism is verified; only its home is unresolved.
- **Proposed change:** one clause in `falkor-chat/docs/manuals/llm-provider-config.md` §2 — the
  section already tells an operator that a missing `{env:}` variable fails startup, but not that
  this fires for a provider nothing ever resolves to. **Not** `falkor-chat/AGENTS.md`: an
  always-loaded context file is the wrong price for a fact that binds only when someone writes a
  new live-harness driver, and the workaround is already commented at the point of use in
  `server/tests/eval/test_guard_calibration_live.py`.
- **Notes:** `manuals/` is `tico`-owned and outside `cobb`'s write remit, so this needs routing by
  the human or by `teco`. The entry is also tagged `MENTIONS → tico` in `kaizen_team`, so it will
  resurface in `tico`'s own distillation pass if it isn't routed sooner.

### K-004 — Correct the committed "one ULP" wording: the pinned `_Z_95` is **two** ULPs away
- **Status:** 🔵 proposed
- **Priority:** medium
- **Origin:** kaizen entry `7f3c1a92-5d64-4b0e-9a11-c8e2f0b47d31` (2026-09-03), kept open in the
  U15 distillation pass — see `history.md`, 2026-09-07, chunk C.
- **Rationale:** This is **not** a promotion request — the fact is already published in the
  strongest form available to it, as the committed executable assertion
  `test_z_95_matches_the_inverse_normal_cdf` (`model-bench/tests/test_stats.py`, line 83 as of
  2026-09-07), which asserts both `_Z_95 != NormalDist().inv_cdf(0.975)` and
  `abs(...) < 1e-12`, and whose docstring forbids tightening the comparison to `==`. What is open
  is an **arithmetic error inside that published statement**: every copy says the two doubles are
  *one* ULP apart. They are **two**. The delta is `4.440892098500626e-16`;
  `math.ulp(1.9599639845400536)` is `2.220446049250313e-16`. Confirmed three ways in the U15 pass
  — `delta / ulp == 2.0`, IEEE-754 bit distance 2, and `math.nextafter(inv_cdf, +inf)` applied
  **twice** landing exactly on the pin. The bottom line is untouched: the doubles are unequal, so
  `==` fails and `< 1e-12` passes; only the ULP count is wrong.
- **Proposed change:** replace "one ULP" with "two ULPs" (and, where the delta is quoted, keep
  `4.44e-16` — it is right) in the three committed places carrying the wording. Line numbers are
  as of 2026-09-07 and drift under the concurrent session; the anchors are the names:
  - `model-bench/tests/test_stats.py:89` — the `test_z_95_matches_the_inverse_normal_cdf`
    docstring (*"4.44e-16 — one ULP"*).
  - `model-bench/docs/HISTORY.md:471` — *"pinned literal is one ULP from
    `NormalDist().inv_cdf(0.975)` and must not be tightened to `==`"*.
  - `docs/reviews/small-model-benchmarking-ml.md:247` — finding **n-ML-3**, whose heading carries
    the same claim.
- **Notes:** open rather than fixed because all three paths are **outside `cobb`'s write remit**
  *and* were under a concurrent session's active edit during U15, so nothing under `model-bench/**`
  or `docs/**small-model-benchmarking*` was written. `teco` routes the correction once that
  session lands. Note the split ownership: `docs/reviews/small-model-benchmarking-ml.md` is a
  document **`data-scientist` itself owns** under root `AGENTS.md`'s by-kind table (`-ml` reviews),
  so that one can be fixed by this agent directly; the two `model-bench/` files belong to whoever
  is delivering that component. No `MENTIONS` tag was added in the graph — tagging would misdirect
  a correction that is half this agent's own and half a human routing decision.

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
