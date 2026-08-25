# Kaizen — Improvement Plan: qa-engineer

> Forward-looking backlog for the `qa-engineer` agent.
> Status: 🔵 proposed · 🟡 in-progress · ✅ done (then moved to history.md) · ⚪ rejected/deferred
> Last reviewed: 2026-07-27

## Active

| ID | Added | Priority | Status | Summary |
|------|------------|----------|--------|---------|
| K-001 | 2026-07-01 | med | 🔵 | Ship a reusable test-plan + test-report markdown template pair (as skill or in-repo doc) so structure is consistent across runs |
| K-002 | 2026-07-01 | med | ✅ | Define the `docs/test-plans/` + `docs/test-reports/` convention explicitly in `falkor-chat/AGENTS.md` (currently inferred by the agent) — done 2026-07-11 via the docs-unification pass (see history) |
| K-003 | 2026-07-01 | low | 🔵 | Consider a handoff protocol: qa-engineer files defects → coder/tdd-engineer fix → qa-engineer re-runs (regression loop) |
| K-004 | 2026-07-01 | low | 🔵 | Capture a first-run smoke-eval as a repeatable check; document the "new subagent isn't routable until a new session" registry-reload gotcha where users will see it |
| K-005 | 2026-08-24 | low | ✅ | Phase 2's lead-in still offers component-negotiable doc *naming* — the one place negotiability no longer survives. Re-point it at structure and IDs. |
| K-006 | 2026-08-24 | low | ✅ | Phase 3 bullet 2 is one ~60-word compound sentence carrying four separable rules; `tdd-engineer` step 3 carries identical content as labeled sub-bullets. |

### K-001 — Reusable plan/report templates
- **Status:** 🔵 proposed
- **Priority:** medium
- **Rationale:** the prompt describes the plan/report structure prose-only; a concrete template (skill or doc) would make output consistent and speed each run.
- **Proposed change:** author a small `qa-templates` skill (or a `docs/_templates/` pair) with the test-plan and test-report skeletons the agent fills in.
- **Notes:** keep it lean; progressive-disclosure skill is the natural home if it grows.

### K-002 — Pin the artifact-location convention in component docs
- **Status:** ✅ done 2026-07-11 — the repo-wide docs unification defined the module documentation convention (active `docs/test-plans/`+`docs/test-reports/` vs. frozen `docs/archive/`) in the root `AGENTS.md` and the `falkor-chat/AGENTS.md` key-docs table; the agent's PLAN bullet was updated to match (see history 2026-07-11). **Superseded in part, 2026-07-27:** the active-vs-`archive/` *move* half of that convention is gone (D4 — a frozen document stays put and gets `Status: archived`), and the convention now also fixes a repo-wide filename grammar the agent may not renegotiate per component. Still done; the current statement is the root `AGENTS.md` bullet.
- **Priority:** medium
- **Rationale:** the agent currently *detects* where to write plans/reports. Writing the convention into `falkor-chat/AGENTS.md` (and other components as they gain QA needs) removes ambiguity and drift.
- **Proposed change:** add a short "Test plans & reports live in `docs/test-plans/` and `docs/test-reports/`, kebab-case per feature" note to the relevant component `AGENTS.md`.

### K-003 — Defect → fix → re-run handoff
- **Status:** 🔵 proposed
- **Priority:** low
- **Rationale:** QA is most valuable in a loop with implementation. A light protocol (report format that `coder`/`tdd-engineer` consume, plus a re-run pass) closes it.
- **Notes:** `teco` could orchestrate; verify subagent-to-subagent handoff patterns before hardwiring. **Update 2026-07-09:** the teco side is now in teco's prompt (roster entry with path-handoff conventions + defect→re-brief→re-run in its integrate-&-verify step); remains open pending a live orchestrated cycle. **Update 2026-07-12:** the designated live cycle is falkor-chat **K-022→K-025** (the first fully-gated coordinated run — see teco K-003): if the analyst impl review or the K-025 acceptance pass returns findings, drive the loop there and log how the handoff held up. teco K-002 is separately evaluating `SendMessage` continuation to make the re-run half cheaper (continue the original implementer instead of re-spawning cold).

### K-004 — First-run smoke-eval + document the registry-reload gotcha
- **Status:** 🔵 proposed
- **Priority:** low
- **Rationale:** the first-spin (2026-07-01) confirmed the agent works but had to be **proxy-run** because a freshly-created subagent isn't in the session's registry until a new session starts. Users will hit this; it belongs in the deploy/testing notes, not tribal memory.
- **Proposed change:** add a one-line "restart the session to route to a newly added agent" note to `claude/README.md` deployment section (or `cobb/TESTING.md`), and keep the M1 pass as a lightweight smoke reference.

### K-005 — Phase 2's lead-in is the last invitation to component-negotiable doc naming
- **Status:** 🔵 proposed
- **Priority:** low
- **Rationale:** C5 removed ", or the component's convention" from phase 4's test-report path,
  because a report filename **is** filename grammar — repo-wide per root `AGENTS.md`, and stated as
  non-negotiable twice in this prompt already. That leaves phase 4 absolute and phase 2 negotiable-
  with-exception. Phase 2's exception clause is correct and m17-certified (see the judged-and-kept
  note below) and must stay; the problem is the *lead-in* above it — "matching the project's naming
  conventions (discover them — don't impose)" — which is the first thing a skimming agent reads and
  now points at the one dimension that is no longer negotiable at all.
- **Proposed change:** re-point the lead-in at **structure and IDs** (test-item structure, the
  component's backlog-ID form, how it references specs) rather than *naming*. That is where genuine
  component negotiability still lives, and it leaves the exception clause doing its job unaided.
- **Provenance:** `cobb` §7 lint during C5 of `claude/docs/plans/prompt-waste-reduction.md`.
  Pre-existing; not bundled into C5 because it is a rule change.

### K-006 — Phase 3 bullet 2 packs four rules into one sentence
- **Status:** 🔵 proposed
- **Priority:** low
- **Rationale:** "Run the existing suite and scripts" is a single ~60-word compound sentence
  carrying four separable rules: establish a green baseline first; never pile onto a red baseline;
  never pile onto an un-runnable one; and report/propose/ask, with the delegated-subagent fallback.
  `tdd-engineer`'s Workflow step 3 carries the same content as three labeled sub-bullets
  (greenfield / already red / can't run here) and is markedly more followable.
- **Proposed change:** mirror that structure. Purely structural — no rule added, none removed, and
  no net word cost either way.
- **Provenance:** `cobb` §7 lint during C5. Pre-existing; a restructure, not a compression, so not
  bundled into that commit.

## Parking lot / ideas

- **Judged and kept, do not re-litigate (2026-08-24, C5 lint).** Four restatements in this file
  will read as class-7 duplicates to any future dedup sweep. All four are keeps, and the reasons
  differ:
  - **The doc-convention override, stated twice with its exception** — once in phase 2 ("Detect the
    convention first"), once in the "Match the project" principle. This one is not a judgment call:
    `docs/reviews/doc-reference-convention.md` **m17** found the second clause and ruled that it
    must carry the exception too, *"or the rewritten `:28` is contradicted from 26 lines below."*
    Removing either is a regression against a completed review.
  - **The environment-mutation rule + its subagent fallback, in phase 3 and in Guardrails** — the
    strongest-looking candidate in the file (22 w) and still a keep: phase 3's instance fires inside
    the baseline procedure, Guardrails' is the standing rule read at a different moment. Also
    protected structurally — the `agent-maintenance` skill §4 check 3 requires *every* "ask" phrasing
    to carry its own delegated-subagent carve-out, so thinning one is a certification regression.
  - **"Never report a pass you didn't observe" (Principles) vs. "never invent a passing run"
    (Guardrails)** — the one genuine "same moment, same actor" instance in the file, i.e. the only
    one that fails the finding-5 test. Kept anyway: anti-fabrication is the last category to thin
    for a 7-word gain.
  - **"Match the component's existing framework, layout, naming…" (phase 3) vs. "Match the project"
    (Principles)** — authoring a test file vs. general orientation.
- **Two accuracy nits, noted not fixed (2026-08-24, C5 lint).** (1) The second hook bullet's
  "nothing else" is very slightly wrong: the shared core `claude/scripts/guard-doc-writes.sh`
  appends `/tmp/*` to every wrapper's allowlist, so three path classes are allowed, not two.
  Behaviorally immaterial under `on_mismatch=pass`; recorded so a future enforcement-parity pass
  doesn't score it as drift. (2) The retained "This hook never escalates" is arguably 4 words of
  waste — both branches are already exhaustively enumerated by the two sentences before it, and its
  intended payload (contrast with every other agent's guard) is a contrast this agent cannot see,
  since it never reads another agent's prompt. Left alone in C5 rather than re-editing a bullet
  whose lint had already run; available to a future unit under its own gate.
- **Corpus non-conformance to reconcile elsewhere (2026-08-24).** `falkor-chat/docs/test-reports/`
  holds five filenames that don't match the repo-wide grammar — `graphrag-eval-2026-08-15.md`,
  `graphrag-eval-2026-08-16.md`, `guard-judge-calibration-2026-08-17.md`,
  `guard-judge-calibration-2026-08-21.md` (dated, no `-report` role) and
  `docs/test-reports/kaizen-agent-ontology.md` (no `-report`). C5's edit removed the prompt's only
  textual license for them, so the prompt is now correct and the corpus is not — while the agent is
  separately told to learn from the corpus. **Not a `qa-engineer` prompt item**: the case those
  files represent (a second run of the same test plan) is already answered by root `AGENTS.md`
  collision rule 5, which is auto-loaded. It belongs to whoever next reconciles `falkor-chat/docs/`.
- Optional non-functional playbooks (perf smoke via `GRAPH.PROFILE`, basic security/permission probes) as an on-demand skill rather than resident prompt weight.
- A `qa-engineer` ↔ `saul`/`dra-claudia`-style workdir option if the user later wants reports kept out of version control (currently in-repo `docs/` per user's choice).
