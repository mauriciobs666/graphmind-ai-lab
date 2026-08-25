# Kaizen — Improvement Plan: qa-engineer

> Forward-looking backlog for the `qa-engineer` agent.
> Status: 🔵 proposed · 🟡 in-progress · ✅ done (then moved to history.md) · ⚪ rejected/deferred
> Last reviewed: 2026-07-27

## Active

| ID | Added | Priority | Status | Summary |
|------|------------|----------|--------|---------|
| K-001 | 2026-07-01 | med | 🔵 | Ship a reusable test-plan + test-report markdown template pair (as skill or in-repo doc) so structure is consistent across runs |
| K-003 | 2026-07-01 | low | 🔵 | Consider a handoff protocol: qa-engineer files defects → coder/tdd-engineer fix → qa-engineer re-runs (regression loop) |
| K-004 | 2026-07-01 | low | 🔵 | Capture a first-run smoke-eval as a repeatable check; document the "new subagent isn't routable until a new session" registry-reload gotcha where users will see it |

### K-001 — Reusable plan/report templates
- **Status:** 🔵 proposed
- **Priority:** medium
- **Rationale:** the prompt describes the plan/report structure prose-only; a concrete template (skill or doc) would make output consistent and speed each run.
- **Proposed change:** author a small `qa-templates` skill (or a `docs/_templates/` pair) with the test-plan and test-report skeletons the agent fills in.
- **Notes:** keep it lean; progressive-disclosure skill is the natural home if it grows.

### K-003 — Defect → fix → re-run handoff
- **Status:** 🔵 proposed
- **Priority:** low
- **Rationale:** QA is most valuable in a loop with implementation. A light protocol (report format that `coder`/`tdd-engineer` consume, plus a re-run pass) closes it.
- **Notes:** The `teco` side is shipped — its roster entry carries the path-handoff convention and its integrate-and-verify step carries defect→re-brief→re-run. What remains is an assessment of the loop **in practice**, and the vehicle designated for it has already passed unassessed: falkor-chat K-022→K-025 ran to a QA acceptance PASS on 2026-07-21 and did return findings (K-027 was filed out of that pass), but how the handoff itself held up was never written down. Close on a deliberate look at one completed defect→fix→re-run cycle — that one retrospectively, or the next one live.

### K-004 — First-run smoke-eval + document the registry-reload gotcha
- **Status:** 🔵 proposed
- **Priority:** low
- **Rationale:** the first-spin (2026-07-01) confirmed the agent works but had to be **proxy-run** because a freshly-created subagent isn't in the session's registry until a new session starts. Users will hit this; it belongs in the deploy/testing notes, not tribal memory.
- **Proposed change:** add a one-line "restart the session to route to a newly added agent" note to `claude/README.md` deployment section (or `cobb/TESTING.md`), and keep the M1 pass as a lightweight smoke reference.

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
