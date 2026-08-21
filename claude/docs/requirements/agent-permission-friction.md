# Agent permission-escalation friction — Feature Requirements
> **Status:** Interviewing · **Owner:** `tico` · **Tracks:** — · **Last updated:** 2026-08-20

## Intent
Reduce permission-escalation prompts that fire on legitimate, safe agent actions, without
weakening the guardrails that exist to catch a genuine accidental drift (an agent editing
something outside its remit, or a coder-type agent doing something destructive/irreversible).
The stakeholder is running a live session and will relay concrete escalation instances as they
happen; this document is being built from real evidence rather than hypothetical ones.

## Problem & current state
Two escalation mechanisms are in play, and both have been flagged as firing too often or on the
wrong things:

1. **The five doc-scoped write guards** (`architect`, `analyst`, `data-scientist`, `teco`, `tico`
   — thin wrappers over `claude/scripts/guard-doc-writes.sh`) escalate any `Write`/`Edit` outside
   a narrow per-agent allowlist to a human "ask" prompt. Each allowlist currently exempts the
   agent's own `kaizen/inbox.md` — but since the 2026-08-20 kaizen consolidation, raw learnings
   capture writes directly to the shared `kaizen_team` FalkorDB graph via `mcp__cypher__query`
   (a call the `Write|Edit`-matched guard never sees), and `kaizen/inbox.md` itself is now a
   frozen historical snapshot nobody writes to. Meanwhile `kaizen/history.md` and `kaizen/plan.md`
   — per the current documented convention, curated by `cobb`, not self-edited by the agent — are
   **not** in any of the five allowlists. The stakeholder has hit an escalation trying to get an
   agent to add to its own `history.md` (and possibly other kaizen files).
2. **`coder`** (no doc-scoped write guard; `permissionMode: acceptEdits`, otherwise Claude Code's
   default tool-permission behavior) is, per the stakeholder, asking for permission "much" more
   than wanted. Specific trigger(s) not yet pinned down — collecting live examples.
3. **The plain default "confirm before Edit/Write" prompt** fires on any agent that lacks
   `permissionMode: acceptEdits` — independent of, and in addition to, both mechanisms above.
   `coder` is the only agent in the roster with that setting today; every other agent (including
   ones with no custom write guard at all, like `cobb`, and the five with a doc-scoped guard) gets
   a manual confirmation on every `Edit`/`Write`, even for an action squarely inside its normal,
   documented remit. First live instance (below) confirms this fires even when the guard mechanism
   doesn't apply at all.

## User stories
- As a stakeholder, I don't want to be interrupted approving an agent's `Write`/`Edit` when it's
  squarely that agent's own normal, documented work — regardless of which agent it is.
- As a stakeholder, I want `cobb` in particular to be able to edit anything across the agent team
  (definitions, kaizen files, related docs) without being stopped, since touching everyone else's
  files is its entire job, not an edge case.
- As a stakeholder, I still want to be asked when an agent does something genuinely outside its
  remit, or something destructive/irreversible — the interruption should mean something when it
  happens.
- As a stakeholder, I want `coder` to run with looser permission requirements so it isn't stopping
  to ask for things that don't warrant an interruption — specifics pending live examples.

## Functional requirements
Settled: applies **team-wide** (stakeholder, 2026-08-20: "applies to all legitimate cases"), not
agent-by-agent. FR-2 is the governing requirement; FR-1 and FR-3 are its two evidenced instances,
kept as their own line items because each has concrete, confirmed evidence behind it.

- **FR-2 (governing):** An agent performing a `Write`/`Edit` that stays within its own
  already-documented remit/deliverable-path convention must not require a manual per-action
  confirmation. An agent performing a `Write`/`Edit` genuinely outside that remit must still be
  escalated for approval — this requirement narrows *when* confirmation fires, it does not remove
  the safety net for a genuine accidental drift.
- **FR-1 (instance of FR-2):** `cobb` specifically gets no path restriction at all — "cobb can edit
  anything on the agents" (stakeholder, 2026-08-20), confirmed to include `cobb`'s own file
  (instance 10) as well as every other agent's. Unlike every other agent, cobb's whole job is
  touching other agents'/the team's files, so there's no meaningful "in vs. out of remit" line to
  draw for it. Evidenced by instances 1–3, 5, 6, 10 below.
- **FR-3 (instance of FR-2):** `qa-engineer` writing to its own stated deliverable paths
  (`docs/test-plans/`, `docs/test-reports/`) must not require a manual per-write confirmation.
  Evidenced by instance 4 below.

## Instances observed (live, from the concurrent `teco` session)
1. **2026-08-20 — `cobb`, `Edit` on `claude/analyst/analyst.md`.** No custom write guard applies
   to `cobb` (unrestricted tools). Stakeholder confirmed: this was `cobb` doing its normal
   agent-maintenance job; approved. Mechanism: default Claude Code "confirm before Edit" prompt
   (mechanism 3 above), not a custom guard hook. → supports FR-1.
2. **2026-08-20 — `cobb`, `Edit` on `claude/data-scientist/data-scientist.md`.** Same shape as
   instance 1 (no custom guard on `cobb`; default confirm-before-Edit prompt). Confirmed by the
   stakeholder as legitimate.
3. **2026-08-20 — several further `cobb` edits, all to other agents' own system-prompt files**
   (`<name>/<name>.md`). Stakeholder: "there were several from cobb all while trying to edit
   other agents' system prompts which is his purpose" — every one legitimate, none an accidental
   drift. Pattern is now well-evidenced: `cobb` editing another agent's own `<name>.md` is its core
   job, not an edge case.
4. **2026-08-20 — `qa-engineer`, `Create` (Write) on `docs/test-plans/generic-cypher-mcp2.md`.**
   Exactly `qa-engineer`'s stated deliverable path (its versioned test plan). No custom write guard
   restricts `qa-engineer`'s `Write`/`Edit` paths (its guard is destructive-ops-only, Bash-scoped).
   Confirmed legitimate. → same mechanism-3 friction as the `cobb` instances, on a different agent:
   the "confirm before Edit/Write" default isn't cobb-specific, it's team-wide except `coder`.
   (Confirmed again shortly after on the other half of its deliverable pair — `Create` on
   `docs/test-reports/generic-cypher-mcp2-report.md` — matching AC-2 exactly.)
5. **2026-08-20 — `cobb`, `Edit` on `claude/graph-dba/kaizen/history.md`.** Directly resolves the
   original kaizen-file thread (Open questions 1–2, below): `cobb` — not the individual doc-scoped
   agent — is the one editing `kaizen/history.md`/`plan.md`, per the documented "curated by cobb"
   convention. Confirmed legitimate. → folded into FR-1: `cobb`'s remit explicitly includes kaizen
   curation across the team, not just agent-prompt/README edits.
6. **2026-08-20 — `cobb`, `Edit` on `cypher-mcp/README.md`.** Not an agent system prompt this
   time — a component README. Cobb's documented remit explicitly covers MCP wiring/cross-tool
   standards, so this is still inside its job, not a drift. Stakeholder confirmed: legitimate,
   approved. → broadens FR-1: the friction isn't confined to `<name>/<name>.md` edits, it's any
   routine `cobb` edit across its documented remit (agent/skill files, MCP/agent-standards
   documentation wherever it lives, kaizen curation for the team).
7. **2026-08-20 — `tdd-engineer`, `Edit` on `server/tests/test_guards.py`, recurring.** Squarely
   its own core job (writing/editing tests); no custom guard applies to `tdd-engineer` at all.
   Confirmed legitimate (per the "only sharing approved cases" rule). The *same* file was edited
   again moments later — consistent with a red→green→refactor TDD cycle hitting one test file
   repeatedly. → third distinct agent hitting the identical mechanism-3 friction on its own core
   work (`cobb`: team maintenance, `qa-engineer`: test plans, `tdd-engineer`: test code) — and
   shows the cost compounds *per edit*, not per task: a single legitimate TDD cycle can trigger
   several separate confirmations against the same file.
8. **2026-08-20 — `tdd-engineer`, `Edit` on `server/falkorchat/guards.py`.** Source/implementation
   code this time, not a test file — but still core TDD work (red→green: writing the simplest
   code to make a failing test pass). Confirmed legitimate. → broadens the evidence past
   docs/test-file edits: FR-2's "in-remit work shouldn't need confirmation" applies equally to an
   implementer agent editing actual source code as part of its sanctioned task, not just to
   doc-authoring agents. (Edited again shortly after — same red→green→refactor compounding-cost
   pattern as instance 7, now on a source file rather than a test file.)
9. **2026-08-20 — `data-scientist`, `Create` on `tests/eval/probe_ministral_judge.py` — NOT
   friction, the guard working correctly.** `data-scientist` is advisory-only ("never implements"),
   and its write guard only allows `docs/plans/*`/`docs/reviews/*` — a `tests/eval/` script is
   genuinely outside that. Stakeholder, on reflection: "this one is not his role." Kept in this
   document as a **counter-example**, not FR-2 evidence: it validates AC-4 (escalation must stay
   for genuinely out-of-remit work) and shows the distinction this feature draws — in-remit
   friction goes away, a real drift still gets caught — is a real, live distinction, not just a
   theoretical carve-out.
10. **2026-08-20 — `cobb`, `Edit` on `claude/cobb/cobb.md` (its own file).** Confirms FR-1's
    "anything on the agents" extends to `cobb` editing itself, not just other agents — closes a
    reading ambiguity in the earlier phrasing.

## Out of scope
- **The destructive-ops guards** (`devops`, `graph-dba`, `qa-engineer`'s Bash-pattern guard over
  `claude/scripts/guard-destructive-ops.sh` — `GRAPH.DELETE`, `FLUSHALL`/`FLUSHDB`, volume wipes,
  `docker rm -f`, `pipeline.sh ... --reset`) are unaffected. Those exist to catch genuinely
  irreversible actions, not routine doc/code writes — no evidence or stakeholder ask has touched
  this mechanism, and it stays exactly as strict as it is today.
- **Git-commit authority scoping** (only `tico`/`teco` may `git add`/`git commit`, per the
  2026-07-30 stakeholder decision) is unaffected — this feature is about `Write`/`Edit`
  confirmation, not who may version-control what.
- **The doc-scoped guards' "ask" behavior for genuinely out-of-remit paths stays** — e.g. if
  `architect` tries to edit something outside `docs/plans/`, that should still escalate. This
  feature removes the *redundant* confirmation on work that's already in-scope, not the guard's
  actual catch for a genuine drift.

## Acceptance criteria
- **AC-1 (FR-1):** Given `cobb` performs any `Write`/`Edit`, when the tool call runs, then no
  manual confirmation prompt appears.
- **AC-2 (FR-3):** Given `qa-engineer` writes/edits a file under `docs/test-plans/` or
  `docs/test-reports/`, when the tool call runs, then no manual confirmation prompt appears.
- **AC-3 (FR-2, general):** Given any other agent performs a `Write`/`Edit` within its own
  already-documented deliverable-path convention (e.g. `tdd-engineer` editing a test file,
  `architect` editing `docs/plans/*`), when the tool call runs, then no manual confirmation prompt
  appears.
- **AC-4 (FR-2, safety net preserved):** Given any agent performs a `Write`/`Edit` genuinely
  outside its documented remit, when the tool call runs, then a manual "ask" confirmation still
  appears.
- **AC-5 (out of scope, regression check):** Given a destructive operation via `devops`/
  `graph-dba`/`qa-engineer` (e.g. `GRAPH.DELETE`, `FLUSHALL`, a volume wipe), when the command
  runs, then a manual confirmation still appears, unchanged from today.

## Open questions
1. For `coder`: which specific actions are triggering unwanted prompts? Still no live example —
   `coder` already has `permissionMode: acceptEdits`, so its friction (if real) is likely a
   *different* mechanism (e.g. Bash/terminal-command confirmations) than the one this document has
   otherwise nailed down (`Write`/`Edit` confirmation). Not folding it into FR-2 until there's a
   concrete instance to confirm that.

## Decision log
- 2026-08-20 — Stakeholder: "my agents ask too much permission when editing plans reviews and
  other files" → opened as a requirements interview (Mode 1); grounded in the existing
  `guard-doc-writes.sh` design (architect kaizen, 2026-07-08) before asking anything.
- 2026-08-20 — Stakeholder: kaizen inbox is now on the graph; wants agents able to add to
  `history.md` "and other relevant files"; separately, `coder` needs much looser permission rules
  → recorded as two threads in this doc; clarifying questions asked, not yet answered.
- 2026-08-20 — Stakeholder is running a concurrent `teco` session and will relay each
  permission-escalation instance here as it happens, rather than answering hypothetically →
  interview proceeds evidence-first; this doc updates per instance.
- 2026-08-20 — Instances 1–2 (`cobb` editing `analyst.md`/`data-scientist.md`) confirmed
  legitimate and approved individually.
- 2026-08-20 — Stakeholder: "there were several from cobb all while trying to edit other agents'
  system prompts which is his purpose" (instance 3) → pattern established for FR-1.
- 2026-08-20 — Asked whether to save an FR-1 improvement item directly into `cobb/kaizen/plan.md`
  → declined: outside tico's Write/Edit scope (requirements docs + manuals only) and outside its
  sanctioned `Agent`-delegation uses; pointed the stakeholder to relay FR-1 to their concurrent
  `teco`/`cobb` session directly, with this doc as the evidence trail either way.
- 2026-08-20 — Instance 4 (`qa-engineer` → `docs/test-plans/generic-cypher-mcp2.md`) and instance
  5 (`cobb` → `claude/graph-dba/kaizen/history.md`) confirmed legitimate.
- 2026-08-20 — Stakeholder: "im only sharing approved cases" → every instance from here on is
  treated as already-confirmed-legitimate without re-asking per instance.
- 2026-08-20 — Instance 6 (`cobb` → `cypher-mcp/README.md`) relayed under the above rule.
- 2026-08-20 — Asked whether the fix should be team-wide or agent-by-agent → Stakeholder: "cobb
  can edit anything on the agents" (settles FR-1: no path restriction for `cobb`) → then "applies
  to all legitimate cases" (settles FR-2: team-wide default, not agent-by-agent).
- 2026-08-20 — Instance 7 (`tdd-engineer` → `server/tests/test_guards.py`, recurring) and instance
  8 (`tdd-engineer` → `server/falkorchat/guards.py`, source code) relayed; broadened FR-2 to
  source-code edits and to per-edit (not per-task) friction cost.
- 2026-08-20 — Instance 9 (`data-scientist` → `tests/eval/probe_ministral_judge.py`) flagged by
  `tico` as a likely-genuine guard hit rather than friction, since it falls outside
  `data-scientist`'s "never implements" contract and its write-guard allowlist. Stakeholder agreed
  on reflection: "this one is not his role" → kept as a counter-example (AC-4), not folded into
  FR-2/FR-3 evidence.
