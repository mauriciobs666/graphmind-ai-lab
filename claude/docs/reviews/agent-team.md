# Review — The Claude Agent Team user manual (factual/architectural half)

> **Status:** active · **Owner:** `analyst` · **Tracks:** —

## Scope & verdict

**Reviewed:** `claude/docs/manuals/agent-team.md` (working-tree rewrite, uncommitted, 446 lines), the
factual/architectural claims plus clarity for someone *directing* the team. Checked against every
`claude/<name>/<name>.md` frontmatter, `claude/AGENTS.md`, `claude/teco/teco.md`, `claude/tico/tico.md`,
the hook wrappers/cores under `claude/<name>/hooks/` and `claude/scripts/`,
`claude/teco/hooks/ensure-environment.sh`, root `AGENTS.md`, `falkor-chat/docs/BACKLOG.md`, root
`docs/BACKLOG.md`/`HISTORY.md`. The two one-liners were executed. Walkthrough behaviour is
`qa-engineer`'s half and was not exercised.

**Verdict: approve with suggestions.** No blockers. The core claims hold, all verified: 13 agents;
12 carry `PreToolUse` hooks and `frontend-engineer` has none; ledger thresholds and the
seven-value status set; the docs-only vs. code commit timing; `teco` lists backlog removals and the
human applies them; root `docs/BACKLOG.md`/`HISTORY.md` belong to `cpg/` (their own H1s say so);
each of the 13 agents has `kaizen/plan.md`; a `tico` consult is announce-then-proceed and an
architect handoff waits for an offer to be accepted (`tico.md` L42, L35, L181). One major finding
and seven minor ones.

CPG: considered, not relevant — the only loaded code graph is `cpg_falkorchat`; the claims under review live in `claude/` Markdown and Bash hooks, which were read directly.

## Findings

### Major

**M1 — Backlog markers and "removed, not ticked" contradict the real legend and the manual itself.**
Claim (§How tasks are tracked, table): *"Markers: 🔵 proposed · 🟡 in progress · ⚪ rejected/deferred.
… a delivered item is removed, not ticked."*
Evidence: `falkor-chat/docs/BACKLOG.md:16` has a fourth marker: `✅ delivered (→ HISTORY.md)`.
Root `AGENTS.md` makes removal a milestone-close step (teco lists, human applies). The manual's own
diagram says the same: *"at milestone close, delivered items are removed from the backlog"*. So
between delivery and close a reader will see ✅ items that the manual says cannot exist.
Fix: add `✅ delivered (→ HISTORY.md)` to the marker list. Reword to: "a delivered item is marked ✅
and pointed at `HISTORY.md`. At milestone close it is removed, never kept as a ticked row."

### Minor

**m1 — "not a repo-wide one" is false for cross-component features.**
Claim (§One name per feature): *"All of these live under the component's own `docs/` folder —
`falkor-chat/docs/plans/x.md`, not a repo-wide one."*
Evidence: root `AGENTS.md` and `tico.md` L72 say cross-component work goes in repo-root `docs/`.
Running the manual's own `find` for `salesperson-ui` returns its whole core family under root
`docs/` (`requirements/`, `plans/`, `reviews/`, `test-plans/`, `test-reports/`). `model-bench`'s
requirements and plan are there too.
Fix: "…under the component's own `docs/` folder, or repo-root `docs/` when a feature spans several
components." Put this next to the `cpg/` trap at L255, because the two facts together are the
confusing part: root `docs/requirements/` and `docs/plans/` are shared, but root
`BACKLOG.md`/`HISTORY.md` belong to `cpg/`.

**m2 — Backlog numbering is not always `K-0NN`.**
Claim: *"Items are numbered `K-0NN`."*
Evidence: root `docs/BACKLOG.md` (the `cpg/` backlog that L255 points readers to) uses `C-3NN` and
`C-5NN` (`C-310`, `C-320`, `C-507`).
Fix: "Items carry a numbered id (`K-0NN` in most components; `C-NNN` in the root `cpg/` backlog)."

**m3 — The approval table leaves `coder`/`tdd-engineer` out of the destructive-ops row.**
Claim: row "A destructive operation… | `devops`, `graph-dba`, `qa-engineer`".
Evidence: `coder`/`tdd-engineer` each have a Bash hook (`guard-*-broad-bash.sh`) that passes the
command through `guard-destructive-ops.sh` and relays its `ask` (`claude/AGENTS.md`, Hook machinery).
Smaller gaps: the `teco`/`tico` row also asks for writes into `.git/`, and `qa-engineer`'s write
guard never asks (`on_mismatch=pass`). The table leaves `qa-engineer` out of the write rows, which is
correct.
Fix: add `coder`, `tdd-engineer` to the destructive row.

**m4 — The commit diagram undersells what `tico` may commit and where docs-only batching applies.**
Claim: box T *"tico — its requirements docs, manuals, ledgers, and reports from experts it
consulted"*. The docs-only / code split (`K`) hangs only off the "delegated by teco" branch.
Evidence: `tico.md` L177 lists two more grants. (d) covers deliverables of units tico is
coordinating, e.g. architect's plan in a tico-run chain. (c) covers a one-off it wrote this
session with the stakeholder's explicit approval. Docs-only batching applies to tico's chains too
(`tico.md` L58; `claude/AGENTS.md`, docs-only batching).
Fix: box T → "…ledgers, and the plans/reviews produced in a docs-only chain it runs". Route T into
the "docs only → one commit when the chain finishes" leaf as well.

**m5 — The document-status diagram sends every kind to `archived` at milestone close, manuals included.**
Claim: `[*] --> active: every other kind` … `active --> archived: milestone closes`.
Evidence: `tico.md` L154: a manual stays `active` for its whole life and is `archived` only when the
feature it documents is removed. Living documents (BACKLOG, HISTORY, DESIGN, README) never freeze
(root `AGENTS.md`). `tico` may also archive a requirements doc itself when design and
implementation land in the same session (`tico.md` L114).
Fix: change the edge label to "milestone closes (plans, reviews, test docs, requirements)". Add one
line under the diagram: "Manuals stay active until the feature is removed; backlog/history/design
docs are living and never archived."

**m6 — "Approved means frozen" is stated as applying to *all* documents.**
Claim (§Technical vs. user documentation, "A few rules apply to all of them"): *"Once a document
has been approved or acted on, a change becomes a new document (`x2.md`)…"*
Evidence: root `AGENTS.md` collision rule 5 covers versioned artifacts (requirements, plans,
coordination docs). Living documents (BACKLOG, HISTORY, DESIGN, AGENTS/README) and manuals are
edited in place by design. This manual is itself an in-place rewrite.
Fix: scope it: "Approved plans, requirements and reports are frozen… Living documents (backlog,
history, design docs, manuals) are edited in place."

**m7 — The "Who calls whom" legend misreads `teco → tico`.**
Claim: *"A dotted arrow is a handover of work already done"*. The diagram draws
`teco -. "manual update, facts already known" .-> tico` dotted.
Evidence: `teco.md` routing table, the manual row: "delegable to `tico`" for a self-contained manual
write. That is a live delegation that returns a result, not a handover of finished work.
Fix: make that edge solid (`teco -- "manual update (facts known)" --> tico`).

**m8 — `claude/` tracks itself through more than kaizen.**
Claim: *"`claude/` has no backlog of its own; each agent has `claude/<name>/kaizen/plan.md`… and
`history.md`."*
Evidence: true as far as it goes (no `claude/docs/BACKLOG.md`/`HISTORY.md`; 13 `kaizen/plan.md`).
But team-wide changes run the normal feature path under `claude/docs/`: `requirements/`, `plans/`
(incl. `*-coordination.md` ledgers), `reviews/`, `test-plans/`, `test-reports/`. Example:
`agent-knowledge-base-strategy*` (25 files). Its delivery is logged in
`falkor-chat/docs/HISTORY.md` (e.g. the 2026-09-20 "K-030 item 2" entry).
Fix: add "Team-wide changes still use the normal requirements → plan → review → ledger path, under
`claude/docs/`."

### Nits

- **n1 — The `find` one-liner over-matches by prefix.** `find . -path '*docs*' -name 'agent-team*'`
  also returns `falkor-chat/docs/plans/agent-team-ingestion-graph.md` and two unrelated
  `claude/docs/reviews/agent-team-*` docs. Add "(it may also catch other slugs that start the same
  way)". The `grep -m1 -H 'Status:' docs/plans/*.md` one-liner works as written when run inside a
  component (tested in `falkor-chat/` and the repo root).
- **n2 — Add the `-impl` suffix to the "Specialist notes add a suffix" list.** It is what the path
  diagram's "analyst reviews the diff" step produces (`reviews/x-impl.md`, e.g.
  `docs/reviews/salesperson-ui-impl.md`), and readers will meet it often.
- **n3 — Audience trim.** "Context files stay small… about 2,500 words max" is a rule for doc
  authors, not for someone directing the team, and root `AGENTS.md` calls it "smells, not gates", so
  "max" overstates it. Suggest dropping the bullet. The ledger's `Cost` column and its example
  `180k tok` could also be dropped or glossed in a few words. Keep `Agent id`: the explanation at
  L200–202 earns it.
- **n4 — Header.** `**Tracks:** — (—)` should be `**Tracks:** —` (root `AGENTS.md` header format).

## What's solid

- The roster and hook facts are exact: 13 agents, `frontend-engineer` unguarded, the exploitation
  guard is `security-expert` only, and the dispatch guards and repo-bounded writes are on
  `teco`/`tico` only.
- The ledger section matches `teco.md` step 2/4: the three triggers, the reactive pause trigger,
  `paused` reusing the Deliverable column, resuming the same agent by its id, no deadline, and a
  paused unit stalling only itself.
- The milestone-close split is right: `teco` flips and lists, the human applies after checking
  `HISTORY.md`.
- The `tico` boundary is right: it never calls an implementer, hands over the whole remaining
  chain, consults are announce-then-proceed, and handoffs wait for your OK.
- The environment claims match `ensure-environment.sh`: additive only, never restarts, CPG
  freshness is report-only, and the `/mcp` reconnect FAQ is correct.
- Mid-run escalation criteria match teco's brief text (scope / irreversible / substantial
  downstream waste).
- Tone and diagrams suit the audience. The walkthroughs are task-shaped and avoid internals.

## Open questions

- None that block. A `claude/docs/test-reports/agent-team-report.md` (qa-engineer's half) appeared
  during this review. Fold both halves together before the manual's commit, per `tico.md`
  "Commit at document boundaries".
