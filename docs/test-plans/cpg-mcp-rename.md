# CPG MCP server/tool rename — Test Plan

> **Status:** archived · **Owner:** `qa-engineer` · **Tracks:** C-605 (M6)

## 1. Scope & objective

Acceptance pass (unit U7 of `docs/plans/cpg-mcp-rename-coordination.md`, step 4/C-605 of
`docs/plans/cpg-mcp-rename.md` §4) for the `cpg-mcp-rename` delivery (M6) — verifying **at
behavior/acceptance altitude** that the four implementation units (steps 1/2/3a/3b, commits
`e00b9f6`, `59a03c4`, `acecb34`, `cd4142f`) actually produce AC-1…AC-6 from
`docs/requirements/cpg-mcp-rename.md`, by driving the live/relocated `cypher-mcp/` server, the
live FalkorDB instance, the `claude`/`git`/`docker` CLIs, and the delivered documentation
directly — not by re-reading the plan or trusting the prior gates' self-reports.

Everything upstream of this pass is already gated and not re-litigated here:
- Plan gate (`docs/reviews/cpg-mcp-rename.md`, 2 passes, final verdict **approve with
  suggestions**) — the discovery/sweep mechanism's design (§3.2's widened `git grep`) and the
  exact string-mapping table (§3.3).
- `teco`'s independent spot-check of each of the four implementation units before commit (file
  counts, grep sweeps, JSON parses per the coordination ledger).

What none of those gates could confirm — because none of them drove the actually-relocated,
actually-rebuilt server or exercised the live Claude Code MCP wiring end to end — is whether the
delivered rename survives contact with the running system. That is this document's one job.

## 2. References

- `docs/requirements/cpg-mcp-rename.md` — AC-1…AC-6 (source of test items below), FR-1…FR-7 for
  rationale.
- `docs/plans/cpg-mcp-rename.md` (Version 1.1) §3.2 (discovery/sweep + classification rules),
  §3.3 (exact string mappings), §4 (step table, step 4/C-605's done-condition), §5 (the adopted
  per-AC test strategy table — this plan expands each row into concrete test items).
- `docs/reviews/cpg-mcp-rename.md` — plan-gate history (2 passes, approve with suggestions; not
  re-litigated).
- `docs/plans/cpg-mcp-rename-coordination.md` — unit ledger, U1…U6 delivered scope.
- Delivered artifact under live test: `cypher-mcp/server.py` (via the running container and the
  `claude` CLI, not read directly for its own sake — this pass drives it).
- `cypher-mcp/README.md` — the container debug recipe (§"Container debug recipe") and the
  documented offline/in-container test-count baselines.

## 3. CPG relevance check (per this agent's own standing orientation)

Live-checked before writing test items, not assumed: a probe against a deliberately nonexistent
graph name via `mcp__cpg__query` (the still-live, pre-restart session tool — see §4) returns
`Loaded graphs: ws:test, cpg_falkorchat, reference, ws:qa-tico-workflows-manual, ws:acme,
cpg_salesperson, ws:eval, kaizen_graph_dba` — identical to the set the plan's own §"CPG:" line and
the plan-gate review both recorded. None of these represents this repo's own docs/scripts/prompts.

`CPG: considered, not relevant — this delivery is a text/config/build-script identity rename
across the monorepo's own docs, agent prompts, and MCP plumbing, not a question about application
code semantics (matching the plan's and the plan-gate review's own CPG line for this component).
Confirmed live via the probe above that no loaded graph represents this repo's own
docs/scripts/prompts — the CPGs that exist (cpg_falkorchat, cpg_salesperson) are for
falkor-chat/salesperson application code, unrelated to this rename's subject matter. A
call-graph/data-flow tool has nothing to offer a rename verification that is entirely `git grep`,
direct file reads, and driving the running MCP server/CLI.` No freshness-marker check applies
since no CPG is in use for this pass's own reasoning.

## 4. Live-environment grounding (confirmed before writing test items)

- **Known constraint, confirmed live, not just assumed:** this QA session's own interactive tool
  surface still resolves the *old* tool name — `mcp__cpg__query` — and a live call against it
  (`MATCH (m:METHOD) RETURN count(m) AS n` on `cpg_falkorchat`) succeeds, returning `n=2915`. This
  is **session staleness, not a shipped-config defect**: the session's stdio connection to the MCP
  server process was established before `.mcp.json`/`.claude/settings.json` were edited, and that
  long-lived process keeps answering independent of what's now on disk. Confirmed structurally,
  not just asserted: `cpg/mcp/docker-run.sh` no longer exists on disk (`git status`/`ls` — the
  directory is gone, relocated to `cypher-mcp/`), so the only way this session's `mcp__cpg__query`
  call can still be answering is a still-running process from before the rename, not a live
  re-resolution of the current config.
- **Found a way around the constraint the brief anticipated:** `claude mcp list` / `claude mcp
  get <name>`, run via `Bash` as fresh, independent CLI invocations, are **not** bound to this
  interactive session's stale MCP client state — each spawns its own health check against the
  *current* `.mcp.json` from scratch. This turned AC-3/AC-6's live checks from "may be
  session-blocked" into fully live-verifiable, confirmed empirically before writing the AC-3/AC-6
  test items below (§6).
- `cypher-mcp/.venv` already exists (pre-built) and `cypher-mcp:dev`/`cypher-mcp:test` images
  already exist in the local Docker image store, alongside old `cpg-mcp:*`-labeled images left as
  inert history (no collision — confirmed both namespaces coexist harmlessly, per the plan's own
  §3.5 prediction).
- Git HEAD at pass time: `cd4142f` (step 3b, the last implementation commit).
  `docs/plans/cpg-mcp-rename-coordination.md` shows as modified-but-uncommitted (ledger bookkeeping
  only, not a deliverable file). The pre-rename baseline for AC-1's `git diff`/`git grep` intent is
  `63384fb` (the plan-gate-approval commit, immediately before step 1's `e00b9f6`).

## 5. Risk assessment & coverage strategy

The highest risk at this altitude is exactly what no prior gate could see: whether the **widened**
discovery sweep (§3.2's `git grep` pattern, the plan-gate review's own B1 fix) actually leaves
**zero unexplained tool-identity hits** when run fresh against the committed tree — not the
pattern's design (already reviewed), but its live output — and whether the **relocated, rebuilt**
server genuinely answers identically to before, which only a live protocol-level call can show.

**Coverage decisions, explicit:**
- AC-1 gets the deepest test item — the full widened sweep, re-run fresh, every surviving hit
  individually triaged against the plan's own classification rules (§3.2), not sampled. This
  mirrors the plan's own §5 sizing note ("expect this list to be real triage volume, not a short
  one").
- AC-2, AC-4 are cheap, deterministic static checks — one test item each.
- AC-3, AC-6 get two independent live angles each (a raw container-level JSON-RPC probe bypassing
  the stale session entirely, plus a fresh `claude mcp list`/`get` CLI invocation) precisely
  because the brief flagged these as the two criteria most at risk from session staleness — worth
  the extra angle to close that risk for real rather than report it as blocked by default.
- AC-5 is a single `git diff` against the pre-rename baseline, scoped to the FR-7-protected paths
  plus a repo-wide `cpg_<component>` literal check — cheap and conclusive.
- The regression floor (offline suite count) and two flagged residual-risk spot-checks (the
  README's stale example count, the deliberately-preserved `generic-cypher-mcp2.md` judgment call)
  are included as their own test items since the brief specifically asked for them, even though
  they are not separately-numbered ACs.

**Deliberately not tested (and why):**
- No re-review of the plan-gate's own findings (B1/M1/minors) — already independently reproduced
  twice by `analyst` (2 passes), not this pass's job to re-litigate.
- No fuzzing of `authorize_write()`'s write-path mechanics — out of scope for this delivery per
  the requirements doc's own Out of scope ("Any change to the tool's actual behavior... is
  unaffected"); the write path was already proven live during the M5 (`generic-cypher-mcp`)
  acceptance pass and this delivery makes zero behavioral changes to it.
- No exhaustive read of all ~120 files the widened sweep matches — the triage in TP-001 covers
  every survivor after mechanical exclusion (archived docs, `docs/archive/`, the `cpg-mcp-rename*`
  family, confirmed CPG-domain vocabulary), which is the sweep's own designed proof gate, not a
  sample of it.

## 6. Test items

### TP-001 — AC-1: widened repo-wide sweep, every surviving hit triaged

**Preconditions:** Git HEAD at `cd4142f` or later (all four implementation units committed).

**Steps:** (a) `git grep -c 'mcp__cpg__query' -- . ':!.git'` (the literal AC-1 wording). (b) The
full widened discovery pattern from `docs/plans/cpg-mcp-rename.md` §3.2:
`git grep -zlE 'mcp__cpg__query|cpg/mcp|"cpg"|CPG_MCP_|cpg-mcp|cpg_mcp_|\bcpg\b' -- . ':!.git'`.
(c) Every file the widened pattern matches is classified per §3.2's ordered rules: skip if under
`docs/archive/`; skip if `docs/{requirements,plans,reviews,test-plans,test-reports}/` with a
basename starting `cpg-mcp-rename`; skip if the file carries `Status: archived`; for every
remaining hit, read the actual line(s) and classify tool-identity (must be renamed) vs.
CPG-domain vocabulary (must stay) vs. legitimate "renamed-from-X" historical narration in an
active living log or the M6 backlog section (surgical-edit rule, §3.2 rule 4 — leave as written).

**Expected result:** (a) Zero hits outside archived/family documents. (b)/(c) Every surviving hit
in a non-archived, non-family file resolves to CPG-domain vocabulary or legitimate historical
narration — no residual, unaddressed old-identity reference in any active document, agent prompt,
or skill. Any hit that is none of these is a defect.

**Priority:** Critical. **Type:** Acceptance (static, full sweep + manual triage).

### TP-002 — AC-2: `.mcp.json` server key

**Preconditions:** None beyond repo access.

**Steps:** Direct read of `.mcp.json`; confirm the `mcpServers` object's only key.

**Expected result:** Key is `"cypher"`; no `"cpg"` key present anywhere in the file.

**Priority:** High. **Type:** Static, direct read.

### TP-003 — AC-3: live call against the relocated/rebuilt server, two independent angles

**Preconditions:** `cypher-mcp:dev` image present (or buildable via `cypher-mcp/build.sh`);
FalkorDB reachable.

**Steps:** (a) **Container-level, bypassing Claude Code's MCP client entirely:** the README's
"Container debug recipe" — `docker run --rm --add-host=host.docker.internal:host-gateway
cypher-mcp:dev python -c "import server; print(server.run_query('cpg_falkorchat', 'MATCH
(m:METHOD) RETURN count(m) AS n'))"`. (b) **Full protocol probe:** pipe a JSON-RPC
`initialize`/`tools/list`/`tools/call` sequence at `cypher-mcp/docker-run.sh` directly (README's
own recipe), inspecting `serverInfo.name`, the `tools/list` result, and the `tools/call` result
for the same query. (c) Compare both results against the stale-session `mcp__cpg__query` call
already run for §4's grounding (`n=2915`, pre-rename tool name, same query).

**Expected result:** (a) Returns a real row count via the relocated code path, not an error. (b)
`serverInfo.name` reads `"cypher"`; `tools/list` shows exactly one tool, `query`; its
`tools/call` result matches (a). (c) The renamed path's result is identical to the stale
old-named session's result — same query, same graph, same answer — proving behavioral parity
across the rename, not just that *a* server answers.

**Priority:** Critical. **Type:** Acceptance (live, two independent protocol-level angles).

### TP-004 — AC-4: spot-check active docs/agent prompts for the new tool name

**Preconditions:** None beyond repo access.

**Steps:** `grep` each of: `claude/AGENTS.md`, `skills/cpg-analysis/SKILL.md` (frontmatter
`allowed-tools:` + body mentions), and the `tools:`/body text of the six CPG-consuming agents'
own prompts (`claude/analyst/analyst.md`, `claude/architect/architect.md`,
`claude/qa-engineer/qa-engineer.md`, `claude/coder/coder.md`,
`claude/tdd-engineer/tdd-engineer.md`, `claude/frontend-engineer/frontend-engineer.md`) for
`mcp__cypher__query` vs. `mcp__cpg__query`. For any of the six with no `tools:` line and no
literal mention, additionally check `git log` on that file for whether it ever named the tool
literal (to distinguish "correctly renamed" from "never named it, nothing to rename").

**Expected result:** `claude/AGENTS.md` and `skills/cpg-analysis/SKILL.md` read
`mcp__cypher__query` throughout, zero `mcp__cpg__query` residue. `analyst`/`architect` (which
carry an explicit `tools:` allowlist) show `mcp__cypher__query` in that line. The other four
agents either show the new name or — if they never explicitly named the tool literal in their own
prompt history (relying on unrestricted tool access instead) — show nothing to rename, which is
not a defect.

**Priority:** High. **Type:** Static, targeted reads + one `git log` check per file with no hit.

### TP-005 — AC-5: FR-7-protected paths, diff-clean

**Preconditions:** Pre-rename baseline commit `63384fb` available in history.

**Steps:** (a) `git diff 63384fb -- skills/cpg-analysis/ skills/joern-cpg/ claude/graph-dba/`. (b)
Repo-wide `git diff 63384fb -- . ':!.git' | grep -E '^[+-].*cpg_[a-z]+'` to catch any
`cpg_<component>` graph-name literal change anywhere, not just in the three named paths.

**Expected result:** (a) The only changes present are tool-identity string substitutions
(`mcp__cpg__query`→`mcp__cypher__query`, `cpg/mcp`→`cypher-mcp` path references, `` `cpg` MCP
tool``→`` `cypher` MCP tool``) — no change to CPG-domain mechanics, the `cpg-analysis`/`joern-cpg`
skill descriptions' own subject matter, or `graph-dba`'s pipeline docs. (b) Every matched line's
`cpg_<component>` literal itself (`cpg_falkorchat`, `cpg_salesperson`) is unchanged on both sides
of every diff hunk — only adjacent tool-identity text changed.

**Priority:** High. **Type:** Static diff review.

### TP-006 — AC-6: old tool name unavailable, two independent live angles

**Preconditions:** Same as TP-003.

**Steps:** (a) `claude mcp list` (fresh CLI process, not the interactive session). (b) `claude mcp
get cpg` and `claude mcp get cypher`. (c) TP-003(b)'s protocol probe's `tools/list` result,
re-inspected for any tool/server literally named `cpg`.

**Expected result:** (a) Exactly one server, `cypher`, `✔ Connected` — no `cpg` entry. (b) `claude
mcp get cpg` fails with a "no such server" message (exit ≠ 0), listing `cypher` as the only
configured server; `claude mcp get cypher` succeeds, `✔ Connected`. (c) No `cpg`-named tool or
server anywhere in the protocol surface.

**Priority:** Critical. **Type:** Acceptance (live, two independent fresh-process angles).

### TP-007 — Regression floor: offline suite count unchanged

**Preconditions:** `cypher-mcp/.venv` present (or created via `cypher-mcp/setup.sh`).

**Steps:** `cypher-mcp/.venv/bin/pytest tests -q` from the new location.

**Expected result:** `84 passed, 7 deselected` — identical to the pre-rename baseline recorded in
the plan (§2/§5) and confirmed by the plan-gate review's own independent run. Any delta is a
regression, not just a naming issue.

**Priority:** Critical. **Type:** Automated suite run (not new tests — the existing suite,
establishing the green baseline this pass builds on).

### TP-008 — Residual risk: README's stale in-container example count

**Preconditions:** `cypher-mcp:test` image present (or buildable).

**Steps:** (a) `grep -n '74 passed\|75 passed' cypher-mcp/README.md`. (b) `docker run --rm
cypher-mcp:test python -m pytest tests -q` — the actual in-container figure. (c) `git log -p` on
the README's example-output line across its history (both under `cpg/mcp/README.md` and
`cypher-mcp/README.md`) to confirm when the drift was introduced.

**Expected result:** (a) README's example gate output reads "74 passed, 7 deselected". (b) Actual
in-container run produces "75 passed, 7 deselected" — a real, pre-existing one-test drift. (c)
`git log -p` shows the line was already "74 passed" before this delivery's own commits (`e00b9f6`
onward only changed `cpg-mcp:test`→`cypher-mcp:test` in the same line, not the count) — confirming
the drift is genuinely pre-existing and not introduced by this rename.

**Priority:** Medium (flagged residual risk, not a blocking AC). **Type:** Static + live count +
historical diff.

### TP-009 — Residual risk: `generic-cypher-mcp2.md`'s deliberately-preserved judgment call

**Preconditions:** None beyond repo access.

**Steps:** Read `docs/requirements/generic-cypher-mcp2.md` lines ~144–148 (the "Renaming the MCP
server/tool itself" out-of-scope bullet) and ~267–273 (the decision-log entry recording the
stakeholder's original naming remark) in full context.

**Expected result:** Both passages read sensibly as historical/contextual record of the *state at
the time this M6 document was written* (before the rename shipped) — not as a stray miss. Neither
claims the tool is *currently* named `cpg` in a way that would mislead a reader of the *current*
document; both are explicitly framed as "at the time," tracking this very rename as a named
follow-on.

**Priority:** Medium (flagged residual risk, not a blocking AC). **Type:** Static read, contextual
judgment.

## 7. Environment & data setup

- No environment bring-up needed — FalkorDB (`falkordb-dev`) is already running with the same
  graphs the plan-gate review recorded; `cypher-mcp/.venv` and `cypher-mcp:dev`/`:test` images are
  already built from a prior step.
- No destructive operation anywhere in this plan — every test item is a read, a static diff, a
  `git grep`/`git log`, a suite run, or a live query against `cpg_falkorchat` (`MATCH ... RETURN
  count(...)`, no write). TP-003/TP-006's container probes are ephemeral (`docker run --rm`),
  leaving no residue.
- This pass does not restart the interactive Claude Code session — §4 documents the workaround
  found (fresh `claude mcp list`/`get` CLI calls) that makes AC-3/AC-6 fully live-verifiable
  without one.

## 8. Entry/exit criteria

**Entry:** Plan gate closed (`docs/reviews/cpg-mcp-rename.md`, 2 passes, verdict **approve with
suggestions**); steps 1/2/3a/3b all committed (`e00b9f6`, `59a03c4`, `acecb34`, `cd4142f`); live
FalkorDB connection confirmed (§4, done).

**Exit:** All nine test items (TP-001…TP-009) executed and recorded pass/fail/blocked with
evidence in the test report. Any AC where observed live behavior diverges from the requirement is
filed as a defect, severity by user/stakeholder impact.

## 9. Explicitly out of scope

- Re-running or re-reviewing the plan-gate's own findings (B1/M1/m1/m2) — independently
  reproduced twice already by `analyst`; this pass verifies the *shipped result*, not the design
  process that produced it.
- Any test of the write-path mechanics (`authorize_write()`, curator-clear semantics) — zero
  behavioral change in this delivery (requirements doc's own Out of scope), already proven live
  during the M5 acceptance pass.
- A full read of every one of the ~120 files the widened sweep matches — TP-001's triage covers
  every survivor after the mechanical exclusions, which is the sweep's own designed completeness
  proof, not a sample.
- Restarting the interactive Claude Code session to force a same-session `mcp__cypher__query`
  call — unnecessary once §4's CLI-based workaround is confirmed to close the same risk live.
