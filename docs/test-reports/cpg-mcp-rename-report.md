# CPG MCP server/tool rename — Test Report

> **Status:** archived · **Owner:** `qa-engineer` · **Tracks:** C-605 (M6)

## Summary

Acceptance pass (unit U7 of `docs/plans/cpg-mcp-rename-coordination.md`, step 4/C-605 of
`docs/plans/cpg-mcp-rename.md`) against `docs/test-plans/cpg-mcp-rename.md` (TP-001…TP-009),
executed 2026-08-19. Target: git HEAD `cd4142f` (all four implementation units — `e00b9f6`,
`59a03c4`, `acecb34`, `cd4142f` — committed; `docs/plans/cpg-mcp-rename-coordination.md` shows as
modified-but-uncommitted, ledger bookkeeping only, not a deliverable file). Pre-rename baseline
for diff/grep comparisons: `63384fb` (the plan-gate-approval commit, immediately before step 1).

All nine test items were exercised **live or by direct static verification**: a full repo-wide
`git grep` sweep with every surviving hit individually triaged, two independent live protocol-level
probes against the relocated/rebuilt `cypher-mcp` server (a raw container-level JSON-RPC exchange
and fresh `claude mcp list`/`get` CLI invocations — both bypass this session's stale MCP-client
binding), the offline pytest suite run from the new location, a `git diff` against the pre-rename
baseline scoped to every FR-7-protected path, and direct reads of the two flagged residual-risk
passages. No file was read as a substitute for driving the actual grep/build/CLI/protocol
surfaces.

**Known-limitation note, resolved better than expected:** the brief anticipated AC-3/AC-6 might be
session-blocked (this interactive session predates the `.mcp.json`/`.claude/settings.json` edits,
and its own `mcp__cpg__query` tool call still resolves and returns live data — confirmed, see
TP-003/§"AC-3 detail" below, a genuine artifact of session staleness, not a shipped-config defect).
This pass found a way around it: `claude mcp list` / `claude mcp get <name>`, run as fresh `Bash`
subprocesses, are independent CLI invocations that read the *current* `.mcp.json` from scratch and
are not bound to the interactive session's stale connection. Combined with a raw container-level
JSON-RPC probe (bypassing Claude Code's MCP client entirely), both AC-3 and AC-6 are **fully live-
verified in this report**, not deferred to a post-restart follow-up.

`CPG: considered, not relevant — this delivery is a text/config/build-script identity rename
across the monorepo's own docs, agent prompts, and MCP plumbing, not a question about application
code semantics. Confirmed live via a probe against a deliberately nonexistent graph name (using
the still-live, pre-restart session tool): loaded graphs are ws:test, cpg_falkorchat, reference,
ws:qa-tico-workflows-manual, ws:acme, cpg_salesperson, ws:eval, kaizen_graph_dba — none represents
this repo's own docs/scripts/prompts (the CPGs that exist are for falkor-chat/salesperson
application code). A call-graph/data-flow tool has nothing to offer a rename verification that is
entirely git grep, direct file reads, and driving the running MCP server/CLI — matching the plan's
and the plan-gate review's own CPG line for this exact component.`

**Overall verdict: PASS, with one low-severity defect found and two confirmed-benign residual
risks.** AC-1…AC-6 all hold under live/static exercise. The regression floor holds exactly
(84 passed/7 deselected, unchanged). One real, minor, comment-only defect was found outside the
plan's own file-list scope (stale `cpg/mcp/` path references in `mcp-monitor/`'s source comments —
not docs, not caught by the plan's step 3b file list, no functional impact). Both flagged
residual-risk items (the README's stale "74 passed" example, the `generic-cypher-mcp2.md`
judgment call) were confirmed as claimed — genuinely pre-existing drift and a sensible deliberate
preservation, respectively, neither a defect.

## Results table

| ID | AC | Result | Evidence |
|---|---|---|---|
| TP-001 | AC-1 | **PASS (1 defect found, see Defects)** | Literal check: `git grep -c 'mcp__cpg__query'` → 2 surviving hits outside archived/family docs (`docs/BACKLOG.md:50`, `docs/requirements/generic-cypher-mcp2.md:145-146`), both confirmed legitimate "renamed-from-X" historical narration (BACKLOG's M6 milestone-map row; the M6 doc's own out-of-scope bullet, see TP-009) — not defects. Widened sweep: 122 files matched `mcp__cpg__query\|cpg/mcp\|"cpg"\|CPG_MCP_\|cpg-mcp\|cpg_mcp_\|\bcpg\b`; after mechanically excluding `docs/archive/`, `Status: archived` docs, and the 4-member `cpg-mcp-rename*` family, narrowed the identity-pattern subset to 16 candidate hits and the bare-`\bcpg\b` subset to 11 additional candidates — every one individually read and triaged. All resolved to CPG-domain vocabulary or legitimate historical narration **except** 5 lines in 3 `mcp-monitor/` source files, filed as a defect below. |
| TP-002 | AC-2 | **PASS** | Direct read of `.mcp.json`: `"mcpServers": {"cypher": {...}}` — `cypher` is the only key, `args` points at `$CLAUDE_PROJECT_DIR/cypher-mcp/docker-run.sh`. No `"cpg"` key anywhere in the file. |
| TP-003 | AC-3 | **PASS (live, two angles)** | (a) Container debug recipe: `docker run --rm --add-host=host.docker.internal:host-gateway cypher-mcp:dev python -c "import server; print(server.run_query('cpg_falkorchat','MATCH (m:METHOD) RETURN count(m) AS n'))"` → `graph=cpg_falkorchat · rows=1 · 8.4ms` / `n=2915`. (b) Full JSON-RPC protocol probe piped at `cypher-mcp/docker-run.sh`: `initialize` → `"serverInfo":{"name":"cypher","version":"1.28.1"}`, instructions text opens `"The \`cypher\` server exposes a single tool, \`query\`..."` and cites `CYPHER_MCP_CURATOR_AGENTS` (confirms the plan-gate's B1 fix landed in the real running code, not just docs); `tools/list` → exactly one tool, `query`; `tools/call` on the same query → `n=2915`, `0.7ms`. (c) Parity check: this session's stale `mcp__cpg__query` call (same query, same graph) also returned `n=2915` — identical result across old/new names, confirming behavioral parity, not just that *a* server answers. |
| TP-004 | AC-4 | **PASS** | `claude/AGENTS.md` and `skills/cpg-analysis/SKILL.md` (frontmatter `allowed-tools:` + 3 body mentions) read `mcp__cypher__query` throughout, zero `mcp__cpg__query` residue. `claude/analyst/analyst.md:4` and `claude/architect/architect.md:4` both carry `tools: ..., mcp__cypher__query`. The other four agents (`qa-engineer`, `coder`, `tdd-engineer`, `frontend-engineer`) carry no `tools:` allowlist and no literal tool-name mention in their current prompt text; `git log -p` on each confirms **zero** historical occurrences of `mcp__cpg__query` ever in any of the four files — nothing to rename, not a gap (these four rely on unrestricted tool access + the `cpg-analysis` skill reference, which correctly stays untouched per FR-7). |
| TP-005 | AC-5 | **PASS** | `git diff 63384fb -- skills/cpg-analysis/ skills/joern-cpg/ claude/graph-dba/` → 7 files changed, 21/21 insertions/deletions, every hunk a pure tool-identity substitution (`mcp__cpg__query`→`mcp__cypher__query`, `cpg/mcp/README.md`→`cypher-mcp/README.md` path citations, `` `cpg` MCP tool``→`` `cypher` MCP tool``) — zero change to CPG-domain mechanics, skill descriptions' own subject matter, or `graph-dba`'s pipeline docs. Repo-wide `git diff 63384fb | grep -E '^[+-].*cpg_[a-z]+'` → every matched hunk shows the `cpg_falkorchat`/`cpg_salesperson` literal itself unchanged on both sides; only adjacent tool-identity text changed. |
| TP-006 | AC-6 | **PASS (live, two angles)** | (a) `claude mcp list` (fresh CLI process) → `cypher: bash -c exec "$CLAUDE_PROJECT_DIR/cypher-mcp/docker-run.sh" - ✔ Connected` — exactly one server, no `cpg` entry. (b) `claude mcp get cpg` → `No MCP server named "cpg". Configured servers: cypher` (exit 1). `claude mcp get cypher` → `Status: ✔ Connected`. (c) TP-003(b)'s `tools/list` result carries no `cpg`-named tool or server anywhere in the protocol surface. |
| TP-007 | Regression floor | **PASS** | `cypher-mcp/.venv/bin/pytest tests -q` from the new location → `84 passed, 7 deselected, 1 warning in 0.86s` — exact match to the pre-rename baseline recorded in the plan and the plan-gate review's own independent run. |
| TP-008 | Residual risk (not blocking) | **CONFIRMED, not a defect of this delivery** | `cypher-mcp/README.md:489` reads `# 74 passed, 7 deselected`. Live in-container run (`docker run --rm cypher-mcp:test python -m pytest tests -q`) → actual `75 passed, 7 deselected, 1 warning in 0.75s` — real 1-test drift, confirmed. `git log -p` on the line's history: it read "74 passed" (jumped from "53 passed") in a commit predating this delivery entirely; this delivery's own commit (`e00b9f6`) only changed `cpg-mcp:test`→`cypher-mcp:test` in that same line, never touched the count. Pre-existing, not introduced by this rename. |
| TP-009 | Residual risk (not blocking) | **CONFIRMED, reads sensibly** | `docs/requirements/generic-cypher-mcp2.md` lines 144-148 (out-of-scope bullet: *"Renaming the MCP server/tool itself (`cpg` → something... the tool is currently `mcp__cpg__query`)... the stakeholder chose to track it as its own, separate follow-on"*) and lines 267-273 (decision-log entry recording the original stakeholder remark that seeded this very delivery) both read as accurate historical/contextual record of the state *at the time this M6 document was written* — explicitly framed as "raised in this session," "currently," "at the time" — not a stray miss and not misleading to a current reader. |

## AC-1 detail — the mcp-monitor defect

TP-001's widened sweep (122 files matched) was narrowed by mechanical exclusion (archived docs,
`docs/archive/`, the 4-member `cpg-mcp-rename*` family) to two candidate sets, each read in full:

1. The 6-axis identity pattern (`mcp__cpg__query|cpg/mcp|"cpg"|CPG_MCP_|cpg-mcp|cpg_mcp_`) minus
   excluded files → 16 lines. 13 of 16 resolved cleanly (historical narration in `AGENTS.md`,
   `docs/BACKLOG.md`, kaizen inbox entries; the confirmed-benign `generic-cypher-mcp2.md` judgment
   call; CPG-domain `--graph default="cpg"` in the `joern-cpg` pipeline scripts, correctly
   untouched per FR-7). **3 files, 5 lines, did not resolve cleanly:**
   - `mcp-monitor/fake_mcp_server/server.py:17` — `` Modeled on `cpg/mcp/server.py`'s mechanics... ``
   - `mcp-monitor/fake_mcp_server/server.py:50` — `` (same posture as `cpg/mcp/server.py`). ``
   - `mcp-monitor/pyproject.toml:7` — `` # pyproject.toml, cpg/mcp/requirements.txt): both existing MCP integrations in ``
   - `mcp-monitor/setup.sh:9` — `` # Mirrors cpg/mcp/setup.sh. ``
   - `mcp-monitor/setup.sh:17` — `` with falkor-chat/server/.venv or cpg/mcp/.venv) — untracked, the repo-root ``
2. The bare-word `\bcpg\b` subset, filtered to remove known-safe substrings (`cpg-analysis`,
   `cpg_<component>` literals, `cpg-model`, `.cpg-artifacts`, `CPG:` convention line, "Code
   Property Graph," and the topic-slug family names already excluded) → 11 lines, all confirmed
   CPG-domain (top-level `cpg/` directory identity, unrelated `cpg-test-gap` naming, an unrelated
   prior doc-sweep's own history, `/tmp/cpg-src/` Joern build path) or historical narration
   already covered above (`docs/BACKLOG.md`'s M6 section itself, `generic-cypher-mcp2.md`). No
   further defects found in this subset.

The 5 lines in `mcp-monitor/`'s source comments are genuine stale `cpg/mcp/`-path references —
confirmed by direct comparison against `mcp-monitor/AGENTS.md:50,54,56` and
`mcp-monitor/README.md:141`, which correctly read `cypher-mcp/` today (step 3b did update these).
`mcp-monitor/fake_mcp_server/server.py`, `mcp-monitor/pyproject.toml`, and `mcp-monitor/setup.sh`
are **not** docs — they're source/config files with design-rationale comments — and none of them
appears in `docs/plans/cpg-mcp-rename.md` §4's step 3b Files column
(`mcp-monitor/{AGENTS,README}.md`, `mcp-monitor/docs/{BACKLOG,HISTORY}.md` only). Filed as a
defect below.

## Defects

### D-1 — Stale `cpg/mcp/` path references survive in three `mcp-monitor/` source-comment files

**Severity: Low.** Comment-only (design-rationale prose referencing a directory that no longer
exists), no functional/behavioral impact — `mcp-monitor` does not import or execute anything from
the old path, it only cites it as design precedent in a comment. Inconsistent with the sibling
`mcp-monitor/AGENTS.md`/`README.md`, which were correctly updated in step 3b, creating a minor
"half-renamed" impression for a reader of the source comments specifically (the exact class of
confusion FR-6 exists to prevent, just at low stakes here since it's prose, not code).

**Steps to reproduce:**
1. `grep -n 'cpg/mcp' mcp-monitor/fake_mcp_server/server.py mcp-monitor/pyproject.toml mcp-monitor/setup.sh`

**Expected:** No reference to the old `cpg/mcp/` path remains anywhere in the active tree outside
the `cpg-mcp-rename*` document family and archived documents (per AC-1's intent and the plan's
own proof-gate language in §3.2: *"any other surviving hit is a defect"*).

**Actual:**
```
mcp-monitor/fake_mcp_server/server.py:17:Modeled on `cpg/mcp/server.py`'s mechanics (a `FastMCP` stdio server built the
mcp-monitor/fake_mcp_server/server.py:50:    reconnected mid-session (same posture as `cpg/mcp/server.py`).
mcp-monitor/pyproject.toml:7:# pyproject.toml, cpg/mcp/requirements.txt): both existing MCP integrations in
mcp-monitor/setup.sh:9:# smoke test. Mirrors cpg/mcp/setup.sh.
mcp-monitor/setup.sh:17:# with falkor-chat/server/.venv or cpg/mcp/.venv) — untracked, the repo-root
```

**Root cause (as observed, not fixed by this pass):** the plan's step 3b (§4's Files column) named
only `mcp-monitor/{AGENTS,README}.md` and `mcp-monitor/docs/{BACKLOG,HISTORY}.md` — it never
enumerated `mcp-monitor/`'s own source/config files, and the widened `git grep` sweep that *would*
have caught these (§3.2) was, per this delivery's own step 3a/3b done-conditions, scoped to
"`claude/`or `skills/`" (3a) and the named docs list (3b) — neither globs into
`mcp-monitor/fake_mcp_server/`, `mcp-monitor/pyproject.toml`, or `mcp-monitor/setup.sh`. This is
structurally the same shape of gap the plan-gate review's own Pass 2 flagged as a
non-blocking observation for `claude/docs/requirements/security-expert.md` (a file outside every
step's Files-column glob, saved there only by a broader done-condition wording) — here, no
broader done-condition happened to cover it.

**Recommendation:** A follow-up one-line-per-file fix (`cpg/mcp/server.py`→`cypher-mcp/server.py`,
`cpg/mcp/requirements.txt`→`cypher-mcp/requirements.txt`, `cpg/mcp/setup.sh`→`cypher-mcp/setup.sh`,
`cpg/mcp/.venv`→`cypher-mcp/.venv`), owned by `cobb` or `coder` per whichever precedent this repo
prefers for a small residual-sweep fix. Not blocking — no behavior depends on these comments, and
they're accurate as *historical* design-precedent citations even before the fix, just no longer
accurate as *current*-path citations.

## Coverage & gaps

**What this pass covered:**
- Every one of AC-1…AC-6, each with direct, first-party evidence (a full sweep + manual triage, a
  direct file read, two independent live protocol probes per AC-3/AC-6, a diff, a suite run).
- Both angles the brief flagged as most at risk from session staleness (AC-3, AC-6) — resolved
  fully live via a workaround (fresh CLI processes) rather than deferred to a post-restart
  follow-up, which is a stronger result than the brief's own worst-case expectation.
- The regression floor (offline suite) and both explicitly flagged residual-risk items (README
  drift, the `generic-cypher-mcp2.md` judgment call).
- A genuinely wider net than the plan's own step 3a/3b Files columns — the D-1 defect was found
  specifically *because* this pass re-ran the full widened sweep repo-wide rather than trusting
  the implementers' own file lists, which is exactly this pass's job per §1.

**What this pass did not cover, deliberately, and why that's an acceptable residual risk:**
- Any re-test of the write-path mechanics (`authorize_write()`, curator-clear) — zero behavioral
  change in this delivery (requirements doc's own Out of scope), already proven live during the
  M5 (`generic-cypher-mcp`) acceptance pass; re-testing here would duplicate that pass, not add
  information.
- A full manual read of all ~120 files the widened sweep matched — TP-001's triage covers every
  survivor after the mechanical exclusions (archived, family, `Status: archived`), which is the
  sweep's own designed completeness proof, not a sample; the one gap that slipped through (D-1)
  slipped through the *implementers'* file-list scoping, not this pass's triage — it was caught
  precisely because this pass re-ran the sweep at full repo width instead of trusting the plan's
  step 3a/3b Files columns.
- Restarting the interactive Claude Code session to get a literal same-session
  `mcp__cypher__query(...)` call — unnecessary once the CLI-based workaround (§Summary) closed the
  same live-verification need for AC-3/AC-6 without one. A future session, once naturally
  restarted, will show the new name in its own tool surface; no separate follow-up action is
  needed purely to force that.

## Feedback & recommendations

1. **File D-1 as a small follow-up** (owner's call — `cobb` for a doc/comment sweep, or `coder`
   for a source-file touch since two of the three files are `.py`/`.toml`, not markdown). Low
   severity, cheap fix, three files, five lines.
2. **Design observation for future wide-rename deliveries, not specific to this one:** this
   delivery's own step 3a/3b Files columns (like the plan-gate review's Pass-2 observation on
   `claude/docs/requirements/security-expert.md`) show the same recurring pattern — a
   hand-enumerated Files column drifting slightly narrower than the "re-run the grep, sweep
   everything it matches" done-condition the plan itself argues is the actual source of truth
   (`docs/plans/cpg-mcp-rename.md` §3.2's own words: *"the grep is the truth, not the file
   list"*). D-1 is exactly this pattern's second live instance in the same delivery (the first,
   `security-expert.md`, was caught by the plan-gate review before it could ship a gap; D-1 is a
   gap that *did* ship, just a low-severity one). Worth considering, for a future delivery of this
   shape: make the acceptance pass's own re-sweep (what TP-001 did here) an explicit part of every
   wide-sweep unit's own done-condition, not just the final acceptance gate — so a Files-column
   gap surfaces at step-commit time, not at the very end.
3. **The session-staleness workaround (fresh `claude mcp list`/`get` CLI calls bypass an
   interactive session's stale MCP-client binding) is worth capturing as a durable technique** for
   any future MCP-rename or MCP-reconfiguration acceptance pass run from within the same session
   that made the config edit — it turns what looked like an unavoidable BLOCKED verdict into a
   fully live one. (Captured in this agent's kaizen inbox per the standard learning-capture
   convention.)
4. **Recommend `teco` proceed to close milestone M6** — `docs/BACKLOG.md`'s C-601…C-605 items are
   all still marked 🔵 proposed even though C-601-604 are committed and C-605 (this report) now
   closes clean; flipping them to ✅ and writing `docs/HISTORY.md`'s dated close-out entry is
   `teco`'s milestone-close coordination per `docs/plans/cpg-mcp-rename.md` §3.6, not this pass's
   job to perform.

## Traceability

Plan: `docs/test-plans/cpg-mcp-rename.md` (TP-001…TP-009). Requirements:
`docs/requirements/cpg-mcp-rename.md` (AC-1…AC-6, FR-1…FR-7). Design:
`docs/plans/cpg-mcp-rename.md` (Version 1.1) §3.2 (sweep mechanism), §3.3 (mapping table), §5
(adopted per-AC strategy). Prior gates: `docs/reviews/cpg-mcp-rename.md` (plan gate, 2 passes,
approve with suggestions — B1/M1 blockers/majors independently re-verified closed in Pass 2, not
re-litigated here). Coordination: `docs/plans/cpg-mcp-rename-coordination.md`, unit U7 (this
report), step 4/C-605.
