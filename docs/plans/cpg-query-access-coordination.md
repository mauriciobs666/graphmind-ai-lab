# CPG query access — coordination log

> `teco` coordination doc for the **CPG query access** feature (MCP tool replaces
> `redis-cli GRAPH.QUERY` on the CPG **read** path).
> Requirements: [`../requirements/cpg-query-access.md`](../requirements/cpg-query-access.md) (Ready for design, no open questions).
> Component backlog: [`../BACKLOG.md`](../BACKLOG.md) · history: [`../HISTORY.md`](../HISTORY.md).
> Started 2026-07-24.

## Goal / definition of done

AC-1…AC-4 of the requirements are met live:
- AC-1 — cold agent session answers a CPG question in **one tool call** (graph name + Cypher as
  parameters), no shell quoting.
- AC-2 — multi-line Cypher accepted verbatim, same result as its single-line form.
- AC-3 — M2 acceptance queries reproduce their recorded numbers against `cpg_falkorchat`.
- AC-4 — `joern-cpg-pipeline.md` FR-9 updated to point here; no contradiction between the docs.

Plus the repo's doc convention: catalogs, skill, agent wiring, BACKLOG + HISTORY land with the change.

## Environment facts (verified by teco, 2026-07-24)

- FalkorDB up: container `falkordb-dev`, `falkordb/falkordb:v4.18.11`, port 6379.
- `GRAPH.LIST` → `cpg_falkorchat`, `cpg_salesperson`, `ws:acme`, `reference`, `ws:test`.
- No MCP servers configured anywhere yet: `~/.claude.json` global `mcpServers` = `[]`, project
  entry for this repo `mcpServers` = `[]`, and there is no `.mcp.json` in the repo.

## ⚠ Requirements defect found at decomposition (teco, 2026-07-24)

**AC-3 cites a stale number.** The requirements say "AC-8 test-gap = **30 untested methods**",
citing `m2-cpg-analysis-coordination.md`. That 30 was the *pre-correction* figure: analyst's
Gate-2a review found it wrong, and the corrected, re-verified figure is **39 rows / 32 distinct
method names** (see the coordination log's W2-fix entry, 2026-07-19, and the M2 entry in
`HISTORY.md`). AC-2 callers = **21** is correct.
→ Treat **39 rows / 32 distinct names** as the AC-3 test-gap baseline; the requirements doc's
AC-3 needs a correction (user/tico call — raised in teco's report).

## Inventory findings (Explore, 2026-07-24) — constraints on the design

**Runtime (decision-relevant):**
- `python3` = **3.12.3** system-wide; `pip` 24.0; **`uv` is NOT installed**.
- **`node` is NOT on PATH** on the Linux side — `npm`/`npx` resolve only to the Windows install via
  WSL interop. An `npx`-launched off-the-shelf MCP server is therefore **not** a clean option;
  effectively **Python-only** for a Linux-side server.
- `redis-cli` at `/usr/bin/redis-cli`.

**Prior art in-repo (closest reference implementation):**
- `falkor-chat/server/falkorchat/mcp.py` — a working **FastMCP** server (`mcp>=1.28,<1.29`,
  `falkordb>=1.6,<1.7`, Python ≥3.12) with tests (`test_mcp.py`, `test_mcp_client.py`) and a
  documented Streamable-HTTP client config (`falkor-chat/docs/DESIGN.md` §15, `:1053`).
- Known trap from its test report: `POST /mcp` 405s — only `/mcp/` (trailing slash) works
  (`falkor-chat/docs/archive/test-reports/m1-chat-mcp-report.md` DEF-1).
- `skills/joern-cpg/scripts/cpg-to-falkordb.py` is a **pure-stdlib RESP client** (no `redis`/
  `falkordb` dependency) — an existing in-repo pattern for talking to FalkorDB dependency-free.

**Scaffolding:** repo root has **no** Python/Node project (no `pyproject.toml`/`package.json`/venv).
`cpg/` exists as a **gitignore-only placeholder** (ignores `.cpg-artifacts/`, currently absent) —
the one repo-root location already carved out for CPG state.

**Harness wiring gaps:** no `.mcp.json` and no `mcpServers` key in any settings file in the repo;
OpenCode has **no** MCP wiring anywhere and `skills/agent-standards/` has **no OpenCode MCP
section** (a gap if OpenCode must be wired). `kiro/DESIGN.md:203-213` has an *illustrative*
(not live) Kiro `mcpServers` example. `skills/agent-standards/claude-code.md` (`:38`, `:53`,
`:58`, `:110-111`, `:168`) is the authoring source for Claude Code MCP specifics.

**Extra stale-doc hits beyond my first scan:**
- `skills/cpg-analysis/SKILL.md` — `:5` description, `:15` `allowed-tools`, `:34` `GRAPH.LIST`,
  `:43` connection block, `:48`, `:56-57` EXPLAIN/PROFILE, **`:92` the whole §3 "no `--param`
  binding, use literal substitution" warning — a shell-layer artifact that an MCP tool may make
  obsolete**.
- `skills/cpg-analysis/references/impact-analysis.md:13` — the **only** recipe with `redis-cli`
  framing (rca/code-review/test-gap carry raw Cypher only).
- `skills/README.md:19`, root `AGENTS.md:74-75`; **CORRECTION (teco, 2026-07-24)** — an earlier
  version of this doc said `claude/README.md` rows do *not* mention `cpg-analysis`. **Wrong**:
  rows **9 (architect), 16 (qa-engineer), 17 (analyst)** each carry a `cpg-analysis` clause and all
  three need updating. Caught by analyst's review.
- `claude/{analyst,architect,qa-engineer}/kaizen/history.md:16-17` — C-207 description edits;
  a new entry is due when the descriptions change again.
- `claude/tico/kaizen/inbox.md:19` already flags the FR-9 "chosen over MCP tool" contradiction.
- `claude/graph-dba/falkordb-quirks.md:159-165` — **`GRAPH.QUERY` materializes an empty key**;
  relevant to any read path, including an MCP tool (a typo'd graph name creates a graph).
- Historical records that will read inconsistently (low priority, do not rewrite history):
  `docs/HISTORY.md:10`, `docs/BACKLOG.md:66`, the M2 plan/review docs.

## Documentation-impact scan (curated by teco)

| Doc | Impact |
|---|---|
| `docs/requirements/joern-cpg-pipeline.md` | **FR-9** reversed (redis-cli → MCP tool) + decision-log entry — AC-4 |
| `docs/requirements/cpg-query-access.md` | AC-3 stale number (see above); status when delivered |
| `skills/cpg-analysis/SKILL.md` | §1 connection section + frontmatter `description` and `allowed-tools` (currently `Bash, Read`) |
| `skills/cpg-analysis/references/*.md` | recipes keep raw Cypher (decision log: not reshaped) — check only for `redis-cli` framing |
| `skills/README.md` | catalog row if the access mechanism/description changes |
| root `AGENTS.md` | `cpg-analysis` bullet mentions Cypher access; MCP server may need a mention |
| `claude/README.md` + `analyst`/`architect`/`qa-engineer` descriptions | CPG-capability lines reference the skill |
| `docs/BACKLOG.md` | new milestone + `C-3xx` items for this feature |
| `docs/HISTORY.md` | dated entry on delivery |

## Units

| # | Unit | Owner | Status |
|---|---|---|---|
| U1 | Design plan → `docs/plans/cpg-query-access.md` | architect | ✅ delivered (723 L) — **build**, not buy |
| U2 | Review gate on the plan → `docs/reviews/cpg-query-access.md` | analyst | ✅ **needs changes** — 3 blockers, 6 majors, 6 minors, 2 nits |
| U2b | Plan rework against the review | architect | ✅ delivered — 1,124 L, §10 rework log, S1–S10 |
| U2c | Re-gate of the reworked plan | analyst | ✅ **approve with suggestions** — 0 blockers, 2 majors, 5 minors, 2 nits |
| U2d | Plan patch: N-1 (S8 guard), N-2 (truncation), n-3 (comment-blind sniff) | architect | ✅ plan now **v2.1** — read S8/S2 at v2.1 |
| U3a | S1 venv/deps/`setup.sh` + smoke | devops | ✅ **teco-verified**: `cpg/mcp/.venv` imports mcp 1.28.1 + falkordb 1.6.2; PII grep clean |
| U3b | S2 `server.py` + `run.sh` + tests | coder | ✅ **teco-verified** (suite re-run + behaviour probed, see below) |
| U3c | S3 `.mcp.json` + settings + live connect (1 human approval) | devops | ⏸ blocked on S2 |
| U4 | S4/S5/S7 skill + agent + agent-standards surfaces | cobb | ✅ **teco-verified** (see below) |
| U5 | S6 requirements edits (FR-9 reversal, AC-1/AC-3 + n-7 decision log) | coder | ✅ **teco-verified** (see below) |
| U6 | S8 CPG rebuild + fresh baseline | joern | ✅ **teco-verified**: 110,048 nodes / 1,019 test methods / callers=21; all 5 graphs intact |
| U7 | S9 live acceptance AC-1…AC-4 + test plan/report | qa-engineer | ⏸ blocked on U3–U6 |
| U8 | S10 BACKLOG M3 (C-301…C-307 + C-308/309/310) + HISTORY | coder | ⏸ blocked on U7 |

## ✅ Decisions RULED by the stakeholder (2026-07-25)

- **D1 → rebuild approved.** User: *"dont worry about data in it, you can delete and recreate."*
  Destructive `GRAPH.DELETE` of `cpg_falkorchat` authorized. Rebuild from
  `falkor-chat/server/{falkorchat,tests}` **including tests** (today's graph has zero test methods).
  Per analyst Blocker 3 the M2 numbers are unreachable anyway (8 commits since) → **record fresh
  numbers as the new baseline**; AC-3 satisfied by **tool ≡ `redis-cli` equivalence** on that graph.
- **D2 → correct it.** 30 → 39 rows / 32 distinct names; superseded in practice by the D1 rebaseline.
- **D3 → direct callers for AC-1** (teco recommended, user deferred to the explanation). This feature
  changes *how Cypher is transmitted*, not how powerful Cypher is. The bounded transitive
  upward-closure query is **deferred to its own `C-3xx` backlog item owned by `graph-dba`**.
- **D4 → `EXPLAIN`-only; `PROFILE` removed** (teco recommended, user deferred). Read-only wins;
  `graph-dba` keeps `PROFILE` via `redis-cli`. Plan must state that `GRAPH.RO_QUERY` is what makes
  the main path safe.

**⚠ Ownership constraint fed into rework:** the plan assigned S6 (FR-9 reversal) to `teco` — **not
possible**: teco's `Write`/`Edit` is harness-restricted to `docs/plans/`. All doc-editing steps
reassigned to `cobb` (agent/skill surfaces, `claude/README.md`, kaizen histories) or `cobb`/`coder`
(requirements, BACKLOG, HISTORY). Nothing stays owned by teco.

## ⛔ Original decision framing (raised 2026-07-24, now resolved above)

- **D1 — the AC-3 baseline graph is gone.** Verified by teco: live `cpg_falkorchat` is
  **29,447 nodes / 185,517 edges** (M2 recorded 79,581 / 522,182), has **zero** test-file METHOD
  nodes, `FILENAME`s are bare basenames with no dir prefix, and `cpg/.cpg-artifacts/` is absent —
  so AC-3 cannot pass today **for reasons unrelated to MCP**. Options:
  **(A)** `joern` rebuilds the CPG from `falkor-chat/server/{falkorchat,tests}` — needs an approved
  destructive `GRAPH.DELETE`; reproduces AC-3's literal numbers.
  **(B)** re-baseline against the current graph and instead prove **tool ≡ `redis-cli`** by diffing
  both paths — a stronger test of *this* feature, but not AC-3 as written.
- **D2 — AC-3's "30 untested methods" is stale** (correct: 39 rows / 32 distinct names). A
  requirements edit → user/`tico`, not an implementer.
- **D4 — is `PROFILE` worth a read-only hole?** (analyst Blocker 1, **teco-verified live**.) The plan
  routes a `PROFILE …` prefix to `GRAPH.PROFILE`, which *executes* the query including writes. teco
  reproduced it: `GRAPH.PROFILE _teco_probe "MATCH (n:T) DELETE n"` really deleted the node
  (`count(n)` → 0 afterwards), while the tool would advertise `readOnlyHint=True`. Options:
  **EXPLAIN-only** (recommended — `GRAPH.EXPLAIN` is the planning tool; `graph-dba` keeps `PROFILE`
  via `redis-cli`), or explain-first-and-reject-write-operators (more code, more failure modes).
  *(Probe graph `_teco_probe` was created and deleted by teco; `GRAPH.LIST` restored to its prior
  five graphs.)*
- **D3 — AC-1's example question** ("who calls `post_message`, **transitively**") is not answerable
  in one query with today's recipes (they iterate). Either graph-dba writes a bounded upward
  name-closure query (plan S4; architect's first live attempt returned 0 rows — real work), or
  AC-1 is demonstrated with the direct-caller question.

## Verified by teco (2026-07-24, read-only)

Architect's blocker claims all confirmed: node/edge counts, zero test methods, bare-basename
`FILENAME`s, missing `cpg/.cpg-artifacts/`. Also confirmed **`claude/analyst/analyst.md:5` and
`claude/architect/architect.md:5` declare explicit `tools:` allowlists** (`qa-engineer` does not) —
the MCP tool is **invisible** to those two agents until their frontmatter is edited (plan S5).

## ⏸ PAUSED — resume anchor (user: out of credits, 2026-07-25)

**Instruction:** *"pause when the agents are done, we are out of credits."* teco launched **no**
further delegations after S2/S4/S5/S7. Nothing is mid-flight at the pause point.

**State at pause:** **S1 · S2 · S4 · S5 · S6 · S7 · S8 all ✅ and teco-verified on disk.**
Remaining: **S3** (wiring), **S9** (acceptance), **S10** (backlog/history).
**The server is built and works, but is not yet wired** — no `.mcp.json` exists, so no agent can
reach `mcp__cpg__query` yet even though `analyst`/`architect` now allowlist it. S3 is the switch-on.

**Extra carry-forwards for S3/S9 (from cobb, verified live against the docs 2026-07-25):**
- Tool search is **on by default** and MCP tools are **deferred** → a `ToolSearch` event is
  expected in a cold-session transcript. **`qa-engineer` must add `ToolSearch` to AC-1's permitted
  events or AC-1 fails on a good run.**
- What tool search reads is the **server `instructions`** (not the tool description), truncated at
  2 KB — which is what `coder`'s n-4 `FastMCP(instructions=…)` addresses. `alwaysLoad: true` loads
  tools upfront but **blocks startup until the server connects**.
- Claude Code MCP output limits: warn >10k tokens, cap **25k** (`MAX_MCP_OUTPUT_TOKENS`), over
  threshold the result is **persisted to disk and replaced by a file reference**; per-tool escape
  is `_meta["anthropic/maxResultSizeChars"]`, ceiling 500,000.
- `opencode debug skill` returns a **non-deterministic subset** (7–9 of 9) of skills per run, no
  error — harness behaviour, present before and after the change.
**Nothing is committed** — the whole feature is uncommitted working-tree changes (`git status`
shows modified `AGENTS.md`, `claude/README.md`, `claude/{analyst,architect}/*`, `skills/*`,
`docs/requirements/*`, plus untracked `cpg/mcp/` and this doc).

**To resume, in order:**
1. **S3** (`devops`) — repo-root `.mcp.json` + `.claude/settings.json`, then a live connect.
   Needs **one human approval**. Depends on S2's `server.py`.
2. **S9** (`qa-engineer`) — live acceptance AC-1…AC-4 against the rebuilt `cpg_falkorchat`,
   using the S8 baseline below; writes `docs/test-plans/` + `docs/test-reports/`.
3. **S10** (`coder`) — `docs/BACKLOG.md` M3 (C-301…C-307) + follow-ups **C-308** (transitive
   closure), **C-309** (audit gate red), **C-310** (OpenCode/Kiro MCP), **C-311** (guard blind to
   scripts), plus the new `FILENAME`-parse-root item; then `docs/HISTORY.md`.
4. **Re-verify** the two doc claims that currently describe unbuilt artifacts (`.mcp.json`,
   `cpg/mcp/tests`) and the **C-308 forward reference** from the requirements docs.

**S8 baseline for S9** (`cpg_falkorchat`, 2026-07-25): 110,048 nodes · 734,929 edges · 1,968
METHODs · 1,019 test-file METHODs (512 `test_*`) · direct callers of `post_message` = **21** ·
test-gap = **50 rows / 43 distinct names** (do not collapse to one number).

## Integration checks by teco (read on disk, not taken on the producer's word)

- **S1** ✅ `cpg/mcp/{requirements.txt,requirements-dev.txt,setup.sh,README.md}` + `.venv`;
  `./cpg/mcp/.venv/bin/python -c "import mcp.server.fastmcp, falkordb"` → exit 0;
  mcp **1.28.1** / falkordb **1.6.2** / pytest **9.1.1**; direct PII grep of the new files → clean.
- **S2** ✅ teco re-ran the suite itself: **53 passed / 7 deselected** offline, **7 passed** live,
  in 0.3s each. Behaviour probed directly against the rebuilt graph:
  `PROFILE` (even comment-prefixed) → **refused before any client call**; `EXPLAIN` → plan only,
  no data; a normal query → `rows=1 · count(m)=1968`, matching S8's baseline exactly; and a
  **typo'd graph name returns a curated "does not exist" listing the loaded graphs and does NOT
  materialise a key** (`GRAPH.LIST` unchanged at five) — the FalkorDB empty-key quirk is closed
  as designed. Direct PII grep of all new files → clean (they're untracked, so `audit-team.sh`
  cannot see them).
  ⚠ **Open item for S3/S9:** `coder` implemented re-gate finding **n-4** (`FastMCP(instructions=…)`)
  which the plan's v2.1 rework log records as **neither fixed nor rejected** — a real
  review-vs-plan gap. Confirm intent at the next architect/analyst touch.
- **S4/S5/S7** ✅ **the inertness fix is real**: `claude/analyst/analyst.md:5` and
  `claude/architect/architect.md:5` both now end `…, Agent, mcp__cpg__query`;
  `skills/cpg-analysis/SKILL.md:16` → `allowed-tools: mcp__cpg__query, Bash, Read`. Root
  `AGENTS.md` ends cleanly (the stray committed `</content></invoke>` XML trailer that was sitting
  in **every session's prompt** via `CLAUDE.md` → `@AGENTS.md` is gone). Audit: **2 FAIL, both
  pre-existing** (C-309) — teco counted 2, cobb's report says 3; harmless miscount, same set,
  **no new failures** either way.
  Also delivered: `agent-standards/claude-code.md` §MCP rewritten + a new OpenCode MCP section
  (divergence: OpenCode uses `<server>_<tool>`, `command` as array, `env`→`environment`), and a
  cross-tool rule that **MCP wiring does not port** — a shared skill routing through MCP needs a
  documented non-MCP fallback or it works in exactly one harness.
- **S6** ✅ `grep -rn "30 untested" docs/requirements/` → **empty** (stale figure gone);
  `joern-cpg-pipeline.md:76-77` FR-9 now routes through `mcp__cpg__query` with `redis-cli` as the
  documented fallback; `cpg-query-access.md` AC-1 now reads **direct** callers with the C-308
  deferral recorded; AC-3 restated as tool ≡ `redis-cli` equivalence; decision-log entries dated
  2026-07-25 present. **AC-4 satisfied** as far as static reading goes (qa re-checks in S9).

## ✅ RESOLVED — teco guardrail change was the user's own (2026-07-25)

**User, 2026-07-25:** *"dont worry about teco changes, i did that in another session."* Authorship
confirmed as the stakeholder, deliberately, outside this feature. **Not** `cobb`, **not** part of
S5. No action needed; the note below is kept as the audit trail for why it was queried.

While integration-checking S8, teco found uncommitted modifications to its own agent definition
and safety hook, unrelated to this feature:

- `claude/teco/teco.md` — description, role paragraph and routing table changed from *"no writing
  code"* to allow teco to **make "genuinely trivial single-file no-brainer" fixes directly**
  (new first row in the routing table).
- `claude/teco/hooks/guard-coordination-doc-writes.sh` — the escalation message now **advises the
  human to approve** such trivial source fixes. (The allowlist globs are unchanged —
  `docs/plans/*` + `teco/kaizen/inbox.md` — so escalation still fires; what changed is the
  guidance the human reads when deciding.)
- Also modified: `claude/teco/kaizen/{history,plan}.md`.

**Why it was queried:** an unrequested loosening of the coordinator's own write guard is a
governance change, and teco's standing instruction is that it never edits its own definition —
so it flagged rather than silently benefited. **Answer: the stakeholder made it.** Note for future
runs: the on-disk definition now permits teco a *genuinely trivial* single-file fix (typo,
one-liner, config value, rename); the hook still escalates every such write for human approval,
and anything needing design judgment, spanning files, or touching security/data-model/tests still
routes to a specialist.

## ⚠ Carry-forward risks (teco)

- **nn-2 forward reference:** both requirements docs now cite backlog item **C-308**, which **S10
  creates**. If S10 slips or renumbers, these citations dangle. Verify at S10.
- **Docs currently describe artifacts that don't exist yet.** Root `AGENTS.md` documents the
  repo-root `.mcp.json` (created by **S3**, not yet run) and `cpg/mcp/tests` pytest commands
  (created by **S2**, in flight). Expected mid-delivery, but **must be re-verified true when S2/S3
  land** — if the feature stopped here the docs would be describing fiction.
- **Root cause of the broken graph, found by joern at S8 — bigger than this feature.**
  `FILENAME` is **relative to the parse root**, so the parse root alone silently decides whether
  every `STARTS WITH 'tests/'` recipe filter works, **and the failure is invisible in node/edge
  counts**. That, not the missing tests, is why the old graph was useless. Belongs in
  `skills/joern-cpg/SKILL.md` as a post-load check — the `joern-cpg` producer path is
  **out of scope** for this feature, so file it as a backlog item at S10 rather than expanding scope.
- **Audit blind spot (found by devops at S1):** `audit-team.sh` check 7 uses `git grep`, which sees
  **tracked files only** — every new untracked artifact under this plan is invisible to it until
  committed. Each implementer must grep its own new files directly; the differential-audit gate is
  a *post-commit* signal for new files. Belongs in project docs (candidate for S10 / C-309).

## Log

- 2026-07-24 — teco: read requirements (complete, no OQs), verified environment, built the
  doc-impact scan, found the AC-3 stale-number defect. Launched U1 (architect) + a read-only
  inventory sweep (Explore).
- 2026-07-24 — Explore ✅ inventory returned; folded into "Inventory findings" above. Headlines:
  **no Linux-side `node`** (Python-only for a server), **FastMCP prior art** in
  `falkor-chat/server/falkorchat/mcp.py`, `cpg/` placeholder dir, zero MCP wiring repo-wide,
  and `SKILL.md` §3's literal-substitution warning as an extra shell-layer artifact.
  → teco to check the architect's plan against these; re-brief if the plan assumes `npx`.
- 2026-07-24 — U1 ✅ architect delivered `docs/plans/cpg-query-access.md` (723 L): **build, not buy**
  — official `@falkordb/mcpserver` v1.3.0 exposes **7 tools** incl. `delete_graph` with no
  tool-filtering (flat FR-2 violation) and needs Node ≥18. Contract
  `mcp__cpg__query(graph, cypher) -> str`, stdio, `.mcp.json`, `GRAPH.RO_QUERY`, S1–S9.
  Raised D1/D2/D3. teco verified all its blocker evidence independently.
- 2026-07-24 — U2 ✅ analyst: **needs changes** (`docs/reviews/cpg-query-access.md`) — 3 blockers,
  6 majors, 6 minors, 2 nits. Blockers: (1) `PROFILE` executes writes → read-only hole (**teco
  reproduced live**, now D4); (2) checked-in `.mcp.json` absolute home path trips `audit-team.sh`
  — **teco ran it: already `RESULT: FAIL` on pre-existing leaks** (`.claude/settings.json:4-5`,
  two kaizen inboxes, `docs/plans/m2-cpg-analysis-skill.md:327`), so S5's done-condition must be
  "no *new* failures"; (3) S7's done-condition unreachable — 8 commits to `falkor-chat/server`
  since the M2 measurement, so a rebuild is a CPG of *different code*. Analyst independently
  recommends **Option B**, and offers **A′**: build from a worktree pinned at the M2-era commit
  into a *separate* graph key (FR-4 makes that free) — no `GRAPH.DELETE` at all.
  Also validated: the build-vs-buy call, the `tools:`-allowlist trap, the EXPLAIN-prefix finding.
  Corrected teco's `claude/README.md` error (see above).
