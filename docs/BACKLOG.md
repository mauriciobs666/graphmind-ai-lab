# Backlog — CPG code-graph component

> **Forward-looking only.** Open work for the repo-root **CPG / code-graph** component
> (Joern → FalkorDB). Requirements in
> [`requirements/joern-cpg-pipeline.md`](./requirements/joern-cpg-pipeline.md) and, for the read
> path, [`requirements/cpg-query-access.md`](./requirements/cpg-query-access.md).
> **Delivered work is not kept here.** A closed item's record is [`HISTORY.md`](./HISTORY.md) —
> which indexes every `C-` item by milestone — or, when it is a live constraint on the system
> rather than a record of work, the design surface that owns it (`cypher-mcp/README.md` for the
> MCP server, the requirements docs for scope decisions).
> Item IDs use the `C-` prefix (distinct from falkor-chat's `K-`); the hundreds digit tracks the
> milestone the item was opened under (C-2xx = M2, C-3xx = M3).
> Status: 🔵 proposed · 🟡 in-progress · ⚪ deferred
> Last reviewed: 2026-08-25.

**Where the component is:** M1…M8 delivered, M8 (kaizen agent/learning-note ontology) closed
2026-08-22. Each milestone's scope, items and gate trail is one entry in
[`HISTORY.md`](./HISTORY.md).

## Open

### Proposed 🔵

- **C-310 — OpenCode + Kiro MCP wiring for the `cypher` server.** 🔵 `.mcp.json` and
  `enabledMcpjsonServers` are **Claude Code only**; OpenCode and Kiro configure MCP through their
  own files and neither is wired. `skills/cpg-analysis` is a *shared* skill, so today it reaches
  the MCP path in exactly one harness and `redis-cli GRAPH.QUERY` remains the **only** path under
  OpenCode/Kiro. Includes the `allowed-tools` portability result from S4 (an unknown entry is
  ignored, not rejected — spot-checked, not exercised by a real OpenCode invocation).
  The launch command is `cypher-mcp/docker-run.sh` (containerized at C-320). That keeps the
  property this item depends on — the launch surface is *a single command*, and a script ports
  where a JSON `args` array does not — while changing the per-host question from "is there a
  working Python 3.12 venv" to "is there a Docker daemon". So this item carries two obligations
  beyond the wiring itself: Docker is a prerequisite on any harness host (`run.sh` is what ports to
  a Docker-less one), and **`MCP_TIMEOUT` is a Claude-Code knob** — OpenCode's and Kiro's own
  startup budgets have to be established here. Owner: `cobb` / `devops`.

- **C-507 — AC-5's append-before-delete ordering is enforced procedurally, not mechanically.** 🔵
  `cobb`'s 4-step distillation sequence (append to `history.md`/knowledge base, confirm, only then
  curator-clear) is a documented discipline, not a tool-enforced invariant — `mcp__cypher__query` has
  no way to require or check the ordering of two independent write calls
  (`docs/plans/generic-cypher-mcp.md` §9 names this explicitly as procedural, not mechanical,
  enforcement). U7's acceptance pass could confirm only end-state consistency, not the raw sequence
  of API-call timestamps (`docs/test-reports/generic-cypher-mcp-report.md`, "AC-5 detail" section
  and Feedback & recommendations #1). U7's one real dispatch behaved correctly, but a single
  successful run is weaker long-run assurance than a mechanically-enforced invariant would be. No
  action needed for this delivery (the trade-off was already named and accepted at plan-gate time),
  but if this pattern extends to a second curator agent or a higher-volume distillation cadence,
  consider a tool-side "last write timestamp" queryable independently of the dispatched agent's own
  narration, rather than relying on end-state consistency plus self-report. Owner: `architect`
  (next time this tool's write path is revisited).

- **C-809 — A `cypher-mcp` rebuild does not affect an already-running MCP connection.** 🔵
  Live-confirmed during S6: `docker-run.sh` resolves the image by content hash per new connection,
  so a long-running Claude Code session (or a subagent inheriting its parent's stdio pipe) keeps
  talking to whichever container it started with, indefinitely, until that connection restarts —
  a rebuild is not retroactive. At S6 time, `docker ps` showed 3 live `cypher-mcp` containers on 2
  different images simultaneously (one fresh, two pre-M8, 10h/21h old) with nothing in the tool's
  responses to let a caller detect which build it's actually talking to. Two concrete
  improvements worth considering: (a) `cypher-mcp/README.md` should state this explicitly rather
  than imply "resolves automatically"; (b) surface a short image/version marker in the tool's
  responses (or a dedicated no-op diagnostic query) so a caller can positively confirm its build
  without shelling out to `docker ps`/`docker inspect`. Owner: `devops` (README + possible
  server-side change).

- **C-810 — Decide whether to restart the two containers found still running the pre-M8 image.**
  🔵 Found live during S6 (`cypher-mcp:aa088de045e2`, containers `recursing_lamarr`/
  `vibrant_knuth`, 10h/21h old at the time) — for as long as they keep running, whatever session
  each is bound to is silently exposed to the pre-M8 cross-clause smuggling gap this milestone
  closes (C-803), and cannot use any of the 4 new write shapes. Not acted on during this milestone
  since each belongs to a different long-running session that may still be in active use — a
  restart decision needs the owning session's own context, not a unilateral call from this
  coordination. Owner: whoever owns those sessions, or `devops` if they're confirmed abandoned.

- **C-811 — The review's literal Attack A/C reproduction text is not valid live Cypher grammar.**
  🔵 `docs/reviews/kaizen-agent-ontology.md`'s Attack A/C text (`CREATE (...) MATCH (...) DETACH
  DELETE/SET/REMOVE ...`) parses as a genuine FalkorDB grammar error live (a `WITH` must separate
  an updating clause from a following `MATCH`) — it never reaches `authorize_write()` on this
  engine, though the offline pytest suite is unaffected (it calls the function directly, no real
  parser involved). If these strings are ever reused as a live fixture, they need a bridging
  `WITH 1 AS _dummy` to actually exercise anything. No owner assigned — informational, act on it
  only if/when these strings get copy-pasted into a live context.

- **C-812 — Consider making `cobb`'s partial-vs-full deletion branch programmatic, not manual.** 🔵
  S6's dry-run self-caught one execution slip: the read-then-decide branch
  (`skills/agent-maintenance/SKILL.md` §5) is documented correctly, but a human/agent computing
  `otherRemaining` by hand and choosing the branch is easy to get wrong under procedural discipline
  alone (caught and corrected in the same run, no lasting effect). Low-priority: worth a future
  look at whether `cobb`'s tooling could compute the count and select the branch automatically
  rather than relying purely on manual arithmetic every pass. Owner: `cobb`/`architect`, next time
  this distillation mechanism is revisited.

### Deferred ⚪

- **C-323 — Bulk repath of the remaining module-anchored references to full root-anchoring (S5).**
  ⚪ **Deliberately deferred — recorded, not scheduled.** C-322 normalised the **live guidance**
  files and left the module-anchored `` `docs/…` `` citations that sit inside **dated records and
  per-item ledgers**, where the module-relative spelling is arguably correct as written;
  `falkor-chat/docs/HISTORY.md` and `falkor-chat/docs/BACKLOG.md` account for most of them. A full
  conversion is a **~60-file, judgement-heavy sweep** — each citation must be resolved against its
  citing file before it can be rewritten — and the plan's cost decomposition puts the return at
  **≤4.5% of future archival churn**, a churn D4 has *already* removed by keeping frozen documents
  in place. **Do not schedule this.** Un-defer only on a measured, repeated failure to resolve one
  of these citations. Cost analysis: `docs/plans/doc-reference-convention.md` §1.2, §2.1 and §12
  *"Not scheduled"*. Owner: unassigned.

- **C-409 — No live dispatch had observed a populated `:CpgBuildInfo` marker.** ⚪ **Narrowed,
  not fully closed** — `graph-dba` rebuilt `cpg_falkorchat` on request; `qa-engineer`'s targeted
  follow-up (`docs/test-plans/cpg-agent-adoption2.md`, `docs/test-reports/
  cpg-agent-adoption2-report.md`, 2026-08-17) dispatched `coder` against it and confirmed, live:
  the freshness marker query now returns a real populated row (not zero rows); the agent correctly
  falls back on `SOURCE_COMMIT`/`SOURCE_DIRTY` being absent (this graph's known `.git`-less
  scratch-copy build pattern, `docs/plans/cpg-agent-adoption-graph.md` §6) without erroring or
  misreading the absence; and it correctly avoids a false-positive stale claim on a genuinely
  fresh marker (the mirror of AC-4's positive branch) — PASS, 4/4, zero defects. That closes two
  of the three edges this item named: "no marker at all" (covered since Pass 1/Pass 2) and "fresh,
  populated marker" (covered now). **What remains open, and why it's deferred rather than
  re-triggered:** a *genuinely stale, populated* marker actually producing a concrete refresh
  suggestion has still never been observed live — `cpg_falkorchat`'s marker was minutes old at
  this follow-up's dispatch time, and there's no organic source drift to observe that branch
  against without fabricating a stale timestamp/commit, which the follow-up correctly declined to
  do. This edge is inherently time-dependent (needs either real elapsed time + independent commits
  on a rebuilt graph, or a future rebuild that happens to land already behind current source) —
  not something to chase with another proactive rebuild ping. Deferred as an accepted residual
  risk; re-open only if a future dispatch happens to hit this condition organically, or if a
  stakeholder decides the branch is worth deliberately engineering a real (not fabricated)
  drift scenario for. Owner: `qa-engineer` (this pass, closed); no active trigger owner while
  deferred.
