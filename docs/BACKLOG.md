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
> Last reviewed: 2026-11-24.

**Where the component is:** M1…M8 delivered, M8 (kaizen agent/learning-note ontology) closed
2026-08-22. Each milestone's scope, items and gate trail is one entry in
[`HISTORY.md`](./HISTORY.md).

## Open

### In progress 🟡

### C-1001 — Connected code graph: code maps linked across languages, components and repositories, serving five use cases

- **Status:** 🟡 in-progress

The use cases are impact mapping, root-cause analysis,
code review, test coverage map and security. Requirements: `docs/requirements/connected-code-graph.md`;
architecture proposal: `docs/plans/connected-code-graph.md`. **Phases P0 and P1 are delivered** (record: `HISTORY.md`, the
2026-11-24 P1 close entry; per-AC status: `docs/test-reports/connected-code-graph-p1-closure-report.md`). **What remains:
the P1 close-out unit, then P3 onward** in the stakeholder's order P0 → P1 → P3 → P4 → P5 → P2 (P2, generality, is
scheduled last, not descoped), then the Arduino `.ino` slot, which needs P2's C/C++ extractor and gates no phase. The
close-out unit (stakeholder decision on `docs/reviews/connected-code-graph-p1-impl.md` Q1) is scheduled before P3 and holds:
scripts and wiring linking (shell exec, URL and script-hop facts, DEF-1), the held-out AC-5 gate, the AC-8 real-component
run with Q-E, and the AC-24 retry. The feature stays open until P2 is built, and AC-8 closes at P2 exit. The
whole-component `cpg/` live spec is owed at the feature close after P2 (`analyst` checks it). Owner: `teco`; `architect`
plans the close-out unit. New item IDs are allocated by `teco` with `wi` at the close.

**The P1 close-out unit (owner `architect` to plan, `teco` to coordinate; trigger: scheduled before P3).**

- **Scripts and wiring linking** (`architect`, then `tdd-engineer`): shell exec, URL and script-hop facts, and DEF-1, the
  shell/Markdown `curl` URL source (AC-1's one dev miss,
  `docs/test-reports/connected-code-graph-p1-closing-items-report.md` §3.3); closes AC-4 and the AC-2 wiring half (scripts
  0/49 and wiring 0/7 asked at P1).
- **Held-out AC-5 gate** (`teco`; trigger: scripts and wiring link): one combined gate over the four strata. Held-out http 26
  plus mcp 46 is 72 counting items, 3 under 75; the held-out list is unspent
  (`docs/test-reports/connected-code-graph-p1-heldout-strata-counts-report.md`). Which list the first gate spends is a
  question for the stakeholder (review Q1).
- **Gate harness minors** (`tdd-engineer`; trigger: before the gate's preregistration): P5-2 and P5-3
  (`docs/reviews/connected-code-graph-p1-32b-impl.md`), the stale `STRATA_GATED`/`MIN_COUNTING` and `ac5-gate-1` record,
  the dev-gold pre-gate probe, the per-kind miss breakdown, a no-workaround rehearsal rerun, and the data-scientist HA-6
  row in EP §5.2.1 (`docs/plans/connected-code-graph-p1-gate-marker-ml.md` §3).
- **S1 noise-reduction bar** (`data-scientist`; trigger: before the gate's preregistration): kind-wide, key-scoped or both
  for the at-least-5x bar (`docs/reviews/connected-code-graph-p1-noise-export-ml.md`).
- **`ccg-oracle-v5` and a runner test** (`tdd-engineer`; trigger: before any oracle capture in the unit): a roughly
  20-line `oracle_runner.py` test pinning the per-phase fence and the 16390 exclusion, and its pre-registration record.
- **AC-24 retry** (`data-scientist`, `qa-engineer`; trigger: the unit's start): not met at P1 (dev 9/12 and 8/12, closure
  report AC-24 row); a larger-model try, R4b out of sample, the "path line N" span residual; any lower local target is
  set from that measurement.
- **AC-8 real-component run and Q-E** (`devops`, `security-expert` advisory; Q-E: `tico` to the stakeholder; trigger: the
  unit builds a component image and a local-model path reachable from a no-route network): P1-40's four spec §8.1 features
  (`docs/plans/connected-code-graph-egress-spec3.md` §8.1, `docs/test-reports/connected-code-graph-p1-ac8-report.md` §4).
- **`raw_query` memory hardening** (`devops`; trigger: before the unit's gate run or any unattended `raw_query` use): set
  `QUERY_MEM_CAPACITY` (256 MiB recommended) in `cpg/infra/ccg-falkordb.sh up` and its `status` drift check, after
  confirming the engine's and the build's heaviest queries stay under it; today a raw query can exhaust instance memory, a
  documented P1 limit (`docs/plans/connected-code-graph-p1-raw-query-memory-graph.md`); per-user limits are P4 (FR-24).

**Product and close-document items (trigger: stated per line).**

- **`refresh` strands the store after the swap rename** (`tdd-engineer`; trigger: before `refresh` is used unattended): P5-1
  of `docs/reviews/connected-code-graph-p1-32b-impl.md`; a failure inside `refresh` after `ccg/store/swap.py` renames
  build to live leaves the live key in place with no remedy. Product code, reachable from the MCP `refresh` tool.
- **P1's own red tests** (`coder`; trigger: the P1 close commit): `test_p1_11b_the_repos_own_mcp_json_names_its_script`
  (re-baseline or exempt the `ccg` server's `bash -c` launcher) and `test_every_data_file_under_ccg_is_covered_by_a_package_data_glob`
  (one `pyproject.toml` glob for `ccg/front/prompts` and `ccg/front/grammar`); both are being fixed now. Not P1's: the
  `test_units` golden. Untraced, owner unverified: the `test_ac8_lib` signal flake, the leaf-harness survivor-tell,
  `test_scip_parse` x4 (did not reproduce in isolation).
- **AC-10 observed** (`qa-engineer`; trigger: a live model reachable through the broker): AC-10 closed on review, behaviour
  unobserved; inspect the first `teco`-coordinated brief on a covered system against the three behaviours.
- **AC-25 control half** (`architect`; trigger: python flow flips by an accepted re-measurement record, which the
  close-out unit's plan decides whether to run): V-both ships; the guarded-code rows are not listed
  (`docs/plans/connected-code-graph-p1-flow-fallback-result-ml.md`).
- **Rule-pack spec rows** (`architect`; trigger: the feature close after P2): `docs/plans/connected-code-graph-p1.md` §11
  (the shell `mcp-call`, `python-http-client` and `ts-mcp-client` rows, the `ownSite`/`urlLike`/`wrapperSite` words).
- **`skills/joern-cpg/SKILL.md`** (`graph-dba`; trigger: the P1 close commit): the build stage is reused by the `ccg`
  Joern extractor and the `neo4jcsv` path remains for optional `cpg_*` graphs.
- **Pack and linking work for P2** (`architect`; trigger: the P2 plan): `library_api` linking (Q-DYN B: AC6-DYN-1..3,
  TW-1), indexer adoption (the corpus measurement and the `ccg-p1-indexer-v1` record), AC-17's queue-name group. For the
  close-out unit's linking plan if it touches them: a string-only guard for `ts-mcp-client` (a pack-format change), JS
  `import.meta.env` environment records (AC6-EX-2), the `/api` base-constant join, Python string-constant prompt mentions
  (AC6-PM-3's original shape), a code consumer binding a config URL (AC6-EX-1's original `launch` shape), a receiver check
  for the TypeScript `store.fetch`, AC-9b's wider rule (an unbound crossing into a never-built unit), and one sentence in
  `extractors/joern/README.md` (D13: the default file filter has no left boundary on `test`/`spec`/`conf`, so `latest.ts`
  is `frontend-ignore`).
- **Untracked `.env` fixture** (`tdd-engineer`; being fixed now, delete this line when it lands): `cpg/tests/data/config/repo1/.env`
  is excluded by root `.gitignore:138` and untracked, so 4 `[.env]` tests in `test_config_extractor` fail on a clean checkout.
- **Absolute-path-dependent tests** (`tdd-engineer`; trigger: before CI or a second checkout): the default-audit-dir tests in
  `test_broker_audit.py` and the derived-fixtures tests in `test_eval_tools.py` depend on the checkout's absolute path.
- **3 `test_no_network.py` failures** (not P1's; owner: the llm-gateway feature's session; trigger: its close):
  `cpg/ccg/gateway/brokertransport.py:27` (llm-gateway D10) imports `BrokerClient` outside P1's import-boundary pin.
- **`skills/joern-cpg/scripts/test-stamp-wiring.sh` parse error** (`coder`; minor): a tree-sitter parse error.
- **S10 product defect** (`falkor-chat`/`salesperson` owners; belongs in `falkor-chat/docs/BACKLOG.md` on filing): the
  salesperson JoinScreen posts `displayName ''` after a reset and gets 422; the SPA must resend the name or the server
  accept a language-only re-join (`docs/reviews/connected-code-graph-p1-47-impl.md`). With it, the `falkor-chat`
  `test_queries.sh` §14.8 failure (460/461, `SUPERSEDES.matchId` relationship index, same review m3).
- **Delete or vendor `cpg/spikes/vertical-slice/`** (`coder`): frozen, not deleted. Blocked by (a) `cpg/tests/slice/*`
  and `tests/test_engine_answer_v18.py` oracle imports (trigger: P2 retires the slice equivalence gates, or the oracles
  are vendored as golden-free fixtures); (b) `cpg/spikes/store/{bfs,synth,lock_replay}.py` and
  `store/tests/test_store_spikes.py` line 404 (re-point to vendored copies or retire the store spikes with their
  results); (c) `s28_run.py`'s default `EXTRACT` and `tests/test_s28_run_extract.py` (re-point only if the S28 record is
  retired). Done when `git grep -n vertical-slice` outside `docs/` shows only historical hits and the suite is green
  (`docs/test-reports/connected-code-graph-p1-spike-dispositions-report.md`).
- **Git-fetch path** (`architect`; trigger: a manifest first naming an https `remote`): deferred by Q1 = C; design inputs in
  `docs/plans/connected-code-graph-egress-spec3.md`.
- **TS/JS flow remedy** (`architect`; trigger: JavaScript flow is measured): `switch`-case cond synthesis.
- **Teardown of `ccg-falkordb-tw-16390`** (`devops`; trigger: the user's `--yes`).

### Proposed 🔵

### C-310 — OpenCode + Kiro MCP wiring for the `cypher` server

- **Status:** 🔵 proposed

`.mcp.json` and
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


### C-507 — AC-5's append-before-delete ordering is enforced procedurally, not mechanically

- **Status:** 🔵 proposed

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


### C-809 — A `cypher-mcp` rebuild does not affect an already-running MCP connection

- **Status:** 🔵 proposed

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


### C-810 — Decide whether to restart the two containers found still running the pre-M8 image

- **Status:** 🔵 proposed

Found live during S6 (`cypher-mcp:aa088de045e2`, containers `recursing_lamarr`/
`vibrant_knuth`, 10h/21h old at the time) — for as long as they keep running, whatever session
each is bound to is silently exposed to the pre-M8 cross-clause smuggling gap this milestone
closes (C-803), and cannot use any of the 4 new write shapes. Not acted on during this milestone
since each belongs to a different long-running session that may still be in active use — a
restart decision needs the owning session's own context, not a unilateral call from this
coordination. Owner: whoever owns those sessions, or `devops` if they're confirmed abandoned.


### C-811 — The review's literal Attack A/C reproduction text is not valid live Cypher grammar

- **Status:** 🔵 proposed

`docs/reviews/kaizen-agent-ontology.md`'s Attack A/C text (`CREATE (...) MATCH (...) DETACH
DELETE/SET/REMOVE ...`) parses as a genuine FalkorDB grammar error live (a `WITH` must separate
an updating clause from a following `MATCH`) — it never reaches `authorize_write()` on this
engine, though the offline pytest suite is unaffected (it calls the function directly, no real
parser involved). If these strings are ever reused as a live fixture, they need a bridging
`WITH 1 AS _dummy` to actually exercise anything. No owner assigned — informational, act on it
only if/when these strings get copy-pasted into a live context.


### C-812 — Consider making `cobb`'s partial-vs-full deletion branch programmatic, not manual

- **Status:** 🔵 proposed

S6's dry-run self-caught one execution slip: the read-then-decide branch
(`skills/agent-maintenance/SKILL.md` §5) is documented correctly, but a human/agent computing
`otherRemaining` by hand and choosing the branch is easy to get wrong under procedural discipline
alone (caught and corrected in the same run, no lasting effect). Low-priority: worth a future
look at whether `cobb`'s tooling could compute the count and select the branch automatically
rather than relying purely on manual arithmetic every pass. Owner: `cobb`/`architect`, next time
this distillation mechanism is revisited.


### C-324 — Decide what to do with the `cpg_deprecated_salesperson` worked example in the CPG skills

- **Status:** 🔵 proposed

The graph is deleted (2026-09-18) and the component it described is gone from the
repo (removed after commit `3a686012`), yet five live skill files still use it as their
illustrative example of a frozen/retired-component graph and of the `:CpgBuildInfo` key set:
`skills/joern-cpg/references/cpg-model.md`, `skills/cpg-analysis/references/freshness.md` (two
places), and comments in `skills/joern-cpg/scripts/git-provenance.sh` and `pipeline.sh`. The
findings they cite (labels absent on `pysrc2cpg` graphs; marker key set) were observed there, so
the call is whether to keep them as provenance, reword them to cite `cpg_falkorchat` evidence
only, or trim. Only prose and comments, no behavior. Owner: unassigned (skills are `graph-dba`'s
and `cobb`'s to judge).


### Deferred ⚪

### C-323 — Bulk repath of the remaining module-anchored references to full root-anchoring (S5)

- **Status:** ⚪ deferred

**Deliberately deferred — recorded, not scheduled.** C-322 normalised the **live guidance**
files and left the module-anchored `` `docs/…` `` citations that sit inside **dated records and
per-item ledgers**, where the module-relative spelling is arguably correct as written;
`falkor-chat/docs/HISTORY.md` and `falkor-chat/docs/BACKLOG.md` account for most of them. A full
conversion is a **~60-file, judgement-heavy sweep** — each citation must be resolved against its
citing file before it can be rewritten — and the plan's cost decomposition puts the return at
**≤4.5% of future archival churn**, a churn D4 has *already* removed by keeping frozen documents
in place. **Do not schedule this.** Un-defer only on a measured, repeated failure to resolve one
of these citations. Cost analysis: `docs/plans/doc-reference-convention.md` §1.2, §2.1 and §12
*"Not scheduled"*. Owner: unassigned.


### C-409 — No live dispatch had observed a populated `:CpgBuildInfo` marker

- **Status:** ⚪ deferred

**Narrowed,
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
