# Review — CPG query access implementation plan

> Reviewer: `analyst`, 2026-07-24. Artifact: [`../plans/cpg-query-access.md`](../plans/cpg-query-access.md)
> (723 lines, author `architect`). Baseline: [`../requirements/cpg-query-access.md`](../requirements/cpg-query-access.md)
> (FR-1…FR-6, AC-1…AC-4 + decision log), [`../plans/cpg-query-access-coordination.md`](../plans/cpg-query-access-coordination.md)
> (teco), [`../requirements/joern-cpg-pipeline.md`](../requirements/joern-cpg-pipeline.md) FR-9,
> `skills/cpg-analysis/**`, `falkor-chat/server/falkorchat/mcp.py` + `pyproject.toml`, root `AGENTS.md`,
> `claude/AGENTS.md`, `claude/scripts/audit-team.sh`.

## Verdict

**Needs changes** — 3 blockers, 6 majors, 6 minors, 2 nits.

The design is well-grounded, the build-vs-buy call is correct and its evidence is real (I
re-verified the npm package and the absence of any per-server tool-filtering mechanism in Claude
Code). The blockers are not about the shape of the tool; they are (1) a read-only guarantee that
the `PROFILE` directive silently defeats, (2) a checked-in absolute home path that fails the
repo's own `audit-team.sh` — which the plan then uses as a done-condition, and (3) an acceptance
step (S7) whose target numbers cannot be reproduced because the source tree they were measured on
has moved on eight commits, while the step carries a destructive `GRAPH.DELETE` with no rollback.

What I verified vs. inferred is marked per finding. Live probes were read-only
(`GRAPH.RO_QUERY`/`GRAPH.QUERY` reads against `cpg_falkorchat`, `audit-team.sh`, `git log`,
introspection inside `falkor-chat/server/.venv`). I ran no writes and created no graphs.

---

## Answers to the six questions in the brief

1. **Requirements coverage.** FR-1…FR-6 each have a step. AC-2 and AC-4 are cleanly covered.
   **AC-1 is covered only on paper** (B-3 / M-1: the one-query transitive-callers deliverable it
   depends on has no step, no owner row and no done-condition), and **AC-3 is not deliverable as
   planned** (B-3).
2. **Build vs. buy.** Sound, and the evidence is real — see *What's solid*. The maintenance cost
   is stated but understated (m-6).
3. **Tool contract.** FR-2 survives literally — I built the tool in the pinned SDK and the schema
   has exactly two required string properties. The `EXPLAIN`/`PROFILE` prefix is in-band mode
   signalling, not a third parameter; the problem is not FR-2 but that `PROFILE` **executes
   writes** and defeats D4a (B-1). `GRAPH.RO_QUERY` exists and behaves as claimed (verified).
   The empty-key quirk is handled *incidentally* on the main path — `RO_QUERY` never materialises
   a key — but **re-opens on the directive path** (B-1), and the plan never says this out loud.
   Truncation is honestly instrumented but under-specified in two places (m-1).
4. **Step decomposition.** S1…S9 are mostly independently checkable; the S4–S6 ∥ S1–S3
   parallelism *is* safe (the contract is frozen and no doc edit reads code). Two dependency-table
   errors (m-4), one missing step (M-1), one impossible done-condition (B-3).
5. **Documentation completeness.** Four gaps against teco's inventory: the two agents' kaizen
   histories (M-4), `claude/tico/kaizen/inbox.md:19` (M-4), `skills/agent-standards/` (M-5), and
   the OpenCode-MCP gap that R-7 states but assigns to nobody (M-5).
6. **Risk realism.** Option **B is the sounder verification of *this* feature** and should be
   promoted to primary; option A tests the CPG *build*, not the access path, and cannot hit its
   own numbers today (B-3). The AC-1 direct-caller fallback does weaken the acceptance and is a
   `tico` decision, not an implementer's (M-1).
7. **Portability.** Honestly scoped out in §4.3/R-7 and the `redis-cli` fallback is retained for
   the right reason. Two loose ends: nobody owns the `agent-standards` OpenCode-MCP gap (M-5), and
   nobody verifies that an unknown `mcp__cpg__query` entry in `allowed-tools` is harmless in
   OpenCode/Kiro (m-3).

---

## Blockers

### B-1 · `PROFILE` by directive prefix executes writes — D5 silently defeats D4a, and the tool advertises `readOnlyHint=True`

**Evidence.** Plan §4.4 D4a (lines 285–291) claims the read path is bounded to reads because it
calls `Graph.ro_query`, and §4.4 D5 (293–302) routes a `PROFILE …` prefix to `Graph.profile()`.
`Graph.profile()` issues `GRAPH.PROFILE`, which the FalkorDB docs state explicitly: *"Unlike
`GRAPH.EXPLAIN`, `GRAPH.PROFILE` actually executes the query including any write operations
(CREATE, DELETE, SET)"* (verified against docs.falkordb.com, 2026-07-24). So
`PROFILE MATCH (n) DETACH DELETE n` against `graph="ws:acme"` runs for real — on the same instance
that holds `falkor-chat`'s `ws:acme`, `ws:test` and `reference` graphs. The plan itself names that
blast radius in R-6 and then declares it bounded by read-only mode.

Two consequences the plan does not carry:
- `annotations=ToolAnnotations(readOnlyHint=True)` (§4.4, line 277–278) becomes a false
  declaration to any harness or model that trusts it; the plan's own error table row *"This tool
  is read-only (GRAPH.RO_QUERY)"* is then wrong for one of the three code paths.
- The empty-key quirk resurfaces. `claude/graph-dba/falkordb-quirks.md:159-165` (verified):
  `GRAPH.QUERY` on a non-existent graph **materialises the key**, `GRAPH.RO_QUERY` does not.
  The main path is therefore safe by construction — a fact worth stating as a design *reason* —
  but `GRAPH.EXPLAIN`/`GRAPH.PROFILE` are not read-only commands, so a typo'd graph name on the
  directive path can leave a junk graph in `GRAPH.LIST` (inferred from the same quirk; I did not
  probe it, because probing it would create the key).

**Suggested fix (owner: `architect`, contract change; then `coder`).** Simplest proportional
option: **support `EXPLAIN` only** — it is plan-only and safe — and document that measured
profiling stays on the `redis-cli` fallback (`GRAPH.PROFILE`), which is exactly the kind of thing
the fallback exists for. If `PROFILE` must stay, gate it: call `Graph.explain()` first (planning
only), reject when the plan text contains a write operator (`Create`, `Merge`, `Delete`, `Update`,
`Set`), and only then profile — and drop or qualify `readOnlyHint`. Either way, add to §4.4 the
one-line statement that `RO_QUERY` is what keeps a typo'd graph name from materialising a key, and
add a robustness-pass row for it in §7.3.

### B-2 · The checked-in `.mcp.json` absolute path fails `claude/scripts/audit-team.sh`, which S5 lists as its own done-condition

**Evidence.** Plan §4.3 (line 232) commits `"command":
"<repo-abs-path>/cpg/mcp/run.sh"` to a tracked `.mcp.json`, justified in §2.5
(122–124) by the precedent in `.claude/settings.json`. `audit-team.sh` check 7 (script lines
116–137) greps **all tracked files** for the maintainer's home path and username and fails the
audit. I ran it:

```
FAIL  repo: username leaked into tracked files — genericize it
FAIL  repo: home path leaked into tracked files — genericize it
      .claude/settings.json:4: "Bash(<repo-abs-path>/claude/scripts/audit-team.sh)"
      …
RESULT: FAIL — fix the items above, then re-run.
```

So the cited precedent is itself a current lint violation, and S5's done-condition (`bash
claude/scripts/audit-team.sh` passes, line 491) is **unachievable today and would be made worse by
this change**. The plan doc itself also carries two absolute home paths (lines 232 and 517) that
will add two more hits the moment teco commits it.

**Suggested fix (owner: `devops` for the config, `teco` for the done-condition).** Promote the
plan's own alternative to **primary**: `"command": "bash", "args": ["-c", "exec
\"$CLAUDE_PROJECT_DIR/cpg/mcp/run.sh\""]`. The official docs support both halves of that
reasoning (verified 2026-07-24, `code.claude.com/docs/en/mcp`): *"Claude Code sets
`CLAUDE_PROJECT_DIR` in the spawned server's environment to the project root"*, and the supported
expansion syntax is only `${VAR}` / `${VAR:-default}` — a bare `$CLAUDE_PROJECT_DIR` is passed
through untouched and expanded by bash from the server env. Keep the absolute path only as a
documented last resort in `cpg/mcp/README.md`, never in a tracked file. Restate S5's
done-condition as *"`audit-team.sh` reports no **new** failures relative to the pre-change
baseline"* and file the pre-existing `.claude/settings.json` leak as its own backlog item. Use a
`<repo-root>` placeholder in the plan/review prose.

### B-3 · S7 cannot meet its done-condition: the M2 numbers were measured on source that has moved eight commits, and the step is destructive with no rollback

**Evidence.** S7 (lines 511–532) rebuilds `cpg_falkorchat` from `falkor-chat/server` with
`--reset` (a `GRAPH.DELETE`) and declares done at *"within a few percent of 79,581 / 522,182"*,
*"≈336 `tests/` methods"* and *"21 rows"* for the `post_message` callers. Those figures were
recorded on 2026-07-19. `git log --since=2026-07-18 -- falkor-chat/server` (verified) shows eight
commits since, four of them after the measurement date, touching `api.py`, `app.py`, `executor.py`,
`guards.py`, `llm.py`, `proof_defs.py`, `repository.py`, `schemas.py`, `services.py` and nine test
modules — including whole new modules (`guards.py`, `proof_defs.py`, `test_process_flow.py`,
`test_process_input.py`). A CPG of today's tree is a CPG of *different source*: node/edge counts,
the test-entrypoint count and the caller count will legitimately differ, and 139 `post_message`
mentions live in the tree today. The `joern` agent would iterate against an unreachable target.

Compounding: `cpg/.cpg-artifacts/` no longer exists (plan §2.3), so once `--reset` deletes the
current graph there is **no restore path** and no recorded command to rebuild even the current
(non-M2) graph. The plan puts this on the critical path — S8 depends on S7 (line 407) — for a
feature that is about the *access mechanism*.

**Suggested fix (owner: `teco` to re-sequence, `tico` for the AC-3 wording, `joern` if a rebuild
is still wanted).** Make **R-2 option B primary**: AC-3 becomes *"each M2 acceptance query returns
byte-identical value sets through `mcp__cpg__query` and through `redis-cli GRAPH.QUERY` on the same
graph"* — that is the only formulation that tests this feature, it is graph-independent, and it
needs no destructive op. Take S7 off the critical path. If the literal 21 / 39 / 32 figures are
still wanted, the correct way to get them is to build the CPG from a **git worktree pinned at the
M2-era commit** (e.g. `4f69a16`, 2026-07-19) into a **separate graph key** (`cpg_falkorchat_m2`) —
FR-4 makes the graph name a caller parameter, so nothing else has to change, and no `GRAPH.DELETE`
is needed. Whichever route, record in the plan the exact command that reproduces *today's* graph
before anything is deleted.

---

## Majors

### M-1 · AC-1's one-query transitive-callers deliverable has no step, no owner and no done-condition — and the escape hatch quietly rewrites the acceptance

**Evidence.** §7.2 (lines 600–608) discovers that the impact recipe answers transitive callers by
*iterating Q1 by name* (confirmed at `skills/cpg-analysis/references/impact-analysis.md:76-80`),
i.e. many tool calls, so AC-1 fails literally. It then assigns `graph-dba` a bounded upward
name-closure query "as an S4 sub-task", buried in the test-strategy section — while §5's step
table (398–412) and §6's doc table (555–573) both attribute `impact-analysis.md` solely to S4 /
`cobb`. The plan itself records that a naive composition *"returned 0 on the live graph"*, so this
is real Cypher work with a live-verification requirement, gating the headline acceptance
criterion, with no done-condition anywhere. The offered alternative — "AC-1 can instead be
demonstrated with the direct-caller question" — changes what AC-1 asserts; AC-1 says
"transitively" and the requirements' decision log records AC-1…AC-4 as accepted *as written*.

**Suggested fix (owner: `architect` to amend the plan; `graph-dba` to own the query).** Promote it
to a first-class step **S4b — upward name-closure query · `graph-dba`**, depending on the
availability of a queryable CPG, with a done-condition of the form *"the closure query, run live,
returns the known caller set for `post_message` at depth ≥2 and is documented in
`references/impact-analysis.md` with its collision caveat and the `WITH`-splitting idiom"*. Add
the row to §6. If it cannot be delivered, the AC-1 downgrade goes to `tico` as a requirements
amendment — it is not the implementer's or qa-engineer's call.

### M-2 · AC-1's verification instrument will fail for reasons unrelated to the feature

**Evidence.** §7.2 (587–598) proposes `claude -p … --output-format stream-json`, then *"count
`tool_use` events: **exactly one**"*. A cold session answering a CPG question legitimately emits
other tool events — the `Skill` invocation and `Read`s of `SKILL.md` and the recipe (which
`cpg-analysis` is explicitly designed to require: SKILL.md §4 sends the agent to open a recipe).
The assertion as written fails on a perfectly good run. Second problem: the run is headless, and
the docs are explicit that project-scoped servers stay at `⏸ Pending approval` until the workspace
is trusted interactively — so the AC-1 run must come *after* S3's human approval in the same
workspace, or it will silently test a session with no `cpg` server at all.

**Suggested fix (owner: `qa-engineer`, in the S8 test plan).** Restate as: *exactly one
graph-query tool call* (`name == "mcp__cpg__query"`, `input` keys exactly `{graph, cypher}`),
**zero** `Bash` events, and no shell quoting anywhere in the transcript. Add a precondition check
to the test plan: `claude mcp list` shows `cpg` **connected** in the same workspace before the
headless run.

### M-3 · `-> str` does **not** give unstructured content on the pinned SDK — every response is duplicated

**Evidence.** §4.4 (275–276): *"Return-annotate `str` so FastMCP emits unstructured text content
(no JSON-schema wrapper)."* Verified false on the pinned `mcp 1.28.1` (built the exact tool in
`falkor-chat/server/.venv`):

```
inputSchema : {'properties': {'graph': …, 'cypher': …}, 'required': ['graph','cypher'], …}
outputSchema: {'properties': {'result': {'type': 'string'}}, 'required': ['result'], …}
```

With an `outputSchema` present, FastMCP returns the payload **twice** — as text content and as
`structuredContent` — so a 60 000-char capped result becomes ~120 000 chars in context. That
directly defeats the token economy that motivates the truncation caps.

**Suggested fix (owner: `coder`).** Register with `@mcp.tool(..., structured_output=False)` —
verified on the same SDK to yield `outputSchema: None` — and add an S2 unit assertion that
`list_tools()[0].outputSchema is None` alongside the "exactly one tool / exactly two required
params" check.

### M-4 · Agent-edit doc convention is breached: kaizen histories and tico's inbox have no owning step

**Evidence.** S5 (484–493) edits `claude/analyst/analyst.md` and `claude/architect/architect.md`
frontmatter. `claude/AGENTS.md:35` (verified): *"Adding/editing/renaming/removing an agent →
update the agent source, its `kaizen/{plan,history}.md` …, the full catalog entry in README.md,
and the name rosters … **in the same change**."* Teco's coordination inventory flagged the same
files (`claude/{analyst,architect,qa-engineer}/kaizen/history.md:16-17`). Neither §5 nor the §6
documentation table mentions any `kaizen/` file. Separately, `claude/tico/kaizen/inbox.md:19`
holds the open note that FR-9 contradicts the MCP decision — the note AC-4 resolves; nobody clears
it.

**Suggested fix (owner: `cobb`, inside S5).** Add §6 rows for
`claude/{analyst,architect}/kaizen/history.md` (dated entry: `tools:` gains `mcp__cpg__query`,
with the C-3xx reference) and for `claude/tico/kaizen/inbox.md` (mark the FR-9 entry resolved by
this change, or leave it for cobb's next distillation with a pointer). Confirm whether
`claude/AGENTS.md` needs a line — its roster is capability-level, so probably not, but the
convention says decide it in the same change.

### M-5 · The perishable Claude-Code MCP facts have a canonical home and the plan doesn't put them there

**Evidence.** §2.4 collects genuinely durable, non-obvious facts (scope precedence and the trust
dialog, `enabledMcpjsonServers`, `mcp__<server>__<tool>` naming, `tools:` allowlists hiding MCP
tools, stdio servers not auto-reconnecting, per-server `timeout` semantics, `CLAUDE_PROJECT_DIR`
being set only in the *server's* env). `skills/agent-standards/claude-code.md` is the declared
authoring source for exactly these specifics — and its `## MCP` section (line 168) is **three
lines of prose** with none of it. Teco's inventory also recorded that `skills/agent-standards/`
has **no OpenCode MCP section**, which is precisely the remediation R-7 (700–703) needs and does
not assign. When this plan is archived, the facts go with it.

**Suggested fix (owner: `cobb`, as an S5 sub-task).** Fold §2.4 into
`skills/agent-standards/claude-code.md` §MCP (scopes + approval/trust, `.mcp.json` shape and
expansion rules, per-server `timeout`, tool naming, subagent `tools:` interaction, stdio
lifecycle, and the verified absence of per-server tool filtering), and add a short OpenCode MCP
subsection recording that OpenCode configures servers under `opencode.json` `mcp` and that this
repo wires none. Add both to §6.

### M-6 · The EXPLAIN/PROFILE path has no query timeout

**Evidence.** §4.4 (331–333) sets `CPG_MCP_TIMEOUT_MS` *"passed as `ro_query(..., timeout=…)`"*.
Verified signatures in `falkordb` 1.6.x:

```
ro_query(self, q, params=None, timeout: Optional[int] = None)
explain(self, query, params=None)          # no timeout
profile(self, query, params=None)          # no timeout
```

So the directive path is unbounded server-side and only stops at the 60 s harness wall — and per
B-1 a `PROFILE` is really executing. §7.3's "deep traversal `*1..12` times out at 30 s, not at the
60 s harness wall" therefore holds only for the plain path.

**Suggested fix (owner: `coder`).** If `EXPLAIN`-only survives B-1 this is nearly moot (planning
is cheap) — say so in §4.4. If `PROFILE` stays, either bound it with a client-side deadline or
document explicitly that `CPG_MCP_TIMEOUT_MS` does not apply to it, and add a §7.3 row that
asserts the actual behaviour rather than the assumed one.

---

## Minors

- **m-1 · Truncation is under-specified in two places, and one of them can mislead an analysis.**
  §4.4 (323–330) defines the row cap notice precisely, but (a) never says what happens when
  `CPG_MCP_MAX_CHARS` (60 000) is the binding cap — which rows are dropped and whether the same
  honest notice is emitted; and (b) the first 200 rows of an **unordered** result set are
  arbitrary, so an agent that reads "showing 200 of 79581" may still draw a conclusion from a
  non-deterministic sample. *Fix (`architect`/`coder`):* specify the char-cap behaviour (drop
  whole rows from the tail, emit the same notice with both counts) and extend the notice text with
  "results are unordered unless the query has ORDER BY". Also note in §4.4 that truncation is
  **display-only** — the client materialises the full result set first, so memory and latency are
  bounded by the query, not by the caps.
- **m-2 · Graph discovery has no path under a one-tool contract.** SKILL.md §1 currently tells the
  agent to confirm the graph with `redis-cli GRAPH.LIST` (`skills/cpg-analysis/SKILL.md:34,48`).
  With FR-2 forbidding a `list_graphs` tool, the only in-tool discovery is to *fail* a query and
  read the graph list out of the error message (§4.4, 343). That is a clever affordance but it
  should be stated as the intended workflow. *Fix (`cobb`, S4):* one explicit sentence in the
  rewritten §1 — graph names come from the caller; to discover them, use the `redis-cli
  GRAPH.LIST` fallback or read the tool's not-found error.
- **m-3 · Nobody verifies that `allowed-tools: mcp__cpg__query` is harmless in OpenCode/Kiro.**
  Root `AGENTS.md` warns that the SKILL.md *format* ports but "tool-gating & activation behavior
  do not — verify per tool". §4.5 adds a Claude-only tool name to a file that is symlinked into
  three harnesses. *Fix (`cobb`, S4 done-condition):* load the skill once under OpenCode and
  confirm the unknown tool name is ignored rather than rejected; record the result in
  `skills/README.md` or `agent-standards`.
- **m-4 · Two dependency-table errors.** §5: S8 verifies AC-4 by grepping the requirements docs
  (§7.2, 628–630) but does not depend on **S6**, which produces them; and S5's done-condition
  ("a cold `analyst` **and** `architect` can each call `mcp__cpg__query`", 491–492) depends on
  **S3**, while the table says "plan". *Fix (`architect`):* `S8 ← S3,S4,S5,S6,S7`; split S5 into
  the edit (parallel, depends on plan) and its live spot-check (in S8).
- **m-5 · A cheaper scoping option is not weighed.** `skills/agent-standards/claude-code.md:38`
  records that subagent frontmatter supports `mcpServers` (name ref or inline config). Scoping the
  `cpg` server to the three consumer subagents is an alternative to loading it into every session
  in the repo, and it interacts with the `tools:` allowlist question in R-3. Not necessarily
  better (per-subagent process spawn, plugin subagents ignore the field) — but it deserves the one
  line of rationale §4.3 gives to the scopes it did reject.
- **m-6 · Maintenance cost is understated and has no regression signal.** §3.3 books the cost as
  "~150 lines and its upgrade path". The actual artifact set is `server.py` + `run.sh` +
  `setup.sh` + `requirements.txt` + `README.md` + a test module + a venv lifecycle + `.mcp.json` +
  two agent frontmatters, in a repo with **no root-level test runner** — so nothing will tell
  anyone when it breaks except an agent's failed query. *Fix (`architect`/`devops`):* state the
  honest artifact count, and give the component a one-command smoke check
  (`cpg/mcp/.venv/bin/pytest cpg/mcp/tests -q` plus a live `-m live` run) referenced from
  `cpg/mcp/README.md` and from root `AGENTS.md`'s key-commands section, so the fallback is not the
  only failure detector.

## Nits

- **n-1 · `requirements.txt` has no "dev extra".** S1 (418–420) writes `mcp`, `falkordb` and
  *"dev extra `pytest>=9.1,<10`"* into a `requirements.txt`, which has no extras concept. Use a
  second `requirements-dev.txt`, or a `pyproject.toml` mirroring `falkor-chat/server`'s
  `[project.optional-dependencies]` (which is the in-repo precedent the plan otherwise follows).
- **n-2 · Redundant relative link.** S6 (500) inserts
  `[cpg-query-access.md](../requirements/cpg-query-access.md)` into a document that already lives
  in `docs/requirements/`; `./cpg-query-access.md` is the correct sibling form.

---

## What's solid

- **The build-vs-buy call, and its evidence.** I re-verified both load-bearing claims:
  `@falkordb/mcpserver` is v1.3.0, published 2026-07-01, `engines.node >=18.0.0`, deps
  `@modelcontextprotocol/sdk ^1.17.0` / `falkordb ^6.3.0` / `zod ^4.3.6` (npm registry); and the
  Claude Code MCP docs document **no** per-server tool-filtering mechanism, so a bought server's
  7 tools genuinely cannot be reduced to satisfy FR-2. Shipping `delete_graph` into every agent's
  tool list on an instance holding `falkor-chat`'s live graphs is a correctly weighted objection.
  The reversal trigger recorded in §3.3 is exactly the right artifact to leave behind.
- **§2.4 is the most valuable section of the plan.** The `tools:`-allowlist trap is real — I
  confirmed `analyst` and `architect` declare allowlists while `qa-engineer`, `graph-dba`, `coder`,
  `cobb`, `joern` and `devops` do not — and R-3 would have been a silent half-shipped feature.
- **§2.2's EXPLAIN finding is correct and I reproduced it**: `GRAPH.QUERY cpg_falkorchat "EXPLAIN
  MATCH (m:METHOD) RETURN count(m)"` returns `747`, not a plan. `GRAPH.RO_QUERY` works as claimed.
  The plan's instinct to make the divergence loud in *both* the tool description and SKILL.md is
  right.
- **The doc-impact table (§6) is genuinely owned row-by-row**, including the "no change, and here
  is why" rows for `rca.md`/`code-review.md`/`test-gap.md` and `cpg-model.md` — I spot-checked
  those and they are correct.
- **R-6's honest accounting** ("do not oversell the performance framing"; truncation is a new
  failure mode) is the kind of self-criticism that makes a plan trustworthy. Keep it in HISTORY.
- One correction *in the plan's favour*: teco's coordination doc says `claude/README.md` rows for
  architect/qa-engineer/analyst "do **not** mention `cpg-analysis` today". They do — rows 9, 16
  and 17 each link the skill. The plan's §2.5 reading is the accurate one; the coordination doc is
  the stale document here.

## Open questions (for the caller / stakeholder, not for the implementer)

1. **AC-3's definition** — with B-3 established, does the stakeholder accept AC-3 restated as a
   tool ≡ `redis-cli` equivalence proof (option B), or does the literal 21 / 39 / 32 reproduction
   matter enough to fund a pinned-commit rebuild into a separate graph key? (`tico` / user;
   subsumes the plan's own R-1 correction, which should be made in the same edit.)
2. **AC-1's wording** — if the upward name-closure query (M-1) cannot be delivered live, does AC-1
   get amended to "direct callers", or does the feature wait? (`tico` / user.)
3. **`PROFILE` in the tool** — is measured profiling worth the read-only hole (B-1), or is
   `EXPLAIN`-only plus the `redis-cli` fallback the accepted trade? (`architect`, then confirm
   with the user, since the plan's R-6 explicitly asked the reviewer to sanity-check this trade.)

---
---

# Re-gate — 2026-07-25

> Reviewer: `analyst`, 2026-07-25. Artifact: [`../plans/cpg-query-access.md`](../plans/cpg-query-access.md)
> **v2** (1,124 lines, `architect`), reworked in place against the review above plus stakeholder
> decisions **D1–D4** ([`../plans/cpg-query-access-coordination.md`](../plans/cpg-query-access-coordination.md)).
> Scope of this pass: (1) does every §10 rework-log row actually land in the plan body, (2) are the
> rejections/supersessions sound, (3) targeted re-checks of B-1/B-2/B-3, (4) risk introduced by the
> rework itself (723 → 1,124 lines, new S7, three new follow-ups), (5) the S6/S10 ownership call.
> **Not re-litigated:** D1–D4 — judged for conformance only.
> All work read-only: live `GRAPH.RO_QUERY` probes, script reads, `code.claude.com/docs/en/mcp`.

## Verdict

**Approve with suggestions** — 0 blockers, 2 majors, 5 minors, 2 nits.

The three v1 blockers are genuinely closed in the design, not just in the log. Implementation may
start on **all** steps, with two cheap conditions that do not need another gate:

- **S8 must not start until N-1 is corrected** (two lines in the step): the safety prompt the step
  tells `joern` to expect **will not fire**. Verified below.
- **S2 should absorb N-2 and n-3 before the truncation/directive code is written** — both are
  parameter-level, and S2 has not started.

Everything else (S1, S3, S4, S5, S6, S7, S9, S10) is clear to proceed as written.

## §10 rework log — verified against the body

I checked every row for a landing site in the design, not just a claim. **All 24 rows land.**
Spot-evidence: B-1 → §4.4 D4a + D5 table + §7.3 rows 1–3; B-2 → §4.3 `bash -c` block, local-scope
fallback, §5 before/after audit-diff recipe, C-309; B-3 → S8 staged copy + *"Explicitly not a
done-condition"* + §7.2 AC-3; M-3 → the frozen decorator with `structured_output=False` **and** the
S2 assertion `outputSchema is None`; M-4 → S5 bullets *and* four §6 rows; M-5 → the new S7 + two §6
rows; m-4 → `S9 ← S3,S4,S5,S6,S8` in the table with S5's live check explicitly deferred to S9.
Grounding re-verified by file: `claude/analyst/analyst.md:5` and `claude/architect/architect.md:5`
carry the `tools:` allowlists (qa-engineer has none), `claude/README.md` rows 9/16/17 carry the
`cpg-analysis` clause, `skills/agent-standards/claude-code.md:168` §MCP is three lines of prose,
`:38`/`:46`/`:110-111` say what §4.3 quotes them as saying, `docs/requirements/cpg-query-access.md:72`
still holds the stale "30 untested methods", `joern-cpg-pipeline.md` FR-9 and the SKILL.md/recipe
line references are all accurate. No "fixed" row is fictional.

**Rejections and supersessions — judged sound.** (a) *A′ dropped* (pinned-worktree rebuild into
`cpg_falkorchat_m2`): correct under D1 — the numbers are disowned, so the extra build buys a figure
nobody will cite; the plan still records the recipe for a future reader, which is the right residue.
(b) *M-1 superseded by D3*: correct, and C-308 carries the "naive composition returned 0 rows"
warning forward, so the real work is not lost. (c) *`claude/AGENTS.md` needs no line — decided, not
skipped*: correct — that file's roster is capability-level and no agent is added or renamed here.
(d) *m-5 rejected*: the decisive ground is real (`claude-code.md:110-111`, "ignored for teammates").
Nothing is waved away that still bites.

## Blocker re-checks

**B-1 / D4 — closed, and the architect's reasoning is confirmed by probe.** I re-ran the directive
question read-only against `cpg_falkorchat` (FalkorDB v4.18.11): `GRAPH.RO_QUERY … "PROFILE MATCH
(m:METHOD) RETURN count(m)"` returns **747** — results, not a profile, and no error. Same for
`EXPLAIN …`, for lowercase `profile …`. So a *passive* drop of `PROFILE` would indeed produce a
wrong answer rather than a failure, and the **active refusal is load-bearing exactly as §10 claims**.
`readOnlyHint=True` is now honest: `ro_query` on the plain path, plan-only on the `EXPLAIN` path,
no server call at all on the `PROFILE` path. The `GRAPH.LIST` pre-check does close the empty-key
hole on the `EXPLAIN` path (the only way `GRAPH.EXPLAIN` could materialise a key is a name that
isn't in `GRAPH.LIST`, and the pre-check returns the not-found error before `explain()` is reached),
and §7.3 asserts `GRAPH.LIST` is unchanged afterwards — the right check. One spelling does still get
through: see **n-3**.

**B-2 — closed, and the mechanism is confirmed in the official docs** (`code.claude.com/docs/en/mcp`,
fetched 2026-07-25, verbatim): *"Claude Code sets `CLAUDE_PROJECT_DIR` in the spawned server's
environment to the project root"* and *"This variable is set in the server's environment, not in
Claude Code's own environment, so referencing it via `${VAR}` expansion in the `command` or `args`
of a project-scoped `.mcp.json` … requires a default"*. The plan's unbraced-`$CLAUDE_PROJECT_DIR`-
inside-`bash -c` idiom is therefore correct and is the *only* form that works here (a
`${CLAUDE_PROJECT_DIR:-.}` default would silently resolve to cwd and break subdirectory sessions —
worth knowing if S3 is tempted to "fix" it that way). `audit-team.sh` check 7 is `git grep` over the
**whole** repo (script lines 116–137), so the plan's premise is right, the config is audit-clean,
and the "no *new* failures" done-condition with the before/after diff (§5) is concretely checkable —
`diff` on two audit dumps is a real gate, not a vibe. The local-scope fallback (`claude mcp add
--scope local`, `~/.claude.json`, untracked) is workable and has no audit surface. Residual, already
covered by S3's own done-condition: if the `env` block in `.mcp.json` were to shadow the inherited
server env, the launcher would fail loudly at connect time — S3's "same works from a session started
in a subdirectory" check catches it, and the fallback is one command away.

**B-3 — closed, and the staged copy does produce the prefixes the filters need.** `pipeline.sh`
passes `$SRC` to `build-cpg.sh` → `joern-parse "$SRC"` verbatim (no exclusions — the plan's premise
holds), and `cpg-to-falkordb.py` never rewrites `FILENAME` (its only `os.path.basename` is
`label_from`, applied to the *CSV filename*, not to node properties). So `FILENAME` is whatever
Joern emits relative to the parse root: copying exactly `{falkorchat, tests}` into `$SRC` yields
`falkorchat/…` and `tests/…`, which is what `STARTS WITH 'tests/'` needs. The counter-evidence
agrees: today's graph, built from `…/server/falkorchat` as root, shows bare basenames
(`api.py`, `guards.py`, `services.py` — re-confirmed live). Two safety properties I'll credit that
the plan doesn't claim: `pipeline.sh` only issues the reset when **both** `--reset` and `--load` are
set **and** after the export-non-empty assertion passes, so a failed parse cannot delete the graph.

## Findings

### N-1 · major · S8's destructive step relies on a guard hook that will not fire · owner `architect` (2-line edit), then `joern`

**Evidence.** S8 states: *"`--reset` issues `GRAPH.DELETE` and is escalated to human approval by
`joern`'s `guard-destructive-ops.sh` hook — expected; approve deliberately, and **only** for
`cpg_falkorchat`."* The guard (`claude/scripts/guard-destructive-ops.sh`, lines 34–58) extracts
`.tool_input.command` and pattern-matches **the command string**:
`grep -qiE "(^|[^[:alnum:]])(FLUSHALL|FLUSHDB)([^[:alnum:]]|$)|GRAPH\.DELETE([^[:alnum:]]|$)"`.
The command S8 prescribes is `skills/joern-cpg/scripts/pipeline.sh "$SRC" --graph cpg_falkorchat …
--reset --load` — it contains no `GRAPH.DELETE` token, so **no prompt appears and the delete runs
unattended**. The deletion itself is authorised by D1; what fails is the *scoping* safeguard the
step leans on — a mistyped `--graph cpg_salesperson` would be executed silently, and "the other four
graphs untouched" is only a post-hoc done-condition.

**Suggested fix.** Replace the `--reset` flag with an explicit pre-step that *does* trip the guard,
and drop `--reset` from the pipeline invocation:

```bash
redis-cli -p 6379 GRAPH.LIST                      # snapshot: five graphs expected
redis-cli -p 6379 GRAPH.DELETE cpg_falkorchat     # ← trips guard-destructive-ops.sh; approve here
skills/joern-cpg/scripts/pipeline.sh "$SRC" --graph cpg_falkorchat --language pythonsrc \
  --workdir /tmp/cpg-work/falkorchat --load
```

and correct the sentence to say the guard matches command text, so wrapper scripts that delete
internally bypass it. (Worth a separate backlog note for `cobb`/`devops`: the same blind spot
applies to any future script that wraps a destructive command.)

### N-2 · major · The 60 000-char cap collides with Claude Code's own MCP output handling, which can swallow the truncation notice · owner `architect`/`coder` (S2)

**Evidence** (`code.claude.com/docs/en/mcp`, "MCP output limits and warnings", fetched 2026-07-25):
Claude Code *"displays a warning when any MCP tool output exceeds 10,000 tokens"*, the default
maximum is **25,000 tokens** (`MAX_MCP_OUTPUT_TOKENS`), and — decisively — *"Without the annotation,
results that exceed the default threshold are **persisted to disk and replaced with a file reference
in the conversation**."* A 60 000-char CPG table is ~17–24 k tokens (identifier-dense text runs
~2.5–3.5 chars/token), i.e. **always above the warning threshold and plausibly at or over the
default limit**. The failure mode is precise and ugly: the run that binds the char cap is exactly
the run whose honest truncation notice matters, and that notice is the **last line** of the payload —
so it is the first thing lost to a harness-side cut or a file-reference substitution. The plan's
truncation design (§4.4) never mentions this layer.

**Suggested fix (any one, cheapest first).** (a) Emit the truncation notice as the **first** line as
well as the last, so it survives any tail-side clipping; (b) lower the `CPG_MCP_MAX_CHARS` default
to ~30 000 (≈10 k tokens — under the warning threshold, still ~150 CPG rows); (c) if large results
are genuinely wanted, declare `_meta["anthropic/maxResultSizeChars"]` on the tool (documented
per-tool escape, ceiling 500 000 chars, applies independently of `MAX_MCP_OUTPUT_TOKENS`). Record
whichever is chosen in `cpg/mcp/README.md` next to the "truncation is display-only" note, and add a
§7.3 row: *a result that binds the char cap arrives with its notice intact and is not replaced by a
file reference*.

### n-3 · minor · The directive sniff is comment-blind — a commented `EXPLAIN`/`PROFILE` reaches the server and executes · owner `coder` (S2)

**Evidence.** §4.4 D5 specifies the sniff as "trimmed, case-insensitive", and S2's test list covers
leading whitespace, `\n`, `\t` and case. Verified live (read-only, v4.18.11): both
`// hi⏎PROFILE MATCH (m:METHOD) RETURN count(m)` and `/* hi */ PROFILE MATCH …` are accepted by
`GRAPH.RO_QUERY` and return **747** — results. So a query whose directive is preceded by a Cypher
comment classifies as "plain", falls through to `ro_query`, and the caller gets results where it
asked for a profile or a plan. No write hazard (RO_QUERY still rejects writes, and the graph key is
still not materialised) — this is the *wrong-answer* class D5 exists to prevent, plus the "let me
just explain this" → "run the heavy traversal" hazard §2.2 names.

**Suggested fix.** Specify the sniff precisely in §4.4: strip leading whitespace **and** leading
`//`-to-EOL and `/* … */` comment blocks, then match `^(EXPLAIN|PROFILE)\b` case-insensitively.
Add the two comment-prefixed cases to S2's test 2/3 list.

### n-4 · minor · Tool search is on by default; the plan never sets server instructions or considers `alwaysLoad` · owner `coder` (S2) / `devops` (S3)

**Evidence.** Same docs page: *"Tool search is enabled by default. MCP tools are deferred rather
than loaded into context upfront"*; server instructions *"help Claude understand when to search for
your tools"*; `"alwaysLoad": true` in the server's `.mcp.json` entry loads its tools at session
start. The plan freezes an excellent ~350-char tool description but never mentions FastMCP's
`instructions=` (the server-level field tool search reads) — the one string that helps a *cold*
session find this tool, which is precisely AC-1's scenario. Two knock-ons: S3's done-condition
*"`/mcp` shows a tool count of 1"* is still fine, but AC-1's transcript will legitimately contain a
`ToolSearch` event, which §7.2's permitted-extras list (`Skill`, `Read`) doesn't mention.

**Suggested fix.** Set `FastMCP(name="cpg", instructions=…)` with one or two sentences ("query a
loaded Joern CPG or any FalkorDB graph with read-only OpenCypher…"); add `ToolSearch` to §7.2's
permitted extras; consider `"alwaysLoad": true` in `.mcp.json` given the server has exactly one
small tool.

### n-5 · minor · AC-1's headless runs may hit a permission prompt no one can answer · owner `qa-engineer` (S9) / `devops` (S3)

**Evidence.** MCP tools are named in permission rules as `mcp__cpg__query` (docs, "Use this full
name when referencing the tool in permission rules"). `.claude/settings.json` today carries only the
`audit-team.sh` Bash allows. The plan's `enabledMcpjsonServers` handles **server** approval and the
skill's `allowed-tools` pre-approves the tool **for the turn that invokes the skill** — but AC-1's
`claude -p` run asserts a tool call that may be issued before or without the skill turn, and a
non-interactive run cannot answer a prompt.

**Suggested fix.** Either add `"mcp__cpg__query"` to `.claude/settings.json` `permissions.allow` in
S3 (visible, reviewable, and it makes the tool usable in every headless run), or pass
`--allowedTools mcp__cpg__query` in S9's AC-1 command and say so in the test plan. Add "the tool call
was permitted, not denied" to the precondition list next to the `claude mcp list` check.

### n-6 · minor · AC-3's equivalence proof compares two different renderings · owner `qa-engineer` (S9)

**Evidence.** §7.2 AC-3 requires *"byte-identical value sets"* between `mcp__cpg__query` and
`redis-cli … --no-raw`. The tool's output is a ` | `-joined table with newlines/tabs escaped and a
**300-char cell cap** (`CPG_MCP_MAX_CELL`); `redis-cli --no-raw` prints quoted, one-value-per-line
output with no cap. The plan handles the row cap ("run on a narrowed form") but not the cell cap or
the format difference — a `CODE` or long `FULL_NAME` value would differ for a reason that is not a
defect.

**Suggested fix.** State the normalisation in the test plan: compare **parsed value sets** (split
the tool's row on ` | `, strip redis-cli's quoting), and restrict the equivalence queries to
projections whose cells fit under `CPG_MCP_MAX_CELL` — B3/B4 already do. Add one deliberate
long-cell case to §7.3 instead, asserting the `…(+N chars)` marker rather than equivalence.

### n-7 · minor · S6 amends accepted acceptance criteria without leaving a decision-log entry · owner `coder` (S6)

**Evidence.** `docs/requirements/cpg-query-access.md` ends with a Decision log whose last line is
*"2026-07-19 — Definition of 'solved' → AC-1…AC-4 accepted as written."* S6 rewrites AC-1 and AC-3
and is told to add a dated decision-log entry only to `joern-cpg-pipeline.md`. Without the matching
entry, the requirements doc will contradict its own log and the next reader cannot tell a
stakeholder ruling from an implementer's convenience.

**Suggested fix.** S6 appends to that log: *"2026-07-25 — D1/D2/D3 (stakeholder): destructive
rebuild approved, M2 figures superseded by a fresh recorded baseline; AC-3 restated as tool ≡
`redis-cli` equivalence; AC-1 demonstrated with the direct-caller question, transitive closure
deferred to C-308."*

### Nits

- **nn-1** — S1's "append `.venv/` to `cpg/.gitignore`" is redundant: root `.gitignore:140` already
  ignores `.venv`. Harmless; keep it only if the intent is local documentation.
- **nn-2** — C-308/C-309/C-310 are referenced as done-condition escape hatches in S4 and §5, but
  S10 (which creates them) runs last. Either seed the BACKLOG rows early or accept the forward
  reference knowingly.

## The ownership call: S6 and S10 to `coder`, not `cobb` — **upheld**

The split is drawn in the right place. `cobb`'s remit (per `claude/AGENTS.md` and the
`agent-maintenance`/`agent-standards` skills) is agent, skill and prompt **surfaces** —
`claude/**`, `skills/**`, the catalogs, the kaizen files — and the plan keeps every one of those
with `cobb` (S4, S5, S7). `docs/requirements/`, `docs/BACKLOG.md` and `docs/HISTORY.md` are module
docs under the repo's module-documentation convention, they carry no agent semantics, and `coder`
has unrestricted `Write` (no doc-scoped guard hook), so the step is executable as assigned. Two
qualifications, neither changing the owner:

1. The requirements doc is originally `tico`'s artifact, and the strongest competing claim is
   tico's. But D1–D3 are already ruled, so S6 is **transcription, not judgement** — the plan says
   so, and it should stay that way: if `coder` finds itself choosing wording that changes meaning,
   that is a stop-and-escalate, not an edit. Add n-7's decision-log entry so the provenance is on
   the record.
2. The loop back to `tico` is already covered by S5's close-out of `claude/tico/kaizen/inbox.md:19` —
   keep those two steps' sequencing (S5's inbox bullet after S6) as the plan states.

## What's solid in v2

- **The rework did not break the dependency graph.** S1→S2→S3; S4–S7 parallel on the frozen
  contract; S8 independent (correctly, since it has the longest latency); `S9 ← S3,S4,S5,S6,S8`;
  S10 last. The only implicit edge — S5's inbox close-out wanting S6 — is stated in the row itself.
  No doc is edited by two steps: §6 gives every row exactly one owner, and the three artifacts that
  span steps (`cpg/mcp/README.md` skeleton→fill, the smoke command in two files, the OpenCode
  spot-check feeding S4 and S7) each name which step writes what.
- **The new S7 earns its place.** §2.4's Claude Code mechanics are the most perishable and most
  reusable content in the plan; routing them to `skills/agent-standards/claude-code.md` (three lines
  of prose today) is the difference between a durable fact and an archived one.
- **The audit-hygiene block (§5) is the model answer to B-2**: it converts a red gate into a usable
  differential gate with a copy-pasteable recipe, and files the cleanup as its own item rather than
  smuggling it into this feature.
- **S8's "Explicitly not a done-condition"** paragraph — naming the ghost an implementer would
  otherwise chase — is worth more than the numbers it replaces.
- **§10 itself.** A rework log that maps every finding ID to fixed/superseded/rejected *with the
  reason*, including the reviewer's own alternative it declined, is what made this re-gate cheap.

## Open questions

**None blocking.** The plan's two non-blocking notes (C-308 deferred, C-309 leaves the audit gate
differential rather than binary) are correctly recorded and need no answer to start.
