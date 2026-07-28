# Kaizen — Learnings Inbox: architect

> Append-only capture of durable, non-obvious environment facts the `architect` agent
> discovers during runs — raw observations, not conclusions. The maintainer (cobb)
> periodically distills this inbox (agent-maintenance skill §5): verifies each entry,
> routes it (prompt / knowledge base / project docs / discard), logs the promotion in
> `history.md`, and clears it. The agent only appends here; it never promotes.
>
> Entry format (append at the end):
>
> ```markdown
> ## YYYY-MM-DD — <the fact, one line>
> - **Evidence:** what was run/read/observed (command, file:line, output)
> - **Context:** the task where it surfaced, one line
> - **Suggested home:** prompt | knowledge base | project docs | unsure
> ```

## 2026-07-19 — falkor-chat's "byte-identity lock" on `executor._drive_loop` reproduces only by SHA, and only via a line-number-independent extraction

- **Evidence:** `falkor-chat/docs/plans/m3-executor-coordination.md` quotes the lock as SHA
  `71055f756280` with three different byte counts (2839, 2844, 2860); only 2860 is correct. The SHA
  reproduces from `sed -n '333,392p' server/falkorchat/executor.py | sha256sum | cut -c1-12` — i.e.
  it is pinned to *line numbers*, which shift whenever anything above the method changes. Verified
  equivalent that survives edits elsewhere in the file:
  `awk '/^    def _drive_loop/{f=1} /^    # ── seams/{f=0} f' server/falkorchat/executor.py | sed -e :a -e '/^\n*$/{$d;N;};/\n$/ba' | sha256sum | cut -c1-12`.
- **Context:** designing K-024's `kind:'process'` proof flow, whose hard constraint is "do not touch
  `_drive_loop`" — every unit's done-condition needs a verification command that stays valid.
- **Suggested home:** project docs (`falkor-chat/AGENTS.md`, next to where the lock is quoted)


## 2026-07-24 — A `MERGE … ON CREATE SET` "create-only / immutable" write is only create-only for *properties*; its MERGE **patterns** still create structure on re-run

- **Evidence:** `falkor-chat/server/falkorchat/repository.py:937 _PUBLISH_CYPHER`. Re-publishing the
  same `(key, version)` with an edited spec is documented (falkor-chat/AGENTS.md, QUERIES §11.1) as a
  silent no-op, and QA confirmed the property half (`docs/archive/test-reports/m3-workflow-engine-report.md`
  §5 DEF-1: `201` returned, old `name`/`kind`/step `config` retained). But `MERGE (st:Step {stepUid:…})`,
  `MERGE (from)-[rel:TRANSITION {on, order}]->(to)` and `MERGE (d)-[:START]->(start)` are *patterns*:
  an added step, a changed `to`, or a changed start step **creates** new nodes/edges beside the old
  ones. So "immutable per version" is really "monotonically additive per version".
- **Context:** designing K-031, a read surface whose whole purpose is making that trap detectable —
  the additive half was undocumented and unasserted, and it is the more dangerous half (the executor
  then drives a def carrying both the old and the new edges).
- **Suggested home:** project docs (`falkor-chat/AGENTS.md` + `docs/QUERIES.md` §11.1), and possibly
  the architect prompt as a general review question about MERGE-based immutability claims.

## 2026-07-24 — In an OpenCypher `RETURN a, b, collect(DISTINCT …)`, a non-aggregated field from an `OPTIONAL MATCH` is a **grouping key** — the "collapses to one row" property is conditional, not structural

- **Evidence:** `falkor-chat/docs/QUERIES.md` §11.2 documents the collapse and states the premise out
  loud — *"`start.key` is constant across the fan-out so the grouping is well-defined"*. The consumer
  (`repository.py:976 _read_subgraph`) then takes `result_set[0]` unconditionally. If a second
  `(d)-[:START]->()` edge ever exists (reachable — see the entry above), the query yields one row per
  start key and the reader silently picks one. A verified-and-documented query can therefore carry a
  latent multi-row hazard that only fires on a data shape the write path was assumed to prevent.
- **Context:** K-031 design; drove a decision to have the new read consume all meta rows rather than
  inherit `result_set[0]`.
- **Suggested home:** knowledge base (`claude/graph-dba/falkordb-quirks.md`) — pending live
  confirmation on this build, which the plan schedules as verification V-1.

## 2026-07-24 — In falkor-chat, a **def publish has no graph seam**, so any live experiment on publish semantics must be run on the *snapshot* side of the same query constant

- **Evidence:** `repository.publish_def` (`server/falkorchat/repository.py:1011`) writes to
  `self._reference()` (`:132-134`) → `db.reference_graph` (`db.py:87-94`) = `select_graph("reference")`
  — a hardcoded literal with no parameter, env var or `config` override. There is no per-workspace
  def publish, so "publish a probe def into a throwaway graph" is impossible and the obvious
  improvisation writes into the global `reference`. The escape hatch: `materialize_snapshot`
  (`:1470-1490`) formats **the same `_PUBLISH_CYPHER` constant** with `label="WorkflowDefSnapshot"`
  against `self._graph(ws)`, and `_READ_META_CYPHER` is likewise label-templated — so the identical
  query text can be exercised in a throwaway `ws:<probe>` (bootstrap → 2 calls → `GRAPH.DELETE`) and
  the result transfers to the `WorkflowDef` side unchanged. Gotcha found while specifying it: the
  probe def needs **≥ 1 transition** — `_PUBLISH_CYPHER` ends in `UNWIND $transitions` and an empty
  list collapses the row stream, so `result_set[0]` raises `IndexError`.
- **Context:** revising the K-031 plan after an `analyst` gate flagged the scheduled live
  verification as unexecutable as written (its stated "isolated throwaway workspace" precondition
  could not exist).
- **Suggested home:** project docs (`falkor-chat/AGENTS.md`, near the `seed_workflows.sh` row that
  already warns about `reference` vs `ws:<id>` staleness) — plus, as a general architect habit, "check
  that a planned live probe has a graph/tenancy seam before scheduling it".

## 2026-07-24 — FalkorDB **silently ignores** an `EXPLAIN`/`PROFILE` prefix inside `GRAPH.QUERY` and executes the query for real

- **Evidence:** `redis-cli -p 6379 GRAPH.QUERY cpg_falkorchat "EXPLAIN MATCH (m:METHOD) RETURN count(m)" --no-raw`
  → returns `747` (the result), not a plan; same for `PROFILE`. Plans come only from the separate
  `GRAPH.EXPLAIN` / `GRAPH.PROFILE` commands (or `falkordb-py`'s `Graph.explain()` / `Graph.profile()`).
  So the Neo4j habit of prefixing is a footgun here: "let me just explain this" runs the heavy traversal.
  `skills/cpg-analysis/SKILL.md:56-57` correctly says *prepend the command*, not the keyword — worth
  keeping that wording exact.
- **Context:** designing the one-tool MCP query surface for the CPG read path (`docs/plans/cpg-query-access.md`),
  where any Cypher pass-through has to decide what to do with a leading `EXPLAIN`.
- **Suggested home:** project docs (`skills/joern-cpg/references/cpg-model.md` consumer-query facts, or
  `skills/cpg-analysis/SKILL.md` §1)

## 2026-07-24 — Two of this team's agents (`analyst`, `architect`) declare an explicit `tools:` allowlist, so **any new MCP tool is invisible to them until their frontmatter is edited**

- **Evidence:** `claude/analyst/analyst.md` and `claude/architect/architect.md` both carry
  `tools: Read, Grep, Glob, Bash, Write, Edit, WebFetch, WebSearch, Agent`; `qa-engineer`, `graph-dba`,
  `joern`, `cobb` omit `tools:` and inherit everything (incl. MCP tools). Claude Code's MCP docs
  (`code.claude.com/docs/en/mcp`, fetched 2026-07-24) confirm the callable name `mcp__<server>__<tool>`
  is what must appear in a subagent's `tools` field, a skill's `allowed-tools`, permission rules and
  hook matchers. Related: a `SKILL.md` `allowed-tools` list **pre-approves** for the invoking turn —
  it does **not** restrict (`code.claude.com/docs/en/skills`), so `allowed-tools: Bash, Read` is a
  permission grant, not a sandbox.
- **Context:** planning the CPG MCP tool rollout — this is the cheapest way to ship a feature that
  silently fails for two of its three named consumers.
- **Suggested home:** prompt (architect/cobb checklist) or project docs (`claude/AGENTS.md`)

## 2026-07-25 — In this repo, "`claude/scripts/audit-team.sh` passes" is an unusable plan done-condition: it already returns `RESULT: FAIL` on pre-existing leaks

- **Evidence:** teco ran it 2026-07-25 → `RESULT: FAIL` from check 7 (personal-info leak), which
  `git grep`s **every tracked file** for `$HOME`, `id -un`, git user.name/email and hostname
  (`claude/scripts/audit-team.sh:116-137`). Current hits: `.claude/settings.json:4-5`,
  `claude/devops/kaizen/inbox.md:24`, `claude/joern/kaizen/inbox.md:19`,
  `docs/plans/m2-cpg-analysis-skill.md:327`. Usable form instead: capture the output before and
  after the change and assert **no new FAIL lines** in the diff.
- **Context:** reworking `docs/plans/cpg-query-access.md`, whose v1 used "audit passes" as a step's
  done-condition — unachievable regardless of the change.
- **Suggested home:** prompt (architect done-condition checklist) or project docs (`claude/AGENTS.md`)

## 2026-07-25 — Check 7 also means a plan document itself must never contain an absolute home path — and `.mcp.json` has a portable form that avoids one

- **Evidence:** the audit greps tracked files indiscriminately, so a `docs/plans/*.md` that quotes
  `/home/<user>/…` adds its own audit hits the moment it is committed (v1 of the plan carried two).
  For MCP config the portable substitute is `{"command": "bash", "args": ["-c", "exec
  \"$CLAUDE_PROJECT_DIR/<path>\""]}`: Claude Code expands only `${VAR}` / `${VAR:-default}`, so the
  unbraced `$CLAUDE_PROJECT_DIR` passes through and **bash** expands it from the spawned server's
  env, where Claude Code does set it (`code.claude.com/docs/en/mcp`, verified 2026-07-24). The
  per-machine escape hatch is `claude mcp add --scope local` → `~/.claude.json`, untracked.
- **Context:** same rework; the review's Blocker 2 was a checked-in absolute path in `.mcp.json`.
- **Suggested home:** project docs (`skills/agent-standards/claude-code.md` §MCP) + prompt

## 2026-07-25 — `teco` cannot own any plan step that edits files outside `docs/plans/` — its Write/Edit is harness-restricted

- **Evidence:** teco's own course correction, 2026-07-25: "my `Write`/`Edit` is harness-restricted
  to `docs/plans/`; any edit outside it escalates to the human." A v1 plan had assigned it the
  `docs/requirements/` reversal and the `docs/BACKLOG.md`/`HISTORY.md` updates. Working assignment
  in this team: `cobb` for agent/skill/prompt surfaces (`skills/`, `claude/`, agent frontmatter,
  catalogs, `kaizen/history.md`), `cobb` or `coder` for module docs (`docs/requirements/`,
  `docs/BACKLOG.md`, `docs/HISTORY.md`).
- **Context:** reassigning owners during the `cpg-query-access` plan rework.
- **Suggested home:** prompt (architect ownership rules when writing step tables)

## 2026-07-25 — `guard-destructive-ops.sh` matches the Bash *command string*, so a destructive op wrapped in a script bypasses the approval prompt entirely

- **Evidence:** `claude/scripts/guard-destructive-ops.sh:34-58` extracts `.tool_input.command` from
  the PreToolUse payload and greps it for `FLUSHALL|FLUSHDB|GRAPH\.DELETE`. It never inspects what
  a script does. `skills/joern-cpg/scripts/pipeline.sh --reset` runs `redis-cli … GRAPH.DELETE`
  internally (its lines 66-72), so the token never appears in the command the hook sees: no prompt,
  the wipe runs unattended. Same blind spot for any future wrapper. Workaround a plan can specify:
  run the destructive command as its own Bash call first (which *does* trip the guard, and puts the
  target name in the text the human approves), then invoke the wrapper without its reset flag.
- **Context:** re-gate fix N-1 on `docs/plans/cpg-query-access.md` — S8 told `joern` to expect an
  approval prompt that could not fire.
- **Suggested home:** prompt (architect: never treat a hook prompt as a gate without checking what
  the hook matches on) + project docs (`claude/scripts/` README or the joern-cpg skill)

## 2026-07-25 — Claude Code silently persists over-threshold MCP results to disk; a trailing truncation notice is exactly what gets lost, and `_meta["anthropic/maxResultSizeChars"]` is settable on the pinned SDK

- **Evidence:** `code.claude.com/docs/en/mcp` §"MCP output limits and warnings" (fetched
  2026-07-25): warning above 10,000 tokens, default limit 25,000 (`MAX_MCP_OUTPUT_TOKENS`), and
  *"Without the annotation, results that exceed the default threshold are persisted to disk and
  replaced with a file reference in the conversation."* Tools declaring
  `_meta["anthropic/maxResultSizeChars"]` use that char value for text content *regardless of*
  `MAX_MCP_OUTPUT_TOKENS`, hard ceiling 500,000. Design consequence: any truncation or caveat notice a
  server emits must be at the **head** of the payload, not only the tail. Verified live in
  `falkor-chat/server/.venv` (`mcp 1.28.1`): `FastMCP.tool()` takes `meta: dict[str, Any] | None`,
  `mcp.types.Tool` has a `meta` field, and a probe registration emitted
  `"_meta": {"anthropic/maxResultSizeChars": 60000}` in the `tools/list` entry with
  `outputSchema` absent under `structured_output=False`.
- **Context:** re-gate fix N-2 on `docs/plans/cpg-query-access.md` — a 60,000-char result cap
  collided with the harness limit and could swallow its own truncation notice.
- **Suggested home:** project docs (`skills/agent-standards/claude-code.md` §MCP)

## 2026-07-25 — An MCP server's `instructions=` string is injected into every Claude Code session as an "MCP Server Instructions" block, so an agent can verify it is live from its own context

- **Evidence:** `cpg/mcp/server.py:399` constructs `FastMCP(name="cpg", instructions=SERVER_INSTRUCTIONS)`
  (the 408-char string at `server.py:103`). That exact string appears verbatim in this session's
  context under a `## cpg` heading inside a **"# MCP Server Instructions"** block — no tool call
  needed to observe it. Confirmed independently by driving the handshake read-only:
  `initialize` over `bash -c 'exec ./cpg/mcp/run.sh'` (one JSON-RPC line on stdin, read one line of
  stdout) returns `result.instructions` = the same 408 chars. Two consequences: (1) the string is
  *always* loaded, not only when tool search fires — it is session-prompt real estate, so keep it
  short and about *when to reach for the tool*, distinct from the tool `description`, which loads
  only with the tool; (2) it is truncated at 2 KB, and the one-shot `initialize` probe is a
  side-effect-free way to verify any stdio MCP server's advertised strings without a running backend
  (the `cpg` server does not connect to FalkorDB at import).
- **Context:** closing the n-4 plan-vs-code audit gap on `docs/plans/cpg-query-access.md` — the plan's
  rework log had no record of a finding the implementer had shipped.
- **Suggested home:** project docs (`skills/agent-standards/claude-code.md` §MCP) + prompt (architect:
  a shipped MCP server's strings can be audited from the session context / a one-line `initialize` probe)

## 2026-07-26 — Root-relative markdown links (`/docs/x.md`) are **not** agent-followable: a leading `/` is filesystem-absolute to `Read`, even though GitHub and VS Code both resolve it to the repo/workspace root

- **Evidence:** `Read("/docs/BACKLOG.md")` in this repo returns *"File does not exist. Note: your
  current working directory is /home/mauricio/prg/graphmind-ai-lab"*, while `Read("docs/BACKLOG.md")`
  succeeds. Contrast with the documented renderer behavior: GitHub Docs (*Basic writing and formatting
  syntax*) states *"Links starting with `/` will be relative to the repository root"*, and VS Code
  resolves a leading `/` to the **workspace-folder** root (as-designed; with an open validation wart
  for `SKILL.md`-language-mode files, microsoft/vscode#299488). So there is **no single spelling that
  both a file-read tool and a markdown renderer treat as repo-root-anchored**: `/docs/x.md` is
  renderer-correct + agent-broken; bare `docs/x.md` is agent-correct + renderer-broken (renderers
  resolve it relative to the citing file). The only form serving both is a *composed* one —
  ``[`repo/root/path.md`](../relative.md)`` — backtick label for agents/grep, relative target for the
  renderer.
- **Context:** designing the repo-wide cross-document reference convention
  (`docs/plans/doc-reference-convention.md`), where "use a shorter root-anchored path" was the
  stakeholder's proposed fix and `/docs/...` was the obvious candidate.
- **Suggested home:** project docs (root `AGENTS.md`, stated **with the reason** so it isn't
  re-proposed) + prompt (architect/analyst: never recommend `/`-anchored markdown links in an
  agent-read repo)

## 2026-07-26 — This repo cites documents with **backticked path strings, not markdown links** (92% / 2,322 of 2,525), so a markdown link-checker validates 8% of references and misses ~97% of the defects

- **Evidence:** Read-only census of all 179 tracked/untracked `*.md` (fenced code masked; every
  reference resolved three ways — citing-file-relative, repo-root, module-root). Spellings: 2,179 bare
  backticked path strings + 143 backticked-inside-a-link-label + 185 explicit relative links + 18 bare
  relative links + **0** leading-slash links. Defects: **3** broken relative markdown links (all
  pre-existing, `falkor-chat/docs/BACKLOG.md:785,787,895`) vs. **87** dead path-bearing backticked
  `.md` citations. Of those 87, **15 point at a pre-archival path whose `docs/archive/…` twin exists**
  — and **all 15 are backticked, zero are links** (13 in `claude/*/kaizen/`, i.e. the cross-module rot
  the analyst predicted in `falkor-chat/docs/reviews/m3-archive-sweep.md` O-2). Corollary: the repo runs
  **two silently competing anchoring conventions** — 652 backticked refs resolve only from the repo
  root, **408 only from their module root** (63 in `falkor-chat/docs/HISTORY.md`, 58 in its
  `BACKLOG.md`), so 408 citations fail verbatim for an agent reading from the repo root.
- **Context:** quantifying why archiving two documents cost 22 path-string edits across 8 files
  (`9bbfbb5`); the answer was that the cost lives in the un-checkable spelling.
- **Suggested home:** project docs (root `AGENTS.md` citation convention) + prompt (architect/analyst:
  when auditing doc-link health in this repo, grep backticked path strings — a link-checker is nearly
  blind here)

## 2026-07-27 — Root `AGENTS.md` reaches **subagents** too (via root `CLAUDE.md`'s `@AGENTS.md`), so "point at the convention" and "inline the convention in each prompt" do not differ in reachability — only in drift risk

- **Evidence:** Root `CLAUDE.md` is exactly one line, `@AGENTS.md`. Observed directly in this run:
  as an `architect` **subagent** in an isolated context, the full text of root `AGENTS.md` arrived in
  the injected `claudeMd` context block before any tool call — i.e. the import is resolved for
  subagent sessions, not just the primary session. Consequence for design: a prompt line that says
  *"use the header block from root `AGENTS.md`"* costs the agent **no** extra lookup, because the
  target text is already in context; the only real trade-off against inlining the block in each of
  N prompts is **copy drift** (N+2 copies to keep in sync vs. 2).
- **Context:** ruling `analyst` finding M20 on `docs/plans/doc-reference-convention.md` — inline the
  header template in six agent prompts, or keep the single normative copy and fix the check.
- **Suggested home:** knowledge base (agent-design trade-offs) or prompt (architect/cobb: when
  deciding "duplicate the rule into prompts vs. point at root `AGENTS.md`" in this repo, the pointer
  is already resolved for every Claude Code agent, subagents included)
