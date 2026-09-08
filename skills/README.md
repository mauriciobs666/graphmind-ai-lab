# Skills

[Agent Skills](https://agentskills.io) home for this repo's Claude-Code-oriented and
cross-tool capabilities — one `SKILL.md` package per folder (with optional `references/`,
scripts, and storage). This is the open, cross-tool skill standard read by **Claude Code**,
**OpenCode**, and **Kiro**; OpenCode-authored skills used only by OpenCode agents live
separately in [`opencode/skills/`](../opencode/skills/), not here — see that directory's own
catalog.

> **Portability:** the `SKILL.md` *format* ports across all three tools, but *behavior* does not
> automatically — tool-gating frontmatter (e.g. Claude Code's `allowed-tools`) and
> activation/matching semantics differ per tool. Each tool also keeps `name` ≤ 64 chars (must
> match the folder) and `description` ≤ 1024 chars. Re-verify any skill you deploy to a new tool.

## Catalog

| Skill | What it does | When to use | Origin |
|-------|--------------|-------------|--------|
| [`agent-maintenance`](./agent-maintenance/SKILL.md) | Kaizen plan/history upkeep, dual-audience documentation, file-location conventions, the drift audit/reconcile method, the team-coherence certification pass (§4; deterministic checks scripted in `claude/scripts/audit-team.sh`), the learnings-inbox distillation procedure (§5 — the team's capture→distill self-improvement loop), and the single-artifact prompt-quality lint (§7 — semantic review of one prompt over seven dimensions: contradiction, ambiguity, persona, cognitive load, coverage, composition conflict, prompt waste). | Creating/editing/renaming/removing/reviewing any agent, subagent, skill, steering doc, or memory file; certifying an agent team; processing learnings inboxes; linting a single prompt's quality. | cobb machinery |
| [`agent-standards`](./agent-standards/SKILL.md) | Perishable per-tool reference: exact frontmatter fields, directory paths, inclusion modes, config keys, what-loads-where tables, **settings precedence** (per key, list keys unioned, rules deny→ask→allow), and **MCP wiring per tool** (scopes/approval, `.mcp.json` shape and `${VAR}` expansion, tool naming, timeouts, tool search + `alwaysLoad`, output limits, and how MCP meets subagent `tools:` allowlists and skill `allowed-tools`) for Claude Code/Kiro/OpenCode. Every fact `Verified:`-stamped. | Producing/porting/debugging a concrete artifact and needing exact field names/paths rather than mental models. | cobb machinery |
| [`joern-cpg`](./joern-cpg/SKILL.md) | Operates the **Joern** toolset to turn a source repo into a Code Property Graph and export/load it into **FalkorDB** as Cypher: scripts for parse → export (neo4jcsv) → transform → load (pinning `JOERN_HOME`/`JAVA_HOME`), the default CPG→FalkorDB model (shared `:CpgNode` label + indexed `id`), and a CPGQL cheat-sheet (`references/cpg-model.md` for the deeper schema). Stamps the `:CpgBuildInfo` provenance marker `cpg-analysis`'s freshness recipe reads, capturing commit/tree/dirty **before the parse and scoped to the source** (`scripts/git-provenance.sh`) so a multi-hour build cannot record a commit it never parsed; `--source-origin` supplies that identity when the parse root is a staged copy. The stamp is a **map assignment** (`SET b = {…}`), so it replaces the marker's whole property set and a rebuild silently erases any hand-authored annotation on it — closure by construction, no list of keys to maintain; a post-stamp assertion re-checks it on every build and fails the run if anything the stamp didn't write is still there. | Building a CPG for a codebase, running CPGQL queries, or exporting/ingesting a repo's code graph into FalkorDB / Cypher. | `graph-dba` (on demand — CPG generation is rare) |
| [`cpg-analysis`](./cpg-analysis/SKILL.md) | **Consumer** counterpart to `joern-cpg`: queries an already-loaded CPG in FalkorDB with read-only Cypher through the **`mcp__cypher__query` MCP tool** (one tool, two parameters: `graph`, `cypher`; `EXPLAIN` supported, `PROFILE` refused), with **`redis-cli GRAPH.QUERY` kept as the documented fallback**. Lean `SKILL.md` core (query surface + shared CONTAINS→CALL / `REACHING_DEF` / symbol def-ref idioms + the topology gotchas) plus five copy-adaptable `references/` recipes — impact analysis, root-cause analysis, code review (taint), test-gap, and the freshness check (`teco`'s dispatch-time gate on whether a graph is current enough to trust). Cites `joern-cpg/references/cpg-model.md` as the one schema source; live-verified against a `pysrc2cpg` CPG. ⚠ **The MCP wiring is Claude-Code-only** (repo-root `.mcp.json`); under OpenCode/Kiro the `redis-cli` fallback is the only path — see the portability note below. | Impact / root-cause / taint / test-gap questions over a loaded CPG — when analyst, architect, qa-engineer, coder, tdd-engineer, or frontend-engineer need call-graph or data-flow answers instead of reading files. Building/loading a CPG routes to `graph-dba`. | `graph-dba` (M2) |
| [`python-web-quirks`](./python-web-quirks/SKILL.md) | Live-verified Python gotchas, version-pinned per entry — mostly web/async-framework, plus several general pytest/import-timing traps that surfaced in the same codebases: `asyncio.create_task` fire-and-forget GC-safety, Starlette/FastAPI `BackgroundTasks`' bounded thread-pool concurrency (anyio's default limiter, `total_tokens` 40, settable but only from inside a running event loop) vs. an unbounded raw `threading.Thread`, uvicorn defaulting `proxy_headers=True` so `ProxyHeadersMiddleware` rewrites `scope["client"]` from `X-Forwarded-For` — falsifying any "this request came from localhost" gate, FastAPI/pydantic `response_model_exclude_unset` silently dropping defaulted fields on **nested** models, FastAPI's four built-in doc routes (and which constructor kwargs suppress them) falsifying any "this app registers only routes X" claim while an *included* router's routes are absent from `app.routes` altogether (and `include_router(prefix=…)` never reaches the inner `.path`), so a route-table assertion can be unfalsifiable rather than merely wrong, a FastAPI `GET` route not matching `HEAD` at all (starlette's `Route` adds it, `APIRoute` does not) so `HEAD` falls through to any path-matching `Mount` — reading as a **404** where every other method reads as a `405` indistinguishable from a real method rejection, `responses={...}` being keyed by status code alone so two error bodies at one status collapse into one declaration — while an `x-` extension inside a status entry rides through verbatim onto `route.responses` and into `app.openapi()`, which is how a finer-than-status discriminator can still be asserted, FastAPI/Starlette's exception-handler registry (MRO lookup, the default keyed on `starlette.exceptions.HTTPException` and not `fastapi.HTTPException`, and a second `add_exception_handler` on one type replacing the first silently but readably), stdlib `urllib`'s `HTTPError`/`URLError`/`TimeoutError` exception taxonomy alongside redis-py's *sibling* `TimeoutError`/`ConnectionError` — both falsifying a handler map claimed total by exception type, an OpenAI-compatible local server's HTTP-200 error envelope on a missing `/v1` prefix, a fence-fragile `json.loads` LLM-judge parser, an env var set *after* import — a `monkeypatch.setenv` or a plain script's own `os.environ` assignment inside `main()` — as a no-op against an import-time-frozen constant, a bare `pytest.raises` with no `match=` passing on an unrelated same-type raise from an earlier check in the same validator, a whole-module-only test that gates itself on `config.option.keyword` instead of on what `request.session.items` collected — so a node id, `--lf` and `--deselect` all slip past the guard, a function-local deferred import re-resolving fresh on every call vs. a def-time-bound default argument, a one-way circular import failing in every load order unless the deferred import sits in a function body, and `TestClient`'s teardown cancelling every still-running task regardless of the app's own lifespan, bounding a call whose deadline the code under test computes with a **daemon thread + `join(timeout)`** rather than an elapsed-time assert a hang never reaches — stamping the start instant *inside* the thread body, since one taken before `Thread.start()` absorbs the scheduling gap and passes on a call that did nothing, and an application lock held across `ThreadPoolExecutor.submit()` delaying interpreter exit by exactly the lock's hold time rather than deadlocking it — nothing ever holds `_global_shutdown_lock` across a join, so the plausible three-party cycle does not exist. | Writing, reviewing, or planning asyncio fire-and-forget scheduling, background-task dispatch, a FastAPI response model relying on `exclude_unset`/`exclude_none`, an HTTP client against `urllib`/an OpenAI-compatible endpoint, an LLM-as-judge JSON parser, an assertion over an app's route table or its `responses={...}` declarations, a method-matching or static-mount-shadowing question, a test that must bound a possibly-hanging call, an exception-handler/error-envelope map over a web app, a lock-ordering or shutdown-deadlock question over a thread pool, or a pytest fixture/monkeypatch touching an env var or a deferred/circular import — for `coder`, `tdd-engineer`, `architect`, or `analyst` in a Python codebase. | `cobb` (analyst + coder inbox distillations, 2026-08-09 / 2026-08-11) |

The five OpenCode-authored skills previously cataloged here (`comparison-driver`,
`python-coding`, `skill-builder`, `user-preferences`, `write-tutorial`) moved to
[`opencode/skills/`](../opencode/skills/README.md).

## Deployment

Skills live here, version-controlled, and are surfaced to Claude Code and Kiro via a whole-dir
symlink from each tool's global config — so **every tool sees every skill in this directory** and
edits here are picked up live. OpenCode instead symlinks to
[`opencode/skills/`](../opencode/skills/), which holds its own (disjoint) set:

| Tool | Symlink |
|---|---|
| Claude Code | `~/.claude/skills` → `skills/` |
| OpenCode | `~/.config/opencode/skills` → `opencode/skills/` |
| Kiro | `~/.kiro/skills` → `skills/` |

Recreate on a new machine with `ln -s <repo>/skills <target>` (Claude Code, Kiro) or
`ln -s <repo>/opencode/skills <target>` (OpenCode). Skills are progressively-disclosed (only the
`description` is always-on), so exposing every skill in a tool's directory costs ~nothing; unused
ones simply never activate. If you later want per-tool scoping within this directory, switch a
tool to per-skill symlinks (the pattern the `claude/` agents use) instead of the whole-dir link.

### Portability notes (verified 2026-07-25)

- **Claude-only frontmatter is ignored, not rejected.** `cpg-analysis` declares
  `allowed-tools: mcp__cypher__query, Bash, Read` — an MCP tool name that exists only under Claude
  Code. OpenCode still discovers and parses the skill with its description intact (`opencode debug
  skill`, 6 runs, no warning on stderr), and its docs state that unknown `SKILL.md` frontmatter
  fields are ignored (`opencode.ai/docs/skills`; its recognized set is `name`, `description`,
  `license`, `compatibility`, `metadata` — gating is done with `permission.skill` patterns, not
  `allowed-tools`). **Not** exercised: an actual OpenCode *invocation* of the skill — tracked as
  **C-310** with the OpenCode/Kiro MCP wiring.
- **⚠ OpenCode's discovery over a whole-dir symlink was observed to be non-deterministic** when it
  pointed at this directory (repeated `opencode debug skill` runs returned *different subsets*, no
  error or warning). Recorded here as a harness behaviour to watch for, not a defect in any skill —
  it has not been re-measured since OpenCode's symlink moved to `opencode/skills/`.

## Maintenance

`agent-maintenance` and `agent-standards` are cobb's machinery; changes to them are logged in
[`claude/cobb/kaizen/history.md`](../claude/cobb/kaizen/history.md). Keep this catalog and the
root [`AGENTS.md`](../AGENTS.md) in sync when adding/editing/removing a skill.
