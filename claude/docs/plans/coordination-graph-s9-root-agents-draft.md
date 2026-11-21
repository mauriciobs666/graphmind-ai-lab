# S9 root `AGENTS.md` draft (AC-1, net-zero words)

> **Status:** active · **Owner:** `cobb` · **Tracks:** coordination-graph S9 (plan `claude/docs/plans/coordination-graph.md` §4 row S9)

Wording only: `cobb` does not edit root `AGENTS.md`; `teco` applies it (S9 row). Each pair is an exact
replace-this-with-that on today's root `AGENTS.md`; every `Replace` text occurs exactly once, so a literal
string replace works. Apply all 13 or none. Pairs 1-3 carry the content: pair 1 is AC-1 (backlogs are
mirrored in the graph and headed there, history stays a file); pairs 2-3 are the S11 remainder (Structure
bullet and component-table row for `coordination-graph/`). Pairs 4-13 are the offsetting trims; each drops
only connective prose, no fact another document does not hold. Counts are `len(text.split())`, the `wc -w`
rule, checked by applying the pairs to a scratch copy of the file.

| # | Change | old words | new words | delta |
|---|---|---|---|---|
| 1 | AC-1 backlog sentence | 29 | 28 | -1 |
| 2 | Structure bullet `coordination-graph/` (inserted before `kiro/`) | 8 | 54 | +46 |
| 3 | Component-table row (inserted before `kiro/`) | 4 | 13 | +9 |
| 4 | Trim `salesperson/` bullet | 24 | 17 | -7 |
| 5 | Trim `opencode/` bullet | 24 | 19 | -5 |
| 6 | Trim `mcp-monitor/` bullet | 38 | 30 | -8 |
| 7 | Trim `cypher-mcp/` bullet | 38 | 27 | -11 |
| 8 | Trim `cpg/` bullet | 15 | 9 | -6 |
| 9 | Trim `skills/` bullet | 11 | 10 | -1 |
| 10 | Trim `model-bench/` bullet | 21 | 15 | -6 |
| 11 | Trim `falkor-chat/` bullet | 14 | 11 | -3 |
| 12 | Trim `kiro/` bullet | 11 | 9 | -2 |
| 13 | Trim `cypher-mcp/` wiring clause | 17 | 12 | -5 |
| | **Total** | | | **+0** |

Root `AGENTS.md`: 3,200 words now, 3,200 after the 13 pairs.

## 1. AC-1 backlog sentence

Replace:

```
Backlogs are headed for the graph, the way team kaizen already is; keeping
    finished work out of them is what makes that move a migration rather than a cleanup.
```

with:

```
Backlogs are mirrored in the coordination graph (`coordination-graph/`; the file stays
    the source) and headed there; history stays a file (`HISTORY.md`), so finished work stays out of them.
```

## 2. Structure bullet `coordination-graph/` (inserted before `kiro/`)

Replace:

```
- `kiro/` — A checked-in Kiro CLI agent
```

with:

```
- `coordination-graph/` — `coordgraph` CLI mirroring the coordination files (teco ledgers, reviews,
  requirements, `BACKLOG.md`, agent `kaizen/plan.md`, registered KB files) into a FalkorDB `coordination`
  graph: read views (in flight, unit status, planned, milestone proposal), a post-edit hook (registration is the stakeholder's), KB freshness.
  Files are the truth.
- `kiro/` — A checked-in Kiro CLI agent
```

## 3. Component-table row (inserted before `kiro/`)

Replace:

```
| `kiro/` | `kiro/README.md`
```

with:

```
| `coordination-graph/` | `coordination-graph/README.md` · `coordination-graph/AGENTS.md` · `skills/coordination-graph/SKILL.md` |
| `kiro/` | `kiro/README.md`
```

## 4. Trim `salesperson/` bullet

Replace:

```
  FastAPI process (one process, one port, no CORS). Delivered per `docs/requirements/salesperson-ui.md` /
  `docs/plans/salesperson-ui.md`; delivery record, the defect-fix wave, and open items are in
  `falkor-chat/docs/HISTORY.md`/`BACKLOG.md`.
```

with:

```
  FastAPI process (one process, one port, no CORS). Per `docs/requirements/salesperson-ui.md` /
  `docs/plans/salesperson-ui.md`; record and open items: `falkor-chat/docs/HISTORY.md`/`BACKLOG.md`.
```

## 5. Trim `opencode/` bullet

Replace:

```
  can bring up/tear down one Docker-Compose environment it itself started
  (`opencode/agents/tank/README.md`). The manual
  covering local LM Studio setup, `opencode/docs/manuals/local-llm.md`, is what `tank` builds on.
```

with:

```
  can bring up/tear down one Docker-Compose environment it itself started
  (`opencode/agents/tank/README.md`); it builds on the LM Studio manual `opencode/docs/manuals/local-llm.md`.
```

## 6. Trim `mcp-monitor/` bullet

Replace:

```
*client*, Python 3.12 + `mcp` SDK, TOML config, one `asyncio.Task` per watch. Driven by
  auto-waking a headless agent CLI on an `@mention` in a `falkor-chat` thread (zero
  falkor-chat-side changes needed); also proven generic against a fake MCP server.
```

with:

```
*client*, Python 3.12 + `mcp` SDK, TOML config, one `asyncio.Task` per watch. Wakes a headless
  agent CLI on an `@mention` in a `falkor-chat` thread (no falkor-chat-side change); also proven generic.
```

## 7. Trim `cypher-mcp/` bullet

Replace:

```
**Runs
  containerized** via `cypher-mcp/docker-run.sh` (built by `build.sh`); image tag is a **content
  hash of the build inputs**, so a stale image is unrepresentable, and it reaches FalkorDB over the
  host's published port rather than the shared `falkordb-dev` container.
```

with:

```
**Runs
  containerized** via `cypher-mcp/docker-run.sh` (built by `build.sh`; image tag = **content hash of
  the build inputs**), reaching FalkorDB over the host's published port, not the shared `falkordb-dev`.
```

## 8. Trim `cpg/` bullet

Replace:

```
despite living at the repo root where they look like a project-wide log — a
```

with:

```
though they sit at the repo root — a
```

## 9. Trim `skills/` bullet

Replace:

```
`python-web-quirks` (live-verified asyncio/Starlette/
  FastAPI/pydantic gotchas, incl. test-harness ones, for `coder`/`tdd-engineer`/`architect`/
  `analyst`/`qa-engineer`).
```

with:

```
`python-web-quirks` (live-verified Python web/async/test-harness gotchas for the implementers and reviewers).
```

## 10. Trim `model-bench/` bullet

Replace:

```
Live spec: `model-bench/docs/DESIGN.md`, `FORMATS.md`. Requirements/plan live at the **repo root**, not `model-bench/docs/`; working
  state is in `model-bench/AGENTS.md`, delivery record in `model-bench/docs/HISTORY.md`.
```

with:

```
Live spec: `model-bench/docs/DESIGN.md`, `FORMATS.md`. Requirements/plan live at the **repo root**; working
  state: `model-bench/AGENTS.md`, record: `model-bench/docs/HISTORY.md`.
```

## 11. Trim `falkor-chat/` bullet

Replace:

```
Design and
  query library are locked and live-verified. Milestone status is authoritative in
  `falkor-chat/docs/BACKLOG.md`.
```

with:

```
Design and
  query library are locked and live-verified; milestone status: `falkor-chat/docs/BACKLOG.md`.
```

## 12. Trim `kiro/` bullet

Replace:

```
demoing Kiro-to-falkor-chat MCP
  connectivity; plus a broader, still-Draft multi-agent Kiro vision
```

with:

```
demoing that
  connectivity; plus a broader, still-Draft multi-agent vision
```

## 13. Trim `cypher-mcp/` wiring clause

Replace:

```
Repo-root `.mcp.json` wiring is
  **Claude-Code-only** — OpenCode/Kiro wire MCP their own way, neither wired yet (backlog C-310).
```

with:

```
Repo-root `.mcp.json` wiring is
  **Claude-Code-only**; OpenCode/Kiro are not wired yet (backlog C-310).
```
