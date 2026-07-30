# graphmind-ai-lab

A monorepo of independent, self-contained components — there is **no root-level build/test
script**. Each component carries its own docs and run instructions. Two themes run through the
repo: **graph-backed AI apps** (FalkorDB + LLMs) and **agent/skill engineering** (Claude Code
and OpenCode artifacts).

## Structure

- `salesperson/` — Streamlit sales-assistant chatbot. FalkorDB knowledge graph + LangChain +
  LangGraph; optional local LLM via LM Studio. See `salesperson/AGENTS.md`.
- `falkor-chat/` — Hybrid chat system (humans + AI) where **FalkorDB is the single store for
  everything**: chat history, workspace/reference data, workflow definitions and execution
  traces. GraphRAG (in-graph vector + traversal) and graph-state-machine workflows. Design and
  query library are locked and live-verified; M0 complete. See `falkor-chat/README.md` and
  `falkor-chat/AGENTS.md`.
- `opencode/` — Personal OpenCode configuration: custom agents and OpenCode-only skills.
  - `agents/` — `rpg`, `coding-senior`, and `severino/` (a full LM-Studio-backed local agent project).
  - `skills/` — OpenCode-authored `SKILL.md` packages (`comparison-driver`, `python-coding`,
    `skill-builder`, `user-preferences`, `write-tutorial`); OpenCode's global config symlinks here,
    not to the repo's shared `skills/`. See `opencode/skills/README.md`.
  - `local-llm.md` — notes on running OpenCode against a local LM Studio server.
- `cpg/` — Code-Property-Graph component code home. `cpg/mcp/` is the **`cpg` MCP server** (stdio,
  Python) exposing the single read-only tool `mcp__cpg__query(graph, cypher)` over FalkorDB; build
  artifacts (`.cpg-artifacts/`) and its `.venv/` are gitignored. It **runs containerized** — the
  launch surface is `cpg/mcp/docker-run.sh`, whose image tag is a **content hash of the build
  inputs**, so a stale image is unrepresentable rather than merely unlikely; `cpg/mcp/build.sh` is
  the supported build step. The container reaches FalkorDB over the host's published port via
  `--add-host=host.docker.internal:host-gateway`, so it does **not** touch the shared `falkordb-dev`
  container. The **host venv is retained** (`setup.sh`/`run.sh`) as the fast test loop and the
  fallback. See `cpg/mcp/README.md`. The repo-root `.mcp.json` that wires it is the repo's **first MCP
  wiring, and it is Claude-Code-only** — OpenCode and Kiro configure MCP through their own files and
  neither is wired (backlog C-310).
- `claude/` — Custom Claude Code subagents (one folder per agent, each with a `kaizen/` plan +
  history + learnings inbox the agent appends to during runs; `cobb` distills the inboxes). See `claude/README.md` (human catalog) and `claude/AGENTS.md` (agent context;
  `claude/CLAUDE.md` is a `@AGENTS.md` import stub).
- `skills/` — **Agent Skills home** (`SKILL.md` packages, the open `agentskills.io` standard)
  for the repo's cross-tool / Claude-Code-oriented capabilities: `agent-maintenance` +
  `agent-standards` (cobb's machinery), `joern-cpg` (drives `graph-dba`'s on-demand Joern
  CPG→FalkorDB pipeline), `cpg-analysis` (the consumer side). OpenCode-authored skills used only
  by OpenCode agents live separately, in `opencode/skills/`. See `skills/README.md`. Format
  ports across Claude Code/OpenCode/Kiro; tool-gating & activation behavior do not — verify per
  tool.

## Component docs (read before working in a component)

| Component | Entry doc(s) |
|---|---|
| `salesperson/` | `salesperson/AGENTS.md` · `salesperson/README.md` |
| `falkor-chat/` | `falkor-chat/README.md` · `falkor-chat/AGENTS.md` · `falkor-chat/docs/DESIGN.md` · `falkor-chat/docs/QUERIES.md` |
| `opencode/` | `opencode/agents/severino/README.md` · `opencode/local-llm.md` · `opencode/skills/README.md` |
| `cpg/` | `cpg/mcp/README.md` · `docs/requirements/cpg-query-access.md` · `skills/cpg-analysis/SKILL.md` |
| `claude/` | `claude/README.md` · `claude/AGENTS.md` (Claude Code reads it via the `claude/CLAUDE.md` import) |
| `skills/` | `skills/README.md` · `skills/*/SKILL.md` |

## Working in this repo

- **Chatbot tasks** → `salesperson/`, follow `salesperson/AGENTS.md`. No pytest/lint scripts; manual checks.
- **FalkorDB chat platform** → `falkor-chat/`, follow `falkor-chat/AGENTS.md` (`falkor-chat/CLAUDE.md`
  imports it). This is FalkorDB
  OpenCypher (not Neo4j): no APOC/GDS, vector indexes via DDL, index-before-constraint. Keep the
  query suite green (`./scripts/test_queries.sh`).
- **OpenCode agent tasks** → `opencode/`, follow the severino docs / `opencode/local-llm.md`.
- **Skill tasks** → `skills/` for cross-tool/Claude-Code-oriented skills, `opencode/skills/` for
  OpenCode-only ones; follow each `<name>/SKILL.md` and the directory's `README.md`.
- **Claude subagent / skill tasks** → `claude/` (agents) and `skills/` (skills), follow
  `claude/AGENTS.md`. Adding/editing/renaming an agent or skill means updating its source, its
  `kaizen/{plan,history,inbox}.md`, the relevant catalog (`claude/README.md` for agents, `skills/README.md`
  for skills), and `claude/AGENTS.md` in the same change.
- **Module documentation convention** — all of a module's documentation lives under
  `<module>/docs/`: `BACKLOG.md` (living backlog; `K-`numbered items), `HISTORY.md` (dated
  change log — append an entry for every delivered change), plus `requirements/`, `plans/`,
  `reviews/`, `test-plans/`, `test-reports/`, `manuals/`. `falkor-chat/` is the reference
  implementation; other modules adopt the structure when they first need it. Modules do **not**
  use `kaizen/` dirs — that convention exists only for agent folders (`claude/<agent>/kaizen/`).
  - **`manuals/` is the one end-user-facing kind** — everything else in this list is engineering
    process documentation (requirements/design/review/test artifacts for the people building the
    system). A `manuals/<slug>.md` document explains how to *use* the shipped product: screens,
    workflows, recovery from mistakes — never internal architecture or file layout. Owned by
    `tico`, illustrated with Mermaid diagrams wherever a picture replaces a paragraph of
    narration (`claude/tico/tico.md`, Mode 3). Unlike the other kinds, a manual doesn't have to
    shadow one feature's topic slug — it's often broader (a whole workflow or subsystem from the
    user's point of view); when a manual does document one feature end-to-end, it reuses that
    feature's slug per the family rule below.
  - **A document that freezes does not move.** It gets `Status: archived` in its own header
    block and stays exactly where it is. The status marker replaces the old "move it to
    `archive/` when the milestone closes" rule — and with it the inbound-link repair that move
    required.
  - **The existing `archive/` trees are read-only history of the previous convention.** Nothing
    is ever moved into them again, and nothing is un-archived.
  - **Citing another document:** write a **backticked path from the repo root** —
    `docs/plans/cpg-query-access.md`. A markdown link is **permitted and never required**; if
    you write one, its target must be **relative**, never `/docs/…`: a leading slash resolves
    against the filesystem root, so agents cannot follow it.
  - **Filename grammar:** `<component>/docs/<kind>/<topic-slug>[-<role>].md`.
  - **The prohibition:** a new document's basename **never begins with `m<digit>`, `k<digit>`,
    or a date.** The one exception — when a topic genuinely *is* a milestone or a recurring
    per-milestone activity, the milestone token goes **inside the slug, never as a prefix**
    (`followups-m4-coordination.md`); the test is *the topic has no name without it*.
  - **The closed role set:** *(none)* · `-coordination` · `-ml` · `-graph` · `-rca` · `-impl` ·
    `-report`. Everything else is part of the topic slug, not a role.
  - **Collision rules.** (1) The primary key is `(component, kind, topic-slug, role)`. (2) The
    **same slug across several kinds is the family** — required, not merely tolerated:
    `requirements/x.md` → `plans/x.md` → `plans/x-coordination.md` → `reviews/x.md` →
    `test-plans/x.md` → `test-reports/x-report.md` → optionally `manuals/x.md` (only when the
    manual documents that exact feature end-to-end; a manual with broader scope is its own topic
    slug, not a family member); a downstream document that invents a new slug is a defect. (3) A
    topic slug is never reused for a different topic. (4) The same
    basename in two directories is safe **because every citation carries a directory** — rules
    2 and 4 are a matched pair: adopt both or neither. (5) For a second document of the same
    kind and topic, the selector is one question — **has the earlier document been approved,
    gated, or executed against?** **No** → revise it in place (bump the optional `Version:` and
    add a dated revision note; a review gets a dated `## Pass N` section). **Yes** → it stays
    intact and you write a successor with the **ordinal on the slug** (`executor2.md`,
    `executor2-coordination.md`), *even while the earlier one is still `active`*. Two pointer
    pairs, never mixed: a successor that **replaces** carries `Supersedes:` and the earlier
    document gains `Superseded by:` and flips to `superseded`; a successor that **adds to** an
    earlier document that stays authoritative carries `Extends:` and the earlier one gains
    `Extended by:` with its `Status:` unchanged. **A header pointer is metadata, not an
    amendment** — it is the one edit permitted on an `archived` document.
  - **The header block** — one line, **immediately under the H1** (a blank line between them is
    permitted; nothing else may precede it), bolded labels, ` · ` separator:

    ```markdown
    # <Document title>

    > **Status:** <token> · **Owner:** `<agent>` · **Tracks:** <id(s)> (<M<n>>)
    ```

    `Owner:` is the producing agent, backticked; `Tracks:` is the backlog ID(s) plus milestone
    (`K-022 (M3)`), or `—`. **The canonical `Status:` token is the first thing after
    `Status:`**; free text is preserved *after* the token, never before it. Optional fields
    take the same bolded form and follow the three: `Version:`, `Supersedes:` /
    `Superseded by:` / `Extends:` / `Extended by:`, `Last updated:` (`tico` keeps it; nobody
    else needs it), `Reviews:`.
  - **The closed `Status:` set — five values, and who flips each:** `Interviewing` and
    `Ready for design` (`requirements/` only — `tico` owns both, the second only on explicit
    stakeholder confirmation) · `active` (the producing agent, at creation; amendable in place
    until the document has been approved, gated, or executed against) · `superseded` (whoever
    writes the successor) · `archived` (**the document's own owner, at milestone close, on
    `teco`'s coordination**). The owner who performs the `archived` flip, by kind:
    `plans/<slug>.md` → `architect` · `plans/<slug>-coordination.md` → `teco` ·
    `plans/`+`reviews/<slug>-ml.md` → `data-scientist` · `plans/<slug>-graph.md` → `graph-dba` ·
    `reviews/*` → `analyst` · `requirements/*` and `manuals/*` → `tico` · `test-plans/*` and
    `test-reports/*` → `qa-engineer`. **`teco` coordinates the close; it does not perform the flips** — its write
    guard reaches `docs/plans/*` only, so any other kind would raise a human approval prompt
    per file.
  - **The whole lifecycle, one line:** `grep -m1 -H 'Status:' docs/plans/*.md`.
  - **An existing `m<n>-` filename prefix is part of a name, not a lifecycle claim** — nobody
    should read meaning into it, and nobody should "fix" it.
- The root `CLAUDE.md` contains only `@AGENTS.md` — this file is the single source of truth for
  root-level context; per-component context files carry the detail.
