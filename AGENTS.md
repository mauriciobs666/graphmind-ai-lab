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
  query library are locked and live-verified. Milestone status is authoritative in
  `falkor-chat/docs/BACKLOG.md`. See `falkor-chat/README.md` and `falkor-chat/AGENTS.md`.
- `opencode/` — Personal OpenCode configuration: custom agents and OpenCode-only skills.
  - `agents/` — `rpg`, `coding-senior`, and `severino/` (a full LM-Studio-backed local agent project).
  - `skills/` — OpenCode-authored `SKILL.md` packages (`comparison-driver`, `python-coding`,
    `skill-builder`, `user-preferences`, `write-tutorial`); OpenCode's global config symlinks here,
    not to the repo's shared `skills/`. See `opencode/skills/README.md`.
  - `local-llm.md` — notes on running OpenCode against a local LM Studio server.
- `cpg/` — Code-Property-Graph component code home: durable CPG reload artifacts
  (`.cpg-artifacts/`, gitignored) for the Joern-built graphs (`cpg_<component>`) loaded into
  FalkorDB. The MCP server is **not** here — it is the top-level `cypher-mcp/` (below), a generic
  Cypher-query tool rather than a CPG-specific one.
- `cypher-mcp/` — the **`cypher` MCP server** (stdio, Python) exposing the single read-only tool
  `mcp__cypher__query(graph, cypher)` over FalkorDB — generic (not limited to `cpg_*` graphs). It
  **runs containerized** — the launch surface is `cypher-mcp/docker-run.sh`, whose image tag is a
  **content hash of the build inputs**, so a stale image is unrepresentable;
  `cypher-mcp/build.sh` is the supported build step. The container reaches FalkorDB over
  the host's published port via `--add-host=host.docker.internal:host-gateway`, so it does **not**
  touch the shared `falkordb-dev` container. The **host venv is retained** (`setup.sh`/`run.sh`)
  as the fast test loop and the fallback. See `cypher-mcp/README.md`. The repo-root `.mcp.json`
  that wires it is **Claude-Code-only** — OpenCode and Kiro configure MCP through their own files
  and neither is wired (backlog C-310).
- `claude/` — Custom Claude Code subagents (one folder per agent, each with a `kaizen/` plan +
  history). Every agent's raw capture writes directly into one shared `kaizen_team` FalkorDB
  graph, so one query reaches every agent's raw learnings; `cobb` distills it. See
  `claude/README.md` (human catalog) and `claude/AGENTS.md` (agent context — including the
  `:KaizenEntry` graph shape; `claude/CLAUDE.md` is a `@AGENTS.md` import stub).
- `kiro/` — A checked-in Kiro CLI agent (`falkor-chat-demo`) that connects to `falkor-chat`'s MCP
  server as a client, restricted to `send_message`/`read_messages`, for a live demo of
  Kiro-to-falkor-chat MCP connectivity; plus a broader, still-Draft multi-agent Kiro vision in
  `kiro/DESIGN.md`. See `kiro/README.md` and `kiro/DESIGN.md`.
- `mcp-monitor/` — Standalone, generic MCP tool-result watcher: polls a configured MCP tool on an
  interval, matches its result against a regex, and launches a configured command line on match —
  its own MCP *client*, Python 3.12 + `mcp` SDK, TOML config, one `asyncio.Task` per watch. The
  driving scenario is auto-waking a headless agent CLI on an `@mention` in a `falkor-chat` thread
  (zero falkor-chat-side changes needed), but genericity is proven against a second, purpose-built
  fake MCP server too. See `mcp-monitor/README.md` and `mcp-monitor/AGENTS.md`.
- `skills/` — **Agent Skills home** (`SKILL.md` packages, the open `agentskills.io` standard)
  for the repo's cross-tool / Claude-Code-oriented capabilities: `agent-maintenance` +
  `agent-standards` (cobb's machinery), `joern-cpg` (drives `graph-dba`'s on-demand Joern
  CPG→FalkorDB pipeline), `cpg-analysis` (the consumer side), `python-web-quirks` (live-verified
  asyncio/Starlette/FastAPI/pydantic gotchas for `coder`/`tdd-engineer`/`architect`/`analyst`).
  OpenCode-authored skills used only
  by OpenCode agents live separately, in `opencode/skills/`. See `skills/README.md`. Format
  ports across Claude Code/OpenCode/Kiro; tool-gating & activation behavior do not — verify per
  tool.

## Component docs (read before working in a component)

| Component | Entry doc(s) |
|---|---|
| `salesperson/` | `salesperson/AGENTS.md` · `salesperson/README.md` |
| `falkor-chat/` | `falkor-chat/README.md` · `falkor-chat/AGENTS.md` · `falkor-chat/docs/DESIGN.md` (graph) · `falkor-chat/docs/SERVER.md` (server process) · `falkor-chat/docs/QUERIES.md` |
| `opencode/` | `opencode/agents/severino/README.md` · `opencode/local-llm.md` · `opencode/skills/README.md` |
| `cpg/` | `docs/requirements/cpg-query-access.md` · `skills/cpg-analysis/SKILL.md` |
| `cypher-mcp/` | `cypher-mcp/README.md` |
| `claude/` | `claude/README.md` · `claude/AGENTS.md` (Claude Code reads it via the `claude/CLAUDE.md` import) |
| `kiro/` | `kiro/README.md` · `kiro/docs/requirements/kiro-demo-agent.md` · `kiro/DESIGN.md` (Draft/vision, not the built system's spec) |
| `mcp-monitor/` | `mcp-monitor/README.md` · `mcp-monitor/AGENTS.md` |
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
  `kaizen/{plan,history}.md`, the relevant catalog (`claude/README.md` for agents,
  `skills/README.md` for skills), and `claude/AGENTS.md` in the same change.
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
    block and stays exactly where it is.
  - **A living document is compacted at milestone close, not only appended to.** A document read
    **whole** to be used — `BACKLOG.md`, `DESIGN.md`, `AGENTS.md`, `README.md` — can never
    freeze, so it never sheds weight on its own; one read **by lookup** — `HISTORY.md`,
    `QUERIES.md`, `reviews/`, closed `plans/`, `test-reports/` — may grow without bound, and that
    is correct. So at milestone close, in the same pass that flips that milestone's documents to
    `archived`, **`teco` lists what should go** — every delivered item, plus each section of a
    living document that tracks **work status** rather than the system itself: a "currently in
    progress" header, a plan-doc row for a document that now exists, a delivered-ticket
    annotation. **The human applies the list** — the by-kind owner table below routes none of
    these document kinds, and `teco`'s write guard allows only the `archived` flip.
    **Verify it is in `HISTORY.md` before deleting** — a closeout is a move, not a discard.
  - **`BACKLOG.md` is forward-looking only — a delivered item does not stay in it, not even as an
    index row.** Its record is `HISTORY.md`, which indexes every delivered item by milestone; the
    exception is a fact that is a **live constraint on the system** rather than a record of work
    (an interface limit, a rejected option with a reversal trigger), which belongs to the design
    surface that owns it — the component's `DESIGN.md` or `README.md`, or the requirements doc for
    a scope decision. Backlogs are headed for the graph, the way team kaizen already is; keeping
    finished work out of them is what makes that move a migration rather than a cleanup.
  - **An open item is rewritten, not appended to.** Revisiting one produces a *replacement* body:
    fold what you learned into it so the item reads as one present-tense statement of what is true
    now and what remains — **never a dated `Update:` clause stacked under the previous one.** After
    three of those the current ask is the last sentence of a wall of text and the two above it are
    false. What an update supersedes is owed nothing: a stale guess at what remained was never
    acted on, and any real change it reported has its own `HISTORY.md` entry — verify that, then
    drop it. What the revisit *found* is different and stays, folded in as the item's content
    rather than as news: a decision taken, a constraint discovered, a partial precedent set,
    evidence that moves the call. Where the fact of the revisit is itself worth recording, it is
    **one dated line, not a narrative** — the same rule collision rule 5 applies to a document
    revised in place, and deliberately the same words.
  - **A milestone-map row, wherever one is kept, says what the milestone is and when it landed.**
    Gate sequences, defect trails and superseded framings belong in `HISTORY.md`.
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
    kind, topic, **and role**, the selector is one question — **has the earlier document been
    approved, gated, or executed against?** **No** → revise it in place (bump the optional
    `Version:` and add a dated revision note — **one dated line, not a narrative**; a review gets
    a dated `## Pass N` section). **Yes** → it stays
    intact and you write a successor with the **ordinal on the slug** (`executor2.md`,
    `executor2-coordination.md`), *even while the earlier one is still `active`*. **A `reviews/`
    document is the exception:** it revises in place regardless of the selector's answer — a
    re-review's value is pass 1 and pass 2 read together. **A later `## Pass N` is compact
    by rule:** the verdict and genuinely new findings in full, while a finding already reported
    gets **one disposition line** — fixed / not fixed / superseded, plus the evidence you
    rechecked — never re-argued prose. Two pointer
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
    writes the successor) · `archived` (**`teco`, at milestone close** — `teco` performs the
    mechanical flips itself: its write guard auto-allows an `Edit` on a `docs/**.md` file whose
    old/new strings differ only in the canonical `Status:` field flipping to `archived`). For any
    `archived` flip that is **more than** that mechanical one-token edit — bundled with other
    changes, disputed, or needing judgment about what freezes — the flip still routes to the
    by-kind owner:
    `plans/<slug>.md` → `architect` · `plans/<slug>-coordination.md` → `teco` ·
    `plans/`+`reviews/<slug>-ml.md` → `data-scientist` · `plans/<slug>-graph.md` → `graph-dba` ·
    `reviews/*` → `analyst` · `requirements/*` and `manuals/*` → `tico` · `test-plans/*` and
    `test-reports/*` → `qa-engineer`. **This table controls where it disagrees with a document's own
    `Owner:` field** — a `reviews/*` document authored by some other specialist still flips via
    `analyst`; only the `-ml` and `-graph` rows follow the owner, and they do so because they are
    named rows here, not by inference.
  - **The whole lifecycle, one line:** `grep -m1 -H 'Status:' docs/plans/*.md`.
  - **An existing `m<n>-` filename prefix is part of a name, not a lifecycle claim** — nobody
    should read meaning into it, and nobody should "fix" it.
- The root `CLAUDE.md` contains only `@AGENTS.md` — this file is the single source of truth for
  root-level context; per-component context files carry the detail.
