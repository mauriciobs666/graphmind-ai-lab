# CPG MCP server/tool rename — Feature Requirements
> **Status:** Ready for design · **Owner:** `tico` · **Tracks:** — · **Last updated:** 2026-08-19

## Intent
The MCP server and its single tool are still named after CPG (`cpg/mcp/`, `.mcp.json` server key
`"cpg"`, tool `mcp__cpg__query`), even though M5 (`docs/requirements/generic-cypher-mcp.md`)
already widened it into a general, graph-agnostic Cypher tool — it now reaches any FalkorDB
graph, not just Code Property Graphs, and M6 (`docs/requirements/generic-cypher-mcp2.md`) is
rolling that generic use out to the whole agent team's working memory. The stakeholder noticed
the name no longer describes what the tool does, and wants it renamed to reflect that.

## Problem & current state
`cpg` was the right name when the tool was purpose-built and scoped to CPG analysis
(`docs/requirements/cpg-query-access.md`). Since M5, the tool is mechanically and by-decision
generic — any named FalkorDB graph, read and (narrowly, attributed) write — but every visible
surface still says "cpg": the directory (`cpg/mcp/`), the `.mcp.json` server key, the tool name
an agent actually invokes (`mcp__cpg__query`), and the README/docs describing it. A newcomer (or
an agent reading its own routing description) sees "cpg" and reasonably assumes CPG-only scope,
which is no longer true.

This name is referenced across **60+ files** repo-wide — not just the `cpg/` component itself,
but `claude/AGENTS.md`, multiple agents' operative prompts and kaizen history, `skills/`
(`cpg-analysis`, `joern-cpg`, `agent-maintenance`), and a long list of `docs/plans`, `docs/reviews`,
`docs/test-plans`, `docs/test-reports` — even `mcp-monitor/`'s own docs, which cite it as an
example. This is a wide, cross-component rename, not a cosmetic single-file tweak.

## User stories
- As **any agent**, I want the MCP tool's name to reflect what it actually does (generic Cypher
  over any graph, not CPG-only), so I don't misjudge its scope just from the name.
- As **a reader of `claude/AGENTS.md` or an agent's own routing description**, I want the tool's
  name to match its real, current capability, so I don't need to already know the M5/M6 history to
  understand what it does.
- As **the team**, we want the rename to land as one coordinated change, not a partial state where
  some references say "cpg" and others say "cypher."
- As **`graph-dba`** (owner of the actual Joern CPG pipeline), I want the genuinely CPG-specific
  naming — the `cpg-analysis` skill, `cpg_<component>` graph names, the top-level `cpg/` directory
  — left untouched, so this rename doesn't blur what's actually CPG-scoped versus generic.

## Functional requirements
- **FR-1** — The MCP tool agents invoke is renamed from `mcp__cpg__query` to
  `mcp__cypher__query`.
- **FR-2** — The `.mcp.json` server key (which produces the tool's `mcp__<key>__query` prefix) is
  renamed from `"cpg"` to `"cypher"`, consistent with FR-1.
- **FR-3** — The `cpg/mcp/` subdirectory itself is renamed/relocated to match the new identity —
  the top-level `cpg/` directory, and its role as the CPG component's home (including
  `.cpg-artifacts/`), is **unaffected**; only the generic-tool subdirectory moves.
- **FR-4** — Every currently **active** (non-archived) document, agent prompt, and skill that
  describes the tool by its old name (`mcp__cpg__query`, the `cpg` server key, or `cpg/mcp/` as a
  path) is updated to the new name. **Archived documents are not edited** — they remain an
  accurate historical record of the tool's name at the time they were written, per this repo's
  own archived-document convention (a frozen document is not amended for a later rename).
- **FR-5** — Code that implements or exercises the tool (`cpg/mcp/server.py`, its test suite,
  `docker-run.sh`/`build.sh` and any embedded references) is updated to the new name/location.
- **FR-6** — The rename lands as a **single atomic change** — no dual-name or compatibility-alias
  period. Once shipped, the old tool name (`mcp__cpg__query`) no longer resolves.
- **FR-7** — Genuinely CPG-specific naming is explicitly **not** part of this rename: the
  `cpg-analysis` skill, the `joern-cpg` skill, `graph-dba`'s Joern build pipeline, and
  `cpg_<component>` graph names all keep "cpg" — this delivery renames only the MCP tool/server
  that used to be scoped to CPG and now is generic.

*Context for the architect (not a requirement):* where `cpg/mcp/`'s new physical home actually is
(renamed in place under `cpg/`, promoted to its own top-level component, something else) is a
design decision, not specified here. So is whether the Docker image-tag scheme (a content hash of
build inputs, per root `AGENTS.md`) needs any adjustment as a result of the move.

## Out of scope
- **Editing archived documents.** They stay exactly as written, describing the tool by whatever
  name was current when they were authored (FR-4).
- **Any change to the tool's actual behavior or capability.** This is a naming-only change — the
  read/write mechanics, author/curator write enforcement, and (per M6) the team-wide query
  surface are unaffected.
- **CPG-specific naming** (`cpg-analysis`, `joern-cpg`, `cpg_<component>` graphs, the top-level
  `cpg/` directory identity) — untouched, see FR-7.
- **A compatibility/alias period for the old name.** Explicitly rejected — see FR-6.
- **Any change to `graph-dba`'s Joern CPG pipeline or the `cpg-analysis` skill's own mechanics.**
  Naming only; not touched by this delivery.

## Acceptance criteria
- **AC-1** — A repo-wide search for `mcp__cpg__query` finds zero hits outside archived documents.
- **AC-2** — `.mcp.json`'s server key is `"cypher"`, not `"cpg"`.
- **AC-3** — `cpg/mcp/` no longer exists as a directory; the relocated/renamed server starts,
  connects to FalkorDB, and answers a live `mcp__cypher__query` call identically to how
  `mcp__cpg__query` did before the rename.
- **AC-4** — Every active (non-archived) document, agent prompt, and skill that referenced the
  tool by its old name now references it by the new one — spot-checked against
  `claude/AGENTS.md`, `skills/cpg-analysis/SKILL.md`, and each agent prompt that cites the tool.
- **AC-5** — A diff confirms no unintended changes to the `cpg-analysis` skill, the `joern-cpg`
  skill, `graph-dba`'s Joern pipeline docs, or any `cpg_<component>` graph name (FR-7).
- **AC-6** — After the rename ships, a call using the old tool name (`mcp__cpg__query`) fails or
  is otherwise unavailable — not silently supported alongside the new one (FR-6).

## Open questions
*(none)*

## Decision log
- 2026-08-19 — Session opened. Raised mid-interview during the M6 (`generic-cypher-mcp2`)
  session: stakeholder noted "cpg doesn't reflect that it is now generic" and asked `tico` for a
  naming suggestion. `tico` offered an informal opinion (`cypher` over `graph` or a
  `falkordb`-flavored name, the latter risking confusion with the *official* `@falkordb/mcpserver`
  that M5's requirements doc already discusses and rejects), explicitly flagged as a suggestion,
  not a decision. Stakeholder chose to track the rename as its **own** delivery rather than fold
  it into M6, given the 60+-file blast radius — new document opened here, own topic slug (not a
  family member of `generic-cypher-mcp`/`generic-cypher-mcp2`, since this is about the tool's own
  identity, not the kaizen-inbox rollout).
- 2026-08-19 — New name? → **`cypher`** — `mcp__cypher__query`, matching the "generic Cypher MCP"
  language already used for the underlying feature.
- 2026-08-19 — Rename depth: tool name only, or the component's full identity (directory,
  `.mcp.json` key, docs)? → **Full identity, everywhere.**
- 2026-08-19 — Does genuinely CPG-specific naming (`cpg-analysis`, `joern-cpg`,
  `cpg_<component>` graphs) also change? → **No, untouched** — those stay "cpg" because it's
  still accurate for them; only the now-generic MCP tool/server is renamed.
- 2026-08-19 — Does the top-level `cpg/` directory itself rename, given it's documented
  repo-wide as "the CPG component home" and also holds `.cpg-artifacts/` (real CPG build
  output), separate from `cpg/mcp/`? → **Only `cpg/mcp/` moves/renames** — the top-level `cpg/`
  directory keeps its name and CPG-component identity.
- 2026-08-19 — Atomic rename vs. a transition/compatibility period? → **Atomic** — one
  coordinated change, no dual-name alias period; the old tool name stops resolving once this
  ships.
- 2026-08-19 — Readback delivered and confirmed. Stakeholder: "yes please." **Status → Ready for
  design.** No material assumption left unconfirmed; Open questions is empty.
