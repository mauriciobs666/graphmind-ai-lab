# Kaizen — Change History: graph-dba

> Dated log of actual changes to the `graph-dba` agent. Most recent first.

## 2026-08-11 — Inbox distillation: 7 entries — 3 promoted to `falkordb-quirks.md`, 2 discarded as already self-corrected, 2 discarded as already covered in `cpg-model.md`

- **What:** `cobb` processed all 7 entries in `graph-dba/kaizen/inbox.md` (§5).
- **Promoted:**
  - `redis-cli`'s `CYPHER` preamble needing quoted literals (not `k=v` trailing args), and a
    non-aggregated `OPTIONAL MATCH` fan-out key being a real grouping key (not a safe-to-assume
    constant) beside `collect(DISTINCT …)` → two new entries in `claude/graph-dba/
    falkordb-quirks.md`. The second one also **fixed an incorrect invariant claim** in
    `falkor-chat/docs/QUERIES.md` §11.2's own footnote, which asserted `start.key` is constant "so
    the grouping is well-defined" as if it were an engine property — it's a schema-level premise
    (exactly one `START` edge) that K-034 is what actually keeps true.
  - `pipeline.sh --reset` running `GRAPH.DELETE` invisibly to the destructive-ops `PreToolUse`
    guard — **checked against the live guard script before writing this up, and the gap this
    entry reported (2026-07-30) was already closed 2026-08-08 (C-311)**: `guard-destructive-ops.sh`
    now basename+flag-matches `pipeline.sh ... --reset` directly (per `claude/AGENTS.md`'s "Hook
    machinery" section, which already documented the fix). Rewrote the `falkordb-quirks.md` entry
    to state the fix + the generalizable lesson (a command-string guard needs an explicit clause
    per destructive wrapper script) instead of re-filing an already-closed gap as an open one. This
    is the third `falkordb-quirks.md` entry, closing out the "3 promoted" count above.
- **Discarded — already covered in `skills/joern-cpg/references/cpg-model.md` (2, both 2026-07-19):**
  the `pysrc2cpg` call-graph directional-asymmetry finding (caller matching by `CALL.NAME`, not the
  resolved `CALL` edge — already in "Consumer-query facts" `:116`/`:121`) and the `FILENAME`/`AST`
  vs. `CONTAINS`/`REACHING_DEF` intraprocedural-scope finding (already in the same section, `:140`).
  Both were re-checked against the live file, not assumed stale.
- **Discarded (already self-corrected in-run, before this pass):** the `db.indexes()`
  vector-dimension entry — this one actually came from **`teco`'s** inbox, not `graph-dba`'s (see
  `claude/teco/kaizen/history.md`'s own 2026-08-11 entry for its disposition; removed from this
  entry, which previously double-claimed it — nit n-2 in `docs/reviews/kaizen-distillation-2026-08.md`).
  The reachability-sandbox correction pair (two entries narrating the same finding, both
  `graph-dba`'s own) — already self-corrects in the second entry's own text; folded as one line
  into `skills/cpg-analysis/SKILL.md`'s query-usage guidance ("probe reachability, don't assume").
- **Also folded in from other agents' inboxes (not counted in this entry's "7 entries," logged in
  those agents' own history entries):** the FalkorDB `RESULTSET_SIZE` silent-cap finding
  (`qa-engineer`'s inbox) → `falkordb-quirks.md`, `cpg/mcp/README.md` (corrected an overclaim —
  "the `rows=` figure is always the true total" was false above 10k rows), and
  `skills/cpg-analysis/SKILL.md`'s gotcha list (new #6). The **no-string-repetition-operator**
  finding is `coder`'s own inbox entry and its promotion is logged solely in `coder`'s history
  entry — this entry previously double-claimed it too (nit n-1); removed here.
- **M-4 follow-up, verified closed:** `cpg/mcp/server.py`'s module docstring ("Display-only
  truncation" bullet, `:20-22`) carried the same now-corrected `rows=`-is-always-exact overclaim as
  `cpg/mcp/README.md`, in the most authoritative of the three sites (flagged by `analyst`'s review,
  M-4). `teco` fixed it directly — confirmed present and reads correctly: "the reported row count
  is exact below FalkorDB's `RESULTSET_SIZE` (default 10000), at or above which it is itself a
  cap." All three sites (`falkordb-quirks.md`, `cpg/mcp/README.md`, `cpg/mcp/server.py`) are now
  consistent; `skills/cpg-analysis/SKILL.md` was always correct on this point.
- **Verified:** `bash claude/scripts/audit-team.sh` clean. `db.indexes()` correction and
  `RESULTSET_SIZE` figures cross-checked against the entries' own cited commands/outputs, not
  re-run live (no FalkorDB access from this session).
- **Docs touched:** `claude/graph-dba/{kaizen/{history,inbox,plan},falkordb-quirks.md}` ·
  `falkor-chat/docs/QUERIES.md` · `cpg/mcp/README.md` · `skills/cpg-analysis/SKILL.md`.

## 2026-07-28 — `joern` agent retired; CPG generation folded in as an on-demand capability
- **What:** The standalone `joern` subagent (CPG specialist) was retired at the user's
  request — CPG generation work is genuinely rare, not frequent enough to justify a
  dedicated standing agent/persona. `claude/joern/` (agent, hooks, kaizen) was deleted
  entirely and its `~/.claude/agents/joern` deployment symlink removed. Its capability
  — driving the `joern-cpg` skill's parse → export → transform → load pipeline — folds
  into `graph-dba` as a small, explicitly on-demand addition:
  - **Frontmatter `description`:** the old "routes to joern" clause replaced with a
    direct capability clause — graph-dba drives Joern itself via the `joern-cpg` skill,
    stated as rare/on-demand so routing agents (and this agent itself) don't start
    suggesting CPG generation proactively. JDK/Joern-toolchain provisioning still routes
    to `devops` (added to that boundary bullet).
  - **Body:** a short "CPG generation (rare, on-demand)" paragraph after the knowledge-base
    bullets — a pointer to `skills/joern-cpg/SKILL.md`, not a restatement of its pipeline
    (joern's own prompt detail lived almost entirely in the skill already). No new hook —
    graph-dba's existing `guard-destructive-ops.sh` already covers `GRAPH.DELETE` for a
    CPG reload, so the destructive-ops step just gained a one-clause example.
  - **`joern:graph-dba` removed from `claude/scripts/audit-team.sh` `BOUNDARY_PAIRS`**
    (the pair no longer exists — folded into one agent, not two bordering ones).
- **Learnings distilled from `joern/kaizen/inbox.md` before deletion** (agent-maintenance
  skill §5 — verify → route → log; no "clear" step since the source is gone):
  - Three 2026-07-17 entries (Python frontend token `pythonsrc`, `pipeline.sh` masking a
    frontend failure as exit 0, per-statement `redis-cli` failing at scale) were already
    fixed in-skill and documented in `skills/joern-cpg/SKILL.md` Gotchas — verified current,
    discarded as duplicates, no new home needed.
  - Two 2026-07-19 entries (pysrc2cpg call-graph sparseness/asymmetry;
    framework-invoked entrypoints needing transitive test-gap reachability) were already
    folded into `skills/joern-cpg/references/cpg-model.md` "Consumer-query facts" and
    `skills/cpg-analysis/references/test-gap.md` respectively — verified current, discarded
    as duplicates.
  - The count-extraction gotcha (`redis-cli --no-raw` output must be parsed with
    `awk '/^[0-9]+$/{last=$0} END{print last}'`, not `grep -oE '[0-9]+' | tail -1`, which
    reads the stats line as data) → **added to `falkordb-quirks.md`** (the existing
    "read via `GRAPH.QUERY` materializes an empty key" entry), the concrete command the
    entry's prose was missing.
  - `FILENAME` being relative to the **parse root** (not the repo root) — a CPG can look
    correct by node/edge counts yet be silently useless to every `STARTS WITH` filter —
    → **added to `skills/joern-cpg/SKILL.md` Gotchas** (producer-side, actionable at build
    time) with a short cross-reference from `cpg-model.md`.
  - No `--exclude`/ignore mechanism in `build-cpg.sh`/`pipeline.sh` (scoping a parse means
    staging a copy of the wanted subtrees first) → **added to `SKILL.md` Gotchas**.
  - `cpg-to-falkordb.py --load` always re-transforms the export (no "replay this
    `.cypher`" mode) → **added to `SKILL.md` Gotchas**.
  - Sizing data point (~2,700 nodes / ~18,000 edges per Python source file with default
    overlays, from a real 41-file run) → **folded into `SKILL.md`'s existing "Scale"
    Gotchas bullet**, which now also points the streaming-loader concern at this agent's
    own kaizen plan instead of the retired agent's.
  - "Joern distribution not installed on this box despite the pinned-path assumption"
    (observed missing after a prior session had verified it — disk pressure or a wiped
    scratch dir) → **added as a caution to `SKILL.md` Prerequisites**: verify before
    running, treat a missing binary as a `devops` blocker, don't reinstall ad hoc.
  - The FalkorDB-start-script / v4.18.11 confirmation entry was already fully covered by
    this agent's own "This deployment" pin — discarded as a duplicate.
- **Plan items carried forward from `joern/kaizen/plan.md`** (opened as K-005, K-006 below;
  the four remaining parking-lot ideas — int-array columns stored as strings, a `graphml`
  export alternative, incremental re-CPG, `--repr` presets — were reviewed and **not**
  promoted: each is a speculative, low-value script-level idea already preserved verbatim
  in `joern/kaizen/history.md`'s final "Created" entry, which this repo's git history keeps).
- **Cross-references updated in the same change:** `claude/README.md` (dropped the `joern`
  row, updated the `graph-dba` row and the Kaizen/Hooks sections), `claude/AGENTS.md`
  (roster line, hook-machinery four-guards → three-guards), root `AGENTS.md` (roster line,
  `skills/` bullet and catalog entry), `claude/teco/teco.md` (routing table row + handoff
  contract), `skills/joern-cpg/SKILL.md` (description + guard reference + Gotchas above),
  `skills/joern-cpg/references/cpg-model.md` (the one "(the `joern` agent)" mention →
  `graph-dba`), `skills/cpg-analysis/SKILL.md` and its four `references/*.md` recipes (every
  "routes to the `joern` agent" / "(the `joern` agent)" phrase → `graph-dba`), and
  `skills/README.md` (both skill rows' "used by" column).
- **Team-coherence certification run after the change** (agent-maintenance skill §4):
  `claude/scripts/audit-team.sh` green (12 agents, no FAIL; `joern` folder absence is
  expected — the script discovers agents from disk, so it needed no code change beyond
  the `BOUNDARY_PAIRS` edit above). Judgment checklist: roster accuracy, handoff symmetry,
  and boundary reciprocity re-checked against the files listed above; no dangling `joern`
  agent references found in a repo-wide grep after the edits (see cobb's own kaizen history
  for the full certificate).
- **Why:** User decision after a short design discussion — CPG generation is rare enough
  that a dedicated standing persona isn't warranted; `graph-dba` already bordered the
  capability (it owned the loaded graph's FalkorDB model) and the retired agent's own
  procedural detail already lived almost entirely in the `joern-cpg` skill, so the merge
  is a small, mostly-pointer addition, not a restatement.
- **Plan items:** opens K-005 (streaming loader for large-repo CPGs) and K-006 (CPGQL
  script library) — see plan.md.

## 2026-07-27 — Unpinned from `model: opus` (team-wide)
- **What:** Removed the `model: opus` frontmatter line. The field is now absent, so the agent runs on Claude Code's default — `model` **defaults to `inherit`** (re-verified 2026-07-27 against `code.claude.com/docs/en/sub-agents`), i.e. the model the session/system default selects. No other frontmatter or body change.
- **Why:** User no longer wants the team locked to Opus. Model choice belongs at the session level (one decision, changeable with `/model`), not duplicated across 13 frontmatter files where it silently overrides whatever the user picked.
- **Plan items:** closes the standing "is opus warranted vs. sonnet?" revisit item — model tier is no longer this agent's decision.

## 2026-07-27 — Design notes open with the canonical header block (step 2 of `docs/plans/doc-reference-convention.md`)
- **What:** One line added to *How you work* item 7 (design work hands off by path): *"Open the document with the header block from root `AGENTS.md`."* It sits inside the item, so it binds to `<component>/docs/plans/<slug>-graph.md` and not to the inline consults the same item excludes. No frontmatter, hook, `description` or catalog change.
- **Why:** `docs/plans/doc-reference-convention.md` v1.4 §9.6 makes a three-field header (`Status:` · `Owner:` · `Tracks:`) the repo's lifecycle signal, replacing the milestone filename prefix and the move-to-`archive/` rule. `-graph` is in the closed role set and the design note is co-located with the architect's plan, so it needs the same header as its neighbours — and `graph-dba` is the only agent that writes one (zero exist today, which makes this the cheapest possible moment to fix the form). The line is a **pointer, not an inlined template** (v1.4 M20) — root `AGENTS.md` already reaches every agent through the root `CLAUDE.md` `@AGENTS.md` import — and byte-identical across the six producing prompts, because the convention's coverage check greps for it literally. `claude/README.md` row 12 re-checked — it cites the design-note path and the destructive-ops hook, not document structure; no edit needed.
- **Plan items:** none.

## 2026-07-24 — Description slimmed further (second team-wide token-cost pass)
- **What:** Frontmatter `description` compressed 919 → 832 chars (-9%): tightened phrasing, dropped restated detail, kept every routing/boundary clause. `claude/scripts/audit-team.sh` boundary-pair symmetry (graph-dba↔devops, graph-dba↔data-scientist, graph-dba↔joern) re-verified green. No body/catalog change.
- **Why:** All 13 agents' descriptions are auto-injected into every session and subagent spawn; the roster grew to 13 (graph-dba, joern added) since the first pass on 2026-07-11, and per-agent `/context` output showed room to cut further. User-requested via a `/context` token audit.
- **Plan items:** none.

## 2026-07-24 — Frontmatter: `permissionMode: acceptEdits`
- **What:** Added `permissionMode: acceptEdits` to the frontmatter, matching the same-day change across the team (`coder`, `tdd-engineer`, `frontend-engineer`, `architect`, `qa-engineer`, `analyst`, `devops`). File-edit/write approvals are session-scoped in Claude Code (unlike Bash approvals, which persist permanently per repo+command), so users otherwise have to re-grant write permission every session even with a global `Edit`/`Write` allow rule in `~/.claude/settings.json`.
- **Why:** Verified against current Claude Code docs (`hooks-guide.md` "Hooks and permission modes") that this doesn't weaken `graph-dba`'s own guard: its `guard-destructive-ops.sh` hook matches Bash command patterns (`GRAPH.DELETE`, `FLUSHALL`/`FLUSHDB`, volume wipes, `docker rm -f`), unrelated to `acceptEdits` (which only covers Edit/Write and common filesystem commands) — and `PreToolUse` hooks fire before any permission-mode check regardless, so a hook `"ask"` decision would survive even if the two overlapped.
- **Plan items:** none.

## 2026-07-17 — Two live-verified quirks added to `falkordb-quirks.md` (graph lifecycle)
- **What:** added to the "Ops, config & tooling" section: (1) a read via `GRAPH.QUERY`
  **materializes an empty graph key** (shows up in `GRAPH.LIST` with 0 nodes), whereas
  `GRAPH.RO_QUERY` on a non-existent graph returns `ERR Invalid graph operation on empty key`
  and creates nothing — so `RO_QUERY` is the side-effect-free emptiness probe; (2) never scan
  the whole `redis-cli` reply for digits to gauge emptiness (the execution-time stat line makes
  everything look non-empty — parse the lone integer output line). Both stamped verified
  2026-07-17 on v4.18.11.
- **Why:** surfaced during the `joern` CPG loader's live load test (K-001); these are generic
  FalkorDB engine facts, not joern-specific, so they belong in graph-dba's quirks KB (the
  established home) per the fold-in rule. cobb promoted them same-run.
- **Plan items:** none. (No change to `graph-dba.md` itself.)

## 2026-07-16 — Boundary reciprocity with new `joern` agent
- **What:** Appended a clause to the frontmatter `description`: generating a repository's Code Property Graph / operating the Joern toolset routes to the new `joern` agent (which owns CPG generation + the mechanical load), while graph-dba owns the code graph's FalkorDB model and tuning.
- **Why:** `joern` was created (CPG → FalkorDB pipeline) and borders graph-dba; the `joern:graph-dba` pair was added to `audit-team.sh` BOUNDARY_PAIRS, which requires each side's description to name the other (routing-contract symmetry). No other graph-dba behavior changed.
- **Plan items:** none.

## 2026-07-12 — Learning-capture loop: kaizen inbox + closing protocol (quirks-file exception kept)
- **What:** Added `kaizen/inbox.md` (append-only learnings inbox, seeded empty) and a "Learning capture" closing-protocol section to the prompt. Live-verified quirks of the pinned FalkorDB build keep their established direct home (`falkordb-quirks.md`, dated); the inbox captures everything else (client-SDK gotchas, lab conventions, non-FalkorDB tool quirks).
- **Why:** Team-wide self-improvement loop (agent-maintenance skill §5, added the same day) — graph-dba's quirks file was the pattern the loop generalizes, so it stays first-class rather than being rerouted through the inbox. Requested by the user.
- **Plan items:** none.

## 2026-07-11 — Design-note handoff contract + destructive-ops guard (certification fixes)
- **What:** Two additions from the same-day team-coherence certification. (1) "How you work" gained step 7: implementer-bound design work (data model, schema/DDL, ingestion/migration) is written to `<component>/docs/plans/<slug>-graph.md` and handed off by path (mirroring data-scientist's `-ml.md`); quick consults stay inline. teco's "Handoff contracts" list gained the matching entry in the same change. (2) Frontmatter now wires a `PreToolUse` Bash guard — `graph-dba/hooks/guard-destructive-ops.sh`, a thin wrapper over the new shared core `scripts/guard-destructive-ops.sh` — escalating `GRAPH.DELETE`/`FLUSHALL`/`FLUSHDB`/volume wipes/container force-removal to human approval; step 8 describes it (enforcement parity). Catalog rows updated (`claude/README.md`, `claude/AGENTS.md` hook machinery).
- **Why:** Certification found graph-dba was the only design-producing specialist without a written-deliverable path (its designs were the one paraphrased handoff in the teco pipeline), and the shared live FalkorDB was guard-protected only when `devops` acted. The guard also answers deferred K-001's revisit trigger ("starts mutating live FalkorDB data in ways that warrant a guardrail") with a narrower, destructive-shapes-only gate instead of a tool allowlist.
- **Plan items:** K-004 done (moved from plan.md); implements cobb K-011 on this agent's side.

## 2026-07-11 — Deep reference moved to on-demand falkordb-reference.md (token-cost pass, part 2)
- **What:** The "Core expertise" reference detail — LPG modeling patterns, the supported Cypher surface, index/constraint DDL, the `algo.*` catalog, config knobs, sizing/persistence/replication/cluster ops, ingestion, GraphRAG patterns — moved out of the prompt body into a new on-demand file `falkordb-reference.md` (7,459 chars), following the `falkordb-quirks.md` precedent; the body (18,656 → 9,440 chars, −49%) keeps what must always load: FalkorDB fundamentals (GraphBLAS/RAM-bound/no-APOC), the pinned deployment (v4.18.11, falkordb-py 1.6.x, vectorset caveat, version-line distinction), boundaries with devops/data-scientist, both knowledge-base pointers, how-you-work, and principles. Frontmatter unchanged; full audit pass.
- **Why:** graph-dba.md was the team's heaviest prompt and loads on every spawn, but the deep reference is only needed for the task area at hand (~2,300 tokens saved per spawn; the reference costs its ~1,900 only when read). Quirks stay separate: quirks are live-verified divergences of the pinned build, the reference is general practice — quirks win on conflict.
- **Plan items:** none.

## 2026-07-11 — Description slimmed (team-wide token-cost pass)
- **What:** Frontmatter `description` compressed from 1324 to 694 chars: capability lists tightened, reciprocal boundary prose reduced to short route-away clauses that still name the counterpart agents (audit check 6 boundary symmetry preserved — full pass green), and "how I work" detail dropped from the description since the prompt body already carries it. Routing semantics unchanged; no body/catalog changes needed.
- **Why:** All 12 agents' descriptions are auto-injected into every session and into every subagent spawn that carries the `Agent` tool; team-wide they cost 12,609 chars (~3.1K tokens) per injection. The pass cut them to 7,036 chars (~44%), saving ≈1,400 tokens per session/spawn with the same routing contract.
- **Plan items:** none.

## 2026-07-09 — data-scientist boundary clause (description + GraphRAG bullet)
- **What:** Frontmatter `description` and a new GraphRAG-section bullet state the split with the new `data-scientist` agent: graph-dba owns the in-graph mechanics (vector-index DDL, `db.idx.vector` queries, fusing similarity with traversal, their performance); the ML method above them — which embedding model, chunking strategy, how to evaluate retrieval quality — is the data-scientist's to design; GraphRAG layers get designed together. Pair `graph-dba:data-scientist` added to `claude/scripts/audit-team.sh` `BOUNDARY_PAIRS` (check 6, description symmetry).
- **Why:** The `data-scientist` agent was created 2026-07-09; "build/improve the GraphRAG layer" plausibly matched both agents, so the boundary must live in both descriptions (mirrors the devops split done the same day).
- **Plan items:** none.

## 2026-07-09 — Deployment pinned to v4.18.11 (edge retired)
- **What:** The lab's FalkorDB moved from `falkordb/falkordb:edge` (module `999999`) to the tagged release **`v4.18.11`** (module `41811`, Redis 8.6.3, released 2026-06-24). Rewrote the "This deployment" bullet (pinned release, reason from v4.18.11's documented behavior instead of moving-target/latest-`main` caveats; `vectorset` still loaded) and updated the quirks-section pointer. Re-stamped `falkordb-quirks.md`'s header: pinned build identified, quirks re-verified via the falkor-chat query suite (193/193 green on the new build); entries not exercised by the suite keep their edge-build dates pending individual re-probes. Catalog current-state refs updated (root `AGENTS.md`, `claude/AGENTS.md`, falkor-chat docs).
- **Why:** User decided to pin the latest release (cost/verification churn of tracking edge; the prompt's verify-live posture existed largely because the build was a moving target). The quirks file's own rule — re-verify on any tagged-release upgrade — was executed via the canonical suite.
- **Plan items:** none.

## 2026-07-09 — devops boundary clause (description + ops bullet)
- **What:** Frontmatter `description` and a new "Architecture & operations" bullet state the split with `devops`: graph-dba *designs* the deployment (RAM sizing, persistence choice, replication/cluster topology, ACLs) and owns everything inside the database; the container/Compose plumbing that runs it (service bring-up, volumes, networking, CI wiring) routes to `devops` — mirroring devops's existing deferral of data-model/query design here. The pair is mechanically enforced by `claude/scripts/audit-team.sh` check 6 (boundary-pair description symmetry). Catalogs synced (`claude/AGENTS.md`, `claude/README.md`, root `AGENTS.md`).
- **Why:** Description-symmetry sweep after teco's roster→routing-table restructure (same day): "spin up FalkorDB" plausibly matched both agents, and only devops's description named the boundary.
- **Plan items:** none.

## 2026-07-09 — Subagent-awareness on "ask one sharp question" (teco interface review follow-up)
- **What:** "How you work" step 1's "ask one sharp question" now carries the delegated-run fallback: when running as a subagent (e.g. delegated by `teco`), return the sharp question as the result instead of trying to ask mid-run — subagents can't ask. Catalog entry (`claude/AGENTS.md`) updated.
- **Why:** Sweep after the 2026-07-09 teco interface review found the "ask" phrasing assumed an interactive session across several delegates (same fix applied to coder, tdd-engineer, qa-engineer the same day).
- **Plan items:** none (out-of-band, driven by teco's 2026-07-09 review).

## 2026-07-05 — Absorbed generic FalkorDB engine quirks from falkor-chat/AGENTS.md
- **What:** `falkor-chat/AGENTS.md` had a "Live-verified FalkorDB facts" section mixing generic
  engine/dialect quirks (vector index DDL, index-before-constraint ordering, composite
  constraints, cross-graph edge no-op, union-label syntax, `length(path)` in ORDER BY, fulltext +
  `algo.*` confirmation, `GRAPH.RO_QUERY`/Bolt port, `TIMEOUT` default + write-path behavior,
  empty-`UNWIND` row collapse, the `FOREACH(CASE...)` idiom, the `exists()` pattern bug,
  `OR`-as-scan-anchor tuning, `GRAPH.MEMORY USAGE` under-reporting, `labels(coalesce())[0]`
  subscripting) with falkor-chat-specific corollaries (repository function names, mention
  write-block internals, keyset predicate profiling), generalized away from falkor-chat's specific
  property/label names. `falkor-chat/AGENTS.md` was trimmed to keep only the project-specific
  corollaries, each pointing back here for the general fact.
- **Mechanism (revised same day):** first draft inlined the ~20 quirks as a "Verified engine
  quirks" subsection in `graph-dba.md`; on review that bloats the always-on prompt with a
  *perishable, growing* fact list. Split instead into a **resource file** —
  `claude/graph-dba/falkordb-quirks.md` — modeled on the `agent-standards` skill's discipline
  (dated verification stamp, "cache not source of truth," re-verify on tagged-release upgrade,
  build sentinel `999999`). `graph-dba.md` keeps only a short stable-framing pointer that tells the
  agent to read the KB before writing/debugging Cypher/DDL/ops against this build. The whole agent
  folder is symlinked into `~/.claude/agents/graph-dba`, so the sibling file is reachable at both
  the repo path and `~/.claude/agents/graph-dba/falkordb-quirks.md`. `falkor-chat/AGENTS.md`'s
  back-reference was repointed from the prompt section to the resource file.
- **Why:** User: "the section ## Live-verified FalkorDB facts should be part of
  ../claude/graph-dba" — these are reusable DBA knowledge for *any* project on this FalkorDB
  build, not just falkor-chat, and belong on the agent so other projects benefit too. Resource-file
  form (not inline, not a shared skill) was the user's explicit call: keeps the prompt lean and the
  KB in the agent's own folder as a growing, curated store.
- **Plan items:** —

## 2026-06-05 — Deferred K-001 & K-002 (documentation-only for now)
- **What:** No agent/prompt change. User said "just document for now," so recorded the decision: **K-001** (tool permissions) → keep tools unconstrained, no `tools` key; **K-002** (live-FalkorDB profiling skill) → not building it yet, agent stays advice-only. Both marked ⚪ deferred with revisit triggers; active backlog is now empty.
- **Why:** User chose the documentation-only path rather than building tooling or restricting permissions. Logged so the items aren't re-proposed.
- **Plan items:** K-001 ⚪ deferred, K-002 ⚪ deferred.

## 2026-06-05 — Identified the deployment (edge engine on Redis 8 + Vector Sets); closed K-003
- **What:** User ran `redis-cli MODULE LIST` / `GRAPH.QUERY`. Findings: the **`graph` module reports version `999999`** = FalkorDB's **edge/untagged build** sentinel (a tagged release encodes as an integer, e.g. `41809` = v4.18.9), so the engine tracks latest `main`. It runs on **Redis 8.x**, evidenced by the separately-loaded **`vectorset`** module = **Redis Vector Sets** (`VADD`/`VSIM`), confirmed via redis.io docs. Module args observed: `MAX_QUEUED_QUERIES=25`, `TIMEOUT=1000`, `RESULTSET_SIZE=10000`. Edits to the agent: expanded the "This deployment" note (edge build → assume newest but verify + test live; Redis 8 base; `vectorset` present) and added a GraphRAG bullet distinguishing **FalkorDB's in-graph vector index** (`db.idx.vector.*` over `vecf32`, fuses with traversal — default for hybrid retrieval) from **standalone Redis Vector Sets** (`vectorset`/`VADD`/`VSIM`, not traversable — only when embeddings needn't live on the graph).
- **Why:** Closes K-003. An edge build can't be pinned to a semver, so the right move is to record the deployment reality and lean on verify-and-test rather than a release's notes. The dual vector stores on one box are a real GraphRAG footgun worth disambiguating.
- **Plan items:** K-003 ✅ (done). Active backlog now: K-001 (tool permissions, open), K-002 (optional live-FalkorDB skill, low).

## 2026-06-05 — Pinned the falkordb-py client + added version-line literacy (K-003 partial)
- **What:** User answered "version is 1.6.0." Verified via PyPI that this is the **`falkordb-py` Python client** (1.6.0 = 2026-02-21; 1.6.1 latest), not an engine version — the FalkorDB **module/server is on a separate `v4.x` line** (v4.18.9 as of 2026-06). Edited the agent's "Clients & ecosystem" bullet to pin the project's client at **`falkordb-py` 1.6.x** (with the `FalkorDB(...) → select_graph → query/ro_query` API shape and RESP+Bolt), and added a new bullet **"Mind the two version lines"** so the agent never conflates a client version with an engine version and reasons about dialect from the engine (v4.x) but client code from the SDK (1.6.x).
- **Why:** "1.6.0" is exactly the trap that makes an agent assume a wrong engine version; the dialect specifics it encodes are governed by the engine line, not the client. The doc-verified dialect details remain valid for current FalkorDB.
- **Plan items:** K-003 → 🟡 in-progress (client identified/pinned; remaining: confirm the deployed engine v4.x version and reconcile `GRAPH.*`/dialect specifics).

## 2026-06-05 — Repivoted from Neo4j-first to FalkorDB-first (major overhaul)
- **What:** Rewrote the agent to specialize in **FalkorDB** instead of Neo4j, after the user confirmed the lab uses FalkorDB. Verified specifics against docs.falkordb.com (two web searches + two doc fetches) before writing. Changes: new `description` (FalkorDB/Redis-module/GraphBLAS/GraphRAG triggers); added a **"What makes FalkorDB different"** section (sparse-matrix/GraphBLAS traversal as matrix multiplication, in-memory RAM-bound sizing, Redis-module ops model, multi-graph multi-tenancy, OpenCypher *subset* with no APOC/GDS/Fabric, `GRAPH.*` command surface). Reworked all core-expertise sections: modeling (added matrix-aware supernode reasoning + one-graph-per-tenant guidance); **Cypher on FalkorDB** (OpenCypher dialect, `GRAPH.QUERY`/`GRAPH.RO_QUERY`, `GRAPH.EXPLAIN`/`GRAPH.PROFILE` instead of Neo4j `PROFILE` prefix, built-in `algo.*` procedures replacing GDS, batched `UNWIND` writes); **indexing & constraints** (range/full-text `db.idx.fulltext.*`/vector `db.idx.vector.*`, `GRAPH.CONSTRAINT` unique/mandatory); **architecture & operations** (RAM sizing first, RDB/AOF persistence, primary/read-replica async replication, Redis Cluster with *graph-per-shard* and no single-graph sharding, Sentinel, `GRAPH.CONFIG`/`THREAD_COUNT`, `GRAPH.SLOWLOG`, Redis ACL/TLS security, SDK/Cloud ecosystem); and a dedicated **GraphRAG/knowledge-graphs** section (vector+graph hybrid retrieval, multi-tenant KGs, GraphRAG-SDK). Updated working method, principles, and communication style to FalkorDB realities. Kept LPG cross-awareness (Neo4j/openCypher/GQL) for porting and the RDF/SPARQL boundary.
- **Why:** User: "we will use falkordb not neo4j, please review everything." Almost every Neo4j-specific claim (APOC, GDS, Fabric, causal cluster/Raft, `neo4j-admin import`, page cache) was wrong for FalkorDB and had to be replaced.
- **Plan items:** reframes K-002 (companion skill now FalkorDB-specific: `redis-cli`/`GRAPH.PROFILE`) and promotes GraphRAG from parking-lot idea to a core section. Updated README.md and CLAUDE.md catalogs.

## 2026-06-05 — Dropped tenure-boast framing
- **What:** Removed "with decades of hands-on experience running graph databases in production" from the opening line; it now reads "You are a **graph database administrator and data architect** who runs graph databases in production." Kept the role label (sets altitude) but cut the tenure brag.
- **Why:** User feedback — the "decades of experience" framing makes agents sound cocky and adds nothing to behavior. Applied collection-wide (also tdd-engineer, dra-claudia).
- **Plan items:** —

## 2026-06-05 — Agent created
- **What:** Initial authoring of the `graph-dba` agent — a senior graph database administrator and data architect. Frontmatter `name: graph-dba`, `model: opus`, and a routing-oriented `description` with proactive-use triggers (design a graph model, write/optimize Cypher/GQL, plan cluster architecture/sizing/sharding, set up indexes/constraints, tune slow traversals, plan migrations/imports, ops questions). Body covers four core-expertise areas (graph data modeling; Cypher/GQL mastery; indexing & constraints; architecture & operations incl. GraphRAG/vector), a six-step working method (access-patterns-first, match existing conventions, show the model concretely, justify by traversal cost, prove perf via PROFILE, respect engine/version/edition boundaries), seven principles, and a communication style. Scoped **Neo4j/Cypher-first** but explicitly aware of the wider LPG world (openCypher, ISO GQL, Memgraph, Neptune) and honest about the RDF/SPARQL boundary.
- **Why:** User asked for a new agent: "graph database administrator who knows cypher query, data modeling, architecture and best practices." Fits the lab's focus (repo `graphmind-ai-lab`).
- **Plan items:** seeded K-001 (tool-permissions decision), K-002 (optional live-Cypher companion skill), plus parking-lot ideas (GraphRAG depth, PROFILE operator cheat-sheet, opus-vs-sonnet, multi-engine portability).

## 2026-06-05 — Docs updated (discoverability)
- **What:** Registered `graph-dba` in the collection catalog `claude/README.md` (table row + kaizen index link) and in the agent-context file `claude/CLAUDE.md` (Agents list).
- **Why:** Dual-audience documentation rule — keep humans (README) and other agents (CLAUDE.md) in sync the moment the agent is created.
- **Plan items:** —
