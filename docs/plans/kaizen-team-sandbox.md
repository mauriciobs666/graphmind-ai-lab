# Agent-team graph sandbox — Implementation Plan

> **Status:** active · **Owner:** `architect` · **Tracks:** — (M<n> TBD) · **Version:** 5 · **Reviews:** `docs/reviews/kaizen-team-sandbox.md`

*2026-08-31: revised per `docs/reviews/kaizen-team-sandbox.md` (needs changes) — decided `.mcp.json`
sandbox-entry git lifecycle (§3.2, §4 steps 1–3), closed the AC-5 recording gap for undocumented
work (§3.3, §4 step 6, §5), fixed AC-2's verification to not leave cruft on production (§5), and
adopted both minor suggestions (image-ref sourced live, default resource limits — §4 step 1, §6).*

*2026-08-31: revised again per `docs/reviews/kaizen-team-sandbox.md` Pass 2 (needs changes) — gave
the `.mcp.json`-editing scripts a `jq`→`python3` fallback matching `claude/scripts/*.sh` convention
(§4 steps 1–2), completed the undocumented-work kaizen-entry template to all seven standard fields
(§3.3), and made the port-reuse-on-re-run fix that actually makes §6's idempotent-re-provisioning
mitigation true (§4 step 1, §6).*

*2026-08-31: revised again per `docs/reviews/kaizen-team-sandbox.md` Pass 3 (needs changes) —
replaced the Pass-2 `jq`→`python3` JSON-round-trip mechanism, empirically shown to reformat the
untouched `cypher` entry, with a text-anchored splice that touches only the owned
`cypher-sandbox-<slug>` block and drops the `jq`/`python3` dependency entirely (§4 steps 1–3, §3.2,
§6); made provisioning's `.mcp.json` write "add-or-correct" instead of "add-if-missing" so a stale
entry left by an incomplete teardown gets fixed automatically (§4 steps 1–2, §6).*

*2026-08-31: revised again per `docs/reviews/kaizen-team-sandbox.md` Pass 4 (approved with
suggestions) — named the safe raw-file-read splice idiom (temp file + `sed '/anchor/r tempfile'`)
in §4 step 1 explicitly, warning against passing the literal JSON block through `awk -v`, which
silently mangles its `\"` escapes.*

## 1. Goal & scope

Give `kaizen_team` — the shared FalkorDB working-memory graph every Claude Code agent writes
`:KaizenEntry` nodes into — a development/production isolation boundary, so that development work
on the upcoming bigger agent/graph integration (and any future work that touches `kaizen_team`)
cannot corrupt production data, conflict with production schema, or degrade/interrupt production's
availability. Scope is exactly the three interference vectors named in
`docs/requirements/kaizen-team-sandbox.md` (FR-1/2/3): data, schema, engine. Out of scope, per that
doc: `cpg_*` graphs, `falkor-chat`'s own workspace/production concerns, a fixed universal promotion
recipe, continuous/automatic sync, and a mandatory sandbox-for-everything policy.

This plan designs the standing **mechanism** (FR-8) — provisioning, MCP wiring, a request/negotiate
convention, a promotion template — not a one-time setup for the upcoming integration project alone,
and not a piece of tooling any single future feature needs to reinvent.

**CPG:** considered, not relevant — `GRAPH.LIST` (live-checked this session) shows no
`cpg_cypher-mcp`/`cpg_claude` graph; only `cpg_falkorchat` and `cpg_salesperson` are loaded. This
design's changes are provisioning scripts, `.mcp.json` config, and docs in `cypher-mcp/`/`claude/`,
not source in a component with a loaded CPG.

## 2. Context & findings

- **Today's shared instance.** `falkordb-dev` is a single Docker container
  (`falkor-chat/scripts/start_falkordb.sh`, image `falkordb/falkordb:v4.18.11`, ports 6379/3000,
  named volume `falkordb-data`) hosting every graph used across the repo, `kaizen_team` included
  (confirmed live: `cpg_falkorchat`, `cpg_salesperson`, `kaizen_team`, several `ws:*` workspaces,
  `reference`, `test` all coexist on it — `GRAPH.LIST` output above). `start_falkordb.sh` is
  **already parameterized** for exactly the override this plan needs: `FALKORDB_IMAGE`,
  `FALKORDB_PORT`, `FALKORDB_WEB_PORT`, `FALKORDB_CONTAINER_NAME` are all env-overridable, and a
  second container on a different port/name/volume is a proven, live-used pattern in this repo
  already — `falkor-chat/docs/reviews/unique-constraint-oversized-value-crash-rca.md` reproduced a
  crash bug via `docker run -d --name falkordb-repro-k049 -p 16379:6379 falkordb/falkordb:v4.18.11`,
  explicitly "never `falkordb-dev`," precisely to avoid touching the shared instance.
- **The access path is the `cypher` MCP server, and it is already multi-instance-ready.**
  `cypher-mcp/docker-run.sh` forwards `FALKORDB_HOST`/`FALKORDB_PORT` from its own environment into
  the container unmodified (`cypher-mcp/README.md` §"Environment variables": "All are read once at
  import... set for Claude Code in `.mcp.json`"). `.mcp.json` is a project-scoped Claude Code config
  file that can hold multiple named MCP server entries; each becomes a distinct tool namespace
  (`mcp__<server-name>__query`). Nothing in `cypher-mcp/server.py`'s write-authorization logic
  (`authorize_write()`) is instance-aware — it is identical code regardless of which FalkorDB
  process it's pointed at.
- **Write authorization is unaffected by this design.** The tool recognizes exactly the shapes
  described in its own MCP instructions: producer-write (`MERGE (:Agent)` + `CREATE
  (...)-[:PRODUCED]->(:KaizenEntry {...})`), the legacy `author:`-property write, and three curator
  shapes (`MENTIONS`-write, edge-resolve, full-node clear by `entryId`) restricted to
  `CYPHER_MCP_CURATOR_AGENTS` (default `cobb`). This logic is per-server-process, not
  per-graph-instance — pointing a second launch of the same, unmodified image at a second FalkorDB
  backend reproduces it exactly, so **no code change to `cypher-mcp/server.py` is needed** for this
  plan.
- **Schema DDL is rejected by the MCP tool outright, by design, on every instance.** Confirmed live
  during M7 (`docs/plans/kaizen-agent-ontology-graph.md` §6) and M8's S0 unit
  (`docs/plans/generic-cypher-mcp2-coordination.md`): `CREATE INDEX`/`GRAPH.CONSTRAINT CREATE` get
  the same rejection as any other non-`:KaizenEntry` write, with no curator carve-out. Production
  `kaizen_team`'s schema was provisioned via **direct `redis-cli GRAPH.QUERY`** against the
  container, by `graph-dba` — not through `mcp__cypher__query`. The exact DDL, index-before-
  constraint (FalkorDB requires a supporting exact-match index before a `GRAPH.CONSTRAINT CREATE`
  will accept the corresponding `UNIQUE` constraint), constraint keyword `NODE` (not `LABEL`):

  ```cypher
  CREATE INDEX FOR (e:KaizenEntry) ON (e.entryId)
  ```
  ```
  GRAPH.CONSTRAINT CREATE kaizen_team UNIQUE NODE KaizenEntry PROPERTIES 1 entryId
  ```
  ```cypher
  CREATE INDEX FOR (a:Agent) ON (a.agentId)
  ```
  ```
  GRAPH.CONSTRAINT CREATE kaizen_team UNIQUE NODE Agent PROPERTIES 1 agentId
  ```

  Constraint creation is async (`PENDING` → poll `CALL db.constraints()` for `status = OPERATIONAL`).
  A freshly provisioned sandbox instance needs this exact block replayed on it before it behaves
  like production for schema-dependent testing.
- **The `guard-destructive-ops.sh` hook is instance-agnostic, which is a feature here.**
  `claude/scripts/guard-destructive-ops.sh` (wired to `devops`, `graph-dba`, `qa-engineer` via
  frontmatter `hooks:`) escalates `docker volume rm/prune`, `docker rm -f`, `compose down -v`, and
  `FLUSHALL`/`FLUSHDB`/`GRAPH.DELETE` to the human regardless of which container the command
  targets. Tearing down a sandbox instance still gets a human-approval pause — a safety backstop
  this plan gets for free, not something it needs to build.
- **Root `docs/` is the established home for `kaizen_team`/`cypher-mcp` design work.** Despite
  `docs/BACKLOG.md`'s header framing it as "CPG code-graph component," prior `kaizen_team`-specific
  plans (`kaizen-agent-ontology.md`, `generic-cypher-mcp2.md`, and this task's own requirements doc)
  all live at repo-root `docs/`, not under `claude/docs/` (which holds agent-*prompt*/process design,
  e.g. `agent-permission-friction.md`, `security-expert.md`) or under a per-component `docs/`. This
  plan follows that precedent and the task's own instruction: `docs/plans/kaizen-team-sandbox.md`.
- **No existing mechanism for cross-instance data copy is documented or verified in this repo.**
  Nothing under `claude/graph-dba/` or elsewhere records whether FalkorDB supports `MIGRATE`/`DUMP`+
  `RESTORE` for a single graph key across two Redis processes. This plan does not assert an answer —
  it names both the Cypher-level fallback (always works, already the tool's normal write path) and
  the Redis-level option as something `graph-dba` verifies against current FalkorDB docs at first
  use (§6, S3).

## 3. Design & rationale

### 3.1 Isolation mechanism: a separate FalkorDB *instance* per sandbox request, not a separate graph key on the shared instance

**Chosen:** each sandbox is its own Docker container running the **same pinned FalkorDB image** as
`falkordb-dev`, its own named volume, its own port, holding a graph still named `kaizen_team`
(not `kaizen_team_sandbox`) inside that separate engine. Reached via a second, additively-registered
`.mcp.json` MCP server entry pointing at that container's port, giving a distinct tool name
(`mcp__cypher-sandbox-<slug>__query`) alongside the untouched `mcp__cypher__query`.

**Why this over a same-instance, separate-graph-key sandbox (e.g. `kaizen_team_sandbox` on
`falkordb-dev` itself).** That alternative satisfies FR-1 (data) and FR-2 (schema) cheaply — FalkorDB
is multi-tenant at the graph-key level, and a `CREATE`/DDL against one key cannot touch another
key's data or schema. It fails FR-3 outright: a single Redis/FalkorDB process is one address space
and one query executor: heavy load or a crash in the sandbox's graph-key workload degrades or takes
down the *engine*, and every other key on it — including production `kaizen_team` — with it. FR-3
is explicit ("load, crashes, or restarts caused by development work must not degrade or interrupt
production's availability") and the acceptance criteria test it directly ("heavy query load or a
crash in the sandbox... agents' real-time reads/writes against production `kaizen_team` continue to
succeed"). Only a separate process satisfies that structurally. This also matches the stakeholder's
own stated preference ("separate FalkorDB instances") and the repo's own precedent for exactly this
scenario (the crash-repro RCA above, run off `falkordb-dev` for the identical reason).

**Why keep the graph key named `kaizen_team` inside the sandbox, rather than `kaizen_team_sandbox`.**
The isolation boundary is the *instance* (a distinct `FALKORDB_HOST:PORT`, pinned into a distinct MCP
server process at launch), not the graph name — so every existing query, script, and the
`cobb`/agent-facing Cypher shapes in `claude/AGENTS.md`'s Learning-capture block work against the
sandbox completely unmodified. There is no sandbox-specific code path anywhere, and no risk of a
copy-pasted query silently landing in production by graph-name typo on a shared engine, because the
sandbox tool's connection never opens a socket to production's port at all — it is a structural
guarantee, not a naming discipline.

**Rejected: a Redis logical-DB (`SELECT n`) split, or a FalkorDB Cloud/managed replica.** The former
still shares one engine process (same FR-3 failure as the graph-key alternative, and is not how
FalkorDB's graph tenancy model works — graphs are Redis keys within one logical DB, not separated by
`SELECT`). The latter is real infra cost and operational complexity disproportionate to a
case-by-case, non-mandatory dev sandbox (FR-5) — worth reconsidering only if demand for concurrent
sandboxes becomes high enough that ad hoc `docker run` containers stop being cheap enough (§7).

**Trade-off accepted:** the sandbox needs an explicit schema-bootstrap replay (§2's DDL block) and,
for FR-3 to hold under host-level, not just engine-level, contention, benefits from container
resource limits (§6, hardening note) — both are real but small, one-time-per-instance costs, in
exchange for a structural (not disciplinary) isolation guarantee on all three vectors.

### 3.2 Per-request naming, not one perpetual shared sandbox slot

Each sandbox is named for the requesting feature (`falkordb-sandbox-<slug>`, MCP entry
`cypher-sandbox-<slug>`), provisioned on request and torn down after promotion/abandonment — not one
long-lived shared "the sandbox" container. FR-8 makes the capability open to **any** agent or
contributor, not just this integration project; a single shared instance would let two unrelated
concurrent efforts collide on each other's data/schema, recreating the exact "no boundary between
trying something out and what's relied on" problem this plan exists to remove, one level down.
**Trade-off:** `.mcp.json` accumulates one entry per active sandbox; provision/teardown (§4 steps 1–2)
add/remove it programmatically, so the working tree returns to its pre-sandbox state once teardown
runs — accepted, because a stale unreachable entry (if teardown is skipped) fails safely (the tool
already documents "FalkorDB unreachable" as an ordinary, non-crashing error), whereas silent
collision between two sandboxes would not fail safely at all. If concurrent sandbox demand turns out
to be consistently low in practice, devops may simplify to a single reused slot later — that is an
operational call within devops's FR-9 ownership, not a decision this plan makes now.

**`.mcp.json` git lifecycle: sandbox entries are deliberately never committed.** `.mcp.json` is a
tracked, committed repo file, but its *sandbox* entries are host/port-specific, per-request state —
a `FALKORDB_PORT` bound to one contributor's own `docker run` container is meaningless (unreachable)
to anyone else's checkout, and FR-8's "any agent or contributor" plus §3.2's own per-request naming
mean several unrelated, concurrent sandboxes could otherwise collide as churn/merge conflicts in one
shared committed file for zero benefit — nobody but the requester can reach another contributor's
container anyway. So the entry is a local, uncommitted working-tree edit, made and removed by the
provision/teardown scripts themselves (§4 steps 1–2), never by a devops-owned commit. The only
committed content this plan touches is the documentation in `cypher-mcp/README.md` (§4 step 4). The
existing `cypher` entry itself is never modified, not even reformatted (FR-4) — the scripts perform a
**text-anchored splice**, not a JSON parse/re-serialize round trip, specifically so nothing outside
the one `cypher-sandbox-<slug>` block they own is ever touched (mechanism: §4 step 1). Residual risk:
a git operation that touches the working tree mid-session (`checkout`, `stash`, `reset`) can drop the
local sandbox entry before teardown runs; the mitigation is cheap re-provisioning, not data loss —
the entry is disposable config pointing at a still-running, unaffected container/volume, so step 1's
script can be re-run (**add-or-correct**, not merely add-if-missing: it always writes the current,
authoritative entry, so it fixes a missing entry and a stale one identically) to regenerate exactly
the same entry without touching the container or its data. This residual risk is also logged in §6.

### 3.3 The requester/devops split (FR-9) reuses the existing plan-doc convention — no new document kind

FR-9 requires the requester to specify scope/isolation-needs/risk-call without provisioning
anything themselves. Rather than inventing a new document kind or filename role (the repo's role set
— `(none)`/`-coordination`/`-ml`/`-graph`/`-rca`/`-impl`/`-report` — is closed, and a "-sandbox" role
would violate it), this plan adds one required subsection, **"Sandbox & promotion"** (template in
§4, step 4), to any `docs/plans/<slug>.md` whose work touches `kaizen_team`. That is exactly the
document the requester (typically `architect`, for a designed feature) already produces, so nothing
new is asked of a requester beyond one more subsection in a document they were writing anyway. Devops
reads scope/isolation-requirements straight out of that subsection when provisioning — the same
architect→specialist handoff this repo already uses everywhere (e.g. `docs/plans/
generic-cypher-mcp2-coordination.md`'s S0/G1 units). For smaller work with no `docs/plans/<slug>.md`
of its own, FR-5's "no fixed checklist... case by case" governs *how* the risk call is made, not
*whether* it is recorded — AC-5 has no size carve-out ("a recorded decision... on whether it goes
through the sandbox — not an assumption either way," for any work touching the graph). So
undocumented work still leaves a minimal recorded artifact: a one-line kaizen entry, written via the
requester's own existing producer-write shape (no new mechanism, no devops involvement), e.g.:

```cypher
MERGE (a:Agent {agentId: '<requester>'})
CREATE (a)-[:PRODUCED]->(:KaizenEntry {
  entryId: '<uuid4>', date: '<YYYY-MM-DD>',
  fact: 'kaizen_team schema/data touched directly, no sandbox — <one-line risk call>',
  evidence: '<what was actually run against production, and how it was confirmed — e.g.
    "ran <DDL/write> directly against kaizen_team, confirmed via <check>">',
  context: '<one-line what/why>',
  suggestedHome: '<a real routing guess for cobb distillation — prompt | knowledge base |
    project docs | unsure>',
  createdAt: '<ISO-8601>'
})
```

All seven of `:KaizenEntry`'s standard fields, matching every other producer-write example in this
repo (not the abbreviated five this template shipped with in the prior revision) — `suggestedHome`
in particular is what `cobb`'s distillation routing reads. This is still proportionate to FR-5's "no
checklist" spirit — one entry, not a subsection — while satisfying AC-5's recording requirement for
the undocumented case.

### 3.4 Promotion (FR-7) is a template, not new tooling

Building bespoke promotion/sync tooling would contradict the requirements doc directly ("no fixed
universal promotion recipe," "no continuous/automatic sync"). Instead, promotion reuses the two
write mechanisms that already exist and are already trusted: `graph-dba` replays the exact validated
DDL against production directly (the same `redis-cli` path already used to provision *any*
`kaizen_team` schema, sandbox or production), and data — where "data" means new/changed
`:KaizenEntry`/`:Agent` records — moves via the tool's own existing producer-write shape, i.e. the
promoted entries are simply written to production the normal way. The only new artifact is the
per-feature migration-impact analysis that decides *which* of those two applies (§4, step 4's
template) — a decision, not a tool.

## 4. Step-by-step implementation

Sized for `teco` to break into units and route (infra → `devops`, schema/DDL/data-copy method →
`graph-dba`, team-doc pointer → `cobb`), mirroring how `kaizen-agent-ontology`/`generic-cypher-mcp2`
were delivered. A solo implementer can also execute steps 1→6 in order; step 7 is a documentation
convention adopted going forward, not a one-time build task.

1. **`devops` — provisioning script.** Add `cypher-mcp/scripts/provision-kaizen-sandbox.sh`:
   - Args: `--slug <name>` (required; becomes the container/volume/MCP-entry suffix), `--port <n>`,
     `--image <ref>` (default: parsed live from `falkor-chat/scripts/start_falkordb.sh`'s own
     `FALKORDB_IMAGE:-<pin>` default at script run-time — e.g. `grep -oP 'FALKORDB_IMAGE:-\K\S+'
     .../start_falkordb.sh` — never a hardcoded copy, so the two scripts cannot drift apart; see §6),
     `--cpus <n>` (default `1`) and `--memory <size>` (default `1g`) (conservative host-resource caps
     serving FR-3/AC-3, overridable for a load-test scenario that needs headroom), and
     `--bootstrap-schema` (default on; runs the DDL block from §2 via `redis-cli` against the new
     container once it answers `PONG`).
   - Behavior, **recovery path** — invoked with *only* `--slug` (no `--port`/`--image`/`--cpus`/
     `--memory` overrides) and a container named `falkordb-sandbox-<slug>` is already running:
     treat this as repairing a dropped `.mcp.json` entry (§6's residual risk), not a new
     provisioning request. Reuse the running container's actual bound port (`docker inspect`), skip
     `docker run`/schema-bootstrap entirely (container/schema/data are untouched), and go straight to
     the `.mcp.json` step — this is what makes re-running the script after a lost entry regenerate
     the *same* entry, closing the gap in §6's idempotent-re-provisioning mitigation.
   - Behavior, **new provisioning** — otherwise (no container of that name running, or one is running
     but the invocation passed an explicit `--port`/`--image`/`--cpus`/`--memory` that doesn't match
     what's already running): if a container of that name is already running with mismatched
     parameters, **fail loudly** ("`falkordb-sandbox-<slug>` is already running with different
     parameters — tear it down first (step 2) or choose a different `--slug`") rather than silently
     reusing or overwriting it — this is the §5 slug-collision edge case, and it stays distinct from
     the bare-rerun recovery path above precisely because a bare rerun never supplies conflicting
     overrides. Otherwise, proceed: `docker run -d --name falkordb-sandbox-<slug> -p <port>:6379
     --cpus=<cpus> --memory=<memory> -v falkordb-sandbox-<slug>-data:/var/lib/falkordb/data <image>`
     (mirrors `start_falkordb.sh`'s proven volume-mount path, `/var/lib/falkordb/data` — **not**
     `/data`, which persists nothing on this image per that script's own documented finding); default
     `--port`, if not given, is the first free port at/above 16380 (avoiding the RCA precedent's
     16379 and `falkordb-dev`'s 6379); wait for `redis-cli -p <port> ping` → `PONG`; if
     `--bootstrap-schema`, run the four DDL statements in order, polling `CALL db.constraints()` for
     `OPERATIONAL` before returning.
   - `.mcp.json` wiring — **text-anchored splice, add-or-correct, no JSON library at all:** the
     script never parses/re-serializes the whole file (a `json.load`→dict-mutate→`json.dump` round
     trip was tried and rejected — confirmed empirically to reformat the pre-existing `cypher`
     entry's `args` array from one line to four even when no value changed, breaking AC-4's
     byte-unmodified check and the "sandbox entries are the only diff" claim in §3.2; `jq`'s default
     pretty-printer has the same well-documented multi-line-array behavior). Instead: (1) if a block
     already exists for this slug — found by locating the line `"cypher-sandbox-<slug>": {` and its
     balanced closing `},` (brace-depth counting, not a naive first-`}` match, since the entry's own
     `"env": { ... }` sub-object contains braces too) — delete it first; (2) insert the entry (step
     3's shape, with the resolved port) as new lines immediately after the `"mcpServers": {` line.
     This is **add-or-correct**, not add-if-missing: a stale entry (wrong/dead port, e.g. left behind
     by an incomplete teardown, §4 step 2) is corrected the same way a missing one is added, by the
     same delete-then-insert. `bash` + `awk`/`sed`/`grep` is sufficient for this — no `jq`, no
     `python3`, no version-detection/fallback chain, no new prerequisite to document. **Insert the
     block via a raw-file read, never a shell-expanded variable:** write the entry text to a temp
     file and splice with `sed '/"mcpServers": {/r tempfile'` (or an equivalent raw-file-read
     splice) — passing the literal JSON block (which contains escaped quotes, `\"`) through
     `awk -v block="$block_text"` instead silently reinterprets the backslash escapes, collapsing
     `\"` to `"` and producing invalid JSON. If the
     `"mcpServers": {` anchor line isn't found (a malformed or hand-edited `.mcp.json`), the script
     fails loudly with a clear message rather than guessing where to insert. The script prints the
     entry it wrote plus the required session-restart reminder. This entry is a **local, uncommitted**
     working-tree edit (rationale: §3.2); the script never runs `git add`/`git commit` against it.
   - "Done" = a fresh (or reused, on re-provisioning) container running, `PONG`-reachable, with
     `kaizen_team`'s `Agent.agentId`/`KaizenEntry.entryId` indexes+constraints confirmed
     `OPERATIONAL`, and its `.mcp.json` entry present in the working tree (uncommitted).
2. **`devops` — teardown script.** Add `cypher-mcp/scripts/teardown-kaizen-sandbox.sh --slug <name>`:
   `docker stop`/`docker rm` the container, then `docker volume rm falkordb-sandbox-<slug>-data`
   (this last command is caught by `guard-destructive-ops.sh`'s `docker volume rm` pattern for any
   agent it's wired to — expected, not a bug, and the reason this script does not need its own
   confirmation prompt), then removes the corresponding entry from the (uncommitted) `.mcp.json` using
   the same brace-balanced text-anchored delete step 1 uses to correct a stale entry (no `jq`,
   `python3`, or any other new dependency — this is the same delete routine, just not followed by a
   re-insert), and prints the required session-restart reminder. "Done" = container and volume gone,
   and the `.mcp.json` entry is actually absent from the working tree — not just a printed reminder.
   If the container/volume teardown succeeds but the process is interrupted before the `.mcp.json`
   delete runs (or a contributor tears the container down manually, outside the script), the entry is
   left stale — harmless by design, since step 1's add-or-correct write will fix it (silently
   replacing the dead port) the next time this slug is provisioned; no separate detection/cleanup
   step is needed for that case.
3. **`devops` — `.mcp.json` wiring convention.** Document (in `cypher-mcp/README.md`, step 4 below)
   the entry shape step 1's script writes, and its git lifecycle:
   ```json
   "cypher-sandbox-<slug>": {
     "command": "bash",
     "args": ["-c", "exec \"$CLAUDE_PROJECT_DIR/cypher-mcp/docker-run.sh\""],
     "env": { "FALKORDB_HOST": "host.docker.internal", "FALKORDB_PORT": "<port>" },
     "timeout": 60000
   }
   ```
   Added alongside, never replacing, the existing `cypher` entry — this is the mechanical guarantee
   behind FR-4/AC-4, and step 1's text-anchored splice (not a JSON parse/re-serialize) is what makes
   it true byte-for-byte, not just in substance: no line outside the added/removed
   `cypher-sandbox-<slug>` block is ever touched, confirmed by a plain before/after file diff of the
   `cypher` entry, independent of git tracking. **Never committed:** the entry is local, per-contributor, per-request state (rationale
   and residual-risk mitigation: §3.2); a contributor running `git status` while a sandbox is active
   will see it as an ordinary uncommitted diff on `.mcp.json` until teardown (step 2) removes it. A
   session restart is required to pick up a new/changed `.mcp.json` entry (`cypher-mcp/README.md`'s
   existing documented fact); both scripts print this reminder.
4. **`devops` — `cypher-mcp/README.md` update.** Add a "## Sandbox instances" section (after
   "Environment variables") covering: why per-instance not per-graph-key (§3.1's rationale,
   condensed), the two scripts and their flags, the `.mcp.json` entry shape and its uncommitted git
   lifecycle (step 3), the schema bootstrap block (§2), the undocumented-work kaizen-entry template
   (§3.3) for work with no plan doc of its own, and — verbatim, for any `docs/plans/<slug>.md`
   touching `kaizen_team` to copy — the **"Sandbox & promotion" subsection template**:
   ```markdown
   ## Sandbox & promotion

   - **Scope:** <what this work touches — schema shape / bulk data / query behavior / etc.>
   - **Isolation requested:** <sandbox (data+schema+engine) | direct-to-prod — low risk>
   - **Risk/blast-radius call (FR-5):** <one-line judgment + why>
   - **Stakeholder negotiation, isolate-or-not (FR-6a):** <date — who — outcome>
   - **Seed strategy, if sandboxed:** <clean | copy-from-prod (scope) + why — graph-dba's call, §3.4>
   - **Migration-impact analysis, filled in at promotion (FR-7):** <what moves — schema only /
     schema+data / neither — and how each moves>
   - **Stakeholder sign-off for structural change, only if promoting a schema-shape change (FR-6b):**
     <date — who — outcome>
   ```
5. **`graph-dba` — schema/data-copy verification.** Provision one real sandbox instance via step 1's
   script as a live check; confirm the DDL block behaves identically to production's provisioning
   history (`db.indexes()`/`db.constraints()` → `OPERATIONAL`, matching M7's S0 precedent). Then
   check current FalkorDB documentation for whether a single graph key can be moved instance-to-
   instance via `MIGRATE`/`DUMP`+`RESTORE` (unverified in this repo today, §2) — if yes, document it
   in `cypher-mcp/README.md`'s new section as the fast path for a "copy-from-prod" seed; if no or
   unclear, document the Cypher-level fallback (read production via `mcp__cypher__query`, replay as
   `CREATE`s against the sandbox) as the supported method — it always works since it is the tool's
   normal write path, just slower for large data.
6. **`cobb` — team-doc pointer.** Add two sentences to `claude/AGENTS.md`'s existing `kaizen_team`
   paragraph (the one describing the `PRODUCED`/`MENTIONS` write shapes and `cobb`'s distillation):
   (a) a sandbox is available for `kaizen_team` dev work, request via `devops`, mechanics in
   `cypher-mcp/README.md` §"Sandbox instances"; (b) for small work with no plan doc of its own that
   still touches `kaizen_team` directly, record the risk call as a one-line kaizen entry (§3.3's
   template) instead — recording is not optional just because the work is undocumented (AC-5). This
   is what makes FR-8 ("standing capability... not limited to the stakeholder who requested it") and
   AC-5's recording requirement actually discoverable by any agent reading the file every agent
   already reads.
7. **Convention adoption, ongoing (no dedicated build step).** From this point on, any
   `docs/plans/<slug>.md` whose work touches `kaizen_team` includes the "Sandbox & promotion"
   subsection (step 4's template) — starting with the upcoming bigger agent/graph integration
   project's own plan. Recommended, not built by this plan: a one-line pointer to that convention in
   root `AGENTS.md`'s "Module documentation convention" section, since that file is where a reader
   would otherwise expect to find such a rule — flagged in §6 as a follow-up for the stakeholder/
   `teco` to decide on, not assumed here (root `AGENTS.md` is root law; this plan does not edit it).

## 5. Test strategy

Infra/process design, not application code — verification is live operational checks against the
acceptance criteria, executable by `qa-engineer` or `graph-dba` once steps 1–6 land. No unit-test
surface exists (no `cypher-mcp/server.py` change, per §2).

| AC (from requirements doc) | Verification |
|---|---|
| AC-1: destructive op in sandbox → prod unaffected | Provision a sandbox (step 1). Run a deliberate `GRAPH.DELETE`/`DETACH DELETE` directly against it via `redis-cli`. Confirm via `mcp__cypher__query` against **production** `kaizen_team`: node/relationship counts and `GRAPH.LIST` unchanged before/after. |
| AC-2: schema change in sandbox → prod schema unaffected until promoted | Add a throwaway index/constraint to the sandbox instance directly (e.g. `CREATE INDEX FOR (e:KaizenEntry) ON (e.throwaway)` + `GRAPH.CONSTRAINT CREATE kaizen_team UNIQUE NODE KaizenEntry PROPERTIES 1 throwaway`). Confirm production's `CALL db.indexes()`/`CALL db.constraints()` for `kaizen_team` is byte-identical before/after. Then deliberately replay the same DDL against production (the promotion path, §3.4) and confirm it now appears there — then immediately drop it from production, constraint before index (reverse of create order): `GRAPH.CONSTRAINT DROP kaizen_team UNIQUE NODE KaizenEntry PROPERTIES 1 throwaway` then `DROP INDEX ON :KaizenEntry(throwaway)` (verified syntax, docs.falkordb.com/commands/graph.constraint-drop.html and /cypher/indexing.html). This teardown is the check's own last step, not optional cleanup — the check is not complete until production's schema is back to its pre-check state. |
| AC-3: heavy load/crash in sandbox → prod reads/writes unaffected | While a sandbox instance is running, drive a heavy query loop or `docker kill` it. Concurrently issue a normal `mcp__cypher__query` producer-write against production `kaizen_team` and confirm it succeeds with no latency/error attributable to the sandbox event. Residual risk noted in §6 (host-level resource contention is not fully eliminated by container isolation alone; step 1's default `--cpus=1 --memory=1g` narrows but does not close it). |
| AC-4: normal kaizen-writing lands in prod, unaffected by sandbox's existence | With a sandbox `.mcp.json` entry present, run an ordinary `mcp__cypher__query` producer-write (no special config) and confirm it lands in production `kaizen_team` exactly as before this plan — a plain before/after file diff of `.mcp.json`'s `cypher` entry (independent of git tracking, since sandbox entries are themselves never committed — §3.2) confirms it is byte-unmodified. |
| AC-5: risk/blast-radius call recorded with stakeholder | For documented work: pick a future `kaizen_team`-touching plan (the integration project's own, once written) and confirm its "Sandbox & promotion" subsection (step 4 template) has a filled-in risk call and a dated stakeholder-negotiation line — not blank, not assumed. For undocumented work: pick a small direct-touch change made under §3.3's undocumented-work path and confirm the one-line kaizen entry recording the risk call exists in `kaizen_team` (`fact` starting "kaizen_team schema/data touched directly, no sandbox —"). |
| AC-6: promotion carries a migration-impact analysis + sign-off for structural changes | Once a sandbox validates a schema-shape change, confirm the same subsection's migration-impact-analysis and stakeholder-sign-off fields are filled in and dated **before** the DDL is replayed against production (step ordering, not just presence). |

Edge cases worth exercising once the scripts exist: provisioning with a `--slug` that collides with
an already-running sandbox started with *different* `--port`/`--image`/`--cpus`/`--memory` (should
fail loudly, not silently reuse/overwrite — §4 step 1's "new provisioning" path); a bare
`--slug`-only re-run against an already-running sandbox of that name, simulating a dropped
`.mcp.json` entry (should succeed via §4 step 1's "recovery path," restoring the same port/entry
without touching the container); provisioning a `--slug` whose `.mcp.json` entry already exists but
points at a stale/dead port — e.g. left behind by a teardown interrupted before its `.mcp.json`
delete ran (should succeed via add-or-correct, overwriting the stale entry rather than leaving it or
erroring on "already present" — §4 steps 1–2); teardown of a sandbox whose `.mcp.json` entry was
never added (script should still succeed — the entry is optional bookkeeping, not a dependency of
the container); running either script against a `.mcp.json` missing the `"mcpServers": {` anchor
line, e.g. hand-corrupted (should fail loudly rather than guess an insertion point — §4 step 1); a
sandbox left running with no promotion or teardown for an extended period (no automatic expiry in
this design — flagged in §6). No `jq`/`python3` edge case remains — §4 steps 1–3's text-anchored
splice has no such dependency to test.

## 6. Risks & open questions

- **Host-level resource contention is not fully eliminated by container-per-instance isolation.**
  Two independent `docker run` containers still share host CPU/RAM/disk I/O; a sandbox load test
  saturating the host could still cause *some* production slowdown, even though the two FalkorDB
  *processes* never share memory or crash together. §4 step 1 now defaults sandbox containers to
  `--cpus=1 --memory=1g` (overridable) to narrow this — a conservative cap, not a full close, of the
  residual risk; a deliberately headroom-heavy load test can still saturate shared host I/O even
  under a per-container CPU/RAM cap.
- **Image-version drift between `falkordb-dev` and a sandbox is now closed structurally, not just
  documented.** §4 step 1's provisioning script parses `start_falkordb.sh`'s own `FALKORDB_IMAGE:-`
  default live, at each run, rather than hardcoding a copy — so the two cannot silently diverge.
  Residual risk is narrower: if that parse ever breaks (e.g. `start_falkordb.sh`'s variable syntax
  changes shape), the script should fail loudly rather than silently fall back to a stale default;
  `cypher-mcp/README.md`'s new section (step 4) should still note this as a standing thing to keep an
  eye on, not a one-time fix.
- **A sandbox's uncommitted `.mcp.json` entry (§3.2, §4 steps 1–3) can be dropped by a git operation
  mid-session** (`checkout`, `stash`, `reset` touching the working tree). Low severity by design: the
  entry is disposable, host-local config pointing at a still-running, unaffected container/volume —
  re-running step 1's script regenerates the *same* entry without touching the container or its data,
  because it now checks for an already-running `falkordb-sandbox-<slug>` container and reuses its
  bound port before falling back to auto-selecting a new one (§4 step 1) — without that check, a bare
  re-run's "first free port" default would pick a *different* port than the original (still bound by
  the still-running container), producing a mismatched entry instead of the same one. No
  teardown/data-loss risk, just a re-provisioning step the requester needs to know about.
- **No automatic sandbox expiry.** A forgotten, never-torn-down sandbox is not itself a production
  risk (it's a fully isolated instance) but is host-resource waste and a stale `.mcp.json` entry.
  Out of scope for this plan to solve (no continuous/automated lifecycle was requested) — worth a
  future backlog item if it becomes a real nuisance, not something to build speculatively now.
- **A teardown interrupted between destroying the container and removing its `.mcp.json` entry
  leaves a stale entry — self-healing, not a standing risk.** §4 step 2 destroys the
  container/volume before the `.mcp.json` delete, so a hard failure (or a contributor tearing the
  container down manually, outside the script) in that gap leaves a dead-port entry behind. This is
  not a residual risk requiring its own mitigation: §4 step 1's provisioning is **add-or-correct**
  (not add-if-missing), so the next time that slug is provisioned — including via the recovery path
  — the stale entry is overwritten with the current, correct one as a side effect of normal use, no
  separate detection/cleanup step needed. A stale entry that's never re-provisioned just fails safely
  in the meantime (`.mcp.json` unreachable-port error, same as any other never-torn-down sandbox
  above).
- **The `.mcp.json`-editing mechanism is a text-anchored splice, not a JSON parse/re-serialize round
  trip — deliberately, after the JSON-library approach was tried and found to reformat the untouched
  `cypher` entry.** A `json.load`→mutate→`json.dump` round trip (the originally specified `jq`/
  `python3` mechanism) pretty-prints the *entire* file on every write, expanding the pre-existing
  `cypher` entry's `args` array across multiple lines even when its value didn't change — breaking
  AC-4's byte-unmodified check and the "sandbox entries are the only diff" claim (§3.2). The
  text-anchored splice (insert after `"mcpServers": {`; delete/replace by brace-balanced match on
  `"cypher-sandbox-<slug>": {` — §4 step 1) touches only the block it owns, closing this structurally
  rather than by picking a different diffing method. Side benefit: no `jq`/`python3` dependency to
  declare or fall back on at all, which was itself a new prerequisite the box this plan's own live
  checks ran on didn't have.
- **The cross-instance data-copy method (§4 step 5) is genuinely unverified** — this plan
  deliberately does not assert whether FalkorDB supports `MIGRATE`/`DUMP`+`RESTORE` for a single
  graph key between two engine processes; `graph-dba` needs to check current docs before a
  "copy-from-prod" seed request relies on it. The Cypher-export/replay fallback is guaranteed to work
  (it's the tool's ordinary write path) but is $O(\text{entries})$ round-trips, not a bulk copy — fine
  at today's `kaizen_team` scale (a few dozen entries), worth revisiting if the graph grows much
  larger.
- **Open governance question, not decided here:** whether the "Sandbox & promotion" subsection
  convention (§4 step 4/7) should be promoted from "this plan says so" into root `AGENTS.md`'s
  documentation-convention section, making it repo law rather than a convention future architects
  need to know to look up. Flagged for the stakeholder/`teco` — this plan does not edit root
  `AGENTS.md` itself.
- **FR-5's "case by case, no checklist" is honored as-is** — this plan deliberately does not
  introduce risk scoring/thresholds. If a future need for a lighter-weight mechanized version
  emerges, that is a fresh requirements conversation (a checklist is explicitly out of scope of the
  requirements doc this plan implements), not something to retrofit here.

## Traceability

| Requirement | Where addressed |
|---|---|
| FR-1 (data isolation) | §3.1 — separate engine instance/volume |
| FR-2 (schema isolation) | §3.1, §2's DDL block — separate engine, DDL never crosses instances except at deliberate promotion (§3.4) |
| FR-3 (engine isolation) | §3.1's rationale for rejecting a same-instance graph-key split; residual risk noted §6 |
| FR-4 (unaffected normal operation) | §3.1/§4 step 3 — `cypher` `.mcp.json` entry never modified, only additive entries |
| FR-5 (risk-based, no blanket rule) | §3.3, §4 step 4 template's "Risk/blast-radius call" field; §6 explicitly declines to mechanize it |
| FR-6 (stakeholder negotiation) | §4 step 4 template's negotiation/sign-off fields (a) and (b) |
| FR-7 (migration-impact analysis) | §3.4, §4 step 4 template's migration-impact field |
| FR-8 (standing capability, open to anyone) | §3.2, §4 steps 1–3 (reusable scripts) + step 6 (discoverability via `claude/AGENTS.md`) |
| FR-9 (responsibility split) | §3.3 (requester writes scope/isolation/risk into their own plan doc) + §4 steps 1–3 (devops owns scripts/containers/`.mcp.json`) + step 5 (graph-dba owns schema/data-copy execution) |
| AC-1..AC-6 | §5 table |
