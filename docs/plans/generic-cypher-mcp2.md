# Generic Cypher MCP — team-wide kaizen inbox rollout — implementation plan

> **Status:** active · **Owner:** `architect` · **Tracks:** — (M7)

Design for [`../requirements/generic-cypher-mcp2.md`](../requirements/generic-cypher-mcp2.md)
(FR-1…FR-14, AC-1…AC-13), per
[`generic-cypher-mcp2-coordination.md`](./generic-cypher-mcp2-coordination.md) unit U1. Builds on,
and does not redesign, the M5 mechanism —
[`generic-cypher-mcp.md`](./generic-cypher-mcp.md) (tool mechanism: the optional `agent`
parameter, the two enforced write shapes) and [`generic-cypher-mcp-graph.md`](./generic-cypher-mcp-graph.md)
(the `:KaizenEntry` schema, `author` as a plain property, curator-clear semantics) — both taken as
given and cited by path, not re-derived. Written directly against the post-M6 tool identity
(`cypher-mcp/`, `mcp__cypher__query`) — no dual-naming transition needed.

**CPG:** considered, not relevant — this delivery touches `cypher-mcp/server.py` (two frozen
doc-string constants, no logic), eleven agents' `kaizen/inbox.md` files, eleven agents' own
operative-prompt Markdown, and three repo-wide convention docs (`claude/AGENTS.md`,
`claude/README.md`, `docs/BACKLOG.md`) plus `skills/agent-maintenance/SKILL.md`. None of this is
application source code a Joern CPG would model, and a live check confirmed no CPG covers it: `MATCH
(n) RETURN count(n)` against a graph named `cypher-mcp` returned "Graph 'cypher-mcp' does not
exist. Loaded graphs: ws:test, cpg_falkorchat, reference, ws:qa-tico-workflows-manual, ws:acme,
cpg_salesperson, ws:eval, kaizen_graph_dba" — the only two loaded CPGs (`cpg_falkorchat`,
`cpg_salesperson`) are unrelated application codebases. Investigation was direct file reads plus
live FalkorDB queries against the actual `kaizen_graph_dba` graph and the `cypher-mcp/server.py`
source — a call-graph tool would add nothing here, mirroring M5's own finding on the same
component.

---

## 1. Goal & scope

Roll the graph-backed kaizen working-memory pattern M5 proved on `graph-dba` alone out to the
other eleven agents — `analyst`, `architect`, `cobb`, `coder`, `data-scientist`, `devops`,
`frontend-engineer`, `qa-engineer`, `tdd-engineer`, `teco`, `tico` — on **exactly the mechanism M5
built** (FR-1: no new write mechanism), plus a new team-wide read surface (FR-7) and a new
session-ID field on future entries (FR-8a). In scope:

- A one-time import of each of the eleven agents' current `kaizen/inbox.md` into a graph, followed
  by deletion of that file (FR-2/FR-3/FR-4, reversing M5's "freeze, don't delete" choice — see the
  requirements doc's decision log).
- `graph-dba`'s own already-frozen `kaizen/inbox.md`, deleted too (FR-14/AC-11).
- The team-wide query surface (FR-7) — resolved below as a data-organization decision, not a new
  tool mechanism.
- The FR-8a session-ID mechanism — resolved below as a concretely verified environment variable.
- Every doc describing the kaizen-inbox convention, brought back to describing reality (FR-11), and
  the new-agent-creation convention, updated so a new agent is born graph-backed (FR-12).
- A sequencing/batching model sized for FR-13's incremental-delivery requirement.

Out of scope: unchanged from the requirements doc's own Out of scope section — falkor-chat
integration, documents-as-graph-data, `BACKLOG.md`-as-graph, the stakeholder's own direct
read/write access, guaranteed semantic search, hardened/cryptographic access control, git-history
rewriting, redesigning the write mechanism itself, and the MCP server/tool rename (already
delivered separately as M6, closed).

---

## 2. Context & findings

- **The write-authorization mechanism in `cypher-mcp/server.py` (read in full, 791 lines) is
  already graph-name-agnostic — confirmed by direct code reading, not inferred.** `authorize_write()`,
  `_author_claims()`, `_kaizen_entry_create_map_spans()`, and `_CURATOR_CLEAR_RE` (lines 225–380)
  operate purely on the **Cypher text** (the `:KaizenEntry` label, the `author:` literal inside a
  `CREATE` map body, the exact `MATCH (...) DETACH DELETE` skeleton) and the declared `agent`
  parameter — none of them inspect or branch on `graph`. `grep -n "kaizen_graph_dba"
  cypher-mcp/server.py` finds it in only two places: the two frozen doc-string constants
  (`TOOL_DESCRIPTION`, `SERVER_INSTRUCTIONS`, lines 116–144) and inline comments — never in
  executable authorization logic. **This means every one of the eleven agents' migrations, and the
  new team-wide query surface, can be built on the existing mechanism with zero changes to
  `authorize_write()` or its helpers** — the only code touch needed anywhere in `cypher-mcp/` is
  updating those two doc-strings' example graph name (§4.2 below), which is documentation text, not
  a new write mechanism (FR-1 satisfied to the letter).
- **`kaizen_graph_dba` currently holds zero `:KaizenEntry` nodes.** Live query,
  `MATCH (e:KaizenEntry) RETURN e.entryId, e.date, e.author, e.createdAt ORDER BY e.date` against
  `kaizen_graph_dba` → `rows=0`. `cobb`'s ongoing distillation has already promoted/discarded every
  entry M5's import created. This matters directly for §3.2's design: there is no live data to
  migrate out of `kaizen_graph_dba` — consolidating it into a shared graph is a pure
  schema/provisioning move, not a data-copy operation.
- **`CLAUDE_CODE_SESSION_ID` is a real, live-observed environment variable, available in a
  subagent's own process environment.** Verified in this very investigation: `env | grep
  CLAUDE_CODE_SESSION_ID` in this session returned `CLAUDE_CODE_SESSION_ID=7315f2d5...`, which
  matches — character for character — the UUID segment of this session's own scratchpad path
  (`/tmp/claude-1000/.../7315f2d5-1ceb-48b3-836e-dbf764c16fe0/scratchpad`), confirming it is the
  actual Claude Code session identifier, not some unrelated value. `CLAUDE_CODE_CHILD_SESSION=1`
  was also observed (a boolean flag, presumably marking this process as a spawned subagent rather
  than the main session). **This is not a documented public API** — I did not find it in Claude
  Code's published docs, only by direct `env` inspection of a live session — so it is reported here
  as a live-verified fact, not an asserted contract, exactly as this agent team's own convention
  requires (see graph-dba's "never present a fabricated function... as fact"). It directly answers
  the requirements doc's open question 4 (§3.3 below).
- **Current `kaizen/inbox.md` sizes** (live `wc -l`, re-measured during this investigation, closely
  matching but not identical to the requirements doc's own count — small drift is expected and
  immaterial): `analyst` 59, `architect` 19, `cobb` 19, `coder` 21, `data-scientist` 118, `devops`
  18, `frontend-engineer` 18, `qa-engineer` 48, `tdd-engineer` 40, `teco` 41, `tico` 47 lines.
- **`claude/AGENTS.md` and `claude/README.md`** (both read in full) already carry the exact
  phrasing this delivery must generalize: both describe the file-based inbox as the rule "**except**
  `graph-dba`," with graph-dba's mechanism spelled out as the one exception. Post-M7 that phrasing
  inverts — the graph-backed pattern becomes the rule, with any not-yet-migrated agent (during the
  incremental window, FR-13) as the temporary exception.
- **`skills/agent-maintenance/SKILL.md` §1 "Creating" procedure** (line 60) is the concrete governing
  text for FR-12/AC-9: *"In collections that run the learning-capture loop (§5 — graphmind-ai-lab's
  `claude/` does), also seed an empty `inbox.md` from the §5 template."* This is the one sentence
  that must change so a newly created agent is never given a markdown inbox at all.
- **Neither `cobb` nor any of the ten other newly-migrating agents (besides `architect`, `analyst`,
  `data-scientist`, `teco`, `tico`) carries a restrictive `tools:` allowlist** — mirroring M5's
  finding for `graph-dba`/`cobb`. The five doc-scoped-write-guard agents (`architect`, `analyst`,
  `data-scientist`, `teco`, `tico`) already have `mcp__cypher__query` in their explicit allowlist
  (confirmed live in `claude/README.md`'s catalog entries, e.g. architect's: *"hence the
  `mcp__cypher__query` entry in this agent's `tools:` allowlist"*). **No agent-wiring/allowlist step
  is needed anywhere in this rollout.**
- **A genuine self-edit wrinkle for `cobb`.** `cobb` is both one of the eleven migrating agents
  *and* the standing owner of every other migrating agent's operative-prompt edit (FR-11's
  Learning-capture section update). Every agent prompt in this repo that states the convention
  explicitly forbids self-editing one's own agent-definition source (`graph-dba.md`: *"never edit
  your own agent definition"*) — except `cobb`'s own prompt, which already carves out one narrow
  self-maintenance exception: *"you are the maintainer, so same-run promotion with full §1/§2
  bookkeeping is in-bounds for you alone."* Whether that existing carve-out extends to `cobb`
  editing its **own** `cobb.md` Learning-capture section (as opposed to just its own kaizen
  inbox/history bookkeeping) is not, on a plain reading, obviously the same thing — flagged as an
  open item, §6.
- **No relevant prior art for FR-7 exists in this repo** — confirmed by the requirements doc's own
  decision log ("`tico` explained the tool mechanics... and the resulting trade-off... left to the
  architect") and by inspection: `mcp__cypher__query(graph, cypher)` takes exactly one graph key per
  call, with no fan-out parameter, and FalkorDB itself has no cross-graph `JOIN`/`UNION` (graph-dba's
  own fundamentals: *"one instance holds many independent named graphs (each a Redis key)"* —
  confirmed in `claude/graph-dba/graph-dba.md`).

---

## 3. Design & rationale — the four points left to the architect

### 3.1 / 3.2 — FR-7's team-wide query surface, and whether to follow `kaizen_<agent>` naming

**Decision: one shared graph, `kaizen_team`, with `author` (already part of the locked FR-8 schema)
doubling as the per-agent partition key — not per-agent graphs plus a federated fan-out helper.**
This single decision resolves both open questions 1 and 2 together, because they are the same
trade-off seen from two angles.

**Why not per-agent graphs (`kaizen_<agent>`, mirroring `graph-dba`'s own precedent).** FalkorDB
has no native cross-graph query (§2's finding) — so if the team-wide surface (FR-7) had to reach N
separate graph keys, "one query" (AC-7's literal wording: *"in one query — not eleven-plus separate
lookups"*) could only be delivered by a **new** mechanism: either a second MCP tool that loops
`GRAPH.QUERY` across every `kaizen_*` graph and merges results, or a `graph='kaizen_*'` fan-out mode
bolted onto the existing tool. Either shape is genuinely new server logic — graph discovery
(`list_graphs()` + prefix filter, or a hardcoded 12-name list that drifts every time an agent is
added/removed), N separate round-trips, result-shape merging, and partial-failure handling for an
agent that hasn't migrated yet. That is real, untested surface area for a problem the alternative
below doesn't have.

**Why one shared graph.** `author` is already on every `:KaizenEntry` node (FR-8's locked schema,
unchanged) — it was designed into the schema specifically as a plain, cheaply-filterable property
(`generic-cypher-mcp-graph.md` §2: *"a plain equality predicate, no traversal or extra lookup
needed"*). Reusing it as the partition key means:
- **FR-7 costs zero new server code.** The team-wide query is exactly `MATCH (e:KaizenEntry) RETURN
  e.author, e.date, e.fact, e.evidence, e.context, e.suggestedHome ORDER BY e.date` — one ordinary
  `mcp__cypher__query` call, today's tool, unchanged. A caller who wants one agent's slice adds
  `{author: '<agent>'}` to the pattern — one extra clause, not a different tool.
- **`authorize_write()` is untouched** (§2's finding: it never inspects `graph`) — the exact same
  author-write / curator-clear enforcement applies verbatim to a shared graph. `cobb`'s
  curator-clear (FR-9) becomes *simpler*, not harder: today it must know which per-agent graph an
  `entryId` lives in before clearing it; with one shared graph, `MATCH (x:KaizenEntry
  {entryId:'<id>'}) DETACH DELETE x` addresses any agent's entry the same way, regardless of who
  authored it.
- **Scale is a non-issue** — the decision log already closed this ("No — same accepted trade-off as
  M5, not a new sizing concern"), and §2's line-count survey (max 118 lines / one agent) puts the
  eventual total working set at, per M5's own §6 footprint math, tens of KB — several orders of
  magnitude below this instance's real consumers (`cpg_falkorchat` alone: ~167K nodes).
- This is *not* the "tenant-property mega-graph" shape `graph-dba`'s own modeling principle warns
  against (*"prefer one graph per tenant over a tenant-property mega-graph"*) — that principle
  guards against genuinely large, growing-without-bound tenant datasets sharing one graph; twelve
  agents' *raw, bounded-by-clear-on-promote* working memory (M5 §6: *"this graph doesn't
  accumulate"*) is a fundamentally smaller, self-pruning shape.

**Consequence for `graph-dba`'s existing `kaizen_graph_dba` graph.** For FR-7's "one query reaches
every migrated agent's working memory — twelve, counting graph-dba" (the decision log's own
framing) to be literally true, `graph-dba`'s entries must also live in `kaizen_team`. Since
`kaizen_graph_dba` is currently **empty** (§2), this is not a data migration — it is: (a)
`kaizen_team` gets `graph-dba`'s existing `entryId` index + uniqueness constraint provisioned
(exact DDL M5 already used, just re-targeted), and (b) the now-empty `kaizen_graph_dba` graph key is
retired (`GRAPH.DELETE`, `graph-dba`'s own destructive-ops hook gates it — a normal, human-approved
op, no different in kind from the file-deletions FR-4/FR-14 already require). This is **not**
literally demanded by any FR/AC in the requirements doc (FR-14 only names the `inbox.md` *file*) —
it is this plan's own hygiene-consistency call, flagged as open item 2 (§6).

**Naming.** `kaizen_team` (not `kaizen_graph_dba` reused, and not `kaizen_all`/`kaizen`) — a clean
name distinct from the retired per-agent key, avoiding the exact naming trap M5's own plan flagged
and predicted: *"Revisit if this pattern extends past `graph-dba` to a second agent's working
memory... Not before."* That trigger has now fired; this is the revisit.

**Trade-off named plainly.** This does diverge from `graph-dba`'s own recorded default recommendation
(`generic-cypher-mcp-graph.md` §0: *"one graph per agent... architect can override it if the
generic MCP tool's graph-discovery UX wants something else"*) — which explicitly anticipated and
pre-authorized exactly this kind of override. The override is made here because FR-7 (new in M7,
absent from M5's one-agent pilot) is the deciding factor `graph-dba`'s note didn't have in front of
it.

### 3.3 — FR-8a's session-ID mechanism

**Decision: read the `CLAUDE_CODE_SESSION_ID` environment variable at write time** (e.g. a
one-line `Bash` call, or any other environment-inspection surface the agent's harness exposes) and
include its value as the new `sessionId` property on a newly created `:KaizenEntry` node — omitted
entirely if unavailable, at the same self-reported trust level as `author` (FR-8a's own stated
bar). §2 live-verified this variable exists and holds the actual session identifier in a subagent's
process environment; it is not documented in Claude Code's public docs, so this is reported as a
live-verified, not an asserted, fact — flagged for `qa-engineer`'s acceptance pass to re-confirm
(and, if it ever changes across a harness upgrade, for whoever's kaizen entry catches the drift to
route to `graph-dba`/`cobb`'s knowledge bases). No other candidate mechanism (a tool-level parameter,
a config file) was found or is needed — the value is already sitting in every session's own
environment.

**Every agent's post-migration Learning-capture recipe** (the entry-creation half; mirrors
`graph-dba`'s own block in `graph-dba.md`, with the `sessionId` line added and the graph name
updated):

```cypher
CREATE (k:KaizenEntry {
  entryId: '<uuid4>', date: '<YYYY-MM-DD>', fact: '<the fact, one line>',
  evidence: '<what was run/read/observed>', context: '<the task where it surfaced, one line>',
  suggestedHome: 'prompt | knowledge base | project docs | unsure',
  author: '<agent-slug>', createdAt: '<ISO-8601 write time>',
  sessionId: '<value of $CLAUDE_CODE_SESSION_ID, or omit this key entirely if unavailable>'
})
```
called as `mcp__cypher__query(graph='kaizen_team', cypher=<that text>, agent='<agent-slug>')`.

### 3.4 — Sequencing/batching for FR-13's incremental-delivery requirement

**Decision: fifteen independently-dispatchable units — no unit blocks on more than one
predecessor, and no unit's own value depends on any other unit landing.** Full table in §4. Design
choices behind the shape:

- **Each of the eleven agents' migrations is its own unit, each with two co-dispatched halves that
  land together** (not two separate units): (a) the agent's own data migration — it reads its own
  `kaizen/inbox.md`, writes into `kaizen_team` attributed to itself, verifies the count, deletes its
  own `kaizen/inbox.md` — mirroring exactly how M5 had `graph-dba` dogfood the write path for its
  own migration (`generic-cypher-mcp.md` §3.4); and (b) `cobb`'s edit to that same agent's own
  operative-prompt Learning-capture section. These two halves are bundled into one unit, not split
  across two independently-landable ones, because splitting them would create a real (if transient)
  state where an agent's file is gone, its data is in the graph, and its own prompt still tells it
  to append to a file that no longer exists — a genuine AC-8 contradiction, even if brief. Bundling
  keeps every unit's "done" state internally consistent, which is what AC-10 actually asks for
  ("independently satisfies AC-1…AC-6 for themselves").
- **A repo-wide docs pass (FR-11's catalog half + FR-12) is its own unit, done once, early, not
  re-touched per agent.** `claude/AGENTS.md` and `claude/README.md` are rewritten from "file-based,
  except `graph-dba`" to a shape that stays true for the entire incremental window: "graph-backed is
  the standing convention (`kaizen_team`); any agent not yet migrated still appends to its own
  `kaizen/inbox.md` — see `docs/BACKLOG.md`'s M7 section for the current roster." This is the same
  technique M5's own catalog entries already used for the one-agent case; generalizing the sentence
  once means it never needs a second edit as agents land one by one. `docs/BACKLOG.md`'s M7 section
  becomes the living, per-agent-granular status table (mirroring M5's own C-50x item list with ✅/🔵
  markers) — that document, not the two catalog files, is where FR-13's "which agents have migrated
  right now" question is actually answered.
- **The `entryId` index/constraint + `kaizen_graph_dba` retirement (§3.2's consequence) is its own
  unit, owned by `graph-dba`, recommended first but not a hard dependency of the other eleven** — any
  agent's migration write auto-materializes `kaizen_team` via the same "empty key + agent" branch
  M5's migration already exercised (`generic-cypher-mcp.md` §3.1), so no unit is blocked waiting for
  this one; it only means the uniqueness constraint isn't engine-enforced until this unit lands
  (`uuid4` collision risk is negligible in the interim — the same trade-off M5 accepted while its
  own index/constraint step ran).
- **The `cypher-mcp/server.py` doc-string update (§4.2) is its own unit**, independent of every
  other unit — it only needs the name `kaizen_team` decided (already true, from this document), not
  any migration to have landed.
- **A final acceptance pass, `qa-engineer`**, is deliberately **not** deferred to "all eleven done" —
  the plan recommends at least one interim pass once a handful of agents have migrated
  (concretely exercising AC-10, not just asserting it), plus a closing pass once all eleven have
  landed.

### 3.5 — Why this plan does not commission a companion `-graph.md` design note

M5 split ownership between a `graph-dba`-authored schema note (`generic-cypher-mcp-graph.md`) and
this `architect`-authored mechanism note, because M5 was inventing a schema from nothing. FR-8 locks
that schema as-is for M7 (no field redesign beyond the additive `sessionId`), and §3.1–3.2 above are
the only genuinely new graph-modeling decisions this delivery makes — both are covered here, both
reuse M5's exact DDL pattern just re-targeted at a new graph name. No fresh schema design work
remains for `graph-dba` to own in a separate document; its role in this rollout is implementation
(the index/constraint provisioning + old-key retirement unit, §4) rather than design. Flagged as
open item 6 (§6) in case the plan-gate reviewer judges otherwise.

---

## 4. Implementation step table

Fifteen units. "Depends on" lists **hard** blockers only (§3.4); anything not listed is safely
parallel/independent, sized for FR-13's per-agent-or-per-batch requirement. Suggested batches are a
dispatch convenience for whoever coordinates execution (`teco`, per its own coordination-ledger
threshold at 3+ units — already opened at
`docs/plans/generic-cypher-mcp2-coordination.md`), not a hard sequencing rule.

### 4.1 Docs & substrate units (recommended early, all mutually independent)

| # | Owner | Files | Depends on | Done-condition |
|---|---|---|---|---|
| **D1** | `cobb` | `claude/AGENTS.md`, `claude/README.md`, `docs/BACKLOG.md` (new M7 section, §4.4 below) | — | Both catalog files describe the graph-backed pattern as the standing convention with a generic, don't-need-re-editing-per-agent sentence (§3.4); `docs/BACKLOG.md` M7 section added with a per-agent status table (C-701…C-71x, see §4.4) |
| **D2** | `cobb` | `skills/agent-maintenance/SKILL.md` (§1 "Creating" procedure, §5 "Learnings inboxes", frontmatter `description`) | — | §1's inbox-seeding sentence replaced with: seed a newly created agent's Learning-capture section directly with the graph-backed recipe (§3.3 above), target `kaizen_team`, no `inbox.md` ever created; §5 rewritten so the graph-backed pattern is the described default (not `graph-dba`'s exception) with the markdown "Inbox template" block kept only as a fallback for any not-yet-migrated agent, citing `docs/BACKLOG.md` M7 for current roster; frontmatter `description` line no longer singles out `graph-dba` |
| **D3** | `coder` | `cypher-mcp/server.py` (`TOOL_DESCRIPTION`, `SERVER_INSTRUCTIONS`), `cypher-mcp/README.md` | — | Both doc-strings' `kaizen_graph_dba` example replaced with `kaizen_team`, "graph-dba's kaizen working memory" generalized to "the team's kaizen working memory"; `test_server_instructions_are_present_and_bounded` re-verified green (≤2000 chars, no content pinned beyond bound); `cypher-mcp/README.md`'s three `kaizen_graph_dba` mentions (parameter table, prose, both example calls) updated to match; `cypher-mcp/build.sh` run once by hand, in-container test gate green |
| **G0** | `graph-dba` | (no tracked files — live graph DDL/ops only) | — | `CREATE INDEX FOR (e:KaizenEntry) ON (e.entryId)` + `GRAPH.CONSTRAINT CREATE kaizen_team UNIQUE LABEL KaizenEntry PROPERTIES 1 entryId` issued against `kaizen_team` (index-before-constraint ordering, `falkordb-quirks.md`) — defensively, since another agent's migration may have already materialized `kaizen_team` first (verify via whatever idempotent-check FalkorDB's build actually supports before re-issuing, per this agent's own "never assert unverified behavior" rule); re-confirm `kaizen_graph_dba` still holds zero `:KaizenEntry` nodes (this plan's live check found 0; re-check immediately before deleting in case anything landed since); `GRAPH.DELETE kaizen_graph_dba` (own destructive-ops hook approval); `claude/graph-dba/graph-dba.md`'s Learning-capture section updated in place (`kaizen_graph_dba` → `kaizen_team`, `sessionId` line added per §3.3) — **`graph-dba` may edit this itself**, since it is modifying its own already-graph-backed section, not switching mechanisms (unlike the eleven agents below) |

### 4.2 Per-agent migration units (each independent; suggested batches shown, not required)

Each unit N: agent `<X>` reads `claude/<X>/kaizen/inbox.md`, parses its entries into the standard
five-field template, generates one `entryId` (`uuid4`) per entry and one shared `createdAt`
(import-run timestamp), builds `UNWIND [<one map per entry, no per-row author>] AS e CREATE
(k:KaizenEntry {entryId: e.entryId, date: e.date, fact: e.fact, evidence: e.evidence, context:
e.context, suggestedHome: e.suggestedHome, author: '<X>', createdAt: e.createdAt})` (the `author`
literal lives once in the `CREATE` clause, exactly mirroring M5 §3.4's post-fix shape — never
per-row), and calls `mcp__cypher__query(graph='kaizen_team', cypher=<that text>, agent='<X>')`.
Verifies `MATCH (e:KaizenEntry {author:'<X>'}) RETURN count(e)` matches the parsed entry count, then
deletes `claude/<X>/kaizen/inbox.md`. Co-dispatched with `cobb`'s matching prompt edit (§3.4).

| # | Agent(s) | Suggested batch | Files | Depends on | Done-condition |
|---|---|---|---|---|---|
| **A1** | `devops` | 1 | `claude/devops/kaizen/inbox.md` (deleted), `claude/devops/devops.md` (Learning capture, by `cobb`) | — | Entry count verified; file gone from working tree, recoverable via `git log`; prompt's Learning-capture section directs new learnings to `kaizen_team` (§3.3 recipe), no remaining `inbox.md`-append instruction |
| **A2** | `frontend-engineer` | 1 | same pattern | — | same shape |
| **A3** | `architect` | 1 | same pattern | — | same shape (self-migrates its own data; `cobb` edits `architect.md`, not `architect` itself) |
| **A4** | `cobb` | 2 | `claude/cobb/kaizen/inbox.md` (deleted), `claude/cobb/cobb.md` (Learning capture) | — | Same data-migration shape; **prompt-edit ownership is the open item flagged in §2/§6** — this plan's default is `cobb` edits its own section (mirroring its existing self-maintenance carve-out), pending plan-gate confirmation |
| **A5** | `coder` | 2 | same pattern | — | same shape |
| **A6** | `teco` | 2 | same pattern | — | same shape |
| **A7** | `analyst` | 3 | same pattern | — | same shape |
| **A8** | `qa-engineer` | 3 | same pattern | — | same shape |
| **A9** | `tdd-engineer` | 3 | same pattern | — | same shape |
| **A10** | `data-scientist` | 4 | same pattern | — | same shape (largest file, 118 lines — budget more parsing/verification time) |
| **A11** | `tico` | 4 | same pattern | — | same shape |

### 4.3 Acceptance

| # | Owner | Files | Depends on | Done-condition |
|---|---|---|---|---|
| **Q1** | `qa-engineer` | — (interim check, no deliverable file required) | ≥3 of A1…A11 | AC-1, AC-4, AC-6, AC-7, AC-10 exercised live against whichever agents have migrated so far — concretely proves partial-state validity, not just asserts it |
| **Q2** | `qa-engineer` | `docs/test-plans/generic-cypher-mcp2.md`, `docs/test-reports/generic-cypher-mcp2-report.md` | D1, D2, D3, G0, A1…A11 | AC-1…AC-13 each exercised live (§5's mapping); test plan + report delivered |

### 4.4 `docs/BACKLOG.md` — M7 section proposal (for D1)

Mirror the M5/M6 section format exactly; add after M6:

```markdown
## M7 — Generic Cypher MCP, team-wide rollout

`mcp__cypher__query`'s write path (M5) is rolled out from `graph-dba` alone to all twelve agents,
sharing one graph (`kaizen_team`) with `author` as the per-agent partition — no new write
mechanism (FR-1), zero `cypher-mcp/server.py` logic changes. Requirements:
[`requirements/generic-cypher-mcp2.md`](./requirements/generic-cypher-mcp2.md) (FR-1…FR-14 /
AC-1…AC-13) · plan: [`plans/generic-cypher-mcp2.md`](./plans/generic-cypher-mcp2.md) · coordination:
[`plans/generic-cypher-mcp2-coordination.md`](./plans/generic-cypher-mcp2-coordination.md).

### Items
- **C-701 — Repo-wide catalog docs.** 🔵 `claude/AGENTS.md`, `claude/README.md` generalized off the
  "except graph-dba" phrasing. Owner: `cobb`.
- **C-702 — Agent-creation convention.** 🔵 `skills/agent-maintenance/SKILL.md` §1/§5 updated so a
  new agent is born graph-backed. Owner: `cobb`.
- **C-703 — Server doc-strings.** 🔵 `cypher-mcp/server.py`/`README.md` `kaizen_graph_dba` →
  `kaizen_team`. Owner: `coder`.
- **C-704 — Shared-graph provisioning + old-key retirement.** 🔵 `kaizen_team` index/constraint;
  `kaizen_graph_dba` deleted (empty, confirmed live). Owner: `graph-dba`.
- **C-705…C-715 — Per-agent migration** (one item per agent: devops, frontend-engineer, architect,
  cobb, coder, teco, analyst, qa-engineer, tdd-engineer, data-scientist, tico). 🔵 each.
- **C-716 — Acceptance pass.** 🔵 AC-1…AC-13 exercised live.
```

Status markers flip to ✅ as each unit lands — this table **is** the live, per-agent-granular
answer to "how much of M7 is done" that D1/D2's generalized catalog prose deliberately stops trying
to restate.

---

## 5. AC-1…AC-13 verification mapping

| AC | Verification approach | Altitude |
|---|---|---|
| AC-1 | Live query against `kaizen_team` filtered to a migrated agent's `author`, from a different agent's context, no distillation gate | Live |
| AC-2 | Per-agent: parsed `inbox.md` entry count vs. `MATCH (e:KaizenEntry {author:'<X>'}) RETURN count(e)`; spot-check 1–2 entries' field content verbatim | Live, per-unit self-check + `qa-engineer` sample |
| AC-3 | `git status`/`git diff` shows `claude/<X>/kaizen/inbox.md` absent post-migration; `git log -- claude/<X>/kaizen/inbox.md` recovers it | Static |
| AC-4 | Agent writes one new entry via the §3.3 recipe; independent second read confirms it, `inbox.md` stays absent | Live |
| AC-5 | `cobb` runs a real distillation pass (append to `history.md`, confirm, curator-clear) on ≥1 live raw entry from a newly migrated agent | Live, full workflow |
| AC-6 | Mismatched `author`/`agent` write attempt rejected (mirrors M5's own unit tests 3/4, now against `kaizen_team`) | Live |
| AC-7 | One `MATCH (e:KaizenEntry) RETURN e.author, e.date, ... ORDER BY e.date` (no author filter) against `kaizen_team`, one tool call, returns entries spanning ≥2 distinct authors | Live, the direct FR-7 proof |
| AC-8 | `grep -rln 'kaizen/inbox\.md\|append.*inbox' claude/ skills/agent-maintenance/SKILL.md` before/after each docs unit (mirrors M5's own close-out method); every post-migration hit is either an already-updated file or confirmed still-correct for a not-yet-migrated agent | Static, repeatable |
| AC-9 | Read `skills/agent-maintenance/SKILL.md` §1 post-D2: confirm no `inbox.md`-seeding step remains for a newly created agent | Static |
| AC-10 | Q1 (interim pass, §4.3): re-run AC-1/AC-4/AC-6 scoped only to whichever agents have migrated at that point | Live, exercised mid-rollout, not just asserted |
| AC-11 | `git status`/`ls claude/graph-dba/kaizen/` confirms `inbox.md` absent; `git log` recovers it | Static |
| AC-12 | `MATCH (e:KaizenEntry) RETURN DISTINCT keys(e)` (or per-agent sampled `RETURN keys(e)`) across ≥2 different agents' entries in `kaizen_team`, confirm identical key sets modulo `sessionId` | Live |
| AC-13 | One entry with `sessionId IS NOT NULL` (a new, post-migration write) and one with `sessionId IS NULL` (an imported entry) both present and distinguishable on that basis | Live |

Verification depth per agent (independent pass for all eleven vs. sampled) is left to
`qa-engineer`'s own test-plan judgment, per the requirements doc's own decision log — this mapping
states the checks, not the rigor level.

---

## 6. Open items for the plan-gate reviewer

1. **The central architectural call: one shared `kaizen_team` graph (author-partitioned) instead
   of per-agent `kaizen_<agent>` graphs.** §3.1/3.2 lay out the reasoning (FalkorDB has no
   cross-graph query, so "one query" for FR-7 is materially cheaper this way — zero new server
   code vs. a new fan-out mechanism). This is the single biggest judgment call in this plan; it
   diverges from `graph-dba`'s own recorded default recommendation (though that recommendation
   explicitly pre-authorized an override). Please scrutinize directly.
2. **Retiring the (currently empty) `kaizen_graph_dba` graph key** is not literally required by any
   FR/AC — FR-14 only names the `inbox.md` file. This plan adds it for hygiene/consistency with the
   file-deletion philosophy already established. Confirm this extension is wanted, not just
   tolerated.
3. **`CLAUDE_CODE_SESSION_ID` as FR-8a's mechanism** is live-verified in this one investigation
   session but not found in official Claude Code documentation. Recommend `qa-engineer`'s
   acceptance pass independently re-confirm it's available in at least one other agent's session
   before the pattern is baked into eleven prompts' Learning-capture sections.
4. **Unit bundling (data migration + prompt edit landing together per agent, §3.4)** trades a
   larger per-unit diff for avoiding a transient AC-8 contradiction window. If the reviewer judges
   FR-13's incrementality is meant to tolerate that window (docs catching up shortly after data,
   not atomically with it), the eleven per-agent units could each be split into two, doubling unit
   count but shrinking each diff.
5. **Repo-wide catalog docs (D1/D2) rewritten once, early, in agent-count-agnostic language**
   rather than re-touched after every single agent migration — confirm this satisfies AC-8's "no
   reader finds it silent or contradicting" bar for the whole incremental window, or whether
   per-agent specificity is wanted in `claude/AGENTS.md`/`claude/README.md` too (this plan puts that
   granularity in `docs/BACKLOG.md`'s M7 section instead, §4.4).
6. **No companion `-graph.md` design note from `graph-dba` for M7** (unlike M5's split ownership) —
   §3.5's reasoning is that FR-8 locks the schema and this plan's §3.1/3.2 already cover the only
   new graph-modeling decisions. Confirm `graph-dba`'s role here (implementation unit G0 only, no
   separate design deliverable) is sufficient.
7. **`cobb` editing its own `cobb.md` Learning-capture section (unit A4)** — flagged in §2 and again
   in unit A4's done-condition. This plan's default is that `cobb`'s existing self-maintenance
   carve-out extends to this case; an explicit second opinion is wanted before any implementer
   treats it as settled, since editing one's own governing agent-definition source is a materially
   different act than clearing one's own kaizen entries.
8. **`GRAPH.CONSTRAINT CREATE`'s idempotency on a graph another unit may have already provisioned**
   (unit G0's "defensively... verify via whatever idempotent-check FalkorDB's build actually
   supports" instruction) is deliberately left unverified in this plan, per this agent team's
   standing rule against asserting unverified FalkorDB command behavior as fact — `graph-dba` is the
   right agent to close this gap at implementation time, not this plan.
