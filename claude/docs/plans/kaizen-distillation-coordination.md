# Kaizen distillation — team-wide pass

> **Status:** active · **Owner:** `teco` · **Tracks:** — (—)

Routine curation pass over the shared `kaizen_team` FalkorDB graph
(`skills/agent-maintenance/SKILL.md` §5): for every agent with raw
`:KaizenEntry` nodes (current-shape `PRODUCED` edges and/or legacy `author`
string entries), `cobb` verifies each entry, routes it (prompt / knowledge
base / project docs / discard / kept-open), logs the disposition in that
agent's own `kaizen/history.md` (and `plan.md` for kept-open actionable
items, with the dedup check), tags any `MENTIONS` edges for entries that are
really about a different agent, and clears each entry from the graph via the
curator `DETACH DELETE` shape once logged.

Dispatched **agent-per-agent, in parallel batches of up to 6** (raised from
an initial sequential-only plan, at the user's direction) — each unit is
scoped to one agent's own disjoint entries, so concurrent `cobb` passes don't
contend on the same graph nodes. The one cross-cutting effect, a `MENTIONS`
tag added during one agent's pass landing after a concurrently-running
mentioned agent's pass already read the graph, is explicitly a non-issue per
SKILL.md §5: "a tagged entry then surfaces again in the mentioned agent's own
future distillation pass" — deferred discovery, not data loss.

No independent review gate: this is `cobb`'s sole-owned, already-specified
procedure (SKILL.md §5 embeds its own verification step), not a design or
implementation deliverable. The team certification pass (§4 of the same
skill) is the periodic audit of `cobb`'s distillation work, run separately
on request.

Snapshot at open (raw entry counts, current-shape `PRODUCED` + legacy
`author`, queried directly against `kaizen_team`):

| Unit | Agent | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| U1 | analyst (20 raw: 8 current + 12 legacy) | `a6d9f06a660c8b9bf` | accepted | `claude/analyst/kaizen/history.md`+`plan.md`, `claude/analyst/review-techniques.md`, `claude/graph-dba/falkordb-quirks.md`, `skills/agent-standards/claude-code.md`, `skills/python-web-quirks/SKILL.md`, `falkor-chat/docs/SERVER.md`, `claude/AGENTS.md`, graph cleared | none (see above) → — | 240.8k tok, 95 tools |
| U2 | coder (10 raw: 6 current + 4 legacy) | `ac1d3df8cfe965afa` | accepted | `claude/coder/kaizen/history.md`+`plan.md` (new K-005, live bug), `skills/python-web-quirks/SKILL.md`, 3x `MENTIONS`→graph-dba, graph cleared | none → — | 207.9k tok, 83 tools |
| U3 | architect (6 raw: 3 current + 3 legacy) | `a2bc974e99f7e1f52` | accepted | `claude/architect/architect.md` (Guardrails clause), `falkor-chat/docs/DESIGN.md` (2 review boxes), `claude/architect/kaizen/history.md`+`plan.md`, `MENTIONS`→graph-dba, graph cleared | none → — | 165.8k tok, 61 tools |
| U4 | teco (6 raw: 3 current + 3 legacy) | `a87da3f83077fa9f5` | accepted | `claude/teco/teco.md` (2 guardrail clauses), `skills/agent-standards/claude-code.md`, `claude/teco/kaizen/history.md`+`plan.md`, graph cleared | none → — | 173.7k tok, 67 tools |
| U5 | qa-engineer (4 raw: 3 current + 1 legacy) | `adc3dcf7825e15838` | accepted | `claude/qa-engineer/kaizen/history.md`+`plan.md`, `claude/qa-engineer/qa-testing-techniques.md`, `cypher-mcp/README.md`, `falkor-chat/docs/SERVER.md`, `falkor-chat/scripts/start_server.sh`, graph cleared | none → — | 153.4k tok, 72 tools |
| U6 | tico (3 raw: 3 current) | `abd5373c05f90760b` | accepted | `claude/tico/kaizen/history.md`, `claude/tico/tico.md` (guardrails clause), graph cleared | none → — | 127.5k tok, 22 tools |
| U7 | data-scientist (3 raw: 3 current) | `a566d8fdc87cb1eaf` | accepted | `claude/data-scientist/kaizen/history.md`, graph cleared | none → — | 120.5k tok, 34 tools |
| U8 | cobb (2 raw: 1 current + 1 legacy) | `a40a8f36c6017c124` | accepted | `claude/cobb/kaizen/history.md`+`plan.md`, 1x `MENTIONS`→devops, graph cleared | none → — | 114.9k tok, 48 tools |
| U9 | tdd-engineer (2 raw: 1 current + 1 legacy) | `a60cb9d39d06ee660` | accepted | `claude/tdd-engineer/kaizen/history.md`, graph cleared | none → — | 117.2k tok, 34 tools |
| U10 | graph-dba (1 raw legacy + 4 current-shape, all reached via `MENTIONS`: 1 from architect's U3 + 3 from coder's U2, both landed before this unit read the graph) | `a6e910e4200ac19b0` | accepted | `claude/graph-dba/falkordb-quirks.md` (Concurrency & atomicity section + 2 Cypher-dialect bullets), `claude/graph-dba/kaizen/history.md`+`plan.md`, graph cleared | none → — | 160.1k tok, 42 tools |

`devops`, `frontend-engineer`, `security-expert` had zero raw entries at
open — no unit dispatched for them.

**Follow-ups surfaced mid-pass (not part of the original 10-unit scope):**

- U2 (`coder`) surfaced a **live bug in shipped code**, kept open as
  `claude/coder/kaizen/plan.md` K-005: `verify_workflows.sh` false-negatives
  an intact `ws:<id>` snapshot when the paired `reference` graph is fully
  deleted (unguarded `ro_query` in `falkor-chat/server/falkorchat/services.py`
  / `repository.py`) — needs a doc fix + a code fix, both outside `cobb`'s
  write remit. **Needs routing** (new `teco`-coordinated unit, likely
  `tdd-engineer`: reproduction test first).
- U8 (`cobb`) tagged one surviving entry `MENTIONS`→`devops` (a
  `cypher-mcp/server.py` trailing-`RETURN` trap undocumented in
  `TOOL_DESCRIPTION`/`SERVER_INSTRUCTIONS`) — `devops` had zero entries at
  this pass's open, so no unit was dispatched; it awaits a future pass.
- U2 (`coder`) added 3 more `MENTIONS`→`graph-dba` edges (count(*)
  undercounting parallel edges; undirected+property-filter degrading to
  directed, plus a REFINES entry) after U10 (`graph-dba`)'s dispatch, but
  before U10 actually read the graph (U10 ran ~15 min, well past U2's
  completion) — **U10 caught all 4** (this trio plus architect's), no
  deferral needed after all; see U10's row.
