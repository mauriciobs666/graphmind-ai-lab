# Kaizen agent/learning-note ontology — Coordination

> **Status:** archived · **Owner:** `teco` · **Tracks:** — (`docs/requirements/kaizen-agent-ontology.md`, M8)

Coordinating delivery of `docs/requirements/kaizen-agent-ontology.md` (Status: Ready for design,
M8) — replacing the plain `author` string property on `:KaizenEntry` nodes (in the shared
`kaizen_team` FalkorDB graph) with real `:Agent` nodes and two locked relationships,
`(:Agent)-[:PRODUCED]->(:KaizenEntry)` and `(:KaizenEntry)-[:MENTIONS]->(:Agent)`, per FR-1..FR-8/
AC-1..AC-7. M7 (`docs/requirements/generic-cypher-mcp2.md`) is confirmed archived/landed, so M8's
FR-7/AC-6 design-start gate is satisfied.

Scope confirmed by grep before drawing units: 13 agent prompt files under `claude/<agent>/*.md`
each carry a "Learning capture" section with a `CREATE (:KaizenEntry {..., author: '<agent>'})`
template; `cypher-mcp/server.py`'s `authorize_write()` gates writes by statically scanning for a
literal `author: '<agent>'` inside a `CREATE (...:KaizenEntry {...})` clause (`_author_claims`) —
this mechanism is the load-bearing security boundary for the whole write path and must change in
lockstep with the schema, not as an afterthought.

## Ledger

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| U1 | `graph-dba` | `a6228218048d49ae3` | delivered | `docs/plans/kaizen-agent-ontology-graph.md` | — | 166k tok / 12 tools |
| U2 | `architect` | `a271ad522143223af` | delivered | `docs/plans/kaizen-agent-ontology.md` | — | 203k tok / 27 tools |
| U3 | `analyst` | `aaa49a46ba2c9dab0` | gated | `docs/reviews/kaizen-agent-ontology.md` | plan review → approve w/ suggestions | 158k tok / 20 tools |
| U2b | `architect` (resumed) | `a271ad522143223af` | delivered | `docs/plans/kaizen-agent-ontology.md` (v2) | — | 246k tok / 16 tools |
| U3b | `analyst` (resumed) | `aaa49a46ba2c9dab0` | gated | `docs/reviews/kaizen-agent-ontology.md` (Pass 2) | plan re-review → needs changes | 204k tok / 3 tools |
| U2c | `architect` (resumed) | `a271ad522143223af` | delivered | `docs/plans/kaizen-agent-ontology.md` (v3) | — | 275k tok / 21 tools |
| U3c | `analyst` (resumed) | `aaa49a46ba2c9dab0` | accepted | `docs/reviews/kaizen-agent-ontology.md` (Pass 3) | plan re-review → **approve** | 238k tok / 2 tools |
| U4 (S0) | `graph-dba` | `a1416c8bc97d38cf8` | accepted | live DDL: `Agent.agentId` index+constraint on `kaizen_team` — OPERATIONAL | — | 49k tok / 8 tools |
| U5 (S1) | `tdd-engineer` | `a10500612b74ead50` | delivered | `cypher-mcp/server.py`, `cypher-mcp/tests/test_server.py`, `cypher-mcp/README.md` (uncommitted) | analyst (diff) → — | 241k tok / 79 tools |
| U6 (S2) | `analyst` | `a333587fc1ddaad28` | accepted | `docs/reviews/kaizen-agent-ontology-impl.md` | diff review → **approve** | 151k tok / 46 tools |
| U7 (S3) | `cobb` (haiku) | `a85a6819d7a4a2bba` | accepted | all 13 files retargeted — committed `4da588a` | teco verified (per-file diff) → pass | 118k tok / 36 tools |
| U8 (S4) | `cobb` | `a911056e40fd4fdb7` | accepted | `skills/agent-maintenance/SKILL.md` §5 rewrite — committed `b7520f0` | teco verified (fit/scope) → pass | 150k tok / 15 tools |
| U9 (S5) | `cobb` | `a7dbdc093901e9521` | accepted | 16 files, committed `e0eabf0` | teco verified (all 16 files, 0 remaining occurrences) → pass | 196k tok / 33 tools |
| U10 (S6) | `qa-engineer` | `a42914d9822d27b11` | accepted | `docs/test-plans/kaizen-agent-ontology.md`, `docs/test-reports/kaizen-agent-ontology.md` | live dry-run → **PASS** | 175k tok / 42 tools |

Plan's step table: S0 (`graph-dba`, DDL) ∥ S1 (`tdd-engineer`, `authorize_write()` redesign +
tests + README) → S2 (`analyst`, diff review, gates S3/S4) → S3+S4 (`cobb`, 13 prompts + skill doc,
both after S2 approves) → S5 (`cobb`, catalog docs, after S3+S4). U4-U9 dispatched only after U3
(plan review) returns an approve/approve-with-suggestions verdict; U6 (S2) is U5's own review gate,
distinct from U3 (the plan gate). Full detail, rationale, and file-level design: read
`docs/plans/kaizen-agent-ontology.md` directly, not this table.

U1 and U2 are sequenced (U2 consumes U1's note by path). U3 gates both before any implementation
unit is drawn/dispatched. U4+ deliberately left undrawn until U2's plan exists — the plan's step
table determines how many implementation units there are and their file boundaries, per the
step-sizing rule; drawing them now would be guessing.

## Notes

- **2026-08-22 — S0+S1 verified and committed.** `cypher-mcp/server.py`,
  `cypher-mcp/tests/test_server.py`, `cypher-mcp/README.md` committed at `e01045b` after S2's
  approve verdict and teco's independent re-verification (re-ran the offline suite, spot-checked
  the kaizen-entry write, confirmed zero selftest residue in `kaizen_team`). `Agent.agentId`'s
  DDL is live and `OPERATIONAL` on `kaizen_team`. The deployed `cypher-mcp` MCP container still
  runs the **pre-M8 image** — `cypher-mcp/build.sh` + restart has not run yet; S6 (`qa-engineer`)
  depends on that rebuild happening first, per the plan's own dependency note.
- **2026-08-22 — container rebuilt.** Stakeholder confirmed proceeding despite the shared-MCP-tool
  interruption risk. `./cypher-mcp/build.sh` run by teco: new image `cypher-mcp:cb712173ab57`
  (alias `:dev`, content-hash tag per `image-tag.sh`), in-image test gate green (97 passed / 10
  deselected — the 9-test gap is the documented `test_build_inputs.py` host-only exclusion, not a
  regression). Since the image is resolved by content hash per MCP connection, no already-running
  session was disrupted; S6 picks up the new image automatically on its own fresh connection.
- **2026-08-22 — Milestone closed.** S6 **PASS** — all 3 adversarial attacks the plan review
  traced (A, B, C) correctly rejected on the deployed, rebuilt image; full distillation dry-run
  completed cleanly; graph left at its exact pre-run baseline. Two non-regression findings
  recorded as follow-ups, not blockers: a long-running session's MCP connection can stay silently
  bound to a pre-rebuild image (`docs/BACKLOG.md` C-809/C-810 — two such stale containers found
  still live, left running since each belongs to a different session not confirmed abandoned), and
  the plan review's literal attack-text needs a `WITH`-bridge to be valid live Cypher (C-811). All
  ten units (U1–U10) delivered, verified, and committed:
  `e01045b` (S1), `b7520f0` (S4), `4da588a` (S3), `e0eabf0` (S5) — S0/S6 are live-DB/QA actions
  with no tracked-file diff. `docs/BACKLOG.md` (M8 row + C-801…C-812) and `docs/HISTORY.md` (closing
  entry) updated by `teco`. Every M8 document's `Status:` flipped to `archived` in this same close
  (requirements, graph design, plan, both reviews, test plan, test report, this coordination doc).

- **2026-08-22 — Stakeholder decision on U3 Finding 1** (a self-attributed decoy `CREATE` chained
  with an unrelated second clause smuggles a curator-only or attribution-forging write past
  `authorize_write()`): **close it**, not accept-and-pin. `architect` is revising the plan (§3.1)
  to generalize the new shape's "nothing else follows" anchoring onto the existing author-write
  check too. A `qa-engineer` dry-run step (new S6) is being added to the plan per U3's
  recommendation, accepted. U3's other two findings (an unstated MENTIONS-before-count ordering
  invariant; a table-vs-prose wording mismatch on S2→S3/S4's dependency strength) are folded into
  the same revision. Plan will be re-reviewed by `analyst` (U3b) before any implementation unit
  (U4+) is dispatched.

- Cross-cutting feature: no single component owns it. Design docs live at repo-root `docs/`,
  matching the `generic-cypher-mcp*` and `graph-ontology` families already there.
- FR-2/FR-3 relationship names and directions are **locked** (stakeholder decision, see
  requirements doc Decision log) — not open for the architect or graph-dba to relitigate.
- The `author` property drop (FR-2) and the `authorize_write()` static-text-scan mechanism are in
  tension: the current authorization shape keys off a literal `author: '<agent>'` inside the
  `CREATE` clause. U1 must produce a concrete, `authorize_write()`-compatible write shape (or an
  explicit recommendation for how that function's authorization logic itself must change) — not
  just an abstract schema — since this is the write path's actual security gate.
