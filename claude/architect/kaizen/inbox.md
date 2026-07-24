# Kaizen — Learnings Inbox: architect

> Append-only capture of durable, non-obvious environment facts the `architect` agent
> discovers during runs — raw observations, not conclusions. The maintainer (cobb)
> periodically distills this inbox (agent-maintenance skill §5): verifies each entry,
> routes it (prompt / knowledge base / project docs / discard), logs the promotion in
> `history.md`, and clears it. The agent only appends here; it never promotes.
>
> Entry format (append at the end):
>
> ```markdown
> ## YYYY-MM-DD — <the fact, one line>
> - **Evidence:** what was run/read/observed (command, file:line, output)
> - **Context:** the task where it surfaced, one line
> - **Suggested home:** prompt | knowledge base | project docs | unsure
> ```

## 2026-07-19 — falkor-chat's "byte-identity lock" on `executor._drive_loop` reproduces only by SHA, and only via a line-number-independent extraction

- **Evidence:** `falkor-chat/docs/plans/m3-executor-coordination.md` quotes the lock as SHA
  `71055f756280` with three different byte counts (2839, 2844, 2860); only 2860 is correct. The SHA
  reproduces from `sed -n '333,392p' server/falkorchat/executor.py | sha256sum | cut -c1-12` — i.e.
  it is pinned to *line numbers*, which shift whenever anything above the method changes. Verified
  equivalent that survives edits elsewhere in the file:
  `awk '/^    def _drive_loop/{f=1} /^    # ── seams/{f=0} f' server/falkorchat/executor.py | sed -e :a -e '/^\n*$/{$d;N;};/\n$/ba' | sha256sum | cut -c1-12`.
- **Context:** designing K-024's `kind:'process'` proof flow, whose hard constraint is "do not touch
  `_drive_loop`" — every unit's done-condition needs a verification command that stays valid.
- **Suggested home:** project docs (`falkor-chat/AGENTS.md`, next to where the lock is quoted)


## 2026-07-24 — A `MERGE … ON CREATE SET` "create-only / immutable" write is only create-only for *properties*; its MERGE **patterns** still create structure on re-run

- **Evidence:** `falkor-chat/server/falkorchat/repository.py:937 _PUBLISH_CYPHER`. Re-publishing the
  same `(key, version)` with an edited spec is documented (falkor-chat/AGENTS.md, QUERIES §11.1) as a
  silent no-op, and QA confirmed the property half (`docs/archive/test-reports/m3-workflow-engine-report.md`
  §5 DEF-1: `201` returned, old `name`/`kind`/step `config` retained). But `MERGE (st:Step {stepUid:…})`,
  `MERGE (from)-[rel:TRANSITION {on, order}]->(to)` and `MERGE (d)-[:START]->(start)` are *patterns*:
  an added step, a changed `to`, or a changed start step **creates** new nodes/edges beside the old
  ones. So "immutable per version" is really "monotonically additive per version".
- **Context:** designing K-031, a read surface whose whole purpose is making that trap detectable —
  the additive half was undocumented and unasserted, and it is the more dangerous half (the executor
  then drives a def carrying both the old and the new edges).
- **Suggested home:** project docs (`falkor-chat/AGENTS.md` + `docs/QUERIES.md` §11.1), and possibly
  the architect prompt as a general review question about MERGE-based immutability claims.

## 2026-07-24 — In an OpenCypher `RETURN a, b, collect(DISTINCT …)`, a non-aggregated field from an `OPTIONAL MATCH` is a **grouping key** — the "collapses to one row" property is conditional, not structural

- **Evidence:** `falkor-chat/docs/QUERIES.md` §11.2 documents the collapse and states the premise out
  loud — *"`start.key` is constant across the fan-out so the grouping is well-defined"*. The consumer
  (`repository.py:976 _read_subgraph`) then takes `result_set[0]` unconditionally. If a second
  `(d)-[:START]->()` edge ever exists (reachable — see the entry above), the query yields one row per
  start key and the reader silently picks one. A verified-and-documented query can therefore carry a
  latent multi-row hazard that only fires on a data shape the write path was assumed to prevent.
- **Context:** K-031 design; drove a decision to have the new read consume all meta rows rather than
  inherit `result_set[0]`.
- **Suggested home:** knowledge base (`claude/graph-dba/falkordb-quirks.md`) — pending live
  confirmation on this build, which the plan schedules as verification V-1.

## 2026-07-24 — In falkor-chat, a **def publish has no graph seam**, so any live experiment on publish semantics must be run on the *snapshot* side of the same query constant

- **Evidence:** `repository.publish_def` (`server/falkorchat/repository.py:1011`) writes to
  `self._reference()` (`:132-134`) → `db.reference_graph` (`db.py:87-94`) = `select_graph("reference")`
  — a hardcoded literal with no parameter, env var or `config` override. There is no per-workspace
  def publish, so "publish a probe def into a throwaway graph" is impossible and the obvious
  improvisation writes into the global `reference`. The escape hatch: `materialize_snapshot`
  (`:1470-1490`) formats **the same `_PUBLISH_CYPHER` constant** with `label="WorkflowDefSnapshot"`
  against `self._graph(ws)`, and `_READ_META_CYPHER` is likewise label-templated — so the identical
  query text can be exercised in a throwaway `ws:<probe>` (bootstrap → 2 calls → `GRAPH.DELETE`) and
  the result transfers to the `WorkflowDef` side unchanged. Gotcha found while specifying it: the
  probe def needs **≥ 1 transition** — `_PUBLISH_CYPHER` ends in `UNWIND $transitions` and an empty
  list collapses the row stream, so `result_set[0]` raises `IndexError`.
- **Context:** revising the K-031 plan after an `analyst` gate flagged the scheduled live
  verification as unexecutable as written (its stated "isolated throwaway workspace" precondition
  could not exist).
- **Suggested home:** project docs (`falkor-chat/AGENTS.md`, near the `seed_workflows.sh` row that
  already warns about `reference` vs `ws:<id>` staleness) — plus, as a general architect habit, "check
  that a planned live probe has a graph/tenancy seam before scheduling it".
