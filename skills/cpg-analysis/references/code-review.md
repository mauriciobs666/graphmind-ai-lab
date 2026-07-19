# Recipe: code review (input → risky-sink taint)

> Back to [`../SKILL.md`](../SKILL.md) · schema in
> [`../../joern-cpg/references/cpg-model.md`](../../joern-cpg/references/cpg-model.md).
> **Consumer:** analyst. **Covers:** FR-12 / AC-7.

**Purpose.** Check whether externally-influenced input can reach a risky sink
(query/exec/eval/system/…). A tainted path is a finding; a genuinely clean target
returns **none**. **Change one parameter** — the method `FULL_NAME` under review,
and the **sink NAME list** (`$SINKS`) you consider risky.

`$SINKS` starter set (edit for the codebase): `['query','execute','exec','eval',
'system','Popen','run','call','subprocess','os_system']`. Here `query` is the
DB-write sink.

## Pattern A — Intraprocedural taint (primary, high precision)

Does a parameter of the method reach a risky sink's argument, within that method?
`REACHING_DEF` is intraprocedural, so this is exact when source and sink share a
method. Note `ARGUMENT_INDEX` is a **node** property on the argument (the
`ARGUMENT` edge has no properties).

```cypher
MATCH (m:METHOD {FULL_NAME: 'falkorchat/repository.py:<module>.Repository.post_first_message'})
      -[:AST]->(src:METHOD_PARAMETER_IN)
MATCH (m)-[:CONTAINS]->(sink:CALL)-[:ARGUMENT]->(arg)
WHERE sink.NAME IN ['query','execute','exec','eval','system','Popen','run']
MATCH (src)-[:REACHING_DEF*1..20]->(arg)
RETURN DISTINCT src.NAME AS taintedParam, sink.NAME AS sink, sink.LINE_NUMBER AS line
ORDER BY line
```

**AC-7 positive — verified.** Target `Repository.post_first_message`: the `text`
parameter reaches the `query` sink at **L320** (7 distinct `REACHING_DEF` paths).
Finding: caller-supplied message text flows into the write query.

> **Source set is *every* `METHOD_PARAMETER_IN`, not just external input — filter
> before you report.** `src` matches all of the method's parameters, so this exact
> query returns **9** `taintedParam` rows for `post_first_message` (`thread_id`,
> `role`, `msg_id`, `self`, `mentions`, `created_at`, `author_id`, `ws`, `text`) —
> `self` and internally-derived params are flagged purely for being arguments that
> reach the sink. That over-reports: the analyst must judge which sources are
> genuinely externally-influenced (here, `text`). Narrow the source set when you
> know the external inputs — e.g. add `WHERE src.NAME <> 'self'`, or pin the
> intended source(s) with `WHERE src.NAME IN ['text', …]` — or report the full set
> and annotate which parameters are actually attacker-controlled. Do **not** quote
> the raw row count as "N tainted parameters."

**AC-7 negative — verified.** Target `Services.ping` (a clean method): **0**
tainted paths. Empty result = no intraprocedural source→sink flow.

**Disambiguate real sinks from test doubles.** Constrain sink methods to
production files, e.g. `AND sink … ` runs inside a prod method — filter the
enclosing method by `m.FILENAME = 'falkorchat/repository.py'` (as above) so
`FakeRepo.*` doubles under `tests/` are excluded.

## Pattern B — Interprocedural reach (approximation)

When source and sink are in **different** methods, combine call-graph reach with
per-method flow. `REACHING_DEF` will not cross the call by itself; bridge over the
`CALL` edge, matching the call-site argument `ARGUMENT_INDEX` to the callee param
`INDEX`:

```cypher
// caller param crosses one resolved call into a callee param
MATCH (caller:METHOD)-[:AST]->(cp:METHOD_PARAMETER_IN)
MATCH (caller)-[:CONTAINS]->(cs:CALL)-[:CALL]->(callee:METHOD)
WHERE caller.IS_EXTERNAL = false AND callee.IS_EXTERNAL = false
MATCH (cs)-[:ARGUMENT]->(arg) WHERE arg.ARGUMENT_INDEX = cp.INDEX
MATCH (cp)-[:REACHING_DEF*1..10]->(arg)
MATCH (callee)-[:AST]->(kp:METHOD_PARAMETER_IN) WHERE kp.INDEX = cp.INDEX
RETURN DISTINCT caller.NAME AS caller, callee.NAME AS callee, kp.NAME AS crossedParam
```
Then resume Pattern A from `kp` inside `callee` to reach a sink.

**Interprocedural anchor — verified (guards the negative).** The bridge crosses a
real resolved edge: e.g. `create_channel → _graph` crosses parameter `ws`,
`read_def_subgraph → _read_subgraph` crosses `graph`. Because the crossing query
returns non-empty where an edge exists, a Pattern-A "clean → none" result is a
**true** clean, not an artifact of the query being unable to cross a call.

## Fidelity limit — read before trusting a clean result

This is **static graph reachability**, an approximation of Joern's `reachableBy`
taint engine, and this frontend's call resolution is sparse:

- **Same-object `self.x()` calls resolve; cross-object dispatch does not.** On
  `cpg_falkorchat` the real request path `api → Services → Repository.query`
  is **not** connected over the resolved `CALL` edge — the `Services→Repository`
  hop is cross-object dispatch (a repository handed to the service) and the final
  `.query()` is a method on an external handle. Verified: no production entrypoint
  reaches a `query` sink over the resolved call graph.
- Consequently Pattern B **under-reports** cross-object taint. A clean Pattern-B
  result across a cross-object boundary is **inconclusive**, not proof of safety.
- **For deep or cross-object taint, escalate to `joern`/CPGQL** (`reachableBy`) in
  the REPL — that is the recipe's failure path, not a bigger Cypher query.

Report both the positive findings (Pattern A paths, with lines) **and** the
coverage caveat (which boundaries Cypher could not cross), so a "no findings"
verdict is honest about its scope.
