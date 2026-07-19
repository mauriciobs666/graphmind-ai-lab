# Recipe: root-cause analysis (data flow + symbol def/ref)

> Back to [`../SKILL.md`](../SKILL.md) · schema in
> [`../../joern-cpg/references/cpg-model.md`](../../joern-cpg/references/cpg-model.md).
> **Consumer:** analyst. **Covers:** FR-11 / AC-4 (data flow), AC-5 (cross-file symbol).

**Purpose.** Two RCA questions: (1) *where did this bad value come from* — trace a
symptom back to the definitions that reach it; (2) *where is this symbol defined
and used* — locate a symbol's definition(s) and every reference across files.
**Change one parameter** — the enclosing method's `FULL_NAME` (data flow) or the
symbol `NAME` (def/ref).

## A. Data-flow slice — back from a symptom (AC-4)

`REACHING_DEF` links definitions to uses **within a single method**
(intraprocedural — see the boundary note). Trace **backward** from a symptom node
(a wrong `RETURN`, a suspect `CALL` argument) to the defs that feed it:

```cypher
MATCH (m:METHOD {FULL_NAME: 'falkorchat/api.py:<module>.build_router.post_message'})
      -[:CONTAINS]->(sym:RETURN)
MATCH (def)-[:REACHING_DEF*1..12]->(sym)
WHERE def.LINE_NUMBER IS NOT NULL AND NOT def:METHOD
RETURN DISTINCT def.LINE_NUMBER AS line, labels(def) AS kind, def.CODE AS code
ORDER BY line
```

Trace **forward** from a parameter instead (what a tainted input touches) by
swapping the anchor to a param and reversing the arrow:
```cypher
MATCH (m:METHOD {FULL_NAME: $full})-[:AST]->(p:METHOD_PARAMETER_IN {NAME: 'body'})
MATCH (p)-[:REACHING_DEF*1..12]->(use)
RETURN DISTINCT use.LINE_NUMBER, labels(use), use.CODE ORDER BY use.LINE_NUMBER
```

**Expected shape.** Line-ordered def/use nodes on the slice. **Verified**
(`build_router.post_message`, backward from `return posted`): reached the
definition of `posted` at L143 (`posted = services.post_message(...)`) and its
transitive feeders — `services` (param), `ctx`, `body.text`, the `_safe_*`
method refs — i.e. exactly the values that determine the returned object.

### Interprocedural boundary (must state in findings)

`REACHING_DEF` **does not cross a call**. The forward slice of `body` above ends
at the call site `services.post_message(...)` (L143) — it does not continue into
`Services.post_message`. To follow flow into a callee, bridge over the `CALL`
edge (`callSite -[:CALL]-> callee -[:AST]-> METHOD_PARAMETER_IN`, matching
argument `ARGUMENT_INDEX` to param `INDEX`) and resume `REACHING_DEF` there. That
bridge is only as complete as call resolution: **same-object `self.x()` resolves;
cross-object dispatch does not** (see cpg-model.md). When an RCA needs to cross a
cross-object boundary, say so and escalate the interprocedural slice to Joern's
`reachableBy` (the `joern` agent) — do not present an intraprocedural slice as if
it were the whole path.

## B. Cross-file symbol definition & references (AC-5)

**Definitions** (methods and types carry `FILENAME`; a `LOCAL` binding does not,
so resolve it through its enclosing method):
```cypher
MATCH (d) WHERE d.NAME = 'hybrid_search' AND (d:METHOD OR d:TYPE_DECL)
RETURN DISTINCT labels(d) AS kind, d.FULL_NAME AS fullName, d.FILENAME AS file, d.LINE_NUMBER AS line
ORDER BY file, line
```

**References across files** — references are `IDENTIFIER`/`CALL`/`METHOD_REF`
nodes with empty `FILENAME`; resolve each to the method that contains it:
```cypher
MATCH (ref) WHERE ref.NAME = 'hybrid_search'
  AND (ref:IDENTIFIER OR ref:CALL OR ref:METHOD_REF OR ref:FIELD_IDENTIFIER)
MATCH (owner:METHOD)-[:CONTAINS]->(ref)
RETURN DISTINCT owner.FILENAME AS file, labels(ref) AS refKind, ref.LINE_NUMBER AS line
ORDER BY file, line
```

> **`LOCAL` bindings are excluded from the ref query — state it.** The ref match
> deliberately omits `LOCAL`, so any `LOCAL` node named for the symbol is silently
> dropped (`hybrid_search` has **5** such `LOCAL` bindings that do **not** appear
> in the ref output). This is intentional — `LOCAL`s are scope-local declarations,
> not cross-file *uses*, and including them adds same-scope noise — but it means
> the ref query answers "where is this symbol *invoked/referenced*," not "every
> node that carries the name." If you need the local declarations too, add
> `d:LOCAL` to the **def** query (as `SKILL.md §3` does) and resolve through the
> enclosing method. Note this boundary in AC-5 findings.

**Expected shape.** Def query: one row per definition site — note Joern models a
callable as a co-located `METHOD` **and** `TYPE_DECL` at the same file:line, so a
function def yields two rows sharing a location (dedupe on `file`+`line` if you
only want sites); a synthetic `<returnValue>` method node with empty `FILENAME`
can also appear (filter `WHERE d.FILENAME <> ''` to drop it). Ref query: one row
per referencing location, keyed by the *owning method's* file. **Verified**
(`hybrid_search`, a genuinely cross-file first-party helper): definitions returned
the two production sites — `falkorchat/repository.py` L658 (`Repository.hybrid_search`)
and `falkorchat/services.py` L396 (`Services.hybrid_search`), each as a co-located
`METHOD`+`TYPE_DECL` pair — plus the test doubles (`FakeServices`/`FakeRepo`/
`StubServices`). References resolved to **three production files** —
`falkorchat/responder.py` L97, `falkorchat/services.py` L408, and
`falkorchat/tools.py` L293 (`CALL` sites) — demonstrating "references across
files" with real downstream call usage, plus test-file references
(`test_graphrag.py`, `test_services.py`, …). The 5 `LOCAL` `hybrid_search`
bindings are absent, as expected.

## Limits

- **Symbol matching is by `NAME`** and over-returns across scopes — two unrelated
  locals named `ctx` collide. Prefer `FULL_NAME` when the symbol is a method/type;
  for locals/identifiers, scope the result by the enclosing method/`TYPE_DECL`
  (the ref query already groups by owner) and state the collision caveat.
- **Data-flow is intraprocedural** (see boundary note) — a backward slice that
  returns "nothing upstream" may mean the cause is in a *caller*, not that the
  value is clean. Cross the boundary explicitly or escalate.
- `FIELD_IDENTIFIER` matches attribute names (`.get_context`), which can add
  unrelated hits; drop it from the ref query if the symbol is a free function.
