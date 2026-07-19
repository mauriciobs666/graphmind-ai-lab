# Recipe: impact analysis

> Back to [`../SKILL.md`](../SKILL.md) · schema in
> [`../../joern-cpg/references/cpg-model.md`](../../joern-cpg/references/cpg-model.md).
> **Consumers:** analyst, architect. **Covers:** FR-10 / AC-2, AC-3.

**Purpose.** Given a function, find who calls it, what it calls, and what a change
to it could transitively break. **Change one parameter** — the target's short
`NAME` (and `FILENAME` to disambiguate) or its `FULL_NAME`.

Set the graph key once (caller-supplied; never hardcode):
`GRAPH=<graph>; PORT=${FALKORDB_PORT:-6379}`, then
`redis-cli -p "$PORT" GRAPH.QUERY "$GRAPH" "<cypher>" --no-raw`.

## Q1 — Direct callers (AC-2)

Callers are matched by call-site `NAME`; the caller is the containing method.
This is the reliable direction (inbound `CALL`-edge resolution is too sparse).

```cypher
MATCH (caller:METHOD)-[:CONTAINS]->(c:CALL {NAME: 'post_message'})
RETURN DISTINCT caller.FULL_NAME AS caller, caller.FILENAME AS file, caller.LINE_NUMBER AS line
ORDER BY file, line
```

**Expected shape.** One row per calling method, with its file + line. If the
name collides across classes and you want only callers of a *specific* target,
add `WHERE c.METHOD_FULL_NAME CONTAINS '<ClassName>'` (best-effort — that key is
inconsistent, see cpg-model.md) or post-filter by the caller files you expect.

**Verified** (`cpg_falkorchat`, target `post_message`): returned **21 distinct
callers** — the two production callers
`falkorchat/api.py:…build_router.post_message` and `falkorchat/mcp.py:…send_message`,
plus 19 `tests/…` callers. To keep only production callers, add
`WHERE NOT caller.FILENAME STARTS WITH 'tests/'`.

## Q2 — Direct callees (AC-2)

Two flavours — pick by what you need:

**(a) Resolved first-party callees** (precise, but omits dynamic/cross-object
dispatch the frontend can't resolve):
```cypher
MATCH (m:METHOD {FULL_NAME: 'falkorchat/services.py:<module>.Services.post_message'})
      -[:CONTAINS]->(:CALL)-[:CALL]->(callee:METHOD)
RETURN DISTINCT callee.FULL_NAME AS callee ORDER BY callee
```

**(b) All invoked names** (broader; includes unresolved external calls; filter
Joern's synthetic operators):
```cypher
MATCH (m:METHOD {FULL_NAME: 'falkorchat/services.py:<module>.Services.post_message'})
      -[:CONTAINS]->(c:CALL)
WHERE NOT c.NAME STARTS WITH '<operator>'
RETURN DISTINCT c.NAME AS callee, c.METHOD_FULL_NAME AS resolvedTo ORDER BY callee
```

**Verified** (`Services.post_message`): (a) returned `_dispatch_write`,
`_next_ts`, `_validate_and_derive_role`. (b) additionally surfaces the same three
by name. Note the repository writes (`post_first_message` / `post_subsequent_message`)
are **not** direct callees — they are reached from `_dispatch_write` via a dynamic
`write` dispatch the frontend does not statically resolve (see Limits).

## Q3 — Transitive downstream reach (AC-3)

"What could break if I change `$full`." Only `:METHOD` nodes are reachable across
a `CALL` edge, so terminating the mixed walk at `:METHOD` yields real call reach.
**Always bound the depth** and filter to first-party to keep the frontier sparse.

```cypher
MATCH (m:METHOD {FULL_NAME: 'falkorchat/services.py:<module>.Services.post_message'})
      -[:CONTAINS|CALL*1..8]->(reached:METHOD)
WHERE reached.IS_EXTERNAL = false AND reached <> m
RETURN DISTINCT reached.FULL_NAME AS reached ORDER BY reached
```

For transitive **upstream** ("everything that depends on X"), the resolved
`CALL` edge is unreliable inbound. Iterate Q1 by name instead: run Q1 for the
target, then re-run Q1 for each caller's `NAME`, to the depth you need. State the
name-collision caveat when you do (two `post_message` methods share a `NAME`).

**Verified** (`Services.post_message`, depth 8): returned `_dispatch_write`,
`_next_ts`, `_validate_and_derive_role`, and (depth 2) `_dedup` — the resolved
first-party downstream set. Reverse transitive reach over the same edge returns
only synthetic module/adapter nodes, confirming the inbound-resolution limit.

## Limits

- **Downstream is trustworthy; upstream over the `CALL` edge is not.** Callers
  must be found by `NAME` (Q1), not by reversing Q2/Q3. See cpg-model.md
  "inbound call resolution is sparse."
- **Dynamic / cross-object dispatch is invisible.** Calls through a handed-in
  object, a dispatch table, or `getattr` (here: `_dispatch_write` → `write`) have
  no resolved `CALL` edge, so downstream reach stops early. Treat Q3 as a *lower
  bound* on reach. For a fuller call graph, escalate to Joern's CPGQL
  (`.caller` / `.callee` / `.reachableBy`) in the REPL via the `joern` agent.
- **Scope:** AC-3 is call-graph reach over the `CALL` relationship only. Type /
  import / inheritance dependency reach is out of scope for this recipe.
