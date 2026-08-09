# Recipe: impact analysis

> Back to [`../SKILL.md`](../SKILL.md) · schema in
> [`../../joern-cpg/references/cpg-model.md`](../../joern-cpg/references/cpg-model.md).
> **Consumers:** analyst, architect. **Covers:** FR-10 / AC-2, AC-3.

**Purpose.** Given a function, find who calls it, what it calls, and what a change
to it could transitively break. **Change one parameter** — the target's short
`NAME` (and `FILENAME` to disambiguate) or its `FULL_NAME`.

Pass the graph key (caller-supplied; never hardcode) and the Cypher below as the
two parameters of `mcp__cpg__query` — `graph` and `cypher`. Outside Claude Code,
or if the tool is unavailable, use the `redis-cli` fallback in
[`../SKILL.md`](../SKILL.md) §1.

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
`CALL` edge is unreliable inbound — see Q4 below for the bounded, name-based
closure that replaces manual iteration.

**Verified** (`Services.post_message`, depth 8): returned `_dispatch_write`,
`_next_ts`, `_validate_and_derive_role`, and (depth 2) `_dedup` — the resolved
first-party downstream set. Reverse transitive reach over the same edge returns
only synthetic module/adapter nodes, confirming the inbound-resolution limit.

## Q4 — Transitive upward call closure (AC-3, bounded, name-based)

"Who calls `X`, transitively" — everything that could be affected by a change to
`X`'s signature or contract, several hops up the call graph. **Do not** try to
get this by reversing the `CALL` edge (`(:CALL)-[:CALL]->(:METHOD)` walked
backward) or by wrapping Q1 in a variable-length pattern — both depend on
*inbound* call resolution, which cpg-model.md documents as sparse on this
frontend (~1.3k of ~20k call sites resolved, cross-object dispatch excluded).
A naive composition along those lines returned **0 rows** on the live graph for
a real target. This recipe instead climbs by **`CALL.NAME`**, one level at a
time, the same `WITH`-splitting idiom `test-gap.md` uses for its downward
closure — just walked in the opposite direction: each level asks "who contains
a call site named after someone in the previous level," instead of "what does
someone in the previous level call."

Bounded to 3 expansion levels (extend by copying the middle block for more
depth, same as Q3/`test-gap.md`).

```cypher
// L1: names of methods that directly call the target (call-site NAME match —
// the reliable direction; inbound CALL-edge resolution is too sparse to trust)
MATCH (c1:METHOD)-[:CONTAINS]->(:CALL {NAME: 'post_message'})
WITH collect(DISTINCT c1.NAME) AS L1
// L2: names of methods that call anything named in L1 (one hop further up)
MATCH (c2:METHOD)-[:CONTAINS]->(call2:CALL) WHERE call2.NAME IN L1
WITH L1, collect(DISTINCT c2.NAME) AS L2raw
WITH L1, [x IN L2raw WHERE NOT x IN L1] AS L2new
WITH L1 + L2new AS L12
// L3: one more hop up
MATCH (c3:METHOD)-[:CONTAINS]->(call3:CALL) WHERE call3.NAME IN L12
WITH L12, collect(DISTINCT c3.NAME) AS L3raw
WITH L12, [x IN L3raw WHERE NOT x IN L12] AS L3new
WITH L12 + L3new AS closure
// resolve the NAME closure back to concrete caller methods
MATCH (caller:METHOD) WHERE caller.NAME IN closure
RETURN DISTINCT caller.FULL_NAME AS caller, caller.FILENAME AS file, caller.LINE_NUMBER AS line
ORDER BY file, line
```

> Do **not** add `AND caller.NAME <> '<target>'` to guard against "self-recursion"
> — a first attempt at this recipe did, and it silently dropped a legitimate
> caller (see the verified example below). A method that truly calls itself
> belongs in its own upward closure; Q1 doesn't filter it out either, and
> neither should this.

> **FalkorDB idiom note (same one `test-gap.md` documents).** Do not fold an
> aggregation and a reference to a prior list into the same `WITH` — this build
> raises `_AR_EXP_UpdateEntityIdx: Unable to locate a value with alias`. Each
> level therefore splits into two `WITH` steps: one that aggregates
> (`collect(...)`), one that filters against the running set
> (`[x IN … WHERE NOT x IN …]`).

**Expected shape.** One row per method that transitively reaches the target
through the call graph, up to 3 levels up, with its file + line — the upward
mirror of Q1's shape.

**Verified** (`cpg_falkorchat`, target `post_message`, 39.2ms): returned **24
rows**, against Q1's **21** direct callers for the same target. The diff proves
both halves of this recipe work, and both are visible in the same run:

- **+1 genuine transitive addition** — `tests/test_workflow_live.py:test_triage_flow_runs_end_to_end_against_live_llm`
  is *not* a direct caller of `post_message` (absent from Q1's result); it calls
  `_post_and_trigger`/`_seed_conversation`, which *are* direct callers. It only
  surfaces at L2, confirming the `WITH`-splitting climb actually reaches a
  second hop and isn't just re-listing L1.
- **+2 name-collision artifacts** — `falkorchat/services.py:…Services.post_message`
  (the target's **own definition**) and a Joern synthetic
  `falkorchat/services.py:…Services.__init__.<returnValue>.post_message`
  (`IS_EXTERNAL = true`, empty `FILENAME`) both appear. Neither is a real
  upward caller. They enter because `falkorchat/api.py:…build_router.post_message`
  — a genuine, distinct caller — happens to have the **same short `NAME`**
  (`post_message`) as the target, so `'post_message'` itself lands in `L1`, and
  the final `caller.NAME IN closure` match pulls in *every* method named
  `post_message`, target and synthetic stub included. Confirmed directly:
  `MATCH (m:METHOD) WHERE m.NAME = 'post_message' RETURN m.FULL_NAME, m.IS_EXTERNAL`
  returns exactly those 3 methods (the route handler, the target, the synthetic).

**State the name-collision caveat precisely** (sharper than "a caller of one
reported as a caller of the other" — this is the live-verified shape it takes):
when the target's own short `NAME` is shared by another real caller, the
resolved closure will include the **target's own definition** and any
Joern-synthetic same-named stub as if they were callers. Drop the noise by
adding `AND caller.IS_EXTERNAL = false` (removes synthetic stubs) and
excluding the target's own site by `FILENAME`+`LINE_NUMBER` once you know it
(removes the target's own definition) — do **not** filter by `caller.NAME <>
'<target>'`, which over-removes (see the query note above).

## Limits

- **Downstream is trustworthy; upstream over the `CALL` edge is not.** Callers
  must be found by `NAME` (Q1 for direct, Q4 for transitive), not by reversing
  Q2/Q3. See cpg-model.md "inbound call resolution is sparse."
- **Q4's name-based closure can resolve back to the target's own definition
  and Joern-synthetic stubs**, not just real callers, whenever the target's
  short `NAME` collides with another caller's `NAME` — live-verified on
  `post_message` (see Q4). Filter `IS_EXTERNAL = false` and, if needed, the
  target's own `FILENAME`+`LINE_NUMBER` out of the result.
- **Dynamic / cross-object dispatch is invisible.** Calls through a handed-in
  object, a dispatch table, or `getattr` (here: `_dispatch_write` → `write`) have
  no resolved `CALL` edge, so downstream reach stops early. Treat Q3 as a *lower
  bound* on reach. For a fuller call graph, escalate to Joern's CPGQL
  (`.caller` / `.callee` / `.reachableBy`) in the REPL via `graph-dba`.
- **Scope:** AC-3 is call-graph reach over the `CALL` relationship only. Type /
  import / inheritance dependency reach is out of scope for this recipe.
