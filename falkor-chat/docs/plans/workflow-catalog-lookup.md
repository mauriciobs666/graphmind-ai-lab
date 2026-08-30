# Structured catalog/reference lookup for workflows — Implementation Plan

> **Status:** archived · **Owner:** `architect` · **Tracks:** K-052 (M6)

Turns `docs/requirements/workflow-catalog-lookup.md` (FR-1..FR-7, AC-1..AC-5) into an ordered,
staged build. **This document is also the canonical owner of the shared demo-agent scaffold** —
the single `salesperson` `WorkflowDef`, its one `agent` step, and the tool-registry wiring
pattern — that `docs/plans/workflow-cart-and-totals.md`, `docs/plans/workflow-durable-profile.md`
and `docs/plans/workflow-nl-query-generation.md` each extend by adding tools and bumping the
def's version. Those three plans cite this document by path for the scaffold rather than
re-describing it; do not read them without this one.

## 1. Goal & scope

**Goal.** Give a workflow author a fixed set of query shapes (exact-name fact lookup,
category/price-range filter) against a new, seed-script-only, workspace-independent electronics
catalog, and prove it inside the single combined "salesperson" demo agent shared with
cart-and-totals and durable-profile.

**In scope:** FR-1..FR-7, AC-1..AC-5; the `Product` schema in `reference`; the demo
`WorkflowDef` scaffold (shared canonical design, this document); `scripts/seed_catalog.sh` +
`scripts/verify_catalog.sh`; `scripts/seed_salesperson.sh` (v1 of the shared def).

**Out of scope** (per the requirements doc, §"Out of scope"): mutating catalog data from a
workflow; a runtime catalog CRUD API; per-workspace catalogs; attributes beyond name/category/
price; arbitrary-phrasing questions (`docs/plans/workflow-nl-query-generation.md`); rebuilding
`salesperson/`.

**CPG:** considered, not relevant — this is new-code design for `falkor-chat/server`, not an
impact analysis over existing call graphs. `cpg_falkorchat` is also stale as of this session
(brief: built 2026-08-26T22:27:22Z, one commit since); the coordinator's brief already directed
reading `executor.py`/`tools.py` directly rather than leaning on the CPG for structural claims,
which is what this plan is grounded in.

## 2. Context & findings

### 2.1 The comparison case and the gap

`salesperson/cypher.py`'s `cypher_qa` (free-form LLM→Cypher, guarded only by a regex keyword
blocklist, `_DANGEROUS_PATTERN`) is explicitly the wrong model to copy for *this* capability —
the requirements doc splits fixed-shape lookup (this document) from arbitrary-phrasing
generation (`workflow-nl-query-generation.md`) precisely because the two need different
mechanisms. `salesperson/cart.py`'s `_lookup_pastel` (a `MATCH (p:Pastel) RETURN ...` full scan +
in-Python fuzzy `SequenceMatcher` ranking, `cart.py:133-183`) is the nearer analog for *this*
capability's exact-lookup need, but it has no category/price-filter shape at all and runs
in-process against a single dev graph — not the multi-graph, per-`reference`-vs-`ws:{id}` model
this codebase uses.

### 2.2 Where the data belongs (`docs/DESIGN.md` §3, §4)

`reference` is explicitly the graph with "a placeholder box for domain reference data / ontology
/ catalogs" (`docs/DESIGN.md` §3) that "nothing is built there" yet — this is exactly that box.
FR-6 ("single, shared, global dataset — not scoped per workspace") maps directly onto `reference`
rather than `ws:{id}`: per §4, "large reference catalogs that are only *looked up* (not traversed
from workspace nodes) stay in `reference` and are reached by property key — no materialization."
`Product` is looked up by the demo agent's tools via ordinary parameterized `GRAPH.RO_QUERY`
reads against `reference` — no per-workspace materialization needed (unlike `WorkflowDef`, which
*is* traversed from workspace-local `WorkflowRun`s and therefore must be materialized).

### 2.3 Layering and the existing tool-registry pattern (`tools.py`, `executor.py`)

`tools.py` already establishes the exact shape a new lookup capability must take:
`PostMessageTool`/`GraphragRetrieveTool` are plain classes exposing `name`, `schema` (an OpenAI
function-calling JSON schema) and `run(arguments, *, ctx, run) -> str`, registered into a
`ToolRegistry` (`tools.py:99-140`) and dispatched by `executor._handle_tool_call`
(`executor.py:810-882`), which already enforces the AC-6 ungranted-tool fence and required-arg
validation before a tool ever runs. **Neither existing tool is exposed via `mcp.py`/`api.py`** —
they are purely internal node capabilities, dispatched only from inside `_run_agent_node`'s tool
loop. This plan (and the three sibling plans) follow the identical pattern: new tools are
`Tool`-protocol classes registered into the same registry, never new MCP/REST routes — consistent
with catalog-lookup's own "no runtime API" out-of-scope bullet and the analogous bullets in the
other three documents.

### 2.4 The single-step scaffold constraint (`services._validate_def_spec`, `executor.py`)

All four requirements docs describe the demo as "one orchestrating `agent` step, many tools" —
narrower than the *existing* `triage@v1` proof (`scripts/seed_workflows.sh:166-257`), which
actually spans **three** `agent` steps (`intake`→`research`→`answer`), each with its own tool
grant. The salesperson demo is one persistent `agent` step (`waitsForHuman: true`, no distinct
research/answer phase — one conversational loop that gains tools as sibling capabilities land).

Two engine facts constrain how that single step can be published, both verified by reading
`services.py`/`guards.py` directly (not the CPG, which is stale and uninvolved here):

1. **`_validate_def_spec` requires ≥ 1 transition** (`services.py:1376-1379`, "a def must carry
   **at least one** transition (K-024 U4b, O-6)"), a requirement the `docs/BACKLOG.md` item
   **K-030** (still open/proposed) would relax but has not yet. A def with one step and zero
   transitions is rejected at publish time today.
2. **An unconditional transition (`guard: ""`) always fires** (`guards.py:223-224`,
   `evaluate_guard`'s `if not guard: return GuardVerdict(decision=True, ...)`), and `_drive_loop`
   only takes the park path (OUTCOME B, `waitsForHuman`) when **no** transition fired
   (`executor.py:505-520`: OUTCOME A is checked before OUTCOME B). So an unconditional
   self-transition on the `agent` step would make it advance every turn instead of parking for
   the next customer message — silently breaking the whole "wait for the customer's next message"
   design the demo depends on.

**Resolution (this plan's scaffold decision, binding for all three sibling plans):** the single
step declares exactly **one conditional transition** to a second, terminal `decision`-typed step,
guarded on a `ctx` key nothing in this demo's tool set ever sets:
`{"kind":"cmp","path":"ctx.endConversation","op":"truthy"}`. This satisfies invariant 1 (a real
transition exists) without violating invariant 2 (a `cmp`-guard only fires when `ctx.
endConversation` is truthy, which never happens in this demo's acceptance path). This is not a
hack invented for this plan — it mirrors an existing, precedented pattern in this exact codebase:
`tools.HumanHandoffTool`/`HumanHandoffSignal` is "a registered capability that signals suspend...
present, not exercised: no triage node grants it" (`tools.py:338-353`). The `ended` transition is
the same kind of forward-looking, structurally-required-but-not-exercised affordance (a future
"the agent ends the conversation" extension has somewhere to go), not dead code smuggled in to
satisfy a validator.

### 2.5 The version-bump discipline this scaffold depends on (`docs/DESIGN.md` §4, K-034)

Per `docs/DESIGN.md` §4, "topology (steps, transitions, start) is immutable per version... a
re-publish/re-materialize whose topology differs from what's stored is rejected (`409`)...
Properties (`name`, `kind`, step `config`, transition `guard`) are **create-only** — a differing
resubmit of those stays a silent no-op, unchanged." This has a sharp, load-bearing consequence for
a def that gains tools incrementally across four plans: **re-publishing `salesperson@v1` with an
extra tool added to `config.tools` would silently no-op** — the property change is swallowed, and
the new tool would never actually reach a running agent. Each sibling plan therefore **bumps the
def's version** (`v1`→`v2`→`v3`→`v4`) when it adds tools, republishing the full cumulative step
config at the new version — topology (one `agent` step + the `ended` decision step + the one
`ctx.endConversation` transition) stays byte-identical across all four versions, so the `409`
topology-conflict path is never hit; only `config.tools`/`systemPrompt` differ, which is exactly
what a version bump is for (§9's write-paths table: "Bump version to change either").

## 3. Design & rationale

### 3.1 `Product` schema in `reference`

```
(:Product {productId, name, nameNormalized, category, price})
```

- `productId` — server-minted uuid, range index + UNIQUE constraint (the standard `{label}Id`
  identity pattern, `falkor-chat/AGENTS.md` schema conventions; `docs/DESIGN.md` §7.2).
- `nameNormalized` — case-folded, whitespace-collapsed `name`, computed with
  `extraction.normalize_name` (`server/falkorchat/extraction.py:67-78`) — **reused, not
  reimplemented**: that function's own docstring calls it "the ONE shared normalization helper"
  specifically to prevent a second, independently-written normalizer from silently drifting
  (`document-ingestion-ml.md` §3.2 lesson). Range index, no constraint (two products could in
  principle share a normalized name in a different category — not this demo's data, but the
  constraint should not assume it). Backs FR-1/AC-1/AC-4's exact-name lookup with a real `=`
  comparison, exactly the same tier-mechanics precedent `document-ingestion-graph.md` §2.3 already
  established for `Entity.nameNormalized` — not routed through RediSearch fuzzy matching, because
  an exact lookup should not depend on a search engine's tokenizer/stemmer behavior.
- `category`, `price` — plain range indexes, no constraint (FR-2/FR-3's filter/list shapes).
- No full-text index is added: FR-3 fixes three query *shapes*, not free-text search: AC-4's
  wording tolerance ("how much is the X" vs. "what's the price of the X") is handled by the LLM's
  own argument extraction when it calls the lookup tool (both phrasings extract the same product
  name argument), not by fuzzy matching in the database — the database side stays a strict `=`
  on the normalized name.

**DDL placement:** `scripts/bootstrap_schema.sh`'s `bootstrap_reference` function
(`bootstrap_schema.sh:37-70`), which — per `falkor-chat/AGENTS.md`'s "Key scripts" table — already
runs unconditionally ahead of the per-workspace loop and is exclusively `CREATE INDEX`/
`GRAPH.CONSTRAINT CREATE` (no `MERGE`/`CREATE (n)`/`DELETE`), so adding `Product`'s index-then-
constraint pair there is safe by the same reasoning already documented for that function. Index
before constraint (the live-verified ordering rule, `docs/DESIGN.md` §7.1 rule 1).

This is a routine data-modeling call, not delegated to `graph-dba`: three scalar properties, one
identity index/constraint pair and two hot-filter indexes, structurally identical to patterns
`bootstrap_schema.sh` already has for a dozen other labels. The coordinator's brief commissions
`graph-dba` notes only for `workflow-cart-and-totals-graph.md` and
`workflow-durable-profile-graph.md` (durable, workspace-scoped, frequently-*mutated* state) —
`Product` is global, seed-script-write-only, structurally simple, and does not warrant a separate
delegation. If a reviewer wants an independent DBA gate on this DDL before implementation, it is
a trivial one to add — flagged, not assumed away (§6).

### 3.2 Seed data path (FR-5)

A dedicated `scripts/seed_catalog.sh [<wsId>]` (the `<wsId>` argument is accepted only for CLI
convention parity with `seed_demo.sh`/`seed_workflows.sh`; the catalog itself is workspace-
independent) — mirrors `seed_demo.sh`'s role as *data*, separate from `seed_workflows.sh`'s role
as *workflow defs* (the same separation of concerns this codebase already keeps). It writes a
fixed catalog of ~15 consumer-electronics products (name/category/price triples — a Python or
shell-embedded literal, not fetched from anywhere) via one `UNWIND $rows AS row MERGE
(p:Product {productId: row.productId}) ON CREATE SET p.name = row.name, p.nameNormalized =
row.nameNormalized, p.category = row.category, p.price = row.price` — a plain guarded `MERGE`
backed by the `productId` UNIQUE constraint (AGENTS.md rule: "every `MERGE` is backed by a
uniqueness constraint"; no HEAD/TAIL pointer is involved, so the special guarded-`CREATE`-inside-
`FOREACH(CASE...)` idiom reserved for that hazard does not apply here — this is the same posture
`WorkflowDef` publish already takes for a create-only write). Idempotent: re-running the script
after a `test_queries.sh` teardown (which `GRAPH.DELETE`s `reference`, per `falkor-chat/AGENTS.md`)
re-creates the same rows with the same ids (deterministic `productId`s derived from a stable slug
of the product name, not `uuid4()` — so a re-seed after a wipe reconstructs byte-identical data,
unlike the non-idempotent Channel/Thread create precedent).

`scripts/verify_catalog.sh` — read-only (`GRAPH.RO_QUERY` only, mirroring `verify_workflows.sh`'s
posture), checks the expected product count and a couple of named products exist with the
expected category/price. Exit `0`/`1` with a printed diagnosis, same contract as
`verify_workflows.sh`.

### 3.3 The shared `salesperson` `WorkflowDef` (v1, this plan)

```python
# server/falkorchat/proof_defs.py — new constant, alongside ACCESS_REQUEST_DEF
SALESPERSON_DEF: dict[str, Any] = {
    "key": "salesperson",
    "version": "v1",              # bumped in place by each sibling plan's stage
    "name": "Salesperson",
    "kind": "conversation",
    "steps": [
        {
            "key": "assistant",
            "type": "agent",
            "start": True,
            "config": {
                "waitsForHuman": True,
                "systemPrompt": (
                    "You are a helpful electronics-store assistant chatting with a "
                    "customer.\n\n"
                    "You can answer factual questions about specific products (name, "
                    "category, price) and list products matching a category or price "
                    "range, using your catalog tools. Never guess a price or category "
                    "you have not retrieved from a tool; if nothing matches, say so "
                    "plainly rather than inventing an answer.\n\n"
                    "Deliver every reply by calling the `post_message` tool; text you "
                    "merely write is never seen by the customer. Never pass `mentions`; "
                    "omit that argument entirely."
                ),
                "tools": ["post_message", "lookup_product_fact", "filter_products"],
                "requiredTools": ["post_message"],
                "maxIterations": 8,
            },
        },
        {"key": "ended", "type": "decision", "config": {}},
    ],
    "transitions": [
        {
            "from": "assistant", "to": "ended", "on": "ended", "order": 0,
            "guard": {"kind": "cmp", "path": "ctx.endConversation", "op": "truthy"},
        },
    ],
}
SALESPERSON_MAX_STEPS = 40  # generous: one long-lived conversational loop, many turns
```

Placed in `proof_defs.py`, **not** inlined in the seed script — deliberately following
`ACCESS_REQUEST_DEF`'s convention rather than `triage@v1`'s still-inline-in-`seed_workflows.sh`
pattern, which `docs/BACKLOG.md`'s **K-029** already flags as a known, undesirable divergence
("the two seeded defs use two different source conventions... filed out of K-024"). Building a
brand-new def against the pattern K-029 wants everyone converged on, rather than the one it wants
retired, avoids adding a third inline copy for K-029 to later have to consolidate.

`maxIterations: 8` (vs. `triage`'s per-node cap of 4): calibration seed, not load-bearing — this
one step does everything `triage` split across three nodes (intake+research+answer), so it
plausibly needs more tool-call rounds per customer turn. Coder/QA calibrate on real runs, same
posture `tools.py`'s own τ/cap/k constants document for `GraphragRetrieveTool`.

`SALESPERSON_MAX_STEPS = 40`: a `WorkflowRun.maxSteps` budget, not to be confused with
`schemas.MAX_STEPS = 200` (the publish-time cap on a def's *step count*, an unrelated axis). This
demo runs many customer turns over one long-lived run (unlike `triage`'s few-turn intake), so a
larger step budget than `ACCESS_REQUEST_MAX_STEPS`'s 24 is appropriate — the number is a tripwire,
not an SLA (`docs/DESIGN.md` §6.2's "What `maxSteps` actually means" note applies unchanged).

### 3.4 Seed script for the def: `scripts/seed_salesperson.sh`

New script, mirroring `seed_workflows.sh`'s shape: publish `SALESPERSON_DEF` (imported from
`proof_defs.py`, not re-typed) into `reference`, then materialize into `ws:<id>` (default
`ws:acme`, override via the same `WS_ID` convention `seed_workflows.sh` uses). This script is
**edited in place** by each sibling plan's implementation stage to bump `SALESPERSON_DEF`'s
`version` and `config.tools` — it is the same evolving artifact `seed_workflows.sh` itself is
across K-022/K-024/etc., not a new script per capability.

`scripts/verify_salesperson.sh` — mirrors `verify_workflows.sh`: confirms `reference` and
`ws:<id>` agree on `salesperson@<expected-version>`, one start key, right topology.

### 3.5 The two catalog tools (FR-1/FR-2/FR-3/FR-4)

New file `server/falkorchat/tools.py` additions (peers of `PostMessageTool`/
`GraphragRetrieveTool`, same file — no new module needed for two small tools):

```python
class LookupProductFactTool:
    name = "lookup_product_fact"
    # schema: {"name": {"type": "string", "description": "The product's name, as the "
    #   "customer referred to it."}}, required: ["name"]
    def run(self, arguments, *, ctx, run) -> str:
        row = self._services.lookup_product(ctx, name=arguments.get("name", ""))
        if row is None:
            return json.dumps({"found": False})
        return json.dumps({"found": True, **row})  # {name, category, price}

class FilterProductsTool:
    name = "filter_products"
    # schema: {"category": {"type":"string"}, "minPrice": {"type":"number"},
    #   "maxPrice": {"type":"number"}} — all optional, at least one should be supplied
    #   (not enforced structurally; an all-omitted call just lists the whole catalog,
    #   bounded by DEFAULT_FILTER_LIMIT below — acceptable for a ~15-row demo catalog)
    def run(self, arguments, *, ctx, run) -> str:
        rows = self._services.filter_products(
            ctx, category=arguments.get("category"),
            min_price=arguments.get("minPrice"), max_price=arguments.get("maxPrice"),
            limit=DEFAULT_FILTER_LIMIT,
        )
        if not rows:
            return json.dumps({"items": [], "finding": "no matching products found"})
        return json.dumps({"items": rows})
```

`DEFAULT_FILTER_LIMIT = 20` (a demo-scale cap; the real result-set cap is FalkorDB's own
`RESULTSET_SIZE` default of 10000, irrelevant at this catalog's size — `claude/graph-dba/
falkordb-quirks.md`). Both tools' abstention shape (`{"found": false}` / `{"items": [],
"finding": "no matching products found"}`) mirrors `GraphragRetrieveTool`'s own "no relevant
context found" abstention convention (`tools.py:317-318`) — one consistent "nothing matched, say
so plainly" idiom across every lookup tool in this codebase, directly satisfying FR-4.

**`services.py` additions** (new methods, following the existing `hybrid_search`-style shape —
thin, `repository`-delegating, no Cypher of their own): `lookup_product(ctx, *, name) -> dict |
None` (normalizes `name` via `extraction.normalize_name`, calls `repository.lookup_product`);
`filter_products(ctx, *, category, min_price, max_price, limit) -> list[dict]` (calls
`repository.filter_products`). Neither takes `ctx.ws` into the query (the catalog lives in
`reference`, not `ws:{id}`) — `ctx` is accepted only for signature symmetry with every other tool-
backing service method and for future auditing (e.g., logging which actor asked), not because the
catalog itself is workspace-scoped.

**`repository.py` additions**: `lookup_product(name_normalized) -> dict | None` and
`filter_products(category, min_price, max_price, limit) -> list[dict]`, both against
`self._reference()` (`repository.py:171-173`, the existing `db.reference_graph()` accessor —
**not** a new graph seam; publish already goes through the same accessor), both `ro_query` reads.
Cypher (illustrative, exact param names/ordering are the implementer's to finalize against
`QUERIES.md`'s conventions):

```cypher
-- lookup_product
MATCH (p:Product {nameNormalized: $name})
RETURN p.name AS name, p.category AS category, p.price AS price
LIMIT 1

-- filter_products (category/min/max each optional; build the WHERE clause from
-- only the supplied params, never string-interpolate the values themselves)
MATCH (p:Product)
WHERE ($category IS NULL OR p.category = $category)
  AND ($minPrice IS NULL OR p.price >= $minPrice)
  AND ($maxPrice IS NULL OR p.price <= $maxPrice)
RETURN p.name AS name, p.category AS category, p.price AS price
ORDER BY p.price ASC
LIMIT $limit
```

Both additions get entries in `docs/QUERIES.md` (new subsection) and a `test_queries.sh` baseline
bump, per the standing "keep the query suite green" rule.

## 4. Step-by-step implementation

1. **`scripts/bootstrap_schema.sh`** — add `Product`'s index-then-constraint pair + the two
   hot-filter indexes to `bootstrap_reference` (§3.1). Done: `bootstrap_schema.sh <anyWsId>` runs
   clean; `CALL db.indexes()`/`db.constraints()` show the new entries.
2. **`docs/QUERIES.md`** — add the `lookup_product`/`filter_products` Cypher (§3.5) as a new
   subsection; **`scripts/test_queries.sh`** — add matching assertions (baseline count bump).
3. **`server/falkorchat/repository.py`** — `lookup_product`, `filter_products` (§3.5).
4. **`server/falkorchat/services.py`** — `lookup_product`, `filter_products` (§3.5).
5. **`server/falkorchat/tools.py`** — `LookupProductFactTool`, `FilterProductsTool` (§3.5);
   extend `build_builtin_registry` (or a new `build_salesperson_registry` — implementer's call,
   since the salesperson demo's tool set is disjoint from triage's) to register them.
6. **`scripts/seed_catalog.sh`** + **`scripts/verify_catalog.sh`** (§3.2) — new.
7. **`server/falkorchat/proof_defs.py`** — `SALESPERSON_DEF` (§3.3), imported (not copied) by:
8. **`scripts/seed_salesperson.sh`** + **`scripts/verify_salesperson.sh`** (§3.4) — new.
9. **`falkor-chat/AGENTS.md`** — add the four new scripts to the "Key scripts" table, per the
   repo's documentation rule (`SERVER.md` §1.6's reminder applies equally here).

Sequenced so the tree stays buildable: steps 1-2 are pure DDL/data-shape groundwork; 3-5 are the
read path + tool wiring (independently unit-testable against a fake `Services`, mirroring
`test_services.py`'s `FakeRepo` pattern — `falkor-chat/AGENTS.md`'s "review-safe pytest subset"
note); 6 seeds data; 7-8 wire the demo def; 9 is documentation hygiene.

**Done (this plan):** `seed_catalog.sh` + `seed_salesperson.sh` (v1) run clean; a live
`@mention`-triggered run of `salesperson@v1` answers a single-item fact question and a
category/price-range filter question correctly, and states plainly when nothing matches.

## 5. Test strategy

| AC | What proves it | Altitude |
|---|---|---|
| AC-1 (single-item fact) | `Repository.lookup_product` against a seeded `ws:test`-adjacent `reference` fixture; `LookupProductFactTool.run` with a fake `Services`; live `@mention` asking "how much does the X cost" | repository/service unit + tool unit + live e2e |
| AC-2 (filter/list) | Same three altitudes for `filter_products`, incl. a category filter and a price-range filter as two separate cases | repository/service unit + tool unit + live e2e |
| AC-3 (abstention) | A name/category absent from the seeded catalog returns `{"found": false}` / `{"items": [], "finding": ...}`, never a fabricated row | repository/service unit |
| AC-4 (wording tolerance) | Two live `@mention` turns with differently-worded questions for the same product ("how much is the X" / "what's the price of the X") both resolve to the same `lookup_product_fact(name="X")` tool call and the same answer | live e2e (this is inherently about the LLM's own argument extraction, not the DB layer — the DB-layer test only proves the exact-match lookup is correct once given the right name) |
| AC-5 (seed + verify) | `seed_catalog.sh` + `seed_salesperson.sh` run clean; `verify_catalog.sh` + `verify_salesperson.sh` report in-sync | script-level (mirrors `verify_workflows.sh`'s existing role) |

**Additional coverage:** `_validate_def_spec` accepting `SALESPERSON_DEF` at publish (the one
conditional transition + the `decision` terminal step); a republish of `salesperson@v1` with a
byte-identical topology succeeding as a no-op (proves the version-bump discipline §2.5 depends on
is real, not assumed); the `ctx.endConversation` transition never firing across an ordinary
multi-turn conversation (regression guard for the §2.4 scaffold decision — a false-positive fire
would silently end the demo run mid-conversation).

## 6. Risks & open questions

- **DDL not independently DBA-gated.** §3.1 makes a routine call rather than delegating to
  `graph-dba`, reasoned from this codebase's existing patterns. If the coordinator wants an
  independent review before implementation, route the DDL (only, not the whole plan) to
  `graph-dba` for a quick sanity pass — cheap to add, not assumed necessary here.
- **`extraction.normalize_name` living in a module named for a different feature.** Reusing it
  (§3.1) is the right call (avoids a second normalizer drifting from the first), but the import
  (`from .extraction import normalize_name` inside `repository.py`/`services.py`, for a feature
  that has nothing to do with document extraction) is a minor naming smell. Not blocking; an
  implementer or a later pass could relocate it to a neutral `textutils.py` and update both call
  sites — noted so it isn't mistaken for an oversight.
  Since this plan is written before that decision needs
  to be made, the maxIterations/step-budget constants above are calibration seeds only.
- **The `ctx.endConversation` transition is genuinely unreachable in this milestone's acceptance
  path** (§2.4) — by design, mirroring the precedented `human_handoff`-present-not-exercised
  pattern, but worth flagging explicitly so a future reviewer doesn't mistake it for dead code to
  delete: removing it would make `SALESPERSON_DEF` unpublishable under the current (pre-K-030)
  engine.
- **K-030 dependency, non-blocking.** If `docs/BACKLOG.md`'s K-030 (allow zero-transition defs)
  ships before this plan is implemented, the `ended` step + transition become optional rather than
  required — leave them in anyway for forward-compatibility (a real "end conversation" affordance)
  rather than removing them opportunistically.
