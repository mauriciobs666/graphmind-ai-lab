# Natural-language query generation over structured graph data — Implementation Plan

> **Status:** active · **Owner:** `architect` · **Tracks:** K-055 (M6) · **Version:** 1.1

> **Revision note (2026-08-26).** `security-expert`'s `docs/reviews/workflow-nl-query-generation-security.md`
> (verdict: approve with suggestions, no blocker) independently reproduced this plan's Layer 2
> claim live and found two MAJOR gaps in Layer 1's own specification — this revision closes both,
> plus the two minors, in §3.1/§3.4/§5/§6 below. No design change: the two-layer architecture and
> the DSL-over-free-form-Cypher decision (§2.3) stand as reviewed.

Turns `docs/requirements/workflow-nl-query-generation.md` (FR-1..FR-5, AC-1..AC-5) into an
ordered, staged build. The golden-set evaluation methodology and passing threshold (FR-4/AC-4) are
delegated to `data-scientist`'s `docs/plans/workflow-nl-query-generation-ml.md`; **this plan owns
the query-generation mechanism itself (FR-1/FR-2) and a concrete, structural design for FR-3's
non-mutation guarantee (FR-3/FR-3a)** — reviewed by `security-expert`
(`docs/reviews/workflow-nl-query-generation-security.md`) prior to the `analyst` plan gate.
**Read `docs/plans/workflow-catalog-lookup.md` first** (the shared `salesperson` `WorkflowDef`
scaffold) — not re-described here.

## 1. Goal & scope

**Goal.** Let the combined demo agent answer an arbitrarily-phrased question against structured
graph data — not limited to `workflow-catalog-lookup.md`'s fixed query shapes — via a mechanism
that (a) generalizes across datasets without code changes and (b) is *structurally* incapable of
a mutating/destructive graph operation, regardless of phrasing.

**In scope:** FR-1..FR-5, AC-1..AC-5; the query-generation mechanism; the FR-3/FR-3a structural
safety design (mechanism-level; the adversarial test-case set itself is `security-expert`'s to
design, per the decision log); the AC-2 second-dataset verification plan against the shipped
document-ingestion schema.

**Out of scope** (per the requirements doc): the golden-set metric/threshold (`data-scientist`);
a comprehensive independent security audit beyond FR-3a's baseline adversarial set
(`security-expert`, later, additive); any write/mutating capability; `workflow-catalog-lookup.md`'s
own fixed shapes; rebuilding `salesperson/`.

**Documentation nit (flag, don't block on):** `workflow-nl-query-generation.md`'s "Related work"
section describes `docs/requirements/document-ingestion.md` as "active `teco` coordination in
flight." Per the coordinator's brief, that coordination closed 2026-08-25 (M5 delivered,
`docs/HISTORY.md` "K-050 M5 closes"); the requirements doc's own text is stale. This does not
change AC-2's design (§3.5) — `tico` should correct the requirements doc's wording at the next
touch, but this plan does not block on that edit.

**CPG:** considered, not relevant — new-code design over the current tree; `cpg_falkorchat` is
stale and uninvolved (same posture as the three prior plans in this set). `services.py`/
`repository.py`/`tools.py` and `claude/graph-dba/falkordb-quirks.md` (the live-verified FalkorDB
engine facts this plan's structural argument depends on) were read directly.

## 2. Context & findings

### 2.1 The comparison case, precisely (`salesperson/cypher.py`)

`cypher_qa` (`salesperson/cypher.py:131-172`) is the mechanism this document's Problem & current
state explicitly rejects as a model to copy: an LLM generates free-form Cypher text
(`cypher_chain.invoke(...)`), which is then checked against `_DANGEROUS_PATTERN` — a regex
matching the bare words `create|delete|set|remove|drop|detach|merge` anywhere in the query
(`cypher.py:32-35`) — and, separately, against `_ALLOWED_CYPHER_KEYWORDS`, a set the code only
ever uses to *log* unrecognized words, never to reject them (`cypher.py:66-70`, `_validate_safe_cypher`
returns `(True, None)` regardless of what `unknown` contains). This is a **keyword blocklist**,
the exact mechanism FR-3's decision log names as unacceptable, and it is also trivially defeated
by construction — e.g. a query using `remove` as a *label or property name* rather than a clause
keyword still matches the regex and is rejected (a false positive), while nothing in the regex
stops a generated call to a write-capable **procedure** that doesn't contain any of those seven
words at all (a false negative this plan does not need to construct hypothetically, because §2.3
shows the mechanism below never reaches procedure calls at all).

### 2.2 The live-verified engine fact this plan's structural backstop rests on

`claude/graph-dba/falkordb-quirks.md` ("Ops, config & tooling" section) records a **live-verified**
FalkorDB behavior on this deployment's pinned build (v4.18.11): **`GRAPH.RO_QUERY` refuses a write
query at the engine level**, returning the error "`graph.RO_QUERY is to be executed only on
read-only queries`" — confirmed by graph-dba's own workaround note for syntax-checking a write
query with no graph to point `GRAPH.EXPLAIN` at ("run the write query via `GRAPH.RO_QUERY` against
any existing graph — it parses the query first and only then rejects it for being a write"). This
is an **engine-enforced** property, not an application-level check: `repository.py` already relies
on this distinction everywhere (every read method in the codebase calls `.ro_query()`, every write
calls `.query()` — `grep -c '\.ro_query(' server/falkorchat/repository.py` finds dozens of existing
call sites, §"Context" investigation). The one documented way this guarantee can be bypassed is a
`GRAPH.PROFILE`-prefixed string, which *executes* a write despite looking like a read directive
(`falkordb-quirks.md`'s "GRAPH.PROFILE is not read-only" entry) — but that bypass is about which
**top-level Redis command** wraps a query string, not about what the query string itself contains.
§3.2's mechanism never lets the query string reach anything but a hardcoded `.ro_query(...)` call
— the app code never inspects the generated text to decide which Redis command to issue, so there
is no sniffing step for a `PROFILE`/`EXPLAIN` prefix to defeat.

### 2.3 Why a constrained DSL, not an AST parser over free-form Cypher

The coordinator's brief names two candidate structural mechanisms: "restrict a generated query's
clause set at the AST/grammar level to read-only forms, or use a constrained query-builder DSL
instead of free-form LLM-generated Cypher." This plan picks the **second**, for a concrete reason
found while investigating the first: building a real (non-regex) Cypher grammar/AST validator
in Python, robust against the exact evasions this codebase has already live-verified against
naive prefix/keyword sniffing (comment-hiding a directive, `falkordb-quirks.md`'s "an
`EXPLAIN`/`PROFILE`/`profile` prefix... including after a `//` or `/* */` comment... is silently
ignored" finding), means either vendoring a full Cypher grammar (this project has no ANTLR/Cypher-
parser dependency today, and FalkorDB's own OpenCypher subset is not a byte-for-byte match for any
off-the-shelf openCypher grammar) or hand-rolling a tokenizer good enough to be trusted for
security purposes — a large, hard-to-fully-audit surface for a capability whose acceptance bar
(FR-3a) is specifically "prove this can't be talked into a mutating operation." The DSL approach
instead makes the unsafe class of query **inexpressible in the model's own output schema**: the
LLM never produces Cypher text at all. Its structured output can only ever populate a small set of
fields (which label, which property, which comparison operator, which literal value) that a
hand-written, five-line compiler turns into one of a fixed handful of clause shapes
(`MATCH`/`WHERE`/`RETURN`/`ORDER BY`/`LIMIT`) — there is no code path through which any string the
model produces becomes a Cypher **keyword**; a model's attempt to inject one (e.g. supplying
`"label": "Product) DETACH DELETE (x"`) is validated as an **identifier** against a closed,
per-dataset allowlist (§3.3) and rejected outright, never concatenated into the query text. This is
a strictly smaller, more mechanically-auditable surface than parsing arbitrary Cypher text well
enough to trust — the right trade for a capability whose next reviewer is specifically checking
"can this be talked into doing something destructive."

## 3. Design & rationale

### 3.1 The mechanism, end to end

1. The combined demo agent's `agent` step offers a new tool, `query_graph_data`
   (`{question: string, dataset: enum}` — §3.4), to the model, alongside every other tool the
   sibling plans have already granted.
2. The model's function-call **arguments are not the answer** — they select a **dataset** and
   restate the question; the tool itself makes a **second, internal, non-agent-loop LLM call**
   (via the same `ModelGateway` `step` kind `_run_agent_node` already resolves,
   `executor.py:702-707`) whose **only** allowed output shape is the constrained DSL below,
   requested as a structured tool-call/JSON-schema completion, not free text — mirroring how
   `extraction.py`'s structured-output extraction call already works for a different feature
   (`document-ingestion.md` §3.3's `extract_own_line_json_object` pattern), reused here for the
   same reason: a fence-tolerant, schema-constrained parse of a small local model's structured
   output is a solved, proven problem in this codebase, not a new one to re-invent.
3. The **DSL** (Pydantic-validated immediately on receipt, before any compilation step runs).
   **Revised per `security-expert`'s MAJOR 1/MAJOR 2 and MINOR 2 findings**
   (`docs/reviews/workflow-nl-query-generation-security.md`): `var`'s pattern is now an
   **enforced field constraint**, not a comment (MAJOR 2); `returns`/`order_by` are validated by
   **fully-anchored** (`re.fullmatch`, never `re.match`/`re.search`) grammar regexes rather than
   left as an unspecified "`\"var.property\"`-shaped string" (MAJOR 1); every model carries
   `extra="forbid"` so an unexpected/smuggled field (e.g. a `"raw_cypher"` key) fails parsing
   loudly instead of being silently ignored (MINOR 2):

```python
# server/falkorchat/querygen.py — new module, pure (no I/O; takes a DatasetSchema + a
# validated QueryRequest, returns a CompiledQuery or raises)

_VAR_RE = r"[a-z][a-z0-9]{0,7}"                       # the one shared identifier grammar —
_PROP_RE = r"[a-z][a-zA-Z0-9]{0,31}"                  # var/property patterns are defined ONCE
                                                        # here and reused by every field below,
                                                        # never redefined per-field
_PROJECTION_RE = re.compile(rf"^({_VAR_RE})\.({_PROP_RE})$")
_AGGREGATE_RE = re.compile(
    rf"^(count|avg|min|max)\(({_VAR_RE})(?:\.({_PROP_RE}))?\)$"
)

class QueryFilter(BaseModel):
    model_config = ConfigDict(extra="forbid")
    property: str = Field(pattern=rf"^{_PROP_RE}$")
    op: Literal["=", "<>", "<", "<=", ">", ">="]   # a closed, six-op whitelist — no `contains`/
                                                     # regex ops in v1 (keeps every value a bound
                                                     # scalar param, never a pattern string)
    value: str | float | int | bool

class QueryMatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # MAJOR 2 fix: an ENFORCED Pydantic constraint, not a comment. `var` has no per-dataset
    # registry to allowlist against (any short lowercase token is a priori a legal Cypher
    # identifier) — unlike `label`/`property`, this regex IS the entire safety property for
    # this field, so it must be wired up, not merely documented.
    var: str = Field(pattern=rf"^{_VAR_RE}$")
    label: str          # validated against DatasetSchema.labels (§3.3) — exact match, reject
                         # anything else; never coerced/fuzzy-matched
    filters: list[QueryFilter] = Field(default_factory=list, max_length=4)

class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset: str
    matches: list[QueryMatch] = Field(min_length=1, max_length=1)   # v1: single-label match only
                                                                      # — no relationship traversal
                                                                      # (§3.6)
    # MAJOR 1 fix: `returns`/`order_by` are compound expressions (`"var.property"` or
    # `"count(var[.property])"`), not bare tokens like `label`/`property`/`var` — they cannot be
    # allowlisted directly, they must be DECOMPOSED first, and the decomposition regex must be
    # fully anchored (`_PROJECTION_RE`/`_AGGREGATE_RE` above both open with `^` and close with
    # `$` — `re.fullmatch` semantics even when called via `.match()`, so a string that merely
    # *starts with* a valid shape but carries trailing garbage — e.g.
    # `"count(v)) DETACH DELETE (v) //"` — is rejected whole, never truncated to its matching
    # prefix). Field-level regex only checks *shape*; `compile()` (below) additionally runs each
    # decomposed `var`/`property` through the **same** `DatasetSchema` allowlist check
    # `QueryFilter.property`/`QueryMatch.label` use — never a second, independently-written check.
    returns: list[str] = Field(min_length=1, max_length=6)
    order_by: str | None = None
    order_dir: Literal["ASC", "DESC"] = "ASC"
    limit: int = Field(default=20, ge=1, le=50)

    @field_validator("returns")
    @classmethod
    def _returns_shape(cls, values: list[str]) -> list[str]:
        for v in values:
            if not (_PROJECTION_RE.fullmatch(v) or _AGGREGATE_RE.fullmatch(v)):
                raise ValueError(f"returns entry {v!r} does not match a projection or aggregate shape")
        return values

    @field_validator("order_by")
    @classmethod
    def _order_by_shape(cls, v: str | None) -> str | None:
        if v is not None and not _PROJECTION_RE.fullmatch(v):
            raise ValueError(f"order_by {v!r} must be a bare projection (\"var.property\")")
        return v
```

4. **`querygen.compile(request, schema) -> CompiledQuery`** (MINOR 2: a **nominal frozen
   dataclass**, `CompiledQuery(cypher: str, params: dict)` — no public constructor outside this
   module — replacing the plain `tuple[str, dict]` this plan originally sketched; a bare tuple
   gave `run_readonly_query` no type-level way to distinguish "the compiler's own output" from
   "any string a future caller assembled by hand," which is exactly the residual §6 already
   flagged). `compile` is a hand-written, fixed-template function. It never string-formats a
   value into the query text (every `QueryFilter.value` and `limit` becomes a bound `$pN`
   parameter, per `falkor-chat/AGENTS.md` rule 1); it validates every `label`/`property`/`var`
   entry — **including each `var`/`property` decomposed out of `returns`/`order_by` by the
   `field_validator`s above** — against `schema` (§3.3) with an **exact-match allowlist check** —
   not a sanitizer, not an escape function, a hard reject on anything not already a known-good
   identifier for that dataset — before splicing it into the query text as an identifier (Cypher
   has no way to parameterize a label/property name; this is the one place model-influenced text
   becomes part of the query string, and it is closed by allowlist, not by blocking a "bad"
   pattern). Per MAJOR 2's second half, `compile` also **independently re-checks** that every
   `var` referenced anywhere in `matches`/`returns`/`order_by`/`filters` resolves to the one
   declared match variable (v1 has exactly one) — a defense-in-depth check against a future
   refactor constructing a `QueryMatch`/`QueryRequest` by some path other than the public,
   validated schema (mirrors this plan's own "two independent layers" philosophy, applied one
   level down, per the reviewer's suggestion). The compiler's own code has **no branch that can
   ever emit** `CREATE`, `MERGE`, `SET`, `DELETE`, `REMOVE`, `DROP`, `FOREACH`, or `CALL` — those
   tokens do not appear anywhere in the compiler's template strings, so no value the model
   supplies can produce them: this is what "structural," not "filtered," means for FR-3.
5. **`repository.run_readonly_query(graph_key, compiled: CompiledQuery, *, timeout: int =
   DEFAULT_QUERY_TIMEOUT_MS) -> list[dict]`** — one new, generic method, typed to accept a
   `CompiledQuery` (not a bare `(str, dict)`) so "only `querygen.compile`'s output ever reaches
   this method" is a type-checker-visible constraint, not only a `grep`-style code-review gate
   (MINOR 2) — the grep-style test (§5) stays as a belt-and-suspenders regression check, not the
   sole enforcement. `DEFAULT_QUERY_TIMEOUT_MS = 2500` (MINOR 1: a concrete, conservative default
   — `claude/graph-dba/falkordb-quirks.md` documents this build's read-`TIMEOUT` enforcement as
   **batch-granular, not a hard per-query cap** ("slightly-over queries can slip through"), so
   this value is a deliberate safety margin, not treated as an exact ceiling; the module docstring
   states this caveat explicitly so nobody downstream mistakes it for a hard guarantee). It calls
   `graph.ro_query(compiled.cypher, compiled.params, timeout=timeout)` — **always** `ro_query`,
   never `query` — which is the independent, engine-level backstop from §2.2: even a defect in
   `querygen.compile` that somehow let a mutating clause through would still be refused by
   FalkorDB itself before any write occurs, because the app never has a code path that calls
   `.query()` on this generated text.
6. The tool formats the rows into the same abstention-or-answer shape every other lookup tool in
   this codebase uses (`{"items": [...]}`/`{"items": [], "finding": "no matching data found"}`,
   mirroring `FilterProductsTool`/`GraphragRetrieveTool`), which the **outer** agent-loop LLM call
   (step 1's routing turn) then turns into a natural-language reply via `post_message`, exactly
   like every other tool result in this codebase.

### 3.2 Two independent layers — reviewed by `security-expert`

- **Layer 1 (primary, structural):** the model's output can only ever populate `QueryFilter`/
  `QueryMatch`/`QueryRequest` fields; the compiler's fixed template set has no code path to a
  mutating clause. This holds even if the model is "talked into" wanting to delete data — there is
  no field in its output schema that could express that intent as anything other than a rejected
  or inert string value. **As originally specified, this layer had two gaps** (`returns`/
  `order_by`'s unspecified decomposition; `var`'s regex stated only as a comment) —
  `security-expert`'s review (`docs/reviews/workflow-nl-query-generation-security.md`, MAJOR 1/2)
  found both and independently proved the underlying risk was real (constructing a syntactically
  valid single-statement Cypher injection completing this section's own §2.3 example — a `MATCH
  ... DETACH DELETE v WITH 1 AS ignore RETURN ...` chain that parses cleanly on this engine). Both
  are closed in §3.1 above (the `field_validator`s + enforced `var` pattern + `compile()`'s
  independent re-check).
- **Layer 2 (backstop, engine-enforced, independent of layer 1's correctness):** every execution
  goes through `graph.ro_query(...)`, which FalkorDB itself refuses to run if it is ever, somehow,
  a write (§2.2). This layer requires **zero trust** in `querygen.compile`'s own correctness — it
  is a property of the database, not of this plan's code. `security-expert` independently
  re-verified this live against the pinned build, including against the exact injection shape
  above and a multi-statement-smuggle attempt (both refused) — this is not a claim taken on faith
  from `falkordb-quirks.md`, it has now been reproduced twice.

Both layers are cheap (layer 2 is the same call every read in this codebase already makes) and
independent (a defect in one does not weaken the other) — the reviewer's own framing, worth
repeating exactly: Layer 2 is what actually saves this design today even where Layer 1 had a gap,
and this section's revision is about making Layer 1 as strong as the design already claimed for
it, not a redesign. `security-expert`'s Groups A-E adversarial set (the requirements doc's FR-3a
bar) attacks **both** layers deliberately: Group A (prompt injection against the internal
structured-completion call) and Group B (direct field-injection against `querygen.compile`, using
the exact live-verified injection shape above) target Layer 1; Group C (a static/grep check plus a
direct hand-crafted-string probe bypassing `querygen` entirely) targets Layer 2 independent of
Layer 1's correctness; Group D covers the malformed/abstention path; Group E is a resource-
exhaustion class the reviewer explicitly tracks separately from FR-3/FR-3a's mutation focus (§5).

### 3.3 Dataset schema registry (FR-2's generality)

A **declarative, per-dataset registry** — not live schema introspection via `db.labels()`/
`db.propertyKeys()` — because this plan needs a reliable **per-label property allowlist**
(which properties exist on which label), and while `db.labels()`/`db.relationshipTypes()` are
live-verified to work on this build (`falkordb-quirks.md`), no per-label property-enumeration
procedure has been live-verified here, and this mechanism's safety argument (§3.2) depends on the
allowlist being trustworthy at the point a query compiles, not on a live introspection call that
could itself return an incomplete or stale set under real-world timing. A declarative registry is
also strictly sufficient for FR-2's actual bar — "generalizes... without being rebuilt specifically
for that schema" means *registering* a new dataset (data), not modifying the compiler/validator
(code):

```python
# server/falkorchat/querygen.py (continued)

@dataclass(frozen=True)
class DatasetSchema:
    graph_key: str                              # "reference" or f"ws:{ws}" — resolved per call
    labels: dict[str, frozenset[str]]           # label -> allowed property names
    aggregates: frozenset[str] = frozenset({"count", "avg", "min", "max"})  # shared, not per-dataset

CATALOG_SCHEMA = DatasetSchema(
    graph_key="reference",
    labels={"Product": frozenset({"name", "nameNormalized", "category", "price"})},
)
KNOWLEDGE_BASE_SCHEMA = DatasetSchema(
    graph_key=None,  # resolved to f"ws:{ctx.ws}" at call time — this dataset is workspace-scoped
    labels={
        "Entity": frozenset({"entityId", "name", "nameNormalized", "type"}),
        "Document": frozenset({"documentId", "title", "sourceFormat"}),
        "Chunk": frozenset({"chunkId", "text", "seq", "documentId"}),
    },
)
DATASET_REGISTRY = {"catalog": CATALOG_SCHEMA, "knowledge_base": KNOWLEDGE_BASE_SCHEMA}
```

The tool's own JSON-schema `description` (offered to the model, §3.4) is generated from this
registry at `ToolRegistry` construction time — one per-dataset paragraph naming its labels and
properties — so adding a third dataset later is: add a `DatasetSchema` entry, extend the tool's
description, done; `querygen.compile` and every validation rule are untouched.

### 3.4 The `QueryGraphDataTool`

```python
class QueryGraphDataTool:
    name = "query_graph_data"
    # schema: {"question": {"type": "string"}, "dataset": {"type": "string",
    #   "enum": list(DATASET_REGISTRY)}}, required: ["question", "dataset"]; description
    # includes the per-dataset label/property rundown from §3.3.
    def run(self, arguments, *, ctx, run) -> str:
        schema = DATASET_REGISTRY.get(arguments.get("dataset"))
        if schema is None:
            return json.dumps({"items": [], "finding": "unknown dataset"})
        request = self._models.resolve_llm("step", ws=ctx.ws) \
            ...structured completion request for QueryRequest, parsed via the same
            fence-tolerant helper `extraction.py` already proved (§3.1 step 2)...
        compiled = querygen.compile(request, schema)   # CompiledQuery, not a bare tuple (MINOR 2)
        graph_key = schema.graph_key or f"ws:{ctx.ws}"
        rows = self._services.run_structured_query(
            graph_key, compiled, timeout=querygen.DEFAULT_QUERY_TIMEOUT_MS  # MINOR 1
        )
        if not rows:
            return json.dumps({"items": [], "finding": "no matching data found"})
        return json.dumps({"items": rows})
```

A malformed/unparseable structured completion (the DS's fence/prose-tolerant parse fails, or
Pydantic validation rejects the fields) returns the same "no matching data found" abstention —
never a fabricated answer, satisfying the same FR-4-style abstention discipline every other lookup
tool in this set already follows, and directly analogous to `evaluate_guard`'s own "bias to
decline, never crash" posture.

### 3.5 AC-2 — the second dataset, concretely

Per the coordinator's brief, `document-ingestion.md`'s `Document`/`Chunk`/`Entity` schema is
"shipped and stable" (M5 closed 2026-08-25) — a real, already-populated second schema, not a
synthetic one invented for this AC. §3.3's `KNOWLEDGE_BASE_SCHEMA` registers it directly. AC-2's
verification: seed at least one document via the existing `ingest_document` MCP/REST path
(`document-ingestion.md` §3.5), then ask the demo agent a question answerable via a single-label
filter against that data without relationship traversal — e.g. "what type of entity is
`<extracted-entity-name>`" compiles to `MATCH (e:Entity) WHERE e.nameNormalized = $p0 RETURN
e.type` — proving the **same** `query_graph_data` tool, `querygen.compile`, and
`run_structured_query` path answers a question against a schema this plan's code has never seen
before, with zero dataset-specific code (only the declarative registry entry, §3.3).

### 3.6 Deliberate v1 non-goal: no relationship-pattern traversal

`QueryMatch` (§3.1) supports exactly one node pattern per request — no `MATCH (a)-[:REL]->(b)`.
This is a genuine, stated scope decision, not an oversight: every AC-1/AC-2 example this plan
needs to prove (a fact lookup, a filter, a cross-schema fact lookup) is answerable with a
single-label filter; adding relationship-pattern support would grow the compiler's template
surface (a new relationship-type allowlist per dataset, direction handling, multi-hop depth caps)
for a capability whose most important property is a *small, fully auditable* grammar (§2.3's own
reasoning). Flagged as a natural, explicitly out-of-scope v2 extension — mirroring
`document-ingestion-ml.md`'s own "embedding-based semantic matching deferred to a scoped v2, not a
v1 precondition" pattern — not something this plan quietly under-builds and hopes nobody notices.

## 4. Step-by-step implementation

Builds on `docs/plans/workflow-catalog-lookup.md`'s scaffold; sequence after
`docs/plans/workflow-durable-profile.md`'s `salesperson@v3` (this plan bumps to `v4`), or in
parallel if the coordinator prefers — same shared-file caveat as the profile plan's §6.

1. **`server/falkorchat/querygen.py`** — `QueryFilter`/`QueryMatch`/`QueryRequest` (Pydantic, all
   `extra="forbid"`), `_PROJECTION_RE`/`_AGGREGATE_RE` (the anchored `returns`/`order_by`
   decomposition grammar) plus their `field_validator`s, `CompiledQuery` (frozen dataclass),
   `DEFAULT_QUERY_TIMEOUT_MS`, `DatasetSchema`/`CATALOG_SCHEMA`/`KNOWLEDGE_BASE_SCHEMA`/
   `DATASET_REGISTRY`, `compile(...) -> CompiledQuery` (§3.1/§3.3, as revised per
   `security-expert`'s review). Pure module except for the registry's `graph_key` resolution —
   unit-testable with no FalkorDB/LLM dependency (feed it hand-built `QueryRequest`s, assert the
   exact `CompiledQuery.cypher`/`.params` produced, and assert every disallowed field value raises
   before any string is built — including the reviewer's specific escape-attempt fixtures, §5).
2. **`server/falkorchat/repository.py`** — `run_readonly_query(graph_key, compiled: CompiledQuery,
   *, timeout: int = querygen.DEFAULT_QUERY_TIMEOUT_MS) -> list[dict]` (§3.1 step 5) — the
   **only** new repository method in this whole four-document effort that takes compiler-produced
   Cypher rather than a query 1:1-mapped to `QUERIES.md`; its signature itself (a `CompiledQuery`
   parameter, not `cypher: str, params: dict`) is now part of the "only `querygen.compile` calls
   this" enforcement, not just a docstring comment.
3. **`server/falkorchat/services.py`** — `run_structured_query(ctx, graph_key, compiled:
   CompiledQuery, *, timeout: int | None = None) -> list[dict]`, a thin pass-through (no
   additional logic — the safety argument is entirely in `querygen`+`repository`, not in a
   services-layer check that could be bypassed by a future direct caller).
4. **`server/falkorchat/tools.py`** — `QueryGraphDataTool` (§3.4); register into the salesperson
   registry.
5. **Wait for `data-scientist`'s `workflow-nl-query-generation-ml.md`** — the structured-
   completion prompt design and the golden-set methodology/threshold (FR-4). This plan's steps
   1-4 do not depend on it (the mechanism is prompt-agnostic — any structured-output prompt that
   fills the `QueryRequest` schema works); step 6 does.
6. **Golden-set harness** — per `data-scientist`'s note, wiring TBD there.
7. **`server/falkorchat/proof_defs.py`** — bump `SALESPERSON_DEF["version"]` to `"v4"`, add
   `"query_graph_data"` to `config.tools`, extend `systemPrompt` ("for a question that doesn't
   match a specific catalog lookup, use `query_graph_data` instead of guessing").
8. **`scripts/seed_salesperson.sh`** / **`scripts/verify_salesperson.sh`** — publish/materialize/
   verify `salesperson@v4`.
9. **Run `security-expert`'s FR-3a adversarial test-case set** (Groups A-E,
   `docs/reviews/workflow-nl-query-generation-security.md`, already designed against this
   mechanism as specified) once `query_graph_data` exists — Group C's static/grep + direct-probe
   checks can run as soon as step 1-2 land, ahead of the live model-dependent Group A cases.

**Done (this plan, pending the two delegated notes):** `salesperson@v4` answers at least one
arbitrarily-phrased catalog question outside `workflow-catalog-lookup.md`'s fixed shapes (AC-5)
and one question against the ingestion knowledge-base dataset (AC-2); the golden-set evaluation
passes `data-scientist`'s bar (AC-4); `security-expert`'s Groups A-E adversarial set passes
(AC-3/AC-3a).

## 5. Test strategy

| AC | What proves it | Altitude |
|---|---|---|
| AC-1 (arbitrary phrasing, two structurally different questions) | Two live `@mention` questions against the catalog dataset that don't match `workflow-catalog-lookup.md`'s fixed shapes — one a filter, one an aggregate (`count`) — both answered correctly | live e2e |
| AC-2 (second dataset) | §3.5's concrete scenario against the seeded ingestion knowledge base | live e2e |
| AC-3, AC-3a (structural non-mutation, adversarial) | **Structural tests this plan specifies directly:** (a) `querygen.compile` raises/rejects for every hand-constructed `QueryRequest` whose `label`/`property` is not in the target `DatasetSchema` — never silently drops or coerces it; (b) a static/code-review check (or a `grep`-based test, mirroring `test_modelconfig.py`'s AST-check discipline) asserts `repository.run_readonly_query` is called **only** from `querygen`-compiled call sites and **always** via `.ro_query(...)`, never `.query(...)`/`.profile(...)` — reinforced, not replaced, by `run_readonly_query`'s own `CompiledQuery`-typed signature (§3.1 step 5); (c) a live probe confirms `graph.ro_query()` does in fact refuse a hand-crafted write string against this deployment's pinned build (re-verifying `falkordb-quirks.md`'s cited fact rather than trusting the doc blindly — `security-expert` already re-ran this live, review §"Live-verified evidence" points 1-2); (d) **MAJOR 1 regression tests:** `returns`/`order_by` values that merely *start with* a valid shape but carry trailing garbage (`"count(v)) DETACH DELETE (v) //"`, `"v.name//anything"`) are rejected whole, never truncated to a matching prefix; (e) **MAJOR 2 regression tests:** `QueryMatch.var` rejects the reviewer's escape-attempt fixtures (`"x) DETACH DELETE (x"`, `"x WITH 1 AS y MATCH (m) DETACH DELETE m"`, a `var` containing `//`) at the Pydantic layer, and `compile()` independently rejects a hand-built `QueryMatch`/`QueryRequest` whose `var` references disagree with the one declared match variable, bypassing Pydantic entirely — **then** `security-expert`'s full Groups A-E adversarial set (`docs/reviews/workflow-nl-query-generation-security.md`), designed and run against the actual mechanism, asserting **zero rows created/modified/deleted** as the pass condition per Group B/C, not merely "the tool returned an error" | unit (a, d, e) + static/grep + type-signature check (b) + one live probe (c, already independently reproduced) + `security-expert`'s Groups A-E pass |
| AC-4 (golden-set passes) | Per `data-scientist`'s methodology note | per that note |
| AC-5 (live proof) | At least one arbitrarily-phrased question correctly answered inside `salesperson@v4` | live e2e (overlaps AC-1's own proof) |

**Additional, non-AC-mapped coverage (from the security review, tracked separately from FR-3/FR-3a's mutation focus per the reviewer's own Group E framing):** a resource-exhaustion check confirming `DEFAULT_QUERY_TIMEOUT_MS` (or an explicit override) actually bounds a `limit=50`, zero-filter query's wall time to a clean, bounded-time error rather than a long hang (MINOR 1); a confirmation that `limit<=50` plus FalkorDB's own `RESULTSET_SIZE` default (10,000 rows, `falkordb-quirks.md`) together close the result-set-size class regardless of underlying data volume. These gate a resource-exhaustion concern, not FR-3a's own pass/fail.

## 6. Risks & open questions

- **Reviewed by `security-expert`; verdict approve with suggestions, no blocker**
  (`docs/reviews/workflow-nl-query-generation-security.md`). Two MAJOR findings (§3.1's
  `returns`/`order_by` decomposition, `QueryMatch.var`'s enforcement) and two MINORs (a
  conservative `run_readonly_query` timeout, `extra="forbid"` + a `CompiledQuery` nominal type)
  are closed in this revision. The reviewer's own framing: Layer 2 (`GRAPH.RO_QUERY`'s
  independent, engine-level write-refusal, live-reproduced by the reviewer including the exact
  single-statement injection shape §2.3/§3.1 step 4 discuss) is what actually saves this design
  even where Layer 1 had a gap — this revision brings Layer 1 up to the strength the design
  already claimed for it, it is not a redesign. A `security-expert` confirmation pass (not a
  fresh full review) is expected before the `analyst` plan gate.
- **Open from the review, for the next reader to weigh in on:** (a) whether `qa-engineer`/the
  implementer runs Group A (prompt-injection against the live model) against the real configured
  model or a scripted stub — either is fine per the reviewer, but the harness should say which and
  re-run Group A on any model-config change, mirroring `data-scientist`'s existing "re-verify
  small-model behavior on change" posture; (b) whether the `CompiledQuery` nominal-type wrapper
  (adopted in this revision) or the grep-test alone would have been preferred — the reviewer
  recommended adopting it, and this revision does, closing that question in the affirmative.
- **The structured-completion prompt for filling `QueryRequest` is not designed here** — deferred
  to `data-scientist`'s note per the coordinator's delegation; this plan's mechanism is agnostic to
  that prompt's exact wording (any prompt that reliably fills the schema works), but a poorly-
  designed prompt could still produce a *valid-but-wrong* `QueryRequest` (e.g., the wrong label) —
  a correctness risk the golden-set evaluation (AC-4) exists to catch, distinct from the safety
  argument in §3.2.
- **No per-label property-enumeration procedure has been live-verified on this build** (§3.3) —
  this plan's declarative-registry choice is partly a hedge against that uncertainty. If a future
  reviewer live-verifies `db.propertyKeys()`-style introspection works reliably here, a live-
  introspected registry could replace the declarative one without changing `querygen.compile`'s
  interface — noted as a possible future simplification, not pursued now.
- **`run_readonly_query`'s generality is itself a residual risk surface** — it is the one place in
  this whole codebase that accepts compiler-built Cypher rather than a query 1:1-mapped to
  `QUERIES.md`. This plan mitigates it two ways now: the `CompiledQuery`-typed signature (a
  type-checker-visible constraint, this revision) and the `grep`-style regression test (§5b) — but
  a future, unrelated feature that constructs its own `CompiledQuery` by some path other than
  `querygen.compile` (nothing stops a determined caller from instantiating the dataclass directly
  unless it is also given a private/module-internal constructor convention) would reopen the exact
  risk this plan closed. Flag this method's narrow intended use in its own docstring loudly enough
  that a future implementer reads it before reusing it; a stricter close (e.g. a module-private
  sentinel `CompiledQuery` can only be constructed with) is available if a future review wants it,
  not adopted here as it would add ceremony beyond what MINOR 2 asked for.
- **`_PROJECTION_RE`/`_AGGREGATE_RE`'s property-name character class (`_PROP_RE`) allows
  mixed-case** (`[a-zA-Z0-9]`) while `_VAR_RE` is lowercase-only — a deliberate asymmetry (this
  codebase's own property names are `camelCase`, e.g. `nameNormalized`, so a lowercase-only
  property regex would reject legitimate allowlisted properties), not an oversight, but worth
  stating explicitly since it is the one place the two identifier grammars in this module
  intentionally differ.
