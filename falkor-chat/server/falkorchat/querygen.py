"""K-055 (M6) — constrained natural-language-query DSL and compiler.

`docs/plans/workflow-nl-query-generation.md` §2.2-§3.3 is the source design; this
module implements plan step 1 of §4's step table only (`QueryFilter`/`QueryMatch`/
`QueryRequest`, the `returns`/`order_by` decomposition grammar, `CompiledQuery`,
`DatasetSchema`/the dataset registry, and `compile()`). It is a **pure module** —
no I/O, no FalkorDB/LLM dependency — so every function here is unit-testable by
feeding it hand-built objects.

**Why this module exists (FR-3's "structurally incapable", not "filtered"):**
the LLM never produces Cypher text. Its structured output can only ever populate
`QueryFilter`/`QueryMatch`/`QueryRequest` fields, which `compile()` turns into one
of a fixed handful of clause shapes (`MATCH`/`WHERE`/`RETURN`/`ORDER BY`/`LIMIT`).
There is no code path through which model-supplied text becomes a Cypher
**keyword** — `CREATE`, `MERGE`, `SET`, `DELETE`, `REMOVE`, `DROP`, `FOREACH`, and
`CALL` do not appear anywhere in this module's template strings, so no value a
caller supplies can produce them. Every value (`QueryFilter.value`, `limit`)
becomes a bound `$pN`/`$limit` parameter — never string-formatted into the query
text (`falkor-chat/AGENTS.md` rule 1). The one place model-influenced text *does*
become part of the query string is a `label`/`property`/`var` identifier (Cypher
has no way to parameterize those) — each is checked against `DatasetSchema` (or,
for `var`, an enforced Pydantic field pattern) with an **exact-match allowlist**,
a hard reject on anything unknown, never a sanitizer or an escape function.

This is Layer 1 of the plan's two-layer design. Layer 2 — every execution goes
through FalkorDB's `GRAPH.RO_QUERY`, which the engine itself refuses to run if it
is ever, somehow, a write (§2.2) — is independent of this module's correctness
and lives in `repository.run_readonly_query` (a later cluster), not here.

**`DEFAULT_QUERY_TIMEOUT_MS` is a safety margin, not an exact ceiling.** This
deployment's pinned FalkorDB build enforces read `TIMEOUT` at **batch
granularity, not as a hard per-query cap** (`claude/graph-dba/falkordb-quirks.md`)
— a slightly-over-budget query can still slip through before the next batch
boundary is checked. Callers must not treat this value as a guarantee that a
query will be aborted at exactly this many milliseconds.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .extraction import normalize_name

# ── The one shared identifier grammar — defined ONCE, reused everywhere ──────
# `_VAR_RE` is deliberately lowercase-only (this codebase's Cypher variables are
# short lowercase tokens, e.g. `p`, `e`); `_PROP_RE` deliberately allows mixed
# case because this codebase's own property names are `camelCase`
# (`nameNormalized`) — a lowercase-only property regex would reject legitimate
# allowlisted properties. This asymmetry is intentional, not an oversight.
_VAR_RE = r"[a-z][a-z0-9]{0,7}"
_PROP_RE = r"[a-z][a-zA-Z0-9]{0,31}"

# Fully anchored (`^...$`) — `re.fullmatch` semantics even if a call site ever
# used `.match()` by mistake. This closes the security review's MAJOR 1 finding:
# an unanchored decomposition regex would let a string like
# `"count(v)) DETACH DELETE (v) //"` pass as a valid prefix match. Every call
# site below uses `.fullmatch()` explicitly — never `.match()`/`.search()`.
_PROJECTION_RE = re.compile(rf"^({_VAR_RE})\.({_PROP_RE})$")
_AGGREGATE_RE = re.compile(
    rf"^(count|avg|min|max)\(({_VAR_RE})(?:\.({_PROP_RE}))?\)$"
)

# Bare-token identifier regexes (`label`/`op`/`var` are single tokens, never
# compound expressions — no decomposition needed, just a shape check before the
# schema-allowlist check in `compile()`).
_VAR_FULL_RE = re.compile(rf"^{_VAR_RE}$")
_PROP_FULL_RE = re.compile(rf"^{_PROP_RE}$")


class QueryFilter(BaseModel):
    """One `WHERE var.property OP $pN` clause. `value` is always bound as a
    parameter, never spliced into the query text."""

    model_config = ConfigDict(extra="forbid")

    property: str = Field(pattern=rf"^{_PROP_RE}$")
    # A closed, six-op whitelist — no `contains`/regex ops in v1 (keeps every
    # value a bound scalar param, never a pattern string).
    op: Literal["=", "<>", "<", "<=", ">", ">="]
    value: str | float | int | bool


class QueryMatch(BaseModel):
    """The single `MATCH (var:label)` pattern v1 supports (§3.6 — no
    relationship traversal)."""

    model_config = ConfigDict(extra="forbid")

    # MAJOR 2 fix: an ENFORCED Pydantic constraint, not a comment. `var` has no
    # per-dataset registry to allowlist against (any short lowercase token is a
    # priori a legal Cypher identifier) — unlike `label`/`property`, this regex
    # IS the entire safety property for this field.
    var: str = Field(pattern=rf"^{_VAR_RE}$")
    label: str  # validated against DatasetSchema.labels in compile() — exact
    # match, reject anything else; never coerced/fuzzy-matched.
    filters: list[QueryFilter] = Field(default_factory=list, max_length=4)


class QueryRequest(BaseModel):
    """The top-level structured-completion output the model's second, internal
    LLM call must produce (§3.1 step 2/3)."""

    model_config = ConfigDict(extra="forbid")

    dataset: str
    matches: list[QueryMatch] = Field(min_length=1, max_length=1)  # v1: single-
    # label match only — no relationship traversal (§3.6).
    # `returns`/`order_by` are compound expressions (`"var.property"` or
    # `"count(var[.property])"`), not bare tokens — they cannot be allowlisted
    # directly, they must be decomposed first (the validators below), and
    # `compile()` independently re-runs the same decomposition + allowlist
    # check rather than trusting that this validator ran.
    returns: list[str] = Field(min_length=1, max_length=6)
    order_by: str | None = None
    order_dir: Literal["ASC", "DESC"] = "ASC"
    limit: int = Field(default=20, ge=1, le=50)

    @field_validator("returns")
    @classmethod
    def _returns_shape(cls, values: list[str]) -> list[str]:
        for v in values:
            if not (_PROJECTION_RE.fullmatch(v) or _AGGREGATE_RE.fullmatch(v)):
                raise ValueError(
                    f"returns entry {v!r} does not match a projection or aggregate shape"
                )
        return values

    @field_validator("order_by")
    @classmethod
    def _order_by_shape(cls, v: str | None) -> str | None:
        if v is not None and not _PROJECTION_RE.fullmatch(v):
            raise ValueError(f"order_by {v!r} must be a bare projection (\"var.property\")")
        return v


@dataclass(frozen=True)
class CompiledQuery:
    """The compiler's own output — a nominal type, not a bare `tuple[str,
    dict]`, so a type checker (and a future reader) can distinguish "the
    compiler's own output" from any string a future caller assembled by hand.
    Only `compile()` in this module should construct one."""

    cypher: str
    params: dict[str, object]


# A conservative safety margin, not an exact ceiling — see the module docstring.
DEFAULT_QUERY_TIMEOUT_MS = 2500


@dataclass(frozen=True)
class DatasetSchema:
    """A declarative, per-dataset registry — not live schema introspection.
    Adding a new dataset is: add a `DatasetSchema` entry (data), not a compiler
    change (code) — see the plan §3.3 for why this is a deliberate choice over
    `db.labels()`/`db.propertyKeys()` introspection.

    `labels` maps each allowed property name to its declared Python type
    (U29f fix A, `docs/reviews/workflow-nl-query-generation-rca.md` §4.1) —
    not just an allowlist of names. `compile()` uses the declared type to
    coerce a numeric-looking *string* filter value (a JSON-serialization
    quirk some models produce, e.g. `"50"` instead of `50`) before binding
    it as a parameter, and to reject one that genuinely doesn't parse as
    that type. `dict.__contains__`/`in` against this mapping still checks
    keys only, so every existing "is this property registered" call site
    (`prop in allowed_props`) is unaffected by this shape change."""

    graph_key: str | None  # "reference", or None when resolved per-call to
    # f"ws:{ws}" (a workspace-scoped dataset, e.g. the knowledge base).
    labels: dict[str, dict[str, type]]  # label -> {property name: declared type}
    aggregates: frozenset[str] = field(
        default_factory=lambda: frozenset({"count", "avg", "min", "max"})
    )


# Matches the actual shipped `Product` node shape (K-052,
# `scripts/seed_catalog.sh`): `name`, `nameNormalized`, `category`, `price`
# (the seeded rows also carry `productId`/`categoryNormalized`, not exposed here
# — this registry is a curated query-facing allowlist, not every property that
# exists on the node).
CATALOG_SCHEMA = DatasetSchema(
    graph_key="reference",
    labels={
        "Product": {
            "name": str,
            "nameNormalized": str,
            "category": str,
            "price": float,
        },
    },
)

# Matches the actual shipped document-ingestion schema
# (`server/falkorchat/repository.py` `create_entity`/`create_document`/chunk
# writes): `Entity{entityId, name, nameNormalized, type}`,
# `Document{documentId, title, sourceFormat}`, `Chunk{chunkId, text, seq,
# documentId}`. Each node also carries other properties not exposed here
# (`Entity.createdAt`, `Document.text`/`sourceKind`/`status`/`pendingJobs`/
# `createdAt`) — again, a curated query-facing allowlist.
KNOWLEDGE_BASE_SCHEMA = DatasetSchema(
    graph_key=None,  # workspace-scoped — resolved to f"ws:{ctx.ws}" at call time
    labels={
        "Entity": {"entityId": str, "name": str, "nameNormalized": str, "type": str},
        "Document": {"documentId": str, "title": str, "sourceFormat": str},
        "Chunk": {"chunkId": str, "text": str, "seq": int, "documentId": str},
    },
)

DATASET_REGISTRY: dict[str, DatasetSchema] = {
    "catalog": CATALOG_SCHEMA,
    "knowledge_base": KNOWLEDGE_BASE_SCHEMA,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _resolve_expr(
    expr: str,
    *,
    declared_var: str,
    allowed_props: dict[str, type],
    schema: DatasetSchema,
    allow_aggregate: bool,
) -> tuple[str, bool]:
    """Decompose one `returns`/`order_by` entry and validate every piece it
    carries against the schema allowlist + the declared match variable —
    **independently** of whatever the Pydantic field validators already did,
    per the plan's "never trust a validated-once flag" defense-in-depth
    posture (a hand-built `QueryRequest` via `.model_construct()` never runs
    those validators at all).

    Returns `(expr, is_aggregate)`: the original expression string unchanged
    (it is safe to splice verbatim once every decomposed piece is confirmed
    allowlisted — the expression's *shape* was already constrained to exactly
    `var.property` or `func(var[.property])` by the fully-anchored regexes
    below) and whether it matched the aggregate shape (U29f fix C — `compile()`
    uses this to scope `DISTINCT` to non-aggregate projections only).
    """
    proj = _PROJECTION_RE.fullmatch(expr)
    if proj:
        var, prop = proj.group(1), proj.group(2)
        _require(var == declared_var, f"{expr!r} references unknown var {var!r}")
        _require(prop in allowed_props, f"{expr!r} references unknown property {prop!r}")
        return expr, False

    if allow_aggregate:
        agg = _AGGREGATE_RE.fullmatch(expr)
        if agg:
            func, var, prop = agg.group(1), agg.group(2), agg.group(3)
            _require(func in schema.aggregates, f"{expr!r} uses unknown aggregate {func!r}")
            _require(var == declared_var, f"{expr!r} references unknown var {var!r}")
            if prop is not None:
                _require(prop in allowed_props, f"{expr!r} references unknown property {prop!r}")
            return expr, True

    raise ValueError(f"{expr!r} does not resolve to an allowed projection or aggregate")


def compile(request: QueryRequest, schema: DatasetSchema) -> CompiledQuery:
    """Compile a validated `QueryRequest` into a `CompiledQuery` against
    `schema`. Raises `ValueError` on any field whose value is not already a
    known-good identifier for this dataset — never coerces, never silently
    drops. See the module docstring for the full safety argument.

    This function never string-formats a value into the query text: every
    `QueryFilter.value` and `limit` becomes a bound `$pN`/`$limit` parameter.
    It has no branch, anywhere, that can emit `CREATE`, `MERGE`, `SET`,
    `DELETE`, `REMOVE`, `DROP`, `FOREACH`, or `CALL` — those tokens do not
    appear in any template string this function builds from.
    """
    # Defense-in-depth: re-check the match count even though Pydantic's own
    # `min_length=1, max_length=1` constraint already enforces it — a
    # `.model_construct()`-built request bypasses that constraint entirely.
    _require(len(request.matches) == 1, "query_graph_data v1 supports exactly one match")
    match = request.matches[0]

    _require(
        bool(_VAR_FULL_RE.fullmatch(match.var)),
        f"match var {match.var!r} is not a valid identifier",
    )
    declared_var = match.var

    _require(
        match.label in schema.labels,
        f"label {match.label!r} is not registered for this dataset",
    )
    allowed_props = schema.labels[match.label]

    params: dict[str, object] = {}
    where_clauses: list[str] = []
    for i, filt in enumerate(match.filters):
        _require(
            bool(_PROP_FULL_RE.fullmatch(filt.property)),
            f"filter property {filt.property!r} is not a valid identifier",
        )
        _require(
            filt.property in allowed_props,
            f"property {filt.property!r} is not registered for label {match.label!r}",
        )
        value: object = filt.value
        if (
            isinstance(value, str)
            and filt.property.endswith("Normalized")
            and filt.op in ("=", "<>")
        ):
            # Fix B: enforce the existing `*Normalized` contract server-side
            # (DESIGN.md §5.1) — the model may supply the verbatim question
            # text instead of the lower-cased/whitespace-collapsed stored
            # value; normalize with the SAME shared helper that produced the
            # stored value (`extraction.normalize_name`), never a
            # second, independently-written normalizer.
            value = normalize_name(value)
        else:
            declared_type = allowed_props[filt.property]
            if isinstance(value, str) and declared_type in (int, float):
                # Fix A: the model sometimes serializes a numeric filter
                # value as a JSON string (`"50"` instead of `50`) — coerce by
                # the property's declared type rather than letting a
                # string-vs-numeric comparison silently match zero rows.
                # A value that genuinely doesn't parse is a compile-time
                # error, not a silent abstention.
                try:
                    value = declared_type(value)
                except ValueError as exc:
                    raise ValueError(
                        f"filter value {filt.value!r} for property "
                        f"{filt.property!r} could not be parsed as "
                        f"{declared_type.__name__}"
                    ) from exc
        param_name = f"p{i}"
        where_clauses.append(f"{declared_var}.{filt.property} {filt.op} ${param_name}")
        params[param_name] = value

    return_results = [
        _resolve_expr(
            r,
            declared_var=declared_var,
            allowed_props=allowed_props,
            schema=schema,
            allow_aggregate=True,
        )
        for r in request.returns
    ]
    return_exprs = [expr for expr, _ in return_results]
    # Fix C: DISTINCT only when NONE of `returns` is an aggregate expression —
    # applying it unconditionally would silently change today's correct
    # `count(...)`-style semantics (golden-set-verified: nlq-31/32/33 expect
    # the raw un-fused node count, not a distinct count).
    distinct = not any(is_aggregate for _, is_aggregate in return_results)

    order_expr: str | None = None
    if request.order_by is not None:
        order_expr, _ = _resolve_expr(
            request.order_by,
            declared_var=declared_var,
            allowed_props=allowed_props,
            schema=schema,
            allow_aggregate=False,
        )

    params["limit"] = request.limit

    clauses = [f"MATCH ({declared_var}:{match.label})"]
    if where_clauses:
        clauses.append("WHERE " + " AND ".join(where_clauses))

    needs_tuple_distinct = (
        distinct and order_expr is not None and order_expr not in return_exprs
    )
    if needs_tuple_distinct:
        # graph-dba (`claude/graph-dba/falkordb-quirks.md`, "distinct
        # projection, ordered by a column NOT in the projection" entry —
        # refines the earlier "RETURN DISTINCT ... ORDER BY ..." entry, does
        # not contradict it): the naive `RETURN DISTINCT <returns> ORDER BY
        # <order_by>` plans `Project -> Distinct -> Sort -> Limit` — DISTINCT
        # collapses to the RETURNed columns alone BEFORE Sort runs, so the
        # sort key surviving for a collapsed group is arbitrary and can
        # produce a flat-out WRONG answer, not just a non-deterministic one
        # (live-verified: same data, opposite creation order, one order
        # gives the correct min/max, the other doesn't). Rejecting this shape
        # outright (an earlier version of this fix) was also wrong: it is
        # the DSL's *only* way to express "the <projected column> of the row
        # achieving the min/max of <order_by column>" — a superlative
        # question ("which product is the cheapest?") with no aggregate
        # return. The confirmed fix instead dedups on the FULL tuple (every
        # column actually needed — both requested and the sort key) in the
        # same WITH that carries the ORDER BY/LIMIT, then RETURNs only the
        # originally-requested columns, re-aliased back to their original
        # expression text (`c0 AS \`p.name\``) so callers keep seeing the
        # same "column key = raw expression text" contract every other
        # compiled shape here guarantees (`repository.run_readonly_query`'s
        # own docstring — `querygen.compile` never aliases a RETURN
        # expression).
        aliases = [f"c{i}" for i in range(len(return_exprs))]
        order_alias = f"c{len(return_exprs)}"
        with_items = [
            f"{expr} AS {alias}" for expr, alias in zip(return_exprs, aliases)
        ]
        with_items.append(f"{order_expr} AS {order_alias}")
        clauses.append("WITH DISTINCT " + ", ".join(with_items))
        clauses.append(f"ORDER BY {order_alias} {request.order_dir}")
        clauses.append("LIMIT $limit")
        clauses.append(
            "RETURN " + ", ".join(
                f"{alias} AS `{expr}`" for expr, alias in zip(return_exprs, aliases)
            )
        )
    else:
        return_keyword = "RETURN DISTINCT " if distinct else "RETURN "
        clauses.append(return_keyword + ", ".join(return_exprs))
        if order_expr is not None:
            clauses.append(f"ORDER BY {order_expr} {request.order_dir}")
        clauses.append("LIMIT $limit")

    return CompiledQuery(cypher=" ".join(clauses), params=params)
