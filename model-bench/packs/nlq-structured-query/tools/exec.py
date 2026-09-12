"""The pack-local, stdlib-only validation + execution surface for `nlq-structured-query` (S4 spec
§5.2.2). Re-implements `querygen.py`'s two validation layers (§2.5) by hand — no pydantic import
anywhere in this file (plan §3.3's explicit prohibition on a pack module importing outside stdlib +
`modelbench.tooling`; this module needs neither). Loaded via `Pack.load_tool_module()` — a module,
not a class, entrypoint `compile_and_execute` (`pack.json`'s own `tools.entrypoint`).

This is the LARGER half of the pack (plan §3.8.3's own cost note): it is what makes a
malformed-spec, a schema-violation, and a wrong-but-well-formed answer distinguishable rather than
pooled into "wrong" the way production's own `QueryGraphDataTool.run()` pools them (both collapse
to the same `{"items": [], "finding": "no matching data found"}` abstention there — `model-bench`'s
own scorer needs the three kept apart, §2.5/§4.3).

**Design note beyond the spec's literal execution description (§5.2.2):** `compile_and_execute`
de-duplicates a non-aggregate `returns` projection the same way `querygen.compile`'s default
`RETURN DISTINCT` does — needed for correctness against `ws:nlq-eval`'s real un-fused duplicate
`Entity` rows (multiple raw nodes sharing one `nameNormalized`, e.g. nine `Marlowe Robotics` rows,
`S4 spec §2.5`/the real golden set's own `nlq-21` rationale): a single-fact question over that
dataset needs the duplicate rows collapsed to one before `nlq_scoring.score_pair`'s scalar rule
("exactly one row, exactly one column") can score it at all. An aggregate `returns` (`count`/`avg`/
`min`/`max`) is never deduplicated first, matching `querygen.compile`'s own "DISTINCT only when
NONE of `returns` is an aggregate" rule (confirmed against the real golden set: `nlq-31`'s
`count(e)` is 17 RAW un-fused nodes, not deduplicated).
"""

from __future__ import annotations

import operator
import re
from typing import Any, Mapping, Sequence

# ── The one shared identifier grammar — mirrors querygen.py's own (§2.5) ────────────────────────
_VAR_RE = re.compile(r"^[a-z][a-z0-9]{0,7}$")
_PROP_RE = re.compile(r"^[a-z][a-zA-Z0-9]{0,31}$")
_PROJECTION_RE = re.compile(r"^([a-z][a-z0-9]{0,7})\.([a-z][a-zA-Z0-9]{0,31})$")
_AGGREGATE_RE = re.compile(
    r"^(count|avg|min|max)\(([a-z][a-z0-9]{0,7})(?:\.([a-z][a-zA-Z0-9]{0,31}))?\)$"
)
_OPS = frozenset({"=", "<>", "<", "<=", ">", ">="})
_ORDER_DIRS = frozenset({"ASC", "DESC"})
_NORMALIZED_OPS = frozenset({"=", "<>"})

_TYPE_BY_TOKEN: Mapping[str, type] = {"str": str, "int": int, "float": float}

_REQUEST_KEYS = frozenset({"matches", "returns", "order_by", "order_dir", "limit"})
_MATCH_KEYS = frozenset({"var", "label", "filters"})
_FILTER_KEYS = frozenset({"property", "op", "value"})


class MalformedSpecError(ValueError):
    """Layer A: the reply's JSON does not have `QueryRequest`'s declared shape at all — wrong
    types, an unknown key (`extra="forbid"`'s hand-rolled equivalent), a regex-failing
    var/property/returns/order_by entry, `filters` over 4 long, `returns` outside 1-6, `limit`
    outside `[1, 50]`, `op` outside the six-member whitelist, `matches` not exactly one entry."""


class SchemaViolationError(ValueError):
    """Layer B: the spec has `QueryRequest`'s shape but fails against THIS dataset's schema —
    unregistered label/property, a duplicate `returns` entry, a filter value that does not coerce
    to its property's declared type."""


def _require_a(condition: bool, message: str) -> None:
    if not condition:
        raise MalformedSpecError(message)


def _require_b(condition: bool, message: str) -> None:
    if not condition:
        raise SchemaViolationError(message)


# --------------------------------------------------------------------------------------------
# Layer A: structural validation (mirrors QueryFilter/QueryMatch/QueryRequest)
# --------------------------------------------------------------------------------------------


def validate_structure(spec: Mapping[str, Any]) -> None:
    """Layer A — raises `MalformedSpecError` on the first violation. Every check below has a
    direct counterpart in `querygen.py` (§2.5); none softens or widens the original rule."""
    _require_a(isinstance(spec, Mapping), "spec must be a JSON object")
    extra = set(spec) - _REQUEST_KEYS
    _require_a(not extra, f"unknown top-level key(s): {sorted(extra)!r}")

    matches = spec.get("matches")
    _require_a(
        isinstance(matches, list) and len(matches) == 1,
        "matches must be a list of exactly one entry",
    )
    match = matches[0]
    _require_a(isinstance(match, Mapping), "matches[0] must be a JSON object")
    extra = set(match) - _MATCH_KEYS
    _require_a(not extra, f"matches[0]: unknown key(s): {sorted(extra)!r}")

    var = match.get("var")
    _require_a(
        isinstance(var, str) and bool(_VAR_RE.fullmatch(var)),
        f"matches[0].var {var!r} is not a valid identifier",
    )

    label = match.get("label")
    _require_a(
        isinstance(label, str) and bool(label), "matches[0].label must be a non-empty string"
    )

    filters = match.get("filters", [])
    _require_a(isinstance(filters, list), "matches[0].filters must be a list")
    _require_a(len(filters) <= 4, f"matches[0].filters has {len(filters)} entries, max 4")
    for i, filt in enumerate(filters):
        _require_a(isinstance(filt, Mapping), f"filters[{i}] must be a JSON object")
        extra = set(filt) - _FILTER_KEYS
        _require_a(not extra, f"filters[{i}]: unknown key(s): {sorted(extra)!r}")

        prop = filt.get("property")
        _require_a(
            isinstance(prop, str) and bool(_PROP_RE.fullmatch(prop)),
            f"filters[{i}].property {prop!r} is not a valid identifier",
        )
        op = filt.get("op")
        _require_a(op in _OPS, f"filters[{i}].op {op!r} is not one of {sorted(_OPS)!r}")
        value = filt.get("value")
        _require_a(
            isinstance(value, (str, int, float)),
            f"filters[{i}].value {value!r} must be a string, number, or boolean",
        )

    returns = spec.get("returns")
    _require_a(
        isinstance(returns, list) and 1 <= len(returns) <= 6,
        "returns must be a list of 1-6 entries",
    )
    for i, r in enumerate(returns):
        _require_a(isinstance(r, str), f"returns[{i}] must be a string")
        _require_a(
            bool(_PROJECTION_RE.fullmatch(r)) or bool(_AGGREGATE_RE.fullmatch(r)),
            f"returns[{i}] {r!r} does not match a projection or aggregate shape",
        )

    order_by = spec.get("order_by")
    if order_by is not None:
        _require_a(
            isinstance(order_by, str) and bool(_PROJECTION_RE.fullmatch(order_by)),
            f'order_by {order_by!r} must be a bare projection ("var.property")',
        )

    order_dir = spec.get("order_dir", "ASC")
    _require_a(
        order_dir in _ORDER_DIRS, f"order_dir {order_dir!r} is not one of {sorted(_ORDER_DIRS)!r}"
    )

    limit = spec.get("limit", 20)
    _require_a(
        isinstance(limit, int) and not isinstance(limit, bool) and 1 <= limit <= 50,
        f"limit {limit!r} must be an integer in [1, 50]",
    )


# --------------------------------------------------------------------------------------------
# Layer B: schema-bound validation (mirrors compile())
# --------------------------------------------------------------------------------------------


def _normalize_name(value: str) -> str:
    """Two-line transcription of `extraction.normalize_name` (`extraction.py:67-78`) — whitespace-
    collapse + casefold. No import: this module reaches outside stdlib for nothing."""
    return re.sub(r"\s+", " ", value.strip()).casefold()


def _resolve_expr(
    expr: str, *, declared_var: str, allowed_props: Mapping[str, str], allow_aggregate: bool
) -> tuple[bool, str | None, str | None]:
    """Mirrors `querygen._resolve_expr` — decomposes one `returns`/`order_by` entry and validates
    every piece against `declared_var`/`allowed_props`, independently of Layer A's own regex
    check (defense-in-depth, same posture as `querygen.compile`'s own docstring). Returns
    `(is_aggregate, func_or_none, prop_or_none)`."""
    proj = _PROJECTION_RE.fullmatch(expr)
    if proj:
        var, prop = proj.group(1), proj.group(2)
        _require_b(var == declared_var, f"{expr!r} references unknown var {var!r}")
        _require_b(prop in allowed_props, f"{expr!r} references unknown property {prop!r}")
        return False, None, prop

    if allow_aggregate:
        agg = _AGGREGATE_RE.fullmatch(expr)
        if agg:
            func, var, prop = agg.group(1), agg.group(2), agg.group(3)
            _require_b(var == declared_var, f"{expr!r} references unknown var {var!r}")
            if prop is not None:
                _require_b(prop in allowed_props, f"{expr!r} references unknown property {prop!r}")
            return True, func, prop

    raise SchemaViolationError(f"{expr!r} does not resolve to an allowed projection or aggregate")


def _coerce_filter_value(filt: Mapping[str, Any], declared_type_token: str) -> Any:
    """Mirrors `querygen.compile`'s two filter-value fixes: Fix B (`*Normalized` case-
    fold/whitespace-collapse for `=`/`<>`) takes priority over Fix A (numeric-string coercion),
    exactly like the source's own `if/else` ordering (§2.5). Raises `SchemaViolationError` on a
    genuine coercion failure — the TYPE was legal, the VALUE didn't parse for it."""
    value = filt["value"]
    prop = filt["property"]
    op = filt["op"]
    declared_type = _TYPE_BY_TOKEN.get(declared_type_token, str)

    if isinstance(value, str) and prop.endswith("Normalized") and op in _NORMALIZED_OPS:
        return _normalize_name(value)
    if isinstance(value, str) and declared_type in (int, float):
        try:
            return declared_type(value)
        except ValueError as exc:
            raise SchemaViolationError(
                f"filter value {value!r} for property {prop!r} could not be parsed as "
                f"{declared_type.__name__}"
            ) from exc
    return value


def validate_against_schema(spec: Mapping[str, Any], schema: Mapping[str, Any]) -> None:
    """Layer B — raises `SchemaViolationError`. `schema` is the resolved dataset's OWN block
    (`{"labels": {...}}`, `schema.json["catalog"]` or `["knowledge_base"]` — never the whole
    registry, mirroring `querygen.compile(request, schema: DatasetSchema)`'s own one-dataset
    signature). label registered; every filter/return/order_by property registered for that
    label; no duplicate returns entries; each string filter value against a declared int/float
    property parses.

    Assumes `spec` already passed `validate_structure` — every index/key access below is safe
    only because Layer A guarantees the shape first (querygen's own validate-then-compile
    ordering, §2.5)."""
    labels = schema.get("labels", {})
    match = spec["matches"][0]
    label = match["label"]
    _require_b(label in labels, f"label {label!r} is not registered for this dataset")
    allowed_props = labels[label]
    declared_var = match["var"]

    for i, filt in enumerate(match.get("filters", [])):
        prop = filt["property"]
        _require_b(
            prop in allowed_props,
            f"filters[{i}].property {prop!r} is not registered for label {label!r}",
        )
        _coerce_filter_value(filt, allowed_props[prop])  # raises on a genuine coercion failure

    seen_returns: set[str] = set()
    for r in spec["returns"]:
        _resolve_expr(
            r, declared_var=declared_var, allowed_props=allowed_props, allow_aggregate=True
        )
        _require_b(r not in seen_returns, f"returns contains a duplicate expression: {r!r}")
        seen_returns.add(r)

    order_by = spec.get("order_by")
    if order_by is not None:
        _resolve_expr(
            order_by, declared_var=declared_var, allowed_props=allowed_props, allow_aggregate=False
        )


# --------------------------------------------------------------------------------------------
# Execution — no FalkorDB call, direct in-process filtering over tables.json
# --------------------------------------------------------------------------------------------

_OP_FUNCS: Mapping[str, Any] = {
    "=": operator.eq,
    "<>": operator.ne,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
}


def _row_matches(
    row: Mapping[str, Any], filters: Sequence[Mapping[str, Any]], allowed_props: Mapping[str, str]
) -> bool:
    for filt in filters:
        prop = filt["property"]
        op = filt["op"]
        value = _coerce_filter_value(filt, allowed_props[prop])
        try:
            if not _OP_FUNCS[op](row.get(prop), value):
                return False
        except TypeError:
            return False  # incomparable types (e.g. str vs float) never match
    return True


def _parse_expr(expr: str) -> tuple[bool, str | None, str | None]:
    """Same decomposition as `_resolve_expr`, without the schema check — `compile_and_execute`
    calls this only after `validate_against_schema` already confirmed `expr` resolves cleanly."""
    proj = _PROJECTION_RE.fullmatch(expr)
    if proj:
        return False, None, proj.group(2)
    agg = _AGGREGATE_RE.fullmatch(expr)
    return True, agg.group(1), agg.group(3)


def _aggregate_value(func: str, prop: str | None, rows: Sequence[Mapping[str, Any]]) -> Any:
    if func == "count":
        return len(rows) if prop is None else sum(1 for row in rows if row.get(prop) is not None)
    values = [row.get(prop) for row in rows if row.get(prop) is not None]
    if not values:
        return None
    if func == "avg":
        return sum(values) / len(values)
    if func == "min":
        return min(values)
    if func == "max":
        return max(values)
    raise AssertionError(f"unhandled aggregate func {func!r}")  # pragma: no cover - exhaustive


def compile_and_execute(
    spec: Mapping[str, Any],
    *,
    tables: Mapping[str, list[dict[str, Any]]],
    schema: Mapping[str, Any],
) -> dict[str, Any]:
    """Runs Layer A then Layer B (in order — a structurally-invalid spec never reaches the schema
    check, matching `querygen.py`'s own validate-then-compile ordering) then executes: filters
    `tables[label]`'s rows by the validated filters (coercing a string value against a
    `*Normalized` property via `_normalize_name`, matching `querygen.compile`'s own rule, §2.5),
    projects/aggregates `returns`, applies `order_by`/`order_dir`/`limit`. Returns `{"items":
    [...]}`, never raises past this point — `MalformedSpecError`/`SchemaViolationError` propagate
    to the caller (`extraction.py`'s `score_item`), which is what lets the three failure classes
    stay distinguishable at the call site rather than being caught and pooled here.

    `tables`/`schema` are both the resolved dataset's OWN blocks (`tables[label]` indexes
    directly into a label's row list; `schema["labels"]` indexes directly into a label's property
    map) — the caller (`extraction.py`) resolves `tables_json[dataset]`/`schema_json[dataset]`
    before calling, mirroring `querygen.compile(request, schema: DatasetSchema)`'s own
    single-dataset signature."""
    validate_structure(spec)
    validate_against_schema(spec, schema)

    match = spec["matches"][0]
    label = match["label"]
    allowed_props = schema["labels"][label]
    rows = tables.get(label, [])
    filtered = [row for row in rows if _row_matches(row, match.get("filters", []), allowed_props)]

    returns: list[str] = spec["returns"]
    resolved = [_parse_expr(r) for r in returns]
    is_aggregate_query = any(is_agg for is_agg, _, _ in resolved)

    if is_aggregate_query:
        row_out: dict[str, Any] = {}
        for expr, (is_agg, func, prop) in zip(returns, resolved):
            if is_agg:
                row_out[expr] = _aggregate_value(func, prop, filtered)
            else:
                row_out[expr] = filtered[0].get(prop) if filtered else None
        return {"items": [row_out]}

    order_by = spec.get("order_by")
    order_dir = spec.get("order_dir", "ASC")
    limit = spec.get("limit", 20)

    order_prop: str | None = None
    if order_by is not None:
        _, _, order_prop = _parse_expr(order_by)

    # De-duplicate by the full tuple actually needed (every returned column plus the sort key, if
    # any) BEFORE sorting/limiting — the module docstring's own design note: matches
    # `querygen.compile`'s default `RETURN DISTINCT` for a non-aggregate returns list, needed for
    # correctness against duplicate un-fused knowledge_base rows.
    seen: set[tuple[Any, ...]] = set()
    deduped: list[Mapping[str, Any]] = []
    for row in filtered:
        proj_values = tuple(row.get(prop) for _, _, prop in resolved)
        key = proj_values if order_prop is None else proj_values + (row.get(order_prop),)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)

    if order_by is not None:
        deduped.sort(key=lambda r: r.get(order_prop), reverse=(order_dir == "DESC"))

    limited = deduped[:limit]
    items = [
        {expr: row.get(prop) for expr, (_, _, prop) in zip(returns, resolved)} for row in limited
    ]
    return {"items": items}
