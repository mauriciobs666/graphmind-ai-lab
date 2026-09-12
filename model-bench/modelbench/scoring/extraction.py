"""`nlq-structured-query`'s `ItemScorer` — re-implements `nlq_scoring.py`'s Layer 1 comparison
rules (S4 spec §2.5/§5.2.4) and builds the per-item structured-completion prompt (§2.5's
transcription of `tools.py`). One chat call per item; execution is entirely in-process against
`tables.json` via the pack's own `tools/exec.py` (`Pack.load_tool_module()`) — no FalkorDB.

Design: `docs/plans/small-model-benchmarking-s4-spec.md` §5.2.4, §2.5 (the real mechanisms this
module transcribes, verified directly against the shipped falkor-chat source), §7 Step 3 (this
module's own red-then-green test order, in three passes).
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from modelbench.lmstudio import ChatResult
from modelbench.packs import Pack
from modelbench.results import BinaryMetric, ExtractionAggregates, ItemResult, ItemTiming
from modelbench.runner import ChatMessage
from modelbench.scoring.classification import extract_own_line_json_object

# --- Layer 1 scoring (transcribed from nlq_scoring.py) -------------------------------------------

#: Layer 1's numeric tolerance for float rounding (`nlq_scoring.py`'s own `_NUMERIC_EPSILON`) —
#: never string-equal on a formatted price.
_NUMERIC_EPSILON = 0.01

_WHITESPACE_RE = re.compile(r"\s+")


def _canon_str(value: Any) -> str:
    """Verbatim port of `nlq_scoring._canon_str` (`nlq_scoring.py:87-90`)."""
    return _WHITESPACE_RE.sub(" ", str(value).strip().casefold())


def _scalar_equal(expected: Any, actual: Any) -> bool:
    """Verbatim port of `nlq_scoring._scalar_equal`: numeric epsilon (`0.01` plus `1e-9` boundary
    slop) for two numbers, canonical string equality otherwise, never coerced across the two. A
    `bool` is compared by identity — `isinstance(x, bool)` is checked BEFORE the numeric branch
    because `bool` is a subclass of `int` in Python, so `True == 1` would otherwise silently pass
    a bool/int comparison the source's own rule does not intend."""
    if isinstance(expected, bool) or isinstance(actual, bool):
        return expected is actual
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return abs(float(expected) - float(actual)) <= _NUMERIC_EPSILON + 1e-9
    if isinstance(expected, (int, float)) or isinstance(actual, (int, float)):
        return False  # one numeric, one not — never coerced
    return _canon_str(expected) == _canon_str(actual)


def _extract_scalar(items: list[dict[str, Any]]) -> tuple[Any, bool]:
    """See `nlq_scoring`'s own scalar-extraction rule: exactly one row, exactly one column, or
    `ok=False`. Zero rows, more than one row, or more than one column is scored incorrect rather
    than guessed at."""
    if len(items) != 1:
        return None, False
    row = items[0]
    if len(row) != 1:
        return None, False
    return next(iter(row.values())), True


def _flatten_values(items: list[dict[str, Any]]) -> set[str]:
    """Every value across every row/column, canonicalized — `nlq_scoring`'s own set-extraction
    rule, deliberately column-name-agnostic."""
    return {_canon_str(v) for row in items for v in row.values()}


def score_pair(
    expected: Mapping[str, Any], shape: str, tool_result: Mapping[str, Any]
) -> tuple[bool, str]:
    """Verbatim port of `nlq_scoring.score_pair`'s three `etype` branches (scalar/set/not_found),
    **including the `conflicting-facts` subset-containment exception**
    (`shape == "conflicting-facts"` -> `expected_set.issubset(actual_set)`, `nlq_scoring.py:186-
    195` — NOT set equality; every OTHER set-shaped `shape` uses exact set equality). Returns
    `(correct, reason)`. Raises `ValueError` for an `expected["type"]` outside `scalar | set |
    not_found` — a defect in the caller/fixture, never silently absorbed."""
    etype = expected["type"]
    items = list(tool_result.get("items", []))

    if etype == "not_found":
        correct = len(items) == 0
        reason = (
            "result is genuinely empty"
            if correct
            else f"expected not_found but got {len(items)} row(s): {items!r}"
        )
        return correct, reason

    if etype == "scalar":
        value, ok = _extract_scalar(items)
        if not ok:
            return False, f"expected a single scalar row/column, got {items!r}"
        correct = _scalar_equal(expected["value"], value)
        reason = (
            f"{value!r} matches expected {expected['value']!r}"
            if correct
            else f"{value!r} does not match expected {expected['value']!r}"
        )
        return correct, reason

    if etype == "set":
        actual_set = _flatten_values(items)
        expected_set = {_canon_str(v) for v in expected["values"]}
        if shape == "conflicting-facts":
            # Containment, not exact match: surfacing the required values plus something extra
            # is still correct for this one shape (nlq_scoring.py's own docstring/S4 spec §2.5).
            correct = expected_set.issubset(actual_set)
            reason = (
                "all expected conflicting values present"
                if correct
                else f"missing {sorted(expected_set - actual_set)} from {sorted(actual_set)}"
            )
        else:
            correct = expected_set == actual_set
            reason = (
                "exact set match"
                if correct
                else f"expected {sorted(expected_set)}, got {sorted(actual_set)}"
            )
        return correct, reason

    raise ValueError(f"unknown expected.type {etype!r}")


# --- prompt assembly (transcribed from tools.py) --------------------------------------------------


def _describe_dataset_schema(schema_block: Mapping[str, Any]) -> str:
    """Verbatim port of `tools._describe_dataset_schema` (`tools.py:825-836`): "{label}
    (properties: {sorted, comma-joined property names})" per label, joined with "; ".
    `schema_block` is one RESOLVED dataset's own block (`schema.json["catalog"]` or
    `["knowledge_base"]`, `{"labels": {...}}`) — never the whole two-dataset registry."""
    return "; ".join(
        f"{label} (properties: {', '.join(sorted(props))})"
        for label, props in schema_block["labels"].items()
    )


def build_messages(item_input: Mapping[str, Any], *, pack: Pack) -> list[ChatMessage]:
    """Never reads `item_input["expected"]`/`["rationale"]`/`["answerable"]` — those would leak
    the gold answer into the prompt. Reads `pack.data_path("schema")` for THIS item's dataset's
    schema block, fills `prompts/querygen.md`'s `{dataset_schema}` placeholder (via `str.format`,
    the SAME call `tools._build_query_request_system_prompt` makes — the shipped prompt file is
    the verbatim, still-double-braced `_QUERY_REQUEST_INSTRUCTIONS` template text, so only
    `.format` un-escapes it correctly), returns `[{"role": "system", "content": filled},
    {"role": "user", "content": item_input["question"]}]` — the same two-message shape
    `QueryGraphDataTool.run()` sends (`tools.py:985-988`).

    `prompt.systemPrompt` stays `None` at the manifest level for this pack (§5.2.1) — the real
    system prompt depends on the item's own `dataset`, so it is resolved here directly rather
    than through `pack.prompt_config().systemPrompt`."""
    schema = _load_json(pack.data_path("schema"))
    schema_block = schema[item_input["dataset"]]
    template = (pack.root / "prompts" / "querygen.md").read_text(encoding="utf-8")
    system_prompt = template.format(dataset_schema=_describe_dataset_schema(schema_block))
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": item_input["question"]},
    ]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


# --- per-item scoring (wires build_messages/score_pair above to the pack's own tools/exec.py) ----


def _title_shape(shape: str) -> str:
    """`"single-fact"` -> `"SingleFact"`, `"conflicting-facts"` -> `"ConflictingFacts"` — the
    per-shape exploratory metric name suffix (`f"exactMatchBy{_title_shape(shape)}"`, S4 spec
    §5.2.4)."""
    return shape.title().replace("-", "")


def score_item(
    item_input: Mapping[str, Any], result: ChatResult | None, timing: ItemTiming, *, pack: Pack,
) -> ItemResult:
    """One structured-completion call, three gates in order (S4 spec §5.2.4), each producing a
    DISTINCT `outcome`/`detail` so the three failure classes stay separable at `aggregate()`:

    1. `extract_own_line_json_object(..., require_key="matches")` is `None` ->
       `outcome="parse_failure"`, `detail={"failureClass": "no_json"}`.
    2. `tools_exec.validate_structure(parsed)` raises `MalformedSpecError` -> `outcome="fail"`,
       `detail={"failureClass": "malformed_spec"}`.
    3. `tools_exec.validate_against_schema(parsed, schema_for(dataset))` raises
       `SchemaViolationError` -> `outcome="fail"`, `detail={"failureClass": "schema_violation"}`.

    None of the three gates above declares ANY metric scoreable — they are counted separately, by
    `aggregate()`, straight off `detail["failureClass"]` (never pooled into `layer1ExactMatchRate`'s
    own denominator, which only ever counts items that reached execution).

    Otherwise `tools_exec.compile_and_execute(...)` executes. `item_input["answerable"] is False`
    -> scores `unanswerableAbstainRate` (`1` iff the executed result is empty, correctly
    abstaining; `0` if it fabricated an answer) — never `layer1ExactMatchRate`. An answerable item
    is scored via `score_pair(item_input["expected"], item_input["shape"], tool_result)` into BOTH
    `layer1ExactMatchRate` (scoreable=True always, even on a wrong answer) and a per-shape
    exploratory metric `f"exactMatchBy{shape}"`.

    **Correction (A3, `docs/reviews/nlq-conflicting-facts-answerability-ml.md` Q2/F-1): excluded
    from the denominator does not mean unscored.** The unanswerable branch ALSO calls
    `score_pair(item_input["expected"], item_input["shape"], tool_result)` — every golden item,
    answerable or not, carries `expected`/`shape` — and sets the SAME per-shape exploratory metric
    `f"exactMatchBy{shape}"`, alongside `unanswerableAbstainRate`, never instead of it. This is why
    `relationship-traversal`/`conflicting-facts` now get a real `exactMatchBy{Shape}` entry
    (previously absent, since no item of either shape was ever answerable). If that exploratory
    call scores correct — F-4's degenerate-spec class of outcome, real now that the branch
    executes against production data — `detail["luckyPass"] = True` is also set, so `aggregate()`
    can count it distinctly rather than let it read as an ordinary win. `layer1ExactMatchRate`
    itself is untouched by this: the unanswerable branch still never sets it scoreable.

    `result is None` -> `outcome="fail"` (timeout) / `"unrunnable"` (no_response), per
    `timing.withheldFor`, mirroring `classification.py`/`retrieval.py`'s own precedent —
    `scoreable={}`, no contribution to any metric's denominator."""
    item_id = item_input["itemId"]
    if result is None:
        outcome = "fail" if timing.withheldFor == "timeout" else "unrunnable"
        return ItemResult(
            itemId=item_id, pairingKey=(item_id,), outcome=outcome,
            scoreable={}, counts={}, timing=timing,
        )

    parsed = extract_own_line_json_object(result.message.get("content"), require_key="matches")
    if parsed is None:
        return ItemResult(
            itemId=item_id, pairingKey=(item_id,), outcome="parse_failure",
            scoreable={}, counts={}, timing=timing, detail={"failureClass": "no_json"},
        )

    tool_module = pack.load_tool_module()
    dataset = item_input["dataset"]
    schema = _load_json(pack.data_path("schema"))
    tables = _load_json(pack.data_path("tables"))

    try:
        tool_module.validate_structure(parsed)
    except tool_module.MalformedSpecError:
        return ItemResult(
            itemId=item_id, pairingKey=(item_id,), outcome="fail",
            scoreable={}, counts={}, timing=timing, detail={"failureClass": "malformed_spec"},
        )

    try:
        tool_module.validate_against_schema(parsed, schema[dataset])
    except tool_module.SchemaViolationError:
        return ItemResult(
            itemId=item_id, pairingKey=(item_id,), outcome="fail",
            scoreable={}, counts={}, timing=timing, detail={"failureClass": "schema_violation"},
        )

    try:
        tool_result = tool_module.compile_and_execute(
            parsed, tables=tables[dataset], schema=schema[dataset]
        )
    except (tool_module.MalformedSpecError, tool_module.SchemaViolationError):
        # Defensive only — neither layer should raise here, both already passed above. A boundary
        # defect must not silently swallow or crash the run; classify as a schema violation
        # (compile_and_execute's own docstring: "never raises past this point" is the contract
        # this guards, not an expected path).
        return ItemResult(
            itemId=item_id, pairingKey=(item_id,), outcome="fail",
            scoreable={}, counts={}, timing=timing, detail={"failureClass": "schema_violation"},
        )

    if item_input["answerable"] is False:
        # Correction A3 (`docs/reviews/nlq-conflicting-facts-answerability-ml.md` Q2/F-1):
        # excluded from `layer1ExactMatchRate`'s denominator does NOT mean unscored. Every golden
        # item, answerable or not, carries `expected`/`shape`, so the exploratory score_pair call
        # runs here too, alongside (never instead of) `unanswerableAbstainRate`. A correct result
        # is F-4's degenerate-spec class of outcome — a lucky pass, named as such rather than
        # left to read as an ordinary win.
        abstained = not tool_result.get("items")
        shape = item_input["shape"]
        correct, _reason = score_pair(item_input["expected"], shape, tool_result)
        shape_metric = f"exactMatchBy{_title_shape(shape)}"
        detail: dict[str, Any] = {"dataset": dataset, "shape": shape}
        if correct:
            detail["luckyPass"] = True
        return ItemResult(
            itemId=item_id, pairingKey=(item_id,), outcome="pass",
            scoreable={"unanswerableAbstainRate": True, shape_metric: True},
            counts={"unanswerableAbstainRate": int(abstained), shape_metric: int(correct)},
            timing=timing, detail=detail,
        )

    shape = item_input["shape"]
    correct, reason = score_pair(item_input["expected"], shape, tool_result)
    shape_metric = f"exactMatchBy{_title_shape(shape)}"
    return ItemResult(
        itemId=item_id, pairingKey=(item_id,), outcome="pass",
        scoreable={"layer1ExactMatchRate": True, shape_metric: True},
        counts={"layer1ExactMatchRate": int(correct), shape_metric: int(correct)},
        timing=timing, detail={"shape": shape, "reason": reason},
    )


# --- aggregation -----------------------------------------------------------------------------


def aggregate(items: Sequence[ItemResult], *, pack: Pack) -> ExtractionAggregates:
    """`exactMatch` = `BinaryMetric("layer1ExactMatchRate", ...)` over ONLY the items that
    DECLARED it scoreable — i.e. the answerable items that reached execution (S4 spec §5.2.4,
    corrected per `docs/reviews/nlq-conflicting-facts-answerability-ml.md` A1: **n = 34, the 40
    minus the 6 stamped unanswerable** — 4 `relationship-traversal` + 2 `conflicting-facts`; this
    function itself is fully data-driven off each item's own `scoreable` map, never a hardcoded
    count, so the correction lives in this docstring and the plan text, not in the arithmetic).

    `byShape` carries one `BinaryMetric` per shape name, **pooling BOTH the answerable items
    scored through `layer1ExactMatchRate`'s own `exactMatchBy{Shape}` AND the unanswerable items'
    exploratory `exactMatchBy{Shape}` (`score_item`'s A3 correction)** — computed by filtering
    `items` on each item's own `exactMatchBy{Shape}` scoreable flag, never by gating on
    `layer1ExactMatchRate` the way the pre-correction version did, because a shape's per-shape
    score is real and reported every run regardless of whether the shape counts toward the
    headline denominator. Concretely: `single-fact`/`filter-list`/`compound-filter`/`not-found`/
    `aggregation` each have only answerable-path members; `relationship-traversal`/
    `conflicting-facts` have ONLY exploratory-path members today, since no item of either shape is
    ever answerable — this is not a special case in the aggregation code, it falls out of which
    items declare the metric scoreable. PLUS `unanswerableAbstainRate` (n = 6 in the real pack,
    all six structurally-unanswerable items, not 4).

    `luckyPassCount` = `sum(1 for it in items if it.detail.get("luckyPass") is True)` — an item
    counted here is ALSO counted in its `exactMatchBy{Shape}` metric's successes; the two are
    additive, never alternatives, and any future `report.py` rendering of `luckyPassCount` must
    not let the shape rate stand alone unlabelled.

    `parseFailures`/`malformedSpecCount`/`schemaViolationCount` are each a `sum(1 for it in items
    if it.detail.get("failureClass") == ...)` over the three named classes — the three counts the
    plan's cost note requires, never pooled into one."""

    def declaring(name: str) -> list[ItemResult]:
        return [it for it in items if it.scoreable.get(name)]

    exact_items = declaring("layer1ExactMatchRate")
    exact_match = BinaryMetric(
        name="layer1ExactMatchRate",
        successes=sum(it.counts.get("layer1ExactMatchRate", 0) for it in exact_items),
        n=len(exact_items),
        unit="item",
    )

    # Correction A3 (`docs/reviews/nlq-conflicting-facts-answerability-ml.md` Q2/F-1): pool BOTH
    # the answerable items' `exactMatchBy{Shape}` contributions AND the unanswerable items' now-
    # added exploratory ones into the SAME per-shape metric — filtering on each item's own
    # `exactMatchBy{Shape}` scoreable flag, never on `layer1ExactMatchRate` (the pre-correction
    # gate, which silently dropped every unanswerable-path contribution).
    shapes_seen = sorted({it.detail["shape"] for it in items if "shape" in it.detail})
    by_shape_metrics = []
    for shape in shapes_seen:
        metric_name = f"exactMatchBy{_title_shape(shape)}"
        contributing = declaring(metric_name)
        if not contributing:
            continue
        by_shape_metrics.append(
            BinaryMetric(
                name=metric_name,
                successes=sum(it.counts.get(metric_name, 0) for it in contributing),
                n=len(contributing),
                unit="item",
            )
        )
    by_shape = tuple(by_shape_metrics)

    unanswerable_items = declaring("unanswerableAbstainRate")
    unanswerable = BinaryMetric(
        name="unanswerableAbstainRate",
        successes=sum(it.counts.get("unanswerableAbstainRate", 0) for it in unanswerable_items),
        n=len(unanswerable_items),
        unit="item",
    )

    parse_failures = sum(1 for it in items if it.detail.get("failureClass") == "no_json")
    malformed_spec_count = sum(
        1 for it in items if it.detail.get("failureClass") == "malformed_spec"
    )
    schema_violation_count = sum(
        1 for it in items if it.detail.get("failureClass") == "schema_violation"
    )
    # (§4.3/§5.2.4 correction, A3) — additive with, never an alternative to, the item's own
    # `exactMatchBy{Shape}` success counted above.
    lucky_pass_count = sum(1 for it in items if it.detail.get("luckyPass") is True)

    return ExtractionAggregates(
        exactMatch=exact_match,
        byShape=(*by_shape, unanswerable),
        parseFailures=parse_failures,
        malformedSpecCount=malformed_spec_count,
        schemaViolationCount=schema_violation_count,
        luckyPassCount=lucky_pass_count,
    )
