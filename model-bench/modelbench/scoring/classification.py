"""guard-judge's `ItemScorer` — a module, not a class, satisfying the Protocol structurally
(`scoring/retrieval.py`'s own precedent). Transcribes `falkorchat.app._LlmGuardJudge`'s prompt
construction, `falkorchat.guards._recent_turns`'s turn normalization, and
`falkorchat.llm.extract_own_line_json_object`'s conservative parse as DATA + CODE, never an import
(D1, FR-23 — `model-bench` never reads `falkor-chat` at run time). No live call, no database — one
chat call per item.

Design: `docs/plans/small-model-benchmarking-s4-spec.md` §5.1.3, §2.3-§2.4 (the two real
mechanisms this module transcribes, verified directly against the shipped falkor-chat source), §7
Step 1 (this module's own red-then-green test order).
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from modelbench.lmstudio import ChatResult
from modelbench.packs import Pack
from modelbench.results import BinaryMetric, ClassificationAggregates, ItemResult, ItemTiming
from modelbench.runner import ChatMessage

# --- turn normalization (transcribed from guards.py:538-567, verbatim rule set) -----------------

_RECENT_TURNS_N = 6
_TURN_TEXT_MAX = 400


def _normalize_turns(raw_turns: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    """`guards._recent_turns`'s exact rule (guards.py:538-567): drop rows with no non-empty string
    `text` FIRST, and only THEN take the last `_RECENT_TURNS_N` (K-027 carried finding — filtering
    after slicing would let a malformed row in the tail shrink the usable evidence window).
    `speaker = displayName or authorId or "member"`; `role = role or "user"`; `text` truncated to
    `_TURN_TEXT_MAX` chars. Input is already chronological, so the last N valid rows are "newest
    last"."""
    turns: list[dict[str, str]] = []
    for row in raw_turns:
        text = row.get("text")
        if not isinstance(text, str) or not text:
            continue
        speaker = row.get("displayName") or row.get("authorId") or "member"
        role = row.get("role") or "user"
        turns.append({"speaker": str(speaker), "role": str(role), "text": text[:_TURN_TEXT_MAX]})
    return turns[-_RECENT_TURNS_N:]


# --- prompt rendering (transcribed from app.py:619-654) ------------------------------------------

_JUDGE_USER_MAX_CHARS = 6000


def _render_judge_user(
    condition: str, understanding: Mapping[str, Any], turns: Sequence[Mapping[str, str]]
) -> str:
    """Verbatim port of `app._render_judge_user` (app.py:619-654): CONDITION always present,
    CURRENT STATE iff `understanding` is non-empty (`json.dumps(..., indent=2, sort_keys=True,
    default=str)`), RECENT TURNS iff `turns` is non-empty — capped at `_JUDGE_USER_MAX_CHARS` by
    dropping OLDEST turns first, so the newest turn (the one the condition is usually about)
    always survives; a hard truncation of `base` backstops the case where even the single newest
    turn cannot fit."""
    blocks = [f"CONDITION: {condition}"]
    if understanding:
        state = json.dumps(understanding, indent=2, sort_keys=True, default=str)
        blocks.append(f"CURRENT STATE:\n{state}")
    base = "\n\n".join(blocks)

    lines = [f"{t.get('speaker', 'member')}: {t.get('text', '')}" for t in turns]
    n = len(lines)
    header = "RECENT TURNS (context only):\n"
    kept = n
    total = len(base) + 2 + len(header) + sum(map(len, lines)) + max(0, n - 1)
    while kept > 0 and total > _JUDGE_USER_MAX_CHARS:
        total -= len(lines[n - kept]) + (1 if kept > 1 else 0)
        kept -= 1

    if kept:
        rendered = "\n".join(lines[-kept:])
        return "\n\n".join(blocks + [f"{header}{rendered}"])
    return base[:_JUDGE_USER_MAX_CHARS]


# --- parse (transcribed from llm.py:474-601 — same conservative parser guard-judge's production
# parser uses; re-implemented once here, not imported, per D1/FR-23) -------------------------------

def _strip_code_fence(text: str) -> str:
    """Verbatim port of `llm._strip_code_fence` (llm.py:483-495): strips one surrounding ```/
    ```json fence, if present; otherwise returns `text` unchanged."""
    if not text.startswith("```"):
        return text
    inner = text[3:]
    if inner[:4].lower() == "json":
        inner = inner[4:]
    return inner.rsplit("```", 1)[0].strip()


def _load_json_object(text: str) -> dict[str, Any] | None:
    """`json.loads(text)` when it yields a dict, else None (llm.py:595-601)."""
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _scan_balanced(text: str, open_idx: int) -> tuple[str, int] | None:
    """Verbatim port of `llm._scan_balanced` (llm.py:439-...): returns `(inner, index_after_close)`
    for the `{`/`}` bracket at `open_idx`, ignoring brackets inside JSON string literals (and their
    backslash escapes). `None` on unbalanced input."""
    opener = text[open_idx]
    closer = {"(": ")", "{": "}"}[opener]
    depth = 0
    in_string = False
    escaped = False
    for i in range(open_idx, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[open_idx + 1 : i], i + 1
    return None


#: A JSON object opening a line: optional indentation, then `{` (llm.py:534-537) — the own-line
#: anchor is what distinguishes an ASSERTED verdict from one a sentence merely quotes.
_OWN_LINE_OBJECT_OPEN = re.compile(r"^[ \t]*\{", re.MULTILINE)


def extract_own_line_json_object(
    content: Any, *, require_key: str | None = None
) -> dict[str, Any] | None:
    """Verbatim port of `llm.extract_own_line_json_object` (llm.py:540-601) — the conservative
    sibling of the permissive `extract_json_object`, biased to find *nothing*.

    Accepts, after stripping one surrounding code fence: a reply that is entirely one JSON object
    (bare or fenced), or exactly one JSON object that OWNS ITS LINES — its `{` is the first
    non-whitespace character of a line, nothing but whitespace follows its matching `}` on that
    line — and which carries `require_key` when one is given. Everything else (an object embedded
    mid-sentence, two or more qualifying objects, nothing that parses) returns `None`.

    `-ml`/plan reference this as the pack's declared parse mode `"ownLineJsonObject"` (S4 spec
    §5.1.3)."""
    if not isinstance(content, str):
        return None
    text = _strip_code_fence(content.strip())
    if not text:
        return None

    whole = _load_json_object(text)
    if whole is not None:
        return whole

    matches: list[dict[str, Any]] = []
    consumed = 0
    for match in _OWN_LINE_OBJECT_OPEN.finditer(text):
        if match.start() < consumed:
            continue  # nested inside an object already scanned
        scanned = _scan_balanced(text, match.end() - 1)
        if scanned is None:
            continue
        _, after = scanned
        consumed = after
        if text[after:].split("\n", 1)[0].strip():
            continue  # trailing prose on the closing line -> quoted, not asserted
        parsed = _load_json_object(text[match.end() - 1 : after])
        if parsed is None or (require_key is not None and require_key not in parsed):
            continue
        matches.append(parsed)

    return matches[0] if len(matches) == 1 else None


# --- prompt assembly (the `ItemScorer.build_messages` hook, runner spec §4.1) --------------------


def build_messages(item_input: Mapping[str, Any], *, pack: Pack) -> list[ChatMessage]:
    """Never reads `item_input["expected"]`/`["label_rationale"]`/`["r1_probe"]` — those would
    leak the gold label into the prompt. `pack.prompt_config().systemPrompt` is
    `prompts/judge.md`'s resolved content (S4 spec §4.2); the user message is `_render_judge_user`
    over this item's own `condition`/`understanding` and its `turns` NORMALIZED first
    (`_normalize_turns` — §2.3 finding 2's fix: production normalizes before rendering, and
    skipping that step silently renders every speaker as "member")."""
    system_prompt = pack.prompt_config().systemPrompt
    turns = _normalize_turns(item_input.get("turns") or [])
    user = _render_judge_user(item_input["condition"], item_input.get("understanding") or {}, turns)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ]


# --- per-item metric assignment -------------------------------------------------------------------

#: tier -> the metric name that tier's items contribute to (S4 spec §5.1.3). `boundary`'s metric is
#: exploratory (never in `verdictMetrics` — plan §3.8.2/`-ml` §7.3: "not a verdict metric" at n=15).
_METRIC_BY_TIER: Mapping[str, str] = {
    "clear_suspend": "falseAdvanceRate",     # expected=False; a False->True flip is a false advance
    "clear_advance": "falseSuspendRate",     # expected=True; a True->False flip is a false suspend
    "boundary": "falseAdvanceRateBoundary",  # all expected=False (verified §2.4); descriptive only
}


def score_item(
    item_input: Mapping[str, Any], result: ChatResult | None, timing: ItemTiming, *, pack: Pack,
) -> ItemResult:
    """One judge call. `result is None` -> outcome `"fail"` (timeout) / `"unrunnable"` (no
    response), per `timing.withheldFor`, mirroring `retrieval.py`'s own precedent (S4 spec §2.1) —
    `scoreable={}`, no contribution to any metric's denominator (a call that never happened cannot
    be "the judge advanced/suspended").

    Otherwise: parses `result.message.get("content")` via `extract_own_line_json_object(...,
    require_key="decision")`. `None` -> `advanced=False`, `outcome="parse_failure"` — the SAME
    fallback the real judge applies (app.py:727-728), never a fabricated guess. Parsed -> `advanced
    = bool(parsed.get("decision"))`, `outcome="pass"`.

    Exactly one metric name is scoreable per item (`_METRIC_BY_TIER[item_input["tier"]]`), with
    `counts[metric] = int(advanced)` for `clear_suspend`/`boundary` (expected=False, verified) —
    `advanced=True` IS the false-advance event; for `clear_advance` (expected=True), `advanced=
    False` IS the false-suspend event, so `counts["falseSuspendRate"] = int(not advanced)`.
    `detail` carries `{"path": item_input["path"], "tier": item_input["tier"]}` for the aggregate's
    path-split diagnostic — never anything answer-adjacent."""
    item_id = item_input["itemId"]
    if result is None:
        outcome = "fail" if timing.withheldFor == "timeout" else "unrunnable"
        return ItemResult(
            itemId=item_id,
            pairingKey=(item_id,),
            outcome=outcome,
            scoreable={},
            counts={},
            timing=timing,
        )

    metric = _METRIC_BY_TIER[item_input["tier"]]
    detail = {"path": item_input["path"], "tier": item_input["tier"]}

    parsed = extract_own_line_json_object(
        result.message.get("content"), require_key="decision"
    )
    if parsed is None:
        advanced = False
        outcome = "parse_failure"
    else:
        advanced = bool(parsed.get("decision"))
        outcome = "pass"

    count = int(not advanced) if metric == "falseSuspendRate" else int(advanced)

    return ItemResult(
        itemId=item_id,
        pairingKey=(item_id,),
        outcome=outcome,
        scoreable={metric: True},
        counts={metric: count},
        timing=timing,
        detail=detail,
    )


# --- aggregation ------------------------------------------------------------------------------

_PATHS: tuple[str, ...] = ("understanding", "turns")


def _binary_metric(
    name: str, items: Sequence[ItemResult], *, count_key: str | None = None
) -> BinaryMetric:
    """`items` already the subset that declared `count_key` (default `name`) scoreable — each item
    declares exactly one metric scoreable, `score_item`'s own contract — so `successes` is the raw
    sum of that metric's per-item count and `n` is the subset's own size, never the whole run's.
    `count_key` differs from `name` only for a path-split metric: `counts` still carries the
    item's original metric name (`"falseAdvanceRate"`), never the derived
    `"falseAdvanceRateByUnderstanding"` label this function is asked to print."""
    key = count_key if count_key is not None else name
    successes = sum(it.counts.get(key, 0) for it in items)
    return BinaryMetric(name=name, successes=successes, n=len(items), unit="item")


def aggregate(items: Sequence[ItemResult], *, pack: Pack) -> ClassificationAggregates:
    """One pass. `perClass` carries, in order: `falseAdvanceRate` (n=40 over the real 85-item
    pack, the `clear_suspend` items), `falseSuspendRate` (n=30, the `clear_advance` items) — the
    two `verdictMetrics`, both `BinaryMetric(unit="item")`; `advanceRecall` (the labelled
    complement, `falseSuspendRate.n - falseSuspendRate.successes` over the same n) — printed,
    never in `verdictMetrics`, so `report.py` renders it exploratory automatically (S4 spec §2.1);
    `falseAdvanceRateBoundary` (n=15, the `boundary` tier) — exploratory, descriptive only (same
    mechanism); `falseAdvanceRateByUnderstanding`/`falseAdvanceRateByTurns`,
    `falseSuspendRateByUnderstanding`/`falseSuspendRateByTurns` — the path-split diagnostic (plan
    §3.8.2), four more exploratory `BinaryMetric`s, each filtered by `detail["path"]` within its
    own tier's item set. `parseFailures` = count of items with `outcome == "parse_failure"`. `n` =
    `len(items)`.

    Each metric's own subset is every item that DECLARED that metric scoreable (`item.scoreable`),
    never a raw tier/path lookup off `detail` — a timed-out/unrunnable item carries no `detail` at
    all (`score_item`'s `result is None` branch), so grouping by declared metric is what keeps
    those items out of every `perClass` denominator while still counting toward `n`.

    **No metric here is a pooled figure across tiers or paths — every `BinaryMetric`'s `n` is its
    own named subset's count**, never the whole run's `n` (plan §3.8.2: "no pooled 85-item figure
    anywhere")."""

    def declaring(name: str) -> list[ItemResult]:
        return [it for it in items if it.scoreable.get(name)]

    false_advance = _binary_metric("falseAdvanceRate", declaring("falseAdvanceRate"))
    false_suspend = _binary_metric("falseSuspendRate", declaring("falseSuspendRate"))
    false_advance_boundary = _binary_metric(
        "falseAdvanceRateBoundary", declaring("falseAdvanceRateBoundary")
    )
    advance_recall = BinaryMetric(
        name="advanceRecall",
        successes=false_suspend.n - false_suspend.successes,
        n=false_suspend.n,
        unit="item",
    )

    def by_path(metric_name: str) -> list[BinaryMetric]:
        subset = declaring(metric_name)
        return [
            _binary_metric(
                f"{metric_name}By{path.capitalize()}",
                [it for it in subset if it.detail.get("path") == path],
                count_key=metric_name,
            )
            for path in _PATHS
        ]

    per_class = (
        false_advance,
        false_suspend,
        advance_recall,
        false_advance_boundary,
        *by_path("falseAdvanceRate"),
        *by_path("falseSuspendRate"),
    )
    parse_failures = sum(1 for it in items if it.outcome == "parse_failure")

    return ClassificationAggregates(perClass=per_class, parseFailures=parse_failures, n=len(items))
