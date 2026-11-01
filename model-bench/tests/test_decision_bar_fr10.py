"""typed-decision-models U4.3: `compare`'s bar section on a new-ability pack — the FR-10 per-class
lines with their Wilson intervals, class minimum and per-class `overLimit`, the mean-latency line,
the replicate states and the replicate disagreement line (v1.3 §3.6; plan2 §3.6; plan3 §3.6; note
§Q4.2, §Q5, §Q6), plus the comparison-wide blocking FR-10 shares with the speed-up side (impl
review Pass 18).

Each candidate replicate starts as a real `run_pack` record — `test_runner_handoff`'s new-ability
workspace, its items replaced by 200 `eval` items of the role, driven offline by its stub — and is
then given the items a test needs, built by the real `agreement` scorer, with the aggregates
re-derived by the scorer's and the runner's own aggregation (so the aggregate-consistency check
still passes). Everything is written under `tmp_path`, never under `model-bench/results/`.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
import test_runner_decisions as base
import test_runner_handoff as ho

from modelbench import cli, decision_stats, decisions, report, runner
from modelbench.decisions import MESSAGE_TYPE_LABELS
from modelbench.packs import load_pack
from modelbench.results import ItemResult, ItemTiming, RunResult, store
from modelbench.scoring import agreement

JOIN_IN, MESSAGE_TYPE = "join-in-judge", "message-type-classifier"
CAND = base._CANDIDATE_ID
QWEN = ho.QWEN
SESSION = "s-fr10"
_T0 = datetime(2026, 11, 1, 9, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _fast_bootstrap(monkeypatch):
    """B is a pinned constant (plan2 §3.6); the report reads it at call time, so these tests
    resample less. The one latency interval asserted below is recomputed at this B."""
    monkeypatch.setattr(decision_stats, "BOOTSTRAP_B", 200)


# --- building records -----------------------------------------------------------------------------


def _item(role: str, n: int, expected: Any) -> dict[str, Any]:
    item: dict[str, Any] = {
        "itemId": f"it-{n:03d}", "stratum": "eval", "origin": "synthetic",
        "genRunId": f"run-{n % 4}",
    }
    if role == JOIN_IN:
        item.update(
            agentName="Pastel",
            priorTurns=[{"displayName": "Ana", "role": "user", "text": f"before {n}"}],
            newMessage={"displayName": "Bo", "role": "user", "text": f"message {n}"},
        )
    else:
        item.update(text=f"message {n}", priorTurn={"speaker": "Ana", "text": f"before {n}"})
    item["expected"] = expected
    return item


def _labels(role: str, sizes: dict[Any, int]) -> list[Any]:
    """Each true answer repeated its class size, in the answer space's order."""
    return [label for label, size in sizes.items() for _ in range(size)]


def _workspace(root: Path, role: str, expected: list[Any]) -> Path:
    """`test_runner_handoff`'s new-ability workspace with `expected` as its eval items."""
    pack_dir = ho._new_ability_workspace(root, role)
    rows = [_item(role, n, label) for n, label in enumerate(expected)]
    (pack_dir / "items.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return pack_dir


def _pack(root: Path):
    return load_pack(root / "packs" / base._PACK_ID)


def _skeleton(root: Path, role: str, *, minutes: int) -> RunResult:
    """A real candidate record on the workspace: every item decided by the model."""
    pack = _pack(root)
    responses = [ho._decided_as(role, 0.95) for _ in range(len(list(pack.iter_items())))]
    lms = ho._new_ability_stub(role, decide_responses=responses)
    lms.clock = ho._StubClock()
    moment = _T0 + timedelta(minutes=minutes)
    run, _ = base.run_pack(
        pack, base._cfg(sessionId=SESSION), lmstudio=lms, root=root, clock=lms.clock,
        now=lambda: moment,
    )
    return run


#: How each spec row is scored: `("ok", answer)` decided by the model; `("handoff", answer)` a
#: policy hand-off; `("parse", answer)` a hand-off whose reply did not parse; `("fail", call)` a
#: kept failed call (`"decision"`/`"handoff"`); `("over", answer)` an over-limit hand-off;
#: `("load", answer)` withheld for load.
def _scored(pack, row: dict[str, Any], spec: tuple[str, Any], *, ms: float) -> ItemResult:
    how, value = spec
    kw: dict[str, Any] = dict(decided_by="decision-model", probability=0.95, failed_call=None,
                              reason=None, input_tokens=None)
    outcome, answer, withheld = "pass", value, None
    if how == "handoff":
        kw.update(decided_by="handoff", probability=0.5)
    elif how == "parse":
        kw.update(decided_by="handoff", probability=0.5)
        outcome = "parse_failure"
    elif how == "fail":
        kw.update(decided_by=None, probability=None, failed_call=value)
        outcome, answer = "fail", None
    elif how == "over":
        kw.update(decided_by="handoff", probability=None, reason="over-limit")
    elif how == "load":
        withheld = "load"
    else:
        assert how == "ok", how
    timing = ItemTiming(wallClockMs=ms, calls=(), withheldFor=withheld)
    return agreement.score_answer(row, answer, outcome, timing, pack=pack, **kw)


def _replicate(
    root: Path, role: str, specs: dict[str, tuple[str, Any]] | None = None, *, minutes: int = 10,
    ms: float | Callable[[str], float] = 100.0, default: str = "agree",
) -> RunResult:
    """A candidate replicate: every item answered with its own label (`default="agree"`) or, with
    `default="no"`, `False` on join-in — unless `specs[itemId]` says otherwise — each `ms` long
    (or `ms(itemId)` long); aggregates re-derived as the runner derives them."""
    run = _skeleton(root, role, minutes=minutes)
    pack = _pack(root)
    specs = specs or {}
    ms_of = ms if callable(ms) else (lambda _item_id: ms)
    items = []
    for row in pack.iter_items():
        own = row["expected"] if default == "agree" else False
        spec = specs.get(row["itemId"], ("ok", own))
        items.append(_scored(pack, row, spec, ms=ms_of(row["itemId"])))
    return _reaggregate(pack, run, tuple(items))


def _reaggregate(pack, run: RunResult, items: tuple[ItemResult, ...]) -> RunResult:
    verdict = pack.manifest["metrics"]["verdictMetrics"]
    extra = runner._handoff_metrics(items, handoff_failure="fail-closed", verdict_metrics=verdict)
    aggregates = runner._with_decision_call_metrics(
        agreement.aggregate(items, pack=pack), items, extra=extra
    )
    return replace(run, items=items, aggregates=aggregates)


def _variant(run: RunResult, **fields) -> RunResult:
    """`run` with fingerprint fields replaced (and `basis`/`requestTimeoutSeconds` when given)."""
    top = {k: fields.pop(k) for k in ("basis", "requestTimeoutSeconds") if k in fields}
    return replace(run, fingerprint=replace(run.fingerprint, fields={
        **run.fingerprint.fields, **fields}), **top)


def _fresh_load(run: RunResult) -> RunResult:
    """The run as if its decision model was not resident at start: a load replicate (plan2 §3.6),
    as U7's unload before replicate 2 leaves it."""
    return _variant(run, residentModelsAtStart=[])


def _two_loads(root: Path, role: str, specs=None, specs_2=None, **kw) -> tuple[RunResult, ...]:
    first = _replicate(root, role, specs, minutes=10, **kw)
    second = _fresh_load(_replicate(root, role, specs_2 if specs_2 is not None else specs,
                                    minutes=20, **kw))
    return first, second


def _report(root: Path, *runs: RunResult, models: str = CAND) -> str:
    for run in runs:
        store(run, root)
    argv = ["compare", "--root", str(root), "--pack", base._PACK_ID, "--out", str(root / "r.md"),
            "--models", models, "--session", SESSION]
    assert cli.main(argv) == 0
    return (root / "r.md").read_text()


def _block(text: str, replicate: int = 1) -> str:
    return text.split(f"#### replicate {replicate} —")[1].split("####")[0]


def _class_lines(text: str, replicate: int = 1) -> list[str]:
    return [line for line in _block(text, replicate).splitlines() if line.startswith("  - agree")]


def _ids(role: str, label: Any, sizes: dict[Any, int], count: int) -> list[str]:
    """The first `count` itemIds of class `label`, in a workspace built from `sizes`."""
    expected = _labels(role, sizes)
    return [f"it-{n:03d}" for n, e in enumerate(expected) if e == label][:count]


#: AC-15's set (v1.3 §5 U4): 200 join-in items, 70/30 — 60 `yes`, 140 `no`.
_JOIN_70_30 = {True: 60, False: 140}
#: Note §Q5.4's worked example: 60 per label, 58, 56, 54, 53, 55 agreeing.
_MT_60 = dict.fromkeys(MESSAGE_TYPE_LABELS, 60)
_Q54_AGREEING = dict(zip(MESSAGE_TYPE_LABELS, (58, 56, 54, 53, 55)))
#: The note's own Wilson column for §Q5.4 (each label's `k/60`).
_Q54_WILSON = {
    "order": "[0.886, 0.991]", "question": "[0.841, 0.974]", "complaint": "[0.799, 0.953]",
    "escalation": "[0.778, 0.942]", "other": "[0.819, 0.964]",
}
_OVER = "**FR-10 per-class agreement, over 2 replicate(s) (2 load(s))"
_OVER_LATENCY = "**FR-10 mean latency, over 2 replicate(s) (2 load(s))"


# --- AC-15 and the per-class bar (v1.3 §5 U4; note §Q5.1) ---------------------------------------


def test_ac15_an_always_no_arm_on_a_70_30_set_does_not_meet_the_bar(tmp_path):
    """AC-15: the always-majority arm agrees on every `no` and on no `yes` — `agreeWhenYes = 0`,
    so FR-10 is not met though its pooled accuracy is 0.70. One class missing is enough."""
    _workspace(tmp_path, JOIN_IN, _labels(JOIN_IN, _JOIN_70_30))
    text = _report(tmp_path, *_two_loads(tmp_path, JOIN_IN, default="no"))
    assert _class_lines(text) == [
        "  - agreeWhenYes: 0/60 items [0.000, 0.060]; DE 1.00 (genRunId ICC) — not met",
        "  - agreeWhenNo: 140/140 items [0.973, 1.000]; DE 1.00 (genRunId ICC) — met",
    ]
    assert "- FR-10 per-class agreement (replicate 1): not met" in text
    assert "- FR-10 per-class agreement (replicate 2): not met" in text
    assert f"{_OVER}: not met**" in text
    assert f"{_OVER_LATENCY}: met**" in text


def test_an_arm_at_exactly_0_90_in_each_class_meets_the_bar_and_one_item_less_does_not(tmp_path):
    """v1.3 §5 U4: 54/60 and 126/140 are each exactly 0.90 — met (`10·k ≥ 9·n`), and with two
    loads met overall; replicate 2 one `yes` item worse (53/60) is `unstable — not met`."""
    _workspace(tmp_path, JOIN_IN, _labels(JOIN_IN, _JOIN_70_30))
    misses = {i: ("ok", False) for i in _ids(JOIN_IN, True, _JOIN_70_30, 6)}
    misses |= {i: ("ok", True) for i in _ids(JOIN_IN, False, _JOIN_70_30, 14)}
    text = _report(tmp_path, *_two_loads(tmp_path, JOIN_IN, misses))
    assert [line.split(" [")[0] + " — " + line.rsplit(" — ", 1)[1]
            for line in _class_lines(text)] == [
        "  - agreeWhenYes: 54/60 items — met", "  - agreeWhenNo: 126/140 items — met",
    ]
    assert f"{_OVER}: met**" in text

    worse = dict(misses) | {_ids(JOIN_IN, True, _JOIN_70_30, 7)[6]: ("ok", False)}
    root = tmp_path / "less"
    _workspace(root, JOIN_IN, _labels(JOIN_IN, _JOIN_70_30))
    text = _report(root, *_two_loads(root, JOIN_IN, misses, specs_2=worse))
    assert _class_lines(text, 2)[0].startswith("  - agreeWhenYes: 53/60 items [")
    assert _class_lines(text, 2)[0].endswith("— not met")
    assert "- FR-10 per-class agreement (replicate 2): not met" in text
    assert f"{_OVER}: unstable across replicates — not met**" in text


def test_note_q54_worked_example_misses_on_escalation_with_the_notes_intervals(tmp_path):
    """Note §Q5.4 (plan2 §5 U4): 58, 56, 54, 53, 55 of 60 — `escalation` not met, so FR-10 is
    not met. Every interval is the note's Wilson column, labelled per-class and not simultaneous,
    and the unit is message type's `item` (P21-n1). Each `order` miss is a kept hand-off failure:
    no label, a disagreement (plan2 §3.3), counted in `handoffCallFailures`."""
    _workspace(tmp_path, MESSAGE_TYPE, _labels(MESSAGE_TYPE, _MT_60))
    specs: dict[str, tuple[str, Any]] = {}
    for label, agreeing in _Q54_AGREEING.items():
        for item_id in _ids(MESSAGE_TYPE, label, _MT_60, 60)[agreeing:]:
            wrong = "order" if label == "other" else "other"
            specs[item_id] = ("fail", "handoff") if label == "order" else ("ok", wrong)
    text = _report(tmp_path, *_two_loads(tmp_path, MESSAGE_TYPE, specs))
    assert "per-class 95% Wilson, not simultaneous" in text
    assert _class_lines(text) == [
        f"  - agree[{label}]: {_Q54_AGREEING[label]}/60 items {_Q54_WILSON[label]}; "
        "DE 1.00 (genRunId ICC) — "
        + ("not met" if label == "escalation" else "met")
        for label in MESSAGE_TYPE_LABELS
    ]
    assert "handoffCallFailures 2/300" in _block(text)
    assert "- FR-10 per-class agreement (replicate 1): not met" in text
    assert f"{_OVER}: not met**" in text


# --- the class minimum and per-class `overLimit` (plan2 §3.6; plan3 §3.6) -------------------------


def test_every_class_prints_its_scored_n_beside_the_class_minimum_and_its_over_limit_count(
    tmp_path,
):
    """Each class prints `method minimum 52 (design 60)`; a class below 52 is printed, never
    refused, and still gets its verdict. `overLimit k/n` prints per class, `0/n` included; an
    over-limit item is scored on the hand-off LLM's answer (plan3 §4 U4)."""
    sizes = {True: 51, False: 149}
    _workspace(tmp_path, JOIN_IN, _labels(JOIN_IN, sizes))
    over = _ids(JOIN_IN, True, sizes, 2)
    specs = {over[0]: ("over", True), over[1]: ("over", False)}
    text = _report(tmp_path, *_two_loads(tmp_path, JOIN_IN, specs))
    block = _block(text)
    assert (
        "    - scored n 51 beside method minimum 52 (design 60) — below it: printed, never "
        "refused; overLimit 2/51"
    ) in block
    assert "    - scored n 149 beside method minimum 52 (design 60); overLimit 0/149" in block
    assert _class_lines(text)[0].startswith("  - agreeWhenYes: 50/51 items [")
    assert _class_lines(text)[0].endswith("— met")
    assert "overLimit 2/200" in block
    assert f"{_OVER}: met**" in text


def test_a_class_with_no_scored_item_is_not_assessable_and_never_met(tmp_path):
    """A class with n = 0 has no measured value (FR-11a): its line prints `0/0` and the class
    minimum, and the replicate is not assessable — never met, since FR-10 needs every class."""
    sizes = {label: 50 for label in MESSAGE_TYPE_LABELS if label != "escalation"}
    _workspace(tmp_path, MESSAGE_TYPE, _labels(MESSAGE_TYPE, sizes))
    text = _report(tmp_path, *_two_loads(tmp_path, MESSAGE_TYPE))
    lines = _class_lines(text)
    assert lines[3] == (
        "  - agree[escalation]: 0/0 items [—]; DE 1.00 (genRunId ICC) — not assessable (no scored "
        "eval item in this class)"
    )
    assert "    - scored n 0 beside method minimum 52 (design 60) — below it: printed, never " \
        "refused; overLimit 0/0" in text
    reason = "not assessable (no scored eval item in agree[escalation])"
    assert f"- FR-10 per-class agreement (replicate 1): {reason}" in text
    assert f"{_OVER}: {reason}**" in text


def test_a_missed_class_still_decides_not_met_beside_an_empty_one(tmp_path):
    sizes = {label: 50 for label in MESSAGE_TYPE_LABELS if label != "escalation"}
    _workspace(tmp_path, MESSAGE_TYPE, _labels(MESSAGE_TYPE, sizes))
    specs = {i: ("ok", "other") for i in _ids(MESSAGE_TYPE, "order", sizes, 6)}
    text = _report(tmp_path, *_two_loads(tmp_path, MESSAGE_TYPE, specs))
    assert "- FR-10 per-class agreement (replicate 1): not met" in text
    assert f"{_OVER}: not met**" in text


# --- scored n, latency, replicate states ---------------------------------------------------------


def test_scored_n_below_200_is_not_assessable_whatever_the_classes_say(tmp_path):
    _workspace(tmp_path, JOIN_IN, _labels(JOIN_IN, _JOIN_70_30))
    specs = {_ids(JOIN_IN, False, _JOIN_70_30, 1)[0]: ("load", False)}
    text = _report(tmp_path, *_two_loads(tmp_path, JOIN_IN, specs))
    block = _block(text)
    assert "- scored eval n: 199 of the pack's 200 eval items, beside FR-11's 200; dropped: " \
        "candidate load 1" in block
    reason = decision_stats.SCORED_N_BELOW_MIN
    assert all(line.endswith(f"— {reason}") for line in _class_lines(text))
    assert f"- FR-10 per-class agreement (replicate 1): {reason}" in block
    assert block.rstrip().splitlines()[-1].endswith(f"— {reason}")
    assert f"{_OVER}: {reason}**" in text and f"{_OVER_LATENCY}: {reason}**" in text


@pytest.mark.parametrize(("ms", "state"), [(350.0, "met"), (351.0, "not met")])
def test_the_mean_latency_bar_is_at_most_350_ms_on_the_measured_mean(tmp_path, ms, state):
    _workspace(tmp_path, JOIN_IN, _labels(JOIN_IN, _JOIN_70_30))
    text = _report(tmp_path, *_two_loads(tmp_path, JOIN_IN, ms=ms))
    line = next(line for line in _block(text).splitlines()
                if line.startswith("- FR-10 mean latency:"))
    assert line.startswith(f"- FR-10 mean latency: {ms:.0f} ms [")
    assert "over 200 scored eval items (percentile bootstrap 95%, B=200, seed " in line
    assert line.endswith(f"the bar is ≤ 350 ms on the measured mean — {state}")
    assert f"{_OVER_LATENCY}: {state}**" in text
    assert f"{_OVER}: met**" in text


def _skewed_ms(item_id: str) -> float:
    """Per-item wall clock that is neither constant nor symmetric in `itemId` order, so the seed
    and the value order both move the bootstrap interval (impl review P23-m1)."""
    n = int(item_id.removeprefix("it-"))
    return 40.0 + (n * n) % 401 + (300.0 if n % 7 == 0 else 0.0)


def test_the_mean_latency_interval_uses_the_packs_seed_over_values_in_item_id_order(tmp_path):
    """Note §Q4.2: `ȳ` and its percentile-bootstrap interval are over the scored items' wall
    clocks sorted by `itemId`, resampled with the seed `pack.json` declares (impl review P23-m1).
    The expected figures come from the pack file and `stats.mean_bootstrap_interval`, never from
    the report's own code path."""
    from modelbench import stats

    pack_dir = _workspace(tmp_path, JOIN_IN, _labels(JOIN_IN, _JOIN_70_30))
    seed = json.loads((pack_dir / "pack.json").read_text())["sampling"]["seed"]
    assert seed != 0
    text = _report(tmp_path, *_two_loads(tmp_path, JOIN_IN, ms=_skewed_ms))

    values = [_skewed_ms(i) for i in sorted(row["itemId"] for row in _pack(tmp_path).iter_items())]
    assert len(set(values)) > 1 and values != values[::-1]
    lo, hi = stats.mean_bootstrap_interval(
        values, B=200, seed=seed, levels=(stats.LEVEL_CI95_LO, stats.LEVEL_CI95_HI), support=None
    )
    line = next(line for line in _block(text).splitlines()
                if line.startswith("- FR-10 mean latency:"))
    assert line.startswith(
        f"- FR-10 mean latency: {sum(values) / len(values):.0f} ms [{lo:.0f}, {hi:.0f}] over 200 "
        f"scored eval items (percentile bootstrap 95%, B=200, seed {seed}); "
    )


def test_an_assumed_replicate_is_not_assessable_and_left_out(tmp_path):
    _workspace(tmp_path, JOIN_IN, _labels(JOIN_IN, _JOIN_70_30))
    first, second = _two_loads(tmp_path, JOIN_IN)
    assumed = replace(_replicate(tmp_path, JOIN_IN, minutes=15), basis="assumed")
    text = _report(tmp_path, first, assumed, second)
    assert "- FR-10 per-class agreement (replicate 2): not assessable (load changed mid-run)" \
        in text
    assert all(line.endswith("— not assessable (load changed mid-run)")
               for line in _class_lines(text, 2))
    assert "**FR-10 per-class agreement, over 3 replicate(s) (2 load(s)): met**" in text
    vs = [line.split(":")[0] for line in text.splitlines() if " vs replicate " in line]
    assert vs == ["- replicate 3 vs replicate 1"]


def test_one_load_alone_is_not_assessable(tmp_path):
    _workspace(tmp_path, JOIN_IN, _labels(JOIN_IN, _JOIN_70_30))
    text = _report(tmp_path, _replicate(tmp_path, JOIN_IN, minutes=10))
    assert "**FR-10 per-class agreement, over 1 replicate(s) (1 load(s)): not assessable (only " \
        "1 load replicate(s) (FR-11a needs 2))**" in text


def test_an_unreadable_replicate_is_not_assessable_never_a_crash(tmp_path):
    """A chat call withheld for `timeout` is not a new-ability candidate record (a new ability
    keeps its failed calls), so `agreement_items` refuses it; the bar says so."""
    _workspace(tmp_path, JOIN_IN, _labels(JOIN_IN, _JOIN_70_30))
    first, second = _two_loads(tmp_path, JOIN_IN)
    items = tuple(
        replace(i, scoreable={}, counts={}, outcome="fail",
                timing=replace(i.timing, wallClockMs=None, withheldFor="timeout"))
        if i.itemId == "it-005" else i
        for i in second.items
    )
    text = _report(tmp_path, first, _reaggregate(_pack(tmp_path), second, items))
    assert "- FR-10 per-class agreement (replicate 2): not assessable (cannot read this " \
        "replicate's items: agreement_items: item 'it-005' is withheld for 'timeout'" in text


def test_an_inconsistent_replicate_makes_its_groups_bars_not_assessable(tmp_path):
    _workspace(tmp_path, JOIN_IN, _labels(JOIN_IN, _JOIN_70_30))
    first, second = _two_loads(tmp_path, JOIN_IN)
    items = tuple(replace(i, scoreable={}, counts={}) if i.itemId == "it-050" else i
                  for i in second.items)
    text = _report(tmp_path, first, replace(second, items=items))
    assert "- FR-10 per-class agreement (replicate 2): not assessable (this replicate's " \
        "aggregates disagree" in text
    assert f"{_OVER}: not assessable (replicate 2: aggregates disagree with items)**" in text


# --- comparison-wide blocking, above the bar-kind split (impl review Pass 18) --------------------


def test_a_record_item_the_pack_does_not_hold_is_not_assessable_never_a_crash(tmp_path):
    """Impl review P26-m1: a stored replicate whose item id is not one of the pack's eval items
    (its own content hash intact) reads as not assessable, as FR-1's `pair_items` reads it —
    never an uncaught error out of `compare` (the closed exit-code contract)."""
    _workspace(tmp_path, JOIN_IN, _labels(JOIN_IN, _JOIN_70_30))
    first, second = _two_loads(tmp_path, JOIN_IN)
    items = tuple(
        replace(i, itemId="ghost-item") if i.itemId == "it-000" else i for i in second.items
    )
    text = _report(tmp_path, first, _reaggregate(_pack(tmp_path), second, items))
    assert "- FR-10 per-class agreement (replicate 2): not assessable (cannot read this " \
        "replicate's items: agreement_items: no eval stratum in the pack for items " \
        "['ghost-item']" in text
    assert "- FR-10 per-class agreement (replicate 1): met" in text


def _assert_every_fr10_bar(text: str, reason: str) -> None:
    for i in (1, 2):
        assert f"- FR-10 per-class agreement (replicate {i}): {reason}" in text
    assert f"{_OVER}: {reason}**" in text and f"{_OVER_LATENCY}: {reason}**" in text
    assert "): met" not in text and "— met" not in text


def test_equal_timeouts_off_the_pin_make_every_fr10_bar_not_assessable(tmp_path):
    _workspace(tmp_path, JOIN_IN, _labels(JOIN_IN, _JOIN_70_30))
    runs = [_variant(r, requestTimeoutSeconds=120.0) for r in _two_loads(tmp_path, JOIN_IN)]
    reason = f"not assessable (chat timeout ≠ {decisions.DECISION_POINT_CHAT_TIMEOUT_SECONDS} s)"
    _assert_every_fr10_bar(_report(tmp_path, *runs), reason)


def test_a_replicate_on_other_pack_content_blocks_every_fr10_bar(tmp_path):
    _workspace(tmp_path, JOIN_IN, _labels(JOIN_IN, _JOIN_70_30))
    first, second = _two_loads(tmp_path, JOIN_IN)
    second = _variant(second, packContentHash="0" * 64)
    text = _report(tmp_path, first, second)
    reason = report._pack_content_reason(
        report.DecisionPointPack.from_pack(_pack(tmp_path)), None,
        {CAND: report.ReplicateGroup(runs=(first, second), kind=("load", "load"), loadCount=2,
                                     k=0, earlierCandidates=())},
    )
    assert reason is not None and reason.startswith("not assessable (arms ran on different")
    _assert_every_fr10_bar(text, reason)
    assert "- scored eval n:" not in text


def _reference_arm(root: Path) -> RunResult:
    """U7's reference chat arm: a real hand-off-LLM run on the workspace's pack."""
    lms = ho._new_ability_stub(JOIN_IN, initially_loaded=(QWEN,))
    chat, _ = base.run_pack(
        _pack(root), base._cfg(candidateId=None, modelKey=QWEN, sessionId=SESSION), lmstudio=lms,
        root=root, clock=base._Clock(0.25), now=lambda: _T0,
    )
    return chat


def test_fr10_is_single_arm_and_a_reference_arms_own_results_are_never_read(tmp_path):
    """No incumbent exists on a new-ability point (`incumbent: null`), so `no incumbent` never
    blocks FR-10; U7's reference chat arm is named, and its own aggregates never reach a bar."""
    _workspace(tmp_path, JOIN_IN, _labels(JOIN_IN, _JOIN_70_30))
    runs = _two_loads(tmp_path, JOIN_IN)
    text = _report(tmp_path, *runs)
    assert "Reference chat arm: none (FR-10 is single-arm" in text
    assert "no incumbent run" not in text
    root = tmp_path / "ref"
    _workspace(root, JOIN_IN, _labels(JOIN_IN, _JOIN_70_30))
    chat = _reference_arm(root)
    # An inconsistent reference arm: `compare` excludes it from the generic tables, and no bar
    # reads its results, so FR-10 is unaffected.
    chat = replace(chat, items=tuple(
        replace(i, scoreable={}, counts={}) if i.itemId == "it-050" else i for i in chat.items))
    runs = _two_loads(root, JOIN_IN)
    text = _report(root, chat, *runs, models=f"{QWEN},{CAND}")
    assert f"Reference chat arm: `{QWEN}` — run `{chat.runId}`" in text
    assert f"`{chat.runId}` — aggregates disagree with items" in text
    assert f"{_OVER}: met**" in text


def test_a_reference_arm_on_other_pack_content_blocks_every_fr10_bar(tmp_path):
    """The reference arm's pack content still applies (impl review P23-m2, teco's call): a
    reference run on other content can only withhold a verdict, and it keeps U7's session on one
    pack content. Every FR-10 bar carries the different-content reason."""
    _workspace(tmp_path, JOIN_IN, _labels(JOIN_IN, _JOIN_70_30))
    chat = _variant(_reference_arm(tmp_path), packContentHash="0" * 64)
    first, second = _two_loads(tmp_path, JOIN_IN)
    text = _report(tmp_path, chat, first, second, models=f"{QWEN},{CAND}")
    reason = report._pack_content_reason(
        report.DecisionPointPack.from_pack(_pack(tmp_path)), chat,
        {CAND: report.ReplicateGroup(runs=(first, second), kind=("load", "load"), loadCount=2,
                                     k=0, earlierCandidates=())},
    )
    assert reason is not None and reason.startswith("not assessable (arms ran on different")
    assert "00000000" in reason
    assert (f"Reference chat arm: `{QWEN}` — run `{chat.runId}` (FR-10 is single-arm: no "
            "incumbent, and no bar reads this run's results; its timeout and pack content still "
            "apply).") in text
    _assert_every_fr10_bar(text, reason)


# --- the replicate disagreement line (data-scientist ruling on plan2 §3.6; note §Q6) -------------


def _mt_pack(tmp_path):
    root = tmp_path / "seven"
    _workspace(root, MESSAGE_TYPE, ["order"] * 7)
    return _pack(root)


def _rows(pack) -> dict[str, dict[str, Any]]:
    return {row["itemId"]: row for row in pack.iter_items()}


def test_the_rulings_seven_item_fixture(tmp_path):
    """The ruling's fixture (message type): a order/order; b None/None; c None/complaint;
    d order/question; e None (parse failure)/None; f other/failedCall=handoff;
    g withheldFor=load/order. Excluding `None` would give 1/2; `None` always disagreeing 4/5;
    exclusion on `answer is None` admits f."""
    pack = _mt_pack(tmp_path)
    rows = list(_rows(pack).values())
    first_specs = [("ok", "order"), ("ok", None), ("ok", None), ("ok", "order"),
                   ("parse", None), ("ok", "other"), ("load", "order")]
    other_specs = [("ok", "order"), ("ok", None), ("ok", "complaint"), ("ok", "question"),
                   ("ok", None), ("fail", "handoff"), ("ok", "order")]
    first = [_scored(pack, r, s, ms=10.0) for r, s in zip(rows, first_specs, strict=True)]
    other = [_scored(pack, r, s, ms=10.0) for r, s in zip(rows, other_specs, strict=True)]
    got = report._answer_agreement(first, other)
    assert (got.n, got.answer_differs) == (5, 2)
    assert got.no_label_in_both == 2
    assert (got.failures, got.one_sided_withholds, got.two_sided_withholds) == (1, 1, 0)
    # e's first side was handed off (a parse failure), so `decidedBy` differs there alone.
    assert got.decided_by_differs == 1


def test_a_join_in_parse_failure_false_agrees_with_a_passed_false(tmp_path):
    root = tmp_path / "ji"
    _workspace(root, JOIN_IN, [True, False])
    pack = _pack(root)
    rows = list(_rows(pack).values())
    first = [_scored(pack, rows[0], ("parse", False), ms=1.0),
             _scored(pack, rows[1], ("ok", False), ms=1.0)]
    other = [_scored(pack, rows[0], ("handoff", False), ms=1.0),
             _scored(pack, rows[1], ("ok", False), ms=1.0)]
    assert first[0].outcome == "parse_failure" and other[0].outcome == "pass"
    got = report._answer_agreement(first, other)
    assert (got.n, got.answer_differs, got.decided_by_differs, got.failures) == (2, 0, 0, 0)


def test_a_two_sided_withhold_is_counted_apart_from_a_one_sided_one(tmp_path):
    pack = _mt_pack(tmp_path)
    rows = list(_rows(pack).values())[:2]
    both = [_scored(pack, r, ("load", "order"), ms=1.0) for r in rows]
    got = report._answer_agreement(both, both)
    assert (got.n, got.one_sided_withholds, got.two_sided_withholds) == (0, 0, 2)


def test_the_disagreement_line_prints_no_label_in_both_on_message_type_only(tmp_path):
    sizes = dict.fromkeys(MESSAGE_TYPE_LABELS, 40)
    _workspace(tmp_path, MESSAGE_TYPE, _labels(MESSAGE_TYPE, sizes))
    orders = _ids(MESSAGE_TYPE, "order", sizes, 4)
    specs_1 = {orders[0]: ("ok", None), orders[1]: ("ok", None), orders[2]: ("fail", "decision")}
    specs_2 = {orders[0]: ("ok", None), orders[1]: ("ok", "question"),
               orders[3]: ("load", "order")}
    text = _report(tmp_path, *_two_loads(tmp_path, MESSAGE_TYPE, specs_1, specs_2=specs_2))
    line = next(line for line in text.splitlines() if line.startswith("- replicate 2 vs"))
    assert line.startswith("- replicate 2 vs replicate 1: final answer differs on 1/198 [")
    assert "decidedBy on 0/198 [0.000, 0.019]; no label in both: 1; beside it, failures: 1, " \
        "one-sided withholds: 1, two-sided withholds: 0" in line

    root = tmp_path / "ji"
    _workspace(root, JOIN_IN, _labels(JOIN_IN, _JOIN_70_30))
    text = _report(root, *_two_loads(root, JOIN_IN))
    line = next(line for line in text.splitlines() if line.startswith("- replicate 2 vs"))
    assert "no label in both" not in line
    assert line.endswith("decidedBy on 0/200 [0.000, 0.019]; beside it, failures: 0, one-sided "
                         "withholds: 0, two-sided withholds: 0")


# --- the Wilson seam for U3e.2 (plan4 §3.9) -------------------------------------------------------


def test_the_class_interval_takes_the_design_effect_as_k_over_de_of_n_over_de():
    """Plan4 §3.9: FR-10's Wilson interval at `k/DE` of `n/DE`. At DE = 1.0 (every run today) it
    is the plain Wilson interval; a larger DE widens it."""
    from modelbench import stats

    assert report._fr10_class_interval(54, 60, design_effect=1.0) == stats.wilson_interval(54, 60)
    wide = report._fr10_class_interval(54, 60, design_effect=2.0)
    assert wide == stats.wilson_interval(27.0, 30.0)
    assert wide[1] - wide[0] > (lambda lo_hi: lo_hi[1] - lo_hi[0])(stats.wilson_interval(54, 60))


# --- the per-class design effect (plan4 §3.9, §5 U3e.2; note §Q6 "Design effect") -----------------

_DE_TEXT = r"; DE (\d+\.\d\d) \(genRunId ICC\)"


def _regroup(root: Path, gen_run_ids: dict[str, str]) -> None:
    """Rewrites the pack's `genRunId`s before any run, so every record carries the new content."""
    path = root / "packs" / base._PACK_ID / "items.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows = [{**r, "genRunId": gen_run_ids.get(r["itemId"], r["genRunId"])} for r in rows]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_a_class_whose_misses_share_one_run_widens_its_interval_at_k_over_de(tmp_path):
    """Plan4 §5 U3e.2: the six `yes` misses (54/60, met) spread across the four runs by default;
    regrouped so they fill one eight-item run, the class's ICC is high — `DE > 1` and a wider
    Wilson interval at `k/DE` of `n/DE`, with the same `k/n` and decision (note §Q6 "Scope")."""
    import re

    misses = {i: ("ok", False) for i in _ids(JOIN_IN, True, _JOIN_70_30, 6)}
    lines = {}
    for name, regroup in (("spread", {}), ("clustered", dict.fromkeys(
            _ids(JOIN_IN, True, _JOIN_70_30, 8), "hot"))):
        root = tmp_path / name
        _workspace(root, JOIN_IN, _labels(JOIN_IN, _JOIN_70_30))
        _regroup(root, regroup)
        lines[name] = _class_lines(_report(root, *_two_loads(root, JOIN_IN, misses)))[0]

    spread, clustered = lines["spread"], lines["clustered"]
    de = {k: float(re.search(_DE_TEXT, v).group(1)) for k, v in lines.items()}
    assert de["clustered"] > 1.0 and de["clustered"] > de["spread"]
    lo_s, hi_s = (float(x) for x in re.search(r"\[([\d.]+), ([\d.]+)\]", spread).groups())
    lo_c, hi_c = (float(x) for x in re.search(r"\[([\d.]+), ([\d.]+)\]", clustered).groups())
    assert hi_c - lo_c > hi_s - lo_s
    exact = report._fr10_class_interval(54, 60, design_effect=de["clustered"])
    assert abs(lo_c - exact[0]) < 0.006
    for line in (spread, clustered):
        assert line.startswith("  - agreeWhenYes: 54/60 items [")
        assert line.endswith("— met")
