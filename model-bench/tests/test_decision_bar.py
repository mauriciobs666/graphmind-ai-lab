"""typed-decision-models U3c.2: `compare` on a decision-point pack — `cli._replicate_groups`, the
arm rules and refusals (plan2 §3.6), and `report.py`'s bar section, speed-up side (v1.3 §3.5–§3.6;
plan2 §3.6; plan3 §3.6). Every figure printed is `decision_stats`' (the method note's); these
tests pin what is composed, printed and refused.

Every run below is a real `run_pack` record, driven offline by `test_runner_decisions`' stub on
the fixture guard pack padded to the contract (200 `eval` items + 1 `side-check`), and written
under `tmp_path` — never under `model-bench/results/runs/` (AGENTS, the P11-m2 ruling).
"""

from __future__ import annotations

import json
import re
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import test_runner_decisions as base

from modelbench import cli, decision_stats, stats
from modelbench.lmstudio import ChatResult, ResidentModel
from modelbench.packs import load_pack
from modelbench.report import ReplicateGroup
from modelbench.results import RunResult, load_history, store

QWEN = "qwen/qwen3-4b-2507"
CAND = base._CANDIDATE_ID
SESSION = "s-bar"
_QWEN_RESIDENT = [ResidentModel(id=QWEN, state="loaded")]
_T0 = datetime(2026, 10, 31, 9, 0, 0, tzinfo=timezone.utc)
_FA_U = "falseAdvanceRateByUnderstanding"
_FA_T = "falseAdvanceRateByTurns"


@pytest.fixture(autouse=True)
def _fast_bootstrap(monkeypatch):
    """B is a pinned constant (plan2 §3.6); the report reads it at call time, so every test but
    the one that pins the printed B resamples less. No interval below is asserted at this B."""
    monkeypatch.setattr(decision_stats, "BOOTSTRAP_B", 200)


# --- building real records ------------------------------------------------------------------------


def _root(tmp_path: Path) -> Path:
    base._workspace(tmp_path, contract_valid=True)
    return tmp_path


def _rows(root: Path) -> list[dict]:
    path = root / "packs" / base._PACK_ID / "items.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def _at(minutes: int):
    moment = _T0 + timedelta(minutes=minutes)
    return lambda: moment


class _ChatStub(base.DecisionStub):
    """The incumbent chat arm: one reply per item, in pack order, from `advances`, each timed
    at `ms` (a chat item's wall clock is its call's)."""

    def __init__(self, advances: list[bool], *, ms: float, **kw) -> None:
        super().__init__(**kw)
        self._replies = list(advances)
        self._ms = ms

    def chat(self, messages, *, model, temperature, max_tokens, timeout_s, tools=None):
        self.chat_calls.append({"model": model, "timeout_s": timeout_s})
        advanced = self._replies.pop(0)
        return ChatResult(
            message={"content": json.dumps({"decision": advanced})}, tool_calls=(),
            toolCallForm="prose", stats=None, model_info=None, runtime=None, usage=None,
            wallClockMs=self._ms,
        )


def _answers(root: Path, flips: set[str] = frozenset()) -> list[bool]:
    """Each item's correct answer (advance iff its label is true), flipped on `flips`."""
    return [bool(row["expected"]) != (row["itemId"] in flips) for row in _rows(root)]


def _incumbent(root: Path, *, flips=frozenset(), minutes=0, session=SESSION, ms=400.0, **cfg):
    n = len(_rows(root))
    lms = _ChatStub(
        _answers(root, flips), ms=ms, residency_sequence=[list(_QWEN_RESIDENT)] * (n + 10),
        initially_loaded=(),
    )
    run, _ = base.run_pack(
        load_pack(root / "packs" / base._PACK_ID),
        base._cfg(candidateId=None, modelKey=QWEN, sessionId=session, **cfg),
        lmstudio=lms, root=root, clock=base._Clock(0.25), now=_at(minutes),
    )
    return run


def _candidate(
    root: Path,
    *,
    flips=frozenset(),
    fail: set[str] = frozenset(),
    minutes=10,
    session=SESSION,
    step=0.1,
    resident_at_start=False,
    recheck_differs=False,
    **cfg,
):
    """A `decisions`-surface candidate run: p = 0.9 advances, 0.1 suspends (the band is empty at
    0.8); `fail` items raise a decision-call timeout (a kept failure, plan2 §3.3). With
    `resident_at_start` laya is resident at the start snapshot, else the runner loads it."""
    from modelbench.lmstudio import LMStudioCallTimeout

    responses = []
    for row, advance in zip(_rows(root), _answers(root, flips), strict=True):
        if row["itemId"] in fail:
            responses.append(LMStudioCallTimeout("decision call timed out"))
        else:
            responses.append(base._decided(0.9 if advance else 0.1))
    if recheck_differs:
        responses.append(base._decided(0.123))
    start = [] if not resident_at_start else list(base._LAYA_RESIDENT)
    lms = base.DecisionStub(
        decide_responses=responses, residency_sequence=[start],
        initially_loaded=("laya",) if resident_at_start else (),
    )
    run, _ = base.run_pack(
        load_pack(root / "packs" / base._PACK_ID),
        base._cfg(sessionId=session, **cfg),
        lmstudio=lms, root=root, clock=base._Clock(step), now=_at(minutes),
    )
    return run


def _store_all(root: Path, *runs: RunResult) -> None:
    for run in runs:
        store(run, root)


def _compare(root: Path, *extra: str, models: str | None = f"{QWEN},{CAND}", session=SESSION):
    argv = ["compare", "--root", str(root), "--pack", base._PACK_ID, "--out", str(root / "r.md")]
    if models is not None:
        argv += ["--models", models]
    if session is not None:
        argv += ["--session", session]
    return cli.main(argv + list(extra))


def _report(root: Path, *extra: str, **kw) -> str:
    assert _compare(root, *extra, **kw) == 0
    return (root / "r.md").read_text()


def _variant(run: RunResult, **fields) -> RunResult:
    """`run` with fingerprint fields replaced (and `runId`/`sessionId`/`basis` when given)."""
    top = {k: fields.pop(k) for k in ("runId", "sessionId", "basis") if k in fields}
    return replace(run, fingerprint=replace(run.fingerprint, fields={
        **run.fingerprint.fields, **fields}), **top)


# --- arm rules and refusals (plan2 §3.6) ----------------------------------------------------------


@pytest.fixture(scope="module")
def stored(tmp_path_factory):
    """An incumbent and two load replicates of the candidate, stored in one session."""
    root = _root(tmp_path_factory.mktemp("bar"))
    _store_all(root, _incumbent(root), _candidate(root, minutes=10), _candidate(root, minutes=20))
    return root


@pytest.mark.parametrize(
    ("kwargs", "extra", "needle"),
    [
        ({"session": None}, (), "--session"),
        ({"models": None}, (), "--models"),
        ({}, ("--negative-control",), "--negative-control"),
        ({"models": f"{QWEN},{CAND},{CAND}"}, (), "twice"),
    ],
)
def test_compare_on_a_decision_point_pack_refuses_a_usage_error(
    stored, capsys, kwargs, extra, needle
):
    assert _compare(stored, *extra, **kwargs) == cli.EXIT_USAGE
    assert needle in capsys.readouterr().err
    assert not (stored / "r.md").exists()


def test_rank_is_refused_on_a_decision_point_pack(stored, capsys):
    code = cli.main(["rank", "--root", str(stored), "--pack", base._PACK_ID])
    assert code == cli.EXIT_USAGE
    assert "rank is refused on decision-point pack" in capsys.readouterr().err


def test_a_key_with_no_run_in_the_session_is_unknown(stored, capsys):
    assert _compare(stored, models=f"{QWEN},nope") == cli.EXIT_USAGE
    assert "'nope'" in capsys.readouterr().err


def test_the_decision_point_test_keys_on_the_key_alone():
    # The refusals key on the `decisionPoint` key alone: `_select_arms` is unchanged elsewhere.
    assert not cli._is_decision_point({"packId": "x"})
    assert cli._is_decision_point({"decisionPoint": None})


def _runs(root: Path) -> list[RunResult]:
    valid, invalid = load_history(root, packId=base._PACK_ID)
    assert invalid == []
    return valid


def _arms(runs, models=f"{QWEN},{CAND}", manifest=None):
    manifest = manifest or {"decisionPoint": {"incumbent": QWEN}}
    return cli._decision_point_arms(
        runs, manifest=manifest, models=models, session=SESSION, negative_control=False
    )


def test_two_incumbent_runs_in_one_session_refuse(stored):
    runs = _runs(stored)
    chat = next(r for r in runs if r.armKey == QWEN)
    with pytest.raises(cli.DecisionPointUsageError, match="2 runs of the chat arm"):
        _arms([*runs, _variant(chat, runId="second-incumbent")])


def test_more_than_one_chat_arm_refuses(stored):
    runs = _runs(stored)
    chat = next(r for r in runs if r.armKey == QWEN)
    other = _variant(chat, runId="other-chat", modelKey="other/model")
    with pytest.raises(cli.DecisionPointUsageError, match="2 chat arms"):
        _arms([*runs, other], models=f"{QWEN},other/model,{CAND}")


def test_a_chat_arm_that_is_not_the_incumbent_refuses(stored):
    runs = _runs(stored)
    chat = next(r for r in runs if r.armKey == QWEN)
    other = _variant(chat, runId="other-chat", modelKey="other/model")
    with pytest.raises(cli.DecisionPointUsageError, match="not this speed-up pack's incumbent"):
        _arms([*runs, other], models=f"other/model,{CAND}")


def test_two_content_hashes_in_one_replicate_group_refuse(stored):
    runs = _runs(stored)
    cand = [r for r in runs if r.armKey == CAND]
    edited = _variant(cand[1], candidateContentHash="e" * 64)
    with pytest.raises(cli.DecisionPointUsageError, match="2 content hashes"):
        cli._replicate_groups([cand[0], edited], session=SESSION, arm_keys=[CAND])


def test_arms_at_different_chat_timeouts_refuse(stored):
    runs = _runs(stored)
    changed = [replace(r, requestTimeoutSeconds=120.0) if r.armKey == CAND else r for r in runs]
    with pytest.raises(cli.DecisionPointUsageError, match="120 s, 180 s"):
        _arms(changed)


def test_the_arm_set_is_the_chat_run_and_every_replicate_in_start_order(stored):
    chat, groups = _arms(list(reversed(_runs(stored))))
    assert chat.armKey == QWEN
    assert list(groups) == [CAND]
    starts = [r.fingerprint.get("startedAt") for r in groups[CAND].runs]
    assert starts == sorted(starts) and len(starts) == 2


def test_a_candidate_alone_has_no_incumbent_and_its_bars_say_so(stored):
    text = _report(stored, models=CAND)
    assert "Incumbent: none in this comparison (the pack names `qwen/qwen3-4b-2507`)." in text
    assert "FR-1 (replicate 1): not assessable (no incumbent run in this comparison)" in text
    assert "(2 load(s)): not assessable (no incumbent run in this comparison)**" in text


# --- replicate kinds and K (plan2 §3.6, v1.3 §3.5) ------------------------------------------------


def test_the_u7_sequence_classifies_two_fresh_loads(tmp_path):
    # Plan3 §5 U3c: the incumbent, replicate 1 (laya absent, the runner loads), a fake unload,
    # replicate 2 (laya absent, the runner loads) gives (load, load) and loadCount 2.
    root = _root(tmp_path)
    runs = [_incumbent(root), _candidate(root, minutes=10), _candidate(root, minutes=20)]
    groups = cli._replicate_groups(runs, session=SESSION, arm_keys=[CAND])
    assert groups[CAND].kind == ("load", "load") and groups[CAND].loadCount == 2


def test_laya_left_resident_before_replicate_2_is_a_same_load(tmp_path):
    root = _root(tmp_path)
    runs = [_candidate(root, minutes=10), _candidate(root, minutes=20, resident_at_start=True)]
    group = cli._replicate_groups(runs, session=SESSION, arm_keys=[CAND])[CAND]
    assert group.kind == ("load", "same-load") and group.loadCount == 1


def test_the_first_non_assumed_run_is_a_load_even_when_resident_at_start(tmp_path):
    root = _root(tmp_path)
    first = _candidate(root, minutes=10, resident_at_start=True)
    group = cli._replicate_groups([first], session=SESSION, arm_keys=[CAND])[CAND]
    assert group.kind == ("load",)


def test_an_assumed_run_is_neither_a_load_nor_the_groups_first_load(tmp_path):
    root = _root(tmp_path)
    assumed = _candidate(root, minutes=10, recheck_differs=True)
    assert assumed.basis == "assumed"
    later = _candidate(root, minutes=20, resident_at_start=True)
    group = cli._replicate_groups([assumed, later], session=SESSION, arm_keys=[CAND])[CAND]
    assert group.kind == ("assumed", "load") and group.loadCount == 1


def _k_runs(tmp_path):
    root = _root(tmp_path)
    return root, _candidate(root, minutes=10)


def _cand(run, n, *, minutes, cid=CAND, content=None, eval_set=None, session=SESSION, **extra):
    stamp = (_T0 + timedelta(minutes=minutes)).strftime("%Y-%m-%dT%H:%M:%SZ")
    fields = {"startedAt": stamp, "candidateId": cid, **extra}
    if content is not None:
        fields["candidateContentHash"] = content
    if eval_set is not None:
        fields["evalSetHash"] = eval_set
    return _variant(run, runId=f"run-{n}", sessionId=session, **fields)


def test_interleaved_replicates_take_one_k_the_earliest_runs(tmp_path):
    # Plan2 §5 U3c: D1, H1, D2, H2 gives D one K, equal to D1's, on both replicates.
    _, run = _k_runs(tmp_path)
    d1 = _cand(run, 1, minutes=1, cid="D", content="d" * 64)
    h1 = _cand(run, 2, minutes=2, cid="H", content="h" * 64)
    d2 = _cand(run, 3, minutes=3, cid="D", content="d" * 64)
    h2 = _cand(run, 4, minutes=4, cid="H", content="h" * 64)
    groups = cli._replicate_groups([d1, h1, d2, h2], session=SESSION, arm_keys=["D", "H"])
    assert (groups["D"].k, groups["D"].earlierCandidates) == (0, ())
    assert (groups["H"].k, groups["H"].earlierCandidates) == (1, ("d" * 64,))


def test_k_counts_an_edit_in_place_as_a_second_candidate_and_never_the_run_itself(tmp_path):
    _, run = _k_runs(tmp_path)
    old = _cand(run, 1, minutes=1, content="a" * 64, session="earlier")
    same = _cand(run, 2, minutes=2, content="b" * 64, session="earlier")
    this = _cand(run, 3, minutes=3, content="b" * 64)
    group = cli._replicate_groups([old, same, this], session=SESSION, arm_keys=[CAND])[CAND]
    assert group.earlierCandidates == ("a" * 64,)


def test_k_keeps_a_reimport_and_resets_on_a_changed_label(tmp_path):
    # A re-import under a new packVersion with the same items and labels keeps `evalSetHash`, so
    # K counts across versions; a changed label is a new evaluation set.
    _, run = _k_runs(tmp_path)
    same_set = _cand(run, 1, minutes=1, content="a" * 64, session="v0", packVersion="0.0.9")
    other_set = _cand(run, 2, minutes=2, content="c" * 64, session="v0", eval_set="f" * 64)
    this = _cand(run, 3, minutes=3, content="b" * 64)
    group = cli._replicate_groups([same_set, other_set, this], session=SESSION, arm_keys=[CAND])
    assert group[CAND].earlierCandidates == ("a" * 64,)


def test_a_same_started_at_run_is_not_earlier(tmp_path):
    _, run = _k_runs(tmp_path)
    tie = _cand(run, 1, minutes=3, content="a" * 64, session="other")
    this = _cand(run, 2, minutes=3, content="b" * 64)
    assert cli._replicate_groups([tie, this], session=SESSION, arm_keys=[CAND])[CAND].k == 0


def test_the_reuse_line_prints_k_and_the_hashes_and_the_reuse_adjusted_limit(tmp_path):
    root = _root(tmp_path)
    earlier = _cand(_candidate(root, minutes=5), 0, minutes=5, content="9" * 64, session="old")
    _store_all(root, earlier, _incumbent(root), _candidate(root, minutes=10))
    text = _report(root)
    assert "- Candidates already run on this evaluation set: 1 (99999999)" in text
    assert "reuse-adjusted at K = 1: ≥ " in text


def test_the_k_line_says_when_invalid_records_were_left_out_of_it(stored):
    # K reads valid records only (impl review P18-n1); an excluded one may have seen this
    # evaluation set, so the line says K is over valid records.
    line = "- Candidates already run on this evaluation set: 0 (none)"
    assert f"{line}\n" in _report(stored)
    broken = stored / "results" / "runs" / "broken.json"
    broken.write_text("{not json")
    try:
        text = _report(stored)
    finally:
        broken.unlink()
    assert f"{line} (valid records only; 1 invalid excluded)\n" in text


def test_replicate_groups_refuse_a_key_with_no_run_in_the_session(stored):
    with pytest.raises(cli.UnknownModelKey, match="'nope'"):
        cli._replicate_groups(_runs(stored), session=SESSION, arm_keys=[CAND, "nope"])
    with pytest.raises(cli.UnknownModelKey):
        cli._replicate_groups(_runs(stored), session="other", arm_keys=[CAND])


def test_a_replicate_group_refuses_counts_that_disagree_with_its_kinds(stored):
    run = _runs(stored)[0]
    with pytest.raises(ValueError):
        ReplicateGroup(runs=(run,), kind=("load",), loadCount=2, k=0, earlierCandidates=())


# --- the bar section (v1.3 §3.5–§3.6; plan2 §3.6; plan3 §3.6) -------------------------------------


def _result_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line and not line.startswith("#")]


def test_the_report_states_the_bar_lines_in_order(stored):
    text = _report(stored)
    assert _result_lines(text)[0] == (
        "Labels: anthropic/fixture-labeler, inputs generated by anthropic/fixture-generator, "
        "unreviewed (FR-5). Results measure agreement with this labeler on synthetic inputs, not "
        "verified correctness."
    )
    assert (
        "- fit manifest: 2 inputs (training/calibration/cut-off); 0 overlap with this pack's 201 "
        "items (checked at run time, sha256 over canonical input fields)"
    ) in text
    assert "- Candidates already run on this evaluation set: 0 (none)" in text
    assert "- fit outcome: within-margin (quality target D_c<=0)" in text
    assert (
        "- scored eval n: 200 of the pack's 200 eval items, beside FR-11's 200; none dropped"
    ) in text
    order = [
        text.index(s)
        for s in ("Labels:", "fit manifest:", "Candidates already run", "fit outcome:",
                  "#### replicate 1", "scored eval n:", "FR-1 parity, per cell", "FR-2 speed-up:",
                  "Side check —", "#### replicate 2", "#### Across the replicates")
    ]
    assert order == sorted(order)


def _cell_lines(text: str, replicate: int = 1) -> list[str]:
    block = text.split(f"#### replicate {replicate} —")[1].split("####")[0]
    return [line for line in block.splitlines() if line.startswith("  - falseA")
            or line.startswith("  - falseS")]


def test_four_fr1_lines_each_holds_with_its_detectable_difference_and_minimum(stored):
    text = _report(stored)
    cells = _cell_lines(text)
    assert [line.split(":")[0].strip("- ") for line in cells] == [
        "falseAdvanceRateByUnderstanding", "falseAdvanceRateByTurns",
        "falseSuspendRateByUnderstanding", "falseSuspendRateByTurns",
    ]
    # ψ̂ = 0, so every discordant item is candidate-worse and power is P(Bin(197, δ) ≥ 5): 0.781
    # at 3.3 pp, 0.802 at 3.4 pp (checked by a stdlib script). U is the stats primitive's own.
    upper = stats.mover_d_interval(0, 0, 0, 197, z=1.6448536269514722)[1]
    assert cells[0] == (
        "  - falseAdvanceRateByUnderstanding: candidate 0/197 [0.000, 0.019] vs incumbent 0/197 "
        f"[0.000, 0.019]; diff +0.0 pp; b=0, c=0, p_worse=1.0000; worse by at most "
        f"{upper * 100:.1f} pp (one-sided 95%) — holds (could only detect ≥ 3.4 pp at n = 197, "
        "ψ̂ = 0.00)"
    )
    assert cells[1].endswith("— holds (no worsening is detectable at n = 1: fewer than 5 items)")
    assert "    - scored n 197 beside method minimum 50; overLimit 0/197" in text
    assert (
        "    - scored n 1 beside method minimum 50 — below it: printed, never refused; "
        "overLimit 0/1"
    ) in text
    assert "- FR-1 (replicate 1): met" in text
    assert "- **FR-1 parity, over 2 replicate(s) (2 load(s)): met**" in text


def test_an_empty_cell_still_prints_its_line_and_its_minimum(tmp_path):
    # P16-m1 at the report: the turns false-advance cell's one item is withheld by the incumbent,
    # so the cell is empty — and still prints, with its 0/0 and its method minimum.
    root = _root(tmp_path)
    inc = _incumbent(root)
    items = tuple(
        replace(i, scoreable={}, counts={}, timing=replace(i.timing, wallClockMs=None,
                withheldFor="timeout"), outcome="fail", detail={})
        if i.itemId == "fx-cs-02" else i
        for i in inc.items
    )
    inc = _reaggregate(root, inc, items)
    _store_all(root, inc, _candidate(root, minutes=10))
    text = _report(root)
    cells = _cell_lines(text)
    assert len(cells) == 4
    assert cells[1].startswith(f"  - {_FA_T}: candidate 0/0 [—] vs incumbent 0/0 [—]; diff —")
    assert "    - scored n 0 beside method minimum 50 — below it: printed, never refused; " \
        "overLimit 0/0" in text
    assert "- scored eval n: 199 of the pack's 200 eval items, beside FR-11's 200; " \
        "dropped: incumbent timeout 1" in text


def _reaggregate(root: Path, run: RunResult, items) -> RunResult:
    """A chat-arm `run` with `items`, its aggregates re-derived by the scorer's own aggregation,
    so the aggregate-consistency check still passes."""
    from modelbench.scoring import classification

    pack = load_pack(root / "packs" / base._PACK_ID)
    return replace(run, items=items, aggregates=classification.aggregate(items, pack=pack))


def test_a_worse_cell_makes_the_replicate_and_the_bar_not_met(tmp_path):
    root = _root(tmp_path)
    flips = {f"pad-{n:03d}" for n in range(6)}  # candidate advances on six suspends: b = 6
    _store_all(root, _incumbent(root), _candidate(root, minutes=10, flips=flips),
               _candidate(root, minutes=20))
    text = _report(root)
    assert _cell_lines(text)[0].endswith("b=6, c=0, p_worse=0.0156; worse by at most "
                                         + _cell_lines(text)[0].split("worse by at most ")[1]
                                         .split(" pp")[0] + " pp (one-sided 95%) — not met: "
                                         "candidate worse")
    assert "- FR-1 (replicate 1): not met" in text
    assert "- FR-1 (replicate 2): met" in text
    assert "FR-1 parity, over 2 replicate(s) (2 load(s)): unstable across replicates — not met" \
        in text


def test_scored_n_below_200_is_not_assessable_whatever_the_cells_say(tmp_path):
    # Plan2 §3.6 / note §Q4.3: the scored-n rule comes first. A worse cell at n = 199 is not
    # "not met"; FR-2 is not assessable too.
    root = _root(tmp_path)
    flips = {f"pad-{n:03d}" for n in range(6)}
    inc = _incumbent(root)
    items = tuple(
        replace(i, timing=replace(i.timing, withheldFor="load")) if i.itemId == "pad-100" else i
        for i in inc.items
    )
    inc = _reaggregate(root, inc, items)
    _store_all(root, inc, _candidate(root, minutes=10, flips=flips))
    text = _report(root)
    assert "- scored eval n: 199 of the pack's 200 eval items" in text
    assert "dropped: incumbent load 1" in text
    assert "- FR-1 (replicate 1): not assessable (scored n < 200)" in text
    assert _cell_lines(text)[0].endswith("— not assessable (scored n < 200)")
    assert re.search(r"- FR-2 speed-up: .* — not assessable \(scored n < 200\)", text)


def test_the_fr2_ratio_is_the_incumbents_mean_over_the_candidates_on_the_pinned_b(
    stored, monkeypatch
):
    monkeypatch.setattr(decision_stats, "BOOTSTRAP_B", 10_000)
    text = _report(stored)
    runs = _runs(stored)
    inc = next(r for r in runs if r.armKey == QWEN)
    rep = next(r for r in runs if r.armKey == CAND)
    x = sum(i.timing.wallClockMs for i in inc.items if i.itemId != "sc-01")
    y = sum(i.timing.wallClockMs for i in rep.items if i.itemId != "sc-01")
    assert x / y == pytest.approx(4.0)  # 400 ms chat calls against a 0.1 s item clock
    line = next(line for line in text.splitlines() if line.startswith("- FR-2 speed-up:"))
    assert line.startswith(f"- FR-2 speed-up: {x / y:.2f} [")
    assert "B=10000, seed 20261031" in line and line.endswith("— met")
    assert "**FR-2 speed-up, over 2 replicate(s) (2 load(s)): met**" in text


def test_a_slow_candidate_misses_fr2(tmp_path):
    root = _root(tmp_path)
    _store_all(root, _incumbent(root, ms=400.0), _candidate(root, minutes=10, step=0.3))
    text = _report(root)
    assert re.search(r"- FR-2 speed-up: 1\.33 \[.*— not met$", text, re.M)


def test_the_side_check_section_prints_its_actual_size_and_its_own_drops(tmp_path):
    # P12-n1: the side check's real n (1 here), never v1.3 §3.5's hard-coded 85. A side-check
    # decision failure is dropped and counted in the side check, never in the bar's drops.
    root = _root(tmp_path)
    _store_all(root, _incumbent(root), _candidate(root, minutes=10, fail={"sc-01"}))
    text = _report(root)
    assert "Side check — 1 human-curated items (does not affect pass/fail)" in text
    assert "85" not in text.split("Side check —")[1].split("\n")[0]
    assert "- side check: dropped: candidate side-check decision failure 1" in text
    assert "beside FR-11's 200; none dropped" in text
    assert "decisionCallFailures 1/201" in text


def test_a_side_check_item_scores_exploratory_lines_only(stored):
    text = _report(stored)
    side = text.split("Side check —")[1].split("####")[0]
    assert "- sideCheckFalseSuspendRate: candidate 0/1 [" in side
    assert "exploratory — no significance claim" in side
    assert "falseAdvanceRate (pooled): candidate 0/198" in text
    # Exploratory side-check rates stay in the side check, never among the eval lines (P18-m1).
    assert not re.search(r"^- sideCheck\w* \(pooled\)", text, re.M)


def test_a_guard_decision_failure_is_concordant_in_the_incumbents_cell(tmp_path):
    root = _root(tmp_path)
    _store_all(root, _incumbent(root, flips={"fx-cs-01"}), _candidate(root, fail={"fx-cs-01"}))
    text = _report(root)
    assert _cell_lines(text)[0].startswith(
        "  - falseAdvanceRateByUnderstanding: candidate 1/197 [0.001, 0.028] vs incumbent 1/197"
    )
    assert "b=0, c=0" in _cell_lines(text)[0]
    assert "decisionCallFailures 1/201" in text and "overLimit 0/201" in text


def test_an_assumed_replicate_is_not_assessable_and_left_out(tmp_path):
    # Plan2 §5 U3c: met + assumed + met (fresh load) gives met.
    root = _root(tmp_path)
    _store_all(
        root, _incumbent(root), _candidate(root, minutes=10),
        _candidate(root, minutes=20, recheck_differs=True), _candidate(root, minutes=30),
    )
    text = _report(root)
    assert "#### replicate 2 — run" in text and "basis assumed" in text
    assert "- FR-1 (replicate 2): not assessable (load changed mid-run)" in text
    assert "**FR-1 parity, over 3 replicate(s) (2 load(s)): met**" in text


def test_one_load_alone_is_not_assessable(tmp_path):
    root = _root(tmp_path)
    _store_all(root, _incumbent(root), _candidate(root, minutes=10),
               _candidate(root, minutes=20, resident_at_start=True))
    text = _report(root)
    assert "#### replicate 2 — run" in text and ", not counted as a load" in text
    assert "- replicate 2: same load as an earlier run — not counted as a load" in text
    assert "(1 load(s)): not assessable (only 1 load replicate(s) (FR-11a needs 2))**" in text


def test_equal_timeouts_off_the_pin_make_every_bar_not_assessable(tmp_path):
    root = _root(tmp_path)
    runs = [_incumbent(root), _candidate(root, minutes=10), _candidate(root, minutes=20)]
    _store_all(root, *(replace(r, requestTimeoutSeconds=120.0) for r in runs))
    text = _report(root)
    assert text.count("not assessable (chat timeout ≠ 180 s)") >= 6
    assert "- FR-1 (replicate 1): not assessable (chat timeout ≠ 180 s)" in text
    assert "(2 load(s)): not assessable (chat timeout ≠ 180 s)**" in text


def test_a_legacy_none_timeout_on_every_arm_is_off_the_pin_too(tmp_path):
    root = _root(tmp_path)
    runs = [_incumbent(root), _candidate(root, minutes=10)]
    _store_all(root, *(replace(r, requestTimeoutSeconds=None) for r in runs))
    assert "- FR-1 (replicate 1): not assessable (chat timeout ≠ 180 s)" in _report(root)


def test_a_pairing_error_prints_as_not_assessable_never_a_crash(tmp_path):
    # A runner refuses a stratum-less pack, so the records here claim the edited content they
    # never ran on: the pack-content guard (P18-M1) is quiet and `pair_items`' refusal prints.
    root = _root(tmp_path)
    runs = [_incumbent(root), _candidate(root, minutes=10)]
    rows = _rows(root)
    rows[0].pop("stratum")
    (root / "packs" / base._PACK_ID / "items.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    edited = load_pack(root / "packs" / base._PACK_ID).contentHash
    _store_all(root, *(_variant(r, packContentHash=edited) for r in runs))
    text = _report(root)
    assert (
        "- FR-1 (replicate 1): not assessable (cannot pair with the incumbent: pair_items: no "
        "eval/side-check stratum for items ['fx-cs-01'])"
    ) in text


def test_an_inconsistent_replicate_makes_its_groups_bars_not_assessable(tmp_path):
    root = _root(tmp_path)
    flips = {f"pad-{n:03d}" for n in range(6)}
    bad = _candidate(root, minutes=20)
    items = tuple(
        replace(i, scoreable={}, counts={}) if i.itemId == "pad-050" else i
        for i in bad.items
    )
    _store_all(root, _incumbent(root), _candidate(root, minutes=10, flips=flips),
               replace(bad, items=items))
    text = _report(root)
    assert "- FR-1 (replicate 2): not assessable (this replicate's aggregates disagree" in text
    assert "(2 load(s)): not assessable (replicate 2: aggregates disagree with items)**" in text


def test_the_disagreement_line_compares_each_replicate_with_the_first(tmp_path):
    root = _root(tmp_path)
    _store_all(root, _incumbent(root), _candidate(root, minutes=10),
               _candidate(root, minutes=20, flips={"pad-007"}, fail={"pad-009"}))
    text = _report(root)
    line = next(line for line in text.splitlines() if line.startswith("- replicate 2 vs"))
    assert line.startswith("- replicate 2 vs replicate 1: final answer differs on 1/200 [")
    assert "decidedBy on 0/200 [0.000, 0.019]" in line
    assert line.endswith("1 item(s) with a failed call in either and 0 withheld in one replicate "
                         "only")


def test_a_lone_replicate_has_nothing_to_disagree_with(tmp_path):
    root = _root(tmp_path)
    _store_all(root, _incumbent(root), _candidate(root, minutes=10))
    assert "- replicate disagreement: 1 non-assumed replicate(s), nothing to compare" in _report(
        root
    )


def test_compare_still_exits_zero_and_writes_nothing_outside_reports(stored):
    before = sorted(p.name for p in (stored / "results" / "runs").iterdir())
    assert _compare(stored) == 0
    assert sorted(p.name for p in (stored / "results" / "runs").iterdir()) == before


def test_an_inconsistent_incumbent_makes_every_bar_not_assessable(tmp_path):
    root = _root(tmp_path)
    inc = _incumbent(root)
    items = tuple(
        replace(i, scoreable={}, counts={}) if i.itemId == "pad-050" else i for i in inc.items
    )
    _store_all(root, replace(inc, items=items), _candidate(root, minutes=10),
               _candidate(root, minutes=20, recheck_differs=True))
    text = _report(root)
    reason = "not assessable (the incumbent run's aggregates disagree with its items)"
    assert f"- FR-1 (replicate 1): {reason}" in text
    assert "- FR-1 (replicate 2): not assessable (load changed mid-run)" in text
    assert f"(1 load(s)): {reason}**" in text


# --- the bar's own pack-content guard (impl review P18-M1) ----------------------------------------


_EDITED = {f"pad-{n:03d}" for n in range(6)}


def _edit_inputs(root: Path, *, restore: list[dict] | None = None) -> list[dict]:
    """Edit six eval items' `condition` in place (labels, strata and `path` unchanged), or write
    `restore` back; returns the rows as they were before the write."""
    rows = _rows(root)
    new = restore if restore is not None else [
        {**r, "condition": f"{r.get('condition')} (edited)"} if r["itemId"] in _EDITED else r
        for r in rows
    ]
    (root / "packs" / base._PACK_ID / "items.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in new)
    )
    return rows


def _content_reason(*runs: RunResult, pack_hash: str) -> str:
    hashes = {r.fingerprint.get("packContentHash") for r in runs} | {pack_hash}
    assert len(hashes) == 2
    return f"not assessable (arms ran on different pack content: {', '.join(sorted(
        h[:8] for h in hashes))})"


def _assert_every_bar(text: str, reason: str, replicates: int) -> None:
    for i in range(1, replicates + 1):
        assert f"- FR-1 (replicate {i}): {reason}" in text
        assert f"- FR-2 (replicate {i}): {reason}" in text
    assert f"(2 load(s)): {reason}**" in text
    assert text.count(f"{reason}**") == 2  # FR-1 and FR-2 across the replicates
    assert "): met" not in text and "): not met" not in text and "— met" not in text


def test_an_input_edit_between_the_incumbent_and_the_replicates_blocks_every_bar(tmp_path):
    # The reviewer's probe: the incumbent ran on one content, an input-only edit (no label, no
    # `path`) follows, and both replicates ran on the edited content — which `pair_items` cannot
    # see, and the `runs` banners never reach (candidates arrive only as `replicates`).
    root = _root(tmp_path)
    inc = _incumbent(root)
    _edit_inputs(root)
    reps = [_candidate(root, minutes=10), _candidate(root, minutes=20)]
    _store_all(root, inc, *reps)
    pack_hash = load_pack(root / "packs" / base._PACK_ID).contentHash
    assert reps[0].fingerprint.get("packContentHash") == pack_hash
    assert inc.fingerprint.get("packContentHash") != pack_hash
    _assert_every_bar(_report(root), _content_reason(inc, *reps, pack_hash=pack_hash), 2)


def test_a_replicate_on_other_content_blocks_every_bar_though_the_incumbent_matches_the_pack(
    tmp_path,
):
    # Comparing only the incumbent with the loaded pack would pass this: replicate 1 alone ran on
    # the edited content, and the pack was restored before replicate 2.
    root = _root(tmp_path)
    inc = _incumbent(root)
    original = _edit_inputs(root)
    first = _candidate(root, minutes=10)
    _edit_inputs(root, restore=original)
    second = _candidate(root, minutes=20)
    _store_all(root, inc, first, second)
    pack_hash = load_pack(root / "packs" / base._PACK_ID).contentHash
    assert inc.fingerprint.get("packContentHash") == pack_hash
    assert second.fingerprint.get("packContentHash") == pack_hash
    reason = _content_reason(inc, first, second, pack_hash=pack_hash)
    _assert_every_bar(_report(root), reason, 2)


def test_a_later_replicate_on_other_content_blocks_every_bar(tmp_path):
    # The mirror (impl review P19-m1): the incumbent and replicate 1 match the pack, replicate 2
    # alone ran on edited inputs. `_replicate_groups` refuses mixed candidate hashes, never mixed
    # pack hashes, so only a check over every replicate sees this.
    root = _root(tmp_path)
    inc = _incumbent(root)
    first = _candidate(root, minutes=10)
    original = _edit_inputs(root)
    second = _candidate(root, minutes=20)
    _edit_inputs(root, restore=original)
    _store_all(root, inc, first, second)
    pack_hash = load_pack(root / "packs" / base._PACK_ID).contentHash
    assert {r.fingerprint.get("packContentHash") for r in (inc, first)} == {pack_hash}
    reason = _content_reason(inc, first, second, pack_hash=pack_hash)
    _assert_every_bar(_report(root), reason, 2)


def test_different_content_outranks_an_off_pin_timeout_and_nothing_is_paired(tmp_path):
    # Impl review P19-n1: equal timeouts off the pin plus an input-only edit. The content reason
    # wins on every bar, and no replicate is paired (no scored n, no FR-2 ratio).
    root = _root(tmp_path)
    inc = _incumbent(root)
    _edit_inputs(root)
    reps = [_candidate(root, minutes=10), _candidate(root, minutes=20)]
    runs = [replace(r, requestTimeoutSeconds=120.0) for r in (inc, *reps)]
    _store_all(root, *runs)
    pack_hash = load_pack(root / "packs" / base._PACK_ID).contentHash
    text = _report(root)
    _assert_every_bar(text, _content_reason(*runs, pack_hash=pack_hash), 2)
    assert "chat timeout" not in text
    assert "- scored eval n:" not in text and "- FR-2 speed-up:" not in text


def test_a_pack_edited_after_every_arm_ran_blocks_every_bar(tmp_path):
    # Every arm agrees, but the bar reads strata and counts from the pack on disk, which is now a
    # different content.
    root = _root(tmp_path)
    runs = [_incumbent(root), _candidate(root, minutes=10), _candidate(root, minutes=20)]
    _store_all(root, *runs)
    _edit_inputs(root)
    pack_hash = load_pack(root / "packs" / base._PACK_ID).contentHash
    assert all(r.fingerprint.get("packContentHash") != pack_hash for r in runs)
    _assert_every_bar(_report(root), _content_reason(*runs, pack_hash=pack_hash), 2)


# --- the disagreement line (impl review P18-m1) ---------------------------------------------------


def test_the_disagreement_line_counts_decided_by_and_one_sided_withholds(tmp_path):
    # Replicate 2 hands pad-011 off where replicate 1 decided it, and withholds pad-013 alone;
    # neither changes the final answer, so only their own counts can move.
    root = _root(tmp_path)
    second = _candidate(root, minutes=20, flips={"pad-007"}, fail={"pad-009"})
    items = tuple(
        replace(i, detail={**i.detail, "decidedBy": "handoff"}) if i.itemId == "pad-011"
        else replace(i, scoreable={}, counts={}, outcome="fail",
                     timing=replace(i.timing, withheldFor="timeout")) if i.itemId == "pad-013"
        else i
        for i in second.items
    )
    assert next(i for i in second.items if i.itemId == "pad-011").detail.get("decidedBy") \
        != "handoff"
    _store_all(root, _incumbent(root), _candidate(root, minutes=10), replace(second, items=items))
    text = _report(root)
    line = next(line for line in text.splitlines() if line.startswith("- replicate 2 vs"))
    assert line.startswith("- replicate 2 vs replicate 1: final answer differs on 1/199 [")
    assert "decidedBy on 1/199 [" in line
    assert line.endswith("1 item(s) with a failed call in either and 1 withheld in one replicate "
                         "only")


def test_every_non_assumed_replicate_is_compared_with_the_first_and_an_assumed_one_never(
    tmp_path,
):
    root = _root(tmp_path)
    _store_all(
        root, _incumbent(root), _candidate(root, minutes=10),
        _candidate(root, minutes=20, recheck_differs=True), _candidate(root, minutes=30),
        _candidate(root, minutes=40),
    )
    text = _report(root)
    assert "basis assumed" in text.split("#### replicate 2 —")[1].split("\n")[0]
    vs = [line.split(":")[0] for line in text.splitlines() if " vs replicate " in line]
    assert vs == ["- replicate 3 vs replicate 1", "- replicate 4 vs replicate 1"]
