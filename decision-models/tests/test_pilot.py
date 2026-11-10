"""The guard pilot (typed-decision-models U5.8c): the screen, the cue check, the report, the recheck
and the decision (plan4 `docs/plans/typed-decision-models4.md` §3.8 and §5 row U5.8; plan5
`typed-decision-models5.md` §3.6; plan6 `typed-decision-models6.md` §3.8).

The pure parts (`screen`, `cue_check`, `fit_line`) are tested on hand-built values whose answers
are worked out in the comments. The campaign parts run real generation and label runs through the
fixture's agent (`test_votes.py`'s harness): the labeller's answers are a function of the item's
seed axes, so a vote credited to the wrong item or number moves a figure.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from test_campaign import Clock, pilot_report, stage, write_answer, write_fixture
from test_screen import obedient_item
from test_votes import run_vote, vote_bench

from decisionmodels import campaign, labels, pilot
from decisionmodels.campaign import Campaign, CampaignRefused, Serve, Stop
from decisionmodels.pilot import CueItem, PilotRefused

Answer = Callable[[Any, int], Any]

# --- the screen ----------------------------------------------------------------------------------


def votes_with(n: int, delta: int) -> dict[str, dict[int, bool]]:
    """`n` items carrying votes 1, 2 and the screen vote, `delta` of which differ between vote 1
    and the screen vote while vote 2 agrees with vote 1: so Δ is exactly `delta`."""
    return {f"it{i}": {1: True, 2: True, labels.SCREEN_VOTE: i >= delta} for i in range(n)} \
        if delta >= 0 else {
            f"it{i}": {1: True, 2: i >= -delta, labels.SCREEN_VOTE: True} for i in range(n)}


@pytest.mark.parametrize(("n", "delta", "cap"), [
    (100, 10, 25),     # 10 * 10 >= 100: the equality case sets 25
    (100, 9, 50),      # 90 < 100
    (90, 9, 25),       # 90 >= 90
    (90, 8, 50),
    (80, 8, 25),
    (100, 0, 50),
    (100, -30, 50),    # vote 2 disagrees more than the screen vote does
    (105, 10, 50),     # 100 < 105: the factor is 10, not 11
    (95, 10, 25),
])
def test_the_screen_sets_the_cap_from_ten_times_delta_against_n(n: int, delta: int,
                                                                cap: int) -> None:
    result = pilot.screen(votes_with(n, delta))
    assert (result.n, result.delta, result.cap) == (n, delta, cap)


def test_delta_is_the_screen_vote_gap_less_the_vote_2_gap() -> None:
    votes = {
        "a": {1: True, 2: True, 4: False},     # v1 != screen: +1
        "b": {1: True, 2: False, 4: True},     # v1 != v2: -1
        "c": {1: True, 2: False, 4: False},    # both differ: +1 -1
        "d": {1: False, 2: False, 4: False},   # neither
    }
    result = pilot.screen(votes)
    assert (result.n, result.delta) == (4, 0)


def test_only_items_with_votes_1_2_and_the_screen_vote_count() -> None:
    votes = {"a": {1: True, 2: True, 4: False}, "b": {1: True, 2: True},
             "c": {1: True, 4: False}, "d": {2: True, 4: False}, "e": {}}
    assert pilot.screen(votes).n == 1


@pytest.mark.parametrize(("n", "more"), [(79, True), (80, False), (0, True)])
def test_fewer_than_80_items_need_another_block(n: int, more: bool) -> None:
    assert pilot.screen(votes_with(n, 0)).more_needed is more


# --- the cue check -------------------------------------------------------------------------------


def stratum_items(name: str, form_departs: dict[str, list[bool]]) -> list[CueItem]:
    return [CueItem(name, form, departs)
            for form, flags in form_departs.items() for departs in flags]


@pytest.mark.parametrize(("form", "counter"), [
    ("immaterial", True), ("unlisted", True), ("ambiguous", True),
    ("empty", False), ("indirect", False), ("listed", False),
])
def test_the_three_counter_cue_forms_are_the_counter_cue_class(form: str, counter: bool) -> None:
    assert CueItem("s", form, False).counter_cue is counter


def test_a_stratum_counts_x_m1_y_m0_and_both_rates() -> None:
    items = (stratum_items("s", {"immaterial": [True, True, False], "unlisted": [True],
                                 "empty": [False, False, True], "listed": [False]}))
    (row,) = pilot.cue_check(items).strata
    assert (row.x, row.m1, row.y, row.m0) == (3, 4, 1, 4)
    record = row.as_record()
    assert record["counterCueRate"] == 0.75 and record["concordantRate"] == 0.25
    assert pilot.cue_check([CueItem("s", "empty", True)]).strata[0].as_record()[
        "counterCueRate"] is None


def test_one_stratum_p_is_the_hypergeometric_tail() -> None:
    # m1 = 4, m0 = 6, t = x + y = 4, observed x = 3: (C(4,3)C(6,1) + C(4,4)C(6,0)) / C(10,4)
    # = 25 / 210. Not failed. At x = 4, y = 0: 1 / 210, failed.
    loose = stratum_items("s", {"immaterial": [True] * 3 + [False], "empty": [True] + [False] * 5})
    result = pilot.cue_check(loose)
    assert result.p == Fraction(25, 210) and not result.failed and result.sum_x == 3
    tight = stratum_items("s", {"immaterial": [True] * 4, "empty": [False] * 6})
    result = pilot.cue_check(tight)
    assert result.p == Fraction(1, 210) and result.failed


def test_the_p_value_convolves_the_strata() -> None:
    # Each stratum: m1 = m0 = 2, t = 2, so x is 0, 1, 2 with 1/6, 4/6, 1/6. Observed Σx = 3:
    # P = 2 * (1/6)(4/6) + (1/6)(1/6) = 9/36.
    a = stratum_items("a", {"immaterial": [True, True], "empty": [False, False]})
    b = stratum_items("b", {"immaterial": [True, False], "empty": [True, False]})
    result = pilot.cue_check(a + b)
    assert result.sum_x == 3 and result.p == Fraction(1, 4) and not result.failed


def test_p_equal_to_one_twentieth_fails_and_just_above_passes() -> None:
    # m1 = 1, m0 = 19, one departure and it is the counter-cue item: p = 1/20 exactly.
    edge = stratum_items("s", {"immaterial": [True], "empty": [False] * 19})
    assert pilot.cue_check(edge).p == Fraction(1, 20) and pilot.cue_check(edge).failed
    above = stratum_items("s", {"immaterial": [True], "empty": [False] * 18})
    assert pilot.cue_check(above).p == Fraction(1, 19) and not pilot.cue_check(above).failed


def test_equal_departure_rates_in_each_stratum_do_not_fail() -> None:
    # Answer-dependent noise: stratum "false" departs at 100%, stratum "true" at 0%, for both
    # classes alike. Pooled over strata it would look like cue-following; stratified it is not.
    items = (stratum_items("false", {"immaterial": [True] * 5, "empty": [True] * 5})
             + stratum_items("true", {"immaterial": [False] * 5, "empty": [False] * 5}))
    result = pilot.cue_check(items)
    assert result.p == 1 and not result.failed


def test_no_counter_cue_items_never_fails() -> None:
    result = pilot.cue_check(stratum_items("s", {"empty": [True, True, False]}))
    assert result.p == 1 and result.sum_x == 0 and not result.failed
    assert pilot.cue_check([]).p == 1


def test_a_cue_check_record_prints_the_counts_the_p_value_and_the_verdict() -> None:
    record = pilot.cue_check(stratum_items(
        "s", {"immaterial": [True] * 4, "empty": [False] * 6})).as_record()
    assert record["sumX"] == 4 and record["failed"] is True
    assert record["p"] == pytest.approx(1 / 210)
    assert record["strata"][0] == {"stratum": "s", "x": 4, "m1": 4, "y": 0, "m0": 6,
                                   "counterCueRate": 1.0, "concordantRate": 0.0}


# --- tokens --------------------------------------------------------------------------------------


def test_f_and_c_are_recovered_from_runs_at_two_batch_sizes() -> None:
    points = [(50, 1000 + 20 * 50)] * 3 + [(10, 1000 + 20 * 10)] * 5
    floor, marginal = pilot.fit_line(points)
    assert floor == pytest.approx(1000) and marginal == pytest.approx(20)


def test_f_and_c_need_two_batch_sizes() -> None:
    with pytest.raises(PilotRefused, match="two batch sizes"):
        pilot.fit_line([(50, 100), (50, 120)])


def fake_state(sizes_and_tokens: list[tuple[int, dict[str, int]]]) -> Any:
    batches = {}
    for index, (size, tokens) in enumerate(sizes_and_tokens):
        outcome = {"outcome": "ingested", "run": {"tokens": tokens}}
        batches[f"b{index}"] = SimpleNamespace(
            kind="label", entries=list(range(size)), extra={},
            attempts=[SimpleNamespace(outcome=outcome)])
    return SimpleNamespace(batches=batches)


def test_label_costs_fit_each_token_class_on_its_own() -> None:
    runs = [(size, {"input": 10 * size + 7, "cacheWrite": 5000, "cacheRead": 300 * size,
                    "output": 2 * size + 100}) for size in (50, 50, 10, 10)]
    costs = pilot._label_costs(fake_state(runs))
    assert costs["input"] == pytest.approx({"F": 7, "c": 10})
    assert costs["cacheWrite"] == pytest.approx({"F": 5000, "c": 0})
    assert costs["cacheRead"] == pytest.approx({"F": 0, "c": 300})
    assert costs["output"] == pytest.approx({"F": 100, "c": 2})


def test_label_costs_leave_out_a_run_that_was_not_ingested() -> None:
    state = fake_state([(50, {"input": 1050}), (10, {"input": 250}), (50, {"input": 1050})])
    spoiled = [SimpleNamespace(outcome={"outcome": "void", "run": {"tokens": {"input": 10**9}}}),
               SimpleNamespace(outcome=None)]
    state.batches["void"] = SimpleNamespace(kind="label", entries=list(range(30)), extra={},
                                            attempts=spoiled)
    assert pilot._label_costs(state)["input"] == pytest.approx({"F": 50, "c": 20})


def test_the_generator_mean_is_over_first_pass_runs_only() -> None:
    def gen(kind: str, tokens: int) -> Any:
        outcome = {"outcome": "ingested", "run": {"tokens": {"output": tokens}}}
        return SimpleNamespace(kind="generate", extra={"generation": kind}, entries=[],
                               attempts=[SimpleNamespace(outcome=outcome)])
    state = SimpleNamespace(batches={"a": gen("first", 100), "b": gen("first", 300),
                                     "c": gen("regen", 10**6)})
    assert pilot._generator_mean(state)["output"] == 200
    with pytest.raises(PilotRefused, match="no ingested generation run"):
        pilot._generator_mean(SimpleNamespace(batches={}))


def test_the_cue_items_are_stratified_by_difficulty_and_intended_answer() -> None:
    # Within "clear" the intended-True items all depart (10 counter-cue, 2 concordant) and the
    # intended-False items none do (2 counter-cue, 10 concordant): answer-dependent noise. Pooled
    # by difficulty alone p = 0.0017 and the check would fail; by difficulty x answer p = 1.
    seeds, items, forms = {}, {}, {}
    plan = [(True, "immaterial", 10), (True, "empty", 2), (False, "immaterial", 2),
            (False, "empty", 10)]
    count = 0
    for intended, form, many in plan:
        for _ in range(many):
            count += 1
            sid = f"s{count}"
            seeds[sid] = SimpleNamespace(axes={"evidenceKind": "understanding",
                                               "difficulty": "clear", "intendedAnswer": intended,
                                               "missingForm": form})
            items[sid] = {"genRound": 1}
            forms[sid] = [not intended] * 3 if intended else [intended] * 3
    # A turns item and an item of another round are not cue items.
    seeds["t"] = SimpleNamespace(axes={"evidenceKind": "turns", "difficulty": "clear",
                                       "intendedAnswer": True, "missingForm": None})
    items["t"], forms["t"] = {"genRound": 1}, [False] * 3
    seeds["r"] = SimpleNamespace(axes=seeds["s1"].axes)
    items["r"], forms["r"] = {"genRound": 2}, [False] * 3
    state = SimpleNamespace(items=items, seeds=seeds, three_votes=forms.get)
    cue = pilot._cue_items(state, [1])
    assert len(cue) == 24 and {c.stratum for c in cue} == {"clear|true", "clear|false"}
    result = pilot.cue_check(cue)
    assert result.sum_x == 10 and result.p == 1 and not result.failed


def test_the_projection_prices_runs_item_votes_and_the_cap_per_point() -> None:
    generator = dict.fromkeys(pilot.TOKEN_CLASSES, 0.0) | {"output": 1000.0}
    labeller = {cls: {"F": 0.0, "c": 0.0} for cls in pilot.TOKEN_CLASSES}
    labeller["input"] = {"F": 100.0, "c": 1.0}
    guard = pilot._projection(25, generator, labeller)["guard-judge"]
    # guard's target is 800 and 1.3 x 800 = 1040 items: 42 generation runs of 25, three votes of
    # ceil(1040 / 25) = 42 label runs each, and 3 * 1040 item-votes.
    assert (guard["generationRuns"], guard["labelRuns"], guard["itemVotes"]) == (42, 126, 3120)
    assert guard["tokens"]["output"] == 42 * 1000
    assert guard["tokens"]["input"] == 126 * 100 + 3120
    assert pilot._projection(50, generator, labeller)["guard-judge"]["labelRuns"] == 3 * 21


# --- the campaign --------------------------------------------------------------------------------


def start_bench(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Campaign:
    camp = vote_bench(tmp_path, monkeypatch, cap=50)
    pilot.start(camp)
    return camp


def run_gen(camp: Campaign, action: Serve,
            item_of: Callable[[Any], dict[str, Any]] = obedient_item) -> None:
    session = camp.issue(action)
    stage(camp, session, action.opaque)
    keys = camp.state().opaques[action.opaque].keys
    text = (camp.store / "req" / f"{action.opaque}.jsonl").read_text("utf-8")
    lines = [{"bodySha256": json.loads(text.splitlines()[0])["bodySha256"]}]
    lines += [{"key": key, "item": item_of(camp.state().seeds[seed_id])}
              for key, seed_id in keys.items()]
    write_answer(camp, action.opaque,
                 "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines)
                 .encode("utf-8"))
    camp.ingest(action.opaque)


def drive(camp: Campaign, answer: Answer, limit: int = 120) -> Stop:
    """Serve generation and label runs until `next` stops. `answer(seed, vote)` is the labeller's
    answer for the item of `seed`."""
    for _ in range(limit):
        action = camp.next()
        if not isinstance(action, Serve):
            return action
        if action.kind == "generate":
            run_gen(camp, action)
        else:
            run_vote(camp, action, lambda ref, vote: answer(camp.state().seeds[ref], vote))
    raise AssertionError(f"still serving after {limit} runs")


def faithful(seed: Any, vote: int) -> bool:
    return seed.axes["intendedAnswer"]


def following_the_cue(seed: Any, vote: int) -> bool:
    """The labeller judges a counter-cue `understanding` item by the list: it departs from
    intent."""
    axes = seed.axes
    follows = axes["evidenceKind"] == "understanding" and axes["missingForm"] in \
        pilot.COUNTER_CUE_FORMS
    return (not axes["intendedAnswer"]) if follows else axes["intendedAnswer"]


def with_screen_gap(camp: Campaign, gap: int) -> Answer:
    """Faithful, except that the screen vote of the first `gap` items differs from vote 1."""
    chosen: list[str] = []      # the items exist once generation ends, before the first vote

    def answer(seed: Any, vote: int) -> bool:
        if not chosen:
            chosen.extend(sorted(camp.state().items)[:gap] or ["none"])
        flip = vote == labels.SCREEN_VOTE and seed.seed_id in chosen
        return seed.axes["intendedAnswer"] != flip
    return answer


def label_batches(camp: Campaign, gen_round: int, vote: int) -> list[int]:
    return [len(b.entries) for b in camp.state().vote_batches(gen_round, vote)]


def test_start_draws_a_hundred_seeds_as_four_runs_of_25_for_a_main_block(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    camp = start_bench(tmp_path, monkeypatch)
    state = camp.state()
    (round_one,) = state.rounds
    assert round_one["genRound"] == 1 and round_one["pilotBlock"] == campaign.MAIN_BLOCK
    assert len(round_one["seeds"]) == pilot.FIRST_BLOCK_SEEDS == 100
    assert [len(b.entries) for b in state.batches.values()] == [25, 25, 25, 25]
    assert all(b.extra == {"generation": "first"} for b in state.batches.values())
    assert state.round_votes(1) == (1, 2, 3, labels.SCREEN_VOTE)


def test_start_refuses_a_second_call_and_a_campaign_that_is_not_a_pilot(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    camp = start_bench(tmp_path, monkeypatch)
    with pytest.raises(PilotRefused, match="already has a block"):
        pilot.start(camp)
    other = Campaign(tmp_path / "full", clock=Clock())
    other.init(point="guard-judge", phase="smoke", fixture=_fixture(tmp_path))
    with pytest.raises(PilotRefused, match="pilot campaign"):
        pilot.start(other)


def _fixture(tmp_path: Path) -> Path:
    return write_fixture(tmp_path / "items.jsonl", 2)


def test_a_pilot_block_is_refused_outside_a_pilot_campaign_and_when_unknown(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    camp = vote_bench(tmp_path, monkeypatch)
    with pytest.raises(CampaignRefused, match="pilot block 'other'"):
        camp.begin_round(1, 5, pilot_block="other")
    smoke = Campaign(tmp_path / "smoke", clock=Clock())
    smoke.init(point="guard-judge", phase="smoke", fixture=_fixture(tmp_path))
    with pytest.raises(CampaignRefused, match="pilot block"):
        smoke.begin_round(1, 5, pilot_block="main")
    assert smoke.state().rounds == []


def test_a_main_block_votes_1_to_3_at_50_then_the_screen_vote_at_10(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    camp = start_bench(tmp_path, monkeypatch)
    drive(camp, with_screen_gap(camp, 0))
    state = camp.state()
    assert [label_batches(camp, 1, v) for v in (1, 2, 3)] == [[50, 50]] * 3
    assert label_batches(camp, 1, labels.SCREEN_VOTE) == [10] * 10
    order = [(e["genRound"], e["vote"]) for e in camp.events() if e["event"] == "vote-start"]
    assert order == [(1, 1), (1, 2), (1, 3), (1, 4)]
    assert state.round_voted(1) and pilot.screen(state.screen_votes([1])).n == 100


def test_the_screen_vote_is_never_one_of_the_three_that_give_label_votes_and_the_majority(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    camp = start_bench(tmp_path, monkeypatch)
    drive(camp, with_screen_gap(camp, 100))                 # every screen vote disagrees
    state = camp.state()
    seed_id = next(iter(state.items))
    assert state.label_votes(seed_id) in ({"true": 3, "false": 0}, {"true": 0, "false": 3})
    assert all(sum(state.label_votes(s).values()) == 3 for s in state.items)
    assert state.vote_ties() == []
    assert all(len(state.votes[s]) == 4 for s in state.items)
    assert len(state.reshuffle_seeds()) == 4


def test_seventy_nine_items_issue_one_added_block_voted_at_50_and_screened_at_10(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pilot, "FIRST_BLOCK_SEEDS", 79)
    camp = start_bench(tmp_path, monkeypatch)
    stop = drive(camp, faithful)
    state = camp.state()
    assert stop.reason == "done"
    assert [r["genRound"] for r in state.rounds] == [1, 2]
    assert [len(r["seeds"]) for r in state.rounds] == [79, 50]
    assert {r["pilotBlock"] for r in state.rounds} == {"main"}
    second_runs = [b for b in state.first_pass_runs(2)]
    assert [len(b.entries) for b in second_runs] == [25, 25]
    assert label_batches(camp, 2, 1) == [50] == label_batches(camp, 2, 3)
    assert label_batches(camp, 2, labels.SCREEN_VOTE) == [10] * 5
    assert pilot.screen(state.screen_votes([1, 2])).n == 129      # no third block


def test_blocks_are_added_while_fewer_than_80_items_carry_the_screen_vote_and_numbered_in_order(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pilot, "FIRST_BLOCK_SEEDS", 29)           # 29, then 79, then 129 items
    camp = start_bench(tmp_path, monkeypatch)
    drive(camp, faithful)
    state = camp.state()
    assert [r["genRound"] for r in state.rounds] == [1, 2, 3]
    assert [len(r["seeds"]) for r in state.rounds] == [29, 50, 50]
    assert pilot.screen(state.screen_votes([1, 2, 3])).n == 129
    assert all(state.round_voted(g) for g in (1, 2, 3))


def test_a_block_is_added_only_once_the_whole_earlier_block_is_voted(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pilot, "FIRST_BLOCK_SEEDS", 79)
    camp = start_bench(tmp_path, monkeypatch)
    for _ in range(4):
        run_gen(camp, camp.next())
    for _ in range(6 + 8 - 1):                      # all but the last screen batch of block 1
        run_vote(camp, camp.next(), lambda ref, vote: faithful(camp.state().seeds[ref], vote))
    assert [r["genRound"] for r in camp.state().rounds] == [1]
    assert not camp.state().round_voted(1)
    last = camp.next()
    assert last.kind == "label" and [r["genRound"] for r in camp.state().rounds] == [1]
    run_vote(camp, last, lambda ref, vote: faithful(camp.state().seeds[ref], vote))
    assert camp.next().kind == "generate"
    assert [r["genRound"] for r in camp.state().rounds] == [1, 2]


def test_eighty_items_draw_no_added_block(tmp_path: Path,
                                          monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pilot, "FIRST_BLOCK_SEEDS", 80)
    camp = start_bench(tmp_path, monkeypatch)
    drive(camp, faithful)
    assert [r["genRound"] for r in camp.state().rounds] == [1]


def test_screen_prints_n_delta_and_the_cap(tmp_path: Path,
                                           monkeypatch: pytest.MonkeyPatch) -> None:
    camp = start_bench(tmp_path, monkeypatch)
    drive(camp, with_screen_gap(camp, 10))
    assert pilot.screen_lines(camp) == ["n 100, delta 10, cap 25", "n is enough"]


# --- the report ----------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def passed_pilot(tmp_path_factory: pytest.TempPathFactory) -> tuple[Campaign, Path]:
    """A finished pilot whose labeller is faithful, with ten screen-vote differences."""
    tmp = tmp_path_factory.mktemp("pilot-passed")
    with pytest.MonkeyPatch.context() as patch:
        camp = start_bench(tmp, patch)
        drive(camp, with_screen_gap(camp, 10))
    data = tmp / "data"
    pilot.report(camp, data)
    return camp, data


def read(data: Path) -> dict[str, Any]:
    return json.loads((data / campaign.PILOT_REPORT).read_text("utf-8"))


def test_a_faithful_pilot_passes_with_cap_25_at_delta_10(
        passed_pilot: tuple[Campaign, Path]) -> None:
    _, data = passed_pilot
    got = read(data)
    assert (got["n"], got["delta"], got["cap"]) == (100, 10, 25)
    assert got["missingCueCheck"]["state"] == "passed"
    assert got["missingCueCheck"]["recheck"] is None and got["recheckProjection"] is None
    assert got["missingCueCheck"]["pilot"]["sumX"] == 0
    assert got["blocks"] == {"main": 1, "recheck": 0}
    assert "stakeholderDecision" not in got


def test_the_report_holds_the_rng_seeds_the_pilot_drew_and_reshuffled_on(
        passed_pilot: tuple[Campaign, Path]) -> None:
    camp, data = passed_pilot
    state = camp.state()
    assert read(data)["rngSeeds"] == sorted(state.held_rng_seeds)
    assert set(state.reshuffle_seeds()) <= set(read(data)["rngSeeds"])
    assert len(read(data)["rngSeeds"]) == 5            # one draw and four reshuffles


def test_the_report_holds_counts_only_no_item_text_and_no_seed_id(
        passed_pilot: tuple[Campaign, Path]) -> None:
    camp, data = passed_pilot
    text = (data / campaign.PILOT_REPORT).read_text("utf-8")
    for fragment in ("Mensagem", "budget number", "the user has", "sd_", "gen_", "bat_",
                     "dom_", "cond_"):
        assert fragment not in text
    assert not any(record["item"]["condition"] in text
                   for record in camp.state().items.values())


def test_the_report_prices_the_labeller_by_class_and_the_generator_by_run(
        passed_pilot: tuple[Campaign, Path]) -> None:
    got = read(passed_pilot[1])
    assert set(got["labeller"]) == set(pilot.TOKEN_CLASSES)
    assert all(set(row) == {"F", "c"} for row in got["labeller"].values())
    assert set(got["generatorMeanPerRun"]) == set(pilot.TOKEN_CLASSES)
    assert set(got["reprojectedTokens"]) == set(campaign.ROLES)
    guard = got["reprojectedTokens"]["guard-judge"]
    # 1,040 planned items at the pilot's cap 25: 42 generation runs, 3 * 42 label runs.
    assert (guard["generationRuns"], guard["labelRuns"], guard["itemVotes"]) == (42, 126, 3120)
    assert guard["tokens"]["total"] == sum(guard["tokens"][c] for c in pilot.TOKEN_CLASSES)


def test_the_report_counts_the_labels_per_evidence_kind_and_the_runs(
        passed_pilot: tuple[Campaign, Path]) -> None:
    camp, data = passed_pilot
    got = read(data)
    rates = got["labelRates"]
    assert set(rates) == {"turns", "understanding"}
    assert sum(row["n"] for row in rates.values()) == 100
    for row in rates.values():
        assert sum(row["counts"].values()) == row["n"]
        assert sum(row["p"].values()) == pytest.approx(1)
    kinds = {camp.state().seeds[s].axes["evidenceKind"] for s in camp.state().items}
    assert set(rates) == kinds
    assert got["voids"] == {} and got["reruns"] == 0 and got["voteTie"] == 0
    assert got["screenHits"] == {"firstPass": {}, "regeneration": 0,
                                 "dropped": {"screenHits": 0, "formMismatch": {},
                                             "bindingMismatch": 0}}


def test_the_report_counts_voids_reruns_and_screen_hits(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from test_generation import leak_item
    camp = start_bench(tmp_path, monkeypatch)
    action = camp.next()
    leaking = camp.state().batches[action.batch].entries[0]
    run_gen(camp, action,
            lambda seed: leak_item(seed) if seed.seed_id == leaking else obedient_item(seed))
    spoiled = {"left": 1}

    def answer(seed: Any, vote: int) -> Any:
        if spoiled["left"] and vote == 1:                  # one answer outside the space: a void
            spoiled["left"] -= 1
            return "maybe"
        return faithful(seed, vote)
    drive(camp, answer)
    built = pilot.build_report(camp)
    assert built["screenHits"]["firstPass"] == {"screen": 1}
    assert built["screenHits"]["regeneration"] == 0
    assert list(built["voids"]) == ["label"] and sum(built["voids"]["label"].values()) == 1
    assert built["reruns"] == 1


def test_the_report_refuses_a_block_that_is_not_fully_voted(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    camp = start_bench(tmp_path, monkeypatch)
    with pytest.raises(PilotRefused, match=r"main block\(s\) \[1\] are not fully voted"):
        pilot.report(camp, tmp_path / "data")
    for _ in range(4):
        run_gen(camp, camp.next())
    run_vote(camp, camp.next(), lambda ref, vote: faithful(camp.state().seeds[ref], vote))
    with pytest.raises(PilotRefused, match="not fully voted"):
        pilot.report(camp, tmp_path / "data")
    assert not (tmp_path / "data").exists()


def test_the_report_refuses_while_fewer_than_80_items_carry_the_screen_vote(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pilot, "FIRST_BLOCK_SEEDS", 79)
    camp = start_bench(tmp_path, monkeypatch)
    for _ in range(4):
        run_gen(camp, camp.next())
    for _ in range(14):                              # 6 label runs of votes 1-3, 8 of the screen
        run_vote(camp, camp.next(), lambda ref, vote: faithful(camp.state().seeds[ref], vote))
    with pytest.raises(PilotRefused, match="fewer than 80 items"):
        pilot.report(camp, tmp_path / "data")


# --- the cue check, the recheck and the decision --------------------------------------------------


def failed_pilot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Campaign, Path]:
    camp = start_bench(tmp_path, monkeypatch)
    drive(camp, following_the_cue)
    data = tmp_path / "data"
    pilot.report(camp, data)
    return camp, data


def test_a_labeller_that_follows_the_cue_fails_the_check_and_halts_the_pilot(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    camp, data = failed_pilot(tmp_path, monkeypatch)
    got = read(data)
    cue = got["missingCueCheck"]
    assert cue["state"] == "failed" and cue["pilot"]["failed"] is True
    assert cue["pilot"]["sumX"] > 0 and cue["recheck"] is None
    assert sum(row["y"] for row in cue["pilot"]["strata"]) == 0       # concordant: all faithful
    assert [e["event"] for e in camp.events()].count("cue-check-halt") == 1
    stop = camp.next()
    assert stop.reason == "halted" and "cueCheck" in stop.detail and "None" not in stop.detail
    assert "HALTED: cueCheck (cueCheck)" in "\n".join(camp.status())
    projection = got["recheckProjection"]
    assert (projection["generationRuns"], projection["labelRuns"],
            projection["itemVotes"]) == (4, 6, 300)


def test_a_second_report_on_a_failed_pilot_records_no_second_halt(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    camp, data = failed_pilot(tmp_path, monkeypatch)
    pilot.report(camp, data)
    assert [e["event"] for e in camp.events()].count("cue-check-halt") == 1
    with pytest.raises(CampaignRefused, match="halt is open"):
        camp.record_cue_check_halt()
    camp.resolve(decision="reword the forms")
    with pytest.raises(CampaignRefused, match="already recorded"):
        camp.record_cue_check_halt()


def test_recheck_refuses_before_the_halt_is_resolved_and_unless_the_check_failed(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    camp, data = failed_pilot(tmp_path, monkeypatch)
    with pytest.raises(PilotRefused, match="no `driver resolve`"):
        pilot.recheck(camp, data)
    assert len(camp.state().rounds) == 1
    ok_camp = start_bench(tmp_path / "ok", monkeypatch)
    drive(ok_camp, faithful)
    with pytest.raises(PilotRefused, match="'passed', not 'failed'"):
        pilot.recheck(ok_camp, tmp_path / "ok-data")
    assert len(ok_camp.state().rounds) == 1


def test_a_resolved_failure_draws_the_recheck_block_voted_at_50_with_no_screen_vote(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    camp, data = failed_pilot(tmp_path, monkeypatch)
    camp.resolve(decision="reword the forms")
    assert camp.state().cue_halt_resolved and camp.state().halt is None
    pilot.recheck(camp, data)
    state = camp.state()
    assert [r["genRound"] for r in state.rounds] == [1, 2]
    block = state.rounds[1]
    assert block["pilotBlock"] == campaign.RECHECK_BLOCK and len(block["seeds"]) == 100
    assert state.round_votes(2) == (1, 2, 3)
    assert read(data)["missingCueCheck"]["state"] == "recheck-pending"
    with pytest.raises(PilotRefused, match="recheck block is not fully voted"):
        pilot.report(camp, data)
    drive(camp, faithful)
    state = camp.state()
    assert [label_batches(camp, 2, v) for v in (1, 2, 3)] == [[50, 50]] * 3
    assert (2, labels.SCREEN_VOTE) not in state.vote_starts
    assert pilot.screen(state.screen_votes([1, 2])).n == 100          # the recheck is not counted
    pilot.report(camp, data)
    got = read(data)
    assert got["missingCueCheck"]["state"] == "recheck-passed"
    assert got["missingCueCheck"]["recheck"]["failed"] is False
    assert got["missingCueCheck"]["pilot"]["failed"] is True          # both reach the package
    assert got["blocks"] == {"main": 1, "recheck": 1}
    assert got["rngSeeds"] == sorted(state.held_rng_seeds) and len(got["rngSeeds"]) == 9
    assert got["recheckProjection"]["itemVotes"] == 300
    assert got["labelRates"]["understanding"]["n"] + got["labelRates"]["turns"]["n"] == 100


def test_a_recheck_that_follows_the_cue_again_is_recheck_failed_with_no_second_halt(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    camp, data = failed_pilot(tmp_path, monkeypatch)
    camp.resolve(decision="reword the forms")
    pilot.recheck(camp, data)
    drive(camp, following_the_cue)
    pilot.report(camp, data)
    got = read(data)
    assert got["missingCueCheck"]["state"] == "recheck-failed"
    assert got["missingCueCheck"]["recheck"]["failed"] is True
    assert [e["event"] for e in camp.events()].count("cue-check-halt") == 1
    assert camp.next() == Stop("done", "nothing is pending")


def test_recheck_refuses_a_second_block_and_a_decided_pilot(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    camp, data = failed_pilot(tmp_path, monkeypatch)
    camp.resolve(decision="reword the forms")
    pilot.recheck(camp, data)
    with pytest.raises(PilotRefused, match="'recheck-pending', not 'failed'"):
        pilot.recheck(camp, data)
    other, other_data = failed_pilot(tmp_path / "b", monkeypatch)
    other.resolve(decision="reword the forms")
    pilot.record_decision(other_data, outcome="stop", cap=50, decided_on="2026-11-09",
                          ref="ledger line 9")
    with pytest.raises(PilotRefused, match="stakeholder has decided"):
        pilot.recheck(other, other_data)
    assert len(other.state().rounds) == 1


def test_a_cue_check_halt_takes_a_plain_resolve_and_refuses_skip_and_instruction_repins(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    camp, _ = failed_pilot(tmp_path, monkeypatch)
    with pytest.raises(CampaignRefused, match="safeguardStop"):
        camp.resolve(decision="x", skip_batch=True)
    with pytest.raises(CampaignRefused, match="no transcript"):
        camp.resolve(decision="x", repin_instructions=True)
    lines = camp.resolve(decision="reword the forms")
    assert lines[0].startswith("cleared: cueCheck (cueCheck)")
    assert camp.state().halt is None


# --- record-decision -----------------------------------------------------------------------------


def base_report(data: Path, state: str, **extra: Any) -> None:
    pilot_report(data, missingCueCheck={"state": state}, rngSeeds=[11, 12], cap=25, **extra)


def decide(data: Path, outcome: str = "proceed", **kw: Any) -> list[str]:
    args: dict[str, Any] = {"outcome": outcome, "cap": 25, "decided_on": "2026-11-09",
                            "ref": "ledger line 12"} | kw
    return pilot.record_decision(data, **args)


@pytest.mark.parametrize("state", ["passed", "recheck-passed"])
def test_proceed_is_recordable_when_the_check_passed(tmp_path: Path, state: str) -> None:
    base_report(tmp_path / "d", state)
    decide(tmp_path / "d", cap=50)
    got = read(tmp_path / "d")["stakeholderDecision"]
    assert got == {"outcome": "proceed", "cap": 50, "decidedOn": "2026-11-09",
                   "ref": "ledger line 12"}


def test_proceed_after_a_failed_recheck_needs_the_stakeholder_ref(tmp_path: Path) -> None:
    base_report(tmp_path / "d", "recheck-failed")
    with pytest.raises(PilotRefused, match="no stakeholderRef"):
        decide(tmp_path / "d")
    assert "stakeholderDecision" not in read(tmp_path / "d")
    decide(tmp_path / "d", stakeholder_ref="the stakeholder, 2026-11-09")
    assert read(tmp_path / "d")["stakeholderDecision"]["stakeholderRef"] == \
        "the stakeholder, 2026-11-09"


@pytest.mark.parametrize("state", ["failed", "recheck-pending"])
def test_proceed_is_refused_outside_the_spend_gate_even_with_a_stakeholder_ref(
        tmp_path: Path, state: str) -> None:
    base_report(tmp_path / "d", state)
    with pytest.raises(PilotRefused, match="does not pass the spend gate"):
        decide(tmp_path / "d", stakeholder_ref="the stakeholder")
    assert "stakeholderDecision" not in read(tmp_path / "d")


@pytest.mark.parametrize("state", ["passed", "failed", "recheck-pending", "recheck-failed"])
def test_a_first_stop_is_always_recordable(tmp_path: Path, state: str) -> None:
    base_report(tmp_path / "d", state)
    decide(tmp_path / "d", "stop")
    assert read(tmp_path / "d")["stakeholderDecision"]["outcome"] == "stop"


def test_the_decision_is_single_write_and_a_stop_is_final(tmp_path: Path) -> None:
    base_report(tmp_path / "d", "passed")
    decide(tmp_path / "d", "stop")
    with pytest.raises(PilotRefused, match="single-write"):
        decide(tmp_path / "d")
    assert read(tmp_path / "d")["stakeholderDecision"]["outcome"] == "stop"


def test_record_decision_keeps_the_rest_of_the_report(tmp_path: Path) -> None:
    base_report(tmp_path / "d", "passed")
    before = read(tmp_path / "d")
    decide(tmp_path / "d")
    after = read(tmp_path / "d")
    assert {k: v for k, v in after.items() if k != "stakeholderDecision"} == before


@pytest.mark.parametrize(("kw", "match"), [
    ({"cap": 30}, "--cap"),
    ({"decided_on": "yesterday"}, "YYYY-MM-DD"),
    ({"decided_on": "2026-13-01"}, "YYYY-MM-DD"),
    ({"ref": "  "}, "--ref"),
    ({"stakeholder_ref": " "}, "--stakeholder-ref"),
    ({"outcome": "maybe"}, "proceed or stop"),
])
def test_record_decision_refuses_a_bad_argument_and_writes_nothing(
        tmp_path: Path, kw: dict[str, Any], match: str) -> None:
    base_report(tmp_path / "d", "passed")
    before = (tmp_path / "d" / campaign.PILOT_REPORT).read_bytes()
    with pytest.raises(PilotRefused, match=match):
        decide(tmp_path / "d", **kw)
    assert (tmp_path / "d" / campaign.PILOT_REPORT).read_bytes() == before


def test_record_decision_refuses_without_a_report(tmp_path: Path) -> None:
    with pytest.raises(PilotRefused, match="no pilot report"):
        decide(tmp_path / "none")


def test_a_decided_report_is_not_rewritten_by_report(
        passed_pilot: tuple[Campaign, Path], tmp_path: Path) -> None:
    camp, data = passed_pilot
    copy = tmp_path / "data"
    (copy / "guard-judge" / "pilot").mkdir(parents=True)
    (copy / campaign.PILOT_REPORT).write_bytes((data / campaign.PILOT_REPORT).read_bytes())
    decide(copy)
    with pytest.raises(PilotRefused, match="final"):
        pilot.report(camp, copy)
    assert read(copy)["stakeholderDecision"]["outcome"] == "proceed"


def test_a_recorded_proceed_opens_init_full_with_its_cap_and_the_pilots_rng_seeds(
        passed_pilot: tuple[Campaign, Path], tmp_path: Path) -> None:
    camp, data = passed_pilot
    copy = tmp_path / "data"
    (copy / "guard-judge" / "pilot").mkdir(parents=True)
    (copy / campaign.PILOT_REPORT).write_bytes((data / campaign.PILOT_REPORT).read_bytes())
    decide(copy, cap=50)
    full = Campaign(tmp_path / "full", clock=Clock())
    full.init(point="guard-judge", phase="full", data_dir=copy)
    state = full.state()
    assert state.init["labelBatchCap"] == 50
    assert state.held_rng_seeds == camp.state().held_rng_seeds
    assert campaign.spend_gate_refusal(read(copy)) is None


# --- the command line ----------------------------------------------------------------------------


def test_the_command_line_runs_start_screen_report_and_record_decision(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]) -> None:
    camp = vote_bench(tmp_path, monkeypatch, cap=50)
    root, data = str(camp.root), str(tmp_path / "data")
    assert pilot.main(["start", "--root", root]) == 0
    assert "4 generation runs queued" in capsys.readouterr().out
    drive(camp, with_screen_gap(camp, 9))
    assert pilot.main(["screen", "--root", root]) == 0
    assert capsys.readouterr().out.splitlines()[0] == "n 100, delta 9, cap 50"
    assert pilot.main(["report", "--root", root, "--data-dir", data]) == 0
    assert "missingCueCheck: passed" in capsys.readouterr().out
    assert pilot.main(["record-decision", "--outcome", "proceed", "--cap", "50",
                       "--decided-on", "2026-11-09", "--ref", "ledger line 3",
                       "--data-dir", data]) == 0
    assert read(Path(data))["stakeholderDecision"]["cap"] == 50
    assert pilot.main(["record-decision", "--outcome", "stop", "--cap", "25",
                       "--decided-on", "2026-11-09", "--ref", "x", "--data-dir", data]) == 1
    assert "single-write" in capsys.readouterr().err


def test_the_command_line_refuses_with_one_line_and_exit_1(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert pilot.main(["screen", "--root", str(tmp_path / "nowhere")]) == 1
    assert capsys.readouterr().err.startswith("refused: no campaign at")


# --- the bound on added blocks (plan6 §3.8) and the fix round ----------------------------------


def screen_dropped(camp: Campaign, rounds: tuple[int, ...] | None = None) -> Answer:
    """Faithful, but the screen vote of `rounds` (all if `None`) answers outside the space, so
    each of its batches drops as `retryCapExceeded` and the round's `n` stays 0."""
    def answer(seed: Any, vote: int) -> Any:
        in_round = rounds is None or camp.state().seed_round[seed.seed_id] in rounds
        return "maybe" if vote == labels.SCREEN_VOTE and in_round else faithful(seed, vote)
    return answer


def block_halts(camp: Campaign) -> list[dict[str, Any]]:
    return [e for e in camp.events() if e["event"] == "pilot-block-halt"]


def main_rounds(camp: Campaign) -> list[int]:
    return [r["genRound"] for r in camp.state().rounds]


def test_added_blocks_stop_at_three_and_the_pilot_halts_with_pilot_blocks(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    camp = start_bench(tmp_path, monkeypatch)
    stop = drive(camp, screen_dropped(camp), limit=600)
    state = camp.state()
    assert stop.reason == "halted" and "pilotBlocks" in stop.detail
    assert main_rounds(camp) == [1, 2, 3, 4]                       # the first and three added
    assert [e["mainRounds"] for e in block_halts(camp)] == [4]
    assert state.halt["reason"] == "pilotBlocks" and state.halt["batch"] is None
    assert pilot.screen(state.screen_votes([1, 2, 3, 4])).n == 0
    assert "HALTED: pilotBlocks (pilotBlocks)" in "\n".join(camp.status())
    assert camp.next().reason == "halted"
    assert len(block_halts(camp)) == 1                             # a second `next` adds none
    with pytest.raises(PilotRefused, match="halted .pilotBlocks."):
        pilot.report(camp, tmp_path / "data")


def test_resolve_clears_the_block_halt_and_allows_exactly_three_more_then_halts_again(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    camp = start_bench(tmp_path, monkeypatch)
    drive(camp, screen_dropped(camp), limit=600)
    lines = camp.resolve(decision="the labeller was misconfigured")
    assert lines[0].startswith("cleared: pilotBlocks (pilotBlocks)")
    with pytest.raises(PilotRefused, match="draws another block"):
        pilot.report(camp, tmp_path / "data")
    stop = drive(camp, screen_dropped(camp), limit=600)
    assert stop.reason == "halted"
    assert main_rounds(camp) == [1, 2, 3, 4, 5, 6, 7]
    assert [e["mainRounds"] for e in block_halts(camp)] == [4, 7]


@pytest.mark.parametrize(("first", "n", "halts"), [(30, 80, False), (29, 79, True)])
def test_the_third_added_block_halts_at_79_items_and_not_at_80(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, first: int, n: int, halts: bool) -> None:
    monkeypatch.setattr(pilot, "FIRST_BLOCK_SEEDS", first)
    camp = start_bench(tmp_path, monkeypatch)
    stop = drive(camp, screen_dropped(camp, (2, 3)), limit=600)
    assert pilot.screen(camp.state().screen_votes([1, 2, 3, 4])).n == n
    assert main_rounds(camp) == [1, 2, 3, 4]
    assert bool(block_halts(camp)) is halts
    assert stop.reason == ("halted" if halts else "done")


def test_a_generator_repin_resets_the_block_epoch_and_the_cue_halt_bookkeeping(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    camp = start_bench(tmp_path, monkeypatch)
    drive(camp, screen_dropped(camp), limit=600)
    assert camp.state().block_halt_base == 4
    camp.resolve(decision="a new generator", repin_model=("generator", "claude-sonnet-5-5"))
    assert camp.state().block_halt_base == 0 and camp.state().rounds == []
    pilot.start(camp)
    stop = drive(camp, screen_dropped(camp), limit=600)
    assert stop.reason == "halted"
    assert [e["mainRounds"] for e in block_halts(camp)] == [4, 4]
    assert len(camp.state().rounds) == 4                           # three added blocks, not none


def test_a_generator_repin_after_a_cue_halt_leaves_the_next_pilot_its_own_halt(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    camp, _ = failed_pilot(tmp_path, monkeypatch)
    assert camp.state().cue_halts == 1
    camp.resolve(decision="a new generator", repin_model=("generator", "claude-opus-5-6"))
    state = camp.state()
    assert state.cue_halts == 0 and not state.cue_halt_resolved


def test_both_halt_recorders_refuse_while_a_halt_is_open(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    camp = start_bench(tmp_path / "a", monkeypatch)
    drive(camp, screen_dropped(camp), limit=600)                   # a pilotBlocks halt is open
    for record in (camp.record_cue_check_halt, camp.record_pilot_block_halt):
        with pytest.raises(CampaignRefused, match=r"halt is open \(pilotBlocks"):
            record()
    assert len(block_halts(camp)) == 1
    cued, _ = failed_pilot(tmp_path / "b", monkeypatch)            # a cueCheck halt is open
    for record in (cued.record_cue_check_halt, cued.record_pilot_block_halt):
        with pytest.raises(CampaignRefused, match=r"halt is open \(cueCheck"):
            record()
    assert [e["event"] for e in cued.events()].count("cue-check-halt") == 1
    assert not [e for e in cued.events() if e["event"] == "pilot-block-halt"]


def test_a_dropped_screen_batch_is_reported_apart_from_the_dropped_items(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    camp = start_bench(tmp_path, monkeypatch)
    drive(camp, screen_dropped(camp), limit=600)
    got = pilot._runs_report(camp.state())
    assert got["droppedScreenVote"] == {"items": 250}              # 100 + 3 x 50
    assert got["dropped"]["retryCapExceeded"] == {"seeds": 0, "items": 0}
    assert got["dropped"]["safeguardSkipped"] == {"seeds": 0, "items": 0}


def test_a_crash_after_3_of_10_screen_batches_queues_the_other_7_into_the_same_vote(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    camp = start_bench(tmp_path, monkeypatch)
    real = Campaign.queue_batch
    queued = {"screen": 0}

    def killed_at(self: Campaign, kind: str, entries: Any, **extra: Any) -> str:
        if kind == "label" and extra.get("vote") == labels.SCREEN_VOTE:
            queued["screen"] += 1
            if queued["screen"] == 4:
                raise RuntimeError("killed")
        return real(self, kind, entries, **extra)

    monkeypatch.setattr(Campaign, "queue_batch", killed_at)
    with pytest.raises(RuntimeError):
        drive(camp, faithful)
    monkeypatch.setattr(Campaign, "queue_batch", real)
    state = camp.state()
    start_event = next(e for e in camp.events()
                       if e["event"] == "vote-start" and e["vote"] == labels.SCREEN_VOTE)
    assert len(state.vote_batches(1, labels.SCREEN_VOTE)) == 3
    for _ in range(3):                                           # the three queued batches run
        run_vote(camp, camp.next(), lambda ref, vote: faithful(camp.state().seeds[ref], vote))
    state = camp.state()
    assert all(b.state == "ingested" for b in state.vote_batches(1, labels.SCREEN_VOTE))
    assert not state.round_voted(1)                              # the tail is not queued yet
    drive(camp, faithful)
    state = camp.state()
    assert label_batches(camp, 1, labels.SCREEN_VOTE) == [10] * 10
    held = [seed for b in state.vote_batches(1, labels.SCREEN_VOTE) for seed in b.entries]
    assert held == start_event["items"]                          # the recorded order, each once
    assert [e["vote"] for e in camp.events() if e["event"] == "vote-start"] == [1, 2, 3, 4]
    assert state.round_voted(1)


def test_a_recorded_cap_of_25_is_read_back_as_25_and_opens_init_full_at_25(
        tmp_path: Path) -> None:
    base_report(tmp_path / "d", "passed")
    decide(tmp_path / "d", cap=25)
    assert read(tmp_path / "d")["stakeholderDecision"]["cap"] == 25
    full = Campaign(tmp_path / "full", clock=Clock())
    full.init(point="guard-judge", phase="full", data_dir=tmp_path / "d")
    assert full.state().init["labelBatchCap"] == 25


def test_a_recheck_report_keeps_the_pilots_n_delta_cap_and_projection(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    camp, data = failed_pilot(tmp_path, monkeypatch)
    before = read(data)
    camp.resolve(decision="reword the forms")
    pilot.recheck(camp, data)
    drive(camp, faithful)
    pilot.report(camp, data)
    got = read(data)
    assert (got["n"], got["delta"], got["cap"]) == (100, 0, 50)
    assert got["reprojectedTokens"]["guard-judge"]["labelRuns"] == 63
    assert got["reprojectedTokens"]["guard-judge"]["generationRuns"] == 42
    assert (before["n"], before["cap"]) == (100, 50)


def test_the_report_counts_two_voids_of_one_reason_as_two(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    camp = start_bench(tmp_path, monkeypatch)

    def answer(seed: Any, vote: int) -> Any:
        state = camp.state()
        label_attempts = sum(1 for a in state.attempts if state.batches[a.batch].kind == "label")
        return "maybe" if label_attempts <= 2 else faithful(seed, vote)   # the first two runs void
    drive(camp, answer)
    got = pilot._runs_report(camp.state())
    assert list(got["voids"]) == ["label"] and list(got["voids"]["label"].values()) == [2]
    assert got["reruns"] == 2


def test_the_report_records_the_projections_plan_factor_and_assumptions(
        passed_pilot: tuple[Campaign, Path]) -> None:
    got = read(passed_pilot[1])
    assert got["planFactor"] == 1.3
    assert "1.3" in got["assumes"] and "no regeneration" in got["assumes"]
