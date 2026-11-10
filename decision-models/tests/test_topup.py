"""Rounds and top-up (typed-decision-models U5.9; plan4 `docs/plans/typed-decision-models4.md`
§3.6 "Top-up planner", "Spend halt", "Message-type tranche" and its §5 row U5.9; plan5
`typed-decision-models5.md` §3.2 and its U5.9 row; plan6 `typed-decision-models6.md` §3.8).

Rounds are built straight into the ledger (`synth_*`): a generation or label batch that `next`/
`queue_*` queued is completed by appending the `issued`, `generated`/`votes` and `outcome` events
a clean run would (the audit and the exchange are `test_campaign.py`'s and `test_votes.py`'s), so
a round of hundreds of items costs events, not staged transcripts. The labeller's answers are a
function of the seed, so a cell's count is whatever the test plants.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from test_campaign import GUARD_COMMAND, PROCEED, Clock, pilot_report
from test_votes import BOTH_AS_FIXTURE

from decisionmodels import campaign, driver, exchange, seeds
from decisionmodels.campaign import Campaign, CampaignRefused, Serve, Stop

TOKENS = {"input": 10, "cacheWrite": 20, "cacheRead": 30, "output": 40}   # 100 per run
MESSAGE_TYPES = ("order", "question", "complaint", "escalation", "other")


# --- building rounds in the ledger ---------------------------------------------------------------


def full_campaign(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, point: str) -> Campaign:
    """A `full` campaign for `point` with the fixture agent's expectations."""
    data = tmp_path / "data"
    pilot_report(data, stakeholderDecision=PROCEED, missingCueCheck={"state": "passed"},
                 rngSeeds=[11, 12], tokenProjection={"guard-judge": {"total": 10**12}})
    (data / "guard-judge" / "run").mkdir(parents=True, exist_ok=True)
    (data / "guard-judge" / "run" / "generation.json").write_text("{}", "utf-8")
    monkeypatch.setattr(exchange, "RENDERERS", {"message-type-classifier": lambda i: i["text"],
                                                "guard-judge": lambda i: json.dumps(i)})
    camp = Campaign(tmp_path / "camp", claude_dir=tmp_path / "claude", clock=Clock())
    camp.init(point=point, phase="full", data_dir=data)
    camp.record_expectations(guard_command=GUARD_COMMAND, roles=BOTH_AS_FIXTURE)
    return camp


def synth_run(camp: Campaign, batch_id: str, events: Callable[[str], list[dict[str, Any]]],
              tokens: dict[str, int] = TOKENS) -> None:
    """One clean ingested attempt of `batch_id`: `issued`, the run's own events, `outcome`."""
    batch = camp.state().batches[batch_id]
    attempt = len(batch.attempts) + 1
    opaque = f"op_{batch_id}_{attempt}"
    camp._append({"event": "issued", "batch": batch_id, "opaque": opaque,
                  "sessionId": f"s_{opaque}", "attempt": attempt, "requestSha256": "0" * 64,
                  "keys": {f"k{n}": ref for n, ref in enumerate(batch.entries)}})
    for event in events(opaque):
        camp._append(event)
    camp._append({"event": "outcome", "batch": batch_id, "opaque": opaque,
                  "outcome": "ingested", "voidClass": None, "details": [], "parts": [],
                  "produced": True, "run": {"tokens": dict(tokens)}})


def pending(camp: Campaign, kind: str) -> list[campaign.Batch]:
    return [b for b in camp.state().batches.values() if b.kind == kind and b.state == "pending"]


def message_item(seed: seeds.Seed) -> dict[str, Any]:
    return {"text": f"message {seed.seed_id}", "priorTurn": None}


def guard_item(seed: seeds.Seed) -> dict[str, Any]:
    return {"text": f"guard {seed.seed_id}"}


def generate_all(camp: Campaign, item_for: Callable[[seeds.Seed], dict[str, Any]],
                 tokens: dict[str, int] = TOKENS) -> int:
    """Every pending generation batch ingested, each seed accepted as `item_for(seed)`."""
    done = 0
    for batch in pending(camp, "generate"):
        state = camp.state()

        def events(opaque: str, batch: campaign.Batch = batch,
                   state: campaign.State = state) -> list[dict[str, Any]]:
            return [{"event": "generated", "opaque": opaque, "batch": batch.batch_id,
                     "items": [{"seedId": seed, "genRunId": batch.gen_run_id,
                                "genRound": state.seed_round[seed],
                                "item": item_for(state.seeds[seed])} for seed in batch.entries]}]

        synth_run(camp, batch.batch_id, events, tokens)
        done += 1
    return done


def vote_all_synth(camp: Campaign, answer: Callable[[campaign.State, str, int], Any],
                   tokens: dict[str, int] = TOKENS) -> int:
    """Queue and complete label batches (`queue_votes`) until it queues nothing more; returns the
    batches completed. `answer(state, item, vote)` is the labeller's vote."""
    done = 0
    while True:
        todo = pending(camp, "label")
        if not todo:
            if not camp.queue_votes():
                return done
            continue
        for batch in todo:
            state = camp.state()
            vote, gen_round = batch.extra["vote"], batch.extra["genRound"]

            def events(opaque: str, batch: campaign.Batch = batch, vote: int = vote,
                       state: campaign.State = state, gen_round: int = gen_round
                       ) -> list[dict[str, Any]]:
                return [{"event": "votes", "opaque": opaque, "batch": batch.batch_id,
                         "genRound": gen_round, "vote": vote,
                         "answers": {ref: answer(state, ref, vote) for ref in batch.entries}}]

            synth_run(camp, batch.batch_id, events, tokens)
            done += 1


def intended(state: campaign.State, ref: str, vote: int) -> Any:
    """The labeller agrees with the seed's intent."""
    return state.seeds[ref].axes["intendedAnswer"]


def events_of(camp: Campaign, kind: str) -> list[dict[str, Any]]:
    return [e for e in camp.events() if e["event"] == kind]


# --- the planner, pure ---------------------------------------------------------------------------


def cells(stratum: str | None, label: str, n: int) -> list[tuple[str | None, str]]:
    return [(stratum, label)] * n


def test_a_short_stratum_draws_the_max_over_its_short_cells_of_shortfall_over_the_label_rate(
) -> None:
    # guard: 200 per cell. `turns` holds 100 + 100 labelled: each cell is short by 100 at
    # P = 1/2, so ceil(100 / 0.5) = 200. `understanding` holds 180 true + 20 false of 200:
    # true is short by 20 at P = 0.9 (ceil(22.2) = 23), false by 180 at P = 0.1 (1800): the max.
    labelled = (cells("turns", "true", 100) + cells("turns", "false", 100)
                + cells("understanding", "true", 180) + cells("understanding", "false", 20))
    assert campaign.topup_counts("guard-judge", labelled) == {"turns": 200, "understanding": 1800}


def test_a_full_stratum_draws_nothing_and_a_full_point_plans_nothing() -> None:
    full = (cells("turns", "true", 200) + cells("turns", "false", 250)
            + cells("understanding", "true", 200) + cells("understanding", "false", 200))
    assert campaign.topup_counts("guard-judge", full) == {}
    only_one = (cells("turns", "true", 200) + cells("turns", "false", 200)
                + cells("understanding", "true", 190) + cells("understanding", "false", 250))
    # n(understanding) = 440: ceil(10 * 440 / 190) = ceil(23.2) = 24
    assert campaign.topup_counts("guard-judge", only_one) == {"understanding": 24}


def test_a_point_with_no_design_stratum_plans_one_pool_and_rounds_up() -> None:
    # message type: 120 per label (eval 60 + calibration 60). n = 520; order is short by 20 at
    # P = 100/520 (ceil(104.0) = 104), escalation by 60 at P = 60/520 (520 exactly).
    labelled = [(None, label) for label, n in zip(
        MESSAGE_TYPES, (100, 120, 120, 60, 120), strict=True) for _ in range(n)]
    assert campaign.topup_counts("message-type-classifier", labelled) == {None: 520}
    # 119 of 120 for one label out of 599: ceil(1 * 599 / 119) = ceil(5.03) = 6, not 5
    nearly = [(None, label) for label, n in zip(
        MESSAGE_TYPES, (119, 120, 120, 120, 120), strict=True) for _ in range(n)]
    assert campaign.topup_counts("message-type-classifier", nearly) == {None: 6}


def test_join_in_targets_follow_its_cells_100_per_label_eval_and_calibration_equal() -> None:
    assert campaign.cell_target("join-in-judge") == 200
    assert campaign.cell_target("guard-judge") == 200
    assert campaign.cell_target("message-type-classifier") == 120


def test_a_short_cell_with_no_labelled_item_is_refused_for_a_method_decision() -> None:
    labelled = [(None, label) for label in MESSAGE_TYPES[1:] for _ in range(10)]
    with pytest.raises(CampaignRefused, match=r"\(order, no stratum\).*P\(label \| stratum\) is 0"):
        campaign.topup_counts("message-type-classifier", labelled)


# --- the spend halt, pure ------------------------------------------------------------------------


def round_one(*, label: int, generation: int, items: int, runs: int) -> dict[str, Any]:
    """Round 1's totals with each side spread over distinct classes (the four sum to the figure)."""
    def spread(total: int) -> dict[str, int]:
        return {"input": total // 10, "cacheWrite": total // 5, "cacheRead": total * 3 // 10,
                "output": total - total // 10 - total // 5 - total * 3 // 10}
    return {"items": items, "generationRuns": runs, "generationTokens": spread(generation),
            "labelTokens": spread(label)}


def test_project_topup_reproduces_a_hand_computed_value_over_the_four_classes() -> None:
    # 100 items cost 30,000 label tokens, 4 generation runs 10,000 (2,500 each); a top-up of 40
    # items in 2 runs projects 40/100 * 30,000 + 2 * 2,500 = 17,000 against round 1's 40,000.
    result = campaign.project_topup(
        round_one=round_one(label=30000, generation=10000, items=100, runs=4),
        topup_items=40, topup_runs=2)
    assert result.projected == 17000 and result.round_one == 40000 and not result.halts


def test_the_projection_sums_all_four_token_classes_not_a_subset() -> None:
    classes = {"input": 1000, "cacheWrite": 2000, "cacheRead": 27000, "output": 0}
    spent = {"items": 100, "generationRuns": 1, "labelTokens": classes,
             "generationTokens": {"input": 0, "cacheWrite": 0, "cacheRead": 0, "output": 0}}
    result = campaign.project_topup(round_one=spent, topup_items=100, topup_runs=0)
    assert result.projected == 30000 == result.round_one


def test_exactly_half_of_round_one_does_not_halt_and_one_token_more_does() -> None:
    # items 300 -> 100 top-up items (a third of the labelling, 10,000 of 30,000), 4 -> 4 runs
    # (all of generation, 10,000): projected 20,000 = half of 40,000.
    even = campaign.project_topup(
        round_one=round_one(label=30000, generation=10000, items=300, runs=4),
        topup_items=100, topup_runs=4)
    assert (even.projected, even.round_one, even.halts) == (20000, 40000, False)
    over = campaign.project_topup(
        round_one=round_one(label=30000, generation=10001, items=300, runs=4),
        topup_items=100, topup_runs=4)
    assert (over.projected, over.round_one, over.halts) == (20001, 40001, True)


def test_a_round_one_with_nothing_to_project_from_is_refused() -> None:
    with pytest.raises(CampaignRefused, match="no accepted item or generation run"):
        campaign.project_topup(round_one=round_one(label=0, generation=0, items=0, runs=0),
                               topup_items=5, topup_runs=1)


# --- a planted guard campaign: no intent conditioning, pooled strata, repeated rounds ------------


def guard_round_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, flip: dict[str, int]
                    ) -> Campaign:
    """Round 1 (880 seeds) generated and voted, the labeller agreeing with intent except for the
    first `flip[kind]` `true` seeds of each evidence kind, which it labels `false`."""
    camp = full_campaign(tmp_path, monkeypatch, "guard-judge")
    camp.begin_round(1)
    generate_all(camp, guard_item)
    wrong = flipped_seeds(camp.state(), flip)
    vote_all_synth(camp, lambda st, ref, vote: False if ref in wrong else intended(st, ref, vote))
    return camp


def flipped_seeds(state: campaign.State, flip: dict[str, int]) -> set[str]:
    """The first `flip[kind]` round-1 `true` seeds (in id order) of each evidence kind."""
    return {seed for kind, n in flip.items() for seed in sorted(
        s for s, found in state.seeds.items()
        if found.design_stratum == kind and found.axes["intendedAnswer"] is True
        and state.seed_round[s] == 1)[:n]}


def test_round_one_is_220_per_cell_and_a_planted_shortfall_plans_the_stratum_alone(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    camp = guard_round_one(tmp_path, monkeypatch, flip={"understanding": 30})
    state = camp.state()
    cells_ = state.labelled_cells()
    assert cells_.count(("understanding", "true")) == 190
    assert cells_.count(("understanding", "false")) == 250
    assert cells_.count(("turns", "true")) == cells_.count(("turns", "false")) == 220
    assert campaign.topup_counts("guard-judge", cells_) == {"understanding": 24}


def test_a_short_true_cells_top_up_draws_the_designs_own_intent_mix_within_the_stratum(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    camp = guard_round_one(tmp_path, monkeypatch, flip={"understanding": 30})
    assert camp.queue_topup()
    state = camp.state()
    (start,) = [r for r in state.rounds if r["genRound"] == 2]
    top_up = [state.seeds[s] for s in start["seeds"]]
    assert start["designStratum"] == "understanding" and len(top_up) == 24
    assert {seed.design_stratum for seed in top_up} == {"understanding"}
    # the design weighs true and false 1:1, so 24 seeds are 12 and 12; a sampler that filters on
    # the short cell's intent would give 24 and 0. Difficulty keeps its 2:1 design too.
    assert sum(seed.axes["intendedAnswer"] is True for seed in top_up) == 12
    assert sum(seed.axes["difficulty"] == "clear" for seed in top_up) == 16


def test_the_samplers_signature_names_no_label_or_intent() -> None:
    import inspect
    draw = inspect.signature(seeds.draw_seeds).parameters
    assert list(draw) == ["role", "count", "rng_seed", "design_stratum"]
    assert not {"label", "intent", "intendedAnswer", "answer"} & set(draw)
    plan = inspect.signature(campaign.Campaign.begin_round).parameters
    assert not {"label", "intent", "intendedAnswer", "answer"} & set(plan)


def test_a_top_up_run_pools_every_short_stratum_of_the_round(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    camp = guard_round_one(tmp_path, monkeypatch, flip={"understanding": 30, "turns": 25})
    assert campaign.topup_counts("guard-judge", camp.state().labelled_cells()) == {
        "understanding": 24, "turns": 12}
    batches = camp.queue_topup()
    state = camp.state()
    starts = [r for r in state.rounds if r["genRound"] == 2]
    assert [r["designStratum"] for r in starts] == ["understanding", "turns"]
    assert [len(r["seeds"]) for r in starts] == [24, 12]
    assert len(batches) == 2 and all(state.batches[b].extra["generation"] == "first"
                                     for b in batches)
    for batch_id in batches:                      # 36 seeds, two runs of 18: both strata in each
        kinds = {state.seeds[s].design_stratum for s in state.batches[batch_id].entries}
        assert kinds == {"understanding", "turns"}
    assert state.drawn_in(2) == 36 and sum(len(state.batches[b].entries) for b in batches) == 36


def test_every_stratum_of_a_top_up_round_is_drawn_before_a_run_issues(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    camp = guard_round_one(tmp_path, monkeypatch, flip={"understanding": 30, "turns": 25})
    camp.queue_topup()
    events = camp.events()
    last_start = max(n for n, e in enumerate(events)
                     if e["event"] == "round-start" and e["genRound"] == 2)
    first_issue = next(n for n, e in enumerate(events)
                       if e["event"] == "queued" and n > last_start)
    assert last_start < first_issue                      # and nothing issued has `genRound` 2
    assert not any(e["event"] == "issued" and n > last_start for n, e in enumerate(events))


def test_a_crash_between_two_strata_draws_the_rest_before_any_run_issues(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    camp = guard_round_one(tmp_path, monkeypatch, flip={"understanding": 30, "turns": 25})
    real = Campaign.begin_round
    calls = {"n": 0}

    def killed_on_the_second(self: Campaign, *args: Any, **kwargs: Any) -> list[str]:
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("killed")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Campaign, "begin_round", killed_on_the_second)
    with pytest.raises(RuntimeError):
        camp.queue_topup()
    monkeypatch.setattr(Campaign, "begin_round", real)
    assert [r["designStratum"] for r in camp.state().rounds if r["genRound"] == 2] == [
        "understanding"]
    served = Campaign(camp.root, claude_dir=camp.claude_dir, clock=Clock()).next()
    assert isinstance(served, Serve) and served.kind == "generate"
    state = camp.state()
    assert [r["designStratum"] for r in state.rounds if r["genRound"] == 2] == [
        "understanding", "turns"]
    assert state.rounds[-1]["genRound"] == 2 and state.drawn_in(2) == 36
    kinds = {state.seeds[s].design_stratum for b in state.batches.values()
             for s in b.entries if state.seed_round.get(s) == 2}
    assert kinds == {"understanding", "turns"}


def test_two_top_up_rounds_draw_disjoint_seed_ids_and_never_reuse_an_rng_seed(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    camp = guard_round_one(tmp_path, monkeypatch, flip={"understanding": 30})
    round_one_rng = camp.events()[[e["event"] for e in camp.events()].index("seed-draw")][
        "draw"]["rngSeed"]
    camp.queue_topup()
    generate_all(camp, guard_item)
    # the labeller calls every round-2 true seed false, so the cell stays short and a third
    # round is planned: ceil(10 * 464 / 190) = 25
    wrong = flipped_seeds(camp.state(), {"understanding": 30})
    vote_all_synth(camp, lambda st, ref, vote: False if ref in wrong or st.seed_round[ref] == 2
                   else intended(st, ref, vote))
    assert camp.queue_topup()
    state = camp.state()
    draws = [e["draw"] for e in events_of(camp, "seed-draw")]
    assert [len(r["seeds"]) for r in state.rounds if r["genRound"] > 1] == [24, 25]
    ids = [[s["seedId"] for s in d["seeds"]] for d in draws]
    assert len(ids) == 3 and not (set(ids[0]) & set(ids[1])) and not (set(ids[1]) & set(ids[2]))
    assert not (set(ids[0]) & set(ids[2]))
    assert len({d["rngSeed"] for d in draws}) == 3
    # a draw on round 1's RNG seed is refused whatever its ids
    again = seeds.draw_seeds("guard-judge", 5, round_one_rng, design_stratum="understanding")
    with pytest.raises(CampaignRefused, match=f"RNG seed {round_one_rng} is already held"):
        camp.record_draw(again)
    # and a patched generator that offers a held value first is redrawn from, never used
    values = iter([round_one_rng, 4242])
    monkeypatch.setattr(campaign.secrets, "randbits", lambda bits: next(values))
    assert camp.fresh_rng_seed() == 4242


def test_nothing_is_planned_while_a_cell_is_full_or_a_round_is_unlabelled(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    camp = full_campaign(tmp_path, monkeypatch, "guard-judge")
    assert camp.queue_topup() == []                                  # no round yet
    camp.begin_round(1)
    generate_all(camp, guard_item)
    assert camp.queue_topup() == []                                  # settled, not labelled
    vote_all_synth(camp, intended)
    assert camp.queue_topup() == []                                  # 220 per cell: all full
    assert len(camp.state().rounds) == 1


# --- the spend halt and the decline path, through the campaign -----------------------------------


def message_round_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, target: int,
                      gen_tokens: dict[str, int] = TOKENS,
                      label_tokens: dict[str, int] = TOKENS) -> Campaign:
    """Message type's round 1 of 60 seeds (12 per label under the labeller, whatever the seeds
    intended), the cell target patched to `target` so the top-up's size is the test's."""
    monkeypatch.setattr(campaign, "cell_target", lambda role: target)
    camp = full_campaign(tmp_path, monkeypatch, "message-type-classifier")
    camp.begin_round(1, 60)
    generate_all(camp, message_item, gen_tokens)
    order = sorted(camp.state().items)
    by_label = {seed: MESSAGE_TYPES[n % 5] for n, seed in enumerate(order)}
    vote_all_synth(camp, lambda st, ref, vote: by_label[ref], label_tokens)
    assert camp.state().tranche_passed() and camp.state().round_voted(1)
    return camp


def test_a_top_up_inside_half_of_round_one_issues_with_no_halt(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # target 13: each label (12 of 60) is short by 1 -> ceil(1 * 60 / 12) = 5 seeds, one run.
    camp = message_round_one(tmp_path, monkeypatch, target=13)
    action = camp.next()
    assert isinstance(action, Serve) and action.kind == "generate"
    (plan,) = events_of(camp, "topup-plan")
    assert plan["genRound"] == 2 and plan["strata"] == [[None, 5]]
    assert events_of(camp, "spend-halt") == [] and camp.state().halt is None


def test_a_projection_over_half_halts_with_a_spend_halt_and_issues_nothing(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # target 25: short by 13 each -> ceil(13 * 60 / 12) = 65 seeds in 3 runs. Round 1 cost 3 runs
    # + 9 label runs (batches of 25, 25, 10 per vote), 100 each: 1,200; 65/60 * 900 + 3 * 100 =
    # 1,275 > 600.
    camp = message_round_one(tmp_path, monkeypatch, target=25)
    stop = camp.next()
    assert isinstance(stop, Stop) and stop.reason == "halted" and "spend" in stop.detail
    (halt,) = events_of(camp, "spend-halt")
    assert halt["genRound"] == 2 and halt["projection"]["halts"] is True
    assert halt["projection"]["roundOneTokens"] == 1200
    assert halt["projection"]["projectedTokens"] == pytest.approx(65 * 900 / 60 + 300)
    assert events_of(camp, "topup-plan") == []
    assert not any(r["genRound"] == 2 for r in camp.state().rounds)
    assert camp.state().halt["voidClass"] == "spend"
    assert any("HALTED: spend" in line for line in camp.status())


def test_the_projection_reads_this_points_round_one_and_nothing_outside_its_ledger(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The data directory (with the guard's pilot report and its token projection) is gone by the
    time the round is planned, and the figures follow this ledger's round 1: a round 1 whose
    generation runs cost ten times as much projects 10 x 3 runs' worth more."""
    camp = message_round_one(tmp_path / "a", monkeypatch, target=25)
    shutil.rmtree(tmp_path / "a" / "data")
    assert isinstance(camp.next(), Stop)
    pricey = {"input": 100, "cacheWrite": 200, "cacheRead": 300, "output": 400}
    other = message_round_one(tmp_path / "b", monkeypatch, target=25, gen_tokens=pricey)
    assert isinstance(other.next(), Stop)
    first, second = (events_of(c, "spend-halt")[0]["projection"] for c in (camp, other))
    assert (first["roundOneTokens"], second["roundOneTokens"]) == (1200, 900 + 3000)
    assert second["projectedTokens"] == pytest.approx(65 * 900 / 60 + 3 * 1000)


def test_the_four_classes_of_round_ones_ledger_all_count(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    camp = message_round_one(tmp_path, monkeypatch, target=13)
    spend = camp.state().spend(1)
    assert spend["items"] == 60 and spend["generationRuns"] == 3
    assert spend["generationTokens"] == {k: 3 * v for k, v in TOKENS.items()}
    assert sum(spend["labelTokens"].values()) == 900                 # 3 votes x 3 batches


def test_a_plain_resolve_lets_the_spend_halted_round_through_without_halting_again(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    camp = message_round_one(tmp_path, monkeypatch, target=25)
    assert isinstance(camp.next(), Stop)
    camp.resolve(decision="stakeholder: yes, spend it (ref 9)")
    action = camp.next()
    assert isinstance(action, Serve) and action.kind == "generate"
    assert len(events_of(camp, "spend-halt")) == 1
    assert events_of(camp, "topup-plan")[0]["genRound"] == 2
    assert camp.state().spend_approved == {2}


def test_after_decline_topup_no_top_up_run_is_issued_and_the_store_is_left_intact(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    camp = message_round_one(tmp_path, monkeypatch, target=25)
    assert isinstance(camp.next(), Stop)
    before = camp.ledger_path.read_bytes()
    lines = camp.resolve(decision="stakeholder: no (ref 9)", decline_topup=True)
    assert "declined" in lines[-1]
    for _ in range(3):
        assert camp.next() == Stop("done", "nothing is pending")
    after = camp.ledger_path.read_bytes()
    assert after.startswith(before) and after[len(before):].count(b"\n") == 1   # topup-declined
    assert camp.state().topup_declined and not any(
        r["genRound"] == 2 for r in camp.state().rounds)
    assert not [b for b in camp.state().batches.values() if b.state == "pending"]
    assert any("top-up declined" in line for line in camp.status())


def test_decline_topup_is_for_a_spend_halt_only_and_excludes_the_other_actions(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    camp = message_round_one(tmp_path, monkeypatch, target=13)
    with pytest.raises(CampaignRefused, match="not halted"):
        camp.resolve(decision="no", decline_topup=True)
    with pytest.raises(CampaignRefused, match="at most one"):
        camp.resolve(decision="no", decline_topup=True, skip_batch=True)


def test_the_driver_cli_takes_decline_topup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                            capsys: pytest.CaptureFixture[str]) -> None:
    camp = message_round_one(tmp_path, monkeypatch, target=25)
    assert isinstance(camp.next(), Stop)
    code = driver.main(["resolve", "--root", str(camp.root), "--decision", "no (ref 9)",
                        "--decline-topup"])
    assert code == 0 and "top-up declined" in capsys.readouterr().out
    assert camp.state().topup_declined
    with pytest.raises(SystemExit):                      # argparse: the actions are exclusive
        driver.main(["resolve", "--root", str(camp.root), "--decision", "x",
                     "--decline-topup", "--skip-batch"])


# --- the message-type tranche --------------------------------------------------------------------


@pytest.mark.parametrize(("drops", "size", "failed"), [
    (7, 250, False), (8, 250, True), (0, 250, False),          # 700 <= 750 < 800
    (3, 100, False), (4, 100, True),                           # the equality case: 300 = 3 * 100
])
def test_the_tranche_rule_is_100_drops_over_3_times_the_tranche(
        drops: int, size: int, failed: bool) -> None:
    assert campaign.tranche_failed(drops, size) is failed


def tranche_bench(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, seeds_count: int = 260
                  ) -> Campaign:
    """Message type's round 1 of `seeds_count` seeds, all accepted, no vote taken yet."""
    monkeypatch.setattr(campaign, "cell_target", lambda role: 13)
    camp = full_campaign(tmp_path, monkeypatch, "message-type-classifier")
    camp.begin_round(1, seeds_count)
    generate_all(camp, message_item)
    return camp


def tied(ties: set[str]) -> Callable[[campaign.State, str, int], Any]:
    """Every item answers `order` in every vote, except `ties`, which answer three labels."""
    return lambda st, ref, vote: MESSAGE_TYPES[vote - 1] if ref in ties else "order"


def test_8_ties_in_250_halt_with_a_rubric_halt_and_the_rest_waits(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    camp = tranche_bench(tmp_path, monkeypatch)
    assert camp.queue_votes()                                    # draws the tranche, queues vote 1
    state = camp.state()
    items = state.tranche_items()
    assert len(items) == 250 and set(items) <= set(state.items)
    vote_all_synth(camp, tied(set(items[:8])))
    state = camp.state()
    assert len(state.tranche_drops()) == 8
    assert state.halt["voidClass"] == "rubric" and state.halt["drops"] == 8
    assert state.halt["items"] == 250 and not state.tranche_passed()
    stop = camp.next()
    assert isinstance(stop, Stop) and stop.reason == "halted" and "rubric" in stop.detail
    assert [e["vote"] for e in events_of(camp, "vote-start")] == [1, 2, 3]
    assert all(e.get("tranche") for e in events_of(camp, "vote-start"))
    assert not state.round_voted(1)


def test_7_ties_in_250_pass_and_the_rest_votes_only_after_all_three_tranche_votes(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    camp = tranche_bench(tmp_path, monkeypatch)
    camp.queue_votes()
    items = camp.state().tranche_items()
    rest = sorted(set(camp.state().items) - set(items))
    vote_all_synth(camp, tied({*items[:7], *rest[:2]}))          # two ties in the rest, too
    state = camp.state()
    assert len(state.tranche_drops()) == 7 and len(state.vote_ties()) == 9
    assert state.tranche_passed() and state.halt is None
    assert any(line.startswith("tranche: 250 items, 7 voteTie drops, passed")
               for line in camp.status())
    starts = events_of(camp, "vote-start")
    assert [(bool(e.get("tranche")), e["vote"]) for e in starts] == [
        (True, 1), (True, 2), (True, 3), (False, 1), (False, 2), (False, 3)]
    assert all(set(e["items"]) == set(items) for e in starts[:3])
    assert all(set(e["items"]) == set(state.items) - set(items) for e in starts[3:])
    assert len(state.items) - len(items) == 10 and state.round_voted(1)
    assert [len(e["items"]) for e in starts[3:]] == [10, 10, 10]


def test_the_tranche_is_a_recorded_random_draw_on_a_held_rng_seed(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    camp = tranche_bench(tmp_path, monkeypatch)
    camp.queue_votes()
    (event,) = events_of(camp, "tranche-start")
    state = camp.state()
    assert event["rngSeed"] in state.held_rng_seeds
    every = sorted(state.items)
    shuffled = list(every)
    import random
    random.Random(f"tranche|{event['rngSeed']}").shuffle(shuffled)
    assert event["items"] == sorted(shuffled[:250]) != every[:250]
    assert event["rubric"] == campaign.rubric_hash("message-type-classifier")


def test_a_round_of_fewer_than_250_items_is_all_tranche_and_has_no_rest(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    camp = tranche_bench(tmp_path, monkeypatch, seeds_count=30)
    camp.queue_votes()
    vote_all_synth(camp, tied(set()))
    state = camp.state()
    assert len(state.tranche_items()) == 30 and state.round_voted(1)
    assert [bool(e.get("tranche")) for e in events_of(camp, "vote-start")] == [True] * 3


def test_a_resolve_needs_a_changed_rubric_then_relabels_the_tranche_under_it(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    camp = tranche_bench(tmp_path, monkeypatch)
    camp.queue_votes()
    items = camp.state().tranche_items()
    vote_all_synth(camp, tied(set(items[:8])))
    assert camp.state().halt["voidClass"] == "rubric"
    with pytest.raises(CampaignRefused, match="rubric is unchanged"):
        camp.resolve(decision="reworded")
    monkeypatch.setattr(campaign, "rubric_hash", lambda role: "f" * 64)
    camp.resolve(decision="rubric: tie-break order for escalation vs complaint")
    state = camp.state()
    assert state.halt is None and state.tranche["rubric"] == "f" * 64
    assert state.tranche["relabels"] == 1 and not state.tranche_passed()
    assert all(seed not in state.votes for seed in items)         # the tranche's votes are gone
    old = [b for b in state.batches.values() if b.kind == "label" and b.state == "retired"]
    assert len(old) == 3 * 10                                      # 10 batches x 3 votes of 25
    assert [b for b in state.batches.values() if b.state == "pending"] == []
    vote_all_synth(camp, tied(set()))                              # relabelled: no ties
    state = camp.state()
    assert state.tranche_passed() and state.halt is None and state.round_voted(1)
    starts = events_of(camp, "vote-start")
    assert [(bool(e.get("tranche")), e["vote"]) for e in starts] == [
        (True, 1), (True, 2), (True, 3), (True, 1), (True, 2), (True, 3),
        (False, 1), (False, 2), (False, 3)]
    assert len({e["rngSeed"] for e in starts}) == 9               # each relabel vote reshuffled


def test_a_relabelled_tranche_that_fails_again_halts_again(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    camp = tranche_bench(tmp_path, monkeypatch)
    camp.queue_votes()
    items = camp.state().tranche_items()
    vote_all_synth(camp, tied(set(items[:8])))
    monkeypatch.setattr(campaign, "rubric_hash", lambda role: "e" * 64)
    camp.resolve(decision="first fix")
    vote_all_synth(camp, tied(set(items[:9])))
    assert camp.state().halt["voidClass"] == "rubric" and camp.state().halt["drops"] == 9
    assert camp.state().tranche["relabels"] == 1


def test_the_guard_and_join_in_have_no_tranche_and_neither_does_a_later_round(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    camp = full_campaign(tmp_path, monkeypatch, "guard-judge")
    camp.begin_round(1, 30)
    generate_all(camp, guard_item)
    camp.queue_votes()
    assert events_of(camp, "tranche-start") == [] and not camp.state().tranche_applies(1)
    top = tranche_bench(tmp_path / "m", monkeypatch)
    assert top.state().tranche_applies(1) and not top.state().tranche_applies(2)


def test_a_generator_restart_resets_the_tranche_and_the_top_up_view(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    camp = tranche_bench(tmp_path, monkeypatch)
    camp.queue_votes()
    assert camp.state().tranche is not None
    camp._append({"event": "generation-restart", "action": "repin-model", "role": "generator",
                  "model": "claude-opus-5-9", "batch": None, "reason": None, "decision": "x"})
    state = camp.state()
    assert state.tranche is None and state.tranche_starts == {} and state.topup_plans == []
    assert state.spend_approved == set() and state.topup_declined is False


def test_status_prints_the_tranche_and_its_drops(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    camp = tranche_bench(tmp_path, monkeypatch)
    camp.queue_votes()
    vote_all_synth(camp, tied(set(camp.state().tranche_items()[:8])))
    assert any(line.startswith("tranche: 250 items, 8 voteTie drops, not passed")
               for line in camp.status())
