"""Generation runs, the screen's ledger path and form obedience at ingest (typed-decision-models
U5.8a).

Rows: plan4 `docs/plans/typed-decision-models4.md` §3.6, §5 U5.8 (the Screen part); plan5
`typed-decision-models5.md` §3.2, §3.5, §4 U5.8 (the Obedience and Screen parts). Votes, the pilot
and the cue check are U5.8b/U5.8c.

Seeds are hand-built from drawn ones, each with its own `condition` text and ids, so an item
checked against another seed's axes fails its binding: no fixture value is uniform across seeds.
Runs are staged from the clean transcript fixture (`test_campaign.stage`); the generator's declared
expectations are the labeler's values, which is the fixture's.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest
from test_campaign import GUARD_COMMAND, ROLES, Clock, ledger, outcomes, serve, stage, write_answer
from test_screen import obedient_item

from decisionmodels import campaign, seeds
from decisionmodels.campaign import Campaign, CampaignRefused, IngestResult, Stop
from decisionmodels.seeds import Seed, SeedDraw

#: The fixture's agent is `dm-labeler` on sonnet; a generator expecting that runs clean.
GENERATOR_AS_FIXTURE = {"generator": {**ROLES["labeler"]},
                        "labeler": {**ROLES["labeler"], "model": "claude-haiku-5-5"}}

LEAK = "(near-miss: the budget is withheld)"


def bench(tmp_path: Path) -> Campaign:
    camp = Campaign(tmp_path / "camp", claude_dir=tmp_path / "claude", clock=Clock())
    camp.init(point="guard-judge", phase="pilot")
    camp.record_expectations(guard_command=GUARD_COMMAND, roles=GENERATOR_AS_FIXTURE)
    return camp


_DRAWN = seeds.draw_seeds("guard-judge", 40, 3, design_stratum=None).seeds
_BASE = {kind: next(s for s in _DRAWN if s.axes["evidenceKind"] == kind)
         for kind in ("turns", "understanding")}


def hand_seed(number: int, kind: str = "understanding", form: str | None = "empty",
              turn_count: int = 2) -> Seed:
    """A seed with a unique id and `condition`; `form` only for `understanding`."""
    axes = {**_BASE[kind].axes, "condition": f"the user has stated budget number {number}",
            "missingForm": form if kind == "understanding" else None,
            "turnCount": turn_count if kind == "turns" else None}
    return dataclasses.replace(_BASE[kind], seed_id=f"sd_{number:012d}", axes=axes)


def hand_start(camp: Campaign, hand: Sequence[Seed], gen_round: int = 1) -> list[str]:
    """What `begin_round` records before it queues, for seeds the test built: the draw and
    `round-start`. Returns the seed ids in order."""
    rng_seed = camp.fresh_rng_seed()
    camp.record_draw(SeedDraw(role="guard-judge", rng_seed=rng_seed, count=len(hand),
                              design_stratum=None, seeds=tuple(hand)))
    ids = [seed.seed_id for seed in hand]
    camp._append({"event": "round-start", "genRound": gen_round, "rngSeed": rng_seed,
                  "designStratum": None, "seeds": ids})
    return ids


def hand_round(camp: Campaign, hand: Sequence[Seed], gen_round: int = 1) -> list[str]:
    """`hand_start`, then the balanced first-pass runs `begin_round` queues."""
    ids = hand_start(camp, hand, gen_round)
    return [camp.queue_batch("generate", run, generation="first")
            for run in campaign.balanced_runs(ids)]


def padded(hand: Sequence[Seed], total: int = 25, base: int = 900) -> list[Seed]:
    """`hand` plus obedient filler seeds up to `total` drawn, so one hit stays under 5% (a round
    of 25 tolerates one, of 40 two)."""
    return [*hand, *(hand_seed(base + n) for n in range(total - len(hand)))]


def drain(camp: Campaign, respond: Callable[[Seed], Any]) -> int:
    """Run every batch `next` serves (regeneration included) until it stops; returns the count."""
    runs = 0
    while not isinstance(camp.next(), Stop):
        run_generation(camp, respond)
        runs += 1
    return runs


def run_generation(camp: Campaign, respond: Callable[[Seed], Any]
                   ) -> tuple[campaign.Serve, IngestResult]:
    """next → issue → a clean staged run answering each seed with `respond(seed)` → ingest."""
    action = serve(camp)
    session = camp.issue(action)
    stage(camp, session, action.opaque)
    keys = camp.state().opaques[action.opaque].keys
    text = (camp.store / "req" / f"{action.opaque}.jsonl").read_text("utf-8")
    lines = [{"bodySha256": json.loads(text.splitlines()[0])["bodySha256"]}]
    lines += [{"key": key, "item": respond(camp.state().seeds[seed_id])}
              for key, seed_id in keys.items()]
    write_answer(camp, action.opaque,
                 "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines)
                 .encode("utf-8"))
    return action, camp.ingest(action.opaque)


def leaking(bad: dict[str, Callable[[Seed], dict]]) -> Callable[[Seed], dict]:
    """Obedient for every seed except those named in `bad`, which get the leaking item."""
    return lambda seed: bad[seed.seed_id](seed) if seed.seed_id in bad else obedient_item(seed)


def leak_item(seed: Seed) -> dict:
    item = obedient_item(seed)
    if item["path"] == "turns":
        item["turns"][0]["text"] += f" {LEAK}"
    else:
        item["understanding"]["known"] = [LEAK]
    return item


def form_item(seed: Seed) -> dict:
    """The seed's `missing` the wrong way round for its form."""
    item = obedient_item(seed)
    item["understanding"]["missing"] = [] if item["understanding"]["missing"] else ["x"]
    return item


def binding_item(seed: Seed) -> dict:
    item = obedient_item(seed)
    item["condition"] = "some other condition"
    return item


def events_of(camp: Campaign, kind: str) -> list[dict[str, Any]]:
    return [e for e in camp.events() if e["event"] == kind]


def items_by_seed(camp: Campaign) -> dict[str, dict[str, Any]]:
    return camp.state().items


# --- generation runs -----------------------------------------------------------------------------


def test_round_one_draws_1_1_times_the_target_in_runs_of_25_that_span_the_strata(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    camp = bench(tmp_path)
    monkeypatch.setattr(campaign.secrets, "randbits", lambda bits: 20261104)
    batches = camp.begin_round(1)
    state = camp.state()
    (draw,) = events_of(camp, "seed-draw")
    # guard: 4 cells x 100 eval, calibration equal, so 800; 1.1 x 800 = 880.
    assert draw["draw"]["count"] == len(state.seeds) == 880
    assert draw["draw"]["rngSeed"] == 20261104 and 20261104 in state.held_rng_seeds
    sizes = [len(state.batches[b].entries) for b in batches]
    assert sizes == [25] * 16 + [24] * 20                    # plan6 §3.8: balanced, no short tail
    queued = [seed for b in batches for seed in state.batches[b].entries]
    assert sorted(queued) == sorted(state.seeds) and queued != sorted(queued)
    for b in batches:
        kinds = {state.seeds[s].axes["evidenceKind"] for s in state.batches[b].entries}
        assert kinds == {"turns", "understanding"}, b
    assert {state.batches[b].extra["generation"] for b in batches} == {"first"}
    (start,) = events_of(camp, "round-start")
    assert start["genRound"] == 1 and sorted(start["seeds"]) == sorted(state.seeds)


def test_a_later_round_names_its_count_and_a_bad_round_is_refused(tmp_path: Path) -> None:
    camp = bench(tmp_path)
    with pytest.raises(CampaignRefused, match="explicit seed count"):
        camp.begin_round(2)
    for bad in (0, True, "1"):
        with pytest.raises(CampaignRefused, match="genRound"):
            camp.begin_round(bad, 5)  # type: ignore[arg-type]
    assert events_of(camp, "round-start") == [] and events_of(camp, "seed-draw") == []
    assert camp.begin_round(2, 5) and len(camp.state().rounds) == 1


def test_a_round_draws_on_a_fresh_rng_seed_and_a_held_one_is_never_reused(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    camp = bench(tmp_path)
    values = iter([77, 77, 78])
    monkeypatch.setattr(campaign.secrets, "randbits", lambda bits: next(values))
    camp.begin_round(1, 3)
    camp.begin_round(2, 3)
    assert sorted(camp.state().held_rng_seeds) == [77, 78]


def test_a_cuts_overflow_is_a_run_of_its_own_with_its_own_gen_run_id(tmp_path: Path) -> None:
    camp = bench(tmp_path)
    hand = [hand_seed(n) for n in range(1, 31)]
    ids = hand_start(camp, hand)
    batch = camp.queue_batch("generate", ids, generation="first")   # 30 in one batch: the cap cuts
    run_generation(camp, obedient_item)
    state = camp.state()
    first, overflow = state.batches[batch], state.batches["bat_0002"]
    assert (len(first.entries), len(overflow.entries)) == (25, 5)
    assert overflow.extra["generation"] == "first"
    assert first.gen_run_id != overflow.gen_run_id
    run_generation(camp, obedient_item)
    ids = {rec["seedId"]: rec["genRunId"] for rec in camp.state().items.values()}
    assert {ids[s] for s in first.entries} == {first.gen_run_id}
    assert {ids[s] for s in overflow.entries} == {overflow.gen_run_id}


def test_every_item_records_its_runs_gen_run_id_and_its_seeds_gen_round(tmp_path: Path) -> None:
    camp = bench(tmp_path)
    one = hand_round(camp, [hand_seed(n) for n in range(1, 31)], gen_round=1)
    two = hand_round(camp, [hand_seed(n) for n in range(101, 108)], gen_round=2)
    assert len(one) == 2 and len(two) == 1
    for _ in range(3):
        run_generation(camp, obedient_item)
    state = camp.state()
    assert camp.next() == Stop("done", "nothing is pending")
    run_ids = {b: state.batches[b].gen_run_id for b in (*one, *two)}
    assert len(set(run_ids.values())) == 3
    assert len(state.items) == 37
    for batch, gen_round in ((one[0], 1), (one[1], 1), (two[0], 2)):
        for seed_id in state.batches[batch].entries:
            record = state.items[seed_id]
            assert (record["genRunId"], record["genRound"]) == (run_ids[batch], gen_round)
            assert record["item"] == obedient_item(state.seeds[seed_id])
    (generated,) = [e for e in events_of(camp, "generated") if e["batch"] == two[0]]
    assert {r["genRound"] for r in generated["items"]} == {2}


def test_gen_run_ids_are_unique_across_a_whole_campaign(tmp_path: Path) -> None:
    camp = bench(tmp_path)
    camp.begin_round(1, 60)
    camp.begin_round(2, 30)
    ids = [b.gen_run_id for b in camp.state().batches.values()]
    assert len(ids) == len(set(ids)) == 3 + 2


# --- the screen's path ---------------------------------------------------------------------------


def test_a_screen_hit_queues_its_seed_and_only_its_seed(tmp_path: Path) -> None:
    camp = bench(tmp_path)
    hand = padded([hand_seed(n) for n in range(1, 11)])
    (batch,) = hand_round(camp, hand)
    target = hand[6]
    _, result = run_generation(camp, leaking({target.seed_id: leak_item}))
    assert result.stop is None
    (hit,) = events_of(camp, "screen-hit")
    assert (hit["seedId"], hit["batch"], hit["reasons"], hit["firstPass"]) == (
        target.seed_id, batch, ["screen"], True)
    assert hit["detail"] == "token:near-miss" and hit["genRunId"] == camp.state().batches[
        batch].gen_run_id
    state = camp.state()
    assert set(state.items) == {s.seed_id for s in hand} - {target.seed_id}
    assert state.awaiting_regeneration() == [target.seed_id]
    regen = serve(camp)
    assert [s.seed_id for s in regen.request.keys.values()] == [target.seed_id]
    assert camp.state().batches[regen.batch].extra["generation"] == "regen"


def test_regeneration_waits_for_the_first_pass_and_pools_the_rounds(tmp_path: Path) -> None:
    camp = bench(tmp_path)
    one = padded([hand_seed(1), hand_seed(2), hand_seed(3)], 40)      # runs of 25 and 15
    two = padded([hand_seed(11), hand_seed(12)], 25, base=700)
    runs = hand_round(camp, one, gen_round=1) + hand_round(camp, two, gen_round=2)
    assert len(runs) == 3
    bad = {one[0].seed_id: leak_item, two[1].seed_id: leak_item}
    run_generation(camp, leaking(bad))
    assert camp.state().awaiting_regeneration() == [one[0].seed_id]
    action, _ = run_generation(camp, leaking(bad))           # the next first-pass run, not a regen
    assert action.batch == runs[1]
    run_generation(camp, leaking(bad))
    regen = serve(camp)
    assert [s.seed_id for s in regen.request.keys.values()] == [one[0].seed_id, two[1].seed_id]


def test_a_regenerated_item_keeps_its_seeds_gen_round_and_takes_the_new_gen_run_id(
        tmp_path: Path) -> None:
    camp = bench(tmp_path)
    one = padded([hand_seed(n) for n in range(1, 5)])
    two = padded([hand_seed(n) for n in range(11, 15)], 25, base=700)
    first = hand_round(camp, one, gen_round=1)[0]
    second = hand_round(camp, two, gen_round=2)[0]
    bad = {one[1].seed_id: leak_item, two[3].seed_id: leak_item}
    run_generation(camp, leaking(bad))
    run_generation(camp, leaking(bad))
    action, result = run_generation(camp, obedient_item)     # the regeneration run, pooled
    assert result.stop is None and action.batch not in (first, second)
    state = camp.state()
    new_run = state.batches[action.batch].gen_run_id
    old = {first: state.batches[first].gen_run_id, second: state.batches[second].gen_run_id}
    a, b = state.items[one[1].seed_id], state.items[two[3].seed_id]
    assert (a["genRound"], b["genRound"]) == (1, 2)
    assert a["genRunId"] == b["genRunId"] == new_run and new_run not in old.values()
    assert state.items[one[0].seed_id]["genRunId"] == old[first]
    assert state.items[two[0].seed_id]["genRunId"] == old[second]
    assert len(state.items) == 50 and camp.next() == Stop("done", "nothing is pending")


def test_the_third_hit_drops_the_seed_and_counts_it_under_screen_hits(tmp_path: Path) -> None:
    camp = bench(tmp_path)
    hand = padded([hand_seed(n) for n in range(1, 6)])
    hand_round(camp, hand)
    target = hand[2]
    assert drain(camp, leaking({target.seed_id: leak_item})) == 3
    state = camp.state()
    assert [h["reasons"] for h in state.hits[target.seed_id]] == [["screen"]] * 3
    assert [h["firstPass"] for h in state.hits[target.seed_id]] == [True, False, False]
    assert target.seed_id not in state.items and set(state.dropped_seeds) == {target.seed_id}
    assert state.generation_drops() == {"screenHits": 1, "formMismatch": {}, "bindingMismatch": 0}
    assert camp.next() == Stop("done", "nothing is pending")
    assert len(events_of(camp, "screen-hit")) == 3
    assert any("screenHits" in line for line in camp.status())


def test_a_seeds_two_hits_regenerate_and_an_obedient_third_try_is_kept(tmp_path: Path) -> None:
    camp = bench(tmp_path)
    hand = padded([hand_seed(n) for n in range(1, 4)])
    hand_round(camp, hand)
    target = hand[0]
    for _ in range(2):
        run_generation(camp, leaking({target.seed_id: leak_item}))
    run_generation(camp, obedient_item)
    state = camp.state()
    assert target.seed_id in state.items and state.dropped_seeds == {}


def test_the_cap_counts_every_reason_and_the_drop_takes_the_third_hits(tmp_path: Path) -> None:
    camp = bench(tmp_path)
    hand = padded([hand_seed(1, form="listed"), hand_seed(2, form="empty"),
                   hand_seed(3, form="indirect")], 40)
    hand_round(camp, hand)
    # seed 1: screen, form, binding -> dropped as bindingMismatch; seed 2: form, form, form ->
    # dropped as formMismatch/empty; seed 3: obedient.
    plan = {1: [leak_item, form_item, binding_item], 2: [form_item] * 3}
    seen: dict[str, int] = {}

    def respond(seed: Seed) -> dict:
        number = int(seed.seed_id[3:])
        if number not in plan:
            return obedient_item(seed)
        turn = seen.get(seed.seed_id, 0)
        seen[seed.seed_id] = turn + 1
        return plan[number][turn](seed)

    assert drain(camp, respond) == 2 + 2
    state = camp.state()
    assert [h["reasons"][0] for h in state.hits[hand[0].seed_id]] == [
        "screen", "formMismatch", "bindingMismatch"]
    assert state.generation_drops() == {
        "screenHits": 0, "formMismatch": {"empty": 1}, "bindingMismatch": 1}
    assert set(state.items) == {s.seed_id for s in hand[2:]}
    assert [(e["event"], e["seedId"][3:].lstrip("0")) for e in camp.events()
            if e["event"].endswith(("-hit", "-mismatch"))] == [
        ("screen-hit", "1"), ("form-mismatch", "2"), ("form-mismatch", "1"),
        ("form-mismatch", "2"), ("binding-mismatch", "1"), ("form-mismatch", "2")]


# --- form obedience ------------------------------------------------------------------------------


@pytest.mark.parametrize("form", ["empty", "immaterial", "indirect", "listed", "unlisted",
                                  "ambiguous"])
def test_a_form_mismatch_regenerates_and_an_obedient_item_passes(
        tmp_path: Path, form: str) -> None:
    camp = bench(tmp_path)
    bound, other = hand_seed(1, form=form), hand_seed(2, form="empty")
    hand = padded([bound, other])
    hand_round(camp, hand)
    run_generation(camp, leaking({bound.seed_id: form_item}))
    (hit,) = events_of(camp, "form-mismatch")
    assert (hit["seedId"], hit["form"], hit["reasons"]) == (bound.seed_id, form, ["formMismatch"])
    assert set(camp.state().items) == {s.seed_id for s in hand} - {bound.seed_id}
    assert camp.state().awaiting_regeneration() == [bound.seed_id]
    run_generation(camp, obedient_item)
    assert set(camp.state().items) == {s.seed_id for s in hand}
    assert len(events_of(camp, "form-mismatch")) == 1


def test_the_four_error_wrong_kind_item_is_one_binding_mismatch_and_no_form_mismatch(
        tmp_path: Path) -> None:
    camp = bench(tmp_path)
    seed = hand_seed(1, form="listed")
    other = hand_seed(2, kind="turns", turn_count=3)
    hand_round(camp, padded([seed, other]))

    def wrong_kind(s: Seed) -> dict:
        if s.seed_id != seed.seed_id:
            return obedient_item(s)
        item = obedient_item(hand_seed(9, kind="turns", turn_count=2))   # a turns item
        return item                                          # also another condition: 4 errors

    run_generation(camp, wrong_kind)
    (hit,) = events_of(camp, "binding-mismatch")
    assert hit["seedId"] == seed.seed_id and hit["reasons"] == ["bindingMismatch"]
    assert events_of(camp, "form-mismatch") == [] and events_of(camp, "screen-hit") == []
    state = camp.state()
    assert len(state.hits[seed.seed_id]) == 1
    assert state.round_hits == {(1, "bindingMismatch"): 1} and state.halt is None


def test_a_screen_hit_beats_a_binding_error_for_the_cap_and_the_item_counts_under_each(
        tmp_path: Path) -> None:
    camp = bench(tmp_path)
    seed = hand_seed(1, form="empty")
    hand_round(camp, padded([seed] + [hand_seed(n) for n in range(2, 5)]))

    def both(s: Seed) -> dict:
        item = leak_item(s)
        item["condition"] = "another"
        return item

    run_generation(camp, leaking({seed.seed_id: both}))
    (hit,) = events_of(camp, "screen-hit")
    assert hit["reasons"] == ["screen", "bindingMismatch"]
    assert events_of(camp, "binding-mismatch") == []
    state = camp.state()
    assert len(state.hits[seed.seed_id]) == 1                  # one hit toward the cap
    assert state.round_hits == {(1, "screen"): 1, (1, "bindingMismatch"): 1}


def test_a_schema_failure_voids_the_run_and_records_nothing(tmp_path: Path) -> None:
    camp = bench(tmp_path)
    hand = [hand_seed(n) for n in range(1, 5)]
    (batch,) = hand_round(camp, hand)

    def broken(seed: Seed) -> dict:
        item = leak_item(seed) if seed.seed_id == hand[0].seed_id else obedient_item(seed)
        if seed.seed_id == hand[2].seed_id:
            del item["condition"]
        return item

    _, result = run_generation(camp, broken)
    assert result.outcome["outcome"] == "void" and result.outcome["reason"] == "outOfSpace"
    state = camp.state()
    assert state.items == {} and state.hits == {} and state.round_hits == {}
    assert state.batches[batch].state == "pending" and state.batches[batch].counted == 1
    assert [e for e in camp.events() if e["event"] in ("generated", "screen-hit")] == []
    run_generation(camp, obedient_item)                       # the re-issue is checked afresh
    assert len(camp.state().items) == 4


# --- the round halt ------------------------------------------------------------------------------


def test_four_percent_screen_plus_form_and_four_percent_form_only_halt_on_form_not_screen(
        tmp_path: Path) -> None:
    camp = bench(tmp_path)
    hand = [hand_seed(n, form="empty") for n in range(1, 51)]  # 50 drawn: two runs of 25
    first, _ = hand_round(camp, hand)
    run_one = camp.state().batches[first].entries

    def screen_and_form(seed: Seed) -> dict:
        item = form_item(seed)
        item["understanding"]["known"] = [LEAK]
        return item

    # Per run: one screen-plus-form item and one form-only item. After run 1 the form reason has
    # 2 hits (8% of the 25 seeds ingested, and of the 50 drawn only 4%); screen has 1.
    bad = {run_one[0]: screen_and_form, run_one[1]: form_item,
           hand[25].seed_id: screen_and_form, hand[26].seed_id: form_item}
    _, result = run_generation(camp, leaking(bad))
    assert result.stop == "halted"
    halt = camp.state().halt
    assert halt["reason"] == "formMismatch" and halt["reasons"] == ["formMismatch"]
    assert halt["voidClass"] == "template" and halt["pooled"] is True
    assert camp.state().round_hits == {(1, "screen"): 1, (1, "formMismatch"): 2}
    assert [e["event"] for e in camp.events() if e["event"].endswith(("-hit", "-mismatch"))] == [
        "screen-hit", "form-mismatch"]
    assert camp.next() == Stop("halted", "formMismatch (template) on bat_0001")
    assert any("HALTED" in line for line in camp.status())
    camp.resolve(decision="fixed the form text")
    _, result = run_generation(camp, leaking(bad))           # run 2: screen reaches 2 of 50 = 4%
    halt = camp.state().halt                                  # the round's own pair is new
    assert halt["reasons"] == ["formMismatch"] and halt["pooled"] is False
    assert halt["genRound"] == 1 and result.stop == "halted"
    assert camp.state().round_hits == {(1, "screen"): 2, (1, "formMismatch"): 4}


@pytest.mark.parametrize(("hits", "halts"), [(2, False), (3, True)])
def test_the_halt_fires_after_the_first_run_past_five_percent_and_exactly_five_does_not(
        tmp_path: Path, hits: int, halts: bool) -> None:
    camp = bench(tmp_path)
    hand = [hand_seed(n) for n in range(1, 41)]              # 40 drawn: two runs of 20
    first, second = hand_round(camp, hand)
    in_first = {hand[0].seed_id: binding_item}               # spread, or the pooled halt fires
    in_second = {s.seed_id: binding_item for s in hand[20:20 + hits - 1]}
    _, one = run_generation(camp, leaking(in_first))
    assert one.stop is None and camp.state().halt is None
    _, two = run_generation(camp, leaking(in_second))
    assert (two.stop == "halted") is halts                    # 2 of 40 is exactly 5%
    assert (camp.state().halt is not None) is halts
    assert camp.state().round_hits == {(1, "bindingMismatch"): hits}
    assert camp.state().batches[second].state == "ingested" and first


def test_the_halt_is_against_the_rounds_own_drawn_seeds_not_the_campaigns(tmp_path: Path) -> None:
    camp = bench(tmp_path)
    hand_round(camp, [hand_seed(n) for n in range(1, 21)], gen_round=1)     # 20 drawn
    hand_round(camp, [hand_seed(n) for n in range(101, 181)], gen_round=2)  # 80 drawn
    two = {hand_seed(n).seed_id: binding_item for n in (1, 2)}              # 10% of round 1
    _, result = run_generation(camp, leaking(two))
    assert result.stop == "halted" and camp.state().halt["genRound"] == 1


def test_regeneration_hits_count_toward_the_cap_only(tmp_path: Path) -> None:
    camp = bench(tmp_path)
    hand = [hand_seed(n) for n in range(1, 41)]
    hand_round(camp, hand)                                   # two runs of 20
    bad = {hand[0].seed_id: binding_item, hand[20].seed_id: binding_item}
    run_generation(camp, leaking(bad))
    run_generation(camp, leaking(bad))                       # 2 of 40: exactly 5%
    assert camp.state().halt is None
    drain(camp, leaking(bad))                  # both hit again, twice, in regeneration
    state = camp.state()
    assert state.halt is None and state.round_hits == {(1, "bindingMismatch"): 2}
    (pooled,) = state.pool.values()
    assert pooled["hits"] == {"bindingMismatch": 2} and pooled["seeds"] == 40
    assert [len(state.hits[s.seed_id]) for s in (hand[0], hand[20])] == [3, 3]
    assert set(state.dropped_seeds) == {hand[0].seed_id, hand[20].seed_id}


def test_a_template_halt_clears_with_resolve_and_does_not_fire_again_for_its_round(
        tmp_path: Path) -> None:
    camp = bench(tmp_path)
    hand = [hand_seed(n) for n in range(1, 51)]
    first, second = hand_round(camp, hand)
    bad = {s.seed_id: binding_item for s in hand[:4]}
    _, one = run_generation(camp, leaking(bad))
    assert one.stop == "halted" and isinstance(camp.next(), Stop)
    camp.resolve(decision="the template was fixed")
    assert camp.state().halt is None
    more = {hand[25].seed_id: binding_item}
    action, two = run_generation(camp, leaking(more))
    assert action.batch == second and two.stop is None and camp.state().halt is None
    assert camp.state().batches[first].state == "ingested"


def test_two_reasons_over_five_percent_fire_together(tmp_path: Path) -> None:
    camp = bench(tmp_path)
    hand = [hand_seed(n, form="empty") for n in range(1, 21)]
    hand_round(camp, hand)
    bad = {hand[0].seed_id: binding_item, hand[1].seed_id: binding_item,
           hand[2].seed_id: form_item, hand[3].seed_id: form_item}
    run_generation(camp, leaking(bad))
    halt = camp.state().halt
    assert halt["reasons"] == ["bindingMismatch", "formMismatch"] and halt["reason"] == (
        "bindingMismatch")


# --- the ledger is the state ---------------------------------------------------------------------


def test_events_an_outcome_has_not_committed_change_nothing(tmp_path: Path) -> None:
    camp = bench(tmp_path)
    hand = [hand_seed(n) for n in range(1, 4)]
    hand_round(camp, hand)
    action = serve(camp)
    camp.issue(action)
    camp._append({"event": "generated", "opaque": action.opaque, "batch": "bat_0001",
                  "items": [{"seedId": hand[0].seed_id, "genRunId": "gen_x", "genRound": 1,
                             "item": {}}]})
    camp._append({"event": "screen-hit", "opaque": action.opaque, "batch": "bat_0001",
                  "seedId": hand[1].seed_id, "genRunId": "gen_x", "reasons": ["screen"],
                  "detail": "x", "form": None, "firstPass": True})
    state = camp.state()
    assert state.items == {} and state.hits == {} and state.round_hits == {}


def test_a_batch_queued_without_a_generation_pass_is_not_checked(tmp_path: Path) -> None:
    """`queue_batch("generate", …)` alone (the low-level call older tests use) records no round,
    so the ingest path leaves such a batch to its caller."""
    camp = bench(tmp_path)
    draw = seeds.draw_seeds("guard-judge", 2, 5, design_stratum=None)
    camp.record_draw(draw)
    camp.queue_batch("generate", [s.seed_id for s in draw.seeds])
    before = ledger(camp)
    run_generation(camp, obedient_item)
    assert [e["event"] for e in camp.events()][len(before.splitlines()):] == [
        "issued", "outcome"]
    assert outcomes(camp)[-1]["outcome"] in ("ingested", "void")
