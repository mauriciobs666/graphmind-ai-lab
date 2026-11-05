"""Round halts, the template pool, issue-time expectations and the generator restart
(typed-decision-models U5.8a, plan6 `docs/plans/typed-decision-models6.md` §3.8 and its §5 row
"U5.8a, U5.9"), plus the survivors Pass 18 found in `test_generation.py`.

Helpers and the fixture-run machinery are `test_generation.py`'s: hand-built seeds with their own
`condition` and ids, runs staged from the clean transcript fixture.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from test_campaign import (
    GUARD_COMMAND,
    answer_bytes,
    run_batch,
    serve,
    smoke,
    stage,
    write_answer,
)
from test_generation import (
    GENERATOR_AS_FIXTURE,
    bench,
    binding_item,
    drain,
    form_item,
    hand_round,
    hand_seed,
    leak_item,
    leaking,
    padded,
    run_generation,
    votes_next,
)
from test_screen import MESSAGE_TYPE, obedient_item

from decisionmodels import campaign, screen, seeds
from decisionmodels.campaign import Campaign, CampaignRefused, Stop, generator_template_hash


def prompts(**generator: str) -> dict[str, Any]:
    """A `promptSha256` whose generator entries are `generator`'s (the others fixed)."""
    entries = {"agentDefinition": "a0", "spawnTemplate": "s0", "requestTemplate": "r0"}
    entries.update(generator)
    return {key: {"generator": value, "labeler": "x", "driver": "y"}
            for key, value in entries.items()}


def re_install(camp: Campaign, **kwargs: Any) -> None:
    camp.record_expectations(guard_command=GUARD_COMMAND, roles=GENERATOR_AS_FIXTURE, **kwargs)


def ledger_events(camp: Campaign, *kinds: str) -> list[dict[str, Any]]:
    return [e for e in camp.events() if e["event"] in kinds]


def bad_binding(*numbers: int) -> dict[str, Any]:
    return {hand_seed(n).seed_id: binding_item for n in numbers}


# --- the balanced cut ----------------------------------------------------------------------------


@pytest.mark.parametrize(("count", "sizes"), [
    (0, []), (1, [1]), (25, [25]), (26, [13, 13]), (51, [17, 17, 17]), (75, [25, 25, 25]),
    (76, [19, 19, 19, 19]), (880, [25] * 16 + [24] * 20),
])
def test_the_balanced_cut_makes_ceil_n_over_25_runs_differing_by_at_most_one(
        count: int, sizes: list[int]) -> None:
    ids = [f"s{n}" for n in range(count)]
    runs = campaign.balanced_runs(ids)
    assert [len(r) for r in runs] == sizes
    assert [seed for run in runs for seed in run] == ids


def test_a_regeneration_pool_is_cut_balanced_too(tmp_path: Path) -> None:
    camp = bench(tmp_path)
    hand = [hand_seed(n) for n in range(1, 31)]                # two runs of 15
    hand_round(camp, hand)
    everything = {s.seed_id: binding_item for s in hand}
    run_generation(camp, leaking(everything))                   # halts: the pair is spent below
    camp.resolve(decision="the template was fixed")
    run_generation(camp, leaking(everything))
    assert camp.state().halt is None and len(camp.state().awaiting_regeneration()) == 30
    assert isinstance(camp.next(), campaign.Serve)
    sizes = [len(b.entries) for b in camp.state().batches.values()
             if b.extra.get("generation") == "regen"]
    assert sizes == [15, 15]


# --- the halt: floor, denominator, pooling -------------------------------------------------------


@pytest.mark.parametrize(("hits", "halts"), [(1, False), (2, True)])
def test_a_round_of_24_seeds_halts_at_two_hits_and_not_at_one(
        tmp_path: Path, hits: int, halts: bool) -> None:
    camp = bench(tmp_path)
    hand = [hand_seed(n) for n in range(1, 25)]
    hand_round(camp, hand)
    _, result = run_generation(camp, leaking(bad_binding(*range(1, hits + 1))))
    assert (result.stop == "halted") is halts


def test_two_round_starts_with_one_gen_round_share_one_denominator(tmp_path: Path) -> None:
    camp = bench(tmp_path)
    first = [hand_seed(n) for n in range(1, 26)]
    second = [hand_seed(n) for n in range(101, 126)]
    hand_round(camp, first, gen_round=1)
    hand_round(camp, second, gen_round=1)
    state = camp.state()
    assert state.drawn_in(1) == 50 and len(state.rounds) == 2
    assert set(state.seed_round.values()) == {1}
    run_generation(camp, leaking(bad_binding(1)))
    _, two = run_generation(camp, leaking(bad_binding(101)))
    assert two.stop is None                                    # 2 of 50 is exactly 4%
    assert camp.state().round_hits == {(1, "bindingMismatch"): 2}


def test_a_skipped_or_repeated_gen_round_keys_its_seeds_by_that_number(tmp_path: Path) -> None:
    camp = bench(tmp_path)
    hand_round(camp, [hand_seed(n) for n in range(1, 4)], gen_round=1)
    hand_round(camp, [hand_seed(n) for n in range(11, 14)], gen_round=3)
    state = camp.state()
    assert {state.seed_round[hand_seed(n).seed_id] for n in range(1, 4)} == {1}
    assert {state.seed_round[hand_seed(n).seed_id] for n in range(11, 14)} == {3}
    assert state.drawn_in(2) == 0 and state.drawn_in(3) == 3
    for _ in range(2):
        run_generation(camp, leaking(bad_binding(11)))
    assert {k[0] for k in camp.state().round_hits} == {3}
    items = camp.state().items
    assert {items[hand_seed(n).seed_id]["genRound"] for n in range(1, 4)} == {1}


def small_rounds(camp: Campaign, rounds: range, *, bad: dict[int, Any] | None = None,
                 on_round: Any = None, base: int = 0) -> list:
    """Run one 10-seed round per `genRound` in `rounds`; `bad` maps a `genRound` to the item
    function its first seed answers with. Returns each run's `IngestResult`."""
    results = []
    for gen_round in rounds:
        if on_round is not None:
            on_round(gen_round)
        hand = [hand_seed(base + gen_round * 100 + n) for n in range(10)]
        hand_round(camp, hand, gen_round=gen_round)
        respond = leaking({hand[0].seed_id: bad[gen_round]} if gen_round in (bad or {}) else {})
        results.append(run_generation(camp, respond)[1])
    return results


def test_thirty_10_seed_rounds_with_one_hit_each_halt_in_the_pool_at_the_30th(
        tmp_path: Path) -> None:
    camp = bench(tmp_path)
    results = small_rounds(camp, range(1, 31), bad={g: binding_item for g in range(1, 31)})
    assert [r.stop for r in results[:29]] == [None] * 29       # 290 seeds: under 300
    state = camp.state()
    assert results[29].stop == "halted" and state.template_halts == set()
    halt = state.halt
    assert halt["pooled"] is True and halt["genRound"] == 30 and halt["batch"] == "bat_0030"
    assert halt["reasons"] == ["bindingMismatch"]
    assert "round" not in halt
    (pool,) = halt["pools"]
    assert pool["poolRounds"] == list(range(1, 31)) and pool["poolSeeds"] == 300
    assert pool["poolHits"] == {"screen": 0, "bindingMismatch": 30, "formMismatch": 0}
    assert pool["reasons"] == ["bindingMismatch"] and len(pool["templateHash"]) == 64
    assert halt["reason"] == "bindingMismatch"
    assert state.round_hits == {(g, "bindingMismatch"): 1 for g in range(1, 31)}


def test_a_resolve_that_leaves_the_hash_unchanged_does_not_refire_the_pool(
        tmp_path: Path) -> None:
    camp = bench(tmp_path)
    small_rounds(camp, range(1, 31), bad={g: binding_item for g in range(1, 31)})
    assert camp.state().halt is not None
    camp.resolve(decision="the template was fixed")
    (after,) = small_rounds(camp, range(31, 32), bad={31: binding_item})
    assert after.stop is None and camp.state().halt is None
    (pooled,) = camp.state().pool.values()
    assert pooled["seeds"] == 310 and pooled["hits"]["bindingMismatch"] == 31


def test_thirty_rounds_with_a_hash_change_after_round_15_do_not_halt(tmp_path: Path) -> None:
    camp = bench(tmp_path)
    re_install(camp, prompt_sha256=prompts())

    def change(gen_round: int) -> None:
        if gen_round == 16:
            re_install(camp, prompt_sha256=prompts(requestTemplate="r1"))

    results = small_rounds(camp, range(1, 31), bad={g: binding_item for g in range(1, 31)},
                           on_round=change)
    assert [r.stop for r in results] == [None] * 30 and camp.state().halt is None
    pools = camp.state().pool
    assert sorted(p["seeds"] for p in pools.values()) == [150, 150]
    assert sorted(p["rounds"][0] for p in pools.values()) == [1, 16]


def test_a_hit_of_another_reason_does_not_pool(tmp_path: Path) -> None:
    camp = bench(tmp_path)
    results = small_rounds(camp, range(1, 31), bad={3: binding_item, 17: form_item})
    assert [r.stop for r in results] == [None] * 30
    (pooled,) = camp.state().pool.values()
    assert pooled["hits"] == {"screen": 0, "bindingMismatch": 1, "formMismatch": 1}
    assert pooled["seeds"] == 300


def test_regeneration_hits_do_not_enter_the_pool(tmp_path: Path) -> None:
    camp = bench(tmp_path)
    for gen_round in (1, 2, 3):
        hand_round(camp, [hand_seed(gen_round * 100 + n) for n in range(10)], gen_round=gen_round)
    drain(camp, leaking({hand_seed(100).seed_id: binding_item}))   # hits in the first pass + twice
    (pooled,) = camp.state().pool.values()
    assert pooled == {"rounds": [1, 2, 3], "seeds": 30,
                      "hits": {"screen": 0, "bindingMismatch": 1, "formMismatch": 0}}


def test_a_generator_restart_makes_the_pooled_pair_new(tmp_path: Path) -> None:
    camp = bench(tmp_path)
    small_rounds(camp, range(1, 31), bad={g: binding_item for g in range(1, 31)})
    assert camp.state().pool_halts
    camp.resolve(decision="a new generator", repin_model=("generator", "claude-sonnet-5-5"))
    state = camp.state()
    assert state.pool_halts == set() and state.pool == {} and state.halt is None
    results = small_rounds(camp, range(1, 31), base=10000,
                           bad={g: binding_item for g in range(1, 31)})
    assert results[29].stop == "halted" and camp.state().halt["pooled"] is True


def test_a_25_seed_round_is_outside_the_pool(tmp_path: Path) -> None:
    camp = bench(tmp_path)
    hand_round(camp, [hand_seed(n) for n in range(1, 26)])
    run_generation(camp, obedient_item)
    assert camp.state().pool == {} and camp.state().pooled_rounds == set()


def test_a_small_round_joins_the_pool_when_its_last_run_ingests_not_before(
        tmp_path: Path) -> None:
    camp = bench(tmp_path)
    hand_round(camp, [hand_seed(n) for n in range(1, 13)], gen_round=1)
    hand_round(camp, [hand_seed(n) for n in range(101, 113)], gen_round=1)   # 24 under genRound 1
    run_generation(camp, obedient_item)
    assert camp.state().pool == {} and camp.state().pooled_rounds == set()
    run_generation(camp, obedient_item)
    (pooled,) = camp.state().pool.values()
    assert pooled["rounds"] == [1] and pooled["seeds"] == 24 and camp.state().pooled_rounds == {1}


def sized_rounds(camp: Campaign, sizes: list[int], *, start: int = 1, bad: Any = binding_item,
                 where: tuple[int, ...] = (0,)) -> list:
    """One round per entry of `sizes` (its seed count); each round's run answers `bad` at the
    position `where` names in its `entries` (cycled, so -1 is the run's last entry). Returns
    each run's `IngestResult`."""
    results = []
    for number, size in enumerate(sizes):
        gen_round = start + number
        hand = [hand_seed(gen_round * 100 + n) for n in range(size)]
        (batch_id,) = hand_round(camp, hand, gen_round=gen_round)
        victim = camp.state().batches[batch_id].entries[where[number % len(where)]]
        results.append(run_generation(camp, leaking({victim: bad}))[1])
    return results


def test_the_pool_adds_each_rounds_own_size_not_a_neighbours(tmp_path: Path) -> None:
    camp = bench(tmp_path)
    sizes = [10, 12, 20] * 7 + [10]                          # 21 rounds: 294 seeds, then 304
    results = sized_rounds(camp, sizes)
    assert [r.stop for r in results[:21]] == [None] * 21
    assert results[21].stop == "halted"
    halt = camp.state().halt
    assert halt["pooled"] is True and halt["genRound"] == 22
    (pool,) = halt["pools"]
    assert pool["poolRounds"] == list(range(1, 23)) and pool["poolSeeds"] == 304
    assert pool["poolHits"] == {"screen": 0, "bindingMismatch": 22, "formMismatch": 0}


@pytest.mark.parametrize(("hits", "halts"), [(15, False), (16, True)])
def test_the_pool_cutoff_is_5_percent_not_the_rounds_4(
        tmp_path: Path, hits: int, halts: bool) -> None:
    camp = bench(tmp_path)
    results = small_rounds(camp, range(1, 31), bad={g: binding_item for g in range(1, hits + 1)})
    assert (results[29].stop == "halted") is halts           # 15 of 300 is 5%, not more than 5%
    assert camp.state().halt is None or camp.state().halt["pooled"] is True


def test_a_small_round_split_by_an_install_feeds_each_runs_own_hash(tmp_path: Path) -> None:
    camp = bench(tmp_path)
    re_install(camp, prompt_sha256=prompts())
    sized_rounds(camp, [10] * 28)                            # H0: 280 seeds, 28 binding hits
    first = [hand_seed(9000 + n) for n in range(20)]
    second = [hand_seed(9100 + n) for n in range(4)]
    hand_round(camp, first, gen_round=29)
    hand_round(camp, second, gen_round=29)                   # 24 drawn under genRound 29
    assert run_generation(camp, leaking({first[0].seed_id: form_item}))[1].stop is None
    assert camp.state().pool == {} or all(p["seeds"] == 280 for p in camp.state().pool.values())
    re_install(camp, prompt_sha256=prompts(requestTemplate="r1"))
    _, result = run_generation(camp, obedient_item)
    h0 = generator_template_hash({"promptSha256": prompts()})
    h1 = generator_template_hash({"promptSha256": prompts(requestTemplate="r1")})
    pool = camp.state().pool
    assert pool[h0]["seeds"] == 300 and pool[h1]["seeds"] == 4
    assert pool[h0]["hits"] == {"screen": 0, "bindingMismatch": 28, "formMismatch": 1}
    assert pool[h1]["hits"] == {"screen": 0, "bindingMismatch": 0, "formMismatch": 0}
    assert pool[h0]["rounds"][-1] == 29 and pool[h1]["rounds"] == [29]
    assert result.stop == "halted"
    halt = camp.state().halt
    (pool,) = halt["pools"]
    assert pool["templateHash"] == h0 and halt["genRound"] == 29 and pool["poolSeeds"] == 300


def test_a_small_round_with_a_dropped_run_closes_and_pools_the_ingested_run(
        tmp_path: Path) -> None:
    camp = bench(tmp_path)
    sized_rounds(camp, [10] * 28)                            # 280 seeds, 28 hits
    first = [hand_seed(9000 + n) for n in range(14)]
    second = [hand_seed(9100 + n) for n in range(10)]
    hand_round(camp, first, gen_round=29)
    hand_round(camp, second, gen_round=29)                   # 24 drawn
    run_generation(camp, leaking({first[0].seed_id: binding_item}))
    assert 29 not in camp.state().pooled_rounds              # the second run is still open

    def broken(seed: Any) -> dict[str, Any]:
        item = obedient_item(seed)
        del item["condition"]
        return item

    for _ in range(3):
        _, last = run_generation(camp, broken)
    state = camp.state()
    assert state.batches["bat_0030"].dropped == "retryCapExceeded"
    assert last.stop is None and state.halt is None
    (pool,) = state.pool.values()
    assert pool["seeds"] == 294 and pool["rounds"][-1] == 29 and state.pooled_rounds >= {29}
    assert pool["hits"]["bindingMismatch"] == 29             # the dropped run's seeds are not in
    (after,) = sized_rounds(camp, [10], start=30)
    assert after.stop == "halted" and camp.state().halt["pools"][0]["poolSeeds"] == 304


@pytest.mark.parametrize(("sizes", "halts"), [([10] * 28 + [19], False), ([10] * 28 + [20], True)])
def test_a_pool_of_299_seeds_is_not_tested_and_one_of_300_is(
        tmp_path: Path, sizes: list[int], halts: bool) -> None:
    camp = bench(tmp_path)
    results = sized_rounds(camp, sizes)                      # 29 rounds, a hit in each
    assert [r.stop for r in results[:-1]] == [None] * 28
    assert (results[-1].stop == "halted") is halts
    (pool,) = camp.state().pool.values()
    assert pool["seeds"] == sum(sizes)


def test_a_hit_on_any_entry_of_a_pooled_run_counts_the_last_included(tmp_path: Path) -> None:
    camp = bench(tmp_path)
    results = sized_rounds(camp, [10] * 30, where=(-1, 0, 4, -2, 9))
    assert [r.stop for r in results[:29]] == [None] * 29
    assert results[29].stop == "halted"
    assert camp.state().halt["pools"][0]["poolHits"]["bindingMismatch"] == 30


def two_reason_item(seed: Any) -> dict[str, Any]:
    """A leak (screen) in an item whose condition is also the wrong one (bindingMismatch)."""
    item = leak_item(seed)
    item["condition"] = "some other condition"
    return item


def test_an_item_showing_two_reasons_counts_under_both_in_the_round_and_the_pool(
        tmp_path: Path) -> None:
    camp = bench(tmp_path)
    results = sized_rounds(camp, [10] * 30, bad=two_reason_item)
    assert results[29].stop == "halted"
    state = camp.state()
    assert state.round_hits[(1, "screen")] == 1 and state.round_hits[(1, "bindingMismatch")] == 1
    (pool,) = state.halt["pools"]
    assert pool["reasons"] == ["screen", "bindingMismatch"]
    assert pool["poolHits"] == {"screen": 30, "bindingMismatch": 30, "formMismatch": 0}
    assert state.halt["reasons"] == ["screen", "bindingMismatch"]


def split_round(camp: Campaign, number: int, sizes: tuple[int, int]) -> list[Any]:
    """Round `number` as two runs (`sizes`), seeds numbered apart from the sized rounds'."""
    runs = []
    for part, size in enumerate(sizes):
        hand = [hand_seed(90000 + number * 1000 + part * 100 + n) for n in range(size)]
        hand_round(camp, hand, gen_round=number)
        runs.append(hand)
    return runs


def test_one_close_crossing_two_pools_lists_both_and_marks_every_pair_fired(
        tmp_path: Path) -> None:
    camp = bench(tmp_path)
    h0_template, h1_template = prompts(), prompts(requestTemplate="r1")
    h0 = generator_template_hash({"promptSha256": h0_template})
    h1 = generator_template_hash({"promptSha256": h1_template})
    sizes = [15] * 14 + [10] * 8                             # 290 seeds, 22 hits per hash
    re_install(camp, prompt_sha256=h0_template)
    assert all(r.stop is None for r in sized_rounds(camp, sizes))
    re_install(camp, prompt_sha256=h1_template)
    assert all(r.stop is None for r in sized_rounds(camp, sizes, start=101))
    re_install(camp, prompt_sha256=h0_template)
    first, second = split_round(camp, 200, (10, 10))
    assert run_generation(camp, leaking({first[3].seed_id: binding_item}))[1].stop is None
    re_install(camp, prompt_sha256=h1_template)
    _, result = run_generation(camp, leaking({second[2].seed_id: form_item}))
    assert result.stop == "halted"
    state = camp.state()
    halt = state.halt
    assert "round" not in halt and halt["genRound"] == 200 and halt["pooled"] is True
    assert [p["templateHash"] for p in halt["pools"]] == sorted([h0, h1])
    by_hash = {p["templateHash"]: p for p in halt["pools"]}
    assert by_hash[h0]["poolSeeds"] == by_hash[h1]["poolSeeds"] == 300
    assert by_hash[h0]["poolHits"] == {"screen": 0, "bindingMismatch": 23, "formMismatch": 0}
    assert by_hash[h1]["poolHits"] == {"screen": 0, "bindingMismatch": 22, "formMismatch": 1}
    assert by_hash[h0]["reasons"] == by_hash[h1]["reasons"] == ["bindingMismatch"]
    assert by_hash[h0]["poolRounds"][-1] == by_hash[h1]["poolRounds"][-1] == 200
    assert halt["reason"] == "bindingMismatch"
    assert state.pool_halts == {(h0, "bindingMismatch"), (h1, "bindingMismatch")}
    text = camp.next().detail
    assert h0[:12] in text and h1[:12] in text
    assert h0[:12] in "\n".join(camp.status()) and h1[:12] in "\n".join(camp.status())
    camp.resolve(decision="both templates fixed")
    assert camp.state().halt is None and camp.state().pool_halts == state.pool_halts


def test_a_round_and_a_pool_crossing_together_keep_their_own_reasons(tmp_path: Path) -> None:
    camp = bench(tmp_path)
    sized_rounds(camp, [10] * 29)                            # 290 seeds, 29 binding hits
    hand = [hand_seed(n, form="empty") for n in range(1, 11)]
    hand_round(camp, hand, gen_round=30)
    bad = {hand[0].seed_id: form_item, hand[6].seed_id: form_item}
    _, result = run_generation(camp, leaking(bad))
    assert result.stop == "halted"
    state = camp.state()
    halt = state.halt
    assert halt["round"] == {"genRound": 30, "reasons": ["formMismatch"]}
    (pool,) = halt["pools"]
    assert pool["reasons"] == ["bindingMismatch"] and pool["poolSeeds"] == 300
    # formMismatch sorts after bindingMismatch, yet the round's reason leads
    assert halt["reason"] == "formMismatch"
    assert halt["reasons"] == ["bindingMismatch", "formMismatch"]
    assert halt["genRound"] == 30
    assert state.template_halts == {(30, "formMismatch")}
    assert state.pool_halts == {(pool["templateHash"], "bindingMismatch")}


def test_a_dropped_run_that_closes_a_round_raises_the_pooled_halt_through_ingest(
        tmp_path: Path) -> None:
    camp = bench(tmp_path)
    sized_rounds(camp, [10] * 28)                            # 280 seeds, 28 hits
    first = [hand_seed(9000 + n) for n in range(20)]
    second = [hand_seed(9100 + n) for n in range(4)]
    hand_round(camp, first, gen_round=29)
    hand_round(camp, second, gen_round=29)
    assert run_generation(camp, obedient_item)[1].stop is None

    def broken(seed: Any) -> dict[str, Any]:
        item = obedient_item(seed)
        del item["condition"]
        return item

    stops = [run_generation(camp, broken)[1].stop for _ in range(3)]
    assert stops == [None, None, "halted"]
    halt = camp.state().halt
    assert halt["batch"] == "bat_0030" and halt["pools"][0]["poolSeeds"] == 300


def test_skip_batch_of_a_halted_run_closes_the_round_and_pools(tmp_path: Path) -> None:
    camp = bench(tmp_path)
    sized_rounds(camp, [10] * 28)
    first = [hand_seed(9000 + n) for n in range(20)]
    second = [hand_seed(9100 + n) for n in range(4)]
    hand_round(camp, first, gen_round=29)
    hand_round(camp, second, gen_round=29)
    run_generation(camp, obedient_item)
    run_batch(camp, "refusal")                               # the second run: a safeguardStop
    assert camp.state().halt["reason"] == "safeguardStop"
    camp.resolve(decision="skip the flagged run", skip_batch=True)
    state = camp.state()
    assert state.batches["bat_0030"].dropped == "safeguardSkipped"
    assert 29 in state.pooled_rounds
    (pool,) = state.halt["pools"]
    assert pool["poolSeeds"] == 300 and pool["reasons"] == ["bindingMismatch"]
    camp.resolve(decision="the template is fixed")
    assert camp.state().halt is None and camp.state().pool_halts == state.pool_halts


def test_driver_resolve_prints_the_halt_it_clears_and_the_one_its_drop_raises(
        tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    from decisionmodels import driver
    camp = bench(tmp_path)
    sized_rounds(camp, [10] * 28)
    hand_round(camp, [hand_seed(9000 + n) for n in range(20)], gen_round=29)
    hand_round(camp, [hand_seed(9100 + n) for n in range(4)], gen_round=29)
    run_generation(camp, obedient_item)
    run_batch(camp, "refusal")                               # the second run: a safeguardStop
    capsys.readouterr()
    root = str(camp.root)
    assert driver.main(["resolve", "--root", root, "--decision", "skip", "--skip-batch"]) == 0
    cleared, raised = capsys.readouterr().out.splitlines()
    assert cleared.startswith("cleared: safeguardStop") and "bat_0030" in cleared
    assert raised.startswith("HALTED: ") and "pool " in raised and "300 seeds" in raised
    assert "resumes" not in raised and raised.endswith("clear with `driver resolve`")
    assert driver.main(["resolve", "--root", root, "--decision", "fixed"]) == 0
    cleared, resumes = capsys.readouterr().out.splitlines()
    assert cleared.startswith("cleared: ") and "pool " in cleared
    assert resumes == "resolved; the next `driver run` resumes"


@pytest.mark.parametrize("ingested", [False, True])
def test_extend_is_refused_once_a_first_pass_run_of_the_round_has_issued(
        tmp_path: Path, ingested: bool) -> None:
    """U5.9's planner killed between its two `begin_round` calls while the runner went on."""
    camp = bench(tmp_path)
    camp.begin_round(30, 15, design_stratum="turns")
    if ingested:
        run_generation(camp, obedient_item)
    else:
        camp.issue(serve(camp))
    before = camp.ledger_path.read_bytes()
    with pytest.raises(CampaignRefused, match="issued"):
        camp.begin_round(30, 15, design_stratum="understanding", extend=True)
    assert camp.ledger_path.read_bytes() == before
    assert camp.state().drawn_in(30) == 15


def test_a_stratum_a_lower_round_drew_is_accepted_for_round_3_and_refused_for_round_2(
        tmp_path: Path) -> None:
    camp = bench(tmp_path)
    camp.begin_round(2, 10, design_stratum="turns")
    camp.begin_round(2, 8, design_stratum="understanding", extend=True)
    camp.begin_round(3, 12, design_stratum="turns")
    camp.begin_round(3, 9, design_stratum="understanding", extend=True)
    with pytest.raises(CampaignRefused, match="already holds a draw"):
        camp.begin_round(2, 5, design_stratum="understanding", extend=True)
    state = camp.state()
    assert (state.drawn_in(2), state.drawn_in(3)) == (18, 21)


def test_the_template_hash_is_the_canonical_json_of_the_generator_entries() -> None:
    expectations = {"promptSha256": prompts(agentDefinition="aaa")}
    parts = {"agentDefinition": "aaa", "spawnTemplate": "s0", "requestTemplate": "r0"}
    wanted = hashlib.sha256(json.dumps(parts, sort_keys=True, separators=(",", ":"))
                            .encode()).hexdigest()
    assert generator_template_hash(expectations) == wanted
    base = generator_template_hash({"promptSha256": prompts()})
    for key in ("agentDefinition", "spawnTemplate", "requestTemplate"):
        assert generator_template_hash({"promptSha256": prompts(**{key: "z"})}) != base
    other_roles = prompts()
    for key in other_roles:
        other_roles[key]["labeler"] = "changed"
        other_roles[key]["driver"] = "changed"
    assert generator_template_hash({"promptSha256": other_roles}) == base
    assert generator_template_hash(None) == generator_template_hash({})


def test_a_resolve_after_a_template_halt_records_a_decision_and_draws_nothing(
        tmp_path: Path) -> None:
    camp = bench(tmp_path)
    hand = [hand_seed(n) for n in range(1, 41)]
    hand_round(camp, hand)
    run_generation(camp, leaking(bad_binding(1, 2, 3)))
    assert camp.state().halt is not None
    before = len(camp.events())
    camp.resolve(decision="fixed the generator template")
    (event,) = camp.events()[before:]
    assert event["event"] == "resolved" and event["decision"] == "fixed the generator template"
    assert camp.state().halt is None and "stakeholderRef" not in event
    assert camp.state().template_halts == {(1, "bindingMismatch")}
    # a further hit of the same reason in that round does not halt; another reason does
    _, two = run_generation(camp, leaking({**bad_binding(21), **{
        hand_seed(n, form="empty").seed_id: form_item for n in (22, 23, 24)}}))
    assert camp.state().halt["reasons"] == ["formMismatch"] and two.stop == "halted"


def test_ten_percent_no_rush_brackets_do_not_halt_a_guard_round(tmp_path: Path) -> None:
    camp = bench(tmp_path)
    hand = [hand_seed(n) for n in range(1, 41)]
    hand_round(camp, hand)

    def with_aside(seed: seeds.Seed) -> dict:
        item = obedient_item(seed)
        if item["path"] == "turns":
            item["turns"][0]["text"] += " (no rush)"
        else:
            item["understanding"]["request"] += " (other than login)"
        return item

    asides = {hand[n].seed_id: with_aside for n in (0, 9, 20, 29)}
    assert drain(camp, leaking(asides)) == 2
    state = camp.state()
    assert state.hits == {} and state.halt is None and len(state.items) == 40


def test_ten_percent_order_numbers_do_not_halt_a_message_type_stratum() -> None:
    base = seeds.draw_seeds("message-type-classifier", 40, 4, design_stratum=None).seeds
    ordered = [s for s in base if s.axes["intendedAnswer"] == "order"]
    assert len(ordered) >= 2
    hits = 0
    for index, seed in enumerate(base):
        prior = None if seed.axes["threadPosition"] == "opening" else {
            "speaker": "Pip", "text": "Anything else?"}
        text = f"Where is my order (order #{4400 + index})?" if index % 10 == 0 else "ok"
        check = screen.check_item(MESSAGE_TYPE, seed.axes, {"text": text, "priorTurn": prior})
        hits += bool(check.reasons)
    assert hits == 0 and ordered


# --- issue-time expectations ---------------------------------------------------------------------


def test_a_run_issued_before_a_reinstall_is_audited_against_the_template_at_issue(
        tmp_path: Path) -> None:
    camp = bench(tmp_path)
    re_install(camp, prompt_sha256=prompts())
    hand = [hand_seed(n) for n in range(1, 25)]
    hand_round(camp, hand)
    action = camp.next()
    session = camp.issue(action)
    stage(camp, session, action.opaque)
    write_answer(camp, action.opaque,
                 answer_with_items(camp, action.opaque, bad=bad_binding(1, 2)))
    changed = {role: {**spec, "spawnTemplate": spec["spawnTemplate"] + " Be quick."}
               for role, spec in GENERATOR_AS_FIXTURE.items()}
    camp.record_expectations(guard_command=GUARD_COMMAND, roles=changed,
                             prompt_sha256=prompts(spawnTemplate="s1"))
    result = camp.ingest(action.opaque)
    assert result.outcome["outcome"] == "ingested" and "reason" not in result.outcome
    state = camp.state()
    assert len(state.items) == 22
    old = state.attempts[0].expectations["roles"]["generator"]["spawnTemplate"]
    assert old == GENERATOR_AS_FIXTURE["generator"]["spawnTemplate"]
    assert state.expectations["roles"]["generator"]["spawnTemplate"].endswith("Be quick.")
    # the run's hash is the one in force at its issue, not the re-install's
    assert state.halt["templateHash"] == generator_template_hash({"promptSha256": prompts()})
    assert state.halt["templateHash"] != generator_template_hash(state.expectations)


def answer_with_items(camp: Campaign, opaque: str, bad: dict | None = None) -> bytes:
    text = (camp.store / "req" / f"{opaque}.jsonl").read_text("utf-8")
    state = camp.state()
    lines: list[dict[str, Any]] = [{"bodySha256": json.loads(text.splitlines()[0])["bodySha256"]}]
    lines += [{"key": key, "item": (binding_item if seed_id in (bad or {}) else obedient_item)(
        state.seeds[seed_id])} for key, seed_id in state.opaques[opaque].keys.items()]
    return "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines).encode("utf-8")


# --- begin_round: guards, orphans, a half-written round ------------------------------------------


def test_begin_round_refuses_a_started_gen_round_unless_it_extends_it(tmp_path: Path) -> None:
    camp = bench(tmp_path)
    camp.begin_round(1, 10, design_stratum="turns")
    before = camp.ledger_path.read_bytes()
    with pytest.raises(CampaignRefused, match="already started"):
        camp.begin_round(1, 10, design_stratum="turns")          # a repeat, no extend
    with pytest.raises(CampaignRefused, match="not started"):
        camp.begin_round(2, 10, design_stratum="turns", extend=True)
    with pytest.raises(CampaignRefused, match="name it"):
        camp.begin_round(1, 10, extend=True)                      # no design stratum
    with pytest.raises(CampaignRefused, match="already holds a draw"):
        camp.begin_round(1, 10, design_stratum="turns", extend=True)
    assert camp.ledger_path.read_bytes() == before
    camp.begin_round(1, 10, design_stratum="understanding", extend=True)
    state = camp.state()
    assert state.drawn_in(1) == 20 and len(state.rounds) == 2
    with pytest.raises(CampaignRefused, match="already holds a draw"):
        camp.begin_round(1, 10, design_stratum="understanding", extend=True)


def test_begin_round_records_its_design_stratum(tmp_path: Path) -> None:
    camp = bench(tmp_path)
    camp.begin_round(1, 12, design_stratum="turns")
    state = camp.state()
    (start,) = ledger_events(camp, "round-start")
    (draw,) = ledger_events(camp, "seed-draw")
    assert start["designStratum"] == draw["draw"]["designStratum"] == "turns"
    assert {state.seeds[s].axes["evidenceKind"] for s in start["seeds"]} == {"turns"}


def test_a_crash_after_the_draw_leaves_an_orphan_and_a_retry_succeeds(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    camp = bench(tmp_path)
    real = Campaign._append

    def crash(self: Campaign, event: Any) -> dict:
        if event["event"] == "round-start":
            raise OSError("killed")
        return real(self, event)

    monkeypatch.setattr(Campaign, "_append", crash)
    with pytest.raises(OSError):
        camp.begin_round(1, 12)
    monkeypatch.setattr(Campaign, "_append", real)
    orphans = set(camp.state().seeds)
    assert len(orphans) == 12 and camp.state().rounds == []
    assert camp.begin_round(1, 12)                              # not refused: no round exists
    state = camp.state()
    assert len(state.seeds) == 24 and state.drawn_in(1) == 12
    assert not orphans & {s for r in state.rounds for s in r["seeds"]}
    assert state.unbatched_seeds() == []


def test_next_queues_the_runs_a_crash_after_round_start_left_unqueued(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    camp = bench(tmp_path)
    real, calls = Campaign.queue_batch, []

    def crash(self: Campaign, kind: str, entries: Any, **extra: Any) -> str:
        calls.append(len(entries))
        if len(calls) == 2:
            raise OSError("killed")
        return real(self, kind, entries, **extra)

    monkeypatch.setattr(Campaign, "queue_batch", crash)
    with pytest.raises(OSError):
        camp.begin_round(1, 130)
    monkeypatch.setattr(Campaign, "queue_batch", real)
    assert calls == [22, 22] and len(camp.state().unbatched_seeds()) == 108
    assert isinstance(camp.next(), campaign.Serve)
    state = camp.state()
    sizes = [len(b.entries) for b in state.batches.values()]
    assert state.unbatched_seeds() == [] and sizes == [22, 22, 22, 22, 21, 21]
    held = [s for b in state.batches.values() for s in b.entries]
    assert sorted(held) == sorted(state.seeds)


# --- a generator re-pin restarts the generation --------------------------------------------------


def spent_campaign(tmp_path: Path) -> tuple[Campaign, list]:
    """Items, hits, a third-hit drop, spent halts and, last, a template halt still to resolve."""
    camp = bench(tmp_path)
    hand = [hand_seed(n) for n in range(1, 41)]
    hand_round(camp, hand)
    doomed = hand[0]
    run_generation(camp, leaking({doomed.seed_id: leak_item, **bad_binding(2, 3, 4)}))
    assert camp.state().halt is not None
    camp.resolve(decision="the template was fixed")
    run_generation(camp, leaking({doomed.seed_id: leak_item}))
    drain(camp, leaking({doomed.seed_id: leak_item}))           # the regenerations; doomed drops
    second = [hand_seed(n) for n in range(201, 221)]
    hand_round(camp, second, gen_round=2)
    run_generation(camp, leaking({second[0].seed_id: form_item, second[1].seed_id: form_item}))
    state = camp.state()
    assert state.halt["genRound"] == 2 and state.halt["reasons"] == ["formMismatch"]
    assert state.items and state.hits and state.dropped_seeds == {doomed.seed_id: state.hits[
        doomed.seed_id][2]}
    assert state.template_halts and state.pool and state.pooled_rounds == {2}
    return camp, hand


def test_a_generator_repin_resets_the_folded_generation_view_and_keeps_the_ledger(
        tmp_path: Path) -> None:
    camp, hand = spent_campaign(tmp_path)
    held_seeds = set(camp.state().seeds)
    held_rng = set(camp.state().held_rng_seeds)
    before = camp.ledger_path.read_bytes()
    camp.resolve(decision="a new generator", repin_model=("generator", "claude-sonnet-5-5"))
    state = camp.state()
    assert camp.ledger_path.read_bytes().startswith(before)
    assert (state.items, state.hits, state.dropped_seeds, state.round_hits) == ({}, {}, {}, {})
    assert (state.template_halts, state.pool, state.pool_halts) == (set(), {}, set())
    assert state.pooled_rounds == set()
    assert (state.rounds, state.seed_round, state.staged) == ([], {}, {})
    assert {b.state for b in state.batches.values()} == {"retired"}
    assert state.halt is None and state.awaiting_regeneration() == []
    assert state.dropped()["retryCapExceeded"] == {"seeds": 0, "items": 0}
    assert set(state.seeds) == held_seeds and state.held_rng_seeds == held_rng
    assert camp.next() == Stop("done", "nothing is pending")
    assert not any("retired" in line or "batches: 1" in line for line in camp.status()[1:2])


def test_after_a_restart_round_one_is_drawn_afresh_and_its_own_halt_can_fire(
        tmp_path: Path) -> None:
    camp, hand = spent_campaign(tmp_path)
    old_ids, old_rng = set(camp.state().seeds), set(camp.state().held_rng_seeds)
    camp.resolve(decision="a new generator", repin_model=("generator", "claude-sonnet-5-5"))
    with pytest.raises(CampaignRefused, match="already held"):
        camp.record_draw(seeds.SeedDraw(
            role="guard-judge", rng_seed=next(iter(old_rng)), count=1, design_stratum=None,
            seeds=(hand[0],)))
    camp.begin_round(1, 20)                                     # allowed: no round since restart
    state = camp.state()
    fresh = set(state.seeds) - old_ids
    assert len(fresh) == 20 and state.held_rng_seeds - old_rng
    run_generation(camp, leaking({s: binding_item for s in sorted(fresh)[:2]}))
    halt = camp.state().halt                                    # 2 of 20: fires anew, the round's
    assert halt is not None and halt["genRound"] == 1 and halt["reasons"] == ["bindingMismatch"]


def test_a_labeler_repin_keeps_the_items_and_the_batches(tmp_path: Path) -> None:
    camp, _ = spent_campaign(tmp_path)
    items = dict(camp.state().items)
    camp.resolve(decision="a new labeler", repin_model=("labeler", "claude-haiku-5-6"))
    state = camp.state()
    assert state.items == items and state.template_halts
    assert "retired" not in {b.state for b in state.batches.values()}


def test_a_label_run_open_across_a_labeler_repin_is_audited_against_the_old_pin(
        tmp_path: Path) -> None:
    camp = smoke(tmp_path, items=3)
    open_run = camp.prepare("bat_0001")
    camp.issue(open_run)
    stage(camp, camp.state().attempts[0].session_id, open_run.opaque)
    write_answer(camp, open_run.opaque, answer_bytes(camp, open_run.opaque))
    other = camp.queue_batch("label", ["fx_0001"])
    halting = camp.prepare(other)
    session = camp.issue(halting)
    stage(camp, session, halting.opaque, "refusal")
    assert camp.ingest(halting.opaque).stop == "halted"
    camp.resolve(decision="a new labeler", repin_model=("labeler", "claude-haiku-5-6"))
    assert camp.state().pins["labeler"] == "claude-haiku-5-6"
    result = camp.ingest(open_run.opaque)                       # the fixture's model is sonnet
    assert result.outcome["outcome"] == "ingested" and result.stop is None
    assert camp.state().halt is None
    assert camp.state().attempts[0].pins["labeler"] == "claude-sonnet-5-5"


def test_an_attempt_keeps_the_instructions_in_force_at_issue(tmp_path: Path) -> None:
    camp = smoke(tmp_path, items=2)
    action = camp.prepare("bat_0001")
    camp.issue(action)
    camp._append({"event": "resolved", "action": "repin-instructions", "role": "labeler",
                  "instructions": [["/x/CLAUDE.md", "abc"]], "decision": "x",
                  "batch": "bat_0001", "reason": None})
    state = camp.state()
    batch, attempt = state.batches["bat_0001"], state.attempts[0]
    assert state.instructions["labeler"] == [["/x/CLAUDE.md", "abc"]]
    assert attempt.instructions["labeler"] == []
    assert camp._expected(state, batch, action.opaque, attempt).instructions == frozenset()
    assert camp._expected(state, batch, action.opaque).instructions == frozenset(
        {("/x/CLAUDE.md", "abc")})


def test_a_run_open_at_a_generator_repin_is_recorded_and_changes_nothing(tmp_path: Path) -> None:
    camp = bench(tmp_path)
    hand = [hand_seed(n) for n in range(1, 41)]
    first, second = hand_round(camp, hand)
    open_run = camp.prepare(second)
    session = camp.issue(open_run)
    stage(camp, session, open_run.opaque, "refusal")            # a run that would halt on audit
    halting = camp.prepare(first)
    session = camp.issue(halting)
    stage(camp, session, halting.opaque)
    bad = bad_binding(1, 2)
    text = (camp.store / "req" / f"{halting.opaque}.jsonl").read_text("utf-8")
    keys = camp.state().opaques[halting.opaque].keys
    lines = [{"bodySha256": json.loads(text.splitlines()[0])["bodySha256"]}] + [
        {"key": k, "item": (binding_item if seed_id in bad else obedient_item)(
            camp.state().seeds[seed_id])} for k, seed_id in keys.items()]
    write_answer(camp, halting.opaque,
                 "".join(json.dumps(line) + "\n" for line in lines).encode("utf-8"))
    assert camp.ingest(halting.opaque).stop == "halted"
    camp.resolve(decision="a new generator", repin_model=("generator", "claude-opus-5-6"))
    assert camp.state().batches[second].open_attempt is not None
    assert camp.next() == Stop("done", "nothing is pending")    # it reconciled the open run
    state = camp.state()
    assert state.batches[second].state == "retired" and state.items == {} and state.halt is None
    outcome = state.batches[second].attempts[0].outcome
    assert (outcome["outcome"], outcome["reason"]) == ("halted", "safeguardStop")


# --- survivors from Pass 18 ----------------------------------------------------------------------


def all_three(seed: seeds.Seed) -> dict:
    item = leak_item(seed)
    item["condition"] = "another"
    item["understanding"]["missing"] = ["x"]
    return item


def test_a_drop_counts_under_the_third_hits_first_reason_not_its_last(tmp_path: Path) -> None:
    camp = bench(tmp_path)
    hand = padded([hand_seed(1, form="empty")])
    hand_round(camp, hand)
    target = hand[0].seed_id
    passes = iter([binding_item, binding_item, all_three])
    drain(camp, lambda s: next(passes)(s) if s.seed_id == target else obedient_item(s))
    state = camp.state()
    assert state.hits[target][2]["reasons"] == ["screen", "bindingMismatch", "formMismatch"]
    assert state.generation_drops() == {"screenHits": 1, "formMismatch": {}, "bindingMismatch": 0}


def test_a_void_outcome_applies_none_of_the_events_staged_for_it(tmp_path: Path) -> None:
    camp = bench(tmp_path)
    hand = [hand_seed(n) for n in range(1, 4)]
    hand_round(camp, hand)
    action = camp.next()
    session = camp.issue(action)
    stage(camp, session, action.opaque)
    camp._append({"event": "generated", "opaque": action.opaque, "batch": "bat_0001",
                  "items": [{"seedId": hand[0].seed_id, "genRunId": "gen_x", "genRound": 1,
                             "item": {}}]})
    camp._append({"event": "screen-hit", "opaque": action.opaque, "batch": "bat_0001",
                  "seedId": hand[1].seed_id, "genRunId": "gen_x", "reasons": ["screen"],
                  "detail": "x", "form": None, "firstPass": True})
    write_answer(camp, action.opaque, b'{"bodySha256": "0"}\n')
    assert camp.ingest(action.opaque).outcome["outcome"] == "void"
    state = camp.state()
    assert state.items == {} and state.hits == {} and state.round_hits == {} and state.staged == {}


def test_a_stale_hit_staged_for_an_opaque_does_not_survive_the_real_generated_event(
        tmp_path: Path) -> None:
    camp = bench(tmp_path)
    hand = [hand_seed(n) for n in range(1, 4)]
    hand_round(camp, hand)
    action = camp.next()
    session = camp.issue(action)
    stage(camp, session, action.opaque)
    camp._append({"event": "screen-hit", "opaque": action.opaque, "batch": "bat_0001",
                  "seedId": hand[1].seed_id, "genRunId": "gen_x", "reasons": ["screen"],
                  "detail": "stale", "form": None, "firstPass": True})
    write_answer(camp, action.opaque, answer_with_items(camp, action.opaque))
    camp.ingest(action.opaque)
    state = camp.state()
    assert hand[1].seed_id in state.items and state.hits == {}


def test_a_regeneration_run_dropped_by_the_retry_cap_is_not_requeued(tmp_path: Path) -> None:
    camp = bench(tmp_path)
    hand = padded([hand_seed(1)])
    hand_round(camp, hand)
    run_generation(camp, leaking({hand[0].seed_id: leak_item}))
    for _ in range(campaign.MAX_COUNTED_ATTEMPTS):
        action = camp.next()
        assert isinstance(action, campaign.Serve) and action.opaque
        session = camp.issue(action)
        stage(camp, session, action.opaque)
        write_answer(camp, action.opaque, b'{"bodySha256": "0"}\n')
        camp.ingest(action.opaque)
    state = camp.state()
    regen = [b for b in state.batches.values() if b.extra.get("generation") == "regen"]
    assert [b.dropped for b in regen] == ["retryCapExceeded"]
    assert state.awaiting_regeneration() == []
    assert votes_next(camp).kind == "label"          # the dropped seed is settled: the rest vote


def test_next_names_the_halt_when_it_reconciles_a_run_that_halts(tmp_path: Path) -> None:
    camp = bench(tmp_path)
    hand = [hand_seed(n) for n in range(1, 21)]
    hand_round(camp, hand)
    action = camp.next()
    session = camp.issue(action)
    stage(camp, session, action.opaque)
    text = (camp.store / "req" / f"{action.opaque}.jsonl").read_text("utf-8")
    keys = camp.state().opaques[action.opaque].keys
    lines = [{"bodySha256": json.loads(text.splitlines()[0])["bodySha256"]}] + [
        {"key": k, "item": binding_item(camp.state().seeds[s])} for k, s in keys.items()]
    write_answer(camp, action.opaque,
                 "".join(json.dumps(line) + "\n" for line in lines).encode("utf-8"))
    stop = camp.next()                                          # the crash-recovery ingest
    assert stop == Stop(
        "halted", "bindingMismatch (template) on bat_0001 [round 1: bindingMismatch]")
