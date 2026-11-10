"""`diagnostics.json` (typed-decision-models U5.9; plan4 `docs/plans/typed-decision-models4.md`
§3.6 "Diagnostics", plan5 `typed-decision-models5.md` §3.6 "Round-1 reprint", plan6
`typed-decision-models6.md` §3.8, and the carried ML items P3-n2 and P4-n2).

The campaign is a small hand-built ledger (`test_topup.py`'s `synth_run`): message type's seeds are
drawn once, then given the intents and difficulties below, so every expected figure is worked out
by hand in the comments. The votes (item by vote 1, 2, 3) are:

    i0 intent order     o o o      i3 intent question  o q o
    i1 intent order     o o q      i4 intent complaint c e x
    (o order, q question, c complaint, e escalation, x other)
    i2 intent question  q q q      i5 intent other     x x c
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest
from test_generation_rounds import spent_campaign
from test_topup import full_campaign, synth_run

from decisionmodels import diagnostics, seeds
from decisionmodels.campaign import Campaign
from decisionmodels.diagnostics import FILE_NAME

MODEL_BENCH = Path(__file__).resolve().parents[2] / "model-bench" / "packs"
FULL = {"o": "order", "q": "question", "c": "complaint", "e": "escalation", "x": "other"}
VOTES = {0: "ooo", 1: "ooq", 2: "qqq", 3: "oqo", 4: "cex", 5: "xxc"}
INTENTS = ["order", "order", "question", "question", "complaint", "other"]
DIFFICULTY = ["clear", "near-miss", "clear", "clear", "near-miss", "clear"]
_BASE = seeds.draw_seeds("message-type-classifier", 12, 5, design_stratum=None).seeds


def hand_seed(number: int, intent: str, difficulty: str) -> seeds.Seed:
    axes = {**_BASE[number % len(_BASE)].axes, "intendedAnswer": intent,
            "difficulty": difficulty}
    return dataclasses.replace(_BASE[number % len(_BASE)], seed_id=f"sd_{number:012d}", axes=axes)


def start_round(camp: Campaign, hand: list[seeds.Seed], gen_round: int) -> list[str]:
    rng_seed = camp.fresh_rng_seed()
    camp.record_draw(seeds.SeedDraw(role="message-type-classifier", rng_seed=rng_seed,
                                    count=len(hand), design_stratum=None, seeds=tuple(hand)))
    ids = [seed.seed_id for seed in hand]
    camp._append({"event": "round-start", "genRound": gen_round, "rngSeed": rng_seed,
                  "designStratum": None, "seeds": ids})
    return ids


def generate(camp: Campaign, ids: list[str], accept: list[str],
             hits: dict[str, str] | None = None, text: dict[str, str] | None = None) -> None:
    """One generation run over `ids`: the `accept`ed seeds become items, each seed of `hits`
    (seed id to screen detail) a first-pass screen hit."""
    batch_id = camp.queue_batch("generate", ids, generation="first")
    state = camp.state()

    def events(opaque: str) -> list[dict[str, Any]]:
        batch = state.batches[batch_id]
        out: list[dict[str, Any]] = [{
            "event": "generated", "opaque": opaque, "batch": batch_id,
            "items": [{"seedId": s, "genRunId": batch.gen_run_id,
                       "genRound": state.seed_round[s],
                       "item": {"text": (text or {}).get(s, f"message {s}"), "priorTurn": None}}
                      for s in accept]}]
        for seed_id, detail in (hits or {}).items():
            out.append({"event": "screen-hit", "opaque": opaque, "batch": batch_id,
                        "seedId": seed_id, "genRunId": batch.gen_run_id, "reasons": ["screen"],
                        "detail": detail, "form": None, "firstPass": True})
        return out

    synth_run(camp, batch_id, events)


def label(camp: Campaign, entries: list[str], votes: dict[str, str], gen_round: int = 1) -> None:
    """Votes 1 to 3 over `entries` in that batch order: item `s` answers `votes[s][v - 1]`."""
    for vote in (1, 2, 3):
        batch_id = camp.queue_batch("label", entries, genRound=gen_round, vote=vote)
        synth_run(camp, batch_id, lambda opaque, vote=vote, batch_id=batch_id: [{
            "event": "votes", "opaque": opaque, "batch": batch_id, "genRound": gen_round,
            "vote": vote, "answers": {s: FULL[votes[s][vote - 1]] for s in entries}}])


@pytest.fixture
def report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    camp = full_campaign(tmp_path, monkeypatch, "message-type-classifier")
    first = [hand_seed(n, INTENTS[n], DIFFICULTY[n]) for n in range(6)]
    first += [hand_seed(6, "order", "clear"), hand_seed(7, "question", "near-miss")]
    ids = start_round(camp, first, 1)
    generate(camp, ids, ids[:6], hits={ids[6]: "token:near-miss", ids[7]: "meta:order"},
             text={ids[0]: "Where is my order? It was due yesterday", ids[1]: "Where is my order?"})
    second = [hand_seed(9000, "order", "clear"), hand_seed(9001, "question", "clear")]
    ids2 = start_round(camp, second, 2)
    generate(camp, ids2, ids2)
    label(camp, ids[:6], {s: VOTES[n] for n, s in enumerate(ids[:6])})
    label(camp, ids2, {ids2[0]: "ooo", ids2[1]: "ooo"}, gen_round=2)
    return diagnostics.build(camp.state())


def test_pair_agreement_and_the_three_zero_share_are_per_class_beside_the_ties(
        report: dict[str, Any]) -> None:
    by_class = report["agreement"]["byClass"]
    # order: i0 (3 pairs), i1 (1), i3 (1) of 9 -> 5/9, and one of three is 3-0; the two top-up
    # items (3 pairs each) join: 11/15 over five items, two of five 3-0 plus i0 = 3/5.
    assert by_class["order"] == {"items": 5, "pairAgreement": 11 / 15, "threeZeroShare": 3 / 5}
    assert by_class["question"] == {"items": 1, "pairAgreement": 1.0, "threeZeroShare": 1.0}
    assert by_class["other"] == {"items": 1, "pairAgreement": 1 / 3, "threeZeroShare": 0.0}
    assert by_class["complaint"] == {"items": 0, "pairAgreement": None, "threeZeroShare": None}
    assert report["agreement"]["ties"] == 1 and report["labelled"] == 8


def test_intent_label_agreement_is_per_intended_class_and_leaves_ties_out(
        report: dict[str, Any]) -> None:
    got = report["agreement"]["intentLabelAgreement"]
    # intended order: i0, i1 agree, j0 agrees -> 3/3; question: i2 agrees, i3 (majority order) and
    # j1 (order) do not -> 1/3; other: i5 -> 1/1; complaint: i4 is a tie, so no row.
    assert got["order"] == {"items": 3, "rate": 1.0}
    assert got["question"] == {"items": 3, "rate": 1 / 3}
    assert got["other"] == {"items": 1, "rate": 1.0}
    assert "complaint" not in got
    assert got["overall"] == {"items": 7, "rate": 5 / 7}


def test_position_effect_is_the_class_rate_by_decile_of_the_position_in_the_batch(
        report: dict[str, Any]) -> None:
    # six entries -> deciles 0, 1, 3, 5, 6, 8 (10 * index // 6); three votes each. `order` answers:
    # i0 3 of 3, i1 2 of 3, i2 0, i3 2 of 3, i4 0, i5 0. The two-item round-2 batches add votes
    # at deciles 0 and 5 (10 * 1 // 2): all `order`, three votes each.
    order = report["positionEffect"]["order"]
    assert [n for n, _ in order] == [6, 3, 0, 3, 0, 6, 3, 0, 3, 0]
    assert [r for _, r in order] == [1.0, 2 / 3, None, 0.0, None, 5 / 6, 0.0, None, 0.0, None]


def test_balance_effect_is_the_ols_slope_on_the_mates_share_under_the_next_vote(
        report: dict[str, Any]) -> None:
    # The 18 (item, vote) points of the six-item batches (hand-worked, mates = the other five) give
    # a slope of -47/25 for `order`; the two-item round adds points whose mate is one item.
    only_first = report["balanceEffect"]["order"]
    assert only_first["n"] == 18 + 6 and only_first["slope"] is not None
    assert report["balanceEffect"]["complaint"]["n"] == 24


def test_balance_effect_over_the_first_round_alone_is_minus_47_25(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    camp = full_campaign(tmp_path, monkeypatch, "message-type-classifier")
    first = [hand_seed(n, INTENTS[n], DIFFICULTY[n]) for n in range(6)]
    ids = start_round(camp, first, 1)
    generate(camp, ids, ids)
    label(camp, ids, {s: VOTES[n] for n, s in enumerate(ids)})
    got = diagnostics.build(camp.state())["balanceEffect"]
    assert got["order"]["n"] == 18 and got["order"]["slope"] == pytest.approx(-47 / 25)


def test_a_slope_needs_two_points_and_a_varying_share() -> None:
    assert diagnostics._slope([(0.5, 1)]) is None
    assert diagnostics._slope([(0.5, 1), (0.5, 0)]) is None
    assert diagnostics._slope([(0.0, 0), (1.0, 1)]) == 1.0


def test_round_one_and_top_up_are_compared_per_cell(report: dict[str, Any]) -> None:
    cell = report["cells"]["none|order"]
    # round 1: i0, i1, i3 -> pairs 5/9, intent agreement 2 of 3 (i3 intended question);
    # top-up: j0, j1 -> pairs 1.0, intent agreement 1 of 2 (j1 intended question).
    assert cell["roundOne"] == {"items": 3, "pairAgreement": 5 / 9, "threeZeroShare": 1 / 3,
                                "intentLabelAgreement": 2 / 3}
    assert cell["topUp"] == {"items": 2, "pairAgreement": 1.0, "threeZeroShare": 1.0,
                             "intentLabelAgreement": 1 / 2}
    assert report["cells"]["none|question"]["topUp"]["items"] == 0
    assert report["cells"]["none|question"]["topUp"]["pairAgreement"] is None


def test_screen_hits_are_counted_by_intent_difficulty_and_pattern(report: dict[str, Any]) -> None:
    hits = report["screenHits"]
    # ten seeds were drawn; two were hit: a clear `order` one (token) and a near-miss `question`
    # one (meta position).
    assert hits["byIntent"]["order"] == {"seeds": 4, "hits": 1, "hitSeeds": 1}
    assert hits["byIntent"]["question"] == {"seeds": 4, "hits": 1, "hitSeeds": 1}
    assert hits["byIntent"]["other"] == {"seeds": 1, "hits": 0, "hitSeeds": 0}
    assert hits["byDifficulty"]["clear"] == {"seeds": 7, "hits": 1, "hitSeeds": 1}
    assert hits["byDifficulty"]["near-miss"] == {"seeds": 3, "hits": 1, "hitSeeds": 1}
    assert hits["byPattern"] == {"screen:token": 1, "screen:meta": 1}


def test_prompt_examples_are_counted_verbatim_and_nothing_is_dropped(
        report: dict[str, Any]) -> None:
    got = report["promptExamples"]
    assert got["items"] == 8 and got["matching"] == 2
    assert got["perExample"]["Where is my order?"] == 2
    assert got["perExample"]["Where is my order? It was due yesterday"] == 1
    assert got["perExample"]["Do you deliver on Sundays? If so I'd like two"] == 0
    assert report["labelled"] == 8            # the two matching items are still labelled


def test_join_in_examples_are_read_from_the_new_message() -> None:
    items = [{"newMessage": {"text": "Ask Nova later, please"}, "priorTurns": []},
             {"newMessage": {"text": "ok"}, "priorTurns": [{"text": "Nova, can you check this?"}]}]
    got = diagnostics.prompt_example_counts("join-in-judge", items)
    assert got["matching"] == 1 and got["perExample"]["Ask Nova later"] == 1
    assert got["perExample"]["Nova, can you check this?"] == 0       # a prior turn is not it
    assert diagnostics.prompt_example_counts("guard-judge", items) is None


@pytest.mark.parametrize(("role", "prompt"), [
    ("message-type-classifier",
     "message-type-classifier-synthetic/prompts/message-type-handoff.md"),
    ("join-in-judge", "join-in-judge-synthetic/prompts/join-in-handoff.md"),
])
def test_every_listed_example_is_verbatim_in_its_hand_off_prompt(role: str, prompt: str) -> None:
    path = MODEL_BENCH / prompt
    if not path.is_file():
        pytest.skip("model-bench is absent")
    text = path.read_text("utf-8")
    assert all(example in text for example in diagnostics.HANDOFF_EXAMPLES[role])


# --- guard: the round-1 CueCheck reprint and the turn-cut split -----------------------------------


_GUARD = seeds.draw_seeds("guard-judge", 40, 3, design_stratum=None).seeds
_UNDERSTANDING = next(s for s in _GUARD if s.axes["evidenceKind"] == "understanding")
_TURNS = next(s for s in _GUARD if s.axes["evidenceKind"] == "turns")


def guard_seed(number: int, base: seeds.Seed, **axes: Any) -> seeds.Seed:
    return dataclasses.replace(base, seed_id=f"sd_{number:012d}", axes={**base.axes, **axes})


def guard_campaign(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                   hand: list[tuple[seeds.Seed, dict[str, Any]]],
                   votes: dict[int, tuple[bool, bool, bool]], gen_round: int = 1) -> Campaign:
    camp = full_campaign(tmp_path, monkeypatch, "guard-judge")
    rng_seed = camp.fresh_rng_seed()
    camp.record_draw(seeds.SeedDraw(role="guard-judge", rng_seed=rng_seed, count=len(hand),
                                    design_stratum=None, seeds=tuple(s for s, _ in hand)))
    ids = [s.seed_id for s, _ in hand]
    camp._append({"event": "round-start", "genRound": gen_round, "rngSeed": rng_seed,
                  "designStratum": None, "seeds": ids})
    batch_id = camp.queue_batch("generate", ids, generation="first")
    synth_run(camp, batch_id, lambda opaque: [{
        "event": "generated", "opaque": opaque, "batch": batch_id,
        "items": [{"seedId": s.seed_id, "genRunId": camp.state().batches[batch_id].gen_run_id,
                   "genRound": gen_round, "item": item} for s, item in hand]}])
    for vote in (1, 2, 3):
        label_id = camp.queue_batch("label", ids, genRound=gen_round, vote=vote)
        synth_run(camp, label_id, lambda opaque, vote=vote, label_id=label_id: [{
            "event": "votes", "opaque": opaque, "batch": label_id, "genRound": gen_round,
            "vote": vote, "answers": {s.seed_id: votes[n][vote - 1]
                                      for n, (s, _) in enumerate(hand)}}])
    return camp


def test_the_round_one_cue_check_reprint_covers_all_understanding_items_and_omits_failed(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    common = {"evidenceKind": "understanding", "difficulty": "clear", "intendedAnswer": True}
    hand = [(guard_seed(1, _UNDERSTANDING, **common, missingForm="immaterial"), {}),
            (guard_seed(2, _UNDERSTANDING, **common, missingForm="empty"), {}),
            (guard_seed(3, _UNDERSTANDING, **common, missingForm="empty"), {}),
            (guard_seed(4, _TURNS, evidenceKind="turns", difficulty="clear",
                        intendedAnswer=True, missingForm=None), {"turns": []})]
    votes = {0: (False, False, False), 1: (True, True, True), 2: (True, True, True),
             3: (False, False, False)}
    camp = guard_campaign(tmp_path, monkeypatch, [(s, {"x": n}) for n, (s, _) in enumerate(hand)],
                          votes)
    got = diagnostics.build(camp.state())["missingCueCheck"]
    # one stratum (clear|true): the counter-cue item departs (x = 1 of m1 = 1), the two concordant
    # ones do not (y = 0 of m0 = 2), so P(x = 1) = C(1,1) C(2,0) / C(3,1) = 1/3. The turns item is
    # not an `understanding` item and is left out; `failed` is omitted.
    assert got == {"strata": [{"stratum": "clear|true", "x": 1, "m1": 1, "y": 0, "m0": 2,
                               "counterCueRate": 1.0, "concordantRate": 0.0}],
                   "sumX": 1, "p": pytest.approx(1 / 3)}
    assert "failed" not in got


def test_the_cue_check_reprint_leaves_a_top_up_round_out(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    common = {"evidenceKind": "understanding", "difficulty": "clear", "intendedAnswer": True}
    hand = [(guard_seed(1, _UNDERSTANDING, **common, missingForm="immaterial"), {"x": 1})]
    camp = guard_campaign(tmp_path, monkeypatch, hand, {0: (False, False, False)}, gen_round=2)
    assert diagnostics.build(camp.state())["missingCueCheck"]["strata"] == []


def test_intent_label_agreement_is_split_by_whether_turns_were_cut(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def turns(n: int) -> list[dict[str, str]]:
        return [{"role": "user", "text": f"t{k}"} for k in range(n)]

    base = {"evidenceKind": "turns", "difficulty": "clear", "missingForm": None}
    hand = [(guard_seed(1, _TURNS, **base, intendedAnswer=True), {"turns": turns(8)}),
            (guard_seed(2, _TURNS, **base, intendedAnswer=True), {"turns": turns(7)}),
            (guard_seed(3, _TURNS, **base, intendedAnswer=True), {"turns": turns(6)}),
            (guard_seed(4, _TURNS, **base, intendedAnswer=False),
             {"turns": [*turns(6), {"role": "user", "text": ""}, {"role": "user"}]})]
    votes = {0: (True, True, True), 1: (False, False, False), 2: (True, True, True),
             3: (False, False, True)}
    camp = guard_campaign(tmp_path, monkeypatch, hand, votes)
    got = diagnostics.build(camp.state())["turnCut"]
    # items 1 and 2 (8 and 7 turns) are cut: one agrees with intent, one does not. The six-turn
    # item and the one with six valid turns plus two without text are not cut (6 is the limit).
    assert got["cut"]["items"] == 2
    assert got["cut"]["intentLabelAgreement"]["overall"] == {"items": 2, "rate": 0.5}
    assert got["notCut"]["items"] == 2
    assert got["notCut"]["intentLabelAgreement"]["overall"] == {"items": 2, "rate": 1.0}


def test_message_type_has_no_turn_cut_split_and_no_cue_check(report: dict[str, Any]) -> None:
    assert report["turnCut"] is None and "missingCueCheck" not in report


# --- templates, retired runs, rounds and the file -------------------------------------------------


def test_template_halts_and_the_hash_of_each_run_are_reported_and_a_restart_retires_them(
        tmp_path: Path) -> None:
    camp, _ = spent_campaign(tmp_path)
    before = diagnostics.build(camp.state())
    assert before["templates"]["halts"] and before["templates"]["hashes"]
    halt = before["templates"]["halts"][0]
    assert halt["voidClass"] == "template" and halt["reasons"]
    assert sum(h["items"] for h in before["templates"]["hashes"]) == len(camp.state().items)
    assert before["retired"] == {} and before["rounds"][0]["genRound"] == 1
    camp.resolve(decision="a new generator", repin_model=("generator", "claude-sonnet-5-5"))
    after = diagnostics.build(camp.state())
    assert after["templates"] == {"hashes": [], "halts": []} and after["rounds"] == []
    retired = after["retired"]["generate"]
    assert retired["batches"] > 0 and retired["attempts"] > 0 and sum(
        retired["tokens"].values()) > 0


def test_each_rounds_draw_items_and_tokens_are_reported(report: dict[str, Any]) -> None:
    one, two = report["rounds"]
    assert (one["genRound"], one["drawn"], one["items"]) == (1, 8, 6)
    assert (two["genRound"], two["drawn"], two["items"]) == (2, 2, 2)
    assert sum(one["labelTokens"].values()) == 300 and sum(one["generationTokens"].values()) == 100


def test_write_puts_diagnostics_json_beside_generation_json_and_the_cli_does_too(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    camp = full_campaign(tmp_path, monkeypatch, "message-type-classifier")
    ids = start_round(camp, [hand_seed(n, INTENTS[n], DIFFICULTY[n]) for n in range(6)], 1)
    generate(camp, ids, ids)
    label(camp, ids, {s: VOTES[n] for n, s in enumerate(ids)})
    out = tmp_path / "dataset"
    path = diagnostics.write(camp.state(), out)
    assert path == out / FILE_NAME and json.loads(path.read_text("utf-8"))["labelled"] == 6
    assert diagnostics.main(["--root", str(camp.root), "--out", str(tmp_path / "cli")]) == 0
    assert (tmp_path / "cli" / FILE_NAME).read_bytes() == path.read_bytes()
    assert diagnostics.main(["--root", str(tmp_path / "none"), "--out", str(out)]) == 1
    assert "refused" in capsys.readouterr().err
