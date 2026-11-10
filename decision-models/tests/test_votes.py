"""Votes: majority of three, `labelVotes`, reshuffles and the label batches (typed-decision-models
U5.8b; plan4 `docs/plans/typed-decision-models4.md` §3.6 "Votes" and its §5 row U5.8; plan6
`typed-decision-models6.md` §3.7 and §3.8 "Votes").

Items are accepted through real generation runs over hand-built seeds (`test_generation.py`); the
labeler is the fixture's agent, so a label run stages from the clean transcript fixture. Each
answer is a function of the item and the vote, so a vote credited to the wrong item or the wrong
vote number changes a tally.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from test_campaign import (
    GUARD_COMMAND,
    PROCEED,
    ROLES,
    Clock,
    pilot_report,
    run_batch,
    stage,
    write_answer,
)
from test_generation import hand_round, hand_seed, run_generation
from test_screen import obedient_item

from decisionmodels import campaign, exchange, labels, seeds
from decisionmodels.campaign import Campaign, Serve, Stop

#: The generator is the fixture's agent (its runs stage as they are); the labeler pins another
#: model, which `retarget` writes into each label run's agent files.
LABELER_MODEL = "claude-haiku-5-6"
BOTH_AS_FIXTURE = {"generator": {**ROLES["labeler"]},
                   "labeler": {**ROLES["labeler"], "model": LABELER_MODEL}}
Respond = Callable[[str, int], Any]


def vote_bench(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cap: int = 3) -> Campaign:
    """A guard campaign whose generator and labeler are the fixture's agent, label batches of
    at most `cap`."""
    monkeypatch.setattr(campaign, "DEFAULT_LABEL_BATCH_CAP", cap)
    camp = Campaign(tmp_path / "camp", claude_dir=tmp_path / "claude", clock=Clock())
    camp.init(point="guard-judge", phase="pilot")
    camp.record_expectations(guard_command=GUARD_COMMAND, roles=BOTH_AS_FIXTURE)
    return camp


def settled_round(camp: Campaign, numbers: range, gen_round: int = 1) -> list[str]:
    """Round `gen_round` over hand seeds `numbers`, every one accepted. Returns the seed ids."""
    hand = [hand_seed(n) for n in numbers]
    hand_round(camp, hand, gen_round=gen_round)
    run_generation(camp, obedient_item)
    assert len(camp.state().items) >= len(hand)
    return [seed.seed_id for seed in hand]


def retarget(camp: Campaign, session: str) -> None:
    """Make the staged subagent's model the labeler's current pin."""
    pin = camp.state().pins["labeler"]
    folder = camp.transcript_path(session).with_suffix("") / "subagents"
    for path in folder.iterdir():
        path.write_text(path.read_text("utf-8").replace("claude-sonnet-5-5", pin), "utf-8")


def refuse_next(camp: Campaign) -> None:
    """Serve the next batch and end its run in a safeguard refusal: a halt to resolve."""
    action = camp.next()
    session = camp.issue(action)
    stage(camp, session, action.opaque, "refusal")
    retarget(camp, session)
    assert camp.ingest(action.opaque).stop == "halted"
    assert camp.state().halt["reason"] == "safeguardStop"


def run_vote(camp: Campaign, action: Serve, respond: Respond) -> campaign.IngestResult:
    """One label run for `action`: each key answered with `respond(item ref, vote)`."""
    vote = camp.state().batches[action.batch].extra["vote"]
    session = camp.issue(action)
    stage(camp, session, action.opaque)
    retarget(camp, session)
    keys = camp.state().opaques[action.opaque].keys
    text = (camp.store / "req" / f"{action.opaque}.jsonl").read_text("utf-8")
    lines = [{"bodySha256": json.loads(text.splitlines()[0])["bodySha256"]}]
    lines += [{"key": key, "answer": respond(ref, vote)} for key, ref in keys.items()]
    write_answer(camp, action.opaque,
                 "".join(json.dumps(line) + "\n" for line in lines).encode("utf-8"))
    return camp.ingest(action.opaque)


def vote_all(camp: Campaign, respond: Respond, limit: int = 60) -> list[Serve]:
    """Serve and answer every label batch until `next` stops; returns the batches served. A
    campaign still serving after `limit` batches fails instead of looping."""
    served = []
    for _ in range(limit):
        if not isinstance(action := camp.next(), Serve):
            return served
        assert action.kind == "label", action
        run_vote(camp, action, respond)
        served.append(action)
    raise AssertionError(f"still serving after {limit} label batches")


def vote_events(camp: Campaign, kind: str) -> list[dict[str, Any]]:
    return [e for e in camp.events() if e["event"] == kind]


def scripted(ids: list[str]) -> Respond:
    """Item *i* answers True on votes in a pattern that differs per item (3-0, 2-1 and 0-3)."""
    patterns = [(True, True, True), (True, True, False), (False, True, False),
                (True, False, True), (False, False, False), (False, True, True)]
    return lambda ref, vote: patterns[ids.index(ref) % len(patterns)][vote - 1]


# --- labels.py -----------------------------------------------------------------------------------


MESSAGE_TYPE = ("order", "question", "complaint", "escalation", "other")


def test_vote_keys_spell_a_predicate_in_lowercase_and_a_choice_as_itself() -> None:
    assert labels.vote_key(True) == "true" and labels.vote_key(False) == "false"
    assert labels.vote_key("complaint") == "complaint"


@pytest.mark.parametrize(("answers", "votes", "winner"), [
    ((True, True, True), {"true": 3, "false": 0}, True),
    ((False, True, False), {"true": 1, "false": 2}, False),
    ((True, False, True), {"true": 2, "false": 1}, True),
])
def test_label_votes_cover_the_predicate_space_with_zeros_and_sum_to_three(
        answers: tuple[bool, ...], votes: dict[str, int], winner: bool) -> None:
    assert labels.label_votes((True, False), answers) == votes
    assert labels.majority((True, False), answers) is winner


def test_a_choice_majority_is_two_of_three_and_three_distinct_labels_tie() -> None:
    assert labels.label_votes(MESSAGE_TYPE, ("order", "other", "order")) == {
        "order": 2, "question": 0, "complaint": 0, "escalation": 0, "other": 1}
    assert labels.majority(MESSAGE_TYPE, ("order", "other", "order")) == "order"
    assert labels.majority(MESSAGE_TYPE, ("other", "other", "other")) == "other"
    assert labels.majority(MESSAGE_TYPE, ("order", "other", "question")) is None
    assert sum(labels.label_votes(MESSAGE_TYPE, ("order", "other", "question")).values()) == 3


# --- the votes of a round ------------------------------------------------------------------------


def test_a_settled_round_votes_vote_major_and_labelvotes_sum_to_three(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    camp = vote_bench(tmp_path, monkeypatch, cap=3)
    ids = settled_round(camp, range(1, 9))                       # 8 items: batches of 3, 3, 2
    served = vote_all(camp, scripted(ids))
    state = camp.state()
    order = [state.batches[a.batch].extra["vote"] for a in served]
    assert order == [1, 1, 1, 2, 2, 2, 3, 3, 3]
    sizes = [len(state.batches[a.batch].entries) for a in served]
    assert sizes == [3, 3, 2] * 3
    patterns = {0: (3, 0), 1: (2, 1), 2: (1, 2), 3: (2, 1), 4: (0, 3), 5: (2, 1)}
    for index, seed_id in enumerate(ids):
        wanted_true, wanted_false = patterns[index % 6]
        assert state.label_votes(seed_id) == {"true": wanted_true, "false": wanted_false}
    assert state.vote_ties() == [] and camp.next() == Stop("done", "nothing is pending")
    assert [h for h in vote_events(camp, "vote-start")] and len(vote_events(camp, "votes")) == 9


def test_each_vote_reshuffles_on_a_recorded_held_seed(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    camp = vote_bench(tmp_path, monkeypatch, cap=50)
    ids = settled_round(camp, range(1, 21))
    held_before = set(camp.state().held_rng_seeds)
    vote_all(camp, scripted(ids))
    starts = vote_events(camp, "vote-start")
    assert [(e["genRound"], e["vote"]) for e in starts] == [(1, 1), (1, 2), (1, 3)]
    seeds_used = [e["rngSeed"] for e in starts]
    assert len(set(seeds_used)) == 3 and not set(seeds_used) & held_before
    assert set(seeds_used) <= camp.state().held_rng_seeds
    assert camp.state().reshuffle_seeds() == seeds_used
    for event in starts:
        wanted = sorted(ids)
        random.Random(f"label-votes|{event['rngSeed']}").shuffle(wanted)
        assert event["items"] == wanted                      # the order the seed gives
    orders = [e["items"] for e in starts]
    assert orders[0] != orders[1] != orders[2] and orders[0] != orders[2]
    assert all(sorted(order) == sorted(ids) for order in orders)


def test_a_vote_starts_only_when_every_batch_of_the_one_before_is_done(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    camp = vote_bench(tmp_path, monkeypatch, cap=3)
    ids = settled_round(camp, range(1, 8))
    for _ in range(3):                                           # vote 1: batches of 3, 3, 1
        run_vote(camp, camp.next(), scripted(ids))
        assert [e["vote"] for e in vote_events(camp, "vote-start")] == [1]
    first = camp.next()                                          # the first batch of vote 2
    assert [e["vote"] for e in vote_events(camp, "vote-start")] == [1, 2]
    assert camp.state().batches[first.batch].extra["vote"] == 2


def test_every_label_attempt_has_its_own_key_per_item_and_vote(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    camp = vote_bench(tmp_path, monkeypatch, cap=4)
    ids = settled_round(camp, range(1, 10))
    vote_all(camp, scripted(ids))
    labels_keys = [dict(a.keys) for a in camp.state().attempts
                   if camp.state().batches[a.batch].kind == "label"]
    flat = [key for keys in labels_keys for key in keys]
    assert len(flat) == 27 and len(set(flat)) == 27
    by_item: dict[str, list[str]] = {}
    for keys in labels_keys:
        for key, ref in keys.items():
            by_item.setdefault(ref, []).append(key)
    assert all(len(keys) == 3 for keys in by_item.values())


def test_a_round_does_not_vote_while_a_hit_awaits_regeneration(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from test_generation import leak_item, leaking
    camp = vote_bench(tmp_path, monkeypatch)
    hand = [hand_seed(n) for n in range(1, 7)]
    hand_round(camp, hand)
    run_generation(camp, leaking({hand[1].seed_id: leak_item}))
    state = camp.state()
    assert len(state.items) == 5 and state.awaiting_regeneration() == [hand[1].seed_id]
    assert not state.round_settled(1)                    # a hit awaits regeneration
    regen = camp.next()
    assert regen.kind == "generate" and vote_events(camp, "vote-start") == []
    run_generation(camp, obedient_item)
    assert camp.state().round_settled(1) and not camp.state().round_settled(2)
    action = camp.next()
    assert action.kind == "label"
    (start,) = vote_events(camp, "vote-start")
    assert sorted(start["items"]) == sorted(seed.seed_id for seed in hand)


def test_a_top_up_round_votes_its_own_items_after_the_earlier_round(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    camp = vote_bench(tmp_path, monkeypatch, cap=50)
    first = settled_round(camp, range(1, 6))
    vote_all(camp, scripted(first))
    second = settled_round(camp, range(101, 108), gen_round=2)
    vote_all(camp, scripted(second))
    starts = [(e["genRound"], e["vote"]) for e in vote_events(camp, "vote-start")]
    assert starts == [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]
    by_round = {e["genRound"]: set(e["items"]) for e in vote_events(camp, "vote-start")}
    assert by_round == {1: set(first), 2: set(second)}
    state = camp.state()
    assert all(state.label_votes(seed_id) is not None for seed_id in first + second)
    assert sum(sum(state.label_votes(i).values()) for i in first + second) == 3 * 12


def test_a_dropped_vote_batch_is_counted_once_and_its_items_leave_later_votes(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    camp = vote_bench(tmp_path, monkeypatch, cap=3)
    ids = settled_round(camp, range(1, 8))                       # vote 1 batches: 3, 3, 1
    camp.next()
    second = [b for b in camp.state().batches.values() if b.kind == "label"][1]
    dead = set(second.entries)
    good = scripted(ids)
    vote_all(camp, lambda ref, vote: "maybe" if ref in dead else good(ref, vote))
    state = camp.state()
    assert state.batches[second.batch_id].dropped == "retryCapExceeded"
    assert state.dropped()["retryCapExceeded"] == {"seeds": 0, "items": 3}
    starts = {e["vote"]: e["items"] for e in vote_events(camp, "vote-start")}
    assert set(starts[1]) == set(ids) and set(starts[2]) == set(starts[3]) == set(ids) - dead
    assert all(state.label_votes(seed_id) is None for seed_id in dead)
    assert all(state.label_votes(seed_id) is not None for seed_id in set(ids) - dead)
    assert all(len(state.votes.get(seed_id, {})) == 0 for seed_id in dead)
    assert "retryCapExceeded" in "\n".join(camp.status())


def test_a_skipped_vote_batch_also_takes_its_items_out_of_later_votes(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    camp = vote_bench(tmp_path, monkeypatch, cap=4)
    ids = settled_round(camp, range(1, 7))                       # batches of 4 and 2
    run_vote(camp, camp.next(), scripted(ids))
    refuse_next(camp)                                            # the second batch: a safeguardStop
    camp.resolve(decision="skip it", skip_batch=True)
    skipped = set(camp.state().batches["bat_0003"].entries)
    vote_all(camp, scripted(ids))
    starts = {e["vote"]: e["items"] for e in vote_events(camp, "vote-start")}
    assert len(skipped) == 2 and set(starts[2]) == set(ids) - skipped


def test_a_labeler_repin_clears_the_votes_retires_the_batches_and_repeats_from_vote_1(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    camp = vote_bench(tmp_path, monkeypatch, cap=3)
    ids = settled_round(camp, range(1, 6))
    for _ in range(2):
        run_vote(camp, camp.next(), scripted(ids))               # vote 1 (3 + 2)
    refuse_next(camp)                                            # vote 2's first batch halts
    items_before = dict(camp.state().items)
    old_seeds = camp.state().reshuffle_seeds()
    before = camp.ledger_path.read_bytes()
    camp.resolve(decision="a new labeler", repin_model=("labeler", "claude-opus-5-6"))
    state = camp.state()
    assert camp.ledger_path.read_bytes().startswith(before)
    assert state.votes == {} and state.vote_starts == {} and state.halt is None
    assert state.items == items_before and state.vote_ties() == []
    label_batches = [b for b in state.batches.values() if b.kind == "label"]
    assert {b.state for b in label_batches} == {"retired"} and label_batches
    assert state.dropped()["retryCapExceeded"] == {"seeds": 0, "items": 0}
    assert set(old_seeds) <= state.held_rng_seeds
    vote_all(camp, scripted(ids))
    state = camp.state()
    starts = vote_events(camp, "vote-start")
    assert [e["vote"] for e in starts] == [1, 2, 1, 2, 3]
    assert not set(old_seeds) & {e["rngSeed"] for e in starts[2:]}
    assert all(sum(state.label_votes(i).values()) == 3 for i in ids)


def test_a_generator_repin_clears_the_votes_with_the_items(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    camp = vote_bench(tmp_path, monkeypatch, cap=50)
    ids = settled_round(camp, range(1, 5))
    run_vote(camp, camp.next(), scripted(ids))
    refuse_next(camp)
    camp.resolve(decision="a new generator", repin_model=("generator", "claude-opus-5-6"))
    state = camp.state()
    assert state.items == {} and state.votes == {} and state.vote_starts == {}
    assert isinstance(camp.next(), Stop)


def test_a_label_batch_that_overflows_the_pages_continues_in_the_same_vote(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    camp = vote_bench(tmp_path, monkeypatch, cap=50)
    ids = settled_round(camp, range(1, 10))
    monkeypatch.setattr(exchange, "PAGE_BUDGET_CHARS", 5_500)
    monkeypatch.setattr(exchange, "P_MAX", {"label": 2, "generate": 2})
    served = vote_all(camp, scripted(ids))
    state = camp.state()
    per_vote: dict[int, list[str]] = {1: [], 2: [], 3: []}
    for action in served:
        batch = state.batches[action.batch]
        per_vote[batch.extra["vote"]] += batch.entries
    assert len(served) > 3
    assert all(sorted(entries) == sorted(ids) for entries in per_vote.values())
    order = [state.batches[a.batch].extra["vote"] for a in served]
    assert order == sorted(order)


def message_bench(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Campaign, list[str]]:
    """A full message-type campaign (label batch cap 25 from its pilot report) with six accepted
    items; returns it and their seed ids. These tests are about the vote path alone: the tranche
    and the top-up planner (`test_topup.py`) are switched off, since six items are all tranche and
    every cell short."""
    monkeypatch.setattr(campaign, "TRANCHE_POINT", None)
    monkeypatch.setattr(Campaign, "queue_topup", lambda self: [])
    data = tmp_path / "data"
    pilot_report(data, stakeholderDecision=PROCEED, missingCueCheck={"state": "passed"},
                 rngSeeds=[11, 12])
    (data / "guard-judge" / "run").mkdir(parents=True)
    (data / "guard-judge" / "run" / "generation.json").write_text("{}", "utf-8")
    monkeypatch.setattr(exchange, "RENDERERS", {"message-type-classifier": lambda i: i["text"]})
    camp = Campaign(tmp_path / "camp", claude_dir=tmp_path / "claude", clock=Clock())
    camp.init(point="message-type-classifier", phase="full", data_dir=data)
    camp.record_expectations(guard_command=GUARD_COMMAND, roles=BOTH_AS_FIXTURE)
    draw = seeds.draw_seeds("message-type-classifier", 6, 77, design_stratum=None)
    camp.record_draw(draw)
    ids = [seed.seed_id for seed in draw.seeds]
    camp._append({"event": "round-start", "genRound": 1, "rngSeed": 77, "designStratum": None,
                  "seeds": ids})
    camp.queue_batch("generate", ids, generation="first")

    def message(seed: seeds.Seed) -> dict[str, Any]:
        follow_up = seed.axes["threadPosition"] == "follow-up"
        return {"text": f"message number {ids.index(seed.seed_id)}",
                "priorTurn": {"speaker": "user", "text": "hello"} if follow_up else None}

    run_generation(camp, message)
    assert len(camp.state().items) == 6
    return camp, ids


def test_the_label_batch_cap_comes_from_the_pilot_report_and_a_choice_tie_is_counted(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    camp, ids = message_bench(tmp_path, monkeypatch)
    script = [("order", "order", "question"), ("order", "question", "complaint"),
              ("other", "other", "other"), ("escalation", "order", "escalation"),
              ("question", "complaint", "question"), ("complaint", "other", "escalation")]
    vote_all(camp, lambda ref, vote: script[ids.index(ref)][vote - 1])
    state = camp.state()
    assert state.init["labelBatchCap"] == 25
    assert state.label_votes(ids[0]) == {"order": 2, "question": 1, "complaint": 0,
                                         "escalation": 0, "other": 0}
    assert state.label_votes(ids[2]) == {"order": 0, "question": 0, "complaint": 0,
                                         "escalation": 0, "other": 3}
    assert sorted(state.vote_ties()) == sorted([ids[1], ids[5]])
    assert any("voteTie" in line and "2" in line for line in camp.status())
    assert {11, 12} <= state.held_rng_seeds and state.reshuffle_seeds()


def test_a_voteTie_needs_all_three_votes_so_an_item_at_two_distinct_votes_is_none(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    camp, ids = message_bench(tmp_path, monkeypatch)
    vote_all(camp, lambda ref, vote: {1: "order", 2: "question"}.get(vote, "maybe"))
    state = camp.state()
    assert state.batches[max(state.batches)].dropped == "retryCapExceeded"   # vote 3 dropped
    assert all(len(state.votes[i]) == 2 for i in ids)
    assert state.vote_ties() == [] and all(state.label_votes(i) is None for i in ids)
    assert not any("voteTie" in line for line in camp.status())


def test_the_smoke_fixtures_label_batch_records_no_votes(tmp_path: Path) -> None:
    from test_campaign import smoke
    camp = smoke(tmp_path, items=3)
    run_batch(camp)
    assert vote_events(camp, "votes") == [] and camp.state().votes == {}


def test_votes_recorded_before_a_crash_count_only_once_the_outcome_commits(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    camp = vote_bench(tmp_path, monkeypatch, cap=50)
    ids = settled_round(camp, range(1, 6))
    real = Campaign._append

    def crash_at_outcome(self: Campaign, event: dict[str, Any]) -> Any:
        if event["event"] == "outcome":
            raise RuntimeError("killed")
        return real(self, event)

    action = camp.next()
    monkeypatch.setattr(Campaign, "_append", crash_at_outcome)
    with pytest.raises(RuntimeError):
        run_vote(camp, action, scripted(ids))
    monkeypatch.setattr(Campaign, "_append", real)
    assert len(vote_events(camp, "votes")) == 1 and camp.state().votes == {}
    assert isinstance(camp.next(), Serve)                        # reconciled: vote 2 is served
    state = camp.state()
    assert {seed: dict(votes) for seed, votes in state.votes.items()} == {
        seed_id: {1: scripted(ids)(seed_id, 1)} for seed_id in ids}


@pytest.mark.parametrize("kill", [1, 2, 3])
def test_a_crash_between_a_votes_queued_events_queues_the_missing_items_into_it(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kill: int) -> None:
    camp = vote_bench(tmp_path, monkeypatch, cap=3)
    ids = settled_round(camp, range(1, 9))                       # 8 items: batches of 3, 3, 2
    real = Campaign.queue_batch
    queued = {"label": 0}

    def killed_at(self: Campaign, kind: str, entries: Any, **extra: Any) -> str:
        if kind == "label":
            queued["label"] += 1
            if queued["label"] == kill:
                raise RuntimeError("killed")
        return real(self, kind, entries, **extra)

    monkeypatch.setattr(Campaign, "queue_batch", killed_at)
    with pytest.raises(RuntimeError):
        camp.next()
    monkeypatch.setattr(Campaign, "queue_batch", real)
    vote_all(camp, scripted(ids))
    state = camp.state()
    assert [len(state.votes[i]) for i in ids] == [3] * 8
    assert all(sum(state.label_votes(i).values()) == 3 for i in ids)
    for vote in (1, 2, 3):
        held = [seed for b in state.vote_batches(1, vote) for seed in b.entries]
        assert sorted(held) == sorted(ids)                       # each item once, none twice
    assert [e["vote"] for e in vote_events(camp, "vote-start")] == [1, 2, 3]
    assert camp.next() == Stop("done", "nothing is pending")


def test_a_recovered_vote_keeps_its_recorded_order_and_cap(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    camp = vote_bench(tmp_path, monkeypatch, cap=3)
    settled_round(camp, range(1, 9))
    real = Campaign.queue_batch
    count = {"n": 0}

    def killed_at_second(self: Campaign, kind: str, entries: Any, **extra: Any) -> str:
        count["n"] += 1
        if count["n"] == 2:
            raise RuntimeError("killed")
        return real(self, kind, entries, **extra)

    monkeypatch.setattr(Campaign, "queue_batch", killed_at_second)
    with pytest.raises(RuntimeError):
        camp.next()
    monkeypatch.setattr(Campaign, "queue_batch", real)
    (start,) = vote_events(camp, "vote-start")
    assert camp.state().next_vote(1) is None                     # the first batch is pending
    camp.queue_votes()                                           # in flight: nothing queued
    assert len(camp.state().vote_batches(1, 1)) == 1
    run_vote(camp, camp.next(), scripted(start["items"]))
    assert camp.state().next_vote(1) == (1, start["items"][3:], True)
    queued = camp.queue_votes()
    state = camp.state()
    assert [state.batches[b].entries for b in queued] == [start["items"][3:6], start["items"][6:]]
    assert len(vote_events(camp, "vote-start")) == 1


def test_a_vote_in_flight_in_one_round_holds_back_the_votes_of_another(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    camp = vote_bench(tmp_path, monkeypatch, cap=50)
    settled_round(camp, range(1, 4))
    settled_round(camp, range(101, 104), gen_round=2)
    assert isinstance(camp.next(), Serve)                        # round 1, vote 1: pending
    assert camp.state().next_vote(1) is None and camp.queue_votes() == []
    assert [e["genRound"] for e in vote_events(camp, "vote-start")] == [1]


def test_label_request_shows_each_key_its_own_item_in_every_vote(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    camp = vote_bench(tmp_path, monkeypatch, cap=3)
    ids = settled_round(camp, range(1, 6))
    items = camp.state().items

    def conditions(ref: str) -> str:
        return f"CONDITION: {items[ref]['item']['condition']}\n"

    checked = 0
    while isinstance(action := camp.next(), Serve):
        shown = {json.loads(line)["key"]: json.loads(line)["text"]
                 for line in action.request.text.splitlines()[2:]}
        assert shown.keys() == action.request.keys.keys() and shown
        for key, ref in action.request.keys.items():
            assert shown[key].startswith(conditions(ref))
            assert not any(conditions(other) in shown[key] for other in ids if other != ref)
            checked += 1
        run_vote(camp, action, scripted(ids))
    assert checked == 15                                         # 5 items x 3 votes


def test_a_labeler_repin_retires_a_label_batch_that_holds_no_vote(tmp_path: Path) -> None:
    from test_campaign import smoke
    camp = smoke(tmp_path, items=3)
    refuse_next(camp)                                            # the fixture's batch halts
    plain = camp.queue_batch("label", ["fx_0001"])
    camp.resolve(decision="a new labeler", repin_model=("labeler", "claude-opus-5-6"))
    state = camp.state()
    assert {state.batches["bat_0001"].state, state.batches[plain].state} == {"retired"}
    assert camp.next() == Stop("done", "nothing is pending")
