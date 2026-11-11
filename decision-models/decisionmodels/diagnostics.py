"""`diagnostics.json`, the campaign's exploratory read-out (plan4
`docs/plans/typed-decision-models4.md` §3.6 "Diagnostics"; method note
`docs/plans/typed-decision-models-ml.md` §Q1.2 and §Q1.3; plan5 `typed-decision-models5.md` §3.6
"Round-1 reprint"; plan6 `typed-decision-models6.md` §3.8).

`build(state)` is pure over a folded `campaign.State`; `write(state, out_dir)` writes the file
beside `generation.json` (U5.10's `finalize` calls it; `python -m decisionmodels.diagnostics`
does too, for a campaign in flight). Everything is exploratory: a figure here gates nothing.
Counts, rates and slopes only; an item's text appears nowhere but in `promptExamples`' counts.

A **labelled item** is an accepted item with votes 1 to 3. Its **majority** is `labels.majority`
(`None` for a `voteTie`); its **class** is the majority label, and its **intent** the seed's
`intendedAnswer`. A rate over nothing is `None`.

- **`agreement`** (note §Q1.2): per answer class, `pairAgreement` (the share of the three vote
  pairs of the class's items that agree: a label-noise indicator, not a ceiling) and
  `threeZeroShare` beside it; `intentLabelAgreement` per intended class (the share whose majority
  is the intent; ties leave it, and are counted).
- **`positionEffect`**: per answer class, the share of votes with that answer by decile of the
  item's position in its label batch (`10 · index // batch size`). Positions are random, so a
  trend is causal. `balanceEffect`: per class, the OLS slope of "this item's vote is the class"
  on the share of its batch-mates that carry the class under the next vote (`v % 3 + 1`); a
  negative slope means the labeller balances its answers.
- **`cells`**: round 1 against top-up (`genRound > 1`) for each cell (design stratum, class), the
  planner's cell: items, intent-label agreement and pair agreement (note §Q1.3).
- **`screenHits`**: seeds drawn and hits by intent and by difficulty, and hits by pattern
  (`screen:meta`, `screen:token`, `screen:id`, `bindingMismatch`, `formMismatch`), every hit
  counted, regeneration ones included.
- **`turnCut`** (ML P3-n2): intent-label agreement for items whose rendering cut their earliest
  turns (more than `render.RECENT_TURNS_KEPT` turns with text) against the rest; guard and
  join-in only. It counts the six-turn cut alone: `render_guard_input`'s `MAX_CHARS` drop of
  the earliest turns is ignored (`turns_cut`), so a guard item with very long turns can be cut
  and not counted.
- **`promptExamples`** (ML P4-n2): items whose message contains, verbatim, an example sentence of
  the point's hand-off prompt (`HANDOFF_EXAMPLES`), per example and distinct; it drops nothing and
  is reported beside FR-10. `build(state)` counts every accepted item (a campaign in flight has no
  eval split); `finalize` passes `build` its eval split's items, so a train or calibration item
  carrying an example does not count.
- **`missingCueCheck`** (guard only; plan5 §3.6): round 1's `pilot.CueCheck` over all of its
  `understanding` items, exploratory, with `failed` omitted.
- **`rounds`**, **`templates`** and **`retired`** (plan6 §3.8): each round's draws, items and
  tokens; each generator-template hash with its runs and items, and every template halt raised;
  the batches a restart retired, with their attempts and tokens.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import campaign, labels, pilot, render
from .points import load_point

FILE_NAME = "diagnostics.json"
DECILES = 10

#: The worked example sentences of each new-ability point's hand-off prompt (model-bench's
#: `message-type-handoff.md` and `join-in-handoff.md`; `tests/test_diagnostics.py` holds each
#: verbatim in the prompt when `model-bench/` is present). Full sentences only: the prompts'
#: one- or two-word replies ("Yes", "that one", "thanks") would match unrelated items.
HANDOFF_EXAMPLES: Mapping[str, tuple[str, ...]] = {
    "message-type-classifier": (
        "Do you deliver on Sundays? If so I'd like two",
        "Where is my order?",
        "Where is my order? It was due yesterday",
    ),
    "join-in-judge": (
        "Morning Nova, how's your week going?",
        "Ask Nova later",
        "Nova, can you check this?",
        "Thanks Nova! Bia, can you bring the cake?",
    ),
}


@dataclass(frozen=True)
class Labelled:
    """One labelled item: its seed, round, stratum, the three votes and their majority."""

    seed_id: str
    gen_round: int
    stratum: str | None
    intent: str | None
    votes: tuple[Any, ...]
    majority: Any | None
    item: Mapping[str, Any]

    @property
    def cls(self) -> str | None:
        return None if self.majority is None else labels.vote_key(self.majority)

    @property
    def pairs_agreeing(self) -> int:
        a, b, c = self.votes
        return (a == b) + (a == c) + (b == c)


def labelled_items(state: campaign.State) -> list[Labelled]:
    expected = load_point(state.init["point"]).expected
    out = []
    for seed_id, record in sorted(state.items.items()):
        votes = state.three_votes(seed_id)
        if votes is None:
            continue
        seed = state.seeds[seed_id]
        intent = seed.axes.get("intendedAnswer")
        out.append(Labelled(seed_id, record["genRound"], seed.design_stratum,
                            None if intent is None else labels.vote_key(intent), tuple(votes),
                            labels.majority(expected, votes), record["item"]))
    return out


def _rate(part: int, whole: int) -> float | None:
    return part / whole if whole else None


def pair_agreement(rows: Iterable[Labelled]) -> dict[str, Any]:
    """`items`, `pairAgreement` and `threeZeroShare` over `rows` (module docstring)."""
    rows = list(rows)
    return {"items": len(rows),
            "pairAgreement": _rate(sum(r.pairs_agreeing for r in rows), 3 * len(rows)),
            "threeZeroShare": _rate(sum(r.pairs_agreeing == 3 for r in rows), len(rows))}


def intent_agreement(rows: Iterable[Labelled]) -> dict[str, Any]:
    """Per intended class and overall: the items with a majority, and the share whose majority
    is the intent."""
    rows = [r for r in rows if r.cls is not None and r.intent is not None]
    out: dict[str, Any] = {}
    for intent in sorted({r.intent for r in rows}):
        group = [r for r in rows if r.intent == intent]
        out[intent] = {"items": len(group),
                       "rate": _rate(sum(r.cls == intent for r in group), len(group))}
    out["overall"] = {"items": len(rows), "rate": _rate(sum(r.cls == r.intent for r in rows),
                                                        len(rows))}
    return out


def agreement(rows: Sequence[Labelled], classes: Sequence[str]) -> dict[str, Any]:
    return {"byClass": {cls: pair_agreement(r for r in rows if r.cls == cls) for cls in classes},
            "ties": sum(r.majority is None for r in rows),
            "intentLabelAgreement": intent_agreement(rows)}


def position_effect(state: campaign.State, classes: Sequence[str]) -> dict[str, Any]:
    """Answer rate by position decile, per class, over every recorded vote 1 to 3 (module
    docstring). Each decile holds `[votes, rate]`."""
    counts = {cls: [[0, 0] for _ in range(DECILES)] for cls in classes}
    for batch in state.batches.values():
        vote = batch.extra.get("vote")
        if (batch.kind != "label" or batch.state == "retired" or vote is None
                or not 1 <= vote <= labels.VOTES_PER_ITEM):
            continue
        for index, seed_id in enumerate(batch.entries):
            answer = state.votes.get(seed_id, {}).get(vote)
            if answer is None:
                continue
            decile = DECILES * index // len(batch.entries)
            for cls in classes:
                counts[cls][decile][0] += 1
                counts[cls][decile][1] += labels.vote_key(answer) == cls
    return {cls: [[n, _rate(k, n)] for n, k in rows] for cls, rows in counts.items()}


def balance_effect(state: campaign.State, classes: Sequence[str]) -> dict[str, Any]:
    """Per class, the OLS slope (module docstring) over every (item, vote) whose batch has at
    least one other item with a vote `v % 3 + 1` recorded; `None` when `x` does not vary."""
    points: dict[str, list[tuple[float, int]]] = {cls: [] for cls in classes}
    for batch in state.batches.values():
        vote = batch.extra.get("vote")
        if (batch.kind != "label" or batch.state == "retired" or vote is None
                or not 1 <= vote <= labels.VOTES_PER_ITEM):
            continue
        other = vote % labels.VOTES_PER_ITEM + 1
        for seed_id in batch.entries:
            answer = state.votes.get(seed_id, {}).get(vote)
            mates = [state.votes.get(m, {}).get(other) for m in batch.entries if m != seed_id]
            mates = [labels.vote_key(m) for m in mates if m is not None]
            if answer is None or not mates:
                continue
            for cls in classes:
                points[cls].append((mates.count(cls) / len(mates),
                                    int(labels.vote_key(answer) == cls)))
    return {cls: {"n": len(pts), "slope": _slope(pts)} for cls, pts in points.items()}


def _slope(points: Sequence[tuple[float, int]]) -> float | None:
    if len(points) < 2:
        return None
    mean_x = sum(x for x, _ in points) / len(points)
    spread = sum((x - mean_x) ** 2 for x, _ in points)
    if spread == 0:
        return None
    mean_y = sum(y for _, y in points) / len(points)
    return sum((x - mean_x) * (y - mean_y) for x, y in points) / spread


def cell_comparison(rows: Sequence[Labelled]) -> dict[str, Any]:
    """Round 1 against top-up for each (design stratum, class) cell with an item."""
    out: dict[str, Any] = {}
    cells = sorted({(r.stratum, r.cls) for r in rows if r.cls is not None},
                   key=lambda c: (c[0] or "", c[1]))
    for stratum, cls in cells:
        group = [r for r in rows if (r.stratum, r.cls) == (stratum, cls)]
        entry = {}
        for name, part in (("roundOne", [r for r in group if r.gen_round == 1]),
                           ("topUp", [r for r in group if r.gen_round > 1])):
            entry[name] = {**pair_agreement(part),
                           "intentLabelAgreement": _rate(sum(r.cls == r.intent for r in part),
                                                         len(part))}
        out[f"{stratum or 'none'}|{cls}"] = entry
    return out


def screen_hits(state: campaign.State) -> dict[str, Any]:
    """Seeds drawn and hits by intent and by difficulty, and hits by pattern."""
    def key(seed_id: str, axis: str) -> str:
        value = state.seeds[seed_id].axes.get(axis)
        return "none" if value is None else labels.vote_key(value)

    out: dict[str, Any] = {"byIntent": {}, "byDifficulty": {}, "byPattern": {}}
    for axis, name in (("intendedAnswer", "byIntent"), ("difficulty", "byDifficulty")):
        for seed_id in state.seed_round:
            row = out[name].setdefault(key(seed_id, axis), {"seeds": 0, "hits": 0, "hitSeeds": 0})
            row["seeds"] += 1
            row["hits"] += len(state.hits.get(seed_id, ()))
            row["hitSeeds"] += seed_id in state.hits
    for hits in state.hits.values():
        for hit in hits:
            for reason in hit["reasons"]:           # a hit showing two reasons counts under both
                pattern = (f"screen:{hit['detail'].split(':', 1)[0]}" if reason == "screen"
                           else reason)
                out["byPattern"][pattern] = out["byPattern"].get(pattern, 0) + 1
    return out


def turns_cut(role: str, item: Mapping[str, Any]) -> bool | None:
    """Whether the rendering cuts the item's earliest turns (more than `RECENT_TURNS_KEPT` turns
    with text), `None` for a point with no turn list."""
    field = {"guard-judge": "turns", "join-in-judge": "priorTurns"}.get(role)
    if field is None:
        return None
    valid = [t for t in item.get(field) or [] if isinstance(t.get("text"), str) and t["text"]]
    return len(valid) > render.RECENT_TURNS_KEPT


def turn_cut(role: str, rows: Sequence[Labelled]) -> dict[str, Any] | None:
    if role not in ("guard-judge", "join-in-judge"):
        return None
    out = {}
    for name, flag in (("cut", True), ("notCut", False)):
        group = [r for r in rows if turns_cut(role, r.item) is flag]
        out[name] = {"items": len(group), "intentLabelAgreement": intent_agreement(group)}
    return out


def _message(role: str, item: Mapping[str, Any]) -> str:
    if role == "join-in-judge":
        return str((item.get("newMessage") or {}).get("text", ""))
    return str(item.get("text", ""))


def prompt_example_counts(role: str, items: Iterable[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Items whose message contains a hand-off prompt example verbatim (module docstring), per
    example and distinct; `None` for a point with no examples. It drops nothing."""
    examples = HANDOFF_EXAMPLES.get(role)
    if examples is None:
        return None
    messages = [_message(role, item) for item in items]
    return {"items": len(messages),
            "matching": sum(any(e in m for e in examples) for m in messages),
            "perExample": {e: sum(e in m for m in messages) for e in examples}}


def cue_check_reprint(state: campaign.State) -> dict[str, Any] | None:
    """plan5 §3.6: guard's round-1 `CueCheck` over all its `understanding` items, with `failed`
    omitted (the pilot's decision is not the campaign's)."""
    if state.init["point"] != "guard-judge":
        return None
    record = pilot.cue_check(pilot._cue_items(state, [1])).as_record()
    record.pop("failed")
    return record


def rounds_report(state: campaign.State) -> list[dict[str, Any]]:
    out = []
    for gen_round in sorted({r["genRound"] for r in state.rounds}):
        spend = state.spend(gen_round)
        out.append({"genRound": gen_round, "drawn": state.drawn_in(gen_round),
                    "strata": {str(r["designStratum"]): len(r["seeds"])
                               for r in state.rounds if r["genRound"] == gen_round},
                    "items": spend["items"], "generationRuns": spend["generationRuns"],
                    "generationTokens": spend["generationTokens"],
                    "labelTokens": spend["labelTokens"]})
    return out


def templates_report(state: campaign.State) -> dict[str, Any]:
    """Each generator-template hash in first-use order with its runs and accepted items, and
    every template halt raised since the last restart (plan6 §3.8)."""
    hashes = list(dict.fromkeys(state.run_hash.values()))
    items_by_hash = dict.fromkeys(hashes, 0)
    for record in state.items.values():
        batch = next((b for b in state.batches.values()
                      if b.kind == "generate" and b.gen_run_id == record["genRunId"]), None)
        if batch is not None and batch.batch_id in state.run_hash:
            items_by_hash[state.run_hash[batch.batch_id]] += 1
    return {"hashes": [{"templateHash": h,
                        "runs": sum(v == h for v in state.run_hash.values()),
                        "items": items_by_hash[h]} for h in hashes],
            "halts": [dict(h) for h in state.template_log]}


def retired_report(state: campaign.State) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for batch in state.batches.values():
        if batch.state != "retired":
            continue
        row = out.setdefault(batch.kind, {"batches": 0, "attempts": 0,
                                          "tokens": dict.fromkeys(pilot.TOKEN_CLASSES, 0)})
        row["batches"] += 1
        for attempt in batch.attempts:
            tokens = ((attempt.outcome or {}).get("run") or {}).get("tokens")
            row["attempts"] += 1
            for cls in pilot.TOKEN_CLASSES:
                row["tokens"][cls] += (tokens or {}).get(cls, 0)
    return out


def build(state: campaign.State, *,
          eval_items: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """The whole `diagnostics.json` for a folded campaign (module docstring). `eval_items` is
    `finalize`'s eval split; `None` counts the accepted items."""
    role = state.init["point"]
    classes = [labels.vote_key(v) for v in load_point(role).expected]
    rows = labelled_items(state)
    report: dict[str, Any] = {
        "point": role, "labelled": len(rows),
        "agreement": agreement(rows, classes),
        "positionEffect": position_effect(state, classes),
        "balanceEffect": balance_effect(state, classes),
        "cells": cell_comparison(rows),
        "screenHits": screen_hits(state),
        "turnCut": turn_cut(role, rows),
        "promptExamples": prompt_example_counts(
            role, [r["item"] for r in state.items.values()] if eval_items is None
            else eval_items),
        "rounds": rounds_report(state),
        "templates": templates_report(state),
        "retired": retired_report(state),
    }
    if role == "guard-judge":
        report["missingCueCheck"] = cue_check_reprint(state)
    return report


def write(state: campaign.State, out_dir: Path, *,
          eval_items: Sequence[Mapping[str, Any]] | None = None) -> Path:
    """`diagnostics.json` in `out_dir`, beside `generation.json`."""
    path = out_dir / FILE_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    report = build(state, eval_items=eval_items)
    text = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    campaign._write_atomic(path, text.encode("utf-8"))
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m decisionmodels.diagnostics")
    parser.add_argument("--root", required=True, type=Path, help="the campaign root")
    parser.add_argument("--out", required=True, type=Path, help="the directory to write into")
    args = parser.parse_args(argv)
    try:
        print(write(campaign.Campaign(args.root).state(), args.out))
    except campaign.CampaignRefused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
