"""`driver finalize`: turn a finished `full` campaign into a dataset version (plan4
`docs/plans/typed-decision-models4.md` §3.6 "Cells, then splits" and §3.7; plan2
`typed-decision-models2.md` §3.5; plan6 `typed-decision-models6.md` §3.7 and §3.8 "U5.10
hand-offs"; method note `typed-decision-models-ml.md` §Q1.3).

It writes `data/<role>/<datasetVersion>/` (`--data-dir` moves the root): `items.jsonl`,
`splits.json`, `generation.json` (§3.7's keys) and `diagnostics.json`. Everything is built first and
renamed into place whole, so a refusal writes no dataset and a crash leaves at most a `.partial`
directory. The ledger gains one `finalized` event (the split's RNG seed joins the held set); the
only other writes are a `recheck-halt` and a `floor-short` (below). `--rng-seed` fixes the seed.

**What it refuses** (each names its fix; none writes a file):
1. a campaign that is not `full`, has no install record, has a halt open, has a batch not ingested
   or dropped, or has a round not yet labelled;
2. a message-type point whose `rubric.md` now hashes differently from `state.tranche["rubric"]`
   (plan6 §3.8: both hashes are named; restoring the file clears it; a point with no tranche is
   not checked);
3. a point with a second generator-template hash unless that hash's first run was issued after a
   `resolve` that cleared a template halt (`State.template_resolutions` against each
   `Attempt.epoch`, over live batches only: a generator restart retires the earlier ones), and a
   role whose ingested runs name more than one model, or one model for both roles;
4. a short cell (fewer labelled items than `campaign.cell_target`): with no decline recorded the
   top-up is still due; after `resolve --decline-topup` the refusal names `--short` and
   `resolve --reopen-topup`. **`--short`** needs that decline, then finalizes with the short cells
   printed (refusal 5's floors still apply). `--short` on a point with no short cell is an ordinary
   finalize.
5. **a cell under its floor** (`point.json`'s `evalFloorPerCell`, on every finalize): an eval or
   calibration cell below it after the splits, the shingle drop and the refill. `build_dataset`
   (pure) raises `FloorShort`; `finalize` catches it inside `camp.locked()`, appends `floor-short`
   (per short cell `n` and `need` = `⌈deficit / (1 − d)⌉` more labelled items, and the RNG seed,
   now held) and re-raises. No halt: `State.floor_short` folds the newest event.

**The whole-point recheck** (plan4 §3.6, message type only): the tranche's rule
(`campaign.tranche_failed`) over every labelled item. A failure appends `recheck-halt` and
refuses; `resolve` clears it with a changed rubric (`point-relabel`: every item's three votes are
taken again) and a second failure only with `--stakeholder-ref` (`recheck-accepted`), after which
`finalize` goes ahead with the ties counted as `voteTie` drops.

**The dataset.** A cell is (design stratum, label). `split.make_splits` runs the plan's steps (a
dropped eval slot is refilled from the cell's unused candidates) over the labelled items (those
with three votes and a majority) with `eval` and `calibration` of
`campaign.cell_target // 2` per cell and `train` 0 (note §Q1.3's train column is U8's), so
`splits.json`'s `train` is empty. Only items in a split list reach `items.jsonl`: a surplus,
side-check or shingle-dropped item is omitted. An item is its input fields, `expected` (the
majority), `origin: "synthetic"`, `genRunId`, `genRound`, `seedId`, `labelVotes` and, for message
type, the seed's `threadPosition`. `itemId`s number the kept items in seed-id order. `diagnostics`
counts the hand-off prompt examples over the eval split alone.

**`generation.json`** takes `model` from the ingested runs' transcripts (never the tier alias;
`tier` is the plan's fixed `opus`/`sonnet`), `effort` from the install's pin, `promptSha256` from
the last-issued live attempt's `expectations` with `rubric` rewritten by `campaign.rubric_hash`,
`runs[]` from the live batches' recorded runs, and `dropped` from the folded state and the split.
"""

from __future__ import annotations

import copy
import json
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import UTC
from pathlib import Path
from typing import Any

from . import campaign, diagnostics, exchange, labels, shingles, split
from .campaign import Campaign, CampaignRefused, State
from .points import load_point

#: plan4 §3.7: the constant `surface` and the plan's `tier` per role.
SURFACE = "claude-code-subagent"
TIERS = {"generator": "opus", "labeler": "sonnet"}
ITEM_PREFIX = {"guard-judge": "gj", "join-in-judge": "ji", "message-type-classifier": "mt"}
#: A cell whose eval-assigned drop rate `d` exceeds this is flagged in the printed line.
DROP_RATE_FLAG = 0.25
#: The files of a dataset version, in write order.
ITEMS_FILE, SPLITS_FILE, GENERATION_FILE = "items.jsonl", "splits.json", "generation.json"
_VERSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_PROMPT_KEYS = {"agentDefinition": {"generator", "labeler", "driver"},
                "spawnTemplate": {"generator", "labeler", "driver"},
                "requestTemplate": {"generator", "labeler"}}


@dataclass(frozen=True)
class CellFigures:
    """One cell's printed figures (plan4 §3.6 "Cells, then splits"): the split sizes, the
    remaining `surplus`, and the refill's own counts (all counts of items)."""

    stratum: str | None
    label: str
    sizes: dict[str, int]
    surplus: int
    n: int                 # labelled items in the cell
    eval_assigned: int     # eval slots `assign_splits` filled, before the shingle drop
    assigned_drops: int    # of those, dropped by the eval-against-calibration/train check
    walked: int            # unused candidates the refill walked (`refilled` + `refill_drops`)
    refill_drops: int
    refilled: int

    @property
    def drop_rate(self) -> float:
        return self.assigned_drops / self.eval_assigned if self.eval_assigned else 0.0

    def need(self, floor: int) -> int | None:
        """`⌈deficit / (1 − d)⌉` more labelled items (plan4 §3.6 "Drop-aware top-up") for an eval
        cell below `floor`, in exact integers; 0 for a cell with no eval deficit (calibration
        alone is short only under `cell_target`, which the planner already covers); `None` when
        every assigned eval item was dropped, so no survival rate is observed."""
        deficit = floor - self.sizes["eval"]
        if deficit <= 0:
            return 0
        if self.assigned_drops == 0:
            return deficit
        if self.assigned_drops >= self.eval_assigned:
            return None
        kept = self.eval_assigned - self.assigned_drops
        return -(-deficit * self.eval_assigned // kept)


class FloorShort(CampaignRefused):
    """`build_dataset`'s refusal for a cell below its floor. `cells` holds, per short cell, the
    `floor-short` event's `{stratum, label, n, need}`; `finalize` records it (`build_dataset`
    stays pure)."""

    def __init__(self, message: str, cells: list[dict[str, Any]]) -> None:
        super().__init__(message)
        self.cells = cells


@dataclass(frozen=True)
class Dataset:
    """What `build_dataset` computed: the three files' contents, the eval items' inputs for the
    diagnostics, and the printable per-cell figures."""

    items: list[dict[str, Any]]
    splits: dict[str, Any]
    generation: dict[str, Any]
    eval_items: list[dict[str, Any]]
    cells: list[CellFigures]
    floor: int = 0


def eval_per_cell(role: str) -> int:
    """`evalPerCell`: calibration equals it, so it is half of `campaign.cell_target`."""
    return campaign.cell_target(role) // 2


def eval_floor(role: str) -> int:
    """`evalFloorPerCell` (note §Q1.3's table): what `--short` holds eval and calibration to."""
    return load_point(role).definition["cells"]["evalFloorPerCell"]


def all_cells(role: str) -> list[tuple[str | None, str]]:
    """Every (design stratum, label) cell of the point, in design order, empty ones included."""
    return [(stratum, labels.vote_key(value)) for stratum in campaign.design_strata(role)
            for value in load_point(role).expected]


# --- refusals ------------------------------------------------------------------------------------


def _live_attempts(state: State) -> list[campaign.Attempt]:
    return [a for a in state.attempts if state.batches[a.batch].state != "retired"]


def hash_refusal(state: State) -> str | None:
    """plan6 §3.8's one-hash rule over the live generation attempts in issue order: the first
    attempt of each generator-template hash after the first must follow a template-halt `resolve`
    that came after the previous hash's first attempt."""
    firsts: dict[str, campaign.Attempt] = {}
    for attempt in _live_attempts(state):
        if state.batches[attempt.batch].kind == "generate":
            firsts.setdefault(campaign.generator_template_hash(attempt.expectations), attempt)
    ordered = list(firsts.items())
    for (_, before), (digest, after) in zip(ordered, ordered[1:], strict=False):
        if not any(before.epoch < e <= after.epoch for e in state.template_resolutions):
            return (f"the generator template changed (hash {digest[:12]}, first run "
                    f"{after.batch}) with no `driver resolve` of a template halt between it and "
                    "the previous hash's first run: start a new campaign, or finalize one whose "
                    "change followed a template halt")
    return None


def _short_cells(state: State, role: str) -> list[str]:
    """The cells below the planner's target, named for a refusal; empty when none is short."""
    try:
        need = campaign.topup_counts(role, state.labelled_cells(), state.floor_targets())
    except campaign.UnboundedCell as exc:
        return [f"({exc.label}, {exc.stratum or 'no stratum'}) has no draw count "
                "(no labelled item, or every eval item dropped)"]
    return [f"stratum {stratum or 'none'} needs about {seeds} more seeds"
            for stratum, seeds in need.items()]


def _readiness(state: State, role: str) -> None:
    init = state.init
    if init["phase"] != "full":
        raise CampaignRefused(f"finalize needs a full campaign, not {init['phase']}")
    if state.expectations is None:
        raise CampaignRefused("no `driver install` is recorded: nothing says what generated it")
    if state.halt is not None:
        raise CampaignRefused(f"the campaign is halted ({campaign._halt_text(state.halt)}); "
                              "clear it with `driver resolve`")
    unfinished = [b.batch_id for b in state.batches.values()
                  if b.state not in ("ingested", "dropped", "retired")]
    rounds = {r["genRound"] for r in state.rounds}
    if unfinished or not rounds or not all(state.round_voted(g) for g in rounds):
        raise CampaignRefused(
            "the campaign is not finished ("
            + (f"batches not done: {', '.join(unfinished[:3])}" if unfinished
               else "a round is unlabelled") + "): `driver run` first")
    if state.tranche is not None:
        now = campaign.rubric_hash(role)
        if now != state.tranche["rubric"]:
            raise CampaignRefused(
                f"rubric.md hashes to {now} but the tranche was labelled under "
                f"{state.tranche['rubric']}: restore the file (a rubric fix after the tranche "
                "passed means a new campaign)")
    if (refusal := hash_refusal(state)) is not None:
        raise CampaignRefused(refusal)


def _models(state: State) -> dict[str, str]:
    """One model per role from its ingested runs, refusing a mixture or equal ids."""
    found: dict[str, set[str]] = {"generator": set(), "labeler": set()}
    for attempt in _live_attempts(state):
        run = (attempt.outcome or {}).get("run")
        if run and run["status"] == "ingested" and run["model"]:
            found[campaign.ROLE_OF_KIND[run["role"]]].add(run["model"])
    models = {}
    for role, names in found.items():
        if len(names) != 1:
            raise CampaignRefused(f"the {role}'s ingested runs name {sorted(names) or 'no model'}"
                                  ": generation.json records exactly one")
        models[role] = next(iter(names))
    if models["generator"] == models["labeler"]:
        raise CampaignRefused(f"the generator and the labeler are both {models['generator']}")
    return models


# --- the dataset ---------------------------------------------------------------------------------


def _prompt_hashes(state: State, role: str) -> dict[str, Any]:
    attempts = _live_attempts(state)
    recorded = (attempts[-1].expectations if attempts else state.expectations) or {}
    hashes = copy.deepcopy(recorded.get("promptSha256"))
    ok = isinstance(hashes, dict) and "guardHook" in hashes and all(
        isinstance(hashes.get(key), dict) and set(hashes[key]) == names
        for key, names in _PROMPT_KEYS.items())
    if not ok:
        raise CampaignRefused("the last install recorded no complete promptSha256: run "
                              "`driver install` again before finalize")
    hashes["rubric"] = campaign.rubric_hash(role)
    return hashes


def _runs(state: State) -> list[dict[str, Any]]:
    runs = [copy.deepcopy(a.outcome["run"]) for a in _live_attempts(state)
            if (a.outcome or {}).get("run")]
    return sorted(runs, key=lambda run: run["runId"])


def build_dataset(state: State, *, rng_seed: int, side_check: frozenset[str],
                  generated_at: str) -> Dataset:
    """The dataset of a ready campaign (module docstring). Every cell is held to its floor:
    `FloorShort` (a `CampaignRefused`) names those below it and carries what `finalize` records."""
    role = state.init["point"]
    point = load_point(role)
    per_cell = eval_per_cell(role)
    rows = [row for row in diagnostics.labelled_items(state) if row.cls is not None]
    result = split.make_splits(
        [split.Candidate(row.seed_id, (row.stratum, row.cls),
                         shingles.evidence_text(role, row.item)) for row in rows],
        sizes={"eval": per_cell, "calibration": per_cell, "train": 0},
        rng_seed=rng_seed, side_check=side_check)

    cell_of = {row.seed_id: (row.stratum, row.cls) for row in rows}
    in_cell = {cell: 0 for cell in all_cells(role)}
    for cell in cell_of.values():
        in_cell[cell] = in_cell.get(cell, 0) + 1
    assigned_drops = Counter(cell_of[seed] for seed in result.dropped["shingleSplit"])
    refill_drops = Counter(cell_of[seed] for seed in result.refill_dropped)
    cells = []
    for cell in all_cells(role):
        sizes = {name: result.sizes.get(cell, {}).get(name, 0) for name in split.SPLITS}
        refilled = result.refilled.get(cell, 0)
        cells.append(CellFigures(
            cell[0], cell[1], sizes, result.surplus.get(cell, 0), in_cell[cell],
            sizes["eval"] - refilled + assigned_drops[cell], assigned_drops[cell],
            refilled + refill_drops[cell], refill_drops[cell], refilled))
    floor = eval_floor(role)
    short = [figures for figures in cells
             if figures.sizes["eval"] < floor or figures.sizes["calibration"] < floor]
    if short:
        raise FloorShort(
            f"a cell is below its floor of {floor} ("
            + "; ".join(f"({f.label}, {f.stratum or 'no stratum'}): eval {f.sizes['eval']}, "
                        f"calibration {f.sizes['calibration']}" for f in short)
            + "): no dataset is written; a `floor-short` event records each cell's n and the "
            "labelled items it needs, and holds the RNG seed",
            [{"stratum": f.stratum, "label": f.label, "n": f.n, "need": f.need(floor)}
             for f in short])

    kept = sorted(result.eval + result.calibration + result.train)
    item_id = {seed: f"{ITEM_PREFIX[role]}-{n:04d}" for n, seed in enumerate(kept, 1)}
    by_seed = {row.seed_id: row for row in rows}
    items = []
    for seed_id in kept:
        row, record = by_seed[seed_id], state.items[seed_id]
        item: dict[str, Any] = {"itemId": item_id[seed_id]}
        item.update({name: row.item[name] for name in point.input_fields})
        item.update(expected=row.majority, origin="synthetic", genRunId=record["genRunId"],
                    genRound=record["genRound"], seedId=seed_id,
                    labelVotes=state.label_votes(seed_id))
        if role == "message-type-classifier":
            item["threadPosition"] = state.seeds[seed_id].axes["threadPosition"]
        items.append(item)
    ids = {name: [item_id[seed] for seed in getattr(result, name)] for name in split.SPLITS}
    splits = {"rngSeed": rng_seed, "eval": ids["eval"], "train": ids["train"],
              "calibration": ids["calibration"]}

    models, roles = _models(state), (state.expectations or {})["roles"]
    drops, batch_drops = state.generation_drops(), state.dropped()
    generation = {
        **{role_name: {"surface": SURFACE, "tier": TIERS[role_name], "model": models[role_name],
                       "effort": roles[role_name]["effort"]}
           for role_name in ("generator", "labeler")},
        "votesPerItem": labels.VOTES_PER_ITEM, "labelerTemperature": None,
        "labelBatchCap": state.init["labelBatchCap"],
        "promptSha256": _prompt_hashes(state, role),
        "generatedAt": generated_at, "itemCount": len(items),
        "dropped": {
            "voteTie": len(state.vote_ties()),
            "shingleSplit": len(result.dropped["shingleSplit"]),
            "shingleSideCheck": len(result.dropped["shingleSideCheck"]),
            "screenHits": drops["screenHits"],
            "retryCapExceeded": batch_drops["retryCapExceeded"],
            "safeguardSkipped": batch_drops["safeguardSkipped"],
            "formMismatch": drops["formMismatch"],
            "bindingMismatch": drops["bindingMismatch"]},
        "runs": _runs(state)}
    eval_set = set(ids["eval"])
    return Dataset(items, splits, generation,
                   [item for item in items if item["itemId"] in eval_set], cells, floor)


# --- the command ---------------------------------------------------------------------------------


def _text(value: Any) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _write_version(final: Path, data: Dataset, state: State) -> None:
    stage = final.with_name(final.name + ".partial")
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    lines = "".join(exchange.dump_line(item) + "\n" for item in data.items)
    campaign._write_atomic(stage / ITEMS_FILE, lines.encode("utf-8"))
    campaign._write_atomic(stage / SPLITS_FILE, _text(data.splits))
    campaign._write_atomic(stage / GENERATION_FILE, _text(data.generation))
    diagnostics.write(state, stage, eval_items=data.eval_items)
    stage.rename(final)


def _recheck(camp: Campaign, state: State) -> None:
    """The whole-point tranche rule (module docstring): a failure is recorded and refused."""
    if state.tranche is None or state.recheck_accepted:
        return
    drops, size = len(state.vote_ties()), sum(
        1 for seed in state.items if state.three_votes(seed) is not None)
    if not campaign.tranche_failed(drops, size):
        return
    rubric = campaign.rubric_hash(state.init["point"])
    camp._append({"event": "recheck-halt", "drops": drops, "items": size, "rubric": rubric})
    second = state.recheck_failures >= 1
    raise CampaignRefused(
        f"the tranche's rule fails over the whole point ({drops} voteTie drops in {size} "
        "labelled items, more than 3%): a halt is recorded; "
        + ("this is the second failure, so only the stakeholder clears it: "
           "`driver resolve --stakeholder-ref <who and where recorded>`" if second else
           "fix rubric.md, then `driver resolve` relabels the whole point"))


def finalize(camp: Campaign, *, version: str, data_dir: Path = campaign.DATA_DIR,
             short: bool = False, rng_seed: int | None = None,
             side_check_file: Path = shingles.SIDE_CHECK_FILE) -> list[str]:
    """`driver finalize` (module docstring): the lines to print, or `CampaignRefused`."""
    if not _VERSION.fullmatch(version) or version.endswith(".partial"):
        raise CampaignRefused(f"--dataset-version {version!r} is not a plain name")
    with camp.locked():
        state = camp.state()
        role = state.init["point"]
        target = data_dir / role / version
        if target.exists():
            raise CampaignRefused(
                f"{target} exists: a dataset version is never overwritten"
                + ("" if version in state.finalized_versions else
                   " (the ledger records no `finalized` event for it: an unrecorded version "
                   "from an interrupted finalize; move it aside, or pick another name)"))
        _readiness(state, role)
        shorts = _short_cells(state, role)
        if shorts and short and not state.topup_declined:
            raise CampaignRefused("--short needs a recorded `driver resolve --decline-topup`: "
                                  "the top-up is still the stakeholder's to decline")
        if shorts and not short:
            raise CampaignRefused(
                f"cells are short ({'; '.join(shorts)}): "
                + ("a top-up was declined; `finalize --short` finalizes with every n printed if "
                   "each cell meets its floor, and `driver resolve --reopen-topup` plans the "
                   "top-up again" if state.topup_declined else "`driver run` plans the top-up"))
        _recheck(camp, state)
        seed = rng_seed if rng_seed is not None else camp.fresh_rng_seed()
        if seed in state.held_rng_seeds:
            raise CampaignRefused(f"RNG seed {seed} is already held")
        stamp = camp.clock().astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            data = build_dataset(state, rng_seed=seed, side_check=shingles.load_side_check_hashes(
                side_check_file), generated_at=stamp)
        except FloorShort as exc:
            camp._append({"event": "floor-short", "rngSeed": seed, "cells": exc.cells})
            raise
        target.parent.mkdir(parents=True, exist_ok=True)
        _write_version(target, data, state)
        camp._append({"event": "finalized", "datasetVersion": version, "rngSeed": seed,
                      "items": len(data.items), "dir": str(target)})
    return _report(role, version, target, data, bool(shorts))


def _report(role: str, version: str, target: Path, data: Dataset, short: bool) -> list[str]:
    sizes = {name: len(data.splits[name]) for name in split.SPLITS}
    lines = [f"finalized {role} as {version}: {len(data.items)} items (eval {sizes['eval']}, "
             f"calibration {sizes['calibration']}, train {sizes['train']}) in {target}"]
    if short:
        lines.append("SHORT: finalized under a declined top-up; every cell meets its floor")
    for cell in data.cells:
        lines.append(
            f"  cell {cell.stratum or '-'} / {cell.label}: eval {cell.sizes['eval']}, "
            f"calibration {cell.sizes['calibration']}, surplus {cell.surplus} | eval-assigned "
            f"{cell.eval_assigned}, assigned drops {cell.assigned_drops}, walked {cell.walked}, "
            f"refill drops {cell.refill_drops}, refilled {cell.refilled}, floor {data.floor}, "
            f"d {cell.drop_rate:.2f}"
            + (f" (above {DROP_RATE_FLAG}: a generation-diversity signal)"
               if cell.drop_rate > DROP_RATE_FLAG else ""))
    lines.append("dropped: " + json.dumps(data.generation["dropped"], sort_keys=True))
    return lines
