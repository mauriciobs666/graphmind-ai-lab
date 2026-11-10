"""The guard pilot, `python -m decisionmodels.pilot` (plan4 `docs/plans/typed-decision-models4.md`
§3.8; plan5 `typed-decision-models5.md` §3.6; note §Q1.3 "Pilot").

The pilot is a guard campaign (`driver init --point guard-judge --phase pilot`) whose rounds are
**blocks** (`campaign.MAIN_BLOCK`, `RECHECK_BLOCK`), numbered 1, 2, … in issue order. The generation
runs, the screen and the votes are the campaign's; this module holds the pilot's own decisions.

- **`start`** draws the first main block: `FIRST_BLOCK_SEEDS` (100: 4 generation runs). It takes
  votes 1 to 3 at 50 per batch and the screen vote at 10 (`labels.SCREEN_VOTE`). The driver
  (`Campaign.next`, `queue_pilot_block`) draws each further main block of `ADDED_BLOCK_SEEDS` (50)
  while `screen(...).more_needed` (`n < 80`); `start` is the only block a human draws.
- **`screen`** prints `n`, `Δ` and the cap: over the `n` items carrying votes 1, 2 and the screen
  vote, `Δ` is how many differ between vote 1 and the screen vote, less how many differ between
  vote 1 and vote 2. `10·Δ ≥ n` sets the cap to 25, else 50. Fixed before any label is seen.
- **`cue_check(items)`** is the note's `missing` cue check (pure): per stratum (difficulty ×
  intended answer), `x` counter-cue items whose majority label departs from intent and `y`
  concordant ones that do. Given each stratum's `x + y`, `x` is hypergeometric, so the null law of
  `Σx` is the convolution of the strata's; it fails when `P(Σx ≥ observed) ≤ 0.05`. The p-value is
  exact (`fractions.Fraction`), so the equality case `p = 1/20` fails and is tested as such.
- **`report`** is the one evaluator. It writes `data/guard-judge/pilot/pilot-report.json`
  (`campaign.PILOT_REPORT`), counts only: no item text, no seed text. It refuses a partly voted
  block (the main blocks until `n ≥ 80`, the recheck block until every vote is ingested) and,
  once `stakeholderDecision` is recorded, any rewrite. `missingCueCheck.state` is derived from the
  ledger on every run: `passed`; `failed` (it appends a `cue-check-halt` event, which `driver
  resolve` clears); `recheck-pending`; `recheck-passed`; `recheck-failed`.
- **`recheck`** refuses unless the state is `failed`, a `resolve` has cleared the halt and no
  decision is recorded. It draws the recheck block (`RECHECK_BLOCK_SEEDS`, 100) and sets
  `recheck-pending`.
- **`record-decision`** writes `stakeholderDecision` once (`outcome`, `cap`, `decidedOn`, `ref`,
  and `stakeholderRef` when given). A `proceed` must satisfy the spend gate
  (`campaign.spend_gate_refusal`, the same predicate `init --phase full` applies); a first `stop`
  is always recordable. The store is deleted by hand only after a decision (plan4 §3.8).

**Token model.** A labelling run's tokens in each class (`input`, `cacheWrite`, `cacheRead`,
`output`) are `F + c × items`: `F` and `c` are the least-squares line over the pilot's ingested
label runs, which need two batch sizes (50 and the screen vote's 10). The generator's cost is its
mean per first-pass run (every one is 25 seeds, so no `F`/`c` split). The per-point projection is
the whole labelled volume at the note's planning factor (`PLAN_TENTHS`, 1.3: 2,340 labelled
for 1,800 kept, the generated seeds equal to the labelled items; regeneration runs are left out;
the report records both as `planFactor` and `assumes`), generation runs of 25, three votes at
the pilot's cap.

Pure stdlib (FR-8). `campaign` and this module import each other: `campaign` calls `screen` and
reads the block sizes, and this module calls `campaign` only inside its functions, never at
import time, which is what lets the cycle load.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import campaign, labels, seeds
from .points import ROLES, load_point

if TYPE_CHECKING:
    from .campaign import Campaign, State

#: Note §Q1.3 "Pilot": 100 items from 4 generation runs, then blocks of 50 (2 runs) until
#: `MIN_ITEMS` carry the screen vote. The recheck block is 100 (4 runs), plan5 §3.6.
FIRST_BLOCK_SEEDS = 100
ADDED_BLOCK_SEEDS = 50
RECHECK_BLOCK_SEEDS = 100
MIN_ITEMS = 80
#: plan6 §3.8: added main blocks per halt epoch before `Campaign.queue_pilot_block` halts.
PILOT_ADDED_BLOCK_LIMIT = 3
#: The screen: `CAP_FACTOR · Δ ≥ n` sets the cap to `SCREEN_CAP`, else `DEFAULT_CAP`.
CAP_FACTOR = 10
SCREEN_CAP = 25
DEFAULT_CAP = 50
#: The note's counter-cue forms (plan5 §3.6); every other form is concordant.
COUNTER_CUE_FORMS = frozenset({"immaterial", "unlisted", "ambiguous"})
#: The cue check fails at `p ≤` this (note §Q1.3), compared exactly.
CUE_ALPHA = Fraction(1, 20)
#: The note's labelling volume: about 2,340 labelled for 1,800 kept, so 1.3 × the target.
PLAN_TENTHS = 13
TOKEN_CLASSES = ("input", "cacheWrite", "cacheRead", "output")
PROJECTION_ASSUMES = ("1.3 x the point's target as items, generated and labelled; generation runs "
                      "of 25 seeds; three votes at the pilot's cap; no regeneration or top-up "
                      "runs; F, c and the generator mean as measured here")
VOTES_RECHECK_RUNS = 2 * labels.VOTES_PER_ITEM


class PilotRefused(Exception):
    """A refused pilot command; nothing was written."""


# --- the screen ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Screen:
    """The pilot's screen: `n` items carry votes 1, 2 and the screen vote; `delta` and the cap."""

    n: int
    delta: int
    cap: int

    @property
    def more_needed(self) -> bool:
        return self.n < MIN_ITEMS


def screen(votes: Mapping[str, Mapping[int, Any]]) -> Screen:
    """The pilot's `n`, `Δ` and cap over `votes` (item → vote number → answer), module
    docstring."""
    counted = [v for v in votes.values()
               if 1 in v and 2 in v and labels.SCREEN_VOTE in v]
    delta = (sum(v[1] != v[labels.SCREEN_VOTE] for v in counted)
             - sum(v[1] != v[2] for v in counted))
    return Screen(n=len(counted), delta=delta,
                  cap=SCREEN_CAP if CAP_FACTOR * delta >= len(counted) else DEFAULT_CAP)


# --- the cue check -------------------------------------------------------------------------------


@dataclass(frozen=True)
class CueItem:
    """One labelled `understanding` item: its stratum (difficulty × intended answer), its
    `missing` form, and whether its majority label departs from intent."""

    stratum: str
    form: str
    departs: bool

    @property
    def counter_cue(self) -> bool:
        return self.form in COUNTER_CUE_FORMS


@dataclass(frozen=True)
class Stratum:
    """`x` of `m1` counter-cue items depart, `y` of `m0` concordant ones do."""

    stratum: str
    x: int
    m1: int
    y: int
    m0: int

    def as_record(self) -> dict[str, Any]:
        return {"stratum": self.stratum, "x": self.x, "m1": self.m1, "y": self.y, "m0": self.m0,
                "counterCueRate": self.x / self.m1 if self.m1 else None,
                "concordantRate": self.y / self.m0 if self.m0 else None}


@dataclass(frozen=True)
class CueCheck:
    strata: tuple[Stratum, ...]
    sum_x: int
    p: Fraction
    failed: bool

    def as_record(self) -> dict[str, Any]:
        return {"strata": [s.as_record() for s in self.strata], "sumX": self.sum_x,
                "p": float(self.p), "failed": self.failed}


def _hypergeometric(m1: int, m0: int, total: int) -> dict[int, Fraction]:
    """`P(x)` for `x` counter-cue items among `total` departures over `m1 + m0` items."""
    whole = math.comb(m1 + m0, total)
    return {x: Fraction(math.comb(m1, x) * math.comb(m0, total - x), whole)
            for x in range(max(0, total - m0), min(m1, total) + 1)}


def cue_check(items: Iterable[CueItem]) -> CueCheck:
    """The note's exact stratified conditional test, one-sided (module docstring): fails when
    `P(Σx ≥ observed) ≤ CUE_ALPHA` under the convolution of the strata's hypergeometrics."""
    counts: dict[str, list[int]] = {}
    for item in items:
        row = counts.setdefault(item.stratum, [0, 0, 0, 0])
        row[0 if item.counter_cue else 2] += item.departs
        row[1 if item.counter_cue else 3] += 1
    strata = tuple(Stratum(name, x, m1, y, m0) for name, (x, m1, y, m0) in sorted(counts.items()))
    null: dict[int, Fraction] = {0: Fraction(1)}
    for s in strata:
        law = _hypergeometric(s.m1, s.m0, s.x + s.y)
        merged: dict[int, Fraction] = {}
        for have, p_have in null.items():
            for x, p_x in law.items():
                merged[have + x] = merged.get(have + x, Fraction(0)) + p_have * p_x
        null = merged
    observed = sum(s.x for s in strata)
    p = sum((p for total, p in null.items() if total >= observed), Fraction(0))
    return CueCheck(strata, observed, p, p <= CUE_ALPHA)


# --- tokens ---------------------------------------------------------------------------------------


def fit_line(points: Sequence[tuple[int, int]]) -> tuple[float, float]:
    """Least squares `y = F + c·x` over `(x, y)` points; `PilotRefused` unless two batch sizes."""
    if len({x for x, _ in points}) < 2:
        raise PilotRefused("F and c need label runs at two batch sizes; the pilot has "
                           f"{sorted({x for x, _ in points})}")
    mean_x = sum(x for x, _ in points) / len(points)
    mean_y = sum(y for _, y in points) / len(points)
    slope = (sum((x - mean_x) * (y - mean_y) for x, y in points)
             / sum((x - mean_x) ** 2 for x, _ in points))
    return mean_y - slope * mean_x, slope


def _label_costs(state: State) -> dict[str, dict[str, float]]:
    """`F` and `c` per token class over the ingested label runs (batch size, tokens)."""
    runs = [(len(batch.entries), attempt.outcome["run"]["tokens"])
            for batch in state.batches.values() if batch.kind == "label"
            for attempt in batch.attempts
            if attempt.outcome and attempt.outcome["outcome"] == "ingested"
            and attempt.outcome.get("run")]
    costs = {}
    for cls in TOKEN_CLASSES:
        floor, marginal = fit_line([(size, tokens.get(cls, 0)) for size, tokens in runs])
        costs[cls] = {"F": floor, "c": marginal}
    return costs


def _generator_mean(state: State) -> dict[str, float]:
    """Mean tokens per ingested first-pass generation run, per class."""
    runs = [attempt.outcome["run"]["tokens"]
            for batch in state.batches.values()
            if batch.kind == "generate" and batch.extra.get("generation") == "first"
            for attempt in batch.attempts
            if attempt.outcome and attempt.outcome["outcome"] == "ingested"
            and attempt.outcome.get("run")]
    if not runs:
        raise PilotRefused("no ingested generation run to price")
    return {cls: sum(t.get(cls, 0) for t in runs) / len(runs) for cls in TOKEN_CLASSES}


def _priced(generation_runs: int, label_runs: int, item_votes: int,
            generator: Mapping[str, float], labeller: Mapping[str, Mapping[str, float]]
            ) -> dict[str, Any]:
    tokens = {cls: round(generation_runs * generator[cls] + label_runs * labeller[cls]["F"]
                         + item_votes * labeller[cls]["c"]) for cls in TOKEN_CLASSES}
    return {"generationRuns": generation_runs, "labelRuns": label_runs, "itemVotes": item_votes,
            "tokens": {**tokens, "total": sum(tokens.values())}}


def _projection(cap: int, generator: Mapping[str, float],
                labeller: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
    """The planned volume's tokens per point (module docstring)."""
    out = {}
    for role in ROLES:
        items = -(-PLAN_TENTHS * seeds.target_count(role) // 10)
        out[role] = _priced(-(-items // 25), labels.VOTES_PER_ITEM * -(-items // cap),
                            labels.VOTES_PER_ITEM * items, generator, labeller)
    return out


# --- reading the campaign -------------------------------------------------------------------------


def _rounds(state: State, block: str) -> list[int]:
    return [r["genRound"] for r in state.rounds if r["pilotBlock"] == block]


def _cue_items(state: State, rounds: Sequence[int]) -> list[CueItem]:
    """The labelled `understanding` items of `rounds`: majority of votes 1 to 3 against intent."""
    expected = load_point("guard-judge").expected
    out = []
    for seed_id, record in state.items.items():
        votes = state.three_votes(seed_id)
        axes = state.seeds[seed_id].axes
        if (record["genRound"] not in rounds or votes is None
                or axes["evidenceKind"] != "understanding"):
            continue
        majority = labels.majority(expected, votes)
        if majority is None:
            continue
        out.append(CueItem(f"{axes['difficulty']}|{labels.vote_key(axes['intendedAnswer'])}",
                           axes["missingForm"], majority != axes["intendedAnswer"]))
    return out


def _label_rates(state: State, rounds: Sequence[int]) -> dict[str, Any]:
    """`P̂(label | evidence kind)` over the labelled items of `rounds`, with the counts."""
    expected = load_point("guard-judge").expected
    counts: dict[str, dict[str, int]] = {}
    for seed_id, record in state.items.items():
        votes = state.three_votes(seed_id)
        if record["genRound"] not in rounds or votes is None:
            continue
        majority = labels.majority(expected, votes)
        if majority is None:
            continue
        row = counts.setdefault(state.seeds[seed_id].axes["evidenceKind"],
                                {labels.vote_key(v): 0 for v in expected})
        row[labels.vote_key(majority)] += 1
    return {kind: {"counts": row, "n": sum(row.values()),
                   "p": {label: n / sum(row.values()) for label, n in row.items()}}
            for kind, row in sorted(counts.items())}


def _runs_report(state: State) -> dict[str, Any]:
    """Voids, reruns and hits: counts only."""
    voids: dict[str, dict[str, int]] = {}
    for attempt in state.attempts:
        if attempt.outcome and attempt.outcome["outcome"] == "void":
            role = state.batches[attempt.batch].kind
            reason = attempt.outcome.get("reason") or "unknown"
            voids.setdefault(role, {})[reason] = voids.setdefault(role, {}).get(reason, 0) + 1
    first_pass: dict[str, int] = {}
    for (_, reason), hits in state.round_hits.items():
        first_pass[reason] = first_pass.get(reason, 0) + hits
    regenerated = sum(1 for hits in state.hits.values() for h in hits if not h["firstPass"])
    dropped = {reason: dict(counts) for reason, counts in state.dropped().items()}
    given_up = {"items": 0}
    for batch in state.batches.values():
        if batch.kind == "label" and batch.dropped and (
                batch.extra.get("vote") == labels.SCREEN_VOTE):
            dropped[batch.dropped]["items"] -= len(batch.entries)
            given_up["items"] += len(batch.entries)
    return {
        "voids": voids,
        "reruns": sum(len(b.attempts) - 1 for b in state.batches.values() if b.attempts),
        "screenHits": {"firstPass": first_pass, "regeneration": regenerated,
                       "dropped": state.generation_drops()},
        "dropped": dropped, "droppedScreenVote": given_up, "voteTie": len(state.vote_ties()),
    }


def cue_state(state: State) -> str:
    """The `missingCueCheck.state` the ledger gives (module docstring); `PilotRefused` while the
    block to evaluate is not fully voted."""
    mains = _rounds(state, "main")
    if not mains:
        raise PilotRefused("no main pilot block drawn (run `pilot.py start`)")
    unvoted = [g for g in mains if not state.round_voted(g)]
    if unvoted:
        raise PilotRefused(f"main block(s) {unvoted} are not fully voted")
    if screen(state.screen_votes(mains)).more_needed:
        if state.halt is not None and state.halt["reason"] == "pilotBlocks":
            raise PilotRefused(
                f"fewer than {MIN_ITEMS} items carry the screen vote after "
                f"{PILOT_ADDED_BLOCK_LIMIT} added blocks: the pilot is halted (pilotBlocks); "
                "read the dropped reasons, then `driver resolve`")
        raise PilotRefused(f"fewer than {MIN_ITEMS} items carry the screen vote: `driver run` "
                           "draws another block")
    rechecks = _rounds(state, "recheck")
    if rechecks:
        if not state.round_voted(rechecks[0]):
            return "recheck-pending"
        return "recheck-failed" if cue_check(_cue_items(state, rechecks)).failed \
            else "recheck-passed"
    return "failed" if cue_check(_cue_items(state, mains)).failed else "passed"


# --- the commands ---------------------------------------------------------------------------------


def _report_path(data_dir: Path) -> Path:
    return data_dir / campaign.PILOT_REPORT


def _read_report(data_dir: Path) -> dict[str, Any] | None:
    path = _report_path(data_dir)
    return json.loads(path.read_text("utf-8")) if path.is_file() else None


def _write_report(data_dir: Path, report: Mapping[str, Any]) -> Path:
    path = _report_path(data_dir)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    campaign._write_atomic(path, text.encode("utf-8"))
    return path


def start(camp: Campaign) -> list[str]:
    """Draw the first main block (module docstring). Refuses outside a pilot campaign, and a
    second call (`begin_round` refuses a `genRound` already started)."""
    with camp.locked():
        state = camp.state()
        if state.init["phase"] != "pilot":
            raise PilotRefused("`start` is for a pilot campaign")
        if state.rounds:
            raise PilotRefused("the pilot already has a block")
        return camp.begin_round(1, FIRST_BLOCK_SEEDS, pilot_block="main")


def screen_lines(camp: Campaign) -> list[str]:
    state = camp.state()
    mains = _rounds(state, "main")
    result = screen(state.screen_votes(mains))
    return [f"n {result.n}, delta {result.delta}, cap {result.cap}",
            "another block is needed (n < 80)" if result.more_needed else "n is enough"]


def build_report(camp: Campaign) -> dict[str, Any]:
    """The report the ledger gives, without its `stakeholderDecision`."""
    state = camp.state()
    state_name = cue_state(state)
    if state_name == "recheck-pending":
        raise PilotRefused("the recheck block is not fully voted")
    mains, rechecks = _rounds(state, "main"), _rounds(state, "recheck")
    result = screen(state.screen_votes(mains))
    generator, labeller = _generator_mean(state), _label_costs(state)
    cue: dict[str, Any] = {"state": state_name,
                           "pilot": cue_check(_cue_items(state, mains)).as_record(),
                           "recheck": (cue_check(_cue_items(state, rechecks)).as_record()
                                       if state_name in ("recheck-passed", "recheck-failed")
                                       else None)}
    report: dict[str, Any] = {
        "point": "guard-judge", "blocks": {"main": len(mains), "recheck": len(rechecks)},
        "n": result.n, "delta": result.delta, "cap": result.cap,
        "labeller": labeller, "generatorMeanPerRun": generator,
        "labelRates": _label_rates(state, mains), **_runs_report(state),
        "reprojectedTokens": _projection(result.cap, generator, labeller),
        "planFactor": PLAN_TENTHS / 10,
        "assumes": PROJECTION_ASSUMES,
        "missingCueCheck": cue,
        "recheckProjection": (_priced(RECHECK_BLOCK_SEEDS // 25, VOTES_RECHECK_RUNS,
                                      labels.VOTES_PER_ITEM * RECHECK_BLOCK_SEEDS,
                                      generator, labeller)
                              if state_name != "passed" else None),
        "rngSeeds": sorted(state.held_rng_seeds),
    }
    return report


def report(camp: Campaign, data_dir: Path) -> list[str]:
    """`pilot.py report` (module docstring). A first `failed` appends the `cue-check-halt`
    before the file is written, so a crash between them leaves the halt and a rerun writes."""
    with camp.locked():
        existing = _read_report(data_dir)
        if existing is not None and "stakeholderDecision" in existing:
            raise PilotRefused("the report holds a stakeholderDecision; it is final")
        built = build_report(camp)
        if built["missingCueCheck"]["state"] == "failed" and not camp.state().cue_halts:
            camp.record_cue_check_halt()
        path = _write_report(data_dir, built)
    return [f"wrote {path}", f"missingCueCheck: {built['missingCueCheck']['state']}",
            f"n {built['n']}, delta {built['delta']}, cap {built['cap']}"]


def recheck(camp: Campaign, data_dir: Path) -> list[str]:
    """`pilot.py recheck` (module docstring)."""
    with camp.locked():
        state = camp.state()
        current = cue_state(state)
        current_report = _read_report(data_dir)
        if current != "failed":
            raise PilotRefused(f"the cue check is {current!r}, not 'failed'")
        if not state.cue_halt_resolved:
            raise PilotRefused("no `driver resolve` has cleared the cue-check halt")
        if current_report is None:
            raise PilotRefused("no pilot report: run `pilot.py report` first")
        if "stakeholderDecision" in current_report:
            raise PilotRefused("the stakeholder has decided; a new decision needs a fresh pilot")
        camp.begin_round(max(r["genRound"] for r in state.rounds) + 1, RECHECK_BLOCK_SEEDS,
                         pilot_block="recheck")
        current_report["missingCueCheck"]["state"] = "recheck-pending"
        path = _write_report(data_dir, current_report)
    return [f"recheck block drawn; {path} is recheck-pending",
            "`driver run` labels it; then `pilot.py report`"]


def record_decision(data_dir: Path, *, outcome: str, cap: int, decided_on: str, ref: str,
                    stakeholder_ref: str | None = None) -> list[str]:
    """`pilot.py record-decision`: write `stakeholderDecision` once (module docstring)."""
    existing = _read_report(data_dir)
    if existing is None:
        raise PilotRefused("no pilot report: run `pilot.py report` first")
    if "stakeholderDecision" in existing:
        raise PilotRefused("a stakeholderDecision is already recorded (single-write)")
    if outcome not in ("proceed", "stop"):
        raise PilotRefused(f"--outcome is proceed or stop, not {outcome!r}")
    if cap not in (SCREEN_CAP, DEFAULT_CAP):
        raise PilotRefused(f"--cap is {SCREEN_CAP} or {DEFAULT_CAP}, not {cap!r}")
    try:
        date.fromisoformat(decided_on)
    except ValueError:
        raise PilotRefused(f"--decided-on is a YYYY-MM-DD date, not {decided_on!r}") from None
    if not ref.strip():
        raise PilotRefused("--ref names the ledger line")
    decision: dict[str, Any] = {"outcome": outcome, "cap": cap, "decidedOn": decided_on,
                                "ref": ref}
    if stakeholder_ref is not None:
        if not stakeholder_ref.strip():
            raise PilotRefused("--stakeholder-ref is empty")
        decision["stakeholderRef"] = stakeholder_ref
    if outcome == "proceed":
        refusal = campaign.spend_gate_refusal({**existing, "stakeholderDecision": decision})
        if refusal is not None:
            raise PilotRefused(f"proceed refused: {refusal}")
    _write_report(data_dir, {**existing, "stakeholderDecision": decision})
    return [f"recorded {outcome} (cap {cap}) in {_report_path(data_dir)}"]


# --- the CLI --------------------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m decisionmodels.pilot")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, helptext in (("start", "draw the first main block"),
                           ("screen", "print n, delta and the cap"),
                           ("report", "evaluate the pilot and write pilot-report.json"),
                           ("recheck", "draw the recheck block after a failed cue check")):
        sub = commands.add_parser(name, help=helptext)
        sub.add_argument("--root", required=True, type=Path)
        if name in ("report", "recheck"):
            sub.add_argument("--data-dir", type=Path, default=campaign.DATA_DIR)
    decide = commands.add_parser("record-decision", help="record the stakeholder's decision")
    decide.add_argument("--outcome", required=True, choices=("proceed", "stop"))
    decide.add_argument("--cap", required=True, type=int, choices=(SCREEN_CAP, DEFAULT_CAP))
    decide.add_argument("--decided-on", required=True)
    decide.add_argument("--ref", required=True)
    decide.add_argument("--stakeholder-ref")
    decide.add_argument("--data-dir", type=Path, default=campaign.DATA_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "record-decision":
            lines = record_decision(args.data_dir, outcome=args.outcome, cap=args.cap,
                                    decided_on=args.decided_on, ref=args.ref,
                                    stakeholder_ref=args.stakeholder_ref)
        else:
            camp = campaign.Campaign(args.root)
            if args.command == "start":
                lines = [f"drew the first block: {len(start(camp))} generation runs queued"]
            elif args.command == "screen":
                lines = screen_lines(camp)
            elif args.command == "report":
                lines = report(camp, args.data_dir)
            else:
                lines = recheck(camp, args.data_dir)
    except (PilotRefused, campaign.CampaignRefused) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
