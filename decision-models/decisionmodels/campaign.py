"""The campaign: its append-only ledger, batch states, reconciliation, halts and `resolve` (plan
`docs/plans/typed-decision-models4.md` §3.3 caps and "Idempotence", §3.5; plan5
`typed-decision-models5.md` §3.2 seed identity and §3.6 spend gate; plan6
`typed-decision-models6.md` §3.2 "Per batch", "Resume", the stop switch and §3.5's `--skip-batch`;
and the carried U5.6 rulings in `docs/plans/typed-decision-models-coordination.md`).

**Layout** under a campaign root (outside any git work tree and any `CLAUDE.md`/`AGENTS.md`
ancestor): `exchange/` (`req/`, `ans/`, `empty-mcp.json`; U5.7's `install` adds `.claude/`) and
`store/` (`ledger.jsonl`, `items.jsonl`, `req/` and `ans/` copies, `runs/<sessionId>.*` from the
runner, `pause`, `refusal`, `driver.lock`).

**The ledger** is the state: one JSON object per line, appended and fsync'd, never rewritten.
Every event carries `event` and `at` (UTC, ISO 8601). Batch states fold from it: `queued` →
`pending`; `issued {batch, opaque, sessionId, …}` → `issued`; then one `outcome` event per attempt,
which is the commit point of an ingest (the ledger's U5.6 ruling): `ingested`; `void` (counted:
re-issued with fresh keys, and dropped as `retryCapExceeded` at `MAX_COUNTED_ATTEMPTS`); `reissued`
and `usage-stop` (no attempt, a new opaque name); `halted`. A halt stops `next` until `resolve`
writes `resolved`, `generation-restart` or `labelling-restart`.

**`next` and `ingest` are library entry points for `driver run`** (plan6 §3.2), not subcommands.
`next` returns `Stop` while `store/pause` exists (reading nothing else), on a halt, and when
nothing is pending; otherwise it reconciles every `issued` attempt through its recorded
`sessionId`, requires `exchange/req/` and `exchange/ans/` to be empty, and returns a `Serve` for
the first pending batch. `issue` then records `issued` and writes the request. **A usage stop** is
remembered by the `Campaign` instance whose `ingest` recorded it (`Campaign.usage_stop`), whether
`run`'s step 5 or `next`'s own reconciliation called it, and that instance's `next` then returns
`Stop("usage-stop")`, so plan6 §3.2's "exit when `next` returns stop" launches once per spent
window. The memory is deliberately the instance's, never the ledger's: each `run`
is a new instance, which serves the batch again, and that one launch is how a run learns
whether the window has reset (plan6 §3.2 "Scheduling"). `ingest` also returns
`IngestResult.stop` for a halt or a usage stop.

**Ingest** of an attempt with no outcome yet audits the run (`audit.audit_run`, with `Expected`
built from absolute paths and `Launch` from ledger timestamps and the `launchFailBound` recorded
at `init`), runs the answer checks only when the audit allows (`Audit.ingests_answer`), joins them
(`audit.combine`), appends the `outcome` event, and only then moves the answer parts to `store/`:
copy all, fsync, then unlink. **Re-ingest** of an attempt that has an outcome never calls
`audit_run` or `combine`: a recorded `answerSha256` is re-read through `ingest_answer_files` from
`store/ans/` when it holds every recorded part, else from `exchange/ans/` (`AlreadyIngested`
finishes the move; a different answer raises `AnswerRefused`); a name with no `answerSha256`
(re-issued, usage stop, halt) is retired, and an answer part it did not record, or one that differs
from its stored copy, raises `AnswerRefused`. Either way nothing is appended.

**Generation** (plan4 §3.6, plan5 §3.2/§3.5, plan6 §3.8; `screen.py`). `begin_round` draws a
round's seeds (`record_draw`), appends `round-start {genRound, seeds}` and queues them, shuffled,
as first-pass runs (`generation: "first"`) in a balanced cut of `⌈n/25⌉` runs (`balanced_runs`);
`queue_regeneration` (which `next` calls when nothing is pending) queues the seeds a hit left
awaiting as `"regen"` runs, and `next` first queues any `round-start` seed no batch holds. A run's
`genRunId` is its batch's (`Batch.gen_run_id`); an item's `genRound` is its seed's round, never the
batch's. At ingest, before the `outcome` commits it, each item is checked: a clean run's items go
in one `generated` event, a rejected one in a `screen-hit`/`binding-mismatch`/`form-mismatch`
event (`reasons` lists every reason it shows, first = the one counted toward the seed's 3 hits).
`fold` applies both only at the run's ingested `outcome`, then evaluates the `template` halts
(`State.halt`, `voidClass: "template"`): per (`genRound`, reason) over that round's drawn seeds
after each run (more than 4% and at least two hits), and per (generator-template hash, reason)
over the pool of rounds under 25 drawn seeds (more than 5%, at least 300 pooled seeds), each
joining when its first-pass runs are all ingested or dropped, with each ingested run's seeds and
hits added to the pool of that run's own issue-time hash. Each pair fires once; regeneration
hits are excluded. An attempt carries the `expectations`, model pins and instructions in force at
its `issued` (`Attempt`), which the audit and the hash read. A `generation-restart` (a generator
re-pin) resets the folded generation view and retires every batch (`_restart_generation`). A
batch queued by plain `queue_batch` carries no `generation` and is not checked.

**Votes** (plan4 §3.6, plan6 §3.8; `labels.py`). A round's accepted items vote once every seed of
its `genRound` is accepted or dropped (`State.round_settled`) and no halt is open. `next`, when
nothing is pending, calls `queue_votes`: for the first such round with votes left it starts the
next vote with a `vote-start {genRound, vote, rngSeed, items}` event (a fresh RNG seed, held; the
items reshuffled on it) and queues the items as label batches of at most `labelBatchCap`
(`prepare` cuts what does not fit in `P_MAX` pages into the next batch of that vote). Votes run
vote-major: vote *v+1* starts when every batch of vote *v* is ingested or dropped. A `votes`
event, appended before the run's `outcome` and applied only at its ingested `outcome`, holds each
item's answer. A batch dropped (`retryCapExceeded` or `--skip-batch`) takes its items out of the
round's later votes. `State.label_votes` is an item's `labelVotes` once it has three votes, and
`State.vote_ties` the choice items with three distinct labels (`voteTie`). A `labelling-restart`
clears the votes and retires every label batch; the items stay. A batch queued without `vote`
(the smoke fixture's) records no votes. **The pilot** (`pilot.py`; plan4 §3.8, plan5 §3.6): a
`round-start`'s `pilotBlock` (`MAIN_BLOCK`, `RECHECK_BLOCK`) adds the screen vote (vote 4, batches
of `labels.SCREEN_BATCH_CAP`) to a main block's votes, which is never one of the three that give
`labelVotes`; `next` calls `queue_pilot_block` for each added main block; `pilot.py report` records
the `cue-check-halt` event (`record_cue_check_halt`), a halt with no batch that `resolve` clears.
A vote a crash left half-queued is incomplete until its live batches hold every `vote-start`
item; `queue_votes` queues the rest into it.

**Counting.** `earlier_no_attempt` is the batch's earlier `neverSpawned`/`incomplete` outcomes,
leaving out attempts an `interrupted {batch}` event names (plan6 §3.2; the audit's own flag exempts
only the current one). The `launchFailed` clock reads `issued` times: earlier `launchFailed`/
`launchFailing` attempts, and the last attempt that produced an `Agent` call or a `<synthetic>`.
Both counts restart after a `resolve`: the human's decision is a fresh start for the batch.

Choices this module makes where the plans leave room:
- the child's exit code is `store/runs/<sessionId>.exit`, which U6.3's runner writes atomically
  (a `.partial` then `os.replace`) after `wait()` and before `ingest`: ASCII, an optional `-` and
  decimal digits, optional surrounding whitespace (`-15` and a newline); negative is
  killed by that signal, as `Popen.returncode` gives it. A file that is missing, empty or not in
  that form is an exit never observed: it counts as non-zero (a launch that produced nothing is
  then `launchFailed`, not capped) and is recorded as `launch.exitCode: null`, never raised;
- `--skip-batch` is refused for a halt other than `safeguardStop`, the flag its count names;
- the guard pilot's report is read for `stakeholderDecision.{outcome, cap, stakeholderRef}`,
  `missingCueCheck.state` and `rngSeeds` (plan4 §3.8, plan5 §3.2, §3.6; U5.8 writes them), with
  no defaults: an absent or mistyped key refuses `init --phase full` (`cap` 25 or 50, `rngSeeds` a
  non-empty list of ints, `stakeholderRef` a non-empty string).
"""

from __future__ import annotations

import fcntl
import hashlib
import itertools
import json
import os
import random
import re
import secrets
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from fractions import Fraction
from pathlib import Path
from typing import Any

from . import audit, exchange, labels, pilot, screen
from .ingest import AlreadyIngested, AnswerRefused, Ingested, ingest_answer_files
from .points import ROLES, input_field_errors, load_point
from .seeds import Seed, SeedDraw, draw_seeds, round_one_count

PHASES = ("smoke", "pilot", "full")
#: plan6 §3.2: `batchTimeout` defaults to 45 minutes; §3.4: `launchFailBound` to 6 hours.
DEFAULT_BATCH_TIMEOUT = timedelta(minutes=45)
DEFAULT_LAUNCH_FAIL_BOUND = timedelta(hours=6)
#: plan4 §3.6: label batches of at most 50 items unless the pilot sets 25; generation runs of 25.
DEFAULT_LABEL_BATCH_CAP = 50
GENERATION_RUN_SIZE = 25
#: plan4 §3.3: a batch with 2 counted attempts is re-issued; after 3, its entries drop.
MAX_COUNTED_ATTEMPTS = 3
#: note §Q1.2 "On a hit": a seed's third hit (screen, form or binding) drops it.
MAX_SEED_HITS = 3
#: plan6 §3.8 (note §Q1.2 "Halt"): a round halts when more than 4% of its drawn seeds show one
#: reason in first pass, and at least `HALT_MIN_HITS` do.
HALT_PERCENT = 4
HALT_MIN_HITS = 2
#: plan6 §3.8's pool: rounds of fewer drawn seeds than `POOL_BELOW` (the rounds that cannot halt
#: alone) join it, and it is tested once it holds `POOL_MIN_SEEDS` seeds, at `POOL_PERCENT`, not
#: the round's `HALT_PERCENT`: a backstop for gross leaks, tested after each joining round.
POOL_PERCENT = 5
POOL_BELOW = GENERATION_RUN_SIZE
POOL_MIN_SEEDS = 300
#: The ledger events that record one rejected generated item, by the reason that rejected it.
HIT_EVENTS = {screen.SCREEN: "screen-hit", screen.BINDING_MISMATCH: "binding-mismatch",
              screen.FORM_MISMATCH: "form-mismatch"}
_HIT_EVENT_KINDS = frozenset(HIT_EVENTS.values())
#: plan4 §3.6 "Message-type tranche": round 1's labelling starts with `TRANCHE_ITEMS` items drawn at
#: random; all three votes finish on them before the rest, and the campaign halts (`rubric`) when
#: more than `TRANCHE_DROP_PERCENT` of them are `voteTie` drops. Message type's, in `full`.
TRANCHE_POINT = "message-type-classifier"
TRANCHE_ITEMS = 250
TRANCHE_DROP_PERCENT = 3
#: plan4 §3.6 "Top-up planner": a top-up round is a `genRound` after the last, its first draw
#: plain and each further stratum an `extend`.
#: plan4 §3.6 "Spend halt": the projection halts when it exceeds half of round 1's tokens.
SPEND_HALT_DENOMINATOR = 2
#: plan4 §3.5: `full` needs the previous point's campaign finalized.
FULL_ORDER = ("guard-judge", "message-type-classifier", "join-in-judge")
#: plan5 §3.6's spend gate states.
PASSING_CUE_STATES = ("passed", "recheck-passed")
#: The pilot's two kinds of block (a `round-start`'s `pilotBlock`): a main block takes votes 1 to 3
#: and the screen vote (note §Q1.3 "Pilot" "Screen"); the recheck block (plan5 §3.6) takes
#: votes 1 to 3 only. `pilot.py` owns the numbers of each block.
MAIN_BLOCK = "main"
RECHECK_BLOCK = "recheck"
PILOT_BLOCKS = (MAIN_BLOCK, RECHECK_BLOCK)

ROLE_OF_KIND = {"generate": "generator", "label": "labeler"}
_CLEARING_EVENTS = ("resolved", "generation-restart", "labelling-restart", "tranche-restart",
                    "topup-declined", "probe-requested", "point-relabel", "recheck-accepted")
#: A batch's `retired_by`: what retired it. Only a labelling or tranche relabel's retirements
#: count in round 1's total spend (plan6 §3.8 "The spend projection"); a generation restart
#: discards the round, so its retirements belong to no round.
RETIRED_SPEND_CAUSES = ("labelling-restart", "tranche-restart")
#: plan6 §3.8 "The probe": one generation run of the halted cell's design stratum, and the z of
#: the Wilson bound it prints (95%).
PROBE_SEEDS = 25
WILSON_Z = 1.96
_NOT_PRODUCED = frozenset({"launchFailed", "launchFailing", "neverSpawned"})
_NO_ATTEMPT = frozenset({"neverSpawned", "incomplete"})
_EXIT_CODE = re.compile(r"-?[0-9]+")

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
PILOT_REPORT = Path("guard-judge") / "pilot" / "pilot-report.json"


class CampaignRefused(Exception):
    """A refused command or call; nothing was written."""


class UnboundedCell(CampaignRefused):
    """`topup_counts`' refusal for a short cell with no labelled item (P̂ = 0): the top-up draw is
    unbounded. It carries the cell, so `queue_topup` can record the `unboundedCell` halt."""

    def __init__(self, label: str, stratum: str | None) -> None:
        self.label, self.stratum = label, stratum
        super().__init__(
            f"cell ({label}, {stratum or 'no stratum'}) holds no labelled item, so "
            "P(label | stratum) is 0 and its top-up draw is unbounded: a method decision")


# --- the folded state ----------------------------------------------------------------------------


@dataclass
class Attempt:
    batch: str
    opaque: str
    session_id: str
    attempt: int
    issued_at: datetime
    epoch: int
    keys: Mapping[str, str]
    request_sha256: str
    interrupted: bool = False
    outcome: dict[str, Any] | None = None
    #: The `expectations` event in force when this attempt was `issued` (plan6 §3.8): the audit
    #: reads the template, agent type and effort from it, never from a later `install`.
    expectations: Mapping[str, Any] | None = None
    #: The model pins and instructions in force at `issued` (plan6 §3.8): a later `resolve
    #: --repin-*` applies to attempts issued after it, never to this one.
    pins: Mapping[str, str] = field(default_factory=dict)
    instructions: Mapping[str, Sequence[Sequence[str]]] = field(default_factory=dict)


@dataclass
class Batch:
    batch_id: str
    kind: str
    point: str
    entries: list[str]
    extra: dict[str, Any]
    state: str = "pending"
    attempts: list[Attempt] = field(default_factory=list)
    counted: int = 0
    dropped: str | None = None
    #: What retired it (`retired`'s cause): `generation-restart`, `labelling-restart` or
    #: `tranche-restart`; `None` while live.
    retired_by: str | None = None

    @property
    def open_attempt(self) -> Attempt | None:
        last = self.attempts[-1] if self.attempts else None
        return last if last is not None and last.outcome is None else None

    @property
    def gen_run_id(self) -> str:
        """plan4 §3.6's `genRunId`: unique because a batch id is, and it is the batch's, not
        copied by a cut (`extra` is), so a cut's overflow is a run of its own."""
        return f"gen_{self.batch_id.removeprefix('bat_')}"


@dataclass
class State:
    init: dict[str, Any] | None = None
    expectations: dict[str, Any] | None = None
    pins: dict[str, str] = field(default_factory=dict)
    instructions: dict[str, list[list[str]]] = field(default_factory=dict)
    held_rng_seeds: set[int] = field(default_factory=set)
    finalized_versions: set[str] = field(default_factory=set)
    seeds: dict[str, Seed] = field(default_factory=dict)
    batches: dict[str, Batch] = field(default_factory=dict)
    attempts: list[Attempt] = field(default_factory=list)
    opaques: dict[str, Attempt] = field(default_factory=dict)
    held_keys: set[str] = field(default_factory=set)
    halt: dict[str, Any] | None = None
    epoch: int = 0
    #: Generation (plan4 §3.6, plan5 §3.5, plan6 §3.8): the drawn rounds (`genRound`, seed ids)
    #: in start order, each seed's `genRound`, the items accepted by seed, every hit by seed in
    #: order, the seeds dropped on a third hit (by that hit), the first-pass hits per (`genRound`,
    #: reason), the halts fired, and the events of an attempt not yet committed by its outcome.
    rounds: list[dict[str, Any]] = field(default_factory=list)
    seed_round: dict[str, int] = field(default_factory=dict)  # seed id -> its `genRound`
    items: dict[str, dict[str, Any]] = field(default_factory=dict)
    hits: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    dropped_seeds: dict[str, dict[str, Any]] = field(default_factory=dict)
    round_hits: dict[tuple[int, str], int] = field(default_factory=dict)  # (genRound, reason)
    template_halts: set[tuple[int, str]] = field(default_factory=set)
    #: plan6 §3.8's pooled halt, per generator-template hash: the small rounds pooled (their
    #: `genRound`s, drawn seeds and first-pass hits by reason), the rounds already pooled, and the
    #: (hash, reason) pairs already fired.
    pool: dict[str, dict[str, Any]] = field(default_factory=dict)
    pooled_rounds: set[int] = field(default_factory=set)
    pool_halts: set[tuple[str, str]] = field(default_factory=set)
    #: A generation run's generator-template hash at its issue, set when it ingests (only a
    #: first-pass run feeds a pool), and the template halts raised, in order (diagnostics).
    run_hash: dict[str, str] = field(default_factory=dict)
    template_log: list[dict[str, Any]] = field(default_factory=list)
    #: Votes (plan4 §3.6, plan6 §3.8): each item's answers by vote number, and each (`genRound`,
    #: vote)'s recorded `vote-start` (its reshuffle RNG seed and item order), in start order.
    votes: dict[str, dict[int, Any]] = field(default_factory=dict)
    vote_starts: dict[tuple[int, int], dict[str, Any]] = field(default_factory=dict)
    staged: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: The pilot's `cue-check` halt (plan5 §3.6): raised by `pilot.py report`, and whether a
    #: `resolve` has cleared it (the precondition of `pilot.py recheck`).
    cue_halts: int = 0
    cue_halt_resolved: bool = False
    #: plan6 §3.8's halt epoch for the pilot's added blocks: the number of main blocks when the
    #: last `pilot-block-halt` was recorded (0 before any), so the blocks since are the epoch's.
    block_halt_base: int = 0
    #: plan4 §3.6's message-type tranche: its `genRound`, drawn items, the rubric hash it is
    #: labelled under, whether its rule was checked and passed, and its relabellings; its votes'
    #: starts by vote number (the round's own `vote_starts` hold the rest of the round).
    tranche: dict[str, Any] | None = None
    tranche_starts: dict[int, dict[str, Any]] = field(default_factory=dict)
    #: plan4 §3.6's top-up: the `topup-plan` events (each names its `genRound` and strata), the
    #: rounds a stakeholder let through a spend halt, and whether a `--decline-topup` was recorded.
    topup_plans: list[dict[str, Any]] = field(default_factory=list)
    spend_approved: set[int] = field(default_factory=set)
    topup_declined: bool = False
    #: plan6 §3.8 "The probe": a `probe-requested` event's `designStratum`, kept (as
    #: `{"designStratum": ...}`, since a point with no stratum has `None`) until a `topup-plan`,
    #: a `topup-declined` or a generation restart.
    probe: dict[str, Any] | None = None
    #: plan4 §3.6's whole-point recheck at `finalize` (message type): how many times it has
    #: failed (`recheck-halt` events since the last generator restart), whether a stakeholder
    #: accepted the second failure (`recheck-accepted`), and the `epoch` each cleared template
    #: halt left behind (`finalize`'s one-hash rule compares it with an attempt's `epoch`).
    recheck_failures: int = 0
    recheck_accepted: bool = False
    template_resolutions: list[int] = field(default_factory=list)

    def generation_drops(self) -> dict[str, Any]:
        """plan5 §3.5's `dropped` keys for seeds dropped on a third hit, counted under the third
        hit's reason: `screenHits`, `formMismatch` by form, `bindingMismatch`."""
        counts: dict[str, Any] = {"screenHits": 0, "formMismatch": {}, "bindingMismatch": 0}
        for hit in self.dropped_seeds.values():
            reason = hit["reasons"][0]
            if reason == screen.SCREEN:
                counts["screenHits"] += 1
            elif reason == screen.BINDING_MISMATCH:
                counts["bindingMismatch"] += 1
            else:
                counts["formMismatch"][hit["form"]] = counts["formMismatch"].get(hit["form"], 0) + 1
        return counts

    def awaiting_regeneration(self) -> list[str]:
        """The seeds a hit queued: hit, with no accepted item and fewer than three hits, whose
        latest generation batch is ingested (not still pending, issued, halted or dropped), in
        order of first hit."""
        latest: dict[str, Batch] = {}
        for batch in self.batches.values():
            if batch.kind == "generate" and batch.state != "retired":
                latest.update(dict.fromkeys(batch.entries, batch))
        return [seed for seed in self.hits if seed not in self.items
                and seed not in self.dropped_seeds and seed in latest
                and latest[seed].state == "ingested"]

    def drawn_in(self, gen_round: int) -> int:
        """plan6 §3.8: a round's halt denominator, every seed of every `round-start` that
        carries its `genRound`."""
        return sum(len(r["seeds"]) for r in self.rounds if r["genRound"] == gen_round)

    def first_pass_runs(self, gen_round: int) -> list[Batch]:
        """The first-pass generation batches holding a seed of `gen_round`."""
        return [batch for batch in self.batches.values()
                if batch.kind == "generate" and batch.extra.get("generation") == "first"
                and batch.state != "retired"
                and any(self.seed_round.get(seed) == gen_round for seed in batch.entries)]

    def round_closed(self, gen_round: int) -> bool:
        """plan6 §3.8: every first-pass run of `gen_round` is ingested or dropped, and no seed
        drawn under it lacks a run. A dropped run does not hold the round open."""
        runs = self.first_pass_runs(gen_round)
        held = {seed for batch in runs for seed in batch.entries}
        wanted = {seed for r in self.rounds if r["genRound"] == gen_round for seed in r["seeds"]}
        return bool(runs) and wanted <= held and all(
            batch.state in ("ingested", "dropped") for batch in runs)

    def round_settled(self, gen_round: int) -> bool:
        """plan6 §3.8 "Votes": every seed drawn under `gen_round` is accepted, dropped on a third
        hit, or in a generation batch that dropped; none awaits regeneration."""
        wanted = {seed for r in self.rounds if r["genRound"] == gen_round for seed in r["seeds"]}
        given_up = {seed for batch in self.batches.values()
                    if batch.kind == "generate" and batch.state == "dropped"
                    for seed in batch.entries}
        return bool(wanted) and all(
            seed in self.items or seed in self.dropped_seeds or seed in given_up
            for seed in wanted)

    def vote_batches(self, gen_round: int, vote: int, *, tranche: bool = False) -> list[Batch]:
        """The live label batches of (`gen_round`, `vote`), cut overflow included; `tranche`
        selects the message-type tranche's batches (`queue_votes` marks them) or the rest's."""
        return [batch for batch in self.batches.values()
                if batch.kind == "label" and batch.state != "retired"
                and batch.extra.get("genRound") == gen_round and batch.extra.get("vote") == vote
                and bool(batch.extra.get("tranche")) is tranche]

    def tranche_applies(self, gen_round: int) -> bool:
        """plan4 §3.6: message type's round 1, in a `full` campaign, labels a tranche first."""
        return (self.init is not None and gen_round == 1 and self.init["phase"] == "full"
                and self.init["point"] == TRANCHE_POINT)

    def tranche_due(self, gen_round: int) -> bool:
        """The round is settled and its tranche is not yet drawn."""
        return (self.tranche_applies(gen_round) and self.tranche is None
                and self.round_settled(gen_round)
                and any(rec["genRound"] == gen_round for rec in self.items.values()))

    def tranche_items(self) -> list[str]:
        return list(self.tranche["items"]) if self.tranche else []

    def tranche_drops(self) -> list[str]:
        """The tranche's `voteTie` drops: its items with three votes and no majority."""
        ties = set(self.vote_ties())
        return [seed for seed in self.tranche_items() if seed in ties]

    def tranche_complete(self) -> bool:
        """All three votes of the tranche have run to their end (`_stage_done`)."""
        return self.tranche is not None and self._stage_done(
            self.tranche["genRound"], self.tranche_items(), tranche=True)

    def tranche_passed(self) -> bool:
        return bool(self.tranche and self.tranche["passed"])

    def _stage_items(self, gen_round: int, *, tranche: bool) -> list[str]:
        """The items a stage votes: the tranche's, or the round's accepted items less the
        tranche's (all of them for a round with no tranche), in id order."""
        held = set(self.tranche_items()) if self.tranche_applies(gen_round) else set()
        accepted = sorted(seed for seed, rec in self.items.items() if rec["genRound"] == gen_round)
        return sorted(held) if tranche else [seed for seed in accepted if seed not in held]

    def _stage_done(self, gen_round: int, items: Sequence[str], *, tranche: bool) -> bool:
        """Every vote of the stage (the three, for the tranche) is recorded, its batches all
        ingested or dropped, and every item it names (less a dropped batch's) in one. A stage
        with no item has no vote to take."""
        if not items:
            return True
        out = self._dropped_voters()
        votes = (tuple(range(1, labels.VOTES_PER_ITEM + 1)) if tranche
                 else self.round_votes(gen_round))
        for vote in votes:
            start = (self.tranche_starts if tranche else self.vote_starts).get(
                vote if tranche else (gen_round, vote))
            batches = self.vote_batches(gen_round, vote, tranche=tranche)
            if start is None or any(b.state not in ("ingested", "dropped") for b in batches):
                return False
            held = {seed for batch in batches for seed in batch.entries}
            if any(seed not in out and seed not in held for seed in start["items"]):
                return False
        return True

    def round_votes(self, gen_round: int) -> tuple[int, ...]:
        """The votes `gen_round` takes: the three, and the screen vote after them for a main
        pilot block (`MAIN_BLOCK`)."""
        votes = tuple(range(1, labels.VOTES_PER_ITEM + 1))
        screened = any(r["genRound"] == gen_round and r["pilotBlock"] == MAIN_BLOCK
                       for r in self.rounds)
        return votes + (labels.SCREEN_VOTE,) if screened else votes

    def _dropped_voters(self) -> set[str]:
        """The items of a dropped label batch: out of the later votes of their round."""
        return {seed for batch in self.batches.values()
                if batch.kind == "label" and batch.state == "dropped" for seed in batch.entries}

    def round_voted(self, gen_round: int) -> bool:
        """Every vote `gen_round` takes has run to its end: recorded, its batches all ingested or
        dropped, and every item it names (less a dropped batch's) in one. A settled round with no
        accepted item has no vote to take. A round with a tranche is voted only once the tranche
        has passed its rule, then its rest."""
        if not self.round_settled(gen_round):
            return False
        if not any(rec["genRound"] == gen_round for rec in self.items.values()):
            return True
        if self.tranche_applies(gen_round) and not (self.tranche_complete()
                                                    and self.tranche_passed()):
            return False
        return self._stage_done(gen_round, self._stage_items(gen_round, tranche=False),
                                tranche=False)

    def next_vote(self, gen_round: int) -> tuple[int, list[str], bool] | None:
        """`vote_step` less its tranche flag (the public shape of a round with no tranche)."""
        step = self.vote_step(gen_round)
        return None if step is None else step[:3]

    def vote_step(self, gen_round: int) -> tuple[int, list[str], bool, bool] | None:
        """The vote `gen_round` starts or finishes queuing next, as (vote, the items still to be
        queued into it, whether its `vote-start` is already recorded, whether it is the
        tranche's), or `None` when the round is not settled, a vote is still running, or all are
        done. A recorded vote is incomplete until its live batches hold every item its
        `vote-start` names (a crash between its `queued` events leaves a tail). An item whose
        label batch dropped is out. A round with a tranche votes it first, and its rest only once
        the tranche has been checked and passed."""
        if not self.round_settled(gen_round):
            return None
        if self.tranche_applies(gen_round):
            if self.tranche is None:
                return None
            step = self._vote_step(gen_round, tranche=True)
            if step is not None or not self.tranche_passed():
                return step
        return self._vote_step(gen_round, tranche=False)

    def _vote_step(self, gen_round: int, *, tranche: bool
                   ) -> tuple[int, list[str], bool, bool] | None:
        out = self._dropped_voters()
        votes = (tuple(range(1, labels.VOTES_PER_ITEM + 1)) if tranche
                 else self.round_votes(gen_round))
        for vote in votes:
            batches = self.vote_batches(gen_round, vote, tranche=tranche)
            if any(batch.state not in ("ingested", "dropped") for batch in batches):
                return None
            key = vote if tranche else (gen_round, vote)
            starts = self.tranche_starts if tranche else self.vote_starts
            recorded = key in starts
            items = (starts[key]["items"] if recorded
                     else self._stage_items(gen_round, tranche=tranche))
            held = {seed for batch in batches for seed in batch.entries}
            todo = [seed for seed in items if seed not in out and seed not in held]
            if todo or not batches:
                return vote, todo, recorded, tranche
        return None

    def label_votes(self, seed_id: str) -> dict[str, int] | None:
        """`labelVotes` of an item with all three votes in, else `None`."""
        answers = self.three_votes(seed_id)
        if answers is None:
            return None
        return labels.label_votes(load_point(self.init["point"]).expected, answers)

    def screen_votes(self, rounds: Sequence[int]) -> dict[str, dict[int, Any]]:
        """The votes of the accepted items of `rounds`, by seed id (`pilot.screen`'s input)."""
        return {seed: self.votes.get(seed, {}) for seed, rec in self.items.items()
                if rec["genRound"] in set(rounds)}

    def three_votes(self, seed_id: str) -> list[Any] | None:
        """An item's votes 1 to 3 in vote order, or `None` while one is missing: the screen vote
        is never one of them."""
        answers = self.votes.get(seed_id, {})
        wanted = range(1, labels.VOTES_PER_ITEM + 1)
        return [answers[v] for v in wanted] if all(v in answers for v in wanted) else None

    def vote_ties(self) -> list[str]:
        """The items with three votes and no majority (`voteTie`): a choice item with three
        distinct labels."""
        expected = load_point(self.init["point"]).expected
        return [seed for seed in self.votes
                if (three := self.three_votes(seed)) is not None
                and labels.majority(expected, three) is None]

    def reshuffle_seeds(self) -> list[int]:
        """The RNG seeds of the votes' reshuffles, in start order (the pilot report lists them)."""
        return [start["rngSeed"] for start in
                (*self.tranche_starts.values(), *self.vote_starts.values())]

    def unbatched_seeds(self) -> list[str]:
        """The `round-start` seeds no live generation batch holds (a crash between the round and
        its `queued` events), in round order. A top-up round that drew several strata
        (plan6 §3.8) interleaves them, so that its balanced cut pools them: each run holds seeds
        of every stratum while any is left."""
        held = {seed for batch in self.batches.values() if batch.kind == "generate"
                for seed in batch.entries}
        draws: dict[int, list[list[str]]] = {}
        for r in self.rounds:
            draws.setdefault(r["genRound"], []).append(
                [seed for seed in r["seeds"] if seed not in held])
        return [seed for lists in draws.values()
                for group in itertools.zip_longest(*lists) for seed in group if seed is not None]

    def labelled_cells(self) -> list[tuple[str | None, str]]:
        """plan4 §3.6 "Top-up planner": each labelled item's cell, as (design stratum, label):
        an accepted item with three votes and a majority. The stratum is its seed's, `None` for a
        point with none."""
        expected = load_point(self.init["point"]).expected
        out = []
        for seed_id in self.items:
            votes = self.three_votes(seed_id)
            majority = labels.majority(expected, votes) if votes is not None else None
            if majority is not None:
                out.append((self.seeds[seed_id].design_stratum, labels.vote_key(majority)))
        return out

    def spend(self, gen_round: int) -> dict[str, Any]:
        """The tokens spent on `gen_round` and what they bought, from every attempt that recorded
        a run (void ones spent them too): `items` its accepted items, `seeds` its drawn seeds,
        `generationRuns` and `generationTokens` its live generation batches (a regeneration batch
        counts under its first seed's round), and `labelTokens` its live label batches, four
        classes each (`pilot.TOKEN_CLASSES`). `totalTokens` is plan6 §3.8's round total: the live
        tokens plus those of label batches a `labelling-restart` or `tranche-restart` retired
        (`RETIRED_SPEND_CAUSES`), since the halt bound compares a top-up with what round 1 cost; a
        generation restart's retirements are no round's. The live figures price the top-up, as a
        retired attempt does not recur."""
        classes = pilot.TOKEN_CLASSES
        gen = dict.fromkeys(classes, 0)
        lab = dict.fromkeys(classes, 0)
        retired = 0
        runs = 0
        for batch in self.batches.values():
            if batch.state == "retired" and batch.retired_by not in RETIRED_SPEND_CAUSES:
                continue
            if batch.kind == "generate":
                if self.seed_round.get(batch.entries[0] if batch.entries else "") != gen_round:
                    continue
                target = gen
            elif batch.extra.get("genRound") == gen_round:
                target = lab
            else:
                continue
            for attempt in batch.attempts:
                tokens = ((attempt.outcome or {}).get("run") or {}).get("tokens")
                if tokens is None:
                    continue
                if batch.state == "retired":
                    retired += sum(tokens.get(cls, 0) for cls in classes)
                    continue
                runs += batch.kind == "generate"
                for cls in classes:
                    target[cls] += tokens.get(cls, 0)
        items = sum(1 for rec in self.items.values() if rec["genRound"] == gen_round)
        return {"items": items, "seeds": self.drawn_in(gen_round), "generationRuns": runs,
                "generationTokens": gen, "labelTokens": lab,
                "totalTokens": sum(gen.values()) + sum(lab.values()) + retired}

    def dropped(self) -> dict[str, dict[str, int]]:
        """plan4 §3.7's `retryCapExceeded` and plan6 §3.5's `safeguardSkipped` counts."""
        counts = {reason: {"seeds": 0, "items": 0}
                  for reason in ("retryCapExceeded", "safeguardSkipped")}
        for batch in self.batches.values():
            if batch.dropped is not None:
                unit = "seeds" if batch.kind == "generate" else "items"
                counts[batch.dropped][unit] += len(batch.entries)
        return counts


def _parse_at(text: str) -> datetime:
    return datetime.fromisoformat(text)


def fold(events: Iterable[Mapping[str, Any]]) -> State:
    """The campaign state the ledger's events give (module docstring)."""
    state = State()
    for event in events:
        kind = event["event"]
        if kind == "init":
            state.init = dict(event)
        elif kind == "expectations":
            state.expectations = dict(event)
            for role, spec in event["roles"].items():
                # Only the first event sets the pins: a later `install` accepts a settings edit
                # and must not undo a `resolve --repin-model/--repin-instructions`.
                state.pins.setdefault(role, spec["model"])
                state.instructions.setdefault(
                    role, [list(pair) for pair in spec.get("instructions", [])])
        elif kind == "held-rng-seeds":
            state.held_rng_seeds.update(event["rngSeeds"])
        elif kind == "seed-draw":
            draw = event["draw"]
            state.held_rng_seeds.add(draw["rngSeed"])
            for record in draw["seeds"]:
                state.seeds[record["seedId"]] = Seed(
                    seed_id=record["seedId"], role=record["role"],
                    design_stratum=record["designStratum"], axes=record["axes"],
                )
        elif kind == "round-start":
            for seed_id in event["seeds"]:
                state.seed_round[seed_id] = event["genRound"]
            state.rounds.append({"genRound": event["genRound"], "seeds": list(event["seeds"]),
                                 "designStratum": event.get("designStratum"),
                                 "pilotBlock": event.get("pilotBlock")})
        elif kind == "vote-start":
            state.held_rng_seeds.add(event["rngSeed"])
            start = {"rngSeed": event["rngSeed"], "items": list(event["items"])}
            if event.get("tranche"):
                state.tranche_starts[event["vote"]] = start
            else:
                state.vote_starts[(event["genRound"], event["vote"])] = start
        elif kind == "tranche-start":
            state.held_rng_seeds.add(event["rngSeed"])
            state.tranche = {"genRound": event["genRound"], "rngSeed": event["rngSeed"],
                             "items": list(event["items"]), "rubric": event["rubric"],
                             "checked": False, "passed": False, "relabels": 0}
        elif kind == "topup-plan":
            state.topup_plans.append(dict(event))
            state.probe = None
        elif kind == "spend-halt":
            state.halt = {"event": kind, "batch": None, "reason": "spend",
                          "voidClass": "spend", "genRound": event["genRound"],
                          "projection": event["projection"], "probe": bool(event.get("probe"))}
        elif kind == "unbounded-halt":
            state.halt = {"event": kind, "batch": None, "reason": "unboundedCell",
                          "voidClass": "unboundedCell", "label": event["label"],
                          "stratum": event["stratum"], "items": event["items"],
                          "bound": event["bound"]}
        elif kind == "topup-reopened":
            state.topup_declined = False
        elif kind == "recheck-halt":
            state.recheck_failures += 1
            state.halt = {"event": kind, "batch": None, "reason": "recheck",
                          "voidClass": "rubric", "drops": event["drops"],
                          "items": event["items"], "rubric": event["rubric"],
                          "second": state.recheck_failures >= 2}
        elif kind == "finalized":
            state.held_rng_seeds.add(event["rngSeed"])
            state.finalized_versions.add(event["datasetVersion"])
        elif kind == "votes":
            # Fresh, as for `generated`: a crashed ingest re-run replaces its earlier record.
            state.staged[event["opaque"]] = {"votes": dict(event["answers"]),
                                             "vote": event["vote"]}
        elif kind == "generated":
            # A fresh entry: a crashed ingest re-run may screen differently, and its earlier hits
            # must not survive beside the new items.
            state.staged[event["opaque"]] = {"items": [dict(r) for r in event["items"]],
                                             "hits": {}}
        elif kind in _HIT_EVENT_KINDS:
            _staged(state, event["opaque"])["hits"][event["seedId"]] = dict(event)
        elif kind == "queued":
            extra = {k: v for k, v in event.items()
                     if k not in ("event", "at", "batch", "kind", "point", "entries")}
            state.batches[event["batch"]] = Batch(event["batch"], event["kind"], event["point"],
                                                  list(event["entries"]), extra)
        elif kind == "cut":
            batch = state.batches[event["batch"]]
            rest = [ref for ref in batch.entries if ref not in set(event["keep"])]
            batch.entries = list(event["keep"])
            state.batches[event["overflow"]] = Batch(event["overflow"], batch.kind, batch.point,
                                                     rest, dict(batch.extra))
        elif kind == "issued":
            batch = state.batches[event["batch"]]
            attempt = Attempt(
                batch=batch.batch_id, opaque=event["opaque"], session_id=event["sessionId"],
                attempt=event["attempt"], issued_at=_parse_at(event["at"]), epoch=state.epoch,
                keys=event["keys"], request_sha256=event["requestSha256"],
                expectations=state.expectations, pins=dict(state.pins),
                instructions={role: [list(pair) for pair in pairs]
                              for role, pairs in state.instructions.items()},
            )
            batch.attempts.append(attempt)
            batch.state = "issued"
            state.attempts.append(attempt)
            state.opaques[attempt.opaque] = attempt
            state.held_keys.update(event["keys"])
        elif kind == "interrupted":
            attempt = state.batches[event["batch"]].open_attempt
            if attempt is not None:
                attempt.interrupted = True
        elif kind == "outcome":
            _fold_outcome(state, event)
        elif kind == "cue-check-halt":
            state.cue_halts += 1
            state.cue_halt_resolved = False
            state.halt = {"event": kind, "batch": None, "reason": "cueCheck",
                          "voidClass": "cueCheck"}
        elif kind == "pilot-block-halt":
            state.block_halt_base = event["mainRounds"]
            state.halt = {"event": kind, "batch": None, "reason": "pilotBlocks",
                          "voidClass": "pilotBlocks"}
        elif kind in _CLEARING_EVENTS:
            _fold_resolution(state, event)
    return state


def _staged(state: State, opaque: str) -> dict[str, Any]:
    return state.staged.setdefault(opaque, {"items": [], "hits": {}})


def generator_template_hash(expectations: Mapping[str, Any] | None) -> str:
    """plan6 §3.8: sha256 of the canonical JSON of the generator role's `agentDefinition`,
    `spawnTemplate` and `requestTemplate` entries in an `expectations` event's `promptSha256`."""
    prompts = (expectations or {}).get("promptSha256") or {}
    parts = {key: (prompts.get(key) or {}).get("generator")
             for key in ("agentDefinition", "spawnTemplate", "requestTemplate")}
    text = json.dumps(parts, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _over_threshold(hits: int, drawn: int) -> bool:
    """plan6 §3.8: more than `HALT_PERCENT` of the seeds, and at least two hits."""
    return hits >= HALT_MIN_HITS and 100 * hits > HALT_PERCENT * drawn


def _pool_round(state: State, gen_round: int) -> list[tuple[str, str]]:
    """plan6 §3.8's pooled halt: `gen_round`, closed and under 25 drawn seeds, adds each ingested
    first-pass run's seeds and hits (by reason) to the pool of that run's own template hash; a
    dropped run adds nothing. Each pool fed that holds 300 seeds is then tested. Returns the
    (hash, reason) pairs newly over the threshold."""
    state.pooled_rounds.add(gen_round)
    fed: dict[str, dict[str, Any]] = {}
    for batch in state.first_pass_runs(gen_round):
        digest = state.run_hash.get(batch.batch_id)
        if batch.state != "ingested" or digest is None:
            continue
        pool = fed.setdefault(digest, {"seeds": 0, "hits": dict.fromkeys(screen.REASONS, 0)})
        pool["seeds"] += len(batch.entries)
        for seed in batch.entries:
            for hit in state.hits.get(seed, ()):
                if hit["firstPass"]:
                    for reason in hit["reasons"]:
                        pool["hits"][reason] += 1
    over: list[tuple[str, str]] = []
    for digest, added in fed.items():
        pool = state.pool.setdefault(digest, {"rounds": [], "seeds": 0, "hits": {}})
        pool["rounds"].append(gen_round)
        pool["seeds"] += added["seeds"]
        for reason in screen.REASONS:
            pool["hits"][reason] = pool["hits"].get(reason, 0) + added["hits"][reason]
        if pool["seeds"] >= POOL_MIN_SEEDS:
            over += [(digest, reason) for reason in screen.REASONS
                     if (digest, reason) not in state.pool_halts
                     and pool["hits"][reason] >= HALT_MIN_HITS
                     and 100 * pool["hits"][reason] > POOL_PERCENT * pool["seeds"]]
    return over


def _close_rounds(state: State, batch: Batch) -> tuple[int | None, list[tuple[str, str]]]:
    """After a first-pass run of `batch` is ingested or dropped: each round it belongs to that is
    now closed, under 25 drawn seeds and not yet pooled is pooled (`_pool_round`). Returns the
    last round closed and the pairs over the pooled threshold."""
    closed: int | None = None
    over: list[tuple[str, str]] = []
    if batch.kind == "generate" and batch.extra.get("generation") == "first":
        for g in sorted({state.seed_round[seed] for seed in batch.entries
                         if seed in state.seed_round}):
            if (g not in state.pooled_rounds and state.drawn_in(g) < POOL_BELOW
                    and state.round_closed(g)):
                closed = g
                over += _pool_round(state, g)
    return closed, over


def _raise_template_halt(state: State, batch: Batch, digest: str,
                         rounds: list[tuple[int, str]], closed: int | None,
                         pooled: list[tuple[str, str]]) -> None:
    """The one `template` halt for the pairs a run or a drop newly crossed (plan6 §3.8). Its
    `round` part holds the per-round pairs (`genRound`, `reasons`), absent if none; its `pools`
    part one entry per pool that crossed, in hash order (`templateHash`, `reasons`, `poolRounds`,
    `poolSeeds`, `poolHits`). Every pair is marked fired. `reason` is the first of the round's
    reasons, else the first pool's; `genRound` is the closing round's."""
    if not (rounds or pooled) or state.halt is not None:
        return
    state.template_halts.update(rounds)
    state.pool_halts.update(pooled)
    reasons = [r for r in screen.REASONS if r in {x[1] for x in rounds + pooled}]
    state.halt = {"event": "template-halt", "batch": batch.batch_id, "voidClass": "template",
                  "genRound": rounds[0][0] if rounds else closed, "reasons": reasons,
                  "pooled": bool(pooled)}
    round_reasons = [r for r in screen.REASONS if r in {x[1] for x in rounds}]
    if rounds:
        state.halt["round"] = {"genRound": rounds[0][0], "reasons": round_reasons}
    pools = []
    for pool_digest in sorted({d for d, _ in pooled}):
        pool = state.pool[pool_digest]
        pools.append({"templateHash": pool_digest,
                      "reasons": [r for r in screen.REASONS if (pool_digest, r) in pooled],
                      "poolRounds": list(pool["rounds"]), "poolSeeds": pool["seeds"],
                      "poolHits": dict(pool["hits"])})
    if pools:
        state.halt["pools"] = pools
    state.halt["reason"] = round_reasons[0] if rounds else pools[0]["reasons"][0]
    state.halt["templateHash"] = pools[0]["templateHash"] if pools else digest
    state.template_log.append(dict(state.halt))


def _apply_generation(state: State, attempt: Attempt, batch: Batch,
                      staged: Mapping[str, Any]) -> None:
    """The ingested generation run's items and hits, then plan6 §3.8's halts: after each run, one
    `template` halt per (`genRound`, reason) over that round's drawn seeds; and, when a first-pass
    round under 25 seeds closes, the pooled halt per (generator-template hash, reason).
    Regeneration hits count toward the 3-hit cap only."""
    for record in staged["items"]:
        state.items[record["seedId"]] = record
    touched: set[int] = set()
    for seed_id, hit in staged["hits"].items():
        history = state.hits.setdefault(seed_id, [])
        history.append(hit)
        if len(history) >= MAX_SEED_HITS:
            state.dropped_seeds[seed_id] = hit
        if hit["firstPass"]:
            gen_round = state.seed_round[seed_id]
            touched.add(gen_round)
            for reason in hit["reasons"]:
                key = (gen_round, reason)
                state.round_hits[key] = state.round_hits.get(key, 0) + 1
    rounds = [(g, reason) for g in sorted(touched) for reason in screen.REASONS
              if (g, reason) not in state.template_halts
              and _over_threshold(state.round_hits.get((g, reason), 0), state.drawn_in(g))]
    digest = generator_template_hash(attempt.expectations)
    if batch.extra.get("generation") is not None:
        state.run_hash[batch.batch_id] = digest
    closed, pooled = _close_rounds(state, batch)
    _raise_template_halt(state, batch, digest, rounds, closed, pooled)


def _restart_generation(state: State) -> None:
    """plan6 §3.8 (note §Q1.2 "Reruns"): a generator re-pin discards the generation view. Items,
    hits, drops, counts, halts, rounds and staged events go, and every generation and label batch
    of the point is `retired`: never issued again, neither dropped nor ingested. The ledger, the
    held RNG seeds and seed ids (`state.seeds`), the pins, instructions and expectations stay."""
    state.items.clear()
    state.hits.clear()
    state.dropped_seeds.clear()
    state.round_hits.clear()
    state.template_halts.clear()
    state.pool.clear()
    state.pool_halts.clear()
    state.pooled_rounds.clear()
    state.run_hash.clear()
    state.template_log.clear()
    state.block_halt_base = 0
    state.cue_halts = 0
    state.cue_halt_resolved = False
    state.votes.clear()
    state.vote_starts.clear()
    state.tranche = None
    state.tranche_starts.clear()
    state.topup_plans.clear()
    state.spend_approved.clear()
    state.topup_declined = False
    state.probe = None
    state.recheck_failures = 0
    state.recheck_accepted = False
    state.rounds.clear()
    state.seed_round.clear()
    state.staged.clear()
    for batch in state.batches.values():
        _retire(batch, "generation-restart", override=True)


def _retire(batch: Batch, cause: str, *, override: bool = False) -> None:
    """Retire `batch` for `cause` (`Batch.retired_by`). A batch already retired keeps its cause,
    unless `override`: a generation restart discards the round, so a label batch an earlier
    relabel retired is the discarded round's too, and a later relabel must not claim one a
    generation restart retired."""
    if batch.state != "retired" or override:
        batch.retired_by = cause
    batch.state, batch.dropped = "retired", None


def _settle_drop(state: State, batch: Batch) -> None:
    """A dropped first-pass run may close its round (plan6 §3.8): the round pools what its
    ingested runs added, and the pool is tested. A dropped tranche batch may complete the
    tranche."""
    closed, pooled = _close_rounds(state, batch)
    _raise_template_halt(state, batch, "", [], closed, pooled)
    _check_tranche(state, batch)


def _check_tranche(state: State, batch: Batch | None) -> None:
    """plan4 §3.6: once the tranche's three votes have run to their end, apply its rule once:
    `100 · drops > TRANCHE_DROP_PERCENT · size` halts (`rubric`, naming the drops and the rubric
    hash the tranche was labelled under), anything else passes it. A relabelled tranche is new.
    It runs where a tranche batch ends (`_fold_outcome`, `_settle_drop`), the only places the
    tranche can complete; it has no "wait while another halt is open" clause and `_fold_resolution`
    no re-check. A halt can be open at that moment only if a recorder wrote it outside a batch
    outcome (`record_cue_check_halt`, `record_pilot_block_halt`), and both are called only in a
    pilot campaign, which has no tranche (`tranche_applies` needs message type and `full`). Called
    by hand in a `full` campaign they would be overwritten by the tranche halt: not a shipped
    flow, and not guarded (P27-m2, Pass 29 P29-m1)."""
    tranche = state.tranche
    if tranche is None or tranche["checked"] or not state.tranche_complete():
        return
    tranche["checked"] = True
    drops, size = len(state.tranche_drops()), len(tranche["items"])
    if tranche_failed(drops, size):
        state.halt = {"event": "tranche-halt", "batch": batch.batch_id if batch else None,
                      "reason": "tranche", "voidClass": "rubric", "drops": drops, "items": size,
                      "rubric": tranche["rubric"]}
    else:
        tranche["passed"] = True


def tranche_failed(drops: int, size: int) -> bool:
    """plan4 §3.6: the tranche's rule, `100 · drops > 3 · size` (3 · 250 for a full tranche)."""
    return 100 * drops > TRANCHE_DROP_PERCENT * size


def _fold_outcome(state: State, event: Mapping[str, Any]) -> None:
    attempt = state.opaques[event["opaque"]]
    attempt.outcome = dict(event)
    batch = state.batches[attempt.batch]
    outcome = event["outcome"]
    staged = state.staged.pop(event["opaque"], None)
    if batch.state == "retired":      # a run reconciled after a generator re-pin: recorded only
        return
    if outcome == "ingested":
        batch.state = "ingested"
        if staged is not None and "votes" in staged:
            for seed_id, answer in staged["votes"].items():
                state.votes.setdefault(seed_id, {})[staged["vote"]] = answer
        elif staged is not None:
            _apply_generation(state, attempt, batch, staged)
    elif outcome == "void":
        batch.counted += 1
        if batch.counted >= MAX_COUNTED_ATTEMPTS:
            batch.state, batch.dropped = "dropped", "retryCapExceeded"
            _settle_drop(state, batch)
        else:
            batch.state = "pending"
    elif outcome in ("reissued", "usage-stop"):
        batch.state = "pending"
    elif outcome == "halted":
        batch.state = "halted"
        state.halt = dict(event)
    else:
        raise ValueError(f"unknown outcome {outcome!r} in the ledger")
    if batch.extra.get("tranche"):
        _check_tranche(state, batch)


def _restart_labelling(state: State) -> None:
    """plan6 §3.8 "Votes": a labeler re-pin clears the folded votes and retires every label batch
    (never issued again, neither dropped nor ingested), vote-carrying or not. The items, the
    ledger and the held RNG seeds, reshuffle seeds included, stay, so the votes repeat from
    vote 1 on fresh seeds."""
    state.votes.clear()
    state.vote_starts.clear()
    state.tranche_starts.clear()
    if state.tranche is not None:
        state.tranche.update(checked=False, passed=False)
    for batch in state.batches.values():
        if batch.kind == "label":
            _retire(batch, "labelling-restart")


def _fold_resolution(state: State, event: Mapping[str, Any]) -> None:
    halted = state.halt
    state.halt = None
    state.epoch += 1
    if halted is not None and halted.get("voidClass") == "template":
        state.template_resolutions.append(state.epoch)
    if halted is not None and halted["event"] == "cue-check-halt":
        state.cue_halt_resolved = True
    batch = state.batches.get(halted["batch"]) if halted else None
    if event.get("action") == "skip-batch" and batch is not None:
        batch.state, batch.dropped = "dropped", "safeguardSkipped"
        _settle_drop(state, batch)
    elif batch is not None and batch.state == "halted":
        batch.state = "pending"
    if event["event"] == "generation-restart":
        _restart_generation(state)
    if event["event"] == "labelling-restart":
        _restart_labelling(state)
    if event["event"] in ("generation-restart", "labelling-restart"):
        state.pins[event["role"]] = event["model"]
    if event.get("action") == "repin-instructions":
        state.instructions[event["role"]] = [list(pair) for pair in event["instructions"]]
    if event["event"] == "tranche-restart":
        _restart_tranche(state, event["rubric"])
    if event["event"] == "point-relabel":
        _restart_labelling(state)
        if state.tranche is not None:
            state.tranche.update(rubric=event["rubric"], relabels=state.tranche["relabels"] + 1)
    if event["event"] == "recheck-accepted":
        state.recheck_accepted = True
    if event["event"] == "topup-declined":
        state.topup_declined = True
        state.probe = None
    elif event["event"] == "probe-requested":
        state.probe = {"designStratum": event["designStratum"]}
    elif (event["event"] == "resolved" and halted is not None
          and halted.get("voidClass") == "spend"):
        state.spend_approved.add(halted["genRound"])


def _restart_tranche(state: State, rubric: str) -> None:
    """plan4 §3.6: a tranche whose rule failed is relabelled under the fixed rubric: its items'
    votes go, its label batches are `retired`, and it is checked afresh. The items stay."""
    tranche = state.tranche
    if tranche is None:
        return
    for seed_id in tranche["items"]:
        state.votes.pop(seed_id, None)
    state.tranche_starts.clear()
    for batch in state.batches.values():
        if batch.kind == "label" and batch.extra.get("tranche"):
            _retire(batch, "tranche-restart")
    tranche.update(rubric=rubric, checked=False, passed=False, relabels=tranche["relabels"] + 1)


# --- results -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Serve:
    """The next batch to spawn: `request` is built, not yet written (`Campaign.issue` writes it)."""

    batch: str
    kind: str
    opaque: str
    agent_type: str
    spawn_prompt: str
    request: exchange.BuiltRequest


@dataclass(frozen=True)
class Stop:
    """`next` has nothing to serve: `reason` is `paused`, `halted`, `usage-stop` or `done`."""

    reason: str
    detail: str = ""


@dataclass(frozen=True)
class IngestResult:
    """One `ingest` call. `outcome` is the event appended, or `None` for a re-ingest (nothing
    appended); `stop` is `halted` or `usage-stop` when the runner must stop after it."""

    opaque: str
    batch: str
    outcome: Mapping[str, Any] | None
    stop: str | None


# --- small file helpers --------------------------------------------------------------------------


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    with open(partial, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)
    _fsync_dir(path.parent)


def _now() -> datetime:
    return datetime.now(UTC)


def parse_duration(text: str) -> timedelta:
    """`<int>` seconds, or `<int>s|m|h`: `45m`, `6h`. Positive only."""
    units = {"s": 1, "m": 60, "h": 3600}
    number, unit = (text[:-1], text[-1]) if text and text[-1] in units else (text, "s")
    if not number.isdigit() or int(number) < 1:
        raise ValueError(f"not a positive duration: {text!r}")
    return timedelta(seconds=int(number) * units[unit])


def transcript_slug(exchange_dir: Path) -> str:
    """The project folder Claude Code names for a working directory: every non-alphanumeric
    character of its path becomes `-` (probe report §9.5, H5)."""
    return "".join(c if c.isascii() and c.isalnum() else "-" for c in str(exchange_dir))


# --- init refusals -------------------------------------------------------------------------------


def root_refusal(root: Path) -> str | None:
    """Why `root` cannot hold a campaign (plan4 §3.5): inside a git work tree, or under any
    ancestor (itself included) holding `CLAUDE.md` or `AGENTS.md`."""
    for folder in (root, *root.parents):
        if (folder / ".git").exists():
            return f"{root} is inside a git work tree ({folder})"
        for name in ("CLAUDE.md", "AGENTS.md"):
            if (folder / name).exists():
                return f"{root} is under {folder / name}"
    return None


def spend_gate_refusal(report: Mapping[str, Any]) -> str | None:
    """plan5 §3.6's one predicate for `init --phase full`, or why it fails."""
    decision = report.get("stakeholderDecision")
    if not isinstance(decision, dict):
        return "the pilot report has no recorded stakeholderDecision"
    if decision.get("outcome") != "proceed":
        return f"the stakeholder's decision is {decision.get('outcome')!r}, not 'proceed'"
    cue = report.get("missingCueCheck")
    state = cue.get("state") if isinstance(cue, dict) else None
    if state in PASSING_CUE_STATES:
        return None
    ref = decision.get("stakeholderRef")
    if state == "recheck-failed" and isinstance(ref, str) and ref.strip():
        return None
    if state == "recheck-failed":
        return "missingCueCheck is recheck-failed and the decision has no stakeholderRef"
    return f"missingCueCheck state {state!r} does not pass the spend gate"


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def pilot_report_refusal(report: Mapping[str, Any]) -> str | None:
    """Why a report that passes the spend gate still cannot start `full`: `init` reads
    `stakeholderDecision.cap` (25 or 50, plan4 §3.8) and `rngSeeds` (a non-empty list of ints, plan5
    §3.2; a pilot always draws) with no defaults, so a misnamed key refuses rather than silently
    giving cap 50 or no held seeds."""
    cap = report["stakeholderDecision"].get("cap")
    if not _is_int(cap) or cap not in (25, 50):
        return f"stakeholderDecision.cap is {cap!r}, not 25 or 50"
    seeds = report.get("rngSeeds")
    if not isinstance(seeds, list) or not seeds or not all(map(_is_int, seeds)):
        return f"rngSeeds is {seeds!r}, not a non-empty list of ints"
    return None


def _finalized(data_dir: Path, role: str) -> bool:
    folder = data_dir / role
    return folder.is_dir() and any(folder.glob("*/generation.json"))


def balanced_runs(seed_ids: Sequence[str]) -> list[list[str]]:
    """plan6 §3.8's balanced cut: `⌈n / 25⌉` runs whose sizes differ by at most one, the larger
    ones first (880 seeds: 16 of 25, then 20 of 24); none is a short tail or exceeds 25."""
    runs = -(-len(seed_ids) // GENERATION_RUN_SIZE)
    if not runs:
        return []
    base, extra = divmod(len(seed_ids), runs)
    cuts, start = [], 0
    for index in range(runs):
        size = base + (index < extra)
        cuts.append(list(seed_ids[start:start + size]))
        start += size
    return cuts


def rubric_hash(role: str) -> str:
    """plan5 §3.3: sha256 over the UTF-8 `Point.rubric_body`, the value `promptSha256.rubric`
    holds: the key a message-type tranche is labelled under (plan4 §3.6)."""
    return hashlib.sha256(load_point(role).rubric_body.encode("utf-8")).hexdigest()


def design_strata(role: str) -> list[str | None]:
    """A point's design strata, in design order: the values of its design stratum axis (guard's
    evidence kinds), `[None]` for a point with none (note §Q1.3)."""
    design = load_point(role).design
    axis_name = design["designStratumAxis"]
    if axis_name is None:
        return [None]
    axis = next(a for a in design["stratified"] if a["axis"] == axis_name)
    return [entry["value"] for entry in axis["values"]]


def cell_target(role: str) -> int:
    """A cell's target: eval plus calibration, which equals eval (note §Q1.3's table)."""
    return 2 * load_point(role).definition["cells"]["evalPerCell"]


def topup_counts(role: str, labelled: Iterable[tuple[str | None, str]]) -> dict[str | None, int]:
    """plan4 §3.6 "Top-up planner", pure: the seeds to draw per short design stratum, from the
    labelled items' cells (design stratum, label). A cell is short below `cell_target`. Per short
    stratum `s` it is `max over its short cells ℓ of ⌈shortfall_ℓ / P̂(ℓ | s)⌉`, with `P̂(ℓ | s)`
    the share of `s`'s labelled items that carry ℓ (integer arithmetic). Nothing here names an
    intent: the draw is the design's own, restricted to `s`. A short cell with no labelled item
    has `P̂ = 0` and no draw count: `UnboundedCell` (a `CampaignRefused` carrying the cell; the
    planner turns it into the `unboundedCell` halt)."""
    counts = dict.fromkeys(itertools.product(design_strata(role), (
        labels.vote_key(v) for v in load_point(role).expected)), 0)
    for cell in labelled:
        counts[cell] += 1
    target = cell_target(role)
    out: dict[str | None, int] = {}
    for stratum in design_strata(role):
        size = sum(n for (s, _), n in counts.items() if s == stratum)
        draw = 0
        for (cell_stratum, label), n in counts.items():
            if cell_stratum != stratum or n >= target:
                continue
            if n == 0:
                raise UnboundedCell(label, stratum)
            draw = max(draw, -(-(target - n) * size // n))
        if draw:
            out[stratum] = draw
    return out


def wilson_upper_zero(items: int) -> float:
    """The Wilson 95% upper bound on P(label | stratum) when none of the stratum's `items`
    labelled items has the label: `z² / (n + z²)` (0.133 at 25 items, 0.071 at 50; note §Q1.3)."""
    return WILSON_Z**2 / (items + WILSON_Z**2)


@dataclass(frozen=True)
class TopUpProjection:
    """`project_topup`'s result: the projected tokens of a top-up round, round 1's total tokens
    (retired relabel attempts included, `State.spend`'s `totalTokens`), the live tokens the rate
    came from, and whether the projection exceeds half the total."""

    projected: Fraction
    round_one: int
    live: int

    @property
    def halts(self) -> bool:
        return SPEND_HALT_DENOMINATOR * self.projected > self.round_one

    def as_record(self) -> dict[str, Any]:
        return {"projectedTokens": float(self.projected), "roundOneTokens": self.round_one,
                "roundOneLiveTokens": self.live, "halts": self.halts}


def project_topup(*, round_one: Mapping[str, Any], topup_seeds: int,
                  topup_runs: int) -> TopUpProjection:
    """plan4 §3.6 "Spend halt" (note §Q1.3, plan6 §3.8), pure: `(top-up seeds / round-1 seeds) ×
    round 1's live labelling tokens + top-up generation runs × round 1's mean tokens per live
    generation run`, each side the sum of the four token classes, against round 1's total
    (`State.spend(1)`'s `totalTokens`, retired relabel attempts included). Seeds, not accepted
    items, sit on both sides of the ratio, so drops before labelling are priced at round 1's own
    rate. The round halts when the projection exceeds half of the total: exactly half does not."""
    classes = pilot.TOKEN_CLASSES
    label = sum(round_one["labelTokens"].get(c, 0) for c in classes)
    generation = sum(round_one["generationTokens"].get(c, 0) for c in classes)
    seeds, runs = round_one["seeds"], round_one["generationRuns"]
    if seeds < 1 or runs < 1:
        raise CampaignRefused("round 1 has no drawn seed or generation run to project from")
    projected = Fraction(topup_seeds * label, seeds) + Fraction(topup_runs * generation, runs)
    return TopUpProjection(projected, round_one["totalTokens"], label + generation)


# --- the campaign --------------------------------------------------------------------------------


class Campaign:
    """One campaign root, resolved to its physical path. `claude_dir` is where the child's
    transcripts land (`$HOME/.claude`); `clock` stamps events (tests pass a fixed one)."""

    def __init__(self, root: Path, *, claude_dir: Path | None = None,
                 clock: Callable[[], datetime] = _now) -> None:
        # The physical path: the child's `getcwd()` names the transcript folder, and a symlink
        # must not carry the root out from under `root_refusal` (plan4 §3.5).
        self.root = Path(os.path.realpath(root))
        self.exchange_dir = self.root / "exchange"
        self.store = self.root / "store"
        self.ledger_path = self.store / "ledger.jsonl"
        self.claude_dir = claude_dir if claude_dir is not None else Path.home() / ".claude"
        self.clock = clock
        #: A usage stop this instance's own `ingest` recorded: `next` then serves nothing more
        #: (module docstring, "a usage stop").
        self.usage_stop: Stop | None = None

    # -- the ledger -------------------------------------------------------------------------------

    def events(self) -> list[dict[str, Any]]:
        """The ledger's events. A line that does not parse, and above all a torn last line (no
        newline: a crash mid-append), refuses every command, naming its byte offset; it is never
        dropped, because it may be a commit point. A human repairs the ledger."""
        if not self.ledger_path.is_file():
            raise CampaignRefused(f"no campaign at {self.root} (run `driver init`)")
        lines = self.ledger_path.read_bytes().split(b"\n")   # never `str.splitlines`: U+2028
        torn = lines.pop()
        events, offset = [], 0
        for line in lines:
            if not line:
                offset += 1
                continue
            try:
                event = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise CampaignRefused(f"{self.ledger_path}: the line at byte {offset} is not "
                                      f"JSON ({exc}); inspect and repair it by hand") from None
            if not isinstance(event, dict) or "event" not in event:
                raise CampaignRefused(f"{self.ledger_path}: the line at byte {offset} is not an "
                                      "event; inspect and repair it by hand")
            events.append(event)
            offset += len(line) + 1
        if torn:
            raise CampaignRefused(
                f"{self.ledger_path}: the last line, at byte {offset}, is torn (no newline; "
                f"{len(torn)} bytes); inspect and repair it by hand")
        return events

    def state(self) -> State:
        return fold(self.events())

    def _append(self, event: Mapping[str, Any]) -> dict[str, Any]:
        stamped = {"event": event["event"], "at": self.clock().isoformat(),
                   **{k: v for k, v in event.items() if k != "event"}}
        line = json.dumps(stamped, ensure_ascii=False, separators=(",", ":")) + "\n"
        if self.ledger_path.is_file() and self.ledger_path.stat().st_size:
            with open(self.ledger_path, "rb") as ledger:
                ledger.seek(-1, os.SEEK_END)
                if ledger.read(1) != b"\n":
                    self.events()                       # refuses, naming the torn line
        with open(self.ledger_path, "ab") as handle:
            handle.write(line.encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        return stamped

    @contextmanager
    def locked(self):
        """plan6 §3.2: commands that append take `store/driver.lock` without blocking, and refuse
        while it is held."""
        self.store.mkdir(parents=True, exist_ok=True)
        with open(self.store / "driver.lock", "a+b") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise CampaignRefused("the campaign is locked by a running `driver run`") from None
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    # -- init -------------------------------------------------------------------------------------

    def init(self, *, point: str, phase: str, fixture: Path | None = None,
             batch_timeout: timedelta = DEFAULT_BATCH_TIMEOUT,
             launch_fail_bound: timedelta = DEFAULT_LAUNCH_FAIL_BOUND,
             data_dir: Path = DATA_DIR) -> None:
        """`driver init` (plan4 §3.5): refuse, writing nothing, or create the layout and record the
        `init` event (with `batchTimeout` and `launchFailBound` in seconds, plan6 §3.2, §3.4)."""
        report, items = self._init_checks(point, phase, fixture, data_dir)
        for duration, name in ((batch_timeout, "batchTimeout"),
                               (launch_fail_bound, "launchFailBound")):
            if duration <= timedelta(0):
                raise CampaignRefused(f"{name} must be positive")
        for folder in (self.exchange_dir / "req", self.exchange_dir / "ans", self.store):
            folder.mkdir(parents=True, exist_ok=True)
        with self.locked():
            (self.exchange_dir / "empty-mcp.json").write_text('{"mcpServers": {}}\n', "utf-8")
            cap = DEFAULT_LABEL_BATCH_CAP
            if report is not None:
                cap = report["stakeholderDecision"]["cap"]
            self._append({"event": "init", "point": point, "phase": phase,
                          "batchTimeout": int(batch_timeout.total_seconds()),
                          "launchFailBound": int(launch_fail_bound.total_seconds()),
                          "labelBatchCap": cap})
            if report is not None:
                self._append({"event": "held-rng-seeds", "source": "pilot-report.json",
                              "rngSeeds": list(report["rngSeeds"])})
            if items is not None:
                refs = [f"fx_{n:04d}" for n in range(1, len(items) + 1)]
                lines = "".join(json.dumps({"itemId": ref, "item": item}, ensure_ascii=False)
                                + "\n" for ref, item in zip(refs, items))
                _write_atomic(self.store / "items.jsonl", lines.encode("utf-8"))
                self._append({"event": "queued", "batch": "bat_0001", "kind": "label",
                              "point": point, "entries": refs})

    def _init_checks(self, point: str, phase: str, fixture: Path | None,
                     data_dir: Path) -> tuple[dict[str, Any] | None, list[Any] | None]:
        if point not in ROLES:
            raise CampaignRefused(f"unknown point {point!r}")
        if phase not in PHASES:
            raise CampaignRefused(f"unknown phase {phase!r}")
        refusal = root_refusal(self.root)
        if refusal is not None:
            raise CampaignRefused(refusal)
        if self.root.exists() and any(self.root.iterdir()):
            raise CampaignRefused(f"{self.root} is not empty")
        if (phase == "smoke") != (fixture is not None):
            raise CampaignRefused("--fixture is required with --phase smoke, and only with it")
        if phase == "pilot" and point != "guard-judge":
            raise CampaignRefused("the pilot phase is guard's (plan4 §3.8)")
        report = None
        if phase == "full":
            report = self._pilot_report(data_dir)
            position = FULL_ORDER.index(point)
            if position and not _finalized(data_dir, FULL_ORDER[position - 1]):
                raise CampaignRefused(f"{FULL_ORDER[position - 1]}'s campaign is not finalized")
        items = self._fixture_items(point, fixture) if fixture is not None else None
        return report, items

    @staticmethod
    def _pilot_report(data_dir: Path) -> dict[str, Any]:
        path = data_dir / PILOT_REPORT
        if not path.is_file():
            raise CampaignRefused(f"no guard pilot report at {path}")
        try:
            report = json.loads(path.read_text("utf-8"))
        except json.JSONDecodeError as exc:
            raise CampaignRefused(f"{path} is not JSON: {exc}") from None
        if not isinstance(report, dict):
            raise CampaignRefused(f"{path} is not a JSON object")
        refusal = spend_gate_refusal(report)
        if refusal is not None:
            raise CampaignRefused(f"the spend gate refuses: {refusal}")
        refusal = pilot_report_refusal(report)
        if refusal is not None:
            raise CampaignRefused(f"the pilot report refuses: {refusal}")
        return report

    @staticmethod
    def _fixture_items(point: str, fixture: Path) -> list[Any]:
        items = [json.loads(line) for line in fixture.read_text("utf-8").splitlines() if line]
        if not items:
            raise CampaignRefused(f"{fixture} holds no items")
        definition = load_point(point)
        for number, item in enumerate(items, 1):
            errors = input_field_errors(definition, item)
            if errors:
                raise CampaignRefused(f"{fixture} line {number}: {errors[0]}")
        return items

    # -- expectations, seeds and batches ----------------------------------------------------------

    def record_expectations(self, *, guard_command: str,
                            roles: Mapping[str, Mapping[str, Any]],
                            driver_template: str | None = None,
                            user_settings_digests: Mapping[str, str] | None = None,
                            prompt_sha256: Mapping[str, Any] | None = None,
                            settings_sha256: str | None = None) -> None:
        """The declared expectations `install` writes (plan4 §3.5; U5.7): per role (`generator`,
        `labeler`) its `agentType`, `model`, `effort`, `spawnTemplate` and `instructions`
        (`[path, sha256]` pairs), plus the guard's `command`. `driver run` also needs
        `driverTemplate` (plan6 §3.3's message, `{agent_type}` and `{spawn_prompt}` slots) and
        `userSettingsDigests` (`runner.user_settings_digests`, plan6 §3.2), and refuses without
        them. `install` also records `promptSha256` (plan4 §3.7's `agentDefinition`,
        `spawnTemplate`, `guardHook`, `requestTemplate` and `rubric` hashes) and
        `settingsSha256`. The latest event wins, except that only the first sets the per-role
        model pins and declared `instructions` (`fold`): a later `install`
        must not undo a `resolve --repin-model/--repin-instructions`."""
        if set(roles) != set(ROLE_OF_KIND.values()):
            raise CampaignRefused("expectations need exactly the generator and labeler roles")
        event: dict[str, Any] = {"event": "expectations", "guardCommand": guard_command,
                                 "roles": {role: dict(spec) for role, spec in roles.items()}}
        if driver_template is not None:
            event["driverTemplate"] = driver_template
        if user_settings_digests is not None:
            event["userSettingsDigests"] = dict(user_settings_digests)
        if prompt_sha256 is not None:
            event["promptSha256"] = json.loads(json.dumps(prompt_sha256))
        if settings_sha256 is not None:
            event["settingsSha256"] = settings_sha256
        self._append(event)

    def fresh_rng_seed(self) -> int:
        """plan5 §3.2: `secrets.randbits(32)`, redrawn while the ledger holds the value."""
        held = self.state().held_rng_seeds
        while True:
            value = secrets.randbits(32)
            if value not in held:
                return value

    def record_draw(self, draw: SeedDraw) -> None:
        """plan5 §3.2: append a `seed-draw` event, or refuse, writing nothing and naming the clash,
        when the draw's RNG seed or any of its `seedId`s is already held."""
        state = self.state()
        if draw.rng_seed in state.held_rng_seeds:
            raise CampaignRefused(f"RNG seed {draw.rng_seed} is already held")
        clash = sorted(seed.seed_id for seed in draw.seeds if seed.seed_id in state.seeds)
        if clash:
            raise CampaignRefused(f"seedId {clash[0]} is already held")
        self._append({"event": "seed-draw", "draw": draw.to_record()})

    def queue_batch(self, kind: str, entries: Sequence[str], **extra: Any) -> str:
        """Queue a `pending` batch of item ids (label) or seed ids (generate); returns its id."""
        if kind not in ROLE_OF_KIND:
            raise ValueError(f"unknown batch kind {kind!r}")
        state = self.state()
        batch_id = f"bat_{len(state.batches) + 1:04d}"
        self._append({"event": "queued", "batch": batch_id, "kind": kind,
                      "point": state.init["point"], "entries": list(entries), **extra})
        return batch_id

    def begin_round(self, gen_round: int, count: int | None = None, *,
                    design_stratum: str | None = None, extend: bool = False,
                    pilot_block: str | None = None, queue_runs: bool = True) -> list[str]:
        """plan4 §3.6 "Generation runs", plan6 §3.8: draw `count` seeds on a fresh RNG seed
        (plan5 §3.2: `record_draw` refuses a held one), record the round (`round-start`: its
        `genRound` and drawn seeds, which add to the halt denominator of that `genRound`), shuffle
        the seeds on that RNG seed and cut them into balanced runs (`balanced_runs`), each queued
        as a first-pass generation batch. Round 1's `count` is ⌈1.1 × target⌉
        (`seeds.round_one_count`); a later round names it. A `genRound` already started since the
        last generator restart is refused. Only U5.9's planner passes `extend=True`, for the
        second and later strata of one top-up round (each its own RNG seed and `round-start`):
        it is refused unless the `genRound` is started with no first-pass run issued, a
        `design_stratum` is given, and that stratum has no `round-start` in that `genRound` yet. A
        retry after a crash needs no `extend`: after `round-start` a repeat is refused as a
        duplicate (`next` queues the stranded seeds), and after the draw alone the stratum has no
        `round-start`, so it draws afresh. `pilot_block` (`MAIN_BLOCK` or `RECHECK_BLOCK`, a pilot
        campaign's only; `pilot.py` passes it) is recorded on the `round-start`: it decides which
        votes the round takes (`State.round_votes`). `queue_runs=False` records the draw and
        queues nothing, for a top-up round's strata that `queue_topup` cuts as one pool.
        Returns the queued batch ids."""
        if isinstance(gen_round, bool) or not isinstance(gen_round, int) or gen_round < 1:
            raise CampaignRefused(f"genRound must be a positive integer, not {gen_round!r}")
        state = self.state()
        if pilot_block is not None and (pilot_block not in PILOT_BLOCKS
                                        or state.init["phase"] != "pilot"):
            raise CampaignRefused(f"pilot block {pilot_block!r} needs a pilot campaign and "
                                  f"{' or '.join(PILOT_BLOCKS)}")
        started = any(r["genRound"] == gen_round for r in state.rounds)
        if started and not extend:
            raise CampaignRefused(f"genRound {gen_round} is already started (extend adds a draw)")
        if extend and not started:
            raise CampaignRefused(f"genRound {gen_round} has not started; nothing to extend")
        if extend and design_stratum is None:
            raise CampaignRefused("extend adds one design stratum to a round: name it")
        if extend and any(batch.attempts for batch in state.first_pass_runs(gen_round)):
            raise CampaignRefused(
                f"genRound {gen_round} has a run issued; extend draws before the first issues")
        if extend and any(r["genRound"] == gen_round and r["designStratum"] == design_stratum
                          for r in state.rounds):
            raise CampaignRefused(
                f"genRound {gen_round} already holds a draw of {design_stratum!r}")
        role = state.init["point"]
        if count is None:
            if gen_round != 1:
                raise CampaignRefused("a round after round 1 needs an explicit seed count")
            count = round_one_count(role)
        rng_seed = self.fresh_rng_seed()
        draw = draw_seeds(role, count, rng_seed, design_stratum=design_stratum)
        self.record_draw(draw)
        ordered = [seed.seed_id for seed in draw.seeds]
        random.Random(f"generation-runs|{rng_seed}").shuffle(ordered)
        self._append({"event": "round-start", "genRound": gen_round, "rngSeed": rng_seed,
                      "designStratum": design_stratum, "seeds": ordered,
                      **({"pilotBlock": pilot_block} if pilot_block else {})})
        return self._queue_runs(ordered, "first") if queue_runs else []

    def queue_regeneration(self) -> list[str]:
        """Note §Q1.2 "On a hit": the seeds a hit queued, pooled across rounds and design strata,
        cut into balanced generation runs (`[]` when none is awaiting). A regenerated item keeps
        its seed's `genRound` (`ingest` reads it per seed, never from the batch) and takes its new
        run's `genRunId`."""
        return self._queue_runs(self.state().awaiting_regeneration(), "regen")

    def queue_votes(self) -> list[str]:
        """plan6 §3.8 "Votes": start the next vote of the first settled round that has one (`[]`
        when none can start, or a vote is still running). The items are reshuffled on a fresh RNG
        seed that `vote-start` records (it joins the held set, plan5 §3.2), then queued in
        batches of at most `labelBatchCap`. A vote a crash left half-queued gets its missing items
        queued into it, in the recorded order (the generation half's `unbatched_seeds`). Message
        type's round 1 draws its tranche first (`tranche-start`) and votes it before the rest
        (plan4 §3.6): its batches and `vote-start`s carry `tranche`."""
        state = self.state()
        if any(batch.kind == "label" and batch.extra.get("vote") is not None
               and batch.state not in ("ingested", "dropped", "retired")
               for batch in state.batches.values()):
            return []
        for gen_round in sorted({r["genRound"] for r in state.rounds}):
            if state.tranche_due(gen_round):
                self._draw_tranche(state, gen_round)
                state = self.state()
            step = state.vote_step(gen_round)
            if step is None or not step[1]:
                continue
            vote, order, recorded, tranche = step
            if not recorded:
                rng_seed = self.fresh_rng_seed()
                random.Random(f"label-votes|{rng_seed}").shuffle(order)
                self._append({"event": "vote-start", "genRound": gen_round, "vote": vote,
                              "rngSeed": rng_seed, "items": order,
                              **({"tranche": True} if tranche else {})})
            cap = (labels.SCREEN_BATCH_CAP if vote == labels.SCREEN_VOTE
                   else state.init["labelBatchCap"])
            return [self.queue_batch("label", order[start:start + cap], genRound=gen_round,
                                     vote=vote, **({"tranche": True} if tranche else {}))
                    for start in range(0, len(order), cap)]
        return []

    def _draw_tranche(self, state: State, gen_round: int) -> None:
        """plan4 §3.6: `TRANCHE_ITEMS` of round `gen_round`'s accepted items, drawn at random on
        a fresh RNG seed (held, plan5 §3.2), with the rubric hash the tranche is labelled under.
        A round with fewer items is all tranche."""
        items = sorted(seed for seed, rec in state.items.items() if rec["genRound"] == gen_round)
        rng_seed = self.fresh_rng_seed()
        random.Random(f"tranche|{rng_seed}").shuffle(items)
        self._append({"event": "tranche-start", "genRound": gen_round, "rngSeed": rng_seed,
                      "items": sorted(items[:TRANCHE_ITEMS]),
                      "rubric": rubric_hash(state.init["point"])})

    def queue_pilot_block(self) -> list[str]:
        """Note §Q1.3 "Pilot" "Screen" (plan4 §3.8): in a pilot campaign whose main blocks have
        all been voted, `n < 80` draws one more main block of `pilot.ADDED_BLOCK_SEEDS` seeds,
        numbered after the last round; `[]` otherwise. The first block is `pilot.py start`'s.
        Once `pilot.PILOT_ADDED_BLOCK_LIMIT` added blocks have been drawn in the halt epoch
        (plan6 §3.8) it draws nothing and records the `pilot-block-halt` instead."""
        state = self.state()
        mains = [r["genRound"] for r in state.rounds if r["pilotBlock"] == MAIN_BLOCK]
        if (state.init["phase"] != "pilot" or not mains
                or not all(state.round_voted(g) for g in mains)
                or not pilot.screen(state.screen_votes(mains)).more_needed):
            return []
        if len(mains) - max(state.block_halt_base, 1) >= pilot.PILOT_ADDED_BLOCK_LIMIT:
            self.record_pilot_block_halt()
            return []
        return self.begin_round(max(r["genRound"] for r in state.rounds) + 1,
                                pilot.ADDED_BLOCK_SEEDS, pilot_block=MAIN_BLOCK)

    def queue_topup(self) -> list[str]:
        """plan4 §3.6 "Top-up planner" (phase `full`; plan6 §3.8): when every round is labelled
        and a cell is short, plan the next `genRound`. A pending probe (`resolve --probe-topup`)
        is planned first, before `topup_counts`: 25 seeds of its stratum alone. A short cell
        with no labelled item raises `UnboundedCell`, which records the `unboundedCell` halt
        (naming the cell, with the Wilson bound over the stratum's labelled items) and queues
        nothing. Before it is issued the spend projection is compared (`project_topup`); over
        half of round 1's tokens it records the `spend-halt` (`resolve` lets it through,
        `--decline-topup` ends top-ups) and queues nothing. Otherwise it records the
        `topup-plan` and draws every stratum before any run issues, the first with `begin_round`
        and each further one with `extend=True`, then cuts the pooled seeds into runs
        (`_queue_unbatched`). `[]` when nothing is short, a round is unlabelled, a halt is open
        or top-ups were declined."""
        state = self.state()
        if (state.init["phase"] != "full" or state.topup_declined or state.halt is not None
                or not state.rounds
                or not all(state.round_voted(g) for g in {r["genRound"] for r in state.rounds})):
            return []
        gen_round = max(r["genRound"] for r in state.rounds) + 1
        probe = state.probe is not None
        if probe:
            need: dict[str | None, int] = {state.probe["designStratum"]: PROBE_SEEDS}
        else:
            try:
                need = topup_counts(state.init["point"], state.labelled_cells())
            except UnboundedCell as cell:
                items = sum(1 for stratum, _ in state.labelled_cells()
                            if stratum == cell.stratum)
                self._append({"event": "unbounded-halt", "genRound": gen_round,
                              "label": cell.label, "stratum": cell.stratum, "items": items,
                              "bound": wilson_upper_zero(items)})
                return []
        if not need:
            return []
        runs = len(balanced_runs(["seed"] * sum(need.values())))
        projection = project_topup(round_one=state.spend(1), topup_seeds=sum(need.values()),
                                   topup_runs=runs)
        record = {"genRound": gen_round, "strata": [[s, n] for s, n in need.items()],
                  "projection": projection.as_record(), **({"probe": True} if probe else {})}
        if projection.halts and gen_round not in state.spend_approved:
            self._append({"event": "spend-halt", **record})
            return []
        self._append({"event": "topup-plan", **record})
        return self._draw_topup()

    def _resume_topup(self) -> list[str]:
        """The strata of the last `topup-plan` that no `round-start` of its `genRound` holds yet
        (a crash between the plan and its draws), drawn before any run issues; `[]` when none
        remains."""
        state = self.state()
        if not state.topup_plans or state.halt is not None:
            return []
        return self._draw_topup()

    def _draw_topup(self) -> list[str]:
        """Draw the strata of the last `topup-plan` that no `round-start` holds, then cut the
        pooled seeds into runs. The plan's run guard is gone: `todo` is empty once any run has
        issued, since runs queue only after every stratum is drawn (Pass 27 P27-m2)."""
        state = self.state()
        plan = state.topup_plans[-1]
        gen_round = plan["genRound"]
        drawn = {r["designStratum"] for r in state.rounds if r["genRound"] == gen_round}
        todo = [(s, n) for s, n in plan["strata"] if s not in drawn]
        if not todo:
            return []
        for stratum, count in todo:
            self.begin_round(gen_round, count, design_stratum=stratum,
                             extend=any(r["genRound"] == gen_round for r in self.state().rounds),
                             queue_runs=False)
        return self._queue_unbatched()

    def _queue_unbatched(self) -> list[str]:
        """plan6 §3.8 "A half-written round": queue the `round-start` seeds no batch holds, in
        the same balanced cut, as first-pass runs."""
        return self._queue_runs(self.state().unbatched_seeds(), "first")

    def _queue_runs(self, seed_ids: Sequence[str], generation: str) -> list[str]:
        return [self.queue_batch("generate", run, generation=generation)
                for run in balanced_runs(seed_ids)]

    def record_cue_check_halt(self) -> None:
        """plan5 §3.6: `pilot.py report` halts the pilot when the `missing` cue check fails;
        `resolve` clears it. Refused while any halt is open, and when one is already recorded."""
        state = self.state()
        if state.halt is not None:
            raise CampaignRefused(f"a halt is open ({_halt_text(state.halt)})")
        if state.cue_halts:
            raise CampaignRefused("the cue-check halt is already recorded")
        self._append({"event": "cue-check-halt"})

    def record_pilot_block_halt(self) -> None:
        """plan6 §3.8: the pilot's added blocks reached `pilot.PILOT_ADDED_BLOCK_LIMIT` with
        `n < 80`; the event records the main-block count, which starts the next epoch. Refused
        while any halt is open."""
        state = self.state()
        if state.halt is not None:
            raise CampaignRefused(f"a halt is open ({_halt_text(state.halt)})")
        mains = sum(1 for r in state.rounds if r["pilotBlock"] == MAIN_BLOCK)
        self._append({"event": "pilot-block-halt", "mainRounds": mains})

    def record_interrupted(self, batch: str) -> None:
        """plan6 §3.2: `pause --now` names the batch it stopped (the runner holds the lock)."""
        if self.state().batches[batch].open_attempt is None:
            raise CampaignRefused(f"{batch} has no issued attempt to interrupt")
        self._append({"event": "interrupted", "batch": batch})

    # -- next, prepare, issue ---------------------------------------------------------------------

    @property
    def pause_path(self) -> Path:
        return self.store / "pause"

    def next(self) -> Serve | Stop:
        """plan6 §3.2 step 1 (module docstring)."""
        if self.pause_path.exists():
            return Stop("paused", "store/pause exists")
        state = self.state()
        if state.halt is not None:
            return Stop("halted", _halt_text(state.halt))
        for attempt in list(state.attempts):
            if attempt.outcome is None or self._exchange_names(attempt.opaque):
                result = self.ingest(attempt.opaque)
                if result.stop is not None:
                    halt = self.state().halt
                    detail = (_halt_text(halt) if result.stop == "halted" and halt is not None
                              else result.outcome.get("reason", "") if result.outcome else "")
                    return Stop(result.stop, detail)
        stray = self._exchange_names()
        if stray:
            raise CampaignRefused(f"exchange/ holds files no attempt accounts for: {stray}")
        if self.usage_stop is not None:
            return self.usage_stop
        self._resume_topup()
        state = self.state()
        if state.unbatched_seeds() and self._queue_unbatched():
            state = self.state()
        pending = [b for b in state.batches.values() if b.state == "pending"]
        if not pending and self.queue_regeneration():
            state = self.state()
            pending = [b for b in state.batches.values() if b.state == "pending"]
        if not pending and self.queue_votes():
            state = self.state()
            pending = [b for b in state.batches.values() if b.state == "pending"]
        if not pending and self.queue_topup():
            state = self.state()
            pending = [b for b in state.batches.values() if b.state == "pending"]
        if not pending and self.state().halt is not None:        # the spend halt (`spend`)
            return Stop("halted", _halt_text(self.state().halt))
        if not pending:
            queued = self.queue_pilot_block()
            state = self.state()
            if state.halt is not None:           # the bound halted it (`pilot-block-halt`)
                return Stop("halted", _halt_text(state.halt))
            if queued:
                pending = [b for b in state.batches.values() if b.state == "pending"]
        if not pending:
            return Stop("done", "nothing is pending")
        return self.prepare(pending[0].batch_id)

    def prepare(self, batch_id: str) -> Serve:
        """Build the request for a pending batch, cutting what does not fit into a new batch."""
        state = self.state()
        batch = state.batches[batch_id]
        if batch.state != "pending":
            raise CampaignRefused(f"{batch_id} is {batch.state}, not pending")
        if state.expectations is None:
            raise CampaignRefused("no declared expectations (run `driver install`)")
        definition = load_point(batch.point)
        if batch.kind == "label":
            items = self._items()
            built = exchange.build_label_request(
                definition,
                [(ref, items[ref] if ref in items else state.items[ref]["item"])
                 for ref in batch.entries],
                exchange_dir=self.exchange_dir, cap=state.init["labelBatchCap"],
                held_keys=state.held_keys)
            kept = list(built.keys.values())
        else:
            built = exchange.build_generation_request(
                definition, [state.seeds[ref] for ref in batch.entries],
                exchange_dir=self.exchange_dir, cap=GENERATION_RUN_SIZE,
                held_keys=state.held_keys)
            kept = [seed.seed_id for seed in built.keys.values()]
        if built.overflow:
            self._append({"event": "cut", "batch": batch_id, "keep": kept,
                          "overflow": f"bat_{len(state.batches) + 1:04d}"})
        expected = self._expected(state, batch, built.opaque)
        return Serve(batch_id, batch.kind, built.opaque, expected.agent_type,
                     expected.spawn_prompt, built)

    def issue(self, serve: Serve, *, session_id: str | None = None) -> str:
        """plan6 §3.2 step 2: record `issued {batch, opaque, sessionId}` (fsync'd), then write the
        request. Returns the session id (a fresh uuid4 unless given)."""
        state = self.state()
        batch = state.batches[serve.batch]
        if batch.state != "pending":
            raise CampaignRefused(f"{serve.batch} is {batch.state}, not pending")
        if serve.opaque in state.opaques:
            raise CampaignRefused(f"{serve.opaque} is already issued")
        refs = [getattr(stored, "seed_id", stored) for stored in serve.request.keys.values()]
        if not set(refs) <= set(batch.entries):
            raise CampaignRefused(f"the request is stale for {serve.batch}")
        session = session_id if session_id is not None else str(uuid.uuid4())
        _write_atomic(self.store / "req" / f"{serve.opaque}.jsonl",
                      serve.request.text.encode("utf-8"))
        self._append({"event": "issued", "batch": serve.batch, "opaque": serve.opaque,
                      "sessionId": session, "attempt": len(batch.attempts) + 1,
                      "requestSha256": serve.request.request_sha256,
                      "keys": dict(zip(serve.request.keys, refs))})
        exchange.write_request(serve.request)
        return session

    # -- ingest -----------------------------------------------------------------------------------

    def transcript_path(self, session_id: str) -> Path:
        """`<claude_dir>/projects/<slug>/<sessionId>.jsonl` (H5)."""
        return (self.claude_dir / "projects" / transcript_slug(self.exchange_dir)
                / f"{session_id}.jsonl")

    def ingest(self, opaque: str) -> IngestResult:
        """plan6 §3.2 step 5 (module docstring)."""
        state = self.state()
        attempt = state.opaques.get(opaque)
        if attempt is None:
            raise CampaignRefused(f"{opaque} was never issued")
        if attempt.outcome is not None:
            self._reingest(attempt)
            return IngestResult(opaque, attempt.batch, None, None)
        batch = state.batches[attempt.batch]
        expected = self._expected(state, batch, opaque, attempt)
        launch = self._launch(state, attempt)
        result = audit.audit_run(
            self.transcript_path(attempt.session_id), expected,
            earlier_no_attempt=self._earlier_no_attempt(batch, attempt), launch=launch,
            interrupted=attempt.interrupted,
        )
        parts = self._exchange_names(opaque, folder="ans")
        answer = None
        if result.ingests_answer:
            answer = ingest_answer_files(self._request_text(opaque), self.exchange_dir / "ans",
                                         opaque, ingested_answer_sha256=None)
        verdict = audit.combine(result, answer)
        event = self._outcome_event(state, batch, attempt, result, verdict, launch, parts)
        if (verdict.status == "ingested" and isinstance(answer, Ingested)
                and batch.state != "retired"):
            self._record_generation(state, batch, attempt, answer.answers)
            self._record_votes(batch, attempt, answer.answers)
        recorded = self._append(event)
        self._move_to_store(opaque, parts)
        stop = {"halted": "halted", "usage-stop": "usage-stop"}.get(event["outcome"])
        if batch.state == "retired":      # recorded only: it must not stop the campaign
            stop = None
        after = (self.state().halt if event["outcome"] in ("ingested", "void")
                 and batch.state != "retired" else None)
        if (after is not None and after.get("voidClass") in ("template", "rubric")
                and state.halt is None):
            stop = "halted"
        if stop == "usage-stop":
            self.usage_stop = Stop(stop, event.get("reason", ""))
        return IngestResult(opaque, batch.batch_id, recorded, stop)

    def _record_votes(self, batch: Batch, attempt: Attempt, answers: Mapping[str, Any]) -> None:
        """plan6 §3.8 "Votes", at label ingest and before the run's `outcome` commits it: one
        `votes` event holds each item's answer for the batch's vote. A batch queued without a
        `vote` (the smoke fixture's) records nothing."""
        if batch.kind != "label" or batch.extra.get("vote") is None:
            return
        self._append({"event": "votes", "opaque": attempt.opaque, "batch": batch.batch_id,
                      "genRound": batch.extra["genRound"], "vote": batch.extra["vote"],
                      "answers": {ref: answers[key] for key, ref in attempt.keys.items()}})

    def _record_generation(self, state: State, batch: Batch, attempt: Attempt,
                           answers: Mapping[str, Any]) -> None:
        """plan5 §3.5 at generation ingest, before the run's `outcome` commits it: each answered
        item is screened and checked against its seed's bindings. An item that shows nothing is
        recorded in one `generated` event with the run's `genRunId` and the seed's `genRound`; one
        that shows a reason is a `screen-hit`, `binding-mismatch` or `form-mismatch` event under
        its first reason, listing every reason it shows. A batch not queued as a generation run
        (`begin_round`, `queue_regeneration`) is not checked. Appended before the outcome, folded
        only when it commits: a crash between them leaves events a re-ingest overwrites."""
        generation = batch.extra.get("generation")
        if batch.kind != "generate" or generation is None:
            return
        definition = load_point(batch.point)
        accepted: list[dict[str, Any]] = []
        hits: list[dict[str, Any]] = []
        for key, seed_id in attempt.keys.items():
            seed, item = state.seeds[seed_id], answers[key]
            check = screen.check_item(definition, seed.axes, item)
            if check.primary is None:
                accepted.append({"seedId": seed_id, "genRunId": batch.gen_run_id,
                                 "genRound": state.seed_round[seed_id],
                                 "item": item})
            else:
                hits.append({"event": HIT_EVENTS[check.primary], "opaque": attempt.opaque,
                             "batch": batch.batch_id, "seedId": seed_id,
                             "genRunId": batch.gen_run_id, "reasons": list(check.reasons),
                             "detail": check.detail, "form": check.form,
                             "firstPass": generation == "first"})
        self._append({"event": "generated", "opaque": attempt.opaque, "batch": batch.batch_id,
                      "items": accepted})
        for hit in hits:
            self._append(hit)

    def _outcome_event(self, state: State, batch: Batch, attempt: Attempt, result: audit.Audit,
                       verdict: audit.Verdict, launch: audit.Launch,
                       parts: Sequence[str]) -> dict[str, Any]:
        if verdict.halts:
            outcome = "halted"
        elif verdict.reissues_under_new_name:
            outcome = "usage-stop" if result.action == audit.WAIT else "reissued"
        else:
            outcome = "ingested" if verdict.status == "ingested" else "void"
        reasons = {finding.reason for finding in result.findings}
        runs = sum(1 for a in state.attempts if a.outcome and a.outcome.get("run"))
        run = audit.run_record(result, verdict, run_id=f"run_{runs + 1:04d}", role=batch.kind,
                               batch_id=batch.batch_id, attempt=attempt.attempt,
                               request_sha256=attempt.request_sha256)
        event: dict[str, Any] = {"event": "outcome", "batch": batch.batch_id,
                                 "opaque": attempt.opaque, "outcome": outcome}
        if verdict.void_reason is not None:
            event["reason"] = verdict.void_reason
        if verdict.answer_sha256 is not None:
            event["answerSha256"] = verdict.answer_sha256
        # No `countsAttempt`: `outcome == "void"` is the counted attempt, the one fact `fold` reads
        # (`combine` sets `counts_attempt` exactly then), so a second copy could only drift.
        event.update({"voidClass": verdict.void_class, "details": list(verdict.details),
                      "parts": list(parts), "produced": not reasons & _NOT_PRODUCED, "run": run})
        if verdict.void_reason in ("launchFailed", "launchFailing"):
            event["launch"] = self._launch_evidence(attempt.session_id)
        return event

    def _reingest(self, attempt: Attempt) -> None:
        """The carried ruling's re-ingest path: never `audit_run` or `combine`; nothing appended."""
        opaque, recorded = attempt.opaque, attempt.outcome
        names = list(recorded.get("parts", []))
        exchange_ans, store_ans = self.exchange_dir / "ans", self.store / "ans"
        late = sorted(set(self._exchange_names(opaque, folder="ans")) - set(names))
        if late:
            raise AnswerRefused(f"{opaque} is closed; answer parts it did not record: {late}")
        for name in names:
            mine, kept = exchange_ans / name, store_ans / name
            if mine.is_file() and kept.is_file() and mine.read_bytes() != kept.read_bytes():
                raise AnswerRefused(f"{opaque}: {name} differs from its stored copy")
        sha = recorded.get("answerSha256")
        if sha is not None:
            whole = all((store_ans / name).is_file() for name in names)
            source = store_ans if whole else exchange_ans
            again = ingest_answer_files(self._request_text(opaque), source, opaque,
                                        ingested_answer_sha256=sha)
            if not isinstance(again, AlreadyIngested):
                raise AnswerRefused(f"{opaque}: the re-read is not the recorded answer")
        self._move_to_store(opaque, names)

    def _move_to_store(self, opaque: str, parts: Sequence[str]) -> None:
        """Move the answer parts as a set (copy all, fsync, then unlink), then drop the exchange
        copy of the request, whose store copy `issue` wrote first."""
        source, target = self.exchange_dir / "ans", self.store / "ans"
        target.mkdir(parents=True, exist_ok=True)
        for name in parts:
            if (source / name).is_file() and not (target / name).is_file():
                _write_atomic(target / name, (source / name).read_bytes())
        for name in parts:
            (source / name).unlink(missing_ok=True)
        request = self.exchange_dir / "req" / f"{opaque}.jsonl"
        request.unlink(missing_ok=True)
        request.with_name(f".{request.name}.partial").unlink(missing_ok=True)

    def _exchange_names(self, opaque: str | None = None, *, folder: str | None = None) -> list[str]:
        """Files in `exchange/req/` and `exchange/ans/` (one of them with `folder`), all of them or
        those of `opaque`'s request and answer (a write's `.partial` included)."""
        found = []
        for name in (folder,) if folder else ("req", "ans"):
            path = self.exchange_dir / name
            for entry in sorted(path.iterdir()) if path.is_dir() else []:
                bare = entry.name.removeprefix(".")
                if opaque is None or bare.startswith(f"{opaque}."):
                    found.append(entry.name)
        return found

    def _request_text(self, opaque: str) -> str:
        return (self.store / "req" / f"{opaque}.jsonl").read_text("utf-8")

    def _items(self) -> dict[str, Any]:
        path = self.store / "items.jsonl"
        lines = path.read_text("utf-8").splitlines() if path.is_file() else []
        return {record["itemId"]: record["item"] for record in map(json.loads, lines) if record}

    def _expected(self, state: State, batch: Batch, opaque: str,
                  attempt: Attempt | None = None) -> audit.Expected:
        """The audit's expectations with absolute paths (Pass 9: relative ones breach). For an
        issued `attempt` everything comes from what was in force at its `issued` (plan6 §3.8):
        the `expectations` event (template, agent type, effort, guard command), the model pin and
        the instructions. Without one (a request about to be served), the folded latest."""
        expectations = state.expectations if attempt is None else attempt.expectations
        pins = state.pins if attempt is None else attempt.pins
        instructions = state.instructions if attempt is None else attempt.instructions
        role = ROLE_OF_KIND[batch.kind]
        spec = expectations["roles"][role]
        return audit.Expected(
            agent_type=spec["agentType"], model=pins[role], effort=spec["effort"],
            spawn_template=spec["spawnTemplate"],
            request_path=str(self.exchange_dir / "req" / f"{opaque}.jsonl"),
            answer_path=str(self.exchange_dir / "ans" / f"{opaque}.jsonl"),
            guard_command=expectations["guardCommand"],
            instructions=frozenset(tuple(pair) for pair in instructions[role]),
        )

    def _launch(self, state: State, attempt: Attempt) -> audit.Launch:
        """`Launch` from the ledger's `issued` times and `init`'s `launchFailBound` (Pass 9)."""
        earlier = [a for a in state.attempts if a is not attempt and a.outcome is not None
                   and a.epoch == attempt.epoch and a.issued_at <= attempt.issued_at]
        failures = tuple(a.issued_at for a in earlier
                         if a.outcome.get("reason") in ("launchFailed", "launchFailing"))
        produced = [a.issued_at for a in earlier if a.outcome.get("produced")]
        return audit.Launch(
            fail_bound=timedelta(seconds=state.init["launchFailBound"]),
            exit_code=self._exit_code(attempt.session_id), at=attempt.issued_at,
            earlier_failures=failures, last_produced=max(produced, default=None),
        )

    @staticmethod
    def _earlier_no_attempt(batch: Batch, attempt: Attempt) -> int:
        """The batch's earlier never-spawned and incomplete outcomes since the last resolve,
        leaving out interrupted attempts (plan6 §3.2; Pass 9)."""
        return sum(1 for a in batch.attempts if a is not attempt and a.outcome is not None
                   and a.epoch == attempt.epoch and not a.interrupted
                   and a.outcome.get("reason") in _NO_ATTEMPT)

    def _runs_file(self, session_id: str, suffix: str) -> Path:
        return self.store / "runs" / f"{session_id}{suffix}"

    def _observed_exit(self, session_id: str) -> int | None:
        """The child's exit code from `store/runs/<sessionId>.exit`, or `None` when it was never
        observed: the file is missing, or does not hold the format (module docstring)."""
        path = self._runs_file(session_id, ".exit")
        try:
            text = path.read_bytes().decode("ascii").strip()
        except (OSError, UnicodeDecodeError):
            return None
        return int(text) if _EXIT_CODE.fullmatch(text) else None

    def _exit_code(self, session_id: str) -> int:
        observed = self._observed_exit(session_id)
        return -1 if observed is None else observed  # unobserved counts as non-zero

    def _launch_evidence(self, session_id: str) -> dict[str, Any]:
        """plan6 §3.4: stdout, stderr and exit code, recorded as is (never parsed); `exitCode` is
        `None` when the exit was never observed."""
        def text(suffix: str) -> str | None:
            path = self._runs_file(session_id, suffix)
            return path.read_text("utf-8", errors="replace") if path.is_file() else None

        return {"exitCode": self._observed_exit(session_id), "stdout": text(".json"),
                "stderr": text(".err")}

    # -- resolve ----------------------------------------------------------------------------------

    def resolve(self, *, decision: str, skip_batch: bool = False,
                repin_instructions: bool = False, decline_topup: bool = False,
                reopen_topup: bool = False, probe_topup: bool = False,
                repin_model: tuple[str, str] | None = None,
                stakeholder_ref: str | None = None) -> list[str]:
        """`driver resolve` (plan4 §3.3 "Halt", plan6 §3.5): clear the halt with one recorded
        decision. Retry re-issues the halted batch under a new name; `--skip-batch` drops it
        into `safeguardSkipped`; a re-pin records its new model or instructions set. A `rubric`
        halt (the tranche's rule failed) is cleared by `tranche-restart` and needs the rubric
        hash to have changed since the tranche was labelled, so the relabelling means something;
        a `spend` halt is let through by a plain resolve (the stakeholder's "yes") or ended by
        `--decline-topup` (the "no", plan4 §3.6): no top-up round is issued after it. An
        `unboundedCell` halt (a short cell with no labelled item) is refused a plain resolve,
        since nothing changed and it would halt again: it takes `--probe-topup` (records
        `probe-requested`: one run of 25 seeds of the cell's stratum), `--decline-topup` or
        `--repin-model generator`. `--reopen-topup` is the one action valid with no halt open: it
        needs a recorded decline and appends `topup-reopened`, so the next `driver run` plans
        again. A `recheck` halt (`finalize`'s whole-point tranche rule failed, plan4 §3.6) is
        cleared, like the tranche's, by a changed rubric hash and relabels the whole point
        (`point-relabel`); its second failure is cleared only by `--stakeholder-ref` alone (a re-pin
        is refused), which records the stakeholder's acceptance (`recheck-accepted`) and relabels
        nothing. Returns the lines
        it prints (plan6 §3.8): the halt it cleared and, when the drop
        closed a round or pool, the halt that raised, else the line that `driver run` resumes."""
        if not decision.strip():
            raise CampaignRefused("--decision needs a text")
        if (skip_batch + repin_instructions + decline_topup + reopen_topup + probe_topup
                + (repin_model is not None)) > 1:
            raise CampaignRefused("give at most one of --skip-batch, --repin-instructions, "
                                  "--decline-topup, --reopen-topup, --probe-topup and "
                                  "--repin-model")
        with self.locked():
            state = self.state()
            halt = state.halt
            if reopen_topup:
                if halt is not None:
                    raise CampaignRefused(f"a halt is open ({_halt_text(halt)}): clear it first")
                if not state.topup_declined:
                    raise CampaignRefused("no top-up decline is in force; nothing to reopen")
                self._append({"event": "topup-reopened", "decision": decision,
                              "action": "reopen-topup"})
                return ["top-up reopened; the next `driver run` plans the top-up again"]
            if halt is None:
                raise CampaignRefused("the campaign is not halted")
            second_recheck = halt.get("reason") == "recheck" and halt.get("second")
            if stakeholder_ref is not None and not second_recheck:
                raise CampaignRefused("--stakeholder-ref is for the second failure of "
                                      "`finalize`'s whole-point tranche recheck (plan4 §3.6)")
            if second_recheck and (repin_model is not None or repin_instructions):
                raise CampaignRefused(
                    "the second failure of the whole-point recheck is cleared only by "
                    "`--stakeholder-ref`, alone: a re-pin neither clears it nor carries the ref")
            event: dict[str, Any] = {"event": "resolved", "decision": decision,
                                     "batch": halt["batch"], "reason": halt.get("reason")}
            if decline_topup:
                if halt.get("voidClass") not in ("spend", "unboundedCell"):
                    raise CampaignRefused(
                        "--decline-topup is for a spend or unboundedCell halt (plan4 §3.6)")
                event["event"], event["action"] = "topup-declined", "decline-topup"
            elif probe_topup:
                if halt.get("voidClass") != "unboundedCell":
                    raise CampaignRefused("--probe-topup is for an unboundedCell halt (plan6 §3.8)")
                event.update(event="probe-requested", action="probe-topup",
                             designStratum=halt["stratum"])
            elif skip_batch:
                if halt.get("reason") != "safeguardStop":
                    raise CampaignRefused("--skip-batch is for a safeguardStop halt (plan6 §3.5)")
                event["action"] = "skip-batch"
            elif repin_instructions:
                run = halt.get("run")
                if not run:
                    raise CampaignRefused("the halted run has no transcript to re-pin from")
                event.update(action="repin-instructions",
                             role=ROLE_OF_KIND[state.batches[halt["batch"]].kind],
                             instructions=[
                    [entry["path"], entry["sha256"]] for entry in run["instructions"]])
            elif repin_model is not None:
                event.update(self._repin_model(state, *repin_model))
            elif second_recheck:
                if not (stakeholder_ref or "").strip():
                    raise CampaignRefused(
                        "the recheck failed a second time, so the label set may not separate: "
                        "only the stakeholder clears it; give --stakeholder-ref "
                        "(who decided, and where it is recorded)")
                event.update(event="recheck-accepted", action="accept-recheck",
                             stakeholderRef=stakeholder_ref.strip())
            elif halt.get("voidClass") == "rubric" and halt.get("reason") == "recheck":
                now = rubric_hash(state.init["point"])
                if now == state.tranche["rubric"]:
                    raise CampaignRefused(
                        "the rubric is unchanged since the point was labelled "
                        f"({now[:12]}): fix it first, then resolve")
                event.update(event="point-relabel", action="relabel-point", rubric=now)
            elif halt.get("voidClass") == "rubric":
                now = rubric_hash(state.init["point"])
                if now == state.tranche["rubric"]:
                    raise CampaignRefused(
                        "the rubric is unchanged since the tranche was labelled "
                        f"({now[:12]}): fix it first, then resolve")
                event.update(event="tranche-restart", action="relabel-tranche", rubric=now)
            elif halt.get("voidClass") == "unboundedCell":
                raise CampaignRefused(
                    "a plain resolve changes nothing for an unboundedCell halt and it would halt "
                    "again: give --probe-topup, --decline-topup or --repin-model generator")
            else:
                event["action"] = "retry"
            self._append(event)
            raised = self.state().halt
        return [f"cleared: {_halt_text(halt)}",
                f"HALTED: {_halt_text(raised)}; clear with `driver resolve`" if raised
                else "top-up declined; no top-up round will be issued" if decline_topup
                else f"probe requested: the next `driver run` draws {PROBE_SEEDS} seeds of "
                     f"{halt['stratum'] or 'the point'}" if probe_topup
                else "resolved; the next `driver run` resumes"]

    @staticmethod
    def _repin_model(state: State, role: str, model: str) -> dict[str, Any]:
        if role not in ("generator", "labeler"):
            raise CampaignRefused(f"--repin-model takes generator or labeler, not {role!r}")
        other = "labeler" if role == "generator" else "generator"
        if not model or model == state.pins.get(other):
            raise CampaignRefused(f"the {role} cannot pin {model!r}: it equals the {other}'s")
        restart = "generation-restart" if role == "generator" else "labelling-restart"
        return {"event": restart, "action": "repin-model", "role": role, "model": model}

    # -- status -----------------------------------------------------------------------------------

    def status(self) -> list[str]:
        """`driver status`: read-only, needs no lock (plan6 §3.2)."""
        state = self.state()
        counts: dict[str, int] = {}
        for batch in state.batches.values():
            if batch.state != "retired":
                counts[batch.state] = counts.get(batch.state, 0) + 1
        init = state.init
        lines = [f"point {init['point']}, phase {init['phase']}",
                 "batches: " + (", ".join(f"{n} {s}" for s, n in sorted(counts.items()))
                                or "none queued")]
        if state.halt is not None:
            lines.append(f"HALTED: {_halt_text(state.halt)}; clear with `driver resolve`")
        if self.pause_path.exists():
            lines.append("PAUSED: store/pause exists; `driver resume` removes it")
        # `run` refuses this root before taking the lock, so it writes no `store/refusal`: the
        # reason is computed here, live (plan6 §3.2, "`status` prints it").
        root = root_refusal(self.root)
        if root is not None:
            lines.append(f"REFUSED: {root}")
        refusal = self.store / "refusal"
        if refusal.is_file():
            lines.append(f"REFUSED: {refusal.read_text('utf-8').strip()}")
        if state.attempts:
            last = state.attempts[-1]
            lines.append(f"last issued: {last.batch} session {last.session_id}")
        dropped = {k: v for k, v in state.dropped().items() if any(v.values())}
        if dropped:
            lines.append(f"dropped: {json.dumps(dropped, sort_keys=True)}")
        if state.vote_ties():
            lines.append(f"voteTie (three different labels): {len(state.vote_ties())} items")
        if state.tranche is not None:
            lines.append(f"tranche: {len(state.tranche['items'])} items, "
                         f"{len(state.tranche_drops())} voteTie drops, "
                         f"{'passed' if state.tranche_passed() else 'not passed'} "
                         f"(rubric {state.tranche['rubric'][:12]})")
        if state.topup_declined:
            lines.append("top-up declined: no top-up round will be issued; "
                         "`driver resolve --reopen-topup` reopens it")
        if state.probe is not None:
            lines.append(f"probe pending: {PROBE_SEEDS} seeds of "
                         f"{state.probe['designStratum'] or 'the point'} at the next top-up")
        if state.dropped_seeds:
            lines.append("seeds dropped on a third hit: "
                         + json.dumps(state.generation_drops(), sort_keys=True))
        return lines


def _halt_text(halt: Mapping[str, Any]) -> str:
    """One line; a template halt lists its round part and every pool that crossed (plan6 §3.8)."""
    text = f"{halt.get('reason')} ({halt.get('voidClass')})"
    if halt.get("batch") is not None:
        text += f" on {halt['batch']}"
    parts = []
    if halt.get("voidClass") == "rubric":
        scope = "labelled items of the whole point" if halt.get("reason") == "recheck" \
            else "tranche items"
        parts.append(f"{halt['drops']} voteTie drops in {halt['items']} {scope} under "
                     f"rubric {halt['rubric'][:12]}")
    if halt.get("voidClass") == "spend":
        projection = halt["projection"]
        live = projection.get("roundOneLiveTokens")       # absent in a pre-fix-round ledger
        parts.append(f"round {halt['genRound']} {'probe' if halt.get('probe') else 'top-up'} "
                     f"projected {projection['projectedTokens']:.0f} tokens against round 1's "
                     f"total {projection['roundOneTokens']}"
                     + (f" (live {live}; the bound is half the total)" if live is not None
                        else ""))
    if halt.get("voidClass") == "unboundedCell":
        parts.append(f"cell ({halt['label']}, {halt['stratum'] or 'no stratum'}) holds no "
                     f"labelled item; Wilson 95% upper bound for P(label | stratum) "
                     f"{halt['bound']:.3f} over {halt['items']} labelled items of the stratum")
    if "round" in halt:
        parts.append(f"round {halt['round']['genRound']}: " + ", ".join(halt["round"]["reasons"]))
    for pool in halt.get("pools", ()):
        hits = ", ".join(f"{reason} {pool['poolHits'][reason]}" for reason in pool["reasons"])
        parts.append(f"pool {pool['templateHash'][:12]} ({pool['poolSeeds']} seeds, "
                     f"rounds {pool['poolRounds']}): {', '.join(pool['reasons'])} [{hits}]")
    return text + (" [" + "; ".join(parts) + "]" if parts else "")
