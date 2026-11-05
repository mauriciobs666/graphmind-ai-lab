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
(`State.halt`, `voidClass: "template"`): per (`genRound`, reason) over that round's drawn seeds,
and per (generator-template hash, reason) over ingested first-pass seeds, each needing two hits
and over 5%, each firing once, regeneration hits excluded. An attempt carries the `expectations`
in force at its `issued` (`Attempt.expectations`), which the audit and the hash read. A
`generation-restart` (a generator re-pin) resets the folded generation view and `retire`s every
batch (`_restart_generation`). A batch queued by plain `queue_batch` carries no `generation` and
is not checked.

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
from pathlib import Path
from typing import Any

from . import audit, exchange, screen
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
#: plan5 §3.5: a round halts when more than 5% of its drawn seeds show one reason in first pass.
HALT_PERCENT = 5
#: plan6 §3.8: a round or a template pool halts on at least this many hits, and over 5%.
HALT_MIN_HITS = 2
#: The ledger events that record one rejected generated item, by the reason that rejected it.
HIT_EVENTS = {screen.SCREEN: "screen-hit", screen.BINDING_MISMATCH: "binding-mismatch",
              screen.FORM_MISMATCH: "form-mismatch"}
_HIT_EVENT_KINDS = frozenset(HIT_EVENTS.values())
#: plan4 §3.5: `full` needs the previous point's campaign finalized.
FULL_ORDER = ("guard-judge", "message-type-classifier", "join-in-judge")
#: plan5 §3.6's spend gate states.
PASSING_CUE_STATES = ("passed", "recheck-passed")

ROLE_OF_KIND = {"generate": "generator", "label": "labeler"}
_CLEARING_EVENTS = ("resolved", "generation-restart", "labelling-restart")
_NOT_PRODUCED = frozenset({"launchFailed", "launchFailing", "neverSpawned"})
_NO_ATTEMPT = frozenset({"neverSpawned", "incomplete"})
_EXIT_CODE = re.compile(r"-?[0-9]+")

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
PILOT_REPORT = Path("guard-judge") / "pilot" / "pilot-report.json"


class CampaignRefused(Exception):
    """A refused command or call; nothing was written."""


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
    #: plan6 §3.8's pooled halt, per generator-template hash: first-pass seeds ingested and hits
    #: by reason; and the (hash, reason) pairs already fired.
    pool: dict[str, dict[str, Any]] = field(default_factory=dict)
    pool_halts: set[tuple[str, str]] = field(default_factory=set)
    staged: dict[str, dict[str, Any]] = field(default_factory=dict)

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

    def unbatched_seeds(self) -> list[str]:
        """The `round-start` seeds no live generation batch holds (a crash between the round and
        its `queued` events), in round order."""
        held = {seed for batch in self.batches.values() if batch.kind == "generate"
                for seed in batch.entries}
        return [seed for r in self.rounds for seed in r["seeds"] if seed not in held]

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
            state.rounds.append({"genRound": event["genRound"], "seeds": list(event["seeds"])})
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
                expectations=state.expectations,
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
    """plan5 §3.5 and plan6 §3.8: more than 5% of the seeds, and at least two hits."""
    return hits >= HALT_MIN_HITS and 100 * hits > HALT_PERCENT * drawn


def _apply_generation(state: State, attempt: Attempt, batch: Batch,
                      staged: Mapping[str, Any]) -> None:
    """The ingested generation run's items and hits, then plan6 §3.8's halts after each run: one
    `template` halt per (`genRound`, reason) over that round's drawn seeds, and one per
    (generator-template hash, reason) over the ingested first-pass seeds. Regeneration hits count
    toward the 3-hit cap only."""
    for record in staged["items"]:
        state.items[record["seedId"]] = record
    first_pass = batch.extra.get("generation") == "first"
    digest = generator_template_hash(attempt.expectations)
    pool = state.pool.setdefault(digest, {"seeds": 0, "hits": {}}) if first_pass else None
    if pool is not None:
        pool["seeds"] += len(attempt.keys)
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
                pool["hits"][reason] = pool["hits"].get(reason, 0) + 1
    rounds = [(g, reason) for g in sorted(touched) for reason in screen.REASONS
              if (g, reason) not in state.template_halts
              and _over_threshold(state.round_hits.get((g, reason), 0), state.drawn_in(g))]
    pooled = [] if pool is None else [
        (digest, reason) for reason in screen.REASONS
        if (digest, reason) not in state.pool_halts
        and _over_threshold(pool["hits"].get(reason, 0), pool["seeds"])]
    if (rounds or pooled) and state.halt is None:
        state.template_halts.update(rounds)
        state.pool_halts.update(pooled)
        reasons = [r for r in screen.REASONS if r in {x[1] for x in rounds + pooled}]
        state.halt = {"event": "template-halt", "batch": batch.batch_id, "reason": reasons[0],
                      "reasons": reasons, "voidClass": "template",
                      "genRound": rounds[0][0] if rounds else None, "pooled": bool(pooled),
                      "templateHash": digest}


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
    state.rounds.clear()
    state.seed_round.clear()
    state.staged.clear()
    for batch in state.batches.values():
        batch.state, batch.dropped = "retired", None


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
        if staged is not None:
            _apply_generation(state, attempt, batch, staged)
    elif outcome == "void":
        batch.counted += 1
        if batch.counted >= MAX_COUNTED_ATTEMPTS:
            batch.state, batch.dropped = "dropped", "retryCapExceeded"
        else:
            batch.state = "pending"
    elif outcome in ("reissued", "usage-stop"):
        batch.state = "pending"
    elif outcome == "halted":
        batch.state = "halted"
        state.halt = dict(event)
    else:
        raise ValueError(f"unknown outcome {outcome!r} in the ledger")


def _fold_resolution(state: State, event: Mapping[str, Any]) -> None:
    halted = state.halt
    state.halt = None
    state.epoch += 1
    batch = state.batches.get(halted["batch"]) if halted else None
    if event.get("action") == "skip-batch" and batch is not None:
        batch.state, batch.dropped = "dropped", "safeguardSkipped"
    elif batch is not None and batch.state == "halted":
        batch.state = "pending"
    if event["event"] == "generation-restart":
        _restart_generation(state)
    if event["event"] in ("generation-restart", "labelling-restart"):
        state.pins[event["role"]] = event["model"]
    if event.get("action") == "repin-instructions":
        state.instructions[event["role"]] = [list(pair) for pair in event["instructions"]]


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
        `spawnTemplate` and `guardHook` hashes) and `settingsSha256`. The latest event wins, except
        that only the first sets the per-role model pins and declared `instructions` (`fold`): a
        later `install` must not undo a `resolve --repin-model/--repin-instructions`."""
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
                    design_stratum: str | None = None, extend: bool = False) -> list[str]:
        """plan4 §3.6 "Generation runs", plan6 §3.8: draw `count` seeds on a fresh RNG seed
        (plan5 §3.2: `record_draw` refuses a held one), record the round (`round-start`: its
        `genRound` and drawn seeds, which add to the halt denominator of that `genRound`), shuffle
        the seeds on that RNG seed and cut them into balanced runs (`balanced_runs`), each queued
        as a first-pass generation batch. Round 1's `count` is ⌈1.1 × target⌉
        (`seeds.round_one_count`); a later round names it. A `genRound` already started since the
        last generator restart is refused unless `extend` adds a draw to it (a top-up round draws
        one stratum per call, each its own RNG seed); `extend` needs the round to exist. A crash
        after the draw leaves an orphan draw and no round; the retry draws afresh. Returns the
        queued batch ids."""
        if isinstance(gen_round, bool) or not isinstance(gen_round, int) or gen_round < 1:
            raise CampaignRefused(f"genRound must be a positive integer, not {gen_round!r}")
        state = self.state()
        started = any(r["genRound"] == gen_round for r in state.rounds)
        if started and not extend:
            raise CampaignRefused(f"genRound {gen_round} is already started (extend adds a draw)")
        if extend and not started:
            raise CampaignRefused(f"genRound {gen_round} has not started; nothing to extend")
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
                      "designStratum": design_stratum, "seeds": ordered})
        return self._queue_runs(ordered, "first")

    def queue_regeneration(self) -> list[str]:
        """Note §Q1.2 "On a hit": the seeds a hit queued, pooled across rounds and design strata,
        cut into balanced generation runs (`[]` when none is awaiting). A regenerated item keeps
        its seed's `genRound` (`ingest` reads it per seed, never from the batch) and takes its new
        run's `genRunId`."""
        return self._queue_runs(self.state().awaiting_regeneration(), "regen")

    def _queue_unbatched(self) -> list[str]:
        """plan6 §3.8 "A half-written round": queue the `round-start` seeds no batch holds, in
        the same balanced cut, as first-pass runs."""
        return self._queue_runs(self.state().unbatched_seeds(), "first")

    def _queue_runs(self, seed_ids: Sequence[str], generation: str) -> list[str]:
        return [self.queue_batch("generate", run, generation=generation)
                for run in balanced_runs(seed_ids)]

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
        state = self.state()
        if state.unbatched_seeds() and self._queue_unbatched():
            state = self.state()
        pending = [b for b in state.batches.values() if b.state == "pending"]
        if not pending and self.queue_regeneration():
            state = self.state()
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
                definition, [(ref, items[ref]) for ref in batch.entries],
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
        expected = self._expected(state, batch, opaque, attempt.expectations)
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
        recorded = self._append(event)
        self._move_to_store(opaque, parts)
        stop = {"halted": "halted", "usage-stop": "usage-stop"}.get(event["outcome"])
        if batch.state == "retired":      # recorded only: it must not stop the campaign
            stop = None
        after = (self.state().halt if event["outcome"] == "ingested"
                 and batch.state != "retired" else None)
        if after is not None and after.get("voidClass") == "template" and state.halt is None:
            stop = "halted"
        if stop == "usage-stop":
            self.usage_stop = Stop(stop, event.get("reason", ""))
        return IngestResult(opaque, batch.batch_id, recorded, stop)

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
                  expectations: Mapping[str, Any] | None = None) -> audit.Expected:
        """The audit's expectations with absolute paths (Pass 9: relative ones breach). The agent
        type, effort, spawn template and guard command come from `expectations`: for an issued
        attempt, the event in force at its `issued` (plan6 §3.8; default: the latest). The model
        pin and the instructions come from the folded `state`, which `resolve` re-pins."""
        expectations = state.expectations if expectations is None else expectations
        role = ROLE_OF_KIND[batch.kind]
        spec = expectations["roles"][role]
        return audit.Expected(
            agent_type=spec["agentType"], model=state.pins[role], effort=spec["effort"],
            spawn_template=spec["spawnTemplate"],
            request_path=str(self.exchange_dir / "req" / f"{opaque}.jsonl"),
            answer_path=str(self.exchange_dir / "ans" / f"{opaque}.jsonl"),
            guard_command=expectations["guardCommand"],
            instructions=frozenset(tuple(pair) for pair in state.instructions[role]),
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
                repin_instructions: bool = False,
                repin_model: tuple[str, str] | None = None) -> None:
        """`driver resolve` (plan4 §3.3 "Halt", plan6 §3.5): clear the halt with one recorded
        decision. Retry re-issues the halted batch under a new name; `--skip-batch` drops it
        into `safeguardSkipped`; a re-pin records its new model or instructions set."""
        if not decision.strip():
            raise CampaignRefused("--decision needs a text")
        if skip_batch + repin_instructions + (repin_model is not None) > 1:
            raise CampaignRefused("give at most one of --skip-batch, --repin-instructions and "
                                  "--repin-model")
        with self.locked():
            state = self.state()
            halt = state.halt
            if halt is None:
                raise CampaignRefused("the campaign is not halted")
            event: dict[str, Any] = {"event": "resolved", "decision": decision,
                                     "batch": halt["batch"], "reason": halt.get("reason")}
            role = ROLE_OF_KIND[state.batches[halt["batch"]].kind]
            if skip_batch:
                if halt.get("reason") != "safeguardStop":
                    raise CampaignRefused("--skip-batch is for a safeguardStop halt (plan6 §3.5)")
                event["action"] = "skip-batch"
            elif repin_instructions:
                run = halt.get("run")
                if not run:
                    raise CampaignRefused("the halted run has no transcript to re-pin from")
                event.update(action="repin-instructions", role=role, instructions=[
                    [entry["path"], entry["sha256"]] for entry in run["instructions"]])
            elif repin_model is not None:
                event.update(self._repin_model(state, *repin_model))
            else:
                event["action"] = "retry"
            self._append(event)

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
        if state.dropped_seeds:
            lines.append("seeds dropped on a third hit: "
                         + json.dumps(state.generation_drops(), sort_keys=True))
        return lines


def _halt_text(halt: Mapping[str, Any]) -> str:
    return f"{halt.get('reason')} ({halt.get('voidClass')}) on {halt['batch']}"
