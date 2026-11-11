"""A fixture campaign for `finalize` (typed-decision-models U5.10b): rounds built straight into the
ledger, like `test_topup.py`'s, but with the run records the audit gives.

`run_records()` audits the transcript fixtures of `test_audit.py` once (a clean run, an incomplete
one whose agent transcript is cut after the prompt entry, a `spawnPromptMismatch` void from an empty
agent transcript, a no-`tool_result` incomplete run) and returns the `run_record` each yields; a
generation run is the clean record with the opus model, a label run the sonnet one. A seed's item
carries words unique to the seed, so no two items share an 8-token span unless a test plants it.
"""

from __future__ import annotations

import copy
import functools
import json
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest
from test_audit import audit_of, clean_run, drop_tool_result
from test_campaign import GUARD_COMMAND
from test_generation import hand_start
from test_topup import full_campaign
from test_votes import BOTH_AS_FIXTURE

from decisionmodels import audit, campaign, finalize, ingest
from decisionmodels.points import load_point
from decisionmodels.seeds import Seed

GENERATOR_MODEL, LABELER_MODEL = "claude-opus-5-5", "claude-sonnet-5-5"
HASHES = {
    "agentDefinition": {"generator": "a1" * 32, "labeler": "a2" * 32, "driver": "a3" * 32},
    "spawnTemplate": {"generator": "b1" * 32, "labeler": "b2" * 32, "driver": "b3" * 32},
    "requestTemplate": {"generator": "c1" * 32, "labeler": "c2" * 32},
    "rubric": "d1" * 32, "guardHook": "e1" * 32}
LABELS = {"guard-judge": ("true", "false"), "join-in-judge": ("true", "false"),
          "message-type-classifier": ("order", "question", "complaint", "escalation", "other")}
STRATA = {"guard-judge": ("understanding", "turns"), "join-in-judge": (None,),
          "message-type-classifier": (None,)}
Cell = tuple[str | None, str]


@functools.cache
def run_records() -> dict[str, dict[str, Any]]:
    """The audit's `run_record` for each fixture shape, plus the outcome and answer it needs."""
    out: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for name, edit in (("clean", None), ("cut", lambda r: setattr(r, "agent", r.agent[:1])),
                           ("empty", lambda r: setattr(r, "agent", [])),
                           ("incomplete", drop_tool_result)):
            run = clean_run()
            if edit is not None:
                edit(run)
            folder = Path(tmp) / name
            folder.mkdir()
            result = audit_of(folder, run)
            answer = (ingest.Ingested(answers={}, answer_sha256="a" * 64) if name == "clean"
                      else ingest.Void("noAnswer", (), "a" * 64) if result.ingests_answer
                      else None)
            verdict = audit.combine(result, answer)
            out[name] = {"record": audit.run_record(
                result, verdict, run_id="run_x", role="label", batch_id="bat_x", attempt=1,
                request_sha256="f" * 64), "action": result.action}
    return out


def run_record(kind: str, shape: str = "clean", *, run_id: str, batch: str,
               attempt: int) -> dict[str, Any]:
    """A fixture run's record for batch `batch` (`kind` `generate` or `label`)."""
    record = copy.deepcopy(run_records()[shape]["record"])
    record.update(runId=run_id, role=kind, batchId=batch, attempt=attempt)
    if kind == "generate" and record["model"] is not None:
        record["model"] = GENERATOR_MODEL
    return record


def unique(number: int, words: int = 12) -> str:
    """Words no other seed's text shares: tokens `s<number>w<k>`."""
    return " ".join(f"s{number}w{k}" for k in range(words))


def seed_number(seed_id: str) -> int:
    return int(seed_id.removeprefix("sd_"))


def make_seed(role: str, number: int, stratum: str | None, intent: str, **axes: Any) -> Seed:
    answer: Any = {"true": True, "false": False}.get(intent, intent)
    base = {"intendedAnswer": answer, "difficulty": "clear", **axes}
    if role == "message-type-classifier":
        base.setdefault("threadPosition", "opening" if number % 2 else "follow-up")
    if role == "guard-judge":
        base.setdefault("evidenceKind", stratum)
        base.setdefault("missingForm", "empty" if stratum == "understanding" else None)
        base.setdefault("condition", f"the user has stated budget number {number}")
    return Seed(f"sd_{number:012d}", role, stratum, base)


def item_for(role: str, seed: Seed, text: str | None = None) -> dict[str, Any]:
    """An item that conforms to the point's input fields."""
    body = text if text is not None else unique(seed_number(seed.seed_id))
    if role == "guard-judge":
        turns = seed.design_stratum == "turns"
        return {"condition": "the user has said which order the refund is for",
                "understanding": {} if turns else {"request": body, "known": [], "missing": []},
                "turns": [{"role": "user", "displayName": "Ana", "text": body}] if turns else [],
                "path": "turns" if turns else "understanding"}
    if role == "join-in-judge":
        return {"agentName": "Pastel", "priorTurns": [],
                "newMessage": {"role": "user", "displayName": "Bo", "text": body}}
    follow = seed.axes["threadPosition"] == "follow-up"
    return {"text": body, "priorTurn": {"speaker": "Pastel", "text": "Your order shipped."}
            if follow else None}


class Bench:
    """A `full` campaign for one point with tiny cells (`per_cell` eval items per cell, `floor`)."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, role: str, *,
                 per_cell: int = 2, floor: int = 1, tranche: int | None = None) -> None:
        self.tmp_path, self.monkeypatch, self.role = tmp_path, monkeypatch, role
        monkeypatch.setattr(campaign, "cell_target", lambda _role: 2 * per_cell)
        monkeypatch.setattr(finalize, "eval_floor", lambda _role: floor)
        if tranche is not None:
            monkeypatch.setattr(campaign, "TRANCHE_ITEMS", tranche)
        stub = tmp_path / "data" / "message-type-classifier" / "run" / "generation.json"
        stub.parent.mkdir(parents=True)
        stub.write_text("{}", "utf-8")            # join-in's predecessor is finalized
        self.camp = full_campaign(tmp_path, monkeypatch, role)
        self.install()
        self.data = tmp_path / "out"
        self.side = tmp_path / "side-check.txt"
        self.side.write_text("# none\n", "utf-8")
        self.texts: dict[str, str] = {}
        self.item_for: Callable[[Seed], dict[str, Any]] = lambda seed: item_for(
            role, seed, self.texts.get(seed.seed_id))

    def install(self, request_generator: str | None = None) -> None:
        """An `expectations` event with `promptSha256` (a later one is a second install)."""
        hashes = copy.deepcopy(HASHES)
        if request_generator is not None:
            hashes["requestTemplate"]["generator"] = request_generator
        roles = {**BOTH_AS_FIXTURE,
                 "generator": {**BOTH_AS_FIXTURE["generator"], "effort": "medium"}}
        self.camp.record_expectations(guard_command=GUARD_COMMAND, roles=roles,
                                      prompt_sha256=hashes)

    # -- seeds and rounds -------------------------------------------------------------------

    def seeds(self, counts: dict[Cell, int], start: int = 1,
              label_of: Callable[[Seed], str] | None = None) -> list[Seed]:
        """`counts[(stratum, intent)]` seeds of each cell, numbered from `start`."""
        out, number = [], start
        for (stratum, intent), n in counts.items():
            for _ in range(n):
                out.append(make_seed(self.role, number, stratum, intent))
                number += 1
        return out

    def start(self, seeds: Sequence[Seed], gen_round: int = 1) -> None:
        """`begin_round`'s records for hand-built seeds: the draw, `round-start`, balanced runs."""
        ids = hand_start(self.camp, seeds, gen_round)
        for run in campaign.balanced_runs(ids):
            self.camp.queue_batch("generate", run, generation="first")

    # -- completing batches -----------------------------------------------------------------

    def complete(self, batch_id: str, events: Callable[[str], list[dict[str, Any]]], *,
                 shape: str = "clean", outcome: str = "ingested", **extra: Any) -> str:
        """One attempt of `batch_id`: `issued`, the run's own events, `outcome` (with the audit
        record of `shape`). Returns the opaque name."""
        camp = self.camp
        batch = camp.state().batches[batch_id]
        attempt = len(batch.attempts) + 1
        opaque = f"op_{batch_id}_{attempt}"
        camp._append({"event": "issued", "batch": batch_id, "opaque": opaque,
                      "sessionId": f"s_{opaque}", "attempt": attempt, "requestSha256": "0" * 64,
                      "keys": {f"k{n}": ref for n, ref in enumerate(batch.entries)}})
        for event in events(opaque):
            camp._append(event)
        run = run_record(batch.kind, shape, run_id=f"run_{len(camp.state().attempts):04d}",
                         batch=batch_id, attempt=attempt)
        camp._append({"event": "outcome", "batch": batch_id, "opaque": opaque,
                      "outcome": outcome, "voidClass": run["voidClass"], "details": [],
                      "parts": [], "produced": True, "run": run, **extra})
        return opaque

    def generate(self, hits: Callable[[Seed, int], tuple[str, str | None] | None] | None = None
                 ) -> int:
        """Every pending generation batch ingested (and regenerations, while hits queue them).
        `hits(seed, n)` is `(event kind, form)` for the seed's `n`th generation, else accepted."""
        done = 0
        while True:
            pending = [b for b in self.camp.state().batches.values()
                       if b.kind == "generate" and b.state == "pending"]
            if not pending:
                if self.camp.queue_regeneration():
                    continue
                return done
            for batch in pending:
                self.complete(batch.batch_id, self._generation_events(batch, hits))
                done += 1

    def _generation_events(self, batch: campaign.Batch, hits: Any
                           ) -> Callable[[str], list[dict[str, Any]]]:
        state = self.camp.state()

        def events(opaque: str) -> list[dict[str, Any]]:
            accepted, rejected = [], []
            for seed_id in batch.entries:
                seed = state.seeds[seed_id]
                nth = len(state.hits.get(seed_id, ())) + 1
                hit = hits(seed, nth) if hits else None
                if hit is None:
                    accepted.append({"seedId": seed_id, "genRunId": batch.gen_run_id,
                                     "genRound": state.seed_round[seed_id],
                                     "item": self.item_for(seed)})
                else:
                    kind, form = hit
                    reason = {"screen-hit": "screen", "form-mismatch": "formMismatch",
                              "binding-mismatch": "bindingMismatch"}[kind]
                    rejected.append({"event": kind, "opaque": opaque, "batch": batch.batch_id,
                                     "seedId": seed_id, "genRunId": batch.gen_run_id,
                                     "reasons": [reason], "detail": "meta:x", "form": form,
                                     "firstPass": batch.extra.get("generation") == "first"})
            return [{"event": "generated", "opaque": opaque, "batch": batch.batch_id,
                     "items": accepted}, *rejected]
        return events

    def vote(self, answer: Callable[[campaign.State, str, int], Any]) -> int:
        """`queue_votes` and complete label batches until none is left; `answer(state, ref, vote)`
        is each vote."""
        done = 0
        while True:
            todo = [b for b in self.camp.state().batches.values()
                    if b.kind == "label" and b.state == "pending"]
            if not todo:
                if not self.camp.queue_votes():
                    return done
                continue
            for batch in todo:
                state = self.camp.state()
                self.complete(batch.batch_id, lambda opaque, batch=batch, state=state: [
                    {"event": "votes", "opaque": opaque, "batch": batch.batch_id,
                     "genRound": batch.extra["genRound"], "vote": batch.extra["vote"],
                     "answers": {ref: answer(state, ref, batch.extra["vote"])
                                 for ref in batch.entries}}])
                done += 1

    def agree(self, state: campaign.State, ref: str, vote: int) -> Any:
        """The labeller agrees with the seed's intent."""
        return state.seeds[ref].axes["intendedAnswer"]

    # -- finalize ---------------------------------------------------------------------------

    def finalize(self, version: str = "v1", **kwargs: Any) -> list[str]:
        kwargs.setdefault("side_check_file", self.side)
        kwargs.setdefault("rng_seed", 20261110)
        return finalize.finalize(self.camp, version=version, data_dir=self.data, **kwargs)

    def dataset(self, version: str = "v1") -> dict[str, Any]:
        folder = self.data / self.role / version
        return {"items": [json.loads(line) for line in
                          (folder / "items.jsonl").read_text("utf-8").splitlines()],
                "splits": json.loads((folder / "splits.json").read_text("utf-8")),
                "generation": json.loads((folder / "generation.json").read_text("utf-8")),
                "diagnostics": json.loads((folder / "diagnostics.json").read_text("utf-8"))}

    def full_round(self, per_cell_seeds: int = 5, *, start: int = 1) -> list[Seed]:
        """One round with `per_cell_seeds` seeds per cell, generated and voted unanimously."""
        counts = {(stratum, label): per_cell_seeds for stratum in STRATA[self.role]
                  for label in LABELS[self.role]}
        seeds = self.seeds(counts, start)
        self.start(seeds)
        self.generate()
        self.vote(self.agree)
        return seeds


def point_cells(role: str) -> list[Cell]:
    return finalize.all_cells(role)


def definition(role: str) -> Any:
    return load_point(role).definition
