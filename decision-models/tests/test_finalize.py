"""`driver finalize` (typed-decision-models U5.10b; plan4 `docs/plans/typed-decision-models4.md`
§3.6 "Cells, then splits", §3.7 and its §5 row U5.10; plan6 `typed-decision-models6.md` §3.7, §3.8
"U5.10 hand-offs" and its §5 row U5.10).

Campaigns are built by `finalize_bench.Bench` (rounds in the ledger, run records from the audit
over the transcript fixtures) with tiny cells, so a dataset costs a few dozen events. Numbers in
the assertions are planted and distinct (a count under the wrong `dropped` key, a list under the
wrong split, a cell keyed by intent instead of the majority label each give a different figure).
`shape_errors` is plan6 §3.7's comparison of `finalize`'s output against the contract vector's
line for the role (`tests/fixtures/decision-contract/dataset-shape.jsonl`).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import finalize_bench
import pytest
from finalize_bench import (
    GENERATOR_MODEL,
    HASHES,
    LABELER_MODEL,
    LABELS,
    STRATA,
    Bench,
    make_seed,
    unique,
)

from decisionmodels import campaign, diagnostics, driver, finalize, shingles, split
from decisionmodels.campaign import Campaign, CampaignRefused
from decisionmodels.points import ROLES

VECTOR = {line["role"]: line for line in (json.loads(raw) for raw in (
    Path(__file__).parent / "fixtures" / "decision-contract" / "dataset-shape.jsonl"
).read_text("utf-8").splitlines() if raw)}


def cell_lines(lines: list[str]) -> set[str]:
    """Each printed cell line up to its refill figures (` | eval-assigned ...`)."""
    return {line.split(" | ")[0] for line in lines if line.startswith("  cell")}


# --- §3.7's shape comparison ---------------------------------------------------------------------


def jtype(value: Any) -> str:
    """`json.loads`' types; a boolean is never an integer."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    for name, kind in (("integer", int), ("number", float), ("string", str), ("array", list),
                       ("object", dict)):
        if isinstance(value, kind):
            return name
    raise AssertionError(value)


def entry_errors(where: str, entries: list[Any], vector: list[Any]) -> list[str]:
    """A data-sized container: each output entry against all of the vector's entries."""
    errors: list[str] = []
    for number, entry in enumerate(entries):
        spot = f"{where}[{number}]"
        if not vector:
            errors.append(f"{spot}: no counterpart in the vector")
        elif isinstance(entry, dict):
            peers = [v for v in vector if isinstance(v, dict)]
            if not any(set(entry) == set(v) for v in peers):
                errors.append(f"{spot}: keys {sorted(entry)} are not the vector's")
                continue
            for key, value in entry.items():
                if jtype(value) not in {jtype(v[key]) for v in peers if key in v}:
                    errors.append(f"{spot}.{key}: type {jtype(value)} is not one the vector shows")
        elif jtype(entry) not in {jtype(v) for v in vector}:
            errors.append(f"{spot}: type {jtype(entry)} is not one the vector shows")
    return errors


def tree_errors(where: str, out: Any, vec: Any, data: dict[str, Any]) -> list[str]:
    """Outside the data-sized containers: equal key sets, recursively, and equal types."""
    if jtype(out) != jtype(vec):
        return [f"{where}: type {jtype(out)}, vector {jtype(vec)}"]
    if not isinstance(out, dict):
        return []
    if set(out) != set(vec):
        return [f"{where}: keys {sorted(out)}, vector {sorted(vec)}"]
    errors: list[str] = []
    for key in out:
        path = f"{where}.{key}"
        if path in data:
            errors += data[path](out[key], vec[key])
        else:
            errors += tree_errors(path, out[key], vec[key], data)
    return errors


def runs_errors(out: list[Any], vec: list[Any]) -> list[str]:
    errors = entry_errors("runs", out, vec)
    for number, run in enumerate(out):
        if isinstance(run, dict) and isinstance(run.get("tokens"), dict) and not any(
                set(run["tokens"]) == set(v["tokens"]) for v in vec):
            errors.append(f"runs[{number}].tokens: keys {sorted(run['tokens'])}")
    errors += entry_errors("runs[].instructions",
                           [i for run in out if isinstance(run, dict)
                            for i in (run.get("instructions") or [])],
                           [i for run in vec for i in run["instructions"]])
    return errors


def shape_errors(output: dict[str, Any], line: dict[str, Any]) -> list[str]:
    """`finalize`'s `generation`, `items` and `splits` against the vector line (plan6 §3.7)."""
    data = {"generation.dropped.formMismatch": lambda o, v: (
                [] if isinstance(o, dict) and jtype(v) == "object"
                else ["formMismatch is not an object"]) + entry_errors(
                    "formMismatch", list(o.values()) if isinstance(o, dict) else [],
                    list(v.values())),
            "generation.runs": runs_errors}
    errors = tree_errors("generation", output["generation"], line["generation"], data)
    errors += entry_errors("items", output["items"], line["items"])
    vector_votes = [set(item["labelVotes"]) for item in line["items"]]
    errors += [f"items[{n}].labelVotes: keys {sorted(item['labelVotes'])}"
               for n, item in enumerate(output["items"])
               if isinstance(item.get("labelVotes"), dict)
               and set(item["labelVotes"]) not in vector_votes]
    splits, vec = output["splits"], line["splits"]
    errors += tree_errors("splits", splits, vec, {f"splits.{name}": (
        lambda o, v, name=name: entry_errors(name, o, v)) for name in ("eval", "train",
                                                                       "calibration")})
    return errors


def shape_of(bench: Bench, version: str = "v1") -> list[str]:
    data = bench.dataset(version)
    return shape_errors(data, VECTOR[bench.role])


# --- campaigns -----------------------------------------------------------------------------------


def shape_bench(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, role: str, *,
                form_dropped: bool = False) -> Bench:
    """A finished campaign whose first run took four attempts: an incomplete one (no tool result,
    re-issued), a `maxTurns` void cut after the prompt (four facts null), and a
    `spawnPromptMismatch` void (all five null; halted, resolved, retried)."""
    bench = Bench(tmp_path, monkeypatch, role)
    counts = {(stratum, label): 5 for stratum in STRATA[role] for label in LABELS[role]}
    bench.start(bench.seeds(counts))
    (batch, *_) = bench.camp.state().batches
    bench.complete(batch, lambda opaque: [], shape="incomplete", outcome="reissued")
    bench.complete(batch, lambda opaque: [], shape="cut", outcome="void")
    bench.complete(batch, lambda opaque: [], shape="empty", outcome="halted",
                   reason="spawnPromptMismatch")
    bench.camp.resolve(decision="retry the mismatch")
    def hits(seed: Any, nth: int) -> tuple[str, str | None] | None:
        return ("form-mismatch", "immaterial") if form_dropped and seed.seed_id.endswith(
            "003") else None
    bench.generate(hits)
    bench.vote(bench.agree)
    return bench


@pytest.mark.parametrize("role", ROLES)
def test_a_finalized_dataset_has_the_vectors_shape_for_its_role(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, role: str) -> None:
    bench = shape_bench(tmp_path, monkeypatch, role)
    bench.finalize()
    assert shape_of(bench) == []
    data = bench.dataset()
    statuses = {run["status"] for run in data["generation"]["runs"]}
    assert {"ingested", "incomplete", "void"} <= statuses            # the audit's three shapes
    empty = next(r for r in data["generation"]["runs"] if r["voidReason"] == "spawnPromptMismatch")
    assert [empty[k] for k in ("claudeCodeVersion", "model", "sessionContextSha256", "thinking",
                               "effort")] == [None] * 5
    assert data["generation"]["dropped"]["formMismatch"] == {}


def test_a_campaign_that_dropped_a_form_matches_the_guard_line(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bench = shape_bench(tmp_path, monkeypatch, "guard-judge", form_dropped=True)
    bench.finalize()
    assert shape_of(bench) == []
    assert bench.dataset()["generation"]["dropped"]["formMismatch"] == {"immaterial": 1}


def mutate(bench: Bench, edit: Callable[[dict[str, Any]], None]) -> list[str]:
    data = bench.dataset()
    edit(data)
    return shape_errors(data, VECTOR[bench.role])


def finished(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, role: str = "guard-judge") -> Bench:
    bench = Bench(tmp_path, monkeypatch, role)
    bench.full_round()
    bench.finalize()
    return bench


def test_the_comparison_reddens_on_each_planted_departure(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bench = finished(tmp_path, monkeypatch)
    assert mutate(bench, lambda d: None) == []
    assert mutate(bench, lambda d: d["items"][0]["labelVotes"].__setitem__(
        "True", d["items"][0]["labelVotes"].pop("true")))
    assert mutate(bench, lambda d: d["splits"].pop("rngSeed"))
    assert mutate(bench, lambda d: d["generation"]["runs"][0].__setitem__("attempt", "1"))
    assert mutate(bench, lambda d: d["items"][0].__setitem__("genRound", True))
    assert mutate(bench, lambda d: d["generation"]["runs"][0]["tokens"].pop("output"))
    assert mutate(bench, lambda d: d["generation"].__setitem__("itemCount", "3"))
    assert mutate(bench, lambda d: d["generation"]["dropped"].pop("safeguardSkipped"))


def test_a_form_entry_on_a_join_in_output_has_no_counterpart(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bench = finished(tmp_path, monkeypatch, "join-in-judge")
    errors = mutate(bench, lambda d: d["generation"]["dropped"]["formMismatch"].update(empty=1))
    assert errors and "no counterpart" in errors[0]


def test_an_output_entry_is_compared_with_all_the_vectors_entries_not_the_first(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The vector's runs are one ingested and one incomplete: a null `model` is a type its second
    entry shows, so an incomplete run in the output is no departure."""
    bench = shape_bench(tmp_path, monkeypatch, "guard-judge")
    bench.finalize()
    assert any(r["model"] is None for r in bench.dataset()["generation"]["runs"])
    assert shape_of(bench) == []


def test_an_empty_train_list_is_not_refused_by_the_comparison(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bench = finished(tmp_path, monkeypatch)
    splits = bench.dataset()["splits"]
    assert splits["train"] == [] and VECTOR["guard-judge"]["splits"]["train"]
    assert shape_of(bench) == []


# --- the dataset ---------------------------------------------------------------------------------


def test_the_splits_are_split_make_splits_over_the_labelled_items_and_partition_items_jsonl(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bench = Bench(tmp_path, monkeypatch, "guard-judge")
    seeds = bench.full_round(5)
    bench.finalize()
    data = bench.dataset()
    state = bench.camp.state()
    expected = split.make_splits(
        [split.Candidate(s.seed_id, (s.design_stratum, "true" if s.axes["intendedAnswer"]
                                     else "false"),
                         shingles.evidence_text("guard-judge", state.items[s.seed_id]["item"]))
         for s in seeds], sizes={"eval": 2, "calibration": 2, "train": 0}, rng_seed=20261110)
    by_item = {item["itemId"]: item["seedId"] for item in data["items"]}
    assert [by_item[i] for i in data["splits"]["eval"]] == list(expected.eval)
    assert [by_item[i] for i in data["splits"]["calibration"]] == list(expected.calibration)
    assert data["splits"]["train"] == [] and data["splits"]["rngSeed"] == 20261110
    listed = data["splits"]["eval"] + data["splits"]["calibration"]
    assert sorted(listed) == sorted(by_item) and len(set(listed)) == len(listed) == 16
    assert [i["itemId"] for i in data["items"]] == sorted(by_item)        # ids number the kept
    assert [i["seedId"] for i in data["items"]] == sorted(i["seedId"] for i in data["items"])
    assert data["generation"]["itemCount"] == 16                          # 20 labelled, 4 surplus


def test_a_cell_is_keyed_by_the_majority_label_not_the_seeds_intent(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bench = Bench(tmp_path, monkeypatch, "guard-judge")
    counts = {(stratum, label): 8 for stratum in STRATA["guard-judge"]
              for label in ("true", "false")}
    seeds = bench.seeds(counts)
    bench.start(seeds)
    bench.generate()
    flipped = {s.seed_id for s in seeds if s.design_stratum == "turns"
               and s.axes["intendedAnswer"] is True and seed_index(s) % 2}
    bench.vote(lambda st, ref, vote: (not st.seeds[ref].axes["intendedAnswer"]
                                      if ref in flipped else st.seeds[ref].axes["intendedAnswer"]))
    lines = bench.finalize()
    data = bench.dataset()
    by_seed = {item["seedId"]: item for item in data["items"]}
    assert flipped and all(by_seed[s]["expected"] is False for s in flipped if s in by_seed)
    assert all(by_seed[s]["labelVotes"] == {"true": 0, "false": 3} for s in flipped if s in by_seed)
    # turns/true held 8 - |flipped| labelled, turns/false 8 + |flipped|: the surplus differs
    surplus = {"true": 8 - len(flipped) - 4, "false": 8 + len(flipped) - 4}
    for label, left in surplus.items():
        assert f"  cell turns / {label}: eval 2, calibration 2, surplus {left}" in cell_lines(lines)


def test_items_jsonl_has_no_raw_line_separator_and_the_text_round_trips(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """model-bench's importer reads the file with `str.splitlines()`: U+2028, U+0085 and U+2029
    must be escaped, so one line is one item."""
    bench = Bench(tmp_path, monkeypatch, "guard-judge")
    odd = "a\u2028b\u0085c\u2029d"
    bench.item_for = lambda seed: finalize_bench.item_for(
        "guard-judge", seed, f"{unique(seed_index(seed))} {odd}")
    bench.full_round(5)
    bench.finalize()
    raw = (bench.data / "guard-judge" / "v1" / "items.jsonl").read_text("utf-8")
    assert not any(char in raw for char in "\u2028\u0085\u2029")
    lines = raw.splitlines()
    assert len(lines) == raw.count("\n") == 16
    loaded = [json.loads(line) for line in lines]
    assert all(odd in json.dumps(item["understanding"] or item["turns"], ensure_ascii=False)
               for item in loaded)
    assert loaded == bench.dataset()["items"]


def test_an_item_carries_the_input_fields_and_no_other_key_of_the_generated_item(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bench = Bench(tmp_path, monkeypatch, "guard-judge")
    bench.item_for = lambda seed: {**finalize_bench.item_for("guard-judge", seed),
                                   "extraKey": "not an input field"}
    bench.full_round(5)
    bench.finalize()
    assert all("extraKey" not in item for item in bench.dataset()["items"])


def test_item_ids_count_from_one_under_the_roles_prefix(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for role, prefix in (("guard-judge", "gj"), ("join-in-judge", "ji"),
                         ("message-type-classifier", "mt")):
        bench = Bench(tmp_path / role, monkeypatch, role)
        bench.full_round(5)
        bench.finalize()
        ids = [item["itemId"] for item in bench.dataset()["items"]]
        assert ids == [f"{prefix}-{n:04d}" for n in range(1, len(ids) + 1)] and ids


def test_gen_round_is_the_round_the_item_was_generated_in(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bench = short_bench(tmp_path, monkeypatch, turns_true=5)
    bench.start(bench.seeds({("turns", "true"): 4}, start=900), gen_round=2)   # the top-up
    bench.generate()
    bench.vote(bench.agree)
    bench.finalize()
    state = bench.camp.state()
    items = bench.dataset()["items"]
    rounds = {item["seedId"]: item["genRound"] for item in items}
    assert set(rounds.values()) == {1, 2}
    assert all(rounds[s] == state.items[s]["genRound"] == (2 if seed_index_of(s) >= 900 else 1)
               for s in rounds)
    assert all(item["genRunId"] == state.items[item["seedId"]]["genRunId"] for item in items)
    assert len({item["genRunId"] for item in items}) > 2


def seed_index(seed: Any) -> int:
    return seed_index_of(seed.seed_id)


def seed_index_of(seed_id: str) -> int:
    return int(seed_id.removeprefix("sd_"))


def test_expected_and_the_provenance_keys_are_the_majority_and_the_runs(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bench = Bench(tmp_path, monkeypatch, "message-type-classifier")
    bench.full_round(5)
    bench.finalize()
    data = bench.dataset()
    state = bench.camp.state()
    for item in data["items"]:
        record = state.items[item["seedId"]]
        assert (item["genRunId"], item["genRound"]) == (record["genRunId"], 1)
        assert item["origin"] == "synthetic"
        assert item["expected"] == state.seeds[item["seedId"]].axes["intendedAnswer"]
        assert item["labelVotes"][item["expected"]] == 3 and sum(item["labelVotes"].values()) == 3
        assert item["threadPosition"] == state.seeds[item["seedId"]].axes["threadPosition"]
        assert (item["priorTurn"] is None) == (item["threadPosition"] == "opening")
        assert set(item) == {"itemId", "text", "priorTurn", "expected", "origin", "genRunId",
                             "genRound", "seedId", "labelVotes", "threadPosition"}
    assert {i["threadPosition"] for i in data["items"]} == {"opening", "follow-up"}


def test_generation_json_records_models_efforts_hashes_and_the_runs(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bench = Bench(tmp_path, monkeypatch, "guard-judge")
    bench.full_round(5)
    bench.finalize()
    gen = bench.dataset()["generation"]
    state = bench.camp.state()
    assert gen["generator"] == {"surface": "claude-code-subagent", "tier": "opus",
                                "model": GENERATOR_MODEL, "effort": "medium"}
    assert gen["labeler"] == {"surface": "claude-code-subagent", "tier": "sonnet",
                              "model": LABELER_MODEL, "effort": "high"}
    assert (gen["votesPerItem"], gen["labelerTemperature"]) == (3, None)
    assert gen["labelBatchCap"] == state.init["labelBatchCap"] == 25
    assert gen["generatedAt"].endswith("Z") and gen["generatedAt"].startswith("2026-11-0")
    assert gen["promptSha256"] == {**HASHES, "rubric": campaign.rubric_hash("guard-judge")}
    assert {run["role"] for run in gen["runs"]} == {"generate", "label"}
    assert len(gen["runs"]) == sum(len(b.attempts) for b in state.batches.values())
    assert [r["runId"] for r in gen["runs"]] == sorted(r["runId"] for r in gen["runs"])


def test_the_runs_are_listed_in_run_id_order_whatever_the_order_they_ran_in(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real = finalize_bench.run_record
    def backwards(kind: str, shape: str = "clean", **kw: Any) -> dict[str, Any]:
        number = int(kw["run_id"].removeprefix("run_"))
        return {**real(kind, shape, **kw), "runId": f"run_{9999 - number:04d}"}
    monkeypatch.setattr(finalize_bench, "run_record", backwards)
    bench = Bench(tmp_path, monkeypatch, "guard-judge")
    bench.full_round(5)
    state = bench.camp.state()
    ran = [a.outcome["run"]["runId"] for a in state.attempts]
    assert ran != sorted(ran)                                        # the ledger is not in order
    bench.finalize()
    ids = [run["runId"] for run in bench.dataset()["generation"]["runs"]]
    assert ids == sorted(ran) and len(ids) == len(ran)


def test_the_dataset_hashes_are_the_last_issued_runs_and_the_rubric_the_current_one(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bench = Bench(tmp_path, monkeypatch, "guard-judge")
    bench.full_round(5)
    bench.install(request_generator="f0" * 32)         # after the last run: not in force at issue
    bench.finalize()
    hashes = bench.dataset()["generation"]["promptSha256"]
    assert hashes["requestTemplate"]["generator"] == HASHES["requestTemplate"]["generator"]
    monkeypatch.setattr(campaign, "rubric_hash", lambda role: "9" * 64)
    bench.finalize("v2", rng_seed=1)
    assert bench.dataset("v2")["generation"]["promptSha256"]["rubric"] == "9" * 64


def test_the_side_check_drop_surplus_and_split_drop_land_under_their_own_keys(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every dropped count is distinct, so a count under the wrong key is a red test. (Floor 0:
    the cell the shingle drop empties is not what this test is about.)"""
    bench = Bench(tmp_path, monkeypatch, "guard-judge", per_cell=1, floor=0)
    for name in ("HALT_PERCENT", "POOL_PERCENT"):
        monkeypatch.setattr(campaign, name, 100)         # no template halt: the hits are planted
    shared = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
    texts = {}
    cells = {("understanding", "true"): 2, ("understanding", "false"): 4,
             ("turns", "true"): 5, ("turns", "false"): 2}
    seeds = bench.seeds(cells)
    first = {c: [s for s in seeds if (s.design_stratum, "true" if s.axes["intendedAnswer"]
                                      else "false") == c] for c in cells}
    for seed in first[("understanding", "true")]:        # both items of a 2-item cell share text
        texts[seed.seed_id] = shared
    planted = first[("understanding", "false")][:2]      # two side-check hits
    for seed in planted:
        texts[seed.seed_id] = f"planted {unique(seed_index(seed))}"
    bench.side.write_text("\n".join(sorted(
        h for s in planted for h in shingles.shingle_hashes(texts[s.seed_id]))) + "\n", "utf-8")
    bench.texts.update(texts)
    extra = bench.seeds({("turns", "true"): 3}, start=100)         # screen hits: 3 seeds dropped
    forms = bench.seeds({("turns", "false"): 4}, start=200)
    binds = bench.seeds({("turns", "true"): 5}, start=300)
    bench.start([*seeds, *extra, *forms, *binds])
    nth = {"screen": {s.seed_id for s in extra}, "form": {s.seed_id for s in forms},
           "bind": {s.seed_id for s in binds}}

    def hits(seed: Any, n: int) -> tuple[str, str | None] | None:
        if seed.seed_id in nth["screen"]:
            return "screen-hit", None
        if seed.seed_id in nth["form"]:
            return "form-mismatch", "empty" if seed_index(seed) < 203 else "immaterial"
        if seed.seed_id in nth["bind"]:
            return "binding-mismatch", None
        return None
    bench.generate(hits)
    bench.vote(bench.agree)
    bench.finalize()
    dropped = bench.dataset()["generation"]["dropped"]
    assert dropped == {
        "voteTie": 0, "shingleSplit": 1, "shingleSideCheck": 2, "screenHits": 3,
        "retryCapExceeded": {"seeds": 0, "items": 0},
        "safeguardSkipped": {"seeds": 0, "items": 0},
        "formMismatch": {"empty": 3, "immaterial": 1}, "bindingMismatch": 5}
    data = bench.dataset()
    kept = {i["seedId"] for i in data["items"]}
    assert not kept & {s.seed_id for s in planted}                  # the side-check hits are out
    assert len([s for s in first[("turns", "true")] if s.seed_id in kept]) == 2   # 5 -> 1 + 1 per
    # understanding/true: 2 items sharing text: eval dropped, calibration kept
    assert len(data["splits"]["eval"]) + len(data["splits"]["calibration"]) == len(kept)


def test_batches_given_up_are_counted_under_their_own_dropped_keys(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bench = Bench(tmp_path, monkeypatch, "guard-judge")
    bench.full_round(5)
    two = bench.seeds({("turns", "true"): 2}, start=500)
    bench.start(two, gen_round=2)
    (batch,) = [b for b in bench.camp.state().batches.values()
                if b.state == "pending" and b.kind == "generate"]
    for _ in range(3):                                   # three counted voids: retryCapExceeded
        bench.complete(batch.batch_id, lambda opaque: [], shape="cut", outcome="void")
    three = bench.seeds({("turns", "false"): 3}, start=600)
    bench.start(three, gen_round=3)
    (skipped,) = [b for b in bench.camp.state().batches.values() if b.state == "pending"]
    bench.complete(skipped.batch_id, lambda opaque: [], shape="empty", outcome="halted",
                   reason="safeguardStop")
    bench.camp.resolve(decision="skip the flagged run", skip_batch=True)
    bench.finalize()
    dropped = bench.dataset()["generation"]["dropped"]
    assert dropped["retryCapExceeded"] == {"seeds": 2, "items": 0}
    assert dropped["safeguardSkipped"] == {"seeds": 3, "items": 0}


def test_vote_ties_are_counted_and_their_items_are_in_no_split(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bench = Bench(tmp_path, monkeypatch, "message-type-classifier", per_cell=1)
    seeds = bench.seeds({(None, label): 14 for label in LABELS["message-type-classifier"]})
    bench.start(seeds)                                   # 70 items: 2 ties are 2.9%, under 3%
    bench.generate()
    ties = {seeds[0].seed_id, seeds[37].seed_id}
    labels_ = LABELS["message-type-classifier"]
    bench.vote(lambda st, ref, vote: labels_[vote - 1] if ref in ties
               else st.seeds[ref].axes["intendedAnswer"])
    bench.finalize()
    data = bench.dataset()
    assert data["generation"]["dropped"]["voteTie"] == 2
    assert not {i["seedId"] for i in data["items"]} & ties


def test_the_diagnostics_json_is_written_beside_the_dataset(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bench = finished(tmp_path, monkeypatch)
    folder = bench.data / "guard-judge" / "v1"
    assert sorted(p.name for p in folder.iterdir()) == [
        "diagnostics.json", "generation.json", "items.jsonl", "splits.json"]
    assert bench.dataset()["diagnostics"]["point"] == "guard-judge"


# --- the hand-off prompt examples are counted over the eval split --------------------------------


def test_prompt_example_counts_run_over_the_eval_split_only(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    role = "message-type-classifier"
    bench = Bench(tmp_path, monkeypatch, role, per_cell=1)
    example = diagnostics.HANDOFF_EXAMPLES[role][1]                    # "Where is my order?"
    seeds = bench.seeds({(None, label): 3 for label in LABELS[role]})
    target = seeds[0]
    bench.texts[target.seed_id] = f"{example} {unique(1)}"
    bench.start(seeds)
    bench.generate()
    bench.vote(bench.agree)
    state = bench.camp.state()
    candidates = [split.Candidate(
        s.seed_id, (None, s.axes["intendedAnswer"]),
        shingles.evidence_text(role, state.items[s.seed_id]["item"])) for s in seeds]
    # a draw that puts the planted item in calibration, none of it in eval
    for rng_seed in range(1, 200):
        result = split.make_splits(candidates, sizes={"eval": 1, "calibration": 1, "train": 0},
                                   rng_seed=rng_seed)
        if target.seed_id in result.calibration:
            break
    else:
        raise AssertionError("no draw puts the item in calibration")
    bench.finalize(rng_seed=rng_seed)
    data = bench.dataset()
    in_eval = {i["seedId"] for i in data["items"] if i["itemId"] in data["splits"]["eval"]}
    assert target.seed_id not in in_eval
    assert diagnostics.build(state)["promptExamples"]["matching"] == 1     # all accepted items
    assert data["diagnostics"]["promptExamples"]["matching"] == 0          # the eval split
    assert data["diagnostics"]["promptExamples"]["items"] == len(in_eval)


def test_an_eval_item_carrying_an_example_is_counted(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    role = "message-type-classifier"
    bench = Bench(tmp_path, monkeypatch, role, per_cell=1)
    example = diagnostics.HANDOFF_EXAMPLES[role][1]
    seeds = bench.seeds({(None, label): 3 for label in LABELS[role]})
    target = seeds[0]
    bench.texts[target.seed_id] = f"{example} {unique(1)}"
    bench.start(seeds)
    bench.generate()
    bench.vote(bench.agree)
    state = bench.camp.state()
    candidates = [split.Candidate(
        s.seed_id, (None, s.axes["intendedAnswer"]),
        shingles.evidence_text(role, state.items[s.seed_id]["item"])) for s in seeds]
    for rng_seed in range(1, 200):
        if target.seed_id in split.make_splits(
                candidates, sizes={"eval": 1, "calibration": 1, "train": 0},
                rng_seed=rng_seed).eval:
            break
    bench.finalize(rng_seed=rng_seed)
    assert bench.dataset()["diagnostics"]["promptExamples"]["matching"] == 1


# --- refusals ------------------------------------------------------------------------------------


def ledger(bench: Bench) -> bytes:
    return bench.camp.ledger_path.read_bytes()


def refused(bench: Bench, match: str, *, floor_short: bool = False, **kwargs: Any) -> None:
    """`finalize` is refused with `match` and no dataset is written. The ledger is byte-equal,
    except that a floor refusal (`floor_short=True`) appends exactly one `floor-short` event."""
    before = ledger(bench)
    with pytest.raises(CampaignRefused, match=match):
        bench.finalize("vx", **kwargs)
    after = ledger(bench)
    if floor_short:
        assert after.startswith(before)
        appended = after[len(before):].splitlines()
        assert len(appended) == 1 and json.loads(appended[0])["event"] == "floor-short"
    else:
        assert after == before
    assert not list(bench.data.glob("*/*")) if bench.data.exists() else True


def test_finalize_refuses_a_campaign_that_is_not_full(tmp_path: Path) -> None:
    camp = Campaign(tmp_path / "camp")
    camp.init(point="guard-judge", phase="pilot")
    with pytest.raises(CampaignRefused, match="full campaign"):
        finalize.finalize(camp, version="v1", data_dir=tmp_path / "out")


def test_finalize_refuses_an_unfinished_campaign_and_a_halted_one(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bench = Bench(tmp_path, monkeypatch, "guard-judge")
    bench.start(bench.seeds({("turns", "true"): 5}))
    refused(bench, "not finished")
    bench.generate()
    refused(bench, "unlabelled")
    bench.vote(bench.agree)
    bench.complete("bat_0001", lambda opaque: [], shape="empty", outcome="halted",
                   reason="safeguardStop")
    refused(bench, "halted")


def test_a_dataset_version_is_never_overwritten_and_its_name_is_plain(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bench = finished(tmp_path, monkeypatch)
    before = ledger(bench)
    with pytest.raises(CampaignRefused, match="never overwritten"):
        bench.finalize("v1", rng_seed=5)
    assert ledger(bench) == before
    with pytest.raises(CampaignRefused, match="plain name"):
        bench.finalize("../v1")
    with pytest.raises(CampaignRefused, match="plain name"):
        bench.finalize("v2.partial")


def test_finalize_records_its_rng_seed_in_the_ledger_and_never_reuses_a_held_one(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bench = Bench(tmp_path, monkeypatch, "guard-judge")
    bench.full_round(5)
    bench.finalize()
    (event,) = [e for e in bench.camp.events() if e["event"] == "finalized"]
    assert (event["datasetVersion"], event["rngSeed"], event["items"]) == ("v1", 20261110, 16)
    assert 20261110 in bench.camp.state().held_rng_seeds
    with pytest.raises(CampaignRefused, match="already held"):
        bench.finalize("v2")


def test_a_stale_partial_directory_from_a_crash_before_the_rename_is_removed_on_retry(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bench = Bench(tmp_path, monkeypatch, "guard-judge")
    bench.full_round(5)
    stale = bench.data / "guard-judge" / "v1.partial"
    stale.mkdir(parents=True)
    (stale / "junk").write_text("x", "utf-8")
    bench.finalize()
    assert not stale.exists() and (bench.data / "guard-judge" / "v1" / "items.jsonl").is_file()


def test_a_crash_between_the_rename_and_the_event_leaves_an_orphan_the_retry_diagnoses(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bench = Bench(tmp_path, monkeypatch, "guard-judge")
    bench.full_round(5)
    real = bench.camp._append

    def crash(event: dict[str, Any]) -> None:
        if event["event"] == "finalized":
            raise OSError("disk gone")
        real(event)
    monkeypatch.setattr(bench.camp, "_append", crash)
    with pytest.raises(OSError, match="disk gone"):
        bench.finalize()
    monkeypatch.setattr(bench.camp, "_append", real)
    orphan = bench.data / "guard-judge" / "v1"
    assert (orphan / "items.jsonl").is_file()
    assert not [e for e in bench.camp.events() if e["event"] == "finalized"]
    before = (orphan / "items.jsonl").read_bytes()
    with pytest.raises(CampaignRefused, match="never overwritten.*no `finalized` event.*move it"):
        bench.finalize()
    assert (orphan / "items.jsonl").read_bytes() == before            # never overwritten
    assert bench.finalize("v2")[0].startswith("finalized guard-judge as v2")
    with pytest.raises(CampaignRefused) as recorded:                  # a recorded one: no hint
        bench.finalize("v2", rng_seed=3)
    assert "never overwritten" in str(recorded.value) and "interrupted" not in str(recorded.value)


def test_more_than_one_ingested_model_in_a_role_is_refused_naming_both(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bench = short_bench(tmp_path, monkeypatch, turns_true=8)
    monkeypatch.setattr("finalize_bench.GENERATOR_MODEL", "claude-opus-5-6")
    bench.start(bench.seeds({("turns", "true"): 2}, start=900), gen_round=2)
    bench.generate()
    bench.vote(bench.agree)
    refused(bench, r"generator's ingested runs name \['claude-opus-5-5', 'claude-opus-5-6'\]")


def test_a_run_that_was_not_ingested_names_no_model_for_the_dataset(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A `modelMismatch` void run whose model is foreign is no evidence of what generated."""
    bench = Bench(tmp_path, monkeypatch, "guard-judge")
    counts = {(stratum, label): 5 for stratum in STRATA["guard-judge"]
              for label in LABELS["guard-judge"]}
    bench.start(bench.seeds(counts))
    (batch, *_) = bench.camp.state().batches
    real = finalize_bench.run_record

    def foreign(kind: str, shape: str = "clean", **kwargs: Any) -> dict[str, Any]:
        record = real(kind, shape, **kwargs)
        if kwargs["batch"] == batch and kwargs["attempt"] == 1:
            record.update(status="void", voidReason="modelMismatch", model="claude-foreign-9")
        return record
    monkeypatch.setattr(finalize_bench, "run_record", foreign)
    bench.complete(batch, lambda opaque: [], outcome="void")
    monkeypatch.setattr(finalize_bench, "run_record", real)
    bench.generate()
    bench.vote(bench.agree)
    assert bench.finalize()[0].startswith("finalized")
    data = bench.dataset()
    assert data["generation"]["generator"]["model"] == GENERATOR_MODEL
    assert "claude-foreign-9" in {run["model"] for run in data["generation"]["runs"]}


def test_two_models_in_one_role_or_one_model_for_both_are_refused(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("finalize_bench.GENERATOR_MODEL", LABELER_MODEL)
    bench = Bench(tmp_path, monkeypatch, "guard-judge")
    bench.full_round(5)
    refused(bench, "both claude-sonnet-5-5")


# --- --short and the decline ---------------------------------------------------------------------


def short_bench(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, turns_true: int,
                per_cell: int = 4, floor: int = 2) -> Bench:
    """Guard with three full cells of 8 and `turns/true` holding `turns_true` labelled items."""
    bench = Bench(tmp_path, monkeypatch, "guard-judge", per_cell=per_cell, floor=floor)
    counts = {("understanding", "true"): 8, ("understanding", "false"): 8,
              ("turns", "true"): turns_true, ("turns", "false"): 8}
    bench.start(bench.seeds(counts))
    bench.generate()
    bench.vote(bench.agree)
    return bench


def decline(bench: Bench) -> None:
    """A spend halt, then the stakeholder's no (`resolve --decline-topup`)."""
    bench.camp._append({"event": "spend-halt", "genRound": 2, "probe": False,
                        "projection": {"projectedTokens": 9.0, "roundOneTokens": 10}})
    bench.camp.resolve(decision="no top-up: spend", decline_topup=True)


def test_a_short_cell_with_no_decline_is_a_top_up_still_due(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bench = short_bench(tmp_path, monkeypatch, turns_true=5)
    refused(bench, "cells are short.*`driver run` plans the top-up")
    refused(bench, "--short needs a recorded", short=True)


def test_after_a_decline_the_refusal_names_short_and_reopen_topup(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bench = short_bench(tmp_path, monkeypatch, turns_true=5)
    decline(bench)
    refused(bench, "--short.*--reopen-topup")


def test_short_finalizes_when_every_cell_meets_its_floor_and_prints_every_n(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bench = short_bench(tmp_path, monkeypatch, turns_true=5)         # eval 4 + calibration 1
    decline(bench)
    with pytest.raises(CampaignRefused, match="below its floor of 2 .*turns.*calibration 1"):
        bench.finalize("vx", short=True)
    assert not (bench.data / "guard-judge" / "vx").exists()
    bench2 = short_bench(tmp_path / "b2", monkeypatch, turns_true=6)  # eval 4 + calibration 2
    decline(bench2)
    lines = bench2.finalize(short=True)
    assert lines[1] == "SHORT: finalized under a declined top-up; every cell meets its floor"
    assert "  cell turns / true: eval 4, calibration 2, surplus 0" in cell_lines(lines)
    assert "  cell understanding / true: eval 4, calibration 4, surplus 0" in cell_lines(lines)
    data = bench2.dataset()
    assert len(data["splits"]["eval"]) == 16 and len(data["splits"]["calibration"]) == 14


def test_an_eval_cell_the_shingle_drop_empties_is_under_its_floor_whatever_calibration_holds(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bench = Bench(tmp_path, monkeypatch, "guard-judge", per_cell=4, floor=2)
    counts = {("understanding", "true"): 8, ("understanding", "false"): 8,
              ("turns", "true"): 5, ("turns", "false"): 8}
    seeds = bench.seeds(counts)
    shared = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
    for seed in seeds:
        if seed.design_stratum == "understanding" and seed.axes["intendedAnswer"] is True:
            bench.texts[seed.seed_id] = shared           # every item of the cell repeats it
    bench.start(seeds)
    bench.generate()
    bench.vote(bench.agree)
    decline(bench)
    with pytest.raises(CampaignRefused, match=r"understanding\): eval 0, calibration 4"):
        bench.finalize(short=True)


def test_a_cell_under_its_floor_refuses_every_finalize_short_cell_or_not(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every cell has its target; the shingle drop empties one eval cell below the floor. The
    floor gates this finalize whatever `--short` says: nothing is written, and the one ledger
    change is the `floor-short` record (a cell whose eval was all dropped has no drop rate to
    size a top-up from: `need` is null)."""
    bench = Bench(tmp_path, monkeypatch, "guard-judge", per_cell=4, floor=2)
    counts = {(stratum, label): 8 for stratum in STRATA["guard-judge"]
              for label in LABELS["guard-judge"]}
    seeds = bench.seeds(counts)
    shared = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
    for seed in seeds:
        if seed.design_stratum == "understanding" and seed.axes["intendedAnswer"] is True:
            bench.texts[seed.seed_id] = shared
    bench.start(seeds)
    bench.generate()
    bench.vote(bench.agree)
    for short in (False, True):
        refused(bench, r"below its floor of 2 \(\(true, understanding\): eval 0, calibration 4\)",
                floor_short=True, short=short, rng_seed=100 + short)
    cells = bench.camp.state().floor_short["cells"]
    assert cells == [{"stratum": "understanding", "label": "true", "n": 8, "need": None}]


def test_an_empty_cell_is_printed_with_zeros(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A cell with no labelled item is absent from the split's sizes and surplus, not from the
    printed n."""
    bench = short_bench(tmp_path, monkeypatch, turns_true=0, floor=0)
    decline(bench)
    lines = bench.finalize(short=True)
    assert ("  cell turns / true: eval 0, calibration 0, surplus 0 | eval-assigned 0, assigned "
            "drops 0, walked 0, refill drops 0, refilled 0, floor 0, d 0.00") in lines
    assert len([line for line in lines if line.startswith("  cell")]) == 4


def test_a_floor_refusal_leaves_the_store_intact_for_a_later_round(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bench = short_bench(tmp_path, monkeypatch, turns_true=3)    # eval 3, calibration 0
    decline(bench)
    refused(bench, "below its floor", floor_short=True, short=True)
    bench.camp.resolve(decision="reopen: more generation", reopen_topup=True)
    assert not bench.camp.state().topup_declined
    refused(bench, "`driver run` plans the top-up")


def test_short_on_a_point_with_no_short_cell_is_an_ordinary_finalize(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bench = short_bench(tmp_path, monkeypatch, turns_true=8)
    decline(bench)
    lines = bench.finalize(short=True)
    assert not any(line.startswith("SHORT") for line in lines)
    bench2 = short_bench(tmp_path / "b2", monkeypatch, turns_true=8)
    assert bench2.finalize()[0].startswith("finalized guard-judge as v1")


def test_after_a_reopen_and_a_completed_top_up_finalize_and_short_both_run(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bench = short_bench(tmp_path, monkeypatch, turns_true=5)
    decline(bench)
    bench.camp.resolve(decision="reopen", reopen_topup=True)
    more = bench.seeds({("turns", "true"): 4}, start=900)
    bench.start(more, gen_round=2)                                   # the top-up round
    bench.generate()
    bench.vote(bench.agree)
    assert bench.finalize("v1")[0].startswith("finalized")
    assert bench.finalize("v2", short=True, rng_seed=7)[0].startswith("finalized")
    assert not (bench.data / "guard-judge" / "v3").exists()


def test_an_unbounded_cell_is_short_and_a_floor_failure(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bench = short_bench(tmp_path, monkeypatch, turns_true=0)
    decline(bench)
    refused(bench, "--short.*--reopen-topup")
    refused(bench, "below its floor of 2 .*turns", floor_short=True, short=True)


# --- the eval refill, the floor gate and `floor-short` (U5.10c) ---------------------------------

SEED = 20261110
Cell = tuple[str, str]


def plain_split(bench: Bench, per_cell: int) -> tuple[split.SplitResult, dict[str, Cell]]:
    """The split `finalize` would make over `bench`'s campaign, and each seed's cell."""
    state = bench.camp.state()
    rows = [row for row in diagnostics.labelled_items(state) if row.cls is not None]
    candidates = [split.Candidate(row.seed_id, (row.stratum, row.cls),
                                  shingles.evidence_text(bench.role, row.item)) for row in rows]
    result = split.make_splits(candidates, sizes={"eval": per_cell, "calibration": per_cell,
                                                  "train": 0}, rng_seed=SEED)
    return result, {c.seed_id: c.cell for c in candidates}


def planted_bench(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                  counts: dict[Cell, int], plant: dict[Cell, tuple[int, int]], *,
                  per_cell: int = 6, floor: int = 6) -> Bench:
    """Guard with `counts` labelled items per cell. `plant[cell] = (k, u)` plants a distinct span
    in the cell's first `k` eval items (of the plain split at `SEED`) and a calibration item each
    (so `k` assigned drops), and the first of them in `u` of its unused items (`u` refill drops
    when walked)."""
    def build(root: Path, texts: dict[str, str]) -> Bench:
        bench = Bench(root, monkeypatch, "guard-judge", per_cell=per_cell, floor=floor)
        seeds = bench.seeds(counts)
        bench.texts.update(texts)
        bench.start(seeds)
        bench.generate()
        bench.vote(bench.agree)
        return bench

    base = build(tmp_path / "plain", {})
    result, cell_of = plain_split(base, per_cell)
    texts: dict[str, str] = {}
    everyone = set(result.eval) | set(result.calibration)
    for cell, (k, u) in plant.items():
        evals = [x for x in result.eval if cell_of[x] == cell]
        cals = [x for x in result.calibration if cell_of[x] == cell]
        unused = sorted(x for x in cell_of if cell_of[x] == cell and x not in everyone)
        spans = [" ".join(f"sp{cell[0][:2]}{cell[1][0]}{i}w{j}" for j in range(10))
                 for i in range(k)]
        for i, span in enumerate(spans):
            texts[evals[i]], texts[cals[i]] = span, "x " + span + " y"
        for seed_id in unused[:u]:
            texts[seed_id] = spans[0] + " z"
    return build(tmp_path / "planted", texts)


US, UF, TS, TF = ("understanding", "true"), ("understanding", "false"), ("turns", "true"), (
    "turns", "false")


def test_finalize_refills_an_eval_slot_from_the_cells_surplus_and_prints_every_figure(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    counts = {US: 14, UF: 12, TS: 13, TF: 12}
    bench = planted_bench(tmp_path, monkeypatch, counts, {US: (2, 0), TS: (1, 1)}, floor=5)
    lines = bench.finalize(rng_seed=SEED)
    head = "eval 6, calibration 6, surplus 0 | eval-assigned 6, assigned drops "
    expect = {
        "understanding / true": head.replace("assigned drops ", "assigned drops 2, walked 2, "
                                              "refill drops 0, refilled 2, floor 5, d 0.33 "
                                              "(above 0.25: a generation-diversity signal)"),
        "understanding / false": head + "0, walked 0, refill drops 0, refilled 0, floor 5, d 0.00",
        "turns / true": "eval 5, calibration 6, surplus 0 | eval-assigned 6, assigned drops 1, "
                        "walked 1, refill drops 1, refilled 0, floor 5, d 0.17",
        "turns / false": head + "0, walked 0, refill drops 0, refilled 0, floor 5, d 0.00"}
    for cell, text in expect.items():
        assert f"  cell {cell}: {text}" in lines
    dropped = bench.dataset()["generation"]["dropped"]
    assert dropped["shingleSplit"] == 3                       # assigned drops only: 2 + 1
    splits = bench.dataset()["splits"]
    assert len(splits["eval"]) == 6 + 6 + 5 + 6 and len(splits["calibration"]) == 24


def test_a_refill_never_enters_through_a_dropped_surplus_a_refill_drop_is_not_counted_assigned(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    counts = {US: 14, UF: 12, TS: 13, TF: 12}
    bench = planted_bench(tmp_path, monkeypatch, counts, {TS: (1, 1)}, floor=5)
    bench.finalize(rng_seed=SEED)
    assert bench.dataset()["generation"]["dropped"]["shingleSplit"] == 1


def test_an_eval_cell_under_its_floor_after_the_refill_refuses_and_records_floor_short(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Eval size 6, floor 5, and per cell a different `n`, drop rate and deficit, so a figure
    taken from another cell, or the size for the floor, goes wrong. understanding/true: 12 items,
    3 of 6 dropped (d = 1/2, eval 3, deficit 2: need 4); understanding/false: 12, 2 of 6 (d =
    1/3, eval 4, deficit 1: need 2); turns/true: 14, 4 of 6 dropped and both surplus items
    overlap (d = 2/3, eval 2, deficit 3: need 9); turns/false: 12, none dropped."""
    counts = {US: 12, UF: 12, TS: 14, TF: 12}
    bench = planted_bench(tmp_path, monkeypatch, counts, {US: (3, 0), UF: (2, 0), TS: (4, 2)},
                          floor=5)
    refused(bench, "below its floor of 5", floor_short=True, rng_seed=SEED)
    event = json.loads(ledger(bench).splitlines()[-1])
    assert event["rngSeed"] == SEED
    by_cell = {(c["stratum"], c["label"]): (c["n"], c["need"]) for c in event["cells"]}
    assert by_cell == {US: (12, 4), UF: (12, 2), TS: (14, 9)}          # turns/false is not short
    state = bench.camp.state()
    assert SEED in state.held_rng_seeds
    with pytest.raises(CampaignRefused, match=f"RNG seed {SEED} is already held"):
        bench.finalize("vy", rng_seed=SEED)
    assert not (bench.data / "guard-judge").exists()
    assert state.floor_short["rngSeed"] == SEED


@pytest.mark.parametrize(("sizes_eval", "assigned", "drops", "floor", "need"), [
    (4, 6, 2, 6, 3),      # deficit 2, d = 1/3: 2 / (2/3) = 3 exactly (a float ceil gives 4)
    (5, 6, 1, 6, 2),      # deficit 1, d = 1/6: 1.2 up to 2
    (2, 6, 3, 6, 8),      # deficit 4, d = 1/2
    (3, 4, 0, 5, 2),      # no drop: the deficit itself
    (0, 4, 4, 2, None),   # every assigned item dropped: no survival rate observed
    (6, 6, 0, 6, 0),      # at the floor: nothing needed
])
def test_need_is_the_deficit_over_the_survival_rate_rounded_up_in_exact_integers(
        sizes_eval: int, assigned: int, drops: int, floor: int, need: int | None) -> None:
    figures = finalize.CellFigures(
        "s", "x", {"eval": sizes_eval, "calibration": 6, "train": 0}, 0, 12, assigned, drops,
        drops, 0, 0)
    assert figures.need(floor) == need


def test_a_later_floor_short_replaces_the_earlier_and_both_seeds_stay_held(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    counts = {US: 12, UF: 12, TS: 12, TF: 12}
    bench = planted_bench(tmp_path, monkeypatch, counts, {US: (2, 0)})
    refused(bench, "below its floor", floor_short=True, rng_seed=SEED)
    later = [{"stratum": "turns", "label": "false", "n": 9, "need": 4}]
    bench.camp._append({"event": "floor-short", "rngSeed": SEED + 1, "cells": later})
    state = bench.camp.state()
    assert state.floor_short == {"rngSeed": SEED + 1, "cells": later}
    assert {SEED, SEED + 1} <= state.held_rng_seeds
    assert state.halt is None                                          # a plain fold: no halt


def test_a_cell_at_its_floor_passes_and_one_above_it_refuses(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    counts = {US: 12, UF: 12, TS: 12, TF: 12}
    at = planted_bench(tmp_path / "at", monkeypatch, counts, {US: (2, 0)}, floor=4)   # eval 4
    assert at.finalize(rng_seed=SEED)[0].startswith("finalized")
    above = planted_bench(tmp_path / "above", monkeypatch, counts, {US: (2, 0)}, floor=5)
    refused(above, "below its floor of 5", floor_short=True, rng_seed=SEED)


def test_a_refusal_for_a_cell_with_no_eval_deficit_records_need_zero(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A calibration shortfall has no `need` (the planner's own target covers it)."""
    bench = short_bench(tmp_path, monkeypatch, turns_true=5)          # eval 4 + calibration 1
    decline(bench)
    refused(bench, "calibration 1", floor_short=True, short=True, rng_seed=SEED)
    event = json.loads(ledger(bench).splitlines()[-1])
    assert event["cells"] == [{"stratum": "turns", "label": "true", "n": 5, "need": 0}]


def test_build_dataset_stays_pure_the_event_is_finalizes_to_append(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    counts = {US: 12, UF: 12, TS: 12, TF: 12}
    bench = planted_bench(tmp_path, monkeypatch, counts, {US: (2, 0)})
    before = ledger(bench)
    with pytest.raises(CampaignRefused, match="below its floor"):
        finalize.build_dataset(bench.camp.state(), rng_seed=SEED, side_check=frozenset(),
                               generated_at="2026-11-12T00:00:00Z")
    assert ledger(bench) == before


def test_rng_seed_fixes_the_seed_on_the_command_line_and_a_held_one_is_refused(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bench = Bench(tmp_path, monkeypatch, "guard-judge")
    bench.full_round(5)
    argv = ["finalize", "--root", str(bench.camp.root), "--dataset-version", "v1",
            "--data-dir", str(bench.data), "--rng-seed", "4242"]
    assert driver.main(argv) == 0
    assert bench.dataset()["splits"]["rngSeed"] == 4242
    assert bench.camp.state().held_rng_seeds >= {4242}
    argv[argv.index("v1")] = "v2"
    assert driver.main(argv) == 1                                      # 4242 is held


# --- the mid-tranche rubric edit -----------------------------------------------------------------


def mt_bench(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> Bench:
    bench = Bench(tmp_path, monkeypatch, "message-type-classifier", per_cell=1, **kwargs)
    counts = {(None, label): 4 for label in LABELS["message-type-classifier"]}
    bench.start(bench.seeds(counts))
    bench.generate()
    return bench


def test_a_changed_rubric_hash_is_refused_naming_both_hashes_and_restoring_clears_it(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bench = mt_bench(tmp_path, monkeypatch)
    bench.vote(bench.agree)
    tranche = bench.camp.state().tranche["rubric"]
    monkeypatch.setattr(campaign, "rubric_hash", lambda role: "7" * 64)
    refused(bench, f"{'7' * 64}.*{tranche}")
    monkeypatch.undo()
    monkeypatch.setattr(campaign, "cell_target", lambda role: 2)
    monkeypatch.setattr(finalize, "eval_floor", lambda role: 1)
    assert bench.finalize()[0].startswith("finalized message-type-classifier")


def test_a_point_with_no_tranche_is_not_rubric_checked(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bench = Bench(tmp_path, monkeypatch, "guard-judge")
    bench.full_round(5)
    assert bench.camp.state().tranche is None
    monkeypatch.setattr(campaign, "rubric_hash", lambda role: "7" * 64)
    bench.finalize()
    assert bench.dataset()["generation"]["promptSha256"]["rubric"] == "7" * 64


# --- the whole-point recheck ---------------------------------------------------------------------


def recheck_bench(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Bench, set[str]]:
    """Message type: a tranche of 20 with no tie, and 2 ties among the other 20 items (2 of 40:
    5% > 3%, while the tranche passes)."""
    bench = Bench(tmp_path, monkeypatch, "message-type-classifier", per_cell=1, tranche=20)
    bench.start(bench.seeds({(None, label): 8 for label in LABELS["message-type-classifier"]}))
    bench.generate()
    bench.camp.queue_votes()
    tranche = set(bench.camp.state().tranche_items())
    rest = [s for s in bench.camp.state().items if s not in tranche]
    ties = set(rest[:2])
    labels_ = LABELS["message-type-classifier"]
    bench.tied = lambda st, ref, vote: (labels_[vote - 1] if ref in ties
                                        else st.seeds[ref].axes["intendedAnswer"])
    bench.vote(bench.tied)
    assert bench.camp.state().tranche_passed() and bench.camp.state().halt is None
    return bench, ties


def test_a_failing_whole_point_recheck_halts_and_names_the_relabel(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bench, _ = recheck_bench(tmp_path, monkeypatch)
    with pytest.raises(CampaignRefused, match="2 voteTie drops in 40 labelled.*relabels the whole"):
        bench.finalize()
    state = bench.camp.state()
    assert state.halt["reason"] == "recheck" and not state.halt["second"]
    assert not list(bench.data.glob("*/*")) if bench.data.exists() else True
    with pytest.raises(CampaignRefused, match="halted"):
        bench.finalize()                                             # still halted: no second event
    assert [e["event"] for e in bench.camp.events()].count("recheck-halt") == 1


def test_the_recheck_passes_at_the_tranche_boundary(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """3 ties in 100 labelled items: 300 = 3 * 100 is not over the rule."""
    assert not campaign.tranche_failed(3, 100) and campaign.tranche_failed(4, 100)
    bench = Bench(tmp_path, monkeypatch, "message-type-classifier", per_cell=1, tranche=20)
    bench.start(bench.seeds({(None, label): 8 for label in LABELS["message-type-classifier"]}))
    bench.generate()
    bench.vote(bench.agree)                                          # no ties at all
    assert bench.finalize()[0].startswith("finalized")


def test_a_changed_rubric_relabels_the_whole_point_and_a_second_failure_needs_the_stakeholder(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bench, ties = recheck_bench(tmp_path, monkeypatch)
    with pytest.raises(CampaignRefused):
        bench.finalize()
    with pytest.raises(CampaignRefused, match="rubric is unchanged"):
        bench.camp.resolve(decision="reworded")
    with pytest.raises(CampaignRefused, match="only for the second failure|second failure"):
        bench.camp.resolve(decision="x", stakeholder_ref="somebody")
    monkeypatch.setattr(campaign, "rubric_hash", lambda role: "6" * 64)
    bench.camp.resolve(decision="rubric: tie-break order")
    state = bench.camp.state()
    assert state.halt is None and state.votes == {} and state.tranche["rubric"] == "6" * 64
    assert state.tranche["relabels"] == 1 and not state.tranche_passed()
    bench.vote(bench.tied)                                           # every item, three votes
    state = bench.camp.state()
    assert all(state.three_votes(seed) is not None for seed in state.items)
    with pytest.raises(CampaignRefused, match="second failure.*--stakeholder-ref"):
        bench.finalize()
    assert bench.camp.state().halt["second"]
    with pytest.raises(CampaignRefused, match="only the stakeholder"):
        bench.camp.resolve(decision="go on")
    with pytest.raises(CampaignRefused, match="only the stakeholder"):
        bench.camp.resolve(decision="go on", stakeholder_ref="  ")
    bench.camp.resolve(decision="the labels do not separate: accept",
                       stakeholder_ref="the owner, ledger line 3")
    (accepted,) = [e for e in bench.camp.events() if e["event"] == "recheck-accepted"]
    assert accepted["stakeholderRef"] == "the owner, ledger line 3"
    assert bench.finalize()[0].startswith("finalized")
    assert bench.dataset()["generation"]["dropped"]["voteTie"] == len(ties)


def test_a_re_pin_neither_clears_the_second_recheck_failure_nor_carries_the_ref(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bench, _ = recheck_bench(tmp_path, monkeypatch)
    with pytest.raises(CampaignRefused):
        bench.finalize()
    monkeypatch.setattr(campaign, "rubric_hash", lambda role: "6" * 64)
    bench.camp.resolve(decision="rubric: tie-break order")
    bench.vote(bench.tied)
    with pytest.raises(CampaignRefused, match="second failure"):
        bench.finalize()
    before = ledger(bench)
    for extra in ({}, {"stakeholder_ref": "the owner, ledger line 3"}):
        with pytest.raises(CampaignRefused, match="only by `--stakeholder-ref`, alone"):
            bench.camp.resolve(decision="new labeler", repin_model=(
                "labeler", "claude-sonnet-5-6"), **extra)
    assert ledger(bench) == before and bench.camp.state().halt["second"]
    with pytest.raises(CampaignRefused, match="only by `--stakeholder-ref`, alone"):
        bench.camp.resolve(decision="same", repin_instructions=True,
                           stakeholder_ref="the owner")


def test_a_generator_restart_resets_the_recheck_count(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bench, _ = recheck_bench(tmp_path, monkeypatch)
    with pytest.raises(CampaignRefused):
        bench.finalize()
    assert bench.camp.state().recheck_failures == 1
    bench.camp.resolve(decision="regenerate", repin_model=("generator", "claude-opus-5-6"))
    state = bench.camp.state()
    assert (state.recheck_failures, state.recheck_accepted, state.halt) == (0, False, None)


# --- the one-hash rule ---------------------------------------------------------------------------


def template_halt_bench(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *,
                        resolve: bool) -> Bench:
    """Guard, round 1 of 40 seeds in two runs: run 1 (hash H1) has two screen hits, which halts;
    a second install changes `generator.md`'s hash (H2); run 2 and the regenerations carry H2.
    `resolve` False installs H2 with no halt cleared in between."""
    bench = Bench(tmp_path, monkeypatch, "guard-judge", per_cell=2)
    seeds = bench.seeds({(s, label): 10 for s in STRATA["guard-judge"] for label in ("true",
                                                                                     "false")})
    bench.start(seeds)
    first, second = [b for b in bench.camp.state().batches.values()]
    hit_seeds = set(first.entries[:2])
    bench.complete(first.batch_id, bench._generation_events(
        first, lambda seed, nth: ("screen-hit", None) if seed.seed_id in hit_seeds and nth == 1
        else None))
    if resolve:
        assert bench.camp.state().halt["voidClass"] == "template"
        bench.camp.resolve(decision="generator.md: stop the leak")
    bench.install(request_generator="f0" * 32)
    bench.generate()
    bench.vote(bench.agree)
    return bench


def test_a_second_hash_after_a_template_halt_resolve_finalizes_with_per_hash_counts(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bench = template_halt_bench(tmp_path, monkeypatch, resolve=True)
    bench.finalize()
    templates = bench.dataset()["diagnostics"]["templates"]
    h1, h2 = templates["hashes"]
    assert h1["templateHash"] != h2["templateHash"]
    state = bench.camp.state()
    first_items = sum(1 for r in state.items.values() if r["genRunId"] == "gen_0001")
    assert (h1["runs"], h1["items"]) == (1, first_items) and h1["items"] == 18
    assert (h2["runs"], h2["items"]) == (2, 22)                      # run 2 and the regeneration
    assert [(h["reason"], h["genRound"]) for h in templates["halts"]] == [("screen", 1)]
    for batch_id, digest in state.run_hash.items():                  # each genRunId, one hash
        assert sum(1 for h in templates["hashes"] if h["templateHash"] == digest) == 1
        assert batch_id in state.batches
    assert bench.dataset()["generation"]["promptSha256"]["requestTemplate"]["generator"] == (
        "f0" * 32)


def test_a_second_hash_finalizes_when_the_first_hash_ran_again_after_the_resolve(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The licence is the resolve between the two hashes' *first* runs, whatever H1 did later."""
    bench = Bench(tmp_path, monkeypatch, "guard-judge", per_cell=2)
    seeds = bench.seeds({(s, label): 10 for s in STRATA["guard-judge"] for label in ("true",
                                                                                     "false")})
    bench.start(seeds)
    first, second = bench.camp.state().batches.values()
    hit_seeds = set(first.entries[:2])
    bench.complete(first.batch_id, bench._generation_events(
        first, lambda seed, nth: ("screen-hit", None) if seed.seed_id in hit_seeds and nth == 1
        else None))
    bench.camp.resolve(decision="generator.md: stop the leak")
    bench.complete(second.batch_id, bench._generation_events(second, None))     # still H1
    bench.install(request_generator="f0" * 32)
    bench.generate()                                                            # H2
    bench.vote(bench.agree)
    assert finalize.hash_refusal(bench.camp.state()) is None
    bench.finalize()


def test_a_second_hash_with_no_template_halt_resolve_is_refused(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bench = Bench(tmp_path, monkeypatch, "guard-judge")
    seeds = bench.seeds({(s, label): 10 for s in STRATA["guard-judge"] for label in ("true",
                                                                                     "false")})
    bench.start(seeds)
    first = next(iter(bench.camp.state().batches.values()))
    bench.complete(first.batch_id, bench._generation_events(first, None))
    bench.install(request_generator="f0" * 32)               # H2, and no halt was ever resolved
    bench.generate()
    bench.vote(bench.agree)
    refused(bench, "generator template changed.*no `driver resolve` of a template halt")


def test_a_resolve_of_another_halt_does_not_license_a_second_hash(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bench = Bench(tmp_path, monkeypatch, "guard-judge")
    seeds = bench.seeds({(s, label): 10 for s in STRATA["guard-judge"] for label in ("true",
                                                                                     "false")})
    bench.start(seeds)
    first = next(iter(bench.camp.state().batches.values()))
    bench.complete(first.batch_id, bench._generation_events(first, None))
    second = [b for b in bench.camp.state().batches.values() if b.state == "pending"][0]
    bench.complete(second.batch_id, lambda opaque: [], shape="empty", outcome="halted",
                   reason="safeguardStop")
    bench.camp.resolve(decision="retry")                      # a resolve, but not a template halt
    bench.install(request_generator="f0" * 32)
    bench.generate()
    bench.vote(bench.agree)
    refused(bench, "generator template changed")


def test_a_template_halt_that_the_second_hashs_own_run_raised_does_not_license_it(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """H2's first run was issued before the resolve: the resolve came too late to be the fix."""
    bench = Bench(tmp_path, monkeypatch, "guard-judge", per_cell=2)
    seeds = bench.seeds({(s, label): 10 for s in STRATA["guard-judge"] for label in ("true",
                                                                                     "false")})
    bench.start(seeds)
    first, second = bench.camp.state().batches.values()
    bench.complete(first.batch_id, bench._generation_events(first, None))
    bench.install(request_generator="f0" * 32)
    hit_seeds = set(second.entries[:2])
    bench.complete(second.batch_id, bench._generation_events(
        second, lambda seed, nth: ("screen-hit", None) if seed.seed_id in hit_seeds and nth == 1
        else None))
    assert bench.camp.state().halt["voidClass"] == "template"
    bench.camp.resolve(decision="generator.md: stop the leak")
    bench.generate()
    bench.vote(bench.agree)
    refused(bench, "generator template changed")


def test_a_template_halt_resolve_before_the_first_hashs_first_run_does_not_count(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The resolve must fall between the two hashes' first runs, not before both."""
    bench = template_halt_bench(tmp_path, monkeypatch, resolve=True)
    state = bench.camp.state()
    state.template_resolutions[:] = [0]                       # a resolve before every run
    assert finalize.hash_refusal(state) is not None
    assert finalize.hash_refusal(bench.camp.state()) is None


def test_the_one_hash_rule_reads_only_the_runs_after_the_last_generator_restart(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bench = Bench(tmp_path, monkeypatch, "guard-judge")
    seeds = bench.seeds({(s, label): 10 for s in STRATA["guard-judge"] for label in ("true",
                                                                                     "false")})
    bench.start(seeds)
    first = next(iter(bench.camp.state().batches.values()))
    bench.complete(first.batch_id, bench._generation_events(first, None))
    bench.complete(first.batch_id, lambda opaque: [], shape="empty", outcome="halted",
                   reason="safeguardStop")
    bench.camp.resolve(decision="repin the generator", repin_model=("generator", "claude-opus-5-6"))
    bench.install(request_generator="f0" * 32)               # H2 before the restarted round
    seeds2 = bench.seeds({(s, label): 10 for s in STRATA["guard-judge"]
                          for label in ("true", "false")}, start=1000)
    bench.start(seeds2)
    bench.generate()
    bench.vote(bench.agree)
    assert finalize.hash_refusal(bench.camp.state()) is None   # the pre-restart H1 run is retired
    bench.finalize()
    runs = bench.dataset()["generation"]["runs"]
    assert runs and all(r["batchId"] != first.batch_id for r in runs)
    assert any(r["batchId"] == first.batch_id for a in bench.camp.state().attempts
               for r in [a.outcome["run"]] if r)                # the ledger still holds them


# --- the command line ----------------------------------------------------------------------------


def test_the_cli_finalizes_prints_and_refuses_with_exit_1(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    bench = Bench(tmp_path, monkeypatch, "guard-judge")
    bench.full_round(5)
    root = str(bench.camp.root)
    base = ["finalize", "--root", root, "--data-dir", str(bench.data)]
    assert driver.main([*base, "--dataset-version", "v1"]) == 0
    out = capsys.readouterr().out
    assert "finalized guard-judge as v1: 16 items" in out and "dropped:" in out
    assert driver.main([*base, "--dataset-version", "v1"]) == 1
    assert "never overwritten" in capsys.readouterr().err
    assert (bench.data / "guard-judge" / "v1" / "splits.json").is_file()
    with pytest.raises(SystemExit):
        driver.main(["finalize", "--root", root])                    # --dataset-version required


def test_resolve_takes_a_stakeholder_ref_flag(tmp_path: Path,
                                              capsys: pytest.CaptureFixture) -> None:
    root = tmp_path / "camp"
    driver.main(["init", "--point", "guard-judge", "--phase", "pilot", "--root", str(root)])
    assert driver.main(["resolve", "--root", str(root), "--decision", "x",
                        "--stakeholder-ref", "r"]) == 1
    assert "not halted" in capsys.readouterr().err


def test_make_seed_helper_builds_a_message_type_seed_with_a_thread_position() -> None:
    assert make_seed("message-type-classifier", 3, None, "order").axes["threadPosition"] == (
        "opening")
