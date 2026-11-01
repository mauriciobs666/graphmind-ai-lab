"""`scripts/import_decision_pack.py` — typed-decision-models U3d, the one-way importer of a
decision-models dataset's eval split into `packs/<role>-synthetic/` and of a candidate with its fit
manifest into `candidates/` (v1.3 §3.4 "Importer"; plan2 §4.1 U3d; plan4 §4.2 U3d and §5's U3d
row; impl review P12 point 2 and P22-n1 via the coordination ledger's U3d brief).

Offline throughout. decision-models has no dataset yet, so each test builds one under `tmp_path`
in the shape the plans give the producer (plan4 §3.7 `generation.json`; items with the role's input
fields, `expected`, `origin`, `genRunId`, `genRound`, `seedId`, `labelVotes`, message type's
`threadPosition`; `splits.json` as the importer's docstring assumes), plus a model-bench root under
`tmp_path` holding copies of the committed prompts and the guard's side-check source. Nothing here
writes under the real `model-bench/`.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from modelbench.decisions import (
    BAR_KIND_BY_ROLE,
    INPUT_FIELDS_BY_ROLE,
    MESSAGE_TYPE_LABELS,
    MIN_EVAL_ITEMS,
    Candidate,
    eval_set_hash,
    input_hash,
    load_fit_manifest,
)
from modelbench.packs import load_pack, validate_pack
from modelbench.scoring.agreement import per_class_agreement_metrics
from modelbench.scoring.classification import PER_KIND_VERDICT_METRICS

_TESTS = Path(__file__).resolve().parent
_MB = _TESTS.parent
_SCRIPTS_DIR = _MB / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import import_decision_pack as imp  # noqa: E402

ROLES = sorted(BAR_KIND_BY_ROLE)
_SIDE_CHECK_SOURCE = _MB / "packs" / "guard-judge-understanding" / "items.jsonl"
_FAKE_SHA = "f" * 40
SIDE = "guard-judge-understanding"

LABELER = {"surface": "claude-code-subagent", "tier": "sonnet", "model": "claude-sonnet-x",
           "effort": "high"}
GENERATOR = {"surface": "claude-code-subagent", "tier": "opus", "model": "claude-opus-y",
             "effort": "medium"}


# --- builders ----------------------------------------------------------------------------------


def _inputs(role: str, n: int) -> dict[str, Any]:
    if role == "guard-judge":
        path = "understanding" if n % 2 else "turns"
        return {
            "condition": f"generated condition {n}",
            "understanding": {"request": f"request {n}", "known": [], "missing": []}
            if path == "understanding" else {},
            "turns": [] if path == "understanding"
            else [{"role": "user", "displayName": "Ana", "text": f"turn {n}"}],
            "path": path,
        }
    if role == "join-in-judge":
        return {
            "agentName": "Pastel",
            "priorTurns": []
            if n % 2 else [{"role": "user", "displayName": "Ana", "text": f"hi {n}"}],
            "newMessage": {"role": "user", "displayName": "Bo", "text": f"message {n}"},
        }
    prior = None if n % 2 else {"speaker": "Ana", "text": f"hi {n}"}
    return {"text": f"message {n}", "priorTurn": prior}


def _expected(role: str, n: int) -> Any:
    if role == "message-type-classifier":
        return MESSAGE_TYPE_LABELS[n % len(MESSAGE_TYPE_LABELS)]
    return n % 2 == 0


def _source_item(role: str, n: int, split: str) -> dict[str, Any]:
    expected = _expected(role, n)
    votes = (
        {label: (3 if label == expected else 0) for label in MESSAGE_TYPE_LABELS}
        if role == "message-type-classifier"
        else {"true": 2 if expected else 1, "false": 1 if expected else 2}
    )
    item = {
        "itemId": f"dm-{n:04d}",
        **_inputs(role, n),
        "expected": expected,
        "origin": "synthetic",
        "seedId": f"sd_{n:04d}",
        "genRunId": f"run_{n // 25:03d}",
        "genRound": 1 + (n % 3 == 0),
        "labelVotes": votes,
        # producer-side fields the pack never carries
        "intent": "x",
        "split": split,
    }
    if role == "message-type-classifier":
        item["threadPosition"] = "opening" if n % 2 else "follow-up"
    return item


def _dataset(base: Path, role: str, *, evals: int = MIN_EVAL_ITEMS, others: int = 10,
             edit_items=None, generation_edit=None, version: str = "2026-11-01") -> Path:
    """`base/data/<role>/<version>/` with `evals` eval items and `others` train/calibration ones,
    interleaved so the importer has to select by `splits.json`."""
    directory = base / "data" / role / version
    directory.mkdir(parents=True)
    items, eval_ids, other_ids = [], [], []
    for n in range(evals + others):
        split = "eval" if n < evals else ("train" if n % 2 else "calibration")
        items.append(_source_item(role, n, split))
        (eval_ids if split == "eval" else other_ids).append(items[-1]["itemId"])
    items = items[1::2] + items[0::2]
    if edit_items is not None:
        edit_items(items)
    (directory / "items.jsonl").write_text("".join(json.dumps(i) + "\n" for i in items), "utf-8")
    (directory / "splits.json").write_text(
        json.dumps({"rngSeed": 7, "eval": eval_ids, "train": other_ids[1::2],
                    "calibration": other_ids[0::2]}), "utf-8")
    generation = {
        "generator": dict(GENERATOR), "labeler": dict(LABELER), "votesPerItem": 3,
        "labelerTemperature": None, "labelBatchCap": 50, "promptSha256": {},
        "generatedAt": "2026-11-01T00:00:00Z", "itemCount": len(items), "dropped": {}, "runs": [],
    }
    if generation_edit is not None:
        generation_edit(generation)
    (directory / "generation.json").write_text(json.dumps(generation), "utf-8")
    return directory


def _mb_root(base: Path) -> Path:
    """A model-bench root holding byte-copies of every committed `packs/<role>-synthetic/` and the
    guard's side-check source."""
    root = base / "model-bench"
    for role in ROLES:
        shutil.copytree(_MB / "packs" / f"{role}-synthetic", root / "packs" / f"{role}-synthetic")
    (root / "packs" / SIDE).mkdir(parents=True)
    shutil.copyfile(_SIDE_CHECK_SOURCE, root / "packs" / SIDE / "items.jsonl")
    return root


def _import(root: Path, role: str, dataset: Path, version: str = "1.0.0") -> imp.DatasetImport:
    return imp.import_dataset(root, role, dataset, version, source_sha=lambda _p: _FAKE_SHA,
                              now=lambda: "2026-11-01T12:00:00Z")


def _rows(pack_dir: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (pack_dir / "items.jsonl").read_text("utf-8").splitlines()]


def _snapshot(directory: Path) -> dict[str, bytes]:
    return {p.relative_to(directory).as_posix(): p.read_bytes()
            for p in sorted(directory.rglob("*")) if p.is_file()}


@pytest.fixture
def root(tmp_path):
    return _mb_root(tmp_path)


# --- the dataset half: an imported pack validates ----------------------------------------------


@pytest.mark.parametrize("role", ROLES)
def test_an_imported_pack_validates_and_holds_the_eval_split_only(tmp_path, root, role):
    dataset = _dataset(tmp_path, role)
    result = _import(root, role, dataset)
    pack_dir = root / "packs" / f"{role}-synthetic"
    pack = load_pack(pack_dir)
    assert validate_pack(pack) == []
    rows = _rows(pack_dir)
    evals = [r for r in rows if r["stratum"] == "eval"]
    eval_ids = json.loads((dataset / "splits.json").read_text())["eval"]
    assert sorted(r["itemId"] for r in evals) == sorted(eval_ids)
    assert result.evalCount == len(evals) == MIN_EVAL_ITEMS
    assert result.contentHash == pack.contentHash
    assert pack.manifest["packId"] == f"{role}-synthetic"
    assert pack.manifest["packVersion"] == "1.0.0"


@pytest.mark.parametrize("role", ROLES)
def test_an_eval_item_keeps_its_provenance_and_its_input_hash(tmp_path, root, role):
    """plan4 §5's U3d row: an imported eval item keeps `genRunId`, `genRound`, `seedId` and
    `labelVotes` (and message type's `threadPosition`); `input_hash` is unchanged by them."""
    dataset = _dataset(tmp_path, role)
    _import(root, role, dataset)
    source = {json.loads(line)["itemId"]: json.loads(line)
              for line in (dataset / "items.jsonl").read_text().splitlines()}
    kept = ["genRunId", "genRound", "seedId", "labelVotes"]
    if role == "message-type-classifier":
        kept.append("threadPosition")
    evals = [r for r in _rows(root / "packs" / f"{role}-synthetic") if r["stratum"] == "eval"]
    for row in evals:
        src = source[row["itemId"]]
        assert {k: row[k] for k in kept} == {k: src[k] for k in kept}
        stripped = {k: v for k, v in row.items() if k not in kept}
        assert input_hash(row, role) == input_hash(stripped, role) == input_hash(src, role)
        assert row["expected"] == src["expected"] and row["origin"] == "synthetic"
    assert eval_set_hash(evals, role) == eval_set_hash(
        [{**source[r["itemId"]], "stratum": "eval"} for r in evals], role)


@pytest.mark.parametrize("role", ROLES)
def test_an_eval_item_carries_exactly_the_projected_fields(tmp_path, root, role):
    _import(root, role, _dataset(tmp_path, role))
    expected = {"itemId", *INPUT_FIELDS_BY_ROLE[role], "expected", "origin", "stratum",
                "genRunId", "genRound", "seedId", "labelVotes"}
    if role == "message-type-classifier":
        expected.add("threadPosition")
    for row in _rows(root / "packs" / f"{role}-synthetic"):
        if row["stratum"] == "eval":
            assert set(row) == expected


@pytest.mark.parametrize(
    ("role", "field"),
    [(role, field) for role in ROLES for field in imp.PROVENANCE_FIELDS_BY_ROLE[role]]
    + [(role, field) for role in ROLES for field in (*INPUT_FIELDS_BY_ROLE[role], "expected",
                                                     "origin")],
)
def test_an_eval_item_lacking_a_copied_field_is_refused_and_nothing_is_written(
    tmp_path, root, role, field
):
    def edit(items):
        victim = next(i for i in items if i["split"] == "eval")
        del victim[field]

    pack_dir = root / "packs" / f"{role}-synthetic"
    before = _snapshot(pack_dir)
    with pytest.raises(imp.ImportRefused, match=field):
        _import(root, role, _dataset(tmp_path, role, edit_items=edit))
    assert _snapshot(pack_dir) == before


@pytest.mark.parametrize(
    ("field", "value"),
    [("genRunId", ""), ("genRunId", 7), ("genRound", 0), ("genRound", True), ("genRound", "1"),
     ("labelVotes", {"true": 2, "false": 2}), ("labelVotes", {"true": 3, "false": False}),
     ("labelVotes", [3])],
)
def test_ill_typed_provenance_is_refused(tmp_path, root, field, value):
    def edit(items):
        next(i for i in items if i["split"] == "eval")[field] = value

    with pytest.raises(imp.ImportRefused, match=field):
        _import(root, "join-in-judge", _dataset(tmp_path, "join-in-judge", edit_items=edit))


def test_a_non_eval_item_lacking_provenance_is_not_read(tmp_path, root):
    def edit(items):
        for item in items:
            if item["split"] != "eval":
                del item["genRunId"]

    _import(root, "join-in-judge", _dataset(tmp_path, "join-in-judge", edit_items=edit))


# --- the manifest ------------------------------------------------------------------------------


@pytest.mark.parametrize("role", ROLES)
def test_the_manifest_states_the_incumbent_explicitly(tmp_path, root, role):
    """Impl review P12 point 2: `incumbent` is written, `null` included, never left absent."""
    _import(root, role, _dataset(tmp_path, role))
    raw = json.loads((root / "packs" / f"{role}-synthetic" / "pack.json").read_text())
    assert "incumbent" in raw["decisionPoint"]
    if BAR_KIND_BY_ROLE[role] == "speed-up":
        assert raw["decisionPoint"]["incumbent"] == "qwen/qwen3-4b-2507"
    else:
        assert raw["decisionPoint"]["incumbent"] is None


def test_the_guard_manifest_declares_exactly_the_four_per_kind_rates(tmp_path, root):
    _import(root, "guard-judge", _dataset(tmp_path, "guard-judge"))
    manifest = json.loads((root / "packs" / "guard-judge-synthetic" / "pack.json").read_text())
    assert manifest["metrics"]["verdictMetrics"] == list(PER_KIND_VERDICT_METRICS)
    assert len(manifest["metrics"]["verdictMetrics"]) == 4
    assert manifest["scorer"] == "classification"
    assert manifest["prompt"]["systemPrompt"] == "prompts/judge.md"


def test_the_guard_prompt_parameters_are_the_incumbent_packs(tmp_path, root):
    """Impl review P24-n2: the guard's synthetic pack runs the incumbent's judge prompt (a
    byte-copy, pinned below), so its `temperature`/`maxTokens` must be the incumbent pack's too —
    FR-1/FR-2 compare arms on the same prompt."""
    _import(root, "guard-judge", _dataset(tmp_path, "guard-judge"))
    imported = json.loads((root / "packs" / "guard-judge-synthetic" / "pack.json").read_text())
    incumbent = json.loads((_MB / "packs" / SIDE / "pack.json").read_text())
    for key in ("temperature", "maxTokens"):
        assert imported["prompt"][key] == incumbent["prompt"][key], key


@pytest.mark.parametrize("role", ["join-in-judge", "message-type-classifier"])
def test_a_new_ability_manifest_names_its_handoff_prompt_and_the_agreement_scorer(
    tmp_path, root, role
):
    """Impl review P22-n1: `prompt.systemPrompt: "prompts/<point>-handoff.md"`, `scorer:
    "agreement"`."""
    _import(root, role, _dataset(tmp_path, role))
    manifest = json.loads((root / "packs" / f"{role}-synthetic" / "pack.json").read_text())
    point = role.removesuffix("-judge").removesuffix("-classifier")
    assert manifest["prompt"]["systemPrompt"] == f"prompts/{point}-handoff.md"
    assert manifest["scorer"] == "agreement"
    space = manifest["decisionPoint"]["answer"]
    assert manifest["metrics"]["verdictMetrics"] == list(per_class_agreement_metrics(space))


@pytest.mark.parametrize("role", ROLES)
def test_labels_carry_the_labeler_and_generator_from_generation_json(tmp_path, root, role):
    """plan4 §4.2: `{provider: "anthropic", model, surface, effort}` each; the tier alias is not
    copied."""
    _import(root, role, _dataset(tmp_path, role))
    labels = json.loads((root / "packs" / f"{role}-synthetic" / "pack.json").read_text())["labels"]
    assert labels == {
        "source": "llm-unreviewed",
        "labeler": {"provider": "anthropic", "model": "claude-sonnet-x",
                    "surface": "claude-code-subagent", "effort": "high"},
        "generator": {"provider": "anthropic", "model": "claude-opus-y",
                      "surface": "claude-code-subagent", "effort": "medium"},
        "inputs": "synthetic",
    }


@pytest.mark.parametrize("key", ["labeler", "generator"])
@pytest.mark.parametrize("field", ["model", "surface", "effort"])
def test_a_generation_block_lacking_a_field_is_refused(tmp_path, root, key, field):
    def edit(generation):
        del generation[key][field]

    with pytest.raises(imp.ImportRefused, match=f"{key}.{field}"):
        _import(root, "join-in-judge",
                _dataset(tmp_path, "join-in-judge", generation_edit=edit))


@pytest.mark.parametrize("cap", [None, 0, True, "50"])
def test_a_bad_label_batch_cap_is_refused(tmp_path, root, cap):
    def edit(generation):
        if cap is None:
            del generation["labelBatchCap"]
        else:
            generation["labelBatchCap"] = cap

    with pytest.raises(imp.ImportRefused, match="labelBatchCap"):
        _import(root, "join-in-judge", _dataset(tmp_path, "join-in-judge", generation_edit=edit))


def test_one_model_as_generator_and_labeler_is_refused(tmp_path, root):
    def edit(generation):
        generation["generator"]["model"] = generation["labeler"]["model"]

    with pytest.raises(imp.ImportRefused, match="one model"):
        _import(root, "join-in-judge", _dataset(tmp_path, "join-in-judge", generation_edit=edit))


# --- PROVENANCE.md -----------------------------------------------------------------------------


@pytest.mark.parametrize("role", ROLES)
def test_provenance_carries_both_blocks_the_batch_cap_and_the_source_hashes(tmp_path, root, role):
    dataset = _dataset(tmp_path, role)
    _import(root, role, dataset, version="2.1.0")
    text = (root / "packs" / f"{role}-synthetic" / "PROVENANCE.md").read_text("utf-8")
    assert "**Pack version:** 2.1.0" in text
    assert "**Dataset version:** 2026-11-01" in text
    labeler_block = text.split("## Labeler")[1].split("##")[0]
    generator_block = text.split("## Generator")[1].split("##")[0]
    for block, source in ((labeler_block, LABELER), (generator_block, GENERATOR)):
        for field in ("model", "surface", "effort"):
            assert f"**{field}:** `{source[field]}`" in block
        assert "**provider:** `anthropic`" in block
    assert "**labelBatchCap:** `50`" in text
    for name in ("items.jsonl", "splits.json", "generation.json"):
        digest = hashlib.sha256((dataset / name).read_bytes()).hexdigest()
        assert f"`{digest}`" in text
    assert f"`{_FAKE_SHA}`" in text


def _provenance_rows(text: str) -> dict[str, list[str]]:
    """`PROVENANCE.md`'s source table: origin path -> [destination, git SHA, SHA-256], unquoted."""
    rows = {}
    for line in text.splitlines():
        cells = [c.strip().strip("`") for c in line.strip().strip("|").split("|")]
        if len(cells) == 4 and cells[0] not in ("Origin", "---"):
            rows[cells[0]] = cells[1:]
    return rows


@pytest.mark.parametrize("role", ROLES)
def test_each_dataset_source_is_git_checked_once_and_its_row_carries_its_own_sha(
    tmp_path, root, role
):
    """Impl review P24-m1: an uncommitted `splits.json` (or either other source) would change the
    pack without `PROVENANCE.md` showing it, so each of the three dataset files gets its own git
    check, and its row records that check's SHA rather than another file's."""
    dataset = _dataset(tmp_path, role)
    calls: list[Path] = []

    def per_path_sha(path: Path) -> str:
        calls.append(path)
        return hashlib.sha1(path.name.encode()).hexdigest()

    imp.import_dataset(root, role, dataset, "1.0.0", source_sha=per_path_sha,
                       now=lambda: "2026-11-01T12:00:00Z")
    names = ("items.jsonl", "splits.json", "generation.json")
    assert sorted(calls) == sorted(dataset / name for name in names)
    rows = _provenance_rows(
        (root / "packs" / f"{role}-synthetic" / "PROVENANCE.md").read_text("utf-8"))
    for name in names:
        origin = (dataset / name).relative_to(tmp_path).as_posix()
        _, git_sha, digest = rows[origin]
        assert git_sha == hashlib.sha1(name.encode()).hexdigest(), name
        assert digest == hashlib.sha256((dataset / name).read_bytes()).hexdigest(), name


def test_the_guard_provenance_names_the_side_check_source(tmp_path, root):
    _import(root, "guard-judge", _dataset(tmp_path, "guard-judge"))
    text = (root / "packs" / "guard-judge-synthetic" / "PROVENANCE.md").read_text("utf-8")
    assert hashlib.sha256(_SIDE_CHECK_SOURCE.read_bytes()).hexdigest() in text
    assert "stratum `side-check` (85 items)" in text


# --- the side check ----------------------------------------------------------------------------


def test_the_guard_pack_carries_the_85_human_curated_items_as_its_side_check(tmp_path, root):
    _import(root, "guard-judge", _dataset(tmp_path, "guard-judge"))
    rows = _rows(root / "packs" / "guard-judge-synthetic")
    side = [r for r in rows if r["stratum"] == "side-check"]
    source = [json.loads(line) for line in _SIDE_CHECK_SOURCE.read_text().splitlines()]
    assert len(side) == len(source) == 85
    stamps = ("stratum", "labelSource")
    assert [{k: v for k, v in r.items() if k not in stamps} for r in side] == source
    assert {r["labelSource"] for r in side} == {"human"}


@pytest.mark.parametrize("role", ["join-in-judge", "message-type-classifier"])
def test_a_new_ability_pack_has_no_side_check(tmp_path, root, role):
    _import(root, role, _dataset(tmp_path, role))
    assert {r["stratum"] for r in _rows(root / "packs" / f"{role}-synthetic")} == {"eval"}


def test_an_eval_item_id_repeating_a_side_check_id_is_refused(tmp_path, root):
    dataset = _dataset(tmp_path, "guard-judge")
    splits = json.loads((dataset / "splits.json").read_text())
    victim = splits["eval"][0]
    splits["eval"][0] = "ca-01"
    (dataset / "splits.json").write_text(json.dumps(splits))
    items = (dataset / "items.jsonl").read_text().replace(f'"{victim}"', '"ca-01"')
    (dataset / "items.jsonl").write_text(items)
    with pytest.raises(imp.ImportRefused, match="side-check"):
        _import(root, "guard-judge", dataset)


# --- what the importer writes, and what it never writes -----------------------------------------


@pytest.mark.parametrize("role", ROLES)
def test_prompts_are_never_written(tmp_path, root, role):
    """Impl review P22-n1: `prompts/` is committed pack data; an import leaves it byte-identical,
    and the pack directory gains only `pack.json`, `items.jsonl` and `PROVENANCE.md`."""
    pack_dir = root / "packs" / f"{role}-synthetic"
    prompts_before = _snapshot(pack_dir / "prompts")
    _import(root, role, _dataset(tmp_path, role))
    assert _snapshot(pack_dir / "prompts") == prompts_before
    assert set(_snapshot(pack_dir)) == {*(f"prompts/{p}" for p in prompts_before),
                                        "pack.json", "items.jsonl", "PROVENANCE.md"}


def test_an_edited_prompt_is_kept_not_overwritten(tmp_path, root):
    prompt = root / "packs" / "join-in-judge-synthetic" / "prompts" / "join-in-handoff.md"
    prompt.write_text("a locally edited prompt\n", "utf-8")
    _import(root, "join-in-judge", _dataset(tmp_path, "join-in-judge"))
    assert prompt.read_text("utf-8") == "a locally edited prompt\n"


def test_a_missing_pack_directory_is_refused_and_not_created(tmp_path, root):
    shutil.rmtree(root / "packs" / "join-in-judge-synthetic")
    with pytest.raises(imp.ImportRefused, match="never creates"):
        _import(root, "join-in-judge", _dataset(tmp_path, "join-in-judge"))
    assert not (root / "packs" / "join-in-judge-synthetic").exists()


@pytest.mark.parametrize("role", ROLES)
def test_a_missing_prompt_is_refused_and_not_written(tmp_path, root, role):
    prompt = root / "packs" / f"{role}-synthetic" / imp.PROMPT_BY_ROLE[role]
    prompt.unlink()
    with pytest.raises(imp.ImportRefused, match="never writes prompts"):
        _import(root, role, _dataset(tmp_path, role))
    assert not prompt.exists()
    assert not (root / "packs" / f"{role}-synthetic" / "pack.json").exists()


def test_a_reimport_under_an_unchanged_pack_version_is_refused(tmp_path, root):
    dataset = _dataset(tmp_path, "join-in-judge")
    _import(root, "join-in-judge", dataset, version="1.0.0")
    pack_dir = root / "packs" / "join-in-judge-synthetic"
    before = _snapshot(pack_dir)
    with pytest.raises(imp.ImportRefused, match="unchanged"):
        _import(root, "join-in-judge", dataset, version="1.0.0")
    assert _snapshot(pack_dir) == before


def test_the_version_gate_reads_provenance_too(tmp_path, root):
    dataset = _dataset(tmp_path, "join-in-judge")
    _import(root, "join-in-judge", dataset, version="1.0.0")
    (root / "packs" / "join-in-judge-synthetic" / "pack.json").unlink()
    with pytest.raises(imp.ImportRefused, match="unchanged"):
        _import(root, "join-in-judge", dataset, version="1.0.0")


def test_a_new_pack_version_overwrites_the_three_files(tmp_path, root):
    """Not exclusive create (P22-n1): a second import under a new version rewrites the files."""
    _import(root, "join-in-judge", _dataset(tmp_path / "a", "join-in-judge"), version="1.0.0")
    second = _dataset(tmp_path / "b", "join-in-judge", evals=MIN_EVAL_ITEMS + 5)
    result = _import(root, "join-in-judge", second, version="1.1.0")
    pack = load_pack(root / "packs" / "join-in-judge-synthetic")
    assert pack.packVersion == "1.1.0" and result.evalCount == MIN_EVAL_ITEMS + 5
    assert validate_pack(pack) == []


def test_a_pack_that_would_not_validate_is_refused_and_nothing_is_written(tmp_path, root):
    pack_dir = root / "packs" / "join-in-judge-synthetic"
    before = _snapshot(pack_dir)
    with pytest.raises(imp.ImportRefused, match="MIN_EVAL_ITEMS"):
        _import(root, "join-in-judge",
                _dataset(tmp_path, "join-in-judge", evals=MIN_EVAL_ITEMS - 1))
    assert _snapshot(pack_dir) == before


def test_a_dataset_of_another_role_is_refused(tmp_path, root):
    with pytest.raises(imp.ImportRefused, match="not a join-in-judge dataset"):
        _import(root, "join-in-judge", _dataset(tmp_path, "message-type-classifier"))


@pytest.mark.parametrize(
    ("splits", "match"),
    [({"eval": ["nope"]}, "absent from items"), ({"eval": "dm-0001"}, "list"), ({}, "list"),
     ({"eval": ["dm-0001", "dm-0001"]}, "repeats")],
)
def test_a_bad_splits_file_is_refused(tmp_path, root, splits, match):
    dataset = _dataset(tmp_path, "join-in-judge")
    (dataset / "splits.json").write_text(json.dumps(splits))
    with pytest.raises(imp.ImportRefused, match=match):
        _import(root, "join-in-judge", dataset)


@pytest.mark.parametrize("version", ["", "1.0 .0", " 1.0.0"])
def test_a_pack_version_that_is_not_one_token_is_refused(tmp_path, root, version):
    with pytest.raises(imp.ImportRefused, match="pack-version"):
        _import(root, "join-in-judge", _dataset(tmp_path, "join-in-judge"), version=version)


# --- the role tables ---------------------------------------------------------------------------


def test_the_role_tables_are_keyed_exactly_like_bar_kind_by_role():
    for table in (imp.INCUMBENT_BY_ROLE, imp.PROMPT_BY_ROLE, imp.PROVENANCE_FIELDS_BY_ROLE):
        assert set(table) == set(BAR_KIND_BY_ROLE)


def test_the_incumbent_is_named_iff_the_bar_is_a_speed_up():
    for role, kind in BAR_KIND_BY_ROLE.items():
        assert (imp.INCUMBENT_BY_ROLE[role] is not None) == (kind == "speed-up"), role


def test_the_guard_incumbent_is_the_decision_point_fixtures():
    fixture = json.loads((_TESTS / "fixtures" / "packs" / "decision_point_guard" / "pack.json")
                         .read_text())
    assert imp.INCUMBENT_BY_ROLE["guard-judge"] == fixture["decisionPoint"]["incumbent"]


def test_every_named_prompt_is_committed_pack_data():
    for role, prompt in imp.PROMPT_BY_ROLE.items():
        assert (_MB / "packs" / f"{role}-synthetic" / prompt).is_file(), role


def test_the_guard_handoff_prompt_is_a_byte_copy_of_the_incumbents_judge_prompt():
    assert (_MB / "packs" / "guard-judge-synthetic" / "prompts" / "judge.md").read_bytes() == (
        _MB / "packs" / "guard-judge-understanding" / "prompts" / "judge.md").read_bytes()


def test_no_provenance_field_is_an_input_field():
    for role, fields in imp.PROVENANCE_FIELDS_BY_ROLE.items():
        assert not set(fields) & set(INPUT_FIELDS_BY_ROLE[role]), role
        assert {"genRunId", "genRound", "seedId", "labelVotes"} <= set(fields)
    assert "threadPosition" in imp.PROVENANCE_FIELDS_BY_ROLE["message-type-classifier"]
    assert "threadPosition" not in imp.PROVENANCE_FIELDS_BY_ROLE["join-in-judge"]


# --- the candidate half ------------------------------------------------------------------------


def _candidate_source(base: Path, *, manifest_lines: list[str] | None = None,
                      candidate_id: str = "guard-laya-band-v1") -> Path:
    """`base/fits/` with a candidate, its fit manifest and a fit report beside them."""
    fits = base / "fits"
    fits.mkdir(parents=True)
    lines = manifest_lines if manifest_lines is not None else sorted(
        hashlib.sha256(str(n).encode()).hexdigest() for n in range(3))
    manifest = "".join(line + "\n" for line in lines).encode()
    (fits / f"{candidate_id}.fit.txt").write_bytes(manifest)
    (fits / f"{candidate_id}.fit-report.json").write_text('{"auc": 0.9}')
    body = {
        "candidateId": candidate_id,
        "role": "guard-judge",
        "decisionModel": "laya",
        "question": {"type": "predicate", "name": "advance", "instructions": "Is it satisfied?"},
        "policy": {"kind": "predicate-band", "yesAtOrAbove": 0.92, "noBelow": 0.08},
        "handoffModel": "qwen/qwen3-4b-2507",
        "decisionTimeoutSeconds": 2.0,
        "fit": {"outcome": "below-margin", "qualityTarget": "mcnemar-nonreject"},
        "fitManifest": {"file": f"{candidate_id}.fit.txt",
                        "sha256": hashlib.sha256(manifest).hexdigest(), "size": len(lines),
                        "purposes": ["calibration"]},
        "provenance": {"builtBy": "decision-models", "datasetVersion": "2026-11-01",
                       "sourceSha": "825d5d93"},
    }
    path = fits / f"{candidate_id}.json"
    path.write_text(json.dumps(body, indent=2))
    return path


def test_a_candidate_and_its_fit_manifest_are_copied_and_load(tmp_path):
    root = tmp_path / "mb"
    root.mkdir()
    source = _candidate_source(tmp_path / "dm")
    result = imp.import_candidate(root, source)
    assert sorted(result.written) == ["guard-laya-band-v1.fit.txt", "guard-laya-band-v1.json"]
    assert sorted(p.name for p in (root / "candidates").iterdir()) == sorted(result.written)
    candidate = Candidate.load(root, "guard-laya-band-v1")
    assert candidate.fit["outcome"] == "below-margin"
    assert load_fit_manifest(root, candidate.fitManifest).size == 3
    assert (root / "candidates" / "guard-laya-band-v1.json").read_bytes() == source.read_bytes()


def test_the_fit_report_is_never_copied(tmp_path):
    root = tmp_path / "mb"
    root.mkdir()
    imp.import_candidate(root, _candidate_source(tmp_path / "dm"))
    assert not any("fit-report" in p.name for p in (root / "candidates").iterdir())


def test_a_reimport_of_identical_bytes_is_a_no_op(tmp_path):
    root = tmp_path / "mb"
    root.mkdir()
    source = _candidate_source(tmp_path / "dm")
    imp.import_candidate(root, source)
    again = imp.import_candidate(root, source)
    assert again.written == () and len(again.unchanged) == 2


def test_a_candidate_with_other_bytes_under_an_existing_id_is_refused(tmp_path):
    root = tmp_path / "mb"
    root.mkdir()
    imp.import_candidate(root, _candidate_source(tmp_path / "a"))
    before = _snapshot(root / "candidates")
    edited = _candidate_source(tmp_path / "b", manifest_lines=[])
    with pytest.raises(imp.ImportRefused, match="other bytes"):
        imp.import_candidate(root, edited)
    assert _snapshot(root / "candidates") == before


def test_a_candidate_that_would_not_load_is_refused_and_nothing_is_written(tmp_path):
    root = tmp_path / "mb"
    root.mkdir()
    source = _candidate_source(tmp_path / "dm")
    body = json.loads(source.read_text())
    body["fitManifest"]["size"] = 4
    source.write_text(json.dumps(body))
    with pytest.raises(imp.ImportRefused, match="would not load"):
        imp.import_candidate(root, source)
    assert not (root / "candidates").exists()


def test_a_candidate_whose_fit_manifest_is_absent_is_refused(tmp_path):
    root = tmp_path / "mb"
    root.mkdir()
    source = _candidate_source(tmp_path / "dm")
    (source.parent / "guard-laya-band-v1.fit.txt").unlink()
    with pytest.raises(imp.ImportRefused, match="cannot read"):
        imp.import_candidate(root, source)


def test_a_candidate_file_not_named_for_its_id_is_refused(tmp_path):
    root = tmp_path / "mb"
    root.mkdir()
    source = _candidate_source(tmp_path / "dm")
    renamed = source.with_name("other.json")
    source.rename(renamed)
    with pytest.raises(imp.ImportRefused, match="candidateId"):
        imp.import_candidate(root, renamed)


# --- CLI and git --------------------------------------------------------------------------------


def test_main_rejects_mixed_or_partial_modes(tmp_path, capsys):
    assert imp.main(["--candidate", "x.json", "--role", "guard-judge"]) == 2
    assert imp.main(["--role", "guard-judge"]) == 2


def test_main_imports_a_candidate_and_reports_a_refusal(tmp_path, capsys):
    root = tmp_path / "mb"
    root.mkdir()
    source = _candidate_source(tmp_path / "dm")
    assert imp.main(["--candidate", str(source), "--root", str(root)]) == 0
    assert imp.main(["--candidate", str(tmp_path / "missing.json"), "--root", str(root)]) == 1
    assert "refused" in capsys.readouterr().err


def test_main_refuses_an_uncommitted_dataset(tmp_path, root, capsys):
    """The real `git_file_sha` path: a dataset outside any git history has no source SHA."""
    dataset = _dataset(tmp_path, "join-in-judge")
    code = imp.main(["--role", "join-in-judge", "--dataset", str(dataset), "--pack-version",
                     "1.0.0", "--root", str(root)])
    assert code == 1
    assert not (root / "packs" / "join-in-judge-synthetic" / "pack.json").exists()


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false", *args],
        cwd=cwd, capture_output=True, text=True, check=True,
    ).stdout.strip()


def test_git_file_sha_names_the_last_commit_and_refuses_an_uncommitted_file(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "a.txt").write_text("one\n")
    _git(repo, "add", "a.txt")
    _git(repo, "commit", "-q", "-m", "a")
    assert imp.git_file_sha(repo / "a.txt") == _git(repo, "rev-parse", "HEAD")
    (repo / "a.txt").write_text("two\n")
    with pytest.raises(imp.ImportRefused, match="modified"):
        imp.git_file_sha(repo / "a.txt")
    (repo / "b.txt").write_text("new\n")
    with pytest.raises(imp.ImportRefused, match="untracked"):
        imp.git_file_sha(repo / "b.txt")
