#!/usr/bin/env python3
"""scripts/import_decision_pack.py — the one-way, human-invoked importer of decision-models'
datasets and candidates into model-bench (typed-decision-models v1.3 §3.4 "Importer", amended by
plan2 §4.1 U3d and plan4 §4.2 U3d; mirrors `scripts/refresh_golden.py`).

**Never imported by any run-path module (FR-23) and never invoked automatically.** It reads
decision-models files as data only — no decision-models code is imported — and writes only under
model-bench. Two exclusive modes, the plan's "both halves":

* **`--role <r> --dataset <dir> --pack-version <v>`** copies the **eval split only** of one
  dataset version (`decision-models/data/<r>/<datasetVersion>/`) into the existing
  `packs/<r>-synthetic/`, writing `pack.json`, `items.jsonl` and `PROVENANCE.md` there by plain
  overwrite (never exclusive create) and nothing else. **`prompts/` is committed pack data and is
  never written**: the pack directory and its role's prompt must already exist, or the import is
  refused. On `guard-judge` the 85 human-curated items of `packs/guard-judge-understanding/` are
  appended as the `side-check` stratum (v1.3 §3.5, FR-15). The result must pass
  `packs.validate_pack` in a staging copy before anything in the pack directory is touched.
  Re-import under an unchanged `packVersion` is refused, as in `refresh_golden.py`.
* **`--candidate <file>`** copies a candidate (carrying `fit`, plan2 §3.2) and the fit manifest
  its `fitManifest.file` names, from the candidate's own directory, into `candidates/`. Both are
  verified by `decisions.Candidate.load` and `decisions.load_fit_manifest` in a staging copy
  first. **The fit report is never copied** (plan2 §3.5): nothing but those two files is read.
  An existing destination file with other bytes is refused; identical bytes are a no-op.

**The source formats this reads.** `generation.json` is plan4 §3.7's; its `labeler`/`generator`
blocks become the pack's `labels.labeler`/`labels.generator` as `{provider: "anthropic", model,
surface, effort}` (plan4 §4.2), and `labelBatchCap` goes to `PROVENANCE.md`. `splits.json` and
`labelVotes` are plan6 §3.7's (`docs/plans/typed-decision-models6.md`): `splits.json` holds
`rngSeed`, `eval`, `train` and `calibration`, and only `eval` is read (each `itemId` once). Each
dataset item is a JSON object with a unique `itemId`. The contract vector
`tests/fixtures/decision-contract/dataset-shape.jsonl` pins all three files' shapes per role;
decision-models' `finalize` (U5.10) checks its output against a byte-copy of it.

**What an eval item keeps** (`_project_eval_item`): `itemId`, the role's
`decisions.INPUT_FIELDS_BY_ROLE` fields, `expected`, `origin`, and the provenance fields
`genRunId`, `genRound`, `seedId`, `labelVotes` (plus `threadPosition` on message type, plan4
§4.2), each required; `stratum` is written as `"eval"`. Any other source field is not copied.
`labelVotes` must be counts summing to 3 whose keys are exactly the role's answer space written as
strings (`label_vote_keys`: `"true"`/`"false"` on a predicate point, lowercase; each label on a
choice point), zeros included.
None of the provenance fields is an input field, so `input_hash` is unchanged by them (plan4 §5).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

from modelbench.decisions import (
    ANSWER_SPACE_BY_ROLE,
    BAR_KIND_BY_ROLE,
    INPUT_FIELDS_BY_ROLE,
    SCORER_BY_ROLE,
    Candidate,
    CandidateError,
    load_fit_manifest,
)
from modelbench.packs import PackConfigError, load_pack, validate_pack

#: scripts/import_decision_pack.py -> model-bench/
_DEFAULT_ROOT = Path(__file__).resolve().parents[1]

#: v1.3 §3.5: `decisionPoint.incumbent`, written explicitly on every import (impl review P12
#: point 2): the guard's incumbent judge on a speed-up point, `None` on a new-ability point.
#: Keyed exactly like `BAR_KIND_BY_ROLE`.
INCUMBENT_BY_ROLE: Mapping[str, str | None] = MappingProxyType(
    {
        "guard-judge": "qwen/qwen3-4b-2507",
        "join-in-judge": None,
        "message-type-classifier": None,
    }
)

#: The committed hand-off prompt each pack names as `prompt.systemPrompt` (impl review P22-n1:
#: `prompts/<point>-handoff.md` on the new abilities; the guard's is the incumbent's own judge
#: prompt). Pack data the importer never writes. Keyed exactly like `BAR_KIND_BY_ROLE`.
PROMPT_BY_ROLE: Mapping[str, str] = MappingProxyType(
    {
        "guard-judge": "prompts/judge.md",
        "join-in-judge": "prompts/join-in-handoff.md",
        "message-type-classifier": "prompts/message-type-handoff.md",
    }
)

#: plan4 §4.2 U3d / §3.7: the per-item provenance fields copied onto every eval item, per role.
#: None is in `INPUT_FIELDS_BY_ROLE`. Keyed exactly like `BAR_KIND_BY_ROLE`.
_COMMON_PROVENANCE_FIELDS = ("genRunId", "genRound", "seedId", "labelVotes")
PROVENANCE_FIELDS_BY_ROLE: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "guard-judge": _COMMON_PROVENANCE_FIELDS,
        "join-in-judge": _COMMON_PROVENANCE_FIELDS,
        "message-type-classifier": (*_COMMON_PROVENANCE_FIELDS, "threadPosition"),
    }
)

#: The sibling pack whose items become the guard's `side-check` stratum (v1.3 §3.5, FR-15).
SIDE_CHECK_SOURCE_PACK = "guard-judge-understanding"
SIDE_CHECK_ROLE = "guard-judge"

#: The pack's bootstrap seed (`sampling.seed`). The plans leave it open; one fixed value keeps a
#: re-import's intervals comparable with the previous import's.
SAMPLING_SEED = 20261101

#: The labeler/generator blocks' keys copied from `generation.json` (plan4 §4.2).
_MODEL_BLOCK_KEYS = ("model", "surface", "effort")
_PROVIDER = "anthropic"

_PACK_VERSION_RE = re.compile(r"\*\*Pack version:\*\*\s*(\S+)")
_WRITTEN_FILES = ("pack.json", "items.jsonl", "PROVENANCE.md")


class ImportRefused(RuntimeError):
    """The import cannot proceed as asked. Raised before any write to the destination."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_file_sha(path: Path) -> str:
    """The last commit that touched `path` (`git log -1`, as `refresh_golden._git_file_sha`),
    refused when the file is untracked or differs from that commit — an uncommitted source has no
    SHA that names its bytes."""
    cwd = path.parent
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain", "--", path.name],
            cwd=cwd, capture_output=True, text=True, check=True,
        )
        log = subprocess.run(
            ["git", "log", "-1", "--format=%H", "--", path.name],
            cwd=cwd, capture_output=True, text=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ImportRefused(f"cannot read git history of {path}: {exc}") from exc
    if status.stdout.strip():
        raise ImportRefused(f"{path} is untracked or modified; commit it before importing")
    sha = log.stdout.strip()
    if not sha:
        raise ImportRefused(f"{path} has no git history; commit it before importing")
    return sha


# --- reading the dataset -----------------------------------------------------------------------


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ImportRefused(f"cannot read {path}: {exc}") from exc


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ImportRefused(f"cannot read {path}: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise ImportRefused(f"{path}:{number}: not JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ImportRefused(f"{path}:{number}: not a JSON object")
        rows.append(row)
    return rows


def _model_block(generation: Mapping[str, Any], key: str) -> dict[str, str]:
    block = generation.get(key)
    if not isinstance(block, Mapping):
        raise ImportRefused(f"generation.json: {key!r} is absent or not an object")
    out = {"provider": _PROVIDER}
    for field in _MODEL_BLOCK_KEYS:
        value = block.get(field)
        if not (isinstance(value, str) and value):
            raise ImportRefused(f"generation.json: {key}.{field} is absent or empty: {value!r}")
        out[field] = value
    return out


def _label_batch_cap(generation: Mapping[str, Any]) -> int:
    cap = generation.get("labelBatchCap")
    if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
        raise ImportRefused(f"generation.json: labelBatchCap must be an int >= 1, got {cap!r}")
    return cap


def label_vote_keys(role: str) -> frozenset[str]:
    """The exact `labelVotes` key set of a role's items (plan6 §3.7): its answer space written as
    strings. A JSON boolean is never a key, so a predicate's spelling is fixed lowercase."""
    space = ANSWER_SPACE_BY_ROLE[role]
    if space["type"] == "predicate":
        return frozenset({"true", "false"})
    return frozenset(space["labels"])


def _eval_ids(splits: Any) -> list[str]:
    ids = splits.get("eval") if isinstance(splits, Mapping) else None
    if not isinstance(ids, list) or not all(isinstance(i, str) and i for i in ids):
        raise ImportRefused("splits.json: 'eval' must be a list of itemId strings")
    if len(ids) != len(set(ids)):
        raise ImportRefused("splits.json: 'eval' repeats an itemId")
    return ids


def _project_eval_item(row: Mapping[str, Any], role: str) -> dict[str, Any]:
    """One eval item in the pack's form: the fields named in the module docstring, each required,
    in a fixed order, with `stratum: "eval"`."""
    item_id = row.get("itemId")
    required = (
        "itemId",
        *INPUT_FIELDS_BY_ROLE[role],
        "expected",
        "origin",
        *PROVENANCE_FIELDS_BY_ROLE[role],
    )
    missing = [field for field in required if field not in row]
    if missing:
        raise ImportRefused(f"eval item {item_id!r} lacks {missing}")
    gen_run_id, gen_round, votes = row["genRunId"], row["genRound"], row["labelVotes"]
    if not (isinstance(gen_run_id, str) and gen_run_id):
        raise ImportRefused(f"eval item {item_id!r}: genRunId must be a non-empty string")
    if isinstance(gen_round, bool) or not isinstance(gen_round, int) or gen_round < 1:
        raise ImportRefused(f"eval item {item_id!r}: genRound must be an int >= 1")
    if not (
        isinstance(votes, Mapping)
        and all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in votes.values())
        and sum(votes.values()) == 3
    ):
        raise ImportRefused(f"eval item {item_id!r}: labelVotes must be counts summing to 3")
    if set(votes) != label_vote_keys(role):
        raise ImportRefused(
            f"eval item {item_id!r}: labelVotes keys {sorted(votes)} are not the answer space "
            f"{sorted(label_vote_keys(role))}"
        )
    projected = {field: row[field] for field in required}
    projected["stratum"] = "eval"
    return projected


def _side_check_rows(root: Path) -> tuple[list[dict[str, Any]], Path]:
    """The sibling guard pack's human-curated items, stamped `side-check` and `labelSource: human`
    (v1.3 §3.5); otherwise as written there."""
    path = root / "packs" / SIDE_CHECK_SOURCE_PACK / "items.jsonl"
    rows = [
        {**row, "stratum": "side-check", "labelSource": "human"} for row in _read_jsonl(path)
    ]
    if not rows:
        raise ImportRefused(f"{path} holds no side-check items")
    return rows, path


# --- the pack ----------------------------------------------------------------------------------


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return list(value)
    return value


def _verdict_metrics(role: str) -> list[str]:
    """The bar kind's exact `verdictMetrics` (`packs._verdict_metrics_problems`): the guard's four
    per-kind rates, a new ability's per-class agreement set."""
    if BAR_KIND_BY_ROLE[role] == "speed-up":
        from modelbench.scoring.classification import PER_KIND_VERDICT_METRICS

        return list(PER_KIND_VERDICT_METRICS)
    from modelbench.scoring.agreement import per_class_agreement_metrics

    return list(per_class_agreement_metrics(ANSWER_SPACE_BY_ROLE[role]))


def build_manifest(
    role: str, pack_version: str, labeler: Mapping[str, str], generator: Mapping[str, str]
) -> dict[str, Any]:
    return {
        "packId": f"{role}-synthetic",
        "packVersion": pack_version,
        "role": role,
        "schemaVersion": 1,
        "description": (
            f"typed-decision-models: the {role} decision point's synthetic eval split, imported "
            "from decision-models by scripts/import_decision_pack.py. Labels are an unreviewed "
            "LLM's (FR-5)."
        ),
        "scorer": SCORER_BY_ROLE[role],
        "environment": {"requires": ["lmstudio-chat", "lmstudio-decisions"]},
        "prompt": {
            "systemPrompt": PROMPT_BY_ROLE[role],
            "toolSchemas": None,
            "temperature": 0.0,
            "maxTokens": 256,
        },
        "data": {"items": "items.jsonl"},
        "sampling": {"seed": SAMPLING_SEED, "pairingKey": ["itemId"], "analysisUnit": "itemId"},
        "metrics": {"verdictMetrics": _verdict_metrics(role), "headlineMetric": None},
        "decisionPoint": {
            "answer": _plain(ANSWER_SPACE_BY_ROLE[role]),
            "incumbent": INCUMBENT_BY_ROLE[role],
        },
        "labels": {
            "source": "llm-unreviewed",
            "labeler": dict(labeler),
            "generator": dict(generator),
            "inputs": "synthetic",
        },
        "provenance": "PROVENANCE.md",
    }


@dataclass(frozen=True)
class SourceFile:
    origin: str
    destination: str
    gitSha: str
    sha256: str


def _display_path(path: Path, root: Path) -> str:
    """`path` relative to the monorepo root (model-bench's parent) when it lies under it."""
    try:
        return path.resolve().relative_to(root.resolve().parent).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def render_provenance(
    *,
    role: str,
    pack_version: str,
    dataset_version: str,
    imported_at: str,
    sources: Sequence[SourceFile],
    labeler: Mapping[str, str],
    generator: Mapping[str, str],
    label_batch_cap: int,
    eval_count: int,
    side_check_count: int,
) -> str:
    def block(title: str, values: Mapping[str, str]) -> list[str]:
        return [f"## {title}", "", *(f"- **{k}:** `{v}`" for k, v in values.items()), ""]

    lines = [
        f"# Provenance — {role}-synthetic",
        "",
        f"> **Pack version:** {pack_version} · **Imported:** {imported_at} · "
        f"**Dataset version:** {dataset_version}",
        "",
        "Written by `model-bench/scripts/import_decision_pack.py` (typed-decision-models v1.3 "
        "§3.4; plan4 §4.2 U3d): the dataset's eval split only, copied one-way from "
        "decision-models. `prompts/` is committed pack data the importer never writes. "
        "Re-importing needs a new `--pack-version`; an unchanged one is refused.",
        "",
        f"Items: {eval_count} in stratum `eval`, {side_check_count} in stratum `side-check`.",
        "",
        "| Origin | Destination | Source git SHA | Source SHA-256 |",
        "|---|---|---|---|",
        *(f"| `{s.origin}` | {s.destination} | `{s.gitSha}` | `{s.sha256}` |" for s in sources),
        "",
        *block("Labeler (`labels.labeler`)", labeler),
        *block("Generator (`labels.generator`)", generator),
        "## Labelling",
        "",
        f"- **labelBatchCap:** `{label_batch_cap}`",
        "",
    ]
    return "\n".join(lines)


def _recorded_versions(pack_dir: Path) -> set[str]:
    versions: set[str] = set()
    manifest = pack_dir / "pack.json"
    if manifest.is_file():
        version = _read_json(manifest).get("packVersion")
        if isinstance(version, str):
            versions.add(version)
    provenance = pack_dir / "PROVENANCE.md"
    if provenance.is_file():
        match = _PACK_VERSION_RE.search(provenance.read_text(encoding="utf-8"))
        if match:
            versions.add(match.group(1))
    return versions


@dataclass(frozen=True)
class DatasetImport:
    packDir: Path
    evalCount: int
    sideCheckCount: int
    contentHash: str


def import_dataset(
    root: Path,
    role: str,
    dataset: Path,
    pack_version: str,
    *,
    source_sha: Callable[[Path], str] = git_file_sha,
    now: Callable[[], str] = _utc_now_iso,
) -> DatasetImport:
    """The dataset half (module docstring). Every refusal raises `ImportRefused` before the pack
    directory is written."""
    if role not in BAR_KIND_BY_ROLE:
        raise ImportRefused(f"role {role!r} is not a decision point ({sorted(BAR_KIND_BY_ROLE)})")
    if not pack_version or any(c.isspace() for c in pack_version):
        raise ImportRefused(f"--pack-version {pack_version!r} must be one non-empty token")
    pack_dir = root / "packs" / f"{role}-synthetic"
    if not pack_dir.is_dir():
        raise ImportRefused(
            f"{pack_dir} does not exist; the importer never creates a pack directory (its "
            "prompts/ are committed pack data)"
        )
    if not (pack_dir / PROMPT_BY_ROLE[role]).is_file():
        raise ImportRefused(
            f"{pack_dir / PROMPT_BY_ROLE[role]} is missing; the importer never writes prompts/"
        )
    if pack_version in _recorded_versions(pack_dir):
        raise ImportRefused(
            f"packVersion {pack_version!r} is unchanged since the last import into {pack_dir}; "
            "pass a new --pack-version"
        )
    if dataset.parent.name != role:
        raise ImportRefused(
            f"{dataset} is not a {role} dataset version (expected data/{role}/<datasetVersion>/)"
        )

    generation_path = dataset / "generation.json"
    items_path = dataset / "items.jsonl"
    splits_path = dataset / "splits.json"
    generation = _read_json(generation_path)
    if not isinstance(generation, Mapping):
        raise ImportRefused(f"{generation_path}: not a JSON object")
    labeler = _model_block(generation, "labeler")
    generator = _model_block(generation, "generator")
    if labeler["model"] == generator["model"]:
        raise ImportRefused(
            f"generation.json: the generator and the labeler are one model {labeler['model']!r}"
        )
    label_batch_cap = _label_batch_cap(generation)

    rows = _read_jsonl(items_path)
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        item_id = row.get("itemId")
        if not (isinstance(item_id, str) and item_id):
            raise ImportRefused(f"{items_path}: an item has no itemId string")
        if item_id in by_id:
            raise ImportRefused(f"{items_path}: itemId {item_id!r} repeats")
        by_id[item_id] = row
    eval_ids = set(_eval_ids(_read_json(splits_path)))
    unknown = sorted(eval_ids - by_id.keys())
    if unknown:
        raise ImportRefused(f"splits.json names eval items absent from items.jsonl: {unknown[:5]}")
    pack_rows = [_project_eval_item(row, role) for row in rows if row["itemId"] in eval_ids]

    sources = [
        SourceFile(
            _display_path(items_path, root),
            f"`items.jsonl`, stratum `eval` ({len(pack_rows)} items)",
            source_sha(items_path),
            _sha256_bytes(items_path.read_bytes()),
        ),
        SourceFile(
            _display_path(splits_path, root),
            "eval membership",
            source_sha(splits_path),
            _sha256_bytes(splits_path.read_bytes()),
        ),
        SourceFile(
            _display_path(generation_path, root),
            "`pack.json` `labels`; this file",
            source_sha(generation_path),
            _sha256_bytes(generation_path.read_bytes()),
        ),
    ]
    side_count = 0
    if role == SIDE_CHECK_ROLE:
        side_rows, side_path = _side_check_rows(root)
        side_count = len(side_rows)
        clash = sorted({r["itemId"] for r in side_rows} & eval_ids)
        if clash:
            raise ImportRefused(f"eval itemIds repeat side-check itemIds: {clash[:5]}")
        pack_rows.extend(side_rows)
        sources.append(
            SourceFile(
                _display_path(side_path, root),
                f"`items.jsonl`, stratum `side-check` ({side_count} items)",
                "— (model-bench)",
                _sha256_bytes(side_path.read_bytes()),
            )
        )

    manifest = build_manifest(role, pack_version, labeler, generator)
    contents = {
        "pack.json": json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        "items.jsonl": "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in pack_rows),
        "PROVENANCE.md": render_provenance(
            role=role,
            pack_version=pack_version,
            dataset_version=dataset.name,
            imported_at=now(),
            sources=sources,
            labeler=labeler,
            generator=generator,
            label_batch_cap=label_batch_cap,
            eval_count=len(pack_rows) - side_count,
            side_check_count=side_count,
        ),
    }

    with tempfile.TemporaryDirectory(prefix="import-decision-pack-") as scratch:
        staged = Path(scratch) / pack_dir.name
        shutil.copytree(pack_dir, staged)
        for name, text in contents.items():
            (staged / name).write_text(text, encoding="utf-8")
        try:
            problems = validate_pack(load_pack(staged))
        except PackConfigError as exc:
            problems = [str(exc)]
        if problems:
            raise ImportRefused("the imported pack would not validate:\n  " + "\n  ".join(problems))

    for name in _WRITTEN_FILES:
        (pack_dir / name).write_text(contents[name], encoding="utf-8")
    pack = load_pack(pack_dir)
    return DatasetImport(pack_dir, len(pack_rows) - side_count, side_count, pack.contentHash)


# --- the candidate half ------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateImport:
    candidateId: str
    written: tuple[str, ...]
    unchanged: tuple[str, ...]


def import_candidate(root: Path, candidate_file: Path) -> CandidateImport:
    """The candidate half (module docstring): the candidate file and the fit manifest it names,
    verified in a staging `candidates/` before either lands in `<root>/candidates/`."""
    body = _read_json(candidate_file)
    if not isinstance(body, Mapping):
        raise ImportRefused(f"{candidate_file}: not a JSON object")
    candidate_id = body.get("candidateId")
    if candidate_file.name != f"{candidate_id}.json":
        raise ImportRefused(
            f"{candidate_file}: the file name is not <candidateId>.json ({candidate_id!r})"
        )
    fit_manifest = body.get("fitManifest")
    manifest_name = fit_manifest.get("file") if isinstance(fit_manifest, Mapping) else None
    if not (isinstance(manifest_name, str) and manifest_name and "/" not in manifest_name):
        raise ImportRefused(f"{candidate_file}: fitManifest.file is not a plain file name")
    manifest_file = candidate_file.parent / manifest_name
    files = {candidate_file.name: candidate_file, manifest_name: manifest_file}

    with tempfile.TemporaryDirectory(prefix="import-candidate-") as scratch:
        staged = Path(scratch) / "candidates"
        staged.mkdir()
        for name, source in files.items():
            try:
                shutil.copyfile(source, staged / name)
            except OSError as exc:
                raise ImportRefused(f"cannot read {source}: {exc}") from exc
        try:
            candidate = Candidate.load(scratch, candidate_id)
            load_fit_manifest(scratch, candidate.fitManifest)
        except CandidateError as exc:
            raise ImportRefused(f"the candidate would not load: {exc}") from exc
        staged_bytes = {name: (staged / name).read_bytes() for name in files}

    destination = root / "candidates"
    written: list[str] = []
    unchanged: list[str] = []
    for name, data in staged_bytes.items():
        target = destination / name
        if target.exists():
            if target.read_bytes() != data:
                raise ImportRefused(
                    f"{target} exists with other bytes; a candidate edited in place is a new "
                    "candidate, so give it a new candidateId (and fit manifest name)"
                )
            unchanged.append(name)
    destination.mkdir(exist_ok=True)
    for name, data in staged_bytes.items():
        if name not in unchanged:
            (destination / name).write_bytes(data)
            written.append(name)
    return CandidateImport(candidate.candidateId, tuple(written), tuple(unchanged))


# --- CLI ---------------------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="import_decision_pack.py",
        description="Import a decision-models dataset's eval split into packs/<role>-synthetic/, "
        "or a candidate and its fit manifest into candidates/ (typed-decision-models v1.3 §3.4).",
    )
    parser.add_argument("--role", choices=sorted(BAR_KIND_BY_ROLE))
    parser.add_argument("--dataset", help="decision-models/data/<role>/<datasetVersion>/")
    parser.add_argument("--pack-version", help="the new packVersion (a re-import must change it)")
    parser.add_argument(
        "--candidate", help="a candidate JSON file; its fit manifest sits beside it"
    )
    parser.add_argument(
        "--root", default=str(_DEFAULT_ROOT), help="the model-bench directory (default: this one)"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    root = Path(args.root)
    dataset_args = (args.role, args.dataset, args.pack_version)
    if args.candidate is not None:
        if any(a is not None for a in dataset_args):
            parser.print_usage(sys.stderr)
            print("import_decision_pack.py: --candidate excludes --role/--dataset/--pack-version",
                  file=sys.stderr)
            return 2
        try:
            result = import_candidate(root, Path(args.candidate))
        except ImportRefused as exc:
            print(f"import_decision_pack.py: refused: {exc}", file=sys.stderr)
            return 1
        print(
            f"candidate {result.candidateId}: written {list(result.written)}, "
            f"unchanged {list(result.unchanged)}"
        )
        return 0
    if any(a is None for a in dataset_args):
        parser.print_usage(sys.stderr)
        print("import_decision_pack.py: give --role, --dataset and --pack-version, or --candidate",
              file=sys.stderr)
        return 2
    try:
        result = import_dataset(root, args.role, Path(args.dataset), args.pack_version)
    except ImportRefused as exc:
        print(f"import_decision_pack.py: refused: {exc}", file=sys.stderr)
        return 1
    print(
        f"{result.packDir}: {result.evalCount} eval + {result.sideCheckCount} side-check items, "
        f"contentHash {result.contentHash}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
