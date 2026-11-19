"""Fit emission: the calls file, the two candidates and the manifest
(`docs/plans/typed-decision-models4.md` §3.11, §4.4 U6f.9b, §5 row U6f.9b;
`docs/plans/typed-decision-models2.md` §3.2 for the candidate's keys).

**The calls file** `fits/calls-<UTC ts>-<runId>.json` (`write_calls_file`) is written by
`fit.run_fit` after the collection and the end residency read and before any analysis: the
`ScoredCall` fields, `maxProbabilities` and an `endResidency` label (`ok` / `refused`: a model
gone / `unreadable`: the read raised). No item text; not a candidate, not in the manifest. It is
written to a temporary name and linked, so a reader never sees a partial file.

**Candidates** (`build_emission`): one per family, `<role>-<datasetVersion>-handoff-f<N>`
(`handoffModel` = H) and `<role>-<datasetVersion>-only-f<N>` (`handoffModel: null`), with exactly
the keys of model-bench's `Candidate.load`: `candidateId`, `role`, `decisionModel`, `question`
(the chosen text's spec), `policy`, `handoffModel`, `decisionTimeoutSeconds`, `fit`
(`outcome`/`qualityTarget` from `outcome.assemble_outcome` with the role's bar kind),
`fitManifest` (`file`, `sha256`, `size`, `purposes`) and `provenance` (`builtBy`,
`datasetVersion`, `sourceSha`). The ladder's policy maps to the candidate's: `{l, h}` is
`predicate-band {noBelow: l, yesAtOrAbove: h}`, `{t}` (decision-only) the empty band `l == h == t`,
`{c}` `choice-confidence {minConfidence: c}`. **Each candidate's policy and `handoffModel` pass
`policy.check_handoff_rule` before anything is written.**

**`N`** is the fit ordinal of the point: one more than the largest `N` among the
`<role>-<datasetVersion>-(handoff|only)-f<N>.json`, `-f<N>.fit.txt` and `-f<N>.fit-report.json`
files already in `fits/` (1 if none), so a
second fit never reuses an id (the importer refuses an id re-imported with other bytes).

**The manifest** is one file, `<role>-<datasetVersion>-f<N>.fit.txt`, both candidates'
`fitManifest.file`: the `input_hash` of every calibration item the fit loaded (an item a guard
hand-off disposition removed from the analysis was still sent to the models, so it stays),
`purposes` = `["question-tuning", "cutoff"]`. `provenance.sourceSha` is the sha256 of the
dataset's `items.jsonl`: the plan leaves its meaning open, and this one names the bytes the fit
read.

**The report** (`report=True`) is `<role>-<datasetVersion>-f<N>.fit-report.json`, built with the
candidates before anything is written (`report.py`); it is never an entry of the manifest.

**Writing** (`write_files`): every file's bytes are built and checked first; then each is written
to a temporary name and hard-linked to its final name (`os.link`, which refuses an existing
target, never overwriting), the temporaries removed. **This needs a POSIX file system with hard
links** (a failing `os.link` is an `OSError`: `main` reports it as a refusal). The link order is
the manifest, the report, then the candidates (hand-off, only), so a candidate is never visible
without the manifest it names. A failure removes whatever it linked, so nothing partial is left.
The calls file alone falls back to an exclusive create (`open(..., "x")`) when `os.link` fails:
it is about 5N live calls and must not be lost to a file system quirk.

Stdlib + this package only (FR-8). It imports `fit` only for typing, so `fit.run_fit` can call it.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from decisionmodels import manifest as manifest_module
from decisionmodels.fitcalls import CollectedCalls, FitInputs
from decisionmodels.outcome import assemble_outcome
from decisionmodels.policy import check_handoff_rule
from decisionmodels.report import build_report

if TYPE_CHECKING:
    from decisionmodels.fit import FitRun, SelectedText

CALLS_SCHEMA = "decision-models.fit-calls/1"
END_RESIDENCY_LABELS = ("ok", "refused", "unreadable")

BAR_KIND_BY_ROLE: Mapping[str, str] = {
    "guard-judge": "speed-up",
    "join-in-judge": "new-ability",
    "message-type-classifier": "new-ability",
}
SUFFIX_BY_FAMILY = {"handoff": "handoff", "decision-only": "only"}
PURPOSES = ["question-tuning", "cutoff"]
BUILT_BY = "decision-models"


class EmitRefused(RuntimeError):
    """The emission cannot be written as built; nothing was written."""


# --- bytes ---


def _json_bytes(document: Mapping[str, Any], *, strict: bool = True) -> bytes:
    """`document` as UTF-8 JSON. `strict` refuses NaN and infinities (a candidate or report the
    bench could not parse); the calls file is not strict, so a freak value never loses the calls."""
    text = json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=not strict)
    return (text + "\n").encode()


def write_files(directory: Path, files: Mapping[str, bytes]) -> list[Path]:
    """Every file or none (module docstring, "Writing"). A name that exists is `EmitRefused`."""
    directory.mkdir(parents=True, exist_ok=True)
    nonce = uuid.uuid4().hex
    temporaries: list[Path] = []
    linked: list[Path] = []
    try:
        for name, data in files.items():
            temporary = directory / f".{name}.{nonce}.tmp"
            temporary.write_bytes(data)
            temporaries.append(temporary)
        for name, temporary in zip(files, temporaries, strict=True):
            target = directory / name
            try:
                os.link(temporary, target)
            except FileExistsError as exc:
                raise EmitRefused(f"{target} already exists; nothing written") from exc
            linked.append(target)
    except BaseException:
        for path in linked:
            path.unlink(missing_ok=True)
        raise
    finally:
        for temporary in temporaries:
            temporary.unlink(missing_ok=True)
    return linked


# --- the calls file ---


def _stamp(now: Callable[[], datetime] | None) -> str:
    return (now() if now is not None else datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")


def calls_document(
    run_id: str, inputs: FitInputs, collected: CollectedCalls, end_residency: str
) -> dict[str, Any]:
    """The calls file's content: each `ScoredCall` field, `maxProbabilities` as a list of
    `{itemId, textId, maxProbability}` (a tuple cannot be a JSON key), the `endResidency` label."""
    if end_residency not in END_RESIDENCY_LABELS:
        raise ValueError(
            f"endResidency must be one of {END_RESIDENCY_LABELS}, got {end_residency!r}"
        )
    return {
        "schema": CALLS_SCHEMA,
        "runId": run_id,
        "inputs": {
            "role": inputs.role,
            "datasetVersion": inputs.datasetVersion,
            "questionsSha256": inputs.questionsSha256,
            "handoffSha256": inputs.handoffSha256,
            "decisionModel": inputs.decisionModel,
            "handoffModel": inputs.handoffModel,
            "decisionTimeoutSeconds": inputs.decisionTimeoutSeconds,
        },
        "endResidency": end_residency,
        "calls": [
            {
                "itemId": c.itemId,
                "callKind": c.callKind,
                "textId": c.textId,
                "outcome": c.outcome,
                "answer": c.answer,
                "elapsedSeconds": c.elapsedSeconds,
            }
            for c in collected.calls
        ],
        "maxProbabilities": [
            {"itemId": item, "textId": text, "maxProbability": value}
            for (item, text), value in sorted(collected.maxProbabilities.items())
        ],
    }


def write_calls_file(
    fits_dir: Path,
    *,
    inputs: FitInputs,
    collected: CollectedCalls,
    end_residency: str,
    run_id: str | None = None,
    now: Callable[[], datetime] | None = None,
) -> Path:
    """`fits/calls-<UTC ts>-<runId>.json`, complete or absent; the directory is made if absent."""
    run_id = run_id or uuid.uuid4().hex
    document = calls_document(run_id, inputs, collected, end_residency)
    name = f"calls-{_stamp(now)}-{run_id}.json"
    data = _json_bytes(document, strict=False)
    try:
        return write_files(fits_dir, {name: data})[0]
    except OSError:  # no hard links here, or a failed link: the exclusive create still saves it
        path = fits_dir / name
        with path.open("xb") as handle:  # a foreign file at `path` raises here and is left alone
            try:
                handle.write(data)
            except BaseException:
                path.unlink(missing_ok=True)
                raise
        return path


# --- candidates ---


@dataclass(frozen=True)
class EmissionFile:
    """One file to write: its name under `fits/` and its bytes."""

    name: str
    data: bytes


@dataclass(frozen=True)
class Emission:
    """The built, checked emission: the candidates (family -> file), the manifest and `extra`
    (the report); `files()` is every file in link order: manifest, report, then the candidates."""

    ordinal: int
    candidates: Mapping[str, EmissionFile]
    manifest: EmissionFile
    extra: tuple[EmissionFile, ...] = ()

    def files(self) -> dict[str, bytes]:
        every = (self.manifest, *self.extra, *self.candidates.values())
        return {f.name: f.data for f in every}


def next_ordinal(fits_dir: Path, role: str, dataset_version: str) -> int:
    """The point's next fit ordinal `N`: one more than the largest among its candidate, manifest
    and report file names (module docstring)."""
    stem = f"{re.escape(role)}-{re.escape(dataset_version)}"
    pattern = re.compile(
        rf"{stem}-(?:(?:handoff|only)-f(\d+)\.json|f(\d+)\.fit\.txt|f(\d+)\.fit-report\.json)"
    )
    found = [0]
    if fits_dir.is_dir():
        for path in fits_dir.iterdir():
            if match := pattern.fullmatch(path.name):
                found.append(int(next(g for g in match.groups() if g)))
    return max(found) + 1


def candidate_id(role: str, dataset_version: str, family: str, ordinal: int) -> str:
    return f"{role}-{dataset_version}-{SUFFIX_BY_FAMILY[family]}-f{ordinal}"


def candidate_policy(role: str, ladder_policy: Mapping[str, Any]) -> dict[str, Any]:
    """The candidate's `policy` for a ladder's: `{l, h}` -> band, `{t}` -> the empty band at `t`,
    `{c}` -> `choice-confidence`. A policy with none of them is a ladder bug and refuses."""
    if "c" in ladder_policy:
        return {"kind": "choice-confidence", "minConfidence": ladder_policy["c"]}
    if "l" in ladder_policy and "h" in ladder_policy:
        low, high = ladder_policy["l"], ladder_policy["h"]
    elif "t" in ladder_policy:
        low = high = ladder_policy["t"]
    else:
        raise EmitRefused(f"{role}: a ladder policy with no l/h, t or c: {dict(ladder_policy)!r}")
    return {"kind": "predicate-band", "yesAtOrAbove": high, "noBelow": low}


def candidate_document(
    run: FitRun,
    chosen: SelectedText,
    *,
    ordinal: int,
    manifest_name: str,
    manifest_declaration: Mapping[str, Any],
    source_sha: str,
) -> dict[str, Any]:
    """One candidate's JSON; refuses (`ValueError`/`EmitRefused`) before anything is written."""
    role = run.inputs.role
    family = chosen.selection.candidate
    if family not in SUFFIX_BY_FAMILY:
        raise EmitRefused(f"unknown candidate family {family!r}")
    handoff_model = run.inputs.handoffModel if family == "handoff" else None
    policy = candidate_policy(role, chosen.ladder.policy)
    check_handoff_rule(policy, handoff_model)
    outcome, target = assemble_outcome(chosen.ladder, chosen.selection, BAR_KIND_BY_ROLE[role])
    return {
        "candidateId": candidate_id(role, run.inputs.datasetVersion, family, ordinal),
        "role": role,
        "decisionModel": run.inputs.decisionModel,
        "question": dict(chosen.text.spec),
        "policy": policy,
        "handoffModel": handoff_model,
        "decisionTimeoutSeconds": run.inputs.decisionTimeoutSeconds,
        "fit": {"outcome": outcome, "qualityTarget": target},
        "fitManifest": {"file": manifest_name, **manifest_declaration, "purposes": list(PURPOSES)},
        "provenance": {
            "builtBy": BUILT_BY,
            "datasetVersion": run.inputs.datasetVersion,
            "sourceSha": source_sha,
        },
    }


def build_emission(
    run: FitRun,
    fits_dir: Path,
    *,
    input_hashes: Iterable[str],
    source_sha: str,
    extra: Iterable[EmissionFile] = (),
    report: bool = False,
) -> Emission:
    """Everything the fit writes, built and checked and not yet written: the two candidates, the
    manifest of `input_hashes`, `extra` and, with `report`, the fit report (`report.py`). Raises
    before any file exists."""
    if not source_sha:
        raise EmitRefused("provenance.sourceSha is empty")
    role, version = run.inputs.role, run.inputs.datasetVersion
    if role not in BAR_KIND_BY_ROLE:
        raise EmitRefused(f"unknown role {role!r}")
    ordinal = next_ordinal(fits_dir, role, version)
    manifest_bytes = manifest_module.render_manifest(input_hashes)
    manifest_name = f"{role}-{version}-f{ordinal}.fit.txt"
    declaration = manifest_module.declare_manifest(manifest_bytes)
    by_family = {c.selection.candidate: c for c in run.analysis.selected}
    candidates: dict[str, EmissionFile] = {}
    for family in SUFFIX_BY_FAMILY:
        if family not in by_family:
            raise EmitRefused(f"the analysis has no selection for the {family!r} candidate")
        document = candidate_document(
            run,
            by_family[family],
            ordinal=ordinal,
            manifest_name=manifest_name,
            manifest_declaration=declaration,
            source_sha=source_sha,
        )
        candidates[family] = EmissionFile(f"{document['candidateId']}.json", _json_bytes(document))
    files = list(extra)
    if report:
        document = build_report(
            run,
            ordinal,
            {f: Path(c.name).stem for f, c in candidates.items()},
            BAR_KIND_BY_ROLE[role],
        )
        name = f"{role}-{version}-f{ordinal}.fit-report.json"
        files.append(EmissionFile(name, _json_bytes(document)))
    manifest_file = EmissionFile(manifest_name, manifest_bytes)
    return Emission(ordinal, candidates, manifest_file, tuple(files))


def emit(
    run: FitRun,
    fits_dir: Path,
    *,
    input_hashes: Iterable[str],
    source_sha: str,
    extra: Iterable[EmissionFile] = (),
    report: bool = False,
) -> tuple[Emission, list[Path]]:
    """`build_emission`, then `write_files`: the paths written."""
    emission = build_emission(
        run, fits_dir, input_hashes=input_hashes, source_sha=source_sha, extra=extra, report=report
    )
    return emission, write_files(fits_dir, emission.files())
