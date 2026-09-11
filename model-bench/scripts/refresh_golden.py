#!/usr/bin/env python3
"""scripts/refresh_golden.py — the one-way, human-invoked importer of the `embedder-graphrag-
retrieval` pack's golden data from `falkor-chat` (plan §3.1 point 3, §4 S3; this stage's own spec,
`docs/plans/small-model-benchmarking-s3-spec.md` §5, §8 Step 0).

**Never imported by any run-path module (FR-23) and never invoked automatically.** Reads
`falkor-chat` source files as TEXT/AST only (spec §3.4) — never imports `falkor-chat` code, never
touches a `falkor-chat` venv, never opens a database connection. `model-bench` never reads
`falkor-chat` at run time; this script is a maintenance path, not a code path of any `run`.

Three pairwise-exclusive CLI modes (spec §5's `main()`):

* **default (data-import)** — re-copies/re-transforms every `OriginSpec` except the `check-only`
  one, rewrites `PROVENANCE.md` in full, and refuses to run again with an unchanged `packVersion`
  (a content-hash-changing edit under an unchanged version number is exactly AC-3's own
  violation). Never touches `corpus.embeddings.json` — that is `--embed-corpus`'s job alone.
* **`--check-origins`** — read-only, no writes: re-hashes every tracked origin (all five, including
  `test_metrics.py`) and prints `unchanged`/`DRIFTED` per file; exit 0 iff none drifted.
* **`--embed-corpus`** — the one live-LM-Studio mode (spec §8 Step 2). **Not built in this stage**:
  `embed_corpus()` below raises `NotImplementedError` by design — it needs a reachable LM Studio
  and a real `ModelInfo`, neither of which Step 0 (static pack data, no live call) touches.

The pure cache-key computation (`compute_cache_key`, `-ml` §5.5's four components) *is* built and
tested here, offline, against a synthetic corpus/model fixture — it is what `--embed-corpus`
(Step 2) will use to write `corpus.embeddings.json`'s header, and what `scoring/retrieval.py`'s
`prime()` (Step 1) will use to decide cache hit/miss, but computing it needs no live call.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

# scripts/refresh_golden.py -> model-bench/ -> graphmind-ai-lab/ (the monorepo root, where every
# `OriginSpec.originPath` below is rooted — confirmed against root AGENTS.md's own structure,
# `falkor-chat/` as a top-level sibling of `model-bench/`).
_DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2]


class RefreshGoldenError(RuntimeError):
    """A maintenance-script-level failure: an origin file's own shape changed (no `_CORPUS`
    assignment found, a malformed manifest), an unbumped `packVersion`, or an invalid CLI mode
    combination. Raised, never silently worked around."""


# --------------------------------------------------------------------------------------------
# Tracked origins (spec §5)
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class OriginSpec:
    originPath: str  # repo-root-relative
    destPath: str  # pack-root-relative (or, for "check-only", relative to the pack dir)
    kind: Literal["copy", "jsonl-transform", "ast-literal", "check-only"]


_TRACKED_ORIGINS: tuple[OriginSpec, ...] = (
    OriginSpec(
        "falkor-chat/server/tests/eval/golden_retrieval.jsonl",
        "queries.jsonl",
        "jsonl-transform",
    ),
    OriginSpec("falkor-chat/scripts/seed_eval_corpus.py", "corpus.jsonl", "ast-literal"),
    OriginSpec(
        "falkor-chat/server/tests/eval/retrieval_baseline.json",
        "retrieval_baseline.json",
        "copy",
    ),
    OriginSpec(
        "falkor-chat/server/tests/eval/golden_retrieval.embeddings.json",
        "golden_retrieval.embeddings.json",
        "copy",
    ),
    # Tracked for --check-origins drift detection only — this script never writes this file's
    # CONTENT (the 20 cases are hand-transcribed, not extracted, spec §3.1 point 2). It reads the
    # fixture's own recorded `sourceSha256` header and reports drift against a fresh hash of
    # `test_metrics.py`.
    OriginSpec(
        "falkor-chat/server/tests/eval/test_metrics.py",
        "../../tests/fixtures/metrics_agreement.json",
        "check-only",
    ),
)


# --------------------------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ProvenanceRecord:
    originPath: str
    destPath: str
    sourceGitSha: str  # the last commit that touched `originPath`, not the repo's current HEAD
    sourceSha256: str  # sha256 of the ORIGIN file's bytes at copy time
    copiedAt: str  # UTC ISO-8601


_PROVENANCE_VERSION_RE = re.compile(r"\*\*Pack version:\*\*\s*(\S+)")
_PROVENANCE_ROW_RE = re.compile(
    r"^\|\s*`([^`]+)`\s*\|\s*`([^`]+)`\s*\|\s*`([^`]+)`\s*\|\s*`([^`]+)`\s*\|\s*([^|]+?)\s*\|\s*$"
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_file_sha(repo_root: Path, rel_path: str) -> str:
    """The last commit that touched `rel_path`, via `git log -1` in `repo_root` — never the
    repo's own current HEAD, which says nothing about *this file's* own history. Confirmed
    against the plan's own cited SHA for `test_metrics.py` (§3.1 point 2(a)): `git log -1 --
    <path>` for that one file reproduces `9650a3858b9d5c4e7e934f977839fc1a61c84b1b` exactly, while
    `git rev-parse HEAD` does not."""
    result = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", rel_path],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    sha = result.stdout.strip()
    if not sha:
        raise RefreshGoldenError(f"no git history found for {rel_path!r} under {repo_root}")
    return sha


def _write_provenance(
    pack_root: Path, pack_version: str, records: Sequence[ProvenanceRecord]
) -> None:
    """Rewrites `PROVENANCE.md` IN FULL (never appended to) — this file's job is "what does this
    pack's data trace back to right now", a living, wholly-regenerated document, not a history
    log."""
    lines = [
        f"# Provenance — {pack_root.name}",
        "",
        f"> **Pack version:** {pack_version} · **Generated:** {_utc_now_iso()}",
        "",
        "Data files below are copied one-way from `falkor-chat/` by "
        "`model-bench/scripts/refresh_golden.py` (plan §3.1 point 3, D1: \"copy the data, "
        "clean-build the code\"). Re-running the importer requires bumping `pack.json`'s "
        "`packVersion` by hand first — a content-hash-changing edit under an unchanged version "
        "number is refused.",
        "",
        "| Origin | Destination | Source git SHA | Source SHA-256 | Copied at |",
        "|---|---|---|---|---|",
    ]
    for record in records:
        lines.append(
            f"| `{record.originPath}` | `{record.destPath}` | `{record.sourceGitSha}` | "
            f"`{record.sourceSha256}` | {record.copiedAt} |"
        )
    lines.append("")
    (pack_root / "PROVENANCE.md").write_text("\n".join(lines), encoding="utf-8")


def _read_provenance_pack_version(provenance_path: Path) -> str | None:
    match = _PROVENANCE_VERSION_RE.search(provenance_path.read_text(encoding="utf-8"))
    return match.group(1) if match else None


def _read_provenance_records(provenance_path: Path) -> dict[str, str]:
    """`originPath` -> recorded `sourceSha256`, parsed back out of `PROVENANCE.md`'s own table —
    the read half of `_write_provenance`'s write, used by `--check-origins`."""
    out: dict[str, str] = {}
    for line in provenance_path.read_text(encoding="utf-8").splitlines():
        match = _PROVENANCE_ROW_RE.match(line)
        if match:
            origin_path, _dest_path, _git_sha, sha256, _copied_at = match.groups()
            out[origin_path] = sha256
    return out


def _pack_version_gate(pack_root: Path, provenance_path: Path) -> None:
    """Refuses a re-run when `PROVENANCE.md` already exists and its own recorded `packVersion`
    equals `pack.json`'s current one — bumping `packVersion` by hand first is the caller's
    obligation (spec §5's `main()` docstring). A first run, with no `PROVENANCE.md` yet, always
    proceeds: there is nothing to have forgotten to bump."""
    if not provenance_path.exists():
        return
    manifest = json.loads((pack_root / "pack.json").read_text(encoding="utf-8"))
    current_version = manifest["packVersion"]
    recorded_version = _read_provenance_pack_version(provenance_path)
    if recorded_version == current_version:
        raise RefreshGoldenError(
            f"pack.json's packVersion ({current_version!r}) is unchanged since the last refresh "
            f"recorded in {provenance_path}; bump packVersion by hand before re-running this "
            "importer (a content-hash-changing edit under an unchanged version number is AC-3's "
            "own violation)"
        )


# --------------------------------------------------------------------------------------------
# `ast-literal` — reading `_CORPUS` out of `seed_eval_corpus.py`'s TEXT, never by importing it
# --------------------------------------------------------------------------------------------


def _read_corpus_literal(seed_script_text: str) -> list[dict[str, object]]:
    """AST-parses `seed_eval_corpus.py`'s text and literal-evals the one `_CORPUS = [...]` node
    (spec §3.4) — never `exec`/`import`. `_CORPUS` carries a type annotation
    (`_CORPUS: list[dict[str, object]] = [...]`), so its assignment is an `ast.AnnAssign` node,
    not a plain `ast.Assign` — confirmed by parsing the real file this session; a walk that only
    matches `ast.Assign` silently finds nothing on this file, so both forms are handled. Raises
    `RefreshGoldenError` if no such assignment exists (the origin's own shape changed) rather than
    silently returning `[]`."""
    tree = ast.parse(seed_script_text)
    for node in ast.walk(tree):
        target_name: str | None = None
        value_node: ast.expr | None = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                target_name, value_node = target.id, node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target_name, value_node = node.target.id, node.value
        if target_name == "_CORPUS" and value_node is not None:
            literal = ast.literal_eval(value_node)
            if not isinstance(literal, list):
                raise RefreshGoldenError(
                    f"_CORPUS did not literal-eval to a list (got {type(literal).__name__})"
                )
            return literal
    raise RefreshGoldenError(
        "no `_CORPUS = [...]` (or annotated) assignment found in seed_eval_corpus.py's text — "
        "the origin's own shape changed"
    )


def _corpus_rows_from_literal(corpus: Sequence[Mapping[str, object]]) -> list[dict[str, str]]:
    """One row per message: `{"docId": f"eval-{slug}-{n:03d}", "text": msg_text, "topic": slug}` —
    reproduces `seed_eval_corpus.py`'s own `msg_id = f"eval-{slug}-{n:03d}"` convention (confirmed
    at `seed_eval_corpus.py:629`: `n` is 1-indexed per topic, in each topic's own message order)
    so `docId` values match `golden_retrieval.jsonl`'s `relevant_msgIds` verbatim."""
    rows: list[dict[str, str]] = []
    for topic in corpus:
        slug = str(topic["slug"])
        messages = topic["messages"]
        if not isinstance(messages, list):
            raise RefreshGoldenError(f"topic {slug!r} has no messages list")
        for n, message in enumerate(messages, start=1):
            _role, text = message
            rows.append({"docId": f"eval-{slug}-{n:03d}", "text": str(text), "topic": slug})
    return rows


# --------------------------------------------------------------------------------------------
# `jsonl-transform` — `golden_retrieval.jsonl` -> `queries.jsonl` (this pack's own field names)
# --------------------------------------------------------------------------------------------


def _queries_rows_from_golden_retrieval(lines: Sequence[str]) -> list[dict[str, Any]]:
    """One row per `golden_retrieval.jsonl` line, renamed to this pack's own naming convention
    (spec §4.1): `id` -> `itemId` (matches `ANALYSIS_UNIT_FIELD_BY_ROLE["embedder"]`),
    `relevant_msgIds` -> `relevantDocIds` (`docId` is `corpus.jsonl`'s own field name),
    `target_text` -> `targetText`; `query`/`topic`/`rationale` carried through unchanged."""
    rows: list[dict[str, Any]] = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        row = json.loads(line)
        rows.append(
            {
                "itemId": row["id"],
                "query": row["query"],
                "relevantDocIds": row["relevant_msgIds"],
                "topic": row["topic"],
                "targetText": row["target_text"],
                "rationale": row["rationale"],
            }
        )
    return rows


def _write_jsonl(dest_path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with dest_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


# --------------------------------------------------------------------------------------------
# `--check-origins` — read-only drift detection
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class OriginCheckResult:
    originPath: str
    status: Literal["unchanged", "DRIFTED"]
    recordedSha256: str | None
    currentSha256: str


def _fixture_source_sha256(fixture_path: Path) -> str | None:
    """`test_metrics.py`'s own recorded `sourceSha256`, read from `metrics_agreement.json`'s own
    header (spec §3.1 point 2(c)) — the `check-only` origin's provenance lives on the fixture
    itself, not in `PROVENANCE.md`, since this script never writes that fixture's content."""
    if not fixture_path.exists():
        return None
    data = json.loads(fixture_path.read_text(encoding="utf-8"))
    return data.get("sourceSha256")


def _check_origins(repo_root: Path, pack_root: Path) -> list[OriginCheckResult]:
    """Re-hashes every `_TRACKED_ORIGINS` entry's current `originPath` bytes and compares against
    the matching recorded hash: `PROVENANCE.md`'s table for the four copied/transformed files,
    `metrics_agreement.json`'s own header for `test_metrics.py` (spec §3.1 point 2(c)). Read-only;
    never mutates either side."""
    provenance_path = pack_root / "PROVENANCE.md"
    recorded_provenance = (
        _read_provenance_records(provenance_path) if provenance_path.exists() else {}
    )

    results: list[OriginCheckResult] = []
    for origin in _TRACKED_ORIGINS:
        origin_path = repo_root / origin.originPath
        current_sha = _sha256_bytes(origin_path.read_bytes())

        if origin.kind == "check-only":
            fixture_path = (pack_root / origin.destPath).resolve()
            recorded_sha = _fixture_source_sha256(fixture_path)
        else:
            recorded_sha = recorded_provenance.get(origin.originPath)

        status: Literal["unchanged", "DRIFTED"] = (
            "unchanged" if recorded_sha == current_sha else "DRIFTED"
        )
        results.append(
            OriginCheckResult(
                originPath=origin.originPath,
                status=status,
                recordedSha256=recorded_sha,
                currentSha256=current_sha,
            )
        )
    return results


# --------------------------------------------------------------------------------------------
# The embedding cache key (`-ml` §5.5) — pure, offline, no LM Studio call
# --------------------------------------------------------------------------------------------


def compute_cache_key(
    *, model: str, quantization: str, document_prefix: str, corpus_bytes: bytes
) -> dict[str, str]:
    """`-ml` §5.5's cache key, all four components, as a plain dict: `(model id, quantization,
    docPrefix, corpus version)`, with `corpus version` realized as `corpusSha256` — a hash of the
    corpus data's own bytes rather than a hand-maintained version counter, the same "a hash
    cannot be forgotten to bump" discipline `packs.content_hash` already uses for the whole pack.

    Pure: no LM Studio call, no filesystem access beyond what the caller already read. Reused by
    `embed_corpus` (Step 2, below) to write `corpus.embeddings.json`'s own header, and by
    `scoring/retrieval.py`'s `prime()` (Step 1) to compare a live run's key against that header.
    Getting this cache key wrong produces a plausible, invalid comparison with no visible trace
    (`-ml` §5.5) — so it is asserted in a unit test against a synthetic fixture (this stage's own
    `tests/test_refresh_golden.py`), not just implemented."""
    return {
        "model": model,
        "quantization": quantization,
        "documentPrefix": document_prefix,
        "corpusSha256": _sha256_bytes(corpus_bytes),
    }


@dataclass(frozen=True)
class CorpusEmbeddingsFile:
    """`corpus.embeddings.json`'s in-memory shape (spec §5). Vectors are stored RAW — never
    pre-normalized at write time (spec §6.2/§9): the ranking function L2-normalizes at read time,
    and the whole point of this file existing is to let the offline self-test observe the raw norm
    distribution too, so normalizing before writing would throw away the one thing `-ml` §5.2 asks
    to be measured. `"normalized": false` is written into the file itself (spec §9's own risk
    note) so a future reader cannot assume pre-normalization by omission."""

    cacheKey: Mapping[str, str]
    generatedAt: str
    vectors: Mapping[str, Sequence[float]]  # docId -> raw vector

    def to_dict(self) -> dict[str, Any]:
        return {
            "cacheKey": dict(self.cacheKey),
            "generatedAt": self.generatedAt,
            "normalized": False,
            "vectors": {doc_id: list(vector) for doc_id, vector in self.vectors.items()},
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "CorpusEmbeddingsFile":
        return cls(
            cacheKey=dict(d["cacheKey"]),
            generatedAt=d["generatedAt"],
            vectors={doc_id: list(vector) for doc_id, vector in d["vectors"].items()},
        )


def embed_corpus(
    corpus_rows: Sequence[Mapping[str, str]],
    *,
    lmstudio: Any,
    model_key: str,
    document_prefix: str,
    timeout_s: float,
) -> CorpusEmbeddingsFile:
    """Step 2's own live mode (`docs/plans/small-model-benchmarking-s3-spec.md` §8 Step 2, §5):
    one batched `lmstudio.embed()` call over all 121 (prefixed) documents, writing RAW vectors
    plus a `compute_cache_key` header.

    **Not built in this stage.** Step 0 (this file's own scope, spec §8) is "static pack data, no
    live call" — `embed_corpus` needs a reachable LM Studio and a real `ModelInfo.quantization`,
    neither of which any Step 0 test touches. Raises unconditionally so `--embed-corpus` fails
    loudly rather than silently no-opping."""
    raise NotImplementedError(
        "embed_corpus: Step 2's own live mode "
        "(docs/plans/small-model-benchmarking-s3-spec.md §8 Step 2) — not built in Step 0"
    )


# --------------------------------------------------------------------------------------------
# The default (data-import) mode
# --------------------------------------------------------------------------------------------


def _run_import(repo_root: Path, pack_root: Path) -> None:
    provenance_path = pack_root / "PROVENANCE.md"
    _pack_version_gate(pack_root, provenance_path)

    now = _utc_now_iso()
    records: list[ProvenanceRecord] = []
    for origin in _TRACKED_ORIGINS:
        if origin.kind == "check-only":
            continue

        origin_path = repo_root / origin.originPath
        origin_bytes = origin_path.read_bytes()
        dest_path = pack_root / origin.destPath

        if origin.kind == "copy":
            dest_path.write_bytes(origin_bytes)
        elif origin.kind == "jsonl-transform":
            lines = origin_bytes.decode("utf-8").splitlines()
            rows = _queries_rows_from_golden_retrieval(lines)
            _write_jsonl(dest_path, rows)
        elif origin.kind == "ast-literal":
            corpus = _read_corpus_literal(origin_bytes.decode("utf-8"))
            rows = _corpus_rows_from_literal(corpus)
            _write_jsonl(dest_path, rows)
        else:  # pragma: no cover - exhaustive over OriginSpec.kind's Literal
            raise AssertionError(f"unhandled OriginSpec kind {origin.kind!r}")

        records.append(
            ProvenanceRecord(
                originPath=origin.originPath,
                destPath=origin.destPath,
                sourceGitSha=_git_file_sha(repo_root, origin.originPath),
                sourceSha256=_sha256_bytes(origin_bytes),
                copiedAt=now,
            )
        )

    manifest = json.loads((pack_root / "pack.json").read_text(encoding="utf-8"))
    _write_provenance(pack_root, manifest["packVersion"], records)


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="refresh_golden.py",
        description=(
            "One-way importer of the embedder-graphrag-retrieval pack's golden data from "
            "falkor-chat (human-invoked, never on a run path)."
        ),
    )
    parser.add_argument(
        "--repo-root",
        default=str(_DEFAULT_REPO_ROOT),
        help="the monorepo root (default: this script's own resolved ancestor)",
    )
    parser.add_argument("--pack", required=True, help="path to the pack directory")
    parser.add_argument(
        "--check-origins", action="store_true", help="read-only drift check; no writes"
    )
    parser.add_argument(
        "--embed-corpus",
        action="store_true",
        help="Step 2's live mode (not built in this stage)",
    )
    parser.add_argument("--model", help="LM Studio model key (--embed-corpus only)")
    parser.add_argument(
        "--api-base-url", default="http://localhost:1234", help="--embed-corpus only"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.check_origins and args.embed_corpus:
        print(
            "refresh_golden.py: --check-origins and --embed-corpus are exclusive — two "
            "unrelated operations, one flag each, never combined in one invocation",
            file=sys.stderr,
        )
        return 2

    repo_root = Path(args.repo_root).resolve()
    pack_root = Path(args.pack).resolve()

    if args.check_origins:
        results = _check_origins(repo_root, pack_root)
        for result in results:
            print(f"{result.status}: {result.originPath}")
        return 0 if all(r.status == "unchanged" for r in results) else 1

    if args.embed_corpus:
        embed_corpus(
            [],
            lmstudio=None,
            model_key=args.model or "",
            document_prefix="",
            timeout_s=120.0,
        )
        return 0  # unreachable — embed_corpus always raises in this stage

    try:
        _run_import(repo_root, pack_root)
    except RefreshGoldenError as exc:
        print(f"refresh_golden.py: {exc}", file=sys.stderr)
        return 1
    print(f"refreshed: {pack_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
