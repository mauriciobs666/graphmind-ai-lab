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
* **`--embed-corpus`** — the one live-LM-Studio mode (spec §8 Step 2, this stage's own build): one
  batched `lmstudio.embed()` call over the pack's 121 documents, writing `corpus.embeddings.json`
  with its cache-key header. Refuses under an unchanged `packVersion`, exactly like the default
  mode (`_pack_version_gate`, shared verbatim between both write paths — spec §8 Step 2 item 1).

The pure cache-key computation (`compute_cache_key`, `-ml` §5.5's four components) is built and
tested here, offline, against a synthetic corpus/model fixture — `--embed-corpus` uses it to write
`corpus.embeddings.json`'s header, and `scoring/retrieval.py`'s `prime()` independently recomputes
the same shape (its own `_compute_cache_key`, deliberately not imported from here) to decide cache
hit/miss.
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

from modelbench.lmstudio import LMStudio, LMStudioError, ModelInfo
from modelbench.packs import load_pack

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
    destPath: str  # pack-root-relative (or, for "check-only", relative to the pack dir); a
    # "#"-suffixed fragment (`"tables.json#catalog"`) means "JSON-merge this key into that file",
    # never a literal filename (S4 spec §6)
    kind: Literal["copy", "jsonl-transform", "ast-literal", "schema-literal", "check-only"]


#: Per-pack tracked origins, keyed by `packId` (S4 spec §6) — generalized from a single
#: module-level constant because the shipped one-pack shape was embedder-only despite `--pack`
#: already being a required, generic-looking flag: running the old constant against a
#: non-embedder `--pack` path would silently write the wrong origins to the wrong destination.
#: `_origins_for_pack_id`/`_read_pack_id` below are the two resolvers `main()` uses; `_run_import`
#: and `_check_origins` take an already-resolved `origins` tuple rather than reading this mapping
#: themselves, so they stay testable against a synthetic origin list with no `pack.json` at all
#: (the same shape their own tests used before this generalization).
_TRACKED_ORIGINS_BY_PACK_ID: dict[str, tuple[OriginSpec, ...]] = {
    "embedder-graphrag-retrieval": (
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
        # Tracked for --check-origins drift detection only — this script never writes this
        # file's CONTENT (the 20 cases are hand-transcribed, not extracted, spec §3.1 point 2).
        # It reads the fixture's own recorded `sourceSha256` header and reports drift against a
        # fresh hash of `test_metrics.py`.
        OriginSpec(
            "falkor-chat/server/tests/eval/test_metrics.py",
            "../../tests/fixtures/metrics_agreement.json",
            "check-only",
        ),
    ),
    "guard-judge-understanding": (
        # `id` -> `itemId` rename only; every other field carried through unchanged (S4 spec
        # §5.1.2/§6) — see `_items_rows_from_golden_guards`.
        OriginSpec(
            "falkor-chat/server/tests/eval/golden_guards.jsonl", "items.jsonl", "jsonl-transform",
        ),
    ),
    "nlq-structured-query": (
        # `id` -> `itemId` rename; `answerable` is NOT written here — `--stamp-answerability`
        # does that, once `reference_specs.json` exists (S4 spec §6).
        OriginSpec(
            "falkor-chat/server/tests/eval/nlq_golden_set.jsonl", "items.jsonl", "jsonl-transform",
        ),
        # AST-parses the CATALOG heredoc (same technique as seed_eval_corpus.py's _CORPUS, S3
        # spec §3.4), computes nameNormalized/categoryNormalized via the transcribed
        # normalize_name (§2.6), and merges into tables.json's "catalog" key — never touches
        # "knowledge_base" (that half is `--check-tables-shape`'s, from the human-in-the-loop
        # live snapshot).
        OriginSpec("falkor-chat/scripts/seed_catalog.sh", "tables.json#catalog", "ast-literal"),
        # AST-extracts CATALOG_SCHEMA/KNOWLEDGE_BASE_SCHEMA's `labels` dict literals the same
        # execute-nothing way `_read_corpus_literal` does, mapping each Python type name
        # (str/int/float) to its JSON token.
        OriginSpec(
            "falkor-chat/server/falkorchat/querygen.py", "schema.json", "schema-literal",
        ),
    ),
}

def _origins_for_pack_id(pack_id: str) -> tuple[OriginSpec, ...]:
    """Resolve `pack_id` against `_TRACKED_ORIGINS_BY_PACK_ID` (S4 spec §6). Raises
    `RefreshGoldenError`, naming every known pack, on an unregistered one — a silent `{}.get(...,
    ())` would re-open exactly the "wrong origins for the wrong pack" latent bug this
    generalization exists to close."""
    try:
        return _TRACKED_ORIGINS_BY_PACK_ID[pack_id]
    except KeyError:
        known = ", ".join(sorted(_TRACKED_ORIGINS_BY_PACK_ID))
        raise RefreshGoldenError(
            f"no tracked origins registered for pack {pack_id!r}; known packs: {known}"
        ) from None


def _read_pack_id(pack_root: Path) -> str:
    """The target `--pack`'s own declared `packId`, read from its `pack.json` — the key
    `_origins_for_pack_id` resolves against (S4 spec §6)."""
    manifest = json.loads((pack_root / "pack.json").read_text(encoding="utf-8"))
    return manifest["packId"]


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


def _read_provenance_records_full(provenance_path: Path) -> list[ProvenanceRecord]:
    """Every row of `PROVENANCE.md`'s own table, parsed back into `ProvenanceRecord`s (the read
    half of `_write_provenance`'s write) — used by `run_check_tables_shape` to ADD its one new
    row without clobbering the rows a prior default-import run already wrote (S4 spec §6)."""
    records: list[ProvenanceRecord] = []
    for line in provenance_path.read_text(encoding="utf-8").splitlines():
        match = _PROVENANCE_ROW_RE.match(line)
        if match:
            origin_path, dest_path, git_sha, sha256, copied_at = match.groups()
            records.append(ProvenanceRecord(origin_path, dest_path, git_sha, sha256, copied_at))
    return records


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


def _items_rows_from_golden_guards(lines: Sequence[str]) -> list[dict[str, Any]]:
    """One row per `golden_guards.jsonl` line — `id` -> `itemId` rename ONLY (matches
    `ANALYSIS_UNIT_FIELD_BY_ROLE["guard-judge"]`); every other field carried through unchanged
    (S4 spec §5.1.2/§6, D1: "data, not derived")."""
    rows: list[dict[str, Any]] = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        row = json.loads(line)
        rows.append({("itemId" if key == "id" else key): value for key, value in row.items()})
    return rows


def _items_rows_from_nlq_golden_set(lines: Sequence[str]) -> list[dict[str, Any]]:
    """One row per `nlq_golden_set.jsonl` line — `id` -> `itemId` rename ONLY (matches
    `ANALYSIS_UNIT_FIELD_BY_ROLE["nlq-generator"]`); every other field (`dataset`/`question`/
    `shape`/`expected`/`rationale`) carried through unchanged (S4 spec §6, D1: "data, not
    derived"). `answerable` is deliberately absent here — `--stamp-answerability` writes it,
    separately, once `reference_specs.json` exists."""
    rows: list[dict[str, Any]] = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        row = json.loads(line)
        rows.append({("itemId" if key == "id" else key): value for key, value in row.items()})
    return rows


#: Which pure row-transform function a `"jsonl-transform"` origin runs, keyed by the OWNING
#: pack's `packId` — each pack in `_TRACKED_ORIGINS_BY_PACK_ID` declares at most one
#: `"jsonl-transform"` origin today, so one function per pack is unambiguous.
_JSONL_TRANSFORM_BY_PACK_ID: Mapping[str, Any] = {
    "embedder-graphrag-retrieval": _queries_rows_from_golden_retrieval,
    "guard-judge-understanding": _items_rows_from_golden_guards,
    "nlq-structured-query": _items_rows_from_nlq_golden_set,
}


def _write_jsonl(dest_path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with dest_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def _split_dest_fragment(dest_path_str: str) -> tuple[str, str | None]:
    """`"tables.json#catalog"` -> `("tables.json", "catalog")`; a plain path -> `(path, None)`
    (S4 spec §6 — the `#`-fragment convention for a JSON-merge write)."""
    if "#" in dest_path_str:
        file_part, key = dest_path_str.split("#", 1)
        return file_part, key
    return dest_path_str, None


def _write_json_merge_key(dest_path: Path, key: str, value: Any) -> None:
    """Merges `value` into `dest_path`'s JSON object under `key`, preserving every other
    top-level key already there (S4 spec §6: the catalog half of `tables.json` must never
    clobber a `"knowledge_base"` half written separately by `--check-tables-shape`)."""
    data: dict[str, Any] = {}
    if dest_path.exists():
        data = json.loads(dest_path.read_text(encoding="utf-8"))
    data[key] = value
    dest_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------------------------
# `ast-literal` (nlq-structured-query's catalog) — reading `CATALOG` out of `seed_catalog.sh`'s
# embedded Python heredoc TEXT, never by importing or running it
# --------------------------------------------------------------------------------------------


def _extract_python_heredoc(text: str, marker: str = "PY") -> str:
    """Isolates the Python body of a `"$VENV_PY" - <<'PY' ... PY` bash heredoc — `ast.parse`
    cannot run on the surrounding bash syntax directly (S4 spec §6, extending S3's
    `_read_corpus_literal` technique from a plain `.py` file to a shell-embedded one)."""
    lines = text.splitlines()
    start: int | None = None
    end: int | None = None
    for i, line in enumerate(lines):
        if start is None and line.rstrip().endswith(f"<<'{marker}'"):
            start = i + 1
            continue
        if start is not None and line.strip() == marker:
            end = i
            break
    if start is None or end is None:
        raise RefreshGoldenError(f"no <<'{marker}' ... {marker} heredoc found in the origin text")
    return "\n".join(lines[start:end])


def _read_catalog_literal(seed_script_text: str) -> list[tuple[str, str, float]]:
    """AST-parses `seed_catalog.sh`'s embedded heredoc for the one `CATALOG = [...]` assignment
    (a plain `ast.Assign`, unlike `seed_eval_corpus.py`'s annotated `_CORPUS`) — never
    `exec`/`import`. Raises `RefreshGoldenError` if no such assignment exists."""
    heredoc = _extract_python_heredoc(seed_script_text)
    tree = ast.parse(heredoc)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id == "CATALOG":
                literal = ast.literal_eval(node.value)
                if not isinstance(literal, list):
                    raise RefreshGoldenError(
                        f"CATALOG did not literal-eval to a list (got {type(literal).__name__})"
                    )
                return literal
    raise RefreshGoldenError(
        "no `CATALOG = [...]` assignment found in seed_catalog.sh's heredoc text — the origin's "
        "own shape changed"
    )


def _normalize_name_for_refresh(value: str) -> str:
    """Two-line transcription of `extraction.normalize_name` (`extraction.py:67-78`) — whitespace-
    collapse + casefold. No import (D1/FR-23): this maintenance script reaches outside stdlib for
    nothing, and this is the same transcription `tools/exec.py`'s own `_normalize_name` carries."""
    return re.sub(r"\s+", " ", value.strip()).casefold()


def _catalog_rows_from_literal(
    catalog: Sequence[tuple[str, str, float]]
) -> dict[str, list[dict[str, Any]]]:
    """`{"Product": [...]}` — one row per `(name, category, price)` triple, computing
    `nameNormalized`/`categoryNormalized` (S4 spec §2.6/§6), matching `seed_catalog.sh`'s own
    live-write shape (which also stores a `productId` slug this pack's `schema.json` never
    exposes — a curated query-facing allowlist, not every property that exists on the node, same
    discipline `querygen.DatasetSchema`'s own docstring states). Keyed by label (`"Product"`,
    catalog's only one) rather than returned as a bare row list — `tables.json["catalog"]` must
    mirror `schema.json["catalog"]["labels"]`'s own per-label shape, the same `tables[label]`
    indexing `tools/exec.py`'s `compile_and_execute` does at run time."""
    rows = [
        {
            "name": name,
            "nameNormalized": _normalize_name_for_refresh(name),
            "category": category,
            "categoryNormalized": _normalize_name_for_refresh(category),
            "price": price,
        }
        for name, category, price in catalog
    ]
    return {"Product": rows}


# --------------------------------------------------------------------------------------------
# `schema-literal` (nlq-structured-query's schema.json) — AST-extracting
# CATALOG_SCHEMA/KNOWLEDGE_BASE_SCHEMA's `labels` dicts out of querygen.py's TEXT
# --------------------------------------------------------------------------------------------

#: Source variable name -> this pack's own dataset key (schema.json's two top-level keys).
_TARGET_SCHEMA_VARS: Mapping[str, str] = {
    "CATALOG_SCHEMA": "catalog",
    "KNOWLEDGE_BASE_SCHEMA": "knowledge_base",
}

#: The only property-type tokens `tools/exec.py`'s Layer B coercion switches on (§5.2.3) — a
#: schema literal naming any other bare type (e.g. `bool`, not used anywhere today) is the
#: origin's own shape changing in a way this script does not know how to carry forward silently.
_KNOWN_TYPE_TOKENS = frozenset({"str", "int", "float"})


def _labels_dict_from_dataset_schema_call(call: ast.Call) -> dict[str, dict[str, str]]:
    labels_node: ast.expr | None = None
    for kw in call.keywords:
        if kw.arg == "labels":
            labels_node = kw.value
            break
    if labels_node is None or not isinstance(labels_node, ast.Dict):
        raise RefreshGoldenError("DatasetSchema(...) call has no `labels={...}` keyword argument")

    labels: dict[str, dict[str, str]] = {}
    for key_node, value_node in zip(labels_node.keys, labels_node.values):
        if key_node is None:
            raise RefreshGoldenError("labels={...} contains a `**`-unpacked entry, not a literal")
        label = ast.literal_eval(key_node)
        if not isinstance(value_node, ast.Dict):
            raise RefreshGoldenError(f"label {label!r}'s properties are not a dict literal")
        props: dict[str, str] = {}
        for prop_key_node, prop_value_node in zip(value_node.keys, value_node.values):
            if prop_key_node is None:
                raise RefreshGoldenError(f"label {label!r} contains a `**`-unpacked entry")
            prop = ast.literal_eval(prop_key_node)
            is_known_type = (
                isinstance(prop_value_node, ast.Name) and prop_value_node.id in _KNOWN_TYPE_TOKENS
            )
            if not is_known_type:
                raise RefreshGoldenError(
                    f"property {prop!r} on label {label!r} does not carry a known bare type name "
                    f"(str/int/float) — the origin's own shape changed"
                )
            props[prop] = prop_value_node.id
        labels[label] = props
    return labels


def _read_schema_literal(querygen_source_text: str) -> dict[str, dict[str, Any]]:
    """AST-parses `querygen.py`'s text and extracts `CATALOG_SCHEMA`/`KNOWLEDGE_BASE_SCHEMA`'s
    `labels={...}` keyword argument into `schema.json`'s own shape (§5.2.3) — never
    `exec`/`import`. Raises `RefreshGoldenError` naming whichever assignment is missing."""
    tree = ast.parse(querygen_source_text)
    found: dict[str, dict[str, Any]] = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in _TARGET_SCHEMA_VARS
            and isinstance(node.value, ast.Call)
        ):
            dataset_key = _TARGET_SCHEMA_VARS[node.targets[0].id]
            found[dataset_key] = {"labels": _labels_dict_from_dataset_schema_call(node.value)}

    missing = set(_TARGET_SCHEMA_VARS.values()) - set(found)
    if missing:
        raise RefreshGoldenError(
            f"could not find assignment(s) for {sorted(missing)!r} in querygen.py's text — the "
            "origin's own shape changed"
        )
    return found


#: Which pure `origin text -> rows` function an `"ast-literal"` origin runs, keyed by the OWNING
#: pack's `packId` — mirrors `_JSONL_TRANSFORM_BY_PACK_ID`'s own per-pack dispatch shape. Each
#: entry composes that pack's own `_read_*_literal` (the AST read) with its own `_*_rows_from_
#: literal` (the row-shape transform), since the two packs read different origin shapes
#: (an annotated `_CORPUS` list-of-dicts vs. a plain `CATALOG` list-of-tuples).
_AST_LITERAL_ROWS_BY_PACK_ID: Mapping[str, Any] = {
    "embedder-graphrag-retrieval": (
        lambda text: _corpus_rows_from_literal(_read_corpus_literal(text))
    ),
    "nlq-structured-query": lambda text: _catalog_rows_from_literal(_read_catalog_literal(text)),
}


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


def _check_origins(
    repo_root: Path, pack_root: Path, origins: Sequence[OriginSpec]
) -> list[OriginCheckResult]:
    """Re-hashes every entry of `origins`' current `originPath` bytes and compares against the
    matching recorded hash: `PROVENANCE.md`'s table for the copied/transformed files,
    `metrics_agreement.json`'s own header for a `check-only` origin (spec §3.1 point 2(c)).
    Read-only; never mutates either side. `origins` is the caller's already-resolved tuple (`main`
    resolves it from the target pack's own `packId` via `_origins_for_pack_id`, S4 spec §6) —
    kept as an explicit parameter rather than read from the module mapping directly so this stays
    testable against a synthetic origin list with no `pack.json` on disk at all."""
    provenance_path = pack_root / "PROVENANCE.md"
    recorded_provenance = (
        _read_provenance_records(provenance_path) if provenance_path.exists() else {}
    )

    results: list[OriginCheckResult] = []
    for origin in origins:
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


def _find_model_in_catalog(catalog: Sequence[ModelInfo], model_key: str) -> ModelInfo:
    """The catalog-lookup-by-model-key logic `--embed-corpus` needs, mirroring
    `runner._find_model`'s convention rather than inventing a new one (spec's own instruction: "LM
    Studio has its own load/warm-up path the runner already uses elsewhere"): resolves `model_key`
    against a live
    `GET /api/v0/models` catalog. Never checks `state` — this is a lookup, not a residency check;
    the caller loads the model via `LMStudio.warm_up()` afterwards, the same way `run_pack`'s own
    step 4 does. Raises `RefreshGoldenError`, naming every id the catalog does have, if `model_key`
    resolves to none of them — a live, actionable failure rather than a guess at why the batched
    embed call would otherwise fail."""
    for info in catalog:
        if info.id == model_key:
            return info
    available = ", ".join(sorted(m.id for m in catalog)) or "(catalog is empty)"
    raise RefreshGoldenError(
        f"--embed-corpus: model {model_key!r} not found in LM Studio's catalog "
        f"(GET /api/v0/models); available: {available}"
    )


def embed_corpus(
    corpus_rows: Sequence[Mapping[str, str]],
    *,
    lmstudio: LMStudio,
    model_key: str,
    quantization: str,
    document_prefix: str,
    corpus_bytes: bytes,
    timeout_s: float,
) -> CorpusEmbeddingsFile:
    """Step 2's own live mode (`docs/plans/small-model-benchmarking-s3-spec.md` §8 Step 2, §5):
    one batched `lmstudio.embed()` call over all 121 (`document_prefix`-applied) documents — RAW
    vectors, never pre-normalized at write time (spec §6.2/§9: `scoring/retrieval.py`'s ranking
    function L2-normalizes at read time, and the whole point of this file existing is to let the
    offline self-test and `prime()`'s own diagnostic observe the raw norm distribution too —
    normalizing here would throw that away).

    `quantization`/`corpus_bytes` are supplied by the caller (`main()`, which already reads
    `corpus.jsonl`'s bytes to build `corpus_rows` and resolves the live `ModelInfo` via
    `lmstudio.catalog()` + `_find_model_in_catalog`, spec §8 Step 2 item 1) rather than re-derived
    here, so this function stays a pure request/response step with no catalog call or file read of
    its own. `corpus_bytes` MUST be `corpus.jsonl`'s raw file bytes — not a re-serialization of
    `corpus_rows` — because `compute_cache_key`'s `corpusSha256` component must match the SAME
    hash `scoring/retrieval.py`'s own `_compute_cache_key` independently recomputes from
    `pack.data_path("corpus").read_bytes()` at run time (`-ml` §5.5); any other byte sequence here
    is a cache key that can never hit."""
    texts = [document_prefix + row["text"] for row in corpus_rows]
    embed_result = lmstudio.embed(texts, model=model_key, timeout_s=timeout_s)
    if len(embed_result.vectors) != len(corpus_rows):
        raise RefreshGoldenError(
            f"--embed-corpus: lmstudio.embed returned {len(embed_result.vectors)} vectors for "
            f"{len(corpus_rows)} documents — a truncated or malformed batch response"
        )
    vectors = {
        row["docId"]: vector
        for row, vector in zip(corpus_rows, embed_result.vectors, strict=True)
    }
    cache_key = compute_cache_key(
        model=model_key,
        quantization=quantization,
        document_prefix=document_prefix,
        corpus_bytes=corpus_bytes,
    )
    return CorpusEmbeddingsFile(
        cacheKey=cache_key,
        generatedAt=_utc_now_iso(),
        vectors=vectors,
    )


# --------------------------------------------------------------------------------------------
# The default (data-import) mode
# --------------------------------------------------------------------------------------------


def _run_import(
    repo_root: Path, pack_root: Path, pack_id: str, origins: Sequence[OriginSpec]
) -> None:
    provenance_path = pack_root / "PROVENANCE.md"
    _pack_version_gate(pack_root, provenance_path)

    now = _utc_now_iso()
    records: list[ProvenanceRecord] = []
    for origin in origins:
        if origin.kind == "check-only":
            continue

        origin_path = repo_root / origin.originPath
        origin_bytes = origin_path.read_bytes()
        dest_file, dest_key = _split_dest_fragment(origin.destPath)
        dest_path = pack_root / dest_file

        if origin.kind == "copy":
            dest_path.write_bytes(origin_bytes)
        elif origin.kind == "jsonl-transform":
            lines = origin_bytes.decode("utf-8").splitlines()
            rows = _JSONL_TRANSFORM_BY_PACK_ID[pack_id](lines)
            _write_jsonl(dest_path, rows)
        elif origin.kind == "ast-literal":
            rows = _AST_LITERAL_ROWS_BY_PACK_ID[pack_id](origin_bytes.decode("utf-8"))
            if dest_key is not None:
                _write_json_merge_key(dest_path, dest_key, rows)
            else:
                _write_jsonl(dest_path, rows)
        elif origin.kind == "schema-literal":
            schema = _read_schema_literal(origin_bytes.decode("utf-8"))
            dest_path.write_text(
                json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
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
# `--check-tables-shape` (S4 spec §2.6/§6) — local-only, never opens a network socket. Validates
# an ALREADY-PRODUCED `tables.json` (the operator hand-wrote/merged its "knowledge_base" half
# from a live, human-run `ws:nlq-eval` snapshot, §2.6) and writes PROVENANCE.md's entry for it.
# --------------------------------------------------------------------------------------------


def _tables_shape_problems(
    tables_kb: Mapping[str, list[dict[str, Any]]], schema_kb: Mapping[str, Any]
) -> list[str]:
    """Every `tables.json["knowledge_base"]` row under each of `schema_kb["labels"]`'s labels
    (`Entity`/`Document`/`Chunk`) must carry EXACTLY that label's declared properties — no more,
    no fewer (`querygen.DatasetSchema`'s own "curated allowlist" discipline, S4 spec §6)."""
    problems: list[str] = []
    labels = schema_kb.get("labels", {})
    for label, declared_props in labels.items():
        expected_keys = set(declared_props)
        for i, row in enumerate(tables_kb.get(label, [])):
            actual_keys = set(row)
            if actual_keys == expected_keys:
                continue
            missing = sorted(expected_keys - actual_keys)
            extra = sorted(actual_keys - expected_keys)
            parts = []
            if missing:
                parts.append(f"missing {missing!r}")
            if extra:
                parts.append(f"extra {extra!r}")
            problems.append(f"knowledge_base.{label}[{i}]: {' and '.join(parts)}")
    return problems


def run_check_tables_shape(pack_root: Path, *, source_git_sha: str) -> list[str]:
    """Validates `tables.json`'s `"knowledge_base"` half against `schema.json`'s own declared
    properties and, on success, ADDS one `PROVENANCE.md` row for the live snapshot (never
    clobbering the rows a prior default-import run already wrote). Refuses under an unchanged
    `packVersion`, the same `_pack_version_gate` `--embed-corpus` already shares (S4 spec §6).
    Returns the row counts it just counted, one entry per label."""
    provenance_path = pack_root / "PROVENANCE.md"
    _pack_version_gate(pack_root, provenance_path)

    manifest = json.loads((pack_root / "pack.json").read_text(encoding="utf-8"))
    schema = json.loads((pack_root / "schema.json").read_text(encoding="utf-8"))
    tables = json.loads((pack_root / "tables.json").read_text(encoding="utf-8"))
    tables_kb = tables.get("knowledge_base", {})

    problems = _tables_shape_problems(tables_kb, schema["knowledge_base"])
    if problems:
        raise RefreshGoldenError(
            "tables.json's knowledge_base shape does not match schema.json:\n"
            + "\n".join(problems)
        )

    counts = {label: len(rows) for label, rows in tables_kb.items()}
    existing = _read_provenance_records_full(provenance_path) if provenance_path.exists() else []
    new_record = ProvenanceRecord(
        originPath="ws:nlq-eval (live FalkorDB snapshot)",
        destPath="tables.json#knowledge_base",
        sourceGitSha=source_git_sha,
        sourceSha256=_sha256_bytes(
            json.dumps(tables_kb, sort_keys=True).encode("utf-8")
        ),
        copiedAt=_utc_now_iso(),
    )
    _write_provenance(pack_root, manifest["packVersion"], [*existing, new_record])
    return [f"{label}={n}" for label, n in sorted(counts.items())]


# --------------------------------------------------------------------------------------------
# `--stamp-answerability` (S4 spec §6) — nlq-structured-query only. Runs `reference_specs.json`'s
# hand-authored specs through the pack's own `tools/exec.py` and stamps `items.jsonl`.
# --------------------------------------------------------------------------------------------


def run_stamp_answerability(pack_root: Path) -> dict[str, int]:
    """Reads `reference_specs.json` (`itemId` -> a `QueryRequest`-shaped spec), runs each item's
    own reference spec through the pack's declared `tools.entrypoint`
    (`tools_exec.compile_and_execute`) against `tables.json`/`schema.json`, and stamps
    `items.jsonl`'s matching row `"answerable": true` iff the reference spec compiles, executes,
    and its result is non-empty when `expected.type != "not_found"` — a `not_found`-shaped item
    is answerable by construction (the correct answer IS an empty result); `false` otherwise (the
    reference spec itself fails Layer A/B, or executes to empty against a non-`not_found`
    expectation). Refuses on any pack other than `nlq-structured-query` and under an unchanged
    `packVersion`, the same `_pack_version_gate` every other write mode shares."""
    manifest = json.loads((pack_root / "pack.json").read_text(encoding="utf-8"))
    if manifest.get("packId") != "nlq-structured-query":
        raise RefreshGoldenError(
            f"--stamp-answerability is nlq-structured-query-only; refusing on pack "
            f"{manifest.get('packId')!r}"
        )
    provenance_path = pack_root / "PROVENANCE.md"
    _pack_version_gate(pack_root, provenance_path)

    pack = load_pack(pack_root)
    tool_module = pack.load_tool_module()
    entrypoint = getattr(tool_module, manifest["tools"]["entrypoint"])

    schema = json.loads((pack_root / "schema.json").read_text(encoding="utf-8"))
    tables = json.loads((pack_root / "tables.json").read_text(encoding="utf-8"))
    reference_specs = json.loads((pack_root / "reference_specs.json").read_text(encoding="utf-8"))

    items_path = pack_root / "items.jsonl"
    items = [
        json.loads(line)
        for line in items_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    counts = {"answerable": 0, "unanswerable": 0}
    stamped: list[dict[str, Any]] = []
    for item in items:
        spec = reference_specs.get(item["itemId"])
        dataset = item["dataset"]
        answerable = False
        if spec is not None and dataset in tables and dataset in schema:
            try:
                result = entrypoint(spec, tables=tables[dataset], schema=schema[dataset])
            except (tool_module.MalformedSpecError, tool_module.SchemaViolationError):
                answerable = False
            else:
                non_empty = bool(result.get("items"))
                answerable = non_empty or item["expected"]["type"] == "not_found"
        stamped.append({**item, "answerable": answerable})
        counts["answerable" if answerable else "unanswerable"] += 1

    _write_jsonl(items_path, stamped)
    return counts


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
        help="the one live-LM-Studio mode: embed corpus.jsonl and write corpus.embeddings.json",
    )
    parser.add_argument(
        "--check-tables-shape",
        action="store_true",
        help="local-only (nlq-structured-query): validate an already-produced tables.json's "
        "knowledge_base half against schema.json and write PROVENANCE.md's snapshot row "
        "(--source-git-sha required)",
    )
    parser.add_argument(
        "--source-git-sha",
        help="the falkor-chat commit the operator read the live ws:nlq-eval graph at "
        "(--check-tables-shape only)",
    )
    parser.add_argument(
        "--stamp-answerability",
        action="store_true",
        help="nlq-structured-query only: run reference_specs.json's specs through tools/exec.py "
        "and stamp items.jsonl's answerable field",
    )
    parser.add_argument("--model", help="LM Studio model key (--embed-corpus only)")
    parser.add_argument(
        "--api-base-url", default="http://localhost:1234", help="--embed-corpus only"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="seconds allowed for the warm-up call and the batched embed call each "
        "(--embed-corpus only; default 300.0, matching run's own firstCallTimeoutSeconds default)",
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
        try:
            origins = _origins_for_pack_id(_read_pack_id(pack_root))
        except RefreshGoldenError as exc:
            print(f"refresh_golden.py: {exc}", file=sys.stderr)
            return 1
        results = _check_origins(repo_root, pack_root, origins)
        for result in results:
            print(f"{result.status}: {result.originPath}")
        return 0 if all(r.status == "unchanged" for r in results) else 1

    if args.check_tables_shape:
        if not args.source_git_sha:
            print(
                "refresh_golden.py: --check-tables-shape requires --source-git-sha",
                file=sys.stderr,
            )
            return 2
        try:
            counts = run_check_tables_shape(pack_root, source_git_sha=args.source_git_sha)
        except RefreshGoldenError as exc:
            print(f"refresh_golden.py: {exc}", file=sys.stderr)
            return 1
        print(", ".join(counts))
        return 0

    if args.stamp_answerability:
        try:
            counts = run_stamp_answerability(pack_root)
        except RefreshGoldenError as exc:
            print(f"refresh_golden.py: {exc}", file=sys.stderr)
            return 1
        print(", ".join(f"{key}={value}" for key, value in sorted(counts.items())))
        return 0

    if args.embed_corpus:
        if not args.model:
            print("refresh_golden.py: --embed-corpus requires --model", file=sys.stderr)
            return 2

        # Same content-hash-changing-write refusal as the default import mode, and the SAME
        # function — corpus.embeddings.json is just as much this pack's own committed data as the
        # four `_run_import` writes are (spec §8 Step 2 item 1: "packVersion must already be
        # bumped"). Checked BEFORE any LM Studio contact, so a forgotten bump fails instantly and
        # for free, never after a live call has already run.
        provenance_path = pack_root / "PROVENANCE.md"
        try:
            _pack_version_gate(pack_root, provenance_path)
        except RefreshGoldenError as exc:
            print(f"refresh_golden.py: {exc}", file=sys.stderr)
            return 1

        manifest = json.loads((pack_root / "pack.json").read_text(encoding="utf-8"))
        document_prefix = manifest["embedding"]["documentPrefix"]

        corpus_path = pack_root / "corpus.jsonl"
        corpus_bytes = corpus_path.read_bytes()
        corpus_rows = [
            json.loads(line)
            for line in corpus_bytes.decode("utf-8").splitlines()
            if line.strip()
        ]

        client = LMStudio(args.api_base_url)
        try:
            catalog = client.catalog()
            model_info = _find_model_in_catalog(catalog, args.model)
            # The mandatory warm-up (mirrors run_pack's own step 4, runner.py): never assume the
            # model is already resident — LM Studio's own load path is a call landing on it, not
            # something `catalog()` alone triggers. Its content is discarded, exactly like the
            # runner's warm-up; the point is to isolate the cold-load cost from the real batched
            # embed call below, not to time it.
            resident_before = client.residency()
            client.warm_up(
                args.model,
                call_surface="embeddings",
                system_prompt=None,
                was_resident_before=any(r.id == args.model for r in resident_before),
                timeout_s=args.timeout,
            )
            result = embed_corpus(
                corpus_rows,
                lmstudio=client,
                model_key=args.model,
                quantization=model_info.quantization,
                document_prefix=document_prefix,
                corpus_bytes=corpus_bytes,
                timeout_s=args.timeout,
            )
        except RefreshGoldenError as exc:
            print(f"refresh_golden.py: {exc}", file=sys.stderr)
            return 1
        except LMStudioError as exc:
            print(f"refresh_golden.py: --embed-corpus call failed: {exc}", file=sys.stderr)
            return 1

        dest_path = pack_root / "corpus.embeddings.json"
        dest_path.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
        print(f"wrote {dest_path}")
        return 0

    try:
        pack_id = _read_pack_id(pack_root)
        origins = _origins_for_pack_id(pack_id)
        _run_import(repo_root, pack_root, pack_id, origins)
    except RefreshGoldenError as exc:
        print(f"refresh_golden.py: {exc}", file=sys.stderr)
        return 1
    print(f"refreshed: {pack_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
