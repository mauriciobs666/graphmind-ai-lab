#!/usr/bin/env python3
"""Refresh the golden-query embedding cache (K-026, `docs/plans/graphrag-eval.md`
§5 Unit 2a / §3 D3).

**Live, manually-run script — never collected by pytest.** Its filename matches
neither `test_*.py` nor `*_test.py`, and it is never imported by a pytest test, so
`server/tests/conftest.py`'s autouse `_model_config_env` fixture (which redirects
`ModelGateway.from_env()` to the dim-4 test-fixture config for every collected
test) never touches this file — exactly like `scripts/seed_eval_corpus.py` (plan
§3 D7's asymmetry note). That is why `ModelGateway.from_env()` is the *correct*
call here, resolving against the real `config/models.json`, and would be the
*wrong* call inside `test_golden_set_integrity.py`.

What this does:

  1. Load `golden_retrieval.jsonl`.
  2. Load the existing `golden_retrieval.embeddings.json` cache, if any.
  3. Resolve the current embedding model via `ModelGateway.from_env().resolve(...)`.
  4. For each golden query whose cache entry is missing, or whose cached `model`
     doesn't match the current model ref, embed the query live and update the
     cache entry.
  5. Write the updated cache back to disk (only if anything changed).

Run with:

    cd server && .venv/bin/python tests/eval/embed_golden_queries.py

Requires the configured embedding provider (LM Studio in the dev environment) to
be reachable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_EVAL_DIR = Path(__file__).resolve().parent
_SERVER_DIR = _EVAL_DIR.parents[1]  # .../server
_GOLDEN_PATH = _EVAL_DIR / "golden_retrieval.jsonl"
_EMBEDDINGS_PATH = _EVAL_DIR / "golden_retrieval.embeddings.json"

# Defensive fallback for a direct `python3 embed_golden_queries.py` invocation
# outside the venv's own site-packages resolution — mirrors seed_eval_corpus.py.
sys.path.insert(0, str(_SERVER_DIR))

from falkorchat import modelconfig  # noqa: E402


def _load_golden_queries() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with _GOLDEN_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_cache() -> dict[str, dict[str, object]]:
    if not _EMBEDDINGS_PATH.exists():
        return {}
    with _EMBEDDINGS_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    rows = _load_golden_queries()
    print(f"Loaded {len(rows)} golden queries from {_GOLDEN_PATH.name}.")

    cache = _load_cache()
    print(f"Loaded {len(cache)} existing cache entries from {_EMBEDDINGS_PATH.name}.")

    print("Resolving embedding model via ModelGateway.from_env()...")
    gateway = modelconfig.ModelGateway.from_env()
    resolution = gateway.resolve("embedding")
    model_ref = resolution.primary.ref
    print(f"  embedding model: {model_ref}")

    embedder = None  # built lazily — only if something actually needs embedding
    updated = 0
    for row in rows:
        golden_id = str(row["id"])
        entry = cache.get(golden_id)
        if entry is not None and entry.get("model") == model_ref:
            continue  # already current — skip the live call
        if embedder is None:
            embedder = gateway.embedder("embedding")
        query = str(row["query"])
        vector = embedder.embed(query)
        cache[golden_id] = {"model": model_ref, "vector": vector}
        updated += 1
        print(f"  embedded {golden_id!r} ({updated}/{len(rows)} so far)")

    if updated == 0:
        print("Cache already current for every golden query — nothing to do.")
        return

    _EMBEDDINGS_PATH.write_text(
        json.dumps(cache, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"Updated {updated} cache entr{'y' if updated == 1 else 'ies'}; "
        f"wrote {_EMBEDDINGS_PATH.relative_to(_SERVER_DIR.parent)}."
    )


if __name__ == "__main__":
    main()
