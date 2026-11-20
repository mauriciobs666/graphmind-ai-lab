#!/usr/bin/env python3
"""One-off repair (K-102): re-embed `Chunk`s that have no `embedding` in one workspace.

Background: a document's chunks whose embed job failed (typically an LM Studio model-swap
timeout) have `embedding IS NULL` and are invisible to `search_documents`. This script
finds them and re-embeds each through the normal `EmbeddingWorker.embed_chunk` path
(workspace embedding model, FR-19 dimension guard, `Repository.set_chunk_embedding`).

* Default is a **dry run**: it lists the chunks and makes no model call and no write.
  `--apply` performs the embed calls and writes.
* It **never changes `Document.status`**: a document that failed stays `failed`. The
  closing report lists every `failed` document, so you can see which are now fully
  embedded and decide what to do with them.
* Human-started only; embed calls load the embedding model on LM Studio (swapping out
  whatever is resident), so run it when nothing else is using the GPU.

Usage:  ./scripts/reembed_unembedded_chunks.sh <workspaceId> [--apply]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_SERVER_DIR = Path(__file__).resolve().parents[1] / "server"
sys.path.insert(0, str(_SERVER_DIR))


@dataclass
class RunResult:
    found: int
    embedded: int
    errors: list[tuple[str, str]] = field(default_factory=list)  # (chunkId, message)


def run(ws: str, *, repo: Any, worker: Any, apply: bool, out=print) -> RunResult:
    chunks = repo.list_unembedded_chunks(ws)
    mode = "APPLY" if apply else "DRY-RUN (no model calls, no writes; pass --apply)"
    out(f"[{mode}] ws:{ws} — {len(chunks)} chunk(s) without an embedding")
    for c in chunks:
        out(f"  chunk {c['chunkId']}  doc {c['documentId']} [{c['documentStatus']}] "
            f"{c['documentTitle']!r}")
    result = RunResult(found=len(chunks), embedded=0)
    if not apply:
        return result

    for c in chunks:
        try:
            worker.embed_chunk(ws, chunk_id=c["chunkId"], text=c["text"])
        except Exception as exc:  # noqa: BLE001 — report and carry on with the rest
            result.errors.append((c["chunkId"], f"{type(exc).__name__}: {exc}"))
            out(f"  FAILED {c['chunkId']}: {type(exc).__name__}: {exc}")
        else:
            result.embedded += 1
            out(f"  embedded {c['chunkId']}")
    out(f"embedded {result.embedded}/{result.found}; {len(result.errors)} error(s)")

    failed = repo.list_failed_documents(ws)
    done = [d for d in failed if d["unembeddedChunks"] == 0]
    todo = [d for d in failed if d["unembeddedChunks"] > 0]
    out(f"documents still 'failed' with all chunks embedded ({len(done)}) — status untouched:")
    for d in done:
        out(f"  {d['documentId']}  {d['documentTitle']!r}")
    if todo:
        out(f"documents still 'failed' with chunks still unembedded ({len(todo)}):")
        for d in todo:
            out(f"  {d['documentId']}  {d['documentTitle']!r}  ({d['unembeddedChunks']} chunk(s))")
    return result


def _build(ws: str) -> tuple[Any, Any]:
    from falkorchat import db
    from falkorchat.embedding import EmbeddingWorker
    from falkorchat.modelconfig import GraphWorkspaceOverrides, ModelGateway
    from falkorchat.repository import Repository

    repo = Repository(db.connect())
    models = ModelGateway.from_env(workspace_overrides=GraphWorkspaceOverrides(repo))
    return repo, EmbeddingWorker(repo, models=models)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("workspace")
    parser.add_argument("--apply", action="store_true",
                        help="perform the embed calls and writes (default: dry run)")
    args = parser.parse_args(argv)
    repo, worker = _build(args.workspace)
    result = run(args.workspace, repo=repo, worker=worker, apply=args.apply)
    return 1 if result.errors else 0


if __name__ == "__main__":
    sys.exit(main())
