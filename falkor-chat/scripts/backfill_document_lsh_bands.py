#!/usr/bin/env python3
"""One-off backfill (K-091): stamp `Document.lshBand0..7` on documents that lack them.

Background: before K-091 the ingest write stored no LSH bands, so
`Repository.find_update_shortlist`'s band lookup matched nothing. New documents now get
their bands at ingest (`Services.ingest_document`); this script gives every older
`Document` (current or not) the bands its own text yields, through the same
`update_detection.document_lsh_bands` helper the ingest path and detection use.

* Default is a **dry run**: it lists the band-less documents and writes nothing.
  `--apply` writes the bands.
* Idempotent: it selects only documents missing a band, so a second run finds none.
* No model call, no LLM, no embedding: pure Python over the stored `Document.text`.
* **Rollout order:** first restart every server that serves the workspace on K-091
  code (so new ingests store their own bands), then `--apply`. A document ingested by
  a pre-K-091 server in between is band-less; re-running repairs it, which is safe.
* A workspace that fails (e.g. a mistyped id: no such graph) is reported on stderr
  and the remaining workspaces still run; the exit code is then 1.

Usage:  ./scripts/backfill_document_lsh_bands.sh <workspaceId> [<workspaceId> ...] [--apply]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SERVER_DIR = Path(__file__).resolve().parents[1] / "server"
sys.path.insert(0, str(_SERVER_DIR))

from falkorchat import update_detection as ud  # noqa: E402


@dataclass
class RunResult:
    found: int
    written: int


def run(ws: str, *, repo: Any, apply: bool, out=print) -> RunResult:
    docs = repo.list_documents_missing_lsh_bands(ws)
    mode = "APPLY" if apply else "DRY-RUN (no writes; pass --apply)"
    out(f"[{mode}] ws:{ws} — {len(docs)} document(s) without LSH bands")
    for d in docs:
        out(f"  doc {d['documentId']}  {d['title']!r}")
    result = RunResult(found=len(docs), written=0)
    if not apply:
        return result
    for d in docs:
        if repo.set_document_lsh_bands(
            ws, document_id=d["documentId"], lsh_bands=ud.document_lsh_bands(d["text"] or ""),
        ):
            result.written += 1
    out(f"wrote bands on {result.written}/{result.found} document(s) in ws:{ws}")
    return result


def _build() -> Any:
    from falkorchat import db
    from falkorchat.repository import Repository

    return Repository(db.connect())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("workspaces", nargs="+", metavar="workspace")
    parser.add_argument("--apply", action="store_true",
                        help="write the bands (default: dry run)")
    args = parser.parse_args(argv)
    from redis.exceptions import ResponseError

    repo = _build()
    failed = False
    for ws in args.workspaces:
        try:
            run(ws, repo=repo, apply=args.apply)
        except ResponseError as exc:  # a mistyped id ("empty key") or a failed write — report, carry on
            failed = True
            print(f"ERROR ws:{ws}: read or write failed: {exc}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
