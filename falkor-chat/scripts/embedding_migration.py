#!/usr/bin/env python3
"""Embedding-model migration tool (K-042-adjacent, `docs/plans/embedding-migration.md`).

Two subcommands (only `pin` is landed in this file so far — `migrate` is a
follow-up unit, sequenced after this one since it shares this same module):

  * `pin`     — FR-1/FR-2's snapshot/pin-at-birth operation: one workspace,
                idempotent, no model call beyond resolving the current default ref.
  * `migrate` — FR-3/FR-4/FR-5/FR-8/FR-10's re-embed + index-rebuild + override-move
                operation. Not built yet.

Same posture as `scripts/seed_eval_corpus.py`: a plain Python script, run via
`server/.venv/bin/python` (through the `.sh` wrapper), that imports `falkorchat`
directly rather than reimplementing model-config resolution in bash.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]  # .../falkor-chat
_SERVER_DIR = _REPO_ROOT / "server"

# Defensive fallback for a direct `python3 scripts/embedding_migration.py` run
# outside the venv's own site-packages resolution — the .sh wrapper always uses
# `server/.venv/bin/python`, which already resolves `falkorchat` (same pattern as
# `seed_eval_corpus.py`).
sys.path.insert(0, str(_SERVER_DIR))

from falkorchat import db  # noqa: E402
from falkorchat.modelconfig import ModelConfigError, ModelGateway  # noqa: E402
from falkorchat.repository import Repository  # noqa: E402


def _default_repo() -> Repository:
    """Shared setup for every subcommand: a `Repository` over a real connection.
    Factored out so `migrate` (a follow-up unit) can reuse it without duplicating
    the `db.connect()` wiring."""
    return Repository(db.connect())


def pin(
    ws: str, *, repo: Repository | None = None, gateway: ModelGateway | None = None,
) -> dict[str, Any] | None:
    """FR-1/FR-2: idempotently pin `ws`'s `embeddingModelOverride` to the current
    global default (`docs/plans/embedding-migration.md` §3.2/§5 step 1).

    - If `ws` already has an explicit `embeddingModelOverride`, this is a no-op —
      the existing value is returned unchanged, and `write_model_overrides` is
      never called (idempotent, FR-1's "existing workspace unaffected by a later
      default change" acceptance criterion).
    - Otherwise, the workspace is pinned to
      `gateway.resolve("embedding").primary.ref` — the current global default, read
      fresh on every call, never hardcoded — while `agent`/`guard`/`responder`
      overrides are read back and passed through **unchanged** (the read-before-
      write discipline `write_model_overrides` requires: a `None` argument CLEARS
      that override kind, so this must never pass `None` for a kind it isn't
      touching).
    - `ModelGateway.from_env()` failure (`ModelConfigError` — e.g. no valid
      `FALKORCHAT_OPENCODE_CONFIG`) is **non-fatal**: this is deliberate (§3.2's
      Option B mitigation for `FALKORCHAT_ENABLE_AGENT=0`'s documented no-config
      mode) — printed as a WARNING, `None` returned, nothing raised, no graph
      access attempted.

    Returns a `read_model_overrides`-shaped dict (`{agentModel, guardModel,
    embeddingModel, responderModel}`) reflecting the post-call state, or `None`
    when the gateway could not be built at all.
    """
    if gateway is None:
        try:
            gateway = ModelGateway.from_env()
        except ModelConfigError as exc:
            print(
                f"WARNING: could not pin the embedding-model override for "
                f"{ws!r}: {exc} — this workspace is uncovered by FR-1/FR-2's "
                f"safety net until pin_workspace_embedding_model.sh {ws} is "
                f"re-run once a valid model config is available"
            )
            return None

    if repo is None:
        repo = _default_repo()

    current = repo.read_model_overrides(ws)
    if current["embeddingModel"]:
        print(
            f"ws:{ws} already pinned to {current['embeddingModel']!r} — no-op"
        )
        return current

    default_ref = gateway.resolve("embedding").primary.ref
    written = repo.write_model_overrides(
        ws,
        agent=current["agentModel"],
        guard=current["guardModel"],
        embedding=default_ref,
        responder=current["responderModel"],
        at=int(time.time() * 1000),
        by="embedding_migration.pin",
    )
    print(f"ws:{ws} pinned to {default_ref!r}")
    return {
        "agentModel": written["agent"],
        "guardModel": written["guard"],
        "embeddingModel": written["embedding"],
        "responderModel": written["responder"],
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="embedding_migration",
        description=(
            "Embedding-model migration tool — pin a workspace's embedding-model "
            "override, or (a follow-up unit) migrate its stored embeddings to a "
            "new model."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    pin_parser = subparsers.add_parser(
        "pin",
        help="Idempotently pin one or more workspaces' embeddingModelOverride "
        "to the current global default (FR-1/FR-2).",
    )
    pin_parser.add_argument(
        "workspace", nargs="+", help="workspace id(s) to pin (e.g. acme, eval)"
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    if args.command == "pin":
        for ws in args.workspace:
            pin(ws)
        return 0

    raise AssertionError(f"unhandled command {args.command!r}")  # pragma: no cover


if __name__ == "__main__":
    sys.exit(main())
