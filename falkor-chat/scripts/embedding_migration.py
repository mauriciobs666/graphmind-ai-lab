#!/usr/bin/env python3
"""Embedding-model migration tool (K-042-adjacent, `docs/plans/embedding-migration.md`).

Two subcommands:

  * `pin`     — FR-1/FR-2's snapshot/pin-at-birth operation: one workspace,
                idempotent, no model call beyond resolving the current default ref.
  * `migrate` — FR-3/FR-4/FR-5/FR-8/FR-10's re-embed + index-rebuild + override-move
                operation (§5 steps 4-5).

Same posture as `scripts/seed_eval_corpus.py`: a plain Python script, run via
`server/.venv/bin/python` (through the `.sh` wrapper), that imports `falkorchat`
directly rather than reimplementing model-config resolution in bash.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
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


class MigrationAbortedError(RuntimeError):
    """Raised whenever `migrate()` refuses to proceed past a named precondition
    (§3.4 step 0's traffic-stop gate, §4 item 3's count check, §3.3's missing-dim
    guard) — always before the next write/HTTP-call this precondition protects."""


# (label, id-property) pairs `migrate` re-embeds, in the order §3.4 step 1 fixes
# ("all `Message` rows to completion, then all `Chunk` rows"). Never derived from
# caller input (`-graph.md` item 5's own warning about DDL identifiers).
_MIGRATION_LABELS: tuple[tuple[str, str], ...] = (("Message", "msgId"), ("Chunk", "chunkId"))


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


@dataclass
class LabelReport:
    """Per-label outcome of one `migrate()` call (§5 step 4h)."""

    label: str
    migrated: int
    total: int
    skipped: int = 0


@dataclass
class MigrationReport:
    """`migrate()`'s return value (§5 step 4h) — printed AND returned, so the
    `.sh` wrapper can echo a summary and a test can assert on the structure
    directly."""

    ws: str
    target_ref: str
    labels: dict[str, LabelReport] = field(default_factory=dict)
    elapsed_seconds: float = 0.0


# ── §4's confirmed Cypher shapes, verbatim per `-graph.md` — one helper per item.
# Raw queries against `repo._graph(ws)` rather than new Repository methods
# (`-graph.md` item 2: "two new repository methods (or inline queries in
# embedding_migration.py)... either is a small implementer choice") — chosen here
# specifically so this unit never touches `repository.py`, which a concurrent
# session is editing for an unrelated feature. `repo` is always duck-typed: real
# calls get a real `Repository`, tests substitute a fake/spy exposing the same
# surface (`_graph`, `read_index_dimension`, `read_model_overrides`,
# `write_model_overrides`) — this is also what makes the step-0 precondition
# ("zero GRAPH.QUERY calls") assertable via a repo double that raises on any
# method call. ──────────────────────────────────────────────────────────────────

def _read_unmigrated_batch(
    repo: Any, ws: str, *, label: str, id_prop: str, last_id: str,
    target_ref: str, batch_size: int,
) -> list[tuple[str, str]]:
    """§4 item 1: keyset read, `>` only (verified-safe direction per `-graph.md`
    item 1's engine-bug finding — never flip this to `<=`/`<`)."""
    res = repo._graph(ws).ro_query(
        f"MATCH (n:{label}) "
        f"WHERE n.{id_prop} > $lastId AND coalesce(n.embeddingModel, '') <> $target "
        f"RETURN n.{id_prop} AS id, n.text AS text "
        f"ORDER BY n.{id_prop} ASC LIMIT $b",
        {"lastId": last_id, "target": target_ref, "b": batch_size},
    )
    return [(row[0], row[1]) for row in res.result_set]


def _write_embedding(
    repo: Any, ws: str, *, label: str, id_prop: str, id_value: str,
    embedding: list[float], target_ref: str,
) -> bool:
    """§4 item 2: write one row's new embedding + the migration-owned
    `embeddingModel` marker. Returns `False` (skip-and-log, §3.3) on the
    vanished-row no-op (`properties_set == 0`)."""
    res = repo._graph(ws).query(
        f"MATCH (n:{label} {{{id_prop}: $id}}) "
        "SET n.embedding = vecf32($embedding), n.embeddingModel = $target",
        {"id": id_value, "embedding": list(embedding), "target": target_ref},
    )
    return res.properties_set > 0


def _count_unmigrated(repo: Any, ws: str, *, label: str, target_ref: str) -> tuple[int, int]:
    """§4 item 3: two label-scan counts, verified clean on an empty graph."""
    unmigrated_res = repo._graph(ws).ro_query(
        f"MATCH (n:{label}) WHERE coalesce(n.embeddingModel, '') <> $target "
        "RETURN count(n) AS unmigrated",
        {"target": target_ref},
    )
    total_res = repo._graph(ws).ro_query(f"MATCH (n:{label}) RETURN count(n) AS total")
    return unmigrated_res.result_set[0][0], total_res.result_set[0][0]


def _rebuild_vector_index(repo: Any, ws: str, *, label: str, dim: int) -> None:
    """§4 item 5, guarded per `-graph.md`'s resumability hazard: skip the `DROP`
    when no vector index exists at all (`read_index_dimension(...) is None` —
    the crash-between-drop-and-create resume state), never call it
    unconditionally."""
    if repo.read_index_dimension(ws, label=label) is not None:
        repo._graph(ws).query(f"DROP VECTOR INDEX FOR (n:{label}) ON (n.embedding)")
    repo._graph(ws).query(
        f"CREATE VECTOR INDEX FOR (n:{label}) ON (n.embedding) "
        f"OPTIONS {{dimension: {dim}, similarityFunction: 'cosine'}}"
    )


def _reembed_label(
    repo: Any, ws: str, *, label: str, id_prop: str, target_ref: str,
    embedder: Any, batch_size: int,
) -> int:
    """§3.3/§3.4 step 1: the keyset loop for one label, to completion. Returns
    the count of vanished (skip-and-logged, never retried/aborted) rows."""
    last_id = ""
    skipped = 0
    while True:
        batch = _read_unmigrated_batch(
            repo, ws, label=label, id_prop=id_prop, last_id=last_id,
            target_ref=target_ref, batch_size=batch_size,
        )
        if not batch:
            break
        for row_id, text in batch:
            embedding = embedder.embed(text)
            written = _write_embedding(
                repo, ws, label=label, id_prop=id_prop, id_value=row_id,
                embedding=embedding, target_ref=target_ref,
            )
            if not written:
                skipped += 1
                print(
                    f"WARNING: ws:{ws} {label} {row_id!r} vanished between read "
                    "and write — skipping (§3.3's skip-and-log decision; not "
                    "counted as migrated, batch continues)"
                )
        last_id = batch[-1][0]
        print(f"ws:{ws} {label}: migrated {len(batch)} rows (batch ending at {last_id!r})")
    return skipped


def migrate(
    ws: str,
    target_ref: str,
    *,
    batch_size: int = 50,
    traffic_stopped: bool = False,
    repo: Any = None,
    gateway: Any = None,
) -> MigrationReport:
    """FR-3/FR-4/FR-5/FR-8/FR-10: re-embed `ws`'s `Message`/`Chunk` rows to
    `target_ref`, rebuild both vector indexes at the new dimension, and move the
    workspace's `embeddingModelOverride` (`docs/plans/embedding-migration.md` §5
    step 4).

    Idempotent-resume (FR-10): every step below is keyed off the graph's own
    state (`coalesce(embeddingModel,'') <> target_ref`, `read_index_dimension`),
    never off a script-local cursor — re-running after any crash, at any point,
    converges to the same end state and re-processes only what's left.

    a. **Hard precondition (§3.4 step 0).** `traffic_stopped` must be `True` —
       checked before touching `repo`/`gateway` at all (no `_default_repo()`
       connection, no `ModelGateway.from_env()` config parse), so a caller can
       assert zero graph/HTTP calls on this path with a repo/gateway double that
       raises on any use.
    b. Preflight: resolve `target_ref` with **no `ws=`/`overrides=`** (§3.3's
       hard-cap bypass) and refuse to start if its `dim` isn't declared.
    c. Re-embed `Message` to completion, then `Chunk` to completion (§3.4 step 1),
       via `embedder = gateway.embedder("embedding", requested=target_ref)` —
       again no `ws=`, built once and reused for every row.
    d. §4 item 3's count check for both labels; abort before the index rebuild if
       either reports a nonzero unmigrated count.
    e. §4 item 5's guarded drop+recreate at `target_ref`'s declared dimension.
    f. FR-5: move the override (read-before-write discipline, same landmine as
       `pin`).
    g. Print the explicit restart instruction (§2.7/§3.4 step 5 — this script
       cannot restart a process it isn't running as).
    """
    if not traffic_stopped:
        raise MigrationAbortedError(
            f"traffic must be stopped before migrating ws:{ws} — stop the "
            f"falkorchat.app process serving FALKORCHAT_WS_ID={ws!r} (§2.6), "
            "then re-run with --i-have-stopped-traffic (or confirm the "
            "interactive prompt) once it is down"
        )

    if repo is None:
        repo = _default_repo()
    if gateway is None:
        gateway = ModelGateway.from_env()

    start = time.monotonic()

    # b. Preflight — deliberately no `ws=`/`overrides=` (§3.3's hard-cap bypass):
    # the workspace override still names the OLD model until step f below.
    resolution = gateway.resolve("embedding", requested=target_ref)
    target_dim = resolution.primary.dim
    if target_dim is None:
        raise MigrationAbortedError(
            f"models.{target_ref!r}.dim is not declared in the model-config "
            "overlay — declare it (docs/plans/embedding-migration.md §3.3) "
            "before migrating; refusing to start (no HTTP call, no graph write)"
        )
    embedder = gateway.embedder("embedding", requested=target_ref)

    # c. Re-embed, one label to completion at a time.
    skipped: dict[str, int] = {}
    for label, id_prop in _MIGRATION_LABELS:
        skipped[label] = _reembed_label(
            repo, ws, label=label, id_prop=id_prop, target_ref=target_ref,
            embedder=embedder, batch_size=batch_size,
        )

    # d. Count check — abort before the index ever moves if anything is left.
    label_reports: dict[str, LabelReport] = {}
    unmigrated_by_label: dict[str, int] = {}
    for label, _ in _MIGRATION_LABELS:
        unmigrated, total = _count_unmigrated(repo, ws, label=label, target_ref=target_ref)
        unmigrated_by_label[label] = unmigrated
        label_reports[label] = LabelReport(
            label=label, migrated=total - unmigrated, total=total, skipped=skipped[label],
        )
    if any(unmigrated_by_label.values()):
        details = ", ".join(f"{lbl}: {n} unmigrated" for lbl, n in unmigrated_by_label.items())
        raise MigrationAbortedError(
            f"ws:{ws} count check failed after re-embed — refusing to rebuild "
            f"the vector index while rows remain unmigrated ({details})"
        )

    # e. Guarded index rebuild.
    for label, _ in _MIGRATION_LABELS:
        _rebuild_vector_index(repo, ws, label=label, dim=target_dim)

    # f. FR-5 — read-before-write, same discipline as `pin`.
    current = repo.read_model_overrides(ws)
    repo.write_model_overrides(
        ws,
        agent=current["agentModel"],
        guard=current["guardModel"],
        embedding=target_ref,
        responder=current["responderModel"],
        at=int(time.time() * 1000),
        by="embedding_migration.migrate",
    )

    # g. The one step this tool cannot do for the operator.
    print(
        f"ws:{ws} migrated to {target_ref!r} at dim {target_dim} — RESTART the "
        f"falkorchat server process serving this workspace NOW, before resuming "
        "traffic (clears EmbeddingWorker's process-lifetime index-dimension cache)"
    )

    elapsed = time.monotonic() - start
    for label, lr in label_reports.items():
        print(
            f"ws:{ws} {label}: migrated {lr.migrated}/{lr.total} "
            f"(skipped {lr.skipped}) in {elapsed:.1f}s total"
        )
    return MigrationReport(ws=ws, target_ref=target_ref, labels=label_reports, elapsed_seconds=elapsed)


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

    migrate_parser = subparsers.add_parser(
        "migrate",
        help="Re-embed one workspace's Message/Chunk rows to a new model, "
        "rebuild its vector indexes, and move its embeddingModelOverride "
        "(FR-3/FR-4/FR-5/FR-8/FR-10).",
    )
    migrate_parser.add_argument("workspace", help="workspace id to migrate (e.g. eval)")
    migrate_parser.add_argument(
        "target_ref", help="target embedding model ref, e.g. lmstudio/granite-embedding-278m-multilingual"
    )
    migrate_parser.add_argument(
        "--batch-size", type=int, default=50,
        help="rows read per keyset page (default: 50)",
    )
    migrate_parser.add_argument(
        "--i-have-stopped-traffic", dest="traffic_stopped", action="store_true",
        help="confirm the falkorchat.app process serving this workspace (§2.6) "
        "is stopped — required before any graph access; omit to be prompted "
        "interactively when attached to a terminal",
    )

    return parser


def _confirm_traffic_stopped(ws: str) -> bool:
    """Interactive fallback for `--i-have-stopped-traffic` (§3.4 step 0) — only
    ever consulted when stdin is a terminal; a non-interactive run with no flag
    aborts outright (test 6: zero graph access either way)."""
    if not sys.stdin.isatty():
        return False
    prompt = (
        f"Stop the falkorchat.app process serving FALKORCHAT_WS_ID={ws!r} "
        "before continuing (§2.6). Has it been stopped? [y/N] "
    )
    answer = input(prompt).strip().lower()
    return answer in ("y", "yes")


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    if args.command == "pin":
        for ws in args.workspace:
            pin(ws)
        return 0

    if args.command == "migrate":
        traffic_stopped = args.traffic_stopped or _confirm_traffic_stopped(args.workspace)
        migrate(
            args.workspace, args.target_ref,
            batch_size=args.batch_size, traffic_stopped=traffic_stopped,
        )
        return 0

    raise AssertionError(f"unhandled command {args.command!r}")  # pragma: no cover


if __name__ == "__main__":
    sys.exit(main())
