"""`coordgraph propose <component|agent> [--per-milestone N]` (plan section 3.7): the read-only adapter.

The pure planner is `propose.propose`; this module only feeds it: the scope's open items (graph note 7.5 (i)),
each distinct dependency token resolved through `identity` and the graph (7.5 (ii)/7.6 (a): open in another
backlog) or the owning history file on disk (delivery-shaped heading). It issues `RO_QUERY` reads only and
writes nothing (AC-14)."""
from __future__ import annotations

import sys
from pathlib import Path

from . import commands_store, drift, sync
from .discover import discover
from .history import HistoryIndex
from .parse.backlog import STATUS_WORDS, Item
from .propose import DepResolution, propose
from .scope import load_legacy_paths

BACKLOG_EXISTS = "MATCH (b:Backlog {scope: $scope}) RETURN count(b)"
ITEMS = (
    "MATCH (b:Backlog {scope: $scope})-[:HAS_ITEM]->(i:Item) WHERE i.open = true "
    "RETURN i.ref, i.scope, i.statusMarker, i.title, i.priority, i.milestone, i.milestoneOrder, "
    "i.milestoneStarted, i.dependsOn, i.touches, b.path, i.ord ORDER BY i.ord"
)
DEP_ITEMS = "MATCH (d:Item) WHERE d.ref IN $candidates RETURN d.ref, d.scope, d.open, d.backlogPath"


def _word(marker: str | None) -> str | None:
    lowered = (marker or "").casefold()
    return next((w for w in STATUS_WORDS if w in lowered), None)


def fetch_items(graph, scope: str) -> list[Item]:
    rows = graph.ro_query(ITEMS, {"scope": scope}).result_set
    return [
        Item(ref=r[0], item_id=None, scope=r[1], backlog_path=r[10], ord=r[11], line=0, title=r[3], item_hash="",
             status_marker=r[2], status=_word(r[2]), priority=r[4], milestone=r[5], milestone_order=r[6],
             milestone_started=r[7], depends_on=r[8], touches=r[9])
        for r in rows
    ]


def make_resolver(graph, identity, history: HistoryIndex):
    """`propose`'s resolver: candidates from `identity`; open elsewhere from the graph; delivered from history."""

    def resolve(token: str, referrer_scope: str) -> DepResolution:
        res = identity.resolve(token, referrer_scope)
        cands = frozenset(res.candidates)
        rows = graph.ro_query(DEP_ITEMS, {"candidates": sorted(cands)}).result_set
        external = next(((r[0], r[3]) for r in rows if r[2]), None)  # own-backlog items are matched by `propose` first
        hit = history.delivered(token, referrer_scope, identity.candidates)
        reason = identity.reason(token, referrer_scope) or "not in any open backlog"
        return DepResolution(cands, external, hit[0] if hit else None, reason)

    return resolve


def propose_cmd(args) -> int:
    from .identity import load_identity  # noqa: PLC0415 - read side only, deliberately lazy

    try:
        cfg, _key, root, _db, graph = commands_store._target(args)
        root = Path(root)
        if graph.ro_query(BACKLOG_EXISTS, {"scope": args.scope}).result_set[0][0] == 0:
            print(f"propose: no synced backlog for {args.scope!r} (run coordgraph sync?)", file=sys.stderr)
            return 1
        parser = sync.Parser.for_root(root)
        report = drift.check(graph, parser)
        identity = load_identity(root)
        history = HistoryIndex.from_repo(root, [f.path for f in discover(root) if f.kind == "backlog"],
                                         load_legacy_paths(root))
        items = fetch_items(graph, args.scope)
        proposal = propose(items, make_resolver(graph, identity, history), args.per_milestone)
    except Exception as exc:  # noqa: BLE001
        return commands_store._fail("propose", exc)
    print(drift.render_header(report, commands_store._hook_active(cfg), identity.registry_line()))
    print(proposal.render() or "nothing to propose")
    return 0


COMMANDS = {"propose": propose_cmd}


def register(sub) -> None:
    p = sub.add_parser("propose", help="read-only milestone proposal for one component or agent backlog")
    p.add_argument("scope", metavar="COMPONENT_OR_AGENT")
    p.add_argument("--per-milestone", type=int, default=6, metavar="N", help="items per milestone (default 6)")
    p.add_argument("--graph", help="graph key (default: COORDGRAPH_KEY)")
