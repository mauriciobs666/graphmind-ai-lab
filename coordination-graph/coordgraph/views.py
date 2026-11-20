"""The read views: `inflight [--waiting]`, `unit <slug|ID> [<unit>]`, `planned <component|agent> [--all]`
(plan section 3.4/3.4a; graph note v4 sections 7.2, 7.3, 7.4, 7.6; plan3 sections 3.1, 3.2, 3.4).

Every statement runs through `RO_QUERY` and every answer prints the sync header first (`drift`), with
`⚠ drift` on each row whose file row no longer matches the graph. The file wins; the view only labels.

Identity is resolved at read time, never stored (plan3 3.1): `unit` expands an ID argument through
`Identity.resolve` and joins `IN $candidates` on the indexed raw strings (`Source.itemId`, `Unit.itemId`), so
a new ID, an alias and the old legacy name find the same ledger or unit; `inflight` uses the same candidates
to decide whether a requirement is "picked up" by a ledger carrying its `Item:` (note 7.6(d)). The slug
join is the fallback and is never skipped (plan3 3.2). `select_units` is the one lookup rule shared with
the need resolver. Imports no `wilib` itself: the registry arrives as an `Identity` argument.
"""
from __future__ import annotations

import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from . import commands_store, drift
from .keys import norm_value
from .parse.ledger import unit_key_of
from .shapes import UNSCOPED, classify
from .sync import Parser

# --- statements (note 7.2, 7.3, 7.4, 7.6; columns are always aliased) -----------------------------------------

Q_UNITS = """MATCH (l:Ledger)-[:HAS_UNIT]->(u:Unit)
WHERE coalesce(l.docStatus, '') <> 'archived'
RETURN l.component AS component, l.path AS path, l.syncedAt AS syncedAt, u.key AS key,
       u.unit AS unit, u.owner AS owner, u.statusToken AS status, u.statusText AS statusText,
       u.agentId AS agentId, u.deliverable AS deliverable
ORDER BY path, u.tableIdx, u.rowIdx"""

Q_PAUSED = """MATCH (u:Unit {statusToken: 'paused'})
MATCH (l:Ledger {path: u.ledgerPath})
WHERE coalesce(l.docStatus, '') <> 'archived'
RETURN l.component AS component, l.path AS path, u.key AS key, u.unit AS unit, u.statusText AS statusText,
       u.question AS question, u.askedOn AS askedOn, l.syncedAt AS syncedAt
ORDER BY askedOn"""

Q_REVIEWS = """MATCH (r:Review)
WHERE coalesce(r.docStatus, '') <> 'archived' AND r.verdictClass = 'needs-changes'
OPTIONAL MATCH (l:Ledger)
WHERE r.path IN l.resolvedReviews
   OR (l.docStatus = 'archived' AND r.path IN l.citedReviews)
WITH r, collect(l.path) AS closedBy
WHERE size(closedBy) = 0
RETURN r.component AS component, r.path AS path, r.latestPass AS pass,
       r.latestVerdict AS verdict, r.syncedAt AS syncedAt"""

Q_REQUIREMENTS = """MATCH (q:Requirement)
WHERE q.docStatus IN ['Interviewing', 'Ready for design']
OPTIONAL MATCH (l:Ledger {slug: q.slug})
WITH q, l WHERE l IS NULL
RETURN q.component AS component, q.path AS path, q.docStatus AS status, q.itemId AS itemId,
       q.syncedAt AS syncedAt"""

Q_CARRIERS = """MATCH (s:Source) WHERE s.itemId IN $candidates AND s.kind = 'ledger'
RETURN s.path AS path, s.slug AS slug, s.itemId AS itemId"""

_UNIT_COLUMNS = """RETURN l.path AS path, l.docStatus AS docStatus, l.sha256 AS fileSha, l.syncedAt AS syncedAt,
       u.key AS key, u.unitKey AS unitKey, u.unitLabel AS unitLabel, u.statusToken AS status,
       u.statusText AS statusText, u.agentId AS agentId, u.deliverable AS deliverable,
       u.question AS question, u.askedOn AS askedOn, u.rowHash AS rowHash
ORDER BY path, u.tableIdx, u.rowIdx"""
_OR = "WHERE u.unitKey = $unit OR u.unitLabel = $unit\n"
# 7.3 verbatim (slug anchor, the optional filter), its path-anchored twin, and the 7.6(c) task join.
Q_UNIT_SLUG = "MATCH (l:Ledger {slug: $slug})-[:HAS_UNIT]->(u:Unit)\n" + _OR + _UNIT_COLUMNS
Q_UNIT_SLUG_ALL = "MATCH (l:Ledger {slug: $slug})-[:HAS_UNIT]->(u:Unit)\n" + _UNIT_COLUMNS
Q_UNIT_PATH = "MATCH (l:Source {path: $path})-[:HAS_UNIT]->(u:Unit)\n" + _OR + _UNIT_COLUMNS
Q_UNIT_PATH_ALL = "MATCH (l:Source {path: $path})-[:HAS_UNIT]->(u:Unit)\n" + _UNIT_COLUMNS
_TASK_HEAD = "MATCH (u:Unit) WHERE u.itemId IN $candidates\nMATCH (l:Source {path: u.ledgerPath})\n"
Q_UNIT_TASK = _TASK_HEAD + _UNIT_COLUMNS

Q_BACKLOGS = """MATCH (b:Backlog {scope: $scope})
RETURN b.path AS path, b.syncedAt AS syncedAt, b.rowCount AS rowCount
ORDER BY path"""

Q_PLANNED = """MATCH (b:Backlog {scope: $scope})-[:HAS_ITEM]->(i:Item)
WHERE i.open = true
RETURN b.path AS path, b.syncedAt AS syncedAt, i.ref AS ref, i.statusMarker AS marker,
       i.title AS title, i.priority AS priority, i.milestone AS milestone
ORDER BY path, i.ord"""

Q_SCOPES = "MATCH (b:Backlog) RETURN DISTINCT b.scope AS scope ORDER BY scope"

DRIFT_MARK = "⚠ drift"
_TRUNC = 140


EMPTY_MIRROR_HINT = "(no synced mirror: run `coordgraph bootstrap` and `coordgraph sync`)"


class ViewError(RuntimeError):
    """A view that cannot answer (an empty mirror); the message says what to do."""


def records(result) -> list[dict]:
    """A query result as `{alias: value}` dicts (the plan's columns are always aliased)."""
    names = [h[1] if isinstance(h, (list, tuple)) else h for h in getattr(result, "header", [])]
    return [dict(zip(names, row, strict=True)) for row in result.result_set]


def ro(graph, query: str, params: Mapping | None = None) -> list[dict]:
    """Run one read; `RO_QUERY` only, so a view can never write (graph note 7)."""
    return records(graph.ro_query(query, dict(params or {})))


# --- inflight -------------------------------------------------------------------------------------------------

@dataclass
class Inflight:
    units: list[dict]
    paused: list[dict]
    reviews: list[dict]
    requirements: list[dict]


def inflight(graph, identity) -> Inflight:
    """The four reads of note 7.2. A requirement is also "picked up" by a ledger whose `Item:` is any
    candidate of the requirement's `Item:` (7.6(d)): that read is separate from the slug join, since an `OR`
    would plan a label scan."""
    units = ro(graph, Q_UNITS)
    paused = ro(graph, Q_PAUSED)
    reviews = ro(graph, Q_REVIEWS)
    reqs = ro(graph, Q_REQUIREMENTS)
    cands = {r["path"]: _candidates(identity, r["itemId"], r["component"]) for r in reqs if r.get("itemId")}
    if cands:
        pool = sorted(set().union(*cands.values()))
        carried = {c["itemId"] for c in ro(graph, Q_CARRIERS, {"candidates": pool})}
        reqs = [r for r in reqs if not (cands.get(r["path"], set()) & carried)]
    return Inflight(units, paused, reviews, reqs)


def _candidates(identity, token: str, scope: str) -> set[str]:
    """The requirement's `Item:` is read in its own component's scope: a bare `K-065` denotes that home's
    item, not whichever home's ledger happens to cite a bare `K-065` (plan3 3.1 step 3)."""
    return {norm_value(c) for c in identity.resolve(token, scope).candidates}


# --- unit ---------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class UnitHit:
    """One `:Unit` row of note 7.3, with the attributes `identity.select_units` reads."""

    path: str
    doc_status: str | None
    synced_at: int | None
    key: str
    unit_key: str
    unit_label: str
    status: str
    status_text: str | None = None
    agent_id: str | None = None
    deliverable: str | None = None
    question: str | None = None
    asked_on: str | None = None

    @classmethod
    def of(cls, r: Mapping) -> UnitHit:
        return cls(r["path"], r["docStatus"], r["syncedAt"], r["key"], r["unitKey"], r["unitLabel"], r["status"],
                   r.get("statusText"), r.get("agentId"), r.get("deliverable"), r.get("question"),
                   r.get("askedOn"))


@dataclass
class UnitLookup:
    hits: list[UnitHit]
    notes: list[str] = field(default_factory=list)  # why an ID argument did not resolve (never guessed)


def find_units(graph, identity, arg: str, unit_ref: str | None = None) -> UnitLookup:
    """`unit <slug|ID> [<unit>]`: the slug read (always) united by `key` with the ID reads (feature ledgers,
    task units) when `arg` has an ID shape. The lookup rule is `select_units`: exact `unitKey` wins, else the
    label; a trailing `~n` addresses one repeat, tried on the whole token first (plan3 3.4)."""
    from .identity import select_units, unit_query_key  # noqa: PLC0415 - read side only, deliberately lazy

    cleaned = unit_key_of(unit_ref) if unit_ref else None
    notes: list[str] = []
    cands: list[str] = []
    if classify(arg) is not None:
        res = identity.resolve(arg, UNSCOPED)
        cands = sorted({norm_value(c) for c in res.candidates})
        if res.state != "resolved":
            notes.append(f"{arg}: {res.state}: {res.reason}")
    slug = norm_value(arg)
    carriers = [c["path"] for c in ro(graph, Q_CARRIERS, {"candidates": cands})] if cands else []

    def gather(unit_param: str | None) -> list[UnitHit]:
        hits: dict[str, UnitHit] = {}

        def add(rows: Iterable[Mapping]) -> None:
            for r in rows:
                hits.setdefault(r["key"], UnitHit.of(r))

        if unit_param is None:
            add(ro(graph, Q_UNIT_SLUG_ALL, {"slug": slug}))
        else:
            add(ro(graph, Q_UNIT_SLUG, {"slug": slug, "unit": norm_value(unit_param)}))
        for path in carriers:
            if unit_param is None:
                add(ro(graph, Q_UNIT_PATH_ALL, {"path": path}))
            else:
                add(ro(graph, Q_UNIT_PATH, {"path": path, "unit": norm_value(unit_param)}))
        if cands:
            add(ro(graph, Q_UNIT_TASK, {"candidates": cands}))
        return sorted(hits.values(), key=lambda h: h.path)  # stable: table order survives within a ledger

    if cleaned is None:
        return UnitLookup(gather(None), notes)
    hits = select_units(gather(cleaned), cleaned)
    stripped = unit_query_key(cleaned)
    if not hits and stripped != cleaned:
        hits = select_units(gather(stripped), cleaned)
    return UnitLookup(hits, notes)


# --- planned ------------------------------------------------------------------------------------------------

@dataclass
class Planned:
    sources: list[dict]  # every backlog with this scope: path, syncedAt, rowCount
    planned: list[dict]
    parked: list[dict]  # deferred / rejected: a record, not work (P2-m1)
    scopes: list[str] = field(default_factory=list)  # filled only when nothing matched, to say what exists


def _parked(item: Mapping) -> bool:
    marker = (item.get("marker") or "").casefold()
    return "⚪" in marker or "deferred" in marker or "rejected" in marker


def planned(graph, scope: str) -> Planned:
    scope = norm_value(scope)
    sources = ro(graph, Q_BACKLOGS, {"scope": scope})
    if not sources:
        return Planned([], [], [], [r["scope"] for r in ro(graph, Q_SCOPES) if r["scope"]])
    items = ro(graph, Q_PLANNED, {"scope": scope})
    return Planned(sources, [i for i in items if not _parked(i)], [i for i in items if _parked(i)])


# --- rendering ----------------------------------------------------------------------------------------------

def _clip(text: str | None, limit: int = _TRUNC) -> str:
    flat = " ".join((text or "").split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _same_status(raw: str | None, token: str) -> bool:
    """True when the cell adds nothing to the token (`**accepted**`); any other cell text is shown beside it."""
    return not raw or raw.replace("*", "").strip().casefold() == token


def _flag(flagged: set[tuple[str, str]], path: str, key: str) -> str:
    return f"\t{DRIFT_MARK}" if (path, key) in flagged else ""


def _section(title: str, rows: list[str]) -> list[str]:
    return [f"{title} ({len(rows)})"] + (rows or ["  (none)"])


def render_inflight(view: Inflight, flagged: set[tuple[str, str]], waiting: bool = False) -> str:
    paused = []
    for r in view.paused:
        asked = f"asked {r['askedOn']}" if r.get("askedOn") else "asked unknown"
        what = _clip(r.get("question") or r.get("statusText"), 400)
        paused.append(f"  {r['component']}\t{r['path']}\t{r['unit']}\t{asked}\t{what}"
                      + _flag(flagged, r["path"], r["key"]))
    out = _section("Waiting on the stakeholder", paused)
    if waiting:
        return "\n".join(out)
    units: list[str] = []
    last = None
    for r in view.units:
        if r["path"] != last:
            last = r["path"]
            units.append(f"  {r['component']}\t{r['path']}")
        raw = r.get("statusText")
        shown = r["status"] if _same_status(raw, r["status"]) else f"{r['status']} [{_clip(raw, 60)}]"
        units.append(f"    {r['unit']}\t{r.get('owner') or '-'}\t{shown}\t{r.get('agentId') or '-'}\t"
                     f"{_clip(r.get('deliverable'))}" + _flag(flagged, r["path"], r["key"]))
    reviews = [f"  {r['component']}\t{r['path']}\tpass {r.get('pass') or '?'}\t{_clip(r.get('verdict'))}"
               for r in view.reviews]
    reqs = [f"  {r['component']}\t{r['path']}\t{r['status']}" for r in view.requirements]
    return "\n".join(["Units in open coordinations"] + (units or ["  (none)"]) + [""] + out + [""]
                     + _section('Open "needs changes" verdicts', reviews) + [""]
                     + _section("Requirements not yet picked up", reqs))


def render_unit(arg: str, unit_ref: str | None, lookup: UnitLookup, flagged: set[tuple[str, str]]) -> str:
    out = list(lookup.notes)
    if not lookup.hits:
        what = f"{arg} {unit_ref}" if unit_ref else arg
        return "\n".join(out + [f"no unit matches {what}"])
    last = None
    for h in lookup.hits:
        if h.path != last:
            last = h.path
            state = f" · docStatus: {h.doc_status}" if h.doc_status else ""
            out.append(f"{h.path}{state}")
        out.append(f"  {h.unit_key}\t{h.key}" + _flag(flagged, h.path, h.key))
        out.append(f"    status: {h.status}" + ("" if _same_status(h.status_text, h.status)
                                               else f" [{_clip(h.status_text, 200)}]"))
        if h.doc_status == "archived":
            out.append("    (archived ledger: only key, label and status are kept)")
            continue
        out.append(f"    agent: {h.agent_id or '-'}")
        out.append(f"    deliverable: {_clip(h.deliverable, 600) or '-'}")
        if h.question or h.status == "paused":
            asked = h.asked_on or "unknown"
            out.append(f"    question: {_clip(h.question, 600) or '-'} (asked {asked})")
    if len(lookup.hits) > 1:
        out.append(f"{len(lookup.hits)} rows match; cite one by its key or by task ID")
    return "\n".join(out)


def render_planned(result: Planned, scope: str, flagged: set[tuple[str, str]], show_all: bool = False) -> str:
    if not result.sources:
        known = ", ".join(result.scopes) or "none"
        return f"no backlog with scope {scope!r}; known scopes: {known}"
    out = [f"Backlog {s['path']}\t({s.get('rowCount', 0)} items)" for s in result.sources]
    lines = [f"  {i['ref']}\t{i.get('marker') or '-'}\t{i.get('priority') or '-'}\t{i.get('milestone') or '-'}\t"
             f"{i['title']}\t{i['path']}" + _flag(flagged, i["path"], i["ref"]) for i in result.planned]
    out += _section("Planned (proposed / in progress)", lines)
    if show_all:
        parked = [f"  {i['ref']}\t{i.get('marker') or '-'}\t{i['title']}\t{i['path']}"
                  + _flag(flagged, i["path"], i["ref"]) for i in result.parked]
        out += [""] + _section("Deferred / rejected (a record, not planned work)", parked)
    else:
        ids = ", ".join(i["ref"] for i in result.parked) or "none"
        out += ["", f"Deferred / rejected: {len(result.parked)} ({ids}); --all lists them in full"]
    return "\n".join(out)


# --- commands -----------------------------------------------------------------------------------------------

def _header(cfg, root: Path, graph, identity) -> tuple[str, set[tuple[str, str]]]:
    report = drift.check(graph, Parser.for_root(root))
    return (drift.render_header(report, commands_store._hook_active(cfg), identity.registry_line()),
            report.flagged_rows())


def _explain(exc: Exception) -> ViewError | Exception:
    """An unsynced graph key answers `Invalid graph operation on empty key`: add what to do about it."""
    if "empty key" in str(exc).casefold():
        return ViewError(f"{exc} {EMPTY_MIRROR_HINT}")
    return exc


def _run(name: str, args, body) -> int:
    """Shared frame: connect (ephemeral only), load identity, run `body(graph, identity)`, print the header
    first. An absent mirror or a refused endpoint is one clear line, never a traceback."""
    from .identity import load_identity  # noqa: PLC0415 - read side only, deliberately lazy

    try:
        cfg, _key, root, _db, graph = commands_store._target(args)
        identity = load_identity(root)
        answer = body(graph, identity)
        header, flagged = _header(cfg, root, graph, identity)
    except Exception as exc:  # noqa: BLE001 - a failure is one line, never a traceback
        print(f"{name}: {_explain(exc)}", file=sys.stderr)
        return 1
    text, code = answer(flagged)
    print(header + "\n")
    print(text)
    return code


def inflight_cmd(args) -> int:
    def body(graph, identity):
        view = inflight(graph, identity)
        return lambda flagged: (render_inflight(view, flagged, getattr(args, "waiting", False)), 0)

    return _run("inflight", args, body)


def unit_cmd(args) -> int:
    def body(graph, identity):
        lookup = find_units(graph, identity, args.ref, getattr(args, "unit", None))
        return lambda flagged: (render_unit(args.ref, getattr(args, "unit", None), lookup, flagged),
                                0 if lookup.hits else 1)

    return _run("unit", args, body)


def planned_cmd(args) -> int:
    def body(graph, identity):
        result = planned(graph, args.scope)
        return lambda flagged: (render_planned(result, args.scope, flagged, getattr(args, "all", False)),
                                0 if result.sources else 1)

    return _run("planned", args, body)


COMMANDS = {"inflight": inflight_cmd, "unit": unit_cmd, "planned": planned_cmd}


def register(sub) -> None:
    """Add the three subcommands to the CLI's subparsers (each takes `--graph`, like every store-backed one)."""
    inf = sub.add_parser("inflight", help="open coordinations, questions waiting, verdicts, requirements")
    inf.add_argument("--waiting", action="store_true", help="only the questions waiting on the stakeholder")
    un = sub.add_parser("unit", help="one unit's status, agent, deliverable, question; by slug or ID")
    un.add_argument("ref", metavar="SLUG_OR_ID")
    un.add_argument("unit", nargs="?", metavar="UNIT", help="unit key or label (U3, U32~2); omit for every unit")
    pl = sub.add_parser("planned", help="open backlog items of one component or agent")
    pl.add_argument("scope", metavar="COMPONENT_OR_AGENT")
    pl.add_argument("--all", action="store_true", help="list deferred/rejected records in full")
    for p in (inf, un, pl):
        p.add_argument("--graph", help="graph key (default: COORDGRAPH_KEY)")
