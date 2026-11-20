"""Work-item identity, resolved at read time (plan3 sections 3.1 and 3.4).

Every ID in a source file is stored as written; this module decides what a stored string
*denotes*. It is the **read side**: it may import `wilib` (lazily), the sync/hook path never
does and never imports this module, so a source's subgraph stays a pure function of its own
file and a bad `COORDGRAPH_WI_ROOT` cannot break freshness.

`Identity.resolve` maps a token to a `Resolution` whose `candidates` are the strings under
which that identity may be stored; two references denote one item iff their candidate sets
intersect, and joins are `IN $candidates` on indexed raw strings. `Identity.candidates` has
the `shapes.Candidates` signature, so `lint` takes it unchanged.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone

from .lint import default_wi_root
from .shapes import UNSCOPED, classify, qualify

_UNIT_REF = re.compile(r"[a-z0-9][a-z0-9-]*:\S+")


def load_wilib():
    """Import `wilib` from `COORDGRAPH_WI_ROOT` (default: the repo's `skills/work-items/scripts`)."""
    root = str(default_wi_root())
    if root not in sys.path:
        sys.path.append(root)
    import wilib  # noqa: PLC0415 - read side only, deliberately lazy

    return wilib


def _resolve_error():
    load_wilib()
    from wilib.resolve import ResolveError  # noqa: PLC0415

    return ResolveError


@dataclass(frozen=True)
class Resolution:
    """state: resolved | unminted | ambiguous | unknown | retired | unavailable.

    `route` is the shape that picked the resolver (`legacy-bare`, `legacy-qualified`,
    `scheme`, `unit`) or None for an unrecognised token; `type` is the item type letter of a
    resolved item; `reason` names why a token did not resolve.
    """

    token: str
    state: str
    route: str | None = None
    uid: str | None = None
    type: str | None = None
    candidates: frozenset[str] = frozenset()
    reason: str | None = None


class Registry:
    """A `wilib` registry state (or its absence) plus the declared legacy namespaces."""

    def __init__(self, state, namespaces, unflushed: int = 0):
        self.state, self.namespaces, self.unflushed = state, namespaces, unflushed

    def lookup(self, name: str) -> dict:
        """`wilib.resolve.resolve` over the state; raises `ResolveError` (`.code` 3/4/5)."""
        load_wilib()
        from wilib import legacy, resolve  # noqa: PLC0415

        legacy.install(self.namespaces)
        return resolve.resolve(self.state, name)

    def legacy_name(self, uid: str) -> str | None:
        item = self.state.items.get(uid)
        leg = item.legacy if item is not None else None
        return f"{leg['home']}:{leg['name']}" if leg else None


def load_identity(repo_root) -> Identity:
    """An `Identity` over the repo's registry; degraded (`Registry: unavailable`) when `wilib` is not
    importable, the repo has no `.work-items/`, or the registry cannot be read. Never raises."""
    try:
        load_wilib()
        from wilib import alloc, project  # noqa: PLC0415
        from wilib.store import Store  # noqa: PLC0415

        namespaces = project.load_namespaces(repo_root)
        store = Store(repo_root, alloc.load_key(repo_root))
        state = store.read_state()
        tracked = {e["evid"] for e in store.read_tracked()}
        unflushed = sum(1 for e in store.read_journal() if e["evid"] not in tracked)
    except Exception as exc:  # noqa: BLE001 - any failure to read the registry degrades, visibly
        return Identity(None, why=f"{type(exc).__name__}: {exc}")
    return Identity(Registry(state, namespaces, unflushed))


def _unrecognised(token: str) -> Resolution:
    return Resolution(token, "unknown", None, candidates=frozenset({token}), reason="unrecognised token")


class Identity:
    def __init__(self, registry: Registry | None, why: str | None = None):
        self.registry, self.why = registry, why
        self._cache: dict[tuple[str, str, bool], Resolution] = {}

    def resolve(self, token: str, scope: str, needs: bool = False) -> Resolution:
        key = (token, scope, needs)
        if key not in self._cache:
            self._cache[key] = self._resolve(token, scope, needs)
        return self._cache[key]

    def reason(self, token: str, scope: str) -> str | None:
        """Why `token` is not a resolved identity (None when it is): the lint seam's second half."""
        res = self.resolve(token, scope)
        return None if res.state == "resolved" else res.reason

    def registry_line(self) -> str:
        """The sync-header line: the registry's size and freshness, or that it is unavailable."""
        reg = self.registry
        if reg is None:
            return "Registry: unavailable — identities compared as written"
        last = reg.state.last_ts
        day = datetime.fromtimestamp(last / 1e9, timezone.utc).strftime("%Y-%m-%d") if last else "none"
        return (f"Registry: {len(reg.state.items)} items · last event {day} · "
                f"{reg.unflushed} unflushed journal events")

    def candidates(self, token: str, scope: str) -> set[str]:
        """The `shapes.Candidates` callable for `lint`."""
        return set(self.resolve(token, scope).candidates)

    def _resolve(self, token: str, scope: str, needs: bool) -> Resolution:
        route = classify(token)
        if route is None and needs and _UNIT_REF.fullmatch(token):
            return Resolution(token, "resolved", "unit", candidates=frozenset({token}))
        if route is None:
            return _unrecognised(token)
        name = token
        if route == "legacy-bare" and scope and scope != UNSCOPED:
            name = qualify(token, scope)
        return self._through_registry(token, name, route)

    def _through_registry(self, token: str, name: str, route: str) -> Resolution:
        # A bare token the referrer's scope qualified is denoted by the qualified name alone: the bare
        # string would also match another home's `K-065` (plan3 3.1 step 3).
        as_written = frozenset({name})
        if self.registry is None:
            return Resolution(token, "unavailable", route, candidates=as_written,
                              reason=f"registry unavailable ({self.why or 'not loaded'}): "
                                     "identities compared as written")
        try:
            rec = self.registry.lookup(name)
        except _resolve_error() as exc:
            if exc.code == 5:
                return Resolution(token, "unminted", route, candidates=frozenset({name}), reason=str(exc))
            state = "ambiguous" if exc.code == 3 else "unknown"
            return Resolution(token, state, route, candidates=as_written, reason=str(exc))
        cands = {rec["id"], rec["uid"], name, *rec["aliases"]}
        if (legacy := self.registry.legacy_name(rec["uid"])) is not None:
            cands.add(legacy)
        state = "retired" if rec["retired"] else "resolved"
        return Resolution(token, state, route, uid=rec["uid"], type=rec["type"],
                          candidates=frozenset(cands))


@dataclass(frozen=True)
class LedgerRef:
    """The ledger a `Needs` cell is written in: its path, per-kind slug and backlog-scope stand-in."""

    path: str
    slug: str
    scope: str = UNSCOPED


@dataclass(frozen=True)
class UnitRow:
    """One fetched `:Unit` (graph note 7.3/7.6(c)); an archived ledger's unit has no `status_text`."""

    key: str
    ledger_path: str
    slug: str
    unit_key: str
    unit_label: str
    status: str
    item_id: str | None = None
    status_text: str | None = None


@dataclass(frozen=True)
class LedgerRow:
    """One fetched ledger of graph note 7.6(b): `units`/`accepted` count `:Unit`s and `accepted` ones."""

    path: str
    slug: str
    doc_status: str | None
    units: int
    accepted: int
    component: str | None = None
    item_id: str | None = None


@dataclass(frozen=True)
class NeedResult:
    state: str  # met | unmet | unresolvable
    reason: str | None = None
    matches: tuple[str, ...] = ()


def resolve_need(token, ledger: LedgerRef, identity: Identity, graph, read_cell=None) -> NeedResult:
    """Is one `Needs` token met, unmet or unresolvable (plan3 section 3.4)? A need never copies status.

    `graph` answers the reads of graph note 7.3/7.6: `ledgers_carrying(candidates)`,
    `units_with_item(candidates)`, `units_matching(slug, unit)`, `units_in_ledger(path, unit)`.
    `read_cell(unit_row)` fetches an archived unit's raw status cell from its file, or None.
    """
    res = identity.resolve(token, ledger.scope, needs=True)
    if res.route == "unit":
        slug, _, unit_ref = token.partition(":")
        rows = graph.units_matching(slug, unit_query_key(unit_ref))
        return _unit_need(token, select_units(rows, unit_ref), unit_ref, read_cell)
    if res.route is None:
        rows = select_units(graph.units_in_ledger(ledger.path, unit_query_key(token)), token)
        if rows:
            return _unit_need(token, rows, token, read_cell)
        return NeedResult("unresolvable", res.reason)
    if res.state != "resolved":
        return NeedResult("unresolvable", _unresolved_reason(res))
    cands = set(res.candidates)
    if res.type == "F":
        return _feature_need(token, graph.ledgers_carrying(cands))
    return _task_need(token, graph.units_with_item(cands), read_cell)


_REPEAT = re.compile(r"~\d+$")


def unit_query_key(unit_ref: str) -> str:
    """What the one `unitKey = $u OR unitLabel = $u` query is asked for: neither property carries a
    `~n` repeat suffix, so a trailing `~<digits>` (anchored: a real key can hold `~80 tool calls)`) goes."""
    return _REPEAT.sub("", unit_ref)


def select_units(rows, unit_ref: str):
    """The one lookup rule shared by `resolve_need` and the `unit` view (plan3 section 3.4 `Unit label`),
    pure over rows with `key`, `unit_key`, `unit_label`: a trailing `~n` addresses one repeat by its
    `key`; otherwise exact-`unitKey` rows win (every repeat of that key), and only when none exists do
    the `unitLabel` matches count. A key that itself ends in `~n` matches exactly first."""
    rows = list(rows)
    exact = [u for u in rows if u.unit_key == unit_ref]
    if exact:
        return exact
    if unit_query_key(unit_ref) != unit_ref:
        return [u for u in rows if u.key.endswith("#" + unit_ref)]
    return [u for u in rows if u.unit_label == unit_ref]


def _unresolved_reason(res: Resolution) -> str:
    return f"{res.state}: {res.reason}"


def _feature_need(token, ledgers) -> NeedResult:
    if not ledgers:
        return NeedResult("unresolvable", f"no ledger carries {token}")
    paths = tuple(l.path for l in ledgers)
    open_ = [l.path for l in ledgers
             if l.doc_status != "archived" and not (l.units >= 1 and l.accepted == l.units)]
    if open_:
        return NeedResult("unmet", f"{token}: not finished in " + ", ".join(open_), paths)
    return NeedResult("met", None, paths)


def _task_need(token, units, read_cell) -> NeedResult:
    return _unit_need(token, units, token, read_cell, what="unit carries")


def _unit_need(token, units, shown, read_cell, what="unit") -> NeedResult:
    if what == "unit carries" and not units:
        return NeedResult("unresolvable", f"no unit carries {token}")
    if what == "unit" and not units:
        return NeedResult("unresolvable", f"no unit {shown}")
    keys = tuple(u.key for u in units)
    if len(units) > 1:
        return NeedResult("unresolvable", "ambiguous: " + ", ".join(keys), keys)
    (u,) = units
    if u.status == "accepted":
        return NeedResult("met", None, keys)
    if u.status == "unknown":
        cell = u.status_text if u.status_text is not None else (read_cell(u) if read_cell else None)
        where = f"{u.ledger_path}, {cell}" if cell is not None else u.ledger_path
        return NeedResult("unresolvable", f"status unknown ({where})", keys)
    return NeedResult("unmet", f"{token}: {u.status}", keys)


@dataclass(frozen=True)
class NeedsStatus:
    """state: ready | blocked | unresolvable | not-recorded; `reasons` say what is missing."""

    state: str
    reasons: tuple[str, ...] = ()
    results: tuple[NeedResult, ...] = ()


def needs_status(needs, needs_error, ledger: LedgerRef, identity: Identity, graph, read_cell=None) -> NeedsStatus:
    """A unit's `Needs` cell as a whole (AC-3, AC-4): absent or malformed is never ready; any
    certainly-unmet need blocks; otherwise any unresolvable need makes the unit unresolvable."""
    if needs_error:
        return NeedsStatus("not-recorded", (f"needs not recorded: {needs_error}",))
    if needs is None:
        return NeedsStatus("not-recorded", ("needs not recorded",))
    results = tuple(resolve_need(t, ledger, identity, graph, read_cell) for t in needs)
    reasons = tuple(f"{t}: {r.reason}" for t, r in zip(needs, results, strict=True) if r.state != "met")
    states = {r.state for r in results}
    state = "blocked" if "unmet" in states else "unresolvable" if "unresolvable" in states else "ready"
    return NeedsStatus(state, reasons, results)
