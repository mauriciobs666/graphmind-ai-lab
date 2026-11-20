"""The milestone proposal (plan section 3.7, FR-13 to FR-15): a pure, deterministic function.

`propose(items, resolver, per_milestone)` takes the open items of one backlog (`parse.backlog.Item`
shaped: only the recorded facts are read) and a `resolver` for dependency tokens, and returns a
`Proposal`. No store, no `wilib`, no network, no clock, no file read: whatever needs one of those is
answered by the caller's `resolver(token, referrer_scope) -> DepResolution`, so the same input is always
the same output (AC-11) and nothing is ever written (AC-14).

Steps follow the plan: 0 fixed items and insertion, 1 classify, 2 resolve dependencies, 3 rank,
4 waves, 5 milestones, 6 one explaining criterion per placement. Where the plan is silent the choice
is stated at the function that makes it.
"""
from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from fnmatch import fnmatchcase

PRIORITY = {"high": 3, "medium": 2, "low": 1}
_MILESTONE = re.compile(r"M(\d+)(?:\.(\d+))?")
_AHEAD = re.compile(r"ahead-of-M(\d+)(?:\.(\d+))?")
_GLOB = "*?["
_FACT = re.compile(r"(?P<token>.*?)\s*\((?P<mode>changes|relies)\)\s*$", re.I)


@dataclass(frozen=True)
class DepResolution:
    """What the caller knows of one `Depends on` token.

    `candidates`: the strings its identity may be stored under (an item of the backlog is the target
    when its `ref` is among them); `external`: `(ref, backlog path)` when it is open in another backlog;
    `delivered`: the history file holding a delivery-shaped heading; `reason`: why it is unknown.
    """

    candidates: frozenset = frozenset()
    external: tuple | None = None
    delivered: str | None = None
    reason: str | None = None


Resolver = Callable[[str, str], DepResolution]


@dataclass(frozen=True)
class Placement:
    ref: str
    title: str
    path: str
    wave: int | None = None  # None: a fixed item, placed as recorded
    reason: str | None = None  # the one criterion; None for a fixed item
    inserted_ahead_of: str | None = None
    impossible: tuple = ()  # causes, for a fixed item whose recorded placement can no longer hold


@dataclass
class MilestonePlan:
    name: str
    kind: str  # fixed | inserted | proposed
    items: list[Placement] = field(default_factory=list)


@dataclass(frozen=True)
class Entry:
    ref: str
    title: str
    path: str


@dataclass(frozen=True)
class Held:
    ref: str
    title: str
    path: str
    reason: str
    blocker_path: str | None = None


@dataclass(frozen=True)
class NotPlannable:
    ref: str
    title: str
    path: str
    missing: tuple = ()
    unknown: tuple = ()
    unknown_reasons: tuple = ()
    cycle: tuple = ()

    @property
    def text(self) -> str:
        parts = []
        if self.missing:
            parts.append("missing " + ", ".join(self.missing))
        for tok, why in zip(self.unknown, self.unknown_reasons, strict=True):
            parts.append(f"unknown dependency {tok} ({why}) — if it was delivered, drop the ref or add a "
                         "`<ID> delivered` history heading; otherwise fix the id")
        if self.cycle:
            parts.append("in dependency cycle " + ", ".join(self.cycle))
        return "; ".join(parts)


@dataclass(frozen=True)
class Impossible:
    ref: str
    title: str
    path: str
    causes: tuple


@dataclass
class Proposal:
    milestones: list[MilestonePlan] = field(default_factory=list)
    carried: list[Entry] = field(default_factory=list)
    held: list[Held] = field(default_factory=list)
    deferred: list[Entry] = field(default_factory=list)
    not_plannable: list[NotPlannable] = field(default_factory=list)
    impossible: list[Impossible] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def render(self) -> str:
        out: list[str] = []
        for m in self.milestones:
            note = {"fixed": "approved, as recorded", "inserted": "inserted", "proposed": "proposed"}[m.kind]
            out.append(f"Milestone {m.name} ({note})")
            last_wave = object()
            for pl in m.items:
                if pl.wave != last_wave and pl.wave is not None:
                    out.append(f"  wave {pl.wave}")
                last_wave = pl.wave
                line = f"    {pl.ref}  {pl.title}  [{pl.path}]"
                if pl.inserted_ahead_of:
                    line += f"  ⤴ inserted ahead of approved {pl.inserted_ahead_of}"
                if pl.reason:
                    line += f"  — {pl.reason}"
                out.append(line)
                for cause in pl.impossible:
                    out.append(f"      ⚠ approved placement now impossible: {cause}")
        if self.impossible:
            out.append("Approved placements now impossible (a proposed revision; nothing was changed):")
            out += [f"  {e.ref}  {e.title}  [{e.path}]: " + "; ".join(e.causes) for e in self.impossible]
        sections = (
            ("Carried — in progress", self.carried, lambda e: ""),
            ("Held", self.held, lambda e: f" — {e.reason}"),
            ("Deferred — not proposed", self.deferred, lambda e: ""),
            ("Not plannable yet", self.not_plannable, lambda e: f" — {e.text}"),
        )
        for title, rows, tail in sections:
            if rows:
                out.append(f"{title}:")
                out += [f"  {e.ref}  {e.title}  [{e.path}]{tail(e)}" for e in rows]
        out += [f"warning: {w}" for w in self.warnings]
        return "\n".join(out)


# --- overlap (step 4) ------------------------------------------------------------------------------

def _file_covers(general: str, other: str) -> bool:
    """`general` (a path, directory or glob) covers the literal path `other`."""
    if any(c in general for c in _GLOB):
        return fnmatchcase(other, general)
    return other == general or other.startswith(general.rstrip("/") + "/")


def _prefix(glob: str) -> str:
    cut = min((glob.find(c) for c in _GLOB if c in glob), default=len(glob))
    return glob[:cut]


def _file_overlap(a: str, b: str) -> str | None:
    """The covering entry when two `file:` entries overlap, else None. Glob vs glob is conservative
    (m7): the literal prefix of one a prefix of the other's, because `fnmatch` of one glob against
    another is not an intersection test."""
    a_glob, b_glob = any(c in a for c in _GLOB), any(c in b for c in _GLOB)
    if a_glob and b_glob:
        return a if _prefix(a).startswith(_prefix(b)) or _prefix(b).startswith(_prefix(a)) else None
    if _file_covers(a, b):
        return a
    if _file_covers(b, a):
        return b
    if a_glob or b_glob:  # a directory against a glob: the glob lies under it
        glob, lit = (a, b) if a_glob else (b, a)
        if _prefix(glob).startswith(lit.rstrip("/") + "/"):
            return lit
    return None


def _fact(body: str) -> tuple[str, str]:
    m = _FACT.fullmatch(body.strip())
    if m:
        return m.group("token").strip().casefold(), m.group("mode").casefold()
    return body.strip().casefold(), "changes"


def touch_overlap(a: Sequence[str], b: Sequence[str]) -> tuple[str, str] | None:
    """`(kind, value)` of the first overlap between two `Touches` lists, else None (FR-13(b)).

    `file:` equal paths or a covering directory/glob; `graph:` the same key; `fact:` the same
    lower-cased token unless both sides are `(relies)`. `value` is the covering entry for a file."""
    for x in a:
        for y in b:
            for kind in ("file", "graph", "fact"):
                if not (x.startswith(kind + ":") and y.startswith(kind + ":")):
                    continue
                xv, yv = x[len(kind) + 1:].strip(), y[len(kind) + 1:].strip()
                if kind == "file" and (hit := _file_overlap(xv, yv)) is not None:
                    return kind, hit
                if kind == "graph" and xv == yv:
                    return kind, xv
                if kind == "fact":
                    (xt, xm), (yt, ym) = _fact(xv), _fact(yv)
                    if xt == yt and not (xm == "relies" and ym == "relies"):
                        return kind, xt
    return None


# --- the function ----------------------------------------------------------------------------------

def _pr(item) -> int:
    return PRIORITY.get(item.priority, 0)


def _milestone_key(name: str, appearance: dict[str, int]) -> tuple:
    """Fixed milestones: `M<n>[.<m>]` numerically; `ahead-of-Mx` immediately before `Mx` (P3-n1); any other
    token by first appearance, after the numbered ones."""
    if m := _MILESTONE.fullmatch(name):
        return 0, int(m.group(1)), int(m.group(2) or 0), 1
    if m := _AHEAD.fullmatch(name):
        return 0, int(m.group(1)), int(m.group(2) or 0), 0
    return 1, appearance[name], 0, 0


@dataclass
class _Dep:
    token: str
    kind: str  # internal | external | delivered | unknown
    target: object = None  # internal: the Item
    external: tuple | None = None
    reason: str | None = None


def propose(items: Iterable, resolver: Resolver, per_milestone: int = 6) -> Proposal:
    if per_milestone < 1:
        raise ValueError("per_milestone must be at least 1")
    return _Planner(list(items), resolver, per_milestone).run()


class _Planner:
    def __init__(self, items, resolver, n):
        self.work = [i for i in items if i.open]
        self.resolver, self.n = resolver, n
        self.by_ref = {i.ref: i for i in self.work}
        self.out = Proposal()
        self._res: dict[tuple[str, str], DepResolution] = {}
        self.deps: dict[str, list[_Dep]] = {i.ref: self._resolve_all(i) for i in self.work}

    # -- step 2 plumbing
    def _resolve_all(self, item) -> list[_Dep]:
        out = []
        for tok in item.depends_on or []:
            key = (tok, item.scope or "")
            res = self._res.setdefault(key, self.resolver(*key))
            target = next((w for w in self.work if w.ref in res.candidates), None)
            if target is not None:
                out.append(_Dep(tok, "internal", target))
            elif res.external:
                out.append(_Dep(tok, "external", external=tuple(res.external)))
            elif res.delivered:
                out.append(_Dep(tok, "delivered"))
            else:
                out.append(_Dep(tok, "unknown", reason=res.reason or "not in any open backlog"))
        return out

    def run(self) -> Proposal:
        work, out = self.work, self.out
        self.fixed = [i for i in work if i.milestone]
        free = [i for i in work if not i.milestone]
        # step 1
        self.deferred = [i for i in free if i.status in ("deferred", "rejected")]
        self.carried = [i for i in free if i.status == "in-progress"]
        out.deferred = [self._entry(i) for i in self.deferred]
        out.carried = [self._entry(i) for i in self.carried]
        cands = [i for i in free if i not in self.deferred and i not in self.carried]
        self.not_plannable: dict[str, NotPlannable] = {}
        plannable = []
        for i in cands:
            missing = tuple(label for label, value in (("Priority", i.priority), ("Depends on", i.depends_on),
                                                       ("Touches", i.touches)) if value is None)
            unknown = [d for d in self.deps[i.ref] if d.kind == "unknown"]
            if missing or unknown:
                self.not_plannable[i.ref] = NotPlannable(
                    i.ref, i.title, i.backlog_path, missing, tuple(d.token for d in unknown),
                    tuple(d.reason for d in unknown))
            else:
                plannable.append(i)
        # cycles (step 2): over plannable and fixed items
        self.cycles = self._cycles(plannable + self.fixed)
        for i in plannable:
            if i.ref in self.cycles:
                self.not_plannable[i.ref] = NotPlannable(i.ref, i.title, i.backlog_path, cycle=self.cycles[i.ref])
        plannable = [i for i in plannable if i.ref not in self.cycles]
        # holds (step 2)
        plannable, held = self._holds(plannable)
        self.plannable = plannable
        out.not_plannable = [self.not_plannable[i.ref] for i in sorted(
            (self.by_ref[r] for r in self.not_plannable), key=lambda x: x.ord)]
        out.held = held
        self._rank()
        self._place()
        return out

    @staticmethod
    def _entry(i) -> Entry:
        return Entry(i.ref, i.title, i.backlog_path)

    # -- cycles
    def _cycles(self, nodes) -> dict[str, tuple]:
        """ref -> member refs of its dependency cycle (Tarjan; a self-dependency counts)."""
        members = {i.ref for i in nodes}
        edges = {i.ref: [d.target.ref for d in self.deps[i.ref] if d.kind == "internal" and d.target.ref in members]
                 for i in nodes}
        index, low, onstack, stack, counter = {}, {}, set(), [], [0]
        found: dict[str, tuple] = {}

        def visit(v):
            index[v] = low[v] = counter[0]
            counter[0] += 1
            stack.append(v)
            onstack.add(v)
            for w in edges[v]:
                if w not in index:
                    visit(w)
                    low[v] = min(low[v], low[w])
                elif w in onstack:
                    low[v] = min(low[v], index[w])
            if low[v] == index[v]:
                comp = []
                while True:
                    w = stack.pop()
                    onstack.discard(w)
                    comp.append(w)
                    if w == v:
                        break
                if len(comp) > 1 or v in edges[v]:
                    cyc = tuple(sorted(comp, key=lambda r: self.by_ref[r].ord))
                    for r in comp:
                        found[r] = cyc

        for i in nodes:
            if i.ref not in index:
                visit(i.ref)
        return found

    # -- holds
    def _holds(self, plannable):
        held: dict[str, Held] = {}
        live = {i.ref for i in plannable}
        changed = True
        while changed:
            changed = False
            for i in plannable:
                if i.ref in held:
                    continue
                reason = self._hold_reason(i, live, held)
                if reason is not None:
                    held[i.ref] = Held(i.ref, i.title, i.backlog_path, reason[0], reason[1])
                    live.discard(i.ref)
                    changed = True
        kept = [i for i in plannable if i.ref in live]
        return kept, [held[i.ref] for i in plannable if i.ref in held]

    def _hold_reason(self, item, live, held):
        for d in self.deps[item.ref]:
            if d.kind == "external":
                return f"external blocker {d.external[0]} in `{d.external[1]}`", d.external[1]
            if d.kind != "internal":
                continue
            t = d.target
            if t in self.fixed or t in self.carried or t.ref in live:
                continue
            why = ("deferred" if t in self.deferred else "held" if t.ref in held else "not plannable yet")
            return f"waits on {t.ref} ({why})", None
        return None

    # -- step 3
    def _rank(self):
        sset = {i.ref for i in self.plannable} | {i.ref for i in self.fixed}
        self.dependents: dict[str, list[str]] = {r: [] for r in sset}
        for r in sset:
            for d in self.deps[r]:
                if d.kind == "internal" and d.target.ref in sset:
                    self.dependents[d.target.ref].append(r)

        def closure(r):
            seen, todo = [], list(self.dependents[r])
            while todo:
                x = todo.pop(0)
                if x not in seen:
                    seen.append(x)
                    todo += self.dependents[x]
            return seen

        self.trans = {i.ref: closure(i.ref) for i in self.plannable}
        self.eff = {i.ref: max([_pr(i)] + [_pr(self.by_ref[r]) for r in self.trans[i.ref]]) for i in self.plannable}
        self.rank_key = {i.ref: (-self.eff[i.ref], -len(self.trans[i.ref]), -_pr(i), i.ord) for i in self.plannable}

    def _ranked(self, refs) -> list:
        return sorted((self.by_ref[r] for r in refs), key=lambda i: self.rank_key[i.ref])

    # -- steps 0 (insertion), 4, 5, 6
    def _place(self):
        appearance: dict[str, int] = {}
        for i in self.fixed:
            appearance.setdefault(i.milestone, len(appearance))
        self.appearance = appearance
        self.mkey = lambda name: _milestone_key(name, appearance)
        fixed_names = list(appearance)
        for name in fixed_names:
            if not (_MILESTONE.fullmatch(name) or _AHEAD.fullmatch(name)):
                self.out.warnings.append(f"fixed milestone {name!r} is not an M<n> token; ordered by first appearance")
        self.inserted: dict[str, str] = {}  # item ref -> milestone name
        inserted_plans: list[tuple[str, MilestonePlan]] = []
        gaps = sorted((nm for nm in fixed_names if _MILESTONE.fullmatch(nm) and self._not_started(nm)),
                      key=self.mkey)
        high = [i.ref for i in self.plannable if self.eff[i.ref] == PRIORITY["high"]]
        for gap in gaps:
            if f"ahead-of-{gap}" in appearance:
                continue
            pool = [r for r in high if r not in self.inserted]
            pool = self._insertable(pool, gap)
            if not pool:
                continue
            plan = self._sequence(self._ranked(pool), self.n, cap=True)
            if not plan:
                continue
            name = f"ahead-of-{gap}"
            plans = self._plan_milestones(plan, name_of=lambda _k, nm=name: nm, kind="inserted", ahead=gap)
            inserted_plans.append((name, plans[0]))
            for pl in plans[0].items:
                self.inserted[pl.ref] = name
        rest = [i for i in self.plannable if i.ref not in self.inserted]
        seq = self._sequence(self._ranked([i.ref for i in rest]), self.n, cap=False)
        proposed = self._plan_milestones(seq, name_of=lambda k: f"proposed-{k + 1}", kind="proposed")
        self.fixed_plans = self._fixed_plans()
        ordered = sorted(self.fixed_plans + inserted_plans, key=lambda kp: self.mkey(kp[0]))
        self.out.milestones = [m for _k, m in ordered] + proposed

    def _not_started(self, name: str) -> bool:
        members = [i for i in self.fixed if i.milestone == name]
        return not any(i.status == "in-progress" or i.milestone_started for i in members)

    def _insertable(self, pool: list[str], gap: str) -> list[str]:
        """The high items that can sit in the gap before `gap` (plan 3.7 step 0): every internal dependency
        delivered, in a fixed milestone before it, or in the same gap (closure)."""
        pool = list(pool)
        gap_key = self.mkey(gap)
        while True:
            keep = [r for r in pool if self._deps_ok(self.by_ref[r], gap_key, pool)]
            if keep == pool:
                return pool
            pool = keep

    def _deps_ok(self, item, gap_key, pool) -> bool:
        for d in self.deps[item.ref]:
            if d.kind != "internal":
                continue
            t = d.target
            if t in self.fixed:
                if not self.mkey(t.milestone) < gap_key:
                    return False
            elif t.ref in self.inserted:
                continue  # inserted into an earlier gap
            elif t.ref not in pool:
                return False  # in progress, another placement, or not insertable
        return True

    # a scheduled sequence: list of (wave, [items in rank order]) plus per-item bookkeeping
    def _sequence(self, ranked, cap_n, cap):
        seq_refs = {i.ref for i in ranked}
        wave_of: dict[str, int] = {}
        skips: dict[str, tuple] = {}
        waves: list[list] = []
        remaining = list(ranked)
        k = 0
        while remaining:
            k += 1
            in_wave: list = []
            for c in list(remaining):
                seq_deps = [d.target for d in self.deps[c.ref] if d.kind == "internal" and d.target.ref in seq_refs]
                carried_dep = any(d.kind == "internal" and d.target in self.carried for d in self.deps[c.ref])
                if any(t.ref not in wave_of for t in seq_deps) or (carried_dep and k < 2):
                    continue
                blockers = (self.carried if k == 1 else []) + in_wave
                hit = next(((b, ov) for b in blockers if (ov := touch_overlap(c.touches or [], b.touches or []))),
                           None)
                if hit:
                    skips.setdefault(c.ref, (hit[0].ref, *hit[1]))
                    continue
                in_wave.append(c)
                remaining.remove(c)
            if not in_wave and k > len(ranked) + 2:
                raise AssertionError("proposal schedule made no progress")  # unreachable: cycles are removed
            for c in in_wave:
                wave_of[c.ref] = k
            waves.append(in_wave)
        flat = [(wave_of[c.ref], c) for w in waves for c in w]
        if cap:
            flat = flat[:cap_n]
        return {"flat": flat, "wave_of": wave_of, "skips": skips, "seq_refs": {c.ref for _w, c in flat}}

    def _plan_milestones(self, seq, name_of, kind, ahead=None) -> list[MilestonePlan]:
        if not seq["flat"]:
            return []
        plans: list[MilestonePlan] = []
        wave_start: dict[int, int] = {}
        placed_ms: dict[str, int] = {}
        size = 0
        for wave, c in seq["flat"]:
            if not plans or (kind == "proposed" and size == self.n):
                plans.append(MilestonePlan("", kind))
                size = 0
            wave_start.setdefault(wave, len(plans) - 1)
            plans[-1].items.append(c)  # items for now; Placement built below
            placed_ms[c.ref] = len(plans) - 1
            size += 1
        for idx, plan in enumerate(plans):
            plan.name = name_of(idx)
            plan.items = [self._placement(c, seq, idx > wave_start[seq["wave_of"][c.ref]], ahead)
                          for c in plan.items]
        return plans

    def _placement(self, c, seq, spilled, ahead) -> Placement:
        wave = seq["wave_of"][c.ref]
        own = _pr(c)
        seq_refs = seq["seq_refs"]
        constraints = []  # (earliest wave, dep ref, in progress)
        for d in self.deps[c.ref]:
            if d.kind != "internal":
                continue
            if d.target.ref in seq_refs:
                constraints.append((seq["wave_of"][d.target.ref] + 1, d.target.ref, False))
            elif d.target in self.carried:
                constraints.append((2, d.target.ref, True))
        dew, dep_ref, in_progress = max(constraints, key=lambda x: x[0], default=(1, None, False))
        if spilled:
            reason = "(c) milestone capacity"
        elif wave > dew:
            blocker, kind, value = seq["skips"][c.ref]
            reason = f"(b) overlap with {blocker} on {kind} {value}"
        elif dew > 1:
            reason = f"(a) after {dep_ref}" + (" (in progress)" if in_progress else "")
        elif wave == 1 and self.eff[c.ref] > own:
            names = [r for r in self.trans[c.ref] if _pr(self.by_ref[r]) == self.eff[c.ref]]
            reason = "(a) unblocks " + ", ".join(names)
        else:
            reason = f"(c) priority {c.priority}"
        return Placement(c.ref, c.title, c.backlog_path, wave, reason, ahead)

    # -- fixed items
    def _fixed_plans(self) -> list[tuple[str, MilestonePlan]]:
        by_name: dict[str, list] = {}
        for i in self.fixed:
            by_name.setdefault(i.milestone, []).append(i)
        impossible = {i.ref: self._causes(i) for i in self.fixed}
        self.out.impossible = [Impossible(i.ref, i.title, i.backlog_path, tuple(impossible[i.ref]))
                               for i in self.fixed if impossible[i.ref]]
        plans = []
        for name, members in by_name.items():
            members.sort(key=lambda i: (i.milestone_order if i.milestone_order is not None else float("inf"), i.ord))
            plans.append((name, MilestonePlan(name, "fixed", [
                Placement(i.ref, i.title, i.backlog_path, impossible=tuple(impossible[i.ref])) for i in members])))
        return plans

    def _causes(self, f) -> list[str]:
        causes: list[str] = []
        fkey = self.mkey(f.milestone)
        for d in self.deps[f.ref]:
            if d.kind == "external":
                causes.append(f"external blocker {d.external[0]} in `{d.external[1]}`")
            elif d.kind == "unknown":
                causes.append(f"unknown dependency {d.token} ({d.reason})")
            elif d.kind == "internal":
                causes += self._dep_causes(f, fkey, d.target)
        if f.ref in self.cycles:
            causes.append("sits on a dependency cycle: " + ", ".join(self.cycles[f.ref]))
        for g in self.fixed:
            if g is f or g.milestone != f.milestone:
                continue
            distinct = (f.milestone_order is not None and g.milestone_order is not None
                        and f.milestone_order != g.milestone_order)
            if not distinct and (ov := touch_overlap(f.touches or [], g.touches or [])):
                causes.append(f"overlaps {g.ref} on {ov[0]} {ov[1]} with no distinct order between them")
        return causes

    def _dep_causes(self, f, fkey, t) -> list[str]:
        if t in self.fixed:
            out = []
            if t.status in ("deferred", "rejected"):
                out.append(f"depends on {t.ref} ({t.status})")
            if self.mkey(t.milestone) > fkey:
                out.append(f"depends on {t.ref} in later milestone {t.milestone}")
            elif (self.mkey(t.milestone) == fkey and t.milestone_order is not None and f.milestone_order is not None
                  and t.milestone_order >= f.milestone_order):
                out.append(f"depends on {t.ref} at the same or a later order in {t.milestone}")
            return out
        placed = self.inserted.get(t.ref)
        if placed is not None and self.mkey(placed) < fkey:
            return []
        why = ("deferred" if t in self.deferred else "in progress" if t in self.carried
               else "held" if t.ref not in self.not_plannable and t not in self.plannable
               else "not plannable yet" if t.ref in self.not_plannable else "unassigned")
        return [f"depends on {t.ref}, which is {why} and not placed in an earlier milestone"]
