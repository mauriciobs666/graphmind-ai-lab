"""The milestone proposal (plan section 3.7, section 5 proposal tests; AC-11 to AC-14): pure, offline."""
import re

import pytest

from coordgraph.parse.backlog import Item
from coordgraph.propose import DepResolution, propose, touch_overlap

SCOPE = "comp"
PATH = "comp/docs/BACKLOG.md"
UNSET = object()


def it(n, status="proposed", prio="medium", deps=(), touches=(), milestone=None, order=None, started=None):
    """An open item `comp:K-<n>` (file order = n); `None` for prio/deps/touches = fact not recorded."""
    return Item(
        ref=f"{SCOPE}:K-{n:03d}", item_id=f"K-{n:03d}", scope=SCOPE, backlog_path=PATH, ord=n, line=n,
        title=f"Item {n}", item_hash="h", status=status, status_marker=status, priority=prio,
        depends_on=None if deps is None else [f"K-{d:03d}" if isinstance(d, int) else d for d in deps],
        touches=None if touches is None else list(touches), milestone=milestone, milestone_order=order,
        milestone_started=started,
    )


def ref(n):
    return f"{SCOPE}:K-{n:03d}"


def resolver_for(extra=None):
    """Bare `K-n` denotes `comp:K-n`; `extra` maps a token to a `DepResolution` for everything else."""
    extra = extra or {}

    def resolve(token, scope):
        if token in extra:
            return extra[token]
        if re.fullmatch(r"K-\d+", token):
            return DepResolution(frozenset({f"{scope}:{token}"}), reason="not in any open backlog")
        return DepResolution(frozenset({token}), reason="not in any open backlog")

    return resolve


def run(items, n=6, extra=None):
    return propose(items, resolver_for(extra), n)


def where(p, n):
    """`(milestone name, wave, placement)` of `comp:K-n`."""
    for m in p.milestones:
        for pl in m.items:
            if pl.ref == ref(n):
                return m.name, pl.wave, pl
    raise AssertionError(f"{ref(n)} is not placed: {[m.name for m in p.milestones]}")


def waves(p, *ns):
    return [where(p, n)[1] for n in ns]


def refs(entries):
    return [e.ref for e in entries]


CRITERION = re.compile(r"\(([abc])\) ")


# --- AC-11: the fixture -----------------------------------------------------------------------

def ac11_items():
    return [
        it(1),                                                       # A
        it(2, deps=[1]),                                             # B deps A
        it(3, touches=["file:src/a.py"]), it(4, touches=["file:src/a.py"]),                       # C/D
        it(7, touches=["graph:kb"]), it(8, touches=["graph:kb"]),                                 # G/H
        it(9, touches=["fact:Tok (changes)"]), it(10, touches=["fact:tok (relies)"]),            # I/J
        it(5, prio="high", touches=["file:e.py"]), it(6, prio="low", touches=["file:f.py"]),     # E high, F low
    ]


def test_ac11_dependency_pair_waves_overlap_pairs_and_priority():
    p = run(ac11_items(), n=20)
    assert where(p, 1)[1] < where(p, 2)[1]
    for a, b in ((3, 4), (7, 8), (9, 10)):
        assert where(p, a)[1] != where(p, b)[1]
    assert where(p, 5)[1] <= where(p, 6)[1]


def test_ac11_overlap_is_named_in_the_later_items_reason():
    p = run(ac11_items(), n=20)
    assert where(p, 4)[2].reason == f"(b) overlap with {ref(3)} on file src/a.py"
    assert where(p, 8)[2].reason == f"(b) overlap with {ref(7)} on graph kb"
    assert where(p, 10)[2].reason.startswith(f"(b) overlap with {ref(9)} on fact ")


def test_ac11_every_placement_carries_exactly_one_criterion():
    p = run(ac11_items(), n=20)
    placed = [pl for m in p.milestones for pl in m.items]
    assert len(placed) == 10
    for pl in placed:
        assert len(CRITERION.findall(pl.reason)) == 1, pl.reason
        assert CRITERION.match(pl.reason)


def test_two_relies_on_one_fact_do_not_overlap():
    p = run([it(1, touches=["fact:tok (relies)"]), it(2, touches=["fact:TOK (relies)"])])
    assert waves(p, 1, 2) == [1, 1]


# --- AC-12: not plannable yet -----------------------------------------------------------------

@pytest.mark.parametrize("kwargs,missing", [
    ({"prio": None}, "Priority"), ({"deps": None}, "Depends on"), ({"touches": None}, "Touches"),
])
def test_ac12_each_missing_fact_makes_the_item_not_plannable_and_is_named(kwargs, missing):
    p = run([it(1, **kwargs), it(2)])
    assert [e.ref for e in p.not_plannable] == [ref(1)]
    assert p.not_plannable[0].missing == (missing,)
    assert [m.name for m in p.milestones] == ["proposed-1"]
    assert refs(p.milestones[0].items) == [ref(2)]


def test_ac12_none_counts_as_recorded():
    p = run([it(1, deps=[], touches=[])])
    assert p.not_plannable == [] and where(p, 1)[0] == "proposed-1"


# --- AC-13: dependency resolution -------------------------------------------------------------

def test_ac13_external_open_dependency_holds_with_ref_and_path():
    ext = {"other:K-900": DepResolution(frozenset({"other:K-900"}), external=("other:K-900", "other/docs/BACKLOG.md"))}
    p = run([it(1, deps=["other:K-900"])], extra=ext)
    assert refs(p.held) == [ref(1)]
    assert p.held[0].reason == "external blocker other:K-900 in `other/docs/BACKLOG.md`"
    assert p.milestones == []


def test_ac13_found_delivered_in_history_is_satisfied():
    ext = {"other:K-901": DepResolution(frozenset({"other:K-901"}), delivered="other/docs/HISTORY.md")}
    p = run([it(1, deps=["other:K-901"])], extra=ext)
    assert p.held == [] and p.not_plannable == []
    assert where(p, 1)[1] == 1


def test_ac13_history_mention_without_delivery_shape_is_unknown_dependency():
    p = run([it(1, deps=["other:K-902"])])
    assert refs(p.not_plannable) == [ref(1)]
    assert p.not_plannable[0].missing == ()
    assert p.not_plannable[0].unknown == ("other:K-902",)
    assert "unknown dependency other:K-902" in p.not_plannable[0].text
    assert "drop the ref or add a `<ID> delivered` history heading" in p.not_plannable[0].text


def test_every_dependency_token_counts_not_only_the_first():
    ext = {"other:K-900": DepResolution(frozenset({"other:K-900"}), external=("other:K-900", "o/BACKLOG.md"))}
    p = run([it(1), it(2, deps=[1, "other:K-900"])], extra=ext)
    assert refs(p.held) == [ref(2)] and "other:K-900" in p.held[0].reason
    p = run([it(1), it(2, deps=[1, "other:K-902"])])
    assert refs(p.not_plannable) == [ref(2)] and p.not_plannable[0].unknown == ("other:K-902",)


def test_unknown_dependency_reason_from_identity_is_kept():
    ext = {"GAL-T0001": DepResolution(frozenset({"GAL-T0001"}), reason="unminted: no such item")}
    p = run([it(1, deps=["GAL-T0001"])], extra=ext)
    assert "unminted: no such item" in p.not_plannable[0].text


# --- structure: cycles, holds, deferred, carried -----------------------------------------------

def test_cycle_makes_every_member_not_plannable_and_names_the_cycle():
    p = run([it(1, deps=[2]), it(2, deps=[1]), it(3)])
    assert sorted(refs(p.not_plannable)) == [ref(1), ref(2)]
    assert f"{ref(1)}" in p.not_plannable[0].text and "cycle" in p.not_plannable[0].text
    assert refs(p.milestones[0].items) == [ref(3)]


def test_self_dependency_is_a_cycle():
    p = run([it(1, deps=[1])])
    assert refs(p.not_plannable) == [ref(1)]


def test_holding_propagates_to_dependents():
    ext = {"other:K-900": DepResolution(frozenset({"other:K-900"}), external=("other:K-900", "o/docs/BACKLOG.md"))}
    p = run([it(1, deps=["other:K-900"]), it(2, deps=[1]), it(3, deps=[2]), it(4)], extra=ext)
    assert refs(p.held) == [ref(1), ref(2), ref(3)]
    assert p.held[1].reason == f"waits on {ref(1)} (held)"
    assert refs(p.milestones[0].items) == [ref(4)]


def test_item_depending_on_a_not_plannable_item_is_held():
    p = run([it(1, prio=None), it(2, deps=[1])])
    assert refs(p.held) == [ref(2)]
    assert p.held[0].reason == f"waits on {ref(1)} (not plannable yet)"


def test_deferred_item_is_listed_and_its_dependent_held():
    p = run([it(1, status="deferred"), it(2, status="rejected", prio=None), it(3, deps=[1])])
    assert refs(p.deferred) == [ref(1), ref(2)]
    assert p.held[0].reason == f"waits on {ref(1)} (deferred)"
    assert p.milestones == []


def test_carried_item_is_listed_and_not_placed():
    p = run([it(1, status="in-progress", touches=["file:a.py"]), it(2, touches=["file:b.py"])])
    assert refs(p.carried) == [ref(1)]
    assert refs(p.milestones[0].items) == [ref(2)]


def test_carried_item_blocks_an_overlapping_item_from_wave_1():
    p = run([it(1, status="in-progress", touches=["file:src/"]), it(2, touches=["file:src/x.py"]),
             it(3, touches=["file:other.py"])])
    assert waves(p, 2, 3) == [2, 1]
    assert where(p, 2)[2].reason == f"(b) overlap with {ref(1)} on file src/"


def test_dependency_on_in_progress_item_lands_in_wave_2_with_reason():
    p = run([it(1, status="in-progress", touches=["file:a.py"]), it(2, deps=[1], touches=["file:b.py"])])
    assert where(p, 2)[1] == 2
    assert where(p, 2)[2].reason == f"(a) after {ref(1)} (in progress)"


def test_delivered_dependency_in_this_backlog_name_is_not_an_in_progress_dep():
    ext = {"K-050": DepResolution(frozenset({"comp:K-050"}), delivered="comp/docs/HISTORY.md")}
    p = run([it(1, deps=["K-050"])], extra=ext)
    assert where(p, 1)[1] == 1


# --- ranking and waves ---------------------------------------------------------------------------

def test_effective_priority_pulls_a_low_dependency_ahead_of_a_medium_rival():
    items = [it(1, prio="medium", touches=["file:x.py"]),
             it(2, prio="low", touches=["file:x.py"]),
             it(3, prio="high", deps=[2], touches=["file:h.py"])]
    p = run(items)
    assert where(p, 2)[1] == 1 and where(p, 1)[1] == 2


def test_ties_break_by_unblocks_then_own_priority_then_file_order():
    # 1 and 2 conflict; 2 unblocks two items, so it wins wave 1 over file order.
    p = run([it(1, touches=["file:x.py"]), it(2, touches=["file:x.py"]),
             it(3, deps=[2], touches=["file:c.py"]), it(4, deps=[2], touches=["file:d.py"])])
    assert where(p, 2)[1] == 1 and where(p, 1)[1] == 2
    # same unblocks (none), own priority decides: the higher own priority wins; equal -> file order
    q = run([it(1, prio="low", touches=["file:x.py"]), it(2, prio="medium", touches=["file:x.py"]),
             it(3, prio="medium", touches=["file:x.py"])])
    assert waves(q, 2, 3, 1) == [1, 2, 3]


def test_unblocks_counts_transitive_dependents_not_only_direct_ones():
    # 1 and 2 conflict; each has one direct dependent, but 2's has a further one: 2 unblocks 2, 1 unblocks 1.
    p = run([it(1, touches=["file:x.py"]), it(2, touches=["file:x.py"]),
             it(3, deps=[1], touches=["file:c.py"]), it(4, deps=[2], touches=["file:d.py"]),
             it(5, deps=[4], touches=["file:e.py"])])
    assert where(p, 2)[1] == 1 and where(p, 1)[1] == 2


def test_a_pushed_item_keeps_its_rank_in_the_later_wave():
    p = run([it(1, prio="high", touches=["file:x.py"]), it(2, prio="medium", touches=["file:x.py"]),
             it(3, prio="low", touches=["file:x.py"])])
    assert waves(p, 1, 2, 3) == [1, 2, 3]


def test_overlap_items_are_always_in_different_waves_even_without_deps():
    p = run([it(i, touches=["file:same.py"]) for i in range(1, 5)])
    assert sorted(waves(p, 1, 2, 3, 4)) == [1, 2, 3, 4]


def test_dependency_always_sits_in_an_earlier_wave():
    p = run([it(1), it(2, deps=[1]), it(3, deps=[2])])
    assert waves(p, 1, 2, 3) == [1, 2, 3]
    assert where(p, 3)[2].reason == f"(a) after {ref(2)}"


# --- overlap rules -------------------------------------------------------------------------------

@pytest.mark.parametrize("a,b,hit", [
    (["file:src/a.py"], ["file:src/a.py"], True),
    (["file:src/a.py"], ["file:src/b.py"], False),
    (["file:src/"], ["file:src/deep/b.py"], True),            # directory covers path
    (["file:src"], ["file:src/b.py"], True),                  # bare directory covers path
    (["file:srcx"], ["file:src/b.py"], False),                # prefix on a name boundary only
    (["file:src/*.py"], ["file:src/a.py"], True),             # glob covers path
    (["file:src/*.py"], ["file:docs/a.py"], False),
    (["file:src/*.py"], ["file:src/a/*.js"], True),           # glob vs glob: conservative literal prefix
    (["file:tests/*.py"], ["file:src/*.py"], False),
    (["file:*.md"], ["file:docs/*.py"], True),                # empty literal prefix: possible intersection
    (["graph:kb"], ["graph:kb"], True),
    (["graph:kb"], ["graph:KB"], False),
    (["fact:Tok (changes)"], ["fact:tok (relies)"], True),
    (["fact:tok (relies)"], ["fact:tok (relies)"], False),
    (["fact:tok (changes)"], ["fact:tok (changes)"], True),
    (["fact:a (changes)"], ["fact:b (changes)"], False),
    (["fact:Tok"], ["fact:tok"], True),                       # no mode written: counts as changes, case-folded
    (["file:a.py"], ["graph:a.py"], False),
    ([], ["file:a.py"], False),
])
def test_touch_overlap_rules(a, b, hit):
    assert (touch_overlap(a, b) is not None) is hit
    assert (touch_overlap(b, a) is not None) is hit


def test_a_file_entry_covers_a_directory_only_on_a_name_boundary():
    assert touch_overlap(["file:src/a"], ["file:src/ab.py"]) is None
    assert touch_overlap(["file:src/ab.py"], ["file:src/a"]) is None
    assert touch_overlap(["file:src/a"], ["file:src/a/b.py"]) == ("file", "src/a")
    p = run([it(1, touches=["file:src/a"]), it(2, touches=["file:src/ab.py"])])
    assert waves(p, 1, 2) == [1, 1]  # no false overlap pushing the second into a later wave


def test_touch_overlap_names_kind_and_value():
    assert touch_overlap(["file:a.py", "graph:kb"], ["graph:kb"]) == ("graph", "kb")


# --- capacity and explanation precedence -------------------------------------------------------

def test_default_capacity_is_six():
    p = propose([it(i, touches=[f"file:f{i}.py"]) for i in range(1, 9)], resolver_for())
    assert [len(m.items) for m in p.milestones] == [6, 2]


def test_capacity_splits_waves_into_proposed_milestones_default_6_and_other_n():
    items = [it(i, touches=[f"file:f{i}.py"]) for i in range(1, 9)]
    p = run(items)
    assert [(m.name, len(m.items)) for m in p.milestones] == [("proposed-1", 6), ("proposed-2", 2)]
    q = run(items, n=3)
    assert [(m.name, len(m.items)) for m in q.milestones] == [("proposed-1", 3), ("proposed-2", 3), ("proposed-3", 2)]


def test_spilled_item_reports_capacity_and_a_full_wave_boundary_does_not():
    items = [it(i, touches=[f"file:f{i}.py"]) for i in range(1, 4)] + [it(4, deps=[1], touches=["file:g.py"])]
    p = run(items, n=3)   # wave 1 holds 1,2,3 exactly; 4 is wave 2 and opens milestone 2
    assert where(p, 4)[0] == "proposed-2" and where(p, 4)[2].reason == f"(a) after {ref(1)}"
    q = run([it(i, touches=[f"file:f{i}.py"]) for i in range(1, 5)], n=3)  # wave 1 has 4 items: one spills
    assert where(q, 4)[0] == "proposed-2" and where(q, 4)[2].reason == "(c) milestone capacity"


def test_precedence_overlap_beats_dependency_and_capacity_beats_overlap():
    # 3 deps on 1 (dependency-earliest wave 2) but overlaps 2 in wave 2 -> wave 3: the overlap is reported
    p = run([it(1, touches=["file:a.py"]), it(2, deps=[1], touches=["file:b.py"]),
             it(3, deps=[1], touches=["file:b.py"])])
    assert waves(p, 2, 3) == [2, 3]
    assert where(p, 3)[2].reason == f"(b) overlap with {ref(2)} on file b.py"
    # 5 is independent, overlaps 1 (wave 2, dew 1) and, with N=2, wave 2 [2, 3, 5] spills 5 and 3
    items = [it(1, touches=["file:a.py"]), it(2, deps=[1], touches=["file:b.py"]),
             it(3, deps=[1], touches=["file:c.py"]), it(5, touches=["file:a.py"])]
    assert where(run(items), 5)[2].reason == f"(b) overlap with {ref(1)} on file a.py"
    q = run(items, n=2)
    assert where(q, 5)[0] == "proposed-2" and where(q, 5)[2].reason == "(c) milestone capacity"
    assert where(q, 3)[2].reason == "(c) milestone capacity"


def test_capacity_reason_when_a_wave_itself_spills():
    q = run([it(i, touches=[f"file:f{i}.py"]) for i in range(1, 4)], n=2)
    assert where(q, 3)[2].reason == "(c) milestone capacity"


def test_unblocks_reason_when_effective_priority_exceeds_own_in_wave_1():
    p = run([it(1, prio="low"), it(2, prio="high", deps=[1], touches=["file:h.py"])])
    assert where(p, 1)[2].reason == f"(a) unblocks {ref(2)}"


def test_priority_reason_otherwise():
    p = run([it(1, prio="high")])
    assert where(p, 1)[2].reason == "(c) priority high"


# --- fixed milestones (decision log 2026-10-21) ------------------------------------------------

def test_fixed_items_keep_their_milestone_and_order_and_new_items_open_proposed_after_them():
    p = run([it(1, milestone="M1", order=2), it(2, milestone="M1", order=1), it(3, milestone="M2", order=1),
             it(4), it(5, prio="medium")])
    assert [m.name for m in p.milestones] == ["M1", "M2", "proposed-1"]
    assert refs(p.milestones[0].items) == [ref(2), ref(1)]
    assert refs(p.milestones[2].items) == [ref(4), ref(5)]
    assert all(pl.reason is None for pl in p.milestones[0].items)


def test_a_sub_milestone_orders_by_its_second_number():
    p = run([it(1, milestone="M5.10"), it(2, milestone="M5.2"), it(3, milestone="M5.1"), it(4, milestone="M5")])
    assert [m.name for m in p.milestones] == ["M5", "M5.1", "M5.2", "M5.10"]


def test_an_unassigned_item_is_never_added_to_a_fixed_milestone_with_room():
    p = run([it(1, milestone="M1", order=1), it(2)], n=6)
    assert refs(p.milestones[0].items) == [ref(1)]


def test_fixed_milestones_sort_numerically_and_ahead_of_goes_just_before_its_target():
    p = run([it(1, milestone="M10"), it(2, milestone="M2"), it(3, milestone="ahead-of-M10"),
             it(4, milestone="M2.1"), it(5, milestone="Later")])
    assert [m.name for m in p.milestones] == ["M2", "M2.1", "ahead-of-M10", "M10", "Later"]
    assert any("Later" in w for w in p.warnings)


def test_fixed_items_feed_dependency_targets_for_unassigned_items():
    p = run([it(1, milestone="M1", order=1), it(2, deps=[1])])
    assert where(p, 2)[:2] == ("proposed-1", 1)


def test_fixed_priority_pulls_an_unassigned_dependency_forward():
    p = run([it(1, prio="medium", touches=["file:x.py"]), it(2, prio="low", touches=["file:x.py"]),
             it(3, prio="high", deps=[2], milestone="M9", order=1, started="2026-11-01")])
    assert where(p, 2)[1] == 1 and where(p, 1)[1] == 2


# --- insertion (decision log 2026-10-22) -------------------------------------------------------

def test_high_dependency_free_item_is_inserted_ahead_of_a_not_started_fixed_milestone():
    p = run([it(1, milestone="M7", order=1), it(2, prio="high")])
    assert [m.name for m in p.milestones] == ["ahead-of-M7", "M7"]
    name, wave, pl = where(p, 2)
    assert name == "ahead-of-M7" and pl.inserted_ahead_of == "M7"
    assert "inserted ahead of approved M7" in p.render()
    assert len(CRITERION.findall(pl.reason)) == 1


def test_insertion_goes_to_the_earliest_eligible_gap():
    p = run([it(1, milestone="M5", order=1), it(2, milestone="M7", order=1), it(3, prio="high")])
    assert where(p, 3)[0] == "ahead-of-M5"


def test_never_inserted_before_a_started_milestone_by_in_progress_item_or_by_started_marker():
    ip = run([it(1, milestone="M7", status="in-progress"), it(2, prio="high")])
    assert where(ip, 2)[0] == "proposed-1"
    st = run([it(1, milestone="M7", order=2, started="2026-11-01"), it(2, prio="high")])
    assert where(st, 2)[0] == "proposed-1"


def test_a_started_gap_is_skipped_for_the_next_not_started_one():
    p = run([it(1, milestone="M5", started="2026-11-01"), it(2, milestone="M7"), it(3, prio="high")])
    assert where(p, 3)[0] == "ahead-of-M7"


def test_not_inserted_when_a_dependency_sits_in_the_target_or_a_later_milestone():
    p = run([it(1, milestone="M7", order=1), it(2, prio="high", deps=[1])])
    assert where(p, 2)[0] == "proposed-1"
    q = run([it(1, milestone="M8", order=1), it(2, milestone="M7"), it(3, prio="high", deps=[1])])
    assert where(q, 3)[0] == "proposed-1"


def test_not_inserted_when_it_depends_on_an_in_progress_item():
    p = run([it(1, status="in-progress", touches=["file:a.py"]), it(2, milestone="M7"),
             it(3, prio="high", deps=[1])])
    assert where(p, 3)[:2] == ("proposed-1", 2)
    assert where(p, 3)[2].reason == f"(a) after {ref(1)} (in progress)"


def test_inserted_when_the_dependency_sits_in_an_earlier_fixed_milestone():
    p = run([it(1, milestone="M6", order=1), it(2, milestone="M7", order=1), it(3, prio="high", deps=[1])])
    assert where(p, 3)[0] == "ahead-of-M7"


def test_inserted_item_may_depend_on_another_inserted_item_in_an_earlier_wave():
    p = run([it(1, milestone="M7"), it(2, prio="high"), it(3, prio="high", deps=[2])])
    assert where(p, 2)[:2] == ("ahead-of-M7", 1) and where(p, 3)[:2] == ("ahead-of-M7", 2)


def test_high_item_depending_on_a_medium_unassigned_item_pulls_it_in_and_both_insert():
    p = run([it(1, milestone="M7"), it(2, prio="medium"), it(3, prio="high", deps=[2])])
    assert where(p, 2)[0] == "ahead-of-M7" and where(p, 3)[0] == "ahead-of-M7"


def test_medium_item_is_never_inserted():
    p = run([it(1, milestone="M7"), it(2, prio="medium")])
    assert where(p, 2)[0] == "proposed-1"


def test_insertion_overflow_beyond_n_falls_back_to_proposed():
    items = [it(1, milestone="M7")] + [it(i, prio="high", touches=[f"file:f{i}.py"]) for i in (2, 3, 4)]
    p = run(items, n=2)
    assert [m.name for m in p.milestones] == ["ahead-of-M7", "M7", "proposed-1"]
    assert len(p.milestones[0].items) == 2 and len(p.milestones[2].items) == 1


def test_inserted_items_precede_proposed_ones_that_depend_on_them():
    p = run([it(1, milestone="M7"), it(2, prio="high"), it(3, prio="low", deps=[2])])
    assert where(p, 3)[:2] == ("proposed-1", 1)


def test_an_existing_ahead_of_milestone_is_not_proposed_over():
    p = run([it(1, milestone="M7"), it(2, milestone="ahead-of-M7"), it(3, prio="high")])
    assert where(p, 3)[0] == "proposed-1"


# --- approved placements now impossible --------------------------------------------------------

def impossible(p):
    return {e.ref: e.causes for e in p.impossible}


def test_fixed_item_depending_on_unassigned_item_is_impossible_and_stays_put():
    p = run([it(1, milestone="M1", order=1, deps=[2], prio="medium"), it(2, prio="low", touches=["file:z.py"])])
    assert ref(1) in impossible(p) and "depends on" in impossible(p)[ref(1)][0] and ref(2) in impossible(p)[ref(1)][0]
    assert where(p, 1)[0] == "M1"


def test_fixed_item_depending_on_an_unassigned_item_inserted_ahead_of_it_is_fine():
    p = run([it(1, milestone="M7", order=1, deps=[2]), it(2, prio="medium")])
    # fixed item feeds its own priority? it has medium; effective stays medium -> not inserted -> impossible
    assert ref(1) in impossible(p)
    q = run([it(1, milestone="M7", order=1, deps=[2], prio="high"), it(2, prio="medium")])
    assert where(q, 2)[0] == "ahead-of-M7" and impossible(q) == {}


def test_fixed_item_depending_on_a_later_fixed_milestone_or_a_later_order_is_impossible():
    p = run([it(1, milestone="M1", order=1, deps=[2]), it(2, milestone="M2", order=1)])
    assert ref(1) in impossible(p) and "M2" in impossible(p)[ref(1)][0]
    q = run([it(1, milestone="M1", order=1, deps=[2]), it(2, milestone="M1", order=2)])
    assert ref(1) in impossible(q)
    eq = run([it(1, milestone="M1", order=1, deps=[2]), it(2, milestone="M1", order=1)])
    assert ref(1) in impossible(eq)
    named = run([it(1, milestone="Alpha", deps=[2]), it(2, milestone="Beta")])
    assert ref(1) in impossible(named)  # other tokens order by first appearance: Beta is later
    ok = run([it(1, milestone="M2", order=1, deps=[2]), it(2, milestone="M1", order=5)])
    assert impossible(ok) == {}
    ok2 = run([it(1, milestone="M1", order=2, deps=[2]), it(2, milestone="M1", order=1)])
    assert impossible(ok2) == {}


@pytest.mark.parametrize("dep_status", ["deferred", "rejected"])
def test_fixed_item_depending_on_a_deferred_item_is_impossible(dep_status):
    p = run([it(1, milestone="M1", order=2, deps=[2]), it(2, status=dep_status, milestone="M1", order=1)])
    assert ref(1) in impossible(p) and dep_status in impossible(p)[ref(1)][0]


def test_fixed_item_depending_on_an_externally_open_item_is_impossible():
    ext = {"other:K-900": DepResolution(frozenset({"other:K-900"}), external=("other:K-900", "o/docs/BACKLOG.md"))}
    p = run([it(1, milestone="M1", order=1, deps=["other:K-900"])], extra=ext)
    assert "external blocker other:K-900" in impossible(p)[ref(1)][0]


def test_fixed_items_on_a_cycle_are_impossible():
    p = run([it(1, milestone="M1", order=1, deps=[2]), it(2, milestone="M1", order=2, deps=[1])])
    assert set(impossible(p)) == {ref(1), ref(2)}
    assert any("cycle" in c for c in impossible(p)[ref(1)])


def test_two_overlapping_fixed_items_without_distinct_order_are_flagged():
    t = ["file:a.py"]
    p = run([it(1, milestone="M1", order=1, touches=t), it(2, milestone="M1", order=1, touches=t)])
    assert any("overlaps" in c for c in impossible(p)[ref(1)])
    ok = run([it(1, milestone="M1", order=1, touches=t), it(2, milestone="M1", order=2, touches=t)])
    assert impossible(ok) == {}
    other = run([it(1, milestone="M1", order=1, touches=t), it(2, milestone="M2", order=1, touches=t)])
    assert impossible(other) == {}


# --- purity, determinism, output --------------------------------------------------------------

def test_same_input_twice_is_identical_output():
    items = ac11_items() + [it(11, milestone="M3", order=1), it(12, prio="high")]
    assert run(items, n=4) == run(items, n=4)
    assert run(items, n=4).render() == run(items, n=4).render()


def test_input_items_are_not_modified():
    items = ac11_items()
    before = [(i.ref, i.depends_on, i.touches, i.milestone) for i in items]
    run(items)
    assert [(i.ref, i.depends_on, i.touches, i.milestone) for i in items] == before


def test_per_milestone_must_be_positive():
    with pytest.raises(ValueError):
        run([it(1)], n=0)


def test_closed_items_are_not_proposed():
    d = it(1)
    d.open = False
    assert run([d, it(2)]).milestones[0].items[0].ref == ref(2)


def test_render_lists_every_section_with_id_and_source_file():
    ext = {"other:K-900": DepResolution(frozenset({"other:K-900"}), external=("other:K-900", "o/docs/BACKLOG.md"))}
    items = [it(1, milestone="M1", order=1), it(2), it(3, status="in-progress"), it(4, status="deferred"),
             it(5, prio=None), it(6, deps=["other:K-900"])]
    out = run(items, extra=ext).render()
    for needle in ("M1", "proposed-1", "Carried", "Held:", "Deferred", "Not plannable yet", ref(1), ref(2),
                   ref(3), ref(4), ref(5), ref(6), PATH, "o/docs/BACKLOG.md", "Priority"):
        assert needle in out, needle


def test_the_planner_imports_only_pure_standard_library_modules():
    """Purity (plan 3.7): no store, `wilib`, identity, network, file, clock or randomness; the caller's resolver
    does the I/O. The import set is the whole claim: any other module would have to appear in it."""
    import ast
    import sys
    from pathlib import Path

    from coordgraph import propose as module

    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert not [n for n in imports if isinstance(n, ast.ImportFrom) and n.level]  # no relative (package) import
    roots = {a.name.split(".")[0] for n in imports if isinstance(n, ast.Import) for a in n.names}
    roots |= {n.module.split(".")[0] for n in imports if isinstance(n, ast.ImportFrom)}
    assert roots and roots <= set(sys.stdlib_module_names), roots
    impure = {"os", "pathlib", "io", "socket", "urllib", "http", "subprocess", "time", "datetime", "random",
              "sqlite3", "shutil", "tempfile", "uuid", "secrets"}
    assert not roots & impure, roots & impure
