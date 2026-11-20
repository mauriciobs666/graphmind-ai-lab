"""`identity`: read-time resolution of stored-raw ID tokens through `wilib` (plan3 sections 3.1, 3.4).

The registry is a **fake**: a real `wilib.events.fold` over in-memory events, never the
registry's write path or a `.work-items` directory.
"""
import re

import pytest

from coordgraph.identity import Identity, Registry, load_wilib

NAMESPACES = {"K-": ["architect", "falkor-chat", "teco"], "C-": ["cpg"]}
_clock = iter(range(1, 100_000))


def ev(kind, **kw):
    return {"evid": f"e{next(_clock)}", "ts": next(_clock), "ev": kind, **kw}


def new(id_, type_, parent=None, legacy=None, home="repo"):
    extra = {"legacy": legacy} if legacy else {}
    return ev("new", id=id_, type=type_, parent=parent, home=home, **extra)


def move(uid, old, new_id, parent=None):
    return ev("move", uid=uid, parent=parent, moves=[{"uid": uid, "old": old, "new": new_id}])


def make_registry(*evs, unflushed=0, namespaces=NAMESPACES):
    load_wilib()
    from wilib import events

    return Registry(events.fold(list(evs)), namespaces, unflushed=unflushed)


def identity(*evs, **kw):
    return Identity(make_registry(*evs, **kw))


# GAL-I02 (kaizen) holds K items; falkor-chat:K-065 minted as GAL-I02-K0042.
KAIZEN = [new("GAL-I02", "I")]
MINTED_K065 = new("GAL-I02-K0042", "K", parent="GAL-I02", legacy={"home": "falkor-chat", "name": "K-065"})


class TestRouting:
    def test_bare_legacy_is_qualified_with_the_referrers_scope(self):
        r = identity(*KAIZEN, MINTED_K065).resolve("K-065", "falkor-chat")
        assert (r.state, r.route) == ("resolved", "legacy-bare")
        assert r.uid == "GAL-I02-K0042"

    def test_bare_legacy_in_another_scope_is_a_different_item(self):
        # architect:K-065 is declared but unminted; falkor-chat's mint must not leak
        r = identity(*KAIZEN, MINTED_K065).resolve("K-065", "architect")
        assert r.state == "unminted"
        assert r.candidates == {"architect:K-065"}

    def test_qualified_legacy_is_resolved_as_written(self):
        r = identity(*KAIZEN, MINTED_K065).resolve("falkor-chat:K-065", "cpg")
        assert (r.state, r.route) == ("resolved", "legacy-qualified")

    def test_scheme_id_is_never_scope_qualified(self):
        r = identity(*KAIZEN, MINTED_K065).resolve("GAL-I02-K0042", "cpg")
        assert (r.state, r.route) == ("resolved", "scheme")
        assert not any(c.startswith("cpg:") for c in r.candidates)

    def test_scheme_requirement_path_resolves_through_its_feature(self):
        reg = (*KAIZEN, new("GAL-F0003", "F"))
        r = identity(*reg).resolve("GAL-F0003-FR4", "cpg")
        assert r.state == "resolved" and "GAL-F0003-FR4" in r.candidates

    def test_unit_reference_is_routed_only_in_a_needs_cell(self):
        idn = identity(*KAIZEN)
        assert idn.resolve("connected-code-graph:U7", "cpg", needs=True).route == "unit"
        # same text as a Depends-on token: not a legacy shape, not a needs cell -> unrecognised
        assert idn.resolve("connected-code-graph:U7", "cpg").state == "unknown"

    def test_a_qualified_legacy_token_in_needs_is_legacy_not_a_unit(self):
        r = identity(*KAIZEN, MINTED_K065).resolve("falkor-chat:K-065", "cpg", needs=True)
        assert r.route == "legacy-qualified"

    def test_free_text_is_unrecognised(self):
        r = identity(*KAIZEN).resolve("see the plan", "cpg")
        assert (r.state, r.route, r.reason) == ("unknown", None, "unrecognised token")
        assert r.candidates == {"see the plan"}


class TestExitMapping:
    def test_ambiguous_repeats_the_clis_candidate_list_verbatim(self):
        # no referrer scope: bare K-065 reaches wilib as written and exits 3
        r = identity(*KAIZEN, MINTED_K065).resolve("K-065", "unscoped")
        assert r.state == "ambiguous"
        for home in ("architect:K-065", "falkor-chat:K-065 -> GAL-I02-K0042", "teco:K-065"):
            assert home in r.reason

    def test_ambiguous_is_not_resolved_to_a_guessed_home(self):
        r = identity(*KAIZEN, MINTED_K065).resolve("K-065", "unscoped")
        assert r.uid is None and "GAL-I02-K0042" not in r.candidates

    def test_unminted_backlog_ref_compares_as_written_and_carries_the_mint_text(self):
        r = identity(*KAIZEN).resolve("falkor-chat:K-065", "cpg")
        assert r.state == "unminted"
        assert r.candidates == {"falkor-chat:K-065"}
        assert "wi mint-legacy falkor-chat:K-065" in r.reason

    def test_unminted_bare_ref_names_the_mint_command_too(self):
        r = identity(*KAIZEN).resolve("C-310", "cpg")
        assert r.state == "unminted"
        assert r.candidates == {"cpg:C-310"}
        assert "wi mint-legacy cpg:C-310" in r.reason

    def test_unknown_scheme_id_is_unknown_and_compares_as_written(self):
        r = identity(*KAIZEN).resolve("GAL-F0042", "cpg")
        assert r.state == "unknown" and "GAL-F0042" in r.candidates

    def test_undeclared_home_is_unknown(self):
        assert identity(*KAIZEN).resolve("bogus:K-1", "cpg").state == "unknown"

    def test_retired_item_is_retired_with_its_candidates_kept(self):
        r = identity(*KAIZEN, MINTED_K065, ev("retire", uid="GAL-I02-K0042")).resolve("GAL-I02-K0042", "cpg")
        assert r.state == "retired" and "falkor-chat:K-065" in r.candidates

    def test_registry_unavailable_compares_as_written_qualified_when_the_scope_qualified_it(self):
        r = Identity(None).resolve("K-065", "falkor-chat")
        assert r.state == "unavailable"
        assert r.candidates == {"falkor-chat:K-065"}
        assert "unavailable" in r.reason
        assert Identity(None).resolve("K-065", "unscoped").candidates == {"K-065"}

    def test_unknown_bare_legacy_keeps_only_its_qualified_form(self):
        r = identity(*KAIZEN).resolve("K-065", "bogus")  # `bogus` is not a declared home
        assert r.state == "unknown" and r.candidates == {"bogus:K-065"}


class TestPlantedRename:
    """Plan3 section 6 test 6: identity survives `wi move` and minting without any stored string changing."""

    def moved(self):
        # GAL-F0007 moved under a new initiative becomes GAL-I03-F0007
        return identity(*KAIZEN, new("GAL-I03", "I"), new("GAL-F0007", "F"),
                        move("GAL-F0007", "GAL-F0007", "GAL-I03-F0007"))

    def test_old_id_and_new_id_share_candidates_after_a_move(self):
        idn = self.moved()
        old, cur = idn.resolve("GAL-F0007", "cpg"), idn.resolve("GAL-I03-F0007", "cpg")
        assert old.candidates & cur.candidates
        assert {"GAL-F0007", "GAL-I03-F0007"} <= old.candidates

    def test_old_id_written_in_depends_on_joins_an_item_stored_under_the_new_id(self):
        idn = self.moved()
        assert "GAL-I03-F0007" in idn.candidates("GAL-F0007", "cpg")  # Item.ref seek

    def test_old_id_in_a_header_item_joins_the_new_one_symmetrically(self):
        idn = self.moved()
        assert idn.candidates("GAL-I03-F0007", "cpg") & {"GAL-F0007"}

    def test_legacy_dependency_joins_the_item_headed_with_its_scheme_id_once_minted(self):
        idn = identity(*KAIZEN, MINTED_K065)
        assert "GAL-I02-K0042" in idn.candidates("falkor-chat:K-065", "cpg")
        assert "GAL-I02-K0042" in idn.candidates("K-065", "falkor-chat")  # bare, referrer scope

    def test_scheme_reference_joins_the_item_still_headed_with_the_legacy_name(self):
        idn = identity(*KAIZEN, MINTED_K065)
        assert "falkor-chat:K-065" in idn.candidates("GAL-I02-K0042", "cpg")

    def test_one_number_minted_in_two_scopes_gives_disjoint_candidates(self):
        """Plan3 section 3.1 step 3 / test 9: two distinct items never share a candidate."""
        cobb = new("GAL-I02-K0043", "K", parent="GAL-I02", legacy={"home": "teco", "name": "K-065"})
        idn = identity(*KAIZEN, MINTED_K065, cobb)
        a, b = idn.candidates("K-065", "falkor-chat"), idn.candidates("K-065", "teco")
        assert a & b == set()
        assert "K-065" not in a and "K-065" not in b
        assert a == {"GAL-I02-K0042", "falkor-chat:K-065"} and b == {"GAL-I02-K0043", "teco:K-065"}

    def test_bare_without_a_scope_keeps_itself_and_is_ambiguous(self):
        r = identity(*KAIZEN, MINTED_K065).resolve("K-065", "unscoped")
        assert r.state == "ambiguous" and r.candidates == {"K-065"}

    def test_a_stored_bare_string_no_longer_joins_a_qualified_reference(self):
        # ruling consequence: the stored bare `K-065` is not a candidate of `falkor-chat:K-065`
        idn = identity(*KAIZEN, MINTED_K065)
        assert "K-065" not in idn.candidates("falkor-chat:K-065", "cpg")

    def test_unminted_legacy_does_not_join_the_scheme_world(self):
        idn = identity(*KAIZEN)
        assert idn.candidates("falkor-chat:K-065", "cpg") == {"falkor-chat:K-065"}

    def test_cross_backlog_scheme_id_joins_an_item_in_another_backlog_never_qualified(self):
        # AC-13: a cpg backlog depends on a falkor-chat item by its scheme ID
        idn = identity(*KAIZEN, MINTED_K065)
        cands = idn.candidates("GAL-I02-K0042", "cpg")
        assert "GAL-I02-K0042" in cands and "cpg:GAL-I02-K0042" not in cands


class TestRegistryLine:
    def test_counts_items_last_event_date_and_unflushed_events(self):
        # ts is nanoseconds since the epoch (wilib); 2026-11-20T13:00Z
        last = dict(ev("set", uid="GAL-I02", attr="title", value="x"), ts=1_795_179_600_000_000_000)
        idn = identity(*KAIZEN, MINTED_K065, last, unflushed=2)
        assert idn.registry_line() == "Registry: 2 items · last event 2026-11-20 · 2 unflushed journal events"

    def test_unavailable_registry_line_says_so(self):
        assert Identity(None).registry_line() == "Registry: unavailable — identities compared as written"


@pytest.fixture
def scratch(tmp_path):
    """A scratch git repo with its own `wi init` (never the real registry)."""
    import json
    import subprocess

    load_wilib()
    from wilib import project

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    project.init(tmp_path, "SCR", ["falkor-chat", "repo"])
    pj = tmp_path / ".work-items" / "project.json"
    data = json.loads(pj.read_text())
    data["legacy_namespaces"] = {"K-": ["falkor-chat", "teco"]}
    pj.write_text(json.dumps(data))
    return tmp_path


class TestLoad:
    def test_loads_the_registry_of_a_repo_and_resolves_through_it(self, scratch):
        from coordgraph.identity import load_identity
        from wilib import legacy

        legacy.mint(scratch, "falkor-chat:K-065", "K")
        idn = load_identity(scratch)
        r = idn.resolve("falkor-chat:K-065", "cpg")
        assert r.state == "resolved" and r.uid.startswith("SCR-")

    def test_counts_unflushed_journal_events(self, scratch):
        from coordgraph.identity import load_identity
        from wilib import legacy

        legacy.mint(scratch, "falkor-chat:K-065", "K")  # journal only: not flushed
        assert "1 unflushed journal events" in load_identity(scratch).registry_line()

    def test_a_repo_without_a_registry_is_unavailable(self, tmp_path):
        from coordgraph.identity import load_identity

        idn = load_identity(tmp_path)
        assert idn.registry_line().startswith("Registry: unavailable")
        r = idn.resolve("GAL-F0042", "cpg")
        assert r.state == "unavailable" and r.candidates == {"GAL-F0042"}
        assert ".work-items/project.json" in r.reason

    def test_a_bad_wilib_root_is_unavailable_not_an_error(self, scratch):
        import subprocess
        import sys
        import textwrap

        code = textwrap.dedent(f"""
            from coordgraph.identity import load_identity
            idn = load_identity({str(scratch)!r})
            print(idn.registry_line())
        """)
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             env={"COORDGRAPH_WI_ROOT": "/nonexistent", "PYTHONPATH": ".", "PATH": ""},
                             cwd=str(__import__("pathlib").Path(__file__).resolve().parents[1]))
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() == "Registry: unavailable — identities compared as written"


# ---------------------------------------------------------------- needs (plan3 section 3.4)
from coordgraph.identity import LedgerRef, LedgerRow, UnitRow, resolve_need  # noqa: E402


def unit(path, key, status="queued", item_id=None, label=None, text=None):
    slug = path.rsplit("/", 1)[-1].removesuffix("-coordination.md")
    base = re.sub(r"~\d+$", "", key)
    return UnitRow(f"{path}#{key}", path, slug, base, label or base.split()[0], status, item_id, text)


class FakeGraph:
    """The four reads of graph note 7.3/7.6, answered from fixed rows."""

    def __init__(self, units=(), ledgers=()):
        self.units, self.ledgers = list(units), list(ledgers)

    def units_with_item(self, cands):
        return [u for u in self.units if u.item_id in cands]

    def ledgers_carrying(self, cands):
        return [l for l in self.ledgers if l.item_id in cands]

    def units_matching(self, slug, u):
        return [x for x in self.units if x.slug == slug and u in (x.unit_key, x.unit_label)]

    def units_in_ledger(self, path, u):
        return [x for x in self.units if x.ledger_path == path and u in (x.unit_key, x.unit_label)]


ME = LedgerRef("claude/docs/plans/mine-coordination.md", "mine", "cpg")
TASKS = [new("GAL-F0002", "F"), new("GAL-F0002-T01", "T", parent="GAL-F0002"),
         new("GAL-F0002-T02", "T", parent="GAL-F0002")]


def need(token, graph, *evs, ledger=ME, read_cell=None):
    return resolve_need(token, ledger, identity(*KAIZEN, *TASKS, *evs), graph, read_cell)


class TestTaskNeed:
    OTHER = "falkor-chat/docs/plans/other-coordination.md"

    def test_accepted_unit_is_met_even_in_another_ledger(self):  # AC-4 variant D
        g = FakeGraph([unit(self.OTHER, "U1", "accepted", "GAL-F0002-T01")])
        r = need("GAL-F0002-T01", g)
        assert (r.state, r.matches) == ("met", (self.OTHER + "#U1",))

    @pytest.mark.parametrize("status", ["queued", "in-flight", "delivered", "gated", "paused", "abandoned"])
    def test_any_other_status_is_unmet(self, status):
        g = FakeGraph([unit(self.OTHER, "U1", status, "GAL-F0002-T01")])
        assert need("GAL-F0002-T01", g).state == "unmet"

    def test_need_cited_by_an_alias_after_a_move_is_met(self):  # AC-4 variant D'
        g = FakeGraph([unit(self.OTHER, "U1", "accepted", "GAL-F0002-T01")])
        r = need("GAL-F0002-T09", g, move("GAL-F0002-T01", "GAL-F0002-T01", "GAL-F0002-T09"))
        # the ledger still says T01, the need cites the new ID: candidates meet
        assert r.state == "met"
        r2 = need("GAL-F0002-T01", FakeGraph([unit(self.OTHER, "U1", "accepted", "GAL-F0002-T09")]),
                  move("GAL-F0002-T01", "GAL-F0002-T01", "GAL-F0002-T09"))
        assert r2.state == "met"

    def test_unknown_status_is_unresolvable_with_the_raw_cell_never_met_or_unmet(self):
        g = FakeGraph([unit(self.OTHER, "U1", "unknown", "GAL-F0002-T01", text="blocked (waiting)")])
        r = need("GAL-F0002-T01", g)
        assert r.state == "unresolvable"
        assert r.reason == f"status unknown ({self.OTHER}, blocked (waiting))"

    def test_unknown_status_on_an_archived_unit_reads_the_cell_from_the_file(self):
        g = FakeGraph([unit(self.OTHER, "U1", "unknown", "GAL-F0002-T01")])
        r = need("GAL-F0002-T01", g, read_cell=lambda u: "blocked (file)")
        assert r.reason == f"status unknown ({self.OTHER}, blocked (file))"

    def test_unknown_status_falls_back_to_no_cell_when_the_file_or_row_is_gone(self):
        g = FakeGraph([unit(self.OTHER, "U1", "unknown", "GAL-F0002-T01")])
        assert need("GAL-F0002-T01", g, read_cell=lambda u: None).reason == f"status unknown ({self.OTHER})"
        assert need("GAL-F0002-T01", g).reason == f"status unknown ({self.OTHER})"

    def test_no_unit_carrying_the_task_is_unresolvable(self):
        r = need("GAL-F0002-T01", FakeGraph())
        assert (r.state, r.reason) == ("unresolvable", "no unit carries GAL-F0002-T01")

    def test_two_units_carrying_the_task_are_listed_never_guessed(self):
        g = FakeGraph([unit(self.OTHER, "U1", "accepted", "GAL-F0002-T01"),
                       unit("a/b-coordination.md", "U9", "queued", "GAL-F0002-T01")])
        r = need("GAL-F0002-T01", g)
        assert r.state == "unresolvable" and r.reason.startswith("ambiguous:")
        assert "a/b-coordination.md#U9" in r.reason and self.OTHER + "#U1" in r.reason

    def test_a_minted_legacy_task_is_found_by_its_scheme_id(self):
        g = FakeGraph([unit(self.OTHER, "U1", "accepted", "GAL-I02-K0042")])
        assert need("falkor-chat:K-065", g, MINTED_K065).state == "met"

    def test_a_task_id_is_never_looked_up_as_a_feature(self):
        g = FakeGraph(ledgers=[LedgerRow("p", "s", "archived", 0, 0, item_id="GAL-F0002-T01")])
        assert need("GAL-F0002-T01", g).state == "unresolvable"


class TestFeatureNeed:
    def ledger(self, path, doc_status, units, accepted):
        return LedgerRow(path, path, doc_status, units, accepted, item_id="GAL-F0002")

    def test_met_when_every_carrying_ledger_is_archived_even_with_abandoned_units(self):
        g = FakeGraph(ledgers=[self.ledger("a", "archived", 3, 1), self.ledger("b", "archived", 0, 0)])
        assert need("GAL-F0002", g).state == "met"

    def test_met_when_a_non_archived_ledger_has_all_units_accepted(self):
        assert need("GAL-F0002", FakeGraph(ledgers=[self.ledger("a", "active", 2, 2)])).state == "met"

    def test_abandoned_or_unknown_unit_in_a_non_archived_ledger_is_unmet(self):
        r = need("GAL-F0002", FakeGraph(ledgers=[self.ledger("a", "active", 3, 2)]))
        assert r.state == "unmet" and "a" in r.reason

    def test_a_non_archived_ledger_with_no_units_is_unmet_not_vacuously_met(self):
        assert need("GAL-F0002", FakeGraph(ledgers=[self.ledger("a", "active", 0, 0)])).state == "unmet"

    def test_one_unfinished_ledger_among_archived_ones_is_unmet(self):
        g = FakeGraph(ledgers=[self.ledger("a", "archived", 1, 1), self.ledger("b", "active", 2, 1)])
        r = need("GAL-F0002", g)
        assert r.state == "unmet" and r.matches == ("a", "b")

    def test_no_ledger_carrying_the_feature_is_unresolvable(self):
        r = need("GAL-F0002", FakeGraph())
        assert (r.state, r.reason) == ("unresolvable", "no ledger carries GAL-F0002")

    def test_feature_cited_by_alias_after_a_move_meets_the_ledgers_header(self):
        g = FakeGraph(ledgers=[self.ledger("a", "archived", 1, 1)])
        evs = (new("GAL-I03", "I"), move("GAL-F0002", "GAL-F0002", "GAL-I03-F0002"))
        assert need("GAL-I03-F0002", g, *evs).state == "met"


class TestUnresolvableTokens:
    def test_unminted_legacy_names_the_mint_command(self):
        r = need("falkor-chat:K-065", FakeGraph())
        assert r.state == "unresolvable" and r.reason.startswith("unminted:")
        assert "wi mint-legacy falkor-chat:K-065" in r.reason

    def test_ambiguous_bare_legacy_lists_the_homes(self):
        r = need("K-065", FakeGraph(), ledger=LedgerRef("p", "s", "unscoped"))
        assert r.state == "unresolvable" and r.reason.startswith("ambiguous:")
        assert "teco:K-065" in r.reason

    def test_bare_legacy_uses_the_ledgers_scope(self):
        g = FakeGraph([unit("x/y-coordination.md", "U1", "accepted", "GAL-I02-K0042")])
        assert need("K-065", g, MINTED_K065, ledger=LedgerRef("p", "s", "falkor-chat")).state == "met"

    def test_unknown_id_is_unresolvable_with_the_reason(self):
        r = need("GAL-F0042", FakeGraph())
        assert r.state == "unresolvable" and r.reason.startswith("unknown:")

    def test_retired_item_is_unresolvable(self):
        r = need("GAL-I02-K0042", FakeGraph(), MINTED_K065, ev("retire", uid="GAL-I02-K0042"))
        assert r.state == "unresolvable" and r.reason.startswith("retired")

    def test_registry_unavailable_is_unresolvable_not_met_or_unmet(self):
        g = FakeGraph([unit("x/y-coordination.md", "U1", "accepted", "GAL-F0002-T01")])
        r = resolve_need("GAL-F0002-T01", ME, Identity(None), g)
        assert r.state == "unresolvable" and "registry unavailable" in r.reason

    def test_free_text_is_unresolvable(self):
        r = need("when the moon is full", FakeGraph())
        assert (r.state, r.reason) == ("unresolvable", "unrecognised token")


class TestUnitNeed:
    P = "connected-code-graph/docs/plans/connected-code-graph-coordination.md"

    def graph(self, *units):
        return FakeGraph(list(units))

    def test_slug_unit_accepted_is_met_and_queued_unmet(self):
        met = need("connected-code-graph:U7", self.graph(unit(self.P, "U7", "accepted")))
        unmet = need("connected-code-graph:U7", self.graph(unit(self.P, "U7", "queued")))
        assert (met.state, unmet.state) == ("met", "unmet")

    def test_exact_key_wins_over_label_matches(self):
        g = self.graph(unit(self.P, "S6", "accepted"), unit(self.P, "S6 gate", "queued"))
        r = need("connected-code-graph:S6", g)
        assert (r.state, r.matches) == ("met", (self.P + "#S6",))

    def test_every_repeat_of_the_exact_key_is_a_match_so_it_is_ambiguous(self):
        g = self.graph(unit(self.P, "U32", "accepted"), unit(self.P, "U32~2", "queued"))
        r = need("connected-code-graph:U32", g)
        assert r.state == "unresolvable" and r.reason.startswith("ambiguous:")
        assert self.P + "#U32" in r.reason and self.P + "#U32~2" in r.reason

    def test_tilde_n_addresses_one_repeat_by_key(self):
        g = self.graph(unit(self.P, "U32", "accepted"), unit(self.P, "U32~2", "queued"))
        r = need("connected-code-graph:U32~2", g)
        assert (r.state, r.matches) == ("unmet", (self.P + "#U32~2",))

    def test_label_only_matches_are_all_listed_when_no_exact_key_exists(self):
        g = self.graph(unit(self.P, "U1 K-091 band", "accepted"), unit(self.P, "U1 (LM Studio fix)", "queued"))
        r = need("connected-code-graph:U1", g)
        assert r.state == "unresolvable" and r.reason.count("#U1") == 2

    def test_a_label_alone_resolves_when_only_one_row_carries_it(self):
        r = need("connected-code-graph:U1", self.graph(unit(self.P, "U1 K-091 band", "accepted")))
        assert r.state == "met"

    def test_same_slug_in_two_components_lists_both_paths(self):
        other = "kiro/docs/plans/connected-code-graph-coordination.md"
        g = self.graph(unit(self.P, "U7", "accepted"), unit(other, "U7", "accepted"))
        r = need("connected-code-graph:U7", g)
        assert r.state == "unresolvable" and self.P in r.reason and other in r.reason

    def test_unknown_unit_is_unresolvable(self):
        r = need("connected-code-graph:U99", self.graph(unit(self.P, "U7", "accepted")))
        assert r.state == "unresolvable" and "U99" in r.reason

    def test_unit_with_unknown_status_is_unresolvable(self):
        r = need("connected-code-graph:U7", self.graph(unit(self.P, "U7", "unknown", text="blocked (x)")))
        assert (r.state, r.reason) == ("unresolvable", f"status unknown ({self.P}, blocked (x))")

    def test_bare_label_is_a_unit_of_the_referring_ledger_only(self):
        mine = unit(ME.path, "U3", "accepted")
        theirs = unit(self.P, "U3", "queued")
        assert need("U3", self.graph(mine, theirs)).state == "met"

    def test_bare_token_matching_no_local_unit_is_unrecognised(self):
        r = need("U3", self.graph(unit(self.P, "U3", "accepted")))
        assert (r.state, r.reason) == ("unresolvable", "unrecognised token")

    def test_unit_slug_form_beats_nothing_when_token_is_a_legacy_name(self):
        # falkor-chat:K-065 has the slug:unit shape but is a legacy name (routing (b) first)
        g = self.graph(unit(self.P, "K-065", "accepted"))
        assert need("falkor-chat:K-065", g).reason.startswith("unminted")


class TestUnitNeedsStatus:
    """AC-3 / AC-4: what a unit's whole `Needs` cell says about its readiness."""

    OTHER = "falkor-chat/docs/plans/other-coordination.md"

    def status(self, needs, error=None, graph=None):
        from coordgraph.identity import needs_status

        g = graph or FakeGraph([unit(self.OTHER, "U1", "accepted", "GAL-F0002-T01"),
                                unit(self.OTHER, "U2", "queued", "GAL-F0002-T02")])
        return needs_status(needs, error, ME, identity(*KAIZEN, *TASKS), g)

    def test_absent_cell_is_not_recorded_and_never_ready(self):
        s = self.status(None)
        assert (s.state, s.reasons) == ("not-recorded", ("needs not recorded",))

    def test_malformed_cell_lists_what_is_missing(self):  # AC-3
        s = self.status(None, "empty token between commas")
        assert s.state == "not-recorded"
        assert s.reasons == ("needs not recorded: empty token between commas",)

    def test_none_cell_is_ready(self):
        assert self.status([]).state == "ready"

    def test_all_needs_met_is_ready_including_another_ledger(self):  # AC-4 B/D
        assert self.status(["GAL-F0002-T01"]).state == "ready"

    def test_one_unmet_need_blocks(self):  # AC-4 C
        s = self.status(["GAL-F0002-T01", "GAL-F0002-T02"])
        assert s.state == "blocked" and any("GAL-F0002-T02" in r for r in s.reasons)

    def test_a_need_that_cannot_be_found_is_unresolvable(self):  # AC-4 E
        s = self.status(["GAL-F0002-T01", "nowhere:U1"])
        assert s.state == "unresolvable" and any("nowhere:U1" in r for r in s.reasons)

    def test_a_certainly_unmet_need_blocks_even_beside_an_unresolvable_one(self):
        assert self.status(["GAL-F0002-T02", "nowhere:U1"]).state == "blocked"


class TestLintSeam:
    """`lint_repo` takes the identity's candidates and its reason, so an unresolvable
    `Depends on` says *why* (unminted / unknown / ambiguous / registry unavailable)."""

    @pytest.fixture
    def repo(self, tmp_path):
        import json

        (tmp_path / ".work-items").mkdir()
        (tmp_path / ".work-items" / "project.json").write_text(json.dumps(
            {"key": "GAL", "legacy_paths": {"*/docs/BACKLOG.md": "$1"}}))
        (tmp_path / "falkor-chat" / "docs").mkdir(parents=True)
        (tmp_path / "falkor-chat" / "docs" / "BACKLOG.md").write_text(
            "# B\n\n### K-002 — T\n\n- **Status:** 🔵 proposed\n- **Depends on:** falkor-chat:K-065\n")
        return tmp_path

    def messages(self, repo, idn):
        from coordgraph.lint import lint_repo

        return [f.message for f in lint_repo(repo, ["falkor-chat/docs/BACKLOG.md"], idn.candidates,
                                             reason=idn.reason)
                if f.code == "unresolvable-dependency"]

    def test_an_unminted_dependency_reports_the_mint_command(self, repo):
        (msg,) = self.messages(repo, identity(*KAIZEN))
        assert "wi mint-legacy falkor-chat:K-065" in msg

    def test_a_degraded_registry_is_named_in_the_finding(self, repo):
        (msg,) = self.messages(repo, Identity(None, why="gone"))
        assert "registry unavailable (gone)" in msg

    def test_a_resolved_dependency_has_no_reason_and_a_renamed_one_joins(self, repo):
        idn = identity(*KAIZEN, MINTED_K065)
        assert idn.reason("falkor-chat:K-065", "falkor-chat") is None
        (repo / "falkor-chat" / "docs" / "BACKLOG.md").write_text(
            "# B\n\n### GAL-I02-K0042 — T\n\n- **Status:** 🔵 proposed\n\n"
            "### K-002 — T\n\n- **Status:** 🔵 proposed\n- **Depends on:** falkor-chat:K-065\n")
        assert self.messages(repo, idn) == []

    def test_without_a_reason_callable_lint_is_unchanged(self, repo):
        from coordgraph.lint import lint_repo

        f = [f for f in lint_repo(repo, ["falkor-chat/docs/BACKLOG.md"])
             if f.code == "unresolvable-dependency"]
        assert len(f) == 1 and "wi mint-legacy" not in f[0].message


class TestSelectUnits:
    """The one pure lookup rule shared by `resolve_need` and the `unit` view (plan3 3.4 `Unit label`)."""

    P = "x/docs/plans/s-coordination.md"

    def rows(self, *keys):
        return [unit(self.P, k) for k in keys]

    def sel(self, rows, ref):
        from coordgraph.identity import select_units

        return [u.unit_key for u in select_units(rows, ref)]

    def test_exact_key_wins(self):
        assert self.sel(self.rows("S6", "S6 gate"), "S6") == ["S6"]

    def test_label_matches_when_no_exact_key(self):
        assert self.sel(self.rows("U1 K-091 band", "U2"), "U1") == ["U1 K-091 band"]

    def test_label_only_ambiguity_returns_every_holder(self):
        rows = self.rows("Pass 2 re-gate", "Pass 3 final")
        assert self.sel(rows, "Pass") == ["Pass 2 re-gate", "Pass 3 final"]

    def test_a_trailing_tilde_n_picks_one_repeat_by_key(self):
        from coordgraph.identity import select_units

        rows = self.rows("U32", "U32~2")
        assert [u.key for u in select_units(rows, "U32~2")] == [self.P + "#U32~2"]

    def test_a_tilde_inside_a_real_key_is_not_a_repeat_suffix(self):
        rows = [unit(self.P, "U7 (~80 tool calls)")]
        assert self.sel(rows, "U7") == ["U7 (~80 tool calls)"]

    def test_query_key_strips_only_a_trailing_tilde_n(self):
        from coordgraph.identity import unit_query_key

        assert unit_query_key("U32~2") == "U32"
        assert unit_query_key("U7 (~80 tool calls)") == "U7 (~80 tool calls)"
        assert unit_query_key("A~1x") == "A~1x"

    def test_a_key_that_really_ends_in_tilde_n_is_found_exactly(self):
        rows = [UnitRow(self.P + "#V~3", self.P, "s", "V~3", "V~3", "queued")]
        from coordgraph.identity import select_units

        assert [u.key for u in select_units(rows, "V~3")] == [self.P + "#V~3"]

    def test_local_token_anchors_on_the_referring_ledger_path_not_the_slug(self):
        mine = ME.path
        twin = "kiro/docs/plans/mine-coordination.md"  # same slug, other component
        g = FakeGraph([unit(mine, "U3", "accepted"), unit(twin, "U3", "queued")])
        r = need("U3", g)
        assert (r.state, r.matches) == ("met", (mine + "#U3",))


class TestSyncPathStaysOffline:
    """Plan3 section 6 test 12: hook-time sync imports neither `identity` nor `wilib`."""

    SYNC = ["coordgraph.discover", "coordgraph.keys", "coordgraph.config", "coordgraph.scope",
            "coordgraph.history", "coordgraph.migrate", "coordgraph.shapes",
            "coordgraph.parse.backlog", "coordgraph.parse.ledger", "coordgraph.parse.review",
            "coordgraph.parse.requirements", "coordgraph.parse.kb", "coordgraph.parse.header"]

    def test_importing_the_sync_modules_loads_neither_identity_nor_wilib(self):
        import subprocess
        import sys
        from pathlib import Path

        code = ("import sys\n" + "".join(f"import {m}\n" for m in self.SYNC)
                + "bad = [m for m in ('coordgraph.identity', 'wilib') if m in sys.modules]\n"
                  "assert not bad, bad\n")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             env={"COORDGRAPH_WI_ROOT": "/nonexistent", "PYTHONPATH": ".", "PATH": ""},
                             cwd=str(Path(__file__).resolve().parents[1]))
        assert out.returncode == 0, out.stderr
