"""The read views over a fake `ro_query` graph (offline): which statement runs with which parameters, how the
rows are selected through `identity.select_units`, how identity joins are built, and how each answer renders.
The Cypher itself is exercised live (`test_views_live.py`). The fake has **no `query` method**: a view that
reached for a non-read-only call would raise.
"""
import subprocess
import sys
import textwrap

import pytest

from coordgraph import views
from coordgraph.cli import build_parser, main
from coordgraph.identity import Identity
from tests.test_identity import KAIZEN, MINTED_K065, identity, move, new

X = "a/docs/plans/x-coordination.md"


class RoResult:
    def __init__(self, rows):
        self.header = [[1, name] for name in (rows[0] if rows else {})]
        self.result_set = [list(r.values()) for r in rows]


class FakeGraph:
    """`ro_query` answers from `{statement: rows | callable(params) -> rows}`; every call is recorded."""

    def __init__(self, answers=None):
        self.answers = answers or {}
        self.calls: list[tuple[str, dict]] = []

    def ro_query(self, q, params=None):
        params = params or {}
        self.calls.append((q, params))
        ans = self.answers.get(q, [])
        return RoResult(ans(params) if callable(ans) else ans)

    def ran(self, q):
        return [p for c, p in self.calls if c == q]


def unit_row(key, unit_key, label=None, status="queued", path=X, **kw):
    return {"path": path, "docStatus": kw.pop("docStatus", "active"), "fileSha": "s", "syncedAt": 1, "key": key,
            "unitKey": unit_key, "unitLabel": label or unit_key.split()[0], "status": status,
            "statusText": kw.pop("statusText", None), "agentId": kw.pop("agentId", None),
            "deliverable": kw.pop("deliverable", None), "question": kw.pop("question", None),
            "askedOn": kw.pop("askedOn", None), "rowHash": "h"}


def by_unit_param(rows):
    """What the engine does for `WHERE u.unitKey = $unit OR u.unitLabel = $unit`."""
    return lambda p: [r for r in rows if p["unit"] in (r["unitKey"], r["unitLabel"])]


# --- unit: the lookup rule and the identity joins ------------------------------------------------------------

class TestUnitLookup:
    def test_exact_key_wins_over_a_label_sibling(self):
        rows = [unit_row(f"{X}#S6", "S6"), unit_row(f"{X}#S6 gate", "S6 gate", "S6")]
        g = FakeGraph({views.Q_UNIT_SLUG: by_unit_param(rows)})
        hits = views.find_units(g, identity(), "x", "S6").hits
        assert [h.key for h in hits] == [f"{X}#S6"]
        assert [h.key for h in views.find_units(g, identity(), "x", "S6 gate").hits] == [f"{X}#S6 gate"]

    def test_a_label_with_no_exact_key_returns_every_label_match_in_table_order(self):
        rows = [unit_row(f"{X}#U1 a", "U1 a", "U1"), unit_row(f"{X}#U1 b", "U1 b", "U1")]
        g = FakeGraph({views.Q_UNIT_SLUG: by_unit_param(rows)})
        assert [h.key for h in views.find_units(g, identity(), "x", "U1").hits] == [f"{X}#U1 a", f"{X}#U1 b"]

    def test_tilde_n_is_tried_as_written_then_stripped_and_picks_one_repeat(self):
        rows = [unit_row(f"{X}#U32", "U32"), unit_row(f"{X}#U32~2", "U32")]
        g = FakeGraph({views.Q_UNIT_SLUG: by_unit_param(rows)})
        hits = views.find_units(g, identity(), "x", "U32~2").hits
        assert [h.key for h in hits] == [f"{X}#U32~2"]
        assert [p["unit"] for p in g.ran(views.Q_UNIT_SLUG)] == ["U32~2", "U32"]  # whole token first, then stripped

    def test_a_key_that_really_ends_in_tilde_digits_is_found_on_the_first_try(self):
        rows = [unit_row(f"{X}#U ~80", "U ~80", "U")]
        g = FakeGraph({views.Q_UNIT_SLUG: by_unit_param(rows)})
        assert [h.key for h in views.find_units(g, identity(), "x", "U ~80").hits] == [f"{X}#U ~80"]
        assert len(g.ran(views.Q_UNIT_SLUG)) == 1

    def test_the_unit_argument_is_cleaned_like_the_stored_unit_key(self):
        g = FakeGraph({views.Q_UNIT_SLUG: by_unit_param([unit_row(f"{X}#U3", "U3")])})
        assert views.find_units(g, identity(), "x", "**U3** — note").hits
        assert g.ran(views.Q_UNIT_SLUG) == [{"slug": "x", "unit": "U3"}]

    def test_no_unit_argument_reads_every_unit_without_a_filter(self):
        g = FakeGraph({views.Q_UNIT_SLUG_ALL: [unit_row(f"{X}#U1", "U1"), unit_row(f"{X}#U2", "U2")]})
        assert len(views.find_units(g, identity(), "x").hits) == 2
        assert g.ran(views.Q_UNIT_SLUG) == []

    def test_a_plain_slug_never_consults_identity(self):
        class Boom(Identity):
            def resolve(self, *a, **k):
                raise AssertionError("slug lookups must not touch the registry")

        g = FakeGraph({views.Q_UNIT_SLUG: by_unit_param([unit_row(f"{X}#U3", "U3")])})
        assert len(views.find_units(g, Boom(None), "connected-code-graph", "U3").hits) == 1

    def test_an_id_argument_joins_on_every_candidate_old_alias_new_and_uid(self):
        idn = identity(*KAIZEN, new("GAL-I04", "I"), new("GAL-I02-F0001", "F", parent="GAL-I02"),
                       move("GAL-I02-F0001", "GAL-I02-F0001", "GAL-I04-F0001", parent="GAL-I04"))
        g = FakeGraph()
        for token in ("GAL-I02-F0001", "GAL-I04-F0001"):  # the old id and the new one: one identity
            g.calls.clear()
            views.find_units(g, idn, token)
            (carry,) = g.ran(views.Q_CARRIERS)
            assert {"GAL-I02-F0001", "GAL-I04-F0001"} <= set(carry["candidates"])
            assert g.ran(views.Q_UNIT_TASK) == [carry]

    def test_ledgers_carrying_the_id_are_read_by_path_and_their_units_united_by_key(self):
        idn = identity(*KAIZEN, new("GAL-I02-F0001", "F", parent="GAL-I02"))
        carriers = [{"path": X, "slug": "x", "itemId": "GAL-I02-F0001"}]
        row = unit_row(f"{X}#U1", "U1")
        g = FakeGraph({views.Q_CARRIERS: carriers, views.Q_UNIT_PATH_ALL: [row], views.Q_UNIT_TASK: [row],
                       views.Q_UNIT_SLUG_ALL: [row]})
        hits = views.find_units(g, idn, "GAL-I02-F0001").hits
        assert [h.key for h in hits] == [f"{X}#U1"]  # one row reached three ways: shown once
        assert g.ran(views.Q_UNIT_PATH_ALL) == [{"path": X}]

    def test_a_unit_inside_the_ledgers_of_an_id_uses_the_filtered_path_read(self):
        idn = identity(*KAIZEN, new("GAL-I02-F0001", "F", parent="GAL-I02"))
        rows = [unit_row(f"{X}#S1", "S1")]
        g = FakeGraph({views.Q_CARRIERS: [{"path": X, "slug": "x", "itemId": "i"}],
                       views.Q_UNIT_PATH: by_unit_param(rows)})
        assert [h.key for h in views.find_units(g, idn, "GAL-I02-F0001", "S1").hits] == [f"{X}#S1"]
        assert g.ran(views.Q_UNIT_PATH) == [{"path": X, "unit": "S1"}]

    def test_an_unresolved_id_says_why_and_still_tries_what_was_written(self):
        g = FakeGraph()
        out = views.find_units(g, identity(*KAIZEN), "GAL-F9999")
        assert out.hits == [] and out.notes and out.notes[0].startswith("GAL-F9999: unknown")
        assert g.ran(views.Q_CARRIERS) == [{"candidates": ["GAL-F9999"]}]

    def test_an_ambiguous_bare_legacy_id_is_reported_never_guessed(self):
        other = new("GAL-I02-K0043", "K", parent="GAL-I02", legacy={"home": "teco", "name": "K-065"})
        out = views.find_units(FakeGraph(), identity(*KAIZEN, MINTED_K065, other), "K-065")
        assert out.notes and "ambiguous" in out.notes[0]

    def test_an_unavailable_registry_compares_as_written_and_the_slug_read_still_runs(self):
        g = FakeGraph({views.Q_UNIT_SLUG_ALL: [unit_row(f"{X}#U1", "U1")]})
        out = views.find_units(g, Identity(None, why="no registry"), "GAL-F0002")
        assert out.notes[0].startswith("GAL-F0002: unavailable")
        assert g.ran(views.Q_CARRIERS) == [{"candidates": ["GAL-F0002"]}] and g.ran(views.Q_UNIT_SLUG_ALL)

    def test_views_only_call_ro_query(self):
        g = FakeGraph()
        assert not hasattr(g, "query")
        views.find_units(g, identity(), "x", "U1")
        views.inflight(g, identity())
        views.planned(g, "a")


# --- inflight: picked-up through identity ----------------------------------------------------------------------------

def req(path, item=None):
    return {"component": "a", "path": path, "status": "Interviewing", "itemId": item, "syncedAt": 1}


class TestInflightPickedUp:
    def idn(self):
        return identity(*KAIZEN, new("GAL-I04", "I"), new("GAL-I02-F0001", "F", parent="GAL-I02"),
                        move("GAL-I02-F0001", "GAL-I02-F0001", "GAL-I04-F0001", parent="GAL-I04"))

    def test_a_requirement_whose_item_is_carried_by_a_ledger_under_another_name_is_picked_up(self):
        g = FakeGraph({views.Q_REQUIREMENTS: [req("a/docs/requirements/w.md", "GAL-I04-F0001"),
                                              req("a/docs/requirements/q.md")],
                       views.Q_CARRIERS: [{"path": X, "slug": "x", "itemId": "GAL-I02-F0001"}]})
        view = views.inflight(g, self.idn())
        assert [r["path"] for r in view.requirements] == ["a/docs/requirements/q.md"]

    def test_the_carrier_read_gets_the_union_of_the_requirements_candidates_and_is_skipped_without_items(self):
        g = FakeGraph({views.Q_REQUIREMENTS: [req("a/q.md", "GAL-I04-F0001")]})
        views.inflight(g, self.idn())
        (call,) = g.ran(views.Q_CARRIERS)
        assert {"GAL-I02-F0001", "GAL-I04-F0001"} <= set(call["candidates"])
        g2 = FakeGraph({views.Q_REQUIREMENTS: [req("a/q.md")]})
        assert views.inflight(g2, self.idn()).requirements and g2.ran(views.Q_CARRIERS) == []

    def test_a_requirement_with_an_item_no_ledger_carries_stays_listed(self):
        g = FakeGraph({views.Q_REQUIREMENTS: [req("a/q.md", "GAL-I04-F0001")],
                       views.Q_CARRIERS: [{"path": X, "slug": "x", "itemId": "GAL-I09-F0009"}]})
        assert len(views.inflight(g, self.idn()).requirements) == 1


# --- planned ----------------------------------------------------------------------------------------------------------

def item(ref, marker, title="t", path="a/docs/BACKLOG.md"):
    return {"path": path, "syncedAt": 1, "ref": ref, "marker": marker, "title": title, "priority": None,
            "milestone": None}


class TestPlanned:
    def test_rejected_and_deferred_are_parked_everything_else_is_planned(self):
        g = FakeGraph({views.Q_BACKLOGS: [{"path": "a/docs/BACKLOG.md", "syncedAt": 1, "rowCount": 4}],
                       views.Q_PLANNED: [item("a:K-1", "🔵 proposed"), item("a:K-2", "🟡 in-progress"),
                                         item("a:K-3", "⚪ rejected"), item("a:K-4", "⚪ deferred"),
                                         item("a:K-5", "🔵 proposed — x")]})
        out = views.planned(g, "a")
        assert [i["ref"] for i in out.planned] == ["a:K-1", "a:K-2", "a:K-5"]
        assert [i["ref"] for i in out.parked] == ["a:K-3", "a:K-4"]

    @pytest.mark.parametrize("marker", ["deferred", "rejected — record", "⚪"])
    def test_a_status_written_with_only_the_word_or_only_the_emoji_is_still_parked(self, marker):
        g = FakeGraph({views.Q_BACKLOGS: [{"path": "p", "syncedAt": 1, "rowCount": 1}],
                       views.Q_PLANNED: [item("a:K-1", marker)]})
        out = views.planned(g, "a")
        assert out.planned == [] and len(out.parked) == 1

    def test_scope_is_normalised_and_an_unknown_one_lists_the_known_scopes(self):
        g = FakeGraph({views.Q_SCOPES: [{"scope": "cpg"}, {"scope": "a"}]})
        out = views.planned(g, "  nope ")
        assert g.ran(views.Q_BACKLOGS) == [{"scope": "nope"}] and g.ran(views.Q_PLANNED) == []
        assert out.scopes == ["cpg", "a"] and not out.sources


# --- rendering ---------------------------------------------------------------------------------------------

def inflight_rows():
    return views.Inflight(
        units=[{"component": "a", "path": X, "syncedAt": 1, "key": f"{X}#U1", "unit": "U1", "owner": "coder",
                "status": "accepted", "statusText": "**accepted**", "agentId": "ag1", "deliverable": "shipped"},
               {"component": "a", "path": X, "syncedAt": 1, "key": f"{X}#U2", "unit": "U2", "owner": None,
                "status": "unknown", "statusText": "blocked (K-9)", "agentId": None, "deliverable": None}],
        paused=[{"component": "a", "path": X, "key": f"{X}#U3", "unit": "U3", "statusText": "**relayed**",
                 "question": "which port?", "askedOn": "2026-11-01", "syncedAt": 1},
                {"component": "b", "path": "b/y.md", "key": "b/y.md#Q", "unit": "Q", "statusText": "paused",
                 "question": None, "askedOn": None, "syncedAt": 1}],
        reviews=[{"component": "a", "path": "a/docs/reviews/r.md", "pass": 2, "verdict": "needs changes",
                  "syncedAt": 1}],
        requirements=[{"component": "a", "path": "a/docs/requirements/q.md", "status": "Interviewing",
                       "itemId": None, "syncedAt": 1}])


class TestRender:
    def test_all_four_kinds_with_component_and_path(self):
        out = views.render_inflight(inflight_rows(), set())
        assert f"  a\t{X}" in out and "    U1\tcoder\taccepted\tag1\tshipped" in out
        assert f"  a\t{X}\tU3\tasked 2026-11-01\twhich port?" in out
        assert "  b\tb/y.md\tQ\tasked unknown\tpaused" in out
        assert "  a\ta/docs/reviews/r.md\tpass 2\tneeds changes" in out
        assert "  a\ta/docs/requirements/q.md\tInterviewing" in out

    def test_an_unknown_status_is_listed_with_its_raw_cell_never_dropped(self):
        assert "    U2\t-\tunknown [blocked (K-9)]\t-\t" in views.render_inflight(inflight_rows(), set())

    def test_waiting_is_the_paused_questions_only(self):
        out = views.render_inflight(inflight_rows(), set(), waiting=True)
        assert "U3\tasked 2026-11-01" in out
        assert "U1" not in out and "needs changes" not in out and "Requirements" not in out

    def test_a_drifted_row_is_marked_and_only_that_row(self):
        out = views.render_inflight(inflight_rows(), {(X, f"{X}#U2"), ("b/y.md", "b/y.md#Q")})
        marked = [line for line in out.splitlines() if line.endswith("\t⚠ drift")]
        assert [line.split("\t")[0].strip() for line in marked] == ["U2", "b"] and len(marked) == 2

    def test_an_empty_section_says_none(self):
        out = views.render_inflight(views.Inflight([], [], [], []), set())
        assert out.count("(none)") == 4

    def test_unit_shows_the_question_with_its_date_and_marks_drift_by_key(self):
        h = views.UnitHit(X, "active", 1, f"{X}#U3", "U3", "U3", "paused", "**relayed**", "ag", "d", "which port?",
                          "2026-11-01")
        out = views.render_unit("x", "U3", views.UnitLookup([h]), {(X, f"{X}#U3")})
        assert "question: which port? (asked 2026-11-01)" in out and "agent: ag" in out
        assert "status: paused [**relayed**]" in out and out.splitlines()[1].endswith("\t⚠ drift")
        assert "⚠" not in views.render_unit("x", "U3", views.UnitLookup([h]), {(X, "other#U3")})

    def test_an_archived_unit_shows_only_what_the_graph_keeps(self):
        h = views.UnitHit(X, "archived", 1, f"{X}#A1", "A1", "A1", "accepted")
        out = views.render_unit("x", "A1", views.UnitLookup([h]), set())
        assert f"{X} · docStatus: archived" in out and "archived ledger" in out
        assert "agent:" not in out and "deliverable:" not in out

    def test_several_matches_are_all_shown_with_a_count_never_guessed(self):
        hs = [views.UnitHit(X, "active", 1, f"{X}#P{i}", f"P{i}", "P", "queued") for i in (1, 2, 3)]
        assert "3 rows match" in views.render_unit("x", "P", views.UnitLookup(hs), set())

    def test_no_match_carries_the_identity_notes(self):
        out = views.render_unit("GAL-F1", None, views.UnitLookup([], ["GAL-F1: unknown: not in registry"]), set())
        assert out.splitlines() == ["GAL-F1: unknown: not in registry", "no unit matches GAL-F1"]

    def test_planned_groups_and_parks_records(self):
        res = views.Planned([{"path": "a/docs/BACKLOG.md", "syncedAt": 1, "rowCount": 3}],
                            [item("a:K-1", "🔵 proposed", "One")], [item("a:K-3", "⚪ rejected", "Three")])
        brief = views.render_planned(res, "a", set())
        assert "a:K-1\t🔵 proposed" in brief and "K-3\t" not in brief
        assert "Deferred / rejected: 1 (a:K-3); --all" in brief
        full = views.render_planned(res, "a", set(), show_all=True)
        assert "a:K-3\t⚪ rejected\tThree" in full


# --- the CLI seam ------------------------------------------------------------------------------------------

class TestCli:
    def test_the_three_subcommands_parse_with_their_arguments_and_graph_option(self):
        p = build_parser()
        assert p.parse_args(["inflight", "--waiting", "--graph", "g"]).waiting is True
        a = p.parse_args(["unit", "x", "U3"])
        assert (a.cmd, a.ref, a.unit) == ("unit", "x", "U3")
        assert p.parse_args(["unit", "GAL-F0002"]).unit is None
        b = p.parse_args(["planned", "a", "--all"])
        assert (b.scope, b.all) == ("a", True)

    def test_every_view_command_is_dispatched(self, monkeypatch):
        seen = []
        for name in views.COMMANDS:
            monkeypatch.setitem(views.COMMANDS, name, lambda args, n=name: seen.append(n) or 0)
        import coordgraph.cli as cli

        monkeypatch.setattr(cli.views, "COMMANDS", views.COMMANDS)
        for argv in (["inflight"], ["unit", "x"], ["planned", "a"]):
            assert main(argv) == 0
        assert seen == ["inflight", "unit", "planned"]

    @pytest.mark.parametrize("argv", [["inflight"], ["unit", "x", "U1"], ["planned", "a"]])
    def test_a_refused_endpoint_is_one_clear_line_and_exit_1(self, argv, monkeypatch, capsys):
        monkeypatch.delenv("COORDGRAPH_EPHEMERAL_PORT", raising=False)
        monkeypatch.setenv("COORDGRAPH_FALKORDB_PORT", "6379")
        assert main(argv) == 1
        cap = capsys.readouterr()
        assert cap.out == "" and cap.err.startswith(f"{argv[0]}: refusing") and "Traceback" not in cap.err

    def test_importing_the_cli_loads_neither_identity_nor_wilib(self):
        code = textwrap.dedent("""
            import sys
            import coordgraph.cli, coordgraph.views
            bad = [m for m in sys.modules if m in ("coordgraph.identity", "wilib") or m.startswith("wilib.")]
            print(bad)
        """)
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             env={"PYTHONPATH": ".", "PATH": ""}, check=True)
        assert out.stdout.strip() == "[]"
