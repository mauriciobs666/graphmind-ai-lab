import pytest

from coordgraph.parse.ledger import (
    map_columns,
    parse_ledger,
    parse_needs,
    parse_paused,
    unit_key_of,
)

HEADER = "# X coordination\n\n> **Status:** {status} · **Owner:** `teco` · **Tracks:** K-1\n\n"
COLS = "| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict | Cost |\n|---|---|---|---|---|---|---|\n"


def ledger(rows, status="active", cols=COLS, path="falkor-chat/docs/plans/x-coordination.md",
           reviews=None):
    text = HEADER.format(status=status) + cols + "\n".join(rows) + "\n"
    return parse_ledger(path, text.encode(), reviews)


# --- shape ----------------------------------------------------------------------------

def test_standard_row_maps_by_header_name():
    d = ledger(["| U1 — plan | `architect` | `a1b2` | accepted | `docs/plans/x.md` | `analyst` → approve | 10k / 3 |"])
    (u,) = d.units
    assert (u.unit, u.unit_key, u.owner, u.agent_id) == ("U1 — plan", "U1", "`architect`", "`a1b2`")
    assert (u.status_text, u.status_token) == ("accepted", "accepted")
    assert (u.deliverable, u.gate_verdict, u.cost) == (
        "`docs/plans/x.md`", "`analyst` → approve", "10k / 3")
    assert (u.ledger_path, u.slug, u.component) == (
        "falkor-chat/docs/plans/x-coordination.md", "x", "falkor-chat")
    assert (u.table_idx, u.row_idx) == (1, 1)
    assert u.key == "falkor-chat/docs/plans/x-coordination.md#U1"


def test_header_variants_agent_for_owner_agent_scope_and_no_cost():
    assert map_columns(["Unit", "Agent", "Agent id", "Status", "Deliverable", "Gate → verdict"]) == {
        "unit": 0, "owner": 1, "agentId": 2, "status": 3, "deliverable": 4, "gateVerdict": 5}
    cols = map_columns(["Unit(s)", "Agent (scope)", "Status"])
    assert cols == {"unit": 0, "owner": 1, "status": 2}
    # `Agent` is the owner only when there is no `Owner` column
    both = map_columns(["Unit", "Owner", "Agent", "Status"])
    assert both["owner"] == 1 and "agentId" not in both
    d = ledger(["| U1 | `coder` | in-flight | — |"],
               cols="| Unit | Agent | Status | Deliverable |\n|---|---|---|---|\n")
    u = d.units[0]
    assert u.owner == "`coder`" and u.cost is None and u.agent_id is None


def test_absent_column_is_absent_not_empty_but_an_empty_cell_is_empty():
    d = ledger(["| U1 | | accepted |"], cols="| Unit | Owner | Status |\n|---|---|---|\n")
    u = d.units[0]
    assert u.owner == "" and u.deliverable is None and u.cost is None
    props = u.props()
    assert props["owner"] == "" and "deliverable" not in props and "cost" not in props


def test_pre_convention_table_without_status_column_warns_and_does_not_crash():
    d = ledger(["| U1 | someone |"], cols="| Unit | Who |\n|---|---|\n")
    assert d.units[0].status_token == "unknown"
    assert any("table 1: no Status column" in w for w in d.warnings)


def test_multiple_tables_and_non_unit_tables_are_handled():
    text = (HEADER.format(status="active") + COLS + "| U1 | a | b | accepted | — | — | — |\n\n"
            "| Metric | Value |\n|---|---|\n| cost | 3 |\n\n"
            + COLS + "| U2 | a | b | queued | — | — | — |\n")
    d = parse_ledger("docs/plans/x-coordination.md", text.encode())
    assert [(u.unit_key, u.table_idx, u.row_idx) for u in d.units] == [("U1", 1, 1), ("U2", 3, 1)]


def test_duplicate_unit_label_gets_tilde_n_and_a_warning():
    d = ledger(["| U32 | a | b | accepted | — | — | — |",
                "| U32 | a | b | queued | — | — | — |",
                "| U32 | a | b | gated | — | — | — |"])
    assert [u.key.split("#")[1] for u in d.units] == ["U32", "U32~2", "U32~3"]
    assert [u.unit_key for u in d.units] == ["U32"] * 3  # lookup value carries no suffix
    assert sum("duplicate unit U32" in w for w in d.warnings) == 2


def test_unit_key_strips_emphasis_and_backticks_and_cuts_at_first_em_dash():
    assert unit_key_of("**U3** — Stage 6 — pass 2") == "U3"
    assert unit_key_of("`U3b` — x") == "U3b"
    assert unit_key_of("U1 K-091 band write") == "U1 K-091 band write"  # no ` — `: whole cell
    assert unit_key_of("U1—x") == "U1—x"  # a dash without spaces is not the cut
    assert unit_key_of("~~U9~~ — dropped") == "U9"


def test_a_600_char_cell_is_kept_whole_and_a_long_unit_key_is_hashed():
    long = "x" * 700
    d = ledger([f"| U1 | a | b | accepted | {long} | — | — |",
                f"| {'L' * 700} | a | b | queued | — | — | — |"])
    assert d.units[0].deliverable == long
    assert d.units[1].key.startswith("h:") and len(d.units[1].key.encode()) <= 512


def test_escaped_pipe_stays_inside_the_cell_and_surplus_cells_warn():
    d = ledger(["| U1 | a | b | accepted | `x\\|y` | — | — |"])
    assert d.units[0].deliverable == "`x|y`" and d.units[0].status_token == "accepted"
    d = ledger(["| U1 | a | b | accepted | one | two | three | four |"])
    assert any("8 cells, header has 7" in w for w in d.warnings)


def test_short_rows_are_padded_without_a_warning():
    d = ledger(["| U1 | `coder` | `a1` | accepted (committed) |"])
    assert d.units[0].status_token == "accepted" and d.units[0].cost == ""
    assert not any("cells" in w for w in d.warnings)


def test_tables_inside_code_fences_are_ignored():
    text = HEADER.format(status="active") + "```\n" + COLS + "| U1 | a | b | accepted | — | — | — |\n```\n"
    assert parse_ledger("docs/plans/x-coordination.md", text.encode()).units == []


# --- status ---------------------------------------------------------------------------

def test_unknown_status_is_listed_with_raw_text_and_a_warning():
    d = ledger(["| U1 | a | b | blocked (on architect) | — | — | — |"])
    u = d.units[0]
    assert (u.status_token, u.status_text) == ("unknown", "blocked (on architect)")
    assert any("U1" in w and "blocked (on architect)" in w for w in d.warnings)


def test_relayed_row_reads_paused():
    d = ledger([
        '| U3 | `coder` | — | **relayed** | ⏸ "backfill NULLs to 0 or error?" — relayed 2026-08-22 | — → — | — |'
    ])
    u = d.units[0]
    assert u.status_token == "paused"
    assert u.question == "backfill NULLs to 0 or error?" and u.asked_on == "2026-08-22"


def test_paused_with_and_without_a_relay_date():
    assert parse_paused('⏸ "which?" — relayed 2026-08-22') == ("which?", "2026-08-22")
    assert parse_paused("⏸ which one? — relayed 2026-09-01") == ("which one?", "2026-09-01")
    assert parse_paused('⏸ "since 2026-01-01, which?" — relayed 2026-08-22') == (
        "since 2026-01-01, which?", "2026-08-22")  # the relay date, not any date in the question
    assert parse_paused('⏸ "which?"') == ("which?", None)
    assert parse_paused("—") == ("—", None)
    d = ledger(['| U3 | a | b | paused | ⏸ "which?" | — | — |'])
    assert d.units[0].asked_on is None
    assert any("paused without a relay date" in w for w in d.warnings)


# --- review citations -----------------------------------------------------------------

REVIEWS = {"claude/docs/reviews/a.md", "falkor-chat/docs/reviews/b.md", "docs/reviews/c.md"}


def test_cited_and_resolved_reviews_for_a_live_ledger():
    d = ledger([
        "| U1 | a | b | accepted | `claude/docs/reviews/a.md` | — | — |",
        "| U2 | a | b | in-flight | x | `docs/reviews/c.md` → approve | — |",
        "| U3 | a | b | abandoned | `falkor-chat/docs/reviews/b.md#pass-2` | — | — |",
    ], reviews=REVIEWS)
    assert d.cited_reviews == ["claude/docs/reviews/a.md", "docs/reviews/c.md", "falkor-chat/docs/reviews/b.md"]
    assert d.resolved_reviews == ["claude/docs/reviews/a.md", "falkor-chat/docs/reviews/b.md"]


def test_a_delivered_or_gated_row_cites_a_review_without_resolving_it():
    d = ledger(["| U1 | a | b | delivered | `claude/docs/reviews/a.md` | — | — |",
                "| U2 | a | b | gated | `docs/reviews/c.md` | — | — |"], reviews=REVIEWS)
    assert d.cited_reviews == ["claude/docs/reviews/a.md", "docs/reviews/c.md"]
    assert d.resolved_reviews == []


def test_the_same_lists_are_emitted_for_an_archived_ledger():
    d = ledger(["| U1 | a | b | accepted | `claude/docs/reviews/a.md` | — | — |",
                "| U2 | a | b | in-flight | `docs/reviews/c.md` | — | — |"],
               status="archived", reviews=REVIEWS)
    assert d.archived
    assert d.cited_reviews == ["claude/docs/reviews/a.md", "docs/reviews/c.md"]
    assert d.resolved_reviews == ["claude/docs/reviews/a.md"]


def test_cited_path_is_retried_under_the_ledgers_component_dir_else_dropped_with_a_warning():
    d = ledger(["| U1 | a | b | accepted | `docs/reviews/b.md` | — | — |",
                "| U2 | a | b | accepted | `docs/reviews/nowhere.md` | — | — |"], reviews=REVIEWS)
    # `docs/reviews/b.md` -> `falkor-chat/docs/reviews/b.md` (the ledger's component)
    assert d.cited_reviews == ["falkor-chat/docs/reviews/b.md"]
    assert any("docs/reviews/nowhere.md not found" in w for w in d.warnings)
    # a root-tree ledger has no component dir to retry under
    root = ledger(["| U1 | a | b | accepted | `docs/reviews/b.md` | — | — |"],
                  path="docs/plans/x-coordination.md", reviews=REVIEWS)
    assert root.cited_reviews == []


def test_cites_lists_repo_paths_in_deliverable_and_gate_cells():
    d = ledger([
        "| U1 | a | b | accepted | `docs/plans/x.md`, `not a path`, `falkor-chat/server/a.py` "
        "| `docs/reviews/c.md#p2` → ok | — |"
    ])
    assert d.units[0].cites == ["docs/plans/x.md", "docs/reviews/c.md", "falkor-chat/server/a.py"]


# --- Item / Needs columns (plan3 3.4) --------------------------------------------------

ITEM_COLS = "| Unit | Owner | Status | Item | Needs |\n|---|---|---|---|---|\n"


def test_item_and_needs_columns_map_by_name_and_store_tokens_verbatim():
    d = ledger(["| U1 | a | queued | `GAL-F0042-S01-T03` | GAL-F0007, other-ledger:U2, U1b |"], cols=ITEM_COLS)
    u = d.units[0]
    assert u.item_id == "GAL-F0042-S01-T03"
    assert u.needs == ["GAL-F0007", "other-ledger:U2", "U1b"] and u.needs_error is None


def test_opacity_planted_malformed_ids_and_free_text_are_stored_verbatim():
    d = ledger(["| U1 | a | queued | GAL-F42 | GAL-F42, wait for the migration |"], cols=ITEM_COLS)
    u = d.units[0]
    assert u.item_id == "GAL-F42"
    assert u.needs == ["GAL-F42", "wait for the migration"]


def test_needs_none_is_empty_list_and_missing_is_absent():
    d = ledger(["| U1 | a | queued | — | none |", "| U2 | a | queued | — | — |", "| U3 | a | queued | | |"],
               cols=ITEM_COLS)
    assert d.units[0].needs == []
    assert d.units[1].needs is None and d.units[2].needs is None
    assert d.units[0].item_id is None
    p = d.units[0].props()
    assert p["needs"] == [] and "itemId" not in p
    assert "needs" not in d.units[1].props()
    # a ledger without the columns is unchanged
    plain = ledger(["| U1 | a | b | queued | — | — | — |"])
    assert plain.units[0].needs is None and "needs" not in plain.units[0].props()
    assert "itemId" not in plain.units[0].props()


def test_an_unusable_item_cell_is_not_stored_and_warns():
    d = ledger([f"| U1 | a | queued | {'G' * 600} | — |"], cols=ITEM_COLS)
    assert d.units[0].item_id is None
    assert any("unusable Item cell" in w for w in d.warnings)


def test_malformed_needs_cell_sets_needs_error_and_a_warning():
    d = ledger(["| U1 | a | queued | — | A,,B |"], cols=ITEM_COLS)
    u = d.units[0]
    assert u.needs is None and "empty token" in u.needs_error
    assert u.props()["needsError"] == u.needs_error and "needs" not in u.props()
    assert any("U1" in w and "empty token" in w for w in d.warnings)
    assert parse_needs("A,") == (None, "empty token in Needs cell 'A,'")


def test_needs_backticks_around_a_token_are_markup_not_part_of_it():
    assert parse_needs("`GAL-F0007`, `a:U2`") == (["GAL-F0007", "a:U2"], None)


# --- archived ledgers keep minimal units -----------------------------------------------

def test_archived_unit_props_are_minimal():
    d = ledger(["| U1 | `coder` | queued | `GAL-F0042-S01-T03` | GAL-F0007 |"],
               status="archived", cols=ITEM_COLS)
    (props,) = d.unit_props()
    assert set(props) == {
        "key", "unitKey", "unitLabel", "statusToken", "itemId", "ledgerPath", "slug", "component",
        "tableIdx", "rowIdx",
    }
    # the row was still parsed in full
    assert d.units[0].needs == ["GAL-F0007"] and d.units[0].owner == "`coder`"


def test_archived_malformed_needs_adds_no_warning_because_needs_is_not_mirrored():
    d = ledger(["| U1 | a | queued | — | A,,B |"], status="archived", cols=ITEM_COLS)
    assert not any("empty token" in w for w in d.warnings)


def test_live_unit_props_are_full():
    d = ledger(["| U1 | `coder` | `a1` | accepted | `docs/plans/x.md` | — | 1k |"])
    props = d.unit_props()[0]
    assert {"unit", "statusText", "owner", "agentId", "deliverable", "gateVerdict", "cost",
            "cites", "rowHash"} <= set(props)


def test_row_hash_is_whitespace_normalised_and_changes_with_any_cell():
    a = ledger(["| U1 | a | b | accepted | one  two | — | — |"]).units[0].row_hash
    b = ledger(["| U1 | a | b | accepted | one two | — | — |"]).units[0].row_hash
    c = ledger(["| U1 | a | b | accepted | one three | — | — |"]).units[0].row_hash
    assert a == b and a != c and len(a) == 64


# --- :Source properties -----------------------------------------------------------------

def test_ledger_source_props():
    d = ledger(["| U1 | a | b | accepted | — | — | — |"], reviews=set())
    p = d.source_props()
    assert p["kind"] == "ledger" and p["slug"] == "x" and p["component"] == "falkor-chat"
    assert p["docStatus"] == "active" and p["owner"] == "`teco`" and p["tracks"] == "K-1"
    assert p["citedReviews"] == [] and p["resolvedReviews"] == [] and p["rowCount"] == 1
    assert p["aliases"] == [] and "itemId" not in p and p["warnings"] == []
    assert len(p["sha256"]) == 64 and p["bytes"] > 0


def test_header_item_and_aliases_reach_the_source_props():
    text = ("# X\n\n> **Status:** active · **Owner:** `teco` · **Item:** `GAL-F0002` · "
            "**Aliases:** `GAL-F0001`\n\n" + COLS + "| U1 | a | b | queued | — | — | — |\n")
    p = parse_ledger("docs/plans/x-coordination.md", text.encode()).source_props()
    assert p["itemId"] == "GAL-F0002" and p["aliases"] == ["GAL-F0001"]


def test_a_ledger_without_a_header_block_has_no_status_and_no_aliases():
    text = "# X\n\n" + COLS + "| U1 | a | b | queued | — | — | — |\n"
    d = parse_ledger("docs/plans/x-coordination.md", text.encode())
    p = d.source_props()
    assert not d.archived and "docStatus" not in p and "aliases" not in p


# --- unitLabel (plan3 section 3.4 `Unit label`) ---

@pytest.mark.parametrize("key,label", [
    ("U1", "U1"),
    ("U1 (LM Studio backend fix)", "U1"),
    ("U1(x)", "U1"),
    ("U1 K-091 band write + backfill", "U1"),
    ("U10b Note: later", "U10b"),
    ("A7 Note: x", "A7"),
    ("U2.3 between", "U2.3"),
    ("C-graph-dba (data half)", "C-graph-dba"),
    ("Queued: foo", "Queued"),
    ("U3:", "U3"),
    ("U3,;", "U3"),
    ("", ""),
])
def test_unit_label_rule(key, label):
    from coordgraph.parse.ledger import unit_label_of
    assert unit_label_of(key) == label


def test_every_unit_carries_a_label_archived_or_not():
    text = ("# L\n\n> **Status:** {s} · **Owner:** `teco` · **Tracks:** —\n\n"
            "| Unit | Status |\n|---|---|\n| U1 (a) — x | accepted |\n| U1 (b) — y | accepted |\n")
    for status, archived in (("active", False), ("archived", True)):
        d = parse_ledger("docs/plans/x-coordination.md", text.format(s=status).encode())
        assert d.archived is archived
        assert [p["unitLabel"] for p in d.unit_props()] == ["U1", "U1"]
        assert [p["unitKey"] for p in d.unit_props()] == ["U1 (a)", "U1 (b)"]


# --- pins for the row hash, the width cut, the item bound ---------------------------------

def test_row_hash_is_sha256_of_the_normalised_cells_as_json_including_the_last_cell():
    import hashlib
    import json
    d = ledger(["| U1  — é | `a` | `id`  | accepted |  `d` | gate | 10k /\t3 |"])
    cells = ["U1 — é", "`a`", "`id`", "accepted", "`d`", "gate", "10k / 3"]
    expected = hashlib.sha256(json.dumps(cells, ensure_ascii=False).encode("utf-8")).hexdigest()
    assert d.units[0].row_hash == expected
    d2 = ledger(["| U1  — é | `a` | `id`  | accepted |  `d` | gate | 20k |"])
    assert d2.units[0].row_hash != expected  # a last-column edit is visible to row-level drift


def test_an_over_wide_row_is_cut_to_the_header_width_with_a_warning():
    import hashlib
    import json
    d = ledger(["| U1 | `a` | `id` | accepted | `d` | gate | 5k | stray | more |"])
    cells = ["U1", "`a`", "`id`", "accepted", "`d`", "gate", "5k"]
    assert d.units[0].row_hash == hashlib.sha256(
        json.dumps(cells, ensure_ascii=False).encode("utf-8")).hexdigest()
    assert d.units[0].cost == "5k"
    assert any("9 cells, header has 7" in w for w in d.warnings)


def test_an_item_cell_is_bounded_by_bytes_not_characters():
    d = ledger(["| U1 | a | queued | — | — |"], cols=ITEM_COLS)
    assert d.units[0].item_id is None  # sanity: no value, no warning
    cjk = "漢" * 200  # 200 characters, 600 bytes
    d = ledger([f"| U1 | a | queued | `{cjk}` | — |"], cols=ITEM_COLS)
    assert d.units[0].item_id is None and any("unusable Item" in w for w in d.warnings)


def test_a_row_without_a_status_column_has_no_status_text_but_an_unknown_token():
    d = ledger(["| U1 | `a` | x |"], cols="| Unit | Owner | Deliverable |\n|---|---|---|\n")
    u = d.units[0]
    assert u.status_text is None and u.status_token == "unknown"
    assert "statusText" not in u.props() and u.props()["statusToken"] == "unknown"
