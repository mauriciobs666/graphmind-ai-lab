#!/usr/bin/env python3
"""pwq_kb_selftest.py -- selftest and mutation target for `pwq_kb.py` (plan T1-T9).

Plain assertion script (no pytest under `claude/`; same house style as
`check_content_loss_selftest.py`). Expected values come from the pinned source (via `git show
<sha>:path` and an independent `grep`) and from the plan's tables, never from the code under test.

Run: `python3 claude/cobb/scripts/pwq_kb_selftest.py`   -> "ALL PASS", exit 0; else exit 1.
Mutation runs: point `PWQ_KB_PATH` at a mutated copy of `pwq_kb.py`.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import re
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PIN = "3fcb171cf03065f99aca800d9ea30ff6a2a7ba95"
SRC_PATH = "skills/python-web-quirks/SKILL.md"

_spec = importlib.util.spec_from_file_location("pwq_kb_under_test", os.environ.get("PWQ_KB_PATH", HERE / "pwq_kb.py"))
kb = importlib.util.module_from_spec(_spec)
sys.modules["pwq_kb_under_test"] = kb
_spec.loader.exec_module(kb)

_cache: dict = {}


def pin_text() -> str:
    if "pin" not in _cache:  # independent of kb.read_pinned
        _cache["pin"] = subprocess.run(
            ["git", "-C", str(REPO), "show", f"{PIN}:{SRC_PATH}"], capture_output=True, check=True
        ).stdout.decode()
    return _cache["pin"]


def source():
    return kb.parse_source(pin_text())


def units():
    return kb.build_units(source())


def copy_text() -> str:
    if "copy" not in _cache:
        s = source()
        _cache["copy"] = kb.render_copy(s, kb.build_units(s), "2026-11-09", PIN)
    return _cache["copy"]


def verify(text: str) -> list[str]:
    return kb.verify_copy(text, pin_text())


def replace_once(text: str, old: str, new: str) -> str:
    assert text.count(old) >= 1, f"fixture bug: {old!r} not in text"
    return text.replace(old, new, 1)


def unit_block(copy: str, uid: str) -> str:
    start = copy.index(f"<!-- unit {uid} -->")
    nxt = copy.find("<!-- unit ", start + 5)
    return copy[start : nxt if nxt != -1 else len(copy)]


def raises(exc, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc as e:
        return e
    raise AssertionError(f"expected {exc.__name__}")


# ----------------------------------------------------------------------------- T1
def t1_pin_parses_42_headings_and_preamble():
    grep = subprocess.run(
        ["grep", "-c", "^## "], input=pin_text().encode(), capture_output=True
    ).stdout.decode().strip()
    src = source()
    assert len(src.sections) == int(grep) == 42, (len(src.sections), grep)
    assert [s.index for s in src.sections] == list(range(1, 43))
    assert src.sections[0].heading.startswith("`asyncio.create_task` fire-and-forget")
    assert src.sections[0].line == 33  # grep -n of the pin
    assert "# Python web/async-framework quirks" in src.preamble
    assert "**This is a cache, not the source of truth.**" in src.preamble
    assert src.front_matter.startswith("---\nname: python-web-quirks") and "name: python-web-quirks" not in src.preamble
    # blank lines after a heading belong to the body (verbatim)
    assert src.sections[0].body.startswith("\n\n**The docs' warning:**")


def t1_dropping_a_heading_changes_the_count():
    text = pin_text().replace("\n## Version rot ahead", "\nVersion rot ahead", 1)
    assert len(kb.parse_source(text).sections) == 41


def t1_unclosed_front_matter_and_pin_must_be_a_sha():
    raises(kb.KBError, kb.parse_source, "---\nname: x\n\n## A\nbody\n")
    raises(kb.KBError, kb.read_pinned, "HEAD")
    raises(kb.KBError, kb.read_pinned, "main")


def t1_pin_is_read_from_the_explicit_sha_not_head():
    with tempfile.TemporaryDirectory() as d:
        run = lambda *a: subprocess.run(["git", "-C", d, *a], check=True, capture_output=True).stdout.decode().strip()
        run("init", "-q")
        run("config", "user.email", "t@t")
        run("config", "user.name", "t")
        p = Path(d) / SRC_PATH
        p.parent.mkdir(parents=True)
        p.write_text("---\nname: x\n---\n\n# H\n\n## First\n\nalpha\n")
        run("add", ".")
        run("commit", "-qm", "one")
        first = run("rev-parse", "HEAD")
        p.write_text("---\nname: x\n---\n\n# H\n\n## First\n\nbeta\n")
        run("commit", "-qam", "two")
        second = run("rev-parse", "HEAD")
        assert "alpha" in kb.read_pinned(first, repo=Path(d)) and "beta" not in kb.read_pinned(first, repo=Path(d))
        assert "beta" in kb.read_pinned(second, repo=Path(d))
        # the CLI renders the pinned content, and verify-copy against the other pin fails
        out = Path(d) / "copy.md"
        assert kb.main(["render-copy", "--pin", first, "--repo", d, "--out", str(out)]) == 0
        assert "alpha" in out.read_text() and "beta" not in out.read_text()
        with contextlib.redirect_stdout(io.StringIO()):
            assert kb.main(["verify-copy", "--pin", first, "--repo", d, "--copy", str(out)]) == 0
            assert kb.main(["verify-copy", "--pin", second, "--repo", d, "--copy", str(out)]) == 1


# ----------------------------------------------------------------------------- T2
PLAN_HIT_LINES = [587, 624, 633, 660, 670, 744, 754, 891, 951, 974]  # plan §2 (D2-vocabulary subset)


def t2_probe_prints_the_known_hits_and_every_one_is_ruled():
    hits = kb.find_hits(pin_text())
    assert [h.line for h in hits] == PLAN_HIT_LINES, [h.line for h in hits]
    for h in hits:
        assert kb.ruling_for(h, kb.RULINGS) is not None, (h.line, h.text)
    units()  # the real pin builds


def t2_unruled_hit_fails_the_build():
    text = pin_text() + "\n## A scratch section\n\nThe bounded-pool entry above is about total parallelism.\n"
    src = kb.parse_source(text)
    e = raises(kb.UnruledHit, kb.build_units, src)
    assert "bounded-pool entry above" in str(e) or "entry above" in str(e)


def t2_probe_window_is_40_chars_in_either_order():
    gap = lambda n: "x" * n
    near = f"the section {gap(30)} above"  # 30-char gap
    far = f"the section {gap(45)} above"
    rev = f"above {gap(30)} the entry"
    assert len(kb.find_hits(f"## H\n{near}\n")) == 1
    assert len(kb.find_hits(f"## H\n{far}\n")) == 0
    assert len(kb.find_hits(f"## H\n{rev}\n")) == 1
    # exactly at the window: 40 chars between noun and direction is a hit, 41 is not
    assert len(kb.find_hits(f"## H\nsection {gap(38)} above\n")) == 1  # 1 space + 38 + 1 space = 40
    assert len(kb.find_hits(f"## H\nsection {gap(39)} above\n")) == 0
    assert kb.find_hits("## H\nSection ABOVE\n")[0].text == "Section ABOVE"  # case-insensitive


def t2_hit_carries_its_enclosing_section():
    h = {x.line: x for x in kb.find_hits(pin_text())}
    assert h[891].section == 34 and h[891].heading.startswith("Starlette `BackgroundTasks` runs one request")
    assert h[624].section == 23  # the heading line itself belongs to its section


# ----------------------------------------------------------------------------- T3
D2_UNITS = [[2, 34], [21, 22, 23], [24, 25], [29, 39], [30, 40]]


def t3_units_equal_d2_table():
    us = units()
    merged = [[m.index for m in u.members] for u in us if len(u.members) > 1]
    assert merged == D2_UNITS, merged
    assert len(us) == 36 and sum(len(m) for m in merged) == 11
    assert sorted(m.index for u in us for m in u.members) == list(range(1, 43))  # a partition
    assert [u.id for u in us][:3] == ["u01", "u02", "u03"] and "u34" not in [u.id for u in us]
    assert us[1].title.startswith("Starlette/FastAPI `BackgroundTasks` (sync callable)")


def t3_without_the_891_ruling_the_build_fails_not_silently_unmerges():
    rulings = {k: v for k, v in kb.RULINGS.items() if k[1] != "entry above"}
    e = raises(kb.UnruledHit, kb.build_units, source(), rulings)
    assert "line 891" in str(e)


def t3_a_ruling_naming_no_section_fails():
    rulings = dict(kb.RULINGS)
    rulings[(kb.H_PYC2, "section above")] = ("referent", "No such heading")
    raises(kb.KBError, kb.build_units, source(), rulings)


# ----------------------------------------------------------------------------- T4
def t4_copy_opens_with_preamble_and_note_fences_and_verifies_clean():
    text = copy_text()
    lines = text.split("\n")
    assert lines[0] == "<!-- preamble -->"
    note_at = lines.index("<!-- preamble-note -->")
    first_unit = next(i for i, ln in enumerate(lines) if ln.startswith("<!-- unit "))
    assert 0 < note_at < first_unit
    c = kb.parse_copy(text)
    assert c.preamble.startswith("# Python web/async-framework quirks")
    assert "cache, not the source of truth" not in c.preamble + c.note  # waived span in neither block
    assert "Migrated: 2026-11-09" in c.note and not any(ln.startswith("## ") for ln in c.note.split("\n"))
    assert verify(text) == [], verify(text)


def t4_single_and_merged_header_grammar():
    text = copy_text()
    single = kb.parse_copy(text).units[0]
    hl = single.header.split("\n")
    assert hl[0].startswith("Gotcha: `asyncio.create_task`")
    assert [h.split(":")[0] for h in hl] == ["Gotcha", "Versions", "Verified"]
    stored = single.text
    assert stored.startswith("Gotcha: ") and "<!-- unit" not in stored  # fence is not stored
    merged = {u.id: u for u in kb.parse_copy(text).units}["u21"]
    hl = merged.header.split("\n")
    assert hl[1] == "Members:" and [x[:2] for x in hl[2:5]] == ["- "] * 3
    assert [h.split(":")[0] for h in hl[5:]] == [
        "Versions[1]", "Verified[1]", "Versions[2]", "Verified[2]", "Versions[3]", "Verified[3]"
    ]
    src_heads = [h.group(1) for h in re.finditer(r"^## (.*)$", pin_text(), re.M)]
    want = [src_heads[20], src_heads[21], src_heads[22]]  # sections 21-23, source order
    assert [x[2:] for x in hl[2:5]] == want and hl[0] == "Gotcha: " + want[0]
    assert [h for h, _ in merged.parts] == want


def t4_reordering_merged_members_fails_verify():
    block = unit_block(copy_text(), "u21")
    heads = [ln for ln in block.split("\n") if ln.startswith("## ")]
    # swap the first two member bodies' headings in place (copy order differs from source order)
    swapped = block.replace(heads[0], "@@A@@").replace(heads[1], heads[0]).replace("@@A@@", heads[1])
    probs = verify(copy_text().replace(block, swapped))
    assert any(p.startswith("u21:") for p in probs), probs
    # and swapping the Members lines only
    members = [ln for ln in block.split("\n") if ln.startswith("- ")]
    swapped2 = block.replace(members[0], "@@A@@").replace(members[1], members[0]).replace("@@A@@", members[1])
    probs = verify(copy_text().replace(block, swapped2))
    assert any(p.startswith("u21: header mismatch") for p in probs), probs


def t4_preamble_edits_and_note_grammar_are_checked():
    text = copy_text()
    assert any("preamble" in p for p in verify(replace_once(text, "Facts confirmed", "Facts confirmd")))
    bad_note = replace_once(text, "<!-- unit u01 -->", "## stray\n<!-- unit u01 -->")
    assert any("preamble-note" in p for p in verify(bad_note))
    no_date = text.replace("Migrated: 2026-11-09", "Migrated: n/a")
    assert any("Migrated" in p for p in verify(no_date))


# ----------------------------------------------------------------------------- T5
def _dates(body):
    return kb.extract_dates(body)


def t5_only_phrase_qualified_dates_are_verified():
    assert _dates("Verified 2026-01-02 on x. " + "y" * 120 + " Observed 2026-03-04.") == (["2026-01-02"], ["2026-03-04"])
    assert _dates("observed 2026-08-26 in a repo") == ([], ["2026-08-26"])
    for phrase in ("verified", "measured", "confirmed", "reproduced", "re-run", "checked", "tested"):
        assert _dates(f"{phrase} 2026-01-02")[0] == ["2026-01-02"], phrase
    assert _dates("MEASURED 2026-01-02")[0] == ["2026-01-02"]
    assert _dates("untested 2026-01-02")[0] == []  # word boundary
    assert _dates("verified 2026-05-05; Measured 2026-01-02") == (["2026-01-02", "2026-05-05"], [])  # ascending


def t5_verification_window_is_100_chars():
    ok = "verified " + "x" * 91 + "2026-01-02"  # date starts 100 chars after the phrase ends (1 space + 91 + ...)
    pre = "verified"
    for pad, want in ((99, True), (100, True), (101, False)):
        body = pre + " " * pad + "2026-01-02"
        assert (_dates(body)[0] == ["2026-01-02"]) is want, (pad, want)
    assert ok  # silence unused


def t5_real_provenance_date_is_other_not_verified():
    c = {u.id: u for u in kb.parse_copy(copy_text()).units}
    h16 = c["u16"].header.split("\n")
    assert h16[1:] == ["Versions: none stated", "Verified: unverified, migrated 2026-11-09", "Other-dates: 2026-08-26"], h16
    # and the phrase-qualified ones are verified: s1-ish (measured 2026-09-08) in u02
    assert "Verified[1]: 2026-09-08" in c["u02"].header


def t5_verify_copy_recomputes_headers():
    text = copy_text()
    block = unit_block(text, "u16")
    moved = block.replace("Verified: unverified, migrated 2026-11-09\nOther-dates: 2026-08-26", "Verified: 2026-08-26")
    probs = verify(text.replace(block, moved))
    assert any(p.startswith("u16: header mismatch") for p in probs), probs
    # unverified swapped onto a dated unit
    b2 = unit_block(text, "u35")
    assert "Verified: 2026-11-06" in b2
    probs = verify(text.replace(b2, b2.replace("Verified: 2026-11-06", "Verified: unverified, migrated 2026-11-09")))
    assert any(p.startswith("u35: header mismatch") for p in probs), probs
    # the migration date feeds the unverified text
    assert verify(text.replace("Migrated: 2026-11-09", "Migrated: 2026-12-01")) != []


# ----------------------------------------------------------------------------- T6
def t6_versions_extracted_with_lib_and_token():
    assert kb.extract_versions("Verified against starlette 1.6.0 / httpx 0.28.1 and **CPython** 3.12.3.") == [
        ("starlette", "1.6.0"), ("httpx", "0.28.1"), ("CPython", "3.12.3")
    ] or kb.extract_versions("starlette 1.6.0 / httpx 0.28.1; CPython 3.12.3") == [
        ("starlette", "1.6.0"), ("httpx", "0.28.1"), ("CPython", "3.12.3")
    ]
    assert kb.extract_versions("returned in **0.2 ms**; total_tokens = 0.5") == []
    assert kb.extract_versions("uvicorn 0.52.x") == [("uvicorn", "0.52.x")]
    assert kb.extract_versions("anyio 4.14.1, anyio 4.14.1") == [("anyio", "4.14.1")]  # unique


def t6_every_real_versions_token_is_a_body_substring_with_lib_within_40():
    c = kb.parse_copy(copy_text())
    exp = {u.id: u for u in units()}
    seen = 0
    for cu in c.units:
        for lib, tok in kb.header_version_pairs(cu.header):
            seen += 1
            assert any(tok in m.body and kb.version_pair_ok(lib, tok, m.body) for m in exp[cu.id].members), (cu.id, lib, tok)
    assert seen > 40  # plan: 32 sections carry a version-like token


def t6_lib_must_precede_token_within_window():
    assert kb.version_pair_ok("starlette", "1.2.3", "starlette " + "x" * 20 + " 1.2.3")
    assert not kb.version_pair_ok("starlette", "1.2.3", "starlette " + "x" * 60 + " 1.2.3")
    assert not kb.version_pair_ok("starlette", "1.2.3", "1.2.3 then starlette")  # lib after the token
    assert not kb.version_pair_ok("starlette", "9.9.9", "starlette 1.2.3")


def t6_an_invented_token_fails_verify():
    text = copy_text()
    block = unit_block(text, "u35")
    assert "Versions: CPython 3.12\n" in block
    probs = verify(text.replace(block, block.replace("Versions: CPython 3.12\n", "Versions: CPython 3.99\n")))
    assert any("invented version CPython" in p for p in probs), probs
    probs = verify(text.replace(block, block.replace("Versions: CPython 3.12\n", "Versions: CPython 3.12; flask 3.12\n")))
    assert any("invented version flask" in p for p in probs), probs


# ----------------------------------------------------------------------------- T7
def t7_exactly_one_delimiter_per_real_unit_and_never_in_a_body():
    c = kb.parse_copy(copy_text())
    assert len(c.units) == 36 and all(u.delim_count == 1 for u in c.units)
    assert all(kb.BODY_DELIM not in b for u in c.units for _, b in u.parts)
    assert pin_text().count(kb.BODY_DELIM) == 0


def t7_a_body_containing_the_delimiter_or_a_fence_fails():
    for hostile in ("<!-- body -->", "<!-- unit u99 -->", "<!-- preamble -->"):
        src = kb.parse_source(pin_text() + f"\n## Scratch\n\nbefore\n{hostile}\nafter\n")
        raises(kb.KBError, lambda s=src: kb.render_copy(s, kb.build_units(s)))
    text = copy_text()
    block = unit_block(text, "u10")
    doubled = block.replace("<!-- body -->", "<!-- body -->\n<!-- body -->", 1)
    probs = verify(text.replace(block, doubled))
    assert any(p.startswith("u10:") and "delimiter" in p for p in probs), probs
    nodelim = block.replace("<!-- body -->\n", "", 1)
    probs = verify(text.replace(block, nodelim))
    assert any(p.startswith("u10:") and "delimiter" in p for p in probs), probs


# ----------------------------------------------------------------------------- T8
def t8_one_changed_body_character_fails_naming_the_unit():
    text = copy_text()
    block = unit_block(text, "u16")
    body_only = block.split("<!-- body -->", 1)[1]
    assert "**Consequence:**" in body_only
    probs = verify(text.replace(block, block.replace(body_only, body_only.replace("**Consequence:**", "**Consequense:**", 1), 1)))
    assert any(p.startswith("u16: body differs") for p in probs), probs
    # a merged unit's third member
    b21 = unit_block(text, "u21")
    tail = b21.rstrip().rsplit("\n", 2)
    probs = verify(text.replace(b21, b21.rstrip()[:-3] + "xyz\n"))
    assert any(p.startswith("u21: body differs") and "Version rot" in p for p in probs), probs


def t8_missing_extra_and_whitespace_changes_are_caught():
    text = copy_text()
    block = unit_block(text, "u10")
    assert any(p.startswith("u10: unit missing") for p in verify(text.replace(block, "")))
    probs = verify(text.replace(block, block.replace("## ", "### ", 1)))
    assert any(p.startswith("u10:") for p in probs), probs
    # an inner blank line removed is a body change (heading-adjacent blank lines are verbatim)
    probs = verify(text.replace(block, block.replace("\n\n", "\n", 1) if block.count("\n\n") else block))
    assert any(p.startswith("u10:") for p in probs), probs


def t8_verify_cli_exit_codes():
    with tempfile.TemporaryDirectory() as d:
        good, bad = Path(d, "good.md"), Path(d, "bad.md")
        good.write_text(copy_text())
        b16 = unit_block(copy_text(), "u16")
        body16 = b16.split("<!-- body -->", 1)[1]
        bad.write_text(copy_text().replace(body16, body16.replace("**Consequence:**", "**Consequense:**", 1), 1))
        with contextlib.redirect_stdout(io.StringIO()) as out:
            assert kb.main(["verify-copy", "--pin", PIN, "--copy", str(good)]) == 0
            assert kb.main(["verify-copy", "--pin", PIN, "--copy", str(bad)]) == 1
        assert "FAIL: u16: body differs" in out.getvalue()
        with contextlib.redirect_stderr(io.StringIO()):
            assert kb.main(["verify-copy", "--pin", "HEAD", "--copy", str(good)]) == 2


# ----------------------------------------------------------------------------- T9
def t9_listing_matches_real_grep_and_is_within_bound():
    text = copy_text()
    with tempfile.NamedTemporaryFile("w", suffix=".md") as f:
        f.write(text)
        f.flush()
        real = subprocess.run(
            ["grep", "-n", "-E", r"^(<!-- (unit |preamble)|## )", f.name], capture_output=True
        ).stdout.decode()
    assert kb.listing(text) == real
    assert len(real.encode()) == kb.listing_bytes(text) <= 10_000 and kb.LISTING_BOUND == 10_000
    lines = real.split("\n")
    assert sum(1 for ln in lines if re.match(r"\d+:## ", ln)) == 42
    assert sum(1 for ln in lines if re.match(r"\d+:<!-- unit ", ln)) == 36
    assert sum(1 for ln in lines if re.match(r"\d+:<!-- preamble", ln)) == 2  # both preamble fences
    assert verify(text) == []


def t9_twenty_five_extra_200_char_titles_break_the_bound():
    text = copy_text()
    extra = "".join(f"<!-- unit u9{i:02d} -->\nGotcha: x\n<!-- body -->\n## {'T' * 200}\n\nb\n" for i in range(25))
    big = text + extra
    # measured against the real grep, independent of listing()
    with tempfile.NamedTemporaryFile("w", suffix=".md") as f:
        f.write(big)
        f.flush()
        n = len(subprocess.run(["grep", "-n", "-E", r"^(<!-- (unit |preamble)|## )", f.name], capture_output=True).stdout)
    assert n > 10_000
    assert any("title listing" in p for p in verify(big)), verify(big)


def t9_bound_is_inclusive_at_the_boundary():
    text = copy_text()
    n = kb.listing_bytes(text)
    assert kb.verify_copy(text, pin_text(), bound=n) == []
    assert any("title listing" in p for p in kb.verify_copy(text, pin_text(), bound=n - 1))


# ---------------------------------------------------------------------- units CLI
def t9b_units_cli_prints_hits_units_and_chunk_table_matching_the_chunker():
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        assert kb.main(["units", "--pin", PIN, "--chunks"]) == 0
    s = out.getvalue()
    assert "== cross-reference hits (10) ==" in s and "== units (36 units from 42 sections) ==" in s
    assert "UNRULED" not in s
    spec = importlib.util.spec_from_file_location("chk", REPO / "falkor-chat/server/falkorchat/chunking.py")
    chk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(chk)
    rows = re.findall(r"^(u\d\d)\t(\d+)\t", s.split("== chunks")[1], re.M)
    assert len(rows) == 36
    for u in units():
        want = len(chk.split_into_chunks(kb.render_unit(u, "2026-11-09"), size=1000, overlap=150))
        assert dict(rows)[u.id] == str(want), (u.id, want)
    total = int(re.search(r"total chunks (\d+)", s).group(1))
    assert total == sum(int(n) for _, n in rows)
    assert re.search(r"units with >= 4 chunks: (\d+)", s)


def t9c_units_cli_fails_on_an_unruled_hit():
    with tempfile.TemporaryDirectory() as d:
        run = lambda *a: subprocess.run(["git", "-C", d, *a], check=True, capture_output=True).stdout.decode().strip()
        run("init", "-q"); run("config", "user.email", "t@t"); run("config", "user.name", "t")
        p = Path(d) / SRC_PATH
        p.parent.mkdir(parents=True)
        p.write_text("# H\n\n## A\n\nthe section above says\n")
        run("add", "."); run("commit", "-qm", "x")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            assert kb.main(["units", "--pin", run("rev-parse", "HEAD"), "--repo", d]) == 1
        assert "UNRULED" in out.getvalue() and "BUILD FAILS" in out.getvalue()



# ======================================================================== S3a: T16-T23
import hashlib  # noqa: E402
import json  # noqa: E402

PROD_URL, TEST_URL = "http://fake-prod:8200", "http://fake-test:8210"
FAKE_TARGETS = {"agent-team": PROD_URL, "pwq-test": TEST_URL}
MINI_SRC = "# H\n\n> waived blockquote\n\n## Alpha title\n\nalpha body\n\n## Beta title\n\nbeta body\n\n## Gamma title\n\ngamma body\n"
SCHEMA_FIELDS = None


def schema_fields() -> set[str]:
    """IngestDocumentIn's field names, read from falkor-chat's schemas.py (the fake is built from them)."""
    global SCHEMA_FIELDS
    if SCHEMA_FIELDS is None:
        src = (REPO / "falkor-chat/server/falkorchat/schemas.py").read_text()
        body = src.split("class IngestDocumentIn(BaseModel):", 1)[1].split("\nclass ", 1)[0]
        SCHEMA_FIELDS = set(re.findall(r"^    (\w+): ", body, re.M))
    return SCHEMA_FIELDS


class FakeServer:
    """One workspace's REST server + Cypher fetcher. REST body handling follows pydantic's default:
    declared fields kept, unknown keys silently ignored, `text` required and non-empty."""

    def __init__(self, base):
        self.base, self.docs, self.n = base, {}, 0

    def post(self, body):
        fields = {k: v for k, v in body.items() if k in schema_fields()}
        if not isinstance(fields.get("text"), str) or not fields["text"]:
            return 422, {"detail": "text required"}
        if fields.get("producedBy") != "cobb":
            return 400, {"detail": "fake: producedBy must be 'cobb'"}
        for d in self.docs.values():  # byte-identical text auto-supersedes the older current doc
            if d["current"] and d["text"] == fields["text"]:
                d["current"] = False
        self.n += 1
        did = f"{self.base[-4:]}-doc{self.n}"
        self.docs[did] = dict(documentId=did, title=fields.get("title"), text=fields["text"], status="processing",
                              current=True, chunks=2, embedded=2, producedBy=fields["producedBy"], fields=fields)
        return 201, {"documentId": did, "chunkCount": 2, "status": "processing"}


class FakeWorld:
    def __init__(self):
        self.servers = {PROD_URL: FakeServer(PROD_URL), TEST_URL: FakeServer(TEST_URL)}
        self.requests: list[tuple[str, str]] = []

    def srv(self, ws):
        return self.servers[FAKE_TARGETS[ws]]

    def transport(self, method, url, body=None):
        self.requests.append((method, url))
        base = next(b for b in self.servers if url.startswith(b + "/"))
        srv, path = self.servers[base], url[len(base) + 1:]
        if method == "POST" and path == "documents":
            return srv.post(body or {})
        if method == "DELETE" and path.startswith("documents/"):
            return (200, {"deleted": True}) if srv.docs.pop(path.split("/", 1)[1], None) else (404, {})
        return 404, {"detail": "fake has no such route"}

    # fetcher interface
    def current_docs(self, ws, titles, ids):
        return [dict(documentId=d["documentId"], title=d["title"], text=d["text"], status=d["status"])
                for d in self.srv(ws).docs.values() if d["current"] and (d["title"] in titles or d["documentId"] in ids)]

    def embedding_counts(self, ws, ids):
        return {i: (None if i not in self.srv(ws).docs else
                    dict(chunks=self.srv(ws).docs[i]["chunks"], embedded=self.srv(ws).docs[i]["embedded"],
                         status=self.srv(ws).docs[i]["status"])) for i in ids}

    def post_count(self):
        return sum(1 for m, _ in self.requests if m == "POST")

    def delete_count(self):
        return sum(1 for m, _ in self.requests if m == "DELETE")


def mini_copy() -> str:
    s = kb.parse_source(MINI_SRC)
    return kb.render_copy(s, kb.build_units(s), "2026-11-09", "abc1234")


def run_ingest(world, copy, ws="pwq-test", manifest=None, confirm=False, base=None):
    return kb.ingest(copy, ws=ws, base_url=base or FAKE_TARGETS[ws], manifest_path=manifest, transport=world.transport,
                     fetcher=world, confirm_production=confirm, targets=FAKE_TARGETS,
                     read_source=lambda pin: MINI_SRC)


def tmpm():
    d = tempfile.mkdtemp()
    return Path(d, "m.json")


def actions(results):
    return {r.id: r.action for r in results}


def t16_ingest_sends_producedby_cobb_and_the_stored_text_verbatim():
    w, m, copy = FakeWorld(), tmpm(), mini_copy()
    res = run_ingest(w, copy, manifest=m)
    assert actions(res) == {"u01": "ingest", "u02": "ingest", "u03": "ingest"}, actions(res)
    docs = list(w.srv("pwq-test").docs.values())
    assert [d["producedBy"] for d in docs] == ["cobb"] * 3
    cu = kb.parse_copy(copy).units
    assert [d["text"] for d in docs] == [u.text for u in cu]  # byte-identical stored text
    assert [d["title"] for d in docs] == ["Alpha title", "Beta title", "Gamma title"]
    assert all(d["fields"].get("sourceFormat") for d in docs)
    assert all(set(d["fields"]) <= schema_fields() for d in docs)  # only declared IngestDocumentIn fields
    assert {"producedBy", "sourceFormat", "text", "title"} <= schema_fields()
    # every request hit the validated base URL, only POST /documents
    assert w.requests == [("POST", TEST_URL + "/documents")] * 3, w.requests
    mf = json.loads(m.read_text())
    assert list(mf) == ["pin", "migratedOn", "units", "waivedSpans"], list(mf)
    assert mf["pin"] == "abc1234" and mf["migratedOn"] == "2026-11-09" and mf["waivedSpans"] == ["> waived blockquote"]
    assert [list(u) for u in mf["units"]] == [["id", "title", "headings", "documentId", "sha256"]] * 3
    assert [u["documentId"] for u in mf["units"]] == [d["documentId"] for d in docs]
    assert [u["sha256"] for u in mf["units"]] == [hashlib.sha256(u.text.encode()).hexdigest() for u in cu]
    assert mf["units"][0]["headings"] == ["Alpha title"]


def t16_a_server_rejection_is_an_error_not_a_silent_skip():
    w = FakeWorld()
    bad = lambda method, url, body=None: (400, {"detail": "nope"})
    e = raises(kb.KBError, kb.ingest, mini_copy(), ws="pwq-test", base_url=TEST_URL, manifest_path=tmpm(), transport=bad,
               fetcher=w, targets=FAKE_TARGETS, read_source=lambda p: MINI_SRC)
    assert "400" in str(e)


def t17_rerun_skips_every_unit_and_posts_nothing():
    w, m, copy = FakeWorld(), tmpm(), mini_copy()
    run_ingest(w, copy, manifest=m)
    before = m.read_text()
    w.requests.clear()
    res = run_ingest(w, copy, manifest=m)
    assert actions(res) == {"u01": "skip", "u02": "skip", "u03": "skip"}, actions(res)
    assert w.post_count() == 0 and m.read_text() == before


def t17_edited_unit_with_a_registered_current_document_is_refused_not_reposted():
    w, m, copy = FakeWorld(), tmpm(), mini_copy()
    run_ingest(w, copy, manifest=m)
    w.requests.clear()
    edited = copy.replace("beta body", "beta body EDITED")
    res = run_ingest(w, edited, manifest=m)
    assert actions(res) == {"u01": "skip", "u02": "refuse", "u03": "skip"}, actions(res)
    old = json.loads(m.read_text())["units"][1]["documentId"]
    r = next(r for r in res if r.id == "u02")
    assert f"curl -X DELETE {TEST_URL}/documents/{old}" in r.note, r.note
    assert w.post_count() == 0


def t18_identical_current_documents_are_adopted_without_a_post():
    w, copy = FakeWorld(), mini_copy()
    run_ingest(w, copy, manifest=tmpm())  # ingest, then lose the manifest
    ids = {d["title"]: d["documentId"] for d in w.srv("pwq-test").docs.values()}
    w.srv("pwq-test").docs[ids["Alpha title"]]["status"] = "ready"  # one ready, the others still processing
    w.requests.clear()
    m2 = tmpm()
    res = run_ingest(w, copy, manifest=m2)
    assert actions(res) == {"u01": "adopt", "u02": "adopt", "u03": "adopt"}, actions(res)
    assert w.post_count() == 0
    mf = json.loads(m2.read_text())
    assert {u["title"]: u["documentId"] for u in mf["units"]} == ids


def t18_a_failed_identical_document_is_not_adopted():
    w, copy = FakeWorld(), mini_copy()
    run_ingest(w, copy, manifest=tmpm())
    failed = next(d for d in w.srv("pwq-test").docs.values() if d["title"] == "Beta title")
    failed["status"] = "failed"
    w.requests.clear()
    m2 = tmpm()
    res = run_ingest(w, copy, manifest=m2)
    assert actions(res) == {"u01": "adopt", "u02": "ingest", "u03": "adopt"}, actions(res)
    assert w.post_count() == 1
    new_id = next(u["documentId"] for u in json.loads(m2.read_text())["units"] if u["id"] == "u02")
    assert new_id != failed["documentId"]
    # (auto-supersede in the fake demoted the failed one: the old document is no longer current)
    assert failed["current"] is False


def t18_a_failed_identical_document_in_production_refuses_with_the_curl_line():
    w, copy = FakeWorld(), mini_copy()
    run_ingest(w, copy, ws="agent-team", manifest=tmpm(), confirm=True)
    failed = next(d for d in w.srv("agent-team").docs.values() if d["title"] == "Beta title")
    failed["status"] = "failed"
    w.requests.clear()
    res = run_ingest(w, copy, ws="agent-team", manifest=tmpm(), confirm=True)
    r = next(r for r in res if r.id == "u02")
    assert r.action == "refuse" and f"curl -X DELETE {PROD_URL}/documents/{failed['documentId']}" in r.note, (r.action, r.note)
    assert w.post_count() == 0


def t18_a_current_document_with_different_text_under_the_title_refuses():
    for ws in ("pwq-test", "agent-team"):
        w, copy = FakeWorld(), mini_copy()
        w.srv(ws).post(dict(text="Gotcha: Beta title\nsomething else\n", title="Beta title", producedBy="cobb"))
        stray = next(iter(w.srv(ws).docs.values()))
        res = run_ingest(w, copy, ws=ws, manifest=tmpm(), confirm=True)
        assert actions(res) == {"u01": "ingest", "u02": "refuse", "u03": "ingest"}, (ws, actions(res))
        r = next(r for r in res if r.id == "u02")
        assert f"curl -X DELETE {FAKE_TARGETS[ws]}/documents/{stray['documentId']}" in r.note, r.note
        assert w.post_count() == 2 and len(w.srv(ws).docs) == 3  # u02 never posted over it


def t18_a_renamed_unit_whose_registered_document_has_the_old_title_refuses():
    w, m, copy = FakeWorld(), tmpm(), mini_copy()
    run_ingest(w, copy, manifest=m)
    w.requests.clear()
    renamed = copy.replace("Beta title", "Beta retitled")
    res = run_ingest(w, renamed, manifest=m)
    assert actions(res)["u02"] == "refuse" and w.post_count() == 0, actions(res)


def t19_target_table_pairs_are_enforced_on_the_url_actually_written_to():
    cases = [
        (("pwq-test", PROD_URL, False), "does not match"),
        (("agent-team", TEST_URL, True), "does not match"),
        (("agent-team", "http://localhost:8200x", True), "does not match"),
        (("nope", TEST_URL, False), "unknown workspace"),
        (("agent-team", PROD_URL, False), "--confirm-production"),
    ]
    for (ws, url, conf), msg in cases:
        w = FakeWorld()
        e = raises(kb.TargetRefused, run_ingest, w, mini_copy(), ws=ws if ws in FAKE_TARGETS else "pwq-test", base=url,
                   manifest=tmpm(), confirm=conf) if ws in FAKE_TARGETS else raises(
            kb.TargetRefused, kb.check_target, ws, url, FAKE_TARGETS, confirm_production=conf)
        assert msg in str(e), (ws, url, str(e))
        assert w.requests == []  # refused before any request
    t = kb.check_target("pwq-test", TEST_URL + "/", FAKE_TARGETS)
    assert t.base_url == TEST_URL and not t.production
    t = kb.check_target("agent-team", PROD_URL, FAKE_TARGETS, confirm_production=True)
    assert t.production
    # the committed table is exactly the plan's
    assert kb.TARGETS == {"agent-team": "http://localhost:8200", "pwq-test": "http://localhost:8210"}
    # non-production runs need an explicit manifest so a test run cannot rewrite the production one
    e = raises(kb.TargetRefused, kb.ingest, mini_copy(), ws="pwq-test", base_url=TEST_URL, manifest_path=None,
               transport=FakeWorld().transport, fetcher=FakeWorld(), targets=FAKE_TARGETS, read_source=lambda p: MINI_SRC)
    assert "--manifest" in str(e)


def t19_production_ingest_posts_to_the_production_url_only():
    w = FakeWorld()
    run_ingest(w, mini_copy(), ws="agent-team", manifest=tmpm(), confirm=True)
    assert w.requests == [("POST", PROD_URL + "/documents")] * 3 and len(w.srv("pwq-test").docs) == 0


def t20_replace_and_retire_refuse_production_with_and_without_the_flag():
    for conf in (False, True):
        w, m, copy = FakeWorld(), tmpm(), mini_copy()
        run_ingest(w, copy, ws="agent-team", manifest=m, confirm=True)
        did = json.loads(m.read_text())["units"][1]["documentId"]
        w.requests.clear()
        for fn in (kb.retire, kb.replace):
            e = raises(kb.TargetRefused, fn, copy, unit="u02", ws="agent-team", base_url=PROD_URL, manifest_path=m,
                       transport=w.transport, fetcher=w, confirm_production=conf, targets=FAKE_TARGETS,
                       read_source=lambda p: MINI_SRC)
            assert f"curl -X DELETE {PROD_URL}/documents/{did}" in str(e), str(e)
            assert "production" in str(e)
        assert w.requests == [] and did in w.srv("agent-team").docs  # nothing deleted, nothing posted


def t20_a_production_client_cannot_delete_at_all():
    w = FakeWorld()
    c = kb.RestClient(kb.check_target("agent-team", PROD_URL, FAKE_TARGETS, confirm_production=True), w.transport)
    raises(kb.TargetRefused, c.delete_document, "x")
    assert w.requests == []
    c2 = kb.RestClient(kb.check_target("pwq-test", TEST_URL, FAKE_TARGETS), w.transport)
    raises(kb.KBError, c2.delete_document, "missing-id")  # 404 surfaces as an error
    assert w.requests == [("DELETE", TEST_URL + "/documents/missing-id")]


def t20_retire_and_replace_on_the_test_target():
    w, m, copy = FakeWorld(), tmpm(), mini_copy()
    run_ingest(w, copy, manifest=m)
    old = {u["id"]: u["documentId"] for u in json.loads(m.read_text())["units"]}
    w.requests.clear()
    kb.retire(copy, unit="u03", ws="pwq-test", base_url=TEST_URL, manifest_path=m, transport=w.transport, fetcher=w,
              targets=FAKE_TARGETS, read_source=lambda p: MINI_SRC)
    assert w.requests == [("DELETE", f"{TEST_URL}/documents/{old['u03']}")]
    assert [u["id"] for u in json.loads(m.read_text())["units"]] == ["u01", "u02"]
    w.requests.clear()
    edited = copy.replace("beta body", "beta body EDITED")
    kb.replace(edited, unit="u02", ws="pwq-test", base_url=TEST_URL, manifest_path=m, transport=w.transport, fetcher=w,
               targets=FAKE_TARGETS, read_source=lambda p: MINI_SRC)
    assert [mm for mm, _ in w.requests] == ["DELETE", "POST"], w.requests  # delete first, then post
    assert w.requests[0][1].endswith(old["u02"])
    new = next(u for u in json.loads(m.read_text())["units"] if u["id"] == "u02")
    assert new["documentId"] != old["u02"] and "EDITED" in w.srv("pwq-test").docs[new["documentId"]]["text"]
    assert new["sha256"] == hashlib.sha256(w.srv("pwq-test").docs[new["documentId"]]["text"].encode()).hexdigest()
    assert old["u02"] not in w.srv("pwq-test").docs


class Clock:
    def __init__(self):
        self.t, self.sleeps = 0.0, 0

    def monotonic(self):
        return self.t

    def sleep(self, s):
        self.sleeps += 1
        assert self.sleeps < 200, "poll loop never ends"
        self.t += s


def gate_setup(**over):
    w, m = FakeWorld(), tmpm()
    run_ingest(w, mini_copy(), manifest=m)
    mf = kb.load_manifest(m)
    return w, mf


def run_gate(w, mf, clock=None, **kw):
    clock = clock or Clock()
    return kb.run_gate(w, "pwq-test", mf, sleep=clock.sleep, monotonic=clock.monotonic, **kw), clock


def t21_gate_passes_when_every_chunk_is_embedded_and_prints_a_per_unit_table():
    w, mf = gate_setup()
    (ok, table), clock = run_gate(w, mf)
    assert ok and clock.sleeps == 0
    lines = table.strip().split("\n")
    assert len(lines) == 3 + 1 and lines[0].split() == ["unit", "chunks", "embedded", "status"]
    assert lines[1].split()[:3] == ["u01", "2", "2"], lines[1]


def t21_gate_fails_on_one_null_embedding_after_the_timeout():
    w, mf = gate_setup()
    w.srv("pwq-test").docs[mf["units"][1]["documentId"]]["embedded"] = 1
    (ok, table), clock = run_gate(w, mf, timeout=60, poll=5)
    assert not ok and "u02" in table and "UNEMBEDDED 1/2" in table, table
    assert 11 <= clock.sleeps <= 13  # 60s window / 5s poll, then stop
    assert "u01" in table and "ok" in table.split("u01")[1].split("\n")[0]


def t21_gate_waits_for_embeddings_that_arrive_late():
    w, mf = gate_setup()
    doc = w.srv("pwq-test").docs[mf["units"][0]["documentId"]]
    doc["embedded"] = 0
    clock = Clock()
    orig = clock.sleep

    def sleep(s):
        orig(s)
        if clock.sleeps == 3:
            doc["embedded"] = 2

    ok, table = kb.run_gate(w, "pwq-test", mf, sleep=sleep, monotonic=clock.monotonic, timeout=600, poll=5)
    assert ok and clock.sleeps == 3, (ok, clock.sleeps, table)


def t21_gate_fails_at_once_on_failed_status_and_on_a_missing_document():
    w, mf = gate_setup()
    w.srv("pwq-test").docs[mf["units"][2]["documentId"]]["status"] = "failed"
    (ok, table), clock = run_gate(w, mf)
    assert not ok and clock.sleeps == 0 and "FAILED" in table.split("u03")[1], table
    w, mf = gate_setup()
    del w.srv("pwq-test").docs[mf["units"][0]["documentId"]]
    (ok, table), clock = run_gate(w, mf)
    assert not ok and "MISSING" in table.split("u01")[1].split("\n")[0], table


def t21_gate_with_zero_chunks_fails_and_defaults_are_ten_minutes():
    w, mf = gate_setup()
    w.srv("pwq-test").docs[mf["units"][0]["documentId"]].update(chunks=0, embedded=0)
    (ok, table), _ = run_gate(w, mf, timeout=10, poll=5)
    assert not ok and "NO CHUNKS" in table
    assert kb.GATE_TIMEOUT == 600 and kb.GATE_POLL == 5


def t21_gate_cli_exit_codes():
    w, m = FakeWorld(), tmpm()
    run_ingest(w, mini_copy(), manifest=m)
    mf = kb.load_manifest(m)
    with contextlib.redirect_stdout(io.StringIO()):
        assert kb.main(["gate", "--ws", "pwq-test", "--manifest", str(m)], fetcher=w, targets=FAKE_TARGETS) == 0
        w.srv("pwq-test").docs[mf["units"][0]["documentId"]]["status"] = "failed"
        assert kb.main(["gate", "--ws", "pwq-test", "--manifest", str(m)], fetcher=w, targets=FAKE_TARGETS) == 1


def diverge_world():
    w, m, copy = FakeWorld(), tmpm(), mini_copy()
    run_ingest(w, copy, manifest=m)
    return w, m, copy


def run_diverge(w, m, copy, ws="pwq-test"):
    return kb.diverge(copy, ws=ws, manifest=kb.load_manifest(m), fetcher=w)


def t22_diverge_is_clean_on_a_faithful_workspace():
    w, m, copy = diverge_world()
    assert run_diverge(w, m, copy) == []


def t22_missing_stale_duplicate_orphan_and_title_drift_each_by_unit():
    w, m, copy = diverge_world()
    ids = {u["id"]: u["documentId"] for u in kb.load_manifest(m)["units"]}
    srv = w.srv("pwq-test")
    del srv.docs[ids["u03"]]
    assert run_diverge(w, m, copy) == [("MISSING", "u03", "Gamma title")]
    w, m, copy = diverge_world()
    w.srv("pwq-test").docs[ids["u01"]]["text"] += " "  # trailing whitespace is a difference
    assert run_diverge(w, m, copy) == [("STALE", "u01", "Alpha title")]
    w, m, copy = diverge_world()
    w.srv("pwq-test").post(dict(text="Gotcha: Beta title\nother\n", title="Beta title", producedBy="cobb"))
    assert run_diverge(w, m, copy) == [("DUPLICATE", "u02", "Beta title")]
    w, m, copy = diverge_world()
    gone = kb.render_copy(kb.parse_source(MINI_SRC.replace("## Gamma title\n\ngamma body\n", "")), kb.build_units(
        kb.parse_source(MINI_SRC.replace("## Gamma title\n\ngamma body\n", ""))), "2026-11-09", "abc1234")
    assert run_diverge(w, m, gone) == [("ORPHAN", "u03", "Gamma title")]
    w, m, copy = diverge_world()
    w.srv("pwq-test").docs[ids["u01"]]["title"] = "Alpha retitled"
    assert run_diverge(w, m, copy) == [("TITLE-DRIFT", "u01", "Alpha title")]


def t22_a_43rd_unit_is_missing_until_it_is_ingested():
    w, m, copy = diverge_world()
    s = kb.parse_source(MINI_SRC + "\n## Delta title\n\ndelta body\n")
    bigger = kb.render_copy(s, kb.build_units(s), "2026-11-09", "abc1234")
    assert run_diverge(w, m, bigger) == [("MISSING", "u04", "Delta title")]
    assert "Delta title" in kb.listing(bigger)  # shown in the D6 listing
    run_ingest(w, bigger, manifest=m)
    assert run_diverge(w, m, bigger) == []


def t22_diverge_cli_exit_codes_and_report_lines():
    w, m, copy = diverge_world()
    cp = Path(tempfile.mkdtemp(), "copy.md")
    cp.write_text(copy)
    args = ["diverge", "--copy", str(cp), "--ws", "pwq-test", "--manifest", str(m)]
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        assert kb.main(args, fetcher=w, targets=FAKE_TARGETS) == 0
        w.srv("pwq-test").docs[kb.load_manifest(m)["units"][0]["documentId"]]["text"] += "x"
        assert kb.main(args, fetcher=w, targets=FAKE_TARGETS) == 1
    assert "STALE u01 Alpha title" in out.getvalue()


def t23_a_unit_hidden_from_the_listing_is_caught_by_diverge():
    w, m, copy = diverge_world()
    block = unit_block(copy, "u02")
    hidden = block.replace("<!-- unit u02 -->\n", "").replace("## Beta title", "### Beta title")
    found = run_diverge(w, m, copy.replace(block, hidden))
    assert ("UNLISTED", "-", "Beta title") in found, found
    assert "Beta title" not in "".join(ln for ln in kb.listing(copy.replace(block, hidden)).split("\n") if "Beta" in ln and "###" not in ln)
    # a fence whose heading was demoted but still fenced is a stale body, found as such
    demoted = block.replace("## Beta title", "### Beta title")
    found = run_diverge(w, m, copy.replace(block, demoted))
    assert [k for k, u, _ in found if u == "u02"] == ["STALE"], found
    assert run_diverge(w, m, copy) == []  # control


def t23_real_transport_speaks_http_to_a_loopback_listener():
    import http.server, threading
    seen = []

    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers["Content-Length"])
            seen.append((self.command, self.path, self.headers["Content-Type"], json.loads(self.rfile.read(n))))
            self.send_response(201); self.send_header("Content-Type", "application/json"); self.end_headers()
            self.wfile.write(b'{"documentId": "x1"}')

        def do_DELETE(self):
            seen.append((self.command, self.path, None, None))
            self.send_response(404); self.send_header("Content-Type", "application/json"); self.end_headers()
            self.wfile.write(b'{"detail": "gone"}')

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        assert srv.server_address[1] not in (8200, 8210)
        status, body = kb.urllib_transport("POST", base + "/documents", {"text": "t", "producedBy": "cobb"})
        assert (status, body) == (201, {"documentId": "x1"})
        assert seen[0] == ("POST", "/documents", "application/json", {"text": "t", "producedBy": "cobb"})
        assert kb.urllib_transport("DELETE", base + "/documents/q", None) == (404, {"detail": "gone"})
    finally:
        srv.shutdown()
        srv.server_close()


def t23_real_fetcher_snippets_compile_and_name_the_gate_query():
    compile(kb.FETCH_SNIPPET, "fetch", "exec")
    for frag in ("-[:HAS_CHUNK]->(c:Chunk)", "count(c.embedding)", "d.status", "currentVersion"):
        assert frag in kb.FETCH_SNIPPET, frag


def t19_cli_refusals_exit_3_before_any_request_and_say_why():
    w, m = FakeWorld(), tmpm()
    cp = Path(tempfile.mkdtemp(), "copy.md")
    cp.write_text(copy_text())  # the CLI reads the pinned source for the waived span, so use the real pin
    base = ["--copy", str(cp), "--manifest", str(m)]
    kw = dict(transport=w.transport, fetcher=w, targets=FAKE_TARGETS)
    for args, msg in (
        (["ingest", "--ws", "pwq-test", "--base-url", PROD_URL], "does not match"),
        (["ingest", "--ws", "agent-team", "--base-url", PROD_URL], "--confirm-production"),
    ):
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            assert kb.main(args + base, **kw) == 3, args
        assert msg in err.getvalue(), (args, err.getvalue())
        assert w.requests == []
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        assert kb.main(["ingest", "--ws", "pwq-test", "--base-url", TEST_URL] + base, **kw) == 0
    assert out.getvalue().count(" ingest ") == 36 and w.post_count() == 36


def t20_cli_production_retire_prints_the_curl_line_and_deletes_nothing():
    w, m = FakeWorld(), tmpm()
    cp = Path(tempfile.mkdtemp(), "copy.md")
    cp.write_text(copy_text())
    kw = dict(transport=w.transport, fetcher=w, targets=FAKE_TARGETS)
    with contextlib.redirect_stdout(io.StringIO()):
        assert kb.main(["ingest", "--copy", str(cp), "--manifest", str(m), "--ws", "agent-team", "--base-url", PROD_URL,
                        "--confirm-production"], **kw) == 0
    did = kb.load_manifest(m)["units"][0]["documentId"]
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        assert kb.main(["retire", "--copy", str(cp), "--manifest", str(m), "--unit", "u01", "--ws", "agent-team",
                        "--base-url", PROD_URL], **kw) == 3
    assert f"curl -X DELETE {PROD_URL}/documents/{did}" in err.getvalue()
    assert w.delete_count() == 0


def t23_real_pin_copy_ingests_with_the_pinned_waived_span_and_diverges_clean():
    w, m = FakeWorld(), tmpm()
    res = kb.ingest(copy_text(), ws="pwq-test", base_url=TEST_URL, manifest_path=m, transport=w.transport, fetcher=w,
                    targets=FAKE_TARGETS, read_source=lambda pin: pin_text())
    assert len(res) == 36 and {r.action for r in res} == {"ingest"}
    mf = kb.load_manifest(m)
    want = "\n".join(ln for ln in pin_text().split("\n") if ln.startswith(">"))
    assert mf["waivedSpans"] == [want] and want.startswith("> **This is a cache, not the source of truth.**")
    assert mf["pin"] == PIN and len(mf["units"]) == 36
    assert kb.diverge(copy_text(), ws="pwq-test", manifest=mf, fetcher=w) == []
    assert all(len(d["text"]) < 500_000 and len(d["title"]) <= 200 for d in w.srv("pwq-test").docs.values())


def t16_manifest_keeps_a_unit_dropped_from_the_copy_registered_for_orphan_detection():
    w, m, copy = FakeWorld(), tmpm(), mini_copy()
    run_ingest(w, copy, manifest=m)
    src = MINI_SRC.replace("## Gamma title\n\ngamma body\n", "")
    shrunk = kb.render_copy(kb.parse_source(src), kb.build_units(kb.parse_source(src)), "2026-11-09", "abc1234")
    run_ingest(w, shrunk, manifest=m)
    mf = kb.load_manifest(m)
    assert [u["id"] for u in mf["units"]] == ["u01", "u02", "u03"], mf["units"]
    assert kb.diverge(shrunk, ws="pwq-test", manifest=mf, fetcher=w) == [("ORPHAN", "u03", "Gamma title")]


def t22_a_manifest_unit_missing_from_the_copy_whose_document_is_gone_is_not_an_orphan():
    w, m, copy = diverge_world()
    mf = kb.load_manifest(m)
    del w.srv("pwq-test").docs[next(u["documentId"] for u in mf["units"] if u["id"] == "u03")]
    src = MINI_SRC.replace("## Gamma title\n\ngamma body\n", "")
    gone = kb.render_copy(kb.parse_source(src), kb.build_units(kb.parse_source(src)), "2026-11-09", "abc1234")
    assert run_diverge(w, m, gone) == []


def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if re.match(r"t\d", n) and callable(f)]
    tests.sort(key=lambda nf: nf[0])
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception:  # noqa: BLE001 - report every failing test
            failed += 1
            print(f"FAIL {name}\n" + "".join(traceback.format_exc().splitlines(True)[-4:]))
    print(f"{len(tests) - failed}/{len(tests)} passed")
    print("ALL PASS" if not failed else "FAILURES")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
