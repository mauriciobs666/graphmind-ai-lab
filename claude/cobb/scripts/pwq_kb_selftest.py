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
    extra = "".join(f"<!-- unit u9{i:02d} -->\nGotcha: x{i}\n<!-- body -->\n## {'T' * 200}\n\nb\n" for i in range(25))
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
    cp = Path(tempfile.mkdtemp(), "copy.md")
    cp.write_text(mini_copy())
    mf = kb.load_manifest(m)
    argv = ["gate", "--ws", "pwq-test", "--manifest", str(m), "--copy", str(cp)]
    with contextlib.redirect_stdout(io.StringIO()):
        assert kb.main(argv, fetcher=w, targets=FAKE_TARGETS) == 0
        w.srv("pwq-test").docs[mf["units"][0]["documentId"]]["status"] = "failed"
        assert kb.main(argv, fetcher=w, targets=FAKE_TARGETS) == 1


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
    # the edited copy also no longer matches the registered sha256 (F4): reported on top of the STALE body
    assert sorted(k for k, u, _ in found if u == "u02") == ["REGISTRY-STALE", "STALE"], found
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


# ======================================================================== S3b: T24-T30 (+ T21 eval gate clauses)
SKILL_PATH = REPO / "skills/agent-kb-retrieval/SKILL.md"
EXPECTED_PREFIX = ("Instruct: Given a coding agent's description of its current situation, retrieve the distilled "
                   "technique or rule that applies to it.\nQuery: ")


def ck(doc, score, seq=0):
    return dict(chunkId=f"{doc}-{seq}-{score}", text="t", documentId=doc, seq=seq, score=score)


class FakeSearch:
    """Scripted `search_documents` + `get_document`. `script` is a dict situation->chunks or a function."""

    def __init__(self, script=None, world=None, ws="pwq-test"):
        self.script, self.world, self.ws, self.calls, self.titles = script or {}, world, ws, [], {}

    def search(self, ws, query, limit):
        self.calls.append((ws, query, limit))
        sit = query.split("Query: ", 1)[1] if "Query: " in query else query
        got = self.script(sit) if callable(self.script) else self.script.get(sit, [])
        return list(got)[:limit]  # the server honours `limit`

    def get_document(self, ws, did):
        if did in self.titles:
            return {"documentId": did, "title": self.titles[did]}
        d = self.world.srv(ws).docs.get(did) if self.world else None
        return {"documentId": did, "title": d["title"]} if d else None


def evaluate(pos, neg, **kw):
    kw.setdefault("floor", 0.43)
    kw.setdefault("feature_docs", {"D1", "D2", "D3", "D4", "U21"})
    return kb.evaluate(pos, neg, **kw)


def P(id_, section, chunks, tail=False):
    return dict(id=id_, section=section, tail=tail, chunks=chunks)


def N(id_, kind, chunks):
    return dict(id=id_, kind=kind, chunks=chunks)


SEC_DOC = {1: "D1", 2: "D2", 3: "D3", 4: "D4"}


def t24_recall_floor_recall_mrr_and_distinct_documents():
    pos = [
        P("p1", 1, [ck("D1", 0.30), ck("X", 0.35)]),
        P("p2", 2, [ck("X", 0.20), ck("D2", 0.43)]),  # exactly at the floor: kept (reject only score > floor)
        P("p3", 3, [ck("X", 0.20), ck("Y", 0.30), ck("D3", 0.44)]),  # raw hit, floor miss
        P("p4", 4, [ck("X", 0.20), ck("X", 0.25, 1)]),  # miss
    ]
    m = evaluate(pos, [], unit_doc=SEC_DOC)
    assert (m["raw_hits"], m["floor_hits"], m["n_pos"]) == (3, 2, 4), m
    assert m["recall_raw"] == 0.75 and m["recall_floor"] == 0.5
    assert abs(m["mrr"] - (1 + 0.5 + 1 / 3 + 0) / 4) < 1e-12, m["mrr"]
    assert m["distinct_docs_mean"] == (2 + 2 + 3 + 1) / 4  # distinct documents among the scored chunks, per query


def t24_floor_value_and_direction_matter():
    pos = [P("p", 1, [ck("D1", 0.44)])]
    assert evaluate(pos, [], unit_doc=SEC_DOC, floor=0.43)["floor_hits"] == 0
    assert evaluate(pos, [], unit_doc=SEC_DOC, floor=0.44)["floor_hits"] == 1  # inclusive
    assert evaluate(pos, [], unit_doc=SEC_DOC, floor=0.50)["floor_hits"] == 1
    assert evaluate([P("p", 1, [ck("D1", 0.10)])], [], unit_doc=SEC_DOC, floor=0.05)["floor_hits"] == 0
    assert evaluate(pos, [], unit_doc=SEC_DOC, floor=0.43)["raw_hits"] == 1


def t24_head_and_tail_probe_recall_are_reported_separately():
    pos = [P("h1", 1, [ck("D1", 0.3)]), P("h2", 2, [ck("X", 0.3)]),
           P("t1", 3, [ck("D3", 0.3)], tail=True), P("t2", 4, [ck("D4", 0.3)], tail=True)]
    m = evaluate(pos, [], unit_doc=SEC_DOC)
    assert m["head_recall"] == 0.5 and m["tail_recall"] == 1.0 and (m["n_head"], m["n_tail"]) == (2, 2)


def t25_merged_unit_hit_counts_for_each_member_section():
    unit_doc = {21: "U21", 22: "U21", 23: "U21", 3: "D3"}
    pos = [P("a", 21, [ck("U21", 0.3)]), P("b", 22, [ck("U21", 0.3)]), P("c", 23, [ck("U21", 0.3)]),
           P("d1", 3, [ck("X", 0.3)]), P("d2", 3, [ck("D3", 0.3)])]
    m = evaluate(pos, [], unit_doc=unit_doc)
    assert m["covered_sections"] == 4 and m["n_sections"] == 4 and m["floor_hits"] == 4
    pos = [P("a", 21, [ck("U21", 0.3)]), P("b", 22, [ck("U21", 0.3)]), P("c", 23, [ck("X", 0.3)]),
           P("d1", 3, [ck("X", 0.3)]), P("d2", 3, [ck("Y", 0.3)])]
    m = evaluate(pos, [], unit_doc=unit_doc)
    assert (m["covered_sections"], m["n_sections"]) == (2, 4)  # 23 and 3 are uncovered


def t26_false_accept_counts_only_feature_documents_under_the_floor():
    neg = [N("n1", "off", [ck("TEAM", 0.30)]),            # not a feature document
           N("n2", "near", [ck("D1", 0.42)]),             # feature document under the floor: accept
           N("n3", "near", [ck("D1", 0.44)]),             # over the floor: rejected
           N("n4", "off", [ck("X", 0.2), ck("D2", 0.43)]),  # exactly at the floor: accept
           N("n5", "near", [])]
    m = evaluate([], neg, unit_doc=SEC_DOC)
    assert m["false_accepts"] == {"off": 1, "near": 1}, m["false_accepts"]
    assert m["n_neg"] == {"off": 2, "near": 3}


def t24_gate_verdict_thresholds_are_inclusive():
    base = dict(recall_floor=0.9, n_pos=10, floor_hits=9, covered_sections=3, n_sections=3,
                false_accepts={"off": 0, "near": 3}, n_neg={"off": 15, "near": 15})
    assert kb.ac3_verdict(base)["pass"] is True  # 9/10 and 12/15 rejected: both exactly at the targets
    assert kb.ac3_verdict(dict(base, floor_hits=8))["pass"] is False
    assert kb.ac3_verdict(dict(base, false_accepts={"off": 0, "near": 4}))["pass"] is False
    assert kb.ac3_verdict(dict(base, false_accepts={"off": 1, "near": 0}))["pass"] is False
    assert kb.ac3_verdict(dict(base, covered_sections=2))["pass"] is False


def t27_prefix_is_read_from_the_fenced_literal_by_string_extraction():
    tmpl = kb.extract_prefix(SKILL_PATH.read_text())
    assert tmpl == EXPECTED_PREFIX + "{situation}", repr(tmpl)
    assert kb.build_query(tmpl, "my situation") == EXPECTED_PREFIX + "my situation"
    assert kb.build_query(tmpl, "a {situation} b") == EXPECTED_PREFIX + "a {situation} b"  # no re-substitution


def t27_an_altered_literal_is_followed_and_a_malformed_one_aborts():
    text = SKILL_PATH.read_text()
    altered = text.replace("retrieve the distilled technique", "retrieve the OTHER technique")
    assert "OTHER technique" in kb.extract_prefix(altered)  # read from the file, not a constant
    for bad in (text.replace("\\nQuery: {situation}", " Query: {situation}"),           # no newline escape
                text.replace("{situation}\"\n```", "\"\n```", 1),                       # no placeholder
                text.replace('f"Instruct:', '"Instruct:', 1),                         # not an f-string literal
                text.replace("The query-instruction prefix", "A different heading"),  # fence not found
                text.replace("\\nQuery: {situation}", "\\tQuery: {situation}"),       # unknown escape
                text.replace("retrieve the distilled", "retrieve\\tthe distilled")):    # extra escape beside a good \\n
        raises(kb.KBError, kb.extract_prefix, bad)


def t27_extraction_never_evaluates_code():
    text = SKILL_PATH.read_text().replace("{situation}\"\n```", "{situation}{__import__('os').getpid()}\"\n```", 1)
    assert "getpid" in kb.extract_prefix(text)  # kept as inert text


def t28_set_b_path_comes_only_from_the_argument_or_the_env_var():
    raises(kb.KBError, kb.resolve_set_b_path, None, {})
    assert kb.resolve_set_b_path("/tmp/x/b.json", {}) == Path("/tmp/x/b.json")
    assert kb.resolve_set_b_path(None, {"PWQ_SET_B": "/tmp/y/b.json"}) == Path("/tmp/y/b.json")
    assert kb.resolve_set_b_path("/tmp/x/b.json", {"PWQ_SET_B": "/tmp/y/b.json"}) == Path("/tmp/x/b.json")  # arg wins
    e = raises(kb.KBError, kb.resolve_set_b_path, str(REPO / "claude/cobb/scripts/fixtures/b.json"), {})
    assert "outside the repo" in str(e)


def write_set(path, positives, negatives):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"positives": positives, "negatives": negatives}))
    return Path(path)


def eval_env():
    w, m = FakeWorld(), tmpm()
    kb.ingest(copy_text(), ws="pwq-test", base_url=TEST_URL, manifest_path=m, transport=w.transport, fetcher=w,
              targets=FAKE_TARGETS, read_source=lambda p: pin_text())
    docs = {u["id"]: u["documentId"] for u in kb.load_manifest(m)["units"]}
    sec_unit = {s.index: u.id for u in units() for s in u.members}
    return w, m, (lambda n: docs[sec_unit[n]])


def default_copy_path() -> str:
    if "copy_path" not in _cache:
        _cache["copy_path"] = str(Path(tempfile.mkdtemp(), "copy.md"))
        Path(_cache["copy_path"]).write_text(copy_text())
    return _cache["copy_path"]


def run_eval(extra, search, world, m, ws="pwq-test", fetcher=None):
    out, err = io.StringIO(), io.StringIO()
    if "--copy" not in extra:  # the gate compares the manifest with the copy's unit ids (F7)
        extra = extra + ["--copy", default_copy_path()]
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = kb.main(["eval", "--ws", ws, "--base-url", FAKE_TARGETS[ws], "--manifest", str(m)] + extra,
                       searcher=search, fetcher=fetcher or world, targets=FAKE_TARGETS)
    return code, out.getvalue(), err.getvalue()


def b_parts(doc_of):
    """A well-shaped Set B (84 positives, 2 per section 1-42; 30 negatives, 15 near / 15 off) plus the
    scripted search results. B01/B02/B03/BN1/BN2 are the named probes the tests tweak; the rest is filler."""
    pos = [dict(id="B01", section=1, query="sit one"), dict(id="B02", section=2, query="sit two", tail=True),
           dict(id="B03", section=22, query="sit twentytwo")]
    neg = [dict(id="BN1", kind="off", query="neg off"), dict(id="BN2", kind="near", query="neg near")]
    script = {"sit one": [ck(doc_of(1), 0.30), ck("TEAM", 0.4)], "sit two": [ck("TEAM", 0.3), ck(doc_of(2), 0.41)],
              "sit twentytwo": [ck(doc_of(22), 0.2)], "neg off": [ck("TEAM", 0.3)], "neg near": [ck(doc_of(5), 0.40)]}
    have = {1: 1, 2: 1, 22: 1}
    k = 0
    for n in range(1, 43):
        for _ in range(2 - have.get(n, 0)):
            k += 1
            pos.append(dict(id=f"BF{k:02d}", section=n, query=f"fill {k}"))
            script[f"fill {k}"] = [ck(doc_of(n), 0.2)]
    for j in range(14):
        for kind in ("off", "near"):
            neg.append(dict(id=f"BN{kind[0]}{j:02d}", kind=kind, query=f"neg {kind} {j}"))
            script[f"neg {kind} {j}"] = [ck("TEAM", 0.3)]
    return pos, neg, script


def b_set(tmpdir, doc_of):
    pos, neg, script = b_parts(doc_of)
    return write_set(Path(tmpdir, "sealed", "b.json"), pos, neg), script


def t28_probe_prints_rank1_source_and_score_only_writes_no_sentinel_and_needs_no_gate():
    w, m, doc_of = eval_env()
    tmpd = tempfile.mkdtemp()
    bp, script = b_set(tmpd, doc_of)
    w.srv("pwq-test").docs[doc_of(1)]["embedded"] = 0  # the gate would fail: probe is exempt
    srch = FakeSearch(script, w)
    code, out, err = run_eval(["--set", "B", "--probe", "--set-b-path", str(bp), "--timeout", "0"], srch, w, m)
    assert code == 0, (code, err)
    t1 = w.srv("pwq-test").docs[doc_of(1)]["title"]
    assert any(ln.startswith("B01 ") and doc_of(1) in ln and t1 in ln and "0.3" in ln for ln in out.split("\n")), out
    assert "B03 " in out and "BN1 " in out
    assert "recall" not in out.lower() and "mrr" not in out.lower() and "false" not in out.lower(), out
    assert not Path(str(bp) + ".scored").exists() and list(Path(tmpd, "sealed").iterdir()) == [bp]
    assert len(srch.calls) == 114 and all(c[0] == "pwq-test" for c in srch.calls)


def t28_probe_flags_a_suspiciously_close_rank1_as_a_leak_sign():
    w, m, doc_of = eval_env()
    bp, script = b_set(tempfile.mkdtemp(), doc_of)
    script["sit one"] = [ck(doc_of(1), 0.24)]
    code, out, _ = run_eval(["--set", "B", "--probe", "--set-b-path", str(bp)], FakeSearch(script, w), w, m)
    line = next(ln for ln in out.split("\n") if ln.startswith("B01 "))
    assert "LEAK?" in line and not any("LEAK?" in ln for ln in out.split("\n") if ln.startswith("B02 "))
    script["sit one"] = [ck(doc_of(1), 0.25)]
    out = run_eval(["--set", "B", "--probe", "--set-b-path", str(bp)], FakeSearch(script, w), w, m)[1]
    assert "LEAK?" not in next(ln for ln in out.split("\n") if ln.startswith("B01 "))  # threshold is < 0.25


def t28_sealed_once_scores_writes_the_sentinel_and_a_second_run_refuses():
    w, m, doc_of = eval_env()
    tmpd = tempfile.mkdtemp()
    bp, script = b_set(tmpd, doc_of)
    srch = FakeSearch(script, w)
    args = ["--set", "B", "--sealed-once", "--set-b-path", str(bp), "--floor", "0.43"]
    code, out, err = run_eval(args, srch, w, m)
    assert code == 0, (code, err, out)
    sentinel = Path(str(bp) + ".scored")
    assert sentinel.is_file() and json.loads(sentinel.read_text())["floor"] == 0.43
    assert "floor=0.43" in out and "recall_floor" in out and "recall_raw" in out
    n = len(srch.calls)
    code, out2, err2 = run_eval(args, srch, w, m)
    assert code == 5 and "already scored" in err2 and len(srch.calls) == n and "recall" not in out2
    # the sentinel belongs to this Set B file: another file is not blocked
    bp2, script2 = b_set(tempfile.mkdtemp(), doc_of)
    assert run_eval(["--set", "B", "--sealed-once", "--set-b-path", str(bp2), "--floor", "0.43"], FakeSearch(script2, w), w, m)[0] == 0
    assert Path(str(bp2) + ".scored").is_file()


def t28_sealed_report_has_aggregates_only_no_per_query_failures():
    w, m, doc_of = eval_env()
    bp, script = b_set(tempfile.mkdtemp(), doc_of)
    script["sit one"] = [ck("TEAM", 0.3)]  # a miss
    code, out, _ = run_eval(["--set", "B", "--sealed-once", "--set-b-path", str(bp), "--floor", "0.43"], FakeSearch(script, w), w, m)
    assert code == 0 and "B01" not in out and "sit one" not in out and "BN2" not in out
    assert "floor_hits=83" in out and "n_pos=84" in out


def t28_a_failing_gate_refuses_before_the_sentinel_is_burnt():
    w, m, doc_of = eval_env()
    bp, script = b_set(tempfile.mkdtemp(), doc_of)
    w.srv("pwq-test").docs[doc_of(1)]["embedded"] = 0
    srch = FakeSearch(script, w)
    code, out, err = run_eval(["--set", "B", "--sealed-once", "--set-b-path", str(bp), "--floor", "0.43", "--timeout", "0"], srch, w, m)
    assert code == 4 and "gate" in err and srch.calls == []
    assert not Path(str(bp) + ".scored").exists()


def t28_set_b_needs_a_mode_and_a_path():
    w, m, doc_of = eval_env()
    bp, script = b_set(tempfile.mkdtemp(), doc_of)
    srch = FakeSearch(script, w)
    code, out, err = run_eval(["--set", "B", "--set-b-path", str(bp), "--floor", "0.43"], srch, w, m)
    assert code == 2 and "--probe or --sealed-once" in err and srch.calls == []
    env = dict(os.environ)
    os.environ.pop("PWQ_SET_B", None)
    try:
        code, out, err = run_eval(["--set", "B", "--probe"], srch, w, m)
    finally:
        os.environ.clear(); os.environ.update(env)
    assert code == 2 and "Set B path" in err


def t29_exclusion_takes_a_fixed_50_chunk_window_then_drops_then_truncates():
    assert kb.EVAL_WINDOW == 50 and kb.EVAL_TOPK == 5
    chunks = [ck("X", 0.10 + i * 0.001, i) for i in range(3)] + [ck(f"G{i}", 0.2 + i * 0.01, 0) for i in range(8)]
    srch = FakeSearch({"q": chunks})
    got = kb.window_chunks(srch, "pwq-test", "q", excluded={"X"})
    assert [c["documentId"] for c in got] == ["G0", "G1", "G2", "G3", "G4"], got  # five scored, X's three dropped first
    assert srch.calls == [("pwq-test", "q", 50)]
    # one excluded document holding far more than 5 + |excluded| of the top chunks (silent window arg)
    many = [ck("X", 0.1 + i * 0.001, i) for i in range(40)] + [ck(f"G{i}", 0.3 + i * 0.01) for i in range(10)]
    got = kb.window_chunks(FakeSearch({"q": many}), "pwq-test", "q", excluded={"X"})
    assert [c["documentId"] for c in got] == ["G0", "G1", "G2", "G3", "G4"]
    # no exclusions: the first five, in the server's order
    assert [c["documentId"] for c in kb.window_chunks(FakeSearch({"q": chunks}), "pwq-test", "q", excluded=set())] == [
        "X", "X", "X", "G0", "G1"]


def t29_report_states_the_limit_is_exceeded_lists_excluded_ids_and_gives_unfiltered_numbers():
    w, m, doc_of = eval_env()
    tmpd = tempfile.mkdtemp()
    a = write_set(Path(tmpd, "a.json"), [dict(id="A01", section=1, query="s1")], [dict(id="NA1", kind="off", query="n1")])
    script = {"s1": [ck("CAPTURE", 0.10), ck("CAPTURE", 0.11, 1), ck(doc_of(1), 0.3)], "n1": [ck("CAPTURE", 0.2)]}
    args = ["--set", "A", "--set-a-path", str(a), "--floor", "0.43", "--exclude", "CAPTURE"]
    srch = FakeSearch(script, w)
    code, out, err = run_eval(args, srch, w, m)
    assert code == 0, err
    assert srch.calls and all(c[2] == 50 for c in srch.calls), srch.calls  # the fixed window, whatever |excluded|
    assert "limit=5 deliberately exceeded" in out and "window=50" in out and "excluded: CAPTURE" in out
    assert "filtered: " in out and "unfiltered: " in out
    f = next(ln for ln in out.split("\n") if ln.startswith("filtered: "))
    u = next(ln for ln in out.split("\n") if ln.startswith("unfiltered: "))
    assert "floor_hits=1" in f and "mrr=1" in f and "floor_hits=1" in u and "mrr=0.3333" in u  # rank 3 unfiltered


def t30_bm25_scores_match_the_formula_and_ranking_is_deterministic():
    corpus = [("a:0", "cat sat mat"), ("b:0", "cat cat dog"), ("c:0", "dog bird")]
    got = dict(kb.bm25_rank(corpus, "cat"))
    import math
    n, df, avg = 3, 2, (3 + 3 + 2) / 3
    idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
    want = lambda tf, ln: idf * tf * 2.5 / (tf + 1.5 * (1 - 0.75 + 0.75 * ln / avg))
    assert abs(got["a:0"] - want(1, 3)) < 1e-12 and abs(got["b:0"] - want(2, 3)) < 1e-12 and got["c:0"] == 0
    assert kb.BM25_K1 == 1.5 and kb.BM25_B == 0.75


def t30_bm25_ties_are_broken_by_key_independent_of_hash_seed():
    code = (
        "import importlib.util,sys\n"
        f"spec=importlib.util.spec_from_file_location('k', {str(Path(kb.__file__))!r}); m=importlib.util.module_from_spec(spec)\n"
        "sys.modules['k']=m; spec.loader.exec_module(m)\n"
        "corpus=[(k,'same words here') for k in ('zz:0','aa:0','mm:0','bb:0','qq:0','cc:0')]\n"
        "print(','.join(k for k,_ in m.bm25_rank(corpus,'words')))\n"
    )
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=dict(os.environ, PYTHONHASHSEED=str(sd))
                           ).stdout.strip() for sd in (0, 1, 2, 3, 4, 5, 6)}
    assert outs == {"aa:0,bb:0,cc:0,mm:0,qq:0,zz:0"}, outs


def t30_bm25_baseline_cli_scores_a_set_over_the_copys_chunks_after_the_gate():
    w, m, doc_of = eval_env()
    tmpd = tempfile.mkdtemp()
    cp = Path(tmpd, "copy.md"); cp.write_text(copy_text())
    cu = {u.id: u for u in kb.parse_copy(copy_text()).units}
    # a query made of words only unit u10's body has
    a = write_set(Path(tmpd, "a.json"), [dict(id="A10", section=10, query="whitespace-only strings pydantic min_length")],
                  [dict(id="NA", kind="off", query="kubernetes readiness probe")])
    srch = FakeSearch({}, w)
    code, out, err = run_eval(["--baseline", "bm25", "--set", "A", "--set-a-path", str(a), "--copy", str(cp)], srch, w, m)
    assert code == 0, err
    assert "baseline=bm25" in out and "recall_raw=1.0" in out and srch.calls == []  # offline
    w.srv("pwq-test").docs[doc_of(1)]["embedded"] = 0
    assert run_eval(["--baseline", "bm25", "--set", "A", "--set-a-path", str(a), "--copy", str(cp), "--timeout", "0"], srch, w, m)[0] == 4


def t21_scoring_modes_refuse_on_a_failing_gate_and_search_nothing_exempt_modes_run():
    w, m, doc_of = eval_env()
    tmpd = tempfile.mkdtemp()
    cp = Path(tmpd, "copy.md"); cp.write_text(copy_text())
    pos = [dict(id=f"A{n:02d}", section=n, query=f"sit {n}") for n in range(1, 43)]
    a = write_set(Path(tmpd, "a.json"), pos, [dict(id="NA1", kind="off", query="neg")])
    bp, script = b_set(tmpd, doc_of)
    w.srv("pwq-test").docs[doc_of(3)]["embedded"] = 0
    srch = FakeSearch({}, w)
    common = ["--timeout", "0"]
    refused = {
        "ac2": ["--ac2", "--set-a-path", str(a), "--copy", str(cp), "--floor", "0.43"],
        "sweep": ["--sweep", "--set", "A", "--set-a-path", str(a), "--floor", "0.43"],
        "setA": ["--set", "A", "--set-a-path", str(a), "--floor", "0.43"],
        "setB-sealed": ["--set", "B", "--sealed-once", "--set-b-path", str(bp), "--floor", "0.43"],
        "bm25": ["--baseline", "bm25", "--set", "A", "--set-a-path", str(a), "--copy", str(cp)],
    }
    for name, args in refused.items():
        code, out, err = run_eval(args + common, srch, w, m)
        assert code == 4 and "gate" in err and "u03" in err, (name, code, err)
        assert srch.calls == [], name
    for name, args in {"B-probe": ["--set", "B", "--probe", "--set-b-path", str(bp)],
                       "A-before": ["--set", "A", "--before", "--set-a-path", str(a), "--floor", "0.43"]}.items():
        code, out, err = run_eval(args + common, FakeSearch(script, w), w, m)
        assert code == 0, (name, code, err)
    # and with no manifest at all (before any ingest) the exempt modes still run
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = kb.main(["eval", "--ws", "pwq-test", "--base-url", TEST_URL, "--manifest", str(Path(tmpd, "none.json")),
                        "--set", "B", "--probe", "--set-b-path", str(bp)], searcher=FakeSearch(script, w), fetcher=w, targets=FAKE_TARGETS)
    assert code == 0, err.getvalue()


def t24_set_a_scoring_cli_reports_recall_and_names_the_best_document_for_misses():
    w, m, doc_of = eval_env()
    tmpd = tempfile.mkdtemp()
    pos = [dict(id="A01", section=1, query="s1"), dict(id="A02", section=2, query="s2")]
    a = write_set(Path(tmpd, "a.json"), pos, [dict(id="NA1", kind="near", query="n1")])
    script = {"s1": [ck(doc_of(1), 0.3)], "s2": [ck(doc_of(5), 0.31), ck(doc_of(2), 0.5)], "n1": [ck(doc_of(7), 0.42)]}
    code, out, err = run_eval(["--set", "A", "--set-a-path", str(a), "--floor", "0.43"], FakeSearch(script, w), w, m)
    assert code == 0, err
    assert "floor=0.43" in out and "floor_hits=1" in out and "raw_hits=2" in out and "n_pos=2" in out
    assert "false_accepts near=1/1 off=0/0" in out
    t5 = w.srv("pwq-test").docs[doc_of(5)]["title"]
    miss = next(ln for ln in out.split("\n") if ln.startswith("MISS A02"))
    assert doc_of(5) in miss and t5 in miss  # best distinct document, named via get_document
    assert not any(ln.startswith("MISS A01") for ln in out.split("\n"))


def t24_name_hit_takes_the_best_scoring_distinct_document():
    s = FakeSearch({}, None)
    s.titles = {"D2": "Two", "D9": "Nine"}
    chunks = [ck("D9", 0.31), ck("D2", 0.2), ck("D9", 0.1, 1)]
    assert kb.name_hit(s, "pwq-test", chunks) == ("D9", "Nine", 0.1)  # lowest score wins, not first position
    assert kb.name_hit(s, "pwq-test", []) is None


def t24_before_mode_lists_what_already_answers_the_queries():
    w, m, doc_of = eval_env()
    tmpd = tempfile.mkdtemp()
    a = write_set(Path(tmpd, "a.json"), [dict(id="A01", section=1, query="s1"), dict(id="A02", section=2, query="s2")],
                  [dict(id="NA1", kind="off", query="n1")])
    script = {"s1": [ck("CAPTURE1", 0.30), ck("OTHER", 0.5)], "s2": [ck("OTHER", 0.5)], "n1": [ck("OTHER", 0.5)]}
    code, out, err = run_eval(["--set", "A", "--before", "--set-a-path", str(a), "--floor", "0.43"], FakeSearch(script, w), w, m)
    assert code == 0, err
    assert "DUPLICATE-COVERAGE A01 CAPTURE1 0.3" in out and "DUPLICATE-COVERAGE A02" not in out and "NA1" not in out
    assert "queries_with_a_hit_under_floor=1/3" in out and "recall" not in out.lower()


def t24_sweep_covers_0_38_to_0_50_and_judges_adequacy_by_the_ml_criteria():
    assert kb.SWEEP_FLOORS == [0.38, 0.39, 0.40, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.50]
    pos = [P("p1", 1, [ck("D1", 0.40)]), P("p2", 2, [ck("D2", 0.30)])]
    neg = [N("n1", "near", [ck("D1", 0.45)]), N("n2", "off", [ck("D3", 0.47)]), N("n3", "near", [ck("X", 0.3)])]
    rows = {r["floor"]: r for r in kb.sweep(pos, neg, unit_doc=SEC_DOC, feature_docs={"D1", "D2", "D3", "D4"}, keep=0.43)}
    assert rows[0.38]["recall_floor"] == 0.5 and rows[0.40]["recall_floor"] == 1.0 and rows[0.50]["recall_floor"] == 1.0
    assert rows[0.44]["near_fa"] == 0 and rows[0.45]["near_fa"] == 1 and rows[0.46]["off_fa"] == 0 and rows[0.47]["off_fa"] == 1
    assert rows[0.43]["adequate"] is True      # recall within 0.03 of raw, no off FA, near FA 0, worst positive 0.40 <= 0.405
    assert rows[0.40]["adequate"] is False     # worst positive 0.40 is not 0.025 below 0.40
    assert rows[0.47]["adequate"] is False     # an off-domain false accept
    pos2 = [P("p1", 1, [ck("D1", 0.41)])]
    r = kb.sweep(pos2, [], unit_doc=SEC_DOC, feature_docs={"D1"}, keep=0.43)
    assert next(x for x in r if x["floor"] == 0.43)["adequate"] is False  # 0.43 - 0.41 = 0.02 < 0.025
    pos3 = [P("p1", 1, [ck("D1", 0.405)])]
    r = kb.sweep(pos3, [], unit_doc=SEC_DOC, feature_docs={"D1"}, keep=0.43)
    assert next(x for x in r if x["floor"] == 0.43)["adequate"] is True    # exactly 0.025: allowed
    # the recall criterion: a miss at the floor that raw top-5 would have found by more than 0.03
    pos4 = [P(f"p{i}", 1, [ck("D1", 0.44 if i == 0 else 0.3)]) for i in range(10)]
    r = kb.sweep(pos4, [], unit_doc=SEC_DOC, feature_docs={"D1"}, keep=0.43)
    assert next(x for x in r if x["floor"] == 0.43)["adequate"] is False


def t24_sweep_cli_prints_one_row_per_floor():
    w, m, doc_of = eval_env()
    a = write_set(Path(tempfile.mkdtemp(), "a.json"), [dict(id="A01", section=1, query="s1")], [dict(id="NA1", kind="off", query="n1")])
    script = {"s1": [ck(doc_of(1), 0.40)], "n1": [ck("TEAM", 0.3)]}
    code, out, err = run_eval(["--sweep", "--set", "A", "--set-a-path", str(a), "--floor", "0.43"], FakeSearch(script, w), w, m)
    assert code == 0, err
    rows = [ln for ln in out.split("\n") if ln.startswith("floor ")]
    assert len(rows) == 13 and rows[0].startswith("floor 0.38 ") and rows[-1].startswith("floor 0.5 ")
    assert "recall_floor=0 " in rows[0] + " " and "recall_floor=1 " in rows[3] + " "


def t31_ac2_checks_title_situation_and_last_paragraph_per_unit_and_the_title_via_get_document():
    w, m, doc_of = eval_env()
    tmpd = tempfile.mkdtemp()
    cp = Path(tmpd, "copy.md"); cp.write_text(copy_text())
    pos = [dict(id=f"A{n:02d}", section=n, query=f"sit {n}") for n in range(1, 43)]
    a = write_set(Path(tmpd, "a.json"), pos, [])
    ents = kb.copy_entries(copy_text())
    sec_unit = {s.index: u.id for u in units() for s in u.members}
    doc_by_unit = {e.id: doc_of(next(n for n, u in sec_unit.items() if u == e.id)) for e in ents}
    script = {}
    for e in ents:
        script[e.title] = [ck(doc_by_unit[e.id], 0.3)]
        script[e.text.rstrip().split("\n\n")[-1]] = [ck(doc_by_unit[e.id], 0.3, 2)]
    for n in range(1, 43):
        script[f"sit {n}"] = [ck(doc_by_unit[sec_unit[n]], 0.3)]
    args = ["--ac2", "--set-a-path", str(a), "--copy", str(cp), "--floor", "0.43"]
    code, out, err = run_eval(args, FakeSearch(script, w), w, m)
    assert code == 0, (err, out[-600:])
    assert "ac2: 36/36 units pass" in out and "u21 title=ok situation=ok tail=ok name=ok" in out
    # tail query returns another document -> that unit fails on tail only
    e5 = next(e for e in ents if e.id == "u05")
    bad = dict(script); bad[e5.text.rstrip().split("\n\n")[-1]] = [ck(doc_by_unit["u02"], 0.3)]
    code, out, err = run_eval(args, FakeSearch(bad, w), w, m)
    assert code == 1 and "u05 title=ok situation=ok tail=MISS name=ok" in out and "ac2: 35/36" in out
    # over the floor is a miss too
    bad = dict(script); bad["sit 5"] = [ck(doc_by_unit["u05"], 0.44)]
    assert "u05 title=ok situation=MISS" in run_eval(args, FakeSearch(bad, w), w, m)[1]
    bad = dict(script); bad["sit 22"] = [ck("OTHER", 0.3)]  # one of u21's three Set A queries misses
    assert "u21 title=ok situation=MISS" in run_eval(args, FakeSearch(bad, w), w, m)[1]
    # the unit's stored title is what get_document must return
    w.srv("pwq-test").docs[doc_by_unit["u07"]]["title"] = "something else"
    code, out, err = run_eval(args, FakeSearch(script, w), w, m)
    assert code == 1 and "u07 title=ok situation=ok tail=ok name=WRONG" in out


def t24_set_files_are_validated_and_sections_must_exist():
    d = tempfile.mkdtemp()
    ok = write_set(Path(d, "ok.json"), [dict(id="p", section=1, query="q")], [dict(id="n", kind="near", query="q")])
    s = kb.load_set(ok)
    assert s["positives"][0]["tail"] is False and s["negatives"][0]["kind"] == "near"
    for pos, neg in (([dict(id="p", section=1)], []), ([], [dict(id="n", kind="weird", query="q")]),
                     ([dict(id="p", section="x", query="q")], []), ([dict(id="p", section=1, query="")], [])):
        raises(kb.KBError, kb.load_set, write_set(Path(d, "bad.json"), pos, neg))
    Path(d, "junk.json").write_text("not json")
    raises(kb.KBError, kb.load_set, Path(d, "junk.json"))


def t24_rest_searcher_speaks_the_search_and_get_routes_of_the_validated_target():
    seen = []

    def tr(method, url, body=None):
        seen.append((method, url))
        return (200, [ck("D1", 0.2)]) if "/search" in url else (200, {"documentId": "D1", "title": "T"})

    s = kb.RestSearcher(kb.check_target("pwq-test", TEST_URL, FAKE_TARGETS), tr)
    assert s.search("pwq-test", "a b&c\nQuery: x", 50)[0]["documentId"] == "D1"
    assert seen[0] == ("GET", TEST_URL + "/documents/search?q=a+b%26c%0AQuery%3A+x&limit=50")
    assert s.get_document("pwq-test", "D1")["title"] == "T" and seen[1] == ("GET", TEST_URL + "/documents/D1")
    raises(kb.KBError, kb.RestSearcher(kb.check_target("pwq-test", TEST_URL, FAKE_TARGETS), lambda *a: (503, {})).search, "pwq-test", "q", 5)


def t24_mrr_uses_the_first_expected_chunk_rank():
    pos = [P("p", 1, [ck("X", 0.1), ck("D1", 0.2), ck("D1", 0.3, 1), ck("X", 0.4)])]
    assert evaluate(pos, [], unit_doc=SEC_DOC)["mrr"] == 0.5


def t30_bm25_top_returns_at_most_five_positive_scoring_chunks_best_first():
    corpus = [(f"u{i:02d}:000", "alpha " * (i + 1) + "filler words") for i in range(8)] + [("z:000", "nothing relevant")]
    top = kb.bm25_top(corpus, "alpha")
    assert len(top) == 5 and all(c["score"] > 0 for c in top) and "z" not in {c["documentId"] for c in top}
    assert [c["score"] for c in top] == sorted((c["score"] for c in top), reverse=True)
    few = kb.bm25_top([("a:000", "alpha"), ("b:000", "beta"), ("c:000", "gamma")], "alpha")
    assert [c["documentId"] for c in few] == ["a"]  # zero-score chunks are never returned
    assert kb.bm25_top(corpus, "alpha", k=2) == top[:2] and kb.EVAL_TOPK == 5


def _adequate(pos, neg, floor=0.43, feature=("D1", "D2")):
    rows = kb.sweep(pos, neg, unit_doc=SEC_DOC, feature_docs=set(feature), keep=0.43)
    return next(r for r in rows if r["floor"] == floor)["adequate"]


def t24_sweep_adequacy_boundaries_each_isolated():
    ok_pos = [P(f"p{i}", 1, [ck("D1", 0.30)]) for i in range(100)]
    # near-domain false accepts: 2 of 7 allowed, 3 of 7 not
    near = lambda k: [N(f"n{i}", "near", [ck("D1", 0.40)] if i < k else [ck("X", 0.3)]) for i in range(7)]
    assert _adequate(ok_pos, near(2)) is True and _adequate(ok_pos, near(3)) is False
    # an off-domain false accept is never allowed
    assert _adequate(ok_pos, [N("o", "off", [ck("D1", 0.40)])]) is False
    assert _adequate(ok_pos, [N("o", "off", [ck("D1", 0.44)])]) is True
    # recall within 0.03 of raw: 3 of 100 lost to the floor is allowed, 4 is not
    lost = lambda k: [P(f"p{i}", 1, [ck("D1", 0.44 if i < k else 0.30)]) for i in range(100)]
    assert _adequate(lost(3), []) is True and _adequate(lost(4), []) is False
    assert _adequate(lost(4), [], floor=0.50) is True  # at 0.50 nothing is lost and the worst found (0.44) is 0.06 below




# ======================================================================== G1 round (review F1-F8)
# ----------------------------------------------------------------------------- F1: version headers
def _hdr_block(uid="u21"):
    return unit_block(copy_text(), uid)


def t90_f1_extractor_follows_a_bold_list_continuation_after_one_lib():
    got = kb.extract_versions("starlette **1.3.1** (`a/.venv`) and **1.6.0** (`b/.venv`, `c/.venv`), both on httpx 0.28.1")
    assert got == [("starlette", "1.3.1"), ("starlette", "1.6.0"), ("httpx", "0.28.1")], got
    got = kb.extract_versions("starlette 1.3.1, 1.4.0 and 1.6.0 differ")
    assert got == [("starlette", "1.3.1"), ("starlette", "1.4.0"), ("starlette", "1.6.0")], got


def t90_f1_continuation_needs_a_dotted_token_and_never_reads_a_measurement():
    assert kb.extract_versions("starlette 1.3.1 and 2 retries") == [("starlette", "1.3.1")]
    assert kb.extract_versions("starlette 1.3.1 and 0.5 s later") == [("starlette", "1.3.1")]
    assert kb.extract_versions("starlette 1.3.1, then 1.6.0") == [("starlette", "1.3.1")]  # no separator word
    assert kb.extract_versions("took starlette-free 0.2 ms") == []


def t90_f1_extractor_crosses_one_hard_line_wrap_but_not_a_paragraph_break():
    assert kb.extract_versions("Measured on CPython\n3.12.3 across") == [("CPython", "3.12.3")]
    assert kb.extract_versions("Measured on **CPython**\n3.12.3") == [("CPython", "3.12.3")]
    assert kb.extract_versions("uses pytest\n\n9.1 is next") == []


def t90_f1_the_pin_header_for_u21_member_3_names_both_starlette_pins_and_cpython():
    block = _hdr_block("u21")
    line = next(ln for ln in block.split("\n") if ln.startswith("Versions[3]:"))
    pairs = set(line.split(": ", 1)[1].split("; "))
    assert {"starlette 1.3.1", "starlette 1.6.0", "httpx 0.28.1", "CPython 3.12.3"} <= pairs, line


def _with_versions_line(uid, label, new_value):
    text = copy_text()
    block = unit_block(text, uid)
    old = next(ln for ln in block.split("\n") if ln.startswith(label + ":"))
    return text.replace(block, block.replace(old, f"{label}: {new_value}", 1)), old


def t90_f1_verify_copy_accepts_a_header_that_adds_a_valid_hand_pair():
    # python 3.12.3 is not extracted for u26 (the body says CPython) but passes version_pair_ok
    text, old = _with_versions_line("u26", "Versions", "CPython 3.12.3; python 3.12.3")
    assert old == "Versions: CPython 3.12.3", old
    assert verify(text) == []


def t90_f1_verify_copy_rejects_an_invented_hand_pair_and_names_it():
    text, _ = _with_versions_line("u26", "Versions", "CPython 3.12.3; flask 3.12.3")
    probs = verify(text)
    assert any(p.startswith("u26:") and "invented version flask" in p for p in probs), probs
    text, _ = _with_versions_line("u26", "Versions", "CPython 3.12.3; python 3.99")
    assert any("invented version python '3.99'" in p for p in verify(text)), verify(text)


def t90_f1_verify_copy_rejects_a_header_that_omits_an_extracted_pair():
    text, _ = _with_versions_line("u21", "Versions[3]", "starlette 1.3.1; httpx 0.28.1; CPython 3.12.3")
    probs = verify(text)
    assert any(p.startswith("u21:") and "omits" in p and "starlette" in p and "1.6.0" in p for p in probs), probs
    text, _ = _with_versions_line("u26", "Versions", "none stated")
    assert any(p.startswith("u26:") and "omits" in p for p in verify(text)), verify(text)


def t90_f1_a_hand_pair_is_judged_against_its_own_member_body_not_a_sibling():
    # u21 member 1 body never says 'httpx 0.28.1'... but member 2 does: member 1 may not borrow it
    text, old = _with_versions_line("u21", "Versions[1]", "starlette 1.3.1; fastapi 0.139.0; anyio 4.14.1; httpx 0.28.1")
    assert any(p.startswith("u21:") and "invented version httpx" in p for p in verify(text)), verify(text)


def t90_f1_other_header_lines_are_still_compared_exactly():
    text, _ = _with_versions_line("u21", "Verified[3]", "2020-01-01")
    assert any(p.startswith("u21: header mismatch") for p in verify(text)), verify(text)
    text, _ = _with_versions_line("u26", "Versions", "CPython 3.12.3; python 3.12.3")
    assert verify(text) == []
    text = text.replace("Versions: CPython 3.12.3; python 3.12.3\n", "Versions: CPython 3.12.3; python 3.12.3\nVersions: x 1.0\n", 1)
    assert any(p.startswith("u26: header mismatch") for p in verify(text)), verify(text)


# ----------------------------------------------------------------------------- F5: copy uniqueness
def t91_f5_parse_copy_rejects_a_duplicate_unit_id():
    text = copy_text()
    dup = text + unit_block(text, "u01")
    e = raises(kb.KBError, kb.parse_copy, dup)
    assert "duplicate unit id" in str(e) and "u01" in str(e), e
    assert raises(kb.KBError, kb.copy_entries, dup) is not None
    assert any("duplicate unit id" in p for p in verify(dup))


def t91_f5_parse_copy_rejects_a_duplicate_title_under_another_id():
    text = copy_text()
    twin = unit_block(text, "u01").replace("<!-- unit u01 -->", "<!-- unit u99 -->", 1)
    e = raises(kb.KBError, kb.parse_copy, text + twin)
    assert "duplicate title" in str(e) and "u99" in str(e) and "u01" in str(e), e


def t91_f5_distinct_ids_and_titles_still_parse():
    assert len(kb.parse_copy(copy_text()).units) == 36


# ----------------------------------------------------------------------------- F7: gate on a partial manifest
def t92_f7_run_gate_fails_on_a_unit_the_copy_expects_but_the_manifest_lacks():
    w, mf = gate_setup()
    ids = [u["id"] for u in mf["units"]]
    (ok, table), _ = run_gate(w, mf, expected_ids=ids)
    assert ok and "ABSENT" not in table
    (ok, table), clock = run_gate(w, mf, expected_ids=ids + ["u99"])
    assert not ok and clock.sleeps == 0, (ok, clock.sleeps)
    assert "ABSENT" in table.split("u99")[1].split("\n")[0], table
    # the check is by id, not by count: one unit swapped for another is still absent
    (ok, table), _ = run_gate(w, mf, expected_ids=ids[1:] + ["u99"])
    assert not ok and "u99" in table


def t92_f7_gate_cli_fails_when_the_manifest_is_partial_and_needs_the_copy():
    w, m = FakeWorld(), tmpm()
    copy = mini_copy()
    cp = Path(tempfile.mkdtemp(), "copy.md")
    cp.write_text(copy)
    run_ingest(w, copy, manifest=m)
    base = ["gate", "--ws", "pwq-test", "--manifest", str(m), "--copy", str(cp)]
    with contextlib.redirect_stdout(io.StringIO()) as out:
        assert kb.main(base, fetcher=w, targets=FAKE_TARGETS) == 0
    mf = kb.load_manifest(m)
    dropped = mf["units"].pop(1)
    m.write_text(json.dumps(mf))
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        assert kb.main(base, fetcher=w, targets=FAKE_TARGETS) == 1
    assert "ABSENT" in out.getvalue() and dropped["id"] in out.getvalue() and "FAILED" in out.getvalue(), out.getvalue()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
        code = kb.main(base[:-1] + [str(cp) + ".missing"], fetcher=w, targets=FAKE_TARGETS)
    assert code == 2 and "copy" in err.getvalue(), err.getvalue()


def t92_f7_eval_scoring_refuses_on_a_partial_manifest():
    w, m, doc_of = eval_env()
    mf = kb.load_manifest(m)
    gone = mf["units"].pop(3)
    m.write_text(json.dumps(mf))
    bp, script = b_set(tempfile.mkdtemp(), doc_of)
    srch = FakeSearch(script, w)
    code, out, err = run_eval(["--set", "B", "--sealed-once", "--set-b-path", str(bp), "--floor", "0.43", "--timeout", "0"], srch, w, m)
    assert code == 4 and gone["id"] in err and "ABSENT" in err and srch.calls == [], (code, err)
    assert not Path(str(bp) + ".scored").exists()


# ----------------------------------------------------------------------------- F3: coverage is any-of-two
def t93_f3_a_section_is_covered_when_either_of_its_two_queries_hits_in_either_order():
    hit, miss = [ck("D1", 0.30)], [ck("X", 0.30)]
    for first, second in ((hit, miss), (miss, hit)):
        m = evaluate([P("a", 1, first), P("b", 1, second)], [], unit_doc=SEC_DOC)
        assert (m["n_sections"], m["covered_sections"], m["covered_sections_raw"]) == (1, 1, 1), (first, m)
    m = evaluate([P("a", 1, miss), P("b", 1, miss)], [], unit_doc=SEC_DOC)
    assert (m["n_sections"], m["covered_sections"], m["covered_sections_raw"]) == (1, 0, 0), m
    # several sections at once, mixed orders: 1 hit/miss, 2 miss/hit, 3 miss/miss, 4 hit/hit
    pos = [P("a1", 1, hit), P("b1", 1, miss), P("a2", 2, [ck("X", 0.3)]), P("b2", 2, [ck("D2", 0.3)]),
           P("a3", 3, miss), P("b3", 3, miss), P("a4", 4, [ck("D4", 0.3)]), P("b4", 4, [ck("D4", 0.3)])]
    m = evaluate(pos, [], unit_doc=SEC_DOC)
    assert (m["n_sections"], m["covered_sections"], m["covered_sections_raw"]) == (4, 3, 3), m


def t93_f3_floor_coverage_and_raw_coverage_are_accumulated_separately():
    over = [ck("D1", 0.90)]  # the expected document is returned, but past the floor
    for first, second in ((over, [ck("X", 0.3)]), ([ck("X", 0.3)], over)):
        m = evaluate([P("a", 1, first), P("b", 1, second)], [], unit_doc=SEC_DOC)
        assert (m["covered_sections"], m["covered_sections_raw"]) == (0, 1), m
    m = evaluate([P("a", 1, over), P("b", 1, [ck("D1", 0.3)])], [], unit_doc=SEC_DOC)
    assert (m["covered_sections"], m["covered_sections_raw"]) == (1, 1), m


# ----------------------------------------------------------------------------- F2: Set B shape before the seal
def _malformed_cases(doc_of):
    """(why-substring, mutate(pos, neg)) pairs: each breaks exactly one clause of the ML §1 shape."""
    def drop_last_section(pos, neg):  # 84 positives over 41 sections: section 42's pair moves to 41
        for p in pos:
            if p["section"] == 42:
                p["section"] = 41

    def outside_range(pos, neg):  # 84 positives, 2 per section, but one section is 43 instead of 42
        for p in pos:
            if p["section"] == 42:
                p["section"] = 43

    return [
        ("positives", lambda pos, neg: pos.pop()),  # 83
        ("positives", lambda pos, neg: pos.append(dict(id="BX", section=1, query="x"))),  # 85
        ("section", drop_last_section),
        ("section", outside_range),
        ("section", lambda pos, neg: pos.__setitem__(-1, dict(pos[-1], section=1))),  # 84, but section 1 holds 3, 42 holds 1
        ("split", lambda pos, neg: next(n for n in neg if n["kind"] == "near").update(kind="off")),  # 16 off / 14 near
        ("split", lambda pos, neg: next(n for n in neg if n["kind"] == "off").update(kind="near")),  # 14 off / 16 near
        ("negatives", lambda pos, neg: neg.pop()),  # 29
        ("negatives", lambda pos, neg: neg.append(dict(id="BNX", kind="near", query="x"))),  # 31
        ("duplicate", lambda pos, neg: pos.__setitem__(0, dict(pos[0], id=pos[1]["id"]))),
    ]


def t94_f2_validate_set_b_accepts_the_spec_shape_and_names_each_violated_clause():
    w, m, doc_of = eval_env()
    pos, neg, _ = b_parts(doc_of)
    kb.validate_set_b({"positives": pos, "negatives": neg})
    assert (kb.SET_B_SECTIONS, kb.SET_B_PER_SECTION, kb.SET_B_NEAR, kb.SET_B_OFF) == (42, 2, 15, 15)
    for why, mutate in _malformed_cases(doc_of):
        pos, neg, _ = b_parts(doc_of)
        mutate(pos, neg)
        e = raises(kb.KBError, kb.validate_set_b, {"positives": pos, "negatives": neg})
        assert why in str(e), (why, str(e))


def t94_f2_a_malformed_set_b_refuses_in_probe_sealed_and_baseline_before_any_search_or_sentinel():
    w, m, doc_of = eval_env()
    for why, mutate in _malformed_cases(doc_of):
        for mode in (["--probe"], ["--sealed-once", "--floor", "0.43"], ["--baseline", "bm25", "--floor", "0.43"]):
            pos, neg, script = b_parts(doc_of)
            mutate(pos, neg)
            bp = write_set(Path(tempfile.mkdtemp(), "sealed", "b.json"), pos, neg)
            srch = FakeSearch(script, w)
            code, out, err = run_eval(["--set", "B", "--set-b-path", str(bp)] + mode, srch, w, m)
            assert code == 2 and why in err, (why, mode, code, err)
            assert srch.calls == [] and not Path(str(bp) + ".scored").exists(), (why, mode)
            assert "recall" not in out and "B01" not in out, (why, mode, out)


def t94_f2_a_refused_set_b_leaves_the_seal_for_the_corrected_set():
    w, m, doc_of = eval_env()
    pos, neg, script = b_parts(doc_of)
    bp = Path(tempfile.mkdtemp(), "sealed", "b.json")
    write_set(bp, pos[:-1], neg)
    args = ["--set", "B", "--sealed-once", "--set-b-path", str(bp), "--floor", "0.43"]
    assert run_eval(args, FakeSearch(script, w), w, m)[0] == 2
    write_set(bp, pos, neg)  # the corrected file at the same path scores: nothing was burnt
    assert run_eval(args, FakeSearch(script, w), w, m)[0] == 0
    assert Path(str(bp) + ".scored").is_file()


# ----------------------------------------------------------------------------- F8: cheap harness items
def t95_f8_distinct_docs_mean_is_over_positive_queries_only():
    pos = [P("p1", 1, [ck("D1", 0.3), ck("X", 0.3)]), P("p2", 2, [ck("D2", 0.3)])]  # 2 and 1 distinct
    neg = [N("n1", "off", [ck(f"Z{i}", 0.5) for i in range(5)]), N("n2", "near", [ck("Y", 0.5)])]  # 5 and 1
    m = evaluate(pos, neg, unit_doc=SEC_DOC)
    assert m["distinct_docs_mean"] == 1.5, m["distinct_docs_mean"]
    assert evaluate([], neg, unit_doc=SEC_DOC)["distinct_docs_mean"] == 0.0


def t95_f8_bm25_is_fed_the_raw_situation_never_the_prefixed_query():
    w, m, doc_of = eval_env()
    tmpd = tempfile.mkdtemp()
    cp = Path(tmpd, "copy.md"); cp.write_text(copy_text())
    a = write_set(Path(tmpd, "a.json"), [dict(id="A10", section=10, query="whitespace-only strings pydantic")],
                  [dict(id="NA", kind="off", query="kubernetes readiness probe")])
    seen = []
    real = kb.bm25_top
    kb.bm25_top = lambda corpus, query, *a, **k: (seen.append(query), real(corpus, query, *a, **k))[1]
    try:
        code, out, err = run_eval(["--baseline", "bm25", "--set", "A", "--set-a-path", str(a), "--copy", str(cp)], FakeSearch({}, w), w, m)
    finally:
        kb.bm25_top = real
    assert code == 0, err
    assert seen == ["whitespace-only strings pydantic", "kubernetes readiness probe"], seen
    assert all("Instruct:" not in q and "Query:" not in q for q in seen)


def t95_f8_the_probe_requests_the_full_window_so_exclusions_cannot_empty_it():
    w, m, doc_of = eval_env()
    bp, script = b_set(tempfile.mkdtemp(), doc_of)
    script["sit one"] = [ck("CAPTURE", 0.10 + i * 0.001, i) for i in range(40)] + [ck(doc_of(1), 0.31)]
    srch = FakeSearch(script, w)
    code, out, err = run_eval(["--set", "B", "--probe", "--set-b-path", str(bp), "--exclude", "CAPTURE"], srch, w, m)
    assert code == 0, err
    assert next(ln for ln in out.split("\n") if ln.startswith("B01 ")).split()[1] == doc_of(1), out
    assert srch.calls and all(c[2] == 50 for c in srch.calls), {c[2] for c in srch.calls}


# ----------------------------------------------------------------------------- F4: the manifest's write-only fields get a reader
def t96_f4_diverge_reports_a_registry_whose_sha256_is_not_the_copys_unit_hash():
    w, m, copy = diverge_world()
    mf = kb.load_manifest(m)
    assert run_diverge(w, m, copy) == []
    mf["units"][1]["sha256"] = "0" * 64
    m.write_text(json.dumps(mf))
    assert run_diverge(w, m, copy) == [("REGISTRY-STALE", "u02", "Beta title")]
    del mf["units"][1]["sha256"]  # an absent hash is not a free pass
    m.write_text(json.dumps(mf))
    assert run_diverge(w, m, copy) == [("REGISTRY-STALE", "u02", "Beta title")]


def t96_f4_diverge_reports_a_registry_built_from_another_copy_pin_or_date():
    w, m, copy = diverge_world()
    for key, val in (("migratedOn", "2020-01-01"), ("pin", "deadbee")):
        mf = kb.load_manifest(m)
        mf[key] = val
        m.write_text(json.dumps(mf))
        found = run_diverge(w, m, copy)
        assert len(found) == 1 and found[0][0] == "REGISTRY-STALE" and found[0][1] == "-" and key in found[0][2], (key, found)
        run_ingest(w, copy, manifest=m)  # re-registering repairs it
        assert run_diverge(w, m, copy) == []


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
