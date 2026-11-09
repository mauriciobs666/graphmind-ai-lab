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
