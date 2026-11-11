"""P1-15 — `extractors/indexers/scip.py`: a SCIP index to ``(file, line, col) -> target``.

Plan: `docs/plans/connected-code-graph-p1-extraction.md` § P1-15 (Files, Behaviours). The `.scip` file is hostile
input (the process that wrote it may have executed staged code): a size cap, bounded memory, and no path read from
it used for a write. The golden is a stored index built by hand (`tests/scip_helpers.py`, not by the indexer).
"""
from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

import scip_helpers as h
from extractors.indexers import scip
from extractors.indexers.scip import ScipError

DATA = Path(__file__).resolve().parent / "data" / "scip"
SRC = Path(scip.__file__)


def doc_with(path: str) -> bytes:
    return h.index([h.document(path, [h.occurrence([0, 0, 1], h.HELPER)])])


# --- golden ------------------------------------------------------------------------------------------------------


def test_the_stored_golden_is_what_the_hand_encoder_writes():
    assert (DATA / "tiny.scip").read_bytes() == h.tiny_index()


def test_golden_references_are_keyed_by_file_one_based_line_and_column():
    idx = scip.parse_scip(DATA / "tiny.scip")
    got = {k: tuple(t.symbol for t in v) for k, v in idx.refs.items()}
    assert got == {
        ("pkg/b.py", 4, 4): (h.HELPER,),
        ("pkg/b.py", 7, 8): (h.BOX_PUT,),
        ("pkg/b.py", 8, 4): (h.LEN,),
        ("pkg/b.py", 9, 0): (h.LOCAL,),
        ("pkg/b.py", 10, 4): (h.BOX,),
        ("pkg/b.py", 11, 4): (h.HELPER,),         # unpacked, four-element range
    }
    assert idx.tool == ("scip-python", "0.6.8")
    assert idx.documents == 2 and idx.rejected_documents == 0


def test_an_import_occurrence_is_not_a_reference():
    idx = scip.parse_scip(DATA / "tiny.scip")
    assert ("pkg/b.py", 1, 5) not in idx.refs            # role 2: no call site sits on an import


def test_definitions_are_not_references():
    idx = scip.parse_scip(DATA / "tiny.scip")
    assert not any(f == "pkg/a.py" for f, _l, _c in idx.refs)
    assert ("pkg/b.py", 3, 4) not in idx.refs            # `run`'s own definition


def test_target_carries_definition_file_chain_and_flags():
    idx = scip.parse_scip(DATA / "tiny.scip")
    put = idx.refs[("pkg/b.py", 7, 8)][0]
    assert (put.file, put.chain, put.callable, put.local) == ("pkg/a.py", ("Box", "put"), True, False)
    helper = idx.refs[("pkg/b.py", 4, 4)][0]
    assert (helper.file, helper.chain, helper.callable) == ("pkg/a.py", ("helper",), True)
    box = idx.refs[("pkg/b.py", 10, 4)][0]
    assert (box.file, box.chain, box.callable, box.is_type) == ("pkg/a.py", ("Box",), False, True)
    ext = idx.refs[("pkg/b.py", 8, 4)][0]
    assert ext.file is None and ext.chain == ("len",)      # defined in no indexed document: external
    loc = idx.refs[("pkg/b.py", 9, 0)][0]
    assert loc.local and loc.chain is None


def test_two_symbols_at_one_position_are_both_kept():
    raw = h.index([h.document("x.py", [h.occurrence([0, 0, 1], h.HELPER), h.occurrence([0, 0, 1], h.BOX_PUT)])])
    idx = scip.parse_bytes(raw)
    assert sorted(t.symbol for t in idx.refs[("x.py", 1, 0)]) == sorted([h.HELPER, h.BOX_PUT])


# --- the descriptor grammar --------------------------------------------------------------------------------------


@pytest.mark.parametrize("sym,chain,callable_,is_type", [
    ("scip-typescript npm salesperson 0.0.0 src/`api/endpoints.ts`/getHealth().", ("getHealth",), True, False),
    ("scip-python python p 0.1 `a.b`/C#m().", ("C", "m"), True, False),
    ("scip-python python p 0.1 `a.b`/C#", ("C",), False, True),
    ("scip-python python p 0.1 `a.b`/C#attr.", ("C", "attr"), False, False),
    ("scip-python python p 0.1 `a.b`/f().(x)", None, False, False),                   # a parameter: no chain
    ("scip-python python p 0.1 `a.b`/`weird``name`().", ("weird`name",), True, False),   # escaped backtick
    ("scip-python python pkg  with  space 0.1 `a`/f().", ("f",), True, False),         # escaped spaces in the package
    ("scip-python python p 0.1 garbage(", None, False, False),                             # unparsable: external-like
])
def test_symbol_descriptors(sym, chain, callable_, is_type):
    p = scip.parse_symbol(sym)
    assert (p.chain, p.callable, p.is_type) == (chain, callable_, is_type)


# --- hostile input -----------------------------------------------------------------------------------------------


def test_size_cap_refuses_before_reading(tmp_path, monkeypatch):
    big = tmp_path / "big.scip"
    with open(big, "wb") as f:
        f.truncate(10 * 1024 * 1024)                       # sparse: reading it would be 10 MiB
    for name in ("read", "readv"):
        monkeypatch.setattr(os, name, lambda *a: pytest.fail("the file was read although its size is over the cap"))
    with pytest.raises(ScipError, match="exceeds the size cap"):
        scip.parse_scip(big, max_bytes=1024 * 1024)


def test_size_cap_boundary(tmp_path):
    p = tmp_path / "t.scip"
    raw = h.tiny_index()
    p.write_bytes(raw)
    assert scip.parse_scip(p, max_bytes=len(raw)).documents == 2
    with pytest.raises(ScipError, match="exceeds the size cap"):
        scip.parse_scip(p, max_bytes=len(raw) - 1)


def test_a_file_that_grows_after_stat_is_refused(tmp_path, monkeypatch):
    p = tmp_path / "g.scip"
    p.write_bytes(h.tiny_index())
    real = os.fstat
    monkeypatch.setattr(os, "fstat", lambda fd: type("S", (), {"st_size": 10, "st_mode": real(fd).st_mode})())
    with pytest.raises(ScipError, match="changed while it was read"):
        scip.parse_scip(p, max_bytes=100)


def test_a_symlink_is_not_followed(tmp_path):
    link = tmp_path / "link.scip"
    os.symlink(DATA / "tiny.scip", link)
    with pytest.raises(ScipError, match="cannot open"):
        scip.parse_scip(link)


def test_not_a_regular_file_is_refused(tmp_path):
    fifo = tmp_path / "f.scip"
    os.mkfifo(fifo)
    with pytest.raises(ScipError, match="regular file"):
        scip.parse_scip(fifo)


@pytest.mark.parametrize("path", ["../x.py", "/etc/passwd", "a/../../b.py", "a\x00b.py", "a\nb.py", "a\x1fb.py", "a\x7fb.py", "", "a//b.py", "./a.py", "C:\\x.py",
                                  "a/./b.py", "x" * 5000])
def test_a_document_path_that_could_escape_is_dropped_and_counted(path):
    idx = scip.parse_bytes(doc_with(path))
    assert idx.refs == {} and idx.rejected_documents == 1


def test_a_normal_nested_path_is_kept():
    idx = scip.parse_bytes(doc_with("src/api/a-b_c.d/e.ts"))
    assert ("src/api/a-b_c.d/e.ts", 1, 0) in idx.refs


def test_parsing_writes_nothing_and_never_touches_a_path_from_the_file(tmp_path, monkeypatch):
    victim = tmp_path / "victim"
    victim.mkdir()
    (tmp_path / "evil").mkdir()
    raw = h.index([h.document(f"../{victim.name}/pwn.py", [h.occurrence([0, 0, 1], h.HELPER)]),
                   h.document(str(victim / "abs.py"), [h.occurrence([0, 0, 1], h.HELPER)])])
    monkeypatch.chdir(tmp_path / "evil")
    scip.parse_bytes(raw)
    assert os.listdir(victim) == [] and os.listdir(tmp_path / "evil") == []


def test_the_module_has_no_write_capable_call():
    """A structural guard: nothing in scip.py can write or open a path (the one read is `os.open` read-only)."""
    tree = ast.parse(SRC.read_text())
    banned = {"write", "write_text", "write_bytes", "mkdir", "makedirs", "unlink", "remove", "rename", "replace",
              "symlink", "copy", "copyfile", "copytree", "rmtree", "touch", "chmod", "truncate", "mkfifo"}
    hits = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            f = n.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
            if name in banned:
                hits.append((name, n.lineno))
            if name in {"open", "fdopen"}:
                mode = [a for a in n.args[1:2]] + [k.value for k in n.keywords if k.arg in ("mode", "flags")]
                src = " ".join(ast.unparse(m) for m in mode)
                if any(w in src for w in ("O_WRONLY", "O_RDWR", "O_CREAT", "O_APPEND", "O_TRUNC")) or re.search(r"['\"][rb]*[wax+]", src):
                    hits.append((name, n.lineno))
    assert hits == []


@pytest.mark.parametrize("cut", [1, 5, 40, 300, 700])
def test_a_truncated_index_is_an_error_not_a_partial_result(cut):
    raw = h.tiny_index()
    with pytest.raises(ScipError):
        scip.parse_bytes(raw[:cut])


def test_malformed_wire_is_an_error():
    with pytest.raises(ScipError, match="varint"):
        scip.parse_bytes(b"\x12" + b"\xff" * 11)                      # an 11-byte varint
    with pytest.raises(ScipError, match="length"):
        scip.parse_bytes(b"\x12\xff\xff\xff\x7f")                     # a length past the end
    with pytest.raises(ScipError, match="wire type"):
        scip.parse_bytes(b"\x13")                                     # a group: not in scip.proto


def test_an_occurrence_with_a_bad_range_is_skipped_and_counted():
    raw = h.index([h.document("a.py", [h.occurrence([1], h.HELPER), h.occurrence([-1 & 0xFFFFFFFF, 0, 1], h.HELPER),
                                       h.occurrence([0, 0, 1], h.HELPER)])])
    idx = scip.parse_bytes(raw)
    assert list(idx.refs) == [("a.py", 1, 0)] and idx.rejected_occurrences == 2


def test_an_overlong_symbol_is_skipped_and_counted():
    raw = h.index([h.document("a.py", [h.occurrence([0, 0, 1], "x" * 5000), h.occurrence([1, 0, 1], h.HELPER)])])
    idx = scip.parse_bytes(raw)
    assert list(idx.refs) == [("a.py", 2, 0)] and idx.rejected_occurrences == 1


def test_occurrence_count_cap():
    occs = [h.occurrence([i, 0, 1], h.HELPER) for i in range(50)]
    raw = h.index([h.document("a.py", occs)])
    assert len(scip.parse_bytes(raw, max_occurrences=50).refs) == 50
    with pytest.raises(ScipError, match="occurrence cap"):
        scip.parse_bytes(raw, max_occurrences=49)


def test_document_count_cap():
    raw = h.index([h.document(f"f{i}.py") for i in range(5)])
    assert scip.parse_bytes(raw, max_documents=5).documents == 5
    with pytest.raises(ScipError, match="document cap"):
        scip.parse_bytes(raw, max_documents=4)


def test_memory_is_bounded_by_the_input_not_by_what_it_claims(tmp_path):
    """A 1 MB input that claims a 2 GB field is refused without allocating it."""
    import tracemalloc
    tracemalloc.start()
    with pytest.raises(ScipError):
        scip.parse_bytes(b"\x12" + h.varint(2 * 1024 ** 3) + b"x" * 1000)
    _cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 5 * 1024 * 1024


def test_unknown_fields_are_skipped():
    extra = h.vi(15, 7) + h.ld(14, b"zz")
    raw = h.index([h.document("a.py", [h.occurrence([0, 0, 1], h.HELPER)])]) + extra
    assert ("a.py", 1, 0) in scip.parse_bytes(raw).refs


# --- expansion probes: the work and memory one unit of input costs ---------------------------------------------------


def one_document(body: bytes, path: str = "a.py") -> bytes:
    return h.ld(2, h.ld(1, path.encode()) + body)


def test_many_distinct_symbols_at_one_position_cost_linear_time_and_are_capped():
    n = 50_000
    body = b"".join(h.ld(2, h.occurrence([0, 0, 1], f"scip-python python p 0.1 `m`/f{i}().")) for i in range(n))
    t0 = time.monotonic()
    idx = scip.parse_bytes(one_document(body))
    assert time.monotonic() - t0 < 2.0
    assert len(idx.refs[("a.py", 1, 0)]) == scip.MAX_TARGETS_PER_POSITION
    assert idx.rejected_occurrences == n - scip.MAX_TARGETS_PER_POSITION


def test_the_same_symbol_repeated_at_one_position_is_one_target():
    body = h.ld(2, h.occurrence([0, 0, 1], h.HELPER)) * 1000
    idx = scip.parse_bytes(one_document(body))
    assert len(idx.refs[("a.py", 1, 0)]) == 1 and idx.rejected_occurrences == 0


def test_a_range_with_many_elements_is_rejected_without_collecting_them():
    body = h.ld(2, b"".join(h.vi(1, 0) for _ in range(5000)) + h.ld(2, h.HELPER.encode()))
    idx = scip.parse_bytes(one_document(body))
    assert idx.refs == {} and idx.rejected_occurrences == 1


def test_symbol_information_cap():
    raw = one_document(b"\x1a\x00" * 10)
    assert scip.parse_bytes(raw, max_symbol_infos=10).documents == 1
    with pytest.raises(ScipError, match="symbol cap"):
        scip.parse_bytes(raw, max_symbol_infos=9)


def test_the_caps_are_counted_before_anything_is_stored_even_in_a_document_that_is_dropped():
    raw = one_document(b"\x12\x00" * 10, path="../x")
    with pytest.raises(ScipError, match="occurrence cap"):
        scip.parse_bytes(raw, max_occurrences=9)


LIMITED = """
import resource, sys
resource.setrlimit(resource.RLIMIT_AS, (1_500_000_000, 1_500_000_000))
sys.path.insert(0, {cpg!r})
sys.path.insert(0, {tests!r})
from extractors.indexers import scip
import scip_helpers as h
{setup}
open({out!r}, "wb").write(raw)
del raw
try:
    scip.parse_scip({out!r})
except scip.ScipError as e:
    print("ScipError", e)
except MemoryError:
    print("MemoryError")
print("peak_mb", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)
"""


def run_limited(tmp_path, setup: str) -> list[str]:
    cpg = Path(__file__).resolve().parent.parent
    code = LIMITED.format(cpg=str(cpg), tests=str(cpg / "tests"), setup=setup, out=str(tmp_path / "x.scip"))
    p = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stderr
    return p.stdout.splitlines()


@pytest.mark.parametrize("setup,what", [
    ('raw = h.ld(2, h.ld(1, b"a.py") + b"\\x12\\x00" * 12_000_000)', "occurrence cap"),       # a 24 MB file of empty occurrences
    ('raw = h.ld(2, h.ld(1, b"a.py") + b"\\x1a\\x00" * 20_000_000)', "symbol cap"),           # a 40 MB file of empty symbol infos
    ('raw = h.ld(2, h.ld(1, b"../x") + b"\\x12\\x00" * 12_000_000)', "occurrence cap"),       # ... in a document that is dropped
])
def test_a_file_of_empty_sub_messages_is_a_scip_error_not_a_memory_error_under_a_1_5_gb_limit(tmp_path, setup, what):
    out = run_limited(tmp_path, setup)
    assert out[0].startswith("ScipError") and what in out[0]
    assert int(out[1].split()[1]) < 300


def test_the_cap_respecting_worst_case_peaks_under_half_the_limit(tmp_path):
    """``DEFAULT_MAX_OCCURRENCES`` (1 000 000) occurrences, every one a distinct symbol at a distinct position: the file (48 MB) and the
    parse together stay under 750 MB, half of a 1.5 GB address-space limit."""
    setup = ('raw = h.ld(2, h.ld(1, b"a.py") + b"".join(h.ld(2, h.occurrence([i // 200, i % 200, i % 200 + 1], '
             'f"scip-python python p 0.1 `m`/f{i}().")) for i in range(scip.DEFAULT_MAX_OCCURRENCES)))')
    out = run_limited(tmp_path, setup)
    assert out[0].startswith("peak_mb"), out
    assert int(out[0].split()[1]) < 750


def test_an_occurrence_with_millions_of_range_elements_is_rejected_without_collecting_them(tmp_path):
    out = run_limited(tmp_path, 'raw = h.ld(2, h.ld(1, b"a.py") + h.ld(2, b"\\x0a\\x00" * 10_000_000))')    # 20 MB of empty packed ranges
    assert out[0].startswith("peak_mb") and int(out[0].split()[1]) < 300


def test_the_shipped_size_cap_refuses_a_file_just_over_it(tmp_path):
    assert scip.DEFAULT_MAX_BYTES == 64 * 1024 * 1024
    big = tmp_path / "big.scip"
    with open(big, "wb") as f:
        f.truncate(scip.DEFAULT_MAX_BYTES + 1)                                     # sparse; refused before any read
    with pytest.raises(ScipError, match="exceeds the size cap"):
        scip.parse_scip(big)
