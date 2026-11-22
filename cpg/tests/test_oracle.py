"""P1-36 runtime-oracle scorer (``docs/plans/connected-code-graph-p1-ml.md`` §6.5; P1-47 gate rulings).

Offline: the Link set is an injected list of ``(element, keyNorm)`` pairs; rows have the shape of run2's ``all.jsonl``."""
import json
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "eval" / "tools"
sys.path.insert(0, str(TOOLS))
import oracle as O  # noqa: E402

EP = "salesperson/src/api/endpoints.ts"


def row(method, path, file=None, func=None, src="py", cls="app", host="127.0.0.1", port=16500):
    return {"cls": cls, "frame": {"file": file, "func": func} if file else None, "host": host,
            "method": method, "path": path, "port": port, "src": src}


def spa(method, path):
    return row(method, path, EP, None, "spa", host=None, port=None)


# ---------------------------------------------------------------- segment matcher (ruling 2)
@pytest.mark.parametrize("static,method,raw,want", [
    ("GET /api/workspaces/{?}/threads", "GET", "/api/workspaces/123/threads", True),
    ("GET /threads/{}/messages", "GET", "/threads/t_abc123/messages", True),
    ("GET /t/{#}/info", "GET", "/t/77/info", True),
    ("GET /api/workspaces/{?}/threads", "GET", "/api/workspaces/123/other", False),   # literal mismatch stays unmatched
    ("GET /api/workspaces/{?}/threads", "GET", "/api/workspaces/123", False),          # segment count differs
    ("GET /api/workspaces/{?}", "GET", "/api/workspaces/123/threads", False),
    ("GET /api/workspaces/{?}", "GET", "/api/workspaces/", False),                      # placeholder needs a non-empty segment
    ("POST /x", "GET", "/x", False),                                                    # method differs
    ("* /x", "GET", "/x", True),                                                        # any-method verb
    ("GET /shop/api/state", "GET", "/shop/api/state?since=0", True),                    # query is not part of the key
    ("GET /a/b", "GET", "/a/B", False),                                                 # literals compare exactly
    ("GET /files/{**}", "GET", "/files/a/b/c", True),                                   # catch-all: one or more segments
    ("GET /files/{**}", "GET", "/files/a", True),
    ("GET /files/{**}", "GET", "/files", False),
    ("GET /files/{**}", "GET", "/files/", False),
    ("GET /files/{**}", "GET", "/other/a", False),
    ("GET /f/{**}/tail", "GET", "/f/a/b/tail", True),                                   # mid-path catch-all backtracks
    ("GET /f/{**}/tail", "GET", "/f/a/b/nope", False),
    ("GET /f/{**}/tail", "GET", "/f/tail", False),                                      # catch-all never matches zero segments
    ("GET /{?}/shop/api/messages", "GET", "/shop/api/messages", False),                # consumer-form wrapper key: not accepted
])
def test_key_matches_per_segment(static, method, raw, want):
    assert O.key_matches(static, method, raw) is want


# ---------------------------------------------------------------- in-scope rule (ruling 3)
@pytest.mark.parametrize("r,want", [
    (spa("GET", "/shop/api/state"), "in_scope"),
    (row("GET", "/shop/api/health", "scripts/x.py", "f"), "in_scope"),                  # any client of the storefront API
    (row("POST", "/threads/abc/messages", "falkor-chat/scripts/load_append.py", "_worker"), "out_of_scope"),
    (row("GET", "/health", "falkor-chat/scripts/load_test.sh", None, "sh"), "out_of_scope"),
    (row("POST", "/v1/chat/completions", "falkor-chat/server/falkorchat/transport.py", "_transport",
         host="localhost", port=1234), "out_of_scope"),
    (row("GET", "/shop/apix/state", "a.py", "f"), "out_of_scope"),                      # prefix is a path boundary
    (row("GET", "/shop/api/state", "falkor-chat/server/tests/test_x.py", "request", host="testserver", port=None,
         cls="testclient"), "testclient"),
    (row("GET", "/shop/api/state", "x.py", "f", host="testserver", port=None), "testclient"),   # decided by host alone
    (row("GET", "/shop/api/state", "falkor-chat/server/tests/test_x.py", "t"), "testclient"),   # or a frame under /tests/
])
def test_scope_rule(r, want):
    assert O.classify(r) == want


# ---------------------------------------------------------------- grain (ruling 1)
def test_spa_row_matches_at_file_grain_whatever_the_link_function():
    rows = [spa("GET", "/shop/api/state")]
    rep, _ = O.score(rows, [(EP + "::getState", "GET /shop/api/state")])
    assert rep["classes"]["file"]["edges"]["k"] == 1


def test_spa_row_does_not_match_a_link_in_another_file():
    rep, misses = O.score([spa("GET", "/shop/api/state")], [("salesperson/src/other.ts::getState", "GET /shop/api/state")])
    assert rep["classes"]["file"]["edges"]["k"] == 0 and len(misses) == 1


def test_function_grain_row_needs_the_same_function():
    r = row("GET", "/shop/api/health", "scripts/c.py", "ping")
    assert O.score([r], [("scripts/c.py::pong", "GET /shop/api/health")])[0]["classes"]["function"]["edges"]["k"] == 0
    assert O.score([r], [("scripts/c.py::ping", "GET /shop/api/health")])[0]["classes"]["function"]["edges"]["k"] == 1


def test_grain_classes_are_reported_separately_and_a_pooled_line_is_labelled():
    rows = [spa("GET", "/shop/api/state"), row("GET", "/shop/api/health", "scripts/c.py", "ping")]
    rep, _ = O.score(rows, [(EP + "::g", "GET /shop/api/state")])
    assert rep["classes"]["file"]["edges"]["n"] == 1 and rep["classes"]["function"]["edges"]["n"] == 1
    assert rep["classes"]["file"]["edges"]["k"] == 1 and rep["classes"]["function"]["edges"]["k"] == 0
    assert rep["pooledAcrossGrains"]["edges"]["k"] == 1 and rep["pooledAcrossGrains"]["edges"]["n"] == 2


# ---------------------------------------------------------------- numeric ids against placeholders (ruling 2, end to end)
def test_runtime_numeric_id_hits_a_placeholder_link():
    r = row("GET", "/api/workspaces/123/threads", "/shop/api/x.py", "f")
    r["path"] = "/shop/api/workspaces/123/threads"
    rep, _ = O.score([r], [("/shop/api/x.py::f", "GET /shop/api/workspaces/{?}/threads")])
    assert rep["classes"]["function"]["edges"]["k"] == 1


def test_literal_mismatch_is_a_miss():
    r = row("GET", "/shop/api/workspaces/123/threads", "x.py", "f")
    rep, _ = O.score([r], [("x.py::f", "GET /shop/api/workspaces/{?}/members")])
    assert rep["classes"]["function"]["edges"]["k"] == 0


# ---------------------------------------------------------------- counts: edges vs keys, rows, denominators
def test_rows_collapse_to_distinct_edges_and_keys():
    rows = [spa("GET", "/shop/api/state")] * 5 + [spa("GET", "/shop/api/messages?since=0"), spa("GET", "/shop/api/messages?since=4")]
    links = [(EP + "::a", "GET /shop/api/state")]
    rep, _ = O.score(rows, links)
    f = rep["classes"]["file"]
    assert (f["edges"]["k"], f["edges"]["n"], f["keys"]["k"], f["keys"]["n"]) == (1, 2, 1, 2)
    assert rep["inScopeRows"] == 7


def test_key_counts_differ_from_edge_counts_when_a_key_has_several_consumers():
    rows = [row("GET", "/shop/api/health", "a.py", "f"), row("GET", "/shop/api/health", "b.py", "g")]
    rep, _ = O.score(rows, [("a.py::f", "GET /shop/api/health")])
    f = rep["classes"]["function"]
    assert (f["edges"]["k"], f["edges"]["n"]) == (1, 2)
    assert (f["keys"]["k"], f["keys"]["n"]) == (1, 1)          # the key is linked by some consumer


def test_edge_is_hit_when_any_of_its_raw_rows_matches():
    rows = [row("GET", "/shop/api/t/123", "a.py", "f"), row("GET", "/shop/api/t/124", "a.py", "f")]
    rep, _ = O.score(rows, [("a.py::f", "GET /shop/api/t/123")])
    assert rep["classes"]["function"]["edges"]["k"] == 1


def test_out_of_scope_and_testclient_stay_outside_the_denominator():
    rows = [spa("GET", "/shop/api/state"),
            row("POST", "/v1/chat/completions", "t.py", "_t", host="localhost", port=1234),
            row("GET", "/x", "tests/t.py", "t", host="testserver", port=None, cls="testclient")]
    rep, _ = O.score(rows, [])
    assert (rep["rows"], rep["testclient"], rep["outOfScope"], rep["inScopeRows"]) == (3, 1, 1, 1)
    assert rep["pooledAcrossGrains"]["edges"]["n"] == 1 and rep["outOfScopeEdges"] == 1


def test_unattributable_in_scope_frame_is_counted_not_guessed():
    r = row("GET", "/shop/api/state")           # no frame
    rep, misses = O.score([r, spa("GET", "/shop/api/health")], [(EP + "::g", "GET /shop/api/health")])
    assert rep["unattributed"] == 1
    assert rep["pooledAcrossGrains"]["edges"]["n"] == 1      # not in the denominator, not a miss
    assert misses == []


def test_empty_denominator_has_no_interval():
    rep, _ = O.score([], [])
    assert rep["pooledAcrossGrains"]["edges"]["s"] is None


# ---------------------------------------------------------------- Wilson
def test_wilson_known_values():
    w = O.wilson(28, 28)
    assert w["lower"] == pytest.approx(0.879, abs=1e-3) and w["upper"] == 1.0
    w = O.wilson(4, 4)
    assert w["lower"] == pytest.approx(0.510, abs=1e-3)
    w = O.wilson(0, 10)
    assert w["lower"] == 0.0 and w["upper"] == pytest.approx(0.278, abs=1e-3)
    w = O.wilson(5, 10)
    assert (w["lower"], w["upper"]) == (pytest.approx(0.237, abs=1e-3), pytest.approx(0.763, abs=1e-3))


def test_report_carries_wilson_for_edges_and_keys():
    rep, _ = O.score([spa("GET", "/shop/api/state")], [(EP + "::g", "GET /shop/api/state")])
    e = rep["pooledAcrossGrains"]["edges"]
    assert e["s"] == 1.0 and e["lower"] == pytest.approx(0.207, abs=1e-3)


# ---------------------------------------------------------------- unrun entry points (ruling 4), report shape (ruling 5)
def test_report_lists_the_six_unrun_http_clients():
    rep, _ = O.score([], [])
    assert set(rep["unrunEntryPoints"]) == {
        "falkor-chat/web/app.js", "salesperson/scripts/load_demo.py", "salesperson/scripts/swap_bench.py",
        "salesperson/scripts/language_salience_probe.py", "salesperson/scripts/qwen_watch.sh", "salesperson/scripts/install_node.sh"}


def test_report_is_counts_only_and_names_no_miss():
    rows = [spa("GET", "/shop/api/state"), row("GET", "/shop/api/health", "secret/path.py", "hidden_fn")]
    rep, misses = O.score(rows, [])
    text = json.dumps(rep)
    assert "hidden_fn" not in text and "secret/path.py" not in text and "/shop/api/state" not in text
    assert rep["misses"] == 2 and len(misses) == 2


def test_misses_carry_element_and_key_for_the_custodian():
    _, misses = O.score([row("GET", "/shop/api/health", "a.py", "f")], [])
    assert misses == [{"element": "a.py::f", "keyNorm": "GET /shop/api/health", "grain": "function"}]


# ---------------------------------------------------------------- CLI: misses go to a separate custodian-side file, never stdout
def _write(p, rows):
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_cli_writes_misses_to_file_and_only_counts_to_stdout(tmp_path, capsys):
    cap, links, out = tmp_path / "all.jsonl", tmp_path / "links.jsonl", tmp_path / "cust" / "misses.jsonl"
    _write(cap, [spa("GET", "/shop/api/state"), row("GET", "/shop/api/health", "zz.py", "leakme")])
    _write(links, [{"element": EP + "::g", "keyNorm": "GET /shop/api/state"}])
    assert O.main(["--capture", str(cap), "--links", str(links), "--misses-out", str(out)]) == 0
    stdout = capsys.readouterr().out
    assert "leakme" not in stdout and json.loads(stdout)["misses"] == 1
    assert "leakme" in out.read_text()


def test_cli_refuses_a_misses_file_inside_the_repo(tmp_path, capsys):
    cap, links = tmp_path / "a.jsonl", tmp_path / "l.jsonl"
    _write(cap, []), _write(links, [])
    inside = Path(O.REPO) / "cpg" / "oracle-misses.jsonl"
    assert O.main(["--capture", str(cap), "--links", str(links), "--misses-out", str(inside)]) == 2
    assert not inside.exists()
