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
    assert misses == [{"element": "a.py::f", "keyNorm": "GET /shop/api/health", "grain": "function", "class": "extractor"}]


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


# ================================================================ P1-36 store-side note §7 (F5): link status, linked-no-provider
# (``docs/plans/connected-code-graph-p1-noise-export-graph.md`` §2, §7; ``docs/reviews/connected-code-graph-p1-noise-export-ml.md`` F5)
LINK_ROW = {"consumerKey": "GET /{?}/shop/api/x", "element": "h.ts::h", "keyNorm": "GET /shop/api/x", "linkId": "L1",
            "path": "h.ts", "status": "confirmed"}                                         # the §2 fixture, verbatim
UNPROVIDED_ROW = {"consumerKey": "GET /{?}/shop/api/x", "element": "h.ts::h", "keyNorm": "GET /{?}/shop/api/x", "linkId": "L3",
                  "path": "h.ts", "status": "possible"}                                    # no TO: ON key is the consumer's


def hrow(method, path, file="h.ts", func=None):
    return row(method, path, file, func, "spa", host=None, port=None)


def test_hits_are_split_by_the_best_status_of_a_matching_link():
    rows = [hrow("GET", "/shop/api/a", "a.ts"), hrow("GET", "/shop/api/b", "b.ts"), hrow("GET", "/shop/api/c", "c.ts"),
            hrow("GET", "/shop/api/d", "d.ts")]
    links = [("a.ts::f", "GET /shop/api/a", "confirmed"), ("b.ts::f", "GET /shop/api/b", "possible"),
             ("c.ts::f", "GET /shop/api/c", "possible"), ("c.ts::g", "GET /shop/api/c", "confirmed"),     # best wins, either order
             ("zz.ts::f", "GET /shop/api/d", "confirmed")]                                                # other file: a miss
    rep, _ = O.score(rows, links)
    assert rep["pooledAcrossGrains"]["byStatus"]["edges"] == {"confirmed": 2, "possible": 1, "other": 0}
    assert rep["classes"]["file"]["byStatus"]["keys"] == {"confirmed": 2, "possible": 1, "other": 0}
    assert rep["classes"]["file"]["edges"]["k"] == 3 and rep["classes"]["function"]["byStatus"]["edges"] == {"confirmed": 0, "possible": 0, "other": 0}


def test_best_status_does_not_depend_on_link_order():
    r = [hrow("GET", "/shop/api/c", "c.ts")]
    for links in ([("c.ts::f", "GET /shop/api/c", "possible"), ("c.ts::g", "GET /shop/api/c", "confirmed")],
                  [("c.ts::g", "GET /shop/api/c", "confirmed"), ("c.ts::f", "GET /shop/api/c", "possible")]):
        assert O.score(r, links)[0]["pooledAcrossGrains"]["byStatus"]["edges"] == {"confirmed": 1, "possible": 0, "other": 0}


def test_a_link_without_a_status_is_counted_as_other_never_as_confirmed():
    rep, _ = O.score([hrow("GET", "/shop/api/c", "c.ts")], [("c.ts::f", "GET /shop/api/c")])
    assert rep["pooledAcrossGrains"]["byStatus"]["edges"] == {"confirmed": 0, "possible": 0, "other": 1}


def test_function_grain_hits_are_split_by_status_too():
    r = row("GET", "/shop/api/health", "c.py", "ping")
    rep, _ = O.score([r], [("c.py::ping", "GET /shop/api/health", "possible")])
    assert rep["classes"]["function"]["byStatus"]["edges"] == {"confirmed": 0, "possible": 1, "other": 0}


# ---- linked-no-provider
def _miss_class(runtime_rows, unprovided, links=()):
    rep, misses = O.score(runtime_rows, list(links), unprovided)
    return rep, [m["class"] for m in misses]


def test_a_miss_matching_an_unprovided_consumer_key_in_the_same_file_is_linked_no_provider():
    rep, cls = _miss_class([hrow("GET", "/shop/api/x")], [("h.ts::h", "GET /{?}/shop/api/x")])
    assert cls == ["linked-no-provider"]
    assert rep["misses"] == 1 and rep["missClasses"] == {"linked-no-provider": 1, "extractor": 0}
    assert rep["pooledAcrossGrains"]["edges"]["k"] == 0               # classification only, never scoring


def test_a_miss_with_no_unprovided_row_stays_extractor():
    rep, cls = _miss_class([hrow("GET", "/shop/api/x")], [])
    assert cls == ["extractor"] and rep["missClasses"] == {"linked-no-provider": 0, "extractor": 1}


@pytest.mark.parametrize("unprov,why", [
    ([("other.ts::h", "GET /{?}/shop/api/x")], "another file"),
    ([("h.ts::h", "POST /{?}/shop/api/x")], "another verb"),
    ([("h.ts::h", "GET /{?}/shop/api/y")], "literal differs"),
    ([("h.ts::h", "GET /{?}/v2/shop/api/x")], "key longer than the runtime path"),
    ([("h.ts::h", None)], "keyless row"),
    ([("h.ts::h", "GET /{?}")], "nothing after the wrapper"),
])
def test_a_miss_stays_extractor_when_no_unprovided_row_fits(unprov, why):
    assert _miss_class([hrow("GET", "/shop/api/x")], unprov)[1] == ["extractor"], why


@pytest.mark.parametrize("key", ["GET /{?}/{?}/shop/api/x", "GET /{?}/shop/api/{?}", "GET /{?}/shop/{}/x", "GET /{?}/shop/api/{#}",
                                 "* /{?}/shop/api/x", "GET /shop/api/x", "GET /{?}/api/x"])
def test_unprovided_key_forms_that_match_the_runtime_path(key):
    assert _miss_class([hrow("GET", "/shop/api/x")], [("h.ts::h", key)])[1] == ["linked-no-provider"]


def test_the_tail_is_compared_to_the_last_runtime_segments_per_segment():
    r = [hrow("GET", "/shop/api/x/42")]
    assert _miss_class(r, [("h.ts::h", "GET /{?}/x/{?}")])[1] == ["linked-no-provider"]       # last two segments x, 42
    assert _miss_class(r, [("h.ts::h", "GET /{?}/api/x")])[1] == ["extractor"]                # api/x are not the last two


def test_a_placeholder_tail_segment_needs_a_non_empty_runtime_segment():
    assert _miss_class([hrow("GET", "/shop/api/")], [("h.ts::h", "GET /{?}/shop/api/{?}")])[1] == ["extractor"]


def test_function_grain_miss_is_classified_by_file_not_function():
    r = row("GET", "/shop/api/x", "h.ts", "caller")
    assert _miss_class([r], [("h.ts::other", "GET /{?}/shop/api/x")])[1] == ["linked-no-provider"]


def test_an_edge_with_several_raw_rows_is_linked_no_provider_when_any_row_fits():
    u1, u2 = "11111111-1111-4111-8111-111111111111", "22222222-2222-4222-8222-222222222222"
    rows = [hrow("GET", f"/shop/api/t/{u1}"), hrow("GET", f"/shop/api/t/{u2}")]      # one edge: the capture masks UUIDs
    rep, misses = O.score(rows, [], [("h.ts::h", f"GET /{{?}}/shop/api/t/{u2}")])
    assert len(misses) == 1 and misses[0]["class"] == "linked-no-provider"           # only the second raw row fits


def test_a_hit_has_no_miss_class_and_the_provided_link_wins_over_an_unprovided_one():
    rep, cls = _miss_class([hrow("GET", "/shop/api/x")], [("h.ts::h", "GET /{?}/shop/api/x")], [("h.ts::h", "GET /shop/api/x", "confirmed")])
    assert cls == [] and rep["missClasses"] == {"linked-no-provider": 0, "extractor": 0} and rep["misses"] == 0


def test_the_report_names_no_unprovided_row():
    rep, _ = O.score([hrow("GET", "/shop/api/x")], [], [("secret/h.ts::hidden", "GET /{?}/shop/api/x")])
    assert "hidden" not in json.dumps(rep)


# ---- CLI with the §2 format
def test_cli_reads_status_and_the_unprovided_file_and_writes_the_class_to_the_misses_file(tmp_path, capsys):
    cap, links, unp, out = tmp_path / "all.jsonl", tmp_path / "links.jsonl", tmp_path / "links-unprovided.jsonl", tmp_path / "cust" / "m.jsonl"
    _write(cap, [hrow("GET", "/shop/api/x"), hrow("GET", "/shop/api/y", "h.ts"), hrow("GET", "/shop/api/z", "k.ts")])
    _write(links, [LINK_ROW])
    _write(unp, [dict(UNPROVIDED_ROW, keyNorm="GET /{?}/shop/api/y", consumerKey="GET /{?}/shop/api/y")])
    assert O.main(["--capture", str(cap), "--links", str(links), "--links-unprovided", str(unp), "--misses-out", str(out)]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["pooledAcrossGrains"]["byStatus"]["edges"] == {"confirmed": 1, "possible": 0, "other": 0}
    assert rep["missClasses"] == {"linked-no-provider": 1, "extractor": 1}
    got = {json.loads(l)["keyNorm"]: json.loads(l)["class"] for l in out.read_text().splitlines()}
    assert got == {"GET /shop/api/y": "linked-no-provider", "GET /shop/api/z": "extractor"}


def test_cli_without_an_unprovided_file_classifies_every_miss_extractor(tmp_path, capsys):
    cap, links, out = tmp_path / "all.jsonl", tmp_path / "links.jsonl", tmp_path / "m.jsonl"
    _write(cap, [hrow("GET", "/shop/api/y")])
    _write(links, [LINK_ROW])
    assert O.main(["--capture", str(cap), "--links", str(links), "--misses-out", str(tmp_path / "cust" / "m.jsonl")]) == 0
    assert json.loads(capsys.readouterr().out)["missClasses"] == {"linked-no-provider": 0, "extractor": 1}


def test_cli_exits_4_on_an_unprovided_file_row_without_an_element(tmp_path, capsys):
    cap, links, unp = tmp_path / "a.jsonl", tmp_path / "l.jsonl", tmp_path / "u.jsonl"
    _write(cap, []), _write(links, []), _write(unp, [{"keyNorm": "GET /x"}])
    assert O.main(["--capture", str(cap), "--links", str(links), "--links-unprovided", str(unp),
                   "--misses-out", str(tmp_path / "cust" / "m.jsonl")]) == 4
