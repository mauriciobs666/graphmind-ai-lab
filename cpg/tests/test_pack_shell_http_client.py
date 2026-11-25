"""CO-3 (DEF-1): the `curl`/`wget` URL consumer pack (`cpg/packs/shell-http-client/`), run by the P1-16 pack runtime.

Plan: `docs/plans/connected-code-graph-p1-closeout.md` section 3.3. A `curl`/`wget` command in a shell script, or an inline
`curl ...` span of a prompt Markdown file (call kind `mention`), is an `http` consumer of the URL it names. The URL is any
argument position (`curl -sf -m 5 URL`), so the pack lifts every argument position and keeps the URL-like alternatives
(`urlLike`); a flag, a number or a word is not one.

Three kinds of evidence, as in `test_pack_shell_mcp_call.py`:

* hand-built records (`packrt_helpers`) pin each clause one at a time;
* the REAL shell and Markdown extractors turn text into records, so a comment, a quoted `echo` argument, a heredoc line or a
  fenced block is what the extractor says it is (no regular expression on text here);
* `behaviour_failures(pack)` is the whole contract as one function; the leaf-mutation registration mutates every leaf of the
  pack file and requires it to fail for each.

Known limits (pinned, not endorsed): see the `test_known_limit_*` tests. The command lifting is `urlLike: scheme`, so an operand that is
a path (`-o /tmp/out`, `-d /health`) or a non-HTTP URL (`ftp://`, `file://`) is no consumer.
"""
from __future__ import annotations

import copy
import pathlib
import tempfile

import pytest
import yaml

from packrt_helpers import CPG, PACKS, call, file_, pack, sym, vs
from ccg.packrt import run_packs
from co3_helpers import assemble_http
from ccg.rulepack import load_pack, pack_errors
from extractors.markdown.extract import extract_markdown
from extractors.markdown.globs import load_globs
from extractors.shell import extract_shell

PID = "shell-http-client"
NEGATIVES = PACKS / PID / "negatives.yaml"
HISTORY = lambda path, spans: ("2024-01-01T00:00:00Z", "a" * 40)   # noqa: E731
GLOBS = load_globs(CPG / "eval" / "eval-manifest.yaml")
PROMPT = "claude/demo/demo.md"          # a `prompt-md` path of the eval manifest
URL = "http://127.0.0.1:8200/health"
DYN_PORT = {"template": [{"const": "http://127.0.0.1:"}, {"unknown": "runtime-built"}, {"const": "/health"}]}
DYN_ORIGIN = {"template": [{"unknown": "runtime-built"}, {"const": "/health"}]}


def shell_records(script: str, *, name: str = "a.sh") -> list[dict]:
    with tempfile.TemporaryDirectory(dir=CPG / ".ccg-work" if (CPG / ".ccg-work").is_dir() else None) as d:
        (pathlib.Path(d) / name).write_text(script)
        return extract_shell(pathlib.Path(d), [name], repo_id="r", unit_id="u", history=HISTORY, warnings=[])


def md_records(text: str, *, path: str = PROMPT) -> list[dict]:
    return extract_markdown(path, text.encode(), repo_id="r", unit_id="u", globs=GLOBS, history=HISTORY("", ())).records


def consumers(records: list[dict], the_pack: dict | None = None):
    return run_packs([records], [the_pack or pack(PID)]).consumers


def view(c: dict) -> tuple:
    return (c["kind"], c["key"], c["status"], c["reason"])


def planted(callee: str, *args, kind: str = "command", line: int = 4) -> list[dict]:
    """One script file, its `<script>` symbol and one call site of `callee`; `args` are (position, value) pairs (a const or an element)."""
    recs = [file_("f1", "s.sh", lang="shell", role="primary"),
            sym("sc", "<script>", fid="f1", path="s.sh", kind="script"),
            call("c1", "sc", callee, line=line, kind=kind)]
    recs += [vs("c1", f"a{i}", v) for i, v in args]
    return recs


# ---------------------------------------------------------------------------------------------
# the pack file
# ---------------------------------------------------------------------------------------------


def test_the_pack_loads_and_passes_the_validator_and_is_a_paper_status_v1_pack():
    p = load_pack(PACKS / PID / "pack.yaml")
    assert pack_errors(p, dirname=PID) == []
    assert (p["id"], p["status"], p["packFormat"]) == (PID, "paper", 1)


# ---------------------------------------------------------------------------------------------
# the command rule: the URL is any argument position
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("callee", ["curl", "wget"])
def test_a_literal_url_is_a_confirmed_http_consumer_keyed_by_the_path_with_the_origin_dropped(callee):
    got = consumers(planted(callee, (0, URL)))
    assert [view(c) for c in got] == [("http", "GET /health", "confirmed", "lifted-role")]
    assert (got[0]["packId"], got[0]["line"], got[0]["qname"]) == (PID, 4, "s.sh::<script>")


@pytest.mark.parametrize("pos", [1, 3, 7])
def test_the_url_is_found_after_any_number_of_flags(pos):
    flags = [(i, f) for i, f in enumerate(["-sf", "-m", "5", "-o", "x", "-w", "%{http_code}", "--max-time", "2"][:pos])]
    assert [c["key"] for c in consumers(planted("curl", *flags, (pos, URL)))] == ["GET /health"]


def test_a_url_with_a_dynamic_port_is_possible_partial_key_with_the_path_key():
    got = consumers(planted("curl", (0, "-fsS"), (1, DYN_PORT)))
    assert [view(c) for c in got] == [("http", "GET /health", "possible", "partial-key")]


def test_a_url_with_a_dynamic_origin_is_possible_partial_key_with_the_path_key():
    got = consumers(planted("curl", (0, "-sf"), (1, "-m"), (2, "5"), (3, DYN_ORIGIN)))
    assert [view(c) for c in got] == [("http", "GET /health", "possible", "partial-key")]


def test_an_argument_that_is_wholly_unknown_names_no_url():
    assert consumers(planted("curl", (0, {"unknown": "runtime-built"}))) == []


def test_the_query_string_is_not_part_of_the_key():
    assert [c["key"] for c in consumers(planted("curl", (0, "http://127.0.0.1:8200/health?x=1")))] == ["GET /health"]


@pytest.mark.parametrize("callee", ["curls", "xcurl", "CURL", "./curl", "curl ", "wget2", "mycurl", "libcurl"])
def test_the_command_name_is_matched_exactly(callee):
    assert consumers(planted(callee, (0, URL))) == []


def test_only_a_command_or_mention_call_kind_matches_a_call_in_another_language_does_not():
    assert consumers(planted("curl", (0, URL), kind="call")) == []


# ---------------------------------------------------------------------------------------------
# the Markdown rule (call kind `mention`)
# ---------------------------------------------------------------------------------------------


def test_a_markdown_curl_mention_with_a_url_is_a_possible_prompt_mention_consumer():
    got = consumers(planted("curl", (0, URL), kind="mention"))
    assert [view(c) for c in got] == [("http", "GET /health", "possible", "prompt-mention")]
    assert got[0]["ruleId"] != consumers(planted("curl", (0, URL)))[0]["ruleId"]


def test_a_markdown_curl_mention_with_no_url_is_no_consumer():
    assert consumers(planted("curl", kind="mention")) == []


def test_the_markdown_rule_reads_only_the_first_argument_the_extractor_wrote():
    assert consumers(planted("curl", (1, URL), kind="mention")) == []


def test_the_real_markdown_extractor_inline_span_gives_the_consumer():
    got = consumers(md_records("Poll `curl -sf http://127.0.0.1:8200/health` until 200.\n"))
    assert [view(c) for c in got] == [("http", "GET /health", "possible", "prompt-mention")]
    assert got[0]["qname"] == f"{PROMPT}::<document>"


# ---------------------------------------------------------------------------------------------
# the real shell extractor feeding the pack
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("script,want", [
    ('curl -fsS "$BASE_URL/health" >/dev/null\n', [("GET /health", "possible", "partial-key")]),
    ('curl -sf -m 5 "${BASE}/health" || exit 1\n', [("GET /health", "possible", "partial-key")]),
    ('code="$(curl -s -o /dev/null -w \'%{http_code}\' --max-time 2 "http://127.0.0.1:$PORT/health" 2>/dev/null)"\n',
     [("GET /health", "possible", "partial-key")]),
    ('wget -qO- http://localhost:8200/health\n', [("GET /health", "confirmed", "lifted-role")]),
    ('if curl -sf http://127.0.0.1:8200/health; then echo up; fi\n', [("GET /health", "confirmed", "lifted-role")]),
])
def test_the_dev_shapes_through_the_real_shell_extractor(script, want):
    assert sorted((c["key"], c["status"], c["reason"]) for c in consumers(shell_records(script))) == sorted(want)


def test_a_call_inside_a_function_is_anchored_at_the_function():
    got = consumers(shell_records("probe() {\n  curl -sf http://127.0.0.1:8200/health\n}\nprobe\n"))
    assert [(c["key"], c["qname"]) for c in got] == [("GET /health", "a.sh::probe")]


def test_a_function_no_one_calls_is_still_a_consumer_at_its_own_site():
    """`ownSite`: a wrapper no call edge reaches is a consumer where the call is, with its dynamic path piece unknown."""
    got = consumers(shell_records('probe() {\n  curl -sf "http://127.0.0.1:8200/health/$1"\n}\n'))
    assert [(c["key"], c["status"], c["reason"], c["qname"]) for c in got] == [("GET /health/{?}", "possible", "partial-key", "a.sh::probe")]


def test_a_function_that_takes_the_url_is_a_consumer_at_each_call_not_inside_it():
    got = consumers(shell_records('probe() {\n  curl -sf "$1"\n}\nprobe http://127.0.0.1:8200/health\nprobe "$BASE/x"\n'))
    assert sorted((c["key"], c["status"], c["line"], c["qname"]) for c in got) == [
        ("GET /health", "confirmed", 4, "a.sh::<script>"), ("GET /x", "possible", 5, "a.sh::<script>")]


def test_a_script_variable_assigned_once_is_resolved_by_the_extractor():
    got = consumers(shell_records('U=http://127.0.0.1:8200/health\ncurl -sf "$U"\n'))
    assert [view(c)[:3] for c in got] == [("http", "GET /health", "confirmed")]


# ---------------------------------------------------------------------------------------------
# known limits, pinned (a limit is not a negative: each goes red, deliberately, when the format learns to see it)
# ---------------------------------------------------------------------------------------------


def test_known_limit_the_verb_is_always_get_whatever_the_flags():
    """`curl -X POST URL` is keyed `GET`: the pack format has no flag grammar (as `mcp-call --json reindex`, `shell-mcp-call`)."""
    assert [c["key"] for c in consumers(shell_records("curl -X POST -d x http://127.0.0.1:8200/documents\n"))] == ["GET /documents"]


@pytest.mark.parametrize("script,want", [
    ("curl -o /tmp/health http://127.0.0.1:8200/x\n", ["GET /x"]),
    ("curl -sf -o /dev/null -w '%{http_code}' http://127.0.0.1:8200/x\n", ["GET /x"]),
    ("curl -d /health http://127.0.0.1:8200/other\n", ["GET /other"]),
    ("curl -H /health http://127.0.0.1:8200/other\n", ["GET /other"]),
    ("curl --data-raw /health http://127.0.0.1:8200/other\n", ["GET /other"]),
    ("curl -K /etc/curlrc http://127.0.0.1:8200/other\n", ["GET /other"]),
    ("curl ftp://127.0.0.1/pub/file\n", []),
    ("curl file:///etc/passwd\n", []),
    ("curl /health\n", []),
    ("wget -O /work/plugins/x.jar http://127.0.0.1:8200/x\n", ["GET /x"]),
], ids=["o-tmp", "o-devnull", "d-path", "H-path", "data-raw", "K-path", "ftp", "file", "bare-path", "wget-O"])
def test_a_non_url_operand_with_a_leading_slash_or_another_scheme_is_no_consumer(script, want):
    """`urlLike: scheme`: curl needs a host, so a `/`-leading operand is an output path or data, never the URL (M1 of the CO-3 gate)."""
    assert sorted(c["key"] for c in consumers(shell_records(script))) == want


@pytest.mark.parametrize("script,keys", [
    ("curl -sf http://127.0.0.1:8200/health\n", ["GET /health"]),
    ("curl -sf https://h.example/health\n", ["GET /health"]),
    ('curl -sf "$BASE/health"\n', ["GET /health"]),
    ('curl -sf "http://127.0.0.1:$PORT/health"\n', ["GET /health"]),
    ("curl -sf http://u:pw@127.0.0.1:8200/health\n", ["GET /health"]),
], ids=["http", "https", "dynamic-origin", "dynamic-port", "plain-userinfo"])
def test_the_live_twins_of_the_non_url_operands_stay_consumers(script, keys):
    assert [c["key"] for c in consumers(shell_records(script))] == keys


@pytest.mark.parametrize("script,key", [
    ("curl -o /tmp/health http://127.0.0.1:8200/x\n", "GET /tmp/health"),
    ("curl -d /health http://127.0.0.1:8200/other\n", "GET /health"),
    ("curl ftp://127.0.0.1/pub/file\n", "GET /pub/file"),
])
def test_a_non_url_operand_does_not_link_to_a_provider_of_the_same_key(tmp_path, script, key):
    tree = {"svc/api.sh": "#!/bin/bash\necho serve\n", "ops/x.sh": "#!/bin/bash\n" + script}
    assert assemble_http(tmp_path, tree, "svc/api.sh", key).links.links == []


def test_the_twin_of_a_non_url_operand_links_to_a_provider_of_the_same_key(tmp_path):
    tree = {"svc/api.sh": "#!/bin/bash\necho serve\n", "ops/x.sh": "#!/bin/bash\ncurl -o /tmp/out http://127.0.0.1:8200/health\n"}
    assert len(assemble_http(tmp_path, tree, "svc/api.sh", "GET /health").links.links) == 1


@pytest.mark.parametrize("script", ['curl -o "$D/health" http://127.0.0.1:8200/x\n', 'curl -d "$X/health" http://127.0.0.1:8200/x\n'])
def test_known_limit_a_computed_operand_before_a_slash_is_a_possible_partial_key_consumer(script):
    """`urlLike: scheme` keeps a dynamic origin then `/`: `-o "$D/health"` is read as a URL with a computed origin. Counted wrong-`possible` at CO-9."""
    got = sorted((c["key"], c["status"], c["reason"]) for c in consumers(shell_records(script)))
    assert got == [("GET /health", "possible", "partial-key"), ("GET /x", "confirmed", "lifted-role")]


def test_known_limit_a_shell_url_with_plain_userinfo_keeps_its_path_key():
    """The shell extractor keeps a short `user:pw@` constant (it passes the token screen; the Markdown carrier refuses any userinfo).
    The key never holds the userinfo (the authority is dropped); a token-shaped one is `unknown` and gives no consumer (negatives)."""
    assert [c["key"] for c in consumers(shell_records("curl -sf http://u:pw@127.0.0.1:8200/health\n"))] == ["GET /health"]


# ---------------------------------------------------------------------------------------------
# negatives.yaml: per rule id, through the real extractors (a decoy per axis)
# ---------------------------------------------------------------------------------------------


def negatives() -> dict[str, dict[str, str]]:
    return yaml.safe_load(NEGATIVES.read_text())["negatives"]


def records_for(rule: str, text: str) -> list[dict]:
    return shell_records(text) if rule == "command" else md_records(text)


def negatives_failures(the_pack: dict) -> list[str]:
    """`rule/name` of every negative text that the pack turns into a consumer."""
    out = []
    for rule, cases in negatives().items():
        for name, text in cases.items():
            if consumers(records_for(rule, text), the_pack):
                out.append(f"{rule}/{name}")
    return out


def test_every_rule_of_the_pack_has_negatives_and_every_negative_names_a_rule():
    rules = {r["id"] for r in pack(PID)["rules"]}
    assert set(negatives()) == rules
    for rule, cases in negatives().items():
        assert len(cases) >= 5 and all(isinstance(s, str) and s.strip() for s in cases.values()), rule


def test_the_negatives_cover_every_axis():
    assert {"unrelated-url", "comment", "quoted-example", "heredoc", "token-shaped-userinfo"} <= set(negatives()["command"])
    assert {"unrelated-url", "fenced-block", "word-in-prose", "non-http-scheme", "credentials", "token-shaped-userinfo"} <= set(negatives()["mention"])


def test_the_shipped_pack_turns_no_negative_into_a_consumer():
    assert negatives_failures(pack(PID)) == []


#: Per negative: the same text with only the protecting feature removed, which the pack must read (a no-match test needs a decoy).
TWINS = {
    ("command", "unrelated-url"): "curl -sf -m 5 http://example.com/health\n",
    ("command", "scheme-less-host"): "curl -sf http://localhost:8200/health\n",
    ("command", "comment"): "curl -sf http://127.0.0.1:8200/health\n",
    ("command", "quoted-example"): "curl -sf http://127.0.0.1:8200/health\n",
    ("command", "heredoc"): "curl -sf http://127.0.0.1:8200/health\n",
    ("command", "token-shaped-userinfo"): "curl -sf http://svc:pw@127.0.0.1:8200/health\n",
    ("command", "curl-as-argument-of-another-command"): "curl http://127.0.0.1:8200/health\n",
    ("command", "similar-name"): "curl -sf http://127.0.0.1:8200/health\n",
    ("mention", "unrelated-url"): "Probe `curl -sf http://example.com/health` first.\n",
    ("mention", "fenced-block"): "Start it, then `curl -sf http://127.0.0.1:8200/health`.\n",
    ("mention", "word-in-prose"): "Use `curl http://127.0.0.1:8200/health` to poll the server.\n",
    ("mention", "non-http-scheme"): "Fetch it with `curl http://127.0.0.1/health`.\n",
    ("mention", "credentials"): "Probe `curl -sf http://127.0.0.1:8200/health` first.\n",
    ("mention", "token-shaped-userinfo"): "Probe `curl -sf http://127.0.0.1:8200/health` first.\n",
    ("mention", "token-shaped-path"): "Probe `curl -sf http://127.0.0.1:8200/health` first.\n",
    ("mention", "url-outside-the-span"): "The server answers; run `curl -sf http://127.0.0.1:8200/health` against it.\n",
    ("mention", "similar-name"): "Probe `curl -sf http://127.0.0.1:8200/health`.\n",
}


def test_every_negative_has_a_live_twin_that_the_pack_does_read():
    assert set(TWINS) == {(r, n) for r, cases in negatives().items() for n in cases}
    for (rule, name), text in TWINS.items():
        assert consumers(records_for(rule, text)), (rule, name)


def test_every_negative_text_holds_the_decoy_word():
    for rule, cases in negatives().items():
        for name, text in cases.items():
            assert "curl" in text or "wget" in text, f"{rule}/{name}"


# ---------------------------------------------------------------------------------------------
# the contract as one function (used by the leaf-mutation registration)
# ---------------------------------------------------------------------------------------------


def behaviour_failures(the_pack: dict) -> list[str]:
    """What `the_pack` gets wrong, as short strings (empty for the shipped pack)."""
    bad: list[str] = []
    errs = pack_errors(the_pack, dirname=PID)
    if errs:
        return [f"invalid: {errs[0]}"]

    def got(recs):
        return [view(c) + (c["ruleId"],) for c in run_packs([recs], [the_pack]).consumers]

    cases = [
        (planted("curl", (0, URL)), [("http", "GET /health", "confirmed", "lifted-role", "command")]),
        (planted("wget", (0, URL)), [("http", "GET /health", "confirmed", "lifted-role", "command")]),
        (planted("curl", (0, "-sf"), (1, "-m"), (2, "5"), (3, URL)), [("http", "GET /health", "confirmed", "lifted-role", "command")]),
        (planted("curl", *[(i, "-x") for i in range(9)], (9, URL)), [("http", "GET /health", "confirmed", "lifted-role", "command")]),
        (planted("curl", (0, "-fsS"), (1, DYN_PORT)), [("http", "GET /health", "possible", "partial-key", "command")]),
        (planted("curl", (0, DYN_ORIGIN)), [("http", "GET /health", "possible", "partial-key", "command")]),
        (planted("curl", (0, "-sf"), (1, "5")), []),
        (planted("curl", (0, "-o"), (1, "/tmp/health"), (2, URL)), [("http", "GET /health", "confirmed", "lifted-role", "command")]),
        (planted("curl", (0, "/dev/null")), []),
        (planted("curl", (0, "ftp://127.0.0.1/health")), []),
        (planted("curl", (0, {"unknown": "u"})), []),
        (planted("curl"), []),
        (planted("curls", (0, URL)), []),
        (planted("curl", (0, URL), kind="call"), []),
        (planted("curl", (0, URL), kind="mention"), [("http", "GET /health", "possible", "prompt-mention", "mention")]),
        (planted("wget", (0, URL), kind="mention"), [("http", "GET /health", "possible", "prompt-mention", "mention")]),
        (planted("curl", (1, URL), kind="mention"), []),
        (planted("curl", kind="mention"), []),
    ]
    one = [("http", "GET /health", "confirmed", "lifted-role", "command")]
    for callee in ("curl", "wget"):                  # every seed (each position, each callee) is exercised
        cases += [(planted(callee, *[(i, "-x") for i in range(pos)], (pos, URL)), one) for pos in range(10)]
    cases += [(planted(bad, *[(i, "-x") for i in range(pos)], (pos, URL)), []) for pos in range(10) for bad in ("curls", "xwget")]
    cases += [(planted("curl", *[(i, "-x") for i in range(pos)], (pos, URL), kind="call"), []) for pos in range(10)]
    cases.append((planted("curl", *[(i, "-x") for i in range(10)], (10, URL)), []))      # a11 is past the last seed
    own = [file_("f1", "s.sh", lang="shell", role="primary"), sym("sc", "<script>", fid="f1", path="s.sh", kind="script"),
           sym("fn", "probe", fid="f1", path="s.sh", kind="function"), call("c1", "fn", "curl", line=2, kind="command"),
           vs("c1", "a0", {"template": [{"const": "http://127.0.0.1:8200/health/"}, {"unknown": "runtime-built"}]})]
    cases.append((own, [("http", "GET /health/{?}", "possible", "partial-key", "command")]))
    cases.append((planted("curl", (0, "-sf"), kind="mention"), []))
    cases.append((shell_records('probe() {\n  curl -sf "http://127.0.0.1:8200/health/$1"\n}\n'),
                  [("http", "GET /health/{?}", "possible", "partial-key", "command")]))          # the real param/flow records
    for i, (recs, want) in enumerate(cases):
        if got(recs) != want:
            bad.append(f"case {i}: {got(recs)} != {want}")
    bad += [f"negative {n}" for n in negatives_failures(the_pack)]
    return bad


def test_the_shipped_pack_has_no_behaviour_failures():
    assert behaviour_failures(pack(PID)) == []
