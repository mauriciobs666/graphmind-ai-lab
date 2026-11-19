"""P1-26: the default ``front`` wiring of the clients (``clients/common.py``, ``clients/cli/main.py``; ``docs/plans/connected-code-graph-p1-answers.md`` P1-26 Files).

``Client`` builds ``ccg.front.Front`` itself when no ``front=`` is injected, importing ``ccg.front`` lazily, so ``ask --question`` with a parseable anchor is answered;
free prose stays the typed ``not-built`` error until the interpreter (dispatch 2), and so do a diff, a stack trace, a scope and an unbuilt use case (the request echoed).
The CLI's ``--fail-on`` exits 3 on the front's ``ambiguous``/``unresolved`` documents (P1-25 (8b)), now produced by a real front rather than a double.

    cd cpg && .venv/bin/python -m pytest tests/test_front_clients.py
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import test_engine_answer as TA
from front_helpers import FrontReader

from ccg import answer
from clients import common
from clients.cli import main as cli
from clients.common import Client, NotBuilt, StoreError

SYSTEM = "shop"
CPG = Path(__file__).resolve().parent.parent


@pytest.fixture
def g(monkeypatch):
    g, _ = TA.caller_world(flow=True)
    monkeypatch.setattr(common, "system_repos", lambda system: (g.repo,))
    return g


def client(g, opened=None):
    def factory(system, host, port):
        if opened is not None:
            opened.append(system)
        return FrontReader(g)
    return Client(SYSTEM, host="127.0.0.1", port=16399, reader_factory=factory)


def run_cli(argv, c):
    out, err = io.StringIO(), io.StringIO()
    code = cli.main(argv, client_factory=lambda system: c, stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


def test_the_default_front_answers_a_parsed_question_as_the_structured_ask(g):
    """Byte-equal to the structured ask on its own client: the default front builds on a fresh reader, so its recipe lists the walk's reads only (AC-16)."""
    by_question = client(g).ask(question="impact m.py:5")
    by_request = client(g).ask({"useCase": "impact", "anchor": {"kind": "symbol", "repo": g.repo, "qname": "m.py::A"}})
    assert answer.dumps(by_question) == answer.dumps(by_request)
    assert not {q["name"] for q in by_question["recipe"]["queries"]} & {"repo_exists", "file_by_path", "symbols_of_file", "symbol_by_name"}


def test_the_default_front_looks_files_up_in_the_default_manifests_repos():
    assert common.system_repos("graphmind-ai-lab") == ("graphmind-ai-lab",)              # ccg/system.yaml
    assert common.system_repos("shop") == ()


def test_a_question_naming_a_file_of_a_system_with_no_known_repos_is_not_built():
    g, _ = TA.caller_world(flow=True)
    with pytest.raises(NotBuilt) as e:
        client(g).ask(question="impact m.py:5")
    assert e.value.detail == {"surface": "file-without-repos"}


@pytest.mark.parametrize("q,surface", [("who calls A?", "plain-language-question"), ("diff --git a/m.py b/m.py\n@@ -1 +1 @@", "diff"),
                                       ("  File \"m.py\", line 5, in A", "stack-trace")])
def test_what_the_front_cannot_answer_is_the_typed_not_built_error_and_opens_no_reader(g, q, surface):
    opened = []
    with pytest.raises(NotBuilt) as e:
        client(g, opened).ask(question=q)
    assert e.value.detail["surface"] == surface and opened == []
    assert e.value.payload()["error"]["code"] == "not-built"


@pytest.mark.parametrize("q,key", [("security m.py:5", "useCase"), ("scope: svc impact m.py:5", "scope")])
def test_an_unbuilt_use_case_or_a_scope_is_not_built_with_the_request_echoed_and_opens_no_reader(g, q, key):
    opened = []
    with pytest.raises(NotBuilt) as e:
        client(g, opened).ask(question=q)
    assert key in e.value.detail["request"] and opened == []


def test_a_store_failure_inside_the_front_is_the_typed_store_error(g):
    from ccg.engine.reader import ReaderError

    class Broken(FrontReader):
        def read(self, name, params=None):
            raise ReaderError("query 'repo_exists' failed (ConnectionError)")
    c = Client(SYSTEM, host="127.0.0.1", port=16399, reader_factory=lambda *a: Broken(g))
    with pytest.raises(StoreError):
        c.ask(question="impact m.py:5")


def test_an_invalid_document_from_the_front_never_leaves(g, monkeypatch):
    from ccg import front as F
    monkeypatch.setattr(F, "asking_document", lambda reader, outcome: {"answerSchemaVersion": "1.2"})
    with pytest.raises(StoreError):
        client(g).ask(question="m.py:5")


# ------------------------------------------------------------------------------------------------------------------ the CLI

def test_cli_question_with_a_parseable_anchor_is_answered_exit_0(g):
    code, out, err = run_cli(["ask", "--question", "impact m.py:5"], client(g))
    assert code == 0 and err == "" and json.loads(out)["interpretation"]["status"] == "resolved"


def test_cli_fail_on_exits_3_on_the_fronts_ambiguous_and_unresolved_documents_and_0_without_it(g):
    for q, status in (("m.py:5", "ambiguous"), ("impact gone.py:3", "unresolved")):
        code, out, _ = run_cli(["ask", "--question", q, "--fail-on", "any"], client(g))
        assert code == 3 and json.loads(out)["interpretation"]["status"] == status, q
        code, out, _ = run_cli(["ask", "--question", q], client(g))
        assert code == 0 and json.loads(out)["interpretation"]["status"] == status, q


def test_cli_free_prose_is_not_built_exit_2_nothing_on_stdout(g):
    code, out, err = run_cli(["ask", "--question", "who calls A?"], client(g))
    assert code == 2 and out == "" and json.loads(err)["error"]["code"] == "not-built"


def test_cli_text_format_renders_an_ambiguous_document(g):
    code, out, _ = run_cli(["ask", "--question", "m.py:5", "--format", "text"], client(g))
    assert code == 0 and out.strip()


# ------------------------------------------------------------------------------------------------------------------ lazy import and the broker boundary

def test_the_clients_import_no_front_module_until_a_question_and_a_question_loads_no_broker_module():
    code = """
import sys
sys.path[:0] = [%r, %r]
import clients.common as c, clients.cli.main, clients.mcp.server
assert not [m for m in sys.modules if m.startswith('ccg.front')], 'import'
from front_helpers import FrontReader
import test_engine_answer as TA
g, ids = TA.caller_world(flow=True)
c.system_repos = lambda system: (g.repo,)
cl = c.Client('shop', host='h', port=16399, reader_factory=lambda *a: FrontReader(g))
cl.ask({'useCase': 'impact', 'anchor': {'kind': 'symbol', 'repo': g.repo, 'qname': 'm.py::A'}})
assert not [m for m in sys.modules if m.startswith('ccg.front')], 'structured'
cl.ask(question='impact m.py:5')
cl.ask(question='m.py:5')
assert 'ccg.front' in sys.modules
assert not [m for m in sys.modules if m.startswith('ccg.broker')], sorted(m for m in sys.modules if m.startswith('ccg.broker'))
print('ok')
""" % (str(CPG), str(CPG / "tests"))
    r = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, cwd=str(CPG),
                       env={"PATH": os.environ["PATH"], "CCG_CREDENTIALS_DIR": "/nonexistent"})
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr[-600:]


def test_no_markdown_file_under_the_front():
    """D5: prompts and grammars are ``.txt``/``.json``, never ``.md`` (P1-26 behaviour (8))."""
    front = CPG / "ccg" / "front"
    assert front.is_dir() and not [p for p in front.rglob("*") if p.suffix.lower() == ".md"]
