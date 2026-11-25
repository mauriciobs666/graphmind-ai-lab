"""Q8 withheld-shape predicate (``docs/reviews/connected-code-graph-p1-closeout-ml.md`` ``## Q8 ruling``): pure function of
(gold locator, snapshot file text, language tag). Synthetic cases per family axis, each with a twin that must not match; the dev counts
(J 2, K 1, M 1, C 2, B 2, I 2 = 10 of 49, zero built-family items); purity under an import hook that forbids ``ccg.*``."""
import ast
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

CPG = Path(__file__).resolve().parents[1]
REPO = CPG.parent
TOOLS = CPG / "eval" / "tools"
sys.path.insert(0, str(TOOLS))
import withheld_shapes as W  # noqa: E402


def shape(text, lines, lang):
    return W.withheld_shape({"path": "x", "lines": lines}, text, lang)


# ---- C: markdown front matter
FM = "---\nname: a\nhooks:\n  - command: $HOME/.claude/agents/a/hooks/g.sh\n---\n\n# Body\ncommand: ./scripts/x.sh\n"


def test_c_front_matter_command_is_withheld():
    assert shape(FM, [4], "markdown") == "C"


def test_c_twin_same_text_after_front_matter_is_not_c():
    assert shape(FM, [8], "markdown") is None


def test_c_twin_horizontal_rules_mid_document_are_not_front_matter():
    assert shape("# T\n\n---\ncommand: a/b.sh\n---\n", [4], "markdown") is None


# ---- B: markdown fenced block
FENCE = "Run:\n\n```bash\nscripts/export-cpg.sh a b\n```\n\nor `scripts/export-cpg.sh` inline\n"


def test_b_gold_inside_fence_is_withheld():
    assert shape(FENCE, [4], "markdown") == "B"


def test_b_twin_inline_token_outside_fence_is_not_b():
    assert shape(FENCE, [7], "markdown") is None


def test_b_tilde_fence_and_longer_closing_fence():
    assert shape("~~~\nrun.sh\n~~~~\nafter run.sh\n", [2], "markdown") == "B"
    assert shape("~~~\nrun.sh\n~~~~\nafter run.sh\n", [4], "markdown") is None


def test_b_one_gold_line_outside_the_fence_keeps_the_item_built():
    assert shape(FENCE, [4, 7], "markdown") is None


def test_b_fence_in_a_shell_file_is_not_b():
    assert shape("```\nx\n```\n", [2], "shell") is None


# ---- J: -m package
def test_j_python_dash_m_launch():
    assert shape('exec "$PY" -m modelbench "$@"\n', [1], "shell") == "J"
    assert shape("python3 -m mcp_monitor\n", [1], "shell") == "J"


@pytest.mark.parametrize("line", ['grep -m 1 foo f', 'git commit -m "msg"', 'exec "$PY" "$HERE/server.py"', 'tail -m x', 'python3 -m 5'])
def test_j_twins_are_not_j(line):
    assert shape(line + "\n", [1], "shell") is None


# ---- M: embedded python in shell; I: bash -c string
HEREDOC = 'echo hi\n"$PY" - "$URL" <<\'PY\'\nimport x\nawait s.call_tool(\n    "send_message", {})\nPY\necho "$HERE/after.sh"\n'


def test_m_heredoc_to_python_is_withheld():
    assert shape(HEREDOC, [3], "shell") == "M"
    assert shape(HEREDOC, [4, 5], "shell") == "M"


def test_m_twin_non_python_heredoc_and_lines_after_the_terminator():
    assert shape('cat <<EOF\n"$HERE/x.sh"\nEOF\n', [2], "shell") is None
    assert shape(HEREDOC, [7], "shell") is None


def test_m_python_dash_c_multiline():
    assert shape("python3 -c '\nimport os\nprint(1)\n'\necho z\n", [2], "shell") == "M"


BASHC = ("echo a\nMODE=x bash -c '\n  set -e\n  . \"'\"$HERE\"'/git-provenance.sh\"\n  X=1\n'\n"
         "echo \"$HERE/git-provenance.sh\"\nout=\"$(bash -c 'echo '\"$q\"'' )\"\n")


def test_i_bash_c_multiline_string_with_quote_splices_is_withheld():
    assert shape(BASHC, [4], "shell") == "I"


def test_i_twin_line_after_the_string_closes_is_not_i():
    assert shape(BASHC, [7], "shell") is None


def test_i_twin_awk_file_read_is_not_i():
    assert shape("BLOCK=\"$(awk -v s=1 '$0==s{f=1} f{print}' \"$HERE/pipeline.sh\")\"\n", [1], "shell") is None


def test_i_one_line_bash_c_launch_string():
    assert shape("bash -c '. x/y.sh'\nnext\n", [1], "shell") == "I"
    assert shape("bash -c '. x/y.sh'\nnext\n", [2], "shell") is None


# ---- K: python subprocess with an indirect path
KPY = ('import subprocess\n_B = 1\n\ndef f():\n    subprocess.run(\n        ["bash", str(_B), "ws"],\n        check=True,\n    )\n'
       '    subprocess.run(["bash", "scripts/x.sh", str(_B)])\n    foo(["bash", str(_B)])\n')


def test_k_subprocess_with_indirect_path_argument():
    assert shape(KPY, [5, 6], "python") == "K"
    assert shape(KPY, [6], "python") == "K"


def test_k_twin_literal_path_and_indirect_only_in_a_later_position():
    assert shape(KPY, [9], "python") is None


def test_k_twin_not_a_subprocess_call():
    assert shape(KPY, [10], "python") is None


def test_k_unparseable_python_is_not_withheld():
    assert shape("subprocess.run([\n", [1], "python") is None


# ---- built shapes (G, H one-segment and the rest) are never withheld
@pytest.mark.parametrize("text", ['"$REPO_DIR/scripts/x.sh" "$WS"\n', '. "$HERE/git-provenance.sh"\n', 'exec "$PY" "$HERE/server.py"\n',
                                  '. "$(dirname "$0")/shlib/x.sh"\n', '"$HERE/export-cpg.sh" "$CPG"\n', 'curl -fsS "$B/health"\n'])
def test_g_h_d_e_l_shell_lines_are_not_withheld(text):
    assert shape(text, [1], "shell") is None


def test_markdown_inline_token_not_withheld():
    assert shape("see `./scripts/test_queries.sh`.\n", [1], "markdown") is None


def test_unknown_language_and_bad_lines_are_not_withheld():
    assert shape(FM, [4], "yaml") is None
    assert shape(FM, [], "markdown") is None
    assert shape(FM, [99], "markdown") is None
    assert shape(FM, [0], "markdown") is None


def test_language_tag_by_extension():
    assert [W.language_tag(p) for p in ("a/b.md", "a/b.sh", "a/b.py", "a/b.toml")] == ["markdown", "shell", "python", None]


# ---- dev gold
def _dev_items():
    commit = (CPG / "eval" / "EVAL_COMMIT").read_text().split()[0]
    probe = subprocess.run(["git", "-C", str(REPO), "cat-file", "-e", f"{commit}^{{commit}}"], capture_output=True)
    if probe.returncode:
        pytest.skip("EVAL_COMMIT not in this checkout")
    out = []
    for ln in (CPG / "eval" / "dev" / "ac5-scripts.jsonl").read_text().splitlines():
        it = json.loads(ln)
        loc = it["impacted"]
        text = subprocess.run(["git", "-C", str(REPO), "show", f"{commit}:{loc['path']}"], capture_output=True, text=True, check=True).stdout
        out.append((it, {"path": loc["path"], "lines": it["lines"]}, text, W.language_tag(loc["path"])))
    return out


def test_dev_family_counts_match_the_ruling():
    items = _dev_items()
    assert len(items) == 49
    got = Counter(W.withheld_shape(loc, text, lang) for _, loc, text, lang in items)
    assert {k: v for k, v in got.items() if k} == {"J": 2, "K": 1, "M": 1, "C": 2, "B": 2, "I": 2}
    assert sum(v for k, v in got.items() if k) == 10


def test_dev_g_and_h_one_segment_items_are_built_not_withheld():
    by_id = {it["itemId"][:6]: W.withheld_shape(loc, text, lang) for it, loc, text, lang in _dev_items()}
    # H 1-segment (items 32, 43), G 1-segment tails (13, 36, 46), I awk (47), the F/D/E/A built families
    for i in ("c386af", "e8c36e", "6fee71", "d74079", "f33cbe", "f7a509", "7fc6a2", "46cba1", "1ebbf6", "107ea0"):
        assert by_id[i] is None, i
    assert {by_id[i] for i in ("0fbf43", "d105f0")} == {"I"}
    assert {by_id[i] for i in ("c649a7", "d7b414")} == {"B"}


# ---- purity
def test_module_imports_only_the_standard_library_and_never_the_engine():
    tree = ast.parse((TOOLS / "withheld_shapes.py").read_text())
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods.update(a.name.split(".")[0] for a in n.names)
        elif isinstance(n, ast.ImportFrom):
            mods.add((n.module or "").split(".")[0])
    assert mods <= {"__future__", "ast", "re", "typing"}


_HOOK = r"""
import importlib.abc, importlib.util, json, sys
class Ban(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in ('ccg', 'engine', 'assemble', 'extractors', 'resolver', 'gate', 'match_keys'):
            raise ImportError('banned import: ' + name)
sys.meta_path.insert(0, Ban())
spec = importlib.util.spec_from_file_location('ws_iso', sys.argv[1]); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
items = json.load(sys.stdin)
print(json.dumps([m.withheld_shape(i['loc'], i['text'], i['lang']) for i in items]))
assert not [k for k in sys.modules if k.split('.')[0] == 'ccg']
"""


def test_predicate_is_identical_with_ccg_imports_forbidden(tmp_path):
    items = _dev_items()
    payload = json.dumps([{"loc": loc, "text": text, "lang": lang} for _, loc, text, lang in items])
    r = subprocess.run([sys.executable, "-I", "-c", _HOOK, str(TOOLS / "withheld_shapes.py")], input=payload, capture_output=True, text=True,
                       cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout) == [W.withheld_shape(loc, text, lang) for _, loc, text, lang in items]


# ---- killers for mutants found by the copy-aside pass
def test_j_one_plain_gold_line_keeps_the_item_built():
    assert shape('exec "$PY" -m modelbench\n"$HERE/x.sh"\n', [1, 2], "shell") is None


def test_i_lines_after_a_quote_splice_stay_inside_the_string():
    assert shape(BASHC, [5], "shell") == "I"


def test_k_twin_non_subprocess_attribute_call_with_indirect_argv():
    assert shape("import os\nos.run(['bash', str(P)])\n", [2], "python") is None
    assert shape("import subprocess\nsubprocess.run(['bash', str(P)])\n", [2], "python") == "K"


@pytest.mark.parametrize("lang,text", [("shell", 'exec "$PY" -m m\n'), ("python", "subprocess.run(['bash', str(P)])\n"), ("markdown", "```\nx\n```\n")])
def test_out_of_range_gold_lines_are_not_withheld(lang, text):
    n = len(text.split("\n"))
    assert shape(text, [n + 3], lang) is None
    assert shape(text, [0], lang) is None
