"""Q8 withheld-shape predicate (``docs/reviews/connected-code-graph-p1-closeout-ml.md`` ``## Q8 ruling``): pure function of
(gold locator, snapshot file text, language tag). Synthetic cases per family axis, each with a twin that must not match; the dev counts
(J 2, K 1, M 1, C 2, B 2, I 2 = 10 of 49, zero built-family items); purity under an import hook that forbids ``ccg.*``."""
import ast
import re
import shlex
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


# ---- F2: launcher word x flag x opener axes (expected values written by hand, not derived from the module's regexes)
LAUNCHERS = ['python', 'python3', 'python3.12', '/usr/bin/python3', '"$VENV/bin/python"', '"$HERE/.venv/bin/python3"', '$PY', '"$PY"', '${PY}',
             '"${PY[@]}"', '"$VENV_PY"', 'exec "$PY"', 'env FOO=1 "$PY"', 'uv run python']
FLAGS = ["", " -u", " -I", " -u -I", " -X dev", " -W ignore -u"]


@pytest.mark.parametrize("launcher", LAUNCHERS)
@pytest.mark.parametrize("flags", FLAGS)
def test_j_launcher_and_flag_axes(launcher, flags):
    assert shape(f'{launcher}{flags} -m pkg.sub --opt\n', [1], "shell") == "J"


@pytest.mark.parametrize("launcher", LAUNCHERS)
@pytest.mark.parametrize("flags", FLAGS)
def test_m_python_dash_c_and_heredoc_axes(launcher, flags):
    assert shape(f"{launcher}{flags} -c '\nimport os\n'\n", [2], "shell") == "M"
    assert shape(f"{launcher}{flags} <<EOF\nimport os\nEOF\n", [2], "shell") == "M"
    assert shape(f"{launcher}{flags} - a b <<'PY'\nimport os\nPY\n", [2], "shell") == "M"


@pytest.mark.parametrize("launcher", ["bash", "sh", "zsh", "dash", "/bin/bash", "\"$SHELL\"", "\"$BASH\"", "env X=1 bash", "sudo -n bash"])
@pytest.mark.parametrize("flags", [" -c", " -lc", " -ec", " -euc", " -l -c", " -o pipefail -c"])
def test_i_shell_launcher_and_flag_axes(launcher, flags):
    assert shape(f"{launcher}{flags} '\n echo hi\n'\n", [2], "shell") == "I"


@pytest.mark.parametrize("line", [
    'grep -m 1 foo f', 'git commit -m "msg"', 'tail -m x', 'python3 -m 5', 'python3 x.py -m pkg',
    'mypython2 -m x', 'pythonic -m x', '"$HERE/x.sh" -m pkg', '"$CMD" -m msg', 'python -mpkg',
])
def test_j_false_positive_twins(line):
    assert shape(line + "\n", [1], "shell") is None


@pytest.mark.parametrize("line", ['bash script.sh -c x', 'grep -c foo f', 'cat -c', 'python3 x.py -c', 'mash -c "x"'])
def test_i_m_false_positive_twins(line):
    assert shape(line + "\n", [1], "shell") is None


def test_dash_c_spacing_variants():
    assert shape("bash -c  '\nx\n'\n", [2], "shell") == "I"
    assert shape("bash\t-c\t'\nx\n'\n", [2], "shell") == "I"
    assert shape("zsh -c '\nx\n'\n", [2], "shell") == "I"


# ---- heredoc terminator rules and the F6 over-withholding twins
def test_heredoc_dash_form_accepts_a_tab_indented_terminator_plain_form_does_not():
    dash = 'python3 <<-PY\n\timport os\n\tPY\n"$HERE/x.sh"\n'
    assert shape(dash, [2], "shell") == "M" and shape(dash, [4], "shell") is None
    plain = 'python3 <<PY\nimport os\n\tPY\nPY\n"$HERE/x.sh"\n'
    assert shape(plain, [3], "shell") == "M" and shape(plain, [5], "shell") is None


def test_heredoc_terminator_with_carriage_return():
    assert shape('python3 <<PY\r\nimport os\r\nPY\r\n"$HERE/x.sh"\r\n', [4], "shell") is None


@pytest.mark.parametrize("opener", ['python3 - <<< "abc"', "python3 -c 'print(1<<3)'", 'python3 -c "print(1 << 3)"', 'x=$((1<<3))', 'python3 <<<"$X"',
                                    "# bash -c '", "  # python3 <<EOF", 'cat <<EOF'])
def test_f6_here_strings_shifts_and_comments_open_no_region(opener):
    assert shape(f'{opener}\n"$HERE/x.sh"\n', [2], "shell") is None


def test_f6_one_line_string_followed_by_another_command_is_not_wholly_the_string():
    assert shape("bash -c 'echo a' ; \"$HERE/x.sh\"\n", [1], "shell") is None
    assert shape("bash -c 'echo a'\n", [1], "shell") == "I"


# ---- every-gold-line invariant for C, M, I, K (two-line gold, one plain line; all -> any survives otherwise)
@pytest.mark.parametrize("text,lang,inside", [
    ("---\nname: a\ncommand: x/y.sh\n---\nbody x/y.sh\n", "markdown", 3),
    ('python3 <<PY\nimport os\nPY\n"$HERE/x.sh"\n', "shell", 2),
    ("bash -c '\necho a\n'\n\"$HERE/x.sh\"\n", "shell", 2),
    ("import subprocess\nsubprocess.run([P, 'a'])\nsubprocess.run(['bash', 'x.sh'])\n", "python", 2),
])
def test_every_gold_line_invariant(text, lang, inside):
    plain = len(text.split("\n")) - 1
    assert shape(text, [inside], lang) is not None
    assert shape(text, [inside, plain], lang) is None
    assert shape(text, [plain, inside], lang) is None


# ---- K: every subprocess spelling, tuple argv
@pytest.mark.parametrize("fn", ["run", "Popen", "call", "check_call", "check_output"])
def test_k_subprocess_functions(fn):
    assert shape(f"import subprocess\nsubprocess.{fn}(['bash', str(P)])\n", [2], "python") == "K"
    assert shape("import subprocess\nsubprocess.%s(('bash', P))\n" % fn, [2], "python") == "K"


def test_k_literal_argv_is_not_withheld_by_design():
    assert shape("import subprocess\nsubprocess.run(['bash', 'scripts/x.sh'])\n", [2], "python") is None


# ---- F3: Markdown fence and front-matter grammar, edge by edge
def test_fence_nested_in_containers_and_indented():
    assert shape("- ```bash\n  run.sh\n  ```\nrun.sh\n", [2], "markdown") == "B"
    assert shape("> ```bash\n> run.sh\n> ```\nrun.sh\n", [2], "markdown") == "B"
    assert shape("1. a\n\n    ```\n    run.sh\n    ```\n", [4], "markdown") == "B"
    assert shape("      ```\n      run.sh\n      ```\nrun.sh\n", [2], "markdown") == "B"
    assert shape("- ```bash\n  run.sh\n  ```\nrun.sh\n", [4], "markdown") is None


def test_backtick_info_string_with_a_backtick_is_no_fence():
    t = "````` and `code` here\nrun.sh\n```\n"
    assert shape(t, [2], "markdown") is None
    assert shape("```bash\nrun.sh\n```\n", [2], "markdown") == "B"
    assert shape("~~~ a`b\nrun.sh\n~~~\n", [2], "markdown") == "B"


def test_two_backticks_do_not_open_a_fence():
    assert shape("``x``\nrun.sh\n", [2], "markdown") is None


def test_closer_must_match_character_length_and_carry_no_text():
    assert shape("```\na.sh\n~~~\nb.sh\n```\nc.sh\n", [4], "markdown") == "B"      # ~~~ does not close a backtick fence
    assert shape("```\na.sh\n```x\nb.sh\n```\nc.sh\n", [4], "markdown") == "B"     # a closer with text does not close
    assert shape("````\na.sh\n```\nb.sh\n````\nc.sh\n", [4], "markdown") == "B"    # a shorter closer does not close
    assert shape("```\na.sh\n```\nc.sh\n", [4], "markdown") is None


def test_front_matter_needs_a_yaml_first_line_and_the_200_line_cap():
    assert shape("---\nSome prose\n---\ncommand: x\n", [2], "markdown") is None            # a horizontal rule, not YAML
    assert shape("---\n\n# comment\ncommand: x\n---\n", [4], "markdown") == "C"
    assert shape("---\n- a: b\n---\n", [2], "markdown") == "C"
    body = "\n".join(f"k{i}: v" for i in range(198))
    assert shape(f"---\n{body}\n---\n", [100], "markdown") == "C"                          # closes at line 200
    body = "\n".join(f"k{i}: v" for i in range(199))
    assert shape(f"---\n{body}\n---\n", [100], "markdown") is None                         # closes at line 201
    assert shape("---\nname: a\n", [2], "markdown") is None                                # never closes
    assert shape("---\n---\nname: a\n", [3], "markdown") is None


def test_front_matter_and_fences_with_carriage_returns():
    assert shape("---\r\nname: a\r\ncommand: x\r\n---\r\nbody\r\n", [3], "markdown") == "C"
    assert shape("```\r\nrun.sh\r\n```\r\nrun.sh\r\n", [2], "markdown") == "B"
    assert shape("```\r\nrun.sh\r\n```\r\nrun.sh\r\n", [4], "markdown") is None


def test_fence_lines_inside_front_matter_open_nothing():
    assert shape("---\nname: a\n```\n---\nrun.sh\n", [5], "markdown") is None


def test_language_tag_aliases_and_case():
    assert [W.language_tag(p) for p in ("a/B.MD", "a/b.markdown", "a/B.Markdown", "x.SH", "x.bash", "x.PY")] == [
        "markdown", "markdown", "markdown", "shell", "shell", "python"]


# ---- differentials over the repo's tracked corpus (test only: the extractor is read here, never by the module)
def _tracked(glob):
    r = subprocess.run(["git", "-C", str(REPO), "ls-files", glob], capture_output=True, text=True)
    if r.returncode or not r.stdout.strip():
        pytest.skip("no git corpus")
    return [REPO / p for p in r.stdout.split("\n") if p and (REPO / p).is_file()]


def _engine_fenced(E, lines):
    fm = E._front_matter_end(lines)
    out, fence = set(), None
    for n, line in enumerate(lines, 1):
        if fm and n <= fm:
            continue
        run = E._fence_run(line)
        if fence is not None:
            out.add(n)
            if run and run[0] == fence[0] and run[1] >= fence[1] and not run[2].strip():
                fence = None
        elif run and not (run[0] == "`" and "`" in run[2]):
            fence = (run[0], run[1])
            out.add(n)
    return out


def test_markdown_grammar_matches_the_shipped_extractor_over_the_tracked_corpus():
    sys.path.insert(0, str(CPG))
    from extractors.markdown import extract as E
    fm = eng_only = pred_only = 0
    files = _tracked("*.md")
    for f in files:
        try:
            lines = f.read_text(encoding="utf-8").split("\n")
        except UnicodeDecodeError:
            continue
        fm += W._front_matter_end(lines) != E._front_matter_end(lines)
        ef, pf = _engine_fenced(E, lines), W._fenced(lines)
        eng_only += len(ef - pf)
        pred_only += len(pf - ef)
    assert len(files) > 500
    assert (fm, eng_only, pred_only) == (0, 0, 0)


def _python_dash_m_lines(text):
    """An independent probe: tokenise each non-comment line and look for a python launcher, then only flags, then ``-m <pkg>``."""
    out = []
    for n, line in enumerate(text.split("\n"), 1):
        if line.lstrip().startswith("#"):
            continue
        try:
            toks = shlex.split(line, comments=True)
        except ValueError:
            continue
        for k, tk in enumerate(toks):
            if re.fullmatch(r"python[\d.]*", tk.rsplit("/", 1)[-1]) or ("$" in tk and "py" in tk.lower()):
                j = k + 1
                while j < len(toks) and toks[j].startswith("-") and toks[j] != "-m":
                    j += 1
                if j + 1 < len(toks) and toks[j] == "-m" and re.match(r"[A-Za-z_][\w.]*$", toks[j + 1]):
                    out.append(n)
                break
    return out


def test_every_python_dash_m_line_of_the_tracked_shell_corpus_is_withheld():
    total = missed = 0
    for f in _tracked("*.sh"):
        text = f.read_text(encoding="utf-8", errors="replace")
        for n in _python_dash_m_lines(text):
            total += 1
            missed += W.withheld_shape({"lines": [n]}, text, "shell") is None
    assert total >= 15
    assert missed == 0


# ---- killers, second mutation round
def test_c_the_opening_and_closing_rules_are_not_front_matter_content():
    t = "---\nname: a\n---\nbody\n"
    assert shape(t, [1], "markdown") is None and shape(t, [3], "markdown") is None and shape(t, [2], "markdown") == "C"


def test_ordered_list_fence_markers():
    assert shape("1. ```bash\n   run.sh\n   ```\nrun.sh\n", [2], "markdown") == "B"
    assert shape("12) ```bash\n    run.sh\n    ```\nrun.sh\n", [2], "markdown") == "B"


def test_hash_inside_a_word_is_not_a_comment_before_a_launcher():
    assert shape("FOO=a#b bash -c '\nx\n'\n", [2], "shell") == "I"
    assert shape("FOO=a#b python3 <<EOF\nx\nEOF\n", [2], "shell") == "M"
