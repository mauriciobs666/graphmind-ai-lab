"""``Resolver.element``: an imported name beats a same-named "constant of this file" (RCA ``docs/reviews/connected-code-graph-p1-41-rca.md`` c2).

Joern reports an import-bound name as a module-level assignment of the importing file, so ``constants[path]`` holds
``<importing file>::NAME`` for every imported ``NAME``. The import is the truth about where the value lives; a ref naming
the importing file points at a constant that has no record. Expected values are written by hand.
"""
import pytest

from extractors.joern.normalize import core

# language -> (importing file, repo path of it, resolved target file, local constant that is NOT imported)
LANGS = {"python": ("a.py", "svc/a.py", "svc/cfg.py"), "javascript": ("a.ts", "svc/a.ts", "svc/cfg.ts")}

# import form -> (record builder, local name used in the file, expected element for that name)
FORMS = {
    "resolved": (lambda tgt: {"module": ".cfg", "names": [{"name": "TOPIC"}], "resolvedPath": tgt, "_sub": None},
                 "TOPIC", lambda tgt: {"ref": f"{tgt}::TOPIC"}),
    "aliased": (lambda tgt: {"module": ".cfg", "names": [{"name": "TOPIC", "alias": "T"}], "resolvedPath": tgt, "_sub": None},
                "T", lambda tgt: {"ref": f"{tgt}::TOPIC"}),
    "unresolved": (lambda tgt: {"module": "ext", "names": [{"name": "TOPIC"}], "resolvedPath": None, "_sub": None},
                   "TOPIC", lambda tgt: {"importRef": {"module": "ext", "name": "TOPIC"}}),
    "unresolved-aliased": (lambda tgt: {"module": "ext", "names": [{"name": "TOPIC", "alias": "T"}], "resolvedPath": None, "_sub": None},
                           "T", lambda tgt: {"importRef": {"module": "ext", "name": "TOPIC"}}),
    "submodule": (lambda tgt: {"module": "pkg", "names": [{"name": "TOPIC"}], "resolvedPath": "svc/pkg/__init__.py",
                               "_sub": "svc/pkg/TOPIC.py"},
                  "TOPIC", lambda tgt: {"importRef": {"module": "pkg", "name": "TOPIC"}}),
}

def _positions(lang):
    # the text piece of a Python f-string is raw source text (`tests/golden/fstring/probe.json`), not a quoted literal (P1-50)
    text = "p." if lang == "python" else "'p.'"
    return {
        "bare": (lambda name: {"o": "name", "n": name}, lambda el: el),
        "concat-part": (lambda name: {"o": "add", "p": [{"o": "name", "n": name}, {"o": "lit", "c": "'.x'"}]},
                        lambda el: {"concat": [el, {"const": ".x"}]}),
        "template-part": (lambda name: {"o": "fmt", "p": [{"o": "lit", "c": text}, {"o": "name", "n": name}]},
                          lambda el: {"template": [{"const": "p."}, el]}),
    }


POSITIONS = _positions
POSITION_NAMES = ("bare", "concat-part", "template-part")


@pytest.mark.parametrize("position", POSITION_NAMES)
@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize("lang", LANGS)
def test_an_imported_name_is_not_shadowed_by_the_importing_files_own_constant_entry(tmp_path, lang, form, position):
    file, path, target = LANGS[lang]
    make_import, local, expect = FORMS[form]
    expr, wrap = POSITIONS(lang)[position]
    B = core.Builder(lang, "svc", "r", {}, core.Source(tmp_path))
    R = core.Resolver(B, {file: [make_import(target)]}, {path: {f"{path}::{local}"}})   # the import-bound name is in the set
    e = expr(local)
    assert R.element(e, file) == wrap(expect(target))


@pytest.mark.parametrize("lang", LANGS)
def test_a_constant_of_the_file_that_is_not_imported_still_resolves_to_the_file(tmp_path, lang):
    file, path, target = LANGS[lang]
    B = core.Builder(lang, "svc", "r", {}, core.Source(tmp_path))
    imports = {file: [{"module": ".cfg", "names": [{"name": "TOPIC"}], "resolvedPath": target, "_sub": None}]}
    R = core.Resolver(B, imports, {path: {f"{path}::TOPIC", f"{path}::LIMIT"}})
    assert R.element({"o": "name", "n": "LIMIT"}, file) == {"ref": f"{path}::LIMIT"}
