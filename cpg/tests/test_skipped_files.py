"""P1-08 — ``header.skippedFiles`` (`extractors/joern/skipped.py`): the spec4 §1 coverage probe.

Plan: ``docs/plans/connected-code-graph-p1-extraction.md`` § P1-08; rule: ``docs/plans/connected-code-graph-bundle-spec4.md`` §1.
The table is language x file kind x location x override, written out by hand (never derived from the function under
test) and checked for completeness: a missing cell fails. Every cell asserts that the path is listed exactly once with
one reason, or not at all. The "no ``file`` record" half of each cell is the writer's job (the caller drops a listed
path's record); the validator's own check of that is exercised at the end.
"""
from __future__ import annotations

import itertools

import pytest

from ccg.bundle import BundleError, JS_TEST_EXTENSIONS, _Semantics
from ccg.units import Unit
from extractors.joern import skipped
from extractors.joern.skipped import skipped_files

ROOT = "web"
UNIT = {"javascript": Unit("u1", "r", ROOT, "javascript"), "python": Unit("u2", "r", ROOT, "python"),
        "go": Unit("u3", "r", ROOT, "go")}

LANGS = ("javascript", "python", "go")
KINDS = ("prod-parsed", "prod-unparsed", "test-parsed", "test-unparsed")   # test-unparsed = test x frontend-ignore
LOCATIONS = ("under-root", "vendored", "outside-root")
OVERRIDES = (False, True)

#: (language, kind) -> the path under the unit root; the test kinds are shaped like a JS test, whatever the language
_REL = {
    ("javascript", "prod-parsed"): "src/app.ts", ("javascript", "prod-unparsed"): "src/i18n/format.ts",
    ("javascript", "test-parsed"): "src/app.test.ts", ("javascript", "test-unparsed"): "src/i18n/format.spec.ts",
    ("python", "prod-parsed"): "pkg/mod.py", ("python", "prod-unparsed"): "pkg/i18n/mod.py",
    ("python", "test-parsed"): "pkg/mod.test.py", ("python", "test-unparsed"): "pkg/e2e/mod.py",
    ("go", "prod-parsed"): "pkg/mod.go", ("go", "prod-unparsed"): "pkg/i18n/mod.go",
    ("go", "test-parsed"): "pkg/mod.test.go", ("go", "test-unparsed"): "pkg/e2e/mod.go",
}


def _path(lang: str, kind: str, location: str) -> str:
    rel = _REL[(lang, kind)]
    return {"under-root": f"{ROOT}/{rel}", "vendored": f"{ROOT}/node_modules/{rel}", "outside-root": f"other/{rel}"}[location]


#: (language, kind, location, override) -> the expected reason, None (listed in neither) or "raises" (override on, a
#: non-test source the frontend still dropped: no valid reason exists). Written out, 72 cells.
EXPECTED: dict[tuple[str, str, str, bool], str | None] = {}
for _loc in LOCATIONS:
    for _ov in OVERRIDES:
        for _lang in ("python", "go"):                       # no test pattern and no frontend-ignore reason in v1
            for _kind in KINDS:
                EXPECTED[(_lang, _kind, _loc, _ov)] = None
        for _kind in KINDS:
            EXPECTED[("javascript", _kind, _loc, _ov)] = None
for _ov in OVERRIDES:                                        # the only cells with an answer: JS under the root
    EXPECTED[("javascript", "prod-parsed", "under-root", _ov)] = None
    EXPECTED[("javascript", "test-parsed", "under-root", _ov)] = "test"
    EXPECTED[("javascript", "test-unparsed", "under-root", _ov)] = "test"          # precedence over frontend-ignore
EXPECTED[("javascript", "prod-unparsed", "under-root", False)] = "frontend-ignore"
EXPECTED[("javascript", "prod-unparsed", "under-root", True)] = "raises"


def test_the_table_is_complete():
    want = set(itertools.product(LANGS, KINDS, LOCATIONS, OVERRIDES))
    assert set(EXPECTED) == want and len(want) == 72


@pytest.mark.parametrize("lang,kind,location,override", sorted(EXPECTED))
def test_cell(lang, kind, location, override):
    path = _path(lang, kind, location)
    parsed = {path: kind.endswith("-parsed")}
    want = EXPECTED[(lang, kind, location, override)]
    if want == "raises":
        with pytest.raises(ValueError, match=r"override"):
            skipped_files(UNIT[lang], parsed, override)
        return
    got = skipped_files(UNIT[lang], parsed, override)
    assert got == ([] if want is None else [{"path": path, "reason": want}])


# --- the test pattern ---------------------------------------------------------------------------------------------

JS = UNIT["javascript"]


def _only(path: str, parsed: bool, unit: Unit = JS, override: bool = False):
    return skipped_files(unit, {path: parsed}, override)


@pytest.mark.parametrize("ext", JS_TEST_EXTENSIONS)
@pytest.mark.parametrize("kind", ("test", "spec"))
@pytest.mark.parametrize("parsed", (True, False))
def test_every_test_basename_form_is_test_whether_or_not_parsed(ext, kind, parsed):
    p = f"web/src/a.{kind}.{ext}"
    assert _only(p, parsed) == [{"path": p, "reason": "test"}]


@pytest.mark.parametrize("seg", ("e2e", "__tests__"))
@pytest.mark.parametrize("parsed", (True, False))
def test_a_directory_segment_below_the_root_is_test(seg, parsed):
    p = f"web/src/{seg}/a.ts"
    assert _only(p, parsed) == [{"path": p, "reason": "test"}]


def test_the_root_itself_named_e2e_is_not_a_test_segment():
    unit = Unit("u", "r", "web/e2e", "javascript")
    assert _only("web/e2e/src/a.ts", True, unit) == []
    assert _only("web/e2e/src/e2e/a.ts", True, unit) == [{"path": "web/e2e/src/e2e/a.ts", "reason": "test"}]
    assert _only("web/e2e/src/a.ts", False, unit) == [{"path": "web/e2e/src/a.ts", "reason": "frontend-ignore"}]


@pytest.mark.parametrize("rel", ("src/test.ts", "src/latest.ts", "src/a.testing.ts", "src/a.test.json", "src/e2ex/a.ts",
                                 "src/tests/a.ts", "src/a-test.ts", "src/a_spec.ts"))
def test_near_misses_of_the_pattern_are_not_test(rel):
    p = f"web/{rel}"
    assert _only(p, True) == []
    assert _only(p, False) == ([{"path": p, "reason": "frontend-ignore"}] if rel.endswith(".ts") else [])


def test_test_pattern_applies_at_the_repo_root_unit():
    unit = Unit("u", "r", ".", "javascript")
    assert _only("e2e/a.ts", True, unit) == [{"path": "e2e/a.ts", "reason": "test"}]
    assert _only("src/a.ts", False, unit) == [{"path": "src/a.ts", "reason": "frontend-ignore"}]


def test_a_unit_root_prefix_is_a_path_prefix_not_a_string_prefix():
    assert _only("webx/src/a.ts", False) == []


# --- what is listed in neither ------------------------------------------------------------------------------------

@pytest.mark.parametrize("rel", ("src/types.d.ts", "src/types.d.mts", "src/types.d.cts", "__tests__/types.d.ts",
                                 "src/a.test.d.ts", "src/i18n/en.d.ts"))
@pytest.mark.parametrize("parsed", (True, False))
def test_declaration_files_are_listed_in_neither(rel, parsed):
    assert _only(f"web/{rel}", parsed) == []


@pytest.mark.parametrize("rel", ("pkg/a.pyi", "pkg/tests/a.pyi", "pkg/i18n/a.pyi"))
@pytest.mark.parametrize("parsed", (True, False))
def test_pyi_is_listed_in_neither(rel, parsed):
    assert _only(f"web/{rel}", parsed, UNIT["python"]) == []


@pytest.mark.parametrize("seg", ("_vendor", "node_modules", "third-party", "third_party", "vendor", "vendored"))
@pytest.mark.parametrize("parsed", (True, False))
def test_every_vendored_directory_hides_a_file_even_a_test_one(seg, parsed):
    assert _only(f"web/{seg}/a.ts", parsed) == []
    assert _only(f"web/src/{seg}/a.test.ts", parsed) == []


def test_a_vendored_name_as_a_basename_is_not_a_vendored_directory():
    assert _only("web/src/vendor.ts", False) == [{"path": "web/src/vendor.ts", "reason": "frontend-ignore"}]


@pytest.mark.parametrize("rel", ("README.md", "src/data.json", "src/style.css", "src/app.vue", "src/app"))
def test_a_non_source_file_is_listed_in_neither(rel):
    assert _only(f"web/{rel}", False) == []


def test_the_vendored_set_is_the_writers():
    from extractors.joern.normalize import assemble_bundle
    assert sorted(skipped.DEFAULT_VENDORED) == sorted(assemble_bundle.DEFAULT_VENDORED)


# --- shape: sorted by byte order, unique, closed keys -------------------------------------------------------------

def test_result_is_sorted_by_utf8_bytes_with_closed_keys():
    paths = ["web/src/z.ts", "web/src/é.ts", "web/src/B.ts", "web/src/a.test.ts", "web/src/\U0001f600.ts"]
    got = skipped_files(JS, {p: False for p in paths}, False)
    assert [e["path"] for e in got] == sorted(paths, key=lambda s: s.encode("utf-8"))
    assert all(set(e) == {"path", "reason"} for e in got)


def test_mixed_population_one_entry_per_listed_file():
    parsed = {
        "web/src/app.ts": True, "web/src/app.test.ts": True, "web/src/i18n/format.ts": False,
        "web/src/i18n/format.test.ts": False, "web/e2e/a.ts": False, "web/src/types.d.ts": False,
        "web/node_modules/x/a.ts": False, "other/a.ts": False,
    }
    assert skipped_files(JS, parsed, False) == [
        {"path": "web/e2e/a.ts", "reason": "test"},
        {"path": "web/src/app.test.ts", "reason": "test"},
        {"path": "web/src/i18n/format.test.ts", "reason": "test"},
        {"path": "web/src/i18n/format.ts", "reason": "frontend-ignore"},
    ]


def test_override_on_errors_name_the_files():
    with pytest.raises(ValueError, match=r"override.*web/src/i18n/format\.ts"):
        skipped_files(JS, {"web/src/i18n/format.ts": False, "web/src/a.ts": True}, True)


def test_override_on_with_everything_parsed_or_test_is_fine():
    parsed = {"web/src/i18n/format.ts": True, "web/src/a.test.ts": False, "web/src/b.ts": True}
    assert skipped_files(JS, parsed, True) == [{"path": "web/src/a.test.ts", "reason": "test"}]


def test_empty_population_gives_an_empty_list_so_the_header_omits_the_field():
    assert skipped_files(JS, {}, False) == []


def test_unit_language_other_than_js_is_none_for_every_cell():
    parsed = {"web/pkg/a.py": False, "web/pkg/a.test.py": True, "web/pkg/e2e/a.py": True}
    assert skipped_files(UNIT["python"], parsed, False) == []
    assert skipped_files(UNIT["python"], parsed, True) == []


# --- agreement with the S10 validator ----------------------------------------------------------------------------

def _validator(unit: Unit, files: list[str]) -> _Semantics:
    s = object.__new__(_Semantics)
    s.header = {"unit": {"lang": unit.language, "repoId": unit.repoId, "rootPath": unit.rootPath}}
    s.root, s.vendored = unit.rootPath, set(skipped.DEFAULT_VENDORED)
    s.by_type = {"file": {f"f{i}": (i, {"path": p}) for i, p in enumerate(files)}}
    return s


def test_the_validator_accepts_what_the_extractor_lists():
    parsed = {"web/src/app.ts": True, "web/src/app.test.ts": True, "web/src/i18n/format.ts": False,
              "web/src/i18n/format.spec.ts": False, "web/e2e/a.ts": False, "web/__tests__/b.ts": True}
    got = skipped_files(JS, parsed, False)
    assert len(got) == 5
    _validator(JS, ["web/src/app.ts"])._skipped_files(got)


def test_the_validator_rejects_a_listed_path_that_keeps_a_file_record():
    got = skipped_files(JS, {"web/src/app.test.ts": True}, False)
    with pytest.raises(BundleError, match="has a file record"):
        _validator(JS, ["web/src/app.test.ts"])._skipped_files(got)


# --- production cells over the eight extensions (spec3 §3.3), not one ----------------------------------------------

@pytest.mark.parametrize("ext", JS_TEST_EXTENSIONS)
def test_production_parsed_is_never_listed_and_unparsed_is_frontend_ignore(ext):
    p = f"web/src/app.{ext}"
    assert _only(p, True) == []
    assert _only(p, False) == [{"path": p, "reason": "frontend-ignore"}]
    with pytest.raises(ValueError, match="override"):
        _only(p, False, override=True)
    assert _only(p, True, override=True) == []


# --- the candidate walk (F1, F2) ----------------------------------------------------------------------------------

def _stage(tmp_path):
    st = tmp_path / "stage"
    for rel, data in {"src/a.ts": "x", "src/deep/er/b.tsx": "x", "src/i18n/f.ts": "x", "src/empty.ts": "",
                      "src/a.test.ts": "x", "README.md": "x", "src/real.cjs": "x"}.items():
        (st / rel).parent.mkdir(parents=True, exist_ok=True)
        (st / rel).write_text(data)
    (st / "src" / "link.ts").symlink_to(st / "src" / "a.ts")
    return st


def test_candidates_are_repo_relative_sorted_nonempty_regular_files(tmp_path):
    st = _stage(tmp_path)
    assert skipped.candidate_sources(st, "web") == [
        "web/README.md", "web/src/a.test.ts", "web/src/a.ts", "web/src/deep/er/b.tsx", "web/src/i18n/f.ts",
        "web/src/real.cjs"]
    assert skipped.candidate_sources(st, ".")[0] == "README.md"


def test_the_walk_uses_the_runners_predicate_object():
    from extractors.joern import run
    assert skipped.is_staged_source is run.is_staged_source


def test_the_runner_counts_with_the_same_predicate(tmp_path):
    from extractors.joern import run
    st = _stage(tmp_path)
    exts = run.LANGUAGES["javascript"][1]
    assert run._count_sources(st, exts) == len([p for p in skipped.candidate_sources(st, ".") if p.endswith(tuple(exts))])


def test_parsed_map_marks_parsed_and_adds_a_parsed_empty_file(tmp_path):
    st = _stage(tmp_path)
    m = skipped.parsed_map(st, "web", ["web/src/a.ts", "web/src/empty.ts"])
    assert m["web/src/a.ts"] is True and m["web/src/empty.ts"] is True
    assert m["web/src/i18n/f.ts"] is False and "web/src/link.ts" not in m
    assert skipped_files(JS, m, False) == [{"path": "web/src/a.test.ts", "reason": "test"},
                                           {"path": "web/src/deep/er/b.tsx", "reason": "frontend-ignore"},
                                           {"path": "web/src/i18n/f.ts", "reason": "frontend-ignore"},
                                           {"path": "web/src/real.cjs", "reason": "frontend-ignore"}]


def test_an_empty_or_symlinked_unparsed_file_is_not_listed(tmp_path):
    st = _stage(tmp_path)
    got = skipped_files(JS, skipped.parsed_map(st, "web", []), False)
    assert "web/src/empty.ts" not in [e["path"] for e in got] and "web/src/link.ts" not in [e["path"] for e in got]
