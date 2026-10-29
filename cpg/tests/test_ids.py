"""Tests for ``ccg.ids`` — the id hash and canonical tuples.

Spec: ``docs/plans/connected-code-graph-bundle-spec.md`` v1.3 §3.1 (hash), §3.2 (tuples),
§3.6 (golden vectors). Every expected id below is **copied from the spec's §3.6 table**, never
computed by the code under test. Ids that §3.6 states inside a row (the enclosing symbol of
GV-14/GV-17/GV-20 and GV-20's call site) are copied the same way, as ``GV-14s`` etc.
"""

import re

import pytest

from ccg import ids

QNAME_GV3 = "falkor-chat/server/falkorchat/storefront_api.py::build_storefront_router.health"

# --- ids the spec states inside GV-14, GV-17 and GV-20 (copied, not computed) ---
SYM_GET_HEALTH = "8ae57f349ffed0533e458ee5341f2936"  # GV-14's enclosing symbol
SYM_HTTPERR_INIT = "3177d8d58a31424b59c4c7bd458aa801"  # GV-17's enclosing symbol
SYM_TOOLS_CLOSE = "e308e9b40866f15eb9479730c3d2d5fa"  # GV-20's S
CS_CLOSE_SOON_1 = "79f8675f500467a9a3d2871ea327ed78"  # GV-20's call site

# (vector, tuple as given — before normalization, expected id from §3.6)
GOLDEN = [
    ("GV-1", ["unit", "graphmind-ai-lab", "falkor-chat/server", "python"],
     "0fb899742d4d3687495c30357ca16253"),
    ("GV-2", ["file", "graphmind-ai-lab", "falkor-chat/server/falkorchat/storefront_api.py"],
     "a52ba81a97ade6b58f76e1469662bc6a"),
    ("GV-3", ["symbol", "python", "graphmind-ai-lab", QNAME_GV3],
     "a6c2648709f43a0bad2211a1b8bbcd52"),
    ("GV-4", ["symbol", "  python\t", "graphmind-ai-lab ", "\n" + QNAME_GV3 + "  "],
     "a6c2648709f43a0bad2211a1b8bbcd52"),
    ("GV-5", ["symbol", "python", "graphmind-ai-lab", "pkg/café.py::café"],
     "e600f3985246448b452893a572028ae9"),
    ("GV-6", ["symbol", "python", "graphmind-ai-lab", "pkg/café.py::café"],
     "e600f3985246448b452893a572028ae9"),
    ("GV-7", ["symbol", "java", "fx-lib",
              "src/main/java/com/acme/fxlib/Pricing.java::Pricing.quote", "(int,String)"],
     "6b2ed6dbf5678679437067c6d3b079a8"),
    ("GV-8", ["symbol", "java", "fx-lib",
              "src/main/java/com/acme/fxlib/Pricing.java::Pricing.quote",
              "(int,String,Currency)"],
     "7543f67cff9b93877688af1475d91d3c"),
    ("GV-9", ["contractPoint", "http", "GET /shop/api/health"],
     "a359de518acc3d443ff5efa5597f458f"),
    ("GV-10", ["contractPoint", "http", "GET /" + "a" * 5000],
     "329a43a0d3c0223e5a0deadb23275afa"),
    ("GV-11", ["symbol", "javascript", "graphmind-ai-lab",
               "salesperson/src/api/client.ts::apiFetch"],
     "3e0e2e00a2cea8e99fd32ff6c2b1082e"),
    ("GV-12", ["callsite", "3e0e2e00a2cea8e99fd32ff6c2b1082e", "fetch", 0],
     "41d8c78a0fa5b76642dc74e73958888a"),
    ("GV-13", ["point", "3e0e2e00a2cea8e99fd32ff6c2b1082e", "param", 0],
     "3584bd914767bc6ec2cd94a767006185"),
    ("GV-14s", ["symbol", "javascript", "graphmind-ai-lab",
                "salesperson/src/api/endpoints.ts::getHealth"],
     SYM_GET_HEALTH),
    ("GV-14", ["callsite", SYM_GET_HEALTH, "apiFetch", 0],
     "ac74017feb936c98cb9fd4bfbe5737e8"),
    ("GV-15", ["symbol", "c", "fw", "src/util.c::crc8"],
     "493fb56c61dd5e583f5f5221a0e792fa"),
    ("GV-16", ["symbol", "kotlin", "fx-svc-kt", "src/main/kotlin/acme/Orders.kt::slug",
               "String.()"],
     "2f7967620eecf347b542cf76b466c3f5"),
    ("GV-17s", ["symbol", "python", "graphmind-ai-lab",
                "falkor-chat/server/falkorchat/storefront_api.py::StorefrontHTTPError.__init__"],
     SYM_HTTPERR_INIT),
    ("GV-17", ["point", SYM_HTTPERR_INIT, "param", -1],
     "73397835753cc740b7d9ac82faa84ec2"),
    ("GV-18", ["point", SYM_HTTPERR_INIT, "param", " -1 "],
     "0eb4a24f0b98b6081be51a3fbf070305"),
    ("GV-19", ["name", "javascript", "fetch"],
     "541e72d540d11b7c8647528766e38e77"),
    ("GV-20S", ["symbol", "python", "graphmind-ai-lab",
                "falkor-chat/server/falkorchat/tools.py::McpToolClient.close"],
     SYM_TOOLS_CLOSE),
    ("GV-20c", ["callsite", SYM_TOOLS_CLOSE, "call_soon_threadsafe", 1],
     CS_CLOSE_SOON_1),
    ("GV-20", ["fieldAccess", CS_CLOSE_SOON_1, "recv", "read",
               "falkor-chat/server/falkorchat/tools.py::McpToolClient", "_loop"],
     "468fc6cbf90b5f0160ad35d9474afd52"),
    ("GV-21", ["fieldAccess", CS_CLOSE_SOON_1, "a0", "read",
               "falkor-chat/server/falkorchat/tools.py::McpToolClient", "_loop"],
     "17e9de61ca6bca69fdd5738fbef102a5"),
    ("GV-22", ["symbol", "go", "fx-svc-go", "svc/store.go::Store.Get"],
     "afa8ae88b74f3b469133792e4c28b650"),
]


@pytest.mark.parametrize(("vector", "tup", "expected"), GOLDEN, ids=[g[0] for g in GOLDEN])
def test_golden_vector_reproduces_via_hash(vector, tup, expected):
    assert ids.tuple_id(tup) == expected


# The same vectors through the §3.2 constructors (the tag is the constructor's job, not the
# caller's). Each lambda gets the tuple's elements after the tag.
CONSTRUCTED = [
    ("GV-1", lambda: ids.unit_id("graphmind-ai-lab", "falkor-chat/server", "python"),
     "0fb899742d4d3687495c30357ca16253"),
    ("GV-2", lambda: ids.file_id("graphmind-ai-lab",
                                 "falkor-chat/server/falkorchat/storefront_api.py"),
     "a52ba81a97ade6b58f76e1469662bc6a"),
    ("GV-3", lambda: ids.symbol_id("python", "graphmind-ai-lab", QNAME_GV3),
     "a6c2648709f43a0bad2211a1b8bbcd52"),
    ("GV-4", lambda: ids.symbol_id("  python\t", "graphmind-ai-lab ", "\n" + QNAME_GV3 + "  "),
     "a6c2648709f43a0bad2211a1b8bbcd52"),
    ("GV-6", lambda: ids.symbol_id("python", "graphmind-ai-lab", "pkg/café.py::café"),
     "e600f3985246448b452893a572028ae9"),
    ("GV-7", lambda: ids.symbol_id("java", "fx-lib",
                                   "src/main/java/com/acme/fxlib/Pricing.java::Pricing.quote",
                                   signature="(int,String)"),
     "6b2ed6dbf5678679437067c6d3b079a8"),
    ("GV-9", lambda: ids.contract_point_id("http", "GET /shop/api/health"),
     "a359de518acc3d443ff5efa5597f458f"),
    ("GV-12", lambda: ids.callsite_id("3e0e2e00a2cea8e99fd32ff6c2b1082e", "fetch", 0),
     "41d8c78a0fa5b76642dc74e73958888a"),
    ("GV-13", lambda: ids.point_id("3e0e2e00a2cea8e99fd32ff6c2b1082e", "param", 0),
     "3584bd914767bc6ec2cd94a767006185"),
    ("GV-14", lambda: ids.callsite_id(
        ids.symbol_id("javascript", "graphmind-ai-lab",
                      "salesperson/src/api/endpoints.ts::getHealth"), "apiFetch", 0),
     "ac74017feb936c98cb9fd4bfbe5737e8"),
    ("GV-15", lambda: ids.symbol_id("c", "fw", "src/util.c::crc8"),
     "493fb56c61dd5e583f5f5221a0e792fa"),
    ("GV-16", lambda: ids.symbol_id("kotlin", "fx-svc-kt", "src/main/kotlin/acme/Orders.kt::slug",
                                    signature="String.()"),
     "2f7967620eecf347b542cf76b466c3f5"),
    ("GV-17", lambda: ids.point_id(
        ids.symbol_id("python", "graphmind-ai-lab",
                      "falkor-chat/server/falkorchat/storefront_api.py::StorefrontHTTPError.__init__"),
        "param", -1),
     "73397835753cc740b7d9ac82faa84ec2"),
    ("GV-19", lambda: ids.name_key("javascript", "fetch"),
     "541e72d540d11b7c8647528766e38e77"),
    ("GV-20", lambda: ids.field_access_id(
        ids.callsite_id(ids.symbol_id("python", "graphmind-ai-lab",
                                      "falkor-chat/server/falkorchat/tools.py::McpToolClient.close"),
                        "call_soon_threadsafe", 1),
        "recv", "read", "falkor-chat/server/falkorchat/tools.py::McpToolClient", "_loop"),
     "468fc6cbf90b5f0160ad35d9474afd52"),
    ("GV-21", lambda: ids.field_access_id(
        CS_CLOSE_SOON_1, "a0", "read", "falkor-chat/server/falkorchat/tools.py::McpToolClient",
        "_loop"),
     "17e9de61ca6bca69fdd5738fbef102a5"),
    ("GV-22", lambda: ids.symbol_id("go", "fx-svc-go", "svc/store.go::Store.Get"),
     "afa8ae88b74f3b469133792e4c28b650"),
]


@pytest.mark.parametrize(("vector", "build", "expected"), CONSTRUCTED,
                         ids=[c[0] for c in CONSTRUCTED])
def test_golden_vector_reproduces_via_constructor(vector, build, expected):
    assert build() == expected


# --- §3.1 canonical JSON, pinned byte-for-byte -----------------------------------------

def test_canonical_bytes_are_compact_utf8_with_literal_non_ascii():
    tup = ["symbol", "python", "graphmind-ai-lab", "pkg/café.py::café"]
    assert ids.canonical_bytes(tup) == (
        '["symbol","python","graphmind-ai-lab","pkg/café.py::café"]'.encode("utf-8")
    )


def test_gv10_canonical_form_is_5032_bytes():
    assert len(ids.canonical_bytes(["contractPoint", "http", "GET /" + "a" * 5000])) == 5032


def test_integers_are_written_as_json_numbers_not_strings():
    assert ids.canonical_bytes(["point", "s", "param", -1]) == b'["point","s","param",-1]'


def test_nested_arrays_are_normalized_recursively():
    assert ids.canonical_bytes(["x", [" a ", ["café\n", 2]]]) == (
        '["x",["a",["café",2]]]'.encode("utf-8")
    )


# --- §9 ids.py unit tests (P0 plan §9, items 2–6) ---------------------------------------

def test_leading_and_trailing_whitespace_is_stripped_by_str_isspace():
    # U+2003 EM SPACE and U+001F UNIT SEPARATOR are both `str.isspace()` in Python 3.12.
    assert ids.tuple_id(["name", "python", " fetch\x1f"]) == ids.tuple_id(
        ["name", "python", "fetch"])


def test_inner_whitespace_is_kept():
    assert ids.tuple_id(["name", "python", "a b"]) != ids.tuple_id(["name", "python", "ab"])


def test_nfc_and_nfd_forms_give_the_same_id():
    nfc = ids.symbol_id("python", "r", "m.py::Ångström")
    nfd = ids.symbol_id("python", "r", "m.py::Ångström")
    assert nfc == nfd


def test_edge_whitespace_and_decomposed_form_normalize_together():
    assert ids.tuple_id(["name", "python", " é "]) == ids.tuple_id(
        ["name", "python", "é"])


def test_a_10kb_qname_gives_a_32_hex_id():
    qname = "big.py::" + "x" * 10_240
    assert re.fullmatch(r"[0-9a-f]{32}", ids.symbol_id("python", "r", qname))


def test_every_id_is_32_lowercase_hex():
    assert re.fullmatch(r"[0-9a-f]{32}", ids.tuple_id(["name", "python", "FETCH"]))


def test_changing_tuple_order_changes_the_id():
    assert ids.tuple_id(["unit", "r", "p", "python"]) != ids.tuple_id(
        ["unit", "p", "r", "python"])


@pytest.mark.parametrize("lang", ["java", "kotlin", "csharp", "cpp", " java "])
def test_signature_is_accepted_for_overloading_languages(lang):
    with_sig = ids.symbol_id(lang, "r", "A.x::A.f", signature="(int)")
    without = ids.symbol_id(lang, "r", "A.x::A.f")
    assert with_sig != without  # 5 elements vs 4 (types carry none, §4.3)


@pytest.mark.parametrize("lang", ["python", "javascript", "go", "c", "shell", "markdown"])
def test_signature_is_rejected_for_other_languages(lang):
    with pytest.raises(ValueError, match="signature"):
        ids.symbol_id(lang, "r", "m::f", signature="(int)")


def test_optional_signature_is_omitted_not_null():
    assert ids.symbol_id("java", "r", "A.java::A") == ids.tuple_id(
        ["symbol", "java", "r", "A.java::A"])


@pytest.mark.parametrize("bad", [None, True, False, 1.0])
def test_null_bool_and_float_never_appear_in_a_tuple(bad):
    with pytest.raises(TypeError):
        ids.tuple_id(["point", "s", "param", bad])


def test_objects_never_appear_in_a_tuple():
    # §3.1: tuples have no objects, so `sort_keys` can never be observed through a valid id.
    with pytest.raises(TypeError):
        ids.tuple_id(["x", {"b": 1, "a": 2}])


@pytest.mark.parametrize("kind", ["param", "ret", "cond"])
def test_point_kinds_are_param_ret_cond(kind):
    assert ids.point_id("s", kind, 0) == ids.tuple_id(["point", "s", kind, 0])


def test_unknown_point_kind_is_rejected():
    with pytest.raises(ValueError, match="kind"):
        ids.point_id("s", "callsite", 0)


def test_callsite_and_param_point_with_equal_payload_differ_by_tag():
    # BF1: the record-type tag keeps (symbolId, "param", 0) apart across record types.
    assert ids.callsite_id("s", "param", 0) != ids.point_id("s", "param", 0)

# --- §3.2 / §5 constructors: the spec table ---------------------------------------------
# Built from the spec's record list — §3.2's rows plus every §8.3 record type — never from
# `dir(ids)`. Per constructor: its tag and one (type, sample) per position, in spec order.
# Samples are distinct per position so a swap shows, and use the spec's own tokens.
S, I = str, int
SPEC_CONSTRUCTORS = {
    "unit_id": ("unit", [(S, "graphmind-ai-lab"), (S, "falkor-chat/server"), (S, "python")]),
    "file_id": ("file", [(S, "graphmind-ai-lab"), (S, "pkg/m.py")]),
    "symbol_id": ("symbol", [(S, "java"), (S, "fx-lib"), (S, "A.java::A.f"), (S, "(int)")]),
    "callsite_id": ("callsite", [(S, "sym"), (S, "fetch"), (I, 2)]),
    "point_id": ("point", [(S, "sym"), (S, "ret"), (I, 1)]),
    "constant_id": ("constant", [(S, "python"), (S, "r"), (S, "m.py::X")]),
    "config_entry_id": ("configEntry", [(S, "r"), (S, "p.yaml"), (S, "a.b")]),
    "launch_fact_id": ("launchFact", [(S, "r"), (S, "p.sh"), (S, "command"), (I, 2)]),
    "schema_entry_id": ("schemaEntry", [(S, "r"), (S, "s.sql"), (S, "t")]),
    "dep_id": ("dependency", [(S, "PyPI"), (S, "falkordb")]),
    "dependency_record_id": ("dependency",
                             [(S, "PyPI"), (S, "falkordb"), (S, "pyproject.toml"), (I, 12)]),
    "name_key": ("name", [(S, "javascript"), (S, "fetch")]),
    "contract_point_id": ("contractPoint", [(S, "http"), (S, "GET /x")]),
    "link_id": ("link", [(S, "http"), (S, "GET /x"), (S, "P"), (S, "C")]),
    "sighting_id": ("secretSighting", [(S, "r"), (S, "p"), (S, "c0ffee"), (S, "fp")]),
    "value_set_id": ("valueSet", [(S, "cs"), (S, "a0")]),
    "data_id": ("data", [(S, "u"), (S, "d"), (S, "a0"), (S, "ret")]),
    "controlled_by_id": ("controlledBy", [(S, "pt"), (S, "cd")]),
    "decorator_id": ("decorator", [(S, "t"), (I, 1)]),
    "field_access_id": ("fieldAccess", [(S, "pt"), (S, "recv"), (S, "write"), (S, "?"), (S, "f")]),
    "import_id": ("import", [(S, "f"), (I, 3)]),
    "test_marker_id": ("testMarker", [(S, "s"), (S, "pytest")]),
}

# Every tuple shape the spec writes, as (tag, length including the tag), transcribed from the
# §3.2 table and each §5 "Id tuple" line — never from the module. Keyed by shape, not tag, so a
# second shape under an existing tag (§5.15's record id beside `depId`) cannot go missing.
# param/ret/cond are `point` tuples (§5.4–§5.5).
SPEC_SHAPES = {
    ("unit", 4), ("file", 3),
    ("symbol", 4), ("symbol", 5),  # §3.2: signature 5th only for java/kotlin/csharp/cpp (§4.3)
    ("callsite", 4), ("point", 4), ("constant", 4), ("configEntry", 4), ("launchFact", 5),
    ("schemaEntry", 4),
    ("dependency", 3),  # §3.2 depId
    ("dependency", 5),  # §5.15 record id
    ("name", 3), ("contractPoint", 3), ("link", 5), ("secretSighting", 5),
    ("valueSet", 3), ("data", 5), ("controlledBy", 3), ("decorator", 3), ("fieldAccess", 6),
    ("import", 3), ("testMarker", 3),
}
#: Constructors with an optional trailing position, and the shape they give without it.
OPTIONAL_TAIL_SHAPES = {"symbol_id": ("symbol", 4)}


def _samples(name):
    return [sample for _, sample in SPEC_CONSTRUCTORS[name][1]]


@pytest.mark.parametrize("name", sorted(SPEC_CONSTRUCTORS))
def test_constructor_hashes_the_spec_tuple(name):
    tag, _ = SPEC_CONSTRUCTORS[name]
    assert getattr(ids, name)(*_samples(name)) == ids.tuple_id([tag, *_samples(name)])


WRONG = {str: 7, int: "0"}
POSITIONS = [(name, pos) for name in sorted(SPEC_CONSTRUCTORS)
             for pos in range(len(SPEC_CONSTRUCTORS[name][1]))]


@pytest.mark.parametrize(("name", "pos"), POSITIONS, ids=[f"{n}[{p}]" for n, p in POSITIONS])
def test_constructor_rejects_wrong_type_at_every_position(name, pos):
    args = _samples(name)
    args[pos] = WRONG[SPEC_CONSTRUCTORS[name][1][pos][0]]
    with pytest.raises(TypeError):
        getattr(ids, name)(*args)


@pytest.mark.parametrize("name", sorted(n for n in SPEC_CONSTRUCTORS
                                        if I in [t for t, _ in SPEC_CONSTRUCTORS[n][1]]))
def test_integer_positions_reject_bool(name):
    args = _samples(name)
    args[[t for t, _ in SPEC_CONSTRUCTORS[name][1]].index(I)] = True
    with pytest.raises(TypeError):
        getattr(ids, name)(*args)


def test_m1_review_cases_raise_instead_of_hashing():
    with pytest.raises(TypeError):
        ids.callsite_id("3e0e2e00a2cea8e99fd32ff6c2b1082e", "fetch", "0")
    with pytest.raises(TypeError):
        ids.point_id(SYM_HTTPERR_INIT, "param", " -1 ")  # GV-18 stays a raw-tuple vector
    with pytest.raises(TypeError):
        ids.unit_id(1, "p", "python")


def test_table_covers_every_public_constructor():
    public = {n for n, v in vars(ids).items()
              if callable(v) and not n.startswith("_")
              and (n.endswith("_id") or n == "name_key") and n != "tuple_id"}
    assert set(SPEC_CONSTRUCTORS) == public


def test_table_covers_every_spec_tuple_shape():
    shapes = {(tag, 1 + len(positions)) for tag, positions in SPEC_CONSTRUCTORS.values()}
    assert shapes | set(OPTIONAL_TAIL_SHAPES.values()) == SPEC_SHAPES


@pytest.mark.parametrize("pos", range(3))
def test_symbol_id_without_signature_rejects_wrong_type_at_every_position(pos):
    # The 4-element path: the table's row always passes a signature, whose branch would
    # otherwise catch a wrong-typed `lang` by accident (review S09 Pass 2, P2-2).
    args = ["python", "graphmind-ai-lab", "m.py::f"]
    args[pos] = 7
    with pytest.raises(TypeError):
        ids.symbol_id(*args)


def test_symbol_id_without_signature_hashes_the_4_element_tuple():
    assert ids.symbol_id("java", "fx-lib", "A.java::A") == ids.tuple_id(
        ["symbol", "java", "fx-lib", "A.java::A"])


def test_dependency_record_id_is_the_5_element_section_5_15_tuple():
    assert ids.dependency_record_id("PyPI", "falkordb", "pyproject.toml", 12) == ids.tuple_id(
        ["dependency", "PyPI", "falkordb", "pyproject.toml", 12])


def test_one_manifest_declaring_a_package_twice_gives_two_record_ids_one_dep_id():
    runtime = ids.dependency_record_id("PyPI", "falkordb", "pyproject.toml", 12)
    dev = ids.dependency_record_id("PyPI", "falkordb", "pyproject.toml", 20)
    assert runtime != dev
    assert ids.dep_id("PyPI", "falkordb") not in {runtime, dev}


# --- m1 / m3 / m5 -------------------------------------------------------------------------

def test_test_marker_id_is_not_collected_by_pytest():
    assert ids.test_marker_id.__test__ is False


@pytest.mark.parametrize("bad", ["unit", {"b": 1, "a": 2}, []], ids=["str", "dict", "empty"])
def test_tuple_id_rejects_a_non_list_or_empty_tuple(bad):
    with pytest.raises((TypeError, ValueError)):
        ids.tuple_id(bad)


def test_tuple_id_rejects_a_non_string_tag():
    with pytest.raises(TypeError):
        ids.tuple_id([1, "x"])


def test_tuple_id_accepts_a_python_tuple():
    assert ids.tuple_id(("name", "javascript", "fetch")) == "541e72d540d11b7c8647528766e38e77"


@pytest.mark.parametrize("sig", ["", "   ", "\t\n"])
def test_empty_or_whitespace_signature_is_rejected(sig):
    with pytest.raises(ValueError, match="signature"):
        ids.symbol_id("java", "r", "A.java::A.f", signature=sig)


def test_signature_of_wrong_type_is_rejected():
    with pytest.raises(TypeError):
        ids.symbol_id("java", "r", "A.java::A.f", signature=1)


def test_signature_type_is_checked_before_the_language_rule():
    # A wrong-typed signature is a TypeError in any language, not a language-rule ValueError.
    with pytest.raises(TypeError):
        ids.symbol_id("python", "r", "m.py::f", signature=1)
