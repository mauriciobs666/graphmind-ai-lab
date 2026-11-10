"""P1-16 `ccg.packrt.extract` (value alternatives, expressions) and `ccg.packrt.httpkey` (the ``http`` key steps).

Plan: `docs/plans/connected-code-graph-p1-linking.md` § P1-16 behaviours (1), (2); rule-pack spec §1.2, §1.4.
"""
import pytest

from packrt_helpers import call, constant, dec, file_, sym, vs
from ccg import keynorm
from ccg.packrt import httpkey
from ccg.packrt.extract import Corpus, Ctx, UnsupportedExpression, eval_expr, slot_mode
from ccg.packrt.httpkey import UNKNOWN


def corpus(*records):
    return Corpus([list(records)])


# ---- alternatives

def test_const_ref_concat_and_the_unresolved_pieces():
    c = corpus(constant("m::BASE", "/api"), constant("m::A", {"ref": "m::B"}), constant("m::B", {"ref": "m::A"}))
    assert c.alternatives({"const": 7}) == ["7"]
    assert c.alternatives({"ref": "m::BASE"}) == ["/api"]
    assert c.alternatives({"concat": [{"ref": "m::BASE"}, {"const": "/x"}]}) == ["/api/x"]
    assert c.alternatives({"unknown": "point-derived"}) == [UNKNOWN]
    assert c.alternatives({"importRef": "q"}) == [UNKNOWN]
    assert c.alternatives({"ref": "m::missing"}) == [UNKNOWN]
    assert c.alternatives({"ref": "m::A"}) == [UNKNOWN]                    # a cycle stops, it does not loop
    assert c.alternatives({"concat": [{"const": "/a"}, {"unknown": "x"}]}) == ["/a" + UNKNOWN]


def test_a_value_set_is_the_union_of_its_elements_and_a_product_is_capped_at_64():
    c = corpus(constant("m::A", "1", "2"))
    assert c.value_alternatives([{"const": "z"}, {"ref": "m::A"}]) == ["1", "2", "z"]
    wide = {"concat": [{"ref": "m::W"}, {"ref": "m::W"}]}
    c2 = corpus(constant("m::W", *[str(i) for i in range(9)]))
    got = c2.alternatives(wide)
    assert len(got) == 65 and got[-1] == UNKNOWN                            # 64 products, then the unresolved marker


def test_arg_values_read_a_decorators_args_and_a_callsites_value_sets():
    c = corpus(file_("f1", "a.py"), sym("h", "h"), call("c", "h", "g"), vs("c", "a0", "/x"), vs("c", "kw.p", "/p"),
               dec("d", "h", "r.get", ("a0", "/d")))
    d = c.decorators[0]
    assert c.arg_alternatives(d, "a0") == ["/d"] and c.arg_alternatives(d, "a1") is None
    assert c.arg_alternatives(c.callsites["c"], "kw.p") == ["/p"] and c.arg_alternatives(c.callsites["c"], "a1") is None


def test_flow_into_a_prefix_of_the_slot_carries_the_remainder():
    from packrt_helpers import data
    c = corpus(data("u", "d", "a1", "p0.opts"))
    assert [(d["def"], p) for d, p in c.flows_into("u", "a1.method")] == [("d", "p0.opts.method")]
    assert list(c.flows_into("u", "a10")) == []                              # `a1` is not a prefix of `a10`
    assert list(c.flows_into("other", "a1")) == []


# ---- expressions

def ctx(c, record, hit=None, target=None, lifted=None):
    return Ctx(corpus=c, record=record, hit=hit, target=target, lifted=lifted)


def test_firstof_takes_the_first_non_empty_expression_left_to_right():
    c = corpus(file_("f1", "a.py"), sym("h", "h"), dec("d", "h", "mcp.tool", ("a0", "second")))
    d, h = c.decorators[0], c.symbols["h"]
    e = {"firstOf": [{"arg": "kw.name"}, {"arg": "a0"}, {"target": "name"}]}
    assert eval_expr(e, ctx(c, d, target=h)) == ["second"]
    d2 = dec("d2", "h", "mcp.tool", ("kw.name", "named"), ("a0", "second"))
    assert eval_expr(e, ctx(c, d2, target=h)) == ["named"]
    assert eval_expr(e, ctx(c, dec("d3", "h", "mcp.tool"), target=h)) == ["h"]       # falls through to the function name


def test_an_unresolved_first_alternative_is_still_non_empty_so_firstof_stops_there():
    c = corpus(file_("f1", "a.py"), sym("h", "h"), dec("d", "h", "mcp.tool", ("a0", {"unknown": "point-derived"})))
    e = {"firstOf": [{"arg": "a0"}, {"target": "name"}]}
    assert eval_expr(e, ctx(c, c.decorators[0], target=c.symbols["h"])) == [UNKNOWN]


def test_group_literal_target_qname_and_default():
    from ccg.packrt.match import text_hit
    c = corpus(file_("f1", "a.py"), sym("h", "h"), dec("d", "h", "router.post"))
    d, h = c.decorators[0], c.symbols["h"]
    hit = text_hit({"regex": r"\w+\.(get|post)"}, "router.post")
    assert eval_expr({"group": 1, "upper": True}, ctx(c, d, hit)) == ["POST"]
    assert eval_expr({"group": 1}, ctx(c, d, hit)) == ["post"]
    assert eval_expr({"literal": "*"}, ctx(c, d)) == ["*"]
    assert eval_expr({"target": "qname"}, ctx(c, d, target=h)) == ["a.py::h"]
    assert eval_expr({"arg": "kw.prefix", "default": ""}, ctx(c, d)) == [""]         # absent: the default
    assert eval_expr({"arg": "kw.prefix"}, ctx(c, d)) == []


def test_lifted_expression_reads_the_resolved_roles_of_the_node():
    c = corpus(file_("f1", "a.py"), sym("h", "h"), call("c", "h", "g"))
    lifted = {"url": ["/u"], "verb": ["GET", "POST"]}
    assert eval_expr({"lifted": "verb"}, ctx(c, c.callsites["c"], lifted=lifted)) == ["GET", "POST"]


@pytest.mark.parametrize("expr", [{"config": "keyPath"}, {"field": "x"}, {"yield": "c"}])
def test_expression_forms_outside_the_p1_16_runtime_are_refused_not_read_as_empty(expr):
    c = corpus(call("c", "h", "g"))
    with pytest.raises(UnsupportedExpression):
        eval_expr(expr, ctx(c, c.callsites["c"]))


def test_a_slots_mode_is_its_one_nested_on_unknown_else_the_default():
    assert slot_mode({"arg": "a0"}, "drop") == "drop"
    assert slot_mode({"arg": "a0", "onUnknown": "partial"}, "drop") == "partial"
    assert slot_mode({"firstOf": [{"arg": "a0", "onUnknown": "runtime-built"}, {"target": "name"}]}, "drop") == "runtime-built"


# ---- http key steps

@pytest.mark.parametrize("raw,form,want", [
    ("GET /a/%2E%2E/b", "provider", "GET /b"),                  # F-13: percent-decode before dot-segment resolution
    ("get /x/{id}", "provider", "GET /x/{}"),
    ("GET /x/{path:path}", "provider", "GET /x/{**}"),
    ("/x", "provider", "* /x"),                                  # no verb word: any-method
    ("GET /a//b/", "provider", "GET /a/b/"),
    ("GET http://h:80/a/b?q=1#f", "consumer", "GET /a/b"),      # step 2 only on the consumer form
    ("GET /a/" + UNKNOWN, "consumer", "GET /a/{?}"),
    ("GET /v/%2f", "provider", "GET /v/%2F"),                    # a reserved escape stays, uppercased
])
def test_http_key_steps(raw, form, want):
    assert httpkey.normalize_key(raw, form) == want


def test_step_8a_is_the_one_in_ccg_keynorm_and_a_token_shaped_segment_is_masked():
    assert httpkey.mask_http_segments is keynorm.mask_http_segments
    assert httpkey.normalize_key("GET /u/550e8400-e29b-41d4-a716-446655440000", "provider") == "GET /u/{#}"
    assert httpkey.normalize_key("GET /u/550e8400-e29b-41d4-a716-446655440000", "consumer") == "GET /u/{?}"


def test_rfc_normalizer_vectors_of_the_spec_section_1_3():
    assert httpkey.normalize_path_rfc("/mcp/%2E%2E/admin") == "/admin"
    assert httpkey.normalize_path_rfc("/mcp/./x") == "/mcp/x"
    assert httpkey.normalize_path_rfc("/a//b") == "/a/b"
    assert httpkey.normalize_path_rfc("/a/b/.") == "/a/b/" and httpkey.normalize_path_rfc("/a/b/..") == "/a/"   # a trailing dot segment leaves a `/`
    assert httpkey.normalize_path_rfc("/a/b/./c") == "/a/b/c"
    with pytest.raises(ValueError):
        httpkey.normalize_key("GET /x", "neither")
