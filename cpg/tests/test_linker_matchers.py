"""P1-17 dispatch 1: the linker's key matchers (``ccg.linker.matchers``; rule-pack spec §1.3, §1.6 ``link``).

Keys here are already ``keyNorm`` strings, as the pack runtime emits them: a matcher compares, it never normalizes a
writer's key again (bundle spec3 §6 step 6: a key is normalized once). ``http_key`` is the one entry for a raw URL or path
the linker derives itself, and it follows the RFC 3986 order (F-13).
"""
import pytest

from ccg.linker import matchers as m


# -- F-13: percent-decode before dot-segment resolution --------------------------------------------
def test_http_key_decodes_before_resolving_dot_segments():
    assert m.http_key("GET /a/%2E%2E/b") == "GET /b"          # the old order would keep %2E%2E as a literal segment


def test_http_key_consumer_form_marks_unknown_and_takes_the_url_path():
    assert m.http_key("GET http://localhost:8200/mcp/x?q=1", "consumer") == "GET /mcp/x"


# -- exact -----------------------------------------------------------------------------------------
def test_exact_equal_strings_match_with_matcher_exact():
    assert m.match_keys("mcp_tool", "exact", "send_message", "send_message") == m.Match("exact", False)


def test_exact_different_strings_do_not_match():
    assert m.match_keys("mcp_tool", "exact", "send_message", "send_messages") is None


# -- path-template ---------------------------------------------------------------------------------
@pytest.mark.parametrize("provider,consumer,matcher,partial", [
    ("GET /api/health", "GET /api/health", "exact", False),
    ("GET /api/items/{}", "GET /api/items/7", "path-template", False),
    ("GET /files/{**}", "GET /files/a/b/c.txt", "path-template", False),
    ("* /x", "GET /x", "path-template", False),      # an any-method registration serves every verb
    ("GET /api/items/{}", "GET /api/items/{?}", "path-template", True),   # unknown consumer segment: partial
    ("GET /api/items/me", "GET /api/items/{?}", "path-template", True),
    ("GET /api/{#}/x", "GET /api/a/x", "path-template", True),           # masked provider literal: not provably equal
    ("GET /x", "* /x", "path-template", True),                            # unknown consumer verb
    ("GET /files/a/b", "GET /files/{**}", "path-template", True),         # a consumer catch-all: partial
    ("GET /x/", "GET /x", "path-template", True),                         # a trailing slash relates as possible only
    ("GET /x", "GET /x/", "path-template", True),
])
def test_path_template_matches(provider, consumer, matcher, partial):
    assert m.match_keys("http", "path-template", provider, consumer) == m.Match(matcher, partial)


@pytest.mark.parametrize("provider,consumer", [
    ("GET /api/items/{}", "GET /api/items"),            # {} is one non-empty segment
    ("GET /api/items/{}", "GET /api/items/1/2"),
    ("GET /files/{**}", "GET /files/"),                 # {**} is one or more
    ("POST /x", "GET /x"),
    ("GET /a/b", "GET /a/c"),
    ("GET /a/{}/c", "GET /a/b/d"),
    ("GET /api/items/{}", "GET /api/items/"),            # an empty segment is not a value
    ("GET /files/", "GET /files/{**}"),                  # a consumer catch-all needs something to stand for
    ("GET /a/{**}/b", "GET /a/x/b"),                     # {**} is accepted only in the last provider segment (F5)
])
def test_path_template_rejects(provider, consumer):
    assert m.match_keys("http", "path-template", provider, consumer) is None


def test_path_template_never_renormalizes_a_writer_key():
    """``%2%45`` is a key that decodes once to ``%2E``; a second pass would turn it into ``..`` and change the key."""
    key = "GET /a/%2E%2E/b"        # what one normalization of GET /a/%2%45%2%45/b yields (spec3 §6 step 6)
    assert m.match_keys("http", "path-template", key, key) == m.Match("exact", False)
    assert m.match_keys("http", "path-template", key, "GET /b") is None


# -- glob / prefix / regex -------------------------------------------------------------------------
def test_glob_key_form_matches_a_literal_of_the_same_kind():
    got = m.match_keys("topic", "exact", "inhouse-bus:billing.invoice.paid", "inhouse-bus:billing.invoice.*",
                       consumer_form="glob")
    assert got == m.Match("glob", False)


def test_glob_star_does_not_cross_a_slash():
    assert m.match_keys("topic", "exact", "bus:a/b", "bus:*", consumer_form="glob") is None


def test_regex_key_form_applies_to_the_part_after_the_transport():
    got = m.match_keys("topic", "exact", "kafka:orders.created", r"kafka:^orders\..*", consumer_form="regex")
    assert got == m.Match("regex", False)


def test_regex_key_form_requires_the_same_transport():
    assert m.match_keys("topic", "exact", "amqp:orders.created", r"kafka:^orders\..*", consumer_form="regex") is None


def test_regex_is_a_whole_string_match():
    assert m.match_keys("topic", "exact", "kafka:orders.created.x", r"kafka:orders", consumer_form="regex") is None


def test_regex_outside_the_re2_subset_matches_nothing():
    assert m.match_keys("topic", "exact", "kafka:aa", r"kafka:(a)\1", consumer_form="regex") is None
    assert m.match_keys("topic", "exact", "kafka:a", r"kafka:(?=a)a", consumer_form="regex") is None


def test_a_malformed_regex_matches_nothing_and_does_not_raise():
    assert m.match_keys("topic", "exact", "kafka:a", "kafka:(", consumer_form="regex") is None


def test_provider_side_pattern_matches_a_literal_consumer():
    got = m.match_keys("topic", "exact", "bus:a.*", "bus:a.b", provider_form="glob")
    assert got == m.Match("glob", False)


def test_two_pattern_keys_are_never_matched_to_each_other():
    assert m.match_keys("topic", "exact", "bus:a.*", "bus:a.*", provider_form="glob", consumer_form="glob") is None


def test_pack_level_prefix_matcher_stops_at_a_slash_boundary():
    assert m.match_keys("custom", "prefix", "/mcp", "/mcp/x") == m.Match("prefix", False)
    assert m.match_keys("custom", "prefix", "/mcp", "/mcpx") is None


def test_pack_level_glob_and_regex_link_matchers_apply_to_the_provider_key_as_pattern():
    assert m.match_keys("custom", "glob", "a.*", "a.b") == m.Match("glob", False)
    assert m.match_keys("custom", "regex", "a\\..*", "a.b") == m.Match("regex", False)
    assert m.match_keys("custom", "glob", "a.*", "a.b") is not None and m.match_keys("custom", "glob", "a.*", "b.b") is None


def test_unknown_link_matcher_is_refused():
    with pytest.raises(ValueError):
        m.match_keys("http", "fuzzy", "GET /a", "GET /a")


# -- O-5: prefix-only path matching ----------------------------------------------------------------
@pytest.mark.parametrize("mount,url,ok", [
    ("/mcp", "/mcp", True),
    ("/mcp", "/mcp/", True),
    ("/mcp", "/mcp/anything/below", True),            # the inner path is never compared
    ("/mcp", "/mcpx", False),
    ("/mcp", "/mcp/../admin", False),                 # normalized once, RFC order
    ("/mcp", "/mcp/%2E%2E/admin", False),
    ("/mcp", "/mcp/./x", True),
    ("/mcp", "/other", False),
    ("/", "/anything", True),                         # a root mount takes every path
])
def test_mount_prefix_is_prefix_only(mount, url, ok):
    assert m.mount_prefix_matches(mount, url) is ok


def test_mount_prefix_does_not_have_an_exact_mode():
    """O-5 pins prefix-only: a consumer path under the mount matches although the unit's inner path may differ."""
    assert m.mount_prefix_matches("/mcp", "/mcp/never-registered")


# -- behaviour (14): regex keys are repo-derived text under a backtracking engine ------------------
def _regex(pattern, text, refused=None):
    return m.match_keys("topic", "exact", f"kafka:{text}", f"kafka:{pattern}", consumer_form="regex", refused=refused)


def test_a_catastrophic_pattern_returns_at_once_matches_nothing_and_is_reported():
    import time
    refused: set = set()
    t = time.perf_counter()
    assert _regex("(a+)+b", "a" * 40 + "c", refused) is None
    assert time.perf_counter() - t < 0.5 and refused == {"(a+)+b"}


@pytest.mark.parametrize("pattern", ["(a+)+$", "(a|b)*c", "(a|b)?c", "((a|b)c)*", "(a+?)+", "(?:a*)*", "a+b+c+d+e",
                                     "a{2,}b{2,}c{2,}d{2,}", "(?P<x>a+)*"])
def test_ambiguous_or_heavily_quantified_patterns_are_refused(pattern):
    refused: set = set()
    assert _regex(pattern, "aaaa", refused) is None and refused == {pattern}
    assert m.regex_refusal(pattern, "aaaa") is not None


@pytest.mark.parametrize("pattern,text", [(r"^orders\..*", "orders.created"), ("(ab)?c", "abc"), ("(?:ab)+", "abab"),
                                          ("a+b+c+", "aabbcc"), ("a+?b+?c+?", "abc"), ("[a+]+", "a+a"), ("[^]+*+*]", "a"), ("[]+*+*]", "]"), (r"\(a+\)+", "(aa)))"), ("a{2,3}", "aaa"), (r"a\+\+b", "a++b")])
def test_plain_patterns_still_match_and_are_not_reported(pattern, text):
    refused: set = set()
    assert _regex(pattern, text, refused) == m.Match("regex", False) and refused == set()


def test_a_text_longer_than_256_characters_is_refused():
    refused: set = set()
    assert _regex("a+", "a" * 257, refused) is None and refused == {"a+"}
    assert _regex("a+", "a" * 256) == m.Match("regex", False)


def test_exactly_three_unbounded_quantifiers_are_allowed_and_four_are_not():
    assert m.regex_refusal("a*b*c*", "x") is None
    assert m.regex_refusal("a*b*c*d*", "x") is not None
