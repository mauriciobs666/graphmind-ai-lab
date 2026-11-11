"""P1-17 dispatch 1: the linker's key matchers (``ccg.linker.matchers``; rule-pack spec §1.3, §1.6 ``link``).

Keys here are already ``keyNorm`` strings, as the pack runtime emits them: a matcher compares, it never normalizes a
writer's key again (bundle spec3 §6 step 6: a key is normalized once). ``http_key`` is the one entry for a raw URL or path
the linker derives itself, and it follows the RFC 3986 order (F-13).
"""
import re

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
                                          ("a+b+", "aabb"), ("a+?b+?", "ab"), ("[a+]+", "a+a"), ("[^]+*+*]", "a"), ("[]+*+*]", "]"), (r"\(a+\)+", "(aa)))"), ("a{2,3}", "aaa"), (r"a\+\+b", "a++b")])
def test_plain_patterns_still_match_and_are_not_reported(pattern, text):
    refused: set = set()
    assert _regex(pattern, text, refused) == m.Match("regex", False) and refused == set()


def test_a_text_longer_than_256_characters_is_refused():
    refused: set = set()
    assert _regex("a+", "a" * 257, refused) is None and refused == set()          # F12: the pattern is fine, the text is not
    assert _regex("a+", "a" * 256) == m.Match("regex", False)


def test_exactly_two_unbounded_quantifiers_are_allowed_and_three_are_not():
    assert m.regex_refusal("a*b*", "x") is None
    assert m.regex_refusal("a*b*c*", "x") is not None


# --- F10 / F12 (analyst Pass 2): every quantifier counts; the refusal cause is reported apart --------
ANALYST = r"a{1,3}(?:a)a{1,3}[ab]{0,5}a*[ab]?(?:a){2,}b?a+"


def test_f10_the_analysts_stacked_bounded_quantifier_pattern_is_refused_at_once():
    import time
    refused: set = set()
    t = time.perf_counter()
    assert _regex(ANALYST, "a" * 255 + "!", refused) is None
    assert time.perf_counter() - t < 0.3 and refused == {ANALYST}


def test_f10_bounded_quantifiers_count_toward_the_total_cap_which_is_four():
    # literal rows (SEC M1 / analyst N2): the expected cap is written here, never read from the constant under test
    assert m.regex_refusal("a?a?a?a?", "x") is None
    assert m.regex_refusal("a?a?a?a?a?", "x") is not None
    assert m.regex_refusal("a{1,2}a{1,2}a{1,2}a{1,2}", "x") is None
    assert m.regex_refusal("a{1,2}a{1,2}a{1,2}a{1,2}a{1,2}", "x") is not None
    assert m.regex_refusal("(?:ab){2}" * 5, "x") is not None


def test_the_unbounded_cap_is_two_not_three_and_a_bounded_quantifier_is_not_unbounded():
    assert m.regex_refusal("a*b+", "x") is None
    assert m.regex_refusal("a*b+c{2,}", "x") is not None


def test_f10_a_lazy_marker_is_part_of_its_quantifier_not_a_second_one():
    assert m.regex_refusal("a??b??c??d??", "x") is None and m.regex_refusal("a??b??c??d??e??", "x") is not None


def test_f10_a_seeded_fuzz_finds_no_admitted_pattern_slower_than_the_budget():
    import random, signal, time

    class Slow(Exception):
        pass

    def alarm(*_):
        raise Slow
    atoms = ["a", "b", "(?:a)", "(?:ab)", "[ab]", ".", "[a-c]"]
    quants = ["", "", "*", "+", "?", "{2}", "{1,3}", "{0,5}", "{2,}", "*?", "+?", "{1,2}?"]
    texts = ["a" * 255 + "!", "a" * 256, "ab" * 128, "a" * 200 + "b" * 55 + "!"]
    rng, old = random.Random(11), signal.signal(signal.SIGALRM, alarm)
    try:
        for _ in range(600):
            pattern = "".join(rng.choice(atoms) + rng.choice(quants) for _ in range(rng.randint(2, 14)))
            if m.regex_refusal(pattern, "x" * 256) is not None:
                continue
            for text in texts:
                signal.setitimer(signal.ITIMER_REAL, 0.5)
                t = time.perf_counter()
                try:
                    re.fullmatch(pattern, text)
                except Slow:
                    pytest.fail(f"admitted but over 0.5 s: {pattern!r}")
                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                assert time.perf_counter() - t < 0.3, (pattern, text)
    finally:
        signal.signal(signal.SIGALRM, old)


def test_f12_a_fine_pattern_with_a_too_long_text_is_not_counted_as_a_refused_pattern():
    refused, long_text = set(), set()
    got = m.match_keys("topic", "exact", "kafka:" + "a" * 300, "kafka:ab+", consumer_form="regex",
                       refused=refused, refused_text=long_text)
    assert got is None and refused == set() and long_text == {"a" * 300}


def test_f12_a_bad_pattern_is_counted_as_a_pattern_and_not_as_a_text():
    refused, long_text = set(), set()
    m.match_keys("topic", "exact", "kafka:aaaa", "kafka:(a+)+$", consumer_form="regex", refused=refused, refused_text=long_text)
    assert refused == {"(a+)+$"} and long_text == set()


@pytest.mark.parametrize("pattern", ["a+(?:a)*?[a-c]+a{0,5}a{1,2}?a", "a{0,5}(?:a)+(?:a)+[a-c]*?[a-c]{0,5}", "[ab]{0,5}[a-c]{2,}a+?a{0,5}a*?"])
def test_f10_patterns_the_fuzz_found_slow_under_a_cap_of_five_are_refused(pattern):
    assert m.regex_refusal(pattern, "x") is not None


# --- SEC B1 (P1-17 security review): the quantifier grammar `re` really accepts ---------------------
# `{,N}` and `{,}` are `{0,N}` and `*` to Python's parser; the scanner once read them as literal braces.
def _timed_none(pattern, text):
    import time
    refused: set = set()
    t = time.perf_counter()
    got = _regex(pattern, text, refused)
    return got, time.perf_counter() - t, refused


@pytest.mark.parametrize("pattern", ["(a+){,}b", "(a+){,99}b", "(a|aa){,99}b", "(a*){,}b", "(a+){,}?b", "(a*){0,}b"])
def test_b1_a_quantified_group_holding_a_quantifier_is_refused_in_every_brace_spelling(pattern):
    got, took, refused = _timed_none(pattern, "a" * 30 + "c")
    assert got is None and took < 0.3 and refused == {pattern}


@pytest.mark.parametrize("pattern", [r"\w{,255}" * 6 + "x", r"\w{,}" * 8, "a{,}a{,}a{,}", "a{3,}a{4,}a{5,}", "a{,5}" * 5])
def test_b1_stacked_brace_quantifiers_hit_the_caps_whatever_their_spelling(pattern):
    got, took, refused = _timed_none(pattern, "a" * 255)
    assert got is None and took < 0.3 and refused == {pattern}


def test_b1_the_control_spellings_are_refused_too():
    assert m.regex_refusal("(a+)+b", "x") is not None and m.regex_refusal("(a|a)*b", "x") is not None


@pytest.mark.parametrize("pattern,text", [("a{,5}", "aaa"), ("a{,}", "aaaa"), ("a{2,}", "aaa"), ("a{,5}b{,5}", "aab"), ("a{,}?b", "aab"),
                                          (r"a\{,5}", "a{,5}"), ("[a{,5}]+", "a"), ("a{x}", "a{x}"), ("a{}", "a{}"), ("a{ 5}", "a{ 5}"),
                                          ("a{,", "a{,")])
def test_b1_a_real_quantifier_is_counted_and_a_literal_brace_is_not_refused(pattern, text):
    refused: set = set()
    assert _regex(pattern, text, refused) == m.Match("regex", False) and refused == set()


def test_b1_a_brace_that_is_a_quantifier_counts_and_one_that_is_a_literal_does_not():
    assert m.regex_refusal("a{,5}" * 4, "x") is None and m.regex_refusal("a{,5}" * 5, "x") is not None      # quantifiers: 4 vs 5
    assert m.regex_refusal("a{x}" * 9, "x") is None and m.regex_refusal(r"a\{,5}" * 9, "x") is None       # literals: no count


@pytest.mark.parametrize("pattern", ["a{,}+", "a*+b", "a{2}+", "a?+"])
def test_b1_possessive_forms_are_refused(pattern):
    assert m.regex_refusal(pattern, "x") is not None


@pytest.mark.parametrize("pattern", ["(?=a)b", "a(?!b)", "(?<=a)b", "(?<!a)b", r"(a)\1", "(?P<n>a)(?P=n)", "(?>a)b", "(a)?(?(1)b|c)"])
def test_b1_constructs_outside_the_re2_subset_are_refused(pattern):
    assert m.regex_refusal(pattern, "x") is not None


def test_b1_a_pattern_the_parser_chokes_on_is_refused_not_raised():
    for pattern in ("a{99999999999}", "a{1,99999999999}", "(" * 300 + "a" + ")" * 300, "(a"):
        assert m.regex_refusal(pattern, "x") is not None


def test_b1_a_construct_the_guard_cannot_classify_is_refused_by_default(monkeypatch):
    monkeypatch.setattr(m, "_parse", lambda pattern: [("SOME_FUTURE_OPCODE", None)])
    assert m.regex_refusal("a", "x") is not None


def test_b1_with_no_parser_to_classify_with_every_pattern_is_refused(monkeypatch):
    monkeypatch.setattr(m, "_parse", None)
    assert m.regex_refusal("a", "x") is not None


# -- the second bypass of the same guard: unquantified alternation groups multiply (2^n) -------------
@pytest.mark.parametrize("pattern", ["(a|a)" * 30 + "b", "(?:a|a)" * 30 + "b", "(a|ab)" * 30 + "b"])
def test_b1_many_alternation_groups_in_a_row_are_refused_they_backtrack_exponentially(pattern):
    got, took, refused = _timed_none(pattern, "a" * 30 + "c")
    assert got is None and took < 0.3 and refused == {pattern}


def test_b1_a_few_alternations_still_match():
    assert _regex(r"orders\.(created|updated|deleted)", "orders.updated") == m.Match("regex", False)
    assert m.regex_refusal("(a|bc)(d|ef)(g|hi)", "x") is None             # 3 bars
    assert m.regex_refusal("(a|bc)(d|ef)(g|hi)(j|kl)", "x") is None       # 4 bars
    assert m.regex_refusal("(a|bc)(d|ef)(g|hi)(j|kl)(m|no)", "x") is not None


# -- a hard work budget across (pattern, literal) pairs ----------------------------------------------
def _budget_run(budget, pairs):
    refused: set = set()
    out = [m.match_keys("topic", "exact", f"kafka:{lit}", f"kafka:{pat}", consumer_form="regex", refused=refused, budget=budget)
           for pat, lit in pairs]
    return out, refused


def test_a_pair_budget_stops_evaluation_after_its_limit_and_counts_the_rest():
    budget = m.RegexBudget(3)
    pairs = [("a+", "a" * i) for i in range(1, 7)]
    got, _ = _budget_run(budget, pairs)
    assert [g is not None for g in got] == [True, True, True, False, False, False]
    assert budget.exhausted == 3


def test_a_repeated_pair_is_evaluated_once_and_costs_one():
    budget = m.RegexBudget(2)
    got, _ = _budget_run(budget, [("a+", "aa")] * 5 + [("b+", "bb")])
    assert all(g is not None for g in got) and budget.exhausted == 0


def test_a_refused_pattern_costs_no_budget():
    budget = m.RegexBudget(1)
    got, refused = _budget_run(budget, [("(a+)+b", "aa")] * 3 + [("a+", "aa")])
    assert got[-1] is not None and refused == {"(a+)+b"} and budget.exhausted == 0


def test_the_default_budget_is_finite_and_literal():
    assert m.MAX_REGEX_PAIRS == 800


# -- a wide bounded quantifier costs as much as `*` (n^4 admitted at 0.46 s per pair before) -----------
@pytest.mark.parametrize("pattern", [r"\w*\w*\w{0,255}\w{0,255}x", r"\w{,255}\w{,255}\w{,255}\w{,255}x", r"[\s\S]*[\s\S]*[\s\S]{,255}[\s\S]{,255}x",
                                     r"\w*\w{,255}\w{,255}\w{,255}x"])
def test_wide_bounded_quantifiers_count_as_unbounded_and_the_n4_patterns_are_refused_at_once(pattern):
    got, took, refused = _timed_none(pattern, "a" * 255 + "!")
    assert got is None and took < 0.3 and refused == {pattern}


def test_the_wide_cap_is_two_and_a_span_of_eight_is_narrow_nine_is_wide():
    assert m.regex_refusal("a{0,9}a{0,9}", "x") is None and m.regex_refusal("a{0,9}a{0,9}a{0,9}", "x") is not None
    assert m.regex_refusal("a{0,8}" * 4, "x") is None and m.regex_refusal("a{3,12}a{3,12}a{3,12}", "x") is not None
    assert m.regex_refusal("a{3,11}" * 4, "x") is None


# -- what the tree sees and the text scan cannot: a verbose-mode comment hides a paren from the text --------------
def test_b1_a_quantified_alternation_group_hidden_from_a_text_scan_by_a_verbose_comment_is_refused():
    pattern = "(?x)(a|aa#)\n)*b"
    assert re.fullmatch(pattern, "aab") and not re.fullmatch(pattern, "c")          # the engine reads (a|aa)*b
    got, took, refused = _timed_none(pattern, "a" * 30 + "c")
    assert got is None and took < 0.3 and refused == {pattern}


@pytest.mark.parametrize("raised", [OverflowError, RecursionError, re.error, ValueError])
def test_b1_a_parser_that_raises_anything_gives_a_refusal(monkeypatch, raised):
    def boom(pattern):
        raise raised("x")
    monkeypatch.setattr(m, "_parse", boom)
    assert m.regex_refusal("a", "x") is not None


def test_a_repeated_pair_is_matched_by_the_engine_once(monkeypatch):
    calls = []
    real = re.fullmatch
    monkeypatch.setattr(m.re, "fullmatch", lambda p, t, *a: calls.append((p, t)) or real(p, t, *a))
    budget = m.RegexBudget(10)
    _budget_run(budget, [("a+", "aa")] * 5 + [("b+", "bb")] * 2)
    assert calls == [("a+", "aa"), ("b+", "bb")]


# -- the cost model (security Pass 2 R1): the product of the factors, refused above a ceiling ---------------------------
CORNER_1 = "(?:a.|.a)" * 4 + r"\w*\w*\w{0,8}\w{0,8}x"                 # 147 ms admitted before
CORNER_2 = "(?i)" + CORNER_1                                              # 406 ms
CORNER_3 = r"(?is)\S{0,8}[a-z]{2,9}\S{3,}a{,}(?:a|.)(?:.|a)(?:a|.)(?:.|a)x"  # 440 ms


@pytest.mark.parametrize("pattern", [CORNER_1, CORNER_2, CORNER_3])
def test_r1_the_known_slow_admitted_patterns_are_refused_at_once(pattern):
    got, took, refused = _timed_none(pattern, "a" * 255 + "!")
    assert got is None and took < 0.05 and refused == {pattern}


def test_r1_the_cost_ceiling_is_a_product_with_literal_rows():
    ok = lambda p: m.regex_refusal(p, "x") is None
    assert ok(r"\w*\w*") and ok(r"\w*\w{0,3}")                          # 257*257 = 66k; 257*4 = 1k... both under 100k
    assert not ok(r"\w*\w*\w?")                                         # 132k: a third overlapping factor
    assert ok(r"\w*\w{0,8}\w{0,8}") and not ok(r"\w*\w{0,9}\w{0,9}\w{0,3}")      # 257*81 = 20.8k; 257*100*4 = 103k
    assert ok("(?:a.|.a)" * 4) and ok("(a|bc)(d|ef)(g|hi)(j|kl)")        # branch multiplicities: 16 and 16
    assert not ok("(?:a.|.a)" * 4 + r"\w*\w*")                           # 16 * 66k


def test_r1_a_widening_flag_multiplies_the_cost_by_four():
    ok = lambda p: m.regex_refusal(p, "x") is None
    assert ok(r"\w*\w{0,8}\w{0,8}") and ok(r"(?i)\w*\w{0,8}")           # 20.8k; 257*9*4 = 9.3k
    assert ok(r"\w*\w*") and not ok(r"(?i)\w*\w*")                      # 66k; 264k
    assert ok(r"(?i)\w*\w{0,8}\w{0,8}") and not ok(r"(?i)\w*\w{0,8}\w{0,8}\w{0,1}")   # 257*81*4 = 83k; *2 more = 167k (a factor of 2 would admit it)
    assert not ok(r"(?s)\w*\w*") and not ok(r"(?:(?i:\w*\w*))")


def test_r1_the_pair_budget_is_sized_from_the_measured_worst_pair_not_the_typical_one():
    assert m.MAX_REGEX_PAIRS == 800 and m.MAX_COST == 100_000         # worst admitted pair measured 12.2 ms: 800 pairs bound a run near 10 s


def test_p5_the_cost_ceiling_boundary_is_inclusive_and_the_widening_factor_is_four():
    ok = lambda p: m.regex_refusal(p, "x") is None
    assert m.WIDENING_FACTOR == 4
    assert ok(r"a{0,249}b{0,199}c?")                                    # 250 * 200 * 2 = 100000 = MAX_COST exactly: admitted (>, not >=)
    assert not ok(r"a{0,249}b{0,200}c?")                                # 250 * 201 * 2 = 100500
    assert ok(r"(?i)a{0,249}b{0,49}c?")                                 # 25000 * 4 = 100000 exactly: admitted
    assert ok(r"(?i)a{0,249}b{0,47}c?")                                 # 24000 * 4 = 96000 (a factor of 5 would refuse it)
    assert not ok(r"(?i)a{0,249}b{0,58}c?")                             # 29500: x4 = 118000 refused, x3 = 88500 would admit it
