"""Tests for ``ccg.keynorm.mask_http_segments`` (step 8a).

Oracle: ``docs/plans/connected-code-graph-eval-protocol2-ml.md`` §2, rows K1-K22 and K30-K32, in
provider form. Each row's *input* here is the writer's string after steps 1 and 4-7 (the earlier
steps are not this function's job); the expected value is copied from the row's output. Where a
row's earlier steps matter, the test says which, and the post-step input is stated by hand with
the RFC 3986 order (percent-decode, then resolve dot segments; BS2 §3 F-13).
"""

import pytest

from ccg import keynorm
from ccg.keynorm import mask_http_segments as mask

# (id, post-step-7 input, expected output of 8a in provider form)
ROWS = [
    ("K1", "GET /shop/api/health", "GET /shop/api/health"),
    ("K2", "POST /hooks/x7k2q9w4m1p8z3v6", "POST /hooks/{#}"),
    ("K3", "POST /hooks/a1b2c3d4e5f6g7h8", "POST /hooks/{#}"),
    ("K4", "POST /hooks/{}", "POST /hooks/{}"),
    ("K5", "POST /notify/x7k2-q9w4-m1p8-z3v6", "POST /notify/{#}"),
    ("K6", "POST /notify/x7k2+q9w4+m1p8+z3v6", "POST /notify/{#}"),
    ("K7", "GET /builds/3f2a9c1e5b7d", "GET /builds/{#}"),
    ("K8", "DELETE /items/550e8400-e29b-41d4-a716-446655440000", "DELETE /items/{#}"),
    ("K9", "POST /cb/sk-abcdefghijklmnopqrs", "POST /cb/{#}"),
    (
        "K10",
        "POST /services/T024BE7LD/B01ABCDEFGH/abcdefghijklmnopqrstuvwx",
        "POST /services/T024BE7LD/B01ABCDEFGH/{#}",
    ),
    ("K11", "GET /t/3f2a9c1e5b7d/x7k2q9w4m1p8z3v6/info", "GET /t/{#}/{#}/info"),
    # K12: steps 1, 4, 5 give this; the trailing "/" and its empty segment are kept.
    ("K12", "POST /hooks/x7k2q9w4m1p8z3v6/", "POST /hooks/{#}/"),
    # K13: step 6 decodes %6d (-> "m") before 8a, so the screen sees the decoded run.
    ("K13", "GET /hooks/x7k2q9w4m1p8z3v6", "GET /hooks/{#}"),
    # K14: %2F stays, uppercased, and splits the runs, so the segment passes.
    ("K14", "GET /a/x7k2q9w4%2Fm1p8z3v6", "GET /a/x7k2q9w4%2Fm1p8z3v6"),
    # K15: step 7 turned "{x7k2q9w4m1p8z3v6}" into "{}", which is not screened.
    ("K15", "GET /x/{}", "GET /x/{}"),
    ("K16", "GET /static/{**}", "GET /static/{**}"),
    ("K17", "GET /ntfy/kxqzmwvptryhbnjdlc", "GET /ntfy/kxqzmwvptryhbnjdlc"),
    (
        "K18",
        "GET /wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        "GET /wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    ),
    (
        "K19",
        "GET /release-2024-notes/s3-bucket-v2-us-east-1",
        "GET /release-2024-notes/s3-bucket-v2-us-east-1",
    ),
    ("K20", "POST /hooks/{#}", "POST /hooks/{#}"),
    ("K21", "GET /x/{?}", "GET /x/{?}"),
    ("K22", "GET /x7k2q9w4m1p8z3v6", "GET /{#}"),
    # K30: step 7 rewrites only a wholly-parameter segment; this one is a literal.
    ("K30", "GET /f/{x7k2q9w4m1p8z3v6}.json", "GET /f/{#}"),
    # K31 / K32: step 5 already resolved the dot segments (K31's masked segment is gone).
    ("K31", "GET /a/b", "GET /a/b"),
    ("K32", "GET /a/x7k2q9w4m1p8z3v6", "GET /a/{#}"),
    ("K33", "POST /hooks/x7k2q9w4m1p8", "POST /hooks/x7k2q9w4m1p8"),
]


@pytest.mark.parametrize("rid,given,expected", ROWS, ids=[r[0] for r in ROWS])
def test_oracle_provider_form(rid, given, expected):
    assert mask(given, "{#}") == expected


@pytest.mark.parametrize("rid,given,expected", ROWS, ids=[r[0] for r in ROWS])
def test_oracle_rows_are_fixed_points(rid, given, expected):
    assert mask(expected, "{#}") == expected
    assert mask(expected, "{?}") == expected


@pytest.mark.parametrize("rid,given,expected", ROWS, ids=[r[0] for r in ROWS])
def test_consumer_form_differs_only_in_the_placeholder(rid, given, expected):
    if "{#}" in given:  # K20: an existing provider placeholder is kept as is
        assert mask(given, "{?}") == given
    else:
        assert mask(given, "{?}") == expected.replace("{#}", "{?}")


def test_consumer_form_example_from_spec():
    # BS2 §1.8's example: the last segment fails T2, a consumer key gets {?}.
    got = mask("POST /services/T024BE7LD/B01ABCDEFGH/EXAMPLEexampleEXAMPLEexample0000", "{?}")
    assert got == "POST /services/T024BE7LD/B01ABCDEFGH/{?}"


def test_masking_provider_placeholder_input_untouched_in_consumer_form():
    # An already-masked provider key is not rewritten into the consumer form.
    assert mask("POST /hooks/{#}", "{?}") == "POST /hooks/{#}"
    assert mask("POST /hooks/{?}", "{#}") == "POST /hooks/{?}"


def test_screen_is_the_one_in_bundle(monkeypatch):
    # The function consults the module's screen_fails (imported from ccg.bundle): a stub changes it.
    monkeypatch.setattr(keynorm, "screen_fails", lambda s: s == "plain")
    assert mask("GET /plain/x7k2q9w4m1p8z3v6", "{#}") == "GET /{#}/x7k2q9w4m1p8z3v6"


def test_screen_fails_is_imported_not_copied():
    import ccg.bundle as bundle

    assert keynorm.screen_fails is bundle.screen_fails


def test_guards_hold_when_every_segment_fails_the_screen(monkeypatch):
    # With a screen that always fails, only the empty-segment and placeholder guards protect these.
    monkeypatch.setattr(keynorm, "screen_fails", lambda s: True)
    assert mask("GET /a//{}/{**}/{?}/{#}/b/", "{#}") == "GET /{#}//{}/{**}/{?}/{#}/{#}/"
    assert mask("GET /a//{}/{**}/{?}/{#}/b/", "{?}") == "GET /{?}//{}/{**}/{?}/{#}/{?}/"


# --- gaps the row leaves open: decisions are documented in the function's docstring ---


def test_bare_path_without_verb():
    assert mask("/hooks/x7k2q9w4m1p8z3v6", "{#}") == "/hooks/{#}"
    assert mask("/", "{#}") == "/"
    assert mask("", "{#}") == ""


def test_verb_is_never_screened():
    # A verb that would fail the screen (a 24-letter run) is still left alone.
    verb = "A" * 24
    assert mask(f"{verb} /ok", "{#}") == f"{verb} /ok"


def test_empty_segments_are_kept_verbatim():
    # Consecutive slashes are step 5's business; 8a keeps whatever empty segments it is given.
    assert mask("GET //a//x7k2q9w4m1p8z3v6/", "{#}") == "GET //a//{#}/"


@pytest.mark.parametrize("ph", ["{}", "{**}", "x", "", "{#}{#}"])
def test_rejects_any_placeholder_but_the_two_forms(ph):
    with pytest.raises(ValueError):
        mask("GET /a", ph)


def test_adjacent_placeholders_are_not_screened_even_if_inside_a_long_run():
    assert mask("GET /{**}", "{#}") == "GET /{**}"


# --- F-13: RFC 3986 order. A percent-encoded dot segment resolves AFTER decoding, so 8a
# --- sees the already-resolved path; 8a itself never decodes or resolves anything.


def test_encoded_dot_segments_are_not_this_functions_business():
    # Step 6 then step 5 turn "GET /a/%2E%2E/b" into "GET /b" (K42, F-13 order). The input here
    # is that result; a not-yet-resolved form is passed through unchanged, never resolved by 8a.
    assert mask("GET /b", "{#}") == "GET /b"
    assert mask("GET /a/%2E%2E/b", "{#}") == "GET /a/%2E%2E/b"
    assert mask("GET /a/../b", "{#}") == "GET /a/../b"


def test_masked_segment_hidden_behind_dots_is_gone_before_8a():
    # K31 under F-13: decode, then resolve -> the failing segment was removed by step 5.
    # With the pre-8a result as input, nothing remains to mask.
    assert mask("GET /a/b", "{#}") == "GET /a/b"


def test_leading_slash_means_no_verb_even_if_the_path_has_a_space():
    # A bare path is never split at a space: "/a" here is a segment, not a verb.
    assert mask("/a b/x7k2q9w4m1p8z3v6", "{#}") == "/a b/{#}"
    assert mask("/x7k2q9w4m1p8z3v6 b", "{#}") == "/{#}"


def test_segment_mixing_failing_run_and_percent_escape_is_masked():
    assert mask("GET /x7k2q9w4m1p8z3v6%2Fabc", "{#}") == "GET /{#}"


def test_8a_never_decodes_percent_escapes():
    # %6d is "m" only after step 6; 8a screens it as written, where the escape splits the run.
    assert mask("GET /x7k2q9w4%6d1p8z3v6", "{#}") == "GET /x7k2q9w4%6d1p8z3v6"


def test_verb_bearing_path_containing_a_space():
    assert mask("GET /a b/x7k2q9w4m1p8z3v6", "{#}") == "GET /a b/{#}"
    # The verb ends at the first space, so a failing segment before a later space is still screened.
    assert mask("GET /x7k2q9w4m1p8z3v6 b/c", "{#}") == "GET /{#}/c"
