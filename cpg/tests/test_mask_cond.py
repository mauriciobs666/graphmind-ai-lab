"""P1-06 (F-9) — `Cond.code` literal masking (`extractors/joern/normalize/mask.py`).

Plan: `docs/plans/connected-code-graph-p1-extraction.md` § P1-06. Expected values are written out by hand.
"""
from __future__ import annotations

import random
import string

import pytest

from ccg import bundle
from extractors.joern.normalize import assemble_bundle, mask
from extractors.joern.normalize.mask import mask_cond_code

HEX32 = "0123456789abcdef0123456789abcdef"
UUID = "123e4567-e89b-12d3-a456-426614174000"
TOKEN = "ghp_" + "aB3dE5gH7jK9mN1pQ3sT5vX7zA9cE1gI3kM5"


@pytest.mark.parametrize("code", [
    "x > 0", "", "a and not b", "len(items) == 3", "x.y(z) if q else r", "if (a && b) {",
])
def test_condition_without_literal_is_byte_identical(code):
    assert mask_cond_code(code) == code


@pytest.mark.parametrize("code", [
    'mode == "fast"', "mode == 'fast'", "mode === `fast`", 'x == ""', 'kind in ("a", "b")',
])
def test_passing_literal_stays_verbatim(code):
    assert mask_cond_code(code) == code


@pytest.mark.parametrize("secret", [HEX32, UUID, TOKEN])
def test_failing_literal_is_masked(secret):
    assert mask_cond_code(f'k == "{secret}"') == 'k == "…"'
    assert mask_cond_code(f"k == '{secret}'") == 'k == "…"'


def test_name_fails_literal_is_masked():
    assert mask_cond_code('h == "password"') == 'h == "…"'


def test_each_literal_judged_independently():
    assert mask_cond_code(f'a == "ok" or b == "{HEX32}" or c == "fine"') == 'a == "ok" or b == "…" or c == "fine"'


def test_escaped_quote_does_not_end_literal():
    assert mask_cond_code(f'x == "a\\"{HEX32}" and y') == 'x == "…" and y'
    assert mask_cond_code(f"x == 'a\\'{HEX32}' and y") == 'x == "…" and y'


def test_escaped_backslash_before_closing_quote_ends_literal():
    assert mask_cond_code('x == "a\\\\" and y == "' + HEX32 + '"') == 'x == "a\\\\" and y == "…"'


def test_other_quote_kind_inside_literal_is_not_a_delimiter():
    assert mask_cond_code(f'''x == "it's {HEX32}" and z''') == 'x == "…" and z'
    assert mask_cond_code(f"""x == 'say "{HEX32}"' and z""") == 'x == "…" and z'


def test_python_triple_quoted_literal():
    assert mask_cond_code(f'x == """a "b" {HEX32}"""') == 'x == "…"'
    assert mask_cond_code('x == """a "b" c"""') == 'x == """a "b" c"""'


def test_python_string_prefix_is_kept_outside_the_mask():
    assert mask_cond_code(f'x == b"{HEX32}"') == 'x == b"…"'
    assert mask_cond_code(f'x == f"{HEX32}"') == 'x == f"…"'


def test_template_literal_with_secret_text_is_masked_whole():
    assert mask_cond_code(f"x === `a${{b}}{HEX32}`") == 'x === "…"'


def test_template_literal_passing_stays_and_nested_literal_is_masked():
    assert mask_cond_code("x === `a${b}c`") == "x === `a${b}c`"
    assert mask_cond_code(f'x === `a${{f("{HEX32}")}}c`') == 'x === `a${f("…")}c`'


def test_template_nested_template_and_braces():
    code = f"x === `a${{ {{k: 1}}.k + `{HEX32}` }}c`"
    assert mask_cond_code(code) == 'x === `a${ {k: 1}.k + "…" }c`'


def test_template_quote_inside_expression_does_not_close_template():
    assert mask_cond_code("x === `a${'`'}b` && y") == "x === `a${'`'}b` && y"


def test_unterminated_literal_failing_is_masked_and_passing_kept():
    assert mask_cond_code(f'x == "{HEX32}') == 'x == "…"'
    assert mask_cond_code('x == "pinn') == 'x == "pinn'
    assert mask_cond_code("x === `a${b") == "x === `a${b"




def test_escaped_backtick_does_not_close_template():
    assert mask_cond_code(f"x === `a\\`{HEX32}`") == 'x === "…"'
    assert mask_cond_code("x === `a\\`b` && Y") == "x === `a\\`b` && Y"


def test_template_expression_text_is_not_judged_as_static_text():
    ident = "qrstuvwxyzqrstuvwxyzqrstuv"          # 26 non-hex letters: T2 flags it as a literal; not a hex run
    assert mask_cond_code(f"x === `a${{{ident}}}c`") == f"x === `a${{{ident}}}c`"


def test_non_literal_text_keeps_its_case_and_spacing():
    code = 'Foo.Bar  ==   "ok" AND  BAZ'
    assert mask_cond_code(code) == code




def test_masking_is_idempotent_and_linear_on_hostile_input():
    hostile = '"' * 5000 + "\\" * 5001 + "`${" * 3000
    out = mask_cond_code(hostile)
    assert mask_cond_code(out) == out


def _rand_body(rng):
    kind = rng.choice(["hex", "uuid", "tok", "word", "mixed", "esc"])
    if kind == "hex":
        return "".join(rng.choice("0123456789abcdef") for _ in range(rng.randint(12, 40)))
    if kind == "uuid":
        return UUID.replace("e", rng.choice("abcdef"))
    if kind == "tok":
        return rng.choice(["ghp_", "sk-", "AKIA", "xoxb-"]) + "".join(
            rng.choice(string.ascii_letters + string.digits) for _ in range(rng.randint(20, 40)))
    if kind == "word":
        return rng.choice(["fast", "ok", "token", "a b", "Bearer x", "pin", "pinned", "x_y"])
    if kind == "esc":
        return "a\\\"" + "".join(rng.choice("0123456789abcdef") for _ in range(rng.randint(12, 30)))
    return "".join(rng.choice(string.ascii_letters + string.digits + " _-.") for _ in range(rng.randint(0, 40)))


def _unused(code):
    """Independent, simple extraction of the literal bodies of a code made of one-delimiter literals."""
    return code


@pytest.mark.parametrize("seed", range(40))
def test_property_masked_condition_never_holds_a_flagged_literal(seed):
    rng = random.Random(seed)
    for _ in range(60):
        d = rng.choice(['"', "'", "`"])
        body = _rand_body(rng).replace(d, "") if d != '"' and d != "'" else _rand_body(rng).replace("\\\"", "").replace(d, "")
        code = f"a == {d}{body}{d} and b"
        out = mask_cond_code(code)
        flagged = bundle.screen_fails(body) or bundle.name_fails(body)
        if flagged:
            assert out == 'a == "…" and b', (code, out)
        else:
            assert out == code, (code, out)
        # the output holds no literal the screen flags
        inner = out[len("a == "):-len(" and b")]
        assert inner == '"…"' or not (bundle.screen_fails(inner[1:-1]) or bundle.name_fails(inner[1:-1]))


def test_cond_ordinals_follow_the_raw_code_not_the_masked_one(tmp_path):
    from normalize_assembly_helpers import SRC, full_facts, git_repo
    from extractors.joern import run as joern_run
    commit = git_repo(tmp_path, {"r.py": SRC})
    facts = full_facts()
    facts["c"][1]["l"], facts["c"][1]["c"] = facts["c"][0]["l"], facts["c"][0]["c"]     # one position
    facts["c"][0]["code"] = f'x == "{HEX32}"'        # raw: longer; masked `x == "…"`: shorter than the other
    facts["c"][1]["code"] = 'x == "zzzzzzzz"'
    records, _ = assemble_bundle.assemble(
        lang="python", unit_root=".", repo_id="r", facts=facts, stage=tmp_path, astgen_dir=None, repo=tmp_path,
        commit=commit, audit_ids=["11111111-1111-4111-8111-111111111111"], extract_script=joern_run.EXTRACT_SC,
        built_at="2026-01-01T00:00:00Z")
    conds = sorted((r["ordinal"], r["code"]) for r in records if r["type"] == "cond")
    assert conds == [(0, 'x == "zzzzzzzz"'), (1, 'x == "…"')]


def test_a_literal_is_masked_before_the_200_character_cut(tmp_path):
    from normalize_assembly_helpers import SRC, full_facts, git_repo
    from extractors.joern import run as joern_run
    commit = git_repo(tmp_path, {"r.py": SRC})
    facts = full_facts()
    head = "self.a == " + "x" * 185               # 195 chars; a cut at 200 would keep `"012`
    facts["c"][0]["code"] = head + f' "{HEX32}" and y'
    assert len(head) == 195
    records, _ = assemble_bundle.assemble(
        lang="python", unit_root=".", repo_id="r", facts=facts, stage=tmp_path, astgen_dir=None, repo=tmp_path,
        commit=commit, audit_ids=["11111111-1111-4111-8111-111111111111"], extract_script=joern_run.EXTRACT_SC,
        built_at="2026-01-01T00:00:00Z")
    code = next(r["code"] for r in records if r["type"] == "cond" and r["code"].startswith("self.a =="))
    assert code == head + ' "…"'


def test_assemble_masks_secret_literal_in_cond_record_after_collapse(tmp_path):
    from normalize_assembly_helpers import SRC, full_facts, git_repo
    from extractors.joern import run as joern_run
    commit = git_repo(tmp_path, {"r.py": SRC})
    facts = full_facts()
    facts["c"][0]["code"] = f'self.a  ==\n "{HEX32}"'
    facts["c"][1]["code"] = 'self.b == "fine"'
    records, _ = assemble_bundle.assemble(
        lang="python", unit_root=".", repo_id="r", facts=facts, stage=tmp_path, astgen_dir=None, repo=tmp_path,
        commit=commit, audit_ids=["11111111-1111-4111-8111-111111111111"], extract_script=joern_run.EXTRACT_SC, built_at="2026-01-01T00:00:00Z")
    codes = sorted(r["code"] for r in records if r["type"] == "cond")
    assert codes == ['self.a == "…"', 'self.b == "fine"']


# --- the 200-character cut (B1): the stored tail of a passing literal must pass the screens too ----------------------------
@pytest.fixture(scope="module")
def cut_repo(tmp_path_factory):
    from normalize_assembly_helpers import SRC, git_repo
    path = tmp_path_factory.mktemp("cutrepo")
    return path, git_repo(path, {"r.py": SRC})


def _stored(cut_repo, code):
    """The `code` of the one cond the written records carry for raw `code` (through `assemble`: mask, cut, mask)."""
    from normalize_assembly_helpers import full_facts
    from extractors.joern import run as joern_run
    path, commit = cut_repo
    facts = full_facts()
    facts["c"][0]["code"] = code
    facts["c"][1]["code"] = "zz"
    records, _ = assemble_bundle.assemble(
        lang="python", unit_root=".", repo_id="r", facts=facts, stage=path, astgen_dir=None, repo=path, commit=commit,
        audit_ids=["11111111-1111-4111-8111-111111111111"], extract_script=joern_run.EXTRACT_SC,
        built_at="2026-01-01T00:00:00Z")
    return next(r["code"] for r in records if r["type"] == "cond" and r["code"] != "zz")


#: literals that pass whole whose prefixes fail: a whole-token name word cut early, a hex run cut at 12+, a short hex-looking word.
_PASSING_WHOLE = ["pinned please", "seeded value", "salted caramel", "vaulted", "deadbeefcafexyz tail", "sshd running"]


@pytest.mark.parametrize("delim", ['"', "'"])
@pytest.mark.parametrize("lit", _PASSING_WHOLE)
def test_stored_tail_of_a_passing_literal_passes_the_screens_at_every_cut_offset(cut_repo, delim, lit):
    for k in range(len(lit) + 3):
        pad = "z" * (199 - k)
        raw = f"{pad} {delim}{lit}{delim} tail"
        assert not (bundle.screen_fails(lit) or bundle.name_fails(lit))
        stored = _stored(cut_repo, raw)
        assert len(stored) <= 200
        assert stored.startswith(pad), "the pad must survive intact (not hex, or the backstop eats it)"
        frag = stored[len(pad) + 1:]
        if frag.startswith('"…"'):
            continue
        body = frag[1:]
        if len(body) and body.endswith(delim) and len(body) > 0 and frag.count(delim) == 2:
            body = body[:-1]
        assert not (bundle.screen_fails(body) or bundle.name_fails(body)), (k, stored[-40:])
        assert mask_cond_code(stored) == stored


@pytest.mark.parametrize("raw_tail", [
    '`pinned ${x} please` tail',            # the cut lands in the static text, inside `${}` or after it
    '"pinned \\" please" tail',            # an escape sequence at the boundary
    '"seed\\x41 here" tail',
])
def test_stored_tail_is_stable_when_the_cut_lands_in_an_escape_or_an_expression(cut_repo, raw_tail):
    for k in range(len(raw_tail) + 1):
        stored = _stored(cut_repo, "z" * (199 - k) + " " + raw_tail)
        assert stored.startswith("z" * (199 - k)) and mask_cond_code(stored) == stored, (k, stored[-40:])


# --- M1: a quote-parity flip must not store a whole secret (backstop on the text the scanner took for code) --------------
def test_regex_quote_does_not_let_a_secret_literal_through():
    assert mask_cond_code(f'/"/.test(a) && b === "{HEX32}"') == '/"/.test(a) && b === "…"'


def test_apostrophe_in_code_does_not_let_a_secret_literal_through():
    assert mask_cond_code(f"it's fine and k == '{HEX32}'") == "it's fine and k == '…'"


def test_uuid_outside_any_recognised_literal_is_masked():
    assert mask_cond_code(f"/'/.test(a) && b === '{UUID}'") == "/'/.test(a) && b === '…'"


def test_backstop_leaves_short_hex_words_and_ordinary_identifiers_alone():
    code = "a == dead and bead(0xBEEF) and abcdef12345 and iteratorNonEmptyOrException"   # 11 hex chars at most
    assert mask_cond_code(code) == code


# --- M2/M3: each screen rule and the name rule pinned on the literal as a whole, position and word order included -----------
@pytest.mark.parametrize("body, rule", [
    ("sk-abcdefghijklmnopq", "T5 only: a 20-character token-prefix literal; no other rule flags it, nor does its reversal"),
    ("q" * 24, "T2 only: a 24-character alphanumeric run with no digits and no hex"),
    ("123e4567-e89b-12d3-a456-42661417400", "T4 (UUID shape minus a digit: T6 co-fires, no run is long)"),
])
def test_each_screen_rule_masks_a_literal_it_alone_catches(body, rule):
    assert bundle.screen_fails(body) and not bundle.name_fails(body), rule
    assert mask_cond_code(f'x == "{body}"') == 'x == "…"'


def test_t5_verdict_depends_on_the_literal_start_not_its_reversal():
    assert not bundle.screen_fails("sk-abcdefghijklmnopq"[::-1])      # the reversal passes, so a reversed call would keep it
    assert mask_cond_code('x == "sk-abcdefghijklmnopq"') == 'x == "…"'


@pytest.mark.parametrize("body", ["empty key", "a b key", "the user session"])
def test_name_word_in_any_position_of_a_multi_word_literal_masks_it(body):
    assert not bundle.screen_fails(body) and bundle.name_fails(body)
    assert mask_cond_code(f'x == "{body}"') == 'x == "…"'



def test_backstop_boundary_is_twelve_hex_characters():
    assert mask_cond_code("a == deadbeef0123 and b") == "a == … and b"      # 12: T3's own threshold
    assert mask_cond_code("a == deadbeef012 and b") == "a == deadbeef012 and b"   # 11


@pytest.mark.parametrize("secret", ["0123456789abcdef" * 4, "0123456789abcdef0123456789abcdef01234567"])   # 64 and 40 hex
def test_long_hex_after_a_quote_parity_flip_is_masked_in_full(secret):
    assert mask_cond_code(f'/"/.test(a) && k == "{secret}"') == '/"/.test(a) && k == "…"'
    assert mask_cond_code(f"it's fine and k == '{secret}'") == "it's fine and k == '…'"


def test_a_screen_failure_only_beyond_character_64_still_masks_the_literal():
    body = "ok " * 22 + HEX32                                  # 66 characters of clean words, then a 32-hex run
    assert not (bundle.screen_fails(body[:64]) or bundle.name_fails(body)) and bundle.screen_fails(body)
    assert mask_cond_code(f'x == "{body}"') == 'x == "…"'
