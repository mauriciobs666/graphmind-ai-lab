"""P1-03 — `ccg.sysname`: the system-name whitelist and the only key and ACL-user constructors.

Spec: `docs/plans/connected-code-graph-p1-graph.md` §7 (whitelist, users, grants) and §8 P1-19 (c).
Plan: `docs/plans/connected-code-graph-p1-extraction.md` § P1-03 and P1-48 (2) (the script's same-verdict check).

Expected values here are written out, never read back from the constants under test, except the two
cross-checks whose point is agreement with an independent artifact (the shell script's own check).
"""
from __future__ import annotations

import ast
import itertools
import re
import subprocess
from pathlib import Path

import pytest

from ccg import sysname
from ccg.sysname import SystemNameError

CPG = Path(__file__).resolve().parent.parent
SCRIPT = CPG / "infra" / "ccg-falkordb.sh"

VALID = ["a", "shop", "shop2", "a-b", "a_b", "a1-b2_c3", "graphmind-ai-lab", "0", "9-9", "x" * 48]
INVALID = [
    "", "_", "-", "shop_", "_shop", "-shop", "shop-", "a__b", "a--b", "a_-b", "a-_b", "Shop", "SHOP",
    "sh op", "a.b", "a:b", "a*b", "a?b", "a[b", "a]b", "a{b}", "a\\b", "a/b", "shop\n", "\nshop", "shóp",
    "x" * 49, "x" * 100, "default ", "a\x00b",
]


@pytest.mark.parametrize("name", VALID)
def test_check_system_name_returns_a_valid_name_unchanged(name):
    assert sysname.check_system_name(name) == name


@pytest.mark.parametrize("name", INVALID)
def test_check_system_name_refuses_every_malformed_name(name):
    with pytest.raises(SystemNameError):
        sysname.check_system_name(name)


@pytest.mark.parametrize("value", [None, 1, b"shop", ["shop"]])
def test_check_system_name_refuses_a_non_string(value):
    with pytest.raises(SystemNameError):
        sysname.check_system_name(value)


def test_system_name_error_is_a_value_error():
    assert issubclass(SystemNameError, ValueError)


def test_the_length_bound_is_48_inclusive():
    assert sysname.check_system_name("a" * 48)
    with pytest.raises(SystemNameError, match="48"):
        sysname.check_system_name("a" * 49)


def test_a_trailing_newline_is_refused_although_dollar_would_match_before_it():
    # Positive control: Python's `$` admits it, so only a full-string anchor refuses it.
    assert re.match(r"^[a-z0-9]+$", "shop\n")
    with pytest.raises(SystemNameError):
        sysname.check_system_name("shop\n")


def test_roles_are_the_four_fixed_tokens_in_order():
    assert sysname.ROLES == ("build", "verdict", "engine", "raw")


def test_key_constructors_form_the_spec_names():
    assert sysname.live_key("shop") == "ccg_shop"
    assert sysname.build_key("shop", "1") == "ccg_shop__build_1"
    assert sysname.build_key("shop", "a1b2c3") == "ccg_shop__build_a1b2c3"
    assert sysname.overlay_key("shop", "7f") == "ccg_shop__ov_7f"
    assert sysname.lock_key("shop") == "ccg_shop:verdict_lock"


@pytest.mark.parametrize("role", ["build", "verdict", "engine", "raw"])
def test_acl_user_is_ccg_role_system(role):
    assert sysname.acl_user(role, "shop") == f"ccg_{role}_shop"
    assert sysname.acl_user(role, "a-b_c") == f"ccg_{role}_a-b_c"


@pytest.mark.parametrize("role", ["admin", "", "Build", "build_", "default", "raw ", None])
def test_acl_user_refuses_a_role_outside_the_four(role):
    with pytest.raises(ValueError):
        sysname.acl_user(role, "shop")


@pytest.mark.parametrize("bad", ["shop_", "a__b", "", "x" * 49, "Shop"])
def test_every_constructor_checks_the_system_name(bad):
    for call in (
        lambda: sysname.live_key(bad),
        lambda: sysname.build_key(bad, "1"),
        lambda: sysname.overlay_key(bad, "1"),
        lambda: sysname.lock_key(bad),
        lambda: sysname.acl_user("build", bad),
    ):
        with pytest.raises(SystemNameError):
            call()


@pytest.mark.parametrize("bad", ["", "a_b", "a b", "a*", "a:b", "a\n", "-a", "a-", "a--b", "a/b", "A_"])
def test_build_and_overlay_ids_are_one_safe_token(bad):
    with pytest.raises(ValueError):
        sysname.build_key("shop", bad)
    with pytest.raises(ValueError):
        sysname.overlay_key("shop", bad)


def test_the_name_whitelist_keeps_systems_isolated_under_the_acl_patterns():
    """§7 isolation: a key of `t` matches `ccg_s` / `ccg_s__*` only if `t` is `s__…` or `s_`; both are refused names.

    Exhaustive over every name to length 4 over {a, b, _, -}: for every pair of distinct VALID names s, t, no key
    of t (live, build, overlay, lock) is matched by s's selector patterns. The positive control is `shop_`: the
    pattern alone would grant it, and the whitelist refuses it.
    """
    alphabet = "ab_-"
    names = [
        "".join(p) for n in range(1, 5) for p in itertools.product(alphabet, repeat=n)
        if _valid(("".join(p)))
    ]

    def selectors(s):
        # `~ccg_<s>` is exact; `~ccg_<s>__*` is a glob that `fnmatch` reads the same way for these names.
        return lambda key: key == f"ccg_{s}" or key.startswith(f"ccg_{s}__")

    for s in names:
        match = selectors(s)
        for t in names:
            if t == s:
                continue
            for key in (sysname.live_key(t), sysname.build_key(t, "1"), sysname.overlay_key(t, "1")):
                assert not match(key), (s, t, key)
    # Positive control: the refused name really is granted by the pattern alone.
    assert selectors("shop")("ccg_shop___build_1")
    with pytest.raises(SystemNameError):
        sysname.check_system_name("shop_")


def _valid(name: str) -> bool:
    try:
        sysname.check_system_name(name)
    except SystemNameError:
        return False
    return True


# --- same verdict as the script's own check (P1-48 (2)) ------------------------------------------------------------


def _script_acl_check(users: list[str]) -> list[bool]:
    """Run `acl_user_ok` exactly as `cpg/infra/ccg-falkordb.sh` defines it, over every user, in one bash."""
    text = SCRIPT.read_text()
    m = re.search(r"^ACL_USER_RE=.*?^acl_user_ok\(\) \{.*?^\}\n", text, re.S | re.M)
    assert m, "ACL_USER_RE / acl_user_ok not found in the script"
    prog = m.group(0) + 'while IFS= read -r u; do if acl_user_ok "$u"; then echo 1; else echo 0; fi; done\n'
    out = subprocess.run(
        ["bash", "-c", prog], input="\n".join(users) + "\n", capture_output=True, text=True, check=True
    ).stdout.split()
    assert len(out) == len(users)
    return [o == "1" for o in out]


def _py_acl_check(user: str) -> bool:
    if user == "default":
        return True
    m = sysname.acl_user_re().fullmatch(user)
    return bool(m) and 1 <= len(m.group("system")) <= 48


def _users() -> list[str]:
    alphabet = "ab_-"
    systems = ["".join(p) for n in range(1, 6) for p in itertools.product(alphabet, repeat=n)]
    users = [f"ccg_{r}_{s}" for r in sysname.ROLES for s in systems]
    users += ["ccg_build", "ccg_raw_", "ccg_admin_x", "ccg___", "ccg_build___", "ccg_build_a_", "ccg_build_a__b",
              "default", "defaults", "", "ccg_build_Shop", "ccg_Build_a", "xccg_build_a", "ccg_build_a b"]
    users += [f"ccg_{r}_{'x' * n}" for r in sysname.ROLES for n in (47, 48, 49, 50)]
    return users


def test_acl_user_re_gives_the_same_verdict_as_the_script_over_every_role_and_name():
    users = _users()
    assert len(users) > 5000  # the sweep is real, not a stub
    script = _script_acl_check(users)
    mine = [_py_acl_check(u) for u in users]
    assert script == mine
    # Anchors on both sides: the verdicts are not all one value, and the length edges are where they flip.
    assert sum(mine) > 500 and not all(mine)
    verdict = dict(zip(users, mine))
    assert verdict["ccg_build_" + "x" * 48] and not verdict["ccg_build_" + "x" * 49]
    assert verdict["default"] and not verdict["ccg_build_a_"] and not verdict["ccg_build_a__b"]


def test_acl_user_re_agrees_with_acl_user_for_every_role_and_valid_name():
    for role in sysname.ROLES:
        for name in VALID:
            m = sysname.acl_user_re().fullmatch(sysname.acl_user(role, name))
            assert m and m.group("system") == name and m.group("role") == role


def test_acl_user_re_is_built_from_roles_and_the_name_pattern_not_a_second_literal():
    """No second regex: the pattern's text contains `SYSTEM_NAME_RE`'s body and each role token."""
    pat = sysname.acl_user_re().pattern
    for role in sysname.ROLES:
        assert role in pat
    assert sysname.SYSTEM_NAME_BODY in pat


# --- the formatting sweep ---------------------------------------------------------------------------------------


def _ccg_text_sites(tree: ast.AST) -> list[int]:
    """Line numbers where `ccg_…` text is formatted: an f-string, `%`, `.format`, or `+` on a `ccg_` literal."""

    def is_ccg(node) -> bool:
        return isinstance(node, ast.Constant) and isinstance(node.value, str) and "ccg_" in node.value

    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr) and any(is_ccg(v) for v in node.values):
            hits.append(node.lineno)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mod, ast.Add)) and (
            is_ccg(node.left) or is_ccg(node.right)
        ):
            hits.append(node.lineno)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("format", "join")
            and (
                is_ccg(node.func.value)
                or any(is_ccg(a) or (isinstance(a, (ast.List, ast.Tuple)) and any(is_ccg(e) for e in a.elts))
                       for a in node.args)
            )
        ):
            hits.append(node.lineno)
    return hits


@pytest.mark.parametrize(
    "src",
    [
        'x = f"ccg_{s}"',
        'x = "ccg_%s" % s',
        'x = "ccg_{}".format(s)',
        'x = "ccg_" + s',
        'x = s + "__ccg_x"',
        'x = f"{r}ccg_build_{s}"',
        'x = "".join(["ccg_", s])',
    ],
)
def test_the_sweep_detects_each_formatting_shape(src):
    assert _ccg_text_sites(ast.parse(src))


@pytest.mark.parametrize("src", ['"""ccg_<s> docs"""', 'x = "ccg_"', 'y = f"{a}-{b}"', 'z = "ccg" + s'])
def test_the_sweep_ignores_plain_text_and_unrelated_formatting(src):
    assert not _ccg_text_sites(ast.parse(src))


def test_no_python_file_under_ccg_or_clients_forms_ccg_key_or_user_text_except_sysname():
    offenders = []
    files = 0
    for top in ("ccg", "clients"):
        for path in sorted((CPG / top).rglob("*.py")):
            files += 1
            if path.name == "sysname.py" and path.parent.name == "ccg":
                continue
            for line in _ccg_text_sites(ast.parse(path.read_text(), filename=str(path))):
                offenders.append(f"{path.relative_to(CPG)}:{line}")
    assert files > 5
    assert offenders == []


def test_the_sweep_does_see_sysname_itself():
    """Positive control: the exempted module is the only place that forms the text, so the sweep must flag it."""
    tree = ast.parse((CPG / "ccg" / "sysname.py").read_text())
    assert _ccg_text_sites(tree)
