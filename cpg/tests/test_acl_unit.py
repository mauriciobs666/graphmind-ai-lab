"""P1-19 dispatch 2, offline: ``ccg.store.acl`` reads the ACL lines from its data file (one source, pinned to the plan's §7 by a drift test), checks every
template against a closed grammar, builds the ``ACL SETUSER`` arguments, writes the credentials file and never lets a password reach an exception, a log
or stdout. The server-side grants are ``test_acl_live.py``'s."""
from __future__ import annotations

import logging
import os
import stat
import time
from pathlib import Path

import pytest

from ccg import atrest, sysname
from ccg.store import acl

NOTE = Path(__file__).resolve().parents[2] / "docs" / "plans" / "connected-code-graph-p1-graph.md"


def section7_block(text: str) -> str:
    return text.split("## 7. ACL users", 1)[1].split("```", 2)[1]


@pytest.fixture(autouse=True)
def creds(tmp_path, monkeypatch):
    d = tmp_path / "cred"
    monkeypatch.setenv("CCG_CREDENTIALS_DIR", str(d))
    return d


class FakeAdmin:
    """Records every command. ``fail_on`` is the 1-based call index that raises (Redis echoes the modifier in its ACL errors); ``boom`` is an
    exception class to raise there instead (``KeyboardInterrupt``); ``absent`` are users ``ACL GETUSER`` reports as not existing."""

    def __init__(self, fail_on=None, boom=None, absent=()):
        self.calls: list[tuple] = []
        self.fail_on, self.boom, self.absent = fail_on, boom, set(absent)
        self.connection = self

    def execute_command(self, *args):
        self.calls.append(args)
        if self.fail_on == len(self.calls):
            if self.boom:
                raise self.boom()
            modifier = next((a for a in args if isinstance(a, str) and a.startswith(">")), "x")
            raise RuntimeError(f"Error in ACL SETUSER modifier '{modifier}': Syntax error")
        if args[:2] == ("ACL", "GETUSER"):
            return None if args[2] in self.absent else [b"flags"]
        return b"OK"

    def kinds(self):
        return [c[:2] if c[0] in ("ACL", "CLIENT") else c[:1] for c in self.calls]


def setusers(admin):
    return [c for c in admin.calls if c[:2] == ("ACL", "SETUSER")]


# ------------------------------------------------------------------------------------------------ the data file and the plan

def test_the_template_is_the_data_file_with_one_secret_per_role():
    t = acl.parse_acl_template()
    assert tuple(t) == sysname.ROLES and acl.DATA_FILE.name == "acl_users.txt"
    assert all(toks.count("><secret>") == 1 for toks in t.values())
    assert "%W~ccg_<s>" in t["build"] and "~ccg_<s>__*" in t["build"] and "+rename" in t["build"]
    assert "+graph.query" in t["verdict"] and "(~ccg_<s>:verdict_lock" in t["verdict"] and "+eval)" in t["verdict"]   # continuation line joined
    assert "+graph.ro_query" in t["engine"] and t["engine"] == t["raw"]


def test_the_plans_section_7_block_equals_the_data_file():
    """A rename or edit of the plan fails here, instead of silently changing production grants (the data file is what ships)."""
    plan = acl.parse_acl_lines(section7_block(NOTE.read_text()))
    assert plan == acl.parse_acl_template()
    norm = lambda t: [" ".join(l.split()) for l in t.strip().splitlines() if l.strip()]
    assert norm(section7_block(NOTE.read_text())) == norm(acl.DATA_FILE.read_text())


def test_a_drifted_plan_block_is_detected():
    block = section7_block(NOTE.read_text())
    assert acl.parse_acl_lines(block.replace("+eval)", "+eval +flushall)", 1)) != acl.parse_acl_template()


def test_there_is_no_copy_of_the_acl_lines_in_the_module_source():
    src = Path(acl.__file__).read_text()
    assert "ACL SETUSER ccg_" not in src and "resetkeys resetchannels -@all +graph" not in src


@pytest.mark.parametrize("mutate", [
    lambda b: b.replace("ACL SETUSER ccg_raw_<s>   ", "ACL SETUSER ccg_ops_<s>   "),                          # a role outside ROLES
    lambda b: b.replace("ACL SETUSER ccg_engine_<s>", "ACL SETUSER ccg_build_<s>", 1),                        # a role twice
    lambda b: b + "ACL SETUSER ccg_build_<s> on ><secret>\n",                                                    # a duplicate user line
    lambda b: b.replace("ccg_verdict_<s>", "ccg_verdict_x<s>", 1),                                              # a malformed name
    lambda b: "",                                                                                                # nothing
])
def test_a_changed_line_shape_fails_loudly(mutate):
    with pytest.raises(acl.AclError):
        acl.parse_acl_lines(mutate(acl.DATA_FILE.read_text()))


def test_an_unreadable_data_file_is_an_acl_error(tmp_path):
    with pytest.raises(acl.AclError):
        acl.parse_acl_template(tmp_path / "missing.txt")


# ------------------------------------------------------------------------------------------------ the closed grammar

def _with(role, edit):
    t = {r: list(v) for r, v in acl.parse_acl_template().items()}
    t[role] = edit(t[role])
    return t


def sub(tokens, old, new):
    """Replace the token ``old`` by the token list ``new``."""
    out = []
    for x in tokens:
        out += new if x == old else [x]
    return out


TAMPERS = {
    "catch-all keys and +@all (engine)": ("engine", lambda t: t + ["~*", "+@all"]),
    "a second password": ("engine", lambda t: t + [">evilpw-SECRET123"]),
    "a password hash token": ("raw", lambda t: t + ["#" + "a" * 64]),
    "allkeys allcommands": ("raw", lambda t: t + ["allkeys", "allcommands"]),
    "nopass": ("build", lambda t: t + ["nopass"]),
    "an extra system's key pattern": ("build", lambda t: t + ["~ccg_shop2"]),
    "an extra system's key (verdict)": ("verdict", lambda t: t + ["~ccg_shop2__*"]),
    "upper-case command": ("engine", lambda t: [x.replace("+graph.ro_query", "+GRAPH.RO_QUERY") for x in t]),
    "upper-case head": ("engine", lambda t: ["ON"] + t[1:]),
    "unicode lookalike command": ("verdict", lambda t: t + ["+graph.quеry"]),
    "fullwidth key pattern": ("engine", lambda t: t + ["~ccg_ｓhop"]),
    "a category": ("verdict", lambda t: t + ["+@read"]),
    "a second -@all category line": ("verdict", lambda t: t + ["-@all"]),
    "a channel": ("engine", lambda t: t + ["&*"]),
    "allchannels": ("engine", lambda t: t + ["allchannels"]),
    "reset in the line": ("engine", lambda t: ["reset"] + t),
    "resetpass": ("engine", lambda t: t + ["resetpass"]),
    "a command outside the role (engine write)": ("engine", lambda t: t + ["+graph.query"]),
    "flushall (build)": ("build", lambda t: t + ["+flushall"]),
    "config set (build)": ("build", lambda t: t + ["+config|set"]),
    "graph.delete (build)": ("build", lambda t: t + ["+graph.delete"]),
    "a subcommand outside the role": ("build", lambda t: t + ["+graph.config|set"]),
    "read-write on live": ("build", lambda t: [x.replace("%W~ccg_<s>", "~ccg_<s>") for x in t]),
    "the build key patterns on raw": ("raw", lambda t: t + ["~ccg_<s>__*"]),
    "the lock key on engine": ("engine", lambda t: t + ["~ccg_<s>:verdict_lock"]),
    "a repeated command": ("engine", lambda t: t + ["+info"]),
    "a missing head token": ("engine", lambda t: [x for x in t if x != "resetchannels"]),
    "a second secret": ("verdict", lambda t: t + ["><secret>"]),
    "an unknown token kind": ("verdict", lambda t: t + ["banana"]),
    "a selector widened to the system": ("build", lambda t: sub(t, "(~ccg_<s>__build_*", ["(~ccg_<s>*"])),
    "a selector with an extra command": ("build", lambda t: sub(t, "+expire)", ["+expire", "+rename)"])),
    "a selector with a live-key pattern": ("verdict", lambda t: sub(t, "(~ccg_<s>:verdict_lock", ["(~ccg_<s>", "~ccg_<s>:verdict_lock"])),
    "a selector with +@all": ("verdict", lambda t: sub(t, "+eval)", ["+eval", "+@all)"])),
    "an unclosed selector": ("verdict", lambda t: sub(t, "+eval)", ["+eval"])),
    "a nested selector": ("verdict", lambda t: t + ["(~ccg_<s>", "(+get)", ")"]),
    "a selector on a role that has none": ("engine", lambda t: t + ["(~ccg_<s>__ov_*", "+del)"]),
    "a third selector": ("build", lambda t: t + ["(~ccg_<s>:verdict_lock", "+get)"]),
    "a non-ASCII space token": ("engine", lambda t: t + ["+info +flushall"]),
}


@pytest.mark.parametrize("name", list(TAMPERS))
def test_a_widened_or_malformed_grant_is_refused_by_the_grammar_value_free(name):
    role, edit = TAMPERS[name]
    t = _with(role, edit)
    with pytest.raises(acl.AclError) as ei:
        acl.validate_template(t)
    msg = str(ei.value)
    assert role in msg and "SECRET123" not in msg and "evilpw" not in msg and "~*" not in msg
    admin = FakeAdmin()
    with pytest.raises(acl.AclError):
        acl.create_users(admin, "shop", template=t)
    assert admin.calls == []                                                           # zero server calls


@pytest.mark.parametrize("inject", [
    "ACL SETUSER ccg_engine_<s> on ><secret> resetkeys resetchannels -@all ~* +@all\n",                    # a duplicate user line
    "    ~* +@all\n", "    +flushall\n", "    allkeys allcommands nopass\n",                                  # continuation lines (newline injection)
])
def test_text_injected_into_the_lines_is_refused(inject):
    text = acl.DATA_FILE.read_text().rstrip("\n") + "\n" + inject
    with pytest.raises(acl.AclError):
        acl.validate_template(acl.parse_acl_lines(text))


def test_a_tampered_file_is_refused_when_read_from_a_path(tmp_path):
    p = tmp_path / "users.txt"
    p.write_text(acl.DATA_FILE.read_text().replace("~ccg_<s> ~ccg_<s>__ov_*", "~* ~ccg_<s>__ov_*"))
    with pytest.raises(acl.AclError, match="engine|raw"):
        acl.parse_acl_template(p)


def test_the_shipped_lines_and_every_narrower_subset_pass_the_grammar():
    t = acl.parse_acl_template()
    assert acl.validate_template(t) is t
    narrower = _with("engine", lambda toks: [x for x in toks if x != "+graph.explain"])
    assert acl.validate_template(narrower)


# ------------------------------------------------------------------------------------------------ setuser args and create_users

def test_setuser_args_reset_first_and_use_acl_user_and_fill_both_placeholders():
    user, toks = acl.setuser_args("build", "shop", "PW")
    assert user == sysname.acl_user("build", "shop") and toks[0] == "reset" and ">PW" in toks
    assert "%W~ccg_shop" in toks and not any("<" in t for t in toks)
    with pytest.raises(sysname.SystemNameError):
        acl.setuser_args("build", "shop_", "PW")
    with pytest.raises(ValueError):
        acl.setuser_args("admin", "shop", "PW")


@pytest.mark.parametrize("bad", ["shop_", "shop__x", "Shop", "", "a" * 49, "-shop", "sh op", "shop:x", "shop*"])
def test_create_users_refuses_a_bad_system_name_before_any_call_or_file(bad, creds):
    admin = FakeAdmin()
    with pytest.raises(sysname.SystemNameError):
        acl.create_users(admin, bad)
    assert admin.calls == [] and not creds.exists()


def test_create_users_resets_then_kills_sessions_per_role_then_writes_a_0600_file_in_a_0700_dir(creds):
    admin = FakeAdmin()
    users = acl.create_users(admin, "shop")
    assert users == [sysname.acl_user(r, "shop") for r in sysname.ROLES]
    assert [c[:3] for c in setusers(admin)] == [("ACL", "SETUSER", u) for u in users]
    assert all(c[3] == "reset" for c in setusers(admin))
    assert [c for c in admin.calls if c[0] == "CLIENT"] == [("CLIENT", "KILL", "USER", u) for u in users]
    assert admin.kinds() == [("ACL", "SETUSER"), ("CLIENT", "KILL")] * 4                  # each user: set, then its old sessions end
    creds_by_role = acl.read_credentials("shop")
    for role, call in zip(sysname.ROLES, setusers(admin)):
        sent, long_enough = f">{creds_by_role[role]}" in call, len(creds_by_role[role]) >= 43      # 256 bits, urlsafe base64
        assert sent and long_enough
    distinct = len(set(creds_by_role.values()))
    assert distinct == 4
    f = creds / "shop.acl"
    assert stat.S_IMODE(f.stat().st_mode) == 0o600 and stat.S_IMODE(creds.stat().st_mode) == 0o700
    same = f.read_bytes() == atrest.read_acl("shop")
    assert same


def test_a_second_call_rotates_every_password_and_resends_the_same_grants():
    a1, a2 = FakeAdmin(), FakeAdmin()
    acl.create_users(a1, "shop")
    first = acl.read_credentials("shop")
    acl.create_users(a2, "shop")
    second = acl.read_credentials("shop")
    rotated = all(first[r] != second[r] for r in sysname.ROLES)
    strip = lambda calls, pw: [[t for t in c if t != f">{pw[r]}"] for c, r in zip(setusers(calls), sysname.ROLES)]
    same_grants = strip(a1, first) == strip(a2, second)
    assert rotated and same_grants


def test_a_bad_credentials_directory_is_refused_before_the_first_setuser(creds, monkeypatch):
    """An unusable directory on a rotation must not reset live credentials it then cannot store: zero server calls."""
    monkeypatch.setenv("CCG_CREDENTIALS_DIR", str(atrest.REPO_ROOT / "cpg" / "x-creds"))
    admin = FakeAdmin()
    with pytest.raises(acl.AclError, match="credentials directory"):
        acl.create_users(admin, "shop")
    assert admin.calls == []


def test_a_credentials_dir_that_is_a_file_or_not_writable_is_refused_with_zero_calls(tmp_path, monkeypatch):
    f = tmp_path / "afile"
    f.write_text("x")
    monkeypatch.setenv("CCG_CREDENTIALS_DIR", str(f))
    admin = FakeAdmin()
    with pytest.raises(acl.AclError):
        acl.create_users(admin, "shop")
    ro = tmp_path / "ro"
    ro.mkdir(mode=0o700)
    os.chmod(ro, 0o500)
    monkeypatch.setenv("CCG_CREDENTIALS_DIR", str(ro))
    try:
        if os.access(ro, os.W_OK):
            pytest.skip("running as a user that ignores directory modes")
        with pytest.raises(acl.AclError):
            acl.create_users(admin, "shop")
    finally:
        os.chmod(ro, 0o700)
    assert admin.calls == []


def test_a_symlink_planted_at_the_credentials_file_is_refused_with_zero_calls(creds, tmp_path):
    creds.mkdir(mode=0o700)
    victim = tmp_path / "victim"
    victim.write_text("keep")
    (creds / "shop.acl").symlink_to(victim)
    admin = FakeAdmin()
    with pytest.raises(acl.AclError):
        acl.create_users(admin, "shop")
    assert admin.calls == [] and victim.read_text() == "keep"


def test_a_setuser_failure_disables_every_existing_user_kills_sessions_and_carries_no_secret(creds):
    acl.create_users(FakeAdmin(), "shop")                                              # a previous, working rotation: its file exists
    assert (creds / "shop.acl").exists()
    admin = FakeAdmin(fail_on=5)                                                      # set+kill for build and verdict (4 calls), then the engine's SETUSER fails
    with pytest.raises(acl.AclError) as ei:
        acl.create_users(admin, "shop")
    pw = {t[1:] for c in admin.calls[:5] for t in c if isinstance(t, str) and t.startswith(">")}
    text = " ".join([str(ei.value), repr(ei.value), repr(ei.value.__cause__), repr(ei.value.__context__)])
    leaked = any(p in text for p in pw)
    assert len(pw) == 3 and not leaked and "Syntax error" not in text and ei.value.__cause__ is None and ei.value.__context__ is None
    after = admin.calls[5:]
    assert [c[2] for c in after if c[:2] == ("ACL", "SETUSER")] == [sysname.acl_user(r, "shop") for r in sysname.ROLES]   # all four, incl. the unreached ones
    assert all(c[3:] == ("off", "resetpass") for c in after if c[:2] == ("ACL", "SETUSER"))
    assert [c[3] for c in after if c[0] == "CLIENT"] == [sysname.acl_user(r, "shop") for r in sysname.ROLES]
    assert not (creds / "shop.acl").exists()                                           # no stale file naming disabled users
    assert "the users are disabled" in str(ei.value)


def test_a_failed_first_creation_disables_only_users_that_exist():
    absent = {sysname.acl_user(r, "shop") for r in ("verdict", "engine", "raw")}
    admin = FakeAdmin(fail_on=1, absent=absent)
    with pytest.raises(acl.AclError):
        acl.create_users(admin, "shop")
    assert [c[2] for c in setusers(admin) if c[3:] == ("off", "resetpass")] == [sysname.acl_user("build", "shop")]


def test_a_keyboard_interrupt_mid_rotation_disables_the_users_then_propagates(creds):
    acl.create_users(FakeAdmin(), "shop")
    admin = FakeAdmin(fail_on=3, boom=KeyboardInterrupt)
    with pytest.raises(KeyboardInterrupt):
        acl.create_users(admin, "shop")
    assert len([c for c in setusers(admin) if c[3:] == ("off", "resetpass")]) == 4
    assert not (creds / "shop.acl").exists()


class SecondInterrupt(KeyboardInterrupt):
    """A second interrupt, told apart from the first by type: the first one must be the one that propagates."""


class StateAdmin:
    """A stateful fake: four users that exist and are ``on`` with live sessions (a previous rotation); SETUSER/KILL change that state. ``rules`` is a
    list of ``[match(args) -> bool, exc, when, times]``: the command raises ``exc`` ``times`` times, ``when`` ``before`` its effect or ``after`` it
    (an interrupt can land on either side of the server's reply)."""

    def __init__(self, system="shop", rules=()):
        self.users = {sysname.acl_user(r, system): True for r in sysname.ROLES}
        self.live = set(self.users)
        self.rules = [list(r) for r in rules]
        self.connection = self

    def execute_command(self, *args):
        hit = next((r for r in self.rules if r[3] > 0 and r[0](args)), None)
        if hit and hit[2] == "before":
            hit[3] -= 1
            raise hit[1]()
        if args[:2] == ("ACL", "SETUSER"):
            self.users[args[2]] = "off" not in args[3:]
        elif args[:2] == ("CLIENT", "KILL"):
            self.live.discard(args[3])
        if hit:
            hit[3] -= 1
            raise hit[1]()
        return [b"flags"] if args[:2] == ("ACL", "GETUSER") else b"OK"


def _first_setuser(role):
    return lambda a: a[:2] == ("ACL", "SETUSER") and a[2] == sysname.acl_user(role, "shop") and a[3] == "reset"


def _closing(kind, role):
    user = sysname.acl_user(role, "shop")
    return {"GETUSER": lambda a: a[:3] == ("ACL", "GETUSER", user),
            "off": lambda a: a[:2] == ("ACL", "SETUSER") and a[2] == user and a[3:] == ("off", "resetpass"),
            "KILL": lambda a: a[:4] == ("CLIENT", "KILL", "USER", user)}[kind]


def _assert_closed(admin, creds):
    assert not any(admin.users.values()), "a user is still on"
    assert not admin.live, "a session is still alive"
    assert not (creds / "shop.acl").exists()


@pytest.mark.parametrize("when", ["before", "after"])
@pytest.mark.parametrize("kind", ["GETUSER", "off", "KILL"])
@pytest.mark.parametrize("role", sysname.ROLES)
def test_a_second_interrupt_in_any_fail_closed_step_still_disables_everything_and_the_first_propagates(creds, role, kind, when):
    acl.create_users(FakeAdmin(), "shop")
    admin = StateAdmin(rules=[[_first_setuser("build"), KeyboardInterrupt, "after", 1], [_closing(kind, role), SecondInterrupt, when, 1]])
    with pytest.raises(KeyboardInterrupt) as ei:
        acl.create_users(admin, "shop")
    assert type(ei.value) is KeyboardInterrupt
    _assert_closed(admin, creds)


@pytest.mark.parametrize("when", ["before", "after"])
def test_a_second_interrupt_in_the_credentials_file_removal_still_disables_everything_and_the_first_propagates(creds, monkeypatch, when):
    acl.create_users(FakeAdmin(), "shop")
    real, left = atrest.remove_acl, [1]

    def hostile(system, keys_root=None):
        if left[0]:
            left[0] -= 1
            if when == "before":
                raise SecondInterrupt()
            real(system, keys_root)
            raise SecondInterrupt()
        return real(system, keys_root)

    monkeypatch.setattr(atrest, "remove_acl", hostile)
    admin = StateAdmin(rules=[[_first_setuser("build"), KeyboardInterrupt, "after", 1]])
    with pytest.raises(KeyboardInterrupt) as ei:
        acl.create_users(admin, "shop")
    assert type(ei.value) is KeyboardInterrupt
    _assert_closed(admin, creds)


def test_an_interrupt_in_the_first_normal_step_alone_disables_everything_and_propagates(creds):
    acl.create_users(FakeAdmin(), "shop")
    admin = StateAdmin(rules=[[_first_setuser("build"), KeyboardInterrupt, "after", 1]])
    with pytest.raises(KeyboardInterrupt):
        acl.create_users(admin, "shop")
    _assert_closed(admin, creds)


def test_an_interrupt_in_the_fail_closed_path_of_an_ordinary_failure_propagates_after_everything_is_disabled(creds):
    acl.create_users(FakeAdmin(), "shop")
    admin = StateAdmin(rules=[[_first_setuser("engine"), RuntimeError, "before", 1], [_closing("off", "build"), SecondInterrupt, "before", 1]])
    with pytest.raises(SecondInterrupt):
        acl.create_users(admin, "shop")
    _assert_closed(admin, creds)


def test_every_user_is_still_disabled_when_one_disabling_keeps_failing_and_the_error_says_so(creds):
    acl.create_users(FakeAdmin(), "shop")
    admin = StateAdmin(rules=[[_first_setuser("engine"), RuntimeError, "before", 1], [_closing("off", "build"), RuntimeError, "before", 99]])
    with pytest.raises(acl.AclError, match="DISABLING THEM ALSO FAILED"):
        acl.create_users(admin, "shop")
    assert [u for u, on in admin.users.items() if on] == [sysname.acl_user("build", "shop")]       # only the stubborn one; the loop went on
    assert not admin.live and not (creds / "shop.acl").exists()


def test_a_file_write_failure_disables_all_four_users_removes_nothing_stale_and_names_no_secret(monkeypatch, creds):
    admin = FakeAdmin()

    def boom(system, data, keys_root=None):
        raise OSError(f"disk full while writing {data.decode()}")

    monkeypatch.setattr(atrest, "write_acl", boom)
    with pytest.raises(acl.AclError) as ei:
        acl.create_users(admin, "shop")
    pw = {t[1:] for c in setusers(admin)[:4] for t in c if isinstance(t, str) and t.startswith(">")}
    text = " ".join([str(ei.value), repr(ei.value), repr(ei.value.__cause__)])
    leaked = any(p in text for p in pw)
    assert len(pw) == 4 and not leaked and "disk full" not in text
    assert len([c for c in setusers(admin) if c[3:] == ("off", "resetpass")]) == 4


def test_a_disabling_that_itself_fails_is_reported_without_a_secret():
    class Stubborn(FakeAdmin):
        def execute_command(self, *args):
            r = super().execute_command(*args)
            if len(self.calls) > 1 and "off" in args:
                raise RuntimeError("cannot")
            return r

    admin = Stubborn(fail_on=1)
    with pytest.raises(acl.AclError, match="DISABLING THEM ALSO FAILED"):
        acl.create_users(admin, "shop")


def test_a_stale_credentials_temp_of_a_killed_write_is_swept_and_other_files_are_not(creds):
    creds.mkdir(mode=0o700)
    old, young, other = creds / ".shop.acl.deadbeef.tmp", creds / ".shop.acl.cafebabe.tmp", creds / "notes.txt"
    for f in (old, young, other):
        f.write_text("pw")
        os.chmod(f, 0o600)
    t = time.time() - 7200
    os.utime(old, (t, t))
    os.utime(other, (t, t))
    acl.create_users(FakeAdmin(), "shop")
    assert not old.exists() and young.exists() and other.exists()


def test_a_killed_write_leaves_a_temp_that_the_atrest_sweep_removes_by_name_only(creds):
    creds.mkdir(mode=0o700)
    tmp = creds / ".rot.acl.0123abcd.tmp"
    tmp.write_text("pw")
    os.chmod(tmp, 0o600)
    now = time.time() + 7200
    assert atrest.sweep_stale_acl_tmp(now=now) == [tmp] and not tmp.exists()


def test_the_temp_name_write_acl_uses_is_the_one_the_sweep_matches(monkeypatch, creds):
    seen = []
    real = os.replace
    monkeypatch.setattr(os, "replace", lambda a, b: (seen.append(Path(a).name), real(a, b))[1])
    atrest.write_acl("shop", b"x\n")
    assert atrest._ACL_TMP_RE.fullmatch(seen[0])


def test_no_password_reaches_stdout_stderr_or_the_log(capsys, caplog, creds):
    caplog.set_level(logging.DEBUG)
    acl.create_users(FakeAdmin(), "shop")
    pws = set(acl.read_credentials("shop").values())
    with pytest.raises(acl.AclError):
        acl.create_users(FakeAdmin(fail_on=2), "shop")
    out, err = capsys.readouterr()
    blob = out + err + caplog.text
    leaked = any(p in blob for p in pws)
    assert not leaked and out == "" and err == ""


# ------------------------------------------------------------------------------------------------ reading the file back

def test_read_credentials_refuses_a_malformed_or_foreign_file(creds):
    atrest.write_acl("shop", b"ccg_build_shop pw1\n")
    with pytest.raises(acl.AclError):
        acl.read_credentials("shop")
    atrest.write_acl("shop", "".join(f"{sysname.acl_user(r, 'shop2')} pw{i}\n" for i, r in enumerate(sysname.ROLES)).encode())
    with pytest.raises(acl.AclError):
        acl.read_credentials("shop")


def test_a_hardlinked_credentials_file_is_refused_on_read(creds, tmp_path):
    acl.create_users(FakeAdmin(), "shop")
    os.link(creds / "shop.acl", tmp_path / "second-name")
    with pytest.raises(atrest.AtRestError, match="hard link"):
        atrest.read_acl("shop")
    with pytest.raises(atrest.AtRestError):
        acl.read_credentials("shop")


def test_a_single_link_file_still_reads(creds):
    acl.create_users(FakeAdmin(), "shop")
    assert set(acl.read_credentials("shop")) == set(sysname.ROLES)


def test_nothing_is_read_or_written_under_the_default_key_area(creds):
    assert not str(atrest.credentials_dir()).startswith(str(Path.home() / ".config" / "ccg"))
    acl.create_users(FakeAdmin(), "shop")
    assert (creds / "shop.acl").exists()


def test_a_test_that_forgets_the_credentials_env_var_fails_instead_of_writing(monkeypatch):
    """The default area is ``~/.config/ccg``: with the override removed, ``create_users`` must refuse (the ``conftest.py`` guard) with zero server calls."""
    monkeypatch.delenv("CCG_CREDENTIALS_DIR")
    before = sorted(p.name for p in (Path.home() / ".config" / "ccg" / "keys").glob("*")) if (Path.home() / ".config" / "ccg" / "keys").exists() else []
    admin = FakeAdmin()
    with pytest.raises(acl.AclError, match="credentials directory"):
        acl.create_users(admin, "shop")
    assert admin.calls == []
    after = sorted(p.name for p in (Path.home() / ".config" / "ccg" / "keys").glob("*")) if (Path.home() / ".config" / "ccg" / "keys").exists() else []
    assert before == after
