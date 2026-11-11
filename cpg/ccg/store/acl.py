"""The four per-system ACL users of the connected-code-graph store (C-1001 P1-19 dispatch 2).

Contract: ``docs/plans/connected-code-graph-p1-graph.md`` §7 (the user lines, which supersede the flat ``~ccg_<system>*`` form) and §8
P1-19 (c) (the grant matrix ``tests/test_acl_live.py`` asserts, the oracle for the grants).

**Source and ceiling.** The ``ACL SETUSER`` lines live in ONE data file next to the code, ``acl_users.txt`` (:func:`parse_acl_template`);
``tests/test_acl_unit.py`` asserts the plan's §7 block equals it, so editing the plan fails a test instead of silently changing production
grants. Whatever the source, every template is checked against a CLOSED GRAMMAR in this module (:func:`validate_template`) before any server
call: a fixed head, then per role only the commands, key patterns and selectors of that role's CEILING, no password token but the one
``><secret>``, no ``nopass``/``allkeys``/``allcommands``, no ``+@category``, no ``&`` channel, no ``~*``. Anything outside it raises
:class:`AclError` naming the role and the rule, never the token (a token could carry a secret). A grant is therefore narrower than or equal
to the ceiling by construction, and the live matrix decides whether it is exactly right.

**Semantics of** :func:`create_users` **(idempotent in grants, rotating in secrets).** In this order:
1. the system name, the template and the credentials directory are checked (``atrest.prepare_acl_target``: not in the repository, private, no
   link, writable): a bad one raises :class:`AclError` with ZERO server calls, so a configuration typo cannot reset live credentials;
2. a stale ``.<system>.acl.<hex>.tmp`` of a killed earlier write is swept (``atrest.sweep_stale_acl_tmp``);
3. for each role: a fresh password (``secrets.token_urlsafe``, 256 bits), ONE atomic ``ACL SETUSER <user> reset <tokens>`` (``reset`` first:
   without it ``>secret`` would ADD a password and a hand-added grant or the old password would survive), then ``CLIENT KILL USER <user>``
   on the admin connection (Redis does not end an authenticated session on a password change);
4. only after all four succeed, ``<user> <password>`` lines are written through ``atrest.write_acl`` (0600 in a 0700 directory;
   ``$CCG_CREDENTIALS_DIR`` or ``~/.config/ccg/keys``), replacing the previous file.
A second call is a **rotation**: new passwords, old ones dead for new AND established connections. **Fail closed, on any ``BaseException``
(a ``KeyboardInterrupt`` included):** every one of the four users that exists is set ``off resetpass`` and its sessions killed, and the
credentials file is removed, EVERY step tried (twice) even when an earlier one raised or was interrupted, so no user is left holding a password
nobody was told and no file names disabled users; then the original exception (or, for an ordinary failure, the first interrupt met while closing) is re-raised (an :class:`AclError` for an ordinary failure, value-free: user name, exception class, and whether the disabling itself
succeeded). A ``kill -9`` cannot run that path; it leaves, at worst, users with new passwords and the old file (re-run ``create_users``) and
possibly a 0600 temp that the next call sweeps. Two concurrent calls for one system are not serialised (the last writer wins both the users
and the file; run one at a time). Passwords go nowhere else: not stdout, not a log, not a return value, not a committed file.
"""
from __future__ import annotations

import re
import secrets
from pathlib import Path

from ccg import atrest, sysname

DATA_FILE = Path(__file__).with_name("acl_users.txt")
SECRET_BYTES = 32
_PLACEHOLDER_SYSTEM = "zz"          # a valid system name used to read a template user's role back through ``sysname.acl_user_re``


class AclError(RuntimeError):
    """A template outside the grammar, an unusable credentials directory, or a user that could not be created. Never carries a secret."""


def parse_acl_lines(text: str) -> dict[str, list[str]]:
    """``{role: argument tokens}`` of ``ACL SETUSER`` lines in ``text`` (a continuation line joins the previous user's tokens), ``<s>`` and
    ``<secret>`` still in place. The user-name token is not part of the tokens: it is re-formed by ``sysname.acl_user``. Shape only: the
    grammar is :func:`validate_template`."""
    entries: list[list[str]] = []
    for line in text.splitlines():
        if line.startswith("ACL SETUSER "):
            entries.append(line[len("ACL SETUSER "):].split())
        elif line.strip() and entries:
            entries[-1] += line.split()
    template: dict[str, list[str]] = {}
    for entry in entries:
        m = sysname.acl_user_re().match(entry[0].replace("<s>", _PLACEHOLDER_SYSTEM))
        if not m or entry[0].replace("<s>", _PLACEHOLDER_SYSTEM) != sysname.acl_user(m["role"], _PLACEHOLDER_SYSTEM) or m["role"] in template:
            raise AclError("the ACL lines have a user line that is not ccg_<role>_<s>, or one twice")
        template[m["role"]] = entry[1:]
    if tuple(template) != sysname.ROLES:
        raise AclError(f"the ACL lines do not list exactly the roles {sysname.ROLES}, in order")
    return template


# ---------------------------------------------------------------------------------------------------- the closed grammar
_HEAD = ["on", "><secret>", "resetkeys", "resetchannels", "-@all"]
_LOCK = ({"~ccg_<s>:verdict_lock"}, {"+set", "+get", "+del", "+eval"})
#: per role: root commands, root key patterns, selectors (key patterns, commands): each a CEILING (a subset is accepted), never a grant list to copy.
_CEILING = {
    "build": ({"+graph.query", "+graph.ro_query", "+graph.explain", "+graph.constraint", "+graph.config|get", "+rename", "+exists", "+info"},
              {"%W~ccg_<s>", "~ccg_<s>__*"}, [_LOCK, ({"~ccg_<s>__build_*", "~ccg_<s>__ov_*"}, {"+del", "+expire"})]),
    "verdict": ({"+graph.query", "+info"}, {"~ccg_<s>"}, [_LOCK]),
    "engine": ({"+graph.ro_query", "+graph.explain", "+info"}, {"~ccg_<s>", "~ccg_<s>__ov_*"}, []),
    "raw": ({"+graph.ro_query", "+graph.explain", "+info"}, {"~ccg_<s>", "~ccg_<s>__ov_*"}, []),
}


def _rule(token: str, kind: str) -> str:
    """The rule a rejected token broke, by token shape only (the text of the token is never returned)."""
    if not token.isascii():
        return "a non-ASCII token"
    low = token.lower()
    if low[:1] in "><#!" or low == "nopass" or low == "resetpass":
        return "a password token"
    if low in ("allkeys", "allcommands", "allchannels") or low[:2] in ("+@", "-@") or low.startswith(("-", "reset", "clearselectors", "sanitize", "skip")):
        return "a category, reset or blanket-access token"
    if low[:1] == "&":
        return "a channel token"
    if "~*" in low or low in ("~*", "%rw~*"):
        return "a catch-all key pattern"
    return f"a {kind} outside the ceiling"


def _check_selector(role: str, toks: list[str], specs: list) -> None:
    keys = [t for t in toks if t.startswith(("~", "%"))]
    cmds = [t for t in toks if t.startswith("+")]
    other = [t for t in toks if t not in keys and t not in cmds]
    for t in other:
        raise AclError(f"§7 grant for {role}: {_rule(t, 'selector token')}")
    for i, (skeys, scmds) in enumerate(specs):
        if keys and set(keys) <= skeys:
            for t in cmds:
                if t not in scmds:
                    raise AclError(f"§7 grant for {role}: {_rule(t, 'selector command')}")
            if len(set(keys)) != len(keys) or len(set(cmds)) != len(cmds):
                raise AclError(f"§7 grant for {role}: a repeated selector token")
            specs.pop(i)
            return
    raise AclError(f"§7 grant for {role}: a selector outside the ceiling")


def validate_template(template: dict[str, list[str]]) -> dict[str, list[str]]:
    """Check ``template`` (``{role: tokens}``, placeholders in place) against the closed grammar; return it, or raise :class:`AclError`.
    Case-sensitive and ASCII-only on purpose: Redis reads ``+GRAPH.QUERY`` as ``+graph.query``, so only the one lowercase spelling is accepted."""
    if not isinstance(template, dict) or tuple(template) != sysname.ROLES:
        raise AclError(f"a template holds exactly the roles {sysname.ROLES}, in order")
    for role, toks in template.items():
        cmds_ok, keys_ok, selectors = _CEILING[role]
        specs = list(selectors)
        if not isinstance(toks, list) or not all(isinstance(t, str) for t in toks):
            raise AclError(f"§7 grant for {role}: tokens are strings")
        if toks[:len(_HEAD)] != _HEAD:
            raise AclError(f"§7 grant for {role}: the head is not the fixed on / one password / resetkeys / resetchannels / -@all")
        rest, seen, i = toks[len(_HEAD):], set(), 0
        while i < len(rest):
            t = rest[i]
            if t.startswith("("):                                   # a selector: collect to the closing token
                group = []
                while True:
                    if i >= len(rest):
                        raise AclError(f"§7 grant for {role}: an unclosed selector")
                    group.append(rest[i])
                    if rest[i].endswith(")"):
                        break
                    i += 1
                group[0], group[-1] = group[0][1:], group[-1][:-1]
                if any("(" in g or ")" in g for g in group) or not all(g.isascii() for g in group):
                    raise AclError(f"§7 grant for {role}: a nested or malformed selector")
                _check_selector(role, [g for g in group if g], specs)
            else:
                if not t.isascii() or t in seen:
                    raise AclError(f"§7 grant for {role}: {'a non-ASCII token' if not t.isascii() else 'a repeated token'}")
                seen.add(t)
                if t.startswith("+") and not t.startswith("+@"):
                    if t not in cmds_ok:
                        raise AclError(f"§7 grant for {role}: {_rule(t, 'command')}")
                elif t.startswith(("~", "%")):
                    if t not in keys_ok:
                        raise AclError(f"§7 grant for {role}: {_rule(t, 'key pattern')}")
                else:
                    raise AclError(f"§7 grant for {role}: {_rule(t, 'token')}")
            i += 1
    return template


def parse_acl_template(path: Path | None = None) -> dict[str, list[str]]:
    """The validated template of ``acl_users.txt`` (or of ``path``, a file of the same line shape)."""
    src = DATA_FILE if path is None else Path(path)
    try:
        text = src.read_text()
    except OSError:
        raise AclError(f"the ACL lines are not readable in {src.name}") from None
    return validate_template(parse_acl_lines(text))


def setuser_args(role: str, system: str, password: str, template: dict[str, list[str]] | None = None) -> tuple[str, list[str]]:
    """``(user, [rule tokens after the user name])`` for ``ACL SETUSER``: ``reset`` then the template's tokens for ``system`` with the password
    filled. The template is validated (:func:`validate_template`) whether supplied or read."""
    system = sysname.check_system_name(system)
    user = sysname.acl_user(role, system)
    toks = (parse_acl_template() if template is None else validate_template(template))[role]
    return user, ["reset"] + [t.replace("<s>", system).replace("<secret>", password) for t in toks]


def new_password() -> str:
    return secrets.token_urlsafe(SECRET_BYTES)


def _attempt(fn, *args) -> str | None:
    """Run ``fn``; return the exception CLASS NAME on failure, else None. The exception object is dropped here, so no raise below chains to
    it (``__context__`` stays None): its text may hold the ``>secret`` modifier Redis echoes."""
    try:
        fn(*args)
    except Exception as exc:  # noqa: BLE001
        return type(exc).__name__
    return None


def _setuser(admin, user: str, tokens: list[str]) -> None:
    admin.connection.execute_command("ACL", "SETUSER", user, *tokens)


def _kill_sessions(admin, user: str) -> None:
    admin.connection.execute_command("CLIENT", "KILL", "USER", user)


def _closing_step(fn, *args) -> tuple[bool, BaseException | None]:
    """One fail-closed step, tried twice, catching ``BaseException`` (a second ``KeyboardInterrupt`` lands here, not past it). Returns
    ``(succeeded, the first BaseException raised or None)``; the exception is kept only for the caller to re-raise, never read."""
    first: BaseException | None = None
    for _ in range(2):
        try:
            fn(*args)
        except BaseException as exc:    # noqa: BLE001
            first = first or exc
        else:
            return True, first
    return False, first


def _fail_closed(admin, system: str) -> tuple[bool, BaseException | None]:
    """Disable every one of the four users that exists (``off resetpass``), kill its sessions, drop the credentials file: EVERY step runs even
    when an earlier one raised or was interrupted (each is tried twice, ``BaseException`` included). Returns ``(all of it worked, the first
    BaseException raised inside)``; it never raises, the caller re-raises that exception once everything has been tried."""
    ok, first = True, None
    for role in sysname.ROLES:
        user = sysname.acl_user(role, system)
        try:
            exists = admin.connection.execute_command("ACL", "GETUSER", user)
        except BaseException as exc:    # noqa: BLE001
            exists, ok, first = True, False, first or exc
        if exists is None:
            continue
        for step, args in ((_setuser, (admin, user, ["off", "resetpass"])), (_kill_sessions, (admin, user))):
            done, exc = _closing_step(step, *args)
            ok, first = ok and done, first or exc
    done, exc = _closing_step(atrest.remove_acl, system)
    return ok and done, first or exc


def create_users(admin, system: str, *, template: dict[str, list[str]] | None = None) -> list[str]:
    """Create or reset the four users of ``system`` through the ``admin`` connection (a falkordb-py client authenticated as the instance's
    admin) and write their passwords to ``ccg.atrest.write_acl``. Returns the four user names (never a password). See the module docstring
    for the order, the rotation and the fail-closed semantics."""
    system = sysname.check_system_name(system)            # before the template is read and before any key or user exists
    template = parse_acl_template() if template is None else validate_template(template)
    if (bad := _attempt(atrest.prepare_acl_target, system)) is not None:
        raise AclError(f"the credentials directory for {system!r} is unusable ({bad}); no user was touched")
    _attempt(atrest.sweep_stale_acl_tmp)
    passwords: dict[str, str] = {}
    problem: str | None = None
    try:
        for role in sysname.ROLES:
            passwords[role] = new_password()
            user, tokens = setuser_args(role, system, passwords[role], template)
            if (bad := _attempt(_setuser, admin, user, tokens)) is not None:
                problem = f"ACL SETUSER {user} failed ({bad})"
                break
            if (bad := _attempt(_kill_sessions, admin, user)) is not None:
                problem = f"CLIENT KILL USER {user} failed ({bad})"
                break
        else:
            data = "".join(f"{sysname.acl_user(role, system)} {passwords[role]}\n" for role in sysname.ROLES).encode()
            if (bad := _attempt(atrest.write_acl, system, data)) is not None:
                problem = f"the ACL credentials file for {system!r} was not written ({bad})"
    except BaseException:
        _fail_closed(admin, system)                       # a KeyboardInterrupt or SystemExit: disable first, then let THIS one propagate
        raise
    finally:
        passwords.clear()
    if problem is not None:
        closed, interrupt = _fail_closed(admin, system)
        if interrupt is not None and not isinstance(interrupt, Exception):
            raise interrupt                                # an interrupt inside the closing: everything was tried first, now it propagates
        raise AclError(problem + ("; the users are disabled" if closed else "; DISABLING THEM ALSO FAILED: disable them by hand"))
    return [sysname.acl_user(r, system) for r in sysname.ROLES]


_LINE_RE = re.compile(r"\A(?P<user>\S+) (?P<password>\S+)\Z")


def read_credentials(system: str) -> dict[str, str]:
    """``{role: password}`` read back through ``ccg.atrest.read_acl`` (the clients' accessor); the file must hold exactly the four users."""
    system = sysname.check_system_name(system)
    out: dict[str, str] = {}
    for line in atrest.read_acl(system).decode("ascii", "replace").splitlines():
        m = _LINE_RE.match(line)
        m2 = sysname.acl_user_re().match(m["user"]) if m else None
        if not m2 or m2["system"] != system or m2["role"] in out:
            raise AclError(f"the ACL credentials file of {system!r} is malformed")
        out[m2["role"]] = m["password"]
    if tuple(out) != sysname.ROLES:
        raise AclError(f"the ACL credentials file of {system!r} does not hold exactly the four roles")
    return out
