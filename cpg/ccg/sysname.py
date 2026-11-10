"""System names, store keys and ACL user names: the one module that forms them (C-1001 P1-03).

Spec: ``docs/plans/connected-code-graph-p1-graph.md`` §7. A system name is lowercase ASCII letters and digits with
single ``-`` or ``_`` separators, none leading, trailing or doubled, 1-48 characters. That shape is what keeps the
ACL key patterns ``~ccg_<s>`` and ``~ccg_<s>__*`` of one system from matching a key of another, and it excludes
every glob metacharacter and ``:``. No other module formats ``ccg_`` key or user text (a sweep in
``cpg/tests/test_sysname.py`` fails if one does).
"""
from __future__ import annotations

import re

MAX_SYSTEM_NAME_LENGTH = 48
ROLES = ("build", "verdict", "engine", "raw")

#: The name pattern without anchors, so `acl_user_re` embeds it and no second literal of it exists.
SYSTEM_NAME_BODY = r"[a-z0-9]+(?:[-_][a-z0-9]+)*"
#: Anchored with `\A…\Z`: `$` would also match before a trailing newline.
SYSTEM_NAME_RE = re.compile(rf"\A{SYSTEM_NAME_BODY}\Z")

# A build or overlay id is one token of the key's last segment: alphanumerics and single `-`, so it can neither
# end the `ccg_<s>__build_` prefix early nor carry a glob metacharacter, `:` or a second `__`.
_ID_RE = re.compile(r"\A[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*\Z")


class SystemNameError(ValueError):
    """A system name outside the whitelist."""


def check_system_name(name: str) -> str:
    """Return ``name`` if it is a valid system name, else raise `SystemNameError`."""
    if not isinstance(name, str):
        raise SystemNameError(f"a system name is a str, got {type(name).__name__}")
    if len(name) > MAX_SYSTEM_NAME_LENGTH:
        raise SystemNameError(f"a system name is at most {MAX_SYSTEM_NAME_LENGTH} characters, got {len(name)}")
    if not SYSTEM_NAME_RE.match(name):
        raise SystemNameError(
            f"invalid system name {name!r}: lowercase letters and digits with single '-' or '_' separators, "
            "none leading, trailing or doubled"
        )
    return name


def _token(kind: str, value: str) -> str:
    if not isinstance(value, str) or not _ID_RE.match(value):
        raise ValueError(f"invalid {kind} {value!r}: alphanumerics with single '-' separators")
    return value


def live_key(system: str) -> str:
    return f"ccg_{check_system_name(system)}"


def build_key(system: str, snapshot_id: str) -> str:
    return f"ccg_{check_system_name(system)}__build_{_token('snapshot id', snapshot_id)}"


def overlay_key(system: str, ov_id: str) -> str:
    return f"ccg_{check_system_name(system)}__ov_{_token('overlay id', ov_id)}"


def lock_key(system: str) -> str:
    return f"ccg_{check_system_name(system)}:verdict_lock"


def acl_user(role: str, system: str) -> str:
    if role not in ROLES:
        raise ValueError(f"unknown role {role!r}; one of {ROLES}")
    return f"ccg_{role}_{check_system_name(system)}"


def acl_user_re() -> re.Pattern[str]:
    """The ACL user pattern, groups ``role`` and ``system``. It does not hold the length bound; check that apart."""
    return re.compile(rf"\Accg_(?P<role>{'|'.join(ROLES)})_(?P<system>{SYSTEM_NAME_BODY})\Z")
