"""Walk states and access paths (C-1001 P1-22; ``docs/plans/connected-code-graph-p1-answers.md`` P1-22 Files).

A frontier entry is ``(label, id, state kind, state path)`` (``-p0-graph.md`` §3.1, the visited-set key). The kinds are the closed six below; ``function`` is the
state of a function reached only at call level (``-answer-spec2.md`` §5). SEAM(P1-23): the rows that enter and leave ``function`` in a flag-false unit are built
there; the shipped P1-22 table never enters it (every unit is walked with the precise rows), only a row-table variant can.

Access paths are the grammar of ``-bundle-spec3.md`` §1.1/§5.1: ``a<i>``, ``recv``, ``kw.<name>`` (an argument), ``p<i>``, ``recv``, ``r`` (a definition),
``ret``, each followed by fields (``.ident`` or ``["json"]``).
"""
from __future__ import annotations

import json
import re

VALUE = ("value", "")
CONTRACT = ("contract", "")
KEY = ("key", "")
END = ("end", "")
FUNCTION = ("function", "")
SITE = "site"   # a Link reached from a config entry: path is the siteId it binds
KINDS = ("value", "path", "contract", "key", "end", "site", "function")

State = tuple[str, str]  # (kind, path)
Node = tuple[str, str]  # (label, id)
Key = tuple[str, str, str, str]  # (label, id, state kind, state path)


def path_state(path: str) -> State:
    return ("path", path)


def site_state(site_id: str) -> State:
    """The state of a ``Link`` entered from a config entry that binds ``site_id``: ``VIA <-`` from it lists only that site's ``CallSite``."""
    return ("site", site_id)


def path_overlaps(state_path: str, edge_path: str) -> bool:
    """Whether a change at ``state_path`` and a use of ``edge_path`` touch the same value: equal, or one a segment prefix of the other.

    ``p0`` covers ``p0.path``; ``p1`` is not a prefix of ``p10``; ``p0.path`` does not touch ``p0.method``.
    """
    if state_path == edge_path:
        return True
    shorter, longer = sorted((state_path, edge_path), key=len)
    return longer.startswith(shorter) and longer[len(shorter)] in ".["


def data_edge_matches(state: State, from_path: str) -> bool:
    """``value`` (the whole point changed) matches every ``DATA`` edge; a ``path`` state only an overlapping one."""
    return state[0] == "value" or path_overlaps(state[1], from_path)


_POSITIONAL = re.compile(r"a(0|[1-9][0-9]*)(?=$|[.\[])")
_RECEIVER = re.compile(r"recv(?=$|[.\[])")
_KW_IDENT = re.compile(r"kw\.([A-Za-z_$][A-Za-z0-9_$]*)(?=$|[.\[])")
_KW_QUOTED = re.compile(r'kw\[("(?:[^"\\]|\\.)*")\](?=$|[.\[])')


def param_path(arg_path: str, index: int, name: str | None) -> str | None:
    """The definition path, on the callee's ``Param`` (``index``, ``name``), of an argument path of a call site; ``None`` when the argument is not that parameter.

    ``a<i>`` names the parameter of index ``i``; ``recv`` the receiver parameter (index ``-1``); ``kw.<n>`` the parameter called ``n``. The fields after the head
    carry over: ``a0.q`` is ``p0.q``, ``recv.f`` is ``recv.f``. Never raises on a malformed path.
    """
    if not isinstance(arg_path, str) or isinstance(index, bool) or not isinstance(index, int):
        return None
    m = _POSITIONAL.match(arg_path)
    if m:
        return f"p{index}{arg_path[m.end():]}" if index == int(m.group(1)) else None
    m = _RECEIVER.match(arg_path)
    if m:
        return arg_path if index == -1 else None
    if index < 0 or name is None:
        return None
    m = _KW_IDENT.match(arg_path)
    if m:
        kw = m.group(1)
    else:
        m = _KW_QUOTED.match(arg_path)
        if not m:
            return None
        try:
            kw = json.loads(m.group(1))
        except ValueError:
            return None
    return f"p{index}{arg_path[m.end():]}" if kw == name else None


def is_arg_path(path: str) -> bool:
    """Whether ``path`` names an argument of a call site (``a<i>``, ``recv``, ``kw.<n>``, with fields): the only paths that enter a callee. ``r`` (the call's
    result), ``ret`` and a ``p<i>`` definition path are not arguments."""
    return isinstance(path, str) and bool(_POSITIONAL.match(path) or _RECEIVER.match(path) or _KW_IDENT.match(path) or _KW_QUOTED.match(path))
