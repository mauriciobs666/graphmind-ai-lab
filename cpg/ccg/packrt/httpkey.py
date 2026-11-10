"""The ``http`` ``keyNorm`` steps that precede step 8a, for the pack runtime (bundle spec3 §6 steps 2 and 4-8, §1.8 8a).

``ccg.keynorm`` owns step 8a (:func:`ccg.keynorm.mask_http_segments`) and this module calls it, never copies it. The steps
here are the RFC 3986 order (F-13): percent-decode unreserved characters, then collapse ``/`` runs and resolve dot
segments; the old order (dots first) is not implemented anywhere. Step 3 (mount composition) is the caller's.
"""
from __future__ import annotations

import re
import unicodedata

from ccg.keynorm import CONSUMER, PROVIDER, mask_http_segments

VERBS = frozenset({"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "TRACE", "CONNECT"})
UNKNOWN = "\x00"  # an unresolved value piece inside a string under construction

_UNRESERVED = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~")
_ESCAPE = re.compile(r"%([0-9A-Fa-f]{2})")
_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]*://")
_FIXED = frozenset({"{}", "{**}", CONSUMER, PROVIDER})
_WHOLE_PARAM = (
    (re.compile(r"^\{[^{}/]*:path\}$"), "{**}"),
    (re.compile(r"^\{\*[^{}/]*\}$"), "{**}"),
    (re.compile(r"^\[\.\.\.[^\[\]/]*\]$"), "{**}"),
    (re.compile(r"^<path:[^<>/]*>$"), "{**}"),
    (re.compile(r"^\*$"), "{**}"),
    (re.compile(r"^\{[^{}/]*\}$"), "{}"),
    (re.compile(r"^:[A-Za-z_][A-Za-z0-9_]*$"), "{}"),
    (re.compile(r"^<[^<>/]*>$"), "{}"),
    (re.compile(r"^\[[^\[\]/]+\]$"), "{}"),
)


def _decode_unreserved(path: str) -> str:
    def one(m: re.Match) -> str:
        ch = chr(int(m.group(1), 16))
        return ch if ch in _UNRESERVED else "%" + m.group(1).upper()

    return _ESCAPE.sub(one, path)


def _collapse_and_resolve(path: str) -> str:
    """Step 6: collapse ``/`` runs, then RFC 3986 §5.2.4 dot-segment removal (a trailing ``.``/``..`` ends in ``/``)."""
    parts = re.sub(r"/{2,}", "/", path).split("/")[1:]
    out: list[str] = []
    last = len(parts) - 1
    for i, seg in enumerate(parts):
        if seg == ".":
            if i == last:
                out.append("")
        elif seg == "..":
            if out:
                out.pop()
            if i == last:
                out.append("")
        else:
            out.append(seg)
    return "/" + "/".join(out)


def normalize_path_rfc(path: str) -> str:
    """Steps 4-6 only (rule-pack spec §1.3 ``normalize_path``): leading ``/``, decode unreserved, collapse and resolve."""
    path = path.strip()
    if not path.startswith("/"):
        path = "/" + path
    return _collapse_and_resolve(_decode_unreserved(path))


def _param_segment(seg: str) -> str:
    if seg in _FIXED:
        return seg
    for rx, repl in _WHOLE_PARAM:
        if rx.match(seg):
            return repl
    return seg


def normalize_path(path: str, placeholder: str) -> str:
    """Steps 4-7, the consumer step 8 (an ``UNKNOWN``-bearing segment becomes ``{?}``), then 8a."""
    segs = [_param_segment(s) for s in normalize_path_rfc(path).split("/")]
    segs = [CONSUMER if UNKNOWN in s else s for s in segs]
    return mask_http_segments("/".join(segs), placeholder)


def url_path_of(url: str) -> str:
    """Step 2: the path only; scheme and authority, query and fragment are dropped."""
    m = _SCHEME.match(url)
    if m:
        rest = url[m.end():]
        slash = min((i for i in (rest.find("/"), rest.find("?"), rest.find("#")) if i >= 0), default=-1)
        url = "" if slash < 0 else rest[slash:]
    for ch in "?#":
        cut = url.find(ch)
        if cut >= 0:
            url = url[:cut]
    return url


def normalize_key(raw: str, form: str = "provider") -> str:
    """``keyNorm`` of ``"<verb> <path>"``; a string with no leading verb word is a bare path with the any-method verb ``*``.

    ``form`` is ``provider`` (``{#}``) or ``consumer`` (``{?}``). The consumer form first takes the URL's path (step 2), the
    provider form does not (a route is a path already).
    """
    if form not in ("provider", "consumer"):
        raise ValueError(f"form must be 'provider' or 'consumer', got {form!r}")
    raw = unicodedata.normalize("NFC", raw).strip()
    verb, sep, rest = raw.partition(" ")
    if sep and not raw.startswith("/") and (verb.upper() in VERBS or verb == "*"):
        v, path = verb.upper(), rest
    else:
        v, path = "*", raw
    if form == "consumer":
        path = url_path_of(path)
    return f"{v} {normalize_path(path, PROVIDER if form == 'provider' else CONSUMER)}"
