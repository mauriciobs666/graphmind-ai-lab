"""The ``http`` ``keyNorm`` steps that precede step 8a, for the pack runtime (bundle spec3 §6 steps 2 and 4-8, §1.8 8a).

``ccg.keynorm`` owns step 8a (:func:`ccg.keynorm.mask_http_segments`) and this module calls it, never copies it. The steps
here are the RFC 3986 order (F-13): percent-decode unreserved characters, then collapse ``/`` runs and resolve dot
segments; the old order (dots first) is not implemented anywhere. Step 3 (mount composition) is the caller's.

The no-literal-segment rule (P1-55, :func:`names_no_literal_segment`) is the caller's: such a consumer key is keyless.

The unknown-origin key rule (P1-49; rule-pack spec §1.4 and ``docs/plans/connected-code-graph-p1-linking.md`` P1-49 rules 1-3):
a value piece that is not known is a marker, ``UNKNOWN`` (``\\x00``: a name that should have resolved) or ``UNKNOWN_VALUE``
(``\\x01``: a value not statically known). :func:`split_url` drops an unresolved origin from a consumer URL and says so;
:func:`normalize_key_parts` reports it with whether the *path* holds an unresolved piece, and the caller grants ``confirmed``
only when neither is true. :func:`is_wholly_unresolved` (one marker: an alternative that is unknown as a whole) and
:func:`has_unresolved` (a marker anywhere: a string under construction) are the two ways a site asks "unresolved?".
"""
from __future__ import annotations

import re
import unicodedata
from typing import NamedTuple

from ccg.keynorm import CONSUMER, PROVIDER, mask_http_segments

VERBS = frozenset({"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "TRACE", "CONNECT"})
UNKNOWN = "\x00"  # a name that should resolve and did not (an import the resolver could not map, a cycle, the depth cap)
UNKNOWN_VALUE = "\x01"  # a value that is not statically known (parameter, environment read, call result, the alternatives cap)

_UNRESERVED = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~")
_ESCAPE = re.compile(r"%([0-9A-Fa-f]{2})")
_AUTHORITY = re.compile(r"^(?:[A-Za-z][A-Za-z0-9+.\-]*:)?//")     # `scheme://` or a scheme-relative `//`
_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]*://")
_MARKER_RUN_THEN_SLASH = re.compile("^[\x00\x01]+/")
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
    """Steps 4-7, the consumer step 8 (a segment holding either marker becomes ``{?}``), then 8a."""
    segs = [_param_segment(s) for s in normalize_path_rfc(path).split("/")]
    segs = [CONSUMER if has_unresolved(s) else s for s in segs]
    return mask_http_segments("/".join(segs), placeholder)


def is_wholly_unresolved(piece: str) -> bool:
    """``piece`` is exactly one marker: an *alternative* of a value set that is unknown as a whole."""
    return piece == UNKNOWN or piece == UNKNOWN_VALUE


def has_unresolved(s: str) -> bool:
    """Either marker occurs anywhere in ``s``: a *string under construction* that is partly unknown (``\\x01\\x01`` is, and is
    not wholly unresolved)."""
    return UNKNOWN in s or UNKNOWN_VALUE in s


def is_url_like(alt: str) -> bool:
    """Does ``alt`` read as a URL or a URL path: it starts with ``/`` (so ``//``), starts with ``scheme://``, or is a run of
    markers followed by ``/``? A wholly unresolved piece is not URL-like (the caller decides what to do with it)."""
    return alt.startswith("/") or bool(_SCHEME.match(alt)) or bool(_MARKER_RUN_THEN_SLASH.match(alt))


def names_no_literal_segment(key: str) -> bool:
    """P1-55: a normalized consumer key whose path holds at least one ``{?}`` segment and no other segment names nothing: it is
    unknown as a whole (``GET /{?}``, ``GET /{?}/{?}``), not partial. The root ``/`` alone is a key (no unknown segment), and so is
    a path with any literal or known-parameter (``{}``) segment."""
    segs = [s for s in key.partition(" ")[2].split("/") if s]
    return bool(segs) and all(s == CONSUMER for s in segs)


class NormalizedKey(NamedTuple):
    """``key`` is the normalized ``"<verb> <path>"``; ``origin_dropped`` says an unresolved origin was removed from a consumer
    URL; ``unresolved_in_path`` says the path (after the origin drop and the query/fragment cut) holds an unresolved piece."""
    key: str
    origin_dropped: bool
    unresolved_in_path: bool


def _cut_query(url: str) -> str:
    for ch in "?#":
        cut = url.find(ch)
        if cut >= 0:
            url = url[:cut]
    return url


def split_url(url: str) -> tuple[str, bool]:
    """Step 2: ``(path, origin_dropped)`` of a consumer URL; query and fragment are dropped.

    Rule 1: after ``scheme://`` or a leading ``//`` the authority (to the first ``/``, ``?``, ``#`` or the end) is the host and
    is dropped, known or not; a marker in it makes ``origin_dropped`` true. Rule 2: with no scheme, a leading run of
    ``UNKNOWN_VALUE`` immediately followed by ``/`` is the origin and is dropped (``origin_dropped`` true); every other
    position keeps its marker for :func:`normalize_path` to turn into ``{?}``.
    """
    dropped = False
    m = _AUTHORITY.match(url)
    if m:
        rest = url[m.end():]
        end = min((i for i in (rest.find("/"), rest.find("?"), rest.find("#")) if i >= 0), default=len(rest))
        dropped, url = has_unresolved(rest[:end]), rest[end:]
    else:
        run = len(url) - len(url.lstrip(UNKNOWN_VALUE))
        if run and url[run:run + 1] == "/":
            dropped, url = True, url[run:]
    return _cut_query(url), dropped


def url_path_of(url: str) -> str:
    """Step 2: the path only; scheme and authority, query and fragment are dropped."""
    return split_url(url)[0]


def normalize_key_parts(raw: str, form: str = "provider") -> NormalizedKey:
    """``keyNorm`` of ``"<verb> <path>"`` with the two facts the status rule needs (see the module docstring).

    A string with no leading verb word is a bare path with the any-method verb ``*``. ``form`` is ``provider`` (``{#}``) or
    ``consumer`` (``{?}``). The consumer form first takes the URL's path (step 2), the provider form does not (a route is a
    path already, so it never drops an origin).
    """
    if form not in ("provider", "consumer"):
        raise ValueError(f"form must be 'provider' or 'consumer', got {form!r}")
    raw = unicodedata.normalize("NFC", raw).strip()
    verb, sep, rest = raw.partition(" ")
    if sep and not raw.startswith("/") and (verb.upper() in VERBS or verb == "*"):
        v, path = verb.upper(), rest
    else:
        v, path = "*", raw
    dropped = False
    if form == "consumer":
        path, dropped = split_url(path)
    placeholder = PROVIDER if form == "provider" else CONSUMER
    return NormalizedKey(f"{v} {normalize_path(path, placeholder)}", dropped, has_unresolved(path))


def normalize_key(raw: str, form: str = "provider") -> str:
    """The key of :func:`normalize_key_parts`."""
    return normalize_key_parts(raw, form).key
