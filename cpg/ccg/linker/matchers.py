"""The linker's key matchers (P1-17; rule-pack spec §1.3 and §1.6 ``link``; plan ``docs/plans/connected-code-graph-p1-linking.md``).

A matcher compares a provider's ``keyNorm`` with a consumer's and says how strongly: :class:`Match` carries the matcher that
related them (the ``Link.matcher`` value: ``exact`` when the two strings are equal, else the strongest one applied) and
whether the relation is *partial* (an unknown part on the consumer or a masked one on the provider, so ``possible``).

Keys reaching :func:`match_keys` are already ``keyNorm`` strings, as the pack runtime emitted them. They are never normalized
again: a key is normalized once, from the writer's original (bundle spec3 §6 step 6; ``%2%45`` decodes to ``%2E`` and a second
pass would resolve it as a dot segment, changing the key). :func:`http_key` is the one entry for a raw URL or path the linker
derives itself; it follows the RFC 3986 order (percent-decode, then dot segments, F-13) through
:func:`ccg.packrt.httpkey.normalize_path_rfc`, which is the only normalizer.

Pattern keys (``keyForm`` ``glob``, ``regex``, ``template``) are matched by their own form against the other side's literal key
of the same kind, whatever ``link`` says; two pattern keys are never related. A pattern is repo-derived text and Python ``re`` is
not linear time, so a ``regex`` is applied only inside the RE2 subset plus :func:`regex_refusal`'s guard (no nested or ambiguous
quantification, at most 3 unbounded quantifiers, a text of at most 256 characters); a refused pattern matches nothing and is
reported through ``refused`` (``Links.counts["refused_patterns"]``), never dropped silently. A linear-time engine is a P2 item.
"""
from __future__ import annotations

import re
from typing import NamedTuple

from ccg.packrt.httpkey import normalize_key, normalize_path_rfc
from ccg.packrt.match import text_hit
from ccg.rulepack import regex_problem

__all__ = ["LINK_MATCHERS", "Match", "http_key", "match_keys", "mount_prefix_matches", "regex_refusal"]

LINK_MATCHERS = frozenset({"exact", "path-template", "glob", "prefix", "regex"})
_UNKNOWN_SEG, _MASKED_SEG, _ONE, _REST = "{?}", "{#}", "{}", "{**}"


class Match(NamedTuple):
    matcher: str
    partial: bool


def http_key(raw: str, form: str = "consumer") -> str:
    """``keyNorm`` of a raw ``http`` key or URL (``form`` ``provider`` or ``consumer``); RFC 3986 order (F-13)."""
    return normalize_key(raw, form)


# -- path-template (http) -------------------------------------------------------------------------
def _verb_path(key: str) -> tuple[str, str]:
    verb, sep, path = key.partition(" ")
    return (verb, path) if sep else ("*", key)


def _verbs(pv: str, cv: str) -> bool | None:
    """``None``: no match; ``False``: matches; ``True``: matches only because the consumer's verb is unknown."""
    if pv == cv or pv == "*":
        return False
    if cv == "*":
        return True
    return None


def _segments(ps: list[str], cs: list[str]) -> bool | None:
    """Segment-wise template match; ``None`` no match, else whether an unknown or masked segment made it partial."""
    partial = False
    for i, p in enumerate(ps):
        if p == _REST:
            rest = cs[i:]
            if i != len(ps) - 1 or not rest or not "".join(rest):
                return None
            return partial or _UNKNOWN_SEG in rest
        if i >= len(cs):
            return None
        c = cs[i]
        if c == _REST:                           # a consumer catch-all stands for the provider's one or more remaining segments
            return True if "".join(ps[i:]) else None
        if c in (_UNKNOWN_SEG, _ONE):
            if p == "":
                return None
            partial = True
        elif p == _MASKED_SEG:
            if c == "":
                return None
            partial = True
        elif p == _ONE:
            if c == "":
                return None
        elif p != c:
            return None
    return partial if len(cs) == len(ps) else None


def path_template(provider_key: str, consumer_key: str) -> Match | None:
    """``<VERB> <path>`` keys: ``{}`` is one non-empty segment, ``{**}`` the one or more remaining.

    An unknown consumer segment (``{?}``), a masked provider segment (``{#}``), an unknown consumer verb and a trailing slash
    present on one side only each make the relation partial.
    """
    if provider_key == consumer_key:
        return Match("exact", False)
    pv, pp = _verb_path(provider_key)
    cv, cp = _verb_path(consumer_key)
    verb = _verbs(pv, cv)
    if verb is None:
        return None
    got = _segments(pp.split("/"), cp.split("/"))
    if got is None and pp != cp and pp != "/" and cp != "/" and pp.endswith("/") != cp.endswith("/"):
        got = _segments(pp.rstrip("/").split("/"), cp.rstrip("/").split("/"))
        got = None if got is None else True
    if got is None:
        return None
    return Match("path-template", got or verb)


# -- pattern keys ---------------------------------------------------------------------------------
def _split_transport(kind: str, key: str) -> tuple[str, str]:
    """A ``topic`` key is ``<transport>:<name>``; the pattern applies to the name (rule-pack spec §5.2)."""
    if kind == "topic":
        transport, sep, name = key.partition(":")
        return (transport, name) if sep else ("", key)
    return "", key


MAX_TEXT, MAX_UNBOUNDED = 256, 3
_QUANT = "*+?{"


def _scan(pattern: str) -> tuple[int, bool]:
    """``(unbounded quantifier count, ambiguous)`` of ``pattern``; ``ambiguous`` is a quantified group that holds a quantifier or an
    alternation (``(a+)+``, ``(a|b)*``), the shape that backtracks exponentially. A conservative token scan, not a parser."""
    n, i, unbounded, ambiguous = len(pattern), 0, 0, False
    stack: list[list[bool]] = []                 # per open group: [holds a quantifier, holds an alternation]
    closed: list[bool] | None = None             # the group that closed as the previous token
    while i < n:
        c, prev_closed = pattern[i], closed
        closed = None
        if c == "\\":
            i += 2
        elif c == "[":
            i += 1 + pattern.startswith("^", i + 1)
            i += 1 if pattern.startswith("]", i) else 0
            j = pattern.find("]", i)
            i = n if j < 0 else j + 1
        elif c == "(":
            i += 1
            if pattern.startswith("?", i):
                if pattern.startswith("?P<", i):
                    i = pattern.find(">", i) + 1 or n
                elif pattern.startswith("?:", i):
                    i += 2
                else:                            # inline flags: (?i) is no group, (?i:...) is one
                    j = i
                    while j < n and pattern[j] not in ":)":
                        j += 1
                    if j < n and pattern[j] == ":":
                        i = j + 1
                    else:
                        i = j + 1
                        continue
            stack.append([False, False])
        elif c == ")":
            i += 1
            if stack:
                closed = stack.pop()
                if stack:
                    stack[-1][0] |= closed[0]
                    stack[-1][1] |= closed[1]
        elif c == "|":
            i += 1
            if stack:
                stack[-1][1] = True
        elif c in "*+?{":
            unb = c in "*+"
            if c == "{":
                end = pattern.find("}", i)
                m = re.fullmatch(r"(\d+)(,(\d*))?", pattern[i + 1:end]) if end > 0 else None
                if m is None:                    # a literal brace
                    i += 1
                    continue
                unb = m.group(2) is not None and m.group(3) == ""
                i = end
            i += 1
            unbounded += unb
            if prev_closed is not None and (prev_closed[0] or prev_closed[1]):
                ambiguous = True
            if stack:
                stack[-1][0] = True
        else:
            i += 1
    return unbounded, ambiguous


def regex_refusal(pattern: str, text: str) -> str | None:
    """Why ``pattern`` is not applied to ``text``, or ``None`` when it may be (behaviour (14))."""
    if (why := regex_problem(pattern)) is not None:
        return why
    if len(text) > MAX_TEXT:
        return f"text longer than {MAX_TEXT} characters"
    unbounded, ambiguous = _scan(pattern)
    if ambiguous:
        return "a quantified group holds a quantifier or an alternation"
    if unbounded > MAX_UNBOUNDED:
        return f"more than {MAX_UNBOUNDED} unbounded quantifiers"
    return None


def _regex_hit(pattern: str, text: str, refused: set | None) -> bool:
    if (why := regex_refusal(pattern, text)) is not None:
        if refused is not None:
            refused.add(pattern)
        return False
    try:
        return re.fullmatch(pattern, text) is not None
    except re.error:
        return False


def _pattern_match(kind: str, form: str, pattern_key: str, literal_key: str, refused: set | None = None) -> Match | None:
    if form == "template":
        got = path_template(pattern_key, literal_key)
        return None if got is None else Match("path-template", got.partial)
    p_transport, pattern = _split_transport(kind, pattern_key)
    l_transport, literal = _split_transport(kind, literal_key)
    if p_transport != l_transport:
        return None
    if form == "glob":
        return Match("glob", False) if text_hit({"glob": pattern}, literal) else None
    if form == "regex":
        return Match("regex", False) if _regex_hit(pattern, literal, refused) else None
    raise ValueError(f"unknown keyForm {form!r}")


# -- entry ----------------------------------------------------------------------------------------
def match_keys(kind: str, matcher: str, provider_key: str, consumer_key: str, *,
               provider_form: str = "literal", consumer_form: str = "literal", refused: set | None = None) -> Match | None:
    """Relate a provider's key with a consumer's. ``matcher`` is the pack's ``link`` matcher for ``kind`` (default ``exact``).
    ``refused`` (optional) collects the regex patterns the guard refused."""
    if matcher not in LINK_MATCHERS:
        raise ValueError(f"unknown link matcher {matcher!r}")
    if provider_form != "literal" and consumer_form != "literal":
        return None
    if consumer_form != "literal":
        return _pattern_match(kind, consumer_form, consumer_key, provider_key, refused)
    if provider_form != "literal":
        return _pattern_match(kind, provider_form, provider_key, consumer_key, refused)
    if provider_key == consumer_key:
        return Match("exact", False)
    if matcher == "exact":
        return None
    if matcher == "path-template":
        return path_template(provider_key, consumer_key)
    if matcher == "prefix":
        return Match("prefix", False) if text_hit({"prefix": provider_key}, consumer_key, path=provider_key.startswith("/")) else None
    return _pattern_match(kind, matcher, provider_key, consumer_key, refused)


# -- O-5: a mount path against a consumer URL path -------------------------------------------------
def mount_prefix_matches(mount_path: str, url_path: str) -> bool:
    """A consumer path equal to the mount path or under it (at a ``/`` boundary); the inner path is never compared (O-5).

    Both operands are normalized once, from the original text (RFC order, :func:`normalize_path_rfc`).
    """
    mount, url = normalize_path_rfc(mount_path).rstrip("/") or "/", normalize_path_rfc(url_path)
    if mount == "/":
        return True
    return url.rstrip("/") == mount or url.startswith(mount + "/")
