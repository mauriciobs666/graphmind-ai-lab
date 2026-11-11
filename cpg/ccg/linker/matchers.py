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
not linear time, so a ``regex`` is applied only inside the RE2 subset plus :func:`regex_refusal`'s guard, which classifies the pattern
as Python's own parser reads it (``{,N}``, ``{,}`` and ``{N,}`` are quantifiers; anything it cannot classify is refused): no quantified group
holding a quantifier or an alternation, at most :data:`MAX_WIDE` unbounded-or-wide (span over :data:`WIDE_SPAN`) and :data:`MAX_QUANTIFIERS` quantifiers of any kind (bounded
ones included), at most :data:`MAX_ALTERNATIONS` extra alternation branches (unquantified groups multiply too), a cost model (below), a text of at most 256
characters. Across one link run a :class:`RegexBudget` caps the distinct (pattern, literal) pairs evaluated at :data:`MAX_REGEX_PAIRS`
(memoised); a refused pattern matches nothing and is reported through ``refused`` (``Links.counts["refused_patterns"]``), a too-long
text apart through ``refused_text`` (``refused_texts``) and a pair past the budget through ``RegexBudget.exhausted``
(``refused_regex_pairs``), never dropped silently. The cost ceiling (:data:`MAX_COST`, the product model of :func:`_walk`, times :data:`WIDENING_FACTOR` under an ``(?i)``/``(?s)`` flag) bounds one pair: the worst admitted pair a corner-biased search found (188,857 candidates, 53,623 admitted) took 12.2 ms at the 256-character text cap, so a run's stall is bounded by the budget times that, about 10 s (generator, figures and the reviewers' reported 11.4 ms / 2.6 ms in ``cpg/tests/data/regex_guard_evidence.txt``). It is a measured bound, not a proof: Python ``re`` cannot be interrupted; a hard guarantee needs a killable worker or a linear-time engine (P2). A linear-time engine is a P2 item.
"""
from __future__ import annotations

import re
from typing import NamedTuple

from ccg.packrt.httpkey import normalize_key, normalize_path_rfc
from ccg.packrt.match import text_hit
from ccg.rulepack import regex_problem

__all__ = ["LINK_MATCHERS", "MAX_ALTERNATIONS", "MAX_COST", "MAX_QUANTIFIERS", "MAX_REGEX_PAIRS", "Match", "RegexBudget", "http_key", "match_keys", "mount_prefix_matches", "regex_pattern_refusal",
           "regex_refusal"]

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


MAX_TEXT, MAX_WIDE, WIDE_SPAN, MAX_QUANTIFIERS, MAX_ALTERNATIONS, MAX_COST, WIDENING_FACTOR, MAX_REGEX_PAIRS = 256, 2, 8, 4, 4, 100_000, 4, 800

try:                                             # the parser Python's own ``re`` compiles with: one grammar, never a second reading of it
    from re import _constants as _C, _parser as _P
    _parse = _P.parse
    _REPEATS = {_C.MAX_REPEAT, _C.MIN_REPEAT}
    _PLAIN = {_C.LITERAL, _C.NOT_LITERAL, _C.ANY, _C.IN, _C.CATEGORY, _C.AT}
    _WIDENING = int(_C.SRE_FLAG_IGNORECASE | _C.SRE_FLAG_DOTALL)
except ImportError:                              # pragma: no cover - no parser to classify with: every pattern is refused
    _parse = None
    _REPEATS, _PLAIN, _WIDENING = set(), set(), 0


class _Shape:
    """What a parsed pattern costs a backtracking engine: ``wide`` quantifiers (unbounded, or a span ``hi - lo`` over :data:`WIDE_SPAN`:
    ``\\w{0,255}`` costs as much as ``\\w*``), ``total`` quantifiers, ``alternations`` (extra branches), ``cost`` (the cost model of
    :func:`_walk`), ``widened`` (an ``(?i)``/``(?s)`` flag anywhere), ``ambiguous`` (a quantified group holding a quantifier or an
    alternation) and ``unknown`` (a construct not classified)."""
    def __init__(self) -> None:
        self.wide = self.total = self.alternations = 0
        self.cost = 1
        self.ambiguous = self.unknown = self.widened = False


def _walk(items, shape: _Shape) -> tuple[bool, int]:
    """Classify ``items`` (a parsed sub-pattern) into ``shape``; returns ``(holds a quantifier or an alternation, cost)``.

    The cost model is the product, over the elements of a sequence, of: ``min(span, MAX_TEXT) + 1`` for a quantifier (a wide one is the
    text length, a narrow one its span plus one: how many lengths the engine retries); the *sum* of its branches'
    costs for an alternation (every branch may be tried at every position). A repeated body holds no quantifier or alternation (else the
    pattern is refused as ambiguous), so the product is the polynomial the backtracker can reach: ``n^wide * prod(narrow spans + 1) *
    prod(branches)``. It over-counts (disjoint atoms do not multiply), which only refuses more."""
    holds, cost = False, 1
    for op, av in items:
        if op in _PLAIN:
            continue
        if op == _C.SUBPATTERN:
            h, c = _walk(av[3], shape)
            holds |= h
            cost *= c
            shape.widened |= bool((av[1] | av[2]) & _WIDENING)
        elif op == _C.BRANCH:
            shape.alternations += len(av[1]) - 1
            cost *= sum(_walk(branch, shape)[1] for branch in av[1])
            holds = True
        elif op in _REPEATS:
            lo, hi, body = av
            shape.total += 1
            shape.wide += hi - lo > WIDE_SPAN              # an unbounded ``hi`` is MAXREPEAT: wide too
            shape.ambiguous |= _walk(body, shape)[0]          # a repeated body holding anything is refused, so its cost is 1
            cost *= min(hi - lo, MAX_TEXT) + 1
            holds = True
        else:                                    # assertion, backreference, atomic or possessive form, anything newer: refused by default
            shape.unknown = True
    return holds, cost


def _quantified_alternation(pattern: str) -> bool:
    """True iff a group holding a ``|`` (the parser may fold ``(a|b)`` into a set, so this reads the text) is directly followed by
    ``*``, ``+``, ``?`` or ``{``: the conservative form of "a quantified group holds an alternation"; escapes and classes are skipped."""
    n, i, stack, closed = len(pattern), 0, [], False
    while i < n:
        c, was_closed = pattern[i], closed
        closed = False
        if c == "\\":
            i += 2
            continue
        if c == "[":
            i += 1 + pattern.startswith("^", i + 1)
            i += pattern.startswith("]", i)
            while i < n and pattern[i] != "]":
                i += 2 if pattern[i] == "\\" else 1
        elif c == "(":
            stack.append(False)
        elif c == "|" and stack:
            stack[-1] = True
        elif c == ")" and stack:
            closed = stack.pop()
            if stack:
                stack[-1] |= closed
        elif c in "*+?{" and was_closed:
            return True
        i += 1
    return False


def _shape(pattern: str) -> _Shape | None:
    """The :class:`_Shape` of ``pattern`` as the engine itself parses it (``{,N}``, ``{,}``, ``{N,}``, lazy forms, escaped or
    in-class braces included), or ``None`` when it cannot be parsed."""
    try:
        tree = _parse(pattern)
        shape = _Shape()
        shape.cost = _walk(tree, shape)[1]
        shape.widened |= bool(getattr(getattr(tree, "state", None), "flags", 0) & _WIDENING)
    except Exception:                            # re.error, OverflowError, RecursionError, and no parser at all (None): refused, never raised
        return None
    return shape


def regex_pattern_refusal(pattern: str) -> str | None:
    """Why ``pattern`` is never applied, whatever the text, or ``None`` (behaviour (14), the pattern's own fault)."""
    try:
        why = regex_problem(pattern)
    except Exception:
        why = "not a regex the guard can parse"
    if why is not None:
        return why
    shape = _shape(pattern)
    if shape is None:
        return "not a pattern the guard can classify"
    if shape.unknown:
        return "a construct the guard cannot classify"
    if shape.ambiguous or _quantified_alternation(pattern):
        return "a quantified group holds a quantifier or an alternation"
    if shape.wide > MAX_WIDE:
        return f"more than {MAX_WIDE} unbounded or wide quantifiers"
    if shape.total > MAX_QUANTIFIERS:
        return f"more than {MAX_QUANTIFIERS} quantifiers"
    if shape.alternations > MAX_ALTERNATIONS:
        return f"more than {MAX_ALTERNATIONS} alternations"
    if shape.cost * (WIDENING_FACTOR if shape.widened else 1) > MAX_COST:
        return f"cost model over {MAX_COST}"
    return None


def regex_refusal(pattern: str, text: str) -> str | None:
    """Why ``pattern`` is not applied to ``text``, or ``None`` when it may be (behaviour (14))."""
    if (why := regex_pattern_refusal(pattern)) is not None:
        return why
    if len(text) > MAX_TEXT:
        return f"text longer than {MAX_TEXT} characters"
    return None


class RegexBudget:
    """A hard, deterministic work bound across one link run's (pattern, literal) pairs: ``limit`` distinct pairs are evaluated, each
    result is memoised, and every further distinct pair is refused (counted in ``exhausted``) instead of evaluated. Python ``re`` cannot
    be interrupted, so the bound is on the number of pairs, not on time; the guard bounds each pair."""

    def __init__(self, limit: int = MAX_REGEX_PAIRS):
        self.limit, self.exhausted = limit, 0
        self._seen: dict[tuple[str, str], bool] = {}

    def hit(self, pattern: str, text: str) -> bool:
        key = (pattern, text)
        if key in self._seen:
            return self._seen[key]
        if len(self._seen) >= self.limit:
            self.exhausted += 1
            return False
        try:
            got = re.fullmatch(pattern, text) is not None
        except re.error:
            got = False
        self._seen[key] = got
        return got


def _regex_hit(pattern: str, text: str, refused: set | None, refused_text: set | None, budget: RegexBudget | None = None) -> bool:
    if (why := regex_pattern_refusal(pattern)) is not None:
        if refused is not None:
            refused.add(pattern)
        return False
    if len(text) > MAX_TEXT:
        if refused_text is not None:
            refused_text.add(text)
        return False
    return (budget or RegexBudget()).hit(pattern, text)


def _pattern_match(kind: str, form: str, pattern_key: str, literal_key: str, refused: set | None = None,
                   refused_text: set | None = None, budget: RegexBudget | None = None) -> Match | None:
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
        return Match("regex", False) if _regex_hit(pattern, literal, refused, refused_text, budget) else None
    raise ValueError(f"unknown keyForm {form!r}")


# -- entry ----------------------------------------------------------------------------------------
def match_keys(kind: str, matcher: str, provider_key: str, consumer_key: str, *,
               provider_form: str = "literal", consumer_form: str = "literal", refused: set | None = None,
               refused_text: set | None = None, budget: RegexBudget | None = None) -> Match | None:
    """Relate a provider's key with a consumer's. ``matcher`` is the pack's ``link`` matcher for ``kind`` (default ``exact``).
    ``refused`` (optional) collects the regex patterns the guard refused, ``refused_text`` the literals it refused as too long, ``budget`` (a :class:`RegexBudget`) the pair bound of one link run."""
    if matcher not in LINK_MATCHERS:
        raise ValueError(f"unknown link matcher {matcher!r}")
    if provider_form != "literal" and consumer_form != "literal":
        return None
    if consumer_form != "literal":
        return _pattern_match(kind, consumer_form, consumer_key, provider_key, refused, refused_text, budget)
    if provider_form != "literal":
        return _pattern_match(kind, provider_form, provider_key, consumer_key, refused, refused_text, budget)
    if provider_key == consumer_key:
        return Match("exact", False)
    if matcher == "exact":
        return None
    if matcher == "path-template":
        return path_template(provider_key, consumer_key)
    if matcher == "prefix":
        return Match("prefix", False) if text_hit({"prefix": provider_key}, consumer_key, path=provider_key.startswith("/")) else None
    return _pattern_match(kind, matcher, provider_key, consumer_key, refused, refused_text, budget)


# -- O-5: a mount path against a consumer URL path -------------------------------------------------
def mount_prefix_matches(mount_path: str, url_path: str) -> bool:
    """A consumer path equal to the mount path or under it (at a ``/`` boundary); the inner path is never compared (O-5).

    Both operands are normalized once, from the original text (RFC order, :func:`normalize_path_rfc`).
    """
    mount, url = normalize_path_rfc(mount_path).rstrip("/") or "/", normalize_path_rfc(url_path)
    if mount == "/":
        return True
    return url.rstrip("/") == mount or url.startswith(mount + "/")
