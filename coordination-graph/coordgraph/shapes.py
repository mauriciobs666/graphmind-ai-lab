"""ID-token shapes (plan3 section 3.1 steps (a)-(c)): chosen by shape, never validated.

A token is stored as written everywhere; these tests only pick which resolver applies, and
`history`/`lint` use them to *find* ID tokens in prose. Canonical form (widths, parent
matrix) is `wilib.rules`' call, made by `lint`, never by the parsers.
"""
from __future__ import annotations

import re
from collections.abc import Callable

UNSCOPED = "unscoped"  # lint's stand-in for a referrer with no backlog scope
LEGACY_BARE = re.compile(r"[A-Z]-\d+")
LEGACY_QUALIFIED = re.compile(r"[a-z0-9][a-z0-9-]*:[A-Z]-\d+")
SCHEME = re.compile(r"[A-Z]{2,6}(?:-[A-Z]\d+|-FR\d+[a-z]?|-AC\d+[a-z]?)+")

# Resolves a token to the set of strings under which its identity may be stored (plan3 3.1,
# `candidates`). S1d's registry-backed one plugs in here; `offline_candidates` is the default.
Candidates = Callable[[str, str], set[str]]
# Optional companion of `Candidates` (`identity.Identity.reason`): why a token is not a resolved
# identity (unminted, unknown, ambiguous, registry unavailable), or None when it is.
Reason = Callable[[str, str], str | None]


def classify(token: str) -> str | None:
    """`legacy-bare` | `legacy-qualified` | `scheme` | None (anything else is unrecognised)."""
    if LEGACY_BARE.fullmatch(token):
        return "legacy-bare"
    if LEGACY_QUALIFIED.fullmatch(token):
        return "legacy-qualified"
    if SCHEME.fullmatch(token):
        return "scheme"
    return None


def qualify(token: str, scope: str) -> str:
    """The graph's stored form of a reference: bare legacy gains the referrer's scope."""
    return f"{scope}:{token}" if classify(token) == "legacy-bare" else token


def offline_candidates(token: str, scope: str) -> set[str]:
    """Candidates with no registry (plan3 3.1 steps 1(a)/3, what `identity.Identity` yields when unavailable): a
    bare legacy token with a real referrer scope is denoted by its qualified form alone (the bare string would also
    match another home's `K-065`); anything else is the token as written."""
    if scope and scope != UNSCOPED:
        return {qualify(token, scope)}
    return {token}
