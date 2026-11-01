"""Step 8a of the ``http`` ``keyNorm``: mask literal path segments that fail the token screen.

Spec: ``docs/plans/connected-code-graph-bundle-spec2.md`` §1.8 (step 8a) and §1.7.1 (the screen);
``docs/plans/connected-code-graph-eval-protocol2-ml.md`` §1.1 (the one shared implementation).
``assemble.py`` and ``match_keys.py`` import :func:`mask_http_segments`; neither reimplements it.

This is *only* step 8a. The earlier steps (verb, leading ``/``, percent-decoding **then** dot
segments in RFC 3986 order, parameter rewriting) are the caller's; the input is the string they
produced.
"""

from ccg.bundle import screen_fails

PROVIDER = "{#}"
CONSUMER = "{?}"
_PLACEHOLDERS = frozenset({"{}", "{**}", CONSUMER, PROVIDER})


def mask_http_segments(path: str, placeholder: str) -> str:
    """Replace every literal segment that fails the token screen by ``placeholder``.

    ``placeholder`` is ``{#}`` (provider form) or ``{?}`` (consumer form). ``path`` is a
    ``keyNorm`` string after step 7: either a bare path (``/a/b``) or ``VERB /a/b``; a leading
    verb is kept and never screened. Empty segments (a trailing ``/``) and the four placeholders
    ``{}``, ``{**}``, ``{?}``, ``{#}`` are kept. A partly parameterized segment such as
    ``{x}.json`` is a literal and is screened as written.
    """
    if placeholder not in (PROVIDER, CONSUMER):
        raise ValueError(f"placeholder must be {PROVIDER!r} or {CONSUMER!r}, got {placeholder!r}")
    verb, sep, rest = "", "", path
    if not path.startswith("/") and " " in path:
        verb, sep, rest = path.partition(" ")
    segments = [
        placeholder if s and s not in _PLACEHOLDERS and screen_fails(s) else s
        for s in rest.split("/")
    ]
    return verb + sep + "/".join(segments)
