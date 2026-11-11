"""``Cond.code`` literal masking (C-1001 P1-06, F-9).

Each string literal in a condition's code whose body fails ``ccg.bundle.screen_fails`` or ``ccg.bundle.name_fails``
(`docs/plans/connected-code-graph-p1-extraction.md` P1-06) becomes ``"…"``; a passing literal and all non-literal text
stay byte-identical. One left-to-right pass, linear in the input: every step advances the cursor.

Handled: ``"``/``'`` literals with backslash escapes and Python triple quotes; Python string prefixes (kept outside the
mask); TS/JS template literals, whose static text is judged as one string and whose ``${…}`` expressions are scanned
recursively for nested literals; an unterminated literal runs to the end of the text.
Quotes inside regex literals or comments flip the scanner's quote parity; ``_backstop`` then still masks a UUID or a
12+ hex run, but any other secret shape there is not masked (the guarantee is per well-formed literal).
"""
from __future__ import annotations

import re

from ccg import bundle

MASK = '"…"'
_MAX_DEPTH = 32  # template nesting beyond this is left unmasked-by-recursion and judged as plain text


def _flagged(body: str) -> bool:
    return bundle.screen_fails(body) or bundle.name_fails(body)


def _scan_quoted(s: str, i: int, q: str) -> tuple[int, str, bool]:
    """From the opening quote at ``i``: (index after the literal, body, terminated)."""
    n = len(s)
    if q != "`" and s.startswith(q * 3, i):
        delim, j = q * 3, i + 3
    else:
        delim, j = q, i + 1
    start = j
    while j < n:
        if s[j] == "\\":
            j += 2
        elif s.startswith(delim, j):
            return j + len(delim), s[start:j], True
        else:
            j += 1
    return n, s[start:], False


_HEX_RUN = re.compile(r"[A-Za-z0-9]+")


def _backstop(text: str) -> str:
    """Rules T3/T4 of the screen applied to the text the scanner did NOT take for a literal (a quote inside a regex
    literal or a comment flips the scanner's quote parity, so a real secret literal reads as code): a UUID and an
    all-hex run of 12 or more characters become ``…``. Measured on the 4,034 stored slice conds: 0 changes."""
    text = bundle._UUID.sub("…", text)
    return _HEX_RUN.sub(lambda m: "…" if len(m[0]) >= 12 and all(c in bundle._HEXCH for c in m[0]) else m[0], text)


def mask_cond_code(code: str) -> str:
    out, _ = _mask(code, 0, 0, False)
    return _backstop(out)


def _mask(s: str, i: int, depth: int, in_expr: bool) -> tuple[str, int]:
    """Mask from ``i``; when ``in_expr`` stop at the ``}`` closing a template expression (returned index is on it)."""
    out: list[str] = []
    n = len(s)
    braces = 0
    while i < n:
        c = s[i]
        if c in "\"'":
            end, body, _ = _scan_quoted(s, i, c)
            out.append(MASK if _flagged(body) else s[i:end])
            i = end
        elif c == "`":
            text, i = _template(s, i, depth)
            out.append(text)
        elif in_expr and c == "{":
            braces += 1
            out.append(c)
            i += 1
        elif in_expr and c == "}":
            if braces == 0:
                return "".join(out), i
            braces -= 1
            out.append(c)
            i += 1
        else:
            out.append(c)
            i += 1
    return "".join(out), i


def _template(s: str, i: int, depth: int) -> tuple[str, int]:
    n = len(s)
    j = i + 1
    static: list[str] = []
    parts: list[str] = []          # rebuilt text, expressions masked
    closed = False
    while j < n:
        c = s[j]
        if c == "\\":
            seg = s[j:j + 2]
            static.append(seg)
            parts.append(seg)
            j += 2
        elif c == "`":
            j += 1
            closed = True
            break
        elif c == "$" and s.startswith("${", j) and depth < _MAX_DEPTH:
            inner, k = _mask(s, j + 2, depth + 1, True)
            parts.append("${" + inner + ("}" if k < n else ""))
            j = k + 1
        else:
            static.append(c)
            parts.append(c)
            j += 1
    if _flagged("".join(static)):
        return MASK, min(j, n)
    return "`" + "".join(parts) + ("`" if closed else ""), min(j, n)
