"""S35 TP-005 harness: the fixture KN rows against the real step 8a, and two mutants.

Plan: ``docs/test-plans/connected-code-graph-fixtures.md`` §5.2. The correct order is RFC 3986's
(decode, resolve dots, then mask; BS2 §3 F-13). Steps 5-6 are written here by hand because
``ccg.keynorm`` implements only step 8a. Each mutant must fail exactly the rows the plan names.
"""

import re

import pytest

from ccg.keynorm import mask_http_segments

UNRESERVED = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~")


def decode(path):
    def one(m):
        ch = chr(int(m.group(1), 16))
        return ch if ch in UNRESERVED else "%" + m.group(1).upper()

    return re.sub(r"%([0-9A-Fa-f]{2})", one, path)


def resolve_dots(path):
    out, segs = [], path.split("/")[1:]
    for i, seg in enumerate(segs):
        last = i == len(segs) - 1
        if seg == ".":
            if last:
                out.append("")
        elif seg == "..":
            if out:
                out.pop()
            if last:
                out.append("")
        else:
            out.append(seg)
    return "/" + "/".join(out)


def correct(p):
    return "GET " + mask_http_segments(resolve_dots(decode(p)), "{?}")


def mask_before_decode(p):
    return "GET " + resolve_dots(decode(mask_http_segments(p, "{?}")))


def resolve_before_decode(p):
    return "GET " + mask_http_segments(decode(resolve_dots(p)), "{?}")


KN = {
    "KN-1": ("/api/v1/%2E%2E/health", "GET /api/health"),
    "KN-2": ("/api/items/%2E%2E/%2E%2E/api/health", "GET /api/health"),
    "KN-3": ("/api/%68ealth", "GET /api/health"),
    "KN-4": ("/api/files/a%2fb", "GET /api/files/a%2Fb"),
    "KN-5": ("/api/hooks/x7k2%2Dq9w4%2Dm1p8%2Dz3v6", "GET /api/hooks/{?}"),
}


@pytest.mark.parametrize("kid", KN)
def test_correct_order_passes_every_row(kid):
    raw, want = KN[kid]
    assert correct(raw) == want


@pytest.mark.parametrize(
    "mutant, failing",
    [(resolve_before_decode, {"KN-1", "KN-2"}), (mask_before_decode, {"KN-5"})],
)
def test_mutant_fails_exactly_the_named_rows(mutant, failing):
    failed = {k for k, (raw, want) in KN.items() if mutant(raw) != want}
    assert failed == failing
