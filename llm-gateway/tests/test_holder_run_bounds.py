"""Run-bound matching, pinned at both enforcement points (plan §3.4, §3.9; review Pass 13 H1).

The holder (`Holder._bound_for`) and the library (`policy.check_run_bound`) carry separate copies
of the match (the holder imports no library policy). One case table runs against both so they
cannot drift: the first entry in file order whose glob `fnmatchcase`s the run id supplies the
bound (`*`, `?`, `[...]`, case-sensitive)."""

from types import SimpleNamespace

import pytest

from llmgateway.config import GatewayConfig
from llmgateway.keyholder import Holder
from llmgateway.ledger import Refusal
from llmgateway.policy import PolicyRefused, check_run_bound

BOUNDS = [
    ("eval-?", 10),
    ("eval-7*", 20),
    ("eval-*", 30),
    ("[ab]-*", 40),
    ("Case-*", 50),
    ("exact", 60),
    ("n[!0-9]-*", 70),
]
REVERSED_OVERLAP = [("eval-*", 30), ("eval-7*", 20)]

CASES = [
    (BOUNDS, "eval-7", 10),  # `?` is exactly one character
    (BOUNDS, "eval-77", 20),  # `?` does not match two; `eval-7*` precedes `eval-*`
    (BOUNDS, "eval-8x", 30),
    (BOUNDS, "eval-", 30),  # `*` matches empty
    (BOUNDS, "a-1", 40),  # character class
    (BOUNDS, "b-1", 40),
    (BOUNDS, "c-1", None),  # outside the class
    (BOUNDS, "Case-1", 50),
    (BOUNDS, "case-1", None),  # case-sensitive, pattern capitalised
    (BOUNDS, "EVAL-7", None),  # case-sensitive, run id capitalised
    (BOUNDS, "exact", 60),
    (BOUNDS, "exact-1", None),  # no wildcard: not a prefix match
    (BOUNDS, "xeval-7", None),  # anchored at the start
    (BOUNDS, "nx-1", 70),  # negated class
    (BOUNDS, "n7-1", None),
    (REVERSED_OVERLAP, "eval-77", 30),  # file order, not specificity or size, decides
    (REVERSED_OVERLAP, "eval-8", 30),
]


def holder_bound(bounds, run_id):
    policy = SimpleNamespace(run_bounds=tuple(bounds), unmatched="refuse", default_bound=None)
    try:
        bound, cap = Holder._bound_for(object(), policy, run_id)
    except Refusal as exc:
        assert exc.code == "run_unmatched"
        return None
    assert cap is None
    return bound


class Records:
    def __init__(self, used):
        self.used = used

    def tokens_used(self, run_id):
        return self.used


def library_bound(bounds, run_id):
    """The bound `check_run_bound` applies: the lowest token count it refuses at."""
    cfg = GatewayConfig(
        protected=True,
        run_bounds=tuple({"match": m, "max_tokens": n} for m, n in bounds),
    )

    def refused(used):
        try:
            check_run_bound(cfg, run_id, Records(used))
        except PolicyRefused as exc:
            assert exc.reason == "run_bound_reached"
            return True
        return False

    return next((used for used in range(101) if refused(used)), None)


@pytest.mark.parametrize("bounds,run_id,expected", CASES)
def test_holder_matches_by_fnmatchcase_first_entry_wins(bounds, run_id, expected):
    assert holder_bound(bounds, run_id) == expected


@pytest.mark.parametrize("bounds,run_id,expected", CASES)
def test_library_matches_by_fnmatchcase_first_entry_wins(bounds, run_id, expected):
    assert library_bound(bounds, run_id) == expected
