"""Majority of three blind votes and `labelVotes` (plan4 `docs/plans/typed-decision-models4.md`
§3.6 "Votes"; plan6 `typed-decision-models6.md` §3.7 and §3.8 "Votes"; note §Q1.2 "Majority of
three blind votes").

Pure functions over a point's answer space (`Point.expected`) and one item's votes. A vote is an
answer in that space (`ingest` has already refused anything else).

- **`vote_key(answer)`** spells an answer as its `labelVotes` key: `"true"` / `"false"` for a
  predicate (a JSON boolean is never a key, plan6 §3.7), the label itself for a choice.
- **`label_votes(expected, answers)`** counts `answers` over the whole answer space, zeros
  included: the object `labelVotes` is, in definition order. Three votes sum to 3.
- **`majority(expected, answers)`** is the answer at least two votes agree on, or `None`: three
  different labels on a choice point, which the campaign drops and counts as `voteTie`. A
  predicate has two answers, so three votes always have a majority (2-1 or 3-0).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

#: Votes per item (note §Q1.2): the majority of three.
VOTES_PER_ITEM = 3


def vote_key(answer: Any) -> str:
    """`answer` as a `labelVotes` key (module docstring)."""
    if isinstance(answer, bool):
        return "true" if answer else "false"
    return str(answer)


def label_votes(expected: Sequence[Any], answers: Sequence[Any]) -> dict[str, int]:
    """Vote counts over the whole answer space, zeros included."""
    counts = {vote_key(value): 0 for value in expected}
    for answer in answers:
        counts[vote_key(answer)] += 1
    return counts


def majority(expected: Sequence[Any], answers: Sequence[Any]) -> Any | None:
    """The answer two or more of `answers` agree on, or `None` (a `voteTie`)."""
    counts = label_votes(expected, answers)
    for value in expected:
        if counts[vote_key(value)] * 2 > len(answers):
            return value
    return None
