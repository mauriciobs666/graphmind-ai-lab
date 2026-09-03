"""The closed set of roles, and the analysis unit each one's statistics are computed over.

Design: `docs/plans/small-model-benchmarking.md` §1 (FR-21's five roles) and
`docs/plans/small-model-benchmarking-ml.md` §3.3's table, which is the source of the unit column.

The unit kind is a property of the *role*, not of a call site: it is what a resolving-power line
prints ("n=12 effective conversations") and what `verdict()` cross-checks against the paired table
(`-ml` §3.4 Rule 4, precondition 2).
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

ROLES: tuple[str, ...] = (
    "tool-caller",
    "guard-judge",
    "nlq-generator",
    "chat-responder",
    "embedder",
)

#: `-ml` §3.3's unit column, verbatim.
UNIT_KIND_BY_ROLE: Mapping[str, str] = MappingProxyType(
    {
        "tool-caller": "conversation",
        "guard-judge": "item",
        "nlq-generator": "item",
        "chat-responder": "item",
        "embedder": "query",
    }
)


class UnknownRole(ValueError):
    """A role outside FR-21's five. There is no sixth, and no default."""


def unit_kind(role: str) -> str:
    try:
        return UNIT_KIND_BY_ROLE[role]
    except KeyError:
        raise UnknownRole(f"unknown role {role!r}; known roles are {', '.join(ROLES)}") from None
