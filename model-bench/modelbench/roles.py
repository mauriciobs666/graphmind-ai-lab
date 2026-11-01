"""The closed set of roles, and the analysis unit each one's statistics are computed over.

Design: `docs/plans/small-model-benchmarking.md` §1 (FR-21's five roles) and
`docs/plans/small-model-benchmarking-ml.md` §3.3's table, which is the source of the unit column;
typed-decision-models plan v1.3 §3.6 "New roles (FR-20a extension)" adds `join-in-judge` and
`message-type-classifier`, both unit kind `item`, single-call, analysis unit `itemId`.

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
    "join-in-judge",
    "message-type-classifier",
)

#: `-ml` §3.3's unit column, verbatim.
UNIT_KIND_BY_ROLE: Mapping[str, str] = MappingProxyType(
    {
        "tool-caller": "conversation",
        "guard-judge": "item",
        "nlq-generator": "item",
        "chat-responder": "item",
        "embedder": "query",
        "join-in-judge": "item",
        "message-type-classifier": "item",
    }
)


#: §3.3 v1.26's third role-table column: whether a turn under this role is a bounded multi-call
#: loop (`drive`'s `maxIterationsPerTurn` cap applies) or single-call by construction. `True` only
#: for `tool-caller` — the item-level roles score one call per item and carry no cap.
#: `PromptConfig.maxIterationsPerTurn` is required *iff* this column is `True` for the pack's
#: role, and forbidden otherwise (`packs.Pack.prompt_config`, impl review P17-5).
MULTI_CALL_TURN_BY_ROLE: Mapping[str, bool] = MappingProxyType(
    {
        "tool-caller": True,
        "guard-judge": False,
        "nlq-generator": False,
        "chat-responder": False,
        "embedder": False,
        "join-in-judge": False,
        "message-type-classifier": False,
    }
)


#: §3.3's route (iii) column (v1.25): the analysis unit is the *outermost* component of
#: `sampling.pairingKey` — a `pairingKey` **component name**, never `UNIT_KIND_BY_ROLE`'s
#: denominator *noun* (the embedder's row is `itemId` / `query`, and deriving either column from
#: the other is wrong — plan §3.3). `check_sampling_contract` reads this through
#: `analysis_unit_field(role)` and requires `pairingKey[0]` to equal it.
ANALYSIS_UNIT_FIELD_BY_ROLE: Mapping[str, str] = MappingProxyType(
    {
        "tool-caller": "scriptId",
        "guard-judge": "itemId",
        "nlq-generator": "itemId",
        "chat-responder": "itemId",
        "embedder": "itemId",
        "join-in-judge": "itemId",
        "message-type-classifier": "itemId",
    }
)


class UnknownRole(ValueError):
    """A role outside `ROLES` — FR-21's five plus typed-decision-models' two (v1.3 §3.6). There is
    no default."""


def unit_kind(role: str) -> str:
    try:
        return UNIT_KIND_BY_ROLE[role]
    except KeyError:
        raise UnknownRole(f"unknown role {role!r}; known roles are {', '.join(ROLES)}") from None


def analysis_unit_field(role: str) -> str:
    try:
        return ANALYSIS_UNIT_FIELD_BY_ROLE[role]
    except KeyError:
        raise UnknownRole(f"unknown role {role!r}; known roles are {', '.join(ROLES)}") from None
