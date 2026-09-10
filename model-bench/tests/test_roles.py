"""`modelbench.roles` — the closed set of roles and the analysis-unit kind each one carries.

Design: `docs/plans/small-model-benchmarking.md` §1 (FR-21's five roles) and
`docs/plans/small-model-benchmarking-ml.md` §3.3's unit column.

Review Pass 14 (`docs/reviews/small-model-benchmarking-impl.md`), P14-2: `ROLES` and
`UNIT_KIND_BY_ROLE` are two declarations of the same closed set, eight lines apart in
`roles.py`, with nothing binding them to each other. Before this file existed, three of the five
roles were pinned only by accident — some other suite's fixture happened to use them as a
`pack.role`/`unit_kind` value — and dropping `"chat-responder"` from `UNIT_KIND_BY_ROLE`, or
adding a spurious sixth entry to `ROLES`, both left the full suite green.
"""

from __future__ import annotations

import pytest

from modelbench.roles import (
    ANALYSIS_UNIT_FIELD_BY_ROLE,
    MULTI_CALL_TURN_BY_ROLE,
    ROLES,
    UNIT_KIND_BY_ROLE,
    UnknownRole,
    analysis_unit_field,
    unit_kind,
)


def test_multi_call_turn_by_role_domain_is_exactly_roles():
    """Same shape as the unit-kind pin above (P14-2): both directions of drift between the two
    declarations must redden, whichever side changed."""
    assert set(MULTI_CALL_TURN_BY_ROLE) == set(ROLES)


def test_multi_call_turn_by_role_is_true_for_tool_caller_only():
    """§3.3 v1.26's third role-table column: `tool-caller` is the one role whose turn is a
    bounded multi-call loop (`drive`'s `maxIterationsPerTurn` cap applies); the four item-level
    roles score one call per item and are single-call by construction."""
    assert MULTI_CALL_TURN_BY_ROLE["tool-caller"] is True
    for role in ROLES:
        if role != "tool-caller":
            assert MULTI_CALL_TURN_BY_ROLE[role] is False


def test_unit_kind_by_role_domain_is_exactly_roles():
    """The pin: both directions of drift between the two declarations must redden. `ROLES`
    losing or gaining a member, or `UNIT_KIND_BY_ROLE` losing or gaining one, breaks this
    equality regardless of which side changed."""
    assert set(UNIT_KIND_BY_ROLE) == set(ROLES)


@pytest.mark.parametrize("role", ROLES)
def test_unit_kind_resolves_every_declared_role(role):
    """Drives `unit_kind` itself over all five roles, not just the ones some other module's
    fixtures happen to touch."""
    assert unit_kind(role) == UNIT_KIND_BY_ROLE[role]


def test_unit_kind_refuses_a_role_outside_the_five():
    """`UnknownRole`'s own docstring: "There is no sixth, and no default.\""""
    with pytest.raises(UnknownRole, match="tour-guide"):
        unit_kind("tour-guide")


def test_analysis_unit_field_by_role_domain_is_exactly_roles():
    """Same shape as the unit-kind pin above (P14-2), for §3.3's route (iii) column
    (`ANALYSIS_UNIT_FIELD_BY_ROLE`, plan §4 S2 "Done when" `:5078-5082`): both directions of
    drift between the two declarations must redden, whichever side changed."""
    assert set(ANALYSIS_UNIT_FIELD_BY_ROLE) == set(ROLES)


def test_analysis_unit_field_by_role_matches_the_plans_table():
    """Plan §3.3's role table, verbatim: `pairingKey`'s outermost component name per role — a
    component *name* (`scriptId`/`itemId`), never `unit_kind`'s denominator *noun*, which is why
    the embedder's row is `itemId` and not `query`."""
    assert dict(ANALYSIS_UNIT_FIELD_BY_ROLE) == {
        "tool-caller": "scriptId",
        "guard-judge": "itemId",
        "nlq-generator": "itemId",
        "chat-responder": "itemId",
        "embedder": "itemId",
    }


@pytest.mark.parametrize("role", ROLES)
def test_analysis_unit_field_resolves_every_declared_role(role):
    """Drives `analysis_unit_field` itself over all five roles, not just the ones some other
    module's fixtures happen to touch."""
    assert analysis_unit_field(role) == ANALYSIS_UNIT_FIELD_BY_ROLE[role]


def test_analysis_unit_field_refuses_a_role_outside_the_five():
    """`UnknownRole`'s own docstring: "There is no sixth, and no default.\" — same refusal as
    `unit_kind`'s, mirrored for this second table."""
    with pytest.raises(UnknownRole, match="tour-guide"):
        analysis_unit_field("tour-guide")
