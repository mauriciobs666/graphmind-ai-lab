"""`tool-caller` scorer (`ConversationScorer`-shaped, structural — `retrieval.py`'s/
`classification.py`'s own precedent).

Design: `docs/plans/small-model-benchmarking-s5-spec.md` §3.4, §5 Step 0. This module is built
incrementally across S5's seven steps; **Step 0 lands only** the two `-ml` §4.2(f)/§4.3.1 item 8
scoring constants and the cross-module union+disjointness assertion that binds them to
`convo.TURN_DISPOSITIONS` (plan `:5797-5799`, S5 spec §4 S5 *Done when* item (4)). Everything else
`toolcalls.py` will eventually carry — `turn_disposition_scores`, `emission_form`,
`score_conversations`, and the rest of §3.4's public surface — is later S5 steps' own work, not
this one's.
"""

from __future__ import annotations

from modelbench import convo

#: `-ml` §4.2(f)/§4.3.1 item 8, transcribed literally (plan `:5838-5844`): the population of the
#: `I(t)` mean/p95 summary. A **scoring** vocabulary, so it sits here beside the scorer rather than
#: in `convo`, whose `TURN_DISPOSITIONS` is a **mechanism** vocabulary. Written out literally —
#: never derived from `ITERATION_SUMMARY_EXCLUDED` or from `convo.TURN_DISPOSITIONS` — because a
#: derived complement would make the assertion below a tautology.
ITERATION_SUMMARY_DISPOSITIONS: frozenset[str] = frozenset({"replied", "cap-hit"})

#: The complement `-ml` §4.2(f) excludes from that summary, transcribed literally for the same
#: reason `ITERATION_SUMMARY_DISPOSITIONS` is.
ITERATION_SUMMARY_EXCLUDED: frozenset[str] = frozenset(
    {"timed-out", "no-response", "server-rejected"}
)

#: The cross-module union (plan `:5796-5799`, S5 spec §4 S5 *Done when* item (4)): binds two
#: independently authored declarations across two modules, so a sixth mechanism belongs to neither
#: set and reddens before it can be silently included or dropped, and so a mechanism placed in both
#: (an incomplete reclassification) reddens on the disjointness half rather than passing because the
#: union still happens to come out whole.
assert ITERATION_SUMMARY_DISPOSITIONS | ITERATION_SUMMARY_EXCLUDED == convo.TURN_DISPOSITIONS, (
    "ITERATION_SUMMARY_DISPOSITIONS | ITERATION_SUMMARY_EXCLUDED must equal "
    "convo.TURN_DISPOSITIONS exactly — a mechanism outside both sets is a coverage gap"
)
assert not (ITERATION_SUMMARY_DISPOSITIONS & ITERATION_SUMMARY_EXCLUDED), (
    "ITERATION_SUMMARY_DISPOSITIONS and ITERATION_SUMMARY_EXCLUDED must be disjoint — a mechanism "
    "in both is an incomplete reclassification, not a valid scoring vocabulary"
)
