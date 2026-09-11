"""Per-role scorer modules (`ItemScorer`/`ConversationScorer` implementations).

Design: `docs/plans/small-model-benchmarking-s3-spec.md` §3.2/§4. A module resolved by
`runner._load_item_scorer` from a pack's declared `"scorer"` name (`modelbench.scoring.<name>`)
satisfies the `ItemScorer` Protocol structurally — its `score_item`/`aggregate` module-level
functions ARE the Protocol's methods, never an `isinstance`-checked class instance.
"""

from __future__ import annotations
