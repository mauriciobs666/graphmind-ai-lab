"""The leakage key: `input_hash(item, role)` (`docs/plans/typed-decision-models.md` v1.3 §3.5,
held equal across components by §3.2.2's golden vectors).

v1.3 §3.5's formula: the SHA-256 hex digest of the UTF-8 bytes of
`json.dumps({k: item[k] for k in INPUT_FIELDS_BY_ROLE[role] if k in item}, sort_keys=True,
separators=(",", ":"), ensure_ascii=False)` — the item's input fields only, keys sorted at every
level, no whitespace, non-ASCII kept raw, and a field the item lacks omitted rather than written
as `null`. Non-input fields (`itemId`, `tier`, `expected`, `stratum`, `origin`, …) never change
it, so the same input carried by two items hashes the same.

model-bench owns the canonical `input-hash.jsonl`, whose digests were computed by hand from the
formula, not by any `input_hash`; this component asserts against a byte-copy of it under
`tests/fixtures/decision-contract/`. FR-8 forbids importing model-bench, so this is the formula
implemented again, not shared code.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

#: v1.3 §3.5: the item fields `input_hash` covers, keyed by model-bench role. `guard-judge` only:
#: the join-in and message-type rows (plan2 §3.1 for the latter) arrive with their vectors in
#: U5.2b, after model-bench's U4 lands them.
INPUT_FIELDS_BY_ROLE: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {"guard-judge": ("condition", "understanding", "turns", "path")}
)


def canonical_input_json(item: Mapping[str, Any], role: str) -> str:
    """The canonical JSON the hash covers: `role`'s input fields present in `item`, keys sorted
    at every level, no whitespace, non-ASCII raw. An unknown role is a `ValueError`."""
    try:
        fields = INPUT_FIELDS_BY_ROLE[role]
    except KeyError:
        raise ValueError(f"no input fields are defined for role {role!r}") from None
    inputs = {name: item[name] for name in fields if name in item}
    return json.dumps(inputs, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def input_hash(item: Mapping[str, Any], role: str) -> str:
    """The SHA-256 hex digest of `canonical_input_json(item, role)` encoded as UTF-8."""
    return hashlib.sha256(canonical_input_json(item, role).encode("utf-8")).hexdigest()
