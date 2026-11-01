"""The label smoke's fixture (plan4 §4.3 U5.7 row; the ledger's P5-F5 ruling in
`docs/plans/typed-decision-models-coordination.md`).

`at_cap_items(count)` returns `count` guard `understanding` items whose rendering is exactly
`render.MAX_CHARS` (6,000) characters of double-escaped pt-BR text: the renderer ASCII-escapes the
state block and the request line escapes it again, the costliest realistic line for Read's
25,000-token cap. `driver smoke-fixture` writes them as the `items.jsonl` of `driver init --phase
smoke --fixture`; the 50 items make a full 10-page label batch.
"""

from __future__ import annotations

import json
from typing import Any

from . import render

PHRASE = ("O cliente já informou que precisa de orçamento para a reforma da cozinha, com "
          "instalação elétrica nova e acabamento em azulejo português")


def at_cap_item(index: int) -> dict[str, Any]:
    return {
        "condition": "the user has stated a budget for the request",
        "path": "understanding",
        "understanding": {
            "request": f"Reforma da cozinha nº {index} com orçamento até R$ 15.000",
            "known": [f"{PHRASE} ({index}.{n})" for n in range(60)],
            "missing": [],
        },
        "turns": [],
    }


def at_cap_items(count: int = 50) -> list[dict[str, Any]]:
    """`count` items, each rendering to exactly `render.MAX_CHARS`; anything else is a defect."""
    items = [at_cap_item(index) for index in range(count)]
    lengths = {len(render.render_guard_item(item)) for item in items}
    if lengths != {render.MAX_CHARS}:
        raise ValueError(f"the fixture items render to {sorted(lengths)}, not {render.MAX_CHARS}")
    return items


def fixture_text(count: int = 50) -> str:
    return "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in at_cap_items(count))
