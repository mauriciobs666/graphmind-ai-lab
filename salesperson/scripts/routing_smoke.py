#!/usr/bin/env python3
"""K-066 U3 Q1 routing smoke: per language, one fresh storefront session that asks the
three catalog questions meant to steer the `assistant` toward `query_graph_data`
(an aggregate, a superlative, and golden pair `nlq-12`'s compound filter).

Reuses `language_salience_probe.run_participant` (same join/post/poll/read plumbing,
one flushed JSONL line per turn) with Q1's scripted turns instead of the TP-007 ones.
Languages run one after another (not concurrently), so a miss is attributable to the
model, not to concurrency. Snapshots `/api/v0/models` (loaded) before and after.

Which tool answered is read afterwards from `ws:demo` `Message.toolsUsed` (plan §7 Q1).

    python3 routing_smoke.py --out q1.jsonl [--languages en,pt-BR,es] [--extra-variant N]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import language_salience_probe as probe  # noqa: E402

PROMPTS = {
    "en": [
        "How many products do you have in Wearables?",
        "What's your cheapest product?",
        "Show me Wearables products that cost under $200.",
    ],
    "pt-BR": [
        "Quantos produtos vocês têm na categoria Wearables?",
        "Qual é o produto mais barato de vocês?",
        "Mostre-me produtos da categoria Wearables que custam menos de US$ 200.",
    ],
    "es": [
        "¿Cuántos productos tienen en la categoría Wearables?",
        "¿Cuál es su producto más barato?",
        "Muéstrame productos de la categoría Wearables que cuesten menos de 200 dólares.",
    ],
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--lmstudio-url", default="http://localhost:1234")
    ap.add_argument("--languages", default="en,pt-BR,es")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    w = probe.JsonlWriter(a.out)
    run_id = uuid.uuid4().hex
    w.write({"type": "run_start", "run_id": run_id, "at": probe.now_iso(), "epoch_ms": int(time.time() * 1000),
             "lmstudio_pre": probe.lmstudio_snapshot(a.lmstudio_url)})
    for i, lang in enumerate(s.strip() for s in a.languages.split(",") if s.strip()):
        probe.run_participant(a.base_url, run_id, i, lang, PROMPTS[lang], w, probe.DEFAULT_TURN_WAIT_CEILING_S)
    w.write({"type": "run_end", "run_id": run_id, "at": probe.now_iso(),
             "lmstudio_post": probe.lmstudio_snapshot(a.lmstudio_url)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
