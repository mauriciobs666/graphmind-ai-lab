#!/usr/bin/env python3
"""K-066 per-switch swap bench: chat x N -> embed x M -> chat x K against LM Studio.

Re-creation of U0's lost scratch `bench.py`
(`falkor-chat/docs/test-reports/salesperson-single-model-fit-report.md`, "Throughput";
review `falkor-chat/docs/reviews/salesperson-single-model.md` P3-1), used by U3's Q5(b).

Chat: `POST /api/v0/chat/completions`, temperature 0, `max_tokens` 256, system prompt =
a synthetic bakery catalog with a per-request nonce at the very start (defeats prefix
caching), user asks for three recommendations (~3.1k prompt tokens). LM Studio's
native `/api/v0` response carries `stats` (`tokens_per_second`, `time_to_first_token`),
which is what is reported. Embedding: `POST /api/v0/embeddings`, one short Portuguese
sentence; wall-clock latency is reported.

One JSON line per request is appended (and flushed) to `--out`, so a killed run keeps
every completed request. Stdlib only.

    python3 swap_bench.py --model mistralai/ministral-3-3b \\
        --embed-model text-embedding-qwen3-embedding-0.6b \\
        --sequence chat:4,embed:5,chat:2 --out swap.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
import uuid
from datetime import datetime, timezone

_ITEMS = [
    ("Sourdough Loaf", "bread", 8.5), ("Croissant", "pastry", 3.2), ("Pain au Chocolat", "pastry", 3.8),
    ("Baguette", "bread", 4.0), ("Cinnamon Roll", "pastry", 4.5), ("Rye Bread", "bread", 7.0),
    ("Lemon Tart", "dessert", 5.5), ("Carrot Cake Slice", "dessert", 5.0), ("Focaccia", "bread", 6.5),
    ("Almond Biscotti", "cookie", 2.5), ("Chocolate Chip Cookie", "cookie", 2.0), ("Eclair", "pastry", 4.2),
]


def _catalog(nonce: str) -> str:
    lines = [f"Request nonce {nonce}.", "You are a helpful bakery assistant. Catalog:"]
    i = 0
    # ~3.1k tokens of catalog lines, same order of magnitude as U0's prompt.
    while len(lines) < 76:  # ~3.1k prompt tokens with ministral's tokenizer (U0: 3,106-3,167)
        name, cat, price = _ITEMS[i % len(_ITEMS)]
        lines.append(
            f"- SKU B{i:04d}: {name} (batch {i // len(_ITEMS)}), category {cat}, price ${price + (i % 7) * 0.25:.2f}, "
            f"baked fresh daily, contains wheat, best within {1 + i % 3} days."
        )
        i += 1
    return "\n".join(lines)


def _post(url: str, body: dict, timeout: float = 300.0) -> tuple[int, dict, float]:
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = json.loads(r.read().decode())
            status = r.status
    except urllib.error.HTTPError as e:  # type: ignore[attr-defined]
        status, payload = e.code, {"error": e.read().decode(errors="replace")[:500]}
    return status, payload, time.monotonic() - t0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://localhost:1234")
    ap.add_argument("--model", default="mistralai/ministral-3-3b")
    ap.add_argument("--embed-model", default="text-embedding-qwen3-embedding-0.6b")
    ap.add_argument("--sequence", default="chat:4,embed:5,chat:2")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    seq = [(k, int(n)) for k, n in (part.split(":") for part in args.sequence.split(","))]
    with open(args.out, "a", encoding="utf-8") as out:
        idx = 0
        for phase_no, (kind, n) in enumerate(seq):
            for j in range(n):
                rec: dict = {"idx": idx, "phase": phase_no, "kind": kind, "pos_in_phase": j,
                             "at": datetime.now(timezone.utc).isoformat()}
                if kind == "chat":
                    status, p, wall = _post(f"{args.base_url}/api/v0/chat/completions", {
                        "model": args.model, "temperature": 0, "max_tokens": args.max_tokens, "stream": False,
                        "messages": [
                            {"role": "system", "content": _catalog(uuid.uuid4().hex)},
                            {"role": "user", "content": "Recommend three items for a breakfast for two, with prices."},
                        ],
                    })
                    stats, usage = p.get("stats") or {}, p.get("usage") or {}
                    rec.update(status=status, wall_s=round(wall, 3),
                               ttft_s=stats.get("time_to_first_token"), tok_s=stats.get("tokens_per_second"),
                               prompt_tokens=usage.get("prompt_tokens"), completion_tokens=usage.get("completion_tokens"),
                               error=p.get("error"))
                elif kind == "embed":
                    status, p, wall = _post(f"{args.base_url}/api/v0/embeddings", {
                        "model": args.embed_model, "input": "Quero um mouse sem fio barato para o escritório.",
                    })
                    dims = len(((p.get("data") or [{}])[0]).get("embedding") or [])
                    rec.update(status=status, wall_s=round(wall, 3), dims=dims, error=p.get("error"))
                else:
                    raise SystemExit(f"unknown kind {kind!r}")
                out.write(json.dumps(rec) + "\n")
                out.flush()
                print(json.dumps(rec), flush=True)
                idx += 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
