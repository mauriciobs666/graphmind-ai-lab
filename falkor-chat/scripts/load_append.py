#!/usr/bin/env python3
"""K-011 load harness — drive the M1 service-layer append path through REST.

Measures the *live request path* for a message post:
`POST /threads/{threadId}/messages` → `services.post_message` (validate actor +
mentions, derive role, dispatch the §4 v2 write) → FalkorDB. This is deliberately
NOT the K-007 bulk-`UNWIND` ingestion datapoint (single client, batched Cypher):
every message here is one HTTP round trip through the full service stack.

Concurrency model: `--workers` concurrent posters, each owning its own thread
(one channel, N threads). All writes still land in one workspace graph, so
sustained throughput is graph-write-bound regardless of thread fan-out — the
per-thread split just removes artificial first-post/TAIL race dispatch from the
latency sample. Reports p50/p99 append latency and sustained msg/s.

Talks only to the REST API of an already-running server (start it against an
isolated `ws:load` workspace — see scripts/load_test.sh). Writes a JSON summary
(with a sample channelId/threadId/msgId for the GRAPH.PROFILE step) to --emit.

Usage:
  load_append.py [--base-url URL] [--workers N] [--messages N]
                 [--text-bytes N] [--emit PATH]
Env fallbacks: LOAD_BASE_URL, LOAD_WORKERS, LOAD_MESSAGES, LOAD_EMIT.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter

import httpx


def _percentile(sorted_vals: list[float], pct: float) -> float:
    """Nearest-rank percentile (pct in [0,100]) over a pre-sorted list."""
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = pct / 100.0 * (len(sorted_vals) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = rank - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def _worker(base_url: str, thread_id: str, count: int, text: str) -> tuple[list[float], int]:
    """Post `count` messages to one thread; return (latencies_ms, errors)."""
    latencies: list[float] = []
    errors = 0
    with httpx.Client(base_url=base_url, timeout=30.0) as client:
        for _ in range(count):
            t0 = perf_counter()
            try:
                r = client.post(f"/threads/{thread_id}/messages", json={"text": text})
                dt_ms = (perf_counter() - t0) * 1000.0
                if r.status_code == 201:
                    latencies.append(dt_ms)
                else:
                    errors += 1
            except httpx.HTTPError:
                errors += 1
    return latencies, errors


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default=os.environ.get("LOAD_BASE_URL", "http://127.0.0.1:8000"))
    ap.add_argument("--workers", type=int, default=int(os.environ.get("LOAD_WORKERS", "16")))
    ap.add_argument("--messages", type=int, default=int(os.environ.get("LOAD_MESSAGES", "3000")))
    ap.add_argument("--text-bytes", type=int, default=80)
    ap.add_argument("--emit", default=os.environ.get("LOAD_EMIT", ""))
    args = ap.parse_args()

    base_url = args.base_url.rstrip("/")
    workers = max(1, args.workers)
    total = max(workers, args.messages)

    # A searchable, roughly realistic message body (~text-bytes chars). The
    # "load" token guarantees the §5 full-text PROFILE has hits to rank.
    filler = "load test message body reproducible append path graphrag chat "
    text = (filler * ((args.text_bytes // len(filler)) + 1))[: args.text_bytes]

    with httpx.Client(base_url=base_url, timeout=30.0) as client:
        r = client.get("/health")
        r.raise_for_status()

        ch = client.post("/channels", json={"name": "load"})
        ch.raise_for_status()
        channel_id = ch.json()["channelId"]

        thread_ids: list[str] = []
        for i in range(workers):
            tr = client.post(
                f"/channels/{channel_id}/threads", json={"title": f"load-{i}"}
            )
            tr.raise_for_status()
            thread_ids.append(tr.json()["threadId"])

    # Even split (remainder spread over the first workers).
    base, rem = divmod(total, workers)
    counts = [base + (1 if i < rem else 0) for i in range(workers)]

    print(
        f"load: {total} messages / {workers} workers / {workers} threads "
        f"→ {base_url}",
        file=sys.stderr,
    )

    wall0 = perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_worker, base_url, thread_ids[i], counts[i], text)
            for i in range(workers)
        ]
        results = [f.result() for f in futures]
    wall = perf_counter() - wall0

    all_lat: list[float] = []
    errors = 0
    for lat, err in results:
        all_lat.extend(lat)
        errors += err
    all_lat.sort()

    ok = len(all_lat)
    msg_per_s = ok / wall if wall > 0 else float("nan")
    summary = {
        "base_url": base_url,
        "workers": workers,
        "threads": workers,
        "messages_requested": total,
        "messages_ok": ok,
        "errors": errors,
        "wall_seconds": round(wall, 3),
        "msg_per_s": round(msg_per_s, 1),
        "append_ms_p50": round(_percentile(all_lat, 50), 2),
        "append_ms_p90": round(_percentile(all_lat, 90), 2),
        "append_ms_p99": round(_percentile(all_lat, 99), 2),
        "append_ms_max": round(all_lat[-1], 2) if all_lat else float("nan"),
        "sample_channel_id": channel_id,
        "sample_thread_id": thread_ids[0],
    }

    # A sample msgId from the busiest thread (for the §5/single-message PROFILE,
    # not strictly needed but handy).
    with httpx.Client(base_url=base_url, timeout=30.0) as client:
        rd = client.get(f"/threads/{thread_ids[0]}/messages", params={"limit": 1})
        if rd.status_code == 200 and rd.json():
            summary["sample_msg_id"] = rd.json()[0]["msgId"]

    print(json.dumps(summary, indent=2))
    if args.emit:
        with open(args.emit, "w") as fh:
            json.dump(summary, fh, indent=2)
    if errors:
        print(f"WARNING: {errors} request(s) failed", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
