#!/usr/bin/env python3
"""K-066 U3 analysis of a `language_salience_probe.py` run: turn latency, dead turns,
and failure attribution by call site (Q3, Q4, Q5(a)).

Replaces the salience2 run's lost scratch `attribute_failures.py` and the architect's
p95 computation (review `falkor-chat/docs/reviews/salesperson-single-model.md` P3-1).

* **Latency** — nearest-rank percentiles on sorted `turn_ms` of *replied* turn records
  (`reply_count > 0`), exactly the baseline's definition (plan §7: salience2 p50 6.0 s,
  p95 24.1 s, max 62.3 s). Also per turn index.
* **Dead turns** — `dead_turn` true (the DEF-3 `lastTurn == 'failed'` latch), over turns
  attempted; Wilson 95% interval.
* **Attribution** — every `falkorchat.storefront` failure event in the demo log
  (`storefront turn failed (participantId=…)` plus its traceback, or `drove to a failed run
  without raising (participantId=…, error=…)`) is joined to a dead turn by participantId and
  the turn's `[posted_at, idle_at]` window. The call site is split on the traceback:
  a frame in `tools.py` (`QueryGraphDataTool.run` → `llm.complete`) is the **tool** site;
  otherwise `executor._run_agent_node` → `llm.chat` is the **agent** site. Events within
  `--burst-ms` of each other form one burst.

Stdlib only; reads, never writes (prints text, or JSON with --json).

    python3 single_model_analysis.py RUN.jsonl --demo-log demo.log [--json]
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from datetime import datetime

_TS = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?[+-]\d\d:?\d\d)")
_PID = re.compile(r"participantId=([^,)\s]+)")


def wilson(x: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    p = x / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - h) / d), min(1.0, (c + h) / d))


def nearest_rank(sorted_vals: list[float], q: float) -> float | None:
    if not sorted_vals:
        return None
    k = max(1, math.ceil(q * len(sorted_vals)))
    return round(sorted_vals[k - 1], 1)


def _dt(s: str | None) -> datetime | None:
    return datetime.fromisoformat(s) if s else None


def parse_failures(path: str) -> list[dict]:
    """Group the demo log into failure events (header line + following untimestamped lines)."""
    events: list[dict] = []
    cur: dict | None = None
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = _TS.match(line)
            if m:
                cur = None
                if "falkorchat.storefront" in line and (
                    "storefront turn failed" in line or "drove to a failed run" in line
                ):
                    pm = _PID.search(line)
                    cur = {"at": m.group(1), "participantId": pm.group(1) if pm else None,
                           "header": line.rstrip("\n"), "body": []}
                    events.append(cur)
            elif cur is not None:
                cur["body"].append(line.rstrip("\n"))
    for e in events:
        text = "\n".join([e["header"], *e["body"]])
        e["site"] = "tool" if re.search(r'tools\.py", line \d+, in run', text) else (
            "agent" if "_run_agent_node" in text else "unknown")
        e["context_exceeded"] = "Context size has been exceeded" in text
        em = re.search(r"ProviderCallError: (lmstudio/\S+)", text)
        e["model"] = em.group(1).rstrip(":") if em else None
        last = [ln for ln in e["body"] if ln.strip()] or [e["header"]]
        e["error_line"] = last[-1][:300]
        del e["body"]
    return events


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--demo-log")
    ap.add_argument("--burst-ms", type=float, default=1000.0)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    recs = [json.loads(ln) for ln in open(a.run, encoding="utf-8") if ln.strip()]
    turns = [r for r in recs if r.get("type") == "turn"]
    replied = sorted(r["turn_ms"] / 1000.0 for r in turns if (r.get("reply_count") or 0) > 0)
    dead = [r for r in turns if r.get("dead_turn")]
    lo, hi = wilson(len(dead), len(turns))
    by_idx: dict[int, list[float]] = defaultdict(list)
    for r in turns:
        if (r.get("reply_count") or 0) > 0:
            by_idx[r["turn_index"]].append(r["turn_ms"] / 1000.0)
    trials = sorted({r["trial"] for r in recs if r.get("type") == "trial_end"})
    out: dict = {
        "trials_complete": len(trials),
        "turns": len(turns), "replied": len(replied), "dead": len(dead),
        "dead_rate": len(dead) / len(turns) if turns else None, "dead_wilson95": [lo, hi],
        "no_reply": sum(1 for r in turns if not r.get("reply_count")),
        "timed_out": sum(1 for r in turns if r.get("timed_out")),
        "post_non200": sum(1 for r in turns if r.get("post_status") != 200),
        "errors_nonempty": sum(1 for r in turns if r.get("errors")),
        "latency_s": {"p50": nearest_rank(replied, 0.50), "p95": nearest_rank(replied, 0.95),
                      "max": round(replied[-1], 1) if replied else None},
        "p95_by_turn_index": {i: nearest_rank(sorted(v), 0.95) for i, v in sorted(by_idx.items())},
        "dead_by_language": {},
        "dead_turns": [{k: r.get(k) for k in ("trial", "language", "turn_index", "participantId",
                                               "posted_at", "idle_at", "turn_ms", "reply_count")} for r in dead],
    }
    for r in dead:
        out["dead_by_language"][r["language"]] = out["dead_by_language"].get(r["language"], 0) + 1
    runs = [r for r in recs if r.get("type") in ("run_start", "run_end")]
    out["window"] = [runs[0].get("at") if runs else None, runs[-1].get("at") if runs else None]

    if a.demo_log:
        t0 = _dt(turns[0]["posted_at"]) if turns else None
        t1 = max((_dt(r.get("idle_at")) or _dt(r["posted_at"]) for r in turns), default=None)
        evs = [e for e in parse_failures(a.demo_log)
               if t0 and t1 and t0 <= _dt(e["at"]) <= t1]
        # join to dead turns
        for e in evs:
            e["matched_turn"] = None
            for r in dead:
                if r.get("participantId") != e["participantId"]:
                    continue
                s, end = _dt(r["posted_at"]), _dt(r.get("idle_at")) or t1
                if s <= _dt(e["at"]) <= end:
                    e["matched_turn"] = {"trial": r["trial"], "language": r["language"],
                                         "turn_index": r["turn_index"]}
                    break
        evs.sort(key=lambda e: e["at"])
        burst, prev = 0, None
        for e in evs:
            t = _dt(e["at"])
            if prev is None or (t - prev).total_seconds() * 1000 > a.burst_ms:
                burst += 1
            e["burst"] = burst
            prev = t
        sizes: dict[int, int] = defaultdict(int)
        for e in evs:
            sizes[e["burst"]] += 1
        for e in evs:
            e["burst_size"] = sizes[e["burst"]]
        matched_dead = {(m["trial"], m["language"], m["turn_index"]) for e in evs if (m := e["matched_turn"])}
        out["failures"] = {
            "events": len(evs),
            "by_site": {s: sum(1 for e in evs if e["site"] == s) for s in ("agent", "tool", "unknown")},
            "context_exceeded": sum(1 for e in evs if e["context_exceeded"]),
            "by_model": {m: sum(1 for e in evs if e["model"] == m) for m in {e["model"] for e in evs}},
            "unmatched_events": sum(1 for e in evs if not e["matched_turn"]),
            "dead_turns_unattributed": len(dead) - len(matched_dead),
            "bursts": len(sizes), "multi_event_bursts": sum(1 for v in sizes.values() if v > 1),
            "list": evs,
        }

    if a.json:
        print(json.dumps(out, indent=2, ensure_ascii=False, default=str))
        return 0
    L = out["latency_s"]
    print(f"trials complete {out['trials_complete']}  turns {out['turns']}  replied {out['replied']}  "
          f"dead {out['dead']} ({(out['dead_rate'] or 0):.2%}, Wilson [{lo:.2%}, {hi:.2%}])  by lang {out['dead_by_language']}")
    print(f"no_reply {out['no_reply']}  timed_out {out['timed_out']}  post_non200 {out['post_non200']}  "
          f"errors_nonempty {out['errors_nonempty']}")
    print(f"turn latency (replied, nearest-rank): p50 {L['p50']} s  p95 {L['p95']} s  max {L['max']} s")
    print(f"p95 by turn index: {out['p95_by_turn_index']}")
    for d in out["dead_turns"]:
        print(f"  dead: {d}")
    if "failures" in out:
        F = out["failures"]
        print(f"failure events {F['events']}  by site {F['by_site']}  context_exceeded {F['context_exceeded']}  "
              f"by model {F['by_model']}  unmatched {F['unmatched_events']}  dead unattributed {F['dead_turns_unattributed']}  "
              f"bursts {F['bursts']} (multi-event {F['multi_event_bursts']})")
        for e in F["list"]:
            print(f"  {e['at']} burst#{e['burst']}(size {e['burst_size']}) site={e['site']} ctx={e['context_exceeded']} "
                  f"model={e['model']} turn={e['matched_turn']} | {e['error_line'][:160]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
