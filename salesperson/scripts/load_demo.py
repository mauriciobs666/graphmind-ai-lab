#!/usr/bin/env python3
"""S15 load / concurrency harness (docs/plans/salesperson-ui.md §6.4, S15 row).

Drives N synthetic participants through join -> scripted conversation -> cart
add -> place order against a **running** storefront deployment
(`falkor-chat/scripts/start_demo.sh` or the manual sequence in
`salesperson/README.md`), the same shape `falkor-chat/scripts/load_test.sh`
uses for the platform's own REST append path. Reports latency percentiles
**split by route class** (poll reads vs. agent turns), asserts cross-
participant isolation on every response that carries participant-identifying
content, and — mode `stub` only — measures FalkorDB's queued-query headroom
under a live `reset-all` call.

Zero third-party dependencies: stdlib `urllib`/`concurrent.futures` only, and
`redis-cli` (already on this box, `/usr/bin/redis-cli`) shelled out to for the
`GRAPH.INFO` queue-depth sample — no Python `redis`/`httpx` client is assumed
to exist anywhere under `salesperson/`.

It honours the `409 TurnInProgress` contract rather than firing blind: each
participant waits for `turn.state == 'idle'` (polling `GET /shop/api/state`,
same 2 s cadence a real client uses — R10's own assumption) before posting its
next scripted line, and a `409` it *does* see (a race, not this harness's own
doing) is counted and waited out rather than treated as a hard failure.

Two run shapes (see plan §6.4):

    # Run A — stub LLM, single fixed concurrency, isolates server + graph.
    python3 load_demo.py stub --base-url http://127.0.0.1:8000 \\
        --participants 50 --turns 5 \\
        --presenter-key <key> --with-reset-check \\
        --out /tmp/run-a.json

    # Run B — live LLM, concurrency sweep, publishes the reply-latency curve
    # (and the dead-turn count beside it, never the curve alone).
    python3 load_demo.py live --base-url http://127.0.0.1:8000 \\
        --concurrencies 1,2,4,8,16,32,50 --turns 1 \\
        --out /tmp/run-b.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

POLL_INTERVAL_S = 2.0  # matches the SPA's own poll cadence (R10)
DEFAULT_TURN_WAIT_CEILING_S = 200.0  # > the 180s agent-call timeout (§4.4)

SCRIPTED_TURNS = [
    "Hi there!",
    "What products do you have?",
    "Add the Wireless Mouse Pro to my cart.",
    "I'd like to place my order.",
    "What's the status of my order?",
]

LANGUAGES = ["en", "pt-BR", "es"]


# ── HTTP plumbing (stdlib only) ──────────────────────────────────────────────


def http_json(method: str, url: str, token: str | None = None, body=None, timeout: float = 30.0):
    """Returns (status, parsed_json_or_None, elapsed_ms). Never raises on 4xx/5xx."""
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            elapsed = (time.monotonic() - t0) * 1000.0
            raw = resp.read()
            parsed = json.loads(raw.decode("utf-8")) if raw else None
            return resp.status, parsed, elapsed
    except urllib.error.HTTPError as exc:
        elapsed = (time.monotonic() - t0) * 1000.0
        raw = exc.read()
        try:
            parsed = json.loads(raw.decode("utf-8")) if raw else None
        except json.JSONDecodeError:
            parsed = {"_raw": raw.decode("utf-8", errors="replace")}
        return exc.code, parsed, elapsed
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        elapsed = (time.monotonic() - t0) * 1000.0
        return -1, {"_transport_error": str(exc)}, elapsed


# ── shared result accumulator (thread-safe) ─────────────────────────────────


@dataclass
class Results:
    lock: threading.Lock = field(default_factory=threading.Lock)
    samples: dict = field(default_factory=lambda: {"state_poll": [], "agent_turn": [], "write": [], "read": []})
    errors: list = field(default_factory=list)
    isolation_violations: list = field(default_factory=list)
    turn_in_progress_seen: int = 0
    dead_turns_seen: int = 0
    turns_completed: int = 0
    turns_timed_out: int = 0

    def record(self, kind: str, ms: float) -> None:
        with self.lock:
            self.samples[kind].append(ms)

    def error(self, msg: str) -> None:
        with self.lock:
            self.errors.append(msg)

    def violation(self, msg: str) -> None:
        with self.lock:
            self.isolation_violations.append(msg)

    def note_409(self) -> None:
        with self.lock:
            self.turn_in_progress_seen += 1

    def note_dead_turn(self) -> None:
        with self.lock:
            self.dead_turns_seen += 1

    def note_turn_done(self) -> None:
        with self.lock:
            self.turns_completed += 1

    def note_turn_timeout(self) -> None:
        with self.lock:
            self.turns_timed_out += 1

    def percentiles(self, kind: str) -> dict:
        with self.lock:
            data = sorted(self.samples[kind])
        if not data:
            return {"n": 0}
        def pct(p):
            if len(data) == 1:
                return data[0]
            k = (len(data) - 1) * p
            f = int(k)
            c = min(f + 1, len(data) - 1)
            if f == c:
                return data[f]
            return data[f] + (data[c] - data[f]) * (k - f)
        return {
            "n": len(data),
            "min": round(data[0], 1),
            "p50": round(pct(0.50), 1),
            "p95": round(pct(0.95), 1),
            "p99": round(pct(0.99), 1),
            "max": round(data[-1], 1),
        }


# ── one participant's flow ──────────────────────────────────────────────────


def run_participant(base_url: str, idx: int, turns: int, results: Results, turn_wait_ceiling: float) -> dict | None:
    name = f"LoadP{idx:04d}"
    language = LANGUAGES[idx % len(LANGUAGES)]

    status, resp, ms = http_json(
        "POST", f"{base_url}/shop/api/session", body={"displayName": name, "language": language}
    )
    results.record("write", ms)
    if status != 200 or resp is None:
        results.error(f"participant {name}: join failed status={status} body={resp}")
        return None
    if resp.get("displayName") != name:
        results.violation(
            f"participant {name}: join response displayName={resp.get('displayName')!r} != {name!r}"
        )
    # Bearer credential is `<participantId>.<token>` (§5.2), not the bare token.
    participant_id = resp["participantId"]
    token = f"{participant_id}.{resp['token']}"

    for turn_idx in range(turns):
        text = SCRIPTED_TURNS[turn_idx % len(SCRIPTED_TURNS)]
        status, resp, ms = http_json(
            "POST", f"{base_url}/shop/api/messages", token=token, body={"text": text}
        )
        results.record("write", ms)
        if status == 409:
            results.note_409()
            # A turn is already in flight for this participant (a race, not
            # this harness's own doing — it always waits for idle first). Do
            # not fire another post; just fall through to polling for idle.
        elif status != 200:
            results.error(f"participant {name}: post_message status={status} body={resp}")
            continue

        turn_start = time.monotonic()
        deadline = turn_start + turn_wait_ceiling
        last_state = None
        while time.monotonic() < deadline:
            time.sleep(POLL_INTERVAL_S)
            sstatus, sresp, sms = http_json("GET", f"{base_url}/shop/api/state", token=token)
            results.record("state_poll", sms)
            if sstatus != 200 or sresp is None:
                results.error(f"participant {name}: state poll status={sstatus} body={sresp}")
                break
            profile_name = (sresp.get("profile") or {}).get("name")
            if profile_name is not None and profile_name != name:
                results.violation(
                    f"participant {name} (token-scoped): /state returned profile.name="
                    f"{profile_name!r}, expected {name!r} — cross-participant leak"
                )
            last_state = sresp
            if sresp.get("turn", {}).get("state") == "idle":
                break
        else:
            results.note_turn_timeout()

        turn_ms = (time.monotonic() - turn_start) * 1000.0
        results.record("agent_turn", turn_ms)
        if last_state is not None:
            if last_state.get("turn", {}).get("state") == "idle":
                results.note_turn_done()
            if last_state.get("turn", {}).get("lastTurn") == "failed":
                results.note_dead_turn()

    status, resp, ms = http_json("GET", f"{base_url}/shop/api/catalog", token=token)
    results.record("read", ms)
    if status != 200:
        results.error(f"participant {name}: catalog read status={status}")

    return {"participantId": participant_id, "displayName": name, "token": token}


# ── GRAPH.INFO queue-depth sampler (Run A's reset-under-load check) ─────────


def sample_waiting_queries(redis_host: str, redis_port: int) -> int:
    """One `GRAPH.INFO` sample -> count of lines under `# Waiting queries`.

    Instance-wide, no graph key (`falkordb-quirks.md`, verified 2026-09-08).
    Counts non-empty lines after the `# Waiting queries` header up to the next
    blank-then-header boundary (`Object Pool` or another `#` section).
    """
    try:
        out = subprocess.run(
            ["redis-cli", "-h", redis_host, "-p", str(redis_port), "GRAPH.INFO"],
            capture_output=True, text=True, timeout=2,
        ).stdout
    except Exception:
        return -1
    lines = out.splitlines()
    try:
        start = next(i for i, ln in enumerate(lines) if ln.strip() == "# Waiting queries")
    except StopIteration:
        return -1
    count = 0
    for ln in lines[start + 1:]:
        stripped = ln.strip()
        if stripped == "" or stripped == "Object Pool" or stripped.startswith("#"):
            break
        count += 1
    return count


def reset_headroom_check(
    base_url: str,
    presenter_key: str,
    redis_host: str,
    redis_port: int,
    poll_seconds_before: float = 3.0,
) -> dict:
    """Fires `presenter/reset-all` while sampling `GRAPH.INFO` at high frequency.

    Assumes `run_participants` has already been called and its participants are
    steadily polling `/state` in the background at the 2 s cadence (the caller
    is responsible for keeping that poll load alive across this call) — this
    function only logs in as presenter, samples, and fires the reset.
    """
    status, resp, _ = http_json(
        "POST", f"{base_url}/shop/api/presenter/session", body={"key": presenter_key}
    )
    if status != 200:
        return {"error": f"presenter login failed status={status} body={resp}"}
    # Bearer credential is `presenter.<token>` (§5.2), not the bare token.
    presenter_token = f"presenter.{resp['token']}"

    peak = -1
    samples = []
    stop = threading.Event()

    def sampler():
        nonlocal peak
        while not stop.is_set():
            depth = sample_waiting_queries(redis_host, redis_port)
            if depth >= 0:
                samples.append(depth)
                peak = max(peak, depth)
            time.sleep(0.02)

    t = threading.Thread(target=sampler, daemon=True)
    t.start()
    time.sleep(poll_seconds_before)  # let steady-state poll load establish
    t0 = time.monotonic()
    rstatus, rresp, rms = http_json(
        "POST", f"{base_url}/shop/api/presenter/reset-all", token=presenter_token, timeout=30
    )
    elapsed_ms = (time.monotonic() - t0) * 1000.0
    time.sleep(0.5)
    stop.set()
    t.join(timeout=2)

    return {
        "reset_status": rstatus,
        "reset_response": rresp,
        "reset_wall_ms": round(elapsed_ms, 1),
        "peak_waiting_queries": peak,
        "num_samples": len(samples),
    }


# ── steady-state poller (keeps participants "on" after their scripted turns) ─


def steady_poll(base_url: str, participant: dict, stop: threading.Event, results: Results) -> None:
    token = participant["token"]
    name = participant["displayName"]
    while not stop.is_set():
        status, resp, ms = http_json("GET", f"{base_url}/shop/api/state", token=token, timeout=10)
        if status == 401:
            # Expected once reset-all invalidates this token — not an error.
            return
        results.record("state_poll", ms)
        if status == 200 and resp is not None:
            profile_name = (resp.get("profile") or {}).get("name")
            if profile_name is not None and profile_name != name:
                results.violation(
                    f"participant {name}: steady-poll /state returned profile.name="
                    f"{profile_name!r} — cross-participant leak"
                )
        time.sleep(POLL_INTERVAL_S)


# ── Run A: stub-LLM, fixed concurrency ───────────────────────────────────────


def run_stub(args) -> dict:
    results = Results()
    with ThreadPoolExecutor(max_workers=args.participants) as pool:
        futures = [
            pool.submit(run_participant, args.base_url, i, args.turns, results, args.turn_wait_ceiling)
            for i in range(args.participants)
        ]
        participants = [f.result() for f in futures]
    participants = [p for p in participants if p is not None]

    report = {
        "mode": "stub",
        "participants_requested": args.participants,
        "participants_joined": len(participants),
        "turns_per_participant": args.turns,
        "latency_ms": {
            "state_poll": results.percentiles("state_poll"),
            "agent_turn": results.percentiles("agent_turn"),
            "write": results.percentiles("write"),
            "read": results.percentiles("read"),
        },
        "turn_in_progress_409_count": results.turn_in_progress_seen,
        "dead_turns_seen": results.dead_turns_seen,
        "turns_completed": results.turns_completed,
        "turns_timed_out": results.turns_timed_out,
        "errors": results.errors,
        "isolation_violations": results.isolation_violations,
    }

    if args.with_reset_check:
        if not args.presenter_key:
            report["reset_headroom_check"] = {"error": "no --presenter-key supplied"}
        else:
            stop = threading.Event()
            with ThreadPoolExecutor(max_workers=len(participants) or 1) as pool:
                poll_futures = [
                    pool.submit(steady_poll, args.base_url, p, stop, results) for p in participants
                ]
                report["reset_headroom_check"] = reset_headroom_check(
                    args.base_url, args.presenter_key, args.redis_host, args.redis_port,
                )
                stop.set()
                for f in poll_futures:
                    f.result(timeout=5)
            report["latency_ms"]["state_poll_incl_steady"] = results.percentiles("state_poll")

    return report


# ── Run B: live-LLM, concurrency sweep ───────────────────────────────────────


def run_live_sweep(args) -> dict:
    sweep_results = []
    for concurrency in args.concurrencies:
        results = Results()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [
                pool.submit(run_participant, args.base_url, i, args.turns, results, args.turn_wait_ceiling)
                for i in range(concurrency)
            ]
            participants = [f.result() for f in futures]
        participants = [p for p in participants if p is not None]

        sweep_results.append(
            {
                "concurrency": concurrency,
                "participants_joined": len(participants),
                "agent_turn_latency_ms": results.percentiles("agent_turn"),
                "state_poll_latency_ms": results.percentiles("state_poll"),
                "turns_completed": results.turns_completed,
                "turns_timed_out": results.turns_timed_out,
                "dead_turns_seen (turn.lastTurn == 'failed')": results.dead_turns_seen,
                "turn_in_progress_409_count": results.turn_in_progress_seen,
                "errors": results.errors,
                "isolation_violations": results.isolation_violations,
            }
        )
        print(
            f"[live sweep] concurrency={concurrency} -> "
            f"agent_turn p50/p95={sweep_results[-1]['agent_turn_latency_ms'].get('p50')}/"
            f"{sweep_results[-1]['agent_turn_latency_ms'].get('p95')} ms, "
            f"dead_turns={results.dead_turns_seen}, timeouts={results.turns_timed_out}",
            file=sys.stderr,
        )

    return {"mode": "live", "turns_per_participant": args.turns, "sweep": sweep_results}


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="run_mode", required=True)

    p_stub = sub.add_parser("stub", help="Run A — stub LLM, fixed concurrency, isolates server+graph")
    p_stub.add_argument("--base-url", default="http://127.0.0.1:8000")
    p_stub.add_argument("--participants", type=int, default=50)
    p_stub.add_argument("--turns", type=int, default=5)
    p_stub.add_argument("--turn-wait-ceiling", type=float, default=DEFAULT_TURN_WAIT_CEILING_S)
    p_stub.add_argument("--with-reset-check", action="store_true")
    p_stub.add_argument("--presenter-key", default="")
    p_stub.add_argument("--redis-host", default="127.0.0.1")
    p_stub.add_argument("--redis-port", type=int, default=6379)
    p_stub.add_argument("--out", default="")

    p_live = sub.add_parser("live", help="Run B — live LLM, concurrency sweep, publishes the latency curve")
    p_live.add_argument("--base-url", default="http://127.0.0.1:8000")
    p_live.add_argument("--concurrencies", default="1,2,4,8,16,32,50")
    p_live.add_argument("--turns", type=int, default=1)
    p_live.add_argument("--turn-wait-ceiling", type=float, default=DEFAULT_TURN_WAIT_CEILING_S)
    p_live.add_argument("--out", default="")

    args = parser.parse_args()

    if args.run_mode == "stub":
        report = run_stub(args)
    else:
        args.concurrencies = [int(x) for x in args.concurrencies.split(",") if x.strip()]
        report = run_live_sweep(args)

    text = json.dumps(report, indent=2, default=str)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)


if __name__ == "__main__":
    main()
