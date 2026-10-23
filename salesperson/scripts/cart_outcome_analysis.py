#!/usr/bin/env python3
"""K-078 S7 outcome analysis of a storefront probe run (plan
`falkor-chat/docs/plans/salesperson-cart-product-id.md` §6.3).

Reads one or more probe JSONL files (`language_salience_probe.py run` or
`refire_probe.py run` — both write `turn` records carrying `participantId`,
`language`, `prompt`, `turn_index`, `turn_ms`, `reply_text`) and joins them to:

* **the graph, read-only** (`GRAPH.RO_QUERY` only): each participant's
  `(:Cart {customerId})-[:HAS_ITEM]->(:CartItem)` and
  `(:Customer {customerId})-[:PLACED]->(:Order)-[:HAS_LINE]->(:OrderLine)` lines
  in `ws:<id>`, the `WorkflowRun.ctx` threadId of every run id named in a demo-log
  hold line, and the `reference` catalog (`productId`, `name`, `price`) for the
  name→price sweep. Filtered by participant id, never "everything in ws:demo"
  (the graph holds carts from earlier runs).
* **the demo log** (optional): `off-turn write held` (K-058) and `same-turn write
  held` (K-061) WARNING lines, attributed run → thread → participant.
* **an LLM request/response log** (optional, `--llm-log`): one JSON line per
  `/v1/chat/completions` call with full `request`/`response` bodies, as written by
  an observe-only proxy in front of LM Studio. Each storefront turn is rebuilt from
  its last (longest) request: the executor resends the turn's own tool calls and
  tool results every iteration, so that request plus its response is the whole
  turn. A turn is keyed by `(threadId from the CONTEXT block, trigger prompt)`.

Measures (per language, Wilson 95% intervals):
  M1 — target product (default `wireless-mouse-pro`) in cart OR a placed order
       (also reported as cart-only and order-only, the plan's alternative views);
  M2 — participants with a line for any other product;
  M3 — participants whose target quantity summed over CartItem + OrderLine > 1;
  holds (log and LLM-log views), K-058 hold recovery, K-061 holds;
  dead turns; nearest-rank p50/p95/max of replied `turn_ms`, overall, per turn
  index, and at the add turn per language;
  tool mix on the add turn; first-lookup miss rate (result-based and
  argument-based); writes on turns whose prompt does not name a product;
  name→price sweep over every reply (each wrong pair tagged with whether its turn
  had a `lookup_product_fact` miss followed by `query_graph_data`, impl review F4);
  non-`$` currency symbols.

Stdlib + the `redis` client (run it with `falkor-chat/server/.venv/bin/python`).
Reads only; prints text, or JSON with `--json`.

    falkor-chat/server/.venv/bin/python salesperson/scripts/cart_outcome_analysis.py \\
        run.jsonl --demo-log demo.log --llm-log proxy.jsonl [--json]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict

TARGET_DEFAULT = "wireless-mouse-pro"
ADD_PROMPT_RE = re.compile(r"\badd\b", re.IGNORECASE)


# ── small helpers ───────────────────────────────────────────────────────────


def wilson(x: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    p = x / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h) * 100, min(1.0, c + h) * 100)


def frac(x: int, n: int) -> str:
    if n == 0:
        return f"{x}/0"
    lo, hi = wilson(x, n)
    return f"{x}/{n} ({x / n * 100:.1f}%, [{lo:.1f}, {hi:.1f}])"


def nearest_rank(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    k = max(1, math.ceil(pct / 100 * len(s)))
    return s[k - 1]


def norm_id(v) -> str:
    return v.strip().lower() if isinstance(v, str) else ""


def norm_name(v) -> str:
    return re.sub(r"\s+", " ", v.strip().lower()) if isinstance(v, str) else ""


def load_jsonl(path: str) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # torn final line of a killed run
    return out


def as_obj(content):
    if isinstance(content, dict):
        return content
    if not isinstance(content, str):
        return None
    try:
        v = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return None
    return v if isinstance(v, dict) else None


# ── graph (read-only) ───────────────────────────────────────────────────────


class Graph:
    def __init__(self, url: str):
        import redis  # server venv

        self._r = redis.Redis.from_url(url, decode_responses=True)

    def ro(self, graph: str, cypher: str, params: dict | None = None) -> list[list]:
        prefix = ""
        if params:
            prefix = "CYPHER " + " ".join(f"{k}={json.dumps(v)}" for k, v in params.items()) + " "
        res = self._r.execute_command("GRAPH.RO_QUERY", graph, prefix + cypher)
        return res[1] if len(res) >= 2 else []


def chunks(xs: list, n: int = 200):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


# ── demo log ────────────────────────────────────────────────────────────────

_HOLD = re.compile(r"off-turn write held: run (\w+) step \S+ did not dispatch (\w+)\((\w+)=(.*)\) — ")
_DEDUP = re.compile(r"same-turn write held: run (\w+) step \S+ did not re-dispatch (\w+)\(")
_TS = re.compile(r"^(\S+) ")


def parse_demo_log(path: str, t_from: str | None, t_to: str | None) -> dict:
    holds, dedups = [], []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m_ts = _TS.match(line)
            ts = m_ts.group(1) if m_ts else ""
            if t_from and ts < t_from:
                continue
            if t_to and ts > t_to:
                continue
            m = _HOLD.search(line)
            if m:
                holds.append({"at": ts, "runId": m.group(1), "tool": m.group(2), "value": m.group(4)})
                continue
            m = _DEDUP.search(line)
            if m:
                dedups.append({"at": ts, "runId": m.group(1), "tool": m.group(2)})
    return {"holds": holds, "dedups": dedups}


# ── LLM log → turns ─────────────────────────────────────────────────────────

_CTX = re.compile(r"CONTEXT:\s*(\{.*\})\s*$", re.DOTALL)


def _trigger(messages: list[dict]) -> tuple[str | None, str | None, str | None]:
    """(threadId, language, prompt) of an agent request, from its last user message."""
    users = [m for m in messages if m.get("role") == "user"]
    if not users:
        return None, None, None
    text = users[-1].get("content") or ""
    if not isinstance(text, str):
        return None, None, None
    m = _CTX.search(text)
    if not m:
        return None, None, None
    try:
        ctx = json.loads(m.group(1))
    except json.JSONDecodeError:
        return None, None, None
    body = text[: m.start()].rstrip()
    last = body.split("\n\n")[-1] if body else ""
    prompt = last.split(": ", 1)[1] if ": " in last else last
    return ctx.get("threadId"), ctx.get("language"), prompt.strip()


def build_turns(llm_records: list[dict]) -> tuple[dict, int]:
    """{(threadId, prompt): turn} plus the count of non-agent (internal) calls."""
    best: dict[tuple[str, str], dict] = {}
    calls: Counter = Counter()
    internal = 0
    for r in llm_records:
        req = r.get("request")
        if not isinstance(req, dict):
            continue
        msgs = req.get("messages") or []
        tid, lang, prompt = _trigger(msgs)
        if not tid:
            internal += 1
            continue
        key = (tid, prompt)
        calls[key] += 1
        if key not in best or len(msgs) >= len(best[key]["request"]["messages"]):
            best[key] = r
    turns = {}
    for key, r in best.items():
        msgs = r["request"]["messages"]
        results = {m.get("tool_call_id"): m.get("content") for m in msgs if m.get("role") == "tool"}
        seq = []
        for m in msgs:
            if m.get("role") == "assistant":
                for tc in m.get("tool_calls") or []:
                    fn = tc.get("function") or {}
                    args = fn.get("arguments")
                    try:
                        args = json.loads(args) if isinstance(args, str) else (args or {})
                    except json.JSONDecodeError:
                        args = {"_raw": args}
                    seq.append({"name": fn.get("name"), "args": args,
                                "result": results.get(tc.get("id")), "executed": True})
        try:
            final = r["response"]["choices"][0]["message"]
        except (KeyError, IndexError, TypeError):
            final = {}
        for tc in final.get("tool_calls") or []:
            fn = tc.get("function") or {}
            args = fn.get("arguments")
            try:
                args = json.loads(args) if isinstance(args, str) else (args or {})
            except json.JSONDecodeError:
                args = {"_raw": args}
            seq.append({"name": fn.get("name"), "args": args, "result": None, "executed": False})
        _, lang, _ = _trigger(msgs)
        turns[key] = {"threadId": key[0], "prompt": key[1], "language": lang,
                      "calls": seq, "iterations": calls[key]}
    return turns, internal


def classify_call(c: dict) -> str:
    name, res = c["name"], c["result"]
    obj = as_obj(res)
    if name == "lookup_product_fact":
        if obj is None:
            return "unknown" if res is None else "error"
        return "hit" if obj.get("found") else "miss"
    if name in ("add_to_cart", "remove_from_cart"):
        if res is None:
            return "unknown"
        if obj is None:
            return "error"  # executor string: missing argument / ungranted / service error
        if obj.get("held"):
            reason = str(obj.get("reason") or "")
            return "held_dedup" if "already succeeded" in reason else "held"
        if obj.get("found") is False:
            return "not_found"
        return "ok"
    return "ok" if res is not None else "unknown"


def turn_facts(turn: dict, catalog_names: set[str], target: str) -> dict:
    calls = turn["calls"]
    for c in calls:
        c["cls"] = classify_call(c)
    lookups = [c for c in calls if c["name"] == "lookup_product_fact"]
    adds = [c for c in calls if c["name"] == "add_to_cart"]
    held = [c for c in adds if c["cls"] == "held"]
    ok_adds = [c for c in adds if c["cls"] == "ok"]
    first_held_idx = next((i for i, c in enumerate(calls) if c["name"] == "add_to_cart" and c["cls"] == "held"), None)
    recovered_any = recovered_target = False
    if first_held_idx is not None:
        later = [c for c in calls[first_held_idx + 1:] if c["name"] == "add_to_cart" and c["cls"] == "ok"]
        recovered_any = bool(later)
        recovered_target = any(norm_id(c["args"].get("productId")) == target for c in later)
    miss_idx = next((i for i, c in enumerate(calls) if c["name"] == "lookup_product_fact" and c["cls"] == "miss"), None)
    miss_then_qgd = miss_idx is not None and any(c["name"] == "query_graph_data" for c in calls[miss_idx + 1:])
    first = lookups[0] if lookups else None
    return {
        "tools": [c["name"] for c in calls],
        "n_lookup": len(lookups),
        "first_lookup_cls": first["cls"] if first else None,
        "first_lookup_arg": (first["args"].get("name") if first else None),
        "first_lookup_arg_exact": (norm_name(first["args"].get("name")) in catalog_names) if first else None,
        "n_add": len(adds), "n_add_held": len(held), "n_add_ok": len(ok_adds),
        "n_add_dedup": sum(1 for c in adds if c["cls"] == "held_dedup"),
        "n_add_error": sum(1 for c in adds if c["cls"] == "error"),
        "n_add_notfound": sum(1 for c in adds if c["cls"] == "not_found"),
        "ok_add_ids": [norm_id(c["args"].get("productId")) for c in ok_adds],
        "held_values": [c["args"].get("productId", c["args"]) for c in held],
        "recovered_any": recovered_any, "recovered_target": recovered_target,
        "miss_then_qgd": miss_then_qgd,
        "n_remove": sum(1 for c in calls if c["name"] == "remove_from_cart"),
        "iterations": turn["iterations"],
    }


# ── name→price sweep ────────────────────────────────────────────────────────

_PRICE = re.compile(
    r"(?P<sym>US\$|R\$|\$|€|£)\s?(?P<a>\d{1,5}(?:[.,]\d{3})*(?:[.,]\d{1,2})?)"
    r"|(?P<b>\d{1,5}(?:[.,]\d{1,2})?)\s?(?P<sym2>\$|€|£|USD|US\$|dólares|dollars|euros|reais)",
    re.IGNORECASE,
)


def _num(s: str) -> float | None:
    s = s.strip()
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+,\d{1,2}", s):  # 1.299,99
        s = s.replace(".", "").replace(",", ".")
    elif re.fullmatch(r"\d+,\d{1,2}", s):  # 29,99
        s = s.replace(",", ".")
    else:
        s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def sweep_reply(text: str, catalog: list[dict]) -> list[dict]:
    """Pairs (catalog name, stated price) per line: each English catalog name is paired
    with the first price after it and before the next name; a line with one name and
    no price after it takes a single price stated before it."""
    pairs = []
    names = sorted(catalog, key=lambda p: -len(p["name"]))
    for line in (text or "").split("\n"):
        low = line.lower()
        hits, taken = [], [False] * len(line)
        for p in names:
            for m in re.finditer(re.escape(p["name"].lower()), low):
                if any(taken[m.start():m.end()]):
                    continue
                for i in range(m.start(), m.end()):
                    taken[i] = True
                hits.append((m.start(), m.end(), p))
        if not hits:
            continue
        hits.sort(key=lambda h: h[0])
        prices = []
        for m in _PRICE.finditer(line):
            v = _num(m.group("a") or m.group("b") or "")
            if v is not None:
                prices.append((m.start(), v, (m.group("sym") or m.group("sym2") or "")))
        for i, (s, e, p) in enumerate(hits):
            nxt = hits[i + 1][0] if i + 1 < len(hits) else len(line) + 1
            after = [pr for pr in prices if e <= pr[0] < nxt]
            chosen = after[0] if after else (prices[0] if len(hits) == 1 and len(prices) == 1 else None)
            if chosen is None:
                continue
            pairs.append({"productId": p["productId"], "name": p["name"], "stated": chosen[1],
                          "symbol": chosen[2], "actual": p["price"],
                          "correct": abs(chosen[1] - float(p["price"])) < 0.005})
    return pairs


_CURRENCY = re.compile(r"€|£|R\$|¥|\beuros?\b|\breais\b|\bEUR\b|\bBRL\b", re.IGNORECASE)


# ── main ────────────────────────────────────────────────────────────────────


def analyse(args) -> dict:
    turns_rec = []
    for p in args.runs:
        turns_rec += [r for r in load_jsonl(p) if r.get("type") == "turn"]
    if args.scripts:
        keep = set(args.scripts.split(","))
        turns_rec = [r for r in turns_rec if r.get("script") in keep]
    parts: dict[str, dict] = {}
    for r in turns_rec:
        pid = r["participantId"]
        parts.setdefault(pid, {"language": r["language"], "trial": r.get("trial"),
                               "script": r.get("script"), "turns": []})["turns"].append(r)
    langs = sorted({p["language"] for p in parts.values()}, key=lambda l: ["en", "pt-BR", "es"].index(l) if l in ["en", "pt-BR", "es"] else 9)
    g = Graph(args.redis_url)
    ws = args.graph
    target = args.target

    # graph lines
    lines: dict[str, list[dict]] = defaultdict(list)
    ids = list(parts)
    for ch in chunks(ids):
        for cid, pid_, q in g.ro(ws, "MATCH (c:Cart)-[:HAS_ITEM]->(i:CartItem) WHERE c.customerId IN $ids "
                                     "RETURN c.customerId, i.productId, i.quantity", {"ids": ch}):
            lines[cid].append({"src": "cart", "productId": pid_, "quantity": int(q or 0)})
        for cid, pid_, q, oid in g.ro(ws, "MATCH (cu:Customer)-[:PLACED]->(o:Order)-[:HAS_LINE]->(l:OrderLine) "
                                          "WHERE cu.customerId IN $ids RETURN cu.customerId, l.productId, l.quantity, o.orderId",
                                      {"ids": ch}):
            lines[cid].append({"src": "order", "productId": pid_, "quantity": int(q or 0), "orderId": oid})
    catalog = [{"productId": a, "name": b, "price": float(c)} for a, b, c in
               g.ro("reference", "MATCH (p:Product) RETURN p.productId, p.name, p.price")]
    catalog_names = {norm_name(p["name"]) for p in catalog}

    out: dict = {"participants": len(parts), "turn_records": len(turns_rec), "languages": langs,
                 "target": target, "per_language": {}, "inspect": {}}
    inspect_m2, inspect_m3 = [], []
    for lang in langs:
        P = {pid: p for pid, p in parts.items() if p["language"] == lang}
        n = len(P)
        cart = sum(1 for pid in P if any(l["src"] == "cart" and l["productId"] == target for l in lines[pid]))
        order = sum(1 for pid in P if any(l["src"] == "order" and l["productId"] == target for l in lines[pid]))
        either = sum(1 for pid in P if any(l["productId"] == target for l in lines[pid]))
        both = sum(1 for pid in P if {l["src"] for l in lines[pid] if l["productId"] == target} == {"cart", "order"})
        m2 = [pid for pid in P if any(l["productId"] != target for l in lines[pid])]
        m3 = [pid for pid in P if sum(l["quantity"] for l in lines[pid] if l["productId"] == target) > 1]
        inspect_m2 += [{"participantId": pid, "language": lang, "lines": lines[pid]} for pid in m2]
        inspect_m3 += [{"participantId": pid, "language": lang, "lines": lines[pid]} for pid in m3]
        T = [t for p in P.values() for t in p["turns"]]
        dead = sum(1 for t in T if t.get("dead_turn"))
        replied = [t for t in T if (t.get("reply_count") or 0) > 0]
        no_reply = len(T) - len(replied)
        out["per_language"][lang] = {
            "n": n, "M1_cart_or_order": either, "M1_cart_or_order_fmt": frac(either, n),
            "cart_only_view": cart, "order_view": order, "both": both,
            "M2_other_product_participants": len(m2), "M3_qty_gt1": len(m3),
            "turns": len(T), "dead_turns": dead, "no_reply_turns": no_reply,
            "timed_out": sum(1 for t in T if t.get("timed_out")),
        }
    out["inspect"] = {"M2": inspect_m2, "M3": inspect_m3}

    # latency
    replied = [t for t in turns_rec if (t.get("reply_count") or 0) > 0]
    lat = {"all": [t["turn_ms"] / 1000 for t in replied]}
    by_idx = defaultdict(list)
    by_idx_lang = defaultdict(list)
    for t in replied:
        by_idx[t["turn_index"]].append(t["turn_ms"] / 1000)
        by_idx_lang[(t["turn_index"], t["language"])].append(t["turn_ms"] / 1000)
    out["latency"] = {
        "all": {"n": len(lat["all"]), "p50": nearest_rank(lat["all"], 50), "p95": nearest_rank(lat["all"], 95),
                "max": max(lat["all"]) if lat["all"] else None},
        "by_turn_index": {i: {"n": len(v), "p50": nearest_rank(v, 50), "p95": nearest_rank(v, 95), "max": max(v)}
                          for i, v in sorted(by_idx.items())},
        "by_turn_index_language": {f"{i}/{l}": {"n": len(v), "p50": nearest_rank(v, 50), "p95": nearest_rank(v, 95),
                                                "max": max(v)} for (i, l), v in sorted(by_idx_lang.items())},
    }
    out["dead_turns_total"] = sum(1 for t in turns_rec if t.get("dead_turn"))
    out["dead_by_turn_index_language"] = dict(sorted(Counter(
        f"{t['turn_index']}/{t['language']}" for t in turns_rec if t.get("dead_turn")).items()))
    dpart = Counter(t["participantId"] for t in turns_rec if t.get("dead_turn"))
    out["participants_with_dead_turn_by_language"] = {
        l: sum(1 for pid, p in parts.items() if p["language"] == l and dpart[pid]) for l in langs}
    out["participants_with_dead_add_turn_by_language"] = {
        l: sum(1 for t in turns_rec if t["language"] == l and t.get("dead_turn") and ADD_PROMPT_RE.search(t["prompt"] or ""))
        for l in langs}

    # thread → participant
    thread_to_pid = {f"th-{pid}": pid for pid in parts}

    # demo log holds
    if args.demo_log:
        t_from = min((t["posted_at"] for t in turns_rec), default=None)
        t_to = max((t.get("idle_at") or t["posted_at"] for t in turns_rec), default=None)
        dl = parse_demo_log(args.demo_log, t_from, t_to)
        run_ids = sorted({h["runId"] for h in dl["holds"] + dl["dedups"]})
        run_thread = {}
        for ch in chunks(run_ids):
            for rid, ctx in g.ro(ws, "MATCH (r:WorkflowRun) WHERE r.runId IN $ids RETURN r.runId, r.ctx", {"ids": ch}):
                try:
                    run_thread[rid] = json.loads(ctx).get("threadId")
                except (TypeError, json.JSONDecodeError):
                    pass
        hl = Counter()
        hl_runs = defaultdict(set)
        vals = defaultdict(Counter)
        unattr = 0
        for h in dl["holds"]:
            pid = thread_to_pid.get(run_thread.get(h["runId"]))
            if pid is None:
                unattr += 1
                continue
            lang = parts[pid]["language"]
            hl[(lang, h["tool"])] += 1
            hl_runs[lang].add(h["runId"])
            vals[lang][h["value"]] += 1
        dd = Counter()
        for h in dl["dedups"]:
            pid = thread_to_pid.get(run_thread.get(h["runId"]))
            dd[parts[pid]["language"] if pid else "unattributed"] += 1
        out["demo_log"] = {
            "window": [t_from, t_to], "off_turn_holds_total": len(dl["holds"]),
            "off_turn_unattributed": unattr,
            "off_turn_by_language_tool": {f"{l}/{t}": c for (l, t), c in sorted(hl.items())},
            "off_turn_distinct_runs_by_language": {l: len(s) for l, s in hl_runs.items()},
            "held_values_by_language": {l: dict(c.most_common()) for l, c in vals.items()},
            "same_turn_holds_total": len(dl["dedups"]), "same_turn_by_language": dict(dd),
        }

    # LLM log
    turn_index_of_prompt = {}
    if args.llm_log:
        llm = load_jsonl(args.llm_log)
        lturns, internal = build_turns(llm)
        out["llm_log"] = {"chat_calls": sum(1 for r in llm if r.get("request")), "internal_calls": internal,
                          "agent_turns_rebuilt": len(lturns)}
        upstream = [r["upstream_ms"] for r in llm if r.get("request") and r.get("upstream_ms") is not None]
        overhead = [r["proxy_total_ms"] - r["upstream_ms"] for r in llm
                    if r.get("request") and r.get("upstream_ms") is not None]
        out["llm_log"]["proxy_in_process_overhead_ms"] = {
            "n": len(overhead), "p50": nearest_rank(overhead, 50), "p95": nearest_rank(overhead, 95),
            "max": max(overhead) if overhead else None}
        out["llm_log"]["upstream_ms"] = {"p50": nearest_rank(upstream, 50), "p95": nearest_rank(upstream, 95)}
        fails = Counter()
        for r in llm:
            if r.get("request") and r.get("status") != 200:
                site = "agent" if _trigger(r["request"].get("messages") or [])[0] else "internal(query_graph_data)"
                body = json.dumps(r.get("response") or r.get("response_raw") or "")
                kind = "context_exceeded" if "Context size has been exceeded" in body else body[:80]
                fails[(site, kind)] += 1
        out["llm_log"]["failed_calls"] = {f"{s} / {k}": v for (s, k), v in fails.most_common()}
        ok_pt = [r["response"]["usage"]["prompt_tokens"] for r in llm
                 if r.get("request") and r.get("status") == 200 and isinstance(r.get("response"), dict)
                 and isinstance(r["response"].get("usage"), dict)]
        out["llm_log"]["prompt_tokens_ok_calls"] = {"n": len(ok_pt), "p50": nearest_rank(ok_pt, 50),
                                                    "p95": nearest_rank(ok_pt, 95), "max": max(ok_pt) if ok_pt else None}
        per_turn = []
        matched = 0
        for t in turns_rec:
            key = (f"th-{t['participantId']}", t["prompt"])
            lt = lturns.get(key)
            if lt is None:
                continue
            matched += 1
            f = turn_facts(lt, catalog_names, target)
            f.update({"participantId": t["participantId"], "language": t["language"], "turn_index": t["turn_index"],
                      "prompt": t["prompt"], "script": t.get("script")})
            per_turn.append(f)
            turn_index_of_prompt[key] = f
        out["llm_log"]["probe_turns_matched"] = matched
        agg = {}
        for lang in langs:
            L = [f for f in per_turn if f["language"] == lang]
            add_turns = [f for f in L if ADD_PROMPT_RE.search(f["prompt"] or "")]
            first_turn_add = [f for f in add_turns]
            with_lookup = [f for f in first_turn_add if f["n_lookup"] > 0]
            held_turns = [f for f in add_turns if f["n_add_held"] > 0]
            other_turns = [f for f in L if not ADD_PROMPT_RE.search(f["prompt"] or "")]
            mix = Counter(x for f in add_turns for x in f["tools"])
            mix_turns = Counter(x for f in add_turns for x in set(f["tools"]))
            agg[lang] = {
                "add_turns": len(add_turns),
                "add_turn_tool_calls": dict(mix.most_common()),
                "add_turn_tool_presence_turns": dict(mix_turns.most_common()),
                "add_turn_iterations_mean": (sum(f["iterations"] for f in add_turns) / len(add_turns)) if add_turns else None,
                "add_turn_iterations_max": max((f["iterations"] for f in add_turns), default=None),
                "add_turns_with_lookup": len(with_lookup),
                "first_lookup_miss_result": sum(1 for f in with_lookup if f["first_lookup_cls"] == "miss"),
                "first_lookup_arg_not_exact": sum(1 for f in with_lookup if f["first_lookup_arg_exact"] is False),
                "first_lookup_args": dict(Counter(f["first_lookup_arg"] for f in with_lookup).most_common()),
                "add_turns_with_ok_add": sum(1 for f in add_turns if f["n_add_ok"] > 0),
                "add_turns_with_held_add": len(held_turns),
                "held_add_calls": sum(f["n_add_held"] for f in add_turns),
                "held_turns_recovered_any": sum(1 for f in held_turns if f["recovered_any"]),
                "held_turns_recovered_target": sum(1 for f in held_turns if f["recovered_target"]),
                "held_values": dict(Counter(str(v) for f in add_turns for v in f["held_values"]).most_common()),
                "add_turns_no_add_call": sum(1 for f in add_turns if f["n_add"] == 0),
                "add_error_calls": sum(f["n_add_error"] for f in add_turns),
                "add_notfound_calls": sum(f["n_add_notfound"] for f in add_turns),
                "dedup_held_calls_all_turns": sum(f["n_add_dedup"] for f in L),
                "writes_on_non_add_turns": [
                    {"participantId": f["participantId"], "turn_index": f["turn_index"], "prompt": f["prompt"],
                     "add_ok": f["n_add_ok"], "add_held": f["n_add_held"], "removes": f["n_remove"],
                     "ok_ids": f["ok_add_ids"]}
                    for f in other_turns if f["n_add"] or f["n_remove"]],
                "turns_with_miss_then_qgd": sum(1 for f in L if f["miss_then_qgd"]),
            }
        out["llm_log"]["per_language"] = agg

    # sweep + currency
    pairs_all, cur = [], []
    for t in turns_rec:
        txt = t.get("reply_text") or ""
        for pr in sweep_reply(txt, catalog):
            f = turn_index_of_prompt.get((f"th-{t['participantId']}", t["prompt"]))
            pr.update({"participantId": t["participantId"], "language": t["language"], "trial": t.get("trial"),
                       "turn_index": t["turn_index"],
                       "miss_then_qgd": (f["miss_then_qgd"] if f else None)})
            pairs_all.append(pr)
        if _CURRENCY.search(txt):
            cur.append({"participantId": t["participantId"], "language": t["language"], "trial": t.get("trial"),
                        "turn_index": t["turn_index"], "symbols": sorted(set(m.group(0) for m in _CURRENCY.finditer(txt)))})
    wrong = [p for p in pairs_all if not p["correct"]]
    out["sweep"] = {
        "pairs": len(pairs_all), "wrong": len(wrong),
        "by_language": {l: {"pairs": sum(1 for p in pairs_all if p["language"] == l),
                            "wrong": sum(1 for p in wrong if p["language"] == l)} for l in langs},
        "wrong_tag_miss_then_qgd": dict(Counter(str(p["miss_then_qgd"]) for p in wrong)),
        "wrong_pairs": wrong,
        "replies_with_wrong": len({(p["participantId"], p["turn_index"]) for p in wrong}),
    }
    out["currency_non_usd"] = {"replies": len(cur), "detail": cur}
    return out


def render(o: dict) -> str:
    L = []
    L.append(f"participants {o['participants']}  turn records {o['turn_records']}  target {o['target']}")
    for lang, d in o["per_language"].items():
        L.append(f"[{lang}] n={d['n']} M1(cart|order)={d['M1_cart_or_order_fmt']} cart={d['cart_only_view']} "
                 f"order={d['order_view']} both={d['both']} M2={d['M2_other_product_participants']} "
                 f"M3={d['M3_qty_gt1']} turns={d['turns']} dead={d['dead_turns']} noreply={d['no_reply_turns']} "
                 f"timedout={d['timed_out']}")
    L.append(f"dead turns total {o['dead_turns_total']}")
    la = o["latency"]
    L.append(f"latency all: {la['all']}")
    for k, v in la["by_turn_index"].items():
        L.append(f"  idx {k}: {v}")
    for k, v in la["by_turn_index_language"].items():
        L.append(f"  idx/lang {k}: {v}")
    if "demo_log" in o:
        L.append("demo log: " + json.dumps(o["demo_log"], ensure_ascii=False))
    if "llm_log" in o:
        ll = dict(o["llm_log"])
        pl = ll.pop("per_language", {})
        L.append("llm log: " + json.dumps(ll))
        for lang, d in pl.items():
            L.append(f"[{lang}] " + json.dumps(d, ensure_ascii=False))
    s = dict(o["sweep"])
    wp = s.pop("wrong_pairs")
    L.append("sweep: " + json.dumps(s, ensure_ascii=False))
    for p in wp:
        L.append("  WRONG " + json.dumps(p, ensure_ascii=False))
    L.append(f"non-USD currency replies: {o['currency_non_usd']['replies']}")
    for c in o["currency_non_usd"]["detail"]:
        L.append("  " + json.dumps(c, ensure_ascii=False))
    L.append("inspect M2/M3: " + json.dumps(o["inspect"], ensure_ascii=False))
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", help="probe JSONL file(s)")
    ap.add_argument("--demo-log")
    ap.add_argument("--llm-log")
    ap.add_argument("--graph", default="ws:demo")
    ap.add_argument("--redis-url", default="redis://localhost:6379")
    ap.add_argument("--target", default=TARGET_DEFAULT)
    ap.add_argument("--scripts", help="only turn records whose `script` is in this comma list (refire_probe runs)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    o = analyse(a)
    print(json.dumps(o, ensure_ascii=False, indent=1, default=str) if a.json else render(o))
    return 0


if __name__ == "__main__":
    sys.exit(main())
