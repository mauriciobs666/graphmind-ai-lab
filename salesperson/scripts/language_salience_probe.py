#!/usr/bin/env python3
"""TP-007 literal-concurrency language-adherence probe (K-065 / K-066).

Drives QA's `docs/test-plans/salesperson-ui.md` TP-007 "literal-concurrency
variant" against a **running** storefront demo
(`falkor-chat/scripts/start_demo.sh` — never `start_server.sh`): per trial, one
participant per configured language (default `en` + `pt-BR` + `es`) joins
concurrently and runs the same 5 scripted turns `load_demo.py` uses (parity),
and every turn's assistant reply text is captured, so the language of every
reply can be classified — and hand-read — after the fact.

Promoted from the lost scratch harness of the 2026-09-18 Mitigation D pass
(`falkor-chat/docs/test-reports/salesperson-language-salience-report.md`,
"Feedback & recommendations" item 4). Zero third-party dependencies (stdlib
only), same HTTP plumbing shape as `load_demo.py`.

Kill-resilient by construction: the `run` subcommand appends **one JSON line
per turn, flushed and fsync'd, the moment that turn finishes** — a killed run
keeps every completed turn. `run --resume` continues an existing file at the
next unused trial index; `aggregate` re-derives every headline figure from the
raw file alone, re-classifying every reply from its stored text (so a
classifier fix never needs new live calls — exactly how the 2026-09-18 pass
corrected its first-cut classifier).

Record types in the output file (`"type"` field):
  run_start   — parameters, harness/classifier version, pre-run LM Studio snapshot
  turn        — one per scripted turn (see `_turn_record`)
  join_error  — a participant whose join failed (no turn records follow)
  trial_end   — written once all participants of a trial finished
  run_end     — post-run LM Studio snapshot

Usage:

    # full run (U6 shape): 3-way concurrent, 5 turns, n trials
    python3 language_salience_probe.py run --trials 60 \\
        --base-url http://127.0.0.1:8000 --out /path/run.jsonl

    # continue a killed run (appends; starts at the next trial index)
    python3 language_salience_probe.py run --trials 60 --resume --out /path/run.jsonl

    # headline metrics (+ every reply's raw text for the mandatory hand-read)
    python3 language_salience_probe.py aggregate /path/run.jsonl --dump-replies

Run the offline tests: `python3 -m unittest salesperson/scripts/test_language_salience_probe.py`
(from the repo root; no LM Studio, no server needed).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import threading
import time
import unicodedata
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

HARNESS_VERSION = "1"
CLASSIFIER_VERSION = "v2"  # v2 = the 2026-09-18 corrected classifier, see classify_language

POLL_INTERVAL_S = 2.0  # the SPA's own poll cadence (R10), same as load_demo.py
DEFAULT_TURN_WAIT_CEILING_S = 200.0  # > the 180 s agent-call timeout (§4.4), same as load_demo.py
MESSAGE_PAGE = 200  # storefront_api.MAX_MESSAGE_PAGE

# Same 5 scripted turns as load_demo.py, verbatim, for TP-007 parity. Kept as a
# literal copy (not an import) so this script stays runnable standalone; the
# offline test asserts the two lists stay identical.
SCRIPTED_TURNS = [
    "Hi there!",
    "What products do you have?",
    "Add the Wireless Mouse Pro to my cart.",
    "I'd like to place my order.",
    "What's the status of my order?",
]

DEFAULT_LANGUAGES = ["en", "pt-BR", "es"]
UNKNOWN = "unknown"

# ── language classifier (diagnostic-grade, v2) ──────────────────────────────

# English function words only — never product vocabulary, so an English product
# name ("Wireless Mouse Pro") inside a pt/es reply contributes nothing.
EN_WORDS = frozenset(
    """
    the an and or but you your yours i i'm i'd i'll we our is are was were be been
    to of for with on in at by from this that these those it it's its there here
    what which who how would could can will shall should may might have has had
    do does did please thank thanks hello hi help if not any some all my
    """.split()
)

# Portuguese-exclusive vocabulary (none of these is also a Spanish word).
PT_WORDS = frozenset(
    """
    você vocês voce não nao é são estão estou sou seu sua seus suas meu minha meus
    minhas eu um uma uns umas com em do da dos das na nas num numa pelo pela isso
    esse essa também tambem já então entao obrigado obrigada olá oi posso possa
    pode deseja gostaria informe endereço endereco carrinho produtos ajudar
    ajudá-lo ajudá-la mais muito muita agora ainda até foi fazer faça seguinte
    seguintes disponíveis disponível
    """.split()
)
# Spanish-exclusive vocabulary (none of these is also a Portuguese word).
ES_WORDS = frozenset(
    """
    usted ustedes tú el los las del al y es son están estoy soy su sus mi mis yo
    un unos unas lo le les hay muy más también ya entonces gracias hola puedo
    puede quiere quieres desea deseas ayudarte ayudarle dirección direccion
    carrito productos tienes tiene tenemos disponibles disponible ahora todavía
    hasta fue hacer haga siguiente siguientes envío cuál cómo qué
    """.split()
)
# Words BOTH languages use — the first-cut (v1) defect was letting these vote
# for Spanish. v2 ignores them entirely: they carry no pt-vs-es signal.
SHARED_IBERO_WORDS = frozenset(
    """
    por favor que para de a o se no entrega pedido está aqui aquí te me una con
    en podemos completo completa confirme confirmar será
    """.split()
)

PT_CHARS = frozenset("ãõçêôà")  # never in Spanish orthography
ES_CHARS = frozenset("ñ¿¡")  # never in Portuguese orthography
CHAR_WEIGHT = 2  # a language-exclusive diacritic outweighs one exclusive word

# pt-only and es-only morphology (suffixes), matched on whole words.
PT_SUFFIX = re.compile(r"(ção|ções|ões|ãos)$")
ES_SUFFIX = re.compile(r"(ción|ciones)$")

_WORD = re.compile(r"[a-zà-öø-ÿ']+", re.IGNORECASE)


def language_scores(text: str) -> dict[str, int]:
    """Raw evidence counts per language (see classify_language for the rules)."""
    lowered = unicodedata.normalize("NFC", text.lower())
    words = _WORD.findall(lowered)
    en = pt = es = 0
    for w in words:
        if w in SHARED_IBERO_WORDS:
            continue
        if w in EN_WORDS:
            en += 1
        if w in PT_WORDS:
            pt += 1
        if w in ES_WORDS:
            es += 1
        if PT_SUFFIX.search(w):
            pt += 1
        if ES_SUFFIX.search(w):
            es += 1
    pt += CHAR_WEIGHT * sum(1 for ch in lowered if ch in PT_CHARS)
    es += CHAR_WEIGHT * sum(1 for ch in lowered if ch in ES_CHARS)
    return {"en": en, "pt-BR": pt, "es": es}


def classify_language(text: str | None) -> str:
    """Classify one reply as ``"en"``, ``"pt-BR"``, ``"es"`` or ``"unknown"``.

    Diagnostic-grade regex heuristic (the same approach the 2026-09-18 pass and
    the `data-scientist` repro used) — **never** a substitute for hand-reading
    every reply, which TP-007's evidence standard still requires
    (`aggregate --dump-replies` prints them all).

    **What counts as wrong-language:** a reply whose dominant signal is a
    *different one* of en / pt-BR / es than the participant's configured
    language — scored as: English function words for ``en``; for ``pt-BR``
    vs ``es``, words exclusive to that language, its exclusive morphology
    (``-ção/-ões`` vs ``-ción``), and its exclusive diacritics
    (``ã õ ç ê ô à`` vs ``ñ ¿ ¡``), each diacritic weighted ``CHAR_WEIGHT``.
    The highest score wins.

    **What it deliberately does NOT count:**

    * **Shared Ibero-Romance words** (``por favor``, ``que``, ``para``,
      ``de``, ``pedido``, ``entrega``, ``está`` … — ``SHARED_IBERO_WORDS``)
      vote for *neither* pt nor es. **This is the v1→v2 fix**: the first-cut
      classifier let them vote Spanish, so "Por favor, informe o endereço de
      entrega completo para que eu possa processar seu pedido corretamente"
      — unambiguous Portuguese — was misread as ``es``.
    * **English product names / non-function words** inside a pt/es reply
      ("Wireless Mouse Pro") — only English *function* words score ``en``.
    * **Shared accents** (``á é í ó ú``) — both languages use them.
    * **No reply / empty text** → ``unknown``; and a **tie** or **no signal at
      all** → ``unknown``. ``unknown`` is never counted as wrong-language; the
      aggregate reports it separately as "unclassified" for hand review.
    """
    if not text or not text.strip():
        return UNKNOWN
    scores = language_scores(text)
    best = max(scores.values())
    if best == 0:
        return UNKNOWN
    winners = [lang for lang, s in scores.items() if s == best]
    return winners[0] if len(winners) == 1 else UNKNOWN


def is_wrong_language(expected: str, text: str | None) -> bool:
    """True only when the reply is confidently classified as another language."""
    got = classify_language(text)
    return got != UNKNOWN and got != expected


# ── statistics ───────────────────────────────────────────────────────────────


def wilson_interval(x: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for x successes in n trials (95% by default).

    Returns (low, high) as proportions in [0, 1]; (0.0, 1.0) when n == 0.
    """
    if n <= 0:
        return (0.0, 1.0)
    if x < 0 or x > n:
        raise ValueError(f"x={x} out of range for n={n}")
    p = x / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = p + z2 / (2.0 * n)
    margin = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    low = (centre - margin) / denom
    high = (centre + margin) / denom
    return (max(0.0, low), min(1.0, high))


def fmt_ci(x: int, n: int) -> str:
    if n == 0:
        return "n/a (n=0)"
    lo, hi = wilson_interval(x, n)
    return f"[{lo * 100:.1f}%, {hi * 100:.1f}%]"


def fmt_rate(x: int, n: int) -> str:
    return "n/a" if n == 0 else f"{x / n * 100:.1f}%"


# ── HTTP plumbing (stdlib only; same shape as load_demo.py) ─────────────────


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


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")


# ── environment evidence ─────────────────────────────────────────────────────


def lmstudio_snapshot(lmstudio_url: str) -> dict:
    """Which LM Studio models are loaded right now (REST `/api/v0/models`).

    Evidence about other consumers: a model loaded that the salesperson def does
    not use (it pins `mistralai/ministral-3-3b`) means something else is on the
    instance. Also lists local uvicorn processes (best-effort, from /proc) — the
    2026-09-18 confound was a second falkor-chat server on :8200.
    """
    snap: dict = {"taken_at": now_iso(), "lmstudio_url": lmstudio_url}
    status, resp, _ = http_json("GET", f"{lmstudio_url.rstrip('/')}/api/v0/models", timeout=10)
    snap["status"] = status
    if status == 200 and isinstance(resp, dict):
        models = resp.get("data") or []
        snap["loaded"] = [
            {
                k: m.get(k)
                for k in ("id", "type", "quantization", "loaded_context_length", "state")
            }
            for m in models
            if m.get("state") == "loaded"
        ]
    else:
        snap["error"] = resp
    snap["local_uvicorn_processes"] = _local_uvicorn_processes()
    return snap


def _local_uvicorn_processes() -> list[dict]:
    procs = []
    try:
        pids = [p for p in os.listdir("/proc") if p.isdigit()]
    except OSError:
        return procs
    for pid in pids:
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmd = f.read().replace(b"\0", b" ").decode("utf-8", errors="replace").strip()
        except OSError:
            continue
        if "uvicorn" in cmd and "falkorchat" in cmd:
            procs.append({"pid": int(pid), "cmdline": cmd})
    return procs


# ── JSONL writer ─────────────────────────────────────────────────────────────


class JsonlWriter:
    """Append-only, one record per line, flushed + fsync'd per record, thread-safe."""

    def __init__(self, path: str):
        self._lock = threading.Lock()
        self._f = open(path, "a", encoding="utf-8")

    def write(self, record: dict) -> None:
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            self._f.write(line + "\n")
            self._f.flush()
            os.fsync(self._f.fileno())

    def close(self) -> None:
        with self._lock:
            self._f.close()


def read_records(path: str) -> list[dict]:
    """Every parseable record; a torn final line (killed mid-write) is skipped."""
    records = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"warning: {path}:{lineno} unparseable line skipped", file=sys.stderr)
    return records


# ── one participant within one trial ────────────────────────────────────────


def _assistant_rows_since(base_url: str, token: str, since: int) -> tuple[list[dict], int, str | None]:
    """(assistant rows, new cursor, error) for this participant's own thread."""
    status, rows, _ = http_json(
        "GET", f"{base_url}/shop/api/messages?since={since}&limit={MESSAGE_PAGE}", token=token
    )
    if status != 200 or not isinstance(rows, list):
        return [], since, f"messages read status={status} body={rows}"
    cursor = since
    for r in rows:
        ts = r.get("createdAt")
        if isinstance(ts, int) and ts > cursor:
            cursor = ts
    return [r for r in rows if r.get("role") == "assistant"], cursor, None


def run_participant(
    base_url: str, run_id: str, trial: int, language: str, turns: list[str],
    writer: JsonlWriter, turn_wait_ceiling: float,
) -> None:
    name = f"LSP-{run_id[:6]}-t{trial:03d}-{language}"
    status, resp, _ = http_json(
        "POST", f"{base_url}/shop/api/session", body={"displayName": name, "language": language}
    )
    if status != 200 or not isinstance(resp, dict) or "token" not in resp:
        writer.write({
            "type": "join_error", "run_id": run_id, "trial": trial, "language": language,
            "participant": name, "status": status, "body": resp, "at": now_iso(),
        })
        return
    participant_id = resp["participantId"]
    token = f"{participant_id}.{resp['token']}"  # §5.2 bearer shape

    # Start the cursor after anything already in the thread at join time.
    _, cursor, _ = _assistant_rows_since(base_url, token, 0)

    for turn_idx, text in enumerate(turns):
        rec = {
            "type": "turn", "run_id": run_id, "trial": trial, "language": language,
            "participant": name, "participantId": participant_id,
            "turn_index": turn_idx, "prompt": text, "errors": [],
            "posted_at": now_iso(),
        }
        t0 = time.monotonic()
        pstatus, presp, pms = http_json(
            "POST", f"{base_url}/shop/api/messages", token=token, body={"text": text}
        )
        rec["post_status"] = pstatus
        rec["post_ms"] = round(pms, 1)
        if pstatus not in (200, 409):
            rec["errors"].append(f"post_message status={pstatus} body={presp}")

        last_state = None
        timed_out = False
        if pstatus in (200, 409):  # 409: a turn is in flight — wait it out, never re-post
            deadline = t0 + turn_wait_ceiling
            while True:
                if time.monotonic() >= deadline:
                    timed_out = True
                    break
                time.sleep(POLL_INTERVAL_S)
                sstatus, sresp, _ = http_json("GET", f"{base_url}/shop/api/state", token=token)
                if sstatus != 200 or not isinstance(sresp, dict):
                    rec["errors"].append(f"state poll status={sstatus} body={sresp}")
                    break
                last_state = sresp
                if (sresp.get("turn") or {}).get("state") == "idle":
                    break
        turn = (last_state or {}).get("turn") or {}
        rec["idle_at"] = now_iso() if turn.get("state") == "idle" else None
        rec["turn_ms"] = round((time.monotonic() - t0) * 1000.0, 1)
        rec["timed_out"] = timed_out
        rec["last_turn"] = turn.get("lastTurn")
        rec["dead_turn"] = turn.get("lastTurn") == "failed"

        rows, cursor, err = _assistant_rows_since(base_url, token, cursor)
        if err:
            rec["errors"].append(err)
        rec["reply_count"] = len(rows)
        rec["reply_msg_ids"] = [r.get("msgId") for r in rows]
        # epoch ms — lets a reviewer check each reply landed inside [posted_at, idle_at]
        rec["reply_created_at"] = [r.get("createdAt") for r in rows]
        rec["reply_text"] = "\n\n".join(r.get("text") or "" for r in rows) if rows else None
        rec["classified_at_run"] = classify_language(rec["reply_text"]) if rows else None
        rec["classifier_version"] = CLASSIFIER_VERSION
        writer.write(rec)


# ── run subcommand ───────────────────────────────────────────────────────────


def next_trial_index(path: str) -> int:
    if not os.path.exists(path):
        return 0
    trials = [r.get("trial") for r in read_records(path) if isinstance(r.get("trial"), int)]
    return (max(trials) + 1) if trials else 0


def cmd_run(args) -> int:
    languages = [s.strip() for s in args.languages.split(",") if s.strip()]
    turns = SCRIPTED_TURNS[: args.turns]
    if os.path.exists(args.out) and os.path.getsize(args.out) > 0 and not args.resume:
        print(f"error: {args.out} exists; pass --resume to continue it", file=sys.stderr)
        return 2
    start = next_trial_index(args.out) if args.resume else 0
    run_id = uuid.uuid4().hex
    writer = JsonlWriter(args.out)
    try:
        writer.write({
            "type": "run_start", "run_id": run_id, "at": now_iso(),
            "harness_version": HARNESS_VERSION, "classifier_version": CLASSIFIER_VERSION,
            "base_url": args.base_url, "trials_requested": args.trials,
            "first_trial": start, "languages": languages, "turns": turns,
            "turn_wait_ceiling_s": args.turn_wait_ceiling, "resume": bool(args.resume),
            "lmstudio_pre": lmstudio_snapshot(args.lmstudio_url),
        })
        for trial in range(start, args.trials):
            started = now_iso()
            with ThreadPoolExecutor(max_workers=len(languages)) as pool:
                futures = [
                    pool.submit(run_participant, args.base_url, run_id, trial, lang, turns,
                                writer, args.turn_wait_ceiling)
                    for lang in languages
                ]
                for f in futures:
                    f.result()
            writer.write({"type": "trial_end", "run_id": run_id, "trial": trial,
                          "started_at": started, "ended_at": now_iso()})
            print(f"[probe] trial {trial} done ({trial + 1}/{args.trials})", file=sys.stderr)
        writer.write({"type": "run_end", "run_id": run_id, "at": now_iso(),
                      "lmstudio_post": lmstudio_snapshot(args.lmstudio_url)})
    finally:
        writer.close()
    return 0


# ── aggregate subcommand ─────────────────────────────────────────────────────


def aggregate(records: list[dict], include_partial: bool = False) -> dict:
    """Every headline figure of the 2026-09-18 report, from raw records only.

    A trial counts only once its `trial_end` record exists (a killed run's
    in-flight trial would bias the trial-level metric), unless include_partial.
    Classification is recomputed from `reply_text` with the current classifier.
    """
    complete = {r["trial"] for r in records if r.get("type") == "trial_end"}
    all_turns = [r for r in records if r.get("type") == "turn"]
    turns = [t for t in all_turns if include_partial or t["trial"] in complete]
    excluded = len(all_turns) - len(turns)
    join_errors = [r for r in records if r.get("type") == "join_error"
                   and (include_partial or r["trial"] in complete)]

    languages: list[str] = []
    for t in turns:
        if t["language"] not in languages:
            languages.append(t["language"])

    per_lang: dict = {}
    per_trial: dict = {}
    for lang in languages:
        lt = [t for t in turns if t["language"] == lang]
        replied = [t for t in lt if t.get("reply_text")]
        wrong = [t for t in replied if is_wrong_language(lang, t["reply_text"])]
        unclassified = [t for t in replied if classify_language(t["reply_text"]) == UNKNOWN]
        trials_with_reply = sorted({t["trial"] for t in replied})
        trials_affected = sorted({t["trial"] for t in wrong})
        per_lang[lang] = {
            "turns": len(lt),
            "replied": len(replied),
            "adherent": len(replied) - len(wrong) - len(unclassified),
            "wrong": len(wrong),
            "unclassified": len(unclassified),
            "trials": len({t["trial"] for t in lt}),
            "trials_with_reply": len(trials_with_reply),
            "trials_affected": len(trials_affected),
            "affected_trial_ids": trials_affected,
            "no_reply": sum(1 for t in lt if not t.get("reply_text")),
            "dead_but_replied": sum(1 for t in replied if t.get("dead_turn")),
            "dead_total": sum(1 for t in lt if t.get("dead_turn")),
            "no_reply_not_dead": sum(1 for t in lt if not t.get("reply_text") and not t.get("dead_turn")),
            "timed_out": sum(1 for t in lt if t.get("timed_out")),
            "post_errors": sum(1 for t in lt if t.get("post_status") not in (200, 409)),
        }
        for t in lt:
            cell = per_trial.setdefault(t["trial"], {}).setdefault(
                lang, {"adherent": 0, "replied": 0, "dead": 0, "wrong": 0})
            if t.get("reply_text"):
                cell["replied"] += 1
                if is_wrong_language(lang, t["reply_text"]):
                    cell["wrong"] += 1
                elif classify_language(t["reply_text"]) != UNKNOWN:
                    cell["adherent"] += 1
            if t.get("dead_turn"):
                cell["dead"] += 1

    totals = {
        "turns": len(turns),
        "no_reply": sum(v["no_reply"] for v in per_lang.values()),
        "dead_but_replied": sum(v["dead_but_replied"] for v in per_lang.values()),
        "dead_total": sum(v["dead_total"] for v in per_lang.values()),
        "no_reply_not_dead": sum(v["no_reply_not_dead"] for v in per_lang.values()),
    }
    return {
        "trials_complete": len(complete),
        "turn_records_excluded_partial": excluded,
        "join_errors": len(join_errors),
        "languages": languages,
        "per_language": per_lang,
        "per_trial": per_trial,
        "totals": totals,
    }


def render(agg: dict, records: list[dict], dump_replies: bool, include_partial: bool) -> str:
    out = []
    starts = [r for r in records if r.get("type") == "run_start"]
    ends = [r for r in records if r.get("type") == "run_end"]
    out.append(f"Run segments: {len(starts)} run_start, {len(ends)} run_end; "
               f"complete trials: {agg['trials_complete']}; "
               f"turn records excluded (partial trials): {agg['turn_records_excluded_partial']}; "
               f"join errors: {agg['join_errors']}; classifier: {CLASSIFIER_VERSION}")
    for s in starts:
        loaded = [m.get("id") for m in (s.get("lmstudio_pre") or {}).get("loaded") or []]
        out.append(f"  pre-run  LM Studio loaded ({s.get('at')}): {loaded}")
    for e in ends:
        loaded = [m.get("id") for m in (e.get("lmstudio_post") or {}).get("loaded") or []]
        out.append(f"  post-run LM Studio loaded ({e.get('at')}): {loaded}")
    out.append("")
    out.append("Turn-level adherence (of replied turns; dead/no-reply turns excluded):")
    out.append("| Language | Replied | Adherent | Wrong | Unclassified | Adherence | Wilson 95% CI | Wrong rate | Wilson 95% CI |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for lang in agg["languages"]:
        v = agg["per_language"][lang]
        out.append(f"| `{lang}` | {v['replied']} | {v['adherent']} | {v['wrong']} | {v['unclassified']} | "
                   f"{fmt_rate(v['adherent'], v['replied'])} | {fmt_ci(v['adherent'], v['replied'])} | "
                   f"{fmt_rate(v['wrong'], v['replied'])} | {fmt_ci(v['wrong'], v['replied'])} |")
    out.append("")
    out.append("Trial-level (trial affected = >=1 wrong-language replied turn), of trials with >=1 reply:")
    out.append("| Language | Trials w/ >=1 reply | Affected | Rate | Wilson 95% CI | Affected trial ids |")
    out.append("|---|---|---|---|---|---|")
    for lang in agg["languages"]:
        v = agg["per_language"][lang]
        out.append(f"| `{lang}` | {v['trials_with_reply']} / {v['trials']} | {v['trials_affected']} | "
                   f"{fmt_rate(v['trials_affected'], v['trials_with_reply'])} | "
                   f"{fmt_ci(v['trials_affected'], v['trials_with_reply'])} | {v['affected_trial_ids']} |")
    out.append("")
    out.append("Dead-turn accounting (dead = turn.lastTurn == 'failed'):")
    out.append("| Language | Turns | No-reply | Dead-but-replied | Dead total | No-reply, not dead | Timed out | Post errors |")
    out.append("|---|---|---|---|---|---|---|---|")
    for lang in agg["languages"]:
        v = agg["per_language"][lang]
        out.append(f"| `{lang}` | {v['turns']} | {v['no_reply']} | {v['dead_but_replied']} | {v['dead_total']} | "
                   f"{v['no_reply_not_dead']} | {v['timed_out']} | {v['post_errors']} |")
    t = agg["totals"]
    out.append(f"| **all** | {t['turns']} | {t['no_reply']} ({fmt_rate(t['no_reply'], t['turns'])}) | "
               f"{t['dead_but_replied']} | {t['dead_total']} | {t['no_reply_not_dead']} | | |")
    out.append("")
    out.append("Per trial — adherent/replied (dead=N); `!W` marks wrong-language turns:")
    out.append("| Trial | " + " | ".join(f"`{l}`" for l in agg["languages"]) + " |")
    out.append("|---|" + "---|" * len(agg["languages"]))
    for trial in sorted(agg["per_trial"]):
        cells = []
        for lang in agg["languages"]:
            c = agg["per_trial"][trial].get(lang)
            if c is None:
                cells.append("—")
            else:
                w = f" !W{c['wrong']}" if c["wrong"] else ""
                cells.append(f"{c['adherent']}/{c['replied']} (dead={c['dead']}){w}")
        out.append(f"| {trial} | " + " | ".join(cells) + " |")
    if dump_replies:
        complete = {r["trial"] for r in records if r.get("type") == "trial_end"}
        out.append("")
        out.append("Every reply, for the mandatory hand-read (trial/lang/turn -> classified):")
        for r in records:
            if r.get("type") != "turn" or not r.get("reply_text"):
                continue
            if not include_partial and r["trial"] not in complete:
                continue
            got = classify_language(r["reply_text"])
            flag = "WRONG " if is_wrong_language(r["language"], r["reply_text"]) else ""
            out.append(f"--- t{r['trial']} {r['language']} turn{r['turn_index']} -> {flag}{got} "
                       f"{language_scores(r['reply_text'])}")
            out.append(r["reply_text"])
    return "\n".join(out)


def cmd_aggregate(args) -> int:
    records = read_records(args.path)
    agg = aggregate(records, include_partial=args.include_partial)
    if args.json:
        print(json.dumps(agg, indent=2, ensure_ascii=False))
    else:
        print(render(agg, records, args.dump_replies, args.include_partial))
    return 0


# ── CLI ───────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="drive the live demo, appending one JSON line per turn")
    p_run.add_argument("--trials", type=int, default=25, help="total trial count (with --resume: target total)")
    p_run.add_argument("--languages", default=",".join(DEFAULT_LANGUAGES),
                       help="comma-separated participant languages per trial (one participant each, concurrent)")
    p_run.add_argument("--turns", type=int, default=len(SCRIPTED_TURNS), choices=range(1, len(SCRIPTED_TURNS) + 1),
                       metavar=f"1..{len(SCRIPTED_TURNS)}")
    p_run.add_argument("--base-url", default="http://127.0.0.1:8000")
    p_run.add_argument("--lmstudio-url", default="http://localhost:1234")
    p_run.add_argument("--turn-wait-ceiling", type=float, default=DEFAULT_TURN_WAIT_CEILING_S)
    p_run.add_argument("--out", required=True, help="JSONL output path (appended to with --resume)")
    p_run.add_argument("--resume", action="store_true", help="append to --out, starting at the next trial index")

    p_agg = sub.add_parser("aggregate", help="print the report's metrics from a JSONL file")
    p_agg.add_argument("path")
    p_agg.add_argument("--include-partial", action="store_true",
                       help="also count turns of trials with no trial_end record")
    p_agg.add_argument("--dump-replies", action="store_true", help="print every reply's raw text")
    p_agg.add_argument("--json", action="store_true", help="machine-readable aggregate")

    args = parser.parse_args(argv)
    return cmd_run(args) if args.cmd == "run" else cmd_aggregate(args)


if __name__ == "__main__":
    sys.exit(main())
