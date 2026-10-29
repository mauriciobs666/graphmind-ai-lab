#!/usr/bin/env python3
"""K-078 C2 add-then-add re-fire probe (plan
`falkor-chat/docs/plans/salesperson-cart-product-id.md` §6.3 C2).

Drives two-turn English conversations through the storefront API (`/shop/api/session`
with `language`, `/shop/api/messages`), reusing `language_salience_probe.py`'s plumbing
(`run_participant`, `JsonlWriter`, `lmstudio_snapshot`) unchanged. Each trial is one
conversation per configured language, run concurrently (the probe's shape), on one
script, a fresh participant each time:

  same-category-immediate  "Add 1 Wireless Mouse Pro to my cart." → "Also add 1 Mechanical Keyboard K200."
  distinct-immediate       "Add 1 Wireless Mouse Pro to my cart." → "Also add 1 Portable SSD 1TB."
  confirm                  "Do you have the Wireless Mouse Pro?"  → "Yes, add it to my cart."
                           (impl review Pass 3: a confirmation-turn add, which neither
                           TP-007 nor the two C2 scripts contain; reported, not gated)
  browse                   "Do you have any periféricos?"  (one turn; id2 design note §6.2,
                           review m4: D8's category-miss `catalog` on a browse request;
                           reported, not gated — classify each reply by hand)
  browse-type              "Vocês têm algum mouse?"  (one turn; K-086 plan
                           `falkor-chat/docs/plans/salesperson-query-graph-data.md` §8.3 B5:
                           a product-type word, not a category; run with --languages pt-BR only)

The schedule interleaves the scripts (`--schedule`, default 12 × the two C2 scripts,
alternating), so trial `t` always runs the same script and `--resume` continues at the next
unused trial index. Every turn record carries `script` and `conversation` fields on top
of the probe's own (`participantId`, `language`, `prompt`, `turn_ms`, `reply_text`, …).

Ground truth is the graph, never reply text: a **re-fire** is a participant whose final
Wireless Mouse Pro quantity, summed over `CartItem` and `OrderLine`, is > 1. Count it with
`cart_outcome_analysis.py RUN.jsonl --scripts <name> …` (its M3 column is exactly that).

    python3 salesperson/scripts/refire_probe.py run --out c2.jsonl \\
        --languages en,pt-BR,es --schedule same-category-immediate,distinct-immediate --repeat 12
    python3 salesperson/scripts/refire_probe.py run --out c2.jsonl --resume ...   # continue a killed run

Stdlib only.
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import language_salience_probe as probe  # noqa: E402

SCRIPTS: dict[str, list[str]] = {
    "same-category-immediate": [
        "Add 1 Wireless Mouse Pro to my cart.",
        "Also add 1 Mechanical Keyboard K200.",
    ],
    "distinct-immediate": [
        "Add 1 Wireless Mouse Pro to my cart.",
        "Also add 1 Portable SSD 1TB.",
    ],
    "confirm": [
        "Do you have the Wireless Mouse Pro?",
        "Yes, add it to my cart.",
    ],
    "browse": [
        "Do you have any periféricos?",
    ],
    "browse-type": [
        "Vocês têm algum mouse?",
    ],
}
HARNESS_VERSION = "2"


class TaggingWriter:
    """Adds `script`/`conversation`/`harness` to every record before the probe's writer."""

    def __init__(self, inner: probe.JsonlWriter, **tags):
        self._inner, self._tags = inner, tags

    def write(self, record: dict) -> None:
        self._inner.write({**record, **self._tags})


def build_schedule(names: list[str], repeat: int) -> list[str]:
    for n in names:
        if n not in SCRIPTS:
            raise SystemExit(f"unknown script {n!r}; known: {', '.join(SCRIPTS)}")
    return [n for _ in range(repeat) for n in names]


def cmd_run(a) -> int:
    languages = [s.strip() for s in a.languages.split(",") if s.strip()]
    schedule = build_schedule([s.strip() for s in a.schedule.split(",") if s.strip()], a.repeat)
    if os.path.exists(a.out) and os.path.getsize(a.out) > 0 and not a.resume:
        print(f"error: {a.out} exists; pass --resume to continue it", file=sys.stderr)
        return 2
    start = probe.next_trial_index(a.out) if a.resume else 0
    run_id = uuid.uuid4().hex
    w = probe.JsonlWriter(a.out)
    try:
        w.write({"type": "run_start", "run_id": run_id, "at": probe.now_iso(), "harness": "refire_probe",
                 "harness_version": HARNESS_VERSION, "base_url": a.base_url, "languages": languages,
                 "schedule": schedule, "scripts": {n: SCRIPTS[n] for n in set(schedule)},
                 "first_trial": start, "resume": bool(a.resume),
                 "lmstudio_pre": probe.lmstudio_snapshot(a.lmstudio_url)})
        for trial in range(start, len(schedule)):
            script = schedule[trial]
            started = probe.now_iso()
            tw = TaggingWriter(w, script=script, conversation=trial, harness="refire_probe")
            with ThreadPoolExecutor(max_workers=len(languages)) as pool:
                futs = [pool.submit(probe.run_participant, a.base_url, run_id, trial, lang,
                                    SCRIPTS[script], tw, a.turn_wait_ceiling) for lang in languages]
                for f in futs:
                    f.result()
            w.write({"type": "trial_end", "run_id": run_id, "trial": trial, "script": script,
                     "started_at": started, "ended_at": probe.now_iso()})
            print(f"[refire] trial {trial} ({script}) done ({trial + 1}/{len(schedule)})", file=sys.stderr, flush=True)
        w.write({"type": "run_end", "run_id": run_id, "at": probe.now_iso(),
                 "lmstudio_post": probe.lmstudio_snapshot(a.lmstudio_url)})
    finally:
        w.close()
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--resume", action="store_true")
    r.add_argument("--languages", default="en,pt-BR,es")
    r.add_argument("--schedule", default="same-category-immediate,distinct-immediate")
    r.add_argument("--repeat", type=int, default=12)
    r.add_argument("--base-url", default="http://127.0.0.1:8000")
    r.add_argument("--lmstudio-url", default="http://localhost:1234")
    r.add_argument("--turn-wait-ceiling", type=float, default=probe.DEFAULT_TURN_WAIT_CEILING_S)
    a = ap.parse_args(argv)
    return cmd_run(a)


if __name__ == "__main__":
    sys.exit(main())
