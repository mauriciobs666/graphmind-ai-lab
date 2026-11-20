# model-bench defects — Coordination

> **Status:** active · **Owner:** `teco` · **Tracks:** — (follow-on to typed-decision-models; backlog `model-bench/docs/BACKLOG.md` "Defects the live spec recorded")

Goal: fix the five defects the live spec recorded (`model-bench/docs/DESIGN.md` §2.2, §2.8, §2.9;
audit `model-bench/docs/reviews/backlog-compaction.md`). Definition of done: each fixed with a test that
fails without the fix, suite green, live spec and manual updated at the feature's close, delivered
items out of `BACKLOG.md` and into `docs/HISTORY.md`.

## Stakeholder decisions (2026-11-20)

- D-ref: remove `--reference` from `run` (argparse error, exit 2); it stays on `rank`.
- D-cmp: a plain `compare` with no `--models` refuses (exit 2) unless exactly two arms exist; the arms table
  may still list all arms.
- D-strict: `validate --strict` refuses cleanly (exit 2, one-line "not implemented", no traceback).
- No decision needed: the exit-5 message stops naming `results/transcripts/<runId>.jsonl`; a start-of-run LM Studio timeout
  (`residency()`/`catalog()`) becomes a `RunRefused` exit 3 like the probes.

## Ledger

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| U1 five defect fixes + tests (leaves DESIGN/FORMATS/README/manual alone) | `tdd-engineer` | `ab317a6b89a52b0a1` | delivered 2026-11-20 (124k / 33); suite 3661 passed, re-run by teco; defect 3 refuses only for >2 arms (0/1 arms stay a report, §3.6a) — pending user ruling | `model-bench/modelbench/{cli,runner}.py`, `tests/` | `analyst` → — | — |
| U2 live-spec update (DESIGN §2.x), README/AGENTS pointers, BACKLOG removals | `coder` | — | queued (after U1 gate) | `model-bench/docs/DESIGN.md`, `README.md`, `BACKLOG.md` | `analyst` impl check vs shipped diff → — | — |
| U3 manual update | `tico` | — | queued (after U2) | `model-bench/docs/manuals/small-model-benchmarking.md` | `analyst` + `qa-engineer` → — | — |

## Log

- 2026-11-20 Opened. Sequenced behind the item-1 hand-off re-runs (queued script reads `model-bench/modelbench/`; code edits
  mid-run would change what those runs execute). Close: `HISTORY.md` entry last; archive this doc and the checks.
