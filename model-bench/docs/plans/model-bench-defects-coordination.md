# model-bench defects — Coordination

> **Status:** archived · **Owner:** `teco` · **Tracks:** — (follow-on to typed-decision-models; backlog `model-bench/docs/BACKLOG.md` "Defects the live spec recorded")

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
| U1 five defect fixes + tests (leaves DESIGN/FORMATS/README/manual alone) | `tdd-engineer` | `ab317a6b89a52b0a1` | accepted | `model-bench/modelbench/{cli,runner}.py`, `tests/` | `analyst` `af062019452306049` → approve with suggestions (Pass 3); R2 "newest = last file name" documented in spec + BACKLOG, not fixed | — |
| U2 live-spec update (DESIGN §2.x), README/AGENTS pointers, BACKLOG removals | `coder` | `ab2cbc09ab5bcf9f9` | accepted | `model-bench/docs/DESIGN.md`, `README.md`, `BACKLOG.md` | `analyst` impl check → approve with suggestions (fixed) | — |
| U3 manual update | `tico` | `a7a7eec6d49d79178` | accepted | `model-bench/docs/manuals/small-model-benchmarking.md` | `analyst` Pass 3 approve with suggestions (fixed) + `qa-engineer` re-walk PASS | — |

## Log

- 2026-11-20 Opened. Sequenced behind the item-1 hand-off re-runs (queued script reads `model-bench/modelbench/`; code edits
  mid-run would change what those runs execute). Close: `HISTORY.md` entry last; archive this doc and the checks.
- 2026-11-20 Closed: U1 gated (3 passes), U2 spec updated and impact-checked (approve with suggestions, findings fixed), U3 manual updated, analyst Pass 3 + QA re-walk. HISTORY entry written; this document and the two defects reviews archived.
