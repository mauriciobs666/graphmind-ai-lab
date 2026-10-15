# Extract job transient-error retry — Coordination

> **Status:** active · **Owner:** `teco` · **Tracks:** K-067

Goal: close `falkor-chat/docs/BACKLOG.md` **K-067**. Today one transient LM Studio failure during a
chunk's extract call (HTTP 400 `{"error":"terminated"}`, then 500, then recovery within seconds) marks the
whole `Document` as `failed` for good. The cause is `background.py::_safe_extract`, which reports
`success=False` once and does not retry, combined with `repository.py::report_document_job_done`, which
treats the first failure as terminal by design (K-051).
**Definition of done:**
- a transient provider error is retried with bounded backoff;
- a genuinely bad chunk still fails fast;
- an exhausted retry budget still ends in `failed`;
- a retried extract is proven idempotent on the chunk and document state;
- both review gates are passed, plus a live proof run;
- the docs are updated.

Baseline: `704f590c`.
CPG: `cpg_falkorchat` was built `2026-10-15T00:39:12Z`. Its `source-origin` is `falkor-chat/server`.
Its `sourceTree` `9e41122e…` equals `HEAD:falkor-chat/server`, with no changes since and a clean
worktree, so it is **fresh**.

## Documentation impact

- `falkor-chat/docs/BACKLOG.md`: remove K-067 at close. Its record moves to `HISTORY.md`.
- `falkor-chat/docs/HISTORY.md`: add one entry for the delivered change.
- `falkor-chat/docs/SERVER.md`: update if it describes the background-job or document-status
  lifecycle. The implementer checks and reports.
- `falkor-chat/AGENTS.md`: update only if a new runtime knob/env var or invariant lands.
- Docstrings in `background.py`/`repository.py`, which own the K-051 contract text.

## Ledger

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| U1 | `architect` | `a3878c5f291cd084b` | accepted | `falkor-chat/docs/plans/extract-transient-retry.md` v1→v2 (widens scope to chunk embed, plan §decision 4: **stakeholder approved 2026-10-15**) | `analyst` (U2) → Pass 2 approve w/ suggestions | 187k+258k tok / 55+46 tools |
| U2 | `analyst` | `aac9e86c2d6a811c4` | accepted | `falkor-chat/docs/reviews/extract-transient-retry.md` Pass 1 → needs changes; Pass 2 → **approve w/ suggestions** + stopping condition (six items N-1..N-7) | — | 183k+240k tok / 52+13 tools |
| U1b | `architect` (fresh: U1 context >250k) | `ad060137934b29957` | accepted | plan v3 (§8.2): all six items accepted; N-5 condition amended (WARNING or gate INFO) | `analyst` targeted check (U2b) → approve w/ suggestions | 108k tok / 41 tools |
| U2b | `analyst` | `aac9e86c2d6a811c4` | accepted | review Pass 3 (targeted) → **approve w/ suggestions**; P3-1, P3-2 carried verbatim into U5 (no v4) | — | — |
| U3a | `tdd-engineer` | — | queued | plan step 1 (error fields, rung 3b, `retry.py`) | `analyst` (U4a) → — | — |
| U3b | `tdd-engineer` | — | queued | plan step 2 (extract wiring, gate method, live tests) | `analyst` (U4b) → — | — |
| U3c | `tdd-engineer` | — | queued | plan step 3 (embed wiring, docs) | `analyst` (U4c) → — | — |
| U5 | `qa-engineer` | — | queued | live proof per plan: proxy legs, then real-fault legs (e)–(g) **only after the stakeholder's go-ahead** (they disrupt the shared LM Studio) | — | — |

U3a–U3c follow the plan's step table, one dispatch per step (sizing rule). Each step is resumed on the same `tdd-engineer` where its context allows, gated by `analyst` on that step's diff, and committed before the next step starts.

The units run strictly in sequence. Each one depends on the artifact produced by the one before it.
