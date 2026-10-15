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
| U3a | `tdd-engineer` | `aedc20a61b45fb469` | accepted (resumed once after 429 kill, no files lost) | plan step 1: `transport.py`, `llm.py`, `embedding.py`, new `retry.py` + 4 test files; 173 passed after P4 fixes (teco re-ran; teco mutants on rung-1 `status`/`detail` both killed) | `analyst` (U4a, `af49f6c63460dc1b3`) → Pass 4 **approve w/ suggestions**; P4-1/P4-2/599 row sent back to U3a; P4-3 (R-6 list, plan text) → close; other P4-4 nits kept | 151k tok / 45 tools |
| U3b | `tdd-engineer` | `aedc20a61b45fb469` | accepted | plan step 2: `ingestion.py`, `repository.py` + 3 test files; 62 + 5 passed after P5 fixes (teco re-ran; teco mutant on gate `ws` killed; `extract(text[:1])` survived, pre-existing, closed by P5-2) | `analyst` (U4b, `af49f6c63460dc1b3`) → Pass 5 **approve w/ suggestions**; P5-1/P5-2 sent back to U3b; P5-3 nits kept | 230k tok cumulative / 39 tools |
| U3c | `tdd-engineer` (fresh: U3a/b context ~238k) | `aba640fb7455f820d` | accepted | plan step 3: `embedding.py`, docstrings, `QUERIES.md` §14.1a + 2 test files; 246 passed on 6 files (teco re-ran; teco mutant on `write_label` killed) | `analyst` (U4c, `af49f6c63460dc1b3`) → Pass 6 **approve w/ suggestions**; P6-1 (test comments) + P6-2 (SERVER.md `--reload` clause) closed by U3c; plan-text side of P6-1 + P4-3 → `architect` at close; `test_queries.sh` case: **no** (gate recommendation, accepted) | 132k+12k tok / 39+8 tools |
| U5 | `qa-engineer` | `a2ac7e86e7b763370` | accepted (Half A) | `falkor-chat/docs/test-plans/extract-transient-retry.md` + `falkor-chat/docs/test-reports/extract-transient-retry-report.md`: Half A 12/12 pass, no K-067 defect; D-1 (fusion RediSearch escaping) out of scope; Half B (e)–(g) specified, **not run**, awaits stakeholder at machine | teco evidence check: proxy log 58 POST / 28 fwd, per-nonce hit counts and 6 gate-INFO / 1 give-up lines match report; `ws:retryprobe` gone; `ws:agent-team` 340/12 unchanged | 224k tok / 64 tools |
| C1 | `coder` | `a7072d6998e9129d4` | accepted | `falkor-chat/docs/BACKLOG.md` (K-067 rewritten: Half B outstanding; new K-068…K-072) + `falkor-chat/docs/HISTORY.md` 2026-10-15 entry | teco integration check → verified (K-numbers unique, figures sourced, no line >700 chars) | 133k tok / 23 tools |

U3a–U3c follow the plan's step table, one dispatch per step (sizing rule). Each step is resumed on the same `tdd-engineer` where its context allows, gated by `analyst` on that step's diff, and committed before the next step starts.

**Integration (teco, 2026-10-15):** steps 1–3 committed `91b65fed` / `128941a6` / `6519db83`. The full
offline suite (`falkor-chat/server`, solo run, no concurrent agents) gave `2988 passed, 14 deselected`.
`reference` was already wiped before this coordination: a `timers-stale-key` test leftover was found and
every `verify_*` script failed. My re-seed (`bootstrap_schema` → `seed_demo` → `seed_workflows` →
`seed_catalog` → `seed_salesperson`, for `acme` and `demo`) was **blocked by the permission classifier
(Modify Shared Resources)** and escalated to the stakeholder. U5 waits on the stakeholder's go-ahead.

The units run strictly in sequence. Each one depends on the artifact produced by the one before it.

## State at session end (2026-10-15)

The code is delivered and gated, and Half A passed. **This coordination stays `active`** until Half B runs.
Remaining work is in `falkor-chat/docs/BACKLOG.md` K-067. Starting it cold needs no agent ids: the
test plan (`falkor-chat/docs/test-plans/extract-transient-retry.md` §6) is self-contained. The plan
text nits P4-3 and the P6-1 done-condition row are owned by `architect`, and the next plan revision
can fold them in. The stakeholder restores `reference` by hand (re-seed blocked by the permission
classifier).
