# Document ingestion — update & delete — Coordination

> **Status:** active · **Owner:** `teco` · **Tracks:** —

Sequencing/gating log for implementing `falkor-chat/docs/plans/document-ingestion2.md` (Status:
active, Owner: `architect` — every Cypher/index shape in it is live-verified by `graph-dba` across
two passes, §0/§7 of that plan; no open design blockers remain). Read that plan directly for full
design rationale — this document cites it, never restates it.

## Sequencing

Five stages per the plan's §4, dispatched **one at a time** (not parallel) even though A/B/C are
mutually independent per the plan's own dependency note — `repository.py`, `services.py`,
`mcp.py`, `api.py`, `schemas.py`, and `scripts/bootstrap_schema.sh` are touched by more than one
stage, so same-file serialization applies regardless. Each stage gets a diff-scoped `analyst`
review gate immediately after its delivery, before the next stage is dispatched. No separate
plan-gate review was dispatched before implementation: the plan's own two `graph-dba`
live-verification passes (§0) already exercised every Cypher/index shape against a real instance,
which is a stronger form of design verification than a static plan read would add on top for the
graph-facing half of this work; the `analyst` diff gates below still cover blast-radius/edge-case
review for the application-layer half.

**Environment:** FalkorDB confirmed up (`redis-cli -p 6379 ping` → `PONG`) before dispatch.
**CPG:** none consulted — the plan's own §2 scopes this out ("new-code design... rather than an
impact-analysis question a call graph would answer faster"), and that reasoning holds unchanged
for the coordination.

## Ledger

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| Stage A | `coder` | `a9d89d22889116f71` | accepted | commit `4a6186b` | `analyst` (`a6e97e7a616f4e17d`) → approve (`reviews/document-ingestion2-impl.md` Pass 1, commit `609f5e2`) | 292883 tok / 169 tools |
| Stage B | `coder` | `a9f6e53aab977e853` | accepted | commit `aa1c9be` | `analyst` (`a439cd511546da526`) → approve (Pass 2, commit `f61d193`) | 284816 tok / 128 tools |
| Stage B-fix | `coder` | `aeae785dc198f920c` | delivered, commit held | not committed — see Notes | — → — | 132737 tok / 42 tools |
| Stage C | `coder` | `aac114c8aba5b13b3` | gated | commit `6443365` (+ RCA doc) | `analyst` (RCA, `ac32535b34417b58f` → root-cause; diff gate, `a4b85a78a0b012216`) → **needs changes** (`document-ingestion2-impl.md` Pass 3) | 232999 tok / 114 tools |
| Stage C-testinfra | `tdd-engineer` | `a86e9ffd7e45830fb` | accepted | commit `bedae6f` | `analyst` (`a2ddb8f16f010aa2d`) → approve (Pass 4, minor+nit only) | 181930 tok / 61 tools |
| Stage C-fix | `tdd-engineer` | — | queued (after Stage C-testinfra gate) | — | `analyst` → — | — |
| Stage D | `coder` | — | queued | — | `analyst` → — | — |
| Stage E | `qa-engineer` | — | queued | — | — | — |

## Notes

- **Stage A committed as `4a6186b`, 11 files, deliberately excluding
  `falkor-chat/server/falkorchat/storefront_api.py`,
  `falkor-chat/server/tests/test_storefront_api.py`, and `falkor-chat/docs/HISTORY.md`.** These
  three carry the concurrent, not-yet-committed `salesperson-ui` S9e work (a different, independently
  running `teco` session, `session_01Pw13NhkZVMBmuCvLBEwfXP` per its own commits) mixed with a small,
  legitimate `document-ingestion2` hunk (a `DocumentNotFoundError` import + `SERVICE_ERRORS_UNREACHABLE`
  entry in `storefront_api.py`, plus the corresponding `test_storefront_api.py` reachability-guard
  update). That other session's own S9e HISTORY.md entry (still uncommitted at Stage A's delivery
  time) states explicitly it reverted this exact hunk from its own diff and **"left [it] for the
  document-ingestion2 session to land as its own change."** `coder` had re-applied the hunk after
  noticing it was "clobbered" mid-run, not realizing the removal was deliberate — it is currently
  sitting correctly in the shared working tree (suite green with it present) but is **not yet
  committed by either side**. Committing `storefront_api.py`/`test_storefront_api.py` now would sweep
  the other session's large, ungated, in-flight diff into this commit under this message — not done.
  **Follow-up, not blocking Stage B:** once the salesperson-ui S9e work commits (its own analyst gate
  was already dispatched per that coordination's ledger), pick up the remaining
  `storefront_api.py`/`test_storefront_api.py` diff (by then just the `DocumentNotFoundError`
  registration) and commit it as a small standalone follow-up to this coordination. No `SendMessage`
  route to that session was available (no agent-name handle, only a session id seen in commit
  trailers) — coordinating asynchronously via git history/HISTORY.md, the pattern that session itself
  already used.
- **Follow-up resolved, then re-blocked by a third wave (S10).** The salesperson-ui S9e work
  committed as `dcec3f2` (reviewed/approved, Pass 27-28) and that coordination closed/split
  (`docs/plans/salesperson-ui2-coordination.md` opened for post-S9 work) — `storefront_api.py`/
  `HISTORY.md` were briefly clean at `HEAD` again. Stage B independently hit the same shape of issue
  (a second new `ServiceError` subclass, `DocumentUpdateNotFoundError`, also unclassified) — `coder`
  was told not to touch that file and correctly didn't. A follow-up unit (Stage B-fix) landed both
  Stage A's and Stage B's classification entries in the shared working tree, verified the true family
  size empirically (**12**, not the stale `11` left over from an earlier abandoned attempt), and
  confirmed via mutation test — but by the time it finished, a **third, unrelated, concurrent unit**
  (salesperson-ui S10, presenter-login/reset-all) had started actively rewriting
  `storefront.py`/`storefront_api.py`/`config.py`/`test_storefront_api.py` in this same shared tree
  (`git diff HEAD --stat` shows storefront_api.py alone at 247 changed lines, dwarfing the ~4-line
  classification fix). **The classification fix is verified correct and sitting in the working tree,
  but withheld from every commit so far** — committing either file now would sweep S10's large,
  ungated, uncommitted work in under a document-ingestion2 message. Stages A and B are both committed
  and gated without it; picking it up is a standing, low-priority follow-up whenever
  `storefront_api.py`/`test_storefront_api.py` next go quiet. Not blocking Stage C/D/E — nothing in
  those stages depends on this fix landing.
- **Stage C delivered but held uncommitted — a real, reproducible suite failure, not concurrent-
  session noise.** `coder` reported "723 passed" but that run did not exercise the combination that
  fails: `pytest -q tests/test_repository.py tests/test_services.py tests/test_api.py tests/test_mcp.py
  tests/test_graphrag.py` reproducibly fails 4 tests (identical set, two consecutive runs); the full
  `pytest -q` fails 6 (adds `test_responder.py::test_ac5_…`, `test_tools.py::test_graphrag_retrieve_…`).
  Every failing test **passes in isolation** — pattern is "a `documentCurrent`-true chunk not found by
  `search_chunks`'s ANN scan," consistent with this build's known tiny-corpus ANN-recall degradation
  (`falkor-chat/docs/SERVER.md` §1.7, and `coder`'s own kaizen finding from this same stage). **Timeline
  evidence points at Stage C causing this, not pre-existing flakiness**: Stage A's implementer and
  reviewer both ran the **full** suite clean (2680/2680 passed, zero failures) when `test_graphrag.py`
  already existed but the `documentCurrent` filter did not — so whatever is happening started with
  Stage C's change. Verified via a throwaway `git worktree` at `aa1c9be` (removed after) that this
  monorepo's PEP-660 editable install resolves `import falkorchat` to the **current working tree**
  regardless of `PYTHONPATH` override, not the worktree's checked-out commit — so that specific
  diagnostic attempt was inconclusive and abandoned rather than trusted; noted here so a future
  attempt doesn't repeat it. Dispatched `analyst` for RCA rather than guessing at a fix.
- **RCA returned and independently spot-checked before acting on it** (`docs/reviews/document-ingestion2-rca.md`).
  Root cause: pre-existing FalkorDB HNSW recall degradation under cumulative node churn on
  `ws:test`'s session-scoped (never mid-session-rebuilt), dim-4 `Chunk.embedding` vector index —
  **not** caused by Stage C's `WHERE seed.documentCurrent = true` filter or the
  `SEARCH_DOCUMENTS_OVERFETCH` multiplier. Evidence: (1) a correctly-isolated `git worktree` +
  independent venv baseline at `aa1c9be` reproduces 2 of the 4 current failures byte-for-byte with
  zero Stage-C code present; (2) `GRAPH.PROFILE` shows `Filter` strictly after `ProcedureCall` in
  both plans — refutes a pre-filter-pushdown hypothesis, confirms the plan's own §3.3 assumption;
  (3) a clean-room churn probe shows ANN recall for small `k` degrading monotonically with
  cumulative create/delete cycles on the same persistent index, independent of any
  `documentCurrent` code. I independently re-ran the exact repro command myself
  (`pytest -q tests/test_repository.py tests/test_services.py tests/test_api.py tests/test_mcp.py
  tests/test_graphrag.py`) before accepting the finding and got the identical result reported in
  the RCA (`4 failed, 739 passed`, same 4 test names) — corroborates the headline number
  independently rather than accepting it on the RCA's word alone. Confirmed `HEAD` unmoved on every
  Stage C target file since Stage B (`git diff aa1c9be HEAD --stat` empty) before committing, so no
  concurrent-session sweep risk. **Stage C committed as `6443365`** (code unchanged from delivery,
  RCA doc included) — the diff itself needed no fix. Opened a new unit, **Stage C-testinfra**, to
  fix the actual defect (test-infra: `conftest.py`'s `_schema` fixture is session-scoped, so churn
  accumulates unboundedly across the whole suite) — routed to `tdd-engineer` per the RCA's §5
  suggested fix (function/module-scoped vector-index rebuild boundary for ANN-sensitive test
  modules, reproduction test adapted from the RCA's Appendix B churn script) — judged and not
  merely forwarded: this is a bug fix with a clear, RCA-stated behavior contract, exactly
  `tdd-engineer`'s routing case. Sequenced in parallel with Stage C's diff-gate review (disjoint
  files: `conftest.py`/new fixture vs. the already-committed `repository.py`/`services.py`).
  Judgment call on RCA's own open question ("fix now or file as a ticket"): fixing now, ahead of
  Stage D, because Stage D's own tests will add more embedding-touching churn and could otherwise
  reproduce this exact same false-attribution scare against Stage D's diff instead.
  Also flagged, not yet routed: RCA's suggestion to add the churn-vs-recall curve to
  `claude/graph-dba/falkordb-quirks.md` — a `graph-dba` documentation follow-up, non-blocking.
- **Stage C diff-gate (Pass 3) returned `needs changes` — two real blockers, both independently
  spot-checked before acting.** (1) Every `Document`/`Chunk` ingested before Stage A landed
  (`4a6186b`, 2026-09-11) has neither `currentVersion` nor `documentCurrent` set at all — the new
  `WHERE seed.documentCurrent = true` filter treats the missing property as falsy, so
  `search_documents`/`GET /documents/search` now silently return **zero results forever** for any
  pre-existing workspace. I re-ran the reviewer's exact live probe myself
  (`MATCH (d:Document) RETURN count(d), count(d.currentVersion)` against `ws:acme`) and got the
  identical `29, 0` — confirms this is real, not a review artifact. (2) `services.hybrid_search`
  (the chat-grounding retrieval path behind `GraphragRetrieveTool`) calls the same filtered
  `repository.search_chunks` but never got the `SEARCH_DOCUMENTS_OVERFETCH` treatment — confirmed
  by grep and independently by a `cpg_falkorchat` call-graph query for `search_chunks` callers
  (`services.py:1096` unfixed, `:1300` fixed) — same under-fill defect Stage C exists to close,
  left open on the agent's own retrieval path. Full findings + the reviewer's two named fix options
  for Blocker 1 (backfill migration matching this codebase's own `backfill_thread_ids.sh`
  precedent, vs. a null-tolerant filter as a silent semantics change) are in
  `document-ingestion2-impl.md` Pass 3 — read there, not restated. **Decision (mine, not a passthrough
  of the reviewer's suggestion):** backfill, not a null-tolerant filter — the plan's own §3.3
  wording specifies plain equality, this codebase already has a precedent script shape for exactly
  this kind of property backfill, and a silent semantics change would make "current" mean "created
  before or after this feature" in a way nobody decided on purpose. Opened **Stage C-fix**
  (`tdd-engineer`) for: the backfill (+ a regression test with an unset-property fixture, the axis
  Pass 3 found untested), the `hybrid_search` over-fetch parity fix + its own hard-negative test,
  and the docstring nit. **Queued behind Stage C-testinfra**, not parallel — real file-overlap risk
  (both likely touch `test_graphrag.py`), so serializing rather than risking a same-file collision
  for the sake of running two units at once.
- **Stage C-testinfra delivered** (`conftest.py`'s `rebuild_vector_indexes`/`fresh_vector_index`
  fixture applied to every real-ANN test module, plus a new churn-guard test file). Independently
  re-verified before dispatching its gate: confirmed via `git diff --stat` that
  `repository.py`/`services.py`/`storefront_api.py`/etc. are untouched, and re-ran the RCA's exact
  5-file repro command myself — `743 passed`, matching the delegate's own reported number exactly.
  Dispatched `analyst` (diff-gate, **Pass 4** in the same `-impl` review file) rather than
  committing on the delegate's self-report alone. Held uncommitted pending that verdict.
- **Stage C-testinfra gated `approve`** (Pass 4 — minor: a runtime-delta reporting discrepancy in
  the delegate's own summary, real number smaller not larger, no regression hidden; nit: a
  duplicated label-tuple literal; no blockers/majors). The reviewer independently reproduced the
  mutation-test claim itself via `git stash` rather than just judging it plausible, and a
  `cpg_falkorchat` call-graph audit surfaced two `tests/eval/` callers a flat grep missed — both
  confirmed correctly out of scope. Before committing I re-checked `git diff 6443365 HEAD --stat`
  on every target file (empty — no concurrent-session drift since the Stage C commit) and confirmed
  no stray `git stash` entry was left behind. **Committed as `bedae6f`.**
  **Stopping here for the day, per explicit user instruction** — Stage C (code + RCA + test-infra
  fix) is now fully committed and gated. **Stage C-fix (the two Pass-3 blockers — pre-existing-data
  backfill and the `hybrid_search` over-fetch parity gap) is queued but NOT dispatched.** Stage C is
  therefore not fully closed: its Pass 3 verdict is still `needs changes` until Stage C-fix lands
  and gets its own review pass. Next session should pick up by dispatching Stage C-fix
  (`tdd-engineer`) per the Notes entry above ("Decision (mine...): backfill, not a null-tolerant
  filter..."), then re-gate, then proceed to Stage D/E.
