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
| Stage C | `coder` | `aac114c8aba5b13b3` | accepted (closed via Stage C-fix) | commit `6443365` (+ RCA doc) | `analyst` (RCA, `ac32535b34417b58f` → root-cause; diff gate, `a4b85a78a0b012216`) → needs changes (Pass 3), superseded by Pass 5 approval | 232999 tok / 114 tools |
| Stage C-testinfra | `tdd-engineer` | `a86e9ffd7e45830fb` | accepted | commit `bedae6f` | `analyst` (`a2ddb8f16f010aa2d`) → approve (Pass 4, minor+nit only) | 181930 tok / 61 tools |
| Stage C-fix | `tdd-engineer` | `ae7d3fff734c8f6f9` | accepted | commits `b9c4b66` + `f64b3c4` | `analyst` (`a3018e15374301a62`) → approve with suggestions (Pass 5) | 191189 tok / 84 tools (+125848 tok / 56 tools review) |
| Stage D | `coder` | `a82458d12eeedad41` | accepted | commit `0c0fa4a` | `analyst` (`af1c21373dcf0a02d`) → approve with suggestions (Pass 6) | 409851 tok / 138 tools (+211974 tok / 54 tools review) |
| Stage D-fix | `tdd-engineer` | `a21c31b2be06c85b3` | accepted | commit `854f0b5` | `analyst` (`ab817d518c2689a3e`) → approve (Pass 7, zero new findings) | 167316 tok / 55 tools (+117663 tok / 32 tools review) |
| Stage D-fix-graph | `graph-dba` | `a89b7b2fec6d7d88c` | accepted (no code change) | `claude/graph-dba/falkordb-quirks.md` entry, 2026-09-13 | — (diagnostic consult) → clean, closes Pass 6 Finding 1 | 90604 tok / 17 tools |
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
- **Resumed 2026-09-13.** Verified environment (FalkorDB `PONG`, tree clean, `HEAD` at `bedae6f`)
  and re-confirmed both Pass 3 blockers still live in the current tree (`services.py:1096`
  `hybrid_search` still unfixed `k=k`; `services.py:1300` `search_documents` still fixed) before
  dispatching. **Stage C-fix dispatch was blocked by the auto-mode permission classifier**
  ("Modify Shared Resources") — the brief asks the delegate to run a live backfill write against
  the shared `ws:acme`/`ws:nlq-eval` FalkorDB workspaces. Escalated to the user rather than
  substituting my own judgment (this is exactly the "write the harness itself gates behind human
  approval" case) — user chose **"approve the live backfill now"**. Re-dispatched
  `tdd-engineer` (`ae7d3fff734c8f6f9`) with the same brief, now carrying explicit user approval
  for the write against those two named workspaces specifically (not a blanket approval — a third
  workspace surfacing mid-run is flagged in the brief as a stop-and-ask fork, not an extension of
  this approval).
- **Stage C-fix delivered, both blockers fixed, independently re-verified before gating.**
  `git diff --stat` matches the delegate's own file list exactly (`services.py`,
  `test_graphrag.py`, `test_services.py` + new `scripts/backfill_document_current.sh`;
  `repository.py` untouched). Full suite re-run by me independently: `2749 passed, 14 deselected`
  — identical to the delegate's reported figure. Live backfill counts re-probed by me via
  read-only `mcp__cypher__query` against both workspaces (not taken on the delegate's word):
  `ws:acme` 29/29 documents with `currentVersion`, 87/87 chunks with `documentCurrent`;
  `ws:nlq-eval` 12/12 documents, 12/12 chunks — both match the delegate's reported before/after
  table exactly. Read the actual `services.py` diff directly (not just the summary): Blocker 2's
  fix reuses `SEARCH_DOCUMENTS_OVERFETCH` at `hybrid_search`'s `chunk_hits` call with a comment
  explaining the shared reasoning, and the docstring nit is corrected — matches the report.
  Confirmed the delegate's `kaizen_team` write landed (`tdd-engineer` → new entry dated
  2026-09-13, ANN-recall-on-non-embedding-property-write finding). **Non-blocking side note:**
  my own independent full-suite re-run reproduced the documented `reference`-graph wipe hazard
  (`docs/SERVER.md` §1.7) — `./scripts/verify_workflows.sh acme` now reports both defs `MISSING`
  from `reference`. The standard remedy (`./scripts/seed_workflows.sh acme`) was blocked by the
  same auto-mode classifier as the backfill write; not re-escalating for this — it's a
  pre-existing, well-documented hazard triggered by *any* pytest run (every prior stage's runs
  did this too), not specific to Stage C-fix, and doesn't affect any of the counts/tests verified
  above. Flagging as a standing, low-priority follow-up (re-seed `reference` next time a script
  write is approved) rather than blocking this gate on it.
  Dispatching `analyst` for the Pass 5 diff-gate re-review of both blockers now.
- **Pass 5 returned approve with suggestions** (no blockers). Independently reproduced both
  mutation-test claims (file-copy + md5-verified restore) and the same-revision-fixes interaction
  check (no double-counting between the two blockers' fixes). One Minor: the docstring correction
  added by Stage C-fix (itself transcribing this same review's own Pass 3 nit) mischaracterized
  `repository.hybrid_search`'s scope join as an `OPTIONAL MATCH` — it's actually two required
  `MATCH`es; the underlying conclusion (no over-fetch currently needed there) still holds for a
  different, correct reason. I independently re-verified this against `repository.py:826-874`
  myself before accepting the finding (confirmed only the `Entity` co-occurrence expansion is
  `OPTIONAL MATCH`) — genuinely trivial, single-file docstring correction, fixed directly rather
  than routed to a specialist, using the reviewer's own suggested replacement wording (verified,
  not applied blind). Full targeted suite re-run after the fix: 288 passed.
  **Stage C-fix committed as `b9c4b66`** (code + tests + docstring fix + Pass 5 review doc)
  **+ `f64b3c4`** (the backfill script — a separate commit because `git add` on the new script
  file alone was denied by the auto-mode classifier as "Self-Modification"; escalated to the user,
  who granted it, then the add/commit succeeded). Live backfill counts re-verified via read-only
  `mcp__cypher__query` before either commit (`ws:acme` 29/29 docs, 87/87 chunks; `ws:nlq-eval`
  12/12 docs, 12/12 chunks). **Stage C is now fully closed** — Pass 3's `needs changes` is
  superseded by Pass 5's approval of the fix that closes both its blockers.
  **Non-blocking follow-up still open:** the `reference`-graph pytest-wipe hygiene item noted
  above (re-run `seed_workflows.sh acme` next time such a write is approved).
  **Next: Stage D (`coder`), then Stage E (`qa-engineer`).**
- **Stage D delivered, independently spot-checked before gating.** `git diff 5f71231 HEAD --stat`
  on every Stage D target file is empty (no concurrent-session drift). Two unrelated, currently-
  running concurrent sessions are active in the same shared tree on disjoint files
  (`salesperson/` — a `salesperson-ui2` coordination — and `model-bench/` commits) — left entirely
  untouched, not part of this dispatch or this commit. Full suite re-run by me independently:
  `2812 passed, 14 deselected`, identical to the delegate's own figure (net +63 over the Stage
  C-fix baseline of 2749). Re-ran the concurrency probe alone
  (`test_create_document_with_auto_supersede_concurrent_calls_produce_exactly_one_edge`): passes.
  Read `create_document_with_auto_supersede`'s actual Cypher directly (not just the delegate's
  summary): matches the plan's two-separate-`FOREACH`-blocks shape exactly, `doSupersede = ok AND
  candidate IS NOT NULL` (not `candidate IS NOT NULL` alone, per the plan's own actor-safety
  note), re-`MATCH`-by-`documentId` before the guarded blocks (a Cypher scoping necessity the
  delegate documented rather than silently working around). Noted deviation, judged reasonable:
  the delegate substituted a different real mutation target for the "unlabeled-endpoint
  discipline" mutation test, since `create_document_with_auto_supersede` only `CREATE`s
  `SUPERSEDES` (never `MATCH`es one) — the discipline literally doesn't apply to this new code;
  Stage B's existing `MATCH`-based `SUPERSEDES` reads are unchanged. Dispatching `analyst` for the
  Stage D diff-gate review (Pass 6) now.
- **Pass 6 returned approve with suggestions** (no blockers). Reviewer independently re-mutation-
  tested the compound guard itself (not just judged plausible) and ran `EXPLAIN` live against
  `ws:test` for `find_update_shortlist`'s actual query. Three Minor findings, none blocking:
  (1) the shipped `find_update_shortlist` Cypher includes a `currentVersion` predicate that
  wasn't in the literal shape `graph-dba` profiled in the plan's §0 pass — `EXPLAIN` still shows a
  single `Node By Index Scan`, no label scan, but `EXPLAIN` can't confirm real selectivity the way
  `GRAPH.PROFILE` with planted probe rows can — recommends a `graph-dba` follow-up; (2) the
  "skip suggested-tier scheduling when already auto-superseded" guard is present (verified by
  reading) but has no direct regression test distinguishing "guard present" from "guard silently
  removed"; (3) `scripts/test_queries.sh` — which plan §5 explicitly calls for updating with every
  new Cypher shape this stage adds — is untouched, a real plan-mandated-deliverable gap (not a
  functional defect: `EXPLAIN`/pytest already prove the shapes work, this is the standing
  live-instance regression-guard layer specifically). **Stage D committed as `0c0fa4a`**
  (code + tests + Pass 6 review doc) — none of the three findings are correctness blockers, and
  Stage D's own AC-1/AC-2 coverage is solid per both the delegate's and reviewer's independent
  verification. **Opened Stage D-fix** to close all three before Stage E, since Stage E's QA
  acceptance pass benefits from a complete `test_queries.sh` baseline and a fully-closed review.
  Not yet dispatched — next session (or continuation) picks up: (a) `graph-dba` quick consult,
  `GRAPH.PROFILE` with planted current/non-current documents sharing an LSH band, to confirm
  finding 1 for real; (b) a small `coder`/`tdd-engineer` follow-up for the missing scheduling-guard
  regression test + the `test_queries.sh` §14.10 additions (mirroring the existing §14.7/§14.8/
  §14.9 per-stage pattern) — route fresh (not resumed) given Stage D's delegate is already at
  ~410k tokens, well past the resume-vs-fresh threshold for self-contained follow-up work.
- **Stage D-fix (`tdd-engineer`, findings 2+3) delivered**, test-only diff (`test_api.py` + 1 new
  `test_queries.sh` §14.10 section), left uncommitted per brief. Independently re-verified before
  gating: read the full diff; confirmed the new `test_api.py` test's monkeypatch/args indexing
  against `_schedule_update_detection`'s actual 4-arg signature (`background.py:295`) — `args[3]`
  is `document_id`, matches the call sites in `api.py`; confirmed `§14.10`'s
  `CREATE_DOC_WITH_AUTO_SUPERSEDE` Cypher string is a verbatim copy of the shipped query
  (`repository.py:1779-1838`, diffed clause-by-clause); ran my own mutation test (not just
  re-trusting the delegate's) — removed the `autoSuperseded` guard in `api.py`, new test failed as
  expected, restored from a `cp` backup, md5-confirmed byte-identical; re-ran the full Python suite
  myself (2813 passed, 14 deselected — matches delegate's report) and `./scripts/test_queries.sh`
  myself (458/459 — matches; the one failure is the same pre-existing §14.8 `Edge By Index Scan`
  helper gap the delegate already isolated via `git stash`/`stash pop` bracketing, logged as a
  `:KaizenEntry`, confirmed present in `kaizen_team`). Dispatched `analyst` (`ab817d518c2689a3e`)
  for a diff-scoped Pass 7 gate before committing, per this coordination's per-stage gate
  discipline. **Pass 7 returned approve, zero new findings** (both mutation-tested the guard
  test itself, independently of my own earlier mutation test — same result). Committed as
  `854f0b5` (test_api.py + test_queries.sh §14.10 + review doc). **Stage D-fix is now fully
  closed** — all three Pass 6 findings resolved (Finding 1 via `graph-dba`/`19ad080`, Findings 2+3
  via this unit/`854f0b5`). Next: Stage E (`qa-engineer`).
