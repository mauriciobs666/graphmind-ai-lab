# Document ingestion — update & delete — Test Plan

> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** —

Stage E (final) of `docs/plans/document-ingestion2-coordination.md`. Acceptance pass for
`docs/plans/document-ingestion2.md`'s AC-1..AC-8, driven against the real running system —
extending past the offline suite (2813 passed, 14 deselected at dispatch, `HEAD` = `854f0b5`/
`c805dc8`), which already proves the mechanics stage-by-stage, independently diff-gated
(`docs/reviews/document-ingestion2-impl.md`, Passes 1-7). Mirrors the K-050 predecessor's own
Stage-E shape (`docs/test-plans/document-ingestion.md` / `docs/test-reports/document-ingestion-
report.md`).

## References

- Plan: `docs/plans/document-ingestion2.md` — §3 (design), §4 Stage E (this pass's own assignment),
  §5 (test-strategy AC→altitude table + "additional, non-AC-mapped" bullets).
- ML note: `docs/plans/document-ingestion2-ml.md` — §4.1's named v1 known-gap (a heavily-rewritten
  edit that also changes its title enough escapes both candidate-generation signals, becomes an
  independent document silently) and §4.1's "candidate generation must not be actor-scoped" claim
  — both exercised live below (TP-14, TP-05/TP-08).
- Review history: `docs/reviews/document-ingestion2-impl.md` (Passes 1-7, all blockers closed).
- Coordination ledger: `docs/plans/document-ingestion2-coordination.md` (Stage A `4a6186b`, B
  `aa1c9be`, C `6443365`, C-fix `b9c4b66`+`f64b3c4`, D `0c0fa4a`, D-fix `854f0b5`).
- Precedent: `docs/test-plans/document-ingestion.md` / `docs/test-reports/document-ingestion-
  report.md` (K-050 M5's own Stage-E pass — same family-slug convention, same actor-restart
  technique for M1's single-hardcoded-tenant seam).

## CPG

Considered, not relevant — same reasoning the architect plan itself gave (§2): this is a
behavior/acceptance pass against a small, already-well-documented, already-twice-reviewed feature;
a call-graph query would not answer "does the shipped system satisfy AC-1..AC-8" faster than
reading `server/falkorchat/*.py` directly and driving the running server. No CPG query issued.

## Risk assessment / prioritization

Six implementation stages are already diff-gated and independently spot-checked by `teco`
(re-run suite counts matching the delegate's own, live counts re-probed via read-only Cypher) —
the residual risk this pass exists to close is not "does the mechanism work in isolation" but
**whether it holds when driven through the real front doors, under the real M1 single-tenant
constraint, and across two genuinely different actors** — the exact motivating shape the ML note
names (§4.1: an agent-team KB file ingested by one actor, edited by another). Priority order:

1. **The two zero/low-review-cost guarantees (AC-1 auto-supersede, AC-6/AC-8 delete)** — these are
   the highest-blast-radius paths per the ML note's own F7 (a false auto-supersede hides
   independent content from default search; a delete is irreversible content removal). A live miss
   here is the highest-impact defect this pass could find.
2. **AC-2/AC-3 cross-actor suggested-tier + review lifecycle** — the feature's actual reason for
   existing (agent-team KB re-ingestion by a different actor), and the one behavior with no
   analogous coverage in the K-050 predecessor pass (entities there are actor-agnostic by
   construction; documents carry an explicit `ingestedBy` this feature must not gate candidate
   generation on).
3. **AC-4 (search exclusion)** — a correctness claim with a concrete, previously-fixed regression
   history (Stage C's Pass-3 blockers, both re-verified closed) — re-confirmed live, not assumed
   fixed forever.
4. **AC-5 (history) and AC-7 (transport parity)** — solid existing coverage (both transports'
   independent offline suites, per plan §5's own stated altitude for AC-7); re-confirmed live
   rather than re-derived from scratch.
5. **The five "additional, non-AC-mapped" items (plan §5)** — pure-unit coverage (cite, don't
   re-derive — already comprehensive offline), the concurrency regression (add a live HTTP-altitude
   corroboration beyond the existing `threading.Barrier` repository-level proof), the
   delete-during-in-flight-background race (genuinely uncovered offline — drive live), the
   cascade-non-effect regression (covered offline at repository level — add one live end-to-end
   corroboration reusing real extraction), and the `SUPERSEDES` reopen-on-corroboration path
   (covered offline — add one live repository-altitude corroboration, same live-acceptance-
   technique limitation the K-050 predecessor named for the analogous `SAME_AS` case, §"Deliberate
   scope calls" below).
6. **The ML note's named known-gap scenario (§7)** — exercised explicitly as an accepted-behavior
   test case, not chased as a defect.

**Deliberate scope calls:**

- **The `SUPERSEDES` automatic reopen-on-corroboration path (OQ-3 path 1) cannot be forced through
  two ordinary live ingests**, for the identical structural reason the K-050 predecessor named for
  `SAME_AS` (`document-ingestion.md` test plan, "Deliberate scope calls"): `create_or_reopen_
  supersede_suggestion` reopens an edge only when the **same** `(new_document_id,
  candidate_document_id)` pair re-derives, but every ingest mints a fresh `documentId` — a third
  ingestion can never reproduce the first two's exact pair, only create new ones. This is a
  live-acceptance-**technique** limit, not a testability gap in the system: the repository method
  itself is directly, live-drivable (it takes plain scalar ids, no ingestion pipeline involved) —
  TP-09 drives it directly against the real, disposable workspace (two real `GRAPH.QUERY` round
  trips, not a re-run of the offline pytest), corroborating
  `test_create_or_reopen_supersede_suggestion_reopens_a_rejected_edge` (`test_repository.py:1155`)
  rather than substituting for it.
- **AC-7's transport-parity claim is not doubled for every single call in the live fixture.** Per
  the plan's own §5 test-strategy row for AC-7, the altitude is "REST + MCP integration... proven
  by both transports' independent suites... not a bespoke parity harness" — confirmed by grep that
  every new document-ingestion2 method has both a `test_api.py` and a `test_mcp.py` case (below).
  The live fixture still **alternates** transports throughout (REST and MCP each write and each
  read at least once) for genuine live corroboration beyond the offline citation, matching the
  K-050 predecessor's own AC-6 technique (a real out-of-process MCP Streamable-HTTP client, not an
  in-process FastMCP call).
- **The delete-during-in-flight-background race (TP-16) is a best-effort timing race, reported
  honestly either way it lands** — a background scheduler under real load may complete before the
  delete request round-trips; this pass fires the delete with no artificial wait and reports what
  was actually observed (crash vs. no crash), not a guaranteed reproduction of the race window
  itself. The plan's own accepted residual (§2.1 — a rare orphaned `Entity` node if a chunk is
  deleted mid-extraction) is not independently provable on demand; this pass asserts the two things
  that *are* provable regardless of whether the exact window was hit: the server does not crash and
  the document is genuinely gone.

## Environment

- **Disposable workspace `ws:docingest2qa`** — not `ws:acme`/`reference`, per the brief's own
  guidance (no shared-state risk, no re-seed obligation). `FALKORCHAT_WORKFLOW_ENABLED=0` (this
  feature has no workflow-engine dependency, plan §1 "no web UI work," and disabling it removes an
  unrelated LM-Studio consumer from this pass's concurrency footprint).
- **Actor setup (M1's single-hardcoded-tenant seam, `config.get_context()`):** both a `User`
  (`u1`) and an `Agent` (`assistant`) are seeded once (`seed_demo.sh`, idempotent, seeds both
  regardless of which one `FALKORCHAT_USER_ID` runtime-resolves to). Because `CallContext.actor` is
  a **process-wide** constant (`config.USER_ID`, read once at import), exercising two genuinely
  different actors requires **two server runs** against the same workspace: Run 1 with
  `FALKORCHAT_USER_ID=u1`, Run 2 with `FALKORCHAT_USER_ID=assistant` — the same restart-based
  technique the K-050 predecessor's own multi-actor drives would have needed had that feature's ACs
  required it (they didn't; this one's motivating scenario, per the ML note, explicitly does).
- **LM Studio reachable** (`http://localhost:1234/v1/models` — confirmed, embedding model
  `text-embedding-qwen3-embedding-0.6b` present, matching `config/models.json`'s configured
  `lmstudio/text-embedding-qwen3-embedding-0.6b` @ dim 1024). **Same stale-gateway-IP environment
  quirk the K-050 pass already documented**: the shared `~/.config/opencode/opencode.json` still
  has `baseURL: http://192.168.0.69:1234`, unreachable from this WSL2 box. Worked around
  identically — `FALKORCHAT_OPENCODE_CONFIG` pointed at a corrected local copy
  (`baseURL: http://localhost:1234`, otherwise byte-identical); the shared file itself is **not**
  modified.
- **`--reload` disabled** (`UVICORN_ARGS="--timeout-keep-alive 5"`, a non-empty override — an empty
  string does not defeat the script's `:-` default, K-050 report's own documented gotcha) — writing
  this plan/report under `falkor-chat/docs/` mid-run must not kill in-flight background threads.
- **Baseline cited, not re-run in full**: offline `pytest -q` **2813 passed, 14 deselected** and
  `./scripts/test_queries.sh` **458/459** (one pre-existing, already-diagnosed `Edge By Index Scan`
  gap, `docs/plans/document-ingestion2-coordination.md`'s Stage D-fix note) — both current per the
  dispatch brief and the coordination ledger's own independently-reproduced numbers; not
  re-litigated (would also wipe the shared `reference` graph other concurrent sessions depend on,
  for no benefit — this pass's own workspace is disposable and untouched by that hazard). **`tests/
  test_update_detection.py` is re-run directly by this pass** (pure functions, no FalkorDB fixture,
  zero shared-state risk) as live-executed evidence for the plan's "pure-unit" coverage bullet,
  not merely cited.

## Test items

| ID | AC / item | Title | Altitude | Priority |
|---|---|---|---|---|
| TP-01 | AC-1 | Byte-identical-modulo-whitespace re-ingest auto-supersedes, no confirmation, same actor, REST→MCP | e2e | High |
| TP-02 | AC-5 | History + direct old-version read after TP-01's supersede | integration | High |
| TP-03 | AC-4 | Default search excludes the superseded version's chunks, includes the new one | e2e (real embedder) | High |
| TP-04 | AC-2 | Cross-actor plausible edit lands a pending suggestion; candidate generation is not actor-scoped | e2e, background-completion-aware | High |
| TP-05 | AC-3 | Confirm branch: confirm the TP-04 suggestion, old non-current, chunks flipped | contract (REST) | High |
| TP-06 | AC-5 | History reflects the 3-version chain after TP-05's confirm | integration | Medium |
| TP-07 | AC-2/AC-3 | Second cross-actor pair (Beta family) + reject branch: both stay independent, current, searchable | e2e + contract (MCP) | High |
| TP-08 | AC-3 | Manual recheck (OQ-3 path 2) reopens the TP-07 rejected suggestion to pending | contract (REST) | Medium |
| TP-09 | additional | `SUPERSEDES` automatic reopen-on-corroboration — direct repository-level live drive | repository integration (live) | Medium |
| TP-10 | AC-6/AC-8 | Delete a non-current (superseded) version; deletion audit trail | e2e | High |
| TP-11 | additional | Cascade-non-effect: `Entity`/`RELATES_TO` extracted from TP-10's deleted document survive | integration (live, real extraction) | High |
| TP-12 | AC-6/AC-8 | Delete the current tip of a chain; excluded from list/search; deletion audit | e2e | High |
| TP-13 | additional | Known v1 gap (ML note §7): title+content rewrite escapes both candidate signals, becomes independent doc | e2e (accepted-behavior case) | Medium |
| TP-14 | additional | Concurrency regression, live HTTP altitude: two near-simultaneous identical-content POSTs → exactly one confirmed edge | e2e (concurrency) | High |
| TP-15 | additional | Delete racing in-flight background processing — no crash, content genuinely gone | e2e (best-effort race) | Medium |
| TP-16 | additional | Pure-unit `update_detection.py` coverage — executed directly | unit (live-executed) | Low |
| TP-17 | AC-7 | Transport-parity citation (offline `test_api.py`/`test_mcp.py` per new method) + live transport alternation across TP-01..TP-15 | contract (REST + MCP) | High |

### TP-01 — AC-1 auto-supersede, no confirmation
**Preconditions:** Run 1 (`FALKORCHAT_USER_ID=u1`) server up, `ws:docingest2qa`.
**Steps:** `POST /documents` (REST, actor u1) with distinctive text "Alpha family v1..." (Family
Alpha V1). Immediately re-ingest the **same** text with different surrounding whitespace/case via
the MCP `ingest_document` tool (real Streamable-HTTP client), same actor.
**Expected:** MCP receipt has `autoSuperseded: true`, `supersededDocumentId` = Alpha V1's id.
`GET /documents/{alphaV1}` (REST) → `currentVersion: false`, `supersededAt` set, `supersededBy:
'system'`. `GET /documents/{alphaV2}` → `currentVersion: true`. Direct Cypher: `SUPERSEDES{status:
'confirmed', decidedBy:'system', confidence:1.0, technique:'exact_normalized_text_hash'}` edge
Alpha V2 → Alpha V1.

### TP-02 — AC-5 history + direct old-version read
**Steps:** `GET /documents/{alphaV1}/history` (REST) and MCP `get_document_history(alphaV1)`.
`GET /documents/{alphaV1}` directly (REST).
**Expected:** both history calls return the 2-version chain (V1, V2) with correct `currentVersion`/
`supersededAt`/`supersededBy` per row; direct `GET` on the superseded V1 still returns its full
`text` (unchanged — only default *search* excludes it, not direct lookup, per plan §3.3).

### TP-03 — AC-4 search excludes superseded content
**Preconditions:** Alpha V2's chunks embedded (poll `GET /documents/{alphaV2}` or retry the search
call — embedding is async, real LM Studio call).
**Steps:** `GET /documents/search?q=<Alpha-distinctive-term>&limit=10` (REST); MCP
`search_documents`.
**Expected:** results include a chunk from Alpha V2, **never** a chunk from Alpha V1 (superseded).

### TP-04 — AC-2 cross-actor suggested-tier detection
**Preconditions:** Run 2 (`FALKORCHAT_USER_ID=assistant`), same `ws:docingest2qa`.
**Steps:** `POST /documents` (REST, actor `assistant`) with a **plausible-but-not-identical** edit
of Alpha V2's text (same distinctive subject, materially overlapping wording, above the 0.1 Jaccard
noise floor but not identical) → Alpha V3, `ingestedBy = assistant`. Poll `GET /document-updates/
pending` (no LLM dependency — detection is pure Python/Cypher, expected fast) until a suggestion
appears or a bounded retry budget is exhausted.
**Expected:** `SUPERSEDES{status:'pending', technique:'shingled_jaccard_overlap'}` edge Alpha V3 →
Alpha V2, **even though V2 was ingested by `u1` and V3 by `assistant`** — direct live proof of the
ML note's §4.1 "candidate generation must not be actor-scoped" requirement, not merely a repository-
level assertion. `confidence` is the raw Jaccard ratio (0 < value < 1, not exactly 1.0). Alpha V2
still `currentVersion: true` and still returned by TP-03-style search (not yet superseded — only a
suggestion).

### TP-05 — AC-3 confirm branch
**Steps:** `POST /document-updates/{matchId}/confirm` (REST, actor `assistant`) on TP-04's
suggestion.
**Expected:** `200`, `status:'confirmed'`. Direct read: Alpha V2 `currentVersion:false`,
`supersededBy` = the deciding actor (`assistant`, **never** `'system'` on this manual path); its
chunks' `documentCurrent` flipped `false`. Alpha V3 `currentVersion:true`.

### TP-06 — AC-5 3-version history after confirm
**Steps:** `GET /documents/{alphaV1}/history` (any id in the chain works, per plan §3.7).
**Expected:** all three versions (V1→V2→V3), correctly ordered, correct `currentVersion`/audit
fields per row.

### TP-07 — AC-2/AC-3 cross-actor pair + reject branch (Family Beta)
**Steps:** `POST /documents` (REST, actor `u1`, Run 1 — or re-use Run 2's `assistant` context for
the first Beta document and switch for the edit; whichever ordering the live run lands on, both
Beta documents must be ingested by **different** actors, mirroring TP-04) — Beta V1. A plausible
edit — Beta V2, different actor. Poll for the pending suggestion (MCP `list_pending_document_
updates`). `POST /document-updates/{matchId}/reject` (MCP `reject_document_update`).
**Expected:** suggestion lands cross-actor (same live proof as TP-04, second corroboration).
Reject → `status:'rejected'`. Both Beta V1 and V2 remain `currentVersion:true`; both are returned
by a search for their shared distinctive term (neither hidden).

### TP-08 — AC-3 manual recheck (OQ-3 path 2)
**Steps:** `POST /document-updates/{betaMatchId}/recheck` (REST) on TP-07's rejected suggestion.
**Expected:** `status:'pending'` again, `resuggestCount` incremented, `lastResuggestedAt` stamped.
Both Beta documents still independent/current (recheck alone doesn't confirm).

### TP-09 — `SUPERSEDES` automatic reopen-on-corroboration (repository-altitude, live)
**Preconditions:** a small Python one-liner using the server's own venv, invoked against the live
`ws:docingest2qa` FalkorDB connection directly (`falkorchat.repository`, not through HTTP) — the
plan's own §5 altitude for this item ("mirrors the existing `create_or_reopen_match` test shape").
**Steps:** call `repository.create_or_reopen_supersede_suggestion(...)` for a fresh, disclosed pair
of pre-existing `currentVersion` documents (Gamma V1/V1b, written earlier in this pass for TP-13,
reused here as disclosed test data — their content shape is irrelevant to this item, only their
ids). Reject the resulting edge. Call `create_or_reopen_supersede_suggestion` **again with the
identical `(new_document_id, candidate_document_id, match_id)` triple**.
**Expected:** the second call reopens the same edge to `status:'pending'` (never straight to
`'confirmed'`), `resuggestCount` bumped on the **original** `matchId`, no duplicate `SUPERSEDES`
edge between the pair (`MATCH (a)-[r:SUPERSEDES]->(b) WHERE ... RETURN count(r)` = 1). Corroborates
`test_create_or_reopen_supersede_suggestion_reopens_a_rejected_edge` (`test_repository.py:1155`)
against the real live instance rather than substituting for it.

### TP-10 — AC-6/AC-8 delete a non-current version
**Steps:** `DELETE /documents/{alphaV1}` (REST) — Alpha V1, already superseded (non-current) since
TP-01.
**Expected:** `200`. `GET /documents/{alphaV1}` → `404`. `GET /documents/{alphaV1}/deletion` →
`{deletedBy:'u1', deletedAt:...}`. Direct Cypher: `Document{documentId:alphaV1}` and its `Chunk`s
gone; `(:DocumentDeletion{documentId:alphaV1})` present.

### TP-11 — Cascade-non-effect regression (live, real extraction)
**Preconditions:** Alpha V1's text includes a clear entity-bearing sentence (e.g. "Alpha Systems
acquired Northgate Robotics in 2024.", mirroring the K-050 predecessor's own reliable fixture
shape); background extraction confirmed complete **before** TP-10's delete (poll for `Entity`
presence).
**Steps:** direct Cypher read, after TP-10's delete: `MATCH (e:Entity) WHERE e.name IN ['Alpha
Systems','Northgate Robotics'] RETURN e`; `MATCH ()-[r:RELATES_TO]->() WHERE r.sourceDocumentId =
$alphaV1 RETURN r`.
**Expected:** both `Entity` nodes and the `RELATES_TO` edge survive the document/chunk hard delete
untouched (plan §3.6's "zero cascade on delete" decision) — a live regression proof, not just
design-note prose, corroborating `test_delete_document_leaves_entities_and_relates_to_untouched`
(`test_repository.py:990`).

### TP-12 — AC-6/AC-8 delete the current tip
**Steps:** `DELETE /documents/{alphaV3}` (MCP `delete_document`) — the current tip of the Alpha
chain (post TP-05's confirm).
**Expected:** MCP receipt `{documentId:alphaV3, deleted:true}`. `GET /documents` (REST, `list_
documents`) no longer lists Alpha V3. `GET /documents/search` for Alpha's distinctive term returns
nothing (the whole Alpha family is now either deleted or superseded). MCP `get_document_
deletion(alphaV3)` returns the audit record.

### TP-13 — Known v1 gap: title+content rewrite escapes both candidate signals
**Steps:** `POST /documents` — Gamma V1, title "Quarterly Ops Notes", body a few distinctive
sentences. A second, **substantially rewritten** document — different subject-appropriate title
("Facility Maintenance Log") and body reworded enough that its shingle set shares negligible
overlap with Gamma V1's (different vocabulary, different sentence structure, same rough real-world
topic only in the loosest sense) — Gamma V2. Poll `GET /document-updates/pending` with a bounded
retry budget.
**Expected (accepted behavior, not a defect — ML note §7):** **no** `SUPERSEDES` suggestion forms
between Gamma V1 and V2. Both remain independent, `currentVersion:true`, both searchable. Reported
as the named, accepted v1 limitation exercised live, exactly per the plan's Stage E brief — **if**
a suggestion unexpectedly *does* form, that is worth noting as an interesting (better-than-
documented) outcome, not a failure either way.

### TP-14 — Concurrency regression, live HTTP altitude
**Steps:** two `POST /documents` calls with **identical** text, fired near-simultaneously (shell
`curl ... & curl ... & wait`, no artificial delay) against the running server.
**Expected:** exactly one of the two receipts has `autoSuperseded:true` (or, if both land before
either commits, the two race for which becomes "old" — either outcome is acceptable as long as
exactly one confirmed edge results); direct Cypher: exactly one `SUPERSEDES{status:'confirmed'}`
edge between the pair, never zero, never duplicated. Corroborates `test_create_document_with_
auto_supersede_concurrent_calls_produce_exactly_one_edge` (`test_repository.py:1670`, a
`threading.Barrier`-precise proof) under realistic concurrent HTTP request pressure instead.

### TP-15 — Delete racing in-flight background processing
**Steps:** `POST /documents` with a multi-paragraph document (forces several chunks, several
background embed/extract jobs); immediately (no wait) `DELETE /documents/{id}`.
**Expected (best-effort — see "Deliberate scope calls"):** the `DELETE` call itself succeeds
(`200`) or 404s cleanly if it lost the race to a same-millisecond `GET`-visibility window — either
is acceptable; what is **not** acceptable is a `5xx`/crash. Server stays responsive afterward
(`GET /health`). `GET /documents/{id}` → `404` once the delete has landed. Server log/stdout
checked for an unhandled exception traceback around the event window (background jobs must degrade
safely per plan §2.1's `MATCH`-anchored discipline).

### TP-16 — Pure-unit `update_detection.py` coverage
**Steps:** `.venv/bin/python -m pytest -q tests/test_update_detection.py`.
**Expected:** all cases pass, including the ML note's own named cases (empty text, whitespace-only
variance, identical-modulo-case-and-whitespace pairs, a hard-negative shared-boilerplate pair) —
executed directly by this pass, not merely cited from the dispatch-brief baseline.

### TP-17 — AC-7 transport parity
**Steps:** `grep -c "^def test_.*document\|^def test_.*supersede\|^def test_.*update" tests/
test_api.py tests/test_mcp.py` scoped to the nine new document-ingestion2 methods (delete, list,
history, deletion, list-pending, list-all, confirm, reject, recheck); cross-check against the live
transport log for TP-01..TP-15 (which calls used REST vs. MCP).
**Expected:** both `test_api.py` and `test_mcp.py` carry at least one case per new method (already
confirmed present by this plan's own reconnaissance — cited, not re-derived); the live fixture
above used **both** transports for writes (TP-01 REST+MCP, TP-04 REST, TP-07 MCP) and both for
reads (TP-02 REST+MCP, TP-12 MCP) — genuine live cross-transport corroboration, not a single-
transport run dressed up as parity coverage.

## Entry / exit criteria

**Entry:** FalkorDB reachable (`PONG`); offline baseline current per the dispatch brief (not
re-run, see Environment); `ws:docingest2qa` freshly bootstrapped + seeded (both `User u1`/`Agent
assistant` present); LM Studio reachable with the embedding model listed.

**Exit:** all seventeen items executed to pass/fail/blocked with evidence; TP-13's outcome (gap
reproduced or not) reported honestly either way; TP-15's race outcome reported as observed, not
assumed; `ws:docingest2qa` left in place (disposable, not torn down — nothing else depends on it,
unlike `reference`/`ws:acme`) for any follow-up inspection.

## Out of scope

- Re-deriving unit/integration coverage the offline suite (2813 passed/14 deselected, independently
  diff-gated stage-by-stage, Passes 1-7) already proves — this pass extends past it, not under it.
- Forcing the `SUPERSEDES` *automatic* reopen path through two ordinary live ingests (a
  live-acceptance-technique limit, not a system gap — see "Deliberate scope calls"; TP-09 drives
  the mechanism directly instead).
- Web UI — no UI work is in scope for this feature (plan §1).
- Load/scale/performance testing beyond the concurrency regression item already named.
- Re-litigating OQ-1's original entity-fusion methodology or the K-050 feature's own already-closed
  defects (`Document.status` terminal-state, `DESIGN.md` currency) — orthogonal to this feature,
  checked only incidentally if encountered.
