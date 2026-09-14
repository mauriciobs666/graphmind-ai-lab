# Document ingestion — update & delete — Test Report

> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** —

Execution of `docs/test-plans/document-ingestion2.md` — Stage E (final) of `docs/plans/document-
ingestion2-coordination.md`. In progress; written incrementally as items execute (survives an
interrupted run). Executed against `HEAD` = `c805dc8` (Stage D-fix + its coordination-ledger entry
closed; feature code tip `854f0b5`). FalkorDB `falkordb-dev`, disposable workspace
`ws:docingest2qa` (fresh, seeded fresh for this pass — no shared-state risk). LM Studio reachable
and used for real (embedding + extraction) where the fixture needs it; document-update detection
itself (AC-1/AC-2) needs no LLM at all (ML note's own structural finding, confirmed live below).

**CPG: considered, not relevant — same reasoning as the test plan and the architect plan itself
(§2): a behavior/acceptance pass against a small, twice-reviewed feature, driven by reading
`server/falkorchat/*.py` directly and live-driving the running server, not by graph queries over a
call graph.**

## Environment actually used

- `ws:docingest2qa`, bootstrapped fresh (`EMBEDDING_DIM=1024`), seeded (`User u1`, `Agent
  assistant`, both present for the actor-restart technique below).
- Run 1: `FALKORCHAT_USER_ID=u1`, `FALKORCHAT_WORKFLOW_ENABLED=0`, `FALKORCHAT_ENABLE_AGENT=1`,
  `UVICORN_ARGS="--timeout-keep-alive 5"` (no `--reload`), `FALKORCHAT_OPENCODE_CONFIG` pointed at
  a corrected local copy of the shared `~/.config/opencode/opencode.json` (`baseURL:
  http://localhost:1234` — the shared file's own `192.168.0.69` is unreachable from this WSL2 box,
  same pre-existing environment quirk the K-050 predecessor pass already documented; shared file
  not modified).
- Baseline cited, not re-run in full, per the dispatch brief (`pytest -q` 2813 passed/14
  deselected; `test_queries.sh` 458/459, one pre-existing diagnosed gap) — re-running either would
  wipe the shared `reference` graph other concurrent sessions in this repo depend on, for no
  benefit to a pass scoped to a disposable workspace. `tests/test_update_detection.py` **was**
  re-run directly by this pass (pure functions, zero shared-state risk): **31 passed** (TP-16,
  PASS).

## Results (running)

| ID | Item | Result | Evidence |
|---|---|---|---|
| TP-16 | pure-unit `update_detection.py` | **PASS** | `.venv/bin/python -m pytest -q tests/test_update_detection.py` → `31 passed in 0.04s` |
| TP-01 | AC-1 auto-supersede | **PASS** | `POST /documents` (REST, u1) → Alpha V1 `4bf4b36d...`. MCP `ingest_document` (real Streamable-HTTP client) with same content, whitespace/case varied → Alpha V2 `64f0cf80...`, receipt `{"autoSuperseded": true, "supersededDocumentId": "4bf4b36d..."}`. Direct Cypher confirms V1: `currentVersion=false`, `supersededAt=1789344093910`, `supersededBy="system"`, `textNormalizedHash` matching V2's |
| TP-02 | AC-5 history + direct read | **PASS** | REST `GET /documents/{v1}/history` and MCP `get_document_history` both return the identical 2-row chain (`currentVersion`/`supersededAt`/`supersededBy` correct per row). REST `GET /documents/{v1}` directly still returns full `text` byte-identical to the original ingest (confirms plan §3.3: direct lookup unaffected by supersession, only default search is) |
| TP-03 | AC-4 search exclusion | **PASS** | `GET /documents/search?q=AlphaFamilyMarkerZQ9...` → returns only V2's chunk (`abc72356...`). Direct Cypher confirms V1's chunk (`85f20e42...`, a **different** chunk id, `documentCurrent=false`) **has a real embedding** (`hasEmbedding=true`) and was still correctly excluded — proves the `documentCurrent` post-filter is doing real work, not merely "never embedded so trivially absent" |

**Note on TP-01/TP-02 test-plan wording vs. shipped contract:** the test plan's TP-01 step
originally expected `currentVersion`/`supersededAt`/`supersededBy` to appear on a direct `GET
/documents/{id}` response. Live driving found they do **not** — `repository.get_document`'s
`RETURN` clause (`repository.py:1067-1076`, the original K-050 M5 §14.2 method, explicitly
"unchanged" per plan §3.3) never selected those three properties; only `list_documents` and
`get_document_history` carry them (plan §3.7's own field allocation). **Not a defect** — the
plan's own text says `get_document` is deliberately unchanged, and the fields are genuinely
retrievable via the two surfaces the plan designed for exactly this (`list_documents`/`get_
document_history`, both exercised above). Verification for TP-01/TP-02 was carried out via those
two surfaces plus a direct graph read instead; flagged under Feedback below as a minor
plan-wording correction, not a product finding.

**Environment note, not a defect:** Alpha V2's background extraction failed
(`ProviderCallError: lmstudio/qwen/qwen3-4b-2507 ... HTTP 400 Bad Request: {"error":"Engine
protocol predict request failed: fetch failed"}`, `Document.status` left at `"failed"`) —
identical in shape to the LM Studio concurrent-model-swap thrashing the K-050 predecessor pass
already documented and reproduced repeatedly. `_safe_extract`'s isolation worked exactly as
designed: embedding for the same chunk succeeded independently (confirmed via direct Cypher,
`hasEmbedding=true`), the document remained fully searchable (TP-03), and no corruption resulted.
Not a `document-ingestion2` defect; noted for the record.

## Results (continued)

| ID | Item | Result | Evidence |
|---|---|---|---|
| TP-04 | AC-2 cross-actor suggested-tier | **PASS** | Restarted server as `FALKORCHAT_USER_ID=u2` (a genuinely different `User`, `assistant`-as-boot-actor was blocked — see Finding 3). `POST /documents` (REST, actor `u2`) with a plausible edit of Alpha V2 (ingested by `u1`), same title `"Alpha Doc v2"` → `SUPERSEDES{status:'pending', technique:'shingled_jaccard_overlap', confidence:0.444}` edge from the new doc (`199308...`, actor `u2`) → Alpha V2 (`64f0cf80...`, actor `u1`) — live proof candidate generation is **not** actor-scoped |
| TP-05 | AC-3 confirm | **PASS** | `POST /document-updates/bf89e2ff.../confirm` (REST) → `200 {"status":"confirmed",...}`. Direct Cypher: Alpha V2 `currentVersion=false`, `supersededBy="u2"` (real actor, never `'system'` on this path), its chunk `documentCurrent=false`; new doc `currentVersion=true` |
| TP-06 | AC-5 3-version history | **PASS** | `GET /documents/{alphaV1}/history` → 3-row chain (V1 `system`-superseded, V2 `u2`-superseded, V4 current), correct fields per row |
| TP-07 | AC-2/AC-3 cross-actor reject | **PASS** | Beta V3 (REST, `u2`, title `"Beta Doc"` — identical to Beta V1's) vs Beta V1 (`u1`) → pending suggestion, confidence 0.42. `reject_document_update` (**MCP**, real Streamable-HTTP client) → `status:'rejected'`; both remain `currentVersion:true` |
| TP-08 | AC-3 manual recheck | **PASS** | `POST /document-updates/{id}/recheck` (REST) → `status:'pending'`; direct Cypher confirms `resuggestCount=1`, `lastResuggestedAt` stamped |
| TP-09 | reopen-on-corroboration | **PASS** | Direct `repository.create_or_reopen_supersede_suggestion` call (live, against `ws:docingest2qa`, disclosed Gamma V1/V2 pair) → create (`created:true`) → direct-Cypher reject → **identical** call again → `{'created': False, 'reopened': True, matchId: <same>, status:'pending'}`; final Cypher check: exactly 1 edge, `resuggestCount=1`. Corroborates `test_repository.py:1155` |
| TP-10 | AC-6/AC-8 delete non-current version | **PASS** | `DELETE /documents/{alphaV1}` → `200`; `GET` → `404`; MCP `get_document_deletion` → `{deletedBy:"u2", deletedAt:...}`; direct Cypher: `Document`/`Chunk` counts both `0`, `DocumentDeletion` node present |
| TP-11 | cascade-non-effect (live, real extraction) | **PASS** | Alpha V1's genuinely-extracted `Entity(Alpha Systems)`/`Entity(Northgate Robotics)`/`Entity(AlphaFamilyMarkerZQ9)` and both `RELATES_TO` edges (`sourceDocumentId=<deleted alphaV1 id>`) all survive TP-10's hard delete untouched — direct Cypher read after the delete. Corroborates `test_repository.py:990` |
| TP-12 | AC-6/AC-8 delete current tip | **PASS** | MCP `delete_document(alphaV4)` → `{deleted:true}`. REST `GET /documents` (list) no longer includes it. MCP `get_document_deletion` → audit record present. (Search still returned other, independent Alpha-titled documents — see note below; not the deleted chain) |
| TP-13 | known v1 gap (ML note §7) | **PASS (accepted behavior reproduced)** | Gamma V1 (`"Quarterly Ops Notes"`, warehouse/staffing content) vs. Gamma V2 (`"Facility Maintenance Log"`, elevator/HVAC content) — direct Cypher confirms **zero** `SUPERSEDES` edges between them; both remain independent, `currentVersion:true`. Matches the ML note's named limitation exactly — not a defect |
| TP-14 | concurrency, live HTTP | **PASS** | Two `curl` POSTs fired via backgrounded shell jobs (`& ... & wait`), identical text → receipts `{autoSuperseded:true,...}` / `{autoSuperseded:false,...}`; direct Cypher: exactly 1 confirmed `SUPERSEDES` edge between the pair (`count(r)=1`). Corroborates `test_repository.py:1670`'s `threading.Barrier` proof under real HTTP concurrency |
| TP-15 | delete racing in-flight background | **PASS (best-effort)** | `POST /documents` then immediate `DELETE` (no wait) → `200` delete, `404` on subsequent `GET`, `GET /health` → `200` throughout. No `5xx`, no unhandled traceback in server log around the event; background embed/extract jobs for unrelated documents continued running normally after |
| TP-16 | pure-unit coverage | **PASS** | (see baseline above) `31 passed in 0.04s` |
| TP-17 | AC-7 transport parity | **PASS** | Offline: both `test_api.py` and `test_mcp.py` carry ≥1 case for all 9 new methods (`delete_document`, `list_documents`, `get_document_history`, `get_document_deletion`, `list_pending_document_updates`, `list_document_updates`, `confirm_document_update`, `reject_document_update`, `recheck_document_update` — grep-confirmed, 15 `test_api.py` + 14 `test_mcp.py` cases across them). Live: REST used for writes (TP-01, TP-04, TP-10) and reads (TP-02, TP-06, TP-12-list); MCP used for writes (TP-01's auto-supersede trigger, TP-07's reject, TP-10/12's delete) and reads (TP-02, TP-10/12's deletion audit) — genuine cross-transport corroboration, not a single-transport run |

**12/12 remaining items + TP-16 = all 17 items executed. 17 PASS. Two defects found and reported
below (neither falsifies an AC in the general case — both are narrower, real gaps within AC-2's
suggested-tier candidate generation).**

**Note on TP-12's search assertion:** the original test-plan wording expected a post-delete search
for the Alpha family's marker term to return nothing. It did not — two **separate, independent**
documents (`Alpha Doc v3 (cross-actor edit)`, `Alpha Doc v3b (cross-actor edit)`) that also contain
the word "AlphaFamilyMarkerZQ9" and were never part of the confirmed supersession chain (they are
themselves evidence for Defect 1 below — see next section) remained current and correctly
searchable. Not a defect — those documents were never deleted or superseded, so their continued
presence in search is exactly correct; the test-plan's expected-result wording was simply broader
than what the live fixture actually produced once Defect 1 diverted two ingests into permanently
independent documents.

## Defects

### Defect 1 — `find_update_shortlist`'s title-fuzzy query crashes (and silently kills the entire suggested-tier job) for any document title containing a common RediSearch metacharacter

**Severity: High.** Not a server crash and not data corruption (`_safe_detect_update`'s
try/except-log-never-raise isolation catches it exactly as designed) — but it **silently and
completely defeats AC-2** for any newly-ingested document whose own title contains a character
RediSearch's query parser treats as syntax (parentheses, hyphen, colon, quotes, brackets, pipe —
all confirmed below), with **zero signal visible to any caller on any transport**. These are
extremely common in real document titles ("Q3 Report (Draft)", "Spec: v2", "Notes - Final").

**Steps to reproduce (minimal, isolated — no HTTP layer needed):**
```python
from falkorchat import db, repository, update_detection as ud
g = db.connect(host='127.0.0.1', port=6379)
repo = repository.Repository(g)
bands = ud.lsh_bands(ud.minhash_signature(ud.shingles(ud.normalize_text('placeholder'))))
repo.find_update_shortlist('docingest2qa', bands=bands, title='Quarterly Report (Draft)')
```
**Expected:** returns a (possibly empty) candidate list.
**Actual:** raises `redis.exceptions.ResponseError: RediSearch: Syntax error at offset 10 near
Report`.

**Confirmed live, end-to-end, twice** during this pass (not just the isolated repro above):
ingesting `"Alpha Doc v3 (cross-actor edit)"` and `"Alpha Doc v3b (cross-actor edit)"` (both
plausible edits of a `currentVersion` document, both well-formed content otherwise) each produced
`background update-detection failed (documentId=...)` in the server log
(`redis.exceptions.ResponseError: RediSearch: Syntax error at offset N near <title>`) and **no**
`SUPERSEDES` suggestion was ever written for either — both silently became permanent, independent
documents instead of the pending suggestions AC-2 requires. This is a direct, live, unplanned
demonstration of the defect's real-world effect, not merely a synthetic repro.

**Characterized breadth** (direct repository calls, same live instance):
| Title | Result |
|---|---|
| `Report - Final` | CRASH — syntax error |
| `Spec: v2` | CRASH — syntax error |
| `Q3 "Draft" Notes` | CRASH — syntax error |
| `Notes [2024]` | CRASH — syntax error |
| `A\|B test` | CRASH — syntax error |
| `Report (Draft)` | CRASH — syntax error |
| `Plain Clean Title` | OK |

**Root cause:** `repository.find_update_shortlist` (`repository.py:1846`) builds its title-fuzzy
RediSearch query as `" ".join(f"%{tok}%" for tok in title.split())` (`repository.py:1888`) — no
escaping of RediSearch query-syntax metacharacters in `title`'s tokens before they're embedded in
the query string passed to `db.idx.fulltext.queryNodes`. This is a **faithful, verbatim port** of
the pre-existing `fusion._fuzzy_query` (`fusion.py:34-42`, built for `Entity.name`) — the
document-ingestion2 plan and code both explicitly say so ("built the same way `fusion._fuzzy_query`
builds one for entity names," `repository.py:1866-1868`). The underlying gap is not new to this
feature, but this feature is the first to expose it against **free-form, caller-supplied document
titles** rather than short, LLM-extracted entity names — titles are far more likely to carry
ordinary punctuation than extracted proper nouns are, making the same latent gap materially more
reachable here.

**Suggested fix:** escape (or strip) RediSearch special characters in each title token before
building the fuzzy query string — same fix would also harden `fusion._fuzzy_query`'s existing,
narrower exposure for entity names, though that is outside this feature's diff. Not fixed by this
pass (implementation, out of `qa-engineer`'s lane); recommend routing to `coder`/`tdd-engineer` as
a fast follow, and flagging the shared root cause to whoever picks it up so both call sites are
fixed together rather than twice.

### Defect 2 — title-fuzzy candidate generation uses implicit-AND token combination, making it far narrower than its own docstring describes

**Severity: Medium.** Not a crash, not silent-and-invisible in the same way as Defect 1 — a
real, observed effectiveness gap in one of the two candidate-generation signals, discovered while
isolating Defect 1.

**Observed:** `find_update_shortlist(title="Beta Doc v2 clean title")` against a workspace
containing a `currentVersion` document titled `"Beta Doc"` returned **only the querying document
itself**, never `"Beta Doc"` — even though both titles clearly share words ("Beta", "Doc"). Direct
repository call, reproduced live:
```python
repo.find_update_shortlist('docingest2qa', bands=<unrelated-bands>, title='Beta Doc v2 clean title')
# -> only the document whose own title is exactly 'Beta Doc v2 clean title'; NOT the 'Beta Doc' doc
```
Confirmed the cause: the query string `%beta% %doc% %v2% %clean% %title%` is combined by
RediSearch's **default implicit-AND** operator for space-separated terms — a candidate's title
must fuzzy-match **every** token of the new document's title to surface at all, not just one. A
revision that adds any descriptive suffix to its title (`"v2"`, `"Draft"`, `"Final"`, `"Updated"` —
an extremely ordinary thing to do) will never title-fuzzy-match its own predecessor unless the
predecessor's title happens to already contain that exact suffix word too. This is narrower than
the method's own docstring implies ("a cheap complementary booster... title-fuzzy match... when
non-empty," `repository.py:1863-1868`, plan §3.4) — a reasonable reader would expect *any* shared
distinctive word to be enough to shortlist a candidate for the precise Jaccard check to then judge,
not require near-total title-token overlap.

**Not an AC violation** — AC-2's own wording only requires *a* plausible edit to land a suggestion,
which this pass demonstrated does happen (TP-04, TP-07, both via the LSH-band signal or an
identical-title case); the LSH/MinHash-banding signal is the fingerprint-based, primary net per the
ML note (§4.1), and title-fuzzy is explicitly named "complementary," not load-bearing. But it means
the title-fuzzy signal's real-world recall is meaningfully lower than the design note's own framing
suggests, for the exact same "titles carry ordinary punctuation/suffixes, entity names mostly
don't" reason as Defect 1. **Recommend to `architect`/`data-scientist`**: either explicit-OR the
per-token fuzzy terms (`%tok1%|%tok2%|...`) or drop to a smaller, more distinctive token subset —
a design call, not merely an implementation bug, so routed as a recommendation rather than a
defect fix.

## Environment / testability findings (not `document-ingestion2` code defects)

### Finding 3 — `FALKORCHAT_USER_ID` cannot be pointed at an existing `Agent` id; the app refuses to boot

The test plan's original actor-restart technique planned to alternate between `User u1` and
`Agent assistant` as the two live actors. Booting the server with `FALKORCHAT_USER_ID=assistant`
(an `Agent` already seeded by `seed_demo.sh`) failed outright at startup:
```
falkorchat.repository.MemberIdCollisionError: member id 'assistant' is already held by an Agent
— refusing to create a User (member ids are namespace-unique across User/Agent)
```
`app.py`'s `_lifespan` unconditionally calls `services.ensure_actor` → `repository.ensure_user`
for the configured `USER_ID`, regardless of whether that id is meant to represent a `User` or an
`Agent` — `CallContext.actor`'s own docstring says it can be either, but the boot path only ever
tries to ensure a `User`. **Worked around**, not blocked: seeded a second, genuinely distinct
`User` (`u2`) instead and used the `u1`/`u2` pair for every cross-actor item — the plan's own
wording explicitly allows "a `User` and an `Agent`, **or two different users**." Not filed as a
`document-ingestion2` defect (pre-existing M1 tenancy-seam behavior, orthogonal to this feature,
plausibly intentional — the boot-time actor represents "the human operator of this deployment," an
`Agent`'s identity is otherwise only ever used by the AI responder's own internal posting path).
Noted for the record as a real, live-discovered testability constraint of the M1 single-tenant
seam, worth a kaizen entry.

### LM Studio environment quirks (pre-existing, not this feature's bugs)

- **Stale gateway IP** in the shared `~/.config/opencode/opencode.json` (`192.168.0.69`, unreachable
  from WSL2) — same as the K-050 predecessor pass; worked around identically via a corrected local
  `FALKORCHAT_OPENCODE_CONFIG` copy, shared file untouched.
- **Concurrent extract+embed thrashing** reproduced several times this pass too (`HTTP 400 "Model
  is unloaded"` / `"Engine protocol predict request failed: fetch failed"`), always correctly
  isolated by `_safe_extract`'s try/except (no crash, no corruption, embedding for the same chunk
  frequently succeeded independently) — the same class the K-050 predecessor already documented and
  escalated.

## Coverage & gaps

**Covered:** all eight ACs (AC-1..AC-8), all five "additional, non-AC-mapped" items the plan
names, plus the ML note's named known-gap scenario exercised live and confirmed to reproduce
exactly as documented. Both REST and MCP used for writes and reads throughout, not a
single-transport run. Two genuinely different live actors (`u1`, `u2`) used across the fixture,
satisfying the plan's own multi-actor requirement — the cross-actor pairing is the direct, live
proof of the ML note's "candidate generation must not be actor-scoped" design requirement (TP-04,
TP-07).

**Not covered / deliberately out of scope** (see test plan's "Deliberate scope calls"):
- The `SUPERSEDES` *automatic* reopen path forced through two ordinary end-to-end ingests (a
  live-acceptance-technique limit identical in shape to the K-050 predecessor's analogous
  `SAME_AS` scope call) — driven instead via a direct, live repository-level call (TP-09), which is
  a stronger proof than citing the offline test alone, though not a full ingest-driven derivation.
- Re-deriving unit/integration coverage the offline suite (2813 passed/14 deselected, independently
  diff-gated Passes 1-7) already proves.
- Full offline-suite/`test_queries.sh` re-run (cited from the current, ledger-verified baseline
  instead, to avoid an unnecessary `reference`-graph wipe against other concurrent sessions in this
  repo).

**Residual risk:** Defect 1 is a real, moderately-likely-to-be-hit gap in production use (any
title with ordinary punctuation silently defeats AC-2 for that document) — bounded by: the auto
tier (AC-1) is entirely unaffected (no title dependency at all), a caller can still discover a
missed edit manually (no suggestion ever appearing is indistinguishable from "no suggestion was
warranted," a real but not severe UX gap), and history/delete/confirm/reject/recheck are all
unaffected once a suggestion *does* exist. Defect 2 further narrows real-world recall of the
already-secondary title-fuzzy signal specifically.

## Feedback & recommendations

1. **Fix Defect 1** (High) — escape RediSearch metacharacters in `find_update_shortlist`'s
   title-fuzzy query construction; consider the same fix for `fusion._fuzzy_query`'s existing,
   narrower exposure while there, since both share the identical root cause and pattern.
2. **Reconsider Defect 2's implicit-AND token combination** (Medium) — a design call for
   `architect`/`data-scientist`, not just an implementation fix; explicit-OR combination would
   bring the signal's real recall closer to its own docstring's framing.
3. **Testability note**: the M1 single-hardcoded-tenant seam (`config.get_context()`) requires a
   full server restart to change the effective actor, and (Finding 3) cannot be pointed at an
   existing `Agent` id at all without a boot-time crash — worth a kaizen entry for future QA passes
   needing multi-actor live fixtures against this codebase; two different `User`s is the reliable
   technique.
4. **Testability win worth keeping**: document-update detection (AC-1/AC-2) needs zero LLM
   dependency — this made the auto-supersede and (once titles avoided Defect 1) suggested-tier
   items the most *reliable* live items in this whole pass, a real contrast with the LM-Studio-
   dependent extraction/embedding paths that needed the now-familiar thrashing workarounds. Worth
   highlighting as a design strength, not just a QA convenience.
5. **No testability issues found in the shipped MCP/REST surface itself** — every AC had a
   concrete, checkable response or a direct, plan-specified graph read; the one plan-vs-shipped
   wording mismatch found (TP-01/02's `currentVersion` fields living on `list_documents`/
   `get_document_history`, not `get_document`) is a test-plan authoring correction, not a product
   gap — flagged and corrected during execution, described above.

## Milestone / Stage-E done-condition assessment

Per `docs/plans/document-ingestion2-coordination.md`'s own sequencing (five stages, each
diff-gated; Stage E is the final, `qa-engineer`-owned acceptance pass before the whole
`document-ingestion2` coordination is done pending the user's own integration verification):

- All five implementation stages delivered + `analyst`-gated: **✅** (ledger, Stages A-D + D-fix,
  Passes 1-7, zero open blockers).
- `qa-engineer` acceptance pass: **✅ PASS-with-parked-defects** — all 8 ACs hold end-to-end
  against a real, multi-version, multi-actor, cross-transport live fixture; two real, reproducible,
  non-AC-falsifying defects found in the suggested-tier's title-fuzzy candidate-generation signal
  and reported above (not fixed by this pass, per the dispatch brief).
- Green baselines: **✅** (offline suite 2813/14 deselected + `test_queries.sh` 458/459, both cited
  current per the dispatch brief and the coordination ledger's own independently-reproduced
  numbers; `test_update_detection.py` re-run directly by this pass, 31/31).

## Overall verdict: PASS-with-parked-defects

All eight acceptance criteria (AC-1..AC-8) hold, live-verified end-to-end against a real,
disposable FalkorDB workspace, a real multi-version/multi-actor document chain, and both REST and
MCP transports. The ML note's named v1 known-gap (title+content rewrite escaping both candidate
signals) was reproduced live exactly as documented — accepted behavior, not a defect. Two real
defects were found and are reported above, both scoped to the suggested tier's title-fuzzy
candidate-generation signal specifically (Defect 1, High — a RediSearch-metacharacter crash that
silently defeats AC-2 for common real-world titles; Defect 2, Medium — an implicit-AND
token-combination effectiveness gap in the same signal) — neither falsifies any AC in the general
case (the LSH-band signal and the auto tier are both unaffected), but Defect 1 in particular is
worth a fast follow given how ordinary the triggering titles are.

## Artifacts

- Test plan: `falkor-chat/docs/test-plans/document-ingestion2.md`
- This report: `falkor-chat/docs/test-reports/document-ingestion2-report.md`
- Scratch harness (not part of the permanent test suite, kept for reference):
  `/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/8a99d39c-05e5-4728-9683-42324e138a1e/scratchpad/di2-qa/`
  (`mcp_call.py` — generic real MCP Streamable-HTTP tool caller; `opencode-local.json` — corrected
  LM Studio `baseURL` copy; `server-run1.log`/`server-run2.log` — full server logs for both actor
  runs).
- Disposable workspace `ws:docingest2qa` left in place (not torn down — no shared-state risk,
  available for follow-up inspection; contains this pass's own fixture data only).
