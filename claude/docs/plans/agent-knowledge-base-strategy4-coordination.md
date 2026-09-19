# Agent knowledge-base strategy — Track 2 implementation coordination (Stages 6-9)

> **Status:** active · **Owner:** `teco` · **Tracks:** K-030 (`claude/cobb/kaizen/plan.md`)

Implements `claude/docs/plans/agent-knowledge-base-strategy.md` §3 "Track 2 — distilled-knowledge
ingestion (FR-2–FR-7), after Track 1" (Stages 6-9). Track 1 (Stages 1-5, raw-capture migration)
is fully delivered and closed — `claude/docs/plans/agent-knowledge-base-strategy3-coordination.md`
(archived). This is a fresh coordination doc, not a continuation ledger of that one; unit ids
below restart at U1 and are scoped to this document only. Stage numbers (6-9) are the parent
plan's own identifiers and stay globally unique across the whole K-030 effort.

**Blocking discovery made before any Track 2 unit was dispatched (2026-09-18):** the LM Studio
embedding backend behind `ws:agent-team`'s dedicated falkor-chat process (port 8200,
`falkor-chat-agent-team`) is unreachable at its currently-configured address
(`http://192.168.0.69:1234` — confirmed unreachable, `curl` times out). `http://localhost:1234`
answers 200 from this WSL2 session, consistent with the standing WSL2↔LM Studio mirrored-networking
note. This is not a caveat to route around: the parent plan's own §7 test strategy requires a live
`ingest_document` → `search_documents` smoke round-trip **before** Stage 6's bulk migration, and
Stage 7's retrieval-convention skill and Stage 8's entire golden-set evaluation both need working
`search_documents` to exist at all. Treated as a real environment-blocker unit (U1 below), not a
caveat noted and worked around.

## Ledger

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| U1 (LM Studio backend fix + durability hardening) | `devops` | `a9b89bff4200d229c` | accepted | `falkor-chat/scripts/start_agent_team.sh` (3-tier config fallback), commit `d5ba1d2`; reachability fix live, `opencode.local.json` outside repo | skipped, trivial/low-risk infra fix (§Guardrails) — independently re-verified by teco (diff scope, live health, cold-restart WARNING branch reasoning, `ws:agent-team` doc states) instead | 147.7k+176.5k tok / 90 tools / 924s |
| U2 (Stage 6 candidate-flagging tool) | `coder` | `a2e077f01efe309ac` | accepted | `claude/cobb/scripts/flag_split_candidates.py` (+ selftest + fixture), commit `a909032` | skipped, trivial/low-risk (§Guardrails) — independently re-run and known-answer-verified by teco instead | 213.1k tok / 43 tools / 896s |
| U3 (Stage 6 migration pass, minus Step 4) | `cobb` | `a16e315007fa77211` | accepted-checkpoint (closed — no further resumes of this agentId) | `claude/cobb/scripts/kb-claim-manifest.json` (9/13 files fully done, 134 claims; `falkordb-quirks.md` 4/5 headings done, 67 claims; heading 5 stopped 3/13 bullets in; `review-techniques.md` not started) — 206 ready/31 failed in `ws:agent-team`, teco-reverified exact match | n/a — checkpoint, not a final deliverable; remainder continues under U3f+ as fresh dispatches | 792.7k tok / 484 tools / 1659s (final) |
| U3f (finish `falkordb-quirks.md` + migrate `review-techniques.md` headings 1-11) | `cobb` | `afea468968adff31e` | accepted-checkpoint (closed — no further resumes; already 295.6k tok in one cycle) | `falkordb-quirks.md` complete (**86** claims — corrected 2026-09-19 per U3k's finding, see Notes; was misreported as 83, teco-reverified exact against the manifest's own nested documentId count; 236 ready/31 failed teco-reverified exact at the time); `review-techniques.md` 11/54 headings done (14 claims); qa-testing-techniques.md "and"-drop fixed, teco-reverified byte-exact via `substring()`; lm-studio-model-notes.md 2nd finding judged false-positive (legit split, reasoning recorded) | n/a — checkpoint; remainder continues under U3g (fresh) | 295.6k tok / 83 tools / 1309s (final) |
| U3g (continue `review-techniques.md` from heading 12, fresh dispatch, checkpoint tightened to ~80 tool calls) | `cobb` | `abb03228b0495e968` | accepted-checkpoint (closed — no further resumes of this agentId) | `review-techniques.md` 32/54 headings done (44 claims: 14 prior + 30 this dispatch), all byte-exact, zero `Document.status:"failed"`; heading 33 (~2170w, the other flagged giant) fully read and split into 11 claims, not yet ingested — see manifest `_resume_from_heading_33` | n/a — checkpoint; remainder continues under U3h (fresh) | 283.3k tok / 65 tools / 1740s (final) |
| U3h (finish `review-techniques.md` from heading 33 through 54) | `cobb` | `ab5c286d2579b4574` | accepted | `review-techniques.md` COMPLETE: 54/54 headings, 81 claims (14 U3f + 30 U3g + 37 this dispatch), all byte-exact, zero `Document.status:"failed"`. **All 13 KB files now attempted** — 303 ready/31 failed in `ws:agent-team`, teco-reverified exact match (266+37) | n/a — Stage 6's own diff-scoped `analyst` gate covers the whole migration once U3i closes, not per-dispatch | 293.4k tok / 90 tools / 26961s |
| U3i (final low-contention single-item retry pass over the 31 deferred-failed documents) | `cobb` | `a80ef622846074815` | accepted | manifest updated with final documentIds/status for all 31 (`claude/cobb/scripts/kb-claim-manifest.json`); 26/31 resolved to `ready`, 5/31 permanently `Document.status:"failed"` after the one retry (byte-exact content confirmed on all 5). **teco-reverified 2026-09-19 against `ws:agent-team` directly (`mcp__cypher__query`) and against the manifest's own documentId inventory: the delegate's self-reported "308 documents ready" aggregate in its hand-back/kaizen entries was an arithmetic error — actual corpus (all 332 manifest-tracked KB claims across the 13 files): 327 `ready` + 5 permanently `failed` = 332. (`ws:agent-team`'s grand total is 334 = 332 KB claims + 2 unrelated pre-existing raw-learning documents from the teco/cobb kaizen-pilot, both `ready`, correctly outside this manifest and outside Stage 6's scope.) 303 (U3h) + 26 (this dispatch's successes) = 329 documents transitioned to `ready` across Stage 6's full run, cross-checks cleanly.** Correction dispatched back to `cobb` to fix the "308" figure at its source (kaizen `history.md`/`plan.md`, the manifest's `_status` field) — see Notes below. | n/a — folds into Stage 6's overall gate, which this closes | 255.7k tok / 157 tools / ~8788s |
| U3e (Stage 6 Step 4 — content-loss checker + mutation test) | `coder`+`analyst` | `a4e15ab1f754defab` / `ad3ac3714fa9519c4` | **accepted, closed** | `claude/cobb/scripts/check_content_loss.py` (+selftest, 16 checks), commit `a2abc99`; review Pass 1+2, commit `fe53332` | `analyst` → **approve** (Pass 2, all 6 Pass-1 findings independently reverified by execution, no new findings) | coder 235.1k+302.7k tok/105 tools; analyst 124.7k+175.8k tok/38 tools |
| U3b (diagnose embedding-backend throughput under sustained load; unblock U3's retry decision) | `devops` | `a98ff226c19d0968e` | accepted | root cause: transient LM Studio crash of extractor model `qwen3-4b-2507` under cross-model contention (14 extract failures, 0 embed failures — teco-reverified via log grep) — recommends retry now, smaller batches. Diagnosis only, no server/process action taken | skipped, diagnostic finding independently re-verified by teco against raw log + graph state instead | 112.4k tok / 23 tools / 243s |
| U3c (independent read: push-through vs. fix-pipeline-first, unfiltered from teco's/user's own view) | `devops` | `abd3fbd98d3a5f850` | accepted | recommends (a) push through + defer, matching the user's independently-made choice; flags a concrete, cheap-to-check lead (concurrent `model-bench` LM Studio sweeps as the contention source) and recommends filing `background.py` retry/backoff as a separate backlog item rather than a live mid-migration change | skipped, advisory-only opinion, cross-checked against user's independent answer instead | 57.3k tok / 1 tool / 57s |
| U3d (file the deferred `background.py` extract-job retry/backoff hardening as a `falkor-chat` backlog item) | `devops` | `a3f5fdea8dff4a9c7` | accepted | `falkor-chat/docs/BACKLOG.md` K-067, commit `5743ab1` | skipped, trivial single-file backlog write following existing convention exactly — teco-verified via diff before commit | 64.6k tok / 4 tools / 65s |
| U3j (correct the "308" arithmetic error at its source in cobb's own kaizen files/manifest `_status`) | `cobb` | `aea51f90e1e7d33d0` | accepted | `claude/cobb/kaizen/history.md`, `claude/cobb/kaizen/plan.md` (2 spots), `claude/cobb/scripts/kb-claim-manifest.json` `_status` field, all corrected to 327 ready/5 failed/332 total; JSON re-validated; grep-swept `claude/cobb/`/`claude/docs/` for any other "308 documents" occurrence, none found | skipped, purely mechanical text correction with numbers pre-supplied by teco's own re-verification (§Guardrails, trivial/low-risk) — teco re-grepped and re-validated JSON before commit instead | 103.6k tok / 22 tools / 837s |
| U3k (Stage 6 overall migration review — diff-scoped, whole corpus) | `analyst` | `adcc61ffdd4e9e570` | accepted | `claude/docs/reviews/agent-knowledge-base-strategy4-stage6.md` | self → **approve with suggestions** — corpus tally/attribution/split-boundary/permanent-failure-gap independently re-derived by execution and teco-spot-checked afterward (manifest recount, `INGESTED_BY`/`SUPERSEDES`/title-uniqueness queries all matched); 2 non-blocking findings (Major: `check_content_loss.py` false-positive mode on the shared-duplicated-label split shape + `review-techniques.md`/`qa-testing-techniques.md`'s AC-4/AC-5 obligation only now closed by manual reconciliation; Minor: `falkordb-quirks.md` claim count mis-recorded 83→should be 86) routed to U3l | 190.5k tok / 82 tools / 1937s |
| U3l (fix U3k's two findings: falkordb-quirks.md count correction + check_content_loss.py LIMITATIONS doc + AC-4/AC-5 closure note) | `cobb` | `afc9b730600d9aaf9` | in-flight | — | n/a — teco-verified directly, not a Stage-7 blocker | — |
| U4 (Stage 7 retrieval-convention skill) | `cobb` | `acdb6bcd73d53b8d2` | accepted | `skills/agent-kb-retrieval/SKILL.md` (new); `claude/scripts/audit-team.sh` check 11; one-line pointer in 9 agent prompts; `skills/README.md` + `claude/AGENTS.md` catalog updates; kaizen `history.md`/`plan.md` — all teco-reverified: SKILL.md content read directly and cross-checked against `ws:agent-team` (the "family-slug — claim-title" split-sibling title claim spot-checked live, holds); check 11 independently mutation-tested (corrupt → FAIL, byte-identical restore → PASS); full `audit-team.sh` run shows only the 5 pre-existing, unrelated FAILs (none of this dispatch's files); all 9 pointer lines present exactly once, `devops.md`'s flagged fix reads clean; symlink deployment confirmed live | `analyst` → — | 219.7k tok / 69 tools / 2043s |
| U4a (Stage 7 review — diff-scoped) | `analyst` | `ad88bd61f047fd085` | accepted | `claude/docs/reviews/agent-knowledge-base-strategy4-stage7.md` | self → **needs changes** — 1 blocker (teco-reverified directly against frontmatter): the retrieval pointer went to 9 agents but only 5 inherit every tool by default; `teco`/`architect`/`analyst`/`data-scientist`'s explicit `tools:` allowlists never gained `search_documents`/`get_document`, so the pointer is silently inert for them (data-scientist is U5's own owner — Stage 8 was about to hit this wall). Everything else (prefix byte-exactness, score floor discipline, check 11, pointer placement, Step 0 reasoning, README.md open question) holds, teco-spot-checked. Routed to U4b | 168.0k tok / 30 tools / 2179s |
| U4b (fix U4a's blocker: add the 2 MCP tools to 4 agents' `tools:` allowlists) | `cobb` | `a04f6f292c8ee11a0` | accepted | `teco.md`/`architect.md`/`analyst.md`/`data-scientist.md` `tools:` lines fixed, commit `ffd1809`; teco-reverified frontmatter content directly + re-ran `audit-team.sh` (same 5 pre-existing FAILs only) | n/a — closed by teco's own live tool-visibility probe: a fresh `data-scientist` spawn (agentId `a8aca814a2ca4a35e`) confirmed `search_documents`/`get_document` both present and accessible, per the review's own suggested closing method | 94.3k tok / 21 tools / 330s |
| U5 (Stage 8 golden-set design + pilot calibration) | `data-scientist` | `a54a41942db93aa4f` | accepted | `claude/docs/plans/agent-knowledge-base-strategy-ml.md` "Stage 8 Phase 1" section (45-pair golden-set design, all 13 KBs, 5 strata a-e; 16-pair pilot executed live); `skills/agent-kb-retrieval/SKILL.md` floor section updated (0.42 cosine distance, no longer provisional/disabled) — teco-reverified: Wilson CI [0.529,0.978] recomputed by hand from n=8/x=7 and matches exactly; 5 documentIds spot-checked live against `ws:agent-team`, all real/ready/title-matching; pilot's 16 ✓-tagged design rows cross-counted against the reported 8 single-answer + 4 families(8 docs) + 4 negatives = 16, consistent; floor-derivation arithmetic (0.409/0.446/0.42 margins) correct; `audit-team.sh` re-run clean (same 5 pre-existing FAILs, check 11 still passes) | `analyst` → — | 231.5k tok / 81 tools / 1104s |
| U5a (Stage 8 Phase 1 review — diff-scoped) | `analyst` | `a108e34078a14a3ba` | accepted | `claude/docs/reviews/agent-knowledge-base-strategy4-stage8-phase1.md` | self → **approve with suggestions** — Wilson CI/MRR independently re-derived (different method), 11 disjoint documentIds spot-checked, R3/h39 crowding-out + 2 negative queries live-reran and reproduced exactly; no blockers. 2 Minors: a prose inaccuracy (ml.md:417-418's "6 plus one held in reserve" — table actually uses all 7 named families), R9's query paraphrases its target's title too closely (explicitly deferred, no rerun needed now). Prose fix routed to U5b | 101.2k tok / 23 tools / 318s |
| U5b (fix U5a's Minor: ml.md:417-418 prose correction) | `data-scientist` | `a060769a2e0655dd4` | accepted | `claude/docs/plans/agent-knowledge-base-strategy-ml.md`:417-418 corrected, commit `7342f9e` — teco-reverified exact text matches the review's suggested replacement | n/a — mechanical, exact replacement text supplied by the review | 37.4k tok / 4 tools / 30s |
| U6 (Stage 8 Phase 2 — full ~45-pair AC-2 regression gate) | `qa-engineer` | `a67e8c1654cd18e1d` | **gated (needs changes)** | `claude/docs/test-plans/agent-knowledge-base-strategy-ac2.md`, `claude/docs/test-reports/agent-knowledge-base-strategy-ac2-report.md` (commit `f8558f2`); `skills/agent-kb-retrieval/SKILL.md` floor re-derived 0.42→0.43 (commit `658e29b`) — teco's own pass verified Wilson CI/MRR arithmetic (matched exactly against the report's own stated inputs), 5 documentIds' existence/status/topical fit, `SKILL.md` diff hygiene, and `audit-team.sh`. **Correction (2026-09-19): teco's arithmetic check validated the *math* against the report's *stated* 0.4201/0.446 inputs, not the 0.4201 figure's own truth against the live tool — U6a's live reproduction (below) found that input itself does not reproduce.** The shipped 0.43 floor and the "0/41 wrongly dropped" claim are not yet trustworthy — see U6a/U6b. | `analyst` → **needs changes** (1 blocker, 2 majors, see U6a) | 227.2k tok / 51 tools / 1096s |
| U6a (Stage 8 Phase 2 review — diff-scoped, live-reproduction of pivotal claims) | `analyst` | `acf9ea5b7873e729a` | accepted | `claude/docs/reviews/agent-knowledge-base-strategy4-stage8-phase2.md`, commit `1fa49a6` | self → **needs changes** — 1 Blocker: R6/h40's second sibling score (the sole number driving the 0.42→0.43 floor re-derivation) does not reproduce live — reported 0.4201/rank 3, live 0.4405/rank 5 (twice, byte-identical), while 3 other spot-checked rows (F4/N4/R4-h14) matched to the 4th decimal, isolating the discrepancy to this one row; live 0.4405 would flip "0/41 wrongly dropped" at the shipped 0.43 floor. 2 Majors: DEF-1's X1 reproduction cites the wrong documentId (right KB file `test-design-techniques.md`, wrong claim — `31e22f72d811485eaad5a0aadd9c1d30` not T1's `aa72815433924cfea2760062daa22b40`); crowding-out "isolated (1/7)" judgment only checked the 7 designed stratum-(e) rows — analyst found the same duplicate-chunk mechanism live in X1 (non-stratum-e), so the true denominator is uncounted. Routed to U6b | 153.0k tok / 34 tools / 790s |
| U6b (fix U6a's blocker + 2 majors: re-verify R6's score, correct DEF-1's X1 citation, run a pooled crowding-out coverage probe) | `qa-engineer` | `a67e8c1654cd18e1d` | in-flight | — | n/a — same delegate resumed via SendMessage | — |
| U7 (Stage 9 distillation-ingestion hook) | `cobb` | — | queued | — | `analyst` → — | — |

U1 and U2 are independent (no file/state overlap) — dispatched in parallel. U3 depends on both.
U4 depends on U3 (needs real embedded content per plan §3). U5 depends on U3+U4. U6 depends on U5
and full Stage 6 coverage. U7 depends on U6 (plan's own Stage 8→9 sequencing).

## Notes

- KB file inventory for Stage 6 (five/six pre-existing + Stage 0's four interim) is `cobb`'s own
  call to confirm against its catalog at migration time, not asserted here as fact — candidates
  observed in the tree: `claude/analyst/review-techniques.md`,
  `claude/architect/plan-authoring-techniques.md`,
  `claude/data-scientist/{lm-studio-model-notes,statistical-method-techniques}.md`,
  `claude/devops/ops-quirks.md`, `claude/frontend-engineer/frontend-quirks.md`,
  `claude/graph-dba/{falkordb-reference,falkordb-quirks}.md`,
  `claude/qa-engineer/qa-testing-techniques.md`,
  `claude/tdd-engineer/{estimator-test-fixtures,test-design-techniques,guard-testing-techniques}.md`,
  `claude/teco/coordination-techniques.md`.
- Plan closing item (b): `cobb` confirms, as part of U4, that `skills/agent-kb-retrieval/` is the
  right home (vs. folding into an existing agent's KB) and that `scripts/audit-team.sh` is the
  right place for the drift check (vs. a standalone script) — `cobb`'s call, not pre-decided here.
- Plan closing item (c): U7's implementer resolves the claim→`documentId` tracking mechanism
  (manifest vs. `list_documents` scan) — not resolved here.
- **Checkpoint 2026-09-18/19 (teco-initiated, context-size, not a migration problem).** `cobb`'s
  session was stopped mid-file at ~800k tokens and handed back immediately per explicit
  instruction, with no further migration attempted first. Exact state, verified in the manifest:
  **9/13 files fully done** (`plan-authoring-techniques.md`, `statistical-method-techniques.md`,
  `test-design-techniques.md`, `falkordb-reference.md`, `frontend-quirks.md`,
  `qa-testing-techniques.md`, `estimator-test-fixtures.md`, `coordination-techniques.md`,
  `lm-studio-model-notes.md` — 134 claims, zero embedding failures). `graph-dba/falkordb-quirks.md`
  **partially done**: 4 of 5 headings complete (67 claims — Indexing/DDL 16, Concurrency 1, Cypher
  dialect 39, Query tuning 11 — all byte-exact, zero failures), heading 5 "Ops, config & tooling"
  stopped after 3 of its 13 bullets; the manifest's `_note4` records each remaining bullet's exact
  line number and one-line summary for a clean resume. `analyst/review-techniques.md` **not
  started at all** (54 sections, 26 flagged — the densest file per the plan's own
  characterization). **Still owed once migration completes:** the single low-contention
  one-item-at-a-time final retry pass across every document still `Document.status:"failed"` —
  currently known: `ops-quirks.md`'s 12 + `guard-testing-techniques.md`'s 19 = 31, likely plus
  whatever recurs finishing the remaining work. Content-loss checker is U3e's (`coder`), not part
  of this unit's remaining scope. A fresh dispatch should resume directly from
  `claude/cobb/scripts/kb-claim-manifest.json`, not re-read `falkordb-quirks.md` cold.
- **U3 paused mid-run 2026-09-18, not delivered.** Confirmed file list (13, matching this doc's own
  candidate list above exactly): 4 files fully migrated and byte-exact verified
  (`architect/plan-authoring-techniques.md`, `data-scientist/statistical-method-techniques.md`,
  `tdd-engineer/test-design-techniques.md`, `graph-dba/falkordb-reference.md`); `devops/
  ops-quirks.md` (12 claims) ingested and byte-exact but stuck at `Document.status:"failed"` on
  retrieval — checked twice with a delay, not transient, and not explained by "batch calls
  categorically fail" (three earlier, comparably- or larger-sized `ingest_documents` batches this
  same run succeeded, and an immediate single-item control probe succeeded too);
  `frontend-engineer/frontend-quirks.md` analyzed (split boundaries decided) but not yet ingested;
  7 files not started; the content-loss-check script not built. Paused per the migration brief's
  own explicit stop condition on the `"failed"` signal rather than guessed past. Open question for
  whoever resumes U3 (`cobb` or a fresh dispatch): retry the 12 stuck `ops-quirks.md` documents now
  (delete-then-recreate, per §4.5) vs. wait for `devops` to confirm the embedding backend's
  throughput under sustained load first — not `cobb`'s call to make unilaterally under this brief.
  Full detail: `claude/cobb/kaizen/history.md`, 2026-09-18 entry. Also resolved as part of this
  unit, ahead of the pause: Step 1 (pre-migration smoke test) passed; Step 1b (the stuck Track 1
  pilot document, `e3ddf8bccbf34454876431f09d2a2482`) recovered and re-ingested successfully, new
  id `4da25923b0214ed0ba20a91dd2132c50`, no classifier block encountered on the delete.
- **Update 2026-09-18 (same day, after `teco` routed the open question to `devops`).** Root cause
  confirmed: a transient LM Studio crash of the extraction-role model, not the embedder — see
  `teco`'s own message in the thread and `claude/cobb/kaizen/history.md` for the full diagnosis.
  `cobb` retried the 12 `ops-quirks.md` documents (delete-then-recreate, 3 sub-batches of 4 per
  the diagnosis's caution) — **all 12 reproduced the identical `Document.status:"failed"`
  signature**, byte-exact text otherwise. Stopped per the diagnosis's own explicit "don't
  blind-retry a third time" condition rather than attempting a third pass. U3 is paused a second
  time on this one file's retrieval status; `frontend-quirks.md` and the 7 not-started files are
  untouched.
- **Update 2026-09-18 (third pass, after `teco` resolved the open question and resumed the
  unit).** `cobb` completed 4 more files cleanly (`frontend-quirks.md`, `qa-testing-techniques.md`,
  `estimator-test-fixtures.md`, `coordination-techniques.md` — 68 claims, zero embedding failures
  except 2 isolated retries on `frontend-quirks.md` that succeeded on individual retry), bringing
  the clean total to 8/13 files, 117 claims. Then `tdd-engineer/guard-testing-techniques.md`'s
  entire 19-claim ingest (8 headings, several genuine multi-way splits, the densest single heading
  in this corpus split 8 ways) came back `Document.status:"failed"` on **all 19** — a total
  wipeout, not a partial one, and larger than either prior recurrence. This happened immediately
  after `coordination-techniques.md`'s 39-claim, 10-batch ingest had succeeded with zero failures
  moments earlier, so `cobb`'s own request pacing does not obviously explain the difference.
  Stopped per `teco`'s own pre-authorized "same stop-and-report instruction... if you hit another
  genuine fork" — not retried. `analyst/review-techniques.md`, `data-scientist/
  lm-studio-model-notes.md`, and `graph-dba/falkordb-quirks.md` remain not started; the
  content-loss-check script (Step 4) not yet built.
- **Update 2026-09-19 (U3f, fresh dispatch resumed from the 2026-09-18 context-size checkpoint).**
  Resumed purely from the manifest's `_note4` — no cold re-read of `falkordb-quirks.md` needed.
  Finished heading 5's remaining 10 bullets (16 claims, 2 bullets required a fresh split-boundary
  judgment call), completing `falkordb-quirks.md` in full (**86** claims — corrected 2026-09-19,
  was misreported as 83 at the time; see U3k/U3l — all byte-exact, zero failures at ingest-time
  check). Started `analyst/review-techniques.md`: re-ran
  `flag_split_candidates.py` fresh (25/54 flagged, refined from the earlier 26 estimate), confirmed
  all 54 heading line numbers, migrated headings 1-11 (14 claims, including a genuine 4-way split
  on heading 2's labeled `(a)`/`(b)`/`(c)`/`(d)` enumeration), all byte-exact. **Stopped at the
  mandatory ~100-tool-call self-checkpoint** — a clean per-heading boundary, heading 12 not
  started. Exact resume point and split-judgment notes through heading 27 are in the manifest's
  `_analyst_review_techniques_IN_PROGRESS` block. **Mid-run, `teco` relayed two findings from
  `coder`'s new `check_content_loss.py` (U3e) run against the already-migrated corpus:** (1) a
  real dropped-word defect in `qa-testing-techniques.md`'s model-bench-attest/run split, fixed via
  delete-then-recreate (new id `1034d23fe5b04503b90c53128e7cfb17`), byte-exact re-verified; (2)
  `lm-studio-model-notes.md`'s heading-5 split flagged as 3 NOT_FOUND + 1 large UNACCOUNTED gap —
  assessed via full word-by-word reconciliation against the live source and found to be a false
  positive (a legitimate reorganization: one sentence correctly relocated across the claim
  boundary, plus a joint numbered-list paragraph split into two independent labels, dropping only
  structural numerals/framing, no facts). Not re-ingested. Both detailed in
  `claude/cobb/kaizen/history.md`, 2026-09-19. **Still owed:** `review-techniques.md` headings
  12-54; the final low-contention retry pass (unchanged scope, still not started).
- **Update 2026-09-19 (U3g, fresh dispatch resumed from the tightened-checkpoint instruction).**
  Migrated headings 12-32 of `review-techniques.md` (30 claims), all byte-exact verified via
  `get_document` immediately after ingest, **zero** instances of the `Document.status:"failed"`
  signature this run. Notable split calls made fresh (no near-identical prior precedent):
  heading 14's "this already exists" claim split along its own Origin's `(1)(2)` vs `(3)(4)`
  numbered examples; heading 22's two numbered traps split as genuinely distinct failure modes
  with the shared Origin duplicated in both (it doesn't split by trap); heading 24 kept **whole**
  despite 3 bolded sub-paragraphs because all three share one single Origin citation (contrast
  with the split cases, which each had per-item distinct origins); heading 26 split into the main
  hashing technique plus an explicitly-labeled, unrelated "companion trap" finding; heading 27 (the
  first big ~1287w flagged giant) turned out to have **more** structure than the prior checkpoint's
  guess of "2 sub-techniques" — consolidated to 4 claims (deletion-flavour, AST-flavour bundling
  3 paragraphs about the same worked example, an explicitly-labeled "Third flavour", and a general
  gating-discipline capstone that explicitly extends "the same move" to a shell-harness example
  rather than being a 5th flavour). **Stopped at the tightened ~80-tool-call self-checkpoint** —
  heading 33 (the *other* ~2170w flagged giant, "A grep-pinned edit table is an edit list, not a
  completeness proof") was fully read and split-decided (9 independently-attributed numbered items
  + a "two derived checks" claim + one claim bundling 3 trailing re-deriving-caveat paragraphs =
  11 claims planned) but **not yet ingested** — the next dispatch can go straight to ingest. Full
  per-heading documentId breakdown and the heading-33 split plan are in the manifest's
  `_analyst_review_techniques_IN_PROGRESS` block (`headings_12_32_done_2026_09_19` and
  `_resume_from_heading_33`). **Still owed:** `review-techniques.md` headings 33-54; the final
  low-contention retry pass (unchanged scope, still not started).
- **Update 2026-09-19 (U3h, fresh dispatch resumed from U3g's checkpoint — the heading-33 split
  plan, fully worked out in advance, needed no re-reading or re-deciding).** Ingested heading 33's
  11 already-decided claims exactly as planned (9 numbered items, the "two derived checks" claim,
  and the bundled trailing re-deriving-caveats claim), then continued fresh through headings 34-54
  (26 more claims across 21 headings, 9 of them flagged and split, the rest migrated whole).
  Notable split calls: h35 (delegate-refusal-retirement vs. diff-collected-test-IDs, two
  independently re-derived instruments); h37 (the module-wide-walk case vs. the CPG-provenance-
  stamp shell incident, one shared Origin across two narrative beats, kept as a pair); h39 (the
  main revision-history-hashing technique vs. an explicitly-labeled "header's blanket claim"
  companion with its own separate, later-dated Origin); h40 (two remedy shapes, each attributed to
  its own named Pass of the same four-pass gate — same shape as h13/h21's precedent); h45 (sha-
  pinning every review count vs. an explicitly-labeled "same hazard applies to a narrative claim"
  companion with its own separate Origin); h54 kept **whole** (two numbered mutants share one
  Origin covering both, unlike h13/h21/h40's per-item-Origin split cases). **Zero** instances of the
  `Document.status:"failed"` signature this entire dispatch — all 37 new claims byte-exact verified
  via `get_document` immediately after ingest, status `ready` (not `processing`) at every check.
  `review-techniques.md` closes at **81 claims across all 54 headings**, and this was the **last of
  the 13 KB files** — Stage 6's content migration is now complete for the whole corpus. Manifest
  updated: `_analyst_review_techniques_IN_PROGRESS` renamed to `_analyst_review_techniques_DONE`
  with the full `headings_33_54_done_2026_09_19` documentId/reasoning block, top-level `_status`
  rewritten to reflect all 13 files attempted, `_remainingFiles` cleared, `_stillOwedAfterMigration`
  tightened to name the final, closed count (31 documents). **Still owed, and now the only
  remaining unit for the whole Stage 6 effort:** the final low-contention one-item-at-a-time retry
  pass over `ops-quirks.md`'s 12 + `guard-testing-techniques.md`'s 19 still-`Document.status:
  "failed"` documents — per the dispatching brief, this is the trigger to dispatch that unit now
  that all 13 files have been attempted. Team coherence certification is unaffected (no
  agent/skill roster or prompt changed this dispatch — only a KB `.md` file's distilled content was
  migrated to `ws:agent-team`, and the manifest/kaizen/coordination bookkeeping that tracks it).
- **Update 2026-09-19 (U3i, the final retry pass — Stage 6 is now FULLY CLOSED).** Pre-flight
  backend check first: `localhost:1234` responded 200, no `model-bench` LM Studio sweeps running,
  and a throwaway sanity ingest/verify/delete cycle round-tripped cleanly (`processing` → `ready`
  in ~20s). Ran the single low-contention one-item-at-a-time retry exactly once over all 31
  deferred documents: read each existing (byte-exact, `failed`) document's text via `get_document`
  first (31 parallel pure reads, no embedding-pipeline contention), then serially
  delete-then-re-ingest each with identical text and `produced_by='cobb'`, one `ingest_document`
  call per document, no batching. Re-checked status after a real delay (several minutes of other
  work in between, not an instant re-check). **Result: 26/31 resolved to `Document.status:"ready"`.
  5/31 reproduced the identical `"failed"` signature a second time, even under confirmed
  low-contention conditions** — re-verified a second time after further delay to rule out a timing
  artifact, still `failed`. All 5 confirmed byte-exact in content at the final check; only
  retrievability is affected. Per the brief's explicit "genuinely once" instruction, none of the 5
  were retried a third time — they are now a **closed, permanent, named gap**: 4 in
  `devops/ops-quirks.md` (`docker run` stdout-clean; cached-`docker build` `FROM`-metadata round
  trip; PID-1-bare-interpreter `SIGTERM`; interrupted-`docker-build`-resumes) and 1 in
  `tdd-engineer/guard-testing-techniques.md` (the coverage-probe-silence-criterion claim). No
  `[Irreversible Deletion]` classifier block encountered on any of the 31 deletes. **Final corpus
  tally, all 13 KB files' migration claims (manifest-tracked): 327 documents `ready`, 5 permanently
  `failed` (332 total claims).** *Correction 2026-09-19 (teco): cobb's own hand-back and kaizen
  entries stated this as "308 ready" — independently re-verified against `ws:agent-team` directly
  (329 `ready` + 5 `failed` = 334 documents total in the workspace) and against the manifest's own
  documentId inventory (332 distinct claim ids, all 5 permanently-failed ids among them → 327
  ready); the 2-document gap between 329 (workspace-wide) and 327 (manifest-scoped) is exactly the
  2 unrelated, pre-existing raw-learning documents from the teco/cobb kaizen-pilot (not KB-content
  claims, correctly untracked by this manifest) — both `ready`, ids `4da25923b0214ed0ba20a91dd
  2132c50` and `d1f94b35119a400bbe166dd5e9229f44`. "308" does not match either framing and was a
  plain arithmetic slip; corrected here and routed back to `cobb` to fix at its source (kaizen
  `history.md`/`plan.md`, the manifest's own `_status` field).* Manifest,
  `claude/cobb/kaizen/history.md` (new dated entry), and `claude/cobb/kaizen/plan.md` (K-030
  rewritten in place) all updated to record this closure — **the "308" figure in those three
  locations is stale pending cobb's correction dispatch (see U3i row above).** **Stage 6 (K-030
  Track 2's first stage) is now fully closed — this triggers Stage 7 per this document's own
  dependency chain** (`U4 depends on U3`, and U3's whole chain — U3 through U3i — is what "full
  Stage 6 coverage" refers to). Team coherence certification unaffected (no agent/skill roster or
  prompt changed; only `ws:agent-team` document state and manifest/kaizen/coordination
  bookkeeping).

- **Update 2026-09-19 (U3j — "308" correction, and Stage 6's artifacts committed).** U3j (fresh
  `cobb` dispatch, haiku, mechanical) fixed the "308" figure at its source in
  `claude/cobb/kaizen/history.md`, `plan.md` (2 spots), and the manifest's `_status` field to the
  corrected 327 ready/5 failed/332 total; re-validated the manifest as well-formed JSON afterward.
  teco grep-swept `claude/cobb/` and `claude/docs/` post-dispatch and confirmed no remaining
  "308 documents" occurrence (the only "308" hits left are this correction's own explanatory prose
  and an unrelated `C-308` backlog ticket ID). **Stage 6's full artifact set is now committed** —
  `a1eb03d` (`kb-claim-manifest.json` + this coordination doc) and `62824af` (cobb's
  `kaizen/history.md` + `kaizen/plan.md` correction) — `git status` confirmed clean of Stage 6
  files afterward, only the unrelated concurrent `model-bench/` session's diffs remain untouched.
  **Next: U3k, a diff-scoped `analyst` review of the whole Stage 6 migration** (content-fidelity
  spot-checks, the 5-document permanent-failure gap's documentation, manifest/kaizen consistency)
  before Stage 7 (U4) opens — per the standing plan recorded in U3h/U3i's own rows ("Stage 6's own
  diff-scoped analyst gate covers the whole migration once U3i closes, not per-dispatch").

- **Update 2026-09-19 (U6 — Stage 8 Phase 2, the full AC-2 gate, delivered and teco-verified).**
  `qa-engineer` executed all 29 design-only golden-set rows live against `ws:agent-team` and pooled
  with Phase 1's 16: pooled recall@5 0.875 [Wilson 0.719,0.950] (point estimate unchanged from
  Phase 1, CI tightened as designed), code/prose split now statistically distinguishable (code
  11/11=1.000 vs. prose 19/15=0.789), stratum-e pooled set-recall 13/14=0.929. Two real findings:
  (1) the 0.42 floor did not survive out-of-sample validation — a genuine true positive (R6/h40's
  second sibling) scored 0.4201, over the floor by 0.0001 — re-derived to **0.43** (landed in
  `SKILL.md`, commit `658e29b`), 0/41 pooled true positives wrongly dropped at the new value; (2)
  the C1 prose-retrieval miss Phase 1 flagged as "unexplained" is confirmed a real, evidenced
  pattern, not a one-off — all 4 misses across the full 45-pair set are `b-prose`-tagged, 0 of 11
  `b-code`-tagged queries missed. The h39 duplicate-chunk crowding-out defect is judged **isolated**
  (0 of the 3 newly-tested stratum-e families reproduce it; 6 of 7 designed families show no
  crowding at all) — recommends against a `graph-dba`/`cobb` de-duplication follow-up on this
  evidence. Deliverables committed `f8558f2` (test plan + report) and `658e29b` (SKILL.md floor).
  **Not yet gated** — `analyst` review dispatched next (U6a); the C1/DEF-1 prose-retrieval gap is
  flagged as the most actionable open follow-up, not this gate's to fix.
