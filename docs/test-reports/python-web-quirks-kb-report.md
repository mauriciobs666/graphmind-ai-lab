# python-web-quirks on the knowledge base — Test Report

> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** — (python-web-quirks-kb, S11)

Report for `docs/test-plans/python-web-quirks-kb.md` (S11 scope: TP-001 to TP-008, plan T31-T38; AC-2, AC-5, AC-7, AC-8, AC-12), executed 2026-11-10 against the test workspace `ws:pwq-test` on `http://localhost:8210` only. Later steps (S14, S15c) append their own sections. Sections land as each T-test completes.

## 1. Summary

- Target: `claude/cobb/scripts/pwq_kb.py` (selftest 145/145 observed at start), copy of record `claude/knowledge/python-web-quirks.md` (85,707 bytes, 36 `<!-- unit` fences), pin `3fcb171cf03065f99aca800d9ea30ff6a2a7ba95`.
- Working manifest: a copy of the S10 scratch manifest under the session scratchpad (`m-work.json`); the repo's `pwq-kb-manifest.json` was never used or written.
- Fences respected: no `:8200`, no `ws:agent-team` access, no `--confirm-production`, no curl DELETE; the committed copy was never edited (all mutants/plants worked on copies in the scratchpad).
- CPG: considered, not relevant — black-box live-workspace checks of a tooling/KB migration; no code-level test-gap question was asked.
- Overall verdict: PASS with one defect (D-1, the T31 last-chunk mutant is not red); see section 10.

Baseline before any test: `pwq_kb_selftest.py` 145/145; `gate --ws pwq-test` clean (36 units, 129 chunks, 129 embedded); `diverge` exit 0 ("diverge: clean").

## 2. TP-001 / T31 (AC-2): per-unit retrieval

Commands (cwd repo root, `M=<scratchpad>/m-work.json`):

```
python3 claude/cobb/scripts/pwq_kb.py eval --ac2 --ws pwq-test --base-url http://localhost:8210 --manifest $M --floor 0.43
python3 claude/cobb/scripts/pwq_kb.py eval --ac2 ... --exclude <the 14 S7 raw-capture ids, one flag each>   # excluded run
```

Observed: `ac2: 36/36 units pass` in both runs (unfiltered and S7-excluded). Each unit: title query, every Set A situation query for its sections (merged units u02, u21, u24, u29, u30 are scored per member section), and the last-paragraph query all return the unit's document among the five chunks at or under 0.43; `get_document(documentId).title` equals the unit's `Gotcha:` (name=ok, 36/36). Gate: 36 units, 129 chunks, 129 embedded.

Independent rank table (my own script over the same REST search; scratchpad `ranks.py`, output `t31_ranks.txt`): title query rank 1 for 36/36 (scores 0.056-0.191); last-paragraph query rank 1 for 34/36, rank 2 for u35 (0.219) and u41 (0.137); `get_document` title equals the `Gotcha:` line 36/36.

**Mutant (plan T31: delete the last chunk of one unit in the test graph; the last-paragraph query must turn red): NOT RED. See defect D-1.**
Procedure: u04 (document `7f0a4437...`, 6 chunks seq 0-5); a Cypher `DETACH DELETE` of the `seq=5` Chunk node in `ws:pwq-test` (scratch script via `cypher-mcp/.venv`), `eval --ac2` re-run: `ac2: 36/36 units pass` (exit 0). I then also deleted seq 4 and 3 (3 of 6 chunks left): still `36/36`. Cause (observed with a direct search of the tail paragraph, 416 chars): the unit's other chunks still score under the floor for that query (seq 0: 0.348, seq 2: 0.3775, seq 1: 0.4064 with chunks 3-5 gone), and the check is document-level ("any chunk of the unit's document in the five"). `gate` also passed (3/3 embedded), since it counts embedded against existing chunks. Restore: `replace --unit u04` (33 s, "replace u04: done"), `gate` clean (u04 6/6), `diverge` clean, `ac2` 36/36 (u04's documentId changed to `1a9eb5d1...` in the working manifest only).
A mutant that does turn T31 red in this design is deleting the unit's whole document (not run: that is the `retire` path, exercised as the retired-not-returned check in section 8).

**Exploratory charter** (five off-wordings and one adjacent library fact; informational, no pass/fail; `t_expl.txt`). Top-5 sources by query (u = unit; "other" = non-unit team document):

| Query | Result |
|---|---|
| E1 NaN in a number field (u12) | u12 rank 1 (0.329), 2 (0.340) |
| E2 whitespace-only form field (u10) | u10 rank 1 (0.412), 2 (0.428) |
| E3 skip a test by absolute path (u36) | two non-unit documents rank 1-2 (0.313, 0.340), u36 rank 3 (0.344), u17 rank 4-5 |
| E4 `request.client.host` behind a proxy (u09) | u09 ranks 1-4 (0.324-0.400) |
| E5 same-second `.pyc` edit (u29) | u29 ranks 1-3 (0.367-0.396) |
| E6 adjacent fact: ThreadPoolExecutor worker count (no unit) | nothing under 0.43 (best u02 at 0.451, u26 0.500): a correct "no gotcha matched" |

No query returned a unit above its own for a different gotcha; E3 shows non-unit documents (raw captures or other team documents) outranking the unit when the wording is off.

## 3. TP-002 / T32 (AC-8): header facts

Derived from the pin (not from the tool): `git show 3fcb171c...:skills/python-web-quirks/SKILL.md` split on `^## ` gives 42 sections; applying the D3 rule with an independent regex (an ISO date within 100 characters after `verified|measured|confirmed|reproduced|re-run|checked|tested`) gives **15 sections with a phrase-qualified date and 27 undated**; 16 sections carry any ISO date (the 16th is the provenance date, correctly not a verification; the requirements say "16 of 42 carry a date").

Observed in the stored documents (`GET /documents/<id>` of the 36 documents, member titles mapped back to pin sections, merged units read per member `Verified[n]`): **15 dated, 27 `unverified, migrated 2026-11-10`, 0 mismatches**; every dated value equals the derived date list (e.g. section 26: `2026-09-08, 2026-09-09`). Versions: 30 `Versions` lines with values, 12 `none stated`; all 57 `<lib> <token>` pairs are body substrings with the library name within 40 characters; the four "none stated" lines whose unit body contains a number-like string are the second member of merged units u02/u24, `0.364` (a recall figure, u14) and `1.0` (a seconds figure, u28), none a library version.

Mutant (scratch copy with u05's `Verified: 2026-09-08` swapped for `unverified, migrated 2026-11-10`): header count of the copy goes 15 dated / 27 unverified to 14 / 28 (mismatch against the derived 15 / 27) and `verify-copy --pin ... --copy <mutant>` exits 1: `FAIL: u05: header mismatch: copy 'Verified: unverified, migrated 2026-11-10' vs recomputed 'Verified: 2026-09-08'`. Control `verify-copy` on the committed copy: `verify-copy: clean`, exit 0.

**TP-002: PASS** (mutant red, control green).

## 4. TP-003 / T33 (AC-8): title listing

`grep -n -E '^(<!-- (unit |preamble)|## )' claude/knowledge/python-web-quirks.md`: **80 lines = 42 `## ` titles + 36 unit fences + 2 preamble fences, 6,619 bytes** (bound 10,000; expected from the pin: 42 headings, 36 units per the plan's D2, 2 preamble fences, so 80). The sorted list of the 42 titles equals the sorted pin headings (empty diff). Normal mode: all 36 documents' titles are reachable by `get_document` (T31, 36/36 `name=ok`).
Mutant (regex without `preamble`): 78 lines, 0 preamble fences, the two preamble fences missing, count no longer 80: red.

**TP-003: PASS.**

## 5. TP-004 / T34 (AC-8): 43rd title

Scratch copy `w_copy.md` = committed copy + a synthetic `<!-- unit u43 -->` (title begins `QA43 synthetic probe gotcha: a quokka-lattice token ...`). Listing on it: 82 lines (80 + the fence + the `## `), 6,759 bytes, `u43` visible in the last two lines. `pwq_kb.py diverge --copy w_copy.md --ws pwq-test --base-url http://localhost:8210 --manifest <m-work.json>`: `MISSING u43 QA43 synthetic probe ...`, `diverge: 1 finding(s)`, **exit 1**. Control (committed copy): `diverge: clean`, exit 0.
Mutant (T23 shape): a scratch copy with the 43rd gotcha as a `###` heading with no fence: the listing regex still returns 80 lines (hidden), but `diverge` exits 1 with `STALE u42` and `REGISTRY-STALE u42` (the hidden lump is absorbed into the last unit's body), so the listing's blind spot is caught by `diverge`. Note the finding names u42, not the hidden heading (the diagnostic points at the neighbour; observation, not a defect against the AC).

**TP-004: PASS.** The 43rd unit is later ingested in section 8 (TP-006) and removed again at restore.

## 6. TP-008 / T38 (AC-12): write guard

Invocation shape read from `claude/cobb/hooks/guard-cobb-topic-writes.sh` (it `exec`s `claude/scripts/guard-doc-writes.sh`): PreToolUse JSON on stdin, `.tool_input.file_path`, output JSON `hookSpecificOutput.permissionDecision` (`allow`/`ask`; no output = pass). Run with `printf '{"tool_input":{"file_path":"<p>"}}' | bash claude/cobb/hooks/guard-cobb-topic-writes.sh`, relative and absolute forms:

| Path | Decision |
|---|---|
| `claude/knowledge/python-web-quirks.md` (rel and abs) | allow |
| `skills/python-web-quirks/SKILL.md` (rel and abs) | ask |
| `docs/BACKLOG.md` (rel and abs) | ask |

Non-comment lines of the guard naming `skills/python-web-quirks`: 0 (the only occurrence is the explanatory comment at lines 82-84, which cites `claude/*/*.md` as covering the new path). Mutant (scratch copy of the guard plus `guard-doc-writes.sh` under the scratchpad, with `skills/python-web-quirks/*|*/skills/python-web-quirks/*` re-added to the glob list): the copy path stays allow, **`skills/python-web-quirks/SKILL.md` becomes allow**, `docs/BACKLOG.md` stays ask: red as the plan requires.

**TP-008: PASS.**

## 7. TP-005 / T35 (AC-5): rebuild from the copy

Working manifest `m-t35.json` (copy of the S10 scratch manifest). Baseline before: checker `--source-format copy --live --ws pwq-test` exit 0 (44 scopes clean).

1. Remove: 36 x `pwq_kb.py retire --ws pwq-test --base-url http://localhost:8210 --manifest m-t35.json --unit <id>`: 36 `retire <id>: done`, manifest left with 0 units; `diverge` then printed `MISSING` for all 36 (`diverge: 36 finding(s)`); a title query for u12 returned only non-unit documents.
2. Rebuild: `pwq_kb.py ingest --ws pwq-test --base-url http://localhost:8210 --manifest m-t35.json --timeout 1800 --poll 10` from the committed copy: 36 `ingest <id> <documentId>` lines, exit 0, **18 min 26 s** wall time (paced, one document at a time).
3. Gate: `gate: clean`, 36 units, 129 chunks, 129 embedded. `diverge: clean`, exit 0.
4. Manifest regenerated: 36 units, `pin`/`migratedOn` equal to before, every unit's `title` and `sha256` equal to before, all 36 `documentId`s new (none reused).
5. Checker: `check_content_loss.py skills/python-web-quirks/SKILL.md --stored-format pwq --manifest m-t35.json --copy claude/knowledge/python-web-quirks.md --source-format copy --live --ws pwq-test`: **exit 0**, 44 `clean: 1 claim(s), fully accounted, no overlaps.` lines.
6. TP-001 rerun: `eval --ac2 ... --floor 0.43`: `ac2: 36/36 units pass`.

Mutant (scratch copy with one body character changed, `whitespace-only` to `whitespaCe-only` in u10; checker run live against the rebuilt state): **exit 1**, `[NOT_FOUND] <u10 documentId>#pydantic ... whitespace-only strings`, with the `- whitespaCe-only / + whitespace-only` diff, `44 scope(s) checked, 1 with findings`. Red as required (this mutant was run before the wipe, against the equivalent pre-wipe state; the rebuilt documents are text-identical, sha256 equal).

**TP-005: PASS.**

## 8. TP-006 / T36 (AC-7): add, edit, retire (`ws:pwq-test` only)

Working copy `w_copy.md` (scratchpad; the committed copy untouched), working manifest `m-work.json` (the rebuilt state of section 7). `B` = `--ws pwq-test --base-url http://localhost:8210 --manifest m-work.json`.

| Step | Command | Observed |
|---|---|---|
| Add | `pwq_kb.py ingest $B --copy w_copy.md` (copy + synthetic `u43`, a fresh never-used id) | 36 existing units `skip`, `u43 ingest a919f438...`, 14 s; `gate`: `u43 1 1 ok`, `gate: clean`; `diverge` clean; title query rank 1 (0.207), last-paragraph query rank 1 (0.281); `get_document` title equals the `Gotcha:` line. A situation-style wording ("a handle from frobnicate_widget returns an empty list on the third read") ranked u43 first at 0.447, **above** the 0.43 floor (informational: a one-chunk synthetic unit). |
| Edit (mutant first) | edit one word in u10's body in `w_copy.md` (`false-safety` to `false-sense-of-safety`), edit NOT ingested: `diverge $B --copy w_copy.md` | **exit 1**, `REGISTRY-STALE u10` and `STALE u10` (mutant red). |
| Edit (plain ingest) | `ingest $B --copy w_copy.md` | `u10 refuse  current document(s) with different text; the stakeholder deletes first: curl -X DELETE http://localhost:8210/documents/1fa5a8c7...`, exit 1: the T18 refusal, the edit cannot go through `ingest`. |
| Edit (apply) | `replace $B --copy w_copy.md --unit u10` | `replace u10: done` (22.6 s); `gate` `u10 2 2 ok`; `diverge` clean (exit 0); stored text has `false-sense-of-safety`, not `false-safety`; title rank 1 (0.065), tail rank 1 (0.121), situation query ranks u10 first (0.413). |
| Retire | `retire $B --copy w_copy.md --unit u12` | `retire u12: done`. With u12 still in the copy: `gate` row `u12 - - ABSENT`, `gate: FAILED` (exit 1) and `diverge` `MISSING u12` (confirms the plan's statement that the unit must also leave the copy). After deleting u12's 1,163-byte block from `w_copy.md`: `gate: clean`, `diverge: clean`, exit 0. |
| Retired not returned | title and situation queries for u12 | no u12 chunk in the top 5 (top: two non-unit chunks, then u14/u02/u37). |

Mutant for the edit (skip the ingest of the edit): `diverge` stale, exit 1, as required. For add (skip the ingest): section 5 shows `MISSING u43`, exit 1. A bonus observation: `diverge` of the committed copy against the end state (u43 added, u10 edited, u12 retired) printed `STALE u10`, `MISSING u12` and `ORPHAN u43`, exit 1, i.e. all three lifecycle states are named.

**TP-006: PASS.**

## 9. TP-007 / T37 (AC-7): planted divergence

Control first, nothing planted (`diverge $B --copy w_copy.md`, the consistent state after section 8): `diverge: clean`, **exit 0**. Plant: a scratch copy `w_plant.md` with one body character changed in u31 (`derivable` to `derivabIe`): `diverge` printed `REGISTRY-STALE u31 A typing.Protocol's own declared methods ...` and `STALE u31 ...`, `diverge: 2 finding(s)`, **exit 1**, naming the unit. The plan's alternative plant (editing a stored document directly) was not used: the REST surface has no edit route, and a direct Cypher write to stored text was out of the allowed operations.

**TP-007: PASS.**

## 10. Restore of `ws:pwq-test`, final gate, defects, coverage, verdict

### Restore (the 36-unit clean state)

`retire $B --copy w_copy.md --unit u43` (done); `replace $B --unit u10` from the **committed** copy (22 s, done); `ingest $B` from the committed copy (re-added u12, `u12 ingest e2e3f587...`, 27 s, other 35 `skip`). Final `gate --ws pwq-test --base-url http://localhost:8210 --manifest m-work.json`, exit 0:

```
unit chunks embedded status
u01 3 3 ok   u02 7 7 ok   u03 3 3 ok   u04 6 6 ok   u05 3 3 ok   u06 2 2 ok
u07 3 3 ok   u08 5 5 ok   u09 4 4 ok   u10 2 2 ok   u11 5 5 ok   u12 2 2 ok
u13 3 3 ok   u14 3 3 ok   u15 4 4 ok   u16 2 2 ok   u17 3 3 ok   u18 3 3 ok
u19 3 3 ok   u20 3 3 ok   u21 9 9 ok   u24 6 6 ok   u26 10 10 ok  u27 3 3 ok
u28 4 4 ok   u29 7 7 ok   u30 4 4 ok   u31 2 2 ok   u32 2 2 ok   u33 2 2 ok
u35 1 1 ok   u36 2 2 ok   u37 3 3 ok   u38 1 1 ok   u41 2 2 ok   u42 2 2 ok
gate: clean
```

36 units, 129 chunks, 129 embedded (the raw output is one row per line; wrapped here). `diverge` clean (exit 0); `eval --ac2` 36/36; checker `--source-format copy --live --ws pwq-test` exit 0; a title-and-id lookup finds 36 current documents (all `ready`) and no `QA43` document; `:8210` health `{"status":"ok"}`. **Note for S12 and any later reader: every unit's `documentId` in `ws:pwq-test` changed during this run** (S10's ids are gone: all 36 were rebuilt in T35; u04, u10, u12 changed again). The current manifest is the scratchpad `m-work.json` (`<scratchpad>/m-work.json`); the S10 scratch manifest in the same directory is now stale (its ids no longer exist). The repo's `pwq-kb-manifest.json` was never created or touched. The committed copy `claude/knowledge/python-web-quirks.md` is unchanged (sha256 prefix `7d0fc38c054beb25` before and after; `git status` shows it clean).

### Results

| TP | T | AC | Result | Mutant / control |
|---|---|---|---|---|
| TP-001 | T31 | AC-2 | PASS (36/36 units, 129/129 chunks embedded) | Mutant "delete last chunk" **NOT red** (D-1); control and a restored run green |
| TP-002 | T32 | AC-8 | PASS (15 dated / 27 unverified, 0 mismatches) | red (verify-copy exit 1, count 14/28) |
| TP-003 | T33 | AC-8 | PASS (80 lines, 6,619 bytes) | red (78 lines, no preamble fences) |
| TP-004 | T34 | AC-8 | PASS (43rd in listing; `diverge` exit 1 until ingested) | red (hidden `###` heading caught: STALE u42, exit 1) |
| TP-005 | T35 | AC-5 | PASS (rebuild 18m26s, gate/diverge/checker clean, ac2 36/36) | red (checker exit 1 on a 1-char corruption) |
| TP-006 | T36 | AC-7 | PASS | red (skipped edit: STALE u10, exit 1) |
| TP-007 | T37 | AC-7 | PASS | control exit 0, plant exit 1 naming u31 |
| TP-008 | T38 | AC-12 | PASS | red (old path allowed in the mutant guard) |

Exploratory charter: section 2 (informational).

### Defects

**D-1 (medium, test-design/oracle weakness, TP-001 / T31): the tail-probe check cannot detect a missing tail chunk.** Repro: in `ws:pwq-test`, `DETACH DELETE` the `seq=5` Chunk of u04's document (6 chunks), then `pwq_kb.py eval --ac2 --ws pwq-test --base-url http://localhost:8210 --manifest <m> --floor 0.43`. Expected (plan T31 mutant): the u04 last-paragraph query turns red. Actual: `ac2: 36/36 units pass`, exit 0; deleting seq 3, 4 and 5 (half the chunks) also stays 36/36. Evidence: with those chunks gone, the tail paragraph (416 chars) still ranks u04's remaining chunks 1-3 at 0.348 / 0.3775 / 0.4064, all under the floor, and `returns()` accepts any chunk of the document; `gate` also stays clean (3/3 embedded of the remaining chunks). Impact: AC-2's "tail" guarantee (ML section 7 risk 1) is only a document-level guarantee, and a partially ingested unit (chunks lost, K-102-style) is invisible to `gate`, `diverge` (which compares document text) and `--ac2`. Recommendation: for the tail query require that a chunk containing the paragraph is among the five (compare chunk text or `seq` of the tail chunk), or have `gate` compare the chunk count with `split_into_chunks` of the stored text. The Set B `tail` flags in S14 rely on the same document-level matching; the S14 report should say so.

No other defects. Observations (not defects): (a) `diverge` names the neighbouring unit (u42) rather than the hidden `###` heading when a fence-less section is appended to the last unit; (b) T36 mutant for edit shows `ingest` cannot apply an edit in the test workspace (it refuses by design, T18) and `replace` is the path; (c) E3 of the exploratory charter shows non-unit documents outranking u36 for an off-wording query (retrieval quality, S12's area).

### Coverage and gaps

Covered: AC-2, AC-5, AC-7, AC-8, AC-12 per the plan's T31-T38, with the mutants named by the plan. Gaps: the plan's T31 mutant is not red (D-1); T33's "normal mode" is evidenced through T31's `get_document` titles rather than a KB-side listing call; the T37 plant is in the scratch copy, not in a stored document; the T35 corrupted-copy mutant was run before the wipe (same state by sha256); TP-009 to TP-015 and S14/S15c items were not run (other steps). The hook mutant in T38 is a scratch copy of the guard; the real guard file was not edited.

### Verdict (S11)

**PASS with one defect (D-1)** on the oracle's sensitivity: all of AC-2 (as the tool measures it), AC-5, AC-7, AC-8 and AC-12 hold; the plan's exit criterion "TP-001 to TP-008 pass with their mutants red, or each failure a reported defect" is met, with the T31 mutant reported as D-1 rather than waived. Planted divergence (u31, exit 1) and the 43rd title (u43, listing + `diverge` exit 1) are shown. `ws:pwq-test` restored to the clean 36-unit state (final gate above).

## S14 — sealed Set B scored once (2026-11-10, production `ws:agent-team` on `:8200`, read-only)

Executes TP-010 / TP-011 (plan S14, ML §1 gate). Aggregates only: no Set B query, answer or per-query line appears here or anywhere in the repo. Set B sha256 `163efb061834345d3d34c72501c96dd11e8c20fdd48bcbd433ba4c617d5fd15c`. CPG: not applicable — documentation/retrieval-evaluation task with no code-level component under test.

### Pre-checks (all observed this run)

- Set B file: `sha256sum` equals the ledger value, mode `-r--r--r--`, outside the repo, `git ls-files | grep -i set-b` empty; no `.scored` sentinel before the run.
- `gate --copy claude/knowledge/python-web-quirks.md --ws agent-team --base-url http://localhost:8200 --manifest claude/cobb/scripts/pwq-kb-manifest.json`: 36 units, every unit's stored chunk count equal to its embedded count, `gate: clean`, exit 0.
- Exclusions: the 17 ids are the 17 `curl -X DELETE` lines of `docs/plans/python-web-quirks-kb-captures.md` §4 (17 distinct); `GET /documents/<id>` returned 404 for each of the 17; `458d329e2ba44e45a5fd82301a47504a` returned 200 and was not excluded. The exclusions are therefore a no-op on the stored state (deleted at H1); they are passed as the plan specifies, and the filtered and unfiltered runs are identical. I did not enumerate the workspace for captures created after S13b; none was named in the brief, so none is excluded.
- `--floor 0.43` equals the value in `skills/agent-kb-retrieval/SKILL.md` (`reject any returned hit with score > 0.43`).
- Seal accounting: the first invocation omitted `--set-b-path` and exited 2 (`Set B path: pass --set-b-path or set PWQ_SET_B`) before reading Set B; it burned nothing. The second invocation scored (exit 0) and wrote `pwq-set-b.json.scored` (`floor 0.43`, sha256 as above, `scoredAt 2026-11-10T15:56:03`). No second scoring was run.

### Command (single scoring)

`pwq_kb.py eval --copy claude/knowledge/python-web-quirks.md --ws agent-team --base-url http://localhost:8200 --manifest claude/cobb/scripts/pwq-kb-manifest.json --set B --set-b-path <sealed path> --sealed-once --expect-sha256 163efb06…fd15c --floor 0.43 --exclude <17 ids>`. Tool header: `floor=0.43 window=50 topk=5`; note printed: `limit=5 deliberately exceeded for this measurement` (50-chunk window, excluded documents' chunks dropped, first five scored).

### Results (headline = filtered run; unfiltered identical on every figure)

| Measure | Observed | Plan threshold (ML §1 item 2 / AC-3) | Result |
|---|---|---|---|
| Positive recall at floor 0.43 (top-5, document-level) | 73 of 84 = 0.8690 | ≥ 0.90 (≥ 76 of 84) | FAIL (3 short) |
| Positive recall raw top-5 (no floor) | 77 of 84 = 0.9167 | informational | — |
| Per-section coverage, floor-applied | 41 of 42 | all 42 | FAIL (one section missed by both its queries) |
| Per-section coverage, raw | 42 of 42 | informational | — |
| Near-domain false accepts (unit documents only) | 3 of 15 (rejection 12/15) | ≥ 12/15 | PASS, at the limit |
| Off-domain false accepts | 0 of 15 | 15/15 | PASS |
| Negative rejection overall | 27 of 30 | ≥ 27 of 30 | PASS, at the limit |
| MRR / mean distinct documents in top five | 0.9087 / 2.93 | informational | — |
| Unit document inside the 50-chunk window | 84 of 84 | informational | — |

`ac3` line from the tool: `recall=False coverage=False off_domain=True near_domain=True pass=False`.

### Displacement (aggregate only)

Queries where a non-unit chunk outranks the unit's best chunk: **7 of 84**. The unit was in the 50-chunk window for all 84. The tool prints the ids of non-unit documents involved (not Set B content); I do not reproduce that list. Reading the figures together: raw top-5 recall is 77/84 (7 positives miss the top five even with no floor, equal in count to the 7 displacement queries; I did not verify they are the same queries because the tool prints no per-query data), and the 0.43 floor removes 4 more (77 to 73). Per the ML §4.2 ruling this is reported, with no threshold applied and no retuning.

### Tail versus head (document-level matching)

Matching is document-level throughout (any chunk of the unit counts; S11 defect D-1 still applies: recall does not check which chunk was returned). Tool figures: head-probe recall 0.8451, tail-probe recall 1.0000. Tail does not trail head (the plan's 0.15 trigger for an `architect` decision is not met; the gap is in the opposite direction, tail higher by 0.1549). Derived counts (not printed by the tool): 0.8451 = 60/71 and 1.0000 = 13/13, consistent with 84 positives. The ML §7 risk 1 (tail chunks carrying no title) is not borne out for retrieval; head probes are the weaker class. Because of D-1 this says nothing about whether a tail chunk itself was the one returned.

### BM25 on B (lexical baseline over the 129 unit chunks, no non-unit competitors)

`--baseline bm25 --set B --floor 0.43` (read-only; sentinel not involved): 84 of 84 hits (recall 1.0000, floor-independent), coverage 42/42, MRR 0.9821, mean distinct documents 3.10, head recall 1.0000, tail recall 1.0000. **BM25 beats the semantic arm on B** on recall (84/84 against 73/84 at the floor, 77/84 raw), coverage and MRR (0.9821 against 0.9087). The comparison is unfavourable to the semantic arm in design (BM25 competes only against the unit chunks, the semantic arm against the whole workspace plus a floor), but ML §4.3's stated condition (semantic must beat lexical on recall at equal top-5, else GraphRAG's value is in question) is **not met on B**. This reverses the Set A observation (semantic 42/42 vs BM25 41/42) and Set A could not resolve it.

### Verdict (AC-3, S14)

**FAIL on the positive arm; negative arm passes at its limits.** Floor-applied recall 0.8690 (< 0.90) and coverage 41/42 (< 42/42) miss the plan's gate; off-domain 15/15 and near-domain 12/15 pass. Per plan §3 and ML §5, a failed B is final: no re-run and no retuning (floor, chunking, queries) on this set; the next step is a B2 by a different author, and the stakeholder decides on the floor (4 of 77 raw hits lost to it), the BM25 result and the displacement finding. The seal is spent.

### Coverage, gaps, recommendations

- Not covered: per-query diagnosis (sealed by design); whether raw misses are the displaced queries; captures created after S13b (not enumerated); which chunk (head or tail) was returned (D-1, document-level).
- The raw-vs-floor gap (77 to 73) and the 7 displacement queries are separate loss mechanisms; a B2 and a decision on whether a Set-A-derived diagnosis (not Set B) can inform either is the owner's (`data-scientist`) call.
- Tool note: `eval --sealed-once` requires `--set-b-path` (or `PWQ_SET_B`) even though the brief names the path; the S14 brief template should list it. Raw output kept outside the repo in the session scratchpad only.
