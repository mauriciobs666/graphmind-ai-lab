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

## S14b — sealed Set B2 scored once (2026-11-10, production `ws:agent-team` on `:8200`, read-only)

Replaces the failed S14 Set B result as the AC-3 instrument (`docs/plans/python-web-quirks-kb-b2-ml.md` section 4). Aggregates only: no Set B2 query, target or per-query result appears here or anywhere in the repo. Set B2 sha256 `cd14d4e2d949f277934090f5682b390ab382082c0e83a592fc95a0321d5b5f00`. CPG: not applicable — documentation/retrieval-evaluation task with no code-level component under test.

### Pre-checks (observed this run)

- Set B2 file: `sha256sum` equals the expected value, mode `-r--r--r--`, outside the repo; no `pwq-set-b2.json.scored` before the run.
- Frozen configuration, sha256 prefixes (all equal the brief's): manifest `claude/cobb/scripts/pwq-kb-manifest.json` `c48fbecf9ead08e8`; `skills/agent-kb-retrieval/SKILL.md` `a8c1192359b7ddbe`; `claude/knowledge/python-web-quirks.md` `7d0fc38c054beb25`; `git log -1 --format=%h -- claude/cobb/scripts/pwq_kb.py` = `d1c81826`. Floor 0.43, limit 5 (default).
- `GET /documents/<id>` returned 404 for two of the 17 deleted capture ids (`14745ca4…`, `1bfe7423…`); `458d329e2ba44e45a5fd82301a47504a` returned 200 and was not excluded. No `--exclude` passed.
- `pwq_kb.py gate` (same ws/manifest/copy flags): every unit's stored chunk count equals its embedded count, last line `gate: clean`, exit 0.
- Seal accounting: one scoring invocation (exit 0); it wrote `pwq-set-b2.json.scored` (`floor 0.43`, sha256 as above, `scoredAt 2026-11-10T16:27:56`). No second scoring, no `--probe`.

### Command (single scoring)

`pwq_kb.py eval --ws agent-team --base-url http://localhost:8200 --manifest claude/cobb/scripts/pwq-kb-manifest.json --copy claude/knowledge/python-web-quirks.md --set B2 --sealed-once --set-b2-path <sealed path> --expect-sha256 cd14d4e2…5f00 --floor 0.43`. Tool header: `set=B2 floor=0.43 window=50 topk=5`.

### Results

| Measure | Observed | Decision rule | Result |
|---|---|---|---|
| Positive recall at floor 0.43 (top-5, document-level) | 110 of 126 = 0.8730 | ≥ 116/126 | FAIL (6 short) |
| Positive recall raw top-5 (no floor) | 119 of 126 = 0.9444 | informational | — |
| Per-section coverage, floor-applied | 42 of 42 | all 42 | PASS |
| Per-section coverage, raw | 42 of 42 | informational | — |
| Near-domain false accepts | 9 of 30 | ≤ 6/30 | FAIL (3 over) |
| Off-domain false accepts | 0 of 30 | 0/30 | PASS |
| Overall false accepts (9 + 0) | 9 of 60 | ≤ 6/60 | FAIL |
| MRR / mean distinct documents in top five | 0.8861 / 3.03 | informational | — |
| Unit document inside the 50-chunk window | 126 of 126 | informational | — |

Tool line: `b2: recall=False coverage=True near_domain=False off_domain=True overall=False pass=False`.

### Displacement (aggregate only)

Queries where a non-unit chunk outranks the unit's best chunk: **17 of 126** (unit in the 50-chunk window for all 126). The tool prints the non-unit document ids involved; not reproduced. Raw top-5 recall is 119/126 (7 misses with no floor); the 0.43 floor removes 9 more (119 to 110). Per-query data is not printed, so I did not verify overlap between displacement queries and misses. Reported only, no threshold, no retuning.

### Tail versus head (document-level matching, D-1)

Matching is document-level (any chunk of the unit counts; S11 defect D-1 still applies, so recall does not show which chunk was returned). Tool figures: head-probe recall 0.8661, tail-probe recall 0.9286. Tail is higher than head by 0.0625, so tail does not trail head (the 0.15 trigger is not met). Counts were not printed and are not derived here.

### BM25 comparison (informational)

`--set B2 --baseline bm25 --floor 0.43` (read-only): recall 125 of 126 = 0.9921 (raw equals floor, floor-independent), coverage 42/42, MRR 0.9487, mean distinct documents 3.26, head recall 0.9911, tail recall 1.0000. BM25 exceeds the semantic arm on recall (125/126 against 110/126 at the floor, 119/126 raw) and MRR (0.9487 against 0.8861). Handicap: BM25 ranks over the 129 unit chunks only, with no non-unit competitors and no floor, so it is favoured by design; the semantic arm competes against the whole workspace plus the floor. Direction matches S14 (BM25 84/84 there).

### Pooled with Set B (informational, not part of the verdict)

Printed line: `pooled with Set B (informational, not part of the verdict): (73+110)/210 = 0.8714`. The decision rule's pooled-0.90 equivalence (189/210) is not met, consistent with the B2-alone failure; the pooled figure is clustered by section and the tool printed no interval.

### Verdict (AC-3, S14b)

**FAIL, as printed by the tool.** Of the rule's clauses: floor-applied recall 110/126 (< 116) FAIL; every section hit 42/42 PASS; near-domain 9/30 (> 6) FAIL; off-domain 0/30 PASS; overall 9/60 (> 6) FAIL. Per the B2 protocol a failed B2 is final: the seal is spent, no re-run, no retuning. The finding the protocol names for this outcome is that score-only retrieval at the unchanged configuration does not reach the target on this content; the path (deviation, weak tier, FR-9/FR-11 fallback) is the stakeholder's call. `skills/python-web-quirks/` must not be deleted on this result.

### Coverage, gaps

- Not covered: per-query diagnosis (sealed by design); whether raw misses are the displaced queries; which chunk was returned (D-1); captures created after the S13b derivation (not enumerated; none named).
- Raw tool output kept only in the session scratchpad.


## W3 — staging fidelity check (4b, 2026-11-10)

Unit W3 of `docs/plans/python-web-quirks-kb-path-ml.md` §6 (gate in §5). Read-only against `http://localhost:8210`, `ws:pwq-test` (4b, 2560 dims). Set A only; no sealed file, `--sealed-once`, `--probe` or `:8200` call. Reads of the FalkorDB graphs `ws:pwq-test` / `ws:agent-team` (read-only Cypher, for item 4 only); no writes, no git. Raw output: `/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/c7ed45a3-38f2-4f0c-af72-d53860ccdf25/scratchpad/w3-*.txt`.

CPG: not applicable — black-box fidelity check of a retrieval configuration; no code-level change under test.

Hashes (sha256, as computed in this run): copy `claude/knowledge/python-web-quirks.md` `7d0fc38c054beb25827f2e98ec04cda689457d69e31a6bcb0e70fbfde3099479`; tool `claude/cobb/scripts/pwq_kb.py` `1d768401a58143a652158c6d1d5a907fdf123fb16bdfc0e06fed7a72657112b1`; skill `skills/agent-kb-retrieval/SKILL.md` `a8c1192359b7ddbed95de008d42097a5a44c676f20cde2c0d683da919aece2a4`; manifest `m-work.json` `a405a93c72febf93a74a3e7a95e73f8f41fc87e782eacfcf313240009a586a61`. The staging document-list sha256 was not computed (the list endpoint caps at 200 of 518); the id list is in `/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/c7ed45a3-38f2-4f0c-af72-d53860ccdf25/scratchpad/w3-ids-ws:pwq-test.csv`.

### 1. Gate

`pwq_kb.py gate --manifest m-work.json --copy … --ws pwq-test --base-url http://localhost:8210`: exit 0, last line `gate: clean` (every unit row `chunks == embedded`, `ok`). **PASS.**

### 2. Set A through the server (floor 0.42 explicit)

Command: `eval --set A --floor 0.42 [--sweep] --manifest … --copy … --ws pwq-test --base-url http://localhost:8210`. The tool's Set A mode scores only the **full form** of the 42 positives. The perturbed forms of the embedder note (first half, second half, every third word dropped) have no tool mode, so I generated them mechanically from `claude/cobb/scripts/fixtures/pwq-golden-set-a.json` into scratchpad files (script `w3_perturb.py`; positives only, negatives unchanged) and scored them with the same command plus `--set-a-path`. The note does not fix the rounding of "half"; I ran both and report both. This is a scoring-input choice made before looking at which matches, not a floor tune (floor stayed 0.42).

| Form | Observed raw top-5 (all-workspace) | Offline note | Observed floor-applied hits @0.42 | Offline note | near FA | off FA |
|---|---|---|---|---|---|---|
| full | 42/42 | 42 | 42/42 | 42 | 0/7 | 0/7 |
| first half (n//2 words) | 41/42 | 41 | 37/42 | 37 | 0/7 | 0/7 |
| first half (ceil(n/2) words) | 41/42 | 41 | 38/42 | 37 | 0/7 | 0/7 |
| second half | 38/42 (both roundings) | 38 | 26/42 (both) | 26 | 0/7 | 0/7 |
| every 3rd word dropped | 41/42 | 41 | 36/42 | 36 | 0/7 | 0/7 |

Full-form sweep (`--sweep`, rows 0.38 to 0.50): `recall_floor` 0.9286 / 0.9524 / 0.9524 / 1 at 0.38 / 0.39 / 0.40 / 0.41, 1 from 0.41 up; `near_fa` 0 through 0.42, 1 at 0.43 to 0.47, 2 from 0.48; `off_fa` 0 throughout. Consistent with the note (max best-unit positive 0.408, min near-negative 0.422; the sweep step of 0.01 cannot resolve those two values exactly, and I did not print them).

Floor-applied sums: 42/37/26/36 against the note's 42/37/26/36 (n//2 rounding, exact) or 42/38/26/36 (ceil, +1 on the first-half form). Raw: 42/41/38/41 against 42/41/38/41, exact. Plan §5 tolerance is within 1 per form, near FA at most 1/7, off 0/7, gate clean: all met under either rounding.

**Verdict: server reproduces offline: yes. Differences: none under n//2 rounding; +1 on first-half floor-applied (38 vs 37) under ceil rounding, inside tolerance.** Not reproduced by the tool, by design: nothing beyond Set A (no sealed data touched).

### 3. Query construction (read from `claude/cobb/scripts/pwq_kb.py`, sealed files not read)

- `extract_prefix` (line ~1004) pulls the fenced f-string literal under the skill's "query-instruction prefix" heading by pure string handling; `build_query` (~1031) is `template.replace("{situation}", situation, 1)`. Executed on the live skill: `build_query(t,'SIT')` = `'Instruct: Given a coding agent's description of its current situation, retrieve the distilled technique or rule that applies to it.\nQuery: SIT'`, identical to the literal at `skills/agent-kb-retrieval/SKILL.md` line 46. The same string goes to `GET /documents/search?q=…&limit=50` for every item (`fetch_all`, `searcher.search`).
- Set loading (`load_set`, ~1035) reads each item's `query` field verbatim and nothing else; the tool prepends the template itself, so a set must store the situation text only. **Gap: nothing in the tool rejects a stored query that already starts with `Instruct:`** (grep for "instruct" finds only the skill-extraction code; the Set A fixture has 0 occurrences). A set stored with the prefix would be silently double-prefixed. I cannot confirm from code how B2's or B3's author stored their queries (sealed files not read, as instructed); that is the author-side assertion in the brief, unverified here. Recommendation: the scorer checks the B3 set for a leading `Instruct:` by a non-content probe (e.g. count of queries that start with it) before the seal, or W2 adds a validator line.

### 4. What staging holds besides the 36 units

Read-only Cypher via `GRAPH.RO_QUERY` (python, `w3-ids-*`) over `ws:pwq-test` and `ws:agent-team`:

- Staging: 518 `Document`, all `currentVersion`; 36 are the unit documents (all 36 manifest ids present), 482 others. `sourceKind`: 514 `agent`, 4 `document`. No document title contains "pwq" and no test-only document exists that is absent from production, apart from the S7 captures below.
- Production `ws:agent-team`: 505 `Document`: 36 unit documents (different documentIds from staging's), 465 shared with staging, 4 present only in production (`8a3f5632…` Joern 4.0.579 .pyi capture, `458d329e…` GRAPH.COPY, `ff3cedc8…` decision-models promptSha256, `e017d016…` leave-one-unit-out negatives): none restate a python-web-quirks unit (judged from titles).
- **Staging-only others: 17, and they are exactly the 17 S7 capture ids** that H1 deleted from production (`docs/plans/python-web-quirks-kb-captures.md` §4; count of the 17 ids found in staging by id: 17, in production: 0). They restate unit gotchas, so with them present staging has *more* near-duplicate competition than production.
- Where they bite on Set A: with all 17 present, full-form displacement is 2 positives (`non_unit_outranks_unit=2`, MRR 0.9722); excluding them it is 0 (MRR 1.0000). Perturbed-form figures under the exclusion: first half (n//2) floor-applied 37, raw 41; second half 26/38; drop-3rd 36/41; near 0/7, off 0/7 for all: identical hit counts to the unfiltered runs (the floor decides, as the path note says), only rank statistics improve.

**Recommendation: yes, use `--exclude` for B3, the 17 S7 ids** (one `--exclude` per id, ids in `/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/c7ed45a3-38f2-4f0c-af72-d53860ccdf25/scratchpad/w3-s7-ids.txt`), because production has them deleted and the staging corpus should equal production's competitor set; it moves no hit count on Set A, and it is the same exclusion B used. No other exclusion: no staging-only test documents or newer captures exist. The 4 production-only captures are absent from staging; none is on-topic, so no action, but a capture newly added to `ws:agent-team` after this date is a competitor staging lacks (note for W4/W5 drift check). B3 scoring uses the same manifest.

### 5. Latency sanity (read-only, 4b)

Five distinct situation queries (skill prefix, `limit=50`) to `GET :8210/documents/search`: 0.059 s, 0.044 s, 0.063 s, 0.044 s, 0.057 s wall time each, 50 chunks returned each. Each call embeds the query with the 4b through LM Studio, so the default `--timeout` is ample. No timeouts or errors in any eval above (each B-size set is 156 queries, 12 s at this rate; the four forms of Set A ran in seconds).

### Result

W3 gate met: gate clean, Set A through :8210 reproduces the offline 4b figures (raw exact; floor-applied exact or +1 depending on the unspecified half-rounding), near FA 0/7, off 0/7. Open items for the caller: (a) `--exclude` the 17 S7 ids at B3; (b) the tool does not reject a pre-prefixed stored query (item 3); (c) staging document-list sha256 not recorded here. Floor stays 0.42; nothing was tuned.


## W5 — sealed Set B3 scored once (2026-11-10, staging :8210, 4b)

Unit W5 of `docs/plans/python-web-quirks-kb-path-ml.md` (§5 protocol, rule R1). **This is a STAGING result: `ws:pwq-test` on `http://localhost:8210`, `text-embedding-qwen3-embedding-4b` (2560 dims). Production `:8200` was not used; production is still the 0.6b and nothing was promoted.** Aggregates only: no Set B3 query, target or per-query result appears here. Set B3 sha256 `e455c292db188dada3a0f3d05f3bf49c29eb6297ff25fefabfb0e96d5917cab9`. No git commands, no workspace writes, no server process touched. Raw output (aggregates plus the tool's own printed lines) in the session scratchpad: `/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/c7ed45a3-38f2-4f0c-af72-d53860ccdf25/scratchpad/w5-gate.txt`, `w5-b3-scored.txt`, `w5-b3-bm25.txt`.

CPG: not applicable — retrieval-evaluation task with no code-level component under test.

### Pre-checks (observed this run)

- Set B3 file: `sha256sum` equals the expected value; mode `-r--r--r--`; no `.scored` sentinel before the run.
- Frozen-configuration hashes (first 16 hex, all equal to the brief): `skills/agent-kb-retrieval/SKILL.md` `a8c1192359b7ddbe`; `claude/knowledge/python-web-quirks.md` `7d0fc38c054beb25`; `falkor-chat/config/models.pwq-staging.json` `b7bb9a881b49b37b`; `git log -1 --format=%h -- claude/cobb/scripts/pwq_kb.py` = `f4ae8d07`; production manifest `claude/cobb/scripts/pwq-kb-manifest.json` `c48fbecf9ead08e8` (reference only). Scoring manifest: the staging `m-work.json` (W3 sha256 `a405a93c…6a61`, not recomputed). Floor 0.42 explicit, limit 5 (default).
- `GET :8210/health` = 200. Read-only Cypher on `ws:pwq-test`: 902 chunks, 902 embedded, all with `embeddingModel = lmstudio/text-embedding-qwen3-embedding-4b`.
- `pwq_kb.py gate` (staging, m-work.json, copy): exit 0, `gate: clean`.
- Exclusions: 17 `--exclude` flags, one per id of `h1del.txt` (17 lines). They exist in staging, production deleted them (W3 §4).
- Seal accounting: one scoring invocation (exit 0); it wrote `pwq-set-b3.json.scored` (`floor 0.42`, sha256 as above, `scoredAt 2026-11-10T17:53:22`). No second scoring, no `--probe`.

### Command (single scoring)

`pwq_kb.py eval --ws pwq-test --base-url http://localhost:8210 --manifest <m-work.json> --copy claude/knowledge/python-web-quirks.md --set B3 --sealed-once --set-b3-path <sealed path> --expect-sha256 e455c292…cab9 --floor 0.42 --exclude <17 ids>`. Tool header: `set=B3 floor=0.42 window=50 topk=5`.

### Results (headline = filtered run; unfiltered identical on every figure except MRR and distinct-documents mean)

| Measure | Observed | R1 clause | Result |
|---|---|---|---|
| Positive recall at floor 0.42 (top-5, document-level) | 115 of 126 = 0.9127 | >= 114/126 | PASS (1 above the bar) |
| Positive recall raw top-5 (no floor) | 123 of 126 = 0.9762 | informational | — |
| Per-section coverage, floor-applied | 42 of 42 | every section | PASS |
| Per-section coverage, raw | 42 of 42 | informational | — |
| Near-domain false accepts | 4 of 30 | <= 6/30 | PASS |
| Off-domain false accepts | 0 of 30 | 0/30 | PASS |
| Overall false accepts (4 + 0) | 4 of 60 | <= 6/60 | PASS |
| MRR / mean distinct documents in top five | 0.9524 / 2.89 (unfiltered 0.9028 / 3.02) | informational | — |
| Unit document inside the 50-chunk window | 125 of 126 | informational | — |
| B2 bar (116/126), printed as non-verdict figure | `b2_bar_116: False` | not part of the verdict | 115 < 116 |

Tool line: `b3: recall=True coverage=True near_domain=True off_domain=True overall=True pass=True`.

### Displacement (aggregate only)

Queries where a non-unit chunk outranks the unit's best chunk: **6 of 126** (unit in the 50-chunk window for 125 of 126). The tool prints the non-unit document ids involved; not reproduced. Raw 123/126 means 3 misses with no floor; the 0.42 floor removes 8 more (123 to 115). The tool prints no per-query data, so overlap between displacement queries and misses is not verified. Reported only.

### Tail versus head (document-level matching, D-1)

Matching is document-level (any chunk of the unit counts; S11 defect D-1 still applies, so recall does not show which chunk was returned). Tool figures: head-probe recall 0.9018, tail-probe recall 1.0000. Tail is higher than head by 0.0982; tail does not trail head (the 0.15 trigger is not met). Counts were not printed and are not derived here.

### BM25 comparison (informational)

`--set B3 --baseline bm25 --floor 0.42 --set-b3-path <sealed path>` (read-only; no sentinel involved): recall 123 of 126 = 0.9762 (raw equals floor), coverage 42/42, MRR 0.9474, mean distinct documents 3.08, head recall 0.9732, tail recall 1.0000. The semantic arm at the floor (115/126, 0.9127) is below BM25 (123/126); semantic raw (123/126) equals BM25 on recall; semantic MRR 0.9524 (filtered) is slightly above BM25's 0.9474. Handicap: BM25 ranks over the 129 unit chunks only, with no non-unit competitors and no floor, so it is favoured by design; the semantic arm competes against the whole staging workspace plus the floor. The tool prints no false-accept figures for BM25.

### Verdict (R1, as printed by the tool)

**PASS.** Floor-applied recall 115/126 (>= 114) PASS; every section hit 42/42 PASS; near-domain 4/30 (<= 6) PASS; off-domain 0/30 PASS; overall 4/60 (<= 6) PASS. The margin on recall is one query. No retuning, no second scoring, no reinterpretation; the seal is spent. R2 was not registered and is not applicable.

**Scope of the claim: staging only.** The pass is for the 4b configuration in `ws:pwq-test` (staging, with the 17 deleted capture ids excluded to match production's competitor set). Production `ws:agent-team` on `:8200` is still embedded by the 0.6b and was not touched; nothing was promoted, and no production floor, embedder or manifest change follows from this report alone. Promotion is a separate stakeholder-gated step (path note §6).

### Coverage, gaps

- Not covered: per-query diagnosis (sealed by design); whether raw misses are the displaced queries; which chunk was returned (D-1); captures created in production after W3 (not enumerated; staging does not hold the 4 production-only captures per W3 §4); whether Set B3's author stored queries without a leading `Instruct:` (W3 §3 gap, not checked here because the sealed file is not to be inspected beyond scoring; the aggregates, with raw recall 0.9762, show no sign of a double-prefix collapse).
- Recall clears the bar by one query, so the result is sensitive to one query's outcome; the path note's own estimate (§4) and its multiplicity caveat (§7: the 4b is the best of many Set A looks) still apply to any claim beyond "B3's R1 is met".

## Step 0c — paired pre-gate (W7d, 2026-11-10)

Rule: `docs/plans/python-web-quirks-kb-prod-migration-ml.md`, sha256 verified before anything else = `308677fcbac108f277b728910455ac5678a51a661326e622ae94b5f3cea4c74e` (match). Runner `claude/cobb/scripts/kb_golden_runner.py`, fixture `claude/cobb/scripts/fixtures/kb-golden-45.json`. One scored run per configuration, no retries, no floor changes; both runs exit 0 with empty stderr. Raw JSON: `/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/c7ed45a3-38f2-4f0c-af72-d53860ccdf25/scratchpad/w7d-baseline.json` (A: `:8200` `ws:agent-team`, 0.6b, floor 0.43) and `/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/c7ed45a3-38f2-4f0c-af72-d53860ccdf25/scratchpad/w7d-candidate.json` (B: `:8210` `ws:pwq-test`, 4b, floor 0.42). Independent recomputation (throwaway `/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/c7ed45a3-38f2-4f0c-af72-d53860ccdf25/scratchpad/w7d-verify.py`, from the raw docs lists plus fixture expected ids, not the runner's aggregates): **no difference** from the runner's aggregates (singles hit 28 / 29, siblings 12 / 13, negatives admitted 1 / 0, closest negative 0.4100 / 0.5114).

CPG: not applicable — a retrieval-metric run over a fixture with no code-level component.

**Fixture validity:** 32 single / 7 family / 6 negative (the rule's counts hold; all 32 singles `applies`). **Readiness limit (declared):** the runner does not check document status, and I stayed within the brief's "GET /documents/search only", so I did not call `GET /documents/{id}`. From the search results alone, of the 46 expected ids, 42 appeared in A's results and 44 in B's. Not seen in A: R3 sibling `ccefd08d`, C1 `9870c48f`, P1 `08a422ae`, G2 `f29bddad`; not seen in B: R6 sibling `5b1b477f`, G2 `f29bddad`. Every one of those ids was seen in at least one of the two runs except **G2 `f29bddad` and C1 `9870c48f`, never seen in either top five**; those two are the expected-id `ready` check I could not complete. If either is not `ready` in its workspace the rule calls the run invalid. Closing it takes a read-only `GET /documents/{id}` on both, which I did not make.

### Paired table (32 singles; hit = expected id in top-5 AND best score <= floor)

| | B hit | B miss |
|---|---|---|
| **A hit** | 28 | **b = 0** |
| **A miss** | **c = 1** (P1) | 3 (C1, X1, G2) |

A hits 28/32, B hits 29/32, net loss b - c = **-1** (a net gain of one). Raw top-5 table (floor-free): A 28, B 29, raw b 0, raw c 1; floor-applied equals raw on both. Exact one-sided sign p on the discordant pair (b=0, c=1; n=1): 1.0, descriptive. Wilson 95%: A 28/32 = 0.875 [0.719, 0.950]; B 29/32 = 0.906 [0.758, 0.968].

### Negatives (hard guard)
A (0.6b, 0.43) top-1 scores: N1 0.6474, N2 0.5542, **N3 0.4100 (admitted)**, N4 0.4466, N5 0.5930, N6 0.4968: **1 admitted**. B (4b, 0.42): N1 0.6101, N2 0.5508, N3 0.5114, N4 0.5266, N5 0.7019, N6 0.5742: **0 admitted**. B admits no more than A: guard **passes**. Closest negative: A 0.4100 (margin to its floor 0.0200, inside it, admitted); B 0.5114 (margin 0.0914). Note: the rule expected 0 admitted at baseline and a closest pooled negative of 0.446; today's 0.6b baseline admits N3 at 0.4100, a corpus-drift or fixture-vs-corpus observation to report, not a stop condition.

### STOP conditions
1. net loss >= 4: b - c = -1, **not met**. 2. B admits more negatives than A: 0 vs 1, **not met**. 3. Fixture counts or readiness: counts hold; readiness partly unverified (above).

### Reported, not judged
- **Lost pairs (b):** none. No raw rank/score to list. Pairs in (0.42, 0.43] for the 4b: none (no single's best 4b score lies in that band).
- **Missed at both (concordant):** C1, X1, G2 (expected id not in the document-level top-5 at either configuration); P1 missed at A, found at B rank 1, score 0.3991.
- **Borderline (within 0.02 of the floor):** A: R6 second sibling `5b1b477f` rank 5, 0.4393 (0.0093 above 0.43, not found); C4 rank 1, 0.4128 (hit); R1 second sibling rank 2, 0.4115 (found). B: S1 rank 1, 0.4118 (hit, 0.0082 under 0.42); R6 second sibling not in the 4b top-5 at all (score none; it was found by A only at rank 5 above the floor). P1 at B 0.3991 and negative N3 at A 0.4100 sit just outside / inside the band (0.0209 / 0.0200).
- **R6:** first sibling `af9ffb19` rank 1 on both (A 0.2813, B 0.2150); second sibling A 0.4393 rank 5 (over floor), B not in top-5. R6 recall 1/2 on both.
- **Family set-recall (sibling count under the floor):** A 12/14 (R1 2/2, R2 2/2, R3 1/2, R4 2/2, R5 2/2, R6 1/2, Q1 2/2); B 13/14 (R1 2/2, R2 2/2, R3 2/2, R4 2/2, R5 2/2, R6 1/2, Q1 2/2). R3's second sibling: A not in top-5, B 0.2693.
- **Baseline drift:** A single hits 28 against the recorded 28/32 (floor-applied equals raw): drift 0, under the 3 threshold.

### Lock rehearsal witness
`bash falkor-chat/scripts/rehearse_agent_team_wrapper.sh --root /tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/c7ed45a3-38f2-4f0c-af72-d53860ccdf25/scratchpad/w7d-rehearsal` (one run, not repeated): **SUMMARY pass=23 fail=13 mutant=none**. Failing items (log `/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/c7ed45a3-38f2-4f0c-af72-d53860ccdf25/scratchpad/w7d-rehearsal/tree/run/results.log`): a2.3, a2.5, a4.1, a6.1-a6.7, a3.5, a3.6, a5.2. Two anomalies in the log that look like harness/state faults rather than a locking verdict: a2.5 and a4.1 show an EXEC timestamp of about -1.79e12 ms (an unset/epoch-relative value), and a6.2 shows the dummy's cwd under an earlier `.../scratchpad/teco-rehearsal2/tree/falkor-chat` rather than this run's tree (a stale listener or state from a previous rehearsal seems to have answered on the scratch port, though a8.1 reports no 8298/8299 listener at the end). Not diagnosed. The load average was about 3.7 on 8 cores. I did not rerun (no retries). a8.3/a8.4 pass: real :8200/:8210 listeners, the real lock and `falkor-chat/` were untouched. This witness does **not** reproduce W7c's rehearsal result and is not a PASS; it does not enter the 0c verdict (the rule is retrieval-only) but the caller should look at it before the outage step.

### Verdict: **GREEN** on the frozen paired rule (net -1, negatives 0 vs 1), conditional on the one open validity item
Green supports only: no gross floor-applied top-5 retrieval regression on these 32 author-chosen Stage 8 pairs (a loss of up to 3 net pairs would also have passed; 4 of the rows were the 0.43 floor's pilot, the 4b floor was never calibrated on them). It says nothing about production quality, queries unlike these 32, or the smoke list (§7 non-pwq, step 0c's other half), which I was not asked to run here. Open before relying on it: `ready` status for G2 and C1 expected ids (not seen in any top-5), and the failed rehearsal witness above.
