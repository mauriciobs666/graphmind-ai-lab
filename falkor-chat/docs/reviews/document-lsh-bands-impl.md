# K-091 implementation review — `Document.lshBand0..7` stored at ingest, plus backfill

> **Status:** archived · **Owner:** `analyst` · **Tracks:** K-091

## Scope & verdict

**Scope.** This review covers the uncommitted `falkor-chat/` working-tree diff for K-091:
- `server/falkorchat/{update_detection,repository,services,background}.py`, `AGENTS.md`, `scripts/test_queries.sh`;
- the tests in `server/tests/{test_update_detection,test_services,test_repository,test_api}.py`;
- the untracked `scripts/backfill_document_lsh_bands.{sh,py}` and `server/tests/test_backfill_document_lsh_bands_script.py`.

The baseline is `BACKLOG.md` `### K-091` (scope and done-condition), with root and `falkor-chat/` `REVIEW.md` applied. There is no plan document. Out of scope: `docs/plans/graph-config-defaults-graph.md`, everything outside `falkor-chat/`, and live-spec edits, which are deferred to the close. The spec sentences that are now wrong are listed below.

**Verdict: approve with suggestions.** There is no blocker. The write site is correct, and so is the single band pipeline. Ingest, detection and the backfill all compute bands the same way. The suite is green and the end-to-end band-signal tests have a working positive control. Five argument- and fixture-level mutants survive. All five are about pinning (F1, F2), not wrong behaviour today. The live half of the done-condition is still open (F4).

CPG: considered, not relevant — no `cpg_falkorchat` freshness was stated for this uncommitted diff, and every claim here needed the working-tree text, so I read it, grepped it and mutated it directly.

**Checks run (all verified):**
- **pytest:** 842 passed over `test_update_detection`, `test_services`, `test_backfill_document_lsh_bands_script`, `test_repository` and `test_api` (`cd falkor-chat/server`). These request `repo`/`conn` against `ws:test`. `conftest.py` forces `reference-test`, so the live `reference` was not touched. The `wf_repo` tests in other files were not run.
- **Query suite:** `./scripts/test_queries.sh` passed 475/475. It wipes only `ws:test` and `reference-test`; no re-seed needed.
- **Mutation:** 11 mutants, each applied to a copy-aside file and restored by copy. The sha256 restore check passed after every batch.
- **Live backfill dry runs:** run against `acme` and `agent-team`. A dry run is read-only (`ro_query`).

## Findings

### F1 — minor — The backfill listing's two contract properties are pinned only on a hand copy, not on the shipped method

`Repository.list_documents_missing_lsh_bands` claims to list every document *missing any band*, *current or not* (`repository.py:1984-1995`). Two mutants of the shipped method both left the 842 tests green:
- **MA:** only `lshBand0 IS NULL` is checked (`range(1)`).
- **MJ:** `d.currentVersion = true AND (…)` is added.

The only assertions on these properties are in `test_queries.sh`: `asd1` non-current, the `LIST_MISSING_BANDS` copy. Mutating `repository.py` does not change that copy, so those assertions cannot redden on a regression in the method.

The pytest live test (`test_live_backfill_stamps_bands_only_on_band_less_documents`) has two gaps:
- its only band-less document is current (written by `create_document`);
- the banded document carries all 8 bands or none.

Live impact today is low: no partially-banded documents exist, and no path re-flips `currentVersion` to `true`. However, the done-condition says *every* Document is banded.

A third survivor is **MB**: `set_document_lsh_bands` hard-wired to `return True`. The missing-document branch is never exercised, so the script's `wrote N/M` count is unpinned.

**Fix.** Extend the live pytest so that it plants three kinds of document and asserts the exact listed set. Then add one `set_document_lsh_bands` call on a non-existent id that asserts `False`. The probe covers the listing's own axes: band presence (none, some, all) × `currentVersion` (true, false).

| Planted document | Bands | `currentVersion` | Expected |
|---|---|---|---|
| (a) | 0..6, `lshBand7` absent | true | listed |
| (b) | none | false (superseded, e.g. via `_auto_supersede` with a shared hash, then a band-less raw `CREATE`, or `SET d.currentVersion=false`) | listed |
| (c) | all 8 | false | not listed |

MA, MJ and MB must each turn the extended test red.

### F2 — minor — Ingest-side band storage is unpinned on the `produced_by` branch and for multi-chunk text

Two more mutants survived the full five-file run:
- **ML:** `band_props` emitted only `if not produced_by`. The `produced_by` branch is how agents ingest into `ws:agent-team`, which is the largest corpus with 522 documents.
- **MC2:** ingest bands computed from `text[:2000]`. Every *exact* band assertion uses sub-chunk text: `test_services` (40 words), `test_api` auto-supersede (34 chars), `test_repository` (`"x"`). The one long-text test, the API near-duplicate test, asserts only that a suggestion appears. A truncated-prefix band still collides, so it stays green.

Today's code is correct on both counts. `band_props` is appended after the prefix choice (`repository.py:1920-1921,1938`), and `services.py:1319` passes the full `text`. These pin future regressions; they are not current bugs.

**Fix.**
- In `test_repository.py`, add a `produced_by` twin of `test_create_document_with_auto_supersede_stores_each_lsh_band_on_the_document`: seed an `Agent`, pass `produced_by=` and assert the 8 bands.
- In `test_services.py`, give one exact-band test a text of more than 2,000 characters (several chunks), with the expected value computed explicitly, as the test already does.

ML and MC2 must each turn red.

### F3 — minor — The band writers trust `len(lsh_bands)` (brief claim 2)

Both writers build their query text from `len(lsh_bands)`:
- `create_document_with_auto_supersede` (`repository.py:1921`);
- `set_document_lsh_bands` (`:2005`).

Building it this way is safe from injection, because only integer indexes are interpolated. It is not safe for the invariant. A short list, `[]` included, silently creates a document with fewer bands. The backfill would then re-list that document and patch it, but only if someone runs the backfill again. A longer list writes unindexed `lshBand8+`. The listing already pins the count to `LSH_BAND_COUNT` (`:1988`), so the writer and the listing can disagree.

**Fix.** In both writers, raise `ValueError` when `len(lsh_bands) != _LSH_BAND_COUNT`, then build from `range(_LSH_BAND_COUNT)`.
- Add a test first that passes 7 bands and asserts `ValueError`. Today's code fails that test, so it can redden.
- The expected count in that test must be the literal `8`, not `LSH_BAND_COUNT`.

### F4 — minor — Rollout order and the live done-condition are not yet satisfied or documented

A live read on 2026-11-20 found every document still band-less: 522 of 522 in `ws:agent-team` and 29 of 29 in `ws:acme`, with no null `text` in either. The dry runs list exactly those counts. So the servers serving those workspaces run pre-K-091 code. Any document ingested between `--apply` and the restart will be band-less.

The backfill is idempotent, so a second run repairs this. Nothing tells the operator to restart first, though: not the `AGENTS.md` row and not the script header. The done-condition's "near-duplicate with an unrelated title is suggested through the band signal **on a live instance**" has not been exercised. The API tests run on `TestClient` against `ws:test`.

**Fix.**
- Add "run after restarting every server on K-091 code; re-run is safe" to the script header. In the `AGENTS.md` row, add only a clause.
- At the close, record the live near-duplicate probe and the post-`--apply` dry-run count of 0 per workspace.

### F5 — minor — The `test_queries.sh` copy is correct now, but nothing stops it drifting again

I rendered the shipped `create_document_with_auto_supersede` query (`ingestedBy` branch) by calling the method against a capturing fake graph. It is whitespace-normalized equal to `CREATE_DOC_WITH_AUTO_SUPERSEDE`. `LIST_MISSING_BANDS` and `SET_DOC_BANDS` are also equal (verified; method in Appendix A).

The pre-diff copy had already drifted: it lacked `ingestorIsUser`, which the diff now fixes. **Fix.** Add a pytest that renders each copied query this way and compares it with the `test_queries.sh` literal. The old stale copy is the planted input that reddens it.

### Nits

- **N1.** The `create_document_with_auto_supersede` docstring (`repository.py:1888`) says the caller computes bands via `update_detection.lsh_bands`. The caller uses `document_lsh_bands`, the one pipeline, so cite that.
- **N2.** A mistyped workspace in a multi-workspace run prints a raw `ResponseError: Invalid graph operation on empty key` traceback (verified with `nosuch`). It also aborts the remaining workspaces. Catch the error per workspace and report it. The listing is an `ro_query`, so no graph is created.
- **N3.** The listing pulls every band-less document's full `text` in one result, even in a dry run. That is up to 500k characters per document (`MAX_DOCUMENT_CHARS`). This is fine at today's sizes (largest 13k characters, 551 documents). One `SET` round trip per document is also fine at that scale.

  For a large workspace, page by `documentId` instead (`WHERE … AND d.documentId > $after … LIMIT $n`), and return no text in a dry run. Not needed for this one-off.
- **N4.** `test_queries.sh` labels these assertions `§14.10`, but QUERIES puts update detection at §14.9. This mislabel predates the diff.

## Claim checks

1. **Ingest, not detection, as the write site: confirmed by tracing all three reasons.**
   - Detection is skipped for an auto-superseded ingest: `api.py:193,234`, `mcp.py:371,413` (`if not receipt.get("autoSuperseded")`).
   - There is no detection repo unless an embed worker or pipeline is wired: `app.py` `repo_for_detection = getattr(embed_worker,"repo",None) or getattr(ingestion_pipeline,"repo",None)`, then `_schedule_update_detection` returns on `None` (`background.py:315-319`).
   - A failure is swallowed: `background.py` `except Exception: _log.exception`.

   A fourth reason: a detection-time write would run *after* this document's own shortlist read. The ingest write needs no extra round trip.
2. **The query text built from `len(lsh_bands)`:** injection-safe, invariant-unsafe. See F3.
3. **One band pipeline: confirmed.** An unfiltered `grep -rn "lsh_bands\|minhash_signature\|lshBand"` over `falkor-chat/` (excluding `.venv` and tests) finds band computation only in `update_detection.document_lsh_bands`. Its callers are `services.py:1319`, `background.py:273` and `backfill_document_lsh_bands.py:49`. `background.py:272` still shingles separately, for Jaccard only.

   The tests compute their expected bands independently through the explicit four-step chain (`_bands_of`), as `REVIEW.md` requires.
4. **`create_document` has no non-test caller: confirmed.** I swept `create_document(` and `CREATE (d:Document` over the whole component:
   - production `CREATE (d:Document` sites: only `repository.py:1077` (`create_document`) and `:1932`;
   - every `create_document(` call is under `server/tests/`;
   - scripts ingest over REST (`seed_nlq_eval_corpus.py`).

   QUERIES §14.1's "no server caller" stays true.
5. **The backfill:**
   - **Dry run by default:** confirmed. The MD mutant, which makes a dry run write, is killed.
   - **Idempotent:** confirmed (live pytest and `test_queries.sh`).
   - **"Missing any band, current or not":** true of the code but unpinned (F1).
   - **`text` null → bands of `""`:** right. Skipping the document would re-list it forever and break idempotency. `""` gives the all-zero-signature bands. Ingest rejects blank text (`EmptyDocumentError`, `services.py:1295`), so no probe can ever compute those bands. The constant therefore cannot crowd a real shortlist, and no null-text document exists live.
   - **Performance:** one `SET` per document is fine (N3).
6. **`test_queries.sh` copies match: confirmed** for all three copies (F5). Only the `ingestedBy` prefix is copied; the `produced_by` prefix has no shell copy, as before the diff.
7. **Mutants.** See Appendix B. Surviving mutants: MA, MJ, MB (F1) and ML, MC2 (F2). All the others are killed.

## Live-spec sentences for the close

| # | Location | Sentence | Why it is now wrong |
|---|---|---|---|
| 1 | QUERIES §14.9 | "**No server path writes `Document.lshBand0..7`.** The bands are computed only for this read…" | Now false. |
| 2 | QUERIES §14.9 | The shared-body `CREATE (d:Document {…})` | Lacks `lshBand0..7: $lshBand0..7`. |
| 3 | QUERIES §14.9 | The prose "§14.1's shape plus `textNormalizedHash`" | Should also mention the bands. |
| 4 | QUERIES §14.9 | `find_update_shortlist`'s "(`update_detection.lsh_bands`, 8 bands)" | Should name `document_lsh_bands`, shared with ingest. |
| 5 | QUERIES | (missing) | No entry exists for the two new canonical queries, `list_documents_missing_lsh_bands` and `set_document_lsh_bands`, both PROFILE-asserted in `test_queries.sh`. |
| 6 | DESIGN §5.1 | The `(:Document {…})` property list | Lacks `lshBand0..7`. |
| 7 | DESIGN §5.1 | Key-properties bullet: "**no write path sets them**… only the title-fuzzy signal shortlists a candidate today" | Now false. It should also name `scripts/backfill_document_lsh_bands.sh`, as the `threadId` bullet names its backfill. |
| 8 | DESIGN §7.1 | Index table row: "no write path sets these today (§5.1)" | Now false. This is the same fact as row 7, stated twice in DESIGN. Per `REVIEW.md`, keep one home and have the other cite it. |
| 9 | DESIGN | The "Ingest document" operations row | Omits that the create stores the bands. |
| 10 | `BACKLOG.md` K-064 | "no path writes the bands (K-091)… title-fuzzy alone decides" | The premise is now stale. The done-condition already names this one. |

## What's solid

- The write site is correct, and the docstring states the reason for it.
- The single helper removed a real two-way-drift risk: before this diff, the test fixture shingled independently of detection.
- The backfill reuses the existing script pattern: a thin `.sh` wrapper over a testable `run(ws, repo=…, apply=…)` with a fake-repo seam.
- The ingest write remains one parameterised `GRAPH.QUERY`.
- The near-duplicate API tests are well designed:
  - the title signal is disproven before the second ingest;
  - the case/whitespace variant would catch a raw-text pipeline on either side;
  - with the band write removed, both turn red (positive control, verified).
- The stale `test_queries.sh` copy is regenerated, and the new band-write shape is PROFILE-checked for its index anchor.

## Proposed REVIEW.md lines

Target: `falkor-chat/REVIEW.md`, `## Always check`. Both lines are additions. No existing line is enforced elsewhere, so there is no prune.

- *Addition* (source F1). The current gap in this file is that it says nothing about a hand-copied query:
  > A `test_queries.sh` assertion runs a hand copy of a repository query, so it pins the copy, not the method. A contract claimed for the method (a `WHERE` disjunct, a filter that is present or absent) needs a pytest over the real method, and a mutant of the method must turn it red.
- *Addition* (source F2, mutant ML):
  > A property added to a query that has alternative literal prefixes or branches (`$ingestedBy` / `$producedBy`) is asserted as written on every branch, not only on the default one.

## Open questions

- Should F4's live near-duplicate probe and the post-`--apply` counts be part of this unit or of the feature close? The backlog done-condition requires both before K-091 closes.

## Appendix

### A. Copy-equality method (F5, claim 6)

I subclassed `Repository` with `_graph` returning a fake that captures `query` and `ro_query` text. Then I called `create_document_with_auto_supersede(…, lsh_bands=[8 strings])`, `list_documents_missing_lsh_bands`, and `set_document_lsh_bands`. Each captured text was compared with the `test_queries.sh` literal: `'"'"'` was un-escaped, and runs of whitespace were collapsed (`re.sub(r"\s+"," ",s).strip()`). All three were equal. Captured parameter keys: `chunks, createdAt, documentId, ingestedBy, lshBand0..7, matchId, sourceFormat, text, textNormalizedHash, title`.

### B. Mutation table

The run covered the five pytest files above (842 tests), restored by copy with sha256 verified.

| ID | Mutation | Result |
|---|---|---|
| MA | listing checks `lshBand0` only | **survived** |
| MJ | listing adds `currentVersion = true` | **survived** |
| MB | `set_document_lsh_bands` returns `True` | **survived** |
| ML | ingest band props omitted when `produced_by` | **survived** |
| MC2 | ingest bands from `text[:2000]` | **survived** |
| MC | ingest bands from `chunk_texts[0]` | killed (1 failed) |
| MK | band 7 written with band 0's value | killed |
| MBG | detection bands from `title` | killed |
| MS | ingest bands from `title or source_label or text` | killed |
| MD | dry run falls through to writes | killed |
| M0′ | ingest writes no band props | killed (both band-signal API tests) |

## Pass 2

**Verdict: approve with suggestions.** All five Pass 1 survivors are now killed. The F3 guard, the F5 copy guard and N2 are consistent with each other. The one new finding, F6, is minor: the F5 copy guard reads the copies with a regex instead of evaluating them the way bash does.

**Re-run on 2026-11-20 against the current tree:**
- pytest: 859 passed over `test_query_copies`, `test_backfill_document_lsh_bands_script`, `test_repository`, `test_services`, `test_api` and `test_update_detection`. These use `ws:test` and `reference-test`; the `wf_repo` tests in other files were not run.
- `./scripts/test_queries.sh`: 475/475. It wipes only `ws:test` and `reference-test`.
- Mutants: applied by copy-aside and restored by copy, with sha256 verified after each one.

**Dispositions**
- **F1: fixed.** MA (2 failed) and MJ (2 failed) are killed by `test_list_documents_missing_lsh_bands_lists_any_missing_band_current_or_not` and by the copy guard. The test plants none/some/all bands × current/non-current. MB is killed (1 failed) by `test_set_document_lsh_bands_stamps_all_8_and_reports_whether_the_document_exists`, which asserts `ghost` returns `False`.
- **F2: fixed.** ML is killed by the new `produced_by` band-storage test (`test_repository.py:1596`). MC2 is killed by `test_ingest_document_bands_cover_the_whole_text_of_a_multi_chunk_document`.
- **F3: fixed.** `_lsh_band_params` raises `ValueError` unless there are exactly 8 bands, and both writers build their query text from `range(_LSH_BAND_COUNT)`. I probed the guard with a fake graph that fails on any query:
  - 7, 9 or 0 bands raise on both ingest branches, default and `produced_by`;
  - 7 bands raise on `set_document_lsh_bands`;
  - 0 queries ran.

  (a) The copy guard renders the post-guard text, and it equals all five shell copies.
- **F4: docs half fixed.** The rollout order is stated in the script header and in a clause of the `AGENTS.md` row. The live half is still open for the close: the live near-duplicate probe and a dry-run count of 0 per workspace after `--apply`.
- **F5: fixed, with a gap.** `tests/test_query_copies.py` covers five copies, and its positive control is the pre-K-091 stale copy. Each copy is defined once and used twice (`grep -cF`). (c) The gap is F6.
- **N1: fixed.** The docstring now cites `document_lsh_bands`.
- **N2: fixed.** (b) The catch is safe:
  - **`ValueError`:** propagates, so the run fails with a traceback.
  - **`ResponseError` from the write:** the script exits 1, not 0. It does mislabel the error as "no such workspace graph?". That is a nit; the message could say "write or read failed".
  - **Other redis errors** (`ConnectionError`): not caught.

  I verified these with a fake repo whose `set_document_lsh_bands` raises.
- **N3: skipped.** Accepted, since it is a one-off.
- **N4: fixed.** `grep -c '§14.10' scripts/test_queries.sh` returns 0.

### F6 — minor — The F5 guard compares a regex reading of the source, not the value bash runs

`shell_copy` regex-parses the first `^NAME=` line and undoes the quoting itself. Two mutants of `test_queries.sh` left `test_query_copies.py` green (6 passed):
- **C1:** `\$fuzzyQuery` changed to `$fuzzyQuery` in the double-quoted `FIND_SHORTLIST_TITLE`. Bash expands the unset variable to nothing, so the suite runs `queryNodes('Document', )`, but the parser maps both spellings to `$fuzzyQuery`.
- **C2:** a second `LIST_MISSING_BANDS=` that checks only `lshBand0`, inserted before its first use. The regex reads the first definition and bash runs the second. This is MA at the copy level, and the suite's own assertions (lists `asd1`, excludes `asd2`) stay green under it too.

Separately, the band-OR shortlist read that this diff added inline (`test_queries.sh:1893`) is a sixth copy with no variable, so nothing guards it.

**Fix.** Get each copy's value from bash: `bash -c "$(grep "^NAME=" test_queries.sh); printf %s \"\$NAME\""`. For C1 this yields `…queryNodes('Document', ) …`, which differs from the shipped text (verified). Also assert exactly one `^NAME=` definition, which C2 would fail. Then route `:1893` through the `FIND_SHORTLIST_BANDS` variable. C1 and C2 must turn the guard red.

## Close check

**Scope.** I checked the uncommitted spec edits against the shipped commit `4d35b84e`:
- `git diff -- docs/QUERIES.md docs/DESIGN.md docs/BACKLOG.md`;
- the writer's claim that `SERVER.md` needs no change.

The working-tree code equals `4d35b84e` (`git diff --stat 4d35b84e -- server scripts` is empty). I did not run the suites, because `ws:test` is held by K-096. The one test run was `test_query_copies` driven against a scratch copy of `test_queries.sh`. It is graph-free: `conftest.py`'s only autouse fixture sets model-config env vars.

**Verdict: approve with suggestions.** There is no blocker. Every Cypher block matches what the code sends, and rows 1–10 are all resolved. Two minor wording findings, C-1 and C-2, can be fixed at the close or left for later. Neither is a contract contradiction.

**Cypher blocks** (I rendered each query with a capturing fake graph and compared it with the QUERIES §14.9 blocks):

| Block | Result |
|---|---|
| Default prefix + shared body | Equal to the shipped text once whitespace, including whitespace around punctuation, is ignored |
| `produced_by` prefix + shared body | Same as the default prefix |
| `list_documents_missing_lsh_bands` | Equal after collapsing runs of whitespace |
| `set_document_lsh_bands` | Equal after collapsing runs of whitespace |
| Band shortlist | Unchanged; its `…` elision is the pre-existing, declared shape |

**Absolute words** (each verified by an unfiltered grep over `falkor-chat/`, excluding `.venv` and `server/tests/`):
- **"the one band pipeline … the shortlist read and the backfill also use":** `document_lsh_bands(` is called only at `services.py:1319`, `background.py:273` and `backfill_document_lsh_bands.py:54`.
- **"Called only by the backfill script":** the only non-test callers are `backfill_document_lsh_bands.py:44,53`.
- **"this write is what bands every ingested document":** true. Every ingest goes through `Services.ingest_document`, which is called by REST, MCP and the batch path (`services.py:1437`). `create_document` has no non-test caller.
- **"every `Document` missing any band, current or not":** true, and pinned by the F1 test.
- **"exactly 8 bands … `ValueError` before any query runs":** true for both writers (Pass 2 probe). Over-scoped in one place: see C-1.

**Rows 1–10:** all resolved. I grepped the living docs (`docs/*.md`, excluding `archive/`, `reviews/`, `plans/` and `HISTORY`) for "No server path writes", "no write path sets" and "no path writes the bands". No band-related hit remains outside the K-091 backlog item itself, which leaves `BACKLOG.md` at the close as a delivered item.

DESIGN §7.1 now cites §5.1 instead of restating the fact, which resolves the row 8 duplication.

**Numbering and history:**
- No heading line changed in either spec file.
- `K-091` appears nowhere in QUERIES or DESIGN.
- "A document ingested before the create stored them lacks them" describes a live data condition an operator acts on, not change history. It follows the `Message.threadId` bullet's precedent.

**`SERVER.md`: the writer's claim holds.**
- Its ingest rows (`:278-279`, `:1090-1091`) cite §14.9 by number, and the REST/MCP surfaces, errors and §1.12 scheduling are unchanged.
- `ValueError` cannot be reached from ingest, which always passes 8 bands.
- `SERVER.md` lists no operator scripts.

**F6: fixed** in `4d35b84e`. The copy is now evaluated by bash, and the guard asserts exactly one definition. Pass 2's mutants C1 and C2, run against a scratch copy, now turn red (`FIND_SHORTLIST_TITLE` and `LIST_MISSING_BANDS` respectively). The baseline stays green. Line 1893 now uses `$FIND_SHORTLIST_BANDS`.

### C-1 — minor — "The repository takes exactly 8 bands" is broader than the code

QUERIES §14.9 (create entry): "The repository takes exactly 8 bands: any other count raises `ValueError` before a query runs." `find_update_shortlist` has no such guard. It builds its query from `len(bands)` (`repository.py:2070-2072`), and the band-signal comment in the same section says so ("generated for i in 0..len(bands)-1").

**Fix.** Scope the sentence to "Both band writes (this create and `set_document_lsh_bands`) take exactly 8 bands…". The `set_document_lsh_bands` note can then cite it instead of repeating it.

### C-2 — minor — The backfill script's behaviour is stated in two spec sections

Both DESIGN §5.1's `lshBand` bullet ("dry run without `--apply`; idempotent") and QUERIES §14.9's backfill entry ("dry run by default, `--apply` to write"; "a re-run writes nothing it already wrote") describe the script. Both copies match the code (`backfill_document_lsh_bands.py` `run`/`main`).

Per `REVIEW.md` ("a fact stated in two sections"), keep the behaviour in QUERIES §14.9. The DESIGN bullet should only name the script and cite §14.9.

**Nit (outside the spec, not caused by this diff):** the rewritten K-064 item still quotes a docstring at `repository.py:1863-1868` ("a cheap complementary booster…"). That text no longer exists in `repository.py` (`grep -n complementary` finds nothing). Drop the quotation or re-point it at the next rewrite.

### Proposed REVIEW.md lines (final)

Target: `falkor-chat/REVIEW.md`, `## Always check`. Both are additions. No line in that file is now enforced elsewhere, so there is no prune.

`tests/test_query_copies.py` enforces line 1 only for the five §14.9 copies, so the rule still changes what a reviewer does for every other copy. F6 extended line 1's wording.

- Sources: F1 and F6 (this review):
  > A `test_queries.sh` assertion runs a hand copy of a repository query, so it pins the copy, not the method. A contract claimed for the method needs a pytest over the real method that a mutant of the method turns red. A check that a copy matches the method reads the copy's value as bash evaluates it, from exactly one definition, never by parsing the source.
- Source: F2, mutant ML (this review):
  > A property added to a query that has alternative literal prefixes or branches (`$ingestedBy` / `$producedBy`) is asserted as written on every branch, not only on the default one.
