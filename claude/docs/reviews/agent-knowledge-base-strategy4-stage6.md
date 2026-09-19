# K-030 Track 2 Stage 6 — KB-content migration review

> **Status:** active · **Owner:** `analyst` · **Tracks:** K-030 (Stage 6)

**CPG:** considered, not relevant — no `cpg_claude` graph is loaded (`GRAPHS` shows
`cpg_falkorchat` plus the workspace/`kaizen_team` graphs only); the two Python scripts this
migration produced (`check_content_loss.py`, `flag_split_candidates.py`) were read directly and
exercised live against the real corpus, which is a stronger check than a CPG would add for
scripts this small.

## Scope & verdict

Diff-scoped review of the whole Stage 6 migration (`claude/docs/plans/agent-knowledge-base-strategy.md`
§3/§6, Track 2), built across the U1–U3k dispatch chain in
`claude/docs/plans/agent-knowledge-base-strategy4-coordination.md`. Reviewed by independent
execution, not by re-reading `teco`'s summary: re-derived the `ws:agent-team` status tally myself
(`mcp__cypher__query`), ran `claude/cobb/scripts/check_content_loss.py --live` against **all 13**
KB files (not just the 2 largest + spot-checks the brief asked for), reconciled the manifest's own
documentId inventory against the live graph, spot-checked claim fidelity and split-boundary
judgment directly against source `.md` files, and confirmed the 5-document permanent-failure gap
and its filed backlog item (`falkor-chat/docs/BACKLOG.md` K-067).

**Verdict: approve with suggestions.** Nothing found here changes the "Stage 6 is fully closed,
safe to open Stage 7" conclusion — every apparent defect I found while independently re-running
the corpus's own fidelity checker turned out, on manual reconstruction against the source file, to
be content-preserving. But the checker's blind spot is real and was silently unexercised against
the densest file in the corpus until this review ran it; that is worth fixing and recording before
Stage 9 relies on the same tool for its own ongoing delete-then-recreate hook. See Finding 1.

## Findings

### Major — `check_content_loss.py` has an undocumented false-positive mode, and it was never run to completion against `review-techniques.md` before this review

**Evidence.** The coordination doc's own history (`claude/cobb/kaizen/history.md:206-218`) records
exactly one live run of the checker against the migrated corpus, made mid-flight during U3f — at
that point `review-techniques.md` had only headings 1–11 (14 of its eventual 81 claims) migrated.
Headings 12–54 (67 more claims, produced by U3g/U3h) were never checked afterward; no later run is
recorded anywhere in `kaizen/history.md` or the coordination doc. I ran it myself, live, against
the full 81-claim set:

```
python3 claude/cobb/scripts/check_content_loss.py claude/analyst/review-techniques.md --live \
  --documents <all 81 ids>
# -> 11 NOT_FOUND + 3 UNACCOUNTED (14 findings total)
```

and against `qa-testing-techniques.md` (previously touched by a real, fixed defect — U3e's
"and"-drop): 2 more NOT_FOUND findings on the same heading the earlier fix landed in.

**Root cause, confirmed by manual reconstruction (not inferred).** Every one of these 16 findings
traces to the same tool limitation: `locate_claim` requires a claim's *entire* stored text to be
one contiguous substring of the source body. `cobb`'s own documented, deliberate split convention
for two claims that share one context sentence/citation — "shared Origin duplicated in both since
it doesn't split by trap" (manifest note, h22; same shape at h13/h14/h21/h39/h40 and
`qa-testing-techniques.md`'s model-bench-attest heading) — produces exactly this shape: claim A
gets [shared intro/point 1 + its own half of the duplicated sentence], claim B gets [point
2/3/etc. + the *other* half of the duplicated sentence]. In the source file, claim B's content
sits between claim A's two halves, so neither claim's stored text is literally contiguous in
isolation — even though the **union** of A and B, once you strip the duplicated shared prefix,
exactly reconstructs the source with nothing missing, added, or altered. I verified this by hand
for three representative pairs (full reconstruction shown, byte-for-byte):
- `d41d743e…`/`5c2ef405…` (review-techniques.md, "state-machine guard" heading, h13)
- `f0a93c99…`/`af288858…` (review-techniques.md, "git provenance" heading, h22)
- `781c055d…`/`1034d23f…` (qa-testing-techniques.md, model-bench-attest heading — also
  independently confirms the earlier "and"-drop fix is still correctly in place)
The remaining pairs (h14, h21, h39, h40) match the identical manifest-documented shape and were
cross-checked against their recorded titles/notes rather than re-derived byte-for-byte; given 3/3
fully-verified pairs came back clean with the identical mechanism, I have high confidence the rest
do too, but this is inference from pattern-consistency, not independent re-derivation for every
pair — see Appendix for the full finding list if that gap needs closing further.

**Why this matters even though no actual content was lost.** AC-4/AC-5's own bar (plan §6) is "a
scripted diff... run once per migrated file" — for the corpus's single densest, most-split file,
that obligation was silently left half-done (11/54 headings checked, not 54/54), and nobody caught
it because the tool's own review (U3e, `analyst` approve Pass 2) predates the corpus having any
real instance of this split shape to exercise against. The gap was only found because this Stage-6
gate happened to re-run the tool from scratch. Stage 9 (`claude/AGENTS.md`'s Track 2 hook,
delete-then-recreate on every future KB revision) will lean on this same script going forward; a
false NOT_FOUND on a routine future split will either get investigated every time (avoidable
toil) or eventually get rubber-stamped as "probably another false positive" without the manual
reconstruction this review actually did (real risk).

**Suggested fix.**
1. Add this false-positive mode to `check_content_loss.py`'s own `LIMITATIONS` docstring section,
   next to the existing "first occurrence" and "whole-file mode can't attribute to heading"
   caveats — name the "shared duplicated label/Origin split" shape explicitly, with a pointer to
   this review or to `kaizen/history.md`'s h13/h22 pairs as a worked example.
2. Record, in `claude/cobb/kaizen/history.md` or the coordination doc, that
   `review-techniques.md`'s and `qa-testing-techniques.md`'s AC-4/AC-5 obligation is now formally
   closed — by this review's manual reconciliation, not by a prior automated pass — since no
   record of either currently states that.
3. Not required before Stage 7 opens (content is confirmed intact), but should land before Stage
   9's hook leans on this checker unattended for the next real revision.

### Minor — `falkordb-quirks.md`'s own claim-count bookkeeping is off by 3 everywhere it's narrated

**Evidence.** The manifest's `_graph_dba_falkordb_quirks_DONE._note` states "FILE COMPLETE (all 5
headings, 67+16=83 claims)"; the identical "83" figure is repeated in the coordination doc (U3f
row: "`falkordb-quirks.md` complete (83 claims...")") and `claude/cobb/kaizen/history.md:191`
("all 83 documentIds"). The manifest's own nested structure, however, holds **86** unique
documentIds for this file: heading 1–4 (16+1+39+11=67) plus heading 5's own two sub-lists —
`first_3_bullets_by_prior_instance` (3 ids) **and** `remaining_10_bullets_16_claims...` (16 ids) —
which sum to 19, not the 16 the note credits heading 5 with. I confirmed the 3 "extra" ids
(`8b73235f…`, `8180743780…`, `fae4b822…`) are real, `ready`, content-bearing documents in
`ws:agent-team` (e.g. "GRAPH.RO_QUERY routes to read replicas" — genuine falkordb-quirks.md
content), not stray/orphaned entries.

**Why this is minor, not major.** The corpus-wide tally (332 manifest-tracked claims, 327
ready/5 failed) that Stage 6 actually gates on is unaffected — recomputing it from the raw
documentId inventory across all 13 files (using the true 86 for this file) lands on exactly 332,
matching the live `ws:agent-team` count independently. So the global closure claim is correct; only
this one file's own prose self-report drifted, the same shape of error U3i/U3j's "308" correction
was dispatched specifically to fix — it just wasn't caught for this file.

**Suggested fix.** A small, mechanical `cobb` follow-up (comparable scope to U3j) correcting "83"
to "86" (and "67+16=83" to "67+19=86") at its two known sources:
`claude/cobb/scripts/kb-claim-manifest.json`'s `_graph_dba_falkordb_quirks_DONE._note`, and
`claude/cobb/kaizen/history.md:191`. Worth a grep-sweep for any other "83" occurrence describing
this file before closing, mirroring U3j's own verification step.

## What's solid

- **The corpus-wide tally is correct, independently re-derived twice.** `ws:agent-team` itself
  (`MATCH (d:Document) RETURN d.status, count(*)`) gives 329 ready + 5 failed = 334; subtracting
  the 2 confirmed-unrelated kaizen-pilot documents (`4da25923…`, `d1f94b35…` — verified by content
  and by `INGESTED_BY` attribution: one `cobb`, one `teco`, both genuinely raw-capture entries
  predating this migration, not KB claims) lands on exactly 327 ready + 5 failed = 332, matching
  the manifest's own documentId count exactly. `teco`'s U3i re-verification and the "308"
  correction (U3i/U3j) both check out.
- **10 of 13 files' migrations are completely clean** under `check_content_loss.py` — zero
  findings for `plan-authoring-techniques.md`, `statistical-method-techniques.md`,
  `test-design-techniques.md`, `falkordb-reference.md`, `falkordb-quirks.md` (86/86 claims, fully
  contiguous, no findings despite this file's own bookkeeping issue above),
  `ops-quirks.md` (all 12 claims including the 4 permanently-failed ones — content confirmed
  present and correctly partitioned regardless of `Document.status`), `frontend-quirks.md`,
  `estimator-test-fixtures.md`, `coordination-techniques.md`, and
  `guard-testing-techniques.md` (all 19, including its 1 permanently-failed claim).
- **The 5-document permanent-failure gap is exactly as documented and genuinely harmless to
  content.** Live-queried: all 5 have zero (4 of them) or partial (1) `Chunk.embedding`, confirming
  they are real, currently unsearchable via `search_documents`'s ANN path — but `get_document`/
  direct Cypher reads their full, byte-exact text, and the checker confirms that text is correctly
  partitioned against its source heading. Root cause filed and accurately described at
  `falkor-chat/docs/BACKLOG.md` K-067.
- **Split-boundary judgment is sound on every sample checked.** `flag_split_candidates.py`
  re-run fresh against `review-techniques.md` reproduces the recorded 25/54 flagged count exactly.
  Spot-checked "kept whole despite being flagged" calls (heading 1's AST-hash technique; h24/h54's
  shared-Origin whole-heading calls) read as genuinely one coherent claim each, not under-split.
  Every split pair I verified line-by-line partitions its source heading completely — no bundled
  independent facts, no fragment losing context alone.
- **No orphaned, duplicate-titled, or misattributed documents.** `INGESTED_BY` resolves to exactly
  two `Agent` nodes (`cobb`: 333, `teco`: 1) with no unresolved/`User`-coalesced writes; zero
  `SUPERSEDES` edges exist (matches the plan's own delete-then-recreate convention, never the
  auto-supersede path); a title-uniqueness query over all `ready`/`failed` documents returns zero
  duplicates.

## Open questions

- None that block Stage 7. The two findings above are both actionable without further input —
  neither needs a design decision, just a small correction pass (Finding 2) and a docstring/kaizen
  note (Finding 1).

## Appendix A — full `review-techniques.md` NOT_FOUND/UNACCOUNTED list (whole-file run, 81 claims)

```
[NOT_FOUND] d41d743e5c344eccbf46715f54cbab44   — h13 pair A (verified byte-for-byte, false positive)
[NOT_FOUND] 5c2ef4055ff34ea6a1ba5f7eb329ae4c   — h13 pair B (verified byte-for-byte, false positive)
[NOT_FOUND] 138ec31887af425ca7f0bd3eb87817c4  — h14 pair A (matches documented shared-split shape)
[NOT_FOUND] 7fbe0fe06e824180a6bf4937bfe88542  — h14 pair B (matches documented shared-split shape)
[NOT_FOUND] 0158a990da6d46b58b0eea3f32047669  — h21 pair A (matches documented shared-split shape)
[NOT_FOUND] ba1e9b45d83a443c8fde5f94dbe51daa  — h21 pair B (matches documented shared-split shape)
[NOT_FOUND] f0a93c99dace45029538dd2bbb78a390  — h22 pair A (verified byte-for-byte, false positive)
[NOT_FOUND] 5a6763497f2740e8aa21efac72b3a1c6  — h39 pair A (matches documented shared-split shape)
[NOT_FOUND] ccefd08d853247cd941de116946cda5b  — h39 pair B (matches documented shared-split shape)
[NOT_FOUND] af9ffb191cbd4659a77dbf3a1139c289  — h40 pair A (matches documented shared-split shape)
[NOT_FOUND] 5b1b477ff67e4e3b81c57f14899bbe48  — h40 pair B (matches documented shared-split shape)
[UNACCOUNTED] source[24650:31422] — h13's own non-contiguous span (same root cause, not new content)
[UNACCOUNTED] source[39504:42910] — h21's own non-contiguous span (same root cause, not new content)
[UNACCOUNTED] source[94345:100565] — h39's own non-contiguous span (same root cause, not new content)
```

`af288858…` (h22 pair B) located cleanly and is not in this list — it happens to sit last in its
heading's source order, so its own span is contiguous even though its sibling's is not.

`qa-testing-techniques.md`'s model-bench-attest heading: `781c055d0be5468f98dbe6143d5ac551` and
`1034d23fe5b04503b90c53128e7cfb17` (the U3e-fixed claim) — both NOT_FOUND, both verified
byte-for-byte as the same shared-duplicated-`**Technique:**`-label shape, confirming the earlier
"and"-drop fix is still correctly in place (the word "and" is present in the live text).
