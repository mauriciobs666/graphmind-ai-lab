# python-web-quirks on the knowledge base — implementation review (G1)

> **Status:** active · **Owner:** `analyst` · **Tracks:** — · **Reviews:** `docs/plans/python-web-quirks-kb.md` (S1–S3b; S4–S8 in Pass 3)

## Pass 1

### Scope and verdict

**Reviewed:** `claude/cobb/scripts/pwq_kb.py` (1,437 lines), `pwq_kb_selftest.py` (1,596),
`check_content_loss.py` pwq mode + `check_content_loss_selftest.py` + `fixtures/pwq_*`, diff
`a5215c45..HEAD` limited to `claude/cobb/scripts` (commits `5064db3b`, `5c855e30`, `b0c27709`,
`92a46502`; no later change to those paths), against plan §3 D1–D11, §4 S1–S3b/G1, §5 T1–T30, ML note,
requirements. Root `REVIEW.md` read; `claude/REVIEW.md` does not exist. Not reviewed: S4+ (nothing exists yet),
`pwq-golden-set-a.json` (S5, absent), the copy of record (S4).

**Verdict: approve with suggestions.** No blocker: every production-safety guard held under mutation, the
metrics match the ML note, and the independent sweep found all 36 units intact. Two majors must close
before the step they gate: **F1 before S4** (S4's "review every header" has no mechanism to act on a finding)
and **F2 before S8/S14** (the one-shot Set B seal can be burnt by a malformed Set B).

CPG: considered, not relevant — stdlib scripts and docs; no CPG graph for `claude/cobb/scripts`, and the checks were traced in source and by mutation.

**Observed this run (figures only):**
- `python3 claude/cobb/scripts/pwq_kb_selftest.py` → `98/98 passed`, `ALL PASS` (2.1 s).
- `python3 claude/cobb/scripts/check_content_loss_selftest.py` → `ALL PASS (22/22)` (`check_content_loss.py` has no `--selftest` flag; the selftest is that file).
- `units --pin 3fcb171c`: 10 hits, 36 units from 42 sections; `--chunks`: 129 chunks, 12 units with ≥ 4 chunks (inside the plan's 127–134 / 11–12); header sizes 157–949; rendered copy 85,641 bytes, listing 6,619 bytes (bound 10,000).
- `render-copy` then `verify-copy` at the pin: `clean`. `check_pwq` over that copy with a manifest built from it: 44 scopes, 0 dirty, in both `source` and `copy` source-formats.
- Mutation pass (scratch `git clone --local` under the session scratchpad; the repo was never touched): 57 mutants, 48 killed, 9 survived (list in F3, F8, F9; two survivors are equivalent).

### Findings

**F1 — major — a `Versions`/`Verified` header cannot be corrected, and the extractor has two real misses at the pin.**
Verified (ran). D3 says a missed pin "is a `cobb` review judgement" and "`cobb` rules any disputed date at S4";
S4's done-condition is "review every header, write the copy, `verify-copy`". But `verify_copy` recomputes
every header and compares equality (`pwq_kb.py:477-483`), and the extractor is the only source. I rendered the
copy, hand-added `starlette 1.6.0` to `Versions[3]` of `u21`, and `verify-copy` returned
`FAIL: u21: header mismatch` (rc 1). So the only way to act on S4's review is to change `VERSION_RE` /
`extract_dates`, after which every other unit's header changes too. Two concrete misses exist at the pin
(coverage probe: every `x.y.z` token of every section versus its unit's `Versions`; all other uncovered tokens are
IP addresses): §23 (`u21` member 3) reads `starlette **1.3.1** (...) and **1.6.0**` and `CPython\n3.12.3`;
the header says `Versions[3]: starlette 1.3.1; httpx 0.28.1`, dropping the one pin that the gotcha is about
(two venvs, two Starlette versions, `CPython 3.12.3`). Causes: `[ \t]*` between lib and token (`pwq_kb.py:271`)
cannot cross the source's hard line wrap; a bolded list continuation (`and **1.6.0**`) has no lib before it.
Suggested fix, in a round *before* S4: (a) widen the pattern to `\s*` and add the `lib **a** (...) and **b**`
continuation form; (b) give `verify-copy` a ruled-override table (data in the script, keyed by unit and member,
like `RULINGS`) or relax the rule to "header equals the recomputed pairs plus overrides, each override still
passing T6's substring check"; (c) a selftest fixture per axis the pin varies on (wrapped pair, bold,
list continuation, comma list) instead of a list of the two found shapes. Library allow-list (a plan silence):
at the pin the 13-name list is complete (the only non-listed `<word> <version>` hits are `in **0.2` and similar
non-libraries), so the defect is the adjacency form, not the list; still, a new library in a future unit is
silently unpinned — name that in the plan as a `cobb` review item at each add.

**F2 — major — Set B's shape is not validated before the irreversible seal is burnt; coverage is relative to the set.**
Verified (read + mutant). `ac3_verdict` tests `covered_sections == n_sections` where `n_sections = len(by_section)`
is derived from whatever sections the loaded set mentions (`pwq_kb.py:1003,1021`); `load_set` checks only field
types. A Set B with 41 sections, or 83 positives, or 16/14 off/near, scores "coverage ok" and writes the sentinel
(`pwq_kb.py:1209-1216`); ML §1's gate needs 84+30, 2 per section across all 42, 15/15 split. The mutant
`"coverage": True` was killed, but nothing pins the denominator. Suggested fix, in a small round *before* S8 (not
inside the sealed step: authored there, its declared set would contain the new value from birth and could never
redden): a `validate_set(kind, data, sections_of_pin)` that `--probe` also runs (S8 learns of a malformed B before
freezing) and `--sealed-once` runs before the sentinel; mutants: B missing one section, B with 85 positives, B
whose `kind` split is 16/14, A with 41 positives must each refuse (exit 2, no sentinel).

**F3 — major — the per-section coverage rule ("hit by at least one of its two queries") is unpinned.**
Verified (mutant). Replacing the any-of-two accumulation (`by_section[s] = by_section.get(s, False) or hit_fl`,
`pwq_kb.py:985`) with last-query-wins survived all 98 tests; T25 covers merged-unit membership, not this clause,
which is a named AC-3 gate clause. Fix: three planted cases in the selftest: first hits/second misses, first
misses/second hits (both covered), both miss (uncovered); same for `by_section_raw`.

**F4 — major — ruling on the checker and the manifest fields: `pin` must be tied; `sha256` and `migratedOn` are write-only.**
Verified (grep: nothing reads them). `check_content_loss.py` takes the source as a positional *file*; nothing ties
that file to `manifest.pin`, so AC-1 / S13's "AC-1 clean at the pin" can pass against a drifted working-tree file
(the skill stays editable until S16d). Ruling: (1) `pin` — **yes, verify**: add `--pin`, read the source with
`git show <pin>:<path>` (or compare `git hash-object` to `git rev-parse <pin>:<path>`) and fail unless equal to
`manifest["pin"]`; mutant: a one-character-edited source file. (2) `sha256` — **not in the checker** (it proves
parts against the source, which is stronger), but the field is dead as written: `diverge` hashes both sides
itself (`pwq_kb.py:825`) and never reads it. Either make `diverge` also report a manifest whose `sha256` differs
from the copy unit's hash (registry staleness) or drop the field; do not leave a write-only integrity field.
(3) `migratedOn` — drop it, or have `verify-copy`/`diverge` cross-check it against the note's `Migrated:` line.
Related, same checker: `LIVE_FETCH_SNIPPET` fetches by `documentId` with no `currentVersion` filter
(`check_content_loss.py:196-199`), so a superseded document satisfies AC-1 while the current one differs; filter
or report `currentVersion` in pwq mode (the `diverge` fetcher already filters).

**F5 — minor — the copy side has no id/title uniqueness or id grammar; `verify-copy` is valid only at the pin.**
Verified (ran). Appending a second `u01` block to the rendered copy: `copy_entries` returned 37 entries with no
error; only `verify-copy` noticed, and only because the id list differs from the pin's. After the first legitimate
add (T36/D8: "fresh, never-reused id") `verify-copy` can never pass again (`unit ids/order differ`), and
`ingest`/`diverge` accept any id. Suggested: `parse_copy` rejects duplicate ids, duplicate titles and ids not
`^u\d{2,}$`; state in the plan that `verify-copy --pin` is the migration check, and let it accept units beyond the
pin's set (recomputing their headers) so it keeps working after an add. Two-digit ids (plan silence): fine to u99,
no tool breaks at u100.

**F6 — minor — the target guard's premise (URL ↔ workspace) is never checked by the tool.**
Inferred from `check_target` + D8's own note that `/health` does not report the workspace. If `:8210` were
started with `FALKORCHAT_WS_ID=agent-team`, every "test" `ingest`/`replace`/`retire` would act on production
with all guards green, and the fetcher (host/port from `FALKORDB_HOST`/`FALKORDB_PORT`, not from the table)
could read a different instance than the server writes to. Suggest: after the first POST on any target,
`fetcher.current_docs(ws, [], [receipt id])` must return it, else stop; and S6b must prove it (POST a probe
document, see it in `ws:pwq-test` and absent from `ws:agent-team`).

**F7 — minor — `gate` is clean on a partial manifest.**
Verified (read). A unit that `ingest` refused and that has no manifest entry is simply absent from
`manifest["units"]` (`pwq_kb.py:720-735`), so `gate` prints clean for 35 units (`ingest` exits 1, but the two
commands are run separately). Mitigated at S13 by the checker's heading-set completeness, but the K-102 gate is the
FR-4 report. Suggest `gate`/eval compare manifest ids with the copy's unit ids and fail on a difference.

**F8 — minor — harness details.** (a) `distinct_docs_mean` pools positives and negatives
(`pwq_kb.py:990,996`); ML §7 risk 2 is slot crowding on positive queries; report it for positives only, or both
separately. (b) BM25 is fed the raw query (correct); the mutant that prepends the retrieval prefix survived — pin it
(T30 only pins determinism). (c) `--probe` fetches with the window; the mutant `window=5` survived. (d) The
sentinel is keyed to the Set B *path* (`<path>.scored`); a copy of B at another path scores again, and
`--set A --set-a-path <B path>` prints per-query MISS lines for B. The seal is procedural by design (D11), so record
the sha256 the sentinel stores and have S14 compare it with the ledger's frozen sha256; no mechanical fix needed.
(e) `--manifest` is mandatory for any non-production run even for `--probe`/`--before`, which read none; S8/S9
briefs must say "pass a scratch path".

**F9 — minor — checker mutants that survived.** `heading_mismatch` off and `strip()==` → `in` for the delimiter
both survived the 22 checks; the first is largely redundant with `not_found` plus completeness, the second admits a
header line that merely quotes `<!-- body -->`. Also `dict(parse_sections(...))` (`check_content_loss.py`, pwq
branch) collapses two sections of the same heading (checked: it fails safe, reporting the duplicate's body as not
located). Take or leave.

**F10 — nit — the cross-reference probe is line-based** (`find_hits` runs the regex per line). I re-ran it across
newlines: no multi-line hit exists in the pin, so it is a latent limit for a future pin, not a defect now.

### Independent cross-reference sweep (from the SKILL.md at `3fcb171c`, 76,291 bytes, 42 sections)

Done from the source, not the implementer's table: (1) the plan's vocabulary at 40 and 80 characters, per line and
across newlines; (2) a wider vocabulary (`above|below|earlier|previous|preceding|following|next|elsewhere|sibling|
companion|half|ahead of|§N|section N|see …|complement|related|second face|same trap|…`), every hit read;
(3) the tool's own `units`. **Result: all 36 units keep every cross-reference intact; no missed referent.**
- Inter-section, all ruled and merged: 587 (§22→§21), 624/633 (§23→§21,§22), 670 (§25→§24), 891 (§34→§2), 951
  (§39→§29), 974 (§40→§30). Semantic cousins without a direction word resolve the same way: "Second face of the
  class the section above belongs to" (587), "Complements the same-second section above" (951).
- Intra-section: 660, 744, 754, 172, 649 (`above the \`with\``, literal code), 962, 657 "Same trap in reverse", 409
  "Related trap in the same code path".
- Outside the file: 37, 260, 420 (asymmetric-judge note), 439, 861, 987, 999. Not a reference: 916 "earlier-reported".
- The widened window (80) adds only two §26 intra hits (753, 754). Of 11 merged sections, none is a referent of a
  section outside its unit.

### Plan conformance (S1–S3b done-conditions)

| Step | Verdict | Evidence |
|---|---|---|
| S1 (T1–T9) | met, with F1 | all listed functions/subcommands exist; selftest `t1`…`t9c` present; the unruled-hit build failure is exercised (`t9c_units_cli_fails_on_an_unruled_hit`) |
| S2 (T10–T15) | met, with F4/F9 | changes 1–6 present; old behaviour unchanged (22 = 16 old + 6 new); the real `LIVE_FETCH_SNIPPET` runs against a fake `falkordb` module for `--ws` |
| S3a (T16–T23) | met, with F6/F7 | guard order checked before any request; `replace`/`retire` refuse production (the second refusal in `_test_only_unit` is equivalent to the client's, both fire before any request); production `ingest` needs the flag; the fake derives its fields from `schemas.py` |
| S3b (T24–T30) | met, with F2/F3/F8 | all D9 modes present; `--set B` accepts only `--probe`/`--sealed-once`/`--baseline`; `--sweep`/`--ac2`/`--before` on B are refused |

**Production-safety guards (traced, then mutated):** target guard (URL mismatch, unknown ws, missing
`--confirm-production`) — mutants killed; production never deletes (client guard and `_test_only_unit`); non-production
runs refuse the default manifest; Set B path refusal (`resolve()` follows symlinks into the repo) — killed; sentinel
pre-check and `O_EXCL` write — killed; gate runs before every scoring mode and not for `--probe`/`--before` — killed;
exit codes 3 (target), 4 (gate), 5 (sealed) — killed. The sentinel is written after the metrics exist and before
anything is printed, and a failed gate burns nothing.

**Eval metrics against the ML note (traced + mutants):** window 50 then exclusion then first five
(`scored_chunks`; `limit=5+|excluded|` and exclusion-after-truncation mutants killed); `score <= floor`; recall gate
`floor_hits*10 >= n_pos*9` (76/84 passes, 75 fails) and near `*5 >= *4` (12/15), both boundary mutants killed; MRR
over first expected chunk (raw, informational); false accept counts only feature documents (killed); sweep 0.38–0.50
in 13 steps with the ML §6 criteria (3-point recall gap, 0 off-domain, ≤ 2/7 near, 0.025 margin; margin and near
mutants killed); BM25 k1=1.5, b=0.75, deterministic tie-break on key (killed). Open: F2, F3, F8a.

### Live paths beyond the fakes

- `urllib_transport`: real loopback HTTP listener in T23 (port asserted ≠ 8200/8210).
- `extract_prefix` ran against the real `skills/agent-kb-retrieval/SKILL.md` literal: correct template.
- `FETCH_SNIPPET`: in the selftest only compiled and substring-checked (`t23_real_fetcher_snippets_compile…`). I ran it
  read-only against `ws:agent-team` (stdin JSON through `cypher-mcp/.venv`): `counts` returned
  `{"chunks":2,"embedded":2,"status":"ready"}` for a known document and `null` for an unknown id; `docs` returned the
  document with title, text, status. So both queries are valid on the live graph.
- `RestSearcher.search/get_document`: fake transport only in the selftest; run read-only against `:8200`: `limit=50`
  returns 50 rows with keys `chunkId, documentId, score, seq, text`, ascending by score; `get_document` returns
  `title`, `text`, `status`. The server ties `k = limit * 3`, so the window-50 pool differs from the convention's
  `limit=5` pool in ANN depth only.
- **Never exercised live:** `POST /documents` (201, receipt keys, `producedBy=cobb` resolving to an `Agent` in the test
  workspace), `DELETE`, the gate during real embedding, the `:8210` ↔ `ws:pwq-test` mapping, `GRAPH.COPY`.

**What S6a/S6b (the live test run) must cover, on `:8210` only:** (1) `POST` of one unit with `producedBy=cobb`: status
201, `documentId` in the receipt, `cobb` seeded in `ws:pwq-test`; (2) the stored `Document.text` byte-identical to the
posted text, through `CypherFetcher.current_docs` (trailing newline, non-ASCII such as `⊂`) — otherwise `ingest`
refuses and `diverge` reports STALE forever; (3) re-POST of identical text auto-supersedes and the fetcher then sees
exactly one current document; an edited text lands as a pending suggestion — record whether that suggestion is
`currentVersion` (it drives the `differing` refusal); (4) `gate` while embeddings land, and with one planted failed
document; (5) `DELETE` response shape (the client expects HTTP 200) and a `replace`/`retire` round; (6) F6's mapping
proof; (7) `eval --probe`/`--before` against the test server with a scratch `--manifest`.

### What's solid

Guard ordering and refusal semantics; the `RestClient` has no production delete at all; the fake REST server derives
its accepted fields from `schemas.py` rather than from the code under test; the AC-3 arithmetic is integer-exact at
both boundaries; mutation coverage of the safety paths is high (every guard, sentinel, gate and exit-code mutant
killed); the unit derivation is robust at the pin; `check_content_loss` pwq mode keeps the old CLI byte-compatible and
fails safe.

### Open questions

1. F1(b): override table or relaxed `verify-copy`? Either needs `architect`/stakeholder sign-off because D3 says
   "`cobb` rules" without a mechanism.
2. F4(2)/(3): drop `sha256`/`migratedOn` or give them a reader?

### Growth-stops check

New document at its first pass; no prior revision to compare, so no growth to classify. Findings cite plan sections
and IDs only; nothing restated from the plan beyond the cited values.

### Proposed REVIEW.md lines

Target: root `REVIEW.md`, "Always check". Source: F2 (and F3).
- Add: "A harness with a one-shot seal (a sentinel, a burnt hold-out) validates the shape of the sealed input (counts,
  strata, coverage denominator) before it burns the seal, and the denominator of a coverage clause is a constant of
  the spec, never derived from the data under test."
No prune proposed: nothing in the file is enforced elsewhere.

## Pass 2

Scope: commit `e0b84854` (the only commit after `a482c2ac` touching `claude/cobb/scripts/`) against `92a46502`,
in a fresh `git clone --local` under the scratchpad; repo untouched. Observed: `pwq_kb_selftest.py` 123/123,
`check_content_loss_selftest.py` 25/25. My own mutants: 25 killed of 26 (list below); Pass 1's survivors
re-run (coverage-last, BM25-prefix, probe-window) are now killed.

**Verdict: approve with suggestions.** No blocker, no major remains. F1 and F2 are closed, so S4 and S8 may proceed.

### Dispositions

- **F1 fixed (ran).** `render-copy` now emits `Versions[3]: CPython 3.12.3; starlette 1.3.1; starlette 1.6.0; httpx 0.28.1`;
  `verify-copy` clean. At the pin the new extractor adds exactly those two pairs and removes none (diffed old vs new over
  all 42 sections). Dropping an extracted pair fails ("header omits extracted version"); an invented pair, and a pair valid
  only in another member's body, fail. A hand-added pair that passes the member's own body is accepted.
- **F2 fixed (ran, all three modes).** With a 82-positive, 85-positive, 16/14 or duplicate-id B, `--probe`,
  `--sealed-once` and `--baseline` each exit 2 with 0 searches and no sentinel; a well-formed B proceeds to the next
  check. Validation runs before the sentinel pre-check and before the gate; the constants are spec constants.
  Residual: it does not check the `tail` flags, so S8's "tail probe for every ≥ 4-chunk unit" stays on the S8 brief.
- **F3 fixed** (not in the implementer's list): last-query-wins mutants on `by_section` and `by_section_raw` are killed.
- **F4 fixed, one limit.** The pin tie works (matching file `None`; one-character edit, non-sha pin, and a file outside
  the repo each refused with a reason). The `currentVersion` filter is valid on the live graph (read-only run: a known
  document returns both ways); no superseded document exists to show the negative live, so that rests on the fake.
  `diverge` now reads `pin`, `migratedOn` and `sha256` (planted stale pin and sha: two `REGISTRY-STALE` rows). Limit: the
  tie needs the source *inside* the repo; after S16d deletes the skill, re-running AC-1 needs `--allow-unpinned`, which
  switches the tie off. Minor: add a `--pin-from-git` source mode.
- **F5 fixed (ran).** Duplicate ids and titles raise; the 36 real rendered titles are unique, and `verify-copy` reports the
  error instead of crashing. Pass 1's other F5 points (id grammar `^u\d{2,}$`; `verify-copy` valid only at the pin's unit
  set, so it fails after the first legitimate add) are not addressed: still minor.
- **F6 not addressed** (not claimed): URL↔workspace proof stays with S6b. **F9, F10 not addressed:** take-or-leave.
- **F7 fixed (ran).** A 35-unit manifest against the 36-unit copy gives `u42 - - ABSENT`, `gate: FAILED`, rc 1; an
  unreadable `--copy` gives rc 2. Scoring modes pass the copy too.
- **F8 (a)(b)(c)(e) fixed or pinned** (distinct-docs mutant, BM25-prefix and probe-window mutants killed). **(d) ruled below.**

### Same-revision interactions

1. **`version_pair_ok` extension + superset rule — latent, minor.** An extractor false positive can never be omitted
   (omission fails) or corrected. None exists at the pin, but the extractor reads `pytest 3.5` from "pytest 8.2, 3.5 times
   slower" and `fastapi 12.50` from "fastapi 0.1.2, 12.50 percent" (synthetic probes); the unit blocklist is only
   `ms|s|sec|seconds|MB|KB|%`. It bites only on a unit that is re-rendered by the extractor from a source (a later pin),
   not on hand-added units. Suggest an exclusion table keyed like `RULINGS`, applied by both render and verify. The
   extension itself is a no-op on the superset path (extras are never extractor reads); one mutant (member-specific body →
   member 1) survived. Nit.
2. **F7 contract change vs S9/S13 — consistent, document it.** `--set A --before` (S9) and `--probe` return before the
   gate and never read the copy, so S9 and S8 are unaffected. S10, S12, S13 and S14 all run after S4, when the default
   copy path exists. The plan (D9, T21) says nothing about the copy, so add one sentence to D9: "`gate` and every scoring
   mode read the copy's unit ids and fail on an unregistered unit". The T36 retire step must remove the unit from the copy
   too, or the next gate correctly fails `ABSENT`.
3. **`diverge` registry check vs step order — consistent.** `ingest` rewrites the manifest (`pin`, `migratedOn`, per-unit
   `sha256`) on every run, including all-skip runs, so S10/S13/T35 produce a fresh registry, and `replace` rewrites its
   unit's `sha256`. In the D8 edit flow (copy edited, then `diverge`, then stakeholder `curl`, then `ingest`) the stale unit
   shows twice (`STALE` and `REGISTRY-STALE`) until `ingest`: noisy, correct.
4. **F5 duplicate titles vs real titles — consistent.** 36 of 36 unique, and the title is the first member's heading.

### Rulings

- **F8d (sentinel keyed on path plus content hash): skip accepted.** The sentinel already stores `setSha256` of the scored
  file, and any agent with Bash can delete any sentinel, so a better key adds no mechanical protection (D11: the seal is
  procedural). The real gap is different and cheap: `--sealed-once` records the file's hash but never compares it with the
  hash frozen at S8, so a B edited after freezing is scored silently. Suggest `--expect-sha256 <ledger value>`, checked
  before the sentinel; S14's brief passes it. Minor.
- **Set A shape validator: not needed for the seal, but add a light one.** A has no seal and is meant to be re-run. But
  `n_sections` is derived from the set for A too, so a partial A reports coverage "ok", and A's numbers decide the floor at
  S12. A `validate_set_a` (42 positives, one per section; 14 negatives, 7 near / 7 off) in the same style as B, applied to
  the scoring modes (not `--before`), closes it. Minor; S5's "42+14 rows" done-condition covers only the file, not the tool.

### Mutants run this pass (scratch clone)

`pwq_kb.py`: omit-rule off, invented-rule off, member-body → member 1 (**survived**), per-section count, section set,
negative count, duplicate ids, validation skipped for probe / baseline, ABSENT rows off, eval gate without copy ids,
CLI gate without copy ids, `diverge` sha off, pin off, migratedOn off, duplicate id / title off, distinct includes
negatives, continuation off, line-wrap off, continuation accepts units. `check_content_loss.py`: pin check off,
`currentOnly` false, `WHERE` dropped, hash comparison ignored. All others killed.

### Growth-stops check

Compact pass: one disposition line per Pass 1 finding; the Pass 1 text is unchanged. New material is the four interaction
checks and two rulings (new scope). No restated facts.

## Pass 3 (G2: S4–S8)

### Scope and verdict

Reviewed: the copy `claude/knowledge/python-web-quirks.md` (commit `10dd7c63`) against pin `3fcb171c`; the guard diff; Set A (`pwq-golden-set-a.json`), `pwq_leakcheck.py`; `docs/plans/python-web-quirks-kb-graph.md` (S6) against `ws:pwq-test`/`ws:agent-team` (reads only); `docs/plans/python-web-quirks-kb-captures.md` and commit `c98648a6` (S7); the ML note's S7b rulings and recomputed figures against `pwq_kb.py` (U14/U15 code, working tree equals HEAD); `docs/test-plans/python-web-quirks-kb.md` and the ledger's U17 row (S8). Set B was not read and not looked for. Root `REVIEW.md` applied; `claude/REVIEW.md` does not exist. No write to any FalkorDB or `ws:agent-team`; a scratch `git clone --local` for mutants.

**Verdict: approve with suggestions.** No blocker. One major (F11, before S12) and several minors/stale sentences. S9 may start.

CPG: considered, not relevant: stdlib scripts, docs and data; checks traced in source, by mutation and by read-only queries.

**Observed:** `pwq_kb_selftest.py` 130/130, `pwq_leakcheck_selftest.py` 11/11, `check_content_loss_selftest.py` 25/25; `verify-copy --pin 3fcb171c... --copy ...` clean; `units --chunks` 129 chunks, 12 units with 4+; chunks over the copy's own entries also 129/12; listing regex 80 lines, 6,619 bytes; `pwq_leakcheck.py --pin` 42 checked, 0 shared 4-grams. Scratch mutants on U15 code: tail-flag check off, Set A validation off, sha check off, exclusion-after-truncation: 4 of 4 killed (130 baseline, 128/129 after).

### Dispositions of Pass 1/2 items touched by this range

- F1/F2/F3/F4/F5/F7/F8 (a)(b)(c)(e): remain fixed (Pass 2); selftests green.
- Pass 2 minors: Set A validator **fixed** (U15, mutant killed); `--expect-sha256` **fixed** (U15: missing, non-64-hex, upper-case and a wrong hash each exit 2 with no sentinel; run with a scratch Set B); tail-flag check **fixed, sealed runs only** (see stale sentences); D9 `--copy` sentence **fixed** (plan line 258); T36 retire-from-copy **fixed** (T36); `--pin-from-git` **assigned to S16d**; extractor exclusion table: **not addressed** (latent); F6 **not addressed** (see F16).

### Findings

**F11 — major — the "duplicate coverage" reading is wrong for most of the 36, so S7's exclusion list does not make the scored state the post-H1 state it is described as.**
Verified (ran). The S7 list covers 11 sections, so at most 22 of 84 B queries can be explained by it, yet 36 of 84 have a rank-1 under 0.43 before ingestion (test plan §5, ledger U17). I ran `eval --set A --before --floor 0.43` read-only on `:8210` with a scratch `--manifest`: 27 of 56 queries (26 positives) have a hit under the floor; of the 26, only 9 have a best document on the S7 list, 17 do not (e.g. A01 at 0.4107 hits `eb6e3699...` about a manual sweep, unrelated). So most are unrelated team documents under a loose floor (ML §2's xdist probe, §6), not restating captures. Consequences. (1) Validity of the sealed score: it cannot inflate recall (a hit counts only for the unit's own document; false accepts count unit documents only, `evaluate` lines 1155-1171); the risk is deflation if a non-unit chunk displaces a unit chunk from the five. Units score about 0.23-0.35 in the probes, so displacement by a 0.41 chunk is unlikely but unmeasured. (2) ML §4.2 and D11 say the excluded run "measures the post-H1 state": true only if every excluded id is actually deleted at H1 (the stakeholder may delete only the four of S7 §1), and false for the 17 non-listed documents, which H1 never touches.
Fix: (a) S12 (Set A, inspectable) reports per positive the rank of the first unit chunk and whether any non-unit chunk scores below it, aggregated, with the named documents; if none ever outranks a unit chunk, the 36 are inert and recorded as such. (b) S14 passes `--exclude` only ids the ledger records as deleted at H1 or created after S13b, and reports the same aggregate "non-unit outranks unit" count (no per-query lines; B stays sealed). (c) Re-word "duplicate-coverage" in the plan, ML §5 and the ledger to "pre-existing hit under floor"; restating captures are a subset. (d) `--sweep` prints no unfiltered numbers, contradicting ML §4.2/D11 (`run_eval`, sweep branch returns before the filtered/unfiltered print at lines 1405-1412): print them, or edit the sentence.

**F12 — minor — stale plan sentences (cross-document consistency).** Verified against code and ledger:
1. S8 row: "`validate_set_b` does not read the flags" is false since U15: it does, on `--sealed-once` only (`run_eval` 1308-1313); `--probe`/`--baseline` still do not.
2. §2 (line 36-38): "an edited text lands as a *pending* suggestion" contradicts S6b: an edited re-POST is a second current document, no pending edge (re-checked: no `SUPERSEDES` row in `ws:pwq-test`; `lshBand0` on 0 of 482). D8's refusal still holds; the premise is wrong.
3. D11 (line 314-315): "`eval` reports recall with and without them (unembedded chunks)": no such mode exists in `pwq_kb.py`; the gate only refuses. Delete or build.
4. §2 (line 40-42): "header comment lines 82-93, glob list line 105" and "the one stale entry": the entry was retired at S4.
5. §2 (line 56): "250-400 char header": real headers are 157-982 characters (`units --chunks`).
6. G2 row done-condition "Pass 2": this is Pass 3.
7. D9 never states the Set A/B shape validators exist (an omission: the F2/F3 contract lives only in ML §3/§8 and the code); D2/D11 still use the pre-build names `u-bt`/`u-tc`/`u-st`/`u-pyc`/`u-in`, while code, copy and Set files use `u02`/`u21`/`u24`/`u29`/`u30`.
The ML note has its own: §2 line 42 "476 documents, 767 chunks" (now 483/775, in `ws:agent-team`); §4.2 "the 4 known captures cannot" (17 documents, 23 chunks in `ws:pwq-test`; the conclusion holds, the count is stale); §3 "Today the tool has no such check for A (code change owed, §8)" and the §8 "Code owed" bullet (all three delivered by U15).

**F13 — minor — the test plan.** (a) `TP-012` is cited (exploratory charter) but has no row; define it or drop the id. (b) S15c is qa-engineer's step for AC-6, AC-10, AC-13 (plan §4), yet the test plan puts them out of scope (§1, §7) and nothing else holds a plan for them: add rows or say that plan T39 is the contract. (c) The S14 pre-check order differs from the code: `run_eval` checks the sha first (exit 2), then Set B shape and tail flags (loads the production manifest, exit 2 if absent), then the sentinel (exit 5), then the gate (exit 4); the plan lists gate before sentinel and names no exit codes. Add: a hash mismatch is exit 2, not 5, and burns nothing; the manifest and `--copy` must be readable; `--exclude` must be a subset of the ledger's deleted ids (F11). Verified otherwise: no query text appears; AC coverage is complete across the plan, with AC-1, -4, -6, -9, -10, -13, -14 explicitly owned elsewhere.

**F14 — minor — F6 is still open and the S6 note reads as if it were closed.** `check_target` is string equality on `--base-url` (`pwq_kb.py:579-588`); nothing verifies after the first POST that the receipt id is visible in the target workspace (grep: no such check). S6's proof is for this launch only. Note `localhost` vs `127.0.0.1` is refused (safe). Either implement the post-receipt check for S13, or keep it a procedural item in the S13 brief ("POST, then see the id in `ws:agent-team` and nowhere else").

**F15 — nit — figures.** S6 says "14 failed + 1 processing" documents: observed (both graphs identical) 17 `failed` documents and 1 `processing`; 14 of the 17 hold unembedded chunks, 3 are fully embedded; unembedded chunks total 16, matching. The captures document cites line numbers as `§803`; readers will take them for section numbers. The ledger's "Next:" and "PAUSED" notes are stale.

### Answers to the specific checks

- **S4, copy vs pin:** `verify-copy` clean. Independently (own parser, whitespace-normalised): 42 of 42 source sections found as copy members, 36 units, 0 differences, including u02, u21, u24, u29, u30; preamble block equals the source H1 and scope paragraph, with the waived blockquote in neither. Headers: every ISO date in a unit's body is in `Verified`/`Other-dates` (all 36 units); every `x.y.z` token is in `Versions` except IP addresses (u09, u13). u10 `pydantic 2.x` is a body substring (accepted by the superset rule); u02 `uvicorn 0.52.x`, u21[3] (the F1 fix) and dated (u05 `2026-09-08`) and undated (u03 `unverified, migrated 2026-11-10`) units are right. The u10 edit lives only in the copy: a fresh `render-copy` emits `none stated`, which `verify-copy` also accepts; chunk counts are unaffected.
- **S4, guard:** allow for `claude/knowledge/python-web-quirks.md` (relative and absolute); ask for `skills/python-web-quirks/SKILL.md` and `docs/BACKLOG.md` (both forms); `bash -n` ok; `git show 10dd7c63 -- claude/cobb/hooks` removes only the `skills/python-web-quirks/*` comment block and glob pair.
- **S5:** all 42 positives and 14 negatives equal ML §3 (queries, sections; kinds 7/7); leakcheck 0; selftests green.
- **S6:** `ws:pwq-test` 482 documents / 773 chunks / 757 embedded, 0 `PWQ-PROBE`, `:8210` and `:8200` both ok. `ws:agent-team` is 483/775/759: the delta is exactly one document, `458d329e...` (graph-dba's own learning capture, 2 chunks, not a gotcha), which the plan's exemption allows; S6 changed nothing else. The target table maps `agent-team` to `:8200` and `pwq-test` to `:8210`; mismatches refuse (F14 for the limit).
- **S7:** sampled 6 listed captures by `get_document` (`1bfe7423`, `3ab71d82`, `c590ed3a`, `35b0029b`, `60aba8db`, `629325c2`): each restate their named section; `35b0029b` states the general unknown-route form of §13 (accepted). A keyword sweep (about 70 terms) returned 36 further current documents outside the list; read three: `297b3ed4` (max/min NaN), `3074a9fa` (`-k` vs `-m`), `e3c6b8fc` (PID 1 SIGTERM) are related but distinct gotchas, correctly unlisted; the other 33 were not read (a bound, not a proof). The 17 ids in §1+§2 equal §4's list; 11 sections is the count of the listed sections; K-063 item 3 (973 characters matches the plan) and `GAL-I02-K0001` (registry line and quote present in the copy) match commit `c98648a6`, which touches only those. Listed `c590ed3a` is `processing`, three more are `failed`: unembedded, inert for vector scoring, still deletable at H1.
- **S7b:** every ML figure re-derived equal (129 chunks and distribution, 12 units and their list, 83,751 / 835 / 1,723.5 / 6,554, 73,825 / 668 / 1,416.5 / 6,150). Ruling (i) is `LEAK_SCORE=0.25` with `LEAK?` printing; ruling (ii) is `scored_chunks`; both as stated.
- **S8 (aggregates):** ledger U17 and the test plan agree (84+30, 15/15, 12 tail units, 0 shared 4-grams, 2 positives under 0.25 kept per ruling (i), 36 kept, sha256 `163efb06...fd15c`). Kept ruling (i) positives are consistent with F11: the two Set A analogues under 0.25 (A41, A42) hit listed captures.
- **14 failed + 1 processing documents:** do not touch any scored number or the gate: the gate and `feature_docs` are unit documents only; a document with an unembedded chunk cannot rank by vector; BM25 runs over the copy's units. Failed documents with all chunks embedded can rank but are never counted.
- **S12 with the S7 list excluded vs H1 timeline:** the 17 ids exist in `ws:pwq-test` (23 chunks, 20 embedded), well inside the 50-chunk window; `--exclude` is accepted by every mode; the gap is F11 (b)/(d).

### What S9 needs

Nothing blocks S9. Pass a scratch `--manifest` (a non-production run refuses the default; my `--set A --before` run did, and wrote no file). Owed: F11 before S12; F11(b), F13(c) before S14; F12 plan/ML edits at the architect's next revision.

### Growth-stops check

Pass 3 is new scope (S4-S8); Pass 1/2 text is unchanged. No findings restate plan content beyond cited lines.

### Proposed REVIEW.md lines

Target: root `REVIEW.md`, "Always check". Source: F11.
- Add: "A figure that a ruling labels by cause (for example 'duplicate coverage') is checked against that cause's inventory: if the labelled set cannot account for the count, the label is wrong and the remedy built on it covers only part."
No prune.
