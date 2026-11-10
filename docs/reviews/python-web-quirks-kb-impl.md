# python-web-quirks on the knowledge base — implementation review (G1)

> **Status:** active · **Owner:** `analyst` · **Tracks:** — · **Reviews:** `docs/plans/python-web-quirks-kb.md` (S1–S3b)

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
