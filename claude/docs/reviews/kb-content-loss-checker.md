# Content-loss checker (`check_content_loss.py`) — review

> **Status:** active · **Owner:** `analyst` · **Tracks:** K-030 (`claude/cobb/kaizen/plan.md`)

Reviews `claude/cobb/scripts/check_content_loss.py` (+ `check_content_loss_selftest.py`, +
`fixtures/content_loss_test_kb.md`), committed at `a0dfeb4` ("feat(claude): add Stage 6
content-loss checker for KB-migration claims (K-030 Track 2)"). This is K-030 Track 2 Stage 6
Step 4, per `claude/docs/plans/agent-knowledge-base-strategy.md` §6 ("Content-loss check
(AC-4/AC-5's own bar)... a scripted diff of 'source section text' against 'concatenation of its
family's claim texts'"); coordination context is `claude/docs/plans/agent-knowledge-base-strategy4-coordination.md`
unit U3e, which routes this deliverable through `analyst` ("gated"), unlike its sibling U2
(`flag_split_candidates.py`), which the ledger records as reviewed skipped as "trivial/low-risk."
That's the precedent this review follows for where to land: no `claude/cobb/docs/` tree exists,
and `claude/docs/reviews/` is the established home for `claude/`-subagent-authored tool reviews
(e.g. `agent-team-curator-hook-stage5.md`, `mid-run-escalation.md` — bare slugs, no plan doc under
the same slug to pair with, same shape as this tool).

**Scope.** Static review plus direct execution: ran the existing selftest, wrote and ran several
of my own probes against `check_partition`/`locate_claim`/`main()` directly (not just reading the
code) to verify claims made in the module docstring and to hunt for the defect-masking risks named
in the review brief. Did not attempt a live `--live` run against `ws:agent-team` (no FalkorDB
session available in this context); the commit message's "live-validated... surfaced one genuine
fidelity slip" claim is accepted as reported, corroborated by the coordination ledger (U3e row:
"2 real fidelity findings on already-migrated claims relayed to cobb's U3f for fix/judgment").

**CPG: not applicable — this is a two-file Python utility with no loaded Joern CPG for `claude/`,
and the review question is about the checker's own algorithm/coverage, not impact analysis across
a larger codebase graph.**

**Verdict: needs changes.** No single finding is a "must-fix-before-any-use" blocker — the tool
is already catching real defects in production use — but two of the majors below are
demonstrated, not speculative (both reproduced by direct execution), and one of them is exactly
the shape this team's own testing standard warns against (`guard-testing-techniques.md`'s
"docstring states more than the body" pattern): the module docstring spends ~15 lines justifying
the whitespace-normalization mechanism as load-bearing, and the selftest carries zero coverage
that would catch its removal.

## Findings

### Major — duplicate-content ("overlap") detection misses non-adjacent nested overlaps

`check_partition` (check_content_loss.py:319-324) sorts located spans by start offset and then
compares only *adjacent* pairs (`spans[i-1]` vs `spans[i]`) for overlap. This misses a genuine
overlap between two spans that aren't adjacent in sort order once a third span is interposed
between them. Reproduced directly:

```
spans (after sort): wide=(0,60)  narrow1=(10,30)  narrow2=(40,60)
```
`narrow2` is fully nested inside `wide` (a genuine duplicate — `wide`'s claim re-contains
`narrow2`'s entire content) but the adjacent-pairs loop only checks `(wide,narrow1)` and
`(narrow1,narrow2)`; the real `wide`↔`narrow2` overlap is never added to `report.overlaps`. See
Appendix A for the full reproduction script and output.

**Important nuance, also verified:** this never produced a false "clean" verdict in any
construction I tried — the wide span, by definition covering the gap between the two others,
always overlaps *some* neighbor in sort order, so `report.clean` still ends up `False`. The
practical harm is under-diagnosis, not masking to green: a human fixing the *reported* overlap
(say, deleting `narrow1`) could walk away believing the section is now clean while `narrow2` is
still a live, unreported duplicate of `wide`.

**Suggested fix:** replace the adjacent-only comparison with a proper interval-overlap check —
for each span (sorted by start), compare against every later span until one starts at or after
the current span's end (this is still cheap: an early break once `spans[j][0] >= spans[i][1]`,
and per-heading claim counts are small). Add a coverage-probe case with 3+ claims arranged so a
wide span nests two non-adjacent narrower ones, asserting the reported overlap set names *all*
pairs, not just that `report.overlaps` is non-empty (the current mutation test's assertion style —
"expected a DUPLICATE overlap, got none" — would pass even with this bug present).

### Major — the tool's own signature mechanism (whitespace normalization) has zero regression coverage

`build_normalized`/`locate_claim` exist specifically because a real migrated file
(`coordination-techniques.md`) showed a claim re-wrapped at a different column than the source's
own hard-wrap — the docstring (check_content_loss.py:207-236) spends real space justifying this.
None of the five selftest checks exercise it: every baseline/mutated claim text is sliced
byte-exact out of the fixture body with no re-wrap, so a naive `body.find(text)` would locate them
identically. Verified by monkeypatching `build_normalized` to a no-op passthrough and re-running
`check_content_loss_selftest.main()`: all 5 checks still pass, exit 0 (Appendix B). A regression
that silently reverted the normalization feature — the exact defect shape it exists to fix — would
ship with a fully green selftest.

**Suggested fix:** add a positive coverage-probe case — a claim text built from a source bullet
but re-wrapped at a different column (same words, different newline placement) — asserting it
still locates cleanly; and, to keep the match genuinely exact rather than accidentally
permissive, a companion case where the re-wrapped claim also drops one word, asserting that
*still* reports NOT_FOUND/gap. This is the "axis the artifact varies on" (line-wrap placement,
present vs. absent) rather than one more fixed shape.

### Major — the CLI/manifest layer (both schema forks, `--live`/`--dump`, `missing_document`) is entirely untested

`check_content_loss_selftest.py` calls `check_partition`/`render_report` directly and never
touches `main()`, `load_manifest`, `claims_from_manifest_file_entry`,
`claims_from_manifest_flat_key`, `combined_body`, `make_dump_fetcher`, `make_live_fetcher`, or the
`missing_document` outcome status (confirmed by grep — zero hits for any of these names in the
selftest file). This is the code implementing the module docstring's entire "MANIFEST SCHEMA — a
real fork" section, which the docstring itself frames as a first-class design decision (~25 lines
of justification), yet it has no automated check at all. I exercised it by hand — both the
`--documents`/`--dump` whole-file path and the primary `--manifest`/`--dump` per-heading path,
including a deliberately-unresolvable `documentId` — and all worked correctly (Appendix C), so
this is a coverage gap rather than a live bug, but it's the same shape as the two majors above: a
documented, justified code path with nothing to catch its regression.

**Suggested fix:** a coverage probe over the CLI's own axes — manifest shape (primary per-heading
vs. flat-key vs. `--documents`), fetch source (`--dump` is testable without FalkorDB; `--live`
can at least be smoke-tested for the "binary missing" error path), and per-claim outcome status
(`found`/`not_found`/`missing_document`) — using small synthetic manifest/dump JSON files under
`fixtures/`, run through `check_content_loss.main()` with `argv` supplied directly (no subprocess
needed). At minimum, add `missing_document` to the existing `check_partition`-level tests, since
that requires no CLI plumbing at all.

### Minor — an empty/whitespace-only claim `text` reports `found` at a zero-length span, not a distinct defect

`locate_claim` normalizes `text.strip()`; if that's empty, `norm_body.find("")` returns `0`
unconditionally, producing a `(0, 0)` span reported as `status="found"`. Reproduced: a
completely empty claim against a real body still reports `found (0, 0)`, with the actual content
correctly showing up as a gap for unrelated reasons (Appendix D) — so this doesn't cause a false
"clean" verdict in the cases I could construct, but a document that got ingested with genuinely
empty/whitespace text (a plausible ingestion-bug shape, not just this checker's edge case) is
silently folded into the "found, no worries" bucket rather than being called out as its own
defect, which matters once a report has many claims and a human is scanning for the flagged ones.
**Suggested fix:** in `check_partition`, treat a claim whose stripped `text` is empty as its own
outcome status (e.g. `"empty_text"`) before calling `locate_claim`, rather than letting it fall
through to the general path.

### Minor — whole-file/flat-key mode without `--headings` silently treats every unmigrated heading as 100%-unaccounted noise

`combined_body(sections, headings=None)` concatenates *every* `## ` section's body when
`--headings` is omitted. For a manifest flat-key entry whose claim list covers only some headings
of a multi-heading file (the exact "partial migration" scenario the module's own USAGE section
names and provides a `--headings`-restricted example for), forgetting `--headings` doesn't error —
it produces a report where every unmigrated heading's entire body shows up as one or more
UNACCOUNTED gaps. This is a false positive, not a masked false negative (it can't hide a real
defect — if anything it's over-cautious), so it's the noisier failure mode, but it can still bury
a small genuine gap in the middle of an expected, large, "not migrated yet" one. **Suggested
improvement:** when `--headings` is omitted in flat-key/`--documents` mode, print a one-line
warning naming the source file's total heading count so a partial-migration run isn't silently
treated as whole-file scope; this is cheap (the section count is already computed) and doesn't
require the tool to guess which headings the id list actually covers.

### Nit — `--documents` silently wins over `--manifest-flat-key` if both are passed

`main()`'s dispatch (check_content_loss.py:557-566) checks `args.documents is not None` before
`args.manifest_flat_key is not None`, so passing both silently ignores the flat-key argument with
no warning. Cheap fix: put the two in an `argparse` mutually-exclusive group, or print a one-line
note when both are set.

### Nit — the module docstring's "MANIFEST SCHEMA" section is already stale against the live manifest

The docstring (check_content_loss.py:59-63) names `_graph_dba_falkordb_quirks_IN_PROGRESS` with
"two separately-named partial lists." The current `kb-claim-manifest.json` (untracked — this file
isn't part of the reviewed commit, confirmed via `git log` on it returning nothing) has since
renamed that key to `_graph_dba_falkordb_quirks_DONE` with **five** heading-specific sub-keys, not
two, and still no flat `documentIds` — so `--manifest-flat-key` on it still correctly fails loud
(verified) rather than silently degrading. Not a code defect — the docstring is honestly labeled
"as of this writing (2026-09-18)" — but worth a one-line callout since a reader could otherwise
assume the docstring describes the manifest's current shape.

## What's solid

- **The core coverage/gap algorithm is sound as a global accounting argument**, not just on the
  four scripted mutations. I specifically probed the brief's named worry — two adjacent bullets
  both dropped, absorbed into one larger gap — and confirmed it's real content correctly reported
  (Appendix E), not masked; I also probed whether whole-file/flat-key mode's larger search space
  could let the named "first occurrence" limitation actually hide a drop when two headings share
  identical text, and it didn't (Appendix F) — the position-covering design means a genuinely
  missing occurrence always leaves *some* span uncovered, even when `str.find`'s first-match
  behavior misattributes which literal occurrence a given claim gets credited for.
- **The four scripted mutation cases are each proven load-bearing** in the way the selftest's own
  docstring claims — the specific assertions (gap location matches the dropped/truncated content,
  `not_found` specifically rather than any-non-clean, the two duplicate document IDs both named)
  go well past "not clean," which is exactly the team's stated mutation-testing bar.
- **The manifest-schema fork is handled the right way for a tool whose whole job is refusing to
  guess** — both fallbacks (`--manifest-flat-key`, `--documents`) fail loud with an actionable
  message rather than attempting to parse the irregular keys' free-text `_note` fields.
- **No security/perf concerns.** The live-fetch subprocess snippet takes no interpolated
  input (host/port read from env inside the subprocess, ids passed via JSON over stdin) — verified
  by reading the snippet, there's nothing to inject. Claim counts per heading/file are small
  (tens), so the `O(n)` coverage bytearray and even an `O(n²)` all-pairs overlap fix (suggested
  above) are non-issues.
- **Already caught a real defect in production use**, per the coordination ledger (U3e: "2 real
  fidelity findings on already-migrated claims relayed to cobb's U3f") — external validation this
  isn't just theoretically sound.

## Open questions

- None that block a fix-and-re-review cycle — the three majors are all concrete, independently
  actionable, and don't depend on a design decision only the stakeholder could make.

## Appendix

**A — non-adjacent overlap miss:**
```python
import check_content_loss as ccl
Claim = ccl.Claim
body = 'A'*10 + 'B'*20 + 'C'*10 + 'D'*20 + 'E'*40  # len 100
store = {
    'wide':    body[0:60],   # (0,60)
    'narrow1': body[10:30],  # (10,30), nested in wide
    'narrow2': body[40:60],  # (40,60), nested in wide, non-adjacent to narrow1 in sort order
}
claims = [Claim(document_id=k) for k in store]
report = ccl.check_partition(body, claims, lambda k: {'title': k, 'text': store[k]})
# report.clean == False; report.overlaps == [(wide, narrow1)] only -- wide<->narrow2 never reported
```

**B — normalization has zero coverage:**
```python
import check_content_loss as ccl
def naive_normalized(body):
    return body, list(range(len(body) + 1))
ccl.build_normalized = naive_normalized      # simulate reverting the normalization feature
import check_content_loss_selftest as t
t.main()   # -> ALL PASS (5/5), exit 0, with normalization fully disabled
```

**C — manual CLI smoke tests (both schema paths + missing_document), both exit as expected:**
`--documents ... --headings ... --dump <fixture-slice.json>` → clean, exit 0.
`--manifest <primary-shape.json> --dump <empty.json>` → `[MISSING_DOCUMENT]` + full-body gap,
exit 1. (Full commands run during this review, not reproduced verbatim here for length.)

**D — empty claim text:**
```python
report = ccl.check_partition(
    'Some real content here that should be fully claimed by one document.',
    [ccl.Claim(document_id='empty-claim')],
    lambda k: {'title': 'oops', 'text': ''},
)
# outcome status == 'found', span == (0, 0); real content still shows as an UNACCOUNTED gap
```

**E — two adjacent dropped bullets, real fixture:** dropping bullets 1 and 2 of the fixture's
"Three independent facts about connection pooling" heading (keeping only bullet 3) produces
exactly one gap, 509 characters, containing both dropped bullets' full text verbatim — not
absorbed into silence.

**F — whole-file mode with a duplicated common sentence across two headings, one heading's copy
dropped:** both whole-file (flat-key-shaped) and per-heading checks correctly report the missing
occurrence as a gap; `str.find`'s first-occurrence behavior changes which claim gets "credit" for
which literal occurrence, but never causes the drop to go unreported.

## Pass 2 — 2026-09-19

**Verdict: approve.**

Read the diff directly (not just the coordinator's summary) and reran
`check_content_loss_selftest.py` myself — 16/16 pass, matching the coordinator's own independent
rerun — plus my own targeted rechecks below, reusing the exact constructions from Pass 1's
Appendix. All six Pass-1 findings are fixed and independently reverified by execution.

- **Major 1 (non-adjacent nested overlap):** fixed. `check_partition`'s overlap loop
  (check_content_loss.py:345-352) now scans each span against every later span until one starts
  at/after the current span's end. Reran Pass 1's own Appendix A construction against the current
  code: both `wide↔narrow1` and `wide↔narrow2` are now reported (previously only the first). The
  new `check_nonadjacent_nested_overlap` test asserts the full overlap-pair *set*, closing the
  exact "would pass even with the bug present" gap called out in Pass 1.
- **Major 2 (whitespace-normalization had zero coverage):** fixed. Reran Pass 1's regression
  probe (monkeypatch `build_normalized` to a no-op, rerun the selftest): `check_rewrapped_claim_
  still_locates` now fails as expected (1/16 FAILED, with a legible word-level diff) — the exact
  signal that was silent in Pass 1. The negative companion
  (`check_rewrapped_and_word_dropped_still_not_found`) confirms the tolerance stays exact rather
  than becoming permissive.
- **Major 3 (CLI/manifest layer untested):** fixed. 8 new tests drive `main(argv=...)` end-to-end
  (both manifest-schema forks, `--documents`, `missing_document` via `--dump`, and the `--live`
  missing-venv error path) plus a `check_partition`-level `missing_document` test. All pass on my
  own rerun.
- **Minor (empty claim text):** fixed. `empty_text` is now its own `ClaimOutcome.status`, set
  before `locate_claim` is ever called (check_content_loss.py:314-326) — this also removes a
  latent side effect not called out in Pass 1: an empty-text claim no longer contributes a bogus
  `(0,0)` span to the overlap-candidate list at all (it `continue`s before `spans.append`), so it
  can no longer spuriously "overlap" whatever claim starts at position 0.
- **Minor (silent full-file scope):** fixed. Confirmed the stderr warning fires exactly when
  `--headings` is omitted in `--documents`/`--manifest-flat-key` mode (visible in the selftest's
  own captured stderr during my rerun).
- **Nit (mutual exclusion):** fixed. `--manifest-flat-key`/`--documents` are now an argparse
  mutually-exclusive group; confirmed directly (`argument --manifest-flat-key: not allowed with
  argument --documents`, exit code 2) via both the new test and my own ad hoc rerun.
- **Nit (stale docstring):** fixed. The MANIFEST SCHEMA section now names itself a snapshot and
  states neither fallback depends on the specifics staying accurate (check_content_loss.py:64-69)
  — accurate as written.

**Open judgment call, settled:** whether generating the Major-3 CLI fixtures via
`tempfile.TemporaryDirectory()` per test, rather than persisting them under `fixtures/`, is an
acceptable substitution. **It is — no change requested.** The one existing persisted fixture
(`content_loss_test_kb.md`) is a different kind of artifact: real KB prose demonstrating the
checker's three substantive split shapes, worth hand-inspection and reuse as the matching logic
evolves. The new CLI tests exercise argparse/manifest-parsing *plumbing* with trivial,
fully-deterministic JSON literals written inline in each test function — reading the test source
already is reading the fixture; nothing is generated or hidden, and a failing assertion's message
already names the synthetic ids/paths involved. Persisting them under `fixtures/` would add files
with no independent inspection value over the test code itself.

No new findings from this pass. All six Pass-1 findings are closed; nothing outstanding.
