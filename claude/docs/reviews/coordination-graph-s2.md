# coordination-graph S2 gate (KB prerequisite, narrowed K-058)

> **Status:** active · **Owner:** `analyst` · **Tracks:** K-058 (coordination-graph S2)

## Scope & verdict

**Verdict: approve with suggestions** (0 blockers, 1 major, 3 minor, 1 nit).

Reviewed the uncommitted S2 diff by `cobb` against `claude/docs/plans/coordination-graph.md` §4 S2 row, §2.1, §3.5 and
`claude/docs/plans/coordination-graph-graph.md` §6.4, plus the parser `coordination-graph/coordgraph/parse/kb.py`.
Files: `claude/cobb/scripts/kb-claim-manifest.json`, `claude/cobb/scripts/check_content_loss.py`,
`coordination-graph/kb-files.txt`, the 6 `<!-- kb-split: bullets -->` markers
(`claude/graph-dba/falkordb-reference.md`, `claude/frontend-engineer/frontend-quirks.md`),
`claude/cobb/kaizen/{plan,history}.md`. Out of scope per brief: `claude/graph-dba/falkordb-quirks.md`
(2 uncommitted insertions from another session, 0 markers) and `claude/docs/plans/coordination-graph-graph.md`.
No writes to `ws:agent-team` or any graph; all graph access was `RO_QUERY`/`get_document`.

CPG: considered, not relevant — docs, a JSON manifest and a stdlib script; no loaded graph covers `claude/`.

Root `REVIEW.md` read and applied (evidence bar; "no X remains" sweeps; check-can-fail rule).

## Findings

### M1 (major) — K-058's "open" list is not exhaustive; 4 more claim gaps in the *keyed* files, and 1 miscount

`kb-claim-manifest.json` is the evidence S2 wrote, and K-058 / `history.md` present the open list as complete
("Open, for the plan's first `kb-sync`/`kb-reconcile`"). It is not. I hashed every parsed section of all 13 files
(`parse_kb` claim hash) against every `currentVersion` document in `ws:agent-team` and probed 3 text windows per
unmatched section (Appendix A):

- **Never synced, no claim anywhere in the graph** (0 title hits, 0 hash hits, 0/3 text-window hits, no manifest mention):
  `ops-quirks.md` "falkor-chat's `ingest_document` MCP tool auto-supersedes…" (l.193); `qa-testing-techniques.md`
  "Parse-checking Mermaid blocks from WSL…" (l.285; only the raw capture `d88545b5` exists);
  `test-design-techniques.md` "A hand-built "rejected alternative" mutant…" (l.77). All three are in the 8 `files`-keyed files
  that S2 did not audit for drift. This is exactly K-058's failure mode (b), still live.
- **Drifted claim** in a keyed file: `plan-authoring-techniques.md` "Reconciling a diverging delegated note…" — stored
  `c0bf904a` (1003 chars) differs from the section (1022 chars): different words, not whitespace.
- **Miscount in the delegate's own open list:** K-058 item (2) says "4 sections with no claim"; the S2 mapping has
  **5** empty headings — it omits `lm-studio-model-notes.md` "A live-run report's provenance…" (`documentIds: []`).

Why it matters: S6's first `kb-sync` will pick these up (added/changed sections), so nothing is lost — but the record
that says what S6 inherits is wrong, and AC-8's "every KB section mirrored" baseline cannot be read off K-058.
**Fix (owner `cobb`):** rewrite K-058 item (2)/(3) from the table in Appendix A (9 claim-less sections: teco 1,
review-techniques 3, lm-studio 1, ops-quirks 1, qa 1, test-design 1; drifted: review-techniques 11 docs, lm-studio,
plan-authoring 1). Better, since S2's acceptance is "stale claims re-synced", re-ingest those 3 missing claims by the
§5 procedure now (one-at-a-time, embedder resident) rather than leave them to S6 — the narrowing's point is that S6
starts from a mirror without known gaps.

### m1 (minor) — S2's row is only partly met: `falkordb-quirks.md` is neither re-synced nor marked, and the K-058 text does not gate S6 on it

Plan §4 S2 requires `falkordb-quirks.md` re-synced "once its pending edit is committed" and "`<!-- kb-split: bullets -->`
on every heading already split bullet-per-claim". The deferral is legitimate (the file has uncommitted edits from another
session; the brief excluded it) and K-058 names it. But the consequence is not stated: without the 4 markers, S6's first
sync parses its 4 big headings as `whole` and replaces 84 hand-split bullet claims with 5 multi-chunk documents
(`needs-resplit`, plan §3.5) — the loss of retrieval granularity the markers exist to prevent. Evidence: my forced-bullets
comparison shows those 4 headings now hold 22/40/15/34 bullets against 17/39/11/16 stored claims (Appendix B), so they are
bullet-split with drift, i.e. markers belong there. **Fix:** K-058 states "S6's first run on `falkordb-quirks.md` waits for
its markers (cobb) once the file is committed"; the S6 row's ledger entry carries it.

### m2 (minor) — S6 flag: confirmed, but half-stated; neither side is specified, recommend keeping the parser

Executed check (Appendix C): for the 17 claims in `falkordb-reference.md`'s three marked headings, the stored
`textNormalizedHash` equals `claim_hash` of the **marker-stripped** bullet (17/17) and never of `parse_kb`'s marker-bearing
text (0/17). The delegate's flag is right for that file. What it omits: the **6 claims of `frontend-quirks.md`'s marked
headings were stored *with* the `- ` marker and match `parse_kb` today (6/6, 0 re-ingests)**, and so do the 2 whole-mode
`falkordb-reference.md` sections. The store is internally inconsistent; "marker-stripped" (`check_content_loss.py` docstring,
lines ~25–30) is a half-truth that generalised from one file.

Which side is right per spec: **none is written down.** Plan §3.5 says only "one document per top-level bullet, keyed by
bullet hash"; note §6.4 / §2.4 define `claim_hash` over the claim text without saying whether it carries the marker. The
parser's behaviour is pinned by S1a tests (`coordination-graph/tests/test_kb.py:80,92,97`) and is consistent with `whole`
mode (claim = verbatim slice of the file). Recommendation: **keep the parser; accept the 17-document re-ingest as ordinary
first-sync work** (append-before-delete, so no claim is ever absent; the marker-bearing and marker-less texts hash
differently, so no auto-supersede collision), and record the rule once. Reversing it would instead re-ingest 6 and force a
test and spec change for a cost that is not smaller in kind. Owners: **`architect`** states the rule in plan §3.5 (one
sentence: "a bullet claim's text includes its list marker; `claim_hash` is over that text"); **`coder` (S6)** pins it in
the sync test and expects the 17 re-ingests; **`cobb`** corrects the checker docstring and K-058 item (4) (6 stored with the
marker, 17 without). The checker is unaffected either way: `locate_claim` tries bare and marker-prefixed text.

### m3 (minor) — `load_sections` directive-drop is correct for today's files but is untested and broader than the parser

Judged as asked. It is needed: running the pre-change checker (HEAD copy, run from the scratchpad with `REPO_ROOT`
patched) against both marked files exits 1; the changed checker exits 0 with 0 findings on both (19 and 9 claims). Selftest:
`check_content_loss_selftest.py` 25/25 PASS, **but the selftest file is unchanged** (`git diff` empty) — nothing pins the
new behaviour, so the regex could be deleted and the suite stays green (only a live run on the two real files would
notice). Divergences from the parser: `_SPLIT_MARKER_LINE` drops a directive line **anywhere** in a section, including
inside a fenced code block or mid-body, whereas `parse_kb` treats only the heading-line marker or the first non-blank line
as a directive. A KB entry that documents the marker syntax in a code block would be silently altered in the checker's body
and its claim would then fail to locate. No such section exists today (grep over the 13 files). **Fix (owner `cobb`):**
add a selftest fixture pair (bullets section with marker-alone line: clean; same without the drop = gap) and restrict the
regex to the first non-blank line of a section, as the parser does.

### m4 (minor) — `python-web-quirks.md` exclusion is right, but unrecorded

`claude/knowledge/python-web-quirks.md` does carry claims (36 documents in `ws:agent-team` carry the `<!-- body -->`
header; `claude/cobb/scripts/pwq-kb-manifest.json`, 36 units). Excluding it from `kb-files.txt` is correct: its stored text is a
unit header plus body, mirrored per unit by `pwq_kb.py`, whereas `parse_kb` would treat its `## ` lines as sections whose
bodies absorb the next unit's `<!-- unit -->` header, and no hash could ever match. Nothing records this, so a future cobb
who "adds every KB file" would create 36 permanent mismatches. **Fix:** one comment line in `kb-files.txt` ("excluded:
`claude/knowledge/python-web-quirks.md` — unit-format copy, mirrored by `pwq_kb.py`") — `discover.kb_files` skips `#` lines
(`coordination-graph/coordgraph/discover.py:81-90`, verified by running it: 13 paths).

### n1 (nit) — K-058 row restates delivery history

`kaizen/plan.md` K-058 (~1.5k chars) opens with a "Done:" narrative that `history.md` already holds; root `AGENTS.md`
says a backlog item is forward-looking and rewritten as one present-tense statement. Trim "Done: …" to a clause citing the
`history.md` entry once M1/m1/m2 are folded in.

## What's solid

- **Mapping recovery agrees with an independent method.** The delegate matched by hash/title; I matched by normalised
  *substring containment* of each current document's text in each section body (HEAD text), over the 5 files
  (Appendix A): teco 41/41 ids, guard-testing 19/19, falkordb-quirks 84/84 placed identically; lm-studio and
  review-techniques differ only where expected (review-techniques: 71 of 82 ids placed identically, the other 11 are the
  6 title-matched "drifted" headings whose text no longer appears in the file, which a containment test cannot place by
  construction; lm-studio: 1 stale pre-merge document `0651fa6f` that the delegate left *unplaced* and containment places in
  the measurement-surface heading — same conclusion, both stale). Zero contradictions; 0 ambiguous placements.
- **The 5 permanently-failed ids are correctly reconciled.** Old ids `d442fa32`, `cc22190d`, `44758856`, `c9383a8b`,
  `23527571` are absent; replacements `336901fb`, `59650809`, `e3c6b8fc`, `2dd8c101` (ops-quirks) and `abb4a255`
  (guard-testing) are `ready`, `currentVersion=true`, and their text is an exact substring of the file section
  (compared by my own script, not the delegate's). All 90 manifest `files` ids and all `_s2` mapping ids exist,
  current and ready; no two current documents share a `textNormalizedHash`.
- **Stale teco / missing devops claims are fixed:** no current document carries "Environment readiness: bring FalkorDB up
  yourself" outside the K-058 evidence doc `92e938a3`; the ~13.1 s spawn-stall claim is `50dc6b7e`.
- **Markers are only on truly 1:1 headings.** `parse_kb` on both files: marked headings give 5/4/8 and 2/2/2 claims
  against 5/4/8 and 2/2/2 top-level bullets, 0 warnings (no stray non-bullet text). Unmarked `falkordb-reference.md`
  "Graph data modeling" (4 bullets → 1 claim) and "GraphRAG" (3 → 1), and `frontend-quirks.md`'s three 1-bullet headings, are
  correctly `whole`. I also forced bullets on every multi-claim section of all 13 files: the only other 1:1 candidates are the
  `falkordb-quirks.md` ones (m1). `lm-studio-model-notes.md` "measurement surface" (8 stored vs 9 bullets, plus intro and
  trailing prose that bullets mode would drop with a warning) and `review-techniques.md` "grep-pinned edit table" (mixed
  prose/bullets) are rightly left unmarked.
- **`kb-files.txt` is exactly the KB set:** 13 paths, all exist, all carry claims, equal to the manifest's 8 `files` keys + the
  5 unkeyed files; `discover.kb_files` returns them; `parse_kb` yields 0 warnings on all 13.
- **Executed re-runs:** `check_content_loss.py <file> --live` on both marked files: exit 0, 19/19 and 9/9 claims clean;
  selftest 25/25.
- `history.md` accurately states what was and was not done (no `ws:agent-team` write/delete; deferrals named).

## Open questions

None blocking. If the stakeholder prefers the cheaper reading of m2 (strip the marker in `parse_kb`), that is a spec decision
for `architect` and costs 6 re-ingests plus changing 3 pinned S1a assertions; I recommend against it.

## Appendix

### A. Independent mapping method and results

`RO_QUERY` on `ws:agent-team` for all `currentVersion` documents (528); section split on `## ` outside fences, text from
`git show HEAD:<file>`; document→section by normalised (whitespace-collapse, casefold, optionally list-marker-stripped)
substring containment, length > 40, unique candidate required (0 ambiguous). Compared with the manifest's
`_s2_heading_mapping_2026_11_20`:

| file | sections | manifest ids | independently placed | differences |
|---|---|---|---|---|
| teco coordination-techniques | 40 | 41 | 41 | none |
| tdd guard-testing-techniques | 8 | 19 | 19 | none |
| falkordb-quirks | 5 | 84 (+4 unplaced) | 84 | none |
| lm-studio-model-notes | 10 | 16 (+3 unplaced) | 17 | `0651fa6f` (stale pre-merge doc) placed by me, unplaced by delegate |
| analyst review-techniques | 58 | 82 | 71 | 11 ids under 6 headings, all title-matched "drifted" (text changed since ingest) — not placeable by containment |

Claim-less sections by hash over all 13 files (parse_kb hash vs current docs, then 3-window text probe): teco 1, review-techniques 3,
lm-studio 1, ops-quirks 1, qa 1, test-design 1, falkordb-quirks 4 (the deferred file); drifted single-claim section:
plan-authoring 1.

### B. Forced-bullets comparison (multi-claim sections currently `whole`)

`falkordb-quirks.md`: Indexing 17 stored / 22 bullets; Cypher dialect 39 / 40; Query tuning 11 / 15; Ops 16 / 34 (0 stray lines in
each). Other multi-claim `whole` sections (review-techniques, guard-testing, teco "Mutation-test the green-on-arrival tests") are
paragraph-split: bullets 0 or ≠ stored, correctly unmarked.

### C. Marker hashing check

For every claim `parse_kb` produces under a `bullets` heading, tested `claim_hash(text)` and `claim_hash(text with leading list
marker removed)` against current `textNormalizedHash`: falkordb-reference 17 claims → marker-bearing 0 / stripped 17;
frontend-quirks 6 claims → marker-bearing 6 / stripped 0. Stored text starts `- **` for all 9 frontend documents and for the 2 whole-mode
falkordb-reference sections `- Model…`, `- Combine…`; `**…`/bare text for the 17.
