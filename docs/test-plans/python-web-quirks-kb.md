# python-web-quirks on the knowledge base — Test Plan

> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** — (python-web-quirks-kb, S8/S11/S14/S15c)

Contract for the QA runs of `docs/plans/python-web-quirks-kb.md`: S11 (live checks in `ws:pwq-test`, report `docs/test-reports/python-web-quirks-kb-report.md`), S14 (the one sealed scoring, same report family) and S15c (AC-6, AC-10, AC-13 black-box, no workspace written). The plan's T-table (§5, T1-T39) is the check list; this document does not restate it. It says what each QA run does with those rows, in which order, with which evidence. No Set B query text appears here, by design (D11): only B's sha256 and aggregate figures.

## 1. Scope and objective

Verify, from the outside, that the migrated knowledge base behaves as AC-2, AC-3, AC-5, AC-7, AC-8, AC-11 and AC-12 of `docs/requirements/python-web-quirks-kb.md` require. In scope: T31-T38 (S11, `ws:pwq-test` on `:8210`), the sealed scoring of AC-3 (S14, production, read-only), Set B integrity, and the AC-11 ledger evidence, and S15c: AC-6 (T39), AC-10, AC-13 (TP-013 to TP-015, run after S15b1/S15b2). Not in scope: T1-T30 (unit and contract layer, `tdd-engineer`/`analyst` G1), AC-1/AC-9/AC-14 (S13, G3).

## 2. References

- Plan: `docs/plans/python-web-quirks-kb.md` — D2 (units), D3 (document format), D6 (listing/regex), D8 (target guard, add/edit/retire), D9 (script modes), D11 (evaluation design, seal), §5 T-table.
- Method note: `docs/plans/python-web-quirks-kb-ml.md` — §4 (gates, negatives scoring), §5 (authoring, hold-out, probe ruling), §7 (tail/head risk).
- Test workspace record: `docs/plans/python-web-quirks-kb-graph.md`.
- Requirements: `docs/requirements/python-web-quirks-kb.md` (ACs above).
- Tool: `claude/cobb/scripts/pwq_kb.py` (`eval`, `gate`, `diverge`, `replace`, `retire`, `ingest`, `verify-copy`), `claude/cobb/scripts/pwq_leakcheck.py`, the copy of record `claude/knowledge/python-web-quirks.md`.

## 3. Risk assessment

| Risk | Likelihood x impact | Where it is tested |
|---|---|---|
| A unit's tail chunk is not retrievable (no title in a body fragment; 12 units have at least 4 chunks) | medium x high | TP-001 (last-paragraph query), TP-010 (tail vs head) |
| An unembedded chunk reads as a retrieval miss (K-102) | medium x high | TP-001 gate, TP-010 pre-check |
| Set B contaminated (read, edited, scored twice, leaked from a restating capture) | medium x high (the seal is procedural) | TP-011, TP-010 pre-checks |
| Raw captures displace unit chunks from the five slots, or inflate recall | high x medium | TP-010 exclusion run beside the unfiltered run |
| Header facts wrong (`Verified`/`unverified`, versions) so a hit misleads | low x high | TP-002 |
| Listing misses a unit or exceeds the bound; a 43rd title goes unnoticed | medium x medium | TP-003, TP-004 |
| Rebuild from the copy diverges from the pin; add/edit/retire leaves copy and KB inconsistent | medium x high | TP-005, TP-006, TP-007 |
| Write guard allows the retired skill path or blocks the copy | low x medium | TP-008 |
| Production touched by a test run | low x very high | environment rules (§5) |
| A consuming agent cannot call the KB tools, or the discovery pointer is missing from one of the eight | medium x high | TP-014, TP-015 |
| Degraded mode reads the whole copy or does not announce the outage | medium x medium | TP-013 |

Prioritisation: seal, tail probe and gate (TP-010, TP-001) first, then the lifecycle checks (TP-005 to TP-007), then the read-only header/listing checks; S15c (TP-013 to TP-015) runs after the agent edits land.

## 4. Test items

Each item cites the plan's T-row for steps and mutant; "Evidence" is what the report must quote. Priority P1 = blocks the verdict, P2 = must run, P3 = informational.

| ID | Title | AC / T | Type | Pri | Preconditions | Steps and expected result | Evidence |
|---|---|---|---|---|---|---|---|
| TP-001 | Per-unit retrieval: title, situation, last paragraph | AC-2, T31 | functional, e2e | P1 | S10 `ingest` + `gate` clean in `ws:pwq-test` | For each of the 36 units, three queries through the retrieval prefix (title from the listing; the unit's Set A situation queries, which are dev data; a phrase from the unit's last paragraph taken from the copy). Each must return the unit's document among the five; the hit's `get_document` title begins `Gotcha:`; the merged units (u02, u21, u24, u29, u30) are returned as one document for each member section. `gate` reports 0 unembedded chunks for the 129 chunks. Mutant: delete the last chunk of one unit in the test graph, the last-paragraph query for it must turn red, then restore. | per-unit table (unit, 3 hit ranks, title string), `gate` table |
| TP-002 | Header facts: dates, `unverified`, versions | AC-8, T32 | functional | P2 | TP-001 | Count units whose body carries a phrase-qualified date and compare with the `Verified:` lines in the stored documents; the rest read `unverified, migrated <date>`; a unit with library versions shows them. Counts are derived from the pin, never from the tool's own output. Mutant: swap one `Verified` for `unverified` in a scratch copy, count mismatch. | derived counts vs observed counts |
| TP-003 | Title listing: all titles, bounded, normal and degraded | AC-8, T33 | contract | P2 | copy of record present | Apply the D6 regex to the copy: one title per pin heading, one fence per unit, both preamble fences, at most 10,000 bytes (expected values: the pin's, not the plan's). In normal mode the same titles are reachable from the KB. Mutant: regex without `preamble`, two fences missing. | `grep -c`, byte count |
| TP-004 | 43rd title | AC-8, T34 | functional | P2 | scratch copy (never the committed one) | Add a 43rd gotcha heading; it must appear in the degraded listing, and `diverge --ws pwq-test` must exit 1 until it is ingested. | listing line, `diverge` exit code and stdout |
| TP-005 | Rebuild from the copy | AC-5, T35 | e2e | P1 | S10 state | Remove the units from `ws:pwq-test` (REST `DELETE` through the tool's test-only path), rebuild with `ingest` from the copy, regenerate the manifest, run the content-loss checker with `--source-format copy --ws pwq-test` (clean), then repeat TP-001. Mutant: one corrupted body character in a scratch copy turns the checker red. | checker output, manifest diff, TP-001 rerun |
| TP-006 | Add, edit, retire | AC-7, T36 | e2e | P1 | TP-005 green | In `ws:pwq-test` only: add one gotcha, edit one, retire one (`retire` also removes it from the scratch copy). Added and edited: TP-001 passes for them and the stored text equals the copy; retired: not returned, `diverge` clean. Mutant: skip the ingest of the edit, `diverge` reports it stale. | per-step command, `diverge` output |
| TP-007 | Planted divergence | AC-7, T37 | functional | P2 | TP-006 | Plant one divergence (edit a stored document directly in `ws:pwq-test`, or the scratch copy): `diverge` exits 1 and names the unit; with nothing planted it exits 0 (control, run first). | both exit codes, named unit |
| TP-008 | Write guard | AC-12, T38 | contract | P2 | S4 done | Feed the guard the paths `claude/knowledge/python-web-quirks.md` (allow), `skills/python-web-quirks/SKILL.md` and `docs/BACKLOG.md` (ask), reading the guard's real invocation shape from its selftest; confirm the guard file no longer lists `skills/python-web-quirks/*`. Mutant: re-add the entry in a scratch copy of the guard, the old path is allowed. | three decisions, `grep -c` of the old entry |
| TP-009 | Capture coverage, ruling, deletion | AC-11 | acceptance | P2 | S7, S13b, H1 recorded | Read the ledger: the derived capture list exists (ids, titles, coverage by unit), the ruling and promotion text for `38cd49cd…` is recorded in `claude/cobb/kaizen/plan.md`, and the stakeholder's deletion ids are in the ledger; re-derive the list at S14's start and compare. Read-only. | list, ruling pointer, deletion ids, any new capture since S13b |
| TP-010 | The sealed scoring | AC-3 | acceptance | P1 | TP-011 green, production gate clean, floor decided in S12/S15a | Once, in production, read-only: `eval --ws agent-team --base-url http://localhost:8200 --set B --sealed-once --expect-sha256 <§5 value> --floor <explicit> --exclude <S7 capture ids>` with the pre-checks of §5. Report the headline (excluded run) and the unfiltered run side by side. Gate figures as approved (ML §4, not restated here); additionally report BM25 on B, per-section coverage, mean distinct documents per query, and tail-flagged recall against head recall (a gap over 0.15 is a design decision for `architect`, ML §7). A failed B is final: B2 by a different author, never a re-run. | full tool output, sentinel file, hash check line |
| TP-011 | Set B integrity | AC-3, D11 | functional | P1 | S8 done | Before S14: file is outside the repo, mode `-r--r--r--`, `sha256sum` equals the recorded value, `pwq_leakcheck.py` reports 0 shared 4-grams, `validate_set_b` with the pin's tail units passes, `git ls-files` and `git grep` find no B file in the repo. | the five command outputs |
| TP-013 | Degraded mode: server unreachable and retrieval error | AC-6, T39 | e2e | P2 | S15b1 landed; copy committed | No KB workspace is written; `:8200` is never stopped. (a) A tool-denied session of one consuming agent, and (b) a session using a throwaway MCP config whose falkor-chat URL points at a dead port. Ask each a gotcha question from the pin. Expected: the reply starts "knowledge base unreachable", the agent reads one unit from the copy with `Read` offset/limit (via the listing), and never reads the whole file. Mutant: scratch copy of the retrieval guidance without D6 item 4, the session reads the whole file or does not report. | transcript excerpt (first line, the `Read` call arguments), the mutant transcript |
| TP-014 | The eight consuming agents can call both tools | AC-10 | contract | P2 | S15b1 landed | For each of `coder`, `tdd-engineer`, `architect`, `analyst`, `qa-engineer`, `frontend-engineer`, `tico`, `security-expert`, read `claude/<a>/<a>.md` frontmatter: either no `tools:` line, or a line naming both `mcp__falkor-chat-agent-team__search_documents` and `mcp__falkor-chat-agent-team__get_document`. Expected: 8 of 8 satisfy; `tico` and `security-expert` now list both. Mutant: remove one name in a scratch copy, that agent fails the same read. | per-agent table (agent, `tools:` state, both names yes/no) |
| TP-015 | Discovery pointer in each agent; audit check 12 | AC-13 | contract | P2 | S15b1, S15b2 landed | `grep -F` the fixed pointer literal (named in plan D7) across the eight agent files: 8 hits, one per agent (an existing-line case for `frontend-engineer`). Run `claude/scripts/audit-team.sh` on the real tree (check 12 green), then on a scratch copy of the tree with the literal removed from one agent (check 12 red, naming it). Expected: as stated; the real tree is never edited. | `grep -c` per file, both audit outputs and exit codes |

Exploratory charter (time-boxed, S11): after TP-001, query five units by an off-wording and one by an adjacent library fact, record the hit rank, and report anything that returns a unit above its own for a different gotcha. Findings are informational, recorded in the report as the exploratory charter result; it has no test-item ID, no pass/fail and no mutant.

## 5. Environment, data and Set B record

- Test workspace: `ws:pwq-test` on `http://localhost:8210` (`docs/plans/python-web-quirks-kb-graph.md`); a non-production run needs an explicit scratch `--manifest`. If `:8210` is down, restart per that document only after `GRAPH.LIST` confirms the graph exists; never `GRAPH.COPY` over it.
- Never: `:8200` or `--confirm-production` in S11; `ws:agent-team` writes; any Set B mode other than `--probe` before S14; a tree-mutating git command; the committed copy as a mutation target.
- Data: the copy of record, the pin `3fcb171cf03065f99aca800d9ea30ff6a2a7ba95`, Set A (committed fixture) for TP-001 situations only.
- **Set B record (S8, 2026-11-10):** stored outside the repo (path in D11); sha256 `163efb061834345d3d34c72501c96dd11e8c20fdd48bcbd433ba4c617d5fd15c`; 84 positives (2 per section 1-42), 30 negatives (15 near-domain, 15 off-domain); unique ids; read-only mode. Tail-flagged positives exist for all 12 units with at least 4 chunks (u02, u04, u08, u09, u11, u15, u21, u24, u26, u28, u29, u30); `validate_set_b` with `tail_unit_sections` passes. Leak check: 0 shared 4-grams over 84 positives. Probe (`eval --set B --probe`, `ws:pwq-test`, pre-ingestion): after one rewrite of six queries under the 0.25 tripwire (ruling (i)), 2 positives remain under it and are kept; 36 of 84 positives already surface a non-unit document under 0.43 (a pre-existing hit under floor, kept and recorded; the S7 capture list explains at most 22 of them, so most are unrelated team documents); no negative surfaces under 0.43. Per-query ids are in the S8 report to `teco`, not in a repo document.
- **S14 pre-checks, in the order `run_eval --sealed-once` runs them** (read from `claude/cobb/scripts/pwq_kb.py`; exit codes are `main`'s mapping: `KBError` 2, `TargetRefused` 3, `GateRefused` 4, `SealedRefused` 5):
  1. `sha256sum` of the B path equals the recorded value: the tool compares `--expect-sha256` first. A mismatch (or an unreadable B, or a missing/malformed `--expect-sha256`) is **exit 2** and burns nothing. A missing `--floor` is also exit 2.
  2. Set B shape and tail flags: the tool loads the **production manifest** (`--manifest`, default `claude/cobb/scripts/pwq-kb-manifest.json`) and the pinned source to derive the tail units, then validates B. An absent or unreadable manifest, or a B that fails shape or tail checks, is **exit 2**.
  3. The sentinel (`<B path>.scored`) is absent: present means **exit 5**, a second scoring is refused (B2 by a different author, never a re-run).
  4. Production gate clean: `--copy` must be readable (the expected unit ids come from it; unreadable is exit 2) and the gate must pass (an ABSENT, unembedded or failed unit is **exit 4**). The sentinel is written only after results exist, so exits 2, 4 and 5 before it burn nothing.
  Before invoking, and not enforced by the tool: (a) re-derive the S7 capture list at S14's start; `--exclude` must be a **subset of the ledger's ids deleted at H1 or created after S13b** (never an id still live in production that H1 did not delete, F11); (b) `--floor` explicit and equal to the value in the retrieval skill; (c) the S14 brief names the B path, no other brief does; (d) `--base-url http://localhost:8200` with `--ws agent-team` (a mismatch is exit 3). The report also gives the aggregate count of queries where a non-unit chunk outranks a unit chunk (no per-query lines; B stays sealed).

## 6. Entry and exit criteria

- S11 entry: S10 gate table clean; `:8210` healthy; G2 not "needs changes". Exit: TP-001 to TP-008 pass with their mutants red, or each failure is a reported defect with reproduction steps; the report records the planted divergence and the 43rd title.
- S15c entry: S15b1 and S15b2 landed (agent edits and audit check 12). Exit: TP-013 to TP-015 pass with their mutants red, or each failure is a reported defect; same report family, under a heading for S15c.
- S14 entry: TP-011 and TP-009 green, S13 gate clean, S15a floor change (if any) landed. Exit: AC-3 verdict from one `--sealed-once` run, reported with the unfiltered figures; defects and the BM25/tail findings in `docs/test-reports/python-web-quirks-kb-report.md`.

## 7. Out of scope

T1-T30 and the unit/contract layer; AC-1, AC-4, AC-9, AC-14 (other steps); retrieval quality tuning (S12); embedding model comparison; load or latency testing; any production write; re-scoring Set B. Not tested deliberately: whole-unit embedding (ML §7 says it does not occur), K-100 chunk-size behaviour for messages.
