# Generic Cypher MCP — Coordination

> **Status:** active · **Owner:** `teco` · **Tracks:** — (M5 proposed, `docs/BACKLOG.md`)

## Goal & definition of done

Deliver `docs/requirements/generic-cypher-mcp.md` (Status: Ready for design, FR-1…FR-11 /
AC-1…AC-8). Turn `mcp__cpg__query`'s mechanism into a generic, graph-name-agnostic Cypher MCP tool
that can both read and write, and pilot it on one narrow slice: `graph-dba`'s kaizen working
memory moves from `inbox.md` (append-only markdown) to the graph as the raw-capture layer, while
`history.md` stays exactly as today and `inbox.md` is frozen after a one-time import. FR-8 adds a
two-shape write model (author vs. curator) on trusted, self-reported identity — no hardened auth,
matching the rest of the repo's trust level.

Design ownership split, mirroring the M4 precedent (`docs/plans/cpg-agent-adoption-coordination.md`):
`graph-dba` designs the graph data model first — entry schema (FR-7's fields), the attribution data
shape FR-8's enforcement will sit on top of, curator-clear semantics (FR-9), and the migration
shape for FR-3 — because it's a narrow, self-contained technical question with no dependency on
the tool-mechanism decisions. `architect` designs the MCP tool mechanism second (FR-1 tool shape,
FR-8 enforcement logic, FR-4 frozen-marker signal, container/build implications, BACKLOG.md M5
proposal, and the implementation step table), citing `graph-dba`'s delivered note by path so the
two are coherent. Both plans then go through one combined `analyst` plan-gate before any
implementation is dispatched; a second, diff-scoped `analyst` re-gate follows implementation per
standing practice.

## Unit ledger

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict |
|---|---|---|---|---|---|
| U1 | `graph-dba` | `aad409ffcd504fc08` | delivered | `docs/plans/generic-cypher-mcp-graph.md` — verified: `:KaizenEntry` schema (5 markdown fields + `entryId`/`author`/`createdAt`), `author` as plain property (not `:Agent` node), curator-clear = hard `DETACH DELETE` by `entryId` with append-before-delete ordering flagged as non-negotiable, 6-row `UNWIND` migration (all 6 current inbox entries confirmed to map cleanly), `entryId` index+uniqueness constraint, footprint negligible/bounded-by-design. Graph key `kaizen_graph_dba` proposed, flagged overridable. §7 gives U2 a clean summary. | — |
| U2 | `architect` | `af0c8170c622adc00` | delivered | `docs/plans/generic-cypher-mcp.md` — verified: extends `cpg/mcp/server.py` in place (1 tool, +optional `agent` param), write-detection reuses graph-dba's live-verified "ro_query"/"empty key" probe technique, FR-8 enforcement = static regex on `author:` literals + one curator-clear skeleton (`CURATOR_AGENTS` env-configurable), FR-4 frozen note drafted verbatim, `.mcp.json` unchanged, `cpg-query-access.md` AC-8 edit scoped to header-only (correctly reasoned from its `archived` status), BACKLOG.md M5/C-501..505 proposed, 5-step table (coder×2, graph-dba, cobb, qa-engineer) with 12-case unit test list + AC-1..8 strategy. §9 explicitly flags 5 risks/open items for the plan gate, including an open question inviting analyst pushback on the regex-enforcement approach. | — |
| U3 | `analyst` | `adab0c51beb8447d6` | delivered | `docs/reviews/generic-cypher-mcp.md` — combined U1+U2 plan gate. U1 (graph-dba's schema) holds up fully, no findings. U2 (architect's tool design) has: **B1** (blocker) step 4's file list omits `claude/graph-dba/graph-dba.md` and `claude/cobb/cobb.md` — both agents' own *operative* prompts still instruct the pre-feature behavior (append to inbox.md), so graph-dba would keep writing to the frozen file on its first post-delivery run regardless of code correctness; **M1** (major) `_AUTHOR_LITERAL_RE` scans the whole query text, so an `author:`-shaped substring embedded in free-text `evidence`/`context` can cause a false rejection of a legitimate write; **M2** (major) the author-write shape authorizes `SET .author=` against any matched node, not just newly-created entries — broader than FR-8's "creates new entries attributed to itself only," inert at 1-author pilot scale but activates the moment FR-10 (2nd author) lands with no further review gate; **M3** (major) the "empty key" branch can't distinguish a genuine read from a write once `agent` is (accidentally) supplied, misrouting a plain read against a missing/mistyped graph into a confusing write-rejection message; 2 minors (Superseded-by field used cross-topic-slug; test list gap for M1/M2). | plan gate → **needs changes** |
| U2-fix | `architect` | `af0c8170c622adc00` | delivered | `docs/plans/generic-cypher-mcp.md` (Version 1.1) — teco read in full. B1 fixed: step 4 split into 4a/4b, adds `claude/graph-dba/graph-dba.md`/`claude/cobb/cobb.md`/`claude/README.md`, close-out is now a `grep`-based before/after sweep, not a fixed list. M1 fixed: `_author_claims()` scans only inside a `CREATE (...:KaizenEntry {...})` map body, excluding nested string-literal spans (decoy substrings in evidence/context no longer misread). M2 fixed: same redesign categorically excludes `SET`-based reassignment (zero spans found → always rejected). M3 fixed: `_looks_like_write()` keyword-scans the Cypher text before entering enforcement on the "empty key" branch — `agent` alone no longer implies write intent. m1/m2 addressed (plain `**Note:**` prose instead of `Superseded by:`; tests 14/15 added). Open question resolved with stated rationale (ordering lives only in cobb's side). Dated §10 revision note ties every change back to its finding. | — |
| U3-regate | `analyst` | `adab0c51beb8447d6` | delivered | `docs/reviews/generic-cypher-mcp.md` §"Pass 2 — 2026-08-17" — teco read in full. Verification was by **executing** the plan's own Python (transcribed into a scratch script), not just re-reading claims. B1/M2/M3/m1/m2 confirmed closed by direct execution/re-read. **M1 only partially closed — new Major (M1-residual)**: `_kaizen_entry_create_map_spans`'s CREATE-keyword *location* step isn't string-literal-aware, so a free-text field containing a complete `CREATE (...:KaizenEntry {author:...})`-shaped substring is misread as a second real clause — causing both a false-rejection variant and a more serious false-acceptance/under-enforcement variant (a decoy authorizing a write whose real top-level clause has no `author:` at all). Analyst supplied and verified a one-method fix (whole-text `_string_literal_spans` pre-filter before locating CREATE candidates) with before/after execution proof, no regression against tests 8/14/15. New minor m3 (sweep-noise framing understates ~35-file triage volume vs. "five named files" phrasing). | plan gate → **needs changes** (Pass 2) |
| U2-fix2 | `architect` | `af0c8170c622adc00` | delivered | `docs/plans/generic-cypher-mcp.md` (Version 1.2) — teco spot-checked `_kaizen_entry_create_map_spans()` directly: analyst's verified whole-text `_string_literal_spans` pre-filter landed exactly as specified (CREATE matches inside a string-literal span are now skipped before body-extraction). Test 16 added reproducing both adversarial repro strings from Pass 2. m3 addressed (close-out note now sets ~30-hit expectation). §11 Pass-3 revision note added. | — |
| U3-regate2 | `analyst` | `adab0c51beb8447d6` | accepted | `docs/reviews/generic-cypher-mcp.md` §"Pass 3" — re-verified by re-executing the patched functions from `generic-cypher-mcp.md` Version 1.2 directly (not eyeballing the diff): both Pass-2 adversarial cases now behave correctly, all 8 prior regression cases unchanged, test 16 matches executed behavior, m3 wording confirmed adequate, full 906-line re-read found no drift elsewhere. | plan gate → **approve** |

Plan gate closed (U3/U3-regate/U3-regate2, 3 passes, final verdict **approve**). Implementation
units below are sized 1:1 against `docs/plans/generic-cypher-mcp.md` §7's own step table (already
sized to the team's ≤3-file/≤3-step boundary by the architect) — steps 1+2 dispatch together
(same owner, no dependency between them, well under the sizing threshold); 4a+4b dispatch together
(same owner, same dependency on 3, plan explicitly calls them an adjacent cluster).

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict |
|---|---|---|---|---|---|
| U4 (steps 1+2) | `coder` | `a65b0a5578cddadcb` | delivered | `cpg/mcp/server.py` (+333/-…), `cpg/mcp/tests/test_server.py` (+298, 83 passed/7 deselected offline, 74/7 + 7/74 live in-container), `cpg/mcp/README.md` (+89/-…, param table + "Writing through this tool"); `docs/requirements/cpg-query-access.md` header-only edit (5 lines, teco-verified via direct `git diff`) — teco verified file scope matches self-report exactly, both required mutation-tests (M2 SET-reassignment, Pass-3 nested-decoy-CREATE) run with captured failure output then reverted, `build.sh` run twice (final tag `2dadf10c24b0`), in-container offline+live gates green, one genuinely stale pre-existing live test (`test_live_write_is_rejected_server_side`) found and fixed along the way (flagged, not silent). | code re-gate (analyst) → **approve with suggestions** |
| U4-fix | `coder` | `a65b0a5578cddadcb` | accepted | `cpg/mcp/tests/test_server.py` — added `test_set_map_merge_author_reassignment_is_always_rejected` (colon-form `SET e += {author: '...'}` against a matched, not newly-created, node) as the real M2 regression pin; kept dot-assignment test 15 for independent coverage. Reproduced the M2 mutation surgically (CREATE-scoping only, M1 decoy exclusion preserved) — new test failed as expected, test 15 still passed (confirms M-A diagnosis exactly), reverted, `server.py` byte-identical to backup. teco independently confirmed: `server.py` diff untouched, `test_server.py` +19 net lines, both tests present, suite green **84 passed/7 deselected**. | spot-check (teco) → **accepted** |
| U5 (step 3) | `graph-dba` | `a9e581d8ad9a4690e` | delivered | User reconnected `/mcp` (session-level, unblocking); `graph-dba` re-verified the connection was genuinely fresh (new container `gifted_shamir`, image `cpg-mcp:7c4bf453a970`, updated server-instructions banner) before trusting it, then replayed the migration for real: `write ok (labels_added=6, nodes_created=6, properties_set=48)`. teco independently verified via direct `redis-cli`: `kaizen_graph_dba` has exactly 6 `KaizenEntry` nodes; `entryId` index + uniqueness constraint both `OPERATIONAL`; `inbox.md` frozen note present (dated 2026-08-18, correct — plan's placeholder date not used), `history.md` untouched, diff scope exactly `inbox.md` + `graph-dba`'s own `falkordb-quirks.md` kaizen entry, matching the self-report. One live-verified plan-text defect found and worked around: `docs/plans/generic-cypher-mcp.md` §3.4/`generic-cypher-mcp-graph.md` §5's literal `GRAPH.CONSTRAINT ... UNIQUE LABEL KaizenEntry ...` fails on this build ("Invalid constraint entity type") — this build's keyword is `NODE`, not `LABEL`; corrected live, captured in `falkordb-quirks.md` as the durable record (plan doc itself is past its amendable window — already approved and executed against — so left as-is per doc-lifecycle convention, correction lives in the actively-maintained KB instead). | — |
| U6 (steps 4a+4b) | `cobb` | `ae9d6979fdd1f8986` | accepted | 9 files: `claude/AGENTS.md`, `claude/README.md`, `docs/BACKLOG.md` (M5 section, C-501…C-506), `claude/graph-dba/graph-dba.md`, `claude/cobb/cobb.md`, `skills/agent-maintenance/SKILL.md` §5 (graph-dba's 4-step distillation sequence, append-before-delete ordering lives only here per plan §3.5), both agents' `kaizen/history.md`, root `AGENTS.md`'s `claude/` bullet (self-flagged gap, folded in). Grep-sweep: 36/36, all triaged. `audit-team.sh`: 96 PASS, 2 pre-existing unrelated FAILs. | code re-gate (analyst) → **approve with suggestions** |
| U6-fix | `cobb` | `ae9d6979fdd1f8986` | accepted | Fixed both re-gate findings: M-B (`docs/BACKLOG.md` C-501…C-505 flipped `🔵`→`✅` per M4 precedent `50f9aaa`, C-506 correctly left `🔵`, milestone row `🟡` with outstanding-work note) and m-B (`claude/cobb/kaizen/history.md` U6 entry extended to mention the root `AGENTS.md` edit). teco independently confirmed via `git diff` — exact match to self-report, only `docs/BACKLOG.md` + `claude/cobb/kaizen/history.md` touched. | spot-check (teco) → **accepted** |
| U7 (step 5) | `qa-engineer` | `a5d8991a5e8bb278f` | accepted | `docs/test-plans/generic-cypher-mcp.md` (TP-001…TP-008) + `docs/test-reports/generic-cypher-mcp-report.md` — all 8 ACs PASS under live exercise, no defects. AC-5 (the one requiring a real dispatch, not a stand-in) genuinely ran `cobb`'s full 4-step distillation on a real migrated entry — teco independently re-verified: graph count 6→5, `history.md` new entry, `cpg-model.md` knowledge-base edit both present and correct. AC-4 self-cleaned (6→7→6), zero permanent graph pollution beyond the one legitimate AC-5 promotion. teco cross-checked graph count and both file diffs directly — matches report exactly. | acceptance → **PASS** |

Sequencing (per plan §7): U4 → U5 → U6 → U7. U4's two steps have no dependency on each other so
dispatch together; U5 depends on U4 (needs the live write path deployed); U6 depends on U5 (docs
must describe graph-dba's actual, now-migrated behavior); U7 depends on all four.

## Notes

- Requirements doc's own "Context for the architect" note explicitly flags FR-8's enforcement
  mechanism and FR-4's frozen-marker signal as architect calls, not specified by the stakeholder —
  U2's brief carries this verbatim.
- `docs/BACKLOG.md` numbering: hundreds digit = milestone (C-4xx = M4), so this delivery is
  **C-5xx** under a new `## M5 — …` section, per M4 precedent — folded into U2's step table as a
  proposal, actually added to BACKLOG.md by whichever implementation unit closes out docs (mirrors
  M4's U4b-5).

## Milestone close (post-U7)

U7 (`qa-engineer`) passed all 8 ACs with no defects — delivery is complete. Closing units mirror
the M4 precedent exactly (`1101d07`, `4cd45ae`, `8517197`, `a1ebd9f`): per-document `Status:
archived` flips routed to each doc's owner per root `AGENTS.md`'s by-kind table, then a
`docs/BACKLOG.md`/`docs/HISTORY.md` close-summary commit.

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict |
|---|---|---|---|---|---|
| C1 | `tico` | `a324675f67bf66538` | delivered | `docs/requirements/generic-cypher-mcp.md` — teco-verified via `git diff`: single-token `Status:` flip (`Ready for design`→`archived`), nothing else touched. | — (mechanical, no gate) |
| C2 | `architect` | `a4df08b378d413d75` | delivered | `docs/plans/generic-cypher-mcp.md` — teco-verified via `git diff`: single-token `Status:` flip (`active`→`archived`), nothing else touched. | — (mechanical, no gate) |
| C3 | `graph-dba` | `acec67709b24433b5` | delivered | `docs/plans/generic-cypher-mcp-graph.md` — teco-verified via `git diff`: single-token `Status:` flip (`active`→`archived`), nothing else touched. | — (mechanical, no gate) |
| C4 | `analyst` | `a7345d743610a128c` | delivered | `docs/reviews/generic-cypher-mcp.md` — teco-verified via `git diff`: single-token `Status:` flip (`active`→`archived`), nothing else touched. | — (mechanical, no gate) |
| C5 | `qa-engineer` | `a25a00ee86d9239fa` | delivered | `docs/test-plans/generic-cypher-mcp.md` + `docs/test-reports/generic-cypher-mcp-report.md` — teco-verified via `git diff`: single-token `Status:` flip (`active`→`archived`) on both, nothing else touched. | — (mechanical, no gate) |
| C6 | `cobb` | `ad237c56b9bd79517` | in-flight | `docs/BACKLOG.md` (C-506 `🔵`→`✅`, M5 milestone-map row `🟡`→`✅`, gate-status prose updated to record the full U1…U7 sequence, mirroring `a1ebd9f`'s pattern) + `docs/HISTORY.md` (dated M5 close entry). | — |
| C7 | `teco` | (self) | queued | `docs/plans/generic-cypher-mcp-coordination.md` — this file, flip `Status:` to `archived` once C1…C6 land. | — |
