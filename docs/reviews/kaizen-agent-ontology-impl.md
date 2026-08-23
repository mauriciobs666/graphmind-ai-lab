# Kaizen agent/learning-note ontology — Implementation Review (M8, S1)

> **Status:** archived · **Owner:** `analyst` · **Tracks:** — (M8)

**CPG:** considered, not relevant — this is a code-level task (`cypher-mcp/server.py`'s
`authorize_write()`), but `GRAPH.LIST` against the live FalkorDB instance (re-queried directly for
this review) shows only `kaizen_team`, `cpg_salesperson`, `cpg_falkorchat`, and several `ws:*`/
`reference` graphs loaded — no `cpg_cypher-mcp`/`cpg_claude` graph exists, so the `cpg-analysis`
skill has nothing to query here (same conclusion the plan's own `CPG:` line reached).

## Scope

Independent diff-scoped code review of `tdd-engineer`'s S1 delivery (U5 in
`docs/plans/kaizen-agent-ontology-coordination.md`) against the approved Version-3 plan
(`docs/plans/kaizen-agent-ontology.md`, approved Pass 3 in my own prior review,
`docs/reviews/kaizen-agent-ontology.md`) — a **different** review from that plan-gate one: this
judges the actual code, not the design. Producer ≠ reviewer.

Read/verified, by path and by execution, not by trusting `tdd-engineer`'s own report:

- `git diff cypher-mcp/server.py cypher-mcp/tests/test_server.py cypher-mcp/README.md`
  (uncommitted, working tree) — the full diff, all three files, read in full.
- The plan's §3.1/§3.1a design and §5 test list (23 items), re-read fresh against the diff, not
  from memory of my own three plan-review passes.
- My own prior review (`docs/reviews/kaizen-agent-ontology.md`, all 3 passes) — for Attacks A/B/C's
  exact reproduction text, re-traced here against the *implemented* code.
- `docs/plans/kaizen-agent-ontology-coordination.md` for unit/ledger context.
- Ran the offline suite (`cypher-mcp/.venv/bin/pytest tests -q`), the live suite (`-m live`, against
  the real, reachable FalkorDB on `:6379`), and independently re-performed all 4 claimed mutation
  tests myself (not merely read `tdd-engineer`'s table), with a `sha256sum` byte-identity check
  after each revert.

Not reviewed (out of the brief's stated scope): the 13 agent prompts, `skills/agent-maintenance/
SKILL.md`, `claude/README.md`/`claude/AGENTS.md` — none of these are touched by this diff (S3/S4/S5
are still queued per the coordination ledger), confirmed by `git status --short`. One unrelated,
pre-existing uncommitted change also sits in the working tree,
`falkor-chat/docs/requirements/document-ingestion.md` (+7 lines) — unrelated to this diff, not
reviewed, noted only so it isn't mistaken for scope creep on re-read.

**Verdict: approve.**

## Findings

No blockers, no majors. Two minor/nit items, neither gating.

### 1 (Minor, cosmetic — not gating) — a comment's absolute line-number citation is already stale

`server.py:474`: `# Same keyword set as \`_WRITE_KEYWORD_RE\` (line 239), minus \`CREATE\`...` —
`_WRITE_KEYWORD_RE` is actually defined at line 253 now, not 239. The docstring updates earlier in
the file (module docstring, `TOOL_DESCRIPTION`, `SERVER_INSTRUCTIONS`) added lines above this
comment, shifting every subsequent line number down by 14 without the comment being updated to
match — this is inherited verbatim from the plan's own §3.1a code block (which also says "line
239", written against the pre-diff file). Purely cosmetic (the keyword-set claim itself is still
correct — `_WRITE_KEYWORD_RE = re.compile(r"\b(CREATE|MERGE|SET|DELETE|REMOVE)\b", ...)` at line 253
does still equal `_FOREIGN_TRIGGER_RE`'s set plus `CREATE`), but a stale line number in a comment
that exists specifically to help a future reader cross-reference is a small paper cut waiting to
mislead someone. **Suggested improvement:** drop the parenthetical line number entirely (`"Same
keyword set as _WRITE_KEYWORD_RE, minus CREATE"`) rather than maintain a number that drifts on every
edit above it — cheap for `tdd-engineer` or whoever touches this file next to fold in.

### 2 (Minor, documentation hygiene — pre-existing, not introduced by this diff) — the "16 existing cases" count carried by the plan and my own prior review is actually 17

Both `docs/plans/kaizen-agent-ontology.md` (§2.4, §4 S1's done-condition) and my own
`docs/reviews/kaizen-agent-ontology.md` describe the pre-M8 write-authorization suite as "16
parametrized/individual cases." Counting `def test_` occurrences in `cypher-mcp/tests/test_server.py`
lines 562-815 (section 8, both at `HEAD` and in the current diff — the range is byte-for-byte
identical in both) gives **17**, not 16. This predates S1 entirely (confirmed identical count against
`git show HEAD:cypher-mcp/tests/test_server.py`) — it is not something `tdd-engineer` introduced or
miscounted; the "16" figure has simply been repeated across three plan-review passes and the plan
itself without anyone re-counting `def test_` lines directly. Doesn't affect this review's central
question — the byte-for-byte-unmodified claim holds regardless of whether the count is 16 or 17, and
I verified that directly via `git diff` (zero hunks touch lines 1-815 except one new `import time` at
line 16) rather than via the count. **Suggested improvement:** a cheap, low-priority fix for whoever
next touches this plan family — correct "16" to "17" in both documents, or drop the specific number
in favor of "the existing suite" to avoid re-drifting.

## What's verified solid

- **The deliberate deviation (kept the old `"neither an author-write..."` substring verbatim in the
  new 6-shape catch-all rejection, rather than replacing it per the plan's literal pseudocode) is
  sound and low-risk, confirmed by direct comparison, not just by the 4 pinned tests passing.** The
  plan's own §3.1 pseudocode shows the catch-all fully replaced (`"this write matches none of the
  recognized shapes — an author-write, a producer-write..."`); the shipped code instead extends the
  original sentence (`"this write is neither an author-write (...) , a producer-write (...), nor a
  recognized curator shape (...)."`). Traced: this is the same unconditional-rejection branch, fired
  under the same condition (nothing above it matched) — the set of *inputs* that reach this branch,
  and therefore get rejected, is unchanged; only the prose differs. `git show
  HEAD:cypher-mcp/server.py` confirms the exact pre-existing substring `"neither an author-write (no
  literal \`author: '{agent}'\` found inside a CREATE (...:KaizenEntry {{...}}) clause) nor the
  recognized curator-clear shape."`, and tests 7 (`test_unrecognized_write_shape_is_rejected`), 15/15b
  (the two `SET`-reassignment pins), and 16 (`test_nested_create_decoy_in_free_text_is_excluded`) —
  all four in the untouched pre-815 range — only assert `"neither an author-write" in out`, a
  substring that survives verbatim in the new message. No rejection case the plan intended is
  silently dropped: I independently re-ran the offline suite (all four pass unchanged) and manually
  traced the new message text against the plan's stated 6-shape enumeration — it does name all four
  new pieces (producer-write, curator shapes) that the plan's own catch-all names, just phrased as an
  extension of the old sentence rather than a fresh one.
- **`authorize_write()`'s new control flow matches the plan's §3.1 pseudocode line-for-line** — the
  ordering (`claims` check → foreign-trigger closure inside the matching-claims branch →
  `_producer_write_agent_id` → curator-clear → 3 new curator regexes → catch-all) is identical, and
  the three new curator regexes (`_MENTIONS_WRITE_RE`, `_PRODUCER_EDGE_RESOLVE_RE`,
  `_MENTION_EDGE_RESOLVE_RE`) are **byte-identical** to the plan's own code block (§3.1) — diffed by
  eye, character for character, no drift in translation from the design I already hand-verified
  against graph-dba's recipes across three review passes.
- **Attacks A, B, and both Attack-C sub-cases (SET and REMOVE) re-traced against the actual, running
  code, not trusted from the plan's claim or the test suite's green result.** I ran
  `authorize_write()` directly (via `.venv/bin/python3 -c ...`, importing the real `server` module)
  against all four attack reproductions verbatim from `docs/reviews/kaizen-agent-ontology.md`, plus a
  legitimate producer-write (match and mismatch) and a decoy-before-producer-write variant — every
  one rejects or authorizes exactly as designed. `_FOREIGN_TRIGGER_RE` is confirmed widened to
  `r"\b(?:MERGE|DELETE|SET|REMOVE)\b"` exactly as Pass 3 approved.
- **`_producer_write_agent_id`'s algorithm matches plan §3.1 steps 2a-2f exactly**, including the
  `\A`-anchored `MERGE` clause, the `re.escape(var)`-based binding check between `MERGE` and
  `CREATE`, the optional `{sessionId:...}` map via the shared `_scan_matched_brace()`, and the
  `\s*\)\s*;?\s*\Z` end-anchor for "nothing else follows." Verified by both reading and mutation
  (below), not by reading alone.
- **The existing pre-M8 test range (`test_server.py` lines 1-815) is confirmed byte-for-byte
  unmodified** — `git diff` on the test file produces exactly two insertion hunks (one two-line
  docstring/import hunk at the top adding `import time`, needed only for the new live fixture's
  polling loop; one pure-append block after line 815 for the new offline tests; one pure-append block
  after the pre-existing live section) and zero modification hunks anywhere in between. This is a
  stronger confirmation than re-reading test bodies by eye: a diff with no hunk in a range is proof of
  byte-identity in that range, not an inference from it.
- **Mutation-testing directive independently re-performed, not merely read from `tdd-engineer`'s
  report.** I could not replay their exact session, so I ran my own: for each of the 4 claimed
  mutations, patched `server.py` in place, ran the specific tests claimed to catch it, confirmed the
  failure, then restored the original file and verified `sha256sum` matched the pre-mutation hash
  before moving to the next. All 4 confirmed exactly as claimed:
  1. Dropping `\1`/`\2` backreferences from `_MENTIONS_WRITE_RE` → caught by
     `test_mentions_write_with_mismatched_backreference_is_rejected` (item 27/12).
  2. Removing the `_PRODUCER_WRITE_TRAILER_RE` end-anchor check in `_producer_write_agent_id` →
     caught by `test_producer_write_with_trailing_extra_clause_is_rejected` (item 22/7).
  3. Deleting the `_has_foreign_trigger_outside_strings()` call from shape 1's accept branch →
     caught by all four Finding-1 tests (18, 19, 22, 23 — Attacks A, B, C-SET, C-REMOVE).
  4. Narrowing `_FOREIGN_TRIGGER_RE` back to `r"\b(?:MERGE|DELETE)\b"` → caught **specifically** by
     items 22/23 (the SET/REMOVE-chained Attack-C variants), while 18/19 (Attacks A/B, which only
     use `MERGE`/`DELETE`) correctly kept passing — exactly the granularity the plan's mutation
     directive demanded ("must be caught by 22 and/or 23, not by 18/19").
  After all four reverts, `sha256sum server.py` matched the pre-mutation hash exactly, and the full
  suite re-ran green (106 passed, 10 deselected) with the restored file.
- **Full offline suite independently re-derived: 106 passed, 10 deselected** (`cypher-mcp/.venv/bin/
  pytest tests -q`) — matches `tdd-engineer`'s reported number exactly, not merely cited from their
  report. Baseline (via `git stash`, running the suite against `HEAD`) is 84 passed/7 deselected;
  delta is +22 offline / +3 live, matching plan §5 items 2-23 (22 new offline tests, confirmed by
  direct count: `grep -c "^def test_"` over the new block returns exactly 22) and the "Live,
  automated" section's 3 new shapes bundled across 3 new test functions.
  **Went further than the brief asked and also ran the live suite** (`-m live`, against the real,
  reachable FalkorDB on `localhost:6379`): all 10 live tests pass (7 pre-existing + 3 new — producer-
  write + traversal read, mismatched-agent rejection, and MENTIONS-write + both edge-resolves + final
  curator-clear on one disposable entry). Re-queried `GRAPH.LIST` afterward via the `cypher` MCP tool
  — no leftover `_cypher_mcp_selftest_*` scratch graph, and `kaizen_team` is unchanged, confirming the
  live fixture's cleanup (`graph.delete()`) actually ran and no real team data was touched.
- **README/docstring/`TOOL_DESCRIPTION`/`SERVER_INSTRUCTIONS` updates accurately describe all 6
  shapes**, not still describing 2 — checked each of the four spots individually: the module
  docstring (lines ~18-23), `TOOL_DESCRIPTION`, `SERVER_INSTRUCTIONS`, and `query()`'s own docstring
  all now enumerate producer-write, legacy author-write, MENTIONS-write, both edge-resolves, and
  curator-clear. `cypher-mcp/README.md`'s "Writing through this tool" section is rewritten with
  worked examples for all 6 shapes (renumbered 1-6), keeps the existing "schema DDL is rejected
  unconditionally" paragraph restated to also cover `Agent.agentId`'s index/constraint, and adds a new
  paragraph documenting the cross-clause-smuggling closure in prose, cross-referencing the review
  findings by name.
- **Scope containment is correct**: `git status --short` shows only the three files this review was
  scoped to (`cypher-mcp/server.py`, `cypher-mcp/tests/test_server.py`, `cypher-mcp/README.md`) as
  modified, plus the plan-family docs (already uncommitted from earlier units) as untracked — no
  agent prompt, no `SKILL.md`, no `claude/README.md`/`claude/AGENTS.md` touched, correctly leaving S3/
  S4/S5 for `cobb` as the coordination ledger's queued units show.

## Open questions

None — this review is conclusive. `tdd-engineer`'s S1 delivery matches the approved Version-3 plan
faithfully (including the one reported deviation, confirmed sound), the mutation-testing directive
was genuinely performed (independently re-confirmed, not just asserted), the pre-existing regression
suite is byte-for-byte unmodified, and both the offline and live suites are green. Ready for S2 to
close and S3/S4 to be dispatched per the coordination ledger.
