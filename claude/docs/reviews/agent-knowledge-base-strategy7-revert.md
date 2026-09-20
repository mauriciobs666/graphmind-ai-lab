# K-030 item 2 — revert review (U7)

> **Status:** active · **Owner:** `analyst` · **Tracks:** K-030 (M5)

## Scope & verdict

Diff-scoped review of `tdd-engineer`'s revert of the hybrid lexical+semantic RRF fusion feature
(K-030 item 2, commits `af535ecb` + `b56e80ea`), currently uncommitted on disk. **Not** a
re-review of whether reverting was the right call — that's settled (user + independent
`architect` consult both confirmed revert; see
`claude/docs/plans/agent-knowledge-base-strategy7-coordination.md` U5/U6). In scope: is the
revert itself clean, complete, and safe — the seven mechanically-reverted files, the deliberate
non-blind `DESIGN.md` edit, the new `HISTORY.md` entry, suite health, and leftover-symbol sweep.
Baseline: `af535ecb^` (pre-fusion state) plus the intentional `Document.title` fix `af535ecb`
bundled in.

**Verdict: approve.**

**CPG:** considered, not relevant — `cpg_falkorchat` is stale (parsed 2026-09-18, predates both
the fusion commits and this revert), and a revert-to-known-prior-state is verified more reliably
by direct diffing against the pre-fusion commit than by graph traversal of a stale snapshot; used
direct `git diff`/`grep`/pytest instead, per the brief's own guidance.

## Findings

No blockers or majors. Two informational notes, no action required from this gate.

**1. (info) Byte-identical revert confirmed independently on all seven files.**
`git diff af535ecb^ -- falkor-chat/server/falkorchat/{services,repository,mcp}.py
falkor-chat/scripts/bootstrap_schema.sh falkor-chat/docs/QUERIES.md
skills/agent-kb-retrieval/SKILL.md falkor-chat/server/tests/test_services.py` produces zero
output. Re-confirmed myself rather than taking `teco`'s prior confirmation on trust.

**2. (info) `DESIGN.md`'s non-blind judgment call is correct and grounded.** Pre-`af535ecb`
(`git show af535ecb^:falkor-chat/docs/DESIGN.md`), the full-text register listed only
`Message.text`/`Entity.name` — `Document.title`'s fulltext index had existed in
`bootstrap_schema.sh` since commit `0c0fa4aa` (document-ingestion2 Stage D) but was never
documented. `af535ecb` fixed that omission in the same paragraph it added the `Chunk.text`/fusion
prose. The current working-tree text (`falkor-chat/docs/DESIGN.md:616-621`) keeps exactly the
`Document.title` fix and drops exactly the `Chunk.text`/fusion additions — verified against
`bootstrap_schema.sh`'s actual `echo "[fulltext] ..."` lines (`Message.text`, `Entity.name`,
`Document.title`; no `Chunk.text` remains, `grep -n 'fulltext\]' bootstrap_schema.sh`). Right
call: a blind `git checkout af535ecb^ -- DESIGN.md` would have silently reintroduced the stale
omission the fusion commit had legitimately fixed.

## What's solid

- **`HISTORY.md` accuracy, verified against source evidence, not narrative alone.** The rewritten
  entry (`falkor-chat/docs/HISTORY.md:120`) claims: G2 returned zero admissible candidates (not
  merely low-ranked); C2 regressed from a clean rank-1 hit to absent-from-top-5; and a
  RediSearch-syntax-error crash on "roughly 9%" of realistic queries. Cross-checked each against
  `claude/docs/plans/agent-knowledge-base-strategy-ml.md`'s "Item 2 — U5 re-test, second attempt"
  section: G2 table row confirms "0 rows returned — nothing cleared the gate at all"; C2's
  regression table confirms "absent from top-5"; the crash-rate claim traces to "hitting 4 of 45
  rows" (4/45 = 8.9%, rounds to "roughly 9%," matches). All three claims check out.
- **The immediately-preceding, unrelated `3c975f39` embedding-migration entry is untouched.**
  `git diff 3c975f39 -- falkor-chat/docs/HISTORY.md` shows the diff begins exactly at the K-030
  item 2 heading; every line of the embedding-migration entry above it is byte-identical.
- **Full suite green, no fusion residue.** `.venv/bin/python -m pytest -q` in
  `falkor-chat/server`: 2876 passed, 14 deselected, 0 failed. `grep -n '^def test_'
  tests/test_services.py | grep -i 'rrf\|fusion\|fulltext\|lexical'` returns nothing. `ruff
  check` clean on `services.py`, `repository.py`, `mcp.py`, `tests/test_services.py`.
- **Leftover-symbol sweep clean.** Grepped `_fuse_chunk_hits_rrf`, `search_chunks_fulltext`,
  `RRF_K`, `HYBRID_OVERFETCH_K`, `VECTOR_ADMISSIBILITY_FLOOR`, `LEXICAL_ADMISSIBILITY_RANK`,
  `rrfScore`, `vectorRank`, `lexicalRank` across `falkor-chat/server`, `falkor-chat/scripts`,
  `skills/agent-kb-retrieval`, `docs/QUERIES.md`, `docs/DESIGN.md` — zero hits. A monorepo-wide
  sweep surfaces hits only in `HISTORY.md`'s own narrative and in
  `falkor-chat/docs/plans/embedding-migration-coordination.md:74` (a different, concurrent
  session's own document, passing mention, correctly out of scope per the brief) plus this
  feature's own plan/review history (`agent-knowledge-base-strategy{-ml,-graph,7-impl,7-coordination}.md`,
  `agent-knowledge-base-strategy7-{diff,impl}.md` reviews) — expected: a record of what was built
  and reverted doesn't get scrubbed because the code was reverted.
- **Nothing committed, nothing out-of-scope touched.** `git log --oneline -1` still shows
  `01dadf5a` (no new revert commit). The only files touched relative to `HEAD` are the seven
  mechanically-reverted files, `DESIGN.md`, `HISTORY.md`, plus two pre-existing unstaged edits
  (`claude/docs/plans/agent-knowledge-base-strategy-ml.md`,
  `...strategy7-coordination.md`) that were already modified before this review started (U5's
  analysis and `teco`'s own ledger) — not part of `tdd-engineer`'s revert unit, confirmed by
  their content (U5 evaluation narrative, ledger rows) rather than assumed from the file list
  alone.

## Open questions

None for this gate. Noted for the record, already tracked and sequenced correctly in
`claude/docs/plans/agent-knowledge-base-strategy7-coordination.md` (not a finding against this
unit): the live `ws:agent-team` MCP server (PID 436673) still serves the pre-revert fusion code
in production until `U9` restarts it, and `U9` is already correctly sequenced behind this gate's
approval and the eventual commit — no action needed from `analyst` here.
