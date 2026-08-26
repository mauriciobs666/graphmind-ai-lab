# `workflow-diff-absent-key` — coordination

> **Status:** active · **Owner:** `teco` · **Tracks:** K-005 (`claude/coder/kaizen/plan.md`)

## Goal

Fix a false-negative in `verify_workflows.sh`/`services.diff_def_snapshot`: after
`test_queries.sh` fully `GRAPH.DELETE`s the `reference` graph (not just empties it), a
read against the now-nonexistent key raises an uncaught `redis.exceptions.ResponseError`
("empty key") deep inside `Repository._read_structure`, which propagates out of
`diff_def_snapshot` entirely and gets caught by `verify_workflows.sh`'s outer `read()`
wrapper — collapsing the **whole** diff to the `ABSENT` sentinel
(`defPresent: False, snapshotPresent: False`) even though only the `reference` side
actually failed; the `ws:<id>` **snapshot** read never runs (or its result is discarded),
so an intact snapshot is misreported as missing too. Root cause already independently
verified (not just re-confirmed the citation) by `coder` during the team kaizen
distillation pass — see `claude/coder/kaizen/plan.md` K-005 for the full trace, this doc
does not restate it. No RCA unit needed; root cause is settled, this coordinates the fix.

**Two fixes, not decided by K-005 — left to the implementer's judgment, see brief:**
1. **Code** — stop the exception from defeating both presence checks; return `None`
   per-side on an absent key, matching what `diff_def_snapshot`'s own docstring already
   promises.
2. **Docs** — `falkor-chat/AGENTS.md`'s `test_queries.sh` row currently says "re-run
   `seed_workflows.sh <wsId>` afterward," which is incomplete once `reference`'s
   indexes/constraints were destroyed by the `GRAPH.DELETE` too (needs the full
   `bootstrap_schema.sh` → `seed_demo.sh` → `seed_workflows.sh` sequence); the
   `verify_workflows.sh` row should note the false-negative-on-full-delete case so an
   operator doesn't mistake it for real data loss.

**CPG note:** `cpg_falkorchat` exists but is **stale** for this unit — built
2026-08-17T00:40:42Z, scratch-staged (`sourcePath /tmp/cpg-src/falkor-chat-server`, no
`sourceCommit`), and `git log --oneline --since=2026-08-17T00:40:42Z -- falkor-chat/server`
shows 14 commits since, including several feature milestones. Don't lean on it for this
unit — the affected functions are pinned by file:line below; a live `Read` is authoritative
over the graph here.

**Known concurrent-tree wrinkle:** `falkor-chat/server/falkorchat/repository.py` and
`falkor-chat/server/tests/test_repository.py` already carry **unrelated, uncommitted**
changes from another in-flight session (K-051 `document-status-terminal` work — see the
untracked `falkor-chat/docs/plans/document-status-terminal-coordination.md`), touching
`Document`/`pendingJobs` code around line ~1026–1120. This unit's edit site is
`_read_structure` (~line 1717) and its two callers (~1817, ~2525) — a different region of
the same file. No line-range collision expected, but the eventual commit for this unit
will need care (see step 5 note below) since `git diff`/`git add <path>` on this file will
currently show **both** sessions' changes mixed together.

## Ledger

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| U1 | `tdd-engineer` | `ae8f61957e01bd1fd` | accepted | `repository.py` `_read_structure` fix, 4 new tests in `test_repository.py`, `falkor-chat/AGENTS.md` doc notes (both rows), `falkor-chat/docs/HISTORY.md` entry, `claude/coder/kaizen/plan.md` K-005 → ✅ | `analyst` → needs changes (1 Major), fixed & reverified | 145.1k+147.8k tok, 75+6 tools |
| U2 | `analyst` | `af78de5ae00579a76` | accepted | `falkor-chat/docs/reviews/workflow-diff-absent-key.md` | — | 83.4k tok, 32 tools |

**U2 verdict: needs changes (1 Major).** `verify_workflows.sh`'s `AGENTS.md` row (fix
item 2's second half) was never edited, but `HISTORY.md`'s new entry claims it was.
Fix dispatched back to `tdd-engineer` (same agent, resumed by id) as a same-unit
follow-up per the review's own recommendation — not a new unit.

## Notes

- Below the 3-unit auto-threshold, but opened anyway because U1 carries a review gate.
- Integration commit is deferred pending a clean way to separate this unit's hunks from
  the concurrent session's uncommitted K-051 changes to the same two files — see the
  wrinkle above. Resolve at step 5, not before.
