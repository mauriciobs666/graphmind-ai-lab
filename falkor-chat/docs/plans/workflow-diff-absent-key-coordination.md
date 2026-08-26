# `workflow-diff-absent-key` — coordination

> **Status:** archived · **Owner:** `teco` · **Tracks:** K-005 (`claude/coder/kaizen/plan.md`)

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
- **Docs half integrated 2026-08-25, commit `259757d`** — `falkor-chat/AGENTS.md`, `falkor-chat/docs/HISTORY.md`,
  `claude/coder/kaizen/plan.md`, plus this coordination doc and the review doc: all fully
  isolated from the concurrent K-051 session, staged/committed by explicit path with a
  status re-check immediately before commit.
- **Code half (`repository.py` `_read_structure` fix + its 4 tests in
  `test_repository.py`) is verified/accepted but still uncommitted** — both files still
  carry the concurrent K-051 session's own unrelated, unfinished, uncommitted work mixed
  in on disk (that session was still expanding — into `background.py`, `embedding.py`,
  `ingestion.py`, `QUERIES.md`, `test_api.py`, `test_background.py` — as of this check),
  so a plain `git add <path>` on either file would sweep in work this coordination never
  reviewed. Leaving this coordination `active` (not `archived`) until that's resolved.
  Options once the K-051 session commits or otherwise clears: (a) if K-051 lands first,
  re-diff and commit the remaining K-005-only hunk cleanly; (b) if this needs to close
  first, a surgical `git add -p`/patch-based partial-file stage would be needed — not
  attempted here, treat as a live open question for whoever next touches this coordination,
  not a silent default.
- **2026-08-25, factual update from the K-051 session's own coordinator (`teco`, not this
  session):** option (a) above happened, but not cleanly — this code half is now
  **committed**, as part of K-051's integration commit `d41da78`, not a standalone
  follow-up commit. The K-051 coordinator staged only its own hunks correctly
  (`git apply --cached`, verified) but then ran `git commit -- <paths>` including
  `repository.py`/`test_repository.py` among the pathspecs — which re-stages a matching
  path's full *working-tree* content before committing, not just what was in the index, so
  this unit's already-`analyst`-accepted `_read_structure` fix + 4 tests rode along
  unintentionally. Confirmed present and byte-identical to what U1/U2 above verified
  (`git show d41da78 -- falkor-chat/server/falkorchat/repository.py`). No corrective git
  surgery was attempted (destructive, and the content is legitimate). Full detail in
  `document-status-terminal-coordination.md`'s own Close-out section. This coordination's
  code-half row (U1) can be treated as delivered-and-integrated; whoever next touches this
  doc should verify no local uncommitted `repository.py`/`test_repository.py` diff remains
  before assuming otherwise, and decide independently whether to flip this coordination's
  own `Status:` to `archived`.
- **Closed 2026-08-25.** Independently re-verified the note above rather than taking it on
  trust: working tree clean (`git status --short`, nothing pending), `d41da78` confirmed to
  contain the `_read_structure` try/except fix (`git show d41da78 -- repository.py`), and
  all four K-005 tests re-run fresh against committed `HEAD` — 4 passed. Both fixes (docs in
  `259757d`/`c62dffd`, code in `d41da78`) are integrated and verified. `Status` flipped to
  `archived`.
