# Team learnings distillation 3 — review

> **Status:** active · **Owner:** `analyst` · **Tracks:** — · **Extends:** `docs/reviews/kaizen-inbox-distillation2.md`

## Scope & verdict

Reviewed the uncommitted diff (22 files) of `cobb`'s distillation sweep against `docs/reviews/team-learnings-distillation3-sweep.md` and `docs/plans/team-learnings-distillation3-coordination.md`; root `REVIEW.md` applied (no component `REVIEW.md` is touched). Not mutated: git tree, `ws:agent-team` documents.

**Verdict: needs changes** (one blocker on the acceptance criterion in (d), two majors on false/stale text in promoted content).

CPG: used cpg_falkorchat — re-ran the cpg-analysis skill's `post_message` caller query and compared it with a grep of production code.

## Findings

### B1 (blocker, verified) — 6 of the 7 ingested KB claims are `failed`, not `ready`
Queried `ws:agent-team` by id. Only `50dc6b7e` (ops-quirks 13.1 s) is `ready`. `failed`: `940f699c` (readiness replacement; 4 chunks, 0 embedded), `bbd7665b` (heredoc ask; 0 embedded), `ba4488e6` (ORDER BY; 0), `7beddce9` (full-text; 1 of 2), `bf9fa3f2` (LM Studio; 2 of 3), and the fence replacement `f4cfc792` (1 of 2). `pendingJobs=0` so none will recover on their own. The server log shows `HTTP 400 ... Model has not started loading/has been unloaded` from the embedding endpoint (another session unloaded the embedder), so cause is environmental, but the sweep's acceptance, "confirm they reach ready" (manifest `_note`, K-061), is unmet. The sweep report says "all but one were still `processing`", which is now stale.
Consequence: the stale `ddfbb4bb` ("bring FalkorDB up yourself") is still `ready` and served, while its replacement `940f699c` cannot be retrieved. `search_documents` therefore returns the superseded advice.
Fix: with the embedder loaded, delete and re-ingest each failed claim one at a time (the pattern in the manifest's 2026-09-19 retry notes), then delete `ddfbb4bb` once `940f699c` is `ready`. Update the new manifest ids. Re-check status after a delay.
Also: the manifest key `_cobb_distillation3_2026_11_06` lists 6 ids, not 7. The 7th (the fence claim `f4cfc792`) was swapped into `_teco_coordination_techniques`. The `_note` says "All seven"; say "six here, one in place".

### M1 (major, verified) — SERVER.md bullet describes tells of a reverted feature
`falkor-chat/docs/SERVER.md` new bullet: "Two tells for the hybrid fusion: `rrfScore`/`vectorRank`/`lexicalRank` absent from every row...". Hybrid RRF fusion was reverted (`7a270329`, 2026-09-20). `git grep lexicalRank -- server/falkorchat` finds nothing, and the only hit in `SERVER.md` is this new line. A reader would conclude a healthy live server is stale. Also unverified: "a schema/DDL change is visible to a fresh connection at once".
Fix: drop the "two tells" sentence (keep the process-start-time check), or restate it for the current vector-only `search_documents`. Drop or mark the DDL sentence as unverified. This is a live spec, so keep it free of dated observations beyond what it needs.

### M2 (major, verified) — LM Studio claim states a superseded fact as current
`claude/data-scientist/lm-studio-model-notes.md` (and ingested `bf9fa3f2`): "the salesperson storefront calls `ministral-3-3b` and `qwen3-4b-2507`, plus the embedding model". That is the situation `df558abf` described, which cobb itself rejected (row 36) as superseded by K-066 (`ad01aee7`, 2026-10-22; `tools.py:1198` passes `run.get("stepModel")`; `falkor-chat/AGENTS.md` says qwen is outside the demo set). The source rerun (2026-10-20) predates K-066. The claim is internally inconsistent with the sweep's own rejection.
Fix: state the residency hazard without the two-model premise (ministral plus the embedder; qwen is only the `:8200` agent-team default), or date it "before K-066".

### m1 (minor, verified) — rejection of row 21 is right, but for a different reason than reported
Report: "fixed; a query with `()`, `$()`, hyphens and `%` returned results today". I reproduced the no-error result (`search_documents` with `model-bench (rank) $(x) 100% a-b %_%` returned 3 rows). But there is no lexical path left to fix: the fusion that fed raw text to RediSearch was reverted in `7a270329`; the `Chunk` label has no full-text index (`db.indexes()`). The rejection stands (entry describes code that no longer exists), but a future reader should not conclude the escaping was fixed. Note the same RediSearch hazard does live in `Message`/`Document`/`Entity` full-text calls, which the promoted `falkordb-quirks.md` claim covers.

### m2 (minor, verified) — sweep report adds a home-path leak
`docs/reviews/team-learnings-distillation3-sweep.md` line 91 contains the literal home path (the only untracked file that does). The audit's leak count therefore rises by one from the claim's own artefact (I measured 147 files / 744 lines; the report says 146 / 743). Not the cause of the pre-existing failure (see (e)), but genericize it.

### n1 (nit, verified) — "sub-30 ms"
The cpg-analysis addition says the caller query is "sub-30 ms". Cold: 39.8 ms, warm: 12 ms. Say "tens of ms".

## Checks on the brief's items

**(a) Promoted claims.**
- python-web-quirks `BackgroundTasks`: `inspect.getsource` in the server venv prints `for task in self.tasks: await task()`. Verified.
- `InvalidURL`: MRO `InvalidURL -> HTTPException -> Exception`, `issubclass(..., ValueError)` is `False`, `urlopen('http://localhost:abc/x')` raises `InvalidURL: nonnumeric port: 'abc'`; `transport.py:159` has rung 3a before rung 3. Verified.
- falkordb-quirks ORDER BY: re-ran with `UNWIND` of maps on `ws:test`. ASC gave a, b, c, null, null; DESC gave null, null, c, b, a. Verified; the "tie order not guaranteed" half is a negative claim and properly worded. Full-text claim: cobb's live probes cited; I did not re-run `%a-b%` or `%__%` per character (inferred from K-068/K-075 and `fusion.py` docstring), only confirmed the tokenization-related `Entity.name`/`Document.title` indexes exist.
- cpg-analysis caller exhaustiveness: the query returns 3 production callers (`api.py`, `mcp.py`, `storefront_api.py`) plus tests; a grep for `post_message(` call sites in `server/falkorchat` finds the same three. The stated bound (by name, as of build, not dynamic dispatch) is accurate and the CPG's build date (2026-10-31 artifacts) is the right thing to check. Verified; the claim is correctly bounded.
- `:8200` log holds only the current launch: first line of the log is the 2026-10-15 launch header. Verified. `start_agent_team.sh` states `--reload` OFF (lines 73, 108, 155). Verified. The "two full runs failed different sets (2026-09-20)" claim is an observation I cannot re-derive; it is worded as observed and bounded, acceptable.
- `PostToolUseFailure`: documentation-based, flagged unverified for Bash non-zero. Honest bound; no objection.

**(b) Rejections sampled.** Row 36 (storefront two LLMs): rejection upheld (`tools.py:1198-1199`, `executor.py:936-941`, commit `ad01aee7`), but see M2 for its inconsistency with the promoted text. Row 21: upheld, see m1. Row 15 (truncated cells) corroborated: I saw `(+1202 chars)` and `(+1019 chars)` in this run. Not sampled: rows 3, 8, 10, 17, 30.

**(c) Size discipline.** `claude/AGENTS.md` changed by one in-place rewritten line (+12 words, 3,417 to 3,429; already over the smell before). No new line over 700 chars in the diff (the two long lines in `falkor-chat/AGENTS.md` are untouched, pre-existing). The `kaizen/history.md` additions are short dated entries (not always-loaded files). No changelog prose added to context files. Pass. One caution: `SERVER.md` is a live spec; its new bullets embed "observed 2026-09-20 / 2026-10-15" dates, mild but acceptable as evidence bounds.

**(d)** Fails, see B1. The 7 claims match the title of their `.md` headings and the manifest ids resolve, but status is not `ready`.

**(e) `audit-team.sh` failures pre-existing: confirmed.** Without touching the tree: `git grep -l` for the maintainer's home path at `HEAD` gives 139 files / 673 lines; the diff adds 0 lines containing the user name; `git grep -il "$(hostname)" HEAD` gives 2 files (`docs/reviews/typed-decision-models4-impl.md`, `docs/test-reports/typed-decision-models-harness-probe-report.md`), matching the "2 files, 2 lines" hostname failure. Re-running the script now: only those two FAILs; prompt-size NOTEs for analyst, teco, tico, as reported. Delta from the sweep itself is m2.

**(f) REVIEW.md.** No new check was added by the sweep; the "no X remains" and "can it fail" bars do not apply, except that the K-061 acceptance ("confirm ready") was written as a note, not run, and fails when run (B1). No `REVIEW.md` lines proposed.

## What's solid
Most promotions are accurate and tightly bounded (the `BackgroundTasks`, `InvalidURL`, ORDER BY, log-lifetime and CPG claims all reproduce). Rewrites are in place with history cited rather than duplicated. Attribution loss (`u1`) and the remaining 98 entries are disclosed rather than hidden. Raw documents were not deleted, as instructed.

## Open questions
- Should the 6 failed claims be re-ingested by `cobb` once the embedder is loaded, or is the embedding backend shared with other sessions' fault injection (the LM Studio claim's own subject)? Recommend re-ingest after confirming `/v1/models` lists the embedder.

## Pass 2

**Verdict: approve with suggestions.** Rechecked against the live tree and graph (no deletions, no git mutations).

- **B1 fixed.** All seven claims are `ready` with every chunk embedded: `114ec156` (ORDER BY), `9d093333` (full-text), `d81f3752` (readiness), `dec4c284` (heredoc ask), `b18286cb` (fence), `b0d1d59f` (LM Studio), `50dc6b7e` (13.1 s). The manifest key `_cobb_distillation3_2026_11_06` lists six ids that all match graph ids and titles; the seventh, `b18286cb`, is in `_teco_coordination_techniques.documentIds` (39 ids, replacing `f6a73bf7`); the JSON parses and its `_note` now says six plus one. The six failed ids no longer exist. Residual (known, owed, not a defect of this pass): stale `ddfbb4bb` is still `ready` and served until its delete is approved (K-061).
- **M1 fixed.** The new `falkor-chat/docs/SERVER.md` bullet no longer mentions `rrfScore`/`lexicalRank` (`git grep` of the file: 0 hits) or the DDL claim; it keeps the process-start-time check. It stays consistent with `SERVER.md` line 695 (`start_server.sh` uses `--reload`; `start_agent_team.sh` does not).
- **M2 fixed.** `lm-studio-model-notes.md` and the re-ingested `b0d1d59f` carry identical text: the demo needs `ministral-3-3b` plus the embedder, and the qwen premise is dated as pre-K-066. Consistent with `falkor-chat/AGENTS.md` (demo set excludes qwen) and the rejection of row 36.
- **m1 stands as a note** (no change needed); the rejection reason is unchanged and still correct.
- **m2 fixed in the report, one leak remained: mine.** The sweep report has 0 home-path lines now; the only untracked leak left was a literal in this review, now genericized. `audit-team.sh` still prints 147 files / 744 lines and 2 hostname files; at `HEAD` the tracked baseline is 139 files, the diff adds 0 matching lines, so the two FAILs remain pre-existing. After this edit the count should fall by one (not re-run after the edit to avoid self-reference; re-run on commit).
- **n1** not changed (nit; "sub-30 ms"), take or leave.
- **Interaction between same-revision fixes:** SERVER.md, the LM note, the manifest note and the sweep report agree with one another (qwen outside the demo set, `:8200` without `--reload`, six plus one ids); no contradiction found.
- **Full-text probes re-run live** on `ws:agent-team` `Document` index: empty query returns 0 rows without error; `%a-b%` raises `RediSearch: Syntax error`; `%_%` raises a syntax error; `%__%` parses (3 rows); `%orderr%` parses (6). All match the promoted claim.
- **Rejections sampled.** Row 3 (module attr patch vs `from m import f`): reproduced (`1 2`), generic Python, rejection holds. Row 8 (`cmd | tail; echo $?`): `false | tail -1` gives rc 0, with `pipefail` rc 1; generic shell, holds. Row 17: the server venv does import `falkordb`, but `falkor-chat/AGENTS.md` does not state it (cobb's "visible from AGENTS.md" is slightly off; the rejection still holds as a trivial convenience). Row 10: the `GRAPH.DELETE` and `docker rm -f` `ask` patterns exist in `.claude/settings.json`; the compound-call denial itself was not reproducible by me, so that rejection is accepted but not independently verified. Row 30 (reloader parent survives): not re-run, since it would start servers; unverified by me.
