# Kaizen — Learnings Inbox: coder

> Append-only capture of durable, non-obvious environment facts the `coder` agent
> discovers during runs — raw observations, not conclusions. The maintainer (cobb)
> periodically distills this inbox (agent-maintenance skill §5): verifies each entry,
> routes it (prompt / knowledge base / project docs / discard), logs the promotion in
> `history.md`, and clears it. The agent only appends here; it never promotes.
>
> Entry format (append at the end):
>
> ```markdown
> ## YYYY-MM-DD — <the fact, one line>
> - **Evidence:** what was run/read/observed (command, file:line, output)
> - **Context:** the task where it surfaced, one line
> - **Suggested home:** prompt | knowledge base | project docs | unsure
> ```

## 2026-07-16 — A published falkor-chat workflow def is immutable per version; editing a seed prompt and re-seeding silently keeps the OLD prompt
- **Evidence:** `repository.py:916` `_PUBLISH_CYPHER` uses `MERGE (st:Step …) ON CREATE SET st.config = s.config` — an existing Step's `config` is never updated on re-publish/re-materialize. After editing `scripts/seed_workflows.sh` prompts and re-running `seed_workflows.sh acme`, the reference def refreshed (`created`) but the `ws:acme` snapshot reported `already present — no-op` and still returned the OLD `s.config` (verified with `GRAPH.RO_QUERY ws:acme "MATCH (s:Step {key:'answer'}) RETURN s.config"`). Two independent tiers go stale: the global `reference` def AND each workspace's `WorkflowDefSnapshot`.
- **Compounding trap:** `./scripts/test_queries.sh` drops `reference` but NOT `ws:acme`, so after the suite a re-seed refreshes `reference` from the new source while `ws:acme` keeps its stale snapshot — a split-brain where the def and its materialized snapshot disagree.
- **Operator recovery (no version bump):** `GRAPH.QUERY ws:<id> "MATCH (d:WorkflowDefSnapshot {key:'triage',version:'v1'}) DETACH DELETE d"` + `MATCH (s:Step) WHERE s.stepUid STARTS WITH 'triage:v1:' DETACH DELETE s`, then re-run `seed_workflows.sh <id>` (re-materializes fresh). For `reference` the same, deleting `WorkflowDef` instead. The live e2e (`test_workflow_live.py`) is immune only because its fixture `.delete()`s the whole `ws:live` each run — a measurement harness editing prompts MUST drop the `reference` def between runs or it measures the stale def and looks like a null result.
- **Context:** K-022 M3 D8/S5 + Defect C — measuring a prompt change on the live triage flow; a stale def would have made the whole measurement meaningless.
- **Suggested home:** project docs (falkor-chat AGENTS.md `seed_workflows.sh` row / DESIGN §11 materialize note)

## 2026-07-19 — falkor-chat's "network-free" pytest baseline WIPES the global `reference` graph when FalkorDB is up (same hazard as `test_queries.sh`)
- **Evidence:** `server/tests/conftest.py:93` — the `wf_repo` fixture runs `db.reference_graph(conn).query("MATCH (n) DETACH DELETE n")` on **every** test that requests it (used by `test_repository.py`, `test_api.py`, `test_executor.py`, `test_executor_produced.py`). `_schema` (conftest.py:57) additionally `.delete()`s the whole `ws:test` graph once per session. So a plain `.venv/bin/python -m pytest -q` with FalkorDB running destroys any published `WorkflowDef` in `reference` (e.g. `triage@v1`), while leaving each `ws:<id>` snapshot intact — the exact def/snapshot split-brain that `AGENTS.md` currently attributes only to `test_queries.sh`.
- **Corollary:** with FalkorDB **down** the same command reports `171 passed, 177 skipped, 1 deselected` instead of `348 passed, 1 deselected` — 177 tests skip on a reachability guard rather than fail, so a "green" pytest line is NOT evidence the graph-backed half ran. Always read the skip count, not just the absence of failures.
- **Context:** K-022 M3 D14 S5 revert — instructed to run the pytest baseline but explicitly NOT `test_queries.sh` because it drops `reference`; pytest turns out to carry the same destructive effect.
- **Suggested home:** project docs (falkor-chat AGENTS.md — the M1 server pytest bullet should carry the same ⚠️ as the `test_queries.sh` row)

## 2026-07-19 — The `_drive_loop` byte-identity lock reproduces only with an AST **line-range** slice (`def`..`end_lineno`); the byte count quoted in the gate review is wrong
- **Evidence:** `falkor-chat/docs/plans/m3-executor.md` §2.1 and every gate review quote `_drive_loop` SHA-256(12) = `71055f756280`, **2844 bytes**. Reproducing it: parse `server/falkorchat/executor.py`, find the `FunctionDef`, and hash `"".join(lines[n.lineno-1 : n.end_lineno])` → `71055f756280` but **2860 bytes**. Using `ast.get_source_segment` instead (dedents/trims) gives `f39c370c5556`/2855, and slicing from `n.body[0].lineno` gives `9dab84ba35ff`/2787. Only the raw `lineno..end_lineno` line-range slice matches the locked hash.
- **Why it matters:** the hash is the check that actually holds; the byte count in the docs does not, so an implementer who verifies "2844 bytes" concludes the lock is broken when it isn't. Verify by hash, and additionally diff the function against `git show HEAD:` — cheaper and unambiguous.
- **Context:** K-022 Landing 2 M-2 fix — the brief hard-required proving `_drive_loop` was untouched.
- **Suggested home:** project docs (falkor-chat `docs/plans/m3-executor.md` §2.1 — drop the byte count or correct it to 2860 and state the extraction recipe)

## 2026-07-20 — falkor-chat: publishing/materializing a workflow def with ZERO transitions raises `IndexError`, not a clean error (empty-UNWIND row collapse)
- **Evidence:** `server/falkorchat/repository.py:916` `_PUBLISH_CYPHER` ends `… WITH d, stepCount UNWIND $transitions AS tr … WITH d, stepCount, count(rel) AS transitionCount RETURN …`. With `$transitions = []` the `UNWIND` collapses the row stream **before** the `RETURN`, so `res.result_set` is empty and both `publish_def` (`repository.py:998`) and `materialize_snapshot` (`:1397`) blow up on `row = res.result_set[0]` → `IndexError: list index out of range`. Reproduced while writing `tests/test_executor_process.py`: a single-step terminal def (steps=[…], transitions=[]) failed this way; adding one transition fixed it. The steps + `START` written earlier in the same query DO land, so the def is left half-written with a stack trace and no named error.
- **Why it was invisible:** every existing publish test with the *real* repository carries ≥1 transition, and the zero-transition service tests use `FakeRepo` or raise in `_validate_def_spec` first, so no test ever reaches the real query with an empty list. This is the exact `UNWIND []` class `falkor-chat/AGENTS.md` already documents for the §4 mention write-block (which guards it with `CASE WHEN $x = [] THEN [null] ELSE $x END`); the workflow publish path has no such guard.
- **Context:** K-024 U2 (typed step handlers) — needed a def whose only step is terminal; had to reshape every fixture to carry a transition. Not fixed (out of unit scope); reported to teco.
- **Suggested home:** project docs (falkor-chat `docs/QUERIES.md` §11.1 + a `_PUBLISH_CYPHER` guard, same shape as the §4 mention block)

## 2026-07-21 — falkor-chat pytest wipes `reference` at test *setup*, so the LAST test's published defs survive the run and can masquerade as a seeded def
- **Evidence:** `server/tests/conftest.py:93` (`wf_repo`) does the `MATCH (n) DETACH DELETE n` on the `reference` graph as **setup**, never teardown. Observed directly: after `.venv/bin/python -m pytest -q` (523 passed) the *first* `./scripts/seed_workflows.sh acme` printed `reference def access-request@v1 … (already present — no-op)` while its snapshot line printed `materialized` — the def in `reference` was the leftover from `tests/test_process_flow.py`'s fixture, not something the seed created. A second pytest run + re-seed then printed `(created)` for both defs, confirming the mechanism (the leftover only exists when the *last* `reference`-writing test published that key).
- **Why it matters:** the seed script's `already present — no-op` line is the operator's only signal that a def is unchanged, and published defs are `MERGE … ON CREATE SET` (create-only). A test-fixture leftover therefore silently *wins* over the seed source — the exact def/snapshot split-brain hazard, entered from the direction the docs do not describe (they warn that pytest **wipes** `reference`, not that it **leaves data** there).
- **Practical rule:** treat "pytest → seed" as producing an untrustworthy `already present` verdict for any def key a test publishes; re-run pytest once more, or drop the def subgraph, before reading the seed output as evidence.
- **Context:** K-024 U4 (falkor-chat) — verifying that `seed_workflows.sh` seeds both `triage@v1` and the new `access-request@v1`.
- **Suggested home:** project docs (falkor-chat AGENTS.md — the pytest bullet / `seed_workflows.sh` row)

## 2026-07-21 — falkor-chat `server`'s ruff baseline is already RED (one pre-existing I001 in `llm.py`), so `ruff check .` is not a usable pass/fail gate
- **Evidence:** `cd falkor-chat/server && .venv/bin/ruff check .` on a clean tree reports exactly one error — `I001 Import block is un-sorted or un-formatted --> falkorchat/llm.py:13:1` (`import urllib.request` sits after `from dataclasses import …`). It is pre-existing: `llm.py` is untouched by K-024 U0–U4b. `pyproject.toml` configures ruff (`[tool.ruff.lint]`, dev dep `ruff>=0.14,<0.15`) but nothing runs it in a script or hook.
- **Why it matters:** an implementer who runs ruff as a post-change check reads that error as their own regression and "fixes" an unrelated file, widening the diff. Read the file path before acting; the real gate in this component is `pytest` (+ `scripts/test_queries.sh`, coordinator-run).
- **Context:** K-024 U4b (falkor-chat) — ran ruff as a self-check after editing `services.py`/`executor.py`.
- **Suggested home:** project docs (falkor-chat AGENTS.md, M1 server section — either state that ruff's baseline is one known error in `llm.py` or land the one-line import fix so the gate is green)
