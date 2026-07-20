# Kaizen — Learnings Inbox: analyst

> Append-only capture of durable, non-obvious environment facts the `analyst` agent
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

## 2026-07-19 — `falkor-chat/server` pytest silently self-skips ~half the suite when FalkorDB is down, so "N passed" alone is not evidence
- **Evidence:** `cd falkor-chat/server && .venv/bin/python -m pytest -q` with no FalkorDB container running → `171 passed, 177 skipped, 1 deselected` (skip reason: "FalkorDB not reachable — start it with ./scripts/start_falkordb.sh"); with the DB up the same command is reported as `348 passed, 1 deselected`. `171+177+1 = 349` collected either way, so the *collected* count is stable while the *passed* count halves. `docker ps` is the one-command precheck.
- **Context:** analyst implementation gate on `aa8b813` (K-022 Landing 2) — a claimed-green suite had to be reported as partially unverified, and the drive-level Defect-B regression pin (`tests/test_executor.py`, `wf_repo` fixture) was among the skipped.
- **Suggested home:** project docs (`falkor-chat/AGENTS.md` M1-server section, next to the existing "needs FalkorDB up" note — make the skip-vs-fail behavior and the expected pass/skip split explicit)

## 2026-07-19 — Byte-identity of a "locked" function is cheap to verify mechanically with an AST line-range hash, and separates docstring-only edits from real ones
- **Evidence:** `ast.parse` → walk for the `FunctionDef` by name → hash `src.splitlines()[node.lineno-1:node.end_lineno]` across `git show <rev>:<path>` extracts. On this run it proved `executor._drive_loop` identical (`71055f756280`, 2844 bytes) across `514346b`/`c3cc239`/`aa8b813` despite its line offset moving 310→324, which a `git diff` line-range check would have mis-read. A second pass comparing `ast.dump` with docstring `Expr` nodes stripped showed `_drive` was docstring-only.
- **Context:** the gate's mandatory "the §2.1 A/B/C drive loop must be byte-for-byte unchanged" confirmation.
- **Suggested home:** prompt (analyst — a standard technique for "this artifact is locked, prove it didn't change" confirmations)
