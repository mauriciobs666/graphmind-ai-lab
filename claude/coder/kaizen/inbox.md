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

## 2026-07-24 — FastAPI: to OMIT an optional response key (not emit `null`), use `response_model_exclude_unset=True`, not `exclude_none=True`
- **Evidence:** `falkor-chat/server/falkorchat/api.py` `get_workflow_def_structure` / `get_snapshot_structure`. A field declared `startKeys: list[str] | None = None` serializes as `"startKeys": null` — *present* — under the default settings. `response_model_exclude_none=True` omits it, but it also omits **every** other `None`-valued field, including the sibling `startKey`, which is legitimately nullable. `response_model_exclude_unset=True` keys off `model_fields_set` instead: when the service returns a dict that simply lacks the key, the field is unset ⇒ omitted; when the dict carries `startKey: None`, that field *is* set ⇒ still emitted as `null`. So "omit when absent, keep an explicit null" is expressible, and it is the shape an observability endpoint wants (hiding a null is hiding the anomaly).
- **Context:** K-031 def/snapshot structure read surface — the response contract required `startKeys` present only for a multi-`START` root, pinned by an exact-key-set contract test.
- **Suggested home:** knowledge base

## 2026-07-24 — In this repo, a `pytest --collect-only` baseline can move under you mid-run when another agent is working the same tree
- **Evidence:** measured **552/553 collected, 1 deselected** at K-031 start; final run **595 passed, 1 deselected**. K-031's own contribution is **+32** (verified by `git diff … | grep -cE '^\+def test_'` per file plus the parametrize expansion). The remaining **+11** were `tests/test_llm.py`/`tests/test_app.py` additions from a concurrent K-027 gate pass that landed after the measurement — confirmed by that unit's own HISTORY.md entry ("entry baseline 533; 552 after the first draft's 19, 563 after this gate pass's 11"). Source-file mtimes (`ls -l --time-style=full-iso`) were what made the concurrency visible.
- **Context:** two units running in parallel on one working tree under a coordinator; the delivery report had to state entry→exit numbers.
- **Suggested home:** prompt (report the *attributed* delta, not just entry→exit, whenever a run shares a tree)

## 2026-07-24 — A green pytest exit code is not evidence an integration suite actually ran — read the skip count
- **Evidence:** `falkor-chat/server/tests/conftest.py:54` (`_falkordb_reachable()`) turns the entire graph-backed half of the suite into `pytest.skip(...)` when the DB is unreachable, rather than failing. `pytest -q` still exits 0 with roughly half the tests silently skipped in that case — the exit code alone cannot distinguish "everything passed" from "half the suite never ran."
- **Context:** falkor-chat AGENTS.md doc-restructure audit — found the project's own docs had to warn "always report/read `N passed, M skipped`, never just the absence of failures" because this is easy to miss when reporting a run as green.
- **Suggested home:** prompt (the "Verify and report" step should say to check skip counts, not just exit status, whenever a suite has an environment-reachability gate)

## 2026-07-25 — A FastMCP stdio server drops the response to the LAST request when the client closes stdin immediately — a smoke-test artifact, not a server bug
- **Evidence:** driving `cpg/mcp/run.sh` with `subprocess.run(input=<newline-delimited JSON-RPC>)` on `mcp 1.28.1`: stderr logged `Processing request of type CallToolRequest` for every call, but the final request's response never appeared on stdout (reproduced twice — first with `tools/call` last, then with a different `tools/call` last). Adding a trailing throwaway message (`{"method":"ping"}`) made the previously-lost response appear. EOF on stdin tears the anyio session down before the last write flushes. Handshake required for any of it to work: `initialize` → `notifications/initialized` → real calls.
- **Context:** S2 of the CPG query-access plan — the step's done-condition is a manual `initialize` + `tools/list` + `tools/call` stdio round trip, which silently "loses" the very result it is meant to prove.
- **Suggested home:** knowledge base (MCP/agent-standards: how to smoke-test a stdio MCP server from a script)

## 2026-07-25 — FalkorDB has no string-repetition operator: `CREATE (:T {code: 'x' * 400})` fails with a type error, so bulk test fixtures must pass long strings as parameters
- **Evidence:** `UNWIND range(1,500) AS i CREATE (:Big {i:i, code:'x' * 400})` → `redis.exceptions.ResponseError: Type mismatch: expected Integer, Float, or Null but was String` (FalkorDB v4.18.11 via falkordb-py 1.6.2). `... {code: $c}` with `params={"c": "x"*400}` works. Second-order gotcha: the failed `GRAPH.QUERY` still **materialised the graph key**, leaving a junk graph behind that had to be deleted by hand — the same empty-key quirk that makes `GRAPH.RO_QUERY` the safe read path.
- **Context:** building a wide result set to demonstrate the MCP query tool's char-cap truncation against a live FalkorDB.
- **Suggested home:** knowledge base (`claude/graph-dba/falkordb-quirks.md`)

## 2026-07-27 — Counting `audit-team.sh` failures with `grep -c FAIL` overcounts by one: the trailing `RESULT: FAIL` summary line matches too
- **Evidence:** `bash claude/scripts/audit-team.sh | grep -c FAIL` → **3**; `| grep -c '^FAIL'` → **2**. The script emits one `FAIL  <check>` line per failing check plus a final `RESULT: FAIL — fix the items above, then re-run.` summary. The repo currently sits at exactly 2 real FAIL lines (both check 7 personal-info leaks, C-309a), so a done-condition phrased "still exactly 2 FAIL lines" reads as a regression under the naive grep. Anchoring with `^FAIL` is the correct count; comparing the count against a `git stash`ed baseline is the correct *regression* test, since the absolute number is nonzero by default.
- **Context:** doc-convention step 3 (25-document header backfill) whose done-condition was "audit still shows exactly 2 FAIL lines, a third is a regression you introduced".
- **Suggested home:** knowledge base (`skills/agent-maintenance` — how to read the audit script's output), or the audit script itself (emit a machine-countable summary)
