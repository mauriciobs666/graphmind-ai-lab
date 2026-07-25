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
