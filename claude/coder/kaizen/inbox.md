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

## 2026-08-10 — A pytest autouse fixture's `monkeypatch.setenv(...)` is a no-op for a module-level constant already computed from `os.environ.get(...)` at import time
- **Evidence:** `falkor-chat/server/falkorchat/config.py` resolves `OPENCODE_CONFIG_PATH`/`MODEL_CONFIG_PATH` (and every other config constant — `WS_ID`, `ENABLE_AGENT`, etc.) once at module import, per the codebase's own stated design ("read once, no reload path"). `falkorchat.config` is already imported (by other test modules, at collection time) before any per-test autouse fixture runs, so `monkeypatch.setenv("FALKORCHAT_OPENCODE_CONFIG", ...)` inside the fixture never reaches code that reads `config.OPENCODE_CONFIG_PATH` — the attribute was frozen long before. The existing codebase convention for this (seen in `test_app.py`, e.g. `monkeypatch.setattr(app_mod.config, "ENABLE_AGENT", True)`) is to `monkeypatch.setattr` the **module attribute** directly, not the env var. Fix: the autouse fixture must set both — the env var (for anything reading `os.environ` fresh, or a subprocess) **and** `monkeypatch.setattr(config, "ATTR", value)` (for the frozen-at-import constant).
- **Context:** K-042 Landing 1 (`falkor-chat` LLM provider config) — a `conftest.py` autouse fixture pointing `FALKORCHAT_OPENCODE_CONFIG`/`FALKORCHAT_MODEL_CONFIG` at offline test fixtures for the whole suite.
- **Suggested home:** knowledge base (a general pytest/config-module gotcha, not falkor-chat-specific) — maybe a `python-web-quirks`-adjacent skill, or this repo's own testing-hazards doc pattern (`docs/DESIGN.md` §14.7-style sections exist per component).

## 2026-08-10 — A function-LOCAL `from .module import name` re-resolves the name fresh on every call, so `monkeypatch.setattr("pkg.module.name", fake)` intercepts it — even though a function's DEFAULT ARGUMENT bound to that same name at def-time does NOT
- **Evidence:** `falkor-chat/server/falkorchat/modelconfig.py`'s `_build_llm`/`_build_embedder` do `from .transport import make_http_transport` as the first line inside the function body (not a top-level import — deliberately, to break a `modelconfig.py`⇄`llm.py`/`embedding.py` circular import). Because it's a statement executed at call time, it performs a fresh `getattr(transport_module, "make_http_transport")` every call, so `monkeypatch.setattr("falkorchat.transport.make_http_transport", fake_factory)` transparently intercepts every future `.llm()`/`.embedder()` resolution — useful for asserting the URL/model/timeout a resolved client would actually send, with zero real network and no changes to `ModelGateway`'s public API. Contrast: `transport.py`'s own `make_http_transport(..., opener=urllib.request.urlopen)` binds that default *at function-definition time* (Python evaluates defaults once), so monkeypatching `urllib.request.urlopen` afterward does NOT reach it — that seam has to be injected explicitly via the `opener=` kwarg instead.
- **Context:** K-042 Landing 1 — testing that two different `ModelGateway.resolve()` calls (different `requested=` refs, or different kinds with different per-kind defaults) actually produce different transport calls (URL/model/timeout), without a real HTTP layer.
- **Suggested home:** knowledge base (general Python import-binding-timing gotcha, reusable across any codebase doing deferred/local imports to break a cycle) — candidate for `skills/python-web-quirks` even though it isn't web/async-specific, since it's exactly the kind of "looks obvious once you see it, bites you the first time" fact that skill collects.
