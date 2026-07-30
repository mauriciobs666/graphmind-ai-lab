# Kaizen — Learnings Inbox: qa-engineer

> Append-only capture of durable, non-obvious environment facts the `qa-engineer` agent
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

## 2026-07-21 — falkor-chat: the M3 workflow executor is wired only when `FALKORCHAT_ENABLE_AGENT` is truthy; `FALKORCHAT_WORKFLOW_ENABLED=1` alone gives a 503
- **Evidence:** `server/falkorchat/app.py` `_build_default_app()` returns `create_app(services)` early on `if not config.ENABLE_AGENT:` — the `if config.WORKFLOW_ENABLED:` block that builds the executor/trigger and calls `services.set_executor(...)` is **nested inside** the agent branch. Without an executor, `services._require_executor()` raises `WorkflowEngineDisabledError` → **503** on `POST /workflow-runs`. Confirmed by running the app with both flags on (process-flow REST worked) — the LLM-free `access-request@v1` flow needs no model but still needs the agent flag set.
- **Context:** K-025 M3 acceptance pass — planning an LLM-free black-box run of the `kind:'process'` proof flow.
- **Suggested home:** project docs (`falkor-chat/AGENTS.md` M1-server section / `server/.env.example`) — the two flags read as independent and are not.

## 2026-07-21 — falkor-chat: `pytest -m live` does NOT wipe the global `reference` graph, unlike the default offline `pytest`
- **Evidence:** AGENTS.md warns that a `server` pytest run leaves `reference` cleared by the `wf_repo` fixture. After `./scripts/test_queries.sh` + default `pytest`, `seed_workflows.sh acme` printed `(created)` for both `reference` defs — wiped, as documented. After `pytest -m live` (which deselects all 533 offline tests, so `wf_repo` never runs) `reference` still held all four defs, and the live test seeds its own throwaway `ws:live`. So the re-seed obligation attaches to the *default* run, not to a live-only run.
- **Context:** K-025 M3 acceptance pass — sequencing suite runs against the documented `reference`-wipe trap.
- **Suggested home:** project docs (`falkor-chat/AGENTS.md`, the `seed_workflows.sh` scripts-table row, which currently says "after a `server` pytest run" without the marker distinction).


## 2026-07-25 — `redis-cli GRAPH.QUERY … --no-raw` returns a FLAT one-scalar-per-line stream, and filtering blank lines silently corrupts null/empty cells
- **Evidence:** against FalkorDB v4.18.11, `redis-cli -p 6379 GRAPH.QUERY cpg_falkorchat '<3-col query>' --no-raw` produced exactly `3 header names + (21 rows × 3 cells) + 2 trailing lines` = 68 lines — **no** `1) 1)` RESP nesting, no per-row grouping. Parsing recipe that worked: drop the first N lines (N = column count), drop the last 2 (`Cached execution:` / `Query internal execution time:`), regroup the remainder into N-tuples. **The trap:** a `null` cell and an empty-string cell both render as an *empty line*, so a `[l for l in out if l != ""]` filter (the obvious first attempt) drops cells and silently shifts every subsequent row into the wrong column — my first harness reported false divergences for `null`/`""` until I stopped filtering. Also: `redis-cli` renders `null` **indistinguishably from `''`**, which is precisely why the `cpg` MCP tool renders `None` as the literal `null`.
- **Context:** S9 live acceptance of the `cpg` MCP server — AC-3 required diffing tool output against the `redis-cli` fallback path, which needs a parser for both formats.
- **Suggested home:** project docs (`skills/cpg-analysis/SKILL.md` §1 fallback block, or `cpg/mcp/README.md` next to its "tools diffing this output against redis-cli" note)

## 2026-07-25 — FalkorDB accepts Cypher boolean literals case-insensitively (`False` ≡ `false`), so Python-style boolean rendering round-trips fine
- **Evidence:** `MATCH (m:METHOD) WHERE m.NAME='post_message' AND m.IS_EXTERNAL = false RETURN count(m)` → `2`; the same query with `= False` → also `2`, no error. Worth knowing because `skills/cpg-analysis/SKILL.md` gotcha #2 (*"Booleans are real booleans: `= false` not `'false'`"*) reads as if capitalisation matters — it does not; **quoting** is what breaks (`'false'` is a string). I nearly filed a Medium "broken round-trip" defect against the MCP tool's `True`/`False` rendering before testing the claim; it downgraded to cosmetic.
- **Context:** S9 acceptance — characterising tool-vs-`redis-cli` cell-rendering divergences by value type.
- **Suggested home:** project docs (`skills/cpg-analysis/SKILL.md` gotcha #2 — clarify that the hazard is quoting, not case)

## 2026-07-30 — FalkorDB's own `RESULTSET_SIZE` (default 10000) silently caps the `cpg` MCP tool's reported `rows=` total, contradicting the "always the true total" claim
- **Evidence:** `redis-cli -p 6379 GRAPH.CONFIG GET RESULTSET_SIZE` → `10000`. `cpg_falkorchat` has **110048** nodes (`MATCH (n) RETURN count(n)`), but `mcp__cpg__query(cpg_falkorchat, "MATCH (n) RETURN n")` reports `rows=10000` — not the true 110048 — with no indication this is itself a cap rather than an exact count. Confirmed the FalkorDB-level cap (not the tool's own `CPG_MCP_MAX_ROWS`) is what binds: raw `redis-cli GRAPH.RO_QUERY cpg_falkorchat 'MATCH (n) RETURN n.id LIMIT 50000'` also returns only ~10,000 rows despite the explicit `LIMIT 50000`. Below 10,000 true rows the tool's own accounting is exact (verified: `MATCH (m:METHOD) RETURN m.NAME` on 1968 methods reported `rows=1968` correctly while rendering only 200). So the tool's row-cap/char-cap machinery is honest; it's blind to a *second*, server-level cap sitting beneath it that neither `cpg/mcp/README.md` nor `docs/manuals/cpg-getting-started.md` mentions.
- **Context:** QA verification pass on `docs/manuals/cpg-getting-started.md` (TP-008) — testing the manual's "the underlying count is always the true one" claim about truncated results.
- **Suggested home:** project docs (`cpg/mcp/README.md`'s "Truncation is display-only" section, which currently states the `rows=` figure is always true without qualification — and `skills/cpg-analysis/SKILL.md` §2, as a 6th gotcha, since it silently returns a wrong-but-plausible number rather than an empty/null result).

## 2026-07-25 — In a `pysrc2cpg` CPG, `METHOD.CODE` holds only short signatures; the wide text lives on `LITERAL`/`BLOCK`/`CALL`
- **Evidence:** `MATCH (n) WHERE n.CODE IS NOT NULL RETURN labels(n)[1], max(size(n.CODE)), count(n) ORDER BY 2 DESC` on `cpg_falkorchat` → `LITERAL 4314 (12,223 nodes)` · `BLOCK 2715` · `CALL 2552` · `RETURN 958` · `METHOD` not in the top 8. Consequence for payload-size testing: the intuitive "wide projection" `MATCH (m:METHOD) RETURN m.CODE` yields only **1,951 chars** across all 1,968 methods, so a row cap binds long before any char/size cap — a test written that way exercises the wrong branch and passes vacuously. A genuine binder is `MATCH (n:LITERAL) WHERE size(n.CODE) > 400 RETURN size(n.CODE), n.CODE` (29,890 chars). Docstrings are `LITERAL` nodes, which is why they dominate.
- **Context:** S9 acceptance — plan §7.3 named the METHOD/CODE query as the char-cap probe; it does not bind the char cap, so the case was re-derived (filed as a test-design defect).
- **Suggested home:** knowledge base (`skills/joern-cpg/references/cpg-model.md` consumer-query facts — where `CODE` is actually wide)
