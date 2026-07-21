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

