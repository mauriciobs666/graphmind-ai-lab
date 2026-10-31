# Embedder context and MCP message cap — coordination

> **Status:** active · **Owner:** `teco` · **Tracks:** K-099

Scope and the stakeholder decision (embedder at 4096, chat model stays 8192) live in
`falkor-chat/docs/BACKLOG.md` K-099; this ledger only tracks units.

## Units

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| U1 — probe LM Studio over-context embed + load embedder at 4096 (`scripts/ensure_demo_models.{py,sh}`) | `devops` | `a6e53d08434c67adf` | in-flight (P2-1..5 fixes) | `docs/test-reports/embedder-context-report.md` + `scripts/ensure_demo_models.{py,sh}`, `server/tests/test_ensure_demo_models_script.py`, `AGENTS.md` | `analyst` → — (`a3bd902c46e6e0aaa`) → approve w/ suggestions, Pass 2 (P2-1..3 minor, P2-4/5 nit) | 130k tok / 42 tools |
| U2 — enforce `MAX_TEXT_LEN` on MCP `send_message` | `tdd-engineer` | `a50e7e24fe5ba9303` | accepted (Pass 3 approve; P3-1 nit → `tdd-engineer`/haiku `a65117d1d30d80c74`) | `server/falkorchat/{services,mcp,storefront_api}.py` + tests | `analyst` (`a3bd902c46e6e0aaa`) → approve w/ suggestions, `docs/reviews/embedder-context.md` Pass 1 (F1 minor, F2 nit) | 134k tok / 44 tools |
| U3 — re-measure VRAM/paging vs `docs/test-reports/salesperson-single-model-fit-report.md` | `devops` | — | queued (after U1) | `docs/test-reports/embedder-context-report.md` | — (measurement) | — |
| U4 — feature close: SERVER.md MCP surface update, impl check, HISTORY | implementer / `analyst` | — | queued | `docs/reviews/embedder-context-impl.md` | `analyst` → — | — |

## Documentation impact

- `falkor-chat/AGENTS.md` — `ensure_demo_models.sh` row (says "ministral's context isn't 8192"; loads at 8192). U1.
- `scripts/ensure_demo_models.{py,sh}` header comments. U1.
- `falkor-chat/docs/SERVER.md` MCP `send_message` contract — live spec, feature close only (U4).
- `falkor-chat/docs/HISTORY.md` entry; K-099 removed from `BACKLOG.md` at close.

## Notes

- A concurrent session has untracked `falkor-chat/server/tests/test_decision_contract.py`; U2's pytest
  shares `ws:test`/`reference` with it — solo re-run before integrating.
- CPG `cpg_falkorchat` built 2026-10-31T17:34Z at `c0eeb348`, no package changes since — fresh.
- Open question for stakeholder: `post_agent_answer` (LLM replies: responder, workflow `post_message` tool) stays uncapped — a long model reply still reaches the embedder. Not in K-099 scope as written.
- U1 probe: `laya` (model-bench's model) vanished from LM Studio right after U1's probe load (`unloadPreviousJITModelOnLoad: true`); cause unconfirmed, not reloaded — relayed to stakeholder 2026-10-31.
- 2026-10-31: teco's U2 gate brief wrongly called `tests/test_storefront_api.py` review-safe (its `seeded` fixture requests `wf_repo`); analyst's and teco's runs emptied `reference`. Re-seeded by `devops` (`a30120a7c9cc41974`, 52k tok / 12 tools): 11/11 exit 0; teco re-verified read-only — `reference` = 15 Product, 4 WorkflowDef, 15 Step; verify_workflows/salesperson/catalog OK. Prior content unknown, so anything beyond the four seeded defs is not restored.
