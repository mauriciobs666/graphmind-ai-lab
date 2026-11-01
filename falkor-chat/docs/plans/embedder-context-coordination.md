# Embedder context and MCP message cap — coordination

> **Status:** archived · **Owner:** `teco` · **Tracks:** K-099, K-101

Scope and the stakeholder decision (embedder at 2048 — reversed from 4096 the same day — chat model stays 8192) live in
`falkor-chat/docs/BACKLOG.md` K-099; this ledger only tracks units.

## Units

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| U1 — probe LM Studio over-context embed + load embedder at 4096 (`scripts/ensure_demo_models.{py,sh}`) | `devops` | `a6e53d08434c67adf` | accepted (Pass 4 approve) | `docs/test-reports/embedder-context-report.md` + `scripts/ensure_demo_models.{py,sh}`, `server/tests/test_ensure_demo_models_script.py`, `AGENTS.md` | `analyst` → — (`a3bd902c46e6e0aaa`) → approve w/ suggestions, Pass 2 (P2-1..3 minor, P2-4/5 nit) | 157k tok / 60 tools (cumulative) |
| U2 — enforce `MAX_TEXT_LEN` on MCP `send_message` | `tdd-engineer` | `a50e7e24fe5ba9303` | accepted (Pass 3 approve; P3-1 nit → `tdd-engineer`/haiku `a65117d1d30d80c74`) | `server/falkorchat/{services,mcp,storefront_api}.py` + tests | `analyst` (`a3bd902c46e6e0aaa`) → approve w/ suggestions, `docs/reviews/embedder-context.md` Pass 1 (F1 minor, F2 nit) | 134k tok / 44 tools |
| U3 — re-measure VRAM/paging vs `docs/test-reports/salesperson-single-model-fit-report.md` | `devops` | `a6e53d08434c67adf` (resumed) | accepted — same-day A/B: embedder −753 MiB dedicated, ministral paging −738 MiB; teco cross-checked headline figures against raw phase logs + fit report | `docs/test-reports/embedder-context-report.md` | — (measurement) | — |
| U5 — RCA: REST bounds not enforced on MCP/service paths (K-101) | `analyst` | `ae01eb536bc013c11` | accepted | `docs/reviews/mcp-input-bounds-rca.md` | — (RCA; teco spot-checked root-cause commit, SERVER.md:1074 claim, 7 over-200 titles) | 182k tok / 60 tools |
| U6a — K-101 parity test + service checks (incl. whitespace reject, K-074) | `tdd-engineer` | `a5742b9891ed4ef62` | accepted (Pass 2 approve w/ suggestions; P2-1/P2-2 carried into U6b) | `server/falkorchat/{services,storefront_api}.py` + tests incl. new `test_input_bound_parity.py` | `analyst` (`acca5992382a000e3`) → approve w/ suggestions, `docs/reviews/mcp-input-bounds.md` Pass 1 (P1-1 major) → Pass 2 approve w/ suggestions (P2-1 minor: `propertyNames` unwalked; P2-2 nit: twin prefix match); full suite 3574 passed (teco, solo) | 281k tok / 115 tools (cumulative — U6b goes to a fresh dispatch) |
| U6b — K-101 MCP `Annotated` tool-signature bounds + review P2-1 (closed-world collector) / P2-2 | `tdd-engineer` | `ac40cbad3a000d415` | accepted (Pass 3 approve w/ suggestions; P3-1/P3-2 nits fixed + mutation-verified, re-gate skipped as nit-level; full suite 3634 passed, teco solo) | code + tests | `analyst` → — | — |
| U3b — same-day 2048 measurement + cutoff (stakeholder ask 2026-10-31; decision input, no script change) | `devops` | `a6e53d08434c67adf` (resumed) | accepted — 2048 vs 4096: embedder −352 MiB, paging −35%; cut at 2048: prose 9,681 / code 7,960 / URL-heavy 5,733 chars; teco cross-checked against raw logs | `docs/test-reports/embedder-context-report.md` | — (measurement) | 209k tok / 84 tools (cumulative) |
| U1b — switch embedder to 2048 (stakeholder decision 2026-10-31, reverses 4096) | `devops` | `a6e53d08434c67adf` (resumed) | accepted (Pass 5 approve w/ suggestions; P5-1/P5-2 text fixes applied by teco) | script + tests + `AGENTS.md` + report headline; live embedder at 2048 (loaded by hand — script refused, qwen resident; ~21 s embedder gap 22:04:18–39) | `analyst` → — | 231k tok / 107 tools (cumulative — next devops work goes fresh) |
| U7 — capture-template title ≤ 200 chars (stakeholder yes 2026-10-31; follows K-101) | `cobb` | `a6f06e9efe3cde0b6` | accepted — `a5cc041c`, manual `c6e33c9d`, follow-ups `0821d09d`, F5 `bf6039d4` (review Pass 3 approve) | 13 prompts + 13 kaizen histories + `claude/AGENTS.md` + `skills/agent-maintenance/SKILL.md` (three-way check: `claude/`,`skills/` clean at session start, no commits there since) | `analyst` → — | 70k tok / 14 tools |
| U4 — feature close: SERVER.md update (step 2) | `tdd-engineer` | `a580b55ad5d6b2d9f` | accepted — spec `3a9564f1` (close checks Pass 3 approve), REVIEW.md `62b75cc0`, HISTORY entries + backlog removal + archive in the close commit | `docs/reviews/embedder-context-impl.md` | `analyst` → — | — |

## Documentation impact

- `falkor-chat/AGENTS.md` — `ensure_demo_models.sh` row (says "ministral's context isn't 8192"; loads at 8192). U1.
- `scripts/ensure_demo_models.{py,sh}` header comments. U1.
- `falkor-chat/docs/SERVER.md` §1.4 "Bounds" and §2.2 "No schema layer on MCP" — live spec, feature close only (U4); K-101 lands in the same sections, so it closes with K-099.
- `falkor-chat/docs/HISTORY.md` entry; K-099 removed from `BACKLOG.md` at close.

## Notes

- A concurrent session has untracked `falkor-chat/server/tests/test_decision_contract.py`; U2's pytest
  shares `ws:test`/`reference` with it — solo re-run before integrating.
- CPG `cpg_falkorchat` built 2026-10-31T17:34Z at `c0eeb348`, no package changes since — fresh.
- Open question for stakeholder: `post_agent_answer` (LLM replies: responder, workflow `post_message` tool) stays uncapped — a long model reply still reaches the embedder. Not in K-099 scope as written.
- U1 probe: `laya` (model-bench's model) vanished from LM Studio right after U1's probe load (`unloadPreviousJITModelOnLoad: true`); cause unconfirmed, not reloaded — relayed to stakeholder 2026-10-31.
- 2026-10-31: teco's U2 gate brief wrongly called `tests/test_storefront_api.py` review-safe (its `seeded` fixture requests `wf_repo`); analyst's and teco's runs emptied `reference`. Re-seeded by `devops` (`a30120a7c9cc41974`, 52k tok / 12 tools): 11/11 exit 0; teco re-verified read-only — `reference` = 15 Product, 4 WorkflowDef, 15 Step; verify_workflows/salesperson/catalog OK. Prior content unknown, so anything beyond the four seeded defs is not restored.
- 2026-10-31 stakeholder: K-101 fixes the full class and absorbs K-074; whitespace-only message text rejected on every front door (REST 201 → 400). K-100 (chunk long agent replies) filed, not in this coordination.
- Feature close (U4) must apply `docs/reviews/mcp-input-bounds.md` Pass 1's proposed root `REVIEW.md` addition (schema-walking guards tested per composition axis) verbatim, plus SERVER.md items it lists beyond RCA §6 (§1.4 REST row for POST messages, §1.4 Bounds non-blank text + the pre-existing false `{mid}` cap claim, §2.2 malformed-item bullet).
- 2026-10-31: U6a committed `16f76f46` after teco's solo full suite (3574 passed); `reference` re-seeded by `devops` (`ad9904cbed4a9a6fe`), 11/11 exit 0.
- 2026-10-31 stakeholder: embedder at **2048** (reverses 4096); K-100 widened to chunk every message over the window. `MessageTooLongError` docstring in `services.py` corrected by teco (trivial, comment-only), committed with U1b.
- Follow-up for report: `ws:agent-team` holds 11 `failed` documents with unembedded chunks (oldest 2026-10-24, latest 2026-10-31 21:38 — before U1b's embedder gap, so not caused by it); pre-existing, out of scope.
- 2026-10-31: U6b committed `7490f634`; reseeded by `devops` (`a8938d19f23bce720`), 11/11 exit 0; teco re-verified `reference` (15 Product, 5 WorkflowDef, 19 Step). Close steps 3–5 pending.
- 2026-10-31 stakeholder: yes to K-102 (filed, commit after this line) and to U7.
- 2026-10-31: feature close complete for K-099 and K-101; K-100 and K-102 remain open in `BACKLOG.md`.
