# Coordination — bare-call `name`/`action`/`tool` argument-key shadowing (K-035)

> **Status:** archived · **Owner:** `teco` · **Tracks:** K-035 (post-M3 follow-up, delivered 2026-09-01, not a milestone gate)

## Goal

`falkor-chat/docs/BACKLOG.md` K-035: in `server/falkorchat/llm._parse_content_tool_calls`, the
JSON probe runs before the bare-call probe, and `_normalize_tool_call` maps `name`/`action`/`tool`
loosely. A bare call whose **argument object** happens to carry one of those keys is mistaken for
the call envelope itself, and the real call name is lost — e.g.
`create_user({"name": "bob"})` parses as `ToolCall(name='bob', arguments={})`. Not currently
reachable (no registered tool takes such a parameter today), but the failure mode is silent and
manufactures a tool call named after a user-supplied value — `executor._handle_tool_call`'s AC-6
check then rejects it as an ungranted tool, burning a re-prompt iteration on an undebuggable trace.
A tripwire comment already lives at the site (`llm.py:285-292`) naming this item.

Reproduction (verbatim from `docs/reviews/k027-parse-robustness.md` finding M-2):

| model emits | parsed as |
|---|---|
| `create_user({"name": "bob"})` | `ToolCall(name='bob', arguments={})` |
| `run_tool({"action": "delete"})` | `ToolCall(name='delete', arguments={})` |
| `x({"tool": "y", "args": {"a": 1}})` | `ToolCall(name='y', arguments={"a": 1})` |

Backlog names three candidate remedies, cheapest first — this is a precedence decision, not a
mechanical bug fix, hence the `architect` unit before implementation:
1. In `_normalize_tool_call`, skip the loose `name`/`action`/`tool` mapping when the surrounding
   content also matches `_BARE_CALL_OPEN` — partial hardening, available today.
2. Run the bare-call probe **first** — reorders the content fallback, needs its own regression
   pass over the JSON shapes it must not break.
3. Pass the granted tool names down as a **recognition filter** — closes most of the review's M-1
   residual too, but is a real layering decision (`llm.py`'s note that name validation belongs to
   the agent loop is deliberate).

**Also a secondary purpose of this dispatch:** the user is live-observing whether Claude Code
permission prompts fire on ordinary subagent Write/Edit calls under the repo's
`bypassPermissions` default (`.claude/settings.json`, pinned 2026-08-29) — this is a real,
independently-valuable backlog fix, not throwaway work, chosen because its shape (design pick →
test-first implementation → review) exercises that path end to end.

## Units

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| U1 | `architect` | `a32e8f024ac9dde46` | delivered | `docs/plans/bare-call-key-shadowing.md` | — → — | 120k tok, 35 tools |
| U2 | `tdd-engineer` | `a900b002b6a314590` | delivered | `llm.py`+`test_llm.py`+doc housekeeping | `analyst` → — | 110k tok, 40 tools |
| U3 | `analyst` | `a2a4f368a40dc9fb6` | accepted | `docs/reviews/bare-call-key-shadowing.md` | `analyst` → approve | 108k tok, 31 tools |

All three units delivered and verified. K-035 closed. `teco` fixed one trivial plan-doc typo
("Five" → "Six" test count, plan §1) flagged by the review directly, per the routing table's
trivial-single-file-no-brainer exception — no independent review, by construction.

**Permission-prompt observation (secondary purpose of this dispatch):** across U1-U3, prompts
fired on ordinary Write/Edit calls from all three subagent types (`architect`, `tdd-engineer`,
`analyst`) despite `.claude/settings.json` pinning `defaultMode: bypassPermissions`. Tally from
U2 alone (`tdd-engineer`): 6 prompts (4× `llm.py`, 1× `test_llm.py`, 1× `HISTORY.md`) — no sign
of one approval covering the rest of the run, worse than the prior session's `acceptEdits`
finding. See conversation for full detail; a kaizen entry is warranted.

U2 depends on U1's delivered plan. U3 (review gate) depends on U2's diff.

## Environment note

FalkorDB (`falkordb-dev`) was found down at dispatch time (needed for the offline pytest suite
per `falkor-chat/AGENTS.md`) — started by `teco` via `./scripts/start_falkordb.sh -d` before U1
was dispatched.
