# Kaizen — Improvement Plan: architect

> Forward-looking backlog for the `architect` agent.
> Status: 🔵 proposed · 🟡 in-progress · ✅ done (then moved to history.md) · ⚪ rejected/deferred
> Last reviewed: 2026-09-07

## Active

| ID | Added | Priority | Status | Summary |
|------|------------|----------|--------|---------|
| K-004 | 2026-09-07 | medium | 🔵 | Undocumented `llm.py` probe-order collision: a bare call whose sole argument is a `tool_calls` list |
| K-005 | 2026-09-07 | medium | 🔵 | `ws:acme` is a populated live workspace, not a scratch graph — not warned about in `falkor-chat/AGENTS.md` |

### K-004 — `falkor-chat` `llm.py`: the `tool_calls`-argument probe collision is real and untested
- **Status:** 🔵 proposed
- **Priority:** medium
- **Rationale:** In `server/falkorchat/llm.py`, `_parse_content_tool_calls` runs the JSON probe
  before the bare-call probe, and its `{"tool_calls": [...]}` envelope branch runs *before* the
  K-035 `_BARE_CALL_OPEN` guard. So a content string of the form `x({"tool_calls": [...]})` is
  simultaneously a bare-call match and a native-envelope match, and only the probe order decides
  which wins. Re-verified 2026-09-07 against `llm.py:284-322` and the 857-line
  `server/tests/test_llm.py`: the six K-035 pins cover `name`/`action`/`tool` shadowing only, and
  the two `tool_calls`-envelope tests use non-bare-call content — **nothing pins this shape.** The
  function's docstring names a different residual (a genuine envelope beside an *unrelated*
  bare-call line); this one is not mentioned. Anyone reordering the probes flips the branch with a
  green suite.
- **Proposed change:** file it in `falkor-chat/docs/BACKLOG.md` as a test-gap item — one
  characterization pin driving `llm.chat(...)` with `x({"tool_calls":[{...}]})`, plus one sentence
  in `_parse_content_tool_calls`'s docstring residual paragraph naming the collision. Both targets
  are `falkor-chat` source/docs, outside `architect`'s and `cobb`'s write remit: route via `teco`
  to `qa-engineer` (backlog item) and `tdd-engineer` (the pin).
- **Notes:** from `kaizen_team` entry `a3f5c8e2-4b1d-4e9a-9c7f-6d2b1a8e5f30` (2026-09-01),
  distillation pass 2 unit U7.

### K-005 — Warn in `falkor-chat/AGENTS.md` that `ws:acme` is populated, not scratch
- **Status:** 🔵 proposed
- **Priority:** medium
- **Rationale:** `config.WS_ID` defaults to `acme` (`server/falkorchat/config.py:16`) and `ws:acme`
  is the served demo tenant, not an empty graph. Live label counts re-read 2026-09-07: 544 Entity,
  87 Chunk, 78 StepRun, 52 Message, 29 Document, 29 Step, 21 WorkflowRun, 13 TraceEvent, 11
  WorkflowDefSnapshot, 2 Channel, 2 Thread, 1 Agent, 1 ReadCursor, 1 User. Any script or design
  that reads `$FALKORCHAT_WS_ID` without pinning it operates on that data. `docs/SERVER.md` §1.3
  documents the *tenancy decision* ("there is no `FALKORCHAT_DEMO_WS`, and there never will be")
  and `AGENTS.md` mentions `ws:acme` only in passing (the burned-`v6` note); neither warns that the
  default target is populated.
- **Proposed change:** one row/clause in `falkor-chat/AGENTS.md`'s scripts section stating that the
  default `$FALKORCHAT_WS_ID` (`acme`) is the populated demo tenant and that any destructive or
  seeding script must pin a workspace explicitly. `falkor-chat/AGENTS.md` is outside `cobb`'s write
  remit — route via `teco` to whoever owns that context file.
- **Notes:** from `kaizen_team` entry `b7d5e214-0a93-4c68-9f37-1e4c8a06b2d9` (2026-09-02), U7. That
  entry's second half — a test-design corollary — was `MENTIONS`-tagged to `qa-engineer` in the
  same pass, so the node survives for `qa-engineer`'s own distillation.

## Parking lot / ideas
- **Live-probe seam check (parked, 2026-08-09 — ex-inbox entry 4).** Before scheduling a plan step that runs a live probe/experiment, check the target actually has a graph/tenancy/environment seam that makes the probe isolated (e.g. a throwaway `ws:<probe>` workspace) — surfaced designing K-031 (falkor-chat's `publish_def` writes to a hardcoded `reference` graph with no per-workspace override; `materialize_snapshot`'s shared query constant against a throwaway `ws:` graph was the workaround). Judged narrow/single-occurrence, not promoted to Guardrails — revisit if a second instance turns up.
- A short self-review checklist before delivering a plan (every step concrete & file-specific, alternatives recorded, risks listed, handoff summary present) — and, since 2026-07-27, the canonical header block present and its `Status:`/`Owner:`/`Tracks:` filled.
- **`architect` owns one recurring flip it isn't told about yet (noted 2026-07-27).** Root `AGENTS.md`'s routing table makes the architect the performer of the `Status: archived` flip on `plans/<slug>.md` at milestone close, on `teco`'s coordination. Today that reaches the agent only through the closing unit's brief; if a close ever ships with plans left `active`, the fix is one line in this prompt.
- Optionally delegate wide codebase sweeps to the Explore agent by default for large repos.
- Extend `hooks/guard-plan-doc-writes.sh` to cover Bash write patterns (`sed -i`, `>` redirects, `git commit`, package installs) **only if** the prompt-guarded Bash ever proves leaky in practice — deliberately left out on 2026-07-08 (see history).
