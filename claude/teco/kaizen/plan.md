# Kaizen — Improvement Plan: teco

> Forward-looking backlog for the `teco` agent.
> Status: 🔵 proposed · 🟡 in-progress · ✅ done (then moved to history.md) · ⚪ rejected/deferred
> Last reviewed: 2026-07-12

## Active

| ID | Added | Priority | Status | Summary |
|------|------------|----------|--------|---------|
| K-002 | 2026-06-20 | medium | 🔵 | Evaluate Claude Code *agent teams* / *background agents* as a better substrate for teco than nested subagent calls. |
| K-003 | 2026-07-12 | high | 🔵 | Review-gate invariant: prove it on a fully-gated run (falkor-chat K-022) or renegotiate the prompt honestly. |

> **K-001 — validate nested delegation end-to-end — ✅ done 2026-07-09** (moved to history.md).
> Live run: falkor-chat M3 slice 1 through teco → architect → graph-dba → tdd-engineer, all
> checklist items passed. Launch brief + observation checklist preserved at
> [`k001-run-brief.md`](./k001-run-brief.md).

### K-002 — Consider agent teams / background agents
- **Status:** 🔵 proposed
- **Priority:** medium
- **Rationale:** Claude Code now has first-class *agent teams* (sessions that communicate) and *background agents* — possibly a cleaner orchestration substrate than a subagent calling subagents via the `Agent` tool, especially for parallel or long-running work.
- **Proposed change:** Read `code.claude.com/docs/en/agent-teams` and `/en/agent-view`; assess whether teco should be reframed as a team lead / use background tasks. Update `agent-standards` skill with findings.
- **Notes:** Surfaced during creation-time verification (2026-06-20). New primitives not yet in cobb's resident notes. **2026-07-12:** concrete sub-case to evaluate — the harness now supports `SendMessage` continuation of a previously spawned agent with its context intact; teco's defect→fix→re-run loop currently re-spawns cold agents with fresh briefs each iteration (re-orientation cost every cycle). Continuing the original implementer with the review/report path may be materially cheaper — assess alongside agent teams.

### K-003 — Review-gate invariant: prove it or renegotiate it
- **Status:** 🔵 proposed
- **Priority:** high
- **Rationale:** Team assessment 2026-07-12: the prompt's core quality commitment — "work ships independently reviewed; when you trim ceremony, the review gate is the last thing to go" — is contradicted by the only live run (K-001, 2026-07-09), where code review was the *first* thing trimmed ("left to the user") and QA was deemed not warranted. The likely pressure is cost (~100k subagent tokens / ~45 min for the ungated 2-item slice; the default gates roughly double the delegation count), and teco currently resolves the review-by-default vs. right-altitude tension ad hoc — so far always by skipping review. An invariant that never fires is hopeful prose.
- **Proposed change:** Make falkor-chat **K-022** (executor implementation, next on that component's critical path) the first fully-gated run: brief teco with the analyst post-implementation review as a non-negotiable done-condition (now baked into the K-022 item in `falkor-chat/docs/BACKLOG.md`), capture the cost datapoint against the K-001 baseline, and observe whether the gate catches anything worth its tokens. Then either (a) keep the invariant, with the datapoint as its justification, or (b) rewrite the guardrail to match reality (e.g. review gates on explicit risk signals rather than by default) — but stop carrying a default the coordinator doesn't apply.
- **Notes:** Counterparts: `falkor-chat/docs/BACKLOG.md` K-022 review-gate note; `analyst` K-001 (the same run is its code-review shakedown); `qa-engineer` K-003 (defect→fix→re-run loop, if the review returns findings). Evidence trail: `teco/kaizen/history.md` 2026-07-09 K-001 entry ("review left to the user").

## Parking lot / ideas
- ~~A routing cheat-sheet / decision tree teco self-checks before delegating (which specialist for which signal), to reduce mis-routing between `coder` and `tdd-engineer`.~~ *(✅ Resolved 2026-07-09: the roster is now an explicit routing table — task shape → owner → tie-breaker — with the coder-vs-tdd efficiency rule on both implementer rows, plus a separate handoff-contracts list. See history.md.)*
- Guard against over-orchestration: a heuristic for "this is a single-specialist job, skip the breakdown."
