# Agent knowledge-base strategy — Track 1 implementation coordination

> **Status:** active · **Owner:** `teco` · **Tracks:** K-030 (`claude/cobb/kaizen/plan.md`)

Third coordination document on this topic slug (ordinal per `AGENTS.md` collision rule 5 — the
prior two, `agent-knowledge-base-strategy-coordination.md` (Stage 0) and
`agent-knowledge-base-strategy2-coordination.md` (substrate design revision), are both archived
and executed against, so this is a successor, not a revision-in-place).

Coordinates **implementation** of Track 1 (FR-8/FR-9, raw-capture migration into a new, dedicated
`ws:agent-team` falkor-chat workspace) per `claude/docs/plans/agent-knowledge-base-strategy.md`
§3's Track 1 table (Stages 1-5) and its "Ready to implement" closing section. This is a
**code-implementation chain** (touches `falkor-chat/server` source, tests, config, and ops
scripts) — commits land per verified unit, not batched at the end. Track 2 (Stages 6-9,
distilled-knowledge ingestion) is explicitly out of scope here, sequenced after Track 1 completes
per FR-9 — not dispatched from this coordination.

**CPG note (carried forward, not re-derived):** `cpg_falkorchat` is stale for this work (built
2026-09-12, source `ca25a20`, predates `document-ingestion2`'s full shipping `8906878`) — recorded
once in the prior coordination doc (`agent-knowledge-base-strategy2-coordination.md`), still true,
not rebuilt for this coordination; every falkor-chat-side unit reads current source directly.

## Sequencing plan (per the plan's own Track 1 table, §3)

- **Stage 1** (`produced_by` extension) — `graph-dba` design note → `coder` implements → `analyst`
  diff-scoped review. The interface itself (§4.1) was already independently reviewed twice at the
  parent-plan level (Pass 1/Pass 2) — no separate pre-implementation gate on the design note itself,
  one diff-scoped gate on the actual implementation before merge.
- **Stage 2** (`ws:agent-team` bootstrap + seed script) — `graph-dba`/`devops`, sequenced after
  Stage 1's design note lands (the plan's own stated dependency, §3 table) so the seed script's
  `Agent` node shape matches exactly what Stage 1's design resolves `produced_by` against.
- **Stage 3** (dedicated `FALKORCHAT_WS_ID=agent-team` process) — `devops` (design now, in
  parallel with Stage 1/2; implementation/bring-up only once Stage 1 code is merged and Stage 2's
  workspace exists, per the plan's own explicit dependency), `graph-dba` reviews naming/config.
- **Stage 4** (write-convention rollout, pilot 1-2 agents) — `cobb`, after Stages 1-3 complete.
- **Stage 5** (curator hook: `agent-maintenance` SKILL.md §5 gains the falkor-chat read/clear step)
  — `cobb`, after Stage 4.

## Ledger

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| U1 | `graph-dba` | `a90059122a9d47f3c` | accepted | `falkor-chat/docs/plans/agent-team-ingestion-graph.md` (Stage 1 design note) | verified directly by `teco` (grep for `create_document`/`create_document_with_auto_supersede` call sites; independent `EXPLAIN` re-run against `ws:demo` matching both claimed plans; `claude/graph-dba/falkordb-quirks.md:639` entry confirmed) | 231804 tok / 39 tools |
| U2 | `devops` | `a70b7b50d8ada1d33` | accepted | `falkor-chat/scripts/start_agent_team.sh` (draft, port 8200, not run) | `graph-dba` (`a808e3665f094b8f8`) → sound as-is, no changes | 145746 tok / 20 tools |
| U3 | `graph-dba` | `a808e3665f094b8f8` | accepted | naming/config review of U2's script | verdict: sound as-is, no changes | 106815 tok / 13 tools |

| U4 | `coder` | `a61c94585706ce329` | accepted | Stage 1 implementation: `produced_by` on `ingest_document`/`ingest_documents` + `AgentNotFoundError`, full §7 test coverage (14 new tests), mutation-tested (raise-deletion, rejected-alternative-reimplementation, silent-arg-swap — all caught/reproduced); committed `8a1449a` | `analyst` (`a8dd9342917ec8068`) → approve | 219590 tok / 71 tools |
| U5 | `graph-dba` | `a90059122a9d47f3c` (resumed) | accepted | Stage 2: `ws:agent-team` bootstrap (live) + `seed_agent_team.sh` (live, idempotent); committed `de4158e` | verified directly by `teco` (`mcp__cypher__query` against `ws:agent-team`: 13 `Agent` nodes matching `claude/AGENTS.md`'s roster exactly, `Agent.agentId` RANGE index present; `seed_agent_team.sh` + `falkor-chat/AGENTS.md` row confirmed on disk) | 270105 tok / 19 tools |
| U6 | `analyst` | `a8dd9342917ec8068` | accepted | diff-scoped review, `falkor-chat/docs/reviews/agent-team-ingestion-produced-by.md`; committed `8a1449a` | verdict: approve (1 minor — constant placement, 1 nit — docstring wrap; both fixed directly by `teco`, re-verified: full suite 2844 passed/14 deselected/0 failed after the fix, shared `reference`-graph wipe hazard repaired and re-verified both times) | 117445 tok / 44 tools |

## Notes

- U1 and U2 dispatched in parallel: independent files/components (falkor-chat design doc vs.
  ops/process design), no shared file, no shared DB state, no claim one depends on the other to
  state — safe to run concurrently per this coordination's own file-collision/claim-collision
  check.
- U2 is design-only for now; Stage 3's actual bring-up/deployment is a later unit, sequenced after
  Stage 1 (coder-implemented, `produced_by` merged) and Stage 2 (workspace bootstrapped) land.

## Stage 1/2 close-out (2026-09-18)

Stages 1-2 both `accepted` and committed (`fb10a7a`, `cfe1a78`, `de4158e`, `8a1449a`). U4's `coder`
run was interrupted mid-work by a session-wide rate-limit kill and resumed via `SendMessage` to its
recorded `agentId` rather than re-dispatched — its on-disk state and reasoning survived intact;
`teco` independently re-verified the resumed deliverable in full (diff --stat, full suite rerun,
new-test-name greps, unmodified-regression-test check, kaizen entry) before treating it as
delivered. `analyst`'s diff-scoped gate (U6) approved with two low-stakes findings, both fixed
directly by `teco` as genuinely trivial single-file no-brainers (moving two query-text constants
to sit beside their consumer method, mirroring this file's own `_SINCE_PLAIN`/`_SINCE_KEYSET`
precedent; a docstring line-wrap nit) — re-verified green (2844 passed/14 deselected/0 failed)
after each of two full-suite reruns (mine and `analyst`'s), each of which triggers the documented
`falkor-chat/AGENTS.md` "default pytest wipes `reference`" hazard; both repaired and independently
re-verified via `verify_workflows.sh`/`verify_salesperson.sh`/`verify_catalog.sh` on `ws:demo`/
`ws:acme`.

**Next:** Stage 3 bring-up (`start_agent_team.sh`, already designed+reviewed in U2/U3) can now run
for real — Stage 1's code is merged and Stage 2's workspace exists, the plan's own stated
dependency for it. Not yet dispatched.
