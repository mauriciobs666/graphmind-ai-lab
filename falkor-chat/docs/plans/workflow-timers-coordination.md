# K-028 workflow timers — coordination log

> **Status:** active · **Owner:** `teco` · **Tracks:** K-028

Goal: deliver `falkor-chat` backlog item K-028 — durable due-time on a parked run plus a ticking
mechanism, reusing the existing resume CAS so a timer wakeup and a concurrent human signal cannot
double-resume. Requirements source: `falkor-chat/docs/BACKLOG.md` §K-028. Stakeholder pre-decision:
the architect's recommended ticking mechanism is pre-approved unless the plan gate contests it or a
product-behavior fork appears.

## Context established at decomposition (2026-08-21)

- CPG: `cpg_falkorchat`, built 2026-08-17T00:40:42Z, **stale** — 6 commits touched
  `falkor-chat/server` since build (K-027 work: executor/guard paths). Specialists should prefer
  reading files for anything K-027 touched.
- Resume CAS: `server/falkorchat/executor.py` `Executor.resume()` (~:367), repo
  `resume_run`/`resume_run_with_ctx`; signal endpoint `POST /workflow-runs/{run_id}/input`
  (`server/falkorchat/api.py:262`).
- Component constraints: FalkorDB OpenCypher (no APOC/GDS), index-before-constraint, RAM callout
  for any new index (AGENTS.md rule 6), `./scripts/test_queries.sh` must stay green.

## Documentation-impact scan

| Doc | Why |
|---|---|
| `falkor-chat/docs/DESIGN.md` §6 (and §14 if REST surface changes) | wait semantics gain a timer release path |
| `falkor-chat/docs/QUERIES.md` | any new due-time sweep query |
| `falkor-chat/AGENTS.md` | new env vars/flags/scripts, executor notes |
| `falkor-chat/docs/BACKLOG.md` | K-028 status flip; K-024 D-C cross-references (lines ~299) |
| `falkor-chat/docs/HISTORY.md` | delivery entry |
| `falkor-chat/docs/manuals/workflows.md` | states "waiting is **not a timer**" — invalidated; `tico` owns |

## Units

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| U1 design plan | `architect` | `a8cc3fd8459fad64a` | accepted | `falkor-chat/docs/plans/workflow-timers.md` | U2 → approve w/ suggestions | 154k tok / 22 tools |
| U2 plan gate | `analyst` | `ac36b94c8b4b3f340` | accepted | `falkor-chat/docs/reviews/workflow-timers.md` | — (verdict: approve w/ suggestions; M-1 + m-1..m-4 + nits routed into U3a/U3b/U3c briefs) | 131k tok / 28 tools |
| U3a impl steps 1–2 (repository + executor + QUERIES §12.x + SHA note) | `coder` | — | queued | code + unit tests green | U4 | — |
| U3b impl steps 3–4 (services + acceptance/CAS-contention tests) | `coder` | — | queued | code + tests green | U4 | — |
| U3c impl steps 5–6 (ticker, config, app, REST route + docs/ops sweep) | `coder` | — | queued | code + tests + docs per scan | U4 | — |
| U4 diff re-gate | `analyst` | — | queued | `falkor-chat/docs/reviews/workflow-timers-impl.md` | — | — |
| U5 acceptance QA (incl. manual walkthrough check) | `qa-engineer` | — | queued | `falkor-chat/docs/test-plans/workflow-timers.md` + `docs/test-reports/workflow-timers-report.md` | — | — |
| U6 manual update | `tico` | — | queued | `falkor-chat/docs/manuals/workflows.md` (in-place edit) | U7 + U5 walkthrough | — |
| U7 manual factual gate | `analyst` | — | queued | pass folded into a dated section of an existing review or short verdict | — | — |

Sequencing: U1 → U2 → U3 → U4 → U6 → (U7 ∥ U5). Conditional units: `devops` if the chosen ticking
mechanism lands on an external process surface; `graph-dba` design note only if graph modeling
exceeds a single due-time index (plan gate judges). Integrated baseline (full suites) run by
coordinator before close.
