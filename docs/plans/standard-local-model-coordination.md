# One standard local chat model (qwen) — Coordination

> **Status:** active · **Owner:** `teco` · **Tracks:** —

Requirements: `docs/requirements/standard-local-model.md` (Ready for design; review
`docs/reviews/standard-local-model.md`). Plan: `docs/plans/standard-local-model.md` (U1).

## Ledger

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| U1 | `architect` | `a02ab5cafe7d4c60f` (v1, v1.1); `a08345b8cad192b2c` (v1.2) | accepted (v1.3) | `docs/plans/standard-local-model.md` | `analyst` `a23589673358ed2fd` → needs changes (Passes 3–4); fresh `analyst` `ae4be96bee955416a` → needs changes (Pass 5: P5-M1 probe sized in chunks not concurrent calls per event, P5-M2 overlap order; stopping rule fires in letter only, compact Pass 6 next); v1.3 delivered (`a08345b8cad192b2c`, 8,724 words; S0 P3 pass depth `D` provisional 8, pending stakeholder) → approve with suggestions (Pass 6; stopping signal holds: 5 grep/`wc` checks, no further pass); Pass 6 sweep applied (8,718 words), stopping checks 1–5 verified by `teco` | 176k+236k / 166k+229k+248k tok; Pass 5 182k tok / 58 tools |
| U1-log | `tico` | `af984a9f2f907ec5d` (decision-log fold-in 2026-10-31: `D`=8, P5-R1 keep with setup, P5-R2 first nit; fold-in not re-gated — verbatim stakeholder answers; entry-2 plan-step pointer left, log append-only) | gated (FR-7/AC-1 overlap clause, stakeholder 2026-10-31; relayed to U1 architect same day) | `parallel` 1 decisions + FR-7/AC-1 overlap clause (P4-m4) | `analyst` `ae4be96bee955416a` → approve with suggestions (Pass 5; P5-R1 needs a stakeholder decision-log line, P5-R2 nits) — paused on stakeholder: probe size 8 / larger, P5-R1 | 50k tok |
| S0 | `devops` + operator | `a98295eb439e35794` | accepted (2026-11-01) | Evidence: scratchpad `s0-evidence.md`; per-model default honoured by JIT (path settled: per-model, `N` 16384, no fallback), `numParallelSessions` is the parallel key, global `defaultContextLength` 8192 custom unchanged. P1 pass (JIT reload at 16384/parallel 1, TTL 10h). P2 pass (instance count 1; two concurrent requests ran serially, 2.27 s and 1.16 s). P3 pass: `D` = 8 (stakeholder), 21-chunk probe, **k\* = 21/21**, all received 13:27:12, last finished 13:29:08 (116 s), 0 client disconnects, document `ready`, probe deleted; embedder left untouched (already explicit, no TTL, 2048). `laya` (8192/parallel 4) unloaded with approval, not reloaded. `teco` re-verified log counts, receipt second, last finish, 0 disconnects, probe gone, live qwen 16384/1 | — → pass | 106k tok / 28 tools |
| S1 | `tdd-engineer` | `a21c8694ebfb0e917` | accepted (F1 fixed; `teco` re-ran 39/39 and 23/23, own mutant of F1 killed by 5a2+9b2; was: `teco` re-ran: model-preflight 37/37, wrapper-scripts 23/23, `bash -n` clean; two own mutants killed: parallel constant 1→2, context floor 16384→16385; implementer's 14-mutant table reported) | plan §4 S1 (model preflight) | `analyst` `a4bbb528314bab413` (S1+S2 together, `docs/reviews/standard-local-model-impl.md`) → approve with suggestions (Pass 1: F1 fixed, F2 accepted, F5 ministral note carried to S3/S5b) | 76k tok / 9 tools |
| S2 | `coder` | `a691b36af57405d29` | accepted (gated with S1; was: `teco` re-verified: nemotron grep empty, provider map exactly ministral + qwen, `agent.tank.model` ministral, falkor-chat/README diff one clause; README Troubleshooting `≥` aligned to the script's `>=` by `teco`) | plan §4 S2 (nemotron removal, docs) | `analyst` → — | — |
| S3 | `devops` + operator | `a142ca5b2fd1eeafc` | accepted (no analyst gate: one-constant mechanical change, covered by the 39-test suite and `teco`'s own mutant 20481, killed) | **Settled `N` = 20,480** (`TANK_MIN_CONTEXT_LENGTH`, committed). Sizing: peak prompt 16,148 over four runs → rule band 20,480. Queue check: OpenCode aborts at ~300 s and auto-retries, run completed exit 0 (~6.3 min). Overlap check (reverse order, D=8): `tank s3-overlap` exit 0, 8 extractions queued ~30–34 s behind it, all `ready`, 0 client disconnects, no context error, probe deleted; caveat: ingest started ~18 s after the 2nd completion's slot start, and tank's next turn never queued behind extractions (run had finished) — a weaker form of the plan's check, accepted. Re-measure at 20,480: peak 16,147, TP-001 final `reason: stop`, not truncated. Preflight `OK … context 20480, parallel 1` against the live copy (`teco` re-ran). Evidence: scratchpad `s3-evidence.md`. S4 notes: score final-step `reason` (a `length` truncation exits 0); redirect stdin from `/dev/null` for `opencode run` (a foreground launch hung once); denied tool calls carry `status: error` + `error`, no `output`. Earlier findings (a)-(c) above stand | — → — | 87k + 127k tok |
| S4 | `qa-engineer` | `a51098d2a9af14090` | in-flight (stakeholder go 2026-11-01; `lms unload --all`, ministral load, embedder restore approved) | plan §5 trial report | — (report is the verdict) | — |
| S5a/S5b | `coder` | — | queued | plan §4 S5a or S5b, per S4 | `analyst` → — | — |
| S6 | `qa-engineer`, `teco` | — | queued | plan §6.2 acceptance; HISTORY | — → — | — |
| S7 | `tico` | — | queued | optional manual note (plan §7) | — → — | — |

## Constraints

- U3 and U5 are live LM Studio runs: the GPU is shared with concurrent work (e.g. the
  `typed-decision-models` model-bench coordination) — run only with LM Studio otherwise idle.

## Documentation impact

`opencode/agents/tank/README.md`, `opencode/agents/tank/opencode.json`,
`opencode/agents/tank/scripts/health-check.sh` (comment), `opencode/docs/BACKLOG.md` (nemotron
item), `opencode/docs/HISTORY.md`; falkor-chat docs/`HISTORY.md` if the FR-7 mechanism touches
falkor-chat; `opencode/docs/manuals/local-llm.md` (owner `tico`) if model guidance there changes.

## Baseline

AC-4 diff base: `185e326b` (plan v1.3 accepted; no S1–S7 change yet).
