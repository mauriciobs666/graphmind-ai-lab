# One standard local chat model (qwen) — Coordination

> **Status:** active · **Owner:** `teco` · **Tracks:** —

Requirements: `docs/requirements/standard-local-model.md` (Ready for design; review
`docs/reviews/standard-local-model.md`). Plan: `docs/plans/standard-local-model.md` (U1).

## Ledger

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| U1 | `architect` | `a02ab5cafe7d4c60f` (v1, v1.1); `a08345b8cad192b2c` (v1.2) | accepted (v1.3) | `docs/plans/standard-local-model.md` | `analyst` `a23589673358ed2fd` → needs changes (Passes 3–4); fresh `analyst` `ae4be96bee955416a` → needs changes (Pass 5: P5-M1 probe sized in chunks not concurrent calls per event, P5-M2 overlap order; stopping rule fires in letter only, compact Pass 6 next); v1.3 delivered (`a08345b8cad192b2c`, 8,724 words; S0 P3 pass depth `D` provisional 8, pending stakeholder) → approve with suggestions (Pass 6; stopping signal holds: 5 grep/`wc` checks, no further pass); Pass 6 sweep applied (8,718 words), stopping checks 1–5 verified by `teco` | 176k+236k / 166k+229k+248k tok; Pass 5 182k tok / 58 tools |
| U1-log | `tico` | `af984a9f2f907ec5d` (decision-log fold-in 2026-10-31: `D`=8, P5-R1 keep with setup, P5-R2 first nit; fold-in not re-gated — verbatim stakeholder answers; entry-2 plan-step pointer left, log append-only) | gated (FR-7/AC-1 overlap clause, stakeholder 2026-10-31; relayed to U1 architect same day) | `parallel` 1 decisions + FR-7/AC-1 overlap clause (P4-m4) | `analyst` `ae4be96bee955416a` → approve with suggestions (Pass 5; P5-R1 needs a stakeholder decision-log line, P5-R2 nits) — paused on stakeholder: probe size 8 / larger, P5-R1 | 50k tok |
| S0 | `devops` + operator | — | queued — **paused by stakeholder 2026-10-31: LM Studio unavailable; resume on stakeholder trigger** | plan §4 S0 (P1–P3); pass depth **`D` = 8** (stakeholder 2026-10-31). Step 1 operator save done 2026-10-31 21:17 (read-only check by `teco`): qwen per-model default file created then (no prior qwen default existed), keys `llm.load.contextLength` 16384, `llm.load.numParallelSessions` 1; a stray `llm.load.llama.evalBatchSize` 2048 (operator: unintended) was reset by the operator, file re-read at 21:20 holds only the two planned keys | — → — | — |
| S1 | `tdd-engineer` | — | queued | plan §4 S1 (model preflight) | `analyst` → — | — |
| S2 | `coder` | — | queued | plan §4 S2 (nemotron removal, docs) | `analyst` → — | — |
| S3 | `devops` + operator | — | queued | plan §4 S3 (LM Studio config, live probes) | — → — | — |
| S4 | `qa-engineer` | — | queued | plan §5 trial report | — (report is the verdict) | — |
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
