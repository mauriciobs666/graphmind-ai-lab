# Change History — opencode

> Dated log of actual changes to the `opencode` component. Most recent first.

## 2026-11-01 — Standard local model: qwen trialled for `tank`, failed; ministral stays; `tank` gets a model preflight

Coordination: `docs/plans/standard-local-model-coordination.md`; requirements
`docs/requirements/standard-local-model.md`; plan `docs/plans/standard-local-model.md`; reviews
`docs/reviews/standard-local-model.md` (requirements/plan, Passes 1-6) and
`docs/reviews/standard-local-model-impl.md` (Passes 1-2, approve with suggestions); trial and
acceptance in `docs/test-reports/standard-local-model-report.md`.

**Trial outcome: `fail` (FR-4/AC-2).** `qwen/qwen3-4b-2507` made up findings in all three
TP-001 (health check) runs under every definition of "unsupported" the review tried (13, 10, 6 or
4 across the three runs; every run at least one). It met FR-4(b) (3/3 clean refusals on TP-002
and TP-003), but those texts echo the prompt's own examples, so the fail rests on FR-4(a).
Ministral also embellishes (8 unsupported findings on the same rows under the strict
definition), so the finding is that qwen failed the absolute bar, not that it is worse.
`tank` keeps `mistralai/ministral-3-3b`, the second accepted exception to the one-chat-model
rule alongside the salesperson demo (AC-5). The nemotron provider entry is removed; the qwen
entry stays provisioned in `opencode/agents/tank/opencode.json`.

**What shipped.**
- `opencode/agents/tank/scripts/model-preflight.sh`, called by `health-check.sh`, `bring-up.sh`
  and `tear-down.sh` before `opencode run`: refuses unless exactly one copy of the model is
  loaded with context >= `TANK_MIN_CONTEXT_LENGTH` (20480) and parallel ==
  `TANK_REQUIRED_PARALLEL` (1). **A copy off that parallel refuses every `tank` run**, and a
  ministral loaded at the demo's 8192 is refused by design. Reads only; `TANK_MODEL_PREFLIGHT=0`
  bypasses. Tests: `opencode/agents/tank/tests/model-preflight.sh` (45).
- `N` = 20480, settled in S3: TP-001's peak prompt is 16,147 tokens; at 16384 a run exited 0 with
  a truncated final step (`reason: length`).
- FR-7 mechanism: LM Studio per-model defaults (qwen: context 20480, `numParallelSessions` 1)
  are honoured by JIT loads, so `tank` and falkor-chat's `:8200` share one copy. With one slot
  their requests queue: 21 concurrent extraction calls all finished in 116 s with no client
  disconnect (S0, depth 8 chosen); an 8-chunk ingest during a `tank` run queued ~30-34 s and
  reached `ready` (S3, a weaker overlap than planned: the `tank` run had finished before it
  needed to wait). OpenCode aborts a request queued ~300 s and retries automatically; a run
  can exit 0 with `reason: length`, so score the final step's `reason`.
- Docs: `opencode/agents/tank/README.md` (Prerequisites: load ministral by hand at 20480 /
  parallel 1, unload before a demo; Choosing a model; Troubleshooting), the manual
  `opencode/docs/manuals/local-llm.md` (floor 20480, "Using `tank` with ministral"),
  `falkor-chat/README.md` (one clause).

**Measured, not gated.** AC-6: qwen + embedder (2048) fit the 6 GiB GPU with no spill
(dedicated 5.7-5.8 GiB of 6141 MiB, shared 218-268 MiB; single readings, `tank` overlap not
measured); conditional on the embedder staying explicit at 2048. AC-4: no commit of this change
touches `falkor-chat/server` or `falkor-chat/scripts`; the demo was not re-run end to end.

## 2026-09-13 — `tank`: headless, local-model OpenCode DevOps agent shipped

Closes this coordination — no formal M-number here, tracked as M0 per the plan's own
`Tracks: — (M0)` convention. Full multi-unit trail:
`opencode/docs/plans/devops-opencode-headless-coordination.md`.

**What shipped.** `opencode/agents/tank/` — a headless OpenCode agent (`opencode run --agent
tank`, no live Claude Code session, no human present mid-run) giving the `devops` role an
unattended mode: a read-only health/hygiene check by default, plus bring-up/teardown of one
Docker-Compose environment it itself brought up. Three mechanisms (full design and rationale:
`opencode/docs/plans/devops-opencode-headless.md` §3, Version 3):

1. **Wrapper-script security mechanism** (§3.2/§3.3) — `tank` never builds a `docker compose`
   invocation from a model-authored string. Its `bash` tool may only invoke three fixed,
   repo-committed scripts (`opencode/agents/tank/scripts/{compose-status,compose-up,
   compose-down}.sh`) by exact absolute path, each taking exactly one argument (a slug) looked up
   by true key equality against `opencode/agents/tank/environments.json` — the scripts construct
   the entire compose argv themselves, in code. Replaced an earlier glob-pattern-based design
   after a live-reproduced smuggling bypass (see Review below). An ownership marker
   (`opencode/agents/tank/state/<slug>.json`, written only by the scripts' own code) with a
   freshness bound gives bring-up/teardown symmetry.
2. **Shared-persona split** (`claude/devops/devops-persona.md`, `claude/devops/devops.md`,
   `claude/devops/scripts/sync-persona.sh`) — one canonical, runtime-agnostic persona; `tank`'s
   `opencode.json` live-includes it via `{file:...}`, `claude devops` regenerates its
   marker-wrapped copy via the sync script.
3. **Static permission-table safety net** (`opencode/agents/tank/opencode.json`,
   `permission.bash`) — deny-by-default, then a small read-only allow-list plus the three
   wrapper-script patterns, then a defense-in-depth deny tier (destructive-command catalog,
   repeated-scoping-flag net, shell-metacharacter net). Enforcement lives at this mechanical
   layer, never in the prompt.

Outer scripts (`opencode/agents/tank/scripts/{health-check,bring-up,tear-down}.sh`) are the
reviewed, scriptable entry point. Doc entry points: `opencode/AGENTS.md`/`CLAUDE.md`,
`opencode/agents/tank/README.md`; root `AGENTS.md` updated to match.

**Live-testing arc** (full transcripts/evidence: `opencode/docs/test-reports/
devops-opencode-headless-report.md`, 4 dated revision notes):
- **DEF-1** (environment misconfiguration, resolved) — LM Studio's loaded model had a context
  window below `tank`'s own resolved prompt size; every run 400'd before any user message.
  Not an artifact defect — fixed by reloading the model with adequate context.
- **DEF-2** (context-overflow bug, fully fixed) — the model's own choice of
  `docker system df --verbose`, allowed by a wildcard permission pattern, could dump enough
  output to overflow the context on the next turn. Narrowed to an exact-match
  `"docker system df"` allow entry, forcing the compact form regardless of model choice.
- **DEF-3** (behavioral-compliance gap, partially resolved) — asked to do something destructive
  or arbitrary/unauthorized, the local model sometimes asked for confirmation, retried after a
  denial, or wandered into an unrequested tangent before ending on a question — never executing
  anything, but defeating a clean unattended signal. Two rounds of addendum strengthening (an
  explicit "completely unattended" framing plus two labeled wrong/right contrastive examples)
  closed both plan-literal test scenarios cleanly, but the fix does not generalize to arbitrary
  unseen phrasings on this 3B model — capped at one further round by the stakeholder after an
  independent `architect` opinion that the plan's actual acceptance criteria ("refused, reported,
  never executed") were already met, and that further iteration risked chasing a model-capability
  ceiling rather than a wording gap.
- **Auto-title 500** (upstream OpenCode bug, worked around) — background session-title
  generation 500s against this model's strict-alternation chat template (unfixed upstream issue,
  cited in `opencode/agents/tank/README.md`); not a `tank` defect. Worked around by passing
  `opencode run --title ...` from the outer scripts.

Mechanical safety net (the permission table + wrapper scripts) held in every sampled run across
every round — no destructive or unauthorized action was ever actually executed, including during
DEF-3's unresolved residual.

**Review.** Security review (`opencode/docs/reviews/devops-opencode-headless.md`): Pass 1 found
one blocker (a live-reproduced glob-smuggling bypass of the original permission design) and two
majors (marker basename collision, no marker tamper/freshness check), all fixed by the
wrapper-script redesign and independently re-verified live in Pass 2 (approve with suggestions)
and Pass 3/G2 against the shipped code (approve). Implementation review
(`opencode/docs/reviews/devops-opencode-headless-impl.md`, G1, `analyst`): approve with
suggestions, no blocker/major.

**Verified independently, not taken on report** (`teco`, throughout — full detail in the
coordination ledger cited above): the wrapper scripts' mutation-testing claim corroborated with
an independently-run mutation; every `opencode.json` change re-checked live via
`opencode debug agent tank` (resolved permission-table shape, entry count/order, prompt content);
Docker/volume state confirmed unchanged before/after every live-model round; the shared
`falkordb-dev` stop/restart bracket needed for live testing independently confirmed clean before
and after.
