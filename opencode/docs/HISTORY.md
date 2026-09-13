# Change History — opencode

> Dated log of actual changes to the `opencode` component. Most recent first.

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
