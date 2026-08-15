# CPG agent adoption — Feature Requirements
> **Status:** Interviewing · **Owner:** `tico` · **Tracks:** — (M<n> TBD) · **Last updated:** 2026-08-15

## Intent
Turn the CPG from a narrowly-wired, largely dormant capability into something any
agent doing code-level work routinely benefits from — without turning any of the
machinery around it into something automatic or proactive. Concretely: widen who can
reach it beyond the original three consumers, make *discovering* it a normal, default
step in an agent's orientation rather than something that has to be pointed out, give
agents a way to judge (and flag) how stale a loaded graph might be, and let broader
component coverage happen through the existing on-demand model, just exercised more
often. The MCP access path, auto-rebuild, proactive build-out, and usage dashboards
are all explicitly not part of this — see Out of scope.

## Problem & current state
The `cpg-analysis` skill (reads a Joern Code Property Graph loaded in FalkorDB) was
scoped at M2 (`docs/plans/m2-cpg-analysis-skill.md`, delivered ✅) to exactly three
consumer agents — `analyst`, `architect`, `qa-engineer` — plus `graph-dba` as the
producer. `docs/requirements/cpg-query-access.md` (M3, delivered ✅, archived) later
fixed the read path onto the `mcp__cpg__query` MCP tool for that same three-agent
set; its own `Scope` section named `analyst`/`architect`/`qa-engineer` as **in**, and
"general agent access to FalkorDB" as **out**.

**Prior-decision note (surfaced 2026-08-15, before this interview's first question):**
this feature request — widening which agents benefit from the CPG — touches ground
already ruled on at M2/M3. It is not a reversal of what M2/M3 *built* (the MCP tool,
the skill's recipes), but it does revisit the *consumer-scope* boundary those
milestones drew deliberately. Whatever this interview settles must say explicitly
whether it extends that boundary (new consumers added, M2/M3 untouched) or supersedes
it — the resulting requirements doc and any downstream plan need to reconcile with
`m2-cpg-analysis-skill.md` and `cpg-query-access.md` rather than silently diverge from
them (`docs/BACKLOG.md` M2/M3 rows are the canonical status log for both).

Investigation ahead of this interview (tico, 2026-08-15) found, concretely:
- Only `analyst`, `architect`, `qa-engineer` (consumers) and `graph-dba` (producer) are
  wired to `cpg-analysis` / `mcp__cpg__query` today. `coder`/`tdd-engineer` were
  considered for a similar skill extension once before (`python-web-quirks`, 2026-08-09)
  but `cpg-analysis` was not carried along at that time.
- Routing language is conditional ("when a CPG is loaded") — there is no proactive
  discovery step, and `graph-dba`'s own description frames CPG generation as
  deliberately rare/on-demand, not something to suggest.
- Only two components have a CPG actually built and loaded: `cpg_falkorchat` (2,037
  methods) and `cpg_salesperson` (359 methods), confirmed live via `GRAPH.QUERY` on
  2026-08-15. No other component (`cpg/`, `claude/`, `skills/`, `opencode/`,
  `mcp-monitor/`, `kiro/`) has one.
- No evidence of the CPG being consulted in real (non-CPG-meta) task work since it was
  built — e.g. the most recent `analyst` review, `falkor-chat/docs/reviews/graphrag-eval.md`,
  makes no mention of it.
- No documented freshness/rebuild convention — `cpg_falkorchat` traces to M3-era work;
  falkor-chat has had substantial commits since with no recorded CPG refresh tied to them.

## User stories
- As an agent doing code-level work (not limited to `analyst`/`architect`/`qa-engineer`),
  I want to discover whether a relevant CPG exists as part of my normal task
  orientation, so that I can use it without being told or already knowing it's there.
- As the stakeholder, I want to be able to spot-check a task's transcript and see
  evidence the agent considered or used the CPG when it plausibly applied, so that I
  can trust the wiring isn't just decorative.
- As an agent consulting a loaded CPG, I want to know how current the graph might be,
  so that I don't treat possibly-stale results as ground truth.
- As an agent that notices a stale-looking CPG, I want to be able to surface a
  suggestion to refresh it, so that refreshing stays a deliberate, on-demand action
  rather than something silently trusted or silently auto-triggered.
- As the stakeholder, I want the existing on-demand CPG-build model to get invoked
  more often in practice, so that coverage grows organically without a costly
  proactive build-out project.

## Functional requirements
- **FR-1** — Any agent doing code-level work (not limited to the three agents wired
  today) must be able to determine whether a CPG relevant to its current task exists.
- **FR-2** — That determination must happen as a normal part of the agent's task
  orientation, not only when a CPG's existence is pointed out by the user or another
  agent.
- **FR-3** — When a relevant, available CPG is found, the agent must be able to use it
  for its task (impact analysis, RCA, review, test-gap, or other code-level reasoning)
  rather than defaulting straight to reading files individually.
- **FR-4** — A discovery check that comes up empty (component has no CPG loaded) must
  not introduce noticeable friction or noise to the task.
- **FR-5** — An agent consulting a loaded CPG must be able to obtain some indication of
  how current the graph is relative to the code it describes.
- **FR-6** — When a CPG appears stale, the agent must be able to surface a suggestion
  to refresh it. It must not silently treat a stale graph as current, and must not
  trigger a rebuild on its own.
- **FR-7** — Refreshing/building a CPG remains a deliberate, on-demand action, unchanged
  from the existing model — this feature increases how often that action gets invoked
  in practice, not the model itself.
- **FR-8** — This feature must not change the CPG read path (`mcp__cpg__query`, its
  parameters, or the `redis-cli` fallback) established by `cpg-query-access.md` (M3).
- **FR-9** — Whatever downstream document/plan implements this must reconcile
  explicitly with the consumer-scope boundary set at M2
  (`docs/plans/m2-cpg-analysis-skill.md`) and M3 (`docs/requirements/cpg-query-access.md`),
  recording this as an **extension** of that scope, not a silent divergence from it.

## Out of scope
- **MCP tool / access path changes** — `mcp__cpg__query`'s shape, parameters, or the
  `redis-cli` fallback. Settled at M3; untouched here.
- **Automatic/unattended CPG rebuild.** Refresh stays a suggested, deliberately
  triggered action (FR-6) — never silent, never agent-initiated on its own.
- **Proactive, wholesale CPG build-out** across every component. Coverage grows only
  through the existing on-demand model, just exercised more often (FR-7).
- **Recurring or automated usage-tracking / dashboards.** Verification stays manual
  spot-checking (a stakeholder reviewing a transcript), not a tracking system.
- **Authentication / access-control changes to FalkorDB.** Already out of scope per
  `cpg-query-access.md`; unchanged here.
- **The actual design** — which agents qualify, prompt/skill/hook wording, how
  discovery and staleness signaling are implemented. That's the downstream design
  pass's job (`cobb` for agent/skill wiring, `graph-dba` for freshness mechanics), not
  this document's.

## Acceptance criteria
- **AC-1** — Given a code-level-work agent beyond the original three (exact roster
  decided at design time) starting a task that touches a component with a loaded CPG,
  when it orients on the task, then it discovers the CPG's existence without being
  told, and consults it when relevant to the task.
- **AC-2** — Given a stakeholder spot-checks a task's transcript for work that
  plausibly touched CPG-covered code, when they look for evidence, then they can see
  either that the agent considered/used the CPG, or an explicit, reasoned "not
  relevant here."
- **AC-3** — Given an agent consults a loaded CPG, when it reports back or acts on the
  results, then it also communicates some signal of how current the graph is.
- **AC-4** — Given that signal indicates the graph may be stale, when the agent
  proceeds, then it surfaces a suggestion to refresh — not a silent rebuild, not
  silent continued use as if current.
- **AC-5** — Given a component with no loaded CPG, when an agent's default discovery
  check runs, then the task proceeds with no material delay or noise attributable to
  that check.
- **AC-6** — Given the downstream plan for this feature, when read against
  `m2-cpg-analysis-skill.md` and `cpg-query-access.md`, then it states explicitly that
  it *extends* — not silently overrides — the consumer-scope boundary those documents
  set.

## Open questions
None outstanding. The exact agent roster (FR-1/AC-1) and the discovery/staleness
mechanism (FR-2, FR-5, FR-6) are deliberately left to the downstream design pass —
not open questions here, but explicit deferrals the stakeholder confirmed.

## Decision log
- 2026-08-15 — Session opened from a Mode-2 explanation ("why aren't the Claude agents
  benefiting from the CPG, how can we improve them") that surfaced enough of a gap to
  warrant a requirements interview. Stakeholder chose to proceed with the interview
  (Mode 1) over going straight to a `cobb` design pass.
- 2026-08-15 — What triggered this now? → **All three** surfaced reasons apply at once:
  a specific review that took longer than it should have, a bug impact-analysis
  would plausibly have caught, and the standing discomfort of an expensive capability
  sitting dormant. Stakeholder did not single one out as primary.
- 2026-08-15 — What does "solved" look like? → **All four** candidate outcomes matter:
  (1) more agents consult the CPG beyond the current three, (2) the already-wired
  agents actually reach for it in real tasks (not just carry the wiring), (3) staleness
  is knowable/managed, (4) coverage extends past `falkor-chat`/`salesperson`. This is a
  broad feature — the FR list will need to cover all four without conflating them.
- 2026-08-15 — Lever 1 (more agents consult it): which agents? → **Not a fixed list** —
  "any agent doing code-level work" is the target, not a named set like
  `coder`/`tdd-engineer` specifically. The design pass downstream of this doc decides
  exactly who qualifies; this doc records the need (a code-level-work agent should be
  able to reach the CPG when one's relevant to its task), not the roster.
- 2026-08-15 — Lever 2 (used, not just wired): how would you verify it's working? →
  **Spot-checking is enough for now** — no recurring/automated usage metric required.
  Being able to look at a task transcript that plausibly touched CPG-covered code and
  see the agent considered/used the graph is sufficient evidence.
- 2026-08-15 — Lever 3 (trustworthy/freshness): marker vs. auto-refresh? → **Show +
  suggest, not automatic.** The consuming agent should be able to tell how stale the
  loaded graph might be (a marker of some kind) and, if it looks stale, surface a
  suggestion to refresh it — not silently trust it, and not silently auto-rebuild
  either. Rebuilding stays a deliberate, on-demand action (consistent with
  `graph-dba`'s existing "rare, not proactive" CPG-generation stance).
- 2026-08-15 — Lever 4 (coverage): every component, or the busy ones? Proactive build-out
  or on-demand? → **On-demand stays the model** — no wholesale build-out of CPGs for
  every component. The change is that on-demand gets *invoked more often* in practice
  (more agents actually asking `graph-dba` to build one when it'd help), not that the
  model itself changes to proactive.
- 2026-08-15 — Discovery trigger: default/unprompted step, or reminder-driven? →
  **Default step.** A code-level-work agent should check whether a relevant CPG exists
  as part of its normal orientation (like reading `AGENTS.md` today), without needing
  to be reminded. This is the mechanism that actually closes lever 2.
- 2026-08-15 — Overhead of a default check coming up empty (most components have no
  CPG today) — does that matter? → **No, acceptable cost.** A quick no-op check on
  components without a CPG is fine; not a constraint to design around.
- 2026-08-15 — Out-of-scope confirmation → **All four confirmed out**: MCP tool/access
  path changes, automatic rebuild, proactive full-repo build-out, and any
  usage-tracking dashboard.
