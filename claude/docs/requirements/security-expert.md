# Security expert (new agent) — Feature Requirements
> **Status:** Ready for design · **Owner:** `tico` · **Tracks:** — · **Last updated:** 2026-08-17

## Intent
Introduce a dedicated security-expert team member to close two gaps the current roster doesn't
cover:

1. **Code/app security review**, deeper than the security/perf step already folded into
   `analyst`'s general code review.
2. **Agent/prompt-safety review** — judging whether an agent's own artifacts (prompts, skills,
   `kaizen/inbox.md` entries, plans) are safe to keep or promote. No agent currently owns this
   judgment: on 2026-07-31, `analyst` flagged a kaizen inbox entry as instruction-poisoning-shaped,
   and `teco` (no adjudication authority of its own) had to route the call to `cobb` ad hoc.

## Problem & current state
- `analyst` reviews code correctness → tests → convention fit → clarity → **security/perf**, in
  that priority order — security is one checklist item among several, not a deep pass.
- `devops` owns secrets hygiene and infra hardening, but not application-level vulnerability
  analysis or prompt/agent-safety judgment.
- Nobody owns agent/prompt-safety adjudication as a standing responsibility — the 2026-07-31
  incident (`claude/analyst/kaizen/history.md`, `claude/cobb/kaizen/history.md`) was handled
  one-off, outside any agent's stated remit.

## User stories
- As a stakeholder, I want a dedicated deep-dive on code-level security issues, so that
  vulnerability-shaped problems don't ride along as a single checklist line in a general review.
- As a stakeholder, I want a standing, named owner for judging whether an agent's own artifacts
  (prompts, skills, kaizen entries, plans) are safe, so incidents like 2026-07-31 have a clear
  home instead of ad hoc routing to whichever agent happens to be available.
- As a stakeholder, I want the code-security review to reuse the project's existing Code
  Property Graph capability when one exists for the component under review, so the analysis can
  answer data-flow/injection questions instead of only reading files by hand.
- As a stakeholder, I want secrets/infra-hardening and compliance-style checklist audits
  available from the same expert, so I have one place to ask "are we exposed here?" across code,
  agents, and infra.
- As a stakeholder, I want the option of an active exploitation attempt against my own local/dev
  instances, gated behind explicit approval, so I can get real proof a vulnerability is
  exploitable, not just a theoretical finding — without that capability ever running
  unsupervised or against anything outside this lab.

## Functional requirements
- **FR-1.** The security expert can perform a deep code/app security review of any component in
  the repo, on demand — going beyond `analyst`'s existing security/perf checklist item
  (vulnerability patterns, dependency/CVE risk, injection paths, unsafe deserialization,
  secrets-in-code, and similar).
- **FR-2.** When a Code Property Graph exists for the component under review, the code-security
  review can use it to answer code-security questions (e.g. data-flow/injection paths) instead of
  reading files by hand — the same pattern other agents in this team already follow.
- **FR-3.** The security expert can review, on demand: kaizen inbox entries before promotion,
  agent/skill prompt definitions, and plans/requirements docs, for agent/prompt-safety concerns
  (e.g. instruction-poisoning-shaped writing, unsafe framing).
- **FR-4.** The security expert can review, on demand, secrets/infra-hardening concerns across the
  repo's components.
- **FR-5.** The security expert can run a structured, checklist-based security/compliance audit on
  demand. No external framework (e.g. OWASP) is mandated by this requirement — the checklist's
  actual content is a design decision downstream.
- **FR-6.** `analyst`'s existing security/perf checklist step is unchanged and keeps running on
  every code review it already covers. The security expert's code-security review (FR-1) is an
  additional, deeper pass — not a replacement — invoked separately.
- **FR-7.** None of the security expert's review capabilities (code-security, agent/prompt-safety,
  secrets/infra, compliance) run automatically as a standing gate on any existing workflow (code
  review, kaizen distillation, agent/skill authoring, or otherwise). Every review is invoked
  explicitly.
- **FR-8.** On the agent/prompt-safety side, the security expert's findings are advisory: `cobb`
  retains final decision authority over what gets promoted into a prompt/skill/doc or shipped. The
  security expert does not have unilateral veto power.
- **FR-9.** On the secrets/infra-hardening side, the security expert's findings are advisory:
  `devops` retains ownership of actually changing infra/secrets configuration. The security expert
  does not make infra changes itself.
- **FR-10.** The security expert can attempt active exploitation of a running system, but only
  against this lab's own local/dev instances (never external, shared, or production systems), and
  only after explicit human approval is given for that specific attempt — consistent with how this
  team already gates destructive/irreversible actions elsewhere (`devops`/`graph-dba`/
  `qa-engineer`'s destructive-ops gates). Every other review capability (FR-1 through FR-5) is
  static and non-destructive, matching how the rest of this team's reviewers work.
- **FR-11.** Every review the security expert performs produces a written, durable findings
  report — what was checked, findings ranked by severity, and a verdict/summary — so the
  stakeholder (or `cobb`/`devops`) can read exactly what was found without re-asking the agent.

## Out of scope
- Automatically gating any existing workflow (code review, kaizen-inbox promotion, agent/skill
  authoring, infra changes) — every review is invoked explicitly, not triggered by default (FR-7).
- Replacing `analyst`'s existing security/perf checklist step, `cobb`'s authority over agent/skill
  promotion, or `devops`'s authority over infra/secrets changes — the security expert is advisory
  in all three relationships (FR-6, FR-8, FR-9).
- Active exploitation against anything other than this lab's own local/dev instances — no
  external, shared, third-party, or production targets, ever (FR-10).
- Active exploitation running without a fresh, explicit approval for that specific attempt — no
  "approve once, run repeatedly" mode (FR-10).
- A named/mandated external compliance framework (e.g. a formal OWASP Top 10 sign-off process) —
  the checklist capability exists, but no specific standard is required by this feature (FR-5).
- Runtime/production security monitoring or incident response — this agent reviews and
  (within FR-10's limits) tests; it does not watch a running system on an ongoing basis.

## Acceptance criteria
- Given a code change or component the stakeholder wants reviewed for security, when the
  security expert is invoked, then it produces a written findings report distinct from and more
  detailed than `analyst`'s existing security/perf checklist line, and `analyst`'s own review is
  unaffected.
- Given a component with an existing Code Property Graph, when the security expert reviews it,
  then its findings report shows evidence of graph-based analysis (e.g. a traced data-flow/
  injection path) rather than file-reading alone.
- Given a kaizen inbox entry, agent/skill prompt, or plan/requirements doc the stakeholder or
  `cobb` wants checked, when the security expert reviews it, then it produces a written opinion
  that `cobb` can act on — and `cobb`'s decision, not the security expert's, is what determines
  whether the material is promoted or shipped.
- Given a secrets/infra-hardening question, when the security expert reviews it, then it produces
  a written opinion that `devops` can act on, and no infra/secrets configuration is changed by the
  security expert itself.
- Given no explicit invocation, when any other workflow in the repo runs (a code review, a kaizen
  distillation pass, agent/skill authoring, an infra change), then the security expert's review
  does not run automatically as part of it.
- Given a request for active exploitation, when the target is anything other than this lab's own
  local/dev instance, then the security expert declines; when the target is a local/dev instance,
  then it proceeds only after a fresh explicit approval for that specific attempt.

## Open questions
None — every open question from the interview was resolved and logged below.

## Decision log
2026-08-17 — What's the intent? → Introduce a security-expert agent covering both (1) deeper
code/app security review than `analyst`'s current security/perf checklist step, and (2)
agent/prompt-safety review. For the code-security side, the stakeholder wants the agent able to
use the project's existing Code Property Graph (the `cpg` MCP tool / `cpg-analysis` skill pattern
other agents already follow) when one exists for the component under review — noted here as a
stated preference for reusing an existing project capability, not a new design decision.
2026-08-17 — What triggered this now? → No specific incident; proactive risk reduction ("ensure
we won't have any problems") rather than a reaction to a known close call. This is about standing
coverage, not patching one discovered hole.
2026-08-17 — Does the security expert replace or layer on top of `analyst`'s existing
security/perf checklist step? → **Layer on top.** `analyst` keeps its lightweight security/perf
check on every review as the first line of defense; the security expert is a separate, deeper
pass invoked when security is the actual concern.
2026-08-17 — Which components does the code-security side cover? → **All of them**, whatever's
under review at the time — no component excluded by default (not scoped down to
untrusted-input surfaces only).
2026-08-17 — What does the agent/prompt-safety side review? → **All three**: kaizen inbox
entries before promotion (the 2026-07-31 scenario), agent/skill prompt definitions themselves,
and plans/requirements docs for security implications before they're acted on.
2026-08-17 — Who has final say when the security expert flags agent/prompt material as unsafe?
→ **`cobb` decides; the security expert advises.** Same shape as the `analyst` boundary —
`cobb` keeps ownership of what actually gets promoted/shipped (kaizen distillation, agent/skill
authoring); the security expert supplies a dedicated security opinion that feeds into `cobb`'s
call, replacing ad hoc routing (as happened 2026-07-31) with a named, standing source of that
opinion. Not a veto.
2026-08-17 — Invocation mode for both sides → **On demand only, neither side is a standing gate.**
Nothing changes automatically before shipping/promotion; the security expert is invoked
explicitly (by the stakeholder, `teco`, `cobb`, or another agent) when a security opinion is
wanted.
2026-08-17 — Scope expansion beyond the original two areas → stakeholder wants **all** of:
vulnerability/dependency review (core of code/app security), secrets/infra hardening, and
compliance/audit checklists, in addition to code/app security review and agent/prompt-safety
review.
2026-08-17 — Boundary for secrets/infra hardening → same advisory shape as `analyst`/`cobb`:
**security expert advises, `devops` decides** and keeps ownership of actually changing
infra/secrets config.
2026-08-17 — What does "compliance/audit checklists" mean? → **No specific framework named** —
a general structured-checklist-audit capability, not tied to OWASP or any other named external
standard right now.
2026-08-17 — Should active/offensive security testing (actually attempting to exploit a running
system) be in scope? → **Yes** — a departure from every other reviewer on this team (`analyst`
included), which is strictly static/non-destructive.
2026-08-17 — Where can active exploitation target? → **This lab's own local/dev instances only**
(e.g. the local FalkorDB container, a locally-run `salesperson`/`falkor-chat` instance) — never
external, shared, or production systems.
2026-08-17 — Does active exploitation require approval before running? → **Yes, explicit
approval each time** — consistent with how this team already gates destructive/irreversible
actions (`devops`/`graph-dba`/`qa-engineer`'s destructive-ops hooks); a human confirms before
the security expert actually attempts exploitation, even though it was invoked for that purpose.
2026-08-17 — What confirms the security expert did its job on a given review? → **A written
findings report per review** — same pattern as `analyst`'s reviews: what was checked, findings
ranked by severity, a verdict/summary, saved as a durable artifact rather than only a
conversational answer.
