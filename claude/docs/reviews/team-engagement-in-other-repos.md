# Review — Team engagement in other repos (requirements)

> **Status:** active · **Owner:** `analyst` · **Tracks:** — · **Last updated:** 2026-11-03

## Pass 1 — 2026-11-03

**Scope:** `claude/docs/requirements/team-engagement-in-other-repos.md` (untracked, `Status: Interviewing`), pre-flip gate to *Ready for design*. Baselines: root `REVIEW.md` (growth-stops check; no `claude/REVIEW.md` exists), root `AGENTS.md` header and status rules, prior decisions in `claude/docs/requirements/`, `docs/requirements/`, `claude/*/kaizen/history.md`.
CPG: considered, not relevant — requirements document; the claims checked are about config files and agent prompts, not code structure.

**Verdict: approve with suggestions.** No blocker. The two majors are about testability and are cheap to fix before the flip. The architect can design from the document as it stands, but the flip should wait for them or carry them as explicit design inputs.

### Premise verification (Problem & current state)

| Claim | Result |
|---|---|
| `~/.claude/agents` symlinks to this repo's `claude/` | Verified: `ls -l` shows `agents -> /home/mauricio/prg/graphmind-ai-lab/claude`. |
| Skills live in repo-root `skills/` | Verified. `~/.claude/skills/` holds only the platform's `synced/` bundles, none of the team's skills. This supports FR-6 being a real gap. |
| Agents name `mcp__falkor-chat-agent-team__ingest_document` | Verified: all 13 `claude/*/*.md` agent files. |
| `ws:agent-team` pinned by `start_agent_team.sh` | Verified (`FALKORCHAT_WS_ID=agent-team`, port 8200). |
| "The `cypher` MCP wiring stays in this repo" | Verified, and stronger than stated. Both `cypher` and `falkor-chat-agent-team` are wired only in repo-root `.mcp.json`; `~/.claude.json` has no user-level `mcpServers`. In another repo neither the capture tool nor the cypher tool exists today. This is the real substance of FR-5/FR-2b and it is stated only implicitly. |

### Findings

**M1 (major, verified) — FR-2b, FR-3 and FR-1's ordering clause have no acceptance criterion, and "on demand" has no observable trigger.** The ACs cover FR-1, 2, 2a, 4, 5, 5a and 6 only. FR-2b ("added on demand, not at kickoff") and FR-3 ("can grow later without re-scaffolding") are untestable as written: no AC says who or what requests the addition, or what "without re-scaffolding" means observably. AC-1 checks the end state, so it cannot show that "no scaffolding precedes" the interview (FR-1).
Suggestion: add an AC for each. For FR-2b: after kickoff the repo contains no `REVIEW.md` and no cypher/graph wiring, and a later stakeholder request or review need adds one without touching existing files. For FR-3: adding a convention leaves the existing `AGENTS.md` and docs files unchanged except for appended content. For FR-1: the transcript or git history shows no file written before the requirements document. Keep each in the Given/When/Then form of AC-1 to AC-4.

**M2 (major, verified) — AC-1 is internally ambiguous.** It requires "a project rules file plus docs layout, and nothing else scaffolded" at the end of a `tico` session, but `tico` is today a requirements and manuals agent (`claude/tico/tico.md`) with no scaffolding remit. The AC does not say what "docs layout" contains (which directories, the status-header block, a `HISTORY.md`/`BACKLOG.md`?). "Nothing else" cannot be checked against an undefined "something". The FR-2 wording ("`AGENTS.md`: what the project is, how to run and test it") also creates a tension: the test-run instruction must be written before the test setup exists (FR-2a defers it to first build).
Suggestion: enumerate the kickoff set in FR-2 or AC-1 (which directories are created, whether the status-header convention is exhibited by the requirements doc itself). State what the `AGENTS.md` "how to run and test" section says before FR-2a runs (for example "to be established at first build"). Who scaffolds stays a design choice, but the AC should name the observable outcome only.

**m1 (minor, inferred) — AC-3 and AC-4 depend on infrastructure the document never states as a requirement.** AC-3 requires the shared pool to be reachable and writable from the new repo; AC-4 requires skills to be discoverable. The "Context for design" note hints at this. The document should say in one sentence that the new project must not need the stakeholder to hand-configure either the pool connection or skill access (AC-4 already says "no setup of its own"; AC-3 does not). Otherwise AC-3 could be satisfied by manual wiring done once per project, which contradicts FR-6's spirit.

**m2 (minor, inferred) — Project-specific content entering the shared pool.** FR-5a separates project-specific from general learnings for `cobb`, but nothing says whether a new project's raw content (code, names, possibly private information) is acceptable in a pool that is global across projects. A stakeholder decision or an explicit "accepted" note is cheap now and expensive after the architect has designed around it. This is not an open-questions item only if the stakeholder already decided it; the decision log shows only "Yes, shared". Raise it with the stakeholder at the flip readback or record it as accepted.

**m3 (minor, verified) — Decision log entries 3 and 5 carry "FRs TBD" and "scope of FRs TBD".** The log is append-only, so these are not edited, but later entries do resolve them. A reader who stops at the log sees unresolved placeholders. Add one closing log line (question → answer) stating that entries 1 to 3 are covered by FR-1 to FR-6; do not rewrite the earlier lines.

**n1 (nit) — FR-4 ("as it does today") and AC-2's "pausing only at genuine decision points" are loose.** FR-4 is verifiable only by comparison with an unstated baseline. AC-2 is the real test and is acceptable; naming the specialist handoffs (requirements to plan to review to build) would sharpen it. Optional.

### Checks that came back clean

- **Solution language.** FRs are phrased as outcomes. `AGENTS.md`/`REVIEW.md`/cypher MCP are existing convention names, not design. FR-5a defers mechanism to design.
- **FR to AC mapping.** Every cited reference is correct (AC-1 cites FR-1/FR-2; AC-2 cites FR-2a/FR-4; AC-3 cites FR-5/FR-5a; AC-4 cites FR-6); only FR-2b, FR-3 and FR-1's ordering are uncovered (M1).
- **Prior decisions.** Grepped `claude/docs/requirements/`, `docs/requirements/` and every `claude/*/kaizen/history.md` for portability, user-scoped, other-projects, green-field and shared-pool decisions. The only precedent is `devops` being made portable and user-scoped (`claude/devops/kaizen/history.md`, 2026-xx portable-agent entry), which this document extends rather than reverses. The shared `ws:agent-team` pool is the settled single capture store (`claude/AGENTS.md`); FR-5 is consistent. Nothing reverses a recorded decision. Skills location (repo-root `skills/`) is not changed by this document, which only asks that they be reachable.
- **Open questions "None".** Legitimate for the stakeholder-facing scope, subject to m2.
- **"Context for design" note.** Stays design-neutral: it names the problems (reaching the pool and skills without editing every agent, attribution) and prescribes no mechanism.
- **Growth-stops check.** The document is new (first revision), so all of its content is new scope; no rework growth, restated history or restated facts found. No prior review exists. The "Problem & current state" paragraph restates no fact owned elsewhere beyond citing it.
- **Header block.** Matches the root `AGENTS.md` convention (`Status: Interviewing`, `Owner`, `Tracks`, `Last updated`).
- **Out of scope** (legacy projects, other stacks) is explicit and consistent with the decision log.

### What's solid

Tight, outcome-phrased FRs; a clean decision log with FR/AC references; correct and verified premises; legacy scope deferred cleanly; "lighter by default" settled as a decision rather than left implicit.

### Open questions for the caller

1. Is it acceptable that a new project's raw learnings (possibly private content) land in the global shared pool? (m2)
2. Flip now with M1/M2 carried as design inputs, or fold the AC additions in first? Recommended: fold in first; it is a few lines.

### Proposed REVIEW.md lines

None.
