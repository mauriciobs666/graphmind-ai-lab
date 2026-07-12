# Kaizen — Change History: coder

> Dated log of actual changes to the `coder` agent. Most recent first.

## 2026-07-12 — Learning-capture loop: kaizen inbox + closing protocol
- **What:** Added `kaizen/inbox.md` (append-only learnings inbox, seeded empty) and a "Learning capture" closing-protocol section to the prompt: durable, non-obvious environment facts discovered during runs are appended as dated, evidence-backed inbox entries; the agent never promotes its own entries.
- **Why:** Team-wide self-improvement loop (agent-maintenance skill §5, added the same day): capture is cheap and unreviewed during runs, promotion is curated — cobb periodically verifies each entry and routes it to the prompt, an on-demand knowledge base, or project docs. Requested by the user.
- **Plan items:** none.

## 2026-07-11 — Description slimmed (team-wide token-cost pass)
- **What:** Frontmatter `description` compressed from 822 to 535 chars: capability lists tightened, reciprocal boundary prose reduced to short route-away clauses that still name the counterpart agents (audit check 6 boundary symmetry preserved — full pass green), and "how I work" detail dropped from the description since the prompt body already carries it. Routing semantics unchanged; no body/catalog changes needed.
- **Why:** All 12 agents' descriptions are auto-injected into every session and into every subagent spawn that carries the `Agent` tool; team-wide they cost 12,609 chars (~3.1K tokens) per injection. The pass cut them to 7,036 chars (~44%), saving ≈1,400 tokens per session/spawn with the same routing contract.
- **Plan items:** none.

## 2026-07-09 — Description: route-away clause to the new `frontend-engineer`
- **What:** the `description`'s routing tail gained a second route-away rule: UI-heavy front-end work (components, styling, accessibility, client-side state, a Streamlit screen) → `frontend-engineer`. Catalog rows (`claude/AGENTS.md`, `claude/README.md`, root `AGENTS.md`) updated to match; the pair was added to `scripts/audit-team.sh` `BOUNDARY_PAIRS`.
- **Why:** a UI-depth specialist implementer (`frontend-engineer`) joined the team; boundary reciprocity requires the adjacent generalist implementer to name it so routers see the contract from both sides.
- **Plan items:** none (driven by frontend-engineer's creation).

## 2026-07-09 — K-001 ✅: efficiency-based routing boundary with `tdd-engineer` (de-personalized)
- **What:** Rewrote the `description`'s routing tail. Was "for strict test-first discipline, prefer tdd-engineer" — a subjective tiebreaker; now routes by **task shape / efficiency**: a detailed plan/spec ready to execute → `coder` (tests alongside); a bug fix, safety-net refactor, test-focused work, or clear-contract feature → `tdd-engineer`. Made symmetric: `tdd-engineer`'s description (which previously shadowed coder's trigger — "whenever the user asks to implement a feature" — and never pointed back) now carries the mirror rule. Synced everywhere the rule is repeated: teco's roster (the "(this user prefers TDD — lean toward tdd-engineer)" note removed), `claude/AGENTS.md`, `claude/README.md`, root `AGENTS.md`, and `cobb/TESTING.md`'s rationale column.
- **Why:** User ruling on the overlap review: use `tdd-engineer` only where test-first is genuinely the efficient path, and when a detailed plan already exists the most efficient implementer wins — plus, personal-preference notes ("this user prefers TDD") don't belong in agent prompts; the user's standing preferences are **quality and efficiency**, encoded as objective routing rules.
- **Plan items:** K-001 ✅ (closed — the descriptions no longer collide; the tiebreaker is objective at routing time).

## 2026-07-09 — Subagent-awareness on the two "ask" spots (teco interface review follow-up)
- **What:** Step 2 (baseline) "ask before installing or mutating the environment" and the "Ask before destructive or environment-changing actions" guardrail now both say what to do when running as a subagent (e.g. delegated by `teco`): return the blocker / request as the result instead of trying to ask mid-run — subagents can't ask. Catalog entry (`claude/AGENTS.md`) updated.
- **Why:** Sweep after the 2026-07-09 teco interface review found the "ask" phrasing assumed an interactive session across several delegates (same fix applied to tdd-engineer, qa-engineer, graph-dba the same day; architect already handled it via questions-as-deliverable).
- **Plan items:** none (out-of-band, driven by teco's 2026-07-09 review).

## 2026-07-08 — Architect handoff arrives as a plan-document path
- **What:** Step 1 (Orient) updated: an `architect` handoff is now a plan **document** at `<component>/docs/plans/<slug>.md` — the coder gets the path in its brief, reads the file itself, and treats it as source of truth (gaps filled by reading code, not guessing). Synced with the architect's same-day change making the plan doc its default deliverable and teco's switch to path-based handoff.
- **Why:** The previous flow had the orchestrator paste the plan into the brief — lossy and unreviewable. Reading the artifact directly makes the handoff lossless regardless of who invokes the coder.
- **Plan items:** advances K-002 (transport fixed; live validation run still pending).

## 2026-06-20 — Dropped "senior" framing
- **What:** Removed "senior" from the `description` ("Senior software engineer" → "Software engineer") and the body opener ("You are a senior software engineer who builds" → "You are a software engineer who builds"). Mirrored in the catalog entries (`claude/README.md`, `claude/CLAUDE.md`, root `AGENTS.md`).
- **Why:** User raised the overconfidence concern with seniority framing; persona-prompting evidence (e.g. Zheng et al. 2024) shows role labels are weak-to-neutral for correctness while authority framing can hurt calibration. Quality is carried by the concrete process + guardrails ("don't fake green," "report only what you ran"), not the title. Chose the most conservative option (drop the word). Goes further than the 2026-06-05 precedent that kept "Senior" as an altitude signal — architect/coder now differ from the rest of the collection until harmonized (flagged to user).
- **Plan items:** —

## 2026-06-20 — Created
- **What:** Created the `coder` subagent (`coder/coder.md`, `model: opus`). Senior implementer that executes an approved plan/spec end-to-end: orients on the plan + code, establishes a green baseline (with explicit handling for already-red and can't-run-here cases), implements in small reversible increments, tests alongside, refactors under green, and reports only results it actually ran. Inherits all tools (no `tools` key), following the `tdd-engineer` K-003 precedent of keeping the implementer flexible.
- **Why:** User asked for two complementary Claude Code subagents, "the architect" and "the coder," with an architect→coder handoff. The `coder` is the implementation half.
- **Plan items:** seeded K-001..K-002.

## Decisions recorded at creation
- **Distinction from `tdd-engineer`:** both implement, but `tdd-engineer` is *strictly* test-first (red→green→refactor as the defining discipline). `coder` is plan-driven and pragmatic: it tests behavior thoroughly and never ships untested code, but doesn't mandate writing the failing test first unless the project requires it. The `description` explicitly routes strict-TDD requests to `tdd-engineer` so auto-delegation doesn't collide. Revisit if the two over-trigger on the same prompts.
- **Why inherit all tools:** an implementer needs Read/Write/Edit/Bash plus the ability to fetch docs and delegate; mirrors the deliberate `tdd-engineer` choice (K-003 there). Revisit only if broad access causes surprise.
