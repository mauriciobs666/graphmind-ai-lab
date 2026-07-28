# Kaizen — Change History: frontend-engineer

> Dated log of actual changes to the `frontend-engineer` agent. Most recent first.

## 2026-07-27 — Unpinned from `model: opus` (team-wide)
- **What:** Removed the `model: opus` frontmatter line. The field is now absent, so the agent runs on Claude Code's default — `model` **defaults to `inherit`** (re-verified 2026-07-27 against `code.claude.com/docs/en/sub-agents`), i.e. the model the session/system default selects. No other frontmatter or body change.
- **Why:** User no longer wants the team locked to Opus. Model choice belongs at the session level (one decision, changeable with `/model`), not duplicated across 13 frontmatter files where it silently overrides whatever the user picked.
- **Plan items:** —

## 2026-07-24 — Description slimmed further (second team-wide token-cost pass)
- **What:** Frontmatter `description` compressed 682 → 574 chars (-15%): tightened phrasing, dropped restated detail, kept every routing/boundary clause. `claude/scripts/audit-team.sh` boundary-pair symmetry (frontend-engineer↔coder) re-verified green. No body/catalog change.
- **Why:** All 13 agents' descriptions are auto-injected into every session and subagent spawn; the roster grew to 13 (graph-dba, joern added) since the first pass on 2026-07-11, and per-agent `/context` output showed room to cut further. User-requested via a `/context` token audit.
- **Plan items:** none.

## 2026-07-24 — Frontmatter: `permissionMode: acceptEdits`
- **What:** Added `permissionMode: acceptEdits` to the frontmatter, matching the same-day change to `coder`. File-edit/write approvals are session-scoped in Claude Code (unlike Bash approvals, which persist permanently per repo+command), so users otherwise have to re-grant write permission on every session even with a global `Edit`/`Write` allow rule in `~/.claude/settings.json`. `acceptEdits` auto-accepts file edits and common filesystem commands for paths in the working directory/`additionalDirectories`, independent of session-level grants.
- **Why:** Same root cause as `coder` (see its 2026-07-24 kaizen entry) — applied to the other implementer agents for consistency, at user request.
- **Plan items:** none.

## 2026-07-12 — Learning-capture loop: kaizen inbox + closing protocol
- **What:** Added `kaizen/inbox.md` (append-only learnings inbox, seeded empty) and a "Learning capture" closing-protocol section to the prompt: durable, non-obvious environment facts discovered during runs are appended as dated, evidence-backed inbox entries; the agent never promotes its own entries.
- **Why:** Team-wide self-improvement loop (agent-maintenance skill §5, added the same day): capture is cheap and unreviewed during runs, promotion is curated — cobb periodically verifies each entry and routes it to the prompt, an on-demand knowledge base, or project docs. Requested by the user.
- **Plan items:** none.

## 2026-07-11 — Description slimmed (team-wide token-cost pass)
- **What:** Frontmatter `description` compressed from 1156 to 678 chars: capability lists tightened, reciprocal boundary prose reduced to short route-away clauses that still name the counterpart agents (audit check 6 boundary symmetry preserved — full pass green), and "how I work" detail dropped from the description since the prompt body already carries it. Routing semantics unchanged; no body/catalog changes needed.
- **Why:** All 12 agents' descriptions are auto-injected into every session and into every subagent spawn that carries the `Agent` tool; team-wide they cost 12,609 chars (~3.1K tokens) per injection. The pass cut them to 7,036 chars (~44%), saving ≈1,400 tokens per session/spawn with the same routing contract.
- **Plan items:** none.

## 2026-07-09 — Created
- **What:** initial version of the agent — front-end specialist implementer: web platform (semantic HTML, modern CSS, JS/TS, React & peers), accessibility, responsive layout, state/data-flow design, front-end performance, front-end testing, plus Streamlit/Python-UI fluency. Orient-first discipline (never assumes a stack), plan-by-path handoff from architect, subagent-aware, `model: opus`, inherits all tools (implementer — no write-scope hook).
- **Why:** the team had generalist implementers (coder, tdd-engineer) but no UI-depth specialist; front-end work (components, styling, a11y, performance, future falkor-chat UI) deserved the same specialist treatment graph-dba gives the data layer.
- **Wiring:** added to teco's routing table + description roster, all three catalogs (`claude/AGENTS.md`, `claude/README.md`, root `AGENTS.md`), symlinked into `~/.claude/agents/`, and paired with `coder` in `scripts/audit-team.sh` `BOUNDARY_PAIRS` (coder's description gained the reciprocal route-away clause).
- **Plan items:** seeded K-001 (shakedown run), K-002 (visual verification tooling).
