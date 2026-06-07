# Kaizen — Change History: cobb

> Dated log of actual changes to the `cobb` agent. Most recent first.

## 2026-06-07 — Added "Drift-resistance" principle + cross-tool subagent comparison; opened K-005 (automated drift check)
- **What:** Added a "Drift-resistance" bullet to the prompt's Principles: keep stable mental models + canonical URLs in the always-on prompt, treat field lists / "who-loads-what" tables / feature availability as perishable (stamp `verified DATE against <url>`, prefer live-verify or an updatable skill), and don't assume one tool's behavior transfers. Verified the cross-tool picture for subagent context-loading: **Claude Code** custom subagents auto-load the `CLAUDE.md` hierarchy; **OpenCode** docs are silent on whether subagents get `AGENTS.md` (subagents inherit the invoking primary's model); **Kiro** has subagents since 0.9 and its docs claim `inclusion: always` steering reaches them, but open issues (#7131, #7758) dispute that in practice — Specs/Hooks definitely don't reach Kiro subagents. Opened **K-005** to automate doc-drift detection (scheduled doc-diff → kaizen item).
- **Why:** User asked whether the subagent CLAUDE.md behavior is Claude-specific or portable, and "how can we ensure the info will not drift?" The answer is that it's tool-specific and in flux — which makes drift-resistance a first-class principle, not just K-001's manual re-verify.
- **Plan items:** K-005 (added); reinforces K-001.

## 2026-06-07 — Sharpened subagent context-loading knowledge (verified against official docs)
- **What:** Expanded the Claude Code "Subagents" bullet in the prompt with facts I verified live at code.claude.com/docs/en/sub-agents: (1) custom subagents **do** auto-load the full `CLAUDE.md`/memory hierarchy via message flow even though the body replaces the default system prompt, and `@`-imports expand into them — built-in **Explore/Plan** are the only ones that skip `CLAUDE.md`+git (not configurable); a **fork** inherits the whole parent conversation instead; (2) subagents don't see parent conversation history/tool results/skills — must be passed in the delegation prompt; (3) added the `memory:` frontmatter field (persistent `agent-memory/<name>/` store, distinct from `CLAUDE.md`) and noted other current frontmatter fields (`disallowedTools`, `permissionMode`, `skills`, `isolation`, `effort`, `inherit`) with a verify-against-docs caveat.
- **Why:** A user question ("do agents like you autoload CLAUDE.md?") exposed a real gap in an area the prompt claims to know cold: it described subagent context only as "isolated," implying CLAUDE.md might not load, and omitted the `memory:` field entirely. Fixed because broken/stale Claude Code specifics make me produce wrong artifacts — the core risk K-001 guards against.
- **Plan items:** advances K-001 (re-verified subagent docs; baseline date refreshed to 2026-06-07).

## 2026-06-07 — Learned the DRY import pattern + drift-audit method (from syncing graphmind-ai-lab's AGENTS.md)
- **What:** While bringing the repo's stale root `AGENTS.md` back in sync (it was missing the entire `falkor-chat/` component, the `graph-dba` agent, and the `severino` OpenCode agent), recommended and created a root `CLAUDE.md` containing just `@AGENTS.md`. Folded that single-source-of-truth rule into the prompt's "Documentation" section ("Don't duplicate the same catalog into two files"). Opened K-004 to capture the standalone audit-&-reconcile method (drift detection via `git ls-files` vs. the doc's claims), flagged as skill material to keep the prompt lean.
- **Why:** Two durable learnings surfaced from the session: (1) when `CLAUDE.md` and `AGENTS.md` would carry the same catalog, importing avoids divergence and keeps `AGENTS.md` as the broadest-reach source; (2) reconciling an already-drifted context doc is a recurring task distinct from the "sync on my own edits" duty already in the prompt.
- **Plan items:** K-004 (added).

## 2026-05-31 — Redacted absolute paths from eval reports (privacy)
- **What:** Fixed a leak where `baseline/01-explain-python.md` contained the full `/home/<user>/...` path (the model echoed the absolute path it was given via `-f`). In `run.sh`: attach fixtures as paths relative to `PROJECT_DIR` (we already `cd` there), plus a `sed` scrub safety-net on the captured body (`PROJECT_DIR/` → relative, remaining `$HOME/` → `~/`). Re-ran and re-blessed all three cases; verified `baseline/` is now free of absolute home paths. Documented as a design invariant in `cobb/TESTING.md`.
- **Why:** User flagged that a committed baseline exposed the username via an absolute path; reports must use relative paths so baselines are safe to commit to a shared repo.
- **Plan items:** —

## 2026-05-31 — Added deterministic assertions (#2) to the agent eval harness
- **What:** Implemented item #2 from the promotion roadmap in `opencode/agents/severino/tests/run.sh`: an optional per-case `expect.md` with `require:`/`reject:` literal-substring directives, checked against the **response body only** (so a prompt quoting forbidden text can't trip a `reject:`). Prints per-assertion `PASS`/`FAIL`, a suite tally, and **gates the exit code** (any failure → exit 1; cases without `expect.md` stay advisory, exit 0). Excluded `expect.md` from attached fixtures; hardened `--help` to print only the leading comment block via awk. Seeded the 3 starter cases with assertions and verified live (clean: 4 passed/exit 0; forced fail: 1 passed 1 failed/exit 1). Updated `tests/README.md` and `cobb/TESTING.md` (assertions section + roadmap: #2 moved to done).
- **Why:** User chose to build #2 next. It's the highest-leverage maturity step — diffs are noisy on a stochastic local LLM, so deterministic substring assertions are what make the harness reword-proof and partially CI-gateable.
- **Plan items:** —

## 2026-05-31 — Authored TESTING.md + hardened the Severino eval harness
- **What:** (1) Created `cobb/TESTING.md` documenting the project's two-altitude testing standard (pytest for deterministic code; eval/bless harness for agent behavior) and the reusable agent-eval-harness pattern. (2) In `opencode/agents/severino/tests/run.sh`, did items #1 and #3 from the promotion review: **#3 decoupled from "severino"** — agent name now auto-derives from the parent dir (`AGENT=${AGENT:-$(basename "$PROJECT_DIR")}`), per-run stderr uses `mktemp` instead of a fixed `/tmp/severino-run.stderr`, and banner/help/comments are agent-agnostic; **#1 proven green** — ran all 3 cases, blessed `baseline/`, and confirmed the diff loop works (a re-run correctly flagged `changed`). Updated `tests/README.md` to document the agent-agnostic `AGENT` override.
- **Why:** User wants to promote the Severino harness into a reusable pattern; agreed to harden it first (prove green + decouple) before extracting a shared template. TESTING.md is cobb's living reference for that pattern. Deferred to roadmap: #2 lightweight `expect.md` assertions (highest leverage — case 02 diagnosed a bug correctly but emitted a broken fix, which a pure diff can't catch), temperature pinning, N-sample runs, strict/CI mode.
- **Plan items:** —

## 2026-05-31 — Restructured to per-agent subdirectories
- **What:** Moved each agent into its own folder (`cobb/cobb.md`, `tdd-engineer/…`, `medicina-alternativa/…`) and moved cobb's kaizen from `kaizen/cobb/` to `cobb/kaizen/`. Simplified the kaizen location rule in `cobb.md`: an artifact with its own folder uses `<folder>/kaizen/{plan,history}.md` (no `<name>` nesting); only a lone file sharing a directory namespaces by `<name>`. Made the README rule collection-level (root catalog). Updated `README.md` and `CLAUDE.md` paths.
- **Why:** User opted for a self-contained folder per agent. Verified Claude Code discovers `.claude/agents/` recursively and identifies agents by the `name:` frontmatter field, not the path — so per-agent subdirectories work natively (names must stay unique across the tree).
- **Plan items:** —

## 2026-05-31 — Added dual-audience documentation responsibility
- **What:** Added the "Documentation — keep both audiences informed" section to `cobb.md`: maintain a human-facing `README.md` catalog and update the project's agent-context convention (`CLAUDE.md` / `AGENTS.md` / `.kiro/steering`) on every create/edit/rename/remove. Bootstrapped `README.md` and `CLAUDE.md` for the `agents/` directory.
- **Why:** User wants agents created/edited by Cobb to always be documented for both the user and for other agents working on them.
- **Plan items:** —

## 2026-05-31 — Added kaizen maintenance responsibility
- **What:** Added the "Kaizen — maintain each agent's improvement plan & history" section to `cobb.md`, defining the `<dev-dir>/kaizen/<name>/{plan,history}.md` convention and templates. Bootstrapped Cobb's own `plan.md` and `history.md`.
- **Why:** User wants Cobb to maintain a living improvement plan and change history for every agent/skill it works on.
- **Plan items:** —

## 2026-05-31 — Agent created
- **What:** Initial authoring of the `cobb` agent — expert in agentic development across Claude Code / Claude Agent SDK, Kiro, and OpenCode, with a mandate to web-search official docs when specifics are version-sensitive. Frontmatter `name: cobb`, `model: opus`, routing-oriented `description`.
- **Why:** User requested a new agent specialized in agentic-development standards used by Claude, Kiro, and OpenCode.
- **Plan items:** seeded K-001 (re-verify docs), K-002 (porting example), K-003 (broaden tool coverage).
