# Kaizen — Change History: cobb

> Dated log of actual changes to the `cobb` agent. Most recent first.

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
