# Kaizen — Change History: tico

> Dated log of actual changes to the `tico` agent. Most recent first.

## 2026-07-27 — Unpinned from `model: opus` (team-wide)
- **What:** Removed the `model: opus` frontmatter line. The field is now absent, so the agent runs on Claude Code's default — `model` **defaults to `inherit`** (re-verified 2026-07-27 against `code.claude.com/docs/en/sub-agents`), i.e. the model the session/system default selects. No other frontmatter or body change.
- **Why:** User no longer wants the team locked to Opus. Model choice belongs at the session level (one decision, changeable with `/model`), not duplicated across 13 frontmatter files where it silently overrides whatever the user picked.
- **Plan items:** —

## 2026-07-27 — Requirements-doc template adopts the canonical bolded header block (step 2 of `docs/plans/doc-reference-convention.md`)
- **What:** One line changed — the status line inside the `Structure:` template of the deliverable section. `> Status: Interviewing | Ready for design · Last updated: YYYY-MM-DD` becomes `> **Status:** Interviewing | Ready for design · **Owner:** \`tico\` · **Tracks:** <id(s)> (<M<n>>) · **Last updated:** YYYY-MM-DD`. Nothing else in the prompt moved: no frontmatter, no hook, and — the step's explicit proof obligation — the readback bullet that flips `Status` to **Ready for design** only on the stakeholder's explicit confirmation is untouched, as is `claude/README.md`'s user-facing statement of that promise (catalog row re-checked, no edit needed: it cites the write guard and the gated flip, not the template's form).
- **Why:** `docs/plans/doc-reference-convention.md` v1.4 §9.6, blocker **B4**. The repo-wide header block is now normative in root `AGENTS.md` and its canonical form is **bolded labels**; `tico`'s template was the unbolded dialect, and every requirements document written from it inherited that form, so the convention's own conformance regex could not have passed the three files it was told not to change. The convention absorbs `tico`'s two `Status` **values** verbatim rather than renaming them to lowercase tokens (§9.6, blocker B3) — the gated transition is product behaviour, not the architect's to change — so only the label's asterisks change, plus the two new fields (`Owner:`, `Tracks:`) the block requires. **Bolding a label is a form change, not a value change.**
- **Plan items:** none.

## 2026-07-24 — Description slimmed further (second team-wide token-cost pass)
- **What:** Frontmatter `description` compressed 593 → 538 chars (-9%): tightened phrasing, dropped restated detail. `tico` has no boundary pairs in `claude/scripts/audit-team.sh`; full audit re-verified green regardless. No body/catalog change.
- **Why:** All 13 agents' descriptions are auto-injected into every session and subagent spawn; the roster grew to 13 (graph-dba, joern added) since the first pass on 2026-07-11, and per-agent `/context` output showed room to cut further. User-requested via a `/context` token audit.
- **Plan items:** none.

## 2026-07-24 — Frontmatter: `permissionMode: acceptEdits`
- **What:** Added `permissionMode: acceptEdits` to the frontmatter, matching the same-day change across the team (`coder`, `tdd-engineer`, `frontend-engineer`, `architect`, `qa-engineer`, `analyst`, `devops`, `graph-dba`, `joern`, `teco`). File-edit/write approvals are session-scoped in Claude Code (unlike Bash approvals, which persist permanently per repo+command), so users otherwise have to re-grant write permission every session even with a global `Edit`/`Write` allow rule in `~/.claude/settings.json`.
- **Why:** Verified against current Claude Code docs (`hooks-guide.md` "Hooks and permission modes") that this is safe: `PreToolUse` hooks fire *before* any permission-mode check, and a hook's `"ask"` decision still forces the prompt even under `acceptEdits`/`bypassPermissions`. `tico`'s `guard-requirements-doc-writes.sh` hook (escalates to ask on any Write/Edit outside the allowed requirements-doc paths) keeps working exactly as before; only writes it would already let through silently stop re-prompting every session.
- **Plan items:** none.

## 2026-07-23 — Tico may commit its own deliverable
- **What:** The Bash guardrail no longer bans all tree mutation — it now carves out `git add`/`git commit`, scoped to exactly the paths the Write/Edit guard already allows (the requirements doc(s), the kaizen inbox), staged by explicit path only (`git add -A`/`.`/`commit -a` still forbidden). Added a "Commit as you go" bullet alongside "Write as you go" and a note in the Handoff section to commit the doc's final state before closing.
- **Why:** User ruling: they treat requirements docs as code and want tico to version its own files as part of authoring them, not leave commits to a human or downstream agent. This is a deliberate narrowing of the prior full Bash-mutation ban (see the 2026-07-09 creation entry and `guard-doc-writes.sh`'s design note on architect K-003) — not a hook change: Bash mutation stays prompt-guarded by convention, only the *scope* of what's permitted moved. No push/reset/rebase/amend; no bulk-staging flags.
- **Plan items:** none.

## 2026-07-12 — Learning-capture loop: kaizen inbox + closing protocol + guard allowlist
- **What:** Added `kaizen/inbox.md` (append-only learnings inbox, seeded empty) and a "Learning capture" closing-protocol section to the prompt; the doc-scoped write guard's allowlist gained exactly the agent's own inbox path (`<name>/kaizen/inbox.md`), with the escalation message updated to match.
- **Why:** Team-wide self-improvement loop (agent-maintenance skill §5, added the same day): capture is cheap and unreviewed during runs, promotion is curated — cobb periodically verifies each entry and routes it to the prompt, an on-demand knowledge base, or project docs. Requested by the user.
- **Plan items:** none.

## 2026-07-11 — Description slimmed (team-wide token-cost pass)
- **What:** Frontmatter `description` compressed from 802 to 585 chars: capability lists tightened, reciprocal boundary prose reduced to short route-away clauses that still name the counterpart agents (audit check 6 boundary symmetry preserved — full pass green), and "how I work" detail dropped from the description since the prompt body already carries it. Routing semantics unchanged; no body/catalog changes needed.
- **Why:** All 12 agents' descriptions are auto-injected into every session and into every subagent spawn that carries the `Agent` tool; team-wide they cost 12,609 chars (~3.1K tokens) per injection. The pass cut them to 7,036 chars (~44%), saving ≈1,400 tokens per session/spawn with the same routing contract.
- **Plan items:** none.

## 2026-07-11 — Guard hook refactored to a thin wrapper over a shared core
- **What:** `guard-requirements-doc-writes.sh` was reduced from a ~60-line standalone script to a thin wrapper that `exec`s the new shared core `claude/scripts/guard-doc-writes.sh` with two parameters — this agent's allowed-path globs (`docs/requirements/*|*/docs/requirements/*`) and its escalation-message template (`__PATH__` placeholder for the offending path). The core carries the shared machinery unchanged: jq→python3 path extraction, fail-open on unparseable input, `/tmp/*` always allowed, `permissionDecision: "ask"` JSON emit. The wrapper resolves the core via `readlink -f "$0"`, so it works when invoked through the `~/.claude/agents/<name>` deployment symlink; the frontmatter hook command is unchanged. Verified: `bash -n`, allowed/denied/scratchpad/fail-open cases through the symlink path, the no-jq python3 fallback, and `claude/scripts/audit-team.sh` all pass.
- **Why:** a repo redundancy audit (2026-07-11) found the five doc-scoped guards (analyst, architect, data-scientist, teco, tico) byte-identical except one `case` glob and one message string — ~250 duplicated lines that had to be patched five times per fix. One parameterized core removes the drift risk. (`devops/hooks/guard-destructive-ops.sh` stays standalone — it matches Bash command patterns, not write paths.)
- **Plan items:** none.

## 2026-07-10 — Hook command made machine-independent (`$HOME` symlink path)
- **What:** the frontmatter `PreToolUse` hook command was rewired from the absolute repo path (`/home/<user>/prg/graphmind-ai-lab/claude/tico/hooks/guard-requirements-doc-writes.sh`) to `$HOME/.claude/agents/tico/hooks/guard-requirements-doc-writes.sh`, which resolves through the user-scope deployment symlink (`~/.claude/agents/tico` → the repo folder). Shell-form hook commands (no `args`) run via `sh -c`, so `$HOME` expands — verified 2026-07-10 against `code.claude.com/docs/en/hooks`. Resolution through the symlink confirmed (`test -x` passes).
- **Why:** the committed agent source leaked the user's personal home path into the repo; the symlink path is identical on any machine that follows the deployment convention (`~/.claude/agents/<name>` → `claude/<name>`), keeping the hook enforceable without machine-specific paths. (`${CLAUDE_PROJECT_DIR}` was rejected: the agents are user-scoped and must guard in any project, where the project dir isn't this repo.)
- **Plan items:** none.

## 2026-07-09 — Redesigned as a first-order conversational agent
- **What:** tico now runs as the **main-session agent** (`claude --agent tico`) — verified 2026-07-09 against `code.claude.com/docs/en/sub-agents`: the main thread takes on the definition's prompt/tools/model, frontmatter hooks still fire in main-session mode, and `initialPrompt` auto-submits as the first user turn. Prompt rewritten around a **live interview**: one thread at a time, reflect-back confirmations, `AskUserQuestion` (added to `tools`) for option picks, the doc updated as the conversation progresses, and a readback + explicit stakeholder confirmation gating the "Ready for design" flip. The round-based protocol shrank to a degraded subagent fallback ("If you are invoked as a subagent anyway"). teco no longer delegates to tico — it consumes the doc by path and pauses to the user when requirements need capturing.
- **Why:** User ruling on the initial design: tico is not a subagent but a first-order agent meant to be conversational — the rounds protocol optimized for the wrong constraint.
- **Plan items:** K-002 ⚪ rejected (continuation machinery moot in first-order mode); K-001 retargeted to a live `claude --agent tico` session.

## 2026-07-09 — created
- **What:** initial version of `tico` — conversational product-owner subagent that interviews the user/stakeholder about a feature request and owns the feature requirements document (`<component>/docs/requirements/<slug>.md`). Round-based interview protocol (subagents can't `AskUserQuestion`, so each invocation folds answers into the doc and returns the next question batch as its deliverable; the doc is the durable state between rounds). Write/Edit scoped to the requirements doc, harness-enforced by `hooks/guard-requirements-doc-writes.sh` (same PreToolUse "ask"-escalation pattern as architect/teco/devops). Model `opus`; tools mirror the architect's investigation set.
- **Why:** the team had design (architect) through delivery (coder/tdd/qa) covered, but nothing upstream capturing WHAT/WHY before the architect decides HOW — vague feature requests went straight to design. Requested by the user 2026-07-09.
- **Plan items:** seeded K-001 (e2e spin), K-002 (SendMessage rounds), K-003 (FR-id traceability).
