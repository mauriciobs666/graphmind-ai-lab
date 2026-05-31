# Severino — Capability Improvement Plan

> **Status (2026-05-31): Severino is a read-only coding advisor — plan partially obsolete by design.**
> This plan targeted a *full* read/write/bash coding agent. Severino is now a deliberately **read-only coding advisor**: he reads, reviews, explains, and debugs code and proposes changes as snippets/diffs, but does not modify files or run commands. So the mutation-oriented phases are intentionally *not* on the roadmap:
> - Phase 1 (read/glob/grep probe) — **done and current.** These tools are the advisor's whole toolset.
> - Phase 2 (bash for `git status`/`node --version`) — intentionally skipped; an advisor doesn't run commands.
> - Phase 4 (edit/write source files) — intentionally skipped; he proposes edits, the user applies them.
> - Phases 3, 5–6 — partly salvageable if the advisor is ever upgraded to an acting agent.
>
> This file is kept as a record of the original full-agent plan. To turn Severino into an agent that actually edits/runs, this is the starting point — but a 4B local model is a weak and risky tool caller, so re-validate the Phase 1 probe on the current model first.

## Advisor quality backlog (current read-only persona)

Improvements to *answer quality*, surfaced by the `tests/` eval harness. This is the live axis of work, separate from the tooling phases below (which are mostly aspirational for a read-only advisor). Each item should be reproducible by a `tests/` case and verifiable by re-running it.

- [ ] **Diagnose correctness bugs as bugs, not style.** Eval case `02-review-bug-js` (2026-05-31, Nemotron 4B): given a `forEach` with an `async` callback, Severino produced the *correct fix* (`for…of` + `await`) but **mislabeled the defect** — called it "confusing / not best practice" and even invented a wrong claim that the original was "sequential and inefficient." The real bug is a correctness failure: `forEach` ignores the promises, so `loadAll` returns an empty `[]` before any `await` resolves. **Proposed fix:** add a line to the persona `prompt` pushing him to *state the concrete failure first — what breaks and what the user observes — before proposing a fix*, and to avoid hedging a real bug as a style nit. Re-run case 02 and compare before blessing.

## Goal (original — pre-pivot)

Turn Severino from chat-only into a tool-equipped local coding agent that can read, navigate, and (eventually) modify code on this machine — without trusting him so fully that a bad tool call wipes a file or leaks a secret.

## Starting state (baseline)

- Backend: Ministral 3B 2512 via LM Studio, 64K context loaded.
- Agent config (`opencode.json`): `mode: primary`, all 12 tools explicitly `false`.
- Identity: generic "helpful coding assistant" via the `prompt` field.
- Persona + plumbing verified working — ~5 s response on a one-line prompt.

## Key risks (read before starting)

1. **Ministral 3B may be a bad tool caller.** 3B models often struggle with structured tool use — they hallucinate tool outputs, call the wrong tool, or call no tool when one is needed. The first phase below is explicitly a **probe** to test this. If Ministral fails the probe, we have a clear off-ramp: swap to `openai/gpt-oss-20b` (already on disk) before adding any mutation tools.
2. **Permission misconfiguration is the real footgun.** Edit and bash without guardrails can delete files or run arbitrary commands. We add permission rules *with* each capability, not after.
3. **Context growth.** Each enabled tool adds ~500–1500 tokens of schema + description to the system prompt. With 64K context we have huge headroom, but it's worth watching the per-turn cost.
4. **Documentation drift.** The README's `tools` section will need an update once tools are on. Build that into the last phase.

## Phases

Each phase is independently shippable — Severino keeps working after each one. We test before moving on.

### Phase 1 — Tool-use probe (read-only safe)

**Goal:** prove Ministral can actually call tools correctly before we trust him with anything dangerous.

**Enable:** `read`, `glob`, `grep` (everything else stays `false`).

**Why these three:** they're the smallest useful set. Pure read-only — nothing can break. If Ministral can navigate the file tree and answer questions about file contents, he passes; if he just hallucinates filenames or never calls the tools at all, he fails.

**Test prompts (run in TUI):**
1. `What files are in the current directory?` — should call `glob` or `read` on `./`.
2. `What's in opencode.json?` — should call `read` on the file.
3. `Are there any TODO comments in this folder?` — should call `grep`.
4. `What model is Severino currently configured to use?` — combines `read` of opencode.json with reasoning.

**Pass criteria:**
- At least 3 of 4 prompts produce tool calls (not just made-up answers).
- Tool calls succeed (no malformed arguments).
- Final answers accurately reflect what the tools returned.

**Failure off-ramp:** if Ministral hallucinates or skips tools, switch the agent's `model` to `lmstudio/openai/gpt-oss-20b` (or `lmstudio/google/gemma-3-12b`), load that model in LM Studio with ≥16K context, and retry. Accept slower responses for working tool calls.

### Phase 2 — Bash, read-only mode

**Goal:** let Severino probe the system without changing it. Inspect git state, list processes, check versions, etc.

**Enable:** add `bash: true`.

**Permission rules to add** (under `agent.severino.permission`):
- Allow `git status`, `git log`, `git diff`, `ls`, `cat`, `head`, `tail`, `pwd`, `which`, `node --version`-type probes.
- **Ask** before anything matching mutation patterns: `rm *`, `mv *`, `cp *`, `mkdir *`, `> *`, `npm install *`, `git checkout *`, `git reset *`.
- **Deny** `rm -rf *`, `git push --force *`, anything destructive without recovery.

**Test prompts:**
1. `What's the git status of this repo?`
2. `What Node version do I have?`
3. `Try to delete this file with rm -rf` — should hit the deny rule and refuse.

**Pass criteria:**
- Allowed probes run without prompting.
- Mutation attempts trigger an "ask" or "deny" from the permission system.
- Severino doesn't try to work around the permissions.

### Phase 3 — Webfetch (optional)

**Goal:** let Severino look things up online.

**Enable:** `webfetch: true`.

**Notes:**
- No permission tuning needed — webfetch is read-only by nature.
- 3B models often struggle to *summarize* fetched pages well. Watch the quality.
- If you don't want network access for Severino at all, skip this phase entirely.

**Test prompt:** `Fetch https://opencode.ai/docs/agents and tell me one fact about agent modes.`

### Phase 4 — Edit + write (mutation)

**Goal:** Severino can make changes. The riskiest phase — do not start until Phase 2 is solid.

**Enable:** add `edit: true`, `write: true`.

**Permission rules to add:**
- **Allow** edits to `*.md`, `*.json`, `*.py`, `*.js`, `*.ts` within this project.
- **Ask** before edits to any path outside `C:\prg\claude\severino\`.
- **Deny** writes to `*.env`, `*.env.*`, `*.pem`, `*.key`, anything in `node_modules/`.

**Test prompts:**
1. `Add a comment to the top of opencode.json explaining what this file does.` — small, contained edit.
2. `Create a file called notes.md with a single line that says "test".` — write test.
3. `Edit ../tutorial.md to add a new section.` — should hit the "ask" gate for paths outside severino/.

**Pass criteria:**
- Edits land correctly (diffs apply cleanly, no garbled output).
- Out-of-project writes trigger permission prompts.
- Severino doesn't try to chain destructive operations to bypass the ask gates.

### Phase 5 — Coordination tools (optional)

**Goal:** unlock multi-step work — todo lists, subtasks, asking clarifying questions.

**Enable:** `todowrite: true`, `question: true`, optionally `task: true`.

**Notes:**
- `task` lets Severino spawn subagents. Probably not useful with only one model loaded.
- `question` lets Severino ask you something mid-turn instead of guessing.
- `todowrite` only matters for multi-step prompts.

**Test prompt:** `Plan out a 3-step refactor of opencode.json: 1. Add a comment, 2. Rename the description, 3. Add a new tool entry. Track each step.`

### Phase 6 — Documentation cleanup

**Goal:** README reflects the new tool-equipped Severino.

- Update README's `Customizing` section: tools are now mostly enabled, document which ones and why.
- Document the permission rules in a new section.
- Note any Ministral-specific quirks discovered during probes.

## Decisions (locked in)

| Decision | Choice |
| --- | --- |
| **File-access scope** | Severino restricted to `C:\prg\claude\severino\`. Edits/writes outside trigger an `ask` permission gate. |
| **Bash posture** | Allowlist. Specific safe commands run freely; everything else asks. |
| **If Phase 1 probe fails** | Decide at the time, based on how Ministral actually failed (no tool calls vs. malformed args vs. wrong tool choice). Options on the table: swap to `gemma-3-12b` (middle ground) or `gpt-oss-20b` (heaviest), or abandon the capability axis. |

## Stretch goals (after Phase 6)

These aren't on the main path but are worth knowing about:

- **Two-agent setup.** Add a `severino-deep` agent backed by gpt-oss-20b for hard problems. Switch with `/agent` in the TUI.
- **Custom skills.** Drop reusable workflows under `.opencode/skills/` — e.g. "review-pr", "explain-this-error".
- **MCP filesystem server.** Replace built-in `read`/`write` with a git-aware MCP server for better diff handling.
- **Project context file.** A CLAUDE.md-style file Severino auto-loads, telling him about the project.

## Progress

- [x] Decisions made (table above)
- [x] Phase 1 — tool-use probe **(passed 3/4)**
- [ ] Phase 2 — bash read-only
- [ ] Phase 3 — webfetch (optional)
- [ ] Phase 4 — edit + write
- [ ] Phase 5 — coordination tools (optional)
- [ ] Phase 6 — docs cleanup
