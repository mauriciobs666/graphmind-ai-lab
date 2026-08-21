# Kaizen — Change History: data-scientist

> Dated log of actual changes to the `data-scientist` agent. Most recent first.

## 2026-08-21 — `kaizen_team` distillation: 8 entries — 3 to the prompt, 3 to `lm-studio-model-notes.md`, 1 discarded as an already-resolved one-off, 1 blocked from its original target by archival and redirected

- **What:** `cobb` processed all 8 `author:'data-scientist'` entries in the shared `kaizen_team`
  graph (agent-maintenance skill §5). Read via single-column `substring()` paging (avoids the
  multi-column chat-rendering corruption documented in `cypher-mcp/README.md`).
  - **Promoted (3) → `data-scientist.md` prompt, folded into existing bullets (not new top-level
    bullets — kept cognitive load flat):**
    - `8f6e20a1` — when the judge collapses onto the same model as the agent-under-test, a
      blanket self-preference caveat conflates a fixed-content sub-pass (little risk) with a
      live-output sub-pass (real risk); split it. Folded into the LLM-as-judge bullet.
    - `11475cca` + `f3b90490` (same "Evaluation engineering" bullet, folded together) — before
      blessing a zero-tolerance small-golden-set gate, compare the one-unit delta (`1/n`) to the
      metric's CI width at that `n`; and when a probe set is gated with AND/OR logic, still
      report each probe's individual outcome in prose, not just the bloc boolean — both are
      reusable pre-sign-off habits, not falkor-chat-specific.
    - `e6f3a1c4` — this lab's small-n pass/fail convention is the Wilson score interval, not
      Clopper-Pearson/rule-of-three (independently re-derived in Python, matches the archived
      `m3-guard-calibration.md` §6 figures to rounding). **Redirected from its own suggested
      target**: the entry's context doc, `golden-set-expansion-ml.md`, is now `Status: archived`
      (frozen except header-pointer metadata) — the fact is durable and reusable well beyond that
      one document, so it went into the "Uncertainty" bullet instead of being lost.
    - Folded `9642e9e0` in kind but routed to the knowledge base instead (next bullet) — its
      LM-Studio-specific verification commands are better as a consultable recipe than prompt
      prose the agent pays for every session.
  - **Promoted (3) → new sections in `claude/data-scientist/lm-studio-model-notes.md`:**
    - `e820b9e0` — Mistral/Ministral GGUF templates enforce strict role alternation (HTTP 400 on
      two consecutive same-role messages); Qwen3 tolerates it silently. Live-verified against
      `falkor-chat`'s real `triage@v1` intake prompt.
    - `a1c2ef6d` — two differently-named LM Studio catalog ids can alias one loaded model slot
      (confirmed via `/api/v0/models` state-flipping and byte-identical completions) — verify
      before assuming two entries are two different weight files.
    - `9642e9e0` — a live-run report's provenance can silently diverge from the repo's static
      model config per box (the provider file is machine-local, outside the repo, and
      `ProviderCatalog` validates only the provider id, not the model id); two reusable habits:
      live-check `/api/v0/models` rather than trusting static config, and grep the repo for
      `temperature` before assuming a pinned sampling parameter.
  - **Discarded (1):** `b3f2a8e0` — a golden-set fixture id-numbering footnote (a draft plan
    section had "used" `tn-08`/`tn-09` ahead of the real fixture file) whose only plausible
    target, `golden-set-expansion-ml.md`, is now archived and the numbering it describes is
    already resolved in the delivered `golden_guards.jsonl`; too narrow and already-moot to carry
    forward into a different document.
  - **Verified:** live-re-derived `e6f3a1c4`'s Wilson figures in Python (stdlib `math` only) before
    promoting — reproduced `wilson_upper(0,10)=27.75%`, `wilson_upper(0,30)=11.35%`, matching the
    entry's own citation.
  - **Docs touched:** `claude/data-scientist/{data-scientist.md,lm-studio-model-notes.md,
    kaizen/history.md}`.
- **Why:** User-requested distillation pass ("who's next?" → data-scientist had the oldest pending
  entries, 2026-08-15).
- **Plan items:** none opened — every entry had a direct promotion target or a clear discard
  rationale.

## 2026-08-21 — Persona fix: dropped stale "senior" framing (team certification, §7 fold-in)

- **What:** Opening line "You are a senior **data scientist and AI/ML specialist**..." →
  "You are a **data scientist and AI/ML specialist**...". Dropped the one word.
- **Why:** Caught during a user-requested full team-coherence certification's §7 lint fold-in.
  The team dropped "senior" framing collection-wide on 2026-06-20 (overconfidence concern;
  persona-prompting evidence shows role labels are weak-to-neutral for correctness —
  `claude/cobb/kaizen/history.md`, 2026-06-20 entry, "Collection harmonization" — applied
  explicitly to `cobb` itself and stated as bringing "the whole Claude collection" in line).
  `data-scientist.md` had never been swept for it; genuine drift against a dated, explicit
  team decision, not a fresh design call.
- **Verified:** `bash claude/scripts/audit-team.sh` — same 113 PASS / 2 pre-existing FAILs before
  and after (diff, not bare gate).
- **Plan items:** none opened — direct fix from a live certification finding.

## 2026-08-20 — Learnings capture migrated to a working-memory graph (`kaizen_data-scientist`), mirroring `graph-dba`; `mcp__cypher__query` granted
- **What:** The "Learning capture" closing-protocol section now writes a `:KaizenEntry` node
  directly into `kaizen_data-scientist` (FalkorDB, via `mcp__cypher__query`) instead of appending
  to `kaizen/inbox.md`. `kaizen/inbox.md` is now a frozen historical snapshot — its 4
  pre-existing entries were parsed out programmatically and imported into the graph verbatim
  (entryId assigned, `author: 'data-scientist'`), preserving every field; its own header explains
  the freeze and gives the live-read query. Frontmatter `tools:` gained `mcp__cypher__query` —
  this agent previously had no MCP tool access at all, needed now for both this capture path and
  any future graph reads. The trailing "Your write guard allows exactly this inbox path" clause
  was dropped — the write guard gates `Write`/`Edit`, not the `mcp__cypher__query` MCP tool, so
  it no longer applies to this capture path.
- **Why:** User-directed team-wide redesign ("I will migrate all agents to write their learnings
  to the graph like graph-dba"), reversing yesterday's file-based Learning-capture dedup (entry
  below) — the user determined the whole team should follow `graph-dba`'s existing graph-based
  capture pattern instead of the file-based inbox convention.
- **Plan items:** —

## 2026-08-19 — Learning-capture paragraph de-duplicated against the inbox's own header
- **What:** Trimmed the "Learning capture" paragraph: dropped "(fact, evidence, suggested home; format in the file header)" and "The inbox is raw capture — the team maintainer verifies and promotes entries into prompts, knowledge bases, or project docs" — both already stated verbatim in `kaizen/inbox.md`'s own header template (agent-maintenance skill §5), which the agent necessarily opens to append. Kept: the discipline-specific fact-kind clause, the inbox path, "skip task-specific details," "never edit your own agent definition," and the write-guard clause. Behavior unchanged.
- **Why:** User-directed prompt-verbosity reduction, item 1 of the parked diagnosis (`cobb/kaizen/plan.md`) — the mechanics were literally duplicated (prompt + inbox header say the same thing), not just similar boilerplate; pointing at the file's own header removes the duplication without losing information, since the agent reads that file to act anyway.
- **Plan items:** —

## 2026-08-11 — Inbox distillation: 4 entries — 1 prompt addition, 1 new knowledge base, 1 to `python-web-quirks`, 1 discarded as stale

- **What:** `cobb` processed all 4 entries in `data-scientist/kaizen/inbox.md` (§5).
- **Promoted:**
  - Bias-to-suspend judges need class-conditional gating (false-advance/advance-recall), not
    κ/accuracy → new clause in "Core expertise → LLM systems → LLM-as-judge".
  - Ministral-3B vs. Qwen3-4B tool-calling reliability → new on-demand knowledge base,
    `claude/data-scientist/lm-studio-model-notes.md`, pointed to from "Core expertise → Model
    selection".
  - Bare `json.loads` on an LLM judge's output being fence-fragile → `skills/python-web-quirks/
    SKILL.md` (general knowledge; the project-specific instance already has an open tracking item,
    K-027, in `falkor-chat/docs/BACKLOG.md`, so no new backlog action needed).
- **Discarded:** `read_thread`'s `authorType` being a list, not a string — the flagged gap was in a
  since-completed M3 plan doc's prose; the live query itself (`labels(author) AS authorType`) is
  self-documenting (`labels()` obviously returns a list) and is already cross-referenced correctly
  in `falkor-chat/docs/HISTORY.md`.
- **Verified:** `bash claude/scripts/audit-team.sh` clean.
- **Docs touched:** `claude/data-scientist/{data-scientist.md,lm-studio-model-notes.md,
  kaizen/{history,inbox}.md}` · `skills/python-web-quirks/SKILL.md`.

## 2026-07-27 — Unpinned from `model: opus` (team-wide)
- **What:** Removed the `model: opus` frontmatter line. The field is now absent, so the agent runs on Claude Code's default — `model` **defaults to `inherit`** (re-verified 2026-07-27 against `code.claude.com/docs/en/sub-agents`), i.e. the model the session/system default selects. No other frontmatter or body change.
- **Why:** User no longer wants the team locked to Opus. Model choice belongs at the session level (one decision, changeable with `/model`), not duplicated across 13 frontmatter files where it silently overrides whatever the user picked.
- **Plan items:** —

## 2026-07-27 — Method notes and methodology reviews open with the canonical header block (step 2 of `docs/plans/doc-reference-convention.md`)
- **What:** One line added to *Your deliverables*, after the two document bullets and before the "return the path" line: *"Open the document with the header block from root `AGENTS.md`."* Placed so it covers both written deliverables (`docs/plans/<slug>-ml.md` and `docs/reviews/<slug>-ml.md`) and not the inline-consultation bullet, which produces no document. No frontmatter, hook, `description` or catalog change.
- **Why:** `docs/plans/doc-reference-convention.md` v1.4 §9.6 makes a three-field header (`Status:` · `Owner:` · `Tracks:`) the repo's lifecycle signal, replacing the milestone filename prefix and the move-to-`archive/` rule; both `-ml` documents are in the closed role set and both are cited by path from an architect plan, so they need the same header as everything they sit beside. The line is a **pointer, not an inlined template** (v1.4 M20) — root `AGENTS.md` is already in every agent's context via the root `CLAUDE.md` `@AGENTS.md` import — and is byte-identical across the six producing prompts because the convention's coverage check greps for it literally. `claude/README.md` row 18 re-checked — it cites both write paths and the hook, not document structure; no edit needed.
- **Plan items:** none. (K-001's first-run shakedown, when it happens, now also exercises the header block.)

## 2026-07-24 — Description slimmed further (second team-wide token-cost pass)
- **What:** Frontmatter `description` compressed 676 → 606 chars (-10%): tightened phrasing, dropped restated detail, kept every routing/boundary clause. `claude/scripts/audit-team.sh` boundary-pair symmetry (data-scientist↔architect, data-scientist↔analyst, data-scientist↔graph-dba) re-verified green. No body/catalog change.
- **Why:** All 13 agents' descriptions are auto-injected into every session and subagent spawn; the roster grew to 13 (graph-dba, joern added) since the first pass on 2026-07-11, and per-agent `/context` output showed room to cut further. User-requested via a `/context` token audit.
- **Plan items:** none.

## 2026-07-24 — Frontmatter: `permissionMode: acceptEdits`
- **What:** Added `permissionMode: acceptEdits` to the frontmatter, matching the same-day change across the team (`coder`, `tdd-engineer`, `frontend-engineer`, `architect`, `qa-engineer`, `analyst`, `devops`, `graph-dba`, `joern`, `teco`, `tico`). File-edit/write approvals are session-scoped in Claude Code (unlike Bash approvals, which persist permanently per repo+command), so users otherwise have to re-grant write permission every session even with a global `Edit`/`Write` allow rule in `~/.claude/settings.json`.
- **Why:** Verified against current Claude Code docs (`hooks-guide.md` "Hooks and permission modes") that this is safe: `PreToolUse` hooks fire *before* any permission-mode check, and a hook's `"ask"` decision still forces the prompt even under `acceptEdits`/`bypassPermissions`. `data-scientist`'s `guard-ds-doc-writes.sh` hook (escalates to ask on any Write/Edit outside the allowed methodology-doc paths) keeps working exactly as before; only writes it would already let through silently stop re-prompting every session.
- **Plan items:** none.

## 2026-07-12 — Learning-capture loop: kaizen inbox + closing protocol + guard allowlist
- **What:** Added `kaizen/inbox.md` (append-only learnings inbox, seeded empty) and a "Learning capture" closing-protocol section to the prompt; the doc-scoped write guard's allowlist gained exactly the agent's own inbox path (`<name>/kaizen/inbox.md`), with the escalation message updated to match.
- **Why:** Team-wide self-improvement loop (agent-maintenance skill §5, added the same day): capture is cheap and unreviewed during runs, promotion is curated — cobb periodically verifies each entry and routes it to the prompt, an on-demand knowledge base, or project docs. Requested by the user.
- **Plan items:** none.

## 2026-07-11 — Description slimmed (team-wide token-cost pass)
- **What:** Frontmatter `description` compressed from 1472 to 674 chars: capability lists tightened, reciprocal boundary prose reduced to short route-away clauses that still name the counterpart agents (audit check 6 boundary symmetry preserved — full pass green), and "how I work" detail dropped from the description since the prompt body already carries it. Routing semantics unchanged; no body/catalog changes needed.
- **Why:** All 12 agents' descriptions are auto-injected into every session and into every subagent spawn that carries the `Agent` tool; team-wide they cost 12,609 chars (~3.1K tokens) per injection. The pass cut them to 7,036 chars (~44%), saving ≈1,400 tokens per session/spawn with the same routing contract.
- **Plan items:** none.

## 2026-07-11 — Guard hook refactored to a thin wrapper over a shared core
- **What:** `guard-ds-doc-writes.sh` was reduced from a ~60-line standalone script to a thin wrapper that `exec`s the new shared core `claude/scripts/guard-doc-writes.sh` with two parameters — this agent's allowed-path globs (`docs/plans/*|*/docs/plans/*|docs/reviews/*|*/docs/reviews/*`) and its escalation-message template (`__PATH__` placeholder for the offending path). The core carries the shared machinery unchanged: jq→python3 path extraction, fail-open on unparseable input, `/tmp/*` always allowed, `permissionDecision: "ask"` JSON emit. The wrapper resolves the core via `readlink -f "$0"`, so it works when invoked through the `~/.claude/agents/<name>` deployment symlink; the frontmatter hook command is unchanged. Verified: `bash -n`, allowed/denied/scratchpad/fail-open cases through the symlink path, the no-jq python3 fallback, and `claude/scripts/audit-team.sh` all pass.
- **Why:** a repo redundancy audit (2026-07-11) found the five doc-scoped guards (analyst, architect, data-scientist, teco, tico) byte-identical except one `case` glob and one message string — ~250 duplicated lines that had to be patched five times per fix. One parameterized core removes the drift risk. (`devops/hooks/guard-destructive-ops.sh` stays standalone — it matches Bash command patterns, not write paths.)
- **Plan items:** none.

## 2026-07-10 — Hook command made machine-independent (`$HOME` symlink path)
- **What:** the frontmatter `PreToolUse` hook command was rewired from the absolute repo path (`/home/<user>/prg/graphmind-ai-lab/claude/data-scientist/hooks/guard-ds-doc-writes.sh`) to `$HOME/.claude/agents/data-scientist/hooks/guard-ds-doc-writes.sh`, which resolves through the user-scope deployment symlink (`~/.claude/agents/data-scientist` → the repo folder). Shell-form hook commands (no `args`) run via `sh -c`, so `$HOME` expands — verified 2026-07-10 against `code.claude.com/docs/en/hooks`. Resolution through the symlink confirmed (`test -x` passes).
- **Why:** the committed agent source leaked the user's personal home path into the repo; the symlink path is identical on any machine that follows the deployment convention (`~/.claude/agents/<name>` → `claude/<name>`), keeping the hook enforceable without machine-specific paths. (`${CLAUDE_PROJECT_DIR}` was rejected: the agents are user-scoped and must guard in any project, where the project dir isn't this repo.)
- **Plan items:** none.

## 2026-07-09 — Created
- **What:** Initial version of the `data-scientist` agent — the team's AI/ML/data-science specialist, created to work alongside `architect` (supplies the ML/DS method inside a design) and `analyst` (methodology review of plans/code). Advisory-only shape chosen by the user over a hands-on (graph-dba-style) shape: read-only on code, `Write`/`Edit` scoped to method notes (`docs/plans/<slug>-ml.md`) and methodology reviews (`docs/reviews/<slug>-ml.md`), harness-enforced by `hooks/guard-ds-doc-writes.sh` (PreToolUse, matcher `Write|Edit`, same contract as the analyst's guard but allowing both doc homes). Tools match architect/analyst (`Read, Grep, Glob, Bash, Write, Edit, WebFetch, WebSearch, Agent`); model opus; subagent-aware (questions return as the deliverable).
- **Why:** The team had no ML/DS-methodology depth — model/embedding selection, RAG/GraphRAG evaluation design, metric choice, experiment design, statistical validity all landed on generalists. This lab's two themes (graph-backed AI apps, agent engineering) make the gap recurring.
- **Boundary pairs declared** (added to `claude/scripts/audit-team.sh` `BOUNDARY_PAIRS`, reciprocal clauses added to partners' descriptions): `architect:data-scientist` (software plan vs. ML method inside it), `analyst:data-scientist` (general static review vs. methodology review), `graph-dba:data-scientist` (in-graph vector mechanics vs. embedding/eval method). teco's routing table + handoff contracts gained a data-scientist row/entry.
