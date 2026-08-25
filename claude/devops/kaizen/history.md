# Kaizen — Change History: devops

> Dated log of actual changes to the `devops` agent. Most recent first.

## 2026-08-24 — Prompt-waste compression, Stage C4: the orientation example deleted after its third rot — file measured at its editorial floor
- **What:** One edit (2,212 → 2,110 w, −102, −4.6%), executed as one unit with `security-expert.md` (`claude/docs/plans/prompt-waste-reduction.md`, Stage C4). Deleted the ~118-w blockquote after the infra-brief paragraph: "**Example of what this yields (the graphmind-ai-lab repo where this agent was authored):** …" — the Docker-centric-monorepo brief citing `falkordb/falkordb:v4.18.11`, ports 6379/3000, the `falkordb-data` volume, the `start_falkordb.sh` delegation, Python ≥3.12, the `pyproject`-vs-`requirements` split, and "greenfield gaps (no Compose, no CI, no app image builds)".
- **Why — three arguments, in ascending order of weight:**
  1. **It had rotted.** Live-verified 2026-08-24: `falkor-chat/compose.yaml` exists (and pins the same `v4.18.11`), `falkor-chat/Dockerfile` and `cypher-mcp/Dockerfile` exist. So "no Compose, no CI, no app image builds" was **2/3 false**. Only "no CI" still holds — `.github/workflows/falkor-chat.yml` was removed in `5a97821`.
  2. **This was its third rot, and refresh-in-place had provably stopped working.** The block was already repaired twice — 2026-07-09 (image tag `edge` → `v4.18.11`) and 2026-07-11 (start-script consolidation). The tag installed by the first repair is *still accurate*; the block rotted in a **different slot** each time. Repairing the fact that broke last time never protected the facts that broke next. A third refresh would have re-armed the same trap.
  3. **`devops` is user-scoped — this is the decisive argument.** The agent runs in every project. A memorized snapshot of *this* repo was a false anchor competing with the real project's orientation in every non-graphmind project, on the one agent whose entire remit is "don't generalize from another repo." It was a counter-example to the file's own thesis, and to its own §Orient rule ("confirm against live state, don't trust the docs blindly").
- **Gate (a) inventory — no rule lost.** The block contained no class-1/2 clause. Its two functions both survive: (i) the **content spec** for a brief is stated abstractly and completely in the paragraph directly above it (*what the stack is, how it's containerized, how a dev boots it, where state and secrets live, what CI/deploy exists, and what's missing*) — six named slots; (ii) the **portability rule** it closed with ("*not* a memorized fact… in a different repo you'd derive a completely different brief") is stated three other times: "You do **not** assume a stack, a toolchain, or a convention", "Never generalize from another repo", and the ranked principle "an assumption carried in from another repo is the most likely way to break something".
- **Gate (b) — every deleted fact traced to a surviving home** (`cobb` verified independently): shared FalkorDB as the one runtime → root `AGENTS.md`; `start_falkordb.sh` + the `falkordb-data` volume → `falkor-chat/AGENTS.md`; salesperson delegating to falkor-chat's script, one instance → `salesperson/README.md`; image tag and ports → `falkor-chat/scripts/start_falkordb.sh` and `falkor-chat/compose.yaml`. Nothing was destroyed — it was already in project docs, which is where this prompt's own routing rule says project facts belong.
- **Counter-argument weighed and rejected:** that the blockquote was the file's only concrete illustration of a finished infra brief, and abstract specs shape output less well than worked examples. `cobb` checked the premise and it is false — "**Diagnose or design** at the right altitude" already gives three worked, non-perishable shapes (a one-line port-clash fix, a Compose file unifying ad-hoc run scripts, a pipeline spec for "add CI"), and the "Shared stateful services" principle keeps a live concrete anchor, correctly hedged ("*In this repo that shared service is the FalkorDB instance; in another it might be Postgres, Redis, or a cloud resource*"). Sharper still: the blockquote **was not an example of a brief** — it was one run-on of repo facts that filled some of the six slots and skipped others, so as output-shaping it was *worse* than the abstract spec above it, modelling "recite what you found" rather than "fill the six slots." No replacement was written. If the form-shaping benefit is ever wanted, the correct artifact is a slot skeleton, not a repo snapshot.
- **Considered and kept:** the `ops-quirks.md` blockquote **including** its three parenthetical trap examples (registry round trips on a "fully cached" build, `-e VAR` deleting an image's `ENV` default, `pipefail` + an early-exiting pipe consumer). Under the plan's finding 8 these are **trigger stubs**, and `cobb` confirmed each is recognizable from the task surface *without already knowing the trap* — you know you're doing a cached build, passing `-e VAR`, or piping to an early-exiting consumer. This file is a **model implementation** of finding 8's progressive disclosure; do not "tidy" those parentheticals away. Also kept: the "**Orient before you act.**" principle despite orient appearing in a section heading, a principle, and step 1 — different decision points (*while* orienting vs. *whether* to orient at all), and its why-clause is unique to it.
- **Plan items:** the parking-lot item "Keep the graphmind-ai-lab example in the prompt lightweight" is **resolved by deletion** (marked ⚪ with the reasoning). Three further parking-lot items were stale in the same way and were corrected: the unifying-Compose item (partially delivered by `falkor-chat/compose.yaml`, re-scoped to extending coverage to `salesperson`; its dangling "the prompt's orientation example already frames it" pointer struck), the app-image item (re-scoped to the salesperson image only), and the CI item (still valid, annotated as a *re*-introduction pointing at the deleted `5a97821` file). The lore had rotted in the backlog as well as the prompt — independent corroboration of argument 2.
- **Residual class-6/7 inventory: ~22 w in 2,110 w (1.0%) — at the editorial floor**, on the same threshold as `analyst.md` after C3. Zero provenance of any kind remains; the residual is two judged-and-kept restatements of don't-generalize across three read surfaces. Separately flagged, not counted: the "Shared stateful services" FalkorDB parenthetical (~28 w) is now the file's **only** repo-specific claim and therefore its whole remaining rot surface — low risk (no version pin, no filename, hedged with the generic alternative), but it is the block to check first if this repo ever changes its shared store.
- **Verified:** `audit-team.sh` PASS (115 PASS, 0 FAIL); `cobb` §7 lint 0 blockers, 0 majors on this file.

## 2026-08-23 — Prompt-waste Stage B wave 2: two boilerplate blocks compressed to pilot shapes
- **What:** Interactive-commit-grant bullet and learning-capture intro/tail compressed to the pilot-validated wordings in `architect.md`/`coder.md` (`claude/docs/plans/prompt-waste-reduction.md` v4, §3 doctrine + Stage B). No CPG-freshness clause exists in this file.
- **Removed (class 5/6, already on record):** the grant's "same as before. Stakeholder decision, 2026-08-21 — see `kaizen/history.md`" — this file's 2026-08-21 grant entry; the tail's inbox-replacement sentence + ", exactly like the old inbox was" — this file's 2026-08-21 inbox-deletion entry; the intro's ":Agent node it's `PRODUCED`-linked to" mechanics restatement (mechanics live in the Cypher template below); the grant parenthetical's "— not spawned via `Agent`/`Task` as an isolated delegate" (moved into the carve-out sentence).
- **Gate (a) inventory — all preserved:** grant scope (own verified infra changes, explicit path), full never-list, delegated-subagent carve-out + audit check-8 tokens, the user-scoped parenthetical ("this graph resolves in every project — you are user-scoped"), the project-fact routing rule ("a fact about *a project* belongs in that project's docs, flagged in your report"), Cypher template + call line verbatim, "raw capture: `cobb` promotes; never edit your own definition".
- **Verified:** `audit-team.sh` PASS; cobb §7 lint pass.

## 2026-08-21 — Interactive-mode commit grant added (team-wide stakeholder decision)
- **What:** New Operating-principles bullet: when running interactively (`claude --agent devops`,
  a human present turn-by-turn — not a delegated subagent), may `git add`/`git commit` its own
  verified infra changes from the session, by explicit path, never bulk-staged/pushed/reset/
  rebased/amended; the grant does not apply when spawned as a delegated subagent.
- **Why:** Direct stakeholder ruling, 2026-08-21, after `tico` hit exactly this gap closing out a
  Mode-3 verification pass (its own commissioned artifacts left uncommitted, since only
  `tico`/`teco` had any commit authority). Rather than pin the fix to those two, the stakeholder
  ruled the exception should reach every agent, gated by invocation mode, not identity — full
  rationale, the `claude/AGENTS.md` rewrite, and the `audit-team.sh` check-8 redesign in
  `claude/cobb/kaizen/history.md`, 2026-08-21 entry.
- **Verified:** `bash claude/scripts/audit-team.sh` — clean, all 13 agents pass check 8.
- **Plan items:** none opened — direct implementation of an explicit stakeholder decision.

## 2026-08-21 — `kaizen/inbox.md` deleted (content already fully captured elsewhere)

- **What:** `cobb` deleted this agent's frozen `kaizen/inbox.md` (git history retains it in full, unaltered) as part of a team-wide cleanup of all 12 agents' frozen inboxes.
- **Why:** user-directed — "no point keeping [it] since it's already git history." Verified lossless first: `kaizen_team` (the shared graph every agent's raw capture routes through since 2026-08-20) was confirmed completely empty before any deletion — every entry any agent ever wrote there has already been distilled and cleared — and this file's own pre-migration content (if any) was already imported into the graph system verbatim back on 2026-08-20 (see that date's entry below). Full rationale and verification method: `claude/cobb/kaizen/history.md`, 2026-08-21 entry.
- **Verified:** see `cobb`'s entry (cross-agent verification, not repeated per file).
- **Plan items:** none opened — pure cleanup, no behavior change.

## 2026-08-20 — Learnings capture migrated to a working-memory graph (`kaizen_devops`), mirroring `graph-dba`
- **What:** The "Learning capture" closing-protocol section now writes a `:KaizenEntry` node
  directly into `kaizen_devops` (FalkorDB, via `mcp__cypher__query`) instead of appending to
  `kaizen/inbox.md`. `kaizen/inbox.md` is now a frozen historical snapshot — it had no
  pre-existing entries to migrate; its own header explains the freeze and gives the live-read
  query. The "a fact about *a project* belongs in that project's docs" distinction and the
  user-scoped-path note were both kept.
- **Why:** User-directed team-wide redesign ("I will migrate all agents to write their learnings
  to the graph like graph-dba"), reversing yesterday's file-based Learning-capture dedup (entry
  below) — the user determined the whole team should follow `graph-dba`'s existing graph-based
  capture pattern instead of the file-based inbox convention.
- **Plan items:** —

## 2026-08-19 — Learning-capture paragraph de-duplicated against the inbox's own header
- **What:** Trimmed the "Learning capture" paragraph: dropped "(fact, evidence, suggested home; format in the file header)" and "The inbox is raw capture — the team maintainer verifies and promotes entries" — both already stated verbatim in `kaizen/inbox.md`'s own header template (agent-maintenance skill §5), which the agent necessarily opens to append. Kept everything devops-specific: the discipline fact-kind clause, the user-scoped path note, the "a fact about *a project* belongs in that project's docs" distinction, and "never edit your own agent definition" (no write-guard clause — devops has no doc-scoped write guard). Behavior unchanged.
- **Why:** User-directed prompt-verbosity reduction, item 1 of the parked diagnosis (`cobb/kaizen/plan.md`) — the mechanics were literally duplicated (prompt + inbox header say the same thing), not just similar boilerplate; pointing at the file's own header removes the duplication without losing information, since the agent reads that file to act anyway.
- **Plan items:** —

## 2026-08-11 — Inbox distillation: 13 entries — new `ops-quirks.md` knowledge base (9 entries), 2 to `skills/joern-cpg/SKILL.md`, 1 folded into `claude/cobb/TESTING.md`, 1 discarded as already covered

- **What:** `cobb` processed all 13 entries in `devops/kaizen/inbox.md` (§5). (A prior version of
  this entry's header said "9 entries" — undercounted by 4, though every entry was in fact already
  described in the prose below; caught by `analyst`'s review, M-1 in
  `docs/reviews/kaizen-distillation-2026-08.md`.)
- **Promoted:** new on-demand knowledge base `claude/devops/ops-quirks.md`, pointed to from
  `devops.md`'s "Core expertise" header, carrying 7 written entries (9 raw inbox facts — the
  `.mcp.json` verification pair and the `CreatedSince`/image-ID-verification pair are each
  bundled into one written entry): `.mcp.json`/`claude mcp list`
  verification, `docker run` stdout-cleanliness, BuildKit's registry round trip for `FROM`
  metadata despite a full cache hit, `SIGTERM` being ignored by a bare-interpreter PID 1, build
  resumability after an interrupted `docker build`, `docker run -e VAR` deleting (not
  falling through to) an image's `ENV` default, and `set -euo pipefail` turning a legitimate
  early-exiting pipe consumer's SIGPIPE into a silent kill or a false "producer failed" — plus the
  `docker image ls` `CreatedSince` staleness-illusion trap, filed in the same entry as the image-ID
  verification rule it motivates. One more entry — the stdin-EOF stdio-MCP response-loss race —
  was folded into `claude/cobb/TESTING.md` instead (a testing-technique fact, not an ops fact).
- **Also promoted (Joern-specific, not Docker):** `pysrc2cpg` ships with no cold-start download,
  and a release's `.sha512` sidecar carries a build-relative path → `skills/joern-cpg/SKILL.md`
  Gotchas.
- **Discarded:** `joern-parse --version` throwing a stack trace — the skill already tells readers
  to use `joern --version` instead (line 18), so the negative case needed no separate callout.
- **Verified:** `bash claude/scripts/audit-team.sh` clean.
- **Docs touched:** `claude/devops/{devops.md,ops-quirks.md,kaizen/{history,inbox}.md}` ·
  `skills/joern-cpg/SKILL.md` · `claude/cobb/TESTING.md`.

## 2026-08-09 — Learnings-inbox entry promoted: `.mcp.json` approval scoping (C-319, cobb)
- **What:** The 2026-07-25 inbox entry "Claude Code MCP: `.mcp.json` discovery walks up to the
  repo root, but project-approval scope is keyed on the session's cwd" was verified (re-checked
  `~/.claude.json`'s `projects` map: still exactly one entry for this repo, none for
  `falkor-chat/`) and promoted into `skills/agent-standards/claude-code.md` §MCP → "Scopes,
  precedence, and the approval gate", citing this inbox entry's original evidence directly rather
  than re-deriving it. Entry removed from `kaizen/inbox.md`.
- **Why:** Backlog item C-319 asked for this durable fact to move from the inbox into the
  standards doc, matching the style/detail level of the doc's other observed-behavior bullets
  (e.g. the Lifecycle section's containerized-stdio-server note). Distillation performed by
  `cobb` per agent-maintenance skill §5.
- **Plan items:** none.

## 2026-07-31 — Boundary note: `tico` may hand off demo-environment lifecycle requests
- **What:** Added one bullet to Boundaries & handoffs: `tico` (stakeholder-facing, all three of its modes) may hand devops a demo-environment bring-up/cleanup request mid-conversation — a stakeholder wants to see a feature live or verify a manual's walkthrough. Framed explicitly as a plain lifecycle op: orient, boot/tear down what's asked, non-destructive by default, this agent's existing destructive-ops gate (hook + prompt-level judgment) applies exactly as it would to any other caller. No new capability, no frontmatter/tool/hook change — devops already does exactly this kind of work; the only thing new is who else may ask for it.
- **Why:** Reciprocal to the same-day `tico` change (`claude/tico/kaizen/history.md`, 2026-07-31): tico can now offer a live demo and delegates the actual bring-up/cleanup to devops via `Agent`. devops's own prompt needed the matching boundary note so the handoff isn't one-sided — tico owns *what* gets shown, devops owns *whether the environment is up*, and devops doesn't take instructions from tico about what to explain or document (the reverse boundary, made explicit).
- **Plan items:** none new here — the e2e verification is tracked on tico's side (K-006, `claude/tico/kaizen/plan.md`), since tico is the one initiating the delegation.

## 2026-07-27 — Unpinned from `model: opus` (team-wide)
- **What:** Removed the `model: opus` frontmatter line. The field is now absent, so the agent runs on Claude Code's default — `model` **defaults to `inherit`** (re-verified 2026-07-27 against `code.claude.com/docs/en/sub-agents`), i.e. the model the session/system default selects. No other frontmatter or body change.
- **Why:** User no longer wants the team locked to Opus. Model choice belongs at the session level (one decision, changeable with `/model`), not duplicated across 13 frontmatter files where it silently overrides whatever the user picked.
- **Plan items:** —

## 2026-07-24 — Description slimmed further (second team-wide token-cost pass)
- **What:** Frontmatter `description` compressed 654 → 572 chars (-12%): tightened phrasing, dropped restated detail, kept every routing/boundary clause. `claude/scripts/audit-team.sh` boundary-pair symmetry (devops↔graph-dba) re-verified green. No body/catalog change.
- **Why:** All 13 agents' descriptions are auto-injected into every session and subagent spawn; the roster grew to 13 (graph-dba, joern added) since the first pass on 2026-07-11, and per-agent `/context` output showed room to cut further. User-requested via a `/context` token audit.
- **Plan items:** none.

## 2026-07-24 — Frontmatter: `permissionMode: acceptEdits`
- **What:** Added `permissionMode: acceptEdits` to the frontmatter, matching the same-day change across the team (`coder`, `tdd-engineer`, `frontend-engineer`, `architect`, `qa-engineer`, `analyst`). File-edit/write approvals are session-scoped in Claude Code (unlike Bash approvals, which persist permanently per repo+command), so users otherwise have to re-grant write permission every session even with a global `Edit`/`Write` allow rule in `~/.claude/settings.json`.
- **Why:** Verified against current Claude Code docs (`hooks-guide.md` "Hooks and permission modes") that this doesn't weaken `devops`'s own guard: its `guard-destructive-ops.sh` hook matches Bash command patterns (volume wipes, `docker rm -f`, etc.), unrelated to `acceptEdits` (which only covers Edit/Write and common filesystem commands) — and `PreToolUse` hooks fire before any permission-mode check regardless, so a hook `"ask"` decision would survive even if the two overlapped.
- **Plan items:** none.

## 2026-07-12 — Learning-capture loop: kaizen inbox + closing protocol
- **What:** Added `kaizen/inbox.md` (append-only learnings inbox, seeded empty) and a "Learning capture" closing-protocol section to the prompt: durable, non-obvious environment facts discovered during runs are appended as dated, evidence-backed inbox entries; the agent never promotes its own entries.
- **Why:** Team-wide self-improvement loop (agent-maintenance skill §5, added the same day): capture is cheap and unreviewed during runs, promotion is curated — cobb periodically verifies each entry and routes it to the prompt, an on-demand knowledge base, or project docs. Requested by the user.
- **Plan items:** none.

## 2026-07-11 — Destructive-ops guard refactored to a shared core (no behavior change)
- **What:** `devops/hooks/guard-destructive-ops.sh` became a thin wrapper (mirroring the doc-guard wrappers) over the new shared core `claude/scripts/guard-destructive-ops.sh`, which takes the agent name for its escalation message; the matching logic is byte-identical. The core is now also wired into `graph-dba` and `qa-engineer` (cobb K-011 — they run against the same shared live FalkorDB). Wrapper + core verified: `docker volume rm` and `GRAPH.DELETE` escalate, read-only commands pass through.
- **Why:** Team-coherence certification (2026-07-11) found the destructive-ops gate protected the shared datastore only when devops was the actor; sharing the core follows the guard-doc-writes consolidation precedent.
- **Plan items:** none (cobb K-011).

## 2026-07-11 — Description slimmed (team-wide token-cost pass)
- **What:** Frontmatter `description` compressed from 1359 to 652 chars: capability lists tightened, reciprocal boundary prose reduced to short route-away clauses that still name the counterpart agents (audit check 6 boundary symmetry preserved — full pass green), and "how I work" detail dropped from the description since the prompt body already carries it. Routing semantics unchanged; no body/catalog changes needed.
- **Why:** All 12 agents' descriptions are auto-injected into every session and into every subagent spawn that carries the `Agent` tool; team-wide they cost 12,609 chars (~3.1K tokens) per injection. The pass cut them to 7,036 chars (~44%), saving ≈1,400 tokens per session/spawn with the same routing contract.
- **Plan items:** none.

## 2026-07-11 — Orientation example refreshed after start-script consolidation
- **What:** the infra-brief example in the prompt no longer describes "the two `start_falkordb.sh` scripts" (falkor-chat's named-volume variant vs. salesperson's ephemeral one, conflicting on host 6379) — `salesperson/start_falkordb.sh` is now a thin wrapper delegating to the canonical `falkor-chat/scripts/start_falkordb.sh`, so the example now states one container / one host port. No behavior change to the agent.
- **Why:** a repo redundancy audit (2026-07-11) consolidated the duplicated start scripts; the example's factual claim would otherwise have gone stale.
- **Plan items:** none.

## 2026-07-10 — Hook command made machine-independent (`$HOME` symlink path)
- **What:** the frontmatter `PreToolUse` hook command was rewired from the absolute repo path (`/home/<user>/prg/graphmind-ai-lab/claude/devops/hooks/guard-destructive-ops.sh`) to `$HOME/.claude/agents/devops/hooks/guard-destructive-ops.sh`, which resolves through the user-scope deployment symlink (`~/.claude/agents/devops` → the repo folder). Shell-form hook commands (no `args`) run via `sh -c`, so `$HOME` expands — verified 2026-07-10 against `code.claude.com/docs/en/hooks`. Resolution through the symlink confirmed (`test -x` passes).
- **Why:** the committed agent source leaked the user's personal home path into the repo; the symlink path is identical on any machine that follows the deployment convention (`~/.claude/agents/<name>` → `claude/<name>`), keeping the hook enforceable without machine-specific paths. (`${CLAUDE_PROJECT_DIR}` was rejected: the agents are user-scoped and must guard in any project, where the project dir isn't this repo.)
- **Plan items:** none.

## 2026-07-09 — FalkorDB image fact updated: edge → v4.18.11
- **What:** The grounding example in `devops.md` (and the devops rows in root `AGENTS.md` / `claude/AGENTS.md`) now cite the shared FalkorDB service as `falkordb/falkordb:v4.18.11` instead of `:edge`. The actual swap: both `start_falkordb.sh` scripts, `falkor-chat/compose.yaml`, and the CI service container pinned to `v4.18.11`; container recreated on the same `falkordb-data` volume (data intact), suites green (193/193 queries, 196 pytest).
- **Why:** Deployment pinned to the latest tagged release (user decision, 2026-07-09); the prompt's example must cite the real image or orientation drifts.
- **Plan items:** none.

## 2026-07-02 — K-001: harness-enforced destructive-op guard (PreToolUse hook)

- **What:** Added a **subagent-scoped `PreToolUse` hook** so the "approval-gate destructive/
  shared-state ops" guardrail is enforced by the harness, not just requested in the prompt. New
  script `devops/hooks/guard-destructive-ops.sh` returns `permissionDecision: "ask"` (escalates to
  the human with a permission dialog) for: `docker volume rm|prune`, `docker system prune`,
  `docker rm -f`/`--force`, `docker compose down -v|--volumes`, and Redis/FalkorDB `FLUSHALL`/
  `FLUSHDB`/`GRAPH.DELETE`. Wired via `hooks: PreToolUse → matcher: Bash` in `devops.md`
  frontmatter (scoped to this subagent only; auto-cleaned when it finishes). Added a note in the
  prompt's "guarded ops" principle that the hook is a backstop, not a substitute for judgment.
- **Why:** K-001. The guardrail was prompt-level (hopeful text); the mandate's own principle is
  "deterministic enforcement beats hope." User asked to implement it.
- **Verification (evidence-over-assertion):**
  - Verified the hook contract live (2026-07-02) against `code.claude.com/docs/en/hooks` (PreToolUse
    stdin `.tool_input.command`; JSON `hookSpecificOutput.permissionDecision` allows
    `allow|deny|ask|defer`) and subagent frontmatter `hooks:` schema against `/en/sub-agents`
    (supported, scoped to the subagent, `matcher` = tool name).
  - Tested the script over a 10-case matrix: all 6 destructive shapes → `ask`; all 4 safe ops
    (`docker build`, `ps`/`volume ls`, `compose up -d`, `logs -f`) → pass.
  - **Caught a real defect by testing:** `jq` is **not installed** on this WSL box (the doc examples
    assume it), so the first cut silently fell back to scanning the raw JSON, where end-anchored
    patterns (`-v$`, `FLUSHALL$`) failed because the token is followed by `"}}`. Fixed by (a) making
    extraction jq-optional (jq → python3 → raw payload) and (b) using non-alphanumeric token
    boundaries so patterns match on either a clean command or the raw payload.
- **Design choices:** `ask` (not hard `deny`) — matches "approval-gated," keeps the human in the
  loop rather than making the agent give up. Absolute script path in frontmatter because a subagent's
  cwd is the *target project*, not the agent's home — a relative path wouldn't resolve cross-project.
- **Portability caveat (logged):** the absolute path is machine-specific (same as the deploy
  symlink); re-point it on a new machine. `jq` optional but `python3` gives the cleanest extraction.
- **Plan items:** K-001 (done → moved out of the active table).

## 2026-07-02 — made project-agnostic (orient-first)

- **What:** Reworked the agent from a graphmind-ai-lab-specific prompt into a **portable, any-project**
  agent. Replaced the hard-coded "This repo's infra reality" section with an **"Orient yourself in
  the project first"** discipline: read the project's README / `AGENTS.md` / `CLAUDE.md` / `docs/` /
  infra & manifest files + confirm live state, form an *infra brief*, then act. The graphmind-ai-lab
  FalkorDB details are now a clearly-labeled **example** of what orientation yields, not baked-in
  truth. Genericized deps (cross-ecosystem, not Python-only), CI (any CI system), the shared-service
  guardrail (any datastore; FalkorDB as this repo's instance), and the DBA handoff (project's DBA /
  graph-dba here). Rewrote the `description` accordingly. Updated all three catalogs
  (`claude/README.md`, `claude/AGENTS.md`, root `AGENTS.md`) to describe the portable, user-scoped agent.
- **Why:** User will use this agent across **other projects**, and stressed it must read the project's
  README and docs to understand context. As authored it was over-fitted to this repo; since it's
  symlinked into **user scope** (`~/.claude/agents/`) it's already active in every project.
- **Key mechanism baked in:** a subagent auto-receives the `CLAUDE.md`/`AGENTS.md` memory hierarchy,
  but **README/`docs/` are NOT auto-loaded** — the prompt makes it actively `Read` them. (Per
  agent-standards `claude-code.md`, verified 2026-06-20.)

## 2026-07-02 — created

- **What:** Created the `devops` agent (`devops/devops.md`, `model: opus`) — a DevOps /
  platform-engineering persona owning environments, containerization, dependencies/config,
  automation, CI/CD, deployment, and observability for the monorepo. Seeded this kaizen pair and
  registered the agent in `claude/README.md`, `claude/AGENTS.md`, and the root `AGENTS.md`
  subagents table. Created the deployment symlink `~/.claude/agents/devops → claude/devops`.
- **Why:** User asked for "a devops persona, responsible for all our environments, containerization
  etc." No infra-focused agent existed (graph-dba covers the DB; coder/tdd-engineer the app code).
- **Design decisions (from user, via AskUserQuestion):**
  - **Name** = `devops` (role name, like `graph-dba`/`qa-engineer`) over a persona name.
  - **Autonomy** = *build + guarded ops*: freely authors/edits infra files and runs build/inspect
    commands; treats destructive/shared-state ops (volume wipes, `system prune`, touching the live
    shared FalkorDB) as approval-gated. Inherits all tools (no `tools:` allowlist) so the guardrail
    is prompt-level, not tool-level — see K-001.
  - **Scope** = *full DevOps remit* (containerization + dev-env + deps/secrets + CI/CD + deploy +
    observability), grounded in the repo's current Docker-only reality.
- **Grounding captured in the prompt:** the two competing `start_falkordb.sh` scripts
  (`falkordb/falkordb:edge`, ports 6379/3000, named container `falkordb-dev` + volume
  `falkordb-data` in falkor-chat vs. unnamed/ephemeral in salesperson; both bind 6379 so can't run
  together), Python ≥3.12, pyproject vs requirements split, and the greenfield gaps (no Compose, no
  CI, no Makefile, no app image builds).
