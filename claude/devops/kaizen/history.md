# Kaizen — Change History: devops

> Dated log of actual changes to the `devops` agent. Most recent first.

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
