# Claude Agents

Custom [Claude Code subagents](https://code.claude.com/docs) for this repo. Each lives in its own folder with the agent source (`<name>.md`, Markdown + YAML frontmatter) and a `kaizen/` folder holding its improvement plan and change history.

| Agent | What it does | When to use it |
|-------|--------------|----------------|
| [`teco`](./teco/teco.md) | Technical coordinator; decomposes a multi-step goal into a sequenced work breakdown and **delegates each piece to the right specialist** (architect, coder, tdd-engineer, frontend-engineer, analyst, data-scientist, qa-engineer, graph-dba, devops, cobb), then integrates results — routing QA passes to `qa-engineer` and environment blockers to `devops`. Also the team's **documentation curator**: scans each goal for documentation impact, makes doc updates part of every unit's done-condition (the unit's owner writes them), and verifies the docs were actually updated at integration. Holds **independent review** as the default: every significant deliverable gets a reviewer other than its producer (`analyst` statically, `data-scientist` for ML methodology, `qa-engineer` at acceptance) — skipping the gate is the justified exception. Hybrid: drives execution but pauses to the user at decision points. Coordinates — doesn't design or code itself; **`Write`/`Edit` scoped to its coordination doc is harness-enforced** by a subagent-scoped `PreToolUse` hook (`teco/hooks/guard-coordination-doc-writes.sh`). | A task that spans several steps/specialties, needs orchestration, or is an end-to-end feature delivery. |
| [`tico`](./tico/tico.md) | Conversational **product owner** — a **first-order agent**: run it as the main-session agent (`claude --agent tico`) and it interviews you live about a feature request, editing the **feature requirements document** (`<component>/docs/requirements/<slug>.md`) as the conversation progresses — intent, problem, user stories, testable requirements, out-of-scope, acceptance criteria; "Ready for design" only on your explicit confirmation. Product altitude only: WHAT/WHY, never HOW. **Requirements-doc-only writes are harness-enforced** by a `PreToolUse` hook (`tico/hooks/guard-requirements-doc-writes.sh`) — frontmatter hooks fire in main-session mode too. The requirements half of a tico→architect handoff; not meant to be delegated (as a subagent it degrades to one interview round per invocation). | Capturing requirements, user stories, or acceptance criteria for a vague or unwritten feature request, before any design or code — launch with `claude --agent tico`. |
| [`architect`](./architect/architect.md) | Software architect; investigates the codebase, weighs trade-offs, and produces a step-by-step implementation plan/spec — **without editing code**. Writes the plan to `<component>/docs/plans/<slug>.md` by default and returns the path + a ready-to-implement summary (lossless handoff). **Read-only on code is harness-enforced:** a subagent-scoped `PreToolUse` hook (`architect/hooks/guard-plan-doc-writes.sh`) gates any `Write`/`Edit` outside `docs/plans/` to human approval. The planning half of an architect→coder handoff. | Wanting a design, an approach, an impact analysis, or a plan before any code is written. |
| [`coder`](./coder/coder.md) | Software engineer who implements an approved plan/spec end-to-end — clean, idiomatic, well-tested code following the repo's conventions; keeps the suite green. The implementation half of an architect→coder handoff. | Building from a ready plan/spec or clear task — the efficient route when a detailed plan exists. (Bug fixes / safety-net refactors / test work → `tdd-engineer`; UI-heavy front-end work → `frontend-engineer`.) |
| [`cobb`](./cobb/cobb.md) | Practitioner of agentic development; deep, current knowledge of Claude Code, Kiro, and OpenCode agent formats and the cross-tool standards (`AGENTS.md`, Agent Skills). | Designing, authoring, reviewing, porting, or debugging agents, subagents, skills, steering docs, slash commands, hooks, or system prompts. |
| [`graph-dba`](./graph-dba/graph-dba.md) | Graph database administrator & data architect specialized in **FalkorDB** (Redis-module, GraphBLAS sparse-matrix engine; RedisGraph successor; built for GraphRAG). Covers its OpenCypher dialect, modeling, vector/full-text indexing, constraints, multi-graph tenancy, in-memory sizing, replication/clustering, and tuning via `GRAPH.PROFILE`. Fluent in the wider LPG world (Neo4j, openCypher, GQL) for porting. Designs the deployment; container/Compose plumbing → `devops`. | Designing a graph data model, writing/optimizing FalkorDB Cypher, indexes/constraints, FalkorDB deployment (RAM sizing, persistence, replication, Redis Cluster), tuning slow traversals, bulk ingestion/migration, building a GraphRAG/knowledge-graph layer, or FalkorDB ops. |
| [`tdd-engineer`](./tdd-engineer/tdd-engineer.md) | Software engineer who implements features and fixes strictly via Test-Driven Development (red → green → refactor), keeping the suite green at every step. | Where test-first is the efficient path: bug fixes (repro test first), refactoring with a safety net, adding/improving tests, clear-contract features. (Detailed plan ready to execute → `coder`.) |
| [`frontend-engineer`](./frontend-engineer/frontend-engineer.md) | Senior front-end engineer — the **UI-depth implementer**. Web platform first (semantic HTML, modern CSS, JS/TS, React & peers), fluent in Python-native UIs like **Streamlit**. **Orients first:** reads the project docs and the actual UI stack before writing a line; matches existing conventions. Owns components, layouts, styling, client-side state & data fetching, accessibility, responsive behavior, and front-end performance — tests alongside, and verifies **in the running UI** with honest reporting. Consumes an architect plan by path like the other implementers. Back-end/API/non-UI code → `coder`/`tdd-engineer`; acceptance QA → `qa-engineer`; build/deploy infra → `devops`. | Building or changing a user interface: components, pages, styling/design-system work, forms, accessibility fixes, responsive/cross-browser issues, front-end performance, or a Streamlit screen. |
| [`qa-engineer`](./qa-engineer/qa-engineer.md) | QA / functional-testing engineer; reasons about risk to build a test strategy, writes it to a versioned **test plan** (`docs/test-plans/<kebab>.md`), executes it (authors automated functional/acceptance tests, runs existing suites, **and** drives the running app black-box), then delivers a **test report** (`docs/test-reports/<kebab>-report.md`) with results, defects, and feedback. Behavior/acceptance altitude — the black-box complement to `tdd-engineer`'s unit-level TDD; static plan/code review (no execution) → `analyst`. | Wanting a test strategy/plan, functional/acceptance/integration/e2e/exploratory testing, a QA pass on a feature or release, or a written report of what was tested and what broke. |
| [`analyst`](./analyst/analyst.md) | Systematic, experienced developer acting as a pure **reviewer and diagnostician** of plans and source code — **without changing either**. Reviews an architect plan before implementation (grounding against the real codebase, completeness, soundness, simpler alternatives, test strategy) and code/diffs/modules (correctness → tests → convention fit → clarity → security/perf), plus plan↔code conformance when given both; also performs **root cause analysis** on defects/failing tests/regressions — reproduces when possible and traces symptom → causal chain → root cause with evidence, to `<component>/docs/reviews/<slug>-rca.md` with a suggested fix + prevention. Reviews deliver severity-ranked, evidence-backed findings with a concrete suggested improvement each, under a verdict (approve / approve with suggestions / needs changes), written to `<component>/docs/reviews/<slug>.md` and handed off by path. **Review-only writes are harness-enforced** by a subagent-scoped `PreToolUse` hook (`analyst/hooks/guard-review-doc-writes.sh`). Findings route to their owners (fixes → `coder`/`tdd-engineer`, design rework → `architect`, black-box verification → `qa-engineer`). | Wanting a second opinion on a plan before building it, a code review of a change or module, improvement suggestions, or a root-cause investigation of a bug or failure before fixing it — especially as a review gate in an architect→coder pipeline. |
| [`data-scientist`](./data-scientist/data-scientist.md) | AI/ML/data-science specialist working as an **advisory scientist** — designs the ML method (model/embedding selection, prompt/context strategy, retrieval & chunking, RAG/GraphRAG evaluation design, golden sets & LLM-as-judge validity, experiment/A-B design, metric choice, statistical rigor, data quality/leakage) and judges methodology — **never implements**. Works alongside `architect` (method note at `<component>/docs/plans/<slug>-ml.md`, folded into the plan) and `analyst` (methodology review at `<component>/docs/reviews/<slug>-ml.md`, same verdict scale — the ML complement to the general static review); every recommendation ships with an evaluation design (metric, data, threshold). **Advisory-only writes are harness-enforced** by a subagent-scoped `PreToolUse` hook (`data-scientist/hooks/guard-ds-doc-writes.sh`). Implementation → `coder`/`tdd-engineer`; in-graph vector mechanics/Cypher → `graph-dba`. | Choosing a model or embedding, designing or judging an evaluation for an LLM/RAG feature, defining quality metrics, designing an experiment or A/B test, checking statistical validity, or diagnosing why a model/retrieval pipeline underperforms. |
| [`devops`](./devops/devops.md) | DevOps / platform engineer; owns environments, containerization, and the delivery lifecycle **in any project**. **Orients first** — reads the project's README/`AGENTS.md`/`docs/`/infra files to learn the real stack before acting (never assumes a toolchain). Docker/Compose, reproducible dev environments, cross-ecosystem deps/venvs, `.env`/secrets hygiene, automation scripts, CI/CD, deploy/release, observability. **Guarded ops (harness-enforced):** builds/edits infra freely, but destructive/shared-state ops (volume wipes, `system prune`, `docker rm -f`, `compose down -v`, Redis/FalkorDB flush) are gated to human approval by a subagent-scoped `PreToolUse` hook (`devops/hooks/guard-destructive-ops.sh`). Defers DB/data-model design → the project's DBA (graph-dba for FalkorDB), app code → coder/tdd-engineer, agents → cobb. *(User-scoped: available in every project, not just this repo.)* | Dockerfiles/Compose, spinning up or debugging a container/service, setting up/fixing a dev environment, managing deps/venvs/secrets, writing automation, designing CI/CD or deployment, or hardening infra. |

## Kaizen

Each agent carries a living improvement plan and change log:

- `teco/kaizen/` — [plan](./teco/kaizen/plan.md) · [history](./teco/kaizen/history.md)
- `tico/kaizen/` — [plan](./tico/kaizen/plan.md) · [history](./tico/kaizen/history.md)
- `architect/kaizen/` — [plan](./architect/kaizen/plan.md) · [history](./architect/kaizen/history.md)
- `coder/kaizen/` — [plan](./coder/kaizen/plan.md) · [history](./coder/kaizen/history.md)
- `cobb/kaizen/` — [plan](./cobb/kaizen/plan.md) · [history](./cobb/kaizen/history.md)
- `graph-dba/kaizen/` — [plan](./graph-dba/kaizen/plan.md) · [history](./graph-dba/kaizen/history.md)
- `tdd-engineer/kaizen/` — [plan](./tdd-engineer/kaizen/plan.md) · [history](./tdd-engineer/kaizen/history.md)
- `frontend-engineer/kaizen/` — [plan](./frontend-engineer/kaizen/plan.md) · [history](./frontend-engineer/kaizen/history.md)
- `qa-engineer/kaizen/` — [plan](./qa-engineer/kaizen/plan.md) · [history](./qa-engineer/kaizen/history.md)
- `analyst/kaizen/` — [plan](./analyst/kaizen/plan.md) · [history](./analyst/kaizen/history.md)
- `data-scientist/kaizen/` — [plan](./data-scientist/kaizen/plan.md) · [history](./data-scientist/kaizen/history.md)
- `devops/kaizen/` — [plan](./devops/kaizen/plan.md) · [history](./devops/kaizen/history.md)

`cobb` additionally maintains [`cobb/TESTING.md`](./cobb/TESTING.md) (agent testing standards).
`graph-dba` additionally maintains [`graph-dba/falkordb-quirks.md`](./graph-dba/falkordb-quirks.md) — a dated, live-verified FalkorDB quirks knowledge base for this lab's edge build, kept out of the always-on prompt and loaded on demand.

## Skills

Skills were unified into the repo-root [`skills/`](../skills/) home — see [`skills/README.md`](../skills/README.md) for the full catalog. `cobb` relies on two of them:

- [**`agent-maintenance`**](../skills/agent-maintenance/SKILL.md) — kaizen/documentation/file-location/drift-audit machinery `cobb` follows when it creates/edits/reviews an agent or skill, plus the **team-coherence certification** pass (rosters, handoff contracts, hook parity). Its deterministic half is scripted: [`scripts/audit-team.sh`](./scripts/audit-team.sh) (read-only; exit 1 on drift).
- [**`agent-standards`**](../skills/agent-standards/SKILL.md) — perishable per-tool reference specifics (frontmatter fields, paths, inclusion modes) for Claude Code, Kiro, and OpenCode, each `Verified:`-stamped.

## Deployment

These agents live in this repo but run from Claude Code's config dir via symlinks:

- **Agents:** `~/.claude/agents/<name>` → `claude/<name>` (one symlink per agent folder).
  - **Hook gotcha (`devops`, `architect`, `teco`, `tico`, `analyst`, `data-scientist`):** their `PreToolUse` guard hooks are referenced by
    **absolute paths** in each agent's frontmatter (`claude/devops/hooks/guard-destructive-ops.sh`,
    `claude/architect/hooks/guard-plan-doc-writes.sh`,
    `claude/teco/hooks/guard-coordination-doc-writes.sh`,
    `claude/tico/hooks/guard-requirements-doc-writes.sh`,
    `claude/analyst/hooks/guard-review-doc-writes.sh`,
    `claude/data-scientist/hooks/guard-ds-doc-writes.sh`). On a new machine or a different clone path,
    re-point those paths (and re-create the symlinks). The scripts prefer `jq`, fall back to
    `python3` — install one for clean extraction. (devops kaizen K-004.)
  - **tico runs first-order:** launch it as the main-session agent — `claude --agent tico` — so the
    interview is a live conversation (frontmatter hooks fire in main-session mode too, so its guard
    still applies; invoking it as a subagent degrades it to one interview round per invocation).
- **Skills:** now in the repo-root [`skills/`](../skills/) home, deployed via `~/.claude/skills` →
  `skills/` (whole-dir symlink; all 7 skills visible to Claude Code). Also symlinked into OpenCode
  and Kiro — see [`skills/README.md`](../skills/README.md#deployment).

So edits here are picked up live, and the definitions stay version-controlled.

### WebFetch / WebSearch allowlist (for `cobb`)

`cobb`'s mandate is to **verify version-sensitive specifics against live official docs** (and the
[`agent-standards`](../skills/agent-standards/SKILL.md) skill re-checks its `Verified:` stamps the
same way). To let those fetches run without a confirmation prompt, the official doc domains are
allowlisted in **`~/.claude/settings.json`** (user scope) under `permissions.allow`:

```json
{
  "permissions": {
    "allow": [
      "WebFetch(domain:code.claude.com)",
      "WebFetch(domain:platform.claude.com)",
      "WebFetch(domain:docs.anthropic.com)",
      "WebFetch(domain:kiro.dev)",
      "WebFetch(domain:opencode.ai)",
      "WebSearch"
    ]
  }
}
```

Notes:
- **User scope, so it's session-wide** — every agent in the session (not just `cobb`) can fetch
  these five hosts and run `WebSearch` unprompted; any other URL still prompts. For a strictly
  cobb-only allowlist you'd use a subagent-scoped `PreToolUse` hook instead (more machinery; see
  cobb's kaizen history).
- This file is **personal/global and not in this repo** — re-add the block on a new machine.
- Takes effect on new sessions; open `/permissions` or restart if it doesn't load immediately.
- Match the *final* host: `WebFetch` returns cross-host redirects to be re-fetched, so the
  redirect destination must also be listed (e.g. `docs.anthropic.com`).

## Conventions

- **Folder per agent:** `<name>/<name>.md` is the source; `<name>/kaizen/{plan,history}.md` track improvements.
- **Frontmatter** drives routing: the `description` says *what the agent does and precisely when to use it* so Claude Code can auto-delegate.
- When you add, edit, rename, or remove an agent, keep this catalog and `AGENTS.md` (the agent-context file; `CLAUDE.md` is a `@AGENTS.md` import stub) in sync, and update the agent's `kaizen/` files.
