# Deprecated

Retired components, kept for reference. **Nothing here is maintained.**

A directory under `deprecated/` is code the repo has stopped developing but deliberately has not
deleted — because its behaviour is still worth reading (a parity reference for its replacement, a
worked example of an approach, an evidence trail behind a design decision). It is preserved, not
supported.

## Rules

- **Not maintained.** No bug fixes, no dependency upgrades, no refactors. Nothing here is expected
  to still run; a retired component's own docs describe it as it was when it was retired.
- **Not a precedent.** Do not copy patterns, conventions, or dependencies out of `deprecated/` into
  new work. A living component's docs are the source of truth for how this repo does things.
- **Read-only in practice.** The only edits that belong here are the retirement banner itself and a
  path fix made necessary by the move.
- **Not for anything still in use.** Something that still has users or callers is a live component
  and stays where it is.

## What is in here

| Directory | What it was | Retired because |
|---|---|---|
| `salesperson/` | Standalone Streamlit sales-assistant chatbot for "Pastel do Mau" — its own FalkorDB graph (`kg_pastel`) plus a LangChain/LangGraph agent, with an optional local LLM via LM Studio. See `deprecated/salesperson/README.md`. | It talks to an older, separate backend, not falkor-chat's workflow engine. It is superseded by a single business-facing salesperson UI built against the workflow-engine-backed `salesperson` agent that `falkor-chat/` hosts — specified in `docs/requirements/salesperson-ui.md` and planned in `docs/plans/salesperson-ui.md`. **That replacement is not built yet**; until it ships, the retired app is the only salesperson UI that exists in this repo. |
| `opencode/` | `opencode/`'s former custom agents — `agents/rpg.md`, `agents/coding-senior.md`, and `agents/severino/` (a full LM-Studio-backed local agent project with its own tests) — plus the five OpenCode-only `SKILL.md` packages they used (`skills/`: `comparison-driver`, `python-coding`, `skill-builder`, `user-preferences`, `write-tutorial`). See `deprecated/opencode/agents/severino/README.md` and `deprecated/opencode/skills/README.md`. | Retired to clear the way for `tank`, the new headless/local-model OpenCode agent meant to eventually become the repo's one canonical OpenCode identity — see `opencode/docs/requirements/devops-opencode-headless.md`. `opencode/docs/manuals/local-llm.md` (the general LM-Studio-with-OpenCode walkthrough these agents used) was **not** retired and still lives at that live path. |

The retired Streamlit app is also the parity reference for its replacement: `docs/plans/salesperson-ui.md`
§2.4 cites its cart, profile and session code for the behaviour the new UI has to match.
