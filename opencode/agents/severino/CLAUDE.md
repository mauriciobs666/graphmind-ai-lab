# CLAUDE.md — Severino project

Brief for Claude (and any other AI agent) working in this folder. Read first.

## What this is

A configuration project for **Severino**, a local OpenCode agent backed by LM Studio. Currently configured as a **read-only coding advisor** — he reads, reviews, and explains code but cannot modify it (mutation tools stay off; he proposes changes as snippets/diffs for the user to apply).

- Backend: Nemotron 3 Nano 4B via LM Studio at `http://localhost:1234/v1`, loaded with 64K context. (Ministral 3B 2512 is kept in `opencode.json` as a fallback — Nemotron has proven noticeably better in practice.)
- Identity lives entirely in `agent.severino.prompt` inside `opencode.json`.
- Tools enabled: `read`, `glob`, `grep` only. All mutation tools (`bash`, `edit`, `write`) deliberately off.

## Do / don't

- **Do** edit `opencode.json` to change Severino's persona, model, or tool flags.
- **Do** keep `README.md`, `tutorial.md`, and the persona prompt consistent when any one of them changes — these three drift together.
- **Don't** add a `name` field to the agent definition — the JSON key (`severino`, lowercase) is the identifier; a `name` field shadows it and breaks `--agent` lookup.
- **Don't** rename the top-level `agent` key to `agents` — singular is required.
- **Don't** try to invoke Severino to do work in this conversation; he's the artifact being configured, not the agent doing the configuring.

## Schema gotchas (learned the hard way)

- Top-level key is `agent` (singular).
- Custom-agent identifier comes from the JSON key, not a `name` field.
- Use `prompt` for the system prompt, not `system`.
- `model` is `<provider-key>/<model-id>`; splits on the first `/`.
- LM Studio's `apiKey` value is a dummy string — required by the adapter, ignored by the server. Safe to commit.
- LM Studio Context Length must be **≥16K** (32K recommended) or OpenCode's system prompt overflows with `n_keep >= n_ctx`.

## Files in this project

| File | Role |
| --- | --- |
| `opencode.json` | The actual agent config — provider, model, persona prompt, tool flags |
| `README.md` | User-facing setup guide + troubleshooting |
| `IMPROVEMENT-PLAN.md` | Historical capability plan from an earlier full coding-agent design; aimed at a read/write/bash agent, so mostly aspirational for today's read-only advisor |
| `tests/` | Eval/kaizen harness — `run.sh` sends `cases/*/prompt.md` through `opencode run`, writes answers to `outputs/`, diffs vs blessed `baseline/`. See `tests/README.md`. |
