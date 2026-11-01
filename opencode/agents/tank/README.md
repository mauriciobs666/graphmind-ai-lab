# tank — headless, local-model DevOps agent

A headless OpenCode variant of the `devops` role: no live Claude Code session, no human present
mid-run, powered by a small local model via [LM Studio](https://lmstudio.ai/). By default it runs
a **read-only** health/hygiene check; on request it can bring up or tear down **one**
Docker-Compose environment it itself brought up. Everything else is auto-denied and reported,
never attempted — see [Permission design](#permission-design-why-the-wrapper-scripts-are-safe)
below for why that's a mechanical guarantee, not a prompt instruction.

Full design and rationale: `opencode/docs/plans/devops-opencode-headless.md`. Security review
(two passes, both closed): `opencode/docs/reviews/devops-opencode-headless.md`.

## Prerequisites

- [OpenCode](https://opencode.ai/) installed (`npm install -g opencode-ai`) and on your `PATH`.
- [LM Studio](https://lmstudio.ai/) installed, with the local server running at
  `http://localhost:1234/v1`, serving `tank`'s model, `mistralai/ministral-3-3b`. General
  walkthrough: [`opencode/docs/manuals/local-llm.md`](../../docs/manuals/local-llm.md); confirm
  with `curl http://localhost:1234/v1/models`.
  - **Load ministral by hand, before a run:** ministral has no per-model default to rely on, so
    load it at a context of at least `TANK_MIN_CONTEXT_LENGTH` (20480) and parallel
    `TANK_REQUIRED_PARALLEL` (1), both in `scripts/model-preflight.sh`:
    `lms load mistralai/ministral-3-3b -c 20480 --parallel 1 --gpu max --identifier
    mistralai/ministral-3-3b -y`. Never load it while `lms ps` already lists it.
  - **Unload it before a demo and before returning to everyday use:** falkor-chat's
    `ensure_demo_models` refuses any ministral context other than 8192
    (`falkor-chat/scripts/ensure_demo_models.py`), and the preflight in turn refuses a ministral
    loaded at the demo's 8192. The two setups cannot share one copy; switch with `lms unload
    mistralai/ministral-3-3b`.
  - **Preflight:** each outer script runs `scripts/model-preflight.sh` before `opencode run` and
    exits 1 with a fix-it message when the copy breaks the above. It only reads; it never loads
    or unloads. `TANK_MODEL_PREFLIGHT=0` skips it (for an LM Studio without `/api/v1`).
- Docker + Docker Compose, for whatever environment(s) `tank` is asked to manage.

If your LM Studio model ID differs from what's in `opencode.json`'s `provider.lmstudio.models`,
edit that map to match — model IDs must match exactly, slashes included.

## Running tank

Always go through the outer scripts in `scripts/` — never call `opencode run` directly. Each
script `cd`s into this directory first (`opencode/agents/tank/`), because OpenCode only loads a
project-scoped `opencode.json` from its own directory (verified live against the installed
binary; plan §2.3).

| Script | What it does |
|---|---|
| `./scripts/health-check.sh` | Read-only health/hygiene check: container/service status, disk usage, dangling images/volumes, build-cache usage. Nothing is ever modified. |
| `./scripts/bring-up.sh <slug>` | Bring up one environment registered in `environments.json` (e.g. `./scripts/bring-up.sh falkor-chat`). Refuses if already running. |
| `./scripts/tear-down.sh <slug>` | Tear down an environment `tank` itself brought up. Refuses if it isn't the one that started it, or if its ownership marker is stale. |

These are **outer** scripts — thin wrappers that just start `tank` with a fixed message. They are
not part of the security boundary themselves and don't touch `docker`; they're distinct from the
**inner** wrapper scripts below, which `tank`'s own `bash` tool invokes.

Each also passes `opencode run` a fixed `--title` — that's not cosmetic. Without it, OpenCode
makes a second, separate background call to auto-generate a human-readable session title, which
reliably fails against `mistralai/ministral-3-3b`'s strict-alternation chat template (`Jinja
Exception: ... roles must alternate...`, HTTP 500, silently swallowed — harmless to `tank`'s own
result, but a wasted failing call and log noise on every invocation). This is a known upstream
OpenCode issue, not specific to this repo — the auto-title (and session-compaction) prompts send
two consecutive user-role messages, which any strict Mistral-family Jinja template rejects
(OpenCode issues [#19840](https://github.com/anomalyco/opencode/issues/19840),
[#18998](https://github.com/anomalyco/opencode/issues/18998); no upstream fix as of 2026-09-13).
Supplying `--title` up front skips that call entirely (verified live, 2026-09-13). If you ever
invoke `tank` a different way, pass `--title` yourself to avoid it — there's no `opencode.json`
equivalent (checked the config schema; the only lever is this CLI flag).

## Adding a second environment

Add one entry to [`environments.json`](environments.json): `composeFile`/`projectDirectory`
(repo-root-relative paths) and a human-readable `label`. Nothing else changes — no
`opencode.json` edit, no script edit. The three inner wrapper scripts
(`scripts/compose-status.sh`, `compose-up.sh`, `compose-down.sh`) and `tank`'s `permission.bash`
allow-list are already generic across every entry in this file; that's what makes `tank` able to
manage any compose-based environment under the repo without reopening its permission table (plan
§3.2/§3.6). Review the addition like any other config change — `environments.json` is the trust
boundary that decides what `tank` can reach at all.

## The state marker

One file per environment, `state/<slug>.json`, written and read **only** by
`scripts/compose-up.sh`/`compose-down.sh`'s own code — never by `tank` via an OpenCode tool call
(`tank`'s `write`/`edit` tools are disabled entirely; see `opencode.json`). It records
`broughtUpAt`, a session nonce, the script's PID, and the host — a staleness/collision safeguard,
not a cryptographic guarantee (plan §3.3). `compose-down.sh` refuses to act on a marker older than
`TANK_MARKER_MAX_AGE_SECONDS` (default `21600`, i.e. 6 hours); override by exporting that variable
before running `tear-down.sh`. `state/` is gitignored except `.gitkeep`
(`opencode/.gitignore`) — it's runtime bookkeeping, not a deliverable.

## Changing tank's persona

Edit **`claude/devops/devops-persona.md`** — that's the one canonical source. Never:

- `opencode.json`'s `prompt` field addendum (the text after the `{file:...}` include) — that's
  headless-specific behavior only (how `tank` narrates a refusal, the bring-up/teardown protocol),
  never persona substance. Add to it only for something genuinely specific to running headless.
- `claude/devops/devops.md` directly — it's generated (marker-wrapped) from `devops-persona.md` by
  `claude/devops/scripts/sync-persona.sh`; a hand-edit inside the markers is overwritten the next
  time that script runs.

`opencode.json`'s `prompt` field reads `devops-persona.md` **live**, via
`{file:../../../claude/devops/devops-persona.md}` — there's no regeneration step on this side; an
edit to `devops-persona.md` takes effect the next time `tank` runs.

## Permission design (why the wrapper scripts are safe)

`tank`'s `permission.bash` allow-list never matches a raw `docker compose` invocation — the only
compose-facing commands it allows are the three inner wrapper scripts
(`scripts/compose-status.sh`, `compose-up.sh`, `compose-down.sh`), each anchored to its exact,
absolute path, written as `{env:HOME}/prg/graphmind-ai-lab/opencode/agents/tank/scripts/…` —
OpenCode substitutes `{env:VAR}` in keys too (an unset variable becomes empty, silently), so a
clone at any other location must edit those three keys to match. Each allow pattern ends in a trailing `*` (e.g. `.../compose-down.sh *`), which
looks broad, but is safe *only* because the pattern's one job is "can this script be reached at
all" — the script itself is what validates the argument: exactly one token, looked up against
`environments.json` by true key equality, never substring/prefix/glob. Anything else (extra
tokens, a smuggled second `-f`/`--project-directory` pair, a `../`-shaped value) is refused inside
the script, before it ever touches `docker`, regardless of what the permission table let through —
independently verified live, twice, in `opencode/docs/reviews/devops-opencode-headless.md`.

**Do not "simplify" this by tightening the trailing `*`.** It isn't slack that needs closing —
the real validation lives in the scripts, not the permission table, and narrowing the pattern
buys nothing while risking breaking a legitimate call. If you're ever tempted to add a new
compose-facing allow pattern here that isn't one of these three exact script paths, don't — that's
exactly the pattern class the original design shipped with and had to redesign away from after a
live-reproduced bypass (`opencode/docs/plans/devops-opencode-headless.md` §3.2, "Revised
2026-09-12").

## Configuration reference

See [`opencode.json`](opencode.json). Key pieces (same shape as the retired `severino` project,
`deprecated/opencode/agents/severino/opencode.json`):

| Field | Purpose |
|---|---|
| `provider.lmstudio.npm` | `@ai-sdk/openai-compatible` — adapter for any OpenAI-shaped HTTP server |
| `provider.lmstudio.options.baseURL` | LM Studio server URL (must include `/v1`) |
| `provider.lmstudio.options.apiKey` | Dummy string (`lm-studio`). The adapter requires it; LM Studio ignores it |
| `provider.lmstudio.models` | Map of model IDs LM Studio reports (`mistralai/ministral-3-3b`, `qwen/qwen3-4b-2507`) |
| `agent.tank.mode` | `primary` — selectable directly with `opencode --agent tank` |
| `agent.tank.model` | `lmstudio/mistralai/ministral-3-3b` (default) |
| `agent.tank.prompt` | `{file:...}`-included shared persona + the headless-specific addendum |
| `agent.tank.tools` | `bash`/`read`/`glob`/`grep` on; everything else (`write`, `edit`, `webfetch`, `websearch`, `task`, `skill`, `todowrite`, `question`) off |
| `agent.tank.permission.bash` | The three-tier allow-list described above |

> Two non-obvious things, inherited from `severino`'s schema lessons: the top-level key is
> **`agent`** (singular), and `tank` deliberately has **no `name` field** — the JSON key (`tank`)
> is what `--agent` looks up; adding `name` would shadow it.

## Choosing a model

`mistralai/ministral-3-3b` is `tank`'s model: it completed the deny+report flow correctly and
quickly in live testing (plan §2.3), and the `qwen/qwen3-4b-2507` trial failed (qwen made up
findings in all three TP-001 runs; `docs/test-reports/standard-local-model-report.md`). It is
the second accepted exception to the one-chat-model rule, alongside the salesperson demo, so
running `tank` means a second chat model in memory.

## Troubleshooting

General LM-Studio-with-OpenCode symptoms (`Unrecognized key: agents`, `Agent not found`, `API key
required`, `Connection refused`, `Model not found`, `n_keep >= n_ctx`, `Only user and assistant
roles are supported`, slow reasoning models, slow JIT-loaded first response) are covered in
[`opencode/docs/manuals/local-llm.md`](../../docs/manuals/local-llm.md) § *Troubleshooting*.
`tank`-specific:

| Symptom | Cause / Fix |
|---|---|
| Agent picker doesn't show `tank` | You ran `opencode` without going through the outer scripts (or without `cd`-ing into this directory first). Project-scoped config is only loaded from the current working directory. |
| A run hangs indefinitely with no output | Something in `permission.bash` resolved to `"ask"` — that hangs forever with no human present (verified live, plan §2.3). This shouldn't happen with the shipped table (`allow`/`deny` only); if you've edited it, check for a stray `"ask"`. |
| `tank` refuses a bring-up/teardown you expected to work | Check the exact error it reported — `not a known environment` (fix `environments.json`), `already running, not started by me`, `not something I brought up`, or `marker present but stale` (see [The state marker](#the-state-marker)) all mean exactly what they say; `tank` never guesses past a refusal. |
| Outer script exits 1 with `REFUSED: <k> copies of <id> loaded (<ids>)` | More than one copy of the model is loaded. Unload the extra one(s): `lms unload <id:n>`. |
| `REFUSED: <id> loaded at <c>, tank needs >= <min>` | The copy was loaded with a smaller context. When nothing else is using it, `lms unload <id>` and reload it by hand as in Prerequisites. |
| `REFUSED: <id> loaded with parallel <p>, tank needs <required>` | Same fix as the row above. |
| `<id> not in LM Studio` / `not reachable at <root>` | Fix the id in `opencode.json`, or start LM Studio's server. |
