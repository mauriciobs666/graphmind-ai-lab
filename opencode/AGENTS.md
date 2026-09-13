# opencode/ — context for AI agents working here

`opencode/` is greenfield: **`agents/tank/`** is the only live agent — a headless, local-model
DevOps variant that runs a read-only health/hygiene check by default and can bring up/tear down
one Docker-Compose environment it itself started (`agents/tank/README.md`). Former custom agents
(`rpg`, `coding-senior`, `severino`) and OpenCode-only `SKILL.md` packages are retired to
`deprecated/opencode/` — read-only history, never extended or copied from. A new OpenCode-only
skill goes in `opencode/skills/` fresh (currently empty).

**Don't hand-edit `tank`'s persona.** `agents/tank/opencode.json`'s `prompt` field live-includes
`claude/devops/devops-persona.md` (`{file:../../../claude/devops/devops-persona.md}`) plus a
small headless-specific addendum. To change what `tank` *is*, edit `devops-persona.md` — it's the
same shared source `claude devops` generates from (`claude/devops/scripts/sync-persona.sh`). Edit
the `opencode.json` addendum only for genuinely headless-only behavior, never persona substance,
and never `claude/devops/devops.md` directly (generated; overwritten by the sync script).

For local LM Studio setup (loading a model, starting the server, picking a model size), see
[`opencode/docs/manuals/local-llm.md`](docs/manuals/local-llm.md).
