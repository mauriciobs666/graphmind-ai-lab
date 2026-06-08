# Kiro (spec-driven, agentic IDE) — reference

> **Verified: 2026-06-07** against `kiro.dev/docs/steering`. Specs/Hooks detail on the
> 2026-05-31 baseline — due for refresh. Re-verify before relying on an exact key.

## Three building blocks

**Steering Docs**, **Specs**, **Hooks**.

## Steering files

- **Location:** `.kiro/steering/` (workspace) or `~/.kiro/steering/` (global).
  Workspace **overrides** global.
- **Default trio** (Kiro generates these): `product.md` (purpose, users, business
  objectives), `tech.md` (frameworks, libraries, tools, constraints),
  `structure.md` (file organization, naming, architecture). Loaded into **every**
  interaction by default.
- **Inclusion modes** via YAML frontmatter (must be first, no leading blank line):

  | Mode | Frontmatter | Behavior |
  |---|---|---|
  | Always (default) | `inclusion: always` | Loads in every interaction |
  | File match | `inclusion: fileMatch` + `fileMatchPattern: "pattern"` | Loads when working with matching files |
  | Manual | `inclusion: manual` | Pulled in on demand via `#steering-file-name` in chat |
  | Auto | `inclusion: auto` + `name:` + `description:` | Auto-included when the request matches (slash-command-like) |

- **File references:** inline a live workspace file with `#[[file:<relative_path>]]`
  — e.g. `#[[file:api/openapi.yaml]]`. Keeps the steering doc pointing at the
  real artifact instead of a stale copy.

## Specs

The heart of Kiro's spec-driven flow: **requirements → design → tasks**. Kiro
generates and iterates these as structured documents that drive implementation.

## Hooks

Agent workflows triggered by **IDE events** (save, create, commit). Like Claude
Code hooks, these are harness-driven automation, not prompt text.

## AGENTS.md

Kiro also supports the `AGENTS.md` standard — **loads always, no inclusion modes**
(unlike steering files, which gate via frontmatter).

## Subagent caveat (disputed — verify per release)

Kiro has subagents (since ~0.9). Docs claim `inclusion: always` steering reaches
them, but open issues have disputed this in practice; Specs/Hooks do **not**
reach Kiro subagents. Treat "what reaches a subagent" as unverified and test it.
