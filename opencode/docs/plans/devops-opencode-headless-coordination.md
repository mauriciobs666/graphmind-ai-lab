# devops-opencode-headless — Docs-only Coordination Ledger

> **Status:** active · **Owner:** `tico` · **Tracks:** — · **Last updated:** 2026-09-12

Coordinates the requirements → plan → review chain for the `tank` headless OpenCode agent. No
unit in this chain touches source, tests, or config — the moment one does, the whole remaining
chain hands off to `teco`.

## Ledger

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| Requirements interview | `tico` | — (interactive) | accepted | `opencode/docs/requirements/devops-opencode-headless.md` | — | — |
| Implementation plan | `architect` | `ad6b25b909e4bf140` | revised, ready for review | `opencode/docs/plans/devops-opencode-headless.md` | `security-expert` review dispatched | 371,583 tokens · 101 tool uses · ~65.7 min |
| Security review | `security-expert` | `ace5d715ef79ccdb0` | dispatched | `opencode/docs/reviews/devops-opencode-headless.md` | pending | — |

## Notes

- **2026-09-12 — Plan returned.** `architect` delivered the plan (shared-persona mechanism via
  OpenCode's `{file:...}` live-include; a static, deny-by-default OpenCode permission table as the
  safety net rather than trusting the local model's judgment; a self-written bring-up/teardown
  ownership marker for FR-5a/5b symmetry). Verified two load-bearing OpenCode mechanics
  empirically against the installed binary + a live LM Studio server rather than assuming them
  from docs. Confirmed nothing forecloses the long-term `tank`-absorbs-`claude-devops` merge.
  Flagged two items: (1) which concrete environment FR-5a/5b's "demo/dev environment" means —
  routed to the stakeholder, not architect's call; (2) a residual glob-pattern smuggling risk in
  the permission allow-list, worth a `security-expert` advisory pass before promotion.
- **2026-09-12 — Environment-scope clarified.** Stakeholder: any environment, `falkor-chat` is the
  current practical example, not an exclusivity constraint — logged in the requirements doc's
  decision log (no FR text changed, it was already generic). Relaying to `architect` to correct
  the plan's assumption.
- **2026-09-12 — Security review gate accepted.** Stakeholder agreed to the `security-expert`
  advisory pass architect recommended, on the permission-table's glob-pattern smuggling risk.
  Dispatching once the plan's environment-scope revision lands.
- **2026-09-12 — Plan revised.** `architect` generalized FR-5a/5b's design off `falkor-chat`
  specifically: repo-root-scoped compose globs instead of a literal `falkor-chat/compose.yaml`
  path, a state marker keyed to the requested environment's `--project-directory` (not a fixed
  filename), bring-up/teardown scripts taking the target environment as an argument, and new
  genericity + repo-root-scoping tests in §5. No requirements-doc change needed. Dispatching
  `security-expert` next on the plan's permission-table glob-smuggling risk.
