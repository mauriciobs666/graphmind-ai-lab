# devops-opencode-headless — Docs-only Coordination Ledger

> **Status:** active · **Owner:** `tico` · **Tracks:** — · **Last updated:** 2026-09-12

**Chain complete: plan approved with suggestions, nothing blocking.** Handoff to implementation
(needs `coder`/`tdd-engineer` — out of `tico`'s coordination scope; see Notes) is the next step.

Coordinates the requirements → plan → review chain for the `tank` headless OpenCode agent. No
unit in this chain touches source, tests, or config — the moment one does, the whole remaining
chain hands off to `teco`.

## Ledger

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| Requirements interview | `tico` | — (interactive) | accepted | `opencode/docs/requirements/devops-opencode-headless.md` | — | — |
| Implementation plan | `architect` | `ad6b25b909e4bf140` | revised, ready for review | `opencode/docs/plans/devops-opencode-headless.md` | `security-expert` review dispatched | 371,583 tokens · 101 tool uses · ~65.7 min |
| Security review | `security-expert` | `ace5d715ef79ccdb0` | gated | `opencode/docs/reviews/devops-opencode-headless.md` | verdict: **needs changes** (1 blocker, 2 major, 2 minor) | 139,607 tokens · 37 tool uses · ~18 min |
| Plan revision 2 (blocker fix) | `architect` | `ad6b25b909e4bf140` | revised (v3), ready for re-review | `opencode/docs/plans/devops-opencode-headless.md` | re-review by `security-expert` dispatched | 314,917 tokens · 30 tool uses · ~206 min |
| Security re-review (Pass 2) | `security-expert` | `ace5d715ef79ccdb0` | accepted | `opencode/docs/reviews/devops-opencode-headless.md` | verdict: **approve with suggestions**, no blocker/major remaining | 198,234 tokens · 14 tool uses · ~6.6 min |

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
- **2026-09-12 — Security review returned: needs changes.** Blocker, live-reproduced (not
  theoretical): a second, smuggled `-f`/`--project-directory` pair inside the already-allowed
  compose `down`/`up*` glob's own wildcard span retargets the command at an out-of-repo-scope
  stack — defeats FR-8's repo-scoping and the ownership-marker mechanism's entire premise
  simultaneously, no shell metacharacters or model compromise needed. Two major findings compound
  it: the marker slug is basename-only (collides across environments sharing a leaf dirname with
  no attacker input needed) and the marker has no tamper/freshness/provenance check (any
  filesystem writer, or a stale leftover, can make `tank` believe it owns something it didn't
  start). Two minor findings (asymmetric `up*` wildcard permissiveness vs. `down`'s exact-match;
  an unconfirmed but likely-shared path-traversal gap on `write: state/*`). Reviewer also raised an
  open question for the stakeholder: does the blocker change the risk calculus on shipping the
  generic-any-environment carve-out at all, vs. starting narrower (a hardcoded single-stack
  allow-list)? Routing to stakeholder before dispatching architect for a revision, since this
  affects design approach and possibly scope, not just a tightened regex.
- **2026-09-12 — Stakeholder decision: fix it properly, stay generic.** Adopt the reviewer's
  design-level fix (wrapper script builds the entire `docker compose ... {up|down}` argv itself
  from a pre-validated slug/allow-list; no model-authored string ever reaches the permission glob)
  rather than narrowing scope to a single hardcoded environment. `tank` stays able to target any
  compose environment. Dispatching `architect` for a second plan revision.
- **2026-09-12 — Plan revised to v3, blocker fixed architecturally.** `architect` removed every
  raw `docker compose` glob pattern; added a repo-committed `environments.json` allow-list
  (slug → compose file/project directory, exact-match only) plus three fixed wrapper scripts that
  build the compose argv themselves in code — `tank`'s `permission.bash` now only allows invoking
  those scripts by exact path, eliminating the vulnerable pattern class rather than tightening it.
  Marker slug now the human-assigned `environments.json` key (collision requires a caught-at-review
  authoring mistake, not a runtime hazard); added a freshness bound + provenance fields to the
  marker; `up`/`down` now share identical wrapper discipline; `tools.write`/`.edit` set to `false`
  entirely (eliminates the write-traversal surface rather than verifying it, flagged explicitly for
  the re-reviewer to judge). Architect sanity-checked the new logic against a bash mock but did
  **not** re-run the reviewer's live tear-down reproduction against the new config — explicitly
  left for the re-review pass. Dispatching `security-expert` (same agent/thread) for Pass 2,
  continuing the same review document per the reviews/ family convention.
- **2026-09-12 — Pass 2 complete: approve with suggestions.** `security-expert` independently
  rebuilt v3's permission table and wrapper-script logic, then re-ran the *exact* live reproduction
  technique that found the original blocker (not architect's bash-only mock) — the original payload
  now denies, and five further live attack attempts against the new wrapper-script design (unquoted
  smuggling at the script boundary, quoted single-token smuggling, a path-traversal-shaped slug,
  chaining, and a full up→attempted-smuggle→legitimate-teardown cycle against a real running
  fixture) were all correctly refused, with no new bypass found. Both majors confirmed fixed by
  construction; both minors resolved (write-traversal closed by elimination, independently
  re-verified — no substitute write primitive exists in the bash allow-list). Both of architect's
  open judgment calls resolved in the plan's favor: keep the now-non-load-bearing defense-in-depth
  deny net (free, costs nothing), and treat "eliminated, not verified" as a fully closed answer.
  Two non-blocking nits recorded for implementation time (a comment explaining the wrapper allow
  pattern's trailing wildcard is safe; a unit test asserting `resolve_slug` is exact-key-equality,
  never substring/prefix). Plan (v3) and the Pass 2 review both committed. **Chain complete** —
  requirements → plan → security review, all three units accepted, nothing blocking.
