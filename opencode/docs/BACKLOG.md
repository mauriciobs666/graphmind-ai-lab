# Backlog — opencode

> **Status:** active · **Owner:** `teco` · **Tracks:** —

> **How to read this.** Forward-looking only — what is proposed but unbuilt. *When* something
> changed and *what* it involved live in [`HISTORY.md`](./HISTORY.md), one dated entry per
> delivered item — **a delivered item is not kept here at all, not even as an index row**
> (root `AGENTS.md`). This is the first backlog for `opencode/` (adopted alongside `HISTORY.md`
> at the close of the `tank` build, the first delivered feature here) — no `K-`numbering exists
> yet for this component, so items below are tracked by name rather than an ID.

## Open

- **Wire `tank`'s outer scripts into a scheduler.** The plan named this explicitly as the natural
  next increment but out of scope to build (`opencode/docs/plans/devops-opencode-headless.md` §4
  step 9): a cron entry or systemd timer calling `opencode/agents/tank/scripts/health-check.sh`
  (and, less routinely, `bring-up.sh`/`tear-down.sh`) on a schedule. No design work done yet —
  the outer scripts are already the intended hook point, so this is mostly a deployment/ops task
  once someone wants `tank` running unattended on a timer rather than invoked by hand. Constraint from the standard-local-model trial: `tank` needs its model loaded by hand at the preflight minimum, and with one slot a queued request is aborted by OpenCode at ~300 s and retried, so a scheduled run must not overlap heavy `:8200` extraction.
