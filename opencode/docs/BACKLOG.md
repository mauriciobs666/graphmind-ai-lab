# Backlog — opencode

> **Status:** active · **Owner:** `teco` · **Tracks:** —

> **How to read this.** Forward-looking only — what is proposed but unbuilt. *When* something
> changed and *what* it involved live in [`HISTORY.md`](./HISTORY.md), one dated entry per
> delivered item — **a delivered item is not kept here at all, not even as an index row**
> (root `AGENTS.md`). This is the first backlog for `opencode/` (adopted alongside `HISTORY.md`
> at the close of the `tank` build, the first delivered feature here) — no `K-`numbering exists
> yet for this component, so items below are tracked by name rather than an ID.

## Open

- **Evaluate `nvidia/nemotron-3-nano-4b` as an alternate model for `tank`.** Already provisioned
  in `opencode/agents/tank/opencode.json`'s `provider.lmstudio.models`, but never conclusively
  exercised — the plan flagged this as "validated shallowly"
  (`opencode/docs/plans/devops-opencode-headless.md` §7), and the live-testing arc's DEF-3
  findings (`opencode/docs/HISTORY.md`, 2026-09-13 entry; full evidence in
  `opencode/docs/test-reports/devops-opencode-headless-report.md`) independently suggest the
  current partial resolution — closes both plan-literal test scenarios but doesn't generalize to
  unseen phrasings — may be a `ministral-3-3b`-specific capability characteristic rather than a
  ceiling on the addendum wording. Framed as an open A/B against the same TP-002/TP-003-shaped
  probes, not a commitment to switch the shipped default.
- **Wire `tank`'s outer scripts into a scheduler.** The plan named this explicitly as the natural
  next increment but out of scope to build (`opencode/docs/plans/devops-opencode-headless.md` §4
  step 9): a cron entry or systemd timer calling `opencode/agents/tank/scripts/health-check.sh`
  (and, less routinely, `bring-up.sh`/`tear-down.sh`) on a schedule. No design work done yet —
  the outer scripts are already the intended hook point, so this is mostly a deployment/ops task
  once someone wants `tank` running unattended on a timer rather than invoked by hand.
