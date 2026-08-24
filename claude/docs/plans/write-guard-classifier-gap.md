# Write-guard auto-mode classifier gap — Design

> **Status:** active · **Owner:** `cobb` · **Tracks:** —

**Component:** `claude/` · **Owner altitude:** cobb (design only — no implementation in this unit) ·
**Reviewer:** `analyst` (gate before implementation)
**Background:** `claude/docs/requirements/agent-permission-friction2.md` open question 3 (two
independent 2026-08-23 regressions — `analyst` on `docs/reviews/document-ingestion-impl.md`,
`tdd-engineer` on `cypher-mcp/tests/test_server.py` — both statically correct guards, both still
prompting) and the root-cause investigation committed at `6193083`. This document doesn't
re-derive either; it picks up from their conclusions.
**Goal:** design (not implement) whether and how a `permissions.allow` **rule** in
`.claude/settings.json` can close the gap the root-cause investigation found — an explicit
`PreToolUse` hook `"allow"` decision does not reliably suppress the confirm-before-Write/Edit
prompt for a Task/Agent-delegated write under `auto` mode (this account's default) — without
weakening the per-agent escalation guarantee (`agent-permission-friction.md` AC-4) any of the
existing guards currently provide.

---

## 1. Recap: what's already established (not re-derived here)

From the `6193083` investigation, verified against `code.claude.com/docs/en/permissions` and
`.../permission-modes` (fetched 2026-08-24):

- Both 2026-08-23 instances were statically correct (guard scripts unchanged since 2026-08-21,
  deployment symlinks live, no stale version — reproduced on Claude Code v2.1.240 and v2.1.241
  both) and traced, via the actual session transcripts, to a genuine multi-minute/multi-hour
  human-decision gap between `tool_use` and its `toolUseResult` — a real prompt, not a
  misremembered report. The parent (`teco`) session's transcript carries an explicit
  `"type":"permission-mode"` record showing it stayed in `auto` mode continuously across both
  incidents.
- Auto mode's own documented decision order (`permission-modes` doc, "How the classifier evaluates
  actions") auto-approves a non-protected-path working-directory file edit at step 2, with **no**
  classifier involvement and **no** prompt, regardless of hooks — which the two instances directly
  contradict. Neither instance's target is a protected path.
- Nothing in that decision order, nor in "How auto mode handles subagents," names a `PreToolUse`
  hook's `"allow"` as exempting an action from classifier review. Hooks ("Extend permissions with
  hooks") are a separate extension layer; their interaction with the auto-mode classifier for a
  **subagent-delegated** write specifically is undocumented.
- A prior, unrelated risk flagged in `agent-permission-friction.md` §1.2/§10.1 — an undocumented
  blanket `~/.claude/settings.json` `"permissions":{"allow":["Edit","Write","NotebookEdit"]}` rule
  (no path scoping, tool-level) — is confirmed gone as of this document (`~/.claude/settings.json`
  mtime 2026-08-21 23:20:56, predating the 2026-08-23 instances; current content is only the
  WebFetch/WebSearch domain allowlist). No interfering prior rule, no counter-evidence against the
  hypothesis below sitting in that file.

## 2. Docs finding this document adds: `Edit(path)`/`Read(path)`-only rule matching, and where a rule sits in the decision order

Re-read `code.claude.com/docs/en/permissions` specifically for the **rule** mechanism (not hooks),
2026-08-24:

**2.1 — A rule resolves earlier than a hook does.** The classifier's decision order (`permission-modes`,
"How the classifier evaluates actions") is:

> "1. Actions matching your **allow, ask, or deny rules** resolve immediately. Writes to protected
> paths route to the classifier even when an allow rule matches... 2. Read-only actions and file
> edits in your working directory are auto-approved, except writes to protected paths. 3.
> Everything else goes to the classifier."

Step 1 names settings.json permission **rules** — not hook decisions — as the thing that resolves
an action before the classifier is ever invoked. "How auto mode handles subagents" says a
subagent's actions go through the classifier "with the same rules as the parent session," which
reads naturally as: step-1 rule resolution applies to a delegated write the same way it applies to
the parent's own. **This is the mechanistic case for why a rule might succeed where a hook's
`"allow"` didn't** — it hits an earlier, classifier-bypassing point in the decision tree that hooks
are never named as reaching.

**2.2 — Rules only key off `Edit(path)`/`Read(path)`, never `Write(path)`.** Verbatim:

> "Claude Code checks file permissions against `Edit(path)` and `Read(path)` rules only. If you
> write a path rule for `Write`, `NotebookEdit`, `Glob`, or the legacy `MultiEdit` tool instead,
> Claude Code accepts the rule but never consults it... Use `Edit(docs/**)` in place of
> `Write(docs/**)`, `NotebookEdit(docs/**)`, or `MultiEdit(docs/**)`."

A naive translation that added both `Write(...)` and `Edit(...)` allow rules per existing guard
glob would silently no-op the `Write(...)` half. Every rule below uses `Edit(...)` only.

**2.3 — Pattern-syntax note carried over from the hook globs.** The existing guard globs are
doubled (bare + `*/`-prefixed) specifically because `tool_input.file_path` can arrive absolute or
relative. Rule syntax's nearest equivalent is `**` (crosses path segments, matches at any depth) —
`Edit(docs/reviews/**)` anchors at the settings source (`<project root>/docs/reviews/**` for a
project-settings rule), while `Edit(**/docs/reviews/**)` matches at any depth regardless of anchor.
Both forms are included per candidate below, mirroring the doubled convention.

## 3. Split-verdict design overview

| Guard shape | Agents | Rule-based supplement? | Why |
|---|---|---|---|
| Allow-list (`guard-doc-writes.sh`) | `architect`, `analyst`, `data-scientist`, `teco`, `tico`, `cobb`, `qa-engineer`, `security-expert`'s review guard | **Candidate — designed below** | Narrow, maintained glob list per agent; translates to a small, enumerable set of `Edit(...)` rules |
| Deny-list (`guard-broad-write.sh`) | `tdd-engineer` (and any future `frontend-engineer`/`devops`/`graph-dba` guard of the same shape) | **Excluded — see §5** | Rules aren't agent-scoped; the only literal translation opens the whole repo to everyone |

## 4. The scoping tradeoff (why this isn't a free win)

A `PreToolUse` hook is scoped to *the agent whose frontmatter wires it* — `docs/reviews/*` is only
silently approved when `analyst` (or `security-expert`) is the one writing there; any other agent,
or the top-level human session, hitting that same path still goes through ambient-mode escalation.
A `.claude/settings.json` `Edit(...)` allow rule has **no agent scoping mechanism at all** — it's a
path match, full stop, and applies to every session and every agent, including ones that were never
granted that remit.

Concretely: adding `Edit(docs/reviews/**)` as a project-wide allow rule to close the classifier gap
for `analyst`'s writes would, as a side effect, silently auto-approve **anyone's** write to
`docs/reviews/*` — narrowing `agent-permission-friction.md` AC-4's "genuine escalation on an
out-of-remit path" guarantee from *per-agent* to *per-path*, for every glob a rule is added for.
That's a real, deliberate trade, not a side effect to wave through — each rule added under §5 is a
named exposure decision for the review gate, not a mechanical copy of an existing hook glob.

The hook stays in place regardless (§6) — this design adds rules *alongside* hooks, never in place
of them, so the per-agent "ask on mismatch" behavior for a genuinely out-of-remit path is unchanged
for every agent/path combination that doesn't get a supplementary rule.

## 5. Candidate design: allow-list guards → `Edit(...)` rules

For each existing `guard-doc-writes.sh` wrapper, translate its allowed-glob string (already
maintained, already reviewed) into one or more `Edit(...)` allow rules in `.claude/settings.json`,
doubled the same way the hook globs are:

| Wrapper (source of truth for the glob) | Existing hook glob | Candidate rule(s) |
|---|---|---|
| `claude/analyst/hooks/guard-review-doc-writes.sh` | `docs/reviews/*\|*/docs/reviews/*` | `Edit(docs/reviews/**)`, `Edit(**/docs/reviews/**)` |
| `claude/architect/hooks/guard-*.sh` | `docs/plans/*\|*/docs/plans/*` (verify exact glob against the live wrapper before implementing — not re-read in this design pass) | `Edit(docs/plans/**)`, `Edit(**/docs/plans/**)` |
| `claude/data-scientist/hooks/guard-*.sh` | `docs/plans/*\|docs/reviews/*` shape (ML-scoped — verify exact glob) | `Edit(...)` mirroring the live glob |
| `claude/teco/hooks/guard-*.sh` | `docs/plans/*-coordination.md` shape (verify exact glob) | `Edit(...)` mirroring the live glob |
| `claude/tico/hooks/guard-*.sh` | `docs/requirements/*\|docs/manuals/*` shape (verify exact glob) | `Edit(...)` mirroring the live glob |
| `claude/cobb/hooks/guard-cobb-topic-writes.sh` | topic-bounded union (agent defs, kaizen, catalogs, cobb's own skills, `cypher-mcp/README.md`) | one `Edit(...)` rule pair per union member, same list |
| `claude/qa-engineer/hooks/guard-qa-doc-writes.sh` | `docs/test-plans/*\|docs/test-reports/*` | `Edit(docs/test-plans/**)`, `Edit(**/docs/test-plans/**)`, `Edit(docs/test-reports/**)`, `Edit(**/docs/test-reports/**)` |
| `security-expert`'s review guard | `docs/reviews/*` (same core as `analyst`) | Same rules as the `analyst` row — a duplicate/no-op if `analyst`'s rule is already present, since rules aren't agent-scoped anyway |

**Note on the incomplete glob citations above:** this design pass reused the globs already
recorded in `agent-permission-friction.md` §2's file table and did not re-read every live wrapper
script to confirm each one hasn't drifted since. **Before implementation, re-`grep` each
`claude/*/hooks/guard-*.sh` for its actual current glob string** rather than trusting this table —
getting a rule's pattern wrong is a silent under- or over-scope, not a loud failure.

**Sequencing (design-level, not a build order commitment):**
1. Add rules for one low-stakes agent first (`qa-engineer`'s `docs/test-plans/*`/`docs/test-reports/*`
   is a reasonable pilot — narrow, low ambient traffic from other agents) and validate empirically
   (§7) before rolling out the rest.
2. Roll out the remaining allow-list guards' rules only after the pilot confirms the mechanism
   actually closes the gap for a Task-delegated write — not on docs-reading confidence alone.
3. Leave every hook wrapper untouched — rules are additive, not a replacement.

## 6. Excluded: `tdd-engineer` / broad-implementer (deny-list) shape

`guard-broad-write.sh`'s shape is "allow almost everything, escalate only on a short deny-list of
other-specialists' paths" — safe today *only* because it's scoped to while `tdd-engineer` is
active. Rules have no equivalent scoping. The literal translation — a blanket `Edit(**)`-shaped
allow rule plus per-glob `ask` rules mirroring the deny-list (rules are evaluated deny, then ask,
then allow, per `code.claude.com/docs/en/permissions` "Manage permissions," so a matching `ask`
rule would still win over the blanket allow) — would open the **entire repository** to
unconditional auto-approval for **every session and every agent**, not just `tdd-engineer`. That's
an unacceptable, repo-wide escalation-suppression change wearing a narrow fix's clothes.

**Recommendation: do not add rules for this shape.** Leave `tdd-engineer` (and any future
`frontend-engineer`/`devops`/`graph-dba` guard of the same shape) hook-only. The classifier gap
stays open there — a real, acknowledged limitation of this design, not a solved case. If this
becomes painful enough to revisit, the honest options are narrowing `tdd-engineer`'s own remit to
an allow-list-shaped guard (a bigger behavior change, its own design) or accepting the friction.

## 7. Still open: empirical validation

Docs analysis (§2) supports the hypothesis that a rule resolves before the classifier and would
apply to a delegated write. **This was not empirically confirmed.** The attempted test — an
isolated git worktree, a throwaway headless `claude -p --permission-mode auto` run with the
candidate rule passed via the ephemeral `--settings` CLI flag (never persisted to any settings
file), delegating to `analyst` via the Agent tool to reproduce the exact Task-delegation shape that
failed — was blocked before it could run, by the auto-mode classifier governing the *investigating*
session itself: spawning a nested `claude -p` process from Bash was denied outright ("Permission
for this action was denied by the Claude Code auto mode classifier... STOP and explain to the user
what you were trying to do and why you need this permission. Let the user decide how to proceed.").
No file was created; the worktree was fully removed; nothing in the repository was touched.

**This needs a human running an actual interactive session** — the stakeholder adds one candidate
rule (§5's pilot row is the smallest, lowest-risk one to try first) to `.claude/settings.local.json`,
then reproduces a Task-delegated write to that path from a concurrent `auto`-mode session, and
observes whether it prompts. **Nothing in §5 or §6 should be implemented until that test lands** —
this document's recommendation is a docs-supported hypothesis, not a confirmed fix.

## 8. Decisions for the review gate (`analyst`)

1. Is the per-agent → per-path scoping narrowing (§4) an acceptable trade for each candidate glob
   in §5, or should some/all of them be rejected even after empirical confirmation?
2. Is leaving `tdd-engineer`'s classifier gap open (§6) acceptable, or does it need its own
   follow-up design?
3. Does the pilot-then-roll-out sequencing (§5) match how `analyst`/`teco` want this staged, or
   should the empirical test (§7) run against every candidate before any of them ship?
