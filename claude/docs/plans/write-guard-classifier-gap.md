# Write-guard auto-mode classifier gap — Design

> **Status:** active · **Owner:** `cobb` · **Tracks:** —
> **Version:** 2 (revised 2026-08-24, folding in `docs/reviews/write-guard-classifier-gap.md`'s findings) · **Reviews:** `claude/docs/reviews/write-guard-classifier-gap.md` (verdict: needs changes → corrections folded in below)

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

**Caveat carried forward from the review (`docs/reviews/write-guard-classifier-gap.md`, "Note"
finding):** that same sentence — "the same rules as the parent session" — is also consistent with a
**narrower** reading: every subagent action reaches the classifier regardless, and that phrase
describes only what *context* the classifier consults, not that rules can short-circuit the
classifier for a subagent action the way §2.1 assumes they do for a top-level one. Docs prose alone
can't disambiguate these two readings. This doesn't change anything here — §7 already treats the
whole premise as unconfirmed pending a live test — but whoever runs that test needs to know there
are **two live hypotheses to distinguish, not one**: "rules bypass the classifier for subagent
writes" vs. "rules bypass it only for top-level writes, and subagent writes always reach the
classifier regardless of rules" (which would mean this design's fix doesn't close the gap for
exactly the Task/Agent-delegated case that motivated it, even though it might still help a
same-agent-run-interactively write).

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
| Allow-list (`guard-doc-writes.sh`) — doc-kind globs | `architect`, `analyst`, `data-scientist`, `teco`, `tico`, `qa-engineer`, `security-expert`'s review guard | **Candidate — designed below (§5), risk-differentiated per glob (§5.2)** | Narrow, maintained glob list per agent; translates to a small, enumerable set of `Edit(...)` rules |
| Allow-list (`guard-doc-writes.sh`) — cobb's topic-bounded union | `cobb` | **Split — see §5.3.** Half candidate (kaizen/catalog docs), half **excluded** (agent-definition and skill-package globs) | The union isn't homogeneous — see §5.3 |
| Deny-list (`guard-broad-write.sh`) | `tdd-engineer` (and any future `frontend-engineer`/`devops`/`graph-dba` guard of the same shape) | **Excluded — see §6** | Rules aren't agent-scoped; every translation considered opens the whole repo to everyone |

## 4. The scoping tradeoff (why this isn't a free win)

A `PreToolUse` hook is scoped to *the agent whose frontmatter wires it* — `docs/reviews/*` is only
silently approved when `analyst` (or `security-expert`) is the one writing there; any other agent,
or the top-level human session, hitting that same path still goes through ambient-mode escalation.
A `.claude/settings.json` `Edit(...)` allow rule has **no agent scoping mechanism at all** — it's a
path match, full stop, and applies to every session and every agent, including ones that were never
granted that remit.

This is one tradeoff, but it isn't one uniform severity across every candidate glob — see §5.2 for
the per-glob differentiation (a doc-kind glob's exposure is a documentation-integrity risk
backstopped by version control; the agent-definition/skill-package globs excluded in §5.3 are a
categorically worse, privilege-escalation-shaped exposure, not a bigger instance of the same risk).
The hook stays in place regardless (every candidate below is additive, never a replacement for its
hook) — so the per-agent "ask on mismatch" behavior for a genuinely out-of-remit path is unchanged
for every agent/path combination that doesn't get a supplementary rule.

## 5. Candidate design: allow-list guards → `Edit(...)` rules

### 5.1 Per-agent glob table (corrected against the live wrapper scripts)

Every row below was re-read directly from its live `claude/*/hooks/guard-*.sh` this revision
(the prior version cited two of these from `agent-permission-friction.md`'s summary table without
re-checking the scripts — flagged and corrected per the review's Major finding):

| Wrapper (source of truth for the glob) | Live glob (`grep`-verified 2026-08-24) | Candidate rule(s) |
|---|---|---|
| `claude/architect/hooks/guard-plan-doc-writes.sh` | `docs/plans/*\|*/docs/plans/*` | `Edit(docs/plans/**)`, `Edit(**/docs/plans/**)` |
| `claude/analyst/hooks/guard-review-doc-writes.sh` | `docs/reviews/*\|*/docs/reviews/*` | `Edit(docs/reviews/**)`, `Edit(**/docs/reviews/**)` |
| `security-expert`'s review guard | `docs/reviews/*\|*/docs/reviews/*` (same core as `analyst`, same glob) | Same two rules as the `analyst` row — adding it again is a no-op, since rules aren't agent-scoped |
| `claude/teco/hooks/guard-coordination-doc-writes.sh` | `docs/plans/*\|*/docs/plans/*` — **byte-identical to `architect`'s.** (The prior revision of this table described a `docs/plans/*-coordination.md` suffix restriction; the live wrapper has no such restriction — its coordination-doc convention is enforced by where `teco` chooses to write, not by the guard's glob.) | Same two rules as the `architect` row — not an independent decision |
| `claude/data-scientist/hooks/guard-ds-doc-writes.sh` | `docs/plans/*\|*/docs/plans/*\|docs/reviews/*\|*/docs/reviews/*` — the **exact union of `architect`'s and `analyst`'s globs**, byte-for-byte. (The prior revision described an "ML-scoped," `-ml.md`-suffix-restricted glob; the live wrapper has no such restriction.) | Same four rules already covered by the `architect` and `analyst` rows — not an independent decision |
| `claude/tico/hooks/guard-*.sh` | `docs/requirements/*\|*/docs/requirements/*\|docs/manuals/*\|*/docs/manuals/*` (verify exact glob string before implementing — not re-read this revision; unlike the `teco`/`data-scientist` rows, nothing in this revision's re-check found a description problem here, but it also wasn't independently re-`grep`'d) | `Edit(docs/requirements/**)`, `Edit(**/docs/requirements/**)`, `Edit(docs/manuals/**)`, `Edit(**/docs/manuals/**)` |
| `claude/qa-engineer/hooks/guard-qa-doc-writes.sh` | `docs/test-plans/*\|*/docs/test-plans/*\|docs/test-reports/*\|*/docs/test-reports/*` (verify exact glob before implementing — cited from `agent-permission-friction.md` §5, not re-read this revision) | `Edit(docs/test-plans/**)`, `Edit(**/docs/test-plans/**)`, `Edit(docs/test-reports/**)`, `Edit(**/docs/test-reports/**)` |
| `claude/cobb/hooks/guard-cobb-topic-writes.sh` | See §5.3 — split, not a single row |

**Net effect of the correction:** this is not eight independently-scoped exposure decisions — it's
**four distinct glob surfaces** (`docs/plans/**`, `docs/reviews/**`, `docs/requirements/**` +
`docs/manuals/**`), each already claimed by multiple agents' existing hooks. That's a point in the
design's favor (less incremental surface than a naive per-row count implies — `docs/plans/**`
gets exposed once regardless of how many agents' hooks already write there), but the table above
states the actual shape rather than presenting `teco`'s and `data-scientist`'s rows as narrower,
agent-specific carve-outs.

**Remaining "verify before implementing" caveat:** the `tico` and `qa-engineer` rows above were not
independently re-`grep`'d this revision (only the two rows the review flagged as description-wrong
were re-checked). Re-confirm both against their live wrapper scripts before implementation, the
same discipline that caught the `teco`/`data-scientist` errors.

### 5.2 Per-glob risk differentiation (not one blanket "acceptable trade" judgment)

The four glob surfaces above don't carry the same stakes. Differentiated per the review's table:

| Glob surface | Risk | Reasoning |
|---|---|---|
| `docs/reviews/**` (`analyst`, `security-expert`, `data-scientist`) | **Self-approval risk — flag explicitly, don't wave through.** | A review document is the artifact recording an independent verdict on someone else's work (`AGENTS.md`'s "Owner altitude" convention). If any agent can silently write here, the agent whose work is under review can, in principle, edit its own review's verdict with zero escalation — the one thing this doc kind exists to prevent. |
| `docs/test-reports/**` (`qa-engineer`) | **Same self-approval category as reviews.** | A test report is QA's independent verification result — same risk shape, same recommendation: don't lump with `docs/test-plans/**`. |
| `docs/requirements/**` (`tico`) | **Moderate-to-major — worth the same explicit flag.** | Requirements carry the acceptance criteria a deliverable is judged against; silently loosening AC text is the "move the goalposts unnoticed" risk one step upstream of a review self-edit. |
| `docs/plans/**` (`architect`, `data-scientist`, `teco`) | **Moderate, acceptable.** | A silently-edited plan could mask a mid-implementation deviation, but plans get a companion review (`reviews/<slug>.md`) in the normal family flow — a real backstop the review/test-report kinds don't have to the same degree. |
| `docs/test-plans/**` (`qa-engineer`) | **Fine — this is the pilot.** | A test *plan* is forward-looking, not a pass/fail verdict — lower stakes, and already the chosen low-risk starting point (narrow, low ambient traffic) independent of this table. |
| `docs/manuals/**` (`tico`) | **Fine.** | End-user-facing prose, no gate/verdict role. |

None of this rises to the §5.3 blocker below — these are documentation-integrity risks backstopped
by version control, not a privilege-escalation vector — but `docs/reviews/**`, `docs/test-reports/**`,
and `docs/requirements/**` need an explicit "yes, we accept losing the write-time self-approval
guard here" sign-off from the stakeholder, separately from `docs/plans/**`/`docs/test-plans/**`/
`docs/manuals/**`, rather than one blanket answer for the whole candidate set. Carried into §8.

### 5.3 `cobb`'s row — split, not a single candidate

`guard-cobb-topic-writes.sh`'s allowlist is not homogeneous. It bundles:

- **Lower-stakes, candidate half:** `claude/*/kaizen/history.md`, `claude/*/kaizen/plan.md`,
  `claude/README.md`, `claude/AGENTS.md`, `claude/CLAUDE.md`, `cypher-mcp/README.md` — raw-capture
  and cross-component pointer docs, same general shape as the doc-kind rows above. **Explicit
  callout:** `AGENTS.md`/`CLAUDE.md` auto-load into every subagent's context via the always-loaded
  memory hierarchy (`skills/agent-standards/claude-code.md`) — a bad silent edit there has unusually
  wide blast radius (every session reads it), even though it isn't itself a privilege-escalation
  vector the way the excluded half is.
- **Excluded half:** `claude/*/*.md` (every agent's own definition file — system prompt, tool
  grants, `hooks:` wiring, `permissionMode`) and both skill-package globs, `skills/agent-maintenance/*`,
  `skills/agent-standards/*` (bodies injected into any session that references them — same trust
  class as an agent definition). **These do not proceed as rule candidates.** Claude Code's
  protected-path carve-out (which forces classifier review even past a matching allow rule) only
  catches dot-prefixed paths (`.git`, `.claude`, etc.) — this repo's actual agent definitions live
  at the plain, non-dot-prefixed `claude/<name>/<name>.md`, confirmed symlinked in from
  `~/.claude/agents/<name>/`, so nothing about a rule matching this glob would route through that
  carve-out. A standing `Edit(claude/**/*.md)` allow rule would let any session — any agent, or a
  prompt-injected/compromised one — silently rewrite its own or another agent's system prompt, hook
  wiring, or tool grants, or the `agent-maintenance`/`agent-standards` skill bodies that define the
  team's own maintenance protocol, with **zero** classifier involvement and zero human visibility.
  That reopens, as a standing rule instead of a one-off attempt, the exact hazard
  `skills/agent-standards/claude-code.md` already documents as caught and correctly stopped (the
  2026-08-20 `cobb` incident, "Auto-Mode Bypass/Self-Modification": a delegate proposing a
  persistent bypass rule to its own settings, citing a coordinator's authorization rather than the
  user's). The classifier that caught that one-off attempt would never see a future one, because the
  write would resolve at rule-step-1, before the classifier is ever invoked. **Same conclusion and
  reasoning §6 applies to `tdd-engineer`: leave these hook-only, accept the friction — the exposure
  this glob group would create is worse than what it removes.**

## 6. Excluded: `tdd-engineer` / broad-implementer (deny-list) shape

`guard-broad-write.sh`'s shape is "allow almost everything, escalate only on a short deny-list of
other-specialists' paths" — safe today *only* because it's scoped to while `tdd-engineer` is
active. Rules have no equivalent scoping.

**Alternative 1 considered — blanket allow, mirrored deny-list as `ask` rules.** Rules evaluate
deny → ask → allow, first match wins (`code.claude.com/docs/en/permissions`, "Manage permissions"),
so a targeted `ask` rule would still win over a blanket `Edit(**)`-shaped allow. But the blanket
allow itself is the problem: it would open the **entire repository** to unconditional auto-approval
for **every session and every agent**, not just `tdd-engineer` — an unacceptable, repo-wide
escalation-suppression change wearing a narrow fix's clothes.

**Alternative 2 considered — `ask`-only rules mirroring the deny-list, no companion blanket allow.**
This doesn't help either, and it's worth showing why rather than only presenting Alternative 1 as
the sole option: `tdd-engineer`'s allow side is *unbounded* — any source/test file in any current or
future component directory, by design (new component directories are supposed to be auto-covered
without a guard-list edit). An `ask`-only rule set only ever fires on paths that were *already*
correctly escalating under the hook — it does nothing for the actual friction source, which is the
overwhelming majority of in-remit writes that would still fall through to decision-order steps 2/3,
i.e. still riding the same unreliable classifier path they're on today. The only way to get
in-remit writes onto the rule-resolves-first path is an allow rule that covers them — and since the
allow side is unbounded by design, enumerating it as discrete `Edit(<dir>/**)` rules is
operationally close to `Edit(**)` anyway (most of the repo, modulo the same short exclusion list),
adding an ongoing maintenance burden the deny-list shape was specifically built to avoid.

**Conclusion: no narrower rule-based option exists for this shape.** Leave `tdd-engineer` (and any
future `frontend-engineer`/`devops`/`graph-dba` guard of the same shape) hook-only. The classifier
gap stays open there — a real, acknowledged limitation of this design, not a solved case. If this
becomes painful enough to revisit, the honest options are narrowing `tdd-engineer`'s own remit to an
allow-list-shaped guard (a bigger behavior change, its own design) or accepting the friction.

## 7. Still open: empirical validation

Docs analysis (§2) supports the hypothesis that a rule resolves before the classifier and would
apply to a delegated write — **but see §2.1's caveat: there are two live hypotheses about whether
this holds for a subagent-delegated write specifically, and docs prose alone can't disambiguate
them.** Neither was empirically confirmed. The attempted test — an isolated git worktree, a
throwaway headless `claude -p --permission-mode auto` run with the candidate rule passed via the
ephemeral `--settings` CLI flag (never persisted to any settings file), delegating to `analyst` via
the Agent tool to reproduce the exact Task-delegation shape that failed — was blocked before it
could run, by the auto-mode classifier governing the *investigating* session itself: spawning a
nested `claude -p` process from Bash was denied outright ("Permission for this action was denied by
the Claude Code auto mode classifier... STOP and explain to the user what you were trying to do and
why you need this permission. Let the user decide how to proceed."). No file was created; the
worktree was fully removed; nothing in the repository was touched.

**This needs a human running an actual interactive session** — the stakeholder adds one candidate
rule (`docs/test-plans/**`, §5.2's pilot, is the smallest, lowest-risk one to try first) to
`.claude/settings.local.json`, then reproduces a Task-delegated write to that path from a concurrent
`auto`-mode session, and observes whether it prompts — this single test also distinguishes the two
hypotheses in §2.1's caveat: if the write is silently approved, rules do bypass the classifier for a
delegated write; if it still prompts, the narrower reading is correct and this design's mechanism
doesn't close the gap it targets at all, regardless of which candidate glob is chosen. **Nothing in
§5 or §6 should be implemented until that test lands** — this document's recommendation is a
docs-supported hypothesis, not a confirmed fix.

## 8. Decisions for the review gate (`analyst`)

1. **Self-approval trades, decided individually, not as a block.** Does the stakeholder accept
   losing the write-time self-approval guard for `docs/reviews/**`, `docs/test-reports/**`, and
   `docs/requirements/**` — each on its own merits (§5.2), given each is a documentation-integrity
   risk backstopped by git history rather than a privilege-escalation vector? `docs/plans/**` (backed
   by a companion review), `docs/test-plans/**` (the pilot), and `docs/manuals/**` carry materially
   lower stakes and don't need the same individual scrutiny.
2. **`cobb`'s split (§5.3).** Confirm the kaizen/catalog half (`claude/*/kaizen/{history,plan}.md`,
   `claude/README.md`/`AGENTS.md`/`CLAUDE.md`, `cypher-mcp/README.md`) proceeds as a candidate on the
   same terms as the doc-kind rows, with the `AGENTS.md`/`CLAUDE.md` wide-blast-radius callout noted
   but not treated as disqualifying; confirm the agent-definition/skill-package half (`claude/*/*.md`,
   both `skills/agent-*` globs) is excluded outright, not merely deferred.
3. **`tdd-engineer`'s classifier gap (§6).** Accept that it stays open — no rule-based fix exists
   without either an unacceptable repo-wide exposure or a separate remit-narrowing redesign — or
   flag it as needing its own follow-up design track.
4. **Sequencing.** Does the pilot-then-roll-out order (§5.2: `docs/test-plans/**` first, empirically
   validated per §7, before any other candidate) match how `analyst`/`teco` want this staged, or
   should the empirical test in §7 run against every remaining candidate glob before any of them
   ship, rather than pilot-then-generalize?
