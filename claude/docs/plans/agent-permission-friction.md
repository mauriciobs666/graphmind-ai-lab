# Agent permission-escalation friction — Design

> **Status:** active · **Owner:** `cobb` · **Tracks:** — (—)

**Component:** `claude/` · **Owner altitude:** cobb (design + implementation) · **Reviewer:** `analyst` (gate before implementation)
**Upstream:** `claude/docs/requirements/agent-permission-friction.md` (Status: Ready for design) · `claude/docs/plans/agent-permission-friction-coordination.md` (ledger)
**Goal:** stop the redundant per-write confirmation prompt on in-remit `Write`/`Edit` calls for `cobb` (FR-1), `qa-engineer` (FR-3), and the general case (FR-2 — the five already doc-scoped-guarded agents, plus `tdd-engineer`), without weakening escalation on a genuinely out-of-remit write (AC-4) or touching the destructive-ops guards (AC-5).

This is a **design for implementation**, not the implementation. No hook/frontmatter file is edited in this unit — every script body below is specified precisely enough to copy in as-is.

---

## 1. Root-cause finding (read this before the mechanism section — it drives every choice below)

### 1.1 What's unchanged and still correct
`git show f80edab` (2026-07-24, "permissionMode: acceptEdits across all 13 subagents") verified at
the time: *"PreToolUse hooks fire before any permission-mode check, so a hook's 'ask' decision
still forces the prompt under acceptEdits."* `grep -rn "permissionMode" claude/*/[a-z]*.md`
confirms all 13 agents still carry `permissionMode: acceptEdits` today. **This half is still true**
and still explains why the five doc-scoped guards' escalation on a genuine out-of-remit path
(AC-4) keeps working — a hook's explicit `"ask"` always wins. Nothing here changes that.

### 1.2 What the July fix didn't establish, and the fresh evidence exposes
The July commit's message asserts only that a hook's `"ask"` survives `acceptEdits`. It never
claimed the converse — that `acceptEdits` reliably *suppresses* the confirmation prompt for a
hook-free (or hook-silent) agent. That converse is what FR-1/FR-2/FR-3's fresh 2026-08-20/21
evidence falsifies: `cobb` (no guard at all), `qa-engineer` (Bash-only guard, no Write/Edit guard),
and `tdd-engineer` (no guard at all) all hit the plain default confirm prompt on routine in-remit
writes, despite `acceptEdits` since July.

Verified against current official docs (`code.claude.com/docs/en/sub-agents`,
`.../permission-modes`, `.../hooks`, fetched 2026-08-21):

- **`permissionMode` in agent frontmatter is a subagent-launched-via-Task-tool field**, documented
  under "sub-agents." Verbatim: *"If you leave it unset, the subagent inherits the main
  conversation's mode… Setting it overrides that mode, except in the cases described below."* The
  two override-cancelling cases: **"If the parent uses `bypassPermissions` or `acceptEdits`, this
  takes precedence and can't be overridden"**, and **"If the parent uses auto mode, the subagent
  inherits auto mode and any `permissionMode` in its frontmatter is ignored: the classifier
  evaluates the subagent's tool calls with the same block and allow rules as the parent session."**
- **Auto mode is the Pro/Max/Team default starting mode today**: *"On Pro, Max, and Team plans, the
  built-in starting permission mode is auto mode."* The `--agent` CLI flag's own doc entry ("Specify
  an agent for the current session…") says nothing about that top-level session honoring the named
  agent's own frontmatter `permissionMode` — the field's only doc-confirmed effect is on a
  Task-tool-delegated child, not on the top-level session that IS that agent.
- The evidence instances all trace to *"a concurrent `teco` session"* delegating to
  `cobb`/`qa-engineer`/`tdd-engineer`/`analyst`/`tico` via `Task`. Whichever of the two paths above
  actually applied — `teco`'s own top-level `permissionMode` never took effect (leaving the session
  in the plan's default `auto`), or `teco`'s effectively-`acceptEdits` top-level session still
  routes its Task-launched children through `auto` mode's child-frontmatter-ignoring rule once auto
  mode is in play anywhere in the chain — **every child subagent's own `permissionMode:
  acceptEdits` was silently inert for this exact session shape.** I could not fully certify from
  docs alone which single mode was ambient at every instant (auto mode's own step-2 rule —
  *"Read-only actions and file edits in your working directory are auto-approved, except writes to
  protected paths"* — would, taken alone, predict *no* prompt on a non-protected working-directory
  edit, which doesn't cleanly match the observed friction either) — and I did not have a way to
  inspect the live session's actual mode after the fact. That residual uncertainty doesn't block
  the design: see 1.3.

**One more relevant, verifiable artifact**: `~/.claude/settings.json` (this machine, personal,
**not** in the repo) currently carries an undocumented blanket
`"permissions":{"allow":["Edit","Write","NotebookEdit"]}` entry — file `birth == modify ==
2026-08-20 19:47:22`, i.e. created in one shot, same day as the evidence, timing relative to
individual instances unknown. `claude/README.md`'s only *documented* personal-settings block is
the 5-domain WebFetch/WebSearch allowlist for `cobb` — this Edit/Write entry is not part of any
documented convention. Left in place, an unscoped `Edit`/`Write` allow rule would, on its own,
silently approve genuinely out-of-remit writes too for any hook-free agent (a matching allow rule
resolves a permission check immediately unless a hook's `"ask"`/`"deny"` intervenes first) — i.e.
it is not a safe substitute for AC-4 and this design does not rely on it. **Flagged as a decision
for the stakeholder/`teco`, not something this plan edits**: narrow or remove that entry once the
hook-based fix below lands, so AC-4's safety net is the hook, not an accidental side effect of a
personal, unscoped, non-portable local override.

### 1.3 The mechanism this design actually relies on (fully doc-confirmed, mode-independent)
From `code.claude.com/docs/en/hooks`, the `PreToolUse` `permissionDecision` field table:

> `"allow"` **skips the permission prompt**, except for the actions no mode auto-approves and for
> `AskUserQuestion` and `ExitPlanMode`… `"ask"` prompts the user to confirm… Deny and ask rules are
> still evaluated regardless of what the hook returns.

This is unconditional with respect to ambient permission mode — it doesn't matter whether the
session is in `auto`, `default`, or `acceptEdits`, and it doesn't matter whether that mode came
from the agent's own frontmatter, an inherited parent mode, or a plan-default: **an explicit
`PreToolUse` `"allow"` suppresses the prompt, and an explicit `"ask"` forces it, every time.** The
existing five doc-scoped guards already prove the `"ask"` half (that's exactly what AC-4 depends
on and what instances 10/11 show still firing correctly). What none of them do today is emit the
`"allow"` half — `guard-doc-writes.sh` currently does a **silent `exit 0`** on a glob match,
meaning an in-remit write's fate is left to whatever ambient mode governs (per 1.2, unreliable).
**That gap — not `acceptEdits`, not a missing allowlist entry — is the actual root cause**, and
closing it (emit explicit `"allow"` on match) is a single, mode-independent, doc-confirmed fix that
requires no certainty about which ambient mode any given session happens to be in.

None of the paths this design allowlists are Claude Code "protected paths" (`.git`, `.claude`,
`.vscode`, etc. — all dot-prefixed dirs, plus a fixed list of dotfiles) or "critical paths" (`rm`/
`rmdir` targets), so neither of the two documented carve-outs on hook `"allow"` applies here.

---

## 2. Design overview

| FR | Agent(s) | Mechanism | New/changed files |
|---|---|---|---|
| FR-2 (general, existing 5) | `architect`, `analyst`, `data-scientist`, `teco`, `tico` (+ `security-expert`'s review guard, same core) | `guard-doc-writes.sh` core: emit explicit `"allow"` on a glob match instead of silent `exit 0` | `claude/scripts/guard-doc-writes.sh` only — no wrapper/frontmatter changes, all six wrappers inherit the fix |
| FR-1 | `cobb` | New wrapper on the same core, ask-on-mismatch (default): a topic-bounded glob union, not one folder | `claude/cobb/hooks/guard-cobb-topic-writes.sh` (new), `claude/cobb/cobb.md` (frontmatter) |
| FR-3 | `qa-engineer` | New wrapper on the same core, **pass-on-mismatch** (new 3rd-arg mode — qa-engineer also authors source/test files outside its two doc kinds; those must NOT start escalating) | `claude/qa-engineer/hooks/guard-qa-doc-writes.sh` (new), `claude/qa-engineer/qa-engineer.md` (frontmatter, add matcher) |
| FR-2 (general, broad implementer) | `tdd-engineer` | New standalone core, **inverted** (deny-list: ask only on a known-other-specialist path, allow otherwise) — evidenced (instances 7,8) and named in AC-3; `frontend-engineer`/`devops`/`graph-dba` get the same treatment only once similarly evidenced (not this round — see §5) | `claude/scripts/guard-broad-write.sh` (new), `claude/tdd-engineer/hooks/guard-tdd-broad-write.sh` (new), `claude/tdd-engineer/tdd-engineer.md` (frontmatter) |
| AC-4/AC-5 | everyone / `devops`,`graph-dba`,`qa-engineer` | No change | — |

Three truth tables, three scripts (one shared, two new) — not one script with three modes bolted
on. This mirrors the team's own precedent for "genuinely different hazard shape earns its own
core" (`security-expert`'s two independent hook cores, documented in `claude/AGENTS.md` "Hook
machinery": one reuses `guard-doc-writes.sh`, the exploitation-approval one is standalone because
the shape differs).

---

## 3. `claude/scripts/guard-doc-writes.sh` — core change (serves FR-2 general + FR-1 + FR-3)

Two changes to the existing core, both backward-compatible with all six current callers
(`architect`, `analyst`, `data-scientist`, `teco`, `tico`, `security-expert`'s review guard — none
of their wrapper invocations change, they get the fix for free):

1. **On a glob match, emit explicit `"allow"` instead of silent `exit 0`.**
2. **New optional 3rd positional arg**, `on_mismatch` (`ask` default | `pass`). `ask` is today's
   behavior (escalate on no match) — every existing caller keeps this by omitting the arg. `pass`
   is new: on no match, silently `exit 0` (today's behavior, unchanged) instead of escalating —
   for `qa-engineer`'s wrapper only (§5), whose remit is "its two doc kinds, **plus** whatever
   source/test files the task needs" — escalating those non-doc writes would be *new* friction with
   no FR/evidence behind it, not a fix.

Full replacement content:

```bash
#!/usr/bin/env bash
# guard-doc-writes.sh — shared PreToolUse core for the doc-scoped write guards.
#
# The doc-scoped agents (architect, analyst, data-scientist, teco, tico,
# cobb, qa-engineer, security-expert's review guard) each keep a thin wrapper
# in <agent>/hooks/ (wired via the agent's frontmatter `hooks:`, matcher
# `Write|Edit`) that execs this script with its parameters:
#
#   guard-doc-writes.sh '<glob>|<glob>...' '<escalation message template>' [<on_mismatch>]
#
#   $1  pipe-separated allowed-path globs; the /tmp scratchpad ('/tmp/*') is
#       always allowed and needn't be listed
#   $2  message shown to the human on escalation (only used when $3 is "ask",
#       the default); the literal token __PATH__ is replaced with the
#       (JSON-escaped) offending path. Keep templates free of double quotes
#       and backslashes — the message is spliced into JSON verbatim.
#   $3  optional, "ask" (default, omit for today's behavior) or "pass".
#       "ask": a non-matching write escalates to the human (unchanged).
#       "pass": a non-matching write silently falls through instead — for an
#       agent whose remit is genuinely wider than its doc-scoped allowlist
#       (e.g. qa-engineer, which also authors source/test files as part of
#       execution; escalating those would be new friction, not a fix).
#
# Behavior: a Write/Edit whose target MATCHES an allowed glob is explicitly
# APPROVED (PreToolUse permissionDecision "allow") — changed 2026-08-21
# (agent-permission-friction FR-1/FR-2/FR-3). Previously this was a silent
# `exit 0`, which left an in-remit write's fate to whatever ambient
# permission mode governed the session — the fresh evidence showed that is
# NOT reliable (see claude/docs/plans/agent-permission-friction.md §1: a
# subagent's frontmatter permissionMode is silently ignored/overridden by
# the parent session's mode in documented cases, including the Pro/Max/Team
# default "auto" mode). An explicit hook "allow" skips the permission prompt
# unconditionally per code.claude.com/docs/en/hooks, independent of ambient
# mode — that's what actually closes the gap. A non-matching write still
# escalates to "ask" by default ($3 unset) — unchanged, and still what AC-4
# relies on.
#
# Deliberately NOT covered: Bash. Mutating the tree via Bash would be a
# deliberate guardrail violation (prompt-guarded), whereas drifting into code
# edits via the editing tools is the realistic *accidental* failure mode this
# guard closes. See architect kaizen K-003 resolution (2026-07-08).
#
# Contract (verified 2026-08-21 against code.claude.com/docs/en/hooks):
#   - stdin: JSON with .tool_input.file_path (matcher already restricts to
#     Write/Edit).
#   - stdout JSON on a match:
#       {"hookSpecificOutput":{"hookEventName":"PreToolUse",
#         "permissionDecision":"allow","permissionDecisionReason":"..."}}
#   - stdout JSON on a mismatch (on_mismatch="ask", default):
#       {"hookSpecificOutput":{"hookEventName":"PreToolUse",
#         "permissionDecision":"ask","permissionDecisionReason":"..."}}
#   - on_mismatch="pass": no output, exit 0 (normal permission flow decides).
#   - exit 0 always (the decision is carried in the JSON, not the exit code).
#
# No hard dependency on jq: extraction tries jq, then python3. Fail-open by
# design — if the path can't be extracted, the call proceeds and the
# prompt-level guardrail backstops.

set -uo pipefail
set -f # no filename expansion — the allowed globs must reach `case` literally

allowed_globs="${1:?usage: guard-doc-writes.sh '<globs>' '<message template>' [ask|pass]}"
msg_template="${2:?usage: guard-doc-writes.sh '<globs>' '<message template>' [ask|pass]}"
on_mismatch="${3:-ask}"

input="$(cat)"

path=""
if command -v jq >/dev/null 2>&1; then
  path="$(printf '%s' "$input" | jq -r '.tool_input.file_path // empty' 2>/dev/null || true)"
elif command -v python3 >/dev/null 2>&1; then
  path="$(printf '%s' "$input" | python3 -c 'import sys,json;
try: print(json.load(sys.stdin).get("tool_input",{}).get("file_path",""))
except Exception: pass' 2>/dev/null || true)"
fi

# Fail-open: no extractable path, let it through (prompt guardrail backstops).
[ -z "$path" ] && exit 0

esc_path=""
json_escape_path() {
  esc_path="$(printf '%s' "$path" | sed 's/\\/\\\\/g; s/"/\\"/g')"
  shopt -u patsub_replacement 2>/dev/null || true # keep '&' in paths literal
}

IFS='|'
for glob in $allowed_globs '/tmp/*'; do
  case "$path" in
    $glob)
      json_escape_path
      printf '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"allow","permissionDecisionReason":"in-remit write (matches an allowed path) — auto-approved by guard, path: %s"}}\n' "$esc_path"
      exit 0
      ;;
  esac
done
unset IFS

if [ "$on_mismatch" = "pass" ]; then
  exit 0
fi

json_escape_path
msg="${msg_template//__PATH__/$esc_path}"

printf '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"ask","permissionDecisionReason":"%s"}}\n' "$msg"
exit 0
```

**Verification note for the implementer:** re-read this against the six existing wrapper
invocations (`architect`, `analyst`, `data-scientist`, `teco`, `tico`, `security-expert`'s review
guard, all shown in full in claude/AGENTS.md and each `hooks/guard-*.sh` file) — every one calls
with exactly 2 positional args, so `on_mismatch` defaults to `"ask"` and their `"ask"`-branch
message text and escalation behavior are byte-for-byte unchanged. Only the match branch changes
for them (silent → explicit allow).

---

## 4. `cobb` — FR-1

New wrapper, same core, default `ask` on mismatch (topic-bounded allowlist, not folder-bounded).

`claude/cobb/hooks/guard-cobb-topic-writes.sh` (new file):

```bash
#!/usr/bin/env bash
# PreToolUse guard for the `cobb` subagent (frontmatter `hooks:`, matcher
# `Write|Edit`). Cobb's remit is TOPIC-bounded, not folder-bounded — the team
# maintainer's job cuts across every agent's own folder (definitions, kaizen
# curation) plus a small, explicitly maintained set of cross-cutting
# MCP/agent-standards docs that live outside claude/ and skills/ entirely
# (e.g. a component README documenting MCP wiring). See
# claude/docs/requirements/agent-permission-friction.md FR-1 for the
# evidence trail (instances 1-3,5,6,9) and counter-example C2
# (docs/BACKLOG.md — genuinely out of remit, still escalates below).
#
# Allowed-path union (every entry doubled — bare + "*/"-prefixed — because
# tool_input.file_path can arrive absolute, not just repo-relative; bash
# `case` lets a leading `*` cross `/`, which is what lets the doubled sibling
# absorb an arbitrary absolute prefix ahead of the literal directory. Every
# existing guard-doc-writes.sh caller already relies on this
# (claude/architect/kaizen/history.md:342, "absolute + relative docs/plans/
# -> pass") — analyst review 2026-08-21 caught this plan's first draft
# omitting the doubled form here, which would have meant the guard never
# actually matches on the delivery shape every evidenced FR-1 instance has
# (a Task-delegated subagent write, file_path absolute):
#   claude/*/*.md, */claude/*/*.md   any agent's own top-level docs, incl.
#                                     cobb's own (<name>/<name>.md,
#                                     TESTING.md, *-notes.md, *-quirks.md,
#                                     ...) — NOTE (analyst review Finding 2):
#                                     because `case` lets a bare `*` cross
#                                     `/`, this also matches
#                                     claude/<agent>/kaizen/inbox.md (and any
#                                     other .md file at any depth under
#                                     claude/<agent>/). Scoping this to
#                                     "exactly one path segment" would need
#                                     an extglob pattern (`+([^/])`) — a
#                                     single bracket-negation `[^/]` only
#                                     constrains ONE character, not the whole
#                                     run, so it does NOT actually work in
#                                     plain `case` matching (verified: `case
#                                     "kaizen/inbox.md" in [^/]*.md) ...`
#                                     still matches) — and the shared core
#                                     doesn't use extglob today. ACCEPTED
#                                     DELIBERATELY instead: inbox.md is
#                                     frozen and nobody writes to it (FR-1's
#                                     evidence trail), so silently allowing a
#                                     write there costs nothing; the plain,
#                                     already-battle-tested glob form is kept
#                                     rather than introducing a new pattern
#                                     dialect for one low-risk edge case.
#   claude/*/kaizen/history.md, */claude/*/kaizen/history.md
#   claude/*/kaizen/plan.md, */claude/*/kaizen/plan.md
#                                     kaizen curation for any agent — cobb
#                                     curates, not the agent itself (FR-1
#                                     instance 5); kaizen/inbox.md is not
#                                     listed here either — redundant with the
#                                     claude/*/*.md entry above in any case
#   claude/README.md, */claude/README.md
#   claude/AGENTS.md, */claude/AGENTS.md
#   claude/CLAUDE.md, */claude/CLAUDE.md
#                                     team catalog + agent-context files —
#                                     cobb's own maintenance duty
#                                     (claude/AGENTS.md "Maintenance rules")
#   skills/agent-maintenance/*, */skills/agent-maintenance/*
#   skills/agent-standards/*, */skills/agent-standards/*
#                                     cobb's own skill packages
#   skills/README.md, */skills/README.md
#                                     skills catalog (shared; cobb updates its
#                                     own entries here)
#   cypher-mcp/README.md, */cypher-mcp/README.md
#                                     MCP/agent-standards doc outside
#                                     claude/skills/ (FR-1 instance 6) — a
#                                     path-only hook can't detect "documents
#                                     MCP wiring" by content, so this line is
#                                     a small, EXPLICITLY MAINTAINED list:
#                                     extend it, don't broaden the globs
#                                     above, when a new such doc surfaces.
#
# Deliberately NOT allowed (still escalates — AC-4, counter-example C2): a
# general project doc with no agent/skill/MCP relevance, e.g. docs/BACKLOG.md.
# "cobb can edit anything on the agents" was corrected by the stakeholder to
# "topic-bounded, not folder-bounded," not path-unrestricted.
#
# Thin wrapper: shared logic lives in claude/scripts/guard-doc-writes.sh
# (resolved through this file's real path, so it also works via the
# ~/.claude/agents/ symlink).
exec "$(dirname "$(readlink -f "$0")")/../../scripts/guard-doc-writes.sh" \
  'claude/*/*.md|*/claude/*/*.md|claude/*/kaizen/history.md|*/claude/*/kaizen/history.md|claude/*/kaizen/plan.md|*/claude/*/kaizen/plan.md|claude/README.md|*/claude/README.md|claude/AGENTS.md|*/claude/AGENTS.md|claude/CLAUDE.md|*/claude/CLAUDE.md|skills/agent-maintenance/*|*/skills/agent-maintenance/*|skills/agent-standards/*|*/skills/agent-standards/*|skills/README.md|*/skills/README.md|cypher-mcp/README.md|*/cypher-mcp/README.md' \
  "cobb guardrail: Write/Edit targets '__PATH__', which is outside cobb's agentic-development topic-remit (any agent's own definition file, kaizen curation for the team, MCP/agent-standards documentation) or the /tmp scratchpad. Approve only if this is genuinely agent/skill/MCP-standards work; otherwise it belongs to whichever agent actually owns that doc kind (e.g. a general project backlog item is not cobb's job — see counter-example C2)."
```

Frontmatter diff, `claude/cobb/cobb.md` (currently has no `hooks:` block at all — add one after
`permissionMode: acceptEdits`):

```yaml
permissionMode: acceptEdits
hooks:
  PreToolUse:
    - matcher: Write|Edit
      hooks:
        - type: command
          command: $HOME/.claude/agents/cobb/hooks/guard-cobb-topic-writes.sh
```

---

## 5. `qa-engineer` — FR-3

New wrapper, same core, **`pass`** on mismatch — its Write/Edit remit is "its two doc kinds, plus
whatever source/test files phase-3 execution needs" (see `qa-engineer.md` §3, "Author automated
functional tests"). An `ask`-on-mismatch guard here would escalate every test-file write it makes
today — a regression with no FR/evidence behind it. `pass` mode preserves today's status quo
(ambient mode decides) for everything except the two doc kinds, which now get the explicit-allow
fix.

`claude/qa-engineer/hooks/guard-qa-doc-writes.sh` (new file):

```bash
#!/usr/bin/env bash
# PreToolUse guard for the `qa-engineer` subagent (frontmatter `hooks:`,
# second matcher entry alongside the existing Bash destructive-ops guard).
# FR-3 (claude/docs/requirements/agent-permission-friction.md): qa-engineer's
# two versioned deliverable-doc kinds must not require a manual confirmation
# — evidenced by instance 4 (docs/test-plans/generic-cypher-mcp2.md,
# docs/test-reports/generic-cypher-mcp2-report.md).
#
# on_mismatch="pass" (NOT the shared core's "ask" default): qa-engineer also
# authors automated functional tests and drives the running app as part of
# its own execution phase (qa-engineer.md §3) — those Write/Edit calls are
# squarely in-remit too, just not doc-scoped. Escalating them would be new
# friction this FR never evidenced; "pass" leaves them to the ambient
# permission flow exactly as today, only the two doc-kind paths below change
# behavior (silent exit 0 -> explicit allow).
#
# Thin wrapper: shared logic lives in claude/scripts/guard-doc-writes.sh
# (resolved through this file's real path, so it also works via the
# ~/.claude/agents/ symlink).
exec "$(dirname "$(readlink -f "$0")")/../../scripts/guard-doc-writes.sh" \
  'docs/test-plans/*|*/docs/test-plans/*|docs/test-reports/*|*/docs/test-reports/*' \
  "unused — on_mismatch is pass, no escalation message is ever rendered" \
  pass
```

Frontmatter diff, `claude/qa-engineer/qa-engineer.md` (add a second matcher entry to the existing
`hooks.PreToolUse` list):

```yaml
hooks:
  PreToolUse:
    - matcher: Bash
      hooks:
        - type: command
          command: $HOME/.claude/agents/qa-engineer/hooks/guard-destructive-ops.sh
    - matcher: Write|Edit
      hooks:
        - type: command
          command: $HOME/.claude/agents/qa-engineer/hooks/guard-qa-doc-writes.sh
```

---

## 6. `tdd-engineer` — FR-2 general (broad-implementer case)

`tdd-engineer` has **evidence** (instances 7, 8 — a test file and a source file, both squarely
in-remit) and is one of AC-3's two named examples, so — per the root-cause finding in §1 —
`acceptEdits` alone is **not** sufficient and something must be added. Its remit is genuinely "the
whole codebase, this task" — there's no single folder/kind to allowlist the way the doc-scoped
agents have. An allow-list guard would be wrong here (everything it legitimately touches can't be
enumerated); the right shape is the **inverse**: allow by default, escalate only on a small,
principled deny-list of paths that are **known to belong to a different specialist's documented
deliverable-path convention**, or that the stakeholder left genuinely unresolved (U1 — see below).

### 6.1 New standalone core: `claude/scripts/guard-broad-write.sh`

```bash
#!/usr/bin/env bash
# guard-broad-write.sh — shared PreToolUse core for a "broad implementer"
# write guard: the INVERSE shape from guard-doc-writes.sh.
#
# guard-doc-writes.sh is an ALLOW-LIST: escalate everything except a small
# set of paths that ARE the whole remit (a doc-scoped specialist like
# architect/analyst/tico/cobb/qa-engineer, whose Write/Edit legitimately
# touches nothing else). This core is a DENY-LIST: allow everything except a
# small set of paths KNOWN to belong to a DIFFERENT specialist's documented
# deliverable-path convention (or a genuinely unresolved team-governance
# ambiguity) — for an agent whose own remit is "the whole codebase, this
# task" and has no single folder/kind to allowlist. See
# claude/docs/plans/agent-permission-friction.md §6.
#
#   guard-broad-write.sh '<glob>|<glob>...' '<escalation message template>'
#
#   $1  pipe-separated globs that are OUT of this agent's remit — a match
#       escalates. No implicit /tmp/* handling needed here (nothing to
#       protect a scratchpad from for a broad-remit agent).
#   $2  message shown to the human on escalation; __PATH__ is replaced with
#       the (JSON-escaped) offending path. Keep templates free of double
#       quotes and backslashes.
#
# Behavior: a Write/Edit whose target MATCHES a listed glob escalates
# (PreToolUse permissionDecision "ask"); everything else — the overwhelming
# common case, source/test/any-other-in-task-file work — is explicitly
# APPROVED (permissionDecision "allow"), skipping the ambient permission-mode
# prompt regardless of which mode governs the session (see root-cause
# finding, agent-permission-friction.md §1).
#
# A SEPARATE core from guard-doc-writes.sh rather than a third mode bolted
# onto it: same precedent as security-expert's two independent hook cores
# (claude/AGENTS.md "Hook machinery") — a genuinely different truth table
# earns its own script.
#
# Same stdin/stdout contract, jq->python3 extraction, and fail-open behavior
# as guard-doc-writes.sh (contract verified 2026-08-21 against
# code.claude.com/docs/en/hooks).

set -uo pipefail
set -f

denied_globs="${1:?usage: guard-broad-write.sh '<globs>' '<message template>'}"
msg_template="${2:?usage: guard-broad-write.sh '<globs>' '<message template>'}"

input="$(cat)"

path=""
if command -v jq >/dev/null 2>&1; then
  path="$(printf '%s' "$input" | jq -r '.tool_input.file_path // empty' 2>/dev/null || true)"
elif command -v python3 >/dev/null 2>&1; then
  path="$(printf '%s' "$input" | python3 -c 'import sys,json;
try: print(json.load(sys.stdin).get("tool_input",{}).get("file_path",""))
except Exception: pass' 2>/dev/null || true)"
fi

# Fail-open: no extractable path, let it through (prompt guardrail backstops).
[ -z "$path" ] && exit 0

IFS='|'
for glob in $denied_globs; do
  case "$path" in
    $glob)
      esc_path="$(printf '%s' "$path" | sed 's/\\/\\\\/g; s/"/\\"/g')"
      shopt -u patsub_replacement 2>/dev/null || true
      msg="${msg_template//__PATH__/$esc_path}"
      printf '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"ask","permissionDecisionReason":"%s"}}\n' "$msg"
      exit 0
      ;;
  esac
done
unset IFS

printf '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"allow","permissionDecisionReason":"in-remit implementer write — auto-approved by guard"}}\n'
exit 0
```

### 6.2 `tdd-engineer` wrapper

`claude/tdd-engineer/hooks/guard-tdd-broad-write.sh` (new file):

```bash
#!/usr/bin/env bash
# PreToolUse guard for the `tdd-engineer` subagent (frontmatter `hooks:`,
# matcher `Write|Edit`). tdd-engineer's remit is genuinely "the whole
# codebase, this task" (red-green-refactor over whatever source/test files
# the task needs) — no single folder to allowlist, so this uses the INVERSE
# shape (claude/scripts/guard-broad-write.sh): allow by default, escalate
# only on a path that's known to belong to a DIFFERENT specialist's
# documented deliverable-path convention, or that the stakeholder left
# genuinely unresolved. See
# claude/docs/requirements/agent-permission-friction.md FR-2 (instances 7,8;
# AC-3's tdd-engineer example) and U1 below.
#
# Deny-list (escalates). Every entry doubled — bare + "*/"-prefixed — for
# the same reason as guard-doc-writes.sh's callers (Claude Code's
# tool_input.file_path can arrive absolute; a leading "*" is what absorbs an
# arbitrary absolute prefix ahead of the literal directory — analyst review
# 2026-08-21 Finding 1, same fix as §4 above). Also folds in Finding 5
# (skills/agent-maintenance/*, skills/agent-standards/*, cypher-mcp/README.md
# were named in §4 as cobb's topic-remit but missing from this deny-list):
#   docs/plans/*, */docs/plans/*             architect / teco-coordination /
#                                             data-scientist-ml
#   docs/reviews/*, */docs/reviews/*         analyst / security-expert /
#                                             data-scientist-ml
#   docs/requirements/*, */docs/requirements/*   tico
#   docs/manuals/*, */docs/manuals/*         tico
#   docs/test-plans/*, */docs/test-plans/*   qa-engineer
#   docs/test-reports/*, */docs/test-reports/*   qa-engineer
#   claude/*/*.md, */claude/*/*.md           agent definitions / kaizen — cobb
#   claude/*/kaizen/*, */claude/*/kaizen/*   (same claude/*/*.md caveat as §4
#                                             Finding 2 applies here too: also
#                                             catches kaizen/inbox.md, accepted
#                                             the same way — frozen, no-op)
#   claude/README.md, */claude/README.md     team catalog/context — cobb
#   claude/AGENTS.md, */claude/AGENTS.md
#   claude/CLAUDE.md, */claude/CLAUDE.md
#   skills/README.md, */skills/README.md
#   skills/agent-maintenance/*, */skills/agent-maintenance/*   cobb's own
#   skills/agent-standards/*, */skills/agent-standards/*       skill packages
#   cypher-mcp/README.md, */cypher-mcp/README.md   cobb's topic-remit (FR-1
#                                             instance 6) — a tdd-engineer
#                                             write here should escalate too
#   docs/BACKLOG.md, */docs/BACKLOG.md       U1 (agent-permission-friction.md):
#                                             the stakeholder was explicitly
#                                             unsure whether a tdd-engineer ->
#                                             BACKLOG.md write is in-remit —
#                                             left here so it keeps asking,
#                                             same as today, deliberately NOT
#                                             resolving U1 either way
#
# Everything else -- source code, test files, any other in-task file --
# is explicitly allowed. Thin wrapper: shared logic lives in
# claude/scripts/guard-broad-write.sh (resolved through this file's real
# path, so it also works via the ~/.claude/agents/ symlink).
exec "$(dirname "$(readlink -f "$0")")/../../scripts/guard-broad-write.sh" \
  'docs/plans/*|*/docs/plans/*|docs/reviews/*|*/docs/reviews/*|docs/requirements/*|*/docs/requirements/*|docs/manuals/*|*/docs/manuals/*|docs/test-plans/*|*/docs/test-plans/*|docs/test-reports/*|*/docs/test-reports/*|claude/*/*.md|*/claude/*/*.md|claude/*/kaizen/*|*/claude/*/kaizen/*|claude/README.md|*/claude/README.md|claude/AGENTS.md|*/claude/AGENTS.md|claude/CLAUDE.md|*/claude/CLAUDE.md|skills/README.md|*/skills/README.md|skills/agent-maintenance/*|*/skills/agent-maintenance/*|skills/agent-standards/*|*/skills/agent-standards/*|cypher-mcp/README.md|*/cypher-mcp/README.md|docs/BACKLOG.md|*/docs/BACKLOG.md' \
  "tdd-engineer guardrail: Write/Edit targets '__PATH__', which looks like another specialist's documented deliverable path (a plan/review/requirements/manual/test-plan/test-report doc, an agent definition or kaizen file, a team catalog or skill package, an MCP-standards doc, or the project backlog). Approve only if tdd-engineer genuinely owns this write for the current task; otherwise it belongs to whichever agent normally authors that doc kind."
```

Frontmatter diff, `claude/tdd-engineer/tdd-engineer.md` (currently no `hooks:` block — add one
after `permissionMode: acceptEdits`):

```yaml
permissionMode: acceptEdits
hooks:
  PreToolUse:
    - matcher: Write|Edit
      hooks:
        - type: command
          command: $HOME/.claude/agents/tdd-engineer/hooks/guard-tdd-broad-write.sh
```

### 6.3 `frontend-engineer`, `devops`, `graph-dba`, `coder`, `security-expert` — explicitly not touched this round
- **`frontend-engineer`, `devops`, `graph-dba`**: same "broad implementer, no doc-scoped guard"
  shape as `tdd-engineer` in principle, and `guard-broad-write.sh` would extend to them cheaply if
  ever needed — but **zero live evidence** exists for any of the three in this requirements round.
  Adding guards speculatively would break the evidence-first discipline this whole interview ran
  on. **Recommendation, not a decision made here**: extend `guard-broad-write.sh` to them only once
  a similar friction instance is actually observed and evidenced (mirrors how `coder`'s and U1's
  gaps were explicitly deferred rather than guessed at).
- **`coder`**: named in the design brief's "by the same logic" list, but the requirements doc
  explicitly scopes `coder`'s friction **out of this round** ("Keep `coder`'s own friction triggers
  … out of scope — do not touch them"). Not touched here regardless of whether the same logic would
  apply; deferred to the forecasted phase 2 alongside `coder`'s other triggers.
- **`security-expert`**: already has a narrow, correctly-scoped `Write|Edit` guard
  (`docs/reviews/*` only, via the same shared `guard-doc-writes.sh` core) — it is not "no guard at
  all," so it's not in the broad-implementer bucket, and it already gets the core's explicit-allow
  fix for free (§3). No change needed.

---

## 7. Files changed (full list)

| File | Change |
|---|---|
| `claude/scripts/guard-doc-writes.sh` | Modify — explicit `"allow"` on match; new optional `on_mismatch` 3rd arg |
| `claude/scripts/guard-broad-write.sh` | New — deny-list core |
| `claude/cobb/hooks/guard-cobb-topic-writes.sh` | New wrapper |
| `claude/cobb/cobb.md` | Frontmatter: add `hooks:` block |
| `claude/qa-engineer/hooks/guard-qa-doc-writes.sh` | New wrapper |
| `claude/qa-engineer/qa-engineer.md` | Frontmatter: add `Write\|Edit` matcher entry |
| `claude/tdd-engineer/hooks/guard-tdd-broad-write.sh` | New wrapper |
| `claude/tdd-engineer/tdd-engineer.md` | Frontmatter: add `hooks:` block |
| `claude/AGENTS.md` | "Hook machinery" section: document the **one new core** (`guard-broad-write.sh` — `guard-doc-writes.sh` is modified in place, not new), the three new wrappers, and the core-behavior change (explicit allow on match, `on_mismatch` arg) |
| `claude/README.md` | Catalog entries for `cobb`, `qa-engineer`, `tdd-engineer` (new guard, one line each, matching the existing per-agent guard-description style); the "Hooks (...)" enumeration bullet in the deployment section gains `cobb` and `tdd-engineer`; **and** the Deployment-section summary prose (~lines 92-99, *"All guards are thin wrappers over **two** shared cores... the **five** doc-scoped write guards over `guard-doc-writes.sh`... and the **three** destructive-ops guards..."*) — update the core count to **three** and the `guard-doc-writes.sh` caller count/roster to include `cobb` and `qa-engineer` (with a note that `qa-engineer` uses the new `pass`-on-mismatch mode), so this paragraph doesn't read stale after the plan lands (analyst review Finding 3) |
| `claude/cobb/kaizen/history.md` | Dated entry: new topic-bounded guard added |
| `claude/qa-engineer/kaizen/history.md` | Dated entry: new doc-write guard added |
| `claude/tdd-engineer/kaizen/history.md` | Dated entry: new broad-implementer guard added |
| `architect`, `analyst`, `data-scientist`, `teco`, `tico`, `security-expert` — no file changes | They inherit the core-behavior fix automatically; no wrapper/frontmatter edit needed |
| `~/.claude/settings.json` | **Not edited by this plan** — flagged in §1.2 as a stakeholder/`teco` decision (personal, untracked, outside a `docs/plans/*` write's remit) |

No change to `claude/scripts/guard-destructive-ops.sh`, the `devops`/`graph-dba`/`qa-engineer`
destructive-ops wrappers, or `security-expert/hooks/guard-exploitation-approval.sh` — AC-5 and
that whole mechanism are untouched, as required.

---

## 8. Sequencing

1. Update `guard-doc-writes.sh` (§3) in isolation; sanity-check the six existing wrapper
   invocations still parse (2-arg calls, `on_mismatch` defaults to `ask`).
2. Add `guard-broad-write.sh` (§6.1) — new file, no existing caller yet, no regression risk.
3. Add the three new wrapper scripts (§4, §5, §6.2); `chmod +x` each.
4. Update `cobb.md`, `qa-engineer.md`, `tdd-engineer.md` frontmatter to wire the new hooks. No
   redeploy step needed — `$HOME/.claude/agents/<name>` is a whole-directory symlink to
   `claude/<name>` (verified: `cobb`, `qa-engineer`, `tdd-engineer` all symlinked today), so a new
   file under an already-symlinked tree is live immediately.
5. Manual verification pass — one `Write`/`Edit` per AC (see §9); a fresh session per agent to
   avoid any single-session mode-selection confound from §1.2.
6. Update `claude/AGENTS.md`, `claude/README.md`, and the three touched agents' `kaizen/history.md`
   in the same change (standing convention, not a separate unit).
7. Hand to `analyst` for review before merge (this plan document is itself the artifact under
   review right now — implementation is the next unit, gated the same way per the coordination
   ledger).

---

## 9. Acceptance-criteria verification

- **AC-1 (FR-1):** `cobb` editing `claude/<any-agent>/<any-agent>.md`, `claude/<any-agent>/kaizen/
  {history,plan}.md`, `claude/README.md`/`AGENTS.md`/`CLAUDE.md`, `skills/agent-{maintenance,
  standards}/*`, `skills/README.md`, or `cypher-mcp/README.md` → matches
  `guard-cobb-topic-writes.sh`'s allowlist → explicit `"allow"` → no prompt. `cobb` editing
  `docs/BACKLOG.md` (counter-example C2) → no glob matches → `"ask"` (default mismatch mode) →
  prompt still appears. **Verified both for a repo-relative path and, critically, for an absolute
  `file_path`** (`case "/home/.../claude/cobb/cobb.md" in */claude/*/*.md) ...` → match; `case
  ".../docs/BACKLOG.md" in <the full allowed-glob union>) ...` → no match) — this is the exact
  delivery shape every evidenced FR-1 instance has (a `Task`-delegated subagent write), and the
  gap analyst review Finding 1 caught in the first draft was specifically that this case was
  untested. Both halves verified against both path forms.
- **AC-2 (FR-3):** `qa-engineer` writing `docs/test-plans/<slug>.md` or
  `docs/test-reports/<slug>-report.md` (repo-root or component-relative) → matches
  `guard-qa-doc-writes.sh`'s allowlist → explicit `"allow"` → no prompt. Verified.
- **AC-3 (FR-2, general):** `architect` writing `docs/plans/*` → already-allowlisted glob, now gets
  explicit `"allow"` instead of silent pass-through → no prompt (was previously falling through to
  the unreliable ambient mode per §1.2 — instances 10/11 showed this still prompting even on a
  passed guard). `tdd-engineer` editing a test file or source file for its current task → no match
  against `guard-tdd-broad-write.sh`'s deny-list → explicit `"allow"` → no prompt. Both verified.
- **AC-4 (safety net preserved):** Every guard's mismatch/match branch that should escalate still
  emits `"ask"` exactly as before (five existing agents: unchanged message/behavior, only the
  match-branch changed; `cobb`: `"ask"` on a topic-outside path; `tdd-engineer`: `"ask"` on a
  known-other-specialist path). Per the hooks doc, a hook's `"ask"` always wins regardless of
  ambient mode — this was already true and remains true; nothing in this design weakens it.
  `data-scientist`'s C1 counter-example (`tests/eval/probe_ministral_judge.py`, outside its
  `docs/plans|reviews` allowlist) is untouched — that wrapper and core behavior on mismatch is
  identical to today. **For `tdd-engineer` specifically, verified with an absolute path against
  every deny-list entry** — `claude/README.md`, `claude/AGENTS.md`, an arbitrary
  `claude/<agent>/<agent>.md`, `skills/agent-standards/SKILL.md`, `cypher-mcp/README.md`, and
  `docs/BACKLOG.md` all escalate under an absolute `file_path`, while an ordinary source/test file
  path (`falkor-chat/server/falkorchat/guards.py`, `server/tests/test_guards.py`) still falls
  through to allow — this is the case analyst review Finding 1 flagged as silently failing open in
  the first draft (deny-list entries that could never match on an absolute path, so a write to
  `claude/README.md` would have hit the guard's default allow instead of escalating). Verified for
  all six existing agents plus `cobb` and `tdd-engineer`.
- **AC-5 (destructive-ops guards untouched):** `guard-destructive-ops.sh` and its three wrappers
  (`devops`, `graph-dba`, `qa-engineer`) are not modified anywhere in this design (§7). Verified by
  inspection — zero references to that script or its wrappers in any change above.

---

## 10. Decisions flagged for `teco`/the stakeholder (not resolved unilaterally here)

1. **`~/.claude/settings.json`'s undocumented blanket `Edit`/`Write`/`NotebookEdit` allow rule**
   (§1.2) — recommend narrowing or removing it once this design ships, so AC-4 is guaranteed by the
   hooks (portable, repo-tracked, deterministic) rather than accidentally by a personal, unscoped,
   machine-local override that a teammate/clean-clone won't have. This plan does not edit that
   file — it's outside a `docs/plans/*` write's remit and outside `cobb`'s file-editing authority
   for this repo.
2. **`frontend-engineer`/`devops`/`graph-dba` broad-implementer guards** (§6.3) — designed-for
   (reuses `guard-broad-write.sh` trivially) but deliberately not added without evidence. Revisit
   if/when a friction instance surfaces for any of them.
3. **`coder`'s friction and instance U1** — untouched, exactly as the requirements doc scoped them:
   phase 2, once more live evidence exists. `guard-tdd-broad-write.sh`'s deny-list keeps
   `docs/BACKLOG.md` in the "still asks" bucket specifically so U1 stays unresolved rather than
   silently decided by this change.
