# `defaultMode` as a lever for the Task/Agent-delegation classifier gap — Design

> **Status:** active · **Owner:** `cobb` · **Tracks:** —

**Background (not re-derived here):** `claude/docs/plans/write-guard-classifier-gap.md` (design,
refuted) and its `claude/docs/plans/write-guard-classifier-gap-coordination.md` (ledger, close-out)
root-caused and then empirically tested a `permissions.allow` `Edit(path)` settings-rule supplement
to close a gap in the original RCA (`skills/agent-standards/claude-code.md`, 2026-08-24; also
`claude/docs/requirements/agent-permission-friction2.md` open question 3): a `PreToolUse` hook's
explicit `"allow"` does not suppress the auto-mode classifier's confirmation prompt for a
Task/Agent-delegated write. The empirical test (coordination doc §U7) **refuted** the rules
approach too — a matching `Edit(path)` allow rule in `.claude/settings.local.json` did not
suppress the prompt for a delegated write either. Both suppression mechanisms this team had tried
are confirmed dead for the delegated-write case. This document investigates the one lever both
docs left open and unresolved: **changing `defaultMode` away from `auto`.**

---

## 1. What the current docs say (verified 2026-08-24 against `code.claude.com/docs`)

### 1.1 The mode table

Fetched fresh from `code.claude.com/docs/en/permission-modes` and `.../permissions`:

| Mode | What runs without asking | Classifier involved? |
|---|---|---|
| `default` (labeled **Manual**) | Reads only | No — every edit/Bash/network action prompts |
| `acceptEdits` | Reads, file edits, and a small fixed Bash allowlist (`mkdir`, `touch`, `rm`, `rmdir`, `mv`, `cp`, `sed`) in the working directory | **No** — no classifier exists in this mode |
| `plan` | Reads, plus classifier-approved commands **only when auto mode is available** | Conditionally, for exploration commands only |
| `auto` | Everything, background-classifier-reviewed | **Yes** — a second model reviews every non-trivial action |
| `dontAsk` | Only pre-approved tools | No |
| `bypassPermissions` | Everything | No |

`acceptEdits` is the only mode besides `auto` itself that is remotely relevant here: it is the one
mode where file edits are silently approved **without any classifier ever being invoked** — not
because a rule or hook out-races the classifier (the mechanism the refuted design relied on), but
because in `acceptEdits` there is no classifier in the loop at all for that decision.

### 1.2 The subagent inheritance rule — the actual mechanism, precisely quoted

This is the passage neither `write-guard-classifier-gap.md` nor its coordination doc had in hand.
From `code.claude.com/docs/en/sub-agents`, "Permission modes" subsection (verbatim):

> "If the parent uses `bypassPermissions` or `acceptEdits`, this takes precedence and can't be
> overridden. If the parent uses [auto mode], the subagent inherits auto mode and any
> `permissionMode` in its frontmatter is ignored: the classifier evaluates the subagent's tool
> calls with the same block and allow rules as the parent session."

And from `code.claude.com/docs/en/permission-modes`, "How auto mode handles subagents":

> "1. Before a subagent starts, the delegated task description is evaluated... 2. While the
> subagent runs, each of its actions goes through the classifier with the same rules as the parent
> session, and any `permissionMode` in the subagent's frontmatter is ignored. 3. When the subagent
> finishes, the classifier reviews its full action history..."

Reading both together resolves `write-guard-classifier-gap.md` §2.1's undecided ambiguity (whether
"the same rules as the parent session" meant rules can bypass the classifier for a subagent action,
or that subagent actions always reach the classifier regardless of rules) — but not in the design's
favor. The empirical refutation (§U7) already settled that a settings-rule doesn't bypass the
classifier for a delegated write. What these two passages add is the **positive** mechanism: it is
specifically **`auto` mode as the parent's own ambient mode** that forces every dispatched
subagent's actions through the classifier, discarding whatever the subagent's own frontmatter
`permissionMode` says. **If the parent session is not in `auto`** — specifically if it's in
`acceptEdits` or `bypassPermissions` — the subagent doesn't go through the classifier at all; it
runs under the parent's mode, "and can't be overridden." Mode inheritance, not a rule race, is the
one documented path that plausibly closes this gap. This is a materially different mechanism than
the refuted design (§5 there), and isn't undermined by that refutation.

**Gap in the docs:** neither passage states what happens when the parent is in `default`, `plan`,
or `dontAsk` — only `bypassPermissions`/`acceptEdits` (takes precedence) and `auto` (forces auto)
are covered. Not load-bearing for this document's analysis (§3 below only considers `acceptEdits`
as the candidate), but worth flagging as genuinely undocumented if a future design considers those
modes.

### 1.3 Hooks are mode-independent — confirms `acceptEdits` doesn't lose the escalation guarantee

From `code.claude.com/docs/en/permissions`, "Extend permissions with hooks" (verbatim):

> "When Claude Code makes a tool call, `PreToolUse` hooks run before the permission prompt... The
> hook output can deny the tool call, force a prompt, or skip the prompt to let the call proceed."

Nothing here is gated on permission mode — a hook's `"ask"` decision is documented to force a
prompt regardless of mode, the same mechanism that already makes AC-2 (`agent-permission-friction2.md`
— "safety net preserved") hold today. This means switching a delegating parent to `acceptEdits`
would **not** weaken the per-agent escalation guarantee the guard hooks provide for a genuinely
out-of-remit path: the hook's `"ask"` still fires; only the in-remit `"allow"` path changes, from
"allow, then re-litigated by a classifier that may re-prompt anyway" (today, under `auto`) to
"allow, then no classifier exists to re-litigate it" (under `acceptEdits`).

---

## 2. A finding this investigation surfaced that neither prior document had: the fix has already been "declared" team-wide, and it doesn't work

Every one of the 13 agents under `claude/` already carries `permissionMode: acceptEdits` in its own
frontmatter (`grep -l 'permissionMode: acceptEdits' claude/*/*.md` — all 13 match, confirmed
2026-08-24). If a custom subagent's own frontmatter `permissionMode` controlled its starting mode
when run as the **primary** session (`claude --agent <name>`), every one of these agents would
already be starting in `acceptEdits`, and this whole investigation would be moot for the top-level
case (only the delegated-dispatch case, governed by the §1.2 inheritance rule, would remain open).

It doesn't. Two independent pieces of evidence, read together:

- **The documented decision order for what mode a session starts in**
  (`code.claude.com/docs/en/permission-modes`, "Which mode a session starts in") is: (1) the
  `--permission-mode` flag, (2) `permissions.defaultMode` in a settings file, (3) the built-in
  default. **A custom agent's own frontmatter is not a step in this order at all** — `permissionMode`
  in frontmatter is documented only as a field consulted by the §1.2 subagent-dispatch inheritance
  rule, never as a session-start input.
- **Direct empirical confirmation**, already on record in `write-guard-classifier-gap.md` §1 (Recap):
  the `teco` session's own transcript across both 2026-08-23 incidents carries an explicit
  `"type":"permission-mode"` record showing it stayed in **literal `auto`** mode continuously — despite
  `teco/teco.md`'s frontmatter declaring `permissionMode: acceptEdits`.
- **This repo's actual live setting, read directly this session:** `~/.claude/settings.json`
  contains `"permissions": {"defaultMode": "auto", ...}` — an explicit, deliberate pin (not merely
  the Pro/Max/Team built-in default falling through), at **user/global scope**. No project-level
  `.claude/settings.json` or `.claude/settings.local.json` in this repo currently sets
  `permissions.defaultMode` at all.

Put together: every agent's `permissionMode: acceptEdits` frontmatter line is, today, dead
configuration for controlling that agent's own top-level starting mode — the actual mode is decided
purely by `~/.claude/settings.json`'s explicit `auto` pin, with the frontmatter never consulted for
that decision. It only becomes live at all under the §1.2 dispatch-time rule, and even there, since
the delegating parent is *always* itself running in `auto` (nothing in this repo currently sets
`defaultMode` any other way), the rule's `auto`-forces-`auto` branch fires every time — the
`acceptEdits`-branch (where the subagent's frontmatter would matter) never gets a chance to apply.
**Today, 100% of delegated dispatch in this repo goes through the classifier, regardless of any
agent's frontmatter, because the one setting that actually decides it (`~/.claude/settings.json`)
says `auto`.**

This reframes the investigation: it isn't "should we add a new mode declaration nobody has tried,"
it's "the team already tried exactly this, at exactly the frontmatter layer, and it silently never
took effect because that's not how session-start mode is decided." Fixing it means changing the
setting that's actually load-bearing — `defaultMode` in a settings file (or `--permission-mode` per
launch) — not the frontmatter, which is already there and already inert for this purpose.

---

## 3. Blast radius and scope options

`defaultMode` can be set at three scopes that matter here, per
`code.claude.com/docs/en/permission-modes` ("Start in a different permission mode") and this
session's direct read of the live files:

| Scope | File | Who's affected | Current state |
|---|---|---|---|
| Global / user | `~/.claude/settings.json` | **Every Claude Code session on this machine, in every project** — not just `graphmind-ai-lab`. Any other repo the maintainer works in inherits whatever this says. | `"permissions.defaultMode": "auto"` — explicit, live today |
| Project | `<repo root>/.claude/settings.json` | Every session started inside this repo — every one of the 13 agents (top-level *and* delegated), plus any ad hoc human session with no `--agent` flag at all. **Not scoped to `teco` or to coordination sessions specifically** — no mechanism keys `defaultMode` by which agent is running. Outranks the user-level file per settings precedence. | Not set (file exists, no `permissions.defaultMode` key) |
| Per-launch | `--permission-mode acceptEdits` CLI flag | Exactly one session, exactly once. Nothing persisted, nothing to roll back. | N/A — opt-in every time |

**No finer scope exists.** There is no per-agent, per-tool-call, or "only when this session
delegates" setting — `defaultMode`/`permissionMode` is a whole-session scalar, and the §1.2
inheritance rule hands that same scalar down to every subagent a session dispatches. Two corollaries
worth stating plainly:

- **You cannot keep `auto`'s classifier coverage for a coordinator's own actions while getting
  `acceptEdits`-style silent delegation for what it dispatches.** Switching the parent's mode
  switches *all* of that parent's own actions too, not just its `Agent`/`Task` calls.
- **Runtime toggling around each delegation isn't automatable.** Claude Code documents plainly:
  "Asking Claude in chat to change the permission mode doesn't work" — `Shift+Tab` (or the
  `--permission-mode` flag at launch) is a human/CLI action, not something an agent can invoke on
  itself mid-turn. A "flip to `acceptEdits` right before dispatching, flip back after" workflow
  would require the human operator to do it by hand around every single `Agent` call in a
  `teco`-coordinated run — not a realistic standing practice.

**Other consumers checked (root `AGENTS.md`, `claude/AGENTS.md` roster):** `opencode/` and `kiro/`
are separate tools with their own permission models entirely — `defaultMode` is Claude-Code-specific
and doesn't touch them. `cypher-mcp`, `mcp-monitor`, `salesperson`, `falkor-chat` are components, not
distinct Claude Code identities; whichever of the 13 agents or ad hoc human sessions work in them are
already covered by the project/global rows above. The one real "other consumer" is: **any other
project on this machine**, if the global file is the one changed — genuinely unknowable blast radius
from inside this repo, since it depends on what else the maintainer uses Claude Code for outside
`graphmind-ai-lab`.

---

## 4. What a switch actually trades away

`acceptEdits`'s Bash auto-approval is a **small, fixed, filesystem-only allowlist** — `mkdir`,
`touch`, `rm`, `rmdir`, `mv`, `cp`, `sed`. Everything else — `pytest`, `npm test`, `git commit`,
`docker`/`redis-cli` calls (`graph-dba`, `devops`), `./scripts/audit-team.sh`, any project script —
reverts to a plain per-call confirmation prompt, because there is no classifier in `acceptEdits` to
silently clear them the way `auto`'s decision order does today. `auto` mode's classifier also
silently blocks (not just approves) a long, actively-maintained list of dangerous patterns — force
push, `git reset --hard`, `curl | bash`, secret exfiltration, IaC destroy, and more
(`code.claude.com/docs/en/permission-modes`, "What the classifier blocks by default") — entirely
without a human in the loop. None of that exists under `acceptEdits`; a human would see a prompt for
each instead, which is safer in the sense that nothing gets silently *allowed* that shouldn't be, but
is a large step back in the specific thing `auto` mode was adopted to reduce: interruption volume.

This matters more than it looks, because of the §1.2 inheritance rule: switching a delegating
session (`teco`, or any of `architect`/`analyst`/`data-scientist`/`security-expert`/`tico`, all of
which also carry the `Agent` tool per the roster) to `acceptEdits` doesn't just change *that
session's own* Bash friction — it hands the same loss of classifier coverage to **every subagent it
dispatches**, for the duration of that delegation. A typical `teco`-coordinated unit routinely
involves `coder`/`tdd-engineer` running test suites, `qa-engineer` or `graph-dba` shelling out to
`redis-cli`/`docker`, or `analyst` grepping and diffing — none of that is the guarded doc-write
problem this investigation is chasing, but all of it would newly prompt under `acceptEdits`, where
today it's silently classifier-cleared. The write-confirmation friction being solved (guard-scoped,
occasional, one prompt per guarded write) is very plausibly smaller in total volume than the Bash
friction this trade would introduce across a whole delegation chain (unbounded, one prompt per
non-trivial shell command any dispatched agent runs). This is the central reason this document does
not recommend the switch as a standing default, at either scope in §3's table.

---

## 5. Recommendation

**Do not change `defaultMode` at global or project scope.** The mechanism is real and now precisely
understood (§1.2) — unlike the refuted rules approach, this one is documented to work by removing
the classifier from the loop entirely rather than trying to out-race it — but the blast radius at
either persisted scope is "every session, every Bash call, indefinitely," to fix a narrower,
occasional, already-partially-mitigated (hooks correctly emit `"allow"`/`"ask"`; only the
delegated-write case is affected) friction. §4's cost-benefit doesn't clear the bar at either scope
in §3's table, and there's no narrower persisted scope available to reach for instead.

**If the mechanism is worth empirically confirming anyway** (research value: nobody has verified
`acceptEdits`-parent inheritance the way `write-guard-classifier-gap-coordination.md` §U7 verified —
and refuted — the rules approach), the only low-commitment way to do it is the per-launch flag:
start one `teco` session as `claude --agent teco --permission-mode acceptEdits`, deliberately for a
single, low-Bash-volume coordination unit, and observe whether a delegated `analyst`/`tdd-engineer`
write is silently approved this time. Nothing persists; reverting is simply not passing the flag next
time. This is optional — the recommendation to not adopt a standing default holds regardless of the
outcome, given §4 — but it would turn "docs say this should work" into the same empirical footing
`write-guard-classifier-gap-coordination.md` reached (and refuted) for the rules approach.

**Recommended standing position: stay on `auto`, accept the documented delegated-write friction.**
This is the same shape of conclusion `agent-permission-friction2.md` AC-3 already reached for
`coder` specifically ("a pre-existing condition shared with all five phase-1-fixed agents, not a
regression... verifying or closing it depends on the live isolation test... outside this document's
scope") — this document extends that framing to the whole team and closes out the `defaultMode`
question the earlier RCA had left open, rather than reopening a path to fix it. The gap is now fully
explained (§1–2), not mysterious; there is no clean, narrowly-scoped fix; the honest answer is to
document the limitation and move on, the same way `write-guard-classifier-gap-coordination.md`'s
close-out already did for the rules approach.

**Separate, smaller follow-up worth deciding (not resolved here):** §2's finding that every agent's
`permissionMode: acceptEdits` frontmatter line is currently dead configuration for its stated
purpose is a piece of misleading team-wide configuration independent of whether `defaultMode` ever
changes. Worth a future decision — remove it team-wide as inert, or leave it as declared intent in
case a session-start-honors-frontmatter behavior is ever documented — but that's a 13-file cleanup
with its own blast-radius question, out of scope for this document. Logged to `cobb`'s own
`kaizen/plan.md`.

---

## 6. Rollback

- **Per-launch pilot (§5):** nothing to roll back — don't pass the flag on the next session.
- **If a scope from §3 were ever adopted despite this recommendation:** revert is a single JSON key
  edit (`permissions.defaultMode` back to `"auto"`, or delete the key to fall back to the built-in
  default) in whichever file was changed — mechanically trivial; the cost analyzed in this document
  is behavioral (prompt volume during the time it's live), not structural or hard to reverse.

## Cross-references

- `claude/docs/plans/write-guard-classifier-gap.md` — the refuted `permissions.allow` rule design;
  this document picks up the one lever it left open.
- `claude/docs/plans/write-guard-classifier-gap-coordination.md` — the ledger and empirical
  refutation (§U7) of the rules approach; §1 and §2 above build on its Recap and Close-out without
  re-deriving them.
- `claude/docs/requirements/agent-permission-friction2.md` — open question 3, the original trace
  back to this whole line of investigation; AC-3's framing is what §5's recommendation extends to
  the whole team.
- `skills/agent-standards/claude-code.md` — carries the original classifier/hook RCA; this document's
  §1.2/§1.3 quotes and §2 finding are being folded in as a durable addition in the same change (see
  `claude/cobb/kaizen/history.md`).
