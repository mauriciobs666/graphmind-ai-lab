# `defaultMode` as a lever for the Task/Agent-delegation classifier gap — Design

> **Status:** archived · **Owner:** `cobb` · **Tracks:** —
> **Version:** 2 (revised 2026-08-24, folding in `claude/docs/reviews/permission-default-mode.md`'s
> findings) · **Reviews:** `claude/docs/reviews/permission-default-mode.md` (verdict: approve with
> suggestions → corrections folded in below)

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

**The docs do resolve the other parent modes — by a general rule stated one paragraph earlier, not
a gap.** From `code.claude.com/docs/en/sub-agents`, "Permission modes," the paragraph immediately
preceding the passage quoted above (verbatim):

> "Set `permissionMode` to choose the permission mode a subagent runs in... If you leave it unset,
> the subagent inherits the main conversation's mode, which starts as auto mode on Pro, Max, and
> Team plans unless your settings or your organization change it. **Setting it overrides that mode,
> except in the cases described below.**"

"The cases described below" are exactly the two branches already quoted
(`bypassPermissions`/`acceptEdits`-parent takes precedence; `auto`-parent forces `auto`). That's an
exhaustive if/else: for every *other* parent mode — `default` (Manual), `plan`, `dontAsk` — the
general rule applies, and a dispatched subagent's own frontmatter `permissionMode` governs
independently of the parent's. See §3 for why this third branch doesn't change this document's
recommendation despite looking, at first glance, like a narrower option than `acceptEdits`-parent.

### 1.3 Hooks are mode-independent on paper — but this repo already has live counter-evidence that `"ask"` enforcement is unconfirmed under `auto`, and that's not yet tested under `acceptEdits`

From `code.claude.com/docs/en/permissions`, "Extend permissions with hooks" (verbatim):

> "When Claude Code makes a tool call, `PreToolUse` hooks run before the permission prompt... The
> hook output can deny the tool call, force a prompt, or skip the prompt to let the call proceed."

Nothing here is gated on permission mode — on the docs' own account, a hook's `"ask"` decision
forces a prompt regardless of mode. But `skills/agent-standards/claude-code.md` — the exact file
this document's §1.2/§1.3 findings are folded into, in the same change — already carries a dated,
live-reproduced, filed-upstream finding that contradicts relying on that account at face value:
four isolated live tests (2026-08-21, Claude Code 2.1.238, under `auto` mode) found a `PreToolUse`
hook, confirmed correctly wired and confirmed to compute `"ask"` in isolation, **did not** pause
execution for the real matching command, from either a Task-dispatched subagent or the main session
itself — "matcher-agnostic and context-agnostic, not a narrow subagent-dispatch-only gap." That
file's own working hypothesis is that `auto` mode's classifier layer is what silently overrides the
`"ask"` decision — which would mean removing the classifier (switching to `acceptEdits`) could
plausibly fix this as a side effect, consistent with what this section originally concluded. But
that's a hypothesis, not a confirmed fact: all four tests ran under `auto`; nobody has verified
whether a hook's `"ask"` reliably fires under `acceptEdits` specifically. This document's claim that
"the hook's `"ask"` still fires" under `acceptEdits` is therefore a docs-supported inference, not an
empirically closed question — and the KB entry, from the moment it's folded into the same file,
should not be read as having settled it.

**Consequence for the argument, not the conclusion:** this doesn't overturn §5's recommendation —
if anything, an unresolved `"ask"`-reliability risk is one more reason to stay on `auto` rather than
switch, since it means the out-of-remit safety net's actual behavior under `acceptEdits` is unknown,
not merely "small-cost." It does mean §5's optional pilot, if ever run, needs to check both
directions, not just the one this document originally described — see §5.

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

**A fourth option the §1.2 general rule opens — a `default`-parent — is dominated by `acceptEdits`,
not a free win.** Since setting a subagent's `permissionMode` frontmatter "overrides that mode,
except" the two named branches (§1.2), a coordinator running in plain `default` (Manual) would
leave every dispatched subagent's already-declared `permissionMode: acceptEdits` frontmatter live
at dispatch time — no classifier, same net effect as the `acceptEdits`-parent candidate above. But
this doesn't beat that candidate: both land the dispatched subagent in the identical place (its own
`acceptEdits`, either by the general rule or by the takes-precedence branch), while `default` is
strictly worse for the *parent's own* actions — Manual mode auto-approves nothing at all (every
edit, every Bash command, every network call prompts), where `acceptEdits` at least silently clears
file edits and the small filesystem allowlist. A `default`-parent buys nothing an
`acceptEdits`-parent doesn't already buy, and costs strictly more for the parent's own work.
Dominated; not carried forward as a separate row in §4's cost analysis.

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
session — `teco`, or in principle any of the other 12 agents, since **all 13** can dispatch via
`Agent` (six — `architect`/`analyst`/`data-scientist`/`security-expert`/`tico`/`teco` — declare it
explicitly in a `tools:` list; the other seven — `coder`/`tdd-engineer`/`qa-engineer`/`graph-dba`/
`devops`/`frontend-engineer`/`cobb` — omit `tools:` entirely and so inherit the full built-in set,
`Agent` included) — to `acceptEdits` doesn't just change *that session's own* Bash friction — it
hands the same loss of classifier coverage to **every subagent it dispatches**, for the duration of
that delegation, through a wider set of possible delegating parents than a glance at explicit
`tools:` lists alone would suggest. A typical `teco`-coordinated unit routinely
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
occasional, already-partially-mitigated (hooks are *designed* to emit `"allow"`/`"ask"` correctly —
§1.3 flags that whether `"ask"` reliably *enforces* is itself unconfirmed under `auto`, and untested
under `acceptEdits`; only the delegated-write case is affected either way) friction. §4's
cost-benefit doesn't clear the bar at either scope in §3's table, and there's no narrower persisted
scope available to reach for instead.

**If the mechanism is worth empirically confirming anyway** (research value: nobody has verified
`acceptEdits`-parent inheritance the way `write-guard-classifier-gap-coordination.md` §U7 verified —
and refuted — the rules approach), the only low-commitment way to do it is the per-launch flag:
start one `teco` session as `claude --agent teco --permission-mode acceptEdits`, deliberately for a
single, low-Bash-volume coordination unit, and observe **both** directions, not just the one this
document originally proposed testing: (a) whether a delegated `analyst`/`tdd-engineer` in-remit
write's hook `"allow"` is now silently approved, and (b) — per §1.3's unresolved `"ask"` question —
whether a delegated write genuinely *outside* that agent's remit still produces a hook `"ask"`
confirmation prompt under `acceptEdits`, rather than assuming the safety net holds just because it's
undocumented as mode-gated. Nothing persists; reverting is simply not passing the flag next time.
This is optional — the recommendation to not adopt a standing default holds regardless of the
outcome, given §4 — but it would turn "docs say this should work" into the same empirical footing
`write-guard-classifier-gap-coordination.md` reached (and refuted) for the rules approach, on both
the `"allow"` and `"ask"` paths rather than just one.

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
- `skills/agent-standards/claude-code.md` — carries the original classifier/hook RCA, including the
  2026-08-21 `PreToolUse` "ask"-enforcement-gap finding §1.3 now cross-references; this document's
  §1.2/§1.3 quotes and §2 finding were folded in as a durable addition in commit `773328c` (see
  `claude/cobb/kaizen/history.md`). The KB entry itself has since been backported with the matching
  caveat (its "Resolution/update, 2026-08-24" entry now cross-references the "ask"-enforcement-gap
  callout the same way §1.3 here does), closing the split-across-two-documents gap
  `claude/docs/reviews/permission-default-mode.md` flagged as an open question.
- `claude/docs/reviews/permission-default-mode.md` — `analyst`'s review (approve with suggestions)
  of v1 of this document; both Major findings and the Minor finding are folded into this v2
  in place, per the same precedent as `write-guard-classifier-gap.md`'s v2 revision.
