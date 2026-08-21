# Agent permission-escalation friction — Coordination

> **Status:** archived · **Owner:** `teco` · **Tracks:** — (—)

## Close-out
Feature delivered, reviewed twice (plan gate + diff-scoped implementation gate, both `analyst`,
both approve), and committed: `93c3a39` (implementation), `4a35a48` (requirements doc archived by
`tico`), `949c41b` (plan + review docs archived by `architect`/`analyst`). One process deviation
occurred and was corrected mid-run (see "Process deviation" below) — assessed as low content-risk,
both affected deliverables were independently re-verified by a properly-typed specialist before
acceptance. `~/.claude/settings.json`'s undocumented blanket Edit/Write/NotebookEdit allow rule
(flagged in `docs/plans/agent-permission-friction.md` §1.2) remains a stakeholder decision, not
resolved by this coordination — relayed to the user directly, not tracked as an open unit here.

## Goal
Implement the requirements at `claude/docs/requirements/agent-permission-friction.md`
(Status: Ready for design as of 2026-08-21) — reduce redundant permission-escalation prompts on
legitimate, in-remit agent `Write`/`Edit` actions (FR-1 `cobb` topic-bounded remit, FR-2 general
governing rule, FR-3 `qa-engineer` test-plan/report paths) while preserving the safety net for
genuinely out-of-remit writes (AC-4) and leaving the destructive-ops guards untouched (AC-5).
`coder`'s specific friction and instance U1 are explicitly out of scope this round (deferred to a
forecasted phase 2).

## Material fact for the design step
All 13 subagents already carry `permissionMode: acceptEdits` in frontmatter, added
2026-07-24 (`f80edab`, "permissionMode: acceptEdits across all 13 subagents"), whose own commit
message states — as a verified fact at the time — "PreToolUse hooks fire before any
permission-mode check, so a hook's 'ask' decision still forces the prompt under acceptEdits."
That explains escalation on genuinely out-of-remit paths (correct, per AC-4). It does **not**
explain the requirements doc's evidence: fresh 2026-08-20/21 instances of the **plain default**
confirm-before-Edit prompt firing on hook-free, in-remit writes by `cobb`, `qa-engineer`, and
`tdd-engineer` — all three already `acceptEdits` since July. This contradiction is unresolved and
is the first thing the design step must run down before proposing a fix — a candidate hypothesis
(unconfirmed) is that a subagent's own `permissionMode` frontmatter may not apply when it is
launched via `Task`/`Agent` delegation from an orchestrating session (e.g. a `teco` session), as
opposed to being run as the top-level `claude --agent <name>` session — i.e. permission mode may
not inherit down through delegation the way a `PreToolUse` hook (which is wired per spawned agent
regardless of nesting) does. Verify against current docs rather than assuming either explanation.

## Units

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict |
|---|---|---|---|---|---|
| U1 | `cobb` | `a3671bdce543d0dc1` | accepted | `claude/docs/plans/agent-permission-friction.md` | `analyst` (`aed057e50a0a6a24c`) → approve (Pass 2) |
| U2 | `cobb` | `a3671bdce543d0dc1` | accepted | hook/frontmatter/catalog diff — committed `93c3a39` | `analyst` (`a725ec2c6b0040d1e`, fresh + typed) → approve |
| U3 | `architect`/`analyst`/`tico` | `abcbe65feffdc8eb9` / `ac176f94c85a532e6` / `a72f61c09de86d66f` | accepted | archive-status flips (plan, review, requirements) — committed `949c41b` (plan+review), `4a35a48` (requirements, tico self-committed) | — |

U2 depends on U1's plan being accepted. U3 is closing bookkeeping, dispatched only once U2 is
accepted.

## Documentation-impact scan
- `claude/scripts/guard-doc-writes.sh` (shared core, doc'd behavior may change) and any new
  guard script(s) for `cobb`/`qa-engineer`.
- `claude/AGENTS.md` "Hook machinery" section — describes the shared cores and per-agent wrappers
  today; must reflect any new guard(s) or core behavior change.
- Each touched agent's `<name>/<name>.md` frontmatter (`hooks:`, possibly `permissionMode`).
- Each touched agent's `kaizen/history.md` — dated entry, per the standing "adding/editing an
  agent" convention in `claude/AGENTS.md`.
- `claude/README.md` catalog entry for any agent whose hook/permission behavior is now
  user-visible-different (only if the catalog documents that level of detail today — verify).
- No `claude/docs/HISTORY.md`/`BACKLOG.md` exist yet (`claude/` hasn't adopted that convention) —
  nothing to update there.

## U1 outcome — root-cause finding (verified independently by `teco`)
`cobb` traced the friction to a real gap: `guard-doc-writes.sh` only ever emitted an explicit
`"ask"` on a mismatch; on a match it did a **silent `exit 0`**, leaving the write's fate to
whatever ambient permission mode governed the session — which, per current docs (`sub-agents`,
`permission-modes`, `hooks`), a `Task`-delegated subagent's own frontmatter `permissionMode` does
**not** reliably control (it's overridden/ignored when the parent session is in `bypassPermissions`/
`acceptEdits`, or in `auto` mode — the documented Pro/Max/Team default). An explicit hook `"allow"`
is unconditional regardless of ambient mode and is the actual, doc-confirmed fix.

Also surfaced, and independently confirmed by `teco` (`cat`/`stat` on `~/.claude/settings.json`):
an undocumented, unscoped `"permissions":{"allow":["Edit","Write","NotebookEdit"]}` entry, file
timestamp 2026-08-20 19:47:22 (same day as the evidence). This is a personal, machine-local,
non-repo file outside any documented convention — the plan correctly does not edit it, and flags
it as a decision for the stakeholder (narrow/remove it once the hook fix ships, so AC-4's safety
net rests on the portable, repo-tracked hooks rather than this accidental local override).

## Review 1 outcome (`analyst`, `claude/docs/reviews/agent-permission-friction.md`)
**Verdict: needs changes.** High-severity, mechanically-fixable bug: every existing
`guard-doc-writes.sh` caller writes each allowed-path glob as a **doubled pair** (bare +
`*/`-prefixed sibling, e.g. `docs/plans/*|*/docs/plans/*`) because `tool_input.file_path` can
arrive absolute — `claude/architect/kaizen/history.md:342` records this was deliberately
smoke-tested both ways. The plan's two new glob lists (§4 `guard-cobb-topic-writes.sh`, §6.2
`guard-tdd-broad-write.sh`'s deny-list) both introduce bare-only `claude/*`/`skills/*`/
`cypher-mcp/*` entries with no doubled sibling — for cobb this means the guard never actually
matches on an absolute path (fails safe: still asks, but FR-1/AC-1 never fires); for tdd-engineer
it's worse — the same missing prefix means those deny-list entries never match either, so writes
to `claude/README.md`/`claude/AGENTS.md`/agent definitions/etc. silently fall through to the
guard's **default allow**, contradicting the deny-list's own intent and §9's AC-4 claim. Two
smaller findings fold into the same fix pass: `claude/*/*.md` also (once fixed) matches the frozen
`kaizen/inbox.md` (Moderate, contradicts §4's own comment) and tdd-engineer's deny-list omits
`skills/agent-maintenance/*`/`skills/agent-standards/*`/`cypher-mcp/README.md` even though §4
names them as cobb's remit (Minor). Two documentation findings: `claude/README.md`'s existing
"two shared cores"/"five doc-scoped guards" summary prose isn't in §7's doc-impact list and will
go stale (Moderate); §7 misdescribes "two new cores" when only one is new (Minor). Everything else
— backward compatibility with all six existing callers, AC-2/AC-3/AC-5 delivery, `qa-engineer`'s
`pass`-mode reasoning, U1 preservation, general script mechanics, sequencing — verified sound.
Recommendation: a revision pass on the two new glob lists + §7's doc list, not a redesign;
re-review can be scoped to just the touched globs.

## Process deviation — noted, not hidden
`teco` omitted `subagent_type` on the `Agent` dispatches for U1 (design) and Review Pass 1, so
both ran as `general-purpose` rather than the actual `cobb`/`analyst` subagents — no real
`cobb.md`/`analyst.md` system prompt, tool restrictions, or `PreToolUse` write-guard hooks were
active, despite the briefs instructing each to act as that persona. Checked both deliverables'
actual file writes: both stayed within the paths the real guards would have allowed anyway
(`claude/docs/plans/*`, `claude/docs/reviews/*`), and `analyst`'s findings were independently
verifiable (executed `case` tests, a `teco`-reproduced `~/.claude/settings.json` check) — no
content defect found, but the specialist guardrails/knowledge base genuinely weren't in play for
U1 and Review Pass 1. Review Pass 2 and U2 (via `SendMessage` resume) inherit whatever identity
the original dispatch had (`general-purpose`) — can't be corrected mid-thread. Correction applied
going forward: any *fresh* `Agent` dispatch in this coordination now explicitly passes
`subagent_type`.

## Notes
- Requirements doc (`claude/docs/requirements/agent-permission-friction.md`) is the upstream
  artifact — read it by path, don't paraphrase from this coordination doc.
- Out of scope, unaffected: destructive-ops guards (`guard-destructive-ops.sh` and its three
  wrappers), `coder`'s own friction triggers, instance U1's classification, git-commit authority
  scoping.
