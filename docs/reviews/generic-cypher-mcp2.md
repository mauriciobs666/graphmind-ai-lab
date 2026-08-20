# Generic Cypher MCP — team-wide kaizen inbox rollout — plan-gate review

> **Status:** active · **Owner:** `analyst` · **Tracks:** — (M7)

## Scope & verdict

Static plan-gate review of `docs/plans/generic-cypher-mcp2.md` (Status: active, owner
`architect`) against `docs/requirements/generic-cypher-mcp2.md` (FR-1…FR-14, AC-1…AC-13, Status:
Ready for design). Read both documents in full. Independently re-verified the plan's own
live-verification claims (did not take them at face value): `cypher-mcp/server.py`'s
write-authorization logic (read in full, 792 lines), the live `kaizen_graph_dba` graph state, the
`CLAUDE_CODE_SESSION_ID` environment variable, current `kaizen/inbox.md` line counts, the doc
files cited (`claude/AGENTS.md`, `claude/README.md`, `skills/agent-maintenance/SKILL.md`,
`claude/graph-dba/graph-dba.md`, `claude/cobb/cobb.md`), every one of the eleven migrating agents'
frontmatter `tools:` lists, all five doc-scoped write-guard hook wrapper scripts, and
`claude/scripts/audit-team.sh` (run live). Cross-checked FR-1…FR-14/AC-1…AC-13 against the plan's
§4 unit table and §5 AC-mapping for silent drops, and spot-checked the plan's file lists against
actual repo state for the M5-B1-shaped failure mode (an operative-prompt file that *directs*
behavior, omitted from the touched-file list).

**CPG:** considered, not relevant — independently re-ran the plan's own live check
(`mcp__cypher__query(graph='cypher-mcp', cypher='MATCH (n) RETURN count(n)')`) and got the
identical result: no graph named `cypher-mcp` (or any `cpg_claude`-shaped graph) is loaded; the
only two loaded CPGs (`cpg_falkorchat`, `cpg_salesperson`) are unrelated application codebases.
This delivery touches agent prompts, skill docs, hook scripts, and kaizen data — none of it code a
Joern CPG models.

**Verdict: needs changes.** Two blockers: a factually incorrect claim that no agent-wiring step is
needed (three of the eleven migrating agents cannot call `mcp__cypher__query` under their current
frontmatter), and a stated acceptance criterion (AC-11) with no unit that actually executes it.
Four more major findings — all in the same failure class the review brief asked to check for
(M5's own B1: a fixed/enumerated file list under-covering real scope) — are below.

---

## Findings

### Blockers

**B1 — Three of the eleven migrating agents (`teco`, `tico`, `data-scientist`) do not have
`mcp__cypher__query` in their frontmatter `tools:` allowlist; the plan's claim that "no
agent-wiring/allowlist step is needed anywhere in this rollout" (§2) is false for them.**

Verified directly by reading each file's frontmatter in full:
- `claude/data-scientist/data-scientist.md:4` — `tools: Read, Grep, Glob, Bash, Write, Edit, WebFetch, WebSearch, Agent`
- `claude/teco/teco.md:4` — `tools: Read, Grep, Glob, Bash, Agent, SendMessage, ListAgents, Write, Edit`
- `claude/tico/tico.md:4` — `tools: Read, Grep, Glob, Bash, Write, Edit, WebFetch, WebSearch, Agent, AskUserQuestion`

None lists `mcp__cypher__query`. Only `architect` and `analyst` (of the five "doc-scoped-write-guard"
agents the plan names as a group) actually carry it — confirmed both by frontmatter and by
`claude/README.md`, which states the "hence the `mcp__cypher__query` entry in this agent's
`tools:` allowlist" reasoning **only** for `architect` and `analyst` (lines 9 and 16); it makes no
such claim anywhere for `data-scientist`, `teco`, or `tico`. The plan's §2 finding conflates "has a
doc-scoped write guard" (governs `Write`/`Edit` targets only) with "has the MCP tool in its
allowlist" (a separate, unrelated frontmatter property) — the two properties happen to coincide for
`architect`/`analyst` but not for the other three.

Consequence: units **A6** (`teco`), **A10** (`data-scientist`), and **A11** (`tico`) as currently
scoped cannot execute — a restrictive `tools:` list makes an unlisted MCP tool invisible to the
agent, so none of these three could make the migration write, the FR-2 write path afterward, or a
future FR-8a-carrying entry.

Suggested fix: add `mcp__cypher__query` to all three agents' frontmatter `tools:` lists as a new,
explicitly-owned step (naturally folds into `cobb`'s D1/D2 pass, or into each of A6/A10/A11's own
done-condition) — and re-audit the other seven non-restrictive agents once more to be sure none of
them silently *do* carry a restrictive list the plan's "mirrors M5's finding" shortcut missed (this
review independently confirmed `cobb`, `coder`, `devops`, `frontend-engineer`, `qa-engineer`,
`tdd-engineer` all currently declare no `tools:` line at all — i.e. "All tools" — so only these
three are affected, but that should be stated as directly-verified, not inferred by category).

**B2 — AC-11 (delete `claude/graph-dba/kaizen/inbox.md`) has no owning unit anywhere in §4's
implementation table.**

The plan states in §1 ("`graph-dba`'s own already-frozen `kaizen/inbox.md`, deleted too
(FR-14/AC-11)") and §5 (AC-11's row: "`git status`/`ls claude/graph-dba/kaizen/` confirms
`inbox.md` absent") that this file deletion is in scope and will be verified. But unit **G0**
(`graph-dba`, the only unit that could plausibly own it) lists its Files as "(no tracked files —
live graph DDL/ops only)" and its done-condition covers only: index/constraint provisioning,
re-confirming zero `:KaizenEntry` nodes, `GRAPH.DELETE kaizen_graph_dba`, and
`claude/graph-dba/graph-dba.md`'s Learning-capture edit. No other unit (D1–D3, A1–A11, Q1, Q2)
touches `claude/graph-dba/kaizen/inbox.md` either. Grepped the entire plan text for the string —
the only three hits are the two scope claims above and the AC-mapping row; none is an executing
step.

Suggested fix: add the file deletion (with the AC-2-style "content already present in
`kaizen_graph_dba`, confirmed" gate FR-14 itself requires) to G0's Files/Done-condition explicitly.

### Major

**M1 — `claude/scripts/audit-team.sh` check 1 unconditionally requires `kaizen/inbox.md` to exist
for every agent, and no unit updates it — this plan will break the team's own certification
tooling.** Read the script directly (`claude/scripts/audit-team.sh:76`):
`[ -f "$CL/$a/kaizen/plan.md" ] && [ -f "$CL/$a/kaizen/history.md" ] && [ -f "$CL/$a/kaizen/inbox.md" ]`
— no exception for any agent. Ran it live: it currently reports `PASS  graph-dba: kaizen plan +
history + inbox present`, specifically *because* `graph-dba`'s frozen `inbox.md` still exists on
disk today. The moment this plan deletes even one migrated agent's `kaizen/inbox.md` (FR-3/FR-4),
that agent's check 1 flips to `FAIL`; once all twelve are migrated it fails for the whole team,
permanently — and the same unconditional check would also fail for FR-12's own goal (a *newly
created* agent, born graph-backed, never gets a `kaizen/inbox.md` at all). `agent-maintenance`
`SKILL.md` §4 treats this script's `FAIL`/exit-1 result as the gate to certification ("Fix any FAIL
before judging the rest") — this is a real, permanent regression to the team's own tooling that
none of AC-1…AC-13 would catch (they verify the feature, not the auditor). No unit's Files column
names this script. Suggested fix: add updating check 1 (accept either the file-triple or, for a
migrated agent, plan/history plus a live `kaizen_team` presence check) as part of D2 or G0's scope.

**M2 — Five doc-scoped write-guard hook wrapper scripts hardcode `<agent>/kaizen/inbox.md` as an
allowed `Write`/`Edit` path and name it in the human-facing escalation message; none is touched by
any unit.** Confirmed by reading all five in full:
`claude/architect/hooks/guard-plan-doc-writes.sh`, `claude/analyst/hooks/guard-review-doc-writes.sh`,
`claude/data-scientist/hooks/guard-ds-doc-writes.sh`, `claude/teco/hooks/guard-coordination-doc-writes.sh`,
`claude/tico/hooks/guard-tico-doc-writes.sh` — each execs `guard-doc-writes.sh` with an
allowed-glob string containing `<agent>/kaizen/inbox.md|*/<agent>/kaizen/inbox.md`, and an
escalation message describing it by name ("...the agent's own `kaizen/inbox.md`, or the `/tmp`
scratchpad"). Two consequences once these five agents migrate: (a) the escalation text a human
sees on any future `Write`/`Edit` outside the plan-doc directory is now stale/misleading — it
references a file that no longer exists; (b) more concretely, the glob still *allows* a `Write` to
that path — if a migrated agent ever attempted (stale habit, drift, a bug) to write to its own
now-deleted `kaizen/inbox.md`, the guard would **silently permit it** rather than escalate,
quietly resurrecting a file FR-4's whole design says should stay deleted with git history as the
only archive. Suggested fix: fold narrowing/removing the `kaizen/inbox.md` glob entry (and the
matching escalation-message clause) into each of A3/A4(cobb's edit for architect is N/A — architect
is A3)/A7/A10/A11/A6's own done-condition, since these are exactly the five agents whose units
already touch their own prompt file.

**M3 — `claude/teco/teco.md`'s own coordination-duty prose (outside its "Learning capture"
section) generically describes every delegate's raw capture as `claude/<agent>/kaizen/inbox.md`,
and unit A6 doesn't reach it.** Two load-bearing passages, confirmed by direct read: line 72
("Fencing carve-out" — "carve out that delegate's own `claude/<agent>/kaizen/inbox.md`
explicitly... a delegate's inbox resolves into the excluded tree via its deployment symlink") and
line 88 ("Learnings ride the handoff" — "confirm it filed a dated entry in its own learnings inbox
(`claude/<agent>/kaizen/inbox.md`) — a one-line check, not a gate"). Both are teco's *own*
operational checklist for handling every other delegate, not teco's self-description of its own
learning capture — so they sit outside the "Learning-capture section" scope A6's done-condition
targets ("prompt's Learning-capture section directs new learnings to `kaizen_team`... no remaining
`inbox.md`-append instruction"). Once agents migrate, both checks become wrong for those delegates:
there is no file to carve a fence around, and no `inbox.md` to confirm a dated entry in — the
correct check becomes a `kaizen_team` read filtered by `author`. This is the same failure shape as
M5's own B1 finding (`docs/reviews/generic-cypher-mcp.md` B1): an operative-prompt passage that
*directs* behavior, silently missing from the migration file list, because the list was built
around "each agent's own Learning-capture section" rather than "every place this convention is
referenced." Confirms the review brief's suspicion directly: running the plan's own §5 AC-8
verification command literally (`grep -rln 'kaizen/inbox\.md\|append.*inbox' claude/
skills/agent-maintenance/SKILL.md`) surfaces `claude/teco/teco.md`, all five hook wrapper scripts
from M2, and `claude/scripts/audit-team.sh` from M1 — meaning `qa-engineer`'s Q1/Q2 pass, using the
plan's own prescribed method, would hit all three of these gaps as unplanned defects at acceptance
time, not implementation time. Suggested fix: add these two teco.md passages (and a corresponding
audit of any other coordinator-level cross-reference) to A6's scope explicitly, and change AC-8's
verification note to say what to *do* with a hit that isn't a Learning-capture section, not just
how to classify it.

**M4 — Open item 7 (cobb self-editing its own `cobb.md` Learning-capture section, unit A4):
second opinion is that the existing self-maintenance carve-out does not clearly cover this case,
and the safer default is to route this one edit through a different agent.** Read `cobb.md:86` in
full: "you are the maintainer, so same-run promotion with full §1/§2 bookkeeping is in-bounds for
you alone." That carve-out is scoped to cobb's own **judgment call** about promoting a specific raw
kaizen entry it has already verified — not to mechanically applying an already-approved external
plan's edit to its own governing prompt section. Every one of the other ten migrating agents gets
this exact class of edit made by a *different* agent (`cobb`); `cobb` doing its own is the one
asymmetric case in the whole rollout, and it is also the one case with no independent check that
the edit landed correctly (the other ten get `cobb`'s edit reviewed implicitly by the fact a
different actor did it against a written recipe). Recommend A4's prompt-edit half be reassigned to
a different owner (e.g. `coder`, mirroring D3's role touching frozen doc-strings verbatim from a
recipe, or `architect`) rather than defaulting to self-edit — or, if the stakeholder/plan-gate
prefers keeping it self-edited, record that as an explicit decision rather than a default.

### Minor

**m1 — §2's per-agent `kaizen/inbox.md` line-count survey has already drifted for `tico`, beyond
the "immaterial" band the plan itself allows.** Live re-measured: `tico` 18 lines today, not 47 as
both the plan and the requirements doc's "Problem & current state" context state — this is a real
`cobb` distillation pass (commit `5c79e32`, "chore(tico): distill kaizen inbox"), not measurement
noise. `architect` similarly drifted (24 lines today vs. 19 stated) — from the architect's own
`CLAUDE_CODE_SESSION_ID` investigation being logged there mid-plan. Immaterial to migration
mechanics (each unit reads whatever's actually present at execution time), but worth flagging so
whoever executes A11/A3 isn't surprised by a smaller-than-documented file, and so the "small drift
is expected and immaterial" framing isn't over-applied to a case that's actually a full distillation
pass.

**m2 — D2's done-condition understates the scope of the `skills/agent-maintenance/SKILL.md` §5
rewrite.** Read §5 in full (~140 lines): it is graph-dba-specific procedural detail throughout —
the literal `kaizen_graph_dba` graph name and `agent='graph-dba'` appear repeatedly in a
step-by-step four-step distillation script, and the "Inbox template" is currently positioned as
`graph-dba`'s one named exception, not a general fallback. Generalizing this to "the graph-backed
pattern is the described default... Inbox template kept only as fallback for any not-yet-migrated
agent" (D2's done-condition) is a substantial content rewrite — every `kaizen_graph_dba` /
`agent='graph-dba'` occurrence needs to become an `author`-filtered `kaizen_team` operation, and the
per-agent four-step recipe needs a name-parameterized rewrite. Not wrong, just under-scoped as
written; an implementer following the done-condition literally could plausibly ship a smaller diff
than the section actually needs.

**m3 — Unit ownership concentrates almost entirely on `cobb`: 13 of 15 units (D1, D2, A1–A11) name
`cobb` as full or half owner, including `cobb`'s own A4 migration.** This doesn't violate FR-13
(each unit still stands alone), but the plan's own "independently dispatchable" framing implicitly
suggests throughput independence that doesn't actually exist here — real rollout velocity is gated
by `cobb`'s availability across nearly the whole plan. Worth surfacing as a sequencing risk for
whoever coordinates execution (`teco`), even if it doesn't change the plan's shape.

---

## What's solid

- **The core mechanism-reuse claim is correct and independently verified**: `cypher-mcp/server.py`'s
  `authorize_write()`, `_author_claims()`, `_kaizen_entry_create_map_spans()`, and
  `_CURATOR_CLEAR_RE` (read in full) never inspect the `graph` parameter — the shared-graph decision
  genuinely costs zero server logic changes, exactly as claimed.
- **`kaizen_graph_dba` is genuinely empty** — live-queried and confirmed zero `:KaizenEntry` nodes —
  so §3.2's "schema-only, not a data migration" framing for retiring the key holds.
- **`CLAUDE_CODE_SESSION_ID` is real and independently re-confirmed in a second, unrelated session**
  (this review's own): `env | grep CLAUDE_CODE_SESSION_ID` returned a value matching this session's
  own scratchpad UUID segment, exactly the same corroboration pattern the plan used, now reproduced
  from a cold session — directly answers the plan's own open item 3 request for independent
  re-confirmation before baking the mechanism into eleven prompts.
- **The shared-`kaizen_team`-graph vs. per-agent-graphs trade-off (§3.1/3.2) is well-reasoned and
  correctly traces back to M5's own pre-authorized override and revisit trigger** — verified both
  quotes directly in `generic-cypher-mcp-graph.md` (§0's override language, §2/§5's "revisit if this
  pattern extends past `graph-dba`" trigger) and in `graph-dba.md`'s modeling-principle quote; the
  "not a tenant-property mega-graph" distinction is sound given the working set is self-pruning by
  design (M5 already established this).
- **The nine other migrating agents' own operative-prompt Learning-capture sections are each a
  single, self-contained, correctly-scoped edit target** — spot-checked all nine directly; no
  cross-referencing surprises there (unlike `teco`, per M3 above).
- **AC-1…AC-10, AC-12, AC-13's verification mapping (§5) is concrete and mostly at the right
  altitude** — live checks where live checks are the actual proof (AC-7's one-query team-wide read
  is the direct FR-7 evidence), static checks where static suffices.

---

## Open questions

- Whether the `mcp__cypher__query` frontmatter addition for `teco`/`tico`/`data-scientist` (B1)
  should be its own tiny unit or folded into each of A6/A10/A11 — either works, but it needs an
  explicit owner before those three units are dispatched.
- Whether `docs/BACKLOG.md`'s M7 per-agent status markers (C-705…C-715, §4.4) get flipped as part of
  each A-unit's own done-condition, or purely via `teco`'s standing documentation-curator duty at
  integration time — the plan doesn't say either way, and it isn't clearly a defect either way, but
  the plan-gate reviewer or `teco` should pick one explicitly rather than leaving it implicit.
