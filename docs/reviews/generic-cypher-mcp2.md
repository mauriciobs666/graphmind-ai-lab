# Generic Cypher MCP — team-wide kaizen inbox rollout — plan-gate review

> **Status:** archived · **Owner:** `analyst` · **Tracks:** — (M7)

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

---

## Pass 2 — Re-gate of Version 2 (2026-08-20)

*(Everything above is Pass 1, against Version 1 of the plan. It is left intact; this section is a
fresh, independent pass against Version 2 by a different `analyst` instance, cold session.)*

### Scope & verdict

Plan-gate re-review of `docs/plans/generic-cypher-mcp2.md` **Version 2** (Status: active, owner
`architect`) against `docs/requirements/generic-cypher-mcp2.md` (FR-1…FR-14, AC-1…AC-13) and the
stakeholder decision recorded in `docs/plans/generic-cypher-mcp2-coordination.md` ("Stakeholder
decision (2026-08-20)"). Read all four documents in full, plus Pass 1's own review above.

Independently re-verified every claim V2 makes about Pass 1's findings, rather than accepting its
self-summary — and re-derived the state V2 describes from the live system:

- Read `claude/{teco,tico,data-scientist}`'s frontmatter `tools:` lines directly (B1).
- Ran `claude/scripts/audit-team.sh` live, and read its check-1 logic at
  `claude/scripts/audit-team.sh:74-79` (M1).
- Read all five doc-scoped write-guard wrappers in full (M2).
- Read `claude/teco/teco.md:72` and `:89` verbatim (M3).
- Ran `git show --stat ccf9c8b` and read the full 43-file list (M4, and V2's §2 claims about what
  the commit did and did not touch).
- **Executed** `authorize_write()` from `cypher-mcp/server.py` against the plan's own §4.2
  migration query shape, in `cypher-mcp/.venv` (§3.6's load-bearing author-binding claim).
- Live-queried the FalkorDB instance for the current `kaizen_*` graph inventory and entry counts.
- Grepped all 12 agent prompts, `skills/agent-maintenance/SKILL.md`, `cypher-mcp/server.py`,
  `cypher-mcp/README.md`, `cypher-mcp/tests/test_server.py`, `docs/BACKLOG.md`, and every
  `claude/*/kaizen/inbox.md` for the occurrences §2/§4 enumerate — and **ran the plan's own AC-8
  verification command verbatim** to see what it actually surfaces.

**CPG:** not applicable — this delivery is agent-prompt, FalkorDB-kaizen-schema and documentation
work; no Joern-modelled application code is touched. (Confirmed incidentally by the live
loaded-graph list obtained for the `kaizen_*` inventory: the only CPGs present are
`cpg_falkorchat` and `cpg_salesperson`, unrelated application codebases. The dispatch brief also
directed skipping the CPG check for this artifact.)

**Verdict: needs changes.** V2 is a large, genuine improvement — its re-triage of Pass 1 is
honest and, on the six items I re-checked, five are exactly right; §3.6 is new, correct, and I
confirmed it by execution rather than by reading. But two blockers remain, both in the same
failure class Pass 1 named and V2 explicitly set out to close: **an enumerated list asserting a
state of the repo that the repo does not actually have.**

- **P2-B1** — the plan asserts FR-12/AC-9 is already delivered. It provably is not; the new-agent
  convention still seeds a `kaizen/inbox.md`, and the AC-9 check the plan prescribes cannot detect
  that. This is the *third* requirement the never-delete decision collides with, and V2's
  otherwise-correct "the decision is blanket" reasoning stopped one clause short.
- **P2-B2** — two of the 19 units (`C-graph-dba`, `C-architect`) assign a self-edit that the
  agents' own operative prompts forbid in so many words, and that `architect`'s write guard would
  escalate to a human on every write. V2 rests `C-graph-dba` on a `ccf9c8b` precedent that does not
  exist (the commit never touched `claude/graph-dba/graph-dba.md`).

Four majors and six minors follow. The plan's core design — one shared `kaizen_team`,
`author`-partitioned — I have no quarrel with; every finding below is about executability and
coverage, not the design.

---

### Re-verification of V2's re-triage of Pass 1 (what the brief asked me to check)

| Pass-1 finding | V2's claim | My independent result |
|---|---|---|
| **B1** — `teco`/`tico`/`data-scientist` lack `mcp__cypher__query` | closed by `ccf9c8b` | **Confirmed closed.** `claude/teco/teco.md:4`, `claude/tico/tico.md:4`, `claude/data-scientist/data-scientist.md:4` each now end their `tools:` line with `mcp__cypher__query`. |
| **B2** — AC-11 unowned | moot, AC dropped | **Confirmed, and dropped cleanly.** No dangling reference survives in the plan: §1's Out-of-scope states the reversal explicitly, §5.1's FR-14 row and §5.2's AC-11 row both carry a stated reason (not silent absence), and `Q2`'s done-condition requires confirming AC-3/AC-11 are *dropped*, not absent. Good practice. (One dangling reference does survive **outside** the plan — see P2-M3.) |
| **FR-4/AC-3 must drop too** (V2's own added finding) | blanket decision, not graph-dba-specific | **Reasoning holds.** Requirements FR-4 mandates deleting the eleven agents' `inbox.md`; the decision's wording is *"No `kaizen/inbox.md` file is ever deleted, for any agent"*. FR-4/AC-3 is the same requirement one clause earlier and dies the same death. **But V2 did not catch them all** — FR-12/AC-9 is the third collision. See P2-B1. |
| **M1** — `audit-team.sh` check 1 requires `kaizen/inbox.md` | moot by construction | **Half confirmed, half silently dropped.** Ran it live: all 12 agents `PASS  <agent>: kaizen plan + history + inbox present`; the only two `FAIL`s are the pre-existing home-path/username leaks in `falkor-chat/docs/test-reports/graphrag-eval-report.md`. The *deletion* half of M1 is genuinely moot. The *FR-12* half — Pass 1's own sentence, "the same unconditional check would also fail for FR-12's own goal (a newly created agent, born graph-backed, never gets a `kaizen/inbox.md` at all)" — is untouched by the never-delete decision and V2 does not address it. See P2-B1. |
| **M2** — hook scripts hardcode `kaizen/inbox.md` | cosmetic only | **Substantially confirmed**, with one correction — see P2-m6. The `Write`-resurrection failure mode Pass 1 flagged is genuinely gone (nothing is deleted). |
| **M3** — `teco.md`'s coordination prose | still a real defect, fixed in `C-teco` | **Confirmed on both halves.** `claude/teco/teco.md:72` and `:89` are verbatim as V2 quotes them, and both are wrong under the graph mechanism regardless of file persistence. `C-teco`'s done-condition specifies the fix concretely ("both stale cross-reference passages now describe a `kaizen_team`/`author`-filtered check, not a file check") — this is a real fix, properly specified, not an acknowledgement. |
| **M4** — `cobb` self-editing `cobb.md` | resolved by `ccf9c8b` precedent | **Confirmed for `cobb` only.** `git show --stat ccf9c8b` lists `claude/cobb/cobb.md | 15 ++-`. It does **not** list `claude/graph-dba/graph-dba.md` at all. V2 extends the precedent to `graph-dba.md` "alike" (§3.6, §4.2) — that extension is unsupported. See P2-B2. |

---

### Findings

#### Blockers

**P2-B1 — The plan asserts FR-12/AC-9 is already delivered. It is not, and the AC-9 check the plan
prescribes cannot detect the gap. FR-12/AC-9 is the third requirement the never-delete decision
collides with, and the plan neither drops it nor delivers it.**

Requirements FR-12: *"a newly created agent is born directly on the graph-backed pattern — **no
`kaizen/inbox.md` is ever created for it**"*. AC-9: *"...with **no `kaizen/inbox.md` step in it**"*.

The plan says (§5.1's FR-12 row): *"Already delivered by `ccf9c8b`'s SKILL.md rewrite, wrong graph
name in the seeded text"*, and `S3`'s done-condition states: *"§1's 'Creating' procedure
(new-agent convention, FR-12) **confirmed to already describe seeding the Learning-capture section
directly (no `inbox.md` step)** — only the graph name in that seeded text needs the `kaizen_team`
update"*.

That is false, verified by direct read. `skills/agent-maintenance/SKILL.md:57-61`, §1's Creating
procedure, step 1:

> 1. **Creating:** create both files. Seed `history.md` with a dated "created" entry and `plan.md`
>    with improvements you already foresee. In collections that run the learning-capture loop (§5
>    — graphmind-ai-lab's `claude/` does), **also seed an `inbox.md` from the §5 template's
>    frozen-stub variant** — the agent's own `kaizen_<name>` graph needs no pre-creation…

and `skills/agent-maintenance/SKILL.md:433-435` supplies the template to seed with:

> **Inbox template** (**seed on creation**, for the frozen `kaizen/inbox.md` triad member every
> agent still carries…)

whose body says the file *"exists only to satisfy the standard kaizen triad (`audit-team.sh` check
1)"*.

So a new agent created under today's convention **does** get a `kaizen/inbox.md` — FR-12 and AC-9
are unmet. And the plan's own AC-9 verification (§5.2) is *"Read `skills/agent-maintenance/SKILL.md`
§1 post-S3: confirm the seeded new-agent text targets `kaizen_team`"* — it checks only the graph
name, never the "no `inbox.md` step" half, so `Q2` would sign AC-9 off green with the requirement
unmet.

This is not a simple "add a step" fix, because there is a genuine three-way conflict the plan must
resolve explicitly:

1. FR-12/AC-9 says a new agent gets no `inbox.md`.
2. `audit-team.sh:76` unconditionally requires the file for every agent (ran it live — this is why
   the stub exists at all).
3. The stakeholder's never-delete decision has just re-established `inbox.md` as a *permanent
   historical record*, which is a coherent argument that a **brand-new** agent — with no history to
   preserve — should not get one at all, and equally a coherent argument that the triad is now a
   structural invariant.

Suggested fix — pick one, in the plan, explicitly:

- **(a) Amend FR-12/AC-9** the same way §5.1/§5.2 amend FR-4/FR-14: record that the never-delete
  decision makes the frozen-stub triad member a standing structural invariant, so a new agent *does*
  get a seeded empty `inbox.md`, and the FR-12 requirement that survives is only "born graph-backed,
  no retrofit". Cheapest, matches shipped reality, needs a one-line stakeholder confirmation via
  `tico` (see P2-M3). Then `S3`'s done-condition should say "retarget the seeded stub's graph name",
  and drop the false "no `inbox.md` step" assertion; and AC-9's check becomes "seeded text targets
  `kaizen_team`" only, correctly scoped.
- **(b) Deliver FR-12 as written** — remove the seed step from `SKILL.md` §1 and the Inbox template
  from §5, **and** add a unit updating `claude/scripts/audit-team.sh:76` so check 1 accepts
  plan+history without inbox (Pass 1's M1, the half V2 declared moot). This is the option that keeps
  the requirement intact; it costs one extra unit and touches the certification script.

Either way, AC-9's verification approach in §5.2 needs a second clause that actually tests the
"no `inbox.md` step" property (or explicitly records it as dropped).

**P2-B2 — `C-graph-dba` and `C-architect` assign each agent a self-edit of its own operative prompt.
Every non-`cobb` agent's prompt — `architect.md` and `graph-dba.md` included — ends its
Learning-capture section with the literal standing instruction "never edit your own agent
definition", and `architect`'s `PreToolUse` write guard would escalate every such write to a human.
The `ccf9c8b` precedent V2 cites covers `cobb` only.**

Three independent pieces of evidence, all directly verified:

1. **The prohibition is in the prompts themselves.** `claude/architect/architect.md:67` and
   `claude/graph-dba/graph-dba.md:87` both end with: *"…the team maintainer (`cobb`) reads it,
   verifies, and promotes entries; **never edit your own agent definition**."* `grep -rln "never
   edit your own agent definition" claude/` returns all 11 non-`cobb` agent prompts (plus kaizen
   histories) — and **not** `claude/cobb/cobb.md`. `cobb` is the single agent whose prompt omits the
   clause, because `cobb` is the maintainer. That is the actual shape of the carve-out, and it does
   not generalize.
2. **`architect` is harness-blocked.** `claude/architect/hooks/guard-plan-doc-writes.sh:10` allows
   only `docs/plans/*|*/docs/plans/*|architect/kaizen/inbox.md|*/architect/kaizen/inbox.md`.
   `claude/architect/architect.md` matches none of those, so a self-edit escalates to a human
   approval prompt — the unit does not execute cleanly as written.
3. **The claimed precedent is absent for `graph-dba`.** `git show --stat ccf9c8b` lists 43 files;
   `claude/graph-dba/graph-dba.md` is not among them (it was already graph-backed and needed no
   edit). So §3.6's *"M4's resolution … applies this to `cobb.md` and `graph-dba.md` alike"* and
   §4.2's `C-graph-dba` done-condition ("prompt self-edited to `kaizen_team`") rest on a precedent
   that never happened.

Note that the `C-architect` half is *flagged* by the plan (§6 open item 1) and the `C-graph-dba`
half is **not** — it is asserted as settled. That is the part that makes this a blocker rather than
an answered open question.

Suggested fix: reassign the prompt-edit half of **both** `C-architect` and `C-graph-dba` to `cobb`,
identical to the other nine. `cobb` then edits 11 of 12 prompt files and self-edits only `cobb.md`
— which is exactly the shape `ccf9c8b` already shipped and the shape `cobb.md`'s own missing
prohibition sanctions. This also answers §6 open item 1 definitively: **no**, `architect` should not
self-edit; the `cobb`/`graph-dba` symmetry V2 reached for does not exist.

#### Major

**P2-M1 — AC-8's verification command, as literally written in §5.2, cannot see the occurrences this
delivery exists to fix. Ran both patterns; the plan's own command misses 10 of the 12 agent prompt
files.** §5.2's AC-8 row prescribes:

```
grep -rln 'kaizen_<agent>\|kaizen_graph_dba' claude/ skills/ cypher-mcp/
```

`kaizen_<agent>` is a *template placeholder*. The repo does not contain that string in any agent
prompt — it contains `kaizen_analyst`, `kaizen_coder`, `kaizen_teco`, … . Run verbatim, the command
returns 13 files, of which only two are agent prompts (`graph-dba.md`, `cobb.md`, both hit via the
literal `kaizen_graph_dba`). It **misses** `analyst.md`, `architect.md`, `coder.md`,
`data-scientist.md`, `devops.md`, `frontend-engineer.md`, `qa-engineer.md`, `tdd-engineer.md`,
`teco.md`, `tico.md` — i.e. the majority of what every `C-<agent>` unit is supposed to fix. AC-8 is
the plan's only static, repeatable sweep; as written it would report the rollout complete with ten
prompts still pointing at retired graph keys.

Suggested fix — replace with a pattern that matches an actual graph key (this one was run before
writing it here; it returns all 12 agent prompts plus 51 other files):

```
grep -rlE 'kaizen_[A-Za-z{<][A-Za-z_{}<>-]*' claude/ skills/ cypher-mcp/ docs/ AGENTS.md
```

The `{`/`<` alternatives matter — `SKILL.md` uses both `kaizen_<name>` and `kaizen_{name}` spellings
(see P2-M4). Widening the search roots to `docs/` and root `AGENTS.md` also matters: `S2` already
owns root `AGENTS.md`, and it is outside the three roots the current command searches.

**P2-M2 — The 12 frozen `kaizen/inbox.md` header notes each name `kaizen_<agent>` prescriptively and
hand out a copy-pasteable query against it; no unit touches them, and the never-delete decision makes
those pointers permanent.** Read `claude/analyst/kaizen/inbox.md:3-9` (all 12 carry the same block,
added by `ccf9c8b`):

> **FROZEN — 2026-08-20.** … Its 5 entries … were imported into the `kaizen_analyst` FalkorDB graph
> …; `analyst` no longer appends here. **New raw learnings are written directly into the graph and
> are immediately queryable by any agent:** `mcp__cypher__query(graph='kaizen_analyst', cypher='MATCH
> (e:KaizenEntry) …')`.

That text is **prescriptive about current behaviour**, not a past-tense historical record — so it
falls squarely inside FR-11/AC-8 ("no doc … contradicting that agent's actual behavior"), and by
AC-8's own classification rule in §5.2 it is "a real remaining gap", not "genuinely historical". After
`C-<agent>` retargets and `G1` deletes the key, all 12 files will instruct any reader to query a
graph that no longer exists — permanently, because these files are now never deleted. §1's scope
list and §4's `C-<agent>` Files columns (which name only `<agent>.md` at specific line numbers) do
not reach them.

This is the identical failure shape Pass 1's M3 named and V2 fixed for exactly one file (`teco.md`).
Suggested fix: add `claude/<agent>/kaizen/inbox.md`'s header note to each `C-<agent>` unit's Files
column and done-condition — the retarget is mechanical and lands with the same actor (`cobb`) who
edits the prompt. While there, fold in the wording S2 already prescribes for `claude/README.md`
("permanent frozen snapshot", not "required to exist").

**P2-M3 — Four requirements (FR-4, FR-14, AC-3, AC-11) are being dropped and a fifth (FR-12/AC-9)
needs amending, but no unit routes any of that back to `tico`, who owns
`docs/requirements/generic-cypher-mcp2.md`. The requirements doc will state them as binding
forever.** The plan drops the ACs in its own §5, which is correct as far as the plan's authority
reaches — `architect` cannot edit a `tico`-owned requirements document. But nothing in §4's 19-unit
table names the requirements doc, and FR-11/AC-8's own scope is explicitly `claude/AGENTS.md`,
`claude/README.md`, `docs/BACKLOG.md`, and agent prompts — requirements docs are outside it. Two
further live stale statements make this concrete:

- `docs/requirements/generic-cypher-mcp2.md` FR-4, FR-14, AC-3, AC-11 all still read as binding, and
  its Out-of-scope bullet still argues *"which is exactly why an in-repo frozen copy is no longer
  needed"* — a rationale the stakeholder has now reversed.
- `docs/plans/generic-cypher-mcp2-coordination.md`'s own "Goal & definition of done" (written before
  the decision) still says *"then deletion of that file once the import is confirmed"* and
  *"`graph-dba`'s own already-frozen `kaizen/inbox.md` … is also deleted as part of this delivery"* —
  directly contradicting the decision recorded further down the same file. That one is `teco`'s to
  fix, not `architect`'s, but it belongs on the same list.

Suggested fix: add a unit **`T1` — owner `tico`** — amending `docs/requirements/generic-cypher-mcp2.md`
per the repo's own second-document rule (this document has been executed against, so: in-place
revision note vs. successor is `tico`'s call, per root `AGENTS.md` collision rule 5). Its scope:
mark FR-4/AC-3/FR-14/AC-11 superseded with the decision and date, void the git-history Out-of-scope
rationale, and settle FR-12/AC-9 per P2-B1's chosen branch. Sequence it early — P2-B1's branch (a)
needs the stakeholder confirmation it produces.

**P2-M4 — `S3`'s occurrence list, explicitly advertised as "full grep-confirmed … not a re-scoped
guess", is itself incomplete — and the two lines it misses are precisely the FR-12-critical seeded
template.** `grep -c "kaizen_" skills/agent-maintenance/SKILL.md` → **14** matching lines: 3, 61,
201, 324, 327, 328, 347, 391, 417, 426, 436, **444**, **445**, 460. `S3` enumerates twelve of them
("10 occurrences", itself an undercount of its own list) and omits 444–445 — which are inside the
§5 "Inbox template" block seeded into every new agent:

```
> `kaizen_{name}`, as `:KaizenEntry` nodes (agent-maintenance skill §5),
> immediately queryable by any agent: `mcp__cypher__query(graph='kaizen_{name}',
```

They were missed because they use the curly-brace spelling `kaizen_{name}`, while the enumeration was
built from the angle-bracket forms. This is Pass 1's m2 re-asserted as fixed but not fixed, and it
matters more than a line-count nit because these are the exact lines FR-12/AC-9 turns on.

Suggested fix: replace `S3`'s hardcoded line list with the grep itself as the done-condition ("no
`kaizen_` occurrence other than a past-tense Origin note survives, verified by
`grep -n 'kaizen_' skills/agent-maintenance/SKILL.md`") — a line-number list in a plan goes stale
the first time anyone edits the file, and this one was already wrong on the day it was written.

#### Minor

**P2-m1 — `G1`'s gate is a count check against data it then irreversibly destroys.** §4.2 step 2's
verification is `MATCH (e:KaizenEntry {author:'<X>'}) RETURN count(e)` matched against the source
count; `G1` then `GRAPH.DELETE`s the source key. A count match cannot detect field-level corruption,
and the migration query is a **hand-built literal** — each entry's `fact`/`evidence`/`context` is
re-emitted as a quoted Cypher string, and the live entries genuinely contain backticks and
apostrophes (e.g. `kaizen_analyst`'s "No `SendMessage` tool means…"). A quote-escaping slip
truncates a field while the count still reconciles. (`cobb`'s own `ccf9c8b` commit message notes it
extracted *programmatically, not hand-transcribed,* "to avoid silent drops" — the same care is
warranted here and the plan doesn't ask for it.)

Mitigating: each agent's `kaizen/inbox.md` still holds the original markdown, so a third copy
survives even after `G1`. Suggested fix: make the `C-<agent>` verification a content comparison
(`RETURN e.entryId, e.fact ORDER BY e.entryId` on both sides, diffed) rather than a count, and say
so in the done-condition; and note in `G1` that its gate is the content check, not the count.
This is also the substantive answer to §6 open item 3 (`G1` batching): **batching is not the risk** —
one unit is fine — the risk is that `G1` is the delivery's only irreversible step and its gate is
weaker than the thing it protects.

**P2-m2 — The plan's live snapshot was already stale on the day it was written: `kaizen_architect`
now exists.** Live-queried the loaded-graph list: `kaizen_graph_dba, kaizen_analyst,
kaizen_data-scientist, kaizen_qa-engineer, kaizen_teco, **kaizen_architect**`. `kaizen_architect`
holds 1 entry, `author: 'architect'`, `date: '2026-08-20'` — written by `architect` itself, at or
just after the very revision whose §2 states *"the other 7 agents' `kaizen_<agent>` graph keys
(`architect`, …) **do not exist**"*. So `G1`'s "5 currently-existing keys" is now 6, and
`C-architect`'s "0 (graph key doesn't exist)" is wrong.

This is **not** a design defect — it is the plan's own FR-13 incremental-window scenario firing
exactly as predicted, and both `C-architect` and `G1` already carry live-recheck instructions that
handle it correctly. Credit where due. But the enumerated counts in §1, §2, §4.2 and `G1` should be
demoted from "the state" to "the state as of 2026-08-20, re-derive at dispatch", and `G1`'s
done-condition should lead with the re-list, not treat it as an "also".

**P2-m3 — `S1`'s file/line enumeration under-counts both files.** `cypher-mcp/server.py` contains
`kaizen_graph_dba` at lines **118, 134, 251, 763** — `S1`'s Files column names only "lines 116–144",
missing the two code comments at 251 (`_looks_like_write`'s docstring: *"a populated
`kaizen_graph_dba` never returns 'empty key' again after the one-time import"*) and 763 (*"materializing
`kaizen_graph_dba` for the one-time import"*). Both become wrong once `G1` deletes that key.
`cypher-mcp/README.md` contains **5** occurrences, not the "3 mentions" `S1` states. `S1`'s
done-condition ("every `kaizen_graph_dba` mention replaced") is right and saves it — but an
implementer working the Files column will ship a short diff. Same fix as P2-M4: state the grep, not
the line numbers. (`cypher-mcp/tests/test_server.py`'s 15 occurrences are arbitrary fixture graph
names and correctly out of scope — verified they assert nothing about the instruction text.)

**P2-m4 — `S0` is explicitly *not* a hard blocker on the `C-<agent>` units, but its uniqueness
constraint is the only thing preventing a re-run of a `C-<agent>` unit from silently duplicating
entries** (§4.2 uses `CREATE`, not `MERGE`). A retry after a partial/ambiguous failure — the ordinary
case that motivates FR-13's incremental design — would double-write. Suggested fix: make `S0` a hard
predecessor of every `C-<agent>` unit. It costs one serialization edge, removes the duplicate class
entirely, and — usefully — makes §6 open item 2 (FalkorDB constraint idempotency when re-issued
against an already-provisioned graph) **moot for this delivery**, since `S0` then always runs against
an empty, never-before-written `kaizen_team`. That is a better resolution than deferring the
unverified-behaviour question to implementation time.

**P2-m5 — `S2` adds a `## M7` section to `docs/BACKLOG.md` but not a row to its `## Milestone map`
table.** Read `docs/BACKLOG.md:41-53`: every one of M1–M6 has both a map row and a body section.
One-line addition; name it in `S2`'s done-condition so it isn't left to the implementer to notice.

**P2-m6 — One correction to V2's M2 re-triage: the staleness is not confined to the header
comments.** V2 states the escalation text "stays accurate". Read all five wrappers; the escalation
strings describe the frozen file as a live write target — e.g.
`claude/analyst/hooks/guard-review-doc-writes.sh:12`: *"its `Write`/`Edit` are for review documents
**and its learnings inbox** only"*, and `claude/tico/hooks/guard-tico-doc-writes.sh:12`: *"**its
learnings inbox is the one other allowed target**"*. And the glob at each script's line 10-11 still
*permits*, without escalation, a `Write` to a file the convention now declares frozen and never
written to — a weaker version of the Pass-1 concern, but not zero. Agreed this is low-priority and
doesn't warrant its own dispatch; §6 open item 4's framing should just say "header comments **and**
escalation text", and note the glob permits appends to a frozen file.

#### Nit

**P2-n1 — `G1` doesn't name the execution surface for `GRAPH.DELETE`.** §3.6 correctly establishes
that it is *not* one of `mcp__cypher__query`'s two authorized write shapes — but then `G1`'s
done-condition just says "`GRAPH.DELETE <key>` (own destructive-ops hook approval)" without saying
the call goes via `redis-cli` against the FalkorDB container. `graph-dba` will know; a one-clause
addition removes the ambiguity for a cold dispatch.

---

### What's solid

- **§3.6's `authorize_write()` author-binding claim is correct, and I confirmed it by execution, not
  by reading.** Loaded `cypher-mcp/server.py` in `cypher-mcp/.venv` and ran the plan's own §4.2
  migration shape through `authorize_write()`:
  - `authorize_write(<UNWIND … CREATE (k:KaizenEntry {…, author: 'analyst', …})>, 'analyst')` → `None`
    (authorized).
  - the same query with `agent='cobb'` → *"Rejected: this write attributes an entry to author
    'analyst', but the call declared agent='cobb'."*
  - `_author_claims()` correctly extracts exactly `['analyst']` — the `UNWIND` list's own map
    literals produce no spans, so the shape is safe; and a decoy `CREATE (k:KaizenEntry {author:
    'cobb'})` embedded inside an entry's `fact` free text does **not** desync it (also executed).

  So the constraint is real: **`cobb` cannot do the data-migration half for anyone**, and the
  two-actor `C-<agent>` unit shape is forced by the mechanism, not chosen. This was the single most
  load-bearing new claim in V2 and it holds exactly as stated.
- **`G1` is correctly identified as destructive and correctly routed.** Verified `GRAPH.DELETE` is
  outside the tool's two write shapes (it is a Redis command, not Cypher — the tool only issues
  `GRAPH.QUERY`/`GRAPH.RO_QUERY`), and that `guard-destructive-ops.sh` exists for exactly
  `graph-dba`, `devops`, and `qa-engineer` (`ls claude/*/hooks/`). `graph-dba` is the right owner and
  no migrating agent could run it itself.
- **§5's FR/AC mapping is structurally complete.** All 15 FR rows (FR-1…FR-14 plus FR-8a) and all 13
  AC rows are present; both dropped ACs carry a stated reason rather than vanishing; and `Q2`'s
  done-condition requires confirming the *absence* of a deletion rather than treating AC-3/AC-11 as
  simply out of mind. That is the right way to drop an AC. (The one gap is FR-12/AC-9 — P2-B1 — which
  is a wrong status, not a missing row.)
- **Every agent-prompt line number in §2 and §4.2 is accurate.** Spot-checked all 12 files with
  `grep -n "kaizen_"`: `analyst.md:91,102`, `architect.md:56,67`, `cobb.md:71,86,97`,
  `coder.md:40,51`, `data-scientist.md:91,102`, `devops.md:100,111`,
  `frontend-engineer.md:86,97`, `graph-dba.md:87`, `qa-engineer.md:80,91`, `tdd-engineer.md:58,69`,
  `teco.md:122,133` (+`:72,89`), `tico.md:151,162`. No drift.
- **The FR-4/AC-3 extension is `architect`'s own finding and it is right.** The brief named only
  FR-14/AC-11; reading the decision's wording and the requirements text independently, FR-4/AC-3 is
  the same requirement for the other eleven agents and dies with it. Catching that unprompted is
  exactly what a revision pass should do.
- **The plan is honest about what it did and didn't re-derive** — §2 explicitly states which V1
  findings were re-read versus carried forward on the strength of `ccf9c8b` not touching the file, and
  §4.2/`G1` build live-rechecks into the units rather than trusting the plan's own snapshot. P2-m2 is
  proof that discipline was warranted, and that it works.
- **`audit-team.sh` is genuinely clean under this plan**: ran live, 12/12 check-1 `PASS`, two
  pre-existing unrelated `FAIL`s. M1's deletion half really is moot by construction.

### Answers to §6's open items

1. **`C-architect` self-edit** → **No.** Route `architect.md`'s edit to `cobb`, and `graph-dba.md`'s
   too. Evidence in P2-B2: the prohibition is written into both prompts, `architect`'s write guard
   blocks it, and `ccf9c8b` set no `graph-dba` precedent. `cobb.md` is the sole legitimate self-edit,
   because `cobb.md` is the sole prompt without the prohibition clause.
2. **`S0` constraint idempotency** → make `S0` a hard predecessor of all 12 `C-<agent>` units
   (P2-m4). The unverified-behaviour question then never arises for this delivery, and the duplicate-
   on-retry hole closes with it. If `graph-dba` still wants the fact for the knowledge base, that's a
   `falkordb-quirks.md` entry, not a gate on this plan.
3. **`G1` batching** → keep it as one unit; the sizing is fine. The framing needs changing, not the
   batching: `G1` is this delivery's only irreversible step and it is gated on a count check
   (P2-m1). Strengthen the gate to a content comparison, and lead its done-condition with the live
   re-list rather than the snapshot's five keys (P2-m2).
4. **M2 residual** → agreed it's low priority; widen the description to cover the escalation text and
   the permissive glob, not just the header comments (P2-m6).
5. **`cobb` concentration** → concur with the plan; a real throughput note for `teco`, not a defect.
   P2-B2's fix adds two more prompt files to `cobb`'s load (11 of 12), which sharpens it slightly.
6. **"13 agents" vs 12** → confirmed 12 (`ls claude/`, and `ccf9c8b`'s file list). Correctly flagged
   and correctly immaterial.

### Open questions

- **Which branch of P2-B1?** (a) amend FR-12/AC-9 to accept the permanent frozen stub as a structural
  invariant, or (b) deliver FR-12 as written and update `audit-team.sh:76`. This is a stakeholder
  question, not an architect one — it re-opens a requirement the stakeholder locked. Route via `tico`
  alongside P2-M3's `T1` unit.
- **Does the never-delete standing rule extend to the `kaizen_<agent>` graph keys themselves?** The
  stakeholder's stated reason for keeping every `inbox.md` is that a historical record should not be
  destroyed. `G1` destroys five (now six) graphs holding the same records. The plan treats this as
  obviously in scope; it may be, since the entries are relocated rather than discarded and the
  `inbox.md` originals survive — but the plan never asks, and it is the one place this delivery's
  design runs against the grain of the decision that reshaped it. Worth one explicit confirmation
  before `G1` is dispatched.
- **Who flips `docs/BACKLOG.md`'s M7 item markers** as each unit lands — carried over unanswered from
  Pass 1's open questions; `S2` creates the section, and nothing says who maintains it. `teco`'s call.

---

## Pass 3 — Re-gate of Version 3 (2026-08-20)

### Scope & verdict

Plan-gate re-review of `docs/plans/generic-cypher-mcp2.md` **Version 3** (Status: active, owner
`architect`), read in full — not as a diff against V2 — against
`docs/requirements/generic-cypher-mcp2.md` and `docs/plans/generic-cypher-mcp2-coordination.md`
(including its "Resolved 2026-08-20" paragraph, read at lines 126-132).

What I ran and read this pass, beyond re-reading the plan:

- Re-read `claude/scripts/audit-team.sh` end to end — the header comment (lines 8-13), the agent
  enumeration (lines 63-67), and check 1 (lines 74-79).
- **Executed `audit-team.sh` against a synthetic agent** in an isolated scratch tree (a relocated
  copy, so the live `claude/` collection was not touched), in both readings of `S4`'s
  done-condition (b). Result below — it does not hold.
- Re-ran `grep -n "kaizen_" skills/agent-maintenance/SKILL.md` to check `S3`'s rebuilt 14-line list
  independently of its stated count.
- Ran V3's new AC-8 pattern verbatim and measured what it actually returns.
- Read all 12 `claude/*/kaizen/inbox.md` header notes, comparing the populated agents'
  (`analyst`, `teco`, `qa-engineer`, `data-scientist`) against an empty one (`coder`).
- Grepped V3 itself for the cross-references its §3 compression leaves behind.

**CPG:** not applicable — agent prompts, FalkorDB kaizen schema, and documentation only; no
Joern-modelled application code in scope (unchanged from Pass 2, and per the dispatch brief).

**Verdict: approve with suggestions.**

Both Pass-2 blockers are genuinely closed, and I confirmed the mechanism of each rather than the
claim. All four majors are addressed, three of them cleanly. Three new majors below are all
**one-or-two-sentence done-condition edits** — none touches the design, the unit set, the
ownership model, or any FR/AC mapping. My recommendation to `teco`: have `architect` apply them
in place (the plan is `active` and has not yet been executed against, so a `Version:` bump plus a
dated note is the right mechanism) and **dispatch without a fourth gate**. A Pass 4 would cost more
than it can find.

---

### Verification of the fixes (what the brief asked me to check)

**1. P2-B1 — `S3` + `S4` do close the gap. The units are right; `S4`'s stated verification is not.**

The substance works. `S3` (`skills/agent-maintenance/SKILL.md`) removes the seeding step from §1's
Creating procedure and rewrites §5's "Inbox template" from a "seed on creation" block into a
historical reference; `S4` narrows `claude/scripts/audit-team.sh` check 1 to `plan.md` + `history.md`.
Traced against the real script: check 1 at `audit-team.sh:76` is a plain three-way `-f` conjunction,
so dropping the third conjunct makes plan+history sufficient — a brand-new agent with no `inbox.md`
passes, and the 12 existing agents keep passing because their frozen file is still physically
present. Neither unit touches any `kaizen/inbox.md`. The asymmetry the directive asks for
("existing ones are permanent, new ones are never created") genuinely holds under these two units.
`S4` also correctly catches that the script's **header comment** (lines 8-13, *"has its
kaizen/{plan,history,inbox}.md triple"*) must change alongside the executable logic — that comment
is the documented contract and V3 spotted it without being told.

**But `S4`'s done-condition (b) is not achievable as written**, and `Q2` inherits it — see P3-M1.

**2. P2-B2 — reflected in the step table's owner field, both rows.** Confirmed in §4.2's table, not
only in prose: `C-graph-dba`'s Agent cell reads *"`graph-dba` (data) / `cobb` (prompt)"* and
`C-architect`'s reads *"`architect` (data) / `cobb` (prompt)"*, each with an explicit
done-condition clause (*"**`cobb`**, not `graph-dba`, retargets the prompt (§3.7 — corrected from
V2)"*). §3.7 is a new, correctly-reasoned section, and V3 re-derived both evidence lines itself
rather than citing my review — it re-ran the `grep` and `git show --stat ccf9c8b`. `C-cobb` remains
the single self-edit, correctly justified. This is fully fixed.

**3. The four majors.**

- **P2-M1 (AC-8 pattern) — fixed.** §5.2's AC-8 row now carries the corrected pattern verbatim,
  widened to `docs/` and root `AGENTS.md`. Ran it: it returns 64 files including all 12 agent
  prompts (versus 13 files / 2 prompts for the old placeholder version). The pattern is right; the
  widened roots introduce a small classification problem — P3-m2.
- **P2-M2 (12 frozen `inbox.md` headers) — folded in, and the "frozen ≠ read-only for an accuracy
  edit" carve-out is sound.** The brief asked me to judge the carve-out specifically. It holds, on
  two independent grounds, and the second is stronger than the plan's own argument:
  1. Root `AGENTS.md`'s archived-document convention permits exactly one class of edit to a frozen
     document — *"A header pointer is metadata, not an amendment — it is the one edit permitted on
     an `archived` document."* A graph-name pointer in a header note is precisely that class.
     (Strictly, that convention governs `docs/<kind>/` documents carrying a `Status:` token, which
     these agent working files are not — so it is an analogy, not a rule that literally binds here.)
  2. **The frozen files themselves already scope their own immutability promise to exclude the
     header.** Every one of the 12 ends its header note with: *"**Content below** is preserved for
     historical reference and will not change."* Not "this file"; *content below*. The carve-out
     isn't a new exception the plan is carving — it is what the files have promised since
     `ccf9c8b`. V3 should cite this sentence in §4.2; it settles the question outright and is
     stronger than the "accuracy edit, not new content" framing it currently leans on.

  So: no contradiction with the plan's own use of "frozen", and none with root `AGENTS.md`. What
  *is* wrong is the scope of the retarget instruction — see P3-M3.
- **P2-M3 (`T1`) — fixed and well-scoped.** New unit, `tico`-owned, files
  `docs/requirements/generic-cypher-mcp2.md`, covering FR-4/AC-3/FR-14/AC-11 plus the reversed
  git-history rationale in Out of scope. Two things V3 got right that I would have flagged
  otherwise: it correctly concludes **FR-12/AC-9 need no requirements edit** (branch (b) delivers
  them as originally written rather than superseding them — which is also why choosing branch (b)
  on `teco`'s directive required no fresh stakeholder ruling, while branch (a) would have), and it
  declines to preempt `tico`'s in-place-vs-successor choice under root `AGENTS.md`'s collision rule
  5, citing the requirements doc's own 2026-08-19 "Reconsidered" entries as precedent. Correctly
  routed and correctly bounded. The coordination doc's own stale section is properly excluded as
  outside `architect`'s authority (and is fixed already, per the brief).
- **P2-M4 (`S3`'s occurrence list) — fixed; I re-ran the grep rather than trusting the count.**
  `grep -n "kaizen_" skills/agent-maintenance/SKILL.md` returns exactly **14** lines: `3, 61, 201,
  324, 327, 328, 347, 391, 417, 426, 436, 444, 445, 460` — identical to V3's list, including the
  previously-missed 444/445. Better still, `S3`'s done-condition now makes *the grep itself* the
  verification and demotes the line list to a dated snapshot, and `S1` got the same treatment. That
  is the durable fix, not just the corrected numbers.

**4. `G1` / the never-delete reach — settled, and I treated it as such.** Read the "Resolved
2026-08-20" paragraph at `docs/plans/generic-cypher-mcp2-coordination.md:126-132`: the stakeholder's
answer is yes, delete the keys, and the record explicitly discharges `G1`'s confirmation gate
(*"`graph-dba` does not need to re-ask via `tico` at execution time; cite this line instead"*). I am
**not** re-opening it. `G1` is otherwise well-designed — content-diff gate, `graph-dba`-owned,
live-relist-first, `redis-cli` execution surface named. Only the wording needs a pass (P3-m1).

---

### New findings

#### Major

**P3-M1 — `S4`'s done-condition (b) is unachievable as written, and `Q2` re-runs it as an
acceptance step. Proved by execution, not by reading.** `S4` requires:

> (b) a synthetic scratch agent directory containing only `plan.md`+`history.md` (no `inbox.md`)
> also `PASS`es, proving the FR-12 asymmetry holds for a genuinely new agent.

I copied `audit-team.sh` into an isolated scratch tree (`<scratch>/root/sub/scripts/`, so
`ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"` resolves to `<scratch>/root` and the
live repo is untouched) and ran both readings:

- **Without a `<name>.md`** — the directory is **never enumerated at all**. `audit-team.sh:63-67`
  builds the roster as `for d in "$CL"/*/ … [ -f "$d$name.md" ] && agents+=("$name")`. Output:
  `FAIL  no agents found under …`. It cannot "PASS" check 1, because it is never checked.
- **With a stub `newagent.md` added** so it *is* enumerated, and with check 1 still in its current
  unmodified form:

  ```
  Auditing 1 agents: newagent
  FAIL  newagent: missing kaizen/plan.md, kaizen/history.md, or kaizen/inbox.md
  FAIL  newagent: not symlinked into …/.claude/agents (or points elsewhere)
  FAIL  newagent: NOT mentioned in teco's prompt — roster drift
  FAIL  newagent: missing from claude/AGENTS.md
  FAIL  newagent: missing from claude/README.md
  ```

  After `S4`'s edit the first line flips to `PASS`, but checks 2, 4, 5 and 5b still `FAIL` — a
  synthetic agent is by construction undeployed, unrostered and uncataloged. **The run's overall
  result is `FAIL` either way**, so an implementer taking the done-condition literally cannot
  satisfy it, and `Q2` ("`S3`/`S4`'s synthetic-agent check re-run independently … `S4`'s
  audit-team.sh asymmetry re-verified live one more time at closing") would stall on the same
  impossibility.

There is a third hazard the wording invites: doing this test inside the live `claude/` tree creates
a real directory in the agent collection, which check 7's untracked-file personal-info scan sweeps
and which mutates the certified roster.

Suggested fix — restate (b) as a check-1-scoped assertion plus an explicit location:

> (b) In an **isolated copy of the repo tree** (not `claude/` itself — `ROOT` resolves as
> `dirname($BASH_SOURCE)/../..`, so a copy at `<scratch>/sub/scripts/audit-team.sh` audits
> `<scratch>/claude/`), a scratch agent with `<name>.md` + `kaizen/{plan,history}.md` and **no**
> `inbox.md` produces `PASS  <name>: kaizen plan + history present` on **check 1**. The overall run
> still `FAIL`s on checks 2/4/5 (deployment, roster, catalogs) — expected and irrelevant; check 1's
> line is the assertion.

Mirror the same scoping into `Q2`'s done-condition.

**P3-M2 — V3's §3 compression removed two copy-pasteable Cypher artifacts that other units still
cite by reference, and left one factually wrong pointer at the document's own contents.** §3's new
preamble says §3.1–§3.5 are *"Not reproduced a third time in full here — see Version 2's own text,
preserved in git history, or the live document below §3.6."* Verified consequences:

1. **The FR-7 team-wide query is gone.** V2's §3.1 carried it in full
   (`MATCH (e:KaizenEntry) RETURN e.author, e.date, e.fact, e.evidence, e.context, e.suggestedHome
   ORDER BY e.date`). `grep -n "MATCH (e:KaizenEntry) RETURN e.author" docs/plans/generic-cypher-mcp2.md`
   now returns exactly one line — AC-7, with the field list elided to `...`. Yet `S2`'s
   done-condition instructs `cobb` to document *"the FR-7 one-query recipe (§3.1's `MATCH
   (e:KaizenEntry) RETURN ... ORDER BY e.date`) as a **copy-pasteable** example"* — citing a section
   that no longer contains it, and eliding the very field list that makes it copy-pasteable.
2. **The `UNWIND`/`CREATE` migration query is gone.** §4.2 step 2 now says `<X>` *"builds the
   `UNWIND`/`CREATE` migration query (§4.2 shape unchanged from V2: one `author` literal in the
   outer `CREATE` clause, never per-row)"* — a description of a query the document no longer
   contains. This is the delivery's most escaping-sensitive statement, executed twelve times by
   twelve different agents in twelve isolated contexts, and P2-m1's own finding (hand-built string
   literals over entries containing backticks and apostrophes) is the reason it matters *more* now,
   not less. The `author`-literal placement is also exactly what `authorize_write()` enforces, so a
   reconstruction that gets it wrong fails the write.
3. **§3.1 makes a false claim about this document's own layout**: *"see the Version 2 revision note,
   preserved above the Version 3 note in this document's history."* `grep -n "## Revision note"`
   returns one hit — the Version 3 note. The V2 note is not above it; it is not in the document at
   all.

Compressing the *rationale* was a reasonable call — the design has survived two gates and nobody
needs it re-argued. Compressing the *artifacts* was not: an implementer in an isolated context is
now told to copy-paste something the plan doesn't contain, from a section that doesn't have it.

Suggested fix: restore the two Cypher blocks verbatim (§3.1's FR-7 query, §4.2's `UNWIND`/`CREATE`
shape) — roughly ten lines total — and either drop the "preserved above" clause or replace it with a
commit-pinned pointer (`git show <sha>:docs/plans/generic-cypher-mcp2.md`). §3.3's `CREATE` recipe
survived intact and correctly; these two should sit beside it.

**P3-M3 — The inbox-header retarget, as instructed, would falsify the provenance sentence in the
four populated agents' permanent historical records.** §4.2 step 3 says `cobb` *"retargets
`claude/<X>/kaizen/inbox.md`'s header note **from `kaizen_<X>` to `kaizen_team`**"*. Read all 12
headers: four of them (`analyst`, `teco`, `qa-engineer`, `data-scientist` — the ones that had
entries) contain **two** occurrences with opposite tenses. `claude/analyst/kaizen/inbox.md:3-9`:

> Its 5 entries (as of this date) **were imported into the `kaizen_analyst` FalkorDB graph**
> (`claude/cobb/kaizen/history.md`, 2026-08-20 entry); `analyst` no longer appends here. **New raw
> learnings are written directly into the graph** and are immediately queryable by any agent:
> `mcp__cypher__query(graph='kaizen_analyst', cypher='…')`.

The first is a **past-tense provenance statement and it is true** — those entries *were* imported
into `kaizen_analyst` on 2026-08-20; this delivery *relocates* them afterwards. A blanket
`kaizen_<X>` → `kaizen_team` substitution rewrites it into a false history ("were imported into
`kaizen_team`", a graph that did not exist at that date). The second occurrence is the prescriptive
pointer and is the one that must change. The other eight agents' headers (e.g. `coder`) carry only
the prescriptive form and retarget cleanly.

This matters more than a wording slip because these files are, by the stakeholder's own decision,
**permanent records that are never deleted** — and it is `cobb` applying one mechanical recipe
across twelve files, the exact shape that produces twelve identical errors.

Suggested fix: replace the blanket instruction with a scoped one —

> retarget **only the prescriptive clause** ("New raw learnings are written directly into … /
> `mcp__cypher__query(graph='kaizen_<X>', …)`") to `kaizen_team` with an `author:'<X>'` filter;
> **leave the past-tense provenance clause** ("Its N entries … were imported into the `kaizen_<X>`
> FalkorDB graph") unchanged — it is historically accurate and is exactly the content the header's
> own "Content below is preserved" promise and FR-11's past-tense carve-out protect. Four agents
> (`analyst`, `teco`, `qa-engineer`, `data-scientist`) have both clauses; the other eight have only
> the prescriptive one.

Optionally append a dated line rather than editing in place: *"2026-08-<dd>: relocated into
`kaizen_team` (author-partitioned) — see `docs/plans/generic-cypher-mcp2.md` M7."* That keeps the
record strictly additive, which is the most defensible thing to do to a permanent file.

#### Minor

**P3-m1 — `G1`'s first step and §6 open item 1 are now stale: the confirmation they wait on already
exists in the record.** `G1`'s done-condition opens with *"**Before anything else in this unit**:
confirm with the stakeholder (via `tico`) that the never-delete decision's reach stops at
`inbox.md` files…"*, and §6 item 1 carries it as unresolved. `docs/plans/generic-cypher-mcp2-coordination.md:126-132`
answers it — yes, delete the keys — and states the gate is discharged by that record. Leaving the
wording as-is means `graph-dba` opens `G1` by trying to re-ask a settled question through an
interactive agent, which is a real dispatch stall, not just stale prose. Suggested fix: replace
`G1`'s first step with *"the never-delete-reach question is settled — `kaizen/inbox.md` files only,
not the graph keys (`docs/plans/generic-cypher-mcp2-coordination.md`, 'Resolved 2026-08-20'); no
re-confirmation needed"*, and move §6 item 1 from Open items to a one-line resolved note. V3 was
right to decline to decide this unilaterally; it simply got answered between drafts.

**P3-m2 — AC-8's corrected pattern is right, but its widened roots need a third classification
bucket and a pre-declared exclusion, or it will generate false defects at acceptance.** Ran it as
written: 64 files, of which **19 are under `docs/`** (historical plans, reviews, test plans and
reports for M5/M6 and this delivery's own earlier passes — every one of them correctly frozen
past-tense) and **17 occurrences sit in `cypher-mcp/tests/test_server.py`**, where
`kaizen_graph_dba` is an *arbitrary fixture graph name* passed to `run_query()`, semantically
irrelevant to which graph the team actually uses. AC-8's stated rule offers exactly two buckets —
"genuinely historical (past-tense)" or "a real remaining gap" — and the test fixtures are neither.
A `qa-engineer` applying the rule literally would either file 17 false defects or churn a green
suite renaming fixtures. Suggested fix: add a third bucket ("an arbitrary fixture/example graph
name in test code — semantically irrelevant, out of scope"), name `cypher-mcp/tests/test_server.py`
as pre-classified out of scope in AC-8's own row, and consider narrowing the `docs/` root to
`docs/BACKLOG.md` + `docs/requirements/generic-cypher-mcp2.md` (the only two `docs/` files any unit
touches) so the check stays signal-dense.

**P3-m3 — `S3` and `S4` are both listed "Depends on: —", but the FR-12 asymmetry only holds once
*both* land.** `S3` removes the seed step from the creation convention; `S4` stops the auditor
requiring the file. If `S3` lands alone, any agent created in that window is born without an
`inbox.md` and immediately fails `audit-team.sh` check 1 — Pass 1's M1, reintroduced through the
front door. The window is narrow (no new agent is being created during this rollout) but the edge
is free: make `S4` a hard predecessor of `S3`, or mark them "land together". `S4` alone is harmless
in either order, so the ordering is strictly one-directional.

**P3-m4 — §4.2's table lost the two-actor annotation on five of twelve rows, and carries no
"Depends on" column now that `S0` is a hard predecessor.** `C-graph-dba`, `C-architect`, `C-coder`,
`C-devops`, `C-frontend-engineer`, `C-tdd-engineer` and `C-tico` spell the split out as
*"`<X>` (data) / `cobb` (prompt)"*; `C-analyst`, `C-data-scientist`, `C-qa-engineer`, `C-teco` and
`C-cobb` list a single agent. The §4.2 prose covers it correctly (*"`cobb` (for all 12, including
`architect` and `graph-dba` …; `cobb` self-edits only its own file)"*), so this is presentation, not
substance — but `teco` dispatches from the table, and P2-B2 existed precisely because an ownership
error hid in this column. Same for `S0`: P2-m4 made it a hard predecessor of all twelve units, and
that edge appears in `S0`'s own row and the §4.2 heading but in no `C-<agent>` row. Suggested fix:
annotate all twelve Agent cells uniformly and add a `Depends on: S0` column (or one line under the
table stating it applies to every row).

---

### What's solid

- **The Pass-2 blockers are closed on their merits, not by assertion.** `S3`+`S4` genuinely produce
  the "existing permanent / new never created" asymmetry — I traced check 1's logic and confirmed
  narrowing it to a two-way conjunction is sufficient. `C-architect`/`C-graph-dba` ownership is
  corrected in the table's own owner field, with §3.7 giving the reasoning and re-deriving both
  evidence lines independently.
- **The `S3` line list is now exactly right** — 14 lines, matching my independent grep character for
  character, including the curly-brace spellings it previously missed. And it is now backed by
  grep-as-done-condition, so it can't rot again. `S1` got the same treatment unprompted.
- **The frozen-header carve-out is not merely defensible — the files already authorize it.** Every
  header scopes its own promise to *"Content below … will not change."* V3 reached the right
  conclusion via a weaker argument than the one available to it.
- **`T1` is correctly scoped, correctly owned, and correctly bounded** — including the non-obvious
  call that FR-12/AC-9 need *no* requirements edit under branch (b), and the refusal to preempt
  `tico`'s document-mechanics choice.
- **P2-m1's fix is the right one.** Every `C-<agent>` verification is now a field-by-field content
  diff (`entryId`, `date`, `fact`, `evidence`, `context`, `suggestedHome`, `createdAt`) rather than
  a count — which is what makes `G1`'s irreversibility acceptable.
- **P2-m4's fix does more than close the duplicate hole.** Making `S0` a hard predecessor also
  retires the FalkorDB constraint-idempotency question outright, rather than deferring an
  unverified-behaviour claim to implementation time. That was the better of the two available
  answers and V3 took it.
- **Every minor and the nit were adopted without argument, including P2-m6 — which was a correction
  to V3's own predecessor's claim.** A revision that concedes its predecessor overstated an
  "it's accurate" finding, in writing, is doing the job.
- **The snapshot-vs-live discipline is now explicit throughout**: §4's preamble states every count
  is a 2026-08-20 snapshot re-derived at dispatch, `G1` leads with the live re-list, and
  `C-architect`'s row carries the corrected "1 entry, not 0" that Pass 2 surfaced. The plan now
  behaves correctly under FR-13's incremental window instead of pretending the window is closed.

### Open questions

None blocking. One judgement call for `teco`, stated as a recommendation rather than a question:
the three majors above are done-condition wording fixes with no design content, on a plan that is
still `active` and not yet executed against. **Apply them in place and dispatch**; a fourth full
plan-gate pass would not pay for itself.
