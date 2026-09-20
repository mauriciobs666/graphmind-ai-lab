# Coordination techniques — on-demand

> **On-demand knowledge base for `teco`.** Situational coordination techniques — brief-writing
> discipline, dispatch sizing, in-flight recovery scenarios, and integration-verification traps —
> consulted when that specific situation arises during a coordination, not needed on every unit.
> Core routing, the ledger mechanic, the pause/resume flow, and the standing Guardrails stay in
> `teco.md`; this file holds the on-demand technique depth that used to bloat it inline.
>
> Origin: extracted 2026-09-17 from `teco.md` as part of K-030 Stage 0 (interim knowledge-base
> relief) — a prompt restructure, not a kaizen distillation. `teco.md` was the extreme case K-030
> was raised over: 11,010 words, 32 lines over 700 chars, longest line 3,105 chars.

## Environment readiness: bring FalkorDB up yourself, then just retry

If any unit, gate, or your own learnings capture will touch FalkorDB, a CPG, or
`mcp__cypher__query`, probe once at decomposition — `redis-cli -p 6379 ping`, or any cheap
`mcp__cypher__query` read. Not `PONG` → **start it yourself**,
`./falkor-chat/scripts/start_falkordb.sh -d`, re-probe, then **just retry the failed query** — the
MCP server is expected to recover its connection pool without a session restart
(`docs/plans/cpg-query-access.md` §7.3; designed and documented in the server's own error text,
but the stop/restart path was deliberately left untested —
`docs/archive/test-reports/cpg-query-access-report.md`), and one retry settles it either way. If it
*doesn't* recover, that's a finding worth capturing, not a dead end. `docker start falkordb-dev` is
*not* the path: the script runs `docker run --rm`, so there is usually no stopped container to
start. Bringing a service up is additive and idempotent, graphs live in the named volume
`falkordb-data` and survive, so this is inside your `Bash` grant — unlike anything **destructive**
(`docker rm -f`, `GRAPH.DELETE`, removing a volume, `pipeline.sh … --reset`), which is never yours
and routes to `devops`/`graph-dba` behind their guards. If the bring-up itself fails — daemon down,
port taken, image pull fails — that is a real environment blocker: dispatch `devops`, don't hand it
to the user. A missing service is a task you complete, not a caveat you report.

## Editing the coordination ledger: assert the match, and anchor on a row, not a header

A scripted string replace that asserts nothing no-ops silently and leaves a `Status` cell stale —
the exact field a resumed or post-compaction session reads first to decide what is still queued.
Asserting the count is necessary and **not sufficient**: a document holding two tables makes a
*header* unique text, so a matched, asserted insertion lands in the wrong table. **Anchor a row
insertion on the last data row of the intended table**; a cell-count check validates shape and is
blind to location. Confirm placement afterwards by **line number** — `grep -n '^| U'` and look for
the gap — which is the only check that can see it.

That fix has its own way to fail: anchoring on the row's **prefix** rather than its full text. A
ledger row's own cell content routinely runs to hundreds of words, and can by coincidence contain
a substring matching another row's opening text — so a prefix-only anchor can match *inside* the
wrong row's content instead of at the intended row's boundary, splicing the new row into the
middle of it and orphaning the rest of that row's content onto the end of the newly inserted line,
even though the anchor still resolved to exactly one match (an occurrence-count assertion does not
catch this — the match is unique, just in the wrong place). This happened inserting a later unit's
row using an earlier, already-committed unit's row-start prefix as the anchor; caught only by a
pipe-per-row count (`tr -cd '|' | wc -c` — every row should carry the same cell count) before the
document was ever committed. **Anchor on the full text of the row you're inserting after, or a
trailing-newline-bounded marker — a prefix is never a safe anchor, even when it names the right
row.**

## Stating a prior as a prior, not as background fact

Any figure you put in a brief as background — a total, a running tally, a decomposition — is
unreviewed by construction and lands straight in the delegate's done-condition, where it invites
reproduction instead of measurement. So state a prior **as a prior**, name the measurement that
would refute it, and tell the delegate to recompute rather than adopt it. The tell that it went
wrong is a delegate reporting your own figure back to you with more precision than it could have
derived — and the correction is not more care on your side, since a brief nobody reads cannot be
made trustworthy by being written more carefully.

## A constraint stated as a negation is not a constraint — give it a number

*"One dated line, not a narrative"* is two instructions, one checkable and one not; a delegate
satisfied the checkable half exactly — 150 words on a single physical line — and nothing in the
brief could have distinguished that from what was wanted. When a delegate follows the letter, audit
your own wording before theorising about what else influenced it: the tempting post-mortem there
was that the document's house style had outranked the brief, and measurement killed it (that
file's revision notes ran 84 to 1,581 words — there was no uniform precedent to be captured by),
which matters because it points at the wrong fix. Any length, tone or scope limit goes in as a
threshold **you could check yourself** — *"≤40 words"*, *"at most three bullets"* — and if you
cannot state the number, you have not yet decided what you want. Where you do have a view, mark it
a steer that may be overridden.

## Carry a coordination's accumulated defect lessons as methods, never as exhortation

Not *be careful*, but *compute any load-bearing count under more than one definition and report
the spread*, and *run a failure probe alongside a passing control, so a uniform result cannot be
misread as a working mechanism*. A method transfers because it constrains how the delegate checks
its own work; another unit's **finding** does not, and still goes as a question rather than a
premise.

## Mutation-test the green-on-arrival tests

Ask implementers to break the implementation deliberately and confirm the test fails — a test that
passed before they touched anything proves nothing about whether it exercises the new code. Where
the plan argued down an alternative, name that alternative as the mutant — deleting the chosen
mechanism proves only that the test reaches the code; re-implementing the rejected design is what
proves the decision was load-bearing. Brief the restore as **by copy, after each mutation, never
batched** — a run killed mid-mutation otherwise leaves a broken source file indistinguishable from
ordinary uncommitted work. Verify the backup step itself — don't trust that a later line running
means an earlier one succeeded: the harness's Bash tool doesn't run with `set -e`, so a failed
backup can be followed by output that reads as confirming it worked; chain the backup to its own
check (`&&` a `test -s <backup>`, or inspect `$?`) rather than inferring success from what runs
after it.

A delegate's own mutation table systematically covers control flow and under-covers data plumbing:
it mutates the branches it wrote, not the arguments it passed. So at integration, mutate the one
**argument** whose corruption would be silent — a parameter driving a statistical correction, a
family size, a seed — even when the delegate reports a full table with every mutation caught. A
full table is exactly what a defect in the *fixture* leaves intact, because mutation testing
perturbs the implementation and the fixtures are the half the unit also wrote. Two shapes, both of
which pass every test and kill every mutant: a check keyed on a name the unit **invented** — every
fixture carries the invented key, so the check is dead on any artifact written to the spec and
silent about it — and an assertion satisfied by **uniform stub values**, where
`aggregate >= sum(parts)` holds for any single part because every stub returns the same number.
Both are the same failure: a check whose negative result means *"clean"* and *"I could not see the
artifact"* indistinguishably. So brief two probes the delegate's own table cannot contain — build
the check's input from the **spec's own literal**, never from the unit's fixtures, and confirm no
two stubs in a fixture share a value the assertion depends on.

## Brief for the kill: land incrementally, and make the artifact re-derivable

This harness kills long runs without warning, and the answer is not to shorten or serialize
dispatches — it is to tell every delegate to land its deliverable on disk **incrementally**, so a
kill costs the section in flight rather than the whole run. The durable artifact is whatever
*re-derives* the content, not the prose holding it: where the deliverable is measurement, that is a
re-runnable script (which also catches its own bugs, as a number carried in context cannot); where
it is prose, a skeleton written first and filled section by section. Not into an authoritative
document a gate reads — a delegate is right to refuse skeleton-first there, because you commit by
explicit path and would commit its `pending` markers. The delegate's own working evidence dies with
it too, and that is the half briefs forget: its context goes, and a wiped scratchpad takes the
rest. Tell it to write probes, harnesses and baselines to disk and to redirect command output there
**before** running, not to hold them in context — otherwise what reaches you is a completed-looking
notification reporting work with nothing behind it.

## A reviewer's suggested fix is a finding to judge, not an instruction to apply

Brief the implementer to read the defect's own root-cause docstring/contract and the acceptance
criterion it violates, and to say so when the recommendation itself is wrong. Three shapes it
takes: a live reproduction proves one path is broken, not that it's the only one (a report flagging
a second call site as "likely also affected, not verified" means the narrower fix is probably
incomplete); *"checked, not guessed"* names the **method**, not the **scope** — the identical
escape can sit one file over, so the question that catches it is whether the rule an exemption
*states* is broader than the reach its mechanism *implements*; and folding a proposed guard into
the very step it is meant to catch **deletes the guard while keeping its name** — authored beside
the change it guards, its declared set contains the new value from birth and can never redden, so
split it: the assertion into a round before, the obligation into the later step's row. A gate that
proposes a fix has implicitly routed it, and that routing is its least trustworthy opinion — a
reviewer optimizes for closing its finding, not for which discipline owns the decision. Re-derive
the owner from the artifact the fix would change, and brief an owner you chose over the gate's with
your reasoning stated openly plus an instruction to overrule it.

## Size each dispatch to the plan's own step-table boundaries

A plan's step table (`L1-1..L1-6`, `U1-U9`, etc.) sequencing several files across many steps is a
signal to **split the dispatch to match those step boundaries**, not hand the whole table to one
agent as "one coherent diff" — even when adjacent steps share files, which reads like an argument
for one agent but in practice produces one long, hard-to-checkpoint run that silently drops scope.
Default: a step table spanning **more than ~3 steps or more than ~5 files** is the decomposition
boundary — dispatch one unit per step or small adjacent-step cluster, sequenced as dependent
(same-file) briefs, never as one landing-wide mega-brief. This rule governs *sequentially
dependent* work, and it is not a cost rule — it buys checkpointability and guards against silent
scope drop, both of which a split genuinely delivers when step N's output is step N+1's input. A
**batch of independent items** is the opposite case and sizes the opposite way. Dispatch cost is
dominated by a **per-run floor** — measured on this repo at roughly **143k tokens fixed plus ~8k
per item**, so a one-item dispatch costs about **18×** the marginal item — while a batch whose
items each land on disk as they complete already has its checkpoint granularity *inside* the
dispatch, where a kill costs one item rather than the run. So the sizing question is never how many
items, but whether a kill costs you the run or one item: dependent work splits at the step boundary
and pays a fresh floor for the checkpoint; independent work sizes **up** to the largest coherent
batch the delegate can hold, and a one-or-two-item dispatch is a cost smell to merge into its
neighbour.

## Fencing note: a graph-write fence has to name the exception, not just the subtree

Raw learnings capture is a graph write (`mcp__cypher__query` against `kaizen_team`,
author-partitioned), not a file write, so a brief excluding a subtree (e.g. "don't touch `claude/`"
to dodge a collision) does not block it — no carve-out is needed for that step. If a brief
separately needs a delegate to append to its own `kaizen/history.md` inside an excluded tree (a
distillation step, normally `cobb`'s job), carve that file out explicitly instead. The same trap in
the other direction: a "no graph writes" fence — drawn against destructive/shared-state ops — reads
on the delegate as barring the additive, agent-partitioned `kaizen_team` producer-write every agent
is supposed to make, so carve that out by name too. Fence by file, not by directory, and assume
nothing is obvious: a delegate cannot tell an intentional bar from an over-broad one, and the
correct thing for it to do with the doubt is to comply and report — which costs a round trip to
lift.

## A `completed` notification's result can be a stale mid-task placeholder

This happens when the delegate kicked off its own long-running background step and its own turn
ended before that step's completion reached it. If the result reads like a status line ("in
progress," "will wait for completion," no concrete figures/paths the brief asked for) rather than
an answer to the brief's stated deliverable, `SendMessage` a plain "report your actual current
state" check to the same `agentId` before treating the unit as `delivered`, updating the ledger, or
dispatching a dependent unit on it — one extra round-trip, versus risking a false `delivered` row.

## Recovering from a transient platform failure

A 500, a 529, a timeout, a killed run is not a deficient result, and a fresh respawn is not the
first move: the delegate's file edits **and** its conversation context both survive, while its
notification carries the last line it emitted as the `<result>`, which routinely reads as far less
progress than actually landed. `SendMessage` the `agentId` in its ledger row and let it re-orient
from disk; attempt the send first and treat an addressing error as the non-resolution signal — you
have no agent-enumeration *tool*, though an id missing from the ledger is recoverable from the
session's own subagent transcripts. Only on non-resolution, re-dispatch with a **state-recovery
brief**: inspect `git status`/`git diff` and the on-disk artifacts, continue from actual state
rather than starting over.

## On abnormal termination, diff the tree before checking the deliverable was written

Ahead of checking whether the deliverable was written: a delegate killed mid-mutation-test leaves a
deliberate defect the next unit reads as real code.

## When a unit writes two surfaces, ask whether they agree — not whether anything was written

Where the unit writes two surfaces — an artifact and the record of it — "was anything written?" is
the wrong question, because the answer is yes and it misleads. A diffstat says a file changed; it
does not say *which of the two* changed, and a kill routinely lands between them. Ask instead
whether the artifact and its record now **agree**, and probe with the specific content that should
have moved: a correction that landed on only one surface leaves the retracted figures still
greppable in the other, and a promotion that landed on only one leaves the artifact carrying a rule
its log never records.

## What decides resume-versus-fresh is what the transcript uniquely holds

Resume to recover reasoning that exists *only* there, and dispatch fresh once that reasoning is
pinned by assertions already on disk. A unit killed after a completed TDD **red** phase is the
clean case — the tests moved the design out of the transcript and into the tree, so a fresh
dispatch reconstructs nothing, which is a resilience property worth weighing when you route a unit
likely to be interrupted.

## A delegate's edits can vanish mid-run with no platform failure and no git trail at all

A second, distinct cause of apparent lost work, distinct from a kill. Not delegate error: treat the
report as credible. Recovery is the same discipline as a platform kill — redo from the delegate's
already-settled design, snapshot in-progress artifacts to scratchpad as insurance, and commit by
explicit path promptly once concurrent churn is visible in `git status`, rather than leaving
verified work uncommitted. This promptness is an explicit exception to the docs-only-chain batching
rule ("Commit what you verified") — even mid-chain, a unit whose already-verified work just proved
at risk of a second loss is committed immediately by explicit path, not held for the chain's
terminal commit; protecting it outranks the batching preference. The rest of the chain still
batches normally once this one unit is safe.

## An `agentId` resolves only inside the session that spawned it

It survives a mid-run failure but not a session reboot — `SendMessage` then returns *"No transcript
found for agent ID"*. So a checkpoint written *because* the session is dying is exactly the one
whose recorded ids will not resolve on pickup: give every in-flight unit in it an explicit
cold-start fallback — which upstream artifact paths and section anchors a **fresh** agent needs,
plus what is already done and must not be redone — never just "resume `<agentId>`".

## A finding that invalidates a still-running sibling's premise gets relayed immediately

`SendMessage`d immediately, not held until that sibling delivers — the correction is cheap even
when the sibling reaches the same conclusion independently. But the relay destroys the independence
of any later agreement between the two, so record it in the ledger at the moment you relay it,
never afterwards. A sibling that adopts a relayed conclusion is *informed*, not convergent, and by
delivery time the two results are indistinguishable from independent arrival at the same answer —
which is how a single finding gets written up as its own corroboration.

## A defect class named by one gate is worth folding into another in-flight gate's brief

Even in an unrelated component with no shared files. Each delegate runs in an isolated context and
sees only its own arc, so you are the only party positioned to make the transfer — and a reviewer
told which shape to look for finds it where nobody would have thought to look. Send the
**question** (*"what neighbouring wrong implementation would this guard still pass?"*), never the
other arc's finding as a premise.

## An incoming resume/pause message's intent is authoritative; its factual state claims are not

These can describe the coordination as several dispatch-and-verify cycles more stale than it
actually is by the time you process them — re-read `git log`/`git status` and the coordination
doc's own ledger before acting on what such a message says is currently true, not just before
acting on a directive that might be stale.

## A message describing a task/coordination absent from your own ledger is a misrouting signal

Not just a staleness signal. Cross-session `SendMessage` addresses by bare agent name, which
resolves ambiguously when more than one independently-launched session shares that name — the
sender may simply have the wrong `teco`. Treat a total mismatch as reason to pause and confirm
identity with your own user *before* doing anything in response, even read-only verification.

## Verify a self-reported recovery independently

If a delegate reports rebuilding work it damaged, a green suite proves only that nothing
*test-covered* was lost; diff the symbol inventory against the last commit as the cheap independent
check.

## Your own rebuttal of a delegate's report is the least-checked claim in the coordination

No gate reads it. Before overriding one, confirm your evidence can *see* what the delegate claims:
`git log`/`git show` cannot see live graph state (a shared graph holds snapshots published from
uncommitted trees), and "none of *my* delegates wrote it" is not "nobody did" — another session
commits to this tree.

## A clean, stable, or empty re-derivation is not a refutation — it may be a bug in your check

Reproduce with the code's own expression rather than the units the document prints, and ask the
delegate for its construction before overriding it. Empty is the case that hides, because it reads
exactly like the artifact being missing — a broken instrument and a real absence return the same
thing, so an empty result earns the suspicion a clean one gets, not less.

## Absence measured in one place is not absence in general

Absence also defeats the reach check, which asks what the widest set a sentence covers is: a claim
that something does *not* exist has no members for that probe to bite on, so it passes vacuously.
For a negative the counter-question is a different one — **where would this be if it did exist, and
did I look there?** Absence measured in one directory tree, one file, or one name spelling is never
absence in general.

## Corroboration needs independence of method, not a second agent

Two agents running the same grep are one check, and a grep hit proves the value exists, never that
the entity you named owns it: verify an attribution by reading the enclosing definition. Never tell
a review gate that one of your own conclusions is settled and out of scope; it is the one input a
reviewer has no independent reason to re-check.

## Your own instrument is the least-checked thing in the gate, and agreement is where it hides

A verifier who rebuilds a delegate's measuring instrument tends to rebuild its bug, because the
obvious implementation is the same wrong one — and an instrument that happens to agree with the
figure you were checking costs as much as one that contradicts it, while being far harder to
notice: agreement ends the inquiry where disagreement starts one. Compute any load-bearing count
under **several definitions and report the spread**, rather than trusting the single run that
matches your hypothesis.

## A refutation is not a verification of its neighbours, either

When you find one wrong figure in an evidence line, the remaining figures in that line become
*less* likely to be checked, not more, because the refutation consumes the verification budget and
the survivors inherit the original authority. Check every figure in a citation independently of
what happened to the one beside it.

## Closing an upstream artifact on its gate verdict is not absorbing it

A note or plan that ends with a "handed onward / open items" section creates obligations on
*downstream* documents that its own review never checks — sweep that list into the downstream
artifact item by item at closure and record each disposition, or the downstream step ships a
done-condition that cannot fail.

## When one revision closes several findings, the next defect is the interaction of two correct fixes

And no finding's own re-check can see it, because each fix is sound against the finding it closes.
Brief the re-gate to check same-revision fixes **against each other**, not only each against its
own finding. Corollary: a delegate reporting belt-and-braces — both available repairs applied to
one finding — is the signal to check the interaction, never a reason to relax.

## A green offline suite over hand-built fixtures can hide the same gap a model's free-text output does

Where the behavior depends on a live model's free-text output, a green offline suite is not a
proof run — the suite exercises your own mocks, and the model's casing, wording and tool-call shape
are exactly what it cannot vary; require a live proof run before accepting such a unit. The same
gap can hide behind a purely deterministic pipeline, not just a model's free-text output, when
every offline test drives an intermediate stage directly with hand-built objects and none drives
the real producer→store→load→consumer round trip — a stage can stay green for as long as no test
happens to cross that boundary, so treat the first live end-to-end run through the *actual*
pipeline as a genuine proof point, not a formality, whenever a unit's test estate is built entirely
on hand-constructed fixtures at one stage.

## A repeated gate has a decidable stopping signal, and the reviewer sets it, not your patience

When the marginal finding is produced *by* the fixes rather than found *in* the original artifact,
further static passes have negative expected value — ask the reviewer for a **falsifiable stopping
condition** rather than another fix, promote it *above* the individual findings as the
implementer's done-condition, say in advance what happens if it fails, and move to execution gates.
Read convergence off the *ratio* of findings that required judgement, never off the finding count —
a count is flat and unreadable from inside a sequence of passes. Ask the reviewer to classify each
finding by one question: *would this still exist if the document were mechanically consistent with
itself?* A pass dominated by sweep-class findings buys a **pin discipline**, not another pass; one
still producing design-class findings has not converged whatever its total. Get the branch
pre-stated before the next revision exists, so the stopping call cannot be fitted to the outcome
after the fact. Where the defect is a guard claiming more reach than its mechanism implements, put
**both** closures to the implementer — widen the mechanism, or narrow the claim to what it does —
because a team that has widened repeatedly will not propose retreating on its own.

## An audit's clears are its untrusted half

When a pass audits an inventory against a stated criterion, the items it flags get fixed and
re-checked while the items it cleared are never revisited — so the moment that criterion is itself
found defective, the flagged set is repaired against the corrected wording and the cleared set
silently keeps a verdict issued under the broken one. Dispatch a re-audit of the **cleared** set
explicitly, and resume the *original* reviewer: the closed inventory is knowledge only it holds. A
second cause of the same asymmetry is worth briefing against up front — an audit that mutates in
**one direction only** reports false clears, because an item with no real coverage still reddens
when some unrelated fixture happens to use what you deleted. Brief any coverage- or reach-audit to
pre-state and run **both** directions, shrink *and* widen, or its pass-list carries no information.

## A clean pass on a brand-new mechanism is evidence the scripted scenario holds, not full coverage

Not that the mechanism's full state space is exercised. One scripted "PASS, no defects" QA pass on
a mechanism nobody has used for real yet doesn't mean every real usage pattern got surfaced — a
stakeholder's follow-up ask to run it "for real"/end-to-end after a PASSed report is a legitimate
follow-up unit, not a redundant re-test.

## A pathspec commit protects a concurrent session only where the paths are disjoint

A pathspec commit still takes that path's *whole* working-tree content, so it protects a
concurrent session only where the paths are disjoint: if a concurrent session has also edited a
file your delegate appended to, no pathspec separates the two, and `git add -A` is not the only way
to sweep someone else's state. Leave that file uncommitted, record in the ledger where the work is,
and say so in your report — nothing is lost but attribution.

## Verifying which files are actually yours before an integration commit: the three-way diff check

Which files those are is a thing you measure, not a thing you assume: before every integration
commit, diff each file three ways — baseline sha, `HEAD`, worktree. Only `baseline == HEAD` with
your delta in the worktree proves the delta is yours; `HEAD` having moved means another session
committed *inside* your file; and a file you expected to be dirty that instead matches `HEAD` is
the tell that your content was already committed by someone else — a signal to go and check, not a
proof, since a delegate that never wrote is the other explanation. **Compare content, not size** —
a word count agrees by coincidence and is the same self-deceiving instrument the gate rules
elsewhere warn about. `claude/cobb/kaizen/{history,plan}.md` are structurally shared and no commit
discipline partitions them: every coordination that dispatches `cobb` writes them by construction,
so serializing dispatches does not help and the three-way check is mandatory, not prudent, whenever
`cobb` is in the unit list.

## Holding a shared file out of your commits is a bounded tactic, not a resting state

The other session's commit cadence is not observable, so the cost of holding rises with no natural
end — past two or three units the verified work you are sitting on outweighs the misattribution you
are preventing. Release it then: commit the file with the foreign content **verbatim** — untouched,
unreflowed — and identified in the commit message body. Attribution in prose costs nothing; holding
risks losing every unit's share at once.

## A standing commit grant does not exempt a self-governing file from the platform classifier

Your own git-commit authority (a coordinated specialist's already-verified deliverable, by explicit
path) is a repo-level convention; the harness's auto-mode classifier gates a self-modification-
flagged file — one that changes this session's own trusted tool/config surface — on the file
itself, not on who is asking or how well-verified the diff is. Don't read a delegate's block on
such a file as delegate-specific and assume your own broader grant reaches where theirs didn't.
Observed 2026-09-18 (`docs/reviews/mcp-json-edit-bypass-incident.md`;
`claude/docs/plans/agent-knowledge-base-strategy3-coordination.md`, Stage 3 incident section): a
`devops` delegate's `Edit` on the repo-root `.mcp.json` was denied `[Self-Modification]`; after that
was reported (and its `Bash` workaround identified as a bypass, not acted on), `teco`'s own
`git commit -- .mcp.json` of the resulting, content-verified-clean diff — run only after explicit
user approval — was *separately* denied, reason `[Auto-Mode Bypass]`. The classifier's block was on
the file, not on the tool or the caller. Escalating to the user for a direct, out-of-session write
(what actually resolved it) is the correct next step, not a retry under your own authority.
