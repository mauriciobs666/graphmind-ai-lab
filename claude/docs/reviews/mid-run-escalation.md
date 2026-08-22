# Delegate mid-run escalation — Review

> **Status:** active · **Owner:** `analyst` · **Tracks:** — (—)

## Scope & verdict

Static plan-gate review (pre-implementation) of `claude/docs/plans/mid-run-escalation.md`
(design by `cobb`, Status: active) against its upstream requirements
(`claude/docs/requirements/mid-run-escalation.md`, Status: Ready for design, confirmed 2026-08-21
— FR-1..FR-5, AC-1..AC-5), read in full, and against the current on-disk state of
`claude/teco/teco.md` (the file the plan's §2 diff targets) and `claude/README.md` (the file its
§4 doc-impact list targets). No file under review has been edited — this is a design review, not
a diff review. Baseline: repo state at the time of this review (`main`, 2026-08-21, working tree
clean).

**Verdict: approve with suggestions.** No blocker found: every FR/AC is genuinely satisfied by
concrete `teco.md` text (verified clause-by-clause below, not just trusted from §7's own mapping
table), all three quoted "Current" excerpts match the live file byte-for-byte, the two flagged
judgment calls (§1.2 ledger vocabulary, §2.1 reactive coordination-doc trigger) both hold up under
scrutiny, and the §1.5 zero-grant-changes fact-check reproduces exactly when re-run. One accuracy
issue in the plan's own supporting narrative (Finding 1) and a few completeness/proportionality
suggestions (Findings 2-4) are worth fixing before or shortly after landing, but none change the
design or block implementation.

**CPG: not applicable** — this is a prompt-engineering design for `claude/teco/teco.md` (agent
instructions), not source code sitting in a loaded CPG; the coordination doc for this same slug
independently notes the same thing ("No CPG applies — this is agent-prompt design work, not a
review of CPG-bearing source").

## Findings

### 1 — [Major] The qualifying worked example's "grounded in real history" claim doesn't check out

§1.1 and §2.2 both assert the NULL-backfill worked example is "grounded in this repo's own recent
history (a real NOT-NULL-backfill decision surfaced during the `agent-permission-friction`
investigation...)" — not framed as a plausible hypothetical, but as a citation to an actual past
event in this repo.

I checked it (the task brief said I didn't have to, but it took one grep): there is no such
decision anywhere in this repo's history. `grep -rn -i "backfill" claude/docs/` returns hits only
inside `mid-run-escalation.md` itself; `agent-permission-friction.md`,
`agent-permission-friction-coordination.md`, `docs/reviews/agent-permission-friction.md`, and both
`teco`'s and `cobb`'s `kaizen/history.md` contain no occurrence of "NULL", "backfill", or "NOT
NULL" in any migration/data sense. `agent-permission-friction` was pure hook/permission-guard
engineering (`guard-doc-writes.sh`, `guard-destructive-ops.sh`, frontmatter `hooks:` blocks) — it
never touched a schema, a migration, or production data, so there is no plausible event this could
be paraphrasing either.

This doesn't break the mechanism — the example is well-chosen and legible entirely on its own
terms as a hypothetical illustration of "irreversible on production data" (item 4 of the review
brief asked me to judge exactly that, separately from the grounding claim, and on that axis it
passes). But the plan asserts a specific, checkable provenance for it that is false, in a repo
whose own citation discipline (`AGENTS.md`'s "Citing another document") treats this kind of claim
as load-bearing — and the same document's §1.5 shows `cobb` clearly knows how to actually fact-check
a claim like this (a `grep`, reproduced and cited). The inconsistency is what makes this worth
flagging rather than waving through.

**Suggested fix:** in §1.1 and §2.2, drop "grounded in this repo's own recent history... surfaced
during the `agent-permission-friction` investigation" and replace with a plain "illustrative
example" framing — the worked example itself needs no edit, only the provenance claim around it.
Isolated, one-paragraph fix; doesn't touch the `teco.md` diff text in §2.2's blockquote at all
(the blockquote itself doesn't claim historicity — only the prose introducing it in §1.1 and the
paragraph immediately above the blockquote in §2.2 do).

### 2 — [Minor] The "teco itself as a delegated subagent" resume path is a two-hop chain the plan doesn't spell out

§5 calls the subagent-`teco` case "a structurally consistent extension... not a gap," and that's
right for the *escalation* half (an inner delegate's pause surfaces as `teco`-as-subagent's own
stop-and-ask result, one level up — §2.3's first bullet states the mechanism this relies on). But
the *resume* half isn't addressed: once the stakeholder's answer reaches whoever dispatched
`teco`-as-subagent, resuming the paused *inner* delegate isn't a single `SendMessage` — it
requires first resuming `teco`-as-subagent itself (via `SendMessage` to *its own* `agentId`,
which its own dispatcher must have recorded, mirroring §2.3's second bullet one level up), and
only once `teco`-as-subagent is itself running again does it perform the inner `SendMessage` the
plan describes. The plan implicitly assumes whoever dispatched the subagent-`teco` follows this
same paused/ledger/`SendMessage` discipline, but never says so.

**Suggested improvement:** one added sentence to §5's bullet (or to `teco.md`'s "Pause vs.
proceed" section) stating the two-hop shape explicitly, so a future reader doesn't expect — or
implement — a direct resume of the inner delegate that bypasses `teco`.

### 3 — [Minor] The new brief clause is unconditionally "every brief," unlike the adjacent Model-routing bullet

§2.2's replacement Subagent-awareness bullet says "fold this into every brief close to verbatim,"
with no scoping. The very next bullet in the same list (`teco.md`'s existing "Model routing"
clause, step 3) already distinguishes a **mechanical** unit (apply a stated edit, flip a status
marker) from one that requires judgment — a mechanical unit is structurally incapable of hitting a
"high-stakes fork" (there's no fork to hit), so folding in ~6 sentences of fork-detection guidance
on every such dispatch is pure overhead with no corresponding benefit.

**Suggested improvement:** either scope the new clause the same way ("skip for a unit already
classified mechanical per Model routing") or note explicitly that it's cheap enough to always
include regardless — either is fine, but the plan doesn't make the call, and it's the kind of
proportionality question the plan's own precedent (Model routing) shows `teco.md` already knows
how to answer for an adjacent clause.

### 4 — [Nit] The `paused`-row example doesn't show a full row, and repurposes a path-typed column

§2.1's replacement text gives a `paused` row's `Deliverable`-column value
(`⏸ "does the migration backfill..." — relayed 2026-08-22`) but not the row's `Gate → verdict` or
`Cost` cells — presumably both stay `—`, unchanged from a normal `in-flight` row, but the plan
doesn't say so, unlike `teco.md` step 2's existing sample row, which shows all seven columns.
Separately: every other row's `Deliverable` cell is implicitly a path (the column header and every
existing example are paths); a `paused` row repurposes it to hold free text instead. Low risk
today since the ledger is prose read by `teco` and a human, not parsed by any script — but worth
one line acknowledging the invariant is broken for this one `Status` value, in case that ever
changes.

**Suggested improvement:** add a full seven-column example row for `paused` next to the
`Status`-enum text in §2.1, mirroring `teco.md`'s existing sample row format.

## What's solid

- **Every AC genuinely holds up**, checked against the actual proposed `teco.md` text, not §7's
  table alone: AC-1's Given/When/Then is matched almost clause-for-clause by §2.2's replacement
  bullet (including the "routine ambiguity... unchanged" half); AC-2 by §2.3's first bullet
  (relay, never self-answer); AC-3 by §2.3's second bullet (`SendMessage` by recorded `agentId`,
  never a fresh `Agent` call); AC-4 by §2.3's third bullet (explicit, unqualified "stalls only
  itself"); AC-5 by the §1.5 fact-check, independently reproduced below.
- **All three "Current" quotes in §2 match the live file exactly** — verified by reading
  `claude/teco/teco.md` in full: the `Status`-enum sentence (line 68), the coordination-doc
  trigger clause (line 62), and the Subagent-awareness bullet (line 71) are all byte-accurate
  partial quotes, and the step-4 insertion point ("after `teco.md:89`") correctly identifies the
  misrouting-signal bullet as the list's last item before step 5 begins. A diff built on these
  quotes will apply cleanly.
- **§1.5's fact-check reproduces.** Re-ran `grep -n "^tools:" claude/*/[a-z]*.md`: exactly 6 of
  the 13 agent folders carry an explicit `tools:` line (`analyst`, `architect`, `data-scientist`,
  `security-expert`, `teco`, `tico`), and `SendMessage` appears only on `teco`'s line. The
  zero-grant-changes conclusion for FR-5 is sound, not just asserted.
- **§1.2's ledger-vocabulary decision is well-reasoned**, including its rejected alternative — a
  reused `in-flight` + note column would genuinely defeat the ledger's own "one grep-able word"
  purpose, and the ledger-vocabulary choice is explicitly the requirements doc's Out-of-scope item
  to make.
- **§2.1's reactive coordination-doc trigger is justified, not just declared** — the stated reason
  (a paused unit's `agentId`/question must survive a compaction or interrupted session, and only
  the ledger provides that) is a real, load-bearing requirement given the feature explicitly
  allows an unanswered question to sit indefinitely (FR-4's Out-of-scope "no cap, no deadline").
- **No hook, tool-grant, or frontmatter change is actually needed** — independently confirmed:
  `claude/teco/hooks/guard-coordination-doc-writes.sh` already allows any `docs/plans/*` path
  (bare and `*/`-prefixed), which covers the reactively-created `<slug>-coordination.md` this
  design relies on, with zero change to the guard script.
- **§4's insertion-point citation for `claude/README.md` is accurate** — the sentence it anchors
  to ("...the `agentId` is what `SendMessage` addresses to continue a delegate instead of
  respawning it cold") exists verbatim in the file today.
- **Scope discipline is real, not just asserted** — nothing in §2's diff touches a hook, a
  frontmatter `tools:`/`hooks:` block, or any file outside `teco.md` + `README.md` + (later)
  `kaizen/history.md`, matching §6's stated non-goals exactly.

## Open questions

None that block implementation. Finding 1 is worth a stakeholder-free, cheap fix by `cobb` before
or alongside landing (it's a one-paragraph rewrite, isolated from the `teco.md` diff text itself);
Findings 2-4 are improvements the implementer can fold in during the implementation unit without
another design round.

## Pass 2 — 2026-08-21 (diff-scoped, post-implementation)

Second gate: static review of the **implemented** change — the uncommitted working-tree diff to
`claude/teco/teco.md`, `claude/README.md`, `claude/teco/kaizen/history.md`, plus the Finding-1 fix
to `claude/docs/plans/mid-run-escalation.md` — against the landed text itself, re-derived from a
full read of the file (not trusted from Pass 1's AC-mapping or from `teco`'s own U3 verification
note in `claude/docs/plans/mid-run-escalation-coordination.md`). Baseline: working tree at the time
of this pass (`main`, 2 commits ahead of `origin/main`, otherwise clean per `git status`).

**Verdict: approve with suggestions.** No blocker. Every AC genuinely holds in the landed text, all
three Pass-1 minor/nit findings are correctly folded in, Finding 1's provenance-claim removal
checks out (with one caveat on Item 3 below, which I could not fully close), and scope discipline
is real (`git diff --stat` and `git status` show exactly the four files/three new docs the brief
names, nothing else). One new minor finding on internal ordering; one open evidentiary gap flagged
rather than glossed over.

**CPG: not applicable** — this is a diff review of `claude/teco/teco.md` (agent-prompt text) plus
two catalog/history docs, not source code sitting in a loaded CPG; matches Pass 1's same call and
the coordination doc's own note ("No CPG applies — this is agent-prompt design work").

### Findings

**1 — [Minor] Step 2's new trigger paragraph uses `paused` before the ledger formally defines it**

The new second sentence of step 2's coordination-doc-trigger paragraph (`teco.md:62`) reads "...
before marking anything `paused`" and "A paused unit's `agentId` and open question must survive a
compaction..." — both uses precede the `Status` enum's formal definition of `paused`, which appears
only after the ledger table, later in the same step (`teco.md:69`). A first-time reader hits the
term as an unglossed verb/status before its definition. Low severity: "paused" is ordinary English
and the surrounding sentence is self-explanatory without the formal definition, and the document
already uses forward references elsewhere as an established convention (step 2's own "apply the
step-table sizing rule (§3...)" pointing at step 3; the new Subagent-awareness bullet's "Model
routing below"; "see 'Pause vs. proceed'" in step 4, defined only later in the file) — so this is
consistent with the doc's existing register, not a new defect pattern. Flagged only because the
task brief specifically asked me to check for step-2/3-vocabulary-before-step-4-use ordering, and
this is the one instance I found, just in the opposite direction (step 2 using vocabulary step 2
itself hasn't yet formally defined, not step 4 outrunning step 2/3).

**Suggested improvement:** optional — move the `Status`-enum sentence immediately before the
trigger paragraph, or add "(see `Status` below)" on first use. Cosmetic; does not block approval.

**2 — [Nit] `Deliverable`-column quoting style is straight quotes, not curly — consistent, but worth confirming intentional**

The `paused`-row example (`teco.md:67`) uses straight double quotes around the question text
(`⏸ "does the migration backfill NULLs to 0 or error the row?" — relayed 2026-08-22`), matching the
plan doc's §2.1 example exactly, character-for-character. No inconsistency found — noting only
because a table cell containing an embedded `|` character would break Markdown table parsing, and
I checked: the actual question text contains no `|`. Not a real risk today; worth remembering if a
future `paused`-row question ever needs to quote something containing a pipe (escape it, or the row
silently truncates at render time).

### Item-by-item verification (per the brief's numbered checklist)

1. **AC-1..AC-5, re-derived from the landed text, independent of Pass 1's table:**
   - **AC-1** — `teco.md:71` (Subagent-awareness bullet): the Given/When/Then is matched
     clause-for-clause — the qualifying test ("would change scope, touch something irreversible, or
     waste substantial downstream work") is verbatim from AC-1, the non-qualifying "routine
     ambiguity... states its reasoning and makes the call itself, same as always" preserves the
     unchanged default in the same bullet. Holds.
   - **AC-2** — `teco.md:95` (step 4, first new bullet): `teco` relays via `AskUserQuestion`
     first-order or its own report as a subagent, "Never answer it yourself or silently decide on
     the delegate's behalf." Holds.
   - **AC-3** — `teco.md:96` (second new bullet): `SendMessage` addressed by the ledger-recorded
     `agentId`, "never a fresh `Agent` call re-briefing it cold," answer folded into the message.
     Holds.
   - **AC-4** — `teco.md:97` (third new bullet), unqualified: "a `paused` unit stalls only itself
     ... every other independent, already-dispatched unit ... keeps moving." Holds.
   - **AC-5** — no delegate `SendMessage`-grant text appears anywhere in the new bullets — the
     mechanism is structurally agent-agnostic (`teco` is the sole `SendMessage` caller in every new
     bullet), so the path is available identically regardless of a delegate's own tool grants.
     Holds by construction, same as Pass 1's §1.5 fact-check concluded.

2. **Pass-1 minor/nit findings, re-verified against the landed diff:**
   - **Finding 2 (two-hop chain):** spelled out at `teco.md:95` — "the same 'no live back-and-forth'
     limit that applies to the paused delegate applies to it too: ... resuming the paused *inner*
     delegate is a **two-hop `SendMessage` chain, not one** — the dispatcher must first resume
     teco-as-subagent itself (addressed by *its own* recorded `agentId`) before teco-as-subagent can
     perform the inner resume." The mechanism described is correct: it requires teco-as-subagent's
     own dispatcher to have recorded *its* `agentId` at dispatch time, mirroring the same discipline
     step 3's "Record the identity" bullet already mandates for every `Agent` call — a reasonable,
     structurally consistent extension, not a new assumption invented for this bullet. Correctly
     addressed.
   - **Finding 3 (unconditional brief clause):** now scoped at `teco.md:72` — "except a unit already
     classified mechanical per Model routing below, which is structurally incapable of hitting a
     fork worth stopping for (apply a stated edit, flip a status marker, run a suite and return its
     output — nothing there is a fork)." This is coherent with the Model-routing bullet it's
     compared against: "Model routing below" is a genuine forward reference (Model routing appears
     later in the same step-3 list, `teco.md:83`), and the mechanical/judgment split used is exactly
     the one Model routing itself already draws for its own purpose (haiku-eligible mechanical work
     vs. summarization/judgment work) — same taxonomy reused, not a new one invented. The scoping
     doesn't undermine AC-5's team-wide availability: it's a per-unit distinction (does *this*
     dispatch's work have a fork to hit), not a per-agent-type one — a "mechanical" unit could be
     assigned to any specialist, so no agent type is silently excluded from the mechanism itself.
     Correctly addressed.
   - **Finding 4 (incomplete ledger example):** a full seven-column `U3` row now sits directly below
     the existing `U1` sample row (`teco.md:67`): `U3 | `coder` | `7f3ac91b2e4d8` | paused |
     ⏸ "does the migration backfill NULLs to 0 or error the row?" — relayed 2026-08-22 | — → — | —`.
     `Gate → verdict` and `Cost` both correctly show `—` (no gate has happened yet, matching the
     prose sentence added right after the table: "`Gate → verdict` and `Cost` stay `—`, unchanged
     from any other not-yet-gated row"). Column-formatting conventions match the `U1` row exactly:
     `Owner` and `Agent id` backticked, `Status` plain text, `Gate → verdict` plain dashes when no
     gate has run (same as `U1`'s own `— ` pattern would be, extrapolated). The one deliberate
     deviation — `Deliverable` not backticked, since it's now free text rather than a path — is
     explicitly called out in the surrounding prose ("the only `Status` value that repurposes that
     column away from its normal path-typed use"), exactly matching Finding 4's ask. Correctly
     addressed.

3. **Finding 1 (false provenance claim) — removal confirmed for §1.1; §2.2's original-content claim
   could not be independently confirmed either way.**
   `grep -rn -i "grounded\|agent-permission-friction\|backfill" claude/docs/plans/mid-run-escalation.md`
   returns exactly one hit, in §1.1: "...illustrative example (a hypothetical NOT-NULL-backfill
   decision, and a reversible naming choice)..." — no remaining claim of historicity or a citation
   to the `agent-permission-friction` investigation anywhere in the file. §1.1's fix is confirmed
   clean.

   On the implementer's specific claim ("§2.2 never had this problem in the first place — only §1.1
   did," per the coordination doc's U3-outcome note) — **I could not independently verify this**,
   and flag it as a genuine evidentiary gap rather than accepting `teco`'s own U3-outcome note or
   `cobb`'s report at face value, per the task brief's explicit instruction. `claude/docs/plans/
   mid-run-escalation.md` is **untracked** in git (`git status` — it was never committed, `git log
   --follow` on the path returns nothing, no reflog/stash entry references it), so there is no
   version-controlled "before" state to diff against, and I found no other on-disk backup or copy.
   What I *can* confirm: the **current** §2.2 has no prose paragraph anywhere between its header and
   the "Current (`teco.md:71`):" label, or between "Replace with:" and its blockquote — the section
   follows the same bare header→"Current:"→blockquote→"Replace with:"→blockquote shape as §2.1 and
   §2.3, neither of which carries an intro paragraph either. That's consistent with the implementer's
   claim (no paragraph exists now, and the section's shape gives no structural gap where one was
   excised), but it is not proof the claim is correct — a deleted paragraph leaves no textual trace
   either way, so "no paragraph now" is equally consistent with "one existed and was cleanly
   removed." Pass 1's Finding 1 explicitly asserted the opposite ("the paragraph immediately above
   the blockquote in §2.2" also made the historicity claim) — that specific, textually located claim
   and the implementer's rebuttal cannot both be checked against a surviving artifact. Practically
   moot for the review's verdict either way: the current file, read as it stands, is clean.

4. **Coherence as a whole:** read `teco.md` in full, not just the four edited spans. No internal
   contradiction found between the new text and the surrounding unedited material — the new step-4
   bullets' "Pause vs. proceed" cross-reference is accurate (that section exists at `teco.md:119-121`
   and does describe exactly the `AskUserQuestion`-withheld-as-subagent behavior the new bullet
   relies on); the new step-2 trigger's "(step 4)" and Subagent-awareness's "Model routing below"
   are both genuine forward references matching the document's existing style. One genuine, if
   minor, ordering wrinkle is Finding 1 above (a "step-2-outruns-step-2" case rather than the
   step-4-outrunning-step-2/3 shape the brief specifically asked me to hunt for — I checked for that
   shape too and found no instance of it: every step-4 bullet's vocabulary — `paused`, `agentId`,
   ledger, `SendMessage` — is fully defined by the time step 4 uses it). The `paused`-row example's
   format matches the table's existing column conventions exactly (Finding 4 above, verified
   cell-by-cell).

5. **Scope discipline:** `git status` and `git diff --stat` (run against the whole repo, not just
   the four named files) show modified: `claude/README.md`, `claude/teco/kaizen/history.md`,
   `claude/teco/teco.md`; untracked: `claude/docs/plans/mid-run-escalation-coordination.md`,
   `claude/docs/plans/mid-run-escalation.md`, `claude/docs/reviews/mid-run-escalation.md` — exactly
   the plan/coordination/review docs this feature's process itself produces, nothing else. No hook
   file, no frontmatter `tools:`/`hooks:` block, and no file outside this feature's own docs tree
   touched anywhere. Also ran `bash claude/scripts/audit-team.sh` as an independent corroborating
   check (in-bounds — a read-only script) — full `PASS` on every deterministic check, including the
   `teco` roster/catalog/hook-existence checks and the commit-authority checks; no regression
   introduced by this change.

6. **`README.md` accuracy:** the new sentence (inserted immediately after the existing
   `agentId`/`SendMessage` sentence in the `teco` catalog entry) accurately reflects the landed
   `teco.md` text — the scope test, the `paused` ledger marking, the relay, and the `SendMessage`
   resume are all stated correctly and at a catalog-appropriate altitude (it correctly omits the
   `AskUserQuestion`-vs-report-as-subagent nuance, which belongs in `teco.md`'s own prose, not a
   one-clause catalog summary). No implementation drift found between what the plan proposed and
   what actually landed, on the one axis README.md claims (the described capability itself).

7. **`kaizen/history.md` accuracy:** the new dated entry's "What" section matches the actual diff
   line-for-line on all three edits (verified against the `git diff` directly, not against the
   entry's own claims); the "Verified" section only claims a read-through against AC-1..AC-5 and an
   inspection confirming no hook/frontmatter file was touched — both true and both things a
   read-through can actually establish — and explicitly flags the live dry-run as **not** performed,
   rather than fabricating one. No invented figures, no overclaimed verification.

8. **Security/perf:** agree with the brief's own assessment — this is prompt text with no execution
   path, no user input handling, no secrets, no loop-bound query; the checklist item doesn't apply
   in any material way here.

### What's solid (Pass 2 additions)

- **Every AC re-derives cleanly from the landed text**, independent of trusting Pass 1's own mapping
  table or `teco`'s U3-outcome self-check.
- **All three Pass-1 minor/nit findings are not just present but actually correct** on inspection —
  Finding 2's two-hop mechanism, Finding 3's scoping rationale, and Finding 4's column-format
  fidelity all hold up to the specific mechanical claims the brief asked me to check, not just to
  "was something added here."
- **`audit-team.sh` full PASS** is real corroborating evidence beyond a manual read-through — the
  team-coherence deterministic checks (roster, catalog, hook wiring, commit-authority documentation)
  all still pass after this change.
- **Scope discipline holds** — nothing outside the named files (plus the feature's own doc trio)
  changed anywhere in the repo.

### Open questions

None that block implementation. The one item worth a stakeholder's or `cobb`'s attention, not
because it blocks anything but because it's an evidentiary gap I couldn't close: Item 3 above — the
implementer's "§2.2 never had this problem" claim is unfalsifiable from the surviving artifact
(untracked file, no backup), so it stands as a claim `teco`/`cobb` made and I could not independently
confirm or refute, only note that the current file is consistent with it. Not a blocker: the
*landed* text is clean either way, which is what actually matters for this gate.
