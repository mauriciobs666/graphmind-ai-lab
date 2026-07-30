# Plan review — Web API coverage

> **Status:** archived · **Owner:** `analyst` · **Tracks:** K-036 (M3.5, per the plan's own header)

archived 2026-07-29 — K-036 delivered, M3.5 reached; see `docs/plans/web-api-coverage-coordination.md`.

- **Reviewer:** `analyst` · **Date:** 2026-07-28
- **Artifact reviewed:** `falkor-chat/docs/plans/web-api-coverage.md` (Status: active, Owner: `architect`)
- **Baseline:** `falkor-chat/docs/requirements/web-api-coverage.md` (FR-1..FR-14/AC-1..AC-6,
  committed scope FR-1..FR-10/AC-1..AC-6) and the live codebase at the current `main` tip
  (`server/falkorchat/{api,services,repository,executor,proof_defs,config,trigger}.py`,
  `docs/{DESIGN,QUERIES}.md`, `scripts/{verify_workflows,test_queries,seed_demo}.sh`,
  `web/{index.html,app.js}`).

## Scope & verdict

Static pre-implementation review: FR/AC coverage, Cypher correctness/cardinality, layering
consistency, the FR-9/AC-5 minimalist-default risk, the freshness mechanism, and the plan's own
flagged risks — checked against the real code, not a summary. I did not execute code or run the
test suite (no implementation exists yet to run); every factual claim below was checked by reading
the cited file/line directly.

**Verdict: needs changes.** One blocker (an internal contradiction between the plan's own UI
design and its own acceptance-verification instruction for AC-5) and two majors (one grounding
error in a design rationale, one under-specified test-execution consequence of an already-flagged
risk) should be resolved before build units land. Nothing here invalidates the plan's overall
shape — grounding is unusually strong (every code citation I checked, down to specific line
ranges, matched the real file), the FR/AC-to-unit mapping is complete, and the layering is
idiomatic. These are fixable without a redesign.

## Findings

### Blocker

**B1 — §3.3's design and §5.2's AC-5 verification instruction contradict each other.**

§3.3 explicitly designs several **always-visible** new elements: a "small header button
('Workflow defs')" (§3.3.1, "same visual weight as the existing 'Search' form"), a "small
'Participants (n)' toggle next to `#thread-heading`" (§3.3.4), and "a small badge near the header
`tenant` span... fetched once on load" (§3.3.5). None of these are conditionally rendered — they
are present the instant the page loads, for every user, whether or not they ever touch a workflow
surface. (Only the inline run cue, §3.3.2, is genuinely conditional — "appears **only when** the
thread-runs poll returns a non-empty list.")

§5.2 then instructs `qa-engineer`'s U10 pass to verify AC-5 by "loading the page fresh and
confirming **no new visual element is present until explicitly opened**." Read literally, U10
would fail against the plan's own §3.3 design: the defs button, the participants toggle, and the
readiness badge are all new visual elements present on a fresh, unopened load.

This isn't pedantry — it's exactly the FR-9/AC-5 creep risk this review was asked to check for.
AC-5's actual text is softer than §5.2's instruction ("no more crowded than today's," not "zero new
elements"), so §3.3's design is plausibly fine — but the plan as written gives `qa-engineer` a
verification bar its own design cannot pass, which will surface as a false U10 defect (or a
silent, undocumented reinterpretation) rather than a decision made now.

*Fix:* reconcile the two. Either (a) reword §5.2's AC-5 instruction to match what §3.3 actually
ships — e.g. "confirm the only always-visible additions are the header 'Workflow defs' button, the
participants toggle, and the readiness badge, and that none of them expand/render content until
interacted with" — or (b) if the intent really is zero new chrome by default, redesign the three
always-visible elements in §3.3 (e.g. fold the defs button into an existing header affordance,
make the readiness badge appear only when *not* ready). This is a five-minute fix to the plan
document, not a design rework, but it must be made before U3/U7/U8 (which build the always-visible
pieces) and before U10's acceptance script is written from §5.2's current wording.

### Major

**M1 — §2.3's rationale for "participants = parent channel roster" rests on a claim the code does
not support.**

§2.3 justifies the FR-8 resolution partly on: *"mentions resolve against members, and membership
is scoped at the channel, not derived from posting history."* This is not what the code does.
`resolve_member_kinds` (`server/falkorchat/repository.py:904-920`, QUERIES.md §2 "Member-kind
lookup") is:

```cypher
UNWIND $ids AS id
OPTIONAL MATCH (u:User  {userId:  id})
OPTIONAL MATCH (a:Agent {agentId: id})
RETURN id, CASE WHEN coalesce(u, a) IS NULL THEN null ELSE labels(coalesce(u, a))[0] END AS kind
```

There is no `MEMBER_OF` traversal or filter anywhere in this query, and `_validate_and_derive_role`
(`services.py:438-462`), the only caller, does not check channel membership either — it accepts any
id that resolves to *any* known `User`/`Agent` in the workspace. `MEMBER_OF` is currently
write-only decoration: `scripts/seed_demo.sh:23` says so explicitly ("mention validation ... looks
up `agentId`, not channel membership; `MEMBER_OF` is seeded for roster/scoping"), and a repo-wide
grep confirms `MEMBER_OF` is never read outside the (currently uncalled) "List channel members"
query.

So today, a user can successfully `@mention` **any** workspace member from **any** thread,
regardless of channel roster. The plan's participants surface (channel roster) will therefore show
a list that is **narrower** than "who I can actually mention" — the exact framing FR-8's user story
uses. The other two legs of §2.3's rationale (zero new schema; cheaper than walking `POSTED_BY`
history) stand fine on their own and may well still justify the same decision — but the plan should
not ship a factually wrong justification, and the resulting UX gap (participants list ≠
mentionable set) is worth a line acknowledging it as a known, accepted limitation rather than
silently discovered later.

*Fix:* correct §2.3's rationale (a) to state the actual constraint accurately (there currently is
*no* server-side scoping of who can be mentioned — the channel roster is a *reasonable proxy* for
"who's around," not a technically-derived "who I can mention" set), and add one sentence noting the
gap for the record.

**M2 — Risk #1's consequence for §5.2's U10 test session is not threaded through.**

§7 risk #1 correctly identifies that reaching a run parked on a *structured*-input step (needed for
FR-6/AC-2) requires pointing `FALKORCHAT_TRIGGER_DEF_KEY`/`_VERSION` at `access-request`/`v1`
instead of the default `triage`/`v1` — verified accurate against `config.py:83-84` (`TRIGGER_DEF_KEY`
/`TRIGGER_DEF_VERSION` are module-level constants read once from the environment at import time) and
`trigger.py:41-49` (`WorkflowTrigger` is constructed with one fixed `def_key`/`def_version` for the
whole process). The risk is correctly flagged as needing a decision before U9's acceptance check is
written, not something the plan should silently decide.

What's missing is the knock-on effect on §5.2: AC-1 needs the **plain-chat-reply** resume path,
which only `triage` exercises (per risk #1, `access-request`'s steps declare `fields`/`expects` and
`triage`'s don't — so `access-request` doesn't demo the "answer by plain reply" story AC-1
describes), while AC-2 needs the **structured-form** path, which only `access-request` exercises.
Because `TRIGGER_DEF_KEY` is one process-wide value, **U10 cannot verify both AC-1 and AC-2 in one
continuous browser session** without restarting the server with a different env var in between —
yet §5.2 lists AC-1 through AC-6 as a flat checklist with no such break called out. Whatever risk #1
resolves to (single operator-switched def, or "both chat-triggerable," which would need real
scoping work, not just an env flip), §5.2 should say explicitly how U10 handles the two-def
requirement — otherwise `qa-engineer` discovers this gap live, during the pass, which is exactly
what U0-level planning exists to prevent.

*Fix:* resolve risk #1 (with `teco`/stakeholder as the plan already says) and add one sentence to
§5.2 describing the resulting U10 session shape (one pass with a mid-pass restart, or two separate
passes).

### Minor

**m1 — The §3.1a label-scan fallback (an index on `WorkflowRun.startedAt`) is not verified to
actually change this query's plan.**

The proposed query has **no `WHERE` predicate on `startedAt`** — only `ORDER BY r.startedAt DESC`.
Neither `docs/QUERIES.md` nor `claude/graph-dba/falkordb-quirks.md` documents whether FalkorDB's
planner can use a range index to serve an `ORDER BY` with no accompanying range filter (as opposed
to using it to avoid a label scan when there *is* a `WHERE`, which is the pattern documented and
verified elsewhere, e.g. quirks KB lines ~140-149). It's plausible the fallback does nothing for
this exact query shape, in which case the "cheap, no query-shape change" framing in §3.1a would be
wrong and U1 would need to actually change the query (e.g. add a dummy `WHERE r.startedAt >= 0`
predicate, or reconsider the anchor entirely) rather than just add an index.

*Fix:* have U1's `GRAPH.PROFILE` verification explicitly test the fallback index's effect on *this*
query (not just diagnose the label scan), before treating it as a known-good escape hatch.

**m2 — The structured-input submit flow doesn't reuse the codebase's existing "poll immediately
after a write" idiom.**

`web/app.js:298` already does `await pollMessages();` right after a successful `postMessage()`, so
a just-posted chat message appears without waiting for the next scheduled tick. §3.3.3/U9's
structured-input submit (`POST /workflow-runs/{id}/input`) doesn't call for the equivalent — the
plan's own worst-case freshness math (§3.2: "≈3.0–3.3s, comfortably under the 5s bar") already
assumes waiting for the next poll tick. Reusing the existing idiom (an immediate re-poll of the run
right after a successful submit) is free, matches an established pattern rather than introducing a
gap between two otherwise-parallel flows, and gives AC-2 more margin.

**m3 — The FR-2 cue's "most relevant run" tie-break is real branching logic with only optional test
coverage.**

§3.3.2's rule (non-terminal status beats terminal, ties broken by most recent `startedAt`) is the
one piece of client-side logic in this plan that isn't pure rendering. §5.2 treats a pure-function
unit test for it as "a reasonable, low-cost addition — not a requirement," and U10 is a single
manual black-box pass unlikely to construct a multi-run, mixed-status thread by hand. Given there is
no JS test harness in this codebase at all today (§5.2's own framing), this is exactly the kind of
logic most likely to regress silently. Recommend promoting this one function to a required
dependency-free unit test rather than leaving it optional.

### Nit

**n1 — Stale cross-reference.** U9 (Wave 4, §4) says "(see §6 risk on how such a run is reached in
the demo environment)" — the document's risks live in `## 7. Risks & open questions`, not §6.

## What's solid

- **Grounding is unusually strong.** Every specific code citation I independently checked matched
  exactly: the `_fail_with_note`/`ctx.error` claim (`executor.py:371,411,428,804-815` — confirmed
  every fail path routes through it); `TRIGGER_DEF_KEY`/`TRIGGER_DEF_VERSION` as a single
  startup-read env knob (`config.py:83-84`); `ACCESS_REQUEST_DEF`'s `submit`/`approval` steps
  declaring `config.fields`/`expects` vs. `triage`'s not doing so; the `verify_workflows.sh`
  `DEFS`/`ABSENT`/failure-string shapes the plan proposes reusing verbatim; the "List channel
  members" query and its exact `test_queries.sh:429-432` line citation; the K-036 numbering (K-035
  is indeed the highest filed id in `BACKLOG.md`). This level of care is exactly what an isolated
  implementer needs to trust the plan without re-deriving it.
- **The §12.9-vs-§3.1a reasoning is not circular.** §12.9's "Decision" note (QUERIES.md
  §12.9) rejects the `TRIGGERED_BY` traversal specifically for "find the currently-waiting run"
  because the `waitingThreadId` denorm is simpler and self-contained for that one case. The
  denorm only reflects *live park state*, never history, so it structurally cannot answer "every
  run this thread has ever had" — the plan's §2.2 claim that the traversal isn't an alternative
  there but *is* the only correct shape here is verified correct, not a reuse of rejected
  reasoning in disguise.
- **Layering and idiom reuse are correct.** New routes reuse `ThreadNotFoundError`/404 exactly as
  `_validate_and_derive_role` does; no new error classes, no new `response_model` pattern (matches
  the documented K-031-only convention, `api.py:250-252`); repository/service/route split matches
  DESIGN §14.2/§14.6 exactly; the three-layer test pattern (repository integration / service unit
  with fake repo / API contract) is the one already established by `test_repository.py` /
  `test_services.py` / `test_api.py`.
- **FR/AC → unit traceability is complete.** Every committed FR (FR-1..FR-10) and AC (AC-1..AC-6)
  maps to at least one build unit, and the mappings hold up against what each unit actually builds.
- **The freshness design is proportionate.** Polling on the existing 3000ms cadence, explicitly
  declining to pull K-018 forward, is consistent with DESIGN §14.1's own "real-time deferred to
  M2.5" framing and the requirements doc's own note that FR-4 is a freshness bar, not a transport
  choice.
- **Risk framing (§7) is mostly well-calibrated** — the BACKLOG entry note, the `verify_workflows.sh`
  dedup call, and the "no visual diagram" flag are all correctly scoped as should-do/non-blocking,
  and risk #1 itself is the right thing to flag (see M2 above for what's missing from its
  follow-through).

## Open questions

- Risk #1's resolution (which def reaches FR-6/AC-2, and how) is a stakeholder/`teco` decision the
  plan correctly declines to make unilaterally — this review doesn't resolve it either, but flags
  (M2) that whatever is decided needs to also settle U10's session shape.
- Whether AC-5's bar is "zero new elements" or "no more crowded" is ultimately a product call, not
  a technical one — B1 can be resolved either way, but someone (architect, or `tico` if the
  ambiguity is judged to reach back into the requirements doc) needs to pick one and make §3.3 and
  §5.2 agree.

## Pass 2 — 2026-07-28

**Artifact reviewed:** `falkor-chat/docs/plans/web-api-coverage.md`, `Version: v2` (2026-07-28) —
the revision made after this review's Pass 1 verdict, per the plan's own §8 "Review dispositions"
table. **Baseline:** Pass 1's findings above, re-verified against the live codebase
(`server/falkorchat/{repository,services,config,trigger}.py`, `scripts/seed_demo.sh`,
`web/app.js`) and `falkor-chat/docs/requirements/web-api-coverage.md` directly — not the plan's or
the disposition table's paraphrase of either.

**Verdict: needs changes.** One new blocker. It is not a new idea — it is B1 recurring, in two
places the v2 revision didn't touch when it fixed the two it did.

### New Blocker

**B2 — B1's contradiction was fixed in §3.3/§5.2 but left standing, verbatim in spirit, in three
other places the same document relies on: §3.3's own opening sentence, and U3's and U7's build-unit
done-conditions in §4.**

The v2 fix is real and correctly placed: the new "AC-5 reading, decided here" paragraph
(`docs/plans/web-api-coverage.md:299-313`) and the rewritten §5.2 AC-5 bullet
(`docs/plans/web-api-coverage.md:548-554`) agree with each other, and the reading itself is
defensible against AC-5's actual requirements-doc text (`docs/requirements/web-api-coverage.md:119-120`:
"a chat page whose default layout is no more crowded than today's" — not "zero new elements"; the
plan quotes this accurately). Checked directly, not trusted.

But the disposition table's claim that this "resolved" B1 is only true for the two spots it edited.
Three other passages in the same document still assert the reading §3.3/§5.2 just explicitly
rejected:

1. **§3.3's own opening sentence, two lines above the fix.** Line 296: *"Every new surface is
   **additive and collapsed-by-default** — the 3-column grid ... is untouched."* Line 299
   immediately follows with: *"Three of the five surfaces below ... are **trigger affordances that
   are themselves always visible**."* These two sentences are three lines apart in the same
   paragraph and directly contradict each other — line 296 was never updated when the v2 carve-out
   was inserted right after it.

2. **U3's done-condition (`§4`, lines 412-416):** *"header button opens/closes the overlay; list
   renders; selecting a def renders its steps + transitions; manual check against a running server
   with `triage`/`access-request` seeded; no change to the default (unopened) page."* Read
   literally, this requires the "Workflow defs" header button itself to be absent from the
   unopened page — exactly the reading §3.3.1 explicitly designs against (*"a small header button
   ... same visual weight as the existing 'Search' form"*, always visible per the v2 note).

3. **U7's done-condition (`§4`, lines 447-450):** *"toggle collapsed by default; expanding shows
   both member kinds distinguishably ... collapses again on thread switch; default page unchanged
   when never opened (AC-5)."* Same problem, and worse: it explicitly tags "(AC-5)" onto the
   stricter, rejected reading — asserting the acceptance criterion itself demands an unchanged
   page, which is precisely the claim §3.3's v2 note spends a paragraph refuting.

This is not pedantry restated: if `frontend-engineer` builds U3/U7 to their literal "Done" text
(the section they'll actually work from) rather than re-deriving intent from §3.3's prose four
sections earlier, the "Workflow defs" button and the "Participants" toggle get hidden until some
undesigned trigger exists — which doesn't just relitigate AC-5, it **breaks FR-1** ("the page lists
the workflow definitions ... and lets the user view a chosen def's shape *before* any run is
started" — impossible if there is no discoverable, always-present way to open the defs viewer) and
narrows FR-8/AC-4 the same way. U8 (the readiness banner unit) has no such leftover phrasing and is
internally consistent with §3.3.

*Fix:* sweep the same v2 edit through the two places it missed — reword line 296 to something like
"Every new surface's *content* is collapsed-by-default; three surfaces' *trigger* affordances are
themselves always visible, minimal, and matched to existing header elements (see the AC-5 reading
below)," and reword U3's/U7's done-conditions to match §5.2's already-correct AC-5 bullet (e.g. "the
header button/toggle itself is always present, matched in weight to \[Search / an existing header
element\]; no *content* renders or fetches until clicked/expanded"). This is the same five-minute,
no-redesign class of fix Pass 1's B1 required — it just needs to be applied everywhere the old
reading was written down, not only where the disposition table checked.

### Verification of Pass 1's other dispositions

- **M1 — confirmed correctly and honestly fixed.** Read `repository.py:904-920`,
  `services.py:438-462`, and `scripts/seed_demo.sh:22-23` directly: `resolve_member_kinds` is a
  plain id lookup with no `MEMBER_OF` traversal, `_validate_and_derive_role` is its only caller and
  applies no channel scoping, and the seed script's own comment says exactly what §2.3 v2 now
  quotes. The new "Known, accepted gap" paragraph (§2.3) states the resulting UX gap plainly
  (participants list is narrower than the truly-mentionable set) rather than hand-waving it. No
  residual false claim found anywhere else in the document (`§3.1b`'s participants query and its
  route description don't repeat the old rationale).
- **M2 — resolved, and threaded through correctly.** `config.py:83-84` and `trigger.py:36-51`
  confirm the single-process-wide-env-var constraint exactly as both Pass 1 and v2 describe it.
  §5.2's Pass A / Pass B structure correctly assigns AC-1/AC-4/AC-5/AC-6(+optional AC-3) to Pass A
  (default `triage` config) and AC-2(+AC-3 if not covered) to Pass B (`access-request` config,
  reached by a server restart with the env vars swapped), matching §7 risk #1's resolution
  ("temporary `TRIGGER_DEF_KEY`/`_VERSION` swap, restart between passes, no FR-13 pulled forward")
  verbatim. §6's traceability table rows for AC-1/AC-2/AC-3 were updated to cite the correct pass
  and are consistent with §5.2. One observation, not a defect: the "stakeholder decision (relayed
  via `teco`)" in §7 risk #1 has no separate coordination-doc paper trail in `docs/plans/` — there
  is no `web-api-coverage-coordination.md` — so it's independently unverifiable from this document
  alone; that's expected for a review-driven revision pass at this stage (no `teco`-run build
  sequence has started yet) and not something to hold the plan on.
- **m1 (index/ORDER BY caveat) — confirmed present**, in §3.1a's caveat paragraph, U1's done
  condition (§4), and cross-referenced from §7 risk #2 — all three agree with each other and with
  Pass 1's fix request.
- **m2 (immediate re-poll after structured-input submit) — confirmed present**, in §3.3.3's FR-6
  bullet and U9's done condition (§4); both cite the same `app.js:298` `pollMessages()` idiom Pass
  1 pointed at.
- **m3 (required pure-function test for the run-cue tie-break) — confirmed present and correctly
  promoted from optional to required**, in §5.2's new "Exception ... required not optional"
  paragraph and U6's done condition (§4), which now states it explicitly as part of U6's done
  condition rather than left to `frontend-engineer`'s discretion.
- **n1 (stale §6→§7 cross-reference) — confirmed fixed.** U9's done condition now reads "see §7
  risk #1 (now resolved)".

### What's solid (unchanged from Pass 1, plus the above)

Everything Pass 1 found solid still holds — grounding remains unusually strong (every new v2
citation checked against the real files matched exactly), the FR/AC traceability table is complete
and, per the spot-check above, consistent with the sections it summarizes except for the B2 gap,
and four of the five Pass 1 findings (M1, M2, m1, m2, m3, n1 — six of seven) are genuinely and
cleanly resolved on the first revision pass, which is a good sign for how carefully the rest of the
document was edited.

### Open questions

- Same as Pass 1: whether AC-5's bar is "zero new elements" or "no more crowded" was a product call
  someone needed to make, and v2 makes it (in §3.3/§5.2) — that choice itself is not being
  reopened here. B2 is purely about the document not yet saying the same thing everywhere it
  matters.

## Pass 3 — 2026-07-28

**Artifact reviewed:** `falkor-chat/docs/plans/web-api-coverage.md`, `Version: v3` (2026-07-28) —
the narrow revision made after Pass 2's B2 finding. **Baseline:** Pass 2's B2 finding and its three
named spots, re-verified by reading §1–§8 of v3 in full (not just the three edited passages) plus a
full-document grep for the same contradiction-pattern vocabulary Pass 2 used, cross-checked hit by
hit rather than trusted from the revision note's self-report. Also re-spot-checked every section
Pass 2 already confirmed resolved (§2.3, §3.1a, §5.2, §7 risk #1, §8 rows for M1/M2/m1/m2/m3/n1) to
confirm this narrow pass didn't disturb them.

**Verdict: approve.** B2 is genuinely closed. No new findings. This closes out the review — the
plan is ready to move to implementation.

### Verification of the B2 fix

**All three named spots read correctly now, and agree with each other and with §3.3's/§5.2's
already-correct v2 language:**

1. **§3.3's opening sentence (line 300).** Now: *"Every new surface's *content* is additive and
   collapsed-by-default; three surfaces' *trigger* affordances are themselves always visible —
   small, minimal, and matched in weight to existing header elements (see the AC-5 reading just
   below)."* This states the content/trigger distinction directly, three lines before the existing
   "AC-5 reading, decided here" note (lines 305–319) — the two no longer contradict; they're the
   same claim stated twice, once as a summary and once in full.

2. **U3's done-condition (§4, lines 421–424).** Now ends: *"...the header button itself is always
   present on page load (matched in visual weight to the existing 'Search' form, per §3.3's AC-5
   reading) — no overlay *content* renders or fetches until the button is clicked."* This is the
   opposite of the old "no change to the default (unopened) page" — it now explicitly requires the
   button to be present unopened, matching §3.3.1's design exactly.

3. **U7's done-condition (§4, lines 456–459).** Now ends: *"...the toggle itself is always present
   next to `#thread-heading` (small, per §3.3's AC-5 reading) — no participant *content* renders or
   fetches until expanded."* The old literal "(AC-5)" tag asserting an unchanged page is gone,
   replaced by a citation of the correct reading. Matches §3.3.4's design.

Both done-conditions, §3.3's opening sentence, §3.3's "AC-5 reading" note, and §5.2's AC-5
verification bullet now assert one single reading throughout: three trigger affordances
(defs button, participants toggle, readiness badge) are always visible and minimal; their *content*
is what's gated on interaction; the inline run cue and run detail panel remain genuinely
zero-footprint until a run exists / is opened. No internal disagreement found.

### No fourth spot

Ran the same grep pattern Pass 2 used
(`collapsed|unchanged|unopened|default page|no change to|no new visual|until explicitly opened|always
visible|hidden by default`) against the full v3 document and read every hit in context (13 matches,
listed below with disposition):

- Line 9 (revision-note's own description of B2's old wording) — historical, expected, not a live
  claim.
- Lines 143/148 (§2.5, "Collapsed-overlay idiom" for the *existing* `#results` search-results panel,
  and "no new visual language" for badge tokens) — both describe pre-existing UI idioms being reused
  for *content* panels (the defs overlay, run detail panel), not a claim about trigger visibility.
  Consistent with the AC-5 reading, not a residual contradiction.
- Lines 300/301/308/313 — the fixed opening sentence and the "AC-5 reading" note itself (see above).
- Line 323 (§3.3.1: the defs overlay "styled like `#results`... hidden by default") — this describes
  the overlay *panel content*, immediately after stating the header button is always visible in the
  same bullet. Consistent, not contradictory — content hidden, trigger visible, exactly the shipped
  design.
- Line 359 (§3.3.4: "a small 'Participants (n)' toggle... collapsed by default") — "collapsed" here
  describes the toggle's own closed/open *state* on load (it starts unexpanded, showing a count, not
  the roster), not the toggle's visibility. The toggle itself is stated as always-present at U7's
  done-condition (line 457) and in §3.3.4's own sentence. Same usage as line 456 below — legitimate
  UI-state language, distinct from the rejected "page chrome absent" reading Pass 1/2 were checking
  for.
- Line 416 (U2's done-condition: "`verify_workflows.sh` still passes unchanged against a live
  server") — about a shell script's behavior, unrelated to AC-5/page chrome.
- Line 456 (U7's done-condition: "toggle collapsed by default; expanding shows...") — same "toggle
  state" usage as line 359, immediately followed three clauses later in the same sentence by "the
  toggle itself is always present" (the fixed clause). Internally consistent within one sentence.
- Lines 657/669 (§8 disposition-table rows for B1 and B2) — both are the historical record of what
  the *old*, rejected wording said, which is exactly what a disposition table is supposed to
  preserve. Not live claims about the current design.

No hit asserts, in the document's own present-tense design/done-condition voice, that the defs
button, participants toggle, or readiness badge is itself absent/hidden/unchanged on an unopened
page. The architect's self-reported sweep holds up under independent re-check.

### Confirmation nothing else regressed

Full re-read of §1–§8 against what Pass 2 explicitly verified: §2.3's "Known, accepted gap"
paragraph (M1's fix), §3.1a's index/`ORDER BY` caveat (m1), §3.3.3's immediate re-poll clause (m2),
§5.2's required pure-function test for the run-cue tie-break (m3) and its two-pass U10 session shape
(M2), §7 risk #1's resolved status, and §8's disposition rows for B1/M1/M2/m1/m2/m3/n1 are all
byte-for-byte the same content Pass 2 read and confirmed — only the three named spots, the version
header/revision note, and the new B2 disposition row changed. This was a genuinely narrow,
correctly-scoped revision.

### What's solid (unchanged from Pass 1/2, plus the above)

Same as Pass 2 — grounding remains unusually strong, FR/AC traceability is complete, layering and
idiom reuse are correct, and the document now states one single, coherent reading of AC-5
everywhere it matters: §3.3's opening sentence, its "AC-5 reading" note, §5.2's verification bullet,
§6's traceability row, and U3/U6/U7/U8's done-conditions all agree.

### Open questions

None outstanding. The one open question Pass 1/2 carried (whether AC-5's bar is "zero new elements"
or "no more crowded") was a product call the architect made explicitly in v2 (§3.3's "AC-5 reading,
decided here" note) and is not being reopened by this pass — B2 was purely a consistency defect in
how that already-made decision was written down, and it is now written down consistently
everywhere.
