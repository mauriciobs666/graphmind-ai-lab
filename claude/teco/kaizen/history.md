# Kaizen — Change History: teco

> Dated log of actual changes to the `teco` agent. Most recent first.

## 2026-09-07 — Kaizen distillation, chunk D (the final fourteen 2026-09-07 entries)

- **What:** U18b of `claude/docs/plans/kaizen-distillation2-coordination.md` — `cobb` distilled the
  fourteen remaining `teco` `:KaizenEntry` nodes in `kaizen_team` (all dated 2026-09-07; no date
  filter applied, the whole remainder was chunk D). Result: **9 promoted as 12 statements**
  (11 in-place sharpenings of existing `teco.md` sentences, 0 new `teco.md` bullets, 1 into
  `skills/agent-standards/claude-code.md` merging two entries), **1 discarded**, **4 routed out of
  `teco.md` entirely via `MENTIONS`**, 0 kept open, **5 `MENTIONS` tags** added. Every entry was
  re-derived from primary sources — transcripts, the delivered guard source, the plan and
  coordination documents, and one script I wrote myself. **All fourteen facts held**; one entry's
  *causal mechanism* was falsified by its own sibling, and one chunk-C prediction of coverage was
  wrong.
- **Shape of this pass: no new bullets at all, and four entries kept out of the prompt on the
  "most sessions" bar.** Chunk C concluded that sharpening in place had hit its ceiling. Chunk D
  tested the next lever — **refusing the prompt as a destination** — and it worked for four of
  fourteen: the three static-guard-authoring entries and the AST-equality done-condition are true
  and useful but are not coordination doctrine, so they were `MENTIONS`-tagged to the agents whose
  discipline they belong to instead of costing always-loaded words. `teco.md` still grew, because
  the nine that *are* doctrine are doctrine.
- **Re-derivation that changed a disposition:**
  - `4e91c7a3` (an explicit exact-command allow rule is a materially stronger signal than a broad
    one) — **mechanism not established, and falsified as an inference by its own sibling.** The two
    permission entries describe the *same* incident. Re-derived independently from the session
    transcript (`~/.claude/projects/<proj>/a2d1489d-….jsonl`): `./scripts/seed_workflows.sh` was
    refused at `02:55:22` and the identical command **ran at `16:45:00` the same session with no
    settings change**; the six exact-command rules were added at ~`16:53` and `seed_salesperson.sh`
    ran at `16:53:45`. So the only evidence for the rule helping is one post-hoc success in a
    system already demonstrated to flip without any rule change — n=1, confounded. What *is*
    independently verified and was promoted instead: the tracked `.claude/settings.json` has
    carried a blanket `"Bash"` allow since `c994442` (2026-08-29, committed expressly to "end
    per-command prompt whack-a-mole") — **nine days before** these refusals — so a bare tool-name
    allow demonstrably does not resolve a shell-script invocation at the documented
    "explicit settings rules resolve immediately" step. Also corrected: `settings.local.json` is
    untracked here only because the maintainer's **global** ignore file matches it
    (`git check-ignore -v` → `~/.config/git/ignore`); the repo's own `.gitignore` has no entry.
  - `f3c81a92` — **chunk C predicted this was already covered; it is not.** The step-3 bullet chunk
    C generalized (*"A reviewer's suggested fix is a finding to judge"*) is about the fix's
    **content**; this entry is about its implicit **routing**, which is teco's job alone and which
    no existing sentence touched. Re-derived: coordination row U36 records the `data-scientist`
    (routed there against the gate's suggestion of `architect`) refusing the exemption, **overruling
    teco's own stated reason**, and catching a defect in its own note the gate never saw — §11.2.1's
    1626 divergences measured against the percent spelling `math.ceil(pct / 100 * X)`, where the
    code's numerator-first form `math.ceil(level.numerator * X / level.denominator)` diverges **0**
    times, over levels `n/1000` for `n = 1…999` and `X ≤ 3000` (`small-model-benchmarking-ml.md`
    §11.2.1 table; level-first gives 755). The entry's figures are exact.
- **Promoted → `claude/teco/teco.md` (all in-place word replacements, no new bullets):**
  - `e58c30b7` → step 2, **widening** *"The ledger cites; it does not restate"* to *"The
    coordination doc cites…"* — a row **and your prose around it**; your own summary of a delegate's
    ruling is a secondary source, with the drift tell (a summary naming a code expression where the
    ruling was about placement, sequencing or ownership). Not covered by the two adjacent rules
    despite both predating the incident (`never paraphrase a plan into a brief`, 2026-07-11;
    `the ledger cites`, 2026-08-25): the compression lived in narrative prose, not in the ledger
    table. Re-derived: the v1.22 ledger row rules *"the trigger runs inside the turn-queue
    worker"* (placement); the coordination doc at `:1822` records it as
    `self._services.start_workflow_run` (a spelling); the plan's own S9 row says
    `trigger.maybe_trigger` → `services.start_workflow_run`, with `trigger.py:82` outside all four
    walked scopes — so the paraphrased done-condition was unmeetable.
  - `d471a9c8` (1) → step 3, extending Dispatch axis (c) with the **third option** it lacked: the
    unit of collision is the claim, not the file, so split the *claim* rather than the schedule —
    measurable half to the unit that can measure it, judgment half to the unit that owns it, seam
    and the other's ownership in both briefs — and the units stay parallel. Re-derived verbatim
    from `salesperson-ui-coordination.md` §"Splitting one sentence between two parallel units, at a
    seam": S8f (`coder`) owned the measured fact, U30 (`architect`) the mapping.
  - `f3c81a92` → step 3, closing the reviewer's-suggested-fix bullet (whose third shape was
    compressed to part-pay for it): a gate that proposes a fix has implicitly routed it, and that
    routing is its least trustworthy opinion — re-derive the owner from the artifact the fix would
    change, and brief an owner chosen over the gate's with your reasoning open plus an instruction
    to overrule it.
  - `b7d24e10` → **two halves, two places.** Step 3's mutation-test bullet gains the brief-side
    mitigation (restore **by copy, after each mutation, never batched**); step 4's transient-failure
    bullet gains the detection (**on abnormal termination, diff the tree against the last commit
    first**, ahead of checking whether the deliverable was written). Re-derived: the U40 ledger row
    records the `analyst` `ad216ed80e4e38da2` *"killed by a session rate limit, wrote nothing, left
    no mutation in the tree — verified on disk"* — a near miss, not a hit, which is why the tree
    check is worth a rule. "Never restore via git" was already teco doctrine and was not restated.
  - `7c4e08b1` + `c9a4e60f` (a) → **one doctrine, two moments.** Step 5's rebuttal bullet gains
    *corroboration needs independence of **method**, not a second agent* — two agents running the
    same grep are one check, and a grep hit proves the value exists, never that the entity you named
    owns it (verify an attribution by reading the enclosing definition). Step 4's now-redundant
    trailing clause *"and independent agreement is stronger evidence than either agent alone"* was
    **deleted** to part-pay, because as written it was over-broad in exactly the way this entry
    corrects. "Pause vs. proceed" gains the other moment: put the same question to a qualified
    delegate in parallel with the stakeholder, unfiltered by yours, forming your own view first.
    Re-derived: `api.py:294`'s `Query(10, ge=1, le=50)` belongs to `list_workflow_runs_for_thread`
    (def at `:292`) — `list_thread_participants` (def at `:302`) takes **no limit parameter at
    all**; and `salesperson-ui-coordination.md` §"U31 — an independent read that converged with the
    stakeholder's" records both halves verbatim.
  - `e4d10b73` → step 5, sharpening *"Send the re-check to the same reviewer the same way"*: where
    an **implementer** overruled a finding, the reviewer confirms or refutes by **executing against
    the tree, not re-arguing**, its own recommended fix included. The existing sentence routed the
    re-check but said nothing about what to ask for. Re-derived: U41 (`coder`) *"overruled the
    gate's own recommended fix"*; U45 resumed the Pass 5 author `ad216ed80e4e38da2`, which *"ran a
    mutation applying its own recommended `_ABSENT` design and the suite refused it (2 failed)"* and
    recorded two of its own Pass 5 claims as errors.
  - `c9a4e60f` (b) → Guardrails, **rewriting** *"Neither gate is execution"* to *"Neither gate is
    execution, and neither is a static-analysis artifact"* — a static trace, AST reader, lint rule or
    grep guard testifies only about **text**, and is never evidence about placement, threading or
    ordering. The redundant *"even twice independently"* went with the rewrite. Corroborated twice
    over: U31's *"`_service_layer_reach` reads **source, not threads**"* and the plan's own S9 row
    (*"It is **not** evidence of placement"*).
  - `d471a9c8` (2) + `b2d64f19` (coordinator half) → Guardrails, **replacing** the stopping-signal
    sentence's vaguer *"ask the reviewer to rule on that directly"*: ask for a **falsifiable
    stopping condition** rather than another fix, promote it *above* the individual findings as the
    implementer's done-condition, and say in advance what happens if it fails — plus, where the
    recurring defect is a guard claiming more reach than its mechanism implements, put **both**
    closures to the implementer (widen the mechanism, or narrow the claim), because a team that has
    widened repeatedly will not propose retreating on its own. Re-derived: the coordination doc's
    §"Twelve instances, and the first stated convergence test" (`:2066-2072`), Pass 15 briefed with
    it *"as the bar"* (`:2157`), and §"S8f — the first unit to catch an instance of the defect
    *itself*" (`:2255-2261`) — the probe caught the thirteenth instance pre-gate. The two-closures
    half is in the delivered guard's own docstring: *"the decision to narrow the sentence instead of
    widening the reader"*.
- **Promoted → `skills/agent-standards/claude-code.md`** — `4e91c7a3` + `b3d70e94` merged into one
  three-bullet entry after the existing `defaultMode` resolution: a classifier denial is an **event,
  not a state** (and a settings `allow` rule does not prevent one); the exact-rule lever is
  **not established**; and the placement rule with its verified ignore-scope caveat. Routed there
  rather than to `teco.md` because it is harness mechanics every agent needs, not coordination
  doctrine — and because `teco.md`'s existing guardrails already forbid the two behaviors the
  entries prescribe against (converting a denial into a standing blocker; retrying around it).
  `b3d70e94` was fully re-derived from the transcript and is **true**.
- **Routed out of `teco.md` via `MENTIONS` (4 entries, 5 edges)** — true, verified, and not
  coordination doctrine; each survives with its `MENTIONS` edge after its `PRODUCED` edge is
  resolved, and surfaces in the tagged agent's own distillation pass:
  - `f6b820ae` → `tdd-engineer` — a **coverage probe** and a mutation test answer different
    questions, and only the probe catches the recurring defect; derive the probe's enumeration from
    the language itself and ship it as a test. Verified in the delivered source: eight `ast` node
    types walked as a named constant with the other 19 grammar nodes excluded each with a reason,
    and `test_the_alias_reader_covers_every_binding_form_the_grammar_has` *"takes its enumeration
    from `ast` rather than from a list here"*.
  - `a8f3c521` → `tdd-engineer` — the **two axes** (target axis finishable and derivable from
    `ast`; value axis is alias analysis and is not), so a probe varying only the target axis
    certifies nothing. Verified verbatim in the guard's docstring, numbers included:
    `me = self` then `me._services.start_workflow_run(...)` survives at **185 passed**, where
    `svc = self._services` at the same point is **1 failed / 184**.
  - `b2d64f19` → `tdd-engineer` **and** `analyst` — the gate question is not *"does the guard fire
    on my reproduction?"* but *"what is the smallest edit to production code that satisfies the
    docstring and survives the body?"*. Its coordinator-facing half was promoted (above); the
    review-authoring half belongs to the reviewer.
  - `f3a91c47` → `analyst` — the AST-equality done-condition for a prose-only unit over code.
    **True and exactly reproducible** — my own script over `b720bd3^`→`b720bd3` returns
    stripped-AST equal with identical docstring-owner sets on both files (27 and 129 owners), and
    full-AST equal on `storefront_api.py`. Dropped from `teco.md` on the §5 "most sessions" bar
    after first being drafted into step 2: it is a narrow unit type, and `analyst` is the agent that
    gates a prose-only change.
- **Discarded (1):**
  - `b2d7f309` (file-disjoint units that still conflict) — **already covered, a fortiori**, by the
    step-3 Dispatch axis (c) chunk C wrote for `5d1cca84` and deliberately worded to cover it:
    *"One unit decides a fact the other's deliverable must encode … Read both done-conditions, not
    just their file lists."* Re-derived and true — the coordination doc's own §"S9a is not released
    after all" says *"They conflict through a fact, not through a diff … A file-overlap check does
    not catch that; only reading both done-conditions does"* — but the existing rule covers both
    directions (the brief asserting a fact, and the deliverable encoding one) where the entry covers
    only the second. Its residue, *"brief X to update the marker in the same change"*, is the
    existing **Docs ride in the brief** rule plus serialization, not a third option.
- **Budget:** `teco.md` **6,909 → 7,318 w** (+409), lines past the 700-char smell **21 → 22**.
  `skills/agent-standards/claude-code.md` **8,760 → 9,185 w**, lines past the smell **3 → 3**
  (wrapped at ~100 cols). `claude/AGENTS.md` **untouched at 2,434 w** — every rule here is
  teco-specific or harness-specific, and the file has ~60 w of headroom against its own bar.
  A first draft of this pass came in at **+548 w**; a compression pass over my own additions plus
  dropping `f3a91c47` recovered 139 w. **K-016 stays blocking and the case for it is now stronger,
  not weaker:** two consecutive chunks have grown the file by ~930 w combined while promoting only
  in-place, and the ceiling finding from chunk C is confirmed — the only lever that actually saved
  words this pass was *refusing the destination*, which works for 4 entries in 14 and not for the
  doctrine.
- **Why:** U18b of the team-wide distillation pass; keeps `kaizen_team` as working memory for
  unreviewed capture only. This clears the last `teco`-produced entry.
- **Plan items:** K-016 (updated with this pass's ledger and the refuse-the-destination finding).

## 2026-09-07 — Kaizen distillation, chunk C (the thirteen 2026-09-03 / 2026-09-06 entries)

- **What:** U18 of `claude/docs/plans/kaizen-distillation2-coordination.md` — `cobb` distilled the
  thirteen `teco` `:KaizenEntry` nodes dated 2026-09-03 and 2026-09-06 in `kaizen_team`. Result:
  **11 promoted as 6 statements** (5 in-place sharpenings of existing `teco.md` sentences, 1 new
  `teco.md` bullet, 1 into `skills/agent-standards/claude-code.md`), **2 discarded**, 0 kept open,
  **3 `MENTIONS` tags** added. Every entry was re-derived from primary sources; one entry's stated
  mechanism was **materially wrong** and one other's was **sharper than written**.
- **Deliberate shape of this pass: no new doctrine bullets.** Eleven of thirteen entries were
  coordination doctrine aimed at a file already at 6,388 w with 17 lines past the 700-char smell.
  Siblings were merged into one statement each *before* promotion, and each merged statement
  **replaced words in an existing sentence** rather than being appended beside it. The one new
  bullet (`agentId` resolution scope) exists because it is a hard limit on a mechanism the ledger
  already relies on, and it paid for part of itself by letting the *"Close the loop on the same
  delegate"* bullet shed its now-duplicated `SendMessage`-first clause (**1,436 → 1,312 ch**).
- **Corrections to rules `teco.md` already gave — these outranked every new-bullet candidate:**
  - `c4f8a3d1-7b26-4e95-8a13-6d09f2b5e847` → Guardrails, **correcting** *"To read a baseline use
    `git show <ref>:<path>`"* to require an explicit sha and forbid `HEAD`. **Verdict: the prompt's
    advice was not wrong, it was under-specified in exactly the way that produced the incident** —
    `HEAD` is a legal `<ref>`, and teco's own integrator commit grant is what moves it. Re-derived
    in a throwaway repo in the session scratchpad (never this tree): `git show HEAD:plan.md` at the
    briefing commit returned `Version: v1.7`; two commits later the identical command returned
    `v1.8`, while `git show <sha>:plan.md` still returned `v1.7`. Silent on both sides — the
    delegate believes it followed the brief and the coordinator believes it pinned a version.
  - `1ad53bad-b198-415d-8221-5773d2a95da1` + `7c1e2b04-9a3d-4f61-8e57-2b6a0d94f1c3` (merged) →
    step 4, **reversing the ordering** of the transient-failure bullet, which said *"Re-dispatch
    with a state-recovery brief"* as the first move. The delegate's edits **and** its conversation
    context survive, so `SendMessage` to the recorded `agentId` comes first and the respawn is the
    fallback. This also **resolved a standing contradiction**: step 5 already said to attempt the
    `SendMessage` first, so the two bullets gave opposite first moves. Second half became the new
    bullet: an `agentId` does not survive a session reboot, so a checkpoint written *because* the
    session is dying is exactly the one whose ids will not resolve — each in-flight unit needs a
    cold-start fallback. **Re-derivation changed the mechanism:** the entry says the id does not
    "survive", implying the record is gone. It is not — subagent transcripts live under the
    **parent session's** directory (`~/.claude/projects/<proj>/<parent-session-id>/subagents/
    agent-<agentId>.jsonl`), and the specific reported-unresolvable id `a213382761bc926ec` is still
    on disk under session `0b03bc60-…`. What does not survive is the **resolution scope**: a new
    session has a new id and its own empty `subagents/` index. That corrected mechanism, not the
    entry's, is what was promoted (to `claude-code.md`).
- **Promoted → `claude/teco/teco.md` (in-place sharpenings):**
  - `b8d47f05-1e39-4c72-8a06-3fd2159c7ea4` + `d51c8a73-6f20-4e94-b1a8-7c03e6f9b2d1` (merged) →
    step 3, *"Brief contents"*. One root — **your wording is the delegate's specification and no
    gate reads it** — already stated abstractly there; the two entries supply its operational
    form: put a blocker as *is this premise true, and then what follows* rather than *weigh A
    against B*, and state a failure mode in its likeliest form, not its most vivid. Re-derived:
    `docs/plans/salesperson-ui-coordination.md:126` records the S7 gate **"Dissolved Ruling 1's
    blocker instead of weighing it"** — the reviewer checked the premise and it was false.
  - `b7b3bc96-21fa-4d4c-a2ba-82619bb5d3ad` + `3f8b17d2-6c40-4e93-b1a7-5d29e08c6a44` (merged) →
    step 3, **generalizing** the QA-suggested-fix bullet from `qa-engineer` to *any* reviewer:
    a reviewer's suggested fix is a finding to judge, not an instruction to apply, in three shapes
    (a reproduction proves one path; *"checked, not guessed"* names the method not the scope;
    folding a guard into the step it guards deletes it while keeping its name). Both re-derived
    against `docs/reviews/salesperson-ui-impl.md`: Pass 12's own *"Fix, checked not guessed"* at
    `:3106`, and the review's later independent verdict at `:3542-3547` that S8d2 was right to
    decline the recommendation it was handed. The fold-into-S9 half is corroborated by the
    coordination doc's `:1543-1561` (teco split it) and by `:2119` — S9's done-condition ended up
    **unmeetable anyway** and needed remedial unit U31 (`:165`) to replace it.
  - `5d1cca84-0036-4a28-aa07-80db6684681d` → step 3, *Dispatch*, **restructured** from two prose
    rules into three enumerated axes so the new one fits *inside* the rule rather than beside it:
    same file · same DB/graph key · **one unit decides a fact the other's deliverable must
    encode**. The third is real and distinct — not a third instance of the shared-state principle:
    S8c and the plan-revision unit shared no file and no database, and the coupling ran through
    the *content* of a decision. Re-derived: `salesperson-ui-coordination.md:1752` (*"§5.1's S9 row
    acquired a done-condition that cannot be met"*) and `:2119`. Worded generally enough to also
    cover the still-uncleared chunk-D sibling `b2d7f309-6c14-4e85-a3f0-91e7cc5a2d64`.
  - `b7e41c92-3d5a-4f18-9c60-2a8e17d34f5b` + `fcf3e7a8-1bcd-42a0-86f3-51ad7601dedb` (merged) →
    step 5, sharpening *"Your own rebuttal of a delegate's report is the least-checked claim"*:
    a re-derivation that comes out **clean or stable is not a refutation**. Both fully re-derived.
    `b7e41c92`: `random.Random.choice` really is `seq[self._randbelow(len(seq))]` (index draw), and
    a probe over six permutations of one multiset at a fixed seed produced **2 distinct intervals
    at B=400 but only 1 at B=8000** — confirming the counterintuitive half, that *lower* B makes
    the flip more common. `fcf3e7a8`: swept every exact rational `b/n` for `n<2000` and found
    **99 mismatches** between `floor(x/0.001)` and `floor(x*100/0.1)`, first at `b=7,n=10`.
    **Mechanism sharpened:** the entry frames it as a *units* error; it is a floating-point
    representation error (the double nearest `0.001` sits slightly above it), which is why the
    operational rule promoted is *use the code's own expression*, not *use the code's units*.
    The domain fact itself is already documented at the point of use, in `format_floor_pp`'s
    docstring (`model-bench/modelbench/stats.py:631-647`) — only the coordinator-facing half moved.
  - `e2b7c940-8a15-4d63-9f71-06ad3b5e8c22` → step 5, widening *"Re-verify every summarized number"*
    to *"…and every new identifier"*: a new plan step id is checked against `git log` and the
    ledger, not just the plan. Re-derived: commit `d9d2f2b`'s body opens *"salesperson-ui S7b"* and
    four later commits reference `S7b`/`S7b2`; the coordination doc's U29 row (`:171`) records the
    plan being *"Renamed off the `S7b` collision teco caught"*.
- **Promoted → `skills/agent-standards/claude-code.md`** ("Cross-session peer addressing"), the
  harness half of `1ad53bad`+`7c1e2b04`, with the corrected transcript-storage mechanism above.
  Routed there rather than into `teco.md` because it is a harness fact any agent holding
  `SendMessage` needs, not a coordination rule.
- **Discarded (2):**
  - `58528320-265c-43aa-9dff-029b64fa680d` (parallel subagents share a session-scoped scratchpad)
    — **already promoted**, by U7b/U8 of this same pass, into
    `skills/agent-standards/claude-code.md:516-529`. That entry cites the *same* incident
    (`docs/reviews/small-model-benchmarking-impl.md` Appendix C.5, 2026-09-03), covers the
    sequential-reuse case the raw entry misses, and already carries the briefing consequence
    ("a briefing that dispatches two agents expecting to write scratch files should say so").
    Verbatim-covered a fortiori.
  - `42c89e17-f2bc-497a-ae05-cdb38c35c0a3` (a design rule whose justification only became sound in
    a later revision of the paired document) — **stated mechanism materially wrong.** The entry
    names note **v1.12** as the revision that made §4 S2 rule (iv-b) sound; note v1.12 (`fc2fcf6`)
    is *"the detector is not right-censoring"*, a different subject, and the plan's own record
    (`docs/plans/small-model-benchmarking.md:4707-4709`) names **v1.13** — and says it *"confirmed
    §4 S2 rule (iv-b) by correcting the note rather than the plan… **No plan change was owed, and
    the plan's rule was right before the note's argument for it was**."* So the two gates approved
    a **correct** rule carrying a premature justification; nothing wrong shipped and the pair
    self-corrected one revision later, which is much weaker than the entry's *"invisible to every
    review gate"* framing. Already recorded at the point of use in the plan itself.
- **`MENTIONS` tags added (3)** — each landed and was confirmed committed *before* that entry's
  count-and-decide read, per the §5 ordering invariant; each left the node alive with its
  `MENTIONS` edge after its `PRODUCED` edge was resolved:
  - `3f8b17d2` → `analyst` — *"checked, not guessed"* names the method, not the scope, is a rule
    for the reviewer as much as for the coordinator reading its report.
  - `b7b3bc96` → `analyst` — the reviewer proposing to fold a guard into the step it guards is a
    review-authoring defect, and `analyst` is the agent that authors those recommendations.
  - `b7e41c92` → `data-scientist` — row order being a real input to a seeded bootstrap (and lower
    `B` making the flip more common) is a durable statistics fact in that agent's discipline.
- **Budget:** `teco.md` **6,388 → 6,908 w**; lines past the 700-char smell **17 → 21**. One
  bloated line shortened (*"Close the loop on the same delegate"*, 1,436 → 1,312 ch); four lines
  crossed the smell (`Brief contents`, the reviewer's-suggested-fix bullet, the new `agentId`
  bullet, the own-rebuttal bullet). `claude/AGENTS.md` **untouched** — it sits at 2,434 w with
  ~60 w of headroom, and every rule here is teco-specific or harness-specific. **K-016 escalated
  to the blocking item** it now is; see `plan.md`.
- **Why:** U18 of the team-wide distillation pass; keeps `kaizen_team` as working memory for
  unreviewed capture only.
- **Plan items:** K-016 (updated with this pass's ledger and a concrete first move).

## 2026-09-07 — Kaizen distillation, chunk B (the twelve 2026-09-02 entries)

- **What:** U17 of `claude/docs/plans/kaizen-distillation2-coordination.md` — `cobb` distilled the
  twelve `teco` `:KaizenEntry` nodes dated 2026-09-02 in `kaizen_team`. Result: **8 promoted**
  (6 into `teco.md`, 2 into `claude/AGENTS.md`; six of the eight as in-place clause extensions or
  short bullets, never a paragraph appended after a previous author's), **4 discarded**, 0 kept
  open, **1 `MENTIONS` tag** added. Every entry was re-derived from primary sources rather than
  confirmed against its own cited evidence; one entry's stated mechanism was **falsified** that way.
- **The correction group — `d41e8b07` / `f38a6d15` / `5d8a1c34`, resolved as supersession, one
  promotion.** `5d8a1c34` declared itself a correction to the other two. Re-deriving the underlying
  incident settled it: commit `ef02c7a` (2026-09-02 19:10:15, *"docs: context-file convention +
  repo-wide AGENTS.md bloat sweep"* — a different session's workstream entirely) really did rewrite
  the `seed_salesperson.sh` and `verify_salesperson.sh` rows of `falkor-chat/AGENTS.md`, and its
  version does carry `salesperson@v7` plus its own burned-`v6` clause, exactly as the `coder`
  delegate reported. That commit's own message even states it swept in teco's concurrent L83. So the
  "concurrent writer" was real; **`f38a6d15`'s confabulation diagnosis is false for its only cited
  incident**, and it carries no second instance — discarded as superseded rather than merged (the
  U15 precedent). `d41e8b07` is untouched by the correction and independently true (below). The two
  surviving entries share one root cause — teco's rebuttal of a delegate rested on evidence that
  could not see what the delegate was claiming — so they were promoted **once**, as a single rule
  with both instances as its two clauses.
- **Promoted → `claude/teco/teco.md`:**
  - `d41e8b07-2c95-4f63-a1b8-6e3d0c7f4a29` + `5d8a1c34-7b62-4e09-9f15-3a4c8e2d76b1` → step 5, new
    bullet before *"Close the loop on the same delegate"*: your own rebuttal of a delegate's report
    is the least-checked claim in the coordination. Re-derived independently and both halves hold.
    Graph-vs-git: `git log -S '"v6"' -- proof_defs.py` returns no commits, `git show
    HEAD:…proof_defs.py | grep -c '"v6"'` is 0 and the working tree is 0 — while a live read of
    `ws:acme` returns a `Salesperson v6` `WorkflowDefSnapshot` sitting there right now. A shared
    graph really does hold an artifact that exists in no file and no commit. Scope half: `ef02c7a`
    above. The entry's corollary (never tell a review gate one of your own conclusions is settled
    and out of scope) is kept as the bullet's closing clause.
  - `c04b7f92-6d18-4a35-8e71-93f2c5a08b6d` → step 3, extending the *"Brief contents"* bullet: a
    brief is the one input no gate reads, so route a mandate by stating it and asking where it
    belongs, never by supplying the answer as background fact. Re-derived: `FALKORDB_SOCKET_TIMEOUT`
    is defined at `falkor-chat/server/falkorchat/config.py:29` and consumed at `db.py:44` as the
    FalkorDB client's `socket_timeout=` — a server-side Redis timeout, so the brief's premise that
    it belonged to the SPA's reset UX was wrong at both cited line numbers, exactly as recorded.
    The existing bullet already forbade *paraphrasing a plan* into a brief; it said nothing about
    the coordinator's own asserted premises, which is the gap.
  - `a7d3e619-5c84-4f27-9b13-2e60d8a5c194` → step 5, new bullet: closing an upstream artifact on its
    gate verdict is not absorbing it; sweep its "handed onward" list into the downstream artifact
    item by item at closure. Re-derivation made the entry **stronger** than written:
    `docs/plans/salesperson-ui-graph.md:1021-1027` carries the §12 hand-off list verbatim (including
    the four-part quiesce mandate), the four-part done-condition reached the plan only in the later
    remedial commit `acb5a2a` ("plan v1.3-v1.16 — … S0 mandates …"), and the coordination ledger's
    **U14c** row shows a dedicated remedial unit was commissioned to absorb §12 and found **four**
    unabsorbed items, not the three the entry claims.
  - `c7e41d92-8b3a-4f16-9d02-5a8ef31b7c40` → Guardrails, new bullet before *"A clean pass on a
    brand-new mechanism…"*: a repeated gate has a decidable stopping signal and the reviewer sets
    it, not coordinator patience. Re-derived: `docs/reviews/salesperson-ui.md` carries Passes 2–8,
    all dated 2026-09-02; `:1966` reads *"Mis-ruled — open, and demonstrably still being
    generated. P8-1, P8-2 and P8-3 are three…"*, `:1528` attributes all three to *"the v1.16 delta
    that was meant to close the class"*, and `:1983-1985` is the reviewer itself writing
    *"Recommendation on further passes: stop … A ninth full pass has negative expected value."*
    Nothing in `teco.md` addressed when an iterating gate should stop.
  - `f3a9e21c-7d64-4b08-a5e1-2c9f8b0d6e33` → step 5, sharpening the existing large-context
    **Exception** in place rather than adding a bullet: its carve-out ("doesn't need the delegate's
    own undocumented reasoning") now also reads "**or figures only it observed**". Re-derived from
    the ledger's S6c row — the `coder` was resumed at `a5db169a0966bad59` (~235k tok, i.e. right at
    the threshold that would normally route fresh) and the row records the payoff: it attributed
    three pre-fix survival counts to the review's Appendix J rather than claiming them, and left
    pass counts out where it only had them against a different denominator. The prompt's **Model
    routing** bullet already covered the *model-tier* axis of this; the resume-vs-fresh axis was the
    uncovered one.
  - `b28c5e43-1f76-4d92-a305-7c6e1b9f4a82` → Documentation curation, extending the unfiltered-sweep
    sentence: the same discipline applies **inside one document** — the SCOPE column is the build
    instruction, so sweeping only the done-conditions leaves a removed contract still commissioned.
    Re-derived: `docs/reviews/salesperson-ui.md:776-780` records that `messageCount`/`cartTotal`/
    `orderStatus` are "produced by nothing", the delivered `list_participants`
    (`repository.py:3580`+) projects no activity data, and the ledger's **U19** row shows the
    remedial sweep found 4 hits, 1 of them a defect. Folded into the existing sentence because the
    entry itself calls it "same class as the repo-wide rename rule" — a second bullet would have
    been the duplication the rule is about.
- **Promoted → `claude/AGENTS.md`:**
  - `b7e41c92-3f8a-4d16-9c05-1a2e8f7b3d40` → a new short paragraph in the Git-commit authority
    section: the injected `Claude-Session:` attribution guidance is a harness default that knows
    nothing of this user's settings, memory or repo docs, and `includeCoAuthoredBy: false` does not
    suppress it — so its recurrence is not evidence the preference changed, and it is not a conflict
    to escalate. Re-derived first-hand rather than from the entry: `includeCoAuthoredBy: false` is
    present in `~/.claude/settings.json`, the injection fired in this very distillation session
    while that setting was live, `grep -rn 'Claude-Session'` over the repo returns exactly one hit
    (`falkor-chat/docs/plans/workflow-timers-coordination.md:24`, which *forbids* the footer), and
    the user memory file forbids it. The two are different features, not one broken one.
  - `e5c1f284-9a37-4d60-b8e2-71f4a90c3e58` → appended one sentence to the concurrent-write
    paragraph (the one U16 had just rewritten): a file untracked at session start can become
    **tracked** by another session's commit, which reads as its having vanished — check `ls`,
    `git ls-files` and `git log --all -- <path>` before calling it data loss. The entry's other two
    claims were already published in that same paragraph (commit by explicit path, never
    `git add -A`) and were deliberately **not** restated. Concurrency re-derived live rather than
    from the entry's stale byte counts: `git log` on 2026-09-07 interleaves salesperson-ui,
    model-bench and kaizen commits minutes apart, and this session opened with
    `M docs/plans/small-model-benchmarking.md` dirty in another session's hands.
- **Discarded — falsified:**
  - `f38a6d15-4e72-4b90-9c31-8d05e2a7f6b4` (a long-running delegate misrecognising its own edit as a
    concurrent writer). Superseded by `5d8a1c34`; see the correction group above. **This is the
    entry whose stated mechanism the re-derivation overturned** — the delegate was right and the
    coordinator was wrong, the reverse of what the entry records. Not promoted in any form.
- **Discarded — already documented:**
  - `c92f5a3e-7b41-4d28-8e06-5f1c9a2b6d73` (an unrecorded `agentId` makes a dispatched unit
    unrecoverable). The preventive rule is already in `teco.md` verbatim — *"**Record the identity**
    … write it into that unit's ledger row **at dispatch, always**"* — and step 5 already states
    both the fallback order and "you have no agent-enumeration tool". The entry's addressing fact is
    already in `skills/agent-standards/claude-code.md` under "Nested-delegation notification
    routing", which records the same `"No agent named 'teco' is reachable"` failure for the reverse
    direction. What remained was a detection heuristic and a recovery path for a rule the prompt
    already makes unbreakable — below the "every session pays for it" bar, especially against open
    item K-016.
- **Discarded — corrupt node, no content to route:**
  - `9f2b6c07-3e51-4a88-b174-c6d90e melhor` — `fact`, `evidence` and `context` all the literal
    string `PLACEHOLDER`, `suggestedHome: 'unsure'`, and an `entryId` that is a truncated uuid4 with
    the Portuguese word `melhor` appended **after a literal space** (37 characters). Confirmed to
    match exactly one node before clearing. Nothing to verify or route. The node's *existence* is
    the finding and it is not teco's: the curator write path accepted an all-`PLACEHOLDER` entry
    under a malformed id, so nothing validates entry content or id shape at write time — filed
    against `cobb`'s own machinery in `claude/cobb/kaizen/plan.md`, not here.
- **`MENTIONS` tags added:** one — `b28c5e43…` → `architect`. The incident is teco's (it
  commissioned the trim and re-gated it), which is why the rule was promoted here, but the artifact
  swept is the architect's own step table and the sweep is a plan-editing discipline. Tagged so it
  resurfaces in `architect`'s own pass rather than being lost with teco's clear.
- **Clearing:** all 12 nodes had exactly one `PRODUCED` edge and no `MENTIONS` edge, except
  `b28c5e43…` which was tagged first (ordering invariant: the tag committed before the
  count-and-decide read), leaving `otherRemaining = 1` — so its `PRODUCED` edge was resolved and the
  node kept alive for `architect`. The other 11 had `otherRemaining = 0` and were `DETACH DELETE`d
  whole. Every history append landed before its graph mutation.
- **Cost of this pass, for K-016:** `teco.md` 6,046 → 6,388 words. Two of the six promotions
  extended lines that were already past the 700-char smell (1,266 → 1,447 and 770 → 1,029); the
  count of >700-char lines is unchanged at 17, so no new one was created. Noted in `plan.md`.

## 2026-09-07 — Kaizen distillation, chunk A (8 oldest entries, 2026-08-25 → 2026-09-01)

- **What:** U16 of `claude/docs/plans/kaizen-distillation2-coordination.md` — `cobb` distilled the
  8 oldest `teco` `:KaizenEntry` nodes in `kaizen_team`. Result: **2 promoted** (both as one-clause
  in-place sharpenings of existing text, not new bullets), **6 discarded**, 0 kept open, 0
  `MENTIONS` tags. Every entry was re-derived independently rather than confirmed against its own
  cited evidence.
- **Promoted:**
  - `b3f1a2c4-7e2a-4b9a-9c1e-1a2b3c4d5e6f` (2026-08-25, pathspec `git commit` ignores the index for
    the named paths) → **`claude/AGENTS.md`**, concurrent-write paragraph, replacing its closing
    "if you do stage, re-check `git diff --cached`" sentence. Empirically re-derived in a throwaway
    scratchpad repo (git 2.43.0), never against the shared tree: with only hunk A staged for `f.txt`
    and hunk B added to the working tree afterwards, `git commit -m x -- f.txt` committed **both**
    hunks and left the index clean, while the identical setup committed **without** a pathspec
    honoured the index and committed hunk A alone. A separately-staged unrelated path survived the
    pathspec commit untouched, confirming the remedy the paragraph already mandates. The entry's
    wording ("silently re-stages the current working-tree content") is exactly right — the index is
    rewritten to match, not merely bypassed. The replaced sentence was the defect: it recommended
    `git diff --cached` as the pre-commit check, which under the mandated path-limited form shows
    your hunks and proves nothing about what lands — the precise false assurance behind commit
    `d41da78`, where a concurrent session's `repository.py` content shipped under this team's
    message. Replaced rather than appended per root `AGENTS.md`'s context-file rule.
  - `a1e2c3d4-5f6a-4b7c-8d9e-0f1a2b3c4d5e` (2026-08-28, a live LLM run caught a bug a green offline
    suite and a static plan review both missed) → **`teco.md`**, Guardrails, extending the existing
    "Neither gate is execution" sentence by one clause. The incident itself is already documented at
    the point of use — `falkor-chat/server/falkorchat/repository.py:2740-2751` records the live run
    lowercasing "audio" against the seeded "Audio" and the `categoryNormalized` fix — and
    `docs/reviews/workflow-catalog-lookup-impl.md:88` corroborates the three extra `test_queries.sh`
    assertions U15b added on top. What was **not** written down anywhere is the routing consequence:
    the existing sentence offers "a test **or** the live system" as interchangeable proof of
    execution, so a coordinator could accept "1810 tests green" as satisfying it — which is what
    happened. The clause states that a green offline suite is not that run when the behavior depends
    on a live model's free-text output. Kept to one clause deliberately, given open plan item K-016
    (`teco.md` bloat).
- **Discarded — already published:**
  - `b7c4e9a1-3f52-4d18-9a6e-7c1b0e4f8d92` (2026-09-01, the resolved `bypassPermissions`
    subagent-write gap) and its superseded predecessor `f3e6b1a2-9c4d-4e7a-8b1f-2d5c7a9e0b31`
    (same day, the unresolved observation), handled as one. Every claim of the resolved entry is
    already in `skills/agent-standards/claude-code.md`: the header banner (background subagents by
    default since v2.1.232; `Write`/`Edit` prompts persist under a parent confirmed continuously in
    literal `bypassPermissions`), the `## Hooks` 2026-09-01 resolution block (the four-point root
    cause, including the tool-class-specific finding that not one `Bash` call ever produced a
    confirmation gap), and the record that this repo's `defaultMode` pin was reverted 2026-09-01.
    Re-derived independently: `.claude/settings.json` today carries **no** `defaultMode` key at all
    (`git log` — `6f719ae` pinned `bypassPermissions` 2026-08-29, `4bb96e1` reverted it), so the
    superseded entry's stated premise no longer holds and promoting it would have shipped a false
    config claim. The full investigation trail is `claude/docs/plans/bypass-permissions-subagent-gap.md`.
  - `b3f2a1c4-8e9d-4a5b-9c1e-7d2f6a8b3c05` (2026-08-30, the `BACKLOG.md`/`HISTORY.md` asymmetry at
    milestone close). Root `AGENTS.md` already carries the rule in full — "**The human applies the
    list**" and "`BACKLOG.md` is forward-looking only — a delivered item does not stay in it".
    **The entry's stated mechanism is wrong**: it frames the asymmetry as a write-permission one
    ("appending `HISTORY.md` entries is routine and teco does it directly"), but re-deriving
    `teco/hooks/guard-coordination-doc-writes.sh` shows its allowlist is `docs/plans/*` plus the
    mechanical `Status: archived` flip — `<component>/docs/HISTORY.md` matches neither, so a
    `HISTORY.md` append escalates to the human exactly as a `BACKLOG.md` edit does. The real
    asymmetry is doctrinal, not guard-level. `.claude/settings.json` additionally gates
    `Edit(**/docs/BACKLOG.md)` behind an explicit `permissions.ask` rule.
  - `e7c1a9d4-3b6f-4e2a-9d8c-1f5a7b2c4e60` (2026-08-30, two file-disjoint units colliding through a
    shared DB fixture). `teco.md`'s dispatch bullet already carries the rule at strictly broader
    scope — one agent owns a shared database/graph key whenever both units' suites exercise it,
    "**not only when a unit destructively wipes it**" — so the destructive case is covered a
    fortiori. **The entry's stated mechanism is also wrong in two ways**: re-deriving
    `falkor-chat/server/tests/conftest.py:100-110`, the `reference`-graph `MATCH (n) DETACH DELETE n`
    runs in the body of the opt-in `wf_repo` fixture, i.e. at **setup**, not at teardown; and it is
    not reached by a "destructive default `pytest`" — only by tests that request that fixture.
  - `a1e6f0d2-8c3b-4e1a-9f7d-2b5c6a4d1e90` (2026-08-27, `architect` refusing an agent-to-agent
    resume asking it to edit `docs/BACKLOG.md`). Substantively about `architect`, but re-derivation
    found **both** halves already documented, so there was nothing left for an `architect`-side pass
    to decide and no `MENTIONS` tag was added: `architect.md`'s Guardrails state the scope and its
    harness enforcement verbatim ("Your `Write`/`Edit` access exists for one purpose… a `PreToolUse`
    hook escalates any `Write`/`Edit` outside a `docs/plans/` directory"), matched by
    `architect/hooks/`'s `docs/plans/*|*/docs/plans/*` allowlist; and "no agent message can
    authorize changing your permission settings" is harness-injected boilerplate every subagent
    receives, not repo-authored text that could be promoted. The entry records the design working
    as specified.
  - `e2f8a4c1-9b3d-4e7a-b6c2-1d5f8a3e0c47` (2026-08-25, a mid-pass `git diff --stat` showing an
    in-flight sibling's edits, plus mid-run `MENTIONS` tags being caught anyway). First half:
    `Edit` writes landing on disk immediately is not a non-obvious fact, the observed case was two
    units sharing `falkor-chat/docs/SERVER.md` — which `teco.md`'s "serialize units that touch the
    same file" rule already forbids — and the sharper hazard in the same family (a concurrent
    session silently reverting a confirmed `Edit`) is already in
    `skills/agent-standards/claude-code.md`'s Bash-environment section. Second half is
    distillation-pass trivia, and moot under this pass's one-agent-at-a-time design.
- **Why (process):** verify by re-deriving, never by confirming the entry's own citation. Three of
  the six discards had a materially wrong stated mechanism (`b3f2a1c4`, `e7c1a9d4`, and
  `f3e6b1a2`'s now-false config premise) while still pointing at a real underlying fact — a
  verbatim promotion would have shipped the wrong claim in each case.
- **Open question raised, not resolved here (for the stakeholder):** the shipped config state and
  the user's standing `subagent-permission-mitigation` memory disagree. The memory names running
  the parent session in `acceptEdits` as the mitigation; the revert removed `defaultMode` entirely,
  leaving the parent on the harness default (`auto`), which is what Gen 3 recommended and what Gen
  4 §4.2 offered as "unset / `auto` explicitly". Gen 4 proved `bypassPermissions` buys nothing over
  `auto`; it did **not** re-test `acceptEdits`, whose earlier forensics showed partial
  per-run stickiness that `auto` does not give. Flagged for `cobb`/the stakeholder, not acted on.

## 2026-09-06 — Environment readiness: bring the stack up, don't report it down

- **What:** two additions to `teco.md`, both inside "How you work" (no new section, no structural
  change):
  1. **Step 1 gained an "Environment readiness" paragraph.** When any unit, gate, or teco's own
     learnings capture will touch FalkorDB / a CPG / `mcp__cypher__query`, teco probes once at
     orientation (`redis-cli -p 6379 ping`, or a cheap `mcp__cypher__query` read) and, on a miss,
     **starts the service itself** — `./falkor-chat/scripts/start_falkordb.sh -d`, re-probe, retry
     the failed query. The paragraph carries four facts teco demonstrably didn't have: (a)
     `docker start falkordb-dev` is the wrong path, because that script runs `docker run --rm` so
     there is normally no stopped container to start; (b) graphs live in the named volume
     `falkordb-data` and survive a restart, so this is not data-destructive; (c) a service start is
     additive/idempotent and therefore inside teco's existing `Bash` grant, while `docker rm -f` /
     `GRAPH.DELETE` / volume removal / `pipeline.sh … --reset` are not and stay with
     `devops`/`graph-dba` behind their destructive-ops guards; (d) a *failed* bring-up is an
     environment blocker for `devops` — the routing row that already existed — not something to
     hand back to the user.
  2. **Step 3's CPG-freshness bullet gained a refresh clause.** A stale or absent CPG the task
     genuinely leans on is now a **unit** (`graph-dba` rebuild, freshness evidence in the brief,
     dependent unit sequenced behind it), not a caveat pasted into someone's brief. Explicitly
     contrasted with the FalkorDB case: the rebuild is a multi-minute Joern run whose load step is
     destructively guarded, so it is never teco's to run itself.
- **Why:** stakeholder report — a live `teco` session found FalkorDB not running and reported
  "some things won't work (kaizen_team)" instead of starting it. The prompt already routed
  *implementers'* environment blockers to `devops` ("instead of returning them to the user") but
  said nothing about a blocker teco hits **itself**, at orientation, before any unit exists to
  route. The stakeholder's ask was explicitly proactive: bring up everything the task needs,
  including refreshing the CPG.
- **Why teco-only, not team-wide:** exactly the argument that already centralized CPG-freshness
  checking here (2026-08-19, `docs/plans/cpg-agent-adoption2.md`) — the coordinator sees the whole
  goal before any specialist is spawned, so one check at orientation replaces the same rule
  duplicated across twelve always-loaded prompts. A specialist run standalone still gets neither
  check; that trade was accepted for freshness and is accepted again here.
- **Honesty correction made during authoring:** the first draft asserted "the MCP server's
  connection pool recovers without restarting the session" as fact, citing
  `docs/plans/cpg-query-access.md` §7.3. The §7.3 row is a *plan* row; the matching report
  (`docs/archive/test-reports/cpg-query-access-report.md`) records that the stop/restart case was
  **deliberately not executed** — stopping the shared `falkordb-dev` needs stakeholder approval a
  subagent can't obtain — and lists it as an untested recovery path with residual risk. The
  shipped wording now says the recovery is *expected*, names both documents, and notes that one
  retry settles it either way and a non-recovery is worth capturing. Not live-verified in this
  session either, deliberately: another `teco` session was using the shared instance.
- **Not changed:** decomposition, dispatch, the ledger, gates, guardrails, commit authority,
  hooks, frontmatter. No roster change, so no team-coherence re-certification was triggered.
- **Two composition conflicts found by the §7 lint and fixed in the same pass** — the new rule
  contradicted Guardrails as originally written, in both directions:
  1. The `Bash` guardrail enumerated teco's write actions exhaustively as "one narrow write action:
     integration commits". Now reads "two narrow write actions", naming the service start and
     pointing at step 1.
  2. "Never touch, stage, or commit any file outside the coordination you're actively running, **or
     any running service**" would have forbidden the new rule by its last clause. Rewritten to the
     sharper property the rule actually needs: **a service that is up is untouchable** — never
     stop, restart, or reset one, and never assume you are its only user (`falkordb-dev` is shared
     by other sessions and components) — while starting one that is *down* is the step-1 exception
     and the only one. This is a real safety gain independent of the lint: it was previously only
     implied, and this very session found two other agents' work live in the tree.
- **Cost:** +370 words on the team's heaviest prompt (5,511 → 5,881 body words, measured), which cuts
  against open item K-016. Accepted deliberately: the rule is a **proactive trigger** — teco must
  fire it before it knows anything is wrong — so it cannot live in an on-demand knowledge base
  (the same test K-016's own notes apply to the paused-unit protocol). A follow-up that *does*
  reclaim the words is filed in `plan.md` (K-017).
- **Docs updated:** `claude/README.md` — the teco row's CPG-freshness sentence extended to cover
  environment readiness. `claude/AGENTS.md` needed no change (its teco entry is name-only).
- **Verified:** `bash claude/scripts/audit-team.sh` — see the run recorded below.

## 2026-09-01 — Symmetric update for tico's new docs-only coordination capability
- **What:** small, targeted additions (not a rewrite) so `teco` recognizes the new boundary from
  its own side. Full design and rationale live in `tico/kaizen/history.md`'s 2026-09-01 entry —
  not restated here. Two edits:
  1. **Routing table** gained a new row: a goal that decomposes entirely into
     requirements/plan/review work (no unit ever touches source/tests/config) now routes to
     "pause → user, recommend `tico`" instead of straight to `teco` — tico coordinates that scope
     itself since 2026-09-01.
  2. **Handoff contracts**, tico bullet: noted that tico may now arrive having already run part of
     a docs-only coordination and handed off mid-chain — its `docs/plans/<slug>-coordination.md`
     ledger is `teco`'s state of record for that slug, exactly like one `teco` opened itself. No
     new mechanic needed: step 1's existing "a coordination doc for this slug is the state of
     record, read and reconcile it" already covers a tico-authored ledger without modification.
- **Why:** direct consequence of tico's coordinator capability — `teco` needs to know when to
  defer at intake and how to resume a chain tico started, or the two agents silently duplicate or
  drop work on the same slug.
- **Not changed:** everything else — decomposition, dispatch, the ledger format/thresholds, the
  archived-flip ownership. `teco`'s own K-016 (prompt-size consolidation) is unaffected in scope;
  this added ~90 words, not a new subsystem.
- **Verified:** `bash claude/scripts/audit-team.sh` — see the same-day run recorded once in
  `tico/kaizen/history.md`.
- **Plan items:** none opened on `teco`'s side — the live e2e validation (K-013/K-014) is tracked
  in `tico/kaizen/plan.md` since tico is the initiating side of the handoff.

## 2026-08-25 — Distillation (unit U4, `cobb`): 6 raw `kaizen_team` entries verified, routed, cleared

- **What:** `cobb` ran the `agent-maintenance` §5 distillation procedure against every `teco`-tied
  `kaizen_team` node — the legacy `author:'teco'` read (3 entries) plus the current-shape
  `PRODUCED`/`MENTIONS` read (3 entries), 6 total, all from 2026-08-21 through 2026-08-24. Full
  field text paged via `size()`/`substring()` (the display truncates each cell at ~300 chars).
  Dispositions:
  1. **`teco-20260821-specialist-sendmessage-gap`** (legacy, 2026-08-21 — dispatched specialists
     generally lack `SendMessage`, only `teco`/`tico` have it) → **routed to knowledge base**,
     overriding the entry's own `suggestedHome: prompt`. Live-reverified and broadened: a fresh
     `coder` probe (no `tools:` restriction declared, nominally "all tools") also lacks
     `SendMessage` at runtime — the gap is team-wide, not just `architect`/`analyst`. But the
     K-028 evidence's actual mechanism is the *already-promoted* nested-notification-bubbling fact
     (`skills/agent-standards/claude-code.md`, entry `7994edd7…`, 2026-08-21) — a background
     `teco`'s own turn had ended, so completions bubbled to the live ancestor instead of to
     `teco`, independent of whether the specialist had `SendMessage`. `teco`'s real signal is
     always the platform's completion notification, never a delegate self-report, so the tool
     grant wouldn't change `teco`'s behavior — no `teco.md` edit. Promoted the tool-inventory fact
     itself as a new bullet in that same knowledge-base section (live-tested 2026-08-25). No
     `MENTIONS` tag added — the fact is cross-cutting tool-standards knowledge cobb already owns
     and acted on directly, not a behavior gap in any one other agent's own prompt. `plan.md`
     parking-lot item closed in place with this finding.
  2. **`f3a2e8b1-6c4d-4a9e-8b2f-1d5e7c9a3b6f`** (legacy, 2026-08-21 — K-028: two independent
     `analyst` plan-gate passes approved a mandatory-unconditional-fallback-arm fix that was
     provably unreachable at runtime; only implementation/a live re-trace caught it) → **verified
     true** (re-read `falkor-chat/docs/plans/workflow-timers-coordination.md`'s trail; the
     `executor._drive_loop`/unconditional-guard mechanism the entry describes matches) and
     **promoted to `teco.md`'s "Two gates, not one" guardrail** — new clause: *"Neither gate is
     execution — a static trace, however careful, even twice independently, can approve a
     mechanism that is provably non-functional; state-machine/order-dependent logic needs to
     actually run (a test or the live system), not just be reviewed, before you trust it."* Not
     previously covered — grepped `teco.md` for "static"/"non-functional"/"actually driving"
     first; only the routing-table row existed.
  3. **`b3e4a6a0-6c2e-4f7a-9d1a-8e5f7c2d4a11`** (legacy, 2026-08-23) **+ `c7e2a814-4f1b-4a9d-8e3c-
     2b6f9d1a5c33`** (current-shape, 2026-08-23) — **duplicate observations of the same K-050
     incident** (U9 coder at 284k tokens/137 tool uses, resumed via `SendMessage` for a small,
     fully-specified fix rather than dispatched fresh) → **merged and promoted as one rule** into
     `teco.md` step 5's "Close the loop on the same delegate": new **Exception** clause — when the
     ledger's `Cost` column already shows the delegate carrying a very large context (~250k+
     tokens/100+ tool uses) and the follow-up is small and self-contained, dispatch fresh instead
     of resuming. Ties directly into the existing `Cost` ledger column (added 2026-08-21), so it's
     checkable with no new measurement burden.
  4. **`a1f3c9e2-6b4d-4e2a-9c1a-7d8f2b3e5a10`** (current-shape, 2026-08-23 — K-050: the
     resume-by-`agentId` discipline was lapsed on twice for `architect`/`analyst` fixes despite
     both being freshly in context) → **discarded**. Read against `teco.md`'s existing "Close the
     loop on the same delegate" text: the resume-by-default rule is already stated correctly: the
     lapse was execution discipline on a small-context delegate, not a prompt gap, and the entry
     itself says as much ("inconsistent, not a knowledge gap"). Folded in as supporting evidence
     for item 3's promotion (confirms resume stays the *default*; only the large-context case gets
     the new exception) rather than a separate change.
  5. **`e7f3a1b2-9c4d-4e6a-8f21-3d5c7b9a1e04`** (current-shape, 2026-08-24 — a `permissions.allow
     Edit(path)` rule does not suppress the auto-mode classifier prompt for a Task/`Agent`-
     delegated subagent write, even with a matching `PreToolUse` hook allow) → **discarded**,
     already fully documented. Checked `claude/docs/requirements/agent-permission-friction2.md`
     and `claude/docs/plans/write-guard-classifier-gap(-coordination).md` (both `Status: archived`)
     — this exact finding is the closed investigation's own "Result: refuted" conclusion, with a
     full root cause and no further action expected. This entry is an earlier, now-fully-
     superseded data point of the same closed finding.
- **Why:** routine distillation, batch unit U4 of the team-wide pass coordinated by `teco`
  (`claude/docs/plans/kaizen-distillation-coordination.md`).
- **`MENTIONS` tags added:** none. All 6 entries are substantively about `teco`'s own coordination
  behavior (dispatch discipline, review-gate epistemics, tool-availability assumptions), not about
  another agent's own prompt/behavior gap.
- **Verified:** every entry's full field text paged (not acted on from a truncated read); item 1's
  broadened claim live-tested via a fresh `coder` subagent probe; item 2's incident re-read against
  its own coordination doc; item 5 cross-checked against the two closed, archived documents it
  duplicates. `teco.md` edits are additive clauses inside existing guardrails/bullets — no rule
  removed, no restructuring; `audit-team.sh` not re-run this pass (no roster/hook/tool-grant
  change), consistent with a distillation-only unit.
- **Plan items:** the `SendMessage`-grant parking-lot item closed in place (item 1, above); no new
  K-items opened — items 2 and 3's promotions are direct prompt edits with no open follow-up.
- **Docs touched:** `claude/teco/teco.md` (two guardrail clauses) · `claude/teco/kaizen/{plan,
  history}.md` (this pass) · `skills/agent-standards/claude-code.md` (one new knowledge-base
  bullet).

## 2026-08-25 — The ledger cites, it does not restate (prompt-waste Stage D item, routed here)
- **What:** one sentence into the ledger paragraph at `:69` — *"**The ledger cites; it does not restate** — a row points at the plan section or deliverable path that carries a decision, never re-explains it."* +25 w. Closes the item parked here at Stage D, when the prompt-waste plan specified the rule for `architect.md` but `architect` authors no delegation-summary table — the unit ledger in `plans/<slug>-coordination.md` is **this** agent's artifact.
- **Lexically matched to its sibling on purpose.** `architect.md:41` reads *"a recap table cites, it does not restate"*; this reads *"The ledger cites; it does not restate."* Same construction, same verb pair — which is what will let a future reader recognize them as one rule rather than two coincidences. Deliberately **not** enforced by a script: the two guard different artifacts and were not written to be byte-identical, so a check would invent a constraint in order to make it checkable (the failure mode plan finding 23 warns against). The `agent-maintenance` §4 judgment pass is the right reader.
- **Kept inline rather than routed to K-016, and the reasoning is worth recording.** `teco.md` is 5,286 w and `audit-team.sh` check 9 NOTEs it, so this was the first decision since Stage F where the advisory could have changed an outcome. It correctly didn't. Two reasons: **K-016 is for rare-path rules**, and this one fires on *every ledger row of every unit of every coordination* — finding 8 requires an offloaded reactive rule to leave a trigger stub, and a rule with no rare path has no trigger to stub. And check 9 was built advisory-only precisely so it could not exert this pressure — its header comment says a failing tripwire *"would pressure someone to cut a rule to hit a number."* The tripwire's semantics got their first real test and held.
- **One tension noted and deliberately not "fixed":** the same paragraph says a `paused` row's Deliverable column carries the open question inline rather than a path. That resolves correctly — an open question awaiting an answer is not a *decision* being re-explained, and the carve-out is anchored by *"the only `Status` value that repurposes that column"* — so nobody should later convert the paused row into a pointer.
- **Verified:** `audit-team.sh` PASS; `cobb` §7 lint clean on this unit (0 findings; 2 nits, both analysis).

## 2026-08-25 — `:129`'s description of `tico`'s grant was stale for four weeks (prompt-waste Stage E pass 2, a declared correction)
- **What:** one line, net 0 w. `:129` described `tico`'s commit grant as one "which mirrors its own write-guard exactly because it only ever commits what it itself wrote" → "which is scoped to its own doc kinds plus two narrow cases (`claude/AGENTS.md`, "Git-commit authority")".
- **Why it was wrong.** The claim dates to `eb318d4` (2026-07-30), the original grant formalization, and was never touched again. `tico`'s grant was extended twice afterwards: **2026-08-21** added the returned artifact of a `qa-engineer`/`analyst` verification pass tico itself offered — which by construction tico did *not* write, breaking "only ever commits what it itself wrote" — and **2026-08-24** added a file tico wrote whose `Write`/`Edit` guard escalation the human approved, which is outside the guard's static allowlist and so breaks "mirrors its own write-guard exactly". Both halves of the sentence were false, and `teco` is the agent most likely to act on it: **its own grant is defined by contrast with tico's**, so the comparison is load-bearing here in a way it is nowhere else.
- **How it surfaced — the part worth carrying forward.** Not by a grep for stale facts. Stage E pass 2 deleted, from `claude/AGENTS.md`, the class-6 supersession clause saying extension B "deliberately breaks the write-scope==commit-scope identity `tico` previously held". The C2 anti-trigger test on `tico` correctly returned *No* (its own prompt enumerates all three cases, so it cannot infer the identity) — but that sentence was the **only marker in the corpus** that `:129` was out of date. Deleting it would have left two confident, contradictory statements with no resolution signal. Fixed at the source instead of restoring provenance to a file that had just been cleaned. **The generalizable rule: a grant defined by contrast lives in two files, and widening one side is not done until the other side's sentence is re-read.** Recorded as finding 18 in `claude/docs/plans/prompt-waste-reduction.md`.
- **What was deliberately not changed:** the rest of `:129` — the integrator-role scoping, the "yours and `tico`'s are the only two unconditioned on interactive-vs-subagent mode" clause, the universal-grant pointer, and the two anti-inference sentences — all intact. `:127`'s inline `by explicit path` and `:130`'s no-hook restatement were checked by the lint and confirmed as correct handoff-symmetry duplicates, not waste.
- **Verified:** `audit-team.sh` **PASS** (check 8's `git add`/`git commit` and "delegated subagent" tokens untouched); `cobb` §7 lint raised this as a MAJOR and confirmed the fix; `grep` confirms no restatement of the stale identity claim survives anywhere in `claude/`.

## 2026-08-24 — Prompt-waste C1 pass 2: duplicate restatement removed (5,728 → 5,377 w) — C1 complete
- **What:** The class-7 half of C1, unblocked by pass 1's observation window closing clean. 18 edits, all removing a rule's *second* statement or a trailing rationale that restated the clause it followed. No rule deleted; the canonical statement of every rule stays. Worked from the pass-2 keep-list `cobb` produced at pass 1 (in the entry below), which named the clauses that look like duplication but are mechanism.
- **Removed — the rule survives in one canonical home:**
  - "must survive a compaction … only the ledger, not your context window, guarantees that" (step 2 trigger) → the ledger paragraph's own bolded rule.
  - The mechanical-unit parenthetical in the stop-and-ask exemption → Model routing's definition, two bullets down.
  - "Verify by reading at integration — never accept 'docs updated' as a claim" → step 5's "Documentation is part of done".
  - Guardrail "Briefs must stand alone" → step 3's section lead, which states it verbatim.
  - "Running the project's suites/scripts yourself is in-bounds verification" (step 5) → the `Bash` guardrail and the `qa-engineer` routing row.
  - The `security-expert` routing row's cobb/devops finding-routing → the `security-expert` handoff contract.
  - The `AskUserQuestion`-withheld-from-subagents triple, collapsed exactly as the keep-list directed: the paused-unit bullet's restatement became a cross-ref to **Pause vs. proceed** (canonical); the tool-trust guardrail's copy stayed, because there it is a *tool-availability* fact, not a duplicate of the pause protocol.
  - Step 4's paused-resume fallback → step 5's "Close the loop on the same delegate"; now a pointer.
  - Nine trailing rationale restatements that added no rule (listed in the commit message).
- **Gate (a) — every keep-list item verified present after the edits:** fork blockquote with both worked examples; the three `CPG:` forms; the 7-value `Status` list + `paused`-repurposes-Deliverable + the `U3` example row; all 13 roster names; check-8 tokens (`git add`/`git commit` ×1, "delegated subagent" ×2); the enumerated `stash`/`checkout`/`restore` never-list *and* "including inside an implementer's brief"; the two-hop `SendMessage` chain; the Cypher template.
- **Applied pass 1's own lesson:** every reworded sentence was re-read as a rule diff, not a length diff — that is where pass 1's single defect came from. Where a cut would have changed scope, the sentence was left alone; the commit-grant bullet (the pass-1 regression site) was not touched at all this pass.
- **Verified:** `audit-team.sh` — every teco check green, all 13 roster names and check-8 tokens present (the repo carries one unrelated pre-existing FAIL, a username/home-path leak at `claude/docs/reviews/write-guard-classifier-gap.md:58` from commit `374a350`, another session's in-flight document — deliberately untouched, not this unit's to edit). `cobb` §7 lint: **pass with findings — 1 major, 6 minor, 0 blockers**; all fixed before commit. Final: **5,377 w**.
  - **Major — a class-7 cut where *placement* was load-bearing, the characteristic failure of this pass.** "Delegate wide searches to **Explore**" (step 1) was cut as a duplicate of the `Explore` routing row. It isn't: the routing row governs routing *a unit of work* to Explore — a decision about someone else's task — while the step-1 sentence governed **teco's own orientation reads**, a different act at a different moment. It compounded, because the tool-trust guardrail affirmatively says "use `Bash` (`grep`, `find`) for search and reading", so removing the counterweight left teco pushed toward sweeping the repo in its own context. Restored.
  - **Minors, all fixed:** a pronoun whose antecedent moved ("relay it" → "relay the question"); the **Pause vs. proceed** pointer could be misread as re-opening the escalate/don't-escalate decision the imperative had just settled, so it now points at the *mechanics* explicitly; "Bash command patterns" restored (bare "patterns" reads as file paths, the wrong model for why no hook fires); the rename-sweep rule's generalizer restored — it was not a second why but a **scope extender** ("as must any scan whose purpose is proving a negative"), and cutting it narrowed the rule to rename/removal only, which is pass 1's finding-1 shape recurring at minor severity; and the `tico`-manual routing row regained a 7-word gate reminder, because that row grants an *exception* to "tico is not a delegation target" and its job was to block the inference that the exception extends to the review gate.
- **Two accuracy corrections `cobb` asked be carried here, and they should be:**
  - **Gate (a) discipline gap:** two edits in this pass were not in the enumeration I handed the lint — the `tico`-manual gate sentence and the fork-path rewrite. The mapping exists to catch exactly that; an unenumerated edit is an unreviewed one. Enumerate from the diff, not from memory.
  - **Classification correction:** the `audit-team.sh` check-8 clause was cut as class 7, but it is not restated anywhere in this file — it is class 6 (governance detail about a script teco never runs). Right disposition, wrong reason. Recorded so a later attribution pass doesn't go hunting for a surviving copy that was never there. The fact itself survives in the load-set, at `claude/AGENTS.md`'s "Git-commit authority".
- **Word-count outcome, and the target is what moves:** 5,948 → **5,377** across both passes (−9.6%), against a C1 band of ~4,300–4,600 revised at pass 1. `cobb` re-estimated independently, by a mechanical repeated-5-gram scan rather than impression: **under 200 w of cross-line restatement remains**, most of it class-2 mechanism that *must* repeat (the `<component>/docs/…` path forms across four handoff contracts). Its verdict: the pass-1 estimate was optimistic, the file's editorial floor is **~5,200–5,250 with every rule intact**, and `teco.md` is ~60 distinct rules at ~85 w each — not a narrative file, and not one since pass 1. Per plan §7, a file above target with every rule intact **passes**; the band moves, not the file.
- **~115 w of further cuts identified and deliberately declined.** `cobb` named six defensible trims. I took none: they would be new edits made *after* the lint, shipping unreviewed for a ~2% gain against a floor the lint had just certified — and one of them touches the commit-grant paragraph, the exact site of pass 1's only real regression, which should not be re-edited in the same unit whose lint has already run. Listed in the plan for whoever wants them under their own gate.
- **The only remaining lever is structural, and it already exists:** `kaizen/plan.md` **K-016** (split rare-path rules into an on-demand `coordination-techniques.md`). `cobb` appended this pass's measurement to it, converting it from "the prompt feels heavy" into "editorial means are exhausted at ~5,200; this is the only path below it" — with a caveat to honor: the largest KB candidate by word count is the paused-unit / stop-and-ask protocol (~380 w), but it is **reactive**, so only its mechanics can move. Teco must recognize the trigger in order to know to load the file; move the trigger and you get the failure mode progressive disclosure is prone to.
- **Observation window — target the rewritten clauses, not the deleted ones** (the plan's own carry-forward from pass 1): the paused-unit relay pointer, the resume-fallback pointer, the mechanical-unit exemption, the fork-path trailing sentence, and the no-hook-backstops bullet. Three of the four paused-unit bullets were rewritten here and "paused-unit handling" is already on the §6 watch list, so **round 1 of the next probe should force a pause.**

## 2026-08-24 — C1 pass 1 observation window CLOSED clean (synthetic probe, §6)
- **What:** No organic dispatch occurred after C1 pass 1, so a synthetic probe substituted per `claude/docs/plans/prompt-waste-reduction.md` §6. Two rounds against one teco instance: (1) a planning-only coordination of a real, unbuilt falkor-chat feature (document deletion — cascade to chunks/embeddings plus orphan-entity cleanup, API + MCP + docs); (2) a `SendMessage` resume adding milestone-close units and asking what gets committed by whom. Planning-only by construction: no dispatches, no writes, no commits — `git status` confirmed clean afterward. **No prompt change in this entry.**
- **Result: PASS, no breakage.** Every §6 watch-list rule the probe could trigger fired, including all seven whose narratives pass 1 removed.
  - **`subagent_type`** — every unit routed to a named agent type. **`agentId`** — ledger column present, and re-gates planned as `SendMessage` on the recorded id, not cold respawns.
  - **Gate sequencing** — the strongest signal in the probe: round 2 opened by **self-diagnosing** that round 1 had drawn one gate over two units, naming the rule it broke and giving the surviving why ("leaves U4's delivered core cascade sitting ungated and uncommitted"). That is the exact clause left standing after the K-026 narrative was cut — the compressed form is load-bearing and working.
  - **Step-table sizing** — 7 units split by layer, no landing-wide brief. **Serialization** — U4/U5 scope-disjoint *and* sequenced; it independently caught that both would exercise the shared `reference` graph and wrote the re-seed recovery into every brief (the shared-key rule generalizing correctly without its `ws:test` pointer).
  - **CPG freshness** — recovered from the failed `cpg_<component>` first-guess (`falkor-chat` → the real key is `cpg_falkorchat`), ran the recipe, and reached *stale* on correct evidence: built `2026-08-17T00:40:42Z`, `sourceCommit` null, 10 commits to `falkor-chat/server` since. Matched independently-established ground truth exactly, and it propagated "read the tree directly, do not consult it" into the briefs.
  - **Mutation-testing** — reproduced the surviving why nearly verbatim, plus the no-`checkout`/`stash`-to-revert prohibition *inside* an implementer brief (the tree-mutation rule binding what teco asks of others).
  - **Model routing** — flagged the closeout unit "inherited, NOT `haiku` — this deliverable summarizes numbers and logic", carried the observed-figures clause verbatim, and added "routing a unit cheap never saves the verification".
  - **Milestone-close flip** (rewritten this pass) — kept for itself rather than dispatched, on the guard's auto-allow; and it correctly ruled that dropping "proposed" from a `Tracks:` field is *more* than a `Status:` flip and so falls outside the auto-allow.
  - **Two gates**, **stop-and-ask** (fork template folded into every brief, its qualifying example *adapted* to this task's real fork), **doc-impact scan**, **`tico` not a live-Q&A delegate**, **drift reported not chased** — all fired.
- **The commit grant — the clause `cobb` caught pass 1 narrowing — is behaviorally confirmed, not merely re-read.** Unprompted, it reconstructed both layers correctly: "every one of U1–U9 runs as a delegated subagent, and the universal grant is void as a delegated subagent; my own grant is the role-based one … whether running interactively or as a delegated subagent." It held the never-list (`push`/`checkout`/`restore`/`stash`/`reset`), committed only by explicit path, and drew the right consequence — that it therefore *cannot* clean up a stray delegate edit, which is the one class of mess that reaches the stakeholder.
- **Two behaviors beyond the watch list, worth recording:** it corrected a stale factual claim in its own brief by re-checking `git status` (the "an incoming message's factual state claims are not authoritative" rule), and it challenged the premise of the resume — M5 cannot close with K-050 Stages 4–6 unbuilt, citing the backlog's own done-condition — instead of playing along.
- **One minor, NOT attributable to this pass — new watch item.** The brief's freshness clause read "CPG `cpg_falkorchat` is stale (built 2026-08-17, …)" rather than the canonical `CPG: <graph>, built <builtAt>, stale — <reason>`. The verbatim forms are intact in the prompt and nothing pass 1 cut touched them; the coordination doc's own Context line was much closer to canonical. Logged because the forms are a cross-document interface (`docs/plans/cpg-agent-adoption.md` §3): watch whether real dispatches paraphrase them too, and if so fix it as its own unit — not by re-expanding pass 1's cuts.
- **Consequence:** C1 pass 2 (class-7 dedup) is unblocked. Its keep-list is in the pass-1 entry below.

## 2026-08-24 — Prompt-waste C1 pass 1: incident narratives and provenance removed (5,948 → 5,728 w)
- **What:** Unit C1 pass 1 of `claude/docs/plans/prompt-waste-reduction.md` (v4, §3 doctrine). This file is the team's largest prompt and is split into two passes by the plan's >30%-cut rule: **pass 1 (this one) removes unambiguous class-5 incident narratives and class-6 provenance/governance only**; class-7 dedup and tightening are pass 2, gated on this pass's observation window closing clean. 15 edits, no rule touched. The modest word delta is expected — teco's weight is long-form *rules*, and that is pass-2 scope.
- **Removed (class 5/6, already on record).** Each item verified present in this file before the prompt edit:
  - Gate-sequencing rule: "(the K-026 pauses left four units in that state; `falkor-chat/docs/plans/graphrag-eval-coordination.md`)" — 2026-08-21 optimization-pass entry, "Gate-as-you-go".
  - Mutation-test bullet: "this check has caught real gaps" — bare incident assertion, no behavior.
  - Serialization rule: "(`kaizen/history.md`, 2026-08-16 entry)" — distillation entry, item 3.
  - `subagent_type` rule: the "Confirmed 2026-08-21: two dispatches (a `cobb` design pass, an `analyst` review gate) ran unhooked… caught only by the UI showing 'general-purpose'" narrative + its pointer, and "Since 2026-08-21" on the backstop hook — 2026-08-21 optimization-pass entry.
  - Step-table sizing rule: "standing user directive" (authority citation) and "Origin and cost data (a whole-landing dispatch that ran 458k tokens/222 tool calls and silently dropped scope): `kaizen/history.md`, 2026-08-11 entry (K-042 Landing 1)" — 2026-08-11 entry.
  - Model routing: "Cheap-model doc-closeouts have twice returned confident, fabricated numbers" — 2026-08-11 entry, "two proven failure modes".
  - CPG-freshness bullet: "(stakeholder decision, 2026-08-19)", "an accepted trade-off for a leaner per-agent prompt", and "`mcp__cypher__query` was live-verified 2026-08-21 (fresh-session probe; see Guardrails)" — 2026-08-19 and 2026-08-21 entries.
  - Stale-`<result>` rule: "(`kaizen/history.md`, 2026-08-17 entry)" — distillation entry, item 4.
  - `SendMessage` fallback: "(`ListAgents` was probed absent at runtime and dropped from the frontmatter 2026-08-21)" — 2026-08-21 entry (K-012).
  - Milestone-close flip: "Since 2026-08-21 (stakeholder decision)" and "so a close no longer dispatches one agent per one-token edit (the prior shape, one full spawn per file)" — 2026-08-21 entry, status-flip carve-out.
  - Self-modification rule: "(`kaizen/history.md`, 2026-08-20 entry)" — distillation entry, item 6.
  - Commit-grant boundary bullet: "Stakeholder decision, 2026-07-30", the quoted stakeholder line ("tico and teco are special…"), and the "**Superseded in part, 2026-08-21:**" framing — 2026-07-30 and 2026-08-21 entries.
  - "Trust only probed tool grants": the fresh-session-probe narrative and the declared-but-absent/dropped account — 2026-08-21 entry (K-012).
  - Brand-new-mechanism rule: "(`kaizen/history.md`, 2026-08-18 entry)" — distillation entry, item 5.
  - Two-gates rule: "Skipping either has let real blockers through" — bare incident assertion.
- **Reclassified up, not deleted (class 3/4 — kept):** the `SendMessage`-resume-inherits-the-wrong-identity consequence (it is why prevention must happen at dispatch, and cannot be corrected later); "silently drops scope", folded into the step-table rule's own clause as its ≤1-clause why; the fabricated-breakdown risk, restated absolutely ("a cheap model will otherwise return a confident, fabricated breakdown") instead of as an incident count; the harness's bypass/self-modification classifier rationale.
- **Correctness fix inside an edited sentence:** the commit-grant bullet described the `Write`/`Edit` guard as reaching "the coordination doc and your own inbox" — stale since `kaizen/inbox.md` was deleted 2026-08-21. Corrected to "`docs/plans/*` only", verified against `hooks/guard-coordination-doc-writes.sh`.
- **Gate (a) inventory — all preserved:** routing table (13 rows) and handoff contracts; the 6-step workflow; the coordination-doc trigger (≥3 units / any review gate / any stop-and-ask) and its reactive-backfill rule; the ledger table and all 7 `Status` values incl. `paused`'s Deliverable-column repurposing; the verbatim high-stakes-fork brief block and its mechanical-unit exemption; mutation-test, QA-defect-brief, serialization/shared-key, `subagent_type`, step-table sizing (~3 steps / ~5 files), `agentId`-at-dispatch, `model: "haiku"` routing + the observed-figures clause, fencing note, never-a-tree-mutating-git-command-in-a-brief; the three `CPG: <graph>…` brief forms and the freshness recipe paths; all of step 4's in-flight rules and step 5's integration checks; the documentation-curation duties incl. the unfiltered rename sweep; Pause-vs-proceed's dual mode; every Guardrail incl. the commit grant's never-list, the two-gates default routing, and the `Learning capture` Cypher template. Audit tokens: `git add`/`git commit` ×1, "delegated subagent" ×2, all 13 agent names.
- **Knowing call, recorded so a later breakage attribution doesn't re-derive it:** the `subagent_type` narrative also carried a detection cue ("caught only by the UI showing 'general-purpose' instead of the named agent"), which is arguably class 4. Cut deliberately: `guard-agent-dispatch.sh` now *prevents* the missing-field case deterministically, so a cue for a case a hook blocks is class 5. If a `subagent_type` omission ever lands silently again, this is the clause to reinstate first.
- **Verified:** `audit-team.sh` PASS (exit 0, all 13 check-8 lines green). `cobb` §7 lint: **pass with findings — one major, four minor, no blocker**; all five fixed before commit, then re-verified.
  - **Major (a real regression this pass introduced, now fixed):** rewording the commit-grant bullet changed "every agent now separately carries a narrower universal interactive-mode grant" into "**Every other** agent carries only…", which silently wrote *teco itself* out of the universal grant. That grant is teco's only clean basis for committing the coordination doc it authors — the role grant at Guardrails covers a *specialist's returned deliverable*, not teco's own artifact — while step 5 still says "commit what you verified". It also contradicted `claude/AGENTS.md` ("Git-commit authority"), which puts all 13 agents in that layer and states it "does not touch" teco's broader grant. Restored as "Every agent, you included, *also* carries…".
  - **Doctrine calibration — the lesson of this pass:** the one defect came **not from deleting a story but from rewording a rule while deleting the story attached to it**. Class-5/6 excision is safe; the prose repair around the hole is where scope moves. On the remaining Stage C units, re-read every reworded sentence as a *rule diff*, not just a length diff.
  - **Minors, all fixed:** "(which reaches `docs/plans/*` only)" handed teco a wider mechanical permission than its own behavioral rule at Guardrails → "(which reaches only your coordination doc's directory, `docs/plans/*`)". Bullet heading "Trust only probed tool grants" left "probed" an orphaned term once the probe narrative went → "Trust only tool grants you've verified". "— not a ceremony trade-off" was a bare negation with nothing to contrast → "— a hard boundary, not a ceremony trade-off". (Fourth minor was the line-130 wording, folded into the major above.)
  - **Cobb's residue scan:** zero remaining inline provenance dates, authority markers, supersession trails, or `kaizen/history.md` pointers. The two surviving date-shaped strings are correct keeps — the `U3` example ledger row (class-2 template) and the fencing note's `kaizen/history.md` (normative citation).
- **Pass-2 keep-list (cobb, load-bearing — do not dedup):** the high-stakes-fork blockquote *including both worked examples* (it is a verbatim brief template, not prose); the three `CPG:` forms; the `Status` list + `paused`-repurposes-Deliverable rule + the `U3` example row; all 13 agent names in the routing table (`audit-team.sh:118`); the check-8 tokens (a dedup pass would plausibly take all three "delegated subagent" restatements); the enumerated `stash`/`checkout <path>`/`restore` never-list *and* "including inside an implementer's brief"; the two-hop `SendMessage` chain at step 4 (a distinct mechanism, not a restatement of the resume rule below it); and, of the three `AskUserQuestion`-withheld-from-subagents statements, keep the one in the tool-trust guardrail — there it is a tool-availability fact, not a duplicate of the pause protocol.

## 2026-08-23 — Prompt-waste Stage B wave 2: learning-capture block compressed to pilot shape
- **What:** Learning-capture intro and tail compressed to the pilot-validated wording (`claude/docs/plans/prompt-waste-reduction.md` v4, §3 doctrine + Stage B). Only this block — the broad mode-unconditioned commit-grant paragraph and the centralized CPG-freshness duty are *not* the shared boilerplate shape and stay for Stage C1.
- **Removed (class 5/6, already on record):** the tail's inbox-replacement sentence ("This replaces the earlier `kaizen/inbox.md`-append convention…") and ", exactly like the old inbox was" — this file's 2026-08-21 inbox-deletion entry; the intro's ":Agent node it's `PRODUCED`-linked to" mechanics restatement — the mechanics live in the Cypher template directly below.
- **Gate (a) inventory — all preserved:** capture trigger (durable, non-obvious fact in discipline), full Cypher template + `mcp__cypher__query` call line verbatim, "skip task-specific details and anything already documented", "raw capture: `cobb` reads/verifies/promotes; never edit your own agent definition".
- **Verified:** `audit-team.sh` PASS; cobb §7 lint pass.

## 2026-08-23 — Distillation: producer-write "RETURN clause" rejection (entryId `9a1c7d2e-4b8f-4e10-9c3a-1f6b2d8e5a77`) — promoted, not a bug
- **What:** `cobb` reviewed teco's kaizen entry reporting that every attempted producer-write
  variant against `kaizen_team` was rejected with the generic FR-8 message. Root-caused by reading
  `cypher-mcp/server.py`'s `_producer_write_agent_id()`/`_PRODUCER_WRITE_TRAILER_RE` and confirming
  empirically (`.venv/bin/python` against `server.authorize_write()` directly): the producer-write
  recognizer requires the statement to end immediately after the `KaizenEntry` map's closing
  `}`/`)` — **no trailing `RETURN` or any other clause** — per `docs/plans/kaizen-agent-ontology.md`
  §3.1 step 2e, explicitly flagged there as *intentionally strict*, not a defect
  ("known future-extension seam"), and already pinned by
  `cypher-mcp/tests/test_server.py::test_producer_write_with_trailing_extra_clause_is_rejected`.
  Every one of teco's reported attempts that included `RETURN k.entryId` was rejected for exactly
  this reason; the canonical shape (no `RETURN`, as it appears verbatim in every one of the 13
  agents' own "Learning capture" sections, including teco's own) is unaffected and was confirmed
  authorized. The asymmetric trap — the *legacy* author-write shape tolerates a trailing `RETURN`
  fine, only the newer producer-write shape doesn't — was not previously called out anywhere an
  agent would see it in the moment of calling the tool, so `cobb` added an explicit "Gotcha"
  callout to `cypher-mcp/README.md`'s producer-write section (right after the worked example)
  naming this exact trap and the live 2026-08-23 date it was hit, plus the workaround (issue a
  separate follow-up read for the `entryId` instead of appending `RETURN`).
- **Why:** confirms this specific wall was caller-side (a natural instinct to append `RETURN` for
  write confirmation, absent from the documented recipe) rather than a server regression — closes
  teco's kaizen entry with a verified root cause instead of leaving it open.
- **Also noted, not yet routed:** while diagnosing, `cobb` found the one curator-clear shape
  (`_CURATOR_CLEAR_RE`) is similarly whitespace-strict — it requires a literal space after
  `entryId:` (`entryId: '...'`, not `entryId:'...'`) and rejects the no-space form with the same
  generic message. A `cypher-mcp` code-side improvement (a clearer near-miss hint in
  `authorize_write()`'s fallback message, and/or a regression test naming the `RETURN` trap
  specifically) was identified as worth doing but was **not implemented by `cobb`** — the
  maintainer flagged mid-session that a Python source fix belongs to whoever owns `cypher-mcp`
  day-to-day, not to cobb's remit (agent/skill/prompt/hook standards). Left as a recommendation for
  `devops` (or a `teco`-coordinated unit) rather than actioned here.

## 2026-08-21 — Mid-run escalation: delegates can stop on a high-stakes fork and be resumed, instead of guessing (mid-run-escalation FR-1..FR-5)
- **What:** Three edits to `teco.md`, applied per `claude/docs/plans/mid-run-escalation.md` §2
  (analyst-reviewed, verdict approve with suggestions — `claude/docs/reviews/mid-run-escalation.md`
  — Findings 2-4 folded in during this implementation):
  1. **Step 2** — the ledger `Status` enum gains a new value, **`paused`** (a unit whose delegate
     stopped mid-run with an open question, now relayed and awaiting an answer); a `paused` row
     repurposes the `Deliverable` column to carry the question + relay date instead of a path
     (noted explicitly as the one `Status` value that breaks that column's normal path-typed
     convention — Finding 4), with a full seven-column example row added next to the existing
     sample. The "open a coordination doc" trigger gains a third, **reactive** condition: any unit
     that escalates via stop-and-ask forces a coordination doc into existence (backfilling a ledger
     row per already-dispatched unit) even below the 3-unit/gate threshold, since a paused unit's
     `agentId` and question must survive a compaction and only the ledger — not context — persists
     that.
  2. **Step 3** — the Subagent-awareness bullet now carves out one narrow exception to "cannot ask
     mid-run": a **high-stakes fork** (would change scope, touch something irreversible, or waste
     substantial downstream work if guessed wrong) may be stopped on and returned as the unit's
     result instead of guessed or held for the final report. The brief clause to fold into every
     dispatch (with a qualifying/non-qualifying worked example) is now scoped — skipped for a unit
     already classified **mechanical** per Model routing, since a mechanical dispatch structurally
     cannot hit a fork worth stopping for (Finding 3).
  3. **Step 4** — four new bullets: recognizing a paused result by shape and relaying it (first-order
     via `AskUserQuestion`, or in-report as a subagent — including the two-hop `SendMessage` chain
     this implies when teco itself is a delegated subagent: its own dispatcher must resume
     teco-as-subagent first, before teco-as-subagent can perform the inner resume — Finding 2);
     resuming the same delegate via `SendMessage` by its ledger `agentId` once answered (with the
     existing step-5 addressing-failure fallback cross-referenced); the non-blocking guarantee (a
     `paused` unit stalls only itself and its structural dependents, no cap/deadline/auto-escalation
     — deliberate, per the requirements doc); no fixed cap on stop-and-ask round trips per unit.
  Also updated: `claude/README.md`'s `teco` catalog entry (one clause describing the capability,
  inserted after the existing `SendMessage`/`agentId` sentence).
- **Why:** `claude/docs/requirements/mid-run-escalation.md` (Ready for design, confirmed
  2026-08-21) — the stakeholder wanted to relax the standing "no mid-run questions" rule for
  genuinely high-stakes forks now that `SendMessage`-based resume is proven (K-007, K-013), so an
  undecided fork doesn't get guessed into a deliverable or only surface after the fact. Designed by
  `cobb` per `claude/AGENTS.md`'s routing convention (agent/prompt engineering, not a codebase
  change); gated by `analyst` (approve with suggestions, no blocker) before this implementation.
- **Verified:** Read-through of the three landed `teco.md` edits against each of AC-1..AC-5 and
  against the plan's own §7 mapping; no hook, frontmatter, or tool-grant file touched anywhere in
  this change (confirmed by inspection — `claude/AGENTS.md` needed no edit, per the plan's own
  scope discipline). No automated suite covers prompt text; a live dry-run exercise of the actual
  relay/resume path is still a follow-up, not performed in this pass.
- **Plan items:** none opened.

## 2026-08-21 — Commit-authority note updated: universal interactive-mode grant supersedes "not extended" claim in part
- **What:** The Guardrails "Why the boundary differs from `tico`'s" bullet now (a) states
  explicitly that both teco's integrator grant and tico's own-doc grant are **unconditioned on
  interactive-vs-subagent mode** (they apply either way, tied to role not invocation), and (b)
  corrects the 2026-07-30 "not extended to any other specialist" claim, which the 2026-08-21
  universal grant (below) partially supersedes: every agent now separately carries a narrower
  interactive-only commit grant for its own verified work, void as a delegated subagent — teco's
  and tico's broader, mode-unconditioned grants are unaffected and remain the only ones of that
  shape.
- **Why:** `tico` reported (via a `kaizen_team` entry) that it lacked commit authority over
  subagent deliverables from a Mode-3 verification pass it orchestrated; the stakeholder, put on
  the spot for a decision, ruled beyond that narrow case — every agent gets an interactive-mode
  commit exception, not just tico/teco. Full rationale, the `claude/AGENTS.md` rewrite, and the
  `audit-team.sh` check-8 redesign: `claude/cobb/kaizen/history.md`, 2026-08-21 entry.
- **Verified:** `bash claude/scripts/audit-team.sh` — clean, all 13 agents pass check 8.
- **Plan items:** none opened — direct implementation of an explicit stakeholder decision.

## 2026-08-21 — K-015 ✅ closed: dispatch-sizing rule validated on K-028's real oversized implementation

- **What:** K-028 (falkor-chat workflow timers) supplied the first live instance crossing the
  ~3-step/5-file boundary the 2026-08-11 sizing rule targets — its implementation touched
  `services.py`, `repository.py`, `schemas.py`, `api.py`, `config.py`, `app.py`, `executor.py`,
  plus `QUERIES.md`/`DESIGN.md`/`start_server.sh` and 5 test files (15+ files total), and hit a
  plan-level defect mid-implementation forcing a full mechanism redesign. teco did not hand this
  to one mega-dispatch: it split the implementation into named ledger units by concern —
  **U3a** ("core logic": `services.py`'s sweep/invariant, `executor.py` docstring) vs. **U3b**
  ("wiring": `schemas.py`/`api.py`/`config.py`/`app.py` + docs + tests) — and tracked the
  defect-driven rework as its own distinct rows (**U3a-fix**, **U3c**) rather than silently
  absorbing it back into a growing single dispatch. Per-unit costs stayed well inside the K-042
  baseline (458k tok/222 tools) despite the mid-run redesign: U1 (plan) 212k/42, U2 (plan gate)
  164k/49, **U3a (core logic, the largest single unit) 307k/134**, U5 (QA) 175k/59 — no unit came
  close to K-042's mega-dispatch cost, and no scope was silently dropped (QA: PASS, zero defects,
  12/12 planned items).
- **Why:** K-015 tracked the rule as unproven since 2026-08-11 — "a claim, same epistemic shape as
  K-013's unexercised `SendMessage` loop" — because every prior coordination's dispatches (K-026
  included) had stayed single-unit and never actually crossed the boundary that would exercise it.
- **Disposition:** ✅ confirmed — the rule holds under real, defect-heavy pressure, not just in
  the prompt. One refinement worth carrying into K-016's consolidation pass: the split that
  actually happened here was **by logical concern** ("core logic" vs. "wiring"), not a literal
  file-count tally against the ~3-step/5-file threshold at decomposition time — the concern-based
  split happened to keep every unit's footprint far below the threshold anyway. The rule's intent
  (bounded per-unit cost, no dropped scope) was met; its mechanism, as actually practiced, is
  closer to "cluster by concern" than "count files." Consider restating it that way if K-016
  touches this bullet.
- **Left open:** the parking-lot idea "architect plans annotate dispatch-unit boundaries" was
  *not* exercised here — architect's plan didn't pre-mark U3a/U3b clusters; teco derived the split
  itself at dispatch time. Still open, still worth raising when K-016 or that item is worked.
- **Plan items:** K-015 (✅ done, moved out of Active).

## 2026-08-21 — Optimization pass from a stakeholder-requested in-depth analysis: dispatch guard hook, status-flip carve-out, tool-grant reconciliation (K-012 ✅), Cost ledger column, gate-as-you-go, AskUserQuestion dual-mode

- **What:** seven changes from a stakeholder-requested "analyze teco in depth / optimize its way
  of work" session, three of them stakeholder-decided explicitly (status flips, AskUserQuestion,
  consolidation timing).
  1. **New `PreToolUse` hook `hooks/guard-agent-dispatch.sh` (matcher `Agent|Task`)** — escalates
     any `Agent` dispatch missing `subagent_type` to the human. The 2026-08-21 silent-
     `general-purpose` trap already had a prompt bullet, but prompt discipline alone had let two
     such dispatches through; this makes the omission mechanically impossible to land silently.
     Standalone agent-owned script (the `security-expert` exploitation-guard precedent), fail-open,
     `ask`-only, jq→python3. Tested through the deployment symlink: present/missing/empty
     `subagent_type` and garbage input all behave per contract.
  2. **Status-flip carve-out in `guard-coordination-doc-writes.sh` (stakeholder-approved):**
     before deferring to the shared core, the wrapper auto-allows an `Edit` on a `docs/**.md`
     whose old/new strings differ only in the canonical `**Status:**` field flipping to
     `archived` (python3 masks the field on both strings and requires byte-equality of the rest).
     Rationale: the K-026 close spent **five separate agent spawns** on one-token archival flips.
     Root `AGENTS.md`'s lifecycle section now names `teco` as the performer of the mechanical
     flip (by-kind owner table retained for anything beyond it); teco.md's milestone-close bullet
     and Guardrails updated to match. Tested: pure flip (relative + absolute path) → allow;
     flip+other change, flip to a non-`archived` token, non-docs path, `Write` → escalate.
  3. **Tool-grant reconciliation (K-012 ✅ closed).** Fresh-session probe (spawned teco run):
     runtime tools are exactly `Read, Bash, Agent, SendMessage, Write, Edit, mcp__cypher__query`.
     `ListAgents` absent (like the known `Grep`/`Glob` gap) — all three dropped from frontmatter
     per K-012's own pre-agreed disposition. `mcp__cypher__query` **live-verified** (a
     `kaizen_team` read returned; parking-lot item resolved) — the "not yet live-verified"
     caveats on the CPG-freshness duty removed. Guardrails' runtime-tool-set bullet rewritten:
     frontmatter now matches probed reality; the "a grant is not proof" lesson retained.
  4. **Ledger gains a `Cost` column** (step 2): record the completion notification's reported
     tokens/tool-uses per unit — the data K-015 (dispatch-sizing validation) needs to ever be
     judged against numbers. Feasibility confirmed this session: the probe's own completion
     notification carried `30982 tokens / 1 tool use`.
  5. **Gate-as-you-go** (step 2): sequence each unit's review gate immediately after its
     delivery, never batched at coordination close — K-026's pauses left four delivered units
     ungated and uncommitted across sessions, exactly the crash-exposure this rule shrinks.
  6. **AskUserQuestion dual-mode (stakeholder-approved):** frontmatter gains `AskUserQuestion`;
     Pause-vs-proceed now distinguishes first-order runs (`claude --agent teco` — ask the
     decision as a structured question and continue) from subagent runs (tool withheld by the
     harness — return the decision summary as before). K-026 showed teco frequently runs
     first-order, where every decision point previously cost a full stop-and-report.
  7. **Stale-text fixes:** Guardrails' "and your own kaizen inbox" clause dropped (inbox deleted
     2026-08-21); step 5's "don't assume an enumeration tool" hedge replaced with the probed
     fact. (The hook wrapper's own stale inbox glob/comment turned out to be already fixed by a
     concurrent edit outside this session — found mid-change when the file differed from its
     first read.)
- **Deferred by stakeholder decision:** the full consolidation/KB-split pass filed as **K-016**
  (high) for a dedicated `cobb` pass rather than run in the same session that touched the prompt
  in seven places. Two new parking-lot ideas: architect-side dispatch-boundary annotations
  (feeds K-015), cross-session slug-echo convention.
- **Why:** stakeholder asked for an in-depth analysis of teco and then "let's get to work" on its
  findings. The through-line: promote prompt-level discipline to mechanical enforcement where it
  has already failed (1), stop paying agent spawns for mechanically-verifiable edits (2), align
  declared capability with probed reality (3), and attack the documented credit-burn pattern with
  data capture + smaller exposure windows (4, 5).
- **Verified:** `bash claude/scripts/audit-team.sh` — full PASS before and after (diff, not bare
  gate). Both hook scripts `bash -n` clean and scenario-tested through `~/.claude/agents/teco/`.
- **Docs touched:** `claude/teco/teco.md` · `claude/teco/hooks/guard-agent-dispatch.sh` (new) ·
  `claude/teco/hooks/guard-coordination-doc-writes.sh` · root `AGENTS.md` (lifecycle/flip
  authority) · `claude/AGENTS.md` (hook machinery) · `claude/README.md` (teco row + deployment
  hooks note) · `claude/teco/kaizen/{plan,history}.md`.
- **Plan items:** K-012 ✅ (moved here) · parking-lot `mcp__cypher__query` probe ✅ · K-015
  updated (Cost column feeds it) · K-016 opened · two parking-lot ideas added.

## 2026-08-21 — `kaizen/inbox.md` deleted (content already fully captured elsewhere)

- **What:** `cobb` deleted this agent's frozen `kaizen/inbox.md` (git history retains it in full, unaltered) as part of a team-wide cleanup of all 12 agents' frozen inboxes. In the same session, `kaizen_teco` (the per-agent graph this file's own 2026-08-20 entry below describes) was also retired — `graph-dba` `GRAPH.DELETE`d it after cross-checking all 5 of its entries against this file's 2026-08-21 distillation entry (below), all already promoted or kept-open-as-`K-018`-and-cleared.
- **Why:** user-directed — "no point keeping [it] since it's already git history." Verified lossless first: `kaizen_team` (the shared graph every agent's raw capture routes through since 2026-08-20) was confirmed completely empty before any deletion — every entry any agent ever wrote there (including this agent's own 9-entry distillation, immediately below) has already been distilled and cleared. Full rationale and verification method: `claude/cobb/kaizen/history.md`, 2026-08-21 entry.
- **Verified:** see `cobb`'s entry (cross-agent verification, not repeated per file).
- **Plan items:** none opened — pure cleanup, no behavior change.

## 2026-08-21 — Coverage fix: dropped stale "kaizen-inbox entry" from the commit-authority grant (team certification, §7 fold-in)

- **What:** The commit-authority grant ("The grant" bullet under Guardrails' `Bash` section)
  listed "a plan, review, test plan/report, or kaizen-inbox entry you've verified fits" as the
  deliverable kinds teco may commit for a coordinated specialist. Dropped "or kaizen-inbox entry."
- **Why:** Caught during a user-requested full team-coherence certification's §7 lint fold-in.
  Since the 2026-08-20 team-wide graph migration, no agent produces a fresh `kaizen/inbox.md`
  entry any more — every agent's raw learnings capture writes straight into the shared
  `kaizen_team` graph via `mcp__cypher__query`, not a file. A specialist teco coordinates can
  therefore never hand back a kaizen-inbox-entry deliverable for teco to commit; the clause
  described a delivery shape that no longer exists. Same stale-phrase pattern found and fixed the
  same pass in `tico.md`'s commit-authority grant (`claude/tico/kaizen/history.md`, same date).
- **Verified:** `bash claude/scripts/audit-team.sh` — same 113 PASS / 2 pre-existing FAILs before
  and after (diff, not bare gate).
- **Plan items:** none opened — direct fix from a live certification finding.

## 2026-08-21 — Distilled all 9 pending raw-capture entries from `kaizen_team` (`cobb`, §5 pass)

- **What:** `cobb` ran the full agent-maintenance §5 distillation against every `kaizen_team`
  node with `author:'teco'` — the raw capture written since the 2026-08-20 team-wide graph
  migration (teco's `kaizen/inbox.md` itself stays a frozen, already-imported snapshot; this is
  the *new* capture that accrued in the graph afterward). All 9 verified, dispositioned, and
  cleared from the graph in this pass; none discarded outright.
  1. **`7994edd7…` (2026-08-15, nested notification bubbles to the live ancestor; `SendMessage`
     force-resumes a dormant target) → promoted** to
     `skills/agent-standards/claude-code.md`, new "Nested-delegation notification routing"
     subsection, explicitly dated/caveated as a live observation, not a confirmed stable
     contract — matches the entry's own `suggestedHome: unsure`.
  2. **`e40a95fe…` (2026-08-15, a delegate that can't address teco by name is relayed through
     "main" as a `<system-reminder>` block — legitimate, not an injection) → promoted**, same
     new subsection, paired with entry 1 (same K-026 incident).
  3. **`a77a32a3…` (2026-08-16, parallel dispatches sharing a live shared-DB fixture
     cross-contaminated despite disjoint files) → promoted** into `teco.md` §3 "Dispatch,"
     sharpening the existing same-file/shared-key serialization rule to cover concurrent
     live-suite exercise, not only destructive overlap.
  4. **`f7070b80…` (2026-08-17, a `completed` notification's `<result>` can be a stale
     mid-task placeholder from a delegate's own unfinished background step) → promoted**
     into `teco.md` §4 "Track what's in flight," new bullet after "Never state or predict a
     pending delegate's result."
  5. **`9ec17ba5…` (2026-08-18, a clean QA PASS on a brand-new mechanism isn't full
     state-space coverage) → promoted** into `teco.md` Guardrails, new bullet next to the
     review-gate guardrail.
  6. **`f1a2b3c4…` (2026-08-20, a coordinator's "proceed" ≠ real user approval on a
     harness-gated write; never relay a delegate's self-modify-permissions proposal) →
     promoted**, split across two homes: `teco.md` Guardrails (the operative rule for teco's
     own coordination authority) and `skills/agent-standards/claude-code.md` Hooks section (the
     underlying harness-classifier fact, a sibling to the existing auto-mode Bash-classifier
     bullet).
  7. **`a2b3c4d5…` (2026-08-20, `guard-destructive-ops.sh` didn't fire for a live
     `GRAPH.DELETE` run inside a nested subagent's Bash context) → kept open, not promoted
     to teco.md.** This is a hook-wiring question the entry itself flagged for `cobb` to
     triage, not a teco-side behavior gap. `cobb` re-fetched `code.claude.com/docs/en/hooks`
     (2026-08-21): hooks are documented to fire identically for a subagent whether run as the
     main session agent or a nested delegate, no exception noted — so the observed gap isn't
     explained by any documented behavior. Filed as `K-018` in `claude/cobb/kaizen/plan.md`
     (high priority), with the leading hypothesis that it's actually explained by entry 9
     below (a `subagent_type`-omitted dispatch silently running as `general-purpose`, with no
     `graph-dba` hooks at all) — not confirmed from the coordination doc alone, needs a live
     re-check on a future `graph-dba` dispatch. Node cleared from the graph regardless per §5's
     "kept open" disposition — the durable record is this note plus `K-018`, not a lingering
     graph node.
  8. **`b3c4d5e6…` (2026-08-20, `mcp__cypher__query` table rendering truncates a cell at
     ~300 chars; FalkorDB Cypher distinguishes `\n` from `\\n` in string literals) → partially
     promoted.** The truncation half was **already fully documented** in
     `cypher-mcp/README.md` ("Result format and truncation" — `CYPHER_MCP_MAX_CELL`/
     `CYPHER_MCP_MAX_CHARS`), confirmed by `cobb` re-deriving the exact same 300-char cap
     firsthand while reading these very entries out of the graph, independent of the entry's
     own claim — discarded as a duplicate. The `\n`/`\\n` escaping half was genuinely
     undocumented — promoted to the same README, "Writing through this tool" section.
  9. **`b1e3a1f0…` (2026-08-21, `Agent` silently defaults to `subagent_type: general-purpose`
     when omitted — no error, no hooks/persona/tools for the named agent) → promoted**, high
     priority, into `teco.md` Guardrails as a new bullet next to the existing
     narrower-than-frontmatter runtime-tool-set warning — same "don't trust the frontmatter/the
     brief alone" family. Cross-referenced by entry 7's `K-018` as the likely (unconfirmed)
     root cause of that separate incident.
- **Why:** user asked to "work on teco's inbox" — the file `kaizen/inbox.md` is itself a frozen,
  already-distilled 2026-08-20 snapshot, so the live equivalent is teco's pending raw capture in
  the shared `kaizen_team` graph; first distillation pass against it since the migration.
- **Verified:** every entry's full field text was read via `size()` + multi-column `substring()`
  paging (the harness's own per-cell display truncates at ~300 chars) before any disposition
  decision, not acted on from a truncated partial read; the one live-checkable external claim
  (hook parity for nested subagents, entry 7) was re-checked against current official docs rather
  than taken on the entry's own framing.
- **Plan items:** none opened in `teco/kaizen/plan.md` itself (K-018 opened in `cobb`'s own
  plan.md instead, per its "flagged for cobb" suggested home).
- **Docs touched:** `claude/teco/teco.md`, `claude/teco/kaizen/history.md` (this entry),
  `skills/agent-standards/claude-code.md`, `cypher-mcp/README.md`, `claude/cobb/kaizen/
  {plan,history}.md` (cross-artifact bookkeeping, logged from `cobb`'s side).

## 2026-08-20 — Learnings capture migrated to a working-memory graph (`kaizen_teco`), mirroring `graph-dba`
- **What:** The "Learning capture" closing-protocol section now writes a `:KaizenEntry` node
  directly into `kaizen_teco` (FalkorDB, via `mcp__cypher__query`) instead of appending to
  `kaizen/inbox.md`. `kaizen/inbox.md` is now a frozen historical snapshot — its 5 pre-existing
  entries were parsed out programmatically and imported into the graph verbatim (entryId
  assigned, `author: 'teco'`), preserving every field; its own header explains the freeze and
  gives the live-read query. The trailing "Your write guard allows exactly this inbox path"
  clause was dropped — the write guard gates `Write`/`Edit`, not the `mcp__cypher__query` MCP
  tool, so it no longer applies to this capture path.
- **Why:** User-directed team-wide redesign ("I will migrate all agents to write their learnings
  to the graph like graph-dba"), reversing yesterday's file-based Learning-capture dedup (entry
  below) — the user determined the whole team should follow `graph-dba`'s existing graph-based
  capture pattern instead of the file-based inbox convention.
- **Plan items:** —

## 2026-08-19 — Learning-capture paragraph de-duplicated against the inbox's own header
- **What:** Trimmed the "Learning capture" paragraph: dropped "(fact, evidence, suggested home; format in the file header)" and "The inbox is raw capture — the team maintainer (`cobb`) verifies and promotes entries into prompts, knowledge bases, or project docs" — both already stated verbatim in `kaizen/inbox.md`'s own header template (agent-maintenance skill §5), which the agent necessarily opens to append. Kept: the discipline-specific fact-kind clause, the inbox path, "skip task-specific details," "never edit your own agent definition," and the write-guard clause. Behavior unchanged.
- **Why:** User-directed prompt-verbosity reduction, item 1 of the parked diagnosis (`cobb/kaizen/plan.md`) — the mechanics were literally duplicated (prompt + inbox header say the same thing), not just similar boilerplate; pointing at the file's own header removes the duplication without losing information, since the agent reads that file to act anyway.
- **Plan items:** —

## 2026-08-19 — Step-table sizing rule: incident narrative moved out of operative prompt text
- **What:** The dispatch-sizing bullet (§3, "Delegate with complete briefs") kept its operative rule verbatim (~3-step/5-file decomposition boundary, one unit per step/small cluster) but dropped the inline K-042 incident narrative (the 458k-token/222-tool-call whole-landing dispatch, the dropped-test-files detail, the stakeholder quote) in favor of a dated pointer to this file's own 2026-08-11 entry, which already carries the full story. −~85 words in the prompt body.
- **Why:** User-directed prompt-verbosity reduction, item 2 of the parked diagnosis (`cobb/kaizen/plan.md`) — an origin story belongs in the change log it's already recorded in, not repeated inline in the instruction every session pays to load. The rule itself is unchanged; only the narrative moved.
- **Plan items:** —

## 2026-08-19 — CPG freshness centralized here; `mcp__cypher__query` added
- **What:** Took over the CPG freshness check that `analyst`/`architect`/`coder`/`tdd-engineer`/`frontend-engineer`/`qa-engineer` used to run themselves (`docs/plans/cpg-agent-adoption2.md`, extending the archived `cpg-agent-adoption.md`). Added `mcp__cypher__query` to `tools:`; new §3 bullet — guess the graph key, run the freshness recipe (`skills/cpg-analysis/references/freshness.md`) before dispatching a unit likely to touch a CPG, state the result in the brief. Guardrails note flags the grant as **not yet live-verified** — teco's frontmatter already has a known live-tool-set-narrower-than-declared gap (`Grep`/`Glob`, verified 2026-08-10), so `mcp__cypher__query` needs the same live probe before this duty can be trusted.
- **Why:** User-directed prompt-verbosity reduction surfaced the freshness check as ~130 words duplicated verbatim across six agents; user chose full centralization on teco (accepting the standalone-run capability loss) over a per-agent dedup via a shared skill pointer.
- **Plan items:** new parking-lot item — live-verify the `mcp__cypher__query` grant on a real coordination before relying on the freshness duty.

## 2026-08-16 — K-013 ✅ and K-014 ✅ closed by real evidence from K-026's own coordination (review-only, no prompt change)

- **What:** reviewing `teco/kaizen/plan.md`'s active table against the just-closed K-026
  GraphRAG-eval coordination ledger (`falkor-chat/docs/plans/graphrag-eval-coordination.md`)
  found two of the four open items had actually been exercised, live, during that coordination —
  the datapoints just hadn't been written back yet.
  - **K-013 (exercise `SendMessage` continuation for real) — closed.** The ledger's `U2b-gate`/
    `U2b-fix` rows show `analyst` gated Unit 2b "needs changes" (Blocker B-1, Major M-1) → a fix
    dispatched to `tdd-engineer` → the re-gate row reads `analyst (resume a4b2370c17130742d,
    re-gate)` — `teco` resumed the *same* analyst by its own ledger `agentId` instead of a fresh
    spawn, and that analyst "independently re-verified both fixes itself... re-ran suites...
    checked `GRAPH.LIST` directly for B-1" without being re-briefed on its own earlier findings.
    Real evidence context is preserved across a `SendMessage` resume, not just a claim.
  - **K-014 (agentId ledger cell has no enforcement) — closed, no checker added.** Every one of
    ~20 unit rows in K-026's multi-session coordination has its `agentId` cell filled, with
    exactly one explicitly-justified exception (`U-bug`, inherited from a prior session with
    genuinely no id to carry forward — noted as unresolvable, not silently blank). Self-discipline
    held under sustained real load; per K-014's own stated criterion ("if it doesn't get skipped,
    leave it"), no checker is warranted from this evidence.
  - **K-012 (`ListAgents` materializes) and K-015 (dispatch-sizing rule) — still open.** Neither
    got exercised: nothing in the K-026 coordination invoked `ListAgents`, and every unit
    (including the closing qa-engineer/doc-closeout ones) stayed single-unit, never crossing the
    ~3-step/5-file boundary that would test the sizing rule.
- **Why:** the user asked directly, mid-session, what became of "the teco improvement item we were
  testing" — prompted this reconciliation rather than letting the evidence sit unlogged in a
  component's coordination doc where it would never surface again.
- **Verified:** read the coordination doc's ledger rows directly (not a relayed summary) before
  crediting either closure.
- **Plan items:** K-013 ✅, K-014 ✅ (moved to plan.md's done-notes block); K-012, K-015 remain
  active, both updated with a 2026-08-16 "still no evidence" note.

## 2026-08-16 — Step 4 gains a misrouting-vs-staleness distinction for incoming messages (cobb, cross-session peer-addressing near-miss)

- **What:** one new bullet in step 4 ("Track what's in flight"), right after the existing
  "incoming resume/pause message: intent authoritative, facts aren't" rule: a message describing
  a task/coordination **absent from this session's own ledger or active context** is a
  *misrouting* signal, not merely a staleness one — `SendMessage` addresses peers by bare agent
  name, which resolves ambiguously when more than one independently-launched session shares that
  name, so the sender may simply have the wrong `teco`. Pause and confirm identity with your own
  user before doing *anything* in response, even read-only verification.
- **Why:** a live incident, not a hypothetical — a `teco` session (a different one, mid-coordination
  on `cpg-agent-adoption`) received a full K-026 resume brief from `cobb`, who had picked it off
  `ListAgents` assuming (wrongly, from stale prior-session context) it was the K-026 coordinator.
  That session's own human caught the mismatch; by then it had already done a small amount of safe
  read-only work (a test re-run, a state-restoring reseed) before declining — harmless here, but
  exactly the kind of spend this bullet now heads off explicitly instead of leaving to luck/a human
  catch. Full mechanism writeup lives in `skills/agent-standards/claude-code.md`'s new "Cross-session
  peer addressing" section (`cobb`'s companion promotion, same run).
- **Verified:** `bash claude/scripts/audit-team.sh` clean before and after.
- **Plan items:** none opened — direct fix from a live incident, not a backlog item.

## 2026-08-15 — Review-only pass: filed K-015 (validate the dispatch-sizing rule live)

- **What:** no prompt change. Stakeholder asked what's next for `teco`, specifically flagging
  the "big work packages" episode from the last end-to-end run. Reviewed `plan.md`/`history.md`/
  `inbox.md` (one undistilled entry remains, 2026-08-12 — see below) and confirmed the dispatch-sizing
  rule that answers that exact episode (K-042 Landing 1, 2026-08-11 entry) has shipped in
  `teco.md` §3 but has **zero live-run evidence since** — no coordination has exercised it under
  real conditions. Filed as **K-015** (high priority) in `plan.md`, cross-linked to close
  alongside the still-open **K-012** (`ListAgents` fresh-session probe) and **K-013**
  (`SendMessage` continuation) on the same next live run, since one end-to-end coordination can
  produce evidence for all three at once.
- **Why:** per the kaizen convention, a review-only pass still records ideas surfaced during
  review rather than letting them evaporate at the end of the conversation.
- **Plan items:** K-015 filed (🔵 proposed).

## 2026-08-15 — Distilled the 2026-08-12 inbox entry: discarded from teco (suggested home didn't fit), redirected to `cobb`'s own kaizen

- **What:** ran the agent-maintenance skill §5 procedure on the sole entry in `teco/kaizen/inbox.md`
  (2026-08-12, "a review-gated unit with no coordination doc is nearly unrecoverable after a
  mid-session credit crash").
- **Verified — incident is real, not embellished:** `docs/plans/kaizen-inbox-distillation2-coordination.md`
  (Owner: `teco`, `Tracks: — (no backlog id; stakeholder-triggered cobb sweep)`) and
  `docs/reviews/kaizen-inbox-distillation2.md` confirm it exactly: `cobb` ran a 39-file team-wide
  kaizen distillation directly (not via `teco`), `analyst` gated it "needs changes," the fix pass
  had to be resumed cold (`U1` row: `agentId (prior session, not resumable)`), and the recovering
  session (operating as `teco`) reconstructed state from the review alone, confirmed **K-041**
  already covered one of the review's "open questions" (matches the inbox entry's claim exactly),
  and closed cleanly (commit `db39ade`). The entry's cited filename
  (`kaizen-distillation-2026-08.md`) was the doc's pre-rename name — `analyst`'s own U3 renamed it
  to `kaizen-inbox-distillation2.md`, noted in the coordination doc; not a fabrication, a stale
  filename from before the rename.
- **Why the suggested home ("teco.md step 2/3: open the ledger at first dispatch for any
  review-gated sequence") doesn't fit teco:** that rule **already existed** — step 2 has opened a
  coordination doc whenever *any* unit carries a review gate (not only at the 3-unit threshold)
  since the 2026-08-10 ledger pass, which **predates** this incident. The incident's `U1` (the
  original `cobb` sweep + `analyst` gate) never had a ledger because it was **never coordinated by
  teco at all** — direct stakeholder → `cobb` → `analyst`, by design (`cobb` is meant to be
  directly invokable for agent-maintenance work, per its own `description`). Teco's existing rule
  had no chance to apply; there is no teco-side gap to close.
- **Disposition: discard from teco's inbox, redirect the underlying observation to `cobb`'s own
  kaizen** (`claude/cobb/kaizen/plan.md`, parking lot) as a self-directed note — the actual
  mitigation that saved the recovery (the review's self-sufficient baseline-commit + explicit
  scope list) is already `analyst`'s standing review-header practice, so this is confirmed-good-
  practice-under-fire, not a new rule; logged as a soft parking-lot idea, not a prompt change, per
  §5's "highest bar: every session pays for it" for anything landing in an always-loaded prompt —
  one data point, no repeat, no runtime-behavior gap identified.
- **Plan items:** none advanced in teco's own plan (K-012/013/014/015 unaffected); see
  `claude/cobb/kaizen/{plan,history}.md` for the redirected entry.

## 2026-08-11 — Dispatch-sizing standing rule + 4 smaller promotions from the inbox distillation (stakeholder's "never a landing this big again" directive)

- **What:** `cobb` distilled all six entries in `teco/kaizen/inbox.md` (§5), triggered by a
  stakeholder report of several sessions blowing past 400k tokens. Five promoted into `teco.md`,
  one discarded as already self-resolved.
- **Diagnosis (the actual ask):** verified verbosity vs. orchestration as the cause, not assumed.
  Agent prompt bodies are 42–274 lines (already through two team-wide slimming passes,
  2026-07-11/2026-07-24); `coder.md` is ~1.4k tokens, `teco.md` ~5.7k. The cited incident — K-042
  Landing 1, one `coder` dispatch covering 6 plan steps / ~10 files, **458k tokens / 222 tool
  calls / ~45 min**, per `/context` — is context-length growth from the sheer volume of file
  reads/diffs/test-run output accumulated across one unbroken 6-step, 10-file session, not from a
  large system prompt (which is a small, roughly-constant fraction of that context, not something
  resent 222 times). The same unit's `analyst` gate then found 3 of the 11 test files the plan
  names — 3 of the 5 rewired consumer bindings — silently dropped from its own stated scope — a
  correctness cost, not just a token cost, from the same oversizing. **Verdict: orchestration
  (dispatch sizing), not verbosity, is what's driving these specific blowouts.** No verbosity
  contributor found worth a further slimming pass.
- **Promoted into `teco.md`'s "Delegate with complete briefs" (§3):**
  1. **Dispatch-sizing rule** (the core promotion) — a plan step table spanning more than ~3 steps
     or 5 files is the decomposition boundary: one unit per step/small cluster, sequenced as
     dependent same-file briefs, never one landing-wide mega-brief. Tied explicitly to the
     stakeholder's own words, quoted in the prompt, so the rule can't silently erode.
  2. QA-found-defect fix briefs: read the defect's own root-cause docstring/AC, not just the
     suggested-fix line (a live repro proves one path broken, not the only one).
  3. Documentation-impact scan: for a rename/removal blast radius, sweep unfiltered
     (`grep -rn <token> .`), not `--include='*.ext'` — extension globs silently miss dotfile
     config (`.env.example`).
  4. Track-what's-in-flight: `SendMessage` a premise-invalidating finding to a still-running
     sibling immediately, don't hold it.
  5. Track-what's-in-flight: an incoming resume/pause message's *intent* is authoritative, its
     *factual state claims* are not — re-verify against `git log`/the ledger before acting on them.
- **Discarded:** the "specialist's own knowledge base can be stale in a build-version-specific
  way" entry — already fully self-corrected: `claude/graph-dba/falkordb-quirks.md`'s
  `db.indexes()` entry already carries the "corrects the earlier claim... verified 2026-08-10"
  note the inbox entry was asking for, written by `graph-dba` itself during the same run.
- **Why:** stakeholder-reported context blowouts across recent sessions; the stakeholder's own
  quote ("please never again create a landing so big") had been sitting as an unpromoted inbox
  entry since 2026-08-10 despite being an explicit standing directive.
- **Verified:** `bash claude/scripts/audit-team.sh` clean before and after (diff, not a bare
  gate). No personal identifiers introduced.
- **Bookkeeping note for the next distillation pass:** the inbox held **6 headed (`## `) entries**
  plus one **headless continuation block** — a stray `- **Evidence:**` bullet with no `## ` heading
  of its own, sitting directly under the "K-042 Landing 1... ran past 370k tokens" entry and
  narrating the same unit's *completed* numbers (458k tokens / 222 calls / the dropped-test-files
  finding). Treating it as part of that preceding entry (not a 7th, separately-dispositioned one)
  is correct — don't re-count it as a separate entry in a future pass.
- **Docs touched:** `claude/teco/teco.md` · `claude/teco/kaizen/{history,inbox}.md`.

## 2026-08-10 — Coordination state moves out of the context window: canonical ledger, in-flight tracking, `agentId`-addressed continuation

- **What:** the largest structural pass on `teco.md` since its creation, from a `cobb` review of
  how teco coordinates, tracks units, and routes between *running* and *fresh* agents.
  1. **Frontmatter:** `tools:` gained `ListAgents` — step 5's "fall back to a cold spawn only if
     the identifier no longer resolves" was previously untestable from inside teco.
  2. **Steps 3 and 4 split into sub-bullets** before anything was added to them (two prior
     parking-lot deferrals, §7 dimension 4): each now carries one rule per bullet.
  3. **New step 2 ledger, mandatory at 3+ units or any gated unit** (replacing the unmeasurable
     "for large or long-running work"): `| Unit | Owner | Agent id | Status | Deliverable | Gate →
     verdict |`, with a closed status vocabulary (`queued` · `in-flight` · `delivered` · `gated` ·
     `accepted` · `abandoned`). Stated explicitly: the ledger, not teco's context window, is where
     a unit's state lives.
  4. **New step 1 resume path:** an existing coordination doc for the slug is the state of record —
     read it and reconcile `in-flight` rows against `git log` and the tree before dispatching.
  5. **New step 4, "Track what's in flight":** `Agent` runs in the background by default; never
     state or predict a pending delegate's result; a transient platform failure (500/timeout/kill)
     is not a deficient result but a re-dispatch with a **state-recovery brief** (inspect
     `git status`/`git diff`, continue from actual state); a unit superseded mid-flight is
     `abandoned`, its result discarded.
  6. **Identity recorded at dispatch, always** — every `Agent` call returns an `agentId`; it goes
     in the unit's ledger row immediately, not "when a follow-up seems likely" (which asked teco to
     predict the future). That id is what `SendMessage` addresses; `ListAgents` is how resolution
     is checked.
- **Why (evidence, not opinion):** `SendMessage` appears **42×** across this box's transcripts and
  **0×** in any confirmed teco run — K-007 shipped 2026-07-29 and had never fired. All five real
  coordination docs on disk invent a different Status table and **none** records who is running or
  how to reach them. The delegate id lived only in teco's context, the one thing lost to
  compaction, on exactly the long coordinations where continuation matters.
- **Two proven failure modes turned into rules.** `model:"haiku"` doc-closeouts fabricated numbers
  twice (2026-07-31: a 70% threshold existing nowhere in the codebase; 2026-08-09: a breakdown that
  doesn't arithmetically add up), both caught only by teco's own re-verification. Step 3's model
  routing now confines haiku to **mechanical** units and requires summarizing briefs to carry
  *"state only figures you directly observed in this run's command output; never decompose a total
  into a breakdown you did not observe"*; step 5 pairs it with mandatory re-verification of any
  summarized number — the cheaper model tier never buys out the verification.
- **Five standing practices promoted from the user's AutoMem file into the committed prompt**
  (double analyst gate — plan gate **and** diff-scoped re-gate, replacing the weaker "and/or";
  mutation-testing green-on-arrival tests; shared-file serialization + single ownership of shared
  DB state; independent verification of a self-reported recovery; the tree-mutating-git
  prohibition — `stash`/`checkout <path>`/`restore` — now explicitly binding **inside implementer
  briefs**, with `git show <ref>:<path>` named as the safe baseline read). A live probe confirmed
  **the memory *index* reaches a subagent but not the entry bodies**: teco saw the
  `teco-process-lessons` one-line gloss and nothing behind it, so it had a teaser it could not act
  on. The committed prompt is the right home regardless.
- **Also:** Guardrails' dense commit-authority paragraph split into grant / never-mutate-the-tree /
  boundary-vs-`tico` / no-hook-backstop (the parking-lot note said to split it the next time
  Guardrails gained an addition, and it just did); the milestone-close bullet now states that root
  `AGENTS.md`'s by-kind flip table **controls over a document's own `Owner:`** where they disagree.
- **Verified:** `claude/scripts/audit-team.sh` — 98 PASS / 0 FAIL before, re-run clean after (diff,
  not a bare gate, per `agent-maintenance` §4).
- **Plan items:** closed three parking-lot items (step-3 density, Guardrails bullet split,
  model-routing evidence clause); opened K-012, K-013, K-014.

## 2026-08-10 — Learnings inbox distilled: 5 entries → 2 to teco.md, 2 to project docs, 1 closed by probe

- **What:** processed every pending entry per `agent-maintenance` §5.
  1. **2026-07-31 + 2026-08-09 (haiku doc-closeout fabrications, two independent instances)** →
     **promoted to `teco.md`** (step 3 model routing + step 5 numeric re-verification). Two
     datapoints of the same shape made this the highest-value entry in the inbox.
  2. **2026-08-09 (archival-flip authority: by-kind table vs. a document's own `Owner:`)** →
     **promoted to root `AGENTS.md`** lifecycle section (the table now says it controls) **and** one
     clause in teco's own milestone-close bullet. A fact about the project's doc convention belongs
     in project docs, not hoarded in one agent's prompt.
  3. **2026-07-29 (`node` not on `PATH` on this WSL2 box; two sessions rediscovered the
     workaround)** → **promoted to `falkor-chat/AGENTS.md`**, next to the bootstrap env note.
  4. **2026-07-29 ("continue via SendMessage" not backed by an available tool)** → **closed by
     live probe**, not promoted: a 2026-08-10 read-only probe of a real teco run shows
     `SendMessage` present as a full tool definition with **no** ToolSearch step and **no**
     deferred-tool reminder anywhere in its context. The entry described the pre-K-007 state
     (`SendMessage` was absent from `tools:` until 2026-07-29); it is no longer true.
- **Inbox cleared.**
- **Plan items:** none (distillation).

## 2026-07-30 — Commit authority formalized: `Bash` may now `git add`/`git commit` a coordinated deliverable, by explicit path
- **What:** Documented, for the first time, teco's authority to `git add`/`git commit` a
  specialist's deliverable it is actively coordinating — a plan, review, test plan/report, or
  kaizen-inbox entry it has already verified fits (step 4) — by explicit path, one coherent unit
  per commit, never `git add -A`/`.`/`-a`, never `push`/`reset`/`rebase`/amend, never anything
  outside the coordination it's actively running. Three touch points, no frontmatter/tool change
  (`Bash` was already granted): (1) Guardrails gained a dedicated bullet, placed right after the
  existing "you coordinate, you don't do the specialists' jobs" bullet, which now drops its stale
  "never mutating the tree" clause since that's no longer literally true; (2) step 4 gained one
  sentence — commit a verified deliverable rather than leave it sitting uncommitted; "verified but
  uncommitted" is now explicitly unfinished integration, not a stopping point; (3) cross-referenced
  in `claude/README.md` (teco + tico rows) and a new paragraph in `claude/AGENTS.md`'s Hook
  machinery section explaining this is prompt-level, not hook-backed. **Scoping deliberately
  differs from `tico`'s existing grant** (2026-07-23): tico's commit scope mirrors its own
  Write/Edit guard exactly (it only ever commits what it itself authored); teco's commit scope is
  wider than its own Write/Edit guard (which reaches only the coordination doc + its inbox)
  because its role — integrator of a whole coordinated unit's output — is structurally different
  from tico's. Both grants are pinned to the same stakeholder line so neither reads as
  self-expanding: "tico and teco are special and have coordination rights."
- **Deterministic backstop added:** `claude/scripts/audit-team.sh` gained **check 8** —
  `COMMIT_AUTHORS=("tico" "teco")`; every other agent's `<name>.md` fails the audit if it ever
  comes to claim `git add`/`git commit` authority, and `tico`/`teco` fail if their documented
  grant ever goes missing. This exists because no `PreToolUse` hook can gate a *prose* capability
  the way the doc-scoped/destructive-ops hooks gate Write/Edit paths and Bash command patterns —
  the grep-based check is the only mechanical trip-wire available for "did commit authority
  quietly spread to a third agent." Full audit re-run clean (95+ PASS, 0 FAIL) after the change.
- **Verified the four commits that prompted this change were safe, not just retroactively
  justified.** Read all four (`15d3ad5`, `4fe43a0`, `10f13ae`, `38e020d`) via `git show --stat`:
  each touches exactly the files its subject line names — `docs/reviews/cpg-getting-started.md`
  (analyst's review); `docs/test-plans/cpg-getting-started.md` +
  `docs/test-reports/cpg-getting-started-report.md` (qa-engineer's plan+report, one coherent
  unit); `claude/analyst/kaizen/inbox.md` (analyst's own learning); `claude/qa-engineer/kaizen/inbox.md`
  (qa-engineer's own learning) — no unrelated file in any diff, consistent with explicit-path
  staging rather than `git add -A`/`-a`. No `push`/`reset`/`rebase`/amend in the sequence (four
  distinct hashes, ~30s apart, matching four sequential `git commit` calls, not one commit
  rewritten). This matches the newly-formalized scope precisely — every committed file was a
  verified deliverable from a specialist teco was actively coordinating (the
  `docs/manuals/cpg-getting-started.md` review-gate rollout), none of it teco's own authorship.
  Nothing found that needed flagging as unsafe.
- **Why:** stakeholder decision, relayed via `cobb`: extend commit/coordination authority to
  `tico` and `teco` specifically (declining `cobb`'s earlier 2026-07-30 recommendation to instead
  extend narrow per-doc-kind commit rights to `analyst`/`qa-engineer`) — closes the gap between
  what teco had already done once in this session (four commits, undocumented authority) and what
  its prompt claimed ("never mutating the tree", no carve-out at all).
- **Plan items:** none opened (the fix was direct, not a backlog item); one §7 minor logged to
  `plan.md`'s parking lot (bullet density).

## 2026-07-29 — Manuals join the routing table, handoff contracts, doc scan, and review-gate defaults
- **What:** Four small additions reflecting tico's new Mode 2/3 (didactic explanation + user-manual maintenance, same day): (1) a new routing-table row — live explanations stay pause→user (tico isn't a delegation target), but a self-contained manual write/update is delegable to tico like any other subagent deliverable; (2) the tico handoff-contract line now names `docs/manuals/<slug>.md` alongside the requirements doc; (3) the documentation-impact scan bullet now lists user manuals (flag, don't write — `tico` owns them); (4) the "Work ships independently reviewed" guardrail gained a manuals entry: split by claim — `qa-engineer` verifies walkthroughs against the running app, `analyst` checks architectural/factual claims and clarity. The manuals-delegable routing row also notes the review gate still applies when teco routes a manual update this way.
- **Why:** user ruling following the 2026-07-29 team certification, which flagged manuals as the one doc kind with no independent-review gate; user chose the qa-engineer/analyst split (behavioral vs. everything else) and "mandatory in teco + offered in tico's first-order sessions" for how forced the gate should be.
- **Plan items:** none.

## 2026-07-29 — PII leak fixed in this file (found by the team certification pass)
- **What:** The K-009 entry below (added earlier the same day) had embedded the literal
  flattened `~/.claude/projects/...` transcript-directory path, which leaks the OS username —
  genericized to `<flattened-repo-path>`. Working-tree fix only; the leak reached one shared
  commit (`e7ec4a3`) before being caught — not rewritten, per the repo's don't-rewrite-shared-history
  norm.
- **Why:** Surfaced by `claude/scripts/audit-team.sh` check 7 during the 2026-07-29 team
  certification (see cobb's kaizen history for the full pass).
- **Plan items:** none.

## 2026-07-29 — Learnings inbox distilled: 3 entries → 1 promoted to teco.md, 1 to agent-standards, 1 discarded as duplicate
- **What:** Processed all three pending entries in `kaizen/inbox.md` (agent-maintenance skill §5):
  1. **2026-07-25 — "`.mcp.json` server materializes only at session start; subagents inherit MCP tools from the parent session"** — genuinely new, not previously captured anywhere in the repo (checked `AGENTS.md`, `cypher-mcp/README.md`, `skills/agent-standards/claude-code.md`). **Promoted** to `skills/agent-standards/claude-code.md` § MCP → Lifecycle (a harness-level fact, not teco-specific — belongs in the on-demand reference cobb maintains, not an always-loaded prompt), with a `Verified: 2026-07-25` stamp and the cpg-query-access delivery as evidence.
  2. **2026-07-25 — "verifying 'no new audit failures' needs a diff against the last commit, not a re-read of the gate's verdict"** — checked `skills/agent-maintenance/SKILL.md` §4 and found it **already promoted**, word-for-word disposition, same origin date (2026-07-25) and same task (`docs/plans/cpg-query-access.md` rework). **Discarded** as a duplicate of an already-landed promotion — nothing to do.
  3. **2026-07-27 — "a brief that fences off `claude/` silently disables the delegate's own learnings inbox"** — teco's own coordination mistake, still live risk (any future brief that excludes `claude/` for collision-avoidance repeats it). **Promoted** into teco's own prompt (step 3, appended to the model-routing sentence): carve out the delegate's `kaizen/inbox.md` explicitly, or have the learning come back in the report, whenever a brief fences off a subtree containing it.
- **Why:** user asked to process the inbox after the K-006/008/009/010/011 backlog pass. Each entry got the full §5 treatment (verify still true / not already documented, route to exactly one destination, log, clear) rather than a blanket append.
- **Plan items:** none (inbox distillation, not a plan item).

## 2026-07-29 — K-006, K-008, K-009, K-010, K-011 ✅: all five open plan items closed
- **What:** Worked the full active backlog in one pass.
  - **K-008 (verified, then adopted):** Live-tested whether the `Agent` tool's per-call `model` override reaches a call made *from inside* a subagent — spawned a `general-purpose` agent that itself called `Agent(model:"haiku", run_in_background:true, ...)`; grepped the resulting nested transcript (`agent-<id>.jsonl`) and found `"model":"claude-haiku-4-5-20251001"`, confirming the override is honored one level down. Added a sentence to step 3: pass `model: "haiku"` on cost-insensitive units (routine doc touch-ups, small-diff re-reviews, suite runs); anything with design/code-quality stakes stays on the inherited model.
  - **K-009 (audited, then dropped):** Grepped all 5 of teco's own session transcripts (`~/.claude/projects/<flattened-repo-path>-claude-teco/*.jsonl`) for direct `WebFetch`/`WebSearch` tool_use. Found exactly one hit: 2026-07-24, during the K-002 agent-teams evaluation, teco fetched `code.claude.com/docs/en/agent-teams` and `/agent-view` directly instead of delegating — research that its own routing table already assigns to `cobb` ("Agent/subagent/skill/prompt/hook engineering"). One mis-routed use in the whole history doesn't justify the grant; dropped `WebFetch`, `WebSearch` from `tools:`.
  - **K-006 (decided):** The independent-review guardrail's defaults line named `analyst` for "plans and code" without saying whether that covered `graph-dba` design notes or `cobb` agent/skill deliverables. Made both explicit in the same clause rather than adding a new row: `plans and code (including graph-dba design notes and cobb's agent/skill artifacts) → analyst`.
  - **K-010 (trimmed):** Description's trailing clause ("Does not design solutions and routes non-trivial implementation to a specialist; may fix a genuinely trivial single-file no-brainer directly instead of delegating it" — 170 chars) shortened to "Delegates non-trivial implementation; may fix a trivial single-file no-brainer itself." (88 chars) — the routing table's row 1 already carries the full tie-breaker prose, so the description only needs the routing signal.
  - **K-011 (pruned):** Removed the three single-use Bash allow-rules (exact escaped Cypher literals from the K-001 probe run) from `.claude/settings.local.json`, leaving only the `test_queries.sh` entry.
- **Why:** user asked to work the teco kaizen backlog. K-008 and K-009 were verification-gated/evidence-gated rather than pure opinion calls, so both were resolved empirically (live nested-call test; transcript grep) instead of by inference.
- **Plan items:** K-006 ✅, K-008 ✅, K-009 ✅, K-010 ✅, K-011 ✅ — all moved here; active table now empty.

## 2026-07-29 — Credit/interface analysis backlogged as K-008..K-011 (review only, no source change)
- **What:** A user-requested analysis of teco's interfaces and credit consumption produced four new plan items, filed (not implemented): **K-008** (high) — route cost-sensitive delegations to a cheaper model via the `Agent` tool's own per-call `model` param, distinct from the per-agent frontmatter pin the team already rejected; needs live verification it reaches nested calls before adopting. **K-009** (medium) — audit whether teco itself ever uses its `WebFetch`/`WebSearch` grants (vs. delegating research), drop if unused. **K-010** (low) — scheduled token-cost recompression: description regrew 568→694 chars (+22%) since 2026-07-25, body regrew 9,866→12,656 (+28%) since 2026-07-11, both from legitimate feature additions rather than waste. **K-011** (low, hygiene) — prune three single-use Bash allow-rules left in `.claude/settings.local.json` from the one-off K-001 probe run.
- **Why:** the same analysis session that produced K-007 (above) surfaced these as lower-priority or verification-gated items not to act on immediately; recorded per the agent-maintenance skill's "record new ideas even on a review-only pass" rule rather than left informal in chat.
- **Plan items:** K-008, K-009, K-010, K-011 opened (all 🔵 proposed).

## 2026-07-29 — K-007 ✅: `SendMessage` continuation replaces cold respawn in the defect→fix→re-run loop
- **What:** Three touch points, no catalog change needed (internal execution mechanism, not a routing/deliverable-contract change). (1) `tools:` gained **`SendMessage`** — it was absent, so K-007 as previously worded would have been unshippable even after a step-4 rewrite. (2) Step 3 gained one clause: note the name/id each `Agent` call returns for any unit carrying a review gate or likely to need a follow-up round, since that identifier is what a later `SendMessage` addresses. (3) Step 4's two re-brief paths — the review "needs changes"/qa-defects loop, and the K-004 deficient-result path (errored/out-of-turns/off-brief/empty) — now both `SendMessage` the original delegate by that identifier (resumes from its own transcript, no re-explaining context) instead of a fresh `Agent` call; cold respawn is reserved for when the identifier **no longer resolves** (no name/id was ever returned, or a newer agent has since taken the same name) — the actual boundary condition per `SendMessage`'s own tool description, not the "errored/out-of-turns" split first drafted (see below).
- **Why:** the session's own analysis (prompted by a user request to find credit/interface optimizations) identified this as the highest-value unshipped lever: the defect→fix→re-run loop was re-explaining full context to a cold `Agent` spawn every retry cycle. `SendMessage`'s live tool description (fetched via `ToolSearch` this session, not a cached doc page) resolved K-007's open verification question — it explicitly states a send "resumes it from its transcript" for a named agent, "even after an agent completes," matching the `Agent` tool's own description ("use SendMessage with the agent's ID or name ... resumes it with full context"). This is the current harness's own self-description, stronger evidence than the two doc pages (`agent-teams`, `agent-view`) the original K-007 note flagged as describing two different mechanisms — still worth confirming empirically on the first real re-brief cycle, but no longer blocking.
- **Self-caught fix during drafting:** the first draft of the step-4 rewrite said "fall back to a fresh `Agent` call... when the original agent errored out entirely or exhausted its turn budget" — directly contradicting the same sentence's "deficient" category, which lists "errored, ran out of turns" as `SendMessage`-retry triggers. Caught by a §7-style self-check before this entry was written; corrected to the identifier-resolution boundary instead (see above).
- **Verified no regression:** `claude/scripts/audit-team.sh` re-run clean on teco (no teco-related FAIL; the 4 pre-existing FAILs are root `AGENTS.md` missing `coder`/`devops`/`frontend-engineer`/`tdd-engineer` — unrelated drift, out of scope here, reported not chased).
- **Plan items:** K-007 ✅ done (moved to plan.md's done-notes block).

## 2026-07-27 — Unpinned from `model: opus` (team-wide)
- **What:** Removed the `model: opus` frontmatter line. The field is now absent, so the agent runs on Claude Code's default — `model` **defaults to `inherit`** (re-verified 2026-07-27 against `code.claude.com/docs/en/sub-agents`), i.e. the model the session/system default selects. No other frontmatter or body change.
- **Why:** User no longer wants the team locked to Opus. Model choice belongs at the session level (one decision, changeable with `/model`), not duplicated across 13 frontmatter files where it silently overrides whatever the user picked.
- **Plan items:** —

## 2026-07-27 — Milestone-close freeze becomes a coordination duty; coordination docs open with the header block (step 2 of `docs/plans/doc-reference-convention.md`)
- **What:** Two body edits, no frontmatter and no hook change. (1) *How you work* step 2 gains one line — *"Open the document with the header block from root `AGENTS.md`."* — the canonical sentence, byte-identical across the six producing prompts. (2) *Documentation curation* gains a third bullet: at milestone close, list every document the close freezes and make flipping each one's header to `Status: archived` a **done-condition of the closing unit, routed to that document's owner** (root `AGENTS.md` carries the per-kind routing table); nothing moves; `teco` coordinates and performs only the flip the table assigns it — its own `docs/plans/<slug>-coordination.md`.
- **Why:** `docs/plans/doc-reference-convention.md` v1.4, blocker **B5** and M2. Under D4 a frozen document no longer moves to `archive/` — it gets `Status: archived` in place — which turns "archiving" from a file operation nobody had to schedule into a **flip somebody must be told to perform**, at a moment (`milestone close`) only the coordinator sees. Without this bullet the lifecycle signal the whole convention rests on would simply never be set. Routing rather than performing is forced by the guard topology, not by ceremony: `teco`'s `PreToolUse` allowlist reaches `docs/plans/*` only, so a flip it performed on a review, requirements doc, test plan or test report would raise an interactive human approval prompt **per file** — `falkor-chat/docs/reviews/` alone holds four active documents. The routing table is pointed at, not copied, for the same reason the header block is (v1.4 M20): root `AGENTS.md` is already in every agent's context via the root `CLAUDE.md` `@AGENTS.md` import, so the hop costs nothing while a second copy would drift. `claude/README.md` row 7 re-checked — it already describes `teco` as documentation curator who makes doc updates part of every unit's done-condition, which is exactly what this bullet instantiates; no catalog edit needed.
- **Plan items:** none. (K-006/K-007 untouched.)

## 2026-07-25 — Trivial single-file no-brainer fixes: teco may make them directly instead of delegating
- **What:** Relaxed the "coordinates, never implements" invariant one notch: teco may now make a genuinely trivial, single-file, no-design-needed fix (a typo, an obvious one-liner, a config value, a rename) directly instead of spinning up a specialist for it. Four touch points, no hook-allowlist change: (1) frontmatter `description`'s closing line now reads "does not design solutions and routes non-trivial implementation to a specialist; may fix a genuinely trivial single-file no-brainer directly instead of delegating it" (was an unqualified "Does NOT design or write code itself"); (2) opening persona paragraph states the exception inline; (3) Routing table gained a leading row (trivial single-file no-brainer → teco directly, tie-breaker: multiple files/design judgment/security-data-model-test-critical → delegate instead); (4) Guardrails' coordination bullet and ceremony bullet updated to match. The `PreToolUse` hook (`guard-coordination-doc-writes.sh`) is **unchanged in behavior** — its allowed globs still only cover `docs/plans/` and the kaizen inbox, so a trivial fix still hits the "ask" escalation and needs a one-time human approval; only the escalation *message* was reworded (no longer "deny by default", now "approve if this is genuinely that kind of trivial fix"). This keeps a human check on every non-coordination-doc write teco makes, trivial or not — it just stops teco from having to pretend the option doesn't exist.
- **Why:** User request: too much delegation overhead going to `coder` for small no-brainer changes. Discussed the trade-off first (this reopens ground settled by the 2026-07-08 architect/teco K-003 hook-enforcement work) and the user chose the narrowest of three options offered — prompt-level permission for trivial edits only, hook left as the safety net — over widening the hook's allowlist or just trimming routing ceremony elsewhere.
- **Plan items:** none (out-of-band user request). Worth revisiting if the "ask" escalation for trivial fixes turns out to fire often enough to reintroduce the friction this was meant to remove — that would be the signal to reconsider widening the hook allowlist (the second, rejected option).

## 2026-07-24 — Description slimmed further (second team-wide token-cost pass)
- **What:** Frontmatter `description` compressed 661 → 568 chars (-14%): tightened phrasing, dropped restated detail. `teco` has no boundary pairs in `claude/scripts/audit-team.sh`; full audit re-verified green regardless. No body/catalog change.
- **Why:** All 13 agents' descriptions are auto-injected into every session and subagent spawn; the roster grew to 13 (graph-dba, joern added) since the first pass on 2026-07-11, and per-agent `/context` output showed room to cut further. User-requested via a `/context` token audit.
- **Plan items:** none.

## 2026-07-24 — K-002 ✅: agent-teams evaluation closed — reject team-lead reframe; SendMessage sub-case spun to K-007
- **What:** Read `code.claude.com/docs/en/agent-teams` and `/en/agent-view` (the concrete step K-002 asked for) and closed with disposition: **reject** reframing teco as an agent-teams lead. Agent teams are experimental (opt-in `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS`), built for teammates that talk directly to each other on independent, discussion-benefiting work (parallel review lenses, competing-hypothesis debugging, cross-layer ownership) — the docs are explicit that "for sequential tasks... or work with many dependencies, a single session or subagents are more effective." Teco's actual loop (decompose → sequence on dependencies → delegate → independently-reviewed gate) is exactly that latter shape; teams would add token overhead for no matching benefit. The 2026-07-12 sub-case (defect→fix→re-run re-spawning cold agents) turned out not to be an agent-teams question at all — it's answered by `SendMessage` continuation of the original delegate (confirmed available for `Agent`-tool subagents per the harness's own tool description), independent of the experimental teams flag. Spun off as **K-007**.
- **Why:** User asked to follow through on K-002's own proposed next step (read the docs, assess fit) rather than leave the plan item open indefinitely.
- **Plan items:** K-002 ✅ done (moved here); opened K-007 (adopt SendMessage continuation in step 4's defect→fix→re-run loop).

## 2026-07-24 — Frontmatter: `permissionMode: acceptEdits`
- **What:** Added `permissionMode: acceptEdits` to the frontmatter, matching the same-day change across the team (`coder`, `tdd-engineer`, `frontend-engineer`, `architect`, `qa-engineer`, `analyst`, `devops`, `graph-dba`, `joern`). File-edit/write approvals are session-scoped in Claude Code (unlike Bash approvals, which persist permanently per repo+command), so users otherwise have to re-grant write permission every session even with a global `Edit`/`Write` allow rule in `~/.claude/settings.json`.
- **Why:** Verified against current Claude Code docs (`hooks-guide.md` "Hooks and permission modes") that this is safe: `PreToolUse` hooks fire *before* any permission-mode check, and a hook's `"ask"` decision still forces the prompt even under `acceptEdits`/`bypassPermissions`. `teco`'s `guard-coordination-doc-writes.sh` hook (escalates to ask on any Write/Edit outside the allowed coordination-doc paths) keeps working exactly as before; only writes it would already let through silently stop re-prompting every session.
- **Plan items:** none.

## 2026-07-16 — Applied K-004 + K-005 to teco.md (from the §7 lint)
- **What:** Two surgical prompt additions, user-approved from the same-day §7 lint. **K-004** — Step 4 ("Integrate & verify") gained a *deficient-result* path: when a delegate errors, runs out of turns, or returns something off-brief/empty (explicitly distinct from a *blocker* that changes direction and a review *verdict*), re-brief the same owner once with the gap made explicit, and pause to the user if it recurs or the unit is mis-scoped — "rather than re-spawning blindly". **K-005** — the Documentation-curation "Scan at decomposition" list now names `docs/HISTORY.md` (which takes an entry for every delivered change) and `docs/BACKLOG.md` "where the module uses the convention", closing the gap between teco's curator role and the module-documentation convention in root `AGENTS.md`. No frontmatter/description/catalog change — role unchanged, so the catalog entries still describe teco correctly. K-006 left proposed (not approved).
- **Why:** User approved acting on the two higher-value lint findings; both were surgical additions at teco's existing altitude, not a rewrite.
- **Plan items:** K-004 ✅, K-005 ✅ (moved to the done-notes block in plan.md); K-006 stays open.

## 2026-07-16 — §7 prompt-quality lint (review-only, no prompt change)
- **What:** cobb ran the new `agent-maintenance` §7 single-artifact prompt-lint against `teco.md` across all six dimensions, resolving teco's full load-set (root + `claude/` `CLAUDE.md`→`@AGENTS.md` chain, the injected specialist `description`s, the coordination-doc write guard) for the composition check. **Persona:** clean. **Contradiction / ambiguity / cognitive load:** clean bar minor nits (parked). **Coverage + composition:** three findings filed — K-004 (no deficient/failed-delegate-result path), K-005 (doc-curation scope omits the module `docs/HISTORY.md`/`BACKLOG.md` conventions from `AGENTS.md` — highest-value, surfaced only by the composition load-set resolution), K-006 (no independent reviewer assigned for agent-engineering deliverables). No blocker; no source change.
- **Why:** Smoke test of the §7 procedure cobb authored the same day; teco is a mature, certified prompt so a clean-ish result was expected and validated that §7 surfaces real gaps without manufacturing findings.
- **Plan items:** opened K-004, K-005, K-006; minors parked.

## 2026-07-12 — K-003 ✅: review-gate invariant proven on the first fully-gated run — kept, no prompt change
- **What:** Closed K-003 with disposition **(a) keep the invariant** — "work ships
  independently reviewed; when you trim ceremony, the review gate is the last thing to go."
  falkor-chat **K-022 Landing 1** (U1–U10, committed `3921f87`) ran as the team's first fully-gated
  coordinated delegation with the analyst post-implementation review as a non-negotiable
  done-condition, and the cost datapoint the plan asked for is now recorded in
  `falkor-chat/docs/plans/m3-executor-coordination.md` ("Cost datapoint" table). **No prompt
  change** — the datapoint vindicates the existing guardrail rather than forcing the (b) rewrite
  to risk-signal-gated review.
- **Evidence / reasoning from the datapoint:**
  1. **The gate is cheap.** Analyst review = ~149k tokens / 25 tool uses / ~7 min — ~12% of the
     ~1.20M-token, ~4h gated run, a thin slice on top of the six implementation delegations.
  2. **The gate paid.** On a diff the implementers considered done it returned
     approve-with-suggestions with **1 major (M-1, the drive try/except) + 3 minor + 3 nit** —
     exactly the class of defect the K-020/21 "review left to the user" skip would have shipped
     unseen.
  3. **The headline ~12× vs. the K-001 baseline is a units artifact, not the gate's cost** — 10
     units + independent gate vs. an ungated 2-unit slice (~100k / 23 / ~45 min). Per-unit the run
     is comparable; the review is the cheap part.
  4. Therefore the concern that opened K-003 — "an invariant that never fires is hopeful prose" —
     is resolved: it fired, cheaply, and caught real signal. Keeping review-by-default is the
     right risk posture; the low marginal cost means the default stands even at n=1.
- **Honest caveat (recorded, not blocking):** this is **one** gated run. It proves the gate can pay
  its way and is affordable, not that every gate will catch a major. The cost is low enough that
  "keep the default, skip only with stated justification for genuinely trivial units" remains
  correct pending more datapoints — re-examine if a run of gates comes back all-nits at real cost.
- **Why:** User asked to close the K-003 thread now that K-022 is committed. The experiment ran end
  to end (gate enforced + datapoint captured); the disposition is the last step the plan item
  named ((a) keep / (b) rewrite).
- **Plan items:** **K-003 ✅ done** (moved here). No change to `teco.md`, `README.md`, or the
  context catalogs — behavior/routing unchanged; this is a decision to *keep* the current prompt.
  Counterparts still open on their own agents: `analyst` K-001 (its code-review shakedown — the
  same run validated it; closeable on analyst's side) and `qa-engineer` K-003 (defect→fix→re-run
  loop — **unexercised**, the review returned 0 blockers so no needs-changes loop fired).

## 2026-07-12 — Learning-capture loop: kaizen inbox + closing protocol + guard allowlist + integration check
- **What:** Added `kaizen/inbox.md` (append-only learnings inbox, seeded empty) and a "Learning capture" closing-protocol section to the prompt; the coordination-doc write guard's allowlist gained exactly teco's own inbox path. Step 4 (Integrate & verify) additionally gained the learnings-ride-the-handoff check: when a specialist's result reports a durable environment discovery, confirm it was filed in that agent's inbox (a one-line check, not a gate).
- **Why:** Team-wide self-improvement loop (agent-maintenance skill §5, added the same day): capture during runs, curated promotion by cobb. Teco is the collection point on orchestrated work — the integration check catches learnings a delegate forgot to file. Requested by the user.
- **Plan items:** none.

## 2026-07-11 — graph-dba added to the handoff contracts (certification fix)
- **What:** The "Handoff contracts" list gained the `graph-dba` entry: implementer-bound design work (data model, schema/DDL, ingestion/migration) arrives as a design note at `<component>/docs/plans/<slug>-graph.md`; quick consults and tuning diagnoses stay inline. Matches the same-day addition of the convention to graph-dba's own prompt (its kaizen K-004).
- **Why:** Team-coherence certification (2026-07-11): graph-dba was the only design-producing specialist whose deliverable teco had to paraphrase into the next brief — the exact lossy handoff the "by path, never paraphrased" rule exists to prevent.
- **Plan items:** none (graph-dba K-004 on the producer side).

## 2026-07-11 — Prompt body compressed (token-cost pass, part 2)
- **What:** Body compressed in place, 15,023 → 9,866 chars (−34%): the routing table's per-agent capability prose was cut down to pure routing judgment (tie-breakers, boundaries, pipeline defaults), explicitly leaning on the injected frontmatter descriptions teco already receives at spawn through its `Agent` tool; "How you work", documentation curation, pause rules, and guardrails were tightened without dropping any rule or contract. All 11 specialist names remain in the file (audit check 4 green, full audit pass); frontmatter (description, tools, hook) unchanged. No on-demand reference file — teco uses its whole body every run, so offloading would just add a mandatory Read.
- **Why:** teco.md was the team's second-heaviest prompt and loads on every teco spawn; the injected description catalog already carries each specialist's capabilities, so restating them in the body was pure duplication (~1,450 tokens saved per spawn).
- **Plan items:** none.

## 2026-07-11 — Description slimmed (team-wide token-cost pass)
- **What:** Frontmatter `description` compressed from 1286 to 659 chars: capability lists tightened, reciprocal boundary prose reduced to short route-away clauses that still name the counterpart agents (audit check 6 boundary symmetry preserved — full pass green), and "how I work" detail dropped from the description since the prompt body already carries it. Routing semantics unchanged; no body/catalog changes needed.
- **Why:** All 12 agents' descriptions are auto-injected into every session and into every subagent spawn that carries the `Agent` tool; team-wide they cost 12,609 chars (~3.1K tokens) per injection. The pass cut them to 7,036 chars (~44%), saving ≈1,400 tokens per session/spawn with the same routing contract.
- **Plan items:** none.

## 2026-07-11 — Guard hook refactored to a thin wrapper over a shared core
- **What:** `guard-coordination-doc-writes.sh` was reduced from a ~60-line standalone script to a thin wrapper that `exec`s the new shared core `claude/scripts/guard-doc-writes.sh` with two parameters — this agent's allowed-path globs (`docs/plans/*|*/docs/plans/*`) and its escalation-message template (`__PATH__` placeholder for the offending path). The core carries the shared machinery unchanged: jq→python3 path extraction, fail-open on unparseable input, `/tmp/*` always allowed, `permissionDecision: "ask"` JSON emit. The wrapper resolves the core via `readlink -f "$0"`, so it works when invoked through the `~/.claude/agents/<name>` deployment symlink; the frontmatter hook command is unchanged. Verified: `bash -n`, allowed/denied/scratchpad/fail-open cases through the symlink path, the no-jq python3 fallback, and `claude/scripts/audit-team.sh` all pass.
- **Why:** a repo redundancy audit (2026-07-11) found the five doc-scoped guards (analyst, architect, data-scientist, teco, tico) byte-identical except one `case` glob and one message string — ~250 duplicated lines that had to be patched five times per fix. One parameterized core removes the drift risk. (`devops/hooks/guard-destructive-ops.sh` stays standalone — it matches Bash command patterns, not write paths.)
- **Plan items:** none.

## 2026-07-10 — Independent review made the default mindset
- **What:** independent review is now a standing principle, not an optional gate. Four touch points: (1) a new guardrail — "**Work ships independently reviewed**": no deliverable is accepted on its producer's word alone, teco's own integration check is fit/completeness (not a substitute for review), and every significant deliverable defaults to a reviewer who didn't produce it (plans/code → `analyst`, ML methodology → `data-scientist`, behavior/acceptance → `qa-engineer`); skipping a gate is the justified exception for genuinely trivial, low-risk units, stated explicitly in the report. (2) Step 2 now assigns each unit its **review gate** alongside owner/inputs/done-condition. (3) The typical-feature paragraph flips `analyst` from "slotted in where the stakes warrant it" to the **default review gate**, and the match-ceremony-to-task rule gains "when you trim ceremony, the review gate is the last thing to go, not the first." (4) The frontmatter `description` advertises the default.
- **Why:** User request: teco should "always have in his mindset the need for the work to be independently reviewed." The previous phrasing made review an exception teco had to argue itself into; the risk posture the user wants is the inverse — review by default, skip only with justification.
- **Plan items:** none.

## 2026-07-10 — Standing documentation-curator duty
- **What:** teco is now the team's **documentation curator**, keeping project docs always in sync with delivered work. Four touch points: (1) a new "Documentation curation" section with the standing rules — documentation-impact scan at decomposition (READMEs, `AGENTS.md`/`CLAUDE.md`, design/reference docs, catalogs, recorded in the coordination doc), affected docs named in the unit's brief with same-change updates part of the deliverable (the unit's owner writes them; agent/skill docs → `cobb`), verification by actually reading the flagged docs at integration (stale docs = incomplete unit → re-brief), and pre-existing drift reported as a follow-up rather than silently chased; (2) step 2 runs the scan as part of the breakdown; (3) step 4 makes documentation part of done; (4) the frontmatter `description` advertises the curator duty. teco still never writes these docs itself — `Write`/`Edit` stays hook-scoped to the coordination doc.
- **Why:** User request: teco should "keep track of the docs updates, being the curator for an always updated documentation." Curation (track → brief → verify) fits teco's coordinator identity and existing hook scope; the writing routes to the owner of each change.
- **Plan items:** none.

## 2026-07-10 — Hook command made machine-independent (`$HOME` symlink path)
- **What:** the frontmatter `PreToolUse` hook command was rewired from the absolute repo path (`/home/<user>/prg/graphmind-ai-lab/claude/teco/hooks/guard-coordination-doc-writes.sh`) to `$HOME/.claude/agents/teco/hooks/guard-coordination-doc-writes.sh`, which resolves through the user-scope deployment symlink (`~/.claude/agents/teco` → the repo folder). Shell-form hook commands (no `args`) run via `sh -c`, so `$HOME` expands — verified 2026-07-10 against `code.claude.com/docs/en/hooks`. Resolution through the symlink confirmed (`test -x` passes).
- **Why:** the committed agent source leaked the user's personal home path into the repo; the symlink path is identical on any machine that follows the deployment convention (`~/.claude/agents/<name>` → `claude/<name>`), keeping the hook enforceable without machine-specific paths. (`${CLAUDE_PROJECT_DIR}` was rejected: the agents are user-scoped and must guard in any project, where the project dir isn't this repo.)
- **Plan items:** none.

## 2026-07-09 — Roster: added data-scientist (AI/ML/DS advisory specialist)
- **What:** the routing table gained a `data-scientist` row (AI/ML/data-science **method** questions — model/embedding selection, retrieval strategy, RAG/GraphRAG evaluation design, quality metrics, experiment/A-B design, statistical validity — plus methodology reviews and model/retrieval-underperformance diagnosis; boundary notes: advisory-only — implementation of its recommendations routes to the implementers with its note as the brief, general correctness review stays with `analyst`, in-graph vector mechanics/Cypher with `graph-dba`); the handoff-contracts list gained its two deliverables (method note `docs/plans/<slug>-ml.md`, methodology review `docs/reviews/<slug>-ml.md`, hook-enforced advisory-only writes); the frontmatter parenthetical now includes it.
- **Why:** an AI/ML/data-science specialist joined the team; the orchestrator's roster must enumerate every delegate with its current contract (the drift class the 2026-07-09 interface review exists to catch).
- **Plan items:** none.

## 2026-07-09 — Roster: added frontend-engineer (UI-depth implementer)
- **What:** the routing table gained a `frontend-engineer` row (UI-heavy front-end work — components, styling, accessibility, client-side state, front-end performance, Streamlit screens — with the boundary note that back-end/non-UI code stays with `coder`/`tdd-engineer` and incidental template touches don't need the specialist); the frontmatter parenthetical and the typical-feature pipeline now include it among the implementers.
- **Why:** a front-end specialist joined the team; the orchestrator's roster must enumerate every delegate (the drift class the 2026-07-09 interface review existed to catch).
- **Plan items:** none.

## 2026-07-09 — Roster restructured into an explicit routing table + handoff contracts
- **What:** "The team you coordinate" reformatted from prose bullets into two artifacts: a **routing table** (task shape → owner → tie-breaker/boundary, one row per routable signal, including the "requirements vague → pause, recommend tico" row and the two built-ins) and a **handoff contracts** list (per-agent document paths and by-path handoff rules for tico/architect/analyst/qa-engineer). Content is unchanged — same roster, same routing rules, same contracts — only made scannable and self-checkable; the typical-feature pipeline paragraph kept as-is. Catalogs (`claude/AGENTS.md`, `claude/README.md`, root `AGENTS.md`) describe routing behavior, not prompt format — verified accurate, no edits needed.
- **Why:** User asked how teco decides routing and for a "clear configuration". Routing is LLM judgment over prompt text; the clearest configuration of that judgment is an explicit decision table teco self-checks before each delegation (the parking-lot "routing cheat-sheet" idea, now fully addressed — including the coder-vs-tdd tie-breakers on both implementer rows).
- **Plan items:** parking-lot "routing cheat-sheet / decision tree" ✅ resolved.

## 2026-07-09 — Roster: analyst gained RCA routing
- **What:** analyst's roster entry (and the frontmatter parenthetical) now also routes **cause-unknown defects/failures** to it for a root cause analysis at `<component>/docs/reviews/<slug>-rca.md`, whose suggested fix then briefs the implementer (typically `tdd-engineer`, reproduction test first) by path.
- **Why:** analyst extended with an RCA mode the same day (user request); the orchestrator's roster must describe each specialist's current contract.
- **Plan items:** none.

## 2026-07-09 — Roster: added analyst (plan & code review gate)
- **What:** Added `analyst` to the frontmatter specialist list and the roster, slotted it into the typical-feature pipeline as an optional review gate (after architect on high-blast-radius plans and/or after the implementer before QA), and extended step 4's defect loop to cover a "needs changes" review verdict (re-brief the owner with the review path, then re-review). The roster entry encodes the handoff contract: review doc at `<component>/docs/reviews/<slug>.md`, handed off by path, review-only on code (hook-enforced).
- **Why:** New team member created 2026-07-09 — the orchestrator's roster must be updated in the same change as the agent (agent-maintenance §2 step 3; the qa-engineer/devops roster-drift lesson).
- **Plan items:** none.

## 2026-07-09 — tico reframed: first-order agent, not a delegation target
- **What:** Removed tico from the frontmatter routing list; its roster entry now marks it **not a delegation target** — tico runs as the user's own main-session agent (`claude --agent tico`) and teco **consumes** its requirements doc (`<component>/docs/requirements/<slug>.md`) by path, treating vague/uncaptured requirements as a pause point that recommends a tico interview. Pipeline reads **tico (user-run) → architect → implementers → qa**.
- **Why:** User ruling, same day as the roster addition below: tico is a first-order conversational agent, not a subagent — the interview must be a live conversation, which delegation can't provide.
- **Plan items:** none.

## 2026-07-09 — Roster: added tico (product-owner interviewer, upstream of architect)
- **What:** Added `tico` to the frontmatter specialist list and the roster, and prefixed the typical-feature pipeline with it (**tico → architect → implementers → qa**, skipped when requirements are already clear). The roster entry encodes the round-trip contract: tico's question batches are a pause point — relay to the user verbatim, re-delegate with the answers + the doc path (`<component>/docs/requirements/<slug>.md`); the finished doc hands to the architect by path.
- **Why:** New team member created 2026-07-09 — the orchestrator's roster must be updated in the same change as the agent (agent-maintenance §2 step 3; the qa-engineer/devops roster-drift lesson).
- **Plan items:** none.

## 2026-07-09 — Roster: implementer routing de-personalized (efficiency rule)
- **What:** Replaced the coder/tdd-engineer routing guidance in the roster. Dropped the *"(This user prefers TDD — lean toward `tdd-engineer` for implementation unless told otherwise)"* note; both bullets now carry a task-shape rule — route by **efficiency, not ceremony**: detailed architect plan ready to execute → `coder`; bug fix (repro test first), safety-net refactor, test-focused work, or clear-contract feature → `tdd-engineer`.
- **Why:** User ruling: personal-preference notes don't belong in agent prompts — their standing preferences are quality and efficiency, expressed as objective routing rules. Part of the same-day coder/tdd-engineer boundary fix (coder K-001 ✅).
- **Plan items:** none (out-of-band).

## 2026-07-09 — K-001 ✅: live nested-delegation validation run (falkor-chat M3 slice 1)
- **What:** Ran teco end-to-end on a real assignment — kick off falkor-chat **M3 — Workflow
  engine**, decompose the milestone, deliver slice 1 (K-020 def model + K-021 snapshot
  materialization). Launch brief + observation checklist: `k001-run-brief.md` (executed verbatim).
  Scored against the checklist from the run transcript + independent re-verification:
  1. **Depth — PASS.** teco (opus) spawned architect → graph-dba → tdd-engineer (one `Agent` call
     each, sequenced on their upstream artifacts); all three nested runs completed with no
     depth-related degradation observed.
  2. **Path-based handoff — PASS.** All three delegate briefs carried the plan-doc path
     (`docs/plans/m3-workflow-engine.md`); the plan was never paraphrased wholesale into a brief
     (briefs ~6.7–7.7 KB, self-contained context + path).
  3. **Brief fidelity — PASS.** Every brief included the "this brief is your entire context"
     framing and the blockers-back-as-deliverable reminder. No observed information loss; the
     one plan gap (no `start_key` param on `publish_workflow_def`) was an *architect plan*
     omission, resolved sensibly by the implementer and surfaced by teco as a follow-up —
     exactly the intended behavior.
  4. **Hook enforcement — PASS (unexercised).** teco's own Write/Edit calls (1 Write + 5 Edits)
     all targeted its coordination doc (`m3-workflow-engine-coordination.md`); the
     guard-coordination-doc-writes hook never needed to fire.
  5. **Decision points — PASS.** The §13 guard-expression-language question was correctly
     assessed as *not forced* by slice 1 (opaque strings, evaluated at run time) and deferred to
     K-022's architect pass with an explicit return-to-user; `ws:acme`/`reference` kept
     additive-only; zero scope creep (executor/linkage/proof flows untouched).
  6. **Integration & honesty — PASS.** teco re-ran both suites itself and reported truthfully;
     independently re-verified afterwards: `test_queries.sh` **193/193**, pytest **196** — both
     matching teco's claims. Nothing committed (correct; review left to the user).
- **Why:** K-001 was the open proof that an orchestrator subagent works in practice — depth,
  context-passing fidelity, and result quality were validated on a real deliverable, not a toy.
- **Prompt changes:** **none needed** — the run surfaced no prompt weakness. Deliverables landed
  in falkor-chat (see `falkor-chat/docs/HISTORY.md` 2026-07-09). Run cost datapoint: ~100k
  subagent tokens / 23 tool uses / ~45 min for a 2-item slice with 3 nested specialists.
- **Plan items:** K-001 ✅ done (moved here). Same-run evidence closed **architect K-002**
  (plan executed cold by an isolated implementer) and updated **coder K-002** (contract proven
  via tdd-engineer; coder-specific run still open). K-002 (agent teams) remains the sole active item.

## 2026-07-09 — Interface review: roster completed (qa-engineer, devops) + guard hook + brief/verify upgrades
- **What:** Thorough review of teco and its interfaces produced five prompt changes and one new artifact:
  1. **Roster completed** — `qa-engineer` (with its `docs/test-plans/` / `docs/test-reports/` artifact conventions) and `devops` (environment blockers routed there instead of bounced to the user) added to the roster, the frontmatter `description`, and the typical-feature pipeline (now `architect → implementer → qa-engineer`, `devops` unblocking env issues). Both agents postdate teco's creation (qa-engineer 2026-07-01, devops ~2026-07) and had never been folded in.
  2. **Brief template generalized** (step 3) — path-based handoff is now the rule for *every* document deliverable (architect plan named as the canonical case, qa plan/report as the other standing instance); briefs must remind delegates they can't ask mid-run (blockers/questions come back as the deliverable).
  3. **Parallel-delegation mechanics** (step 3) — independent delegations go out as parallel `Agent` calls in one turn; dependent ones sequence on their upstream artifact.
  4. **Verify step clarified** (step 4) — running the project's suites/scripts is in-bounds read-only verification; acceptance-level verification routes to `qa-engineer`, with the defect→fix→re-run loop (re-brief implementer with the report path, re-run failed items — qa-engineer kaizen K-003's teco side).
  5. **Guard hook (harness enforcement parity with architect)** — new `teco/hooks/guard-coordination-doc-writes.sh` wired in frontmatter (matcher `Write|Edit`): any target outside `docs/plans/` (or `/tmp`) escalates to the human (`permissionDecision: "ask"`); same fail-open jq→python3 contract as the architect/devops hooks. Unit-driven: allowed path passes silently, violating path emits the ask JSON.
  - **Counterpart fixes in the same change:** `tdd-engineer` gained the plan-doc-path handoff line (mirroring coder) + subagent-awareness ("return the question/blocker as your result"); `qa-engineer` gained the same subagent-awareness in its scope step and environment guardrail. Catalogs synced: `claude/AGENTS.md`, `claude/README.md` (teco row + hook-gotcha list), root `AGENTS.md` teco cell.
- **Why:** Review found teco's core design sound but stale at the edges: two specialists were invisible to it (it literally could not route QA or infra work), its doc-scoping guardrail was prompt-only while the identical architect contract is hook-enforced, and the delegation protocol's key rules (path handoff, no-mid-run-questions) existed only as special cases instead of general brief requirements.
- **Plan items:** parking-lot "routing cheat-sheet" idea partially addressed (complete roster + routing signals per entry); K-001 (live nested-delegation run) and K-002 (agent teams) remain open.

## 2026-07-08 — Path-based architect handoff + coordination-doc convention (K-003 ✅)
- **What:** Two prompt changes, synced with the architect's same-day overhaul: (1) step 3 no longer says to pass the architect's plan **verbatim** — the architect now writes its plan to `<component>/docs/plans/<slug>.md` and teco hands the implementer the **path** with an instruction to read the file itself, never a paraphrase; the roster's architect line states the convention. (2) K-003 resolved: teco's coordination/work-breakdown doc gets a fixed convention too — `<component>/docs/plans/<slug>-coordination.md`, co-located with the architect's plan (baked into step 2). Catalog entries updated (`claude/AGENTS.md`).
- **Why:** Design review of the architect found the verbatim copy-through was the weakest link in the teco pipeline: a long plan returned as a subagent message and re-pasted into a brief risks truncation/paraphrase, and leaves no durable artifact. A file handed off by path is lossless, cheap to brief, and reviewable after the fact. The coordination-doc convention rode along since it was the same decision (architect K-001 fixed the location).
- **Plan items:** K-003 ✅ done (moved here); K-001 note updated — the live nested-delegation validation is still pending but no longer needs to stress brief fidelity for the plan itself.

## 2026-07-05 — Added `Edit` (scoped to the coordination doc)
- **What:** Added `Edit` to teco's frontmatter tools (`Read, Grep, Glob, Bash, Agent, Write, Edit, WebFetch, WebSearch`). Updated the guardrail to `Write`/`Edit` = **coordination/work-breakdown document only** (Write to create, Edit to revise in place as steps complete) — still **never** source/tests/config. Also tightened "How you work" step 2 to mention editing the doc in place. Mirrored the wording in `claude/AGENTS.md`.
- **Why:** User asked to give teco the `Edit` tool. With `Write` only, teco could create a coordination doc but had to overwrite it wholesale to update it; `Edit` lets it surgically revise the doc across a long-running orchestration (mark steps done, append findings). Scoped deliberately to the coordination doc — parallels `architect`, which carries `Write`+`Edit` guardrailed to its plan doc — so teco's "coordinate, don't implement" identity is preserved.
- **Plan items:** none (out-of-band user request); relevant to K-003 (coordination-doc convention).

## 2026-06-20 — Created
- **What:** Created the `teco` subagent (`teco/teco.md`, `model: opus`). Technical coordinator / tech lead: decomposes a multi-step goal into a sequenced work breakdown and **delegates each unit to the right specialist** (architect, coder, tdd-engineer, graph-dba, cobb; Explore/Plan built-ins) via the `Agent` tool, then integrates and verifies. **Hybrid mode:** delegates execution itself by default but pauses and returns to the user at genuine decision points / blockers / ambiguity. Tools: `Read, Grep, Glob, Bash, Agent, Write, WebFetch, WebSearch` — **no `Edit`/`NotebookEdit`** (it coordinates, doesn't implement); `Write` is for the coordination doc only; `Bash` read-only by guardrail.
- **Why:** User asked for a third agent on top of the architect→coder pair — "teco the technical coordinator" — to orchestrate the specialist roster.
- **Plan items:** seeded K-001..K-003.

## Decisions & verification recorded at creation
- **Subagents CAN delegate to subagents — verified 2026-06-20** against `code.claude.com/docs/en/sub-agents`. The doc enumerates the tools withheld from subagents (`AskUserQuestion`, `EnterPlanMode`, `ExitPlanMode`, `ScheduleWakeup`, `WaitForMcpServers`); the `Agent`/Task tool is **not** withheld, so an orchestrator subagent is viable. (Older lore said subagents couldn't spawn subagents — that constraint no longer holds per the live doc. Claude Code now also has first-class *agent teams* and *background agents*.)
- **Key limitation baked into the prompt:** `AskUserQuestion` is unavailable to subagents, so teco **cannot ask interactively** — the hybrid design has it *return* to the user with the decision instead of guessing. teco also doesn't see the parent conversation, and delegated agents don't see teco's or each other's context → the prompt mandates **self-contained briefs** (pass the architect's plan verbatim to the implementer, etc.).
- **No `name`-conflict / collection consistency:** dropped any "senior" framing to match the 2026-06-20 harmonized collection. Defaults implementation routing toward `tdd-engineer` given the user's documented TDD preference.
