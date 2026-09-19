# Salesperson `systemPrompt` language salience (Mitigation D, K-065/DEF-6) — Review

> **Status:** active · **Owner:** `analyst` · **Tracks:** K-065 (gates the first live demo only)

## Scope & verdict

Reviewed the uncommitted working-tree diff (`git diff -- falkor-chat/ salesperson/README.md`,
nothing staged/committed) implementing Mitigation D for K-065/DEF-6: a `v8` bump of
`SALESPERSON_DEF` (`falkor-chat/server/falkorchat/proof_defs.py`) adding one redundant
language-salience sentence to `systemPrompt`, plus the accompanying test, three scripts, and two
`AGENTS.md`/`README.md` doc sites. Baseline for grounding: `falkor-chat/AGENTS.md` (read in full),
`docs/plans/salesperson-ui-ml.md` "Mitigation options assessed"/"Recommendation" sections (read in
full), `falkor-chat/docs/BACKLOG.md`'s K-065 entry, and the coordination ledger
(`falkor-chat/docs/plans/salesperson-language-salience-coordination.md`). I independently re-ran
the targeted test file and the mutation check described below; I did not re-run the full offline
suite (already independently verified twice — by the implementer and by the coordinator — and
this diff's only executable-code hunk is the one file I did test directly). No live LLM run was
attempted or judged (explicitly out of scope; that is U3, gated on this review).

**Verdict: approve with suggestions.** The code change itself is correct, narrowly scoped, and
well-tested; the three documented traps all hold. The one thing that should be fixed before this
is treated as closed-and-landed is a documentation gap the diff itself creates: it edits
`salesperson/README.md` twice for `v7`→`v8` but leaves a "not yet mitigated" status line in the
same file (and the matching claim in `falkor-chat/docs/BACKLOG.md`'s K-065 entry, and in
`falkor-chat/AGENTS.md`) now inaccurate the moment this lands.

**CPG:** not applicable — a prompt/doc/test change with no call-graph or data-flow question in
scope; nothing here turns on symbol relationships a CPG would surface.

## Findings

### Major — the diff leaves three "no mitigation dispatched yet" doc claims stale, one in a file it itself edited

`salesperson/README.md:197` reads *"...confirmed to reproduce at LM Studio's own serving layer,
not yet mitigated"* — in the same file's "Status" section the diff edits twice elsewhere
(`salesperson/README.md:106,126` for `v7`→`v8`). Once this diff lands, that sentence is false: a
mitigation *has* been dispatched (this `v8` bump), only not yet live-validated. The same claim,
worded even more strongly, sits in two more places the diff doesn't touch:
- `falkor-chat/AGENTS.md:107` — *"...application code bypassed (`docs/plans/salesperson-ui-ml.md`),
  not yet mitigated."*
- `falkor-chat/docs/BACKLOG.md:56` — *"**No mitigation has been authorized or dispatched.**"*,
  plus `:71` *"Owner: no fix owner yet..."* and `:73` *"Risks/RAM: none — diagnosis only so far, no
  shipped behavior change."*

This is not a style nit: this exact repo has a precedent for exactly this situation, and the
precedent is to update both. `falkor-chat/docs/HISTORY.md`'s `## 2026-08-28 — K-056: ... breadcrumb
reverted` entry — the case K-065's own backlog text explicitly claims to mirror ("same shape as
the existing K-056→AC-10 precedent") — records *"`docs/BACKLOG.md`'s K-056 entry updated to reflect
this precisely"* at the point an intermediate mitigation attempt was implemented and live-tested,
not only at final closure. Root `AGENTS.md`'s own rule is explicit too: *"An open item is rewritten,
not appended to"* — `docs/BACKLOG.md` is forward-looking, always read whole, and a stale "not
dispatched" claim on a filed defect is exactly the "silent and cumulative" drift that rule exists
to prevent. Separately, the module-docs convention (`HISTORY.md`: "append an entry per delivered
change") has no entry for this delivered code change either, again breaking with the K-056
precedent, which got a dated entry at the same implemented-but-not-yet-live-verified stage.

**Suggested fix:** before/alongside landing this diff, rewrite (not append to) the three status
lines above to something like "Mitigation D (`v8`) implemented, awaiting live re-validation
(`docs/test-reports/salesperson-language-salience-report.md`, TP-007 protocol)" and add a dated
`HISTORY.md` entry for the code delivery, mirroring the K-056 2026-08-28 entry's shape. This is
squarely `teco`'s doc-curation lane per the coordination ledger, and naturally sequences as a
follow-up to U2 (this review) rather than blocking it — but it should happen before the ledger
calls the K-065 track done, and ideally before U3 (qa-engineer) starts, so QA isn't working from a
backlog entry that says no mitigation exists.

### What's solid

**Trap 1 (`config.model`/`config.tools` unchanged) — verified, not assumed.** Filtering the diff
to non-comment code lines (`git diff -- proof_defs.py | grep -v systemPrompt-line context`) shows
the *only* code changes are the `"version"` string and the `systemPrompt` tuple gaining four new
lines — no `tools` list or `model` line appears in the diff at all, confirming byte-identical carry
forward from `v7`. `requiredTools`, `maxIterations`, topology (`steps`/`transitions`) are likewise
untouched.

**Trap 2 (no `v6` reuse) — verified.** Version is `v8`, sequential after the documented
`v5`→(burned `v6`)→`v7` history; the module docstring's own `v6`-burned warning is left intact and
unedited.

**Trap 3 (chat-triggered-only property) — verified.** `"kind": "conversation"`, the `assistant`
step's `"start": True`/`"waitsForHuman": True`, and the guarded `ctx.endConversation` transition
are all untouched by the diff.

**The new sentence's placement and mechanism are sound, on the evidence available statically.** I
read `executor._assemble_messages` (`falkor-chat/server/falkorchat/executor.py:1243-1278`):
`systemPrompt` becomes the lone `role: "system"` message prepended once per node execution (i.e.
once per customer turn), and is present on *every* `llm.chat(messages, ...)` call within that
turn's tool loop — while the per-run `language` value only ever reaches the model via the
`CONTEXT:\n{...}` block, which is the *last* `user` turn and, per the same function's own
docstring, gets coalesced into the prior `user` turn via `_append_turn` whenever the thread already
ends on `user`. This independently confirms the ml plan's diagnosis (a JSON key buried at the tail
of a long, mostly-identical coalesced turn) and gives the new sentence a structurally different,
more prominent channel (system role, own paragraph, second position in a six-paragraph prompt) —
a plausible, non-redundant-in-effect lever even though its *text* is redundant with the existing
CONTEXT-block sentence. Whether a 3B model under `llama.cpp`/LM Studio actually weights the
`system` role more heavily than a coalesced `user` turn is an empirical/ML-methodology question
this review can't settle statically — flagging it as a candidate for a `data-scientist` opinion
before or alongside the live QA re-test (U3), not as a blocker on this diff: the plan's own "Risks
& open questions" already anticipates this ("if D alone doesn't move the measured rate, that's
itself evidence the cause sits deeper... than prompt salience can reach").

**The new regression test has real bite — verified by mutation, not just read.** I patched
`SALESPERSON_DEF["systemPrompt"]` in a scratch copy, replacing the new sentence with a
deliberately-weakened paraphrase ("Please also remember to reply in the right language.") that
keeps the same *intent* but drops the pinned phrases, and re-ran
`test_salesperson_v8_carries_v7_forward_and_adds_language_salience_sentence`: it failed exactly as
expected (`AssertionError: assert 'governs your entire reply' in ...`), then restored the file and
confirmed the full targeted suite is back to green (8/8). The test pins both load-bearing phrases
("governs your entire reply", "no drifting into a different language") plus the full `v7` superset
(`config.tools`, both `v7` sentences, topology) — a materially watered-down rewrite of the new
sentence would not pass. Minor, not worth a fix: the test doesn't assert the new sentence's
*position* in the prompt (only its presence), so a future edit that keeps the exact pinned phrases
but moves them to the very end of the prompt (undoing the "not buried in the tail" property this
mitigation is banking on) would still pass. Not requesting a change — the position claim is a
design rationale in the comment, not a contract worth a brittle ordering assertion, but worth
knowing if `v9` ever touches this sentence's placement.

**Doc sweep — independently spot-checked, no new gaps beyond the Major above.** Re-ran an
unfiltered `git grep` for `v7`/`v8` project-wide (not scoped to a pre-filtered list) and checked
every hit: `test_app.py:1001`'s `monkeypatch.setattr(app_mod.config, "TRIGGER_DEF_VERSION", "v7")`
is a generic config-plumbing fixture value, unrelated to the shipped constant (confirmed by
reading the surrounding test — it never touches `SALESPERSON_DEF`); `salesperson/README.md:26`'s
"`react-router-dom` v7" is an unrelated library version; every `docs/plans/salesperson-ui*.md`,
`docs/reviews/salesperson-ui-impl.md`, and `claude/teco/kaizen/history.md` hit is dated/point-in-time
narration of what was true when written (correctly left alone). All three of the actually-live
sites in `salesperson/README.md` (lines 106, 126, 132) and all of `falkor-chat/AGENTS.md`,
`falkor-chat/README.md`, and the three scripts' `v7`→`v8` sites are updated and consistent with
each other and with the shipped `proof_defs.py` constant.

## Open questions

- Should the `HISTORY.md`/`BACKLOG.md` rewrite from the Major finding above land as its own small
  `teco` unit before U3 (qa-engineer) starts, or is it acceptable for QA to work from the
  coordination doc's ledger (which is accurate) while the backlog itself stays stale until the
  whole K-065 track closes? I'd recommend the former (cheap, and QA reading `BACKLOG.md` cold
  should not see "no mitigation dispatched") but it's a sequencing call, not a technical one.
- Worth an explicit `data-scientist` sign-off (or at least a note in `salesperson-ui-ml.md`) on
  whether `system`-role salience is a reasonable mechanism to bank on for this model/serving stack,
  ahead of spending the U3 live-test budget — not a blocker, per the brief's own framing, but the
  plan's "Risks & open questions" already flags the possibility that D doesn't move the needle at
  all, and a cheap methodology sanity check before a 20-30 trial live run seems worth the marginal
  cost.
