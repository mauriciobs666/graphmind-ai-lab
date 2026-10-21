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

## Pass 2 — probe harness (2026-10-20)

### Scope & verdict

**Reviewed:** `salesperson/scripts/language_salience_probe.py` and
`salesperson/scripts/test_language_salience_probe.py`, both uncommitted and frozen (md5
`af33bcf2…` / `18b91d6d…`, unchanged at start and end of this pass), against the 2026-09-18
report's Protocol and Results definitions
(`falkor-chat/docs/test-reports/salesperson-language-salience-report.md`), TP-007
(`docs/test-plans/salesperson-ui.md`, the literal-concurrency variant as recorded in
`docs/test-reports/salesperson-ui-report.md`), `salesperson/scripts/load_demo.py`, and the storefront
routes it drives (`storefront_api.py`: `post_message`, `messages`, `state`; `storefront.py`:
`reserve_turn`, `enqueue_turn`, `_run_turn`, `turn_state`, `_last_turn`; `services.py`:
`read_messages`, `_next_ts`; `repository.py`: `read_thread_since`). Static only, plus scratch Python
against the harness's pure functions and mutation probes on scratch *copies*; no server, LM Studio
or HTTP call. Offline suite run: **21/21 green**.

**Verdict: approve with suggestions** — no blocker. **U6 can proceed and its figures can be
trusted, conditional on one thing:** the mandatory hand-read of *every* reply (`aggregate
--dump-replies`) is load-bearing for the "0 wrong" headline, not a formality — the classifier has
real false negatives in the dangerous direction (F1) and the trial-level table cannot see them
(F2). Both are **re-scorable from the stored `reply_text`**; nothing found requires a re-run. The
turn-driving plumbing is correct by trace (question 1 below); its test gap (F3) has no effect on
U6's figures.

CPG: considered, not relevant — the harness is a standalone stdlib script outside any loaded
`cpg_*` graph; the server paths it depends on were few and read directly.

**The four questions, settled:**

1. **Turn attribution — sound on the normal path.** `post_message` calls `reserve_turn` before
   answering `200`, so the first poll (after a 2 s sleep) cannot see `idle` before the turn starts;
   `enqueue_turn` clears the `lastTurn` latch strictly before `submit` and `_run_turn` marks it
   before its `finally` releases the slot, so `lastTurn` read at `idle` belongs to *this* turn; the
   agent's replies are written inside `maybe_trigger` (synchronous on the worker), so all of them
   exist at `idle`; `services._next_ts` makes `createdAt` strictly monotonic per process, so the
   plain-`>` `since` cursor neither skips nor re-delivers; one thread per participant, and
   `role == "assistant"` is exactly the agent's posts (`services.post_message`'s role rule).
   Attribution can slip only on abnormal paths (F4), and never across participants, so it cannot
   create or hide a wrong-language reply.
2. **Classifier validity — false negatives exist in the dangerous direction** (F1). Not
   compensated anywhere in the aggregate (F2).
3. **Aggregation — matches the report.** Trial-level denominator = trials with ≥1 replied turn
   (`aggregate`'s `trials_with_reply`; my mutant replacing it with all trials was killed),
   partial-trial exclusion by `trial_end`, Wilson matches every published figure (suite). Nothing
   silently dropped from the figures, but `errors` and 409s are never surfaced (F4).
4. **Fixtures — too uniform for plumbing** (F3): `run_participant` has zero coverage; 7 of 9
   mutants I planted survive, including two that would silently mask K-065.

### Findings

**F1 — major — classifier false negatives on genuinely wrong-language replies.**
*U6 impact: re-scorable (no re-run).* Evidence (`language_scores` / `classify_language`, scratch
`cls.py`/`cls2.py`, Appendix A): three distinct lexicon defects, each verified by running it:
(a) `con`, `en`, `una` sit in `SHARED_IBERO_WORDS` but are **Spanish-exclusive** (Portuguese has
`com`, `em`, `uma`), so they vote for nobody — `"Listo, tu orden está en camino."` and
`"Pedido realizado con éxito. Número de pedido: ORD-42."` to an `en` participant → `unknown`;
(b) `EN_WORDS` contains `in`, which occurs in the catalog name `USB-C Hub 7-in-1`
(`falkor-chat/scripts/seed_catalog.sh` `CATALOG`; the only one of 15 names that scores), so
`"Agregado: USB-C Hub 7-in-1 por $39.99"` to an `en` participant classifies **`en`, i.e. adherent**
— the one shape that makes a wrong reply look right, on exactly the turns (2–3) that list/add
products; (c) `do` is in both `EN_WORDS` and `PT_WORDS` (scratch `overlap.py`). Short replies made
of words outside every lexicon (`"Claro, te ayudo con eso."`, `"Buenas! Bienvenido a nuestra
tienda."`) → `unknown`. **Suggested improvement:** move `con`/`en`/`una` to `ES_WORDS`, drop `in` and
`do` from `EN_WORDS` — I ran this on a scratch copy: (a)/(b)'s strings flip to `es`/`unknown`, the
report's pt reply and the suite's clear cases still classify correctly, suite 21/21 green (it
constrains this little; my regression strings are in `fix.py`). That narrows but cannot close the
gap, so do not hand over a lexicon list as the fix: **require a coverage probe** — a small
hand-written corpus of genuine replies, one per (scripted turn × reply language), crossed with
every participant language and with/without each catalog name embedded, asserting no
wrong-language reply classifies as the participant's own language (and counting `unknown`s). Bump
`CLASSIFIER_VERSION` to `v3`. **For U6 now:** the hand-read must cover replies classified
*adherent* too, not only flagged/unclassified ones — (b) is invisible otherwise.

**F2 — major — the trial-level table cannot see unclassified replies.**
*U6 impact: re-scorable.* `aggregate` counts a trial as affected only from `wrong`;
`trials_affected` and the per-trial `!W` mark ignore `unknown`, which F1 shows is where genuinely
wrong replies land. The turn-level table does show `Unclassified` (and `adherent` correctly excludes
it — conservative), but the trial-level headline — the figure the stakeholder decision turns on
(0/23 last pass) — reads clean regardless. **Suggested improvement:** add `trials_with_unclassified`
(+ ids) beside `trials_affected` in `aggregate`/`render`, mark `?U<n>` in the per-trial cells, and
state in the U6 report that "0 affected" holds only once every unclassified reply has been
hand-resolved. Re-derivable from the stored file with no live calls.

**F3 — major (reuse risk) — `run_participant` is untested; 7 of 9 mutants survive.**
*U6 impact: none on figures* — I traced the shipped plumbing as correct (question 1), and this
finding is about the next run. Scratch `mut.py` (Appendix B) planted mutants on copies and ran the
real suite: **survived** — cursor never advances (`since=0` every turn), 409 not waited out, idle
check inverted, `dead_turn` hard-wired `False`, role filter dropped, every participant joined as
`en`, unknown counted adherent; **killed** — trial-level denominator, `next_trial_index` off-by-one.
Two survivors are K-065-masking: dropping the role filter splices the English user prompt into
every reply (an `en` participant's Spanish reply could then tie or win `en`), and the cursor mutant
re-reads earlier replies into later turns. Fixtures are also uniform: `AggregateTests` has no `es`
participant, no `unknown` reply, no multi-row reply, no 409/timeout record. **Suggested
improvement:** a fake storefront on stdlib `http.server` (session/messages/state) that replies in a
language derived from the session's `language`, emits a two-message turn, one `lastTurn:"failed"`
turn and one 409 — enough to kill every survivor above; add an `es` participant and an
unclassified reply to the aggregate fixture.

**F4 — minor — abnormal paths misattribute a reply to the next turn, and the aggregate hides
them.** *U6 impact: no effect on wrong-language figures (attribution never crosses participants);
small effect on turn-level denominators/dead-turn tallies, re-scorable — every trigger is stored.*
`run_participant` reads messages and advances the cursor as soon as the poll loop exits, so on a
**timeout** (ceiling 200 s is measured from the POST, and the 180 s is a per-provider-call
timeout, `llm.py` — a multi-call turn can exceed it), a **state-poll error** (`break`), a
**messages-read error** (cursor kept, reply picked up next turn) or a **POST transport error**
(`-1`, no poll), a later reply lands in the *next* turn's record; that next POST can then return
409, and the record carries a `prompt` that was never posted. A timed-out turn also records
`dead_turn: False` (latch clear while `thinking`), inflating `no_reply_not_dead` — the figure the
last report used as evidence that the DEF-3 latch never misses. **Suggested improvement:** in
`aggregate`, surface `turns_with_errors`, `post_409`, and exclude `timed_out` from
`no_reply_not_dead`; optionally flag records whose `reply_created_at` precede `posted_at`. In
`run_participant`, after a timeout/poll error, keep polling (bounded) before reading, or skip the
next POST while a turn is still in flight.

**F5 — minor — dominant-signal scoring over a joined turn misses code-switching.** *U6 impact:
re-scorable only approximately.* `reply_text` joins all assistant rows with `\n\n` and scores the
whole; `"I've added the Wireless Mouse Pro to your cart. Your total is $29.99. ¿Quieres finalizar
el pedido?"` → `en` (Appendix A). The 2026-09-18 definition is the dominant language, so this is
within contract, but a partial swap is invisible. Row boundaries are lost (`\n\n` also occurs
inside a reply), though `reply_msg_ids` survive. **Suggested improvement:** store `reply_texts`
(list) per record; add a secondary "any paragraph classified wrong" flag in `aggregate`. For U6,
splitting on `\n\n` gives a superset of segments — a conservative re-score.

**F6 — minor — two exceptions escape `http_json` and abort the whole run.** *U6 impact: no effect
on figures (the trial lacks `trial_end` and is excluded; `--resume` continues), costs live window
time.* `json.loads` on a `200` body is outside the `JSONDecodeError` guard, and
`http.client.IncompleteRead` (an `HTTPException`, not `OSError`) is uncaught; either propagates
through `f.result()` in `cmd_run`. **Suggested improvement:** catch both in `http_json` and return
a `-1`-shaped error like the transport branch.

**F7 — nit.** `ajudá-lo`/`ajudá-la` can never match `_WORD` (no hyphen) — dead entries (scratch
`overlap.py`); unaccented `aqui` is Portuguese-only but sits in the shared set; `--resume` after a
mid-trial kill skips that trial index, so the run ends one complete trial short of `--trials`
(harmless, but say so in the report). *U6 impact: none.*

### What's solid

- The protocol plumbing is right where it matters: `since` cursor from server `createdAt`,
  409-never-re-post, poll-until-`idle` and `lastTurn` read at `idle` all line up with the server's
  actual ordering guarantees (traced, question 1).
- The v1→v2 fix is real: shared words no longer vote, the report's misread pt reply classifies
  `pt-BR`, and `unknown` reduces `adherent` rather than inflating it.
- Re-scorability by construction — raw `reply_text`, `reply_msg_ids`, `reply_created_at`,
  `post_status`, `timed_out`, `errors` per turn, fsync'd per record; `aggregate` ignores the stored
  `classified_at_run` (tested with a lying fixture). This is what makes every finding above fixable
  without a re-run.
- Wilson is independently tested (published figures, quadratic roots, symmetry), and partial-trial
  exclusion and the trial-level denominator match the report's definitions.

### Open questions

- Should the U6 report's headline be computed with the shipped v2 classifier (comparability with
  0/23) or with a v3 that applies F1's lexicon fix? My recommendation: report v2 for comparability,
  plus a v3 re-score and the hand-read resolution of every `unknown` — all three come from the same
  stored file.

### Learnings to capture

- *Fact:* a lexicon language classifier's "unknown is never wrong" rule shifts its false negatives
  into the `unknown` bucket, so any trial-level "affected" metric built only from `wrong` is
  anti-conservative; and English product names in a catalog can contain English function words
  (`7-in-1`) that score a Spanish reply as English. *Evidence:* scratch runs above. *Suggested
  home:* knowledge base (qa-engineer / data-scientist).

### Appendix A — classifier probes (scratch, output verbatim, abridged)

```
$ python3 -I cls2.py   # participant -> classified, wrong?, scores
en     -> unknown  wrong=False {'en': 0, 'pt-BR': 0, 'es': 0} | 'Buenas! Bienvenido a nuestra tienda.'
en     -> en       wrong=False {'en': 1, 'pt-BR': 0, 'es': 0} | 'Agregado: USB-C Hub 7-in-1 por $39.99'
en     -> unknown  wrong=False {'en': 0, 'pt-BR': 0, 'es': 0} | 'Pedido realizado con éxito. Número de pedido: ORD-42.'
en     -> unknown  wrong=False {'en': 0, 'pt-BR': 0, 'es': 0} | 'Listo, tu orden está en camino.'
en     -> en       wrong=False {'en': 5, 'pt-BR': 0, 'es': 4} | "I've added the Wireless Mouse Pro to your cart. Your total is $29.99. ¿Quieres finalizar el pedido?"
es     -> unknown  wrong=False {'en': 0, 'pt-BR': 0, 'es': 0} | 'Pronto! Pedido realizado.'
$ python3 -I cls.py
pt-BR  -> unknown  wrong=False {'en': 0, 'pt-BR': 0, 'es': 0} | 'Claro, te ayudo con eso.'
pt-BR  -> unknown  wrong=False {'en': 0, 'pt-BR': 0, 'es': 0} | 'Order placed! Order ID: ORD-1234. Total: $49.99.'
(full-sentence Spanish/Portuguese/English swaps — e.g. 'Tenemos varios productos disponibles: …',
 'Olá! Em que posso ajudar?', 'Your order status is: pending.' — all correctly flagged wrong)
$ python3 -I overlap.py
EN&PT frozenset({'do'})   EN&ES frozenset()   PT&ES frozenset()
PT unmatchable: ['ajudá-lo', 'ajudá-la']
$ python3 -I names.py     # catalog names scored
USB-C Hub 7-in-1 Accessories {'en': 1, 'pt-BR': 0, 'es': 0}   (15 names; only one scores)
$ python3 -I fix.py       # con/en/una -> ES_WORDS; in, do removed from EN_WORDS
'Agregado: USB-C Hub 7-in-1 por $39.99'                  en -> unknown
'Pedido realizado con éxito…' / 'Listo, tu orden está en camino.' / 'Claro, te ayudo con eso.'  -> es (wrong=True)
'Buenas! Bienvenido a nuestra tienda.'                   still unknown
regressions: report pt reply -> pt-BR; 'Hello! How can I help you today?' -> en; 'What do you have in mind?' -> en
(fixed scratch copy: suite 21/21 green)
```

### Appendix B — mutation probes (`mut.py`, scratch copies, real suite)

```
SURVIVED cursor never advances (since=0 every turn)
SURVIVED 409 treated as error, no wait
SURVIVED idle check inverted to thinking
SURVIVED dead_turn from wrong field
SURVIVED role filter dropped (user rows counted)
SURVIVED language not sent at join
KILLED   trial-level denominator = all trials
SURVIVED unknown counted adherent
KILLED   next_trial_index off by one (reuses last)
```

Scratch scripts: `/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/01483502-06e5-42ad-9b05-36efaa8d1ff1/scratchpad/`
(`cls.py`, `cls2.py`, `overlap.py`, `names.py`, `fix.py`, `mut/mut.py`).
