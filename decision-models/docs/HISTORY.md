# Change History — decision-models

> Dated log of actual changes to the `decision-models` component. Most recent first.

## 2026-11-10 — typed-decision-models U5.9 fix round — reopen, unboundedCell, probe, spend projection

**What:** `docs/plans/typed-decision-models6.md` §3.8's decline/reopen, P̂ = 0, probe and
spend-projection bullets and its §5 row "U5.9 fix round" (plan6 v1.25); note §Q1.3; impl Pass 27
(P27-m1, m2, n1, n2) and Pass 28 (P28-M2). Plan, note and live specs untouched; `install.py` and
`finalize` are U5.10's.

- `campaign.py`: `resolve --reopen-topup` (`topup-reopened`; valid only with no halt open and a
  decline in force). `topup_counts` raises the typed `UnboundedCell(CampaignRefused)` carrying
  `(label, stratum)`; `queue_topup` records the `unbounded-halt` event (`voidClass`
  `unboundedCell`; the label, stratum, the stratum's labelled-item count and the Wilson 95% bound
  `z² / (n + z²)`), so `driver run` stops on a halt and writes no `store/refusal`. A plain
  `resolve` is refused for it; `--decline-topup` is valid for `spend` or `unboundedCell`.
  `resolve --probe-topup` appends `probe-requested {designStratum}`: `fold` keeps it in
  `State.probe` until a `topup-plan`, `topup-declined` or `generation-restart`, and `queue_topup`
  plans it (`topup-plan`/`spend-halt` with `probe: true`, 25 seeds of that stratum alone) before it
  calls `topup_counts`, so a probe that spend-halts and is let through plans at the same
  `genRound`. Zero hits re-halts, printing the bound for the stratum's labelled items.
- Spend projection: `project_topup(topup_seeds=...)` divides by round 1's drawn `seeds`
  (`State.spend(1)` gains `seeds` and `totalTokens`) at the live labelling rate; the threshold is
  half of `totalTokens`, which include the retired attempts of a `labelling-restart` or
  `tranche-restart` and not a `generation-restart`'s. New `Batch.retired_by` carries the cause
  (a generation restart overrides an earlier cause; a later relabel never claims a retired
  batch). The spend halt's reason prints both totals.
- Pass 27 minors: m1 `rubric_hash` pinned by an independent sha256 per point and a
  `label_answer_section` edit leaves it unchanged; m2 guard 1 (`ingest`'s stop on the batch that
  trips the tranche rule) and guard 2 (`tranche_due`'s settled conjunct) have constructed cases,
  guards 3 to 5 (`_check_tranche`'s wait on another halt, `_fold_resolution`'s re-check,
  `_draw_topup`'s "a run has issued") are deleted: no shipped flow reaches them (the two halt
  recorders that could be open then are pilot-only, a pilot has no tranche; `begin_round` refuses
  loudly where guard 5 skipped silently), though called by hand in a `full` campaign a cue-check
  halt is now overwritten by the tranche halt (Pass 29 P29-m1); n1
  `screenHits.byPattern` counts every reason of a hit (a seed with two hits and a
  `["screen", "formMismatch"]` hit are tested); n2 `AGENTS.md` and this file name `_draw_topup` as
  the `extend=True` caller (the U5.9 entry below is corrected in place), and `turnCut`'s docstring says it ignores `render_guard_input`'s `MAX_CHARS`
  drop. `begin_round(queue_runs=False)` has a direct test.
- Choices: the events are `unbounded-halt` and `probe-requested`; the cause field is
  `Batch.retired_by` (`generation-restart`, `labelling-restart`, `tranche-restart`).

**Verified:** `pytest -q` from `decision-models/` 1614 passed, none skipped (1588 before: +25 in
`test_topup.py`, +1 in `test_diagnostics.py`); `ruff check .` clean. 36 scratch-copy mutants, 34
redden a test, among them seeds replaced by accepted items in the ratio, the probe stratum taken
from the first or last design stratum or `None`, a live-only or live-rate-for-total threshold,
retired attempts in the rate, a retirement cause that any restart or no restart overrides, a plain
`resolve` clearing `unboundedCell`, an uncaught `UnboundedCell`, a reopen that keeps the decline,
an unpinned or never-cleared probe, and the Wilson bound over seeds drawn. Two survived as
equivalents (a no-op edit, and "first design stratum" when guard's first is the halted one).

## 2026-11-10 — typed-decision-models U5.9 — rounds, top-up, spend halt, tranche, diagnostics

**What:** `docs/plans/typed-decision-models4.md` §3.6 and its §5 row U5.9; plan5 §3.2, §3.6 and its
U5.9 row; plan6 §3.8 and its U5.8a/U5.9 row; carried ML P3-n2 and P4-n2 and impl Pass 26's open
minors. Plan docs and live specs untouched.

- `campaign.py`: `topup_counts` (pure planner), `project_topup` (pure spend projection),
  `Campaign.queue_topup`/`_resume_topup` (called from `next`; `_draw_topup` draws every stratum,
  the first with `begin_round`, the rest `extend=True`, before a run issues, then cuts one
  interleaved pool;
  `begin_round(queue_runs=False)`), the `spend-halt`, `topup-plan` and `topup-declined` events,
  and `resolve --decline-topup` (also `driver.py`). Message type's round 1 votes a 250-item
  tranche first (`tranche-start`; `tranche` on its `vote-start`s and batches): `tranche_failed`
  halts with `rubric`, and `resolve` relabels it (`tranche-restart`) only under a changed
  `rubric_hash`. A generator restart resets all of it. `State` also keeps `template_log`.
- `diagnostics.py`: `diagnostics.json` per plan4 §3.6, plus the round-1 `CueCheck` reprint
  (`failed` omitted), P3-n2's turn-cut split and P4-n2's hand-off-example count (nothing dropped).
- Pass 26 minors: the join-in generator test pins the rule, the item wrappers are pinned by
  value (a slice before the filter now reddens), `agentName` is pinned unstripped, and
  `layout_block` reads only the first indented block (fixed, with a test).
- `tests/test_votes.py`'s message-type bench switches the tranche and the planner off: six items
  are all tranche and every cell short, and those tests are about the vote path.
- No recorded generator-template hash predates the `generator.md` edit: no campaign ledger
  exists outside pytest's temp directories. `install` does not yet record `requestTemplate` or
  `rubric` (U5.10, plan5 §3.3), so the template hash does not cover `generator.md` at all yet.

**Verified:** `pytest -q` from `decision-models/` 1588 passed, no skip or xfail (1527 before:
+39 in `test_topup.py`, +19 in `test_diagnostics.py`, +3 net in `test_render_layout.py`);
`ruff check .` clean. 35 scratch-copy mutants (the intent-filtering sampler, halting at exactly
half, three token classes, the tranche rule at `>=` and 4%, the rest voting before the tranche
passes, a relabel under an unchanged rubric, an unrecorded decline, floor for ceiling, min for
max, one stratum's size for all, unpooled strata, no interleave, `extend` never or always,
spend approval ignored, a reused RNG seed, an unrecorded tranche seed, and the diagnostics and
Pass 26 ones) each redden a test; one (`tranche_drops` counting every tie) first survived, as
the rest's ties come after the check, and a test now holds the rest's ties out of the count.

## 2026-11-10 — typed-decision-models U5.2b-render — new-ability renderers, render vectors, 6-turn rule

**What:** `docs/plans/typed-decision-models5.md` §3.4 and its §4 U5.2b-render row (over plan6 v1.22).
Plan docs and live specs untouched.

- `render.py`: `render_join_in_input`/`render_join_in_item` and `render_message_type_input`/
  `render_message_type_item`, transcribed from model-bench's `modelbench/scoring/agreement.py`
  (U4.1); join-in normalizes its prior turns through `normalize_turns`. `exchange.RENDERERS` maps
  all three roles; the "no renderer yet" refusal now fires only for an unknown role.
- Vectors, byte-copied and `cmp`-equal to model-bench's: `render-join-in.jsonl`,
  `render-message-type.jsonl`, `normalize-turns.jsonl` (the last pins `normalize_turns`, which had
  hand-derived tests only). `tests/test_decision_contract.py` guards each copy and runs each row.
- `tests/test_render_layout.py`: a fixture per new-ability point rendered through `render.py` must
  match the rubric's "What an item looks like" block read from `rubric.md`; the registry; the
  join-in 6-turn cut; the new message's speaker rule (no vector line has both name fields).
- Join-in 6-turn rule: `points/join-in-judge/generator.md` now says the judge sees only the last
  six `priorTurns`, so the agent's job sits inside them. `data-scientist` confirmed it (plan5
  §3.4, 2026-11-10).
- The strict xfail `test_every_point_has_a_label_renderer[...]` is deleted (it XPASSed as planned);
  the no-renderer refusal test uses an unknown role.
- Carried, test-only (`tests/test_pilot.py`): a dropped vote-1 batch stays in
  `dropped.retryCapExceeded.items`, and a screen batch dropped as `safeguardSkipped` is subtracted
  from that count only.

**Verified:** `pytest -q` from `decision-models/` 1527 passed, no xfail (the entry before this one
recorded 1480 passed, 2 xfailed); `ruff check .` clean. Scratch-copy mutants (renderer separator, 6-turn cut, new-message cut,
speaker rule, message-type cuts, item-level argument, registry mapping, generator wording, each
vector copy, and the two `_runs_report` mutants) each redden a test. The speaker-rule mutant first
survived (no vector line carries both name fields on the new message), so a test was added.

## 2026-11-10 — typed-decision-models U5.8c fix round — bounded added blocks, report fields

**What:** impl review Pass 24 (P24-M1, M2, m1, m2, n1 to n3), plan6 v1.22 §3.8 "The pilot's added
blocks are bounded" and the §5 U5.8c row. Plan docs and live specs untouched.

- `campaign.py`: `queue_pilot_block` draws at most `pilot.PILOT_ADDED_BLOCK_LIMIT` (3) added main
  blocks per halt epoch, then `record_pilot_block_halt` appends the batchless `pilot-block-halt`
  (`mainRounds`; reason and `voidClass` `pilotBlocks`; its own event, not `cue-check-halt`). `fold`
  keeps `State.block_halt_base` (the main-block count at the last such halt), so a `resolve` opens
  a new epoch; `next` returns `Stop("halted")` at once. `_restart_generation` resets the epoch and
  the cue-halt count and resolved flag (P24-n2: fixed, not documented). `record_cue_check_halt` and
  `record_pilot_block_halt` refuse while any halt is open (P24-n1).
- `pilot.py`: the `report` refusal for `n < 80` names an open `pilotBlocks` halt; the report gains
  `planFactor` and `assumes` and the docstring says what the factor is (P24-m1); `dropped` leaves
  out dropped screen-vote batches, which are `droppedScreenVote` (P24-n3: fixed; their items keep
  votes 1 to 3).
- The reviewer's suggested fixes all held when executed; none was wrong.
- Tests (`tests/test_pilot.py`, 81 → 94): the bound (blocks 1 to 3 then `pilot-block-halt` with
  `n` = 0, `resolve` then exactly three more and a second halt), 79 against 80 items after block
  3, the epoch and cue bookkeeping reset by a generator re-pin, both recorders refusing with a halt
  open, a crash after 3 of 10 screen batches, `--cap 25` read back (and through `init --phase
  full`), the recheck report keeping the pilot's `n`/`Δ`/cap, two voids counted as two, `planFactor`
  and `assumes`. Suite 1467 → 1480 passed, 2 xfailed. Twenty-one scratch-copy mutants red (limit
  3→2 and 3→4, `>=`→`>`, epoch never re-armed or counting all blocks, no `max(.,1)`, restart keeping
  the epoch or the cue bookkeeping, `next` ignoring the halt, the halt's reason or event reused
  from the cue halt, either recorder without its guard, the refusal text, dropped screen items,
  `planFactor`, `assumes`, Pass 24's M13, M21, M28 and M51).

## 2026-11-09 — typed-decision-models U5.8c — the guard pilot: screen, cue check, report, recheck, decision

**What:** plan4 §3.8 and its §5 row U5.8 (Pilot part), plan5 §3.6 (cue check, report, recheck,
spend gate), plan6 §3.8 (blocks are rounds). Not built: the message-type tranche and top-up (U5.9),
`finalize` (U5.10). Plan docs and live specs untouched.

- `pilot.py` (new; `python -m decisionmodels.pilot start|screen|report|recheck|record-decision`).
  Pure: `screen` (`n`, `Δ`, cap: `10·Δ ≥ n` gives 25, else 50; `n < 80` needs a block),
  `cue_check` (exact stratified conditional test over difficulty × intended answer, `Fraction`
  arithmetic, fails at `p ≤ 1/20`), `fit_line` (labeller `F`, `c` per token class, two batch sizes
  required). Campaign-aware: `start` (the first main block, 100 seeds), `build_report`/`report`
  (the one evaluator; derives `missingCueCheck.state` from the ledger; counts only; refuses a partly
  voted block and a decided report), `recheck` (the recheck block of 100), `record_decision`
  (single-write; a `proceed` goes through `campaign.spend_gate_refusal`, the one predicate).
- `labels.py`: `SCREEN_VOTE` (4) and `SCREEN_BATCH_CAP` (10). `campaign.py`: `begin_round(...,
  pilot_block=)` records `pilotBlock` on `round-start`; `State.round_votes` gives a main block the
  fourth vote and the recheck block three; `State.three_votes` keeps the screen vote out of
  `label_votes`/`vote_ties`; `round_voted`, `screen_votes`; `queue_pilot_block` (called by `next`)
  draws each added main block of 50 while `n < 80`; the `cue-check-halt` event
  (`record_cue_check_halt`) is a batch-less halt that `resolve` clears (`cue_halt_resolved`).
- Choices the plans leave open (the coder's, for the gate): `start` is a `pilot.py` subcommand
  (round 1 is not drawn by `init`, which would change every pilot-phase fixture); the reprojection
  uses 1.3 × target items per point (the note's 2,340 labelled for 1,800 kept), runs of 25 and
  three votes at the pilot's cap; the generator's mean is over first-pass runs; the labeller's fit
  is over every ingested label run, recheck included; `report` is refused once a decision is
  recorded; an added block has no upper bound beyond `n ≥ 80`.
- Tests: `tests/test_pilot.py` (new; 81 tests). Suite 1386 → 1467 passed, 2 xfailed. Scratch-copy
  mutants: 73 run, 71 red; two survive and are argued equivalent in the gate report.

## 2026-11-05 — typed-decision-models U5.8b fix round — half-queued votes, label request, restart, resolve

**What:** impl review Pass 22 (P22-M1, M2, m1 to m3), plan6 v1.21 §3.7 and §3.8 "Votes".

- `campaign.py`: `State.next_vote` treats a recorded vote as incomplete until its live batches hold
  every `vote-start` item (less those of dropped label batches), and `queue_votes` queues the
  missing items into the same vote in the recorded order (P22-M1; the votes half of
  `unbatched_seeds`). `_restart_labelling` retires every `label` batch, not only vote batches
  (P22-m1; this supersedes the "smoke fixture's label batch stays" line of the entry below).
  `Campaign.resolve` returns the lines `driver resolve` prints: the cleared halt and the halt it
  raised, else the "resumes" line (P22-m2).
- Tests: `tests/test_votes.py` (a kill after the 1st, 2nd and 3rd label `queue_batch` of 8 items
  at cap 3; the recovered order and cap; a vote in flight in one round holds back another; the
  request shows each key its own item in all three votes; a plain label batch retired by a
  re-pin; a two-vote item is no `voteTie`; `vote_all` stops at 60 batches instead of looping),
  `tests/test_generation_rounds.py` (`driver resolve` output), `tests/test_install.py` (queues a
  fresh label batch after its repin, since the smoke batch is now retired). Twelve scratch-copy
  mutants red (no held check, tail never queued, first item shown to every key, vote-only restart,
  two-vote ties, either in-flight guard removed, constant `genRound` or `vote`, three `resolve`
  outputs). Both in-flight guards stay: they differ across rounds, and a test now reaches each.

## 2026-11-05 — typed-decision-models U5.8b — votes: majority of three, `labelVotes`, `voteTie`

**What:** plan4 §3.6 "Votes" and its §5 row U5.8 (Votes part), plan5 §3.2, plan6 v1.21 §3.7 and
§3.8 "Votes". Not built: the message-type 250-item tranche (U5.9), the pilot cap/screen vote
(U5.8c).

- `labels.py` (new, pure): `vote_key` (`"true"`/`"false"` for a predicate, the label for a choice),
  `label_votes` (the whole answer space, zeros included, summing to 3), `majority` (two of three,
  or `None`: three distinct choice labels).
- `campaign.py`: `next` calls `queue_votes` when nothing is pending. A round votes once every seed
  of its `genRound` is accepted, dropped on a third hit or in a dropped generation batch
  (`State.round_settled`); only that round's accepted items vote. Each (round, vote) appends
  `vote-start {genRound, vote, rngSeed, items}` (the seed from `fresh_rng_seed()`, so it joins
  `held_rng_seeds`; the items are shuffled on it) and queues label batches of at most
  `labelBatchCap`, with `genRound` and `vote` in their `queued` extras; `prepare` cuts what does
  not fit in `P_MAX` pages into the next batch of the same vote. Votes run vote-major, one request
  at a time, and vote *v+1* starts when every batch of vote *v* is ingested or dropped. A `votes`
  event (item → answer) is appended before the run's `outcome` and applied only at the ingested
  one. A dropped label batch (`retryCapExceeded` or `--skip-batch`) leaves its items out of the
  round's later votes and is counted once by `State.dropped()`. `State.label_votes`,
  `vote_ties` (the `voteTie` drops) and `reshuffle_seeds` (the pilot report's reshuffle seeds,
  for U5.8c) read the folded votes. `labelling-restart` clears the votes and retires the vote
  batches (the smoke fixture's label batch, which holds no vote, stays; items and held seeds kept;
  votes repeat from vote 1); `generation-restart` clears
  the votes with the items. `status` prints the `voteTie` count.
- Event names (the plan leaves them to the coder): `vote-start`, `votes`.
- Tests: `tests/test_votes.py` (new; 19 tests). Twenty-four scratch-copy mutants (a plain majority
  without `voteTie`, no zeros, a boolean key, one shuffle reused, an unheld seed, a dropped or
  skipped batch's items kept, restart variants, the answer map and the vote number, settled
  always, no cap chunking, all rounds' items, an overflow that loses its vote, votes applied at
  the event, the smoke batch recording, no ties, two votes, vote order) are red; two are
  equivalent (a majority of `>= half` over exactly three votes; the vote-start guard that
  `queue_votes` already makes unreachable). Suite 1358 → 1377 passed, 2 xfailed.
  `tests/test_generation.py`'s `drain` now stops when no
  generation is left (before `next` would queue a settled round's votes), and four assertions that
  `next` was `Stop("done")` after generation now expect the round's first vote.

## 2026-11-05 — typed-decision-models U5.8a4 — halt lists every pool, joiner = non-word run

**What:** impl review Pass 21 (P21-m1 to m3, n1), plan6 v1.21 §3.8 and its §5 row "U5.8a, U5.9",
note v1.18 §Q1.2 "Hits".

- `campaign.py`: the `template` halt record has a `round` part (`genRound`, `reasons`; absent if no
  per-round pair crossed) and a `pools` list (`templateHash`, `reasons`, `poolRounds`, `poolSeeds`,
  `poolHits`, in hash order); every listed pair is marked fired. `reason` is the round's first
  reason when `round` is present, else the first pool's; `genRound` is the closing round's.
  `status`, `next` and `resolve` print every entry (`_halt_text`). A pooled halt raised by a drop
  (`retryCapExceeded`) now reaches `IngestResult.stop` as `halted` (P21-n1).
- `screen.py`: rule (a)'s joiner is any run of non-word characters (`(clear; difficulty)` and
  `(clear - advance)` hit) and rule (d) is deleted, since (a) covers it. A bracket like
  `[intended = yes]` now reports `meta:yes` (via (a)), not `meta:intended`.
- Tests: `tests/test_generation_rounds.py` (a close crossing two pools, a per-round and a pooled
  crossing in one close, pools of 299 and 300 seeds, a hit at a run's last, first and middle entry,
  an item with two reasons, the `--skip-batch` close, the drop-raised halt through ingest),
  `tests/test_screen.py`. Eighteen mutants (halt-record parts, the pool order and firing, the
  reason, `genRound`, the drop and `--skip-batch` closes, the 299 and 301 floors, a first or last
  entry, a second reason, the printed pools, the joiner variants and an empty bracket) each redden
  them, in a scratch copy. Suite 1346 → 1358 passed, 2 xfailed.

## 2026-11-05 — typed-decision-models U5.8a3 — pool per run hash at 5%, extend after issue refused

**What:** impl review Pass 20 (P20-m1 to m3, the rule (a) ruling), plan6 v1.20 §3.8 and its §5 row
"U5.8a, U5.9", note v1.17 §Q1.2 "Hits".

- `campaign.py`: `POOL_PERCENT` = 5 for the pooled test only (`HALT_PERCENT` stays 4). A round closes
  when every first-pass run is ingested or dropped (`State.round_closed`); a dropped run adds
  nothing and does not hold it open. At close each ingested run adds its seeds and hits by reason
  to the pool of its own issue-time hash (`State.run_hash`), so a round split by an `install`
  re-run feeds two pools; membership is `drawn_in < 25` at close. A drop (`retryCapExceeded` or
  `--skip-batch`) can close a round and so raise the pooled halt. `begin_round(extend=True)` is
  refused once a first-pass run of that `genRound` has issued; the duplicate-stratum check is per
  `genRound`.
- `screen.py`: rule (a) takes whitespace as a joiner, so `(clear advance)` hits; `(no reply)` and
  `(other question)` are accepted false hits, pinned as such.
- Tests: `tests/test_generation_rounds.py` (pool rounds of 10, 12 and 20 seeds, the 5% cutoff at 15
  and 16 hits in 300 seeds, the `install` split, a dropped run, the interrupted-planner `extend`
  refusal issued and ingested, a stratum reused across rounds), `tests/test_screen.py`. Eighteen
  mutants (pool percent 4 and 6, the completing run's hash, a dropped run holding the round open or
  counted, `extend` after issue, the per-`genRound` check, the hits' source and reason key, the
  pool size source, the membership bound, the 300 floor, the halt's `genRound`, the drop hook,
  `intended` as a keyword, whitespace not a joiner, the two-hit floor) each redden them, in a
  scratch copy. Suite 1331 → 1346 passed, 2 xfailed.
- Open: a close that crosses two pools at once reports the first pool's `poolRounds`, `poolSeeds`
  and `poolHits` in the halt; `diagnostics.json` (U5.10) should record each.

## 2026-11-05 — typed-decision-models U5.8a2 — 4% halt, pooled halt per round, issue-time pins

**What:** impl review Pass 19 (P19-M1, m1 to m4), plan6 v1.19 §3.8 and its §5 row "U5.8a, U5.9",
note v1.16 §Q1.2 "Hits" and "Halt".

- `campaign.py`: `HALT_PERCENT` 5 → 4 (an 880-seed round halts at 36 hits, not 35; a 50-seed round
  at 3, not 2). The pooled halt now pools only rounds under 25 drawn seeds, adds a round when its
  first-pass runs have all ingested (not after each run, the sequential-test defect of P19-M1),
  tests once the pool holds 300 seeds (more than 4%, two hits), once per (hash, reason); the halt
  names the completing round's `genRound` (never `None`), `pooled: true`, and `poolRounds`,
  `poolSeeds`, `poolHits`. A round pools under the hash of the run that completes it. `Attempt`
  keeps the model pins and instructions in force at `issued`, and the audit reads them (P19-m1).
  `begin_round(extend=True)` is refused for an unstarted `genRound`, without a `design_stratum`,
  and for a stratum that already has a `round-start` (P19-m3).
- `screen.py`: the bracket rule is note §Q1.2 "Hits" rules a to d exactly (trailing `. ! ? ;`
  stripped, keyword then listed token within two words forward, `intended` and `difficulty` as
  in-bracket keywords, the two-word `difficulty`/`intended` form) (P19-m2).
- Not done: `diagnostics.json` does not exist yet (U5.10). It must record each pooled halt's hash,
  reason, `poolRounds`, `poolHits` and `poolSeeds`, which the halt dict already carries.
- Tests: `tests/test_generation_rounds.py`, `test_generation.py`, `test_screen.py`. Mutants of the
  new rules (screen rules a to d and the window, the percent, the pool floor and ceiling, the
  completion test, the pool denominator and hit sources, the pins and instructions snapshot, the
  three `extend` refusals, the retired-run stop and the restart clears) each redden them, in a
  scratch copy. Suite 1285 → 1331 passed, 2 xfailed.

## 2026-11-04 — typed-decision-models U5.8a fix round — bracket rule, plan6 §3.8, generator restart

**What:** impl review Pass 18 (judged, not blindly applied), plan6 v1.18 §3.8 and its §5 row
"U5.8a, U5.9", note §Q1.2 "Hits" (amended 2026-11-04).

- `screen.py`: bracket groups are found by a stack, so a stray closer (`:)`, `1)`) no longer
  cancels a later opener (P18-M1: the net-depth count was the defect). A bracket hits only when its
  whole content is listed tokens joined by `/`, `,` or `|`, or holds a listed token followed by `:`
  or `=`; the keyword rule is unchanged. Of Pass 18's strings, "a) (b clear c)" now passes by the
  amended rule; "a) (clear)" pins the same stray-closer case. Not taken (the note's Hits grammar
  does not call for them): U+2011/U+2013 hyphens in `near-miss`, "(near miss)" (prose), uppercase
  ids, "answers: yes", `Intended:` as a keyword, `{}` and fullwidth brackets.
- `campaign.py`: balanced cut (`balanced_runs`, also for regeneration pools); halt needs two hits
  and over 5%, keyed by `genRound` over every `round-start` of it, plus a pooled halt per
  (generator-template hash, reason) over ingested first-pass seeds; `Attempt.expectations`
  snapshots the `expectations` at `issued` and the audit reads it; `next` queues `round-start`
  seeds no batch holds; `generation-restart` resets the folded view and retires every batch
  (open runs are reconciled and recorded, changing nothing); `begin_round` refuses a started
  `genRound` unless `extend=True` (a top-up stratum), a `generated` event replaces its staged
  entry whole (P18-n2), and a recovery re-ingest that halts names the halt (P18-n3).
- Tests: `tests/test_generation_rounds.py` (new), `test_screen.py`, `test_generation.py`. Mutants
  (screen bracket rules, the cut, the floor, the denominator, the pool, the snapshot, the restart,
  the Pass 18 survivors; four plumbing mutants swap the hash, the pool denominator, the stratum and
  the audited expectations) each redden them; sources restored `cmp`-equal. Suite 1224 → 1285
  passed, 2 xfailed.

## 2026-11-04 — typed-decision-models U5.8a — generation runs, the screen, form obedience

**What:** plan4 §3.6 "Generation runs" and "Generated-text screen", §5 U5.8 (Screen); plan5 §3.2,
§3.4, §3.5, §4 U5.8 (Obedience, Screen); method note §Q1.1 "Obedience is checked per item" and
§Q1.2. The first of U5.8's three sequential sub-units: votes and majority (8b) and the pilot
tooling, cue check and `record-decision` (8c) are not built, nor is `render.py`'s join-in and
message-type renderers (U5.2b-render).

- `decisionmodels/screen.py`: `screen_hit(point, item)` (pure; `token:`, `id:`, `meta:` reasons)
  over the item's evidence strings (enums and seed-fixed `condition` text excluded), and
  `check_item`, which adds `seed_binding_errors` and returns every reason the item shows in
  plan5 §3.5's precedence (`screen`, `bindingMismatch`, `formMismatch`).
- `campaign.py`: `begin_round(genRound, count)` (a fresh RNG seed through `record_draw`; round 1
  draws ⌈1.1 × target⌉ seeds; shuffled, runs of 25; a `round-start` event), `queue_regeneration`
  (called by `next` when nothing is pending; pools hits across rounds), `Batch.gen_run_id`, and at
  ingest the `generated`, `screen-hit`, `binding-mismatch` and `form-mismatch` events. `fold`
  keeps accepted items, hits, third-hit drops (`State.generation_drops`) and the `template` halt.
  `status` prints the drops. No change to `ingest.py`'s checks (a schema failure still voids the
  run), `audit.py` or `runner.py`.
- Decided where the plans leave room: items and hits are ledger events, applied at the run's
  `outcome` (a crash between leaves them unfolded); the halt is derived in `fold`, so it survives a
  crash and fires once per (round, reason); a `resolve` clears the halt, not the pair; one `round-start` is one
  "round" for the 5% halt's denominator; `queue_batch` of a generate batch without `generation` is
  left unchecked, as older tests use it.
- Tests: `tests/test_screen.py` (47) and `tests/test_generation.py` (30). Mutants of both
  files' targets (22; the screen's both-counted and binding-first mutants, `>=` at 5%, regeneration
  hits counted, a pooled denominator, and plumbing mutants that swap `genRound`, `genRunId`, the
  hit's seed or the axes checked against) each redden them; sources restored `cmp`-equal. Suite
  1147 → 1224 passed, 2 xfailed (U5.2b-render's).

## 2026-11-01 — typed-decision-models U5.7 definition check: repeated key and wrong `name`

`install.frontmatter` refuses a repeated key (it kept the last, and Claude Code's choice is
unverified) and `definition_error` a `name` other than `dm-driver`/`dm-generator`/`dm-labeler`,
both before any write (plan6 v1.14 §3.3; review Pass 16 D1/D2). Tests plant labeler `maxTurns: 14`
then `2`, a repeated driver `model`, and three wrong names (`dm-labelerx` among them); the shipped
`agents/` still pass. Mutants (repeat check off, no raise, name check off, name prefix-only) each
redden `tests/test_install.py`; `install.py` restored `cmp`-equal. Suite 1141 → 1146 passed,
2 xfailed.

## 2026-11-01 — typed-decision-models U5.7 live smoke outcome; U5.5's `maxTurns` fixture swapped

**What:** the human-started smoke ran (`docs/test-reports/live-smoke-report.md`, against commit
f14aeefa, Claude Code 2.1.316). Figures below are that report's.

- **Batch 1** (50 `pt_br_item_at_cap` items, `maxTurns: 14`): ingested, no void; first spawn
  `completed` / `foreground`; the guard's `hook_success.command` equals `guardCommand` verbatim and
  `$CLAUDE_PROJECT_DIR` expanded (no `hookMissing`); `timeout: 30` accepted; the driver's `Agent`
  input held only `description`, `prompt` and `subagent_type`; no `SubagentHandback` call in
  either transcript, so whether it is hooked is unobserved. The subagent made 4 assistant messages
  (the page Reads in one parallel message), the driver 2. No Read errored, but the largest page's
  tokens could not be measured per Read (parallel Reads); the report estimates about 19,460 by
  proportion, against the 22,500 bound (finding F-1).
- **Batch 2** (`maxTurns: 2` copy of `dm-labeler`): three attempts, each void `maxTurns`
  run-attributable, then dropped as `retryCapExceeded` (50 items). The recorded shape: a header
  `Read`, then nine parallel page `Read`s, all guard-allowed; the subagent transcript ends on a user
  `tool_result` with no final reply, `Write` or `SubagentHandback`; the driver's `Agent` result
  is `completed`, `canContinueAgent: false`, with the note "this agent stopped at its 2-turn limit
  before finishing".
- `tests/test_audit.py`: the provisional `maxTurns` fixture (the clean run truncated after its
  `Write`) is replaced by `max_turns_run`, built in that shape, with the same assertions
  (counted, `run-attributable`, `maxTurns`, one finding) plus checks of the shape itself and a
  positive control (the same run ended by an `end_turn` reply is clean). `audit.py`'s docstring no
  longer calls the row provisional; the row's code is unchanged and does not read the driver's note.
- Mutants of the row (condition removed, condition inverted, class changed to environmental) each
  redden `tests/test_audit.py`; `audit.py` restored `cmp`-equal.

## 2026-11-01 — typed-decision-models U5.7 — agent definitions, guard and `driver install` (smoke not run)

**What:** `docs/plans/typed-decision-models6.md` v1.11 §3.2 "User settings", §3.3 and the U5.7 rows
of §4/§5, over plan4 §3.4, §3.5 and §5 U5.7. The code, tests and docs only: **the human-started
smoke has not run**, nothing here called the real `claude`, and no token was spent.

- `agents/`: `dm-driver.md` (`claude-sonnet-5-5`, `tools: Agent(dm-generator, dm-labeler)`,
  `maxTurns: 3`), `dm-generator.md` (`claude-opus-5-5`, effort `medium`, `Read, Write`,
  `maxTurns: 6`), `dm-labeler.md` (`claude-sonnet-5-5`, effort `high`, `Read, Write`,
  `maxTurns: 14`; both `omitClaudeMd: true`, `permissionMode: acceptEdits`); none carries `hooks`.
  `spawn-generator.txt`/`spawn-labeler.txt` (plan4 §3.4's text, no trailing newline),
  `spawn-driver.txt` (the driver message, slots `{agent_type}` and `{spawn_prompt}`),
  `settings.json` (`autoContinueAtUsageLimit: false`; `PreToolUse`, matcher `*`, one command
  handler `python3 "$CLAUDE_PROJECT_DIR/.claude/guard_hook.py"`, `timeout: 30`,
  `"onFailure": "block"`) and `guard_hook.py`.
- `agents/guard_hook.py`: stdlib only. Decisions print plan4's `hookSpecificOutput` JSON and exit 0;
  an unjudgeable input (missing or unknown `agent_type`, wrong event, missing `tool_name` or
  `tool_input`, bad JSON, any exception, a failed log append) exits 1 with no decision. `dm-driver`:
  `Agent` only, keys limited to `subagent_type`/`prompt`/`description`, the role of the single
  request in `exchange/req/`, and a prompt byte-equal to that role's installed template with the
  request path. Subagents: `Read` strictly inside `req/`, `Write` strictly inside `ans/` (relative
  paths joined to the input's `cwd`, then `os.path.realpath`), `SubagentHandback` allowed for them
  only. Every decision is appended to `store/hook-log.jsonl`.
- `decisionmodels/install.py`, `driver install --root <dir> [--agents-dir <dir>]`: under
  `Campaign.locked()`, `runner.forbidden_settings_refusal` on `~/.claude/settings.json` (reused),
  `runner.user_settings_digests` (reused) recorded as `userSettingsDigests`, the recorded key
  names printed (never a value); sources byte-copied to `exchange/.claude/` and each installed file
  re-hashed against its source; refusal of an edited installed file, a stray file in
  `.claude/agents/`, a `settings.local.json`, a symlinked file, a source that no longer matches the
  first install's `promptSha256`, a definition with `hooks`, a settings wiring that is not one
  `PreToolUse` `*` guard with `onFailure: "block"`, or one without `autoContinueAtUsageLimit: false`.
  A fresh `expectations` event on every run: `guardCommand` (`install.guard_command`), per role the
  pins read from the installed definition, its spawn template and `instructions: []`,
  `driverTemplate`, `userSettingsDigests`, `settingsSha256` and `promptSha256 {agentDefinition,
  spawnTemplate (with `driver`), guardHook}`.
  `Campaign.record_expectations` gained `prompt_sha256` and `settings_sha256`.
- `decisionmodels/smoke.py` and `driver smoke-fixture --out <items.jsonl> [--count 50]`: the P5-F5
  fixture, 50 guard `understanding` items each rendering exactly `render.MAX_CHARS`.
- `README.md` (the `install` section, the guard, the smoke), `AGENTS.md` (current-state line, the
  `agents/` invariant).

**Calls made here:**
- The guard command uses `$CLAUDE_PROJECT_DIR` so the source `settings.json` can be byte-copied
  and hash-checked; the guard finds `exchange/` and `store/` from its own installed path. Whether
  Claude Code expands it, and writes the unexpanded string as `hook_success.command`, is unverified.
- The guard's `Agent` check is an allowlist of keys, stricter than plan6's four named keys; if the
  harness adds a key by default the smoke shows a `guardRejected` halt.
- `SubagentHandback` is allowed for the subagents unconditionally (plan6's condition is unknown
  until the smoke); a deny on it would halt every run as `guardRejected`.
- A failed `hook-log.jsonl` append fails closed (exit 1), per "any exception".
- `install` records `settingsSha256` beside, not inside, `promptSha256`, to keep plan4 §3.7's key set.
- `--agents-dir` is how the smoke's `maxTurns: 2` copy installs; its hashes name the copy.

**Review Pass 15 fixes and plan6 v1.13 (§3.3 "Definition check", "Version floor"):**
- P15-m1: `fold` lets only the first `expectations` event set the model pins and declared
  `instructions` (later events refresh the rest), so re-running `install` keeps a `resolve` repin;
  the plan6 §3.2 rule is the architect's.
- `runner.MIN_CLAUDE_VERSION = (2, 1, 295)` and `claude_version_refusal(environ)`: `claude
  --version` under the child's allowlisted environment, first `x.y.z`, integer-tuple compare;
  below the floor, a non-zero exit, no parsable version or no binary refuses. `install` calls it
  before any write; `Runner._loop` calls it after the pause check and before the settings check
  (a refusal goes through `store/refusal`, exit 1, nothing appended). Floor only. The
  missing-binary case in `tests/test_runner.py` is now a refusal, and the `OSError`-at-launch path
  is tested by planting the failure in `Popen`. The test fake `claude` answers `--version`
  (`tests/conftest.py`'s `fake_claude` for the install and guard files).
- `install.definition_error` (§3.3's key sets and pinned values, `hooks` by name) runs before any
  copy; it subsumes P15-n1's missing `model`/`effort` traceback. P15-n2: a guard test with
  differing role templates, built in a tmp dir.
- Suite 1085 → 1138 passed, 2 xfailed. Mutants, all red but for one that got a test and then went
  red: six for P15 (fold pins, fold instructions, swapped `TEMPLATE_FILES`, and the pin check, now
  the definition check); 24 for the floor and definition check (floor ±1, string compare, no-parse,
  non-zero exit, missing binary, loop order against pause and settings, call removed, install
  order, child environment, and each validated key).

**The smoke** (a person, once, `driver run`, never the timer; `README.md` has the commands). It runs
one Sonnet driver session and one Sonnet labelling run over 50 `smoke-fixture` items, then a second
smoke campaign installed from a copy whose `dm-labeler.md` has `maxTurns: 2`. It records here:
the first spawn's foreground shape (F2); no void and every page Read succeeding, with the largest
page's tokens (transcript usage deltas around each Read) at most 22,500, else the 40,000-character
constant goes to the architect; the `maxTurns: 2` cut-short shape and terminal entry (U5.5 then
swaps its provisional fixture); whether `SubagentHandback` appears and is hooked; that the guard
`hook_success.command` equals `guardCommand` verbatim, that the guard allowed every call with its
own entry, and that the injected-context rows pass; any `Agent` input key beyond the three, and
any safeguards or usage-limit shape met.

**Verified:** from `decision-models/`, `.venv/bin/pytest -q -p no:cacheprovider` 900 passed, 2 xfailed
before; 1085 passed, 2 xfailed after (the delta is the two new files, `tests/test_install.py` and
`tests/test_guard_hook.py`, 185 tests); `ruff check .` clean. Forty-six mutants of `guard_hook.py`,
`install.py` and `settings.json` each redden those two files except one, which is equivalent
(dropping the guard's own `subagent_type` check leaves the role cross-check denying the same calls);
files restored `cmp`-equal. Four first survived (the driver's non-`Agent` tools, a tool use with no
object input, the guard-file check on the settings command, and the above) and three got tests.

## 2026-11-01 — typed-decision-models U6.3b — the user-settings record

**What:** `docs/plans/typed-decision-models6.md` v1.10 §3.2 "User settings" and the U6.3b rows of
§4/§5, replacing U6.3's two-key hash.

- `decisionmodels/runner.py`: `FORBIDDEN_SETTINGS` (`env`, `hooks`, `fallbackModel`,
  `apiKeyHelper`; `disableAllHooks` and `autoContinueAtUsageLimit` when `true`) and
  `UNRECORDED_SETTINGS` (§3.2's closed list) replace `REFUSED_SETTINGS`/`HASHED_SETTINGS`;
  `user_settings_digests` (sha256 of each recorded top-level key's canonical JSON, `permissions`
  without `allow`/`defaultMode`) replaces `user_settings_sha256`; `forbidden_settings_refusal` is
  the check `install` will share; `settings_refusal` runs it first, then names the keys added,
  removed and changed (no values) and ends with the recovery sentence.
  `permissions` is omitted from the digests when nothing is left after stripping `allow` and
  `defaultMode` (absent == empty), so a first allow rule, a lone `defaultMode` or the removal of
  the last allow rule never refuses (review Pass 14, P14-m1).
- A call made: the forbidden-key refusal ends "remove it", not the recovery sentence, because
  `install` refuses to record a forbidden key, so "run `driver install` to accept it" would be
  false there; the recovery sentence ends only the added/removed/changed refusal (P14-n1).
- `decisionmodels/campaign.py`: `record_expectations(user_settings_digests=…)` writes
  `userSettingsDigests`, which `run` requires. No campaign had an `expectations` event, so
  nothing migrates.
- `tests/test_runner.py`: the `Bench` helper records digests; the hash tests are replaced by §5's
  cases (forbidden per key, also when recorded; changed per class; removed key; no value shown;
  not-refused edits; reorder/re-indent; canonical form; pinned set).
- `README.md`'s `run` section and the module docstring.

**Verified:** `pytest -q -p no:cacheprovider` from `decision-models/` 894 passed, 2 xfailed;
`ruff check .` clean. Twelve mutants (§5's list, plus a digest over the wrong key's value) each
redden `tests/test_runner.py`; the file was restored `cmp`-equal.

## 2026-11-01 — typed-decision-models U6.3 — `driver run`, the unattended runner

**What:** `docs/plans/typed-decision-models6.md` v1.8 §3.1, §3.2 and the §4/§5 U6.3 rows, over
U5.6's library (`campaign.py`) and the contracts impl review Passes 10–11 left for the runner
(`docs/reviews/typed-decision-models4-impl.md`).

- `decisionmodels/runner.py`: the loop (one `Campaign` per process; `next` → `issue` → launch →
  wait → `.exit` → `ingest`), exiting 0 on `next`'s stop, on `IngestResult.stop` and after a
  `launchFailed` outcome. The child is §3.2's command verbatim (no `--fallback-model`, no
  `--setting-sources`), in `exchange/`, in its own session and process group, stdin
  `/dev/null`, stdout/stderr in `store/runs/<sid>.json`/`.err`, environment = the allowlist plus
  `CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1`, and the lock's descriptor via `pass_fds`. On
  `batchTimeout` the group gets SIGTERM, then, after 30 s, SIGKILL if any member of the group
  (not only the leader) is left; the wait ends as soon as the group is empty. `.exit` is written
  atomically after `wait()` and before `ingest`. `store/pause` is checked first in each
  iteration, so a paused fire exits 0 whatever the settings hold. User-settings refusals
  (`env`/`hooks` key, `autoContinueAtUsageLimit: true`, a `permissions`/`modelSettings` hash
  other than `install`'s) are checked before every `next` and append nothing; a refusal after
  the lock writes `store/refusal` (ISO 8601 timestamp, reason) for `status`, and a successful
  launch deletes it. The root and no-campaign refusals precede the lock and write nothing;
  `status` prints a root refusal it computes live (`campaign.py`). A signal before a launch
  issues nothing (`stopped: signalled`).
  `store/driver.lock` is held with `LOCK_NB` (held → one line, exit 0); `store/runner.json`
  records `{runnerPid, runnerStartTime}` (field 22 of `/proc/<pid>/stat`) before any launch.
  SIGTERM (`pause --now`) and SIGINT kill a running child group, append `interrupted {batch}`
  once (`Campaign.record_interrupted`, U5.6's API) and exit without ingesting. `pause [--now]`
  and `resume` touch only `store/pause` (and signal the runner only if its PID is alive with the
  recorded start time). The root refusal is `root_refusal`'s, as at `init`.
- `decisionmodels/campaign.py`: `record_expectations` takes optional `driver_template` and
  `user_settings_sha256`, recorded as `driverTemplate`/`userSettingsSha256` in the
  `expectations` event; `run` refuses without them. `status` adds a `REFUSED:` line for a root
  refusal. `decisionmodels/driver.py`: `run`, `pause
  [--now]`, `resume`.
- **Calls made here:** `run` stops after a `launchFailed` outcome (§3.2 "Scheduling": a fire into
  a spent window or a broken environment costs one launch, never a hot loop until
  `launchFailBound`); settings are checked before `next`, so a refusal writes nothing at all;
  refusals exit 1, every stop exits 0; the user-settings hash is sha256 over the canonical JSON
  (sorted keys, no whitespace) of whichever of `permissions`/`modelSettings` is present, and a
  missing settings file is `{}`; a launch that cannot start (`OSError`) writes its error to
  `.err`, no `.exit`, and ingests as `launchFailed`; SIGINT is handled as `pause --now` without
  the pause file; `runner.json` is left in place (the start time guards a recycled PID);
  `pause --now` does not wait for the runner to exit.
- `tests/test_runner.py` (38 functions, 45 tests) with a fake `claude` on a `PATH` of its own and
  a fake `HOME`; nothing calls the real CLI. `README.md` (`run`, `pause`, `resume`, refusals in
  `status`), `AGENTS.md`.

**Verified:** `pytest -q` from `decision-models/`: 854 passed, 2 xfailed (819/2 before);
`ruff check .` clean. 33 mutants of `runner.py`, run in a private copy with a fresh
`PYTHONPYCACHEPREFIX`, each restored `cmp`-equal: 32 redden (allowlist leaks
`ANTHROPIC_API_KEY` 1; `--fallback-model` 1; child-only kill 2; non-atomic `.exit` 1; `.exit`
after ingest 3; fresh `Campaign` per iteration 1; `interrupted` twice 1; lock not passed 2; no
SIGKILL 1; no timeout 1; shared process group 3; blocking lock 2; each settings refusal 1–6;
start-time checks 1–2; no stop after `launchFailed` 2; stdin inherited 1; settings checked once
1; …). The one survivor, dropping the exit on `IngestResult.stop`, is equivalent: the same
instance's `next` stops on the same halt or usage stop.

**Pass 12 fix round** (`docs/reviews/typed-decision-models4-impl.md` P12-m1–m3, P12-n1–n3): the
group SIGKILL above, the pause before the settings check, `status`'s root refusal, a signal before
a launch issuing nothing, `pause --now` tolerating a runner that exits after the start-time check,
and the halt reason on the ingest path; the timeout test's fake ignores SIGTERM first, with a 3 s
`batchTimeout`. Ten new tests (six red against the pre-fix code). `pytest -q`: 864 passed, 2
xfailed; `ruff check .` clean; `tests/test_runner.py` 4/4 green under 64 busy loops on 8 CPUs.
Mutants in a private copy, each restored `cmp`-equal: the review's R16, R35, R28, R20, R10, R11,
R6 and the coder's M25 now redden, and so do 8 of the round's own; R29 survives, equivalent under
the fix (the new signalled check after `next` stops the launch it guarded).

## 2026-11-01 — typed-decision-models U6.8 — dataset contract vector copy re-made

**What:** `docs/plans/typed-decision-models6.md` v1.8 §4's U6.8 row:
`tests/fixtures/decision-contract/dataset-shape.jsonl` re-copied byte for byte from model-bench's
canonical file, which gained each line's second, incomplete `runs[]` entry (§3.7's type exemplar;
model-bench `docs/HISTORY.md` has the detail). `cmp`-equal; the guarded-equality test in
`tests/test_decision_contract.py` is green. Suite 819 passed, 2 xfailed (unchanged); ruff clean.

## 2026-11-01 — typed-decision-models U5.6 — campaign ledger and driver CLI

**What:** `docs/plans/typed-decision-models6.md` (v1.8) §4's U5.6 row and §5 U5.6 row over plan4
`typed-decision-models4.md` §3.3 (caps, halts, idempotence), §3.5 and §5 U5.6, and plan5
`typed-decision-models5.md` §3.2, §3.6's spend gate and §4 U5.6; the carried rulings in
`docs/plans/typed-decision-models-coordination.md` (P5-F4 commit point and cases (a)–(d), P6-F1
case (e), Pass 8 q3, retired names) and impl review Pass 9's caller items; then a fix round for
Pass 10's findings (`docs/reviews/typed-decision-models4-impl.md`).

- `decisionmodels/campaign.py`: the append-only, fsync'd ledger and its fold (`pending`, `issued`,
  `ingested`, counted `void` up to `retryCapExceeded` at 3, `reissued`, `usage-stop`, `halted`);
  `Campaign.next`/`prepare`/`issue`/`ingest` as library entry points for U6.3's `run` (`next`
  stops on `store/pause` before reading anything, on a halt, on a usage stop, and when nothing is
  pending); `issued` carries `sessionId`, and reconciliation reads only that session's
  transcript; `interrupted {batch}` keeps an attempt out of `earlier_no_attempt`;
  `launchFailed`/`launchFailing` outcomes carry stdout, stderr and the exit code, with the clock
  read from `issued` times and `init`'s `launchFailBound`; `Expected` uses absolute paths.
  `record_draw` and `fresh_rng_seed` (plan5 §3.2); `init --phase full` holds the pilot's RNG seeds
  and applies the spend gate; `resolve` retries, `--skip-batch` (`dropped.safeguardSkipped`),
  `--repin-instructions`, `--repin-model` (`generation-restart`/`labelling-restart`).
- **Ingest:** the `outcome` event is appended before parts move (copy all, fsync, unlink). A
  re-ingest checks the ledger first, never calls `audit_run`/`combine`, re-reads through
  `ingest_answer_files` (from `store/` when it holds every recorded part) and appends nothing.
  Plan6's headless sequence wins over the ledger's "`ingest` runs only after the run hands back":
  `run` ingests once the child exits or is killed, and an unfinished run is `incomplete`.
- `decisionmodels/driver.py`: `init`, `resolve`, `status` (`python -m decisionmodels.driver`).
- **Calls made here:** an unobserved child exit counts as non-zero; the no-attempt cap and the
  launch clock restart after a `resolve`; `--skip-batch` is refused for a halt other than
  `safeguardStop`; the pilot report is read for `stakeholderDecision.{outcome, cap,
  stakeholderRef}`, `missingCueCheck.state` and `rngSeeds`.
- **Pass 10 fix round:**
  - P10-M1: `issued.attempt` (plan6 v1.8 §3.7's issue ordinal, skipping numbers),
    `requestSha256`, `answerSha256`, `runId` (located runs only), `role`, `details`, `resolved`'s
    fields and the `issued.keys` binding (label item and generation seed) are pinned by value,
    each expected value computed in the test.
  - P10-M2: `init --phase full` refuses a report whose `stakeholderDecision.cap` is not 25 or 50,
    whose `rngSeeds` is not a non-empty list of ints, whose `stakeholderRef` (when it is needed)
    is not a non-empty string, or that is not a JSON object; no key has a default.
  - P10-m1 (option b): a `Campaign` instance remembers a usage stop its own `ingest` recorded and
    its `next` returns `Stop("usage-stop")`, so plan6 §3.2's loop launches once per spent window.
    The memory is the instance's, not the ledger's: the next run is a new instance and
    serves again, which is plan6 §3.2's "a fire within a spent window costs at most one short
    driver turn".
  - P10-m2: `store/runs/<sessionId>.exit` is ASCII `-?[0-9]+` with optional whitespace; an empty
    or other file is an exit never observed (`launch.exitCode: null`), never an exception.
  - P10-m3: the root is `realpath`, so a symlink neither hides a work tree from `root_refusal` nor
    names the wrong transcript folder.
  - P10-n1: `countsAttempt` is dropped from the `outcome` event: `combine` sets it exactly when
    the outcome is `void`, the one fact `fold` counts, so it could only drift.
  - P10-n2: a torn last ledger line (no newline) or a line that is not an event refuses every
    command naming its byte offset, raised from `events()`, and `_append` refuses to write after
    one. The ledger is split on `\n` only (`str.splitlines` also split on a raw U+2028).
- `tests/test_campaign.py` (139 tests) and `tests/test_driver.py` (9); `README.md`, `AGENTS.md`.

**Verified:** `pytest -q` from `decision-models/`, no deselection: 819 passed, 2 xfailed (761/2
before the fix round, 671/2 before the unit); `ruff check .` clean. Mutants ran in a private copy
(the model-bench byte-copy tests skip there), each file restored `cmp`-equal. The unit's 18:
outcome event appended after the move 6; re-ingest calling `audit_run`+`combine` 8; re-ingest
taken as a first ingest 12; a retired name's late answer accepted 3; a held RNG seed accepted 2;
`interrupted` counted in `earlier_no_attempt` 1; `pause` ignored by `next` 1; a void re-ingested
with `None` 1; `fresh_rng_seed` not redrawing 1; a held `seedId` accepted 1; the 6 h default for
the bound recorded at `init` 1; a relative root 1; the spend gate without `stakeholderRef` 1;
`--skip-batch` on any halt 1; parts moved one by one 2; re-ingest reading `exchange/` only 2; a
halt ignored by `next` 1; any session's transcript read 3. The fix round's 32, every one red:
`attempt` → 1 1, or the counted number 1; `requestSha256` reversed 1, or wrong at `run_record` 1;
keys bound to reversed refs 2; role `label` 1; `runId` constant 1, or over every outcome 1;
`run_record` attempt 1 1; the report refusal dropped 15; the cap check 8; the `rngSeeds` check
4; `stakeholderRef` truthy 3; the usage-stop flag never set 2, or ignored by `next` 2, or set by a
re-ingest 1; bare `int()` on `.exit` 7, or a lax parse 2; `abspath` 1; the launch clock ignoring
`resolve` 1; `issue`'s stale 1 and duplicate 1 refusals; `launchFailing` without evidence 2;
non-ASCII kept in the slug 1; status "last issued" 1; `details` empty 1; a torn line dropped 5,
or appended after 1; `str.splitlines` 6; `resolved.batch` dropped 1; `answerSha256` reversed 8; a
non-event line accepted 1.

## 2026-11-01 — typed-decision-models U6.6 — dataset contract vector copy

**What:** `docs/plans/typed-decision-models6.md` v1.5 §3.7 and §4's U6.6 row:
`tests/fixtures/decision-contract/dataset-shape.jsonl` is a byte-copy of model-bench's canonical
dataset contract vector (`generation.json`, items and `splits.json` per role), which U5.10 checks
`finalize`'s output against. `tests/test_decision_contract.py` adds it to the guarded equality
test and the row-count test, and checks each of its items against this package's points
(`input_field_errors`, `expected` in the answer space, `labelVotes` keys the answer space as JSON
strings).

**Verified:** `pytest -q` from `decision-models/`: 671 passed, 2 xfailed (baseline 666/2); `ruff
check .` clean. Deleting `rngSeed` from the canonical file without re-copying reddens the equality
test (1 failed); re-copying turns it green; both files restored `cmp`-equal.

## 2026-11-01 — typed-decision-models U5.5 — transcript audit

**What:** `docs/plans/typed-decision-models6.md` v1.5 §3.3 (guard bounds), §3.4 (audit rows,
precedence, in-flight rule) and §5 U5.5, over `docs/plans/typed-decision-models4.md` §3.3's location
rule and remaining rows and §3.7's `runs[]` record and token rule; plus impl review Pass 8's
P8-F3–F6 and Pass 9's P9-F1–F4 (fix round U5.5b) (`docs/reviews/typed-decision-models4-impl.md`).
P8-F1/F2 are settled by plan6 (foreground only; one final `SubagentHandback` is a reply).

- `decisionmodels/audit.py`: `audit_run(driver_transcript, Expected, *, earlier_no_attempt, launch,
  interrupted=False)` locates the run through the driver's `Agent` `tool_use` and `.meta.json`
  `toolUseId` and returns an `Audit`: every matching row as a `Finding` in table order, the chosen
  class, and `RunFacts`. `Expected.guard_command` is the declared guard command (P8-F4). `Launch` is
  keyword-only and `fail_bound` (the `launchFailBound` recorded at `init`) has no default, nor has
  `launch` (P9-F4).
- **Rows shipped:** `safeguardStop`, `backgroundSpawn`, `guardRejected` (a guard deny or a blocked
  guard failure, keyed on `permissionDecision` `reject`/`hook`), `guardBreach` (an allowed call
  outside the guard's bounds, recomputed with the audit's own `realpath` canonicalization),
  `inputInvalid` (counted: no decision, `is_error`, no attachment; a hand-back too, P9-F3),
  `hookMissing` (any other decision, incl. D3's `accept`/`config`), `driverModelMismatch`,
  `syntheticError` (either transcript), `rateLimit`, `spawnShapeUnknown` (a non-`foreground` meta
  even in flight, P9-F1), `incomplete`, `neverSpawned`, `launchFailed` (uncapped) and
  `launchFailing` (past `Launch.fail_bound`), then plan4's location, driver, injected-context and
  counted rows (`maxTurns`, `toolOutOfBounds` for hand-back misuse, `pathOutOfBounds`). `hookDenied`
  is removed.
- **Precedence:** halts, usage stop, `spawnShapeUnknown`, re-issues, counted; table order within a
  tier, where the halt rows' order picks the reason (P9-F2). In-flight tool uses (no result while
  the spawn has none) are read by no row, so a killed run is `incomplete`. An `interrupted`
  incomplete is not capped. A final line with no newline that does not parse is truncation (P8-F5).
- **Fixtures** (`tests/fixtures/transcripts/`): the clean run moved to H1's foreground shape (driver
  guard entry, `completed` result, `requestShape`, a Sonnet driver); `guard-decisions.jsonl` holds
  the U6.7 D1/D2/D3 `tool_result` entries and attempt 1's validation error in their recorded key
  sets (probe report §10.3, §7), ids and paths replaced. `tests/test_audit.py`: 174 tests.
  `README.md` line 45 reflowed (P8-F6); `AGENTS.md`.

**Verified:** `pytest -q` from `decision-models/`, no deselection: 666 passed, 2 xfailed (baseline
568/2); `ruff check .` clean. Mutants ran in a private copy, each restored `cmp`-equal;
failures per mutant: `guardRejected` counted 4; deny counted and failure halted, told apart by text
(the rejected alternative) 3; `hookMissing` reading in-flight tool uses 20; in-flight ignoring the
driver's result 1; `async_launched` as completion 1; `finished` not required 2; `spawnShapeUnknown`
first 1; counted tied with the usage stop (A6) 2; counted tied with halt (A1) 4; table order only
(A2) 5; re-issue tied with the usage stop 1; `safeguardStop` counted 6; refusal marker ignored 4;
`refusedMessageId` ignored 2; `guardBreach` counted 23; `normpath` for `realpath` 3; prefix without
separator 6; driver bound unchecked 1; result matched ignoring `tool_use_id` (P1) 2; any command as
the guard 1; any decision as the guard's 1; any hook event 2; `inputInvalid` despite an attachment
1; hand-back not exempt 2; hand-back count unchecked 2; hand-back not a terminal reply 4; driver
model unchecked 2; driver `<synthetic>` ignored 2; `launchFailed` capped 1; launch clock never
resets 1; launch bound `>=` (M9) 1; empty driver transcript as missing (M17) 1; in-flight meta
unchecked 2; first four halt rows reversed 2; hand-back exempt before `inputInvalid` 1; non-zero
exit as never spawned 1; `interrupted` ignored 1; no truncation tolerance 2; any
unparseable final line as truncation 1; empty transcript skipped 1. Accepting `async_launched` in
the shape check survives (0): equivalent, as the background row is tested first.

## 2026-11-01 — typed-decision-models U5.2b-hash — leakage key for join-in and message type

**What:** `docs/plans/typed-decision-models5.md` §3.4's U5.2b-hash (the hashing half of
`docs/plans/typed-decision-models4.md` §4.1's U5.2b; formula v1.3 §3.5).

- `decisionmodels/hashing.py`: `INPUT_FIELDS_BY_ROLE` gains `join-in-judge` (`agentName`,
  `priorTurns`, `newMessage`) and `message-type-classifier` (`text`, `priorTurn`), equal to
  model-bench's `decisions.INPUT_FIELDS_BY_ROLE` at `1554d560`. An opening join-in's
  `priorTurns: []` and a `priorTurn: null` hash as present fields.
- `tests/fixtures/decision-contract/input-hash.jsonl`: re-copied from model-bench (6 new join-in
  and message-type rows), `cmp`-equal; every row recomputes through `input_hash`.
- `tests/test_decision_contract.py`: `test_input_fields_by_role_equals_model_benchs` reads
  model-bench's row literal by AST (never imported; skipped when `model-bench/` is absent) and
  compares `{role: sorted(fields)}`: order-free, since `canonical_input_json` sorts keys, but
  duplicate-sensitive. The reader requires exactly one top-level binding of the name (assignment
  or import), pinned by five planted second bindings. The unknown-role refusal now uses a role no
  table defines.
- `tests/test_points.py`: every point's `inputFields` equals `hashing`'s row as a set. P3-F6's
  two strict xfails (which XPASSed once the rows landed), `PLANNED_INPUT_FIELDS` and its `.get`
  fallback are deleted. `AGENTS.md`, `README.md`.

**Verified:** `pytest -q` from `decision-models/`, no deselection: 492 passed, 2 xfailed (the
U5.2b-render renderer xfails); `ruff check .` clean; `tests/test_fr8_isolation.py` 24 passed.
Mutants ran in a private dir, each file restored `cmp`-equal and sha256-checked afterwards. A
join-in field dropped reddened 5; `priorTurn` moved to join-in 6; an extra join-in field 2; a
duplicated join-in field 1; one copied digest edited 2; a null `priorTurn` omitted like an absent
field 1; the reader taking the first binding 5, ignoring import bindings 2. A hash-neutral
reorder of the join-in row stays green (red, 1, if the model-bench comparison reverts to exact
order). Comparing as sets misses the duplicate (0), as does the point conformance test, which is
set-based by plan5 §3.4; as lists, the point test falsely reddens on a reorder. Review:
`docs/reviews/typed-decision-models4-impl.md` Pass 7 (P7-F1–F3 applied).

## 2026-11-01 — typed-decision-models U5.4 — exchange and ingester

**What:** `docs/plans/typed-decision-models4.md` §4.1's U5.4 — §3.2 in full, §3.3's answer rows,
idempotence and second-answer refusal — with plan5 (`typed-decision-models5.md`) §3.4's id-blind
seed lines, under note `docs/plans/typed-decision-models-ml.md` §Q1.2 "Blind"; review
`docs/reviews/typed-decision-models4-impl.md` Pass 5.

- `decisionmodels/exchange.py`: `build_label_request` / `build_generation_request` (header,
  line 2, then one item or seed per line, at least one; keys of 10 base32 characters from
  `secrets`, redrawn on a collision with the caller-held set or this request's keys; a fresh
  opaque name per request), `write_request` (validates, never overwrites), `request_errors` /
  `validate_request` / `parse_request` (every line a closed key set, a repeated JSON name refused;
  any `*Id` key on a generation request refused, whatever `points/` declares), `body_sha256`
  (layout-free) and `cut_pages` (greedy pages ≤ 40,000 written characters, newline included, cut
  at `P_MAX` 10 / 2 pages with the header page; the rest is `overflow`; a page's limit is at least
  1; a line that cannot be parsed — bad JSON, too deep, an integer past Python's digit limit —
  is a refusal, never an exception). Seed lines are `Seed.axes` with the `*Id` keys projected out; `BuiltRequest.keys` keeps the
  whole seed. Raw U+0085/U+2028/U+2029 are written escaped and refused raw, so no reader splits a
  line. `RENDERERS` holds the guard renderer only.
- `decisionmodels/ingest.py`: `read_answer_parts` (whole file or parts 1…k, no gap),
  `ingest_answer` / `ingest_answer_files` → `Ingested`, `AlreadyIngested` (same answer bytes),
  `Void` (run-attributable, one reason by a fixed order plus every detail) or `AnswerRefused`
  (a second, different answer). A line with a non-string key, a repeated name or nesting too deep
  to parse is a `malformed` void, never an exception, so reconciling the batch again cannot crash
  (P5-F1). The answer space is type-strict (`1` is not `true`); generated items are checked by
  `points.input_field_errors` only (plan5 §3.5: screen and bindings are U5.8's). A CRLF line ending
  reads as LF, so a `\r`-only line is a blank line, `malformed`, as in an LF answer (P6-F2).
- `tests/test_exchange.py`, `tests/test_ingest.py`: the §5 U5.4 rows — the plan4 positive
  controls (`split` on the header, `seedId` on the rubric line, `intent` on an item line), plan5's
  (`domainId` on a generation line; no `*Id` key in a writer's seed line, all three points), the
  budget on 50 pt-BR guard items at the render cap, an independent `bodySha256` recomputation and
  a layout change, a two-part answer, each void reason, and idempotence against a stand-in ledger.
  Scripted-`secrets` tests pin held-key avoidance and uniqueness within a request; a part with no
  final newline hashes as the whole answer; a CRLF answer ingests. Strict xfails for the two
  renderers U5.2b-render adds. `AGENTS.md`, `README.md`.
- **Observed paging:** the guard rubric line is 5,099 characters and a pt-BR item at the render cap
  6,331 written characters, so 50 such items take exactly 10 pages (the first content page holds
  the rubric and 5 items); a 70-item batch is cut at 53.

**Verified:** `pytest -q` from `decision-models/`, with the U5.2b-hash node
`test_the_vector_copy_is_byte_identical_to_model_benchs[input-hash.jsonl]` deselected: 480 passed,
1 deselected, 4 xfailed. `ruff check .` clean; `tests/test_fr8_isolation.py` 24 passed. A
sentinel and 36 mutants ran in a private copy over the two U5.4 test files, each restored
`cmp`-equal, with both mutated files sha256-equal to the live tree afterwards. The sentinel
reddened 13 tests, and every mutant reddened at least one: `*Id` projection removed 13;
`validate_request` accepting `split` 2, `seedId` 4, `intent` 1, `domainId` 1; `*Id` rule removed
5; idempotence broken 3; second-answer refusal removed 2; page budget off by one 1–2 in each of
five forms; non-string key check removed 6; repeated-name hook removed 3; held keys ignored 1;
this request's keys ignored 1; no-final-newline hashing removed 1. After review Pass 6, in a
second private copy: the `\r` strip removed 1; the request-side `ValueError` catch narrowed to
`JSONDecodeError` 1; the request-side `RecursionError` catch disabled 1. The request-side
`offset < 1` check was deleted: the coverage check already refuses any such page.

## 2026-11-01 — typed-decision-models U5.3 — points, rubrics, seeds

**What:** `docs/plans/typed-decision-models4.md` §4.1's U5.3, with plan5
(`typed-decision-models5.md`) §3.4–§3.5, under note `docs/plans/typed-decision-models-ml.md`
v1.11 §Q1.1–§Q1.5; reviews `docs/reviews/typed-decision-models4-impl.md` Passes 3–4 and
`docs/reviews/typed-decision-models-ml.md` Passes 1–2.

- `points/<role>/` for `guard-judge`, `message-type-classifier`, `join-in-judge`: `point.json`
  (answer kind, key and `expected` per v1.3 §3.2 and FR-10; input-field schema with v1.3 §3.5 /
  plan2 §3.1 field names; seed bindings; seed design and weights; note §Q1.3 cells and sizes;
  screen token lists), `rubric.md` (each answer, then boundary cases) and `generator.md` (no
  commentary, no `*Id` field). `points/common/` holds the two answer-file sections.
- Note v1.10–v1.11 design: guard `missingForm` drawn given (answer, difficulty) on
  `understanding` seeds, with per-item obedience as an `emptinessAxis` binding; join-in `cue`
  drawn given the answer, cue_05 narrowed; message-type rulings for off-topic questions,
  cancelling and returns; join-in small-talk ruling; the assistant speaks as `Assistant` /
  `<business> Assistant`; the guard generator adapts the situation, never the condition or the
  business. Generator text per ML Pass 2: the `ambiguous` rule ends "`missing` is `[]`"; a guard
  `understanding` near-miss follows `missingForm`; a message-type follow-up may stand alone.
- `decisionmodels/seeds.py`: `draw_seeds` — hierarchical stratified axes, balanced axes, and
  conditional axes over weighted multi-axis cases, every allocation by unbiased systematic
  rounding with an integer start (docstring); non-word ids; the record carries `pythonVersion`
  (reproduction holds on the same minor version).
- `decisionmodels/points.py`: `load_point`, the composed line-2 texts, `input_field_errors`,
  `seed_binding_errors` returning `BindingError(binding, message)` for U5.8's classification. The
  `missingForm` binding is silent when `understanding` is not a non-empty mapping, so a wrong-kind
  item reports only its `evidenceKind` errors (P4-F1).
- `tests/test_seeds.py`, `tests/test_points.py`: `allocate`'s expectation exact over evenly
  spaced starts, and constant within each integer start's interval; draw means over 1,500 RNG
  seeds at sizes 1–7 within 4 standard errors; restricted round-trip through the record;
  collision and conditional-overlap guards; the wrong-kind item; a text pin tying each form's
  "`missing` is `[]`" to its binding; strict xfails for the two roles `hashing.py` lacks until
  U5.2b (P3-F6); side-check gate fails rather than skips when the pack moves (P3-F5).
  `AGENTS.md`, `README.md`.
- **Digest pins moved**, deliberately: all three re-pinned (guard `341d8815…`, message type
  `299beb51…`, join-in `26109599…`). Message type's `point.json` is unchanged, but systematic
  rounding changes every allocation, so its pin moved too. The integer start left all three
  unchanged: it gave identical shares on 100,000 random (weights, total, RNG seed) cases.

**Verified:** `pytest -q` from `decision-models/` 364 passed, 2 xfailed, and 1 failed:
`test_the_vector_copy_is_byte_identical_to_model_benchs[input-hash.jsonl]`, because
model-bench's canonical `input-hash.jsonl` gained six rows in its working tree (U4 in flight);
this unit changed neither the copy nor the test. Deselected by that node id: 364 passed,
2 xfailed. `ruff check .` clean. Eleven mutants in a private copy, each restored `cmp`-equal and
sha256-checked: largest remainder 84 failed; start ×0.97 21, ×0.95 35, ×0.999 9, ×1.001 86;
`missingForm` guard removed 6; overlap check removed 1; the `ambiguous` rule's "`missing` is
`[]`" removed 1; undefined-form error removed 1; the guard over-silencing 7; the pre-integer
float start restored 0 (equivalent, as intended).

## 2026-10-31 — typed-decision-models U5.2 — guard contract copies

**What:** `docs/plans/typed-decision-models4.md` §4.1's U5.2, under v1.3 §3.2, §3.2.2 and §3.5.

- `decisionmodels/render.py`: `render_guard_input(condition, understanding, turns)` — the guard
  judge user message (CONDITION, then CURRENT STATE iff `understanding` is non-empty, then RECENT
  TURNS; 6,000-char cap dropping oldest turns first, base hard-truncated when no turn fits);
  `normalize_turns` (filter-then-last-6, 400-char text, `displayName`/`authorId`/`member`
  speaker); `render_guard_item(item)` composing the two (v1.3 §3.2's `normalized_turns`).
- `decisionmodels/hashing.py`: `INPUT_FIELDS_BY_ROLE` (`guard-judge` only; the other roles come
  with U5.2b), `canonical_input_json`, `input_hash` (v1.3 §3.5's formula; an unknown role is a
  `ValueError`).
- `tests/fixtures/decision-contract/render-guard.jsonl` (14 rows) and `input-hash.jsonl` (4 rows):
  byte-copies of model-bench's canonical files, `cmp`-equal.
- `tests/test_decision_contract.py`: every vector row against the implementation; the guarded
  byte-equality test per copy (skips only when `model-bench/` is absent); a row-count pin per file
  counted from raw non-blank lines; hand-derived tests for turn normalization and
  `render_guard_item` (including a ten-row turn-order and cap case, Pass 2 P2-F1), which no
  cross-component vector covers.
- `AGENTS.md` (current state; the byte-copy invariant) and `README.md` (status).

**Verified:** `pytest -q` from `decision-models/` 54 passed (26 → 54); `ruff check .` clean.
Twenty-one mutants each redden at least one test, re-run from a private driver after a failed
restore was recovered (`render.py` md5 `2ce93c54…`); the restore failed because a parallel agent and
this unit both wrote `scratchpad/mutate.py`. Independent re-sample:
`docs/reviews/typed-decision-models4-impl.md` Pass 2.

## 2026-10-31 — typed-decision-models U5.1 — component scaffold and the FR-8 isolation test

**What:** `docs/plans/typed-decision-models4.md` §4.1's U5.1 (v1.3 §4 U5's skeleton bullet and its
FR-8 test; v1.3's `labeler.py` is void and not scaffolded).

- `setup.sh` (model-bench's, renamed), `pyproject.toml` (no runtime dependencies; dev extra
  `pytest`, `ruff` at model-bench's pins; `pythonpath = ["."]`), `.gitignore`, `README.md`,
  `AGENTS.md`, and the empty `decisionmodels/` package with `__version__`.
- `tests/test_fr8_isolation.py`: an AST import allowlist (stdlib + `decisionmodels`, relative
  imports; `__import__`/`import_module` count, a non-literal module name is refused) and a scan of
  non-docstring string literals for `falkor-chat`/`falkorchat`/`falkor_chat`, `falkordb`,
  `redis` and `guard-judge-understanding`, plus the integer 6379, over `decisionmodels/` only
  (plan2 §3.5). Fourteen planted-red cases and a negative control.
- `tests/test_package.py`: `__version__` equals `pyproject.toml`'s; `dependencies` is empty.
- Root `AGENTS.md`: one Structure bullet and one Component-docs row.

**Verified:** `pytest -q` from `decision-models/` 20 passed; `ruff check .` clean. Planting
`import redis` (and, separately, `import falkorchat`, `from modelbench import packs`, a
`falkor-chat` path, the side-check items path and `6379`) in `decisionmodels/__init__.py`
reddened `test_decisionmodels_reaches_only_stdlib_and_itself`; the file was restored `cmp`-equal
each time.
- Review Pass 1 fixes (`docs/reviews/typed-decision-models4-impl.md` F1–F3, F5, F6): a string-literal planted case per fragment (dropping any one fragment reddens the suite), word-bounded `redis` with "predisposition"/"redistribuir" in the negative control, bytes literals and `from importlib import import_module as …` aliases scanned, the docstring's list of what the tripwire cannot see, `AGENTS.md`'s side-check-reader wording, and the root bullet naming the requirements doc; 26 passed.
