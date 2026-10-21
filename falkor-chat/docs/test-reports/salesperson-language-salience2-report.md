# Salesperson language salience — dedicated-instance rerun (K-065 / K-066) — Test Report

> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** K-065, K-066 · **Extends:** `falkor-chat/docs/test-reports/salesperson-language-salience-report.md`

## Summary

**What was tested:** a dedicated-instance rerun of TP-007's literal-concurrency protocol against
the deployed `salesperson@v8` (Mitigation D) storefront, at **n=75 trials** (1,125 turns), using
the committed harness `salesperson/scripts/language_salience_probe.py` (U4) after its static review
(U5, `falkor-chat/docs/reviews/salesperson-language-salience.md` `## Pass 2`, approve with
suggestions). Run 2026-10-20, 21:09:53 → 22:16:24, repo `HEAD` `3053a441`. It serves two items:
**K-065**, a tighter bound on wrong-language replies, and **K-066**, a trustworthy dead-turn
baseline.

**K-065 result:** **0 wrong-language replies out of 1,119 replied turns**, by full hand-read
(authoritative) and by the v2 classifier (no disagreement, no `unknown`):

| Measure | Result | Wilson 95% CI |
|---|---|---|
| Trial level, each language | 0/75 | [0.0%, 4.9%] |
| Turn level, each language | 0/373 | [0.0%, 1.0%] |

This is down from the 2026-09-18 bound of 0/23 [0.0%, 14.3%]. Whether this clears the demo bar is
the stakeholder's call.

**K-066 result:** **6 of 1,125 turns dead (0.5%, Wilson [0.2%, 1.2%])**, against 62.7% on
2026-09-18.
- All 6 were `ministral` "Context size has been exceeded" errors, in two simultaneous bursts of
  three (trials 11 and 42).
- There were 0 `qwen` failures, and none of the earlier run's `terminated`/HTTP 500/model-load
  failures.
- This holds with all three demo models **explicitly pinned resident**. JIT warm-up alone could not
  keep them co-resident, and the demo needs both LLMs.

**Correction:** the 2026-09-18 report attributed its `qwen` failures to the `:8200` process. The
storefront demo itself calls `qwen`, through the `query_graph_data` tool's `step`-role model. See
the correction section.

**CPG:** considered, not relevant — this is live behavioral measurement of LLM output and serving
failures. The one code question (which models the demo calls) was answered by reading
`proof_defs.py`/`tools.py` directly, and `cpg_falkorchat` is stale.

## Environment

- **Repo:** `HEAD` `3053a441`; `SALESPERSON_DEF` last changed in `999141f` (Mitigation D, `v8`) —
  the same deployed config as the 2026-09-18 run.
- **Harness:** `salesperson/scripts/language_salience_probe.py`, sha256
  `5949774a8aafefc8f4b6b3afe2cac381ad1bf8b61f0b3bf7abb5e5831bb10787`, unmodified for this run
  (the U4 deliverable as handed to `analyst`'s U5 review; classifier `v2`).
- **Demo:** `falkor-chat/scripts/start_demo.sh` (never `start_server.sh`; `--reload` off),
  `FALKORCHAT_OPENCODE_CONFIG=~/.config/opencode/opencode.local.json` (the `localhost:1234` copy;
  the shared file still names the unreachable LAN address). Bring-up logged
  `verify_catalog` → `RESULT: OK — product catalog in sync (15 products)`,
  `verify_salesperson` → `RESULT: OK — 2 defs in sync between reference and ws:demo`, and the
  storefront preflight `ws=demo agent=assistant def=salesperson@v8 products=15 images=14`.
  Server log kept on disk: `/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/01483502-06e5-42ad-9b05-36efaa8d1ff1/scratchpad/demo-u6.log`.
- **Models the demo calls** (`falkor-chat/config/models.json` + `ws:demo`'s `WorkspaceConfig`,
  which carries only `embeddingModelOverride: lmstudio/text-embedding-qwen3-embedding-0.6b`, no
  `agentModelOverride`): `mistralai/ministral-3-3b` (the `assistant` step's pinned
  `config.model`), `qwen/qwen3-4b-2507` (the shared `step`-role default, reached through the
  `query_graph_data` tool — see the correction section), and
  `text-embedding-qwen3-embedding-0.6b` (embedding, dim 1024).
- **Preload — JIT warm-up was not enough.** At 21:08:39, before anything was touched, LM Studio
  had `qwen/qwen3-4b-2507` loaded by a consumer this pass did not identify (the U4 smoke run's
  post-run snapshot at 21:04:50 showed only `ministral`; candidates are the `:8200` server or this
  demo's own bring-up, not distinguishable from here). One JIT warm-up request each (`ministral` chat, `qwen` chat, embedding) all succeeded,
  but afterwards **only the embedding model was resident** — each JIT load evicted the previous
  one. The models were therefore loaded explicitly with `lms load <model> -c 8192 -y`
  (`ministral` 4.18 GiB, `qwen` 2.33 GiB — default load config, `PARALLEL 4`, no TTL), and one
  warm-up request each was repeated (`ministral` → `'OK! How can I'`, `qwen` → `'OK.'`,
  embedding → 1024-dim vector). `lms ps` then listed exactly the three demo models. Nothing was
  unloaded by this pass.
- **Other consumers:** the `ws:agent-team` server on `:8200` (pid 292702) was running throughout
  (not stopped — other sessions use it); it is visible in both harness snapshots. No other model
  was loaded at run start (see the snapshot section).
- FalkorDB `falkordb-dev` up throughout.

## Protocol

Unchanged from the 2026-09-18 report's "Protocol" section and `docs/test-plans/salesperson-ui.md`
TP-007 (literal-concurrency variant): per trial, one `en` + one `pt-BR` + one `es` participant
join concurrently and each runs the 5 scripted turns of `salesperson/scripts/load_demo.py`; 75
trials (375 turns per language, 1,125 total), run as:

```
python3 salesperson/scripts/language_salience_probe.py run --trials 75 \
    --base-url http://127.0.0.1:8000 --lmstudio-url http://localhost:1234 \
    --out <scratchpad>/salience2.jsonl
python3 salesperson/scripts/language_salience_probe.py aggregate <scratchpad>/salience2.jsonl --dump-replies
```

Differences from 2026-09-18: the harness is the committed rebuild (one flushed JSONL line per
turn, aggregate re-derived from raw data); the classifier is the same `v2` rule (shared
Ibero-Romance words vote for neither pt nor es); a reply that classifies as `unknown` (tie / no
signal) is counted separately as "unclassified", never as wrong; the three models are pinned
resident rather than JIT-loaded. Dead-turn failures were attributed to a model by joining each
dead turn (by `participantId` and the turn's `[posted_at, idle_at]` window) to the demo log's
`falkorchat.storefront` failure line and its `ProviderCallError: lmstudio/<model>` — script kept at
`<scratchpad>/attribute_failures.py` (not committed; diagnostic). As before, every reply was
classified by the heuristic **and** hand-read.

## K-065 — language adherence

**Run size:** 75 complete trials (75 `trial_end` records, no partial trial, no resume), 1,125
turns, 1,119 with a reply. Window 21:09:53 → 22:16:24 (run_start → run_end).

**Authoritative count: the hand-read.** Every one of the 1,119 replied turns, in all three
languages, including every reply the classifier passed, was read in full from
`aggregate --dump-replies`-equivalent output. One judgment per reply is recorded at
`/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/01483502-06e5-42ad-9b05-36efaa8d1ff1/scratchpad/salience2-handread.jsonl`,
next to the run file `salience2.jsonl`. It has 1,119 lines, one per replied turn, no duplicates and
no replied turn missing (checked by script). **Hand-read result: 0 wrong-language replies in any
language.** The v2 classifier agrees with the hand-read on all 1,119 turns, with 0 disagreements
and 0 `unknown`.

Turn level, of replied turns (dead/no-reply turns excluded from the denominator):

| Language | Replied turns | Wrong-language (hand-read) | Wrong-language (v2 classifier) | Unclassified (v2) | Wrong rate | Wilson 95% CI |
|---|---|---|---|---|---|---|
| `en` | 373 | 0 | 0 | 0 | 0.0% | [0.0%, 1.0%] |
| `pt-BR` | 373 | 0 | 0 | 0 | 0.0% | [0.0%, 1.0%] |
| `es` | 373 | 0 | 0 | 0 | 0.0% | [0.0%, 1.0%] |

Trial level (a trial is affected when it has at least one wrong-language replied turn), of trials
with at least one reply:

| Language | Trials w/ ≥1 reply | Affected (hand-read) | Affected (v2) | Rate | Wilson 95% CI |
|---|---|---|---|---|---|
| `en` | 75 / 75 | 0 | 0 | 0.0% | [0.0%, 4.9%] |
| `pt-BR` | 75 / 75 | 0 | 0 | 0.0% | [0.0%, 4.9%] |
| `es` | 75 / 75 | 0 | 0 | 0.0% | [0.0%, 4.9%] |

**Trials with at least one `unknown` reply (U5 review F2): 0.** No reply classified `unknown`, so
there was nothing to resolve by hand, and the "0 affected" trial-level figure does not depend on
how unclassified replies are treated. The review's F1 false-negative shapes (a Spanish reply
scoring `en` through `7-in-1`, or `con`/`en`/`una` voting for nobody) cannot hide a wrong reply
here, because the hand-read covered every reply the classifier passed.

**Comparison** (trial-level, `en` participant, the shape of the original headline):

| Source | Protocol | Affected | Wilson 95% CI |
|---|---|---|---|
| QA original (`docs/test-reports/salesperson-ui-report.md`) | 3-way, n=10, pre-mitigation | 2/10 | [5.7%, 51.0%] |
| 2026-09-18 (`falkor-chat/docs/test-reports/salesperson-language-salience-report.md`) | 3-way, n=25 (23 w/ reply), `v8`, shared/contended LM Studio | 0/23 | [0.0%, 14.3%] |
| **This run** | **3-way, n=75 (75 w/ reply), `v8`, models pinned resident** | **0/75** | **[0.0%, 4.9%]** |

Turn level, the 2026-09-18 run had 61/46/33 replied turns (`en`/`pt-BR`/`es`), all adherent. This
run has 373 per language, all adherent. These are the measured rates; whether a 0/75 trial-level
result (upper bound 4.9%) clears the bar for the live demo is the stakeholder's call.

**Not measured:** language drift inside tool-call arguments that never reach the user (the
2026-09-18 trial-12 sub-observation). This run's evidence is the delivered replies and the demo's
own log, not LM Studio's per-request log.

## K-066 — dead-turn accounting

Dead = `turn.lastTurn == 'failed'` read at `idle` (the DEF-3 latch).

| Language | Turns | No-reply | Dead-but-replied | Dead total | No-reply, not dead | Dead rate | Wilson 95% CI |
|---|---|---|---|---|---|---|---|
| `en` | 375 | 2 | 0 | 2 | 0 | 0.5% | [0.1%, 1.9%] |
| `pt-BR` | 375 | 2 | 0 | 2 | 0 | 0.5% | [0.1%, 1.9%] |
| `es` | 375 | 2 | 0 | 2 | 0 | 0.5% | [0.1%, 1.9%] |
| **all** | **1,125** | **6** | **0** | **6** | **0** | **0.5%** | **[0.2%, 1.2%]** |

For comparison: the 2026-09-18 run had 235/375 no-reply turns (62.7%, Wilson [57.7%, 67.4%]) on a
contended, JIT-loading instance. The U4 smoke run had 20/30 (66.7%), also with nothing pinned
resident.

**By failing model** (each dead turn joined by `participantId` and `[posted_at, idle_at]` to its
`falkorchat.storefront` failure line in the demo log; 6 failure events, all 6 matched, 0 unmatched,
0 failure events on a turn not marked dead):

| Failing model | Dead turns | Error |
|---|---|---|
| `mistralai/ministral-3-3b` | 6 | `HTTP 400 Bad Request: {"error":"Engine protocol predict stream returned an error: {\"code\":500,\"message\":\"Context size has been exceeded.\"…` (all 6, verbatim prefix) |
| `qwen/qwen3-4b-2507` | 0 | — (the demo log has no `qwen` `ProviderCallError` at all) |
| other | 0 | — |

The demo log has 12 `ProviderCallError` lines, which are 6 events (traceback line plus the summary
line each), and no `"terminated"`, `Channel Error`, HTTP 500 or model-load failure, which were the
four shapes behind the 2026-09-18 failures.

**Over time** (dead turns per trial, out of 15; all other 73 trials had 0):

| Trial | Ended at | Dead | Which turns | Failure timestamps |
|---|---|---|---|---|
| 11 | 21:21:28 | 3 | `en` k1, `es` k3, `pt-BR` k4 | 21:21:01.873 / .877 / .878 |
| 42 | 21:48:30 | 3 | `en` k1, `es` k4, `pt-BR` k4 | 21:48:08.924 / .925 / .929 |

There was no warm-up cluster (trials 0–10 had 0 dead) and no degradation trend (trials 43–74 had 0
dead). Both failure events are **simultaneous across all three concurrent participants**: three
different turns, at different turn indices, failed within 5 ms of each other. The `en` turn 1 in
both bursts was a long one (`turn_ms` 40,180 and 40,201). One possible explanation, a hypothesis
not verified here: `ministral` is loaded at context 8192 with `PARALLEL 4` (`lms ps`), so the
concurrent requests share one context budget, and one long conversation overflowing it fails every
in-flight request together. This run can't confirm it (no LM Studio-side log, and no comparison at
a larger context).

**U5 F4 check (abnormal paths that could misattribute a reply), counted from the raw JSONL:**

| Trigger | Count |
|---|---|
| Timed-out turns (`timed_out: true`) | 0 |
| `POST /messages` returning 409 | 0 |
| `POST /messages` non-200, incl. transport `-1` | 0 |
| State-poll errors / messages-read errors (the `errors` field) | 0 / 0 (`errors` is empty on all 1,125 records) |
| Turns that never reached `idle` (`idle_at` null) | 0 |
| Replies with `reply_created_at` outside their own turn's `[posted_at, idle_at]` window | 0 of 1,119 |

**No turn's attribution is suspect.** `no_reply_not_dead` is 0. The review's caveat is that a
timed-out turn would land there with `dead_turn: false`, but no turn timed out. One record looked
odd on reading: trial 42 `en`, where turn 1 is dead and turn 2's reply contains the product list
plus the add-to-cart confirmation. That reply's `createdAt` falls inside turn 2's own window, so
the model produced the list on turn 2, and nothing slipped across turns.

**DEF-3 latch:** every no-reply turn (6/6) carried `lastTurn == 'failed'`, and every failure event
in the log maps to a turn marked dead. The latch never missed under this load.

## LM Studio snapshots

From the run file's `run_start.lmstudio_pre` / `run_end.lmstudio_post` (`/api/v0/models`, state
`loaded`, plus local falkorchat uvicorn processes from `/proc`):

| | Taken at | Loaded models (id · type · quant · ctx) | falkorchat uvicorn processes |
|---|---|---|---|
| Pre-run | 21:09:53.251 | `text-embedding-qwen3-embedding-0.6b` · embeddings · Q8_0 · 2048; `mistralai/ministral-3-3b` · vlm · Q8_0 · 8192; `qwen/qwen3-4b-2507` · llm · Q4_K_M · 8192 | pid 292702 (`--port 8200`, the `ws:agent-team` server); pid 437075 (this demo, :8000) |
| Post-run | 22:16:24.573 | `mistralai/ministral-3-3b` · 8192; `qwen/qwen3-4b-2507` · 8192 | the same two |

- Pre-run: exactly the three demo models were loaded, and nothing else.
- Post-run: the embedding model was **no longer loaded**. It was the only one of the three left on
  its JIT `1h` idle TTL (`lms ps` at preload), and nothing in this pass unloaded it. Its expiry
  within the 66-minute run is consistent with the storefront flow making no embedding calls during
  the run. `SALESPERSON_DEF`'s tool list (`proof_defs.py:448-452`) has no `graphrag_retrieve`. Not
  verified beyond that.
- The `:8200` server was present in both snapshots, so it could have reached LM Studio. This pass
  has no evidence that it did (no foreign model loaded at either point), and none that it didn't.

## Correction to the 2026-09-18 report's `qwen` attribution

The 2026-09-18 report (§"Engine-stability observation") attributed its 102
`lmstudio/qwen/qwen3-4b-2507` failures to **cross-process noise** from the `:8200` `ws:agent-team`
server, reasoning that `SALESPERSON_DEF.steps[0].config.model` is `ministral` only. **That
inference is wrong, or at least incomplete: the storefront demo itself calls `qwen`.**

- **Code:** `SALESPERSON_DEF`'s `assistant` step offers `query_graph_data`
  (`falkor-chat/server/falkorchat/proof_defs.py:448-452`). `QueryGraphDataTool.run`
  (`falkor-chat/server/falkorchat/tools.py`, around line 983) calls `self._models.llm("step",
  ws=ctx.ws)`. That resolves to the shared `step`-role default `lmstudio/qwen/qwen3-4b-2507`
  (`falkor-chat/config/models.json`), because `ws:demo`'s `WorkspaceConfig` carries no
  `agentModelOverride`. The step's own `config.model` pins only the agent's chat model, not the
  models its tools resolve.
- **Live evidence (U4 smoke run, 2026-10-20, this demo process, not `:8200`):** the demo's own log
  recorded `ProviderCallError: lmstudio/qwen/qwen3-4b-2507 … Failed to load model
  "qwen/qwen3-4b-2507". Error: Engine protocol startup was aborted.` with a traceback through
  `executor._handle_tool_call` → `tools.py:985` → `llm.complete`. Six of that smoke run's 21 dead
  turns were attributed to `qwen` by the same join used above.
- **Consequence for the 2026-09-18 figures:** some or all of its 102 `qwen` load failures were
  self-inflicted model swapping by the demo (LM Studio JIT-loading `qwen` and evicting `ministral`,
  and back), not another consumer. This pass saw the same eviction directly: one JIT warm-up per
  model left only the last one resident. That is consistent with the "persistent condition" the
  earlier report described. This run, with all models pinned resident, had **zero** `qwen` failures
  and none of the 2026-09-18 failure shapes.
- The 2026-09-18 report's K-065 figures (0/23) are unaffected, because they count delivered replies
  only.

## Replies classified wrong-language or unclassified (verbatim)

**None.** Under the v2 classifier, 0 of 1,119 replies were classified wrong-language and 0 were
`unknown`. Under the hand-read, 0 were wrong-language. There is no hand-read/classifier
disagreement to list. For reviewers, the observations from the hand-read that did *not* change a
judgment:

- `pt-BR` replies often carry Spanish-looking or Italian-looking product-name renderings in
  otherwise Portuguese text (e.g. `**Altoparlante Bluetooth Mini**`, `**Padão sem fio de
  carregamento**`), and the model's poor product-name translations (`'Mouse Semetral Pro sem fio'`).
  They were judged Portuguese replies: the sentence language is Portuguese throughout.
- `es` replies sometimes quote the English catalog name (`**Wireless Mouse Pro**`,
  `(Wireless Mouse Pro)`), and `pt-BR` ones occasionally do too (`*Wireless Mouse Pro*`). Judged
  per the 2026-09-18 definition (dominant reply language) as adherent.
- `en` replies never contained Spanish or Portuguese sentences.

## Coverage & gaps

- **Covered:** TP-007's literal-concurrency variant (3-way, `en`/`pt-BR`/`es`, 5 scripted turns)
  at n=75 against the deployed `salesperson@v8`, with all three demo models pinned resident on an
  instance the stakeholder confirmed was dedicated for the window. Every replied turn was hand-read.
- **Not covered:** the 40-way heavy variant; C/B mitigations (unauthorized); AC-8 and other ACs;
  tool-call-argument language drift (not observable here).
- **Environment caveat — this is a "pinned-resident" baseline, not a "JIT" one.** The preload
  needed explicit `lms load` (JIT warm-up evicted each model in turn). A demo-day instance left on
  JIT loading would not get these conditions, and the smoke run and the 2026-09-18 run suggest it
  would look like those runs instead (about 63–67% dead turns). The dead-turn figure here holds
  only for the pinned configuration.
- **Context-overflow failures** (the six dead turns) are characterized only from the demo side.
  The shared-context-budget explanation is a hypothesis.
- The `:8200` server ran throughout and is a residual, unmeasured possible consumer.
- The hand-read is one reader's judgment, recorded per reply on disk for audit.

## Feedback & recommendations

1. **K-065: report the measured number to the stakeholder.** 0/75 trials affected per language
   (Wilson upper bound 4.9%), and 0/373 replied turns per language (upper bound 1.0%), confirmed by
   a full hand-read. The sufficiency call is the stakeholder's.
2. **K-066: the dead-turn rate depends on the LM Studio load configuration, and the demo-day
   runbook should pin it.** All of `mistralai/ministral-3-3b`, `qwen/qwen3-4b-2507` and the
   embedding model should be loaded explicitly (`lms load …`, not JIT), because the demo calls both
   LLMs (see the correction section). The 0.5% dead rate measured here holds only in that state.
   Owner suggestion: `devops` (demo bring-up), possibly as a preflight in `start_demo.sh` that
   checks `/api/v0/models` for all three. That change is not made here.
3. **The remaining failure mode is context overflow at 8192.** It came in two bursts, each failing
   all three concurrent participants at once. Worth a follow-up: re-run with `ministral` loaded at
   a larger context, or with `--parallel 1`, and see whether the bursts disappear. Without it,
   long conversations on demo day can still fail every in-flight participant at once.
4. **Correct the record:** K-066's body and the 2026-09-18 report's `qwen` attribution (that
   report is not edited here; this report extends it).
5. **Harness:** it ran without incident (no timeout, 409, transport error or abort, so no
   `--resume` was needed and U5's F7 did not apply). U5's F1/F2/F4/F6 improvements are still worth
   making before the next run. They didn't affect this run's figures, because the hand-read was
   complete and the F4 triggers were all zero. They are a separate follow-up unit and were not
   implemented here.
