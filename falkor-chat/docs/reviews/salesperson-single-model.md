# Review — Salesperson demo on one LLM (`query_graph_data` inherits the calling step's model)

> **Status:** active · **Owner:** `analyst` · **Tracks:** K-066

## Pass 1 (2026-10-21)

### Scope & verdict

Reviewed `falkor-chat/docs/plans/salesperson-single-model.md` (U1, untracked) against its
coordination doc `falkor-chat/docs/plans/salesperson-single-model-coordination.md` and the live tree
at `9b129b18`. I read `executor.py` (`_drive`, `_run_agent_node`, `_handle_tool_call`), `tools.py`
(`ToolRegistry`, `QueryGraphDataTool`, `build_builtin_registry`), `modelconfig.py` (`resolve`,
`_workspace_override_ref`, `StaticModelGateway`), `app.py` wiring, `proof_defs.py`, the eval runner,
`start_agent_team.sh`/`start_demo.sh`, the test stubs, and the model-bench run records. I checked the
`lms` CLI on this box and the docs the plan lists. Nothing was edited. No tree-mutating command was
run. The pytest runs used no DB fixtures (`test_executor_agent.py` in full, plus
`test_tools.py -k query_graph`: 83 passed).

**Verdict: needs changes, confined to §7 (live QA) and one §2.3 omission.** The code design
(§3.1), the steps (§4) and the unit/integration tests (§5) are sound and grounded. `coder` can
implement S1–S8 as written, with the two minor test notes below. §7 must be revised before U3
runs. As written it overwrites a tracked file that also holds its own comparison baseline. It has
no pass bar for the done-condition's "tool/NLQ correctness held". Its GPU-fit criterion can't see
the spill this whole change exists to remove.

CPG: considered, not relevant — `cpg_falkorchat` is stale (8 commits since its 2026-10-15 build) and every claim under review was a handful of call sites, verified by reading source directly.

**Premise checks (the brief's six):**

1. **Consumer set: confirmed.** `ToolRegistry.dispatch` has one production caller,
   `executor.py:1067`. `QueryGraphDataTool` is built only in `build_builtin_registry`
   (`tools.py:1092`, wired only under `WORKFLOW_ENABLED`, `app.py:632-640`) and in the eval runner
   (`run_nlq_golden_set_eval.py:251`, `run={}`). `mcp.py`, `storefront_api.py` and `responder.py`
   have no path to it. No tool writes into `run`, and the only `run[...] =` writes are the two
   `_drive` stamps (`executor.py:569,578`). So a per-node copy is safe. `:8200` exports neither
   workflow flag (`start_agent_team.sh:279-281`). The demo's other LLM paths don't touch qwen
   per turn: salesperson and order-fulfillment guards are all `cmp`, not `llm`, and the responder
   fall-through is off.
2. **`requested_model`: correct.** It's the raw `config.get("model")` (`executor.py:824`), which is
   exactly what the node itself resolves with. The tool resolves through `overrides=` with the
   same drive snapshot. I probed the real `ModelGateway` with the shipped `models.json`. Requested
   ministral plus an all-`None` `agentModel` resolves to ministral/`requested`. An `agentModel`
   override resolves to the override/`workspace`. No requested value resolves to qwen/`default`.
   `StepRun.modelSource` is untouched, since the tool records nothing.
3. **Bench finding: confirmed.** In both ministral run records, `nlq-12` = `malformed_spec`,
   `nlq-27` = `schema_violation`, `scoreable={}`, and `exactMatch` = 32/32. The plan omits a second
   signal (m2 below). The acceptance criteria don't cover the risk (M2 below).
4. **Context/KV and Q1–Q5:** `lms.exe` exists, and `-c`, `--parallel`, `--estimate-only`, `--gpu`
   and `--ttl` are all listed. The probe CLI flags match. Problems are in M1, M3, m3, m4 and m5.
5. **Mutations:** M1–M3 do discriminate. T7 as described can't be built from the double the plan
   recommends, and it asserts arguments, not the client used (m1).
6. **Docs:** mostly complete. Two misses are listed in m6.

### Findings

#### Major

**M1 — Q2 overwrites a tracked file that is also its own comparison baseline.**
`run_nlq_golden_set_eval.py:91,186-195` always writes
`falkor-chat/server/tests/eval/nlq_eval_results.json`. That file is git-tracked: the qwen K-055
run, `2026-08-30`, 33/39, committed in `08c7a5f6`. Q2's claim that it "touches no running server
and no shared file" is false. It dirties the user's tree, and it destroys the very "qwen figure in
this script's last recorded run" Q2 says to compare against. That figure is in this file, not in
`docs/test-reports/`. Q2 also asks for "the raw completion of every miss", but the runner stores
only `toolResult` (the abstention JSON), so nothing in Q2 as written captures the completion.
**Fix:** before running, have QA copy `nlq_eval_results.json` to the scratchpad. After the run,
the file must be restored, and `git status` must show it clean. Better still, route a tiny
`NLQ_EVAL_RESULTS_PATH` env override to `coder` as an S-row; a doc can't fix this. Name the
baseline explicitly: qwen 33/39, `nlq-12`/`nlq-27` both `layer1Correct: true`. For raw
completions, run Q2 through a small scratchpad wrapper that wraps `gateway.llm` to log every
`complete()` reply. Or drop the requirement and say why.

**M2 — The done-condition "tool/NLQ correctness held" has no pass bar.** The coordination DoD
requires correctness *held*. §7's overall acceptance is Q1 + Q3 + Q5. Q1's correctness clause
passes on "correct … *or the miss is recorded verbatim*", so it can't fail. Q2 is report-only.
The known failure (`nlq-12`'s compound filter, which production turns into a false "no matching
data found", §2.3) is exactly the false negative a shopper would see, and no step can turn red on
it. **Fix:** make Q2 a gate relative to the recorded qwen baseline. For example: no catalog-dataset
pair that qwen got right may regress, or any regression is an explicit stakeholder sign-off
recorded in the report before "pass". Make Q1's `nlq-12`-shape turn a hard check in all three
languages. Otherwise, reword the DoD to "measured", via `teco`, so plan and DoD agree.

**M3 — Q5 can't detect the spill it is meant to rule out, and fit is doubtful.** Under Windows
WDDM, CUDA sysmem fallback and LM Studio's automatic `--gpu` ratio both put weights or KV in host
RAM while `memory.used` stays at or below `memory.total`. So "peak `memory.used` < `memory.total`"
passes in exactly the failure case (inferred from driver and LM Studio behaviour, not reproduced).
`lms ps` on this box has no offload column (checked: `DEVICE Local`). The numbers are also tight.
`lms load mistralai/ministral-3-3b -c 8192 --estimate-only` gives **4.84 GiB**. The embedder
estimate is **853 MiB** (confidence LOW). That's ~5.7 GiB against 6,141 MiB total, before
desktop/driver usage (3,787 MiB used now with only qwen loaded). Q4's 12k/16k candidates almost
certainly don't fit. **Fix:** (a) load ministral with `--gpu max` so a non-fit fails loudly instead
of silently splitting. (b) Add a throughput check: ministral generation tok/s under single-request
load should match the bench's ~38 tok/s (`timing.calls[].tokensPerSecond` in the run record), and
a spill shows up as a large drop. (c) Run E2/E3/Q5 **now, before U2**. Fit needs no code (the
assistant step is ministral already), and a non-fit is a stakeholder decision that changes what
U3 should measure.

#### Minor

**m1 — T7's recommended double can't drive the test, and T7 checks plumbing more than routing.**
§5 suggests extending `test_executor_agent.py`'s `RecordingGateway` with `llm(...)`. But its
`resolve_llm` returns a cached `StubChatLLM([ChatResult(text=...)])` per ref
(`test_executor_agent.py:1183-1197`). That emits no tool call, and for the same ministral ref
`llm()` would hand back the same object, which has no `.complete`. **Fix:** spell out a dedicated
T7 double. `resolve_llm` returns a scripted chat stub (one `query_graph_data` call, then text).
`llm` returns a per-ref `StubQueryLLM`. Assert that the **ministral-keyed** query client received
the `complete()` call and that no default-keyed client was ever created. That pins *which client
answered*, not just which args were passed. M1 still reddens it.

**m2 — §2.3 omits ministral's fabrication-direction signal.** In the same records,
`unanswerableAbstainRate` is **0/3** for ministral (`nlq-37..39`: valid spec, non-empty wrong
result) against **1/4** for qwen. So "it isn't a fabrication" holds only for the gated failures.
When ministral does emit a valid spec for a question the schema can't answer, it returns rows. The
items are knowledge_base-shaped, but the tool offers both datasets to the demo agent
(`tools.py` schema enum). **Fix:** one sentence in §2.3 and R3, and have Q2 report the
not-found/unanswerable shapes alongside Layer 1.

**m3 — Q3's "CIs don't overlap" rule tolerates ~3.3× the baseline.** At n≈1,125, up to 20 dead
turns (1.78%, Wilson [1.15%, 2.73%]) still overlaps the baseline's [0.24%, 1.16%] (computed).
**Fix:** state that tolerance explicitly, or set a fixed ceiling (e.g. ≤1.2%, the baseline's
upper bound) for the stakeholder to accept.

**m4 — Q1's dispatch count has no source as written.** Successful tool dispatches aren't logged
(`executor.py` logs only failures, `:1094`). Storefront runs aren't debug runs, so there's no
trace. **Fix:** count from `Message.toolsUsed` with a read-only `GRAPH.RO_QUERY` on `ws:demo`
(persisted by `link_step_emission`, read back at `repository.py:743`).

**m5 — Q4 "reload" and E2 load commands need explicit unload and exact-key steps.** Q4 says to
rerun "with ministral reloaded at N" but gives no `$LMS unload` first. Re-loading an already-loaded
key in LM Studio starts a second instance rather than replacing the first (inferred), which would
corrupt the VRAM reading. Two local models match the key (`mistralai/ministral-3-3b` and
`mistralai_ministral-3-3b-instruct-2512`, `lms ls`), and `-y` loads the first match. E2's `lms ps`
check catches a wrong pick. **Fix:** add `$LMS unload --all` (or a by-identifier unload) before
each reload, and assert the `lms ps` IDENTIFIER equals `mistralai/ministral-3-3b`.

**m6 — Doc list misses two places.** `salesperson/README.md` "Status" (≈l.210: "0.5% with all
three pinned … load them explicitly") becomes wrong; `teco`'s scan names `salesperson/AGENTS.md`,
which has nothing to change. `SERVER.md` §1.8's ASCII diagram (≈l.612) has an executor
`(kind=step)` box but no tool box. Update it together with the binding-sites sentence.

#### Nit

- R2 names only `:8200`. The demo process's own `/mcp` also wires `IngestionPipeline`
  (`app.py:597`), so an `ingest_document` there also resolves `extraction` → qwen. Add one clause
  to the runbook guidance.
- The implicit-`post_message` call site also gets `tool_run`, but no mutation covers it. That's
  harmless (`post_message` ignores `stepModel`). Either say so, or leave that site on `run`.

### What's solid

- §2.1/§2.2 grounding is accurate throughout. The consumer enumeration, the precedence order, the
  `write_model_overrides` clear-all hazard and the `run`-dict carrier all check out against source.
- §3.1 is the minimal correct change. It reuses the gateway's precedence verbatim, and a direct
  `run={}` call (eval runner, existing tests) keeps today's behaviour. I verified that on the real
  gateway. Rejected alternatives B and C are rejected for real, checked reasons.
- T1–T6 and M1–M3 discriminate as claimed. The baseline suites the plan names are green now (83
  passed, no DB fixtures).
- §2.3's byte-identity claim about the prompt and the per-item failure table are correct.

### Open questions

- Should "tool/NLQ correctness held" be a gate (M2's fix) or a reported figure? That's
  stakeholder/`teco`'s call. Plan and DoD currently disagree.
- If the pre-U2 fit check (M3c) shows ministral + embedder don't fit fully on GPU at 8192, does
  the stakeholder still want U2, for instance with a smaller context or `--parallel 2`?

## Pass 2 (2026-10-21)

### Scope & verdict

Re-reviewed plan Version 2 (§4 S9, §5 T7, §7 U0/U3, §8 R6, §9 dispositions) against the tree at
`9b129b18`, plus U0's **in-progress** fit report
`falkor-chat/docs/test-reports/salesperson-single-model-fit-report.md` (read as it stood. Its
throughput section and full-GPU load are filled. The partial-offload contrast and the verdict are
still pending). I checked the fixes against each other as well as one by one.

**Verdict: needs changes, §7 only.** S1–S9 and T1–T7 are ready for `coder` (one minor note on T7).
U3 can't run as written. U0's own measurements already invalidate the Q5 spill criteria, and those
criteria came partly from my Pass 1 suggestions. Q2a has no step in the plan where it can run
without either mutating the tracked file or finding qwen already unloaded.

CPG: considered, not relevant — `cpg_falkorchat` is stale (8 commits since its 2026-10-15 build); every claim rechecked was a handful of call sites, read directly.

**(a) The M2 partial disagreement holds; I confirmed it against the tree.**
- `filter_products` takes `category` plus an inclusive `maxPrice` (`tools.py` schema, ≈l.462).
- The `SALESPERSON_DEF` system prompt sends the agent to `query_graph_data` only when the fixed
  tools don't fit (`proof_defs.py:439-443`).
- So the agent can legitimately answer `nlq-12` without the tool. In that case the tool's own
  correctness on the compound shape is covered by Q2's catalog gate (`nlq-12`, and `nlq-40` if
  Q2a runs).
- The seeded catalog makes "Fitness Tracker Band" the only Wearables item under $200
  (`seed_catalog` rows 94-95: 79.99 vs 249.99), so Q1(c) is a sound check.
- Keeping `tool_run` at the implicit `post_message` site is harmless. `PostMessageTool` reads `run`
  only through `_thread_id_of` (`threadId`/`ctx`).
- Q1's dispatch query runs as written. I ran a read-only query on `ws:demo`: `createdAt` is epoch
  ms, and 200 existing messages already list `query_graph_data`, so Q1(b) is attainable.

### New findings

**B1 (blocker for U3) — Q5's three spill checks pass in the configuration U0 has already shown
pages to RAM.**
- **Q5(a) is refuted.** E2 says "`--gpu max` makes a non-fit fail loudly". U0 shows the load
  succeeds: llama.cpp's fitter logs `failed to fit … abort` and then loads anyway
  (fit report l.37-45).
- **Q5(c) is refuted.** Generation stays at 38.5–40.5 tok/s while about 1.6 GiB of ministral is
  paged out and back in by WDDM (l.69-94). The tok/s check can't see this spill mode.
- **Q5(b) has two problems.** It watches `llama-server` only, but the embedder runs inside the
  LM Studio process (pid 11136, 484 MiB shared, l.55). And "U0's no-spill floor" doesn't exist:
  with both models loaded there is no no-spill state (l.57-60).

Q5 therefore passes a configuration known to swap on every turn. §7's own branch already applies:
"If U0 finds no full-GPU fit at 8192, U3 waits for the stakeholder's call", and U0's interim data
says exactly that.

**Fix:** hold U3 for that call, then rewrite Q5 for whatever the stakeholder chooses.
- **If WDDM paging is accepted:** the measurable cost is the per-switch swap. U0 measured +0.07–0.1 s
  on ministral TTFT and 0.25–1.4 s on the first embedding after a chat. The Q5 criteria become p95
  turn latency and dead turns under that regime, with adapter-total `Shared Usage` (both
  processes) reported, not gated.
- **If it isn't accepted:** re-scope (smaller context, or a smaller quant), and the fit check comes
  first.

Either way, drop "fails loudly" from E2 and the tok/s spill test from Q5. Both rested on my
Pass 1 suggestions (M3 a/b), which U0 has now disproved.

**M4 — Q2a can't run where §7 places it.**
- Q2a must use `NLQ_EVAL_RESULTS_PATH`, which only exists once S9 lands in U2.
- Placed "inside U0's window", it would therefore silently write the tracked
  `nlq_eval_results.json`. The env var is a no-op before S9, and Q2's `git status` check sits in
  Q2's cell, not Q2a's.
- U0 has also already run `lms unload --all` (fit report l.17).
- Inside U3, E2 begins with `unload --all`, then loads only ministral and the embedder. Nothing
  places Q2a before that, and running it afterwards would JIT-load qwen beside them and corrupt
  E3/Q5.

**Fix:** make Q2a the **first U3 step**, after U2 is merged and before E2. Load only qwen
(explicitly) and the embedder, check `git status --short falkor-chat/server/tests/eval/` is clean
afterwards, then proceed to E2. Also correct the stated motive:
- `3d014975` is defense-in-depth that its own message says is unreachable from any production
  caller (`querygen.py`: an op recheck behind the model's `op` field).
- The real reasons for a fresh comparator are `nlq-40` (`d58125ad`, verified: a catalog compound
  filter added after `08c7a5f6`) and the LM Studio runtime drift U0 recorded (2.40.0 → 2.59.0,
  fit report l.14).

**m7 — T7's `RoutingGateway.resolve_llm` must return a `(client, resolution)` tuple.**
`_run_agent_node` unpacks it and reads `resolution.source` (`executor.py:825-829`). The §5 spec
says only "returns a scripted chat stub". **Fix:** say "returns `(scripted_stub,
_FakeResolution(source=...))`", reusing `test_executor_agent.py:1155`.

**Nits.**
- S9's optional unit check depends on "if importing the runner is side-effect-free". It isn't: at
  import it runs `os.environ.setdefault` for `OPENAI_API_KEY`/`FALKORCHAT_OPENCODE_CONFIG` and two
  `sys.path` inserts (`run_nlq_golden_set_eval.py:66-80`), so use the scratch-script or subprocess
  route directly. I found no test that would break from the leak, but don't import it into the
  pytest session.
- Q3's thresholds are counts out of 1,125. State them as rates (≤1.2% pass; fail on Wilson
  non-overlap), because the new run's turn count won't be exactly 1,125. The arithmetic itself is
  right: 13/1,125 = 1.16%, and the CI stays overlapping up to 20.

### Dispositions of Pass 1 findings

- **M1 (Q2 overwrote the tracked baseline; no raw completions): fixed in design.** S9 is sound.
  The proxy wraps only the tool's gateway and leaves `main`'s `gateway.resolve` raw, which is
  correct. The baseline figures check out: tracked file catalog 20/20 true, knowledge_base 13/13
  answerable, 6 misses. Q2a's placement is the new M4.
- **M2 (no correctness pass bar): fixed.** Q2 is now a catalog no-regression gate with recorded
  stakeholder sign-off. The Q1(c) disagreement is confirmed in (a) above. The gate baseline (tracked
  file, or Q2a when it runs) is consistent with Q2a's override rule.
- **M3 (Q5 blind to spill; fit tight): superseded by B1.** The fit was tight, and U0 confirms it
  doesn't fit. The replacement instruments are themselves blind (B1).
- **m1 (T7 double): fixed.** Per-key `query_clients` plus "no other key created" turns T7 red under
  M1 (the tool requests `None`) and under M2 (`stepModel` absent). `_CATALOG_FILTER_REPLY` exists
  (`test_tools.py:740`). See m7 for the tuple.
- **m2 (abstain-rate signal): fixed** (§2.3/R3).
- **m3 (Q3 tolerance): fixed** (see the rate nit above).
- **m4 (Q1 dispatch source): fixed.** I ran the query read-only on `ws:demo`.
- **m5 (unload/identifier): fixed** (E2 and Q4).
- **m6 (two doc sites): fixed** (§6 rows added).
- **Nit, demo `/mcp` ingest: fixed** (R2 and the `falkor-chat/AGENTS.md` row).
- **Nit, implicit `post_message` site: kept by decision.** That's acceptable; confirmed harmless
  in (a).

### Open questions

- For the stakeholder (via `teco`), now concrete rather than hypothetical: U0's interim data shows
  ministral + embedder only co-reside through WDDM paging. Is paging acceptable for the demo (cost:
  a swap on each model switch), or is the single-model goal re-scoped? U3's Q5 can't be written
  until this is answered.

## Pass 3 (2026-10-22)

### Scope & verdict

Reviewed plan Version 3's §6 `falkor-chat/AGENTS.md` row and §7–§9, against the tree and U0's
final fit report. S1–S9 are out of scope: `coder` has the tree mid-diff.

**Verdict: approve with suggestions.** There are no blockers. One major finding (P3-1) should be
fixed before U3 starts. The rest are minor.

CPG: considered, not relevant — `cpg_falkorchat` is stale; every claim here is a config/env lookup or a handful of call sites, read directly.

**(a) Q5's p95 baseline holds up under every definition I tried.** I computed these from
`salience2.jsonl` before the scratchpad was cleared (see P3-1). Turns-level p95 is 24.1 s whether
I took replied turns (n=1,119), all turns (n=1,125) or non-dead turns, and whether I used
nearest-rank or linear interpolation. The 6 dead turns finished in 2–40 s, so they don't move it.
p50 is 6.0 s and max 62.3 s, both as the plan states.

The spread shows up only under other definitions:

| Definition | p95 |
|---|---|
| Per-trial p95 | median 24.1 s, range 16.1–62.3 s |
| Per-session worst turn | 38.2 s |
| By turn index | turn 0: 4.0 s · turn 1: 36.2 s · turn 2: 24.1 s · turns 3–4: 12.1 s |

A trial-cluster bootstrap puts the CI at [22.1, 24.1] s. Values are quantized to roughly 2 s steps
plus 0.1 s, so the ≤29 s pass band really means p95 may rise by at most two poll steps (to 28.1 s).
The band is a fair gate, provided Q3 reuses the same probe and the same 5-turn script.

The plan's stated rationale for the +5 s margin is thin. It cites U0's swap (≤1.4 s), but §7's own
note expects no swaps on the storefront path. The real new latency source is the tool call now
competing for ministral's 4 slots instead of running on qwen's separate process. The band is
empirical, so this doesn't change the gate; the rationale should just say this.

**(b) The `FALKORCHAT_STOREFRONT_TURN_WORKERS=2` lever does what the plan says.** The variable
exists (`config.py:214`, default 4) and sizes the storefront's `ThreadPoolExecutor`
(`app.py:369` → `storefront.py:592`). Each worker runs a whole turn synchronously: the agent's
chat call, then its `query_graph_data` call, one after the other on the same thread
(`storefront.py:1148-1157`). So 2 workers cap in-flight ministral requests from storefront turns
at 2.

Two caveats. First, `start_demo.sh` doesn't set or export the variable, so it must be in the
operator's environment when `start_demo.sh` runs. A prefix or export works, because uvicorn
inherits it. Second, the probe's `turn_ms` includes queue wait, which is what P3-2 is about.

**(c) The "storefront never embeds" premise is true, and stronger than the plan states.**
- `storefront.py`/`storefront_api.py` have no embed call.
- The salesperson def grants no `graphrag_retrieve`, the only tool that embeds (`tools.py:363`).
- `query_graph_data`'s `knowledge_base` path runs `run_structured_query`, which never embeds.
- `start_demo.sh` and its seed/verify scripts make no LM Studio calls.
- Most importantly: with `STOREFRONT_ENABLED=1`, which `start_demo.sh` pins (`:132`), the app
  sets `dev_surface=False`. That un-mounts `api.build_router` *and* `/mcp`
  (`app.py:258-262,580`). So the two front doors that call `_safe_embed` (`api.py:108`,
  `mcp.py:193`) don't exist in the demo process at all.
- The presenter routes live in `storefront_api.py` and don't embed either.

What follows: the demo process can't embed. Open question 1 can be answered "no" now, from the
code. Loading only ministral removes the paging regime entirely. Q5(d) then becomes a confirmation
(expected 0), and Q5(b) measures a swap the demo never pays. The decision belongs to the
stakeholder via `teco`. I recommend deciding it **before** U3, because it changes E2's load set and
makes Q5(b) moot.

### New findings

**P3-1 (major) — The baseline data and U0's instruments live in a session scratchpad that has
already been cleared.** §7 cites `…/scratchpad/salience2.jsonl` as the source of the 24.1 s
baseline, and names `gpumem.ps1` and `bench.py` as the instruments U3 must reuse. As of this pass,
that directory holds only `coder`'s `executor.py.orig`, `tools.py.orig` and `mut.py`. None of
those three files is tracked (`git ls-files`).

So U3 can't "compute the same way" or "reuse U0's instruments" as written. The scratchpad is
per-session and isn't a durable home.

**Fix:** have `qa-engineer`/`teco` recover them, from the salience2/U0 agents' transcripts or by
re-creating them. Then put them somewhere durable: the computation and the scripts, verbatim, in
an appendix of the fit and salience2 test reports, or under `falkor-chat/server/tests/eval/` or
`salesperson/scripts/`. The baseline numbers themselves stay valid, because I re-derived them
before the files went (above).

**P3-2 (minor) — Q4 judges the queueing lever against a band built for paging.** With 2 workers
and 3 participants, the third participant's turn waits in the queue, and `turn_ms` counts that
wait. A p95 above 29 s is therefore likely by construction, so the lever would "fail" Q5(a)'s band
even if it eliminated every overflow. **Fix:** report Q4 as a trade-off: overflows avoided vs p95
and turn-1 p95, since turn 1 is the heaviest index at 36.2 s. Put it to the stakeholder rather
than gating it on Q5(a).

**P3-3 (minor) — My Pass 1 nit about "the demo's own `/mcp`" was wrong.** It is now in R2, in the
§6 row, and already in `falkor-chat/AGENTS.md` (current working-tree text). In storefront mode
`/mcp` isn't mounted (see (c)), so the demo process can't `ingest_document`. The advice is harmless
but false, and it would mislead the next reader about what the demo exposes. **Fix:** keep only
`:8200` (plus any non-storefront falkor-chat process on the same LM Studio) as the qwen risk, and
drop the demo-`/mcp` clause from R2, from the §6 row and from `falkor-chat/AGENTS.md`.

**Nits.**
- R3 still says "against the tracked qwen baseline". Q2 now gates on Q0, with the tracked file as
  fallback; align the wording.
- Q5(b)'s first-embed pass band (≤1.5 s) sits just above U0's single 1.385 s reading, so it will
  flake if Q5(b) is kept. It's moot if the embedder is dropped.

### Dispositions of Pass 2 findings

- **B1 (Q5 spill checks refuted): fixed.** Q5 now measures the accepted paging cost, counts both
  processes and the adapter total, and drops tok/s and the "fails loudly" claim. I checked these
  against the fit report's final text.
- **M4 (Q2a placement): fixed.** Q0 is the first U3 step: after S9, before E2, with qwen and the
  embedder only, followed by `git status`. The motive is corrected (`nlq-40` plus the runtime
  drift).
- **m7 (T7 tuple): deferred to the implementer** per `teco`. That's acceptable; it's out of scope
  for this pass.
- **Nit (S9 import side effects): deferred to the implementer.** Acceptable.
- **Nit (Q3 thresholds as rates): fixed.**

### Open questions

- Should the demo load ministral only? (c) shows the demo process can't embed. That's a
  stakeholder call via `teco`, ideally made before U3's E2.

## Pass 4 (2026-10-22) — implementation (S1–S9, code gate)

### Scope & verdict

This pass reviews the uncommitted `git diff` against plan v3 §3–§6, for these files:
- `falkor-chat/server/falkorchat/{tools,executor,proof_defs}.py`
- `falkor-chat/server/tests/{test_tools,test_executor_agent}.py`
- the new `falkor-chat/server/tests/test_query_graph_data_routing.py`
- `falkor-chat/server/tests/eval/run_nlq_golden_set_eval.py`
- `falkor-chat/AGENTS.md`, `falkor-chat/docs/SERVER.md`, `salesperson/README.md`

`claude/**` and `resume_*.sh` are excluded.

**Verdict: approve.** I found no blockers and no majors. There are two nits and one pre-existing
item for `teco`. Per the root `AGENTS.md` rule, implementation findings belong in a separate `-impl`
review. This pass sits here only because the coordinator asked for it as `## Pass 4`.

CPG: considered, not relevant — `cpg_falkorchat` is stale and the diff is three call sites plus tests, read directly.

**Evidence I ran myself:**
- **Fast offline subset.** `test_executor_agent.py` in full, `test_query_graph_data_routing.py`,
  and `test_tools.py -k query_graph`, run from `falkor-chat/server`: **90 passed**, which is my
  Pass 1 count of 83 plus T1–T7. None of these tests requests `conn` or `wf_repo`. I did not run
  the full suite, as instructed.
- **M2 replay.** I applied M2 in a scratchpad copy of `server/`, not the working tree (main-loop
  site back to `run`): **3 failed** (T4, T5 and T7), then I restored the copy.
- **Implicit `post_message` site mutated:** 90 passed. That is uncovered, as the plan decided (§5).
- **S9 recorder exercised in a subprocess** with a stub gateway and `NLQ_EVAL_RESULTS_PATH` set:
  - The path override is honoured, and `_display_path` prints an absolute path outside the repo.
  - The proxy forwards `kind` unchanged and `requested`/`ws`/`overrides` as given.
  - `toolCompletion` holds the raw text.
  - `take()` resets between pairs.
  - The unknown-dataset path records `None`.

  `git status` shows no change to `nlq_eval_results.json`.

**Against the plan:**
- **§3.1/S2/S4 match exactly.** `tool_run = {**run, "stepModel": requested_model}` is built once,
  right after the node's own resolution. It's passed at both `_handle_tool_call` sites. `run` is
  never written to.
- **The tool's call matches the plan's spec.** It is
  `llm("step", requested=run.get("stepModel"), ws=ctx.ws, overrides=run.get("modelOverrides"))`.
  The shallow copy keeps the same `modelOverrides` object, which is what T7's `is` assertion pins.

**Do the tests prove the routing? Yes.**
- T7 keys query clients by `requested`. It asserts that the set is exactly `{ministral}`, so the
  default `None` key failing to appear is itself checked.
- It asserts that the ministral client's `complete()` ran exactly once, and that the overrides
  object passes through by identity.
- So it reddens under M1 (tool requests `None`), under M2 (no `stepModel`), and under the
  coordinator's `overrides=None` mutant.
- T1's override dict uses the real four `read_model_overrides` keys.
- `RoutingGateway.resolve_llm` returns the `(client, _FakeResolution)` tuple, which closes m7.

**S9 can't change eval behaviour:**
- The proxy only intercepts `complete()`, which is the tool's only call on that client.
- `_render_answer` and `main`'s `gateway.resolve("step")` still use the raw gateway.
- With the env var unset, the path and the overwrite behaviour are as before, and the docstring
  now warns about the overwrite.

**Docs vs code:**
- `SERVER.md` §1.8's "six binding sites" are correct as listed, and the diagram matches.
- The `AGENTS.md` trap (1) and demo paragraph match the code, including the coordinator's P3-3
  edit (`dev_surface` → no `/mcp`, verified in Pass 3).
- The dropped K-065 sentence is history, which the context-file rule says doesn't belong there.
  Its record is `HISTORY.md` 2026-10-21.
- `salesperson/README.md` is consistent with that.

### Findings

**Nits.**
- `run_nlq_golden_set_eval.py` writes the results file only after all ~40 live pairs, which is
  tens of minutes. An `NLQ_EVAL_RESULTS_PATH` whose parent directory doesn't exist therefore fails
  at the very end and loses the run. Either check `parent.is_dir()` up front in `main()`, or have
  Q0/Q2 operators create the directory first.
- T4–T6 drive the node through `_executor(llm=…)`, the `StaticModelGateway` sugar. That's fine:
  they test the carrier, not the resolution. Just noting that T7 is the only test that covers
  resolution.

**Pre-existing, for `teco` (R5): still open, and not yet in the backlog.** `IngestionPipeline`
resolves kind `extraction`. That kind sits outside `modelconfig.KINDS`/`_KIND_TO_OVERRIDE_KEY`, and
it isn't among the "six binding sites". So `SERVER.md` §1.8's "Four closed consumer kinds" sentence
stays false: a workspace override can't reach extraction. `falkor-chat/docs/BACKLOG.md` has no
item for this (grep). It needs one. It is not a defect of this diff.

### Dispositions of Pass 3 findings

- **P3-1 (scratchpad baseline and instruments gone): not fixed.** This was a plan/U3 concern and is
  out of this diff's scope. The scratchpad still holds only `coder`/`teco` files. Recover the
  baseline and instruments before U3.
- **P3-2 (Q4 judged against the paging band): not addressed.** This is a plan §7 item; it doesn't
  affect the code.
- **P3-3 (demo `/mcp` clause): fixed in `falkor-chat/AGENTS.md`** by the coordinator; I checked
  the text in the diff. Plan R2 and the §6 row still carry the old clause, which is plan-only.

## Pass 5 (2026-10-22) — U4 nlq-12 fix (code gate)

### Scope & verdict

This pass reviews the uncommitted diff in `falkorchat/{llm,querygen,tools}.py` and
`tests/{test_tools,test_extraction}.py`, judged against U3's Q2 finding
(`falkor-chat/docs/test-reports/salesperson-single-model-report.md`, "The catalog regression,
verbatim").

**Verdict: needs changes.** There is one major finding with a one-line fix. Everything else is
minor.

CPG: considered, not relevant — `cpg_falkorchat` is stale; the diff is two helpers and one call site, read directly.

**What I ran:**
- **The brief's fixture-free command:** 200 passed, 3 deselected. The implementer reports 201;
  I didn't reconcile the one-test difference.
- **A probe script** that calls the real `QueryGraphDataTool` with a stub LLM and stub services.
  It covered 8 edge shapes, and its results are cited below.

**Implementer claims, checked:**
- **Other callers unchanged: confirmed.** The hook defaults to `None`, so `json.loads` keeps its
  normal last-wins behaviour for the guard judge (`app.py`) and `extraction.py`. That is pinned by
  the new `test_extraction` case.
- **Default-`returns` premise: confirmed.** `QueryRequest.returns` is required with `min_length=1`
  and has no default. `querygen.py`'s history has only one commit touching `"returns"`
  (`6fd89417`, the DSL's birth). So there was nothing for the brief's "follow what querygen
  already does" to point at, and adding a default was the implementer's own choice.

**How the parse paths fail (question 2).** A `ValueError` raised by the hook doesn't reject the
whole reply, only the candidate object that triggered it:
- On the whole-text path it makes `_load_json_object` return `None`, and parsing falls through to
  the line scan.
- The line scan re-parses the same object with the hook, gets the error again, and skips it.
  `consumed` is already set, so nested own-line filter objects aren't re-scanned.
- One consequence (probe): when a conflicting object and a valid one each sit on their own line,
  the valid one is accepted. That matches how the line scan already treats any unparseable
  candidate, so it's acceptable.
- A merged payload with two `matches` passes through `default_returns` unchanged and is rejected by
  validation (probe), which is correct.

### New findings

**P5-1 (major) — Identical repeated lists are concatenated, which contradicts the docstring and
regresses some previously working replies.** `merge_duplicate_keys` tests "both lists" before it
tests "identical". So an identical repeat of a list-valued key is concatenated, not "kept once" as
its docstring promises. Probe results:
- An identical repeated `returns` becomes `["p.name", "p.name"]`. `compile()` rejects the
  duplicate expression, and the tool abstains. Before this diff, last-wins answered it.
- An identical repeated `filters` produces `p.category = $p0 AND p.category = $p1`. The results are
  harmless, but the list counts toward `QueryMatch.filters`' `max_length=4`.

So the fix adds a new false "no matching data found", for exactly the model behaviour (repeating
keys) it exists to absorb.

**Fix:** test `merged[key] == value` first (keep the repeat once), then concatenate lists, then
raise. Add two tests: identical repeated `returns` answers, and identical repeated `filters`
compiles to a single condition.

**P5-2 (minor) — The default `returns` also fires on single-fact questions, not only list ones.**
In the probe, "How much does the Wireless Charging Pad cost?" with no `returns` now compiles to
`WHERE p.name = $p0 RETURN DISTINCT p.name`. The agent gets back the product's name with no price,
where it used to get an abstention. That's a fabrication-shaped hand-off, because the agent may
supply a price itself. The docstring's "known limit" covers aggregates but not this case.

Weighed on its merits, the default is still better than abstaining for `nlq-12`'s list shape. The
prompt prescribes `<var>.name` for list questions, and the salesperson agent has
`lookup_product_fact` for prices. So I don't ask for it to be removed.

**Fix:** either skip the default when a filter targets `name` (a single-entity lookup, where a
name-only answer is uninformative), or document this second limit in the docstring. That's the
implementer's call. No current test pins either behaviour, so whichever is chosen needs one.

**P5-3 (minor) — No positive test goes through the line-scan path.** The verbatim `nlq-12` test is
fenced, and `_strip_code_fence` sends it down the whole-text path. That's why mutant A survives
every positive test, and only the conflicting-scalar test (via fallback) catches it. My probe
shows a prose-wrapped, own-line, repeated-`filters` reply does AND correctly today. **Fix:** add
that case as a test. It's cheap, and it reddens under mutant A on its own merits.

**P5-4 (minor) — The bench's NLQ pack no longer mirrors production parsing.** Plan §2.3 says the
`nlq-structured-query` pack mirrors production parsing. After this diff, production merges
repeated keys and fills an absent `returns`. The pack's `tools/exec.py:126-130` still rejects a
missing `returns` and keeps last-wins, so the bench now under-scores production for this failure
class. **Fix:** route a note to the pack's `PROVENANCE.md`, or a pack version bump, via `teco`. No
change in falkor-chat.

**Gate note (not a code finding).** The fix is proven only against the stubbed verbatim completion.
Closing Q2's stakeholder item needs a live ministral rerun of Q2, with `NLQ_EVAL_RESULTS_PATH`, to
confirm two things: `nlq-12` now passes, and no other pair changes (21 catalog pairs; the
default-`returns` change can move single-fact pairs per P5-2).

**Docs (question 4): the claim holds for living documents.**
- `docs/QUERIES.md:2201` mentions `extract_own_line_json_object`, but only for extraction, whose
  behaviour is unchanged.
- No manual, `SERVER.md` or `DESIGN.md` section describes this parse.
- The K-055 plan (`docs/plans/workflow-nl-query-generation.md`) is `archived` and stays frozen.
- `falkor-chat/AGENTS.md` already says parse tolerance is documented in the `llm.py` docstrings,
  and the diff updates those.

### Dispositions of Pass 4 findings

- **Results-path parent check (nit): not fixed.** It's outside this diff.
- **T4–T6 use static-gateway sugar (nit): unchanged**, as accepted.
- **R5 `extraction` kind (pre-existing, `teco`): still open.** It's outside this diff.

## Pass 6 (2026-10-22) — U4 nlq-12 fix, re-gate of the Pass 5 fixes (code gate)

### Scope & verdict

This pass covers the same uncommitted diff as Pass 5, as revised: `falkorchat/{llm,querygen,tools}.py`
and `tests/{test_tools,test_extraction}.py`.

**Verdict: approve with suggestions.** All three Pass 5 findings in scope are closed, and each
fix is pinned by a test that goes red under a targeted mutant. One new minor finding (P6-1) is
the same class as P5-1, a shape the "identical" check doesn't reach. Its fix is one line and I
have run it, so I recommend landing it in this revision. It doesn't block.

CPG: considered, not relevant — `cpg_falkorchat` is stale; the diff is two helpers and one call site, read directly.

**What I ran:**
- **The brief's fixture-free command:** 207 passed, 3 deselected. This matches the implementer's
  count.
- **Mutants.** Each was copied aside and restored, checked with `cmp`. Afterwards
  `git diff --stat` still showed the same 5 files, +361/−10.
  - A, the line-scan call drops `object_pairs_hook`: 2 red, the prose-wrapped test and the
    conflicting-scalar test.
  - B, P5-1's order reverted (concatenate before the identical check): 2 red, both
    identical-repeat tests.
  - C, `nameNormalized` removed from the skip set: 1 red, the parametrized `[nameNormalized]`
    case.
  - Teco's `value`-for-`property` mutant was already red (per the brief).
- **A probe of the interactions.** It drives the real `QueryGraphDataTool` with stub LLM and
  services over 12 shapes. The script is in the session scratchpad (`probe6.py`), and its results
  are in the Appendix table below.

### Dispositions of Pass 5 findings

- **P5-1 (major): fixed.** `merge_duplicate_keys` checks `merged[key] == value` before the
  list concatenation. Identical repeated `returns` answers, and identical repeated `filters`
  compiles to one condition. Both are tested, and both tests go red under mutant B.
- **P5-2 (minor): fixed**, using the skip option. A filter on `name`/`nameNormalized` leaves
  `returns` absent, so the tool abstains. Both properties are tested via parametrize, and the
  `nameNormalized` case goes red under mutant C.
- **P5-3 (minor): fixed.** The prose-wrapped own-line test can only parse through the line scan.
  It goes red under mutant A on its own merits.
- **P5-4 (minor): deferred** to the backlog by `teco`, per the brief. I didn't recheck it.
- **Pass 4 carry-overs:** unchanged and outside this diff.

### New findings

**P6-1 (minor) — Lists that are repeated with overlapping content are still concatenated with
duplicates, and the docstring says otherwise.** The identical check only catches a repeat that is
equal as a whole list. Probe results:
- `"returns": ["p.name","p.price"]` repeated reordered, or `["p.name"]` then `["p.name","p.price"]`:
  the tool **abstains**. `compile()`'s duplicate-expression guard (`querygen.py:459`) rejects the
  result. Before this diff, last-wins answered both. This is the regression P5-1 described, for the
  next shape along the same axis.
- `filters` as A, B, A, or `[A,B]` then `[B,A]`: the tool answers with redundant `AND` conditions.
  These use up `max_length=4`, so with three or more conditions the tool would abstain.
- The docstring's rule 2 says "two different `returns` lists simply project both". That is false
  when the lists overlap.

**Fix:** remove duplicates element by element during the concatenation:
`merged[key] = merged[key] + [v for v in value if v not in merged[key]]`. I ran this. It closes
every overlap row in the probe, leaves the abstain rows abstaining, and keeps the 207 tests green.
That last point means no existing test constrains it. So add tests along the axis rather than per
shape: one parametrized test over {identical, reordered, partial overlap, disjoint} × {`returns`,
`filters`}, asserting the compiled Cypher. With dedup in place, the identical branch only matters
for scalars, so update rule 1 and rule 2 of the docstring to say so. For `filters`, dedup is
semantically exact, because `X AND X` is `X`.

**P6-2 (nit) — The name skip is not too broad, and it can't collide with `CONTAINS`.** The
brief's question 3 asked whether it could. `QueryFilter.op` is a closed six-op whitelist with no
`CONTAINS` (`querygen.py:81-83`), so no substring-on-name list question can be expressed at all.
- The only list-shaped name filter the DSL can express is `<>` ("everything except X"). In the
  probe it abstains when `returns` is omitted.
- Narrowing the skip to `op == "="` would recover that case. It's take-or-leave, because the
  shape hasn't been observed.
- A merged `filters` list that contains a name condition (category plus name) also skips and
  abstains. That's correct, because it's still a single-entity lookup.

**P6-3 (nit) — The class docstring in `tools.py` abbreviates the skip as "a filter on
`name`".** It omits `nameNormalized`. `querygen.default_returns`' own docstring is the accurate
one, so either name both properties or point to it.

### Repeated different `matches` abstaining (question asked by the brief)

**Judged right.** Two different match objects name two queried entities. Last-wins would
silently answer about the second one, the same hazard the conflicting-scalar rule refuses.
Abstaining is the conservative choice and consistent with that rule.
- One recoverable variant is lost. When both repeats share `var`/`label` and differ only in
  `filters`, the probe shows it abstains, although the intent is plausibly AND, the same as
  `nlq-12` one level up.
- That shape hasn't been observed, so I don't recommend handling it now. Revisit only if the live
  Q2 rerun or a later golden run shows it.

### Docstrings vs behaviour (question 4)

- **`llm.extract_own_line_json_object`: accurate.** It describes the hook and says a
  `ValueError` rejects that object.
- **`querygen.default_returns`: accurate**, including the name-lookup exception and the
  aggregate limit.
- **`querygen.merge_duplicate_keys`: accurate except rule 2's overlap claim** (P6-1).
- **`tools.QueryGraphDataTool`: accurate except the abbreviation** in P6-3.

### Appendix — Pass 6 interaction probe (current tree)

| Shape | Result | Compiled `WHERE`/`RETURN` |
|---|---|---|
| `filters` A, B, A | answer | `cat AND price AND cat` (redundant) |
| `filters` [A,B] then [A] | answer | `cat AND price AND cat` (redundant) |
| `filters` [A,B] then [B,A] | answer | 4 conditions, 2 redundant |
| `returns` reordered repeat | **abstain** | — (duplicate-expression guard) |
| `returns` partial overlap | **abstain** | — (duplicate-expression guard) |
| `returns` disjoint | answer | `p.name, p.price` |
| merged `filters` cat + name, no `returns` | abstain | — (name skip, correct) |
| merged `filters` cat + price, no `returns` | answer | `cat AND price`, `RETURN p.name` |
| `name <> X`, no `returns` | abstain | — (P6-2) |
| same `var`/`label` `matches` × 2, different `filters` | abstain | — (one-match limit) |
| `price < 200` vs `< 200.0` repeat | answer | single condition (`200 == 200.0`) |
| repeated `op` with a different value inside one filter object | abstain | — (conflicting scalar) |

With the P6-1 dedup applied (trial, then restored), the first five rows compile without
duplicates and answer. Every other row is unchanged.
