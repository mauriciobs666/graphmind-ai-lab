# Backlog — falkor-chat

> **Status:** active · **Owner:** `teco` · **Tracks:** K-016…K-081

> **How to read this.** Forward-looking only — what is proposed but unbuilt. *When* something
> changed and *what* it involved live in [`HISTORY.md`](./HISTORY.md), one dated entry per
> delivered item — **a delivered item is not kept here at all, not even as an index row**
> (root `AGENTS.md`). Design lives in [`DESIGN.md`](./DESIGN.md), whose §1 is the standing
> decision register; the query library in [`QUERIES.md`](./QUERIES.md). Completed plan
> documents stay where they are, marked `Status: archived` (root `AGENTS.md`); `archive/` holds
> frozen documents from the previous convention.
>
> **Status of an *open* milestone is authoritative here.** A closed one's record is
> `HISTORY.md`; DESIGN §12's roadmap is the shape of the work, not a status board.
>
> Status markers: 🔵 proposed · 🟡 in-progress · ✅ delivered (→ `HISTORY.md`) · ⚪ rejected/deferred.
> Item IDs keep the `K-` prefix from the former `kaizen/plan.md`.

## Active

Nothing currently active — **M6 (business entities in workflows) closed 2026-08-30**: all four
sibling capabilities (K-052 catalog lookup, K-053 cart/orders, K-054 durable profile, K-055
natural-language query generation) delivered and proven live together inside the combined
`salesperson@v4` demo agent; see `HISTORY.md` for the five closing entries and
`docs/plans/workflow-salesperson-demo-coordination.md` (archived) for the full multi-session trail.

Everything open is a follow-up filed out of a closed milestone (`## Open follow-ups`) or the
deferred M2.5 hardening track.

## Milestones still open

| Milestone | Reaches ✅ when | Items |
|---|---|---|
| **M2.5 — Hardening** ⚪ *(deferred)* | Real auth, transport-level agent path, real-time push | K-016 → K-017, K-018 |

Follow-ups filed out of a closed milestone are **not** green-gates for it; they are listed under
`## Open follow-ups`.

## Open follow-ups

Each was filed out of a closed milestone's gates or a later investigation; none gates M5.

### K-066 — Demo bring-up does not check LM Studio model residency before serving the storefront (🔵 proposed — single-model set adopted and live-measured 2026-10-22, `docs/test-reports/salesperson-single-model-report.md`)

> **What is true now.** The storefront's dead-turn rate is set by whether LM Studio keeps the demo's
> models resident. JIT swapping gave ~63% dead turns (`docs/test-reports/salesperson-language-salience2-report.md`).
> Since `ad01aee7`, `query_graph_data`'s internal call uses the calling step's model
> (`docs/plans/salesperson-single-model.md`), so the demo needs **one chat LLM plus the embedder**:
> `mistralai/ministral-3-3b` (`-c 8192 --gpu max`) and `text-embedding-qwen3-embedding-0.6b`, with
> `qwen` unloaded and the `:8200` server idle (it can JIT-load `qwen`). The demo set-up is
> `falkor-chat/AGENTS.md`. With that set pinned (n=75, 2026-10-22), there were **0/1,125 dead turns**,
> 0 context overflows, 0 wrong-language replies and a p95 of 14.1 s, against 6/1,125 and 24.1 s with
> three models. On the 6 GB GPU it fits only with WDDM paging: ≤1.6 s on the first embed after a
> chat (`docs/test-reports/salesperson-single-model-fit-report.md`). The storefront never embeds.
>
> **What remains.** (1) A preflight in demo bring-up that reads `/api/v0/models` and refuses to
> serve unless exactly the two-model set is loaded. Today the guidance is prose only. (2) Optional:
> a long-conversation stress variant of the probe. Q3 saw 0 overflows, but U0's synthetic three
> concurrent ~3.1k-token requests still exceed the shared 8,192-token KV budget. (3) Optional: a clean
> rerun of the `TURN_WORKERS` lever (Q4 was contaminated by a host suspend; keep 4).
- **Owner:** `devops` (1); `qa-engineer` (2), (3).
- **Test strategy:** TP-007 protocol with `salesperson/scripts/language_salience_probe.py`; residency
  via `salesperson/scripts/qwen_watch.sh` and `lms ps`.

### K-078 — es/pt-BR carts end up empty: the agent translates product names, so `add_to_cart` is held by the off-turn guard (🔵 proposed — `docs/test-reports/salesperson-single-model-report.md` Q3, 2026-10-22)

> In the n=75 run, the demo log held 146 `off-turn write held … add_to_cart(productName=…)`
> warnings, every one with a Portuguese or Spanish rendering of an English catalog name (e.g.
> `Mouse sem fio Pro` 70, `Ratón inalámbrico profesional` 23). The K-058 guard requires the product
> name to appear in the turn's text, and the catalog lookup needs the canonical name, so non-English
> carts mostly stay empty. Agent behaviour that predates K-066 (ministral was already the agent model).
> Serious for an es/pt-BR demo.
- **Owner:** `architect` (prompt vs. canonical-name resolution vs. guard matching), then an implementer.

### K-079 — Storefront replies sometimes state fabricated or scrambled prices (🔵 proposed — `docs/test-reports/salesperson-single-model-report.md` Q3, 2026-10-22)

> Two `en` replies to "What products do you have?" (#156, #286) listed invented prices, e.g. Wireless
> Mouse Pro at $349.99, and some replies quote € instead of $. Agent behaviour, not attributed to
> K-066. `query_graph_data` now abstains rather than return a bare name for a single-item lookup
> with no projection (`querygen.default_returns`), which removes one fabrication-shaped hand-off.
- **Owner:** `analyst` (RCA: prompt vs. tool-result rendering), then an implementer.

### K-080 — Model kind `extraction` sits outside `modelconfig.KINDS`, so a workspace override can't reach it (🔵 proposed — `docs/reviews/salesperson-single-model.md` Pass 4 R5, 2026-10-22)

> `IngestionPipeline` resolves kind `extraction`, which is absent from `modelconfig.KINDS` and
> `_KIND_TO_OVERRIDE_KEY`. `SERVER.md` §1.8's "four closed consumer kinds" is therefore false, and
> `agentModelOverride`-style workspace overrides do not apply to extraction. Pre-existing.
- **Owner:** `architect` (add the kind vs. document it as deliberate), then `tdd-engineer`.

### K-081 — NLQ: ministral answers an entity-type question with the `Product` label (`nlq-27`) (🔵 proposed — `docs/test-reports/salesperson-single-model-report.md` Q2, 2026-10-22)

> "Can you list all the Product-type entities?" (knowledge_base) should match `Entity {type:'Product'}`;
> ministral emits `label: "Product"`, gets the catalog, and finds nothing. qwen passes it, so the
> knowledge_base subset is 12/19 vs 13/19. Catalog is 21/21 on both. Not on the storefront path
> (the shop asks catalog questions only). A tool-prompt fix would follow R3 in
> `docs/plans/salesperson-single-model.md`.
- **Owner:** `data-scientist` (whether to fix by prompt), then an implementer.

### K-077 — Language-salience probe harness: classifier false negatives, a blind trial-level table, and untested `run_participant` (🔵 proposed — `docs/reviews/salesperson-language-salience.md` Pass 2, F1–F7, 2026-10-20)

> **Why it exists.** `salesperson/scripts/language_salience_probe.py`, the committed TP-007 harness,
> was reviewed before the 2026-10-20 rerun and approved with suggestions. Its fixes were deferred so
> they would not touch the live window. They gate nothing: no 2026-10-20 figure depended on them,
> because every reply was hand-read and every F4 trigger counted zero
> (`docs/test-reports/salesperson-language-salience2-report.md`, Feedback item 5). The next run of
> the protocol would rely on them, though. The open findings: **F1** the v2 lexicon has false
> negatives in the dangerous direction (`con`/`en`/`una` vote for nobody; `in` matches the catalog
> name `USB-C Hub 7-in-1`, so a Spanish reply can score `en`; `do` is in both `EN_WORDS` and
> `PT_WORDS`). **F2** the trial-level table ignores `unknown` replies. **F3** `run_participant` has
> no coverage, and 7 of 9 planted mutants survived. **F4** abnormal paths (timeout, poll/read error,
> transport `-1`) misattribute a reply to the next turn, and `aggregate` hides them. **F5** scoring
> the joined turn misses code-switching. **F6** two exceptions escape `http_json` and abort the
> run. **F7** dead lexicon entries, and `--resume` ends one trial short.
- **Owner:** `tdd-engineer` (failing test first).
- **Scope:** the review's suggested fix for each finding (Pass 2, "Findings"). That includes F1's
  coverage-probe corpus with a `CLASSIFIER_VERSION` bump to `v3`, rather than a lexicon edit alone,
  and a fake storefront on stdlib `http.server` that kills every F3 survivor. Re-scoring the stored
  2026-10-20 JSONL with `v3` is optional; it cannot change that run's hand-read headline.
- **Risks/RAM:** none. The harness is an offline-testable dev script with no runtime consumer. A
  `v3` classifier is not comparable with the v2 figures, so report both on the next run.
- **Test strategy:** the 21 offline tests stay green. Add the fake-storefront tests for
  `run_participant`, the F1 coverage probe, and aggregate fixtures with an `es` participant, an
  `unknown` reply, a multi-row reply and a 409/timeout record. Re-run the review's mutants
  (Appendix B) to confirm they are killed.

### K-063 — `SERVER.md` §1.5 "Layout (as built, M1)" is an M1 snapshot presented as current (🔵 proposed — filed out of the salesperson-ui S7→S8g documentation unit, 2026-09-07)

> **Why it exists.** §1.5 lists 8 of the package's modules and omits `storefront.py` and
> `storefront_api.py` along with everything else added since M1. Its title says "as built, M1",
> but it sits in a section a reader reaches looking for the current layout, so the qualifier does
> not do the work it looks like it does.
>
> **The decision is which of two things it should be**, and that is the whole item: *refresh §1.5
> to the current module set* — and accept that it must then be maintained on every module
> addition — *or retitle it explicitly as a frozen M1 snapshot* and point to whatever should be
> read instead. It is not obviously the former: a layout list is exactly the kind of prose that
> goes stale silently, and the repo's own convention (root `AGENTS.md`) is that history belongs
> in `HISTORY.md` rather than in a living document.
>
> **Why it was not fixed in place.** `docs/plans/salesperson-ui.md` §5.0 rules §1.5 out of scope
> for every step of that plan and says this item is to be filed instead. It had not been —
> `teco`'s own action item, missed at the time and caught by the S7→S8g documentation unit.
>
> Two smaller neighbours worth folding in when this is picked up, both found the same way:
> **§1.2's layering diagram** (`api → services → repository → db`) has no storefront layer, and
> **§1.6 "TDD build order"** is an M1 checklist with ✅ marks and a "when this code lands, update
> `AGENTS.md`" note — frozen history sitting in a living document.

### K-060 — `salesperson@v5` sometimes silently drops a genuine match when `filter_products` returns a mixed-category result (🟡 in-progress — root-caused at n=75, low base rate, no fix warranted yet, 2026-08-31)

> **Why it exists.** Live-verifying K-057's wording fix (`docs/HISTORY.md` 2026-08-31,
> `docs/reviews/salesperson-tool-reliability-ml.md` §11) at n=20 found every one of the 4/20
> remaining wrong replies traced to a third mechanism, distinct from both §11 diagnosed: when
> `filter_products` is called with no `category` argument (a mixed-category result set), the model
> sometimes silently drops a genuinely-matching item from its synthesized reply — not a rounding
> error, not a self-contradiction, a synthesis-time omission on an unfiltered result set. A
> larger, isolated diagnosis pass (`ml.md` §14, n=75 across 4 conditions) confirms the mechanism is
> real and reproduces exactly as filed, but finds it **rarer and more self-mitigated than the
> original n=4/20 disclosure implied**: the model spontaneously self-corrects via a second,
> category-scoped `filter_products` call in 66.7% of single-`category`-omitting reps, and that
> self-correction is 100% protective; within the true at-risk population (the ~1-in-3 reps with no
> self-correction), the rate is 10.0% (pooled with the original disclosure sample, 5/50, Wilson 95%
> CI 4.3-21.4% — down from the original 20% point estimate, though the two samples' CIs still
> overlap, most plausibly because `mistralai/ministral-3-3b` — the pinned `assistant`-step model —
> has no `temperature` entry in `config/models.json`, unlike `qwen/qwen3-4b-2507`'s pinned `0`, so
> exact reproducibility run-to-run isn't expected). Item-level mechanism (position/category-
> diversity/list-length) remains genuinely unclear at this n — the one fresh confirmed occurrence
> dropped the cheapest, first-by-price item, a hypothesis worth a future pass's attention, not yet
> a finding.
- **Two independent, reasonable-looking wording attempts already failed to fix this before the
  §14 diagnosis pass — still worth weighing before a third.** A `category`-parameter nudge
  (encouraging the model to always pass `category` when the customer names one) plus a
  `systemPrompt` synthesis-time "check every returned item, never drop a match" safety net were
  live-tested together at n=20: the model still never passed `category` (0/20) and net wrong-reply
  rate went **up** (30% vs. the shipped fix's own 20%), while also suppressing the same
  self-correction pattern §14 later found to be 100% protective on its own. Reverted, never
  shipped.
- **A payload-restructuring lever (an explicit per-item index/flag, or a restructured JSON shape,
  as an alternative to wording) is plausible but not live-tested — the measured rate is too low
  and too compounded (33% opportunity × 10% defect) to power a live A/B affordably through the
  full conversation harness (would need ~150-200 conversations per variant, `ml.md` §14.6).** A
  cheaper design is specified for whoever picks this up next: a **fixed-context, synthesis-only**
  comparison that replays a captured single-call, no-category tool result and drives only the
  final completion twice (shipped payload vs. a restructured candidate) — ~3-5x cheaper per rep,
  affordable at a well-powered n=40-60/arm, but its absolute rate is a payload-shape **delta**
  only, not a substitute for the full-conversation rate (no tool-selection/self-correction
  variance).
- **Owner:** no fix owner yet — `data-scientist` next if a further rate-tightening pass or the
  §14.6 synthesis-only payload comparison is picked up; `tdd-engineer`/`coder` only once a fix
  shape is actually chosen. Per `ml.md` §14.7: acting on a fix now (wording or payload) would
  repeat this thread's own already-learned lesson (guessing a fix shape ahead of a stable rate
  estimate) — the rate just narrowed materially and a third wording attempt is explicitly not
  recommended.
- **Risks/RAM:** none — diagnosis only so far, no graph/schema surface.
- **Test strategy:** if picked up again, the cheapest next step is **not** another full-conversation
  live pass at the isolated condition — `ml.md` §14.6's fixed-context synthesis-only harness design
  (replays a captured tool result, drives only the final completion) is both cheaper per rep and
  the right seam for a payload-shape A/B, should one be attempted. A further n≈30-40 full-conversation
  pass at Condition A's exact prompt would also narrow the pooled CI (currently 4.3-21.4%) if a
  payload-agnostic rate estimate is wanted first.
- **Closed, not open — no further action needed:** the `analyst`-flagged `minPrice` "more than $X"
  regression gap (`docs/reviews/salesperson-tool-reliability-impl3.md`) is now confirmed live at
  100% (11/11, CI 74.1-100%, `ml.md` §14.5) — the shipped K-057 fix's inclusive-bound guidance
  holds symmetrically in both price directions.

### K-062 — `salesperson@v5` sometimes states the wrong reason for a correctly-held write call, even though the correct reason was handed back to it (🟡 in-progress — both candidate fix levers evaluated live and rejected (one no-effect, one actively harmful); no fix shipped, closed pending only a re-worded lever (a), 2026-08-31)

> **Why it exists.** Originally found at 2/24 (8.3%, Wilson CI 2.3-25.8%) in K-061's diagnosis
> pass: the model telling the customer an unrelated `add_to_cart` call was held because the
> product "was not recognized as a product" / "not recognized in the catalog" — factually wrong;
> the product is real and correctly catalogued, and K-058's guard's own held-call result
> (`executor.py:1030-1045`) states the *actual* reason verbatim back to the model ("was not
> mentioned anywhere in this turn's own message"). The model had the accurate explanation
> available and substituted a plausible-sounding wrong one. Distinct from K-061 (which is about
> the cart *state* being wrong); here the cart state is correct throughout and only the
> customer-facing explanation is wrong.
- **Rate history across four independent samples of the identical 3-turn script** (all
  `ml.md`, every candidate reply read in full, never regex-matched): 2/24 (8.3%, original
  disclosure) → 8/25 (32.0%, §15.4 re-screen) → round-5 dedicated diagnosis (§16, n=28): **strict**
  1/28 (3.6%, CI 0.6-17.7%), **broader** 5/28 (17.9%, CI 7.9-35.6%) → lever-eval's own fresh
  baseline arm (§18.3, n=28, run after the `temperature: 0` pin below): strict 0/28 (0.0%),
  broader 5/28 (17.9%) — a fourth independent sample landing on the *same* broader point estimate
  as the third by coincidence of sampling, not by design (same n, same taxonomy — the CI is
  exactly as wide). **`teco` reads the strict/broader split as the more load-bearing fact than any
  single point estimate**: no sample in this whole thread has ever produced the original exact
  filed phrasing at meaningful volume (0/28 in all three of §16's, and §18's own three arms'
  strict counts) — the true, currently-actionable rate is the broader one, high-single-digits to
  high-teens, still too wide to pin further without a fifth dedicated pass, which is not currently
  recommended (see Owner below).
- **A definitional gap contributes to the spread, not fully resolved:** strict vs. broader differ
  2-5x on the same samples. §15.4's own re-screen named its screen as "the §12.5 pattern" without
  re-stating the phrasing threshold, so which definition the 8.3%/32.0% figures actually used
  isn't fully recoverable. **This entry tracks both, explicitly, going forward.**
- **The precondition-occurrence rate (a K-058 hold firing on the held product at all) swings
  large and unpredictably, session-to-session *and* arm-to-arm within one pinned-temperature
  session** — this is now a load-bearing caution for reading any rate in this script family, not
  just K-062's own. Five independent measurements: §12.6 (unpinned) 14/24 (58.3%); §16.3
  (unpinned) 27/28 (96.4%); §18's own three arms, all under the **identical** pinned
  temperature/model/script, differing only in `systemPrompt` text: baseline 15/28 (53.6%), lever
  (a) 5/28 (17.9%), lever (b) 17/28 (60.7%) — baseline vs. lever (a) alone is non-overlapping
  (Fisher exact p≈0.011) despite sharing every variable except prompt wording. **The `temperature:
  0` pin (shipped this round, see below) did not narrow this swing** — §18.7 leaves open whether
  the driver is backend-level float nondeterminism surviving the pin, or a genuine causal effect
  of prompt length/content on turn-3 tool-call behavior; neither is resolved. Practical
  consequence for any future pass in this family: an occurrence-rate difference between two
  sessions or two prompt variants cannot by itself be read as evidence about the variable under
  test without a same-session, same-config comparison at higher n than 28, and even then only
  cautiously — this is why every rate in this entry is reported **conditional on occurrence**
  where that framing is available, not just unconditionally.
- **Mechanism lead — re-checking `view_cart` after a hold correlates with a correct reply —
  replicated a second time, independently, and still only correlational.** Round-5's original
  finding (§16.4, one-sided Fisher p≈0.010, 0/13 vs. 6/14) reproduced cleanly in the lever eval's
  own fresh baseline arm (§18.3, Fisher p≈0.0070, 0/8 defects when the model rechecked vs. 5/7 when
  it didn't) — two independent samples, same association, still not causally tested.
- **Both named candidate levers were evaluated live (`ml.md` §18, U2, n=28/arm, 84 replies read in
  full) and rejected — ship neither:**
  - **Lever (a) (`view_cart`-recheck nudge):** no detectable improvement in the
    conditional-on-occurrence broader-defect rate (40.0%, n=5, vs. baseline's 33.3%, n=15 — CIs
    overlap, Fisher p≈0.42 on the unconditional table). Its own mutation-tested compliance rate —
    whether the model actually rechecks `view_cart` after being told to — is weak: 40.0% (2/5).
    The eval is under-powered on this arm specifically (its own precondition-occurrence rate came
    in far lower than baseline's, 17.9% vs. 53.6%, itself part of the swing finding above), not
    flatly negative — a stronger, more directive wording is the one live option left (see Owner).
  - **Lever (b) (`systemPrompt` line naming the scenario):** actively **worse**. Conditional
    broader-defect rate rises from baseline's 33.3% to **88.2%** (n=17, Fisher p≈0.0028 on a
    same-occurrence-rate — therefore unconfounded — comparison). The added line supplies the model
    a correct-reason template it sometimes applies to the *wrong item pair* (4/15 defects: a novel
    item-swap confabulation not seen in baseline or lever (a)), and even where the reason-sentence
    lands correctly, the itemized cart-contents list immediately following it still omits the held
    item in most cases (10/15 defects) — getting the reason right did not fix the list. **Lever
    (b)'s general approach (name the scenario in prose) should not be retried with different
    wording** — the failure mode is the template-misapplication mechanism, not a wording accident.
- **The reproducibility fix is shipped, but its effect turned out narrower than expected:**
  `config/models.json` pins `temperature: 0` for `lmstudio/mistralai/ministral-3-3b`, mirroring
  the existing `qwen/qwen3-4b-2507` entry — confirmed via `ResolvedModel.params` resolution and
  204 green offline tests. It does **not**, on its own, reliably pin the precondition-occurrence
  rate (see the swing finding above) — narrower payoff than this entry originally expected, though
  still worth keeping (it removes temperature-sampling as *a* variable, even if not the dominant
  one) and still likely relevant to sibling defects in this script family (K-057/K-060/K-061).
- **Owner:** no fix owner right now — closed as "no fix warranted from current evidence," same
  shape as K-060. The only live option, if picked up: a **re-worded lever (a)** naming `view_cart`
  as the *first* required action after any `HELD` event (rather than "before your final reply"),
  with its own fresh targeted eval — not a larger-n rerun of this pass's own wording, and not a
  third wording guess ahead of that eval. Lever (b)'s prose-template approach is explicitly not to
  be revisited without a structurally different intervention.
- **Risks/RAM:** none — diagnosis and eval only, no shipped behavior change beyond the config pin.
- **Test strategy:** if a re-worded lever (a) is picked up, a targeted eval isolating its own
  effect on the `HELD`-then-reply path (both the unconditional and, more importantly, the
  conditional-on-occurrence rate), scored under both the strict and broader definitions above,
  before shipping — `tdd-engineer`/`coder` → `analyst` gating applies to the implementation once a
  lever earns a positive eval, not before.

### K-029 — Converge the seed def sources into `proof_defs.py` (+ the symmetric `decision` publish invariant) (🔵 proposed — filed out of K-024, open item O-5 / gate m-9 / nit n-3)

> **Why it exists.** The two seeded defs use **two different source conventions**, deliberately for
> the K-024 slice: `access-request@v1`'s spec is imported from `server/falkorchat/proof_defs.py`
> (so the seed script and the offline acceptance test provably cannot drift), while **`triage@v1`'s
> literal is still inline in `scripts/seed_workflows.sh`**. Moving `triage`'s def *during* K-024 was
> declined with a reason, since corrected by **K-034**: at the time, published defs were believed
> **create-only** (`MERGE … ON CREATE SET`) end to end, so a byte-diff introduced while relocating a
> **live** def was assumed silently swallowed. As of K-034 that is only true for a **property**-only
> byte-diff (e.g. a `config` field reformatted during the move) — it still silently no-ops, so K-029's
> planned before/after equality check remains load-bearing for that half. A **topology**-changing
> byte-diff (e.g. a retargeted transition introduced while relocating the literal) is now **rejected**
> (`409 WorkflowDefConflictError`, nothing written) rather than swallowed — safer, but a `409` mid-
> deploy is still worse than catching it in a pre-flight check, so the equality check stays this item's
> load-bearing safeguard either way. `reference`/`ws:<id>` can still go stale independently whenever
> one side was never re-published/re-materialized at all. That is a split-brain risk to take on its
> own, with its own verification, not as a rider on a feature slice.
- **Owner:** **`coder`**, with an explicit before/after equality check on the published def subgraph
  (not just "the script ran").
- **Scope:** (1) move `triage@v1`'s inline literal into `proof_defs.py` beside `ACCESS_REQUEST_DEF`,
  leaving `seed_workflows.sh` a pure driver over the service layer for **both** defs; (2) prove the
  move is byte-identical *in the graph*, which given create-only semantics means either verifying
  against a freshly published `reference` or bumping `triage`'s version in lockstep with
  `config.TRIGGER_DEF_KEY`/`TRIGGER_DEF_VERSION` (note `start_server.sh` neither forwards nor exports
  those two vars today — a version bump also needs a script change); (3) fold the `n-A` warning
  (`ACCESS_REQUEST_DEF`'s key set **is** `publish_workflow_def`'s keyword signature) into whatever
  shape both defs end up sharing.
- **Also carries nit n-3 — the symmetric `decision` publish invariant.** K-024 enforces
  "a `human`/`wait` step must declare `config.waitsForHuman: true`" at publish, but **not** its
  mirror: **a `decision` step whose outgoing transitions are *all* conditional and which does not
  declare `waitsForHuman` self-loops until the step budget fails the run.** It is documented as a
  warning in `falkor-chat/AGENTS.md` and deliberately left unenforced because the symmetric check
  would **retro-reject existing test fixtures** (`server/tests/test_services.py`) — the same
  blast-radius problem B-2 caused in K-024, which is precisely why it belongs in an item that can
  budget for the fixture edits.
- **Risks:** touching a live published def is the risk; there is no new graph surface and no RAM cost.
- **Test strategy:** a test that both defs come from importable constants; a publish-equality check
  over the def subgraph; if n-3 is implemented, one ordering pin (it must run **last**, like the other
  three invariants) plus the fixture edits it forces.

### K-030 — Allow zero-transition (single-step) workflow defs; guard the `UNWIND` instead of rejecting (🔵 proposed — filed out of K-024 re-gate findings r-1/r-2)

> **Why it exists.** `repository._PUBLISH_CYPHER` ends in a bare `UNWIND $transitions AS tr …
> RETURN …`. With `$transitions = []` the row stream **collapses** — after the `WorkflowDef`, its
> `Step`s and the `START` edge have already been written — so the caller's `res.result_set[0]` raises
> `IndexError` on a **partially written** def. Because publish is `MERGE … ON CREATE SET`, retrying
> the corrected spec on the same `(key, version)` is a **silent no-op on the half-written def**: the
> version is permanently wrong and cannot be repaired by re-publishing. This is the same empty-`UNWIND`
> class that `AGENTS.md` documents as *guarded* for the §4 mention write-block; this path was not
> guarded. K-024 U4b **closed the reachable route** with a `_validate_def_spec` rule (running last)
> that rejects a transition-less spec **before any repository call** — prevention, not a nicer
> exception.
- **What is still open (re-gate r-1):** the fix is **publish-only**. `services.materialize_def` →
  `repository.materialize_snapshot` (`repository.py:1397`) **reuses the same query shape** and performs
  **no** spec validation, so a def poisoned before U4b — or any zero-transition subgraph read back by
  `read_def_subgraph`, which returns `transitions: []` rather than `None` — is still an unguarded
  `IndexError`/500 on materialize. Low likelihood (materialize is fed by publish, now guarded), but the
  guard is **asymmetric**, and the docs/tests currently imply it isn't:
  `server/tests/test_services.py:916` seeds a `FakeRepo` def with `"transitions": []` and asserts
  materialize **succeeds** — true of the fake, and exactly the shape the real query rejects.
- **Accepted limitation to remove (re-gate r-2):** the U4b rule also **rejects a legitimate shape** —
  a genuine single-step def. All four doc sites state the workaround ("a terminal outcome is a step
  with no *outgoing* transition, never a def with none") but none records it as **debt**. Without a
  `K-` number the next person needing a one-step def will either fight the rule or bypass validation.
- **The known cheap remedy:** guard the trailing `UNWIND` in `_PUBLISH_CYPHER` (and therefore
  `materialize_snapshot`, which reuses it) with the **§4 empty-`UNWIND` `CASE` pattern** this codebase
  already relies on and documents as load-bearing — `UNWIND (CASE WHEN $transitions = [] THEN [null]
  ELSE $transitions END) AS tr` with a `FOREACH` that never filters — then **relax**
  `_validate_def_spec`'s rule, and drop the `transitions=[]` mitigation comments in `proof_defs.py`
  and `tests/test_process_input.py`.
- **Owner:** **`graph-dba`** (the query change needs a gate + a re-PROFILE: the guard must not turn the
  index-anchored publish plan into a scan) → **`coder`** for the service-layer relaxation and the
  fixture/doc cleanup.
- **Risks/RAM:** none — no new node, index or property; a query-shape change only. The risk is plan
  regression, which is what the re-PROFILE is for.
- **Test strategy:** a publish and a materialize of a genuine single-step, zero-transition def, both
  asserted to succeed *and* to leave a complete subgraph (steps + `START` + the returned row); the
  existing ordering pins for the other publish invariants must stay green.

### K-032 — Materialize the workflow def's **data-dependence overlay** (CPG-style READS/WRITES) for publish-time static analysis (🔵 proposed — from a design conversation, 2026-07-22)

> **The framing (Code Property Graph lens).** The def graph is already a control-flow graph:
> `(:Step)-[:TRANSITION {guard, order}]->(:Step)` is a guarded CFG, `HAS_STEP` is one-level AST
> containment, and `(:StepRun)-[:NEXT]->` is an executed CFG path. **What's missing is the
> data-dependence layer (DDG):** which `ctx`/`output` keys each step *reads* (via its `cmp`/`llm`
> guards and, for a `decision`/`human`, the keys it branches on) and which it *writes* (a `human`/
> `wait` step's `config.expects`, a step's declared outputs). That information is **not missing —
> it's trapped inside the opaque `guard`/`config` strings**, and `services._validate_def_spec` +
> `guards.validate_cmp` already walk the `cmp` guard tree at publish, so ~90% of the extraction pass
> exists and is currently thrown away. Materialize it as real edges and three otherwise-impossible
> checks become one-hop Cypher.
- **Why it's worth doing (the payoff).** Publish-time (not live-run) detection of:
  1. **Dangling read** — a guard reads a `ctx`/`output` key no upstream step writes. This is exactly
     the **un-enforced n-3 hazard** AGENTS.md documents (a `decision` step with all-conditional
     outgoing transitions and no `waitsForHuman` **self-loops to budget exhaustion**) — today a live
     discovery, turned into a `WorkflowDefSpecError` at seed time. Overlaps K-029's symmetric-invariant
     proposal; this is the graph-shaped way to get there.
  2. **Unreachable step / dead branch** — plain CFG reachability from `START`.
  3. **Change-impact / blast radius** — "I changed `submit`'s output shape; which downstream guards
     read it?" This matters **specifically because published defs are topology-immutable (K-034) and
     property-create-only**: a def edit costs a version bump + snapshot republish + a
     `reference`↔`ws:{id}` split-brain risk, so knowing the blast radius *before* the bump has real
     value here.
- **Hard constraints (fall out of locked decisions — non-negotiable in any plan):**
  - **Derive at publish, never parse in Cypher.** Rule 8 (`ctx`/`config`/`guard` opaque, never
    filtered in Cypher) holds *iff* publish is treated as a compile step — `joern-parse` builds
    overlays once, queries traverse edges; same contract. Extraction runs app-side in the publish
    validator; only the resulting edges hit the graph.
  - **Overlay edges built inside `_PUBLISH_CYPHER` and `materialize_snapshot`, same query.** A
    separately-written overlay on a `MERGE … ON CREATE SET` def is a **new split-brain axis** on top
    of the `reference`-vs-snapshot one — and per the K-030 note the materialize path still `IndexError`s
    after a partial write, so the overlay must ride the existing atomic publish, not a follow-up write.
  - **Static-only, on the def — never on `StepRun`.** The def graph is tens of nodes (overlay edges
    are single-digit multiples → RAM non-issue, the inverse of a repo CPG where AST/CFG/REACHING_DEF
    fan-out dominates). The run graph is thousands of nodes and RAM-bound; "why did *this run* branch
    here" is a join through `RAN`, not a second copy of the layer.
  - **Honest `READS_UNKNOWN` for what can't be derived statically** — an `agent`/`llm`-guard node
    whose reads aren't extractable gets an explicit marker (Joern marks indirect calls the same way).
    A **feature**: it says precisely which parts of a flow are analyzable vs. trust-the-model. Do
    **not** attempt a probabilistic DDG — an unsound dependence edge produces confident-wrong impact
    answers, worse than none.
- **Owner:** **`graph-dba`** gates the FalkorDB model first (the overlay labels/edge types, whether a
  `CtxKey`-style node is per-def or shared, indexes) → **`architect`** designs the publish-time
  extraction pass + the three validations → **`coder`**/**`tdd-engineer`**. A CPG-model design note
  would land at `falkor-chat/docs/plans/<slug>-graph.md` (graph-dba convention).
- **Scope sketch (to be designed, not decided here):** first slice = extract read/write sets from
  `cmp` guard paths (`ctx.`/`output.` roots) + `config.expects` at publish → materialize
  `(:Step)-[:READS]->` / `(:Step)-[:WRITES]->` a key node → add the **dangling-read** and
  **unreachable-step** publish validations (closes n-3 the graph way). `llm`-guard reads and the
  change-impact query are follow-on slices. No DDL beyond one node label + its index; no rule-8
  violation; no run-side cost.
- **Relationship to neighbours:** complements **K-031** (that exposes def *structure* for reading;
  this *analyzes* it), and overlaps **K-029**'s symmetric-`decision` invariant (K-029 proposes the
  rule; K-032 proposes the graph mechanism that could enforce it). Not an M3-green gate — M3 is ✅.
- **Risks/RAM:** negligible on the def side (see the static-only constraint). Real risk is *scope
  creep* toward a general expression/data-flow engine — the `expr` seam stays a `NotImplementedError`;
  this rides the existing closed `cmp` family only.
- **Test strategy:** offline contract tests — a published def reads back with the exact READS/WRITES
  overlay for its guards/`expects`; a def with a guard reading an unwritten key is **rejected at
  publish**; an unreachable step is **rejected at publish**; a step whose reads can't be derived
  carries `READS_UNKNOWN` rather than silently claiming zero reads.

### K-033 — Make `maxSteps` an exact cap (`>` → `>=` in `_drive_loop`) (🔵 proposed — filed out of K-031, stakeholder decision OQ-1 "document now, fix later", 2026-07-24)

> **Why it exists.** `maxSteps` does not mean what its name says. `executor._drive_loop` records a
> step and *then* checks `rec["stepCount"] > max_steps` — at `executor.py:410` (OUTCOME A, a guard
> fired) and `:427` (OUTCOME C, a legitimate self-loop). With `maxSteps: 2`: step 1 → `1 > 2` false;
> step 2 → `2 > 2` false; **step 3 runs**, `3 > 2` → fail. So the budget means *"at least N, then one
> more"*, and a run executes at most **`maxSteps + 1`** steps. Confirmed by reading and **pinned by a
> passing test** — `tests/test_executor.py:158`, `assert len(trail) == 4  # maxSteps=3 → the 4th
> advance trips the guard`. Harmless for the two proof defs (8/6/6 steps against `maxSteps: 24`), but
> it makes `maxSteps` unusable as an SLA or a cost budget, which is exactly what a caller reaches for
> it for. K-031 shipped the **documentation** of the real semantics at six sites (DESIGN §6,
> QUERIES §12.5 + the two `$maxSteps` comments, `schemas.py`, `AGENTS.md`'s executor-invariants
> block) per the binding stakeholder decision; this item is the **fix**.
- **The change is two characters, in two places** — `>` → `>=` at `executor.py:410` and `:427`.
  Everything else about this item is ceremony, and the ceremony is the reason it was deferred out of
  an observability slice rather than the difficulty of the edit.
- **Both sites are *inside* the SHA-locked `_drive_loop`** (`71055f756280`, still live on the
  tree — recompute with the DESIGN §6.2 recipe). Landing it therefore costs:
  - a lock break + **re-lock ceremony**: recompute the SHA, then re-quote it everywhere it is
    asserted. Scope it with `grep -rn 71055f756280` rather than a count here — the number only
    grows, and it is now well past forty sites across plans, reviews and test plans;
  - **records that assert the SHA was unchanged during their own work must not be rewritten** —
    every `archived` plan/review that quotes it is a historical claim about its own delivery, true
    when written. The re-lock has to be expressed as *"as of K-033 the lock is `<new>`; earlier
    records quote the pre-K-033 value"* — i.e. the lock stops being a single grep-able constant.
    **Decide that framing before editing**; it is the item's only real design question.
  - **test edits**: `tests/test_executor.py:142-158` (the pinned count 4 → 3, and its explanatory
    comment), plus a sweep of `tests/test_process_flow.py`'s step accounting and the
    `access-request@v1` `maxSteps: 24` headroom;
  - **behavioural blast radius**: every existing run's effective budget shrinks by one.
- **There is no bundling opportunity left — settled 2026-08-25.** This item was filed hoping to
  ride the next change that legitimately broke the `_drive_loop` lock, plausibly K-027 item 2 (the
  terminal-node-must-post engine contract). K-027 closed 2026-08-21 and **item 2 shipped without
  breaking the lock** — it was implemented at `_run_agent_node`, outside it, exactly as this item's
  original scepticism predicted. K-033 now pays the full re-lock ceremony on its own, or the honest
  `maxSteps + 1` prose stays in six documents indefinitely. That is the decision to make; nothing is
  waiting on anything else.
- **Also decide (part of the item, not a separate one):** whether the *park* path (OUTCOME B,
  `executor.py:415-421`) and the terminal path stay deliberately unchecked. They are unchecked today
  by design — a parked run cannot self-drive — and K-031's documentation says so explicitly, so
  changing that is a second semantic decision, not a consequence of `>=`.
- **Owner:** **`tdd-engineer`** (a behaviour change pinned by an existing passing test is the
  test-first shape: flip the assertion red, then the operator) — with an **`architect`** call first
  **only** on the re-lock framing above, if the coordinator wants the archive-document question
  settled before code moves. No `graph-dba` gate: no Cypher, no DDL, no index.
- **Risks/RAM:** **none** — no new node type, label, property, index or vector dimension; no query
  changes. The risk is purely behavioural (every run's budget shrinks by one) and procedural (the
  lock/archive framing).
- **Test strategy:** flip `tests/test_executor.py:142-158` to assert exactly `maxSteps` advances
  (3, not 4) and watch it go red before the fix; add the boundary case at `maxSteps = 1`; assert the
  park path is still **not** budget-checked (a parked run at `stepCount == maxSteps` must stay
  `waiting`, not fail); re-run `tests/test_process_flow.py` and the `access-request@v1` acceptance
  flow to confirm the `maxSteps: 24` headroom still covers it.

### K-038 — `refreshRunPanel` has no mutex against overlapping poll-tick/submit-response invocations (🔵 proposed — filed out of K-036's Wave 3+4 analyst re-review gate, `docs/reviews/web-api-coverage-impl.md` Pass 3, findings m6/m7, 2026-07-29)

> **Why it exists.** Pass 3 fixed M1 (the destructive every-tick `renderWaitingForm` rebuild) and,
> while deep-tracing the fix per the task's own request, found two narrower, non-blocking races in
> the same run-panel poll/submit machinery — recorded there rather than chased at gate time, and
> filed here as this item per the K-036 close-out plan.
- **m6 — unordered concurrent `refreshRunPanel` calls; a stale response can transiently overwrite
  a fresher one.** `refreshRunPanel` (`web/app.js`) is not mutex-protected against overlapping
  invocations, and both `startRunPolling`'s periodic tick and `submitRunInput`'s post-submit call
  invoke it independently. If a tick's `refreshRunPanel` is still in flight
  (`Promise.all([GET run, GET step-runs])` awaiting) at the moment a submit resolves and triggers
  its own `refreshRunPanel`, both fetches race against the same `runId`, and whichever's
  `Promise.all` resolves *last* wins the render — regardless of which is fresher. A stale
  (pre-submit) response landing after the fresh one briefly shows an already-superseded step.
- **m7 — the same-step-key rebuild guard can't distinguish "still the same wait" from "revisited
  after an externally-driven round-trip this session's poll never observed."** The
  `state.runWaitingKey` guard (M1's fix) only resets when *this session's own* render observes a
  non-`waiting` status. If some other actor completes a full `waiting → running → waiting`
  round-trip on the *same* `atStepKey` entirely between two of this session's poll ticks, the key
  still matches on the next tick and the box stays hidden behind the early-return guard even
  though it's a genuinely new visit to that step.
- **Non-blocking, self-healing — carry this framing accurately.** Both require timing conditions
  well outside normal single-operator, human-typing-speed form use (a sub-`POLL_MS`, i.e. <3s,
  window for m6; an external actor plus a round-trip faster than one `POLL_MS` for m7), and both
  self-heal within one more `POLL_MS` (≤3s) tick — the next poll fetches true current state again
  and forces a correct rebuild either way. Neither loses user-entered input (unlike M1); neither is
  a permanent inconsistency.
- **Owner:** **`frontend-engineer`**, `web/app.js`.
- **Scope sketch (to be designed, not decided here):** m6 — a request-sequence token (stamp each
  `refreshRunPanel` call with an incrementing counter, ignore a response whose stamp is behind the
  latest issued) would close it; m7 — needs a server-supplied, monotonically-changing identifier
  per wait-instance (not just `atStepKey`) to distinguish "unchanged wait" from "revisited wait",
  which is a server-side surface change, not JS-only.
- **Risks/RAM:** none — web-only, no server/graph surface touched by m6; m7's full fix would need a
  new/changed field on the run-detail response, scoped by whoever picks this up, not decided here.
- **Test strategy:** a DOM-stub harness (the same shape Pass 2's fix verification used) driving two
  overlapping `refreshRunPanel` promises resolving out of order, asserting the later-issued one
  always wins regardless of resolution order (m6); an injected-clock or mocked-response test
  simulating an external round-trip between two ticks, asserting the panel still rebuilds (m7).

### K-040 — `POST /workflow-runs`'s request field is `version`, not `defVersion` — decide whether to rename for consistency (🔵 proposed — found during a `tico` manual-verification pass, 2026-07-31)

> **Why it exists.** `StartWorkflowRunIn` (`server/falkorchat/schemas.py:198-202`) declares the
> field as `version`, while the rest of the def/snapshot vocabulary — `WorkflowRun.defVersion`
> (DESIGN §6.2), the `GET /workflow-defs/{key}/versions/{version}` path segment name notwithstanding,
> and the general "def key + version" phrasing throughout `DESIGN.md` §14.4/§12 — uses `defVersion`.
> The mismatch is easy to get wrong by pattern-matching the graph model rather than the actual
> schema: it produced a real `422 Unprocessable Entity` in a first draft of
> `falkor-chat/docs/manuals/workflows.md`'s API walkthrough, caught only because a `qa-engineer`
> verification pass actually called the endpoint rather than composing the example from the schema
> next to the design doc. Already fixed in the manual (`docs/manuals/workflows.md`, 2026-07-31); this
> item is about the underlying inconsistency, not the doc.
- **The actual decision, not pre-judged here:** either (a) rename `StartWorkflowRunIn.version` →
  `defVersion` for consistency with the rest of the surface — a live API contract change, so it
  needs an assessment of what currently depends on `version` (the web UI does not call this route
  today per `docs/manuals/workflows.md`'s Walkthrough 4 note that it's API-only; MCP tools don't
  cover workflow runs either per DESIGN §15.2 — so the blast radius may be small, but that needs
  confirming, not assuming) — or (b) leave the field as-is and treat this as a documented,
  intentional naming divergence (a one-line callout at `DESIGN.md` §14.4 / `QUERIES.md` §12.12 would
  suffice). Both are legitimate; this item exists so the choice is made deliberately rather than by
  the next person guessing again.
- **Owner:** **`architect`** — assess the rename's blast radius and decide (a) vs (b); if (a),
  **`coder`**/`tdd-engineer` implements + updates every call site and doc reference.
- **Scope:** grep every caller of `POST /workflow-runs` (tests, scripts, docs, the manual) before
  deciding; if renaming, land the schema change, the `test_process_flow.py`/`test_services.py` call
  sites, and the `docs/manuals/workflows.md` example in the same change.
- **Risks/RAM:** none — no graph/DDL surface; purely a request-schema field name. The only risk is a
  silently-missed caller if the rename is chosen without the grep above.
- **Test strategy:** if renamed, flip the existing `test_process_flow.py` start-run assertions to the
  new field name (should go red first, confirming the old name is actually gone) plus a negative test
  that the old `version` name is rejected, not silently ignored.

### K-043 — `compose.yaml`/`Dockerfile` never verified against a real `docker build`/`docker compose` (🔵 proposed — filed out of K-042 close, 2026-08-11)

> **Why it exists.** K-042 Landing 1's L1-5 unit updated `compose.yaml` (the two config-file paths,
> a read-only bind mount of the shared overlay file, `host.docker.internal:host-gateway`) and
> `Dockerfile`-adjacent run instructions on the strength of static review only — no Docker toolchain
> was available anywhere in the coordination pipeline (agents, gates, or the QA acceptance pass), so
> the change was never exercised by an actual `docker build` / `docker compose up`. The risk is
> narrow (a bind-mount path typo or a missing `host.docker.internal` extra_hosts entry would only
> surface in a real container run) but real, and it's the one surface in K-042 that shipped unverified.
- **Owner:** **`devops`** — build the image, bring the stack up via `compose.yaml`, and confirm the
  server inside the container can resolve both config-file paths and reach LM Studio on the host via
  `host.docker.internal`.
- **Scope:** `docker build` against `falkor-chat/Dockerfile`; `docker compose up` against
  `falkor-chat/compose.yaml`; confirm the bind-mounted shared overlay file is readable at the path
  the container expects, and that a chat request round-trips to the host LM Studio instance.
- **Risks/RAM:** none — verification only, no design change expected unless the run surfaces a defect.
- **Test strategy:** a live manual run is the test; if it surfaces a defect, file a fix as its own
  follow-up rather than folding it into this item.

### K-070 — Schedule a document's embeds before its extracts to cut model swaps on a one-model-at-a-time LM Studio (🔵 proposed — filed out of K-067, `docs/plans/extract-transient-retry.md` §6 follow-up F-1, 2026-10-15)

> **Why it exists.** `_schedule_chunk_processing` interleaves `embed(c1), extract(c1), embed(c2), …`.
> On the sequential REST path, with LM Studio holding one model at a time (plan §2.5), that forces
> a model swap on **every** job — and the eviction/reload is what triggers the K-067 faults in the
> first place. Scheduling all of a document's embeds, then all its extracts, would cut swaps per
> document from ~2N to ~2 (plan F-1). It is a behaviour change to scheduling order, so it needs its
> own small design and a live measurement. The MCP-side analogue, per-document job serialisation
> (plan §6 F-4), is not filed: both K-067 leg (g) runs ended `ready` (n = 1 each), so its filing
> rule did not fire (report B.4).
>
> **Live data point** (2026-10-17 K-071 re-drive, `HISTORY.md`): 13 documents re-ingested at once
> through `:8200` drove LM Studio into JIT chat/embed model swapping, whose unload bodies failed
> 11 of them fast (D-3). The sequential re-drive that followed, with both models pre-loaded
> (`lms load`), brought every document to `ready` with no `ProviderCallError` or retry line at
> all. Avoiding swaps removed the failures entirely; that shows the lever matters, but it is not a
> measurement of this item's reordering, which still needs its own.
- **Owner:** `architect` (small design) → `coder`/`tdd-engineer`.
- **Scope:** `_schedule_chunk_processing`'s job order on the REST chain.
- **Risks/RAM:** none in graph terms; changes when each chunk becomes searchable within a document.
- **Test strategy:** an offline test pinning the new order, plus a live measurement of swaps per
  document before and after.

### K-072 — Per-request "provider down" short-circuit for REST batch ingestion (🔵 proposed — filed out of K-067, `docs/plans/extract-transient-retry.md` §6 follow-up F-3 / review Pass 1 m-2, 2026-10-15)

> **Why it exists.** `api.ingest_documents` puts every document's jobs on one sequential
> `BackgroundTasks` chain. In an outage, the K-067 liveness gate bounds the cost **per document**,
> not per request: each document's first faulting job burns its full retry budget before the gate
> can trip, so a 19-document batch costs ≈ 13–19 min of one worker token asleep, with every later
> document stuck in `processing` (review m-2; plan §6 R-4). The named option is one shared flag per
> chain: after one job exhausts its budget on a transient kind, later documents in the same chain
> make one attempt and no sleeps, turning the outage cost into one budget per request (plan F-3).
> Worth doing only if batch latency matters.
- **Owner:** `architect` (design) → `coder`/`tdd-engineer`.
- **Scope:** the REST batch chain only; the MCP path's per-job threads are unaffected.
- **Risks/RAM:** none in graph terms.
- **Test strategy:** an offline test with an always-transient fake provider over a multi-document
  batch, asserting one full budget per request, not per document.

### K-073 — Jobs of an already-failed document still make their first provider call, because the liveness gate is read only between attempts (🔵 proposed — filed out of the D-3 review, `docs/reviews/extract-transient-retry2.md` O-1, 2026-10-17)

> **Why it exists.** Once one job fails a document (K-051: one failure fails it), every remaining
> job of that document still runs attempt 1: `_safe_*` has no status check, and the K-067 liveness
> gate (`Repository.chunk_document_is_processing`) is read only before and after each retry sleep
> (`docs/plans/extract-transient-retry.md` §3.5). Each of those calls is wasted, and under LM
> Studio's one-model-at-a-time JIT swapping each one adds a swap — the trigger of the failures in
> the first place. The 2026-10-17 re-drive log shows such jobs failing transiently and then
> abandoning at the pre-sleep gate read (review O-1). Not a defect: it is the specified design.
> The review's candidate is one gate read before attempt 1. The finding is **inferred** from the
> code; the log cannot show how many of those calls began after the document had failed.
- **Owner:** `architect` (decide whether the pre-attempt read is worth it) → `tdd-engineer`.
- **Related:** K-070 (fewer swaps per document) — the natural companion; design them together.
- **Risks/RAM:** a pre-attempt read moves the gate onto the success path — one more indexed read
  per job, where today it costs nothing until a job fails (plan §6 R-7).
- **Test strategy:** an offline test where a sibling has already failed the document: the job
  makes no provider call and still reports exactly once.

### K-074 — MCP `ingest_document` accepts a title over `MAX_NAME_LEN` (200) that REST `POST /documents` rejects with a 422 (🔵 proposed — found during the K-071 re-drive, 2026-10-17)

> **Why it exists.** REST validates `title` (and `sourceLabel`/`producedBy`) at the pydantic
> boundary, `IngestDocumentIn` in `server/falkorchat/schemas.py`, capped at `MAX_NAME_LEN` = 200.
> MCP `ingest_document` has no schema layer, and `Services.ingest_document` checks only the text
> (empty, `MAX_DOCUMENT_CHARS`), not the title. So the same document is accepted on one front door
> and rejected on the other: `ws:agent-team` document `9b2c6b30…` was ingested through MCP with a
> 331-char title, and its 2026-10-17 re-ingest through REST was rejected with a 422 until the
> title was shortened to 200 chars (`HISTORY.md` 2026-10-17).
- **Owner:** `tdd-engineer` (reproduction test first).
- **Scope:** make the two front doors agree on the title cap (and check the other capped fields).
  The direction is the implementer's call: enforce `MAX_NAME_LEN` in `Services.ingest_document`
  for both paths — the precedent is the text cap, already mirrored there for MCP's sake — or
  relax REST.
- **Risks/RAM:** enforcing the cap on MCP rejects long titles that MCP callers send today.
- **Test strategy:** the same over-length title through both REST and MCP gets the same outcome.

### K-076 — Entity fuzzy query: symbol-only names make junk suggestions, and `\`-escaped punctuation misses its own name (🔵 proposed — `docs/reviews/fusion-fuzzy-escape.md` Pass 2, P2-1/P2-2, 2026-10-17)

> **Why it exists.** Two minor gaps left by K-068's `fusion._fuzzy_query`, both measured live and
> deliberately not fixed in that change. (1) A name with no letter or digit but non-ASCII symbols
> (`∅`, `…`, `→`) builds a one-character fuzzy term: in `ws:agent-team` such a term matches 802
> entities, and for the real `Concept` `∅` the top candidate is `Auth ✔`, so a junk pending
> `SAME_AS` suggestion is written. Not a regression — the pre-K-068 code sent the same `%∅%`.
> (2) The builder deletes every `\`, but the index keeps `\<punct>` as a literal inside the term,
> so `a\-b`, `re\.compile` and `a\%b` miss their own stored name. Zero real misses today.

- **Owner:** `tdd-engineer` (failing test first).
- **Scope:** (1) drop tokens with no letter or digit — trades recall on symbol-only names for no
  junk suggestions, so confirm the trade first; decoy test: seed `∅` plus a same-type `Auth ✔`
  and assert `Auth ✔` is not returned. (2) keep a `\` that precedes ASCII punctuation, delete any
  other; add an `aa\<p>bb` axis to the live coverage test. The review's probes and a tested patch
  for each are in the review's Pass 2.
- **Risks/RAM:** none — query-construction change only.

### K-064 — `find_update_shortlist`'s title-fuzzy signal uses implicit-AND token combination, far narrower than its own docstring implies (🔵 proposed — filed out of `document-ingestion2` Stage E QA pass, `docs/test-reports/document-ingestion2-report.md` Defect 2, 2026-09-13)

> **Why it exists.** `repository.find_update_shortlist`'s title-fuzzy RediSearch query joins one
> `%token%` term per title word with a space, which RediSearch combines via implicit AND — a
> candidate's title must fuzzy-match **every** token of the new document's title to surface at
> all. Live-confirmed: a document titled `"Beta Doc"` was never shortlisted against a query title
> of `"Beta Doc v2 clean title"`, even though both share two distinctive words. Any ordinary
> revision suffix ("v2", "Draft", "Final", "Updated") defeats the signal for its own predecessor
> unless the predecessor's title happens to already contain that exact word. This is narrower than
> the method's own docstring promises ("a cheap complementary booster... title-fuzzy match...
> when non-empty," `repository.py:1863-1868`, plan §3.4) — a reasonable reader expects any shared
> distinctive word to shortlist a candidate for the precise Jaccard check to then judge.
>
> **Not an AC violation** — the LSH/MinHash-banding signal is the primary, fingerprint-based net
> (ML note §4.1); title-fuzzy is explicitly "complementary," not load-bearing, and AC-2 held in
> Stage E's live fixture via the band signal. This narrows the secondary signal's real-world
> recall, it does not remove suggested-tier detection.
>
> **The decision is a design call, not a mechanical fix**: explicit-OR the per-token fuzzy terms
> (`%tok1%|%tok2%|...`) trades recall up against precision down (a title sharing just one common
> word, e.g. "Report", would now shortlist against unrelated documents — bounded in practice by the
> LSH signal already narrowing to true near-duplicates and the exact-Jaccard check downstream
> re-filtering the shortlist, but not zero-cost); dropping to a smaller, more distinctive token
> subset is the other named option. Route to `data-scientist` for a method opinion (RAG/candidate-
> generation recall/precision trade-off, squarely its lane) before `coder`/`tdd-engineer`
> implements — this is exactly the "reviewer's suggested fix is a finding to judge, not an
> instruction to apply" case: the QA report's own two options are plausible, not a decided answer.
>
> **Related:** the entity-name path (`fusion._fuzzy_query`) no longer crashes on punctuation
> (K-068, `HISTORY.md` 2026-10-17). It now builds its terms the way the index tokenizes the name,
> so it adds terms only where the index holds them too. The title path still strips per token
> (`_escape_fuzzy_token`, `HISTORY.md` 2026-09-13); like the entity path, it drops
> underscore-only tokens (K-075, `HISTORY.md` 2026-10-17). Both paths still AND their terms.
> One input for the OR-vs-subset call, if it is ever extended to fusion: a stored glued name (`modelbench`) can be missed by a punctuated new name (`model-bench`)
> under index-aligned tokens, though stripping would have found it.
- **Owner:** **`data-scientist`** (method note) → **`coder`**/**`tdd-engineer`** (implementation).
- **Scope:** `falkor-chat/server/falkorchat/repository.py`'s `find_update_shortlist` title-fuzzy
  branch only; `fusion._fuzzy_query` (entity-name matching) shares the same implicit-AND shape but
  is out of this item's scope — note it as a related, unaddressed instance if picked up.
- **Risks/RAM:** none — query-construction change only, no schema/index change.
- **Test strategy:** a live-driven regression case with a title carrying an ordinary revision
  suffix (mirroring Stage E's `"Beta Doc"` / `"Beta Doc v2 clean title"` pair) must shortlist the
  predecessor after the fix; a hard-negative case (two titles sharing only a common stopword-ish
  term) should not flood the shortlist — both belong in `test_repository.py` plus a
  `test_queries.sh` addition per the plan's own "every new Cypher shape raises the baseline" rule.

## Deferred — M2.5 hardening track

Auth + real-time. Not on the M5 path; no scheduled start.

### K-016 — Real auth/tenancy replacing the hardcoded `get_context` seam (🔵 proposed — M2.5, deferred)

- **Owner:** **`architect`** (design pass — designs the auth mechanism *per* the authoritative-identity decision, now
  resolved: the `identity` graph is authoritative/standalone, DESIGN §1.2) → **`tdd-engineer`** (implement the resolved `get_context`).
- **Inputs/prereqs:** the identity source-of-truth is **decided** (identity graph authoritative/standalone; DESIGN §1.2) —
  K-016 no longer needs the user for that axis; it implements per that decision. Localized by design — only
  `config.get_context` changes (`config.py:43`); everything below already parameterized on `ws`/`actor`.
- **Scope:** token → (user, workspace claim) resolution replacing hardcoded `ws=acme/user=u1`; wire the `identity`
  graph per the §1.2 authoritative-identity decision; keep or replace MCP's `frm`-ignoring rule with authenticated agent identity.
- **Done-condition:** `get_context` resolves a real principal from a credential; multi-tenant isolation test; pytest green.
- **Risks/RAM:** `identity` graph nodes (small). First real trust boundary — MCP endpoint is currently unauthenticated (§15.3).
- **Test strategy:** service/api tests with injected auth contexts; a cross-tenant isolation test.

### K-017 — Transport-level agent-actor path (K-007 QA carry-over) (🔵 proposed — M2.5, deferred · depends on K-016)

- **Owner:** `qa-engineer` (+ small `tdd-engineer`/`coder` fold-in if MCP must express an authenticated agent actor).
- **Scope:** with auth able to express an *agent* principal, drive an external agent authoring over MCP/REST (the M1
  hardcoded seam couldn't) and verify authorship/role/provenance end-to-end.
- **Done-condition:** the K-007 QA carry-over closed — a report showing an externally-authenticated agent authoring
  first-class over the transport.
- **Risks/RAM:** none new. **Test strategy:** black-box over MCP with an agent credential.

### K-018 — Real-time push (Redis Pub/Sub → WebSocket/SSE) (🔵 proposed — M2.5, deferred)

- **Owner:** **`architect`** (design: Pub/Sub fan-out topology; resolve the DESIGN §13 Bolt-vs-RESP gateway question
  here since it touches the transport) → **`coder`/`tdd-engineer`**.
- **Inputs/prereqs:** K-012/K-014 web client (swap polling → push).
- **Scope:** Redis Pub/Sub on message write → WebSocket/SSE endpoint on the same FastAPI process (§14.1: "slots onto
  the same service layer, no schema change") → web client subscribes instead of polling.
- **Done-condition:** a posted message appears in another client without a poll; graceful fallback to polling.
- **Risks/RAM:** no graph RAM; Pub/Sub is transient. Publish *after* the guarded §4 write commits, never inside it (atomicity rule).
- **Test strategy:** integration test of publish-on-write + a WebSocket client receiving it.
- **Related work (client-side polling alternative):** `mcp-monitor/` (`mcp-monitor/docs/requirements/mcp-monitor.md`) has shipped as a separate, polling-based watcher that detects MCP tool-result changes and launches commands — a distinct, complementary approach to K-018's server-side push. K-018 remains its own open item.

## Plan docs still to author

Every other plan named by a delivered item now exists; this is what is left.

| Path | For | Scope |
|---|---|---|
| `docs/plans/auth-tenancy.md` | K-016 (deferred) | Real auth replacing the `get_context` seam, per the §1.2 identity-authoritative decision. |
| `docs/plans/realtime-push.md` | K-018 (deferred) | Pub/Sub → WebSocket/SSE, resolving DESIGN §13's Bolt-vs-RESP question. |

> Both slugs were previously listed as `m2-auth-tenancy.md` / `m2-realtime.md`. Renamed here before
> creation: root `AGENTS.md`'s filename grammar forbids a new document's basename beginning with
> `m<digit>`, and neither topic *is* a milestone.

## Parking lot / ideas

- Verify the K-009 GitHub Action goes green on first push (path-filtered `.github/workflows/falkor-chat.yml`; FalkorDB
  service container). Note the CI baseline echoes in its comments (75/92) predate K-007/K-010's 110/126 — the suites
  themselves are the source of truth. (K-019 fixes the README/DESIGN body numbers; the CI comments are separate.)
- File upstream FalkorDB issues (K-007 OQ6, recommended to the user): `GRAPH.MEMORY USAGE` under-reports vector-index
  memory; one-shot instant-timeout anomaly after a long override run.
- Per-endpoint response schemas (QA, recommended three times now): full-thread / since-reads / search each carry a
  different field subset (all documented/intentional) — a declared schema per endpoint would make the contract testable
  and stop accretion. **The repo is deliberately on a mixed convention today.** The three §11 structure/diff routes
  declare `response_model=` (`WorkflowDefStructureOut` / `WorkflowDiffOut`, `schemas.py`) with exact-key-set contract
  tests; no pre-existing route does. The non-retrofit is deliberate, not pending — FastAPI's `response_model` *filters*
  undeclared fields, so a wrong model silently drops a field the web client reads, which makes a bulk retrofit riskier
  than the accretion it would stop. The §11 routes are the worked precedent for an eventual per-route retrofit.
- **Opportunistic nit — re-slug the K-031 implementation review** (recorded, **not** scheduled work).
  It is filed under the slug `k031-structure-read-impl`, while the rest of its family — the plan and
  the plan review — uses `workflow-def-structure-read`. The filename grammar's family rule (*the same
  slug across several kinds **is** the family; a downstream document inventing a new slug is a
  defect*) is therefore broken by one member. Correcting it to `workflow-def-structure-read-impl`
  spans **four files** as of 2026-08-25 (this backlog, `docs/HISTORY.md`,
  `docs/plans/m3-followups-coordination.md`, `docs/plans/workflow-republish-semantics.md`) — check
  with `grep -rln k031-structure-read-impl docs/` rather than trusting that count. Fold it into a
  change that already opens them; it does not earn a change of its own, and renames in this repo are
  forward-only by ruling.
- DESIGN §13 remaining open questions — resolve as their milestones arrive: real auth (K-016),
  message/embedding retention, cross-workspace analytics, Bolt vs RESP for the gateway (K-018).
  (The workflow guard expression language is **no longer** among them — resolved at DESIGN §6.1,
  and §13 records it struck through.)
- **WSL2 memory cap for the 16GB host** (parked, not applied per user 2026-07-18) — WSL2 runs uncapped at its 8GB
  default (50% of the 16GB host) with `autoMemoryReclaim` off, overcommitting host RAM alongside Windows-side LM Studio;
  likely root cause of the recent memory-overload crashes. Parked fix: set `memory=6GB` + `swap=4GB` +
  `autoMemoryReclaim=gradual` in `C:\Users\mauri\.wslconfig` (keep `networkingMode=mirrored`), then `wsl --shutdown`.
  Full diagnostic + apply procedure: `docs/plans/wsl2-memory-diagnostic.md`. Un-park (apply) if the crashes recur —
  verdict was confirmed-by-defaults, not reproduced live (FalkorDB was down during the diagnostic).
