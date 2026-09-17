# `model-bench` S6 — `tool-caller` pack, part 2: live-run test report

> **Status:** archived · **Owner:** `qa-engineer` · **Tracks:** — (S6)

## Summary

Executed `docs/plans/small-model-benchmarking-s6-spec.md` §4 Steps 7-8 (three live invocations,
item 19a's negative control, item 19b's known-answer validation) plus this stage's acceptance-tier
obligations (§6), against `model-bench/` at commit `702704f` (worktree otherwise clean;
`packs/tool-caller-shop-assistant/pack.json` was temporarily edited for R-3's bisect, below, and
reverted — confirmed byte-identical to its committed state via `git diff` before finishing).

**This pass hit and resolved two forks along the way**, both escalated rather than worked around
unilaterally at the time: a stale-attestation environment blocker (resolved by re-attestation, with
evidence, no field guessed), and a confirmed harness/model-template defect in `convo.py` that made
the first pass's `ministral-run-1` produce zero usable turns. That defect has since been **fixed
and independently gated** (`modelbench/convo.py`'s `assemble()` now merges the system-prompt and
tool-schema text into one `role:"system"` message via `_prologue_system_message`; full suite green,
1625 passed; `analyst` gate: approve with suggestions, no blockers —
`model-bench/docs/reviews/small-model-benchmarking-s6-convo-fix.md`). **All three live runs in this
report are fresh, taken after that fix landed.** The three runs stored during the pre-fix pass
(`...2026-09-17T00:23:01Z`, `...T00:25:50Z`, `...T00:28:36Z`) are left on disk, historical/
diagnostic only — not used anywhere in this report's own comparisons or verdicts.

**Verdict: PASS on this stage's own done-condition.** Item 19a passed cleanly. Item 19b's
known-answer contrast did not reach statistical significance at this pack's n=12 sizing on its own
baseline configuration; per spec §3.7's explicit fallback, R-3's bisect was executed in full (both
named rungs) and its results recorded — neither rung produced the contrast either, so the pack
ships flagged **`known-answer validation: not reproduced`**, exactly the outcome §6's own
done-condition accepts ("gates on 19b being *run and recorded*, never on the contrast appearing").

**CPG:** considered, not relevant — no `cpg_model-bench` graph is loaded on this FalkorDB instance;
this is a code-level task in a component with no CPG.

## Acceptance-tier checks (§6) — all done, all PASS

1. **Baseline suite**, re-confirmed after the `convo.py` fix: `.venv/bin/python -m pytest -q` →
   **1625 passed, 3 deselected**, `ruff check .` clean.
2. **`validate` on the real, fully-authored pack**: `./run.sh validate --pack
   packs/tool-caller-shop-assistant` → `tool-caller-shop-assistant 0.2.0 (tool-caller): valid`,
   exit 0. Re-confirmed clean after reverting the bisect's temporary manifest edits (below).
3. **`validate --strict`**: raises `NotImplementedError` naming runner-spec §9, exit 1 — `cli.py`'s
   own documented, deliberate deferral, not a defect. Confirmed by direct execution.
4. **Independent spot-check of all 12 scripts' `expect` blocks against `tools/sim.py` directly**
   (independent of the prior agent pre-check and the stakeholder review). Every `lookup_product_fact`
   price assertion recomputed against `catalog.json`'s real prices; every `view_cart` total in the
   four Shape-B scripts recomputed against `_add_to_cart`/`_remove_from_cart`/`_place_order`'s real
   arithmetic (B-01: 179.97→49.99; B-02: 202.99→291.99→0.00 after `place_order`; B-03: 44.99→0.00
   after `clear_cart`; B-04: 42.45); every abstention target confirmed genuinely absent from the
   real 14-item catalog. **Zero discrepancies.**
5. **`prose_calibration.jsonl` verified against the real detector**: all 20 rows (10 positive/10
   negative) run through `detect_prose_pseudo_call` directly — every prediction matches its label,
   `prose_detector_precision_recall` returns `(1.0, 1.0)`. Noted under Coverage & gaps: the corpus
   is fully separable by the shipped regex families, a real but undemanding calibration.

## Environment/defect narrative (both forks, both resolved)

**Fork 1 — stale attestation.** The first live-run attempt refused at exit 5
(`EXIT_FINGERPRINT`): `host.json`'s attestation was stale because LM Studio's llama.cpp CUDA
runtime engine had updated (`2.37.0` → `2.39.0`) since the 2026-09-13 attestation. Confirmed via
LM Studio's own on-disk state (`~/.lmstudio/.internal/backend-preferences-v1.json` showing the new
engine version; `~/.lmstudio/.internal/historical-version-info.json` confirming the app version
itself, `0.4.24`, was unchanged) before re-attesting with the same, evidence-confirmed operator
fields — no fingerprint value was guessed.

**Fork 2 — a confirmed `convo.py` defect, now fixed.** The pre-fix pass's `ministral-run-1` scored
0 of 81 turns (all "unrunnable (model channel)"), root-caused by direct reproduction:
`convo.assemble()` sent the model two separate `role:"system"` messages on every turn, and
`mistralai/ministral-3-3b`'s own chat template rejects that shape outright (a live curl replication
showed LM Studio returning the template's own Jinja exception, *"Only user, assistant and tool
roles are supported, got system"*). This was escalated rather than worked around; the coordinator's
decision was **fix now, re-run fresh** — `convo.py`'s `_prologue_system_message` now merges both
pieces of text into one system message, closing the defect for every future run against any model,
not just this one. Confirmed sound by a fresh, independent `analyst` gate and by both `teco`'s and
that gate's own mutation probes beyond the fixing delegate's table (join-order swap, dropped
turn-index branch) — both confirmed the fix is genuinely pinned. This report's own live runs are
all taken after the fix; the two learnings (the attestation trip-wire's exact trigger, and
`ministral-3-3b`'s multi-system-message rejection) are recorded in the `kaizen_team` working-memory
graph for future reference.

## Step 7 — the three live invocations (all fresh, post-fix)

Residency (`GET /api/v0/models`) was re-confirmed immediately before each invocation across this
entire pass (every check showed `state: not-loaded` at that moment, confirming availability-only
per §2.7 — each run's own warm-up call then drove the actual load).

| Run | Model | Session | `runId` | Wall-clock |
|---|---|---|---|---|
| `qwen-run-1` | qwen/qwen3-4b-2507 | `v2-negctrl` | `tool-caller-shop-assistant-qwen_qwen3-4b-2507-2026-09-17T01:11:26Z` | 2m00s |
| `qwen-run-2` | qwen/qwen3-4b-2507 | `v2-negctrl` | `tool-caller-shop-assistant-qwen_qwen3-4b-2507-2026-09-17T01:13:38Z` | 1m56s |
| `ministral-run-1` | mistralai/ministral-3-3b | `v2-known-answer` | `tool-caller-shop-assistant-mistralai_ministral-3-3b-2026-09-17T01:16:02Z` | 4m01s |

All stored under `model-bench/results/runs/`, all exit 0. (`qwen-run-1`/`qwen-run-2` share a
session tag deliberately, so `compare --session v2-negctrl` selects exactly this pair unambiguously
regardless of the historical stale records also present in that directory — CLI arm-selection note,
not a data property.)

**Determinism probe (mechanism 1, §3.6), recorded honestly on all three:**

| Run | `ran` | `identical` | `basis` |
|---|---|---|---|
| `qwen-run-1` | `true` | `false` (A-01 turns 1-5 differ on replay) | `assumed` |
| `qwen-run-2` | `true` | `false` (A-01 turns 1-5 differ on replay) | `assumed` |
| `ministral-run-1` | `true` | `false` (A-01 turns 1, 5, 8 differ on replay) | `assumed` |

No value was overridden for either comparison's convenience — every field is exactly what the live
run recorded.

### 19a — negative control (`qwen-run-1` vs `qwen-run-2`): **PASS**

`./run.sh compare --pack tool-caller-shop-assistant --session v2-negctrl`

**Result: `cleanThroughTurnH`: not distinguishable** — observed difference **+0.0 pp**, 95% CI
**[-24.2, 24.2] pp** (covers zero), discordant counts **b=0, c=0** (exactly equal), McNemar exact
**p=1.000**, **12 of 12 conversations paired** (no censoring this time — cleaner than the pre-fix
pass's own 19a, which had 2 censored turns in one arm). Both arms: `cleanThroughTurnH` 0/12,
`native` 23/75, `rightToolChosen` 20/23, identical on every printed figure. This is exactly the
plan's own required 19a outcome (`:6309`) — **item 19a passes, licensing item 19b's
interpretation.**

Full output: session scratch `s6-runs/step7-19a-compare-v2.md`.

## Step 8 — item 19b, the known-answer validation

### Baseline (pack's own shipped config, `historyReplay: structured-replies-only`)

`./run.sh compare --pack tool-caller-shop-assistant --models qwen/qwen3-4b-2507,mistralai/ministral-3-3b`
(no `--session`; `--models`' own dict-collapse selects the newest stored record per key, which —
verified directly against `load_history`'s own read order — resolved to `qwen-run-2`
(`...T01:13:38Z`) and `ministral-run-1` (`...T01:16:02Z`); `qwen-run-2`'s own figures are identical
to `qwen-run-1`'s on every metric per 19a above, so which of the two negative-control arms gets
reused here changes nothing about the comparison's substance).

**Funnel — real, live data for both arms** (contrast with the pre-fix pass's ministral funnel,
which showed 81/81 turns unrunnable):

| Metric | qwen/qwen3-4b-2507 | mistralai/ministral-3-3b |
|---|---|---|
| turns driven | 81 | 81 |
| unrunnable (model channel) | 0 | 0 |
| native call emitted (of R(t)>=1) | 23/75 | 73/75 |
| rightToolChosen | 20/23 | 66/73 |
| allArgsCorrect | 20/20 | 62/67 |
| duplicateCrossTurn | 1/23 | 17/73 |
| **cleanThroughTurnH (verdict metric)** | **0/12** | **4/12** |

**Per-turn-position** (qwen / ministral, observed k / n still at risk): t=0 1/12 · 1/12→0/12*;
t=1 7/11 · 2/12; t=2 3/4 · 2/10; t=3 1/1 · 4/8; t=4 0/0 · 0/3 (*ministral's own t=0 column in the
report reads `0/12` — no turn-0 failures for ministral). By t=4, all 12 qwen conversations have
already had their first failure (hazard table: f=1,7,3,1 at t=0..3, summing to 12); ministral
still has 3 conversations surviving clean into t=4, then loses more gradually (f=0,1,1 at t=4..6,
2 censored). This is the **directional** shape target 1 (`-ml` §8.2) predicts — qwen collapsing
hard and early, ministral surviving meaningfully longer — but at a **less extreme magnitude**
than target 1's own worked example (which assumed near-total ministral success, `b=12,c=0`).

**Verdict: `cleanThroughTurnH` — not distinguishable.** Discordant pairs **b=0, c=4** (ministral
clean where qwen failed, in 4 of the 12 conversations; qwen clean where ministral failed, in
none). The conservative envelope **excludes zero** ([-60.9, -2.2] pp, ministral ahead
directionally) but McNemar's exact test does **not** reach significance (**p=0.125**), and per
this pack's own Rule 7 (`stats.verdict()`), the exact test is a **necessary condition** on this
decision path — not met, so the verdict is reported as not distinguishable rather than as a
significant finding the envelope alone would suggest. **The known-answer contrast, as §3.7 defines
it (`cleanThroughTurnH` returning `distinguishable`, ministral ahead), did not appear** at this
pack's baseline configuration and n=12 sizing.

Full output: session scratch `s6-runs/step8-19b-compare-v2.md`.

### R-3's bisect, executed in full per spec §3.7's explicit fallback

**No script's `expect` block and no test file was edited in response to this non-reproduction** —
only `pack.json`'s `historyReplay` field, moved one rung at a time exactly as §3.7 specifies, then
reverted to its Step-6 canonical value (`structured-replies-only`) once the bisect concluded
(confirmed via `git diff packs/tool-caller-shop-assistant/pack.json` showing no diff, and a final
`validate` pass, both after reverting).

**Rung 1 — `historyReplay: structured`** (two fresh live runs, `qwen-bisect1` /
`ministral-bisect1`, session `bisect-structured`). Real data both arms (no unrunnable turns).
`cleanThroughTurnH`: qwen **6/12**, ministral **9/12** — replaying the full tool-call scaffolding
sharply reduces qwen's own collapse (0/12 → 6/12), consistent with the collapse mechanism being
about visible tool evidence in history (`-ml` §4.1). **Verdict: not distinguishable** — b=4, c=1,
McNemar exact p=0.375, envelope [-8.7, 58.3] pp covers zero. The contrast did not appear at this
rung either — and moved *further* from significance than the baseline, since both arms' rates
moved closer together.

**Rung 2 — `historyReplay: plaintext`** (two fresh live runs, `qwen-bisect2` /
`ministral-bisect2`, session `bisect-plaintext`). A **different, separate observation** surfaced
here: ministral's run showed 69 of 81 turns "unrunnable (model channel)" (11 native calls
succeeded before most conversations were censored — the hazard table shows all 12 conversations
censored at t=1). This is not the same defect as the earlier `convo.py` fix (that crash was
instantaneous and total, ~22ms per call; this run took 43 seconds wall-clock and did complete 11
calls), so it was **not** re-investigated further given this pass's scope — it is recorded as an
observation for a future pass, not root-caused here. **Verdict: no paired data** — 0 of 12
conversations scoreable for `cleanThroughTurnH` in both arms, so no interval, no verdict; this
rung's own result is honestly recorded as "unable to compare" rather than forced into a reading.

**Bisect summary: neither rung produced the contrast appearing.** Per spec §3.7's own instruction,
the pack ships flagged:

> **`known-answer validation: not reproduced`** — `cleanThroughTurnH` (ministral ahead) did not
> reach McNemar-exact significance at n=12 on the baseline configuration (b=0, c=4, p=0.125) or
> at R-3's rung 1 (`historyReplay: structured`; b=4, c=1, p=0.375); rung 2
> (`historyReplay: plaintext`) produced no paired data at all due to a separate, unresolved
> ministral censoring pattern under that history-replay mode. This is recorded as the pack's
> honest state, not as a defect to fix by adjusting any script.

Full outputs: session scratch `s6-runs/bisect1-compare.md`, `s6-runs/bisect2-compare.md`.

### The ministral duplicate-instruction rate (B-02, §3.7 point 2) — inside or outside this pack's resolving power

B-02's own `cleanThroughTurnH` outcome for ministral in the baseline run is `fail` (its
per-script record: `outcome: "fail"`), consistent with a defect firing somewhere in that script,
but the stored run record does not carry a per-turn dispatch trace rich enough to confirm the
*specific* duplicate-`add_to_cart` re-issue pattern at B-02's own turn 4 in isolation — only the
pooled, turn-level `duplicateCrossTurn` exploratory metric (17/73 = 23.3% of all scored turns in
the baseline run) is available from this report's own tooling, and `report.py` computes **no
formal significance interval for it** (it is listed under "Exploratory metrics — no significance
claim," never the pre-registered verdict metric). **This pack's comparison mechanism does not
compute a paired resolving-power figure for `duplicateCrossTurn` at all** — only for
`cleanThroughTurnH`. Reasoning conceptually from §3.7's own pre-stated expectation instead: the
plan's own worked magnitude for this defect (~30 pp, `-ml` §4.5.3) sits below this pack's stated
50.0 pp observable floor at n=12 for a *between-model* comparison of this rate — so even had this
run shown the duplicate-instruction defect at exactly its previously-documented ~30% conversation
rate, this pack's own sizing could not have resolved it as significant. **The ministral
duplicate-instruction rate, to the extent this run's data speaks to it, sits inside (below) this
pack's stated resolving power** — consistent with §3.7's own pre-registered expectation, not a
surprise this run discovered.

## Evidence on disk

- Stored run records (all 7 post-fix live invocations): `model-bench/results/runs/
  tool-caller-shop-assistant-{qwen_qwen3-4b-2507,mistralai_ministral-3-3b}-2026-09-17T01:{11:26,13:38,16:02,24:42,28:55,37:32,40:36}Z.json`.
- Historical/diagnostic-only pre-fix records (not used in any verdict above):
  `...-2026-09-17T00:{23:01,25:50,28:36}Z.json`.
- `host.json` re-attested `2026-09-17T00:22:48Z` (evidence-based, not guessed — narrative above).
- Comparison markdown: session scratch `s6-runs/step7-19a-compare-v2.md`,
  `s6-runs/step8-19b-compare-v2.md`, `s6-runs/bisect1-compare.md`, `s6-runs/bisect2-compare.md`
  (plus the superseded pre-fix `step7-19a-compare.md`/`step8-19b-compare.md`, kept for the
  historical record).
- Raw run stdout/stderr logs: session scratch `s6-runs/*.std{out,err}.log`.
- Pre-fix root-cause reproduction payloads (superseded by the landed fix, kept for the record):
  `/tmp/ministral_test_payload.json`, `/tmp/ministral_test_payload2.json`.
- `packs/tool-caller-shop-assistant/pack.json`: confirmed reverted to its committed state
  (`git diff` empty) after the bisect.

## Coverage & gaps

**Covered:** the full acceptance-tier checklist (§6); all three Step-7 live invocations, fresh,
post-fix; item 19a in full (clean pass); item 19b's baseline comparison; both of R-3's bisect rungs,
executed and recorded per spec, including rung 2's own honestly-recorded non-comparison.

**Gaps, named rather than hidden:**
- The known-answer contrast (§3.7's target 1/2) did not reach statistical significance at this
  pack's n=12 sizing on any of the three configurations tried — flagged `not reproduced`, per spec.
  A larger `n` (more scripts, or `replicatesPerScript > 1` — a bigger design change) would be the
  natural next lever, not something this stage's own scope covers.
- Rung 2's ministral censoring pattern (69/81 turns unrunnable under `historyReplay: plaintext`) is
  a real, observed fact but not root-caused here — a genuine gap for a future pass, distinct from
  the `convo.py` defect this pass already found and saw fixed.
- The duplicate-instruction rate's own significance could not be formally tested at all with this
  report's current tooling (no paired interval for an exploratory, turn-pooled metric) — reasoned
  about only conceptually against the plan's own pre-stated expectation, not measured directly.

## Feedback & recommendations

- **`report.py` prints no per-script breakdown**, only pooled turn-level exploratory metrics and
  the one per-conversation verdict metric. A future stage wanting to confirm a *specific* script's
  specific turn behavior (as this pass wanted for B-02) needs either a richer stored trace or a
  standalone tool like `scripts/s6_walkthrough.py` run against a live model — worth naming as a
  testability gap, not a blocker.
- **Rung 2's censoring pattern deserves its own root-cause pass** if `plaintext` historyReplay is
  ever considered for real use with `ministral-3-3b` — recommended as follow-up, not actioned here.
