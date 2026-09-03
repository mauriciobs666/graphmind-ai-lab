# Small-LLM benchmarking tool (`model-bench/`) — plan review

> **Status:** active · **Owner:** `analyst` · **Tracks:** — · **Reviews:** `docs/plans/small-model-benchmarking.md`

## Pass 1 — 2026-09-02

*(Current verdict is Pass 2's, below: **approve with suggestions** on plan v1.3 / note v1.3. Pass 1
gated plan v1.1 and is kept intact — the two passes are meant to be read together, and Pass 2's
disposition table is only legible against the findings here.)*

### 1. Scope & verdict

**Reviewed:** `docs/plans/small-model-benchmarking.md` (v1.1, `architect`, 1 125 lines) against
`docs/requirements/small-model-benchmarking.md` (Ready for design, 23 FRs / 5 ACs) and
`docs/plans/small-model-benchmarking-ml.md` (v1.1, `data-scientist`). Weighted toward S0–S3, which
is this pass's build scope per `docs/plans/small-model-benchmarking-coordination.md`, but
design-level findings anywhere are reported because an S1 seam constrains S5.

**Verified, not assumed.** Every factual claim the plan makes about `falkor-chat/server/tests/eval/`
was re-counted from the files (golden-set sizes and compositions, baseline numbers, corpus
provenance, judge calibration, symbol names, line counts, `nlq-34`'s stored record) — **all correct,
without exception**. The LM Studio surface was re-probed live on this box. The CPG blast-radius claim
was re-run. Details in the Appendix.

**Not reviewed:** `model-bench/` itself — `coder`'s S0 was in flight during this review and judging a
half-written tree would produce stale findings. One S0-relevant defect is reported (m1) because it is
actionable now. Behavioural/acceptance verification of the built harness is `qa-engineer`'s.

**CPG:** used `cpg_falkorchat` — independently re-ran the plan's §preamble blast-radius claim
(8 eval callees, callers in `tests/eval/*` only, 0 outside; `METHOD` counts 2 862 `tests/` vs 1 222
`falkorchat/`) and it reproduces exactly; the graph is stale per the brief but the claim is
directory-scoped and unaffected by the three intervening commits.

**Verdict: needs changes.** 3 blockers, 11 majors, 8 minors, 3 nits.

This is a strong plan — better grounded than most, and its honesty architecture (funnel table,
denominators inline, structural refusal to emit a blended number) is the right architecture. The
blockers are not disagreements with the design; they are three places where a guarantee the plan
*states* has no mechanism behind it, and all three land inside S1–S3.

---

### 2. Findings

#### Blockers

**B-1 — The tool-caller's decision instrument and its resolving-power line ignore clustering, which
the `-ml` note calls the feature's largest methodological risk.**
*Evidence:* plan §3.9 point 1 (McNemar exact = the decision rule), point 2 (`This pack resolves
differences of >= X pp …`), S1 signature `min_detectable_difference(n: int) -> float  # ~8/n`, and
`cluster_bootstrap(clusters: Sequence[Sequence[bool]], …)`. Against `-ml` §4.5 (R1, severity
**high**), §7.2 and §3.2(d).
*Why it matters:* the tool-caller pack is 48 conversations clustered in 12 scripts, at
`temperature: 0.0` (plan §3.3's own manifest). `-ml` §7.2 puts its MDD at "16 pp (unclustered) →
**up to ~65 pp (fully clustered)**", and §4.5 says temperature-0 replicates make effective *n*
"closer to 12 than 48". `min_detectable_difference(48)` = 16.7 pp regardless. So the pack whose
headline the whole feature exists for prints an honesty line that can be **4× too optimistic**, and
`verdict()` runs McNemar over 48 correlated observations — anti-conservative, i.e. it can print
*distinguishable* when it is not. Nothing in §3.8.4 or §5 test 7 addresses clustering for
`cleanThroughTurnH`; §3.9 point 4 covers only statistics "pooled across turn positions", which
`cleanThroughTurnH` is not. `cluster_bootstrap`'s signature is also one-level, where `-ml` §4.5
requires **two** (resample scripts, then replicates within each drawn script).
*Fix:* make the resolving-power line cluster-aware — `min_detectable_difference(n, *,
design_effect: float = 1.0)` fed by the observed design effect, and refuse to print a bare `n` for a
clustered pack; give `cleanThroughTurnH` a cluster-bootstrap CI over scripts as the effect size, with
McNemar retained on the script-level aggregate or explicitly labelled anti-conservative. Decide the
temperature/replicate question (see Open questions).

**B-2 — The metric-agreement cross-check, the *sole* discharge of D1's duplication risk, is not
constructible as §3.1 describes it, and S3 states it a second, incompatible way.**
*Evidence:* §3.1 point 2 — "`test_metrics_agreement.py` asserts that `model-bench`'s
`recall_at_k`/`mrr` reproduce, to 1e-12, the values in the copied `retrieval_baseline.json`
**when fed the same ranked lists**"; S3 done-condition — "run `falkor-chat`'s own `test_metrics.py`
fixtures through the copied implementation and require byte-identical output".
*Why it matters:* **the ranked lists do not exist as an artifact.** `retrieval_baseline.json` holds
four aggregate numbers; the ranked lists are produced live by `services.hybrid_search` against the
seeded `ws:eval` graph (`tests/eval/test_retrieval_eval.py:123`) — which `model-bench` must never
touch (FR-23) and which is ANN-approximate anyway. §3.1's version is impossible. S3's version is
possible but unspecified: `test_metrics.py`'s fixtures are hand-written parametrized cases inside a
pytest module, so "run them through" means either importing falkor-chat's test tree (breaks §3.1's
"zero imports in either direction") or hand-copying them (they then drift, defeating the purpose).
D1 — copy the data, clean-build the code — is the plan's central decision and this is the only
mitigation it offers for the risk the requirements explicitly asked to be justified against.
*Fix:* pick one mechanism and specify it. The cheapest defensible one: `refresh_golden.py` copies
`test_metrics.py`'s parameter tables into a pack-versioned `metrics_fixtures.json` at import time,
recording the source git SHA, and `test_metrics_agreement.py` runs the local implementation against
that file. That keeps the copy one-way, hashed, and re-refreshable, and it is diffable against the
origin. Then drop §3.1's "same ranked lists" wording.

**B-3 — The BM25 keyword arm (FR-13 / AC-5) has no representation in the result schema, and the
fingerprint's no-bypass rule forbids the obvious one.**
*Evidence:* §3.8.1 — "reported as a **full paired arm with a confidence interval**"; §3.4 point 1 —
`results.store()` "raises on any missing, empty, or `null` required field. There is no 'save anyway'
flag"; S1 signatures — `compare_report(runs: Sequence[RunResult], *, pack: PackRef)`; §3.4's
`REQUIRED_AUTO` = `modelKey`, `quantization`, `runtimeName`, `runtimeVersion`, `loadedContextLength`, …
*Why it matters:* BM25 has no model, no quantization, no runtime. As a `RunResult` it fails
validation on write; as anything else it cannot reach `compare_report`, whose only input is
`Sequence[RunResult]`. S3's done-condition — "`compare` renders it against the BM25 arm" — is the
endpoint of this build pass and there is no data shape that carries it there. This is the
fingerprint guarantee colliding with a legitimate non-model arm, and it will be resolved badly under
time pressure (a fake `modelKey: "bm25"` with placeholder runtime fields, which is exactly the
"invalid result reaching a report unlabelled" the design exists to prevent).
*Fix:* give the fingerprint an explicit `armKind ∈ {model, deterministic}` discriminator with a
distinct required-field set per kind (`deterministic` requires `packId`/`packVersion`/
`packContentHash`/`benchVersion`/`arm parameters hash`, and *forbids* the model fields), and state it
in §3.4 so `validate()` branches on it rather than on presence. Decide it in S1 — S3 consumes it.

#### Majors

**M-1 — `wilson_interval`'s default `z` diverges from the `-ml` note's mandated constant.**
Plan S1: `def wilson_interval(successes: int, n: int, z: float = 1.96)`. `-ml` §3.2(a): "Reuse
`nlq_scoring.wilson_interval` verbatim (`_Z_95 = 1.959963984540054`; this lab's convention)", and
`falkor-chat/server/tests/eval/nlq_scoring.py:59` defines exactly that. MOVER-D is derived from
Wilson (`-ml` §3.2c), so the five regression fixtures S1 must reproduce were computed at
1.959963984540054. Using 1.96 shifts every bound. (Note the lab is genuinely split — the salesperson
review §8.1 says "z=1.96, this lab's own established convention" — so this needs stating once, not
inferring.) *Fix:* pin `_Z_95 = 1.959963984540054` as a module constant and make `z` keyword-only, as
`nlq_scoring` does.

**M-2 — S1's AC-4 done-condition inverts the amendment's flagship case relative to §5 test 6.**
S1: "AC-4 (a paired-difference interval that includes zero renders the 'not distinguishable at this
sample size' wording — **including the 40/40 vs 34/40 case**…)". §5 test 6: "a paired-difference
interval containing zero produces the … wording, and the 40/40 vs 34/40 case does **not**". The
second is right (`-ml` §3.1: diff +15.0 pp, CI [3.2, 29.1], excludes zero → *distinguishable*). Read
literally, S1's clause pins the inverted verdict on the one case that exists to prove the amendment
works — and `tdd-engineer` builds S1 red-first from its done-condition (coordination U4). *Fix:*
rewrite S1's parenthetical to match §5 test 6 verbatim.

**M-3 — Fingerprint validation as specified rejects legitimate values, and capture timing is
unspecified.** §3.4 point 1 raises on "missing, **empty**, or `null`" required fields, and §5 test 1
is "one test per required field: blank it, assert it is named". But §2.3 records that `lms ps --json`
"returns `[]` when nothing is loaded" — so `residentModelsAtStart: []`, the correct and informative
value on a clean box, fails validation. Live-probed today, two further cases: `capabilities` is
**absent entirely** from several catalog entries (not empty — the key is missing), and
`loaded_context_length` only appears once a model is loaded, while the plan never says whether
`fingerprint.capture()` reads the catalog before or after `lms load`. *Fix:* distinguish
*absent* from *empty* per field in `REQUIRED_AUTO` (an empty resident list is valid; an empty
`modelKey` is not), and state capture ordering: catalog is read **after** load, snapshot residency
before and after.

**M-4 — Schema evolution silently deletes accumulated history, contradicting FR-3.**
§3.4 point 2: "a hand-edited or **older-schema** record must be excluded on read"; §5 test 3: an
unknown `schemaVersion` lands in `invalid`. The fingerprint carries `benchSchemaVersion`. So the
first time a required field is added, every previously stored result is quarantined out of every
comparison — against a requirement whose stated value is that "a new model's result lines up against
models I tested months ago" (FR-3, and the intent section). The exclusion is at least *visible*
(named in the `INVALID RESULTS EXCLUDED` block), which is why this is not a blocker. *Fix:* make the
required-field set a mapping keyed by `benchSchemaVersion`, so an older record is validated against
the contract it was written under; reserve `invalid` for records that fail their *own* schema. Add a
`model-bench migrate` path for genuinely breaking changes, and say in `README.md` that a schema bump
is a deliberate act.

**M-5 — `compare_report`'s signature cannot receive what AC-2 requires it to print.**
§3.4 point 2 says `compare` "prints an `INVALID RESULTS EXCLUDED` block naming each excluded `runId`
and the missing field", and S1's done-condition requires the excluded record be "named in the
report". But `load_history()` returns `tuple[list[RunResult], list[InvalidRecord]]` and
`compare_report(runs: Sequence[RunResult], *, pack: PackRef)` has no parameter for the second half.
*Fix:* `compare_report(runs, *, pack, invalid: Sequence[InvalidRecord] = ())`, and specify
`InvalidRecord` = `(runId | path, schemaVersion, missingFields: list[str], reason)`.

**M-6 — `ItemResult` and `aggregates` are the load-bearing shapes of the paired instrument and
AC-1's structural enforcement, and both are left as `…` / `dict[str, Any]`.**
S1: `@dataclass(frozen=True) class ItemResult: ...` and `aggregates: dict[str, Any]`. Pairing (FR-16)
requires a **stable item identity across runs** — for the tool-caller that is
`(conversationId, scriptId, replicate, turnIndex)` plus a `scoreable` flag per conditional count, and
`-ml` §4.3 additionally needs the paired-*n* intersection and the `asymmetry` count computable from
it. Separately, "`report.py` has no code path that aggregates two roles / produces a blended
percentage" (§3.5, §3.8.4) is claimed as *structural* enforcement, but an untyped `aggregates` dict
means the report renders whatever a pack put in it — the enforcement is back to convention. S1 is
built before any pack exists to constrain the shape (coordination U4 before U5/U6). *Fix:* specify
`ItemResult` fields in the plan, including the pairing key and per-count `scoreable`/`n_a` flags; and
replace `aggregates: dict[str, Any]` with a typed container whose per-role variants are closed
(`RetrievalAggregates`, `ToolCallAggregates`, …), so "no path emits a pooled figure" is a type fact.

**M-7 — The `tool_use` capability gate does not work on this box's actual catalog.**
§3.6: "`tool_use` capability is checked before a tool-caller run … a model lacking it is refused with
a named reason rather than scored 0". Live-probed today (Appendix A.2): `text-embedding-qwen3-
embedding-0.6b`, an `"type": "embeddings"` model, reports `"capabilities": ["tool_use"]` — it passes
the gate. And `google/gemma-3-4b` and `gemma-3-4b-vl-it-…` have **no `capabilities` key at all** —
they fail the gate, though nothing establishes they cannot emit tool calls. So the gate admits models
that certainly cannot and refuses models that plausibly can, which is the exact confound §3.6 exists
to avoid. *Fix:* gate on `type ∈ {llm, vlm}` **and** (`capabilities` absent **or** contains
`tool_use`), and record the raw `type`/`capabilities` in the fingerprint so the gate's decision is
auditable after the fact.

**M-8 — The CLI is the tool's entire user surface and no stage owns it; `model-bench attest` is
required by S3 and built by no stage.** Six commands appear across the plan — `run` (§3.7),
`compare` (§3.9), `validate` (§3.8), `models --tested` (§3.7), `index rebuild` (§3.5), `attest`
(§3.4 point 3) — with no consolidated specification of flags, exit codes, or output. S1 creates
`cli.py` but its done-condition names only `--negative-control`. S3's done-condition ("a stored
result with a **complete** fingerprint") is unreachable without `attest` having written `host.json`
first, and `host.json`'s schema is never given beyond four attested fields plus `lmsPath` plus the
two staleness fields. *Fix:* add a CLI-surface subsection to §3 (command, flags, exit codes — with
"the only non-zero exits are operational" from §1 made explicit per command), give `host.json` a
declared schema, and assign `attest` + `validate` to S1 and `run`/`compare`/`models`/`index` to their
producing stages in the done-conditions.

**M-9 — FR-11 is partially unaccounted for, and the headline latency has no stated measurement
source.** FR-11 names five things; "**prefill cost per 1k prompt tokens**" appears nowhere in the
plan (grep: 0 hits for `prefill`), and "decode tokens/sec is a diagnostic only" is never stated
(0 hits for `tokens/sec`, `decode`). Separately, "end-to-end turn latency (p50/p95)" is the headline
and the plan never says whether it is client wall-clock or LM Studio's own
`stats.generation_time`/`time_to_first_token` — which matters because §3.6 samples
`powershell.exe Get-Process` "on an interval during the run", measured today at **0.2–0.54 s per
invocation** across the WSL boundary (Appendix A.3), i.e. the RAM sampler perturbs the very metric it
runs beside. *Fix:* add `prefillMsPer1kPromptTokens` (derivable from `stats.time_to_first_token` and
`usage.prompt_tokens`) and label `tokens_per_second` diagnostic in §3.6's table; state that headline
latency is client-measured wall clock, and that host-RSS sampling is suspended for the duration of a
timed call (sample between turns, not during).

**M-10 — FR-21a's "format" layer has no data carrier and no done-condition.**
FR-21a scopes `chat-responder` to "latency, **format**, faithfulness to what the tool actually
returned". §3.8.5 maps format to "reply well-formedness and length discipline **as the pack declares
them**" — but the item shape it gives (`{question, context[], referenceAnswer, mustContain[],
mustNotContain[], mustAbstain}`) declares no format or length field, and S7's done-condition covers
only grounding + latency. Two of three parts ship. Separately, `referenceAnswer` is a silent
divergence from `-ml` §6.2, which specifies "a **checklist ground truth** rather than a reference
answer" and whose item shape has no such field — an unscored field that invites a future judge.
*Fix:* add the declared format constraints to the item/manifest shape (`maxWords`, `mustBeSingle
Paragraph`, `forbiddenPatterns`, whatever the pack actually wants) and to S7's done-condition, or
state in §3.8.5 that format is deferred with FR-21a's judged half and record it in `BACKLOG.md`.
Drop `referenceAnswer` or say what scores it.

**M-11 — `cleanThroughTurnH`'s `H` is derived where the note makes it declared, so the headline
metric can change meaning under a fixed name.** Plan §3.8.4: "`H` = the shortest script length, 4
here". `-ml` §3.3: "clean-through-turn-`H` rate (`H` **pack-declared**, default 4)". Under the plan's
derivation, adding one 3-turn script to a future pack version silently redefines the primary metric
from `cleanThroughTurn4` to `cleanThroughTurn3` — and while AC-3 would flag a *version* mismatch, it
flags it as changed data, not as "the headline now measures something else". This is the one failure
mode the tool exists to prevent. *Fix:* follow the note — declare `H` in `pack.json`, have
`validate` fail if `H > min(script length)`, and print `H` beside the metric name.

**M-12 — The `nlq-generator` pack declares no `primaryMetric`, violating the plan's own §3.9 point 3.**
Four packs name one (§3.8.1 MRR, §3.8.2 false-advance, §3.8.4 `cleanThroughTurnH`, §3.8.5 grounding);
§3.8.3's "Reported" list names none, so every number in that pack's report would carry
`exploratory — no significance claim` and the role could never produce a verdict. `-ml` §3.3 supplies
it: **Layer-1 exact-match rate**. *Fix:* state it in §3.8.3, using the note's name (see also m5 on
what Layer 1 actually is).

**M-13 — S6's done-condition is not independently satisfiable.** "**Done when:** step 4 reproduces
the documented contrast" — while R-3 states the reconstruction "will not be turn-for-turn identical"
and the contrast may not appear, with a bisect fallback but **no defined exit**. A stage whose gate
depends on an empirical outcome the plan itself says may not occur can be blocked indefinitely, and
under pressure the reconstruction gets edited until the contrast appears — which is fitting the
instrument to the expected answer. *Fix:* make the done-condition "step 4 is run and its outcome
recorded in `docs/test-reports/`; if the contrast does not appear, the bisect in R-3 is executed and
its result recorded, and the pack ships flagged `known-answer validation: not reproduced` in every
report it generates until it is." That keeps the honesty and unblocks the stage.

#### Minors

**m1 — S0's done-condition is unsatisfiable as a shell chain (actionable now, U2 is in flight).**
"`model-bench/setup.sh && model-bench/.venv/bin/python -m pytest -q` runs and passes with **zero
tests collected**". Verified today: pytest 9.1.1 exits **5** on zero collected tests, so the `&&`
chain fails. *Fix:* ship one smoke test in S0 (`tests/test_smoke.py::test_package_imports`), which
also matches `mcp-monitor/setup.sh`'s import-smoke-test precedent §4 S0 already cites.

**m2 — S1's negative control is tautological.** "`--negative-control` on **two copies of the same
run** reports not-distinguishable with b ≈ c" — two copies give b = c = 0 by construction and cannot
fail. `-ml` §9 and plan §5 test 19a mean two *independent runs* of the same model. Keep the S1 test
as a smoke check but say it is not the control; the control is test 19a.

**m3 — The `stats` fixture threshold is stated two ways and one is unachievable.** S1: "reproduces
the five `(a,b,c,d)` regression fixtures … **exactly**"; `-ml` §9.1: "exact match to **3 decimal
places**". The note's published bounds are 1 dp in pp ([3.2, 29.1]), so "exactly" cannot be asserted.
Relatedly, `min_detectable_difference` coded as `8/n` gives 20.0 pp at n=40 while `-ml` §3.2(e)'s
canonical verdict string says 19.1 pp — pin one and use it in both the function and the fixture.

**m4 — R-1's promised S2 probe is in no stage's done-condition.** §6 R-1: "**Verify during S2:** if
`lms ps --json` on a loaded model … exposes the load configuration, move `kvCacheSetting` from
attested to auto-captured." S2's done-condition covers `catalog()` and `chat()` only; the
`load`/`ps`/`unload` round-trip is §5 test 15, which is not gated on S2. Add the probe and its
recorded outcome to S2's done-condition, or R-1's mitigation is a note nobody executes.

**m5 — The `nlq-generator` executor is sized as "small"; the validation half is not, and one shape
rule is missed.** §3.8.3 costs `tools/exec.py` as "small, pure". Execution is indeed small
(`QueryRequest.matches` is `min_length=1, max_length=1` — no joins), but the plan also requires
"malformed-spec and schema-violation counts **separately** from wrong answers", which means
re-implementing `querygen`'s *validation* surface: `QueryFilter`/`QueryMatch`/`QueryRequest`'s
pydantic constraints (`extra="forbid"`, a six-op whitelist, `filters` max 4, `returns` 1–6 matching
projection/aggregate regexes, `order_by` shape, `limit` 1–50) plus `compile()`'s 165 lines of
allowlist and string→number coercion — in **stdlib only**, since §3.3 forbids pack modules from
importing anything but stdlib and `modelbench.tooling` (no pydantic). Also: `nlq_scoring.score_pair`
is *not* uniformly exact-match — `shape == "conflicting-facts"` scores by **subset containment**
(`nlq_scoring.py:207`), which affects 2 of the 40 items and is not mentioned in §3.8.3 or §5 test 9.

**m6 — `golden_retrieval.embeddings.json` does not isolate what §3.8.1 says it isolates.** §3.8.1 and
`-ml` §5.4 claim it separates "is my ranking code right" from "is my embedding call right". Inspected
today: it holds **38 query vectors only** (`{gr-NN: {model, vector}}`) — the 121 corpus vectors are
still computed live, so a wrong `documentPrefix` or a truncated corpus still contaminates the
self-test. *Fix:* have S3 write the 121 corpus vectors into the pack once (from the same
deterministic embed pass) as `corpus.embeddings.json`, giving a fully fixed ranking-path input.

**m7 — Two pack-safety mitigations under R-6 don't exist as described.** (a) "a `ruff` check … enforce
[s] it" — an import *allowlist* is not expressible under the plan's own `select = ["E","F","W","I"]`
(that is pyflakes/pycodestyle/isort; banned-import rules live in `TID`, unselected, and are a
denylist anyway). Only the `validate_pack` AST check is real; say so and drop the ruff claim.
(b) `validate` is never stated to be a **precondition of `run`**, so on a normal run
`Pack.load_tool_module()` executes pack code without the import check ever firing. Make `run` call
`validate_pack` first and fail closed. While there: state `subprocess.run([...], shell=False)` for
`lms.exe` and `powershell.exe` — model ids reach the argv (`mistralai/ministral-3-3b`) and a
`shell=True` slip is the only injection surface in the tool.

**m8 — §3.3's example manifest contradicts §3.8.4's sizing.** The example is the tool-caller pack
(`"packId": "tool-caller-shop-assistant"`) and carries `"sampling": {"repeats": 15, …}` — 15 is the
prior experiment's replicate count that §3.8.4 explicitly criticises; the design is 4 replicates ×
4 scripts × 3 shapes. There is also no `replicatesPerScript` field, which `-ml` §4.5 requires printed
next to every conversation-level *n*, and no `H` field (M-11). Fix the example and add both fields.

#### Nits

- `def ps(self) -> list[ResidentModel: ...]` (S2, `lmstudio.py`) is not valid Python.
- Several types are named but never defined: `PairedResult` (which determines whether `verdict()` can
  render `-ml` §3.2(e)'s three strings at all), `BootstrapResult`, `PackRef`, `InvalidRecord`,
  `PromptConfig`, `Conversation`, `ConversationTrace`, `DispatchRecord`. Worth one short type appendix.
- `reports/<pack-id>-<date>.md` collides on a same-day re-run, and `runId`'s `<modelSlug>` derivation
  is unspecified while real model keys contain `/` (`qwen/qwen3-4b-2507`). Both need one line each.
- §3.8.2's "false-suspend rate on the 30 `clear_advance` items" is the complement of `-ml` §7.3's
  "advance-recall" — same quantity, two names. Pick the note's.

---

### 3. What's solid

- **Grounding is exceptional.** Every checkable claim about `falkor-chat/server/tests/eval/` is
  correct to the item: golden-set sizes (38/85/10/40), guard tiers (40/30/15) and label marginals
  (30 True / 55 False), NLQ dataset and shape distributions (21/19; 9/8/7/6/4/4/2), relevance
  cardinality (36 of 38 with |R|=1), the pinned baseline to 4 decimals, `corpus_provenance.json`,
  `judge_calibration.json`'s `sameModelAsAgentUnderTest`, the seed-script line numbers, and
  `nlq-34`'s stored `{"items": [], "finding": "no matching data found"}` verbatim. The CPG
  blast-radius claim reproduces exactly. Nothing was pattern-matched.
- **D1 is the right call and is argued on the right axis.** "The two golden sets *must* diverge,
  because one tracks a live corpus and the other must freeze" is a genuinely better argument than the
  usual DRY reflex, and it answers the requirements' open constraint head-on. (Its stated mitigation
  needs B-2 fixed; the decision itself stands.)
- **The honesty architecture is the correct one.** Funnel table first, `k/n` on every rate, `n/a`
  tallies, paired-*n* as an intersection with an `asymmetry` count, the refusal to pool an 85-item
  guard accuracy, unanswerable NLQ items in a named bucket outside the denominator, the negative
  control promoted to a CLI mode. §4.3's laundering failure mode is the one that would have made this
  tool lie, and the plan builds against it deliberately.
- **The `-ml` note is followed, not paraphrased.** The five conclusions in §3.9 are accurate
  restatements, and the places the plan goes *beyond* the note (content-hash pack identity, the
  attestation staleness trip-wire, `refresh_golden.py` forcing a version bump) are additive and good.
- **Stage sequencing is well-reasoned** — S5's synthetic traces before S6's scripts, with the reason
  stated ("a script defect and a scorer defect are indistinguishable"), is exactly right.
- Live-probing the LM Studio surface rather than trusting documentation, and naming the two fields
  with no programmatic source instead of pretending they are captured, is the behaviour you want from
  a plan for a measuring instrument.

---

### 4. Open questions

These need the stakeholder or `data-scientist`, not a plan edit.

1. **Temperature vs. replicates for the tool-caller pack.** `-ml` §4.5: at `temperature: 0.0`,
   4 replicates per script have "an effective n closer to 12 than 48". The pack pins temperature 0
   *and* buys 4 replicates. Either the replicates should run at temperature > 0 (informative, but
   adds a second variance source and changes what FR-18's pinning means), or the budget should buy
   **12 distinct scripts × 1 run** instead of 12 × 4. This changes S6's data cost — the feature's
   long pole — so it should be settled before S5, not during S6. Recommendation: fewer replicates,
   more distinct scripts; `-ml` §4.5's whole argument points that way.
2. **The guard-judge `primaryMetric` was never taken back to the stakeholder.** `-ml` §10 open
   question 2 explicitly says the false-advance-vs-advance-recall choice "depends on which error is
   costlier in the product" and defers to the stakeholder; the plan adopts the note's recommendation
   silently (§3.8.2). The note's other two open questions were closed (FR-21a; backlog); this one is
   still open and pre-registration means it cannot be revisited after results exist.
3. **What does S3 count as done if the harness self-check lands below ~0.85 recall@10?** §3.8.1 and
   `-ml` §5.4 are careful that this is a bug detector, not a gate, and that a disagreement in either
   direction is uninterpretable — but S3's done-condition does not say what happens when it fires.

---

## Pass 2 — 2026-09-02

**Re-gated:** plan **v1.3** (`e69d687`), `-ml` note **v1.3** (`5aa7c83`), requirements as amended
(`aec25c0`), coordination ledger. Baseline: my Pass 1 findings above (3 blockers, 11 majors,
8 minors, 3 nits). Tree confirmed clean at re-dispatch; `dd78e70` is an unrelated session's
falkor-chat commit and was not read.

**CPG:** considered, not relevant — Pass 2 is a document-to-document consistency and arithmetic
re-gate against the two revised specs; no new structural claim about `falkor-chat` was made, and the
one Pass 1 relied on (§preamble's blast radius) was already independently re-run and is unchanged.

**Verdict: approve with suggestions.** All 3 blockers and all 11 majors are closed with mechanisms I
verified rather than accepted; every minor and nit is closed or explicitly withdrawn. Three new
findings, all bounded; **N-1 must go into the S1 dispatch brief**, and neither it nor the other two
justifies another plan revision cycle before S1 starts.

The quality of this revision is high in a specific way worth naming: the `data-scientist` corrected
two of its own published figures and one latent defect by exact search, and the `architect`
withdrew a claim (the ruff enforcement) and downgraded another (B-2's guarantee) rather than
defending them. I re-derived every number both documents changed, from scratch, and **all of them
reproduce exactly** (Appendix B).

### What I tested, on the coordinator's five points

**1. B-1's guard: structural for the shape the gate found, advisory for a caller who mislabels — and
the real closure is at pack validation, not in `stats.py`.** Verified `-ml` §3.4's six rules and the
plan's deferral to them (plan S1 restates none of the cluster-aware surface — the right call, since
restating it is what produced v1.1's divergence). Rules 2, 3, 4 and 5 are genuinely structural:
no-default keyword-only inputs, `n_effective: float` never `n: int`, `verdict()` raising on four
preconditions, and DEFF as a squared ratio. Rule 1 is narrower than it reads — `from_units` raising
on a repeated unit id fires only if the caller passes the **cluster** key as the unit id; 48 distinct
*conversation* ids from 12 scripts are unique and would be accepted. What actually closes that is
Rule 6 (`validate` fails a pack declaring `replicatesPerScript > 1`), which is why N-1 below is about
the pack contract rather than about `stats.py`. Judgement: the specific defect B-1 named is now
unbuildable; the general class is guarded, with one reachable gap.

**2. B-2's re-argued D1 stands, and the concession is correctly scoped.** I counted
`test_metrics.py` myself: **13 `recall_at_k` + 7 `mrr` = 20 cases, 18 value assertions + 2
`ValueError`**, of which exactly **6 are parametrize rows and 14 are literals in test bodies**, and
`check_regression` contributes exactly **6** excluded cases — the plan's numbers are right to the
case, including the "an implementer who ships three cannot call it done" count. `sourceGitSha`
`9650a385…` is the correct last-touching commit for that path. On D1 itself: my Pass 1 finding was
that the agreement test was the *only* mitigation. It no longer is, and the load is now carried by
the right thing — §3.1's point (ii), that the two implementations' outputs are **never compared to
each other as numbers by design**, since `-ml` §5.4 already forbids reading a difference against
`retrieval_baseline.json` in either direction. That reduces the duplication risk from "two numbers
that could silently disagree in a report" to "two places to fix a formula bug", which a 20-case
fixture plus `--check-origins` detects. D1 stands, and it now says what it can prove.

**3. B-3's forbid half does what it claims.** `FORBIDDEN_BY_ARM_KIND` enumerates all fourteen model
fields plus the four attested ones for `deterministic`; `validate()` branches on `armKind` and never
on presence; S1 done-condition 6 pins the exact shortcut case (`modelKey: "bm25"` added to an
otherwise-valid deterministic record → **fails on write**). The downstream consequences are decided
in the same place rather than deferred — `runId` segment, `models --tested` filtering to
`armKind == "model"`, never ranking two deterministic arms, skipping the attestation trip-wire.
Closed.

**4. Vocabulary is aligned — I found one residual divergence, and it is not the one you caught.**
Independently swept both documents: `verdictMetrics` (21/14), `headlineMetric` (24/9),
`falseAdvanceRate`/`falseSuspendRate` (4/4), `replicatesPerScript` (8/7) — consistent. All six
surviving `primaryMetric` occurrences are deliberate *retirement* references, not stale usage.
`advanceRecall` is handled identically in both (printed complement, carries no verdict). The α=0.025
consequence of guard-judge's two-member family propagated correctly into `-ml` §7.3's table
(17.5/21.9 and 23.3/28.7 pp — I recomputed both). The residual is **N-3**: `-ml` §4.6 still defines
`H` as *equal to* `min(script length)` where the plan makes it manifest-declared and validated
`≤ min` — and §7's "where the two disagree the note is right" rule turns that stale sentence into a
live regression of M-11.

**5. M-1: confirmed, your reading is right and mine was wrong on the severity.** Computed the five
MOVER-D fixtures under both constants: the worst bound shift across all ten endpoints is
**3.02 × 10⁻⁴ pp** (Appendix B.2), and all five reproduce the note's published 1-dp bounds under
either. So `1.96` was a typographic rounding, not a numerical defect, and my "breaks the fixtures"
framing was wrong. The pin is still worth having for exactly the reason v1.3 gives — equality
assertions at 1e-12 — and v1.3 makes it a module-level `_Z_95` with keyword-only `z`, which is the
right shape. Disposition below reflects the corrected rationale.

### New findings

**N-1 (major) — the analysis-unit id is unspecified, which is the one reachable gap in B-1's
closure.** `PairedOutcomes.unit_ids` is "the cluster keys, one per row" (`-ml` §3.4 Rule 1) and
`ItemResult.pairingKey` for the tool-caller is `(scriptId, replicate, turnIndex)` (plan S1) — but
**nothing in either document says which component becomes the unit id for a given verdict metric.**
At 12×1 `scriptId` and conversation id are 1:1 so no shipped pack can expose it. A later pack that
raises replicates is the reachable case, and it has two independent ways through: `validate_pack`'s
stated failure conditions (`verdictMetrics`, `headlineMetric`, `H ≤ min(script length)`, import
allowlist) do **not** include a cross-check that `conversations.jsonl` actually contains
`scripts × replicatesPerScript` rows with each `scriptId` appearing exactly that many times — so a
pack can declare `replicatesPerScript: 1` and ship four conversations per script — and if the unit id
is then the conversation id, `from_units` sees 48 unique ids and accepts. Rule 1's guard is bypassed
by a naming choice nobody was told to make. *Fix, all three in the S1 brief:* (a) the manifest
declares the analysis unit per verdict metric (or once, as `sampling.analysisUnit`), and `stats`
takes it from there rather than from the caller; (b) `validate_pack` asserts the row-count identity
above; (c) S1 done-condition 5's synthetic clustered fixture asserts that the unit id used is the
**cluster** key — otherwise that fixture passes while testing nothing.

**N-2 (major) — `basis: "by-construction"` is an unverified attestation, and the note's own
evidence-producing mechanism for it is built by no stage.** `-ml` §3.4 Rule 4 makes McNemar valid
*only* when `design_effect == 1.0 and basis == "by-construction"`, and §4.5.1 grants that basis to
12×1 because each script contributes one observation. But §4.5.1(iii) states that the same design
makes run-to-run variability **unmeasurable**, that LM Studio at temperature 0 is "near-deterministic
but not guaranteed bit-deterministic", and prescribes the fix: "**a determinism probe — re-run 2 of
the 12 scripts a second time, once per model, and report whether the outcome vector is identical …
Cheap mitigation, and it should be built.**" It is not built. Grep of the plan: **zero** occurrences
of the probe under any name; no stage creates it, no done-condition mentions it, and §3.8.4's budget
does not carry its two extra conversations. So the one input that decides whether McNemar may
decide the feature's flagship metric is asserted rather than measured. *Fix:* assign the probe to
S6 (data) with S5's scorer, keep it diagnostic and outside `n` as the note requires, and wire the
outcome to `basis` — a non-identical probe degrades `basis` from `by-construction` to `assumed`,
which via Rule 4 automatically moves McNemar out of the decision seat and the cluster-bootstrap CI
into it. That closes the loop with no new statistics.

**N-3 (major) — `-ml` §4.6 still derives `H` from the data, and §7's conflict rule makes the note
authoritative.** Plan §3.8.4 fixes M-11 correctly and emphatically: `H` is
`metrics.cleanThroughTurnH.H`, "never derived from the data", `validate` fails
`H > min(script length)`, and the report prints the resolved name (`cleanThroughTurn4`). But `-ml`
§4.6's bullet still reads "where `H = min(script length)` across all conditions in the pack". Those
are different contracts — declared-and-bounded versus derived — and plan §7 says "where the two ever
appear to disagree **the note is right**". An implementer applying that rule lands back on derived
`H`, which is precisely M-11: adding one shorter script silently redefines the headline from
`cleanThroughTurn4` to `cleanThroughTurn3` under an unchanged metric name. The plan's version is the
correct one; the fix is one clause in `-ml` §4.6 (`H` is pack-declared and validated
`H ≤ min(script length)`), owner `data-scientist`.

**N-4 (minor, already known to the coordinator, folded in for whoever revises next)** — plan §7's
version-pairing block says "this plan **v1.3** is aligned to the note **v1.2**"; the note is v1.3.
The standing-obligation sentence beside it is the right instrument, so the stale token undercuts
exactly the mechanism it introduces.

### Disposition of Pass 1 findings

| # | Disposition | Evidence rechecked |
|---|---|---|
| **B-1** | **Fixed** | `-ml` §3.4's six binding rules; plan S1 defers the whole cluster-aware surface to them and adds DC-4/DC-5 to detect a non-conforming implementation — including a synthetic clustered fixture kept precisely because no shipped pack is clustered any more. Residual: N-1. |
| **B-2** | **Fixed, with the guarantee honestly downgraded** | 20-case sha-pinned fixture; I re-counted the origin and got 13+7 = 20 = 18+2, 6 parametrized / 14 inline, 6 excluded — exact. D1 re-argued on §3.1(ii); stands. |
| **B-3** | **Fixed** | `FORBIDDEN_BY_ARM_KIND`; S1 DC-6 pins the `modelKey: "bm25"` case failing on write. |
| **M-1** | **Fixed; my rationale was wrong** | Module-level `_Z_95`, keyword-only `z`. Numerically it was a ≤3.02×10⁻⁴ pp rounding (Appendix B.2), not a defect — the pin is for equality-assertion reproducibility. |
| **M-2** | **Fixed** | S1 DC-3 now states the 40/40 vs 34/40 verdict correctly *and* names §5 test 6 as the thing it must not diverge from again. |
| **M-3** | **Fixed** | `REQUIRED_NONEMPTY` / `REQUIRED_PRESENT` tiers, `null` invalid in both, and capture ordering made contractual (residency before load, catalog after). DC-1 tests all three states. |
| **M-4** | **Fixed** | `REQUIRED_BY_SCHEMA` keyed by `benchSchemaVersion`; `invalid` reserved for a record failing *its own* schema or declaring a future one; `SCHEMA VERSIONS IN THIS COMPARISON` banner; `model-bench migrate`. DC-7. |
| **M-5** | **Fixed** | `compare_report(runs, *, pack, invalid=())`; `InvalidRecord` fully specified with a `reason` enum. |
| **M-6** | **Fixed** | `ItemResult` specified incl. `pairingKey`/`scoreable`; `Aggregates` is a closed per-role union, so the no-blended-figure claim is a type fact. (N-1 is the one part of the pairing contract still open.) |
| **M-7** | **Fixed** | Gate is now `type ∈ {llm, vlm}` **and** (`capabilities` absent **or** contains `tool_use`); `modelType`/`modelCapabilities`/`modelCapabilitiesPresent` recorded in the fingerprint; S2's done-condition tests all three real catalog entries that break the naive rule. |
| **M-8** | **Fixed** | §3.6a consolidated CLI table with per-command owning stage; `host.json` schema in §3.4.4; `attest` assigned to S2 *with the reason* (S3's completeness depends on it). |
| **M-9** | **Fixed** | `prefillMsPer1kPromptTokens` added with its formula; `tokensPerSecond` labelled diagnostic; headline pinned to **client wall clock** as a stated decision. |
| **M-10** | **Fixed** | Item gains a `format` block (`maxWords`, `mustBeSingleParagraph`, …) scored separately and never pooled with grounding; `referenceAnswer` removed as diverging from `-ml` §6.2. |
| **M-11** | **Fixed in the plan; reopened by the note** | See N-3. |
| **M-12** | **Fixed** | `verdictMetrics = ["layer1ExactMatchRate"]`, `headlineMetric` set, using the note's Layer-1 name. |
| **M-13** | **Fixed** | S6 now completable either way: the bisect is executed and recorded, and the pack ships flagged `known-answer validation: not reproduced`. The added rationale ("fitting the instrument to the expected answer") is the right reason. |
| **m1** | **Fixed** | S0 ships `tests/test_package.py` asserting `modelbench.__version__` against `project.version` — one real test, so the `&&` chain no longer exits 5, and the assertion is load-bearing for `benchVersion`. |
| **m2** | **Fixed** | DC-9 labels the two-copies control a smoke check *in the test's own docstring* and names §5 test 19a as the real control. |
| **m3** | **Fixed, and better than asked** | Rule 3 replaces `8/n` with exact bisection ceilinged to 0.1 pp. I reproduced the entire §7.1 table and the 0.798-vs-0.8023 rounding argument exactly (Appendix B.1). |
| **m4** | **Fixed** | R-1's probe is now inside S2's done-condition with both outcomes acceptable and "silence does not" stated. |
| **m5** | **Fixed** | §3.8.3 names Layer 1, the `conflicting-facts` subset-containment exception, and the stdlib-only re-implementation of the pydantic validation surface. |
| **m6** | **Fixed** | `corpus.embeddings.json` written into the pack from the same deterministic pass; S3 done-condition 4. |
| **m7** | **Fixed** | Ruff claim **withdrawn** explicitly; single mechanism is `validate_pack`'s AST walk; `run` calls it and fails closed; `shell=False` stated. |
| **m8** | **Fixed** | `sampling: {scripts: 12, replicatesPerScript: 1, seed}`; `repeats: 15` gone; `H` in the `metrics` block. |
| Nits | **Fixed** | `def ps(self) -> list[ResidentModel]`; Appendix A defines every named type (and withdraws v1.2's invented `PairedResult`); `modelSlug` rule and timestamp format given. Report-filename same-day collision: still unaddressed, and I am content to drop it. |

### Open questions carried forward

Pass 1's OQ-1 (temperature/replicates) and OQ-2 (guard-judge metric) were both taken to the
stakeholder and settled — 12×1, and two co-equal verdict metrics with no headline. OQ-3 (what S3
does when the self-check fires) is settled as diagnostic-never-a-gate. Nothing from Pass 1 remains
open. The two items parked for before S6 (48 distinct scripts; the requirements' "~15 pp") are the
coordinator's and out of my scope — I note only that `-ml` §4.5.3's reversal trigger, *"the first
tool-caller comparison that returns 'not distinguishable' with an observed difference in the
15–50 pp band"*, is a well-chosen one: it converts the question into evidence rather than a guess.

---

## Appendix — Pass 2: arithmetic, independently recomputed

Every figure the two documents changed in v1.2/v1.3 was re-derived from scratch (exact McNemar
rejection region, `math.comb`, no library statistics) rather than checked by eye.

**B.1 — `-ml` §3.4 Rule 3 and the resolving-power tables.** Exact MDD₈₀ under the nested
alternative (π_c = 0, reject when `b ≥ b_min(α)`), bisected on δ, ceilinged to 0.1 pp:

| n | exact δ (pp) | ceil 0.1 | note's published | `8/n` (the retired mnemonic) |
|---|---|---|---|---|
| 12 | 57.794 | **57.8** | 57.8 | 66.7 |
| 20 | 36.646 | **36.7** | 36.7 | 40.0 |
| 30 | 25.075 | **25.1** | 25.1 | 26.7 |
| 38 | 20.009 | **20.1** | 20.1 | 21.1 |
| 40 | 19.046 | **19.1** | 19.1 | 20.0 |
| 48 | 15.972 | **16.0** | 16.0 | 16.7 |
| 60 | 12.857 | **12.9** | 12.9 | 13.3 |
| 85 | 9.142 | **9.2** | 9.2 | 9.4 |
| 120 | 6.509 | **6.6** | 6.6 | 6.7 |

All nine reproduce. The rounding argument reproduces too: power at n=40 is **0.7980** at 19.0 pp and
**0.8023** at 19.1 pp, so ceiling — not nearest — is what makes the printed sentence true.
`b_min(c=0)` is **6** at α=0.05 and **7** at α=0.025, so the floor is `6/n` and `7/n` respectively,
as Rule 2 states. Guard-judge at α=0.025: floor 17.5 / 23.3 pp, MDD₈₀ **21.9 / 28.7 pp** — both
match §7.3. Boundary tier at n=15, α=0.05: **47.6 pp**, matching the correction of v1.1's "53 pp".
McNemar exact at b=12, c=0 is **p = 0.000488**, matching §4.5.3's 0.00049. The v1.1 "~65 pp"
correction to 57.8 pp is confirmed as an `8/n` artefact.

Rule 5's identity also holds independently: with m = 7 turns and ρ = 1, `DEFF = 1 + (m−1)ρ = 7`,
width ratio `√7 = 2.646` (v1.1's "≈2.6"), and `n_eff = 280/7 = 40` — exactly the conversation count.
The self-correction is right, and the ρ=1 unit test it prescribes is the correct guard.

**B.2 — M-1, the `z` constant.** MOVER-D on the five `(a,b,c,d)` fixtures under
`z = 1.959963984540054` versus `z = 1.96`:

| fixture | z = 1.9599… | z = 1.96 | max shift |
|---|---|---|---|
| (34,6,0,0) | [3.176, 29.072] | [3.176, 29.073] | 3.02e-04 pp |
| (30,6,0,4) | [3.851, 27.703] | [3.850, 27.703] | 2.61e-04 pp |
| (33,6,1,0) | [−0.986, 26.858] | [−0.987, 26.858] | 3.01e-04 pp |
| (20,8,2,10) | [0.171, 28.779] | [0.171, 28.779] | 2.64e-04 pp |
| (72,10,2,1) | [1.480, 18.213] | [1.480, 18.213] | 1.85e-04 pp |

All five reproduce the note's published 1-dp bounds under either constant; worst endpoint shift
**3.02 × 10⁻⁴ pp**. Contested finding withdrawn, disposition corrected above.

**B.3 — B-2's fixture count, from the origin file.** `test_metrics.py` assertions exercising the two
functions: `recall_at_k` — 8 single-assert bodies, one 3-row parametrize, and two bodies carrying
**two** asserts each (`hit_outside_top_k_window`, `handles_retrieved_shorter_than_k`) = 12 value
assertions + 1 `pytest.raises` = **13**. `mrr` — a 3-row parametrize plus 3 single-assert bodies = 6
value + 1 raise = **7**. Total **20 = 18 + 2**; parametrized **6**, inline **14**;
`check_regression` **6**, all excluded. Matches plan §3.1 point 2 exactly.
`git log -1 -- …/test_metrics.py` → `9650a3858b9d5c4e7e934f977839fc1a61c84b1b`, the recorded
`sourceGitSha`.

## Appendix — Pass 1

### A.1 — Verification performed

| Claim | Source | Result |
|---|---|---|
| eval golden-set sizes / compositions / cardinalities | recounted from the four `.jsonl` files | exact match to plan §2.1 and `-ml` §2 |
| `retrieval_baseline.json`, `corpus_provenance.json`, `judge_calibration.json` | read | exact match |
| `metrics.py`/`nlq_scoring.py` symbols + line counts | `grep`/`wc` | exact match (96 / 267 lines) |
| `_LlmGuardJudge`, `CATALOG_SCHEMA`/`KNOWLEDGE_BASE_SCHEMA`, `DatasetSchema`, tools.py prompt | read | present as described |
| seed-script literals (`_CORPUS`:80, `_CORPUS`:127, `CATALOG` 15 products) | read | present; `CATALOG` is a Python heredoc *inside* `seed_catalog.sh`, not a shell array — copying it is a Python parse, worth one word in §3.8.3 |
| `nlq-34` scored incorrect with `{"items": [], "finding": …}` | `nlq_eval_results.json` | verbatim match |
| CPG blast radius (8 callees / callers all in `tests/eval/`) | `cpg_falkorchat`, re-run | reproduces; `METHOD` 2 862 vs 1 222 exact |
| `salesperson-tool-reliability-ml.md` §8.1/§8.2 (A 9 / B 7 / C 4 turns, n=40/280, ~1.3 s/turn, turn-4 collapse 39/40) | read | exact match |
| `mcp-monitor/pyproject.toml` shape, falkor-chat `addopts = '-ra -m "not live"'` | read | exact match, incl. the `pytest>=9.1,<10` / `ruff>=0.14,<0.15` pins §4 S0 quotes |
| pytest exit code on zero collected tests | ran `pytest -q` on an empty tests dir, pytest 9.1.1 | **exit 5** (finding m1) |

### A.2 — LM Studio catalog, live-probed 2026-09-02

`GET http://localhost:1234/api/v0/models` — 19 models, confirming §2.3. Both
`mistralai/ministral-3-3b` and `mistralai_ministral-3-3b-instruct-2512` present, both `Q8_0`,
different `publisher` (R-8 grounded). `qwen/qwen3-4b-2507` and
`text-embedding-qwen3-embedding-0.6b` both installed, so S3's and S6's validation targets exist.

Two facts the plan does not have (finding M-7):

```json
{"id": "text-embedding-qwen3-embedding-0.6b", "type": "embeddings",
 "quantization": "Q8_0", "state": "not-loaded", "max_context_length": 32768,
 "capabilities": ["tool_use"]}                      <-- an embeddings model advertising tool_use

{"id": "google/gemma-3-4b", "type": "vlm", "quantization": "Q4_K_M",
 "state": "not-loaded", "max_context_length": 131072}
                                                     <-- no "capabilities" key at all
```

`loaded_context_length` is absent from every entry while `state == "not-loaded"` (finding M-3).

### A.3 — `powershell.exe` cost across the WSL boundary (finding M-9)

```
$ time powershell.exe -NoProfile -Command "Get-Process | ... | ConvertTo-Json"   # cold
real 0m0.539s
$ /usr/bin/time -f "%e s" powershell.exe -NoProfile -Command "(Get-Process | Measure-Object WorkingSet64 -Sum).Sum"
0.21 s  /  0.20 s  /  0.18 s        # warm, three consecutive
```

Against `-ml` §4.5's measured ~1.3 s/turn, an interval sampler at this cost is a
double-digit-percentage perturbation of the wall-clock latency it is sampled beside.
