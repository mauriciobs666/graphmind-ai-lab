# Small-LLM benchmarking tool (`model-bench/`) — plan review

> **Status:** active · **Owner:** `analyst` · **Tracks:** — · **Reviews:** `docs/plans/small-model-benchmarking.md`

## Pass 1 — 2026-09-02

*(Current verdict is Pass 3's, below: **needs changes** on plan v1.8 / note v1.8. Pass 2 gated plan
v1.3 with **approve with suggestions**. Pass 1 gated plan v1.1 and is kept intact — the two passes are meant to be read together, and Pass 2's
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

## Pass 3 — 2026-09-03

**Re-gated:** plan **v1.8** (`aebb611`) against note **v1.8** (`27501c9`, read as
`git show aebb611:docs/plans/small-model-benchmarking-ml.md` because `data-scientist` may be editing
the working-tree copy), the requirements as amended (`afe4aef`), and the **shipped S1 tree** where
v1.8 asserts consequences for it. Weighted to v1.8's three changes — the fingerprint's source of
truth (§2.5, §3.4.4a), the JIT warm-up and contamination guard (§3.6, R-14), and P4-4's return to S1
(§4 S1 DC-10, §5 test 11c) — plus a sweep of every section v1.8's own revision line says it touched.
Findings carry the prefix **`G3-`** (plan-gate Pass 3); `P3-*`/`P4-*` belong to
`docs/reviews/small-model-benchmarking-impl.md` and are not reused here.

**CPG:** considered, not relevant — no `model-bench` CPG exists (`cpg_model_bench` is absent from the
loaded-graph list; `cpg_falkorchat` is loaded but v1.8 makes no new structural claim about
`falkor-chat`). The S1 grounding checks below were done by reading `model-bench/modelbench/*.py`
directly, which is a 9-module tree.

**Verified live, not assumed.** Every §2.5/§3.6 claim about the LM Studio surface was re-probed on
this box today, read-only, with no model loaded or unloaded: `GET /api/v0/models` → 19 models,
1.4–1.6 ms, all `state: "not-loaded"`, exactly ten keys and `capabilities` on 15 of 19;
`GET /v1/models` → `{id, object, owned_by}` only; `command -v lms` → exit 1. All reproduce
(Appendix, C.1). The `/api/v0/embeddings` response shape was checked against LM Studio's own v0 REST
docs (C.2).

**Verdict: needs changes.** 2 blockers, 6 majors, 4 minors, 1 nit.

**Yes — something in v1.8 must change before S2 is dispatched**, and two of the findings land inside
the P4-4 unit a `tdd-engineer` is executing *right now* (**G3-6**, **G3-7**): DC-10's selector as
written cannot be evaluated, and its counting call has an uncaught raise path. Those two are the
urgent ones. The two blockers (**G3-1**, **G3-2**) are S2/S3 scope and must be settled before S2 is
briefed, not during it.

The three v1.8 changes are, on their merits, right — the reasoning in §3.4.4a is the best-argued
section in the document and I endorse its conclusion (see *Judgement*, below). The defects are not
in the decisions; they are, again, in **what the instrument says it will do with data it will not
have**: a fingerprint contract written as if one endpoint supplied fields that three sources supply,
and a timing guard specified over one of four timing fields.

### Judgement on the three questions put to this gate

**1. Is "refuse the run" right when `/api/v0/models` is absent? Yes — and it costs less than the
plan argues it does.** §3.4.4a's own third bullet is the decisive one: a `/v1`-populated record
carries `modelKey` and fails `validate()` anyway, so "refuse" is not a policy choice layered on top
of the design, it is what the design already does — the choice is only *where* the refusal happens
and *what it says*. Refusing at the probe, before the first model call, converts a
twenty-minute run into a two-millisecond error with a message that names the actual cause. I would
have flagged the opposite decision. The `/v1/models` discriminator earns its ten lines for exactly
the reason given: "reachable but not LM Studio" is the case a person hits and cannot diagnose.

**What is *not* covered is the generalisation.** §3.4.4a is written as a source-of-truth section but
governs one of the **four** sources the auto-captured half actually has: the catalog, the chat
route's `runtime` object, `host.json`, and the process environment. The two blockers below are both
instances of that gap, not disagreements with the rule.

**2. Is the pre-designed reversal sufficient? The shape is right; the costing is not.** See **G3-8**.
Making the required-field set a function of provider is the correct pattern, but as stated it hides
three costs: the discriminator must be *declared*, not observed; eight of the nine fields have no
analogue on a hosted provider, so provider B's required set collapses to roughly `/v1/models`'s
three fields — the reversal **segregates** strong-fingerprint records from weak ones rather than
preserving fingerprint strength; and nothing in §3.4.3 or `compare_report` labels or refuses a
cross-provider comparison, where the `SCHEMA VERSIONS IN THIS COMPARISON` banner is the obvious
precedent sitting one paragraph away.

**3. Does the anti-contamination guard leave a path? Three.** R-14 names one of them honestly (a
reload that begins and ends inside a single timed call) and I accept that disclosure. The other two
are **G3-3** (the guard nulls `latencyMs` and nothing else, while three more timing figures come from
the same contaminated response and print without denominators) and **G3-4** (the guard's first
comparand is unspecified, and the literal reading drops item 1's latency on every cold-start run).
**G3-5** is adjacent and arguably worse than any of them: a scored call that hits
`requestTimeoutSeconds` has no specified disposition at all, so `latencyMs = 120000` is a legal
implementation and would enter the p95 as a measurement.

**On the drift check — clean.** I searched v1.8's new text for every restatement class v1.7
withdrew: no α, no α count, no `z` literal, no verdict string, no judge threshold, no κ figure, no
rule count, no reproduction of R-9's three numbers. The only `1.96` and `κ = 0.21` occurrences are
the pre-existing negative statement in §4 S1 / §5 test 6 and §2.1's context finding, both cited to
the note. §3.9 and §7's rule 2 are intact. The one drift that *did* occur is the mirror image and is
minor: §7's recap was swept and its owning section (§6 R-13) was not — **G3-11**.

**On the `lmsCliCommit` → `residencySource` claim — the arithmetic is right and the edit list is
short.** I counted `_MODEL_SCHEMA_1` in `model-bench/modelbench/fingerprint.py`: 26 auto + 4 attested
= 30, and the swap keeps it at 30, exactly as §3.4.2 says. "Free only because `results/runs/` does
not exist yet" is **true** — `model-bench/results/` does not exist, nothing under it is tracked, and
`git log --all -- 'model-bench/results/*'` is empty. The *number of edit sites* is understated —
**G3-9**.

### New findings

**G3-1 (blocker) — the `embedder` arm cannot produce a storable fingerprint, and its warm-up call
cannot be issued. S3, the plan's "first end-to-end result", is unbuildable as specified.**
`runtimeName` and `runtimeVersion` are `REQUIRED_NONEMPTY` for every `model` arm (§3.4.2;
`fingerprint.py:75-77`), and their only cited source is the `runtime` object on
`POST /api/v0/chat/completions` (§2.3, §3.6's table). §3.8.1's embedder mechanism issues **only**
`POST /api/v0/embeddings`, which returns no `runtime`, no `model_info` and no `stats` (LM Studio's
own v0 REST docs — Appendix C.2). §3.4.4a forbids a fallback, a substitute and a default; §3.4.5
point 1 makes `store()` refuse. So every embedder run refuses on write, and S3's done-condition ("a
stored result with a complete fingerprint") is unreachable. §3.6's warm-up compounds it: it is
specified as one chat completion carrying "the pack's system prompt plus a fixed probe message at
`temperature: 0, max_tokens: 10`" — an embeddings model serves no such route and the embedder pack
declares no system prompt, so the mandatory per-arm warm-up has no instantiation for the pack S2
exists to unblock. *Fix, both halves in §3.6/§3.4.2:* make `warm_up` **call-surface aware** (one
short `POST /api/v0/embeddings` for a `type: "embeddings"` arm, timed the same way), and make the
required-field set a function of **call surface** as well as `armKind` — §3.4.4a already names that
discriminator pattern for a second provider; this is the same pattern one level earlier, and it is
what lets `runtimeName`/`runtimeVersion` be *required for a chat-surface arm and absent by contract
for an embeddings-surface arm* instead of silently missing. Partly inherited from v1.7 — but v1.8 is
the version that makes the warm-up mandatory for every arm and forbids every escape, so it is v1.8's
to close.

**G3-2 (blocker) — §3.6 says the catalog is the fingerprint's auto half "and its only source"; it is
not, and no section says where the runtime identity comes from or when the staleness trip-wire
fires.** I re-probed `GET /api/v0/models`: ten keys per entry, no `runtime`, no size, no
`loaded_context_length` while nothing is loaded (Appendix C.1). §3.6's `catalog()` row nonetheless
reads "the fingerprint's auto half, and its **only** source (§3.4.4a)" — the sentence an implementer
of `fingerprint.capture()` reads. It is false for `runtimeName`/`runtimeVersion` (chat route),
`temperature`/`maxTokens` (config), `pythonVersion`/`hostOs` (environment) and the four attested
fields. Two live consequences, not stylistic ones. **(i)** §3.4.4a's guarantee is *"refuses before
its first model call and writes nothing"*; its converse — "if the catalog answers, the fingerprint
can be completed" — is what an implementer will assume, and it is untrue. **(ii)** §3.4.5 point 3's
staleness trip-wire compares `runtimeName`/`runtimeVersion`, and §3.4.4 says `host.json` is "read at
the start of every model run", but the earliest those two values exist is *after* the warm-up, whose
response §3.6 says is "discarded" (Appendix A retains it as `LoadResult.discardedResponse`, without
saying it is the runtime source). So the trip-wire either fires after the run has already paid the
21 s load, or — if wired to the first scored call — after items have been consumed, which is the
expensive-refusal case §3.4.4a took care to avoid for residency. *Fix:* one ordering paragraph in
§3.4.4a naming all four sources and pinning the sequence: start probe → `residentModelsAtStart` →
warm-up → read `runtime` from the warm-up response → **trip-wire check here** → catalog re-read for
`loadedContextLength` → first scored item. Correct §3.6's table row to "the catalog half".

**G3-3 (major) — the contamination guard nulls `latencyMs` and nothing else; three more timing
figures come from the same contaminated response and print with no denominator.** §3.6's guard sets
`ItemResult.latencyMs = None` for a load-contaminated item, and the p50/p95 line prints
`latency n = X of Y`. But §3.6's own FR-11 table sources `ttftMs` from `stats.time_to_first_token`,
`prefillMsPer1kPromptTokens` from that same figure over `usage.prompt_tokens`, and `tokensPerSecond`
from `stats.tokens_per_second` — all read off the *same* response, none excluded by the guard, none
printed with a denominator. Whether LM-Studio-side TTFT includes the JIT load is not established:
§2.5 recorded the cold call's **wall clock** (21.068 s) and did not record what `stats` said on it,
so the plan cannot currently claim either way. *Fix, and it is free:* the warm-up against a
non-resident model **is** a known cold load, so record its `stats` beside its wall clock and settle
the question in S2's R-1 probe (same call, one more assertion). If TTFT includes the load, the guard
must null `ttftMs`/`prefill` on the same items and their aggregates need their own `n = X of Y`. The
same measurement gives R-14's acknowledged residual a detector it currently lacks: a reload inside a
single timed call shows up as `wallClockMs − (ttft + generation_time) ≈ 20 000 ms`, which is an order
of magnitude outside anything else and needs no new probe.

**G3-4 (major) — the guard's first comparand is unspecified, and the literal reading discards item 1's
latency on exactly the runs the feature exists for.** §3.6 says `runner` probes `residency()`
"between items" and nulls the latency of "any item whose preceding snapshot did not show the model
resident". Item 1 has no *between-items* predecessor; the only snapshot preceding it is
`residentModelsAtStart`, which §3.4.2 pins as taken **before the warm-up** — i.e. on any cold-start
run it is `[]` **by construction**, because the warm-up is the load. Read as written, every
cold-start run marks its first item `latencyMs = None` and prints `latency n = 37 of 38` with no
cause a reader can recover, and §5 test 15b's `X < Y` assertion goes green on the wrong mechanism.
*Fix, one clause in §3.6:* the guard's baseline for item 1 is a residency probe taken **after the
warm-up returns**, never `residentModelsAtStart`; and test 15b asserts a full-length latency sample
(`X == Y`) on a clean cold run with no mid-run reload, which is the case that currently fails.

**G3-5 (major) — a scored call that hits `requestTimeoutSeconds` has no specified disposition, so a
fabricated 120 000 ms is a conforming implementation.** §3.6 introduces `requestTimeoutSeconds`
(120) and says it "**is** meant to fire on a hung call" — and the plan then never says what happens
when it does. Grep of v1.8: seven occurrences of "timeout", none stating the item's outcome, its
`latencyMs`, or whether the run continues. The three plausible readings differ materially in what the
report *prints*: `latencyMs = 120000` puts a censored observation into the p95 as if it were a
measurement (and it is the largest one, so it lands in exactly the statistic it corrupts);
`latencyMs = None` is honest but silently widens the `Y − X` gap the guard is supposed to explain;
aborting the run discards everything scored so far. *Fix:* state it in §3.6 — a timed-out scored call
is scored per the pack's rule (a `fail`/`parse_failure`, not a skip), its `latencyMs` is **`None`**
because a censored observation is not a measurement, and the report prints the timeout count as its
own line beside `latency n = X of Y` so the two causes of exclusion stay distinguishable. State the
warm-up's own timeout disposition too (exit `3`, nothing written, message naming
`--first-call-timeout`).

**G3-6 (major, in the unit being implemented now) — DC-10's metric selector names two different
vocabularies and, taken literally, never matches, so the cross-check silently never fires.**
§4 S1 DC-10 scopes the check to "each `BinaryMetric` in the pack's `verdictMetrics` family **whose
`unit` is the pack's analysis unit**". Those are two disjoint namespaces in the shipped code:
`BinaryMetric.unit` is a denominator noun — `"item"`, `"conversation"`, `"query"`, `"turn"`,
`"call"` (`results.py:54-71`; `-ml` §3.4's `unit_kind` is the same set) — while `PackRef.analysisUnit`
is a `pairingKey` **component name**, e.g. `"scriptId"` (`packs.py:87-100`, and DC-5(c)'s own
fixture). `metric.unit == pack.analysisUnit` is therefore never true. The predicate the plan means
already exists and is already used: `metric.unit == unit_kind_for_role(pack.role)`
(`report.py:377`). *Fix, three clauses in DC-10 and test 11c:* (a) name that predicate literally;
(b) the 11c fixture must pin a `PackRef.role` whose unit kind is `"item"` — `guard-judge`,
`nlq-generator` or `chat-responder` — or the fixture's `unit="item"` metric is out of scope and the
test passes while testing nothing, the DC-5(c) failure shape again; (c) say **explicitly** that a
turn-pooled or call-pooled `BinaryMetric` is *not* cross-checked and why (its denominator is not an
item count), because the P4-4 defect — a rate printed for a metric no item declares scoreable —
remains printable for a pooled metric, without an interval but with the number.

**G3-7 (major, same unit) — DC-10's counting call has an uncaught raise path that reproduces the
P4-5 shape DC-10 explicitly rejects.** DC-10 counts items "for which `scored_outcome(metric) is not
None`". `ItemResult.scored_outcome` does not return `None` for the sibling malformation: a metric
declared `scoreable: True` with no entry in `counts` raises `IncompleteItemRecord`
(`results.py:144-149`). Nothing catches it — `load_history` validates fingerprints, not
item/aggregate consistency, and `cli.py:_cmd_compare` catches only `PackConfigError`
(`cli.py:144-150`) — so it escapes `compare_report` as a traceback with exit 1, outside §3.6a's
closed exit-code set `{0,2,3,4,5}`. That is precisely the failure DC-10 rejects "raising" for, arriving
through the check's own implementation. *Fix:* DC-10 states that the cross-check treats
`IncompleteItemRecord` as a **mismatch** — exclude the arm, name it in the `INVALID RESULTS EXCLUDED`
block with the offending item and metric — and test 11c gains a third arm carrying that malformation.

**G3-8 (major) — R-15's reversal is the right pattern with three costs it does not name.** "The
required-field set becomes a function of provider as well as `armKind`" is correct in shape, and
naming a reversal in advance is the right instinct. Three things it omits. **(i) The discriminator
must be declared, not observed.** `residencySource` is *captured* — it is the token the probe
answered on — so it cannot select the contract a record is validated against, because validation
would depend on having already probed the provider. A record needs a **declared** `provider`
(from `host.json` / the arm), with `residencySource` remaining the observation. **(ii) Provider B's
required set is not a variation, it is a collapse.** Eight of the nine fields §3.4.4a costs out —
`arch`, `quantization`, `compatibilityType`, `maxContextLength`, `loadedContextLength`, `modelType`,
`modelCapabilities`, `modelPublisher` — have no analogue on a hosted API. So the reversal does not
preserve fingerprint strength across providers; it **segregates** strongly-fingerprinted records
from weakly-fingerprinted ones. That is defensible, and it is a different claim from the one R-15
makes. **(iii) Nothing guards the comparison.** `compare_report` would put a provider-A and a
provider-B arm in one table with no banner and no refusal, while `-ml`'s instruments assume the arms
differ only in the model. §3.4.3's `SCHEMA VERSIONS IN THIS COMPARISON` line is the precedent.
*Fix:* three sentences in R-15 saying (i), (ii) and (iii) — no design work owed now, since the
trigger has not fired; the point is that the reversal's cost is a report-surface change, not a
schema change, and R-15 currently implies the opposite.

**G3-9 (minor) — §3.4.2's "two consequences the implementer must action" undercounts the S1 edit by
two sites, one of which will ship silently wrong.** Three shipped sites carry `lmsCliCommit`, not
two: `modelbench/fingerprint.py:79`, `tests/test_fingerprint.py:162` (the independently-written
literal §3.4.2 does name), and **`tests/conftest.py:38`** — the shared `MODEL_FIELDS` fixture every
blank-one-key test builds from. That third one is self-revealing (the missing `residencySource` fails
`validate()`), so it costs minutes. The fourth is not: `tests/conftest.py:39` declares
`residentModelsAtEnd: [{"modelKey": …, "sizeBytes": 2 << 30}]`, a `lms ps --json` element shape, while
§3.4.4a now specifies `{id, state}` **and** v1.8 removes `sizeBytes`' only source. `residentModels*`
are `REQUIRED_PRESENT`, which checks presence and never element shape, so the stale shape validates
and travels into S2, where `residency()` emits `{id, state}` and the two disagree with nothing to
catch it. *Fix:* name all four edits in §3.4.2's list, and add to S1 DC-1 an assertion on the
residency **element** shape (`{id, state}`, `state` a non-empty string, no `sizeBytes`).

**G3-10 (minor) — `--no-cold-load` survives in §3.6a's CLI table and has no possible meaning under
v1.8.** It is listed on `run` and defined nowhere in v1.7 or v1.8. Its v1.7 sense was "skip the
unload-then-timed-`lms load` step"; under v1.8 there is no unload, the warm-up is unconditional, and
`coldLoadSeconds` is recorded iff the model was not resident at start (§3.6). The only meaning left
to invent is "suppress the `coldLoadSeconds` record", i.e. a flag that hides a measurement — which
this plan's philosophy refuses everywhere else. *Fix:* delete it from the table. If something is
wanted here it is the opposite flag ("refuse to run unless the model is non-resident", for a
deliberate cold-load measurement), and R-14 already routes that to a documented human action.

**G3-11 (minor) — R-13's second input is recorded in §7's recap and not in R-13, which is §7 rule 4
run backwards.** §7's "What S2 inherits" correctly states that p50/p95 are now computed over a
possibly shorter sample than the item count and that "the definition and the denominator are decided
together". §6 R-13 — the risk that *carries* the open decision, and the thing `data-scientist` will
be handed — is **byte-identical to v1.7** (diffed). §7 rule 4 makes the owning section authoritative
and the recap derived, so the sweep went the wrong way. *Fix:* fold the §7 sentence into R-13, and
add the question v1.8 also opens and neither document asks: **is there a minimum surviving-sample
size below which no p95 is printed at all?** Under the guard a bad run can leave four latencies, and
a nearest-rank p95 over four points is the maximum — a number the report will print, honestly
denominated, that means nothing. That floor is the note's call, alongside the definition.

**G3-12 (minor) — test 15b is filed under the `-m live` heading while being the offline test the plan
calls load-bearing.** §5 places 15b between items 15 and 16 in the **Integration (`-m live`, opt-in,
real LM Studio)** block, flagging in its own text that it is offline "despite sitting among the live
ones", and §5's closing paragraph has to carve it out again ("**15b does not**"). §2.4 copies
falkor-chat's `addopts = '-ra -m "not live"'`, so a `live` marker applied by section-adjacency
removes from `pytest -q` the two assertions §5 itself calls "the tests that matter". Two carve-outs
for one item is the signal. *Fix:* move 15b into the unit block as 11d (or 12c) and leave a pointer
where it is now.

**G3-13 (nit) — §3.4.2's tier lists still end in `…` for both tiers, in the section that claims to
own the field set.** §3.4.2 states "**This section owns the model field set**" and then gives
`REQUIRED_NONEMPTY` and `REQUIRED_PRESENT` by example. Tier membership for the elided fields —
`loadedContextLength`, `maxContextLength`, `modelType`, `maxTokens`, `hostRamGb`, the timestamps — is
recoverable today only from `fingerprint.py`. That was tolerable while validation was a formality; it
is less so now that a tier decides whether a run refuses. Low stakes (S1 shipped the table, and it is
right), but an S2 author adding a field has no rule to follow. *Fix:* one sentence — every field not
listed as `REQUIRED_PRESENT` is `REQUIRED_NONEMPTY`, and a new field states its tier in the same edit.

### Disposition of carried findings

Pass 1's fourteen and Pass 2's four were dispositioned in Pass 2's table against plan v1.3; the plan
has since moved v1.3 → v1.8 through the engineering and statistics gates. Only the four still open at
that point are re-checked here.

| # | Disposition | Evidence rechecked |
|---|---|---|
| **N-1** (analysis-unit id) | **Fixed** | §3.3's `sampling` contract (`analysisUnit == pairingKey[0]`, row-count identity); `PackRef.analysisUnit`/`analysisUnitIndex` shipped (`packs.py:87-100`); `report._unit_ids` resolves it with no caller parameter; DC-5(c) asserts *which key* with a negative control. All three fixes I asked for landed. **G3-6 is the residue in a different place** — the `unit`/`analysisUnit` vocabularies were never reconciled in prose. |
| **N-2** (determinism probe) | **Fixed** | §3.8.4's probe exists, `RunResult.basis` is a required field with no dataclass default, `from_dict` carries the reader-side fallback only, `basis` degrades fail-safe and takes the weaker of two arms; §5 test 12b pins all four cases including "probe did not run → `assumed`". |
| **N-3** (`-ml` §4.6's derived `H`) | **Fixed by the note** | Note v1.8 §4.6 line 1065: `H` is `metrics.cleanThroughTurnH.H`, "validated `H ≤ min(script length)`"; §4.4's revision line names N-3 as closed. Plan §3.8.4 and the note now state one contract. |
| **N-4** (stale version-pairing token) | **Fixed** | §7 reads "this plan **v1.8** is aligned to the note **v1.8** (`27501c9`)"; `27501c9` is the note's v1.8 commit and the note's header reads `Version: 1.8`. The standing-obligation sentence beside it survives. |

### What's solid

- **§2.5 is a model of how to re-probe a premise.** Every number in it reproduces on this box today:
  19 models, all `not-loaded`, ten keys, `capabilities` on 15 of 19, sub-2 ms; `/v1/models` returning
  `{id, object, owned_by}`; `command -v lms` exit 1 with the Windows binary still reachable. The
  correction of the framing — the CLI is rejected for being a host-layout accident, **not** for being
  absent — is the right correction and it is stated in the plan rather than quietly applied.
- **§3.4.4a's three-alternatives section is the strongest prose in the document.** The
  `/v1`-only option is costed in fields rather than dismissed, and the fallback is rejected as a
  *source* while being retained as a *diagnosis* with the two error messages spelled out. That is the
  right shape for a decision with a real trade-off, and I would have approved the conclusion on this
  argument alone.
- **The warm-up's five parts are correctly ordered by what they protect.** Warm-up outside the item
  set, two budgets, `coldLoadSeconds` conditional on start-residency, guard, printed denominator —
  and both rejected alternatives are rejected for the contamination they cause rather than for cost.
  `--warmup <n>` surviving only as *extra warm-up calls* and never as *items removed after the fact*
  is exactly right: it keeps the knob and removes its ability to move the reported p50.
- **P4-4's re-attribution is argued, not merely obeyed.** DC-10 re-checks the deferral's two grounds
  and says why each fails, names the failing case in the report's own words ("one report making two
  mutually exclusive statements about the same metric"), rejects both alternatives against prior
  findings the reviews already paid for, and splits S1's net from S2's seam contract without letting
  S2 build a second copy. The `1.7 ms` measurement carrying the between-item probe is the same
  discipline: the design follows the number.
- **The absent-versus-empty table (§3.4.4a) is the right two rows.** Naming that the rule *inverts*
  at the fingerprint boundary — unrepresentable inside, mandatory `None` outside — is the kind of
  thing an implementer gets backwards exactly once, and it is now impossible to read past.

### Open questions

1. **Does LM-Studio-reported `time_to_first_token` include a JIT load?** Nobody has recorded it. It
   decides G3-3's scope and it is one extra assertion on a call S2's R-1 probe already makes. Not a
   design question — a measurement nobody has taken.
2. **R-13 stays `data-scientist`'s, with two inputs now** (definition + denominator), and I add a
   third for that same decision in G3-11: the minimum surviving-sample size below which no p95 is
   printed. Routed, not decided here.
3. **Does the embedder arm's `loadedContextLength` appear on a loaded embeddings model?** §2.3's
   evidence for `loaded_context_length` is a chat model. It is a `REQUIRED_NONEMPTY` field, so if it
   does not appear, G3-1's call-surface discriminator must cover it too. S2's R-1 probe is already
   the right place; it should read a loaded **embeddings** model as well as a loaded chat model.

---

## Appendix — Pass 3: what was probed and read

**C.1 — LM Studio surface, re-probed 2026-09-03, read-only (no model loaded or unloaded).**

```
$ curl -s -w 'HTTP %{http_code} in %{time_total}s\n' http://localhost:1234/api/v0/models
HTTP 200 in 0.002148s          # then 0.001566 / 0.001417 / 0.001542 on three repeats
```

| §2.5 / §3.6 claim | Result |
|---|---|
| 19 models installed | **19** |
| all `state: "not-loaded"` at probe time | **19/19 `not-loaded`** |
| catalog keys per entry | `id, object, type, publisher, arch, compatibility_type, quantization, state, max_context_length` on 19/19; `capabilities` on **15/19** — **no `runtime`, no size, no `loaded_context_length`** |
| "the catalog omits `capabilities` for several models" | 4 omit it: `google/gemma-3-4b` (vlm), `google/gemma-3-12b` (vlm), `gemma-3-4b-vl-it-…` (llm), `text-embedding-nomic-embed-text-v1.5` (embeddings) |
| an `embeddings` model advertising `tool_use` (§3.6's gate) | `text-embedding-qwen3-embedding-0.6b` → `capabilities: ["tool_use"]`. Both halves of M-7's gate still have live counter-examples on this box |
| two catalog ids, same weights (R-8) | `mistralai/ministral-3-3b` (publisher `mistralai`) and `mistralai_ministral-3-3b-instruct-2512` (publisher `bartowski`), both `Q8_0` |
| `GET /v1/models` returns `{id, object, owned_by}` only | confirmed verbatim |
| `command -v lms` → exit 1 | confirmed (exit 1) |
| `GET /api/v0/models` at ~1.7 ms | **1.4–2.1 ms**; plan's 1.6–2.3 ms band reproduces |

**C.2 — `POST /api/v0/embeddings` response shape (G3-1).** From LM Studio's own v0 REST endpoint
documentation: the embeddings response carries `object`, `data` (the vectors), `model` and `usage`,
and **does not include `runtime`, `model_info` or `stats`**. The chat-completions response carries
all three. Not probed live, because probing it would JIT-load an embeddings model, which this run was
not authorised to do; the doc claim is consistent with §2.3, which attributes `runtime` to the chat
route alone and never to the embeddings route.

**C.3 — Shipped-S1 facts the v1.8 claims were checked against** (`model-bench/` at working tree;
`modelbench/stats.py` and two test files were modified by a concurrent `tdd-engineer` and were not
relied on):

| Claim | Where | Result |
|---|---|---|
| swap keeps the model set at 30 | `modelbench/fingerprint.py:62-101` | 26 auto + 4 attested = **30**; removing `lmsCliCommit` and adding `residencySource` keeps 30 |
| "free only because `results/runs/` does not exist yet" | filesystem + `git ls-files` + `git log --all -- 'model-bench/results/*'` | **true** — directory absent, nothing tracked, nothing ever committed |
| §3.4.1's forbidden set "follows for free because it is a derivation" | `fingerprint.py:129-138` | **true** — `frozenset(_MODEL_SCHEMA_1) - frozenset(_DETERMINISTIC_SCHEMA_1)` |
| `runtimeName`/`runtimeVersion` source | `fingerprint.py:75-77` comment | "free from the /api/v0 **chat** route's `runtime` object" — corroborates G3-2 |
| sites carrying `lmsCliCommit` | `grep -rn` | **three**: `fingerprint.py:79`, `tests/test_fingerprint.py:162`, `tests/conftest.py:38` |
| residency element shape in the shipped fixture | `tests/conftest.py:39` | `{"modelKey": …, "sizeBytes": 2 << 30}` — the `lms ps` shape, not §3.4.4a's `{id, state}` (G3-9) |
| DC-10's selector vocabulary | `results.py:54-71`, `packs.py:87-100`, `roles.py:24-32`, `report.py:377` | `BinaryMetric.unit` ∈ {`item`,`conversation`,`query`,`turn`,`call`}; `PackRef.analysisUnit` is a `pairingKey` component name; the working predicate is `metric.unit == unit_kind_for_role(pack.role)` (G3-6) |
| `scored_outcome`'s raise path | `results.py:123-149`, `cli.py:144-150` | raises `IncompleteItemRecord`; `_cmd_compare` catches only `PackConfigError` (G3-7) |
| R-13 unchanged v1.7 → v1.8 | `diff` of the R-13 block across `aebb611^`/`aebb611` | **byte-identical** (G3-11) |
| note-restatement sweep | `grep` for `1.96`, α literals, κ figures, verdict strings, rule counts in v1.8's new text | **none reintroduced**; the only hits are §2.1's cited κ and §4 S1 / §5 test 6's deliberate "**not** `1.96`" |

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

---

## Pass 4 — 2026-09-03

**Re-gated:** plan **v1.9** (`81a3ef7`, +566/−131) against note **v1.10**
(`git show 81a3ef7:docs/plans/small-model-benchmarking-ml.md`, read at the ref because
`data-scientist` may be editing the working-tree copy), the requirements as amended, and the
**shipped S1 tree** (`5878014`, re-run here: **389 passed in 5.50 s**) wherever v1.9 asserts a
consequence for it. Weighted to v1.9's own change list: the five-source table and pinned capture
order, `callSurface`/`armProfile`, the seven-part warm-up and contamination guard, the `LatencyBlock`,
and R-13's closure. Findings carry the prefix **`P4-`** *(plan-gate Pass 4; `P4-*` in
`docs/reviews/small-model-benchmarking-impl.md` are that document's and are not the same series —
see the note on IDs at the end)*. Written by a reviewer who did not write Passes 1–3.

**CPG:** considered, not relevant — no `model-bench` CPG exists (`cpg_model_bench` is not a loaded
graph) and v1.9 makes no structural claim about `falkor-chat`. The S1 grounding below was done by
reading `model-bench/modelbench/*.py` and `tests/` directly.

**Verified live, not assumed.** `GET /api/v0/models` re-probed read-only on this box (no model
loaded or unloaded): 19 models, all `state: "not-loaded"`, key union exactly
`{arch, capabilities, compatibility_type, id, max_context_length, object, publisher, quantization,
state, type}` — **no `runtime`, no `loaded_context_length`**. LM Studio's own v0 REST documentation
was read for the `stats` object's units. The three-profile forbidden-set derivation, the 13+2+2+4+9
arithmetic and the six-member `REQUIRED_PRESENT` list were recomputed from the plan's own sets and
from `fingerprint.py`. Details in Appendix D.

**Verdict: needs changes.** 3 blockers, 5 majors, 5 minors, 1 nit.

**All thirteen Pass 3 findings are closed**, and the two blockers are closed at the general case the
gate asked for: §3.4.4a's five-source table is complete (I recomputed 13+2+2+4+9 = 30 against
`_MODEL_SCHEMA_1`, and the six `REQUIRED_PRESENT` fields match what S1 shipped, exactly), and the
union-of-others-minus-mine generalisation is **sound and yields precisely the four**
(`model:embeddings` forbids `{armId, armParametersHash, runtimeName, runtimeVersion, temperature,
maxTokens}`; `model:chat` still forbids exactly two; a fourth profile behaves — Appendix D.1).
`residencySource` is not yet swapped in the shipped tree and `tests/conftest.py:40` still declares
`{"modelKey", "sizeBytes"}`, which v1.9 correctly presents as outstanding S1 work.

**The new findings are not disagreements with those decisions.** They are, once again, in **what the
instrument says it will do with data it will not have** — and this pass they are concentrated in the
one surface v1.9 grew most: the timing block. Three of them (`P4-1`, `P4-3`, `P4-7`) are different
faces of a single gap, that **v1.9 specified four new timing quantities, a threshold on them and a
printed grammar over them without ever stating their unit, their carrier, or their failure
disposition**. `P4-1` is the sharpest thing in this review: taken literally, v1.9's detector
withholds the latency of every call slower than about a second, which is *every turn of the
tool-caller pack*, and the report then attributes a units bug to model-load contamination in the
note's own published words.

### Judgement on the three deviations v1.9 argued rather than took

**1. `G3-12` — keeping test `15b`'s number while moving it: right, and I would have made the same
call.** Citation stability beats numeric tidiness once a number is in circulation, and the actual
defect my predecessor named (a `live` marker acquired by adjacency) is genuinely fixed by placement
plus the pointer. The supporting claim is loose — "three reviews cite `15b`" is really one review,
one method note (`-ml` §11.9) and one coordination ledger (the other `15b` hits in `docs/` belong to
`kaizen-agent-ontology`, a different component's test) — but the constraint is real either way. One
residue: **P4-12**.

**2. `G3-6`'s disclosed residual — it is neither blocked on S2 nor a defensible deferral, because
the plan already forbids the case it defers.** This was the question put to this gate, so it gets a
direct answer. See **P4-8**: §4 S2's scorer contract says *every* `BinaryMetric.n` is computed as an
item count, which makes a turn- or call-pooled `BinaryMetric` unconstructible by a conforming
scorer; DC-10 simultaneously says a pooled metric's denominator "is not an item count and no
arithmetic over `items` can confirm it". Both cannot be true. Nothing here needs S2 design work —
it needs one of two sentences deleted, today. Disclosing a residual honestly was the right instinct;
the residual just is not the one the plan thinks it has.

**3. `unexplainedMs` folded into the model-load cause rather than opening a third — correct, and
correctly routed.** An in-call reload *is* a model load, so a third entry in `-ml` §11.7's cause
split would name a detector rather than a cause. The plan's reading is right. What follows from it
is the note's and I do not decide it: §11.5's exactness argument ("every withheld call was slower
than every timed call") is stated over **two** producers, and the gap detector is a third whose
right-censoring property is not established — a 1.1 s gap on a slow model is not necessarily above
every timed call. Open question 2 below routes it.

### New findings

#### Blockers

**P4-1 (blocker) — every ms-named field in v1.9's timing block is sourced from a *seconds*-valued LM
Studio field, no conversion is stated anywhere, and the same table contradicts itself. Under the
literal reading `unexplainedMs` withholds the latency of every tool-caller turn and the report blames
model-load contamination.**
*Evidence:* §3.6's FR-11 table sources `ttftMs` from `stats.time_to_first_token` **directly**, and
`prefillMsPer1kPromptTokens` from `1000 × stats.time_to_first_token ÷ (usage.prompt_tokens ÷ 1000)`
— the `× 1000` is a seconds→ms conversion, so one row of one table treats the field as ms and the
row above it treats the same field as seconds. LM Studio's v0 REST documentation gives
`"time_to_first_token": 0.111`, `"generation_time": 0.954` — **seconds** (Appendix D.2). `-ml` §11.7
is explicit that "the unit is milliseconds and the block never prints seconds", and §11.5.1's
threshold is **1 000 ms**.
*Why it matters:* `unexplainedMs = latencyMs − (ttftMs + generationMs)` with unconverted operands is
`latencyMs` minus ~1.1, so **any call slower than ~1 s crosses the threshold**. §2.2's measured pack
turns are ~1.3 s, so the tool-caller withholds 100 % of its latencies, `X = 0`, and `-ml` §11.7's
`X == 0` slot prints *"latency n = 0 of 38 items; no item's timing survived"* under the model-load
cause. FR-11's headline becomes structurally unprintable on the long-pole pack, and it fails
**selectively** — the embedder's 55–115 ms calls stay under the threshold, so the offline suite and
the fast pack look fine.
*Fix:* state the conversion in §3.6's FR-11 table as its own row — `ttftMs = 1000 ×
stats.time_to_first_token`, `generationMs = 1000 × stats.generation_time` — normalise both in
`lmstudio.ChatResult` so no caller ever sees the raw seconds, and add to test 15b an assertion with
a stubbed `stats: {"time_to_first_token": 0.111, "generation_time": 0.954}` requiring
`ttftMs == 111.0` and `unexplainedMs == wallClockMs − 1065.0`. A unit assertion on a stub costs
nothing and is the only thing that catches this class.

**P4-2 (blocker) — `armProfile` is a rewrite of shipped, green, mutation-tested S1 code that no stage
owns, no edit list names, and no done-condition asserts; its mechanical form silently invalidates
every model record.**
*Evidence:* §3.4.1 re-keys `REQUIRED_BY_SCHEMA` and renames `FORBIDDEN_BY_ARM_KIND` →
`FORBIDDEN_BY_ARM_PROFILE`; §4 S1's signature block and Appendix A carry it. The shipped module
derives `ARM_KINDS: frozenset = frozenset(FORBIDDEN_BY_ARM_KIND)` (`fingerprint.py:135`) and
`validate()` opens `if self.armKind not in ARM_KINDS` (`:161`). Re-key the mapping to profiles and
`ARM_KINDS` becomes `{model:chat, model:embeddings, deterministic}`, so `armKind == "model"` —
which §3.4.1 explicitly says is *unchanged* — fails membership and **every model record returns
`FieldProblem("armKind", "unknown")` and refuses on write**. `armKind`/`FORBIDDEN_BY` appear 59
times across six shipped files; `from_dict` strips only `"armKind"`, so `callSurface` needs handling
too. Meanwhile §4 S1 **DC-6 and §5 test 1 still state the retired two-kind contract** ("the two
`armKind` cases"; "a `model` record missing `runtimeName`" fails — false for `model:embeddings`,
where it is *forbidden*), so the acceptance surface asserts the contract v1.9 replaced.
*Why it matters:* v1.9 gives the far smaller `lmsCliCommit` swap a four-row edit table, a
loudly/silently column and a new DC-1 assertion — and gives the larger edit none of it, which reads
as "that one is the retro-edit". S3 cannot store an embedder record until this lands.
*Fix:* give §3.4.2's edit table a second block for the profile re-key (`fingerprint.py`'s mapping,
`ARM_KINDS`'s derivation — decoupled from the forbidden mapping — `validate()`, `from_dict`/
`to_dict`, `tests/test_fingerprint.py`, `tests/conftest.py`), assign it explicitly to the same
stage as the swap, and rewrite DC-6 / §5 test 1 for three profiles: a `model:embeddings` record
without `runtimeName` **validates**, the same record carrying it **fails as forbidden**.

**P4-3 (blocker) — four of the six FR-11 figures v1.9 commits the report to printing have no carrier
anywhere in the result schema.**
*Evidence:* §4 S1's `ItemResult` and the shipped dataclass (`results.py:103-121`) carry
`latencyMs: float | None` and nothing else timing-related; `detail` is specified "scorer-specific,
**never read by `report.py`**". §4 S2's `LatencyBlock` carries only wall-clock figures, two counts,
two withheld counts and `statsCoveredCount`. Yet v1.9 requires: `ttftMs` and prefill "printed with
**their own** denominator", their **medians** taking §11.6's p50 gate; `tokensPerSecond` "still
printed with its denominator"; `unexplainedMs` "**stored per item** and **its maximum reported**".
None of `ttftMs`, `prefillMsPer1kPromptTokens`, `tokensPerSecond`, `unexplainedMs` or its maximum has
a field on `ItemResult`, on `LatencyBlock`, or in the per-role `Aggregates` union (which is scoring
shapes, not timing).
*Why it matters:* this is the shape half of the same gap as P4-1, and it is S1/S2 shared territory —
`ItemResult` is S1's, `LatencyBlock` is S2's, and S2 cannot add a field to a frozen S1 dataclass by
its own done-condition. It also revives M-6's finding in a new place: with no typed home, these land
in `detail`, where the plan's own contract forbids `report.py` from reading them.
*Fix:* add `ttftMs: float | None`, `generationMs: float | None`, `unexplainedMs: float | None` to
`ItemResult` (S1, same edit pass as P4-2), and `ttftMsMedian`, `prefillMsPer1kMedian`,
`tokensPerSecondMedian`, `unexplainedMsMax` plus `statsCoveredCount` to `LatencyBlock` — then
invariant (iv) has something to count and the second denominator line has something to print.

#### Majors

**P4-4 (major) — §3.6 and §6 R-14 give opposite instructions on whether `unexplainedMs` withholds
anything, and R-14 is where an implementer reads the residual.**
*Evidence:* §3.6 — the runner "**withholds `latencyMs` when it exceeds the threshold `-ml` §11.5.1
sets**, counting the item under §11.7's model-load cause". §6 R-14's v1.9 parenthetical — "a
**detector**, though not yet a rule … **No cut-off is set and nothing is withheld on it**: whether a
threshold exists is sample admissibility and belongs with `-ml` §11". The note *does* set it:
§11.5.1 recommends 1 000 ms and §11.9 item 2a states the withholding as a plan dependency.
*Why it matters:* R-14 is the risk entry that documents this residual and is what a reader consults
when the guard behaves unexpectedly. The two readings differ in `latencyTimedCount`,
`latencyWithheldForLoad` and therefore in whether §11.6's level floor refuses a figure — i.e. in
what the report prints. It is also the mirror of `G3-11`: §3.6 was swept for the note's ruling and
R-14 was swept from an earlier draft of the same revision.
*Fix:* rewrite R-14's v1.9 parenthetical to match §3.6 — the detector withholds above the note's
threshold, folded into the model-load cause — and keep only the sentence that is genuinely R-14's:
that the threshold is a starting value to be re-checked against the first real pack run.

**P4-5 (major) — the pinned capture order omits the `catalog()` call that supplies twelve of its
thirteen catalog fields, and omits both refusals that must precede the warm-up.**
*Evidence:* §3.4.4a's ten steps run `host.json` → `probe()` → `residency()` → **warm-up** → runtime →
trip-wire → `residency()` → "**catalog re-read** for `loadedContextLength`" → items → end-residency.
`residency()` is defined in §3.6 as the catalog *filtered on* `state != "not-loaded"`, and on a cold
start the model under test is absent from that result. So `modelKey`, `arch`, `quantization`,
`modelType`, `modelCapabilities` and the rest have no step, and step 8's word "re-read" refers to a
read the list does not contain. Two refusals are likewise unlisted: §3.6's eligibility gate
(`type ∈ {llm, vlm}` and capabilities) and §3.4.4a's own `callSurface`-versus-catalog-`type`
cross-check, which the bullet costs as "one comparison against data step 3 already fetched" —
true of the HTTP response, false of `residency()`'s return type.
*Why it matters:* the list exists to pin where a refusal lands relative to the model load. Read as
written, an implementer reaches the catalog at step 8 — *after* the warm-up — so a model that fails
the eligibility gate or contradicts the pack's surface is refused **after** paying a full JIT load,
which is exactly the expensive refusal §3.4.4a took care to avoid for residency.
*Fix:* insert `3a. **`catalog()`** — the twelve catalog fields, the eligibility gate, and the
`callSurface`/`type` cross-check; either refusal exits before step 4` and change step 8 to read
"catalog re-read (the same call as 3a) for `loadedContextLength` alone". Also name where `startedAt`
is taken.

**P4-6 (major) — the staleness trip-wire's *other* comparand has no capture path: `attest` cannot
observe a runtime with the probe it is specified to use.**
*Evidence:* §3.4.4's `host.json` carries
`observedAtAttestation: {runtimeName, runtimeVersion, residencySource}`, and §3.4.5 point 3 plus
capture-order step 6 compare against it. §3.6a specifies `attest` as "prompts for the four
operator-attested fields, **probes LM Studio (§3.4.4a's two-step probe)**, writes `host.json`" — that
probe is `GET /api/v0/models` then `GET /v1/models`. I re-probed the catalog today: **no `runtime`
key on any of 19 entries** (Appendix D.3); `runtime` exists only on a chat-completions response.
`attest` also takes no model argument (`--api-base-url`, `--set k=v`), so there is no model to issue
a chat call against.
*Why it matters:* S2's done-condition asserts "`attest` writes a `host.json` matching §3.4.4's schema
**and the staleness trip-wire fires when `runtimeVersion` changes**" — unbuildable as specified. The
implementer's two escapes are both wrong: a chat call at attest time JIT-loads an arbitrary model
(3.6–21 s, and it evicts nothing it should), or the fields are written empty and the trip-wire
either never fires or fires on every run. v1.9 newly asserts step 6 is "the first instant its
comparands are available" — only one of the two comparands was checked.
*Fix:* decide it in §3.4.4 and state it in §3.6a. The cheap, honest option: `attest` writes
`observedAtAttestation` with `runtimeName`/`runtimeVersion` **absent**, and the trip-wire
back-fills them on the first `model:chat` run that completes, comparing only from the second run
onward — with the first-run state named in the record rather than silently equal.

**P4-7 (major) — a scored call that fails without timing out has no disposition, and both mandatory
`LatencyBlock` invariants plus `-ml` §11.7's cause split are falsified by it.**
*Evidence:* §3.6 dispositions exactly one failure, `requestTimeoutSeconds`. Grep of v1.9: no
treatment of a non-2xx response, a dropped connection, or an unparseable body (an HTTP 500 on
context overflow, or LM Studio restarting, over a 20-minute run on a 16 GB box). §4 S2's rules
(iii) `latencyWithheldForLoad + latencyWithheldForTimeout == latencyItemCount − latencyTimedCount`
and (iv) `statsCoveredCount == latencyItemCount − latencyWithheldForTimeout` are declared "none of
them optional"; `-ml` §11.7 slot 2 prints exactly two causes and both counts always.
*Why it matters:* such an item has no `stats` and no trustworthy wall clock, so it is withheld under
neither named cause: (iii) fails, (iv) fails, and the printed cause split does not sum to `M`. An
implementer writing these as `assert`s crashes the run on the first 500; writing them as computed
values silently miscounts and the report's cause line becomes false.
*Fix:* give §3.6 a fourth clause beside the timeout one — a call that fails without returning a
response is scored `fail` (never `n_a`), carries no timing, and is counted under
`latencyWithheldForTimeout` **renamed** to `latencyWithheldForNoResponse`, which is what both
producers actually are; then (iii)/(iv) hold unchanged and `-ml` §11.7's two-cause grammar stays
exhaustive. Ask `data-scientist` to confirm the slot-2 wording, since the string is the note's.

**P4-8 (major) — DC-10's disclosed pooled-metric residual contradicts §4 S2's scorer contract; one
of the two sentences is wrong, and neither needs S2 to decide it.**
*Evidence:* §4 S1 DC-10 — "a `BinaryMetric` whose `unit` is `turn` or `call` is **not**
cross-checked, because its denominator is not an item count and no arithmetic over `items` can
confirm it … Closing that needs the scorer to declare a pooled denominator's provenance, which is S2
design work nobody has scoped." §4 S2 — "**Every** `BinaryMetric.n` a scorer emits is *computed* as
the count of items it marked `scored_outcome(metric) is not None`, never counted along a second
path."
*Why it matters:* if S2's contract binds every `BinaryMetric`, then a pooled one whose `n` is not an
item count **cannot be produced by a conforming scorer** and the residual is unreachable rather than
open; if it does not bind them, the contract is narrower than it says and the plan's claim that
DC-10 is "unfalsifiable from inside a correct scorer" is overstated. The disposition matters because
this is the finding the plan carries forward as *blocked*, and the stakeholder's standing principle
turns on distinguishing blocked from deferred. It is neither.
*Fix:* pick one. Either scope §4 S2's sentence to "every `BinaryMetric` whose `unit` is the role's
unit kind", and DC-10's residual stands as written — or keep it unqualified and rewrite DC-10 to say
the pooled case is *unreachable at S2 by contract*, with the cross-check narrowed only because a
pooled metric cannot exist to be checked. Zero design work either way.

#### Minors

**P4-9 (minor) — §3.6 states the note's detector formula in bold and then says it does not.** The
sentence reads "the runner computes **`unexplainedMs = latencyMs − (ttftMs + generationMs)`** …
The metric, the threshold and its basis are `-ml` §11.5.1's and **are not restated here**." The
metric *is* the formula, so this is the restatement class §7 rule 2 forbids, arriving inside the
disclaimer that forbids it — and it restates it in a *third* spelling (`generationMs`, a name that
appears nowhere else in either document; the note writes `generation_time`). The plan's own list of
what it owns here — placement, field name, store-even-when-below — is correct and sufficient.
*Fix:* keep the three ownership claims and the field name, drop the formula, cite `-ml` §11.5.1 for
it; then define `generationMs` once, in §3.6's FR-11 table, as P4-1 requires.

**P4-10 (minor) — union-of-others-minus-mine leaves no room for "permitted but not required", and
the plan's own open field is the case that needs it.** The derivation is sound (I reproduced all
three sets), but it makes *not required by me* mean *forbidden on me*. §3.4.4a leaves
`loadedContextLength` in `model:embeddings` provisionally and says that if S2's probe finds it
absent, "the field moves out of that set" — which under the derivation makes it **forbidden**, so a
future LM Studio build that does return it would refuse a correct capture. The existing machinery
already has the right answer and the plan does not name it: move the field to `REQUIRED_PRESENT` for
that profile (captured `""`), the same way `modelCapabilities` handles a key the catalog omits.
*Fix:* state that schema 1 has no optional fields by design, name the `REQUIRED_PRESENT` route as
the resolution for `loadedContextLength`, and say whether §3.4.2's closed six-member `PRESENT` list
is global or per-profile — as written it is global, so a per-profile tier is not expressible.

**P4-11 (minor) — §2.5 and §3.6 attribute the 6× load spread to "two different models"; the note
attributes it to page cache, and the "same call surface" claim is false on the plan's own text.**
§2.5's 21.068 s was a cold `POST /v1/chat/completions` against `mistralai/ministral-3-3b` (Q8_0);
`-ml` §11.4's 3.625 s was against `qwen/qwen3-4b-2507` (Q4_K_M) and carries a `stats` object, which
only the `/api/v0` route returns — so the two differ in route, model and quantization, and both
documents nonetheless say "the same surface". The note names **page-cache state** as "the obvious
difference"; §3.6 sizes the first-call budget on "a load cost nobody can predict **per model**".
*Why it matters:* the two causes have different consequences for §11.5.1's threshold margin — if a
page-cache-warm reload can be much faster than 3.625 s, the "~3.5× below the smallest cold load"
margin is thinner than stated. *Fix:* correct §2.5 to say the two measurements differ in model,
quantization **and route**, drop "same call surface", and let §3.6 size on "load cost varies by
model and by page-cache state" — then route the threshold-margin question to `data-scientist`
(open question 2).

**P4-12 (minor) — §5's sequencing rule now silently denies `15b` the driving role the same section
calls load-bearing.** v1.9 moved `15b` into the unit block and deleted the closing carve-out
("**15b does not** — it is offline and drives the runner's timing design"). The surviving sentence
reads "Items **13–16** follow the implementation they cover rather than driving it" — and `15b`
sorts inside 13–16 by number, which is precisely why it kept that number (`G3-12`). So the one
sentence that used to protect it now excludes it. *Fix:* write the range as an explicit list —
"Items 13, 14, 15 and 16 follow the implementation they cover; **15b drives it**".

**P4-13 (minor) — `-ml` §11.7's second denominator line is committed to printing on an embeddings
arm, where the figures it denominates do not exist.** Slot 2 prints
`ttft/prefill/tokens-per-second n = <statsCovered> of <Y> items …` "whenever the two differ".
`POST /api/v0/embeddings` returns no `stats`, so on every embedder run `statsCoveredCount` is 0, the
two always differ, and the report prints a coverage line for three LM-Studio-side figures the
surface never produced. §4 S2's invariant (iv) is scoped "on the chat surface" and nothing says the
line is suppressed off it. *Fix:* one clause in §4 S2 — on a `model:embeddings` arm
`statsCoveredCount` is `None` (not 0) and slot 2's second line is not rendered; confirm the
suppression with `data-scientist`, since the grammar is the note's.

#### Nit

**P4-14 (nit) — two small factual slips in v1.9's own supporting prose.** "Three reviews cite `15b`"
is one review, one method note (`-ml` §11.9) and one coordination ledger; the other `15b` hits under
`docs/` belong to `kaizen-agent-ontology`. And §3.4.2's edit table's fourth row is described as the
one that "ships silently wrong", which is right, while the header column asks "Fails loudly if
missed?" and answers "**no — and this is the one that matters**" — the emphasis belongs on the row,
not in the boolean column, and a reader scanning the column reads three yeses and one no.

### Disposition of Pass 3's findings

All thirteen re-checked against v1.9 at `81a3ef7`; where a fix has a residue it is named.

| # | Disposition | Evidence rechecked |
|---|---|---|
| **G3-1** (embedder unbuildable) | **Fixed in design; execution unowned** | `armProfile` ∈ 3 profiles; `model:embeddings` = 26 fields, and I reproduced the derivation — it yields exactly `{armId, armParametersHash, runtimeName, runtimeVersion, temperature, maxTokens}`. Warm-up is call-surface aware. The 26 fields are all obtainable for an embeddings model on this box. Residue: **P4-2** (nobody edits the shipped module), **P4-10** (`loadedContextLength`). |
| **G3-2** (catalog is not the only source) | **Fixed** | Five-source table, 13+2+2+4+9 = 30, recomputed against `_MODEL_SCHEMA_1` — exact. The converse is stated false; §3.6's row corrected to "the catalog half". Residue: **P4-5** (the order omits the catalog read itself). |
| **G3-3** (three timing figures unguarded) | **Fixed, and better than asked** | The free measurement was taken (`-ml` §11.4) and reversed the conservative rule; the three siblings are kept with their own coverage. The reversal is recorded in §3.6 rather than smoothed. Residue: **P4-1**, **P4-3**, **P4-13**. |
| **G3-4** (guard's first comparand) | **Fixed** | Capture-order step 7; §3.6(a) states it explicitly and says why a true number beside a false cause needed fixing rather than catching. Test 15b now asserts `latencyTimedCount == latencyItemCount` on a clean cold run. |
| **G3-5** (timeout disposition) | **Fixed** | Four clauses: scored `fail` never `n_a`, no timing stored, count carried separately, re-probe → exit `3`. Warm-up timeout → exit `3`, message names `--first-call-timeout`. The abort alternative is rejected with a reason. Residue: **P4-7** (the *other* failure mode). |
| **G3-6** (DC-10 selector) | **Fixed in plan and in code** | `metric.unit == roles.unit_kind(pack.role)`; shipped `report.py:565` already uses it, documented at `:179`. Test 11c now pins an `item`-unit role. Residue: **P4-8** (the disclosed residual contradicts §4 S2). |
| **G3-7** (uncaught raise) | **Fixed in plan and in code** | DC-10 treats `IncompleteItemRecord` as a mismatch; shipped `report.py:221` catches it; test 11c gains the third arm. |
| **G3-8** (R-15's three costs) | **Fixed** | (i) declared-not-observed, (ii) collapse-not-variation, (iii) no comparison guard — all three stated, with `callSurface` named as the built precedent for (i). |
| **G3-9** (edit sites) | **Fixed** | Four-row table with a loudly/silently column, and the structural fix correctly placed on **S1 DC-1's element-shape assertion** rather than the fixture edit. Confirmed outstanding: `tests/conftest.py:40` still declares `{"modelKey", "sizeBytes"}` and `:38` still `lmsCliCommit`; suite green at 389 regardless, which is the finding's point. |
| **G3-10** (`--no-cold-load`) | **Fixed** | Deleted from §3.6a with the reason, and the opposite flag explicitly not added. |
| **G3-11** (R-13 recap/section) | **Fixed by closure** | R-13 is closed by `-ml` §11 and rewritten as design carried, not a recap. The third input I asked for — a floor below which no p95 prints — is answered by §11.6's level floor and §11.3's identity floor. |
| **G3-12** (test 15b filed live) | **Fixed, number kept** | Moved to the end of the unit block; both carve-outs gone; pointer left in place. Deviation endorsed (see Judgement 1). Residue: **P4-12**, **P4-14**. |
| **G3-13** (elided tier lists) | **Fixed** | `REQUIRED_PRESENT` closed at six; I counted `_PRESENT` in `fingerprint.py` — six, exactly matching. `REQUIRED_NONEMPTY` is the complement by rule. Residue: **P4-10** (per-profile tiers). |
| **n-ML-9** (§6 R-4's κ) | **Fixed** | R-4's figure withdrawn and replaced with a citation; the only surviving κ is §2.1's attributed inventory row, which Pass 3 already accepted. Full sweep of v1.9's new text for α, `z`, κ, verdict strings, rule counts: **none reintroduced** except **P4-9**. |
| **R-13's four follow-ups** | **All four landed** | `LatencyBlock` + five invariants (§4 S2), `latencyMsMax` + coverage columns (§3.5), both withholding dispositions (§3.6), the detector (§3.6). Cross-checked against `-ml` §11.9's asks 1, 2, 2a, 3, 4 — each has a home. Residues as above. |

### What's solid

- **The five-source table is the right instrument and it is arithmetically exact.** I recomputed it
  from `_MODEL_SCHEMA_1` rather than from the plan: 13 + 2 + 2 + 4 + 9 = 30, and the swap keeps it
  at 30. Stating that *the converse of the refusal rule is false* is the sentence that closes the
  blocker, and it is the sentence an implementer needed.
- **The forbidden-set generalisation is genuinely a generalisation, not a patch.** It had to stop
  being a pairwise difference the moment there were three sides, the plan says so in one clause,
  and the result is that nobody types the four names. A fourth profile behaves correctly under it
  (Appendix D.1). Deriving the answer rather than listing it is what made `model:chat`'s set
  provably unchanged.
- **The load-figure sweep is thorough and the framing is honest.** Every surviving `21.068 s` /
  `3.6 s` is design prose or a stub value in a test; §2.5 pairs the two measurements and says
  outright that nothing may be sized against either; test 15's assertion became a magnitude. The
  one thing the tool would have printed — §3.6's `p95 = … (latency n = 34 of 38)` sketch — is
  **withdrawn rather than corrected**, with the right reason (under §11.6 that run prints no figure
  at all, so the illustration was wrong in the way a restatement goes wrong).
- **Two reversals are recorded rather than smoothed.** §3.6 keeps the sequence by which the
  conservative "withhold all four" rule was right while the question was open and wrong once the
  measurement closed it; §7 says the same about the pairing. A plan that shows its own reversal is
  a plan whose next reader can tell evidence from preference.
- **The deviations were argued to the gate, not taken past it.** All three are defensible on their
  stated grounds, and two are simply right.

### Open questions

1. **The `data-scientist` ruling in flight touches this plan.** §4 S1 still declares
   `paired_bootstrap(diffs, *, B: int, seed: int)` and §3.3's manifest still carries
   `sampling.seed`, while `-ml` §11.2 reason 2 already describes Rule 4 as *replacing* the resampled
   bootstrap with a closed form. If the ruling completes that replacement, `B`/`seed` and the
   manifest's seed contract both move. **Not decided here** — flagged so the plan's next revision
   does not land before it.
2. **Is the `unexplainedMs` detector right-censoring in `-ml` §11.5's sense?** §11.5 establishes
   "every withheld call was slower than every timed call" over two producers; the gap detector is a
   third, and a 1.1 s gap on a slow model need not exceed every timed call. If it does not, slot 3's
   sentence is not true of every render. `data-scientist`'s, alongside the threshold-margin question
   P4-11 raises.
3. **Does `loaded_context_length` appear on a loaded *embeddings* model?** Carried from Pass 3 and
   correctly routed into S2's R-1 probe with a free-either-way resolution. I could not answer it —
   this run was not authorised to load a model, and the field is absent from all 19 entries while
   nothing is loaded (re-confirmed today). P4-10 is about what the *answer* does, not about the
   question.

### A note on finding IDs

This document's Pass 4 findings are `P4-1 … P4-14`. `docs/reviews/small-model-benchmarking-impl.md`
independently uses `P4-*` for its own Pass 4, and both series are already cited in the plan
(`P4-4`, `P4-5` in §4 S1 mean the *impl* review's). To keep the citation stable, plan text should
cite these as **`plan-gate P4-n`** and the impl review's as **`impl-gate P4-n`**. Passes 1–3 here
used `B/M/m/N/G3-`, which do not collide.

## Appendix D — Pass 4: what was probed, read and recomputed

**D.1 — The three-profile forbidden-set derivation, recomputed from the plan's own sets** (Python,
`set` algebra, not read off the table):

| Profile | Required | `⋃ others − mine` | Plan's table |
|---|---|---|---|
| `deterministic` | 11 | 21 fields | "everything in §3.4.2 that is not one of the eleven" ✓ |
| `model:chat` | 30 | `{armId, armParametersHash}` | exactly as stated ✓ |
| `model:embeddings` | 26 | `{armId, armParametersHash, runtimeName, runtimeVersion, temperature, maxTokens}` | exactly the four plus the two ✓ |

Source totals: catalog 13, chat `runtime` 2, run config 2, `host.json` 4, process/pack 9 = **30**,
and `_MODEL_SCHEMA_1` in `modelbench/fingerprint.py:62-101` holds 30 names that map onto them
one-for-one. `_PRESENT` appears on exactly **six** fields, matching §3.4.2's closed list. A
hypothetical fourth profile requiring a strict superset behaves (its extra field becomes forbidden
on the other three); the failure mode is the absence of an *optional* state — **P4-10**.

**D.2 — LM Studio `stats` units** (LM Studio's own v0 REST endpoint documentation, fetched today):
the `POST /api/v0/chat/completions` example response carries
`"tokens_per_second": 51.43709529007664, "time_to_first_token": 0.111, "generation_time": 0.954,
"stop_reason": "eosFound"` — **seconds**. `-ml` §11.7 states independently that
"`ItemResult.latencyMs` is a float in ms" and that §11.5.1's threshold is 1 000 ms, so the note's
symbols are ms and the conversion is owed by the plan (§7 rule 2: the harness surface is the plan's).
Not probed live, because obtaining a `stats` object requires a chat completion, which JIT-loads a
model — outside this run's authorisation.

**D.3 — `GET /api/v0/models`, re-probed 2026-09-03, read-only, nothing loaded or unloaded:**
19 models; every entry `state: "not-loaded"`; key union across all 19 is exactly
`arch, capabilities, compatibility_type, id, max_context_length, object, publisher, quantization,
state, type` — **no `runtime`, no `loaded_context_length`, no size**; `capabilities` present on 15
of 19; types `{llm, vlm, embeddings}`; `text-embedding-qwen3-embedding-0.6b` still advertises
`capabilities: ["tool_use"]` and `text-embedding-nomic-embed-text-v1.5` still omits the key. This is
the evidence for **P4-6**: nothing on the surface `attest` is specified to probe can produce
`runtimeName`/`runtimeVersion`.

**D.4 — Shipped-S1 facts the v1.9 claims were checked against** (working tree, clean for
`model-bench/`; suite re-run):

| Claim | Where | Result |
|---|---|---|
| S1 is green | `./.venv/bin/python -m pytest -q` | **389 passed in 5.50 s** |
| DC-10's predicate is the shipped one | `report.py:36, 179, 208, 565, 607`; `roles.py:39` | `from modelbench.roles import unit_kind as unit_kind_for_role`; selector at `:565` is `metric.unit == unit_kind_for_role(pack.role)` ✓ |
| G3-7's catch landed | `report.py:31, 221` | `except IncompleteItemRecord:` present ✓ |
| the four edit sites are still outstanding | `fingerprint.py:79`, `test_fingerprint.py:162`, `conftest.py:38`, `conftest.py:40` | `lmsCliCommit` still in the schema and both fixtures; `residentModelsAtEnd` still `{"modelKey": …, "sizeBytes": 2 << 30}` — specified, not implemented, as v1.9 says |
| `ARM_KINDS` is derived from the forbidden mapping | `fingerprint.py:135`, `validate()` at `:161` | `ARM_KINDS = frozenset(FORBIDDEN_BY_ARM_KIND)`; a mechanical re-key to profiles makes `armKind == "model"` fail membership — **P4-2** |
| `armKind` blast radius | `grep -c` over six files | 59 occurrences: `test_fingerprint.py` 25, `fingerprint.py` 18, `results.py` 8, `test_results.py` 4, `report.py` 2, `conftest.py` 2 |
| `ItemResult`'s timing fields | `results.py:103-121` | `latencyMs: float | None` only; `detail` defaulted and documented scorer-specific — **P4-3** |
| the two shipped `_percentile` copies | `results.py:573-578`, `stats.py:296` | `int(round(p/100·(X−1)))` — the estimator `-ml` §11.2 explicitly **rejects**; `_index_row` also computes p50/p95 inline from `run.items`, not from `run.latency`. Both are S1 code the note's closure requires rewritten, and neither appears in §3.4.2's edit table — **P4-2**'s pattern, second instance |
| restatement sweep of v1.9's new text | `grep` for α, `1.96`, κ, verdict strings, rule counts, the 1 000 ms threshold | none reintroduced; `1.96` hits are the two deliberate negatives, κ is §2.1's attributed row; the one restatement is the detector formula — **P4-9** |
| load-figure sweep | `grep` for `21.068`, `21 s`, `3.625`, `20 000`, `of the order` | 10 hits, all design prose or a stub value in tests 15b; nothing the tool prints — the sweep holds |
