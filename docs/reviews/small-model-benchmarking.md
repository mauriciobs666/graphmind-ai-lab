# Small-LLM benchmarking tool (`model-bench/`) — plan review

> **Status:** active · **Owner:** `analyst` · **Tracks:** — · **Reviews:** `docs/plans/small-model-benchmarking.md`

## Pass 1 — 2026-09-02

*(Current verdict is **`## Pass 8 (narrow)`**'s, at the end of this document: **needs changes** on
plan v1.14 / note v1.17 — 1 blocker, 2 majors, and an explicit answer on whether the plan is ready
for implementation. Pass 8 is a **narrow** re-check of v1.14's delta, not a full gate; Pass 7 gated
plan v1.13, Pass 6 plan v1.11, Pass 5 plan v1.10, Pass 4 plan v1.9, Pass 3 plan v1.8, all
**needs changes**;
Pass 2 gated plan
v1.3 with **approve with suggestions**. Pass 1 gated plan v1.1 and is kept intact — the passes are
meant to be read together, and each disposition table is only legible against the findings of the
pass before it.)*

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

## Pass 5 — 2026-09-07

**Re-gated:** plan **v1.10** (`3e5dc50`, +738/−134) against note **v1.13** (`5197ce6`), the
requirements as amended, and the **shipped S1 tree** (working tree at `2ba20d8`, `model-bench/`
clean; re-run here: **389 passed in 6.11 s**, `ruff` clean) wherever v1.10 asserts a consequence for
it. Weighted to v1.10's own change list: §3.6's unit boundary, §7's new **rule 5**, §4 **S1e**'s four
grep-pinned tables, `ItemTiming`, `attestationTripWire`, `latencyWithheldForNoResponse`, DC-10's
closure, DC-11, DC-12. Findings carry the prefix **`P5-`** (*plan-gate P5-n*, per §7's ID convention;
`docs/reviews/small-model-benchmarking-impl.md`'s are *impl-gate*). Written by a reviewer who did not
write Passes 1–4.

**CPG:** considered, not relevant — no `model-bench` CPG exists (`cpg_model_bench` is not a loaded
graph; the instance lists `cpg_falkorchat` and `cpg_deprecated_salesperson` only), `cpg_falkorchat`
is stale per the brief, and v1.10 makes no structural claim about `falkor-chat`. All grounding below
was done by reading `model-bench/modelbench/*.py` and `tests/` and by re-running the tables' own
greps.

**Verified, not assumed.** The four tables' pinned counts were re-verified upstream of this pass and
are taken as given; this pass spent its effort on **completeness** instead, re-running each table's
enumerating command and then searching for edit sites those commands cannot reach (Appendix E.1,
E.2). Every claim about the shipped tree below is a `grep`/`sed` result reproduced today. No model
was loaded in LM Studio.

**Verdict: needs changes.** 2 blockers, 4 majors, 4 minors, 0 nits.

**All fourteen Pass 4 findings are addressed in substance and none is unfixed** — the disposition
table below is fourteen *fixed*, which is what v1.10 claims. Five of them carry a residual, and two
of v1.10's own fixes introduced a new defect while closing the old one. **None of the ten findings
below is blocked on unbuilt work**: every one is a plan edit available today, most of them one
sentence. That is the reason the verdict is not *approve with suggestions* — nothing here needs S2,
so nothing here is a residual the standing principle would accept carrying.

**The two blockers are the pass's two assigned questions, answered.** `P5-2` answers *does rule 5
work?* — the rule is the right instrument and its stated completeness property is **false**, in a way
Table B demonstrates now rather than hypothetically. `P5-1` is the P4-5 fix's own collateral: the
capture order it repaired now refuses every embedder run, which is `G3-1`'s defect re-entering
through the sentence written to close `P4-5`.

### New findings

#### Blockers

**P5-1 (blocker) — §3.4.4a capture-order step 3a applies §3.6's *tool-calling* eligibility gate to
every model run, so every `model:embeddings` arm is refused at exit `4` and S3's own done-condition
is unrunnable.**
*Evidence:* the capture order is headed "One `model` run, in order" (§3.4.4a), and its new step 3a
requires "**both refusals that must precede the load**: §3.6's eligibility gate (`type ∈ {llm, vlm}`
and the capabilities rule) and the `callSurface`-versus-catalog-`type` cross-check … Either refusal
exits before step 4 — exit `4`". §3.6 scopes that gate to one role — "**Tool-calling eligibility is
gated before a tool-caller run**" — and its predicate is `eligible ⟺ type ∈ {"llm","vlm"} ∧ …`. An
embeddings model's catalog `type` is `"embeddings"` (§2.5's probe: types `{llm, vlm, embeddings}`).
*Why it matters:* §4 S3 done-condition 1 is "a real run against `text-embedding-qwen3-embedding-0.6b`
produces a stored result" — the exact model §3.6 names as the one the gate **refuses**, and §4 S2's
done-condition asserts that refusal as a unit test. Read as written, S3 — "the cheapest path to a
complete real run" — exits `4` at step 3a before it ever calls the adapter. This is `G3-1` (the
embedder unbuildable) arriving a second time through a different sentence, and it entered in the
revision that fixed `P4-5`.
*Fix:* scope step 3a — "§3.6's eligibility gate, **on a `tool-caller` pack only**" — and state
plainly in §3.6 that the universal pre-load refusal is the `callSurface`/`type` cross-check, of which
the tool-use rule is a role-specific addition. One clause in each section; no design work.

**P5-2 (blocker) — §7 rule 5's completeness property does not hold, and DC-12 asserts it does.
Table B — the largest of the four and the one rule 5 was written for — is incomplete today.**
*Evidence:* rule 5 claims that with an enumerating command, its counts and a residual assertion "the
table is **complete by construction** — a site the author forgot is still in the command's output and
the residual assertion fails until it is gone", and DC-12 restates it. Two independent ways that is
false, both instantiated in §4 S1e: **(a) a site that carries no token is invisible to the command.**
Table A discloses one and covers it with a *second* grep (`sizeBytes` → 1) — that is the correct
pattern. Table B does not: `grep -rFn arm_kind` finds **18 lines, 14 of which carry no `armKind`**
(Appendix E.1), including `tests/test_fingerprint.py:201-207`, the parametrized test pinning the
required-field contract that must gain a third profile, and `:211-213`, which asserts
`set(REQUIRED_BY_SCHEMA[1]) == {"model","deterministic"}`; and the mapping bodies the table edits —
`fingerprint.py:124`, `:131`, `:133` — carry none of the three tokens either, nor do
`test_fingerprint.py:34` and `:41`, the two decorators parametrized over
`REQUIRED_BY_SCHEMA[1]["model"]`. **(b) a rename that
does not *retire* a token has no zero residual to assert.** `armKind` **survives by design**
(§3.4.1 keeps its two values), so Table B's residual covers only `FORBIDDEN_BY_ARM_KIND` → 0 and
`ARM_KINDS`'s derivation — two auxiliary tokens that go to zero from the mapping rename alone, and
that say nothing about the 50 `armKind` lines the table is about.
*Why it matters:* the converse DC-12 needs — *residual zero ⇒ nothing missed* — does not hold, and
this is the mechanism the coordination adopted to stop a **third** incomplete edit list. Stated as a
guarantee it will be trusted by the next revision exactly as v1.8's and v1.9's lists were.
*Fix:* restate rule 5's property honestly — the command is a **no-forgetting guarantee over
token-carrying sites**, and **every site that carries no token must be covered by a second named
command or the table is not complete** (Table A's `sizeBytes` row is the pattern; make it the rule).
Where the token survives, require a **second-form residual** — a command whose count *is* zero after
the edit. Both of these run today and are the ones Table B is missing (verified, Appendix E.1):
`grep -rFn 'REQUIRED_BY_SCHEMA[1]["model"]' modelbench tests --include='*.py'` → **3 → 0**, and
`grep -rFn arm_kind modelbench tests --include='*.py' | grep -cF '"model"'` → **2 → 0**. Then give
Table B its `arm_kind` command with per-file counts, and reword DC-12 to assert what the residual
actually proves. Documentation-only.

#### Majors

**P5-3 (major) — Table D retires `conservative_envelope`'s last production caller and leaves
`paired_cluster_bootstrap` unreachable and undispositioned; its 13 sites are outside all four of the
table's commands.**
*Evidence:* `stats.py:263` — `resample = paired_cluster_bootstrap(diffs, design_effect=…, B=B,
seed=seed)` — is the body of `conservative_envelope`, which Table D collapses to
`conservative_envelope(table, *, design_effect)`. That line carries none of `bootstrap_seed`,
`conservative_envelope`, `cluster-bootstrap` or `DecidedBy`. `paired_cluster_bootstrap` has **13
lines** (`stats.py` 5, `test_stats.py` 8) and, once `:263` goes, no production caller — and neither
does `paired_bootstrap`, whose only production call site is `paired_cluster_bootstrap:187`
(Appendix E.2). Note §3.4 Rule 4 says "Rule 6's `cluster_bootstrap` and `paired_bootstrap` stay
seeded" and names `paired_cluster_bootstrap` nowhere; Table C separately asserts `stats.py:159`
"survives", which is true of the function and not of its reachability.
*Why it matters:* an implementer working Table D verbatim either leaves a public function and four
direct tests (`test_stats.py:1270, 1317, 1323, 1328`) exercising a path nothing reaches, or deletes
them — an edit the table did not authorise — and the residual (`bootstrap_seed` → 0,
`cluster-bootstrap` → 0) is satisfied either way. This is `P5-2`(a) with consequences.
*Fix:* add a Table D row for `paired_cluster_bootstrap` with its own enumerating command and a stated
disposition — **keep it, unreferenced, as §3.2d's clustered continuous interval** (the honest reading
of the note's "stays seeded") or **delete it with its four tests** — and say which, because they are
not the same tree.

**P5-4 (major) — §3.4.2's capture-ordering paragraph still states v1.8's single post-warm-up catalog
read and calls the v1.10 sequence a validation failure.**
*Evidence:* §3.4.2, unswept since v1.8: "`capture()` snapshots `residentModelsAtStart` **before the
warm-up call**, reads the **catalog after the warm-up returns**, and snapshots `residentModelsAtEnd`
after the last scored call. **A `capture()` that reads the catalog first produces a record with a
null `loadedContextLength` and fails its own validation** — which is correct behaviour, and is why
the ordering is stated rather than left to the implementer." §3.4.4a's pinned order now reads the
catalog **first** (step 3a, twelve fields plus both refusals) and re-reads it at step 8 for
`loadedContextLength` alone.
*Why it matters:* the two sections give opposite instructions on the one sequencing question `P4-5`
was raised about, and §3.4.2's version explicitly forbids the fix. An implementer who reads §3.4.2
writes a single post-warm-up read and loses both pre-load refusals — the expensive refusal the
capture order exists to prevent. It is also §7 rule 4's own failure shape (an owning section changed,
its derived surface not swept) occurring in the revision that added rule 5.
*Fix:* rewrite §3.4.2's paragraph to §3.4.4a's two-read sequence, keep the `loadedContextLength`
constraint attached to the **step 8** read, and cite §3.4.4a as the owner rather than restating the
order a third time.

**P5-5 (major) — the plan's pairing is one note revision stale, and note v1.13's co-presence
invariant lands on the plan. Ruling: yes, the plan owes a rule — and it owes a *runtime disposition*,
not only an assertion.**
*Evidence:* §7 states "this plan **v1.10** is aligned to the note **v1.12** (`fc2fcf6`)"; the current
note is **v1.13** (`5197ce6`). v1.13's §11.4 adds: "`ttftMs` and `tokensPerSecond` need a `stats`
object; the prefill figure **additionally** needs a usable `usage.prompt_tokens`. An item carrying
`stats` but no usable token count puts prefill's true `X` below `statsCoveredCount`, so both the gate
and §11.7 slot 2's single denominator line would overstate coverage for one figure of the three —
silently, and in the direction that prints", closed by an assertion and by §11.10 test 7. Plan §4 S2
rule (iv-b) pins `X = statsCoveredCount` for **all three** medians, and `ItemTiming.promptTokens` is
declared `int | None` — so the plan's own type admits the case its denominator rule denies.
*Why it matters:* the note's closure is a *test*-time statement ("an item carrying `stats` but no
usable `usage.prompt_tokens` fails this test, and must"). At run time the same condition is an
assertion with no disposition — precisely `P4-7`'s shape, where an implementer writing it as an
`assert` crashes the run and writing it as a computed value miscounts silently.
*Fix:* two sentences in §4 S2. (1) State co-presence as a named rule beside (iv-b): the three medians
share one count and that count is `statsCoveredCount`. (2) Give the failing item a **runtime**
disposition — recommended: an item with `stats` but `promptTokens` absent or `≤ 0` is **excluded from
`statsCoveredCount` entirely**, which keeps one honest number across all three at the cost of
understating ttft/tps coverage (conservative, and in the safe direction), with the note's alternative
— a separate `prefillCoveredCount` — named as the resolution if it is ever more than a rarity. Then
re-pair §7 to v1.13.

**P5-6 (major) — `censoringExact`'s first clause is unevaluable on the record v1.10 defines: the
`P4-7` merge erased the timeout/other distinction the predicate needs, so every timeout-only run
silently prints slot 3's weaker string.**
*Evidence:* note §11.5.1's predicate is "**true** when every withheld item is a **timeout**, or when
the smallest wall clock among the load-withheld items exceeds the largest among the timed ones;
false otherwise, and false whenever a withheld item's wall clock is not readable" — clause 1 resting
on "a timeout … exhausted a budget every timed call returned inside". Slot 3's true-branch string
says the same in prose. v1.10's `ItemTiming.withheldFor` is
`Literal["load", "no_response"] | None`, and §3.6's fourth disposition folds a fast HTTP 500 into
`no_response` with **no timing at all**. So `report.py` cannot tell a timeout from a 40 ms 500.
*Why it matters:* the plan reads the merge as safe — §4 S2: "the only such item is one that returned
no response, which has no wall clock to compare" — and it *is* safe, but it silently overrides the
note's clause 1: a run whose only withholding is a timeout now renders the **false** branch, where
the note says the true one is owed. Neither document records that consequence, and the plan claims
v1.12's two asks "both land".
*Fix:* give `withheldFor` a third value, `"timeout"`, so clause 1 is evaluable while
`latencyWithheldForNoResponse` stays one counter (§11.7's cause split prints one `no response`
label either way, which the note has already ruled) — then ask `data-scientist` to confirm clause 1
survives against the widened category, since the predicate and both strings are the note's.

#### Minors

**P5-7 (minor) — §4 S2 rule (i)'s "the three figures" was true of v1.9's block and is
under-determined in v1.10's, where it can be read into contradiction with (iv-a). Ruling: yes, this
needs a plan edit.** At v1.9 the block held exactly three `float | None` figures (`latencyMsP50`,
`P95`, `Max`), so (i) was unambiguous. v1.10 added four more and did not sweep (i), while the
sentence immediately above it uses "its three **siblings**" for the *other* trio. Under the sibling
reading, (i) ("`None` exactly when `-ml` §11's gates refuse them") forbids exactly the `None`s
(iv-a) mandates for absence-of-input. `data-scientist` is right that the intended reading is
consistent and right that it is one word away. *Fix:* write "the three **wall-clock** figures" in
(i), and add one clause: the four `stats`-derived figures are `None` under (iv-a) for
absence-of-input as well as under a gate refusal, and the two causes are not distinguished in the
block.

**P5-8 (minor) — §3.6's unit boundary states four `ChatResult` normalisations with no absent-`stats`
case, while §4 S2 (iv) depends on that case existing.** The boundary reads
`ChatResult.ttftMs = 1000 × stats.time_to_first_token` unconditionally, on construction. But (iv)
says `statsCoveredCount` "counts items with a `stats` object" and (iv-a) says `0` "keeps its meaning
on the chat surface, where it says **every response lacked `stats`** and is a real signal" — so a
chat response without `stats` is an expected state, and the stated construction raises on it.
*Fix:* one clause on the boundary — each derived field is `None` when its source key is absent,
never `0`, and `ChatResult` construction never raises on a missing `stats`.

**P5-9 (minor) — §3.6 says a no-response item "carries **no timing at all**", while §4 S2 rule (vi)
counts it by reading `timing.withheldFor`.** Rule (vi) derives both withheld counts as "counts of
`timing.withheldFor` values", which requires an `ItemTiming` to exist on such an item; §4 S1's own
comment says `timing` is `None` "**iff** this arm produces no timings at all (deterministic)". The
two are reconcilable and an implementer should not have to reconcile them. *Fix:* in §3.6 write
"carries an `ItemTiming` whose only populated field is `withheldFor`" rather than "no timing at all".

**P5-10 (minor) — the note's flagged `sampling.seed`-in-the-fingerprint gap has a deadline the note
does not carry, and it is the plan's.** Note §3.4 Rule 4 flags that "§3.2d says *the seed goes into
the environment fingerprint (FR-7)* and it does not — `fingerprint.py` at `5878014` contains no seed
field", calling it pre-existing and unaffected. It is unaffected by *that* ruling; it is not
unaffected by S3. §3.4.2's model set is closed at 30/26 and "free only now, because `results/runs/`
does not yet exist" — the same argument S1e makes four times. If §3.2d is right, schema 1 is one
field short and closing it after S3 costs a `migrate`. *Fix:* state in §3.4.2 that the pack seed
reaches the fingerprint **transitively, through `packContentHash`** (the manifest is hashed and the
seed is a manifest field), which is almost certainly what §3.2d means — and route the sentence's
wording to `data-scientist`, but resolve the plan side before S3.

### Disposition of Pass 4's fourteen

All fourteen re-checked against v1.10 at `3e5dc50`; **fourteen fixed, none unfixed**. Where a fix
leaves a residual or introduced a new defect, it is named.

| # | Disposition | Evidence rechecked |
|---|---|---|
| **P4-1** (seconds/ms) | **Fixed** | §3.6's unit boundary: `ttftMs = 1000 × stats.time_to_first_token`, `generationMs = 1000 × stats.generation_time`, `tokensPerSecond` unconverted, `wallClockMs` client-side, normalised **on `ChatResult` construction** so no caller sees seconds; the FR-11 prefill row drops the second conversion and says why; §5 test 15b asserts `111.0` / `954.0` / `wallClockMs − 1065.0` off a stubbed `stats`; Appendix A's `ChatResult` row carries it. Residue: **P5-8**. |
| **P4-2** (`armProfile` unowned) | **Fixed in design; the mechanism it introduced is not sound** | §4 S1e's four tables + §7 rule 5 + DC-12; `ARM_KINDS` explicitly decoupled with the `FieldProblem("armKind","unknown")` consequence spelled out; `from_dict` told to strip `callSurface`; DC-6 rewritten for three profiles. Residue: **P5-2** (Table B incomplete, rule 5 over-claimed), **P5-3** (Table D). |
| **P4-3** (no carriers) | **Fixed** | `ItemTiming` (7 fields, all `\| None`, never `0`) on `ItemResult`; `latencyMs` demoted to a property with `to_dict`/`from_dict` re-derivation; `LatencyBlock` gains `ttftMsMedian`, `prefillMsPer1kMedian`, `tokensPerSecondMedian`, `unexplainedMsMax`; DC-11 asserts all three properties including "no printed number is computed from `timing.wallClockMs`". Residue: **P5-5**, **P5-7**. |
| **P4-4** (R-14 vs §3.6) | **Fixed** | §6 R-14's v1.10 parenthetical states the withholding, names the model-load cause, and records that v1.9's parenthetical said the opposite; the surviving R-14-owned sentence is the "starting value, re-checked on the first real pack run" one, exactly as asked. |
| **P4-5** (capture order) | **Fixed — and it introduced a blocker** | Step 0 (`startedAt`/`endedAt`), step 3a (`catalog()`, twelve fields, both refusals), step 8 reworded to "the same `catalog()` call as step 3a". Residue: **P5-1** (the gate is unscoped at 3a), **P5-4** (§3.4.2 unswept). |
| **P4-6** (`attest`'s comparand) | **Fixed** | §3.4.4's example carries `observedAtAttestation: {"residencySource": …}` alone; the two runtime keys are absent-never-`""`; the first `model:chat` run back-fills them plus `runtimeObservedAt` and touches nothing else; `RunResult.attestationTripWire` records `compared` / `first-observation` / `unavailable` / `None`; §4 S2's done-condition asserts all four. Writing-empty is rejected with the absent-vs-empty reason. |
| **P4-7** (no-response call) | **Fixed; one consequence unnoticed** | §3.6's fourth disposition, verbatim-from-the-timeout except that it does **not** re-probe or exit `3` on an error response (a connection failure still does); `latencyWithheldForTimeout` → `latencyWithheldForNoResponse` in §4 S2, Appendix A and §5; (iii)/(iv) restated over it; test 15b gains the stubbed 500 with (iii)/(iv) asserted. The note's slot-2 label was confirmed rather than assumed. Residue: **P5-6**. |
| **P4-8** (DC-10 vs S2) | **Fixed — closed rather than disclosed, and correctly so** | `validate_pack` refuses a pooled `verdictMetrics` member; pooled `n` is the sum of reserved `"<metric>#denominator"` contributions with `#` refused in metric names; §4 S2's contract is scoped to two arithmetics in one pass; DC-10 now ranges over **every** `BinaryMetric`; test 11c gains a fourth arm. The gate's offered alternative (scope the sentence, keep the residual) is named and rejected on the stakeholder principle — the right call. |
| **P4-9** (restated formula) | **Fixed** | The formula is gone from §3.6; the three ownership claims and the field name remain with a citation to `-ml` §11.5.1; `generationMs` is defined exactly once, in §3.6's FR-11 table, as the finding asked. |
| **P4-10** (no optional tier) | **Fixed** | §3.4.2: `REQUIRED_PRESENT` resolves per profile — `model:chat` six, `model:embeddings` five (minus `temperature`), `deterministic` none — with the per-profile mechanism named (`FieldSpec.tier` inside `REQUIRED_BY_SCHEMA[schema][armProfile][field]`); "schema 1 has no optional tier" stated as a decision with its cost; `loadedContextLength`'s resolution is the `REQUIRED_PRESENT` route in §3.4.2, §3.4.4a **and** §4 S2's R-1 bullet. |
| **P4-11** (6× spread) | **Fixed** | §2.5 now says the two measurements differ in model, quantization **and route**, drops "same call surface", attributes page-cache state to the note, and records the note's v1.12 withdrawal of the "~3.5× below the smallest cold load" margin; §3.6's budget prose reads "varies by model **and by page-cache state**". |
| **P4-12** (`15b`'s role) | **Fixed** | §5's closing rule is an explicit list — "Items **13, 14, 15 and 16** follow the implementation they cover; **`15b` drives it**" — with the range-by-number trap recorded. |
| **P4-13** (embeddings denominator) | **Fixed** | §4 S2 (iv-a): `statsCoveredCount` is `None`-never-`0` and slot 2's second line is not rendered, **conditioned on the call surface, not the arm profile** — the wider condition, confirmed against note §11.9 ask 5; the four sibling figures and `unexplainedMsMax` are `None` there too, and §3.6 records that the detector is chat-surface-only. |
| **P4-14** (two slips) | **Fixed** | "three reviews" corrected to review + method note + coordination ledger with `kaizen-agent-ontology` named as the other `15b`; Table A's loudly column now carries "**no** — see §3.4.2; the fix is DC-1's assertion, not this row" **on the row**, and the emphasis is out of the column header. |

### The dependency sweep the brief asked for

Pass 4's and Pass 3's shared miss — a plan rule asserted on a note argument that did not yet hold —
was swept for deliberately this pass. Result: **one instance, and it is the same shape with the
polarity reversed.** Plan v1.10 pairs itself to note **v1.12** and is now one revision behind
**v1.13**, which (a) *confirmed* (iv-b) by withdrawing the note's own `tokensPerSecond` exemption —
no plan change owed, and the plan's rule was right before the note's argument for it was — and
(b) added the co-presence invariant, which the plan does not carry (**P5-5**). Nothing else in
v1.10's new text rests on a since-revised note claim: Table C's `percentile(values, *, permille)`
matches §11.2's published signature and its four properties; Table D's "five acceptance tests" are
§3.4 Rule 4's items 1–5, counted; the `DecidedBy` rename is the note's own recommendation with the
architect's decision recorded; §11.6's floors, §11.5's table and §11.3's identity floor are untouched
by v1.12 and v1.13 as §7 claims. The `censoringExact` case (**P5-6**) is the inverse defect — a note
claim that no longer holds against a *plan* change made in the same revision — which is worth naming
as a second face of the same class, since the sweep obligation in §7 rule 1 is symmetric and the
plan-side half has now failed once too.

### What's solid

- **Rule 5 is the right instrument even though its stated property is wrong.** Naming the command,
  pinning per-file counts at a commit, and putting the residual in a done-condition converts an
  un-auditable list into an auditable one, and the `grep -rFc` / `grep -rFo` distinction (50 lines
  vs 57 occurrences) is exactly the kind of precision a table needs to be reproducible. `P5-2` is a
  correction to one sentence of the rule, not a rejection of it.
- **`P4-8` was closed rather than disclosed, and the reasoning is the strongest in the revision.**
  The plan took the gate's harder branch, gave the pooled denominator a per-item provenance
  (`"<metric>#denominator"`) so one arithmetic becomes two without a second path, and rejected the
  cheap alternative *on the stakeholder principle by name*. That is a plan reading its own gate
  correctly.
- **`ItemTiming` is a better answer than the one `P4-3` asked for.** Making `latencyMs` a derivation
  rather than a second stored field removes the invariant that would have had to police them, and
  taking the field over the note's offered reconstruction is justified with the one case the
  reconstruction cannot cover (an embeddings arm, where all three reconstruction operands are `None`
  and the wall clock was measured perfectly well). The reasoning names the failure mode, not the
  preference.
- **`P4-6`'s fix chose the honest state over the convenient one.** Absent-not-`""`, back-fill on
  first observation, and the first run's state *named in the record* (`attestationTripWire`) rather
  than silently indistinguishable from a comparison — the same absent-vs-empty discipline §3.4.2
  applies one file over, applied without being asked to.
- **The suite is green and the tree matches what the plan says about it.** 389 passed, `ruff` clean;
  every "still outstanding" claim v1.10 makes about the shipped tree reproduces.

### Open questions

1. **Does `loaded_context_length` appear on a loaded *embeddings* model?** Carried from Passes 3 and
   4, still correctly routed into S2's R-1 probe with a free-either-way resolution, and `P4-10`'s
   fix now makes both answers safe. Unanswerable here: this run was not authorised to load a model.
   **This is the one genuinely blocked item in the review, and it is blocked on unbuilt work (S2's
   R-1 probe) rather than deferred.**
2. **Does note §11.5.1's `censoringExact` clause 1 survive the widened `no_response` category?**
   `P5-6` proposes making it evaluable; whether it *should* be — a fast HTTP 500 is not
   right-censoring and clause 1 would then need to say "timeout" and not "no response" — is
   `data-scientist`'s, and the predicate and both slot-3 strings are the note's.
3. **Which co-presence resolution does `data-scientist` want in the plan** — dropping the item from
   `statsCoveredCount` (conservative, one number, understates two figures) or a fourth
   `prefillCoveredCount` (exact, a second denominator line to render)? `P5-5` recommends the first
   and names the second; the note's §11.4 sentence leans the other way ("the figure that lost items
   needs its own count"), so the two should agree before the plan writes it.

## Appendix E — Pass 5: what was re-run and read

**E.1 — Table B's blind spot, enumerated** (working tree at `2ba20d8`, `model-bench/` clean).
`grep -rFn arm_kind modelbench tests --include='*.py'` → **18 lines**
(`tests/test_fingerprint.py` 9, `tests/conftest.py` 4, `tests/test_results.py` 3,
`tests/test_report.py` 2), of which **14 carry no `armKind`** and so appear in none of Table B's
three commands:

| Site | Why it is an edit site |
|---|---|
| `test_fingerprint.py:201-207` | parametrized over `[("model", EXPECTED_MODEL_SCHEMA_1), ("deterministic", …)]`, indexing `REQUIRED_BY_SCHEMA[1][arm_kind]` — must gain a `model:embeddings` row and a 26-name literal |
| `test_fingerprint.py:211-213` | `assert set(REQUIRED_BY_SCHEMA[1]) == {"model", "deterministic"}` — the two-kind contract, asserted directly |
| `test_fingerprint.py:21, 97, 102, 267` | `_problems(arm_kind, …)` and the three discriminator tests — semantics change once there are two discriminators |
| `conftest.py:148, 160` | the shared fixture's `arm_kind: str = "model"` branch, which must become profile-aware |
| `test_results.py:62, 71, 239`; `test_report.py:471, 486` | `arm_kind="deterministic"` construction sites |

Also outside all three commands: `test_fingerprint.py:34` and `:41`, the two
`@pytest.mark.parametrize("field", sorted(REQUIRED_BY_SCHEMA[1]["model"]))` decorators that drive one
test per required field (`grep -rFn 'REQUIRED_BY_SCHEMA[1]["model"]' modelbench tests
--include='*.py'` → **3 lines**, run today); and, in the module the table edits, `fingerprint.py:124`
(`1: {"model": _MODEL_SCHEMA_1, …}` — the re-key itself), `:131` and `:133` (the
`FORBIDDEN_BY_ARM_KIND` mapping **bodies**; the grep matches the name at `:128` and `:136`, not the
entries). Most of these fail loudly through the suite, which is a real mitigation — but the table is
what an implementer works from, and DC-12's residual is satisfied without them.

**E.2 — Table D's blind spot.** `grep -rn paired_cluster_bootstrap modelbench tests --include='*.py'`
→ **13 lines** (`stats.py:23, 162, 263, 567, 843`; `test_stats.py:41, 1270, 1317, 1323, 1328, 1330,
1560, 1588`), none of them matching `bootstrap_seed`, `conservative_envelope`, `cluster-bootstrap` or
`DecidedBy`. Call chain read at `stats.py`: `conservative_envelope:263` → `paired_cluster_bootstrap`
→ `paired_bootstrap:187`; `grep -rn 'paired_bootstrap('` finds no other production call site, so
retiring `:263` leaves **both** functions production-unreachable. Note that
`grep -F cluster-bootstrap` (hyphen, the `DecidedBy` token, 27 lines) and `grep -F cluster_bootstrap`
(underscore, the function, 28 lines) are disjoint vocabularies — the residual "`cluster-bootstrap`
→ 0" says nothing about either function.

**E.3 — Shipped-tree facts the v1.10 claims were checked against.**

| Claim | Where | Result |
|---|---|---|
| S1 is green | `./.venv/bin/python -m pytest -q` | **389 passed in 6.11 s** |
| lint clean | `./.venv/bin/python -m ruff check .` | **All checks passed!** |
| `ARM_KINDS` is still coupled | `fingerprint.py:136` | `ARM_KINDS: frozenset[str] = frozenset(FORBIDDEN_BY_ARM_KIND)` — Table B's decoupling row is still outstanding, as the plan says |
| the schema map is still two-kinded | `fingerprint.py:124`, `test_fingerprint.py:213` | `{"model": …, "deterministic": …}`; the test asserts exactly that set |
| both `_percentile` copies are the rejected estimator | `results.py:573-578`, `stats.py:296-299` | `int(round(pct/100.0 * (len(ordered)-1)))` in both — Table C's premise holds |
| `Fingerprint` construction sites | `grep -rFn 'Fingerprint('` | 12 lines, 10 real construction sites; all but one carry `armKind` on the line, and `callSurface` as a required no-default argument breaks every one loudly — rule 5's *add* half is satisfied in substance |
| the embedder's catalog `type` | §2.5 / Pass 4 Appendix D.3 | types across 19 entries are `{llm, vlm, embeddings}`; `text-embedding-qwen3-embedding-0.6b` is `embeddings` and advertises `capabilities: ["tool_use"]` — the evidence for **P5-1** |
| no `model-bench` CPG | `mcp__cypher__query(graph='cpg_model_bench')` | not a loaded graph; instance lists `cpg_falkorchat`, `cpg_deprecated_salesperson` and workspace graphs only |

## Pass 6 — 2026-09-07

**Re-gated:** plan **v1.11** (`85a32e5`, +540/−110) against note **v1.14** (`ca69cb1`), the
requirements as amended, and the **shipped S1 tree** — `model-bench/` at HEAD is byte-identical to
`5878014` (`git diff --stat 5878014..HEAD -- model-bench/` is empty), re-run here: **389 passed in
5.49 s**, `ruff check .` clean. Weighted to the brief's five questions: Pass 5's ten dispositions,
the architect's rejection of `P5-2`'s second residual, rule 5's second formulation, the
fix-reopens-a-finding shape, and Table E plus a sweep for defects of its class. Findings carry the
prefix **`P6-`** (*plan-gate P6-n*; `docs/reviews/small-model-benchmarking-impl.md`'s are
*impl-gate*). Written by a reviewer who did not write Passes 1–5.

**CPG:** considered, not relevant — no `model-bench` CPG exists (the instance loads `cpg_falkorchat`
and `cpg_deprecated_salesperson` only), and v1.11 makes no structural claim about `falkor-chat`. All
grounding below is `grep`/`sed`/`pytest` against `model-bench/` at `5878014`.

**Verified, not assumed.** The sixteen pinned counts were re-verified upstream of this pass and are
taken as given. This pass spent its effort on (a) whether the five tables *satisfy* the rule they now
cite, and (b) the brief's item-5 sweep — shipped S1 code correct for its current caller and wrong for
a caller the plan commits to adding. Every claim below is a command reproduced today (Appendix F) or
a line read at a cited location. No model was loaded in LM Studio.

**Verdict: needs changes.** 1 blocker, 2 majors, 2 minors, 0 nits. **S2 may not be dispatched yet**,
and the reason is `P6-1` specifically: its resolution may change `ItemResult`'s shape, which S2's
runner/scorer seam constructs, so dispatching first risks the rework the S1e "free only now" argument
exists to avoid. `P6-1` is a plan **decision** plus one specification, not unbuilt work.

**All ten Pass 5 findings are fixed and none is carried** — the disposition table below is ten
*fixed*, which is what v1.11 claims, and I checked each one against the tree rather than against the
plan's own summary. That is a real improvement on the previous three revisions. The three findings
below are **not** re-litigations of Pass 5: `P6-2` and `P6-3` are defects *introduced by* v1.11's
fixes to `P5-2` and `P5-3`, and `P6-1` is the sweep the brief asked for, returning a larger instance
of Table E's own defect class.

**The two adjudications the brief asked for, up front.**

**(1) The architect's rejection of `P5-2`'s second residual is correct.** `P5-2` prescribed
`grep -rFn arm_kind modelbench tests --include='*.py' | grep -cF '"model"'` → 2 → 0. The two lines
are `tests/conftest.py:148` (`arm_kind: str = "model"`, a parameter default) and `:160`
(`fields = model_fields() if arm_kind == "model" else deterministic_fields()`). `armKind` keeps both
its values by design (§3.4.1), so `:148` survives any faithful edit outright, and `:160`'s plausible
faithful form — add a `call_surface` parameter, branch on it inside the `model` arm — keeps
`arm_kind == "model"` too. The residual is ≥ 1 after a correct implementation. Rejecting it was
right, and Table B's own row for `conftest.py:148`/`:160` says so on the row, which is the better
place for it than a footnote. **The generalisation — *a residual that fails on a correct edit is a
trap, not a check* — is sound and is the right lesson to have drawn.** It is also the rule v1.11
breaks in the same table: `P6-2`.

**(2) Rule 5, second formulation: sound as stated; not fully applied.** The restatement is honest and
coherent — a no-forgetting guarantee over token-carrying sites, a second named command for every
token-free site, a second-form residual wherever the token survives, plus the two bounds (a command
reaches a construct's body through its head; a residual must not fail on a correct edit). DC-12 is
correctly reduced to the direction a residual proves and says out loud what it does not prove.
**Table B satisfies it**: commands 4–6 are real coverage, the eleven site rows with *found by* are
checkable, and stating that `armKind` has no residual — with the type system named as what stands in
its place — is exactly what 5(b) asks for. **Table E does not** (`P6-3`), and §4 S1e's scope claim
overreaches (`P6-4`). So the rule is not the problem this pass; its application inside the revision
that wrote it is.

### New findings

#### Blocker

**P6-1 (blocker) — no field on the record can carry a per-item continuous value, so the continuous
verdict path the plan commits to (MRR — the embedder pack's *only* verdict metric and its headline —
and `sep_z`) cannot be built, and the shipped comparison path fails silently in two directions
instead of loudly. Table E gates S3 done-condition 2 on the clamp; the interval it protects has no
producer and no renderer.**
*Evidence:* §3.8.1 declares `verdictMetrics = ["mrr"]`, `headlineMetric = "mrr"`; `-ml` §7.2 gives
the embedder's instrument as "paired bootstrap on per-query MRR", floor `n/a (continuous)`, and
§3.2d rules "the CI *is* the test — no separate significance test". §4 S1's target `ItemResult`
carries `counts: Mapping[str, int]`, `scoreable: Mapping[str, bool]` and `detail` — the last
"scorer-specific, **never read by `report.py`**". `paired_cluster_bootstrap` needs `diffs`,
per-query MRR differences; **the plan's own types close every route to them.** Shipped `report.py`
runs one binary loop over the whole family (`:606-655`): `_paired_rows` → `ItemResult.scored_outcome`
→ `PairedOutcomes.from_units` → `mcnemar_exact` → Holm → `stats.verdict`, whose every string is
percentage points (`stats.py:967, 984, 1008, 1015` via `_pp`). Executed, not inferred (Appendix F.2):
`scored_outcome("mrr")` returns `True` for a reciprocal rank of 0.5 and `False` for 0.0.
*Why it matters:* two silent outcomes, both on a green run. A scorer that declares `mrr` scoreable
booleanises it to *did this query retrieve anything*, and the report prints a McNemar `+X pp` verdict
for a metric the note says has no significance test — a different metric under the same name. A
scorer that does not (the natural reading, MRR being a `ContinuousMetric` aggregate) yields
`n_units == 0` for every unit, so `rp is None` and the pack's only verdict metric renders **"No
verdict: no paired data"**. `RetrievalAggregates.named_metrics()` (`results.py:190-195`) likewise
omits `separationZ`/`separationRaw`, so `sep_z` reaches no table either.
*Fix:* three things, all plan edits available today. **(a)** Decide and state where a per-item
continuous value lives — `counts` widened to `Mapping[str, float]`, or a new
`ItemResult.measures: Mapping[str, float]` — and say whether `scored_outcome`'s `bool | None`
contract gains a continuous sibling. **This is S1-local record shape, so §4 S1e's own "free only now,
because `results/runs/` does not exist" argument makes it due with the other five tables, not at S3**;
if it lands as a sixth table it carries rule 5's three things like the rest. **(b)** State in §4 S3
what renders a continuous verdict — `report.py`'s family loop is binary end-to-end, and a continuous
member has no McNemar *p* for the Holm ladder to rank. **(c)** Raise to `data-scientist` (rule 3: the
owner fixes it in their own file) that `-ml` §3.2e publishes three verdict strings, all carrying `pp`
and a McNemar clause, and none fits a pack whose floor is `n/a`. Then correct Table E's deadline
sentence, which currently reads as though the clamp were the obstacle between S3 and a `sep_z`
interval.

#### Majors

**P6-2 (major) — Table B's fourth residual, `{"model", "deterministic"}` → 0, fails on an
implementation Table B itself authorises. It is the same trap the architect just rejected `P5-2`'s
residual for being, in the table written to close `P5-2`.**
*Evidence:* the residual's two sites today are `test_fingerprint.py:213` and `:234` (verified, 2
lines). Table B's `fingerprint.py:137` row prescribes: "`ARM_KINDS` stays `{"model",
"deterministic"}` — **a literal** or a derivation over `REQUIRED_BY_SCHEMA`'s profile keys split on
`:`". Under the literal branch the edit *adds* `frozenset({"model", "deterministic"})` to
`fingerprint.py`, and the residual is 1, not 0 — and any test pinning the decoupling (the row's whole
point, since the coupled form makes every model record refuse) adds a second.
*Why it matters:* DC-12 re-runs this residual and asserts zero, so S1e cannot be signed off under
half the implementations the plan permits — and the implementer's exit is to override a
done-condition, which is precisely the behaviour §7 rule 5(b)'s new sentence exists to prevent.
*Fix:* one of two clauses. Either narrow the residual to the two assertion sites it means —
`grep -rFn '{"model", "deterministic"}' tests --include='*.py'` → 2 → 0, scoped to `tests/` — or drop
the literal branch from the `:137` row and require the derivation, which makes the unscoped residual
honest. The first is smaller; the second also removes a hand-transcribed set that §7 rule 4 would
have to keep swept.

**P6-3 (major) — Table E names four tests that do not call `_widen`, misses the three call sites its
edit actually breaks, and contradicts Table D on `paired_cluster_bootstrap`'s signature. It is
rule 5(a)'s token-free-site gap, in the table added to demonstrate rule 5.**
*Evidence:* `grep -rFn _widen modelbench tests --include='*.py'` → 7 lines; the four in
`test_stats.py` are `:743`, `:773`, `:915`, `:1110` — **`def` lines whose test *names* contain
"widening"/"widened"/"widens"**. All four are `verdict()`-level tests; none calls `_widen` (it is
private) or `paired_cluster_bootstrap`, so Table E's row "the four `_widen`-related tests | keep
their present behaviour under `clamp=(-1.0, 1.0)`" cannot be executed as written. The sites that do
break when `clamp` becomes required-with-no-default are `test_stats.py:1270`, `:1323` and `:1330` —
the three `paired_cluster_bootstrap(...)` test call sites — which carry no `_widen` and appear in
**none** of Table E's commands. Separately, Table D's row says `paired_cluster_bootstrap` and
`paired_bootstrap` are "kept, and **neither is touched** … **No edit**", while Table E's second row
changes `paired_cluster_bootstrap`'s signature at `stats.py:162`.
*Why it matters:* the failure is loud (a `TypeError`), so nothing ships wrong — but the implementer
discovers Table E's site list is fiction by running the suite, which is the trust in the mechanism
that rule 5 was adopted to build. And Table E leaves undecided what those three retained tests pass:
`test_paired_cluster_bootstrap_scales_the_half_widths_by_sqrt_deff` (`:1317-1325`) is the executable
statement of the √DEFF exactness argument Table D says must survive.
*Fix:* give Table E a second command — `grep -rFn 'paired_cluster_bootstrap(' modelbench tests
--include='*.py'` — with its per-file counts, replace the wrong test row with rows for `:1270`,
`:1323`, `:1330` naming the clamp each passes, say at which level the new defect-reproducing test is
written (`_widen` is private; `paired_cluster_bootstrap` is the public surface), and add four words
to Table D's row: *not touched **by this table***.

#### Minors

**P6-4 (minor) — §4 S1e is titled "**The** edit set over shipped code" and DC-12 enumerates "each of
§4 S1e's five tables", while DC-11's `ItemResult.timing` / `latencyMs`-as-property change sits
outside all five with no command and no residual.** `grep -rFc latencyMs modelbench tests` →
26 lines across 5 files (`results.py` 8, `test_results.py` 8, `test_report.py` 8, `conftest.py` 1,
`test_cli.py` 1); `ItemTiming`, `LatencyBlock`, `withheldFor` and `wallClockMs` appear **0 times** in
the tree. It is defensible under rule 5's *adds rather than retires* half — `latencyMs` is a required
positional today, so every construction site breaks loudly — but the section should say that, because
an implementer reading "the edit set" will not go looking for a sixth. *Fix:* one sentence in S1e's
preamble scoping the five tables to *retiring and re-keying* edits and naming §4 S1's signature block
plus DC-11 as the type-system-enforced remainder.

**P6-5 (minor) — Table B's second residual is stated as prose in a table whose thesis is that a
residual is a command with a count.** The row reads "`ARM_KINDS` still derived from the **forbidden
mapping** (`fingerprint.py:137`) | 1 | **0**", which is an eyeball check; it is also doing real work,
since residual 1 (`FORBIDDEN_BY_ARM_KIND` → 0) is satisfied by a rename that leaves `ARM_KINDS`
coupled to the renamed mapping. *Fix:* write it as the command it already is —
`grep -rFn 'frozenset(FORBIDDEN' modelbench --include='*.py'` → **1 → 0** (verified today;
`fingerprint.py:137` is the only match).

### Disposition of Pass 5's ten

All ten re-checked against v1.11 at `85a32e5` and against the tree at `5878014`; **ten fixed, none
unfixed, none carried.** Where a fix introduced a new defect it is named.

| # | Disposition | Evidence rechecked |
|---|---|---|
| **P5-1** (unscoped gate refuses every embedder arm) | **Fixed, on both sides as claimed** | §3.4.4a step 3a scopes the two refusals separately; §3.4.4a's `callSurface` bullet writes the cross-check predicate out and names it the universal pre-load refusal; §3.6's gate bullet carries its own scope paragraph naming `text-embedding-qwen3-embedding-0.6b` and why the redundant half is kept. Both sentences read the same way. |
| **P5-2** (rule 5 over-claimed; Table B incomplete) | **Fixed — and the fix carries a new trap** | §7 rule 5 restated (no-forgetting over token-carrying sites; 5(a) second command; 5(b) second-form residual; two bounds); DC-12 reduced to the sound direction with the converse denied explicitly; Table B at six commands / eleven *found by* rows / four residuals. Residue: **P6-2**. |
| **P5-3** (`paired_cluster_bootstrap` undispositioned) | **Fixed — routed to the note and ruled *keep*** | Table D's new row carries both enumerating commands (13 and 8 lines, reproduced), the disposition *no edit*, the entry-point/engine distinction from `-ml` v1.14 §3.4 Rule 4, and an explicit "deleting them is not authorised". §4 S1's signature block carries the same distinction as a comment. Residue: the collision with Table E — **P6-3**. |
| **P5-4** (§3.4.2 contradicts the capture order) | **Fixed** | §3.4.2's paragraph now cites §3.4.4a as owner and attaches the `loadedContextLength` constraint to the **step 8** read; the v1.8 sentence that forbade step 3a is recorded as withdrawn rather than silently dropped. |
| **P5-5** (co-presence has no runtime disposition) | **Fixed** | §4 S2 rule **(iv-c)**: an item with `stats` but `promptTokens` absent or `≤ 0` leaves `statsCoveredCount` **and** all three medians, "a *disposition*, not an assertion: nothing raises"; rule (iv)'s identity is now the inequality `statsCoveredCount ≤ latencyItemCount − latencyWithheldForNoResponse`; §7 re-paired to note **v1.14** (`ca69cb1`), which is current. |
| **P5-6** (`censoringExact` clause 1 unevaluable) | **Fixed, and the sweep is real** | `withheldFor` is `Literal["load","timeout","no_response"] \| None` at §4 S1 (`:2125`); swept surfaces verified present at §3.6 clause (ii) (`:1317`) and the fourth disposition (`:1340`), §3.6's `censoringExact` bullet (`:1365`), DC-11's four absent cases (`:2528`), §4 S2 (vi) three-states-two-counts (`:2957-2963`), §4 S2's `censoringExact` bullet (`:2970-2977`), §5 test 15b's four branches (`:3330-3357`), Appendix A (`:4079`), §7 (`:4012`). One counter retained. |
| **P5-7** (rule (i) "the three figures") | **Fixed** | (i) now reads "the three **wall-clock** figures", names the four `stats`-derived ones, and states that both `None` causes are indistinguishable in the block with `statsCoveredCount` as the discriminator. |
| **P5-8** (unit boundary raises on absent `stats`) | **Fixed, both homes** | §3.6's unit boundary (`:1207-1208`) and §4 S2's restatement (`:2814`): each derived field is `None` when its source key is absent, never `0`, and `ChatResult` construction never raises on a missing or partial `stats`. |
| **P5-9** ("no timing at all" vs rule (vi)) | **Fixed** | §3.6's timeout clause (ii) and fourth disposition both now say the item carries an `ItemTiming` whose only populated field is `withheldFor`; §4 S1's comment keeps `timing is None` for the arm-level case. |
| **P5-10** (`sampling.seed` in the fingerprint) | **Fixed, before S3 as asked** | §3.4.2 states the seed reaches the fingerprint **transitively through `packContentHash`** (`REQUIRED_NONEMPTY` on all three profiles), adds no field, and names what the transitive route does not buy. |

### The recurring shape, answered

The brief's fourth question — *a fix that reopens a closed finding, now three revisions running.*
Verified rather than accepted: v1.11's `P5-1` fix **is** stated on both sides, and the `P5-6`/`P5-9`
sweep **is** complete across all nine surfaces it claims. So the specific defect Pass 5 named does not
recur. **But the shape does, twice, and both instances are now internal to a single revision rather
than against an earlier one:** Table B's residual 4 is the trap forbidden by rule 5(b), written in the
same edit as rule 5(b) (`P6-2`); Table E contradicts Table D's "neither is touched" and misses the
sites 5(a) requires a second command for, written in the same edit as 5(a) (`P6-3`). The pattern has
migrated from *sweep the other section* to *satisfy the rule you just wrote*. A cheap counter for
v1.12: after writing a residual, ask which authorised implementation makes it non-zero, and run each
of a new table's commands once before publishing its site rows.

### The item-5 sweep

Asked for: other shipped S1 code correct for its current caller and wrong for a caller the plan
commits to adding. **One instance, and it is larger than `_widen`'s** — the whole comparison surface
is binary-only while the plan's first real pack's only verdict metric is continuous (`P6-1`). Its
sub-parts, so the fix is scoped rather than open-ended: `ItemResult.counts` typed `Mapping[str, int]`;
`scored_outcome` returning `bool | None` with `counts[metric] > 0`; `_paired_rows` building `a_ok`
/`b_ok` booleans; `mcnemar_exact` computed for every family member; `stats.verdict`'s four strings in
`pp`; `RetrievalAggregates.named_metrics()` omitting `separationZ`/`separationRaw`. Checked and
**clear**: `results.py`'s `_percentile` and `_index_row` (Table C owns them), `_widen`'s two callers
(both in Table E), the Arms table's `ContinuousMetric` branch (`report.py:578-583`, renders
`n=…, mean, —` and prints no Wilson interval — correct), and DC-10's selector, which skips
`ContinuousMetric` by `isinstance` at `report.py:211` and so is not a source of this defect.

### What's solid

- **The ten dispositions are genuine.** Every one was checked at the tree or at the cited section,
  not against the plan's own change list, and every one holds. Three consecutive revisions of
  fourteen / ten / ten closed findings with nothing carried is a real record.
- **Rule 5's second formulation is the right rule.** It states a narrower property than it wants to,
  denies its own converse in DC-12, and adds two bounds that are both correct. It is now a mechanism
  a later revision can lean on without being misled by it.
- **Table B is the demonstration.** Commands 4–6 close exactly the blind spot Pass 5 measured, the
  *found by* column makes each row checkable in isolation, and the refusal to fake an `armKind`
  residual — naming the type system as what stands in its place — is the honest move.
- **The `P5-3` and `P5-6` routings were correct.** Both were sent to `data-scientist` rather than
  decided by the architect; a public statistical function's deletion and a censoring predicate's
  category are not the plan's to rule, and §7 rule 3 was applied as written.
- **Table E's design reasoning is right even though its enumeration is not.** Required-with-no-default
  over a `(-1.0, 1.0)` default is the correct call for a value that fails silently in the printing
  direction, the rejected manifest-field alternative carries a real reversal trigger, and the residual
  over `max(-1.0, point` is a properly-formed second-form residual.

### Open questions

1. **Is the continuous verdict path S1-local or S3-local?** `P6-1`(a) is a record-shape change, which
   S1e's own "free only now" argument says is due now; `P6-1`(b)/(c) are renderer and method work that
   could legitimately sit at S3. The architect should split them explicitly rather than let the whole
   thing default to S3, because only the first has a deadline that has already started running.
2. **Does `-ml` §3.2e owe a fourth verdict string?** The three published ones all carry `pp` and a
   McNemar clause. The embedder's verdict has neither. That is the note's call, not the plan's, and
   the plan should raise it rather than invent one.

## Appendix F — Pass 6: what was re-run and read

**F.1 — commands, all against `model-bench/` at `5878014`** (`git diff --stat 5878014..HEAD --
model-bench/` empty; suite **389 passed in 5.49 s**; `ruff check .` clean).

| Check | Command / location | Result |
|---|---|---|
| `P5-2`'s rejected residual | `grep -rFn arm_kind … \| grep -F '"model"'` | **2** — `conftest.py:148` (`arm_kind: str = "model"`, a parameter default) and `:160` (`model_fields() if arm_kind == "model"`). Both can survive a faithful edit → the architect's rejection stands |
| Table B residual 3 | `grep -rFn 'REQUIRED_BY_SCHEMA[1]["model"]' …` | **3** — `test_fingerprint.py:34`, `:41`, `:144`; the key is retired by the re-key, so → 0 is sound |
| Table B residual 4 | `grep -rFn '{"model", "deterministic"}' …` | **2** — `test_fingerprint.py:213`, `:234`. The `:137` row's *literal* branch re-adds the string in `fingerprint.py` → **P6-2** |
| Table B residual 2, as a command | `grep -rFn 'frozenset(FORBIDDEN' modelbench --include='*.py'` | **1** — `fingerprint.py:137`. Works; the table states it as prose → **P6-5** |
| `ARM_KINDS` line numbers | `grep -rn ARM_KINDS …` | `fingerprint.py:137`, `:162` — v1.11's correction of v1.10's `:136`/`:160` reproduces |
| Table E's enumeration | `grep -rFn _widen …` | **7** — `stats.py:188`, `:191`, `:264`; `test_stats.py:743`, `:773`, `:915`, `:1110`, all four **`def` lines matching on the test name**, none a call → **P6-3** |
| the sites Table E breaks | `grep -rFn paired_cluster_bootstrap …` | **13** — test call sites `:1270`, `:1323`, `:1330` carry no `_widen` and are in no Table E command → **P6-3** |
| DC-11's edit set | `grep -rFc latencyMs …` / `grep -rn 'ItemTiming\|LatencyBlock\|withheldFor\|wallClockMs'` | **26** lines / 5 files; **0** occurrences of the new types → **P6-4** |
| the clamp | `stats.py:191-202` | `max(-1.0, …)`, `min(1.0, …)`; callers `:188`, `:264`. Table E's premise holds |

**F.2 — `P6-1`, executed.** A three-line probe against the shipped package
(`.venv/bin/python`, `ItemResult` constructed directly):

```
scored_outcome('mrr') with counts={'mrr': 0.5}  ->  True
scored_outcome('mrr') with counts={'mrr': 0.0}  ->  False
scored_outcome('mrr') with no declaration       ->  None
```

`results.py:119` types `counts: Mapping[str, int]`; `:146-152` returns `self.counts[metric] > 0`.
`report.py:606-655` runs `_paired_rows` → `PairedOutcomes.from_units` → `mcnemar_exact` → Holm →
`stats.verdict` for **every** `verdictMetrics` member, and `report.py:637-648` renders
`_NO_PAIRED_DATA` when `outcomes.n_units == 0`. `-ml` §7.2's embedder row: instrument "paired
bootstrap on per-query MRR", floor `n/a (continuous)`; §3.2d: "for continuous metrics the CI *is* the
test". §3.8.1: `verdictMetrics = ["mrr"]`, `headlineMetric = "mrr"`.

## Pass 7 — 2026-09-07

**Re-gated:** plan **v1.13** (`fbe5741`, +300/−91) against note **v1.17** (`1fbdb6f`), the
requirements, and the shipped S1 tree — `model-bench/` at HEAD is byte-identical to `5878014`
(`git diff --stat 5878014..HEAD -- model-bench/` empty), re-run here: **389 passed in 5.49 s**,
`ruff check .` clean (working directory `model-bench/`, per root `AGENTS.md`). Weighted to the
brief's five items: Pass 6's five dispositions under v1.13, whether v1.13 stayed inside its four
deltas, Table G's design, delta 1's negative assertions, and the item-5 sweep. Findings carry the
prefix **`P7-`** (*plan-gate P7-n*; `docs/reviews/small-model-benchmarking-impl.md`'s are
*impl-gate P4-n*). Written by a reviewer who did not write Passes 1–6.

**CPG:** considered, not relevant — no `model-bench` CPG exists (the instance loads `cpg_falkorchat`
and `cpg_deprecated_salesperson` only), and v1.13 makes no structural claim about `falkor-chat`. All
grounding below is `grep`/`sed`/`pytest`/an executed probe against `model-bench/` at `5878014`
(Appendix G).

**Verified, not assumed.** Every count cited below was re-run today. `P7-1`'s three failure modes
were **executed**, not read (G.2). No model was loaded in LM Studio; `loadedContextLength` remains
S2's R-1 probe.

### The question this pass exists to answer

**Is the plan implementable, and may S2 be dispatched?**

**The plan is implementable. S2 may not be dispatched yet, and the gap is one small revision.**

Implementable: every stage's done-conditions are concrete, every §4 S1e table names commands with
counts that reproduce, and the two seams that were open at Pass 6 — the continuous producer's
signature and §3.3's enforcement point — are closed by note v1.16/v1.17. Nothing in this plan now
requires an implementer to re-derive a design.

Not yet dispatchable, for one reason: **`P7-1`**, a blocker in **Table F**, which is S2's own
prerequisite — §7's own S2 bullet says "S2's scorers write it" of `measures`, and S2's runner
constructs the `ItemResult` Table F changes. Table F leaves the **stored form of
`DistributionSummary` undecided** and names three of the four shipped sites that break under it
nowhere, one of which fails **silently**. That is a stored-record-shape decision, and it is due now
on §4 S1e's own *free only now* argument — the same argument that moved `P6-1(a)` into S1. It is a
**plan edit available today**, not unbuilt work: a decided JSON shape, three site rows, one
enumerating command and one DC-13 clause. `P7-2` and `P7-3` are minors and ride in the same
revision.

**S2's own stage is clean.** §4 S2's six rules, the nine `LatencyBlock` invariants, DC-11 and the
R-1 deferral carry no finding this pass. Once `P7-1` lands in the plan, the sequence is the S1 fix
round (Tables A–G + DC-11 + DC-13) and then S2, in that order — and S2 must not be written before
that round, because §4 S1e's "nothing S2 constructs" is true of Tables A–E and false of Table F.

**Verdict: needs changes.** 1 blocker, 0 majors, 2 minors, 0 nits. This is the smallest finding set
of the seven passes and the first in which no finding is a defect of the *mechanism* — rule 5, the
tables and the residuals all did their job this round; `P7-1` is the mechanism's blind spot, which
is a different and more useful thing to have found.

### Disposition of Pass 6's five, under v1.13

All five closed at v1.12 and **all five survive v1.13**, re-checked against the tree rather than
against the plan's change list. v1.13 touches none of the closing text except the two count repairs
noted.

| # | Survives? | Rechecked |
|---|---|---|
| **P6-1** (no carrier for a per-item continuous value) | **Yes — and v1.13 strengthens the renderer half** | Table F intact; §4 S1's loop now hands `diffs` to Rule 8's `continuous_verdict()` rather than describing a producer; DC-13(a)–(d) unchanged, (e) added. **Residue: `P7-1`**, a gap in Table F itself, first readable at this pass — Pass 6 predates Table F |
| **P6-2** (Table B residual 4 was a trap) | **Yes** | Narrowed form re-run: `set(REQUIRED_BY_SCHEMA[1]) == {"model",` → **1** (`test_fingerprint.py:213`). Unwritable after the re-key, and a test pinning a decoupled `ARM_KINDS` by value cannot match it — the trap is genuinely gone under **both** branches of the `:137` row |
| **P6-3** (Table E's fictional site list) | **Yes** | Table E's two commands re-run: `_widen` → **7**, `paired_cluster_bootstrap(` → **5** (`stats.py:162`, `:263`; `test_stats.py:1270`, `:1323`, `:1330`). The four `test_stats.py` `_widen` lines are named as non-sites; Table D's row reads *not touched **by this table***. Residual `max(-1.0, point` → **1** (`stats.py:200`) |
| **P6-4** (§4 S1e scoped as "**the** edit set") | **Yes — and the count inconsistency behind it is repaired** | The preamble's scoping paragraph stands; v1.13 fixed the pre-existing five/six split (see below) and both now read **seven** |
| **P6-5** (a residual stated as prose) | **Yes** | `frozenset(FORBIDDEN` → **1** (`fingerprint.py:137`), the only match |

Table D's and Table F's residuals also reproduce: `bootstrap_seed` **29** lines, `cluster-bootstrap`
**27**, `separationRaw: float | None` **1**, `separationZ: float | None` **1**.

### Did v1.13 stay within its four deltas? Yes, and the self-reported list is complete

I classified all **24 hunks** of `git diff 5b67416 fbe5741` against the four deltas plus the five
items v1.13 reported unprompted. **Every hunk lands in one of the nine, and nothing lands outside.**

- Deltas 1–4 account for hunks at §3.3 (iv), §3.8.1, §4 S1's `compare_report` block and signature
  sketch, DC-12, DC-13(e), the S1e preamble counts, Table D's cross-reference, **Table G**, §5 test
  11d, §7's *Note v1.16* block, §7 rule 5, and Appendix A's `ContinuousMetric`/`DistributionSummary`
  row.
- The five reported extras all reproduce as described: the `-ml` §3.2f trip-hazard raise **closed**
  (Table D, now citing v1.16's six-site sweep); **one new** §7 rule 3 raise (the `support` seam);
  the **pre-existing** five/six preamble split, which v1.12 half-swept — confirmed in the v1.12 text
  (`All five tables below` two paragraphs above `All six are S1-local`) and now both **seven**; two
  mechanical prose repairs (the `P6-4` row's *then-six*, and *a seventh* → *an eighth*); and the
  signature-sketch/α-naming consequence (`levels` on both engine signatures, plus Table G's explicit
  refusal to name an α).
- Version pairing and the change-list paragraph are the revision's own bookkeeping.

**No unreported overreach.** The self-report is accurate, and the pre-existing inconsistency was
repaired rather than propagated — which is the behaviour that makes "stay small" auditable.

### New findings

#### Blocker

**P7-1 (blocker) — `DistributionSummary` has no stored form. Table F retypes `separationRaw` /
`separationZ` to it and puts it into `named_metrics()`, but names none of the four shipped sites
that read a metric's `.mean` or gate on its `"type"` tag — and one of them, `_decode`, fails
*silently*, returning a raw `dict` where a metric belongs. The JSON shape of two published figures
is therefore left to whoever implements S1e, at the one moment §4 S1e says record shape is free.**
*Evidence (executed, G.2).* Shipped `results.py`: `:354-359` `_metric_to_dict` tags `"binary"` and
falls through to a bare `return {"type": "continuous", …, "mean": m.mean, …}` → **AttributeError**;
`:385` `_decode` decodes only `value.get("type") in {"binary", "continuous"}` → a third tag **falls
through and is returned as a `dict`, silently**; `:584` `_index_row`'s metrics cell is
`… if isinstance(m, BinaryMetric) else f"{m.name}={m.mean:.4f}"` over `named_metrics()` →
**AttributeError**, so `index.csv` cannot be written for an embedder run; `:369`
`_metric_from_dict` returns a `ContinuousMetric` for anything not `"binary"` and Table F edits it
only for `support`. **Why Table F's six commands miss them:** command 3 greps the *name*
`ContinuousMetric` (`:359`/`:385` spell it as the string literal `"continuous"`), and command 6 —
the one written as rule 5(a)'s token-free coverage for exactly this defect class — is
`isinstance(metric, BinaryMetric)`, whose variable name is `m` at `results.py:355` and `:584`. So
Table F found **one** of the three bare-`else` `.mean` readers (`report.py:583`) and missed two.
*Why it matters:* S3 done-condition 1 requires a stored result and done-condition 2 requires a
rendered `DistributionSummary` row; a faithful implementation of Table F as written produces a tree
that raises on write and, once "fixed" with a third `"type"` tag, silently round-trips a dict.
Deciding that shape is a **plan** decision under §3.4.3 and under S1e's own *free only now*
deadline — the P6-1(a) argument, unchanged.
*Fix, all plan edits available today:* **(a)** State `DistributionSummary`'s stored shape in
Table F — its `"type"` tag and its keys, and whether `support` is stored or re-stated by the
scorer — and whether `benchSchemaVersion` moves (Table F says it does not, for `measures`; the
`separationRaw`/`separationZ` retype changes an **existing** key's type, which is a different
claim). **(b)** Three site rows: `results.py:354-359` (encoder), `:385` (`_decode`'s literal set,
flagged as the **silent** one), `:584` (`_index_row`'s cell — say what an index cell prints for a
distribution), plus the `"type"`-tag half of the `:369` row. **(c)** A seventh command that reaches
them. Two work and both were run today: `grep -rn '"continuous"' modelbench --include='*.py'` →
**2** (`:359`, `:385`), and — sharper, because it enumerates the defect rather than a spelling —
`grep -rn '\.mean' modelbench --include='*.py'` → **3** (`report.py:583`, `results.py:359`,
`:584`), which is exactly the set of bare-`else` `.mean` readers. **(d)** A DC-13 clause: a
`DistributionSummary` round-trips through `to_dict`/`from_dict` **as a `DistributionSummary`** —
the assertion that fails on `_decode`'s silent pass-through — and one asserting the `index.csv`
cell. **(e)** Correct §4 S1e's "nothing S2 constructs", which is true of Tables A–E and false of
Table F: `ItemResult` is constructed by S2's runner, and `ContinuousMetric` gains a
required-with-no-default `support` that every scorer must state.

#### Minors

**P7-2 (minor) — Table G retires two literals and states a residual over one, and its new test
passes on the half-applied edit its residual misses. The table's claim that accepting-and-ignoring
`levels` is "the only way this edit can be applied unfaithfully and still compile" is false.**
*Evidence:* `stats.py:159` is `return _percentile(means, 2.5), _percentile(means, 97.5)`; the site
row says "replacing the two literals", the residual covers one —
`_percentile(means, 2.5)` → **1 → 0** (verified). An implementer who wires `levels[0]` and leaves
`97.5` compiles, satisfies the residual, and — for a `k = 2` family whose pair is narrower on both
sides — returns an interval that **is** strictly wider by width, so the prescribed new test
("returns a **strictly wider** interval") passes too. The upper bound then takes no family
correction, silently, in the printing direction.
*Fix:* two clauses. Add the symmetric residual `grep -rFn '_percentile(means, 97.5)' modelbench
--include='*.py'` → **1 → 0** (verified: `stats.py:159`, the only match — `means` scoping keeps
`cluster_bootstrap`'s `_percentile(rates, 97.5)` at `:292` out, exactly as for the 2.5 form); and
state that the new test asserts **both bounds move outward**, not that the width grows.

**P7-3 (minor) — delta 1 requires every member of a refused mixed family to print its number under
`exploratory — no significance claim`, and the only shipped home for that label structurally
excludes family members and prints no number.** `report.py:763-777` builds `exploratory` with
`if m.name not in family` (`:767`) and emits
a line reading ``- `<name>` — exploratory — no significance claim`` (`:776`) — a bare name, no figure.
A refused family's members are *in* `family`, so they reach neither. Neither §3.3 (iv), §4 S1's
pass-1 bullet, DC-13(e) nor §5 test 11d names the site or says what changes, so the implementer
decides where the numbers land and then writes test 11d against that choice — the risk being a test
written to match the code rather than the plan. *(The Arms table already prints every member's
figure unconditionally, so one reading of DC-13(e) is satisfied today; that ambiguity is the
finding, not a missing capability.)*
*Fix:* one clause on §3.3 (iv) or DC-13(e) naming `report.py:767`'s family filter and `:776`'s
name-only line as the site, and stating whether a refused member is listed in the Exploratory
section with its figure or labelled in place in the Arms table. **DC-13(e)'s discriminating power
is unaffected either way** — see the adjudication below.

### The four judgements the brief asked for

**(1) The safety classification holds on the note's replacement argument — shipping the interim
`clamp=None` is safe.** Checked against shipped code and against Table E's *specified* semantics,
not against the prose. `_widen` is
`(max(-1.0, point − (point−lo)·scale), min(1.0, point + (hi−point)·scale))` (`stats.py:198-202`),
and Table E's edit makes those two literals the `clamp` argument with `None` meaning *do not clamp*.
At `scale == 1.0` the two multiplications are the identity, so `clamp=None` returns `(lo, hi)` and
`clamp=(-1.0, 1.0)` returns `(max(-1, lo), min(1, hi))`; the two agree whenever the unwidened bounds
lie inside the clamp, and an unwidened bound **is** a bootstrap percentile of per-unit differences,
so it lies inside the difference's own support by construction. **The argument is metric-generic and
census-free, as the note claims**, and it is strictly stronger than v1.13's pack-census version:
it also covers a metric whose difference support is *wider* than `[-1, 1]`, where the census version
says nothing and the shipped hard-coded clamp is actively wrong. Two riders, neither changing the
conclusion. The float identity `a − (a − b)·1.0 == b` is not guaranteed by IEEE-754, though it held
in **20 000/20 000** random pairs here (G.2) — irrelevant to the safety claim, since both candidate
clamps traverse identical arithmetic and differ only in a `max`/`min`. And the interim is now
**moot** rather than merely safe: note v1.17 rules `support` in, so the correct build takes
`support` and derives nothing, and the safe interim only ever has to hold for a build written
against v1.13 alone.
**Table E's required-with-no-default `clamp` still does real work, confirmed independently.**
`verdictMetrics = ["mrr"]` (§3.8.1), so `sep_z` is reported and not verdicted, and §3.8.1 wires its
exploratory cross-model comparison through `paired_cluster_bootstrap` **directly** with
`clamp=None` and `levels=(2.5, 97.5)`. That call site exists in the plan today and takes no
`support`. Two callers, two surfaces, and collapsing either into the other would either re-hard-code
a clamp on the verdict path or hand the exploratory path a metric it has no aggregate to ask.

**(2) Table G — the scoping is principled, and it belongs as its own table.**
*Scoping:* every weaker form fails, and I ran them. `2.5` alone → **6** lines including
`stats.py:242`/`:243` (a docstring) and `:292`; `_percentile(` → **6** including Table C's two
`results.py` copies. `means` is the only available discriminator that both reaches zero on a
faithful edit and cannot be held above zero by `cluster_bootstrap`'s surviving
`_percentile(rates, 2.5)`. It is not a trap in `P6-2`'s sense — a correct edit genuinely retires the
literal at `:159`. One disclosed coupling worth a clause: `means` is a **local variable name**, so a
faithful edit that renamed it would zero the residual without retiring anything. That is the
direction DC-12 already denies (*residual zero ⇏ nothing missed*), so it is a weakness rather than a
defect; `P7-2`'s symmetric residual narrows it further.
*Own table:* yes. Table E's subject is a **clamp**, derived from a metric's support; Table G's is a
**quantile pair**, derived from family size. They share `stats.py:162`'s signature line and three
test call sites, but the plan already has a stated convention for that (`test_stats.py:1270` meets
Tables D and E, "neither owns it alone") and Table G states the landing order is faithful either
way. Merging them would give one table two unrelated "the defect in one sentence" claims and one
residual over two different retirements — strictly worse. Table G's command 2 also reaches
`paired_bootstrap`'s **own** callers (`:187`, `test_stats.py:890`, `:891`, `:1321`), which neither
`_widen`'s name nor `paired_cluster_bootstrap`'s ever touches; that alone justifies a separate
enumeration. All three commands and every site row reproduce (G.1), and every line the three
commands return is dispositioned somewhere in the table — no transcription-without-reading this
time.

**(3) Delta 1's negative assertions do catch the mis-build — and the set is complete against all
four plausible ones.** Mapped against the shipped renderer:

| Mis-build | Caught by |
|---|---|
| DC-10's mechanism — exclude the arms | **both negatives**: "both arms still render" and "`INVALID RESULTS EXCLUDED` empty" |
| v1.12's withdrawn wording — drop the offending members, verdict the rest | "**no member is verdicted**" (the survivors would be) |
| Raise on a mixed family | "both arms still render" (no report is produced at all) |
| Do nothing but suppress verdicts | "each is named with its resolved kind", and the exploratory label |

The brief's reasoning — that the two negatives are what fail under a DC-10 exclusion — is exactly
right, and asserting the positive alone would have caught none of the four. **No vacuity hazard**
of DC-10's kind (whose test row has to pin a role's unit kind or pass while testing nothing): the
only assertion a data-starved fixture could satisfy vacuously is "no member is verdicted", and such
a fixture fails the label assertion, since a member with no paired data renders `_NO_PAIRED_DATA`
(`report.py:637-648`) and not `exploratory — no significance claim`. The one real gap is *where*
that label prints, which is `P7-3` and is a specification gap rather than a discriminating-power
one.

**(4) The convergence question, answered plainly: there was remaining static risk, this pass
reduced it, and the specific class is now exhausted for shipped S1 code.**
`P7-1` is a fifth instance of the shape — shipped code correct for its current caller, wrong for a
caller the plan commits to adding — so the answer to "could a static pass still find one?" was
**yes**, and a Pass 7 that had not swept for it would have shipped it. **Why the previous sweeps
missed it, and why that is fixable rather than luck:** §7 rule 5's commands are *token-based* — they
grep a **name** — and this defect class is *attribute-based*: the shipped code reads `.mean`, or
tests a `"type"` string, or matches on a variable spelled `m` rather than `metric`. Table F's
command 6 was written as rule 5(a)'s token-free coverage and still missed two of the three sites,
because it pinned a variable name. The counter is mechanical and generalises: **for every new type a
table introduces, add a command over the attribute the shipped code reads on that type's siblings,
not only over the type's name** — here, `grep -rn '\.mean' modelbench` returns exactly the three
bare-`else` readers and nothing else.
Having run that procedure over the full set of callers this plan commits to adding —
`DistributionSummary` into `named_metrics()` and the (de)serialisers; `measures` / `scored_value`;
`ContinuousMetric.support`; `levels`; `clamp`; `ContinuousVerdict` into the verdict renderer;
`DecidedBy`'s third member; and `_percentile` at Bonferroni-adjusted fractional levels — **only
`DistributionSummary` yields a finding.** Checked and clear, each executed or read at a cited line:
`_percentile` handles fractional levels correctly (`1.25` → index 25 of 2 000, G.2), so Table G's
new caller is safe; `report.py:763-777`'s exploratory block reads only `.name`, so a
`DistributionSummary` is safe there (its gap is `P7-3`, a different one); `report.py:701-704`'s
`decided_by` branch is correct for a third token because the continuous strings carry their
provenance inside the string; the verdict renderer's `Verdict` field reads sit inside Table F's
`report.py:606-700` row; and `DecidedBy`'s three literal sites are Table D's.
**So: the residual risk is now the kind only execution finds** — LM Studio's real responses, the
runner's timing capture, `loadedContextLength`, and whether the packs' arithmetic reproduces the
copied baseline. **A Pass 8 is not worth a seventh full gate.** What is worth its cost is a narrow
re-check of a v1.14: the three findings above, plus the two plan-side deltas note v1.17 already
owes (swap the census premise for the scale-1.0 identity; make the loop *forward*
`ContinuousMetric.support` and derive nothing). That is a ten-minute confirmation, not a discovery
pass.

### What's solid

- **v1.13 did what a small revision should.** +300/−91 against +521/+540/+738, all four deltas
  landed, five self-reported extras all real, and a pre-existing inconsistency repaired on the way
  past. The self-report is what makes "stay small" checkable, and it checks out.
- **Table G is a well-built table.** Three commands whose union is exactly the edit's site set, every
  returned line dispositioned (including a named non-site at `stats.py:292` with the reason it is
  one), the `stats.py:263` ordering stated as faithful either way, a "why required rather than
  defaulted" paragraph with a rejected alternative, and a residual verified both directions. It is
  the first §4 S1e table to arrive without a gate finding against its enumeration.
- **Citing Rule 8 instead of restating it is the right call**, and §7 rule 2 is now being applied
  rather than quoted: v1.12 described the producer because no specification existed, v1.13 replaced
  the description the moment one did, and named the single thing the note left here (the
  `Verdict | ContinuousVerdict` union) rather than leaving it implied.
- **Delta 1's assertion design.** Choosing the two negatives over the easy positive is the sharpest
  test-design decision in this plan, and it holds up against all four mis-builds I could construct.
- **The `support` raise was raised correctly.** §7 rule 3 routed a method question to its owner with
  a recommendation and an interim, and the owner endorsed the conclusion while replacing the
  premise. That is the mechanism working; the census premise really would have gone stale silently.

### Open questions

1. **Does `separationRaw`/`separationZ` retyping move `benchSchemaVersion`?** Table F rules *no
   bump* for `measures` on the ground that it is additive and absent-safe. The separation retype is
   not additive — it changes an existing key's stored type from a number to an object — and while no
   stored record exists to migrate, the answer belongs in the plan rather than in an implementer's
   head. Part of `P7-1`(a); flagged separately because it is the one part that is a **judgement**
   rather than a transcription.
2. **Which document absorbs note v1.17's two plan-side deltas, and in the same revision as
   `P7-1`?** Both are known to the coordinator and neither is a finding here. Bundling them with
   this pass's three keeps the plan/note pair from going two revisions out of step.

## Appendix G — Pass 7: what was re-run, read and executed

**G.1 — commands, all under `model-bench/` at `5878014`** (`git diff --stat 5878014..HEAD --
model-bench/` empty; **389 passed in 5.49 s**; `ruff check .` clean).

| Check | Command | Result |
|---|---|---|
| Table G cmd 1 | `grep -rFn 97.5 modelbench tests --include='*.py'` | **2** — `stats.py:159`, `:292`. Matches the table, and `:292` is the named non-site |
| Table G cmd 2 | `grep -rFn 'paired_bootstrap(' …` | **5** — `stats.py:148`, `:187`; `test_stats.py:890`, `:891`, `:1321`. Every line has a site row |
| Table G cmd 3 | `grep -rFn 'paired_cluster_bootstrap(' …` | **5** — `stats.py:162`, `:263`; `test_stats.py:1270`, `:1323`, `:1330`. Table E's command 2, verbatim, as claimed |
| Table G residual | `grep -rFn '_percentile(means, 2.5)' modelbench …` | **1** — `stats.py:159`. Symmetric form `…, 97.5)` also **1**, same line → `P7-2` |
| rejected scopings | `grep -rFn 2.5 …` / `grep -rFn '_percentile(' …` | **6** / **6** — both reach `:292` or `results.py`, so neither can reach zero. `means` is the only working discriminator |
| Table E | `_widen` → **7**, `paired_cluster_bootstrap(` → **5**, `max(-1.0, point` → **1** (`stats.py:200`) | `P6-3` closure holds |
| Table B | `frozenset(FORBIDDEN` **1** · `REQUIRED_BY_SCHEMA[1]["model"]` **3** · `set(REQUIRED_BY_SCHEMA[1]) == {"model",` **1** | `P6-2`/`P6-5` closures hold |
| Table D / F | `bootstrap_seed` **29** · `cluster-bootstrap` **27** · `separationRaw: float \| None` **1** · `separationZ: float \| None` **1** | reproduce |
| `P7-1`'s missed sites | `grep -rn '\.mean' modelbench --include='*.py'` | **3** — `report.py:583` (Table F has it), `results.py:359`, `:584` (**neither is in any table**) |
| `P7-1`, second form | `grep -rn '"continuous"' modelbench --include='*.py'` | **2** — `results.py:359`, `:385`; neither spells `ContinuousMetric`, which is why command 3 misses them |
| `P7-1`, why command 6 misses | `grep -rn 'isinstance(.*BinaryMetric' modelbench …` | **6** — 3 in `report.py` (variable `metric`, Table F's command 6) **+ 3 in `results.py`** (variable `m`: `:355`, `:373`, `:584`) |
| `P7-3` | `report.py:767` / `:776` | `if m.name not in family`; the label line emits a bare metric name — no figure, family members filtered out |

**G.2 — executed** (`.venv/bin/python`, a throw-away probe constructing the plan's
`DistributionSummary` exactly as §4 S1 specifies it and calling shipped functions; nothing in
`model-bench/` was modified):

```
encode : RAISES AttributeError 'DistributionSummary' object has no attribute 'mean'   # results.py:359
decode : dict -> {'type': 'distribution', ...}                                        # results.py:385, SILENT
index  : RAISES AttributeError 'DistributionSummary' object has no attribute 'mean'   # results.py:584
pct    : _percentile(xs, 1.25)=0.012506  2.5=0.025013  98.75=0.987494   (B=2000)      # fractional levels OK
_widen identity at scale 1.0: a-(a-b)*1.0 == b in 20000/20000 random pairs
```

**G.3 — read.** Plan v1.13 §3.3 (iv), §3.8.1, §4 S1 (`compare_report` block, signature sketch),
DC-12, DC-13, §4 S1e preamble and Tables D/E/F/G, §4 S3, §5 test 11d, §7 (*Note v1.16* block, the
`support` raise, rule 5, the Pass 6 disposition table), Appendix A. Note v1.17's §3.4 Rule 8 in
full. `git diff 5b67416 fbe5741` in full, classified hunk by hunk (24 hunks). Coordination doc
U32/U33 ledger entries. Shipped `stats.py:140-300`, `results.py:180-250`, `:350-400`, `:570-600`,
`report.py:600-712`, `:758-778`.

## Pass 8 (narrow) — 2026-09-07

**Scope, as commissioned:** the v1.14 delta only — `git diff 57b5715 d6556c3 --
docs/plans/small-model-benchmarking.md` (+288/−75, **25 hunks**, all read and classified) against
note **v1.17** (`1fbdb6f`) and the shipped tree. This is the narrow re-check `## Pass 7` recommended
in place of an eighth full gate; it does **not** re-gate the plan as a whole and reopens nothing
closed in Passes 1–6. Four questions were put to it: is `P7-1` closed, do commands 7 and 8 enumerate
the class, is `P7-2`'s generalisation complete across all seven tables, and are `P7-3` plus note
v1.17's two deltas sound. Findings carry the prefix **`P8-`** (*plan-gate P8-n*).

**CPG:** considered, not relevant — no `model-bench` CPG exists (the instance loads `cpg_falkorchat`
and `cpg_deprecated_salesperson` only) and v1.14 makes no structural claim about `falkor-chat`.

**Verified, not assumed.** Every count v1.14 cites was re-run today against `model-bench/` at
`5878014` (unchanged: `git diff --stat 5878014..HEAD -- model-bench/` empty). **All eleven
reproduce exactly**, including both new residuals, both new commands, the corrected command 6, and
the union claim (Appendix H). Every shipped line the new prose cites was read at its cited number.
No model was loaded in LM Studio.

### The question, answered plainly

**Is the plan ready for implementation, or is another revision needed?**

**Another revision is needed.** Not for anything v1.14 did — all three Pass 7 findings are closed,
two of them completely, and the two note deltas landed correctly. It is needed for a **cross-table
contradiction that predates v1.14 and that no pass has swept for**: §4 S1e **Tables C and G both
edit `stats.py:159`, prescribe incompatible forms for it, and neither names the other** (`P8-1`).
Table C replaces that line's percentile primitive with the note's integer-`permille` `percentile`;
Table G requires the same line to take a family-derived level that is `100·α/(2k)` — **fractional
for every `k ≥ 2`** and inexpressible as an integer permille. The seven tables land as one fix
round by §4 S1e's own new ordering paragraph, so an implementer meets both on one line and has to
invent a statistics decision to proceed.

**Verdict: needs changes.** 1 blocker, 2 majors, 0 minors, 0 nits.

The two majors are the same sweep's other results: §3.3 (iv)'s new "nothing else in the verdict path
changes" is falsifiable at two cited lines (`P8-2`), and **Table C breaks the residual-distinguishing
property v1.14 just wrote** — which is exactly the third table question 3 asked me to look for
(`P8-3`).

**None is blocked on unbuilt work.** All three are plan edits available today; `P8-1` additionally
carries a question that is the note's to rule under §7 rule 3, which is that rule's normal mechanism
and not a deferral.

### Disposition of Pass 7's three

| # | Disposition | Rechecked |
|---|---|---|
| **P7-1** (no stored form for `DistributionSummary`; 3 of 4 sites unnamed, one silent) | **Closed** | Stored form decided in Table F: `"distribution"` tag + six keys; `support` stored on both continuous types as `[lo, hi]`/`null`, read with **no `.get`** (the rule `results.py:364-365` already states for `BinaryMetric.unit` — read, correct); one tag-set home that `_metric_from_dict` dispatches on and `_decode` gates on, **unknown tag raises**; residual `{"binary", "continuous"}` → **1 → 0** (verified, `results.py:385`), which the silence-shipping half-application cannot pass; DC-13(f)'s four assertions + §5 test 11d's stored half; all four sites now rows. `load_history`'s bare `except Exception` (`results.py:489`) does surface the raise as `unparseable`, as claimed. **`benchSchemaVersion` no-bump is right on §3.4.3's own criterion** — read verbatim: increments "when the required-field set or the on-disk record shape changes in a way a *reader* must branch on"; `model-bench/results/` does not exist, so no reader ever sees the numeric form, and `REQUIRED_BY_SCHEMA` carries **fingerprint** field specs keyed by arm profile, so a `[2]` entry could not express an aggregate retype at all. Reversal trigger stated and correct |
| **P7-2** (Table G's one-sided residual; width-only test) | **Closed, and generalised further than asked** | `_percentile(means, 97.5)` → **1 → 0** (verified, `stats.py:159`); the new test now asserts **both bounds move outward**, "never as one about the width", with the fixture chosen so both strict inequalities are checkable. Generalised at §7 rule 5(b) and asserted per table at DC-12, and applied unprompted to **Table E** — `min(1.0, point` → **1 → 0** (verified, `stats.py:201`) — whose half-application would indeed have shipped that table's own defect. Correct, and the right instinct. **Residue: `P8-3`** — the generalisation's own sweep missed Table C |
| **P7-3** (the `exploratory` label has no home that admits family members) | **Closed in part** | Both named sites are right and both were read: `report.py:767`'s `m.name not in family` filter and `:744-751`'s headline fallback, which does read `headline is None` and print `_NO_PAIRED_DATA` — false for a refused family. Widening to "refused whole" rather than "not verdicted" is **correct** and the reason given is exactly right: `report.py:664-669`'s `rp is None` branch renders `_NO_PAIRED_DATA` in the member's own block, and "not verdicted" would migrate it out of there. Name-only line, justified on §7 rule 4. **Residue: `P8-2`** — the same paragraph's claim that nothing else in the verdict path changes is false at two more lines |

### Note v1.17's two plan-side deltas — both sound

**Forward, do not derive.** §4 S1's loop now "**forwards `ContinuousMetric.support` and derives
nothing**", Rule 8 deriving the difference support inside itself and exposing no `clamp`, with the
`lo >= hi` raise placed **there** and explicitly not repeated in the loop. That matches note v1.17
§3.4 Rule 8 exactly and is *less* work than v1.13 specified, as it should be. Table E's `:162` row
now carries both callers on one row, and the §4 S1 paragraph states the two surfaces and why neither
is a simplification of the other — which is the thing most likely to be "simplified" away later.

**The premise swap.** The non-blocking classification is re-rested on `_widen`'s scale-1.0 identity
and the census is retired with its own reasoning recorded rather than deleted. I verified that
argument independently at Pass 7 against shipped `stats.py:198-202` and Table E's specified
`clamp=None` semantics; it holds, it is metric-generic, and it is strictly stronger than the census
it replaces. **No §7 rule 3 raise is left open**, as §7 now states — correct as of v1.14, and
`P8-1` opens one again.

### New findings

#### Blocker

**P8-1 (blocker) — Tables C and G both edit `stats.py:159` and prescribe incompatible forms for it,
and neither names the other. After Table C that line's level parameter is an integer `permille`;
Table G requires it to carry `100·α/(2k)`, which is fractional for every `k ≥ 2`. Table G's own
required test becomes unwritable, and Table G's two residuals are driven to zero by Table C's edit
alone — so DC-12 passes on an implementation that never applied Table G.**
*Evidence.* Table C's site row: "`stats.py:159`, `:292` | call sites **move to the new signature**",
that signature being `-ml` §11.2's `percentile(values, *, permille: int)` — read at note `:2528-2529`
("`permille` is the level x10: p95 is 950") with §11.2.1's exactness expression
`-(-permille * X // 1000)` and its zero-divergence sweep taken over `permille ∈ {25, 500, 950, 975}`,
integers throughout. Table G's site row: `paired_bootstrap` "gains `levels: tuple[float, float]` …
replacing the two literals at `:159`", and **six** of its rows pass `levels=(2.5, 97.5)` — percent,
float. Note §3.3 (`:676-678`) fixes an all-continuous family's percentiles at `100·α/(2k)` and
`100 − 100·α/(2k)`, and Rule 8 (`:1337`) computes exactly those. At α = 0.05: `k = 1` → 2.5 % =
**25 ‰** (integer, which is why nothing binds today); `k = 2` → 1.25 % = **12.5 ‰**; `k = 3` →
**8.33 ‰**; `k = 4` → **6.25 ‰**. **Not an integer for any `k ≥ 2`.**
*Why it matters:* three consequences, all concrete. (i) The two tables land in one round — §4 S1e's
own v1.14 paragraph fixes the order as "Tables A–G with DC-11 and DC-13, **then** S2" — so the
implementer meets both on `:159` and must invent a resolution: make `levels` permille ints (six
Table G rows and the note's own formula then disagree), or convert with a `×10` and a rounding at
the one line §11.2.1's exactness argument lives on. Either is a statistics decision, which §7 rule 2
puts with the note. (ii) **Table G's prescribed new test cannot be written** — it fixes a `k = 2`
family, whose pair is 1.25/98.75, and an integer-permille primitive cannot take it. (iii)
**Table G's residuals are passed by Table C.** `_percentile(means, 2.5)` and `_percentile(means,
97.5)` both go to 0 the moment `:159` reads `percentile(means, permille=25)`, with `levels` never
added — the cross-table form of the weakness `P7-2` found within one table, and it defeats the
residual pair v1.14 just strengthened.
*Fix.* The plan's own convention for two tables meeting on one line (Table D/E on `stats.py:263`,
Table E/G on `:162`) is to name the collision, name the owner, and state whether the order is
faithful either way. Here it is **not** faithful either way, so: **(a)** state the collision on both
rows. **(b)** Note that Table C's `stats.py` half is **wider than the obligation it cites** — `-ml`
§11.10(3) (read at note `:3136-3138`) requires only that "`modelbench.results` exposes no private
percentile helper, and its percentile *is* `stats.percentile`"; it says nothing about `stats.py`'s
own private helper. The cheapest sound resolution is therefore to scope Table C's replacement to
`results.py:573`/`:599`/`:600` plus `stats.py:292` (`cluster_bootstrap`'s fixed 25/975, which *are*
integer permille) and to leave `:159` — **the one percentile call in the package whose level is
family-dependent** — to Table G, stating that exemption and its reason on both tables. **(c)** If
instead one primitive is wanted everywhere, raise to `data-scientist` under §7 rule 3: does
`percentile` gain a form expressing a non-integer level, and what happens to §11.2.1's
integer-rank exactness claim and its sweep when it does? That is the note's to rule, not the plan's.
**(d)** Whichever lands, restate Table G's residuals over the surviving spelling of `:159` — the
present pair is only meaningful against the pre-Table-C form.

#### Majors

**P8-2 (major) — §3.3 (iv)'s new paragraph ends "Nothing else in the verdict path changes", and two
more lines in that path change meaning under a whole-family refusal. Both print a false claim, and
both are the same shape as the headline fallback the paragraph does name.**
*Evidence.* `report.py:711-741`, the **Family-wise error control** block, renders on
`if len(family) > 1:` — and a mixed family is `k ≥ 2` by definition, so it always renders. It prints
the prose "Holm–Bonferroni across the {k} pre-registered verdict metrics, **applied**" (`:722-724`),
which is false when pass 1 refused the family and no ladder ran, and then one row per member whose
decision cell comes from `_decision(v, step)` (`:327-337`); with `v is None` — which is what a
refused member carries — `:329-330` returns **"no verdict — no paired data"**. That is false for
precisely the reason the paragraph gives for rejecting the "not verdicted" filter one screen up:
a refused family *has* paired data and a withheld claim. So the plan diagnoses the defect correctly
at `:744-751` and then asserts it occurs nowhere else, at two lines where it does.
*Why it matters:* the report attributes a **pack-authoring** defect (a mixed `verdictMetrics` list,
one line to fix) to a **data** condition, and states that a family-wise correction was applied when
none was. A reader goes looking for missing runs. DC-13(e) and §5 test 11d assert nothing about
either line, so nothing catches it.
*Fix.* Name both sites in §3.3 (iv) beside the two it already names, and take the two decisions:
whether the Family-wise block is **suppressed** for a refused family (with the one sentence saying
why — there is no ladder to report) or rendered with its prose and its decision cells replaced by
the withheld-claim label; and give `_decision` a **third** input state — *verdict withheld* —
distinct from `v is None` meaning *no paired data*, which is the same distinction `:744-751`
needed. Then extend DC-13(e)/test 11d by one assertion per site, as was done for the first two.

**P8-3 (major) — Table C breaks the residual-distinguishing property v1.14 wrote, and DC-12's own
sweep records it as satisfying it. Its residual is scoped to `results.py` while its edit retires
three further `_percentile` sites in `stats.py`; the half-application leaves both bootstraps on the
estimator `-ml` §11.2 explicitly rejects, silently, with the residual reading zero.**
*Evidence.* Table C's rows retire `_percentile` at **six** shipped sites — `results.py:573` (deleted),
`:599`/`:600` (re-sourced from `LatencyBlock`), `stats.py:296` ("replaced by the note's public
`percentile`"), `:159` and `:292` ("call sites move to the new signature"). Its residual is
`grep -rFc _percentile modelbench/results.py` → **3 → 0** — verified, and verified that
`grep -rFn _percentile modelbench/stats.py` is **3** (`:159`, `:292`, `:296`), none of them covered.
DC-12's new v1.14 paragraph states "**C**'s single command counts all three of its `_percentile`
sites at once, which is the same guarantee in one line" and partitions C into the *already
satisfies* group. There are six, not three.
*Why it matters:* the surviving half is exactly rule 5(b)'s named failure — a number that still
prints. `stats.py`'s `_percentile` is `int(round(p/100·(X−1)))`, which Table C's own prose calls
"the estimator `-ml` §11.2 explicitly **rejects** — `round` is half-to-even, so the tie-break
direction alternates with the sample size". An implementer who does the `results.py` half (which the
residual measures, and which is the half §11.10(3) actually obliges) leaves every bootstrap bound in
`paired_bootstrap` and `cluster_bootstrap` on the rejected estimator, and DC-12 passes.
*Fix.* One residual, scoped the way the first one is: `grep -rFc _percentile modelbench/stats.py`
→ **3 → 0** (verified non-zero today; zero after the prescribed edit, since the replacement is the
note's **public** `percentile`, which carries no leading underscore — the same argument Table C
already makes for `results.py`). Correct DC-12's Table C sentence from "all three" to the six the
table retires, and move C out of the *already satisfies* group into the *gained its second at this
revision* one. **Sequence this with `P8-1`:** if `:159` is exempted from Table C, the residual is
`→ 2 → 0` over `:292`/`:296` and the exemption is what the row must say.

### The two judgements the re-check was asked for

**(1) Commands 7 and 8 enumerate the class, and the rule they instance is stated right.**
Re-run today: command 7 (`grep -rn '\.mean' modelbench --include='*.py'`) → **3**
(`results.py:359`, `:584`, `report.py:583`); command 8 (`grep -rn '"continuous"' modelbench
--include='*.py'`) → **2** (`results.py:359`, `:385`). **Union = exactly the four broken sites**, and
each alone is insufficient in the direction the table claims — 7 misses `_decode`'s tag gate, 8
misses `_index_row` and the Arms table. **Both are 0 under `tests/`**, verified, so the `modelbench`
scoping is a measurement and the table says so. The corrected command 6
(`isinstance(.*BinaryMetric`) → **6**, per file exactly as stated.
*The rule.* §7 rule 5(a)'s new companion — a table adding a new type to an existing union enumerates
by **the attribute the new type lacks** and by **the storage tag**, never by the type name, plus
*a command pins the type, never the variable that holds it* — is the **right statement of the
class**, and it is a better one than Pass 7 proposed: I named only the attribute form, and the
storage-tag form is what reaches `_decode`, the one site that fails silently, while the
variable-pin sub-rule is the actual root cause of command 6's original miss and generalises past
this defect entirely. **It carries its scope obligation the way the rest of rule 5 does** — it
states the obligation in its own text ("states each command's scope, since these forms often return
nothing under `tests/` and a silent scope reads as an oversight") and Table F instances it on both
new commands, checkably. One framing note, not a defect: the rule is written for *a new type joining
a union*, while the defect class is slightly wider — any **retype** of a field breaks readers that
assumed the old type's attributes, union or not. The narrower trigger is the checkable one and Table
F is the only table it fires on, so the choice is defensible; if a future table retypes a field
without widening a union, the rule as written will not fire on it.

**(2) `P7-2`'s generalisation across all seven tables — I checked each myself. It is incomplete: one
table breaks it, and it is Table C.** Re-derived from each table's own rows, not from DC-12's
summary:

| Table | Literals its edit retires | Residuals | Distinguished? |
|---|---|---|---|
| **A** | `lmsCliCommit`, `sizeBytes` | 2 (3 → 0, 1 → 0) | **Yes**, one each. The `residentModelsAtEnd` element shape carries no further discriminating token, and the row already says so — "fails loudly if missed? **no** — the fix is DC-1's assertion" |
| **B** | `FORBIDDEN_BY_ARM_KIND`, `frozenset(FORBIDDEN`, `REQUIRED_BY_SCHEMA[1]["model"]`, the key-set assertion | 4 | **Yes**, one each. `armKind` survives by design and the table states that rather than faking a residual |
| **C** | `_percentile` at **six** shipped sites — `results.py:573`/`:599`/`:600` **and** `stats.py:159`/`:292`/`:296` | 1, scoped to `results.py` (3 → 0) | **No → `P8-3`.** The `stats.py` half is unmeasured and its survival is silent |
| **D** | `bootstrap_seed`, the `cluster-bootstrap` token | 2 (29 → 0, 27 → 0) | **Yes.** `conservative_envelope`'s retired `diffs`/`B`/`seed` parameters need none: a removed *required* parameter is enforced by the call, exactly as rule 5(b)'s closing paragraph says an added one is |
| **E** | `max(-1.0, point`, `min(1.0, point` | 2 at v1.14 | **Yes**, and the second is v1.14's unprompted application. Verified `min(1.0, point` → **1** |
| **F** | the two `float \| None` annotations, `{"binary", "continuous"}` | 3 | **Yes for the silent ones.** The `(BinaryMetric, ContinuousMetric)` tuple at `:373` is retired without a residual, but its survival raises `TypeError` at `json.dump` on the first store and DC-13(f)'s round-trip assertion catches it — the rule's rationale is met, and a residual there would be optional rather than owed |
| **G** | `_percentile(means, 2.5)`, `_percentile(means, 97.5)` | 2 at v1.14 | **Yes as written** — but both are zeroed by Table C's edit alone, which is `P8-1`(iii) |

So the generalisation is right, its application to Table E was the correct unprompted move, and the
sweep that produced DC-12's seven-table partition **miscounted exactly one table** — the one whose
residual is scoped to a single file while its edit spans two.

### What's solid

- **`P7-1` is closed properly, not narrowly.** The stored form is decided rather than gestured at,
  the tag set gets one home so the encoder and the decoder cannot disagree about how many metric
  types exist, and the residual over `{"binary", "continuous"}` is the one an implementer cannot
  pass by adding branches everywhere except `_decode`. The `benchSchemaVersion` ruling is the
  strongest paragraph in the revision: it declines to bump, gives the *criterion* it declines on,
  shows the bump would be inexpressible in `REQUIRED_BY_SCHEMA`, and states the exact condition
  that reverses it.
- **The residual's own limit is stated.** Table F says out loud that its third residual is also zero
  for a third *transcribed* tag in place of the mapping, and names DC-13(f) as what covers the
  difference. A residual that discloses its reach is worth more than one that does not.
- **`P7-2` was generalised rather than patched, and applied to a table nobody asked about.** Finding
  Table E's identical shape unprompted is the behaviour that would have prevented three of the seven
  passes.
- **`P7-3`'s two decisions are both taken correctly**, and the "refused whole", not "not verdicted"
  distinction is a genuinely subtle call made the right way for the right stated reason.
- **The revision-size trend held.** +288/−75 with ~110 lines of it new decision text, and it closed
  all three findings and both note deltas.

### Open questions

1. **Does `percentile` gain a non-integer level form, or is `stats.py:159` exempt from Table C?**
   This is `P8-1`(c) and it is the one part of that finding that belongs to `data-scientist` under
   §7 rule 3 rather than to the architect: §11.2.1's exactness argument and its zero-divergence
   sweep are both stated over integer `permille`, and a `k ≥ 2` continuous family needs a level that
   is not one. The plan can state the exemption without the note's help; it cannot change the
   estimator's signature without it.
2. **Is `Family-wise error control` suppressed or relabelled for a refused family?** `P8-2` names
   the decision but it is a presentation call with a reader-facing consequence, and §3.3 (iv) is
   where this plan has been taking those. Flagged separately only because it is a judgement rather
   than a transcription, like `benchSchemaVersion` was.

## Appendix H — Pass 8: what was re-run and read

**H.1 — every count v1.14 cites, re-run under `model-bench/` at `5878014`** (tree unchanged;
`git diff --stat 5878014..HEAD -- model-bench/` empty).

| Claim | Command | Result |
|---|---|---|
| Table F cmd 6, corrected | `grep -rn 'isinstance(.*BinaryMetric' modelbench tests --include='*.py'` | **6** — `report.py:211`, `:553`, `:564`; `results.py:355`, `:373`, `:584`. Per-file exactly as stated; 0 under `tests/` |
| Table F cmd 7 | `grep -rn '\.mean' modelbench --include='*.py'` | **3** — `results.py:359`, `:584`; `report.py:583`. **0** under `tests/` |
| Table F cmd 8 | `grep -rn '"continuous"' modelbench --include='*.py'` | **2** — `results.py:359`, `:385`. **0** under `tests/`. Union with 7 = the four sites exactly |
| Table F residual 3 | `grep -rFn '{"binary", "continuous"}' modelbench --include='*.py'` | **1** — `results.py:385` |
| Table E residual 2 | `grep -rFn 'min(1.0, point' modelbench --include='*.py'` | **1** — `stats.py:201` |
| Table G residual 2 | `grep -rFn '_percentile(means, 97.5)' modelbench --include='*.py'` | **1** — `stats.py:159` |
| Table C residual, as stated | `grep -rFc _percentile modelbench/results.py` | **3** |
| Table C, the unmeasured half | `grep -rFn _percentile modelbench/stats.py` | **3** — `:159`, `:292`, `:296` → **`P8-3`** |
| `_encode`'s retired tuple | `grep -rFn '(BinaryMetric, ContinuousMetric)' modelbench tests --include='*.py'` | **1** — `results.py:373` |
| stored records exist? | `ls model-bench/results` | **absent** — the no-bump argument's premise holds |

**H.2 — shipped lines read at their cited numbers.** `results.py:326` (the `KeyError` →
`unparseable` comment, quoted correctly), `:354-359`, `:363-365` (the no-`.get` rule for
`BinaryMetric.unit`), `:373`, `:385`, `:471-500` (`load_history`'s bare `except Exception`, `:489`),
`:584`; `report.py:327-337` (`_decision`, `v is None` → "no verdict — no paired data"), `:553-584`,
`:664-673` (`_NO_PAIRED_DATA` in the member's own block, rendered at `:669`), `:711-741` (Family-wise error control,
prose at `:722-724`), `:744-751` (headline fallback), `:763-777`; `stats.py:148-202`, `:296-300`.

**H.3 — documents read.** Plan v1.14 §3.3 (iv) (the whole new paragraph, `:510-527`), §4 S1's
support/clamp block, DC-12, DC-13(f), §4 S1e preamble and ordering paragraph, Tables A–G in full
with every residual statement, §5 test 11d, §7 (rule 5(a)'s new companion, rule 5(b)'s new clause,
the *Note v1.17* block, the Pass 7 disposition table), Appendix A. Note v1.17 §3.3 (`:671-694`),
§3.4 Rule 8 (the level formula at `:1337`, the `support` ruling at `:1352-1400`), §11.2
(`:2528-2551`), §11.10(3) (`:3136`). The v1.14 delta in full,
25 hunks.

## Pass 9 (narrow) — 2026-09-07

**Scope, as commissioned:** the **v1.15 delta only** — `git diff 4cd22b9 083d174 --
docs/plans/small-model-benchmarking.md` (+329/−63, read in full) against note **v1.18** (`bbbf18e`)
and the shipped tree at `5878014`. This is the narrow re-check `## Pass 8` asked for; it does not
re-gate sections v1.15 did not touch and reopens nothing closed in Passes 1–7. Five questions were
put to it (below) plus the three extras the revision disclosed. No new finding IDs are issued.

**CPG:** considered, not relevant — no `model-bench` CPG exists (this instance loads
`cpg_falkorchat` and `cpg_deprecated_salesperson` only), and v1.15 makes no structural claim about
either component. Everything below is grounded in `grep`/`sed`/`awk` against the tree.

**Verified, not assumed.** Every count this pass relies on was re-run today under `model-bench/`
(tree unchanged: `git log -1 -- .` → `5878014`). **All reproduce** — Appendix I. No model was
loaded in LM Studio. Nothing in the plan, the note or `model-bench/` was edited.

### The question, answered plainly

**Is the plan ready for implementation, or is another revision needed?**

**It is ready. No further revision is needed.** All three Pass 8 findings are closed — the blocker
by the route the note ruled rather than by the cheaper one I offered, which was the right call and
is not defended here. I found **no blocker, no major, no minor and no nit** in the delta. Both
questions Pass 8 left open are answered, in the documents that own them.

**Verdict: approve.**

**Which failure mode this is.** I am aware that eight rounds create pressure to approve and that an
invented ninth costs as much as a missed defect. I went looking for a defect rather than for
confirmation: I re-derived the C↔G residual algebra over all four half-application cases, re-read
`report.py:606-780` in full to test the unreachability claim against the code rather than the prose,
and mapped every line in `stats.py`/`test_stats.py` that more than one table edits, hunting for a
second unnamed collision. Three candidates surfaced and all three dissolved under checking; they are
recorded below as observations rather than suppressed. The approve is the finding.

### Disposition of Pass 8's three

| # | Disposition | Rechecked |
|---|---|---|
| **P8-1** (Tables C and G collide on `stats.py:159`; G's residuals zeroed by C's edit alone) | **Closed** | Note v1.18 §11.2.2 ruled the level an exact `Fraction` and **refused** the exemption Pass 8 offered, on a measured cost (the exempted line keeps the *estimator* §11.2 rejects, not a different unit); the plan took the four plan-side edits §11.9 item 6 lists. Collision named on **both** rows (`:3165`, `:3539`), order fixed **C then G** with the reason on C's row, `Fraction(1, 40) = 0.025 ≠ 2.5` confirmed by arithmetic, G's residuals re-derived over the post-C spelling, DC-12 re-running them at the end of the round. **DC-12 can no longer pass on an implementation that never applied Table G** — traced over all four cases in Q1 below |
| **P8-2** (§3.3 (iv)'s "nothing else in the verdict path changes" false at two more lines) | **Closed** | The claim is replaced by an **enumeration**: `awk 'NR>=606 && NR<=780 && (/lines \+=/ \|\| /lines\.append/)' modelbench/report.py` → **11**, re-run, at exactly the eleven line numbers cited (`:666`, `:690`, `:719`, `:738`, `:741`, `:751`, `:756`, `:770`, `:776`, `:777`, `:779`). Each of the six non-sites re-read at its number and each classification is right, including `:756` (claims co-equality, never a verdict) and `:779` (`_OVERLAP_FOOTNOTE`, generic prose about a line this path did not print). `:711` is `if len(family) > 1:` and `:722-724` the `applied` prose — both exact. Suppression is the right of the two options and the rejection reason is correct: every column of that table is a ladder artefact |
| **P8-3** (Table C's residual scoped to `results.py` while its edit spans two files) | **Closed** | Three residuals now, and they partition: `_percentile` in `results.py` **3**, in `stats.py` **3** (`:159`, `:292`, `:296`), and `-ml` §11.10(3)'s package-wide `def [A-Za-z_]*(percentile\|quantile)` → **2**, target **1** — all three re-run. `3 + 3 + 1 = 7` matches the enumerating command's own count, re-run; the seventh line (`tests/test_results.py:507`) has a row and a stated reason for having no residual. DC-12's partition corrected (C out of *already satisfies*), its wording moved to "hits its stated target", §7 rule 5(b) given the non-zero-target clause. **A 2, B 4, C 3, D 2, E 2, F 3, G 2 = 18**, and 18 − G's 2 = the **16** stated over the shipped tree: arithmetic checked, and A's, D's, E's and F's counts re-run and reproduced |

### The five judgements the re-check was asked for

**(1) Does the C/G collision now fail loudly, and can DC-12 pass on an implementation that never
applied Table G?** It fails loudly, and **no**. Traced over the four cases, with the DC-12
conjunction (not either table read alone) carrying the property, which is exactly what the new
"property of the round" clause claims: *both applied* → C 0/0/1, G 0/0 (pass); *C applied, G skipped
or its `:159` wiring omitted* → G's residual **1** (caught); *C's `:159` row skipped, G applied* →
C's `stats.py` residual **non-zero**, `_percentile` surviving there (caught); *neither applied* →
same (caught). A G-first attempt is additionally **not constructible**: six of Table G's rows spell
`LEVEL_CI95_LO`/`LEVEL_CI95_HI`, and those four constants land with **Table C**.

**(2) The residual whose baseline is an intermediate state.** Sound, and unambiguous. §7 rule 5's
"counts at a named commit" is clause **(b)** and it governs the *enumerating* commands — Table G's
three still carry theirs and all three re-run (`97.5` → 2; `paired_bootstrap(` → 5;
`paired_cluster_bootstrap(` → 5). A residual is clause **(c)**, and nothing in the rule requires its
baseline be the shipped tree. The arrow cannot be mis-read: the plan states it in the imperative
("*after Table C, before this table* → *after this table*"), states that both commands return **0**
today (re-run: 0 and 0), and DC-12 carries the exception on its own row with the end-of-round timing
and the fixed order that makes the intermediate state reachable. The one property that is genuinely
weaker — these two are the only residuals never *observed* non-zero anywhere — is disclosed by the
plan and is backstopped behaviourally: Table G's own `k = 2` test asserts both bounds move outward,
which fails if `:159` accepts `levels` and ignores it. See observation (a).

**(3) The non-zero-target clause.** Sound; it does not open a hole. It adds a *disclosure*
obligation (state the target, name the surviving line) on top of rule 5(b)'s two standing
constraints, neither of which it relaxes — a residual must still not fail on a faithful edit, and a
table's set must still defeat its own half-applications. It is if anything tighter than a zero
target: `2 → 1` is an equality, so it fires on over-application as well as under. Table C names its
survivor (`stats.py`'s public `percentile`), and the wrong-survivor case — the def kept in
`results.py` — is caught by that table's first residual. The clause's own justification is correct
on its face: this is the one check in §4 S1e that no half-application passes, precisely because it
is stated over the estimator rather than over a module's private helper.

**(4) The C-then-G ordering: enforceable, and sufficient.** Enforceable by two things an implementer
hits, neither of them prose — the **missing constants** above (G-first is a `NameError` in six of its
own rows), and the **`k = 2` outward-movement test**, which is red for the mis-ordered wiring the
plan warns about (`Fraction(1, 80) → 0.0125` handed to a percent-typed `_percentile` selects near
the bottom of the resample, so the *upper* bound moves the wrong way and the second of the two
assertions fails). **Sufficient — no third table is in that order.** I mapped every multi-table line:
`stats.py:159` {C, G} ordered; `:162` {E, G} both orders faithful; `:263` {D, G}, named on both
sides; `:264` {D, E}; `:292`, `:296` C alone; `test_stats.py:1270` {D, E, G}, `:1323`/`:1330`
{E, G}. Every one of those besides `:159` is a *required-parameter addition* enforced by the type
system, with no residual stated over it, so no order can hide anything. See observation (c).

**(5) `_decision`'s unreachability — verified against the code, not the plan.** The claim holds.
`grep -rFn '_decision(' modelbench --include='*.py'` → **2**: the definition at `report.py:327` and
one call at `:739`, which I read in place and confirmed is lexically inside the `if len(family) > 1:`
block opened at `:711`. So suppressing that block for the two no-ladder conditions removes the only
path on which `v is None` carries *withheld*; the surviving `v is None` is `:660-669`'s empty paired
intersection, for which the string is true — read at its lines. **The plan commits to no second
caller**: S1 adds the continuous branch, whose family renders the replacement line rather than the
table, and the refusal path renders a line rather than a table. The reversal trigger is stated at
the one place that would have to change. The `"no verdict — no paired data"` absence assertion is
also safe from a false positive against `_NO_PAIRED_DATA`, whose text is
`**No verdict: no paired data.**` — different casing and punctuation, read at `:316-320`.

### The three disclosed extras

1. **§3.3 (iv)(3) covering two conditions.** Correct to widen, and the reasoning is right: it is the
   *same shipped line* under the other condition that reaches it, and a fix scoped to the refusal
   would have left `report.py:722-724` printing `applied` for exactly the family Table G exists to
   serve. The supporting claim is grounded — §4 S1's loop does state "`holm_steps` is not called on
   a continuous family", read at its own bullet. Both replacement lines are specified semantically
   (naming the refusal; *correction taken in the interval*) rather than verbatim, which is
   consistent with this plan's standing refusal to restate strings, and DC-13(e) asserts each on a
   checkable condition plus **two absences**, which is the assertion shape a line-presence test
   cannot substitute for.
2. **Table C's row assigning `-ml` §11.10 items 1, 2a, 2b and 10 to S1.** Correct, and the partition
   is right: items 1/2a/2b/10's first half are pure-estimator and pure-level assertions, item 3 is
   the table's own third residual, and the rest of §11.10 is `latency_summary` and rendering, which
   §5's stage table puts in S2. Item 2b's second half — the `k = 2` family deriving exactly
   `(Fraction(1, 80), Fraction(79, 80))` — needs Rule 8's `continuous_verdict()`, and **that is
   available at S1**: §4 S1's loop hands `diffs` to it and forwards `support`. (Table G's phrase
   "which this plan does not write" means *does not specify* — §7 rule 2 — and reads oddly beside
   §4 S1; harmless, and not new in this delta.)
3. **DC-12's standing per-table rule 5(b) sweep, with Table A's result written out.** The right
   move — the sweep's value is that it is standing rather than a reviewer's catch — and the one
   result that needed deriving is derived correctly: Table A's `residentModelsAtEnd` element row
   does retire two keys (`{modelKey, sizeBytes} → {id, state}`, read at the row), `modelKey`
   survives everywhere else (`grep -rFn modelKey modelbench tests --include='*.py'` → **90**,
   re-run), so a residual over it would fail on a faithful edit, and rule 5(b)'s named alternative
   — DC-1's element-shape assertion — is what the row already points at. Writing the result down so
   the next revision does not re-run it is the correct disposition.

### Observations — not findings, no action required

- **(a)** Table G's two residuals are the only ones in §4 S1e never observed non-zero, so a
  mis-typed command string would read as a passing check at the end of the round. It is not a
  defect: the intermediate count is **1 by construction** (Table C prescribes the exact post-edit
  spelling, and prescribing it is the stated reason), and the failure it would let through is red on
  Table G's own `k = 2` test. If a future revision ever wants the belt as well, asserting the pair
  at **1** between the two tables costs two greps and the fixed order already makes that state
  exist.
- **(b)** The `3 + 3 + 1 = 7` derivation reads under the heading *Why three commands and not one*,
  and the third addend is the test-comment row rather than the third command's target — which
  happens also to be `1`. The next bullet says plainly that the seventh line gets no residual, so
  nothing follows from the collision of numerals.
- **(c)** v1.15's new clause asks two tables meeting on a line to name each other **on both rows**;
  on `stats.py:162` Table G names Table E and Table E does not name Table G, and Tables E and G
  meet unremarked on `test_stats.py:1270`/`:1323`/`:1330`. The hazard the clause exists to prevent
  cannot arise there — both orders are faithful, both edits are required-with-no-default parameters
  the type system enforces, and no residual is stated over any of those lines — so this is
  incompleteness in an illustration, not an unclosed collision.
- **(d)** The `606-780` window excludes two emission sites of the same function, `:600` (the
  early-return branch, which a refusal never reaches) and `:604` (the `## Verdicts` header and
  comparison kind, which a refusal does not falsify). The boundary is principled — `:606` is
  `family = list(pack.metrics.verdictMetrics)`, i.e. the family's construction — but the plan does
  not say so.

### What's solid

- **The blocker was closed by the more expensive route and the reasoning is the reason.** The
  exemption Pass 8 offered was legal against §11.10(3)'s v1.17 letter, and the note refused it on a
  measured cost — the exempted line keeps the rejected *estimator*, whose lower-bound order
  statistic differs from type 1 at `k ∈ {1, 2, 5}`, at the only call site whose level is not a
  literal, on the one path where the interval *is* the test. Overruling the gate on evidence is what
  §7 rule 3 is for, and both documents record it as such.
- **`Fraction` closes the level space rather than widening the unit.** The two rejected
  alternatives are both recorded with the reason each fails (a fixed decimal unit cannot exist
  because 3 divides `40k`; outward rounding works and costs a second meaning for *attained level* in
  one report), and the closure argument — every level in the note is a ratio of declared quantities
  — is what makes the choice final rather than current.
- **P8-2 was closed by enumeration, not by adding the two lines I named.** That is the difference
  between a fix and a method, and it is the second time in three revisions this plan has generalised
  a finding instead of patching it. The eleven sites are checkable and I checked them.
- **The unreachability argument is the right shape and it is stated with its trigger.** Adding a
  third input state to `_decision` would have been the easy move and would have shipped a string
  nothing renders; removing the path instead, and naming the one revision that would bring the state
  back, is stronger.
- **The revision disclosed three extras rather than folding them in**, and all three survive
  checking. That discipline is now three revisions old and it is what makes a narrow re-check
  possible at all.

### Open questions

**None.** Both of Pass 8's are answered — `percentile` takes an exact rational (note v1.18 §11.2.2,
the §7 rule 3 raise opened and closed inside one revision), and the `Family-wise error control`
block is suppressed with a one-line replacement rather than relabelled (§3.3 (iv)(3)). Nothing is
carried, nothing is deferred, and nothing is blocked on unbuilt work. The next dispatch is the S1
implementation unit.

## Appendix I — Pass 9: what was re-run and read

**I.1 — every count this pass relies on, re-run under `model-bench/` at `5878014`.**

| Claim | Command | Result |
|---|---|---|
| Table C enumeration | `grep -rFn _percentile modelbench tests --include='*.py'` | **7** — `stats.py:159`, `:292`, `:296`; `results.py:573`, `:599`, `:600`; `tests/test_results.py:507` |
| Table C residual 1 | `grep -rFc _percentile modelbench/results.py` | **3** |
| Table C residual 2 (new) | `grep -rFc _percentile modelbench/stats.py` | **3** |
| Table C residual 3 (new) | `grep -rEn 'def [A-Za-z_]*(percentile\|quantile)' modelbench tests --include='*.py'` | **2** — `stats.py:296`, `results.py:573`; target **1** |
| Table G command 1 | `grep -rFn 97.5 modelbench tests --include='*.py'` | **2**, both `modelbench/stats.py` (`:159`, `:292`) |
| Table G commands 2, 3 | `grep -rFn 'paired_bootstrap(' …` / `'paired_cluster_bootstrap(' …` | **5** and **5**; 2 `def` lines, **8** call sites (`stats.py:187`, `:263`; `test_stats.py:890`, `:891`, `:1270`, `:1321`, `:1323`, `:1330`) — the eight the plan claims |
| Table G residuals, retired pair | `grep -rFc '_percentile(means, 2.5)' …` / `'…97.5)'` | **1** each, `stats.py:159` |
| Table G residuals, re-derived pair | `grep -rFn 'percentile(means, level=LEVEL_CI95_LO)' …` / `…_HI)` | **0** and **0** — the stated *before* is the post-Table-C state |
| §3.3 (iv) enumeration | `awk 'NR>=606 && NR<=780 && (/lines \+=/ \|\| /lines\.append/)' modelbench/report.py` | **11**, at the eleven cited numbers exactly |
| `_decision` call sites | `grep -rFn '_decision(' modelbench --include='*.py'` | **2** — `report.py:327` def, `:739` call |
| DC-12 sweep, Table A | `grep -rFn modelKey modelbench tests --include='*.py'` | **90** |
| DC-12 sweep, A/D/E/F residuals | `lmsCliCommit`; `sizeBytes`; `bootstrap_seed`; `cluster-bootstrap`; `max(-1.0, point`; `min(1.0, point`; `{"binary", "continuous"}` | **3**, **1**, **29**, **27**, **1**, **1**, **1** — all as stated |

**I.2 — shipped lines read at their cited numbers.** `report.py:316-320` (`_NO_PAIRED_DATA`),
`:327-337` (`_decision`), `:588-604` (the two emission sites below the window), `:606-780` in full
(the family loop, the `len(family) > 1` block at `:711`, the `applied` prose at `:722-724`, the
decision-cell append at `:738-739`, the headline block, the Exploratory section, the footnote);
`stats.py:148-202`, `:258-270` (`:263` the resample call, `:264` the MOVER-D arm), `:292-300`.

**I.3 — documents read.** The v1.15 delta in full; plan §3.3 (iv) whole, §4 S1's continuous-branch
bullets, §4 S1e's preamble and ordering paragraphs, Tables A, B, C, D (the kept-functions row), E, G
with every residual statement, DC-12, DC-13(e), §5 test 11d, §7 rule 5 whole, the *Note v1.18* block
and the Pass 8 disposition table. Note v1.18 §11.2 (`:2503-2574`), §11.2.1, §11.2.2 whole, §11.9
item 6, §11.10 whole. Review `## Pass 8 (narrow)` and Appendix H.

## Pass 10 — 2026-09-08

**Scope, as commissioned:** the **v1.19 delta only** — `git diff 9bc3b35 b873cc9 --
docs/plans/small-model-benchmarking.md` (+274/−31, read in full) against note **v1.19** §3.4
**Rule 4a** and the shipped tree pinned at **`7f865e2`**. The plan has moved four times since this
gate last read it (v1.15 at `## Pass 9`); v1.16–v1.18 were driven by the impl gate and are **not**
re-gated here. Every measurement below is taken from `git show 7f865e2:model-bench/…`, never the
working tree — a parallel unit is editing `stats.py` — and no line-pin staleness is raised as a
finding. Findings carry the prefix **`P10-`**. Nothing in `model-bench/`, the plan or the note was
edited; nothing was staged.

**CPG:** considered, not relevant — no `model-bench` CPG exists (this instance loads
`cpg_falkorchat` and `cpg_deprecated_salesperson` only). Everything is grounded in `grep` and in one
read-only simulation of Rule 4a against the `7f865e2` blob (Appendix J.2).

### The question, answered plainly

**Another revision is needed.** Table H is the right shape — the ruling is folded in rather than
re-derived, the two names are the plan's to choose, the disowning-mention trap was caught at design
time, and Table E's "does not move" block is the standing sweep working. But the table's **site list
omits the one shipped test its edit falsifies**, and I measured the failure rather than inferring it:
applying Rule 4a as specified turns `tests/test_stats.py:1345-1364` red on **38** tables at
DEFF 1.2 and **78** at DEFF 2.0, and that test is the executable statement of Rule 4's conservatism
property. Two further rows instruct something the note contradicts.

**Verdict: needs changes.** 1 blocker, 2 majors, 1 minor, 1 nit.

**None is blocked on unbuilt work and none is deferred by choice** — all five are plan edits
available today, against a ruling that already exists. The one item this document cannot close
itself is the §7 rule 3 raise it opens, which is **blocked on `data-scientist`** (the note's
citation is the note's to fix); the plan is right that nothing in it depends on the resolution.

### Findings

#### Blocker

**P10-1 (blocker) — Table H's nine-row site list omits the one existing test its edit falsifies.
`tests/test_stats.py:1345-1364` asserts the printed envelope is never tighter than either arm; once
the arms come back unclamped and the composer clamps, that assertion is false on 38 tables at
DEFF 1.2 and 78 at DEFF 2.0 of its own sweep. The tests row instructs only *additions*.**
*Evidence, measured (Appendix J.2).* I re-implemented Rule 4a points 2 and 4 against the `7f865e2`
blob — `_widen(..., clamp=None)` on both arms, `min`/`max` composition, then `max(-1.0, ·)` /
`min(1.0, ·)` on the result — and re-ran the shipped test's own sweep set (455 tables at n=12 plus
its n=40 stride, 511 in all) at its own three design effects. Under Rule 4a: **0 / 38 / 78**
failures at DEFF 1.0 / 1.2 / 2.0, first failing tables `(0, 0, 9, 3)`, `(0, 0, 10, 2)`,
`(0, 0, 11, 1)`. Control, today's arm-clamped code: **0 / 0 / 0**. The line is `test_stats.py:1361`,
which **command 2 returns** — it is one of the fifteen — so the enumeration reaches it and only the
site list does not. It is not a miss the impl gate could have caught for the plan either: that
review's own next-unit edit list (`## Pass 9` §5, read) enumerates the same five production edits
— `clamp=None`, the composer's clamp and `bound_by`, one subscript, one unpack, four deleted lines —
and likewise names no test.
*Why it matters.* An implementer applying Table H faithfully gets a red suite with no instruction
saying which test changed or why, and the cheap repair — relax the assertion — deletes the
executable statement of `-ml` §3.4 Rule 4 acceptance 4, the property that test's own docstring says
it exists to defend ("what stops a future implementer taking *the wider of* to mean whichever
interval is wider"). No residual can see this: all six of Table H's are over `modelbench` only.
*Fix.* One site row for `tests/test_stats.py:1345-1364`, stating that the comparison moves to the
**clamped** arms — `lo <= max(SUPPORT_DIFF_PROPORTIONS[0], mover[0])` and
`hi >= min(SUPPORT_DIFF_PROPORTIONS[1], mover[1])`, both arms — which is Rule 4a's own restatement of
the property ("Rule 4's own conservatism property survives verbatim: the envelope is never tighter
than the **clamped** MOVER-D arm") and not a weakening. Sequence it with `P10-4`: the same row list
is where the two stale docstrings belong.

#### Majors

**P10-2 (major) — the `report.py:338` row prescribes a rendering the note's assertion 10 forbids.
The row says `"support bound"` renders **bare**; the note's verbatim string is `support bound (-1)`.
The row also omits that the renderer now needs the support *value*, which the table introduces only
in `stats.py`.**
*Evidence.* Plan `:3982`: "the `p=` clause attaches to the exact-bootstrap arm **only**; both
`"MOVER-D"` and `"support bound"` render bare". Note Rule 4a, the rendered bullet and assertion 10,
both read: `- decided by: conservative envelope (lower bound: support bound (-1); upper bound:
MOVER-D)`. The note's own sentence is that a support token never carries a **level** — not that it
carries nothing. `(-1)` is the support's lower component, and `(1)` would be its upper, so the
renderer must reach `SUPPORT_DIFF_PROPORTIONS` (or receive it), which no row says: the constant is
introduced at `stats.py` and `report.py` has no line assigning it a source. Grepped: `support bound`
appears three times in the plan and the parenthetical appears **zero** times. The plan's summary of
the ruling (`:3945`) has the same gap — "must not attach a `p=` clause" is true and incomplete.
*Why it matters.* An implementer follows the row, renders `support bound` bare, and fails the
note's assertion 10 — so the defect is caught, but by a test written from the *other* document,
which is the failure mode §7 rule 2 exists to prevent. Worse, the missing half is the one with a
design consequence: without naming `SUPPORT_DIFF_PROPORTIONS` as the renderer's source, `-1` becomes
a second home for the support in `report.py`, which is the one-arithmetic-two-homes shape this table
is closing `P8-5` for.
*Fix.* Rewrite the row's second clause to the note's string — a `support bound` token renders with
its boundary value and never with a level — and name `SUPPORT_DIFF_PROPORTIONS` as where that value
comes from. Also correct the row's last clause: assertion 10 asserts the **correct** bullet appears
verbatim; `support bound, p=0.025` is what it *kills*, not what it is.

**P10-3 (major) — the `stats.py:413` row claims the post-edit spelling is "prescribed rather than
left open" and never states it. The only statement of it is inside residual commands 2 and 3, which
pin the local names `lo`/`hi` — while the note's own notation for those two quantities is
`u_lo`/`u_hi`. A faithful edit therefore reads 0 against a stated target of 1.**
*Evidence.* Plan `:3977` defers to the residual block; the block (`:3992-3993`) states
`max(SUPPORT_DIFF_PROPORTIONS[0], lo)` and `min(SUPPORT_DIFF_PROPORTIONS[1], hi)`, target **1** each.
`_compose`'s shipped body (`stats.py:413`) has no locals at all — it is a bare
`return min(exact[0], mover[0]), max(exact[1], mover[1])` — so the edit must *introduce* them, and
Rule 4a point 4 names the composed unclamped bounds **`u_lo`** and **`u_hi`** in the very expressions
the implementer is told to implement (`u_lo < L` → `"support bound"`). An implementer who follows the
note names them `u_lo`/`u_hi`; residuals 2 and 3 then read **0**, not 1, and DC-12 fails on a correct
edit. The cited precedent does the opposite of this: Table C's `:159` row writes its post-edit
spelling out **in the row**, which is exactly why Table G's residuals over it are safe.
*Why it matters.* §7 rule 5(b) forbids a residual that fails on a faithful edit, on the stated
ground that it "trains the implementer to override the done-condition"; the direction is inverted
here (0 where 1 is wanted) and the effect is identical. This is the table the revision holds up as
the first written *against* rule 5(b)'s third form rather than corrected into it, so the gap is worth
closing where it is claimed.
*Fix.* Write the two-line post-edit body into the `:413` row verbatim, as Table C's `:159` row does
— including the local names — or restate residuals 2 and 3 over text that does not depend on a name
the plan never fixes (e.g. `SUPPORT_DIFF_PROPORTIONS[0],` and `SUPPORT_DIFF_PROPORTIONS[1],`, whose
counts are still 1 and 1 and which no other construct can spell).

#### Minor

**P10-4 (minor) — two docstring paragraphs state the pre-Rule-4a attribution flow and get no row,
in the two functions the rule restructures. One of them is the sentence the plan claims the edit
makes *true*, and the edit falsifies its other clause.**
*Evidence.* `stats.py:359-361` (`envelope_arms`): "Both `conservative_envelope` and `verdict()` take
their arms from here and compose them through `_compose`, and **`verdict()` reads the attribution
off the same pair**, so no caller recomputes another's arithmetic". After Rule 4a `verdict()` reads
no attribution — it receives one — and the attribution is no longer a function of the arms pair
alone, the third token coming from the composed value against the support. The plan (`:3981`) cites
only the second clause and concludes the edit "finally makes `envelope_arms`'s docstring sentence
true". `stats.py:397-401` (`_compose`, first paragraph): "`verdict()` because it also **needs the
arms themselves for the attribution** and so cannot go through `conservative_envelope`" — the
conclusion survives, the reason does not. The table's docstring row (`:3978`) is scoped to
`_compose`'s **last** paragraph only.
*Why it matters.* A docstring asserting what its body no longer does is impl-gate `P8-5`'s finding,
which this table closes as collateral and gave `_compose` a row for on exactly that argument. No
residual reaches docstring prose, and command 2 returns `stats.py:351` — the head of the function
whose docstring goes stale — with no row against it.
*Fix.* Extend the docstring row to both paragraphs, or add one row for `envelope_arms`'s docstring;
one clause each.

#### Nit

**P10-5 (nit) — "fifteen of `test_stats.py`'s twenty `envelope_arms` lines are the arm-level
assertions Rule 4a's assertion 1 rewrites" (`:3969-3972`) is wrong twice.** Fifteen is
`test_stats.py`'s whole share of the command's twenty, not a subset of a per-file twenty; and of
those fifteen, one is an import (`:38`), two are `parametrize` ids (`:1429`, `:1502`), five are
docstring prose (`:1437`, `:1454`, `:1477`, `:1980`, `:1983`) and two are precondition-raise calls
Rule 4a does not touch (`:1464`, `:1498`) — **four** are arm-value call sites (`:1260`, `:1338`,
`:1361`, `:1422-1423`). The command's coverage claim is unaffected; the sentence describing it is
not what a reader can reproduce. *(This is also where `P10-1` hides: the one line that matters,
`:1361`, is inside the fifteen the sentence waves at.)*

### The four judgements the gate was asked for

**(1) Does Table H's residual set discriminate a half-application?** **Residual 1 does; residuals 2
and 3 discriminate the right thing but are stated over an unfixed spelling (`P10-3`).** Residual 1 is
one command whose count is 2 with target 0 — rule 5(b)'s own named alternative — and `-rFn` prints
the surviving line, so a one-arm application reads 1 and says which. The **trigger for 2 and 3 is
correctly identified**, and I checked it rather than accepting it: a first-form residual over
`_compose`'s current body would be a trap, because a faithful edit can keep
`min(exact[0], mover[0]), max(exact[1], mover[1])` intact as a sub-expression of the new one, so a
retiring count over it reads 1 on a correct edit. Third-form is right. The pair's own split — one
per bound — is right too, and it is not the only guard: the note's assertion 8 (the commutation
sweep) also fires on a one-sided clamp, which is why `P10-3` is a major and not a blocker.

**(2) Does the nine-row site list miss a site?** **Yes — `P10-1`, and `P10-4`.** On the specific
question asked: every consumer of the widened return type *is* enumerated. `_compose` has exactly
two call sites at `7f865e2` — `stats.py:458` inside `conservative_envelope` and `stats.py:1183`
inside `verdict()` — and both have rows (the `conservative_envelope` row takes the first element;
the `:1184-1187` row states `verdict()` "takes both halves"). `stats.py:1360` (`bound_by=bound_by`
in the `Verdict` construction) needs no edit and correctly has none. `envelope_arms`'s own five
`stats.py` lines resolve: `:351` head, `:408` inside `_compose`'s docstring, `:458`, `:517`
(`_check_level`'s docstring, about levels not clamps — unaffected), `:1182` (still returns two arms).
What the list misses is on the **test** side, and it is the one line a residual cannot reach.

**(3) Does the `report.py:338` row cover the `bound_by is None` path?** **The `None` path needs no
edit and its absence from the row is correct; the row is wrong about the path it does cover
(`P10-2`).** `report.py:335-336` returns `f"- decided by: {v.decided_by}"` before any zip, and Rule
4a changes nothing about when `bound_by` is `None` — the token set widens only on the envelope path,
`verdict()` still sets `None` on the `mcnemar-exact` path, and `test_stats.py:1393`'s
`(bound_by is None) == (decided_by == "mcnemar-exact")` iff still holds. The `strict=True` zip is
also safe: `bound_by` stays a **pair** of tokens drawn from a three-token set, not a longer tuple.
So the gap `P8-2` found in that function does not recur here.

**(4) Is the §7 rule 3 raise correctly characterised?** **Yes, and it is slightly stronger than the
plan states.** Note Rule 4a's opening reads "*(v1.19, at plan-gate Pass 8's `P8-1`…)*" — read at the
note. The finding it closes is impl-gate `P8-1`; plan-gate `P8-1` is Tables C and G colliding on
`stats.py:159`, in this document, closed at v1.15. The sharper point the plan does not make:
plan-gate `P8-1` **did** route a question to the note under §7 rule 3, and the note answered it at
v1.18 §11.2.2 — so the citation does not merely name nothing, it names a *real, different,
already-closed* raise from the same review pass, which is the harder kind of citation to unwind two
revisions from now. The plan is **right that nothing depends on the resolution**: Table H derives
from Rule 4a's four points and ten assertions, none of which turn on which review raised it. Routing:
`data-scientist`, one clause in the note's Rule 4a opening.

### What's solid

- **The disowning-mention measurement, taken at design time.** Scoping residual 1 to
  `clamp=(-1.0, 1.0)` rather than the bare tuple is decided by a count — the bare form returns 3 at
  `7f865e2` and the third is `_widen`'s own docstring at `:253` explaining why that default is
  refused. Verified, all three lines. That is the v1.16 trap caught *before* it was written down,
  which is the whole point of a standing sweep.
- **Table E's "does not move" block.** Checked rather than assumed, and the check is the right one:
  Rule 4a changes what callers *pass*, not what `_widen` *does*, so the third-form pair over
  `_widen`'s body survives. I confirmed `_widen`'s body at `:257-261` is untouched by anything Rule
  4a specifies. Reporting a table that did **not** move, with the reason, is worth more than the
  three lines it costs.
- **The ruling is folded in, not re-derived.** The four-line summary, the ten assertions, the two
  witness tables and the commutation property are all cited to the note and none is restated with
  new numbers — the one place the restatement went wrong is `P10-2`, and it went wrong by
  paraphrasing a string the rule states verbatim, which is the case §7 rule 2 already covers.
- **The §7 rule 5(b) *exact text over line pin* clause is paid for, not asserted.** Both halves are
  exhibited — a line pin that broke (`tests/test_results.py:507` → `:543`) and an exact-text pair
  that survived two unrelated edits — which is the evidentiary standard this plan holds itself to.

### Open questions

**None for the caller.** One item is open against another document and is **blocked on its owner,
not deferred**: note `-ml` v1.19 Rule 4a's opening attributes itself to *plan-gate* Pass 8's `P8-1`
where it means *impl-gate* `P8-1` (`P10`, judgement 4). It routes to `data-scientist`; it blocks
neither Table H nor this gate.

## Appendix J — Pass 10: what was re-run and read

**J.1 — counts, all against `7f865e2` via `git show`.** Command 1 (`bound_by`) → **14**:
`stats.py` `:409`, `:896`, `:1168`, `:1184`, `:1360`; `test_stats.py` `:1371`, `:1375`, `:1378`,
`:1386`, `:1390`, `:1393`; `report.py` `:335`, `:339`; `test_report.py` `:1726`. Command 2
(`envelope_arms`) → **20**: `stats.py` `:351`, `:408`, `:458`, `:517`, `:1182`; `test_stats.py` 15
(listed in `P10-5`). Residual *before* values: `clamp=(-1.0, 1.0)` → **2** (`:382`, `:387`); bare
`(-1.0, 1.0)` → **3** (`+:253`, the disowning mention); `max(SUPPORT_DIFF_PROPORTIONS[0], lo)` and
`min(…[1], hi)` → **0**, **0**; `"MOVER-D" if mover_arm[0] <= exact_arm[0]` → **1** (`:1185`);
`arm if arm == "MOVER-D" else` → **1** (`report.py:338`); `tuple[str, str] | None` → **2** (`:896`,
`:1168`). `_compose(` call sites → **2** (`:458`, `:1183`). Both new names free at **0**/**0**.

**J.2 — the one simulation, read-only.** `modelbench/stats.py` at `7f865e2` was copied to a scratch
path and imported (stdlib-only module; nothing in `model-bench/` was read from the working tree or
written to). Rule 4a points 2 and 4 were re-implemented over it — both arms `_widen(…, clamp=None)`,
`min`/`max` composition, `max(-1.0, ·)`/`min(1.0, ·)` on the result — and the shipped test's own
sweep set was rebuilt from its source (455 tables at n=12 plus its n=40 stride = **511**) and scored
at its own three design effects. **Under Rule 4a: 0 / 38 / 78 failures at DEFF 1.0 / 1.2 / 2.0.
Control (today's arm-clamped code): 0 / 0 / 0.** The second arm-value test
(`test_both_arms_are_widened_about_the_same_point_by_the_same_factor`, `:1410-1426`) was scored the
same way and **passes under Rule 4a** on all three of its tables — its hard-coded `max(-1.0, ·)` /
`min(1.0, ·)` expected values are inert at DEFF 4.0 there — so it is named in `P10-4`'s vicinity as
fragile rather than broken, and is not a finding.

**J.3 — shipped lines read at their cited numbers** (all at `7f865e2`): `stats.py:236-262`
(`_widen`, docstring and body), `:351-390` (`envelope_arms` whole), `:392-413` (`_compose` whole),
`:458`, `:507-520`, `:896`, `:1168-1195`, `:1360`; `report.py:327-341` (`_decided_by_line` whole),
`:344-352`; `test_stats.py:1330-1372`, `:1408-1432`, `:1468-1496`.

**J.4 — documents read.** The v1.19 delta in full (+274/−31); plan Table H whole, Table E's two new
blocks, §4 S1e's preamble and count updates, DC-12's five-exception and four-way-partition
paragraphs and both standing-sweep updates, §7 rule 5(b)'s new clause, the *Version pairing* block
and the rule 3 raise, the v1.19 closeout table. Note `-ml` v1.19 §3.4 **Rule 4a** whole — the four
points, the commutation and containment sweeps, the separating case, the four rejected alternatives
and all **ten** assertions. Review `docs/reviews/small-model-benchmarking-impl.md` `## Pass 9` §5 whole
(the `_compose` seam adjudication, its five-item next-unit edit list, and the docstring obligation
Table H carries as its `:405-411` row) — used rather than re-derived, as commissioned.

## Pass 11 — 2026-09-08

**Scope, as commissioned:** two revisions — **v1.20** (`9aafc85`, +80/−14) closing this gate's five
`P10-*`, and **v1.21** (`ed33e92`, +63/−18), the mechanical re-baseline and the two §7 rule 5
conventions it forced. Both read in full. Measured against the worktree, which is clean and equals
**`93b0e42`** (the last commit touching `model-bench/`). Findings carry the prefix **plan-gate
`P11-*`** — the impl gate is numbering its own `## Pass 11` concurrently, which is the collision §7's
prefix convention exists for and the one already open as a §7 rule 3 raise against the note.

**Verdict: needs changes.** 0 blockers, 1 major, 2 minors, 0 nits. All three are closeable today;
**none is blocked on unbuilt work and none is deferred by choice.** The one genuinely blocked item
is unchanged and is not mine to close: the note's Rule 4a citation, **blocked on `data-scientist`**.

**CPG:** considered, not relevant — no `model-bench` CPG exists on this instance.

### Disposition of Pass 10's five — all five closed

| # | Disposition | Rechecked |
|---|---|---|
| **P10-1** (blocker; the site list omits the shipped test the edit falsifies) | **Closed** | The tenth row exists and is **pinned by test name, not by line** — the right call, and made for the right stated reason. `grep -rFn test_neither_printed_bound_is_ever_tighter_than_either_arm` → **1**, re-run. It carries the corrective (comparison moves to the **clamped** arms), cites Rule 4a's own restatement so it cannot be read as a weakening, names why relaxing instead would delete Rule 4 acceptance 4, and records that no residual can see it while command 2 does return the line |
| **P10-2** (major; the renderer row forbids what assertion 10 pins) | **Closed** | The row now states `support bound (-1)` / `(1)`, names `SUPPORT_DIFF_PROPORTIONS` as the renderer's source rather than letting `-1` become a second home, and corrects the last clause — assertion 10 pins the *correct* bullet and kills `support bound, p=0.025`. The table's own point **(iv)** summary is swept with it. `support bound (-1)` now appears in the plan (was **0**) |
| **P10-3** (major; a prescription stated only inside two residual commands) | **Closed, and further than asked** — see `P11-1` | The `:413` row writes the three-line body out in Rule 4a's own `u_lo`/`u_hi` notation and forbids the unpacking variant explicitly; residuals 2 and 3 are additionally restated over a fragment containing no introduced name. Generalised at §7 rule 5(b). The generalisation is right; the second repair introduced `P11-1` |
| **P10-4** (minor; two further stale docstring paragraphs, and an over-claim) | **Closed** | The docstring row extends from one paragraph to **three** and names each one's specific staleness; the v1.19 over-claim is corrected in the deletion row itself — one clause of `envelope_arms`' sentence becomes true, the other false, and only the first is that row's |
| **P10-5** (nit) | **Closed, then superseded by v1.21** | v1.20 re-derived it correctly; `93b0e42` staled the re-derivation within the hour, and v1.21 replaced the whole construction with a site list under its new gloss convention. The correction was right and its lifetime is the argument for the convention |

### Findings

**P11-1 (major) — residuals 2 and 3's new scope collides with a second consumer of the same constant
that the same revision mandates. `SUPPORT_DIFF_PROPORTIONS[0]` is stated over `modelbench` with a
target of **1**; the `report.py:338` row requires the renderer to reach that constant too, and one
plausible faithful spelling makes the count **2**.**
*Evidence.* Residuals 2 and 3 are now `grep -rFn 'SUPPORT_DIFF_PROPORTIONS[0]' modelbench
--include='*.py'` → **0 → 1** and the same for `[1]` (re-run: **0** and **0** today). The scope is
the whole package, and `modelbench/report.py` is inside it. The `report.py:338` row, landed in the
**same revision** by `P10-2`'s fix, says the renderer "needs the support *value*, and its source is
named here rather than re-spelled: `SUPPORT_DIFF_PROPORTIONS`". The renderer must produce `(-1)` on
the lower bound and `(1)` on the upper. Two faithful spellings exist and the plan prescribes neither:
extending the existing generator's `zip` with the support pair (no subscript — count stays 1), or
two explicit branches (`f"{arm} ({stats.SUPPORT_DIFF_PROPORTIONS[0]:g})"` — count becomes 2, since
`-F` matches the substring through the module qualifier). Confirmed the qualifier is what
`report.py` uses: it does `from modelbench import stats` (`:27`) and spells `stats.LEVEL_CI95_LO`
at `:339`, so the row's "imports … exactly as it already imports the two `LEVEL_*` constants" is a
module reference rather than an import, and either way the substring matches.
*Why it matters.* The v1.19 form (`max(SUPPORT_DIFF_PROPORTIONS[0], lo)`) was immune to this: it
pinned the clamp expression, which exists only in `_compose`. The repair removed the introduced
local and in doing so **widened the match to any subscript of the constant anywhere in the package**
— in the one revision that also mandated a second consumer of it. On the explicit-branch spelling
DC-12 fails on a faithful edit, which is `P10-3`'s own shape arriving by a different mechanism, and
the plan's new clause is the one that catches it: where the pinned text contains a name the edit
must introduce — *"a local, a helper, **a constant's spelling**"* — that spelling is fixed in the
row. The renderer's is not fixed anywhere.
*Fix.* Scope residuals 2 and 3 to **`modelbench/stats.py`**, as Table C's first two are scoped to a
file each: it partitions cleanly, keeps the one-per-bound split, is immune to the renderer's
spelling, and still reads 0 → 1 — **checked on a synthetic two-file probe rather than asserted**:
with both spellings planted, the package-scoped command reads **2** and the `stats.py`-scoped one
reads **1**. (The alternative — prescribe the renderer's expression in the
`report.py:338` row too — also works and costs more.) Either way, `P11-1` is the third consecutive
revision in which Table H's residual pair has had to be restated, which is itself worth one sentence
in the table.

**P11-2 (minor) — Table H states two baselines and says nothing about the split, while DC-12 makes
the stronger claim the table's second half does not.** The enumerating commands read "*both re-run
at **`93b0e42`** — re-pointed from `7f865e2` at v1.21 because this revision re-ran them*"; twenty
lines below, the residual block still reads "*all before values re-run at `7f865e2`*". Under
v1.21's own convention 1 that is **permitted** — a baseline is a property of the measurement, and
convention 1 states a necessary condition for moving one, not an obligation — so this is not a
factual error, and I re-ran all six at the worktree myself: **2 / 0 / 0 / 1 / 1 / 2**, unchanged.
What makes it a defect is narrower and is about statement, not fact: **DC-12 asserts, in this same
revision, that "all six of Table H's residuals … are unchanged across it [`93b0e42`]"** — a claim
that can only be made by having run them there. So the document says the six were measured at
`93b0e42` in one place and at `7f865e2` in another, and a reader of the table where baselines are
load-bearing hits both with no explanation. *Fix, one clause, either direction:* re-point the
residual block to `93b0e42` on the strength of the re-run DC-12 already reports, or state on the
table that the split is deliberate and why. I recommend the first — it makes the table
single-baselined again and matches what the revision actually did.

**P11-3 (minor) — the rewritten gloss names its four sites by *line number*, in the file its own
table will insert into, two paragraphs above a row that is name-pinned for exactly that reason.**
The gloss reads "**The four that are arm-value call sites are `:1260`, `:1338`, `:1361` and
`:1422-1423`**" — all four verified at the worktree, correct today. But Table H lands the note's ten
assertions in `tests/test_stats.py`, and their natural home is beside the envelope tests they extend
(`test_the_verdict_records_which_arm_bound_each_printed_bound` at `:1367`,
`test_neither_printed_bound_is_ever_tighter_than_either_arm` at `:1346`) — i.e. **between `:1338` and
`:1422`**, which moves two of the four pins as a direct consequence of applying the table. The gloss
is therefore stale-by-construction the moment its own edit lands. The plan already holds both halves
of this rule: `P10-1`'s row is pinned by test name "because a parallel unit is editing this file",
and §7 rule 5(b) says an exact-text pin is robust to edits elsewhere in a file where a line pin is
not. *Fix.* Name the four by test-function name, as the tenth row does. *(This also repairs
convention 2's rationale — see judgement 4.)*

### The judgements the gate was asked for

**(1) `P10-3`'s two repairs — can they disagree?** **Yes, and that is `P11-1`.** Taken separately
each is sound: the written-out body is a real prescription (it fixes `u_lo`/`u_hi` from the note's
own notation, and explicitly forbids `lo_b, hi_b = SUPPORT_DIFF_PROPORTIONS`, which is the faithful
edit the subscript residuals could not see — a good catch); and a residual over a fragment with no
introduced name is the stronger of the two forms in principle. What was not checked is the
*interaction with the rest of the same revision*: the second repair's fragment is no longer unique
to `_compose`, because `P10-2`'s fix put the same constant in `report.py`. Belt and braces hold only
while the braces are scoped to the belt's trousers.

**(2) Table H's split baseline — defect or correct by construction?** **Both, and I call it a
minor** (`P11-2`). Correct by construction as to *provenance*: convention 1 makes a baseline a
property of a measurement, so a table may legitimately carry two, and neither number is wrong —
verified at the worktree. A defect as to *statement*: the split is unexplained in the one table
where baselines are load-bearing, and it is contradicted by DC-12's own stronger sentence in the
same revision. I did not treat the coordinator's framing as a decision and did not assume the
answer; the deciding evidence is DC-12's "unchanged across it", which is a measurement claim the
table's second half does not reflect.

**(3) Convention 1 — a stated baseline moves only when the revision moving it re-runs the commands.
Sound, and I checked it against the coordinator's own rejected argument rather than inheriting the
agreement.** The architect is right, and this revision is its own witness: had v1.21 re-pointed
Table H to `93b0e42` **without** re-running, the table would have asserted "`envelope_arms` → 20 at
`93b0e42`", and the true count there is **21** (re-run: 21, `stats.py` 5 / `test_stats.py` 16). So
"the implementer runs it against the tree they have" does license publishing a count nobody took,
and it was one revision away from doing so. The two sub-rules follow correctly: a **landed** table's
rows are a record and re-pointing falsifies it; an **unlanded** table's baseline moves in the
revision that re-measures. One observation, not a finding: the gap is carried "once, in DC-12, as
*what has changed since*", and that paragraph is prose with the same staleness exposure as the gloss
— it names `93b0e42` and will be wrong when the next unit lands. It is self-limiting, since DC-12 is
re-run at the end of the round and the implementer re-measures everything then, but it is the one
part of the convention that has no mechanism behind it.

**(4) Convention 2 — a gloss names sites and states no total. It covers both failure mechanisms *as
they occurred*; its stated rationale over-claims, and the rewritten gloss keeps one of the two
exposures it removes.** The rejection of the flat prohibition is right and the narrowing is the
better rule: a gloss is what rule 5(a) asks for in spirit, and Table E's non-site list is the
correct model — banning glosses would have cost the thing that makes these tables usable. On
coverage: mechanism (ii), the stale total, is fully removed — with no total there is nothing an
addition can falsify. Mechanism (i), v1.19's mis-partition, was an arithmetic claim about a total
and cannot be written under the new rule either. **But the rationale — "a site list is stable
against additions in a way a total never is" — is true of a site's *identity* and false of a site's
*line number*, and the gloss states its four sites as line numbers** (`P11-3`). It also keeps a
universally-quantified complement — "*every other line the command returns in that file is an
import, a `parametrize` id, docstring prose or a precondition-raise call*" — which is the same
exhaustiveness claim in words rather than numbers, and which a future addition of a genuine call
site falsifies. Neither is fatal and both are one clause each; and the immediate application was
sound, because I checked what `93b0e42` actually added: the new sixteenth line is `:1465`, docstring
prose — a **non-site** — so the four-site list survived that event on its merits and not by luck.
The catch of a second copy of both counts three sections away is the rule paying for itself on day
one.

### What's solid

- **All five `P10-*` are closed on their merits, and two are closed further than the finding asked.**
  `P10-3`'s row now forbids a faithful-but-invisible variant (`lo_b, hi_b = …`) that I did not name;
  `P10-4`'s correction distinguishes the two clauses of one sentence rather than deleting it.
- **The blocker's row records *why* it is a row** — no residual can see it, while command 2 does
  return the line — which is the distinction between an enumeration failure and a transcription
  failure, and it is the right lesson to have drawn from `P10-1`.
- **v1.21 is the right response to going stale, and it is disciplined about scope**: three facts
  corrected, everything else explicitly re-checked and left alone, with the byte-identity of ten
  pinned `stats.py` lines across `93b0e42` stated as a measurement. I re-ran the six residuals, both
  enumerating commands and the four gloss pins and reproduced all of them.
- **Convention 1 was reached against the coordinator's stated position and is the better answer.**
  A rule that survives its own commissioner's argument, with a demonstrated counter-example one
  revision away, is worth more than one that was never contested.

### Open questions

**None for the caller.** One item remains open against another document and is **blocked on its
owner, not deferred**: note `-ml` v1.19 Rule 4a attributes itself to *plan-gate* Pass 8's `P8-1`
where it means *impl-gate* `P8-1`. It routes to `data-scientist`, blocks nothing, and this pass adds
no second instance of it.

## Appendix K — Pass 11: what was re-run and read

**K.1 — every count relied on, re-run against the worktree (clean, `93b0e42`).** `envelope_arms` →
**21** (`modelbench/stats.py` 5, `tests/test_stats.py` 16); `bound_by` → **14**; the six residuals →
**2 / 0 / 0 / 1 / 1 / 2** in order (`clamp=(-1.0, 1.0)`; `SUPPORT_DIFF_PROPORTIONS[0]`;
`SUPPORT_DIFF_PROPORTIONS[1]`; `"MOVER-D" if mover_arm[0] <= exact_arm[0]`;
`arm if arm == "MOVER-D" else`; `tuple[str, str] | None`);
`test_neither_printed_bound_is_ever_tighter_than_either_arm` → **1**, unique, so the name pin holds;
the four gloss pins read at `:1260`, `:1338`, `:1361`, `:1422-1423` and each is an `envelope_arms`
call — **all four correct**. `report.py` import style: `from modelbench import stats` (`:27`),
`stats.LEVEL_CI95_LO` at `:339`.

**K.2 — what `93b0e42` actually changed under Table H**, diffed line-by-line against the `7f865e2`
blob: `tests/test_stats.py` gains **one** matching line, `:1465`, which is **docstring prose**; the
seven `envelope_arms` lines at or above `:1454` shift by 1 to 23 lines and every one below is
unmoved, which is why the four call-site pins survived. `modelbench/stats.py`'s five are unchanged.
This is the measurement behind judgement 4.

**K.3 — documents read.** The v1.20 delta in full (+80/−14) and the v1.21 delta in full (+63/−18):
Table H's ten site rows, its two enumerating commands and rewritten gloss, its six residuals and the
third-form paragraph, the point-(iv) summary sweep, DC-12's re-baseline paragraph, §7 rule 5(b)'s
three new clauses (the write-out clause at v1.20, and v1.21's baseline and gloss conventions), the
v1.19 closure block's de-duplicated row, and the *Plan gate Pass 10* closure block with its
five-finding table.
