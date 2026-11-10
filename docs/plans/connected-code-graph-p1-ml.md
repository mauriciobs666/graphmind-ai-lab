# Connected code graph — P1 method note (indexers, flow, interpreter, noise, first gate)

> **Status:** active · **Owner:** `data-scientist` · **Tracks:** C-1001 (P1) · **Reviews:** `docs/reviews/connected-code-graph-p1-ml.md`

Unit N-ML of `docs/plans/connected-code-graph-p1.md` §9. It answers that row's six questions for
P1-15, P1-26, P1-34–P1-37, P1-39 and P1-47. It states only what P1 changes or fixes against the
standing contracts:
- `docs/plans/connected-code-graph-ml.md` (ML);
- `docs/plans/connected-code-graph-eval-protocol-ml.md` (EP);
- `docs/plans/connected-code-graph-eval-protocol2-ml.md` (EP2);
- the archived `docs/plans/connected-code-graph-p0-ml.md` (P0-ML).

Revised 2026-11-10 in place (Pass 5, provisioning decision).

A rule changed here applies from P1 on. The note holds method only. It writes no labels and names
no held-out content. Figures are **re-derived** from the P0 artifacts (appendix scripts) unless
marked **carried**.

## 0. Decisions, by question

| # | Decision | Where |
|---|---|---|
| 1 | Indexers: adopt on an offline figure, measured as a delta against P0's online run, with a fixed merge rule | §2 |
| 2 | Flow: entry value 465/481 (dev ceiling 467), so expected **not run**; synthetic conds become transparent in the projection; JavaScript not measured; the fallback rule is a dev superset measurement | §3 |
| 3 | Interpreter: local `qwen/qwen3-4b-2507`, span groups plus a deterministic resolver, repeat 1 scored | §4 |
| 4 | Noise: EP §11 plus a scope filter, two burdens and an operational baseline; thresholds ratified, gating from the next gate | §5 |
| 5 | First gate: record pins, contact definition and check, scope source, custody of held-out-derived artifacts, oracle, flow-fallback labelling | §6 |
| 6 | Scope overlap and catch-all clause | §7 |

## 1. Findings from the P0 artifacts

Sources:
- `cpg/spikes/frontend-fitness/truth/*.flow-final.jsonl` and
  `adjudication/*-flow-{misses,spurious}.adj.jsonl`;
- the python corpus source at `0caf85e3`;
- `cpg/eval/dev/*.jsonl` (dev only);
- `cpg/eval/heldout-manifest.json` (counts only).

**F1 — python flow arithmetic.** There are 481 adjudicated pairs (480 labelled plus the omission at
ordinal 22) in 29 non-empty methods; 308 are connected. The 173 misses split exactly as index §5 states.

| Fixed set (cumulative, on the 30 dev-contact methods) | Connected | Recall | Cluster 95% (P0-ML §7.3, n = 29) |
|---|---|---|---|
| shipped | 308 | 0.640 | [0.565, 0.715] |
| R1 + R3 (counterfactual-verified) | 392 | 0.815 | [0.743, 0.887] |
| R1 + 27 unverified + `for` header + R3 (the index's "455") | 455 | 0.946 | [0.910, 0.982] |
| **everything P1-09 lists** (… + contains-join 4) | **459** | 0.954 | [0.920, 0.988] |
| … + synthetic-cond transparency (§3.4) | **467** | 0.971 | [0.945, 0.996] |

- **The index's 455 omits the contains-join fix**, which P1-09 also ships. Rows from 438 on assume the
  fixes are additive; only R1 and R3 are per-pair counterfactuals.
- **The 9 `definition` misses** were classified from source at `0caf85e3` (an instrument separate
  from the appendix scripts). Each chain runs through one intermediate cond:
  - a `try`-lowering cond in 7 of them: `retry.py:134` ×5 and `services.py:449` ×2;
  - a generator-expression iterator cond in 1: `retry.py:81`;
  - a `for`-statement iterator cond in 1: `test_repository.py:4703`, reached by a loop-carried edge.
- **Residual misses** at 467: 14, in 6 methods (4, 4, 3, 1, 1, 1). The ANOVA intra-method
  correlation of misses is 0.04–0.17 across rows.

**F2 — operating curve for python's real structure (16.6 pairs per method).** P(observed pooled ≥ 0.95)
for a fresh 30-method draw, with method sizes resampled from the empirical m_i:

| True recall | 0.98 | 0.975 | 0.97 | 0.965 | 0.96 | 0.95 | 0.93 | 0.90 |
|---|---|---|---|---|---|---|---|---|
| ρ = 0.1 | 0.98 | 0.96 | 0.91 | 0.85 | 0.76 | 0.54 | 0.17 | 0.01 |
| ρ = 0.2 | 0.96 | 0.92 | 0.87 | 0.80 | 0.73 | 0.56 | 0.25 | 0.04 |

- The index's 0.61 / 0.26 / 0.04 is P0-ML's independent n = 120 line, not python's clustered
  structure.
- **Pair counts.** 8 methods fall below 120 pairs with probability 0.33; 12 methods with probability 0.005.

**F3 — JavaScript.**
- **Pairs.** 152 adjudicated, 61 connected. Frontend misses 62 (52 switch-label endpoints, 9 in the
  `src/i18n` method, 1 JSX); projection 18; join 4; definition 4; undetermined 3.
- **Ceiling.** 83/152 = 0.546. At most 87/152 with transparency. 92/152 = 0.605 even if the i18n
  method were recovered and every one of its pairs connected.
- **Population.** The 5–25-line population (12 methods) was drawn whole and labelled, so it is
  development contact. The 3 tier-2 methods were drawn but never labelled, so they are not contact.
  They cannot reach 120 pairs, though, so **no blind JavaScript flow sample of adequate size exists
  on this corpus**.

**F4 — the as-built flag-false walk against the dev flow items** (a prediction from hop kinds, not
P1-39's measurement).
- `cpg/eval/dev/ac5-flow.jsonl` holds 28 items in 4 anchor clusters.
- The as-built walk lists callers only, to distance 2. Only the 10 pure `return-use` chains end at a
  caller: **10/28 listed** (python 6/8, TS/JS 4/20).
- The 18 other chains contain a `callee-arg`, `control` or `shared-state` hop.

**F5 — catch-all routes.** EP Appendix B's patterns give **0** route-parameter catch-alls:
- at HEAD `a73cf229`, over non-test files outside `cpg/` and `deprecated/`;
- at `EVAL_COMMIT` `cf7b2042`, with tests included (carried from the review's sweep).

The one mount-level fallback (`falkor-chat/server/falkorchat/app.py`, `_SPAStaticFiles`) is not a
route parameter.

**F6 — seed prefixes.** The P0 fitness flow draw used `ccg-p0a-v1-flow|`. The prefix
`ccg-p0f-flow-v1|` belongs to EP §3.6's AC-5 flow anchors. P1-34's "not `ccg-p0f-flow-v1`" names
the wrong one.

## 2. Question 1 — indexers offline (P1-15)

**Why P0 cannot decide.**
- P0's indexer counterfactuals (fitness report §a1.2, carried) were measured with dependencies
  installed, and the reference was half of the truth.
- Python adopts on **precision**: 110/116 = 0.948 [0.892, 0.976], one site short of 0.95.
- TS/JS adopts on **recall**: decision stratum 96/165; 146/165 = 0.885 with the reference.
- No language reaches 0.98 even with the reference, so 0.98 is not P1's adopt bar.

### 2.1 Measurement protocol

**Tools.** The P0 reference versions, `scip-python 0.6.8` and `scip-typescript 0.6.0`
(`corpora.yaml`). A version change confounds M1. If those versions are unavailable, the delta is
reported and M1 is re-baselined.

**Corpus and scoring.** The P0 corpora at `0caf85e3` and the call-site truth
`truth/<lang>.final2.jsonl`, scored by `s27_metrics.py` (imported) on P0's decision stratum, with
pooled beside it.

**Provisioning is an operator step, outside the system** (stakeholder, 2026-11-10).
- The operator installs the dependencies at the corpus commit, hardened per
  `docs/plans/connected-code-graph-egress-spec3.md` §7.4.
- The operator writes `env.json` in scip-python's `--environment` format, which avoids its pip
  discovery. P0's online run used that discovery, a named confound for M1.
- The run records the lockfile and `env.json` sha256s, mounts both read-only, and runs
  `--mode offline`. The argv check asserts no `npm`, `pip` or `uv` invocation.

**What the sandbox runner must provide** (`cpg/sandbox/offline-run.sh`).
- Its closed frontend set (`SBX_FRONTENDS`) has no indexer entry today. That set, with its
  executables and flag rules, is a sandbox contract (`docs/plans/connected-code-graph-egress-spec3.md`
  §7.4). The change is therefore routed to `architect`, as a contract revision plus its
  `offline-run.sh` implementation (index owner). This note states only the requirement.
- The measurement needs:
  1. `scip-python` and `scip-typescript` entries in the closed set, with their executables in the
     `indexer` image;
  2. read-only mounts for `node_modules`, the venv and `env.json`;
  3. allowed-flag lines for `index`, `--project-name`, `--environment`, `--cwd` and `--output`,
     with no install or network flag allowed;
  4. the usual audit record per run, whose digest P1-15's report cites.
- Until this exists, P1-15's done-condition cannot be met. The step reports **"not run: runner"**,
  not "not adopted".

**Measures:**
- **M1 — offline/online delta (primary).** Per labelled site, compare the offline SCIP resolution
  with P0's `truth/<lang>.ref.jsonl`. Report agreement as k/n with Wilson. `data-scientist`
  adjudicates every differing site from source.
- **M2 — merged.** Recall `R_m` (strict P0-ML §2 "found", SCIP targets added). Merged confirmed
  precision `P_m` with n_P. Joern-alone `R_j` and `P_j` from the same run. The confirmed share.
  Fan-out p90.
- **M0 — bare** (no dependencies). Informational only: an unprovisioned build runs Joern alone with
  a `headers-unavailable`-class reason.
- **Cost.** Wall time and peak memory on the 6 GB host.
- **Stated bias.** Truth is the union of hand labels and the online reference. Callees both sources
  missed are invisible, so `R_m` is overstated. Agreement rows were not re-read, which favours
  `P_m`. The result is a screen.

### 2.2 Merge rule (fixed here; implemented, never tuned)

The rule acts per call site, on Joern's resolved in-corpus targets `J` and SCIP's in-corpus target
`S`. A target counts only if it is a symbol under bundle rule 8. SCIP `local` and external results
count as "SCIP external".

| Case | Result |
|---|---|
| `S ∈ J` | `S` confirmed `resolved-call`; others stay `possible` |
| `J` empty, `S` present (incl. no Joern `CALL`) | `S` confirmed `resolved-call` |
| `S ∉ J`, or SCIP external while `J` is in-corpus | every `J` member and `S` `possible` / `resolution-unverified` |
| SCIP silent | Joern unchanged |

- **No candidate is ever dropped**, so `R_m ≥ R_j`, and a test asserts it.
- A rule change after M2 is reported "after adaptation", with both figures.

### 2.3 Adopt and fold criteria (per language, offline M2, decision stratum)

- **Python:** adopt iff `P_m ≥ 0.95` with n_P ≥ 60.
- **TS/JS:** adopt iff the merge closes ≥ `ceil(k_j / 2)` of Joern-alone's strict misses `k_j`
  (P0: 69, so ≥ 35), **and** `P_m ≥ 0.95` with n_P ≥ 60. This is P0-ML §8.1's half-the-misses rule.
- **Fold or list:** P0-ML §8.3 applied to the merged output.
- **Stopping and freeze:**
  - One offline run per language. A tool failure is repaired; a low figure is not re-run.
  - P1-15 lands before the `ac5-gate-1` record or is declared out of gate 1 (index §7).
- **Record `ccg-p1-indexer-v1`** (§8): versions, image digest, lockfile, `env.json` and freeze
  hashes, the merge-rule version, the `s27_metrics.py` sha256, and these criteria.

## 3. Question 2 — flow re-measurement, transparency, the fallback

### 3.1 Entry check (P1-34 step 2)

The entry is a cost guard, computed under zero optimism: fresh-draw recall is assumed equal to dev
recall, although R2, R3 and the unverified causes were designed on those methods.

**Rule.** Enter iff P1-09's python dev recall (481 pairs, the shipped direct-edge scorer, after
§3.4's projection rule) is **≥ 465/481**.
- 465 is the smallest count at the true recall where a 30-method fresh draw passes with probability
  ≥ 0.80, at ρ = 0.2 with resampled sizes (F2: 0.965).
- Across alternative models the threshold spans 461–465 (carried from the review's independent
  instrument): independent misses, ρ 0.1–0.3, fixed or resampled sizes. 465 is the conservative end.

**Outcome.**
- The dev ceiling is 467, so entry needs at most 2 of the 159 fixable misses left open. 27 of those
  fixes are unverified.
- **Expected outcome: not run.** `flowAvailable` python stays false and is reported not closed, with
  the figure.
- P1-09 reports the figure with its cluster interval and the list of fixable misses still open.

### 3.2 JavaScript

**Not measured in P1** (F3): the ceiling is 0.546, and no adequate blind population exists. A
measurement needs either a widened population rule recorded before any draw, or a second TS corpus.
It belongs with the TS/JS remedy backlog item (index §11).

### 3.3 Fresh-draw design (only if entered)

**Frame.**
- P0-ML §6.4's population at `0caf85e3`, minus the 30 P0 methods.
- Minus every held-out AC-5 flow anchor. The custodian applies this exclusion and reports the count
  only.

**Draw.**
- 30 methods, by P0-ML §4.4's hash sort with `seed = "ccg-p1a-v1-flow|python|" + corpusCommit`.
- The spurious sample uses `ccg-p1a-v1-spur|`.
- 30 methods, not 12: better discrimination on both sides of 0.95 (appendix `oc.py`).

**Labelling and adjudication.**
- One fresh blind `qa-engineer`, using P0's brief and patch and P0-ML §6.2's must-not-see list, then
  a Part F-style audit.
- `data-scientist` adjudicates every miss and a 50-pair spurious sample, under P0-ML §8.4 (v1.17),
  fixed before labelling.
- A ruling made after results is disclosed, and the decision is reported under both readings.

**Configurations.** Both run. They are identical by construction, so any discordant pair is a defect.

**Decision**, by P0-ML §8.4: pooled ≥ 0.95 over ≥ 120 pairs in both configurations, with projection
success ≥ 0.95. Reported beside it:
- the cluster interval;
- the omission rate and the truth-incomplete flag;
- the spurious rate;
- the number of methods fully recovered.

A true flag flips the D2 row only through the decision record. Held-out `flow` is then gated only on
items whose chain stays in flag-true units, and fewer than 8 such items means "present, not gated".

**Records:** `ccg-p1-flow-python-v1`, before labelling, and `ccg-p1-flow-python-v1-decision`.

### 3.4 Synthetic-cond transparency (a projection rule for P1-09)

**Rule.** A frontend-synthetic cond is transparent to the projection. When a point's
`CONTROLLED_BY` edge reaches a synthetic cond `S` that is itself controlled by a non-synthetic cond
`C`, the projection emits `point → C` **in place of** `point → S`. This is the cond analogue of R3.
- The synthetic set is a closed per-language list of Joern cond shapes, taken from P0-ML §8.4's
  "not points" rule. Each entry is a recognizer stated on the source construct. `graph-dba`
  implements it as a CPG predicate, and the goldens below decide when the predicate and the text
  disagree.
- **Python `try`-lowering cond.** A CDG source that is not the condition of an `if`, `while`,
  `for`, `match` or ternary, and not a short-circuit operand, whose statement lies directly in a
  `try` body. Joern prints it as that statement's code: `open_ = keep_going()`, `return json.loads(value)`.
- **Python generator/comprehension iterator.** A cond coded `iteratorNonEmptyOrException` whose
  enclosing construct is a generator expression or a comprehension, not a `for` statement.
- **Not synthetic: the `for`-statement iterator.** It carries the same code,
  `iteratorNonEmptyOrException`. Its enclosing construct is a `for` statement, and P1-09's join fix
  maps it to that statement's condition point. The two iterator goldens below are what tell the
  two cases apart.
- **Nested and ungoverned cases.**
  - The redirect walks up through consecutive synthetic conds to the nearest non-synthetic cond, and
    emits one edge per nearest real cond when there are several.
  - If no real cond governs the chain, the point's edge is dropped: the point is unconditioned within
    the method.
  - Each case gets a golden.

**Scoring is unchanged.** Control pairs stay **direct-edge** (P0-ML §7.3; `s28_metrics.py` as is).
The rule changes the edges the projection emits, not the definition of a connected pair. No chain
scorer is needed.

**Why this form.**
- P0-ML §8.4 already calls a pair that touches a synthetic cond spurious. Routing through such a cond
  is a frontend artefact, not a control fact, so redirecting to the governing real cond agrees with
  the spurious-pair rules that the adjudication uses.
- The engine walks the emitted edges. The answer therefore lists exactly what the scorer credits.

**Rejected:**
- Crediting any chain of `CONTROLLED_BY` edges. All 9 `definition` chains run through conds that
  the same rules call non-points, or through a loop-carried edge. Recall and the spurious rate
  would then judge the same edges in opposite ways.
- Materialising the transitive closure, which adds spurious edges and breaks the
  nearest-governing-cond rule.

**Consequence.** The `test_repository.py` pair, whose route is a loop-carried edge through a real
`for` cond, stays a miss. The ceiling is therefore 467, not 468.

**Goldens and controls (P1-09).**
- **Positive:** the 8 pairs (`retry.py:134` ×5, `services.py:449` ×2, `retry.py:81`) connect by a
  direct edge, and no projected edge in those methods ends at a synthetic cond.
- **Joint golden on `services.py::_normalize_opaque`.** Line 449 is both a `try`-lowering cond and a
  call that is itself a CDG source, one of the unverified causes. One golden asserts all three pairs
  together: 447 → `loads` (449), and 447 → the returns at 449 and 451. Neither fix can then undo
  the other.
- **Negative:** the `test_repository.py:4703→4706` pair stays unconnected, and a point under a real
  intermediate cond keeps its direct edge to that cond.
- **Mutant:** with transparency off, all 8 positives fail.
- **P1-09 behaviour (5)'s no-regression subset becomes:**
  - the shipped `controlledBy` records are a subset of the fixed ones, **modulo the rewritten
    synthetic-cond edges**, each listed with its **replacement set**: one edge, several (one per
    nearest real cond), or none (an ungoverned edge dropped);
  - and **no golden pair the shipped projection connects is unconnected by the fixed one**.

  `slice/test_projection_not_regressed.py` carries the same carve-out.
- **Spread:** 467 with both recognizers, 466 with `try` only (carried from the review). No residual
  miss is reachable by the rule.

**Disclosure.** Like R3, this rule was designed after the misses were read. It is fixed here, before
any fresh draw.

### 3.5 The flag-false fallback (P1-39)

**Constraint.** FR-8 requires an impact that cannot be proven to be listed as `possible`. An impact
through a flag-false unit is such an impact. A walk that silently omits it does not meet FR-8. So
the as-built caller-only walk, predicted to list 10/28 (F4), is not an acceptable fallback unless
P1-39 measures it at n/n.

**Measurement** (P1-39, measurement only, on the snapshot built at `EVAL_COMMIT`, where the dev
items' locators were validated):
- For each of the 28 dev flow items: listed or not, distance and chain signature.
- The share `s` pooled, by language and by signature, with anchors as clusters, and the unlisted
  items by id.
- These figures for four walks:
  - the as-built walk;
  - **V-flow**: the flag-true rows (`DATA`, `CONTROLLED_BY`, `arg(i)`/`Ret`) walked in flag-false
    units, every added-row hop `possible`, `flow-unavailable:<lang>`;
  - **V-both**: callers and callees to bound 2, every added-row hop `possible`;
  - **V-union**: V-flow ∪ V-both, computed from the two listed sets.
- Burden per dev anchor, for each walk.

**Rule.**
- Ship the walk with the **lowest dev burden among those with the maximum `s`**.
- "Safe" is claimed only at n/n; 28/28 is a screen, with Wilson lower bound 0.879.
- If no walk reaches n/n, the residual misses are recorded by signature, reported not closed.
- The choice follows from FR-8 and this pre-registered rule. It is a design decision for `architect`
  and P1-23, not a stakeholder question. The proposal's "call-level only" is design text, not a
  requirement.
- A new walk changes the walk definition, so the `algorithm.version` bump follows answer-spec2 §1.2.

**Consequence for AC-25.** AC-25 ("both effects are listed … confirmed before possible") and FR-8
already accept `possible` for an impact that cannot be proven.
- If V-flow or V-union ships, AC-25's control half closes on P1-42's fixture.
- If the as-built walk or V-both ships and does not list the guarded point, AC-25 is not closed. Only
  a stakeholder waiver would close it.

**Record:** `ccg-p1-flow-fallback-v1`, before P1-39 runs.

## 4. Question 3 — the interpreter and the AC-24 smoke

### 4.1 Model

**Recommended:** `qwen/qwen3-4b-2507`, Q4_K_M, local on LM Studio.
- It is in the live catalog (read 2026-11-09) and is `falkor-chat`'s default
  (`falkor-chat/config/models.json`).
- The prior evidence is the nearest measured construct, model-bench's `nlq-structured-query`: it is
  the lead pick on the full 34-item denominator
  (`model-bench/reports/catalog-sweep-2026-09-19-consolidated.md`). That task is not AC-24, and the
  smoke run is the test.

**Rejected:**
- `mistralai/ministral-3-3b`: its template returns HTTP 400 on a repeated role or a second system
  message (`claude/data-scientist/lm-studio-model-notes.md`).
- Thinking variants: zero scoreable observations on the same pack.

The choice is perishable. The gate binds what the endpoint reports.

**Binding.** The bound model is fixed by AC-24's own text: "the model that actually interprets that
codebase's questions (local in any system containing a local-only repository)"
(`docs/requirements/connected-code-graph.md`, AC-24). It is identified at run time by the
provenance fingerprint (§4.2).

### 4.2 Draft, grammar, prompt, sampling (P1-26)

**Division of labour.** The model extracts spans. The deterministic resolver (`resolve.py`) decides
ids, builds the canonical `Request` and produces `options`. The model never sees the symbol list
and never emits an id. The reply is a projection of `Request`, validated against its own schema,
and the built `Request` is validated against the canonical one.

**The `InterpretDraft`:**
- `useCase`: the built `impact`, plus every unbuilt use-case name;
- `direction`: `impact`, `depends-on`, or **`null` when the question states no direction**.
  - A direction is stated by any wording that implies it. **The cues below are examples, not a
    closed list.** For impact: "ripple effect", "consumers of", "what would fail", "o que é
    afetado", "consumidores de". For depends-on: "relies on", "requires", "built on top of",
    "precisa de", "chama o quê".
  - The examples deliberately share no phrase with any dev AC-24 question (checked against
    `cpg/eval/dev/ac24.jsonl`), so the dev entry check is not tuned on its own items.
  - Neutral wording such as "relations" or "relações" states none.
  - The resolver expands `null` into **both** directions as `options` (an ambiguous answer). The
    model never picks one silently, which EP §10.4 would score as a failure;
- `anchors`: 1–3 **span groups** `{spans: [{text, kindHint}]}` (1–3 spans each), where `kindHint`
  is one of `symbol`, `file`, `line`, `route`, `tool`, `server`, `other`. One group names one anchor.
  Spans in a group qualify each other;
- `scope`: a span or null;
- `readings`: 0–3 alternative `{useCase, direction}`, used when the question names two use cases.

**Fabrication guard.** Every `text` must be a verbatim substring of the question. A non-substring
span is dropped. A group left empty makes the answer `unresolved`.

**Resolver join rule, per group.** A group is resolved to a unique anchor, or to `options`:
- `symbol` + `file`: a qualified-name match, where the file matches as a **path suffix**, so
  `decisionmodels/runner.py` matches the repository path;
- `file` + `line`, or `file:line` in one span: the eligible element enclosing that line (`E`, EP
  §2.3);
- `symbol` + `server` or `tool` + `server`: candidates restricted to the unit or MCP server whose
  name matches the span;
- `route`: `keyNorm` (EP2 §1.1);
- several candidates left: `options`.

**Named resolver tests** (dev questions): QW-30 (symbol + file), QW-27 (partial `file:line`), QW-13
(tool + server), QW-7 (file + symbol in reverse order), QW-47 (`direction: null` → both directions
as options). The reverse error, a `null` on a question that does state a direction, counts as
over-asking. QW-3 ("blast radius") and QW-11 ("what calls this route") are negative tests: a
direction stated without one of the example cues must not come back `null`. The dev entry check
measures the rest.

**Grammar.** The draft's JSON Schema, `cpg/ccg/front/grammar/interpret.json`, passed through LM
Studio's `response_format` `json_schema` (strict). P1-26 verifies against current LM Studio docs
that the enforcement is grammar-level for GGUF. If it is not, a schema-invalid reply is
`unresolved` and is never repaired by a second call.

**Prompt** (`cpg/ccg/front/prompts/interpret.txt`, not `.md`, D5):
- one system message with the task, the fields and enums, the span rule, the "not built" list, and
  the rule that two use cases go to `readings`;
- one user message holding the question;
- at most 4 few-shot examples, over fixture names (`fx-*`) only, covering EN and PT, both
  directions, a `null` direction, one qualified group, one unbuilt use case and one two-use-case
  question.

Dev questions are kept for scoring, and held-out questions are used nowhere.

**Sampling.** `temperature` 0, `top_p` 1, `max_tokens` 384, `seed` 0 where honoured. Only a logged
transport retry is allowed. `samplingHash` is the sha256 of the canonical JSON of these parameters.

**Provenance.** The fingerprint is read from `/api/v0/models` at call time:
`id|publisher|arch|quantization|loaded_context_length`.

### 4.3 Smoke protocol (P1-35)

**Counts and thresholds.** These follow EP §10.5 on the split counts. The harness re-counts first,
and on a difference `data-scientist` recomputes them with the same `ceil` rule.

**Dev entry check.** On the 14 dev items, on repeat 1: ≥ 11/12 unambiguous and 2/2 ambiguous. A
failure means no held-out run.

**Held-out run (once).**
- Each item runs 5 times with `cacheBypass` (EP §10.4).
- **Repeat 1 is the scored draw.** A user meets one fresh interpretation per snapshot, which is then
  cached.
- Repeats 2–5 give the flip rate (k/30) and the per-draw understood rate, reported only.

**`cacheHit == true` on repeat 1.** It scores the item "not understood" and **stops the run** as a C5
failure (P1-32b): the cache must be empty for the gate snapshot. A `cacheHit` on repeats 2–5 voids
only those repeats.

**Cap (C7).**
- `capFloor` 5 and `capRule` `max(capFloor, |goldOptions_i|)`, as in EP §10.4. The dev ambiguous
  items have `|goldOptions|` 3 and 2.
- The resolver's did-you-mean list is capped at 5 by construction: edit distance, then qname.

**Record `ccg-prereg-ac24-v1`** (EP §12.3 schema):
- `interpreter` from the final dev run's provenance;
- `devExtras` from that run;
- `cacheBypass` = the input name answer-spec2 §2 fixes.

It is committed after the last prompt change and before the held-out run. The run's outputs live
under the custodian root (§6.4).

## 5. Question 4 — noise (P1-36) and its thresholds

### 5.1 Additions to EP §11

1. **Scope filter.** Out-of-scope items (§6.3) leave the frame before allocation. They are counted
   per answer and per reason, and never labelled for Q.
2. **Burden twice:** in-scope, which is the frame the thresholds were conceived on, and all-listed,
   which is what the user sees. Test-code consumers can dominate the second.
3. **Naive over-lister.** For each held-out anchor, every symbol within bound 2 reachable by every
   name-match candidate of every call hop, plus every provider and consumer of the same contract
   kind, all listed.
   - It is computed deterministically from the same snapshot.
   - Reduction is Σ baseline burden ÷ Σ system burden, with EP §11.3's answer bootstrap.
   - The baseline's recall is computed with the same hit rule. If it is lower than the system's, the
     reduction is flagged.
4. **Noise split by the hop's own edge status** beside the headline. A chain carrying
   `flow-unavailable` is reported with the status its edge would have in a flag-true unit (§6.6).
   This keeps the noise figures comparable across phases whose flags differ.
5. **Labeller.** One fresh `qa-engineer`, not a pack author. EP §11.4's 20-label spot-check goes to
   the stakeholder where available, otherwise to a second fresh labeller. Raw agreement and κ are
   reported; below 0.8 raw agreement, the possible-precision figure is marked "labels unstable".

### 5.2 What the thresholds mean

- **Possible precision ≥ 0.20 implies Q ≤ 4.** With pp = R/(R+W) and Q = W/(RC+R), pp ≥ 0.2 ⇔ W ≤ 4R
  ⇒ W ≤ 4(RC+R). The converse fails when RC > 0. The floor is the stricter bar, and Q ≤ 4 is the bar
  gated meanwhile.
- **At n = 100**, a measured 0.20 has Wilson [0.133, 0.289], so a pooled value within about ±0.08 of
  the floor cannot be told apart from it.
- **p90 burden** at gate 1 is the 56th smallest of 62 answers, reported with its bootstrap interval.

### 5.3 Noise thresholds (decided)

**Stakeholder decision (2026-11-10): ratified as proposed, before gate 1's measurement.** The bars are:
- Q ≤ 4;
- possible precision ≥ 0.20 pooled;
- p90 burden ≤ 100, judged in-scope with the all-listed figure reported;
- reduction ≥ 5×.

**They gate from the next gate on** (P1-43 if it runs, else P2's first gate).
- Gate 1 stays gated at Q ≤ 4 (EP §11.3). The other three are reported there.
- P1-37 reports each value with its interval, both burdens, and the reasons under 0.05 (ML §4.2).
- Record: `ccg-noise-ratify-v1`, cited by the next gate's pre-registration.

**What it overturns.** The decision precedes the first measurement. It therefore overturns the
"provisional until measured" wording in three places:
- AC-5's "(provisional; ratified or adjusted once the first phase has measured the real ratio)";
- decision S8's "provisional until phase 1 measures it";
- EP §11.3's "ratified after P1's first measurement".

From P1 on, this section replaces EP §11.3's sentence. The requirements side is `tico`'s entry
(§10).

## 6. Question 5 — first gate: record, contact, scope, custody, oracle

### 6.1 What this note rules on in `ac5-gate-1`

The rest of the record is EP §12.3.

- **`algorithm.version`** is the exact string the engine emits at the freeze commit, in the
  `MAJOR.MINOR` form and with the value answer-spec2 §1.2 finally fixes. EP §12.3's `"<semver>"`
  placeholder means that string, compared by string equality.
  - The walk changes this note recommends or records (§3.5's fallback walk, §6.6's labelling) are
    walk-definition changes and bump it under that section's rule.
  - The record therefore follows them.
- **The harness that scores the gate is the one at the record's commit.** `cpg/eval/tools/` is
  outside `coreTreeHash`. P1-32b compares the harness files' sha256 with
  `git show <recordCommit>:<path>`.
  - An edit after the record makes the run "after adaptation".
  - The exception is a defect fix shown, by re-scoring with both versions, to change no item's
    score. It is reported.
- **Report units.** Recall is given over items, distinct anchors and clusters. The zero-miss Wilson
  lower bounds are 0.976 (159 items), 0.942 (62 anchors) and 0.938 (58 clusters). With misses,
  P0-ML §7.3's cluster interval applies, with anchors as clusters.
- **Chain length** (ML §3.6's revisit): inter-function hop counts over hit items, median and p90 by
  stratum.

### 6.2 Contact

**Roles, per dispatch.** A dispatch is a **builder** dispatch if it writes engine, extractor, pack,
prompt, grammar, golden or fixture code, or a brief for one of them. It is an **exposed** dispatch
if it reads held-out or held-out-derived material: the custodian, the gate runner, and the method
owner at triage.
- Evaluation tooling (P1-32, P1-35, P1-36) is written in builder-safe dispatches **before** the record
  and pinned by §6.1.
- An exposed dispatch never precedes a builder dispatch of the same tooling.

**An anchor is contacted** (EP §8.1, "computed, asserted or hand-walked a link or an impact from that
anchor") when a builder dispatch does any of these before that gate's first run is recorded:
- reads held-out or held-out-derived material;
- commits an artifact asserting a link or impact from that anchor;
- debugs against that anchor's items after a gate.

EP §8.2's exclusions stand: wrapper and chain hops, and an element appearing as an impacted element
or hop.

**Freeze-time check.** Before the record, the custodian runs `split_manifest.py check-report`
(EP §9.4) once per file, with `--heldout` listing every held-out file plus the oracle candidates,
over the **path set**:
- **The set** is `git diff --name-only <S47> <record>`, with `S47` = `511b14cf`: every file changed
  between the split and the record commit. It covers specs, reviews, skills, agent prompts and
  `cpg/spikes/` edits alike.
- **Excluded**, with the reason recorded per path:
  - paths whose changes in the range all come from **non-P1 commits** (no P1 dispatch touched
    them): no P1 builder wrote there.
    - A commit is P1 iff its message matches `C-1001:? P1`. This catches `C-1001 P1-13` and
      `C-1001: P1 plan set`, but not the `C-1001 P0 close` commits `84ac62d5` and `262161e7`.
    - A mixed commit counts as P1. `teco` puts the tag on every P1 dispatch commit.
  - custodian outputs under `cpg/eval/` (the held-out manifest, `cpg/eval/dev/`, the records in
    `preregistration.jsonl` and `decisions/`): already public as dev or C.
    **`cpg/eval/tools/` is not a custodian output** and stays in the set.

  Nothing else is excluded. In-scope files a P1 dispatch changed stay in, for example P1-27's
  `skills/cpg-analysis/SKILL.md` and `claude/teco/teco.md`, and P1-25's `.mcp.json` proposal.
- **Exit codes.** 0 is clean and 1 is hits. **2 or 4 (usage or unreadable file) fails the check**, and
  is never treated as a skip.
- **Reverse map.** Hits print as 12-hex probe hashes. The custodian recomputes the verb's own probe
  hash over the held-out probes privately, so each hit can be traced to its probe without the output
  leaking.

**Each hit is triaged by the custodian** from the probe it matched:
- *contact*: the artifact asserts a link or impact from a held-out anchor. That cluster moves to dev,
  and the move is reported with its count;
- *mechanism*: an EP §8.2 exclusion. It is recorded, and nothing moves.

**Positive control.** A held-out probe planted in a scratch copy of one artifact must be reported as
exactly one hit.

**Limit.** The check sees committed text only. Uncommitted look-ups are covered by the brief rule:
builders query only dev anchors and fixtures on the proving repo. The gate report states this limit.

### 6.3 Scope

- **Scope source.** `cpg/eval/eval-manifest.yaml`'s `exclusions` at `EVAL_COMMIT`, under which the keys
  were written. It is not the system manifest (P1-03). A disagreement between the two is reported
  as a defect.
- **A listed out-of-scope item** is flagged `out-of-scope`, counted, and never scored as a hit, a miss
  or a wrong-possible item. P1-32b and P1-36 apply the same filter.
- **A held-out item that is out of scope** under this source is triaged `key-error` (EP §12.1), which
  the stakeholder confirms. It is never dropped.

### 6.4 Custody of held-out-derived artifacts

**Rule.** Every artifact derived from held-out items lives under `~/.local/share/ccg-eval/<E12>/`.
`<E12>` is the first 12 hex of `EVAL_COMMIT` (now `cf7b2042cce7`, as S47 wrote it). It is unrelated
to EP decision E12.
That covers:
- gate answers for held-out anchors;
- the miss and triage lists;
- the noise frame, sample and labels;
- AC-24 run outputs;
- oracle edges and misses;
- the next gate's candidate list.

Committed reports carry counts, hashes and classes only.

**Why there.** There is **no read guard**: EP §9.3's manifest detects modification, not reading. The
location is chosen for three reasons:
- it is outside every builder's working tree and sweep paths;
- it is outside `cpg/.ccg-work/`, which builders must use (D9);
- it sits with the custodian's other files.

The brief rule ("builders never read it") is the control.

**Leak check.** `check-report` takes the oracle candidates as additional probes, written in the
key-item schema, beside `heldout/*.jsonl`. Every committed gate, smoke and noise report passes it
before commit.

### 6.5 Runtime oracle (P1-47 capture, P1-36 scoring)

**Observed edge:** `(consumer element, contract key)`.
- The consumer element is the element that fixes the request (EP §3.2), attributed on the client side.
- The key is `ccg.keynorm`-normalized.

**Sources, at `EVAL_COMMIT`.** The code is exported with `git archive` into the custodian root, so
the user's tree is untouched.
1. A scripted pass over every `salesperson` screen and action, written by the gate runner from the
   UI, plus `salesperson/tests/e2e/`. Those requests are issued by application code, so they are in
   scope.
   - The exported SPA's API client records the first frame in `src/api/endpoints.ts`, or the first
     non-wrapper frame, mapped through source maps.
2. Non-test entry points that call the HTTP API or MCP tools.
   - Python callers are attributed through an HTTP-client shim to function grain.
   - Shell callers are attributed at **file grain**: a PATH-shimmed `curl` records the parent script
     from `/proc/$PPID/cmdline`, and the item is the `<script>` symbol.
   - A script that needs live services runs against the throwaway instance or is listed as
     unrunnable, never skipped silently.
3. `falkor-chat`'s suites under the same shim. Their `TestClient` edges are test code and are
   reported as a class outside the denominator.

**Recall and misses.**
- **Oracle recall** = observed in-scope edges present in the gate snapshot as a `Link` ÷ observed
  in-scope edges, with Wilson over edges and over distinct keys.
- It is reported beside seeded recall, never pooled (ML §4.1).
- A frame that cannot be attributed is counted as `unattributed`, never guessed.
- **Misses** are triaged into pack, extractor or dynamic. A confirmed miss that is not already an
  item goes to the next gate's candidate list (§6.4). The report carries counts by class only.

**Fence and record.**
- The fence is as P1-47 states.
- **Record `ccg-oracle-v1`**: `EVAL_COMMIT`, the session script, instrumentation and shim sha256s,
  and the suite list. It is committed before the capture.

### 6.6 Flow-fallback labelling (decided)

**The rule.** The stakeholder decided on 2026-11-09 that, in a flag-false unit, **only the
flow-dependent call-continuation hops become `possible`** (`flow-unavailable:<lang>`). These are
`CALLS ←`, `CALLS_NAME ←` and `TO ←` continuing from a `Symbol`, and §3.5's added rows if they ship.
Contract-level and containment hops keep their own status. answer-spec2 is revised to match.

**Method conditions:**
- The labelled set is a closed list of rows (edge type × direction × to-state) in the walk
  definition, not a function of a state name.
- The change bumps `algorithm.version` (§6.1).

**What this keeps measurable at P1:**
- AC-5's confirmed precision ≥ 0.95, on contract and containment items;
- AC-6's planted statuses;
- the edge-level reason strata of §5.

Its cost is a real gate risk on literal-key links, whose precision on the proving repo is unmeasured.

## 7. Question 6 — open P0 items, before any fresh key-writing (P1-43)

### 7.1 Scope overlap

**Why it matters.** Items catchable from two briefs are more catchable. That heterogeneity biases
capture–recapture N* low, so completeness is overstated.

**Procedure** (custodian, counts only). Each tag mismatch is classed as one of:
- `precedence`: the item also satisfies its briefed stratum's rule, and a higher EP §3.1 rule won.
  This is the rule working as designed;
- `out-of-brief`: it does not, which is real overlap.

Per stratum, completeness is reported twice: the headline, and on in-brief items only.

**Decision.**
- If every stratum's in-brief completeness stays on the headline's side of 0.95, nothing changes.
- If a stratum crosses below 0.95 only on in-brief items, the lower figure is reported and EP §6.5's
  trigger is evaluated on it.
- P1-43's two writers get identical per-stratum briefs.

### 7.2 Catch-all clause

P1-43 re-runs EP Appendix B's patterns at its new `EVAL_COMMIT` and records the count in its brief
(F5: 0 today). If the count is above 0, B.0's `{}` sentence gains this clause, and the briefs are
re-pinned:

> A path parameter that captures the rest of the path — written in the source as `{name:path}`,
> `{*name}`, `[...name]`, `*` or `**` — is written `{**}`. Every other path parameter is `{}`.

A mount-level fallback is not a route parameter. Whether it serves a request is a linker question
(P1-17).

## 8. Pre-registration and decision records

Gate records go to `cpg/eval/preregistration.jsonl`. Decision records go to
`cpg/eval/decisions/<name>.json`, outside `coreTreeHash`. Each is committed before the act it governs.

| Record | Governs | Before | Fixes |
|---|---|---|---|
| `ccg-p1-indexer-v1` | P1-15 | the offline run | §2.3 |
| `ccg-p1-flow-fallback-v1` | P1-39 | the measurement | §3.5's walks and rule |
| `ccg-p1-flow-python-v1` (+ `-decision`) | P1-34 step 3, the D2 row | labelling; the registry edit | §3.3, P0-ML §8.4 |
| `ccg-oracle-v1` | P1-47 | the capture | §6.5 |
| `ccg-prereg-ac24-v1` | P1-35 | the held-out AC-24 run | §4.3 |
| `ccg-prereg-v1` / `ac5-gate-1` | P1-37 | the gate build (freeze) | EP §12.1, plus §6.1 |
| `ccg-noise-ratify-v1` | the next gate's noise bars | that gate's record | §5.3 |

The P1-34 entry value is fixed in §3.1.

## 9. Stakeholder decisions and label custody

No stakeholder question remains open.

- **S36 spot-check (decided 2026-11-10: redo the existing sample).** The stakeholder redoes EP
  §7.3's existing 40-item sample, with its escalation triggers, before the `ac5-gate-1` record.
  - P0's check was done by an AI analyst, a disclosed deviation from S36. EP §12.1 separately
    requires the stakeholder to confirm every `key-error`.
  - The stakeholder is not a builder, so reading these items creates no contact.
  - **Procedure.** The sample was drawn over the union before the split, so its items now sit in
    dev, held-out and reserve.
    - An item marked incorrect is corrected or removed under EP §7.4, and the escalation trigger
      applies as written.
    - A held-out or reserve change means the custodian re-records and commits the held-out manifest
      (`split_manifest.py manifest`, then `verify`) **before** the `ac5-gate-1` record.
    - **Shortfall.** If a removal takes a stratum below its floor of 8 (`wiring` is at 8), a reserve
      cluster of that stratum moves to held-out, and the manifest is re-recorded. Reserve items are
      held-out and never exposed (EP §9.1 step 6).
      - **Cost:** each counting item moved leaves the reserve. That shrinks fill source 1 for
        P1-43's ≥ 35 fresh items (EP §12.1), so the re-gate leans more on the census top-up.
      - Dev cannot top up, because builders have used it.
      - With no reserve cluster in that stratum, the stratum is reported below floor, not gated (as
        EP §9.1 step 4 treats a marked stratum).
    - Counts by stratum and class go into the gate report.
- **Noise thresholds (decided 2026-11-10):** §5.3.
- **Flow-fallback labelling (decided 2026-11-09):** §6.6.
- **AC-25's control half** is decided by AC-25 and FR-8 (§3.5). A waiver is needed only if the
  shipped walk leaves the guarded point unlisted.

## 10. Risks and step-text amendments

**Risks** are stated where they bite: §2.1, §3.1, F4 and §4.1.

**Step-text amendments, for `architect` and `teco`** (one line each):
- **P1-09:** the synthetic-cond transparency rule, its recognizers, goldens and mutant; behaviour (5)
  and `slice/test_projection_not_regressed.py` restated as subset modulo rewritten edges, plus the
  pair-level guard (§3.4); report the dev figure and the open fixable misses.
- **P1-15:** M0, M1 and M2; `env.json` with `--environment`; operator provisioning; the runner
  dependency (§2.1).
- **P1-28 / `-egress-spec3.md` §7.4 and P1-48's file list** (`architect`): the indexer entries,
  read-only mounts and flag lines, and their `offline-run.sh` implementation and sandbox tests (§2.1).
- **P1-22/P1-23:** the fallback walk and the labelled-row list (§3.5, §6.6).
- **P1-26:** the span-group draft, the resolver join rule and the named resolver tests (§4.2).
- **P1-32a:**
  - write the oracle candidates in the key-item schema, passed by `--heldout`, since `--heldout-dir`
    globs `ac5-*.jsonl` only (§6.4);
  - **own the §6.2 freeze-time check** as a verb: the path set, the P1-commit test, per-file
    `check-report`, exit 2/4 as failure, and the private reverse map. The custodian runs it before
    the record.
- **P1-32b:** the harness pin (§6.1); scope from `eval-manifest.yaml`, with an out-of-scope held-out
  item as `key-error` (§6.3); outputs under the custodian root (§6.4).
- **P1-34:** the seed prefix (F6), the frame exclusions and the entry value (§3.1, §3.3).
- **P1-35:** repeat 1 is scored; repeat-1 `cacheHit` stops the run; outputs under the custodian
  root (§4.3).
- **P1-36:** the scope filter, both burdens, the baseline and the status split (§5.1); `oracle.py`
  reads the custodian root and reports counts only.
- **P1-37:** three recall units and chain length (§6.1); oracle misses as counts by class.
- **P1-39:** the four-walk measurement on the `EVAL_COMMIT` snapshot (§3.5).
- **P1-43:** candidates read from the custodian root; the §7 checks in the briefs.
- **P1-47:** the custodian-root path and the shell attribution (§6.4, §6.5); drop "mode 0700" as the
  protection.
- **Index (`architect`):**
  - §4.3 and Q-D: decided (§9);
  - §5 option 5 and §5's "method owner's and stakeholder's call" for the fallback: now ruled
    (§3.4, §3.5);
  - Q-B: decided by AC-25 and FR-8 (§3.5);
  - Q-C: decided (§5.3).
- **Proposal (`architect`):** "every item is shown as *possible*" and "call-level reach only"
  (`docs/plans/connected-code-graph.md`, "Flow availability") now contradict §6.6 and §3.5.
- **Requirements decision log (`tico`)**, one entry per question → answer:
  - 2026-11-09: which hops a flag-false unit makes `possible` → only the flow-dependent
    call-continuation hops (affects AC-5, AC-6; §6.6);
  - 2026-11-10: S36 spot-check for gate 1 → the stakeholder redoes the existing 40-item EP §7.3
    sample (affects S36, AC-5; §9);
  - 2026-11-10: noise thresholds → ratified as proposed, before the first measurement: Q ≤ 4,
    pp ≥ 0.20, p90 ≤ 100, ≥ 5×, applied from the next gate on (§5.3).
    - The entry **overturns S8** ("provisional until phase 1 measures it") and AC-5's clause
      "provisional; ratified or adjusted once the first phase has measured the real ratio".
    - `tico` rewords AC-5's parenthetical to match.
  - 2026-11-10: who provisions indexer dependencies → an operator step, outside the system (affects
    AC-8 and egress Q1 = C; §2.1).

## Appendix. Scripts behind F1–F2 (Python 3.12 stdlib, `python3 -I`)

`flow.py <frontend-fitness dir> <out.json>` reproduces F1's rows, the intra-method correlations and
the residual misses. `oc.py <out.json>` reproduces F2.

```python
# flow.py — F1: python flow scenarios, cluster interval (P0-ML §7.3), ANOVA ICC
import json, collections, sys
D, OUT = sys.argv[1], sys.argv[2]
m = collections.Counter()
for t in map(json.loads, open(D + '/truth/python.flow-final.jsonl')):
    m[t['ordinal']] += len(t['pairs'])
m[22] += 1                                   # the adjudicated omission, added as connected
M = [json.loads(l) for l in open(D + '/adjudication/python-flow-misses.adj.jsonl')]
cause = lambda x: x['proposedCause'].split(' ')[0]
miss = collections.defaultdict(list)
for x in M: miss[x['ordinal']].append(x)
ords = sorted(o for o in m if m[o] > 0); N = sum(m.values())
def scen(fixed):
    y = {o: m[o] - sum(1 for x in miss[o] if not fixed(x)) for o in ords}
    Y = sum(y.values()); R = Y / N; n = len(ords); mb = N / n
    V = sum((y[o] - R * m[o]) ** 2 for o in ords) / (n * (n - 1) * mb * mb)
    t = 2.048                                  # t(0.975, 28)
    return Y, R, (max(0, R - t * V ** .5), min(1, R + t * V ** .5)), y
R1 = lambda x: cause(x).startswith('projection:defsOf')
R3 = lambda x: cause(x).startswith('projection:synthetic')
UNV = lambda x: x['causeSide'] == 'projection' and not (R1(x) or R3(x))
FOR = lambda x: cause(x) in ('join:source:', 'join:sink:')
SYN = lambda x: x['causeSide'] == 'definition' and not x['path'].endswith('test_repository.py')
S = {'shipped': lambda x: False, 'R1+R3': lambda x: R1(x) or R3(x),
     '455': lambda x: R1(x) or UNV(x) or FOR(x) or R3(x),
     'ceiling': lambda x: x['causeSide'] in ('projection', 'join'),
     'ceiling+transparent': lambda x: x['causeSide'] in ('projection', 'join') or SYN(x)}
def icc(y):
    n = len(ords); p = 1 - sum(y.values()) / N
    msb = sum(m[o] * ((m[o] - y[o]) / m[o] - p) ** 2 for o in ords) / (n - 1)
    msw = sum((m[o] - y[o]) * (1 - (m[o] - y[o]) / m[o]) for o in ords) / (N - n)
    m0 = (N - sum(m[o] ** 2 for o in ords) / N) / (n - 1)
    return (msb - msw) / (msb + (m0 - 1) * msw)
for k, f in S.items():
    Y, R, ci, y = scen(f)
    print(f'{k:20s} {Y}/{N}={R:.4f} [{ci[0]:.3f},{ci[1]:.3f}] icc={icc(y):.3f}',
          [(o, m[o] - y[o]) for o in ords if m[o] > y[o]] if k.startswith('ceiling') else '')
json.dump([m[o] for o in ords], open(OUT, 'w'))
```

```python
# oc.py — F2: P(pooled >= 0.95) for a fresh draw with python's per-method pair counts
import json, random, sys
from math import comb, lgamma, exp
m = json.load(open(sys.argv[1]))
def per(n, r, rho):
    q = 1 - r
    if rho == 0: return [comb(n, k) * q**k * r**(n - k) for k in range(n + 1)]
    s = (1 - rho) / rho; a, b = q * s, r * s
    lb = lambda x, y: lgamma(x) + lgamma(y) - lgamma(x + y)
    return [exp(lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1) + lb(k + a, n - k + b) - lb(a, b))
            for k in range(n + 1)]
def conv(p, q):
    out = [0.0] * (len(p) + len(q) - 1)
    for i, x in enumerate(p):
        for j, y in enumerate(q): out[i + j] += x * y
    return out
def ppass(ms, r, rho):
    tot = [1.0]
    for n in ms: tot = conv(tot, per(n, r, rho))
    N = sum(ms); return sum(tot[:N - (-(-95 * N // 100)) + 1])
rng = random.Random('ccg-p1-ml-oc-v1')
for k in (8, 12, 30):
    for rho in (0.1, 0.2):
        draws = [[rng.choice(m) for _ in range(k)] for _ in range(200)]
        print(k, rho, sorted(sum(d) for d in draws)[100], sum(sum(d) < 120 for d in draws) / 200,
              [(r, round(sum(ppass(d, r, rho) for d in draws) / 200, 3))
               for r in (0.98, 0.975, 0.97, 0.965, 0.96, 0.95, 0.93, 0.90)])
```
