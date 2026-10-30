# Typed Decision Models — Implementation Plan

> **Status:** active · **Owner:** `architect` · **Tracks:** — · **Version:** 1.3 · **Reviews:** `docs/reviews/typed-decision-models.md` (requirements Passes 1–2; plan Passes 3–4)

*2026-10-30 — v1.3: answers plan review Passes 3–4 and folds the stakeholder's FR-4 reuse-count and FR-1 detectable-difference answers.*

## 1. Goal & scope

**Goal.** Let falkor-chat answer its typed yes/no and pick-one decisions with a fast local decision
model (Laya-class, served by LM Studio's decisions endpoint), with uncertain cases handed to a
local LLM. Before any of that is used in falkor-chat, model-bench must be able to measure the
combined configuration against the bar for its decision point. This plan turns
`docs/requirements/typed-decision-models.md` (FR-1…FR-15 with FR-11a, AC-1…AC-17) into sequenced
units.

**In scope.**
1. model-bench support for decision models: a `decisions` call surface, a hand-off arm that is one
   measured candidate, decision-point packs labeled by an LLM, the FR-1/FR-2/FR-10 bar lines, and
   two new roles.
2. A new component, `decision-models/`, that builds datasets and candidates: synthetic inputs,
   labels from a hosted frontier model, leakage-safe splits, and fitted hand-off cut-offs. It also
   hosts a fine-tuning spike that runs only if needed (FR-4).
3. falkor-chat runtime integration, one decision point at a time, each one gated on its own
   model-bench result: the guard-judge front (FR-14), the message-type label, and responder
   join-in (FR-12).

**Out of scope** (as in the requirements, plus these design-level exclusions):
- Auto-switching on a result. A deployment config change stays a human edit.
- Hosted decision models at runtime (FR-9). §3.9 shows that nothing here blocks them.
- Acting on a message's type, and entity matching.
- Re-running or re-versioning the existing `guard-judge-understanding` pack. It stays byte-identical
  and is reused, not edited.
- A per-workspace override for decision models. See §3.7.

## 2. Context & findings

CPG: used cpg_falkorchat — caller sweep of `evaluate_guard`, `GuardVerdict`, `_build_llm_judge`, `maybe_respond` and `_should_respond` to size the falkor-chat blast radius (built 2026-10-15 at `5c82170f`; taken as current).

### 2.1 model-bench (read in full where cited)

- **One call surface per pack, two values.** `CallSurface = Literal["chat", "embeddings"]` is
  declared twice, at `lmstudio.py:63` and `fingerprint.py:41`. `packs.derive_call_surface` turns
  `environment.requires` into exactly one of them. `runner._check_call_surface_cross_check`
  requires the catalog `type` to be `llm`/`vlm` for chat and `embeddings` for embeddings. All
  client routes are `/api/v0/*` (`chat`, `embed`, `warm_up`).
- **The fingerprint is profile-keyed.** `fingerprint.REQUIRED_BY_SCHEMA[1]` has three profiles
  (`model:chat`, `model:embeddings`, `deterministic`). Each profile's forbidden fields are a *set
  difference* over the other profiles. `ARM_KINDS`/`CALL_SURFACES` are derived from the profile
  names in `REQUIRED_BY_SCHEMA[1]`. `model-bench/AGENTS.md` has several load-bearing invariants
  here; any edit must re-read them: `ARM_KINDS` must never be derived from the forbidden mapping,
  the `""`/`None` sentinel split in `from_dict`, residency whole-key-set checks, and mutation-pinned
  constants.
- **The runner is single-model.** `run_pack` covers capture-order steps 0–10, one `warm_up`, one
  model's residency, and `_build_fingerprint` for one `ModelInfo`. `_drive_single_call_items`
  makes one call per item. `ItemTiming.calls` is already a tuple, and the tool-caller precedent
  holds several calls per item, so a two-call hand-off item fits the timing shape as it is.
  `RunResult.designEffect`/`basis` are required with no default (`results.py:695-701`).
- **The guard scorer transcribes falkor-chat as data.** `scoring/classification.py` ports
  `app._render_judge_user`, `guards._recent_turns` and `llm.extract_own_line_json_object` verbatim
  (FR-23: no import). `score_item` declares exactly one scoreable metric per item from its `tier`.
  `aggregate` derives the per-path (`understanding`/`turns`) rates from `detail.path`
  (`classification.py:232-276`) as exploratory metrics. Its hand-off parse is
  `bool(parsed.get("decision"))`, which is looser than falkor-chat's `_coerce_verdict`.
- **Roles are a closed table.** `roles.py` has `ROLES`, `UNIT_KIND_BY_ROLE`,
  `MULTI_CALL_TURN_BY_ROLE` and `ANALYSIS_UNIT_FIELD_BY_ROLE`. There are five roles (FR-20a), and
  each table is mutation-pinned.
- **Statistics.** `stats.py` implements `docs/plans/small-model-benchmarking-ml.md` and no other
  source (AGENTS invariant). It has paired McNemar with Holm (`verdict`, `holm_steps`), Wilson
  intervals, and paired and mean bootstraps. `stats.percentile` is the package's only percentile
  estimator. `LatencyBlock` carries p50/p95/max but **no mean**, and FR-2/FR-10 need one.
- **"No pass/fail gate".** This is both SMB's out-of-scope "Hard pass/fail regression gating"
  (`docs/requirements/small-model-benchmarking.md`) and a README non-feature. FR-7 requires that
  "the bar is established there". §3.6 reconciles the two.
- **Stored history.** `results/index.csv` rows 47–48 are the incumbent `qwen/qwen3-4b-2507` on the
  85-item pack: false advance 4/40, false suspend 4/30, p50 653/812 ms.

### 2.2 falkor-chat

- **Guard judge.** `guards.evaluate_guard` calls an injected `judge` and passes the result through
  `_coerce_verdict`. That function biases to suspend: it accepts only a strict `True`, and treats
  any un-negated `_NEGATION_CUES` phrase in the rationale as a contradiction. It sets `tier` on
  `GuardVerdict(decision, rationale, tier)`. Production builds `app._LlmGuardJudge` (resolving
  kind `guard` through `ModelGateway`, `accepts_run = True`) via `app._build_llm_judge(models)`.
  That function has 11 call sites: 1 in production and 10 in tests or eval (CPG sweep). So this
  plan **adds** a builder rather than changing that one.
- **Trace.** `WorkflowExecutor._trace_step` runs inside the drive loop right after `_record`
  (`executor.py:725-726`), with no isolation, so a raise there escapes `_drive`. It records each
  guard verdict as kind `guard_judgment` with payload `"{label} -> {decision}[ [{tier}]]:
  {rationale}"` (`executor.py:1543-1550`).
- **Responder.** `responder.AgentResponder._should_respond`: if `role == "assistant"` it returns
  False, otherwise it returns whether the agent is in `mentions` (the "locked, DECISION 2"
  docstring). `maybe_respond` has exactly two production callers, `trigger.maybe_trigger` and
  `background._safe_respond`. `api.post_message` and `mcp` schedule one of those for **every**
  posted message, and the responder self-decides, so join-in needs no change to scheduling. With a
  workflow trigger wired, `trigger.maybe_trigger` step 4 calls the responder only for messages that
  no workflow claimed. With the storefront fall-through off, the responder is unreachable, so
  join-in is inert there.
- **Model config.** `modelconfig.KINDS = {agent, step, embedding, guard}`, and each has a
  `WorkspaceConfig` hard-cap override. `ModelGateway._resolve_element(ref, kind=…)` resolves a
  direct `provider/model` ref and uses `kind` only for the timeout lookup. Non-`KINDS` kinds
  already exist (`extraction`). `Overlay._KNOWN_KEYS` accepts unknown top-level keys and logs
  them. `test_modelconfig.py:1193` `_FORBIDDEN_NAMES` is the FR-4 AST rule that only
  `modelconfig` may construct provider clients.
- **Message write.** The QUERIES.md §6 "Set a message embedding" shape
  (`MATCH (m:Message {msgId:$msgId}) SET …`, `repository.py:811`) is the precedent for an
  out-of-band property write after posting. Embedding is scheduled by `background._safe_embed`
  from both REST and MCP. The storefront post path schedules none of these. DESIGN §5 states the
  `Message` property schema.

### 2.3 LM Studio decisions endpoint

LM Studio documents `/v1/decisions` (`https://lmstudio.ai/docs/developer/openai-compat/decisions`,
`client.decisions.create`), but its page shows only a predicate request and `answers`. The full
contract below comes from `tico`'s live probe on 2026-10-30 (`ws:agent-team` document
`316050e8…`), which POSTed to `http://localhost:1234/v1/decisions` with model `laya`. Latency was
about 2.5 s cold and about 20 ms warm.

- Request: `{model, input, questions:[{type: predicate|choice|score, name, instructions,
  choices:[{value, instructions}]}]}`. Bare-string choices are rejected.
- Response: `{model, answers:[…], usage:{input_tokens, output_tokens, input_tokens_details,
  output_tokens_details, total_tokens}}`, where `output_tokens` is always 0.
  - A predicate answer is `{type, name, probability}`.
  - A choice answer is `{type, name, choice, probabilities:[{value, probability}], confidence}`.
  - There are **no `stats`, `model_info` or `runtime` fields**.
- Determinism: as recorded in the requirements' Problem section. U0 Q6 consumes it (§3.3).

I also checked the catalog with a read-only GET. `/api/v0/models` lists `laya` as
`{"type":"decision","arch":"modern-bert","compatibility_type":"gguf","quantization":"Q8_0",
"publisher":"ggml-org","max_context_length":8192}` with **no `capabilities` key**.

`decide` uses the verified `/v1/decisions`. Whether an `/api/v0/decisions` variant exists is the
one unverified point, and it is settled by U1a.

## 3. Design & rationale

### 3.1 Shape of the solution

```mermaid
flowchart LR
  DM[decision-models/<br/>generate · label · split · fit] -- eval split + side check<br/>(importer, one-way copy) --> P[model-bench pack]
  DM -- candidate config + fit manifest --> C[model-bench candidates/]
  P --> R[model-bench run<br/>chat arm = incumbent / hand-off target<br/>candidate arm = decision model ± hand-off]
  C --> R
  R --> REP[compare report:<br/>FR-1 parity / FR-2 speed-up / FR-10 bar,<br/>labeler caveat, leakage + reuse lines, side check]
  REP -. human edits config/models.json .-> FC[falkor-chat decisionPoints.*<br/>enabled per point]
```

`decision-models/` builds candidates and data, model-bench measures them, and falkor-chat only
reads its own config. There are no runtime edges between the three components. That preserves
FR-23 (model-bench reads nothing outside itself) and falkor-chat's isolation.

**Rejected: put generation and fitting in model-bench.** model-bench has zero runtime
dependencies, is local-only and is human-started. A hosted-API labeler, and later torch-based
fine-tuning, would break its install-forever reproducibility rule.

**Rejected: generate inside falkor-chat.** FR-8 needs a component that structurally cannot read a
falkor-chat graph. A separate component with an import allowlist makes that a test, not a promise.

### 3.2 The decision-point contract: data, written down once

Each decision point is defined by four things. All four are **data**, and both model-bench and
falkor-chat consume them byte-for-byte:

| Point (model-bench role) | Answer space | Decision-model input (the rendered string) | Hand-off chat prompt | Hand-off parse |
|---|---|---|---|---|
| guard judge (`guard-judge`) | predicate `advance` | `_render_judge_user(condition, understanding, normalized_turns)`, exactly as today | today's `_JUDGE_SYSTEM_PROMPT` + the same user text | `extract_own_line_json_object(require_key="decision")` then `_coerce_verdict` (strict `True`, rationale contradiction). Unparseable means suspend. |
| join-in (`join-in-judge`) | predicate `reply` | `render_join_in_input` (§3.2.1) | `prompts/join-in-handoff.md` (new) | `require_key="reply"`, strict `True`; anything else means no reply |
| message type (`message-type-classifier`) | choice over `order, question, complaint, escalation, other` (FR-10; small talk is `other`) | `render_message_type_input` (§3.2.1) | `prompts/message-type-handoff.md` (new) | `require_key="type"`, and the value must be in the label set; otherwise there is no label |

#### 3.2.1 The two new renderers (canonical text)

- `render_join_in_input(agent_name, prior_turns, new_message)` produces:
  `AGENT: {agent_name}\n\nRECENT TURNS (context only):\n{speaker}: {text}\n…\n\nNEW MESSAGE:\n{speaker}: {text}`.
  `prior_turns` are the thread rows strictly before the new message's position in the
  chronological thread read (not "the thread minus the new message", because concurrent posts may
  follow it). They are normalized by `guards._recent_turns`' rule: filter first, then keep the last
  6, with each turn cut to 400 chars. The block is omitted when there are no prior turns. The new
  message is also cut to 400 chars.
- `render_message_type_input(text)` produces `MESSAGE:\n{text[:2000]}`. The default is the
  message only, with no context. U0 (Q1) may add the preceding turn. If it does, the change is made
  here, once, before U4.

The **question spec** — `{type, name, instructions, choices}` as sent in `questions[]` — belongs
to a **candidate** rather than to the pack, because its instruction text is a parameter that is
tuned on the calibration split (§3.4). The pack fixes only the answer space. The validator checks
that the candidate's spec matches it: same type, and for a choice, the same set of `value`s as the
pack's labels.

#### 3.2.2 How the copies are kept equal

The renderers, hand-off prompts, hand-off parses, `apply_policy` and `input_hash` each exist in up
to three places:
- falkor-chat (runtime);
- model-bench (`scoring/*` and `decisions.py`, transcribed under FR-23 as `classification.py`
  already is);
- decision-models (`render.py`, `fit.py`, `hashing.py`).

**Rejected: one shared library.** It would give model-bench a runtime dependency, which FR-23
forbids.

Instead, **model-bench owns canonical golden vectors**: input → expected output files under
`model-bench/tests/fixtures/decision-contract/`, one file per function family:
- `render-guard.jsonl`, `render-join-in.jsonl` and `render-message-type.jsonl`;
- `handoff-parse-*.jsonl`;
- `apply-policy.jsonl`, which includes `p == l`, `p == h`, `l == h`, `confidence == c` and `c == 0`;
- `input-hash.jsonl`. Its items carry non-input fields (`itemId`, `tier`, `expected`, `stratum`),
  so a widened field table changes the hash and reddens.

**Expected outputs are never produced by the code they pin.** They are derived as follows:
- policy rows, by hand from §3.3's inequalities;
- join-in and message-type render strings, by hand from §3.2.1's canonical text;
- `render-guard.jsonl` and `handoff-parse-guard.jsonl`, by running falkor-chat's real
  `_render_judge_user` and `_coerce_verdict`, which are the source of truth;
- `input-hash.jsonl` hashes, by computing the §3.5 formula by hand over the canonical JSON, not by
  calling `input_hash`.

Each component's own suite asserts its implementation against a byte-copy of these files. A
guarded equality test asserts each copy is byte-identical to model-bench's: in decision-models it
skips when `model-bench/` is absent, and in falkor-chat it skips when `model-bench/` is absent.

**Sequencing, so that nothing is measured before it is pinned:**
- The vectors land before any fitting:
  - U1c lands the policy vectors and the `guard-judge` hash rows;
  - U4 lands the new renderers, parses and the two new roles' hash rows.

  decision-models asserts against them in U5/U6, before U6 fits.
- falkor-chat's existing `_render_judge_user` gets its vector test in U3b, before U7 measures.
- The falkor-chat join-in and message-type renderers and parses are built in U9 against the same
  vectors. They therefore cannot drift from what was measured, even though they are written after
  it.

### 3.3 model-bench: two new call surfaces, one candidate arm

**Surfaces and profiles.**
- `fingerprint.CallSurface` gains `"decisions"` and `"handoff"`. `"handoff"` is a composite arm,
  not an LM Studio route, so `lmstudio.CallSurface` gains only `"decisions"`.
- There are two new fingerprint profiles, `model:decisions` and `model:handoff`, added to
  **`REQUIRED_BY_SCHEMA[1]`**:
  - `model:decisions` has the `model:embeddings` field set (model identity, residency, pack/tool
    identity and attested fields; no runtime or sampling fields), **plus** `candidateId`,
    `candidateContentHash`, `questionSpecHash`, `decisionPolicy` (the policy object),
    `fitManifestHash`/`fitManifestSize` and `evalSetHash` (§3.5). There is no runtime field, because the response
    has no `runtime` object (§2.3).
  - `model:handoff` has the `model:decisions` fields plus the hand-off model's identity, all
    `handoff`-prefixed and flat: `handoffModelKey`, `handoffModelPublisher`, `handoffArch`,
    `handoffQuantization`, `handoffCompatibilityType`, `handoffMaxContextLength`,
    `handoffLoadedContextLength`, `handoffModelType`, `handoffRuntimeName`, `handoffRuntimeVersion`,
    `handoffTemperature`, `handoffMaxTokens`. These are flat fields, not a nested block, so the
    existing tier/forbidden machinery applies unchanged.

*Why schema 1 rather than a schema 2 bump:* the change is purely additive. No existing profile's
required set changes. The new field names are new, so the derived forbidden sets of the old
profiles grow only by names that no stored record carries. All 95 stored records still validate
(confirmed by review Pass 3's in-process probe). A schema-2 bump would instead force
`FORBIDDEN_BY_ARM_PROFILE`, `ARM_KINDS` and `CALL_SURFACES` (all computed over schema 1 today) to
become per-schema. `armKind` stays two-valued (`model` for both new profiles), so `ARM_KINDS` is
unchanged.

**Pack surfaces.** `environment.requires` may now hold exactly one of
`lmstudio-chat`/`lmstudio-embeddings` as today, **optionally plus** `lmstudio-decisions`. A run's
surface is chosen by its arm, not by the pack alone:

| CLI | Arm | Surface | Refusals (exit 4) |
|---|---|---|---|
| `run --pack P --model K` | chat or embeddings (incumbent / hand-off model alone) | the pack's chat/embeddings surface (unchanged) | unchanged |
| `run --pack P --candidate ID`, candidate with no `handoffModel` | decision model alone | `decisions` | pack has no `decisionPoint` block or no `lmstudio-decisions`; catalog `type != "decision"` |
| `run --pack P --candidate ID`, candidate with `handoffModel` | hand-off composite | `handoff` | as above, plus: pack lacks `lmstudio-chat`; hand-off model `type not in {llm, vlm}`; on a speed-up pack, `handoffModel != decisionPoint.incumbent` (FR-3: the rest goes to the existing judge) |

`--model` and `--candidate` are mutually exclusive. `derive_call_surface` keeps its name and
contract for the chat/embeddings token. A new `derive_decision_support(requires) -> bool` covers
the extra token. Existing packs are untouched.

**The candidate file.** `model-bench/candidates/<candidateId>.json` is versioned, and its content
hash is recorded in the fingerprint:

```json
{
  "candidateId": "guard-laya-band-v1",
  "role": "guard-judge",
  "decisionModel": "laya",
  "question": {"type": "predicate", "name": "advance", "instructions": "…"},
  "policy": {"kind": "predicate-band", "yesAtOrAbove": 0.92, "noBelow": 0.08},
  "handoffModel": "qwen/qwen3-4b-2507",
  "fitManifest": {"file": "guard-laya-band-v1.fit.txt", "sha256": "…", "size": 1400,
                  "purposes": ["question-tuning", "cutoff"]},
  "provenance": {"builtBy": "decision-models", "datasetVersion": "…", "sourceSha": "…"}
}
```

Model keys in a candidate are LM Studio catalog ids.

**Policies** live in `modelbench/decisions.py`. These are pure functions, pinned by
`apply-policy.jsonl` (§3.2.2):
- `predicate-band {yesAtOrAbove: h, noBelow: l}` with `0 ≤ l ≤ h ≤ 1`. The answer is *yes* iff
  `p ≥ h`, *no* iff `p < l`, and *hand-off* iff `l ≤ p < h`. `l == h` means an empty band (no
  hand-off), which is required when `handoffModel` is null.
- `choice-confidence {minConfidence: c}` returns `choice` iff `confidence ≥ c`, otherwise hand-off.
  `c == 0` means no hand-off.

**Per-item flow, `handoff` surface** (in the new `runner._drive_decision_items`):

1. Start the item clock.
2. Make the decisions call (timed `CallTiming`, with only `wallClockMs` set).
3. Apply the policy.
4. If the policy says hand off, make the chat call using the scorer's `build_messages` (timed
   `CallTiming` with the chat stats).
5. Parse the hand-off answer with the scorer's `answer_from_chat`, which is the §3.2 parse.
6. Stop the item clock.

The result is `ItemTiming(wallClockMs=both calls end to end, calls=(decision[, chat]))`.
`ItemResult.detail` gains `decidedBy ∈ {"decision-model","handoff"}` and `probability` (a
predicate's p, or a choice's confidence). That is a probability, not reply text, so the
"no raw reply persisted" invariant holds. A decision-call timeout or failure withholds the item
(`fail`/`unrunnable`) exactly as a failed chat call does today, and it is counted and printed.
**It is not handed off.** The bench measures the measured path; production's fallback on error is a
robustness path, handled in §3.7.

**Capture order with two models:**
- Catalog lookups and cross-checks for both models.
- Both are warmed up. The decisions warm-up is one `decide` call with a fixed tiny input and the
  candidate's own question.
- The between-item residency probe must show **both** resident. Otherwise the item is
  `withheldFor="load"`, which is counted, never silently dropped. U0 Q4 decides how FR-2 treats
  such items.
- The attestation trip-wire is fed by the chat warm-up of the hand-off model (`"compared"`). For
  `decisions` alone it is `"unavailable"`, as for embeddings.

**Design effect and basis.** A run's residency withholding keeps every scored item inside one load
of each model. Within one load, Laya repeats exactly (§2.3). So candidate runs record
`designEffect=1.0, basis="by-construction"`, as every item-level role does today. This is
**provisional until U0 Q6 rules**, since two loads differ slightly. Q6 decides whether a bar claim
needs replicate runs across loads. U7 is blocked on Q6.

**`latency_block`** treats `decisions` and `handoff` like `embeddings` for the stats-derived
fields (`statsCoveredCount=None` and so on), because a mixed-surface median of chat-only stats
would mislead. Wall-clock percentiles apply unchanged. The mean that FR-2/FR-10 need lives in
`decision_stats.py` (§3.6), not in `LatencyBlock`, so the nine `LatencyBlock` invariants are
untouched.

**Scorer seam.** `ItemScorer` gains three optional hooks, `getattr`-guarded as `prime`/
`build_messages` already are:
- `decision_input(item_input, *, pack) -> str` — the §3.2 renderer.
- `answer_from_chat(result, *, pack) -> tuple[Answer, Outcome]` — the role's chat parse.
- `score_answer(item_input, answer, outcome, timing, *, pack, decided_by, probability) -> ItemResult`.

**Arm identity.** `RunResult.armKey` returns the fingerprint's `candidateId` if present, otherwise
`modelKey`. `cli._select_arms --models`, `models --tested` and `index.csv` key on `armKey`.
`index.csv` gains `candidateId` and `handoffModelKey` columns, empty for chat arms. Run ids use
`armKey`.

### 3.4 decision-models/: data and candidates (FR-4, FR-5, FR-6, FR-8)

`decision-models/` is a new component. It is Python 3.12 with a stdlib core and a host venv
(`setup.sh`), and it follows `model-bench`'s and `mcp-monitor`'s shape. A `[train]` extra is
added only if U8 goes ahead.

```
decision-models/
  README.md · AGENTS.md · pyproject.toml · setup.sh
  decisionmodels/
    labeler.py      # hosted frontier client (Anthropic Messages API over urllib); key from env only
    points/         # guard_judge.py · join_in_judge.py · message_type_classifier.py — templates + input fields
    generate.py     # scenario seeds → synthetic inputs → blind labeling pass (U0 Q1)
    hashing.py      # input_hash(item, role) — the leakage key (§3.5)
    split.py        # train / calib / eval by scenario cluster; asserts input-hash disjointness
    lmstudio.py     # minimal decisions + chat client (for fitting only)
    fit.py          # question-instruction + policy fitting on calib → candidate + fit manifest
    render.py       # the §3.2 renderers and parses, a vector-pinned copy (§3.2.2)
  data/<role>/<datasetVersion>/   # items.jsonl, splits.json, generation.json (committed)
  docs/BACKLOG.md · docs/HISTORY.md
```

- **Labeler (FR-5, FR-6).** The provider and model are pinned in `generation.json`, along with the
  prompt hashes. `ANTHROPIC_API_KEY` is read from the environment and never written anywhere.
  `generation.json` records `{provider, model, promptSha256, generatedAt, itemCount}`.
- **FR-8 by construction.** A test enforces an AST import allowlist (stdlib + `decisionmodels`) and
  a source scan: no `falkordb`/`redis` import and no path containing `falkor-chat`. Every item
  carries `origin: "synthetic"` and the hash of the generator call that produced it. Generation
  never uses the 85 human-curated side-check items as seeds or examples, so a paraphrase of one
  cannot reach evaluation through the exact-hash check (U0 Q1 states this as a rule).
- **Splits.** Train, calibration and evaluation are disjoint by **scenario cluster**, not by item,
  so paraphrase leakage is structurally reduced. Exact-input disjointness is asserted. The cluster
  definition and the proportions come from U0 Q1.
- **Fitting.** `fit.py` runs the decision model (and the hand-off LLM, if the objective needs its
  answers) on the **calibration split only**. It chooses the question instructions and the policy
  thresholds against the U0 Q2 objective, then writes the candidate JSON plus a fit manifest. The
  manifest is the sorted list of `input_hash` values for every item used for training, calibration
  or question tuning.
- **Re-tuning after an evaluation result** is allowed on the same evaluation set (FR-4). Its cost
  is made visible by the reuse line in §3.5, not prevented.
- **Recalibration is not a separate artifact for predicate points.** Platt or temperature scaling
  is a monotone map of p, so thresholds on calibrated p are exactly thresholds on raw p. The band
  `(l, h)` fitted on raw p is therefore the whole decision-relevant calibration. This does **not**
  hold for choice confidence: temperature scaling can reorder max-probabilities across inputs. U0
  Q2 rules whether message-type needs a calibration map. If it does, the map is a policy field,
  vector-pinned like the policy.
- **Importer** (model-bench side, one-way, human-invoked; mirrors `scripts/refresh_golden.py`):
  `model-bench/scripts/import_decision_pack.py`.
  - `--role <r> --dataset <dir> --pack-version <v>` copies the **eval split only** into
    `packs/<r>-synthetic/items.jsonl` and writes `PROVENANCE.md` (source git SHA, file SHA-256,
    labeler block).
  - `--candidate <file>` copies a candidate and its fit manifest into `model-bench/candidates/`.

  Re-import under an unchanged `packVersion` is refused, as in `refresh_golden.py`.

### 3.5 Honesty mechanics in model-bench (FR-4, FR-5, FR-8, FR-11, FR-13, FR-15)

**Pack manifest additions.** A pack is a **decision-point pack** iff it has a `decisionPoint` block,
and every refusal below keys on that block's presence, never on an optional field inside it:

```json
"decisionPoint": {
  "answer": {"type": "predicate"} | {"type": "choice", "labels": ["…"]},
  "incumbent": "qwen/qwen3-4b-2507" | null
},
"labels": {"source": "llm-unreviewed", "labeler": {"provider": "anthropic", "model": "…"},
           "inputs": "synthetic"}
```

**Bar kind comes from the role, through a pinned table.**
`BAR_KIND_BY_ROLE = {guard-judge: "speed-up", join-in-judge: "new-ability",
message-type-classifier: "new-ability"}`. The `incumbent` field must agree with it: non-null on a
speed-up role, null on a new-ability role. A mismatch refuses. A pack cannot pick its own bar.

**Thresholds are requirement constants in code, not pack data.** These live in
`modelbench/decisions.py`, each citing its FR:
- `MIN_EVAL_ITEMS = 200` (FR-11);
- `FR2_MIN_SPEEDUP = 2.0`;
- `FR10_PER_ANSWER_AGREEMENT = 0.90`;
- `FR10_MEAN_LATENCY_MS_MAX = 350`.

A pack cannot lower them. They and `BAR_KIND_BY_ROLE` are mutation-pinned per the AGENTS.md
constant rule.

**`validate_pack` refusals** (on every decision-point pack). Each refusal is mutation-tested by
deleting the key it reads; the refusal must still fire:
- the `labels` block is absent, or `labels.source != "llm-unreviewed"`;
- `stratum` is absent on any item (no default), or not in `{eval, side-check}`;
- fewer than `MIN_EVAL_ITEMS` items in stratum `eval` (AC-13);
- `labels.labeler.model` is equal to `decisionPoint.incumbent` (AC-4);
- `labels.inputs != "synthetic"`, or any `eval` item whose `origin != "synthetic"` (AC-7). The
  `side-check` stratum is exempt by stratum, since those items are human-curated;
- `incumbent` disagrees with `BAR_KIND_BY_ROLE`;
- a `guard-judge` pack with no `side-check` stratum (AC-17); a `side-check` stratum on any other
  role.

**Declared outcomes, not report filtering.** `report.py` infers nothing (AGENTS invariant), so each
item declares its own metrics in `scoreable`/`counts`:
- **Guard pack.** An `eval` item declares its per-kind verdict metric
  (`falseAdvanceRateByUnderstanding`, `falseAdvanceRateByTurns`, `falseSuspendRateByUnderstanding`
  or `falseSuspendRateByTurns`, chosen from its own label and `path`). It also declares the pooled
  exploratory metric. A `side-check` item declares only `sideCheck*` metrics. Those keep the old
  pack's tier mapping (`clear_suspend`, `clear_advance`, `boundary`).
- **New-ability packs.** Each `eval` item declares the agreement metric for its own true answer
  (§3.6).
- **The bar section prints the scored n beside FR-11's 200** whenever items were withheld.

**Leakage refusal (AC-3) at run time.**
- The candidate's fit manifest is intersected with the `input_hash` of every item in the pack (eval
  and side-check strata). A non-empty overlap refuses with exit 4.
- The fingerprint stores `fitManifestHash` and `fitManifestSize`, and the report prints:
  `fit manifest: N inputs (training/calibration/cut-off); 0 overlap with this pack's M items (checked at run time, sha256 over canonical input fields)`.
- `input_hash(item, role)` = sha256 of
  `json.dumps({k: item[k] for k in INPUT_FIELDS_BY_ROLE[role] if k in item}, sort_keys=True, separators=(",",":"), ensure_ascii=False)`.
- `INPUT_FIELDS_BY_ROLE` is keyed by model-bench role in both components and pinned by
  `input-hash.jsonl` (§3.2.2):
  - `guard-judge`: `condition, understanding, turns, path`;
  - `join-in-judge`: `agentName, priorTurns, newMessage`;
  - `message-type-classifier`: `text` (plus `priorTurn` if U0 adds it).

**Evaluation-set reuse (FR-4).** The evaluation set's identity is its content, not the pack
version:
- `evalSetHash` is the sha256 of the sorted lines `input_hash(item) + ":" + json.dumps(item["expected"], sort_keys=True)`,
  taken over the `eval` stratum.
- `run` computes it and stores it in every candidate fingerprint. A re-import under a new
  `packVersion` with the same items and labels therefore keeps the same hash.

For each candidate arm in a report, the bar section prints, beside that arm's label:
`Candidates already run on this evaluation set: K (<candidateContentHash8 list>)`.
- **Which run.** "This run" is that arm's own run record. In a multi-arm `compare`, each candidate
  arm gets its own K, and chat arms get none.
- **What K counts.** The number of distinct `candidateContentHash` values among stored candidate
  runs of the same `packId` that have the same `evalSetHash` and a `startedAt` strictly earlier
  than this run's, excluding this run's own `candidateContentHash`.
- **Why the content hash.** A candidate file edited in place under the same `candidateId` is
  therefore counted as a new candidate.
- It is a within-pack `load_history(packId)` read across all of the pack's versions, so it stays
  FR-23-clean.

**Labeler caveat (AC-4).** Every report on a decision-point pack prints this as its first result
line:
`Labels: <provider>/<model>, unreviewed (FR-5). Results measure agreement with this labeler on synthetic inputs, not verified correctness.`
The amendment to SMB FR-19 has its single home in the requirements doc's FR-5. model-bench's
README cites it and does not restate it.

**Side check (FR-15, AC-17).**
- The guard pack carries the 85 human-curated items in a `side-check` stratum, copied from the
  sibling pack `guard-judge-understanding` (same tiers, `labelSource: human`).
- They sit in the same pack because model-bench has no cross-pack API by design
  (`load_history(packId)`).
- Their metrics are exploratory, never `verdictMetrics`. The report prints them under
  `Side check — 85 human-curated items (does not affect pass/fail)`.
- They do not count toward FR-11. Their hashes are in the leakage check, so they can never be
  calibrated on.

**Per evidence kind (FR-13, AC-10).** The guard pack's `verdictMetrics` are the four per-kind rates
over the eval stratum. Pooled rates are exploratory. The verdict family, the test direction and
any floor are U0 Q3.

**Hand-off share (FR-3, AC-2).** The candidate arm's aggregates gain an exploratory
`BinaryMetric("handoffShare", handoffs, attempted, unit="item")` and per-path diagnostic error
rates (`…DecidedByModel`, `…HandedOff`). Quality and latency figures cover all items, so both paths
are included by construction.

### 3.6 The bar lines (FR-1, FR-2, FR-10, FR-11a) without becoming a gate

SMB rules out "hard pass/fail regression gating", and the README's non-feature says there is no
threshold and no non-zero exit for an unwelcome result. FR-7 needs model-bench to state whether a
point's requirement bar holds. They are reconciled like this: a report on a decision-point pack
prints **one bar section**. Exit codes are unchanged, nothing is written outside `reports/`, and
adoption stays a human edit. README non-feature 2 is reworded to: *no threshold of its own; a
decision-point pack's report states the requirement's bar and never acts on it.*

**What the bar section prints:**
- **Speed-up point:**
  - `FR-1 parity` — one line per verdict metric. *Holds* means not distinguishably worse.
    **Every `holds` line prints its detectable difference beside it** (FR-1), in the form
    `holds (could only detect ≥ N pp at n = …)`.
  - `FR-2 speed-up: <ratio> (interval)` — the incumbent's mean over the candidate's mean. Both arms
    must share the top-level `run.sessionId` (never `fingerprint.sessionId`, AGENTS invariant).
    Otherwise it prints `not assessable: arms from different sessions (FR-2)`.
- **New-ability point:** agreement within each true answer, each with its Wilson interval, plus the
  mean latency.
- **Both:** the scored n (§3.5) and the reuse line (§3.5).

**FR-11a: every bar is judged on the measured value**, with its interval printed beside it. One
function holds the comparison: `decisions.bar_holds(measured, threshold, *, direction)`.
`direction ∈ {"at-least", "at-most"}` is required with no default. The function never receives the
interval.

**Where the new statistics live.** The parity reading, the per-answer intervals and the
mean-latency estimator come from `docs/plans/typed-decision-models-ml.md`. They go in a new module,
`modelbench/decision_stats.py`, that implements **that note and no other source**, cited by
section. `stats.py` keeps its single-source invariant unchanged.
- `decision_stats` may call `stats` primitives (`wilson_interval`, `percentile`,
  `paired_bootstrap`) as building blocks. It never defines a second percentile estimator.
- Every argument that shapes a decision (`levels`, `clamp`, `seed`, `direction`, family size) is
  required with no default.
- U3c adds one `model-bench/AGENTS.md` invariant line naming `decision_stats.py` and its note.

**New roles (FR-20a extension).** The two new roles are `join-in-judge` and
`message-type-classifier`, both with unit kind `item`, single-call, and analysis unit `itemId`.
Each of the four role tables is extended together with its mutation pins, as is `BAR_KIND_BY_ROLE`.
Their scorer is a new `scoring/agreement.py`:
- predicate: two agreement metrics, `agreeWhenYes` and `agreeWhenNo`;
- choice: `agree[<label>]` for each of the five labels;
- all of them `BinaryMetric(unit="item")`.

The always-majority degenerate arm fails by construction: its minority-answer agreement is 0
(AC-15 has a dedicated test).

### 3.7 falkor-chat runtime

**Config, not a new model kind.** `config/models.json` gains a top-level `decisionPoints` block, and
`Overlay._KNOWN_KEYS` gains the key:

```json
"decisionPoints": {
  "guard":       {"enabled": false, "model": "lmstudio/laya", "question": {…}, "policy": {…},
                  "handoffModel": "lmstudio/qwen/qwen3-4b-2507", "timeoutSeconds": 5,
                  "benchResult": "<model-bench runId(s) that met the bar>"},
  "joinIn":      {"enabled": false, …, "agentName": "…"},
  "messageType": {"enabled": false, …, "labels": ["order", "question", "complaint", "escalation", "other"]}
}
```

- `question` and `policy` are copied verbatim from the measured candidate.
- `model` and `handoffModel` are the candidate's LM Studio catalog ids, each prefixed with the
  provider id whose baseURL is the LM Studio instance it was measured on (`laya` → `lmstudio/laya`).
- `timeoutSeconds` is required and is the decision call's **only** timeout source.
- `benchResult` must be non-empty when `enabled` is true. It is a documentary citation of FR-7 that
  is never read against model-bench (no cross-component runtime edge).

**Rejected: a decision `kind` with a workspace override.** It would let a workspace swap the
decision model alone, which produces an unmeasured composite.

**Clients.** `falkorchat/decisions.py` (new) holds:
- `DecisionClient`, an urllib POST to `{base_url}/decisions` (base URL from the provider catalog,
  which ends in `/v1` for LM Studio) via the existing `transport.make_http_transport`;
- `DecisionPointConfig` (the parsed block);
- `apply_policy`, the two §3.2.1 renderers and the hand-off parses, all vector-pinned (§3.2.2);
- `record_decision(point, decided_by, probability, reason=None)`, the one INFO log line
  `decision_point=… decided_by=… p=… reason=…`. All three points call it on every decision, so
  hand-off frequency is observable in production for each of them (FR-3).

`ModelGateway.decider(ref, *, timeout_s) -> DecisionClient` resolves the provider via
`_resolve_element(ref, kind="decision")` and passes the point's `timeoutSeconds` to the client. The
resolved model's own timeout is not used. `DecisionClient` joins `_FORBIDDEN_NAMES` in the FR-4
AST test.

**The measured-composite rule (applies to all three points).** At call time, the point resolves
its hand-off LLM through the gateway exactly as the incumbent would:
- kind `guard`, with the guard's own `model` if it declared one, for the guard point;
- kind `agent` for join-in and message type.

If the resolved primary ref differs from the configured `handoffModel` (a workspace hard-cap
override, or a guard that names its own model), the decision model is **bypassed** for that call:
- The guard falls back to today's LLM-only judge, traced as `llm-fallback`.
- Join-in and message type **fail closed** to today's behaviour: no unmentioned reply, no label.

The same applies when the decision call errors or times out. This keeps the workspace hard cap
intact. Because model-bench's guard hand-off parse ports `_coerce_verdict` (§3.2), the measured
hand-off path and the runtime one apply the same verdict rules.

**Guard front (FR-3, FR-14, AC-11).**
- `GuardVerdict` gains `probability: float | None = None` and `decided_by: str | None = None`.
  Defaults keep every positional call site byte-compatible.
- `evaluate_guard` copies `probability`/`decidedBy` from the raw judge mapping when they are
  well-typed (float in [0,1]; str in the closed set `{"decision-model","llm-handoff",
  "llm-fallback"}`), the same way it sets `tier`.
- The new `app._DecisionFrontedGuardJudge(llm_judge, decider, cfg, models)` has
  `accepts_run = True` and the same call signature. It:
  1. renders `_render_judge_user(...)`;
  2. calls `decide`;
  3. applies the policy;
  4. if the policy decides, returns `{"decision": bool, "rationale": "decision-model verdict",
     "probability": p, "decidedBy": "decision-model"}`;
  5. on hand-off, calls `llm_judge(...)` and annotates its mapping with `probability=p`,
     `decidedBy="llm-handoff"`;
  6. on bypass, error or timeout, calls `llm_judge(...)` and annotates it with
     `decidedBy="llm-fallback"` and no probability.

  The constant rationale contains none of `_NEGATION_CUES`, and a test pins that.
- `_build_default_app` calls a new `_build_guard_judge(models)`. It wraps `_build_llm_judge(models)`
  only when `decisionPoints.guard.enabled`. `_build_llm_judge` and its 10 test/eval callers are
  untouched.
- **Trace suffix.** The `guard_judgment` payload gains, after the tier segment, ` [{decided_by}]`
  when `decided_by` is set, and ` p={probability:.3f}` inside the brackets **only when `probability`
  is not `None`**. So `llm-fallback` renders `[llm-fallback]`. With no front configured, the payload
  is unchanged.

**Join-in (FR-12, AC-8, AC-16).** `AgentResponder.__init__` gains `join_in: JoinInDecider | None
= None`. `maybe_respond` becomes:
1. If `role == "assistant"`, return `None` **before anything else**: the decider is never
   constructed or called (AC-8).
2. If mentioned, respond exactly as today, with no decision call (AC-16).
3. Otherwise, if `join_in` is `None`, return `None` (today's rule).
4. Otherwise, `join_in.should_join(ctx, thread_id, msg_id, text)` reads the thread
   (`services.read_thread`) and takes the rows strictly before `msg_id`'s position (§3.2.1). It
   renders the input, then applies decide, the policy and, on hand-off, the LLM with the join-in
   prompt. A `True` result continues the existing retrieval and answer path unchanged.

`_should_respond` is replaced by this ordered logic. The trigger's ordered rule is unchanged:
join-in lives entirely in its step-4 fall-through.

**Message-type label (AC-14).**
- `background._safe_classify(classifier, ctx, posted)` is scheduled next to `_safe_embed` in
  `api.post_message` and the MCP post path, only when configured and only for non-`assistant`
  authors. Agent messages get no label. The storefront path schedules no `_safe_embed`, and it
  gets no classifier either.
- The classifier writes `Message.messageType`, `messageTypeBy` (`decision-model`/`llm-handoff`) and
  `messageTypeConfidence` via a new `repository.set_message_type`. Its query is a new QUERIES.md §6
  entry beside "Set a message embedding".
- The thread/message read projections (QUERIES.md §4 reads, §9.1/§9.2) and the response schemas
  gain these optional fields. DESIGN §5's `Message` property line gains them.
- **RAM (falkor-chat rule 6):** three small scalar properties per classified message, about the
  size of `threadId`, with no index. graph-dba confirms this figure in U11.

### 3.8 Interface and contract changes (summary)

| Surface | Change | Compatibility |
|---|---|---|
| model-bench CLI | `run --candidate`; `index.csv` +2 columns | additive |
| model-bench fingerprint | 2 profiles, additive in schema 1 | old records unaffected (§3.3) |
| model-bench roles | +2 roles (FR-20a amended) | additive |
| model-bench pack manifest | `decisionPoint`, `labels`, `lmstudio-decisions`, required `stratum` per item | old packs unchanged |
| falkor-chat `models.json` | `decisionPoints` block | absent = today's behaviour |
| falkor-chat `GuardVerdict` / trace | +2 optional fields; trace suffix | unchanged when no front |
| falkor-chat responder trigger | join-in for unmentioned messages | off by default; @mention and loop guard unchanged |
| falkor-chat `Message` | +3 optional properties; read projections and REST/MCP shapes gain them | additive |

### 3.9 FR-9: hosted decision models later

A hosted decision model would be a third adapter behind the `decisions` surface: the same
`DecisionResult`, a new fingerprint profile whose identity fields replace LM Studio's catalog
fields, and price as a measured field. Nothing in the arm, policy, candidate or bar design assumes
LM Studio beyond the `lmstudio-decisions` requires-token. That token is the one place to extend.

## 4. Units, in order

Each unit leaves its component's suite green and gets its own `HISTORY.md` entry. Units U1–U4 are
model-bench (FR-7 first). They depend on U0 only where marked.

**U0 — `-ml` method note (`data-scientist`) → `docs/plans/typed-decision-models-ml.md`.** This can
run in parallel with U1–U2. Its formulas are what `decision_stats.py` implements (§3.6). It must
answer:
- **Q1 — Data generation.**
  - Scenario taxonomy per point.
  - Eval composition at ≥200 items, including the guard's kind × answer cells and the per-label
    counts for the five message-type labels. Five labels in a 200-item set leave about 40 items per
    label, so Q1 states the per-label minimum and the set size that meets it.
  - Blind relabeling to separate generator from labeler, without filtering the set easy.
  - Cluster definition for splits, and the split sizes.
  - The rule that side-check items are never generation seeds.
  - Whether message type needs the preceding turn.
- **Q2 — Fitting objective** for `(l, h)` / `minConfidence` and the question instructions on the
  calibration split, under the FR-1+FR-2 or FR-10 constraints. Minimum calibration size. Whether
  choice confidence needs a calibration map (§3.4).
- **Q3 — FR-1 parity statistics.**
  - The verdict family for the four per-kind metrics, its α, and the test direction for a claim of
    *no* worse difference.
  - **Constraint:** no multiplicity correction and no floor rule (Holm, `verdict()`'s Rule 7
    `observable_floor`) may make FR-1 easier to pass. A larger family must never make `holds` more
    likely.
  - How the detectable difference that every `holds` line prints is computed, at about 50 items per
    cell.
  - How a reader should weigh the evaluation-set reuse count (FR-4).
- **Q4 — Mean-latency estimator** for FR-2 (paired ratio of means, its interval) and FR-10, and
  how withheld-for-load and timeout items enter it.
- **Q5 — FR-10 statistics.** Per-answer agreement intervals and the five-label family.
- **Q6 — Determinism.** Laya repeats exactly within one load, and two separate loads gave slightly
  different results (false suspend 14/30 vs 12/30). Does `basis="by-construction"` within a run
  stand (§3.3), and does a bar claim need replicate runs across loads?
- **Q7 — Fine-tuning method** (only consumed by U8).

Done = note `active`, reviewed by `analyst`. U1d (provisionally), U3b/U3c, U4, U5, U6 and U7 cite its
sections.

**U1a — model-bench: `lmstudio.decide` and the probe.**
- **Probe (live, human-approved, loads `laya`).** POST one predicate question and one choice
  question to `/api/v0/decisions`. Record the result in `model-bench/docs/HISTORY.md` and the
  `decide` docstring.
  - If the route exists **and** returns `stats`/`runtime` that `/v1` lacks, `decide` uses it and
    U1b adds `runtimeName`/`runtimeVersion` to `model:decisions`.
  - Otherwise, `decide` uses `/v1/decisions`.
- `lmstudio.py`:
  - `CallSurface` += `"decisions"`;
  - `DecisionAnswer` (frozen: `name`, `type`, `probability | None`, `choice | None`,
    `probabilities: tuple[(value, probability), ...] | None`, `confidence | None`);
  - `DecisionResult(answers, model, usage, wallClockMs)`;
  - `LMStudio.decide(input: str, questions: Sequence[Mapping], *, model, timeout_s) ->
    DecisionResult`, through `_raw_post`, with the same failure taxonomy;
  - `warm_up(call_surface="decisions", question=...)`.
- The §2.3 response, verbatim, as the offline fixture. One `-m live` test.

**U1b — model-bench: fingerprint profiles (alone).**
- `fingerprint.CallSurface` += `"decisions"`, `"handoff"`.
- The `model:decisions` and `model:handoff` profiles.
- Mutation pins for `ARM_KINDS` (still `{model, deterministic}`) and `CALL_SURFACES`.
- A regression test that validates **every** stored `results/runs/*.json` record unchanged.

**U1c — model-bench: pure `decisions.py` and the first golden vectors.**
- `Candidate.load(root, candidateId)` with its content hash.
- `apply_policy` and policy validation (band bounds; empty band iff no hand-off).
- `input_hash`, `evalSetHash`, and `INPUT_FIELDS_BY_ROLE` with its `guard-judge` row only.
- The requirement constants, `BAR_KIND_BY_ROLE` with its `guard-judge` row only, and `bar_holds`.
- `tests/fixtures/decision-contract/apply-policy.jsonl` and the `guard-judge` rows of
  `input-hash.jsonl`, with expected outputs derived per §3.2.2.

**U1d — model-bench: plain `decisions` arm wiring** (needs U1a, U1b, U1c, U3a).
- `runner.ItemScorer` Protocol: the three optional hooks `decision_input`, `answer_from_chat` and
  `score_answer` (§3.3). U3a's `classification` functions satisfy them.
- `packs.py`: the `lmstudio-decisions` token, `derive_decision_support`, and the answer-space check.
- `runner.py`:
  - `RunConfig.candidateId`;
  - `_drive_decision_items` for the `decisions` surface;
  - the §3.3 refusals;
  - `latency_block` treats `decisions` like `embeddings`;
  - `_build_fingerprint` for the profile;
  - the provisional `designEffect=1.0, basis="by-construction"` (§3.3), stated in the unit's
    `HISTORY.md` entry as provisional pending U0 Q6.
- `cli.py`: `run --candidate` (mutually exclusive with `--model`); `RunResult.armKey`; the
  `index.csv` columns.

Done when a fixture pack plus a fixture candidate produces a valid stored `model:decisions` run
offline.

**U2 — model-bench: hand-off arm** (needs U1d; both edit `runner.py`, so serialize).
- The `handoff` surface: two-model capture order, both warm-ups, two-model residency withholding,
  two-call `ItemTiming`, and `detail.decidedBy/probability`.
- Fingerprint `handoff*` fields from the second catalog entry and the chat warm-up.
- The leakage refusal (AC-3) and the fingerprint `fitManifest*` fields.
- `handoffShare` and per-path diagnostics (AC-2).

Done when an offline fixture run shows a mix of decided and handed-off items with correct timing
tuples, and an overlapping fit manifest refuses with exit 4.

**U3a — model-bench: `classification` refactor (no behaviour change).** Split `score_item` into
`answer_from_chat` + `score_answer`, with nothing else in the unit. Done when a regression test
re-scores the old pack's fixture replies and the `ItemResult`s are byte-identical.

**U3b — model-bench: decision-point pack contract** (needs U1c, U3a; U0 Q1 for strata sizes).
- The §3.5 `validate_pack` refusals, each mutation-tested.
- Required `stratum`.
- Declared per-kind and side-check metrics.
- The guard hand-off parse mode that ports `_coerce_verdict`/`_rationale_contradicts`/`_is_negated`
  for decision-point packs only. The old pack keeps today's parse.
- `tests/fixtures/decision-contract/render-guard.jsonl` and `handoff-parse-guard.jsonl`. A falkor-chat
  test that asserts `app._render_judge_user` and `_coerce_verdict` against byte-copies of them,
  with the guarded equality test. U3b's gate also runs the falkor-chat suite, with
  `falkor-chat/server` as the working directory.

**U3c — model-bench: report bar section, speed-up side** (needs U2, U3b; U0 Q3/Q4).
- `decision_stats.py` and the AGENTS.md invariant line.
- The labeler caveat, leakage, reuse and scored-n lines.
- The FR-1 lines, each `holds` with its detectable difference.
- The FR-2 same-session ratio.
- The side-check section.
- The README non-feature reword. `model-bench/AGENTS.md` invariants for the new profiles, the bar
  constants and `BAR_KIND_BY_ROLE`.

**U3d — model-bench: importer** (needs U3b; `scripts/import_decision_pack.py`, both halves).

**U4 — model-bench: new roles and FR-10 bar** (needs U3c; U0 Q1/Q4/Q5).
- `roles.py` +2, and the `join-in-judge`/`message-type-classifier` rows of `BAR_KIND_BY_ROLE` and
  `INPUT_FIELDS_BY_ROLE` (the latter per U0 Q1's `priorTurn` ruling). All of them land in this
  unit, with pins that bind each table to `ROLES`.
- The two new roles' rows of `input-hash.jsonl`.
- `scoring/agreement.py` (predicate and choice; the §3.2 renderers and hand-off parses; the two
  hand-off prompts as pack data).
- `render-join-in.jsonl`, `render-message-type.jsonl` and `handoff-parse-{join-in,message-type}.jsonl`.
- The FR-10 bar section.
- The AC-15 always-majority test.

**U5 — decision-models: scaffold, labeler, generation, splits, hashing** (needs U0 Q1).
- Component skeleton: `setup.sh`, `AGENTS.md`, `README.md`, `docs/`.
- The FR-8 import/path test.
- The labeler client: offline-tested against a fake transport, with one live test opt-in under
  `-m live`.
- The per-point generators.
- `split.py` with disjointness asserts.
- `hashing.py` and `render.py`, asserted against byte-copies of model-bench's vectors with the
  guarded equality test.

Root `AGENTS.md` gains a Structure bullet and a Component-docs row for `decision-models/`. Done
when, per point, a dataset version is generated, split, and imported into a model-bench pack that
validates. Guard comes first, then message type, then join-in.

**U6 — decision-models: fitting → candidates** (needs U0 Q2).
- `lmstudio.py` (minimal client) and `fit.py`. `fit.py`'s policy is asserted against
  `apply-policy.jsonl` before any fit runs.
- One candidate per point for off-the-shelf `laya`, with the hand-off model equal to the incumbent
  (guard) or the chosen local LLM (new abilities).
- Imported via the importer.

**U7 — Measurement runs (human-started, live; no code; needs U0 Q6).** Per point, in one session:
- the incumbent or hand-off chat arm, the decision-model-only candidate, and the hand-off
  candidate (plus replicate runs if Q6 requires them);
- `compare` reports stored in `model-bench/reports/`.

The guard needs `qwen/qwen3-4b-2507` resident, so it runs on the dev box, not during a storefront
demo (`falkor-chat/AGENTS.md` residency note). Output is a bar verdict per point. A point that
fails its bar either triggers U8 (or a U6 re-fit) or closes. The human decides which. **A closed
point leaves a record:** a `qa-engineer` test report
`model-bench/docs/test-reports/typed-decision-models-<role>-report.md` stating "does not meet its
bar" with the report lines, plus a `model-bench/docs/HISTORY.md` entry. A re-fit re-runs on the
same evaluation set, and the reuse line shows it (FR-4).

**U8 — Fine-tuning spike (conditional; needs U0 Q7).** Go/no-go question: can a checkpoint
fine-tuned on this lab's training split be exported to a GGUF that LM Studio serves as
`type:"decision"` on `/decisions`, with the same request/response contract? If yes, `fit.py`
gains a training stage, the candidate's `fitManifest.purposes` includes `training`, and the
`[train]` extra lands. If no, report back: FR-4's "trained" half is blocked on tooling, and
recalibration plus hand-off is the only lever.

**U9 — falkor-chat: decision client, config and contract copies** (after the first point meets
its bar).
- `decisions.py`: the client, the config, `apply_policy`, `record_decision`, and the join-in and
  message-type renderers and hand-off parses.
- `ModelGateway.decider`, the `Overlay` key, and the `_FORBIDDEN_NAMES` entry.
- Tests of every contract function against byte-copies of model-bench's vectors, with the guarded
  equality test.
- `models.json` ships the block with every point `enabled: false`.
- `SERVER.md` §1.8 documents the block, the ref mapping and the measured-composite rule.

**U10 — falkor-chat: guard front** (gated on the guard's bar).
- `GuardVerdict` fields, the `evaluate_guard` copy-through, `_DecisionFrontedGuardJudge`,
  `_build_guard_judge`, the trace suffix and `record_decision` calls.

**U11 — falkor-chat: message-type label** (gated on its bar).
- `graph-dba` writes and verifies the QUERIES.md §6 set-query and the §4/§9 read-projection edits,
  adds `test_queries.sh` cases, and updates DESIGN §5's `Message` line and the RAM line
  (DESIGN §7.1/§11).
- `coder` builds the classifier, `_safe_classify`, the repository/service methods, the schema
  fields and the `record_decision` calls.

**U12 — falkor-chat: join-in** (gated on its bar).
- `JoinInDecider`, the `maybe_respond` ordering and the `record_decision` calls.
- **AC-9 doc sweep.** Rewrite each of these sites to state the new rule: @mention always replies,
  agent-authored never triggers, and an unmentioned human message may draw a reply when join-in is
  enabled.
  - `responder.py`: the module docstring ("locked, DECISION 2") and the `_should_respond` comment;
  - `background._safe_respond`'s docstring ("owns the trigger policy (@mention +
    non-agent-authored)");
  - `trigger.py`'s module docstring, step 4;
  - `app.py:564` ("`AgentResponder` replies to an `@mention`");
  - `config.py:85` ("`@mention`-ing it in a message triggers a reply");
  - the post-path comments in `api.py` and `mcp.py`;
  - DESIGN §1.2: a new register row citing FR-12;
  - any matching statement in `falkor-chat/README.md`, `falkor-chat/AGENTS.md` or
    `docs/SERVER.md`.

  Rewording in `falkor-chat/docs/manuals/*` routes to `tico`, not to U12's implementer (e.g.
  `llm-provider-config.md:24,230`, "the assistant that replies to @mentions").

  **The real check is the analyst gate's read of every listed site.** The grep below only confirms
  the known strings are gone, and it must return no hits:
  `grep -rn -E "DECISION 2|replies to an .@mention|triggers a reply|@mention \+ non-agent-authored" falkor-chat/server/falkorchat falkor-chat/docs falkor-chat/README.md falkor-chat/AGENTS.md --exclude-dir=archive --exclude-dir=plans --exclude-dir=reviews --exclude-dir=test-reports --exclude=HISTORY.md`.
  Records are excluded because they stay true as records, for example
  `falkor-chat/docs/plans/demo-environment-bringup.md:100`.

**Routing suggestion (teco's call).**
- U1a–U4: `tdd-engineer` or `coder`, given model-bench's mutation-pin conventions. Each gets an
  `analyst` code gate.
- U5/U6: `coder`. U8: `data-scientist` designs, `coder` executes.
- U10/U12: `tdd-engineer`, since the behaviour contracts are crisp. U11: `graph-dba` + `coder`.
- `runner.py` is shared by U1d → U2, and `report.py` by U3c → U4. Each pair is serialized (§8).

## 5. Test strategy (behaviours, in red→green order per unit)

| Unit | Behaviour to drive | AC |
|---|---|---|
| U1a | `decide` parses the pinned predicate/choice response. Malformed `answers` raises `LMStudioCallFailed`. Timeout raises `LMStudioCallTimeout`. | AC-5 |
| U1b | `model:decisions` record validates. Carrying `temperature` is `forbidden`. Every stored run record still validates. `ARM_KINDS` is still `{model, deterministic}`. | AC-5 |
| U1c | `apply_policy` against `apply-policy.jsonl`: `p == h` → yes; `p == l` → hand-off; `p < l` → no; `l == h` → no band; `confidence == c` → choice. `input_hash`: shrinking **and** widening the `guard-judge` row each redden against `input-hash.jsonl`. | AC-2, AC-3 |
| U1c | `bar_holds`: 350 ms → met, 351 → not met (`at-most`); 0.90 → met, 0.899 → not met; ratio 2.0 → met (`at-least`). Its signature has no interval parameter, and `direction` has no default. Mutating any requirement constant or `BAR_KIND_BY_ROLE` (shrink, widen, move) reddens. | AC-12, AC-6 |
| U1d | `--candidate` whose model's catalog type is `llm` refuses. A pack with no `decisionPoint` refuses `--candidate`. A pack without `lmstudio-decisions` refuses. | AC-5 |
| U2 | Hand-off item: two calls, item wall clock ≥ the sum of the call clocks. Decided item: one call. `handoffShare` = handed/attempted. | AC-2, AC-6 |
| U2 | Fit manifest overlapping any pack item refuses (exit 4). Disjoint passes, and the fingerprint carries hash and size. A speed-up pack with `handoffModel != incumbent` refuses. | AC-3 |
| U2 | Either model non-resident between items gives `withheldFor="load"`, counted. | AC-6 |
| U3a | Old pack fixtures re-score byte-identically. | — |
| U3b | `validate_pack` refuses each of: 199 eval items; labeler == incumbent; `inputs != synthetic`; a synthetic-stratum item with `origin != synthetic`; missing `labels`; missing `stratum`; `incumbent: null` on `guard-judge`; a guard pack with no side check. It accepts 200 eval + 85 side-check items. Deleting the key each refusal reads still refuses. | AC-13, AC-4, AC-7, AC-17 |
| U3b | Guard falkor-chat renderer and `_coerce_verdict` match `render-guard.jsonl` / `handoff-parse-guard.jsonl`. A `"decision": "true"` string is not an advance on decision-point packs. | — |
| U3c | Report prints the labeler caveat first, then the leakage line, the reuse line per candidate arm, the scored n, four per-kind parity lines each `holds` with its detectable difference, and a side-check section marked "does not affect pass/fail". | AC-4, AC-3, AC-10, AC-17, AC-1, AC-13 |
| U3c | FR-2 line: same `run.sessionId` gives a ratio and interval. Different sessions give "not assessable". | AC-6 |
| U3c | Reuse count. A candidate edited in place under the same `candidateId` (new content hash) counts as a second candidate. A re-import under a new `packVersion` with the same eval items and labels keeps K. A changed label changes `evalSetHash` and resets K. A run is never counted against itself, and a same-`startedAt` run is not "earlier". In a two-candidate `compare`, each arm gets its own K. | AC-3 |
| U4 | Role tables, plus the new rows of `BAR_KIND_BY_ROLE` and `INPUT_FIELDS_BY_ROLE`: shrink, widen and move mutations each redden. | — |
| U4 | Always-"no" arm on a 70/30 set: `agreeWhenYes = 0` → bar not met. An arm at exactly 0.90 per class meets it. | AC-15, AC-12 |
| U5 | No import outside the allowlist and no `falkor-chat` path in `decisionmodels/`. Every item has `origin: synthetic`. Splits are input-hash disjoint. `render.py` and `hashing.py` match model-bench's vectors. | AC-7, AC-3 |
| U9 | `decider()` builds a client using `timeoutSeconds`. `DecisionClient` is constructed only in `modelconfig` (AST). Every contract function matches model-bench's vectors. | — |
| U10 | Policy-decided advance gives `decided_by="decision-model"` and p. A band case calls the LLM judge and gives `llm-handoff` plus p. A workspace guard override bypasses the front. The constant rationale does not trip `_rationale_contradicts`. The payload carries `[decision-model p=0.931]`. **A decision error with trace on lets the run advance or suspend normally, with payload `[llm-fallback]` and no `p=`.** With no front configured, the trace is byte-identical to today. Each path emits one `record_decision` line. | AC-11 |
| U11 | Classified message read back via REST/MCP carries `messageType`. An agent-authored message is never classified. A decision error writes no label. Each decision emits one `record_decision` line. | AC-14 |
| U12 | An `assistant`-role message with an always-yes decider gives no reply and the **decider is never called**. A mentioned message with an always-no decider gives a reply and the decider is never called. Unmentioned + yes → reply. Unmentioned + no → none. A message posted after `msg_id` is not in the rendered input. `join_in=None` keeps today's suite green (`test_no_mention_no_response`, …). | AC-8, AC-16 |
| U12 | The pinned sweep grep returns no hits. | AC-9 |

**Live verification** (opt-in `-m live`): one `decide` call (U1a); one hand-off run on a ~10-item
fixture pack (U2). U7 is the real acceptance for AC-1, AC-5, AC-6, AC-10, AC-12 and AC-17.
`qa-engineer` can own a test report over U7's stored runs that checks each AC's printed line.

## 6. Risks & open questions

**Questions for the stakeholder:** none open.

**Risks:**

- **R-1: an endpoint that is new and only partly documented.** Its shape may change across LM
  Studio versions, and the `/api/v0` variant is settled only by U1a. *Mitigation:*
  `lmStudioAppVersion` is attested in every fingerprint, and the verified response is a pinned
  fixture, so a shape change fails loudly in `decide`, not silently.
- **R-2: off-the-shelf Laya may not reach parity even with hand-off.** The informal run
  falsely advanced 14/40 at a 0.5 cut-off. A wide band can always reach parity by handing off
  most items, but FR-2 then fails, because a hand-off share above about 45% erases the 2× gain at
  ~700 ms per LLM call. U7 reveals this. U8 is the lever, and U8 may itself be infeasible on
  tooling. **The guard point may close at "does not meet its bar"** (U7's closure record).
- **R-3: GGUF export of a fine-tuned decision model** (U8) is the least-known step. That is why
  it is a go/no-go spike, not a planned build.
- **R-4: generator == labeler bias** (FR-5's accepted consequence). A frontier model labeling
  inputs it generated may agree with itself trivially. U0 Q1's blind relabeling and the FR-15 side
  check are the counterweights. The labeler caveat line keeps the claim honest.
- **R-5: selection on a reused evaluation set** (FR-4's accepted consequence). Re-fitting until a
  candidate passes the same evaluation set overstates it. The reuse line makes the count visible,
  and U0 Q3 says how to read it.
- **R-6: absence-of-evidence parity** (FR-1). At about 50 items per cell, "holds" can mean "too few
  items to tell". The detectable difference printed on every `holds` line, and U0 Q3's constraint
  against multiplicity loosening it, are the counterweights.
- **R-7: join-in exposure** (requirements review Pass 2, accepted). At the 90% bar, up to about 1 in
  10 unmentioned human messages may draw an unprompted reply or miss a wanted one. It is off by
  default, enabled per deployment, and fails closed on any error or mismatch.
- **R-8: per-message cost.** Join-in and classification add up to two decision calls per human
  message (about 20 ms each, off the request thread), plus LLM hand-offs at the measured share.
  `record_decision` makes that share observable per point.
- **R-9: model residency.** The hand-off composites need the decision model and the LLM co-resident.
  The storefront demo's "exactly the demo's models" posture (`ensure_demo_models.sh`) is unaffected
  while the points stay disabled there. Enabling any point on the storefront needs that script to
  learn about `laya`, which is a follow-up if ever wanted.
- **R-10: skipped cross-checks.** Under §3.2.2's skip rule, whoever gates U3b, U5, U6 and U9–U12
  must run the equality tests un-skipped.
- **Rollback.** Every falkor-chat change is inert until a `decisionPoints.*.enabled` flag is set.
  Rollback is flipping it back plus a restart (the config is read once, FR-15 of the provider-config
  feature). The message-type properties are additive and can be left in place.

**Assumptions taken (not questions):**
- The frontier labeler is Anthropic's API, with the key supplied via `ANTHROPIC_API_KEY`. The
  provider is a one-file change in `labeler.py`.
- Join-in's and message type's hand-off LLM defaults to `qwen/qwen3-4b-2507` (the kind-`agent`
  default). U6 may measure another.

## 7. Traceability

| FR / AC | Where satisfied |
|---|---|
| FR-1 / AC-1 | §3.5 per-kind verdict metrics, §3.6 parity lines with detectable difference; U3c, U7 |
| FR-2 / AC-6 | §3.3 two-call item timing, §3.6 same-session ratio; U2, U3c |
| FR-3 / AC-2 | §3.3 hand-off arm and incumbent hand-off refusal, §3.5 `handoffShare`, §3.7 `record_decision` and trace; U2, U9–U12 |
| FR-4 / AC-3 | §3.4 split/fit manifest, §3.5 run-time leakage refusal and reuse line; U2, U3c, U5, U6, U8 |
| FR-5 / AC-4 | §3.5 `labels` block (required), labeler ≠ incumbent refusal, caveat line; U3b, U3c |
| FR-6 | §3.4 labeler pinned in `generation.json`, printed in the caveat; U5 |
| FR-7 / AC-5 | Units ordered U1a–U7 before U9–U12; `benchResult` citation §3.7 |
| FR-8 / AC-7 | §3.4 import/path test and `origin`, §3.5 per-item `origin` refusal with side-check exemption; U3b, U5 |
| FR-9 | §3.9; every runtime candidate is an LM Studio model, structurally |
| FR-10 / AC-12, AC-15 | §3.6 agreement scorer, bar constants, `bar_holds`; U1c, U4 |
| FR-11 / AC-13 | §3.5 `MIN_EVAL_ITEMS` keyed on `decisionPoint`, scored n printed; U3b, U3c |
| FR-11a | §3.6 `bar_holds` on the measured value, interval printed; U1c, U3c, U4 |
| FR-12 / AC-8, AC-9, AC-16 | §3.7 join-in ordering; U12 sweep with its pinned grep |
| FR-13 / AC-10 | §3.5 four declared per-kind verdict metrics; U3b, U3c |
| FR-14 / AC-11 | §3.7 `GuardVerdict` fields and `None`-safe trace suffix; U10 |
| FR-15 / AC-17 | §3.5 side-check stratum; U3b, U3c |
| AC-14 | §3.7 message-type label; U11 |

## 8. Ready to implement

Plan: `docs/plans/typed-decision-models.md`.

- **Order, as a dependency graph** (`∥` = may run in parallel; `→` = must finish first):
  - **model-bench:** `U1a ∥ U1b ∥ U3a → U1c → U1d → U2`. Then `U1c + U3a → U3b`. Then
    `U2 + U3b → U3c → U4`. Then `U3b → U3d`.
  - **U0** (`data-scientist`) runs alongside. It gates the U0 questions each unit names.
  - **decision-models:** `U3d + U4 + U0 Q1 → U5 → U6` (U0 Q2).
  - **Measurement and after:** `U6 + U0 Q6 → U7 → (U8 if a point misses its bar)`.
  - **falkor-chat:** U9 → U10/U11/U12, each gated on its point's model-bench result.
- **What the plan states once.**
  - The decision-point contract and its golden vectors are §3.2.
  - The arm, fingerprint and basis design is §3.3.
  - The honesty mechanics are §3.5.
  - The bar lines and `decision_stats.py` are §3.6.
  - The runtime and measured-composite rule is §3.7.
