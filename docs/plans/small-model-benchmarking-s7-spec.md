# `model-bench` S7 — `chat-responder` pack — implementation spec

> **Status:** archived · **Owner:** `architect` · **Tracks:** S7 · **Extends:** `docs/plans/small-model-benchmarking.md` (S7)

## 1. Goal & scope

Build the two artifacts `docs/plans/small-model-benchmarking.md`'s "### S7 — `chat-responder`
pack" section requires (plan `:5926-5941`): `packs/chat-responder-grounded-answers/` and
`modelbench/scoring/grounding.py` — the **deterministic layer only** (FR-21a): latency, format,
grounding-by-containment. **Out of scope, explicitly**: any judge (`judged.py`), any calibration
set (`golden_judge_calibration.jsonl`), any faithfulness axis — the deferred design stays exactly
where it already lives (plan §3.8.5's own deferred-design paragraph, `-ml` §6.1-§6.2) and this
document does not touch, restate, or half-build any piece of it. Also out of scope: the FR-19
human-verification *process* itself (who verifies, in what order) — stated as a requirement below
(§3.6) and left for `teco` to put to the stakeholder, exactly as the S6 spec left its own FR-19
process open (`small-model-benchmarking-s6-spec.md` §2.6).

**CPG:** considered, not relevant — no `cpg_model-bench` graph is loaded on this FalkorDB instance
(re-checked live this session via `mcp__cypher__query` `GRAPHS`, which lists every loaded graph and
does not include one for this component); this is a code-level task in a component with no CPG, so
"considered, not relevant" applies rather than "not applicable."

## 2. Context & findings

### 2.1 What already exists and needs no S7 work — confirmed by reading the shipped code directly

- **`GroundingAggregates` already ships** (`modelbench/results.py:644-653`): `kind: Literal
  ["grounding"] = "grounding"`, `checklistPass: BinaryMetric | None`, `perCheck: tuple[BinaryMetric,
  ...]`, `parseFailures: int = 0`, and a `named_metrics()` that returns `checklistPass` plus every
  `perCheck` entry — the exact shape a metric-table-only role needs, with no bespoke renderer
  (§2.4 below). It is a member of `Aggregates`'s closed union and of `_AGGREGATE_BY_KIND["grounding"]`
  already (`results.py:656-670`), and `tests/test_results.py:806-845` already round-trips an *empty*
  `GroundingAggregates()` through `store()`/`load_history()` generically, with its own docstring
  stating plainly: **"`grounding` is inert until an S7 pack exists."** This class was built at S1
  as part of the five-role closed union and needs **no field addition, no `_encode`/`_decode`
  case, no round-trip test** — S7's only obligation here is to *populate* it for real, which is
  `modelbench/scoring/grounding.py`'s job (§3.4), not `results.py`'s.
- **`roles.py` already fully supports `chat-responder`**: `UNIT_KIND_BY_ROLE["chat-responder"] ==
  "item"`, `MULTI_CALL_TURN_BY_ROLE["chat-responder"] is False`, `ANALYSIS_UNIT_FIELD_BY_ROLE
  ["chat-responder"] == "itemId"` (`roles.py:16-65`). No change needed.
- **`runner.py`'s item-level driving loop (`_drive_single_call_items`) is fully generic and
  already dispatches this role correctly with zero new runner code.** Reading `run_pack`
  (`runner.py:865-884`) directly: the branch is `if pack.role == "tool-caller":
  _drive_conversations(...) else: _drive_single_call_items(...)` — `chat-responder` takes the
  `else` branch exactly like `guard-judge`/`nlq-generator`/`embedder` today. `_drive_single_call_
  items` (`runner.py:339-410`) resolves the scorer via `_load_item_scorer` (which resolves
  `modelbench.scoring.<pack.manifest["scorer"]>` — already live since S3), calls the scorer's
  optional `build_messages`/`prime` hooks via `getattr` if defined, and calls `scorer.score_item`/
  `scorer.aggregate`. **Nothing in `runner.py` is chat-responder-specific or needs to become so** —
  the whole seam this stage needs is already proven by `guard-judge`/`nlq-generator` (§2.3 below).
- **`ItemTiming`/`LatencyBlock` are captured for this role exactly as for any item-level role** —
  `_drive_single_call_items`'s per-item timing block (`runner.py:401-409`) and `latency_block()`
  (`runner.py:584-671`) are role-agnostic; a `chat-responder` run's `RunResult.latency` is populated
  the same way an `embedder`/`guard-judge` run's already is. §2.5 below is about whether `report.py`
  *prints* it, which is a separate, real finding.
- **`packs.py` needs no new `validate_pack` axis.** `_row_count_identity_problems`
  (`packs.py:622-704`) already treats `sampling.scripts` genuinely absent from the manifest as
  "the plan's own item-level-pack signal" and returns `[]` immediately — exactly the shape
  `guard-judge-understanding`/`nlq-structured-query`/`embedder-graphrag-retrieval`'s own `pack.json`
  `sampling` blocks already use (`{"seed": ..., "pairingKey": ["itemId"], "analysisUnit":
  "itemId"}`, no `scripts` key). `chat-responder`'s `pack.json` follows the identical shape (§3.2)
  and needs no new check. `_prompt_problems`/`Pack.prompt_config()` (`packs.py:356-416`) already
  forbid `prompt.maxIterationsPerTurn` and skip the `historyReplay` check for any role where
  `MULTI_CALL_TURN_BY_ROLE[role] is False` — `chat-responder` is such a role, so its `pack.json`'s
  `prompt` block only ever needs `systemPrompt`/`temperature`/`maxTokens` (§3.2), the same three
  keys `guard-judge-understanding/pack.json` declares.
- **`_scorer_problems`** (`packs.py:846-878`) already resolves `modelbench.scoring.<name>` at
  `validate` time for any non-`tool-caller` role — `"scorer": "grounding"` needs
  `modelbench/scoring/grounding.py` to exist and import cleanly; no `packs.py` change.

### 2.2 The correct structural precedent is `guard-judge`/`nlq-generator`, not `tool-caller`

`chat-responder` is a single question -> single reply role: one chat call per item, no tool schemas,
no `convo.py`-driven multi-turn loop. Confirmed directly from `roles.py`'s own table
(`MULTI_CALL_TURN_BY_ROLE["chat-responder"] is False`, unlike `tool-caller`'s `True`) and from
`run_pack`'s branch (§2.1): this role takes the **`ItemScorer`** Protocol
(`runner.py:156-208`), never `ConversationScorer` (`runner.py:211-221`, `tool-caller`-only).
`modelbench/scoring/classification.py` (`guard-judge`) and `modelbench/scoring/extraction.py`
(`nlq-generator`) are therefore the closer structural precedent than `scoring/toolcalls.py` — both
are item-shaped, single-chat-call, `ItemScorer`-Protocol modules with their own `build_messages`
hook, exactly the shape `modelbench/scoring/grounding.py` takes (§3.4).

### 2.3 A real, load-bearing design point: the generic per-item message builder would leak the answer key

`_item_chat_messages` (`runner.py:322-336`, the fallback used when a scorer defines no
`build_messages`) builds the user message as `json.dumps(item_input, sort_keys=True)` — **the
whole item row**. For a `chat-responder` item (§3.1's shape: `mustContain`/`mustNotContain`/
`mustAbstain`/`format`/`provenance` alongside `question`/`context`), this would hand the model its
own answer key and format constraints as machine-readable JSON in the same message it is meant to
answer from — exactly the failure `classification.build_messages`'s and `extraction.build_messages`'s
own docstrings each name explicitly ("Never reads `item_input["expected"]`/... — those would leak
the gold answer into the prompt", `classification.py:189`; `extraction.py:144`). **`grounding.py`
therefore must define its own `build_messages`** (the `getattr(scorer, "build_messages", None)`
hook `_drive_single_call_items` already calls, §2.1), reading only `question`/`context` and never
`mustContain`/`mustNotContain`/`mustAbstain`/`format`/`provenance` (§3.4).

### 2.4 A real, confirmed finding: `report.py` needs no new renderer for a metric-table-only role

`compare_report`'s generic "## Arms" table (`report.py:986-1032`) already renders any
`BinaryMetric`/`ContinuousMetric`/`DistributionSummary` a role's `named_metrics()` returns, Wilson
interval included when `metric.unit == unit_kind_for_role(pack.role)` — proven live today by
`ClassificationAggregates`/`ExtractionAggregates`, both of which ship *only* `BinaryMetric`s in
`perClass`/`byShape` with no bespoke table of their own. `GroundingAggregates.named_metrics()`
(§2.1) returns the same shape (`checklistPass` plus every `perCheck` entry, all `BinaryMetric`s
with `unit="item"`), so it renders through the exact same generic path with **zero new `report.py`
renderer function** — unlike `tool-caller`, which needed three (`_render_funnel`/
`_render_per_turn_position`/`_render_hazard`, S5 spec §4.4). `_render_funnel`/`_render_per_turn_
position`/`_render_hazard` all self-gate to `[]` on a non-`ToolCallAggregates` run
(`report.py:727-729`, and the analogous `isinstance` guards in the other two — confirmed by
reading each function directly), so a `chat-responder` comparison report prints no tool-caller
section at all, with no chat-responder-specific gating code needed anywhere in those three
functions.

**Two real, small additions are still needed**, neither a bespoke metric table:

1. **The plan's own Done-when text**, "the report says in words that reply quality is not measured
   by this pack" (plan `:5941`), has no existing mechanism to hang off — grep confirms `report.py`
   carries no per-role prose caveat anywhere today (the closest precedent, `_render_funnel`'s own
   `isinstance`-gated `[]` return, is a *structural* self-gate, not a printed sentence). This needs
   one small, role-gated function (§3.5).
2. **§2.5 below**, a materially larger and cross-cutting finding about `RunResult.latency` never
   being printed anywhere in `compare_report`, for any role, ever.

### 2.5 A real, cross-cutting, pre-existing gap: `compare_report` never prints `RunResult.latency`, for any role, in any stage shipped so far — flagged, not silently absorbed

Traced directly, not inferred: `grep -ni latency modelbench/report.py` returns exactly two hits,
both inside `_render_funnel`'s own docstring, and the docstring's own words are "**no `LatencyBlock`
section exists in this file to piggyback on at all**" (`report.py:710-711`). `grep -ni latency
tests/test_report.py` returns two hits, both an unrelated example metric name
(`"latencyBudgetHits"`) used to test the *generic exploratory-metric* rendering path, not a real
`LatencyBlock` assertion. `RunResult.latency`/`LatencyBlock` — computed for every run since S2
(`runner.latency_block()`, thirteen fields: `latencyMsP50/P95/Max`, `latencyTimedCount`/
`latencyItemCount`, `latencyWithheldForLoad`/`latencyWithheldForNoResponse`, `statsCoveredCount`,
`callCount`, `ttftMsMedian`, `prefillMsPer1kMedian`, `tokensPerSecondMedian`, `unexplainedMsMax`,
`results.py:273-317`) — is **never read by `compare_report` anywhere**, for `embedder` (S3),
`guard-judge`/`nlq-generator` (S4), or `tool-caller` (S5/S6), across three already-gated,
already-delivered stages with published live-run test reports. The only place any latency number
reaches a human-readable surface today is `results.rebuild_index`'s `results/index.csv`
(`results.py:1147-1175`), which **recomputes** a bare `latencyMsP50`/`latencyMsP95` directly from
`RunResult.items`' own `latencyMs` values — it does not read the stored `LatencyBlock` at all, and
prints none of TTFT/prefill/tokens-per-second/withheld-for-load-vs-no-response/coverage, which are
`LatencyBlock`-only figures.

**Why this is S7's to note rather than silently work around or silently build unscoped:** the
plan's own S7 Done-when text is explicit — *"the report prints `groundingRate` as `headlineMetric`
alongside the format counts and **the standard latency block**"* (plan `:5939-5940`) — presupposing
a "standard latency block" renderer that, traced directly, does not exist anywhere in this
codebase today, for any pack. This is not a chat-responder-specific gap (`chat-responder` is,
however, arguably the first role for which end-to-end reply latency is a first-order comparison
axis a reader would look for beside `groundingRate`, which is what surfaces it now rather than
earlier). **Resolved here as a design ruling, not left open, for the same reason the S6 spec
resolved its own `historyReplay`/manifest-correction finding as a ruling rather than a stop-and-ask
fork** (`small-model-benchmarking-s6-spec.md` §2.4/§7): the fix is small, additive, and low-blast-
radius. Concretely, why it clears this document's own high-stakes-fork bar (architect brief) rather
than needing escalation:

- **Reversible at zero cost.** The addition is one new, generic, role-agnostic render function
  (§3.5) gated only on `run.latency is not None` — if `teco`/the stakeholder later decides this
  should have waited for a dedicated stage, dropping the one step that adds it leaves every other
  piece of this document (the pack, the scorer, the format/grounding math) completely unaffected.
- **Additive-only, so it cannot invalidate already-published evidence.** `compare_report`'s output
  is regenerated on demand by re-running `compare`; the three already-published live-run test
  reports (`docs/test-reports/embedder-self-check-report.md`,
  `reports/guard-judge-understanding-20260911-02.md`,
  `reports/nlq-structured-query-20260911-01.md`,
  `docs/test-reports/small-model-benchmarking-s6-report.md`) are static snapshots of markdown
  already committed — a new section appearing in a *future* `compare` invocation changes nothing
  about what those files already say or what they were gated on. S5/S6 already established this
  precedent twice over (adding `_render_funnel`/`_render_per_turn_position`/`_render_hazard` at S5,
  then the prose-detector line and `argsOmittedRequired`/`argsWrongValue` lines at S5's own
  fix-round and S6, all incremental additions to this same shared file, none re-triggering a whole-
  report re-gate).
- **Deliberately narrow scope**, so it does not silently expand into inventing new capture
  machinery: `LatencyBlock` itself carries no `coldLoadSeconds`/peak-RAM field today (confirmed:
  `grep -rn "coldLoad\|peakRam" modelbench/*.py` finds only doc-comment mentions of
  `coldLoadSeconds` as a design intention, never a stored field) — FR-11's cold-load-time and
  peak-RAM clauses are **not** what this addition closes, and this document does not attempt to
  close them. §3.5 prints exactly the thirteen fields `LatencyBlock` already carries, nothing it
  does not.

**Flagged for `teco`'s attention specifically because it is a cross-cutting addition to a shared,
already-gated file, touching what three prior stages' reports will show from here on** — the same
lightweight-confirmation posture the S6 spec used for its own already-gated-file correction, not a
blocking question: the recommendation is to build it now, scoped exactly as §3.5 states, and name
it explicitly at hand-off so a reviewer is not surprised to see a new "## Speed" section appear in
every future `compare` run's output, including a re-run of any of S3/S4/S6's own packs.

### 2.6 A real, confirmed finding: the 30 items must each carry their own copied context, never a live reference into the embedder pack

Traced directly against both `packs.py`'s actual loader and the actual on-disk `embedder-graphrag-
retrieval` pack, per the brief's own instruction not to guess this:

- **`Pack.content_hash(root)`** (`packs.py:473-487`) hashes only the relative paths and bytes of
  files found by walking `root` — a pack's own directory. It has no notion of, and does not follow,
  any path a manifest key might point to outside that directory.
- **`Pack.data_path(key)`** (`packs.py:260-265`) resolves `self.root / data[key]` with **no
  containment check** — confirmed by reading the function directly and by confirming, via
  `grep -n "def _.*problems" modelbench/packs.py`, that `validate_pack`'s checked axes include a
  containment check for `tools.module` (`_tool_module_problems`, `packs.py:733-778`) but **none**
  for any `data.*` key.
- **`_tool_module_problems`'s own docstring states the exact failure class this would reproduce if
  a `data.*` path pointed outside the pack root**, verbatim, from a defect that already happened
  once on the `tools.module` seam and was fixed there specifically (impl review P12-2): *""../
  outside.py" validated CLEAN, `load_tool_module` executed the outside file, and `content_hash`
  never moved when that file changed — falsifying §3.3's "pack code is part of the content hash,
  so a behavior change... is a version change like any other."* A `chat-responder` item whose
  `context` field held a live docId/path reference into `packs/embedder-graphrag-retrieval/
  corpus.jsonl` would be this exact defect class on the `data.*` seam instead of `tools.module` —
  unhashed, unversioned, and (per FR-23, root `AGENTS.md`'s `cypher-mcp`-independent standalone
  rule, restated for this component at `model-bench/AGENTS.md`'s "Standalone — FR-23" hard rule) a
  live cross-pack read this component's own hard rule forbids in spirit even where `validate_pack`
  does not yet check it in code.
- **This settles the brief's own open question**: each of the 30 items' own `context` field must
  carry copied passage text directly — drafted by hand per item from the 121-message corpus already
  living at `packs/embedder-graphrag-retrieval/corpus.jsonl` (S3's work, `{"docId", "text",
  "topic"}` rows, 12 topics x ~10 messages each, confirmed by reading the file directly) — never a
  live `docId`/path reference resolved at run time into that other pack's directory. The corpus is
  **prior art to copy from while drafting**, exactly as `PROVENANCE.md`'s own "authored content, not
  copied content" shape already established for S6's `conversations.jsonl` (`small-model-
  benchmarking-s6-spec.md` §3.5) — not a live dependency.

### 2.7 The canonicalization function the plan names, cited exactly

Plan §3.8.5: grounding scoring uses *"the same canonicalization as the `nlq-generator` scorer."*
Read directly: `modelbench/scoring/extraction.py:34-36` —

```python
def _canon_str(value: Any) -> str:
    """Verbatim port of `nlq_scoring._canon_str` (`nlq_scoring.py:87-90`)."""
    return _WHITESPACE_RE.sub(" ", str(value).strip().casefold())
```

— a verbatim port of `falkor-chat/server/tests/eval/nlq_scoring.py:87-90`'s own `_canon_str`
(confirmed identical by reading both directly): case-fold plus whitespace-collapse plus strip. This
is the exact function `grounding.py`'s containment checks import and reuse (§3.4) — `extraction.py`
already exists inside `model-bench`, so importing its private `_canon_str` is an intra-package
reuse, not a cross-component one (FR-23 is about `falkor-chat`, never about `model-bench`'s own
modules reusing each other). **Not** `_scalar_equal` (numeric-epsilon comparison — irrelevant here,
grounding checks are pure substring containment over free text, never a scalar match).

### 2.8 The closest real prior art for a containment-plus-abstention checklist scorer, and why it is transcribed rather than imported

`falkor-chat/server/tests/eval/nlq_scoring.py:208-233`'s `layer2_contains` — a **secondary,
non-gating** sanity check in that codebase — is structurally the closest existing implementation of
"does a free-text reply contain the expected value(s), and does it correctly signal 'not found'":
per-value normalized-substring containment for `scalar`/`set` shapes, and a fixed tuple of English
abstention phrasings (`_ABSTENTION_MARKERS`, `nlq_scoring.py:68-84`, fourteen phrasings — "not
found", "no matching", "i don't know", etc.) checked by substring against the canonicalized reply
for the `not_found` shape. **`model-bench` never ported `layer2_contains`/`_ABSTENTION_MARKERS`** —
confirmed by `grep -rn "ABSTENTION\|abstain" modelbench/` returning nothing outside
`extraction.py`'s own, structurally different `unanswerableAbstainRate` (which detects abstention
by the *executed tool result being empty*, a structural signal `nlq-generator` has and
`chat-responder` does not, since chat-responder calls no tool). `modelbench/scoring/grounding.py`
therefore **transcribes** `_ABSTENTION_MARKERS` and a small `_looks_like_abstention` helper as DATA
+ CODE (the same D1 discipline `classification.py`'s own module docstring states and `extraction.py`
follows: "transcribes... AS DATA + CODE, never an import" — `classification.py:1-6`), adapted to
this pack's own `mustContain`/`mustNotContain`/`mustAbstain` checklist shape rather than
`nlq_scoring`'s `scalar`/`set`/`not_found` shape (§3.4).

## 3. Design & rationale

### 3.1 Golden item shape — the plan's own literal, unchanged

Thirty items, `packs/chat-responder-grounded-answers/items.jsonl`, one JSON object per line, the
plan's own shape (plan `:2776-2782`) verbatim:

```json
{"itemId": "cr-07", "question": "...", "context": ["..."],
 "mustContain": ["24.99"], "mustNotContain": ["19.99"], "mustAbstain": false,
 "format": {"maxWords": 120, "mustBeSingleParagraph": true,
            "forbiddenPatterns": ["^\\s*[-*]\\s", "```"]},
 "provenance": {"draftedBy": "...", "verifiedBy": "...", "corpusVersion": "..."}}
```

`context` is a list of **copied passage strings** (§2.6) — `[]` for a `mustAbstain: true` item
whose question has no answer in anything the corpus says about that thread, or a list of passages
that do not contain the asked-about fact, per item design. `format` is **optional per item** — an
item that omits it takes the pack-level default in full (§3.2); an item that declares it overrides
only the keys it names, the other two falling back to the pack default (§3.4's `_resolve_format`).
`mustContain`/`mustNotContain` are flat lists of literal substrings (case/whitespace-insensitive via
`_canon_str`, §2.7) — never a structured value comparison (no `nlq-generator`-style `scalar`/`set`
typing here: a chat reply is free text, not a structured extraction).

Drafted per plan §3.8.5/`-ml` §6.2: LLM-drafted from the copied 121-message corpus (topic threads,
`packs/embedder-graphrag-retrieval/corpus.jsonl`), questions as **paraphrases**, never verbatim
copies of a corpus message (FR-19), every item **human-verified** (§3.6). 30, not 20, per `-ml`
§7.1's paired floor, already settled and not reopened here.

### 3.2 `packs/chat-responder-grounded-answers/pack.json`

```json
{
  "packId": "chat-responder-grounded-answers",
  "packVersion": "0.1.0",
  "role": "chat-responder",
  "schemaVersion": 1,
  "description": "Deterministic layer only (FR-21a): grounded-reply containment against retrieved context, plus format compliance and latency. No judge, no reply-quality score.",
  "scorer": "grounding",
  "environment": {"requires": ["lmstudio-chat"]},
  "prompt": {
    "systemPrompt": "prompts/system.md",
    "temperature": 0.0,
    "maxTokens": 512
  },
  "data": {"items": "items.jsonl"},
  "format": {
    "maxWords": 150,
    "mustBeSingleParagraph": true,
    "forbiddenPatterns": ["^\\s*[-*]\\s", "```"]
  },
  "sampling": {
    "seed": 20260917,
    "pairingKey": ["itemId"],
    "analysisUnit": "itemId"
  },
  "metrics": {
    "verdictMetrics": ["groundingRate"],
    "headlineMetric": "groundingRate"
  },
  "provenance": "PROVENANCE.md"
}
```

Three notes, each a design choice this document states rather than leaves implicit:

- **No `prompt.toolSchemas`, no `prompt.historyReplay`, no `prompt.maxIterationsPerTurn`** — this
  role is single-call and tool-free; declaring any of the last two would be **rejected** by
  `Pack.prompt_config()`'s own role-scoping rule (§2.1), and `toolSchemas` is simply not read by
  anything on this role's path (`_drive_single_call_items`'s chat branch calls `lmstudio.chat`
  without a `tools=` argument at all — confirmed by reading `runner.py:369-382` directly, unlike
  `_drive_conversations`'s `llm(...)` closure, which always passes `tools=`).
- **`temperature: 0.0`** — matches every other item-level pack's own choice
  (`guard-judge-understanding`, `nlq-structured-query`), for the same reason: a deterministic
  scorer over a non-deterministic model is already noisy enough without adding sampling variance
  the pack's own sizing (`-ml` §7.1) did not budget for.
- **`format` is a new, pack-level top-level manifest block** (not inside `metrics`, not inside
  `prompt`) — this document's own placement choice, following the same precedent `metrics.
  cleanThroughTurnH.H` (`tool-caller`) and `sampling.determinismProbeScripts` (`tool-caller`) set:
  role-specific configuration that is neither a `sampling` concern nor a `prompt` concern gets its
  own top-level manifest key, read by the scorer (never by `packs.py`'s generic loader, which is
  role-agnostic by design) via `pack.manifest.get("format") or {}` (§3.4).

### 3.3 `prompts/system.md`

Plain prose (content authoring, left to the implementer, same discipline the S5/S6 specs used for
customer-facing script wording): instructs the model it answers questions using **only** the
passages it is given, must say plainly when the passages do not contain the answer rather than
guess, and must follow the reply-shape constraints it will be told per question (word budget,
single paragraph, no bullet points or code fences) — the *general* contract; the *exact* per-item
numeric budget is resolved and appended by `build_messages` (§3.4), never hand-written per item
into this static file, so a pack-level or per-item `format` change never requires editing prose.

### 3.4 `modelbench/scoring/grounding.py` — the scorer

One module (`ItemScorer`-Protocol-shaped, structural, no base class — `classification.py`'s/
`extraction.py`'s own precedent, §2.2), transcribing `_ABSTENTION_MARKERS` (§2.8) and importing
`_canon_str` from `extraction.py` (§2.7):

```python
from modelbench.scoring.extraction import _canon_str

# Transcribed from falkor-chat/server/tests/eval/nlq_scoring.py:68-84 (D1: data + code, never an
# import, per classification.py's own module docstring precedent) — Layer 2's abstention phrasings.
_ABSTENTION_MARKERS: tuple[str, ...] = (
    "not found", "no matching", "couldn't find", "could not find", "don't have", "do not have",
    "no data", "unable to find", "no information", "not available", "no record",
    "i'm not sure", "i don't know", "cannot find", "can't find",
)

_DEFAULT_FORMAT: Mapping[str, Any] = {
    "maxWords": None, "mustBeSingleParagraph": False, "forbiddenPatterns": (),
}
```

- **`looks_like_abstention(reply: str) -> bool`** — `any(marker in _canon_str(reply) for marker in
  _ABSTENTION_MARKERS)`, `layer2_contains`'s own `not_found` branch, adapted (§2.8).
- **`resolve_format(pack_format: Mapping, item_format: Mapping | None) -> dict`** —
  `{**_DEFAULT_FORMAT, **pack_format, **(item_format or {})}`, three keys, pack-level default first,
  item-level override last — the merge §3.1/§3.2 both describe.
- **`checklist_pass(reply: str, *, must_contain: Sequence[str], must_not_contain: Sequence[str],
  must_abstain: bool) -> bool`** — the verdict metric's own predicate, the plan's own three-clause
  definition verbatim (`-ml` §6.2: *"all `mustContain` present **and** no `mustNotContain` present
  **and** abstention matches `mustAbstain`"*): `all(_canon_str(c) in _canon_str(reply) for c in
  must_contain) and not any(_canon_str(c) in _canon_str(reply) for c in must_not_contain) and
  looks_like_abstention(reply) == must_abstain`.
- **`format_checks(reply: str, fmt: Mapping) -> dict[str, bool]`** — the three independent,
  never-pooled format constraints (plan `:2788-2793`): `"maxWords"` — `fmt["maxWords"] is None or
  len(reply.split()) <= fmt["maxWords"]` (a pack/item that declares no `maxWords` trivially passes,
  never refuses); `"mustBeSingleParagraph"` — `not fmt["mustBeSingleParagraph"] or "\n\n" not in
  reply.strip()` (a blank line is this document's own, minimal single-paragraph test — a reply with
  no blank line is one paragraph by construction); `"forbiddenPatterns"` — `not any(re.search(p,
  reply, re.MULTILINE) for p in fmt["forbiddenPatterns"])`, `re.MULTILINE` because the plan's own
  example pattern (`"^\\s*[-*]\\s"`) anchors per line, not per string.
- **`build_messages(item_input: Mapping[str, Any], *, pack: Pack) -> list[ChatMessage]`** — the
  `ItemScorer` hook (§2.3's own load-bearing finding). Never reads `item_input["mustContain"]`/
  `["mustNotContain"]`/`["mustAbstain"]`/`["provenance"]`. Resolves `fmt = resolve_format(pack.
  manifest.get("format") or {}, item_input.get("format"))`, appends one deterministic sentence to
  `pack.prompt_config().systemPrompt` naming the resolved constraints (e.g. `f"For this reply: stay
  under {fmt['maxWords']} words, write a single paragraph, and do not use bullet points, numbered
  lists, or code fences."` when all three constraints are set — the exact wording is this document's
  own synthesis, cheap to revise, §6), then renders the context passages labelled and numbered
  followed by the question:
  ```python
  system_prompt = pack.prompt_config().systemPrompt
  fmt = resolve_format(pack.manifest.get("format") or {}, item_input.get("format"))
  system_prompt += "\n\n" + _format_directive(fmt)   # the sentence above
  passages = "\n".join(f"[{i+1}] {p}" for i, p in enumerate(item_input.get("context") or []))
  user = f"CONTEXT:\n{passages}\n\nQUESTION: {item_input['question']}" if passages else \
      f"CONTEXT: (none provided)\n\nQUESTION: {item_input['question']}"
  return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user}]
  ```
- **`score_item(item_input, result, timing, *, pack) -> ItemResult`** — mirrors `classification.
  score_item`'s/`extraction.score_item`'s own `result is None` branch exactly: `outcome = "fail" if
  timing.withheldFor == "timeout" else "unrunnable"`, `scoreable={}`, `counts={}` (a call that never
  happened cannot pass or fail a checklist). Otherwise: `reply = result.message.get("content"); reply
  = reply if isinstance(reply, str) else ""` (defensive, mirrors `ChatResult`'s own "never raises"
  discipline, `lmstudio.py:174-178`). `checklist_ok = checklist_pass(reply, must_contain=item_input
  ["mustContain"], must_not_contain=item_input["mustNotContain"], must_abstain=item_input
  ["mustAbstain"])`. `fmt = resolve_format(...)` (same call `build_messages` made — recomputed here
  rather than threaded through, since `score_item` gets no side channel from `build_messages` and
  recomputing a pure function over already-available inputs is cheaper than inventing one). `checks
  = format_checks(reply, fmt)`. `outcome = "pass"` always (this role's `outcome` states "the call
  happened", never "the checklist passed" — correctness lives in `counts`, `classification.py`'s
  own established convention, §2.2). Returns:
  ```python
  ItemResult(
      itemId=item_input["itemId"], pairingKey=(item_input["itemId"],), outcome="pass",
      scoreable={"groundingRate": True, "formatMaxWords": True,
                 "formatSingleParagraph": True, "formatNoForbiddenPatterns": True},
      counts={"groundingRate": int(checklist_ok), "formatMaxWords": int(checks["maxWords"]),
              "formatSingleParagraph": int(checks["mustBeSingleParagraph"]),
              "formatNoForbiddenPatterns": int(checks["forbiddenPatterns"])},
      timing=timing,
      detail={"checklistPass": checklist_ok, "abstained": looks_like_abstention(reply),
              "wordCount": len(reply.split())},
  )
  ```
  Every item declares all four metrics scoreable unconditionally (unlike `guard-judge`'s tiered,
  exactly-one-metric-per-item design, §2.2's own precedent difference) — grounding and format are
  independent axes over the *same* reply, never mutually exclusive, so nothing here needs a
  `_METRIC_BY_TIER`-shaped dispatch.
- **`aggregate(items: Sequence[ItemResult], *, pack: Pack) -> GroundingAggregates`** — one pass,
  mirroring `classification.aggregate`'s own `declaring(name)` helper:
  ```python
  def declaring(name): return [it for it in items if it.scoreable.get(name)]
  grounding = declaring("groundingRate")
  checklist_pass_metric = BinaryMetric(
      name="groundingRate",
      successes=sum(it.counts.get("groundingRate", 0) for it in grounding),
      n=len(grounding), unit="item",
  )
  per_check = tuple(
      BinaryMetric(name=name, successes=sum(it.counts.get(name, 0) for it in declaring(name)),
                   n=len(declaring(name)), unit="item")
      for name in ("formatMaxWords", "formatSingleParagraph", "formatNoForbiddenPatterns")
  )
  return GroundingAggregates(checklistPass=checklist_pass_metric, perCheck=per_check,
                              parseFailures=0)
  ```
  **`parseFailures` stays `0` always for this scorer** — a design decision stated explicitly: this
  role scores free text by containment, never by parsing a structured reply (unlike `guard-judge`'s
  `extract_own_line_json_object` or `nlq-generator`'s JSON-object parse), so the "no JSON found"
  failure class the field exists for on the other two roles has no analogue here. The field stays on
  `GroundingAggregates` because it is shared union shape (§2.1), not because this scorer populates
  it.

### 3.5 `report.py` — two small, additive changes

1. **A per-role prose caveat**, closing §2.4 item 1. One new function:
   ```python
   def _render_role_caveat(pack: PackRef) -> list[str]:
       """Plan §3.8.5/S7 Done-when: `chat-responder`'s deterministic layer never measures reply
       *quality* — only grounding-by-containment, format compliance, and latency. Stated once, in
       words, so a reader does not mistake `groundingRate` for a quality score. `[]` for every
       other role (structural self-gate, `_render_funnel`'s own pattern)."""
       if pack.role != "chat-responder":
           return []
       return [
           "> **Reply quality is not measured by this pack.** `groundingRate` is a deterministic "
           "containment check against the retrieved context, never a judgement of how good, "
           "helpful, or well-written a reply is (FR-21a — the judged-quality layer is deferred, "
           "`docs/BACKLOG.md`).",
           "",
       ]
   ```
   Called once, right after the title line, before the funnel-table loop (`compare_report`,
   `report.py:911`) — the first thing a reader of a `chat-responder` comparison sees, before any
   number.
2. **A generic, role-agnostic `LatencyBlock` renderer**, closing §2.5's finding. One new function
   and one new call site, printed for **every** pack that carries a populated `run.latency`
   (retroactively benefiting `embedder`/`guard-judge`/`nlq-generator`/`tool-caller` too, per §2.5's
   own additive-only reasoning):
   ```python
   def _render_speed(runs: Sequence[RunResult], arm_names: Mapping[str, str]) -> list[str]:
       """FR-11's headline block, printed once this component-wide: `RunResult.latency` has
       existed since S2 and this is its first renderer (§2.5). `[]` when every run's `latency is
       None` (a `deterministic` arm, or a role that never times — none exist today, but the guard
       costs nothing). Never a new capture: prints exactly LatencyBlock's own thirteen fields,
       nothing FR-11 asks for that this class does not already carry (cold-load time, peak RAM —
       both out of scope, §2.5)."""
       present = [r for r in runs if r.latency is not None]
       if not present:
           return []
       lines = ["## Speed", "",
                "| arm | p50 | p95/max | timed/n | withheld (load/no-resp) | TTFT median | "
                "prefill ms/1k | tok/s median (diagnostic) |",
                "|---|---|---|---|---|---|---|---|"]
       for r in present:
           lat = r.latency
           p95_or_max = (
               f"{lat.latencyMsP95:.0f}" if lat.latencyMsP95 is not None
               else (f"{lat.latencyMsMax:.0f} (max)" if lat.latencyMsMax is not None else "—")
           )
           ttft = f"{lat.ttftMsMedian:.0f}" if lat.ttftMsMedian is not None else "— (insufficient coverage)"
           prefill = f"{lat.prefillMsPer1kMedian:.1f}" if lat.prefillMsPer1kMedian is not None else "—"
           tps = f"{lat.tokensPerSecondMedian:.1f}" if lat.tokensPerSecondMedian is not None else "—"
           p50 = f"{lat.latencyMsP50:.0f}" if lat.latencyMsP50 is not None else "— (insufficient coverage)"
           lines.append(
               f"| {_arm_label(r, arm_names[r.runId])} | {p50} | {p95_or_max} | "
               f"{lat.latencyTimedCount}/{lat.latencyItemCount} | "
               f"{lat.latencyWithheldForLoad}/{lat.latencyWithheldForNoResponse} | {ttft} | "
               f"{prefill} | {tps} |"
           )
       lines += ["", "*Descriptive only — decode tokens/sec is a diagnostic, never a comparison "
                      "instrument (FR-11).*", ""]
       return lines
   ```
   Called once, after the "## Arms" table and its footnotes and before `_render_per_turn_position`
   (`report.py:1030-1036`) — a natural "one more descriptive block before the verdict machinery"
   slot, matching where the existing per-position/hazard blocks already sit.

### 3.6 FR-19 — stated, not resolved

Every one of the 30 items must be human-verified per FR-19 (`provenance.verifiedBy` filled,
non-empty, before this stage can close) — the plan's own binding requirement (`docs/requirements/
small-model-benchmarking.md` FR-19: *"an LLM may draft labels but a human verifies every one"*), no
different in kind from S6's own FR-19 obligation over its 12 conversation scripts. **This document
does not choose the process.** S6's own coordination settled a specific three-step process for its
12 scripts (agent draft -> independent, non-authoring agent pre-check -> full, non-sampled
stakeholder review, `small-model-benchmarking-s6-spec.md` §2.6) — a strong candidate for this stage
to reuse verbatim (drafting a checklist item and verifying a checklist item against a fixed corpus
are structurally similar tasks, arguably *simpler* to pre-check than a 12-script conversation, since
there is no simulated environment to execute against — only a fixed, already-copied corpus text to
read the claimed `mustContain`/`mustNotContain` substrings against directly). That reuse decision is
`teco`'s to put to the stakeholder when this coordination reaches that step, exactly as the S6 spec
itself deferred it (`small-model-benchmarking-s6-spec.md`, brief's own citation). No conversation
script's — here, no item's — `provenance.verifiedBy` may be filled by anyone but that process's own
final human step, and no run that feeds a published report may claim results drawn from items whose
`verifiedBy` is still empty.

### 3.7 `PROVENANCE.md`

Follows S6's own established shape for **authored, not copied** content
(`small-model-benchmarking-s6-spec.md` §3.5) — a per-item table (`itemId`, `draftedBy`, `basedOn`
naming the corpus `docId`(s)/topic the item paraphrases, `verifiedBy`, `verifiedAt`), plus one
prose paragraph stating plainly that `items.jsonl` is authored from the copied 121-message corpus
(`packs/embedder-graphrag-retrieval/corpus.jsonl`, itself copied one-way from `falkor-chat` at S3)
and is never a live reference to that pack (§2.6). `packVersion` starts at `0.1.0` and is not
version-bumped a second time when `verifiedBy` is filled per item — mirroring S6's own
`packVersion`-bump discipline exactly (a first-delivery version, not a revision of a published one;
per-item `provenance.verifiedBy` moving from empty to filled *does* move `content_hash`, which is
the record FR-6/AC-3 need).

## 4. File/module layout

| File | New/changed | Owner in this stage |
|---|---|---|
| `modelbench/scoring/grounding.py` | new | the scorer (§3.4) |
| `modelbench/report.py` | changed | `_render_role_caveat` (§3.5 item 1), `_render_speed` (§3.5 item 2), two call sites in `compare_report` |
| `packs/chat-responder-grounded-answers/pack.json` | new | §3.2 |
| `packs/chat-responder-grounded-answers/items.jsonl` | new | §3.1, 30 items |
| `packs/chat-responder-grounded-answers/prompts/system.md` | new | §3.3 |
| `packs/chat-responder-grounded-answers/PROVENANCE.md` | new | §3.7 |
| `tests/test_scoring_grounding.py` | new | the scorer's own tests (§5) |
| `tests/test_report.py` | changed | `_render_role_caveat` test (populated/absent by role), `_render_speed` test (populated/`[]`, and a regression fixture over an *existing* pack's fixture data confirming the new section does not alter any existing assertion) |
| `tests/test_runner.py` | changed | one small integration case confirming `_drive_single_call_items` reaches `grounding.build_messages`/`grounding.score_item` end to end against a `FakePack`-shaped `chat-responder` fixture and a stub LLM (mirrors the existing `guard-judge`/`nlq-generator` integration fixtures already in this file) |
| `model-bench/docs/test-reports/small-model-benchmarking-s7-report.md` | new | the live-run record (§5, item 16) |

No change to `modelbench/results.py`, `modelbench/roles.py`, `modelbench/runner.py` (beyond nothing
— confirmed generic, §2.1), or `modelbench/packs.py` (confirmed generic, §2.1).

## 5. Step sequence

Six steps: two small offline code steps, two content-authoring steps, the FR-19 process (stated,
not executed by an agent, §3.6), and the one live run S7 owes. Every code step is independently
red-then-green and entirely offline; only the last step touches LM Studio.

### Step 0 — `modelbench/scoring/grounding.py` (offline)

§3.4 in full: `looks_like_abstention`, `resolve_format`, `checklist_pass`, `format_checks`,
`build_messages`, `score_item`, `aggregate`. Unit-tested against hand-built `item_input`
dicts/`ChatResult`-shaped stand-ins (mirroring `test_scoring_classification.py`'s own fixture
style, no LM Studio), covering: a checklist-passing reply; a reply missing a required `mustContain`
substring; a reply containing a forbidden `mustNotContain` substring; a correctly-abstaining reply
against a `mustAbstain: true` item; an incorrectly-abstaining reply against a `mustAbstain: false`
item (checklist fails even though `mustContain`/`mustNotContain` might otherwise pass, since
`looks_like_abstention(reply) != must_abstain`); each of the three format checks independently
moving (a too-long reply, a two-paragraph reply, a reply matching `forbiddenPatterns`) and each
staying green on an otherwise-identical passing fixture (the mutation-pair discipline, `model-bench/
AGENTS.md`'s guard-reach convention); `resolve_format`'s three-way merge (pack default only, item
override of one key, item override of all three); `build_messages`'s own never-reads assertion
(construct an `item_input` and assert none of `mustContain`/`mustNotContain`/`mustAbstain`/
`provenance`'s values appear anywhere in the returned messages' content — the same shape
`classification`/`extraction`'s own tests already use for this exact claim, if present — confirm by
reading their test files' own pattern and mirror it; if neither test file asserts this today, this
step is the first to, and that is itself worth a one-line note in this stage's own test report);
`score_item`'s `result is None` branch (`timeout` -> `"fail"`, `no_response` -> `"unrunnable"`,
`scoreable={}`); `aggregate`'s four independent `BinaryMetric`s over a small hand-built `items`
list, each `n` reflecting only the items that declared it (trivially all of them here, since
`score_item` always declares all four — still worth a test pinning that invariant, since a future
edit could accidentally special-case one metric). **Done when:** every case above is green,
`ruff check .` is clean, and `modelbench.scoring.grounding` imports cleanly with no dependency
outside stdlib + `modelbench.*` (FR-23; `model-bench/AGENTS.md`'s zero-runtime-dependency rule).

### Step 1 — `report.py`'s two additions (offline)

§3.5: `_render_role_caveat`, `_render_speed`, wired into `compare_report`. Tests in
`test_report.py`: `_render_role_caveat` prints for a `chat-responder`-role `PackRef` fixture and
returns `[]` for every other role fixture already in this file (a widen-and-shrink pair per
`model-bench/AGENTS.md`'s guard-reach convention: add a role to the caveat -> a previously-silent
fixture now prints; remove `chat-responder`'s own case -> the chat-responder fixture goes silent).
`_render_speed` prints a populated table for a hand-built `RunResult` carrying a populated
`LatencyBlock` (both a fully-covered case and an insufficient-coverage case, asserting the "—
(insufficient coverage)" text rather than a computed number, `-ml` §11.6's floor), returns `[]` for
a `RunResult` with `latency=None` (a `deterministic`-armed fixture, if one exists in this file's
fixtures, or a hand-built one), and — the regression half §4's own table names — re-runs at least
one **existing** `compare_report` fixture test from this file (an `embedder` or `guard-judge`
fixture already asserting specific line content) and confirms every previously-asserted line is
still present, unchanged, with the new "## Speed" section appearing as an addition rather than a
replacement of anything. **Done when:** `pytest -q` is green, `ruff check .` is clean, and the
regression assertion above passes without modification to any pre-existing assertion in the file
it runs against.

### Step 2 — `pack.json`, `prompts/system.md` (offline, small)

§3.2, §3.3. Confirm the manifest loads standalone via `pack_ref_from_manifest`/`metrics_from_
manifest` (the S1-level, `items.jsonl`-independent read — `packs.py:223-236`) ahead of authoring the
real items. **Done when:** the manifest-only read succeeds with no error and `metrics.
verdictMetrics == ["groundingRate"]`, `metrics.headlineMetric == "groundingRate"`.

### Step 3 — Author `items.jsonl` (content authoring, offline)

Thirty items per §3.1, drafted from `packs/embedder-graphrag-retrieval/corpus.jsonl`'s twelve
topic threads (plan `:2768`'s own "derived from the copied 121-message corpus"), questions as
paraphrases never verbatim quotes, `context` carrying copied passage text (§2.6) — never a docId
reference. A reasonable spread across the corpus's twelve topics (roughly 2-3 items per topic, so
no single thread dominates the pack), and across the checklist's own three axes: a majority of
items answerable-and-checkable (`mustContain` non-empty, `mustAbstain: false`), a meaningful
minority genuinely unanswerable from their given `context` (`mustAbstain: true`, `context` either
`[]` or passages that do not contain the asked-about fact), and at least a few items whose
`mustNotContain` guards against a specific, plausible fabrication drawn from a *different* topic's
similar-sounding fact (mirroring `tool-caller`'s own `boundaryRule`-confusion-value discipline,
S5 spec §3.3, applied here to a free-text fabrication trap rather than a numeric boundary). Every
item's `provenance.draftedBy`/`basedOn` filled at authoring time; `verifiedBy` left empty (§3.6).
**Done when:** all 30 rows parse via `Pack.iter_items()` with no error, and a hand cross-check
confirms the spread above (a mechanical count, not a claim).

### Step 4 — Write `PROVENANCE.md` (content authoring, offline)

§3.7. `verifiedBy`/`verifiedAt` columns left blank. **Done when:** `PROVENANCE.md` names all 30
items by id with a `basedOn` entry for each, and states plainly that `items.jsonl` is authored, not
copied.

### Step 5 — FR-19 verification (process, not an agent's to execute per se, §3.6)

Whatever process `teco` and the stakeholder settle (§3.6's own recommendation: reuse S6's three-
step shape). **Done when:** all 30 rows carry a non-empty `verifiedBy`, filled only by that
process's own final human step.

### Step 6 — One live run (item 16, §6)

`model-bench run --pack chat-responder-grounded-answers --model <a resident chat model>`, then
`model-bench compare` against a second arm (a negative control, `--negative-control` with the same
model twice, is the cheapest single live-run proof this stage's own report can point to — mirroring
item 19a's own value elsewhere in this component, though S7 owes no numbered item 19 of its own,
§6). Confirm the manifest correction/format-directive/grounding math all behave against a real
model's real replies — in particular, confirm at least one item's reply is long/unstructured enough
to exercise the `maxWords`/`mustBeSingleParagraph`/`forbiddenPatterns` checks for real, and that the
new "## Speed" section and role-caveat line both render on real output. Write
`model-bench/docs/test-reports/small-model-benchmarking-s7-report.md`. **Done when:** the run
stores successfully, `compare`'s output shows `groundingRate` as the headline, the format counts as
exploratory, the "## Speed" section, and the reply-quality caveat, all on real output — the report
states this plainly rather than requiring a reader to re-derive it from raw JSON.

## 6. Test strategy

Per plan §5's stage-ownership table, **S7 owes item 16 only** ("one full run per pack, end to end,
producing a stored, valid result" — plan `:6294`) — the brief's own citation, confirmed by reading
the table's S7 row directly (`| **S7** — chat-responder | **16** | — |`). S7 owes no other numbered
item: it is not the `tool-caller` pack, so items 17/19a/19b (AC-1, the negative control, the
known-answer validation) do not apply; it introduces no new `stats.py`/clustering surface (item 7b/
Rule 5's territory, `tool-caller`-only), and its own pre-registered family is a single verdict
metric (`k=1`, no Holm ladder to test beyond what S1's already-generic machinery already covers).

- **Item 16** → Step 6, this stage's own live run.
- **Unit-tier, this stage's own new surface, not separately numbered** (the S4/S5/S6 specs' own
  precedent for un-numbered scorer-module tests): `grounding.py`'s own functions (Step 0), the two
  `report.py` additions (Step 1), and the small runner-integration case named in §4's file table.
- **No acceptance-tier gate beyond item 16 is owed** — this stage introduces no new statistical
  machinery, no new pack-loader axis, and no new role-dispatch branch (§2.1); everything it touches
  is either brand-new, narrowly-scoped code (the scorer) or a small, additive extension of
  already-tested generic machinery (the two `report.py` functions), covered at the unit tier.

## 7. Risks & open questions

- **The `_render_speed` addition (§2.5/§3.5) is a real, cross-cutting gap this document found and
  resolved as a design ruling rather than escalated** — flagged here for visibility per this
  repository's own high-stakes-fork discipline, even though it clears the bar for "resolve, don't
  escalate" (§2.5 states the reasoning: reversible at zero cost, additive-only, deliberately narrow
  scope). The one thing worth a reviewer's explicit sign-off before implementation: this is the
  **first** `LatencyBlock` renderer in the component's history, so its exact column choices/wording
  (§3.5) are this document's own synthesis, not a transcription of an existing pattern — cheap to
  revise (one function, no stored-data consumer beyond its own printed lines) if a future reviewer
  wants different columns or thresholds.
- **FR-19's verification process for these 30 items is deliberately left open** (§3.6) — not a gap
  in this document, a decision this document correctly does not make on `teco`'s/the stakeholder's
  behalf, mirroring the S6 spec's own identical posture.
- **The format-directive sentence's exact wording (§3.4's `build_messages`) is this document's own
  synthesis**, not a transcription of anything — cheap to revise (one f-string, no stored-record
  consumer) if real replies show the model does not reliably respect a plain-English word-budget
  instruction; the scoring math itself (§3.4's `format_checks`) is unaffected by any wording change,
  since it measures the reply against the resolved numbers regardless of how they were phrased to
  the model.
- **The 30 items' exact question/context/checklist content is not written here** (§3.1's/Step 3's
  own scope note) — deliberate, the same reasoning the S6 spec gave for its own 108 turns of
  scripted customer utterances: a spec that pre-wrote all 30 items would not be reviewable as a spec
  and would leave the implementer no real authoring work.
- **No genuinely unresolvable fork was hit.** The closest candidate — whether to build `_render_
  speed` now versus flag it for a dedicated future stage — is resolved above rather than escalated,
  for the stated reasons (reversible, additive-only, narrow scope, directly named by the plan's own
  S7 Done-when text).

## Ready to implement

Document: `docs/plans/small-model-benchmarking-s7-spec.md` (this file). Six steps (§5), all
offline except Step 6: **Step 0** (`modelbench/scoring/grounding.py` — `looks_like_abstention`,
`resolve_format`, `checklist_pass`, `format_checks`, `build_messages`, `score_item`, `aggregate`,
§3.4); **Step 1** (`report.py`'s `_render_role_caveat` + `_render_speed`, §3.5); **Step 2**
(`pack.json` + `prompts/system.md`, §3.2/§3.3); **Step 3** (author the 30 items, `verifiedBy` left
empty, §3.1); **Step 4** (`PROVENANCE.md`, §3.7); **Step 5** (FR-19 verification — process left for
`teco`/the stakeholder, §3.6); **Step 6** (one live run, item 16, plus the test report). Four real
gaps found and resolved as design rulings, not left open: `GroundingAggregates`/`roles.py`/
`runner.py`/`packs.py` already fully support this role with **zero** production-code change beyond
the scorer itself and two small `report.py` additions (§2.1, a materially smaller footprint than
S5/S6's own tool-caller work); the generic `_item_chat_messages` fallback would leak the answer key
into the prompt, so `grounding.py` must define its own `build_messages` (§2.3); `report.py` has
never printed `RunResult.latency` for any role in any prior stage, a real cross-cutting gap this
document closes with one small, additive, role-agnostic renderer rather than leaving unscoped or
silently working around (§2.5); and the 30 items' own `context` field must carry copied passage
text, never a live reference into `packs/embedder-graphrag-retrieval/`, traced directly through
`content_hash`'s own containment behavior and `_tool_module_problems`'s own prior defect on the
analogous `tools.module` seam (§2.6). FR-19's verification process is stated as a binding
requirement and explicitly left for `teco` to put to the stakeholder, mirroring the S6 spec's own
posture exactly. No genuinely unresolvable fork was hit.

**Proposed next units, in sequence**, consistent with this coordination's own S4-S8 discipline
(stage spec -> `teco`-verified directly against source, no separate `analyst` plan gate ->
implementation -> `analyst` code gate -> `qa-engineer` acceptance -> stage close):

1. **Steps 0-1 (code)** — `tdd-engineer`, offline, test-first (both are small, well-specified
   additions with an obvious red/green shape named in §5's own "Done when" clauses).
2. **Step 2 (manifest + prompt)** — the same agent, immediately after (a direct continuation).
3. **Steps 3-4 (content authoring)** — `tdd-engineer` or `coder` (either fits — content authoring
   against a fixed schema, not new production logic, so the choice is not load-bearing the way it
   was for Step 0).
4. **Step 5 (FR-19 verification)** — dispatched by `teco` once §3.6's process question is put to
   the stakeholder; blocks Step 6 until every item carries a non-empty `verifiedBy`.
5. **Step 6 (the live run + test report)** — `qa-engineer`, once Step 5 clears.
6. **Stage close** — `analyst`'s code gate over Steps 0-1's production-code diff (small: one new
   scorer module, two small `report.py` additions), then the S8 doc-sync unit (already scoped by
   the top-level plan, not this document's to re-specify).
