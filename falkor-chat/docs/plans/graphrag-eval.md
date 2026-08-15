# GraphRAG retrieval + generation evaluation harness — implementation plan

> **Status:** active · **Owner:** `architect` · **Tracks:** K-026 (M2.5-quality) · **Version:** 4

**Revision note (v4, 2026-08-15):** stakeholder-directed, coordinator-relayed design change made
*after* Pass 3 approval — **not** a reopened review finding (v3 was `analyst`-approved, Pass 3,
verdict Approve, before this change). Per a decision the user made directly and `teco` relayed as
coordinator: Unit 3's judge model collapses onto **the same model as the agent-under-test**,
`qwen/qwen3-4b-2507` — replacing D1's prior default judge model, `openai/gpt-oss-20b`. This is a
deliberate real-hardware-constraint call: the LM Studio instance in question does have larger
models registered/reachable (`openai/gpt-oss-20b`, `qwen/qwen3.5-9b`, `google/gemma-4-12b`,
`prism-ml/bonsai-27b` alongside the 4B), but the actual hardware cannot run them for this
workload — "registered" is not "usable" here. The stakeholder made this call with full awareness
that it drops the method note's self-preference-bias safeguard ("never the 4B-under-test judging
itself," `docs/plans/graphrag-eval-ml.md` §"Layer 2"). This plan does not re-litigate that
trade-off — it documents the change and adds an explicit, sign-off-gated methodology/limitation
note (new D1 subsection below) so Unit 3's judge numbers are never read as if the safeguard still
held. Scope of this revision: D1 and the places that cite its judge-model default (§5 Unit 3, §6,
§7 items 2/8) only — D2–D7, Units 1/2a/2b's file specs, and everything else are untouched.

**Revision note (v3, 2026-08-15):** narrow follow-up fix per the `analyst` Pass 2 re-gate
(`docs/reviews/graphrag-eval.md`, Pass 2 verdict *needs changes, narrowly*) as coordinated by
`teco`. Pass 2 confirmed C-1, M-1, and M-2 genuinely closed by v2 — no further action on those. The
one residual: M-3 was only half-fixed — the msgId-existence check correctly moved to Unit 2b, but
Unit 2a's `test_golden_set_integrity.py` still listed a self-retrieval-inflation check ("no query
is a substring of its target message's text") that needs `Message.text` from `ws:eval`, which
`golden_retrieval.jsonl`'s schema didn't carry — silently reintroducing the exact `ws:eval`
dependency Unit 2a's "genuinely network/DB-free" claim said it didn't have. Fixed here (coordinator
adopted the reviewer's leaned-toward recommendation) by adding a `target_text` field to
`golden_retrieval.jsonl`'s schema — the self-retrieval check now runs as a pure string comparison
against the fixture itself, no graph read needed. See §5 Unit 2a and the confirmation below.

**Revision note (v2, 2026-08-15):** revised in place per the `analyst` plan-gate review
(`docs/reviews/graphrag-eval.md`, Pass 1 verdict *needs changes*) as coordinated by `teco`. Fixed: a
Critical grounding gap (`server/tests/conftest.py`'s autouse fixture silently redirects
`ModelGateway.from_env()` away from the real `config/models.json` inside every pytest test — new
§3 D7 resolves it), a factual citation error (`AgentResponder._SYSTEM_PROMPT` is a module-level
constant, not a class attribute — §2/§5 Unit 3), a dispatch-sizing breach (Unit 2's 8 files split
into Unit 2a/2b), and a self-contradiction in the golden-set integrity test's `ws:eval` dependency
(resolved by moving the one check that needs it into Unit 2b — see the v3 note above for the part
of this that Pass 2 found still incomplete). Folded in two coordinator-accepted opinion calls from
the review (a `data-scientist` sign-off gate on the first committed baseline; a recommended real
human spot-check for the judge-calibration set specifically). See §7 for the full disposition of
every review finding.

Turns `docs/plans/graphrag-eval-ml.md` (the `data-scientist` method note, ✅ 2026-07-10) into an
ordered, buildable implementation plan: files, interfaces, sequencing, test strategy. The method
note owns *what to measure and why*; this document owns *how it gets built*. Do not re-litigate the
method note's choices here — see its own file for the two-layer design, the metric formulas, the
acceptance-threshold table, and the risk register.

---

## 1. Goal & scope

Build a re-runnable offline evaluation harness for falkor-chat's GraphRAG retrieval + generation,
so a future change (Entity extraction, hybrid fusion, a seed-relevance cutoff, an embedding-model
swap) ships against a measured baseline instead of on vibes. Concretely, per the backlog (K-026)
and the method note:

1. A representative seeded corpus, `ws:eval` (step 0).
2. `server/tests/eval/golden_retrieval.jsonl` — 30–50 paraphrased `query → relevant_msgId(s)`
   pairs, with a harness assertion against verbatim self-retrieval.
3. A deterministic retrieval-eval module over `Services.hybrid_search` — recall@10 (primary),
   recall@5, MRR — establishing the vector-only @1024 baseline. **Zero network/LLM dependency.**
4. A judge layer — ~15–20 Q&A, LLM-as-judge faithfulness + answer-relevance, calibrated against
   ~10 labeled examples, judge-vs-human agreement reported. **Live-marker-gated; skips cleanly
   (never fails, never fabricates) when the live LLM is unreachable.**
5. A metrics report (`docs/test-reports/graphrag-eval-<date>.md`) a later QA pass can read without
   re-deriving anything.

**Explicitly out of scope** (per the method note's own "rejected alternatives" and the backlog's
"why now"): wiring the Entity-extraction expansion, hybrid full-text fusion, a live seed-distance
cutoff, resolving the grounding-permissive system prompt, or an online A/B harness. This plan
builds the *instrument*; it does not use it to change retrieval behavior.

---

## 2. Context & findings (verified against the current codebase, not assumed from the note)

- **`Repository.hybrid_search`** (`server/falkorchat/repository.py:748`) and its `Services` wrapper
  (`server/falkorchat/services.py:852`) match the method note's description exactly: single-vector
  kNN seed (`db.idx.vector.queryNodes('Message', 'embedding', $k, …)`), cosine distance **ASC**,
  thread-scope traversal, a dormant `Entity` `OPTIONAL MATCH` that no-ops. Returns
  `{msgId, text, role, score, relatedContext}`; `score` is distance (0 = identical), never
  re-sorted; **ANN recall is approximate — never assume exactly `k` rows come back.**
  `Services.hybrid_search(ctx, *, q_vec, k=10, limit=10, channel_id=None)` is the call surface this
  plan uses (not the bare repository method) — it's what `AgentResponder` actually calls, so the
  eval stays representative of production. **Finding: the method note is accurate here — nothing
  stale to flag.**
- **`config/models.json`** declares `defaults.embedding = "lmstudio/text-embedding-qwen3-embedding-0.6b"`
  with `models."lmstudio/text-embedding-qwen3-embedding-0.6b".dim = 1024` — confirms the method
  note's "vector-only @1024" framing. `defaults.agent/step/guard` all point at
  `lmstudio/qwen/qwen3-4b-2507` — the model under test.
- **`modelconfig.ModelGateway`** (`server/falkorchat/modelconfig.py:87`): `KINDS = frozenset({"agent",
  "step", "embedding", "guard"})` — **there is no `judge` kind today**, confirmed by reading the
  frozen set directly, not inferred. See §3 (Design decision D1) for the resolution.
- **`server/tests/conftest.py`'s autouse `_model_config_env` fixture (`:100-123`) redirects
  `ModelGateway.from_env()` to `server/tests/data/models.json` (embedding dim **4**) for every test
  collected under `testpaths = ["tests"]` — including everything this plan adds under
  `server/tests/eval/`, regardless of marker.** Confirmed by reading both files: the real
  `config/models.json` declares the embedding model at dim **1024**; the test fixture's
  `server/tests/data/models.json` declares the *same model ref* at dim **4** (mirrors `ws:test`'s own
  `TEST_EMBEDDING_DIM`). Any code this plan adds that calls `ModelGateway.from_env()` **inside a
  pytest context** silently resolves against the wrong file — this was a Critical grounding gap in
  the plan's first draft (`docs/reviews/graphrag-eval.md` C-1), not caught until the plan-gate
  review. Resolution: §3 D7.
- **FR-4's test-only exception is directory-scoped, not just the two named files.** The AST
  enforcement test (`server/tests/test_modelconfig.py:1188-1213`,
  `test_fr4_only_modelconfig_constructs_openai_compatible_clients_directly`) only walks
  `server/falkorchat/*.py` — it does not reach `server/tests/**` at all. So every module this plan
  adds under `server/tests/eval/` may construct `OpenAICompatibleLLM`/`OpenAICompatibleEmbedder`
  directly, exactly as `server/tests/test_workflow_live.py` already does. `scripts/*.py` is outside
  the scanned tree too, but this plan's corpus-seed script uses `ModelGateway.from_env()` anyway
  (see §5 Unit 1) — that's the more correct choice, not just the permitted one: it makes the seed
  script honor whatever embedding model `config/models.json` actually names, with no separate
  live-only env vars to keep in sync. **This is safe precisely because Unit 1's script is never
  collected by pytest — no autouse fixture ever touches it (see D7's asymmetry note).**
- **`server/tests/test_workflow_live.py`** is the shape to mirror verbatim for every live test this
  plan adds: `pytestmark = pytest.mark.live` (deselected by default via `addopts = '-ra -m "not
  live"'`, `server/pyproject.toml:31`), a module-scoped fixture that probes FalkorDB (`PING`
  equivalent) and the live LLM/embedder, **skips (`pytest.skip`, never fails)** with a clear reason
  on either being unreachable, a throwaway workspace bootstrapped at the **probed** (or here,
  **resolved-from-config**) dimension — never a hardcoded 1024 — and a `KEEP_WS` escape hatch.
  Live-test-only model literals are read from env vars with defaults matching the M2 stack
  (`FALKORCHAT_LIVE_LLM_BASE_URL`, `FALKORCHAT_LIVE_LLM_MODEL`,
  `FALKORCHAT_LIVE_EMBEDDING_MODEL`) — this plan's judge layer adds exactly one more,
  `FALKORCHAT_LIVE_JUDGE_MODEL`, following the identical pattern (§3 D1). **Its clients are
  constructed directly from these literals, never via `ModelGateway`** — its own docstring
  (`:14-19`) says why: "a live run needs no config file." This is the same reason Unit 3's clients
  do the same (D7 mechanism 2).
- **`server/tests/conftest.py`** is the pattern for a *rebuilt-every-session* throwaway workspace
  (`ws:test` at `TEST_EMBEDDING_DIM=4`, wiped node data per test). `ws:eval` deliberately does
  **not** follow this pattern — see §3 D2 for why it must be a *persistent* workspace instead.
- **`server/tests/eval/golden_guards.jsonl`** (K-027, 26 lines) is a same-directory precedent for
  this repo's "golden fixture" JSONL shape (`id`, a discriminating field, the judged input, an
  `expected` boolean, a `label_rationale` string) — useful as a shape reference, **not** a schema to
  reuse verbatim: it judges guard conditions over an `understanding`/`turns` object, not
  `query → relevant_msgId`. `golden_retrieval.jsonl`'s schema is defined fresh in §5 Unit 2a.
- **`docs/archive/plans/m3-guard-calibration.md`** is the closest existing precedent for a
  judge-calibration protocol in this repo (Cohen's κ discussion, a small-N statistical-honesty
  section, a mandatory caveat in the report). This plan borrows its *posture* (report the
  agreement number, mandate a small-N caveat, never claim more than the sample size supports) but
  **not** its κ machinery — see §3 D4 for why a simpler statistic fits K-026's ~10-example
  calibration set better.
- **`scripts/seed_demo.sh`** establishes the idempotent-seed convention this plan's corpus script
  follows for `Channel`/`Thread` (fixed ids, `MERGE`, safe re-run). **`repository.create_channel`/
  `create_thread` are plain, non-idempotent `CREATE`** (`server/falkorchat/repository.py:168-226`,
  confirmed by reading the docstrings: "Ids are server-minted uuids, so a MERGE could never match
  — this is a CREATE... creates are non-idempotent") — so the corpus-seed script must **not** call
  those two repository methods with fixed ids; it needs its own small `MERGE`-based Cypher for
  Channel/Thread (mirroring `seed_demo.sh`'s inline pattern), while messages use
  `Repository.post_first_message`/`post_subsequent_message` directly with **fixed `msgId`s** — the
  §4 v2 write path's own `dupMsg` status *is* idempotency-by-design for a retried identical write
  (`AGENTS.md`: "dupMsg = idempotent retry"), so re-running the seed script against an
  already-seeded `ws:eval` is a safe no-op for messages already present. **`Repository.thread_has_head`
  (`repository.py:243`) is the existing primitive for the first-vs-subsequent write decision** —
  the same one `services._dispatch_write` (`services.py:704`) uses — and the seed script should
  call it directly rather than re-deriving an equivalent check.
- **`scripts/bootstrap_schema.sh`** takes `EMBEDDING_DIM` (default 1536) and creates the `Message`/
  `Chunk` vector indexes at that dimension (`:234-240`). A vector index's dimension is fixed at
  creation (live-verified fact, repeated throughout the codebase) — `ws:eval` must be bootstrapped
  at exactly the embedding model's declared dimension before any message is embedded.
- **`scripts/load_test.sh` + `scripts/load_append.py`** is the existing precedent for a bash
  wrapper (env/precondition checks, calls into the venv) pairing with a Python worker for the parts
  bash can't do (real HTTP/embedding calls) — this plan's `seed_eval_corpus.sh` +
  `seed_eval_corpus.py` follows the same split.
- **`AgentResponder`** (`server/falkorchat/responder.py`) has no side-effect-free "generate only,
  don't post" seam — `maybe_respond` always ends in a real `services.post_agent_answer` write.
  **`_SYSTEM_PROMPT` is a module-level constant (`responder.py:36`), defined *before* the class
  statement (`:43`) — not a class attribute; `AgentResponder._SYSTEM_PROMPT` does not exist and
  raises `AttributeError`** (the plan's first draft cited it as a class attribute — corrected here,
  `docs/reviews/graphrag-eval.md` M-1). The judge layer (§5 Unit 3) reuses
  `AgentResponder._build_prompt(...)` — a real instance method that never reads anything but its
  arguments and already embeds `_SYSTEM_PROMPT` in the message list it returns, so no separate
  import of the constant is needed — via a throwaway instance (`__init__` does no I/O; it only
  wraps `services=None`/no embedder/llm into a `StaticModelGateway` sugar object), rather than
  duplicating the four-line prompt-construction logic — duplicating it would silently drift from
  production the next time the prompt changes. Flagged as a small follow-up opportunity, not
  required for K-026: `AgentResponder` could grow a `generate(...)` method returning `(text,
  seeds)` without posting, which both production (a future "preview" affordance) and this harness
  could share — not building it now to keep this plan's blast radius to `tests/` + two `scripts/*`
  files.
- **`guards.py`**'s `Judge = Callable[..., Any]` / `GuardVerdict(decision, rationale)` shape and
  `llm.extract_own_line_json_object` (the conservative JSON-verdict extractor already used by the
  production guard judge, `app.py:_build_llm_judge`) are the precedent this plan's `judge.py`
  reuses for parsing the faithfulness/relevance verdict — no new JSON-extraction logic needed.

---

## 3. Design & rationale

**Two-layer split, offline-first (per the method note — not re-derived here):** retrieval metrics
are pure functions over `hybrid_search` output and a pre-cached query vector — no LLM, no network,
runs in the default `pytest` session. The judge layer is the only thing behind `pytest -m live`.

### D1 — Judge model config: explicit env-overridable literal, test-only (recommend, per the brief's ask)

**Chosen:** option (b) — a new `FALKORCHAT_LIVE_JUDGE_MODEL` env var, read directly by the eval
harness, constructing `OpenAICompatibleLLM(LIVE_LLM_BASE_URL, judge_model)` the same way
`test_workflow_live.py` constructs its LLM/embedder — never touching `config.py`/`modelconfig.py`.

**Default (changed in v4): `qwen/qwen3-4b-2507` — the same ref as the agent-under-test.** The v2/v3
default was `openai/gpt-oss-20b`, chosen specifically to satisfy the method note's "never the
4B-under-test judging itself" self-preference-bias guidance (`docs/plans/graphrag-eval-ml.md`
§"Layer 2"). **v4 deliberately deviates from that guidance**, per a stakeholder decision relayed by
the coordinator (not this plan's own call, and not an oversight): the actual hardware this harness
runs on cannot serve a second, larger model alongside the 4B for this workload, even though larger
models are registered/reachable on the same LM Studio instance (`openai/gpt-oss-20b`,
`qwen/qwen3.5-9b`, `google/gemma-4-12b`, `prism-ml/bonsai-27b`) — "registered" is not "usable"
here. Collapsing judge and agent-under-test onto one model instance is the pragmatic response to
that real constraint. **See the new subsection immediately below for the required
methodology/limitation note this deviation carries.**

**Rejected:** (a) wiring a fifth `judge` kind through `ModelGateway` — `modelconfig.py:87`'s `KINDS`
frozenset is explicitly documented as closed ("adding a fifth means adding its own override
property... not a change to this set casually"); doing so touches `_KIND_TO_OVERRIDE_KEY`, every
`config/models.json`-shaped fixture in the test suite, and (per `-graph.md`) the workspace-override
property crosswalk — a product-shaped change with a real design surface, for a capability that
today has exactly **one** consumer: this test-only harness. FR-4's own docstring already sanctions
the test-only direct-construction pattern this plan uses, and `test_workflow_live.py` is a live,
working precedent for exactly this shape. If a *product* need for a judge-kind model ever
materializes (e.g., a live-wired guard-judge model distinct from the workflow LLM), that's a
separate K-0xx with its own config-schema design — not a decision to make as a side effect of an
eval harness.

**Self-preference-bias limitation (added in v4) — accepted, named, and sign-off-gated, not an
oversight.** Because Unit 3's judge and agent-under-test are now the *same model instance*
(`qwen/qwen3-4b-2507`), the faithfulness/relevance judgments Unit 3 produces are structurally
exposed to self-preference bias — the method note's own named risk for exactly this configuration.
This is not silently absorbed into the harness's numbers: **any faithfulness/relevance numbers Unit
3 reports must be read with this caveat**, and — mirroring D6's "Sign-off gate on the first
baseline" pattern rather than inventing new phrasing — **`judge_calibration.json`'s numbers require
an explicit `data-scientist` methodology sign-off before they are treated as trustworthy**, the
same way "the harness ran green" is not, by itself, sufficient for D6's baseline. The sign-off
question here is narrower than D6's ("is this a reasonable floor") but the same shape: is a
same-model judge-vs-agent-under-test agreement number *meaningful at all* given the self-preference
risk, or does it need to be reported as a bounded/qualified number (e.g., "judge agreement under a
same-model configuration — read as a lower-confidence signal, not an independent check") — that
call belongs to `data-scientist`, not to this plan. Until that sign-off happens, Unit 3's judge
numbers should be treated the same way an un-signed-off `retrieval_baseline.json` is treated under
D6: produced, but not yet load-bearing.

### D2 — `ws:eval` is a persistent workspace, not rebuilt per test session

Unlike `ws:test`/`ws:live` (dropped and rebuilt every session/run), `ws:eval` is seeded **once**
(and re-seeded only deliberately, via `scripts/seed_eval_corpus.sh`) and then left alone across
pytest runs. Rationale: the retrieval baseline (§5 Unit 2b) is only meaningful if the corpus it was
measured against is stable — rebuilding it every run (with real embedding calls) would (a) make the
default test suite dependent on a live embedder, violating the "zero network dependency" done
condition, and (b) make "did recall regress" ambiguous between "the retrieval code changed" and
"the corpus embeddings drifted slightly between runs." The retrieval-eval test suite therefore
**skips cleanly** (not fails) when `ws:eval` doesn't exist or doesn't look seeded yet — see §5 Unit
2b's `ws_eval` fixture — rather than bootstrapping it inline the way `conftest.py`'s `_schema`
fixture does for `ws:test`.

### D3 — Golden query embeddings: committed cache, model-ref-keyed invalidation

Per the method note's ask for caching + an invalidation story: `golden_retrieval.jsonl` carries no
vectors (keeps the fixture human-diffable); a sibling `golden_retrieval.embeddings.json` maps
`{queryId: {"model": "<provider>/<model-id>", "vector": [...]}}`, generated by a small **live**,
manually-run script (`server/tests/eval/embed_golden_queries.py`), never by pytest itself. A
network-free pytest test loads this cache and asserts every golden query's cached `model` field
equals `config/models.json`'s current `defaults.embedding` — a mismatch fails loudly ("run
embed_golden_queries.py to refresh the cache for `<ref>`"), rather than silently scoring against a
stale vector. **That comparison reads the real config file via D7 mechanism 1
(`modelconfig.Overlay.load(modelconfig.DEFAULT_MODEL_CONFIG_PATH)`), never
`ModelGateway.from_env()`** — inside a pytest context the latter is redirected to the test-fixture
config (§2's C-1 finding), which would silently validate against the wrong source of truth. This is
the concrete invalidation rule the method note asked for but didn't specify: keyed on
**embedding-model identity**, not on a corpus content hash (a corpus edit doesn't invalidate a
query's own cached embedding — only an embedding-model swap does; a corpus edit instead requires
the golden set's `relevant_msgId`s to be re-verified against the new corpus, a separate, manual
re-verification step the msgId-existence check in §5 Unit 2b catches structurally via "does this
msgId still exist," not automatically for "is it still the *right* answer").

### D4 — Judge-calibration statistic: raw agreement rate, not Cohen's κ

The method note's threshold table says "judge–human agreement ≥ ~0.7" without pinning a statistic.
`docs/archive/plans/m3-guard-calibration.md` (a real precedent in this exact codebase) found Cohen's
κ badly behaved even at N≈21–26 (chance-corrected against a case mix the *author* chose, wide
confidence intervals, a threshold that fails a genuinely good judge 20% of the time). K-026's
calibration set is smaller still (~10 examples per the method note) — κ's problems only get worse.
This plan uses **raw percent exact-agreement per axis** (faithfulness, answer-relevance separately)
as the reported number, with a **mandatory small-N caveat** in the generated report (mirroring
`m3-guard-calibration.md`'s own honesty section): at N≈10, a single disagreement swings the number
by 10 points, and "0.7" is a directional read, not a statistically defensible claim. This is a
narrower, cheaper instrument than the guard-calibration protocol's full sensitivity/specificity
split — appropriate here because K-026's judge axes are not a bias-to-suspend safety guard; a wrong
number is a misleading report, not a stuck workflow run. **Flag for the method note's next review
pass:** it would be worth the note explicitly picking a statistic and a verdict scale (see D5) the
way `m3-guard-calibration.md` did for the guard judge — this plan makes both calls but they are
implementation-level defaults, not re-derivations of the method.

### D5 — Judge verdict scale: binary per axis

The method note doesn't pin faithfulness/relevance to a specific scale (1–5 Likert vs. binary).
This plan uses **binary** per axis — `faithfulness: bool | None` (`None` = the answer abstained /
answered from general knowledge, per the method note's explicit "score grounded-when-relevant
separately from abstained/general — don't penalize a correct 'context didn't help'") and
`relevance: bool`. Binary is the simplest instrument that supports the note's stated threshold
("judge–human agreement ≥ ~0.7") and is what the calibration set (`golden_judge_calibration.jsonl`)
labels; it's also a strict subset of a future ordinal scale (extending later doesn't invalidate
existing labels). Flagged as an assumption, not gospel — same caveat as D4.

### D6 — Baseline file is a real regression gate from day one, not just a record

`server/tests/eval/retrieval_baseline.json` is machine-read, not hand-authored. The retrieval-eval
test: if the file doesn't exist, computes recall@10/recall@5/MRR and **writes** it (first-run
baseline-establish mode, per the method note: "first run establishes the numbers; it does not
gate") and passes; if the file **does** exist, it compares current numbers against it using the
method note's own acceptance rule (recall@10 ≥ baseline **and** MRR not down > 5% relative) and
fails loudly on a real regression. An `UPDATE_EVAL_BASELINE=1` env var (mirroring the `KEEP_WS`
convention) forces a deliberate re-baseline after an intentional retrieval change. This means K-026
itself both produces the baseline **and** wires the exact regression check the backlog's "why now"
motivates ("today those would ship unmeasured") — no separate future unit has to build the gate.

**Sign-off gate on the first baseline (coordinator-accepted, `docs/reviews/graphrag-eval.md` M-4):**
"the test passed" is not, by itself, sufficient to treat the *first* committed
`retrieval_baseline.json` as gating for every future run — whether these specific numbers are a
reasonable floor is a methodology call (the corpus behind them is `analyst`-reviewed, not
human-verified, and corpus representativeness is the method note's own highest-ranked risk; a
mediocre first-run number would otherwise silently block a genuinely better future retrieval
change). **The first commit of `retrieval_baseline.json` therefore gets an explicit
`data-scientist` methodology sign-off** — is this a reasonable floor, not just "the harness ran
green" — before it's treated as gating. See §5 Unit 2b's "done when."

### D7 — Real-config reads inside a pytest context bypass `ModelGateway` entirely (resolves C-1)

Two distinct needs this plan has, and two distinct mechanisms for them — neither adds a new
fixture/escape hatch, and neither goes through `ModelGateway`:

1. **Reading a declared value from the real `config/models.json`** (the embedding dimension
   `ws_eval`'s readiness check needs, §5 Unit 2b; D3's model-ref comparison, §5 Unit 2a) — read the
   overlay file directly: `modelconfig.Overlay.load(str(modelconfig.DEFAULT_MODEL_CONFIG_PATH))`.
   `DEFAULT_MODEL_CONFIG_PATH` is a module-level `Path` constant (`modelconfig.py:118`) —
   `conftest.py`'s monkeypatch only ever touches `config.MODEL_CONFIG_PATH`/the env var, never this
   constant, so it's untouched regardless of pytest context. `Overlay.load` reads only the
   falkor-chat overlay file (it never touches `FALKORCHAT_OPENCODE_CONFIG`/the provider catalog),
   so it works standalone with no other config file present. `.default_for("embedding")` gives the
   ref; `.model_settings(ref).get("dim")` gives its declared dimension. Zero network, zero client
   construction — just a plain read of the same JSON file production reads, bypassing the
   redirected env var entirely.
2. **Constructing a live client inside a pytest test** (Unit 3's agent-under-test and judge LLMs) —
   construct `OpenAICompatibleLLM` directly from env-var literals, exactly as
   `test_workflow_live.py` already does (`LIVE_LLM_BASE_URL`/`LIVE_LLM_MODEL`, plus this plan's new
   `LIVE_JUDGE_MODEL`) — never via `ModelGateway` or `StaticModelGateway`.

**Rejected:** a new test-only fixture that restores the real `config/models.json` path for
`tests/eval/` specifically (the review's other suggested option). A third config-resolution path
alongside "the redirected default" and "construct-from-env-literal" is extra maintenance surface
for no benefit here — every real-config need in this harness is satisfiable with a plain file read
(mechanism 1) or the already-established live-literal client pattern (mechanism 2); nothing in
this harness needs the *full* `ModelGateway` resolution machinery (role expansion, workspace
overrides, fallback chains) at all.

**The asymmetry with Unit 1 is intentional, not an inconsistency to "fix" toward either
direction.** Unit 1's `seed_eval_corpus.py` is a bare script, never collected by pytest — `
conftest.py`'s autouse fixture never touches it — so `ModelGateway.from_env()` there correctly
resolves against the real `config/models.json` and is the *more* correct choice (§2). Units 2a/2b/3
are pytest tests under `server/tests/`, inherit the autouse redirection unconditionally, and must
never call `ModelGateway.from_env()` for anything that needs to reflect the real config.

---

## 4. Corpus design (`ws:eval`, Unit 1)

Target: **10–12 topical threads, 8–12 messages each (~110–130 messages total)**, spread across a
handful of channels, so a paraphrased query has genuine distractors. Concretely:

- At least **3 pairs of topically-adjacent threads** (shared vocabulary/domain, different specifics)
  to reproduce the method note's own near-miss example (an orthogonal seed at cosine distance
  0.786) — e.g., two separate incident threads about *different* services with similar
  "deploy → regression → rollback → root cause" shape, two architecture-decision threads in the
  same subsystem, two auth-adjacent threads (token refresh vs. session timeout).
- At least **2–3 clearly orthogonal topics** (e.g., an HR/logistics thread, a product-planning
  thread) so the golden set also has "easy" queries — a corpus of only near-misses would bias
  recall pessimistically and not resemble real usage either.
- Each thread should read like a real chat: 8–12 turns, mixed `user`/`assistant` roles, specific
  facts (service names, versions, dates, numbers) a paraphrased query can target unambiguously —
  the same property `test_workflow_live.py`'s own 5-message `CORPUS` constant already demonstrates
  at small scale (`server/tests/test_workflow_live.py:96-105`), just scaled up and multi-topic.

**Authoring workflow (flag, not resolve — per the brief):** the method note calls for
"LLM-drafts-then-verifies." In this agent-only delivery pipeline, "verify" is satisfied by an
independent **`analyst`** review pass over the drafted corpus (checking topical spread, near-miss
pairing, and that no message content leaks a golden-query answer key) — not a literal human. This
substitution is surfaced here explicitly; it is not this plan's call to resolve, and the downstream
coordinator should decide whether that's an acceptable stand-in or whether a real human spot-check
is warranted before `ws:eval` is treated as load-bearing for the frozen baseline. (§7 item 3
strengthens this for the judge-calibration set specifically, per the plan-gate review's M-5.)

**Corpus provenance recorded, not just the corpus:** the seed script (§5 Unit 1) should print (and
the generated report, §5 Unit 3, should carry) a short provenance line — message count, thread
count, the embedding model ref + dimension used, and the seed date — so a later reader of the
baseline knows what it was measured against without re-deriving it from the graph.

---

## 5. Step-by-step implementation

**Four units** (Unit 2 split into 2a/2b — the original single Unit 2 spanned 8 files, past this
project's ~5-file dispatch-sizing signal; `docs/reviews/graphrag-eval.md` M-2), each independently
dispatchable and independently reviewable:

- **Unit 2a** depends on Unit 1 having been *run* only for its one live-corpus check (see below) —
  its other files/tests are genuinely network/DB-free and can be written, reviewed, and run before
  Unit 1's corpus exists.
- **Unit 2b** depends on Unit 2a's `golden_retrieval.jsonl` schema and `golden_retrieval.embeddings.json`
  existing (for its own code to be written against), not on their content being final/reviewed; it
  owns the `ws_eval` conftest fixture that Unit 2a's one live check also reuses.
- **Unit 3** depends on Unit 2a's `golden_retrieval.jsonl` existing (it samples a subset from it),
  but not on Unit 2b's tests having actually run.

### Unit 1 — `ws:eval` corpus seed (step 0)

**Files:**
- `scripts/seed_eval_corpus.sh` (new) — thin bash wrapper, mirrors `scripts/load_test.sh`'s role:
  checks FalkorDB reachable (`redis-cli PING`), checks the `server/.venv` exists, `exec`s the
  Python script below with `$@` passed through. Env: `FALKORDB_HOST`/`FALKORDB_PORT`,
  `EVAL_WS` (default `eval`), `RESEED=1` (force a full `GRAPH.DELETE ws:eval` + rebootstrap —
  the deliberate reset path when the embedding model or dimension changes).
- `scripts/seed_eval_corpus.py` (new) — the real logic, run via `server/.venv/bin/python` (never
  collected by pytest — `ModelGateway.from_env()` is correct here, D7):
  1. `gateway = modelconfig.ModelGateway.from_env()`; resolve `dim = gateway.resolve("embedding").primary.dim or config.EMBEDDING_DIM`.
  2. If `RESEED` or `repo.read_index_dimension(ws, label="Message") != dim`: `db.connect().select_graph(f"ws:{ws}").delete()` (ignore "doesn't exist"), then `subprocess.run(["bash", ".../bootstrap_schema.sh", ws], env={**os.environ, "EMBEDDING_DIM": str(dim)})` — mirrors `test_workflow_live.py`'s `live_ws` fixture almost exactly, but as a standalone script, not a pytest fixture.
  3. `repo.ensure_user(...)` / `repo.ensure_agent(...)` for the fixed corpus-author ids.
  4. For each topic: `MERGE`-idempotent Channel/Thread (small inline Cypher via `graph.query(...)`, fixed ids — mirrors `seed_demo.sh`'s pattern, not `repository.create_channel`/`create_thread`, which are non-idempotent `CREATE`s per §2's finding).
  5. For each message (fixed `msgId`, e.g. `f"eval-{topic_slug}-{n:03d}"`): pick the first-vs-subsequent write via **`repo.thread_has_head(ws, thread_id=...)`** (the same primitive `services._dispatch_write` uses for this exact decision, `services.py:704` — reuse it directly, don't re-derive an equivalent check), then call `repo.post_first_message`/`post_subsequent_message` accordingly — reruns are safe no-ops via the `dupMsg` status.
  6. For each message not yet embedded (`MATCH (m:Message {msgId:$id}) RETURN m.embedding IS NOT NULL` pre-check, skip if true): `EmbeddingWorker(repo, models=gateway, expected_dim=dim).embed_message(ws, msg_id=..., text=...)` — the **real** embedding path, per the method note's explicit requirement.
  7. Print the provenance line (§4) at the end.
- `scripts/seed_eval_corpus.py`'s `_CORPUS` data (the ~110–130 messages across ~10–12 topics, §4) —
  drafted by the implementer, reviewed by `analyst` per the corpus-review caveat above, before Unit
  2a's golden set is drafted against it.

**Done when:** `./scripts/seed_eval_corpus.sh` run against a live FalkorDB + LM Studio produces a
populated, idempotently-re-runnable `ws:eval`; a second run does no redundant embedding calls
(verify by re-running with a log line or counter showing 0 new embeds).

### Unit 2a — Golden-set authoring

**Files:**
- `server/tests/eval/golden_retrieval.jsonl` (new) — 30–50 lines, schema:
  ```json
  {"id": "gr-01", "query": "<paraphrased user-style question>",
   "relevant_msgIds": ["eval-checkout-v42-003"], "topic": "checkout-v42-incident",
   "target_text": "<verbatim text of the eval-checkout-v42-003 Message>",
   "rationale": "<why this message answers this query>"}
  ```
  **`target_text` (added in v3, per `docs/reviews/graphrag-eval.md` Pass 2 M-3):** a verbatim copy
  of the first-listed `relevant_msgIds` entry's `Message.text`, authored alongside the pair at
  golden-set-authoring time — not a convenience, but the **self-containment mechanism for the
  leakage-inflation guard** (method note finding 5/risk 2): the check that a paraphrased query
  never verbatim-matches its target message must not itself depend on a live `ws:eval` read to get
  that target text, or the guard silently reintroduces the exact dependency Unit 2a exists to avoid.
  For a multi-`relevant_msgIds` pair, `target_text` covers the first (primary) id; the check below
  is scoped accordingly. Drafted against Unit 1's corpus (LLM-drafted, `analyst`-reviewed per the
  same caveat as §4 — flagged, not resolved, here too: "human-verified" in the method note is
  satisfied by that same independent-agent-review gate). Every query paraphrased, never a verbatim
  substring of its target message (enforced structurally by this unit's integrity test against
  `target_text`, not just by authoring discipline).
- `server/tests/eval/embed_golden_queries.py` (new) — **live**, manually-run, not collected by
  pytest (its filename matches neither `test_*.py` nor `*_test.py`, and it's never imported by a
  pytest test): reads `golden_retrieval.jsonl`, for each query missing a cache entry (or whose
  cached `model` doesn't match the current `config/models.json` default), calls
  `ModelGateway.from_env().embedder("embedding")` — correct here for the same reason as Unit 1's
  script (D7's asymmetry note: it's a bare script, not a pytest test) — and writes/updates
  `golden_retrieval.embeddings.json` (D3).
- `server/tests/eval/golden_retrieval.embeddings.json` (generated artifact, committed) —
  `{queryId: {"model": "<ref>", "vector": [...]}}`.
- `server/tests/eval/test_golden_set_integrity.py` (new) — **genuinely network/DB-free**, needs
  neither FalkorDB nor `ws:eval` for **either** of its two checks (resolves the plan's first
  draft's self-contradiction, `docs/reviews/graphrag-eval.md` Pass 1 M-3, and closes the Pass 2
  residual on that same finding — the msgId-existence check moved to Unit 2b in v2, and the
  self-retrieval check below now compares against the fixture's own `target_text` field rather
  than a live `Message.text` read, so no check in this file touches `ws:eval` at all):
  - no `query` is a case-insensitive substring of, or a superstring containing, its own pair's
    `target_text` (self-retrieval-inflation guard, method note finding 5/risk 2) — a pure string
    comparison against the fixture, never a graph read;
  - every golden query id has a cache entry in `golden_retrieval.embeddings.json` whose `model`
    matches `config/models.json`'s current `defaults.embedding`, read via **D7 mechanism 1**
    (`Overlay.load(DEFAULT_MODEL_CONFIG_PATH)`) — never `ModelGateway.from_env()`.

**Confirmed (v3):** with `target_text` in the schema, Unit 2a's "genuinely network/DB-free" claim
now holds for **both** checks in `test_golden_set_integrity.py`, not just the embedding-cache-match
one — neither needs FalkorDB reachable nor `ws:eval` seeded. §7 item 10's "Unit 2a's own code
(fixtures + integrity test) can be written and run before Unit 1 exists at all" is therefore
accurate for the whole file now, not just part of it (see that item's updated wording below).

**Done when:** `cd server && .venv/bin/python -m pytest tests/eval/test_golden_set_integrity.py -q`
passes with zero network/FalkorDB dependency, regardless of whether `ws:eval` exists yet — provable
by running it before Unit 1 has ever been run.

### Unit 2b — Retrieval metrics module + baseline gate

**Files:**
- `server/tests/eval/metrics.py` (new) — pure, network-free functions:
  `recall_at_k(retrieved_msg_ids: list[str], relevant: set[str], k: int) -> float`,
  `mrr(retrieved_msg_ids: list[str], relevant: set[str]) -> float` (0.0 if no hit within the
  returned rows — handles the "ANN may return fewer than k" caveat naturally since it operates on
  whatever list it's given).
- `server/tests/eval/conftest.py` (new, scoped under `tests/eval/`) — `ws_eval` fixture (module or
  session scope): probes FalkorDB reachable, `repo.read_index_dimension(EVAL_WS, label="Message")`
  is not `None` and matches the configured embedding dim (read via **D7 mechanism 1**, not
  `ModelGateway.from_env()`), and a minimum message count (e.g. `MATCH (m:Message) RETURN count(m)`
  ≥ 50) — else `pytest.skip("ws:eval not seeded — run ./scripts/seed_eval_corpus.sh")`. This is the
  D2 seam: no bootstrapping happens here, only a reachability-and-readiness probe, mirroring
  `test_workflow_live.py`'s skip-never-fail posture but without the rebuild.
- `server/tests/eval/test_retrieval_eval.py` (new) — two test functions, both needing `ws_eval`:
  - `test_golden_msgids_exist_in_corpus` (moved here from Unit 2a per M-3): every `relevant_msgId`
    referenced by `golden_retrieval.jsonl` actually exists in `ws:eval` (a light `WHERE m.msgId IN
    $ids` lookup) — catches a golden pair drifting out of sync with the corpus. Reuses the same
    `ws_eval` fixture as the metrics test below, so it skips cleanly (not fails/errors) whenever
    `ws:eval` isn't seeded yet — the "empty key" `ResponseError` pattern `repository.py:740-743`
    shows other code already has to guard against explicitly is exactly what this fixture exists
    to avoid hitting directly.
  - the retrieval-metrics test: for each golden pair, calls `Services.hybrid_search(ctx,
    q_vec=cached_vector, k=10, limit=10, channel_id=None)` once, computes recall@10 over the full
    result, recall@5 over the first 5 (same ordered list, sliced — one graph round-trip per query),
    MRR over the same list. Aggregates recall@10/recall@5/MRR across all golden pairs (mean). Then
    D6's baseline-compare-or-establish logic against `retrieval_baseline.json`.
- `server/tests/eval/retrieval_baseline.json` — **not hand-authored**; produced by the first
  passing run of `test_retrieval_eval.py` against a real, fully-seeded `ws:eval`, then committed
  **after a `data-scientist` methodology sign-off** (D6's sign-off gate) that these specific numbers
  are a reasonable floor, not just "the test passed."

**Done when:** `cd server && .venv/bin/python -m pytest tests/eval -q` (default marker selection,
i.e. `-m "not live"`) passes with `ws:eval` seeded — the retrieval test reports and (pending
`data-scientist` sign-off, D6) commits the baseline; the same command with `ws:eval` absent skips
both tests in this file cleanly (not an error) rather than failing. **The `data-scientist` sign-off
on the first `retrieval_baseline.json` is part of this unit's done-condition, not a follow-up** —
the file is committed only after that review, per D6.

### Unit 3 — Judge layer + calibration + live tests + report

**Files:**
- `server/tests/eval/golden_judge_calibration.jsonl` (new) — ~10 lines, schema:
  ```json
  {"id": "jc-01", "question": "...", "context": ["<retrieved seed text>", "..."],
   "answer": "...", "expected_faithfulness": true, "expected_relevance": true,
   "label_rationale": "..."}
  ```
  Fixed question/context/answer triples (not regenerated live) so the calibration run isolates
  judge variance from generation variance — same "extract-then-judge, evidence fixed" spirit as
  `guards.evaluate_guard`'s injected-judge design. Same human-verification caveat as §4/Unit 2a,
  surfaced again here because it's a separate ~10-example set, not a subset of the golden-retrieval
  set's own verification — **and strengthened here specifically: §7 item 3 recommends a real human
  spot-check for this set** (`docs/reviews/graphrag-eval.md` M-5, coordinator-accepted), even though
  the larger golden-retrieval set stays on the `analyst`-review path.
- `server/tests/eval/judge.py` (new) — `JudgeVerdict` dataclass (`faithfulness: bool | None`,
  `relevance: bool`, `rationale: str`) mirroring `guards.GuardVerdict`'s shape; a prompt-builder
  function; a parser built on `llm.extract_own_line_json_object` (reused, not reimplemented) with
  the same bias documented for the guard judge — an unparseable/ambiguous verdict should resolve to
  the *conservative* reading (`faithfulness=None` "couldn't tell," not a manufactured `True`).
- `server/tests/eval/test_judge_live.py` (new) — `pytestmark = pytest.mark.live`; adds
  `FALKORCHAT_LIVE_JUDGE_MODEL` (default `qwen/qwen3-4b-2507` **— the same ref as
  `FALKORCHAT_LIVE_LLM_MODEL`, per D1's v4 stakeholder-directed change**; the v2/v3 default,
  `openai/gpt-oss-20b`, was chosen specifically to satisfy the method note's "never the model
  judging itself" rule — v4 deliberately trades that safeguard away for a real hardware constraint,
  see D1's methodology/limitation note) alongside the three existing `FALKORCHAT_LIVE_*` vars.
  Fixture mirrors `test_workflow_live.py`'s `live_dim`/reachability-skip shape, extended to also
  probe the judge model is loadable (a `.complete()` no-op call) — skip with a clear reason if not.
  **This fixture behavior is unchanged by v4**: since the judge model now defaults to the exact
  same ref already being probed/loaded as the agent-under-test, the probe is not a *new*
  reachability dependency — a live run that already needs the 4B loaded needs nothing additional
  for the judge role. (An operator who overrides `FALKORCHAT_LIVE_JUDGE_MODEL` back to a distinct,
  stronger model still gets an independent probe/skip for it, unaffected by this change.) **This
  test is a pytest test under `server/tests/`, so it inherits `conftest.py`'s autouse redirection
  like every other test here — both LLM clients below are therefore constructed directly from
  env-var literals (D7 mechanism 2), never via `ModelGateway`/`StaticModelGateway`, which would
  silently answer with whatever `server/tests/data/models.json`'s `defaults.agent` names instead of
  the real M2 stack model.**
  - **Generation sub-pass:** ~15–20 items sampled from `golden_retrieval.jsonl` (a fixed subset,
    e.g. the first 20 by id, so the sample is stable across runs). For each: embed the query
    (live), `Services.hybrid_search` against `ws:eval` (**read-only** — no write, so this never
    mutates the corpus the retrieval baseline depends on), build the prompt via a throwaway
    `AgentResponder(services=None, agent_id=config.AGENT_ID)` instance's
    `_build_prompt(question, seeds)` (reused per §2's corrected finding — this already embeds the
    module-level `_SYSTEM_PROMPT` constant in its returned message list; construction is
    side-effect-free), then call the **agent-under-test** LLM's `.complete(...)` — that LLM
    constructed directly from `FALKORCHAT_LIVE_LLM_BASE_URL`/`FALKORCHAT_LIVE_LLM_MODEL` (D7
    mechanism 2). Then judge the resulting `(question, context, answer)` triple with the judge
    model (constructed from its own env-var literal, same mechanism — as of v4 this defaults to
    the **same ref** as the agent-under-test client above, so two separate client instances are
    still constructed, but they name the same model; see D1's self-preference-bias
    methodology/limitation note).
  - **Calibration sub-pass:** the 10 `golden_judge_calibration.jsonl` triples, judged directly (no
    generation) against their `expected_faithfulness`/`expected_relevance` labels. Computes raw
    percent exact-agreement per axis (D4).
  - Writes `server/tests/eval/judge_calibration.json` (agreement numbers, judge model ref, **and,
    as of v4, an explicit `sameModelAsAgentUnderTest: true` field** whenever the resolved judge ref
    equals the resolved agent ref — the machine-readable carrier of D1's self-preference-bias
    caveat, so the report step below never has to re-derive it from two separate env-var strings)
    — plus sample size, timestamp — **only on an actual completed live run** — never on skip, so
    its *absence* is the unambiguous "not run" signal the report step reads.
- `server/tests/eval/generate_report.py` (new) — **not a pytest test**, manually/CI invoked after
  both suites: reads `retrieval_baseline.json` (required — errors clearly if absent, meaning Unit
  2b was never run against a seeded corpus) and `judge_calibration.json` (optional). Writes
  `docs/test-reports/graphrag-eval-<date>.md` containing, per the backlog's exact done-condition:
  recall@10/recall@5/MRR baseline numbers; judge-human agreement per axis **or** an explicit
  "not run (live LLM unreachable at last suite execution)" marker if the file is absent; corpus
  provenance (§4's line, re-read from a small provenance sidecar Unit 1's seed script writes,
  e.g. `server/tests/eval/corpus_provenance.json`); golden-set size and the self-retrieval-guard
  pass/fail; the D4 small-N caveat verbatim when judge numbers are present; **and, as of v4, when
  `judge_calibration.json`'s `sameModelAsAgentUnderTest` is `true`, a mandatory, clearly-labeled
  self-preference-bias caveat verbatim (D1) alongside an explicit note on whether the required
  `data-scientist` sign-off (D1) has happened yet — mirroring how the report already carries D6's
  baseline sign-off status, not a new report mechanism.**

**Done when:** `pytest -m live -s` (with LM Studio + `ws:eval` up) runs both sub-passes and writes
`judge_calibration.json`; with LM Studio down, the same command skips cleanly with a reason naming
which dependency was unreachable; `generate_report.py` produces a correct report in both cases
(with an explicit not-run marker in the second); `ws:eval`'s message count is unchanged before/after
a live run (verifiable via the same `MATCH (m:Message) RETURN count(m)` the Unit 2b fixture uses).

---

## 6. Test strategy

| Suite | Marker | Network? | What it proves |
|---|---|---|---|
| `test_golden_set_integrity.py` (Unit 2a) | none (default) | none — genuinely, for both its checks (self-retrieval compares against the fixture's own `target_text`, added in v3; no `ws:eval` read anywhere in this file) | Golden fixture internal consistency: no verbatim self-retrieval, embedding cache is current for the real configured model (read via D7, not `ModelGateway`). |
| `test_retrieval_eval.py` (Unit 2b) | none (default) | FalkorDB only (already assumed reachable by the whole default suite per `AGENTS.md`) | `test_golden_msgids_exist_in_corpus`: every golden `relevant_msgId` is present in `ws:eval`. The metrics test: recall@10/recall@5/MRR computed and either establish or check against the committed baseline. Both skip (not fail) if `ws:eval` isn't seeded. |
| `test_judge_live.py` (Unit 3) | `live` | FalkorDB + LM Studio (agent model + judge model, both constructed from env-var literals, D7 — **as of v4, both default to the same ref, `qwen/qwen3-4b-2507`**, so this is one loaded model serving two roles, not two live dependencies) | Faithfulness/relevance over a live-generated Q&A sample, plus judge-vs-human agreement over the fixed calibration set; skips cleanly on either dependency being down. **Numbers carry a self-preference-bias caveat (D1) and are not trustworthy until `data-scientist` sign-off, mirroring D6.** |

Edge cases to cover explicitly in Unit 2b/3 tests:
- A golden query whose `relevant_msgIds` has more than one id (multi-relevant) — recall/MRR must
  handle the set case, not assume a singleton.
- `hybrid_search` returning fewer than `k` rows (ANN's documented non-guarantee) — recall/MRR must
  not index-error or silently overcount.
- A judge response that fails to parse as JSON — must resolve to the conservative `None`/`False`
  reading (per `judge.py`'s design above), and must be visible in the report as a parse-failure
  count, not silently dropped from the agreement denominator.
- The generation sub-pass's retrieved context is empty (e.g. a query with no strong corpus match) —
  the responder's own "answer from general knowledge" path fires; faithfulness must resolve to
  `None` (abstained/general), never scored against non-existent context, matching the method note's
  explicit "don't penalize a correct 'context didn't help'."

---

## 7. Risks & open questions

1. **C-1 (autouse config redirection) — resolved, not deferred.** `docs/reviews/graphrag-eval.md`'s
   Critical finding is fixed by D7: every real-config read inside a pytest context uses a direct
   file read (`Overlay.load(DEFAULT_MODEL_CONFIG_PATH)`) or an env-var-literal client construction,
   never `ModelGateway.from_env()`. Unit 1's script is the one deliberate exception, and it's safe
   because it's never collected by pytest.
2. **D1 (judge-kind config wiring) — resolved, not deferred.** Recommendation: option (b),
   explicit env-overridable literal, test-only. Justification in §3 D1. If a product need for a
   live-wired judge model materializes later, it's a new backlog item with its own config-schema
   design, not a retrofit of this harness. **The config-wiring mechanism itself is unchanged by
   v4** — only D1's *default value* for `FALKORCHAT_LIVE_JUDGE_MODEL` changed (`qwen/qwen3-4b-2507`,
   same ref as the agent-under-test, replacing `openai/gpt-oss-20b`), per a stakeholder
   hardware-constraint decision the coordinator relayed. See §3 D1's methodology/limitation note
   for the self-preference-bias consequence of that value change, and item 8 below.
3. **Corpus authoring + golden-pair/calibration "human verification" — flagged, with one
   strengthened recommendation (per the brief and `docs/reviews/graphrag-eval.md` M-5,
   coordinator-accepted).** All three "human" touchpoints (corpus review, golden-pair verification,
   the ~10-example calibration labels) are, in this agent-only pipeline, satisfied by an
   independent `analyst` review pass rather than a literal human. For the corpus and the 30–50
   golden-retrieval pairs, that substitution is well-matched to what needs checking (topical
   spread, near-miss pairing, no answer-key leakage) and this plan does not push further than
   flagging it. **For the ~10-example `golden_judge_calibration.jsonl` set specifically, this plan
   recommends a real human spot-check before its judge-vs-human agreement numbers are trusted at
   face value** — that number's entire meaning depends on the labels actually being human
   judgments, an LLM-based judge and an LLM-based reviewer can share failure modes a human
   wouldn't, and at ~10 examples it's the cheapest of the three touchpoints to do for real. Still
   the coordinator's/user's call to act on, not a hard gate this plan enforces.
4. **D4/D5 (agreement statistic, verdict scale) are this plan's defaults, not the method note's
   specification.** The note leaves both open; §3 picks raw percent-agreement and a binary scale
   with explicit justification. Flagged for the method note's next review pass as a light addendum
   opportunity — not a re-derivation, and not blocking this plan.
5. **Corpus representativeness is a judgment call by construction** (method note risk #1,
   "highest"). This plan sizes and structures the corpus (§4) to reproduce the note's own
   near-miss finding, but "is this corpus representative enough" is inherently not fully
   verifiable in advance — the `analyst` corpus-review gate (§4) is the best available check, not
   a guarantee. The D6 sign-off gate (item 9 below) is the harness-level backstop against this risk
   silently becoming load-bearing.
6. **`ws:eval` is a new persistent, shared-instance workspace** (D2) — unlike every other
   throwaway eval workspace in this codebase (`ws:test`, `ws:live`, `ws:load`), it is meant to
   survive across sessions so the baseline stays meaningful. This is a deliberate deviation from
   the repo's usual "rebuild every run" convention, and it means `ws:eval` needs the same
   operational awareness as `ws:acme` (don't `test_queries.sh`-style wipe it, don't let an
   unrelated script `GRAPH.DELETE` it). Its vector-index RAM cost at ~120 messages (~12.5 KB/msg
   per the backlog's own K-008 line item) is well under 2 MB — trivial, called out per the
   backlog's RAM-callout convention anyway.
7. **An embedding-model swap is now a two-location change** (`docs/reviews/graphrag-eval.md` N-1).
   D3's "mismatch fails loudly" is correct in isolation, but once `ws:eval`/
   `golden_retrieval.embeddings.json` exist, swapping `config/models.json`'s `defaults.embedding`
   for reasons unrelated to this harness will red the *default, network-free* `pytest -q` run
   (`test_golden_set_integrity.py`) until someone separately re-runs `embed_golden_queries.py` (and
   re-seeds `ws:eval` at the new dimension, per D2). Nothing in the codebase today connects those
   two facts. Recommend a callout in `AGENTS.md`'s model-config section pointing here when an
   embedding-model swap is made — not this plan's own file to edit, but worth the coordinator
   tracking as a small follow-up.
8. **Judge-model choice for the live run is a stakeholder-directed default (changed in v4), not an
   environment capability limit.** `FALKORCHAT_LIVE_JUDGE_MODEL` now defaults to
   `qwen/qwen3-4b-2507` — the same ref as the agent-under-test — **not** because a stronger model
   isn't registered on the LM Studio instance (several are: `openai/gpt-oss-20b`,
   `qwen/qwen3.5-9b`, `google/gemma-4-12b`, `prism-ml/bonsai-27b`), but because the actual hardware
   can't run one of them alongside the 4B for this workload; the user made this trade-off directly,
   aware it drops the method note's self-preference-bias safeguard (D1). Two direct consequences
   this plan tracks, not just states: (a) any faithfulness/relevance numbers Unit 3 produces need
   the explicit self-preference-bias caveat (D1, carried machine-readably by
   `judge_calibration.json`'s `sameModelAsAgentUnderTest` field) and a `data-scientist` sign-off
   before they're treated as trustworthy, mirroring D6's baseline gate; (b) the harness still must
   (and does, via the skip-on-unreachable fixture, unaffected by this change per §5 Unit 3) work
   correctly when no judge-capable model is loaded at all, and an operator who *can* run two
   distinct models is free to override `FALKORCHAT_LIVE_JUDGE_MODEL` back to a stronger one — the
   env var stays the seam, only the shipped default changed.
9. **D6's `data-scientist` sign-off gate on the first `retrieval_baseline.json`** (item 5 above,
   `docs/reviews/graphrag-eval.md` M-4, coordinator-accepted) is part of Unit 2b's done-condition,
   not optional follow-up work — the file is committed only after that review affirms the numbers
   are a reasonable floor.
10. **Sequencing note for the downstream coordinator:** Unit 1 (corpus) should be reviewed
    (`analyst`) before Unit 2a's golden set is drafted against it — drafting golden pairs against a
    corpus that then gets revised would require re-verifying every pair. **Unit 2a's own code
    (fixtures + integrity test) can be written and run before Unit 1 exists at all, in full — both
    of its checks are genuinely network/DB-free as of v3's `target_text` field (Pass 2 M-3), not
    just the embedding-cache-match one** — only the golden pairs' *content* (the actual queries and
    `relevant_msgIds`/`target_text` values) needs the corpus first to be meaningful, not the test
    code itself. Unit 3 can be written in parallel with Unit 2b (it only needs Unit 2a's
    `golden_retrieval.jsonl` to exist, not Unit 2b's tests to have actually run), but its own live
    test naturally can't produce real numbers until `ws:eval` is seeded. Unit 2b depends on Unit
    2a's fixture files existing (schema-wise), not on their content being final.

---

## Ready to implement

Plan: `falkor-chat/docs/plans/graphrag-eval.md` (this document, v4). Four dispatchable units:

1. **Unit 1 — corpus seed**: `scripts/seed_eval_corpus.sh` + `scripts/seed_eval_corpus.py`
   (~110–130 messages across 10–12 topics into a new persistent `ws:eval`, real embeddings via
   `ModelGateway`, idempotent re-run, reusing `Repository.thread_has_head`). Gate: `analyst`
   review of corpus content before Unit 2a.
2. **Unit 2a — golden-set authoring**: `server/tests/eval/golden_retrieval.jsonl` (schema now
   includes `target_text`, v3), `embed_golden_queries.py`, `golden_retrieval.embeddings.json`,
   `test_golden_set_integrity.py`. Genuinely network/DB-free for both its checks; runnable in full
   before Unit 1 exists.
3. **Unit 2b — retrieval metrics + baseline gate**: `metrics.py`, `conftest.py` (`ws_eval`
   skip-fixture), `test_retrieval_eval.py` (msgId-existence check + the recall/MRR test), the
   `retrieval_baseline.json` artifact it produces. Gate: `data-scientist` methodology sign-off on
   the first committed baseline (D6), part of this unit's done-condition.
4. **Unit 3 — judge layer + report**: `golden_judge_calibration.jsonl` (recommend a real human
   spot-check on this ~10-example set — §7 item 3), `judge.py`, `test_judge_live.py`
   (`pytest.mark.live`, `FALKORCHAT_LIVE_JUDGE_MODEL` env var **now defaulting to
   `qwen/qwen3-4b-2507` — same ref as the agent-under-test, v4 stakeholder-directed change**,
   D1/D4/D5/D7), `generate_report.py` → `docs/test-reports/graphrag-eval-<date>.md`.

Key design calls an implementer needs to hold onto: no `judge` kind added to `ModelGateway` (D1);
**Unit 3's judge model defaults to the same model as the agent-under-test as of v4 — a
stakeholder-directed, hardware-constrained trade-off that deliberately drops the method note's
self-preference-bias safeguard, carried by an explicit caveat + `data-scientist` sign-off
requirement before those numbers are trusted (D1's methodology/limitation note — mirror D6's
pattern, don't re-derive)**; `ws:eval` is persistent, not rebuilt per run (D2); golden query
embeddings are a committed, model-ref-keyed cache (D3); judge agreement is raw percent, not κ, with
a mandatory small-N caveat (D4); the baseline file is a live regression gate from first commit,
sign-off-gated (D6); **every real-config read or live-client construction inside a pytest test
bypasses `ModelGateway` entirely — direct file read or env-var-literal construction, never
`ModelGateway.from_env()`/`StaticModelGateway` (D7)**, because `server/tests/conftest.py`'s
autouse fixture silently redirects it to test-fixture config for every test in the suite. Open
items 1–10 in §7 are this plan's explicit flags for the coordinator — none block starting Unit 1.
