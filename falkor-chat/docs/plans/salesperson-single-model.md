# Salesperson demo on one LLM — `query_graph_data` inherits the calling step's model

> **Status:** archived · **Owner:** `architect` · **Tracks:** K-066 · **Version:** 3 · **Reviews:** `falkor-chat/docs/reviews/salesperson-single-model.md`

_Revision 2026-10-21 (v2): analyst Pass 1 + teco rulings applied; §2.3, §4 (S5/S8/S9), §5 (T7), §6, §7 and §8 revised, with dispositions in §9._
_Revision 2026-10-21 (v3): analyst Pass 2, U0's final fit report and the stakeholder's "build it, accept WDDM paging" decision applied; §7, §8, §9 only (§3–§5 and S1–S9 unchanged)._

Coordination: `falkor-chat/docs/plans/salesperson-single-model-coordination.md`.

## 1. Goal & scope

Run the salesperson storefront demo on **one chat LLM**, `lmstudio/mistralai/ministral-3-3b`, plus
the embedder: the `assistant` step already pins ministral via `SALESPERSON_DEF`'s `config.model`;
this plan makes the `query_graph_data` tool's internal structured-completion call use the
**same model as the agent step that called it**, instead of the global `step` default
(`lmstudio/qwen/qwen3-4b-2507`). Motivation: 6 GB RTX 4050 Laptop; ministral (≈4.2 GiB) + qwen
(≈2.3 GiB) + embedder (≈0.6 GB) spill to RAM; ministral + embedder is ≈5.1 GB of weights.

**Out of scope:** any change to `falkor-chat/config/models.json`; any `WorkspaceConfig` write; a
`SALESPERSON_DEF` version bump; the K-066 context-overflow fix itself (this plan *measures* it,
§7); the `:8200` process's own model use (its `extraction` default stays `qwen`, see §2.1 and
risk R2); demo bring-up automation (`start_demo.sh` preflight — K-066 item (1), `devops`).

## 2. Findings

CPG: considered, not relevant — `cpg_falkorchat` is stale (8 commits to `falkor-chat/server`
since its 2026-10-15 build) and every question here was a handful of call sites, answered by
reading `modelconfig.py`/`tools.py`/`executor.py` directly.

### 2.1 Is the "shared defaults" premise true? — Half true

- **The defaults file is shared: true.** `falkor-chat/scripts/start_agent_team.sh` (the `:8200`
  `ws:agent-team` server) deliberately does **not** set `FALKORCHAT_MODEL_CONFIG`, so it reads the
  same `falkor-chat/config/models.json` as `start_demo.sh`/`start_server.sh`. Its header says so
  and warns if the env var is set.
- **But `:8200` does not consume the `step` kind.** It runs with `FALKORCHAT_WORKFLOW_ENABLED` and
  `FALKORCHAT_STOREFRONT_ENABLED` off (`start_agent_team.sh` doesn't export them;
  `config.WORKFLOW_ENABLED` defaults to `False`). Its model consumers are
  `IngestionPipeline` → `llm("extraction", ws=ws)` (`ingestion.py`; document extraction is the
  **`extraction`** kind, its own default in `models.json`), `EmbeddingWorker`/search →
  `embedder("embedding", ...)`, and `AgentResponder` → `llm("agent", ...)` (only on an
  `@bot1` mention). So flipping `defaults.step` would **not** change `:8200`'s document
  extraction.
- **Every `step` consumer** (grep of `server/falkorchat`, `scripts`, `server/tests/eval`):
  1. `WorkflowExecutor._run_agent_node` (`executor.py`):
     `resolve_llm("step", requested=config.get("model"), ws=ctx.ws, overrides=run["modelOverrides"])`.
     This covers every workflow agent step in every workspace. Steps without `config.model`
     (e.g. `triage@v1` in `ws:acme` via `start_server.sh`) use the default.
  2. `QueryGraphDataTool.run` (`tools.py`): `self._models.llm("step", ws=ctx.ws)`. There is **no
     `requested=`**, so outside a workspace override it always gets the global default. That's
     the bug this plan fixes.
  3. `server/tests/eval/run_nlq_golden_set_eval.py`: `tool.run(..., run={})` plus
     `_render_answer` → `gateway.llm("step")`. This is the K-055 NLQ calibration baseline, and it
     prints `step model: <ref>`.
  4. `services.py` publish-time check (`models.resolve(kind, requested=ref)` for declared refs).
     It only validates refs and isn't affected.
- **Conclusion:** a global flip is still wrong, but for different reasons than the brief
  assumed. It would re-model `triage` and every other un-pinned agent step in every workspace served
  from this overlay, and silently move the NLQ eval baseline. It would also leave `:8200`
  untouched anyway, which is the opposite of what the premise feared.

### 2.2 Existing narrow mechanisms in `modelconfig.py`

`ModelGateway.resolve(kind, *, requested=None, ws=None, overrides=None)`, precedence **workspace
override (hard cap) → `requested=` → per-kind default**. For kind `step` the workspace override
key is `agentModel` (`WorkspaceConfig.agentModelOverride`, `_KIND_TO_OVERRIDE_KEY`). `overrides=`
(a pre-fetched dict) is preferred over `ws=` (a fresh graph read). Overlay `roles` (ref without
`/`) give fallback chains, not scoping. Nothing in the overlay is per-def or per-tool (`agents`
is parsed but not honoured).

- **`ws:demo` live state** (read-only query, 2026-10-21):
  `WorkspaceConfig {embeddingModelOverride: 'lmstudio/text-embedding-qwen3-embedding-0.6b'}`, no
  `agentModelOverride`. `ws:demo` also holds snapshots `salesperson@v7`, `salesperson@v8`,
  `order-fulfillment@v1`, **`triage@v1`, `access-request@v1`**.
- `Repository.write_model_overrides` `SET`s **all four** override properties (a `None`
  **clears**). It has **no server/CLI caller** (only `test_queries.sh` §13.1 and
  `embedding_migration.pin`, which preserves the others).
- **How the tool is called:** `_run_agent_node` → `_handle_tool_call(..., run, step, ...)` →
  `self._tools.dispatch(call.name, call.arguments, ctx=ctx, run=run)` → `tool.run(arguments,
  ctx=ctx, run=run)`. The tool sees the per-drive `run` dict (`run["ws"]`,
  `run["modelOverrides"]` are stamped in `_drive`), **not** the calling step or its `config`.
  Nothing in `server/falkorchat` writes `run[...]` except those two stamps in `_drive`.
- **Only `SALESPERSON_DEF` grants `query_graph_data`** (`proof_defs.py`; `seed_salesperson.sh`
  imports it). Its `assistant` step pins `config.model: "lmstudio/mistralai/ministral-3-3b"`,
  carried forward unchanged through `v8` (and therefore also in the `v7` snapshot).

### 2.3 `model-bench` evidence: ministral 32/32 vs qwen 34/34 — what the 2 "unscored" items hide

The consolidated report says "32/32", but that hides **two real failures on answerable items**.
`model-bench/modelbench/scoring/extraction.py:score_item` applies three structural gates
(`no_json` → `parse_failure`; `malformed_spec`; `schema_violation`) **before** scoring. An item
that fails any gate gets `scoreable={}`, so it drops out of `layer1ExactMatchRate`'s
denominator and is only counted in `parseFailures`/`malformedSpecCount`/`schemaViolationCount`.
From `model-bench/results/runs/nlq-structured-query-mistralai_ministral-3-3b-2026-09-20T01:36:40Z.json`:

| Item | Answerable? | Shape | ministral outcome |
|---|---|---|---|
| `nlq-12` "Show me Wearables products that cost under $200." (catalog) | yes | compound-filter | `fail` / `malformed_spec` |
| `nlq-27` "Can you list all the Product-type entities?" (knowledge_base) | yes | filter-list | `fail` / `schema_violation` |
| `nlq-34`, `nlq-36` | no (relationship traversal) | — | `malformed_spec` |
| `nlq-35` | no | — | `parse_failure` / `no_json` |

On the 34 answerable items that's **32/34 = 94.1%** (qwen: 34/34). The `-2512` variant fails the
same two items in the same ways, which points to a deterministic failure at `temperature 0`, not
noise. Raw completions aren't stored in the run record, so *which* structural rule `nlq-12`
broke can't be recovered from it. §7 step Q2 reproduces it live.

**Is the failure mode relevant to `query_graph_data`? Yes, directly.**
`QueryGraphDataTool.run` maps exactly these three classes (unparseable JSON,
`QueryRequest.model_validate` `ValidationError`, `querygen.compile` `ValueError`) to
`{"items": [], "finding": "no matching data found"}`. So in production the failure is a **false
abstention**: the agent is told nothing matches and may tell a shopper "we have no Wearables
under $200". For the gated failures it isn't a fabrication or a crash. **The other direction
does exist, though.** On the three unanswerable items where ministral *did* produce a valid spec
(`nlq-37..39`), it returned rows each time: `unanswerableAbstainRate` 0/3, against qwen's 1/4.
So when the schema can't answer a question, ministral tends to return wrong non-empty rows
rather than abstain. Those items are `knowledge_base`-shaped. The tool offers that dataset to the
demo agent (schema enum), but in `ws:demo` it resolves to `ws:demo`'s own graph, which has 0
`Entity` nodes (read-only query, 2026-10-21). So in the demo any `knowledge_base` query returns
empty rows, which become an abstention, and the fabrication risk applies to future
populated workspaces rather than the demo. The `nlq-12` compound filter (category + price
bound) is a typical storefront question. Mitigating factor: the salesperson agent has
`filter_products`/`lookup_product_fact` for catalog questions and only reaches for
`query_graph_data` for things those don't cover (aggregates, odd phrasings), per its
`systemPrompt`.

**How close is the pack to production?** Very close on the call itself.
`model-bench/packs/nlq-structured-query/prompts/querygen.md` is **byte-identical** to
`tools._QUERY_REQUEST_INSTRUCTIONS` (diffed 2026-10-21; the prompt was last changed in `11d9a192`,
before the pack was built). The golden items, `schema.json` and catalog table are one-way copies
from falkor-chat (`PROVENANCE.md`), and the parse step uses the same
`extract_own_line_json_object(..., require_key="matches")`. Production differs in three ways:
(a) the `question` is **ministral's own restatement** written as the tool-call argument. In the
3-language storefront run it may be in pt-BR/es (not measured), while the golden set is English-only; (b) structural
validation is `querygen.QueryRequest`/`querygen.compile`, not the pack's `tools/exec.py` port;
(c) the pack's "unanswerable" items are moot for the storefront. So the 94% figure is a fair
English-only prior, and (a) is untested.

### 2.4 Context-length: does the step call add to the agent's context?

**No.** It's a separate request. `QueryGraphDataTool.run` calls `llm.complete([...])` with a
fresh two-message list (the system prompt with the dataset schema, plus the restated question),
about 700 prompt tokens (the bench recorded `promptTokens` 694–720). Only the tool's **result
JSON** (`{"items": rows}` or the abstention) is appended to the agent's `messages` as a `tool`
turn, the same as before.

**But it now shares ministral's KV budget.** LM Studio ≥0.4 enables *Unified KV Cache* by
default. With it, the loaded context length (8192) is a single pool shared by all concurrent
predictions (up to `PARALLEL 4`) rather than partitioned per slot. That matches the
salience2 report's hypothesis for its 6 simultaneous `Context size has been exceeded` failures
(external sources, not verified on this box: LM Studio 0.4.0 release notes; llama.cpp discussion
#22658). The step calls that used to land in **qwen's** pool will now land in **ministral's**,
so under 3-way concurrency the overflow risk is expected to rise, not fall. Within one
conversation the tool call runs while its own agent request is already finished (the agent is
waiting on the tool result), so the added pressure comes from the *other* concurrent
participants' requests. This is reasoned from the mechanism, not measured. §7 measures it, and
risk R1 covers mitigations.

## 3. Design & rationale

### 3.1 Chosen: `query_graph_data` inherits the calling agent step's model (code, no data, no def bump)

`query_graph_data`'s internal call resolves **exactly as its calling agent step did**: same
kind (`step`), same `requested=` (the node's `config.model`), same workspace-override snapshot
(`run["modelOverrides"]`). Mechanically:

1. `WorkflowExecutor._run_agent_node` (`executor.py`) builds a **per-node shallow copy**
   `tool_run = {**run, "stepModel": requested_model}` right after its own `resolve_llm(...)`
   and passes `tool_run` (not `run`) to both `_handle_tool_call` call sites. It never mutates
   the per-drive `run`, so a value can't leak to a later node or into the module-level `RUN`
   constant the tests share. `"stepModel"` is always present in `tool_run`, possibly `None`.
2. `QueryGraphDataTool.run` (`tools.py`) resolves with
   `self._models.llm("step", requested=run.get("stepModel"), ws=ctx.ws, overrides=run.get("modelOverrides"))`.
   `resolve()` prefers `overrides` when it isn't `None` and otherwise falls back to the fresh
   `ws=` read, so a direct call with `run={}` (the NLQ eval runner, existing unit tests)
   behaves exactly as today.

**Contract change, stated once:** *a tool that makes its own `step`-kind LLM call uses the model
of the agent node that invoked it.* `query_graph_data` is the only such tool. `run["stepModel"]`
is a new, executor-internal key of the `run` dict tools receive. It's documented on
`ToolRegistry.dispatch`/`QueryGraphDataTool` and isn't persisted anywhere. No public API, graph
schema, def schema or config-file shape changes.

**Effect matrix** (how this lands everywhere):

| Caller | Before | After |
|---|---|---|
| `salesperson@v7`/`v8` `assistant` → `query_graph_data` (any workspace it's seeded in, e.g. `ws:demo`) | `qwen` (step default) | **`ministral`** (the node's `config.model`) |
| A def that grants `query_graph_data` without `config.model` | step default | step default (unchanged) — none exist today |
| Any workspace with `agentModelOverride` set | override (fresh `ws=` read) | override (same drive snapshot the node used) — same model |
| `run_nlq_golden_set_eval.py` (`run={}`) | step default | step default (unchanged) |
| `:8200`, `triage`, every other `step`/`agent`/`extraction`/`guard`/`embedding` consumer | — | unchanged |

It also takes effect for the `v7` snapshot. That's intended and harmless, since `v8` is the
deployed version. It also means a rollback is just a code revert.

### 3.2 Rejected alternatives

| Option | Why rejected (the deciding trade-off) |
|---|---|
| **A. Flip `models.json` `defaults.step` to ministral** | Re-models every un-pinned agent step everywhere this overlay is read (`triage` in `ws:acme`) and silently shifts the K-055 NLQ eval baseline. Its scope is the whole deployment, not the demo. |
| **B. Set `ws:demo`'s `WorkspaceConfig.agentModelOverride` = ministral** (existing FR-16/17 mechanism, zero code) | It's the most direct reuse, but (1) it's a hard cap on **every** `step` resolution in `ws:demo`, which also holds `triage@v1`/`access-request@v1` snapshots, and it flips the `assistant` step's persisted `StepRun.modelSource` from `step` to `workspace` (audit-visible change). (2) There's no operator path: `write_model_overrides` `SET`s all four kinds, and a naive call would **clear** the existing `embeddingModelOverride`. A safe write needs a new script anyway. (3) It's runtime data, not code. A re-created workspace (`create_workspace.sh`) loses it unless `start_demo.sh` re-applies it, so it would also need bring-up changes and a "never flip a running workspace" operational step. It's more moving parts than a two-line code change, with a wider blast radius. |
| **C. New def key (e.g. `config.toolModel`) on the `assistant` step** | Most explicit, but `config` is create-only, so it needs `salesperson@v9` (`v6` is burned; `start_demo.sh` `FALKORCHAT_TRIGGER_DEF_VERSION` default, `seed_salesperson.sh`, `verify_salesperson.sh`, `falkor-chat/AGENTS.md` and the `SALESPERSON_DEF` comment block all carry `v8`). It also needs publish-time validation of the new key in `services._validate_def_spec`, plus a second model knob that would almost always equal `config.model`. Reconsider only if a def ever needs the tool on a *different* model from its agent (trigger to revisit). |
| **D. Pass the step/config into `ToolRegistry.dispatch`** (signature change) | Touches every tool's `run(...)` signature and every registry double in the tests, only to carry one string. The `run` dict is already the per-call carrier, and `_drive` already stamps `ws`/`modelOverrides` on it the same way. |
| **E. Executor hands the tool its already-resolved client** | Same signature churn as D, and it couples tools to the executor's client lifetime. Re-resolving from identical inputs costs ~nothing (resolution is per call by design, `SERVER.md` §1.8). |

The deciding trade-off: §3.1 changes only the one call site that's actually wrong (the tool
ignoring its caller's model choice). It needs no data or version churn, the existing precedence
(override → requested → default) is reused verbatim, and a `git revert` undoes it.

## 4. Implementation steps

Implementer: `coder` (test-alongside is fine; the §5 mutations are mandatory either way). Run the
suite from `falkor-chat/server` (`.venv/bin/python -m pytest -q`; this wipes `reference` at
teardown, see `falkor-chat/AGENTS.md`). Each step leaves the tree green.

| # | Files · symbols | Change | Done when |
|---|---|---|---|
| S1 | `falkor-chat/server/tests/test_tools.py` | Add tests T1–T3 (§5) for `QueryGraphDataTool` with a new recording double `RecordingStepGateway` that implements `.llm(kind, *, requested=None, ws=None, overrides=None)`, records the full call as a dict, and returns a `StubQueryLLM` (the existing helper in this module). Don't reuse the module's embedder-only `RecordingGateway`. | T1/T2 fail against current `tools.py` (T1 sees `requested=None`), T3 passes. |
| S2 | `falkor-chat/server/falkorchat/tools.py` · `QueryGraphDataTool.run` | Replace `self._models.llm("step", ws=ctx.ws)` with `self._models.llm("step", requested=run.get("stepModel"), ws=ctx.ws, overrides=run.get("modelOverrides"))`. Update the class docstring sentence "resolved through the same `ModelGateway` `step` kind `_run_agent_node` already uses" to say it **inherits the calling node's `config.model` and the drive's override snapshot via `run["stepModel"]`/`run["modelOverrides"]`, falling back to the `step` default when called with neither** (e.g. `run={}` from the eval runner). | T1–T3 green; the existing `QueryGraphDataTool` tests (`llm=` stub → `StaticModelGateway`) are unchanged and green. |
| S3 | `falkor-chat/server/tests/test_executor_agent.py` | Add tests T4–T6 (§5). Extend `StubRegistry.dispatch` to also append the received `run` to a new `self.dispatched_runs` list (additive, so existing assertions on `dispatched` stay valid). | T4/T5 fail against current `executor.py`, T6 passes. |
| S4 | `falkor-chat/server/falkorchat/executor.py` · `WorkflowExecutor._run_agent_node` | Right after `model_source_rung = ...`, build `tool_run = {**run, "stepModel": requested_model}` and pass `tool_run` instead of `run` to **both** `_handle_tool_call(...)` calls (the K-039 implicit `post_message` one and the main loop one). Add a short comment: per-node copy, never mutate the per-drive `run`; why (tools that make their own `step` call inherit the node's model, `docs/plans/salesperson-single-model.md`). Leave `_drive`/`_drive_loop` untouched. | T4–T6 green; full suite green. |
| S5 | new `falkor-chat/server/tests/test_query_graph_data_routing.py` | Add T7 (§5): a real `WorkflowExecutor` plus a real `ToolRegistry` holding a real `QueryGraphDataTool`, both wired to **one** dedicated gateway double, `RoutingGateway` (§5). | T7 green; mutation M1 and M2 (§5) each turn it red. |
| S6 | `falkor-chat/server/falkorchat/tools.py` · `ToolRegistry.dispatch` docstring (or the module docstring paragraph that describes `run`) | Document `run["stepModel"]` as an executor-supplied key: "the invoking agent node's `config.model` (may be `None`); present only on executor dispatches". | Docstring present. |
| S7 | `falkor-chat/server/falkorchat/proof_defs.py` · `SALESPERSON_DEF` comment block | In the existing `config.model` create-only note (the `v3` paragraph), add one sentence: `config.model` now also selects the model `query_graph_data`'s internal call uses, so dropping it moves **both** the agent and the tool back to the shared `step` default. **No change to the `SALESPERSON_DEF` dict itself and no version bump.** `verify_salesperson.sh` compares stored config to this constant, and comments aren't config. | `verify_salesperson.sh demo` still prints `RESULT: OK` (read-only; run it if FalkorDB is up). |
| S8 | Docs | §6. | §6 list done. |
| S9 | `falkor-chat/server/tests/eval/run_nlq_golden_set_eval.py` · `_RESULTS_PATH`, `main`, `_run_one` | Two eval-harness changes; production code is untouched. (a) **Results-path override:** `_RESULTS_PATH = Path(os.environ["NLQ_EVAL_RESULTS_PATH"])` when that env var is set, else today's `nlq_eval_results.json`. `_write_results`'s `relative_to(_REPO_ROOT)` print must not crash on a path outside the repo (print the absolute path then). (b) **Raw-completion capture:** wrap the gateway handed to `QueryGraphDataTool` (only that one; `_render_answer` keeps the raw gateway) in a small `_CompletionRecordingGateway` whose `llm(...)` returns a proxy that forwards `complete()` and stores the returned text. `_run_one` adds it to each record as `toolCompletion` (additive key, `None` if no call happened, e.g. unknown dataset). Update the module docstring's "Run:" block to mention `NLQ_EVAL_RESULTS_PATH`. | Offline: with `NLQ_EVAL_RESULTS_PATH=<tmp>`, a stubbed `_write_results` call writes to `<tmp>`. A unit check of the proxy (a stub `llm` returning `"X"` → record `toolCompletion == "X"`) can live in `server/tests/eval/` if importing the runner is side-effect-free; otherwise exercise it in a scratch script and say so in the hand-back. `git status` shows `nlq_eval_results.json` unchanged. |

Commit shape: S1–S2 (tool), S3–S5 (executor plus routing), S9 (eval harness), S6–S8 (docs), or
one commit. Either
way it's small. `teco` commits; the implementer doesn't push.

## 5. Test strategy

All offline (network-free), at unit altitude plus one in-process integration test. Live
behaviour is §7.

| ID | Where | Behaviour |
|---|---|---|
| T1 | `test_tools.py` | `tool.run({"question": q, "dataset": "catalog"}, ctx=CTX, run={"stepModel": "lmstudio/mistralai/ministral-3-3b", "modelOverrides": OV})` makes exactly one gateway call `{"kind": "step", "requested": "lmstudio/mistralai/ministral-3-3b", "ws": CTX.ws, "overrides": OV}`, where `OV` is the all-`None` four-key dict (the `read_model_overrides` shape). |
| T2 | `test_tools.py` | With `OV = {..., "agentModel": "lmstudio/some/override"}`, the call carries that exact `overrides` object, so the gateway's own precedence decides. The tool doesn't drop or re-read it. |
| T3 | `test_tools.py` | `run={}` → `{"kind": "step", "requested": None, "ws": CTX.ws, "overrides": None}`. This pins that the eval-runner path is unchanged. |
| T4 | `test_executor_agent.py` | `_run_agent_node(CTX, RUN, STEP, _config(model="M", tools=["graphrag_retrieve"]), ...)` with an LLM that emits one `graphrag_retrieve` tool call then final text → `registry.dispatched_runs[0]["stepModel"] == "M"`, and every other key of `RUN` is present unchanged in it. |
| T5 | `test_executor_agent.py` | Same with no `model` in config → `"stepModel"` is a key in `dispatched_runs[0]` and its value is `None`. That makes "no model" explicit, so it can't be inherited from anywhere. |
| T6 | `test_executor_agent.py` | After T4's call, the caller's `RUN` dict has **no** `"stepModel"` key (no mutation, no cross-test/cross-node leak). |
| T7 | `test_query_graph_data_routing.py` | `RoutingGateway` (below) drives a node configured with `model="lmstudio/mistralai/ministral-3-3b"` and `tools=["query_graph_data"]`, with `RUN` plus `modelOverrides` = the all-`None` dict. Assert three things. (1) The **ministral-keyed** query client's `complete()` was called exactly once. (2) **No client was ever created for any other key**, including `None`, which is the default. (3) The tool's `llm` call carried the same `overrides` object as the node's `resolve_llm`. That pins *which client answered*, not just the arguments passed: the "one model for both roles" property, end-to-end through the real dispatch path. |

Reuse the existing stubs (`StubQueryLLM`, `StubServices`/`structured_query_calls` in
`test_tools.py`; `StubChatLLM`, `ChatResult`, `ToolCall`, `_executor`, `CTX`, `RUN`, `STEP` in
`test_executor_agent.py`). For T7, build the executor the way the existing K-042 tests do:
`WorkflowExecutor(None, None, models=gateway, tool_registry=ToolRegistry([QueryGraphDataTool(svc, models=gateway)]), guard_judge=None)`.
Executor `services=None` works with `run_ctx={"threadId": "t1"}`, as in
`test_run_agent_node_resolves_step_kind_with_the_steps_own_model_and_ws`, and `svc` is
`test_tools.py`'s `StubServices`-style double with `run_structured_query` (copy it into the new
module if importing it across test modules is awkward). **Don't extend
`test_executor_agent.py`'s `RecordingGateway`.** Its `resolve_llm` returns a cached per-ref
`StubChatLLM` that never emits a tool call, and an `llm()` added to it would hand back that same
object, which has no `.complete`. Write a dedicated `RoutingGateway` instead:

- `resolve_llm(kind, *, requested, ws, overrides)` records the call and returns a scripted chat
  stub. The first `chat()` returns `ChatResult(tool_calls=[ToolCall(id="c1", name="query_graph_data", arguments={"question": "Which products cost less than $50?", "dataset": "catalog"})])`.
  The second returns final text.
- `llm(kind, *, requested, ws, overrides)` records the call and returns a **per-`requested`-key**
  `StubQueryLLM` held in `self.query_clients: dict[str | None, StubQueryLLM]`, created on first
  use. Its reply is a valid `QueryRequest` JSON line, e.g. `test_tools.py`'s
  `_CATALOG_FILTER_REPLY`.

The `post_message`-free `tools` list keeps the K-039 implicit-dispatch branch out of T7.
**Decision on the implicit `post_message` call site:** it also receives `tool_run` (S4), so the
contract holds at every dispatch path. No mutation covers it, because `post_message` ignores
`stepModel`. That's deliberate, not an oversight.

**Mandatory mutations** (run each, confirm red, revert; record the result in the hand-back):

- **M1 (the routing proof):** in `QueryGraphDataTool.run`, revert to
  `self._models.llm("step", ws=ctx.ws)`. **T1, T2 and T7 must fail.** If T7 still passes, it
  isn't exercising the tool's own resolution. Fix the test, not the mutation.
- **M2:** in `_run_agent_node`, pass `run` instead of `tool_run` to the main-loop
  `_handle_tool_call` → T4 and T7 must fail.
- **M3:** replace the copy with `run["stepModel"] = requested_model` (in-place mutation) → T6
  must fail.

**Regression:** the whole `falkor-chat/server` offline suite stays green. In particular
`test_tools.py` `QueryGraphDataTool` tests, `test_executor_agent.py` K-042 `modelSource` tests,
and `test_salesperson_scaffold.py`. Don't run `pytest -m live` (it needs LM Studio and isn't part
of this gate). `./scripts/test_queries.sh` is unaffected (no Cypher change), so no need to run it.

## 6. Docs to update

| Doc | Change | Who / when |
|---|---|---|
| `falkor-chat/AGENTS.md` | **Replace** (don't append to) the "Before running a live, audience-facing storefront demo, pin the demo's models" paragraph. New content: the demo calls **one** chat LLM (`ministral-3-3b`, for the `assistant` step and, inherited, `query_graph_data`) plus the embedder. Load both with `lms load` before serving. `qwen` must stay **unloaded** for the whole demo: `:8200` idle (no `ingest_document`, no `@bot1`) or stopped by its owner, and `lms ps` checked right before going live (U0 saw it re-loaded mid-window, R2). Don't call `ingest_document` during a demo, on `:8200` **or on the demo process's own `/mcp`** (both wire `IngestionPipeline`, whose `extraction` default is still `qwen` and would JIT-load onto the same GPU). Cite K-066. Also, in "Three traps when bumping `SALESPERSON_DEF`" trap (1): dropping `config.model` now reverts **both** the `assistant` step and `query_graph_data` to the shared `step` default. | `coder`, with S8 |
| `falkor-chat/docs/SERVER.md` §1.8 | The binding-sites sentence ("Five binding sites resolve through them: …") omits `QueryGraphDataTool.run` (`step`). Add it as "`step`, inheriting the invoking node's `config.model` and override snapshot via `run["stepModel"]`/`run["modelOverrides"]`". Separate pre-existing inaccuracy to flag for `teco`: `IngestionPipeline` resolves kind **`extraction`**, which is outside `modelconfig.KINDS` and `_KIND_TO_OVERRIDE_KEY`. So the "four closed consumer kinds" claim is false, and a workspace override can't reach extraction. Not fixed by this plan (see R5). Also update the §1.8 ASCII diagram's consumer row: add a `query tool (kind=step, inherits node model)` box beside `executor (kind=step)`. | `coder`, with S8 |
| `salesperson/README.md` "Status" | **Rewrite** the K-066 sentences ("0.5% with all three pinned … load them explicitly"): the demo needs `ministral` + embedder loaded explicitly, and the dead-turn figure is whatever U3 measures. Cite K-066 rather than restating numbers that will move. (`salesperson/AGENTS.md`, named in `teco`'s scan, needs no change.) | `coder`, with S8; numbers filled by `teco` after U3 |
| `falkor-chat/docs/manuals/llm-provider-config.md` §4 ("Naming a specific model for one workflow step or guard") | One sentence: a step's `model` also applies to the internal query call made by the `query_graph_data` tool when that step invokes it. | route to `tico` (manuals owner), after U2 |
| `falkor-chat/docs/BACKLOG.md` K-066 | **Rewrite** the item body (open-item rule, not an `Update:` clause): the preload set is now `ministral` + embedder, the `qwen`-through-`query_graph_data` paragraph goes, plus whatever §7 measures (dead-turn rate, context-overflow behaviour, VRAM fit). | `teco`, after U3 |
| `falkor-chat/docs/HISTORY.md` | One dated entry at close (mechanism, test IDs, §7 headline numbers). | `teco`, at close |
| `falkor-chat/server/tests/eval/run_nlq_golden_set_eval.py` docstring | Covered by S9. | `coder` |
| `falkor-chat/README.md` (line ~35, "LM Studio … `qwen/qwen3-4b-2507`") | Leave as is. It describes the general server default, not the storefront demo. | — |

## 7. Live-QA acceptance (`qa-engineer`)

**U0 (done)** is `falkor-chat/docs/test-reports/salesperson-single-model-fit-report.md`. The
headline results that shape U3:

- **ministral + embedder don't co-reside.** `--gpu max` loads succeed anyway; llama.cpp logs
  `failed to fit … abort` and proceeds. When the embedder runs, WDDM pages ~1.6 GiB of ministral
  out to system RAM, and pages it back on the next chat. `nvidia-smi` stays under total
  throughout (5839/6141 MiB).
- **The swap costs about +0.07–0.1 s on ministral's TTFT, and 0.25–1.4 s on the first embedding
  after a chat.** Generation stays at 38.5–40.5 tok/s, so **tok/s can't see this spill**. It
  only separates real CPU offload: `--gpu 0.5` gives 14 tok/s.
- **Both processes need counting.** The embedder runs inside the LM Studio process, not
  `llama-server`. `llama-server`'s Shared Usage holds a constant ~112 MiB pinned buffer, so
  there's no "no-spill floor" to compare against. Per-process **Dedicated** Usage is what moves.
- **Concurrency:** at `PARALLEL 4` with the unified 8192 pool, **3 concurrent ~3.1k-token
  requests all failed** with `Context size has been exceeded` (twice). 2 concurrent requests
  succeeded.
- The LM Studio runtime is now `2.59.0`; the bench's references used `2.40.0`.

**Stakeholder decision (2026-10-21): build it, and accept the WDDM paging cost.** U3 therefore
**measures** the paging regime; it doesn't test for full residency. **U3 runs after U2 (S1–S9)
is merged.**

**One fact that changes how big that cost is in the demo.** The storefront turn path never
calls the embedder. `storefront_api.py` deliberately schedules no `_safe_embed` (§4.4 measure 3:
`salesperson` has no `graphrag_retrieve`). Executor `post_message` goes through `services`, not
the `api.py`/`mcp.py` front doors that schedule embeds. That matches the salience2 run, where
the JIT-loaded embedder idled out of LM Studio mid-run. So in a pure storefront run the embedder
should stay idle (paged out), ministral should stay resident, and the per-switch swap should be
paid only when some *other* caller embeds. Q5(d) verifies this, and §8 open question 1 follows
from it.

U3 deliverable: `falkor-chat/docs/test-reports/salesperson-single-model-report.md`. Baselines:
- **Salience / dead turns / latency:** `falkor-chat/docs/test-reports/salesperson-language-salience2-report.md`
  (three models pinned, ctx 8192, `PARALLEL 4`, n=75). Results: 0/1,119 wrong-language;
  6/1,125 dead = 0.5% Wilson [0.2%, 1.2%], all ministral `Context size has been exceeded`, in
  two simultaneous bursts.
  **Turn latency** over its 1,119 replied turns, computed by the architect from the raw
  `/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/01483502-06e5-42ad-9b05-36efaa8d1ff1/scratchpad/salience2.jsonl`: `turn_ms` p50 **6.0 s**, p95 **24.1 s**, max 62.3 s. This uses
  nearest-rank on sorted `turn_ms` of replied records, quantized by the probe's 2 s poll; Q5
  uses the same computation.
- **NLQ:** Q0's fresh qwen run. The tracked `falkor-chat/server/tests/eval/nlq_eval_results.json`
  (qwen, 2026-08-30, 33/39, catalog 20/20) stays the fallback comparator.

**Instruments** (`LMS=/mnt/c/Users/mauri/.lmstudio/bin/lms.exe`):
- `nvidia-smi --query-gpu=memory.used,memory.total --format=csv`.
- U0's Windows counters `\GPU Process Memory(*)\Dedicated Usage` / `Shared Usage` for the
  NVIDIA adapter (LUID `0x00011343`, re-check it), read via `powershell.exe Get-Counter` for
  **both** `llama-server` and the LM Studio process, plus the adapter total. Sample every 5 s
  across Q3 into the scratchpad. U0's script was `/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/01483502-06e5-42ad-9b05-36efaa8d1ff1/scratchpad/gpumem.ps1`.
- U0's throughput/swap harness, `/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/01483502-06e5-42ad-9b05-36efaa8d1ff1/scratchpad/bench.py`.
- `$LMS log stream -s runtime` kept to a file for the whole window.

**Steps, in order:**

| ID | What | Pass |
|---|---|---|
| Q0 — fresh qwen NLQ comparator (first U3 step, before E2) | Needs S9 merged. `$LMS unload --all`, then `$LMS load qwen/qwen3-4b-2507 -c 8192 -y` and `$LMS load text-embedding-qwen3-embedding-0.6b -y`. Check `$LMS ps` lists exactly those two. Run `cd falkor-chat/server && NLQ_EVAL_RESULTS_PATH=<scratchpad>/nlq-qwen-current.json .venv/bin/python tests/eval/run_nlq_golden_set_eval.py` (it must print `step model: lmstudio/qwen/qwen3-4b-2507`). Then `git status --short falkor-chat/server/tests/eval/` must be **empty**. **Why a fresh run:** the tracked baseline predates the golden-set pair `nlq-40` (`d58125ad`, a catalog compound filter), and LM Studio's runtime has moved from 2.40.0 to 2.59.0 (U0). (`3d014975` isn't a reason: it's an unreachable defense-in-depth recheck.) | Run completed; tracked file untouched. The result becomes Q2's comparator. If it disagrees with the tracked file on any catalog pair, list those pairs (a runtime-drift signal worth reporting in its own right). |
| E1 — demo up | Demo built from the U2 commit, started with `falkor-chat/scripts/start_demo.sh` (as in the baseline: `FALKORCHAT_OPENCODE_CONFIG=~/.config/opencode/opencode.local.json`, `--reload` off, `verify_catalog`/`verify_salesperson` `OK`). | Preflight OK. |
| E2 — load the single-model set | `$LMS unload --all`, then `$LMS load mistralai/ministral-3-3b -c 8192 --gpu max -y` and `$LMS load text-embedding-qwen3-embedding-0.6b --gpu max -y`, both without `--ttl`. Use the same commands U0 used; `--gpu max` keeps every layer on the GPU, and the overcommit is paged by WDDM rather than refused. Assert that `$LMS ps` lists **exactly two** models and that the chat model's IDENTIFIER is **`mistralai/ministral-3-3b`**, not `mistralai_ministral-3-3b-instruct-2512` (`-y` takes the first match). Cross-check `GET /api/v0/models`. The window must be dedicated and stakeholder-OK'd. Don't stop `:8200`. Record whether it's up, and **re-check `$LMS ps` for `qwen` at E2, before Q3, and after Q3** (R2: U0 saw qwen loaded again by another consumer). | Exactly the two models; qwen absent at all three checks. If qwen appears, record which process requested it from the runtime log, unload it, and restart the affected step. |
| Q1 — routing + catalog answer (smoke) | In each of `en`/`pt-BR`/`es`, a fresh storefront session with turns that steer toward `query_graph_data`: an aggregate ("How many products do you have in Wearables?"), a superlative ("What's your cheapest product?"), and the **`nlq-12` turn** ("Show me Wearables products that cost under $200.", translated). Snapshot `/api/v0/models` before and after. **Dispatch source:** after the run, read-only `GRAPH.RO_QUERY ws:demo "MATCH (m:Message) WHERE m.createdAt >= $t0 AND 'query_graph_data' IN coalesce(m.toolsUsed, []) RETURN m.threadId, m.createdAt, m.toolsUsed, m.text"`. Replace `$t0` with the session start as an epoch-ms literal; the cypher MCP takes no parameters. `Message.toolsUsed` is written by `link_step_emission` onto the message a node posted. It lists the tools that **successfully** dispatched in that node, `post_message` excluded, so a node that dispatched but posted nothing leaves no trace here. | **All hard:** (a) `qwen/qwen3-4b-2507` never appears in `/api/v0/models` and no `qwen` string appears in the demo log. (b) ≥1 `query_graph_data` dispatch per language in `toolsUsed`; if none, rephrase until it's exercised, because unexercised routing isn't a pass. (c) The `nlq-12` turn's reply names **Fitness Tracker Band** and doesn't claim nothing matches, in all three languages, whichever tool the agent picked. The aggregate/superlative answers are checked against `seed_catalog.sh`; a miss there is recorded verbatim and folds into Q2's stakeholder surface. |
| Q2 — NLQ correctness, production path (**gate**) | With the E2 set loaded and **no demo traffic flowing**, run `cd falkor-chat/server && FALKORCHAT_MODEL_CONFIG=<scratchpad copy of config/models.json with defaults.step = lmstudio/mistralai/ministral-3-3b> NLQ_EVAL_RESULTS_PATH=<scratchpad>/nlq-ministral.json .venv/bin/python tests/eval/run_nlq_golden_set_eval.py`. It must print `step model: lmstudio/mistralai/ministral-3-3b`. Afterwards, `git status --short falkor-chat/server/tests/eval/` must be empty. | **Gate:** every **catalog** pair that Q0's qwen run got right (the tracked file's, if Q0 couldn't run) is `layer1Correct: true` for ministral. **Any catalog regression is not a silent pass:** list each one with its `toolCompletion` and `toolResult`, and record the stakeholder's explicit accept or reject before U3 is done. Also report, not gated: knowledge_base answerable Layer 1; the unanswerable shapes' outcomes (rows returned where the right answer is abstention, §2.3); Layer 2; and every miss's raw `toolCompletion`. |
| Q3 — salience + dead turns (the baseline rerun) | `python3 salesperson/scripts/language_salience_probe.py run --trials 75 --base-url http://127.0.0.1:8000 --lmstudio-url http://localhost:1234 --out <scratchpad>/single-model.jsonl`, then `aggregate ... --dump-replies`. Same protocol, harness SHA and hand-read as the baseline. The demo's default `PARALLEL 4` and `FALKORCHAT_STOREFRONT_TURN_WORKERS=4` stay unchanged, so this is the like-for-like comparison. Attribute every dead turn by joining it to the demo log's `ProviderCallError` and **split by call site** (traceback through `executor._run_agent_node` → `llm.chat` vs `tools.py` `QueryGraphDataTool.run` → `llm.complete`). Run the instruments throughout. | Wrong-language: no worse than the baseline (0 by hand-read; report the Wilson bound). **Dead-turn rate (dead / turns attempted):** pass at **≤ 1.2%** (the baseline's Wilson upper bound). Above 1.2% but with the Wilson CI still overlapping the baseline's [0.2%, 1.2%], it's a **stakeholder decision** (that covers up to ~1.8% at n≈1,125). CI non-overlap is a **fail**. For every `Context size has been exceeded` event, report timestamps, call site, and whether it came in a multi-participant burst. |
| Q4 — concurrency lever (re-scoped by U0's finding) | U0 showed that **three** concurrent long requests exhaust the shared 8192 pool and all fail, while **two** succeed. Q3 runs three concurrent participants, and each turn's tool call now lands in the same pool (§2.4). So the lever is capping in-flight ministral requests at 2; a bigger context doesn't fit (R6). Rerun Q3 at **n=25** with **one** of these, chosen in this order: **(i)** `FALKORCHAT_STOREFRONT_TURN_WORKERS=2` (an env var, server-side only; restart the demo; LM Studio is untouched, so other LM Studio clients are unaffected), or **(ii)** if (i) proves insufficient, `$LMS unload mistralai/ministral-3-3b` (confirm it's gone in `$LMS ps`), then `$LMS load mistralai/ministral-3-3b -c 8192 --parallel 2 --gpu max -y` and re-assert the E2 checks. Run Q4 **whether or not** Q3 saw overflows: it is the demo-day settings candidate. | Report the overflow count, dead-turn rate, Q5(a)-style p95 turn latency (two workers queue the third participant), and memory. **Recommend** the demo-day setting. The recommendation is to adopt the lever if it removes overflows with p95 inside Q5(a)'s pass band; otherwise report the trade-off for the stakeholder. |
| Q5 — paging cost (stakeholder-accepted; measured, banded) | **(a) Turn latency:** p95 `turn_ms` of replied turns in Q3, computed as in the baseline. **(b) Per-switch swap:** after Q3, U0's `bench.py` sequence of chat ×4 → embed ×5 → chat ×2. Report ministral TTFT on the first chat after embeddings vs. steady state, and the first-embed-after-chat latency. **(c) Dead turns:** Q3's figure, not double-gated. **(d) Embedder traffic during Q3:** count embedding requests in the runtime log (expected **0**, see the note above). Report the adapter-total and per-process (`llama-server`, LM Studio) Dedicated/Shared Usage over time, which shows whether paging actually happened during Q3. Report only, not gated. | **(a)** pass at p95 **≤ 29 s** (baseline 24.1 s plus 5 s: U0's swap is ≤ 1.4 s, and the 2 s poll quantizes); **29–36 s** (up to 1.5× baseline) is a stakeholder decision; **> 36 s** fails. **(b)** first-chat TTFT penalty: pass at **≤ 0.3 s**, stakeholder at ≤ 1.0 s, fail above that. First-embed-after-chat: pass at **≤ 1.5 s**, stakeholder at ≤ 5 s, fail above that (U0 measured 0.1 s / 1.4 s). **(d)** If Q3 made any embedding requests, identify the caller. A non-zero count from the storefront path contradicts §4.4 measure 3 and goes to `teco` as a defect. |

**Overall acceptance:** Q0 ran and left the tree clean. E2's checks hold, including qwen absent
throughout. Q1 passes all three hard clauses. Q2 passes its gate or has the stakeholder's
recorded decision. Q3 and Q5(a)/(b) are each in their pass band or have the stakeholder's
recorded decision. Q4 yields a demo-day recommendation. No step changes `models.json`, any
`WorkspaceConfig`, the shared OpenCode file, or any tracked file.

## 8. Risks & open questions

- **R1 — Context overflow can get worse (most likely risk).** Every `query_graph_data` call is
  now a ~700-token request into ministral's 8192 pool. U0 confirmed the pool is unified
  (`kv_unified = 'true'`, `n_ctx_slot = 8192`). At `PARALLEL 4`, **three** concurrent ~3.1k-token
  requests all failed, while two succeeded. Three concurrent storefront participants is exactly
  the demo shape, so long conversations can overflow together, as in salience2's bursts.
  Mitigation: cap in-flight requests at 2, with `FALKORCHAT_STOREFRONT_TURN_WORKERS=2` or
  `--parallel 2` (Q4). Both are configuration, not code. The cost is queueing latency for a third
  concurrent shopper. A larger context doesn't fit (R6).
- **R2 — "One model" doesn't hold if qwen gets loaded during the demo, and U0's window showed
  it does.** `qwen` was loaded again by another consumer while U0 was testing (attributed to
  `:8200` per `teco`; U0's report records it resident on arrival and warns that JIT re-loads
  it). **Demo-day guidance:** `qwen` must stay unloaded for the whole demo. That means `:8200`
  is idle (no `ingest_document`, no `@bot1` mention) or stopped by its owner, and `$LMS ps` is
  checked right before going live. The same applies to anything that extracts: `:8200`'s `extraction`
  (and `agent` responder) default is still `qwen`, and so is the demo process's own (`app.py`
  wires `IngestionPipeline` there too, reachable via its `/mcp` `ingest_document`). Either call would
  JIT-load `qwen` on the same GPU and re-create the eviction regime K-066 measured at ~63% dead
  turns. This is operational guidance for the demo runbook (§6 `falkor-chat/AGENTS.md`), not this
  change's scope. A per-process overlay for `:8200` is expressly forbidden by
  `start_agent_team.sh`'s header.
- **R3 — NLQ quality drop.** On the bench prompt, ministral fails 2/34 answerable items where
  `qwen` fails 0. One of them is a catalog compound filter, and production turns that failure into
  a false "no matching data found" (§2.3). In the other direction, ministral abstains on 0/3
  valid-spec unanswerable items (qwen 1/4), which is moot in `ws:demo` (no `Entity` data) but not
  in general. Per `teco`'s ruling this is a **gate** (Q2): any catalog regression against the
  tracked qwen baseline goes to the stakeholder; it's never silently accepted. If it's unacceptable, a fix is
  prompt work on `_QUERY_REQUEST_INSTRUCTIONS` (e.g. a compound-filter example), which would also
  need a `model-bench` pack version bump to keep the bench honest. That's out of scope here.
- **R4 — Behaviour change for any future def.** A future def granting `query_graph_data` with a
  `config.model` will now drive the tool with that model too. That's the intended contract
  (§3.1); revisit option C (§3.2) if a def ever needs them to differ.
- **R5 — Pre-existing, not fixed: `extraction` is a fifth kind.** `ingestion.py` resolves
  `llm("extraction", ...)`, but `modelconfig.KINDS`/`_KIND_TO_OVERRIDE_KEY` list four kinds and
  `SERVER.md` §1.8 says "four closed consumer kinds … adding a fifth means adding its own override
  property". So a workspace override can't reach extraction (the hard cap silently doesn't
  apply). Flagged for `teco` to backlog. It doesn't affect this plan.
- **Rollback:** revert the U2 commit. There's no data, def or config migration to undo.
- **R6 — No full fit; paging accepted (resolved).** U0: ministral + embedder co-reside only
  through WDDM paging (~1.6 GiB of ministral paged out whenever the embedder runs). The
  stakeholder accepted that on 2026-10-21. Q5 measures the cost. Two levers U0 didn't test could
  remove the oversubscription: an embedder loaded with a smaller context (its batch follows
  context, which drove its ~2.1 GiB), or a ministral build without the F16 vision projector. Each
  needs its own measurement, and neither is in scope here.
- **Open questions (stakeholder, via `teco`):**
  1. **Does the demo need the embedder loaded at all?** The storefront path never embeds (§7
     note; `storefront_api.py` §4.4 measure 3). If Q5(d) confirms 0 embedding requests during
     Q3, preloading only ministral removes the paging entirely. Any stray embed call would then
     JIT-load the embedder and pay one swap. I haven't checked whether anything else in a demo
     session (presenter flows, `start_demo.sh` preflight) embeds. U3's data answers it for the
     storefront path. This would change §6's `falkor-chat/AGENTS.md` guidance ("load both"),
     which I haven't edited (v3 scope).
  2. Q3's band between 1.2% and CI non-overlap, Q5's stakeholder bands, and any Q2 catalog
     regression are explicit stakeholder calls.

## 9. Review dispositions

### Pass 1 (2026-10-21) dispositions

| Finding | Disposition |
|---|---|
| M1 Q2 overwrites tracked baseline; no raw completions | **Fixed, harness route.** S9 adds `NLQ_EVAL_RESULTS_PATH` plus `toolCompletion` capture. Q2 writes to the scratchpad and checks `git status`. Baseline named explicitly. I added one point the review didn't raise: the tracked baseline predates `nlq-40` and a `querygen` fix, hence the optional Q2a current-tree qwen run. |
| M2 no correctness pass bar | **Fixed per `teco` ruling.** Q2 is a catalog no-regression gate with mandatory stakeholder surfacing. Q1(c) makes the `nlq-12` turn a hard check in all three languages. **Partial disagreement:** Q1(c) checks the shopper-visible answer, not that `query_graph_data` answered it. The agent may legitimately route that question to `filter_products`, and failing on a correct answer via another tool would test the prompt's routing, not this change. The tool's own correctness on that shape is Q2's job. |
| M3 Q5 blind to spill; fit tight | **Fixed.** `--gpu max` in E2/Q4, U0's Windows `Shared Usage` counter plus the tok/s check in Q5, and the fit check split out to U0 (`teco`). Q4 re-sized to `--parallel 2`; 12k only if U0's headroom allows. |
| m1 T7 double | **Fixed.** Dedicated `RoutingGateway` with per-key query clients; T7 asserts which client answered. |
| m2 abstain-rate signal | **Fixed** in §2.3/R3. I added that it's moot in `ws:demo` (0 `Entity` nodes, verified). Q2 reports the unanswerable shapes. |
| m3 Q3 tolerance | **Fixed.** Fixed ceiling ≤ 13/1,125 (1.2%), a stakeholder band up to CI overlap, fail beyond. |
| m4 Q1 dispatch source | **Fixed.** `Message.toolsUsed` RO query, with its blind spot noted. |
| m5 unload/identifier | **Fixed.** `unload` before every (re)load, plus the exact-IDENTIFIER and two-model assertion. |
| m6 two doc sites | **Fixed.** `salesperson/README.md` Status and the `SERVER.md` §1.8 diagram added to §6. |
| Nit: demo `/mcp` ingest | **Fixed** in R2 and the `falkor-chat/AGENTS.md` row. |
| Nit: implicit `post_message` site | **Kept `tool_run` there, by decision** (§5 note): one contract at every dispatch path, and no mutation needed because `post_message` ignores the key. |

### Pass 2 (2026-10-21) dispositions

| Finding | Disposition |
|---|---|
| B1 Q5 spill checks refuted by U0 | **Fixed.** Q5 is rewritten around accepted paging: p95 turn latency, per-switch swap (TTFT penalty, first embed after a chat) and dead turns, each banded. The "fails loudly", tok/s-as-spill and no-spill-floor checks are gone, and counters now cover both processes plus the adapter total. I added Q5(d): the storefront path never embeds, so the per-turn swap U0 inferred ("a storefront turn alternates embedder and ministral") likely doesn't occur in the demo. That's verified from `storefront_api.py`, but not live, hence Q5(d) and open question 1. |
| M4 Q2a can't run where placed | **Fixed.** It's now **Q0**, the first U3 step: after U2, before E2, with qwen + embedder only, then `git status`. Motive corrected to `nlq-40` plus runtime drift 2.40.0 → 2.59.0. |
| m7 T7 `resolve_llm` tuple | Left to the implementer per `teco` (`coder` builds from v2). The fix is to return `(stub, _FakeResolution(source=...))`. |
| Nit: S9 runner import side effects | Left to the implementer per `teco`: use the scratch-script/subprocess route, and don't import the runner into the pytest session. |
| Nit: Q3 thresholds as rates | **Fixed** (≤ 1.2% pass; Wilson non-overlap fail). |
| Concurrency (U0, via `teco`) | Folded into Q4 (cap in-flight at 2; server-side `TURN_WORKERS=2` first) and R1. |
| qwen re-loaded by `:8200` (U0, via `teco`) | Folded into E2's three qwen checks, R2's demo-day guidance, and the §6 runbook row's intent. |

## Ready to implement

`coder`: implement S1–S9 above, run M1–M3 and report red/green for each. `qa-engineer`: then §7
against the U2 build, with only ministral and the embedder loaded.
