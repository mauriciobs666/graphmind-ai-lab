# Change History — falkor-chat

> Dated log of actual changes to the `falkor-chat` component. Most recent first.
> (Formerly `kaizen/history.md` — older entries may say "kaizen" for what is now
> [`BACKLOG.md`](./BACKLOG.md) + this file; file paths in old entries have been
> updated so they still resolve.)

## 2026-08-30 — K-056 resolved by model swap — `qwen3-4b` skip-and-fabricate mechanism superseded, not scaffold-fixed

**What:** K-056 ("`salesperson` scaffold: live tool-call skip-and-fabricate under extended
conversations") is closed as **resolved by model substitution, not by a scaffold-level fix** — the
two scaffold-level mitigation candidates tried (`tool_choice: "required"` forcing, a tool-use
breadcrumb) were both independently falsified/reverted live (`docs/reviews/
salesperson-tool-reliability-impl.md`). `docs/reviews/salesperson-tool-reliability-ml.md` §8
ran a controlled alternative-model eval and found `mistralai/ministral-3-3b` shows **zero**
skip-and-fabricate instances across 176 live-verified turns (Wilson 95% CI 0-2.1%), against the
pinned `qwen/qwen3-4b-2507`'s confirmed near-certain (87-100% CI) failure by a conversation's 4th
turn — a categorical, not incremental, improvement on this specific mechanism. `salesperson`'s
`assistant` step was re-pointed at `ministral-3-3b` (`v2.1`, commit `03a3c8c`) and every subsequent
version (`v3`, `v4`) carries that pin forward. `qwen/qwen3-4b-2507` remains in use elsewhere
(`query_graph_data`'s internal single-shot structured-completion call, K-055) but that call shape
is not vulnerable to the same multi-turn-conversation mechanism.

**Caveat, not swept under the rug:** this is a substitution, not a proof the underlying scaffold
risk (a model silently skipping a required tool call in a long conversation) can never recur — it
would resurface if this role is ever re-pointed at a less-robust model again. The shipped
observability signal (`executor._note_possible_fabrication`) stays in place as a standing
ops-alerting backstop regardless of which model is pinned. Piloting `ministral-3-3b` surfaced its
own, different, real defect (a follow-up instruction sometimes silently re-firing an earlier,
already-completed tool call) — filed separately as `K-058`, not part of this resolution.

## 2026-08-30 — M6 (business entities in workflows) closed — all four sibling capabilities delivered

**What:** M6's own closing condition ("all four sibling capabilities proven live inside the
combined salesperson demo agent, golden-set/adversarial gates passed for K-055") is met. A final
`qa-engineer` combined e2e pass (`docs/test-reports/workflow-salesperson-demo-report.md`, PASS, no
defects) drove one continuous live conversation with the real `salesperson@v4` agent exercising all
four capabilities together — catalog lookup, cart/order, durable profile, NL query generation —
plus a second, independent thread proving profile and post-order cart state hold together without
re-running the state-creating turns. No cross-capability interference, no regression versus any
capability's own individually-recorded acceptance pass. See the four capability entries below and
`docs/plans/workflow-salesperson-demo-coordination.md` (now archived) for the full multi-session
trail. `K-056` (a separately-filed, still-open model-reliability defect, not one of M6's four
gating items) remains open independent of this closure — see `docs/BACKLOG.md`.

## 2026-08-30 — K-055: natural-language query generation over structured graph data — delivered

**What:** Arbitrary-phrasing question answering via a constrained query-builder DSL
(`server/falkorchat/querygen.py`), never free-form LLM-generated Cypher, executed exclusively
through `GRAPH.RO_QUERY` as a second, engine-enforced non-mutation backstop. Wired into
`salesperson@v4` (`query_graph_data` tool). Full trail: plan
`docs/plans/workflow-nl-query-generation.md` (v1.1) + ml note (v2, corrected golden-set exclusion
rule for permanently out-of-scope shapes); implementation review
`docs/reviews/workflow-nl-query-generation-impl.md` (Pass 2: approve); RCA
(`docs/reviews/workflow-nl-query-generation-rca.md`) root-causing and fixing 4 accuracy defects
found by the golden-set harness; security review
`docs/reviews/workflow-nl-query-generation-security.md` (4 passes — 2 design-time, 2 live — final
verdict: approve, no open finding, after closing a `QueryFilter.op` allowlist gap found live in
Pass 3); golden-set harness `docs/test-reports/workflow-nl-query-generation-report.md` (100% on
every in-scope shape under the corrected gate; `relationship-traversal`/`conflicting-facts` are a
named, permanent, structural gap — extending the DSL to cover them is a separate future scope
decision, not resolved here); live acceptance
`docs/test-reports/workflow-nl-query-generation2-report.md` (PASS WITH DEFECTS — one MAJOR,
intermittent, likely-orchestration-layer defect on a compound-filter question, filed for follow-up,
not gating).

## 2026-08-30 — K-054: durable user-profile data for workflows — delivered

**What:** Durable, workspace-scoped customer name/delivery-address capture as two properties on
the shared `Customer` node (`get_profile`/`save_profile` tools), wired into `salesperson@v3`.
Plan: `docs/plans/workflow-durable-profile.md` + graph note
`docs/plans/workflow-durable-profile-graph.md`. Review: `docs/reviews/workflow-durable-profile.md`
(one BLOCKER found and fixed live — a partial-update contract gap that would have silently erased
previously captured customer data — Pass 2: approve with suggestions) +
`docs/reviews/workflow-durable-profile-impl.md`. Live acceptance:
`docs/test-reports/workflow-durable-profile-report.md` (PASS, all three AC hold).

## 2026-08-30 — K-053: cart, orders, and deterministic totals — delivered

**What:** Durable, workspace-scoped cart/order state (`Cart`/`CartItem`/`Order`/`OrderLine`) and an
LLM-free computation for line-item totals and order snapshots (`view_cart`/`add_to_cart`/
`remove_from_cart`/`clear_cart`/`place_order` tools, plus the `order-fulfillment@v1` process def
for the operator-side lifecycle), wired into `salesperson@v2`. Plan:
`docs/plans/workflow-cart-and-totals.md` (v2) + graph note
`docs/plans/workflow-cart-and-totals-graph.md` (v2). Review:
`docs/reviews/workflow-cart-and-totals.md` (one MAJOR found and fixed — unassigned write ownership
for `ensure_customer`/`ensure_cart` — approve) + `docs/reviews/workflow-cart-and-totals-impl.md`.
Live acceptance: `docs/test-reports/workflow-cart-and-totals-report.md` and, after K-056's Ministral
re-point, the regression re-verification `docs/test-reports/workflow-cart-and-totals2-report.md`
(both PASS WITH DEFECTS, all ten AC hold).

## 2026-08-30 — K-052: structured catalog/reference lookup for workflows — delivered

**What:** Fixed-shape exact-name/category/price-range lookup (`lookup_product_fact`/
`filter_products` tools) against a new, seed-script-only ~15-product `Product` catalog in
`reference`. Ships the shared `salesperson@v1` `WorkflowDef` scaffold the other three M6
capabilities extended by version bump. Plan: `docs/plans/workflow-catalog-lookup.md`. Review:
`docs/reviews/workflow-catalog-lookup.md` + `docs/reviews/workflow-catalog-lookup-impl.md`
(approve). Live acceptance: `docs/test-reports/workflow-catalog-lookup-report.md` and, after
K-056's Ministral re-point, the regression re-verification
`docs/test-reports/workflow-catalog-lookup2-report.md` (both PASS WITH DEFECTS, all five AC hold).
Filed `K-056` out of this capability's own QA gate (D-1: live tool-call skip-and-fabricate under
extended conversations) — still open, tracked separately, not resolved by this delivery.

## 2026-08-30 — K-055: `querygen.compile()` closes the `QueryFilter.op` Layer-1 recheck gap

**What:** `tdd-engineer` closed the one MAJOR finding from the security review's Pass 3 live
adversarial run (`docs/reviews/workflow-nl-query-generation-security.md`, "Finding — MAJOR:
`QueryFilter.op` has no independent `compile()`-level recheck..."). `QueryFilter.op` was the
only splice-worthy DSL field where `compile()` trusted Pydantic's `Literal` constraint alone —
every other field (`label`, `var`, `property`, `returns`/`order_by`) already had an independent
allowlist recheck for the `.model_construct()`-bypass scenario. Not reachable via any current
production caller (the model-driven path always runs full Pydantic validation) — defense-in-depth,
not an active incident.

**Fix:** `compile()` (`server/falkorchat/querygen.py`) now rejects any `filt.op` not in the closed
six-value set (`{"=", "<>", "<", "<=", ">", ">="}`) immediately alongside the existing
`filt.property` allowlist check, before splicing it into the `WHERE` clause template.

**Tests:** one new regression test in `server/tests/test_querygen.py`
(`test_compile_rejects_invalid_op_from_hand_constructed_filter`), mirroring the existing
`.model_construct()`-bypass suite and reproducing the review's exact live payload
(`op="> 0 WITH 1 AS x MATCH (m) DETACH DELETE m WITH 1 AS y RETURN y //"`). Confirmed RED before
the fix (the malicious `op` compiled cleanly), GREEN after, and mutation-tested (temporarily
weakening the new check reproduces the RED failure). Full offline suite: 2290 passed / 14
deselected (was 2289/14 before this change).

## 2026-08-28 — K-056: tool-use breadcrumb + fabrication observability signal — implemented, live-verified as NOT resolving D-1, breadcrumb reverted

**What:** `tdd-engineer` (U37) implemented both pieces of the user-directed K-056 fix pass
(`docs/plans/workflow-salesperson-demo-coordination.md`'s U37, gated on
`docs/reviews/salesperson-tool-reliability-ml.md` §4.1/§4.3/§5) — a tool-use breadcrumb folded
into the replayed conversation history, and a cheap, generalized observability signal for a
fact-bearing answer given with no tool dispatched. **Live verification (2/2 independent runs,
9-turn D-1 repro sequence) shows the breadcrumb does NOT resolve or measurably reduce the
underlying fabrication — collapse still onset at turn 3 in both passes, persisting through
turn 9, including the exact same fabricated `$149.99` price for "Portable SSD 1TB" the ml note's
own repro hit** — and surfaces a new, more concerning failure mode: on every fabricating turn in
both passes, the model's own generated reply text verbatim echoed the replayed-history breadcrumb
format (`"Assistant: <answer> [verified via <tool>]"`) **without ever calling the tool**, i.e. the
customer-visible message now falsely claims tool verification while `Message.toolsUsed` (the real
audit signal) is empty. The observability signal, independently, worked exactly as designed in
both passes: it fired a WARNING on every turn that actually fabricated (turns 3/4/7/8/9, both
passes) and stayed silent on genuine tool-use turns (1/2) and on correct-but-unverified
abstentions (5/6, matching the ml note's own "correct abstention, not tool-verified" distinction).

**Fix (mechanism, initially shipped as-is despite the negative live result — see "Reverted"
below for the breadcrumb's final disposition):**
- `executor.StepResult` gains `toolsUsed: frozenset[str]` — every non-`post_message` tool an
  agent-node execution successfully dispatched (`_run_agent_node`, both the early non-tool-call
  return and the `maxIterations`-exhaustion return).
- `_link_emissions` threads it to `services.link_step_emission` → `repository.link_step_emission`,
  which now also `SET`s it as a `Message.toolsUsed` list property (`docs/QUERIES.md` §12.6) —
  always written, even empty, not left to the read-side `coalesce` fallback alone.
- `repository.read_thread` (`docs/QUERIES.md` §4) now returns `toolsUsed`
  (`coalesce(m.toolsUsed, [])`) alongside the existing message fields.
- `executor._assemble_messages` originally tagged a replayed `assistant` turn with
  `" [verified via <tool>, ...]"` when its `toolsUsed` was non-empty — **reverted, see below.**
- New `executor._looks_fact_bearing` heuristic (a currency-prefixed or two-decimal price-shaped
  number — catalog-vocabulary-free by design) plus `_note_possible_fabrication`, mirroring
  `_note_must_post_violation`'s posture (always logged via `_log.warning`, never gated on a debug
  run; a `possible_fabrication` trace entry on a debug run). Fired whenever a step's own granted
  domain tools (`config.tools` minus `post_message`) are non-empty, none were dispatched this
  execution, and the final text looks fact-bearing — driven entirely off the step's own tool
  grant set, not any hardcoded tool name, so it stays meaningful once K-053/K-054/K-055 add their
  own tools to this scaffold. Kept as-is; the bare two-decimal branch is documented as
  intentionally coarse (advisory `WARNING`-only signal, never blocks) rather than tightened.

**Reverted (U39, `tdd-engineer`, same day):** an `analyst` diff review
(`docs/reviews/salesperson-tool-reliability-impl.md`, MAJOR 1) found the breadcrumb was not a
neutral leftover but a real severity increase on D-1 — the model imitated the breadcrumb's own
surface text in fabricated replies with no tool ever called, and that fabricated text then
replayed into the next turn's history as a self-authored false-verification claim. The tagging
code path in `_assemble_messages` was removed; `StepResult.toolsUsed`, `_link_emissions`'s
threading, `repository.link_step_emission`'s `SET`, and `read_thread`'s surfacing all stay exactly
as shipped — `Message.toolsUsed` is now purely an audit/observability property, never fed back
into a prompt the model reads. Three tests in `test_executor_agent.py` that pinned the breadcrumb
text were reshaped into regression guards pinning its *absence* instead (one of four was folded
into another, net -1 test).

**Tests:** 18 new offline unit/integration tests across `server/tests/test_executor_agent.py`
(breadcrumb tagging/omission — present, absent key, wrong-role defense; `StepResult.toolsUsed`
population on both return paths; the fabrication-warning signal, its negative cases, and its
exhaustion-path coverage), `server/tests/test_repository.py` (`link_step_emission` stamping
`toolsUsed`, its default, `read_thread` surfacing it), and `server/tests/test_executor_produced.py`
(one live-graph integration test driving a real node that dispatches a domain tool then posts,
confirming `Message.toolsUsed` lands correctly end-to-end through executor→services→repository).
Mutation-tested: reverted all three production files (`executor.py`/`repository.py`/`services.py`)
via `git stash`, confirmed all 12 directly-relevant new tests failed for the right reason, restored
and re-confirmed green. Full offline suite (as originally shipped): 1829 passed, 4 deselected (up
from the 1811-passed baseline; +18 new tests, no existing test needed changes).
**After the U39 revert:** 1828 passed, 4 deselected (three breadcrumb-text tests reshaped into two
negative regression guards, net -1); mutation-tested again by temporarily re-adding the tagging
block, confirming the reshaped tests fail for the right reason, then removing it. `./scripts/
test_queries.sh`: 346/346 both times (no query/DDL-shape assertions added there — this is a plain
property `SET`, dialect-verified live via `redis-cli` directly, and exercised for real by the live
`wf_repo`/`conn` pytest fixtures, same posture the module's own testing-hazards doc takes for
non-DDL property additions).

**Live verification (`docs/reviews/salesperson-tool-reliability-ml.md` §2's own method — driving
`services.start_workflow_run`/`WorkflowTrigger.maybe_trigger`→`resume_workflow_run` directly
in-process, `trace=True`, bypassing the `@mention` REST path's default-off trace, no shipped code
changed for this):** fresh throwaway `ws:tdd-d1-fix` (`bootstrap_schema.sh` → `seed_demo.sh` →
`seed_catalog.sh` → `seed_salesperson.sh`), real LM Studio (`localhost:1234`,
`qwen/qwen3-4b-2507`, directly reachable this session — no gateway-IP workaround needed). Ran the
identical 9-turn D-1 sequence twice, independently (fresh thread each pass). Both passes: turns
1-2 correct with real tool calls and a correctly-persisted breadcrumb; turn 3 onward, the model's
first LLM iteration emits zero `tool_calls` and fabricates, exactly the pre-fix pattern, never
recovering through turn 9. `GRAPH.DELETE ws:tdd-d1-fix` after the pass; `reference`'s catalog/def
data (idempotent create-only) and `ws:acme` were untouched by the repro itself, but
`./scripts/test_queries.sh`'s own teardown (run separately, for the regression check) wiped
`reference` as documented — re-seeded (`bootstrap_schema.sh` → `seed_demo.sh acme` →
`seed_workflows.sh acme` → `seed_catalog.sh` → `seed_salesperson.sh acme`) and re-verified in sync
(`verify_workflows.sh acme`, `verify_salesperson.sh acme`, `verify_catalog.sh`, all OK) before
finishing.

**Disposition:** D-1 (K-056) is **not resolved** by this fix pass and is still open. The
observability signal is real, working, generalized value and stays. The breadcrumb mechanism was
live evidence-negative (2/2 runs: no change to the model's underlying skip-and-fabricate behavior)
**and** was found, on review, to actively worsen the defect (the model imitating the breadcrumb's
surface format without doing the verification it advertises, feeding a false claim into the next
turn's replayed history) — so it was reverted rather than left in place. `Message.toolsUsed`
survives the revert as a pure audit/logging property with independent value. Per user direction
("stop after the defect is fixed"), no further mitigation iteration and no K-053 dispatch this
session — routed to `analyst` (U38) for a diff-scoped review, which recommended the revert; U39
executed it. `docs/BACKLOG.md`'s K-056 entry updated to reflect this precisely. `reference`'s
schema/data (wiped by `test_queries.sh`'s teardown, once per verification pass) was re-seeded and
re-verified in sync after both passes.

## 2026-08-26 — K-048: `_assemble_messages` no longer emits two consecutive same-role turns

**What:** `tdd-engineer` closed K-048 — `WorkflowExecutor._assemble_messages`
(`server/falkorchat/executor.py`) unconditionally appended a trailing `user`-role `CONTEXT` block
after the role-mapped thread turns, which crashes a strict-alternation chat template
(live-confirmed: LM Studio's Ministral-3B, HTTP 400) whenever the thread's last turn was itself
`user`-authored — a structural, not incidental, shape: `intake`'s first call and every
`research`→`answer` handoff (`research` never posts, so `answer` always sees a `user`-terminated
thread) both hit it. Implemented exactly per the approved plan,
`docs/plans/assemble-messages-alternation.md`, gated pre-implementation by
`docs/reviews/assemble-messages-alternation.md` (`analyst`, approve with suggestions).

**Fix:** a new module-level helper, `_append_turn(messages, role, content)`, placed next to the
existing `_assistant_turn` helper — merges a turn into the previous message (joined by `"\n\n"`)
instead of appending a new one whenever that would produce two consecutive same-role messages, and
is a no-op whenever the sequence is already alternating. `_assemble_messages` now routes both its
thread-turn appends and its trailing `CONTEXT` append through this helper; its signature is
unchanged. `_assemble_messages`'s docstring was extended (not left stale) to describe the new
coalescing invariant, per the review's suggestion 1.

**Tests:** three new offline unit tests in `server/tests/test_executor_agent.py` — a
characterization test pinning today's already-correct shape (thread ending in `assistant`, 4
messages, zero merges — confirmed passing against unmodified code before any production change);
the crash-shape test (thread ending in `user`, merges to 2 messages, `["system", "user"]`);
and the sibling-shape test (two consecutive `user` thread turns, no `assistant` between them,
merges to one coalesced `user` turn) — promoted from the plan's "recommended" to mandatory per
the review's suggestion 2, reusing the existing `_thread_rows(n)` fixture. Mutation-tested: with
`_append_turn`'s merge condition temporarily disabled, both the crash-shape and sibling-shape
tests failed for the right reason (extra `"user"` entries in the role list); the characterization
test still passed. Full offline suite after reapplying the fix: 1785 passed, 4 deselected (`-m
live` only) — exactly the pre-change baseline (1782 passed) plus the 3 new tests, no existing
test needed changes. SHA-lock on `_drive_loop` re-verified unchanged (`71055f756280`) before and
after. Live regression pass reused the existing test per the plan (no new live fixture):
`pytest -m live -s tests/test_workflow_live.py::test_triage_flow_runs_end_to_end_against_live_llm`
— 1 passed against LM Studio's Qwen, confirming the merged-message shape still drives triage's
intake→research→answer flow correctly.

**Review:** `analyst` gated the plan pre-implementation (approve with suggestions, all three
folded in: the docstring-drift fix, promoting the sibling-shape test to mandatory, and the third
finding — a plan-prose cross-reference nit — had no code implication), then re-gated the
implemented diff (Pass 2, same review doc): **approve**, no new findings, all three Pass-1
findings independently confirmed resolved.

## 2026-08-26 — K-049: app-side guard against the FalkorDB `UNIQUE`-constraint oversized-value crash

**What:** `tdd-engineer` closed K-049 — a shared-instance FalkorDB crash first found during K-028
(an oversized `Step.key` SIGSEGVed the whole `redis-server` process, not just the query). Root
cause had already been established and `analyst`-approved on 2026-08-21
(`docs/reviews/unique-constraint-oversized-value-crash-rca.md`): FalkorDB v4.18.11 segfaults the
entire instance — a null-pointer dereference in `EnforceUniqueEntity` — when a `CREATE`/`MERGE`
commits a value >4096 bytes into a property backed by a `GRAPH.CONSTRAINT ... UNIQUE` constraint.
Independently reproduced three times total, always in disposable containers, never against
`falkordb-dev`: the original RCA, the `analyst` gate on it, and a third confirmation by `graph-dba`
on 2026-08-26 while finalizing the fix design (`docs/plans/oversized-indexed-property-guard-graph.md`)
— `falkor-chat`'s coordination and backlog text had never been updated after the original RCA
landed, so this work opened believing root cause was still unestablished; that was corrected in
place once found (`docs/plans/oversized-indexed-property-guard-coordination.md`).

**Fix:** REST is not exposed to this crash (`schemas.py`'s `MAX_KEY_LEN = 200` already bounds every
field reaching a constrained property via `publish_workflow_def`) — the gap was every non-REST
caller (tests, scripts, a future MCP tool) and the `materialize_def` path, neither of which had a
service-layer length check. A new `Services._validate_key_lengths` helper (reusing `MAX_KEY_LEN`,
mirroring the existing `MAX_CONFIG_LEN` service-layer-mirror precedent) is now called from both
`publish_workflow_def` and `materialize_def`, raising `WorkflowDefSpecError` before any Cypher
reaches the graph. Deliberately does not bound `transition.on` — it never feeds `Step.stepUid`'s
`MERGE` key, so it isn't at risk of this crash class. `api.py`'s materialize route also gained the
`Path(max_length=...)` bound its sibling route already had.

**Tests:** a 5-case parametrized publish test (oversized key/version/step-key/transition-from/
transition-to), a negative-space test proving `transition.on` is deliberately excluded, a
`materialize_def` test planting an oversized key directly into stored `reference` data (simulating
a corrupted/hand-edited def bypassing publish), and two REST 422 tests. Mutation-tested in two
rounds — the first caught a real test-construction bug (two cases were passing for the wrong
reason, via a pre-existing unrelated check) — fixed before the second, clean round. Full offline
suite: 1782 passed, 4 deselected (`-m live` only).

**Review:** `analyst` gated in three passes on the same review document — Pass 1 (2026-08-21,
original RCA, approve with suggestions), Pass 2 (2026-08-26, scoped gate on the finalized design,
approve with suggestions — two minor citation/audit-completeness findings, both folded into the
implementation brief), Pass 3 (2026-08-26, diff review, **approve**, no blockers — independently
re-ran the full suite and confirmed the exact pass count, verified the guard is unbypassable on
both write paths). Upstream FalkorDB issue filing recommended to the user (genuine engine defect);
not filed by any agent.

## 2026-08-26 — K-045: `llm-provider-config` FR-10/AC-8 wording corrected against shipped behavior

**What:** `tico` corrected the stale "the run suspends" wording in
`docs/requirements/llm-provider-config.md`'s FR-10 and AC-8 — both described an unresolvable or
failing use-time model as suspending the run; the actual shipped Landing 2 behavior (confirmed by
the K-042 Landing 2 QA acceptance pass and the D-2 fix, both 2026-08-11, below) is that the run
**fails**, with the cause recorded (`status: 'failed'`, an `error` field) — the executor's ordinary
terminal-failure vocabulary, not a `human`/`wait`-style suspend/park. Because the source document
is `archived` (executed against by K-042), the correction landed as a successor document,
`docs/requirements/llm-provider-config2.md`, carrying `Supersedes:`/`Superseded by:` header
pointers per root `AGENTS.md`'s collision rules, rather than an in-place edit. No code or behavior
change — documentation accuracy only. Closes K-045 (`docs/BACKLOG.md`), filed out of the K-042
close.

## 2026-08-25 — K-051: `Document.status` now reaches a terminal state

**What:** `tdd-engineer` fixed `Document.status` never leaving `'processing'` — no code path
anywhere in `server/falkorchat/*.py` ever wrote `'ready'`/`'failed'`, a defect `qa-engineer`'s
K-050/M5 acceptance pass found live (`docs/test-reports/document-ingestion-report.md` Defect 1)
and filed as K-051. Coordinated via
`docs/plans/document-status-terminal-coordination.md`.

**Fix:** a per-document outstanding-job counter, `Document.pendingJobs`. `create_document`
(`repository.py`) now seeds it to `0`. `background._schedule_chunk_processing` calls the new
`repository.start_document_progress` synchronously, before scheduling any per-chunk job, to set
the real total (one job per chunk per wired worker — embed + extract counted independently); a
zero-chunk document flips straight to `'ready'` in that same call rather than parking forever.
Each of `_safe_embed_chunk`/`_safe_extract` now reports its own outcome back via a shared
`_report_document_job` helper (same failure-isolation discipline as every other `_safe_*`
wrapper — never raises into the scheduler) to the new `repository.report_document_job_done`,
which decrements the counter and flips `status`: to `'failed'` the instant any single job fails
(regardless of how many others are outstanding), or to `'ready'` once every job has succeeded
and none has failed. Both writes guard on `status = 'processing'` so the first terminal write
wins — a late report after a document already reached a terminal state can never flip it back.
`EmbeddingWorker`/`IngestionPipeline` each gained a public `.repo` accessor so
`background.py` can source the repository without threading a new parameter through every
`_schedule_chunk_processing` call site; a worker/pipeline with no `.repo` (every pre-K-051 test
fake) silently skips the report rather than raising.

**Tests:** 19 new tests across `server/tests/test_repository.py`, `test_background.py`, and
`test_api.py` — repository-level counter/status-transition coverage (including the zero-chunk
edge case and late-report-after-terminal-state guards), `background.py` unit tests (including
that `_report_document_job` swallows a raising repo, matching every sibling `_safe_*` wrapper's
existing test convention), and two live integration tests driving the real background pipeline
to a terminal `status` via REST. TDD discipline throughout (RED confirmed before each fix,
mutation-tested after). Full offline suite: 1751 → 1773 passed / 4 deselected.

**Review:** `analyst` gated in two passes (`docs/reviews/document-status-terminal.md`) — Pass 1
approved with suggestions after independently reproducing the fix live against `falkordb-dev`
(one MAJOR: `docs/QUERIES.md` §14 hadn't been kept current; two MINOR: the zero-chunk edge case
and the untested swallow path); Pass 2 confirmed all three resolved and verdict **approve**.
`docs/QUERIES.md` §14.1's `create_document` literal and new §14.1a now document both new
queries.

## 2026-08-25 — K-005: `_read_structure` false-negatived a live snapshot after a fully-deleted `reference` graph key

**What:** `tdd-engineer` fixed a false negative in `Repository._read_structure`
(`server/falkorchat/repository.py`), the shared static helper both
`read_def_structure` (`reference`) and `read_snapshot_structure` (`ws:{id}`)
delegate to. After `./scripts/test_queries.sh` `GRAPH.DELETE`s the `reference`
graph key entirely (not just its node data), a read against the now-nonexistent
key raised an uncaught `redis.exceptions.ResponseError` ("empty key") — a
different failure mode from "key exists, root node absent," which already
returned `None` cleanly. Because `Services.diff_def_snapshot` reads `reference`
before `ws:{id}`, the uncaught exception propagated out before the snapshot side
was ever checked, and `verify_workflows.sh`'s outer catch collapsed the whole
diff to the blanket `ABSENT` sentinel — misreporting an intact `ws:<id>`
snapshot as MISSING. Root cause independently verified by `coder` during team
kaizen distillation (`claude/coder/kaizen/plan.md` K-005); this unit
implemented the fix (coordinated via
`docs/plans/workflow-diff-absent-key-coordination.md`).

**Fix:** `_read_structure` now catches the "empty key" `ResponseError` and
returns `None` — matching what it already returns for an absent root node, and
the same catch-and-return-`None` pattern already used for this exact FalkorDB
behaviour at `read_index_dimension` and `services._read_or_absent`. Any other
`ResponseError` still propagates. One change fixes both callers
(`read_def_structure`, `read_snapshot_structure`) and, transitively,
`get_workflow_def_structure`/`get_snapshot_structure`/`diff_def_snapshot`/
`check_demo_readiness` — `diff_def_snapshot`'s own docstring already promised
this contract ("one side missing is a first-class 200"); it just wasn't true
for a fully-deleted key. The separate `_read_subgraph` helper (feeds
`materialize_def` and the SHA-locked executor path) was not touched.

**Tests:** `server/tests/test_repository.py` — a live reproduction
(`test_read_snapshot_structure_none_when_graph_key_fully_deleted`, a genuine
`GRAPH.DELETE` on a throwaway `ws:<probe>` key, not the shared `reference`
graph) plus two fake-graph unit tests pinning `_read_structure` directly
(`test_read_structure_none_when_graph_key_fully_deleted` for the `reference`
side, `test_read_structure_reraises_response_errors_that_are_not_empty_key` so
the fix doesn't swallow unrelated errors). Confirmed red-then-green and
mutation-tested (removing the `except` reproduces the original failure).

**Docs:** `falkor-chat/AGENTS.md`'s `test_queries.sh` row now names the full
`bootstrap_schema.sh` → `seed_demo.sh` → `seed_workflows.sh` remediation
sequence (indexes/constraints were destroyed along with `reference`'s data, not
just the data); its `verify_workflows.sh` row notes this false-negative case
existed and is fixed.

## 2026-08-25 — K-050 M5 closes: ingestion pipeline & entity fusion — acceptance pass + design sync

**What:** `qa-engineer` ran the milestone-closing acceptance pass over AC-1..AC-10
(`docs/test-plans/document-ingestion.md`, `docs/test-reports/document-ingestion-report.md`) —
**verdict PASS-with-parked-defects**. All ten ACs were live-verified against the running server
(real REST and MCP calls over the wire, real LM Studio) rather than only re-running the offline
suite: AC-9's byte-identical round trip, AC-6's cross-transport MCP-write/any-transport-read (a
genuine out-of-process MCP Streamable-HTTP client), AC-10's traced extraction, AC-1's kept
conflicting facts, AC-2/AC-3's real-LLM auto-merge and fuzzy-pending observations, AC-4's
confirm/reject audit fields, AC-7's manual `recheck_match` path, and AC-8's real bulk `ingest_documents`
cross-document fusion (the new Stage 6a batch-API-altitude coverage). AC-5 was verified by re-running
its existing live-marked test. Full offline suite (1751 passed/4 deselected) and query suite
(320/320) held as unchanged green baselines.

Two Medium defects surfaced, both independently confirmed by `teco`:
- **`Document.status` never reaches a terminal state** (`'ready'`/`'failed'` are never written by
  any code path) — doesn't block any AC; parked as a backlog follow-up (`BACKLOG.md`), not fixed in
  this milestone.
- **`docs/DESIGN.md` §5.1/§7.1 were stale** against the shipped schema (missing the `RELATES_TO`/
  `SAME_AS` edges, `Entity.nameNormalized`, and their indexes) — this one gated M5 directly, since
  the plan's own done-condition names DESIGN §5.1/§7 sync as a closing requirement. `graph-dba`
  fixed it (cross-checked against `bootstrap_schema.sh`/`QUERIES.md`/`repository.py`, independently
  re-verified by `teco` against the same ground truth rather than a fresh `analyst` gate, given how
  tightly scoped and low-risk the factual sync was).

**M5 ✅ — ingestion pipeline & entity fusion is closed.** All six implementation stages
(chunking, chunk embeddings/standalone search, extraction, fusion, chat-grounding integration,
bulk ingestion) delivered, diff-gated, committed; QA acceptance PASS-with-parked-defects on green
baselines; DESIGN.md synced. The full requirements/plan/graph-note/ML-note/coordination/reviews/
test-plan/test-report document family is flipped to `Status: archived` in the same close.

## 2026-08-25 — K-050 M5 Stage 6a: document ingestion — bulk ingestion (FR-11)

**What:** The sixth and final staged slice of the ingestion pipeline (`docs/plans/
document-ingestion.md`, "Stage 6 — Batch hardening + QA acceptance"), implementing `ingest_documents`
(FR-11 bulk ingestion) at all three layers — a real scope gap `teco` found while orienting for Stage
6: `ingest_documents` was never built in Stages 1-5, only the singular `ingest_document`. Per plan
§3.6 (already gated, not a design question): `Services.ingest_documents(ctx, *, documents:
list[dict]) -> list[dict]` loops `Services.ingest_document` per item, returning **one receipt per
item** — no batch-aware fusion logic, since fusion always reads the graph's *current* state
(including sibling documents from the same batch, once their entities land), never a batch-local
view; AC-8's cross-document fusion falls out naturally once each item's independent background
extraction runs. MCP: `ingest_documents(items: list[dict])`. REST: `POST /documents/batch`
(`IngestDocumentsIn`, `schemas.py`). Capped at `MAX_BATCH_SIZE = 20` (`BatchTooLargeError`, maps to
400) — enforced in the service, not only the REST pydantic boundary, mirroring `MAX_DOCUMENT_CHARS`'s
posture. **Per-item failure isolation** (implementer's call, plan left this open): one bad document
does not abort the batch — it comes back as that item's own `{"status": "error", "error": ...,
"errorType": ...}` receipt, and chunk embedding/extraction is scheduled only for items that actually
succeeded. The per-chunk embed+extract scheduling block — previously duplicated inline between
`api.py` and `mcp.py` — is now one shared helper, `background._schedule_chunk_processing`,
parameterized over each transport's own scheduling primitive; backported to both existing singular
call sites too (verified behavior-preserving, not just refactored on faith). New tests across
`test_services.py`, `test_api.py`, `test_mcp.py`, and a batch-altitude AC-8 test in
`test_ingestion.py` (`test_batch_ingest_two_documents_mentioning_the_same_entity_fuse` — drives the
real `Services.ingest_documents` call, not just `extract_chunk` again, then completes extraction
synchronously and asserts a confirmed cross-document `SAME_AS` edge). `docs/QUERIES.md` §14.4 gained
the `ingest_documents` row, batch-semantics paragraph, and shared-helper note.

**Diff-gate fixes (Pass 6, `docs/reviews/document-ingestion-impl.md`):** `analyst`'s diff-scoped code
gate found one BLOCKER, one MAJOR, one MINOR — none in the shipped stage design itself (§3.6
conformance, the shared-helper refactor/backport, the AC-8-at-batch-altitude test, `MAX_BATCH_SIZE`
enforcement, and route/schema conventions all verified clean). **BLOCKER** — `doc["text"]` was
evaluated before the per-item `try`, so a malformed batch item (missing/wrong-shaped `text`) raised a
bare `KeyError` that aborted the *entire* `ingest_documents` call — directly contradicting the
method's own per-item-isolation guarantee, and reachable specifically via the MCP tool (REST is
protected by `IngestDocumentIn`'s pydantic validation; MCP's `items: list[dict]` has no schema layer
at all, exactly the transport the guarantee exists for). Worse than a clean rejection: an
already-ingested sibling item ahead of the malformed one in the list had its `Document` written but
its receipt never returned, since the exception aborted the loop before `return receipts`. Fixed:
each item's shape (`isinstance(doc, dict)` and `isinstance(doc.get("text"), str)`) is validated
explicitly before dispatch, turned into a `{"status": "error", "errorType": "MalformedItemError"}`
receipt rather than raising; the `except` clause is also widened to `(ServiceError, KeyError,
TypeError, AttributeError)` as defense in depth. **MAJOR** — the MCP per-chunk thread fan-out
(already escalated Pass 2 → Pass 3, ~500 → ~1,000-1,200 threads/call) now compounds by up to
`MAX_BATCH_SIZE` (20x, ~23,000 threads for a max-size batch) with no new mitigation or documentation,
in the stage literally named "batch hardening." Not re-mitigated with a scheduling redesign this pass
(judged out of scope for a documentation-flagged MAJOR — REST is unaffected, per-thread failures are
already isolated, and the fuller bounded-pool fix has been deferred three passes running pending a
coordinator scope decision on whether it belongs in this feature or a standalone follow-up); the
compounded number and its reasoning are now documented explicitly in `mcp.py`'s `_default_schedule`
docstring and `docs/QUERIES.md` §14.4, mirroring Pass 3's own precedent for the un-compounded number.
**MINOR** — this entry itself; Stage 6a's diff had no `docs/HISTORY.md` entry, breaking the exact
precedent every one of Stages 1-5 set.

## 2026-08-25 — BACKLOG.md: delivered work leaves the backlog entirely (doc-only)

**What:** `docs/BACKLOG.md` 8,807 → 7,997 w. Three sections removed under the amended
module-documentation convention (root `AGENTS.md`): **a delivered item is not kept in a backlog at
all, not even as an index row** — its record is `HISTORY.md`, or the design surface that owns it
when the fact is a live constraint on the system rather than a record of work.

- **The 25-row `## Delivered` index is gone.** It was created on 2026-08-21 (entry below) as the
  right answer under the then-current rule; the rule changed. Gate check first: all 25 `K-` ids
  (K-008 … K-047) confirmed present in this file before removal — the 2026-08-21 pass had already
  verified body-level coverage, and K-023's one durable fact still sits in `DESIGN.md` §6.2 /
  `QUERIES.md` §12.6.
- **The milestone table keeps only what is open** — M5 🟡 and the deferred M2.5 track. M1/M2/M3/
  M3.5/M4's rows carried delivery dates and QA verdicts that this file and `DESIGN.md` §12 both
  already hold. Renamed `## Milestone-to-green map` → `## Milestones still open`.
- **`## Standing decisions` is gone, and nothing was routed out of it** — all four bullets restated
  `DESIGN.md`: "M2 green = functional GraphRAG" (§12 item 3), the identity-authoritative decision
  (§1.2, which the bullet itself cited), the M2 model stack (§1.3), and the `EMBEDDING_DIM` rule
  (§1.3 + §7.1). The **one** fact not covered anywhere — that `bootstrap_schema.sh`'s `1536`
  default is wrong for every workspace here and `start_server.sh` already passes `1024` — was
  written into `falkor-chat/AGENTS.md`'s bootstrap line, where an agent actually reads it.
- **Header corrected on two counts:** it promised a `## Delivered` index that no longer exists, and
  claimed milestone status is authoritative in the backlog "not in DESIGN.md or any README" — now
  true only of an *open* milestone.

**Why:** stakeholder ruling, 2026-08-25 — *"completed milestone/task related information [does not
belong] along the backlog … in the future we will not even have a file for that (i.e. it will be
moved to the graph, similar to the team kaizen)"*. Keeping finished work out of a backlog is what
makes that migration a move rather than a cleanup. Same pass applied `docs/BACKLOG.md` (root) and
the 13 `claude/*/kaizen/plan.md` files; full record in
`claude/docs/plans/prompt-waste-reduction.md`, Stage G.

**Not touched:** every open item (K-016…K-018, K-029…K-050) verbatim, `## Plan docs still to
author`, and the parking lot.

## 2026-08-25 — K-050 M5 Stage 5: document ingestion — chat-grounding integration (FR-2)

**What:** The fifth of 6 staged slices of the ingestion pipeline (`docs/plans/document-ingestion.md`,
"Stage 5 — Chat-grounding integration"), generalizing the K-013 `EMITTED` agent-answer provenance
edge to cite an ingested `Chunk`, not just a chat `Message`, and wiring GraphRAG retrieval to draw
from both pools. Per graph-dba's finalized design (`docs/plans/document-ingestion-graph.md` §3,
essentially implemented verbatim): the `EMITTED` write-side (`post_agent_answer`/
`post_agent_answer_first`) now resolves `$seedIds` against both `Message.msgId` and `Chunk.chunkId`
via a two-label `OPTIONAL MATCH` + `coalesce` pair — the same bare-id idiom already used for
author/mention resolution, no namespaced id scheme, no parallel `kind` list (both id spaces are
disjoint `uuid4` generators). `read_provenance`'s response shape changes: `{seedMsgId, text, role,
score, rank}` → `{seedKind, seedId, text, role, documentId, documentTitle, score, rank}` — the
`(:Chunk)<-[:HAS_CHUNK]-(:Document)` provenance hop folds into the same read, one extra `OPTIONAL
MATCH`, `s` deliberately kept unlabeled (a plain traversal endpoint, no `Node By Label Scan` risk).
`read_citing_answers`'s `seed_msg_id` kwarg renamed `seed_id`, same resolution idiom, row shape
unchanged (an answer is always a `Message`). `Services.hybrid_search` now merges `repository.
hybrid_search`'s `Message` ANN pool with `repository.search_chunks`'s `Chunk` ANN pool (both
unchanged, called with the same `q_vec`/`k`) **app-side** — no combined-ANN Cypher shape exists for
this, graph-dba's note is scoped to the `EMITTED` generalization only, so the merge (sort by cosine
distance ascending, truncate to `limit`, tag each row `seedKind`) is a deliberate, documented
implementer design choice. `channel_id` scopes only the `Message` pool; a `Chunk` seed has no
channel/thread to scope by. `responder.py`'s `maybe_respond` and the `graphrag_retrieve` MCP/workflow
tool (`tools.py`) both updated to resolve a seed's id generically off `seedKind` instead of assuming
`msgId`.

**Diff-gate findings (Pass 5, `docs/reviews/document-ingestion-impl.md`):** first pass verdict
**needs changes** — one BLOCKER: the merged `hybrid_search` row shape broke `GraphragRetrieveTool.run`
(`tools.py`), a shipped, workflow-granted tool (the `research` node in `scripts/seed_workflows.sh`)
still doing unconditional `r["msgId"]` — `KeyError` on any `Chunk`-seeded hit, live-reproduced, not
covered by any existing test (its stub and live-integration tests were all `Message`-only). Fixed:
the tool now resolves the id the same way `responder.py` does and additionally surfaces `documentId`
for a `Chunk` hit; new stub + live test coverage seeds a real `Chunk` through the fix. Two MINORs,
both resolved rather than deferred: **(1)** the eval harness (`test_retrieval_eval.py`) had the same
unguarded `r["msgId"]` pattern, silently non-triggering only because `ws:eval`'s golden set has no
ingested `Chunk`s — now explicitly filters to `Message`-shaped rows with a documented rationale.
**(2)** the plan's AC-5 test-strategy row asks for both a mocked-LLM integration test and a
live-marked real-LLM e2e; only the former had shipped — added `test_ac5_chat_grounding_live.py`
(`pytest.mark.live`, mirrors `test_workflow_live.py`'s gating, its own throwaway `ws:live5`
bootstrapped at the probed real embedding dimension). One NIT (undocumented merge tie-break + the two
ANN sub-queries running sequentially, not a defect) also folded in as a docstring note. Re-gate:
**approve** — all findings independently confirmed closed, one non-blocking design note (the tool's
new `documentId` is an opaque id, not a title — reasonable for a scoped fix, flagged as a possible
follow-up).

**Why:** AC-5 — an agent's chat answer grounded in ingested content now carries provenance back to
the source chunk/document, traversable exactly like the existing `Message`-seeded `EMITTED` edges,
closing FR-2.

**Tests:** `test_provenance.py` rewritten (not just extended, per the plan's own flag) for the new
response shape, plus new `Chunk`-seeded and mixed-seed forward/reverse read coverage;
`test_services.py` gets `hybrid_search` merge coverage (score-ascending ordering, truncation to
limit, all-Message/all-Chunk/mixed pools, channel-scope-forwarded-to-Message-pool-only);
`test_responder.py` gets Chunk-seeded/mixed-seed id-threading tests plus the named, non-vacuous AC-5
mocked-LLM integration test; `test_tools.py` gets the BLOCKER-pinning stub and live tests;
`test_retrieval_eval.py` gets the explicit Message-only filter; a new `test_ac5_chat_grounding_live.py`
(live-marked, deselected by default). Suite: offline `pytest -q` 1712→1725 passed (3→4 deselected —
the one new live-marked file); the new live test independently run twice (once by the implementer,
once by `teco`, once again by `analyst`'s re-gate) — `1 passed` each time.

Mutation-tested: reverted the write-side `coalesce(seed.msgId, seed.chunkId)` back to bare
`seed.msgId` — the two new `Chunk`-seed provenance tests failed as expected (`teco` independently
reproduced this one too); disabled `services.hybrid_search`'s `merged.sort(...)` — the two
merge-ordering tests failed as expected (`analyst` independently reproduced this one during the
gate); reverted `tools.py`'s id-resolution ternary back to unconditional `r["msgId"]` — all three
`Chunk`-touching `graphrag_retrieve` tests failed with the original crash (reproduced independently
by both `teco` and `analyst`). All reapplied/confirmed passing.

**Process note:** the analyst's own mutation-test spot check during Pass 5 briefly used `git checkout
--` on an uncommitted file mid-review, which would have destroyed the legitimate diff, not just the
planted mutation — caught immediately, recovered by hand from already-captured diff text, confirmed
byte-identical before proceeding. No work was lost, but it's a standing lesson: a mutation-test spot
check on an *uncommitted* diff should copy the file aside (or use `Edit` to plant/revert) rather than
`git checkout --`/`restore`, which reverts to `HEAD` and cannot distinguish "the planted mutation"
from "everything else not yet committed." All three of this session's later mutation-test checks
(by `teco` and by `analyst`'s own re-gate) used the safer approach.

**Scope boundary:** Stage 5 only — no batch hardening or QA acceptance pass (Stage 6).
`document-ingestion.md`/`-graph.md`/`-ml.md`/`BACKLOG.md` untouched (locked design/tracking
documents).

## 2026-08-25 — BACKLOG.md restated as forward-looking: delivered bodies to an index, five stale claims fixed

**What:** `docs/BACKLOG.md` had become a second change log. 1317 of its 2025 lines (65%) were the
bodies of already-delivered items — each ending in a "see HISTORY.md" pointer and then narrating
the implementation anyway. K-027 alone ran 315 lines with all five sub-items closed; K-034 kept its
full pre-implementation blast-radius table of "thirteen shipped assertions [that] are falsified"
three weeks after the sweep that corrected all thirteen. The same accretion had pushed K-050, the
only in-progress item, to line 1746 of 2025.

**Method (mirrors the 2026-08-24 DESIGN.md v1.0 pass):** restate in the present tense, split by
kind, verify before deleting.

- **Coverage check first.** Every delivered item's `HISTORY.md` coverage was confirmed before its
  body was removed. 19 of the 20 largest have a dedicated dated entry. **K-023 has none** — it
  shipped inside K-022 Landing 2 — so its one durable fact, the D2 `PRODUCED`-vs-`EMITTED`
  disambiguation, was confirmed present in `DESIGN.md` §6.2 and `QUERIES.md` §12.6, both marked
  "locked", with the Option-B delivery note in the K-022 Landing 2 entry here.
- **Delivered bodies → a 25-row `## Delivered` index** (item, title, date, milestone). No `K-`
  number was lost; the only identifiers that disappear are K-001/K-002/K-003/K-005, which occurred
  solely as incidental cross-references inside removed bodies (two pointing at other components'
  kaizen plans).
- **Reordered** to Active → milestone map → standing decisions → open follow-ups → deferred M2.5 →
  Delivered → plan docs → parking lot, so the live milestone is first rather than 86% down.
- **Header block** conformed to root `AGENTS.md` (`Status:`/`Owner:`/`Tracks:`), replacing a
  five-generation `Prior review … Prior review …` chain that ran 46 lines before the first content.

**Five stale claims corrected while reading closely:**

| Site | Was | Now |
|---|---|---|
| `## Active` | "Milestone M3 … in progress. Remaining to M3 ✅: K-025 — unblocked, **not yet run**" | M3 closed 2026-07-21, K-025 delivered the same day. Section rewritten to M5's stage state, verified against `docs/plans/document-ingestion-coordination.md` |
| K-033 | "K-027 is itself 🟡 in-progress; items 2–5 remain open" ⇒ K-033 might ride item 2's re-lock | K-027 closed 2026-08-21 and item 2 shipped at `_run_agent_node`, **outside** the lock — as K-033's own scepticism predicted. No bundling opportunity left |
| M5 milestone row | `document-ingestion-graph.md` / `-ml.md` "(not yet authored)" | Both exist |
| `EMBEDDING_DIM` note | "`start_server.sh` guidance defaults to 1536 too — fold the 1024 note into both in the K-008 gate" | `start_server.sh:92` has defaulted to 1024 since that gate closed; only `bootstrap_schema.sh:264` still defaults to 1536 |
| Parking lot | DESIGN §13's "workflow guard expression language" listed open | `DESIGN.md:722` records it resolved and struck through. The `Entity` extraction-pipeline entry is superseded outright by K-050 and dropped |

**Verified, not assumed:** the `_drive_loop` SHA lock recomputed with the DESIGN §6.2 `awk` recipe →
`71055f756280`, intact — which is what establishes that K-027 item 2 never broke it. Every relative
link in the rewritten file resolves.

**Rotting counts replaced by the command that regenerates them** — K-033's re-lock scope
("`BACKLOG.md` ×2, `HISTORY.md` ×2"; actually 4 and 11, and 40+ repo-wide) and the K-031 re-slug
nit ("4 occurrences across 3 files"; actually four files).

**"Recommended plan docs"** listed 16 rows under a heading reading "not yet created"; 13 existed.
Kept the 2 that don't, renamed **before creation** `m2-auth-tenancy.md` → `auth-tenancy.md` and
`m2-realtime.md` → `realtime-push.md`: the root `AGENTS.md` filename grammar forbids a new
basename beginning with `m<digit>`, and neither topic *is* a milestone.

**Not done, deliberately:** open items were left intact. On a close read only K-033 was
over-argued; K-032 and K-035 are dense but every paragraph is a constraint an implementer needs,
so cutting them would have been editing content, not compressing narrative.

**Result:** 2025 → 717 lines (**-65%**). No open item's scope changed. Delivered in two commits —
`0f48b8a` (the coverage-checked deletion) and `88bb71b` (structure + the five fixes).

## 2026-08-24 — K-050 M5 Stage 4: document ingestion — entity fusion (FR-6/7/8/9/10, OQ-1/2/3)

**What:** The fourth of 6 staged slices of the ingestion pipeline (`docs/plans/document-ingestion.md`,
"Stage 4 — Fusion"), following Stage 3's advisory checkpoint (a real-content extraction-quality
review, `docs/reviews/document-ingestion-ml.md` — no design change, but confirmed a real 30%
type-inconsistency rate across repeat entity mentions worth knowing about here). New
`falkorchat.fusion` module: `find_fuzzy_candidates` (RediSearch fuzzy full-text against `Entity.name`,
type-filtered) and `classify_fuzzy` (`'suggested'`/`'none'`, no calibrated threshold in v1 — ML note
§4.3) — the FR-9 suggested tier only. The FR-8 exact tier is **not** a separate lookup: the
plan-gate review's BLOCKER (a three-round-trip check-then-act race that could silently defeat
FR-8's zero-review guarantee under real concurrency) is closed by folding the candidate lookup, the
new `Entity`'s `CREATE`, and its conditional auto-link into one atomic `GRAPH.QUERY`,
`repository.create_entity_with_auto_match` — live-verified ordering (`document-ingestion-graph.md`
§1.8): the candidate `MATCH` binds strictly before the new entity's own `CREATE`, so a call can
never self-match. New `create_or_reopen_match` (the FR-9/OQ-3 find-or-create-or-reopen write, an
undirected lookup + guarded double-`FOREACH`, never a fixed-direction `MERGE`) plus the FR-10 audit
surface — `confirm_match`/`reject_match`/`recheck_match`/`list_pending_matches`/`list_matches` — the
last closing the plan-gate review's own MAJOR finding (a `WHERE $status IS NULL OR r.status =
$status` null-guard silently drops the `SAME_AS.status` index on this build even when bound to a
real value; fixed as two separate query strings instead). `IngestionPipeline.extract_chunk` now
creates every entity via `create_entity_with_auto_match`; when it reports no exact match, a new
`fuse_entity` method runs the fuzzy tier, isolated per-entity via new `background._safe_fuse`
(inline from the per-entity loop, not scheduled as a separate background task — the fuzzy lookup can
only run once the entity exists, one level finer-grained than `_safe_extract`'s per-chunk isolation).
Every `SAME_AS`-anchored query matches its endpoints unlabeled (`(a)`/`(b)`, never `(a:Entity)`) per
the graph note's live-verified planner trap — a bare label forces a full label scan even though the
relationship-property scan alone is fully selective. `services.py`/`mcp.py`/`api.py` wire the five
match-lifecycle operations as MCP tools + REST routes (`GET /matches/pending`, `GET /matches`,
`POST /matches/{id}/{confirm,reject,recheck}`). `bootstrap_schema.sh` gains the `SAME_AS.matchId`/
`SAME_AS.status` relationship-scoped indexes and a `UNIQUE RELATIONSHIP SAME_AS PROPERTIES 1
matchId` constraint, index-before-constraint as always. `docs/QUERIES.md` §14.6 documents all eight
Cypher shapes.

**Diff-gate findings (Pass 4, `docs/reviews/document-ingestion-impl.md`):** verdict approve with
suggestions, no blockers/majors. Two MINORs, both fixed before this lands: **(1)** `docs/QUERIES.md`
had no §14.6 even though the diff's own code comments already cited one — added, mirroring §14.5's
shape, copy-and-cited from the graph note (no new verification needed, the Cypher was already
exact-shape-verified). **(2)** AC-8's named test-strategy scenario (two documents fusing the same
entity, pipeline-level) wasn't literally exercised — the underlying mechanism was already proven at
the repository-primitive altitude (including a real-concurrency variant), but nothing drove
`IngestionPipeline.extract_chunk` twice to prove the pipeline wiring itself. Added
`test_ac8_two_documents_mentioning_the_same_entity_fuse_via_extract_chunk` (`test_ingestion.py`, the
file's one deliberate live-FalkorDB-backed exception).

**A genuine FalkorDB quirk found and filed (`kaizen_team`, two entries, second `REFINES` the
first):** an **undirected** relationship pattern carrying a relationship-property predicate — either
inline (`MATCH (a{...})-[r:TYPE{prop:val}]-(b{...})`) or a separate `WHERE r.prop = val` clause —
silently behaves as if directed on this build, missing the edge when it's actually stored in the
opposite direction from the pattern's node order. Found while writing the concurrency regression
test (a test-authoring bug, not a shipped-code bug — confirmed no shipped Cypher in this diff has
the trigger shape), reproduced deterministically (30/30 and 15/15 runs), and independently
re-verified by `analyst` during the Pass 4 gate and by `teco` during integration.

**Why:** Stage 4 turns Stage 3's deliberately-degenerate "every mention is a fresh node" population
into fused knowledge — conflicting facts survive (never merged edges, AC-1), auto-merge needs no
confirmation (AC-2), suggested matches are listable and confirm/reject-able (AC-3/AC-4), rejection
is reversible both automatically (re-derivation reopens a `rejected` edge) and on demand
(`recheck_match`, AC-7), and a batch of documents fuses against each other as well as existing
knowledge (AC-8) — all provable now.

**Tests:** `tests/test_fusion.py` (+7 new — `find_fuzzy_candidates`/`classify_fuzzy` unit tests);
`tests/test_repository.py` (+23 — all seven new methods, including the exact-tier's ordering
guarantee via black-box behavior across three sequential calls, and the concurrency regression test
via two REAL concurrent calls on separate connections with a `threading.Barrier`);
`tests/test_ingestion.py` (+8 — fusion wiring off `exactMatched`, self-match filtering, and the AC-8
pipeline-level test); `tests/test_background.py` (+2 — `_safe_fuse` isolation); `tests/test_services.py`
(+10), `tests/test_api.py` (+8), `tests/test_mcp.py` (+5) — the five match-lifecycle operations'
service/REST/MCP contracts. Suite: offline `pytest -q` 1650→1712 passed (+62 added, 1 removed net),
3 deselected unchanged; `./scripts/test_queries.sh` 320/320 unchanged (no new entries — same
precedent Stages 2-3 set: the new shapes are covered by the pytest integration suite instead); the
suite's teardown wipe of `reference` was re-seeded and verified in sync both after the initial
delivery and after the diff-gate fixes.

Mutation-tested: reverted `create_entity_with_auto_match` to the historical three-round-trip shape
— the concurrency regression test failed 15/15 as expected, reverted back; dropped the `type` filter
from the exact-tier candidate lookup — the matching-type test failed as expected; made
`classify_fuzzy` always return `'none'` — both the unit test and the ingestion-level fuzzy-write test
failed as expected. All three reapplied/confirmed passing.

**Scope boundary:** Stage 4 only — no chat-grounding change (Stage 5, `test_provenance.py`
untouched), no batch hardening or QA acceptance pass (Stage 6). `document-ingestion.md`/`-graph.md`/
`-ml.md`/`BACKLOG.md` untouched (locked design/tracking documents).

## 2026-08-24 — Milestone status reconciled across four documents (M4 number collision)

**What:** Fixing the one stale line flagged at the close of the `DESIGN.md` v1.0 pass surfaced a
wider disagreement: **four documents held four different accounts of where the project is.**
`docs/BACKLOG.md` is authoritative and was already correct; the other three are now reconciled to
it.

| Document | Said | Now |
|---|---|---|
| root `AGENTS.md` | "M0 complete" | M0–M4 delivered, M5 in progress, + pointer to BACKLOG as authoritative |
| `falkor-chat/README.md` roadmap | M3 `—`, M4 = "Scale & ops" `—` | M3 ✅, M4 ✅ (LLM provider/model config), M5 🟡, scale & ops → *Unscheduled* |
| `docs/DESIGN.md` §12 | M3 unmarked, M4 = "Scale & ops" | same reconciliation + a note that BACKLOG owns status |
| `docs/BACKLOG.md` | M0–M4 ✅, M5 🟡 — **correct** | unchanged |

**The substantive finding — `M4` named two different things.** `DESIGN.md` §12 and `README.md`
both still used **M4 = "Scale & ops"** (Redis Cluster, replicas, Sentinel, ACL/TLS), the original
roadmap's fifth entry. But `BACKLOG.md` reassigned **M4 = "LLM provider & model configuration"**,
delivered 2026-08-11 (K-042). So "M4" resolved to either shipped work or unstarted work depending
on which document a reader opened — the failure mode being an agent or a person reading
`DESIGN.md` §12, concluding M4 is unstarted infrastructure work, and planning against it.
Scale & ops has **no milestone number today**; it is now listed as *Unscheduled* in both places,
with `DESIGN.md` §12 saying explicitly that it was the old M4 before the number was reassigned, so
the next reader who finds an "M4 — Scale & ops" citation in a historical document can resolve it.

**Also:** `DESIGN.md` §12 gained a closing note that **milestone status is authoritative in
`BACKLOG.md`, not in a roadmap list** — the four-way drift is exactly what an un-owned status
marker produces. M3/M4's roadmap entries were also filled in with what actually shipped (both proof
flows for M3; the resolution seam, roles/fallback chains, workspace hard cap, publish-time
validation and resolved-model trace for M4) rather than left as the pre-delivery one-liners.

**Why this was worth doing beyond the one line asked for:** the requested fix was root `AGENTS.md`'s
"M0 complete". Correcting it required establishing what *is* complete, which is what exposed the
collision. Fixing the one line while `README.md` and `DESIGN.md` §12 still disagreed would have
left the repo inconsistent on the same fact, immediately after a session spent making `DESIGN.md`
accurate.

**Verified:** milestone facts read from `docs/BACKLOG.md`'s four `### — Milestone Mn —` rows and
confirmed against `HISTORY.md`'s dated close entries (M2 2026-07-08, M3 2026-07-21 "MILESTONE M3
✅", M4 2026-08-11 "closes M4", M5/K-050 Stage 3 of 6 in progress as of this date). No
`BACKLOG.md` edit was needed.

## 2026-08-24 — `DESIGN.md` v1.0: §1–§4 pass completes the restructure; two stale claims corrected

**What:** Final slice of the same session's documentation restructure (two entries below) — the
§1–§4 pass that the v0.5 entry had explicitly left open. Documentation-only, plus two factual
corrections.

- **§1.2 register de-ticketed.** Five delivered-ticket tags removed from the "Rationale /
  consequence" column (K-007 ×3, K-010). The one that stayed is now bolded as the live pointer it
  is: the identity-source-of-truth row steers the **open K-016** auth work.
- **Every `K-` number left in `DESIGN.md` now points at open work** — K-016/K-017/K-018 (M2.5,
  deferred), K-029 (the unenforced `decision`-step residual), K-033 (the `maxSteps` exact-cap
  proposal). That is the contract the v0.4 header note stated; as of this entry the document
  actually keeps it, end to end.
- **Correction — §1.3 was headed "M2 stack (decided 2026-07-04, pending implementation)."** The
  stack is not pending: M2 shipped and §12 of the same document records it as `✅ ... QA-accepted,
  M2 done`. Retitled **"Model stack (shipped)"** and the preamble rewritten to lead with what an
  operator needs — changing a model is a config-file edit + restart, never a code or env change —
  with `EMBEDDING_DIM` called out as the deliberate exception and why.
- **Correction — §2's supernode rule prescribed a pattern the design rejects.** Item 5 read *"we
  avoid it with a linked-list + **time-bucket** pattern"*, but §1.2 carries `**No DayBucket**
  *(rejected alternative)*` and §5.2 describes only the thread-scoped `NEXT` list. Two sections of
  one document disagreed about a load-bearing topology choice. Now states the thread-scoped
  linked-list pattern and names the time-bucket alternative as rejected, pointing at §1.2.
- **§2's live-verified blockquote** rewritten from provenance-first ("now pinned to…, originally
  probed on edge/main, re-verified 2026-07-09 via…") to finding-first: the four probed behaviours
  are the point, and the silent cross-graph-edge failure — no error, `MATCH` returns 0 rows — now
  reads as the trap it is rather than a clause in a build-history sentence.
- **§3's topology box realigned.** Pre-existing defect, unrelated to this pass: the borders were
  71 chars against 73-char content rows, so the box did not close. Borders widened to 73.

**Why:** the §1–§4 slice was judged lowest-yield when the restructure was scoped, and on
line-count it was. It was not lowest-yield on correctness: a register that a reader is told is
**authoritative** was carrying a stale milestone claim and a topology prescription contradicted
elsewhere in the same file. Reading closely enough to strip ticket noise is what surfaced both.

**Sizes:** `DESIGN.md` 748 → 752 lines. Slightly *up* — the two corrections and the reworked §1.3
preamble cost more than the removed ticket tags saved. Cumulative for the whole restructure:
1,298 → 752 (−42%), with `SERVER.md` (500) and `docs/test-reports/capacity-report.md` (118) split out.

**Verified:** `grep -o 'K-[0-9]\{3\}' docs/DESIGN.md | sort -u` returns exactly the five open
items. §3's box measures a single width (73) across all 21 rows. Every replacement ran through an
asserted single-hit list, so no edit applied silently or twice.

**Adjacent staleness, not fixed (needs a decision):** root `AGENTS.md` still describes
`falkor-chat/` as *"Design and query library are locked and live-verified; **M0 complete**"* — four
milestones behind, since M4/K-042 delivered 2026-08-11. Out of scope for a `DESIGN.md` pass and
left for the owner.

## 2026-08-24 — Doc restructure: `SERVER.md` split out, capacity → test-report, `DESIGN.md` v0.5

**What:** Second and larger documentation-only pass, continuing the same session's v0.4 header +
§5–§9 work (entry below). No technical content changed anywhere.

- **New `docs/SERVER.md` (500 lines).** `DESIGN.md` §14 (M1 application architecture) and §15 (MCP
  transport) moved out whole and renumbered §1/§2 — subsections map straight across (§14.7 → §1.7,
  §15.3 → §2.3). DESIGN.md was serving two readers: "how is the graph modelled and why" and "how is
  the server built and what bites you". It now stops at the graph.
- **`DESIGN.md` §14 is now the redirect table.** ~80 live citations across the repo point at
  `DESIGN.md` §14.x/§15.x, most of them inside **historical** records — `HISTORY.md`, `docs/reviews/*`,
  closed plans, and the agents' own `kaizen/history.md` files. Those were deliberately **not**
  rewritten: a dated record should say what was true when it was written. The 13-row redirect table
  resolves every one of them with a single lookup.
- **New `docs/test-reports/capacity-report.md` (119 lines).** `DESIGN.md` §11/§11.1/§11.2 —
  per-message RAM breakdown, the M1 append-path load test, the hot-read `GRAPH.PROFILE` closeout,
  and the RAM-budget/shard-packing tables — are measurements, i.e. test-report content, not design.
  DESIGN §11 keeps only the three numbers the design turns on (~12.4 KB/message at 1024 dims,
  ~614 msg/s write-serialised ceiling, vector dim fixed at workspace creation) plus the two rules
  that follow from them (`INFO memory` deltas not `GRAPH.MEMORY USAGE`; dim choice is the biggest
  lever). §11 still exists, so citations to it still land.
- **§10–§13 de-ticketed** to match §5–§9. §13's resolved guard-expression question collapsed from
  a 12-line two-layer amendment narrative to two lines pointing at §6.1; the three genuinely open
  questions kept. §12's roadmap prose de-ticketed, with the still-open M2.5 items (**K-016**,
  **K-017**, **K-018**) now bolded as the live pointers they are.
- **Live citations repaired** (`falkor-chat/AGENTS.md` ×2 + its Key-documents table, `README.md`
  ×3, `server/falkorchat/mcp.py:66`, root `AGENTS.md` component-docs row). AGENTS.md's Key-documents
  table gained rows for `SERVER.md` and `capacity-report.md`.

**Why:** DESIGN.md was 1,298 lines carrying four document genres — a decision register, the graph
design, empirical measurements, and a server runbook. Length was never the real problem; genre
mixing and accreted ticket annotation were. Splitting by reader and by kind is what actually
reduces load.

**Sizes:** `DESIGN.md` 1,298 → 752 lines (−42%). New: `SERVER.md` 500, `capacity-report.md` 119.
Total across the three is up slightly (headers, scope statements, the redirect table) — the win is
that each file now answers one question for one reader.

**Verified:** Zero dangling `§14.x`/`§15.x` references remain in `DESIGN.md` outside the redirect
table. Every replacement was applied through an asserted exact-match list (single-hit or reported),
so nothing was silently skipped; residual-marker sweeps for `K-0NN` / `Landing N` / finding IDs
return empty in both `SERVER.md` and `capacity-report.md`. `DESIGN.md`'s remaining `K-` references
are the three open items (**K-016/K-017/K-018** M2.5, **K-029**, **K-033**) plus §1–§4, which has
not had the pass — stated in the header's revision note rather than left implicit.

**Not done (deliberate):** **§1–§4 of `DESIGN.md`** (the decision register, engine constraints,
topology, cross-graph split) still carry delivered-ticket annotations — K-007/K-008/K-010/K-013/
K-034/K-042. §1.2's register is the doc's strongest section and its "Detailed in" column already
enforces no-re-explanation, so this is the lowest-yield remaining slice.

## 2026-08-24 — `DESIGN.md` v0.4: conforming header + §5–§9 restated in present tense

**What:** Documentation-only pass on `docs/DESIGN.md`, no technical content changed.
(1) **Header block now follows the root `AGENTS.md` convention** — a single bolded-label line
(`Status: active` / `Owner: architect` / `Tracks: —` / `Version: 0.4`) immediately under the H1,
replacing the stale free-form `Status: Draft v0.3` + `Date: 2026-06-06` +
`Owner: the repo maintainer` triple, which
had not been touched since before M2 while the body documented work through K-042. A new
"How to read this document" note states the doc's contract: it describes the system **as it is
now**, *when*-and-*which-ticket* lives in `HISTORY.md`, unbuilt work lives in `BACKLOG.md`, and a
`K-` number appears in DESIGN **only** when it points at work not yet done.
(2) **§5–§9 rewritten to that contract.** 34 delivered-ticket citations removed (K-007, K-013,
K-020, K-022, K-024, K-025, K-027, K-028, K-031, K-034); the two open ones kept and now meaningful
(**K-029** the unenforced `decision`-step residual, **K-033** the `maxSteps` exact-cap proposal).
Superseded-wording narration collapsed into plain present-tense statements — "supersedes the old
'expression' wording", "Two corrections to earlier wording", "the old claim was falsified",
"documented — not changed", "corrected for K-028", "(shipped)". §6.3's "Handoff note for K-025"
deleted outright as a verbatim duplicate of §6.1's `wait` bullet plus §6.2's timer paragraph.
§7.1's stale `test_queries.sh (256/256)` count dropped rather than re-stated (QUERIES.md is at
282/282 and is the place that tracks it).

**Why:** DESIGN.md had accreted one ticket citation every ~15 lines, so a reader reconstructed the
current design by mentally replaying a change log that `HISTORY.md` already holds in full. The
rationale prose — §5.2's traversal-cost argument, §6.2's ctx-CAS/silent-wrong-branch argument and
its R-1 residual, the `_drive_loop` SHA lock — is the doc's highest-value content and was kept
intact; only the *when* and the *by-which-ticket* were removed.

**Verified:** All section/subsection numbers unchanged (§5.1–5.4, §6.1–6.3, §7.1–7.3, §8, §9), so
every inbound citation still resolves — `AGENTS.md` cites DESIGN §5.3/§9/§6/§10/§14.7, and QUERIES
/plans/reviews cite the same numbers. A backticked-identifier diff over the rewritten range shows
zero symbols lost. §5–§9 went 470 → 450 lines (4,453 → 4,172 words).

**Not done (deliberate):** §1–§4 and §10–§15 have not had the pass. The three larger structural
moves proposed alongside this one — splitting §14–§15 (414 lines of server architecture + a QA
runbook) into their own document, relocating §11–§11.2's capacity measurements to a new
`docs/test-reports/capacity-report.md`, and moving §14.7's testing hazards to `test-plans/` or
`AGENTS.md` — are **not** done and remain open proposals; the §14–§15 split in particular needs a
naming decision against the `docs/<kind>/<slug>.md` grammar, since DESIGN/QUERIES/BACKLOG/HISTORY
are grandfathered top-level documents.

## 2026-08-24 — K-050 M5 Stage 3: document ingestion — extraction (FR-7a)

**What:** The third of 6 staged slices of the ingestion pipeline (`docs/plans/document-ingestion.md`,
"Stage 3 — Extraction (FR-7a)"). New `falkorchat.extraction` module: one LLM call per chunk
(entities + relationships combined, `data-scientist`'s recommendation adopted as-is,
`docs/plans/document-ingestion-ml.md` §3), parsed via the same fence-tolerant
`llm.extract_own_line_json_object` the K-027 guard judge uses, plus mandatory app-side schema
validation the parser's `require_key` alone does not provide (ML note F1) — a closed 7-value
`Entity.type` taxonomy (`Person, Organization, Location, Product, Event, Concept, Other`) with
out-of-enum/missing types coerced to `Other` rather than rejected; a deterministic stub-repair pass
that synthesizes a `{name, type: "Other"}` entity for any relationship `subject`/`object` name the
model forgot to also list, matched via one shared `extraction.normalize_name` (case-fold +
whitespace-collapse) helper reused, unchanged, by `repository.create_entity`'s `nameNormalized`
write — one normalizer, not two that can drift; and a 20-entities/20-relationships-per-chunk cap
(relationships truncated first off the raw list, then the raw entities list is truncated, THEN
stub-repair runs on top uncapped — fixed mid-build per the `analyst` diff-gate's Pass 3 MAJOR 1
finding, below; the original ordering capped entities *after* repair, which could slice a
just-added stub back off and silently drop the relationship fact it belonged to). New
`repository.create_entity`/`link_chunk_about_entity`/
`create_entity_relationship` (`QUERIES.md` §14.5) populate the `Entity` node and the previously-
dormant `ABOUT` edge, plus the new `RELATES_TO` fact edge (predicate stored as a free-text `label`
property, never its own relationship type — plan §3.3/§3.1); `RELATES_TO` is never deduplicated
(FR-6). New `falkorchat.ingestion.IngestionPipeline` (a peer of `AgentResponder`/`EmbeddingWorker`)
orchestrates one chunk's extraction into writes — **no fusion yet**: every extracted mention becomes
a fresh `Entity` node even across duplicate mentions, the deliberate Stage 3 degenerate case (plan
§3.1/§3.3). New `background._safe_extract` mirrors `_safe_embed_chunk`'s try/except-log-never-raise
isolation. `ingest_document` on both MCP and REST now schedules one `_safe_extract` per chunk
alongside (never instead of, never chained to) that chunk's own `_safe_embed_chunk` schedule — both
independent background jobs for the same chunk. `config/models.json` (and the test fixture mirror,
`tests/data/models.json`) gain a fifth `ModelGateway` kind, `extraction`, defaulted to the same local
model already used for `agent`/`step`/`guard` (`document-ingestion-graph.md` §4 — "zero graph cost,"
resolved through the existing per-kind config, no new resolution mechanism). `bootstrap_schema.sh`
gains `Entity.name`'s full-text index and a plain RANGE index on `Entity.nameNormalized` (no
uniqueness constraint — distinct real entities can share a normalized name+type before fusion runs)
— both bootstrapped-but-dormant until a future fusion stage queries them, the same posture the
original `Chunk` scaffolding demonstrated.

**Diff-gate fixes (Pass 3, `docs/reviews/document-ingestion-impl.md`):** `analyst`'s diff-scoped code
gate (verdict: approve with suggestions, no blockers) found two MAJOR issues, both fixed before this
lands. **MAJOR 1** — `extraction.extract`'s cap ordering (entities truncated *after* stub-repair)
could slice a just-synthesized stub entity back off when the raw entity list was already at the
20-item cap, silently dropping the relationship fact whose endpoint it was — the exact failure
stub-repair exists to prevent (ML note §3.2). Fixed by truncating the *raw* entities to the cap
**before** calling `_repair_stub_entities`, letting repair add on top uncapped (worst case
`MAX_ENTITIES_PER_CHUNK + 2×MAX_RELATIONSHIPS_PER_CHUNK`, still small/RAM-safe). New regression test
`test_extract_stub_repair_is_not_truncated_away_by_the_entity_cap`. **MAJOR 2** — Stage 3 doubles
Stage 2's already-flagged MCP per-chunk `threading.Thread` fan-out (now ~1,000-1,200 raw threads for
a max-size document, up from ~500-600), and `mcp.py`'s `_default_schedule` had no try/except around
`threading.Thread(...).start()` itself — a thread-creation failure (`RuntimeError: can't start new
thread`, a real ceiling near a process's `ulimit -u`/cgroup limit) would propagate unhandled out of
the `ingest_document` tool handler, a new caller-visible failure mode this diff introduced. Fixed
(the gate's cheapest suggested fix (a)+(c), per coordinator direction — the fuller fix, a bounded
thread pool, stays deferred to Stage 6 alongside Pass 2's original finding): wrapped
`_default_schedule`'s thread-start call in a try/except that logs and continues; updated its
docstring and `QUERIES.md` §14.5 to state the doubled fan-out explicitly, so the "accepted for M1
lab-scale" reasoning isn't silently reasoning about half the real number. Two new tests
(`test_default_schedule_swallows_a_thread_start_failure_and_logs`,
`test_default_schedule_still_runs_the_job_on_the_happy_path`). REST is unaffected (`BackgroundTasks
.add_task` is a cheap list append, not a thread spawn) — confirmed by the gate, not re-verified here.

**Why:** Stage 3 makes an ingested document's chunks yield real graph knowledge — `Entity` nodes and
`RELATES_TO` fact edges, each traceable back to its source chunk/document (AC-10) — without yet
attempting identity fusion, which is a genuinely separate axis (plan §3.1: entity identity fusion
vs. fact/relationship provenance) deferred to Stage 4. Building extraction first, against a
provably-duplicating "always create new" entity population, lets FR-7a/AC-10 be proven in isolation
before fusion's added complexity.

**Tests:** `tests/test_extraction.py` (+25 unit — bare/fenced/prose-wrapped JSON parsing, the
explicit empty-result shape, mandatory schema validation for both missing top-level keys and
non-list values, per-item malformed-entry skipping, enum coercion for out-of-enum/missing types,
stub-repair including case/whitespace-insensitive matching and the "never silently drop a
relationship fact" guarantee, the entities/relationships caps including the "a relationship dropped
by the cap gets no stub" ordering, and — added for the Pass 3 MAJOR 1 fix — the "entities at cap +
a relationship needing a not-yet-listed stub" regression); `tests/test_ingestion.py` (+11 unit —
per-chunk orchestration with a fake repository: entity+`ABOUT` writes, `RELATES_TO` writes between
same-extraction entities including stub-repaired ones, fresh id/timestamp minting per entity, no
fusion lookup ever attempted, a dangling relationship endpoint defensively skipped, empty/
unparseable results write nothing, FR-4 `llm=` injection sugar, and two tests pinning that the
shared `extraction.normalize_name` helper — not an independently-written one — drives both the
`nameNormalized` write and relationship-endpoint matching); `tests/test_background.py` (+2 —
`_safe_extract` calls the pipeline, and swallows+logs a failure without raising);
`tests/test_repository.py` (+7, live `ws:test` integration — `create_entity`'s field write and
`{entityId}` return shape, always-a-new-node even for identical `(nameNormalized, type)`,
`link_chunk_about_entity`'s edge write and non-deduplication, `create_entity_relationship`'s
provenance fields and non-deduplication); `tests/test_api.py` (+3 REST contract — every chunk
scheduled for extraction, extraction scheduled independently alongside embedding, and the
no-`ingestion_pipeline`-wired no-crash case); `tests/test_mcp.py` (+5 MCP contract — the Stage 3
scheduling shape (+3, same as `test_api.py`) plus, added for the Pass 3 MAJOR 2 fix,
`_default_schedule`'s thread-start-failure isolation and its happy-path regression guard (+2)).
Suite: offline `pytest -q` 1597→1650 passed (+53), 3 deselected unchanged; `./scripts/test_queries.sh`
320/320 unchanged (no new DDL entered that suite — same precedent Stage 2 set: the new Cypher shapes
are direct analogues of already-verified patterns, covered by the pytest integration suite instead);
the suite's teardown wipe of `reference` was re-seeded (`bootstrap_schema.sh acme` →
`seed_demo.sh acme` → `seed_workflows.sh acme` → `verify_workflows.sh acme`, confirmed in sync) both
before and after the diff-gate fixes.

Mutation-tested (Stage 3 build): removed the stub-repair step (3 tests caught it), removed
enum-coercion (2 tests), removed `_safe_extract`'s try/except (1 test), flipped
`create_entity_relationship`'s never-deduplicated posture to a guarded `MERGE` (1 test), replaced
`IngestionPipeline`'s use of the shared `extraction.normalize_name` helper with an
independently-written naive normalizer lacking whitespace-collapse (2 tests — added during that
pass specifically because the existing suite did not yet catch this drift), and removed the
extraction-scheduling loop from `api.py` (2 tests) — each confirmed to fail the relevant test(s)
before being reverted. Mutation-tested (diff-gate fixes): reverted the MAJOR 1 fix (restored the
old cap-after-repair ordering) — the new `test_extract_stub_repair_is_not_truncated_away_by_the_
entity_cap` failed as expected (`assert False` on the stub not being present), confirmed, then
reapplied; reverted the MAJOR 2 fix (removed `_default_schedule`'s try/except) —
`test_default_schedule_swallows_a_thread_start_failure_and_logs` failed with the unhandled
`RuntimeError` propagating, confirmed, then reapplied.

**Scope boundary:** Stage 3 only — no fusion (`SAME_AS` edges, exact/fuzzy matching, `confirm_match`/
`reject_match`/`recheck_match`, Stage 4), no chat-grounding change (Stage 5), no batch hardening
(Stage 6). `document-ingestion.md`/`-graph.md`/`-ml.md`/`BACKLOG.md` untouched (locked design/
tracking documents). The `Entity.name` full-text index and `Entity.nameNormalized` are wired into
NO query yet — inert until Stage 4's fusion lookups use them, the same "bootstrapped, not yet
queried" posture Stage 1 used for the original `Chunk` scaffolding.

## 2026-08-23 — K-050 M5 Stage 2: document ingestion — chunk embeddings + standalone search

**What:** The second of 6 staged slices of the ingestion pipeline (`docs/plans/document-ingestion.md`,
"Stage 2 — Chunk embeddings + standalone search (FR-3)"). `EmbeddingWorker` (`embedding.py`) gains
`embed_chunk`, a sibling of the existing `embed_message` — both now share a factored-out
`_resolve_and_embed` helper (the FR-19 pre-flight dimension guard + gateway resolution logic was
identical between the two, only the write label differed), so a `Chunk` embed consults only
`Chunk`'s vector index and a `Message` embed only `Message`'s, never the other's. New
`repository.set_chunk_embedding` (mirrors `set_embedding`, same pre-write dimension validation) and
`repository.search_chunks` (a `Chunk`-only ANN read, `QUERIES.md` §14.3 — no scope traversal, no
Entity expansion since `ABOUT` stays dormant until Stage 3). New `background._safe_embed_chunk`
mirrors `_safe_embed`'s try/except-log-never-raise isolation, so one chunk's embedding failure can
never corrupt the `Document` or block sibling chunks. `services.search_documents` embeds the query
text through the injected `ModelGateway` (mirroring `GraphragRetrieveTool`/`AgentResponder`'s own
text→vector step) then ranks via `search_chunks`; raises the new `SearchNotAvailableError` (maps to
REST 503, same precedent as `WorkflowEngineDisabledError`) when no gateway is wired. Wired as
`search_documents` on both MCP and REST (`GET /documents/search?q=`) — registered *before*
`GET /documents/{document_id}` in `api.py`, since Starlette's registration-order route matching would
otherwise let the dynamic path swallow a literal `search` segment as a document id (caught and fixed
during this build, not by a later bug report). `ingest_document` on both transports now schedules a
background chunk-embed for every chunk right after the write returns, via a new internal (non-public)
`repository.list_document_chunks`/`services.list_document_chunks` seam — kept separate from
`ingest_document`'s own return value so that receipt stays at its documented
`{documentId, chunkCount, status}` shape rather than echoing up to ~500,000 characters of chunk text
back into the response body. `bootstrap_schema.sh` needed no change — `Chunk.embedding`'s vector
index has existed since M2, dormant until this stage populates it.

**Why:** Stage 2 makes an ingested document's content retrievable as a standalone knowledge base
(FR-3), independent of chat search (FR-14) — an ingested document's chunks are readable via
`get_document` before their embeddings land (same eventually-consistent posture already used for
posted messages), and `search_documents` returns them ranked once embedding catches up. AC-6 (MCP
write, then read by any agent) is provable end-to-end at this stage, with no entity/fusion work
existing yet (Stage 3/4).

**Tests:** `tests/test_embedding.py` (+7 unit — `embed_chunk` happy path, wrong-dimension rejection,
gateway-resolved dim, the FR-19 mismatch/no-index guards, per-label index-cache isolation, and the
mirror-image "chunk gates only on Chunk, never Message" guarantee); `tests/test_background.py` (+2 —
`_safe_embed_chunk` calls the worker, and swallows+logs a failure without raising);
`tests/test_graphrag.py` (+6, live `ws:test` integration — `set_chunk_embedding` dimension rejection,
ANN retrievability, cosine-distance ranking, denormalized `documentId`/`seq`, `Chunk`↔`Message`
search-surface isolation, and `Chunk`'s own `read_index_dimension`); `tests/test_repository.py` (+2 —
`list_document_chunks` ordering + empty-document case); `tests/test_services.py` (+5 — chunk-list
passthrough, `search_documents` embeds-then-searches, default `limit`, and the no-gateway
`SearchNotAvailableError`); `tests/test_api.py` (+5 REST contract, including the route-ordering
regression guard); `tests/test_mcp.py` (+4 MCP contract, plus the tool-discovery set updated). Suite:
offline `pytest -q` 1566→1597 passed (+31), 3 deselected unchanged; `./scripts/test_queries.sh`
320/320 unchanged (no new DDL entered that suite — mirrors Stage 1's own precedent: the new Cypher
shapes are direct analogues of already-verified §6 patterns, covered by the pytest integration suite
instead). Mutation-tested: deliberately broke the chunk/message index-label mix-up in
`_resolve_and_embed`, removed `_safe_embed_chunk`'s try/except, flipped `search_chunks`'s
`ORDER BY` to DESC, dropped `set_chunk_embedding`'s dimension check, skipped embedding the query text
in `search_documents`, and removed the chunk-embed scheduling loop from both `api.py` and `mcp.py` —
each confirmed to fail the relevant test(s) before being reverted.

**Scope boundary:** Stage 2 only — no extraction (Stage 3), no fusion (Stage 4), no chat-grounding
change (Stage 5). `document-ingestion.md`/`-graph.md`/`-ml.md`/`BACKLOG.md` untouched (locked
design/tracking documents). Two routine implementation judgment calls, not scope changes: (1)
`search_chunks` omits the Entity co-occurrence expansion `hybrid_search` has, since `ABOUT` stays
unpopulated until Stage 3 — the OPTIONAL MATCH would only ever no-op today; (2) a new
`list_document_chunks` internal seam (repository + service) was added, not named in the plan's Stage
2 file list, to let `api.py`/`mcp.py` fetch a just-ingested document's chunks for background
embedding without bloating `ingest_document`'s own return payload.

## 2026-08-23 — K-050 M5 Stage 1: document ingestion — chunking + Document/Chunk write path

**What:** The first of 6 staged slices of the ingestion pipeline (`docs/plans/document-ingestion.md`).
A new pure `falkorchat.chunking.split_into_chunks(text, *, size=1000, overlap=150)` splits text on
paragraph breaks first, falling back to sentence boundaries, then a hard cut carrying `overlap`
characters forward regardless of boundary. New `repository.create_document`/`get_document`
(`QUERIES.md` §14) write a `Document` + its `Chunk`s + `HAS_CHUNK` edges in one guarded `GRAPH.QUERY`
(actor-resolved `INGESTED_BY`, `Document.sourceKind` derived server-side from which label resolved —
never trusted from the caller, same posture as `Message.role`); no `dup` guard — a retried call mints
a second `Document` (deliberate, mirrors the channel/thread-creation precedent). New
`services.ingest_document`/`get_document` mint the document/chunk ids and timestamps, enforce
`MAX_DOCUMENT_CHARS = 500_000` in the service itself (`DocumentTooLargeError`) so an MCP caller is
bound the same as REST, and set `Document.status = 'processing'` (a later stage's background
pipeline flips it to `'ready'` — not built here). Wired as `ingest_document`/`get_document` on both
MCP and REST (`POST /documents`, `GET /documents/{id}`), attributed to the `get_context()` actor
(FR-4) exactly like `send_message`. `bootstrap_schema.sh` needed no change — `Document`/`Chunk`
indexes+constraints and the `Chunk.embedding` vector index already existed since M2; confirmed
against the live script rather than trusted from the plan.

**Why:** Stage 1 lands the fast/deterministic half of ingestion (splitting + verbatim retention,
AC-9) with no LLM dependency, so it's provable in isolation before extraction/fusion/embedding
(Stages 2–4) add complexity. Design fully gated (`analyst` plan-gate, approve, Pass 2) before this
build — `coder` implemented against the locked plan + `graph-dba`'s live-verified Cypher
(`docs/plans/document-ingestion-graph.md` §2.1/§2.4), not designing.

**Tests:** `tests/test_chunking.py` (new, 10 pure unit tests covering the boundary-rule priority
order + overlap carry); `tests/test_repository.py` (+7, live `ws:test` integration — round trip,
actor-kind derivation, unknown-actor no-op, chunk ordering/denorm, non-idempotent retry);
`tests/test_services.py` (+9, `FakeRepo` unit — id/chunk minting, title fallback, size-limit
boundary, unknown-actor error); `tests/test_api.py` (+6 REST contract); `tests/test_mcp.py` (+3 MCP
contract, plus the tool-discovery set updated). Suite: offline `pytest -q` 1529→1563 passed (+34), 3
deselected unchanged; `./scripts/test_queries.sh` 320/320 unchanged (no new DDL/Cypher shape added to
that suite — the Document/Chunk Cypher was already live-verified in `-graph.md`, this stage only
wraps it in `repository.py`/documents it in `QUERIES.md`). Mutation-tested: deliberately broke chunk
minting (dropped a chunk), the actor guard (repository always reporting `ingestorFound=true`), the
`MAX_DOCUMENT_CHARS` boundary (off-by-one), the overlap carry (dropped), and the `sourceKind`
derivation (flipped) — each confirmed to fail the relevant test(s) before being reverted.

**Scope boundary:** Stage 1 only — no chunk embedding/`search_documents` (Stage 2), no extraction
(Stage 3), no fusion (Stage 4), no chat-grounding change (Stage 5). `document-ingestion.md`/
`-graph.md`/`-ml.md`/`BACKLOG.md` untouched (locked design/tracking documents).

## 2026-08-21 — K-028: workflow timers / scheduled wakeups — delivered, QA-accepted

**What:** A `wait`/`human` workflow step may now declare `config.waitForSeconds` (relative) or
`config.waitUntil` (absolute epoch-ms), alongside a required escalation transition guarded on
`ctx.timerFired == "<the step's own key>"`. A periodic sweep (`Services.sweep_due_workflow_runs`,
exposed as `POST /workflow-runs/due` and also ticked automatically in-process on
`FALKORCHAT_WORKFLOW_SWEEP_INTERVAL_S`, default 30s, gated on `WORKFLOW_ENABLED`) finds runs parked
past their due time and resumes them through the **existing** `resume_run_with_ctx` CAS, atomically
writing the step-scoped, reserved `ctx.timerFired` marker so the escalation guard fires
deterministically and exactly once. `TIMER_FIRED_CTX_KEY` is added to `RESERVED_CTX_KEYS`, so no
human/API caller can ever set or spoof it. A step declaring neither key keeps today's forever-park,
signal-only behaviour byte-identical (additive-only) — the existing `access-request@v1` proof flow
is unmodified and unaffected. New `repository.find_due_wait_candidates` (`QUERIES.md` §12.16) reuses
the existing `WorkflowRun.status` index — zero new indexes, zero new node/relationship types.
Publish-time validation rejects a timer-bearing step with no matching escalation transition, and any
caller-supplied `ctx.timerFired`. The sweep's per-candidate resume re-checks `atStepKey` (not just
`status`) to avoid a stale-scan race, and applies the same `MAX_CONFIG_LEN` bound
`submit_workflow_input`'s own ctx merge enforces.

**Why:** `wait` was implemented (K-024) as signal-driven only — parked and released solely by an
external signal on `POST /workflow-runs/{id}/input` — because the system had no scheduler. That
meant an SLA/escalation step ("if no approval in 48h, escalate") couldn't be expressed. K-028 adds
that release mechanism without introducing a second source of truth for run state: the sweep never
writes a new `WorkflowRun` property and derives dueness fresh, every call, from data the engine
already writes (`StepRun.startedAt` + `Step.config`).

**How it got here (worth recording — the design changed mid-flight for a real reason):** the first
gated design (v1/v2, `docs/plans/workflow-timers.md`) closed the churn risk of an automated
sweep-triggered resume by requiring a timer-bearing step to declare an *unconditional* fallback
transition. Two `analyst` plan-gate passes approved this. During implementation, writing the
CAS-contention test surfaced that the fix made the feature **unreachable**: `executor._drive_loop`
evaluates a step's transitions on every pass, including its very first arrival — an unconditional
transition fires immediately, before the step can ever park. `teco` independently re-verified this
against the live source before routing it back. v3 replaces the unconditional arm with the
step-scoped `ctx.timerFired` marker-guard mechanism described above — a genuinely conditional
escalation guard, false until the sweep itself resumes the run. `analyst`'s Pass 3 traced the new
mechanism end to end against live source (not the revision note) and confirmed it works; the
post-implementation diff re-gate independently re-verified the shipped code the same way (own suite
runs, own `GRAPH.PROFILE`, own mutation test via a scratch-copy `services.py` load, a recomputed
SHA-lock hash on `_drive_loop` confirming it, once again, was never touched).

**Verified:** `analyst` — plan gate ×3 passes (`docs/reviews/workflow-timers.md`; Pass 1 needs
changes, Pass 2 approve with suggestions, Pass 3 approve with suggestions) + a post-implementation
diff re-gate (approve with suggestions, independently re-run rather than trusted). `qa-engineer` —
acceptance pass (`docs/test-reports/workflow-timers-report.md`): **PASS, zero defects**, all 12
planned test items, driving the real running process end to end including the automatic periodic
sweep actually ticking (not a stub) and the concurrent human-vs-timer race resolving to exactly one
winner in both orderings. `teco` independently re-ran the full baseline before and after QA.

**Suites:** offline `pytest -q` **1456 → 1529 passed, 3 deselected**; query suite (`test_queries.sh`)
**320/320** (this component's own last-recorded baseline of 256/256 was already stale before this
delivery, unrelated to K-028).

**Docs:** `docs/plans/workflow-timers.md` (v3), `docs/reviews/workflow-timers.md` (3 plan-gate
passes + diff re-gate), `docs/test-plans/workflow-timers.md`, `docs/test-reports/workflow-timers-report.md`,
`docs/plans/workflow-timers-coordination.md` (the full coordination record). `DESIGN.md` §6.1/§6.2/§6.3
and `scripts/start_server.sh` updated in the same change.

**Follow-up filed, not gating this item:** **K-049** — while building the ctx-merge length-bound
test, an oversized value written to an *indexed* graph property (`Step.key`) crashed the shared dev
FalkorDB instance outright (reproduced twice). Unrelated to K-028's own correctness (the shipped
feature never writes an unbounded value to an indexed property; the fix landed on the opaque, ctx
side instead) but a real shared-instance reliability risk, routed to `graph-dba` to root-cause.

## 2026-08-21 — Hygiene: home-path leak genericized in `docs/test-reports/graphrag-eval-report.md`

**What:** Five occurrences of the maintainer's absolute `/home/<user>/prg/graphmind-ai-lab/…`
prefix, left over from literal script output and self-referential deliverable citations, replaced
with repo-root-relative paths (root `AGENTS.md`'s citation convention — a backticked path from the
repo root, no leading slash). No content/verdict change, TP-006/TP-009's own findings untouched.

**Why:** Flagged by `claude/scripts/audit-team.sh` check 7 (personal-identifier scan, run during an
unrelated `cobb` session in `claude/`) as the repo's only FAIL; fixed on request. `git blame` on the
lines showed the file untouched since its original authoring — a plain pre-existing leak, not
introduced by anything recent.

**Verified:** `bash claude/scripts/audit-team.sh` now reports `RESULT: PASS`.

## 2026-08-21 — K-027 item 4: golden-set expansion — delivered; epic closed

**What:** Grew `server/tests/eval/golden_guards.jsonl` from 26 → **85 rows** per the finalized
`data-scientist` method note (`docs/plans/golden-set-expansion-ml.md` v3), gate-approved
unconditionally by `analyst` (`docs/reviews/golden-set-expansion.md` Pass 2, 2026-08-21). This was
the last open scope item in K-027 — landing it closes the epic.

**Composition** (§3.4's target, re-derived from the real file, not just restated): `clear_advance`
30 (18 `understanding` / 12 `turns`), `clear_suspend` 40 (24/16), `boundary` 15 (9/6) — a
Wilson-interval-derived zero-tolerance screen (n=40 `clear_suspend` bounds true false-advance rate
≤8.8% at 95% confidence on zero observed failures), not the backlog's original "~30"/"~50-60"
heuristic. Appended the 59 new rows drafted in plan §6 verbatim (extracted programmatically from
the plan's own fenced JSON blocks, not retyped, to rule out transcription drift) — the existing 26
rows are untouched (`git diff --stat`: 59 insertions, 0 deletions). All 85 ids unique, schema
identical across old and new rows.

**Descope, recorded plainly (2026-08-20, by the user, carried from the plan's own §5/§10):** no
second labeler was available, so the `boundary` tier's independent-second-labeler requirement —
the backlog item's original ask — was dropped. All three tiers are now sourced identically:
LLM-drafted candidate rows, single human spot-check before merge. A `boundary` row's label
therefore carries **no more validation than a `clear_advance`/`clear_suspend` row's does** —
downstream readers of the calibration report should treat every `boundary` label as a spot-checked
draft, never as independently validated ground truth. This is a real, accepted loss of validation
strength for the one tier whose labels are policy calls rather than extracted facts, not a footnote
to omit.

**Harness edits** (`server/tests/eval/test_guard_calibration_live.py`, F3 — the only file with
fixture-size-specific literals): five asserts updated `26→85` (rows), `26→85` (cases),
`26*K_REPLICATES→85*K_REPLICATES` (replicate count), `21→70` (clear-tier cases, 30+40), `5→15`
(boundary-tier cases). No other harness file needed a size-specific edit — `guard_calibration.py`'s
metric functions already group dynamically by `tier`/`path`.

**New offline integrity test:** `server/tests/eval/test_guard_set_integrity.py` (closes F4),
mirroring `test_golden_set_integrity.py`'s pattern for the retrieval set — unique ids, required
schema fields, `tier`/`path` enum validity, `expected` is bool, every `boundary` row is
`expected: false`, and per-stratum/path counts as **minimum** inequalities (not exact literals) so
the check survives the next expansion unedited. Written test-first: run against the pre-expansion
26-row fixture first (RED — failed on the stratum-floor check, 111 other assertions already green),
then against the expanded 85-row fixture (GREEN — 358 passed). Mutation-tested twice: (1) dropped a
required field from one drafted row and flipped one `boundary` row's `expected` to `true` — both
defects caught, then reverted; (2) reverted the fixture-row-count literal edit in
`test_guard_calibration_live.py` back to `26` — the live test failed immediately with the real
85-row fixture, confirming the assert actually exercises the file — then restored.

**Live-verified end-to-end, not just collection-checked** — LM Studio (serving
`qwen/qwen3-4b-2507`) and FalkorDB were both reachable this session, so `pytest -m live` ran the
real 85-case × k=3 = 255-call calibration to completion: **G1 false-advance = 10.0% (n=40 cases /
120 calls — lands right at the ≤10% gate) · G2 advance-recall = 86.7% (n=30 cases) · VERDICT:
wire.** Full report: `docs/test-reports/guard-judge-calibration-2026-08-21.md`. This is a
substantially stronger, but still not certifying, result than the n=10/11 screen it replaces — see
the plan's own §8 item 2 for why a single unlucky replicate among 120 would still fail the gate
outright even against a well-behaved judge.

**Offline suite:** `test_guard_calibration.py` + `test_golden_set_integrity.py` +
`test_guard_set_integrity.py` — 499 passed (baseline was 141 before this unit's own new test file).
Full repo default suite (`pytest -q`, offline): 1456 passed, 3 deselected, no failures. Running the
full offline suite wiped the shared `reference` graph as documented (`AGENTS.md`); re-seeded via
`./scripts/seed_workflows.sh acme`, then confirmed back in sync with `./scripts/verify_workflows.sh`.

**Docs:** `docs/plans/golden-set-expansion-ml.md` (v3, `data-scientist`, unconditional per §7);
`docs/reviews/golden-set-expansion.md` (`analyst`, Pass 2, approve). `docs/BACKLOG.md` K-027 item 4
marked ✅ delivered; **the K-027 header itself flips to ✅ delivered — this was the epic's last open
scope item.**

## 2026-08-20 — K-027 item 5: Ministral re-probe — delivered (block on both axes); K-048 filed

**What:** Re-probed Ministral-3B against current code (post item 1's parse fix, post item 2's
engine contract) per K-027 item 5 / D13 finding 2, reusing item 3's reviewed judge-calibration
harness/protocol rather than a new one. Model identity confirmed live first — LM Studio's two
catalog ids, `mistralai_ministral-3-3b-instruct-2512` and `mistralai/ministral-3-3b`, are the same
underlying weights aliased onto one loaded slot (byte-identical temperature-0 completions,
`/api/v0/models` state flips between the two depending on which was called last).

**Judge calibration: block.** New throwaway driver `server/tests/eval/probe_ministral_judge.py`
(not pytest-collected) drives the real, unmodified `guard_calibration.py` plumbing against
`lmstudio/mistralai/ministral-3-3b` via the workspace-override hard cap
(`run["modelOverrides"]["guardModel"]`) — never a `config/models.json` edit. Same fixture (26 cases
× k=3 = 78 real calls, `golden_guards.jsonl`, sha256 unchanged from the 2026-08-17 Qwen run), same
gates as item 3. **G1 false-advance = 0.0%** (0/30, pass) · **G2 advance-recall = 45.5%** (5/11,
fails the ≥80% gate) · κ=0.442 diagnostic. Improves on D13's fence-tolerant 0.364 (the item-1 parse
fix measurably helped) but stays far under Qwen's 0.818 — a reasoning-quality gap, not a parse
artifact.

**Terminal tool call: Ministral remains the better native caller, but it's moot.** An isolated
`post_message`-schema replay (D13's own protocol, n=5, alternation-safe message shape) scored
Ministral 5/5 native tool calls (reconfirms D13's 3/3); a same-session Qwen replay on the identical
current prompt/schema scored 0/5. Moot in practice: the already-shipped K-039 implicit-dispatch
fallback (2026-07-31) already compensates for Qwen's native weakness at the engine level.

**New finding: a message-alternation crash, filed as K-048.** Before attempting the full live e2e,
live-verified that `executor._assemble_messages`'s unconditionally-appended trailing `user`-role
`CONTEXT` block produces two consecutive `user`-role messages on `intake`'s very first call —
Ministral's LM-Studio-served chat template hard-rejects this (`HTTP 400`, Jinja alternation error),
while Qwen's template tolerates it. `executor._drive`'s fault net turns the crash into an unhandled
`fail_run`. This is model-agnostic message-assembly debt, not specific to the Ministral decision,
so it is filed as its own backlog item (`docs/BACKLOG.md` **K-048**, 🔵 proposed, owner
`architect`→`tdd-engineer`) rather than folded into the Ministral verdict; confirmed **not** inside
the SHA-locked `_drive_loop` (re-ran the lock's own recipe — hash unchanged, `71055f756280`) before
stating that as fact in the filing.

**Verdict: do not wire Ministral for either kind under the current codebase.** Does not change
D13's practical relevance — Qwen remains the right call, but for a different reason than D13
measured: K-039 already neutralizes Qwen's tool-calling weakness, and Ministral's own chat template
introduces a new, more severe failure mode (a hard crash) that Qwen doesn't have.

**Docs:** `docs/plans/ministral-reprobe-ml.md` (method note, `data-scientist`); reviewed
**approve**, 0 blocker/major, `docs/reviews/ministral-reprobe.md` (`analyst`). `docs/BACKLOG.md`
K-027 item 5 marked ✅ delivered; the K-027 heading note updated to name item 4 as the only item
still open, itself blocked on a user decision (`docs/plans/golden-set-expansion-ml.md` §10, not yet
resolved); new **K-048** filed. No `server/` code touched by this entry — the probe driver
(`server/tests/eval/probe_ministral_judge.py`) was added by the re-probe itself, tracked separately
in that unit's own artifacts, not part of this doc-only follow-up.

## 2026-08-20 — K-027: six carried gate findings (m-1/m-2/m-3/n-1/n-2/doc-drift) — delivered

**What:** Closed out the six still-open "Carried findings from the analyst gate"
(`docs/archive/reviews/m3-guard-thread-context-impl.md`) recorded under `docs/BACKLOG.md`'s
`### K-027`, as unit U1 of the `guard-reliability-followups-coordination.md` run (`teco`-coordinated,
alongside U2/U3 golden-set and Ministral advisory notes, out of scope here). Strictly test-first
per finding; `guards.py`'s `_is_negated`/`_rationale_contradicts`/`_coerce_verdict` bias-to-suspend
policy re-derived from the code before changing anything (per the brief's instruction not to trust
the carried BACKLOG text's framing blindly).

**m-1 (false-advance bug, the important one).** `guards._is_negated`'s 12-char negator window let a
negator from an *earlier* clause (e.g. the "not" in `"The user did not say; more info is needed."`)
negate a deficiency cue in a *later* clause across a `;`/`.`/`,` boundary — the false-advance
direction the code's own comment claimed couldn't happen. Fix: the window is now truncated at the
last clause-boundary punctuation before the cue; the comment is corrected to name the true failure
direction. The three rationales from the gate's probe table are pinned into
`SUSPENDING_RATIONALES` (`tests/test_guards.py`). Mutation-tested (revert/red/reapply/green).

**m-2 (evidence-window bug).** `guards._recent_turns` sliced `thread[-n:]` *before* filtering
malformed/empty rows, shrinking the usable evidence window exactly when the judge is on its
degraded fallback tier. Fix: filter first, slice second. Mutation-tested.

**m-3 (tier now traced).** `GuardVerdict` gained an additive `tier: str | None = None` field
(`"understanding"` / `"recent_turns"`, `None` for non-`llm` guards) set in `evaluate_guard`'s `llm`
branch; `executor._trace_step`'s `guard_judgment` payload folds it in as an optional `[{tier}]`
segment when present, unchanged otherwise. `_select_transition`/`_trace_step` are outside the
`_drive_loop` SHA lock — confirmed before touching, and the lock (`71055f756280`) is unchanged
before and after this whole unit. Mutation-tested (both halves reverted together).

**n-1 (already fixed, no code change).** The function-local `import json as _json` this finding
named was already removed by an unrelated earlier commit (`1dd48a0`) that hoisted a top-level
`import json` while touching the same `app.py` functions; verified by grep before closing.

**n-2 (O(n²) cap loop → O(n)).** `app._render_judge_user`'s eviction loop re-joined the whole
candidate message on every dropped turn. Rewritten to accumulate lengths once and evict oldest-first
via O(1) arithmetic per step, verified byte-identical to the old algorithm's output across 2000
randomized cases before landing (a behavior-preserving refactor under green, not a bug fix — no
mutation-test applies). New scale test (`tests/test_app.py`, 300 turns) pins the arithmetic well
beyond the shipped `RECENT_TURNS_N=6` window.

**Doc-drift.** Re-measured the `_drive_loop` byte-identity extraction (`DESIGN.md` §6.2's `awk`
command): **2860 bytes**, confirming `BACKLOG.md`'s existing figure and refuting both `2844` and
`2839`. Repo-wide grep found the wrong figures only inside `docs/archive/` (read-only history,
per `AGENTS.md` — never re-edited); every currently-active doc site already quotes the SHA alone
with no byte count, so nothing needed correcting there.

**Tests added:** `tests/test_guards.py` — 3 new `SUSPENDING_RATIONALES` cases (m-1), 1 malformed-tail
case (m-2), 4 tier cases (m-3). `tests/test_executor.py` — 1 trace-payload tier case (m-3).
`tests/test_app.py` — 1 scale case (n-2). Net +10 tests.

**Suite:** offline `.venv/bin/python -m pytest -q` from `server/` — **1088 passed, 3 deselected**
before this unit's changes; **1098 passed, 3 deselected** after (matches the entry baseline recorded
in `guard-reliability-followups-coordination.md`). `_drive_loop` SHA-lock reconfirmed unchanged
before and after: `71055f756280` both times, byte count **2860** both times (the earlier-quoted
2844/2839 figures were never accurate — see Doc-drift above). `reference` graph re-seeded after the
final run (`./scripts/seed_workflows.sh acme`), verified in sync (`./scripts/verify_workflows.sh
acme` → exit 0, "2 defs in sync").

**Files touched:** `server/falkorchat/guards.py`, `server/falkorchat/executor.py`,
`server/falkorchat/app.py`, `server/tests/test_guards.py`, `server/tests/test_executor.py`,
`server/tests/test_app.py`, `docs/BACKLOG.md` (this run's six K-027 items marked delivered).

## 2026-08-17 — K-027 item 3: judge calibration (D9/D10) — delivered

**What:** Ran the guard-judge calibration protocol (`docs/archive/plans/m3-guard-calibration.md`
§4) against the shipped local `lmstudio/qwen/qwen3-4b-2507`, closing K-027 item 3 ("Judge
calibration (D9/D10)", `docs/BACKLOG.md`, search `### K-027`) — the last open question behind
wiring the `intake → research` fuzzy guard's LLM-as-judge live. Current-code method note:
`docs/plans/guard-judge-calibration-ml.md` (2026-08-17, confirms the archived protocol's method
still sound, one `run`-construction addendum).

**Result:** G1 false-advance = 0.0% (0/30 calls, n=10 `clear_suspend` cases) · G2 advance-recall =
81.8% (9/11 `clear_advance` cases) — both gates pass · **VERDICT: wire**. κ = 0.811, reported
strictly as a diagnostic, not a gate. Per archived §6/§6.1, this is a **one-sided screen at n=21
hand-labeled cases, not a certification** — a pass means "no blocker found at a sample size that
could only have found a large one." Full report:
`docs/test-reports/guard-judge-calibration-2026-08-17.md` (provenance header, verdict line, κ with
marginals/prevalence, per-path breakdown, coercion-flip rate, flip rate, materiality-probe check,
full per-case table with raw/final decisions and rationales).

**Mechanism:** New harness under `server/tests/eval/`: `guard_calibration.py` (fixture-row → real
`guards.evaluate_guard(...)` call assembly per archived §5.1's table, with the ml-note's one
addendum — `run = {"ws": "ws:golden-eval", "modelOverrides": {}}` rather than the archived doc's
literal `run = {}`, which post-K-042 lands in a branch production never reaches; a `RecordingJudge`
wrapper that records the evidence tier actually handed to the judge and asserts it matches the
fixture's declared `path` on every replicate — archived §5.1's "not optional" assertion; the
G1/G2/κ/coercion-flip/flip-rate/materiality-probe/per-path metric functions), plus
`test_guard_calibration.py` (24 offline unit tests, network-free, driven by stub judges) and
`test_guard_calibration_live.py` (the live k=3×26-case run — 78 real judge calls — behind the
established `pytest.mark.live` marker; builds a real `ModelGateway` directly from
`ProviderCatalog.load(...)`/`Overlay.load(...)` rather than `ModelGateway.from_env()`, since
`server/tests/conftest.py`'s autouse `_model_config_env` fixture silently redirects `.from_env()`
to the offline dim-4 test config for every pytest test — matches the already-established
precedent `test_golden_set_integrity.py` documents as "D7 mechanism 1"). Also pins the guard/judge
role's sampling: `config/models.json`'s `models` map gained a new entry,
`"lmstudio/qwen/qwen3-4b-2507": {"temperature": 0}` — verified end-to-end (not assumed) that the
setting flows through `modelconfig._resolve_element` into the resolved request params.

**Tests:** 24 new offline tests in `server/tests/eval/test_guard_calibration.py`, mutation-tested
twice: G1's per-call counting rule swapped to per-case-majority (2 tests went red on the swap,
reverted, green again) and the path-was-taken assertion turned into a no-op (3 tests went red,
reverted, green again). Full offline suite: **1088 passed, 3 deselected, 0 failures**
(`cd server && .venv/bin/python -m pytest -q`, re-verified 2026-08-17) — up from 1064 passed / 2
deselected before this item (the 24 new offline tests plus the new live test as the 3rd deselected
entry).

**Review gates:** both **approve with suggestions, no blocker** —
`docs/reviews/guard-judge-calibration.md` (`analyst`, harness/implementation correctness:
hand-recomputed every number in the report from its own per-case table, independently re-ran both
mutation-test claims) and `docs/reviews/guard-judge-calibration-ml.md` (`data-scientist`,
methodology: independently re-derived the same numbers a second way, confirmed "wire" is the
correct verdict under the protocol's own decision table). Both reviews converged on the same
non-blocking finding, since folded into the report's own "Materiality-probe check" section: the
judge's two G2 misses (`ca-04`, `ca-08`) are themselves the fixture's designed materiality probes,
and their rationales echo the `missing` field almost verbatim rather than reasoning about
research-sufficiency — a real qualitative pattern, but not blocker-grade, since the protocol's
bloc-AND rule (`ca-05` correctly advanced, `cs-04` correctly suspended) never triggers.

**Supersedes a stale number:** `docs/BACKLOG.md` K-027 item 3 previously quoted a "recall 0.818,
false-advance 0.067" diagnostic as "already on record" — that number predates the K-027 item 1
parse fix, bypassed `guards.evaluate_guard` entirely, and used the wrong G1 denominator (15 cases,
not the gate's 10); corrected in place in `docs/BACKLOG.md` (struck through, marked superseded,
full derivation cited to `docs/plans/guard-judge-calibration-ml.md` §3).

## 2026-08-16 — K-027 item 2: terminal-node-must-post engine contract — delivered

**What:** An engine-level guarantee that a must-communicate `agent`-typed step either dispatches
its required tool before ending its turn, or the run records a visible, diagnosable reason it
didn't — closing K-027 item 2 ("Terminal-node-must-post engine contract") together with its own
"Addendum from the K-025 QA pass" (`docs/BACKLOG.md`, search `### K-027`), which broadened the
scope from terminal-node-only to any node whose contract is "post" (the non-terminal `intake` node
hit the identical failure, in the worse "clarifying question never reached the thread" shape).
Plan: `docs/plans/must-post-engine-contract.md`.

**Mechanism:** A new opaque `config.requiredTools: [<tool name>, ...]` declaration on `agent`-typed
steps — a subset of the node's own `config.tools` that must be successfully dispatched at least
once before the node may end its turn; absent/empty means no obligation, byte-identical to prior
behaviour. Enforced inside `server/falkorchat/executor.py`'s `_run_agent_node`, at both of its own
existing exit points: the non-tool-call-text branch (checked *after* the already-shipped K-039
implicit-dispatch fallback has had its chance) and the `maxIterations`-exhaustion fall-through,
which K-039 never covered at all. On a missing required tool: an unconditional `_log.warning`
naming the run id, step key, and missing tool(s) (mirroring the existing `_link_emissions`
"PRODUCED link gap" warning shape), plus — on a debug/traced run only — a new
`("must_post_violation", "...")` trace entry appended to the existing `trace` list (zero changes to
`_trace_step`, `StepResult`'s shape, or any repository call). The run is never failed or parked —
trace-and-continue was chosen over fail/park/retry alternatives (plan §3.3), each considered and
rejected. `post_message`'s satisfaction is read off the existing `emissions` list (the richer "a
`Message` actually got created" signal — accurate even when a dispatch reaches
`PostMessageTool.run` and returns a decline string without raising, e.g. no thread bound); any
other required tool uses a new `satisfied: set[str]`, threaded through `_handle_tool_call`'s two
call sites (one new parameter, no public-surface change). A fourth, deliberately-LAST invariant in
`server/falkorchat/services.py`'s `_validate_def_spec` — inserted between the existing
`waitsForHuman` loop and the `cmp`-family `validate_cmp` loop, matching the plan's specified
ordering — rejects at publish (nothing written) a `config.requiredTools` that isn't a list of
strings, is declared on a non-`agent` step, or names a tool absent from that step's own
`config.tools`. `scripts/seed_workflows.sh`'s `triage@v1` `intake`/`answer` steps now declare
`"requiredTools": ["post_message"]`, with an inline comment citing the K-034 re-publish caveat
below.

**Relationship to K-039:** layers on top of, and does not modify, the already-shipped K-039
implicit-dispatch fallback (`executor.py:677-713`, confirmed byte-for-byte unchanged) — K-039
remains the fast, narrow, best-effort *recovery* attempt for the one tool shape it can safely
synthesize; this item is the general-purpose *detector* that runs regardless of whether K-039
fired, applies to any declared tool, and never leaves a violation with zero signal.

**SHA-lock:** the `_drive_loop` byte-identity lock (`71055f756280`) is unchanged — every change in
this item lands in `_run_agent_node`/`_handle_tool_call` (both already outside the lock) plus
`services._validate_def_spec` (a different module entirely, publish-time only). Independently
reconfirmed unchanged four separate times across this item's units (both implementer units, `teco`,
and the `analyst` diff-scoped re-gate) using the documented line-number-independent recipe
(`docs/DESIGN.md` §6.2) — no re-lock ceremony needed.

**Tests:** 13 new offline tests — 9 in `server/tests/test_executor_agent.py` (8 from the plan's
§10 items 1-8, plus one supplementary test the implementer added on its own initiative after
mutation-testing found a real gap in the plan's own test list —
`test_compliant_node_dispatching_a_non_post_message_required_tool_leaves_no_violation_trace`), 4 in
`server/tests/test_services.py` (plan §10 items 9-12). Full offline suite: **1064 passed, 2
deselected, 0 failures** (`cd server && .venv/bin/python -m pytest -q`).

**Review gates:** design plan-gate — `docs/reviews/must-post-engine-contract.md`, verdict *approve
with suggestions* (all four findings folded into the plan's Version 2 in place: a §11 note that a
non-debug run's only signal is the process log, two added tests, two clarifying notes). Diff-scoped
re-gate — `docs/reviews/must-post-engine-contract-impl.md`, verdict *approve*, no blockers, no
majors.

**Known, deliberately deferred limitation — `ws:acme` snapshot drift:** the shared dev box's
`ws:acme` workspace's `triage@v1` **snapshot** is now out of sync with the freshly-republished
`reference` def — `reference` carries the new `requiredTools` config, `ws:acme`'s pre-existing
snapshot does not, because a config-only re-publish onto an already-published `(key, version)` is a
documented, deliberate no-op (K-034/K-031: `MERGE … ON CREATE SET` only writes on first creation).
`./scripts/verify_workflows.sh acme` reports this divergence. **Nothing is behaviourally broken** —
the workspace snapshot is what actually executes, so `ws:acme`'s live triage runs are unaffected and
continue exactly as before this item landed. This is the exact rollout question the plan's §8/§11
flagged as an open, stakeholder-level decision (wipe-and-reseed `reference` vs. a `triage@v1`
version bump); the stakeholder's initial decision, recorded in
`docs/plans/must-post-engine-contract-coordination.md` (row U9), was **"leave as-is for now,
tracked follow-up, not blocking."** Not resolved by this item — this item only documents the
divergence, it does not touch it; see the coordination doc's ledger for the current status of any
separate follow-up unit tracking reconciliation. Not a defect either way: `ws:acme`'s live triage
behaviour is unaffected regardless of when/whether the snapshot is reconciled.

**Update, same day (2026-08-16):** the divergence above was resolved, via the drop-and-re-materialize
path rather than a `triage@v1` version bump — `graph-dba` dropped `ws:acme`'s stale `triage@v1`
snapshot and re-materialized it fresh from `reference` (which already carried the new
`requiredTools` config), with the stakeholder explicitly authorizing skipping the live/parked-run
check since `ws:acme` is a dev database. A follow-on gap found in the process — a concurrent
`pytest -q` run had wiped `reference` completely, including `access-request@v1` — was closed with
the standard, documented, idempotent re-seed remedy. Final state, independently verified:
`./scripts/verify_workflows.sh acme` now reports **both** `triage@v1` and `access-request@v1` in
sync (`RESULT: OK`).

## 2026-08-16 — K-046: root `server/tests/conftest.py`'s `_falkordb_reachable()` write-mode probe bug — fixed

**What:** `server/tests/conftest.py`'s `_falkordb_reachable()` used write-mode `GRAPH.QUERY`
(`.query("RETURN 1")`) as a mere reachability probe against `ws:test` — per
`claude/graph-dba/falkordb-quirks.md`, a `GRAPH.QUERY` read against a nonexistent graph key
silently materializes an empty graph key as a side effect, which a reachability probe must never
have. Identical bug shape to the one already fixed in `server/tests/eval/conftest.py` (found as
Blocker B-1 at the K-026 Unit 2b `analyst` code gate) — closed the same way, filed as K-046 out of
the K-026 closeout. In practice this rarely bit here because the session-scoped `_schema` fixture
in the same file always rebuilds `ws:test` before the probe runs, but the latent bug was real.

**Fix:** Switched to `.ro_query("RETURN 1")` (`GRAPH.RO_QUERY`, never materializes) with the same
"empty key" `ResponseError` tolerance pattern as the eval-side twin (a `ResponseError` containing
"empty key" still counts as *reachable* — the server responded, there's just no such graph yet).
Also parameterized the function as `_falkordb_reachable(ws: str = TEST_WS)` (mirroring the eval
version's `ws: str = EVAL_WS`) so it's testable against a throwaway ghost workspace without ever
touching the real `ws:test` graph key.

**Test:** New `server/tests/test_conftest_probe.py`, mirroring
`tests/eval/test_conftest_probe.py`'s proof for the eval twin: picks a ghost workspace name
guaranteed not to already exist (asserts the precondition), calls `_falkordb_reachable(ghost_ws)`
and asserts it returns `True` (server responded), then asserts the ghost graph key was **not**
materialized in `conn.list_graphs()` as a side effect, with defensive `finally` cleanup in case the
bug is back. Imported via `from conftest import _falkordb_reachable` (root `tests/` has no
`__init__.py`, so pytest's default `--import-mode=prepend` loads `tests/conftest.py` as a
top-level module literally named `conftest` — the same pattern `test_tools.py`/`test_graphrag.py`
already use for `TEST_EMBEDDING_DIM`), not the eval side's `from eval.conftest import ...` package
path (that would resolve to the wrong module here).

Mutation-tested: hand-reverted `_falkordb_reachable()` to the old write-mode `.query(...)` shape —
the new test correctly failed (`AssertionError: _falkordb_reachable() materialized
ws:k046-reachability-probe-does-not-exist as a side effect of merely probing it`). Restored the fix
and confirmed the test passed again.

**Suite counts:** Baseline (directly observed, before any change this session): `1034 passed, 2
deselected`. After this fix + new test (directly observed): `1051 passed, 2 deselected` — the +17
over baseline is +1 from this item's own new test plus +16 from K-047's `test_generate_report.py`,
which landed concurrently in this same shared working tree during this session (see the K-047 entry
below; its own before/after counts match). Several intermediate full-suite runs during this session
showed unrelated, non-reproducing failures scattered across `test_repository.py`,
`test_services_live.py`, `test_process_flow.py`, `test_graphrag.py`, etc. — traced to that
concurrent session's own `pytest` runs contending for the same live, shared FalkorDB instance
(different files failed on each run, and the suite passed cleanly again moments later with no
change on this item's side); not caused by or related to this change, and out of this item's scope
to fix.

**Scope:** `server/tests/conftest.py`'s `_falkordb_reachable()` and the new
`server/tests/test_conftest_probe.py` only — test-fixture-only, no production code, no graph/DDL
surface, per the backlog item's own risk rating.

## 2026-08-16 — K-047: `generate_report.py` rendering/branching test coverage — delivered

**What:** Added `server/tests/eval/test_generate_report.py` (16 tests), the first dedicated
automated test coverage for `server/tests/eval/generate_report.py`'s rendering/branching logic —
previously verified only by manual/static read at the K-026 Unit 3 `analyst` code gate (Major
M-1, non-blocking) and by `qa-engineer`'s acceptance-pass exploratory execution (TP-011). Covers
the four branches flagged there: (1) the not-run marker when `judge_calibration.json` is absent,
never fabricating numbers; (2) same-model-vs-differs judge caveat selection — the verbatim
`_SAME_MODEL_CAVEAT_TEMPLATE` block adjacent to the generation numbers vs. the plain "differs"
sentence, with a positional check that it lands where the module's own "never a trailing footnote"
docstring requires; (3) the self-retrieval-inflation guard's PASS/FAIL rendering, including a
parametrized check that a leaking golden row is caught whether it's first, middle, or last in the
row list; (4) the missing-`retrieval_baseline.json` `ReportError`, propagated uncaught by
`build_report()` and caught by `main()` (stderr `error: ...`, exit code `1`, no report file
written). Test-only change — `generate_report.py` itself is untouched (`git diff` empty).

Mutation-tested branches 2 and 4 per the task brief: inverting `if same_model:` to
`if not same_model:` correctly failed both same-model-caveat tests; making
`_load_retrieval_baseline` swallow the missing-file case (`return {}` instead of raising) correctly
failed all three missing-baseline tests (the unit-level `_load_retrieval_baseline` test, the
`build_report()`-propagation test, and the `main()`-level exit-code/stderr test). Both mutations
reverted; `git diff` on `generate_report.py` confirmed empty afterward.

**Suite counts:** `pytest tests/eval/test_generate_report.py -q` → 16 passed (network/DB-free, no
FalkorDB dependency). Full suite: `1034 passed, 2 deselected` before → `1051 passed, 2 deselected`
after (exactly +16, confirmed via `pytest --collect-only -q` with and without the new file: 1037
vs. 1053 total collected, 2 deselected in both). Note: intermediate full-suite runs during this
session intermittently showed unrelated failures in `tests/test_graphrag.py` (live-FalkorDB
vector/ANN tests) that reproduced identically with the new test file entirely removed from disk and
passed again in isolation — pre-existing environment flakiness in the shared live FalkorDB
instance, not caused by or related to this change, and out of this item's scope to fix.

## 2026-08-16 — K-026: GraphRAG retrieval + generation evaluation harness — QA-accepted, delivered

**What:** Delivered the K-026 evaluation harness for GraphRAG (`server/tests/eval/`): a retrieval-
metrics layer (recall@10/recall@5/MRR against a committed, `data-scientist`-signed-off baseline)
plus a calibrated LLM-as-judge layer over generation faithfulness/relevance, per the
`data-scientist` method note `docs/plans/graphrag-eval-ml.md`.

**Result: PASS** (`qa-engineer` acceptance pass, `docs/test-reports/graphrag-eval-report.md`, all
eleven test-plan items pass, no new defects). Numbers, quoted exactly as reported:
- Retrieval baseline (n=38): `recall@10=0.9737 recall@5=0.8947 mrr=0.6259`, exact-matching the
  committed `retrieval_baseline.json`.
- Judge–human calibration (fresh live run, N=10): "faithfulness agreement 90.0% (9/10), relevance
  agreement 70.0% (7/10)."
- Generation sub-pass (N=20): "20/20 faithful=true, 20/20 relevant=true, 0 parse failures" —
  flagged in the report itself as the exact pattern the same-model self-preference-bias caveat
  exists to warn a reader against over-trusting, not treated as an unqualified positive.
- Suite counts: default suite `1034 passed, 2 deselected`; live suite `2 passed, 1034 deselected in
  175.92s`; a second, independent `pytest tests/eval -q` re-run afterward reproduced `164 passed, 1
  deselected` identically, confirming re-runnability.

**Delivery chain (multi-session, `teco`-coordinated, per `docs/plans/graphrag-eval-coordination.md`):**
`architect` plan v1 → v4, gated by `analyst` across four passes (Pass 1 needs-changes: 1 Critical/2
Major/1 self-contradiction; Pass 2 needs-changes, narrower; Pass 3 **Approve**; a further v4 revision
for decision **D1** — collapsing agent-under-test and judge onto the single local model
`qwen/qwen3-4b-2507`, a stakeholder hardware-constraint call carrying a required self-preference-bias
limitation note — re-gated **Approve with suggestions**, Pass 4). Implementation ran as four units
across `coder`/`tdd-engineer`: corpus seed (Unit 1), golden-set authoring (Unit 2a, 38 pairs with
real embeddings), retrieval metrics (Unit 2b), and the judge layer (Unit 3). A genuine regression
surfaced and was fixed mid-run: **U-bug** — a `conftest` bare-module-name collision under pytest's
default `--import-mode=prepend` broke the repo-wide default suite's collection once the eval
subtree's own `conftest.py` was added; fixed by packaging `tests/eval/` (`__init__.py`) plus an
explicit `sys.path.insert` for the sibling bare imports the package-ize step broke (commit
`dbd2cdf`), independently re-verified green afterward. Two `analyst` code gates followed: **Unit
2b** first returned needs-changes (Blocker B-1 — `conftest.py`'s `_falkordb_reachable()` used
write-mode `GRAPH.QUERY`, silently materializing an empty `ws:eval` key on a fresh environment;
Major M-1 — the regression-detection branches had zero dedicated test coverage), both fixed
(`ro_query`+"empty key" pattern; an extracted `check_regression()` pure function with 6 new tests)
and re-gated **Approve with suggestions**; **Unit 3** gated **Approve with suggestions, no
blockers** on its first pass (a non-blocking Major on `generate_report.py`'s missing test coverage,
carried forward as a follow-up below). Two `data-scientist` sign-offs closed the methodology side:
the D1 self-preference-bias limitation note, and the `retrieval_baseline.json` (n=38) baseline
itself (Approve with suggestions — a non-blocking note that the zero-tolerance recall@10 gate sits
close to the noise floor at this n). `qa-engineer`'s final acceptance pass then verified everything
above by real execution rather than static review, and found no new defects.

**Document paths delivered:** plan `docs/plans/graphrag-eval.md` (v4); review
`docs/reviews/graphrag-eval.md`; ML method note `docs/plans/graphrag-eval-ml.md` (v2) + ML review
`docs/reviews/graphrag-eval-ml.md`; test plan `docs/test-plans/graphrag-eval.md`; test report
`docs/test-reports/graphrag-eval-report.md`; coordination log
`docs/plans/graphrag-eval-coordination.md`.

**Follow-ups filed at close, both non-blocking:** **K-046** (root `server/tests/conftest.py` carries
the same latent write-mode-`GRAPH.QUERY` bug pattern the Unit 2b B-1 fix corrected in the eval
subtree's own conftest) and **K-047** (`server/tests/eval/generate_report.py` has no dedicated
automated test file for its own rendering/branching logic — all branches manually verified correct
at the Unit 3 gate and independently re-confirmed correct by `qa-engineer`'s acceptance pass via
direct execution).

**Standing recommendation, reiterated not newly discovered:** the coordination ledger's own Notes
section records that golden-set "human verification" (the method note's non-negotiable validity
anchor) was substituted throughout this all-agent pipeline with independent `analyst` review, not a
literal human — the ledger explicitly recommends the user personally spot-check the ~10-example
`golden_judge_calibration.jsonl` set before fully trusting the judge–human agreement numbers above
at face value. The QA report repeats this same recommendation. Not a blocker for this PASS verdict.

## 2026-08-11 — K-044: LLM provider/model configuration admin manual authored

**What:** `tico` authored `docs/manuals/llm-provider-config.md`, resolving — the same session it
was filed — the "is an admin manual wanted?" question the K-042 Landing 2 close-out below had left
open. Covers the two hand-edited config files (the shared `opencode.json`; falkor-chat's own
`models.json` overlay), declaring a provider and its models, per-kind defaults and per-model
settings, naming a model or role on a specific workflow step/guard, roles and fallback chains,
setting a workspace override (a direct Cypher write today — no UI/REST route exists for it),
reading which concrete model actually answered a run off its execution trace, and the system's
failure modes. Illustrated with two Mermaid diagrams (the four-consumer resolution seam; the
workspace → own-choice → per-kind-default precedence chain).

**Bookkeeping note:** this delivery was never reflected in `docs/BACKLOG.md` (K-044 stayed listed
as open) or given its own entry here until closed out 2026-08-26 — a lapsed bookkeeping step, not a
re-litigated decision.

## 2026-08-11 — K-042 Landing 2: QA acceptance pass (U7) + D-2 fix — closes M4

**What:** First execution-based (black-box) verification of Landing 2, driven against the real
running server, real FalkorDB, and real LM Studio — two throwaway workspaces, four genuine server
restarts to exercise FR-15's no-reload-path for AC-6/AC-8, a genuinely-unreachable declared LAN
endpoint for AC-9's fallback-chain proof. `docs/test-plans/llm-provider-config2.md`
(`Extends:` the Landing-1 plan) + `docs/test-reports/llm-provider-config2-report.md`.

**Result: PASS.** All seven in-scope acceptance criteria hold live: AC-4's trace half (two steps,
two models, one non-debug run's `StepRun.resolvedModel`), AC-6 (role remap + restart, no
republish), AC-7 (publish-time rejection, both halves, plus the M-4 409-beats-400 ordering), AC-8
(drive-time unresolvable model, run fails with cause, no fallback used), AC-9 (fallback chain via a
genuinely-unreachable first element), AC-10 (workspace override hard cap across **all four
consumer kinds including `guard`** — the actual payoff of closing finding B-1), and AC-11
(embedding-dimension guard, pre-flight, no vector written on mismatch). Offline suite independently
re-reproduced: 866 passed, 1 deselected, matching the coordination record.

**One defect found, D-2 (Major):** `POST /workflow-runs` (and, by code inspection, `.../input`)
returned a raw `500` traceback instead of the documented `{"status":"failed","error":...}` envelope
for a drive-time `ModelResolutionError` — `services._drive_or_fault`'s caught-exception tuple
predated Landing 2's new drive-time fault classes. The run's own graph state was correct throughout
(confirmed via a follow-up `GET`); this was a REST-layer translation gap, not a correctness defect.
Unlike Landing 1's D-1 (Minor, deferred), D-2 was **fixed the same session** rather than deferred —
the stakeholder's explicit "keep full rigor, no shortcuts" instruction for these closing units,
weighed against the severity (a genuine break in the primary REST contract for exactly the scenario
Landing 2's own AC-8 exists to prove). The fix's scope was independently widened beyond the
QA-suggested one-line patch (`ModelResolutionError` only) after `teco` traced AC-8's own wording
("fails at call time... no fallback chain applies") to `ProviderCallError` — the transport/
fallback-exhaustion half of the same AC, unverified live in this pass but reachable by the
identical code path — and confirmed both belong in the fix by tracing `executor._drive`'s fault net
directly. `services._drive_or_fault` now catches both, with a corrected docstring explaining why
each belongs (and why `ModelConfigError` deliberately does not — it has no drive-time occurrence
path, config is parsed once at construction). Fixed, tested (4 new tests, reproduction-first),
mutation-tested independently by both `teco` and the `analyst` gate, and `analyst`-gated clean
(approve, no blockers/majors/minors) — commit `b3c3019`. Final offline suite: **870 passed, 1
deselected**.

**M4 is now fully reached** — both landings implemented, independently `analyst`-gated at every
step, and QA-accepted. `docs/BACKLOG.md`'s M4 row and K-042 item flipped to ✅. Residual, non-
blocking: `compose.yaml`/`Dockerfile` remain unverified against a real `docker build`/`docker
compose` (no Docker anywhere in this pipeline, carried since Landing 1); whether a
`docs/manuals/llm-provider-config.md` admin manual is wanted remains an open `tico` decision, not
yet raised to the stakeholder.

## 2026-08-11 — K-042 Landing 2: roles, workspace override, resolved-model trace, publish-time rejection, embedding-dimension guard (U8–U12)

**What:** Implemented Landing 2 of the model-resolution seam, `docs/plans/llm-provider-config.md`
§7 (L2-1..L2-6), sequenced as five independently `analyst`-gated units per the standing
"never a landing-wide mega-dispatch again" directive, plus this docs-close unit (L2-7):

- **Roles + ordered fallback chains (FR-7/FR-18, U8, `17c20dc`).** A ref with no `/` now resolves
  as a role name — `Overlay.roles[name]` — to an ordered, settings-applied fallback chain, rejected
  at load (not first use) if a role name contains `/` or a chain element resolves to another role.
  `FallbackClient` (`llm.py`) tries each chain element in order on a `ProviderCallError`, holds no
  mutable per-call state (`__slots__`), and reports the answering model and whether it fell back on
  `ChatResult.model`/`.fallback` — never on client state.
- **The resolved-model trace (FR-8, U8, `17c20dc`).** `StepRun` gains `resolvedModel`,
  `modelSource` (`workspace`/`step`/`default`) and `modelFallback` (nullable bool, orthogonal to
  `modelSource`), written by the existing atomic `record_step_and_advance` and surfaced on
  `GET /workflow-runs/{id}/step-runs` — a durable property, never a `TraceEvent` (those are
  debug-only by construction, `docs/plans/llm-provider-config-graph.md` §1.2). A node that loops
  across models records the last iteration's three fields together.
- **Workspace override + precedence, closing B-1 (FR-16/FR-17, U9, `0801b3c`).** A per-workspace
  `WorkspaceConfig` singleton, read once per drive/responder call and stamped onto
  `run["modelOverrides"]`. `ModelGateway.resolve()` implements the real first-match-wins
  precedence — workspace → the consumer's own requested choice → the per-kind default — with the
  workspace rung a **hard cap** reaching all four consumer kinds, `guard` included (closing the
  design-phase finding that the naive fix would reopen the SHA-locked `_drive_loop`).
- **Publish-time rejection (FR-9, U10, `eb1a60f`).** `_check_models_resolvable`, run immediately
  before `publish_def` and after K-034's topology-conflict check, resolves every declared
  step/guard model reference through the gateway and fails an unresolvable one with a 400 naming
  the step key (or transition endpoints) and the identifier; a def failing both checks returns the
  409, not the 400.
- **Loud use-time failure + embedding-dimension guard (FR-10/FR-19, U11, `44494d5`).** Confirmed
  the run-drive fault net and `background.py::_safe_respond`'s existing blanket try/except already
  satisfy FR-10 (an unresolvable model at drive time fails the run, no fallback model used) —
  zero production code changed, new tests only. `EmbeddingWorker` gained a pre-flight check
  comparing the resolved embedding model's declared dimension against the workspace's introspected
  vector-index dimension (`Repository.read_index_dimension`) before the first embed write per
  `(ws, label)`, refusing loudly (`EmbeddingDimensionError`) rather than writing a silently-
  unreachable vector.
- **Docs + close (U12, this entry).** `docs/DESIGN.md` §14.8 extended with the five items above;
  `falkor-chat/AGENTS.md`'s model/provider paragraph extended; `config/opencode.example.json`'s
  `openai` provider entry gained the missing `options.baseURL` (Landing-1 QA defect D-1).

Each of U8/U9/U10/U11 was independently gated by `analyst` on its own diff (no blockers anywhere
across the whole landing: approve with suggestions / approve / approve with suggestions / approve),
and `teco` independently re-verified suite counts, mutation-tested the hard-cap direction and the
publish-ordering rule, and confirmed the `_drive_loop` SHA lock unchanged before each commit. Full
detail, including two self-flagged findings not adjudicated as defects (a documentation-precision
gap in `-graph.md` §3.2's own worked example, and a flaky `test_queries.sh` assertion fixed
in-flight), is in `docs/plans/llm-provider-config-coordination.md`'s log (U8 through U11 delivered/
gated/committed sections).

**Test results:** offline suite **866 passed, 1 deselected** (from 791 at Landing-1 close); live
`./scripts/test_queries.sh` **320/320** (from 269 baseline), reseeded and verified in sync
(`bootstrap_schema.sh` → `seed_demo.sh` → `seed_workflows.sh` → `verify_workflows.sh acme` → `OK`)
after the last live-suite run. Both figures reproduced independently by `teco`, not taken on
report.

**Why:** K-042 (M4), Landing 2 of two — FR-7..FR-10/FR-16..FR-19. Lets a consumer name a role with
an automatic fallback chain instead of one hardcoded model, records which concrete model actually
answered on the durable execution trace, lets a workspace pin/override the model every consumer in
it uses (including the guard judge, closing finding B-1 from the design-phase review), rejects an
unresolvable model at publish time instead of first use, and refuses to write an embedding whose
dimension cannot match the workspace's vector index instead of silently dropping it out of ANN
retrieval.

**Not covered yet:** the Landing-2 QA acceptance pass (`qa-engineer`, unit U7 in the coordination
ledger) has **not** run as of this entry — AC-6 through AC-11 are implemented and code-reviewed but
not yet independently verified end-to-end against the running system. `compose.yaml`/`Dockerfile`
K-042 changes remain unverified against a real `docker build`/`docker compose` (no Docker anywhere
in this pipeline, carried forward from Landing 1). `docs/plans/local-model-ram-budget-ml.md` has
received its owner's (`data-scientist`) dated amendment noting the env-var mechanism K-042 replaced
(flagged since Landing 1, closed alongside this landing's docs-close work).

**Plan items:** `docs/plans/llm-provider-config.md` §7 (K-042, M4); `docs/BACKLOG.md` M4 row and
K-042 item updated to record Landing 2's implementation as complete and independently gated, QA
acceptance still pending (not flipped to ✅ — that happens on U7's PASS, matching how Landing 1's
own QA unit, U6, closed it out).

## 2026-08-11 — K-042 Landing 1: QA acceptance pass (U6)

**What:** First execution-based (black-box) verification of Landing 1, driven against the real
running server, real FalkorDB, and real LM Studio — distinct from the two prior static/diff-scoped
`analyst` gates on the design and the code diff. `docs/test-plans/llm-provider-config.md`
(TP-001..TP-010) + `docs/test-reports/llm-provider-config-report.md`.

**Result: PASS**, all nine in-scope acceptance criteria (AC-1, AC-4 partial, AC-5, AC-12, AC-13
end-to-end; AC-2/AC-3 structurally, no cloud API key available in this environment) hold against
the real system. Full offline suite independently re-reproduced: 791 passed, 1 deselected,
matching the coordination record's prior report. One defect found and filed: `config/opencode.
example.json`'s `openai` provider entry has no `options.baseURL`, so the shipped cloud-provider
example cannot itself resolve until one line is added (Minor — self-diagnosing, isolated, does not
affect the resolver logic, which was confirmed correct and uniform across all three provider kinds
once the missing key was supplied). All QA state lived in a throwaway `ws:qa-k042` graph, deleted
at teardown; `reference`/`ws:acme`/`ws:test` untouched throughout.

**Not covered (expected, no implementation exists yet):** AC-6..AC-11 (Landing 2). Also open:
`compose.yaml`/`Dockerfile` remain unverified against a real `docker build`/`docker compose` (no
Docker in this environment either — the same gap already flagged at U4); a live hosted-cloud-
provider call (no API key available, stakeholder decision).

## 2026-08-10 — K-042 Landing 1: the model-resolution seam (`ModelGateway`) + FR-20 env-var cutover

**What:** Replaced falkor-chat's four independently-constructed, env-var-configured LLM/embedding
clients with one internal resolution seam, per `docs/plans/llm-provider-config.md` §6 (L1-1..L1-6):

- **`server/falkorchat/transport.py`** (new) — `make_http_transport()`, `ProviderCallError`, the
  ordered §4.9 exception ladder (`HTTPError` before `URLError`, a bare `TimeoutError` named
  explicitly, `ValueError`, then the string-or-object body-level `error` renderer). Both
  duplicated, timeout-less `_urllib_transport` copies in `llm.py`/`embedding.py` deleted.
- **`server/falkorchat/modelconfig.py`** (new) — `Secret`, `ProviderCatalog`/`Overlay` (parse +
  `{env:}`/`{file:}` substitution), the `/v1` normalization rule (§4.3, validate → strip →
  normalize, with the overlay `providers.<id>.baseURL` escape hatch), per-kind defaults, per-model
  settings (reserved `timeout`/`dim`/`protocol`, everything else passed through camelCase→
  snake_case), `ModelGateway`/`StaticModelGateway`, `Resolution`/`ResolvedModel`. A ref with no
  `/` is rejected with a Landing-2-aware message; `roles`/`agents` overlay keys are parsed and
  logged as reserved, not honoured yet.
- **`llm.py`/`embedding.py`** — `LMStudioLLM`/`LMStudioEmbedder` renamed and generalized to
  `OpenAICompatibleLLM`/`OpenAICompatibleEmbedder` (required `base_url`/`model`, no config
  defaults, no back-compat alias); `ChatResult` gained a `model` field carrying the answering ref
  (the FR-8 carrier, unused until Landing 2).
- **All five consumer bindings rewired** onto one gateway: `executor.py` (`_run_agent_node`
  resolves kind `step`; `_drive` stamps `run["ws"]` — the Landing-2 `guard`-kind carrier, landed
  now per §4.10), `guards.py` (forwards `model=`/`run=` to the judge only when the guard/judge
  declare them), `responder.py` (`agent` + `embedding`), `embedding.py`'s `EmbeddingWorker`
  (`embedding`, with the §4.5 dim precedence: explicit override → overlay `dim` →
  `FALKORCHAT_EMBEDDING_DIM`), `tools.py`'s `GraphragRetrieveTool` (`embedding`, a real FR-4
  consumer per M-3, not the type-hint-only change v1 mis-booked it as), `app.py` (`_build_llm_judge`
  now returns an `accepts_run = True` object, not a closure; `_build_default_app` builds one
  `ModelGateway.from_env()`). A directly-injected `llm=`/`embedder=` client is unchanged sugar
  (`StaticModelGateway`) — all 38 `llm=`/24 `guard_judge=` test injections still pass unmodified.
- **FR-20 cutover:** deleted the four legacy env-var constants from `config.py`; added
  `assert_no_legacy_model_env()` (called from `ModelGateway.from_env()`), `OPENCODE_CONFIG_PATH`
  (`FALKORCHAT_OPENCODE_CONFIG`, no product default) and `MODEL_CONFIG_PATH`
  (`FALKORCHAT_MODEL_CONFIG`, defaults to the shipped `config/models.json`). Updated
  `server/.env.example`, `scripts/start_server.sh` (the `$HOME/.config/opencode/...` dev
  convenience default), `compose.yaml` (the two vars, a read-only bind mount of the shared file,
  `host.docker.internal:host-gateway`), `Dockerfile` (`COPY config config`), `README.md`,
  `AGENTS.md`. Shipped `config/models.json` (the overlay defaults) and
  `config/opencode.example.json` (LM Studio + a second LAN host + hosted OpenAI via `{env:}`).
- **Docs:** `docs/DESIGN.md` §1.3 (model rows now point at the shipped config), §14.2 (layering
  diagram gains `modelconfig.py`/`transport.py`), new §14.8 "The model-resolution seam", §14.7
  hazard bullet (two config files now required for a wired agent; `tests/conftest.py`'s
  `_model_config_env` autouse fixture points both at `tests/data/`).

**Test results:** `.venv/bin/python -m pytest -q` → **778 passed, 1 deselected** (the `live`
marker), including two new suites (`test_transport.py`: 13; `test_modelconfig.py`: 51) and
extensions to `test_llm.py`, `test_embedding.py`, `test_app.py`, `test_guards.py`. Re-verified
green with `HOME` pointed at an empty directory (the M-2 done-condition — no dependency on a
developer's real `~/.config/opencode/opencode.json`). The FR-4 enforcement test (an AST scan of
`server/falkorchat/*.py`) confirms no module outside `modelconfig.py` constructs
`OpenAICompatibleLLM`/`OpenAICompatibleEmbedder` directly. The unfiltered legacy-env-var grep
(`docs/plans/llm-provider-config.md` §2.9's method) returns only `config.py`'s own tripwire list,
the K-042 planning documents, and `docs/plans/local-model-ram-budget-ml.md` (explicitly not
rewritten by this unit — routed to its own owner).

**Why:** K-042 (M4), Landing 1 of two — FR-1..FR-6/FR-11..FR-15/FR-20. Replaces four
independently-hardcoded LM Studio clients with one config-file-driven seam so any of the five LLM
consumers can name its own model/provider without a code change, and closes a live-verified silent
failure mode (a missing `/v1` returns HTTP 200 with an error envelope, not an error status).
Landing 2 (roles, ordered fallback chains, workspace override + precedence, trace-recorded
resolved model, publish-time rejection, the embedding-dimension guard) is separately tracked.

**Plan items:** `docs/plans/llm-provider-config.md` (K-042, M4); `docs/BACKLOG.md` M4 row updated
🟡 → 🟡 (Landing 1 delivered, Landing 2 pending).

## 2026-08-09 — Fix: `AGENTS.md` citation nit (`get_snapshot` vs. `_read_subgraph` line numbers)
- **What:** Corrected `AGENTS.md`'s "Probing shared graph state without mutating it" subsection,
  which cited `get_snapshot`/`_read_subgraph` under one shared line number (`repository.py:1031`)
  — that's `_read_subgraph`'s definition only; `get_snapshot` (which calls into it) is actually at
  `:1702`. Now attributes each symbol to its own line.
- **Why:** Flagged as a nit by an independent review (`docs/reviews/kaizen-inbox-distillation.md`)
  of the same-day distillation pass that introduced the subsection.
- **Plan items:** none.

## 2026-08-09 — Docs: safe shared-state probing notes + collect-only counting tip (analyst inbox distillation)

**What:** `AGENTS.md` gained a new "Probing shared graph state without mutating it" subsection
(after the Key scripts table): the `publish_def`/`materialize_snapshot` graph-seam asymmetry (a
def *publish* always hits the global `reference` graph, but the *snapshot* side reuses the same
Cypher constants against a throwaway `ws:<probe>` and can be torn down safely) and
`server/tests/test_services.py` as the review-safe pytest subset (pure `FakeRepo`, no FalkorDB
fixture). The `bootstrap_schema.sh` Key-scripts row gained a note that it always touches
`reference` too (DDL-only, safe). `docs/DESIGN.md` §14.7 gained a note that `pytest
--collect-only -q` is the non-mutating way to check a claimed test count.
**Why:** Distilled from four `analyst` learnings-inbox entries (2026-07-24) surfaced across the
K-031 review and other falkor-chat gates — verification techniques specific to this component's
shared-graph layout, not generic enough for `analyst`'s own prompt. Routed by `cobb` per
`agent-maintenance` skill §5; full disposition rationale in `claude/analyst/kaizen/history.md`
(2026-08-09).
**Plan items:** none.

## 2026-08-01 — K-041: MCP `send_message` now schedules the responder/workflow trigger (D-1 fix)

**What:** Fixed a High-severity defect found by a live QA pass of the unrelated `kiro-demo-agent`
feature (`kiro/docs/test-reports/kiro-demo-agent-report.md`, Defect D-1): a message posted through
the **MCP** `send_message` tool — including an `@mention`-bearing one — never triggered
`assistant`'s reply or the M3 workflow trigger, because the `BackgroundTasks` scheduling that
implements the M3 one-handler guarantee (exactly one of {trigger, responder} handles a posted
message, `embed_worker` always-if-configured) lived only in `api.py`'s REST route, never in
`mcp.py`. The REST route (`POST /threads/{id}/messages`) always worked correctly; every prior
"`@mention` → reply" QA/test pass went through REST, never a real MCP client, so the gap had never
been exercised.

- **New shared module** — `server/falkorchat/background.py` now holds `_safe_embed`/
  `_safe_respond`/`_safe_run_workflow` (moved out of `api.py` verbatim, behavior unchanged), so the
  M3 one-handler policy is defined exactly once and imported by both transports instead of two
  hand-synced copies.
- **`mcp.configure()`** (`server/falkorchat/mcp.py`) now accepts `responder`/`embed_worker`/
  `trigger`, mirroring `api.build_router`'s signature, stored as new module-level state.
- **`mcp.py`'s `send_message`** now calls a new `_schedule_background(ctx, posted)` helper after a
  successful post, replicating `api.py`'s exact scheduling order. Since a plain `@mcp.tool()`
  function has no per-call object like FastAPI's `BackgroundTasks`, scheduling defaults to a daemon
  `threading.Thread` fire-and-forget (`mcp._default_schedule`) — the already-synchronous,
  failure-isolated `_safe_*` functions run off-band so the message write is never blocked. The
  scheduling call itself is a swappable module-level seam (`mcp._schedule`) that tests override for
  deterministic, non-racy assertions.
- **`app.py`'s `create_app()`** now passes `responder=responder, embed_worker=embed_worker,
  trigger=trigger` into `mcp_mod.configure(...)` — the same three objects already passed to
  `api.build_router(...)` a few lines later.
- **Doc cross-references corrected**: three docstring/comment sites that named
  `api._safe_run_workflow` (`services.py`, `executor.py`, `tests/test_process_input.py`,
  `tests/test_workflow_live.py`) now point at `background._safe_run_workflow`, its new home.

**Test counts:** test-first (`tdd-engineer`). New MCP-side `Recording*` doubles
(`RecordingWorker`/`RecordingResponder`/`RecordingTrigger`, mirroring `test_api.py`'s) plus a
`sync_schedule` fixture (swaps the `_schedule` seam for an inline call) added to
`server/tests/test_mcp.py`. Five new tests: `test_send_message_schedules_responder_with_posted_message`,
`test_send_message_trigger_wired_schedules_trigger_not_responder`,
`test_send_message_embeds_independently_of_trigger_or_responder`,
`test_send_message_with_no_wiring_posts_normally`,
`test_send_message_default_scheduling_runs_off_a_background_thread` (asserts the default
threading-based path genuinely runs off a different thread than the caller, confirmed non-flaky
over repeated runs). `pytest -q` **691 → 696 passed, 1 deselected** unchanged.
`./scripts/test_queries.sh` unaffected — **282/282**, no Cypher touched by this fix.

**Files touched:** `server/falkorchat/background.py` (new), `server/falkorchat/api.py` (imports
the three functions instead of defining them), `server/falkorchat/mcp.py` (`configure()` +
`send_message` + `_schedule_background` + `_schedule` seam), `server/falkorchat/app.py`
(`mcp_mod.configure(...)` wiring), `server/falkorchat/services.py`, `server/falkorchat/executor.py`
(doc cross-reference fix), `server/tests/test_mcp.py` (new tests + doubles),
`server/tests/test_process_input.py`, `server/tests/test_workflow_live.py` (doc cross-reference
fix), `docs/BACKLOG.md` (K-041).

## 2026-08-01 — K-034: close the additive-`MERGE` defect — topology-conflict gate on re-publish/re-materialize

**What:** Implemented `docs/plans/workflow-republish-semantics.md` (reviewed, approve with
suggestions). Republishing/re-materializing an existing `(key, version)` with a *structurally*
differing payload — a new/removed step key, a retargeted or removed transition, or a moved start
key — is now rejected (`409 WorkflowDefConflictError`, nothing written) instead of silently
minting parallel `TRANSITION`/`START` structure beside the old edges. A **property-only**
resubmit (`name`, `kind`, a step's `type`/`config`, a transition's `guard`) is unchanged: still a
silent no-op, exactly the K-031-pinned behavior.

- **New gate, service layer only.** `services._structural_diffs` filters K-031's existing
  `_diff_structures` output to topology-changing paths; `services._check_no_structural_conflict`
  reads the existing structure (`Repository.read_def_structure`/`read_snapshot_structure`, both
  pre-existing K-031 reads), diffs it against the candidate, and raises on any structural survivor.
  Wired into `services.publish_workflow_def` (against `reference`) and `services.materialize_def`
  (against `ctx.ws`'s snapshot), both *before* the repository write. `Repository.publish_def`/
  `materialize_snapshot` and `_PUBLISH_CYPHER` itself are **unchanged** — the guarantee is
  enforced one layer up, same layering `_validate_def_spec` already established.
- **New error** — `WorkflowDefConflictError` (`repository.py`, re-exported by `services`), mapped
  to `409` in `app.py` the same way `WorkflowRunNotWaitingError` already is ("state conflict,
  nothing written").
- **Defense-in-depth** — `executor._select_transition`'s sort key gained `to` as a final,
  additive-only tie-break, so a pre-existing or bypass-created duplicate outgoing transition
  resolves deterministically regardless of edge-retrieval order, rather than depending on
  whatever order FalkorDB happened to return. Confirmed outside the SHA-locked `_drive_loop`
  (`71055f756280`, unchanged — verified byte-identical before/after via AST hash).
- **Doc/docstring sweep** — corrected every live (non-archived) site the K-034 backlog table and
  this plan named: `docs/QUERIES.md` §11 preamble/footnotes, `docs/DESIGN.md` (topology diagram,
  §4 decision paragraph, §9 write-paths row), `falkor-chat/AGENTS.md`'s `seed_workflows.sh` row,
  `docs/requirements/agent-import.md`, `docs/requirements/workflow-dependence-overlay.md`,
  `docs/BACKLOG.md`'s K-029 and K-032 premises, `repository.py`'s `publish_def`/
  `materialize_snapshot` docstrings and §11 block comment, `services.py`'s `materialize_def`/
  `publish_workflow_def` docstrings (also documents the residual TOCTOU race for **both**
  `publish_workflow_def` and `materialize_def`, broadening the plan's publish-only framing per
  the review's Minor 2), and `api.py`'s §11 section comment. Corrected framing throughout:
  "create-only on properties, topology-enforced by services (K-034)" — not "immutable"/"idempotent"
  unconditionally. Archived documents, `HISTORY.md`'s own past entries, `docs/reviews/*`, and
  already-executed plans (`workflow-def-structure-read.md`, `m3-followups-coordination.md`) were
  correctly left untouched per root `AGENTS.md`'s frozen-document rule.
- **Review Minor 1 verified, not just trusted.** `tests/test_api.py::
  test_diff_reports_divergence_after_the_documented_reseed_trap` was run as an early canary before
  writing the new `409` tests and confirmed green: the preceding `_wipe_reference` call makes
  `existing_raw is None`, so the gate never fires and the resubmit is treated as a first-time
  create, exactly as the review traced.

**Test counts:** `test_queries.sh` **282/282**, unaffected (`_PUBLISH_CYPHER` untouched, as the
plan required). `pytest -q` **658 → 691 passed, 1 deselected** (+33 new tests: 12 `_structural_diffs`
unit tests, 7 `publish_workflow_def` gate tests + 7 `materialize_def` gate tests (FakeRepo,
service layer), 2 executor tie-break tests (parametrized, reversed order), 1 repository
contract-boundary test (pins that a raw `Repository.publish_def` call stays unsafe on its own),
3 live API end-to-end tests (changed transition `to`, changed start key, materialize-side drift —
all asserting the structure is byte-identical before/after the rejected write), 1 app-wiring test
(409 mapping, isolated from the gate logic).

**Files touched:** `server/falkorchat/repository.py` (new error class, corrected docstrings/
comments), `server/falkorchat/services.py` (`_structural_diffs`, `_check_no_structural_conflict`,
gate wiring + docstrings), `server/falkorchat/app.py` (409 handler), `server/falkorchat/api.py`
(comment), `server/falkorchat/executor.py` (tie-break), `server/tests/test_services.py`,
`server/tests/test_api.py`, `server/tests/test_repository.py`, `server/tests/test_executor.py`,
`docs/QUERIES.md`, `docs/DESIGN.md`, `docs/BACKLOG.md`, `docs/requirements/agent-import.md`,
`docs/requirements/workflow-dependence-overlay.md`, `falkor-chat/AGENTS.md`.

## 2026-07-31 — K-039 item 3: CI blind-spot follow-up — recent triage post-success readiness signal

**What:** Addressed the RCA's contributing-factor-2 gap by building a lagging, informational
post-success signal into the existing `GET /workspaces/{ws}/readiness` route
(`Services.check_demo_readiness`), separate from the deterministic `ready` boolean. Reports the
last 20 terminal `triage@v1` runs' post-success rate (`postSuccess: {defKey, defVersion,
sampleSize, postedCount, rate, status}`, `status` ∈ `"ok"`/`"degraded"`/`"no-data"`), deliberately
kept informational to avoid flipping `ready` on LLM mood swings. The RCA had noted that
`pytest -m live`'s AC-4 answer-post assertion was deterministically RED but excluded from the
default `pytest -q` run, giving false confidence about this exact path; the readiness banner now
surfaces a production-data metric to close that blind spot.

**Implementation:** Built by `graph-dba` (new query, `docs/QUERIES.md` §12.15,
`test_queries.sh` 276→282), `coder` (repository/service wiring, `pytest -q` 647→658, 11 new
tests), and `frontend-engineer` (banner rendering in `web/index.html`/`web/app.js`).

- **New query** — `repository.read_recent_post_success(ws, def_key, def_version, limit)`:
  counts the last-N terminal (`done`/`failed`) runs of a given def and how many produced at least
  one `StepRun -[:PRODUCED]-> Message` edge. Anchors on `WorkflowRun.status` (independently
  indexed, `bootstrap_schema.sh:145-146`); verified live via `GRAPH.PROFILE`.
- **Service layer** — `Services.check_demo_readiness` now composes
  `Repository.read_recent_post_success` into the response alongside the existing `ready`/`defs`
  fields. `postSuccess.status` is `"no-data"` when `sampleSize == 0` (fresh workspace, no runs
  yet), `"ok"` when every sampled run posted (`postedCount == sampleSize`), `"degraded"`
  otherwise (at least one sampled run completed without posting). `rate` is `None` in the
  no-data case, else `postedCount / sampleSize`.
- **Web banner** — K-036's existing ready-to-demo banner now displays the post-success signal
  as a separate status indicator (does not gate the `ready` boolean).

**Test counts:** `test_queries.sh` **276 → 282/282** (+6 query assertions for the new query's
empty/populated/ordering/edge cases); `pytest -q` **647 → 658 passed, 1 deselected** (+11 new
tests: 6 repository integration cases, 5 service unit cases; the two existing `test_api.py`
readiness-route tests were extended in place, not added, including widening
`_READINESS_KEYS` to `{"ready", "defs", "postSuccess"}`).

**Review gates:** Plan-gate (`docs/reviews/mention-reply-delivery.md` Pass 1, 2026-07-31) —
**approve with suggestions** (4 minor + 1 nit, all folded into plan v2). Diff-scoped re-gate
(Pass 2, 2026-07-31) — **approve, no findings**. Both gates checked the edge-case float/int
handling, PROFILE verification against the live engine, and cross-referenced the RCA's own
contributing-factor-2 note.

**QA acceptance:** `qa-engineer` ran a black-box acceptance pass against the delivered code
(no defects found in K-039's own diff). Separately, re-ran `pytest -m live` once with LM Studio
reachable against commit that included K-039 item 1's implicit-`post_message`-fallback fix;
AC-4's answer-post assertion — the same one K-027's D12-B documented as deterministically RED —
now passes, confirming the implicit-fallback fix resolved the failure mode it was designed for.
(Note recorded at `docs/BACKLOG.md` K-027 §5, "Update (2026-07-31...)".)

**Scope completion:** K-039 scope items 1 (immediate mitigation, ✅ 2026-07-31) + 3 (CI
blind-spot follow-up, ✅ 2026-07-31) are both delivered. Item 2 ("do not fold into K-027 item 2")
is a non-deliverable scope clarification (the broader engine contract stays open in K-027). K-039
itself is now **complete**.

Files touched: `falkor-chat/server/falkorchat/repository.py`, `falkor-chat/server/falkorchat/services.py`,
`falkor-chat/web/app.js`, `falkor-chat/web/index.html`, `falkor-chat/docs/QUERIES.md`.

## 2026-07-31 — Cleanup: RCA + QA live-repro test artifacts removed from `ws:acme` demo data

**What:** Removed the live test artifacts two rounds of K-039 reproduction/acceptance testing left
in the demo workspace `ws:acme`, per stakeholder request ("clean up the demo test artifacts").
Both authors had disclosed their artifacts rather than silently leaving them
(`docs/reviews/mention-reply-delivery-rca.md` §2, `docs/test-reports/mention-reply-delivery-report.md`
"Artifacts left in the live demo"). Full live inventory taken before any delete (edge-by-edge,
mirroring the 2026-07-30 K-037 follow-up's methodology below):

1. **`analyst`'s RCA repro**, spliced out of the pre-existing `demo-welcome` thread (22 other
   messages untouched): message `ae8719305b5d4f3bb580b7e4c6d05253` ("analyst-rca-live-repro: what
   is 2+2?") plus the `WorkflowRun` it triggered (`00d95a27ac2a4dc8b74a86ed117b5c95`, `triage@v1`,
   3 `StepRun`s — `intake`/`research`/`answer` — none of which had produced a `Message`, matching
   the RCA's own root-cause finding). The message was the thread's `TAIL`; deleted and relinked
   `TAIL` to the prior message (`07b9e0da006c4893893a150ef27adcc1`) in the same atomic query — no
   `NEXT`-chain gap left behind.
2. **`qa-engineer`'s acceptance pass**, two whole threads under `demo-general` (both created by
   that pass, confirmed via full message-chain walk — nothing pre-existing in either):
   - `4c7eb4368bee4b12a1ea85b4dc18d300` ("qa-mention-reply-delivery"): 13 messages, 3
     `WorkflowRun`s (`ae7b7a4a36754b63a30852bb9e43a7ce`, `58a3933a581f4fd6950fae10879ea641`,
     `9265582e1b8f4c5c994f9a2eb3c71908`), 11 `StepRun`s (the third run resumed twice — 3×`intake`
     + `research` + `answer`), all `PRODUCED` messages included.
   - `b9984e3097c04aacb51f552970036768` ("qa-mention-reply-delivery-clean"): 1 plain message, no
     workflow run.
   Both `Thread` nodes deleted along with their content; **`demo-general` channel itself kept** —
   it is the pre-existing demo channel (it also hosts `demo-welcome`), not something QA created,
   so only its two QA-added threads were removed, not the channel.

**Safety checks (before deleting):** full undirected edge inventory on every message/run/step-run
in scope confirmed the only edges touching them were within-scope (`NEXT`/`HEAD`/`TAIL`/
`TRIGGERED_BY`/`HAS_STEP_RUN`/`LAST_STEP_RUN`/`PRODUCED`/`OF_DEF`) or into shared, never-deleted
nodes (`User`/`Agent` identity nodes via `POSTED_BY`/`MENTIONS_MEMBER`, and the `triage@v1` def's
shared `Step` nodes via `RAN`) — no `AT_STEP` edge from any other run pointed at these step-runs,
and no unrelated thread/message referenced any of this content. The RCA's own separately-noted
historical corroborating runs (`6dea1ba3c5d543cebf5f5a578ad07073` and the other pre-existing
`demo-welcome` mentions that happened to exhibit the same bug, not created by the repro itself)
were explicitly left untouched — out of scope for this cleanup.

**Verified — before/after counts:**
- `ws:acme` overall: **131 nodes / 353 relationships → 96 nodes / 256 relationships** (35 nodes /
  97 relationships removed, exactly matching the three delete queries' own reported counts: 5/15 +
  28/79 + 2/4).
- `demo-welcome`: **23 → 22 messages**; `HEAD` unchanged (`c00eb063af094bce93bf565a5fdf1860`);
  `TAIL` moved `ae8719...` → `07b9e0da006c4893893a150ef27adcc1`.
- `demo-general` channel: **3 → 1 thread** (only `demo-welcome` remains).
- All target msgIds/threadIds/runIds confirmed at **count 0** after.
- `./scripts/verify_workflows.sh acme`: **FAIL** — unchanged from the pre-existing, already-flagged
  state (`reference` MISSING both `triage@v1`/`access-request@v1`, `ws:acme` snapshot present
  both) documented in the RCA/QA report and tracked separately; this cleanup introduced **no new**
  drift (identical MISSING/present signature before and after).

**Why:** Stakeholder-requested demo-data cleanup, explicitly authorized for this dev environment.
Done with the same surgical rigor as the 2026-07-30 K-037 follow-up: full inventory before delete,
safety checks that nothing live still references what's removed, one atomic parameterized
`DETACH DELETE` per logical unit (IDs passed via `CYPHER` parameters, never string-interpolated),
before/after counts recorded as evidence, not asserted.

**Out of scope, untouched:** the pre-existing `reference`/`ws:acme` workflow-def drift (separately
tracked, already flagged to the user); the RCA's historical corroborating runs in `demo-welcome`
unrelated to its own repro; all of `demo-welcome`'s and `demo-general`'s other content.

Files touched: `falkor-chat/docs/HISTORY.md`.

## 2026-07-31 — K-039 immediate mitigation: implicit `post_message` fallback when a granted tool goes uncalled

**What:** Fixed the demo-blocking bug root-caused live in
`docs/reviews/mention-reply-delivery-rca.md`: `@mention`-ing the demo assistant ran `triage@v1` to
`status: done` while posting **zero** chat messages, because the local chat model
(`qwen/qwen3-4b-2507` via LM Studio) routinely ends an `agent` node's turn with plain text instead
of calling its granted `post_message` tool, and `executor.py`'s `_run_agent_node` treated that as a
normal, successful termination — the text was returned as `StepResult.output` and silently
discarded (no `Message` node, no `PRODUCED` edge). This lands the RCA's suggestion-1 "immediate,
demo-scoped mitigation" only; the full engine-level "terminal-node-must-post" contract (K-027 item
2) is unaffected and stays open.

**Fix (`server/falkorchat/executor.py`, `_run_agent_node`):** in the non-tool-call branch (`not
result.is_tool_call`), when the node's granted tools include `post_message`, `result.text` is
non-empty, and the node has not already posted a message earlier in the same loop
(`not emissions` — guards against double-posting after a real explicit call followed by a plain
"done" narration turn), the executor now synthesizes an implicit `ToolCall("post_message", …)` and
dispatches it through the existing `_handle_tool_call` path — the same validation, tracing, and
emission-buffering a real model-initiated call gets, so a successful implicit post still flows
through the normal post-record `StepRun -[:PRODUCED]-> Message` linking
(`_link_emissions`/`_record`). Covers both failure shapes the RCA identified: plain prose with no
tool-call shape at all, and a call whose `mentions` argument gets rejected (a leaked display name)
followed by the model "recovering" by dropping the tool on a later turn — both funnel through the
same branch, so one fix location covers both. A dispatch failure on the implicit call is absorbed
exactly like a real one (logged, traced, run still completes) since there is no further turn left
to re-prompt.

**Verification (test-first, TDD):**
- Reproduction tests added first and confirmed RED for the right reason (no dispatch, no
  emissions) before the fix:
  - `server/tests/test_executor_agent.py`:
    `test_plain_text_with_granted_post_message_is_posted_as_implicit_fallback` (primary repro —
    plain prose, tool never called),
    `test_recovery_after_mention_rejection_still_posts_via_implicit_fallback` (second shape —
    rejected call, then a "recovered" plain-text turn),
    `test_no_implicit_post_when_post_message_not_granted` and
    `test_no_implicit_post_when_final_text_is_empty` (negative guards).
  - `server/tests/test_executor_produced.py`:
    `test_implicit_post_when_tool_not_called_still_creates_produced_edge_live` — the full
    integrated path (real `Services` + real `ToolRegistry`/`PostMessageTool`, live `ws:test`
    graph): asserts a real `Message` node and `StepRun -[:PRODUCED]-> Message` edge now exist,
    mirroring the RCA's live repro exactly.
- All five now pass after the fix; the pre-existing suite (including
  `test_hallucinated_mention_does_not_fail_the_run` and
  `test_agent_node_captures_posted_msg_ids_as_emissions`, both of which exercise adjacent branches
  of this same code) remains green.
- Full offline suite: **642 passed, 1 deselected → 647 passed, 1 deselected** (the 5 new tests; the
  1 deselected is the known `@pytest.mark.live` characterization test, unaffected, still tracked
  under K-027/D12-B — not in scope here).
- `_drive_loop` byte-identity SHA-lock (`docs/DESIGN.md` §6.2 / project invariant) reconfirmed
  unchanged before and after: `71055f756280` both times — the fix lives entirely in
  `_run_agent_node`, below the `# ── seams ──` marker, never touching the locked loop.

**Left alone, by design (per the RCA's own scope split):** the general engine-level
"terminal-node-must-post" contract (K-027 item 2, `architect`-owned), promoting `pytest -m live`
into the default run (RCA §5 item 2), and the two residual test artifacts the RCA's own live repro
left in `ws:acme` (`msgId ae8719305b5d4f3bb580b7e4c6d05253`, `runId
00d95a27ac2a4dc8b74a86ed117b5c95`) — untouched, that decision belongs to whoever owns `ws:acme`'s
demo data.

## 2026-07-30 — K-037 follow-up: surgical cleanup of `ws:acme`'s contaminated `access-request@v1` snapshot

**What:** Removed the 3 spurious `Step` nodes the historical K-037 env-var-collision bug had
grafted onto `ws:acme`'s `access-request@v1` `WorkflowDefSnapshot` — the `ws:acme`-side
contamination the same-day K-037 entry above left open ("the `ws:acme` snapshot side remains
divergent and untouched... scoped narrowly to repairing the `ws:acme` snapshot"). Confirmed via
Cypher against `ws:acme` (not assumed) that the 9 `Step`s under the snapshot were the canonical 6
(`submit`, `route`, `approval`, `provision`, `activate`, `rejected` — cross-checked against
`server/falkorchat/proof_defs.py`'s `ACCESS_REQUEST_DEF`) plus exactly 3 spurious ones with
`stepUid`s `access-request:v1:intake`, `access-request:v1:research`, `access-request:v1:answer`
(the `triage`-shaped literal's step keys). Full edge inventory on the 3 spurious nodes found only
one `HAS_STEP` edge each (inbound, from the snapshot root) plus a `TRANSITION` chain among
themselves (`intake→research→answer`) — no edge connected any spurious node to a real/reachable
step in either direction, matching the 1-`START`-edge-at-`submit` topology. Safety check: zero
`WorkflowRun` in `ws:acme`, live or historical, had an `AT_STEP` edge pointing at any of the 3
spurious step nodes (the only `AT_STEP` edges present target `submit` and `provision` — real
steps) — the "unreachable dead node" assumption from the original QA finding held. Deleted the 3
`Step` nodes with a single atomic, parameterized `DETACH DELETE` (stepUids passed via `CYPHER
spurious=[...]`, never string-interpolated) rather than a full delete+republish of the snapshot
subgraph, since a republish would need to also correctly re-tie any live `WorkflowRun`s already
pointing at this snapshot — unnecessary blast radius for the same surgical outcome.

**Verified — before/after Cypher counts:**
- Before: **9 `Step`s**, **1 `START` edge** (`submit`) under `ws:acme`'s `access-request@v1`
  snapshot.
- Delete query result: `Nodes deleted: 3`, `Relationships deleted: 5` (3 `HAS_STEP` + 2
  `TRANSITION`, exactly matching the edge inventory taken beforehand).
- After: **6 `Step`s** (`activate`, `approval`, `provision`, `rejected`, `route`, `submit` — the
  canonical set, confirmed by key), **1 `START` edge** (`submit`, unchanged). The 3 pre-existing
  `WorkflowRun`→`AT_STEP` edges (targeting `submit`×2 and `provision`×1) confirmed still intact
  and untouched.
- `./scripts/verify_workflows.sh acme`: **`RESULT: OK — 2 defs in sync between `reference` and
  ws:acme`** — both `triage@v1` and `access-request@v1` report `in sync: YES`, closing the
  divergence the same-day K-037 entry's pytest-fixture follow-up had left open.

**Why:** The K-037 backlog item explicitly deferred this as a separate, stakeholder-gated
destructive-data cleanup, distinct from the script fix itself (`falkor-chat/docs/BACKLOG.md`,
K-037: "Route to `graph-dba`... for the `ws:acme`/`reference` cleanup hand — deleting and
republishing... is a destructive shared-state op on live data"). The `reference`-side half of that
cleanup was incidentally resolved by an unrelated pytest-fixture wipe + standard reseed (recorded
in the K-037 entry above); this entry closes the remaining `ws:acme`-side half, done deliberately
and surgically rather than by accident.

No schema/query-shape change; read-only + one scoped delete against live data, explicitly
authorized for this dev environment. `reference`'s already-clean `access-request@v1` (6 `Step`s, 1
`START` edge) was read-verified but not touched.

Files touched: `falkor-chat/docs/HISTORY.md`, `falkor-chat/docs/BACKLOG.md`.

## 2026-07-30 — K-037: decoupled `seed_workflows.sh`'s triage-literal publish identity from `FALKORCHAT_TRIGGER_DEF_KEY`, fixed `start_server.sh`'s stale banner

**What:** Fixed the env-var name collision (Finding 1, major) that let a sanctioned
demo/QA override silently corrupt the `reference` graph. `scripts/seed_workflows.sh` used to read
`FALKORCHAT_TRIGGER_DEF_KEY`/`_VERSION` — the app's real `config.TRIGGER_DEF_KEY`/`_VERSION`
override, which decides which def an `@mention` resolves to — to decide what key/version to
publish its own inline `triage`-shaped step literal under. Overriding the trigger (e.g. to
`access-request@v1`, for a demo) therefore also redirected the triage literal's publish target: the
`WorkflowDef` node reported "already present — no-op" (it already existed), but the per-step
`MERGE` underneath is keyed by `stepUid`, so 3 spurious `Step`s + 1 spurious `START` edge got
silently grafted onto the unrelated, already-published def. Fix: the triage literal's publish
identity now comes from its own dedicated pair, `FALKORCHAT_TRIAGE_DEF_KEY`/
`FALKORCHAT_TRIAGE_DEF_VERSION` (default `triage`/`v1`, matching `config.TRIGGER_DEF_KEY`/
`_VERSION`'s own defaults so an unoverridden run is unaffected), fully decoupled — the triage
literal's publish key/version is never read from `FALKORCHAT_TRIGGER_DEF_KEY`/`_VERSION` again.
Also fixed Finding 2 (minor, cosmetic): `scripts/start_server.sh`'s startup banner hardcoded
`"triage def triage@v1"` regardless of an active trigger override; it had no defaults-section entry
for `FALKORCHAT_TRIGGER_DEF_KEY`/`_VERSION` at all (unlike every other overridable var). Added
defaults (matching `config.py`), exported them to uvicorn, and the banner now interpolates the
actual configured key/version. Both scripts' header-comment env-var docs updated; `docs/BACKLOG.md`
K-037 entry flipped to delivered.

**Why:** Filed out of K-036's Wave 5 QA pass (`docs/test-reports/web-api-coverage-report.md`,
Finding 1 + Finding 2) — the plan's own sanctioned Pass-B demo/QA procedure (restart with
`FALKORCHAT_TRIGGER_DEF_KEY=access-request FALKORCHAT_TRIGGER_DEF_VERSION=v1
./scripts/start_server.sh`) is exactly the sequence that triggers the corruption; every restart
using it further and irreversibly corrupted `reference` (publish/materialize are append-only, so
there is no clean way to undo it short of a destructive delete+republish). Script-only change, no
schema/query-shape change (Risks/RAM: none) — no `graph-dba` gate needed. The already-corrupted
`reference`/`access-request@v1` data (from prior sessions, pre-dating this fix) is untouched —
its cleanup is a destructive shared-state op on live data, held as a separate,
explicitly stakeholder-gated decision per the backlog item's own scope note.

**Verified:** Reproduced the collision against the pre-fix code first (RED), using a throwaway
workspace (`ws:wf037check`) and a throwaway decoy def (`wf037decoy@v1`, seeded fresh via
`FALKORCHAT_PROCESS_DEF_KEY`) rather than the real, already-corrupted `access-request@v1` — setting
`FALKORCHAT_TRIGGER_DEF_KEY`/`_VERSION` to the decoy's key/version and re-running
`seed_workflows.sh` reproduced the exact reported signature: both defs logged under the same key,
decoy went from the canonical 6 `Step`s/1 `START` edge to **9 `Step`s/2 `START` edges**
(`submit`,`intake`). Applied the fix, reran the same scenario against a fresh decoy
(`wf037decoy2@v1`, then again end-to-end through `start_server.sh` itself against
`wf037decoy3@v1`) — confirmed GREEN: decoy stays at the canonical 6 `Step`s/1 `START` edge
(`submit`) after the trigger-key collision, and a real `triage@v1` is still seeded/no-op'd
unconditionally, under its own default key, unaffected by the trigger override. The
`start_server.sh` end-to-end run also confirmed the banner now prints
`Workflow:  enabled=1 (triage def wf037decoy3@v1)` — the actual configured override, not the old
hardcoded `triage@v1` literal. All throwaway test data (`ws:wf037check`, `ws:wf037full`, the
`wf037decoy*` defs) deleted afterward; the real `reference` graph's `access-request@v1` (still at
its pre-existing 9 `Step`s/2 `START` edges from before this fix — untouched, not compounded at that
point) and `triage@v1` (3 `Step`s, correct) confirmed unchanged throughout. `bash -n` clean on both
scripts. This state did not survive intact — see the same-day follow-up immediately below.

**Follow-up (same day) — confirming pytest green wiped `reference`; restored via the standard
reseed.** Running the server pytest suite afterward to confirm it was still green (it was — `641
passed`) triggered the fixture-level hazard the `seed_workflows.sh` header (and this repo's
`AGENTS.md`) already documents: the `wf_repo` fixture wipes `reference` at test setup, so the
finished run left `reference` with **zero** published `WorkflowDef`s — taking the real `triage@v1`
and the real `access-request@v1` (with it, the pre-existing 9-step/2-`START`-edge corruption noted
above) down too. `ws:acme`'s snapshots were confirmed untouched (`triage@v1`, `access-request@v1`
both still present) — the fixture only ever touches `reference`. Restored via the standard,
documented, non-destructive recovery — `./scripts/seed_workflows.sh acme` (create-only, no
override; this is the routine post-pytest reseed the script's own header prescribes, not the
separate stakeholder-gated delete+republish cleanup) — which freshly **created** both defs in
`reference`: `triage@v1` (3 `Step`s) and `access-request@v1` at the **canonical 6 `Step`s/1 `START`
edge** — clean. `./scripts/verify_workflows.sh acme` then reported `triage@v1` in sync, but flagged
`access-request@v1` as **diverging (5 differences)**: `ws:acme`'s *snapshot* still carries the three
spurious steps (`intake`/`research`/`answer`, 9 `Step`s total, 1 `START` edge) — contamination that
predates this whole K-037 session and matches the QA report's own documented "pre-flight" baseline
(`docs/test-reports/web-api-coverage-report.md`), not something Wave 5's own restart added (that
corruption was on the `reference` side, which the wipe just erased). **Net effect, worth recording
explicitly:** the separate stakeholder-gated cleanup this backlog item declined to fold in (deleting
the corrupted `access-request@v1` subgraph before republishing) appears to have **partially resolved
itself on the `reference` side** — a `reference`/`ws:acme` reseed accident, not a deliberate gated
cleanup. The `ws:acme` snapshot side remains divergent and untouched; per `verify_workflows.sh`'s
own instruction ("if they DIVERGE, do NOT re-seed... report it"), no further write was attempted.
The stakeholder-gated cleanup decision is therefore still open, now scoped narrowly to repairing
the `ws:acme` snapshot rather than both graphs.
Script-only change; `./scripts/test_queries.sh` not run (would trigger the same hazard on
`reference` again — no reason to compound it further; the confirmations above already establish
both scripts are syntactically and functionally sound).

**Follow-up 2 (same day) — `analyst` review (approve with suggestions): added the regression
guard, landed three polish fixes, and re-confirmed live state.** Review at
`docs/reviews/wf-trigger-def-key-collision.md` — verdict **approve with suggestions**, decoupling
confirmed correct/complete, doc updates accurate, nothing blocking. Addressed all four items:

- **Major — no automated regression guard for this bug class.** Added
  `server/tests/test_seed_workflows_script.py`
  (`test_trigger_def_key_override_does_not_graft_onto_the_other_def`), following the review's own
  suggested shape (the backlog item's original test-strategy note, automated): shells out to the
  real `scripts/seed_workflows.sh` (never a reimplementation) against `ws:test`, using two
  throwaway, test-only def identities (`k037-triage-guard`, `k037-target-guard`) so the real
  `triage`/`access-request` keys are never touched. `FALKORCHAT_TRIAGE_DEF_KEY`/`_VERSION` are
  always passed EXPLICITLY (belt-and-suspenders, per the review) rather than left to the script's
  default, so a regression that silently stopped reading that pair cannot hide behind matching
  defaults. Seeds once at baseline (both throwaway defs canonical), reseeds a second time with
  `FALKORCHAT_TRIGGER_DEF_KEY`/`_VERSION` pointed at the target's key/version (replaying K-037's
  Pass-B collision against throwaway data), then asserts via `services.get_workflow_def_structure`
  that neither def's structure moved. No `live` marker — network-free, part of the ordinary
  `pytest -q` baseline. Cleans up its own throwaway defs at teardown (`reference` + `ws:test`),
  confirmed via direct Cypher after the run: zero `k037-*-guard` nodes left behind.
  **Proved the test actually catches the regression it guards, not just that it passes**: temporarily
  reverted `seed_workflows.sh`'s fix (`TRIAGE_DEF_KEY` reading `FALKORCHAT_TRIGGER_DEF_KEY` again),
  reran — RED, and for the right reason (`WorkflowDefNotFoundError` on `k037-triage-guard`: the
  buggy code silently stopped honoring `FALKORCHAT_TRIAGE_DEF_KEY` at all, falling through to the
  unset `FALKORCHAT_TRIGGER_DEF_KEY` default and publishing the literal under real `triage@v1`
  instead — exactly the failure mode the explicit-pass design exists to catch). Restored the fix,
  reran — GREEN. Full suite after landing: **642 passed, 1 deselected** (641 + this new test).
- **Minor — header-comment ordering in `seed_workflows.sh`.** Reordered so the K-037 decoupling
  explanation (why the two env-var pairs are independent, only sharing *defaults*) now precedes the
  "TRIAGE def key/version MUST match the trigger config" paragraph, with an explicit "read the next
  paragraph in light of the above" bridge — a top-to-bottom reader no longer hits the
  easy-to-misread "MUST match" framing before the correct explanation.
- **Nit — `HISTORY.md`'s "Files touched" line missing backticks.** Checked: the line already carries
  backticks around every path in the current file (confirmed byte-for-byte via `cat -A`) — no change
  needed. Flagging the discrepancy with the review's specific claim rather than silently
  no-op'ing it.
- **Nit — `start_server.sh` stage 5 doesn't document `FALKORCHAT_TRIAGE_DEF_KEY`/`_VERSION`
  env-inheritance reachability.** Added a one-line note at the stage-5 seed-invocation comment.

**Re-confirmed live `reference`/`ws:acme` state after this follow-up's own pytest runs** (which
re-triggered the same `wf_repo` wipe hazard as before — expected, not new): restored again via
`./scripts/seed_workflows.sh acme`, then `./scripts/verify_workflows.sh acme` →
**`RESULT: OK — 2 defs in sync`** for both `triage@v1` and `access-request@v1`. Direct Cypher
confirms both now canonical on both sides: `reference`'s `access-request@v1` and `ws:acme`'s
snapshot both read exactly 6 `Step`s (`submit`/`route`/`approval`/`provision`/`activate`/`rejected`)
and 1 `START` edge (`submit`) — no `intake`/`research`/`answer` under an `access-request:v1:*`
`stepUid` anywhere. This is a **change from Follow-up 1's observation** (`ws:acme`'s snapshot at 9
`Step`s, diverging from `reference`'s 6) — resolved, not a mystery: while this session was mid-flight
on the review items above, `graph-dba` concurrently landed the `ws:acme`-side surgical cleanup this
same file documents immediately above, under "2026-07-30 — K-037 follow-up: surgical cleanup of
`ws:acme`'s contaminated `access-request@v1` snapshot" — a parameterized `DETACH DELETE` of the 3
spurious `Step`s, explicitly authorized for this dev environment, with its own before/after Cypher
counts. Nothing in *this* session's own commands touched `ws:acme` beyond the two additive, no-op
`seed_workflows.sh acme` reseeds (confirmed by reading `materialize_snapshot`,
`server/falkorchat/repository.py` — `MERGE … ON CREATE SET` only, no delete path) — the change is
fully accounted for by that separate, concurrent, now-documented entry. The stakeholder-gated
`access-request@v1` cleanup decision this backlog item held open is therefore **closed**: both the
`reference`-side half (incidentally resolved by the pytest wipe + reseed, this entry) and the
`ws:acme`-side half (deliberately, surgically resolved, the entry immediately above) are done.

Files touched: `falkor-chat/scripts/seed_workflows.sh`, `falkor-chat/scripts/start_server.sh`,
`falkor-chat/docs/BACKLOG.md`, `falkor-chat/server/tests/test_seed_workflows_script.py` (new).

## 2026-07-29 — K-036 doc gap-fill: missing HISTORY entries for U1 (graph-dba) and U3 (frontend-engineer), Wave 1

**What this entry is:** written today, at K-036's documentation close-out, to record work that
**actually shipped 2026-07-28** in commit `3d2234c` (the same commit as the already-dated
"2026-07-28 — K-036 U2" entry below). U1 and U3 landed in that same commit but never got their own
dated entry — flagged as a known gap at Gate 1 (`docs/reviews/web-api-coverage-impl.md`, finding
m1) and confirmed still open at every check since (the coordination ledger,
`docs/plans/web-api-coverage-coordination.md`, notes it under "Entry state" and again at Gate 1,
and never closes it). This entry closes it now, retroactively and explicitly labeled as such —
content is grounded in `git show 3d2234c --stat`, `git log -1 3d2234c`, `docs/QUERIES.md` §2/§12.14,
and the shipped `web/index.html`/`web/app.js` defs-viewer code, not invented after the fact.

**U1 (`graph-dba`) — two new read queries + one new index, backing FR-2 and FR-8.**
- **`docs/QUERIES.md` §12.14, `find_runs_for_thread`** — every `WorkflowRun` a thread has ever had
  (`MATCH (r:WorkflowRun)-[:TRIGGERED_BY]->(m:Message) WHERE r.startedAt >= 0 AND m.threadId =
  $threadId ...`), backing `GET /threads/{id}/workflow-runs` (FR-2, landed as U4 in Wave 2). Ships
  with a genuinely new, previously-undocumented FalkorDB planner fact: a `WHERE` predicate on one
  pattern variable pulls the label-scan anchor onto *that* variable's label even when a much
  smaller, filter-free label sits elsewhere in the same pattern — the plan's originally-proposed
  shape (no predicate on `r`) anchored on `Node By Label Scan | (m:Message)` (20,003 records in the
  profiled dataset) instead of the much smaller `WorkflowRun` label, and a bare `WorkflowRun.startedAt`
  index alone did **not** move the anchor. The functionally-vacuous `r.startedAt >= 0` conjunct is
  what redirects the anchor back onto `Node By Label Scan | (r:WorkflowRun)` — load-bearing for the
  query plan, not decoration. New `WorkflowRun.startedAt` range index added alongside (small
  cardinality per workspace, negligible RAM). Promoted to the general FalkorDB quirks KB
  (`claude/graph-dba/falkordb-quirks.md`, "Query tuning").
- **`docs/QUERIES.md` §2, "List thread participants"** — a thread's participants, defined as its
  parent channel's roster (`MATCH (c:Channel)-[:HAS_THREAD]->(t:Thread {threadId: $threadId})
  MATCH (u)-[:MEMBER_OF]->(c) RETURN coalesce(u.userId, u.agentId) AS memberId, u.displayName,
  labels(u) AS type ...`), backing `GET /threads/{id}/participants` (FR-8, landed as U5 in Wave 2).
  `PROFILE`-verified: anchors on `Node By Index Scan | (t:Thread)`, no label scan anywhere. Carries
  forward the pre-existing, documented gap that `Agent` nodes have no `.displayName` (comes back
  `null`; `labels(u)` still correctly reads `["Agent"]`).
- `scripts/test_queries.sh` assertions for both: suite went **256 → 276/276**. Also fixed in
  passing: `test_queries.sh`'s `assert_index_scan` helper had a wrong `PROFILE` string match that
  was silently no-op-ing the "no label scan" half of every prior assertion in the suite.

**U3 (`frontend-engineer`) — workflow-defs viewer in `web/`, backing FR-1.** A `#defs-btn` button
in the header (always visible, shares the generic `button` styling with the pre-existing header
buttons — no new visual weight) opens `#defs-panel`, a right-side overlay
(`position:absolute`/`display:none` until opened, same pattern as the existing `#results` panel)
listing every published workflow def (`loadDefsList`, `GET /workflow-defs`). Selecting a def loads
its structure (`GET /workflow-defs/{key}/versions/{version}`) and renders step/transition detail
via `renderDefDetail`. Zero DOM/fetch footprint until the button is clicked — no unconditional
call added to the page's load sequence.

**Baseline at this commit:** server pytest **614 passed**; query suite **276/276** (up from the
256/256 baseline going in). Commit: `3d2234c` (2026-07-28), same commit as U2's already-dated
entry below.

## 2026-07-29 — K-036 U10 (Wave 5): black-box acceptance pass on the Web API Coverage feature

**What:** `qa-engineer`'s black-box acceptance pass against `docs/requirements/web-api-coverage.md`
AC-1..AC-6, per `docs/plans/web-api-coverage.md` §5.2's two-pass session shape — Pass A (default
`triage`/`v1` config: AC-1, AC-4, AC-5, AC-6) → server restart (`FALKORCHAT_TRIGGER_DEF_KEY=
access-request`, `_VERSION=v1`) → Pass B (AC-2, AC-3) → restart back to the default config. No
browser-automation tool was available in this environment; the pass drove the exact REST endpoints
`web/app.js` calls (same sequence/payloads a UI session would trigger), cross-checked against a
direct reading of the corresponding render logic — recorded explicitly as a tooling-constraint
substitution in both artifacts below, not silently treated as a real browser session.

**Result: PASS with parked/non-blocking limitations.** All six ACs satisfied against the delivered
code; no defect found in K-036's own diff (Waves 1-4). Two new, non-blocking observations: (1) a
**Major** operational finding — the plan's own sanctioned Pass-B restart procedure
(`FALKORCHAT_TRIGGER_DEF_KEY=access-request`) causes `scripts/start_server.sh`'s unconditional
`seed_workflows.sh` re-seed to graft `triage`'s steps onto `access-request@v1` in the `reference`
graph (env-var name collision between the chat-trigger override and the seed script's own
triage-literal identity — same duplicate-`START`-edge symptom class as backlog K-034, but a
distinct, previously-undocumented trigger mechanism); confirmed the workspace's readiness endpoint
(FR-10/AC-6) correctly detected and named the resulting drift. Left the graph as-is (QA does not
remediate); recommend a `devops`/`graph-dba`-routed fix + cleanup. (2) A **Minor** cosmetic
finding — `start_server.sh:136`'s startup banner hardcodes "triage def triage@v1" regardless of an
active `FALKORCHAT_TRIGGER_DEF_KEY`/`_VERSION` override (the actual wiring was confirmed correct
via `/proc/<pid>/environ` and a live functional test; only the printed text is stale). Also
recorded: forcing a *chat-triggered* failing run for AC-3 isn't achievable with either seeded demo
def (parked steps are budget-exempt by design, D-C, and neither def contains a self-loop) — AC-3
was validated at the REST/rendering-contract level via a directly-started run instead, a
testability gap rather than a defect.

The pre-existing `reference`/`access-request@v1` drift disclosed ahead of this session (matching
K-034-territory) was confirmed present at pre-flight exactly as described, and used as AC-6's
naturally-occurring negative case rather than re-filed.

**Why:** Wave 5 (U10) of the plan's build sequence — the only unit not yet executed, and the gate
before K-036's milestone close-out.

**Artifacts:** `docs/test-plans/web-api-coverage.md` (v1, written before execution),
`docs/test-reports/web-api-coverage-report.md` (full findings + verdict). Server left running,
restored to the default `triage`/`v1` config (no env override), consistent with what every other
environment (pytest, future manual checks) assumes.

**Plan items:** K-036 Wave 5 (U10) done — all five waves of the build sequence now delivered.
Milestone close-out (BACKLOG.md/plan/review `Status:` flips) is `teco`'s coordination call, not
done here.

## 2026-07-29 — K-036 U6+U7+U8+U9 (Waves 3-4): run cue, run detail panel, participants toggle, readiness banner in the web UI

**What:** the frontend units from `docs/plans/web-api-coverage.md` §4 Waves 3-4, wired entirely
against endpoints already delivered in Waves 1-2 (`3d2234c` U1-U3, the U4+U5 entry directly below)
— no server-side change in this unit.

**U6 — inline run cue + run detail panel shell (FR-2/FR-3/FR-4).** The cue (`#run-cue`, above the
composer) piggybacks on the existing per-thread message poll (`startPolling`, `app.js`) — one more
`GET /threads/{id}/workflow-runs` fetch per tick, zero-footprint when the thread has no runs. Its
"most relevant run" tie-break is `web/run-select.js`'s `selectMostRelevantRun`/
`isTerminalRunStatus` — extracted as a dependency-free pure function per §5.2/review finding m3,
covered by 12 plain-assertion cases in `web/tests/run-select.test.js` (`node
web/tests/run-select.test.js` — 12/12 passing). The run detail panel (`#run-panel`, opened from
the cue's "View" button) polls `GET /workflow-runs/{id}` + `.../step-runs` on its own start/stop
lifecycle (started on open, stopped on close) and renders status, timestamps, and the step-run
list; a "Show trace" toggle lazily fetches `.../trace` only when opened.

**U7 — thread participants toggle (FR-8/AC-4).** `#participants-toggle` next to
`#thread-heading` is always present (disabled until a thread is open, per §3.3's AC-5 reading);
expanding fetches `GET /threads/{id}/participants` and renders a chip row, reusing the existing
`.badge`/`--agent` kind token for `kind === "Agent"`. Collapses on thread switch. **Fixed a latent
CSS scoping bug found while wiring this in:** the `.badge` "AI" pill rule was scoped `.msg .badge`
(only styled inside a chat message); broadened to `.msg .badge, .chip .badge` so the reused token
actually renders inside a participant chip too (`web/index.html`).

**U8 — ready-to-demo banner (FR-10/AC-6).** `#readiness-badge` next to the header `tenant` span
fetches `GET /workspaces/{ws}/readiness` once on load (`DEMO_WS = "acme"` — the `ws` path segment
is descriptive-only per `api.py`'s own comments at that route, tenancy comes from `get_context`;
kept in sync with the demo default). Clicking it toggles a small panel listing `problems` per
offending def; a "Recheck" button re-fetches.

**U9 — waiting-step prompt, structured-input form, failure display (FR-5/FR-6/FR-7).** When a run
is `waiting`, the panel fetches the run's snapshot (`GET
/workspaces/{ws}/snapshots/{defKey}/versions/{defVersion}`) to resolve the parked step's `config`
(an opaque JSON string, parsed client-side) and renders `config.prompt` plus a form built from
`config.fields` (one text input per top-level key; a `config.expects[field]` list renders as a
`<select>`). Submitting `POST`s to `/workflow-runs/{id}/input`; on success the panel **immediately
re-polls** (`refreshRunPanel()`, mirroring the composer's existing `postMessage()` →
`pollMessages()` idiom — plan §3.3.3/review m2) instead of waiting for the next tick; a rejected
submit (400 `WorkflowInputRejectedError`) surfaces via the existing `showError` toast and leaves
the form untouched. When a run is `failed`, `JSON.parse(run.ctx).error` renders as the failure
reason.

**Also fixed:** `#run-panel` was added to `index.html`'s CSS in an earlier session but never
joined the shared `#results, #defs-panel { position:absolute; ...; display:none; }` overlay rule
— it had a width override but no default hidden/positioned state. Folded into the same shared
selector.

**Manual verification (live server, no automated JS test harness — §5.2's own call, since the
front end is thin pass-through UI with no business rules of its own):** started FalkorDB (already
running, `falkordb-dev`) + `./scripts/start_server.sh` with `FALKORCHAT_TRIGGER_DEF_KEY=access-request`,
`FALKORCHAT_TRIGGER_DEF_VERSION=v1` (the plan §7 risk #1 workaround, so `@mention` reaches a
structured-input step rather than `triage`'s plain-chat-reply park) against the seeded `ws:acme`.
Drove every new endpoint with the exact requests the browser code issues (no headless-browser
driver was available in this sandbox — no `chromium`/`playwright` binary and no root to install
one — so this is API-level, not click-level, verification; a follow-on `qa-engineer` black-box
pass, U10, is the plan's own separate Wave-5 unit and was not attempted here):
  - `GET /threads/demo-welcome/participants` → both `Agent` and `User` members, correctly
    `kind`-labeled (U7's exact render inputs);
  - `@assistant …` mention → `GET /threads/demo-welcome/workflow-runs` showed the run within one
    poll tick, `status: "waiting"`, `atStepKey: "submit"` (U6's cue/panel render inputs);
  - `GET /workspaces/acme/snapshots/access-request/versions/v1` confirmed the `submit` step's
    `config` (`prompt`, `fields: ["request"]`) matches what `renderWaitingForm` parses;
  - `POST /workflow-runs/{id}/input` with an undeclared field → `400
    {"error":"WorkflowInputRejectedError", ...}` (U9's toast path); with the declared field →
    `200`, run advanced `submit → route → provision` (still `waiting`, new `atStepKey`);
  - submitted the `provision` step's signal → run reached terminal `status: "done"`;
  - started a second run with `maxSteps: 1` to force budget exhaustion → `status: "failed"`,
    `ctx.error: "step budget exceeded"` (U9's exact FR-7 parse target);
  - `GET /workspaces/acme/readiness` → **surfaced a genuine, pre-existing `ready: false` state**
    in this dev environment (the `reference` `access-request@v1` def has drifted to 2 `START`
    edges and diverges from the `ws:acme` snapshot — unrelated to this change, not touched here;
    flagged for a separate K-034-territory cleanup, not fixed as part of this UI unit) — a
    convenient real fixture for U8's "not ready, names the offending def" rendering path, which
    rendered correctly;
  - static file check: `index.html`/`app.js`/`run-select.js` all served `200` with every new
    element id present in the rendered page; server log showed zero tracebacks across the whole
    session.

**Scope note:** U10 (qa-engineer black-box AC-1..AC-6 pass, two-pass session per §5.2) is
unstarted — the plan's own separate Wave-5 unit, not part of this frontend delivery.

**Addendum (same day) — fixed a poll-driven form-wipe defect (`analyst` review Pass 2, finding
M1) before it reached U10.** `refreshRunPanel()`'s `POLL_MS`-interval tick unconditionally called
`renderWaitingForm()`, which unconditionally did `box.innerHTML = ...` on `#run-waiting-form` —
destroying and rebuilding the live `<form id="run-input-form">` on every tick while the panel was
open and the run stayed parked, silently wiping any text typed but not yet submitted (a real risk
on `access-request@v1`'s free-text `request` field, which plausibly takes longer than the 3s poll
interval to compose). Fixed in `app.js` by tracking the last-rendered `(runId, atStepKey)` as
`state.runWaitingKey` and skipping the rebuild in `renderWaitingForm` when the parked step is
unchanged since the last render; `submitRunInput` clears `runWaitingKey` on a successful submit so
the very next `refreshRunPanel()` always re-renders, keeping the plan §3.3.3 "submit reflects new
state immediately" behavior intact. As a side effect this also resolved review finding m4 (the
waiting-step snapshot was being re-fetched on every tick) for free, since the fetch now only
happens on an actual render. m5 (readiness panel can briefly open empty before the first
`loadReadiness()` response lands) was left as-is — cosmetic, not worth the added complexity.

Verified by extracting the exact, unmodified `refreshRunPanel`/`renderRunPanel`/`renderWaitingForm`/
`submitRunInput` source into a small DOM-stub harness and driving it against a live
`./scripts/start_server.sh` instance (workflow engine + `access-request@v1` parked on `submit`):
across one open + 3 simulated poll ticks with the step unchanged, `run-waiting-form`'s `innerHTML`
was set exactly once and the snapshot endpoint fetched exactly once, and a field value set
mid-"typing" survived unchanged across all 3 ticks; a real successful submit still advanced the run
and re-rendered the panel immediately. Re-running the identical harness against the pre-fix source
reproduced the defect exactly (rebuilt + re-fetched + wiped the typed value on the very first
tick), confirming the harness actually detects the bug rather than passing vacuously. Regression
gates: `node --check web/app.js` clean, `web/tests/run-select.test.js` still 12/12 (untouched by
this fix), and the server suite unaffected (`.venv/bin/python -m pytest -q` → 641 passed, 1
deselected — identical to the review's baseline, no server-side files touched).

## 2026-07-29 — K-036 U4+U5 (Wave 2): thread workflow-run history + thread participants over HTTP

**What:** the two Wave-2 backend units from `docs/plans/web-api-coverage.md` §4, landed together
(same files) — both wire repository/service/API layers on top of the two queries `graph-dba`
authored and `GRAPH.PROFILE`-verified in Wave 1 (U1, `3d2234c`: `docs/QUERIES.md` §12.14
`find_runs_for_thread`, and the new §2 "List thread participants"). No new Cypher.

**U4 — FR-2, the inline run cue's data source.** `repository.find_runs_for_thread(ws, *,
thread_id, limit=10)` is a 1:1 wrapper over §12.14 (carries forward the query's load-bearing
`WHERE r.startedAt >= 0` predicate — the anchor-trap fix U1 found — verbatim).
`services.list_workflow_runs_for_thread(ctx, *, thread_id, limit=10)` validates `thread_exists`
first and raises `ThreadNotFoundError`, reusing the exact guard idiom
`_validate_and_derive_role` already uses rather than inventing a second one. New route
`GET /threads/{thread_id}/workflow-runs?limit=` (1-50, default 10) → 200 list (possibly empty,
newest-first by `startedAt`), 404 via the generic `ServiceError` handler when the thread doesn't
exist. No `response_model` — matches the surface's convention (only the three K-031
structure/diff routes declare one).

**U5 — FR-8, the participants list.** `repository.list_thread_participants(ws, *, thread_id)` is
a 1:1 wrapper over the new §2 query and returns the query's raw columns
(`memberId`/`displayName`/`type`, `type` = `labels(u)` — mirrors `read_thread`'s `authorType`
convention rather than resolving the label in Cypher). `services.list_thread_participants(ctx,
*, thread_id)` applies the same `thread_exists`/`ThreadNotFoundError` guard as U4, then
normalizes `type[0]` down to a single `kind` string (`"User"`/`"Agent"`) — the same derivation
`resolve_member_kinds` already does in Cypher for its own case, done here in Python since this
read's repository method stays a literal query mirror. New route
`GET /threads/{thread_id}/participants` → 200 `[{"memberId", "displayName", "kind"}]`
(empty list, not an error, when the thread's channel has no members — the documented
demo-seed-timing edge case), 404 for an unknown thread.

**Tests:** `server/tests/test_repository.py` — 10 new integration cases against `ws:test` (5 per
method: empty/populated/ordering/limit/other-thread-excluded for `find_runs_for_thread`;
both-kinds/only-human/only-agent/no-members/unknown-thread for `list_thread_participants` — the
latter needed a test-only raw-Cypher `_add_to_channel` helper since no repository method for
`MEMBER_OF` writes exists yet, same gap `seed_demo.sh` fills with raw Cypher today).
`server/tests/test_services.py` — 7 new unit cases against `FakeRepo` (passthrough, missing-thread
error, empty, limit-passthrough for U4; kind-normalization, missing-thread error, empty for U5).
`server/tests/test_api.py` — 10 new `TestClient` contract cases (empty/populated-newest-first/limit
query-param/limit-bounds-422/404-unknown for the runs route; both-kinds/only-human/only-agent/
empty-no-members/404-unknown for the participants route). Full suite: **641 passed, 1 deselected**
(614 baseline + 27 new). `./scripts/test_queries.sh` unaffected (no Cypher/schema change) — still
**276/276**; re-ran `seed_workflows.sh acme` afterward per the documented pytest/`test_queries.sh`
`reference`-graph-wipe interaction (DESIGN §14.7), confirmed back in sync with
`verify_workflows.sh`.

**Scope note:** `web/` (frontend wiring, U6/U7) is explicitly out of scope for this change —
next wave.

## 2026-07-28 — K-036 U2: `GET /workspaces/{ws}/readiness` — "is this workspace ready to demo" over HTTP

**What:** the HTTP form of `scripts/verify_workflows.sh` (FR-10, `docs/plans/web-api-coverage.md`
§3.1c, Wave-1 unit U2 — independent of the plan's other units). New
`services.check_demo_readiness(ctx) -> dict` composes three already-existing service methods —
`diff_def_snapshot`, `get_workflow_def_structure`, `get_snapshot_structure` — over
`DEMO_EXPECTED_DEFS` (`(config.TRIGGER_DEF_KEY, config.TRIGGER_DEF_VERSION)` +
`(proof_defs.ACCESS_REQUEST_DEF["key"/"version"])`, the same pair `seed_workflows.sh` seeds and
`verify_workflows.sh` already checked). **Zero new Cypher, zero repository change.**

Per def: presence + sync from `diff_def_snapshot` (a cold `reference`/`ws:{id}` graph key or an
absent def reads as "nothing there", not a 500 — `_read_or_absent` mirrors the script's own
`read()` helper, including its `WorkflowDefNotFoundError`/`ResponseError("empty key")` catch and
`ABSENT`-shape substitution byte-for-byte), plus the Finding-3 multi-`START` tripwire
(`"startKeys" in structure`, K-034) from the two structure reads. `problems` reuses the script's
exact wording (`"{label}: not published in \`reference\` at this version"`, `"... not
materialized into ws:{ws} at this version"`, `"... diverge (N differences)"`, `"... has N START
edges (...) — see K-034"`) so the endpoint and the script can never disagree about what "ready"
means. `ready` = every def fully present, in sync, and problem-free.

New route `GET /workspaces/{ws}/readiness` → **always 200** (readiness is a report, never a
404/error condition), mirroring `list_snapshots`'/`diff_def_snapshot`'s `ws`-path/tenancy
convention (`ws` is descriptive; tenancy comes from `get_context`).

**Bundled cleanup (plan §3.1c, "should-do"):** `verify_workflows.sh` now imports
`DEMO_EXPECTED_DEFS` from `falkorchat.services` instead of declaring its own inline `DEFS` list —
the two lists would otherwise be a second, silent copy of exactly the kind of drift this feature
exists to catch. Re-verified against a live server after the change: still `RESULT: OK — 2 defs
in sync between \`reference\` and ws:acme`, unchanged in behavior (the pytest run in between, as
documented in DESIGN §14.7, wiped `reference` mid-verification — re-seeding with
`seed_workflows.sh acme` was needed before the final check, not a regression).

**Tests:** `server/tests/test_services.py` — six new cases against a fake repo (all-present/synced,
missing-def, missing-snapshot, both-absent via the `WorkflowDefNotFoundError` catch, diverging,
and the multi-`START` tripwire), all naming the exact offending def/problem string.
`server/tests/test_api.py` — two contract tests against the live `ws:test`/`reference` graphs (200
shape with nothing seeded; `ready: true` once both demo defs are published + materialized under
their real keys). Full suite: `614 passed, 1 deselected`.

## 2026-07-27 — Docs: the repo-wide reference & naming convention applies here — **forward-only**

**What:** the documentation reference and naming convention landed repo-wide
(`docs/plans/doc-reference-convention.md` v1.4, twice reviewed; full entry in `docs/HISTORY.md`
under 2026-07-27). Two consequences for this component, both **forward-only**:

- **A document that freezes no longer moves.** `falkor-chat/docs/archive/` is now **read-only history
  of the previous convention, not a destination** — nothing moves into it again and nothing is
  un-archived. A document that freezes gets `Status: archived` in its own header block and stays
  exactly where it is, which also removes the inbound-link repair the move used to require. The
  `falkor-chat/docs/BACKLOG.md` preamble and the `docs/archive/` row of `falkor-chat/AGENTS.md` were
  corrected to say so. **The dated entries below that describe the old rule are left exactly as
  written** — they were correct when written.
- **Every active plan, review and requirements document here now opens with the canonical header
  block** (`> **Status:** … · **Owner:** … · **Tracks:** …`), so
  `grep -m1 -H 'Status:' falkor-chat/docs/plans/*.md` is a complete lifecycle listing. `Tracks:` is
  where the milestone lives now.

**The naming convention is forward-only, and renames were explicitly declined.** The grammar
`<component>/docs/<kind>/<topic-slug>[-<role>].md` — with its closed role set and its prohibition on
an `m<digit>`/`k<digit>`/date **prefix** — governs **new** documents only. This component's existing
`m1-`/`m2-`/`m3-` names stay. **An existing `m<n>-` prefix is part of a name, not a lifecycle
claim:** nobody should read meaning into it, and nobody should "fix" it.

**Why the rename was declined — recorded so it is not re-litigated.** Renaming the 6 mis-named
documents was measured at **39 occurrences across 15 files** of inbound-citation repair, against
**22 edits across 8 files** for the *entire* M3-close archival sweep that triggered the assessment.
The tidy-up costs nearly twice the whole problem it would tidy, and it buys nothing a reader cannot
already get from the header's `Tracks:` field, from `falkor-chat/docs/BACKLOG.md`, or from this
file — the three maintained places the milestone already lives.

**Also in the same change:** the three broken relative links in `falkor-chat/docs/BACKLOG.md` (an
extra `../` in each — every target existed) are fixed, and the backlog's parking lot records one
opportunistic re-slug nit, deliberately unscheduled.

## 2026-07-24 — Docs: `AGENTS.md` de-bloat — moved restated content to its canonical home, kept only pointers

**What:** `AGENTS.md` had grown to 312 lines, much of it near-verbatim restatement of content that
already lived (often in more detail) in `docs/DESIGN.md`, `docs/QUERIES.md`, or the `server/`
code's own docstrings. Trimmed to 129 lines by replacing four bloated sections with short pointers:
"Decisions locked in" (the 19-row table duplicated `DESIGN.md` §1.2, which is already the stated
authoritative register), "Live-verified FalkorDB facts" (project corollaries already annotated
inline in `QUERIES.md` §2/§4/§9.1 and `DESIGN.md` §10), "Message write paths" (already in
`QUERIES.md` §4 and `services.py`/`repository.py` docstrings), and "M1 server" (architecture
already in `DESIGN.md` §14; K-027 parse-tolerance and executor/workflow-def invariants already in
`llm.py`/`services.py`/`executor.py` docstrings, near-verbatim).

**Genuinely new content surfaced during the audit, given a real home instead of staying in
`AGENTS.md`:**
- `DESIGN.md` §5.3 — the dispatch-loop's 4-attempt tripwire bound and the service's lock-guarded
  monotonic per-process clock for `createdAt` (previously undocumented outside code).
- `DESIGN.md` §6.2 — `_drive_loop`'s SHA-lock (`71055f756280`) and which functions sit outside it.
- `DESIGN.md` §6.1 — the unenforced residual: a `decision` step with all-conditional transitions
  and no `waitsForHuman` self-loops to budget exhaustion (K-029).
- `DESIGN.md` §14.7 (new) — the four `server/` pytest-specific hazards (destructive `reference`-graph
  wipe via the `wf_repo` fixture's *setup-time* `DETACH DELETE`, skip-count-hides-integration-suite,
  `ruff` not being a wired gate, `ws:test`'s fixed dim-4 vector index).
- `claude/graph-dba/falkordb-quirks.md` — two facts generalized from project-specific corollaries:
  an empty-`UNWIND` guard can silently drop an unrelated *required* write downstream, not just the
  guarded list's own edges; and a composite keyset-pagination predicate over one indexed column
  still plans as a bare index scan with no residual filter.
- `claude/coder/kaizen/inbox.md` + `claude/tdd-engineer/kaizen/inbox.md` — two testing-hygiene
  lessons distilled from the same pytest hazards (read skip counts, not just exit code; a fixture
  that wipes shared/global state at setup-not-teardown leaves the last test's leftovers behind),
  routed to the agents' inboxes per the `agent-maintenance` distillation convention rather than
  into their blueprints directly.

Confirmed via `claude/graph-dba/falkordb-quirks.md`'s own stated convention (generic engine facts
belong there; project-specific corollaries stay in the project, pointing back) that most of the
"Live-verified FalkorDB facts" bullets were correctly project-scoped already — the fix was
de-duplicating against `QUERIES.md`/`DESIGN.md`, not relocating them into the agent's knowledge base.

## 2026-07-24 — K-031: def/snapshot **structure** read surface — the create-only split-brain is now detectable

**What:** Three read-only REST routes plus a read-only script, turning the component's most
dangerous documented trap from *documented* into *checkable*. Built from plan
`docs/plans/workflow-def-structure-read.md` **v2** (analyst re-gate: *approve with suggestions*, all
15 round-1 findings closed) → `docs/reviews/workflow-def-structure-read.md`.

| Route | Answers |
|---|---|
| `GET /workflow-defs/{key}/versions/{version}` | *Is what I think is published actually published?* — full structure: `startKey`, steps (`key`/`type`/`config`), transitions (`from`/`to`/`on`/`order`/`guard`) |
| `GET /workspaces/{ws}/snapshots/{key}/versions/{version}` | *Is the workspace running the same thing?* — identical shape, `source: "workspace"` |
| `GET /workspaces/{ws}/snapshots/{key}/versions/{version}/diff` | *Have `reference` and `ws:{id}` gone stale independently?* — one call, `inSync` + an enumerated difference list |

Plus **`scripts/verify_workflows.sh <wsId>`** — the same three checks for both seeded defs at their
**expected** versions (from `config.TRIGGER_DEF_KEY`/`_VERSION` and `proof_defs.ACCESS_REQUEST_DEF`,
the sources `seed_workflows.sh` publishes from, so it cannot drift from the seed), exit 0/1, driving
the **service layer** via a Python one-shot so it works with **no uvicorn running** — which is
exactly when it is most needed, right after a `pytest`/`test_queries.sh` run. Strictly read-only: it
publishes, materializes and deletes nothing, by design and by its own header contract.

**Zero new or modified Cypher.** `repository._read_structure` reuses the existing
`_READ_META_CYPHER`/`_READ_TRANSITIONS_CYPHER` constants — both already `{label}`-templated and
already formatted with **both** `WorkflowDef` and `WorkflowDefSnapshot` — and reads *more rows of
the same result set*. No DDL, no index, no property, no `graph-dba` gate. The plan's tripwire held:
`scripts/test_queries.sh` **256/256, unchanged**. `_read_subgraph`, `read_def_subgraph`,
`services.get_snapshot`, `_PUBLISH_CYPHER` and `executor.py` are byte-identical — they are on the
materialize and SHA-locked executor paths.

**Layering:** Cypher in `repository.py` (two new readers over the existing constants),
canonical ordering + the comparator in `services.py` (`_canonical_structure`, `_diff_structures`,
`get_workflow_def_structure`, `get_snapshot_structure`, `diff_def_snapshot`), HTTP mapping in
`api.py`. Steps sort by `key`, transitions by `(from, order, to, on)`, `startKeys`
lexicographically — the graph returns both unordered by design (F6), and an unsorted list would make
`startKey` nondeterministic between two calls and report false divergences on list order alone.
Transition **identity is the 4-tuple `(from, to, on, order)`**, taken from `_PUBLISH_CYPHER`'s actual
`MERGE` key rather than guessed: a client keying on `(from, to)` would mis-report an added parallel
edge as a modified one, which is the strongest argument for the diff being server-side.
`config`/`guard` are compared and returned **byte-verbatim** (rule 8) — a diff that round-tripped
JSON would hide a whitespace-only divergence. Diff values are **previews, not payloads**
(`MAX_DIFF_PREVIEW = 200`), so the response is O(differences), never O(def).

**One side missing is a 200, not a 404** — `defPresent: false, snapshotPresent: true` is *the*
documented state after a suite wipes `reference` while `ws:{id}` survives; erroring there would push
the operator straight back to raw Cypher. Both sides missing → 404.

**V-1 — live verification, run before any code was written.** A **write** probe in a throwaway
`ws:k031probe` (bootstrapped, then `GRAPH.DELETE`d; never `reference`, `ws:acme` or `ws:test`): two
`materialize_snapshot` calls differing only in `start_key`. **Result, verbatim: 2 `START` edges and
2 meta rows, start keys `{a, b}`, each row carrying the full `steps` collection** — exactly the
plan's assumption, no escalation. So QUERIES §11.2's one-row collapse is **conditional**:
`start.key` is a non-aggregated grouping key beside `collect(DISTINCT …)`, and with two `START`
edges `_read_subgraph`'s `result_set[0]` picks an **arbitrary** start key. The new
`_read_structure` reads **all** rows and surfaces `startKeys`; `verify_workflows.sh` treats a
`startKeys` list as a failure. Recorded in QUERIES §11.2 (mirrored at §11.5).
**Residual closed at the implementation gate** (`docs/reviews/k031-structure-read-impl.md` **M-1**):
the multi-row branch was initially left unpinned because a two-`START` fixture built from two
`materialize_snapshot` calls would assert publish-structure semantics **K-034** owns. The gate found
the third option — `_read_structure` is a `@staticmethod` over an **injected** `graph`, so
`test_read_structure_unions_multi_start_meta_rows_pinning_v1s_live_shape` (plus a null-`startKey`
sibling) replays V-1's exact two-row `result_set` through a ~15-line `_FakeGraph` with **no publish,
no Cypher, no FalkorDB** and therefore no coupling to K-034. Verified to fail under the regression it
guards (`meta.result_set` → `result_set[:1]` ⇒ `start_keys == ['a']`).

**Live state (plan R-1) — no divergence found, and nothing was repaired.**
`./scripts/verify_workflows.sh acme` reports both `triage@v1` and `access-request@v1` **present on
both sides, `inSync: YES`, one start key each**. The script also *caught* the documented trap in
passing: run before the post-suite re-seed it correctly reported `reference def: MISSING` for both
defs while the `ws:acme` snapshots survived, and exited 1.

**`maxSteps` off-by-one — DOCUMENTED, not fixed** (binding stakeholder decision OQ-1).
`executor.py:410`/`:427` untouched, the `_drive_loop` SHA lock `71055f756280` intact,
`tests/test_executor.py:158` keeps its assertion. The real semantics — *a runaway tripwire checked
**after** each recorded step, so a run executes at most `maxSteps + 1` steps; checked only on
OUTCOME A (`:410`) and OUTCOME C (`:427`), deliberately **not** on the park path (OUTCOME B) or the
terminal path* — now land at six sites: DESIGN §6, QUERIES §12.5 + the two `$maxSteps` comments,
`schemas.py`, and `AGENTS.md`'s executor-invariants block. The fix (`>` → `>=`, both inside the
lock) is filed as **K-033**, self-standing, with the "bundle it with K-027 item 2" argument recorded
as an explicit *preference* whose premise is **unverified**.

**K-034 is cross-referenced, not absorbed.** This surface is the **detection** mechanism for the
additive-`MERGE` finding (a re-publish is create-only on *properties* but additive on *structure*);
K-031 deliberately does not test it, and deliberately does not correct the **thirteen** shipped
"immutable/no-op" assertions it falsifies (ten in K-034's original table; three more — the two
`requirements/` docs and K-032's own premise — folded into that table at the implementation gate, so
K-034's done-condition covers them). Both are K-034's. K-034 was filed independently during
this run and carries the analyst's evidence.

**Docs:** DESIGN §14.4 (the §11/§12 exclusion parenthetical extended to name the three routes, plus
the four operator-facing facts and the *receipt counts **submitted**, structure read counts
**stored*** sentence) and §6 (`maxSteps`); QUERIES §11.2 + §11.5 (the conditional one-row collapse,
V-1's result, and the deliberate `start_keys`-vs-`startKey` shape divergence) and §12.5 + the two
`$maxSteps` comments; `AGENTS.md` (a new `verify_workflows.sh` script row, detection pointers added
to the `seed_workflows.sh` and `test_queries.sh` rows — the create-only *wording* left alone for
K-034 — and the `maxSteps` executor invariant); `BACKLOG.md` (K-031 ✅ delivered, **K-033** filed,
the parking-lot response-schema entry annotated with the new mixed convention). `README.md`
enumerates no endpoints (verified by grep) — no change.

**Verified:** `pytest -q` → **596 passed, 1 deselected** (⇒ FalkorDB was up and the integration half
really ran). Entry baseline **552 collected / 1 deselected**, measured non-mutatingly at start;
K-031 contributes **+33** tests (3 repository, 20 service including a 10-case parametrized diff
table, 10 API contract). The gap between 552 + 33 and 596 is the **concurrent K-027 gate pass's +11**,
which landed after this run's baseline measurement — not K-031's. `scripts/test_queries.sh`
**256/256** (the no-new-Cypher tripwire, unchanged). `ruff check .` → `All checks passed!`.
`./scripts/seed_workflows.sh acme` re-run after both suites, then `./scripts/verify_workflows.sh acme`
→ exit **0**. **RAM (rule 6): zero impact** — no new node type, label, property, index or vector
dimension; both structure reads are `GRAPH.RO_QUERY` on unchanged, already-PROFILEd
index-anchored query text, so no re-PROFILE was required.

**Gate + closing pass (2026-07-24).** `analyst` verdict **APPROVE WITH SUGGESTIONS** — 0 blocker ·
1 major · 3 minor · 4 nit (`docs/reviews/k031-structure-read-impl.md`); K-031 **accepted**. Scope
discipline confirmed by execution, not inference: `executor.py` literal zero diff, no hunks on
`_PUBLISH_CYPHER`/`_READ_*`/`_read_subgraph`/`materialize_snapshot`, every new read bottoming out in
the **server-enforced** `GRAPH.RO_QUERY` write barrier. Closed in the pass: **M-1** (the fake-graph
multi-row test above); **m-1** — the `maxSteps` note claimed at `schemas.py` in three places was
never written, and is now at `schemas.py:201` (`StartWorkflowRunIn.maxSteps`, the caller-facing
field), making the "six sites" claims true; **m-2** — the R-3 exact-key-set anti-drift assertion now
covers `/diff` too (`_DIFF_KEYS` on the envelope, `_DIFF_ENTRY_KEYS` on an entry), closing the hole
where `response_model`'s field *filtering* let a dropped `WorkflowDiffOut` field pass all ten API
tests. Nits: **1 accepted** (`WorkflowDefStructureOut`'s docstring now states that `exclude_unset`
propagates into nested models, so both nested models keeping every field required is deliberate);
**3 accepted as a doc clause** (DESIGN §14.4 now says the def/snapshot shape parity is the **200 body
only** — the two routes' 404 shapes differ by plan mandate, §3.2); **2 and 4 declined** —
`verify_workflows.sh`'s double read is 4 extra RO queries at n=2 defs and the alternative
(surfacing `startKeys` on the diff envelope) is an open design question the gate routed to the
coordinator, and `_diff_preview`'s comma join is ambiguous only for a step key containing a comma,
with the full value one structure read away. **m-3 needed no action** (self-healing via K-034's own
grep done-condition). Re-verified: targeted `pytest tests/test_repository.py tests/test_services.py
tests/test_api.py -q` → **219 passed** (217 + 2), `ruff check .` clean, and — because that run wipes
`reference` — `./scripts/seed_workflows.sh acme` then `./scripts/verify_workflows.sh acme` → exit
**0**, `reference` back to **11 nodes**, both defs `inSync: YES`.

## 2026-07-24 — K-027 slice A: parse-layer tolerance for the shapes small local models emit

**What:** Two parse layers widened, both offline, no engine or Cypher change.

1. **Bare function-call syntax in `content`** (K-027 addendum **(a)**, DEF-K027-B).
   `llm._parse_content_tool_calls` recovered only JSON shapes, so the live `intake` step output
   `post_message({"text": "I'm sorry to hear that you're experiencing a broken deploy…"})` was
   lost as prose — the clarifying question never reached the thread while the run parked looking
   healthy. New `llm._parse_bare_call_syntax` recovers `name({json})` / `name()` as a **second**
   probe, after the JSON probe (precedence in `_parse_chat_message` unchanged: structured
   `message.tool_calls` stays authoritative, content probing stays the fallback).
2. **Fence-tolerant guard-judge parse** (K-027 item **1**). `app._build_llm_judge` used a
   bare `json.loads`, so a ```` ```json ````-fenced verdict broke *every* judgment silently — the
   D13 probe scored Ministral **26/26 "unparseable judge output"**, one of them a correct
   `decision:true`. It now parses with `llm.extract_own_line_json_object(…, require_key="decision")`.
   Both function-local `import json as _json` copies in `app.py` are gone, hoisted to a module
   import (gate nit **n-1**, now in full; **n-3** closed).

**Recognition rule — what the code actually enforces** (the analyst gate found the first draft's
rule materially wider than its docstring, with three reachable false positives). A bare call is
recovered only when **all three** hold: (1) the identifier opens a line — leading indentation is
allowed, so an *indented* call is recovered, but a markdown list marker is not; (2) its argument is
empty or a single JSON **object**; and (3) **from the first accepted call onward the fence-stripped
message holds nothing but calls and whitespace** — no prose *between* two calls, and nothing at all
after the last one. Rule 3 is the gate **M-1** fix, widened by gate **N-1**, and is what makes "the
expression owns its lines" true rather than aspirational — without it a call the model was merely
*quoting* (```` ```python\npost_message({…})\n```\nInstead, ask first. ````) fired, dispatched a real
thread write, and — because a recovered call carries `text=None` — threw the model's actual answer
away. Anchoring only the *last* call (the M-1 form) was not enough: `executor._run_agent_node`
dispatches **every** returned call, so *"I considered handing off:\n```\nhuman_handoff()\n```\nBut
instead I will ask:\npost_message({…})"* dispatched **both** — the whole M-1 false-positive family
re-opened whenever the model ended with a genuine call. Multi-line JSON arguments, labelled and
unlabelled code fences, a space before the paren, and prose on lines *before the first* call are
still recovered. Deliberately **rejected as text**: an inline mid-sentence call, trailing prose
on the call's line **or on any later line**, prose *between* two call-shaped expressions,
keyword/positional arguments (`post_message(text="hi")`,
`post_message("hi")`, `post_message(...)`), unbalanced parens, and prose that merely names a tool.
Identical repeated calls collapse to one dispatch (gate **m-5**); distinct calls are all returned.
**Residuals, accepted** (position alone cannot separate either from an intended call, and both are
pinned as characterisation tests): a message whose final line happens to be an illustrative call
still fires, and a *contiguous catalogue* of own-line calls with no prose between them still fires.
Closing them needs the granted tool names as a recognition filter — K-027 open question 4.
Name-against-granted-set and arg-schema validation stay in the agent loop
(`executor._handle_tool_call`), unchanged.

**Two parse seams, not one — and the judge takes the conservative one** (gate **B-1**). The first
draft pointed the judge at the permissive `extract_json_object` and asserted in three places that
"tolerance runs in the safe direction only". **That claim was false.** `extract_json_object` is
order-blind: given *"If the user had named the service I would answer `{"decision": true,
"rationale": "named"}` but they did not, so I answer false"* it lifted the **quoted** verdict and
**advanced** a guard that the previous bare `json.loads` correctly suspended — a strict regression in
the safety-critical direction, on the served guard, in exactly the metric K-027 item 3 exists to
gate. `guards._coerce_verdict` cannot catch it: the quoted rationale reads perfectly clean, so
`_rationale_contradicts` finds no cue, and the judge's real conclusion is discarded before the
coercion ever runs. Fixed by splitting the seam. `extract_json_object` stays permissive and keeps
the **tool-call** path, where the agent loop re-validates name and schema. The new
`extract_own_line_json_object` is **conservative** — it accepts a reply that is entirely one JSON
object (bare or fenced), or **exactly one** `decision`-carrying object that *owns its lines*, and
returns None on everything else: an object embedded mid-sentence, two candidates that disagree, or
nothing that parses. The judge uses it, so an unasserted verdict resolves to `decision: False`.
Tolerance is now applied only to how a verdict is **wrapped**, never to whether one was
**asserted**. Residual, accepted: a hypothetical or schema-echo verdict written on its own line with
no second object to disambiguate it is still read and still **advances** — closing that needs the
model's intent, not a parser. It is now pinned as a characterisation test and named in
`app._build_llm_judge`'s docstring (gate **N-2**), so the boundary is deliberate and visible; if
K-027 item 3's calibration counts this shape in the false-advance rate, that pin is what changes.

**Why:** Both are the same structural defect — a parse layer intolerant of the shapes small local
models actually emit — and `docs/BACKLOG.md` K-027 is explicit that this cheap, offline mitigation
lands **before** the engine-level terminal-node contract (item 2), which is then re-measured.

**Behaviour changes a caller could notice — three, not two** (the third was undisclosed in the
first draft; gate **m-1**):

1. A judge reply that is JSON but not an object (e.g. `[1,2,3]`) now yields the rationale
   `"unparseable judge output"` instead of `"non-object judge output"` — same `decision: False`,
   different trace string in the `guard_judgment` payload.
2. Content that used to come back as `text` with no tool call may now come back as a `ToolCall`
   with `text=None` (the existing embedded-JSON contract, now reached by more inputs), which means
   an agent node keeps iterating instead of terminating on that turn.
3. A `{"tool_calls": [], …}` envelope no longer short-circuits. The `if calls:` guard replaced an
   unconditional `return`, so `{"tool_calls": [], "name": "graphrag_retrieve", "arguments": {…}}`
   now falls through to the sibling name/arguments keys and yields a call where it used to yield
   text. `{"tool_calls": []}` **alone** still yields text. Both directions are now pinned.

**Second gate pass (2026-07-24) — the M-1 rule was only half a rule (gate N-1).** The re-gate closed
the blocker and 12 of 13 dispositions, and re-opened one major. "The **last** accepted call must be
the final non-whitespace content" constrained nothing *between* accepted calls, while
`executor._run_agent_node` dispatches **every** element of `result.tool_calls` — so *"I considered
handing off:\n```\nhuman_handoff()\n```\nBut instead I will ask:\npost_message({…})"* dispatched
**both**, and the entire M-1 false-positive family re-opened whenever the model ended its turn with
a genuine call, which is the *common* shape for a well-behaved turn. An echoed user snippet followed
by a real call meant **two** thread writes, one of them the user's own quoted text. The rule is now:
**from the first accepted call onward, only calls and whitespace may appear** (a three-line guard
beside the accept path in `_parse_bare_call_syntax`; the "nothing follows the last call" half is
unchanged). The blind spot was in the *tests* as much as the code — every M-1 negative pin ended in
prose, so no pin exercised a message ending in a genuine call; three such pins are now the primary
regression guard. Also closed in the same pass: **N-3** (K-027 flipped `🔵 proposed` → `🟡
in-progress`, m-7 from round 1, which had been dropped silently), **N-4** ("would have converted the
observed shape *as recorded*" was literally false — the recorded shape is line-wrapped and still does
not parse; corrected to "*as reconstructed (de-wrapped)*", and the fixture comment in `test_llm.py`
no longer leads with "Verbatim"), **N-5** (`_parse_content_tool_calls`'s docstring no longer calls
the JSON branch "(unchanged)" — the probe *order* is unchanged, the empty-envelope fall-through is
not), **N-2** and **N-6** (three sanctioned residuals — the own-line judge echo, the multi-line
array-wrapped verdict, and `name ({…})` with a space before the paren — pinned as characterisation
tests and stated in the docstrings, so each is a visible boundary rather than an accident).
**Residual after N-1, accepted and pinned:** a *contiguous catalogue* of own-line calls with no prose
between them still fires — it is structurally identical to an intended multi-call, so position alone
cannot separate the two. Closing it needs the granted tool names as a recognition filter (K-035
remedy 3), a real layering decision that is deliberately not taken here.

**Verified:** targeted offline run `pytest tests/test_llm.py tests/test_app.py -q` → **64 passed**
(56 before this second gate pass ⇒ **+8**; 45 before the first gate pass). The full suite was last
observed at **595 passed, 1 deselected, 0 skipped** *before* the second gate pass and was
deliberately **not** re-run afterwards: a plain `pytest` wipes the global `reference` graph and a
concurrent unit was live against the same FalkorDB, while this change is 100 % offline string
parsing. Slice A contributes **+38 tests** (entry baseline 533; 552 after the first draft's 19, 563
after the first gate pass's 11, 571 after this pass's 8) — the 595 figure also carries a concurrent
unit's additions. Of this pass's 8 tests, **3 were red before the fix** (the N-1 shapes: a quoted
fenced call, a disclaimed call and an echoed user snippet, each followed by a genuine call — all
three dispatched **two** calls before, all three stay text now); 5 are characterisation pins that
passed on arrival and are labelled as such (the contiguous catalogue, the own-line judge echo, the
single-line and multi-line array-wrapped verdicts, and the space-before-paren call). Of the 11
first-gate-pass tests, **8 were red before that fix** (the 3 M-1 multi-line false positives, the m-2
nested-identifier case, the m-5 identical-repeat case, and 3 B-1 judge false-advance cases); 3 were
characterisation pins (the two m-1 `tool_calls: []` directions, and the two-disagreeing-verdicts
case, which the old span parse already lost to invalid JSON rather than by design). All 8 positive
bare-call pins and all 3 judge-recovery pins from the first draft survive unchanged, by name and
body. `ruff check .` on the four files this slice owns (`llm.py`, `app.py`, `test_llm.py`,
`test_app.py`) → `All checks passed!`. **No Cypher, DDL, schema or script changed** ⇒
`scripts/test_queries.sh` untouched and unaffected; no new node type, index or vector dimension ⇒
no RAM impact (rule 6).

**Scope held:** K-027 items **2–5** stay open (terminal-node engine contract, judge calibration,
golden-set expansion, Ministral re-probe), as do the carried `guards.py` findings m-1/m-2/m-3.
`executor._drive_loop` (SHA-locked `71055f756280`) not touched.

## 2026-07-24 — Fixed pre-existing ruff error in `llm.py`; `ruff check .` now clean

**What:** Reordered two import lines in `server/falkorchat/llm.py` (`from dataclasses import …`
now sorted before `import urllib.request`) via `ruff check --fix .`, clearing the one pre-existing
`I001` error. `server/` now passes `ruff check .` with zero errors. `AGENTS.md`'s ruff bullet
updated to describe the clean baseline (no gate is wired to run ruff automatically; a future red
result is a real regression, not known noise).

**Why:** Flagged during the 2026-07-24 `coder` kaizen-inbox distillation as a trivial one-line
follow-up outside that pass's remit; landed as its own change per the module convention (small,
independent, reviewable).

**Verified:** `ruff check .` → `All checks passed!`; `pytest -q -k llm` → 18 passed, 2 skipped
(FalkorDB-down guard, expected), 514 deselected.

## 2026-07-22 — M3-close documentation-archival sweep (doc-only)

Housekeeping pass following **milestone M3 — Workflows ✅ (delivered 2026-07-21)**: its completed
planning docs move to `docs/archive/` per the module documentation convention
([`AGENTS.md`](../AGENTS.md) → "Module documentation convention"). **Doc-only — no source, test,
or script file changed** (verified: every touched path ends in `.md`; git shows only renames + the
link edits + this entry).

**Moved (via `git mv`, history preserved) — 20 files:**

- **12 plans** `docs/plans/ → docs/archive/plans/`: `m3-executor.md`, `m3-executor-ml.md`,
  `m3-executor-landing2.md`, `m3-executor-coordination.md`, `m3-guard-calibration.md`,
  `m3-guard-thread-context.md`, `m3-capability-probe-ml.md`, `m3-process-flow.md`,
  `m3-process-flow-coordination.md`, `m3-workflow-engine.md`, `m3-workflow-engine-coordination.md`,
  `m1-cleanup.md`.
- **5 reviews** `docs/reviews/ → docs/archive/reviews/` (new archive subdir created):
  `m3-executor-impl.md`, `m3-executor-landing2-impl.md`, `m3-executor.md`,
  `m3-guard-thread-context-impl.md`, `m3-process-flow.md`.
- **1 test-plan** `docs/test-plans/ → docs/archive/test-plans/`: `m3-workflow-engine.md`.
- **1 test-report** `docs/test-reports/ → docs/archive/test-reports/`: `m3-workflow-engine-report.md`.
- **1 requirement** `docs/requirements/ → docs/archive/requirements/` (new archive subdir created):
  `llm-native-workflows.md` — the tico requirement (FR-1…FR-7 / AC-1…AC-6) that drove M3, now
  realized by the delivered engine.

**Inbound links fixed in the same change:** **157** component-root-relative path strings pointing
at a moved file rewritten to their `docs/archive/…` target — 61 across the live docs (`BACKLOG.md`
36, this file 17, `AGENTS.md` 4, `DESIGN.md` 2, `docs/plans/local-model-ram-budget-ml.md` 2) and
96 in the moved docs' cross-references to each other (the last 4 being the archived plans/review/
test-plan pointers to `llm-native-workflows.md`, repathed with it). Prefix collisions
(the `plans/` copy vs the `reviews/` copy of `m3-executor.md`; the `-ml`/`-impl`/`-landing2`
variants) were handled by full-path-anchored rewrites, not blind name substitution. Bare-filename
prose citations without a `docs/…/` prefix (in `server/**.py`, `scripts/seed_workflows.sh`) are not
path links and were left untouched.

**Deliberately kept active (not completed plans):** the reusable runbook
`docs/plans/demo-environment-bringup.md`; the forward-looking `docs/plans/graphrag-eval-ml.md`
(K-026, not yet built); the two parked standing environment references
`docs/plans/local-model-ram-budget-ml.md` and `docs/plans/wsl2-memory-diagnostic.md`; and the
still-in-progress requirement `docs/requirements/summary-nodes.md` (status *Interviewing*, not yet
built). Only the completed M3 requirement `llm-native-workflows.md` was archived.

## 2026-07-21 — M3 K-025: the QA acceptance pass — verdict **PASS with parked, model-gated limitations** ⇒ **MILESTONE M3 ✅**

Closes **K-025**, the un-run U15, and with it **milestone M3 — Workflows**. `qa-engineer` executed
a risk-based, black-box acceptance pass against commit **`98a3cc8`** (tree clean, no source file
changed by the pass). Artifacts: test plan **`docs/archive/test-plans/m3-workflow-engine.md`** (v1.0,
written *before* execution) and test report
**`docs/archive/test-reports/m3-workflow-engine-report.md`**. Baselines held on entry **and** on exit:
server pytest **533 passed / 1 deselected**, query suite **256/256**.

**Verdict: PASS with parked, model-gated limitations ⇒ M3 ✅. Zero blocking defects.**

- **AC-1 · AC-5 · AC-6 — VERIFIED by execution.** An `@mention` on the served app started a triage
  run read back from the graph as `(:WorkflowRun {status:'waiting', defKey:'triage'})-[:TRIGGERED_BY]->
  (:Message)` anchored to the exact triggering message (AC-1); the same 8-step process flow run with
  `trace:true` recorded **18 `TraceEvent`s** (`node_rationale` ×8, `guard_judgment` ×8, `node_note`
  ×2 — every guard with its verdict *and* its why) against **0** events for the identical flow run
  non-debug, so both halves of AC-5 hold; and the per-node fence held on **both** sides (AC-6) —
  only the granted schema was offered on every iteration, and a scripted ungranted call was
  **defensively rejected without dispatch**, while the corrected granted call still dispatched for
  real. AC-6 was driven through the **real** executor, the **real** builtin `ToolRegistry` and the
  **real** graph with a scripted LLM, not a stub estate.
- **The whole `access-request@v1` process flow — VERIFIED.** All three `m3-process-flow.md` §4.3
  paths reproduce the plan's step-by-step table **exactly**, step counts included: privileged
  (`contractor`) **8** steps `submit,submit,route,approval,approval,provision,provision,activate` →
  `done` at `activate`; standard-hire (`engineer`) **6**, no approval; rejected **6**, terminal
  `rejected`, run **`done` not `failed`**. Nine publish-invariant negatives (missing
  `waitsForHuman` on `human` *and* `wait`, a typo'd `cmp` op, zero transitions, 0/2 start steps, an
  unwhitelisted path root, a dangling endpoint, a duplicate step key) all return **400 and write
  nothing** — the unrepairable zero-transition half-write hazard is closed. The input error map is
  precise (400 rejected / 404 unknown run / 409 not parked) and **every rejection is free**:
  `stepCount` unchanged across empty, undeclared-key, reserved-key and `expects`-violating
  submissions. Budget exhaustion and the `NotImplementedError` typed-handler seam both surface as
  the **D-G `{"status":"failed"}` envelope, never a 500**, with the run correctly terminal in the
  graph.
- **AC-2b · AC-3 · AC-4 — recorded model-gated, structurally demonstrated** (decisions **D12-B** /
  **D7**), not as failures. All three were *observed working* in a live interactive run on `ws:qa`:
  intake parked → a plain reply with **no re-`@mention`** resumed it → the fuzzy guard advanced →
  `research` (which correctly **abstained**, `no relevant context found`, on a near-empty
  workspace) → `answer` posted a real reply `PRODUCED`-linked from its `StepRun` → run `done` in
  ~18 s. `pytest -m live` then failed **2/2** on the AC-4 answer-post assertion (`the answer node
  never posted a reply … posts came from: ['intake', …]`) with every prior AC-1/AC-2/AC-3 assertion
  passing. That is **K-027** (Defect C, live-triage reliability on the local 4B) — a known, filed
  limitation, **not** a new defect and **not** an M3-green gate.
- **Specified behaviour confirmed and explicitly *not* filed as defects:** a parked `wait`
  unchanged after 25 s (`awaiting {"kind":"signal","signal":"provisioned"}`) — signal-driven, no
  scheduler (**D-C**; timers are K-028); `prompt` → `NotImplementedError` (**D-E**); the degraded
  RECENT-TURNS guard tier (**D14**); create-only def publishes — an *edited* re-publish returned a
  clean **`201`** while the stored def kept its old name, kind and step config.
- **No verdict line in the report is sourced from the guard calibration**, so the **D10** caveat is
  attached to no line there; it remains binding for K-027 item 3, which owns that measurement.
- **Two non-blocking findings, both filed rather than left in the report:**
  **K-031** (new) — there is **no black-box way to read a def's or snapshot's structure**
  (`GET /workflow-defs/{key}` is metadata-only), so the component's most dangerous documented
  trap — create-only publishes with independently-stale `reference` and `ws:{id}` — is
  **documented but undetectable**; the pass had to drop to raw Cypher. Carries a nit: the step
  budget overshoots by one (`maxSteps:2` → `stepCount:3`). And an **addendum appended to K-027**:
  the prose-tool-call failure is **not terminal-node-specific** — the *intake* node emitted the
  literal `post_message({...})` in bare function-call syntax, a shape
  `llm._parse_content_tool_calls` does not recover, so the clarifying question never reached the
  thread while the run parked looking healthy. Widening that parser is a cheap, offline-testable
  mitigation that would have converted the observed run — recommended **before** the engine-level
  terminal-node contract.
- **Environment discipline:** `ws:qa` created at the live-probed embedding dim (1024), exercised,
  and **deleted**; `reference`/`ws:acme` additive-only with **no def or snapshot subgraph ever
  deleted**; the documented `pytest → test_queries.sh → seed_workflows.sh → verify → exercise`
  order followed throughout — and the `reference`-wipe trap was **observed live twice**, confirming
  the AGENTS.md warning is accurate and load-bearing. Nothing committed by the pass.

## 2026-07-21 — M3 K-024 (second half): the LLM-free `kind:'process'` proof flow, analyst-gated twice — **M3's last build item ✅**

Closes **K-024** — the `kind:'process'` business-process proof flow the DESIGN §6.3
"coordination is workflow" claim rests on — and with it the last **build** item of M3. Only
**K-025** (QA acceptance) now stands between the component and **M3 ✅**. Delivered by the
teco-coordinated chain over five units: **graph-dba** (U0), **tdd-engineer** (U1), **coder**
(U2, U3, U4, U4b), **teco** (U5 + every integration run), with the **mandatory analyst gate** run
**twice** as a non-negotiable done-condition. Plan `docs/archive/plans/m3-process-flow.md` (v2.1);
coordination log `docs/archive/plans/m3-process-flow-coordination.md`; all three gates in
`docs/archive/reviews/m3-process-flow.md`. New baselines: server pytest **523 → 533 passed / 1 deselected**
(350 at the start of the slice); query suite **241 → 256**.

**The central design claim, and it held: `_drive_loop` was never modified.** SHA `71055f756280`
before the slice, after every edit, and at closeout. A business process fell out of *park-and-branch*
with **no new primitive, no new run state and no scheduler**: a `human` step is just a step whose
outgoing guard reads a `ctx` key that does not exist yet, so the executor's existing "no transition
fired" outcome parks it — and writing that key from outside makes the same guard fire on resume.
Only two capabilities were missing: **read `ctx` in a guard** (U1) and **write `ctx` from outside**
(U3).

- **U0 (graph-dba) — two additive queries, no DDL.** `QUERIES.md` §12.12 `start_run_untriggered`
  (a run with no chat `Message` anchor — finding F-2) and §12.13 `resume_run_with_ctx` (the ctx write
  **folded into** the existing resume CAS, decision D-F, so no window exists where one writer's ctx is
  read by another's in-flight drive). PROFILEs beat the plan's assumption — `start_run_untriggered`
  is a single `Node By Index Scan` and is *cheaper* than §12.1; `resume_run_with_ctx` has no residual
  `Filter`. Zero-row contracts **verified, not assumed** (the CAS loser wrote neither the flip nor the
  ctx). `bootstrap_schema.sh` untouched, RAM ≈ nil. Query suite 241 → 256.
- **U1 (tdd-engineer) — the deterministic `cmp` guard family** (decision D-A):
  `{kind:'cmp', path, op, value}` + `all`/`any`/`not`, whitelisted ops, two whitelisted path roots
  (`ctx.`/`output.`), depth/width/node caps, `validate_cmp` + `render_label`, populated `rationale`.
  **No parser, no `eval`, no new dependency** — and named `cmp`, not `expr`, so DESIGN §13's "no
  expression library is built" stays literally true (`kind:'expr'` still raises). **Strict at publish,
  total at drive** (open item O-1): an unwhitelisted path root is rejected at seed time, but a missing
  path at drive is simply `False`. `test_guards.py` 33 → 143, including a De Morgan contrast pair that
  pins the `ne`-vs-`not` asymmetry side by side.
- **U2 (coder) — typed step handlers + two publish invariants.** `_execute_step` became an explicit
  dispatch: `agent`+LLM → the agent loop; `agent` without an LLM → the preserved empty stub (finding
  F-3, load-bearing for the whole offline estate, now documented as to *why*); `human`/`decision`/`wait`
  → three pure handlers. **⚠️ Behaviour change (R-3): `prompt`/`tool`/`message` and any unknown type
  now raise `NotImplementedError`** naming the plan, where they previously fell through to a **silent
  no-op** — a `decision` node used to "succeed" doing nothing. The M-1 fault net stamps `fail_run` and
  re-raises. `_validate_def_spec` gained both invariants, running **last** and each preceded by
  `_normalize_opaque` so the REST front door (which types `config`/`guard` as `str`) cannot escape
  them. **Named fixture edits, exactly as budgeted by gate findings B-1/B-2:**
  `test_executor.py:345`'s `{"key":"end","type":"task"}` → `type:"agent"` (it is executed by the real
  loop inside the Defect-B regression pin, which the new `NotImplementedError` would otherwise have
  killed), and the `type:"human"` fixtures in `test_api.py` + `test_services.py` that declared no
  `waitsForHuman`; the five affected `pytest.raises` gained `match=` so none went vacuous. Six
  mutations, all killed.
- **U3 (coder) — start-without-trigger + the human-input channel** (decision D-B, REST):
  `POST /workflow-runs` and `POST /workflow-runs/{id}/input`, with the D-G five-handler error map
  (`WorkflowRunNotFoundError` 404 · `WorkflowRunNotWaitingError` 409 · `WorkflowInputRejectedError`
  400 · `WorkflowConfigError` 400 · `WorkflowEngineDisabledError` 503) registered via a table — no
  blanket `RuntimeError` handler. Reserved ctx keys (`threadId`, `error`) are rejected at the
  **service** layer so MCP inherits the rule, closing the latent bug where a caller-set `threadId`
  would let any chat message resume a process run. Submitted input is validated against the parked
  step *before* the merge (D-H), so a mistake costs no step budget. A drive fault reports the run's
  **graph truth** — `status` *and* `ctx` from the same post-fault re-read — and a non-terminal
  re-read re-raises rather than dressing a zombie run as success.
- **U4 (coder) — the proof def, its seed and an offline acceptance test.**
  `server/falkorchat/proof_defs.py` ships `ACCESS_REQUEST_DEF`: **`access-request@v1`**, six steps /
  six transitions, submit→route→approval→provision→activate\|rejected, `human`×2 / `decision`×3 /
  `wait`×1, the four `cmp` ops `exists`/`in`/`eq`/`truthy`, two terminal outcomes,
  `ACCESS_REQUEST_MAX_STEPS = 24`. (The key is **`access-request`**, not `onboarding` — that key
  collides with long-standing test fixtures.) `scripts/seed_workflows.sh` now loops over **both** defs.
  `server/tests/test_process_flow.py` drives all three §4.3 paths — privileged (8 steps), standard hire
  (6, exercising conditional-beats-unconditional from the *losing* side) and rejected (6) — through the
  service layer, **fully offline: no LLM, no network, no `live` marker**. It imports the same constant
  the seed script publishes, so seed and test cannot drift.
- **U4b (coder) — the implementation gate's findings, closed at the right layer.** A drive fault on
  the `@mention` start path was left with **no log line anywhere** once D-G's catch swallowed it —
  fixed with one `logging.exception` inside the existing `except`, envelope byte-unchanged and the
  catch deliberately *not* gated to REST callers. **Open item O-6 was fixed, not filed:** a def
  published with zero transitions collapsed `_PUBLISH_CYPHER`'s trailing `UNWIND` **after** the steps
  and `START` were written, raising `IndexError` on a partial write — and because publish is
  create-only, the corrected retry was a silent no-op on a permanently poisoned `(key, version)`. A
  `_validate_def_spec` rule (running last, like the other two) now rejects it **before any repository
  call**. An empty input body is likewise rejected before it can win the CAS and burn a step. The
  acceptance test moved to the test-only version **`access-request@v1-test`** so a finished pytest
  session can never leave a *test's* publish masquerading as a real seed of the production pair —
  overriding **only** the version key, so the anti-drift property survives.
- **U5 (teco) — closeout.** DESIGN §6.1 rewritten for what the engine actually executes (including
  finding **F-1**'s two-part correction: `TRANSITION.on`/`StepResult.on` are **vestigial and
  descriptive only**, and guards sort by `(guard == "", order)` — *conditional first*, `order` only a
  tie-break within each class), §6.3 gained the proof pointer + the K-025 handoff note, §13 amended
  with `cmp`-not-`expr`. K-024 → ✅, K-025 marked unblocked, and three items filed rather than folded
  in: **K-028** (workflow timers — `wait` is signal-driven because this system has **no scheduler**;
  decision D-C), **K-029** (converge the seed def sources into `proof_defs.py`; `triage@v1`'s literal
  is still inline in `seed_workflows.sh` — carries the unenforced symmetric `decision` publish
  invariant of nit n-3), **K-030** (allow zero-transition defs by guarding the `UNWIND` the way §4's
  mention block does, instead of rejecting them; folds in the residual **publish-only** asymmetry —
  `materialize_snapshot` reuses the same unguarded query).
- **Gates.** Plan gate: **request changes** (2 blocker / 6 major) → plan v2.1, all closed in text and
  re-verified diff-scoped. Implementation gate U0–U4: **approve with suggestions**, 0 blockers, two
  majors (M-A the swallowed stack trace, M-B the reachable O-6 route). Re-gate U4b: **approve with
  suggestions**, no blockers, all seven findings closed *as ruled* — not more, not less. Both gates
  independently re-verified the `_drive_loop` SHA and confirmed the blast radius on the existing test
  estate was exactly the named fixtures, with **no assertion weakened, deleted or made conditional**.
- **Process incident (contained, no work lost).** During U3 the implementer twice reached for a
  tree-mutating git command to *read* or *undo* something; the second (`git checkout <path>`) reverted
  `services.py` to HEAD and destroyed the unit's work in one command. It was reconstructed and
  independently verified against `efdeeb3` — no top-level symbol lost, every diff deletion accounted
  for. **Standing rule now in every implementer brief:** never `stash`/`checkout <path>`/`restore`/
  `reset` a working tree — read baselines with `git show <ref>:<path>`.

## 2026-07-19 — M3 K-022 Landing 2: trigger + triage proof flow (U11–U14), analyst-gated — **U15 not run**

Second landing of **K-022 — LLM-native workflow executor**: the `@mention` trigger, the triage proof
flow, and the two live defects that landing it exposed. Delivered by the teco-coordinated chain —
**tdd-engineer** (U11 trigger wiring + U12 REST run inspection, and later both defect reproduction
suites), **coder** (the U13 workflow seed, the D14 revert, and the gate's code major) — with the
**mandatory analyst review gate** as a non-negotiable done-condition. Plan `docs/archive/plans/m3-executor.md`
+ the U11 design patch `docs/archive/plans/m3-executor-landing2.md`; coordination log
`docs/archive/plans/m3-executor-coordination.md`. New server baseline: **pytest 283 → 350 passed, 0 skipped,
1 deselected** (with FalkorDB up; teco re-verified independently). Query suite unchanged at
**241/241** — Landing 2 has **zero** graph/DDL/QUERIES surface.

- **U11–U12 (tdd-engineer):** `trigger.py` `WorkflowTrigger.maybe_trigger` (§6 ordered rule:
  loop-guard → resume-if-waiting → `@mention`-to-start → fall through to the M2 responder), one
  handler per request (trigger XOR responder), `WORKFLOW_ENABLED`/`TRIGGER_DEF_KEY` config **default
  off** so the baseline stays network-free; **Option B** emission linking (buffer during agent-node
  execution, `StepRun-[:PRODUCED]->Message` after `_record`, keeping the §2.1 A/B/C loop and
  `record_step_and_advance` byte-for-byte); the Landing-1 **M-1** fault net (`_drive` wraps
  `_drive_loop`, so a mid-drive fault can no longer leave a zombie `status='running'` run); agent-node
  thread context (`_read_thread_context`, window 20); `GET /workflow-runs/{id}` + `/step-runs` +
  `/trace`. Its own analyst gate (`docs/archive/reviews/m3-executor-landing2-impl.md`) came back
  **approve-with-suggestions, 0 blocker / 0 major**.
- **U13 (coder):** `scripts/seed_workflows.sh` — publishes + materializes `triage@v1` (three
  `type:'agent'` steps, intake→research→answer) through the **service layer**, not raw Cypher;
  `start_server.sh` seeds it before uvicorn.
- **U14 (tdd-engineer):** `tests/test_workflow_live.py`, the `live`-marked AC-1…AC-4 e2e, delivered
  **deliberately RED** — it pinned two real defects rather than being bent to green.
- **Defect A — the intake→research guard could never fire (fixed).** Structural, not prompt
  calibration: `executor.py` passed `thread=None` and `guards.py` declared the parameter but never
  read it, so the DS-prescribed **recent-turns fallback (N=6) did not exist** and the judge always
  saw an empty understanding. Fixed **at the seam, not in a prompt**: the thread window the agent node
  already reads rides out on `StepResult.thread` → `thread=result.thread` → `guards._recent_turns`,
  with the DS precedence (understanding primary, turns only when empty). Zero extra graph reads; the
  locked `_drive_loop` untouched. Design: `docs/archive/plans/m3-guard-thread-context.md`.
- **Defect B — a hallucinated `@mention` failed the whole run (fixed).** Tool errors are now
  survivable: a failed dispatch returns an error string the model can act on instead of propagating
  to `fail_run`. Pinned at drive level by
  `tests/test_executor.py::test_hallucinated_mention_does_not_fail_the_run`.
- **D14 — S5 reverted.** The intake `{"understanding":{…}}` JSON instruction did reach the *primary*
  extract-then-judge path, but **regressed live intake advancement 10/10 → 3/10** on the shipped
  Qwen3-4B (the model filled `missing` with forensic demands on every turn and the uncalibrated judge
  suspended). It was **surgically reverted** from `scripts/seed_workflows.sh`, with the removal site
  commented so it is not re-added; the separately-measured **Defect-C prompt mitigations were
  retained**. Consequence to carry: the shipped guard runs only on the **degraded RECENT-TURNS tier** —
  a `guard_judgment` citing turn text is expected, not a defect — and the `understanding` primary tier
  is unreachable in this cut by design.
- **D16 — tool-error split (propagate + log).** `UnknownActorError`/`ThreadNotFoundError` — and any
  future `ServiceError` subclass — **propagate** to the M-1 fault net; only an explicit fail-closed
  allowlist (`UnknownMemberError`, `InvalidSearchQueryError`) is absorbed as a model re-prompt, and
  every failed dispatch logs unconditionally. These are deployment misconfigurations, not
  model-correctable arguments; absorbing them produced a run reaching `done` having posted nothing.
  Closes analyst finding M-2.
- **Analyst gate: `approve with suggestions` — 0 blocker / 2 major / 3 minor / 3 nit**
  (`docs/archive/reviews/m3-guard-thread-context-impl.md`, on commit `aa8b813`). All five mandatory
  confirmations affirmative, including the `_drive_loop` byte-identity (SHA `71055f756280`, unchanged
  across `514346b`/`c3cc239`/`aa8b813`). **Both majors closed before the commit:** **M-1** (doc drift —
  `m3-executor.md` §8 documented prompts the reverted script no longer seeds) and **M-2** (the silent
  `ServiceError` catch, closed by D16). The three minors + three nits are carried on
  [`BACKLOG.md`](./BACKLOG.md) under K-027 so they cannot rot.
- **D13 capability probe (data-scientist):** a fits-16GB comparison of the shipped `qwen/qwen3-4b-2507`
  against Ministral 3 3B (Q8_0), config/env-only. **Ministral loses — no model swap.** Intake
  advancement 3/10 vs 0/10; AC-4 terminal post 2/3 vs not-measurable. Two findings routed to K-027: the
  fuzzy-guard judge's **bare `json.loads` is model-fragile** (a fenced ```` ```json ```` reply made all
  26 golden cases unparseable — the shipped Qwen path is unaffected), and Ministral is actually
  *better* at the terminal tool call. Note: `docs/archive/plans/m3-capability-probe-ml.md`.
- **D15 parity repair (graph-dba, user-authorized destructive op on this dev box).** The stale
  `ws:acme` `WorkflowDefSnapshot` was deleted and `triage@v1` republished into **both** `reference` and
  `ws:acme`, resolving a **split-brain** in which the def had been wiped while the stale snapshot — the
  thing the executor actually drives — survived. Throwaway graphs `ws:probe`/`ws:live` dropped. No
  `WorkflowRun` existed, so nothing was severed. Environment-specific authorization: on a shared graph,
  `DETACH DELETE` on a def's `Step`s severs live runs' `AT_STEP`/`OF_DEF` and is a data-loss event.
- **⚠️ Two environment hazards, now documented in `AGENTS.md`:** (1) `server/tests/conftest.py`'s
  `wf_repo` fixture wipes the global `reference` graph — so it is **not only `test_queries.sh`**: a
  plain `pytest` with the DB up destroys the published def while leaving the `ws:acme` snapshot, the
  same silent split-brain, from the command we treat as the routine baseline. Re-run
  `seed_workflows.sh` after either. (2) Published defs are effectively **immutable** —
  `repository._PUBLISH_CYPHER` is `MERGE (st:Step …) ON CREATE SET st.config`, so editing a prompt and
  re-seeding prints a clean `already present — no-op` while the old config stays live.
- **❗ Explicitly NOT done — U15 (qa-engineer acceptance, = K-025) was not run.** Per decision **D12-B**
  the executor **mechanism** is proven (Defect A dead; the flow reaches `done` with the judge reasoning
  from real evidence), while live-triage **reliability** is descoped to K-027: the terminal
  `post_message` call is unreliable on a 4B (**Defect C** — AC-4 posting measured ~2/8, then 0/3 after
  a strengthened prompt, then 2/3 in the probe replay: unreliable in every measurement) and the judge is
  still **uncalibrated**. **Landing 2 is delivered and gated, not accepted** — AC-2b/AC-3/AC-4 remain
  model-gated and only structurally demonstrated, and M3 does not reach ✅ on this landing.

## 2026-07-18 — WSL2 memory diagnostic produced; fix parked (not applied)

Read-only devops diagnostic of the WSL2 memory-overload crashes on the downgraded 16GB host,
persisted at `docs/plans/wsl2-memory-diagnostic.md`. **Verdict: ballooning confirmed by defaults** —
WSL2 runs uncapped at its 8GB default (50% of the host) with `autoMemoryReclaim` off, overcommitting
host RAM alongside Windows-side LM Studio (not reproduced live — FalkorDB was down during the run).
Recommended fix (`memory=6GB` + `swap=4GB` + `autoMemoryReclaim=gradual` in `C:\Users\mauri\.wslconfig`,
keeping `networkingMode=mirrored`; needs `wsl --shutdown`) was **parked, not applied, per the user's
decision** — un-park if the crashes recur. Tracked as a Parking-lot bullet in
[`BACKLOG.md`](./BACKLOG.md) (`## Parking lot / ideas`). Docs-only; no config or code changed.

## 2026-07-12 — M3 K-022 Landing 1: LLM-native workflow executor (U1–U10), analyst-gated

First landing of the reframed **K-022 — LLM-native workflow executor**: the offline executor +
node capabilities (Phases 0–3, units U1–U10). Delivered by the **teco-coordinated
graph-dba → tdd-engineer → coder** chain with a **mandatory analyst review gate** — the team's
first fully-gated coordinated run. Plan `docs/archive/plans/m3-executor.md`; coordination log
`docs/archive/plans/m3-executor-coordination.md`. Trigger + proof flow (Landing 2, U11–U15) is a separate
later run, **not started**. New baselines: **query suite 193 → 241/241**, **server pytest 196 → 283**,
both green (teco re-verified independently).

- **U1–U2 (graph-dba):** `bootstrap_schema.sh` adds `TraceEvent.traceId` index **then** UNIQUE
  (index-before-constraint, idempotent). DESIGN §5.1/§5.2/§6.1/§6.2/§7.1/§13 reconciled — LLM-judged
  guards + the `type:'agent'` node kind, §13 guard-language open question marked resolved, and the
  stale `EMITTED` on StepRun→Message corrected to **`PRODUCED`** (§5.1/§5.2). QUERIES §12 = twelve
  live-verified/PROFILEd run / step-run / trace queries. The M4
  `WorkflowRun-[:LAST_STEP_RUN]->StepRun` tail pointer makes `record_step_and_advance` an O(1)
  atomic advance (no chain-walk). No new index — resume rides the existing `status` index.
- **U3–U5 (tdd-engineer):** `repository.py` — the §12 methods 1:1 + `WorkflowRunNotFoundError` /
  `StepBudgetExceededError`. New `executor.py` — `WorkflowExecutor` (the §2.1 A/B/C loop),
  `Tracer`/`NullTracer`/`GraphTracer`, run-level step budget, monotonic StepRun clock.
  `services.py` — start/resume/read-run methods, tenant seam respected. The slice-1 `start_key` =
  `start:True` contract was kept.
- **U6 (coder):** `llm.chat(messages, tools) -> ChatResult` with dual-shape parsing (native
  `tool_calls` field primary, content-embedded-JSON fallback); `complete()` preserved byte-for-byte.
- **U7–U8 (tdd-engineer):** `guards.py` `evaluate_guard` (DS-note Q1 extract-then-judge,
  `{decision,rationale}`, bias-to-suspend on ambiguity; `""`=unconditional; `expr`/unknown =
  `NotImplementedError` seam, M7). `executor._run_agent_node` — a bounded, tool-scoped agent loop
  with defensive **AC-6** rejection of ungranted/malformed calls (re-prompt, never dispatched) and
  graceful `maxIterations` exhaustion. The §2.1 loop was left byte-for-byte unchanged.
- **U9–U10 (coder):** `tools.py` `ToolRegistry` + `post_message` (§4 write as the agent →
  `PRODUCED` link via `services.link_step_emission`), `graphrag_retrieve` (Q2 τ≈0.5 cutoff / cap 5 /
  floor 1 / abstain), `human_handoff` (registered capability, granted to no node). `McpToolClient`
  MCP-client seam — stub-tested in-memory; real external servers deferred.
- **Analyst gate:** `docs/archive/reviews/m3-executor-impl.md` — **approve-with-suggestions, 0 blockers**
  (1 major, 3 minor, 3 nit). Major **M-1**: `executor._drive` has no top-level `try/except`, so an
  unexpected mid-drive exception leaves the run stuck at `status='running'` — a permanent
  un-resumable zombie once live defs/tools run (not a green-suite blocker; the offline path is
  deterministic). Both deliberately-deferred seams — live `PRODUCED`-link ordering and agent-node
  thread-message context — were ruled **acceptable for Landing 1** and carried to U11.
- Layering held (no Cypher outside `repository.py`); D1–D5 and the M4/M7 decisions honored; AC-5
  (trace on/off) and AC-6 hold by construction; the default app import stays network-free. Cost
  datapoint recorded in the coordination doc: **~1.20M subagent tokens / 238 tool uses / 6
  delegations + the gate** (the first measurable cost/benefit reading for the review gate).

## 2026-07-11 — Docs unification: kaizen/ retired into docs/ (repo module convention)

Unified the component's two documentation homes into one `docs/` tree — the repo-wide module
convention now defined in the root `AGENTS.md` (agent-folder `claude/<agent>/kaizen/` pairs are
a separate convention and unchanged):

- `kaizen/plan.md` → **`docs/BACKLOG.md`** (living backlog; K-numbered items unchanged) and
  `kaizen/history.md` → **`docs/HISTORY.md`** (this file); `kaizen/` removed.
- New **`docs/archive/{plans,test-plans,test-reports}/`** for frozen documents of closed
  milestones. Moved: plans `m1-chat-mcp`, `m2-groundwork`, `m2-groundwork-queries`,
  `m2-graphrag`, `m2-agent-participant`, `doc-consolidation-sweep` (delivered 2026-07-05 —
  header was stale); all four M1/M2 test-plans; all four test-reports. Active M3 plans,
  `m1-cleanup`, `graphrag-eval-ml` (K-026 pending), `demo-environment-bringup` (living
  runbook), and `reviews/` stay in place.
- **Rule going forward:** a plan/test-plan/report moves to `archive/` (same subdir name) when
  its milestone closes, with inbound links fixed in the same change.
- Inbound references rewritten repo-wide (docs, `AGENTS.md` key-docs table, `README.md` tree,
  server source comments, `test_queries.sh`, `.dockerignore`, `.claude/settings.local.json`,
  agent kaizen logs citing concrete paths). Old dated entries below keep their period prose but
  their paths were updated to resolve.

Moved the engine off the floating `falkordb/falkordb:edge` tag to the tagged release
**`v4.18.11`** (module `41811`, Redis 8.6.3): `scripts/start_falkordb.sh`, `compose.yaml`,
and the CI service container (`.github/workflows/falkor-chat.yml`) now pin it; the
salesperson component's `start_falkordb.sh` moved with it. Container recreated on the same
`falkordb-data` volume after an explicit `SAVE` — all graphs (`ws:acme`, `reference`,
`ws:test`) survived. Re-verification per the quirks-file rule: **query suite 193/193,
server pytest 196 passed** on the pinned build. Current-state docs re-stamped (`AGENTS.md`
header, README, `docs/QUERIES.md`, `docs/DESIGN.md` §2 callout,
`claude/graph-dba/falkordb-quirks.md`). Rationale: edge is a moving target that forced
verify-everything churn; a pin makes the live-verified facts durable until a deliberate
upgrade.

## 2026-07-09 — K-020 + K-021: M3 slice 1 (workflow defs + snapshot materialization) delivered

First slice of **M3 — Workflow engine**, delivered end-to-end by the **teco-coordinated
architect → graph-dba → tdd-engineer chain** (the teco K-001 nested-delegation validation run —
see `claude/teco/docs/HISTORY.md` 2026-07-09). Architect decomposed all of M3 into
**K-020…K-025** and wrote the slice-1 plan (`docs/archive/plans/m3-workflow-engine.md`, Part A + Part B);
coordination log at `docs/archive/plans/m3-workflow-engine-coordination.md`. Suites verified
independently after integration: **query suite 149 → 193/193, pytest 156 → 196.**

- **K-020 — def model in `reference`.** *graph-dba gate:* `Step.stepUid = "{defKey}:{version}:{stepKey}"`
  (architect's synthetic key — `Step.key` is unique only within a def) with index + UNIQUE in
  `reference`; one justified model addition — a `-[:HAS_STEP]->` containment edge, because the
  plan's `STARTS WITH stepUid` scoping PROFILEd as a label scan (HAS_STEP gives index-anchored
  O(steps-in-def) reads); canonical `QUERIES.md §11` (publish, read-def, list/get def), live-verified
  + PROFILEd; DESIGN §6.1/§7.1/§7.2 updated. *tdd impl:* `db.reference_graph` seam, reference-graph
  repository methods (1:1 with §11) + typed errors, `services.publish_workflow_def` with spec
  validation **before any write** — `start_key` resolved as "exactly one step declares `start: True`"
  (implementer's call, plan had no param; lock the contract at K-022).
- **K-021 — snapshot materialization.** *graph-dba gate:* workspace `Step.stepUid`/`Step.key` DDL
  in `bootstrap_schema.sh` (additive); materialize / read-snapshot / list-snapshot queries in §11.
  *tdd impl:* two-phase `services.materialize_def` (read `reference` → idempotent MERGE into
  `ws:{id}`; not atomic across the graph boundary, retry completes), size-bounded schemas, thin
  REST surface. **Structural parity proven** (publish → materialize → snapshot `==` reference def)
  + idempotency; reference-wiping test fixture added.
- **Scope discipline:** executor (K-022), chat linkage (K-023), proof flows (K-024) explicitly not
  built; `ws:acme`/`reference` additive-only; §13 guard-language decision confirmed **not forced**
  by slice 1 — returns to the user at K-022's architect pass.

## 2026-07-08 — K-008 + K-013 + K-014 + K-015: M2 GraphRAG delivered → milestone M2 done

End-to-end GraphRAG loop, delivered as the full graph-dba→tdd→coder→qa sequence and
**QA-accepted (K-015, PASS, zero defects)**. Prerequisite: a devops LM-Studio reachability spike
confirmed `http://localhost:1234/v1` reachable from WSL2, embedding dim **1024**, both models live
(`text-embedding-qwen3-embedding-0.6b`, `qwen/qwen3-4b-2507`).

- **K-008 — retrieval core.** *graph-dba gate:* verified the §6 hybrid ANN query + `SET m.embedding`
  live against a 1024-dim workspace, `GRAPH.PROFILE` confirmed the vector index is hit, Entity
  expansion no-ops cleanly; raised `test_queries.sh` **126 → 135**; deliverable `docs/archive/plans/m2-graphrag.md`.
  New quirk logged: a wrong-dimension `vecf32` write is *silently accepted* then drops the node out of
  the ANN index → validate length client-side. *tdd impl:* `repository.set_embedding` (client-side dim
  validation, `EmbeddingDimensionError`) + `repository.hybrid_search` (§6, channel/workspace variants)
  + `services.hybrid_search` (`RAG_QUERY_TIMEOUT_MS`) + `embedding.py` (`Embedder`/`LMStudioEmbedder`/
  `EmbeddingWorker`, injected transport). pytest **110 → 123**.
- **K-013 — AI Agent participant + `EMITTED` provenance.** *graph-dba gate:* defined
  `(answer:Message)-[:EMITTED]->(seed:Message)` with `score`+`rank` props, riding **inside the guarded
  §4 write** (exactly-once under `dupMsg` replay, no relationship constraint needed); canonical
  `QUERIES.md §10`; raised `test_queries.sh` **135 → 149**; deliverable `docs/archive/plans/m2-agent-participant.md`.
  *tdd impl:* `repository.post_agent_answer`/`read_provenance`/`read_citing_answers`, `llm.py`
  (`LMStudioLLM`), `responder.py` (`AgentResponder` — `@mention` trigger, loop guard on
  `role:"assistant"`, LLM/embedder before the guarded write ⇒ failure posts nothing). **Decisions
  (user):** trigger = agent `@mention` only; **every** posted message is embedded out-of-band (corpus
  grows) — both wired via FastAPI `BackgroundTasks`. pytest **123 → 154**.
- **K-014 — live wiring + web.** Served app builds the real embedder/worker/LLM/responder gated on
  `FALKORCHAT_ENABLE_AGENT` (default off → imports/tests stay network-free); `config` gained
  `AGENT_ID`/`AGENT_NAME`/`ENABLE_AGENT` + `LLM_*`; new `scripts/seed_demo.sh` registers the
  `assistant` agent + demo channel/thread; `start_server.sh` now exports `FALKORCHAT_EMBEDDING_DIM=1024`
  + enables the agent + seeds; `server/.env.example` documents runtime env. Web renders assistant
  replies (AI badge) + reader `isMention`; `displayName` added to since-reads (`QUERIES.md §9.1/§9.2`
  in lockstep, suite unaffected). pytest **154 → 156**.
- **Provisioning (ops):** served tenant `ws:acme` dropped and re-bootstrapped at `EMBEDDING_DIM=1024`
  (user-confirmed clean build); vector index verified at 1024.
- **K-015 — QA acceptance (the gate).** Black-box pass across REST + MCP + web + the running
  responder: out-of-band embedding, cosine-ASC ranking, agent answer with `EMITTED` provenance on all
  read surfaces, loop guard, failure isolation, dormant-Entity path — **PASS, no defects.** Plan/report:
  `docs/archive/test-plans/m2-graphrag.md`, `docs/archive/test-reports/m2-graphrag-report.md`.
- **Parked → M2.5 (deferred, not on the M2-green path):** real auth/tenancy (K-016), transport-level
  externally-authenticated agent actor (K-017, K-007 QA carry-over), real-time push (K-018);
  channel-scoped retrieval read (responder currently workspace-wide — trigger self-cites as rank-0);
  `ensure_agent` doesn't persist `displayName`; reverse-provenance not on a public route.
- **Suites:** pytest **156** / query suite **149/149**.
- **Milestone:** closes **milestone M2 — GraphRAG → ✅.** Next milestone: **M3 — Workflow engine.**

## 2026-07-06 — K-012: web request/response UX polish → M1 complete

- **What (client-side only, `web/` — no server/schema/query change):** de-staled the M1
  request/response web path. Three changes in `web/app.js` + `web/index.html`:
  1. **Incremental polling** — the open thread refreshes via `GET …?since=&limit=50` (bounded,
     `since`-anchored, no `NEXT*` walk, no cursor), replacing the full re-fetch-after-post.
  2. **Inline non-blocking toast errors** — replaced **both** `alert()` sites with inline toast
     rendering so a failed post/action no longer blocks the UI.
  3. **Clickable search results** — a search row now opens the message's thread via the `threadId`
     carried on search rows (K-007 denorm).
- **Scope guard:** `web/app.js` + `web/index.html` only — no `.py`, `QUERIES.md`, `test_queries.sh`,
  `bootstrap_schema.sh`, schema, or `scripts/` touched; suites unaffected. Manual-smoke-only per the
  K-005 precedent (no web test harness; `node` not on the box).
- **Parked follow-up → K-014:** polled (`?since=`) message rows carry `authorId` but no
  `displayName` (a `coder` left a code comment in `app.js`); resolving it needs a small server
  change to include `displayName` on since-read rows — folded into the K-014 web-M2 pass.
- **Suites:** pytest **110** / query suite **126/126** (unchanged — no code under test touched).
- **Milestone:** with K-011, closes **milestone M1 — Chat core → ✅**.

## 2026-07-06 — K-011: M1 DoD closeout — append-path load harness + hot-read PROFILE + RAM budget

- **What (devops, with a `graph-dba` PROFILE sub-pass):** closed the M1 append-path load-test +
  hot-read `GRAPH.PROFILE` DoD and folded a per-workspace RAM budget into DESIGN.
  1. **Load harness** — new `scripts/load_test.sh` + `scripts/load_append.py` drive the
     **service-layer append path through REST** (16 concurrent posters, 3000 msgs, 0 errors)
     against an isolated `ws:load` graph. Measured **~614 msg/s; p50/p90/p99 = 24.4/30.6/40.7 ms**.
  2. **Hot-read PROFILE** — `GRAPH.PROFILE` on the four hot reads (§4 thread read, §9.1 & §9.2
     since-reads, §5 search) — **all index-backed (`Node By Index Scan`), none degraded to a
     `NodeByLabelScan`**; raw plans archived by the harness.
  3. **RAM budget** — chat-core floor **~1.06 KB/msg** (measured `INFO memory` `used_memory`
     delta) ⇒ **~101 MB per 100k-msg workspace**; packing table folded into DESIGN §11.1/§11.2.
- **Files:** new `scripts/load_test.sh`, `scripts/load_append.py`; `docs/DESIGN.md` §11.1/§11.2;
  `AGENTS.md` Key-scripts row; `.gitignore` (`.load-out/`).
- **Scope guard:** read-only measurement + docs/harness — **zero new per-workspace RAM cost**;
  no `QUERIES.md`/`test_queries.sh`/`bootstrap_schema.sh`/schema change. Ran against `ws:load`
  (create + delete), never `ws:acme`.
- **Suites:** query suite **126/126** · pytest **110** (green).
- **Milestone:** with K-012, closes **milestone M1 — Chat core → ✅**.

## 2026-07-05 — K-021: §13 open-questions reconciliation + identity-authoritative decision

- **What (doc-only, no code/schema/query/script change):** recorded a newly-made design decision and
  brought `docs/DESIGN.md` §13 "Open questions" back in line with reality.
- **New locked decision — identity source of truth:** the **`identity` graph is authoritative
  (standalone)**, not a projection of an external IdP. The system is self-contained: the `identity`
  graph owns global user identity + auth principals; per-workspace `User` nodes remain membership
  projections of it (consistent with §3 topology). User-approved 2026-07-05; steers K-016 (real auth).
  - Added as a row in **DESIGN §1.2** (the authoritative detailed register; "Detailed in" → §3, §14.3).
  - Added a matching one-line pointer in **`AGENTS.md`**'s decisions index (`… → §3`, no rationale).
- **§13 pruned to genuinely-open questions only:** removed **Embedding model & dimension** (resolved;
  home §1.3) and **Identity source of truth** (now decided; home §1.2) — no resolved-pointers left in
  the "Open questions" list.
- **§13 reworded:** **Bolt vs. RESP** → **Real-time gateway transport** (M1 app driver settled = RESP
  via `falkordb-py`; only the M2.5 push-gateway transport is open, → K-018). **Live config defaults**
  → prefixed **Pre-production config review** and dropped TIMEOUT from the still-to-review set (TIMEOUT
  1000ms already reviewed & kept — K-007, §10; other knobs retained). The three genuinely-open bullets
  tagged with owners: workflow guard expr language (→ M3), retention (→ K-011 data), cross-workspace
  analytics (mechanism open, no milestone).
- **`docs/BACKLOG.md` reconciled:** K-016 "Inputs/prereqs"/Owner/scope now read as **decided** (identity
  graph authoritative; K-016 no longer needs the user for that axis — implements per §1.2); the
  `m2-auth-tenancy.md` recommended-doc row and the milestone-map note updated likewise; removed
  "identity source of truth" from the parking-lot "remaining open questions" line (real auth / K-016 stays).

## 2026-07-05 — K-019: documentation-inconsistency sweep (test counts, embedding decided, M2/M2.5 scope)

- **What (doc-only, no code/schema/query/script change):** reconciled stale numbers and
  contradictory milestone wording in `README.md` and `docs/DESIGN.md`. Counts sourced from a
  **live suite run** (`./scripts/test_queries.sh` → 126/126; `server && pytest -q` → 110 passed)
  with FalkorDB up.
  - **Test counts → true 110 pytest / 126 query suite.** `README.md`: `115/115 passed`→`126/126`
    (step 4 expected output); `(115 assertions)`→`(126)` (repo-layout comment); `(75 tests)`→
    `(110 tests)` and `# 98 passed`→`# 110 passed` (M1 row + pytest example). `DESIGN.md` §12 M1
    roadmap bullet `built and green (70 tests)`→`(110 tests)`. The README M0 roadmap figure
    `(92/92)` was **re-labelled historical** (`92/92 at M0 baseline`), not bumped — it records M0.
  - **Embedding model no longer "open."** `DESIGN.md` §11 RAM line: `default stays 1536
    (embedding model still open, §13)`→`(chosen per workspace); set EMBEDDING_DIM=1024 for the
    decided model (§1.3)`. `DESIGN.md` §13 open-questions "Embedding model & dimension" bullet
    replaced with a resolved pointer to §1.3 (Qwen3-Embedding-0.6B, `EMBEDDING_DIM=1024`). The
    `EMBEDDING_DIM=1536` *default* in scripts was intentionally **left untouched**.
  - **M2-vs-M2.5 scope aligned.** `DESIGN.md` §14.1 Transport/Real-time rows + §14.1 rationale
    note: real-time "deferred to M2"/"M2 real-time" → **M2.5** (agrees with §12 M2 = GraphRAG only
    and the kaizen deferred M2.5 track K-016/K-018). `README.md` M1 roadmap row
    "deferred to M2"→"deferred to M2.5". Auth references (§14.3 "when auth lands", §15.3
    "unauthenticated in M1") were already milestone-agnostic — no contradiction, left as-is.
- **Scope guard:** only `README.md` + `docs/DESIGN.md` (+ these kaizen files) touched — no `.py`,
  `QUERIES.md`, `test_queries.sh`, `bootstrap_schema.sh`, schema, or script changed; pytest 110
  and query suite 126/126 hold by construction (and were re-run green as the count source). The
  K-020 decision register (§1.1/§1.2/§1.3, AGENTS.md pointer index) was only *referenced*, not
  altered.

## 2026-07-05 — K-020: doc-architecture consolidation — DESIGN §1 single decision register

- **What (doc-only, no code/schema/query change):** applied the single-authoritative-home
  discipline (long applied to query bodies) to *design decisions*. `docs/DESIGN.md` §1 is now the
  one authoritative decision register; every other doc points to it.
  - **AGENTS.md decisions → DESIGN §1.2.** The 18-row "Decisions locked in" rationale table
    migrated into a new DESIGN **§1.2** detailed register (16 rows; `Message.role` inline + derived
    merged; "one graph per workspace" already lived in the §1.1 axes table). Each row is a
    statement + rationale + link to the body section (or QUERIES.md) that details the mechanics —
    no re-copied prose. AGENTS.md's section is now a terse two-column `Decision | Home` pointer
    index (rationale removed), kept — not deleted — as the quick do-not-reopen list.
  - **BACKLOG.md M2 stack → DESIGN §1.3.** The user-approved "Locked M2 stack decisions"
    (2026-07-04) graduated into a new DESIGN **§1.3** (embedding model/dim, agent LLM, runtime,
    VRAM, upgrade path); BACKLOG.md keeps a one-line pointer + the `EMBEDDING_DIM=1024` bootstrap
    reminder. K-0xx work items, sequencing, and parking lot untouched.
  - **A1 — GraphRAG dedup.** Deleted the drifted `cypher` block in DESIGN §8 (had lost its
    `LIMIT` and RETURN columns vs. the canonical QUERIES §6); §8 now points to QUERIES §6 in the
    §5.3 "shape-only, link the body" style. §8's design prose kept; QUERIES §6 untouched.
  - **A2 — coordination ADR promotion.** Added DESIGN **§6.3** (coordination is an M3 `WorkflowDef`
    of `kind:'process'`, not a flat `Task` node) with a back-link to `docs/archive/plans/m1-chat-mcp.md`
    Appendix B (which stays the ADR of record).
  - Added one new DESIGN §6.2 body line stating `ctx`/`input`/`output` are flat/serialised (D13).
- **Scope guard:** markdown docs only — no `repository.py`/`services.py`/`QUERIES.md` bodies/
  `test_queries.sh`/schema/scripts touched; pytest 110 and query suite 126/126 hold by
  construction. K-019 boundary respected (stale test counts, §13 "open"→"decided" wording, and
  §12/§14.1 scope left for K-019).

## 2026-07-05 — K-010: QA DEF-1 + DEF-2 closed (K-008 prerequisites)

- **What:** closed both defects from the K-007 QA pass, clearing K-008's gate. Coordinated
  delivery: **graph-dba** authored + live-verified the query layer, **tdd-engineer** wired the
  Python (strict red→green), verification re-run independently.
  1. **DEF-1 — member-id namespace guard (K-008 prerequisite).** Locked rule: member ids are
     **namespace-unique across `User`/`Agent`**. `ensure_user`/`ensure_agent` are now v2
     guarded-CREATE single-query bodies (QUERIES.md §2/§7, verified `Node By Index Scan` on
     both legs) returning an always-present `(created, existed, collided)` status row —
     idempotent re-ensure is a structural no-op; a cross-label collision writes nothing and
     raises `MemberIdCollisionError` (repository-level, re-exported by services);
     `existed AND collided` is a distinct corruption alarm. App startup with a configured
     actor colliding with an existing Agent id now **fails loudly** instead of silently
     minting a shadow `User` that eclipsed the Agent in every `coalesce(u, a)` lookup (the
     exact QA S3 repro). Same-label uniqueness constraints remain the concurrency backstop;
     the one-query-wide cross-label race window is documented, not closed (no engine
     cross-label constraint exists).
  2. **DEF-2 — fail-fast on unreachable FalkorDB.** `db.connect()` now passes
     `socket_connect_timeout`/`socket_timeout` (config-resolved `FALKORDB_CONNECT_TIMEOUT=5`
     / `FALKORDB_SOCKET_TIMEOUT=10`, env-overridable) and wraps failures in
     `FalkorDBUnreachableError` naming host:port + timeout + a start-script hint; a new
     `db.LazyFalkorDB` defers the first connection out of import — **importing
     `falkorchat.app` never touches the network** (the module-level `create_app()` used to
     hang ≥90s with zero output on WSL2's closed-port blackhole). Smoke re-verified: dead
     port → clean exit in ~6s with the actionable error. `app.py` docstring now matches
     reality.
- **Files:** `server/falkorchat/{repository,services,config,db,app}.py`;
  `server/tests/{test_repository,test_services,test_app}.py` + new `test_db.py`;
  `docs/QUERIES.md` §2/§7 (v2 ensures + contract table + locked rule);
  `scripts/test_queries.sh` (11 new DEF-1 assertions incl. PROFILE index checks);
  `AGENTS.md` (new locked-decision row; baselines) + root `AGENTS.md` (baselines).
- **Baselines (independently re-verified):** pytest **98 → 110**, query suite
  **115/115 → 126/126**; `reference` schema restored post-suite; `ws:acme` untouched.
- **Why:** DEF-1's silent misattribution was exactly the failure class K-007 closed, and it
  gated wiring real agent identities in K-008; DEF-2 bought dev/ops diagnosability on the
  README bare-`uvicorn` path (Compose was already shielded by `service_healthy`).

## 2026-07-05 — QA: acceptance pass on K-007 M2 groundwork

- **What:** black-box/acceptance QA pass at `94ab746`, scoped to what the K-007 dev suites
  structurally can't reach: concurrency through the real HTTP stack (single- and
  **two-process** writers), MCP-driven cursor paging over millisecond ties, agent `role` on
  every read surface, `backfill_thread_ids.sh` against real legacy-shaped data, and the
  actor-seam edges. Added `docs/archive/test-plans/k007-m2-groundwork.md` and
  `docs/archive/test-reports/k007-m2-groundwork-report.md`. Isolated `ws:qa` (created + deleted);
  `ws:acme`/`reference` untouched.
- **Result: PASS with two low-severity defects** — 18/18 items executed, 16 clean passes, on
  green baselines (server **98/98**, query suite **115/115**). Highlights: 12-way REST
  first-post hammer and a 20-write race across **two server processes** both yielded exactly
  one HEAD/TAIL and a contiguous chain; the cross-process run produced a **natural same-ms
  `createdAt` tie** and MCP cursor paging (`limit=3`) still delivered all 20 exactly once;
  agent-authored messages read `role: "assistant"` consistently on all five read surfaces;
  backfill script: 2 backfilled, then 0 (idempotent), `threadId: null` tolerated pre-backfill.
- **Defects (parked in `docs/BACKLOG.md`, not fixed here):**
  - **DEF-1 (low now, K-008 hazard):** no cross-label member-id uniqueness — a configured
    actor colliding with an existing `agentId` silently MERGEs a shadow `User` that eclipses
    the Agent in every `coalesce(u, a)` lookup (role derivation, `POSTED_BY`, mentions).
  - **DEF-2 (low, ops):** with FalkorDB unreachable, `uvicorn falkorchat.app:app` hangs
    indefinitely with zero output — `FalkorDB()` connects eagerly (no socket timeout) inside
    the module-level `create_app()`, falsifying the "building the app never requires a
    reachable FalkorDB" intent (hang-vs-refuse is WSL2-flavored; the eager import-time
    connect is real everywhere).
- **Why:** the prior QA report's top residual risks (concurrency/idempotency, agent
  authorship, ms-ties) were exactly K-007's targets — this pass closes that loop before K-008
  puts real agent writers on the system. No code under test changed.

## 2026-07-05 — K-007: M2 groundwork — agent authorship, v2 write-path guards, threadId denorm, composite cursors

- **What:** the six pre-agent-writer correctness/completeness items, landed per the approved
  plan (`docs/archive/plans/m2-groundwork.md`) over the graph-dba's live-verified query deliverable
  (`docs/archive/plans/m2-groundwork-queries.md`); plus the two server fold-ins.
  1. **Agent authorship** — §4 write paths resolve the author label-specifically (two indexed
     `OPTIONAL MATCH`es + `coalesce`), closing the `All Node Scan` *and* the silent no-op that
     made `Agent` authors unwritable; `services.post_message` derives `role` from the author's
     label via the new `repository.resolve_member_kinds` (`User → user`, `Agent → assistant`;
     replaces `existing_members` — one round trip for author + mention validation + role).
  2. **v2 self-guarding write paths (two reproduced defects fixed)** — each path wraps its write
     in a `FOREACH`+`CASE` guard inside the single `GRAPH.QUERY` and always returns a
     `(written, hadHead, dupMsg, authorFound)` status row (`repository.MessageWriteStatus`).
     Defect A: a same-`msgId` retry replay re-ran the relink clauses (NEXT self-loop, doubled
     `POSTED_BY`) — now a structural no-op reported as `dupMsg` = idempotent success. Defect B:
     two racing first-posts created two HEADs — the loser now refuses with `hadHead` and the
     service re-dispatches as subsequent (bounded 4-attempt loop; `Message` writes carry **no
     MERGE** — the uniqueness constraint stays as the verified all-or-nothing backstop).
     `REPLY_TO`-inside-the-guard live-verified in `test_queries.sh` (OQ4); repository fold-in
     waits for a reply surface.
  3. **`Message.threadId` denorm** — stamped inline by both write paths, deliberately unindexed;
     surfaced in §9.1/§9.2 since-reads, `/search`, and `GET /messages/{id}`. One-off
     `scripts/backfill_thread_ids.sh` (QUERIES.md §4.x; idempotent, HEAD-anchored, orphan
     caveat) — run against `ws:acme`: 0 backfilled (expected no-op, 0 messages).
  4. **Millisecond-tie correctness (reproduced page-boundary skip fixed)** — deterministic total
     order `(createdAt, msgId)` on both since-reads; formulation-A composite keyset predicate
     (still a bare `Node By Index Scan`); composite monotonic `ReadCursor`
     (`lastReadAt`, `lastReadMsgId`) — five scenarios verified, pre-K-007 cursors covered by
     `coalesce(…, '')`, no schema change; plus a lock-guarded monotonic per-process message
     clock in `Services` (same-ms ties impossible at the source). Explicit REST `?since=` keeps
     plain-`>` semantics (documented, OQ3).
  5. **TIMEOUT posture (docs-only, live-probed)** — keep legacy `TIMEOUT=1000`; per-query client
     override for future GraphRAG reads; **writes ignore TIMEOUT on this build** — bounded
     batches + input caps are the only write-path protection (DESIGN §10).
  6. **RAM line re-costed at 1024 dims (empirical)** — 12,387 B/message observed ≈ 12.4 KB ⇒
     ~1.25 GB per 100k-message workspace; `GRAPH.MEMORY USAGE` under-reports vector-index memory
     (size from `INFO memory` deltas) — DESIGN §11 rewritten; bootstrap default stays 1536 with
     an explicit choose-before-creation comment.
  - Fold-ins: `db.connect()` late-binds `config.FALKORDB_*`; `create_channel`/`create_thread`
    are plain `CREATE` (server-minted ids — creates documented **non-idempotent**;
    `create_thread` raises on a missing channel anchor).
  - Docs: QUERIES.md §2/§3/§4(+§4.x)/§5/§9 rewritten as the canonical v2 bodies; DESIGN
    §5.1/§5.3/§9/§10/§11/§12 (role values fixed to `user`/`assistant`, the falsified
    "idempotent via MERGE" claim replaced by the status-row contract); AGENTS.md decisions/
    facts/write-path rewrite; README + root AGENTS.md baselines.
- **Why:** prerequisites for AI agents writing concurrently (K-008): agents couldn't author at
  all, a client retry corrupted the thread chain, a first-post race forked it, and same-ms
  `createdAt` ties silently lost messages at cursor page boundaries.
- **Verified:** server suite **98 passed** (was 75; +23 — the plan's ≈95 estimate, exceeded by
  finer-grained regression tests); query suite **115/115** (was 92; +23 exactly as enumerated);
  `ruff check .` clean; defect regressions were watched fail red against the old code (replay →
  `(2 NEXT, 1 self-loop, 2 POSTED_BY)`; race → 2 HEADs) before the v2 queries landed; live
  8-worker concurrency hammer green (1 HEAD, 1 TAIL, contiguous chain of 8); backfill no-op
  proven on `ws:acme`.
- **Plan items:** K-007 ✅ done; K-008 (GraphRAG proper) unblocked; parking-lot fold-ins
  (`db.connect` bind, uuid `MERGE`) delivered. OQ6 (upstream FalkorDB filings: `GRAPH.MEMORY
  USAGE` vector under-report; one-shot instant-timeout anomaly) recommended to the user, not
  filed.

## 2026-07-04 — K-009: containerization (Dockerfile/compose) + CI + `falkordb-data` persistence fix

- **What:** first delivery-lifecycle pass for the component — container images, a compose stack,
  path-filtered CI, dependency pinning, and a critical data-persistence bug fix.
  1. **`falkordb-data` persistence fix (critical)** — `scripts/start_falkordb.sh` mounted the
     named volume at `/data` (the image's legacy `VOLUME`), but `falkordb/falkordb:edge` actually
     writes its Redis `dir` to **`/var/lib/falkordb/data`** (`FALKORDB_DATA_PATH`) — so **no graph
     data ever survived a container stop**; the volume persisted nothing. Live-verified 2026-07-04:
     data written under the `/data` mount vanished on restart; remounted at `/var/lib/falkordb/data`
     it survives. Fixed in the script (with an inline warning comment) and used in `compose.yaml`.
     `ws:acme` schema was re-bootstrapped after the fix (12 indexes).
  2. **`Dockerfile`** — M1 server image (`python:3.12-slim`): build context is the component root
     so the `server/` + `web/` sibling layout survives (app.py resolves `parents[2]/web`), editable
     install, non-root `appuser` runtime (install stays root-owned/read-only), `EXPOSE 8000`, and a
     `HEALTHCHECK` against the K-006 `GET /health` (200 only when FalkorDB answers).
  3. **`compose.yaml`** — two services: `falkordb` (same image/ports/volume as the script; redis-cli
     ping healthcheck) and `server` (built image, `FALKORDB_HOST=falkordb`, `depends_on:
     service_healthy`). The `falkordb-data` volume is declared **`external: true`** — compose must
     never create/re-create/remove the shared dev volume, and `down -v` is explicitly warned
     against. Header warns the script-started `falkordb-dev` container and compose share :6379 and
     the volume — never run both.
  4. **`.dockerignore`** — only `server/` (minus tests/venv/egg-info) + `web/` enter the build
     context; docs, kaizen, scripts, markdown excluded.
  5. **CI (`.github/workflows/falkor-chat.yml`)** — path-filtered to `falkor-chat/**` + the
     workflow itself; single job on ubuntu-latest with a **FalkorDB service container**
     (`falkordb/falkordb:edge`, health-gated) mirroring the local commands: `ruff check server` →
     server pytest (75-baseline) → `./scripts/test_queries.sh` (92/92-baseline). Deliberately
     tracks the floating `:edge` tag — the project's live-verified facts are pinned to it.
     **Never run yet** — first push to GitHub will tell (parking-lot item).
  6. **Dependency pins + ruff adoption** (`server/pyproject.toml`) — compatible-range pins for
     reproducible installs: `fastapi>=0.139,<0.140`, `uvicorn>=0.49,<0.50`, `falkordb>=1.6,<1.7`,
     `mcp>=1.28,<1.29`, `pytest>=9.1,<10`, `httpx>=0.28,<0.29`, `ruff>=0.14,<0.15`; ruff config
     (E,F,W,I / target py312 / line 100). Behavior-neutral import-order (I) fixes across
     `falkorchat/{api,app,services}.py` and `tests/{conftest,test_app,test_repository,test_services}.py`.
  7. **README** — compose run section added alongside the script path.
- **Why:** the component had no image, no one-command stack, and no CI; and the persistence bug
  meant the "durable" dev volume was silently empty — any container stop lost every graph.
- **Verified (2026-07-04 resume session):** fixed script started FalkorDB from a cold stop and
  `GRAPH.LIST` returned **`ws:acme`** — live proof graphs now survive downtime (`ws:k007scratch`
  residue also present, left untouched for the K-007 relaunch). Pins install-verified in a clean
  reinstall (fastapi 0.139.0, uvicorn 0.49.0, falkordb 1.6.1, mcp 1.28.1, pytest 9.1.1,
  httpx 0.28.1, ruff 0.14.14); `ruff check .` clean; server suite **75 passed**; query suite
  **92/92**. Compose stack itself not booted locally (shares :6379 + the volume with the running
  `falkordb-dev`); its build is exercised by CI on first push.
- **Plan items:** K-009 ✅ done; parking lot gains "verify the CI workflow goes green on first
  push". K-007 (graph-dba relaunch) is the next action.

## 2026-07-04 — K-006: post-M1 review follow-ups (navigation, bounds, health)

- **What:** small, high-value fixes from a 2026-07-04 full-project review; the review's larger
  findings went to the parking lot. Adapter/boundary changes only — no `QUERIES.md` query bodies
  or schema touched, so the 92-suite stays a pure regression guard.
  1. **MCP navigation dead-end closed** — `list_channels(limit)` + `list_threads(channel_id,
     limit)` MCP tools (7 total). Before, an agent could not discover an existing channel or
     thread id (workspace-wide `read_messages` rows omit `threadId` — still parked); it could
     only create its own space. Thin wrappers over the existing `Services` methods; discovery
     test updated, list→post→read navigation roundtrip added.
  2. **Input size bounds (RAM rule 6)** — `schemas.py` Pydantic constraints (text ≤ 8000,
     name/title 1–200, mentions ≤ 50) and `Query` bounds on list `limit`s (1–200). Message text
     lands in graph RAM *and* the full-text index; nothing capped it.
  3. **REST thread-read pagination** — `GET /threads/{tid}/messages?since=&limit=` maps to the
     existing §9.1 `read_thread_since` as a **pure read** (`since` defaults to 0 explicitly, so
     a browser poll never consults/advances the member's cursor — cursors stay agent-owned).
     No params keeps the full §4 read contract. Mitigates the unbounded `NEXT*0..` walk vs the
     1000 ms default `TIMEOUT` cliff on long threads (full fix = web client adoption, parked).
  4. **`GET /health`** — `services.ping` → `repository.ping` (`RO_QUERY RETURN 1`); 503 when
     FalkorDB is unreachable. Probe target for compose/CI (both parked).
- **Doc drift fixed (root `AGENTS.md`):** query-suite baseline claims corrected 67/67 → **92/92**
  (×2) — the stale numbers were loaded into every agent session.
- **Verified:** server suite **75 passed** (was 70; +5: MCP navigation roundtrip, health, body
  bounds, limit bounds, pagination — the pagination test injects a counting clock to sidestep the
  known same-ms `createdAt` tie caveat); query suite **92/92**.
- **Docs (same change):** `DESIGN.md` §14.4 REST table (+`/health`, real `?since=&limit=` shape,
  bounds note) and §15.2 tools table (+2 rows); `README.md` tools list + counts 70→75;
  `falkor-chat/AGENTS.md` count 68→75 (was already stale); `BACKLOG.md` parking lot extended,
  Last-reviewed bumped; this entry.

## 2026-07-02 — K-005: M1-final cleanup

- **What:** four small parking-lot items from the 2026-07-02 review, resolved test-first. All
  server changes are **adapter-only** (`mcp.py`, `api.py`) — no `repository.py`, `services.py`,
  `QUERIES.md`, or `test_queries.sh` touched, so the 92-assertion suite stays a pure regression
  guard.
  1. **`search_messages` MCP tool** — the existing `services.search_messages` (REST `GET /search`,
     `QUERIES.md` §5) is now exposed as a 4th MCP tool so agents can keyword-search too. Thin
     adapter; roundtrip test added.
  2. **`create_channel` MCP tool** (Q#4) — 5th tool; agents can now set up their own space
     (channel → thread → post → read) without any REST seeding. Discovery test asserts all 5
     names; full-flow roundtrip added.
  4. **Flat `GET /messages/{msg_id}` route** — replaced the nested
     `GET /threads/{tid}/messages/{mid}`, which ignored `tid` and let a message resolve under any
     thread's URL (a false contract). `Message.msgId` is workspace-unique and `Message` has no
     `threadId`, so resolution is workspace-global by design; the flat route states that truth.
- **Two fork decisions (spec §0):**
  - **Fork 3(a) — dead `isMention` highlight:** *remove it from the JS* rather than make §4 return
    a per-reader `isMention`. `isMention` is a since-read (§9) concept computed only by
    `read_thread_since`/`read_ws_since` (which take `me_id`); the reader-agnostic §4 thread read
    the web UI uses never sends it, so the highlight was dead-falsy. Making §4 reader-aware would
    mutate the locked §4 query, add a per-reader traversal to the hot thread-read path (RAM rule
    6), and force a 92-suite assertion change — not worth restoring a cosmetic highlight on a
    request/response M1 UI. Revisit in M2 with real-time since-reads.
  - **Fork 4 — nested single-message route:** *drop the thread-scoped spelling* for a flat
    `GET /messages/{mid}`. Validating thread membership would need an O(thread-length) HEAD/NEXT
    traversal on a route the web UI does not use, purely to keep a URL shape; the O(1) fix
    (denormalised `Message.threadId`) is a parked schema change (RAM rule 6). Leaving it as-is
    ships a wrong-thread-resolution trap.
- **Verified:** server suite **70 passed** (was 68; +1 search roundtrip, +1 create_channel flow;
  discovery + 2 api tests edited net 0); query suite **92/92** (untouched — regression guard).
- **Docs (same change):** `DESIGN.md` §15.2 tools table (+2 rows), §14.4 REST surface
  (`/messages/{mid}`), §14 test-count 68→70; `README.md` MCP tools list (+`create_channel`,
  +`search_messages`) and counts 68→70; `BACKLOG.md` pruned (4 completed items removed, Last
  reviewed bumped); this entry.
- **Batch B (delivered separately by another implementer):** the two `web/app.js` items —
  removing the dead `isMention` class toggle in `renderMessages`, and making the composer submit
  handler retry a mention-rejected send (`400 UnknownMemberError`) as plain text with a
  non-blocking notice so a typo'd `@handle` no longer drops the whole message. No test harness for
  the web JS; verified manually.

## 2026-07-02 — K-004: M1 hardening — five live-verified defects + QA DEF-1 fixed

- **What:** a full-project review probed the M1 server live (isolated `ws:probe` graph) and
  confirmed five defects the 57-test suite missed — every failing scenario involved state the
  fixtures always seeded (the actor) or parameter combinations never tested (`limit` + cursor).
  All fixed TDD (11 red tests → green):
  1. **Silent no-op writes (worst).** The §4 write queries anchor on `MATCH (author {userId:…})`;
     with the author node absent the whole write no-ops and REST still returned **201 with a fresh
     `msgId`** — on a fresh tenant (nothing ensures `u1`) every send "succeeded" and every thread
     stayed empty. Fix at three layers: `repository._assert_written` raises on zero-row writes;
     `services.post_message` validates the actor resolves to a member (`UnknownActorError`, one
     shared membership lookup with mentions); `create_app`'s lifespan runs `services.ensure_actor()`
     (startup, not import — building the app still needs no live FalkorDB).
  2. **Cursor-vs-limit message loss.** `read_messages` advanced the cursor to the *server clock*,
     permanently skipping rows a `limit` truncated (probe: 5 posted, `limit=2` read → next read 0).
     Fix: since-reads (§9.1/§9.2) are now **chronological** — the truncated page is a contiguous
     prefix — with reader-mentions carried by the `isMention` flag instead of the old
     mention-first sort (which + `LIMIT` is what made pagination lossy); the cursor advances to the
     newest **delivered** `createdAt` (empty page → no write). Ordering change synced in
     `QUERIES.md` §9 (+ rationale note), `test_queries.sh` (1:1 assertion swap), DESIGN §15.2.
  3. **`advance_cursor` IndexError** when the member node didn't exist (empty result indexed) —
     now a no-op returning `None`; noted in QUERIES.md §9.3.
  4. **QA DEF-1 (from the 2026-07-01 report) closed.** `POST /mcp` 405'd (Starlette Mount serves
     only `/mcp/`) — `create_app` adds an ASGI path-alias middleware rewriting `/mcp` → `/mcp/`;
     regression pinned by tightening the existing app test (it had tolerated 405 via `< 500`).
  5. **Search syntax-error 500.** RediSearch parse errors (`q='hello"x'`) surfaced as unhandled
     500s — `services.search_messages` maps `ResponseError` → `InvalidSearchQueryError` → 400.
  - Also: removed a duplicated gotcha comment in `repository.thread_has_head`; fixed the stale
    `exists((t)-[:HEAD]->())` advice in QUERIES.md §4 (contradicted the AGENTS.md live gotcha).
- **Verified:** server suite **68 passed** (was 57; +11); query suite **92/92** (assertion count
  unchanged — ordering assertions swapped 1:1); live probe script re-run: all five defects gone.
- **Docs (same change):** `QUERIES.md` §4 zero-rows + HEAD-check notes, §9 ordering rationale,
  §9.3 no-member note; `AGENTS.md` write-path invariants (+ zero-rows, chronological-cursor
  bullets) and test count; `README.md` counts + `/mcp` slash note; `DESIGN.md` §12/§15.
- **Plan items:** K-004 ✅. Review findings **not** fixed here parked in `BACKLOG.md` (agent
  authorship, `threadId` in §9.2 rows, retry idempotency + first-post race, web-UI mention
  polish, nested-route validation, ms-tie ordering, dependency pins, lint/CI).

## 2026-07-01 — QA: functional test pass on M1 (REST + MCP)

- **What:** first black-box/acceptance QA pass on the M1 server, driving the *running* process
  (curl over REST + a real `mcp` Streamable-HTTP client session) on top of the 57-test baseline.
  Added `docs/archive/test-plans/m1-chat-mcp.md` and `docs/archive/test-reports/m1-chat-mcp-report.md`.
- **Result:** 22/22 functional+contract items PASS · baseline 57/57. Verified both front doors over
  one service layer, error→status mapping (404/404/400), input validation (422), full-text search,
  read-cursor advance vs. explicit-`since` read-only, and REST↔MCP cross-door parity.
- **Defect found (DEF-1, low-med):** MCP endpoint 405s at `POST /mcp`; only `/mcp/` (trailing slash)
  completes the handshake — but README/DESIGN Appendix A advertise `/mcp`. Fix = alias/redirect
  `/mcp`→`/mcp/` **or** correct the docs, plus a regression test. See the report §3.
- **Feedback:** `bootstrap_schema.sh` seeds no members, so the mention happy-path needs manual seeding
  (consider a `seed_demo.sh`); per-endpoint response shapes vary (documented schema would make them
  testable); channel names non-unique. Details in the report §5.
- **Why:** first spin of the new `claude/qa-engineer` agent (proxy-run). No code under test changed.

## 2026-07-01 — K-003: M1 chat core finish — full-text search endpoint + web UI

- **What:** Closed out M1 chat core on top of the K-002 server, TDD and search-first.
  - **Full-text search (red→green per layer):** `repository.search_messages` (workspace-wide
    `db.idx.fulltext.queryNodes('Message', …)`, `QUERIES.md` §5 with the channel-scoping MATCH
    omitted) → `services.search_messages` (thin passthrough) → REST `GET /search?q=&limit=`
    (`q` required via `Query(..., min_length=1)`; `limit` bounded 1–200). **+5 tests** (2 live repo,
    1 fake-repo service, 2 TestClient incl. the `422` missing-`q` guard).
  - **Web UI:** minimal `web/{index.html, app.js}` — vanilla `fetch` over the same-origin REST API:
    channels list/create, threads list/create, thread messages + composer (parses `@id` handles into
    `mentions[]`), and a full-text search panel. HTML-escaped throughout.
  - **Serving:** `app.py` gained a `web_dir` param and mounts `StaticFiles(html=True)` at `/`
    **last** — `/` is a catch-all that must sit behind the REST routes and the `/mcp` mount
    (Starlette matches in registration order). Same-origin ⇒ no CORS. Mount is skipped if `web/` is
    absent. **+1 test** pinning "serves index at `/` **and** `/channels` still returns JSON."
- **Verified:** full server suite **57 passed** (was 51); query suite regression **92/92**. Smoke:
  assembled app serves the real `web/index.html` at `/`, `web/app.js` as `text/javascript`, and
  `/channels` JSON alongside — one process, three front doors (web, REST, MCP).
- **Docs (same change):** `DESIGN.md` §12 roadmap + §14.5 layout/serving note + §14.6 build order
  (steps 3–4 ✅); `README.md` roadmap/layout/run + "open http://localhost:8000/"; `AGENTS.md` server
  surface (static-mount-last rule, `/search`) and test count 51→57.
- **Plan items:** K-003 ✅ → **M1 chat core code-complete.** Parking lot now: `search` over MCP,
  `create_channel` over MCP (Q#4).

## 2026-07-01 — K-002 Step 2: M1 server (repository → services → MCP + REST), one process

- **What:** Built the first application code for the component (greenfield `server/` tree), bottom-up
  and test-first, completing K-002 (`docs/archive/plans/m1-chat-mcp.md`). All against live FalkorDB.
  - **`repository.py`** — every method 1:1 with a verified `QUERIES.md` query: channels/threads (§3),
    `ensure_user`/`ensure_agent` (§2/§7), both message write paths with the atomic `MENTIONS_MEMBER`
    block (§4), `read_thread` (§4), `read_thread_since` (§9.1), `read_ws_since` (§9.2),
    `advance_cursor`/`get_cursor` (§9.3/9.4), `get_message` (§4), plus validation reads
    (`thread_exists`/`channel_exists`/`existing_members`/`thread_has_head`).
  - **`services.py`** — invariants: id/clock generation (server clock), first-vs-subsequent write
    dispatch, mention validation (`UnknownMemberError`), RO/RW `read_messages` dispatch + `cursorId`
    construction, `Channel`/`ThreadNotFoundError`.
  - **`mcp.py`** — FastMCP adapter; tools `send_message`/`read_messages`/`create_thread`, injectable
    service + context (Q#1: `frm` ignored, actor = `get_context()`).
  - **`api.py` + `schemas.py`** — REST surface (DESIGN §14.4) incl. optional `mentions[]` parity;
    `ServiceError` → 404/400.
  - **`app.py`** — `create_app()` mounts REST + MCP on one FastAPI process.
- **Live gotchas found & mitigated (now in AGENTS.md):** (a) `exists((t)-[:HEAD]->())` returns `true`
  with no edge on this build and `count{}` is unsupported → existence via `OPTIONAL MATCH … IS NOT
  NULL`; (b) MCP lifespan wiring (python-sdk #1367) — forward `mcp_app.router.lifespan_context` to
  `FastAPI(lifespan=…)` or the session manager never starts; set `streamable_http_path="/"` so the
  mount lands cleanly at `/mcp`; (c) `call_tool` returns `(content, structured)` with list results
  wrapped as `{"result": […]}`.
- **Env:** no `uv` on the box → `server/.venv` via `python3 -m venv`; deps fastapi/uvicorn/falkordb
  1.6.1/mcp 1.28.1/pytest/httpx.
- **Tests:** **51 passed** — repository (24 live), services (12 unit fake-repo + 2 live), MCP (4
  in-memory), REST (7 TestClient), app-mount/lifespan (2). Query suite regression **92/92**.
- **Verified end-to-end:** REST round-trip through the assembled app; MCP tool discovery lists the
  three tools; mention-prioritised reads; monotonic cursor advance.
- **Plan items:** K-002 Step 2 ✅ → **K-002 complete.** Deferred: web UI (M1), `create_channel` over
  MCP (Q#4), full-text `search` REST endpoint.

## 2026-07-01 — K-002 Step 1 (gate): schema + queries for mentions & read-cursors

- **What:** Landed the graph-dba gate for the M1 Chat MCP transport (`docs/archive/plans/m1-chat-mcp.md`),
  all live-verified against `falkordb/falkordb:edge`. (1) `bootstrap_schema.sh`: added
  `ReadCursor.cursorId` range index + uniqueness constraint (index-before-constraint). (2)
  `QUERIES.md` §4: both message write paths now carry a `$mentions` list and append a
  `MENTIONS_MEMBER` write-block, atomically inside the single write query. (3) `QUERIES.md` new §9:
  `read_messages` since-reads — §9.1 thread-scoped, §9.2 workspace-wide, §9.3 monotonic cursor
  advance, §9.4 cursor read. (4) `test_queries.sh`: +25 assertions.
- **Q#2 resolved (member-match index strategy).** `GRAPH.PROFILE` showed `WHERE n.userId=$x OR
  n.agentId=$x` as a scan anchor degrading to an `All Node Scan`; the write path instead resolves
  each mention with dual `OPTIONAL MATCH (u:User)/(a:Agent)` + `coalesce` → two `Node By Index
  Scan`s. The `OR` form is kept only where `me`/`mem` is already bound (mention-flag, cursor read).
- **Two live gotchas found & mitigated (now in AGENTS.md):** (a) a bare empty `UNWIND` collapses the
  row stream, so `RETURN m` came back empty on a `$mentions=[]` post despite the writes committing —
  guarded with `UNWIND (CASE WHEN $mentions=[] THEN [null] ELSE $mentions END)` + a non-filtering
  `FOREACH`; (b) `collect(DISTINCT coalesce(u,a))` gives free dedup + unknown-skip and collapses the
  per-mention rows back to a single result row. Both proven: `$mentions=[]` is byte-identical to a
  plain post; `['u3','u3','a7','nope']` → 2 edges `[u3,a7]`, one row.
- **Corrections vs. the plan's candidate Cypher:** mention-flag match handles **Agent** readers
  (`me.userId=$meId OR me.agentId=$meId`, not `me {userId:…}`); author id returned via
  `coalesce(author.userId, author.agentId)` so Agent authors aren't null. §9.3 monotonic guard
  (`CASE WHEN $now > coalesce(rc.lastReadAt,0) …`) verified on this build (300 → stale 200 stays
  300 → 400).
- **RAM (rule #6):** +1 range index and +1 constraint per workspace; growth term is one `ReadCursor`
  node per *(member, thread)* read and one `MENTIONS_MEMBER` edge per mention. No new vector
  dimension → no embedding-RAM change.
- **Tests:** suite green at **new baseline 67/67 → 92/92** (+25: mention write-path incl.
  empty/dedup/unknown, §9.1 prioritised since-read + exclusion, §9.2 index-scan proof, §9.3
  monotonic/idempotent cursor + constraint block, §9.4 read + index-scan proof).
- **Plan items:** K-002 Step 1 ✅ (gate passed); Step 2 (repository → services → `mcp.py`/`app.py`
  → REST parity) unblocked.

## 2026-06-11 — K-001: `list_channels` query (list channels in a workspace)

- **What:** Authored and live-verified a `list channels` query and added it to `docs/QUERIES.md`
  §3 (Channels & threads), with assertions in `scripts/test_queries.sh`. Query:
  `MATCH (c:Channel) WHERE c.channelId > '' RETURN c.channelId, c.name, c.createdAt ORDER BY
  c.createdAt DESC LIMIT $limit`. The always-true `c.channelId > ''` predicate (every `channelId`
  is a non-empty string) anchors the listing on the **`Channel.channelId` range index** —
  `GRAPH.PROFILE` confirms `Node By Index Scan`, not `NodeByLabelScan`. Ordered by `createdAt`
  (channel **creation** time, newest-first), which is free once the scan is index-backed. Marked
  `GET /channels → list_channels` resolved in `DESIGN.md` §14.4 (was "gap — owned by graph-dba")
  and flipped the §14.6 prerequisite step to done.
- **Why:** the M1 REST surface (`GET /channels`, DESIGN §14.4) needed a verified query and
  `QUERIES.md` had none — it covered channel *members* (§2) and recent *threads* (§3) but not
  channels. Unblocks the `list_channels` repository method (§14.6 build order).
- **Trade-off noted:** true activity-recency (most-recent message/thread per channel) would need a
  `HAS_THREAD` → `Thread.updatedAt` expansion per channel — the Channel-level edge traversal §5.2
  deliberately avoids — so the cheap, index-backed **creation-time** ordering is used instead, and
  this is documented inline in `QUERIES.md`. No new index or constraint added; **zero per-workspace
  RAM cost** (reuses the existing `Channel.channelId` index).
- **Tests:** suite green at the **new baseline 64/64 → 67/67** (one §3 functional assertion +
  the standard §8 `assert_index_scan` pair; the plan's "65/65" estimate predated counting each
  `assert_*` call — the PROFILE proof is a two-line assertion per the existing §8 convention).
- **Plan items:** K-001 ✅ done.

## 2026-06-11 — Defined the M1 client/server application architecture

- **What:** Pinned the M1 application architecture and documented it as a new `docs/DESIGN.md` §14.
  Decisions: **transport = REST/JSON over FastAPI** (chosen after explicitly re-evaluating and
  rejecting gRPC — the only M1 client is a browser, so gRPC's typed-contract/streaming/service-mesh
  wins go unused and gRPC-Web would be pure bridge tax; WebSocket/SSE is the stronger M2 real-time
  path); **client = minimal web UI**; **real-time deferred to M2**; **single hardcoded tenant**
  (`ws=acme`, `user=u1`) injected at one FastAPI dependency seam so real auth drops in later without
  touching services/repo. Captured the layering (router → service → repository → db → FalkorDB),
  the REST surface → service → `QUERIES.md` mapping, proposed `server/` + `web/` layout, and the
  bottom-up TDD build order. Updated the §12 + README roadmap rows to point at §14.
- **Why:** User wanted the M1 client/server architecture nailed down before any code; the DESIGN
  doc previously only sketched the *operational* topology (§10), not the application code shape.
- **Plan items:** seeded **K-001** (`list_channels` query gap, owned by graph-dba — the one piece of
  the M1 REST surface with no verified query yet).

## 2026-06-11 — Adopted the kaizen plan/history convention

- **What:** Created `kaizen/{plan.md,history.md}` for the component, mirroring the sibling
  `claude/<agent>/kaizen/` projects (forward-looking backlog + dated change log). Replaced a
  short-lived `BACKLOG.md` draft with this structure.
- **Why:** User asked the component to track work the same way the sibling projects do, rather than
  a standalone backlog file.
- **Plan items:** K-001 recorded as the first active item.

## (prior) — M0 baseline

- M0 — Engine up: FalkorDB running (`falkordb/falkordb:edge`, Redis 8.2.2, module `999999`),
  design locked, schema bootstrap + canonical query library live-verified (`test_queries.sh`,
  64/64). Predates this log; see git history (e.g. `feat(falkor-chat): schema, query library,
  tests, and working context`) and `docs/DESIGN.md`.
