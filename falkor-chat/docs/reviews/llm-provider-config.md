# LLM Provider & Model Configuration — Design Review

> **Status:** active · **Owner:** `analyst` · **Tracks:** K-042 (M4) · **Version:** 2

## 1. Scope & verdict

**Reviewed together, as one design:**

- `falkor-chat/docs/plans/llm-provider-config.md` (`architect`, Version 1) — the resolver seam,
  config-file layering, precedence, publish-time validation, two-landing sequencing.
- `falkor-chat/docs/plans/llm-provider-config-graph.md` (`graph-dba`) — FR-8 trace recording,
  FR-16/FR-17 override storage, FR-19 index-dimension guard.

**Judged against:** `falkor-chat/docs/requirements/llm-provider-config.md` (FR-1..FR-20,
AC-1..AC-13 + decision log), `falkor-chat/docs/plans/llm-provider-config-coordination.md`,
`falkor-chat/docs/DESIGN.md` §1/§6/§7/§14, `falkor-chat/AGENTS.md`, root `AGENTS.md`, and the
live tree at `server/falkorchat/` (`config.py`, `llm.py`, `embedding.py`, `app.py`,
`executor.py`, `guards.py`, `services.py`, `responder.py`, `repository.py`, `tools.py`,
`api.py`, `background.py`, `mcp.py`) plus `server/tests/` and `server/.env.example`.

**What I executed** (nothing mutating; `./scripts/test_queries.sh` was **not** run):

- `.venv/bin/python -m pytest -q tests/test_services.py tests/test_app.py::test_default_app_wiring_is_gated_on_enable_agent tests/test_app.py::test_workflow_wiring_is_gated_on_workflow_enabled tests/test_llm.py tests/test_embedding.py` → **168 passed, 0.70 s** (baseline for finding M-2).
- Recomputed the `_drive_loop` SHA lock with the DESIGN §6.2 command → **`71055f756280`**, i.e. the lock is live on this tree and B-1 below is a real constraint, not a stale note.
- Probed `urllib` exception/timeout semantics and `urllib.parse.urlparse` on every `baseURL` form in play (findings B-2, M-6).
- Probed the running LM Studio on `localhost:1234` (read-only `POST` with `{}`) on four endpoint spellings (findings A-3, m-1).
- Read the two real `opencode.json` samples (`~/.config/opencode/opencode.json`, `opencode/agents/severino/opencode.json`) — the plan's §2.4/§2.5 grounding on both is **accurate**.

**Verdict: needs changes.** Two blockers, six majors. Both documents are unusually well-grounded
and the disagreement between them is small and honestly flagged; the blockers are gaps in the
*Landing-2 buildability* claim and in the *new transport's* failure taxonomy, and each is a
paragraph of plan text away from resolved. Landing 1 is otherwise ready to hand to an implementer
once M-1..M-4 are folded in.

---

## 2. The five adjudications you asked for

### A-1. Naming: `StepRun.model` vs `StepRun.resolvedModel` — **`resolvedModel` wins**

**Ruling:** adopt `graph-dba`'s names everywhere — graph property `resolvedModel`, repository
kwarg `resolved_model=`, API field `resolvedModel` on `GET /workflow-runs/{id}/step-runs`. The
plan's L2-2 / §5 (`docs/plans/llm-provider-config.md:395,443`) should be amended.

Three reasons, in order of weight:

1. **The plan already delegated it.** `docs/plans/llm-provider-config.md:395` says
   `record_step_and_advance` gains `model=` — *"**shape owned by `-graph.md`**"* — and §8.1
   (`:457-463`) asks the graph note for the field. The graph note owns the schema; its name
   governs. This is not a tie needing a casting vote.
2. **The collision is on one screen, not just in prose.** `read_step_runs` returns rows keyed by
   `stepKey` (`repository.py:1497`, QUERIES.md §12.8) and each `stepKey` maps to a `Step` whose
   opaque `config` carries the **requested** `model` (plan §2.8). A reviewer comparing "what the
   step asked for" against "what ran" would be reading two fields both called `model`, and they
   differ in exactly AC-9 and AC-10 — the two criteria that exist to make the difference visible.
3. **The plan's own vocabulary already leans that way** — `ResolvedModel`, `Resolution`,
   `last_used` (`:384`, `:442`).

**`modelSource`: adopt it too** (graph-dba offers it as optional, `-graph.md` §1.3). It costs
~10 bytes on LLM steps and no DDL, and it is the only thing that separates *"a fallback answered"*
from *"the workspace cap overruled the step"* — both of which manifest identically as
"`resolvedModel` ≠ the step's declared ref". Caveat to record with it: `modelSource` names the
**precedence rung only** (`workspace|step|default`), so it does **not** distinguish a role's
primary from its fallback — see m-7.

One knock-on: seam 5 in plan §6.1 (`:429`) calls the string `ResolvedModel.label`. Rename that to
`.ref` or `.resolved_ref` so the persisted field and the in-memory field share one word.

### A-2. The `{kind|"*"}` wildcard for FR-16 — **per-kind is a faithful reading, not a narrowing**

**Ruling: `graph-dba` is right to reject the wildcard; the conclusion stands, but the stated
justification is wrong and should not reach `tico` in its present form.**

FR-16's *"everything running in it"* is a **scope quantifier** — the whole workspace, as opposed
to one step, one agent or one guard — not an **arity claim** that one string governs all four
consumers. Two textual anchors in the approved requirement confirm this:

- FR-17's own precedence chain enumerates its middle rung as "the **step/agent/guard**'s own
  choice" and its last rung as "the **per-kind** default". The chain is already kind-indexed at
  two of three rungs; a kind-blind top rung would be the odd one out.
- FR-5 declares four independent namers whose model families are not interchangeable. A single
  string cannot be type-correct against a chat consumer and an embedding consumer at once.

So per-kind is a **refinement**, not a silent narrowing: it can express every *coherent* thing a
blanket could express (set the three chat kinds), and only loses the incoherent one.

**Correct the reasoning, though.** `-graph.md` §2.1 and §7-Q1 say a blanket override *"breaks
FR-19 by construction"* and that *"FR-16 and FR-19 would contradict each other."* That is
overstated. FR-19 would **work exactly as specified** — it would refuse to embed, loudly, forever.
The real objection is narrower and stronger: *a blanket override makes a self-contradictory
configuration expressible, whose only possible outcome is that the workspace can never embed
again.* Left as written, this reaches `tico` as "your approved requirements contradict each
other", which they do not.

**One residual for `tico` (not a blocker):** with per-kind overrides, "everything" is only as
complete as the kind set. Plan §3.1 (`:199`) already declares the set **fixed and closed**, which
closes the hole today; the manual (and ideally one sentence on FR-16) should record that adding a
fifth consumer kind means adding its override property, or the new kind silently escapes a
control the requirement calls a *hard cap*.

### A-3. The `/v1` normalization rule — **right in principle, insufficient as specified**

**Ruling: keep the rule; the plan needs three additions before it is safe.**

I re-derived every row of the §4.3 table with `urllib.parse.urlparse` and all five are correct.
I also re-confirmed the silent-failure premise live against the running LM Studio, and it is
worse than §2.3 records — it applies to the **embeddings** path too:

| Request | HTTP | Body (first bytes) |
|---|---|---|
| `POST /chat/completions` `{}` | **200** | `{"error":"Unexpected endpoint or method. (POST /chat/completions)"}` |
| `POST /v1/chat/completions` `{}` | 400 | `{"error": {"message": "No models loaded…"}}` |
| `POST /embeddings` `{}` | **200** | `{"error":"Unexpected endpoint or method. (POST /embeddings)"}` |
| `POST /v1/embeddings` `{}` | 400 | `{"error":"No models loaded…"}` |

Three gaps:

1. **No scheme/netloc validation** (see M-6). `urlparse("192.168.0.69:1234").path` is
   `'192.168.0.69:1234'` — **non-empty** — so a schemeless `baseURL` takes the "any non-empty
   path ⇒ verbatim" branch and sails through startup.
2. **Trailing-slash ordering is ambiguous.** §4.3 has no row for `http://host:1234/` (path `/` ⇒
   "append `/v1`"), and the rule's own strip clause is written as a *later* step, so a literal
   transcription yields `http://host:1234//v1`. The plan's §4.7 writes Anthropic's base **with**
   a trailing slash (`https://api.anthropic.com/v1/`), so the shape is in play in the plan's own
   text. Pin `http://host:1234/` and `https://api.anthropic.com/v1/` as explicit table rows.
3. **The rule makes falkor-chat interpret the shared file differently from OpenCode.**
   `@ai-sdk/openai-compatible` appends `/chat/completions` to `baseURL` verbatim — it infers no
   `/v1` — so on the stakeholder's own file falkor-chat will POST to `…:1234/v1/…` while OpenCode
   POSTs to `…:1234/…`. That divergence is what makes AC-1 pass (a literal reading gives an
   unusable provider), so it is the right call — but it must be **visible**: emit one INFO line
   per provider at startup naming the declared `baseURL`, the resolved API base, and whether the
   `/v1` rule or an overlay override produced it.

**And on FR-10's body-level detection:** the idea is right and the plan deserves credit for
finding it, but the taxonomy has two holes (blocker B-2) plus a shape hazard (m-1): the same
server returns `error` as a **string** on the wrong-prefix path and as an **object** on the right
one, so `body["error"]["message"]` raises `TypeError` in precisely the case FR-10 exists to
diagnose.

### A-4. FR-4 "one mechanism" vs. the surviving `llm=` injection — **faithful, with one hole to close**

**Ruling: the `StaticModelGateway` wrapping is a faithful satisfaction of FR-4, not a loophole.**

FR-4 forbids a consumer *"read[ing] endpoint/model settings by its own private route"*. Under
plan §3 (`:187-195`) every consumer holds a `ModelGateway` and has exactly one resolution path
inside itself; `llm=` is construction-time dependency injection that reads no file, no env var and
no `config.py` constant. The decision log's *"create an internal abstraction and use it
everywhere"* is satisfied: the abstraction is used everywhere, including on the injected path.
Keeping the 38 `llm=` / 24 `guard_judge=` sites untouched (I counted them: `grep -rn "llm=" tests/
falkorchat/` → **38**, `guard_judge=` → **24**; the plan's 37/23 is off by one each, immaterial)
is a genuine engineering win, not a compromise.

**The hole is elsewhere.** Landing 1 leaves `OpenAICompatibleLLM(base_url, model, …)` publicly
constructible with **required** args and no config source (L1-3, `:414`). A future wiring change
can hand a hand-built client through `llm=` — and that client's `base_url`/`model` must come from
*somewhere*, which is the private route FR-4 forbids. Two cheap closures, both in the plan's own
idiom (L1-5 already uses a `grep` as a done-condition):

1. **Add an enforcement test** to L1-4: no module outside `falkorchat/models.py` (and `tests/`)
   constructs `OpenAICompatible{LLM,Embedder}`. This is the FR-4 invariant made executable, and
   it is what makes "a future consumer gets the capability by using that mechanism" true rather
   than aspirational.
2. **Specify `StaticModelGateway.resolve(kind, requested=…, ws=…)`** — the plan never says what
   it does with a `requested` ref it cannot honour. If it silently ignores it, AC-4 ("each step's
   call goes to its own model") passes under a real gateway and **silently regresses to one
   model** under any `llm=` wiring. Ruling: ignore the ref, but log a WARNING once per
   `(kind, ref)` naming the ref and the fact that a statically-injected client is in use.

### A-5. Landing 1 / Landing 2 boundary — **§6.1 is right on 5 of 6 items; the sixth is load-bearing**

**Ruling: as specified, Landing 1 does *not* leave Landing 2 fully buildable.** Detail in blocker
B-1. Summary of the audit of plan §6.1 (`:419-432`):

| Seam | Verdict |
|---|---|
| 1. `Resolution.chain` tuple, `primary = chain[0]` | **Sound.** Return type is the `LLM` protocol either way; FR-18 becomes a wrapper swap. |
| 2. `resolve(..., ws=…)` threaded from day one | **Sound for 3 of 4 kinds** — fails for `guard` (B-1). |
| 3. No-`/` ref rejected with a Landing-2 message | **Sound.** Grammar is fixed by a real file (`opencode/agents/severino/opencode.json` names `lmstudio/qwen/qwen3-4b-2507`) — verified. |
| 4. `roles`/`agents` parsed-and-reserved | **Sound**, and the right call for AC-1. |
| 5. `ResolvedModel.label` populated in L1 | **Sound**; rename per A-1. |
| 6. *"Every resolution point already has `ctx.ws`/`ws` in scope"* | **False** — see B-1. |

Two further boundary items neither document covers:

- **The *inbound* carrier is unsolved.** `graph-dba` correctly identified that the *outbound*
  resolved model must ride on `StepResult` because `_drive_loop`'s `self._record(...)` call site
  is inside the SHA lock (`-graph.md` §1.5). The symmetric *inbound* problem — how the per-drive
  `WorkspaceOverrides` read (plan L2-3, `:444`) reaches `_execute_step` / `_select_transition` —
  is not addressed by either document. Storing it on `self` is a **thread-safety bug**: the
  executor is a process-wide singleton and every drive runs in the anyio worker threadpool
  (`api.py:105`, sync route + sync `BackgroundTasks`) or a bare `threading.Thread` (`mcp.py:71`).
  **The only lock-free carrier is the `run` dict** — `_drive` (outside the lock) already builds it
  from `repo.get_run(...)` and passes it to `_drive_loop`, which forwards it to both
  `_execute_step` (`executor.py:404`) and `_select_transition` (`:405-407`). Say so explicitly in
  L2-3, or the implementer will reach for `self` or for the lock.
- **`graph-dba` §2.6's read site is mis-located.** It recommends reading the overrides *"at
  `Executor.run` / `resume` entry, alongside the snapshot read that already happens there"*. The
  snapshot read is at `executor.py:376-378` — **inside `_drive_loop`, inside the lock**. `run`/
  `resume` do not read the snapshot. Correct the sentence to name `_drive` (`executor.py:339`),
  which is outside the lock.

---

## 3. Findings

### Blockers

**B-1 — The `guard` consumer has no workspace in scope, and the natural fix edits the SHA-locked
`_drive_loop`. FR-16/FR-17's hard cap cannot reach one of the four kinds as designed.**
*Owner: `architect` (plan §3.1, §6.1 item 6, L1-4, L2-3); `graph-dba` to note the inbound carrier
in `-graph.md` §2.6/§6.1.*

Plan §6.1 item 6 (`docs/plans/llm-provider-config.md:431`) asserts: *"Every resolution point
already has `ctx.ws`/`ws` in scope. Nothing in Landing 2 needs a new parameter to travel through a
function that does not already carry the workspace."* Verified against the tree, that is true for
three kinds and false for the fourth:

| kind | resolution point | has `ws`? |
|---|---|---|
| `agent` | `responder.maybe_respond(ctx, …)` (`responder.py:82`) | ✔ `ctx.ws` |
| `step` | `executor._run_agent_node(ctx, …)` (`executor.py:539`) | ✔ `ctx.ws` |
| `embedding` | `EmbeddingWorker.embed_message(ws, …)` (`embedding.py:86`); `tools.GraphRAGRetrieve.run(…, ctx=…)` (`tools.py:289`) | ✔ |
| `guard` | `guards.evaluate_guard(guard, *, ctx, run, step_output, thread, judge)` (`guards.py:181`) → `judge(condition, *, understanding, recent_turns, ctx, step_output)` (`app.py:388`) | ✘ **none** |

`evaluate_guard`'s `ctx` parameter is the **workflow run ctx dict**, not a `CallContext` — the
call site is `executor.py:805-808`, passing `ctx=run_ctx`. Its caller `_select_transition`
(`executor.py:769-772`) takes `(transitions, run, run_ctx, result)` and has no `CallContext`
either. The `run` dict from `repository.get_run` (`repository.py:1470-1495`) projects twelve
fields and **`ws` is not among them** — it cannot be, the workspace *is* the graph key.

Consequences the plan does not price in:

1. **Landing 1's L1-4** says *"`app._build_llm_judge(models)` resolves kind `guard` per
   evaluation"* (`:415`). It cannot pass `ws=`, so the judge either resolves with `ws=None`
   (harmless in L1, where `ws` is ignored) or the signature changes in L2 — which is exactly the
   "no new parameter travel" the plan promises not to need.
2. **Landing 2's FR-17 hard cap silently does not apply to `guardModelOverride`** —
   `-graph.md` §2.2 defines the property, but nothing can read it at the guard resolution point.
   AC-5 (guard default) would still pass; the *cap* would not, and the failure is invisible
   because a guard with no declared model resolves to the kind default either way.
3. Adding a parameter to `_select_transition` or `evaluate_guard` is legal (both are outside the
   lock), but **the `_select_transition` call site is inside `_drive_loop`**, which I recomputed
   as `71055f756280` — still locked on this tree. So the naive fix costs a deliberate lock reopen
   plus a SHA recompute, against DESIGN §6.2.

**Suggested fix (no lock reopen):** state in L1-4 that `_drive` (outside the lock) stamps
`run["ws"] = ctx.ws` and, in Landing 2, `run["modelOverrides"] = <the per-drive read>` **before**
calling `_drive_loop`; `evaluate_guard` already receives `run` and forwards nothing new is needed
below it, so the judge gains `ws=`/`overrides=` from `run` at `executor.py:805-808` — a change
entirely outside the lock. Whatever carrier is chosen, §6.1 item 6 must be rewritten to say
"three of four resolution points carry the workspace; `guard` is carried on the `run` dict",
because an implementer who believes the current sentence will discover this mid-Landing-2 with the
lock in the way.

---

**B-2 — The FR-10 failure taxonomy, as specified, leaves read timeouts unclassified and makes the
HTTP-status branch unreachable. FR-18's fallback inherits both holes.**
*Owner: `architect` (plan §4.9, L1-1).*

Plan §4.9 (`:358-364`) enumerates the classes `make_http_transport` must raise `ProviderCallError`
for, in this order: *"1. transport/connection failure (`URLError`, timeout); 2. HTTP error status
(`HTTPError`) …"*. Executed on this box's interpreter
(`server/.venv/bin/python`, 3.12):

```
HTTPError is subclass of URLError: True
socket.timeout is TimeoutError:    True
TimeoutError subclass of URLError: False
read-timeout raised: (<class 'TimeoutError'>, <class 'OSError'>, ...)
```

(the last line from a real `urlopen(req, timeout=0.5)` against a deliberately slow local HTTP
server).

Two defects follow:

1. **Ordering.** `urllib.error.HTTPError` **is** a subclass of `URLError`. Transcribed in the
   listed order, `except URLError` swallows every HTTP status error and class 2 becomes dead code
   — so a 401 from a cloud provider (the AC-2/AC-3 path) would be reported as "connection
   failure", with the response body — the only thing that says *why* — discarded.
2. **Read timeouts escape entirely.** A timeout during the **read** phase raises a bare
   `TimeoutError`, which is **not** a `URLError`. A transport catching only `URLError`/`HTTPError`
   lets it propagate unclassified: no `ProviderCallError`, no provider · model · URL in the
   message. This is not hypothetical — the plan *creates* this failure mode. §2.2 (`:56-62`)
   establishes there is **no timeout today**, and FR-14 + L1-1 add one with a 180 s default
   (§9.2). The first requirement the new timeout must satisfy is FR-10's "fails loudly", and as
   specified it does not.
3. **FR-18 inherits it.** L2-1 (`:442`) falls through the chain *"on `ProviderCallError` from
   element n"*. A hung endpoint — the single most likely reason to want a fallback — raises
   `TimeoutError` and therefore **does not trigger the chain**. AC-9's scripted case (endpoint
   unreachable → `ConnectionRefusedError` inside a `URLError`) would pass while the realistic case
   silently would not.

**Suggested fix:** rewrite §4.9's class list as an explicit, ordered `except` ladder —
`HTTPError` **first**, then `URLError`, then `(TimeoutError, OSError)`, then the JSON/body
classes — and add to L1-1's done-condition a `test_transport.py` case per branch, including an
opener that raises `TimeoutError` and one that raises `HTTPError`, each asserting a
`ProviderCallError` naming provider · model · URL. Also add `ValueError` (see M-6).

### Majors

**M-1 — `server/.env.example` sets all four FR-20 variables; plan §2.9 asserts no such site
exists.** *Owner: `architect` (plan §2.9 table), `devops` (U6).*

Plan §2.9 (`:141-143`): *"The four variables named by FR-20 are **only ever read as defaults in
`config.py`** — no script, compose file or CI sets them."* `server/.env.example:20,21,30,31` sets
`FALKORCHAT_EMBEDDING_BASE_URL`, `FALKORCHAT_EMBEDDING_MODEL`, `FALKORCHAT_LLM_BASE_URL`,
`FALKORCHAT_LLM_MODEL`, and its own header (`:5`) instructs *"copy this file to `.env` (and
`source` it) if you run uvicorn by hand"*. With the AC-13 tripwire in place, the shipped example
becomes a **startup brick**: follow the documented instructions and the server refuses to start.
The file is absent from §2.9's change table and from §5's file list.

(Two smaller corrections in the same neighbourhood: `compose.yaml` does **not** currently set any
of the four — the coordination record's blast-radius list is wrong about that, the plan is right.
And `README.md:138,266` reference only `FALKORCHAT_EMBEDDING_DIM`, which survives.)

**Suggested fix:** add `server/.env.example` to the §2.9 table and to L1-5's file list, replacing
the four lines with the two file-path variables; L1-5's `grep` done-condition already covers
`server/`, so it will catch a miss — but the plan must name the file so the change is designed,
not discovered.

---

**M-2 — "pytest is unchanged / the gateway is never built" is false: two existing tests drive
`_build_default_app()` with the agent flags on.** *Owner: `architect` (plan §4.1, §9.1, L1-4).*

Plan §4.1 (`:225-228`): *"With both off (the default, and the pytest baseline) the gateway is
never constructed."* Plan §9.1 (`:488`): *"**`pytest` is unchanged.** Both agent flags are off by
default, so the gateway is never built: no file is read."*

`server/tests/test_app.py:159-162` monkeypatches `config.ENABLE_AGENT = True` and calls
`app_mod._build_default_app()`; `:181-195` does the same for `WORKFLOW_ENABLED`. Both are
network-free **today** because `LMStudioLLM()` reads module constants. I ran them: green, 0.70 s,
no FalkorDB fixture requested. Under Landing 1 they would call `ModelGateway.from_env()`, which by
design **hard-fails when the shared file is missing** — and its default location is
`~/.config/opencode/opencode.json` (§4.1, `:220`), a per-user path outside the repo. Result: two
tests that pass on the stakeholder's box and fail on a clean clone or CI.

The risk is not the breakage; it is the *repair*. The obvious-looking fix — make a missing shared
file non-fatal when it "looks like a test" — would quietly defeat AC-13's whole posture.

**Suggested fix:** add to L1-4's done-condition a fixture strategy — a `conftest.py` autouse
fixture (or per-test `monkeypatch.setenv`) pointing `FALKORCHAT_OPENCODE_CONFIG` /
`FALKORCHAT_MODEL_CONFIG` at the `tests/data/` fixtures L1-2 already ships — and state the
done-condition as *"the full suite passes on a machine with no `~/.config/opencode/opencode.json`"*.
Separately, consider dropping the `~/.config/...` **default** entirely and requiring the variable
when a consumer is wired: a product default that points into a specific user's home is the
"works on my box" failure mode, and AC-13's posture is explicitness.

---

**M-3 — `tools.GraphRAGRetrieve` is a real FR-4 consumer, not a "type hint only" change; as
specified it bypasses the seam and is invisible to FR-16.** *Owner: `architect` (plan §3.1, §5).*

Plan §3.1 (`:206`) lists `tools.GraphRAGRetrieve` under kind `embedding`, *"resolved at per call
(`embed_message(ws, …)`)"* — but `GraphRAGRetrieve` never calls `embed_message`. It holds an
`Embedder` injected at construction (`tools.py:254-259`) and calls `self._embedder.embed(query)`
directly (`tools.py:292`). Plan §5 (`:396`) then books the change as *"`tools.py` (embedder type
hint only)"*.

As specified, the agent-node retrieval path keeps a **statically bound** embedder chosen at
`app._build_default_app` wiring time. That means (a) one LLM consumer does not resolve through the
seam, which is the letter of what FR-4 forbids, and (b) in Landing 2 the workspace's
`embeddingModelOverride` will govern `EmbeddingWorker` but **not** the retrieval query embed — so
a query vector and the corpus vectors could be produced by different models, which is a silent
retrieval-quality failure, not a loud one.

The good news: `GraphRAGRetrieve.run(arguments, *, ctx: CallContext, run)` (`tools.py:289`)
already has `ctx.ws`. **Suggested fix:** give `GraphRAGRetrieve` the gateway (or an
`embedder_for(ws)` callable) instead of an `Embedder`, resolve inside `run()`, and re-book the
`tools.py` row in §5 as a real change with its own done-condition. The same question applies to
`AgentResponder`'s own `self._embedder.embed(text)` (`responder.py:96`) — the plan does cover that
one (L1-4, "resolving `agent` + `embedding` inside `maybe_respond`"), so only `tools.py` is
missing.

---

**M-4 — The FR-9 publish pass is placed *before* the K-034 conflict check, and the plan's stated
rationale for the placement is wrong.** *Owner: `architect` (plan §2.7, L2-4).*

Plan §2.7 (`:123-126`) places the model-resolvability pass *"immediately after `_validate_def_spec`
returns"* and argues *"it preserves the 'last' rule by construction."* It does not.
`publish_workflow_def` (`services.py:878-935`) runs, in order: `_validate_def_spec` (`:906`) →
serialize → `read_def_structure` → **`_check_no_structural_conflict`** (`:922`) → `repo.publish_def`.
The K-034 conflict check is an existing invariant that runs **after** `_validate_def_spec`, so
inserting the new pass at `:907` puts it *ahead* of an older check — the precise inversion the
house rule in `_validate_def_spec`'s docstring (`services.py:787-795`) exists to prevent: *"an
older check must keep failing for its **own** reason."*

Concrete consequence: re-publishing a def whose topology changed **and** whose step names a
now-unresolvable model returns a 400 about the model instead of K-034's 409 "topology is
immutable" — and the 409 is the diagnostically important one, because it tells the author to
publish a new version rather than fix a name.

**Suggested fix:** move the pass to immediately **before** `self._repo.publish_def(...)`
(`services.py:932`), i.e. after `_check_no_structural_conflict`. Nothing is written either way, so
this costs nothing and restores the rule. Keep the rest of L2-4 as written — the use of
`_normalize_opaque` (`services.py:187`) is correct and load-bearing (the M-7 defect), and skipping
the pass when `Services` has no gateway is the right call (see m-8 for one refinement).

---

**M-5 — `FallbackClient.last_used` is mutable per-client state read after the call; concurrent
drives can cross-report the resolved model.** *Owner: `architect` (plan L2-1, L2-2).*

Plan L2-1 (`:442`) specifies *"`FallbackClient` … Exposes `.last_used: ResolvedModel` for FR-8"*,
and L2-2 (`:443`) reads it back (`FallbackClient.last_used.label`). That is a read-after-write on
mutable object state. It is safe **only** if a fresh `FallbackClient` is minted per resolution —
which the plan never states, and which §3.1's "resolved per call" implies but does not promise
(a gateway that caches clients per `(kind, ref)` would be the obvious optimisation and would
introduce the bug).

The concurrency is real, not theoretical: `api.py`'s routes are sync (`def post_message`,
`api.py:84`) and its `BackgroundTasks` are sync callables, so every drive runs in Starlette's
anyio worker threadpool; `mcp.py:71` additionally spawns a bare daemon `threading.Thread` per
posted message. Two runs in the same workspace can be inside `FallbackClient.chat` at the same
instant, and the loser's `StepRun.resolvedModel` records the winner's model — an audit field that
lies, which is worse than an absent one.

**Suggested fix:** adopt `graph-dba` §6.2's carrier instead — put `model: str | None = None` on
`ChatResult` (`llm.py:46`, already a frozen dataclass, so the field is additive and default-safe)
and have the client that made the successful call populate it. The value then travels on the
return value, never on shared state. If `.last_used` is kept for any reason, the plan must state
"one `FallbackClient` per resolution, never cached" as an invariant with a test.

---

**M-6 — No scheme/netloc validation on `baseURL`; a malformed one takes the "verbatim" branch and
fails at call time outside all four classified error classes.** *Owner: `architect` (plan §4.3,
L1-2).*

Verified with `urllib.parse.urlparse`:

| declared `baseURL` | scheme | netloc | path | §4.3 rule says |
|---|---|---|---|---|
| `192.168.0.69:1234` | `''` | `''` | `'192.168.0.69:1234'` | **non-empty path ⇒ verbatim** |
| `localhost:1234/v1` | `'localhost'` | `''` | `'1234/v1'` | **non-empty path ⇒ verbatim** |

Both are plausible admin typos (dropping `http://` is the single most common `baseURL` mistake),
both pass startup silently, and both then fail at first call with
`ValueError: unknown url type` from `urlopen` — which is neither an `HTTPError`, a `URLError`, a
timeout, nor a body-level error, so it escapes the §4.9 taxonomy entirely and reaches the M-1
fault net as `unexpected: ValueError(...)`. FR-9's "reject at publish" and AC-13's "the failure is
explicit" posture both argue for catching this at **load** time.

**Suggested fix:** in L1-2's parser, after substitution and before normalization, require
`urlparse(base).scheme in {"http","https"}` **and** a non-empty `netloc`; otherwise raise
`ModelConfigError` naming the provider, the file and the offending value. Add the two rows above
to the §4.3 test table as *rejected*, and add `ValueError` to the transport's ladder as a
belt-and-braces class.

### Minors

**m-1 — The body-level error classifier must tolerate `error` as a string *or* an object.**
Verified live on one server: the wrong-prefix 200 returns `{"error": "<string>"}` while the
correct-prefix 400 returns `{"error": {"message": …}}`. A classifier written as
`body["error"]["message"]` raises `TypeError` in exactly the case §4.9 class 3 was written for.
Specify: detect on **key presence**, then render `str(err)` when it is not a mapping. *(architect,
§4.9 / L1-1 done-condition.)*

**m-2 — One global 180 s default timeout interacts badly with the anyio threadpool.** Every posted
message schedules `_safe_embed` (`api.py:97`) on the threadpool, whose default capacity is 40
tokens; all REST routes are sync and share it. A stalled LM Studio can therefore pin the pool for
180 s at a time and stall unrelated reads. This is still a **strict improvement** over today's
unbounded wait, so it is not a defect — but per-kind defaults are nearly free and much better
behaved: e.g. `embedding` ≈ 30 s (a short, predictable call on the hot path), `agent`/`step`/
`guard` 180 s. Also note `mcp.py:71`'s unbounded thread-per-message: 180 s timeouts there grow
threads under load. *(architect, §9.2 / `config/models.json`.)*

**m-3 — FR-14's per-model settings are open-ended; the plan's key set is closed.** FR-14 says
*"at minimum request timeout, plus generation settings **such as** temperature and max tokens"*.
L1-2 fixes the set at `timeout`, `temperature`, `maxTokens`, `dim`. Prefer: reserve
`timeout`/`dim`/`protocol` and pass every other key through into the request payload
(camelCase → snake_case), so `top_p`, `max_completion_tokens` (which newer OpenAI models require
*instead of* `max_tokens`), `reasoning_effort` etc. need no plan revision. *(architect, L1-2/L1-3.)*

**m-4 — FR-19's pre-flight should not check `Chunk` at embed time.** `-graph.md` §3.3 requires both
`Message` and `Chunk` dimensions to be checked. `Chunk` appears in **no** server module
(`grep -rn "Chunk" server/falkorchat/*.py` → one comment in `config.py:32`); nothing in the app
writes `Chunk.embedding` today. Checking it on the `Message` write path turns a divergent-but-unused
`Chunk` index into a refusal to embed messages the workspace would happily accept. Put the `Chunk`
check in layer 1 (the startup assertion, where it belongs as an operator warning) and gate each
write on **the label being written**. *(graph-dba, `-graph.md` §3.3; architect, L2-6.)*

**m-5 — `modelSource` cannot distinguish a role's primary from its fallback.** With
`modelSource ∈ {workspace, step, default}`, an operator reading the trace after an AC-9 event sees
`source='step'` and a model the step never named, with nothing saying a degradation occurred.
Either document that explicitly (fallback occurrence is visible only in logs / debug
`TraceEvent`s), or add a separate boolean. Do **not** overload `modelSource` with a `fallback`
value — the rung and the degradation are orthogonal. *(graph-dba, `-graph.md` §1.3/§6.2.)*

**m-6 — A single `StepRun.resolvedModel` cannot represent a step that answered on two models.**
`_run_agent_node` loops up to `config.maxIterations` calling `self._llm.chat` (`executor.py:585`);
with an FR-18 chain, iteration 1 can answer on model A and iteration 3 on model B. This is the same
"one scalar, many calls" problem `-graph.md` §1.6 correctly identified for guard judges, and it
should get the same treatment: pick a rule (recommend **last answering model wins**), state it, and
record it next to §1.6 so the successor does not re-litigate. *(graph-dba + architect, L2-2.)*

**m-7 — FR-9 becomes deployment-dependent when `Services` has no gateway.** L2-4 (`:445`) skips
the pass entirely in that case — correct, since nothing can be resolved — but it means AC-7 holds
on a wired server and silently does not on an unwired one. Add: when the pass is skipped **and**
the def declares any `config.model` or llm-guard `model`, log a WARNING naming the def and the
identifiers, so "validation didn't run" is never invisible. *(architect, L2-4.)*

**m-8 — `server/falkorchat/models.py` is a colliding module name.** In a FastAPI codebase
`models.py` reads as "pydantic/ORM models", and this repo already has `schemas.py` for exactly
that. The new module holds LLM-provider configuration, where "model" means something else again.
`modelconfig.py` or `gateway.py` costs nothing to choose now and something to change later.
*(architect, §5.)*

**m-9 — Citation drift (cosmetic, no action needed beyond a pass at revision time).**
`_build_default_app` is `app.py:244` (plan says `:245-303`); `_drive_loop`'s tracer line is
`executor.py:390` (plan says `:388`, already noted by `teco`); `_normalize_opaque` is
`services.py:187` (plan says `:186-205`). Injection counts are 38/24, not 37/23 (I counted).
Everything substantive I spot-checked — `executor.py:585`, `:821`, `:848`; `services.py:776`,
`:906`; `repository.py:1301`; `embedding.py:59-60`, `:84`; `llm.py:99-100`, `:110`;
`repository.py` `set_embedding`'s `config.EMBEDDING_DIM` default — is **exact**.

---

## 4. What's solid

Worth protecting through the revision, because a lot of it is unusually good:

- **The FR-8 mechanism.** Both documents independently reached "`StepRun` property, not
  `TraceEvent`", from the same evidence (`executor.py:390`), and both are right. `-graph.md`'s
  §1.4 Cypher matches `repository.py:1301-1339`'s real query structure exactly — I compared them
  clause by clause — and the `NULL`-param-omits-the-property verification is the detail that makes
  "nullable and absent by default" free rather than a compromise.
- **`-graph.md` §3.1/§3.2.** Re-probing the pinned build rather than trusting the quirks KB (which
  said the dimension was *not* introspectable, and was wrong for this build) turned L2-6 away from
  error-message parsing and onto `db.indexes()`. That correction is the single highest-value fact
  in either document, and the "re-running `bootstrap_schema.sh` with a new `EMBEDDING_DIM` silently
  does nothing" finding reframes FR-19 as a guard against a real operator error.
- **`StaticModelGateway`.** Absorbing 62 injection sites with zero test churn, and the conditional
  `model=` kwarg for guards, is the right instinct and the reason this is a two-week change rather
  than a two-month one. See A-4 for the one closure it needs.
- **§4.4's "resolve on the provider, not the `models` map."** Grounded in the stakeholder's real
  file (one model listed, seven served — I re-read the file and confirm it), and the asymmetry it
  creates (bad provider → publish-time, bad model id → call-time) is stated plainly rather than
  papered over.
- **The `/v1` live probe.** §2.3 found a genuinely silent failure mode that would otherwise have
  shipped as `KeyError: 'choices'`. I reproduced it and it is worse than recorded (the embeddings
  path too). The design's response — a declared rule plus an always-available per-provider override
  that keeps the shared file pristine — is the right shape.
- **`-graph.md` §2.3's option-D treatment.** Arguing the rejected alternative at full strength,
  and naming the condition under which `architect` should flip the decision, is how a design note
  should handle a close call.
- **The two-landing split** is drawn in the right place: Landing 1 is genuinely demonstrable alone,
  and needs **no graph change at all**.

---

## 5. Open questions

1. **For `tico` (requirements clarification, one sentence):** FR-16's "everything running in it"
   is implemented as four per-kind overrides (A-2). Should FR-16 record that "everything" is
   scoped to the **closed set of consumer kinds**, so that a future fifth consumer must add its
   own override property rather than silently escaping the hard cap?
2. **For the stakeholder / `architect`:** adopt `modelSource` (A-1)? It is not required by any AC,
   costs ~10 bytes per LLM step and no DDL, and is the only thing that makes "the workspace cap
   overruled this step" readable rather than inferable. My recommendation is yes.
3. **For `architect`:** is `FALKORCHAT_OPENCODE_CONFIG`'s default of `~/.config/opencode/opencode.json`
   wanted, or should the variable be required when a consumer is wired (M-2)? This is a
   product-posture call, not a correctness one.
4. **Already settled, not reopened here:** FR-10's "suspends" → `failed`-with-cause (stakeholder,
   2026-08-10); AC-2/AC-3 deferred/model-gated — I checked that the design still *supports* hosted
   providers and `{env:}`/`{file:}` substitution (plan §4.8, §4.7) and it does, with one
   observation: substitution is scoped to the `provider.*` / `providers.*` subtrees only, which
   covers every place a credential can appear today. Native Anthropic Messages API remains a
   declared non-goal.

---

## Pass 2 — 2026-08-10, re-gate of Version 2

Both documents were re-read **in full**, not diffed against Pass-1 memory, per the coordinator's
instruction. Every load-bearing claim in both revision notes was independently re-verified against
the live tree, the running interpreter, and (where the constraints allow) the shared FalkorDB
instance — not accepted on report. What I ran or read for this pass:

- Re-read `docs/plans/llm-provider-config.md` (Version 2, 837 lines) and
  `docs/plans/llm-provider-config-graph.md` (Version 2, 1081 lines) end to end.
- Re-read `server/falkorchat/executor.py:296-436,769-812` and `guards.py:150-233` directly, to
  trace `run()`/`resume()` → `_drive` → `_drive_loop` → `_select_transition` → `evaluate_guard`
  by hand rather than trust §4.10's table.
- Recomputed the `_drive_loop` SHA lock with DESIGN §6.2's command → **`71055f756280`**, unchanged
  from Pass 1 — still live on this tree (`git status` confirms zero source changes since Pass 1).
- Re-ran the `urllib` exception-ladder probe (`HTTPError`/`URLError`/`TimeoutError`/`ValueError`
  dispatched through the exact ordered `except` clauses §4.9 now specifies).
- Re-derived the plan's own §11.2 rebuttal of my Pass-1 M-6 (`urlopen` on a schemeless URL) —
  reproduced `URLError`, not the `ValueError` I originally claimed.
- Re-ran `grep -rn "Chunk" server/falkorchat/*.py` (m-4) and
  `grep -n "FALKORCHAT_LLM_\|FALKORCHAT_EMBEDDING_BASE_URL\|FALKORCHAT_EMBEDDING_MODEL" compose.yaml`
  (item 7a) — both fresh, both against the unchanged tree.
- Grepped both documents for naming stragglers (`StepRun.model`, `.label`, bare `model=`) and for
  every mention of `modelFallback`/`model_fallback`/`ChatResult`/`StepResult`.
- Re-ran `./scripts/verify_workflows.sh acme` before finishing → `OK — 2 defs in sync`. No
  `./scripts/test_queries.sh`. No file written outside `docs/reviews/`. No git command that
  mutates the tree.

**LM Studio was not running during this pass** (`curl` → connection refused, port 1234 has no
listener). I could not literally re-execute the `/v1`-prefix and string-vs-object `error` probes
live a second time; I rely on Pass 1's captured evidence (which both v2 documents also cite
verbatim, consistently with what I recorded) rather than re-asserting it as freshly observed. Flag
this rather than silently reuse it.

### Pass 2 verdict: **needs changes** — one new blocker, everything else from Pass 1 closes clean

Six of the seven items I was asked to adjudicate close cleanly and correctly. One does not: **the
two v2 documents now disagree on the `StepRun` schema itself** — a real, checkable regression
introduced *by* the revision, not a carryover from Pass 1. Separately, one stale passage inside
`-graph.md` still asserts the overstated reasoning its own §2.1 just withdrew. Both are new,
both are concrete, both are cheap to close.

#### 1. B-1 — CLOSED, verified by direct trace, not by report

The plan's §4.10 claim is **exactly right**, and in one respect *better* than it needs to claim
credit for. Traced by hand, not accepted from the table:

- `run()` (`executor.py:300-306`) and `resume()` (`:309-331`) each build `run` fresh from
  `self._repo.get_run(...)` and both call `self._drive(ctx, run)` — confirmed the **single**
  convergence point claim.
- `_drive` (`:339`) runs `run_id = run["runId"]`; `run_ctx = _load_json_obj(run["ctx"])`; `try:
  return self._drive_loop(ctx, run)`. The proposed `run["ws"] = ctx.ws` stamp lands in this body,
  **before** the `try:` — entirely outside `_drive_loop`, which the recomputed SHA lock
  (`71055f756280`, unchanged) confirms spans only `executor.py:375-435` (`awk` bounded on the
  `def _drive_loop` / `# ── seams` markers, same command as Pass 1).
- `_drive_loop` (`:397-399`) forwards the **same** `run` object to `_select_transition` — no new
  parameter, no copy.
- `_select_transition` (`:805-808`) already calls
  `evaluate_guard(guard, ctx=run_ctx, run=run, step_output=result.output, thread=result.thread,
  judge=self._guard_judge)` — **`run=run` is passed today**, on the unmodified v1 code. Better
  than the plan states: this isn't a change to land, it already exists.
- `guards.evaluate_guard`'s signature (`guards.py:185`) already **declares** a `run:
  dict[str, Any]` parameter — and, checked by grepping the function body, **never reads it**. It
  is a currently-dead, already-plumbed-through parameter. The `accepts_run` mechanism the plan
  proposes is therefore not new plumbing; it's finally using plumbing that's already there.

Net: **zero edits inside the SHA-locked body**, confirmed by direct inspection of the boundary,
not by trusting the plan's own table. `-graph.md` §2.6's corrected placement (`_drive`, `:339`,
outside the lock, `run["ws"]`/`run["modelOverrides"]`) matches the plan's §4.10 exactly — the two
documents agree on this mechanism byte-for-byte. **B-1 is resolved.**

#### 2. B-2 — CLOSED, both holes empirically closed by the new ladder

Re-ran the exact ordering (`HTTPError` → `URLError` → `(TimeoutError, OSError)` → `ValueError`)
against one instance of each exception type:

```
HTTPError instance -> HTTPError-branch
URLError instance  -> URLError-branch
bare TimeoutError  -> Timeout/OSError-branch
bare ValueError    -> ValueError-branch
```

Both Pass-1 holes are closed: `HTTPError` is no longer dead code (it's caught before the parent
`URLError` clause reaches it), and a bare `TimeoutError` — which is *not* a `URLError` subclass,
independently reconfirmed — has its own explicit rung and no longer escapes unclassified. The
body-level `error`-as-string-vs-object renderer (m-1, `msg = err.get("message", str(err)) if
isinstance(err, Mapping) else str(err)`) is specified correctly in §4.9 rung 5 and matches the
Pass-1 evidence I couldn't re-probe live this pass (see the LM-Studio-down note above). **B-2 is
resolved.**

One correction to my own Pass-1 finding, which the plan caught and I confirm: M-6's *remedy* (load-
time scheme/netloc validation) was right, but the *failure mode* I cited was wrong — a schemeless
`baseURL` (`"192.168.0.69:1234"`, `"localhost:1234/v1"`) raises **`URLError: unknown url type`**,
not `ValueError`, when passed to `urlopen`. Reproduced fresh this pass:

```
urlopen('192.168.0.69:1234/v1/chat/completions') -> URLError: <urlopen error unknown url type: 192.168.0.69>
urlopen('localhost:1234/v1/chat/completions')    -> URLError: <urlopen error unknown url type: localhost>
```

So it would in fact have been caught by rung 2, with a body-free but provider/URL-bearing message
— less bad than I originally claimed, though still worse than catching it at load time with the
`urlparse`-based check the plan adopted anyway. The plan's §11.2 rebuttal is accurate; my original
citation was wrong. Correctly logged as a partial rebuttal, not silently absorbed as a full
concession — this is the right way to handle a reviewer error and I have nothing to add to it.

#### 3. m-4 / m-5 / m-6 — two right, one **incompletely landed**

**m-4 (Chunk gating) — sound design, adopted correctly.** `grep -rn "Chunk" server/falkorchat/*.py`
re-run fresh: still exactly one hit, a comment (`config.py:32`). Gating layers 2/3 (the write-time
checks) on the label actually being written, while keeping both labels in layer 1 (the startup
assertion, an operator warning) is the right shape — it doesn't block a real write over an
unrelated, unused index, and the forward-compatibility note (the introspection query is already
`$label`-parameterised, so a future `Chunk` writer inherits layers 2/3 for free) is a genuine
design improvement over what I asked for, not just a restatement. No note beyond confirming it.

**m-6 (multi-call "last wins") — sound, and consistently landed in both documents.** `-graph.md`
§1.6's addendum and the plan's L2-2 both state the identical rule ("last answering model wins"),
with the graph note additionally flagging the implementation hazard correctly (a "set once on
first resolution" implementation would silently record the wrong model — the loop must overwrite
pending values each iteration and read them once after the loop exits). This is good design
craftsmanship, not just a restated finding.

**m-5 (`modelFallback`) — the graph note's design is sound; the plan never adopted it.** This is
the substantive new finding of this pass — see §P2-B below.

#### 4. A-2's overstated framing — softened in the two places that matter, **but one stale
restatement survives inside the same document**

`-graph.md` §2.1 and §7-Q1 correctly withdraw the overstated *"breaks FR-19 by
construction... the two requirements would be in direct contradiction"* claim, replace it with the
narrower, stronger argument (a blanket override makes an *incoherent configuration expressible*,
whose only reachable outcome is a permanently-dead embedding path — not a requirements
contradiction), and withdraw the stakeholder escalation. The plan's own §8 restatement (*"the
wildcard is correctly rejected — adjudication A-2"*) never repeats the overstated language at all.

**But `-graph.md` §6.5 (its own §8.2 answer, `-graph.md:927-934`) was not touched by the revision
and still reads:**

> *"A `"*"` blanket that includes the embedding kind forces the embedding worker onto a chat model
> and **breaks FR-19 by construction** (§2.1) — the two requirements would be in direct
> contradiction."*

This is the exact sentence §2.1 just withdrew, verbatim, **citing §2.1 as though it still supports
the claim it now contradicts**. It's a real, checkable internal inconsistency — a reader who
starts at §6.5 (a plausible entry point; it's the "answers to the other document's §8" section)
gets the overstated, withdrawn reasoning and a citation pointing at the correction that
contradicts it. *Routed to: `graph-dba` — a one-paragraph fix, replace §6.5's sentence with §2.1's
corrected argument or a forward reference to it.* Minor severity (the *conclusion* — per-kind, not
blanket — is right and consistent everywhere it's acted on; only the stated *reason* in one
passage is stale), but real, and it's precisely the kind of thing "don't trust the revision note's
own completeness claim" is asking me to catch.

#### 5. Naming (A-1 + `modelFallback`) — consistent **except** the field the graph note added

`resolvedModel` / `resolved_model=` / `modelSource` are used consistently throughout both
documents with no `StepRun.model` stragglers as *live* recommendations — grepped both files for
`StepRun\.[a-zA-Z]+`; the only `StepRun.model` hit is inside `-graph.md`'s §1.3 "Amendment
requested" block, which is a historical quotation of what v1 the plan needed to change, correctly
preserved for traceability, not a live claim. `.label` → `.ref` is likewise clean — the only
`.label` hits are the changelog line and the disposition-table row recording the rename.

**`modelFallback` is not clean — see §P2-B below.** It's a genuine third naming/schema state, not
a stragglers-grep miss.

#### 6. Inbound workspace-override carrier — CLOSED, and consistent between documents

Both v2 documents specify the identical mechanism: `run["ws"] = ctx.ws` and
`run["modelOverrides"] = self._repo.read_model_overrides(ctx.ws)`, stamped in `_drive`
(`executor.py:339`) before `_drive_loop` runs, explicitly rejecting `self` as a carrier (correctly
— the executor is a process-wide singleton driven from the anyio threadpool and from `mcp.py`'s
daemon threads, verified unchanged since Pass 1). The plan's L2-3 additionally covers the two
run-less consumers (responder, embedding worker: "read once per drive / **per responder call**")
and its file list includes `responder.py`, `embedding.py`, `tools.py` — less rigorously
line-cited than the guard path (understandably: those paths don't cross the SHA lock and already
carry `ctx.ws` directly), but not missing. **This is now a solved design problem**, not an open
one — it should not be reopened as its own unit before Landing 2 starts.

#### 7. Two independent spot-checks

**(a) `compose.yaml` sets none of the four FR-20 vars — confirmed, fresh.**
`grep -n "FALKORCHAT_LLM_\|FALKORCHAT_EMBEDDING_BASE_URL\|FALKORCHAT_EMBEDDING_MODEL"
compose.yaml` → no matches; `services.server.environment` (`compose.yaml:35-39`) sets only
`FALKORDB_HOST`, `FALKORDB_PORT`, `FALKORCHAT_WS_ID`, `FALKORCHAT_USER_ID`. The plan is right, the
coordination record's blast-radius list is wrong, and the plan's §11.2 correctly declines to
"fix" `compose.yaml` itself and instead relays the correction to `teco` (the right call — this
plan doesn't own the coordination record).

**(b) `server/.env.example`'s fold-in into FR-20 is genuinely complete, not just mentioned.**
Checked three independent places, not just the changelog line: the §2.9 blast-radius table names
the exact lines (`server/.env.example:20,21,30,31`) and the fix; §5's file list carries the same
row; L1-5's "Files" column lists it and its done-condition is a **positive, executable test**
("copying `.env.example` to `.env`, sourcing it and running uvicorn **starts successfully**"), not
just "update the file." §10's test-strategy item 14 restates the same positive assertion. This is
what a complete fold-in looks like — a finding that shows up in the blast-radius table, the file
list, the unit's done-condition, *and* the test plan, not just the revision note's summary line.

---

### P2-B — New blocker: `-graph.md` v2 adds `StepRun.modelFallback`; the plan v2 never adopted it

**Severity: blocker.** *Owner: `architect` and `graph-dba` jointly (a `teco`-coordinated
reconciliation, since it's a two-document disagreement, not a one-side error).*

`-graph.md`'s response to my Pass-1 m-5 ("`modelSource` cannot distinguish a role's primary from
its fallback") was not a documentation note — it's a new, fully-specified **fourth `StepRun`
scalar property**, threaded through the whole document:

- §1.1's schema block gains `modelFallback` (boolean, nullable, absent by default).
- §1.3 justifies it explicitly over the documentation-only alternative I suggested, with a
  specific, well-reasoned point: `modelSource='workspace'` combined with `modelFallback=true` is a
  *valid, meaningful* combination (a workspace override can itself be a role with its own fallback
  chain), which a comment on `modelSource` alone cannot express.
- §1.4's `CREATE` Cypher gains a third `⊕` line, and reuses the verified `NULL`-omits-the-property
  behavior for it.
- §1.5 adds `modelFallback: bool | None = None` to the `StepResult` carrier requirement.
- §1.7's two read-path queries both gain a `modelFallback` / `fellBack` column.
- §5's RAM table adds a row for it.
- §6.2 (the resolver-facing interface contract — the seam the plan is supposed to implement
  against) states a **binding requirement**: *"`modelFallback` is set by comparing the answering
  call against the chain the winning rung named... `modelFallback = (index of the successful entry
  > 0)`. It is orthogonal to `modelSource`."*

Grepped the plan (`docs/plans/llm-provider-config.md`) for `modelFallback` and `model_fallback`:
**zero hits, in either casing, anywhere in the document.** Concretely, the plan's own artifacts
that would need to carry this field do not:

- §5's file list: `llm.py` row adds only `ChatResult.model: str | None = None` (M-5) — no
  fallback-boolean sibling field, so the resolver has no return-value channel to communicate "was
  this a fallback" up from `FallbackClient`/the client that answered.
- §5's `repository.py` row: `record_step_and_advance` "gains `resolved_model=` / `model_source=`"
  — no `model_fallback=`.
- L2-2's own text (`docs/plans/llm-provider-config.md:594`) still describes the **old,
  documentation-only** answer to m-5 verbatim: *"`modelSource` ... does **not** mark a fallback
  (m-5); record that limitation next to the field and expose fallback occurrence in logs + debug
  `TraceEvent`s."* This is the position the graph note's v2 explicitly moved away from (§1.3: *"This
  is chosen over the review's documentation-only alternative... because AC-9 is a formal acceptance
  criterion, not a debugging concern, and `TraceEvent`s are debug-only by construction"*) — the
  plan's L2-2 has not caught up with that argument at all, let alone rebutted it.

This is not a cosmetic gap. `-graph.md`'s own §0 preamble states *"the two documents agree on
every substantive decision; the single divergence is a property name"* — that sentence was true
of v1 and is **no longer true of v2**: the revision that was supposed to close the naming gap
introduced a second, larger one (a whole additional persisted property, with its own write path,
read path, and resolver-side computation rule, that only one of the two documents knows about). An
implementer handed both documents today would build `StepRun` with three properties per one
document's Cypher and two per the other's file list, and `ChatResult`/`StepResult` would have no
carrier for the third — a partial, silently-incomplete Landing 2, discovered only when someone
tries to wire `record_step_and_advance(..., model_fallback=...)` against a repository method that
was never told to accept it.

**Suggested resolution** (either direction closes it — this is a coordination gap, not a design
defect on either side):

1. **Adopt it** (my recommendation — the design is sound and I said so under item 3 above): add to
   the plan — `ChatResult.fallback: bool` (or fold it into `.model` as a `(model, fallback)` pair),
   `StepResult.modelFallback: bool | None = None`, `record_step_and_advance(...,
   model_fallback=...)` in §5's `repository.py` row, and rewrite L2-2's m-5 sentence to match
   `-graph.md` §1.3/§6.2 instead of contradicting it. Small, mechanical, and the graph note has
   already done the harder design work (the "orthogonal to `modelSource`" reasoning, the
   `index > 0` computation rule).
2. **Reject it** — if `architect` still prefers the documentation-only answer, say so explicitly
   in a v3 exchange and have `graph-dba` withdraw §1.1/§1.3/§1.4/§1.5/§1.7/§5/§6.2's `modelFallback`
   additions accordingly. Silence is the one outcome that isn't acceptable here, because it's what
   produced this gap.

Either way, this must close **before** Landing 2 implementation starts — L2-2's done-condition
("Two steps on two models produce two different `StepRun.resolvedModel` values... a two-model node
records the last") is achievable either way, but AC-9's trace-reading half depends on which
document an implementer trusts for the schema, and right now the two disagree.

---

### Pass 2 summary

| # | Item | Status |
|---|---|---|
| 1 | B-1 (guard carrier / SHA lock) | **Closed** — verified by direct trace of `run()`/`resume()`/`_drive`/`_drive_loop`/`_select_transition`/`evaluate_guard`, not by trusting §4.10's table. Zero edits inside the lock, confirmed against the recomputed hash. |
| 2 | B-2 (exception ladder) | **Closed** — both holes empirically closed by the re-run ladder probe. One correction to my own Pass-1 evidence (M-6's failure mode is `URLError`, not `ValueError`) accepted, correctly logged by the plan as a partial rebuttal. |
| 3 | m-4 | **Closed**, well-designed (Chunk gating, re-verified via fresh grep). |
| 3 | m-5 | **Not closed** — see P2-B (new blocker). Design is sound; adoption is one-sided. |
| 3 | m-6 | **Closed**, well-designed and consistently landed in both documents. |
| 4 | A-2 framing | **Mostly closed** — withdrawn where it's acted on (§2.1, §7-Q1, plan §8); one stale restatement survives at `-graph.md` §6.5 (minor, routed to `graph-dba`). |
| 5 | Naming (A-1) | **Clean** except for the `modelFallback` gap, which is a schema/scope disagreement, not a naming-consistency miss. |
| 6 | Inbound override carrier | **Closed** — consistent, specific, and correctly rejects `self` as a carrier. |
| 7a | `compose.yaml` sets none of the four vars | **Confirmed independently**, fresh grep. |
| 7b | `.env.example` fold-in is complete | **Confirmed independently** — present in the blast-radius table, the file list, the unit's done-condition, and the test plan, not just the revision note. |

**Overall Pass 2 verdict: needs changes.** One blocker (P2-B, `modelFallback` cross-document
disagreement — routed to `architect` + `graph-dba`, `teco`-coordinated), one minor (the stale
`-graph.md` §6.5 restatement — routed to `graph-dba`). Everything else from Pass 1 — both
blockers, all six majors, and eight of nine minors — is genuinely resolved, and resolved well:
the B-1 fix in particular is a better design than what I asked for (it discovered that `run` is
already threaded through to `evaluate_guard` today, turning what could have been a signature
change into a pure additive stamp). This is a small, well-scoped gap standing between the design
and Landing-2 implementation readiness, not a sign of a design in trouble.
