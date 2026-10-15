# Review — Transient provider-error retry for document-ingestion background jobs

> **Status:** active · **Owner:** `analyst` · **Tracks:** K-067

## Pass 1 — plan

**Scope.** I reviewed `falkor-chat/docs/plans/extract-transient-retry.md` (untracked, U1 `architect`)
against baseline `704f590c`, K-067 in `falkor-chat/docs/BACKLOG.md`, and the ledger
`falkor-chat/docs/plans/extract-transient-retry-coordination.md`. I read the code on every layer
between the provider and the two jobs: `transport.py`, `llm.py` (`OpenAICompatibleLLM`,
`FallbackClient`), `modelconfig.py` (`ModelGateway`, `GraphWorkspaceOverrides`, `_build_*`),
`extraction.py`, `ingestion.py`, `embedding.py`, `background.py`, and `repository.py` §14.1a/§14.7.
I also read `app.py` wiring, `mcp._default_schedule`, `executor.py`'s tool-error split,
`services.py`'s drive-fault catch, and the installed Starlette 1.3.1 `background.py`. I re-ran the
§2.6 field query on `ws:agent-team` (read-only), ran EXPLAIN on the gate query, and checked the
CPython 3.12.3 `urllib` `do_open` source. I did **not** run pytest or `test_queries.sh`, per the
brief.

**Verdict: needs changes.** There are no blockers. The core design is sound: structured exception
fields, an allow-list classifier, wrapping only the provider call, and leaving K-051 untouched.
There are four majors. One is a correctness error in the gate's placement, which the plan presents
as closing a race that it does not close. Three concern the embed-scope premise and the live
proof: as written, the live proof cannot detect the failure the brief asked about, a classifier
that never fires in production.

CPG: used cpg_falkorchat — confirmed `extract_chunk`/`embed_chunk` each have exactly one production caller (`_safe_extract`/`_safe_embed_chunk`), `_resolve_and_embed` is shared by `embed_chunk` and `embed_message` (`responder.maybe_respond` calls the latter), and `make_http_transport` has four production construction sites.

### Findings

#### M-1 (major) — The liveness gate runs *before* the sleep, so it never sees a delete or sibling failure that lands *during* the sleep. §3.5 and R-5 claim the opposite.

- **Evidence.** §3.6's contract is: on a transient error, `if … keep_going returns False → re-raise;
  otherwise … sleep(delay), and loop`. §3.5 says the gate is "checked before each retry sleep". So
  the gate is read at time *t* and the next attempt fires at *t + delay*, with no read in between.
- **Why it matters.** A `delete_document` that lands during the 30 s sleep before attempt 4 goes
  undetected. If attempt 4 succeeds, the write loop runs and `create_entity_with_auto_match` (a bare
  `CREATE`) leaves orphan `Entity` nodes, because `link_chunk_about_entity`'s `MATCH` finds no chunk.
  This is exactly the race §3.5 point 2 says the gate "closes most of". R-5 says "the retry widens
  only the sleep window, and the gate closes that". It does not: in this design the gate only
  detects state changes that happened *before the failure*. The same gap means a document failed
  by a sibling during the sleep still costs one more provider call.
- **Fix.** Read the gate **after** each sleep, immediately before the next attempt. Keep the
  pre-sleep read as well, so an already-terminal document skips the sleep (that read is what saves
  time on the sequential REST chain). Update §3.5 point 2, R-5, and the §3.6 contract. Extend U-8b
  with `keep_going` scripted as `[True, False]`: attempt 1 fails transiently → pre-sleep read True
  → sleep → post-sleep read False → re-raise with `fn` called **once** and `sleep` called once.
  That test fails under the plan's current single-read design, so it pins the fix. If the residual
  in-flight race should also shrink, a third optional read can go right before the write loop
  (one indexed read per chunk). Leave that call to the architect, because the race predates this
  change.

#### M-2 (major) — The embed-scope premise is half-proven. The graph shows embed jobs *failed*; it does not show they failed with a shape this classifier retries, or one a 60 s budget clears.

- **What I verified.** I re-ran the query: 12 `failed` documents, all with `pendingJobs = 0`, 27
  chunks, 21 without an embedding. The 12/9/1/5 split matches §2.6 exactly. The inference "an
  un-embedded chunk under `pendingJobs = 0` is a reported embed failure" also holds:
  - every scheduled job runs regardless of document status, because `_safe_*` has no status check
    (`background.py:92-149`);
  - `set_chunk_embedding` always writes when the chunk exists (`repository.py:1202-1229`);
  - nothing in `falkorchat/` or `scripts/` ever clears `Chunk.embedding` (I grepped for it);
  - `EmbeddingDimensionError` is ruled out, because 338 `ready` documents share the same
    workspace, the same `embeddingModelOverride`, and dim 1024.

  So, answering the brief's question 3 directly: no, a sibling failure neither skips embeds nor
  pre-empts them. The missing embeddings are real embed-job failures. Note also that the four
  single-chunk documents in the set (created 2026-09-19 ~11:27 UTC) each have **extraction
  succeeded, embed failed**.
- **What is not shown.** *Which exception* failed those jobs. No server log survives:
  `~/.local/state/graphmind-ai-lab/agent-team-8200.log` starts at 2026-10-15 09:27, and the failures
  date from 2026-09-19/20. The obvious contention shape is a 30 s read timeout on `embedding`
  (`config/models.json` `timeouts.embedding`): an embed request queued behind the resident chat
  model's generation plus a model swap. The plan's `"timeout"` kind is **never retried** (§3.1). If
  that was the shape, the embed half of this change fixes none of the observed incidents. §2.6
  ("That decides the §3.4 scope question") and §3.4 point 1 ("Fixing only extract would leave most
  of the observed K-067 incidents unfixed") assert more than the data shows.
- **My independent view: embed belongs in this change.** The mechanism and the failure contract are
  the same, a documented cold-load 400 on the embedding model is on record, and the marginal cost
  is small. But nobody should claim it fixes the field failures until the shape is captured. Three
  actions:
  - Reword §2.6/§3.4 to "embed jobs demonstrably fail; failure shape unknown".
  - Make the live proof capture the real shape (M-3).
  - Decide timeouts **per kind**. OQ-1's 180 s argument does not carry over to `embedding`: a 30 s
    timeout retried 4 times costs ≤ 4×30 + 60 = 180 s, about one extract attempt. Leave this as a
    decision the live capture triggers, not a default.
- Separately, log `kind`/`status` in the retry WARNING and in a final "giving up after N attempts"
  line. Then the next field failure is self-diagnosing, which this one was not.

#### M-3 (major) — The live proof cannot catch "a classifier that never fires in production". Every injected fault is a byte copy of the string the allow-list was built from.

- **Evidence.** In §5 "Live proof run", the proxy serves exactly the bodies `{"error":"terminated"}`,
  HTML 500, and the §2.2 `exceed_context_size_error`. Those are the same captured strings that
  `TRANSIENT_DETAIL_MARKERS` and U-5 encode. The proxy does drive the real producer → store →
  consumer round trip: REST and MCP ingest, real FalkorDB, real `search_documents` read-back, a
  real `make_http_transport` and `urllib`. So it does prove the wiring, the status contract, the
  gate, and the timing. But a classifier mismatch against what LM Studio *actually* returns today
  passes this test by construction.
- **Fix.** Add **real-fault** legs with no synthetic bodies, and tabulate the logged `kind`/`status`
  for each:
  - **(e)** `lms unload <chat model>` while an extract request is in flight. This is the closest
    reproducible relative of the "terminated" crash.
  - **(f)** `lms server stop`, then `lms server start` within the 2–15 s window. This exercises the
    real connection-refused/reset path.
  - **(g) Unproxied contention.** MCP-ingest a 5–10-chunk document with both models cold, which is
    the actual K-067 trigger (concurrent embed + extract on a one-resident-model server). Record
    the first-attempt exception kinds and the final status. This leg also answers M-2 and M-4.
- **Also:**
  - Add **(d′)**, a proxy leg that sends headers with a `Content-Length` larger than the body and
    then closes. Without it, rung 3b is never exercised live.
  - The proxy's per-path request counter is **global**. On MCP's concurrent path, "first 2
    requests" spread across different jobs, so the per-job assertions in 3(a) ("attempt 1/4 … 2.x s,
    2/4 … 8.x s") will not hold. Use 1-chunk documents for timing assertions, or key the fault
    script per request body.

#### M-4 (major) — Possible non-convergence on the MCP path: the retries re-create the contention that caused the fault

- **Evidence.** §2.1 and §2.5 already establish three facts:
  - MCP runs all of a document's jobs concurrently (`mcp._default_schedule`, one daemon thread per
    job);
  - this LM Studio keeps exactly one model resident;
  - embed and extract evict each other.

  On the MCP path no job reports until its budget ends, so the gate (§3.5) never trips early. All
  of a document's failed jobs therefore retry in the same jittered windows (2–3 s, then +8–12 s,
  then +30–45 s). Each window again mixes embed and chat requests against the same single slot.
  Upward jitter spreads requests *within* a job kind; it does not separate the two kinds.
- **Why it matters.** Inferred, not verified (no LM Studio run). For a multi-chunk MCP document,
  the 4-attempt budget can exhaust *because of* the document's own retry traffic. The change would
  then add up to 60 s of delay and still end in `failed`. F-1 (embeds before extracts) is filed as
  a REST-only follow-up, but scheduling order is moot on MCP, where everything starts at once.
- **Fix.** No redesign is required before implementation. Make leg (g) in M-3 the acceptance
  evidence for the MCP path. Have §6 state the risk explicitly (a new R-8), so that a (g) result of
  "still fails" routes to a follow-up such as per-document job serialization on MCP, rather than
  reading as a defect in this change.

#### m-1 (minor) — The worst-case latency is understated: attempts are not "fast-failing" by construction

§3.2 says "≤ 60 s of sleep, plus 3 extra fast-failing attempts", and R-2/R-3/R-4 inherit that. But
"terminated" is by definition a mid-request crash. A 400/500 can arrive after up to one full
read-timeout's worth of generation. Each attempt is bounded only by `timeouts.extraction` = 180 s.
Recomputed:
- extract job: ≤ 4×180 + 60 = **780 s (13 min)**;
- embed job: ≤ 4×30 + 60 = **180 s**.

The cumulative-wait table itself is correct: (2, 10, 40) nominal → (3, 15, 60) at jitter 0.5, with
an exclusive upper bound because `U < 1`. State both numbers in §3.2. Then R-3's limiter-occupancy
math, and the "≤ 60 s per job" in R-2, rest on the right figure.

#### m-2 (minor) — On REST, the gate bounds the delay per document, not per request. A batch keeps the K-067 multiplication.

`api.ingest_documents` (`api.py:221-235`) puts every document's jobs on **one** `BackgroundTasks`
chain, and that chain runs sequentially (verified: Starlette 1.3.1 `BackgroundTasks.__call__` is
`for task in self.tasks: await task()`). During an outage, each document's first faulting job burns
its full budget before the gate can trip. A 19-document batch, the K-067 field size, therefore costs
≈ 19 × 40–60 s ≈ **13–19 min** of one worker token asleep, with every later document stuck in
`processing`. That is bounded and arguably acceptable for lab load. But §3.5 point 1 and R-4 should
say "per document", not imply one budget per request. A per-request circuit breaker (a shared
"provider down" flag across one chain) is the option to name if this matters. A follow-up is fine.

#### m-3 (minor) — Rung 3 has a sibling gap: a non-`OSError` `http.client.HTTPException` from `getresponse()` also escapes the ladder raw

I verified this against the CPython 3.12.3 source. `AbstractHTTPHandler.do_open` wraps only
`h.request`'s `OSError` into `URLError`. `r = h.getresponse()` sits under a bare `except: h.close();
raise`. `RemoteDisconnected` is an `OSError`, so rung 3 catches it. But `BadStatusLine` and
`LineTooLong` (MRO `HTTPException → Exception`) match none of rungs 1–4 and escape as raw
exceptions, the same defect class rung 3b fixes for `resp.read()`. Fix: add
`http.client.HTTPException` to rung 3's `except` tuple with `kind="network"`, plus one U-1 row. The
plan is already adding `import http.client`.

#### m-4 (minor) — U-9 and I-1 do not pin the retry *boundary*. The rejected alternative (retrying all of `extract_chunk`) passes both.

- **Question 2a: does any write precede the wrapped call?** Answer: the idempotency argument is
  sound. I verified that no write precedes the wrapped call on either path:
  - `extract_chunk`'s pre-call region is `self._models.llm(...)`, which runs
    `GraphWorkspaceOverrides.get` → `read_model_overrides`, a `ro_query` (`repository.py:3300`);
  - `_resolve_and_embed` adds `read_index_dimension`, also `ro_query`;
  - `start_document_progress` runs before scheduling and is never inside a job.
- **Question 2b: would I-1 catch a write moved before the call?** Partly. I-1 and U-9 inject the
  failure only at `llm.complete`, which comes before every write. So:
  - **(i)** Moving a *counted* write (`Entity`/`ABOUT`/`RELATES_TO`/`SAME_AS`) **inside** the retried
    lambda is caught: it runs three times, and U-9's control diff and I-1's count diff both show it.
  - **(ii)** Widening the retry to the whole `extract_chunk`, or to `_safe_extract`, is **not**
    caught: the failure still fires before any write. Yet that widening is precisely the regression
    §3.3's "rejected alternative" paragraph warns about.
  - **(iii)** An uncounted write (a new property on the chunk) inside the lambda escapes I-1.
- **Fix.** Add a **U-9b** test. A `SpyRepo.create_entity_with_auto_match` that raises a
  transient-classified `ProviderCallError(kind="http", status=503)`, standing in for a future
  provider call inside the loop, should make `extract_chunk` raise with `complete` called **once**,
  one write attempt, and zero sleeps. That pins the boundary itself. Also give both U-9 pipelines a
  deterministic `id_gen`: the spec says "exactly the same calls as a control", but the
  `entityId`/`match_id` values are `uuid4`, so a literal comparison can never be equal.

#### m-5 (minor) — The step table omits a file that a done-condition needs

Step 3's done-condition includes **I-3**, which §5 places in `tests/test_background.py`. Step 3's
file list does not include that file; only step 2 lists it. Add it to step 3. Also note for the
implementer: `test_background.py` is fully offline today (fakes only). I-1/I-3/I-4 make it
FalkorDB-requiring through the `repo` fixture. That is safe for `reference`, because `repo` does
not wipe it. But it changes how the file must be run, so the plan's "test-run hazard" list should
say so.

#### Nits

- §2.1/§3.4 call gateway resolution "config only, no I/O". In production, `GraphWorkspaceOverrides`
  does a `ro_query` per `resolve`, so `_resolve_and_embed` makes two graph reads per attempt. This is
  harmless, but the text should say "read-only".
- Step 1 adds `kind="response_shape"` to `OpenAICompatibleEmbedder.embed`, but no test asserts it
  (U-4 covers the LLM only). It is behaviorally invisible, since `None` and `response_shape` are
  both permanent. Add one row or drop the claim.
- R-6 is correct that rung 3b is strictly a fix (see "What's solid"). It should name the two
  observable changes:
  - the executor tool-trace text changes from `ConnectionResetError: …` to `ProviderCallError: …:
    reading response failed: …` (`executor.py:1097`);
  - the workflow start now returns a `failed` envelope instead of a 500 (`services.py:2588-2603`).

### Answers to the brief's questions (summary)

1. **The classifier does fire on what is raised.** No layer re-wraps or swallows the exception:
   - `extraction.extract` has no `try`;
   - `OpenAICompatibleLLM.complete` re-raises only shape errors;
   - `FallbackClient` is the one re-wrapper, and the plan handles it with `causes`;
   - `ModelGateway` returns a bare client for the length-1 default chain;
   - `_safe_*` catches only *above* the retry.

   The 400 "terminated" and 500 cases are rung 1, with `exc.code` and `body` both available;
   `body` is truncated at 500 chars, which is harmless for the markers. Rung 3b is strictly a fix
   for every caller. Only `FallbackClient` catches `ProviderCallError` by type; every other caller
   catches `Exception`. One sibling gap remains (m-3).
2. **The idempotency argument is sound.** No write precedes the call on any path. I-1 catches a
   write moved *into* the lambda, but not a widened retry boundary (m-4).
3. **The premise is half true** (M-2). Embed jobs really failed, and sibling failures do not skip
   embeds. The failure shape is unknown, and the plan's never-retried `timeout` kind is a plausible
   candidate. Embed belongs in scope, but with honest claims and a live capture.
4. **`time.sleep` on the job thread is acceptable.** The Starlette sequential claim is verified, and
   the 100-token limit is confirmed (`config.py:261`). `python-web-quirks` §"BackgroundTasks is
   bounded" agrees and adds nothing contrary. But the gate is mis-placed (M-1), it bounds per
   document rather than per request (m-2), and the worst-case figure is understated (m-1).
5. **Done-conditions are reachable**, apart from I-3's missing file (m-5). Step 1 is coherent as
   one unit. It *could* split into 1a (exception fields + rung 3b + `test_transport`/`test_llm`)
   and 1b (`retry.py` + `test_retry`), since 1b depends only on 1a's fields. I would not force the
   split: it is seven small, cohesive files.
6. **The live proof drives the real round trip but cannot detect a classifier that never fires**
   (M-3); it needs real-fault legs (e)–(g).
7. **Numbers.** All recomputed figures match except the per-job worst case (m-1):
   - cumulative wait 2/10/40 nominal, 2–3 / 10–15 / 40–60 s jittered;
   - the U-6 bounds with `rand = 0.999` (2.999 < 3, 11.996 < 12);
   - 40/21 ≈ 1.9× clearance over the slowest cold load;
   - 20 chunks → 40 jobs × 60 s = 40 min;
   - 588 chunks (`mcp.py:89`) → 1,176 jobs ≈ 19.6 h;
   - 7 files in step 1;
   - 6 production methods and 5 test sites constructing `ProviderCallError`;
   - field counts 12 / 27 / 21 and 12/9/1/5.

   For context on the "no `ABOUT` edge" ambiguity: 116 of 563 chunks (20.6 %) in `ready` documents
   have no `ABOUT` edge, so `_EMPTY` is common.

### What's solid

- §2.2's surfacing table is accurate rung by rung, including the `resp.read()` escape. I verified
  it against `transport.py:109-146` and `do_open`.
- Classifying on structured fields, with policy only in `retry.py`, is the right seam. The message
  strings stay byte-identical.
- Wrapping only the pure provider region makes idempotency structural, not incidental. The rejection
  of whole-`extract_chunk` retry is well argued.
- Keeping `embed_message`/`_resolve_and_embed` untouched protects the responder latency path. The
  CPG confirms `responder.maybe_respond` → `embed_message`.
- Leaving K-051 code untouched, with no new knob and no schema change, is correct. The gate query
  plan is verified: `Node By Index Scan (c:Chunk)` → `Conditional Traverse`.

### Open questions

- **OQ-A (stakeholder, in parallel).** Should the embed scope be confirmed now, with M-2's softened
  claims, or held until leg (g) captures the real shape? My recommendation: confirm now. The cost is
  small and the mechanism is shared.
- **OQ-B (architect).** If leg (g) shows that embed failures are 30 s timeouts, does the plan
  pre-authorize retrying `"timeout"` for `kind == "embedding"` only, or route that back for a design
  change?

## Pass 2 — plan

**Scope.** Plan v2 (`falkor-chat/docs/plans/extract-transient-retry.md`, 1,324 lines, §8
dispositions), re-checked against the same tree (`704f590c`, `falkor-chat/server` unchanged). I
re-ran the executor/services/urllib checks, the `ws:agent-team` status count, `ss`, `lms.exe`
presence, and CPython 3.12.3 behaviour for malformed URLs. Stakeholder ruling recorded: embed is in
scope (OQ-A closed).

**Verdict: approve with suggestions.** U3 (implementation) can start on v2 as written.
- **N-1, N-2 and N-6** are small edits the implementer can absorb. They are stated precisely enough
  to need no redesign.
- **N-3, N-4, N-5 and N-7** are live-proof edits. **N-7 must be fixed before U5 runs legs
  (e)–(g).**
- **Every new finding below was produced by a v2 fix** (rung-3 widening, U-9b, the body-hash
  proxy, (c′), §3.2.1, the shared-instance check). None was found in the underlying design. Another
  full static pass has diminishing value; see "Stopping condition".

CPG: used cpg_falkorchat — Pass 1's caller sweep still holds (no `falkorchat/` change since); no new CPG query was needed for v2's deltas, which are plan-text, test-spec and live-proof changes.

### Dispositions of Pass 1 findings

- **M-1: fixed.** §3.6 step 4 adds a post-sleep read, and U-8b `[True, False]` pins it. I checked
  the architect's three reasons for declining the pre-write read against the code, and all three
  hold:
  - it would run on the happy path;
  - `_safe_extract` has no status check, so entities of a sibling-failed document are written
    today (`background.py:120-149`) and would no longer be;
  - a delete can still land between that read and the writes.

  R-5's window ("during an attempt, or between the successful attempt and the end of the write
  loop … same as a first-try call") is honest. A delete during a *failing* attempt is caught by
  the next pre-sleep read, and one during a sleep by the post-sleep read.
- **M-2: fixed.** §2.6/§3.4 make no prevention claim, and the logs carry `kind`/`status`.
- **M-3: fixed in substance.** Legs (c′), (d′) and (e)–(g) were added, with the timing rule. Two
  defects were introduced by the fix (N-3, N-4).
- **M-4: accepted as R-8.** That is a sound disposition for an inferred risk with a named
  acceptance leg. N-5 covers its routing boundary.
- **m-1: fixed.** Recomputed: 4×180 + 60 = 780 s; 4×30 + 60 = 180 s.
- **m-2: fixed.** Recomputed: 19 × 40–60 s = 12.7–19 min; 19 × 780 s = 14,820 s ≈ 4.1 h;
  20 × 40–60 s = 13.3–20 min; `schemas.MAX_BATCH_SIZE = 20` (`schemas.py:61`).
- **m-3: fixed, but the widening over-catches** (N-1).
- **m-4: fixed, but one claim attached to U-9b is false** (N-2). U-9's deterministic `id_gen` is
  wording-sensitive (nit below).
- **m-5: fixed.** Step 3 lists `tests/test_background.py`, and the run-hazard bullet is present.
- **Nit "config only": fixed.**
- **Nit "embedder `response_shape`": fixed.**
- **Nit R-6: my Pass 1 nit was wrong, and the architect's correction is right on the trace.** The
  correction also goes one step too far (see the nit below).
  - Tool dispatch in `executor.py:1066-1068` is `try: … except ServiceError as exc:`.
    `ProviderCallError` is a `RuntimeError`, and raw `OSError`/`IncompleteRead` are not
    `ServiceError`s either.
  - So neither ever enters the branch. The trace text (`:1097`) does not change.
  - Item 1 of R-6, the `services.py` drive-fault catch, is confirmed.

### New findings

#### N-7 (major; before U5 legs (e)–(g)) — The shared-instance precondition is not checkable as written, and it always degrades to "ask the coordinator"

- **Evidence.** In §5 "Environment", the check is `ss -ltnp | grep 8200` plus `lms ps`, and then
  "confirm nothing else is using LM Studio or get the coordinator's go-ahead".
  - `ss` shows a *listener*, not usage. Port 8200 is listening right now (`uvicorn`, pid 1882),
    because teco's `SessionStart` hook `ensure-environment.sh` brings the `ws:agent-team` server up
    on every coordinated session. So the check is effectively always true.
  - `lms ps` shows loaded models, not in-flight requests.
- **Why it matters.** Legs (e)/(f)/(g) (`lms unload --all`, `server stop`) disrupt every consumer of
  the one LM Studio. The likeliest concurrent consumer is another agent's learning-capture
  `ingest_document` into `ws:agent-team`; I made one during Pass 1. A disrupted ingest there
  produces exactly a new K-067 `failed` document in the shared KB, which then needs a manual
  re-drive. A model-bench sweep (`model-bench/run-sweep.sh`, untracked in the tree) is the other
  known consumer.
- **Fix: replace the check with three conditions QA can evaluate mechanically, each with a pass
  value.**
  1. `MATCH (d:Document {status:'processing'}) RETURN count(d)` on `ws:agent-team` returns
     **0**, read immediately before each leg. Today it is 0: 339 `ready` and 12 `failed`.
  2. `pgrep -af 'run-sweep|model-bench'` returns **nothing**.
  3. `lms log stream` shows **no request for 60 s** before the leg starts.

  Add a coordinator obligation: teco tells in-flight agents not to ingest during the window, or
  runs U5 when none are in flight. Repeat condition (1) **after** each leg; a non-zero `failed`
  delta in `ws:agent-team` is a reportable side effect.

#### N-1 (minor) — Rung 3's new `http.client.HTTPException` clause classifies a permanent malformed URL as transient

- **Evidence (executed, CPython 3.12.3).** `urlopen` on `http://localhost:abc/v1/x` raises
  `http.client.InvalidURL("nonnumeric port: 'abc'")`. A space in the host or path raises
  `InvalidURL("URL can't contain control characters …")`. The MRO is
  `InvalidURL → HTTPException → Exception`, **not** `ValueError`, so rung 4 (the "malformed-URL
  shapes" rung) never sees it.
- Such a URL reaches call time: `modelconfig._validate_url` checks only scheme and non-empty
  netloc, and `urlparse('http://localhost:abc').netloc == 'localhost:abc'`.
- **Consequence.** Under v2's widened rung 3 it becomes `kind="network"` and is retried for a full
  budget per job. The intent stated in §3.1 is the opposite: `malformed_request` is permanent.
  Today it escapes raw, which also fails fast. So the misclassification is a regression introduced
  by the m-3 fix, though a bounded one (R-2-shaped).
- **Fix.** Add an `except http.client.InvalidURL` clause **before** rung 3, mapped to
  `kind="malformed_request"`, or put `InvalidURL` in rung 4's tuple and move that clause above
  rung 3.
- **Add a U-1 row:** an opener raising `http.client.InvalidURL("nonnumeric port")` →
  `kind == "malformed_request"`, and `is_transient` is `False`. That row fails under v2 as written,
  so it pins the fix.
- `BadStatusLine`/`LineTooLong` → `"network"` remains as intended.

#### N-2 (minor) — U-9b does not detect a retry added in `_safe_extract`, though the plan says it does

- **Evidence.** In §5, U-9b "fails if the retry is ever widened to all of `extract_chunk` **or to
  `_safe_extract`**". But U-9b lives in `tests/test_ingestion.py` and calls
  `extract_chunk` directly, so a retry loop placed in `background._safe_extract` is outside the
  code under test.
- **Fix.** Add a sibling U-9c in `tests/test_background.py` (offline, fakes) that drives the same
  scenario through `_safe_extract`, with a pipeline carrying a recording `.repo`. Assert:
  - `complete` is called once;
  - there is one write attempt;
  - there are zero sleeps;
  - exactly one `report_document_job_done(success=False)`.

  Alternatively, narrow the claim to `extract_chunk`.

#### N-3 (minor) — The per-body-hash fault counters collide when a leg runs twice with the same document text

- **Evidence.** In §5 "Timing rule", the proxy counts faults "per distinct request body (sha256)".
  "Each leg runs via both REST `POST` ingest and the MCP `ingest_document` tool." The two runs send
  byte-identical chat and embedding bodies, so the second run finds the counter exhausted and is
  forwarded with **zero** faults.
- A same-text re-ingest also auto-supersedes the first document
  (`services.ingest_document` → `create_document_with_auto_supersede`). That is harmless to status,
  but it is noise.
- QA would notice, because the expected retry WARNINGs would be missing. But a run would be wasted.
- **Fix.** Use one of these:
  - a per-run nonce line in each document's text;
  - a proxy reset (or a counter namespace) per leg-run;

  and state which in §5.

#### N-4 (minor) — Leg (c′)'s "no orphan `Entity`" assertion is vacuous under an always-500 script

- **Evidence.** In (c′), the script is "always `500` on chat". With that script, no attempt can
  ever succeed, so no write can occur, with or without the post-sleep read. The only live signal is
  "no further proxy hit".
- **Fix.** Script `500` ×2, then forward, and delete partway through the second sleep (8–12 s).
  - **Without** the post-sleep read, attempt 3 is forwarded, succeeds, and writes orphans: the
    orphan query returns > 0, and the proxy logs a third hit.
  - **With** it, neither happens.

  Both assertions then discriminate.

#### N-5 (minor) — §3.2.1's done-condition 2 is near-tautological, has a false-fail path, and the routing boundary needs a closure rule

- **Evidence.** Condition 2 says each timed-out embed job "either succeed[s] on a later attempt or
  log[s] `giving up after 4 attempts (kind=timeout)`".
  - Under the flipped policy, every job does one or the other, so the condition cannot fail on the
    thing it is meant to show.
  - It **can** fail spuriously: a job stopped by the gate (a sibling failed the document, which is
    likely on MCP) logs the INFO "abandoning retry", not "giving up".
  - Separately, the flip turns U-5b red by design ("both shipped policies `retry_timeouts is
    False`"), but the done-condition does not list that edit.
- **Fix.**
  - Replace condition 2 with something falsifiable and specific to the flip: *"for every embed job
    whose attempt 1 failed with `kind=timeout`, the log shows a retry WARNING `attempt 1/4 …
    kind=timeout`"*. That WARNING is absent unless the flip took effect.
  - Add "U-5b's shipped-policy row updated for `EMBED_CHUNK_RETRY`" to the list.
- **Routing boundary (brief ask 3).** "A still-failing re-run routes to F-4" is sound *for this
  change*. Its DoD is the retry mechanism, and (g) with transient kinds in the give-up line proves
  the mechanism fired. It is **not** evidence that K-067's user-visible goal is met. So the
  plan (or teco's closeout) should state:
  - if the final (g) document is `failed`, K-067 is closed as "mechanism delivered", **and** F-4 is
    filed **before** K-067 leaves `BACKLOG.md`, carrying (g)'s table as the continuation of
    K-067's goal;
  - `HISTORY.md` must not say the field failure is cured.

  With that rule, the boundary is not a way of declaring success early. Without it, it is.
- A single (g) run that ends `ready` is also weak evidence of convergence (n = 1, nondeterministic).
  Say "converged in this run", not "the MCP path converges".

#### Nits

- **R-6 item 2 overcorrects.** The `tool %s failed: %r` WARNING (`executor.py:1094`) is also inside
  `except ServiceError`, so it does not fire for `ProviderCallError` or for the old raw exceptions.
  The executor shows **no** observable change. Drop item 2, or say "no executor-visible change".
- **U-9 "the same deterministic `id_gen`".** This must be a **fresh** counter per pipeline. One
  shared `itertools.count` instance gives the control pipeline different ids and a false red.

### Stopping condition (v3 approvable iff)

A v3 is approvable when every check below holds. Each is checkable by reading the plan, and none
needs a further design argument:

1. §3.1 and step 1.2 map `http.client.InvalidURL` to `malformed_request` ahead of rung 3, with the
   U-1 row in N-1.
2. §5 has U-9c driving `_safe_extract` (N-2), or U-9b's claim drops "`_safe_extract`".
3. §5 "Timing rule" names the per-run nonce or proxy reset (N-3).
4. (c′) scripts `500` ×2 then forward, with both the hit-count and orphan assertions (N-4).
5. §3.2.1's condition 2 reads as N-5's falsifiable WARNING check, the list includes the U-5b edit,
   and the K-067-closure-with-F-4 rule is stated.
6. §5 "Environment" carries N-7's three mechanical preconditions with pass values, the post-leg
   re-check, and the no-ingest coordinator obligation.

Items 1–2 are needed before U3 finishes step 1/2 (the implementer can absorb them from this
review). Items 3–6 are needed before U5. A v3 that satisfies these needs a **targeted recheck of
those six items, not a third full pass**: every Pass 2 finding came from a v2 fix, none from the
design, which is the diminishing-returns signal.

## Pass 3 — plan (targeted)

**Scope.** Plan v3 (1,474 lines, §8.2), checked only against Pass 2's stopping condition plus the
coordinator's four judgment points. The code is unchanged since `704f590c`, and `retry.py` does not
exist yet, so every order-of-operations claim below is checked against the plan's §3.6 contract.
`background.py` is unchanged and reports only after the worker returns or raises.

### Stopping-condition items

1. **N-1 (rung 3a): pass.**
   - §3.1's table and step 1.2 put `except http.client.InvalidURL` immediately before rung 3, with
     `kind="malformed_request"` and rung 4's message format.
   - U-1 adds the `nonnumeric port` row (`malformed_request`, `is_transient` False), which fails
     under v2.
   - The MRO and the raw escape from `urlopen` were confirmed by execution in Pass 2.
   - `InvalidURL` is raised before any `URLError` wrapping, so no earlier rung can catch it first.
2. **N-2: pass.**
   - U-9c drives a real `IngestionPipeline` through `background._safe_extract`. It asserts
     `complete` ×1, one write attempt, zero sleeps, and exactly one
     `report_document_job_done(success=False)`.
   - That 503 is transient-classified, so any retry that re-invokes `extract_chunk` reddens it.
   - U-9b's claim is narrowed to `extract_chunk`.
3. **N-3: pass, with a nit.** The per-run nonce is stated, recorded, and stateless. But it is "the
   **first line**" of the text, so only chunk 1 carries it. Chunks 2+ of a multi-chunk document
   are byte-identical across the REST and MCP runs, so "its counters simply never see a repeated
   body" is false for them.
   - This is harmless today. The only multi-chunk proxy leg is (c), which is always `500`, and
     counts don't matter for it.
   - Put a nonce in every chunk, or limit the claim to 1-chunk legs.
4. **N-4: pass.**
   - (c′) scripts `500` ×2, then forward.
   - The delete lands 3 s after hit 2, which is after the pre-sleep read (milliseconds after the
     500) and inside the second sleep's 8 s floor.
   - "Exactly 2 hits" and "orphans = 0" each fail without the post-sleep read.
   - The void-run rule handles a delete that misses the window.
5. **N-5: the architect's correction is right; the amended condition has one remaining hole
   (P3-1).**
   - §3.6 orders the checks as: transient? → last attempt? → **pre-sleep gate** → WARNING → sleep.
     So a flipped timeout job stopped by the pre-sleep gate logs only INFO. My Pass 2 condition
     ("WARNING for every such job") would have failed falsely, and accepting the INFO line is
     correct.
   - The amended condition does fail when the flip has no effect, but only if QA can identify the
     timed-out jobs without using the flipped log lines (P3-1).
   - The U-5b edit and the K-067 closure rule are present.
6. **N-7: partial pass.**
   - Checks 1 (`processing` = 0, plus the failed baseline) and 2 (bracketed `pgrep`) are
     mechanical, have pass values, and are re-run after each leg. The coordinator obligation is
     addressed to teco.
   - Check 3 cannot be evaluated as written (P3-2).

**N-6 (the coordinator's question).** There was no N-6. It was a numbering slip in my Pass 2
verdict for the two Pass 2 **nits**: R-6 item 2 overcorrecting, and U-9's fresh counter. §8.2
closes both. The architect's reading is right, and nothing is outstanding.

**Interaction check (rung 3a × the INFO line × the other v3 fixes): clean.**
- `malformed_request` is permanent, so rung 3a's error re-raises before the gate is ever read. It
  can therefore never produce the INFO or WARNING lines that §3.2.1 condition 3 or (c′) count.
- Under `FallbackClient`, a rung-3a cause makes `fallback_exhausted` permanent, as intended.
- The INFO line reads `kind`/`status` via `getattr`. Only transient `ProviderCallError`s reach the
  gate, so that read never sees a foreign exception.
- (c′) expects the INFO line for its own label. Its 500s are `kind=http`, which is unaffected by
  rung 3a.

### New findings (both produced by v3's fix text, not by the design)

#### P3-1 (minor; before a §3.2.1 re-run of leg (g)) — Condition 3's job set is defined only by lines the flip itself produces, so a flip that never took effect reads as "not exercised" and passes

- **Evidence.**
  - Condition 3 quantifies over "every embed job whose attempt 1 failed with `kind=timeout`".
  - Unflipped, a timeout is non-transient and re-raises with **no** retry log (§3.6). The only
    trace is `_safe_embed_chunk`'s `background chunk embed failed` traceback (`background.py:110`).
    Its message is `…: TimeoutError: timed out` (rung 3) or `…: connection failed: timed out`
    (rung 2), and it carries **no `kind=`**.
  - So if QA identifies timeout jobs by `kind=timeout`, a deployment where the flip never took
    effect yields an empty set. §3.2.1 then says "not exercised … the flip stands on U-17 alone".
  - The likeliest form of "never took effect" is exactly that case: uvicorn runs with `--reload`
    **off**, so an un-restarted server keeps the old constant.
- **Fix (text only).**
  - Define the set independently of the flip: every `background chunk embed failed` traceback
    whose `ProviderCallError` message contains `TimeoutError` or `timed out`, **plus** every
    `embed chunkId=…` line carrying `kind=timeout`.
  - Fail condition 3 if any job in that set lacks the WARNING, INFO, or give-up line.
  - Allow "not exercised" only when that independent set is empty.
  - Add a precondition that the server process started after the constant edit (process start
    time later than the `retry.py` mtime).

#### P3-2 (minor; before U5 legs (e)–(g)) — The `lms log stream` quiet check passes vacuously unless it names `--source server` and runs a positive control

- **Evidence (executed 2026-10-15, read-only).**
  - `lms.exe log stream --help` lists `--source model|server|runtime` and `--json`.
  - I ran one `GET /v1/models` with `--source server --json` streaming: it produced a
    `{"type":"server.log","content":"[…][INFO] Returning {…}"}` line.
  - The same GET under `--source model` produced **nothing**.
  - The plan's bare `lms log stream` has an unverified default source. A stream that never logs
    requests reads as "no request for 60 s", so the check passes no matter what is happening.
- **Fix.**
  - Specify `lms.exe log stream --source server --json`.
  - Before trusting silence, require a positive control: one `curl localhost:1234/v1/models` must
    produce a `server.log` line in that stream.
  - If the control does not appear, check 3 is not evaluable. Proceed only on checks 1–2 plus the
    coordinator's explicit confirmation, and record that in the report.
  - Still unverified: whether a `POST` completion request logs under `server`. QA records that
    on the first real request.

### Verdict

**Approve with suggestions.**
- **Approved now:** items 1, 2 and 4 pass outright.
- **Nit only:** item 3.
- **Text fixes needed before the U5 steps they govern:** items 5 and 6 each leave one gap, P3-1
  and P3-2. Neither touches U3's implementation.

Both new findings come from the v3 fix text, and each is a one-paragraph edit with its fix stated
above. Another plan revision cycle is not worth it: QA's U5 test plan can carry P3-1/P3-2 verbatim
and be checked against them at the U5 gate.

## Pass 4 — code, step 1

**Scope.** Unit U4a: the uncommitted working-tree diff for plan Step 1
(`falkor-chat/docs/plans/extract-transient-retry.md` v3, §3.1, §3.2, §3.6, §4 Step 1, §5 Step 1),
against baseline `d497a06d`. Files under `falkor-chat/server/`: `falkorchat/transport.py`,
`falkorchat/llm.py`, `falkorchat/embedding.py`, new `falkorchat/retry.py`,
`tests/test_transport.py`, `tests/test_llm.py`, `tests/test_embedding.py`, new
`tests/test_retry.py`. Steps 2–3 (wiring, gate query, docs) are out of scope.

**Verdict: approve with suggestions.** No blocker and no major. Four minors/nits, all of them
test pins or plan-text follow-ups; none changes behaviour.

CPG: used cpg_falkorchat — confirmed the baseline's production `ProviderCallError` raise sites (7 in `transport._transport`, 2 each in `llm` `chat`/`complete`, 1 in `embedding.embed`) are exactly the sites the diff touches; `retry.py` is uncommitted, so it is not in the graph.

### Evidence gathered (all executed)

- **Suite.** `.venv/bin/python -m pytest tests/test_retry.py tests/test_transport.py
  tests/test_llm.py tests/test_embedding.py` → **172 passed**. Collection splits it **91 baseline
  + 81 new** (`test_u<n>` ids). `git diff d497a06d -- tests` has **0 deleted lines**. The
  implementer's counts hold.
- **Live ladder probe.** I drove the real `urllib.request.urlopen` through `make_http_transport`
  against local raw-socket servers (no FalkorDB, no LM Studio). The table is in the Appendix. Every
  rung lands where §3.1 says. That includes rung 3a for both `InvalidURL` shapes (`nonnumeric
  port`, space in host), `BadStatusLine` → `network`, and `IncompleteRead` from a short body →
  `read` with `__cause__` kept. It also includes a real HTTP 400 `{"error":"terminated"}` →
  `http`/400/transient.
- **Mutation probe**, 13 mutants. Each file was copied aside, mutated, run with `-x`, restored,
  and checked byte-identical with `filecmp.cmp(shallow=False)`. Results are in the Appendix: 9
  killed and 4 survived. The survivors are findings P4-1 to P4-3.
- **Gate-raises re-raise probe.** I scripted the gate as `[RuntimeError, False]`. The re-raised
  object `is` the original provider error, with `__context__`/`__cause__` both `None`. So the
  nested `except` inside `_gate_open` does not disturb the outer bare `raise`.

### Answers to the brief's focus questions

1. **`call_with_retry` vs §3.6: it matches.** The order is:
   1. transient check, with a bare `raise` and no log;
   2. last-attempt check, then a `giving up after %d attempts (%s, kind=%s, status=%s): %s`
      WARNING, then re-raise;
   3. pre-sleep gate;
   4. `delay = delays_s[i]·(1 + jitter·rand())`;
   5. the retry WARNING, in the §3.6 format;
   6. `sleep`;
   7. post-sleep gate;
   8. loop.

   The gate INFO line is `"%s: owning document no longer processing; abandoning retry (kind=%s,
   status=%s)"` with `label`, as v3/N-5 requires. It is emitted from either read.

   Only `Exception` is caught, and U-8 pins `KeyboardInterrupt`. `sleep` and `rand` are both
   injectable keyword parameters with the §3.6 defaults (`time.sleep`, `random.random`), so jitter
   is fully deterministic in tests. Every test passes `rand=lambda: 0.0` or `0.9999999`.

   **A raising gate treated as open, logged via `_log.exception`, is per plan.** §5 U-8b states
   "`keep_going` raises → treated as `True`: retries continue and an error is logged". It is also
   the right call: a gate read fails when FalkorDB is unreachable, and failing the provider retry
   for that reason would turn a recoverable hiccup into a `failed` document. The worst case is
   six ERROR tracebacks per job (pre+post × 3); see P4-4.
2. **Ladder order and scope: correct.**
   - Rung 3a sits immediately before rung 3. Swapping them was killed by the U-1 3a row.
   - Rung 3b wraps only `raw = resp.read()`, and `close()` stays in the `finally`. U-2 asserts
     `resp.closed` for all three read failures.
   - `HTTPError` cannot reach rung 3, because rung 1 catches it first. JSON decode and body checks
     sit after the `try/finally`, so a body-level error is never re-wrapped as `read`.
   - `ProviderCallError` (a `RuntimeError`) is neither `OSError` nor `HTTPException`, so no rung
     re-wraps it.
   - The new `HTTPException` reach of rung 3 covers `UnknownProtocol`, `LineTooLong`,
     `ImproperConnectionState` and similar. All are wire-level, and `network` is defensible for
     them.
3. **R-6: nothing observable beyond R-6, except one omission in its list (P4-3).**
   - Messages: every pre-existing message is byte-identical. The diff only adds keyword arguments,
     and 91 baseline tests that assert the strings are unchanged and green. Only the newly wrapped
     shapes get new messages.
   - Types: `FallbackClient`'s exhausted error is still a bare `ProviderCallError`, still raised
     outside any `except`. Its `__context__` is `None` as before, and the message is identical
     (U-3 asserts the full string).
   - Callers, from a grep for `except.*ProviderCallError|OSError|HTTPException|TimeoutError` in
     `falkorchat/`:
     - The provider path has only `llm.py:217/228` (`FallbackClient`) and the `services.py:2588`
       drive-fault catch that R-6 names.
     - `db.py:46` and `modelconfig.py:303/357` catch `OSError` around Redis and file reads, not
       provider calls, so they are unaffected.
     - `tools.py:929` documents that it does **not** catch `ProviderCallError`, and that is
       unchanged.
4. **Test quality: good.**
   - No test checks only `kind` where the classifier also reads `status`/`detail`. Rung 1 pins
     `status==400` and `detail`, and rung 5 pins `detail=="terminated"` exactly.
   - Stub values are distinct where swaps matter:
     - U-6 uses `http/503`, then `connection/None`;
     - U-7 uses statuses 500–503, so `status=503` in the give-up line pins the **last** error;
     - U-3 uses `http/503` and `connection` with distinct refs, plus an exact message.
   - No fixture invents keys. The verbatim LM Studio bodies come from §2.2.
   - There are three weak pins (P4-1, P4-2, nit in P4-4).
5. **Departures: all sound.**
   - **Per-marker U-5 rows.** Correct and valuable: the verbatim load-failure body carries two
     markers, so without per-marker rows deleting either marker would survive.
   - **Unplanned plumbing test** (`test_u5b_call_with_retry_passes_the_policy_timeout_switch_through`).
     Sound. It is the only step-1 pin that `call_with_retry` forwards `policy.retry_timeouts`, and
     U-17 in step 3 builds on that.
   - **Log levels.** These are WARNING/WARNING/INFO as §3.6 specifies, plus ERROR for a raising
     gate, which the plan's "an error is logged" asks for.
   - **`_exhausted()` refactor.** Sound. `zip(self._refs, caught)` is length-safe, because every
     loop iteration that does not `return` appends exactly once. Dropping `enumerate` in `complete`
     loses nothing.
   - **`jitter < 0` validation.** This is §3.2's stated invariant, not a departure.
   - **Plan note.** No plan revision is needed. One text omission in R-6 (P4-3) can ride along with
     whichever step-2/3 doc edit touches the plan next, or the ledger can record it.

### Findings

#### P4-1 (minor) — U-7 does not pin the third delay, the 30 s point §3.2 derives

- **Evidence.**
  - Mutant `delay = policy.delays_s[min(attempt - 1, 1)]` sleeps `[2, 8, 8]` instead of
    `[2, 8, 30]`. It **survived**: 172 passed.
  - U-6 asserts only the first two delays. U-7 asserts `len(sleep.delays) == 3`, but not the
    values.
- **Why it matters.** Attempt 4 at 40–60 s cumulative is the design's whole argument for clearing
  the ~21 s cold load (§3.2). An off-by-one index or a truncated tuple would quietly move it to
  ~18 s, inside the reload window K-067 warns about.
- **Fix.** U-7 already passes `rand=lambda: 0.0`. Add `assert sleep.delays == [2.0, 8.0, 30.0]`.

#### P4-2 (minor) — The give-up WARNING's `label` is unpinned

- **Evidence.**
  - Mutant: the `giving up after …` line logs `""` in place of `label`. It **survived**.
  - U-7 calls `call_with_retry` with no `label`, so the line it checks has an empty label either
    way.
- **Why it matters.** This line is the only record of a job that spent its whole budget. Live leg
  (g)'s per-job table and the P3-1 job set attribute outcomes by `extract|embed chunkId=…`. A
  label-less give-up line cannot be joined to its job.
- **Fix.** In U-7, pass `label="extract chunkId=c1"` and assert that it appears in
  `giving_up[0]`.

#### P4-3 (nit, plan text) — R-6's list of newly wrapped raw escapes omits the read-phase `TimeoutError`

- **Evidence.** Rung 3b also wraps a `TimeoutError` raised by `resp.read()` (U-2 `timeout` row,
  `kind="timeout"`). Before this change, that error escaped raw. So `FallbackClient` now falls
  through on a body-read timeout, and `services.py`'s drive-fault catch now sees it. R-6 (plan
  §6) lists only `ConnectionResetError`/`IncompleteRead`/`BadStatusLine`/`InvalidURL`.
- **Why it matters.** The behaviour is right; it is the same improvement R-6 describes. But R-6 is
  the reference list of contract changes on a shared seam, and an incomplete list misleads the
  next reader auditing the `FallbackClient` change.
- **Fix.** `architect`/`teco`: add "and a read-phase `TimeoutError` (as `kind=timeout`)" to the
  R-6 first bullet at the next plan touch. This needs no code change.

#### P4-4 (nits; take or leave)

- **5xx upper bound.** Mutant `<= 598` survived, because no U-5 row covers `599`. Add `http 599`
  to transient. Optionally add `http 499` to permanent as the lower neighbour; `500` already pins
  the low edge, since mutant `501 <=` was killed.
- **Rung-1 `detail`.** U-1 uses `"terminated" in exc.detail`. A mutant that sets `detail` to the
  whole message survived. That is harmless to classification today, but `detail` is documented as
  "the provider's own error text". `assert exc.detail == '{"error":"terminated"}'` pins it.
- **Retry WARNING's trailing `%s`** (the exception text). Its removal survived. Optional: assert
  `"boom"` appears in `warnings[0]`.
- **Raising gate logs a full traceback per read.** That is up to 6 ERROR tracebacks per job while
  FalkorDB is down. This is acceptable as specified. If it proves noisy live, log the traceback
  once per call, not per read. No action is needed now.
- **Marker `"terminated"` is a substring of `"unterminated"`** (checked: `'terminated' in
  'unterminated string literal'` → `True`). A permanent 400 whose body says "Unterminated …"
  would burn one budget. This is **inferred, not observed**: I have no LM Studio body showing it,
  and the cost is bounded by §3.2. Route it to `architect` only if leg (g) captures one; if so,
  the fix is a word-boundary match in `_has_marker`. It is plan-level, not an implementation
  defect.

### What's solid

- `retry.py` is a direct, readable transcription of §3.1/§3.6. `_gate_open` factors the two gate
  reads without changing the order. The docstring states the order contract that tests pin.
- `test_u8b_gate_is_read_before_and_after_each_sleep` asserts on a shared event log, and
  `…_never_read_after_the_final_attempt` counts exactly 6 reads. Together they pin the M-1 fix
  structurally rather than by inference.
- The U-2 rows assert `exc.__cause__ is read_exc` and `resp.closed`, which pins rung 3b's scope
  and the `finally`.
- The baseline tests are untouched (0 deleted lines), so the "message byte-identical" pin is the
  existing 91 tests, as the plan intended.

### Open questions

None. P4-1/P4-2 can be folded into U3b's first commit or a one-line follow-up before the step-1
commit, at `teco`'s discretion.

### Appendix — Pass 4 probes

**A. Real `urlopen` through `make_http_transport`** (script:
`/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/cc651677-ac22-4370-9988-76881753007e/scratchpad/u4a/probe_live.py`):

| scenario | kind | status | transient | `__cause__` |
|---|---|---|---|---|
| `http://localhost:abc/…` | `malformed_request` | — | False | `InvalidURL` |
| `http://local host/…` | `malformed_request` | — | False | `InvalidURL` |
| server sends `GARBAGE\r\n\r\n` | `network` | — | True | `BadStatusLine` |
| server closes without reply | `network` | — | True | `RemoteDisconnected` |
| 200, `Content-Length: 100`, 5 bytes sent | `read` | — | True | `IncompleteRead` |
| 400 `{"error":"terminated"}` | `http` | 400 | True | `HTTPError` |
| 200 `{"error":{"message":"Failed to load model x"}}` | `body_error` | — | True | — |
| closed port | `connection` | — | True | `URLError` |

**B. Mutants** (script: `/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/cc651677-ac22-4370-9988-76881753007e/scratchpad/u4a/mut.py`;
all restored, `filecmp` byte-identical):

| mutant | result |
|---|---|
| rung 3a moved after rung 3 | killed |
| rung 3b tuple drops `HTTPException` | killed |
| rung 3 tuple drops `HTTPException` | killed |
| rung 3b `kind` always `"read"` | killed |
| raising gate → closed | killed |
| post-sleep gate removed | killed |
| 5xx lower bound `501` | killed |
| rung 2 `kind` always `"connection"` | killed |
| 5xx upper bound `598` | **survived** (P4-4) |
| third delay `delays_s[min(i,1)]` | **survived** (P4-1) |
| give-up line drops `label` | **survived** (P4-2) |
| retry WARNING drops exception text | **survived** (P4-4) |
| rung 1 `detail` = whole message | **survived** (P4-4) |
