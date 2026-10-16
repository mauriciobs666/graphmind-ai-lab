# Review — Transient provider-error retry for document-ingestion background jobs

> **Status:** archived · **Owner:** `analyst` · **Tracks:** K-067

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

## Pass 5 — code, step 2

**Scope.** Unit U4b: the uncommitted working-tree diff for plan Step 2, against `91b65fed`. The
plan sections are §3.3, §3.5, §3.6, §4 Step 2 and §5 Step 2. The files, under
`falkor-chat/server/`, are:
- `falkorchat/ingestion.py`;
- `falkorchat/repository.py` (new `chunk_document_is_processing`);
- `tests/test_ingestion.py`;
- `tests/test_background.py`;
- `tests/test_repository.py`.

Pass 4's dispositions: P4-1, P4-2 and the 599 row (P4-4) are **fixed** in `91b65fed`, per the
coordinator.

**Verdict: approve with suggestions.** There is no blocker or major. The wiring, the boundary and
the gate are correct. The two minors are three wiring facts that no test pins, each closed by a
small test addition.

CPG: considered, not relevant — the change is a one-call wrap plus one new read query; I read every caller and callee directly, and the CPG predates the uncommitted diff.

### Evidence gathered (all executed)

- **Suites.**
  - `tests/test_ingestion.py tests/test_background.py`, run whole and live on `ws:test`, gave
    **61 passed** with no skips (`-rs`).
  - `test_repository.py -k chunk_document_is_processing` gave **5 passed, 328 deselected**.
  - Both match the implementer's figures. The slowest call is 0.25 s (fixture setup), so nothing
    sleeps for real.
- **Gate EXPLAIN**, run on `ws:test` and `ws:agent-team`:
  `Node By Index Scan (c:Chunk)` → `Conditional Traverse (c)->(d:Document)` → `Project`. This
  matches §3.5 and QUERIES' shape.
  - `HAS_CHUNK` points Document→Chunk. That agrees with I-1's own count query and with the live
    I-2 rows passing.
  - A `null` status evaluates to `null = 'processing'` → `bool(None)` → `False`, which is safe.
  - An empty result set returns `False`.
- **Mutation probe**, 8 mutants. Each file was copied aside and restored, `filecmp`
  byte-identical, with the same md5 before and after. 4 were killed and 4 survived (Appendix C).
- **Model resolution.** `ModelGateway.llm` → `resolve_llm` → `resolve` + `_client_for_chain`.
  Those steps build the clients (`make_http_transport` at `modelconfig.py:575/589` only
  constructs a transport) and make **no provider call**. A workspace-override read is a Redis
  `ro_query`, and its errors are not `ProviderCallError`.

### Answers to the focus questions

1. **Gate Cypher: correct.** The `HAS_CHUNK` direction is right and the query goes through
   `ro_query`. It hits the index and returns `False` on no row; mutants of the `False`-on-no-row
   and status-test branches were killed. Making it read through `.query` instead of `.ro_query`
   survives, but that mutant is behaviour-equivalent for a query with no write, so there is
   nothing to pin.
2. **Retry boundary: exactly the provider call.**
   - `lambda: extraction.extract(text, llm)` is the only thing wrapped. The write loop sits after
     `call_with_retry` returns.
   - U-9b (pipeline altitude) and U-9c (`_safe_extract` altitude) both pin the boundary against
     widening.
   - `llm` resolution sits outside the retry, and that is correct, as the plan intends. Resolution
     makes no provider call, so it cannot raise a transient `ProviderCallError`. Its own failures
     (`ModelResolutionError`, a Redis error from the overrides read) are permanent or
     infrastructure faults, and they fail fast exactly as before. Retrying them would mean
     retrying graph errors, which §3.1 explicitly excludes.
3. **I-1's absolute counts follow from the stub.**
   - The reply has one entity and one relationship whose object (`Acme`/`Bolt`) is not an
     entity. Stub-repair makes that 2 entities, 2 `ABOUT` (each entity linked to the one chunk),
     and 1 `RELATES_TO`.
   - With distinct names there is no exact-tier fusion, so the confirmed `SAME_AS` count is 0.
     That cell is therefore a constant, not a comparison. The plan's distinct-name instruction
     makes this so, it is accepted, and `A == B` on the other three is the load-bearing part.
   - Using distinct *texts* (departure 1) is necessary and correct: same text auto-supersedes.
   - The orphan check is whole-graph, as specified.
4. **Real-sleep hazard (departure 4): genuinely latent today.**
   - Every pre-existing `IngestionPipeline(...)` built without `sleep=` uses either `_ReplyLLM`
     (`test_ingestion.py`) or `_StubExtractionLLM` (`test_api.py:1015`). Neither ever raises. The
     other pipelines in `test_mcp.py`/`test_background.py`/`test_api.py` are recording fakes.
   - `app.py:619` builds a real pipeline only under `ENABLE_AGENT`. No test that sets it ingests
     a document: `grep -l ENABLE_AGENT tests | xargs grep -l 'ingest|documents'` matches only
     `conftest.py`.
   - Nothing sleeps for real now (durations above). **Forward note for U4c:** step 3 brings the
     same hazard to `EmbeddingWorker`. 23 test constructions pass no `sleep=`, and some
     `test_embedding.py` `embed_chunk` tests drive provider failures (around lines 337 and 363).
     If any of those failures is transient-classified, it will sleep 40–60 s for real. U4c checks
     this.
5. **Interaction with step 1.**
   - The label is `extract chunkId={chunk_id}`, which matches §4 Step 2.3 and is the token P3-1
     and §3.2.1 condition 3 match on.
   - The INFO and WARNING lines carry `kind`/`status` from step 1's helper.
   - **But no step-2 test asserts the label** (P5-1).

### Findings

#### P5-1 (minor) — Two pieces of wiring are unpinned: the job label and the injected `retry_policy`

- **Evidence.** Two mutants survived, both at 66 passed:
  - `label=f"extract chunk={chunk_id}"`;
  - `policy=retry.DEFAULT_PROVIDER_RETRY`, which ignores `self._retry_policy`.
- **Why it matters.**
  - The label is the join key the live proof uses (P3-1, §3.2.1 condition 3, leg (c′)). A
    drifted label passes every unit test and then fails silently in U5, where diagnosis is
    expensive.
  - The `retry_policy` seam is the constructor contract §3.2 names. Today it is
    behaviour-equivalent only because nobody injects a non-default policy.
    `test_pipeline_defaults_…` cannot tell "default used" from "injected ignored".
- **Fix.**
  - In U-12 (gate closed), add `caplog` and assert that the INFO line contains
    `extract chunkId=chunk-7`.
  - Add one row:
    `IngestionPipeline(..., retry_policy=RetryPolicy(max_attempts=2, delays_s=(0.0,)), sleep=fake)`
    with an always-transient LLM. Assert `llm.calls == 2`.

#### P5-2 (minor) — The retried call's input is unpinned (the coordinator's `text[:1]` survivor), and this diff now owns that line

- **Evidence.**
  - The `extract(text[:1], llm)` mutant survived, as the coordinator found.
  - In `test_ingestion.py` it is **equivalent by construction**: every new row passes `text="t"`,
    so `"t"[:1] == "t"`.
  - Every LLM stub in the diff ignores its input: `_ScriptedLLM.complete(messages)` and
    `_CountingScriptedLLM.complete(messages)`, like the pre-existing `_ReplyLLM`.
- **Ruling on Q1: fix it in this change, not as a follow-up.** The gap predates step 2, but step 2
  rewrites that exact call into a closure that is re-invoked per attempt. "Every attempt sends
  the same request built from this chunk's text" is now a retry property. A future edit that
  rebuilds or truncates the prompt per attempt would pass every row.
- **Fix.** In U-9:
  - give `_ScriptedLLM` a `seen: list` that records `messages`;
  - pass a multi-word `text` (for example `"Alice works at Acme."`);
  - assert that all three recorded `messages` are equal and contain the text.

  That is three lines, and the same check is optional in I-1.
- **Ruling on Q2: yes, both LLM stubs ignore their input, and nothing else does.**
  - The `SpyRepo` gate records `(ws, chunk_id)`, and U-12 pins it. The `ws`/`chunk_id` mutants
    were killed (6 failed).
  - `_WritePhase503Repo.chunk_document_is_processing` ignores its arguments. That is harmless,
    because U-9c never reaches the gate.
  - The fake sleeps record delays.
  - So the input-blindness is confined to the LLM stubs, and P5-2's fix covers it.

#### P5-3 (nit) — Departures: all sound; one wording point

- **I-1 distinct texts.** This one is required: same text would trigger auto-supersede.
- **Extra live I-4 row.** It is valuable. It is the only end-to-end pin of the post-sleep read
  against real FalkorDB (one call, one sleep, `failed`, 0 entities), so a gate that is wired but
  wrong would turn it red.
- **Default-policy test asserts behaviour, not private attributes.** This is correct in spirit,
  but see P5-1: it cannot distinguish "default" from "ignored injection".
- **`import` blocks mid-file with `# noqa: E402`** in both test files. Both files already put
  live sections at the bottom, so this is consistent enough. If either file gets a later touch,
  hoisting the imports is cleaner.

### What's solid

- The docstring added to `extract_chunk` states the boundary and why it must not widen
  (fresh-uuid `CREATE`s), so the next editor meets the constraint at the call site.
- The I-2 rows seed **two documents in opposite states**, and the delete row keeps a third
  `processing` document alive. A query keyed on the wrong id, or on "any processing document",
  goes red, as the gate mutants confirm (4 failed).
- The I-4 `seen_during_sleep == [("processing", 2)]` check observes the live document **from
  inside the sleep**, which is exactly the "retrying ≠ reported" claim in §3.5.
- U-11/U-13 also assert `gate_reads == []`, which pins "the gate runs only on the transient path"
  at the pipeline altitude.

### Open questions

None.

### Appendix C — Pass 5 mutants

Script:
`/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/cc651677-ac22-4370-9988-76881753007e/scratchpad/u4a/mut2.py`.
Run over `test_ingestion.py`, `test_background.py` and `test_repository.py -k
chunk_document_is_processing` (66 tests).

| mutant | result |
|---|---|
| gate keyed on `document_id` | killed (6 failed) |
| `keep_going=None` | killed (3) |
| gate `RETURN d.status <> 'deleted'` | killed (4) |
| gate `True` on no row | killed (2) |
| label `extract chunk=` | **survived** (P5-1) |
| `retry_policy` not forwarded | **survived** (P5-1) |
| `extract(text[:1], llm)` | **survived** (P5-2) |
| gate `.query` instead of `.ro_query` | survived — behaviour-equivalent, no action |

## Pass 6 — code, step 3

**Scope.** This is unit U4c: the uncommitted diff for plan Step 3 against `128941a6`, read with
plan sections §3.2.1, §3.4, §4 Step 3 and §5 Step 3. It covers:
- `server/falkorchat/embedding.py` (the `EmbeddingWorker` constructor and `embed_chunk` wrap);
- docstring-only changes to `background.py` and `repository.py`;
- `docs/QUERIES.md` §14.1a;
- `tests/test_embedding.py`;
- `tests/test_background.py` (I-3).

The diff also edits the coordination ledger; that is `teco`'s and is not reviewed here.

**Pass 5 dispositions:** P5-1 and P5-2 are fixed in `128941a6`, per the coordinator. Step 3's
U-14 also applies the P5-2 lesson, with a multi-word `_CHUNK_TEXT` and `seen == [text] * 3`.

**Verdict: approve with suggestions.** There is no blocker or major. The wiring is correct and
mirrors step 2. The one minor is that §3.2.1's flip is not the edit the plan says it is (P6-1).

CPG: considered, not relevant — a one-call wrap plus docstrings/docs; I read every function the wrap spans (`_resolve_and_embed`, `_index_dimension`, `ModelGateway.embedder`/`resolve`) directly, and the CPG predates the uncommitted diff.

### Evidence gathered (all executed)

- **Suites.**
  - I ran the six files the coordinator named (`test_embedding`, `test_background`,
    `test_ingestion`, `test_retry`, `test_transport`, `test_llm`) plus `test_responder`: **258
    passed** in 0.60 s. That is the coordinator's 246 plus the responder's 12.
  - The slowest call took 0.02 s, so nothing sleeps for real.
  - Every *pre-existing* `embed_chunk` failure test in `test_embedding.py` (around lines 339–390)
    raises `EmbeddingDimensionError`, which is non-transient. The other test files that construct
    `EmbeddingWorker` without `sleep=` (`test_api`, `test_workflow_live`,
    `test_ac5_chat_grounding_live`) script no provider error.
  - So my Pass 5 forward note resolves to **latent, not live**.
- **Mutation and flip probe.** I ran five mutants on copies; each file was restored and checked
  byte-identical with `filecmp`, and the md5 matched before and after. Appendix D has the
  details. All five were killed, including the §3.2.1 flip itself, which is the subject of P6-1.
- **Identity.** `retry.DEFAULT_PROVIDER_RETRY is retry.EMBED_CHUNK_RETRY` → `False`, while `==` →
  `True`.
- **Docs grep.**
  - `grep -n 'pendingJobs|report_document_job|K-051|_safe_extract|_safe_embed_chunk|retry|Document.status' docs/SERVER.md`
    returns only line 63, an unrelated "idempotent retry".
  - A broader `ingest|background|extract` grep finds no description of the document lifecycle.

### Answers to the focus questions

1. **The pre-flight inside the retry is safe, and it is what the plan specifies.** §3.4 says to
   wrap `_resolve_and_embed` and argues that re-running the pre-flight is cheap. I read every raise
   inside that region:
   - `ModelGateway.embedder`/`resolve` build clients and never make a provider call;
     `make_http_transport` only constructs.
   - Their failures are `ModelResolutionError` or a Redis error from the override `ro_query`.
   - `_index_dimension` → `read_index_dimension` is a Redis `ro_query`.
   - Both dimension checks raise `EmbeddingDimensionError(Exception)`.
   - None of these is a `ProviderCallError`, so `is_transient` is `False`, it re-raises on
     attempt 1, and the gate is never read.
   - A permanent misconfiguration therefore cannot be hidden by a transient error. The pre-flight
     runs **before** the provider call on every attempt, so a permanent fault surfaces on attempt
     1, before any transient error can occur.
   - U-15 pins this: an `index_dim=8` mismatch gives `embedder.calls == 0`, no sleeps, no gate
     reads, and no write.
   - The only cost is that a non-`None` index dimension is cached, so a retry re-reads at most the
     two cheap gateway reads. That is the cost §3.4 states.
2. **The `inspect.signature` identity check is sound.**
   - It is the only check that can tell `EMBED_CHUNK_RETRY` from the equal-valued
     `DEFAULT_PROVIDER_RETRY`, because they are distinct objects. The mutant that defaults to
     `DEFAULT_PROVIDER_RETRY` was killed.
   - Its one brittleness is a refactor to a `None`-sentinel default. That turns it **red**, never
     falsely green, which is the safe direction.
   - `test_worker_uses_an_injected_retry_policy` covers "injection ignored", so between them the
     pair closes the P5-1 class for the worker.
3. **The timeout switch reaches the worker, but the flip is not a one-constant edit (P6-1).**
   - The switch-on U-17 row injects `dataclasses.replace(EMBED_CHUNK_RETRY, retry_timeouts=True)`
     and passes through the worker, so the plumbing is pinned.
   - I flipped `EMBED_CHUNK_RETRY` in a copy of `retry.py`. Two tests went red:
     `test_retry.py::test_u5b_both_shipped_policies_leave_timeout_retry_off`, which §3.2.1
     done-condition 2 names, and `test_embedding.py::test_u17_shipped_embed_policy_does_not_retry_a_timeout`,
     which no done-condition names.
4. **Docs are accurate.**
   - The QUERIES.md §14.1a block's Cypher is byte-identical to `repository.py`.
   - Its EXPLAIN text matches my Pass 5 re-run on `ws:test` and `ws:agent-team`.
   - "Only on the failure path" is slightly loose: precisely, the gate runs only after a *transient*
     failure, and never after the final attempt. The `retry.py` docstring already says this, so it
     is a nit with no action.
   - The `background.py`, `repository.py` and `embed_chunk` docstrings say what the code does.
     "Gateway resolution and the index-dimension read are read-only" is verified above.
   - **"SERVER.md needs no change" is right** by the grep. One adjacent fact is in P6-2.
5. **`test_queries.sh` gate case: recommend no.**
   - §14.1a's sibling progress queries (`start_document_progress`, `report_document_job_done`)
     are `[verified]` by `tests/test_repository.py` alone. `test_queries.sh` line 1641–1642 says
     outright that "§14.1 … has no test_queries.sh harness yet".
   - The gate already has stronger live coverage than a script case would add:
     - five I-2 rows, built from two documents in opposite states;
     - I-4's end-to-end post-sleep row;
     - I-3, all on `ws:test`.
   - A script case would duplicate that coverage.
   - It would also cost a coordinated run of a script that is destructive to `reference`.
   - Revisit only when §14.1 as a whole gets a script harness.
6. **Cross-step consistency: consistent, and the P3-1 join works.**
   - The labels are `extract chunkId={chunk_id}` and `embed chunkId={chunk_id}`. Both use the raw
     id, as does the `_safe_*` traceback line `background … failed (chunkId=%s)`
     (`background.py:114/148`). So one `chunkId=<id>` token joins every line a job emits: retry
     WARNING, give-up, gate INFO and the final traceback.
   - Defaults: extract uses `DEFAULT_PROVIDER_RETRY` and embed uses `EMBED_CHUNK_RETRY`, as §3.2
     says.
   - Both call the same gate method with the same `(ws, chunk_id=chunk_id)`.
   - The label is pinned on both sides: by P5-1 for extract, and for embed by three U-14 rows that
     catch the `extract`/`embed` swap mutant.
   - Note that `id_desc=f"chunkId={chunk_id!r}"` (quoted) appears only inside
     `EmbeddingDimensionError` text. That error is non-transient, so it never meets a retry line,
     and the join is unaffected.

### Findings

#### P6-1 (minor, plan text and a test comment): §3.2.1's flip turns a second, unnamed row red, and the "re-point" comment points at the wrong row

- **Evidence.**
  - The flip probe (Appendix D) fails both `test_u5b_both_shipped_policies_leave_timeout_retry_off`
    and `test_u17_shipped_embed_policy_does_not_retry_a_timeout`.
  - §3.2.1 says "One constant. No other production code changes … The flip turns U-5b's
    shipped-policy row red by design, so the same edit updates that row." Its done-conditions 1–2
    name only U-17's parametrization and U-5b.
  - The comment above the switch-on row says the flip "reuses this row (parametrize over the
    flipped constant)". But that row injects its own policy, so it stays green before and after
    the flip, and re-pointing it adds nothing.
- **Why it matters.** The engineer doing the flip under §3.2.1's pre-authorization would hit an
  unlisted red test. They might "fix" it by weakening it (for example, deriving the expectation
  from `EMBED_CHUNK_RETRY.retry_timeouts`), which would turn the only default-worker timeout pin
  into a test that can never fail. The row that actually proves "the flipped constant reaches a
  default-constructed worker" is the shipped-off row, **inverted**.
- **Fix.**
  - In this change: rewrite the test comment to say that on the §3.2.1 flip,
    `test_u17_shipped_embed_policy_does_not_retry_a_timeout` is red by design. It is then inverted
    to "a default worker retries a timeout" (`calls == 2`, one sleep), with its expectation
    **hard-coded**, never derived from the constant.
  - `architect`/`teco`: add that row to §3.2.1 done-condition 2 alongside U-5b. This is a text
    edit with no code change.

#### P6-2 (nit, U5 setup): the retry widens the window in which the default `--reload` kills an in-flight job

- **Evidence.**
  - `scripts/start_server.sh:108` defaults `UVICORN_ARGS` to `--reload`.
  - `SERVER.md` ~line 560 documents that a reload "silently kills every in-flight background
    daemon thread".
  - Before this change a job was in flight for about one provider call. Now a retrying job stays
    in flight for typically 40–60 s, and up to < 780 s in the worst case (§3.2). If the job is
    killed mid-sleep, it never reports, so its document stays `processing` with `pendingJobs > 0`
    forever.
- **Why it is only a nit.** The plan's live-proof setup already requires `--reload` off (plan line
  ~1136), and production deployment is not `start_server.sh`'s dev default.
- **Optional fix.** Add one clause to that `SERVER.md` gotcha: "a job retrying a transient
  provider error (K-067) is in flight for up to a minute or more, so the exposure is larger". This
  is `teco`'s call at close. It does not change the "no lifecycle text in SERVER.md" conclusion.

#### P6-3 (nit): departures, all sound

- **Extra U-14 rows** (write-phase error not retried, exhausted, gate-closed): sound. They mirror
  U-9b/U-10/U-12 and pin the label on three lines.
- **The `inspect.signature` default check:** sound (see answer 2).
- **The U-17 split and parametrization:** sound as plumbing, but see P6-1 on what its comment
  promises.
- **I-3's second, untouched document:** sound. It catches a gate or write keyed on the wrong
  chunk or document.
- **Pre-flight inside the retried region:** not a departure. §3.4 specifies it (see answer 1).

### What's solid

- U-14's main row pins five things in one pass: `seen == [text] * 3`, one write, the
  `("acme","chunk-7")` gate key × 4, the label prefix on both WARNINGs, and `repo.calls == []` (no
  `Message` write).
- U-16 pins §3.4's exclusion of `embed_message` directly: one call, no sleep, no gate read.
- I-3 checks the live vector with `pytest.approx` and the document reaching `ready, 0`, and it
  confirms that B's chunk and progress are untouched.

### Open questions

None. P6-1's test comment can ride along with this step's commit; the plan line is a text edit.

### Appendix D: Pass 6 mutants and flip probe

Script:
`/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/cc651677-ac22-4370-9988-76881753007e/scratchpad/u4a/mut3.py`.
It runs over `test_embedding`, `test_retry`, `test_background` and `test_responder` (143 tests).

| mutant | result |
|---|---|
| **§3.2.1 flip**: `EMBED_CHUNK_RETRY(..., retry_timeouts=True)` | 2 red: `test_u5b_both_shipped_policies…` **and** `test_u17_shipped_embed_policy…` (P6-1) |
| worker default → `DEFAULT_PROVIDER_RETRY` | killed (1): the identity check |
| label `embed` → `extract` | killed (3): the U-14 rows |
| injected `sleep` ignored | killed (6) |
| `keep_going=None` | killed (2) |

## Pass 7 — code, D-2 marker fix

**Scope.** This pass reviews the uncommitted diff against `1b55e404`. It adds the marker
`"model unloaded"` to `retry.TRANSIENT_DETAIL_MARKERS` and one U-5 transient row using the
verbatim live body. The baseline is D-2 in `falkor-chat/docs/test-reports/extract-transient-retry-report.md`
(Half B, sections B.3 TP-101 and B.6). Pass 6 findings are out of scope here.

**Verdict: approve.** There are no findings above nit.

CPG: considered, not relevant — a one-string allow-list addition; verified by executing the classifier and transport directly.

### Evidence gathered (executed)

- **Suites.** `test_retry.py` and `test_transport.py` give **91 passed**.
- **Mutant.** I deleted the marker line from a copy of the file. Result: **1 failed**, the
  `http 400 model unloaded (D-2)` row. I restored the file and `cmp` confirmed it is identical.
- **Q1: rung 1 does fill `detail`.** I ran `make_http_transport` with an opener that raises
  `HTTPError(400, "Bad Request", fp=<D-2 body>)`. It returned:
  - `kind='http'`, `status=400`;
  - `detail='{"error":"Model unloaded by user or API request."}'`;
  - a message byte-identical to TP-101's traceback line;
  - `is_transient` → `True`.

  The traceback does not show `detail` because `detail` is an attribute, not part of `str(exc)`.
  The message carries the same body, so the live line is enough to confirm the shape. Pass 4
  Appendix A already exercised the same rung through a real `urlopen` socket round-trip.

### Answers

- **Q2: false-match risk is low, and if it happens the cost is bounded.** I ran `_has_marker`
  against these strings:

  | text | matches? |
  |---|---|
  | `No models loaded. Please load a model…` | no |
  | `Model not found` | no |
  | `model is not loaded` | no |
  | `Failed to unload model` | no |
  | the D-2 body | yes |

  The existing permanent row `body_error no marker` ("No models loaded") stays green.
  - The likely follow-on case is an ejected model with JIT loading off. There, attempt 2 should
    get a no-model-loaded style body, which is permanent, so the job fails after one 2–3 s sleep.
    This is **inferred**: I did not run it against LM Studio.
  - This is the R-2 class of risk the plan already accepts.
  - The "by user" trade-off is acceptable. On the REST chain, the document's *next* job would
    JIT-reload the ejected model anyway. Retrying this job does not change what an operator
    experiences after an Eject.
- **Q3: one `http` row is enough.** `_has_marker` is a single function that both the `http`
  branch and the `body_error` branch call. Both branches are already pinned by marker rows
  (`body_error terminated`, `body_error no marker`), so a `body_error` row for the new marker would
  test the same line twice. No 200-with-error-key shape of this body has been observed.
- **The authorization reading is sound.**
  - R-1 (§6) states the pre-authorization generally: "adding a marker is a one-line change …
    pre-authorized by §3.2.1; legs (e)–(g) are the first real-fault sample".
  - Plan §5 (e) calls the unload "the closest reproducible relative of `terminated`".
  - §3.2.1's "(g)" is that bullet's local context, not a scope limit.
  - No plan revision is needed. A one-line ledger entry recording the reading is enough.

### Findings

- **P7-1 (nit).** The marker comment says "(Eject / lms unload / API)". TP-101 tested only
  `lms unload`. Whether a GUI Eject or a TTL/JIT auto-evict produces the same body is untested
  (report B.6). Optional fix: change the comment to "observed on `lms unload`; same body assumed
  for Eject/API".
