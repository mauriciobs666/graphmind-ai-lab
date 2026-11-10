# python-web-quirks on the knowledge base — golden set, targets, floor and chunking

> **Status:** active · **Owner:** `data-scientist` · **Tracks:** — (OQ-1 of `docs/requirements/python-web-quirks-kb.md`; FR-12, AC-3)

## 1. Question and recommendation

**Decision served:** the stakeholder approves a golden set and minimum targets so the requirement
leaves `Interviewing` (OQ-1, FR-12, AC-3). Method only; nothing here is run against `ws:agent-team`
beyond a few read-only probes.

**Recommendation (for approval):**

1. **Two sets, one tuning, one sealed.** Set A (dev, 42 positive + 14 negative, below) is the only
   set anyone may inspect failures on. Set B (sealed, 84 positive + 30 negative, authored by someone
   else, §5) is scored once, at acceptance.
2. **Gate on Set B**, floor-applied, top-5 (chunks, as `search_documents` returns them):
   - positive recall **≥ 0.90** (≥ 76 of 84) **and** every one of the 42 sections hit by at least one
     of its two queries;
   - negative rejection **≥ 0.90** (≥ 27 of 30; false accept counted against the 36 unit
     documents, §4.2), split: **15/15 off-domain** and **≥ 12/15 near-domain**.
3. **Keep the 0.43 floor for now; it is unproven for this content** (§6). A measurement, not an
   argument, decides it, and any change goes into `skills/agent-kb-retrieval/SKILL.md` for all users (FR-7).
4. **No K-100 blocker.** With the shipped chunker no unit is embedded head-only (§7). The
   chunking risks that do exist are tail chunks carrying no title, and one document filling several
   of the five slots.

## 2. Findings from the real system (all observed this run)

- The source at pin `3fcb171c`: 76,291 bytes; 42 `##` sections (split on `^## `); the section
  text totals 73,825 characters. Section size in characters: min 668, median 1,416.5, max 6,150
  (section 26). Over 2,000: 10; over 3,000: 5; over 5,700: 1.
- The migration ingests **36 unit documents** (42 sections; merged units: {2,34}, {21,22,23},
  {24,25}, {29,39}, {30,40}; `pwq_kb.py units --pin 3fcb171c`). A unit's text carries a header block
  (`Gotcha`, `Members`, `Versions`, `Verified`) before the bodies: unit text min 835, median 1,723.5,
  max 6,554 characters (u21), 83,751 in total. Over 5,700: 2 (u21, u26).
- Chunker `falkor-chat/server/falkorchat/chunking.py` default `size=1000, overlap=150`. Applied to
  each unit as its own document it yields **129 chunks** (`units --chunks`): 1 chunk x 2 units, 2 x 10,
  3 x 12, 4 x 4, 5 x 2, 6 x 2, 7 x 2, 9 x 1 (u21), 10 x 1 (u26); longest chunk 1,142 characters. The
  live workspace's longest chunk is 1,139 characters, so ingestion in practice uses that chunker.
- `search_documents` ranks **chunks**, no document dedupe, `score` = cosine distance
  (`falkor-chat/server/falkorchat/repository.py` `search_chunks`). Top-5 means five chunks.
- `ws:agent-team` (read this run): 483 documents, 775 chunks, 759 embedded (K-102 still live; `ws:pwq-test`
  482/773/757). **4 documents already mention `python-web-quirks`**; one of them
  (`38cd49cddb004207a5c7abd10181a4eb`) states the content of section 34 (BackgroundTasks serial
  execution). After migration it competes with the migrated section for the same query.
- Probe (read-only, prefixed per `skills/agent-kb-retrieval/SKILL.md`): a near-domain pytest query
  with no applicable gotcha ("session-scoped fixture ... xdist ...") returned hits at **0.4136 and
  0.4272, inside the 0.43 floor**, from unrelated team documents about shared-store test noise.
  Three other near-domain probes (asyncio.gather cancel, FastAPI dependency overrides, async
  SQLAlchemy session) had closest scores 0.562, 0.469, 0.463. Four probes: wrong hits pass the floor; the rate is unsized.
- Prior art (`claude/docs/plans/agent-knowledge-base-strategy-ml.md`, Stage 8 / DEF-1): 45 pairs,
  4 misses, all in prose-only tags (15/19); code/identifier-heavy tags 11/11. Its 6 negatives were
  off-domain (closest false match 0.446, margin 0.016); that corpus had no Python/pytest content.

## 3. Golden set (Set A, dev)

Scoring unit: the `documentId` of the **unit containing the query's section** appears among the 5
returned chunks (any chunk of it) with `score <= 0.43`; a hit on a merged unit counts for each member
section, so the gate does not separate sections inside a unit (the agent reads the whole unit).
Section numbers below are the 1-based order of the `##` headings (line = heading line).
"Confusable" neighbours, used to diagnose a miss, not to excuse one. Inside one unit (a hit
anyway): {2,34} {21,22,23} {24,25} {29,39} {30,40}. Across units: {5,6} {7,8} {11,35} {15,19,20}
{u21,u24} {26,27} {32,33} {36,38,42}.

**Set A shape (checked by the tool, not by hand):** 42 positives, exactly one per section 1-42, and
14 negatives, 7 `near` and 7 `off`, unique ids. Every scoring mode of Set A (not `--before`) refuses a
set of another shape, because Set A's coverage and false-accept counts decide the floor (§6) and a
partial set would report coverage "ok" (`validate_set_b` is the model). The tool checks this (U15).

### 3.1 Positives (42, one per section)

Written as the situation an agent is in, not the title. Identifiers an agent would really type
(`input()`, `TestClient`, `HEAD`) are allowed; title wording is not.

| id | section (line) | gotcha | situation query |
|---|---|---|---|
| A01 | 1 (L33) | asyncio.create_task fire-and-forget: GC risk is real per the d… | A service starts background work with asyncio and never keeps the object that comes back; a reviewer says the work may vanish mid-run. Should I call that a confirmed bug? |
| A02 | 2 (L55) | Starlette/FastAPI BackgroundTasks (sync callable) is bounded; … | Our API handler kicks off a synchronous job per request. Someone proposes a raw thread per call for throughput. Is the built-in route already capped, and how would I raise the cap? |
| A03 | 3 (L103) | FastAPI response_model_exclude_unset=True drops defaulted fiel… | An endpoint returns nested Pydantic models with exclude-unset switched on, and a client reports that inner keys which have defaults are missing from the JSON. |
| A04 | 4 (L133) | Asserting over a FastAPI app's route table: four built-in doc … | A test counts the routes on a freshly built FastAPI app to prove only my endpoints exist, but the count is higher than expected, and the routes of a sub-router I mounted seem absent from the table. |
| A05 | 5 (L177) | FastAPI's APIRoute does not add HEAD to a GET route — Starlett… | A HEAD request to my FastAPI GET endpoint gets a 405 when another route matches the same path, while the identical setup on plain Starlette answers it. |
| A06 | 6 (L206) | Starlette's StaticFiles.get_response returns its 404.html by d… | My single-page-app static mount overrides error handling to return index.html on a missing file, but a missing path still gets the built-in 404 page when the directory ships one. |
| A07 | 7 (L224) | FastAPI's responses={...} is keyed by status code only — two e… | I listed two different 422 payloads for one route in the OpenAPI responses mapping, but only one of them appears in the generated schema. |
| A08 | 8 (L250) | FastAPI/Starlette's exception-handler registry: MRO lookup, a … | I registered handlers for a base exception and for its subclass; my error-body test passes although the custom handler never ran, and a later registration silently replaced an earlier one. |
| A09 | 9 (L293) | uvicorn rewrites scope["client"] from X-Forwarded-For by defau… | A middleware allows admin endpoints only when the caller address is loopback; behind a reverse proxy any client can look local. Why does that check not hold? |
| A10 | 10 (L328) | pydantic Field(min_length=1) does not reject whitespace-only s… | My request model requires a name of minimum length one, yet a name made only of spaces was accepted and stored. |
| A11 | 11 (L340) | urllib failure taxonomy: HTTPError ⊂ URLError, but a read time… | I wrapped a urlopen call with a timeout in an except for the URL error class and a slow server still crashed my code with an uncaught error. |
| A12 | 12 (L384) | json.loads parses bare NaN/Infinity/-Infinity by default — a n… | A numeric validator parses user JSON and my finiteness check never fires: a bare NaN token got through and broke later comparisons. |
| A13 | 13 (L396) | An OpenAI-compatible local server can answer a missing /v1 pre… | My client pointed at the local model server without the version path segment received a success status, but the body was an error message, and my retry logic counted it as good. |
| A14 | 14 (L414) | An LLM-judge JSON parser that's a bare json.loads(text) is fen… | Code that parses a model's JSON answer began failing on every reply after switching model families; the replies look valid when printed but are wrapped in triple-backtick blocks. |
| A15 | 15 (L433) | Setting an env var *after* import is a no-op for a module-leve… | My test sets an environment variable with monkeypatch to change a config constant, but the code still sees the old value, and only when the whole suite runs, not when the file runs alone. |
| A16 | 16 (L459) | A pytest.raises(SomeError) with no match= can pass on an unrel… | A test expecting a validation error with a given exception type passes, yet when I break the very check it targets the test still passes. |
| A17 | 17 (L478) | A test that is only valid when its whole module ran must gate … | A completeness test over results recorded by sibling tests breaks on a subset run; I skipped it on the keyword filter but a run selecting by node id still fails it. |
| A18 | 18 (L512) | Deleting a parametrized test's def body leaves its @pytest.mar… | After a scripted edit removed a test function, a neighbouring test suddenly runs with odd parameters and fails with argument errors. |
| A19 | 19 (L529) | A function-LOCAL from .module import name re-resolves fresh on… | Patching a name in another module with monkeypatch works for one function that imports it inside its body but has no effect on another function where the same name is a default argument value. |
| A20 | 20 (L546) | A one-way circular import fails in *every* load order, not jus… | Adding an import of module B's name into module A, where B already imports from A, fails in both import orders even though the names are unrelated; moving it to class level did not help. |
| A21 | 21 (L568) | starlette.testclient.TestClient's teardown cancels every still… | My lifespan starts a polling task and never cancels it at shutdown, yet a test using the test client says the task was cancelled once the context-manager block closes, hiding the missing cancellation. |
| A22 | 22 (L585) | TestClient re-raises an unhandled server exception into the ca… | To check what an actual HTTP caller would get on an unhandled server error I asserted a 500 status, but the test itself dies with the original exception. |
| A23 | 23 (L624) | Version rot ahead of the two sections above: starlette depreca… | Importing the Starlette test client prints a deprecation warning telling me to install a different HTTP package, and our CI treats warnings as errors. |
| A24 | 24 (L641) | Anything the lifespan puts on app.state does not exist until t… | I patch an object stored on the application state in a fixture before any request and get an AttributeError saying the state has no such attribute. |
| A25 | 25 (L662) | To force a TestClient response to surface an uncaught exceptio… | A test already holds an entered test client with patches applied through it and I need the 500 behaviour for one assertion; building a second client wrapping the same application object lost my patches. |
| A26 | 26 (L681) | Holding an application lock across ThreadPoolExecutor.submit()… | A worker pool's submit is called while holding our own lock and I worry the interpreter hangs forever at exit; how long does exit actually block? |
| A27 | 27 (L759) | A flag cleared right *after* ThreadPoolExecutor.submit() retur… | A per-caller busy flag is reset right after handing work to a thread pool, and intermittently the end state is wrong because the worker already finished. |
| A28 | 28 (L775) | Bounding a call whose deadline the code under test computes: a… | I want to prove a function finishes inside a deadline it computes itself; my test timed the call and asserted afterwards, but when the function hangs CI just kills the job naming no test. |
| A29 | 29 (L803) | A same-second edit that keeps the file's byte length is invisi… | I changed a one-character constant in a module and re-ran immediately; Python still prints the old value although the file clearly changed, and the don't-write-bytecode variable did not fix it. |
| A30 | 30 (L835) | Under pytest's default capture, an in-test input() call raises… | Code under test calls input() to ask for confirmation; in the test run it throws an OSError about reading from stdin instead of waiting. |
| A31 | 31 (L850) | A typing.Protocol's own declared methods are derivable from va… | I want a test asserting that my fake implements every method a Protocol declares, without keeping a separate hard-coded list of names. |
| A32 | 32 (L865) | ruff check <path> force-parses the given file as Python regard… | I ran the Python linter over a list of changed files that included a markdown changelog and got a flood of syntax errors on ordinary sentences. |
| A33 | 33 (L877) | ruff format <path> reformats the WHOLE file, not just the line… | I ran the code formatter on one file to clear two long-line warnings and my diff now touches hundreds of unrelated lines. |
| A34 | 34 (L889) | Starlette BackgroundTasks runs one request's tasks sequentiall… | Several slow callbacks queued after one response are delayed by the first slow one, even though the thread pool has idle workers. |
| A35 | 35 (L901) | http.client.InvalidURL is an HTTPException, not a ValueError —… | My except ladder for ValueError around a call that opens a URL with a bad port did not catch the failure; it surfaced as an unclassified exception type. |
| A36 | 36 (L910) | pytest --deselect with an absolute path matches nothing, silen… | I told pytest to skip one specific test by passing its absolute path to the deselect option, and it still ran, with no warning. |
| A37 | 37 (L922) | FastMCP validates tool arguments with pydantic — a bound viola… | An MCP tool parameter has a maximum length; a client sending too long a value gets a tool result flagged as an error, not a protocol error, and I cannot find the bound in the advertised schema. |
| A38 | 38 (L940) | pytest -q -q prints no "N failed, M passed" line on a failing … | My script reads the pass and fail counts from the last line of the doubly-quiet pytest output, and a failing run reports zero failures. |
| A39 | 39 (L949) | PYTHONPYCACHEPREFIX=<fresh empty dir> is a stale-.pyc remedy t… | To avoid stale compiled files I pointed the bytecode cache location at an empty temp folder and it worked, but later ordinary runs still show stale code. What did that fix leave behind? |
| A40 | 40 (L966) | Under pytest's default capture, fd 0 is /dev/null, so an in-pr… | A test spawns a subprocess with no stdin argument and prints where its standard input points; it shows the null device, not the pipe my shell gave me, so my stdin-inheritance test proves nothing. |
| A41 | 41 (L977) | Popen.wait() after os.killpg(pgid, SIGTERM) leaves a TERM-igno… | After sending a termination signal to a process group and waiting on the leader, a child that ignores the signal survives and my file lock stays held, blocking the next acquirer. |
| A42 | 42 (L990) | A private copy for mutation testing: pytest imports the editab… | My mutation-testing copy of the repo is ignored when tests are executed via the bare pytest command, which imports the original tree, though python -m pytest picks up the copy. |

Caveat on Set A: I wrote it from each section's opening paragraph, so it is **head-biased**. That is
why Set B must carry the tail probes (§5). Set A is a dev set, not an estimate of recall.

### 3.2 Negatives (14): no gotcha applies

Near-domain (python/web/test topics the 42 sections do not cover; I checked all 42 titles):

| id | situation |
|---|---|
| NA01 | One coroutine in an asyncio.gather call raised, but its siblings keep running in the background. How do I make them cancel when the first one fails? (probed: closest 0.562) |
| NA02 | In tests I need to replace the authentication dependency of a FastAPI route with a fake user, and the override seems ignored for a sub-dependency. (probed: closest 0.469) |
| NA03 | An async SQLAlchemy session is not closed on an error path and under load the connection pool runs dry with timeouts. (probed: closest 0.463) |
| NA04 | A session-scoped pytest fixture seems to run once per worker under xdist, and my shared temporary database is populated several times. (probed: **0.4136**, inside the floor) |
| NA05 | uvicorn --reload does not pick up edits when the source is a bind-mounted Docker volume. |
| NA06 | A pydantic v2 model should accept one field name on input but emit another on output; which setting does that? |
| NA07 | mypy reports an Optional as still possibly None after I narrowed it inside a walrus condition. |

Off-domain (carry-over from Stage 8 plus new; plausible team-agent questions that no KB document answers):

| id | situation |
|---|---|
| NA08 | How should I structure a React Server Components data-fetching boundary to avoid request waterfalls in a Next.js app? (Stage 8 N4, closest then 0.446) |
| NA09 | How do I configure a Kubernetes readiness probe for a FastAPI container whose startup depends on a slow database connection? (Stage 8 N6, closest then 0.463) |
| NA10 | How can I make a multi-stage Dockerfile reuse the dependency layer when only application code changed? |
| NA11 | A git rebase stops with conflicts in a lockfile; what is the safe way to resolve it and continue? |
| NA12 | A CSS grid layout overflows its container on narrow phones even though the columns use fr units. |
| NA13 | A pandas groupby-apply over a few million rows is slow; what is the idiomatic faster form? |
| NA14 | A Postgres table's index keeps growing after heavy updates; how do I reclaim the space without downtime? |

NA01-NA04 were probed live; the other ten were not, so their closest score is unknown until the run.

## 4. Targets and sizing

### 4.1 Positives

Set B has 84 positive queries (2 per section, independent phrasings). The 84 are clustered in 42
sections, so they are not 84 independent draws: the per-query Wilson interval below is optimistic,
and the section-coverage rule (every section hit by at least one of its two queries; 42/42 has Wilson
lower bound 0.916) is the conservative companion criterion. Wilson 95% intervals, computed this run:

| observed on 84 | recall | Wilson 95% |
|---|---|---|
| 84 | 1.000 | 0.956 - 1.000 |
| 80 | 0.952 | 0.884 - 0.981 |
| 78 | 0.929 | 0.853 - 0.967 |
| 76 (gate) | 0.905 | 0.823 - 0.951 |
| 72 | 0.857 | 0.767 - 0.916 |

Probability of passing the gate (>= 76/84) by true recall, exact binomial, computed this run:
0.97 -> 0.999; 0.95 -> 0.975; 0.92 -> 0.772; 0.90 -> 0.534; 0.85 -> 0.100; 0.80 -> 0.008.
So the gate separates a system near 0.95 (passes) from one near 0.85 (fails) and does **not**
separate 0.90 from 0.92; that is the resolution 84 queries buy, and 150 queries would not
sharpen a threshold that is itself a judgment.

**Why 0.90.** The pooled prior-art result is 41 of 45 (4 misses; 0.911 by my arithmetic), with
code/identifier-heavy rows 11/11. This content is code-dense (the rows resembling the 11/11 stratum)
but has dense intra-domain neighbourhoods (5 `TestClient`/lifespan sections, 2 stdin-capture and
2 `.pyc` sections, thread-pool and `BackgroundTasks` pairs). Expected true recall is therefore
>= 0.95 (a prior); 0.90 is the floor below which the migration is worse than the
team's accepted KB retrieval and the user story ("get the one applicable gotcha") is not credibly met.
A miss is not silent (FR-11 title listing and FR-9 section reads are the fallback), so no 0.95 gate.

### 4.2 Negatives

Set B has 30 negatives (15 off-domain, 15 near-domain). 0/30 false accepts has Wilson upper bound
0.114 (2/30 is 0.018-0.213). P(<= 3 false accepts of 30) by true
false-accept rate, computed this run: 0.02 -> 0.997; 0.05 -> 0.939; 0.10 -> 0.647; 0.20 -> 0.123;
0.30 -> 0.009. Gate: **off-domain 15/15** (the prior art's 6/6 plus a real margin; a wrong gotcha for
an unrelated question is the failure mode FR-9 warns about), **near-domain >= 12/15**. Near-domain is
softer: a false accept costs one wrong ~1.4 KB section read, and the xdist probe (0.4136) shows the
shared floor cannot reject them all.

**Scoring definition (decision for the stakeholder, my default stated):** AC-3 says negatives
"return nothing above the score floor". Workspace-wide that is not a property of this feature: the
xdist probe returned unrelated team documents under the floor. I score a false accept as **a hit
under the floor whose `documentId` is one of the 36 unit documents**, and report workspace-wide hits as
information. Needs the stakeholder to confirm, or to reword AC-3.

**Raw captures that restate a gotcha (rule `pwq_kb.py eval` implements).** A raw capture is not a
unit, so (a) a hit on one **never counts as a positive hit** (the expected id is the unit's), and (b) a
hit on one is **never a false accept** for a negative (false accepts count unit documents only). What
a capture can do is displace a unit's chunks from the five slots, which is a measurement artifact, so
captures are removed from the scored state: by deletion at H1 for the captures listed at S13b, and by
id exclusion (`--exclude`, a 50-chunk window, drop the excluded documents' chunks, truncate to five)
for any capture that appeared after that listing. The headline is the excluded run; the unfiltered
numbers are printed beside it (`--sweep` must print them too, an implementation requirement), and the
gap is the pre-existing-hit finding. In `ws:pwq-test` the S12 sweep runs with the S7 list excluded; that
equals the post-H1 state only for ids actually deleted at H1, so S14 passes `--exclude` only ids the
ledger records as deleted at H1 or created after S13b. Most pre-existing hits are not captures: G2
measured 26 of 42 Set A positives with a pre-ingestion hit under the floor, only 9 on the S7 list; the
17 other documents are unlisted team documents H1 never touches. They cannot inflate recall (only the
unit's own document counts); the risk is deflation, a non-unit chunk displacing a unit chunk from the
five. S12 reports, per Set A positive, the rank of the first unit chunk and whether any non-unit chunk
scores better, aggregated with documents named; S14 reports the aggregate only (B stays sealed).
Ruling: reporting, no threshold; the effect is one-sided (conservative), the documents cannot be
removed, and a threshold would be tuned on the sealed set. The existing rule (`scored_chunks`, `evaluate`, the T29 filtered/unfiltered
printout) already matches this ruling in every clause; no code change. Limit: the exclusion window
is 50 chunks, so it fails only if over 45 of a query's nearest chunks belong to excluded documents,
which the 17 known documents (23 chunks, `docs/plans/python-web-quirks-kb-captures.md`) cannot do.

### 4.3 Before and after (FR-12)

- **Before** is not "whole skill loaded" (trivially 100% recall at ~76 KB). It is (i) the same sets
  run **before ingestion**: expected near-zero hits on gotcha documents, and any query whose answer
  already comes back from another ws document is a pre-existing-hit finding (captures are a subset);
  and (ii) a **lexical baseline** (BM25 or ripgrep scoring over the 42 sections, offline) on Set B.
  If semantic retrieval does not beat the lexical baseline on recall at equal top-5, the GraphRAG
  layer is not earning its cost for this content; report it either way.
- **After**: Set B once, plus the two checks on Set A below.
- Report both **raw top-5 recall** (no floor) and **floor-applied recall**; the gap is the floor's
  cost on positives. Also report MRR and rank of first expected-chunk (informational, no gate).

## 5. Authoring without leakage, and hold-out

**Roles.** Set A: `data-scientist` (this note). Set B: **`qa-engineer`** (decided by me as a routine
matter; any author who is neither the migration implementer, nor `cobb`, nor me works). The author of B
works from the section text but writes only the situation an agent would be in.

**Authoring rules (apply to A and B):**

1. Write the situation: the symptom, the surprising observation, or the decision being made, as an
   agent would phrase it without knowing the gotcha exists. Never write the claim, the cause, or the
   title.
2. Public identifiers (library, function, flag names) may appear; the title's non-identifier wording
   and the section's conclusion may not.
3. **Mechanical leak check before freezing**: lowercase word tokens; no 4-gram shared between a
   query and its own section (title or body). Observed on Set A: after rewording 6 queries that
   shared a 4-gram, **0 of 42** share a 4-gram and 0 share a 5-gram with their own section
   (script run this run; the first draft had 5 of 42 sharing a 4-gram).
4. Set B only: each section gets two queries from different framings (one symptom/error, one
   design-time "should I"), and for each of the 12 units with >= 4 chunks (u02 {2,34}, u04, u08,
   u09, u11, u15, u21 {21-23}, u24 {24,25}, u26, u28, u29 {29,39}, u30 {30,40}) **at least one
   positive of the unit's sections must target a fact that sits only in the unit's last chunk** (the
   tail probe, flagged `tail` in the set; `units --chunks` gives each unit's chunk count).
5. Negatives in B: 15 off-domain and 15 near-domain, near-domain ones written to *sound* like a
   section (same library, adjacent behavior) while no section answers them; the author checks that
   against the title list (FR-11), not the bodies.
6. Freeze: B is stored (path to be set by `teco`, not in this repo's docs tree, since the note and
   brief are read by implementers) and its sha256 recorded before the migration implementer starts
   chunk/title decisions. The sealed scoring takes that recorded value as a **required**
   `--expect-sha256` argument and refuses before the sentinel, the gate or any search when the file's
   hash differs (a B edited after freezing otherwise scores silently, and the sentinel's own
   `setSha256` is written after the fact). `--probe`, which runs before the freeze, does not take it. A is committed with this note.

**Hold-out discipline.** A may be run repeatedly and failures inspected; it is the only set used
for any tuning (chunk-header decisions, document title format, any floor sweep). B is run **once**
for the acceptance verdict. If B fails and a fix is made, B is burnt as evidence: write B2 (ordinal
slug) from a different author and gate on it; never re-run B after looking at its misses.
The prefix and top-K stay the repo's literals (FR-7); they are not tunable here. The Stage 8
45-pair set is a regression guard only if a floor or top-K change is proposed (its misses would hit
other KBs).

**When the `--set B --probe` may rewrite a query (ruling).** The probe runs before ingestion, so its
rank-1 document is never a unit; it detects a query that is already near-verbatim in an existing
document. Rewrite a B query **only** if its rank-1 score is **< 0.25** (`LEAK_SCORE`), once, by B's
author, then re-run the 4-gram check and freeze. A query that already surfaces from another document
(rank-1 under the 0.43 floor, e.g. a raw capture) is **kept** and recorded as a pre-existing-hit
finding: selecting queries by how the corpus answers them is tuning on B, and dropping those
queries would inflate recall (captures are neutralized by deletion/exclusion, §4.2, not by editing B). `LEAK_SCORE = 0.25`
is a tripwire, not a calibrated threshold: six Set A probes before ingestion had rank-1 scores
0.3657 to 0.5309, so it sits 0.115 below the lowest and fires only on a near-copy; erring toward not
rewriting is the safe direction for hold-out. It does not detect a title paraphrase (the 4-gram check's
job, rule 3; residual limitation in §8). Do not move it on six points.

**Contamination channels to close:** (i) the existing raw captures that mention the skill (§2),
including one duplicating section 34 (probed this run: three of its chunks returned at 0.4606,
0.4766 and 0.4882 for A34's query, so a restating capture can sit on either side of the floor),
decided by `cobb` under the FR-10 path (retire or keep) before the B run; the scoring rule in §4.2
neutralizes any that remain; (ii) no B query or result
is pasted into any gotcha text, the title listing or the stub skill's description; (iii) the stub
skill description (FR-6) is routing text and must not carry situation phrases lifted from A.

## 6. Is the 0.43 floor adequate? (d)

**Not established for this content.** The floor was calibrated on a corpus with no Python/pytest
content; its closest negative was 0.446, 0.016 above the floor, and the worst accepted true
positive 0.414 (`skills/agent-kb-retrieval/SKILL.md`). Adding 42 dense, same-register, near-domain
documents moves nearby negatives closer (one near-domain probe is already under 0.43 today, §2) and
can move true positives either way. The floor is a corpus-level property, so I will not guess.

**Measurement (run on Set A only; B is not used for choosing):** in a test workspace holding the
full current corpus plus the 42 migrated documents, record per query the best score of the expected
document (positives) and the best score of any gotcha document (negatives), then:

1. Sweep the floor from 0.38 to 0.50 in 0.01 steps; per step: floor-applied recall, near-domain false
   accepts, off-domain false accepts.
2. **Adequate (keep 0.43)** iff, on A: floor-applied recall is within 0.03 of raw top-5 recall,
   AND no off-domain negative has a gotcha hit <= 0.43, AND near-domain false accepts <= 2 of 7, AND
   the worst found-positive score is >= 0.025 below 0.43 (the team's own floor-instability margin rule;
   a row inside it needs multi-session reproduction before it is trusted).
3. Otherwise report the sweep, name the value that satisfies the same criteria, and route a change
   to `agent-kb-retrieval` (FR-7). A change moves every user, so the Stage 8 45-pair set must be
   re-run first and must not regress; a floor that works only for this content is **not**
   adopted privately here. If no value satisfies both, the finding is "score alone cannot separate
   these", and the fallback is a stricter rule scoped to this feature's stub (not a floor edit).
4. Then confirm on B once; B's verdict uses whichever floor step 2/3 set.

A is small for a sweep (42 + 14): it locates a plausible band; the B result carries the evidence.

## 7. Chunking guidance (e)

**Sizes versus window.** The embedder window is about 2,048 tokens (K-100/K-099 in
`falkor-chat/docs/BACKLOG.md`: about 9.7k characters of prose, about 5.7k of URL-dense text).
Ingesting each of the 36 units as one document, with the shipped chunker, gives 129 chunks of at most
1,142 characters, far inside the window. **No unit is embedded head-only** and K-100, which is about
messages, is not a dependency. The "check chunk behaviour on the largest sections first" decision
(requirements decision log) reduces to a count check plus a tail probe, not a truncation fix.

**Where head-only would bite, if someone embedded whole units:** only u21 (6,554 characters) and u26
(6,393) exceed the 5.7k figure; seven more (u02, u04, u08, u09, u11, u24, u29) sit between 2.6k and 5.1k.
Tokens were not measured; before approving any whole-unit embedding, measure them for those 9.

**Risks that do exist, in order of expected effect:**

1. **Tail chunks carry no identity.** 34 of 36 units span >= 2 chunks; only chunk 0 holds the title and
   header (the 150-character overlap is the only bridge). A tail hit is a body fragment without title or
   version pin, so FR-3 ("usable without a second lookup") and FR-5 fail for it; the agent needs
   `get_document`. Measure: Set B tail-probe recall (the 12 units with >= 4 chunks, §5 rule 4) versus head-probe recall; if tail < head by
   more than 0.15, the fix is a design decision for `architect` (a per-chunk title/version header
   line), not a chunker change here (out of scope per the requirements).
2. **Slot crowding.** `search_documents` has no document dedupe; units with 4-10 chunks can fill
   several of the five slots, leaving fewer than five distinct gotchas. Score recall at the
   `documentId` level over the five chunks, and report the mean count of distinct documents per
   query.
3. **Per-unit dilution.** The prior-art DEF-1 mechanism (a dense identifier-heavy worked example
   diluting the embedding) applies most to the 10-chunk u26, the 9-chunk u21 and the other multi-section units; any recall
   miss there is read against that.
4. **K-102.** 16 chunks in `ws:agent-team` have no embedding today (observed at plan time). The per-gotcha
   "every chunk embedded" check (AC-2) is a hard gate before any scoring run; otherwise a
   non-embedded chunk reads as a retrieval miss.

## 8. Risks and open questions for the stakeholder

- **Approval needed:** targets in §1 (recall >= 0.90 + 42/42 coverage; negatives off-domain 15/15,
  near-domain >= 12/15), and Set A. Confidence in them is stated in §4: they discriminate ~0.95 from
  ~0.85 true recall, not finer.
- **AC-3 wording:** confirm that negatives are scored against the 36 unit documents only (§4.2),
  or accept that workspace-wide rejection is not measurable by this feature.
- **Raw captures (§4.2, §5):** retire the captures that mention the skill, or accept that the
  restatement of section 34 can outrank its own migrated section (the scoring rule excludes it either way).
- **Set B author and storage:** `qa-engineer`, stored outside the implementers' reading path; `teco`
  sets the location.
- **Estimates, not measurements:** the expected true recall (>= 0.95) is a prior; token counts per
  section (§7) are unmeasured; none of Set A's positives has been run against an ingested corpus.
- **Dedicated limitation:** the leak check is lexical (n-grams); it does not detect a query that
  paraphrases the title closely in meaning. The before-ingestion run (§4.3) is the semantic check:
  a query whose answer already surfaces from another document is kept and recorded (pre-existing hit); only one whose
  rank-1 score is < 0.25 is re-written (§5 ruling). A title paraphrase is caught only by the post-ingestion
  Set A run, never on B.
- **Code state:** the Set A validator, the required `--expect-sha256` on `--sealed-once` and the Set B tail-flag check are delivered (U15); `--sweep` unfiltered printing is owed (§4.2).
