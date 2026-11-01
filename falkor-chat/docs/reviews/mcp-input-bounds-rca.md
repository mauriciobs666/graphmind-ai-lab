# MCP input bounds: REST-only validation (RCA)

> **Status:** active · **Owner:** `analyst` · **Tracks:** K-101

**Scope.** Every tool in `server/falkorchat/mcp.py` (25 registered tools), each argument mapped to its
REST counterpart in `api.py`/`schemas.py`, and the service method both call (`services.py`). The
baseline is `HEAD` = `6d4f5c05`, which includes K-099 item 3's `MessageTooLongError`. Out of scope:
REST-only routes with no MCP twin (workflow defs/runs, `GET /messages/{mid}`), the storefront
(`/shop/api`, which has its own tighter models), and agent-tool callers (`tools.py`). Each is noted
only where it affects the fix.

**CPG:** considered, not relevant. The question is about field constraints in three files read
directly (`schemas.py`, `mcp.py`, `services.py`), and runtime behavior verified by execution.
Call-graph structure adds nothing here, and `cpg_falkorchat` is stale for exactly `mcp.py`/`services.py`.

**Verdict.** The premise is **confirmed, with a correction.** REST enforces 69 field constraints
(enumerated from the live OpenAPI). 25 tools reach the same service methods via MCP, and MCP
declares **zero** constraints. The service backs up only 4 of the bounds that MCP-mapped arguments
carry. The correction is that MCP **does** have a pydantic layer. FastMCP builds a pydantic argument
model from each tool signature and type-checks every call. It honors `Annotated[..., Field(...)]`
constraints and publishes them in the tool's JSON schema (verified, `mcp` 1.28.1, §2 E2). SERVER.md
§2.2's "No schema layer on MCP. Tool arguments reach the service without Pydantic" is false. That
false premise is what kept this class being patched one bound at a time (§3).

The class is **bounds only**. I found no REST-only *behavior* on MCP-mapped routes, so no scope
question blocks the fix. (REST's 404-on-`None` vs MCP's `null` is a documented, deliberate
difference, SERVER.md §2.2.)

## 1. Symptom & impact

- MCP `send_message` posts an empty body. REST answers 422. (K-101)
- MCP `send_message` accepts more than 50 mentions. REST answers 422. (K-101)
- MCP `ingest_document` stores a title over 200 characters, and a REST re-ingest of the same
  document is then refused. (K-074, open since 2026-10-17.) **Live evidence:** 7 of the 433
  documents in `ws:agent-team` have titles over 200 characters (max 243, read-only query, §2 E4).
  The tool schema an agent sees advertises no cap, so the agent cannot know about one.
- The same class covers 22 more tools: names and titles, document metadata, `limit`, `since`, `query` and id arguments (Appendix A).

**Impact, graded by downstream cost (§3.3).** No gap crashes anything. No MCP argument feeds a
`UNIQUE`-constrained write, so the K-049 SIGSEGV is unreachable (traced, §3.3). The costs:

- **Stored junk:** a blank message, nameless channels and threads.
- **Unbounded RAM:** channel names and thread titles have no cap at any layer below REST.
- **Wasted LLM/embedder work** on an empty message.
- **Misleading errors:** a negative `limit` on `search_messages` surfaces as "invalid query syntax".
- **Front-door disagreement:** the same input succeeds on one door and fails on the other.

## 2. Reproduction & evidence

All runs were offline or read-only. Nothing was written to any graph, and LM Studio was not touched.

- **E1 Service layer** (scratch script `svc_probe.py`, `Services(FakeRepo())` imported from
  `tests/test_services.py`):

  | Status | Inputs |
  |---|---|
  | Accepted | `post_message(text="")`<br>`post_message(text="   ")`<br>51 distinct known mentions<br>`create_channel(name="")` / `"x"*201` / `"x"*10**6`<br>`create_thread(title="")` / `"x"*201`<br>`ingest_document` with `title`, `source_label`, `source_format=""`, or `source_format="x"*51` over cap<br>`ingest_documents([])` → `[]`<br>`read_messages(since=-5, limit=-1)`<br>`search_messages(query="")` |
  | Rejected (controls) | 8001-char message → `MessageTooLongError`<br>`ingest_document(text="")` → `EmptyDocumentError`<br>201-char unknown `produced_by` → `AgentNotFoundError` |

  `list_*` methods are thin passthroughs with no check (read, `services.py:874`, `:1462`, `:1484`,
  `:1491`, `:1577`, `:1583`).
- **E2 FastMCP validation** (scratch `fastmcp_probe.py`, the repo's own `.venv`, `mcp` 1.28.1,
  pydantic 2.13.4). A plain-typed tool accepts `""`, 51 mentions and `limit=-1`, and rejects
  `limit="abc"` and `body=5` with a pydantic `ToolError`. That is the type layer working. The same
  tool with `Annotated[str, Field(min_length=1, max_length=…)]`,
  `Annotated[list[str] | None, Field(max_length=…)]` and `Annotated[int, Field(ge=1, le=200)]`
  rejects all three. Its `inputSchema` carries `minLength`, `maxLength`, `maxItems`, `minimum` and
  `maximum`.
- **E3 Parity enumeration** (scratch `parity_probe.py`). `FastAPI().include_router(
  api.build_router(Services(FakeRepo()))).openapi()` yields 69 constrained REST fields. The live
  `mcp.mcp.list_tools()` returns 25 tools with **0** constrained arguments.
- **E4 Live graph, read-only** (`mcp__cypher__query`, `RO_QUERY`). `ws:agent-team` has 433
  `Document`s, 7 of them with `size(title) > 200`. `ws:acme` has 0 empty messages, channel names or
  thread titles.
- **E5 Engine `LIMIT`** (`RO_QUERY` on `cpg_falkorchat`). `LIMIT $lim` with `lim=-1` gives
  `Limit operates only on non-negative integers` (a `ResponseError`). `lim=0` gives 0 rows.
  `db.idx.fulltext.queryNodes('Message', '')` gives 0 rows (on `ws:acme`).
- **E6** `pytest -q tests/test_services.py`: **288 passed**. This is the review-safe subset only.
  `test_api.py`, `test_mcp.py`, `test_storefront_api.py` and the default suite were **not run**,
  so `reference` was untouched and needs no re-seed.

## 3. Causal chain

**3.1 Why MCP passes `""` through.** Each `mcp.py` tool forwards its arguments verbatim to a
`Services` method (`mcp.py:200-528`). The signatures carry types only (`body: str`,
`mentions: list[str] | None`, `limit: int`), so FastMCP's argument model checks types and nothing
else (E2). `Services.post_message` checks only the upper text bound (`services.py:983`).
`_validate_and_derive_role` dedups mentions and resolves them but never counts them (`:914-925`).

**3.2 Why the class exists (the design decision).** Commit `ccd6f2b9` (K-006, 2026-07-04)
introduced every message, name and title bound and every `limit` bound. It added them
**exclusively** as pydantic `Field`/`Query` constraints. The same commit added two MCP tools
(`list_channels`, `list_threads`) with none. Its spec text, still in SERVER.md §1.4 "Bounds", is
"Request bodies are size-bounded at the Pydantic boundary … so the transport caps it". That files
size bounds as a **transport concern**. SERVER.md §1.2 and §2.1 say "`services.py` owns the
invariants" and "all invariants here; both front doors call the SAME methods". So one transport got
the caps, and the other was specified as a thin adapter that must not hold logic. Neither document
says which layer owns a bound that protects stored data.

**3.3 What each gap costs** (traced unless marked *inferred*):

| Gap (MCP) | Downstream cost |
|---|---|
| Empty message `text` | A permanent blank `Message` in the thread (§4 write). `_safe_embed` sends `""` to the embedder (`background.py:46-57`, `embedding.py:75-79`). *Inferred:* either a provider error, logged and swallowed, leaves the message out of the vector corpus, or a degenerate vector becomes an ANN neighbour. With an @mention, the trigger or responder runs an LLM turn on empty input (`trigger.py:81`, `responder.py:72`). **Data quality plus wasted model work.** |
| Mentions over 50 | Small. Unknown ids are refused (`UnknownMemberError`) and duplicates are deduped (`_dedup`, E1: 60 copies of one id become 1). So stored `MENTIONS_MEMBER` edges are bounded by the workspace's member count. An oversized raw list costs one `UNWIND` lookup before rejection. **Mostly an inconsistency.** |
| Channel `name` / thread `title` (empty or oversized) | Stored as plain, unindexed, unconstrained properties (`bootstrap_schema.sh:123-126`, `:296-300`), with no cap below REST and no HTTP body cap on uvicorn. A 10⁶-char name is accepted (E1) and is returned in every `list_channels`/`list_threads` row. **RAM (rule 6) plus a nameless UI.** |
| Document `title` / `source_format` / `source_label` | `title` and `sourceFormat` are stored. `source_label` is never stored: it is only the `title` fallback (`services.py:1250`). The cost is small RAM. The real cost is **disagreement between the two doors**: E4's 7 titles cannot be re-ingested over REST. |
| `produced_by` over 200 | None. It is used for lookup only, and an unmatched id raises `AgentNotFoundError` (E1). |
| `limit` (12 tools), `since` | A negative value reaches Cypher `LIMIT` and gives a raw `ResponseError` tool error (E5). `search_messages` misreports it as `InvalidSearchQueryError` (`services.py:1101-1104`). A huge `limit` gives an unbounded result. `search_documents` multiplies it into the ANN `k` (`:1436`). `limit=0` gives `[]`, which is harmless. |
| Empty `query` (2 tools) | `search_messages`: 0 rows (E5), harmless. `search_documents`: embeds `""`. *Inferred* cost is the same as the empty message. |
| Path-id lengths (9 tools) | None. These ids are lookup-only, and an unknown id gives `None` or a NotFound error. No MCP id feeds a `UNIQUE`-constrained write. The cursor id is written only after rows are returned, so only for a minted thread (`services.py:1065-1084`), and the `DocumentDeletion` id is copied from the matched node (`repository.py:1345-1352`). **The K-049 crash is unreachable.** |

**3.4 Why it recurred instead of being fixed once.** Each instance was patched where it was found,
in the service, and each patch's docstring repeats the false premise "an MCP caller has no schema
layer":

- `EmptyDocumentError`, `DocumentTooLargeError`, `BatchTooLargeError` (K-050)
- the merged-ctx bound (K-024)
- `_validate_key_lengths` (K-049)
- `MessageTooLongError` (K-099, which reopened the same message model and fixed one of its three
  bounds)

K-049's RCA §6 item 4 proposed a standing rule ("every … caller-supplied … field needs an app-side
bound enforced at the service layer"). It never landed: it is not in `falkor-chat/AGENTS.md`,
`DESIGN.md`, `SERVER.md` or `REVIEW.md`. No test compares the two front doors' bound sets, so a new
REST bound or a new MCP tool opens a new gap silently.

## 4. Root cause

**Root cause (confirmed by execution and history).** Input bounds were classified as transport
validation (SERVER.md §1.4, from `ccd6f2b9`) instead of service invariants. They were then declared
in only one transport's schema layer. Nothing checks that another front door declares or inherits
the same set.

**Trigger.** The `embedder-context` review's Pass 1 found two more instances while checking K-099.

**Contributing factors:**

1. **(a)** SERVER.md §2.2 states that MCP has no pydantic layer. This is false (E2), and it framed
   the gap as structural, so each fix became a one-off service check instead of a declared contract.
2. **(b)** No parity test exists between the two front doors.
3. **(c)** K-049's proposed rule was never adopted.
4. **(d)** The MCP tool schemas advertise no bounds, so a well-behaved agent cannot comply (E4).

## 5. Suggested fix & prevention

### 5.1 Options weighed

| Option | Assessment |
|---|---|
| **A: per-bound service checks** (the `MessageTooLongError` precedent) | Covers every caller (MCP, scripts, future tools). Matches §1.2 "services own the invariants" and six precedents. It is the only layer that keeps `ingest_documents`' per-item error isolation (SERVER.md §2.2): REST validates a batch all-or-nothing, but MCP returns one receipt per item, so a too-long item title must become that item's receipt, which only the service can do. It cannot be compared mechanically against REST, and it advertises nothing to the client. |
| **B: `Annotated[..., Field(...)]` on MCP signatures**, using the `schemas.py` constants | Verified to work (E2). It also **publishes** the bound in the JSON schema the agent's model reads, which prevents the error instead of only reporting it. It exposes constraints in the same JSON-Schema vocabulary as FastAPI's OpenAPI, so parity becomes a **mechanical comparison** (E3). It does not cover in-process callers. Applied to `ingest_documents.items` it would break per-item isolation. |
| **C: build `PostMessageIn(...)` etc. inside each tool** | Field names differ (`body`/`text`, `source_format`/`sourceFormat`), `limit` and `q` are `Query()` parameters with no model, and it advertises nothing. Strictly worse than B. **Rejected.** |

### 5.2 Recommendation: A for stored values, B for request hygiene, one parity probe for the class

1. **Service layer (A): every bound on a value that gets stored.** Add these to `services.py`,
   checked before any repository call:
   - `post_message`: `text` length ≥ 1, and `mentions` ≤ 50, counted on the **raw** list before
     `_dedup`. This matches REST, which counts what was sent.
   - `create_channel` `name`: 1–200.
   - `create_thread` `title`: 1–200.
   - `ingest_document`: `title` ≤ 200, `source_format` 1–50, `source_label` ≤ 200. Through
     `ingest_documents` a violation becomes that item's own error receipt, because the existing
     `except ServiceError` already isolates it.

   Raise one new `InputBoundError(ServiceError)` naming the field, the actual size and the limit. It
   maps to 400 through the generic handler (`SERVER.md` §1.4 "Error envelope"). Leave the existing
   per-bound classes alone. REST is unchanged, because pydantic answers 422 first. The storefront
   is unaffected: it sends nonblank text of at most 2000 characters and `mentions=[agent_id]`
   (`storefront_api.py:1240`). This also closes K-074, which should be folded into K-101.
2. **MCP signature (B): advertise the stored-value bounds, and own the read parameters.** Use the
   `schemas.py` constants.
   - Annotate `send_message.body`/`mentions`, `create_channel.name`, `create_thread.title` and the
     `ingest_document` fields. This duplicates A on purpose, so the agent sees the contract (E4).
     Do **not** type `ingest_documents.items` (see the table above).
   - Annotate every `limit` with its REST twin's range:
     - 1–200 for the list and search tools
     - `read_messages` 1–1000 in both modes (the room-wide read has no REST twin, so it takes the
       thread window's range)
   - Annotate `read_messages.since` ≥ 0, and `search_messages.query` / `search_documents.query`
     length ≥ 1.

   Read parameters stay out of the service on purpose. Internal callers pass their own limits
   (storefront ≤ 200, `tools.py` `hybrid_search`), a violation costs a clean error rather than
   corrupt data, and this mirrors REST's own `Query()` layer.
3. **Not copied** (each recorded as an explicit ruling in the probe's mapping, §5.3):
   - **Path-id lengths** (`document_id`/`match_id`, 9 tools). They are lookup-only, with no cost
     and no K-049 path (§3.3). REST bounds them inconsistently itself (`GET /documents/{id}` is
     unbounded), and copying would replicate that inconsistency.
   - **`produced_by` ≤ 200.** Lookup-only, and already refused as `AgentNotFoundError`.
   - **`ingest_documents` empty batch** (`min_length=1`). `[]` is the correct answer to an empty
     batch, and it writes nothing.
   - **Whitespace-only message text.** REST accepts it too, so rejecting it would *tighten* REST.
     See Open questions.
4. **Fix the spec premise** at the feature close (§6). The model-visible tool schemas change, which
   counts as behavior under `falkor-chat/REVIEW.md`. The change only tightens and advertises
   bounds, but say so in the impl review.

### 5.3 Test strategy (test-first; sequencing matters)

**Round 1: land the class guard before any fix.** It must not be authored in the same step it
gates. `tests/test_input_bound_parity.py` needs no database:

- Build REST OpenAPI from `api.build_router(Services(FakeRepo()))` and walk every constrained field
  (`minLength`, `maxLength`, `minItems`, `maxItems`, `minimum`, `maximum`, `pattern`, `enum`), as in
  E3.
- Enumerate `mcp.mcp.list_tools()` and every tool argument.
- Check both against one `MAPPING` table. Each REST constrained field and each MCP argument has
  exactly one entry: `mcp(tool, arg)`, `service(method, field)`, `not_copied(reason)`,
  `no_mcp_twin` (workflow routes), or, in Round 1 only, `known_gap`.

The test fails when:

- **(i)** a constrained REST field has no entry, as happens when someone adds a `Field(max_length=…)`;
- **(ii)** an MCP tool or argument has no entry, as happens when someone adds an `@mcp.tool()`;
- **(iii)** an `mcp` entry's `inputSchema` constraint differs from REST's;
- **(iv)** a `service` entry does not raise `InputBoundError` at REST's limit+1, or rejects the
  value at the limit. Call it through `Services(FakeRepo())`.

These are two independent declarations compared, not a constant checked against itself.

Round 1 is green with the current gaps listed as `known_gap`. Confirm it can go red with three
planted mutants: an unmapped `Field` on a scratch copy, a dummy `@mcp.tool()`, and a `known_gap`
flipped to `mcp` without the matching annotation.

**Blind spot to record in the file:** custom `field_validator`s (`_bounded_flat_dict`, the
storefront's `_nonblank`) are invisible to OpenAPI. None sits on an MCP-mapped model today, so the
mapping table should say so.

**Round 2 (`tdd-engineer`).** Flip each `known_gap` to its §5.2 layer. Each flip turns a probe leg
red, then the fix makes it green. Add absolute pins that use **literal** values (`""`, 201/200,
51/50, 0/1, -1), never `MAX_*`. Root `REVIEW.md` bans reading expected values from the constant
under test, and the existing K-099 test at `test_mcp.py:244` does exactly that.

- **`test_services.py`** (review-safe): parametrized over every service bound, covering "at limit
  accepted", "limit+1 refused" and "empty refused". Assert that no repository write was recorded.
  Also cover an `ingest_documents` batch where one item has a 201-char title: that item gets an
  `InputBoundError` receipt and its siblings get `processing`.
- **`test_mcp.py`**: a schema assertion per annotated argument through `list_tools()`, using
  literals. Add one `call_tool` rejection each for `limit=-1` and an empty `body`, with the Services
  configured on `FakeRepo` so no database is involved.

`test_mcp.py`'s existing `repo` fixture wipes only `ws:test` (`conftest.py:133`). Keep new tests off
it where possible.

## 6. SERVER.md sections the fix changes (updated at feature close, not by the fix)

- **§1.4 "Bounds"**: say which layer owns which bound (stored-value bounds in the service for every
  front door, read-parameter ranges on each transport). Replace "the transport caps it" and "the
  document and batch caps are also enforced in the service".
- **§1.4 "Error envelope"**: add `InputBoundError` (400).
- **§2.2 "No schema layer on MCP"**: rewrite it. MCP arguments are validated by FastMCP's pydantic
  model with the same bounds as REST, except the not-copied rulings. Remove the K-074 sentence.
- **§2.2 tools table**: optionally note that the bounds are advertised in each tool's `inputSchema`.

Also update the `mcp.py` tool docstrings and the docstrings that repeat "no schema layer"
(`services.py` `DocumentTooLargeError`, `MessageTooLongError`, `BatchTooLargeError`,
`EmptyDocumentError`, `_validate_key_lengths`; the `schemas.py` comments). These are not spec
sections, but they carry the same false premise.

## What's solid

- K-099's `MessageTooLongError` runs before any read or write, and its MCP test checks that nothing
  was posted or scheduled.
- `ingest_documents`' per-item isolation is sound, and the recommendation is designed around it.
- No MCP argument can reach the K-049 crash path.

## Open questions

- **Whitespace-only message text.** `EmptyDocumentError` already rejects blank documents, and the
  storefront rejects blank messages (`_nonblank`). Should `post_message` reject `"   "` too? That
  would also turn dev-REST's 201 into a 400. **Recommendation: yes, as the service's empty check
  (`text.strip()`), for consistency with documents**, but this is the stakeholder's call because it
  tightens REST. The §5.2 baseline is exact parity (`len ≥ 1`).
- **Advertising duplicated bounds (§5.2 item 2, first bullet).** Optional. Recommended because of
  E4, and the parity probe keeps the two declarations in sync.

### Proposed REVIEW.md lines

None. The class guard is the Round 1 parity probe, and root `REVIEW.md` says a rule a test enforces
is not added.

## Appendix A: full inventory (MCP argument → REST bound → who enforces today)

Key: **S** = service enforces · **M** = MCP enforces · **—** = neither. Unbounded = REST declares
no bound either (out of class).

| MCP tool(arg) | REST counterpart & bound | Today | §5.2 |
|---|---|---|---|
| `send_message(body)` | `PostMessageIn.text` 1–8000 | max S (K-099), min — | S + M |
| `send_message(mentions)` | `PostMessageIn.mentions` ≤ 50 items | — | S + M |
| `send_message(re)`, `(frm)` | path `thread_id` unbounded; `frm` ignored | n/a | — |
| `read_messages(since)` | `GET /threads/{tid}/messages` `since` ≥ 0 | — | M |
| `read_messages(limit)` | same route, 1–1000 (room-wide mode has no REST twin) | — | M |
| `read_messages(re)`, `(advance)` | unbounded / no twin | n/a | — |
| `create_thread(title)` | `CreateThreadIn.title` 1–200 | — | S + M |
| `create_thread(channel_id)`, `list_threads(channel_id)` | path unbounded | n/a | — |
| `create_channel(name)` | `CreateChannelIn.name` 1–200 | — | S + M |
| `list_channels(limit)`, `list_threads(limit)` | `Query` 1–200 | — | M |
| `search_messages(query)` | `GET /search` `q` ≥ 1 | — | M |
| `search_messages(limit)` | 1–200 | — | M |
| `ingest_document(text)` | `IngestDocumentIn.text` 1–500,000 | S (stricter: blank refused) | unchanged |
| `ingest_document(title)` | ≤ 200 | — (K-074) | S + M |
| `ingest_document(source_format)` | `sourceFormat` 1–50 | — | S + M |
| `ingest_document(source_label)` | `sourceLabel` ≤ 200 | — | S + M |
| `ingest_document(produced_by)` | `producedBy` ≤ 200 | effectively S (`AgentNotFoundError`) | not copied |
| `ingest_documents(items)` | `IngestDocumentsIn.documents` 1–20 items | max S, min — | max unchanged; min not copied |
| `ingest_documents(items[].*)` | same as `ingest_document` per item | `text` S; others — | S (per-item receipt) |
| `search_documents(query)` | `GET /documents/search` `q` ≥ 1 | — | M |
| `search_documents(limit)` | 1–200 | — | M |
| `get_document(document_id)` | `GET /documents/{id}` unbounded | n/a | — |
| `delete_document`, `get_document_deletion`, `get_document_history` `(document_id)` | path 1–200 | — | not copied |
| `list_documents(limit)` | 1–200 | — | M |
| `list_documents(current_only)` | bool | n/a | — |
| `list_pending_document_updates(limit)`, `list_document_updates(limit)` | 1–200 | — | M |
| `list_document_updates(status)`, `list_matches(status)` | `Query(None)`, no enum | n/a (unbounded both) | — |
| `confirm`/`reject`/`recheck_document_update(match_id)` | path 1–200 | — | not copied |
| `list_pending_matches(limit)`, `list_matches(limit)` | 1–200 | — | M |
| `confirm`/`reject`/`recheck_match(match_id)` | path 1–200 | — | not copied |

**Structural bounds beyond `Field`, checked:**

- No `pattern` and no `enum` on any MCP-mapped REST field. E3 found none: the only `Literal` is
  the storefront's `AdvanceOrderIn.transition`.
- Custom validators exist only on REST-only models: `_bounded_flat_dict` on `StartWorkflowRunIn`
  and `SubmitWorkflowInputIn`, whose merged-ctx bound the service already mirrors.
- FastMCP type checking (string/int/list) already matches REST's type checking (E2).
- The 69 REST constraints include 32 on routes with no MCP twin (workflow defs, runs and snapshots, plus `GET /threads/{tid}/workflow-runs`). These are `no_mcp_twin` in the
  probe. Their K-049-relevant key bounds are already mirrored in `_validate_key_lengths`.
