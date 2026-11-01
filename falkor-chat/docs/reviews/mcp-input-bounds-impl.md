# MCP input bounds — live-spec check (feature close)

> **Status:** active · **Owner:** `analyst` · **Tracks:** K-101

## Scope & verdict

This is close step 3 (root `AGENTS.md`, live-spec bullet) for K-101. I checked the uncommitted
`falkor-chat/docs/SERVER.md` diff (§1.4 table row, §1.4 *Error envelope*/*Bounds*, the §1.4
workflow-run *Bounds* paragraph, the §1.5 layout row, §2.2) against the shipped code in
`16f76f46` (service half) and `7490f634` (MCP half), read at HEAD. Server code has no uncommitted
changes. I also read the unchanged §2.2 lines, and swept `DESIGN.md`, `QUERIES.md`, `README.md`
and `AGENTS.md` for copies of these facts.

**Verdict: needs changes.** There is one blocker (F1), a two-sentence fix. Every bound,
error class, status code and layer assignment in the new text matches the code except where the
findings say otherwise.

CPG: considered, not relevant — a spec-vs-code check of about 20 named symbols, read directly and executed. A call graph would add nothing.

**Evidence run** (scratch scripts, fake repository that raises on any access, so "nothing
written" is observed rather than assumed):
- FastMCP `list_tools()` input schemas, plus `call_tool` through an in-memory client session for
  17 violating calls (`isError`, text and repository access recorded).
- REST via `create_app(mount_mcp=False)` + `TestClient`, for 13 requests, plus the OpenAPI
  parameters of the thread-window route.
- `pytest tests/test_services.py tests/test_input_bound_parity.py tests/test_mcp.py`: **480
  passed**. These do not wipe `reference`. The full suite, `test_storefront_api.py` and
  `test_queries.sh` were not run, as the brief required.

## Findings

### F1 — blocker: §2.2 says `ingest_documents`' `items` is untyped and isolates a non-dict entry; the code rejects the whole call

`SERVER.md` §2.2 (*Argument bounds*: "`items`, which is untyped so that one bad item becomes its
own receipt"; *per-item receipts*: "a non-dict entry comes back as its own …
`MalformedItemError` receipt"). The code is `mcp.py` `ingest_documents(items: list[dict[str, Any]])`.
FastMCP publishes `{"type":"array","items":{"type":"object"}}` and validates it. **Verified:**
`items=[1, {"text":"  "}]` returns `isError=True`, "items.0 Input should be a valid dictionary".
No receipt is produced for either item, the repository is never touched, and REST
`IngestDocumentsIn` also answers `422` for the whole body. The service's non-dict branch can be
reached only by an in-process caller.
This is the receipt contract K-101 changed, restated in rewritten text with a false rationale.
An agent that relies on it gets a whole-batch failure where it expects isolation.
**Fix:** say that `items` is a list of objects with untyped fields, so a non-object entry fails
the whole call as a validation error, while a bad field inside an object becomes that item's
receipt. Remove "a non-dict entry" from the MalformedItemError list.

### F2 — minor: the MalformedItemError list omits non-string `produced_by`

`SERVER.md` §2.2 *per-item receipts* lists only "missing or non-string `text`" (once F1 is fixed).
The code is `services.py` `ingest_documents`, whose guard also covers
`not isinstance(doc.get("produced_by"), (str, type(None)))`. **Verified:**
`{"text":"a","produced_by":5}` returns a `MalformedItemError` receipt. **Fix:** "missing or
non-string `text`, or a non-string `produced_by`".

### F3 — minor: "the exception is whitespace-only message text" — whitespace-only document text is a second one

`SERVER.md` §1.4 *Bounds* names one stored value that passes REST's model and reaches the service.
`IngestDocumentIn.text` is also only `min_length=1`. **Verified:** `POST /documents {"text":"   "}`
returns `400 EmptyDocumentError`, and `/documents/batch` returns `201` with that item's
`EmptyDocumentError` receipt. That receipt is the only bound-violation receipt REST can produce,
because declared bounds `422` the whole batch (verified: a 201-char `title` → `422`). The
sentence "a violating item becomes that item's own error receipt" reads as broader than that.
**Fix:** "The exceptions are whitespace-only text: a message's answers `400 InputBoundError`, a
document's `400 EmptyDocumentError`, or, in `/documents/batch`, that item's receipt; every
other batch violation is a whole-body `422`."

### F4 — minor: §2.2 says "each tool's argument bounds are advertised in its inputSchema"; three are not

`SERVER.md` §2.2 lead sentence. **Verified** (`list_tools`): `ingest_document.text` is
`{"type":"string"}`, with no 500,000 bound and no non-blank rule. `send_message.body` has no
non-blank rule, and `ingest_documents.items` has no `maxItems` 20. The *Argument bounds* bullet
lists exactly these as service-only, so the lead contradicts its own bullet. **Fix:** "Each
*declared* argument bound … is advertised in its `inputSchema`. The service-only ones below are
stated in the tool's description."

### F5 — minor: the MCP error text shape is unstated, and "comes back as an `InputBoundError` tool error" implies a class name the client never sees

Completeness item 2. **Verified:** a service error comes back as `isError` text
`Error executing tool <tool>: <message>` (FastMCP's wrapping), e.g. `Error executing tool
send_message: message text must not be empty or whitespace-only`. The class name is not in it.
A signature violation comes back as `Error executing tool <tool>: 1 validation error for
<tool>Arguments | <arg> | <pydantic message> …`. **Fix:** state the shape once in the §2.2 lead
("…an `isError` result whose text is `Error executing tool <tool>: <message>`; the class name is
not included"), and in *Argument bounds* say "comes back as a tool error", without a class name.

### F6 — minor: the same fact is stated in two or three places

The whitespace-only message rule appears three times: the §1.4 table row for
`POST /threads/{tid}/messages`, §1.4 *Bounds* and §2.2 *Argument bounds*. The MCP ranges are given
in §1.4 ("The MCP tools declare the same ranges (§2.2)") and listed again with their numbers in
§2.2. All copies agree with the code today, which I verified by the probes above. The table row
is still the copy that drifts unnoticed: no other row in that table carries a bound. **Fix:**
drop the table-row clause, since *Bounds* sits directly below it. In §2.2, keep the
argument→bound mapping but cite §1.4 for the values. The room-wide `read_messages` `limit` is
the only MCP-only fact there.

### F7 — nit: "Each bound is declared once, as a `schemas.py` constant" overclaims

`GET /threads/{tid}/workflow-runs` has `Query(10, ge=1, le=50)` as literals (`api.py`
`list_workflow_runs`), and so do `since` `ge=0` and every lower bound. **Fix:** "Each shared
upper bound (one that REST, MCP or the service enforces in more than one layer) is declared
once…".

## Brief's specific questions

- **The `{mid}` correction:** correct. `match_id` is `Path(…, max_length=MAX_ID_LEN)` on all six
  `/document-updates/{match_id}/…` and `/matches/{match_id}/…` routes. A 201-char id returns `422`
  (verified). `GET /messages/{msg_id}` has no bound: a 300-char id reached the service (verified).
- **REST `since` ≥ 0:** true. `read_thread` has `Query(None, ge=0)`, OpenAPI shows `minimum: 0`,
  and `since=-1` returns `422` (verified).
- **`producedBy` as a request-only bound:** true. `IngestDocumentIn.producedBy` has
  `max_length=MAX_NAME_LEN`, and a 201-char value returns `422` (verified). `Services.ingest_document`
  has no length check on `produced_by` and raises `AgentNotFoundError` when
  `ingestorFound` is false (read).
- **Bounds verified by symbol:** `MAX_TEXT_LEN`=8000, `MAX_MENTIONS`=50, `MAX_NAME_LEN`=200,
  `MAX_LIST_LIMIT`=200, `MAX_READ_LIMIT`=1000, `MAX_SOURCE_FORMAT_LEN`=50,
  `MAX_DOCUMENT_CHARS`=500,000, `MAX_BATCH_SIZE`=20, `MAX_ID_LEN`/`MAX_KEY_LEN`=200,
  `MAX_SWEEP_LIMIT`=1000. Every REST `limit` was read in `api.py`, and every MCP schema was
  executed. Service checks run before any repository call; the fake repository confirmed it for
  every violating call. `MessageTooLongError` and `InputBoundError` map to `400` through the
  generic `ServiceError` handler (`app.py`). "No MCP tool fronts a workflow run" (§1.4 workflow
  bounds) is true: there is no workflow tool in `list_tools()`. Empty `items` returns `[]`, and a
  21-item batch returns a `BatchTooLargeError` tool error (both verified).
- **Live-spec form:** passes. The added lines contain no `K-`, version or "now/no longer"
  phrasing, the section numbering is unchanged (§1.1–§1.12, §2.1–§2.3), and the removed "(K-074)"
  history clause was the right cut.

## What's solid

- The three-layer split in §1.4 *Bounds* (stored values in the service, read parameters per
  transport, path segments in REST only) is the actual architecture, and it reads clearly.
- The old false "No schema layer on MCP" bullet and the wrong "every `{mid}`" claim are both
  gone, and their replacements are accurate.
- The per-item receipt list matches the code exactly for `title`/`source_label`/`source_format`,
  including the not-a-string and `null` cases. I verified all of them.

## Open questions

None.

### Proposed REVIEW.md lines

Target: `falkor-chat/REVIEW.md`, *Always check*. Source: F1. The prose rationale and signature
both read as "untyped", and only the generated schema showed otherwise.

- *Addition:* In a spec check of an MCP tool's input contract, read the generated `inputSchema`
  (FastMCP `list_tools()`) and drive one violating `call_tool`, rather than reading the signature or
  docstring. A claim that an argument is untyped or unbounded, or that a bad item becomes its own
  receipt rather than failing the call, must hold in the schema and in the result.

There are no prunes. `tests/test_input_bound_parity.py` pins REST↔MCP schema parity, but nothing
checks spec prose against the schema.

## Pass 2

**Verdict: needs changes.** All seven pass-1 findings are fixed. The revision also gave a
sentence I had approved in pass 1 a new consequence, and I now see that sentence is false (N1).
I re-ran every scratch probe against the tree: 17 MCP calls, the batch receipts, 14 REST requests,
and a `list_tools()` description check. Server code is unchanged since pass 1 (`git status`).

- **F1:** fixed. Re-run: `items=[1, {...}]` returns `isError` "items.0 Input should be a valid
  dictionary", the whole call fails, and the repository is untouched. Each bad field still
  produces its own receipt (8/8).
- **F2:** fixed. A non-string `produced_by` returns a `MalformedItemError` receipt, and the
  rewritten list now names it.
- **F3:** fixed. A whitespace-only message returns `400 InputBoundError`, `POST /documents "   "`
  returns `400 EmptyDocumentError`, and in a batch that becomes the item's receipt (`201`). A
  201-char batch `title` returns a whole-body `422`.
- **F4:** fixed. The service-only bounds are in the tool descriptions: `send_message` says
  "non-blank", `ingest_document` says "non-blank" and "500,000", and `ingest_documents` says
  "20 items". None of them is in the `inputSchema`.
- **F5:** fixed. Service errors and signature violations alike come back as
  `Error executing tool <tool>: <message>`, with no class name.
- **F6:** fixed. The table-row clause is gone, and §2.2 now cites §1.4 for the values. Every
  argument it lists has its value stated in §1.4; the room-wide `read_messages` `limit` maps to
  the thread window's 1–1000. One value copy remains, a nit: §2.2's "≤ 20 batch cap" restates
  §1.4's "≤ 20 documents". Writing "the batch cap" would drop it.
- **F7:** fixed. "Each upper bound shared by more than one layer" holds: every such bound
  (8000/200/50/1000/50/500,000/20) is a `schemas.py` constant, and the remaining literals
  (`le=50` on workflow-runs, `ge=0`) belong to a single layer.
- **Cross-fix consistency:** checked. F3's whitespace exception, F1's receipt bullet and F5's
  error shape agree. A receipt is a successful call, so the `isError` shape does not apply to
  it, and only a non-object entry fails the call.
- **Growth-stops:** passes. Root `AGENTS.md` applies it to requirements, plans and reviews, not
  to a live spec, so I measured instead. `git diff --stat` on `SERVER.md` went from 59 to 56
  inserted lines between passes, so the rework replaced text in place. The one addition is
  `DESIGN.md` §1.3's row, which is new scope (embedder-context F1).

### N1 — blocker: "in-process callers are held to them alike" and "only an in-process caller sees `MessageTooLongError`" are false for the in-process callers that actually write messages

`SERVER.md` §1.4 *Bounds*, stored values. **Traced:** the only callers of `Services.post_message`
are the three front doors: `api.py:100`, `storefront_api.py:1250` and `mcp.py:247`. All three
refuse an over-long text first: REST and MCP at 8000 (verified), and the storefront at 2000
(`storefront_api.py` `MAX_MESSAGE_TEXT_LEN`, which also lists the class in
`SERVICE_ERRORS_UNREACHABLE`). The in-process code that writes messages is the responder
(`responder.py:121`) and the workflow `post_message` tool (`tools.py:278`). Both call
`post_agent_answer`, which has no text or mention check (`services.py` `post_agent_answer`, read
in full). An agent answer longer than 8000 characters is therefore stored, and no shipped caller
can ever see `MessageTooLongError`. The `post_message` docstring says the exclusion is
deliberate (K-099), so the shipped contract is "client-authored messages are capped, agent
answers are not". The spec states the opposite, and it also counts "both front doors" when §1.5
names three.
**Fix:** "Message text written by a client (`post_message`, which every front door calls) is …
≤ 8000. Agent-authored answers (`post_agent_answer`: the responder, the workflow `post_message`
tool) are not capped, which K-100 tracks. Over-long text is `MessageTooLongError`, which none of
the three front doors lets through; it guards direct `Services.post_message` callers." Scope the
mentions cap the same way.
I missed this in pass 1. The "in-process callers … alike" sentence was there, and I checked it
against `post_message` only.

- **Nit:** "each a `400`" now sits on a sentence that covers MCP and in-process callers. It is a
  `400` only on REST, where MCP returns `isError` and an in-process caller gets an exception.
  Suggest "(a `400` on REST)".

## Pass 3

**Verdict: approve.** Scope: N1, the two nits, and the three edited passages (§1.4 stored-values
bullet, §2.2 *Argument bounds*, `DESIGN.md` §1.3 row). Server code is unchanged (`git status`).

- **N1:** fixed. Each claim checked against the code by symbol:
  - `post_message` has exactly three callers: `api.py:100`, `storefront_api.py:1250` and
    `mcp.py:247` (`git grep` over all of `falkor-chat/**/*.py`, tests excluded).
  - All three callers refuse an over-long text first. REST and MCP refuse at 8000, verified in
    passes 1 and 2. The storefront's `PostMessageIn` is `max_length=MAX_MESSAGE_TEXT_LEN` (2000)
    plus `_nonblank`. So no shipped caller reaches `MessageTooLongError`.
  - `post_agent_answer` has no `MAX_*` reference and no length or count check (read in full). Its
    callers are `responder.py:121` and `tools.py:278` (`PostMessageTool`).
  - "Three front doors" now matches §1.5.
- **Writer's front-door-only check:** confirmed by the same sweep. `create_channel` and
  `create_thread` are called only from `api.py` and `mcp.py`. `ingest_document` is called from
  those two and from `Services.ingest_documents`, which is itself called only from `api.py` and
  `mcp.py`. No script calls these service methods: `scripts/load_append.py` and
  `scripts/seed_nlq_eval_corpus.py` go through REST. `seed_eval_corpus.py` writes through
  `Repository`, which is outside the scoped "through `post_message`" claim.
- **Nit "each a `400`":** fixed. The text now reads "a `400` on REST, a tool error on MCP, §2.2".
- **Nit "≤ 20" in §2.2:** fixed. It now reads "the batch cap".
- **Anything new introduced by the edits:** none found. The edited bullets still agree with F1,
  F3 and F5.
- **Growth-stops:** passes. `SERVER.md` went from 56 to 58 inserted lines. The growth is the N1
  fact itself: agent answers are uncapped, a contract the spec had omitted. That is a new fact,
  not restatement or history. "chunking is K-100" is a forward pointer, not change history.
