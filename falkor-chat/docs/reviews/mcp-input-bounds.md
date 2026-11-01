# MCP input bounds (K-101): implementation review

> **Status:** archived · **Owner:** `analyst` · **Tracks:** K-101

## Pass 1 — service bounds + parity test

**Scope.** The uncommitted K-101 unit against `HEAD` (`fb46b6fe`): `server/falkorchat/services.py`,
`server/falkorchat/storefront_api.py`, and the tests in `test_services.py`, `test_mcp.py`,
`test_api.py`, `test_storefront_api.py` and the new `test_input_bound_parity.py`. The baseline is the
RCA `docs/reviews/mcp-input-bounds-rca.md` §5.2–§5.3 (Round 1, plus the service half of Round 2) and
the K-101 stakeholder decisions in `falkor-chat/docs/BACKLOG.md`. Out of scope: the MCP `Annotated`
signature unit, the `mcp.py`/`schemas.py` "no schema layer" docstrings, and the tree's other diffs
(`model-bench/`, `docs/plans/typed-decision-models*`).

**CPG:** considered, not relevant — the questions are which schema shapes one test walks and
where a check sits relative to repository calls; I answered both by reading the touched functions and
running planted mutants. `cpg_falkorchat` predates this uncommitted diff.

**Verdict: approve with suggestions.** The service bounds are correct. Every new check runs before
any repository access. `ingest_documents` keeps its per-item isolation. The REST change is exactly
the agreed whitespace 400, with the standard envelope. The parity guard is real for the shapes the
code has today: it goes red on an unmapped field, an empty `list_tools()`, and an unassigned MCP
constraint. But it does not see the schema shapes it does not recurse into (P1-1). **Fix P1-1 before
the MCP-signature unit lands**, because that unit relies on check (iii).

**Evidence run.**
- `pytest -q tests/test_services.py tests/test_input_bound_parity.py tests/test_mcp.py`: **415 passed**.
- `pytest -q tests/test_api.py -k "whitespace_only_message or empty_message_is_still or nonblank_message_is_still"`:
  **3 passed** (all three use the `fake_client` fixture, which needs no database).
- Six database-free `test_storefront_api.py` functions, called directly with no pytest fixtures:
  all pass. They are the ServiceError partition, the blank-message producer pin, the class-tree
  test, and the three raise-walk tests.
- Not run: the full suite, the rest of `test_storefront_api.py`, and the `wf_repo`/`wf_client`
  nodes in `test_api.py`. Nothing touched `reference`, so no re-seed is needed. `test_mcp.py`'s
  `conn` wiped only `ws:test`.
- Mutants were planted in a scratchpad copy of `server/` imported via `PYTHONPATH`. The
  working-tree files were never edited. The table is in Appendix A.

### Findings

**P1-1 (major) — the parity walk misses constraint shapes. It catches only the ones the code uses
today.** *Verified by two survivors on the REST side and one on the MCP side (Appendix A, M1, M2b,
M6).*
- `_walk_body` (`test_input_bound_parity.py`, "enumeration") records constraints only on a
  property node and its `anyOf` alternatives. It recurses into `items` only by calling
  `_walk_body`, which reads just `$ref`/`properties`.
- So a constraint on a list *element* is dropped. M1 used
  `mentions: list[Annotated[str, Field(max_length=200)]]`. FastAPI publishes it as
  `items.maxLength: 200`, and all 30 tests stay green.
- Parameters go through `_constraints(p["schema"])` only, with no `$ref` resolution. An Enum-class
  `Query` param (M2b) publishes `anyOf:[{$ref}]`, and its `enum` is invisible. A `Literal` param is
  caught (M2a), so the result depends on how the author spelled the type.
- `mcp_arguments()` has the same blind spot. An MCP-only element constraint (M6) passes check
  (iii)'s "MCP advertises no constraint the mapping does not assign it".
- The module docstring claims failure cases (i)–(iii) without this limit.

Why it matters: this test is the RCA's class guard (contributing factor (b)). Its value is catching
the *next* bound someone adds. Element bounds on a list (mention ids, batch item fields) and enums
are ordinary next bounds.

**Fix:** replace the shape-by-shape walk with one recursive collector that both sides share. It
should resolve every `$ref` (body, parameter and `$defs`) and descend through `anyOf`/`oneOf`/
`allOf`, `items`/`prefixItems`, `properties` and `additionalProperties`. It should give each
constraint a path suffix, for example `body.mentions[]`. Then **gate the collector with a coverage
probe, not with a list of shapes:**
- Build a synthetic `FastAPI` router and a synthetic `FastMCP` tool. Give them one constrained field
  for each value of each JSON-Schema composition axis:
  - location: path, query, body;
  - wrapping: bare, Optional, list element, dict value, nested model;
  - spelling: inline `Field`, `Annotated`, `Literal`, `Enum` class.
- Assert that the collector reports every planted constraint.
- Put the probe in its own small round *before* the collector rewrite, so it goes red against
  today's walker first, as M1/M2b/M6 do here.

**P1-2 (minor) — one service check survives a mutant.** *Verified (Appendix A, S5).* Guarding the
`source_label` check with `and title is None` leaves `test_services.py` and the parity test green
(348 passed). Every label probe omits `title`. The data is safe today, because a label is stored
only as the `title` fallback (`services.py` `title or source_label or ""`). But the service would
then accept what REST's `sourceLabel` `maxLength` 422s, which is the front-door disagreement K-101
exists to close.

**Fix:** add one literal test: `ingest_document(text=…, title="t", source_label="x"*201)` raises
`InputBoundError`.

**P1-3 (nit) — the storefront's unreachability reason is factually loose.**
`storefront_api.py` `SERVICE_ERRORS_UNREACHABLE[InputBoundError]` says "no storefront route creates
a channel or thread". `Storefront.join` does create both, but through
`repo.ensure_participant(… thread_title=THREAD_TITLE …)` (`storefront.py:794-805`), not through
`Services`. So the conclusion holds and the reason does not.

**Fix:** reword it to "…creates a channel or thread or ingests a document **through `Services`**".

**P1-4 (nit) — `test_mapping_entries_are_well_formed` allows `Mcp(...)` and `KnownGap("mcp")` in
the same entry.** The follow-up unit could add the `Mcp` entry and forget to drop the gap marker,
and nothing would fail.

**Fix:** mirror the existing `Service`/`KnownGap("service")` exclusion for the MCP layer.

### Divergences from the RCA

- **(a) Channel names and thread titles stay at length ≥ 1, so whitespace-only names are still
  accepted. Accepted.** The stakeholder decision names message text only, and the RCA baseline is
  exact parity. The cost is a blank-looking name, not a crash. The web UI trims and refuses blanks
  client-side (`web/app.js:677`, `:688`). See Open questions for the consistency call.
- **(b) `source_format=None`, or a non-string title or label in a batch item, now gets an
  `InputBoundError` receipt. Accepted, as an improvement.** Before this change a JSON `null`
  `source_format` was stored as a null `sourceFormat`. A non-string title such as `5` survived
  `title or source_label or ""` and was stored as an int. REST 422s both, so this is parity, not
  tightening. The MCP single-`ingest_document` signature already type-refuses `null`. Mutant S1
  (drop the `isinstance` branch) is killed. One nit: for a type error, `limit` carries `min_len`,
  which is not a limit.
- **(c) No-twin routes are listed per route. Accepted.** The list corrects itself in both
  directions:
  - `test_no_mcp_twin_routes_all_exist` catches a stale entry.
  - A new REST route lands in `PARITY_REST` and forces a `MAPPING` decision.
  - A future MCP twin for a listed route fails `test_every_twin_points_at_a_constrained_parity_field`
    until the route is removed from the list.
- **(d) The table is keyed per constraint keyword. Accepted.** It is finer than per field: a
  min/max pair can sit on different layers, as `text` does with `EmptyDocumentError` versus
  `DocumentTooLargeError`. Check (iv) also derives the direction (limit+1 or limit−1) from the
  keyword.

### Checks requested by the brief

1. **Does the parity test prove what it claims?** Yes for today's shapes, no for the shapes in
   P1-1.
   - It cannot pass vacuously on empty enumeration. M4 (`list_tools()` → `[]`) is red through
     `test_every_mcp_args_entry_names_a_live_tool_argument`. An empty REST walk would redden
     `test_every_mapping_entry_names_a_live_rest_constraint`.
   - Measured counts: 125 REST constraint keys, 64 outside the no-twin routes, 42 MCP arguments, 0
     MCP constraints.
   - M5 (annotating `send_message.body` without flipping its `KnownGap`) is red.
   - `test_mcp_layer_constraints_match_rest` is vacuous by construction until the first `Mcp`
     entry exists. Its docstring says so, and M5 shows the reverse direction already bites.
   - Mutants I planted beyond the implementer's list: M1, M2a, M2b, M4, M5, M6, S1, S2, S5.
2. **Ordering. Holds.**
   - `create_channel` and `create_thread` check first. The thread-title check runs ahead of
     `channel_exists`, and S2 (moving it after) is killed by the `_UntouchableRepo` test.
   - `post_message` runs its checks before `_validate_and_derive_role`.
   - `ingest_document` runs them after the two text checks and before `split_into_chunks` and
     any repository call.
   - In MCP, `pytest.raises(ToolError, match=…)` would also catch a repository touch, because the
     `_UntouchableRepo` message would not match.
   - Per-item isolation: an `InputBoundError` is a `ServiceError`, so the existing `except` turns
     it into a receipt. `test_ingest_documents_turns_an_over_long_title_into_that_items_receipt`
     shows the bad item fails while the two items around it land.
3. **REST behavior. The whitespace 400 is the only REST-visible change.**
   - Every other new bound is already a pydantic 422 on REST, including per batch item, before
     the service runs.
   - Whitespace channel names and titles still pass both layers, per (a).
   - The envelope is the generic handler's `{"error": "InputBoundError", "detail": …}` at 400
     (`app.py` `_handle_service_error`), the same shape as `DocumentTooLargeError`. It is pinned
     literally in `test_api.py`.
   - The browser client trims and never sends blank text (`web/app.js:701-702`), so the UI is
     unaffected.
4. **Stored data. No path re-writes a stored title or name through a bounded method.** *Traced.*
   - The only internal callers of `create_channel`, `create_thread`, `post_message` and
     `ingest_document(s)` are `api.py`, `mcp.py`, `storefront_api.py:1248` and the batch loop.
   - `background._safe_detect_update` reads `title` only to pass it to `find_update_shortlist`.
   - Agent replies go through `post_agent_answer`, which is untouched.
   - The seed scripts write through the repository or REST: `seed_eval_corpus.py` uses the
     repository, `seed_nlq_eval_corpus.py` uses `POST /documents`.
   - The 7 long titles in `ws:agent-team` stay readable. Only a deliberate MCP re-ingest that
     reuses one would be refused, which is K-074's intended outcome.
5. **Storefront unreachability. Holds.**
   - The storefront calls exactly one bounded method, `post_message`, with
     `mentions=[agent_id]`.
   - `PostMessageIn._check_text` uses `_nonblank`'s `value.strip()`, the service's own predicate,
     so no string reaches the service blank.
   - The producer test pins `""`, `"   "` and `"\n\t "`. Dropping `_check_text` turns it red.
   - `SERVICE_RAISES_TODAY` and the 15-member family count match. All six database-free partition
     and raise-walk tests pass.
   - The reason text is loose (P1-3).
6. **SERVER.md sections to update at feature close**, beyond RCA §6 (`§1.4` "Bounds", `§1.4`
   "Error envelope", `§2.2` "No schema layer", `§2.2` tools table):
   - **`§1.4` REST table, `POST /threads/{tid}/messages` row**: whitespace-only text is now
     `400 InputBoundError`. The RCA assumed REST unchanged.
   - **`§1.4` "Bounds"**: message text is non-blank, stricter than the model's `min_length=1`.
     Also, "every `{mid}`" is bounded ≤ 200 is false today: `api.py:148-149` declares `msg_id: str`
     with no bound. This contradiction predates the unit, but a spec-update check must fix it,
     because it carries over (root `REVIEW.md`).
   - **`§2.2` malformed-item bullet**: `ingest_documents` items now also get `InputBoundError`
     receipts, for out-of-bound or non-string `title`/`source_label` and for an empty, over-long
     or `null` `source_format`.

### What's solid

- `_check_length` is one small helper with literal-value tests at both edges (empty/1, 200/201,
  50/51 for `source_format`, 50/51 mentions). Mentions are counted as sent, and the 51-copies-of-one-id test pins that.
- Every refusal test runs against `_UntouchableRepo`, so the "before any repository access" claim
  is executed, not asserted.
- The K-099 tests were moved off `MAX_TEXT_LEN` onto literals, which clears the RCA's
  constant-under-test note at `test_mcp.py:244`.
- The parity table's ratchets (`test_no_service_layer_gap_remains`, the well-formedness checks,
  stale-entry checks in both directions) make the Round 2 flips mechanical.
- The new error strings are model-visible through MCP tool errors (`falkor-chat/REVIEW.md`). They
  are neutral: field, size, limit. Nothing in them invites a fabricated value or a bypass.

### Open questions

- **Whitespace-only names and titles (divergence (a)).** Should `create_channel`/`create_thread`
  also refuse whitespace, for consistency with messages and documents? That would be a second
  REST tightening, and it is the stakeholder's call. My recommendation is low priority: the cost is
  cosmetic.
- **Agent-authored blank messages.** `post_agent_answer` (the responder and the workflow
  `post_message` tool) can still write whitespace-only text. That matches the RCA scope (agent
  callers are out) and the decision's wording ("every front door"). It is noted so that the gap is
  recorded as a choice rather than discovered later.

### Proposed REVIEW.md lines

Source: P1-1. Target: root `REVIEW.md`, `## Always check`. This is an addition, and no existing
line is now enforced elsewhere.

- A guard that enumerates a schema or tree (OpenAPI, JSON Schema, AST) to find "every X" is checked
  with one planted X for each composition axis the format allows (nested object, array element,
  map value, `$ref`, `anyOf`/`allOf`), not only the shapes in use today. A walker that recurses
  into only some axes is a bound, not proof.

## Appendix A — mutants (scratchpad copy, restored after each, `__pycache__` cleared)

| ID | Mutation | Suite | Result |
|---|---|---|---|
| M1 | REST `PostMessageIn.mentions` → `list[Annotated[str, Field(max_length=200)]]` (published as `items.maxLength`) | parity | **survived** (30 passed) |
| M2a | `GET /document-updates` `status` → `Literal["pending","done"]` | parity | killed (unmapped REST field) |
| M2b | same param → `Enum`-class (`$ref`) | parity | **survived** (30 passed) |
| M4 | `mcp_arguments()` sees `tools = []` | parity | killed (stale `MCP_ARGS`) |
| M5 | MCP `send_message.body` annotated `min_length=1, max_length=8000`, `KnownGap` not flipped | parity | killed (unassigned MCP constraint) |
| M6 | MCP `send_message.mentions` element `max_length=200` (confirmed in `inputSchema` `items`) | parity | **survived** (30 passed) |
| S1 | drop `_check_length`'s `isinstance` branch | services + parity | killed (`format-none`) |
| S2 | thread-title check moved after `channel_exists` | services + parity | killed (`_UntouchableRepo`) |
| S5 | `source_label` check only when `title is None` | services + parity | **survived** (348 passed) |

## Pass 2 — P1 re-check

**Verdict: approve with suggestions.** All four Pass 1 findings are fixed. My survivors from Pass 1
were re-planted in a fresh scratchpad copy of the changed tree, and all are now killed. There is one
new minor finding (P2-1) and one nit (P2-2). Neither gates the MCP-signature unit.

**Evidence.**
- `pytest -q tests/test_services.py tests/test_input_bound_parity.py tests/test_mcp.py`: **421 passed**.
- Four database-free `test_storefront_api.py` functions, called directly: pass. They are the
  partition, the blank-message producer pin, and two raise-walk tests.
- Nothing else ran, so `reference` was untouched and needs no re-seed.
- Mutants are in Appendix B.

**Dispositions**
- **P1-1: fixed.** M1, M2b and M6 are now killed. M1 and M2b fail on an unmapped REST field; M6
  fails on an unassigned MCP constraint. The coverage probe and the raw-schema tests plant
  constraints across location, wrapping and spelling. The axis set misses dict keys (P2-1).
- **P1-2: fixed.** S5 is killed by
  `test_ingest_document_bounds_source_label_even_when_a_title_is_given`.
- **P1-3: fixed.** The reason now says "through `Services`" and names `Storefront.join`.
- **P1-4: fixed.** `test_mapping_entries_are_well_formed` rejects `Mcp` plus `KnownGap("mcp")`
  in one entry, and more than one `Mcp` per constraint.

**How the fixes interact (checked by execution).**
- **The old tables are unchanged under the new collector.** It reports the same inventory as
  Pass 1: 125 REST keys, 64 in parity, 42 MCP arguments, 0 MCP constraints.
- **No new keyword is absorbed.** No live route emits `uniqueItems`, `const`, `multipleOf` or
  `min/maxProperties` today. `PARITY_REST` filters only `NO_MCP_TWIN_ROUTES`, so a new keyword
  on a twinned route fails (i) loudly. That includes a `uniqueItems` that pydantic emits for any
  `set[...]`, which is a type, not a bound, but forcing a mapping decision for it is the safe
  direction.
- **"Walked once" is per ancestor chain, not global.** It cannot hide a model reused in two
  places. N9 makes it global, and 9 tests go red, because `IngestDocumentIn` sits under both
  `/documents` and `/documents/batch`.
- **Skipping `not`** opens no route that FastAPI or pydantic emits today. P2-1's fix would make
  the skip an explicit allow-list entry.
- **Path suffixes pair correctly in (iii).**
  - Positive control N7b (REST and MCP `mentions[]` both `maxLength` 200, mapped to
    `Mcp("send_message", "mentions[]")`) is green.
  - N7a (the same entry mapped without `[]`) is red.
  - N7c (MCP advertises 199) is red.

**P2-1 (minor, new): dict-key constraints are still invisible.** *Verified (N1).*
- pydantic 2.13.4 publishes `dict[Annotated[str, Field(max_length=5)], str]` as
  `propertyNames: {maxLength: 5}`. I confirmed this in the live OpenAPI on `PostMessageIn.tags`.
- `schema_constraints` never descends into `propertyNames`, so the planted field leaves all 35
  tests green.
- The coverage probe enumerates wrappings as bare, Optional, list element, dict value, tuple
  position and nested model. Dict *key* is the missing value. `contains`, `if`/`then`/`else`,
  `dependentSchemas` and `unevaluated*` are the same kind of gap, not emitted today.

**Fix:** close it by keyword, not by adding one more shape. Make the collector raise on any key it
neither recurses into nor knows to be inert. The inert set is `type`, `title`, `description`,
`default`, `format`, `required`, `examples`, `discriminator`, plus an explicit `not`. Then the next
framework keyword fails the guard instead of being skipped. Add `propertyNames` (suffix `{key}`)
to the recursion and add a dict-key field to both probes. Plant the tripwire's raw-schema test,
with an unknown keyword carrying a bound, before the change.

**P2-2 (nit, new): the twin check matches field names by bare prefix.** *Verified (N3).*
- `test_every_twin_points_at_a_constrained_parity_field` uses `f.startswith(field)`.
- Its `_split(...)[0]` head comparison is `"body"` for every REST body field, so it adds nothing
  there.
- A twin mistyped as `Twin(DOC, "body.source")` passes, because `body.sourceFormat` starts with it.

**Fix:** require `f == field or f[len(field)] in ".[{"`.

## Appendix B — Pass 2 mutants (fresh scratchpad copy, restored after each, `__pycache__` cleared)

| ID | Mutation | Result |
|---|---|---|
| M1 / M2b / M6 | Pass 1 survivors, re-planted verbatim | killed (unmapped REST field / unmapped REST field / unassigned MCP constraint) |
| S5 | `source_label` checked only when `title is None` | killed (new P1-2 test) |
| N1 | REST `PostMessageIn.tags: dict[Annotated[str, Field(max_length=5)], str]` (`propertyNames`) | **survived** (35 passed) |
| N3 | `MCP_ARGS` twin `body.sourceFormat` → `body.source` | **survived** (35 passed) |
| N7a | M1+M6, `MAPPING[(MSG, body.mentions[], maxLength)] = Mcp(send_message, "mentions")` (no suffix) | killed (2 tests) |
| N7b | same with `"mentions[]"`: positive control | green, as it should be |
| N7c | N7b with MCP advertising 199 | killed |
| N9 | recursion guard made a global visited set | killed (9 tests) |

## Pass 3 — MCP signature bounds

**Scope.** The uncommitted second half of K-101 against `HEAD` (`e7ccf17d`; the service half is
`16f76f46`): `mcp.py`, `schemas.py`, `api.py`, `tests/test_input_bound_parity.py` and
`tests/test_mcp.py`. Out of scope: the `services.py` hunk (a K-099 docstring),
`scripts/ensure_demo_models.*` and `tests/test_ensure_demo_models_script.py`.

**CPG:** considered, not relevant — the questions are about annotations, published schemas and
test reach, and I answered each by executing it. `cpg_falkorchat` predates the diff.

**Verdict: approve with suggestions.** Every bound now sits on its §5.2 layer, and none is tighter
than REST, the service, or any known client. A violation reaches the client as a legible `isError`
result. Check (iii) is no longer vacuous. The walker is closed-world for JSON Schema. The two new
findings are nits. Once P3-2's one-line fix lands, **the walker-shape finding class is closed**, and
further walker-shape findings are out of scope for K-101.

**Evidence.**
- `pytest -q tests/test_services.py tests/test_input_bound_parity.py tests/test_mcp.py`: **478 passed**.
- `pytest -q tests/test_api.py -k "<not each of the 24 wf_repo/wf_client tests>"`: **95 passed,
  24 deselected**. The `--setup-plan` output for the selection lists 0 `wf_*` fixtures.
- `reference` was untouched, so no re-seed is needed.
- Mutants ran in a fresh scratchpad copy (Appendix C).

**Dispositions**
- **P2-1: fixed.** `propertyNames` is recursed with the `{key}` suffix and is in both probes. The
  closed-world tripwire covers the rest of the keyword axis (check 1 below).
- **P2-2: fixed.** `_is_constrained_twin` matches at a field boundary, and
  `test_twin_field_matches_only_at_a_field_boundary` pins `body.source` as rejected.

**Checks requested by the brief**

1. **Closed world. It holds for JSON Schema, and a collection error is loud enough.**
   - Planted live keywords: `json_schema_extra={'maxGraphemes': 3}` on REST `CreateChannelIn.name`
     (K1), and `x-maxGraphemes` on MCP `Name` (K3). Both raise `UnknownSchemaKeywordError` at
     import.
   - Making `contains` inert (K5) is killed by the raw-schema tripwire test.
   - `test_live_schemas_use_only_keywords_the_collector_knows` pins that the live walk was not
     empty.
   - **Loud enough:** K1 ran together with `test_services.py`. pytest reported `1 error`, exit 2,
     and *no test of either file ran* (the session stops at collection, and `pyproject.toml` has no
     `--continue-on-collection-errors`). That cannot be misread as the §1.7 hazard: it is not a
     skip and not exit 0. The real cost is the other way: a harmless new keyword stops the whole
     default suite. K2 shows the realistic case, `Field(deprecated=True)`. That is an accepted cost
     of being closed-world, and the error message says exactly what to do.
   - **The remaining gap is one layer above JSON Schema** (P3-2).
2. **No annotation is tighter than REST, the service, or a known client.** *Verified.*
   - `test_value_exactly_at_the_bound_reaches_the_service` passes the boundary value for every
     annotated argument, and the parity check (iii) compares every `Mcp` entry with REST's value.
     M11 (`read_messages` `limit` capped at 200 instead of 1000) is killed.
   - Clients are within bounds:
     - mcp-monitor sends `since=0, limit=200` (`mcp-monitor/config.example.toml:77`) against
       ≥ 0 and 1–1000.
     - `agent-kb-retrieval` sends `search_documents(limit=5)` with a non-empty prefixed query.
     - Kiro sends `send_message` with a non-empty body and 1 mention, and `read_messages(re=…)`.
   - The `claude/` capture template is `title=<the fact, one line>`, with no cap. A 243-character
     title now returns, over an in-memory client session, `isError` with `1 validation error for
     ingest_documentArguments` / `title` / `String should have at most 200 characters`. It is
     legible and names the field and the limit. It is also prevented up front, because the tool
     description now says "at most 200". See Open questions.
3. **`api.py` "byte-identical OpenAPI": false, but the constraints are identical.** *Run.*
   - I dumped the router's OpenAPI from a scratch copy with `HEAD`'s `api.py`/`schemas.py`
     (`git show`) and from the tree. The two dumps differ (35,685 vs 35,811 bytes).
   - A structural diff finds exactly one change: `components.schemas.IngestDocumentsIn.description`.
     That is `schemas.py`'s rewritten docstring, which FastAPI publishes.
   - Every constraint is unchanged. The 12 `Query` swaps are no-ops (P3-1).
4. **The three changed existing tests are justified. None of them weakens a check.**
   - `test_send_message_over_max_text_len_…` now matches pydantic's text, because the signature
     refuses first. The service cap is still pinned in `test_services.py`
     (`test_post_message_rejects_text_over_max_text_len_before_any_repo_call`).
   - The service-via-MCP parametrization shrank to the two whitespace cases. Those are the only
     stored-value refusals still reachable through MCP. The removed cases moved, with literals, to
     `test_client_sees_a_tool_error_naming_the_field_and_the_limit`.
   - The description test swapped `MAX_TEXT_LEN` for a literal, which makes it stronger. M10b
     (`ListLimit` drifting to 201) passes the parity test by design, because both sides read the
     constant. It is killed by 10 literal pins in `test_mcp.py`.
5. **Every service check still has a test that reaches it.**
   - Through MCP, the service is reached only by whitespace-only `body`, which passes
     `min_length=1`, and by `ingest_documents` items, which are untyped.
   - Every other `InputBoundError` branch is now unreachable from MCP and REST alike. Each one is
     still exercised directly: by `test_services.py`'s literal tests and by the parity check (iv)
     probes, all through `Services(FakeRepo())`. In-process callers keep the guarantee.

**P3-1 (nit) — the "byte-identical" claim is wrong, and the REST description now describes MCP
internals.**
- The only OpenAPI change is `IngestDocumentsIn`'s description, which now tells REST readers about
  MCP's untyped `items`.
- **Fix:** correct the claim in the delivery record to "constraint-identical (one description
  changed)". Optionally, move the MCP sentence from the docstring into a `#` comment so it stays
  out of the REST docs.

**P3-2 (nit) — the closed world stops at the OpenAPI parameter object.** *Verified with a planted
spec.*
- `rest_constraints` reads `param.get("schema", {})`. A parameter in the legal `content` form,
  `{"in": "query", "name": "p", "content": {"application/json": {"schema": {"maxLength": 5}}}}`,
  yields `{}`, silently.
- FastAPI does not emit that form today. But it is the one position the tripwire does not cover.
- **Fix:** raise `UnknownSchemaKeywordError` when a parameter has no `schema`, and plant that spec
  in `test_rest_collector_resolves_referenced_parameters_and_request_bodies`'s neighbour first.

**SERVER.md sections to update at feature close**, in addition to the Pass 1 list:
- **`§2.2` tools table**: each signature's bounds, or a note that they are advertised in
  `inputSchema`.
- **`§2.2`**: a bound violation is an `isError` result naming the argument and the limit, before
  any service call. This is a client-visible contract.
- **`§1.4` "Bounds"**: the MCP room-wide `read_messages` `limit` is 1–1000, and it has no REST twin.
- Changed model-visible text (`falkor-chat/REVIEW.md`): the tool descriptions now state limits and
  "non-blank". They invite no fabricated value and no bypass, so no measurement is needed.

**Open questions**
- **The agent capture template** (`claude/*/<agent>.md`,
  `ingest_document(title=<the fact, one line>, …)`) states no 200-character cap. The tool
  description now advertises it, and the error is legible. Should `cobb` add "≤ 200 characters" to
  the template anyway, so a capture is not lost to a model that does not retry?

### Proposed REVIEW.md lines

Nothing new this pass. For *this* guard, the closed-world tripwire now enforces the walker-shape
class. Pass 1's proposed line still stands, because it targets *other* guards that enumerate a
schema or tree, and no test enforces it for those.

## Appendix C — Pass 3 mutants (fresh scratchpad copy, restored after each, `__pycache__` cleared)

| ID | Mutation | Result |
|---|---|---|
| K1 | REST `CreateChannelIn.name` `json_schema_extra={'maxGraphemes': 3}` | killed: collection error, exit 2, 0 tests ran (even with `test_services.py` in the same run) |
| K2 | REST `CreateChannelIn.name` `deprecated=True` (harmless keyword) | collection error. A false alarm, accepted cost of closed world |
| K3 | MCP `Name` alias `json_schema_extra={'x-maxGraphemes': 3}` | killed: collection error |
| K5 | `contains` added to `INERT_KEYS` | killed (raw-schema tripwire test) |
| M10b | `ListLimit` `le=MAX_LIST_LIMIT + 1` | survives parity (by design), killed by 10 `test_mcp.py` literal pins |
| M11 | `read_messages.limit` `le=200` | killed (`test_mcp_layer_constraints_match_rest`) |
| — | planted OpenAPI parameter in `content` form | **silently `{}`** (P3-2) |
