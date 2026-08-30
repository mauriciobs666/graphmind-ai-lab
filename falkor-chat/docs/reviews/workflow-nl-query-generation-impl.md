# `workflow-nl-query-generation` — implementation-diff review

> **Status:** active · **Owner:** `analyst` · **Tracks:** K-055 (M6)

## Scope

Full code review of K-055's implementation mechanism as it stands today — the constrained
query-builder DSL and its consumers — diffed against pre-K-055 base `824c8359` across:
`server/falkorchat/querygen.py`, `repository.py`'s `run_readonly_query`, `services.py`'s
`run_structured_query`, `tools.py`'s `QueryGraphDataTool` (incl. the structured-completion system
prompt), and the corresponding test files (`test_querygen.py`, `test_querygen_live.py`,
`test_repository.py`, `test_services.py`, `test_tools.py`).

This is the **second** of this coordination's two required gates — the implementation-diff review
following the design-time review (`docs/reviews/workflow-nl-query-generation.md` /
`-security.md`, both approved) and the RCA-driven rework
(`docs/reviews/workflow-nl-query-generation-rca.md`) that followed a golden-set failure
(`docs/test-reports/workflow-nl-query-generation-report.md`, 46.2% vs. 85% gate). Read the RCA and
both 2026-08-30 `claude/graph-dba/falkordb-quirks.md` entries ("`RETURN DISTINCT` ... `ORDER BY`"
and "the safe shape for distinct projection, ordered by a column NOT in the projection") in full
before this review, per the brief — their reasoning was independently re-verified below, not
taken on trust.

**Method.** Read all four production files and their tests in full; read the RCA and the two
quirks entries; ran the full offline suite (`test_querygen.py`/`test_repository.py`/
`test_services.py`/`test_tools.py`, 553 passed) and the live suite
(`pytest -m live tests/test_querygen_live.py`, 9 passed) against the real shared instance;
independently probed the tuple-`DISTINCT` compilation live via the `cypher` MCP tool and via
`repo.run_readonly_query` directly (single- and two-column cases, both correct); ran two mutation
tests against a scratchpad copy of `querygen.py` (never against the production file — restored
and diffed clean immediately after each run); re-seeded `reference`/`ws:acme` after the default
`pytest` run's documented teardown wipe (`falkor-chat/AGENTS.md`), verified with
`verify_catalog.sh`/`verify_workflows.sh`/`verify_salesperson.sh`, all `OK`. One incidental stray
graph (`ws:EMBEDDING_DIM=1024`, created by my own bootstrap-script typo, never touched by any
other agent) was deleted immediately — not `reference`/`ws:nlq-eval`/`ws:acme`.

**CPG:** considered, not relevant — `cpg_falkorchat` is confirmed stale (built 2026-08-26T22:27Z,
predates every commit under review); the brief instructs reading the files directly, which this
review did throughout.

## Verdict: **needs changes**

The tuple-`DISTINCT` compilation itself is correct — I independently confirmed it live for both
the single-column and (untested) two-column cases, and it matches the graph-dba-verified safe
shape exactly. Fixes A/B/D are each correctly scoped and match their RCA specification. But two
findings below are real: a MAJOR test-coverage gap in the exact branch the brief flagged as
highest-risk (a column-misalignment mutation ships silently, caught by nothing — unit or live),
and a MAJOR robustness gap where a plausible malformed model completion crashes the entire
workflow run instead of the tool's own documented graceful abstention. Neither invalidates the
mechanism's core safety property (no path to a Cypher write-keyword still holds, independently
re-confirmed), so this does **not** block a golden-set re-run on correctness grounds — the DSL
compiles the golden set's shapes correctly, live-verified against real data. It does block calling
this implementation *done*: ship the two fixes below (both are small, targeted) before treating
K-055 as closed.

## Findings

### MAJOR — the tuple-`DISTINCT` re-aliasing has no test past one column; a column-swap mutation ships silently

`querygen.compile()`'s final `RETURN` in the `needs_tuple_distinct` branch
(`querygen.py:410-414`) maps `zip(return_exprs, aliases)` back to backtick-quoted original
expression text. Every existing test (`test_querygen.py`'s five `tuple_distinct`/regression tests,
all 9 `test_querygen_live.py` cases) exercises `returns` with **exactly one** entry. I constructed
the case the brief asked for — two non-aggregate `returns` columns plus a third `order_by` column
not among them (`returns=["p.name","p.category"], order_by="p.price"`) — and confirmed the real
code produces correct paired output (`{"p.name": "Gaming Mouse Pad XL", "p.category":
"Peripherals"}`, live-verified against `reference`). I then mutated a **scratchpad-only** copy of
`querygen.py` to reverse the alias order in that same `zip()` (`zip(return_exprs,
list(reversed(aliases)))`) and re-ran the full suite (`test_querygen.py` **and**
`test_querygen_live.py`) against it: **53/53 still pass.** Running the mutant live against the
same two-column request produces silently swapped values
(`{"p.name": "Peripherals", "p.category": "Gaming Mouse Pad XL"}`) — a real, silent, wrong-answer
regression that the entire test suite is blind to today, in exactly the branch the RCA flagged as
subtlest and most recently changed.

**Fix:** add a `test_querygen.py` case with 2+ non-aggregate `returns` entries plus an `order_by`
outside them (assert the compiled `RETURN ... AS` mapping is per-column, not just "contains
`WITH DISTINCT`"), and extend one `test_querygen_live.py` case the same way, asserting the actual
paired row values.

### MAJOR — a duplicate `returns` entry crashes the whole workflow run instead of abstaining, contradicting `QueryGraphDataTool`'s own documented contract

Neither `QueryRequest._returns_shape` nor `compile()` rejects a `returns` list with a repeated
expression (e.g. `["p.name", "p.name"]`, or `["count(p)", "count(p)"]` — a duplicated projection
or aggregate a small model could plausibly emit). I confirmed live: `compile()` accepts it, and
`repo.run_readonly_query()` raises `redis.exceptions.ResponseError: Error: Multiple result columns
with the same name are not supported.` — reproduced both in the tuple-`DISTINCT` branch and the
plain-`DISTINCT` branch. `QueryGraphDataTool.run()` (`tools.py:984-990`) only catches
`(ValidationError, ValueError)` around request construction and `compile()`; the call to
`self._services.run_structured_query(...)` (`tools.py:993-995`) is **unwrapped**. This exception
propagates through `services.run_structured_query` → `ToolRegistry.dispatch` (`tools.py:185`,
itself unwrapped) → `executor._handle_tool_call`, which by explicit, documented design
(`executor.py:913-916`, "this is still deliberately NOT a blanket `except Exception`... an engine
fault is never caught here") does **not** absorb a non-`ServiceError` — it propagates to `_drive`'s
outer net (`executor.py:503-505`), which fails the **entire workflow run** (`status=failed`), not
just this one tool call.

This directly contradicts `QueryGraphDataTool`'s own class docstring (`tools.py:913-917`): "Every
failure short of a genuine infrastructure fault ... returns the same abstention shape as 'no
matching data found,' never a fabricated answer and never a crash." A duplicate-`returns`
completion is not a genuine infrastructure fault (no LLM/provider outage, no DB connectivity
issue) — it is a DSL-legal-but-engine-rejected request that `compile()` should have caught. This
predates this session's A/B/C/D fixes (not introduced by the tuple-`DISTINCT` correction), but is
squarely in scope for "the mechanism as it stands" and is exactly the adversarial-input class this
review's brief calls out.

**Fix:** add a uniqueness check on `request.returns` — either a `QueryRequest` field_validator
alongside `_returns_shape`, or a `_require(len(return_exprs) == len(set(return_exprs)), ...)` in
`compile()` — raising `ValueError`, which `tools.run()` already catches and converts to the
intended "no matching data found" abstention. No other engine-runtime-error class was found during
this review (order/aggregate/type shapes all compile-time validated); this one gap is narrow and
cheap to close.

### MINOR — `run_readonly_query`'s "never aliases" docstring claim is now false in one branch

`repository.py`'s `run_readonly_query` docstring states "`querygen.compile` never aliases a
`RETURN` expression, so a key is the raw expression text FalkorDB assigns" — and
`test_repository.py:3482-3484`'s comment repeats the same claim verbatim. Since the
`needs_tuple_distinct` fix, `compile()` **does** alias (`RETURN c0 AS \`p.name\``) — the docstring's
functional promise (dict key = original expression text) still holds, live-confirmed, because the
alias target is the backtick-quoted original text, but the literal "never aliases" claim is wrong
and could mislead a future reader who trusts the docstring over the code.

**Fix:** reword to "`querygen.compile` never aliases a `RETURN` expression to anything **other
than its own original text** — the tuple-`DISTINCT` branch aliases internally (`c0`, `c1`, ...)
but always re-aliases the final `RETURN` back to the original expression, so this contract holds
in both compiled shapes."

### MINOR — a bare boolean filter value against a non-bool property is neither coerced nor rejected

`compile()`'s type-coercion logic (`querygen.py:320-336`) only fires `if isinstance(value, str)`.
A `QueryFilter.value` of `True`/`False` against a `float`- or `str`-typed property (e.g.
`price = true`) passes straight through unvalidated, becomes `$p0 = True`, and silently matches
zero rows — the exact same class of unnoticed-scope failure (silent abstention instead of a
compile-time reject) that RCA fix A was built to close for numeric-string values, left open for
this one adjacent type. Confirmed live: `compile()` accepts it without error, producing
`params={"p0": True}`. Note: the brief's specific concern about `float(True) == 1.0` does **not**
apply here — that coercion path (`declared_type(value)`) is only reached when `isinstance(value,
str)`, so a `bool` value never enters it; this is a distinct, narrower gap (no validation at all,
not a wrong coercion).

**Fix:** low priority (no property in either registered schema is bool-typed today, so this can
only ever produce a false abstention, never a wrong answer) — worth a one-line note in
`DatasetSchema`'s docstring flagging it as a known gap for the next schema that adds a bool-typed
property, or a `_require(type(value) is declared_type or ..., ...)` type-identity check alongside
the existing coercion.

## What's solid

- **The tuple-`DISTINCT` compilation itself is correct**, live-verified against real data for both
  the single-column (`nlq-16`/`nlq-25` shapes, matching golden-set values or documented structural
  intent) and two-column cases I constructed independently. It matches the graph-dba-confirmed
  safe shape (`WITH DISTINCT <tuple> ORDER BY ... LIMIT ... RETURN <re-aliased>`) exactly, and
  correctly declines the tuple path when `order_by` is already in `returns` or when any `returns`
  entry is an aggregate — both confirmed by test and by independent live trace.
- **The typed `DatasetSchema` shape change is transparent everywhere it needed to be** — every
  `prop in allowed_props` call site still does key-membership as before (confirmed by reading and
  by direct execution of `_describe_dataset_schema`, whose `sorted(props)` over a dict correctly
  yields sorted property names, not accidentally the type values).
- **The `*Normalized` fix reuses `extraction.normalize_name` exactly** (confirmed by reading both
  definitions) and is correctly scoped to `=`/`<>` only — a `<`/`>` filter against a `*Normalized`
  property is deliberately left un-normalized, matching the design intent.
- **The hardened prompt is byte-identical to the RCA's own recommended replacement text**
  (`docs/reviews/workflow-nl-query-generation-rca.md` §4 Priority 2) — confirmed no drift, no
  weakened pre-existing constraint (matches cardinality, op whitelist, JSON-only reply all
  survive verbatim), and every one of its four worked examples independently compiles via the
  real `compile()` against the real schemas.
- **Regression coverage asserts real golden-set values, not just "compiles without error"** —
  cross-checked every named test (`nlq-08/02/21/25/26/16/17/20/31`) against
  `nlq_golden_set.jsonl`'s actual `expected` field; all match, including `nlq-25`'s deliberate,
  explained departure (asserts row count/structure, not the golden *name* values, since the
  entityId-vs-name gap is category D's job, not this fix's).
- **The structural no-write-keyword guarantee holds** — `compile()`'s template strings still
  contain none of the forbidden keywords, every value still binds as a parameter, and
  `run_readonly_query` still only calls `.ro_query(...)`, confirmed by the existing AST-based
  static regression tests, which still pass.
- Mutation-testing the "reject the whole branch" shape (disabling `needs_tuple_distinct`
  entirely) **is** caught, decisively, by 4 existing tests — the coverage gap above is narrower
  than "no mutation testing happened here," it's specifically the multi-column re-aliasing path.

## Open questions

- Should the duplicate-`returns` uniqueness check (MAJOR #2) be a blocking fix before any further
  live demo use of `query_graph_data`, or is the golden-set re-run (which doesn't appear to
  exercise this shape, per the RCA's probes) an acceptable interim signal while it's queued as a
  fast follow-up? This review's verdict treats it as needs-changes-before-close, not as blocking
  the re-run itself — but that's a judgment call the coordinator may want to confirm.
