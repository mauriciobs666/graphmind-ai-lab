# Change History — decision-models

> Dated log of actual changes to the `decision-models` component. Most recent first.

## 2026-11-01 — typed-decision-models U6.6 — dataset contract vector copy

**What:** `docs/plans/typed-decision-models6.md` v1.5 §3.7 and §4's U6.6 row:
`tests/fixtures/decision-contract/dataset-shape.jsonl` is a byte-copy of model-bench's canonical
dataset contract vector (`generation.json`, items and `splits.json` per role), which U5.10 checks
`finalize`'s output against. `tests/test_decision_contract.py` adds it to the guarded equality
test and the row-count test, and checks each of its items against this package's points
(`input_field_errors`, `expected` in the answer space, `labelVotes` keys the answer space as JSON
strings).

**Verified:** `pytest -q` from `decision-models/`: 671 passed, 2 xfailed (baseline 666/2); `ruff
check .` clean. Deleting `rngSeed` from the canonical file without re-copying reddens the equality
test (1 failed); re-copying turns it green; both files restored `cmp`-equal.

## 2026-11-01 — typed-decision-models U5.5 — transcript audit

**What:** `docs/plans/typed-decision-models6.md` v1.5 §3.3 (guard bounds), §3.4 (audit rows,
precedence, in-flight rule) and §5 U5.5, over `docs/plans/typed-decision-models4.md` §3.3's location
rule and remaining rows and §3.7's `runs[]` record and token rule; plus impl review Pass 8's
P8-F3–F6 and Pass 9's P9-F1–F4 (fix round U5.5b) (`docs/reviews/typed-decision-models4-impl.md`).
P8-F1/F2 are settled by plan6 (foreground only; one final `SubagentHandback` is a reply).

- `decisionmodels/audit.py`: `audit_run(driver_transcript, Expected, *, earlier_no_attempt, launch,
  interrupted=False)` locates the run through the driver's `Agent` `tool_use` and `.meta.json`
  `toolUseId` and returns an `Audit`: every matching row as a `Finding` in table order, the chosen
  class, and `RunFacts`. `Expected.guard_command` is the declared guard command (P8-F4). `Launch` is
  keyword-only and `fail_bound` (the `launchFailBound` recorded at `init`) has no default, nor has
  `launch` (P9-F4).
- **Rows shipped:** `safeguardStop`, `backgroundSpawn`, `guardRejected` (a guard deny or a blocked
  guard failure, keyed on `permissionDecision` `reject`/`hook`), `guardBreach` (an allowed call
  outside the guard's bounds, recomputed with the audit's own `realpath` canonicalization),
  `inputInvalid` (counted: no decision, `is_error`, no attachment; a hand-back too, P9-F3),
  `hookMissing` (any other decision, incl. D3's `accept`/`config`), `driverModelMismatch`,
  `syntheticError` (either transcript), `rateLimit`, `spawnShapeUnknown` (a non-`foreground` meta
  even in flight, P9-F1), `incomplete`, `neverSpawned`, `launchFailed` (uncapped) and
  `launchFailing` (past `Launch.fail_bound`), then plan4's location, driver, injected-context and
  counted rows (`maxTurns`, `toolOutOfBounds` for hand-back misuse, `pathOutOfBounds`). `hookDenied`
  is removed.
- **Precedence:** halts, usage stop, `spawnShapeUnknown`, re-issues, counted; table order within a
  tier, where the halt rows' order picks the reason (P9-F2). In-flight tool uses (no result while
  the spawn has none) are read by no row, so a killed run is `incomplete`. An `interrupted`
  incomplete is not capped. A final line with no newline that does not parse is truncation (P8-F5).
- **Fixtures** (`tests/fixtures/transcripts/`): the clean run moved to H1's foreground shape (driver
  guard entry, `completed` result, `requestShape`, a Sonnet driver); `guard-decisions.jsonl` holds
  the U6.7 D1/D2/D3 `tool_result` entries and attempt 1's validation error in their recorded key
  sets (probe report §10.3, §7), ids and paths replaced. `tests/test_audit.py`: 174 tests.
  `README.md` line 45 reflowed (P8-F6); `AGENTS.md`.

**Verified:** `pytest -q` from `decision-models/`, no deselection: 666 passed, 2 xfailed (baseline
568/2); `ruff check .` clean. Mutants ran in a private copy, each restored `cmp`-equal;
failures per mutant: `guardRejected` counted 4; deny counted and failure halted, told apart by text
(the rejected alternative) 3; `hookMissing` reading in-flight tool uses 20; in-flight ignoring the
driver's result 1; `async_launched` as completion 1; `finished` not required 2; `spawnShapeUnknown`
first 1; counted tied with the usage stop (A6) 2; counted tied with halt (A1) 4; table order only
(A2) 5; re-issue tied with the usage stop 1; `safeguardStop` counted 6; refusal marker ignored 4;
`refusedMessageId` ignored 2; `guardBreach` counted 23; `normpath` for `realpath` 3; prefix without
separator 6; driver bound unchecked 1; result matched ignoring `tool_use_id` (P1) 2; any command as
the guard 1; any decision as the guard's 1; any hook event 2; `inputInvalid` despite an attachment
1; hand-back not exempt 2; hand-back count unchecked 2; hand-back not a terminal reply 4; driver
model unchecked 2; driver `<synthetic>` ignored 2; `launchFailed` capped 1; launch clock never
resets 1; launch bound `>=` (M9) 1; empty driver transcript as missing (M17) 1; in-flight meta
unchecked 2; first four halt rows reversed 2; hand-back exempt before `inputInvalid` 1; non-zero
exit as never spawned 1; `interrupted` ignored 1; no truncation tolerance 2; any
unparseable final line as truncation 1; empty transcript skipped 1. Accepting `async_launched` in
the shape check survives (0): equivalent, as the background row is tested first.

## 2026-11-01 — typed-decision-models U5.2b-hash — leakage key for join-in and message type

**What:** `docs/plans/typed-decision-models5.md` §3.4's U5.2b-hash (the hashing half of
`docs/plans/typed-decision-models4.md` §4.1's U5.2b; formula v1.3 §3.5).

- `decisionmodels/hashing.py`: `INPUT_FIELDS_BY_ROLE` gains `join-in-judge` (`agentName`,
  `priorTurns`, `newMessage`) and `message-type-classifier` (`text`, `priorTurn`), equal to
  model-bench's `decisions.INPUT_FIELDS_BY_ROLE` at `1554d560`. An opening join-in's
  `priorTurns: []` and a `priorTurn: null` hash as present fields.
- `tests/fixtures/decision-contract/input-hash.jsonl`: re-copied from model-bench (6 new join-in
  and message-type rows), `cmp`-equal; every row recomputes through `input_hash`.
- `tests/test_decision_contract.py`: `test_input_fields_by_role_equals_model_benchs` reads
  model-bench's row literal by AST (never imported; skipped when `model-bench/` is absent) and
  compares `{role: sorted(fields)}`: order-free, since `canonical_input_json` sorts keys, but
  duplicate-sensitive. The reader requires exactly one top-level binding of the name (assignment
  or import), pinned by five planted second bindings. The unknown-role refusal now uses a role no
  table defines.
- `tests/test_points.py`: every point's `inputFields` equals `hashing`'s row as a set. P3-F6's
  two strict xfails (which XPASSed once the rows landed), `PLANNED_INPUT_FIELDS` and its `.get`
  fallback are deleted. `AGENTS.md`, `README.md`.

**Verified:** `pytest -q` from `decision-models/`, no deselection: 492 passed, 2 xfailed (the
U5.2b-render renderer xfails); `ruff check .` clean; `tests/test_fr8_isolation.py` 24 passed.
Mutants ran in a private dir, each file restored `cmp`-equal and sha256-checked afterwards. A
join-in field dropped reddened 5; `priorTurn` moved to join-in 6; an extra join-in field 2; a
duplicated join-in field 1; one copied digest edited 2; a null `priorTurn` omitted like an absent
field 1; the reader taking the first binding 5, ignoring import bindings 2. A hash-neutral
reorder of the join-in row stays green (red, 1, if the model-bench comparison reverts to exact
order). Comparing as sets misses the duplicate (0), as does the point conformance test, which is
set-based by plan5 §3.4; as lists, the point test falsely reddens on a reorder. Review:
`docs/reviews/typed-decision-models4-impl.md` Pass 7 (P7-F1–F3 applied).

## 2026-11-01 — typed-decision-models U5.4 — exchange and ingester

**What:** `docs/plans/typed-decision-models4.md` §4.1's U5.4 — §3.2 in full, §3.3's answer rows,
idempotence and second-answer refusal — with plan5 (`typed-decision-models5.md`) §3.4's id-blind
seed lines, under note `docs/plans/typed-decision-models-ml.md` §Q1.2 "Blind"; review
`docs/reviews/typed-decision-models4-impl.md` Pass 5.

- `decisionmodels/exchange.py`: `build_label_request` / `build_generation_request` (header,
  line 2, then one item or seed per line, at least one; keys of 10 base32 characters from
  `secrets`, redrawn on a collision with the caller-held set or this request's keys; a fresh
  opaque name per request), `write_request` (validates, never overwrites), `request_errors` /
  `validate_request` / `parse_request` (every line a closed key set, a repeated JSON name refused;
  any `*Id` key on a generation request refused, whatever `points/` declares), `body_sha256`
  (layout-free) and `cut_pages` (greedy pages ≤ 40,000 written characters, newline included, cut
  at `P_MAX` 10 / 2 pages with the header page; the rest is `overflow`; a page's limit is at least
  1; a line that cannot be parsed — bad JSON, too deep, an integer past Python's digit limit —
  is a refusal, never an exception). Seed lines are `Seed.axes` with the `*Id` keys projected out; `BuiltRequest.keys` keeps the
  whole seed. Raw U+0085/U+2028/U+2029 are written escaped and refused raw, so no reader splits a
  line. `RENDERERS` holds the guard renderer only.
- `decisionmodels/ingest.py`: `read_answer_parts` (whole file or parts 1…k, no gap),
  `ingest_answer` / `ingest_answer_files` → `Ingested`, `AlreadyIngested` (same answer bytes),
  `Void` (run-attributable, one reason by a fixed order plus every detail) or `AnswerRefused`
  (a second, different answer). A line with a non-string key, a repeated name or nesting too deep
  to parse is a `malformed` void, never an exception, so reconciling the batch again cannot crash
  (P5-F1). The answer space is type-strict (`1` is not `true`); generated items are checked by
  `points.input_field_errors` only (plan5 §3.5: screen and bindings are U5.8's). A CRLF line ending
  reads as LF, so a `\r`-only line is a blank line, `malformed`, as in an LF answer (P6-F2).
- `tests/test_exchange.py`, `tests/test_ingest.py`: the §5 U5.4 rows — the plan4 positive
  controls (`split` on the header, `seedId` on the rubric line, `intent` on an item line), plan5's
  (`domainId` on a generation line; no `*Id` key in a writer's seed line, all three points), the
  budget on 50 pt-BR guard items at the render cap, an independent `bodySha256` recomputation and
  a layout change, a two-part answer, each void reason, and idempotence against a stand-in ledger.
  Scripted-`secrets` tests pin held-key avoidance and uniqueness within a request; a part with no
  final newline hashes as the whole answer; a CRLF answer ingests. Strict xfails for the two
  renderers U5.2b-render adds. `AGENTS.md`, `README.md`.
- **Observed paging:** the guard rubric line is 5,099 characters and a pt-BR item at the render cap
  6,331 written characters, so 50 such items take exactly 10 pages (the first content page holds
  the rubric and 5 items); a 70-item batch is cut at 53.

**Verified:** `pytest -q` from `decision-models/`, with the U5.2b-hash node
`test_the_vector_copy_is_byte_identical_to_model_benchs[input-hash.jsonl]` deselected: 480 passed,
1 deselected, 4 xfailed. `ruff check .` clean; `tests/test_fr8_isolation.py` 24 passed. A
sentinel and 36 mutants ran in a private copy over the two U5.4 test files, each restored
`cmp`-equal, with both mutated files sha256-equal to the live tree afterwards. The sentinel
reddened 13 tests, and every mutant reddened at least one: `*Id` projection removed 13;
`validate_request` accepting `split` 2, `seedId` 4, `intent` 1, `domainId` 1; `*Id` rule removed
5; idempotence broken 3; second-answer refusal removed 2; page budget off by one 1–2 in each of
five forms; non-string key check removed 6; repeated-name hook removed 3; held keys ignored 1;
this request's keys ignored 1; no-final-newline hashing removed 1. After review Pass 6, in a
second private copy: the `\r` strip removed 1; the request-side `ValueError` catch narrowed to
`JSONDecodeError` 1; the request-side `RecursionError` catch disabled 1. The request-side
`offset < 1` check was deleted: the coverage check already refuses any such page.

## 2026-11-01 — typed-decision-models U5.3 — points, rubrics, seeds

**What:** `docs/plans/typed-decision-models4.md` §4.1's U5.3, with plan5
(`typed-decision-models5.md`) §3.4–§3.5, under note `docs/plans/typed-decision-models-ml.md`
v1.11 §Q1.1–§Q1.5; reviews `docs/reviews/typed-decision-models4-impl.md` Passes 3–4 and
`docs/reviews/typed-decision-models-ml.md` Passes 1–2.

- `points/<role>/` for `guard-judge`, `message-type-classifier`, `join-in-judge`: `point.json`
  (answer kind, key and `expected` per v1.3 §3.2 and FR-10; input-field schema with v1.3 §3.5 /
  plan2 §3.1 field names; seed bindings; seed design and weights; note §Q1.3 cells and sizes;
  screen token lists), `rubric.md` (each answer, then boundary cases) and `generator.md` (no
  commentary, no `*Id` field). `points/common/` holds the two answer-file sections.
- Note v1.10–v1.11 design: guard `missingForm` drawn given (answer, difficulty) on
  `understanding` seeds, with per-item obedience as an `emptinessAxis` binding; join-in `cue`
  drawn given the answer, cue_05 narrowed; message-type rulings for off-topic questions,
  cancelling and returns; join-in small-talk ruling; the assistant speaks as `Assistant` /
  `<business> Assistant`; the guard generator adapts the situation, never the condition or the
  business. Generator text per ML Pass 2: the `ambiguous` rule ends "`missing` is `[]`"; a guard
  `understanding` near-miss follows `missingForm`; a message-type follow-up may stand alone.
- `decisionmodels/seeds.py`: `draw_seeds` — hierarchical stratified axes, balanced axes, and
  conditional axes over weighted multi-axis cases, every allocation by unbiased systematic
  rounding with an integer start (docstring); non-word ids; the record carries `pythonVersion`
  (reproduction holds on the same minor version).
- `decisionmodels/points.py`: `load_point`, the composed line-2 texts, `input_field_errors`,
  `seed_binding_errors` returning `BindingError(binding, message)` for U5.8's classification. The
  `missingForm` binding is silent when `understanding` is not a non-empty mapping, so a wrong-kind
  item reports only its `evidenceKind` errors (P4-F1).
- `tests/test_seeds.py`, `tests/test_points.py`: `allocate`'s expectation exact over evenly
  spaced starts, and constant within each integer start's interval; draw means over 1,500 RNG
  seeds at sizes 1–7 within 4 standard errors; restricted round-trip through the record;
  collision and conditional-overlap guards; the wrong-kind item; a text pin tying each form's
  "`missing` is `[]`" to its binding; strict xfails for the two roles `hashing.py` lacks until
  U5.2b (P3-F6); side-check gate fails rather than skips when the pack moves (P3-F5).
  `AGENTS.md`, `README.md`.
- **Digest pins moved**, deliberately: all three re-pinned (guard `341d8815…`, message type
  `299beb51…`, join-in `26109599…`). Message type's `point.json` is unchanged, but systematic
  rounding changes every allocation, so its pin moved too. The integer start left all three
  unchanged: it gave identical shares on 100,000 random (weights, total, RNG seed) cases.

**Verified:** `pytest -q` from `decision-models/` 364 passed, 2 xfailed, and 1 failed:
`test_the_vector_copy_is_byte_identical_to_model_benchs[input-hash.jsonl]`, because
model-bench's canonical `input-hash.jsonl` gained six rows in its working tree (U4 in flight);
this unit changed neither the copy nor the test. Deselected by that node id: 364 passed,
2 xfailed. `ruff check .` clean. Eleven mutants in a private copy, each restored `cmp`-equal and
sha256-checked: largest remainder 84 failed; start ×0.97 21, ×0.95 35, ×0.999 9, ×1.001 86;
`missingForm` guard removed 6; overlap check removed 1; the `ambiguous` rule's "`missing` is
`[]`" removed 1; undefined-form error removed 1; the guard over-silencing 7; the pre-integer
float start restored 0 (equivalent, as intended).

## 2026-10-31 — typed-decision-models U5.2 — guard contract copies

**What:** `docs/plans/typed-decision-models4.md` §4.1's U5.2, under v1.3 §3.2, §3.2.2 and §3.5.

- `decisionmodels/render.py`: `render_guard_input(condition, understanding, turns)` — the guard
  judge user message (CONDITION, then CURRENT STATE iff `understanding` is non-empty, then RECENT
  TURNS; 6,000-char cap dropping oldest turns first, base hard-truncated when no turn fits);
  `normalize_turns` (filter-then-last-6, 400-char text, `displayName`/`authorId`/`member`
  speaker); `render_guard_item(item)` composing the two (v1.3 §3.2's `normalized_turns`).
- `decisionmodels/hashing.py`: `INPUT_FIELDS_BY_ROLE` (`guard-judge` only; the other roles come
  with U5.2b), `canonical_input_json`, `input_hash` (v1.3 §3.5's formula; an unknown role is a
  `ValueError`).
- `tests/fixtures/decision-contract/render-guard.jsonl` (14 rows) and `input-hash.jsonl` (4 rows):
  byte-copies of model-bench's canonical files, `cmp`-equal.
- `tests/test_decision_contract.py`: every vector row against the implementation; the guarded
  byte-equality test per copy (skips only when `model-bench/` is absent); a row-count pin per file
  counted from raw non-blank lines; hand-derived tests for turn normalization and
  `render_guard_item` (including a ten-row turn-order and cap case, Pass 2 P2-F1), which no
  cross-component vector covers.
- `AGENTS.md` (current state; the byte-copy invariant) and `README.md` (status).

**Verified:** `pytest -q` from `decision-models/` 54 passed (26 → 54); `ruff check .` clean.
Twenty-one mutants each redden at least one test, re-run from a private driver after a failed
restore was recovered (`render.py` md5 `2ce93c54…`); the restore failed because a parallel agent and
this unit both wrote `scratchpad/mutate.py`. Independent re-sample:
`docs/reviews/typed-decision-models4-impl.md` Pass 2.

## 2026-10-31 — typed-decision-models U5.1 — component scaffold and the FR-8 isolation test

**What:** `docs/plans/typed-decision-models4.md` §4.1's U5.1 (v1.3 §4 U5's skeleton bullet and its
FR-8 test; v1.3's `labeler.py` is void and not scaffolded).

- `setup.sh` (model-bench's, renamed), `pyproject.toml` (no runtime dependencies; dev extra
  `pytest`, `ruff` at model-bench's pins; `pythonpath = ["."]`), `.gitignore`, `README.md`,
  `AGENTS.md`, and the empty `decisionmodels/` package with `__version__`.
- `tests/test_fr8_isolation.py`: an AST import allowlist (stdlib + `decisionmodels`, relative
  imports; `__import__`/`import_module` count, a non-literal module name is refused) and a scan of
  non-docstring string literals for `falkor-chat`/`falkorchat`/`falkor_chat`, `falkordb`,
  `redis` and `guard-judge-understanding`, plus the integer 6379, over `decisionmodels/` only
  (plan2 §3.5). Fourteen planted-red cases and a negative control.
- `tests/test_package.py`: `__version__` equals `pyproject.toml`'s; `dependencies` is empty.
- Root `AGENTS.md`: one Structure bullet and one Component-docs row.

**Verified:** `pytest -q` from `decision-models/` 20 passed; `ruff check .` clean. Planting
`import redis` (and, separately, `import falkorchat`, `from modelbench import packs`, a
`falkor-chat` path, the side-check items path and `6379`) in `decisionmodels/__init__.py`
reddened `test_decisionmodels_reaches_only_stdlib_and_itself`; the file was restored `cmp`-equal
each time.
- Review Pass 1 fixes (`docs/reviews/typed-decision-models4-impl.md` F1–F3, F5, F6): a string-literal planted case per fragment (dropping any one fragment reddens the suite), word-bounded `redis` with "predisposition"/"redistribuir" in the negative control, bytes literals and `from importlib import import_module as …` aliases scanned, the docstring's list of what the tripwire cannot see, `AGENTS.md`'s side-check-reader wording, and the root bullet naming the requirements doc; 26 passed.
