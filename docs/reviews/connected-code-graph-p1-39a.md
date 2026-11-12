# Review: P1-39a staleness block and stamp (uncommitted) — `cpg/ccg/engine/staleness.py`

> **Status:** active · **Owner:** `analyst` · **Tracks:** C-1001 (P1-39a)

## Scope & verdict

Reviewed the uncommitted `cpg/ccg/engine/staleness.py` (new), `cpg/tests/test_engine_staleness.py` (134 tests) and the one `cpg/README.md` row against `docs/plans/connected-code-graph-p1-answers.md` P1-39 behaviours (1)-(2), `-answer-spec.md` §4.2/§4.3/§6/V9/V11, `-answer-spec2.md` §3.5/§4.3, `-p1-graph.md` §3, requirements FR-7/FR-7a/AC-9/AC-9b, `ccg/answer.py` + `answer-v1.schema.json`, `ccg/staleness.py`, `ccg/refresh.py`, `ccg/assemble/rows.py`, `ccg/engine/reader.py`/`bfs.py`, and the P1-20/P1-21 reviews. Root `REVIEW.md` applied (no `cpg/REVIEW.md`). Not reviewed: the builder (P1-39b), the live tier.

**Verdict: approve with suggestions.** No blocker, no major. The delegate's eight decisions are sound; the three findings are documentation drift (two stale statements in code/README, one spec statement the shipped graph contradicts).

CPG: considered, not relevant — new uncommitted module read in full; no loaded `cpg_*` graph indexes it.

Run (cwd `cpg/`, `ulimit -v 3000000`, `timeout -s KILL`): `tests/test_engine_staleness.py` `134 passed`; `-k "engine and not live"` minus `tests/slice` `925 passed`. Mutation: 24 hand mutants in one unique scratch copy (ccg, tests, packs, infra, pyproject.toml only; `PYTHONPATH` verified to import the copy), 21 killed, 3 survivors, all equivalent (Appendix A). Not run: the live/Docker tier, the slice tier, the full default tier.

## Rulings on the delegate's decisions

1. **No head-behind field — correct, and not a requirement gap (verified against the docs).** FR-7 asks that "how stale the view is must be knowable when it is used"; AC-9 asks for "when the view was last refreshed" and the decision state; FR-7a / AC-9b ask that an answer touching a failed part "says that part is older and why". Per-unit `builtCommit`/`builtAt`, `failedAt`+`reason` and the stamp deliver exactly that. The head comparison is FR-7's *refresh trigger* ("code has moved on too far"), and `ccg/refresh.py:522-525` already consumes `unit_staleness` for it. A head-behind answer field would also need a clone at answer time and break `facts_bytes` determinism (v1 §6: facts are a function of the snapshot). Schema and spec are silent because nothing requires it; no architect routing.
2. **stale-kept / not-mapped / fresh shapes — right.** Matches v1 §4.3 and the assembler: a stale-kept row keeps the last good bundle's `builtCommit` (`rows.py:456-460`), a not-mapped row has only `failReason` (`rows.py:476`). Refusing a fresh row with `failedAt`/`failReason` and a not-mapped row with build fields is the strict reading V9's "status-conditional fields" implies. Narrowing `reason` to `FAIL_REASONS` (v1 §4.3 says "free text") is a subset of the spec and what the assembler enforces (`rows.py:352`); it keeps a value out of the engine.
3. **Commit/time from the unit's own row — correct** (probe + mutant `builtAt<-failed` killed). `:Snapshot.builtAt` is the max over fresh units (`rows.py:495`), so using it would misreport every stale-kept unit.
4. **40-hex — right; the 64-hex in `ccg.staleness`/`ccg.units` is unreachable, not a contradiction to fix here.** `answer-v1.schema.json:771` and `bundle-v1.schema.json:715` (`hex40` for `builtCommit`) reject 64-hex, so no `Unit` row can carry one. A sha256-object-format repo is unsupported end to end today. Time check: regex plus `strptime` refuses `:60`, `02-29`, year 0 (probed). Errors are static ASCII (hostile test includes a bidi/newline plant).
5. **Stamp strictness — right; v1 §4.2 needs a correction (F2).** `-p1-graph.md` §3 says "the engine reads them from the node and fails if they differ", and the assembler already writes both properties (`rows.py:504`). A graph without them (pre-P1/slice) is refused with a static message, acceptable: every P1 graph comes from `build_rows`. `type(offset) is not int` rejecting `False`/`0.0` is correct.
6. **Chain-node selection — right** (traced). V9 (`answer.py:735-758`) is the unit of every item, every hop with a `unit`, and `lifted`/`bound`. V3 gives `unit` to every hop except `ContractPoint` and `Link`; `bfs.py` puts `unit` facts on Symbol, the owner-joined `CallSite`/`Param`/`Ret`/`Cond`, `Unit` (`bfs.py:494`) and `ConfigEntry`, and none on `Link`/`ContractPoint`, so `facts[node]["unit"]` over chain nodes equals V9's hop term; the item's own `unitId` is its last hop's (V7). Using `Walk.units` would add gate-read units (extra) and is correctly avoided. `extra_units` (lifted/bound) and group/folded members are the builder's to pass; a miss fails loud at V9 (missing/extra), never silent (see F4).
7. **Unit hop must be `not-mapped` — right**, same rule as V9 (`answer.py:755-758`), refused earlier with a static message. Mutant killed.
8. **Sort by raw code points — sound for P1-39b.** `ccg.answer.canonicalize` re-sorts `staleness.units` by NFC (`answer.py:56`) and `validate` does not check order, so the engine order is only a stable input. Condition on P1-39b: NFC-normalize `rootPath` before emission and run `canonicalize`; never hash this order.

## Findings

### F1 (minor, verified by grep) — three statements now say staleness and the stamp are not built

`cpg/README.md` bfs/state row (unchanged line: "not built (seams in the docstrings): the answer builder, staleness and the stamp (P1-39)"), `ccg/engine/__init__.py:10` ("the stamp and the staleness derivation are not built here"), `ccg/engine/bfs.py:34` ("SEAM(P1-39): ... `staleness` and the stamp"). The new README row contradicts the first. Suggest: in the same pass as P1-39b, rewrite those three to "builder only (P1-39b)" and name `engine/staleness.py` as where the block and stamp live. `reader.py:20` ("built there") is fine.

### F2 (minor, spec; route to `architect`) — v1 §4.2 and §6 say the engine mints the verdict-log pair, the shipped graph stores it

`-answer-spec.md` §4.2 "Source" ("`:Snapshot` ... has no verdict-log properties ... the engine mints `0` and the constant"), §6 line 318 ("the engine mints the verdict-log pair until P4") and carry-over C8 are superseded by `-p1-graph.md` §3 and `rows.py:504`: the node stores them and the engine reads and verifies. The code is right; the spec text is the stale copy of a fact stated in two places (root `REVIEW.md`, "fact stated in two sections"). Suggest the architect amend §4.2/§6/C8 in place to "stored on `:Snapshot`; the engine refuses a different value before P4". Not a blocker for this unit; fold at P1-39b close.

### F3 (nit, verified) — a 64-hex `builtCommit` is rejected here but accepted by `ccg.staleness`/`ccg.units`

Unreachable today (above). If sha256 repos are ever in scope, `answer-v1.schema.json`, `bundle-v1.schema.json`, `staleness.py` here and `ccg.staleness` change together; a one-line note in the module docstring or backlog would stop the next reader from "fixing" only one. No change needed now.

### F4 (minor, inferred coupling for P1-39b) — nothing here proves group/folded members and lifted/bound units reach `staleness_block`

`staleness_block(reader, items, facts, extra_units)` is correct for the items it is given; its tests pass `w.items()` only. A builder that passes only `Listing.items` (not `groups[].members`) or forgets `extra_units` gets a V9 failure, not a wrong answer. Suggest P1-39b carry one answer-level test per source of a unit (item chain, folded member only, `lifted` only, `bound` only) with the unit on no other path, killed by dropping each source.

## What's solid

- Every unit-row field is checked against the schema shape; mutants on commit pattern (64-hex, `search`), time, reason set, status conditionals, duplicate/extra/missing rows and row-length all die.
- The golden `answer-ac1.json` `staleness` and `stamp` are rebuilt byte-equal from a graph (`test_the_golden_answers_staleness_and_stamp...`, re-run: passes); stale-kept and not-mapped substitutions validate through `answer.validate` (re-run: passes). Reader column order matches `QUERIES["units"]` (`reader.py:79`).
- One staleness vocabulary: `FAIL_REASONS` imported, no second rule; no clone, clock or network touched (suite `test_no_network`-style imports unaffected: 925 pass).

## Open questions

None needing the caller. F2 is the only item with an owner other than the implementer.

## Appendix A — mutation results (scratch copy, 24 mutants)

Killed (21): `builtAt` from `failedAt`; not-mapped allowing build fields / dropping `reason`; `isinstance(offset,int)` (accepts `False`); offset guard, hash guard dropped; 64-hex accepted; `search` instead of `fullmatch`; chain first-node-only; chain last-node-only; Unit-hop check disabled; fresh failure marker allowed; sort by unit id only; free-text reason; missing unit tolerated; unasked-for row tolerated; duplicate row tolerated; snapshot `len(rows) < 1`; time regex without calendar check; `extra_units` dropped; stale-kept `failedAt` unvalidated.

Survivors (3), each judged from the diff:
- *Stamp returns `offset`/`vhash` instead of the constants* — equivalent: the guard just before it forces `offset == 0` (an `int`) and `vhash == EMPTY_VERDICT_LOG_HASH`, so the returned values are equal. Constants in the return are defensible as documentation; no test can distinguish.
- *`read("snapshot", {})` for `read("snapshot")`* — equivalent: `Reader.read(name, params=None)` and `CypherReader.read` normalise `None` through `_check_params`; the test `R` records `dict(params or {})` so both forms log `{}`.
- *`not u.strip()` in `chain_units`* — equivalent in outcome: a whitespace-only unit fact passes `chain_units`, is asked of the `units` read, finds no row and raises `GraphDataError("... has no Unit row")`; the mutant raises one step earlier with a different static message. Facts never carry such a value from the real queries.
