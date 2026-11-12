# Review: P1-21 reader layer and anchors (uncommitted) — `cpg/ccg/engine/`

> **Status:** active · **Owner:** `analyst` · **Tracks:** C-1001 (P1-21)

## Scope & verdict

Reviewed the untracked `cpg/ccg/engine/{__init__,config,reader,anchors}.py`, `tests/test_engine_{reader,anchors,reader_live}.py`, `tests/engine_helpers.py` and the `cpg/README.md` `ccg/engine/` row against `docs/plans/connected-code-graph-p1-graph.md` §5/§7/§8, `docs/plans/connected-code-graph-p1-answers.md` P1-21, `docs/plans/connected-code-graph-p0-graph.md` §1.3-§1.4, and the linker's matchers (`ccg/linker/matchers.py`, `ccg/packrt/match.py`, `packs/*/pack.yaml`). Root `REVIEW.md` applied (no `cpg/REVIEW.md`). Not reviewed: the `graph-dba` query-text review the plan also asks for (this gate checked texts character-for-character, not their plans/perf beyond the live EXPLAIN test); the `cpg/sandbox/ac8` work.

Run: unit tier `201 passed` (reader + anchors, `-m "not live"`); live tier `74 passed` on my own throwaway (`ccg-falkordb-tw-16390`, scratch `CCG_CONFIG_DIR`, `CCG_REQUIRE_LIVE=1`; removed by name afterwards, none of the shared instances touched). 29 mutants run in a unique scratch copy: 25 killed, 4 survived (F2, F3).

**Verdict: approve with suggestions — one major (F1) to fix before P1-26 consumes `resolve_contract`; it does not touch the reader.**

CPG: considered, not relevant — new uncommitted module, read in full; no loaded `cpg_*` graph indexes it.

## Findings

### F1 (major, verified) — `resolve_contract` applies `prefix` to every kind, so it anchors on contract points the linker would never link
`anchors.py:_CONTRACT_MATCHERS = ("path-template", "prefix")` is tried for every `kind`, but the linker uses the *pack's* matcher per kind: shipped packs declare `http: path-template`, `mcp_tool: exact`, `topic: exact` (`packs/*/pack.yaml` `link:`). Ran it: with stored `GET /files`, `resolve_contract(r, "GET /files.json", kind="http")` returns **resolved via `prefix`** (`text_hit` falls back to the `.` separator because the key does not start with `/`; `match_keys("http","prefix","GET /files","GET /files.json")` matches, `"GET /files/x"` does not). Likewise `tool` ⇒ `tool.sub` for `mcp_tool`. The docstring promises "the contract point this key links to"; here it returns one that no link can exist for, and the unit test `test_a_prefix_provider_serves_a_longer_key_by_the_linker_prefix_matcher` plus live `via == "prefix"` pin the wrong behaviour. Effect: a spurious or spuriously-ambiguous anchor (e.g. `cp5`+`cp6` in the unit test) turns a clean question into an `ambiguous` answer or a wrong walk start. Answer to the delegate's question: no, using the matchers as-is is not right for `http`.
Fix: choose matchers per kind, not a fixed tuple — `http` → `path-template` only; other kinds `exact` only until the kind's pack matcher is available to the engine (it is not read today; if P1-26 wants prefix for a kind, say which kind and from which pack row). Tests: `GET /files.json` against stored `GET /files` is `unresolved`; `mcp_tool` `tool.sub` against `tool` is `unresolved`; keep `GET /a/{}` ⇒ `GET /a/b`.
Direction (stored = provider, asked = consumer) matches the linker's `match_keys(kind, matcher, provider_key, consumer_key)` call and is fine.

### F2 (minor, verified by mutation) — location rule: region width and region-vs-span ranking are under-tested
Surviving mutants in `anchors.py:_pick_by_line`: region width `(b - a + 1, 1)` (instead of `b - a`) and `min(spans, key=(kind, width))` both leave 201 green. The tests only put a line in a region *or* a span, never a symbol whose line is in both, and never a region whose width differs from a rival span's by exactly one. Add: (a) a region `[30,34]` (width 4) vs a rival span `[30,35]` (5) at line 32 → the region symbol wins; (b) a symbol whose span and own region both contain the line, with a narrower region → anchored `via="owned-region"`. Axes to probe (not a list of shapes): width delta in {-1,0,+1} × {span,region} × {own, rival} × boundary line {start,end}.

### F3 (minor, verified by mutation) — nothing pins "an error message names no graph data" (focus 4)
Reading finds the messages clean (static strings; only caller-supplied, validated values appear: `unknown repo`, `unknown query`). But a mutant that formats the offending row into `GraphDataError` (`... malformed row {row!r}`) survives all 201 tests, and `_once` chains `from exc`, keeping the server's error text in the traceback. Add one test per failure family (malformed row, server error, `_cell` type, `bound_of_link`) that plants a sentinel in the row / exception text and asserts it is in neither `str(err)` nor `repr(err)`; decide whether `from exc` should become `from None` for the generic failure (the timeout and TIMEOUT_MAX branches are safe to chain).

### F4 (minor, inferred+checked) — `RESULTSET_SIZE` is an unverified assumption with no infra check; `TIMEOUT_MAX` is sound
`TIMEOUT_MAX`: the role cannot `GRAPH.CONFIG GET` (§7), and the constant equals the pin. Live: lowering `TIMEOUT_MAX` to 3000 makes every read raise `ReaderError(... TIMEOUT_MAX)`, not a timeout and not a silent miss — fail-closed, and `for_system(timeout_ms=...)` is the knob. Sound.
`RESULTSET_SIZE`: the image default is 10000 (read live from the throwaway; `up` does not set it). A server configured *lower* returns a cut reply of N < 10000 rows that the reader accepts as complete (silent truncation, the one thing P1-21 forbids). The live test asserts equality only on the throwaway. Fix: have `ccg-falkordb.sh status` report/compare `RESULTSET_SIZE` as it does `TIMEOUT_MAX` (devops row), and say in `config.py` that the constant is the *image default*, not something the instance is made to hold.

### F5 (minor) — `resolve_symbol_by_name(repo=...)` skips the repo check the other repo-scoped resolvers make
`resolve_symbol`, `resolve_location`, `resolve_config` raise `UnknownRepo`; `resolve_symbol_by_name` with an unknown repo returns `unresolved`/`unknown-name`. Pick one (call `_ensure_repo` when `repo` is given). Also `path or ""` fabricates an empty path for a `None` path; keep it `None` or document it.

### F6 (nit) — `name_norm` is a second copy of the bundle's rule
`anchors.name_norm` equals `ccg/bundle.py:1002-1004` (`NFC(casefold(NFC(name))).strip()`; verified by reading, and the unit tests pin two ordering cases). Two copies can drift; add one test that runs the function's output through the bundle's own `nameNorm` validator, or move the rule to one importable home. The live fixture's `nameNorm = name.casefold()` would not catch a divergence on non-ASCII.

## Rulings on the delegate's plan gaps (focus 5)
1. **Unknown repo is raised after one read (`repo_exists`)** — accept. A repo name cannot be validated without a read; plan behaviour (6)'s "before any query" can only hold for the system name (done in the constructor) and for input validation (done, tested with `calls == []`). Plan wording should say "before any *location* read".
2. **Chunk-halving reconciliation** — accept as built: a reply of `>= RESULTSET_SIZE` is discarded, the chunk halves (and stays halved), and one id (or an unchunked read) raises `ResultTooLarge`. Never truncates; the mutants `>` for `>=`, off-by-one halving and chunk slice are all killed.
3. **`resolve_location` with neither line nor symbol** — accept: `unresolved`, `state="mapped"`, `detail="no-line-or-symbol"`, no symbol read. P1-26 must key off `state`, not `status`, so a mapped file is not reported as "not found".

## Focus checks with results
- **(1) Texts vs §5:** all 48 entries of `QUERIES` equal §5 after whitespace normalisation (script over the plan's tables and code blocks; the only differences are trailing annotations such as `($pairs = [linkId, siteId])`). The templated rows (`owner_of`, `data_in/out`, `controlled_out`) were compared by reading. The unit pin and live `test_the_texts_are_the_plans...` also read the document. Read-only: `ro_query` only (a fake `query` raises in the helper), lint at import and at every read, closed name table, params-only with shape checks, hostile `qname`/`repo`/`path` tested unit and live (graph counts unchanged). The `ccg_engine` role holds no write grant (live-tested).
- **(2)** see F4; chunk constants killed by mutation.
- **(3)** tie-breaking matches `-p0-graph.md` §1.4 (narrowest; span over region; larger `depth`; remaining tie → `ambiguous`, options capped, sorted by id); multi-`File` row → `GraphDataError` (mutant killed); `name_norm` per F6; contract direction per F1.
- **(6) Mutants** (scratch copy `scratchpad/mut_an21_*`, deleted): killed — row sort (both paths), chunk/halving/`>=`, `bound_of_link` last-row pick and launch filter, timeout not sent, region/span end and start inclusivity, depth sign, kind ignored, ambiguity-picks-first, options cap, multi-file, inner NFC, prefix removal, unsorted hits, repo-check removal, launch entries as anchors, option order, repo filter, http normalisation. Survived: F2 (2), matcher order (equivalent apart from the `via` label), and a sort-key mutant that is equivalent (all hits share one qname).
- **(7) Seams:** P1-22 (`count(x)` interpretation, no sorting in the walk), P1-23 (`flowAvailable` returned raw), P1-39 (`snapshot`/`units` raw, no stamp) and P1-26 are named in docstrings and built nowhere. `LEVEL_GUARD` and `DEFAULT_DEPTH_BOUND` are defined but unused until P1-22 (nit; stated). The README row names P1-22/23/39 but not P1-26 (nit).

## What's solid
Closed query table plus a two-time lint is a strong read-only design, and the live suite (EXPLAIN index seeks with a positive control, role refusals, cross-system isolation, TIMEOUT_MAX lowered, 10,005-row overflow) tests real server behaviour. The tests are able to fail: 25 of 29 mutants were killed, including the sort-order, region-inclusivity and chunk-size arguments chosen for this gate. The deterministic total row order is applied on both read paths.

## Open questions
- F1: should `resolve_contract` ever use `prefix` (for which kind, from which pack)? If P1-26 needs it, the kind→matcher source must be named.

## Appendix
Probe for F1, run from `cpg/`: `ScriptedReader({"cp_by_key": [], "cp_keys": [("c1", "GET /files")]})`, `resolve_contract(r, "GET /files.json", kind="http")` → `status='resolved', via='prefix'`. `match_keys` results: `http GET /files / GET /files.json` → `prefix` match, `path-template` none; `GET /files / GET /files/x` → none for both.
