# Connected code graph P1-26 dispatch 1: parsers, resolver, lazy front — implementation review

> **Status:** active · **Owner:** `analyst` · **Tracks:** C-1001 (P1-26, dispatch 1)

## Scope and verdict

Reviewed: the uncommitted tree against commit `e200a8a2` — new `cpg/ccg/front/{__init__,parsers,resolve}.py`, `cpg/tests/front_helpers.py`, `test_front_{parsers,resolve,documents,clients}.py`; modified `cpg/clients/common.py`, `cpg/clients/cli/main.py`, `cpg/tests/test_clients.py`, `cpg/tests/test_no_network.py`. Baselines: plan step P1-26 (`docs/plans/connected-code-graph-p1-answers.md`), `docs/plans/connected-code-graph-answer-spec.md` §3 and `-answer-spec2.md` §2, `-p1-ml.md` §4.2/§4.3, `-p1-graph.md` §5, the coordination ledger. Root `REVIEW.md` read and applied; no `cpg/REVIEW.md` exists. Not reviewed: the interpreter/cache/summary (dispatch 2), the ML method (data-scientist's gate).

CPG: considered, not relevant — the change is new Python in a component with no loaded graph worth querying for it; read directly.

**Verdict: approve with suggestions.** No blocker. One major (CAP behaviour contradicts the spec's "every reading" and strands half a candidate), one major decision (QW-27, the open fork), minors below.

Checks run: the 8-file command from the brief = 1724 passed, 4 deselected, 1 xfailed (matches yours). Probes (run): import-boundary (`import ccg.front` loads no `ccg.broker`, no `asyncio`/`anyio`; `urllib`/`_socket` were already pulled in by `ccg.packrt.httpkey`, not by the front); parser timing on pathological 1k–50k-char inputs (all under 0.2 s); one mutant in a COPY (scratchpad, not in place): `path_suffix_match` `"/" + asked` -> `asked` went red (`test_path_suffix_matches_only_at_a_segment_boundary`), so that test constrains the boundary.

## Findings

### M1 (major, verified) — CAP trims complete readings and splits a candidate across directions

`ccg/front/resolve.py:379` `options()` builds candidate-major readings and slices `[:CAP]`. For 4 symbols x null direction it returns `m0 impact, m0 depends-on, m1 impact, m1 depends-on, m2 impact` — m2 appears with one direction, m3 not at all; `test_front_resolve.py:176-183` pins exactly this. Two problems:
- `-answer-spec.md` §3: for `several-symbols` and `two-use-cases` the options "hold every reading"; only `unknown-name`/`no-anchor` did-you-mean lists carry a pre-registered cap (`-p1-ml.md` §4.3: "did-you-mean list is capped at 5 by construction"). The `capRule` `max(capFloor, |goldOptions|)` is the harness's, so a gold set above 5 (e.g. 3 symbols x 2 unstated directions = 6) can never be matched by the resolver. Plan (3) says "never more than the pre-registered cap", so plan and spec disagree; the resolver follows the plan.
- Half a candidate is an incoherent answer: a reader sees m2 offered for one direction only and infers depends-on is not a reading.

Suggested: apply CAP to whole candidates (`candidates[: max(1, CAP // len(dirs))]`) so every kept candidate carries all its directions, and send the plan-vs-spec disagreement to `architect`/`data-scientist` (is the cap on complete readings pre-registered, and as `max(5, N)`?). Replace the test at :176 with the whole-candidate expectation.

### M2 (major decision, the open fork) — QW-27 partial path: pick (a), narrowly specified

Verified the gap: QW-27 asks `falkorchat/api.py:603`; gold path is `falkor-chat/server/falkorchat/api.py` (`eval/dev/ac24.jsonl:5`); `reader.py:68-69` `file_by_path`/`skipped_by_path` match exact `path`, so the resolver returns `unresolved` + `not-found` (pinned by the non-xfail test at :118). The `-p1-ml.md` §4.2 join rule names suffix matching for `symbol`+`file` and a "partial `file:line`" named test; the plan names QW-27. So (c) amends a requirement-named behaviour, and the user-facing form (a path relative to the cwd/unit, as IDEs and stack frames print it) is the common one.

- (b) is weaker than it looks: unit roots come from `discover_units` over a local clone (filesystem, absent in an MCP/other-host client) or from a `units` read that takes ids (so a new read anyway); it would also read the manifest/clone, not the graph snapshot being asked. It only handles "relative to a unit root", not a deeper partial. Rejected.
- (a) cost, inferred (not run against FalkorDB): `ENDS WITH` cannot use the `File.path` index, so it is a label scan filtered by `repoId`; the same shape as the already-accepted `cp_keys` scan (`reader.py:76`), over a File set that is thousands, not millions. It is a fallback only (run after the exact three-read lookup fails), so the hot path is unchanged. Injection: none if the suffix is a `$param` (never concatenated), as every read in the allow-list is; guard length/NUL with the existing `_text` check. Result cap: `ORDER BY f.path LIMIT CAP+1` (deterministic; +1 only to know whether ambiguity is truncated), then the resolver reads `symbols_of_file` per hit (at most CAP+1 reads).
- Required in (a), or it ships a hole: (i) match at a `/` boundary in the query (`f.path = $p OR f.path ENDS WITH $slash_p`), as `path_suffix_match` does client-side; (ii) a matching `SkippedFile` suffix read too, otherwise a partial path naming a test or ignored file reports `not-found` where FR-7a requires "not mapped" with its reason; (iii) a suffix matching several files in one repo is `ambiguous` `several-symbols`/`no-anchor` with options, never a pick; (iv) spec change is real: `-p1-graph.md` §5 row(s), `test_engine_reader` `EXPECTED`, and the live spec at the feature's close (root `REVIEW.md`: an omitted changed contract is a blocker there).

Pick (a), with (i)-(iv). Then drop the xfail and the :118 test, and keep a negative: `api.py` alone (bare filename, many matches) is `ambiguous`, not resolved.

### m1 (minor, verified by reading) — a group that finds nothing is dropped silently when another group resolves

`resolve_draft` (`resolve.py:397-405`): a `suggestion` group (unknown symbol/contract) or an empty non-suggestion group (mapped file, no element at the line) is ignored when exactly one other group resolves; only failed *file* lookups are surfaced (`lookups`). `a.py:3 Typo.baz` resolves `a.py:3` and says nothing about `Typo.baz`. The spec's no-silent-drop rule is stated for paths only, so this is not a violation; but it is the same defect class FR-7a names. Suggested: when any group names something unfound and another resolves, return `ambiguous` (the user's second name needs an answer) or document the rule in the module docstring and pin it with a test. Also: an `unresolved` outcome with some `suggestions` present falls to `ambiguous` and drops that outcome's `lookups` (`:408`); the mixed case is spec'd for `resolved` only, so state it.

### m2 (minor, decision 1) — parser-resolved answers carry no provenance: agree, document it

Spec `provenance?` is "plain-language asks only", and its required fields (model fingerprint, prompt hash) cannot be honestly filled with no model call, so omitting is right. Consequences to record at the feature's close, not code: (1) `--cache-bypass` on a parsed question is a silent no-op like on a structured one (spec2 §2 says bypass is recorded as `provenance.cacheBypass`; with no provenance nothing records it) — say so in spec2 §2; (2) the P1-35 harness must treat "no `provenance`" as "deterministic path, no model call" rather than KeyError, and AC-24's "cacheHit on repeat 1 stops the run" (`-p1-ml.md` after §4.3) must not read an absent field as a miss. Send (2) to `data-scientist`.

### m3 (minor, verified by running) — the narrowed COMMAND_PROSE can still be bypassed by aliasing; the gap is pre-existing but now relevant

`_reviewed_prose` (`test_no_network.py:200`) exempts only `raise ManifestError("<exact text>")` in `ccg/manifest.py`; the 5 new mutant rows pin the variable, other-exception and other-module cases. I planted, via `scan_source`: `import os; ManifestError = os.system; ... raise ManifestError('git fetch is not enabled in P1: ...')` in `ccg/manifest.py` -> `[]`, though that line runs the text as a shell command. Generic form `run = os.system; run(c)` and `run = subprocess.run; run(c)` also -> `[]` (pre-existing: launchers are judged by call-name, not by reference). Because the exemption trusts the name `ManifestError`, require it to be a class defined in that module (`ast.ClassDef`) or import from it, and add a mutant row for the alias; the broader fix (flag a bare reference to a launcher attribute, not only a call) belongs with the P1-25b scanner owner. Risk is low (deliberate aliasing, inert text), hence minor.

### m4 (minor, decision 5) — draft types in `resolve.py`

Verified the reason: `test_the_future_front_passes_unchanged...` (`test_no_network.py:473-486`) copies the scope, writes six named front files and asserts the module set equals them, so a `draft.py` would fail it. Acceptable, but then `parsers.py:6` cites `ccg.front.draft.Draft`, which does not exist (nit-level stale reference), and the cleaner fix is to add `draft.py` to that test's set now (a one-line change) before dispatch 2 adds `interpret.py` and `cache.py`, which also have to live in that set. Decide once; don't let `resolve.py` become the draft's permanent home by default.

### n1 (nit) — trailing bracket stripped from a route span

`_tokens` strips `}`/`]`/`>` from the token end, so `GET /items/{id}` yields the span `GET /items/{id` (also in the not-built echo). Probed: the stored `GET /items/{}` still resolves for `{id}`, `[id]`, `<int:id>` and `:id` forms, because the template matcher is lenient on a final segment, so no wrong answer — only a mangled span in echoes. Strip a closing bracket only when it has no opener in the token.

### n2 (nit) — `-p1-ml.md` §4.2 says an emptied group makes the answer `unresolved`

See decision 2 below: the ML sentence is superseded by the answer spec; amend it at the next revision (data-scientist) so the two stop disagreeing.

## The delegate's decisions

1. **No provenance on parser-resolved answers — agree** (m2: document, and tell the harness).
2. **No anchor named -> `ambiguous` (`no-anchor`), not `unresolved` — right.** `-answer-spec.md` §3 defines `unresolved` as "every file the asker named failed the path lookup" with non-empty `lookups` (V2), and explicitly allows `ambiguous`/`no-anchor` with `options` `[]`. AC-24 reads gold `no-anchor` as `status != "resolved"`, so scoring is unaffected. The ML sentence is the stale one (n2). `asking_document` output validates (`answer.validate` runs inside it; `test_front_documents` covers ambiguous, unknown-name and unresolved incl. a skipped-file lookup).
3. **CAP can trim complete readings — wrong as built** (M1).
4. **Two resolving anchors -> not-built — agree.** The engine builds one anchor per answer in P1 (structured requests take one `anchor`); the answer is typed and echoes the request. The spec's plural `request.anchors` is only the mixed case, which is implemented.
5. **Draft types in `resolve.py` — acceptable, with m4's cleanup.**

## What's solid

- Parser is token-based with a 512-char token bound; no pathological input found (50k-char probes); the "fully read" rule (every non-anchor word must be in the closed direction/unbuilt sets) is conservative: unknown prose returns `None`, never a guessed draft. Diff and stack-trace surfaces take precedence over their embedded `file:line` anchors (tested).
- Resolver: fabrication guard is byte-exact (`guarded`), path-suffix is `/`-boundary (mutant-killed), null direction expands to both, `did-you-mean` is edit distance then rank key, unresolved/not-found lookups are one per repo and sorted, the typed-error mapping in `Client._ask_front` covers `NotBuiltAsk`, `ReaderError`, `AnswerError` and runs `precheck` before any reader opens.
- Lazy import is real: a structured ask loads no `ccg.front`; a question loads no broker module (subprocess test, and my own probe). `resolve_system` now rejects `""`, `{}`, `0`, `False` as `bad-request`.
- Anchor objects match `answer_build`'s shape (`request_anchor` vs `answer_build.py:112-117`), so options equal what a structured ask would send.

## Open questions

- Plan vs spec on the options cap (M1): who rules whether complete readings are capped at all? (`architect` + `data-scientist`.)
- Is a partial path allowed to match across repos of one system (a suffix present in two repos)? Under (a) it should be `ambiguous` with both; confirm before the read is written.

### Proposed REVIEW.md lines

Target `cpg/REVIEW.md` (does not exist; create) under "Always check", source: this review M1 and m3, plus P1-25b review F7 (carried in the ledger):
- "A resolver or ranker that caps its output is checked for what the cap removes: a complete set of readings is capped per whole candidate, never mid-candidate, and the spec section that says 'every reading' is cited beside the cap."
- "A static scanner's exemption is checked with an alias of the exempted name bound to a launcher (`Name = os.system`) as well as a plain reuse."

## Pass 2 (fix round, uncommitted tree vs `e200a8a2`)

**Verdict: approve with suggestions.** No blocker. Re-ran the 9-file set: 1908 passed, 4 deselected (matches yours). Probes run: a throwaway `falkordb/falkordb:v4.18.11` on port 16477 (own container, removed; `falkordb-dev` and `ccg-falkordb-tw-16390` untouched) with the `ddl_v1.cypher` File/SkippedFile indexes and 300+300 planted rows; resolver and scanner probes via `FakeStore`/`scan_source`; two mutants in a scratchpad COPY.

### Dispositions

- **M1 fixed.** `options()` (`resolve.py`, `keep = max(1, OPTIONS_MAX // len(dirs))`) drops whole candidates only past `config.OPTIONS_MAX` (20); `test_front_resolve.py:197-205` pin 25 candidates x null direction = 20 options and every reading below the bound. The plan-vs-spec disagreement stays owed to `architect` (not the coder's).
- **M2 fixed, (a) with (i)-(iv).** Read texts in `reader.py:80-83` match `-p1-graph.md` §5 v4 character for character (compared the two diffs; `$sep` + `ENDS WITH`, no `OR`, `LIMIT 21` = `OPTIONS_MAX + 1`, `ORDER BY repoId, path`); `repos` param type-checked. Live: 300 rows per label, 21 rows returned and the first is the lowest by `(repoId, path)`; suffix `/pi.py` against `api.py` rows returns 0 (the boundary holds). `_locate` order is right: exact read in every repo first, `mapped` wins, suffix only when every failed state is `not found` (an exact skipped hit suppresses it, as §5 says).
- **m1 fixed, with a new finding N1 (below).** `resolve_draft` now returns `ambiguous` (`unknown-name`/`no-anchor`) over the resolved reading plus the did-you-mean list when another group found nothing; `lookups` still ride only on resolved/unresolved.
- **m2 not code; unchanged.** Still owed at the feature's close (spec2 §2 note; harness treats absent `provenance` as no model call; data-scientist).
- **m3 fixed.** Planted `ManifestError = os.system` + the exempt raise in `ccg/manifest.py` now returns `network-tool-command` and `launcher-alias` (was `[]`); `_binds` demands one class definition and no other binding.
- **m4 fixed.** `ccg/front/draft.py` is stdlib-only, `resolve.py` re-exports, the pinned set in `test_no_network.py` includes `draft.py`, `parsers.py` cites it. `import ccg.front` loads no `ccg.broker`/`asyncio`/`anyio` (probe).
- **n1 fixed.** `GET /items/{id}`, `(GET /items/{id})` and `GET /items/<int:id>.` keep their closing bracket in the span.
- **n2 not fixed, not the coder's.** `-p1-ml.md` is untouched; amendment stays with `data-scientist`.

### Fixes against each other (check 2)

- Suffix vs exact: sound as built (above). The skipped-only suffix result is `unresolved` with `skipped-file` lookups carrying the stored path and reason (FR-7a); one file hit plus a skipped hit resolves and carries the skipped row as a lookup (spec mixed case; the lookup's `path` is the stored path, not the asked one, which discloses more than asked and validates). Several file hits (even one with an element at the line) are never resolved: `ambiguous` `no-anchor`, options only for hits with an element, `[]` for none (`test_front_resolve.py:415`, mutant `several=False` killed).
- Mixed-group rule vs answer-spec2/§3: valid (an `unknown-name` option list may hold readings; the resolved reading is option 1, not lost) and it closes the silent drop. It cannot make a gold *resolved* answer unreachable from the deterministic parsers, which emit only anchor-shaped groups. It can from the interpreter: see N1.

### New findings

**N1 (major for dispatch 2, minor today; verified by running).** `unfound = [f for f in found if not f.lookups and (f.suggestion or not f.candidates)]` counts every group that returns an empty `_Found`, including a group whose spans are only `other` or `server` (names no anchor by definition, `_group` ends `return _Found()`). Probed on a store holding `srv/a.py:3`: draft `[("srv/a.py:3","line")]` resolves; adding `[("foo","other")]` or `[("hello","server")]` gives `ambiguous no-anchor`; adding `README.md` (a file with no config entries) also does. The dev entry check counts over-asking against the interpreter (EP §10.4), and a model that emits one spurious extra group will now turn every resolved gold into an ask. Fix: a group with no `symbol`/`file`/`line`/`route`/`tool` span is ignored, not "unfound"; a plain non-config file named beside another anchor should be a lookup or ignored, not an ask. Add a test per axis (other-only, server-only, non-config file) before the interpreter gate; route the over-asking trade-off to `data-scientist`.

**N2 (minor, verified by mutant).** Mutant in a COPY: `if all(res.state == "not found" ...)` -> `any(...)` in `_locate` survives all 152 front tests, yet changes behaviour: repo `r1` holds `a.py` as a skipped file, repo `r2` holds `x/a.py`; `a.py:3` should be `unresolved` (exact skipped hit wins, `-p1-graph.md` §5) and the mutant returns `resolved` on `r2`'s file (probe, both outcomes run). Add that planted case to `test_front_resolve.py` (one skipped exact hit in one repo plus a suffix-only file in another).

**N3 (minor, verified).** `-p1-graph.md` §8 P1-21 (b) says `skipped_by_suffix` "must show `Node By Label Scan`", and §5 "Plan" calls both label scans. Live `GRAPH.EXPLAIN` here: `file_by_suffix` = `Node By Label Scan | (f:File)`; `skipped_by_suffix` = `Node By Index Scan | (s:SkippedFile)` (the planner seeks the `SkippedFile.repoId` index from the DDL, a Filter on the suffix follows). The coder's report is right and the test (`test_engine_reader_live.py`, asserts Index Scan and no Label Scan) is right; the plan text is stale. Owner `graph-dba`: correct both sentences (the 13 ms figure still bounds it; the index only narrows by repo).

### Scanner `launcher-alias` (check 3)

Real tree and the other modules scan clean (the tree-walk test is in the 1908). False-positive risk is low: it fires only on a launcher attribute of `os`/`subprocess`/`pty`/`asyncio` assigned, defaulted, walrus-bound or imported `as`; annotations and `except` clauses are exempt and tested. It is a bound, not proof: launchers passed as an argument (`h(os.system, c)`), held in a dict, wrapped in `functools.partial`, `getattr`-built, or reached through a module alias (`m = os`) all scan `[]` (run). The earlier text rule still catches a literal command through any of these (`D['a']('git fetch x')` flags), so the residue is a non-literal command, the same limit as before. It is scope beyond m3 but small, tested (7 mutant rows, 1 negative test) and in the file m3 already touched; accept in this unit. Docstring should say "bound, not proof" and list the unflagged forms so nobody reads it as closed.

### Open

- The cap ruling for complete readings (plan vs spec §3) is still `architect`'s; the code now follows the spec's "every reading", bounded by the engine's own `OPTIONS_MAX`.
- Live spec update (the reader reads, `ambiguous` over mixed groups, partial paths) and `-p1-graph.md` §5 are at the feature's close, not this unit.

## Pass 3 (dispatch 2: interpreter, cache, summary; uncommitted tree vs `5eae480e`)

**Verdict: approve with suggestions.** No blocker. Re-ran `tests/test_front_*.py`, `test_clients`, `test_no_network`, `test_broker*`: 1767 passed. Probes run: policy and spec reads, summary heuristic probes, import probe, one mutant in a scratchpad COPY (`grammar_hash()` in the interpreter's cache key replaced by a constant: killed, by `test_an_interpreted_answer_is_the_structured_answer_plus_its_provenance` alone; the second mutant I tried, `RESOLVER_VERSION` -> `"1.0"`, is equivalent and tells nothing). No live LM Studio call was made.

### Dispositions of the checks

1. **Egress, `system.yaml` paths: sound, no spec change needed.** `egress-spec3.md` (example block) says "the `paths` list is P1-26's to fix to what its client calls", so widening is the delegated decision; only the example is stale (update it at close). `policy._r0` checks `path in entry["paths"]` (exact string), R1 allows any call to a `local` entry, and `server._upstream` sends the entry's stored string, never the caller's, so there is no path-confusion, query-string or traversal angle; the GET carries `payloadClass none` and no body (R0 method rule), so nothing leaves. The fingerprint read is required: `quantization` and `loaded_context_length` are provenance fields that only `/api/v0/models` reports. Pinned: `test_front_broker.py:73-97` runs the real policy over the shipped manifest (two exact paths, an unlisted path denied). Residual (minor): the allow-list is path-only, so `POST /api/v0/models` and `GET /v1/chat/completions` are also allowed; harmless against a local metadata/inference server, but say so in the spec2 example rather than imply a method pairing.
2. **Run lifecycle: correct.** `Front.answer` builds a `ModelRun` only when free prose or a summary needs one; the broker run opens inside `_token()` at the first `_send` (the fingerprint GET), is shared, and `run.close()` is in `finally` (also on `NotBuiltAsk`/`ModelError`); a parsed question opens none (`test_front_interpret.py:56`). Absent/unreachable/latched broker -> `BrokerUnavailable`/`BrokerError` -> `ModelError` -> client `model-unavailable`; `client.py` has no direct-connection code. Only a transport failure or 5xx is retried once, a denial or 4xx never.
3. **samplingHash: coder's choice is right.** `answer.sampling_hash(temperature_milli, params)` hashes the temperature (`round(0*1000)`) as a first-class input plus `max_tokens`, `seed`, `top_p_milli` (sorted), so the temperature is in it; schema wins over the ML sentence ("sha256 of canonical JSON"): `data-scientist` should amend that line. Uncovered, minor: LM Studio's own sampler defaults for unsent parameters (`top_k`, `min_p`, `repeat_penalty`, the model's preset) are in neither the hash nor the fingerprint, so a changed preset replays an old cache entry; at temperature 0 only `repeat_penalty`-class settings matter. Either send them in `SAMPLING` or record the gap.
4. **Summary: validate-first and notice-by-code are sound; the entity heuristic is weaker than its docstring.** `answer.validate(doc)` runs before any call; the flow notice is `flow_notice_lines(doc)` appended by code whether the summary is kept or dropped. The model receives only the digest (counts), never a name, so it cannot disclose a real entity; the guard only blocks invented ones. Verified by probe against a resolved document with one anchor: passed unflagged: `Checkout is affected by 2 items.`, `Affects 2 items. Billing breaks.` (a capitalized word at a sentence start), `The login service ...` (a plain lowercase noun), `Two callers, 50% of them.`, `Affects three callers` (spelled-out or percent numbers), and a terminal escape sequence (`\x1b`). Flagged: code-shaped names, a mid-sentence capitalized word, digits not in the digest, a URL. Fix (suggested, not run): make it an allow-list, not a pattern list: keep a sentence only if every word is in the digest vocabulary or a small fixed function-word list, and every digit or number word is a digest value; reject control characters and newlines. Add one planted case per axis above.
5. **Cache: key and bypass are right; the cache does nothing in the shipped entry points.**
   - The key holds all seven inputs; entries live under one snapshot (any other drops them); bypass neither reads nor writes and records `cacheBypass: true` with `cacheHit: false`; an unusable reply is not cached; hits are deep-copied and still go through the fabrication guard (`test_front_interpret.py:209-301`).
   - **Finding C1 (major, verified by reading).** The memo lives on the `Client` (`self._default`), but the MCP server builds a fresh `Client` per call (`clients/mcp/server.py:70`, `client_factory(system).ask(...)`) and the CLI is one process per ask, so no ask ever hits the cache in either path; the cache is exercised only by tests and by a caller that holds a `Client`. This is not wrong (AC-24 needs an empty cache), but the behaviour (4) "a cache entry lives exactly as long as its snapshot" is unobservable. Suggested: keep one `Client` (or one default `Front`) per system in the MCP server so the long-lived process caches; leave the CLI uncached and say so in the CLI help.
   - Persistence: do not persist for P1. A disk cache adds a poisoning surface (a stored reply steers interpretation), snapshot-lifetime bookkeeping and cross-process races for a benefit the CLI path (per-ask process) would only get with it; if it is added later, every entry must be re-validated against `interpret.json` on read (`InterpretCache.get` does not, which is fine in memory only) and written under a per-system 0700 directory.
   - Minor: the in-memory dict is unbounded within one snapshot; add a small cap (e.g. 1024, oldest out) before the MCP server memoizes. The fingerprint GET precedes the cache lookup, so every free-prose ask, hit or not, opens a broker run and makes one GET and fails with `model-unavailable` if LM Studio is down; correct (the fingerprint is a key input), just state it.
6. **Unusable reply: correct.** Not JSON, grammar-invalid, or failing the `Draft` rules gives `ambiguous` `no-anchor` with `options: []`, provenance present, one call, no repair, not cached (`test_front_interpret.py:330-347`, `516`).
7. **Prompt hygiene: passes; one nit.** Examples use `fx-*` names only, four examples, EN and PT, null direction, one qualified group, one unbuilt use case and one two-use-case; no dev question or dev anchor appears; files are `.txt`/`.json` (pinned). Nit: the overlap tests compare words, so the prompt's ASCII `relacoes` and the example `relations` pass against the dev question `relações da tool ...` by accent alone; the neutral cue is effectively tuned to that dev item. `data-scientist` to rule whether the dev entry check tolerates it.
8. **Import boundary: sound.** `import ccg.front` (and `clients.common`) loads `draft`, `parsers`, `errors`, `resolve` and no broker or network module; the broker client loads only inside `model_loader`, at the first free-prose question; `test_no_network` pins the real front module set (8 modules, `errors.py` added), that only `interpret`/`summary` may name `ccg.broker.client` (planted in each other module), and the data files.
9. **Mutation:** killed (above).

### Changed existing probes (no weakening)

`test_clients.py`/`test_front_clients.py` swapped the free-prose input `who calls A?` for `security m.py:5` where the test is about the typed `not-built` error (still asserts `not-built`, still no reader opened), and the free-prose CLI test now asserts `model-unavailable` with an unreachable broker socket. The `plain-language-question` surface pin survives at `test_front_documents.py:134` (a `Front` with no interpreter). The old "free prose is not built" behaviour is intentionally gone from the default client.

### Open

- The summary is reachable only through `Front.answer(summary=True)`; no client or CLI flag calls it, so behaviours (5), (9), (10) are tested at the module, not delivered at a surface; confirm that is intended for dispatch 2.
- `data-scientist`: sampling-hash sentence, the `relations`/`relações` cue, and whether LM Studio default samplers need pinning.

## Pass 4 (dispatch 2 fix round; uncommitted vs `5eae480e`)

**Verdict: approve with suggestions.** No blocker, no new major. Re-ran `tests/test_front_*.py`, `test_clients*`, `test_no_network`, `test_broker*`: 1806 passed. Probes run: summary allow-list adversarial strings, MCP import probe, two mutants in a scratchpad COPY of the prompt (see (4)).

### Dispositions of Pass 3

- **C1 fixed.** `clients/mcp/server.py` `_Clients` keeps one `Client` per system (lock, `MAX_CLIENTS = 8`); `test_front_clients.py` drives the real server through `call_tool`: second identical prose ask `cacheHit` True with one completion, two runs opened and closed (one run per ask, not per client); the CLI help now says a one-process ask never hits the cache.
- **Summary guard fixed.** Probed against a resolved document: `Found 2 items, 1 confirmed and 1 possible.` kept; rejected: zero-width space, Cyrillic-led `Аffects`, Arabic-Indic and fullwidth digits, `\n`, NBSP, combining dot, `50%`, spelled-out `two`, `m py`. The set is closed (letters from the digest's keys and enum values, the singular forms, `FUNCTION_WORDS`, `EXT_LANG` names); digest values are `status`/`direction` enums plus integers, so no entity can enter through a digest value; the notice is built from the document after the check, kept or dropped. Residual, stated in the docstring: a kept sentence can be false (swapped `confirmed`/`possible`); the single-letter word `A` passes as the article.
- **Cache bound fixed** (`MAX_ENTRIES = 1024`, oldest out, one lock). **Fingerprint-GET-per-ask:** unchanged and correct. **Persistence:** unchanged (memory only), agreed.
- **samplingHash sentence fixed** in `docs/plans/connected-code-graph-p1-ml.md`. **LM Studio default samplers (`top_k`, `min_p`, `repeat_penalty`) still not pinned or recorded:** not fixed, minor, carried.
- **Prompt cue fixed** (neutral cue `an overview of` / `o panorama de`; depends-on cue subject-worded), see (4).
- **Open: summary reachable only through `Front.answer(summary=True)`:** unchanged; no client surface calls it. Needs the coordinator's confirmation that behaviours (5), (9), (10) are accepted at module altitude.

### Fixes against each other and the new shared surface (check 1)

- **Leaks between calls: none found.** State a client keeps is `_default` (the front and its cache); every `ask` opens its own reader (`Client._reader()`), its own `ModelRun` (broker run opened lazily, closed in `finally`), and builds its own `provenance` dict, copied on use; `InterpretCache` deep-copies both ways under one lock; the grammar validator is stateless. Eight threads on one shared client each got their own run and answer (`test_concurrent_mcp_asks...`).
- **Bypass under sharing:** neither reads nor writes, and a following normal ask still hits (`test_an_mcp_cache_bypass_ask...`). **Snapshot change:** the cache holds one snapshot and drops on the first read or write under another; correct, and a stale in-flight `put` at an old snapshot can only empty the new one (a miss, never a wrong hit, since the key carries the snapshot).
- **Eviction of an in-use client:** `del` removes the dict entry only; an in-flight call keeps its reference and finishes; the lost state is that client's cache.
- **Minor M1 (verified by reading).** Eviction is insertion order, not least-recently-used, and a tool's `system` is caller-chosen, so nine valid-but-unused system names flush a hot system's cache. Cost: a cache miss, nothing else. Suggest moving a hit client to the end (`pop` then re-insert).
- **Nit N1.** `Client._ask_front` sets `_default` without a lock, so two concurrent first questions can build two fronts and drop one; no wrong answer, one wasted model load. Take the same lock the factory uses.
- **Import boundary:** `import clients.mcp.server` and `build_server()` load no `ccg.broker`/`ccg.front` (probe, and `test_the_mcp_server_still_loads_no_broker_module_until_free_prose`).

### `refuse_unbuilt` now refuses the whole ask (check 3): sound

An unbuilt use case in `readings` is refused with the request echoed, never answered for the built half. Consistent with P1-26 behaviour (7) and with the schema: an option is a `Request` with `useCase: "impact"` only, so no field can say what was dropped. Reachability: the dev set has two ambiguous items (several symbols; one tool x two directions, gold options are both `impact`); none names an unbuilt use case, so no dev or QW item becomes unreachable, and two readings that are both `impact` still expand to both directions (`directions()`). Cost: "code review X and who consumes it" is refused whole where the built half could be answered; the example 4 in the prompt now teaches the model to produce exactly a refused draft. Ask `data-scientist` whether any held-out gold expects the built half (not checkable here); if not, example 4 is still the right unbuilt-plus-built illustration, since the refusal is the spec'd outcome.

### Prompt tests (check 4)

Accent-folded n-gram tests (`fold` strips combining marks). Mutants in a COPY: (a) the field-text cue replaced by `"relations" or "relações"`; (b) example 3's question replaced by one containing `relações`. Both killed by `test_the_neutral_null_direction_cue_appears_in_no_dev_question_even_by_accent_or_case`.

### Edited existing tests (check 5)

`test_front_clients.py` and `test_clients.py` keep the typed `not-built` assertions on an unbuilt use case (`security m.py:5`), the `plain-language-question` pin remains at `test_front_documents.py:134` (a `Front` with no interpreter), and the free-prose CLI test now asserts `model-unavailable` with an unreachable broker socket. The new refusal test (`test_front_resolve.py`) asserts the echoed readings, `built == ["impact"]` and zero reads. Intent kept.
