# Connected code graph — P1 plan, part 3: answer engine, clients, plain language (W0 spec, W3)

> **Status:** active · **Owner:** `architect` · **Tracks:** C-1001 (P1)

Part of `docs/plans/connected-code-graph-p1.md` (decisions D1–D11, the flow gate §5, step index §7). Contracts:
answers `docs/plans/connected-code-graph-answer-spec.md` (canonical answer v1, `ccg/answer.py` is its validator),
traversal `-p0-graph.md` §3 (`bfs-v1`), graph `-graph.md` §6.2, evaluation `-eval-protocol-ml.md` §10. The slice's
walk is `cpg/spikes/vertical-slice/impact.py` (`walk`, `Reader`, `CypherReader`, `Item`, `render_item`), ported
under equivalence tests and then extended. Engine code is Python 3.12 under `cpg/ccg/engine/`; every answer
passes `ccg.answer` validation before it leaves the engine. Briefing line: dev items and fixtures only;
no writes to any shared database.

## P1-00 · answer-spec successor `-answer-spec2.md` — `architect`

`Extends:` `-answer-spec.md` (executed against by S34, so a successor with an ordinal, not an in-place edit);
the earlier gains `Extended by:`. It states **only** the changes:
- the stale text the ledger flagged at S50 (the engine reads/hop wording that predates graph model v1);
- the engine's **`cacheBypass` input** (EP §10.4, recorded in provenance as `cacheBypass`), citing the predicate
  rather than restating it (N-3);
- hop kinds: for each of `DEFINED_BY`, `LAUNCHES`, `DEPENDS_ON`, `FROM_DEP`, either the hop kind and its
  fields or an explicit prohibition (EP §5.2.1's four class-X edge types are carried by no P1 answer hop; `LAUNCHES`
  is the `.mcp.json` launch chain and is needed in P1, the others are P3 and are forbidden until then);
- the `algorithm.version` value and bump rule for P1's rows (C6): P1 ships `bfs-v1` at the version this document
  names (the plan expects `1.1`) and retires `function-level-continuation` with the `arg(i)`/`Ret` rows
  (`ENGINE_REASON_CODES` then drops it);
- the depth-bound default (index D3) and `unexplored` counting (C4);
- §5 of the index: how a `flow-unavailable:<lang>` item is built, so P1-23 has one definition.
**Gate:** `analyst`. The code and schema edits it requires are P1-45's.

## P1-45 · answer validator and schema edits — `tdd-engineer`

- **Files:** `cpg/ccg/answer.py`, `cpg/ccg/schemas/answer-v1.schema.json`, `cpg/ccg/registry/reasons.yaml` (only if
  P1-00 adds a code), `cpg/tests/test_answer.py`, `cpg/tests/golden/answer-ac1.json` (only if a golden changes).
- **Behaviours, failing validator tests first (as P0 did; the pin and the field are not authored in one step):**
  (1) `Provenance` accepts `cacheBypass` (bool) and still rejects every other unknown key
  (`additionalProperties: false`); (2) the hop kinds P1-00 adds validate and the ones it forbids are rejected;
  (3) the validator accepts the new `algorithm.version` value P1-00 names (the test that a row added without a bump
  fails is P1-22's); (4) `ENGINE_REASON_CODES` is unchanged until P1-22 removes
  `function-level-continuation` (one-line edit and its test belong to P1-22); (5) the stored golden still
  validates. **Needs:** 00. **Blocks:** P1-21, P1-22, P1-24, P1-26. **Gate:** `analyst`.

## P1-21 · reader layer and anchors — `tdd-engineer`

- **Files:** `cpg/ccg/engine/__init__.py`, `config.py` (`DEFAULT_DEPTH_BOUND = 2`, `LEVEL_GUARD = 64`,
  `CHUNK = 1000`, `RESULTSET_SIZE = 10000`), `reader.py` (`Reader` protocol, `CypherReader` bound to the
  `ccg_engine` credential, recording every query text for the recipe), `anchors.py`, `cpg/tests/test_engine_reader.py`,
  `cpg/tests/test_engine_anchors.py`.
- **Behaviours:** (1) the reader returns for `Param`/`Ret`/`Cond` their `path`, `repo` and `unit` (the slice's
  reads returned only `pointId`, `line`: C3), for a `LIFTED_VIA` expansion its `unit`, `repo` and `siteId`, and for
  an edge its `reason`; each query is parameterized, read-only and named by the recipe pattern
  `^[a-z_]+(:[A-Za-z]+)?$`; ids chunk at ≤ 1,000 and a full reply halves the chunk; a reply reaching
  `RESULTSET_SIZE` raises, never truncates; (2) **code-location anchor:** `resolve_location(reader, repo, path,
  line=None, symbol=None)` runs the three repo-scoped reads in order — `File` → mapped; else `SkippedFile` → "not
  mapped" with its reason; else "not found" (`.d.ts`, `.pyi` and vendored paths answer "not found") — and for a
  file+line fetches **every** symbol of the file via the `File.path` index and `DEFINES` and picks the narrowest
  containing interval, `ownedRegions` included, ties by `depth`; (3) **contract-point anchor:** by `keyNorm`,
  with prefix and template matching client-side through `ccg.linker.matchers` (one implementation); (4) a
  symbol anchor carries `qname` (EP §15.2 Q1); (5) an ambiguous anchor returns `options`, never a guess;
  (6) an unknown repo or a name outside the whitelist is a clean error before any query; (7) the three reads of
  carry-over C3 are enough to derive an item set from a raw recipe re-run (AC-16's raw leg).
- **Done / Gate:** reads exercised on `ccg-falkordb` with the P1-19 loader on a small fixture graph; `analyst`;
  `graph-dba` reviews the query texts (C3).

## P1-22 · BFS rows, guards, depth bound — `tdd-engineer`

- **Files:** `cpg/ccg/engine/bfs.py` (the walk), `cpg/ccg/answer.py` (remove the `function-level-continuation` entry of `ENGINE_REASON_CODES` and its test in `cpg/tests/test_answer.py`), `cpg/ccg/engine/state.py` (`value`, `path(P)`, `contract`,
  `key`, `end`, and the retired `function` state's removal), `cpg/tests/test_engine_bfs.py`,
  `cpg/tests/test_engine_bfs_guards.py`, `cpg/tests/slice/test_engine_equivalence.py`.
- **Behaviours (ordered):** (1) equivalence first: the slice walks (anchor `p0`, `p0.path`, `p0.method`, `p0.query`,
  `p0.nothing`, value walk from `buildUrl`) reproduce the slice's item sets under the slice's recipe; (2) the
  rows of graph note §6.2 as written, plus `PROVIDES <-` in the contract column (distance 0), and contract mode
  reaching consumers by **`CONSUMES <-`** with the `Link` attached through `ON <-`/`VIA`, so a consumer with no
  provider is found and reported `unmatched` (the slice walked `PROVIDES <-`, `TO <-`, `VIA <-` and never read
  `CONSUMES`/`ON`): a fixture graph where a contract walk from a provider-less contract point lists its consumers;
  (3) the **`CONTROLLED_BY` row** with a visited set seeded with the anchor keyed `(label, id, state kind, state
  path)`, re-expanding a node only on a strictly smaller `(possible, distance)`: a `cond → cond` cycle terminates
  (C1, F-12; a mutant replacing `<` by `<=` is killed), a two-point `DATA` cycle and a `DATA` self-edge
  terminate, a chain deeper than 64 levels raises; (4) the **`arg(i)`/`Ret` rows** (`CALLS ->` into a callee,
  index-matched `HAS_PARAM ->`, a `Ret`-reached callee summary onto the call site's `r`; receiver as index `-1`,
  `kw.<n>` by `Param.name`): items are then precise, the `function` state is removed and a caller of a function
  whose return the change does not reach is no longer listed through it; the slice's 21-item function-level sets
  (`-p0-graph.md` §3.3) shrink toward the V2-gated counts that note records (19, 5, 5, 3, 2, 4 across the six
  walks; a prior carried from the S17-L2 gate, not recomputed here), each pinned by a test, and `apiFetch` stays
  `confirmed`; (5) **depth bound:** `DEFAULT_DEPTH_BOUND = 2` applies when the
  request names none; items at distance ≤ bound; every edge the bound cuts from the frontier is counted into
  `depthBound.unexplored` (a cut edge is counted once per edge, not per path) and `unexplored > 0` implies `bound`;
  a distance-2 `flow`-shaped item is reached by default (the reason the bound is 2); (6) `Name` is one compound
  hop carrying the `DISMISSED` anti-join (no verdict log in P1: the join returns empty) and never a BFS node;
  (7) best predecessor: fewest `possible` hops, then distance, then length, then stable id; (8) `algorithm.version`
  is set per P1-00 and a test fails if a row is added without changing it.
- **Done / Gate:** `analyst`; mutation check on the visited-set comparison, the bound and the `unexplored` count.

## P1-23 · flow fallback, distance and grouping, reverse question — `tdd-engineer`

- **Files:** `cpg/ccg/engine/flow.py`, `cpg/ccg/engine/group.py`, `cpg/tests/test_engine_flow.py`,
  `cpg/tests/test_engine_group.py`.
- **Behaviours:** (1) **flow fallback (index §5):** from `Unit.flowAvailable` (assembler, D2): inside a unit with
  the flag true `DATA` and `CONTROLLED_BY` rows are followed `confirmed`; through a unit with the flag false the
  walk follows call-level reach only and every item is `possible`, `reason: flow-unavailable:<lang>`, the reason
  carried by the first hop that crosses into that unit (answer spec §4.1); (2) distance counts inter-function hops
  only (call, link, shared state, config); steps inside a function count 0; direct = distance 1; (3) items are
  grouped repository → component → distance, `confirmed` before `possible`, deterministic order, folded groups
  enumerate every member with its own chain; resolved calls' other candidates stay as `possible`/
  `resolution-unverified` and are folded in rendering only at or above the 0.95 precision threshold (never
  dropped); (4) the **reverse question** ("depends on") is the depends-on column with the same marking and grouping.
- **Gate:** `analyst`. **Needs:** 22, 18.

## P1-39 · staleness, stamp, answer builder, superset measurement — `tdd-engineer`

- **Files:** `cpg/ccg/engine/answer_build.py` (assembles the canonical answer and runs `ccg.answer` validation),
  `cpg/ccg/engine/staleness.py` (**imports** `ccg.staleness` from P1-20; no second implementation),
  `cpg/tests/test_engine_answer.py`, `cpg/tests/test_engine_superset_dev.py` (marked `live`).
- **Behaviours:** (1) staleness: each unit on any chain with its built commit and time, and whether it is stale-kept
  or not mapped with the reason (FR-7, FR-7a; AC-9 code-snapshot stamp from `:Snapshot`); (2) the stamp carries
  `verdictLogOffset` 0 and the empty-log hash until P4; (3) `facts_bytes` is byte-identical for the same request run
  twice and across rebuilds from the same commits; (4) the **superset measurement, on development flow items only**
  (`cpg/eval/dev/ac5-flow.jsonl`): the share of those items that call-level reach alone lists, by language, with the
  items' ids; it is the evidence index §5 sends to the method owner, **not** a gate here; it runs on a throwaway
  instance (index §8).
- **Gate:** `analyst`; `data-scientist` reads the measurement. **Needs:** 23.

## P1-24 · renderers — `coder`

- **Files:** `cpg/ccg/render/__init__.py`, `text.py`, `markdown.py`, `cpg/tests/test_render.py`,
  `cpg/tests/golden/render/` (rendered golden for `cpg/tests/golden/answer-ac1.json`).
- **Behaviours:** deterministic templates only — no fact is model-written; grouped text and a Markdown impact
  report (FR-10, AC-11); each hop shows file and line; a `Link` hop expands to its `BOUND_BY` and `LIFTED_VIA`
  entries; a `depthBound.unexplored > 0` note reads "reach stopped at distance d with k unexplored edges"
  from the two fields; **free-text identifiers from source** (`keyPath`, `Cond.code`, `calleeName`, `qname`, config
  keys, `launchFact.mountPath`/`service`/`envVar`) pass `ccg.answer.mask_identifier` (F-4) and are fenced with
  `fence_data` wherever a model could read them (F-3); a planted `"Ignore previous instructions"` in `Cond.code`
  occurs in the rendered output only inside the fence, and the string never reaches the plain-language
  summary prompt (C2, answer spec §7.1–§7.2); `viaData` is never rendered as "path-matched" for an item whose
  chain has no `DATA` hop.
- **Gate:** `analyst`; mutation check on the fence and the mask.

## P1-25 · clients: `ccg` MCP server, CLI, `raw_query` — `coder`

- **Files:** `cpg/clients/mcp/server.py` (the `mcp` SDK, stdio, FastMCP style; tools `ask`, `resolve`, `refresh`,
  `raw_query`; index D6), `cpg/clients/cli/main.py` (`ccg ask|resolve|refresh|raw` emitting canonical JSON,
  optional `--fail-on possible|any` exit codes), `cpg/clients/common.py` (system resolution, identity),
  `cpg/tests/test_clients.py`, `cpg/tests/test_pin_replay.py`, `cpg/tests/test_no_network.py`; a repo-root `.mcp.json` entry is a **proposal** in
  the step's report (the wiring file is `teco`'s/`cobb`'s to change, P1-27).
- **Behaviours:** (1) `ask(request)` takes the structured request (use case `impact` or `depends-on`, anchor,
  scope, depth bound) and returns the canonical answer; **other use cases return a typed "not built in this
  phase" error**, never an empty answer; (2) `resolve(anchor)` returns candidates with ids (code location,
  contract point); (3) `refresh(system, scope)` calls P1-20; (4) `raw_query(cypher)` runs `GRAPH.RO_QUERY` with
  the `ccg_raw_<system>` credential only; a write, `GRAPH.QUERY` and `GRAPH.PROFILE` are refused by the database
  and the stamp is unchanged (AC-21); a configured time limit is `TIMEOUT_MAX`; (5) the asker is the launching OS
  user; access control is P4 (FR-24), so `raw_query` is available to the local user in P1 and the report says so;
  (6) **AC-16 pin-and-replay** (`-eval-protocol-ml.md` §10.6): the structured request submitted through MCP, the
  CLI and the engine gives identical `facts_bytes`; a raw-recipe re-run of `recipe.queries` reproduces the item set
  (subject ids with `status` and chain `(kind, id)`), reported as the weaker leg by decision; (7) **no client holds a network credential or opens a socket except to FalkorDB**: `cpg/tests/test_no_network.py`
  parses every module under `ccg/engine`, `ccg/render`, `ccg/refresh.py`, `ccg/build.py`, `clients/` and (P1-26) `ccg/front`
  with `ast` and fails on any import or call that can reach the network outside a per-module allow-list
  (`falkordb`/`redis` for the store modules only), including `import urllib.request as u` inside a function,
  `__import__("socket")`, `importlib.import_module`, `subprocess` of `curl`/`wget`, and an import of `ccg.broker`
  outside `ccg/front/interpret.py`'s one broker client; each is a named mutant the test must kill;
- **Gate:** `analyst`; AC-16, AC-21 evidence is P1-42's.

## P1-26 · plain-language front — `coder`

Blocked on `-p1-ml.md` question 3 (model, prompt, grammar, sampling, smoke protocol, `CAP`), on the broker core
(P1-29) and on P1-45 (`cacheBypass` in provenance). Two dispatches (parsers+resolver first, interpreter+cache+summary second).
- **Files:** `cpg/ccg/front/parsers.py` (deterministic: stack traces are P5, so P1 parses routes, `file:line`,
  `path::symbol`, qnames, and diffs only as far as "not built"), `resolve.py` (deterministic resolver over
  `engine/anchors.py`), `interpret.py` (one-shot schema-constrained extraction of the use case and anchor spans),
  `cache.py` (key `ccg.answer.cache_key(normalized_question, snapshot_id, model_fingerprint, prompt_hash, …)`),
  `summary.py` (optional aggregate-only summary), `prompts/*.txt`, `grammar/*.json` (**never `.md`**, index D5),
  `cpg/tests/test_front_*.py`.
- **Behaviours:** (1) deterministic parsers run first; a question they fully resolve makes no model call; (2) for
  free prose, one model call through the broker's structured path (`component: answer.interpret`,
  `purpose: model-call`) with a grammar-constrained JSON reply validated against the canonical `Request` schema;
  (3) ambiguity is answered with the possible readings as `options` with `rank`, never a guess, and never more
  than the pre-registered cap; (4) the interpretation provenance (`endpoint`, model fingerprint, quantization for
  local, prompt/grammar/sampling hashes, `cacheHit`) is in the answer; a cache entry lives exactly as long as its
  snapshot, and the engine's `cacheBypass` input forces a fresh call (EP §10.4); (5) the optional summary is
  aggregate-only and **dropped on any added entity**; (6) `cpg/tests/test_no_network.py` (P1-25) is extended to `ccg/front` with the same named mutants: only `interpret.py`
  and `summary.py` may import the broker's client, and nothing may reach the network any other way; (7) a
  question about a use case P1 does not build is understood and answered "not built in this phase" with the
  request echoed; (8) no file under `cpg/ccg/front/` is `.md` (D5).
- **Gate:** `analyst`; `data-scientist` for the method. The AC-24 smoke run is P1-35.

## P1-27 · `teco` v1 and the usage skill — `cobb`

The unit's questions are in the index §9 (N-COBB). **Files** (all agent-definition or skill text, so `cobb`'s):
`claude/teco/teco.md` (the architect and implementer dispatch seams: an impact answer and its stamp in place of the
`CPG: <graph>, built …` line **for systems `ccg` covers**; falls back to the `cpg-analysis` freshness line for
`cpg_*` graphs), `skills/cpg-analysis/SKILL.md` (usage of the four tools; recipes for optional `cpg_*` graphs and
the freshness recipe stay), `claude/teco/kaizen/` per `claude/AGENTS.md`'s update rule, `claude/AGENTS.md` and
`skills/README.md` rows if wording changes, and the repo-root `.mcp.json` entry for the `ccg` server, which carries **no credential** (the server reads `CCG_CREDENTIALS_DIR`, P1-02; Claude-Code-only wiring, like `cypher`; OpenCode/Kiro wiring is C-310's problem, not P1's).
- **Behaviours (reviewable, not executable):** a brief for an architect or implementer on a covered system names the
  answer, the stamp and the staleness note; an uncovered system keeps the old line; a refresh failure is surfaced
  as "stale-kept" and never silently omitted.
- **Gate:** `analyst`; `cobb` keeps final say on prompts.
