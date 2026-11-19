# Review: P1-25a implementation (clients common code, CLI, `raw_query`)

> **Status:** active · **Owner:** `analyst` · **Tracks:** C-1001 (P1)

## Scope and verdict

**Reviewed:** the uncommitted P1-25a unit: `git diff -- cpg/README.md` (one table row) plus the new, untracked `cpg/clients/common.py` (379 lines), `cpg/clients/cli/__init__.py`, `cpg/clients/cli/main.py` (108), `cpg/tests/test_clients.py` (564). **Baseline:** `docs/plans/connected-code-graph-p1-answers.md` P1-25 behaviours (1)-(5), (8), (9) (behaviours (6), (7) and the MCP server are P1-25b, out of scope) and the P1-26 seam; `-answer-spec2.md` §2, §4.1, §5.1, §9 Q-1; `-egress-spec3.md` §10.1; AC-21. Root `REVIEW.md` read and applied; `cpg/` has no `REVIEW.md`.

**Verdict: needs changes.** One blocker (F1) and four majors (F2-F5). F1, F2 and F3 are small fixes. The behaviours the brief names are met where I could execute them: the refusals reproduce live, the credential is `ccg_raw_<system>` only, and nothing is rendered or printed for an invalid document.

CPG: considered, not relevant — the unit is new code with no counterpart in a loaded `cpg_*` graph, so I read it directly and ran it.

**What I ran** (all by me, this session):
- The coordinator's single invocation (`tests/test_clients.py test_answer test_render test_engine_answer test_engine_anchors test_engine_reader test_assemble_rows`, from `cpg/`): **1495 passed**, matching the figure in the brief. `tests/test_clients.py` alone: 39 passed.
- A live probe on my own throwaway, `ccg-falkordb-tw-16391` (scratch `CCG_CONFIG_DIR`/`CCG_CREDENTIALS_DIR`, system `rev25a`, `engine_helpers.mini_rows` loaded as build then renamed to the live key). It is torn down. `ccg-falkordb-tw-16390`, `falkordb-dev` and ports 6379/6390 were untouched; I checked `docker ps` afterwards.
- A mutation sweep of 68 mutants in a scratch copy (the copy is deleted); results in the Appendix.

## Findings

### F1 · blocker · An unreachable or refusing FalkorDB gives a Python traceback and exit 1, on `ask`, `resolve` and `refresh` (verified live)

`Client._reader` (`clients/common.py:288-294`) catches `ReaderError`, `OSError`, `AtRestError` and `AclError`. `CypherReader.for_system` calls `L.connect`, and `L.connect` constructs `FalkorDB(...)`, which runs `INFO` at construction (`falkordb/falkordb.py:132`) and so raises `redis.exceptions.ConnectionError` outside every handler. `Client.refresh` (`common.py:340-345`) has the same hole: it catches only `RefreshError` and `LoadError`. `cli.main` (`main.py:97-104`) catches only `ClientError` and `AnswerError`.

Reproduced: `CCG_FALKORDB_PORT=16399 python -m clients.cli.main ask --request '{...}' --fail-on any`, and the same for `resolve` and `refresh`, each end with `redis.exceptions.ConnectionError: Error 111 ... Connection refused.` and **exit 1**. Only `raw` survives, because `raw_query` wraps its connect in `except Exception`.

Why it matters:
- Exit 1 is the `--fail-on` code. A CI gate running `ask --fail-on possible` reads a store outage as "possible impacts exist".
- The unit's own header and the README row both promise "2 typed error, JSON on stderr".
- A wrong-password `AuthenticationError` or a disabled ACL user takes the same path.
- The traceback is unbounded text on stderr.

Fix:
1. In `_reader` and `refresh`, map any exception from the connect call to `StoreError`, with a static message and the exception type name only, `from None`. `raw_query`'s shape does this already.
2. Add a backstop in `main()` so that no exit-1 path exists except the `--fail-on` one: `except Exception` → typed `internal-error` payload, exit 2, no message text.
3. Test: a client whose `reader_factory` raises `ConnectionError` and a `connect=` that raises. Assert code 2, empty stdout, a typed JSON error, and, for `ask --fail-on any`, that the code is not 1.

### F2 · major · `raw_query` silently truncates at the server's `RESULTSET_SIZE` and presents the rows as complete (verified live)

Live: `raw --cypher "UNWIND range(1,20000) AS x RETURN x"` returns 10000 rows with no marker, and `... LIMIT 15000` also returns 10000. `config.py:15-17` already records that a reply of exactly `RESULTSET_SIZE` rows "may have been cut short by the server", and `CypherReader` raises for that reason. `raw_query` (`common.py:366-367`) ignores it. The plan's raw re-run is the weaker AC-16 leg (item set reproduction), and a truncated re-run would produce a spurious mismatch or a spurious match.

Fix: if `len(rows) >= config.RESULTSET_SIZE`, either raise `StoreError` (`result-too-large`) as the reader does, or return `"truncated": true`. Pin it with a fake returning exactly `RESULTSET_SIZE` and `RESULTSET_SIZE - 1` rows (a boundary pair).

### F3 · major · The "no fallback to another role" requirement, the `--system` argument, and credential non-leakage are all unpinned (surviving mutants)

Each of the following mutants leaves all 39 tests green:
- **M13**: on any failure of the `ccg_raw_<s>` call, retry as `ccg_engine_<s>`. Brief item (2) asks for this exactly. `opened == [...]` is asserted only on the success path (`test_clients.py:446`).
- **L14**: `cli.main` calls `client_factory(None)` instead of `client_factory(args.system)`. The `--system` flag, and with it the credential and the graph key, is never tested through the CLI. The default factory `lambda system: Client(system)` is untested too.
- **L15**: `refresh --scope` dropped in the CLI.
- **M16b / M50**: the raw password put into an error's `detail`, or an exception's text into `NoCredentials`. Nothing asserts that a password is absent from any payload. By inspection and live, none leaks; the test cannot say so.
- **M40**: `candidatesTotal` dropped. **M47**: the cypher `.strip()`ped, which contradicts `test_raw_query_does_not_alter_the_text` (that fixture has no edge whitespace). **L17**: stdin `-` untested.

Per root `REVIEW.md`, a check that cannot fail is a finding. Fixes, each as a named mutant to kill:
- M13: in the refused-write and `Down` tests assert `opened` has exactly one entry and that its user is `ccg_raw_shop`.
- L14: `main(["--system","alpha","raw",...], client_factory=spy)` and assert `spy` received `"alpha"`. Add a case that drives the real default factory with a `CCG_SYSTEM` env.
- L15: assert `refresh --scope a b` reaches `refresh([...])`.
- Leakage: run every error path with a distinctive planted password and assert it is absent from `payload()`, stdout and stderr.
- M47: use a leading and trailing newline in the text test.

### F4 · major · A raw query can OOM-kill the FalkorDB instance inside its time limit (verified live)

`raw --cypher "UNWIND range(1,100000000) AS x RETURN count(x)" --timeout-ms 200` killed my throwaway (`docker inspect`: `OOMKilled=true`, exit 137, memory cap 2g) within one call. TIMEOUT bounds wall time but not allocation. On `ccg-falkordb` (no persistence) that would drop the loaded snapshot until the next refresh, for every system on it, and the role is open to the local user in P1 (behaviour (5)). It is not a defect in this diff: the diff cannot enforce a server memory limit.

Action: this is a documented-limit item for the plan owner (`architect`, with `devops`/`graph-dba`). State it in the P1-25 plan and the README row (a time limit is not a resource limit), and decide whether P4 (FR-24) or the instance config owns it. Do not try to fix it with a client-side query filter; that would contradict "the database is the enforcer".

### F5 · major · The socket timeout "buffer" is not equivalent to its removal: live, it changes which error you get (a claimed-equivalent, re-derived by counter-shape)

`_RAW_SOCKET_TIMEOUT_S = 60` (`common.py:55`, used at `:235`) is a fixed constant, and the docstring promises "a socket timeout above the query time limit". Mutant M42 (socket timeout 0.001) survives offline, because the fake never reads a socket. Live, with the constant set to 0.05 and `timeout_ms=2000`, the same slow query gives `store-error: the raw query failed (TimeoutError)` instead of `the query timed out after 2000 ms`. With 60 it gives the server's timeout. So the constant is load-bearing and no test pins it.

Also the promise is not general: it is false for any `timeout_ms` above 60000, which the server accepts if its `TIMEOUT_MAX` is raised. On the pinned instance (10000) it is fine today.

Fix: derive it per call, `socket_timeout = timeout_ms / 1000 + margin`, and test that `_open_raw` receives a socket timeout greater than the time limit. A fake `FalkorDB` constructor recording its kwargs would kill M42/M43.

### F6 · minor · `ask` with `--fail-on` exits 0 on `ambiguous` and `unresolved` documents

`fail_on_tripped` (`main.py:34`) gates on `status == "resolved"`. That gate is itself an equivalent mutant (L04): V2 makes `items` and `groups` empty for the other statuses, so removing it changes nothing. Verified over every `b_*` fixture of `test_answer.py`, where `b_amb`-style and `b_unres` documents exit 0 under both levels. This is consistent with spec2 §9 Q-1 ("the exit code means items of that status exist"), but once P1-26 can return these documents, `--fail-on any` reports success for "I could not understand/locate it". Route to `architect` as a spec2 Q-1 follow-up: either say so explicitly, or give those statuses a distinct code (not 1).

### F7 · minor · The seam's docstring promises wiring the code does not have, and P1-26's Files omit the file that must change

`common.py:12-14` says the default `front` wiring is "a lazy import in Client's constructor". `__init__` only stores `front` (`:283`), and the CLI's default factory (`main.py:93`) passes none. So after P1-26 lands, `--question` is still `not-built` unless `clients/common.py` is edited. P1-26's `Files:` list names only `ccg/front/*`, prompts, grammar and tests. See ruling 3c.

### F8 · minor · Importing `clients.common` loads the broker client, `subprocess` and `socket` (verified)

`from ccg import refresh as refresh_mod` (`common.py:34`) pulls `ccg.broker.client`, `ccg.broker.framing`, `subprocess` and `socket` into `sys.modules` for every `ask`/`resolve`/`raw`. I checked `sys.modules` after `import clients.common`. `falkordb` and `redis` are not loaded (lazy), which is what §10.1 asks for. The per-module AST test (P1-25b) will pass, because `common.py` has no direct broker import. But `-egress-spec3.md` §2.1 line 164 says the clients "import no broker module", and at run time that is not so. Cheap fix: import `ccg.refresh` inside `Client.refresh`, and take `DEFAULT_MANIFEST` from where `refresh.py` gets it. Judgment call; flag it to the P1-25b author so the test's wording is decided deliberately.

### F9 · nit
- `Client.__init__` (`common.py:275-276`) does `host or env_host, port or env_port`, so an explicit `port=0` silently becomes the env/default, and an explicit port is not range-checked.
- The README and plan name `ccg ask|resolve|...`; no `[project.scripts]` entry exists (module invocation only). Say so in the row.
- `_plain` stringifies unknown cell types silently (fine), but `rows` are unbounded in size (see F2).
- `_raw_error` classifies on substrings of server text (`"permission"`, `"read-only"`). This is typing only, never enforcement, so it is not a defect. A refusal that the server words differently becomes `store-error`, not `raw-refused`.

## Rulings requested in the brief

**(2) `raw_query` read-only enforcement.** Verified, by running it. The client does no string filtering: `raw_query` passes the text unchanged with `params=None` (`common.py:361`), and the only text inspection is `_raw_error`'s after-the-fact message typing. Enforcement is the database's two layers, both reproduced live on my throwaway:
- The role itself: a direct `ccg_raw_rev25a` connection got `NoPermissionError ... 'graph.QUERY'`, `'graph.PROFILE'` and `'graph.CONFIG'`.
- `RO_QUERY`'s own read-only rule: `CREATE`, `SET`, `DELETE`, `MERGE`, `REMOVE`, `CALL db.idx...createNodeIndex` and `CREATE INDEX` through `raw` each returned `raw-refused`, exit 2. Node count (44) and `snapshotId` were identical before and after (AC-21).

The credential is `ccg_raw_<system>` only. No fallback exists in the code, but no test pins that (F3/M13). The raw role cannot read `TIMEOUT_MAX` (verified: `GRAPH.CONFIG` refused). A client default of `config.QUERY_TIMEOUT_MS` (10_000) is an honest rendering of "a configured time limit is `TIMEOUT_MAX`": it equals the pinned instance's value, which `config.py:13-16` documents. It is refused loudly (`raw-refused`, a TIMEOUT_MAX message) on a lower-configured instance, never silently, and a larger limit (10001 and 99999999999) is refused by the server. Suggested plan wording: "a time limit is sent on every call; its default is the pinned `TIMEOUT_MAX`; the server refuses a larger one". The client itself never claims to read the instance value. Error text carries no credential: messages hold only a role name, the system, the exception type, and the server text capped at 200 characters and `safe_text`-escaped. The server text can echo a fragment of the user's own cypher (a syntax error did), and cypher parameters are never sent. Wrong-password errors come from redis without the password. F4 and F2 are the real raw-path risks.

**(3a) `scope` returns typed `not-built`.** Correct, and better than ignoring it. `build_answer` takes no scope, and spec v1's `Request.scope?` is an optional field the engine never fills, so honouring or dropping a scope would be a silent wrong answer. The plan's "(use case, anchor, scope; ...)" is stale against the engine. Suggested P1-25 (1) text, owner `architect`: "a structured request is `{useCase, anchor}`; `scope` is not a P1 input, a request carrying it is the typed `not-built` error (never ignored), and the answer's `request` carries no `scope`." P1-26 then needs a matching sentence for a question that names a repository scope.

**(3b) Typed errors with candidates instead of canonical ambiguous/unresolved documents.** It does not contradict the answer spec. The spec defines the statuses as documents "that ran no query" and gives no rule that a structured ask must produce one. But the plan has a gap: nothing in P1-25, P1-26 or P1-39 builds them (`build_answer` refuses a non-resolved anchor with `BuildError`, and `grep` of `answer_build.py` for `unresolved` finds nothing). P1-26 (3)/(3b) cover interpretation ambiguity, as `options` with `rank`, through the front's resolver. So the delegate's typed error (`anchor-ambiguous`/`anchor-unresolved`, with `candidates`, `state`, `reason`, `unitId`) is a sound interim for the structured path. Recommend, owner `architect`: P1-25 text says a structured ask with a non-single anchor is a typed error carrying the `resolve` shape; P1-26 owns the canonical `ambiguous`/`unresolved` documents (`lookups`, `options`) for the plain-language path and states whether it also shares the resolver with `resolve`. AC-9b's "not mapped" is about a never-built unit in a resolved answer's `staleness`, not about `lookups`, so no AC wording is contradicted.

**(3c) The seam.** `front(question, *, reader, cache_bypass) -> canonical answer`, validated by `Client.ask` before it leaves, fits P1-26. The reader carries `system`, so the front can read the snapshot for the cache key. The front opens its own broker run (§2.1). `cache_bypass` reaches it and provenance is the front's to set. Two things it needs: F7's file list, and the default wiring (the `Client` or CLI factory must import the front), under the F8 note that `ccg.front` modules are under the import-boundary test. Probes: `ask(question=..)` without a front raises `not-built` and reads nothing (`test_clients.py:192`); with a front the answer is validated (`:203`).

**(3d) Endpoint.** Verified live: port 6379 is refused in `endpoint`, in `Client()` and in `_open_raw` (three mutants M17-M19, all killed), and `CCG_FALKORDB_PORT=6379 ... raw` gives `bad-request`, exit 2. The default is `127.0.0.1:6390` and the env names are `CCG_FALKORDB_HOST`/`CCG_FALKORDB_PORT` (M20, M21 killed). Note the default 6390 is the shared `ccg-falkordb`. The only default-system source is the manifest (`graphmind-ai-lab`); `--system` overrides `$CCG_SYSTEM`, which overrides it (M23 killed; but see F3/L14 for the CLI path).

## (1) Behaviours against code

| # | Verdict | Evidence |
|---|---|---|
| (1) `ask` structured / question / `cacheBypass` / other use cases | met | `parse_request` (`common.py:149`) is strict: unknown key, use case and anchor kind are typed errors. `cache_bypass` is type-checked, passed to the front, and has no effect on a structured request (`test_clients.py:180`). Live: `scope` → `not-built`; `--question` → `not-built` |
| (2) `resolve` | met | live, candidates with ids; ambiguous and unresolved shapes verified for `name` anchors |
| (3) `refresh` | met offline only | the roles are `ccg_build_<s>`/`ccg_verdict_<s>`, pinned by test; I did not run a live refresh |
| (4) `raw_query` | met, with F2, F4, F5 | see ruling (2) |
| (5) asker | met | `asker()` is `pwd`; the raw result carries it (`"asker":"mauricio"` live) |
| (8) validate first | met | `Client.ask` validates (`:330`) and `render_answer` validates first (`:218`). Probe through the real CLI with an invalid document from the builder or the front, for each of `json|text|markdown` and with `--fail-on any`: stdout is empty, exit 2, a typed `store-error` on stderr. The `invalid-answer` branch (`main.py:102`) is reachable only from a client that skips `ask`'s own validation (the `DocClient` test double). |
| (9) notice never dropped | met | `notices` comes back unchanged in JSON, text and markdown. The `fail_on_tripped` function reads items only: mutant L05 (a notice trips) is killed and a zero-item document with a notice exits 0 under both levels |

**`--fail-on` semantics, verified:** `possible` trips on `status == "possible"` for items and folded-group members (`all_items`; L03 killed; `b_group` trips both levels), `any` on any item (L01/L02 killed, the per-status mapping), a `confirmed`-only document trips `any` only, and the level-less run never exits 1 (L06 killed).

## (5) Import boundary

`common.py` has no module-level `falkordb`/`redis`: `import clients.common` leaves both out of `sys.modules` (checked). The lazy import is in `_open_raw` (`:234`), and `ccg.store.loader.connect` lazy-imports it too. §10.1's allow-list names `cpg/clients/common.py` for `falkordb`/`redis`, and an `ast` walk sees a function-level import, so lazy or not the module is allowed as written. `cli/main.py` imports only `argparse`, `json`, `sys`, `ccg.answer`, `ccg.render.common` and `clients`; it opens nothing itself. The one wrinkle is F8, the transitive broker import.

## What's solid
- The module shape: one `Client` with every effect injectable, thin CLI, one `ClientError` hierarchy with a `code`, `from None` on every re-raise so no server text leaks through a chain.
- Strict request parsing, with typed refusal in place of silent defaults; `scope`/depth/unknown keys never ignored.
- The test suite is well aimed: **57 of 68** mutants were killed overall, including every credential-role, graph-key, `RO_QUERY`-versus-`QUERY`, time-limit, 6379, validate-first, `--fail-on` mapping and notice mutant I tried.
- `raw_query` genuinely leaves enforcement to the database, as the plan asks.

## Open questions
1. F4: who owns a memory bound for raw access, the instance config (`devops`/`graph-dba`) or P4? It needs a decision and a line in the plan.
2. F6: should a non-resolved answer be distinguishable from "no items" by exit code? A spec2 Q-1 amendment for `architect`/the stakeholder.
3. F8: is the broker import boundary "no direct import" (what the test checks) or "not loaded at run time" (what §2.1 says)? One of the two should be reworded.

## Appendix · mutation sweep

Method: a scratch copy of `cpg/` (no `.venv`, no `.git`), one mutant at a time from pristine files, `tests/test_clients.py -x`, 68 mutants (the first run's 66 plus M16b and M50, with L04 corrected). I did not have the delegate's list of 28; these are my own, covering the same axes plus the ones the brief names.

- **Killed (57):** M01-M12 (credential role and user, graph key from `asker()`/system, credential system argument, `RO_QUERY`→`QUERY`, time limit not sent / huge default / `<1` bound / bool), M14, M15, M16, M17-M23, M24-M31, M32 (the builder reusing the resolver's reader), M33, M34-M39, M41, M44-M46, M48, M49, L01-L03, L05-L13, L16.
- **Survived (11), by cause:**
  - M13 fallback to engine role, L14 `--system` dropped, L15 `--scope` dropped, M16b/M50 password in `detail`/message, M40 `candidatesTotal`, M47 cypher stripped, L17 stdin `-` — 8 mutants, **test gaps (F3)**.
  - M42 socket timeout 0.001 and M43 retry removed — offline-unobservable; M42 is live-observable (F5).
  - L04 (the `resolved` gate in `fail_on_tripped`) — **equivalent**, V2 makes the lists empty (F6); the gate is dead defensive code.

`L04` as first written was a syntax error and was re-run; the figures above are from the corrected run. The "claimed-equivalent" the brief names (the `buffer` removal) I read as the socket-timeout margin; if the delegate meant another, my re-derivation of it does not apply.

## Pass 2

**Verdict: approve with suggestions.** F1 (blocker) and F2-F7 are fixed and re-run. What remains is minor (below). The coordinator's 10-file invocation was 1753 passed; I re-ran `tests/test_clients.py` alone: 56 passed.

I ran on my own throwaway `ccg-falkordb-tw-16391` (scratch config and credentials dirs, torn down; `ccg-falkordb-tw-16390` and `falkordb-dev` untouched). Mutants ran in a scratch copy (deleted).

| Finding | Disposition | Evidence re-run |
|---|---|---|
| F1 blocker | **fixed** | Closed port 16399: `ask --fail-on any`, `ask`, `resolve`, `refresh`, `raw` each exit **2**, stdout 0 bytes, one typed JSON line (`store-error`: "the store is not reachable (ConnectionError)"; refresh: "the refresh failed against the store"), no traceback. Wrong ACL passwords on the live instance give the same for all four commands, with `AuthenticationError` as the type name and the planted password absent from stdout and stderr. Mutants killed: `_reader`'s broad except dropped, `refresh`'s dropped, the `main` backstop dropped, and the backstop returning 1 (4 tests failed). Exit 1 now comes only from `--fail-on`. |
| F2 major | **fixed** | Live `raw`: `range(1,9999)` returns 9999 rows, exit 0; `range(1,10000)` exits 2 with `result-too-large`, stdout empty. Mutants `>=`→`>`, `>= SIZE+1`, check removed, wrong code: all killed. |
| F3 major (8 survivors) | **fixed** | Re-derived as mutants against the new code: M13 (retry as engine role), L14 (`--system` dropped), L15 (`--scope` dropped), M16b/M16c (password into `detail`, and into the message), M50 (`NoCredentials` text leaks), M40, M47, L17: **all killed**. New survivor M51 (the "store is not reachable" message built from `{exc}` instead of the type name) is a minor gap, see below. |
| F4 major | **accepted limit** | Stakeholder decision recorded in the P1-25 plan; the README row and the module docstring say a time limit bounds wall time, not memory. Not re-run (it kills an instance). |
| F5 major | **fixed** | `raw_connect` receives `socket_timeout = timeout_ms/1000 + 5`: with `timeout_ms=2000` the spy saw 7.0, and the live slow query gives `the query timed out after 2000 ms`. My old probe, forcing a 0.05 s socket timeout, now gives a typed `store-error` ("the raw query failed (TimeoutError)", exit 2), not a traceback or exit 1. Mutants socket = limit, margin 0, fixed 0.001, socket arg ignored, retry removed: all killed (the earlier "survived offline" mutants M42/M43 are now pinned). |
| F6 minor | **fixed**, per plan (8b) | `--fail-on` with `ambiguous` or `unresolved` exits 3, output still printed; without `--fail-on` it exits 0; `--format text` is the same. Mutants exit 3→0, 3→1, without-fail-on, only-unresolved, only-ambiguous: all killed. |
| F7 minor | **fixed** | The docstring now says the front is injected and P1-26 must edit this module or the CLI factory; the plan's P1-26 Files now list `clients/common.py` and `cli/main.py`. |
| F8 minor | **fixed** | After `import clients.common`, and after `resolve`, `raw` and a refused `ask` against the live store, `ccg.broker*` is not in `sys.modules` (`falkordb` is). `Client.refresh` imports `ccg.refresh` inside the method. `DEFAULT_MANIFEST` equals `ccg.refresh.DEFAULT_MANIFEST` (checked). The test is a subprocess run, and it kills the eager-import mutant (verified: `ccg.broker` loaded at import with the mutant; the test fails). Caveat for anyone mutating it: the test runs `python -I`, which resolves `ccg` and `clients` through the editable install, not the cwd, so a mutant in a copy needs the `-I` dropped to be observed. |
| F9 nits | partly | `port or env_port` and the unvalidated explicit port are unchanged (still a nit). The "no `ccg` console script" note is in the README row. |

**Live `refresh`, end to end.** On my throwaway, with ACL users created for a scratch system, `Client(system).refresh()` ran the real `ccg.refresh.refresh` through the real `build` and `verdict` roles and the real swap: 3 units `fresh`, a `snapshotId` returned, and `raw` then read the same `snapshotId` from the live graph (`true`). The one deviation: `Client.refresh` passes only the default manifest, so I wrapped `refresh_mod.refresh` to supply a scratch manifest, repo root and work dirs (`tests/test_refresh.World`) rather than build the real monorepo into the shared `.ccg-work`. The client's own plumbing (lazy import, credentials, connect, `Target`, report mapping) is what ran.

**`_safe` versus `ccg.refresh.safe_text`.** Same rule: 0 differences over 8 hand cases and 300 random strings (controls, backslash, non-ASCII, astral, long) at caps 120 and 50, and the cap 120 equals `_MAX_FIELD`. But no test pins the parity, so a change to either silently diverges (mutants `_safe` cap 5 and `_safe` without escaping survive). Minor.

**Same-round fixes against each other.** F5 and F1: a socket timeout now maps to a typed `store-error` (exit 2), checked with the forced 0.05 s socket. F5 and F2: the row cap is checked after the call, so a timeout and a full reply cannot both be reported; a reply that arrives is capped, one that does not is the timeout. F1 and F6: exit 3 applies only to a delivered document, and typed errors still exit 2. One wording defect: an `AuthenticationError` is reported as "the store is not reachable", which sends an operator to the network, not the credentials. Minor.

### New findings (minor)
1. **No test fails if the "store is not reachable" message carries `{exc}`** (M51); the planted-password test covers the raw and credential paths only. Extend the leak test to the `_reader` and `refresh` messages.
2. **`_safe` duplicates `ccg.refresh.safe_text`** with no parity test (above). Either a one-line test comparing the two, or import it lazily where `refresh` already is.
3. The `AuthenticationError` wording above: a distinct typed code (for example `store-auth`) would be clearer than "not reachable".

None blocks. Suggested before P1-25b: the first two are one-line tests. The F4 limit should appear in the P1 exit report (P1-38), as the plan states.
