# Review: P1-25b implementation (MCP server, AC-16 pin-and-replay, import-boundary test)

> **Status:** active · **Owner:** `analyst` · **Tracks:** C-1001 (P1)

## Scope and verdict

**Reviewed:** the uncommitted P1-25b diff against 6beba275: new `cpg/clients/mcp/{__init__,server}.py`, `cpg/tests/test_pin_replay.py`, `cpg/tests/test_no_network.py`; changed `cpg/clients/common.py` (`dump_json`, `StoreAuthError`, `_is_auth_failure`), `cpg/clients/cli/main.py` (one line), `cpg/tests/test_clients.py`, `cpg/tests/test_atrest.py` (`SAFE_TOOL_PARAMS`), `cpg/README.md` (one row). **Baseline:** `docs/plans/connected-code-graph-p1-answers.md` P1-25 behaviours (1), (6), (7), (8), (9), and `docs/reviews/connected-code-graph-p1-25a-impl.md` Pass 2's three minors. Separate file from the 25a review: a different unit with its own verdict. Root `REVIEW.md` applied; `cpg/REVIEW.md` does not exist.

**Verdict: approve with suggestions.** No blocker. One major (the broker-import rule is evaded by `from ccg import broker`), the rest minor. The MCP wrapper is faithful (16 mutants of it, 15 killed, the survivor equivalent), and the store-auth classification is correct against a live instance.

CPG: considered, not relevant — the questions here were a thin wrapper's argument pass-through, a static import scanner and live error mapping, answered by mutants and live runs, not by call-graph queries.

**Run:** `tests/test_clients.py tests/test_pin_replay.py tests/test_no_network.py tests/test_atrest.py`: 287 passed, 4 deselected (live). `tests/test_pin_replay.py -m live` on my own throwaway `ccg-falkordb-tw-16391`: 4 passed. **`tests/test_scip_parse.py` alone, quiet: 58 passed in 22 s** (the coder's four memory-limit failures did not reproduce; consistent with load, as reported). Mutants ran in a scratch copy of `cpg/` (deleted); the real tree is untouched. Throwaway torn down; `docker ps` afterwards shows `ccg-falkordb-tw-16390` and `falkordb-dev` untouched and no `-16391`.

## Findings

### F1 (major, verified) — `from ccg import broker` evades the broker-import rule

`scan_source` (`cpg/tests/test_no_network.py`) checks a `from X import Y` child only when `X` is already under `ccg.broker` or `X.Y` falls under `NETWORK`/`STORE_DRIVERS`/`MCP_SDK`. `ccg.broker` is in none of those, so the spelling that loads the same package slips through:

| planted source, path | result |
|---|---|
| `from ccg import broker` in `ccg/answer.py` | **no violation** |
| `from . import broker` in `ccg/answer.py`; `from .. import broker` in `ccg/engine/x.py` | **no violation** |
| `from ccg import broker` in `ccg/refresh.py` (where only `ccg.broker.client` is granted) | **no violation** |
| `from ccg.broker import client` in `ccg/answer.py` | caught |

The plan names "an import of `ccg.broker` outside `ccg/refresh.py`, `ccg/front/interpret.py` and `ccg/front/summary.py`" as a rule each named mutant must kill; the module docstring promises it. The helper that would close this, `_imported_names` (line 77, returns `mod` plus `mod.name`), is defined and never called. The named mutants pass because each uses a `from ccg.broker...` spelling. The run-time test (`sys.modules`) covers `Client` paths only, not `ccg/engine` or the future `ccg/front`.

**Fix:** in the `ImportFrom` branch call `_check_import` for `f"{mod}.{alias.name}"` for every alias (that is what `_imported_names` returns, so use it), and add the three rows above as mutants (`from ccg import broker`, `from . import broker`, and the refresh-grant case, which must still flag because only `.client` is granted).

### F2 (minor, verified) — evasions the scanner misses

Tried 70 spellings. The plan's named set and the author's list are all killed. Slipping through (planted in `ccg/answer.py`, no violation):

- **Socket modules beyond the list:** `import _socket`, `import selectors`, `import wsgiref.simple_server`, `import ctypes` (`ctypes.CDLL(None).socket(...)`), `import tornado/flask/fastapi/trio/gevent/twisted/boto3/pexpect`, `webbrowser`.
- **In-process dynamic import without the two banned names:** `getattr(__builtins__, '__import__')('socket')`, `vars(builtins)['__imp'+'ort__']`, `exec('import socket')`, `eval(...)`, `compile(...)`, `runpy.run_module('socket')`, `pkgutil.resolve_name('socket.socket')`, `import imp`, `zipimport`. The docstring calls "the dynamic-import and module rules the backstop for the in-process paths"; these paths are not behind it.
- **Process launches:** `subprocess.run(['sh','-c','true; curl x'])`, `'FOO=1 curl x'`, `'timeout 5 curl x'`, `[sys.executable,'-c','import socket']`, `['git','clone'|'fetch'|'push', url]`, `['pip','install',...]`, and a tool name built by `'cu'+'rl'`. `_tool` looks at the first word of each string only, but the docstring says "any string naming one inside a process-launching call".

Caught as intended: `from importlib import import_module as f`, `from socket import socket as S`, `http.client`, `asyncio.open_connection` (`from asyncio import ...`), `smtplib`, `imaplib`, `http.server`, `xmlrpc.client`, `urllib3`, `ftplib`, `bash -c "curl"`, `env curl`, an aliased `subprocess.run`, `import importlib as I; I.import_module`.

This is a static bound, and the docstring says so for a name built at run time; the rest is cheap to tighten. **Suggest:** add `_socket`, `selectors`, `ctypes`, `wsgiref`, `runpy`, `pkgutil`, `imp`, `zipimport`, `webbrowser` to the banned imports; flag the names `eval`, `exec`, `compile`, `__builtins__`, and string constants equal to `__import__`/`import_module`; in a launcher call match a tool name at any word of any string, and flag `sys.executable`/`python` with `-c`. Then reword the docstring to what is scanned. Coverage probe, not a list: parametrize the mutant table over {import statement, from-import, dynamic call, launcher} x {each banned family}, so a new family is one row.

### F3 (minor, verified) — MCP argument types are validated by pydantic before `Client`, so two contracts diverge

FastMCP validates the declared types before the tool body runs. Verified by calling the tools:

- `raw_query {"timeoutMs": true}` reaches `Client.raw_query` as `timeout_ms=1`; `"5"` becomes 5, `2.0` becomes 2. `Client.raw_query` refuses a bool with `bad-request`, and the CLI cannot send one. Likewise `cacheBypass: 1` and `"yes"` arrive as `True`; `Client.ask` refuses a non-bool.
- A wrong type (`cacheBypass: "maybe"`, `timeoutMs: 2.5`, `scope: "abc"`, `request: "str"`, a missing `anchor`) raises `ToolError`, so the result is `isError` with pydantic's multi-line text (input echoed, a doc URL), not the `{"error": {code, message}}` payload the module docstring and the plan promise for every failure.

No safety consequence (a 1 ms limit is a failed query). It is the silent-coercion class the author's own pass-through test names. **Fix, either:** declare the four arguments `Any` and let `Client` validate (the schema loses precision; `Client` already holds every check), or keep the types and state in the docstring that a schema-level rejection is FastMCP's text. Add one test per choice (`timeoutMs: True` expecting `bad-request`, or expecting the documented text).

### F4 (minor, verified) — AC-16 raw leg: coverage is narrower than the plan's wording and is stated in only one place

`raw_item_set` is independent of the engine: it imports nothing from `ccg.engine` (the engine appears only in `engine_leg`), runs the recipe's own query texts through `Client.raw_query` as `ccg_raw_<s>` (the live test asserts `set(ran) <= recipe texts`), and derives items from row columns. Mutants of the driver in a copy, run live: status always `confirmed`, `name_in` ignored, owner column swapped, chain without the call-site hop, wrong hop kind: all **killed**. Raw leg, shape: flag-true, impact, one `CALLS`/`CALLS_NAME` level and `HAS_CALL`; anything else raises `Underivable` (so it fails, never skips; tested for a flag-false and a depends-on recipe).

Gaps against the wording:

1. Plan P1-25 (6) and EP §10.6 state the raw leg without a coverage qualifier and, per §10.6, unconditionally. The qualifier ("flag-true impact, `CALLS`/`HAS_CALL` only") is in the test docstring alone; the README row says only "the raw-recipe leg is live". The live raw tests run one world and one pin (`m.py::A`). `depends-on`, flag-false and DATA/link walks have **no** raw leg run, only the refusal test.
2. One driver branch is unexercised: mutant `sites[row[1]] = False` (the `calls_in` `row[7] == "possible"` read) **survives**, because the live world has no `CALLS` edge with `possible` status; only the name-matched path is possible-valued. Column 7 is never checked against a real row.
3. The "engine mutant" check I tried (an engine that reports possible items as confirmed) was killed, but by answer validation, not by the raw leg; I did not find an engine mutant that only the raw leg catches.

The plan's done-condition is met for the shipped world. It does not support the claim "AC-16 raw leg holds" for P1-42 without more, and EP §10.6 says the harness's full driver is GC's, so no more is required of this step. **Action:** state the coverage in the README row and in the P1-38 exit report; add a `CALLS` edge with `status="possible"` to the live world, so column 7 is read; owner for the full driver stays GC (P1-42). See Q1.

### F5 (minor, verified) — store-auth classification is correct; one test gap

`_is_auth_failure` matches the class name `AuthenticationError` over the MRO. redis-py 8.1.0 maps both `NOAUTH` and `WRONGPASS` to that class and `NOPERM` to `NoPermissionError` (a `ResponseError`), so a permission refusal on the raw path stays `raw-refused` and is not misread as auth (`_raw_error` checks auth first, and `NOPERM` is not it). No other class in `ccg`, `clients` or `extractors` is named `AuthenticationError` (grep). **Live on my throwaway**, `ccg_*_probe` users, all four commands through the CLI:

| condition | ask / resolve | refresh | raw |
|---|---|---|---|
| password rotated, engine / build / verdict / raw | `store-auth` for engine; others unaffected | `store-auth` for build or verdict | `store-auth` for raw |
| user disabled (engine, raw); engine user deleted | `store-auth` | n/a | `store-auth` for raw |
| unrelated server errors (no graph) | `store-error`, not auth | `refresh-failed` | `store-error` |

Every case exits **2**, stdout empty, one JSON line, the role in the message and no password. F1/F2/F5 of the 25a review are unaffected: exit 1 stays `--fail-on`-only, `result-too-large` is checked after a successful call, and a timeout still maps to `store-error` (auth is tested first and a timeout is not an `AuthenticationError`). Mutants killed: auth check dropped in `_reader`, `refresh`, `_raw_error`; wrong class name; wrong role. **Survivors:** matching the exact type instead of the MRO (`type(exc).__name__ == ...`) passes everything, because the test raises the base class and redis defines no subclass of it, so the MRO walk is unobservable (harmless, but the comment promises more than is tested). A mid-session auth failure inside a query would reach `CypherReader._once` and surface as `store-error` ("query 'x' failed (AuthenticationError)"), not `store-auth`: not reachable today because the connection is made per call, but note it if a reader is ever held open.

### F6 (nit) — consistency

`ask` returns canonical bytes with the trailing newline (`answer.dumps`), the other tools return `dump_json` without one. Both match the CLI byte for byte (the tests pin it), and the module docstring says so; only the wording "the bytes the CLI prints, without its newline" differs for `ask`. No action beyond leaving it.

## Answers to the brief's numbered checks

1. **MCP wrapper (executed).** Canonical bytes equal the CLI's (`test_mcp_ask_returns_the_canonical_bytes...`); notices intact; validate-first. 16 mutants in a copy, tests `test_clients.py` + `test_pin_replay.py`: `system` dropped in each of the four tools, `cacheBypass` dropped, `question` dropped, `request` dropped, `scope` dropped, `timeoutMs` dropped, cypher altered, `isError` flipped both ways, `resolve` output reformatted, exception text leaked, the `AnswerError` branch removed: **15 killed**. Survivor: skipping `render_answer` in the tool (calling `answer.dumps` directly) is equivalent, since `Client.ask` and `answer.dumps` both validate. Typed errors with `isError` hold except for F3's schema-level rejections.
2. **AC-16.** See F4.
3. **`test_no_network`.** See F1, F2. Vacuity: scanner mutants killed: scope drops `clients` or `extractors`, non-recursive walk, a walk that skips `front`, `socket` or `asyncio` removed from the banned set, a broker allow-list widened to `ccg/front/`, a top-level-only AST walk, the command check off, the dynamic-name check off, a broken relative-import base, `importlib.util` allowed (12 of 12). A planted `ccg/front/` directory is picked up (the future-front test and the `copy_root` fixture, both proven by the skip-front mutant).
4. **Store auth.** See F5.
5. **`_safe` parity.** `test_the_client_safe_text_is_the_refresh_reports_rule` runs 310 cases at caps 120, 50, 10 and the defaults. Mutants: default cap 121, no escaping, off-by-one cut: **killed**. Fixed from Pass 2 minor 2.
6. **Interaction with F1/F2/F5/exit codes.** See F5; exit 2 for every store-auth case, verified live.
7. **`.mcp.json` proposal.** The proposal text is not in the tree (only in the coder's report); I checked its likely claims instead. (a) `cwd`: Claude Code's MCP page lists `command`, `args`, `env` for stdio and does not document `cwd`; I could not confirm support from the first 100,000 characters of the page. It is not needed: `.venv/bin/python -m clients.mcp.server` imports from `/` through the editable install (checked). Prefer the `cypher` entry's pattern (`bash -c 'exec "$CLAUDE_PROJECT_DIR/..."'`), and note the docs document the braced `${CLAUDE_PROJECT_DIR:-.}` form only. (b) `test_atrest._collect` judges every `command`, `args`, `env` value, `url` and `cwd` of the `ccg` entry that contains `/ ~ . $` against the kit root, unless it is an exact member of `MCP_LAUNCH_ALLOWLIST`; so P1-27 must add the exact string(s) of the launch line to that set in the same change, and any env value that is a path (e.g. `CCG_CREDENTIALS_DIR`) is also flagged and needs its own entry. The allow-list is empty and `.mcp.json` has no `ccg` entry today, so nothing is exercised yet.
8. **`SAFE_TOOL_PARAMS` addition: right, does not weaken the guard.** `question` (free text), `cacheBypass` (bool), `timeoutMs` (int) name no file, and each is a real parameter of the shipped tools. Verified in a copy: removing `timeoutMs` from the set fails `test_guard_real_allow_lists_stay_inside_the_kit_root`; adding a tool parameter `path` or `outputFile` to `refresh` or `raw_query` also fails it. The set stays name-based and fails closed on any new name (types are not judged; pre-existing).
9. **README row.** Accurate for the tools, arguments, env, exit-on-bad-configuration, `store-auth` and tests. Add the AC-16 raw-leg coverage (F4) and, for the P1-26 implementer, that the scope bans `asyncio`/`anyio`/`socket` imports in `ccg/` and `clients/` (the ledger note exists only in `docs/plans/connected-code-graph-coordination.md`).

## What's solid

- `Client` stays the one place rules live; the server has no logic beyond mapping, so the CLI and MCP cannot drift (`dump_json` is the shared form, and `main.py`'s change is one line).
- Pass 2's three minors are all closed: message-leak test (M51) for the `reader` and `refresh` paths, the `_safe` parity test, the distinct `store-auth` code. Verified live.
- The pin tests are not vacuous: a flipped direction changes facts, a `builtAt` move does not, a dropped item does, and the item-set key sees status and every chain hop; the `system` argument is asserted to reach the client on every leg.
- The stdio server is exercised end to end through the `mcp` SDK client (tools listed, `not-built`, a typed `store-error`) and a bad configuration exits 2 at start.
- The import-boundary test walks the trees (so `ccg/front` needs no edit), requires the module to be clean before each plant, and 12 of 12 scanner mutants die.

## Open questions

- **Q1.** Is the AC-16 raw leg's narrower coverage (F4) acceptable for P1-25 with the qualifier recorded in the README and the P1-38 report, or does `teco` want the plan's P1-25 (6) and P1-42 owner list amended (architect) so nobody reads "AC-16 closed" off this step? I assumed the first.
- **Q2.** F2's wider banned set changes what P1-26 (`ccg/front`) may import; confirm the bans are intended before tightening, since `asyncio`/`anyio` are already banned there.

### Proposed REVIEW.md lines

Target: `cpg/REVIEW.md` (to be created), heading "Always check". Source: F1 and F2 of this review (a path-limited scanner whose named mutants all use one spelling).

- A static import or call scanner is checked with one planted source per import spelling the language allows (`import a.b`, `from a import b`, `from . import b`, `import a as c`, inside a function, via `getattr`/`eval`), for each banned module family and each grant. Mutants that share one spelling are a bound, not proof.

## Pass 2

**Verdict: approve with suggestions.** F1-F5 are fixed. Re-run: `test_pin_replay.py`, `test_clients.py`, `test_atrest.py`: 266 passed, 4 deselected; `test_no_network.py`: 376 passed; `-m live tests/test_pin_replay.py` on my own throwaway `ccg-falkordb-tw-16391`: 4 passed. Throwaway torn down; `docker ps` shows `-16390` and `falkordb-dev` untouched. Mutants ran in scratch copies (deleted), real tree untouched.

| Finding | Disposition | Evidence re-run |
|---|---|---|
| F1 major | **fixed** | Planted: `from ccg import broker` in `ccg/answer.py`, `from . import broker`, `from .. import broker` (in `ccg/engine/`), and both in `ccg/refresh.py`, plus `import ccg.broker as b`, `import ccg; ccg.broker.client`, `from ccg.broker import *`: all `broker-import`. In `ccg/front/interpret.py` and `summary.py`, `from ccg.broker.client import X`, `from ..broker.client import X`, `from ccg.broker import client [as bc]` pass; `from ccg import broker` fails there. `from ccg.broker import client` passes in `ccg/refresh.py`. Mutants: `_imported_names` reduced to the module, attribute-chain check off: both killed. |
| F2 minor | **fixed, with residue below** | Re-ran my 72 evasions: 7 still pass, all benign or the documented limit (`email`/`smtpd`, `psycopg`/`pymongo`, bare `multiprocessing`, `sys.modules.get(...)`, `ping`, and `__import__` through the `importlib.metadata` alias, which has no import). `_socket`, `selectors`, `ctypes`, `wsgiref`, `runpy`, `pkgutil`, `imp`, `zipimport`, the web frameworks, `getattr(__builtins__...)`, `eval`/`exec`/`compile`, `sh -c 'true; curl'`, `FOO=1 curl`, `timeout 5 curl`, `python -c`, `git clone/fetch/push`, `pip install`, `'cu'+'rl'`, a `git clone` argv kept in a variable then launched: all caught. 14 of 16 new scanner mutants killed (survivors in F7). |
| F3 minor | **fixed** | Arguments are `Any`. Through `call_tool` with the real `Client`: `cacheBypass` `"maybe"`/`1`/`"yes"`, `timeoutMs` `true`/`"5"`/`2.5`/`2.0`, `cypher: 5`, `scope: "abc"` or `[1]`, `request: "str"` or `[1]`, `anchor: "x"`, `question: 5`, `system: 123` or `["a"]`: all `bad-request`, `isError`, the CLI's message. Mutants typing each of `timeoutMs`, `cacheBypass`, `scope`, `request` back: all killed. |
| F4 minor | **fixed** | The live world now has a `possible` `CALLS` edge (`F`, `shell-dynamic`); the live raw leg expects it. Driver mutants (`row[7]` read as never possible, always possible, column 6): all killed (the earlier survivor is dead). README row now states the leg is "narrow by design" and lists the shapes. |
| F5 minor | **fixed** | New subclass test; the exact-type mutant is killed. The mid-session auth note stays a note. |

### New findings

**F6 (minor, verified): FastMCP behaviours the `Any` fix leaves, and their cost.** (a) A missing required argument (`resolve {}`) is still FastMCP's pydantic text, not `{"error": ...}`; the server docstring says so. (b) An argument sent as a JSON *string* holding an object/array is parsed first: `refresh {"scope": "[\"a\"]"}` and a JSON-string `request` are accepted, a scalar string is not (checked; documented by the coder). Not a bypass: the parsed value then goes through `Client`'s validation. (c) The cost of `Any`: every tool's `inputSchema` now has no types (`{"cypher": {"title": "Cypher"}, ...}`), so a calling model sees names and docstring only. The docstrings carry the shapes, so acceptable; the alternative is a typed schema plus a wrapper that converts `ToolError`. (d) `system: Any` is falsy-coerced by `resolve_system` (`explicit or env`): `system: ""`, `{}`, `0` silently select the default system rather than `bad-request`. A nit; `Client` could test `is not None`.

**F7 (minor, verified): the new scanner rules have two unpinned arms and one over-narrow exemption.**
- Mutants that survive 376 tests: the concatenated-literal branch off (`msg = "cu" + "rl -s http://x"` in a bare assignment, not inside a launch, is then missed; only the bare-name and in-launch shapes are pinned) and `WRAPPERS` emptied (an argv-shaped list such as `["git", "clone"]` kept in a variable and launched later). Add a mutant row for each shape.
- `COMMAND_PROSE` is exact (module, text), and does not exempt the same text in a launch, an argv, another module, or with a suffix (all re-planted: still `git fetch`). It is narrow. Abuse remaining: in `ccg/manifest.py` the exact message assigned to a variable and passed to `os.system(CMD)` passes, and that string would run `git fetch`. It needs a deliberate edit of that module, and indirection through a variable is the scanner's stated limit; if wanted, restrict the exemption to a `raise ManifestError(<that string>)` node.
- Prose false positives: any multi-word string holding a program word and a subcommand word (`git` + `pull`, `go` + `get`, `python` + `-c`) is flagged, and the dynamic-name rule flags `from re import compile; compile(...)`. The current tree has none besides the one exempted line, so nothing fires today. For P1-26: the broker client in the three granted modules (all spellings) is reachable, `urllib.parse` and `re.compile` are fine; keep prose that mentions a command out of its error messages or add a `COMMAND_PROSE` row.

### Same-round fixes against each other
`Any` typing and the F2 bans: no interaction; `server.py` still passes the scanner and `test_atrest`'s `_mcp_tool_patterns` (names, not types, are judged). `Any` and FastMCP's JSON-string parsing: see F6(b). The F2 string rule (`__import__`/`import_module` constants) and the tree: the full tree scans clean (376 passed), and only an exact string equal to those names fires.
