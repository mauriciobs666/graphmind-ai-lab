# Connected code graph, P1-47: runtime-oracle capture gate

> **Status:** active · **Owner:** `analyst` · **Tracks:** P1-47

## Pass 1

**Scope.** Commits `03f6f8a1`, `49e91814`, `e3f94d33`, `2152e5b7` (tooling, runner, session, records `ccg-oracle-v1..v3`) and the capture
`run2` under `~/.local/share/ccg-eval/cf7b2042cce7/oracle/runs/run2/`, against `docs/plans/connected-code-graph-p1-gates.md` (P1-47),
`docs/plans/connected-code-graph-p1-ml.md` §6.5, root `REVIEW.md`. Counts only; no held-out data read. Nothing under `cpg/` was edited.
`CPG: considered, not relevant — the questions are tool behaviour, a recorded run and a Docker fence, answered by execution; the cpg_* graphs do not model them`

**Verdict: approve with suggestions.** No blocker: the capture is reproducible from its record, the run used the recorded files, the
fence held in the run, and the attribution is honest about its grain. Three majors (M1-M3) are things P1-36 or the next capture must
not inherit silently.

### Questions 1-6, answered

1. **Pins and the recorded run (verified).** `sha256sum` of `oracle_session.mjs`, both shims, `oracle_capture.py`, `oracle_runner.py`,
   `test_oracle_capture.py` and both e2e specs equals every v3 pin; the spec hashes also equal `git show cf7b2042cce7:` content; the
   served bundle and its map in `oracle/spa-dist` equal the recorded sha256s, and `server.out` serves `index-BiENeKUg.js`. `git log`
   shows `2152e5b7` is the last commit touching `cpg/eval/tools/` and the test; every tool file's mtime precedes the run's first
   output (21:32:34) and the record commit is 21:32:26. `session.out` step names are v3's (S08 "round 2 ... one retry", S10 reset
   last), 13/13 ok; `seed_salesperson.out` shows port 16400 (own throwaway, none left in `docker ps`). I recomputed
   `attribute-spa` (byte-identical to `spa.jsonl`), `normalize` (byte-identical to `edges.jsonl`), `all.jsonl` = `py-sh` + `spa`, and
   `summary` = 1038 / 348 / 690 / 0 / 18 / 17. The export's `endpoints.ts` equals the commit's.
2. **Fence.** See M1. `check_fence` refused: unset host/port, `0.0.0.0`, 6379, 6390, a container missing either label (planted live on
   16401 and 16403), and a port with no container. The runner-owned path additionally verifies labels and the 127.0.0.1-only bind
   (`verify_started`); that path is what `run2` used. Offline: `tests/test_oracle_capture.py` 27 pass.
3. **Attribution.** See M2, m1, m2. Python/curl frames probed against source (`load_append.py::_worker` = `client.post` line 60,
   `::main` = lines 90/93/99/153, `load_test.sh`): correct. SPA frames: correct at file grain, wrong-by-wrapper at function grain.
4. **Coverage.** See M3 and the gap table below.
5. **Numeric segments.** See M4: no effect on this capture's in-scope edges; a real hazard for the scorer.
6. **No held-out contact (verified).** No tool, shim, session or runner source mentions held-out paths; importing `oracle_capture` +
   `match_keys` and calling `normalize_contract` under `sys.addaudithook` opens no non-`.py` file; `heldout/` is a sibling of
   `oracle/` under the custodian root and is not in the export (`git ls-tree cf7b2042cce7` has no held-out path).

### Findings

**M1 (major, verified live) The fence is weaker than the record says.** Record v1-v3 `fence`: "never 6379/6390/16390; enforced by
`oracle_capture.py`". `FORBIDDEN_PORTS = (6379, 6390)` (`oracle_capture.py:35`); 16390 is absent, and the only other test is "some
container with both labels publishes this port". Run on the live host: `FALKORDB_HOST=127.0.0.1 FALKORDB_PORT=16390 oracle_capture.py
fence` returns `ok` (container `ccg-falkordb-tw-16390`, which carries both labels), and a planted both-labelled container published on
`0.0.0.0:16402` also passes (`check_fence` does not inspect the bind; only the runner's own `verify_started` does). So
`oracle_capture.py run` and `oracle_runner.py --external` would run the suites against another session's throwaway. `run2` did not
(own container on 16400, removed). No test names 16390. Fix: make the runner own the proof, not the label: give the runner's
container a per-run label (`ccg-oracle-run=<run-id>`) and have `check_fence` require it and the name; also add 16390 to the
forbidden set and a test case per record-named untouchable. Do this before any further capture run; it does not invalidate `run2`.

**M2 (major, verified) SPA edges are file grain, and three routes are attributed to a shared helper.** Every SPA row carries
`{"file": "salesperson/src/api/endpoints.ts", "func": null}`. Mapping the stacks through the source map (my own script over the same
decoder) puts `GET /shop/api/state`, `/messages` and `/catalog` at `endpoints.ts:89`, which is `participantGet`, not
`getState`/`getMessages`/`getCatalog`; the other eight routes land on their own function. The static consumers are function symbols
(`salesperson/src/api/endpoints.ts::getHealth`, `cpg/tests/test_answer.py:162`). The plan only promises "the first frame in
`endpoints.ts`", so this is within the plan, and the tool emits `func: null` rather than a guess. But an oracle scorer that matches
`(element, key)` exactly scores every SPA edge as a miss. P1-36 must (a) match SPA rows at file grain, with that rule written down and
pinned by a planted function-grain Link, and (b) report these 11 distinct edges as a grain-limited class, not pooled with Python's
function-grain edges. Cheaper alternative: resolve the enclosing exported function from the source line (the map gives the line; the
function bounds are in the export's `endpoints.ts`) and skip `participantGet`; that needs a new record version, so prefer (a) unless
function grain is wanted. The same innermost-in-tree rule gives `transport.py::_transport` for the LM Studio rows (a shared closure
from `make_http_transport`), so a miss there is also an attribution artifact.

**M3 (major, verified by sweep) Entry points are skipped silently, which §6.5 forbids ("run, or listed as unrunnable, never skipped
silently").** An unfiltered sweep of the EVAL_COMMIT tree (non-test `.py/.sh/.js/.ts` files matching `curl|urllib.request|httpx|
requests.|http.client|fetch(|urlopen`) finds HTTP clients that are neither in the record's `run` list nor in `listedNotRun`:
`falkor-chat/web/app.js` (the browser client of `/channels`, `/threads/{id}/messages`: the largest consumer after the SPA),
`salesperson/scripts/load_demo.py` (a `/shop/api` client with a `stub` mode that needs no model), `salesperson/scripts/swap_bench.py`,
`language_salience_probe.py`, `qwen_watch.sh`, `install_node.sh`. (`falkor-chat/server/falkorchat/services.py`/`config.py` match on
`urllib` for LM Studio, not as entry points.) The oracle's denominator is observed edges, so this under-covers rather than biases a
ratio, but the listed-or-run rule is the plan's honesty control. Fix: the P1-36 report lists these as unrun with reason (they are not
in the record, and the record is not amended in place), or one more run phase runs `load_demo.py stub` and a Chrome visit of `/` for
`app.js` on the throwaway. Either way it is stated, not omitted.

**M4 (major for the scorer, verified on the runtime side, inferred on the static side) Unmasked short ids.** `normalize_contract`
masks a segment only if `ccg.bundle.screen_fails` fires (T1-T6: long mixed/hex/UUID tokens). `GET /api/workspaces/123/threads`,
`/threads/t_abc123/messages`, `/channels/ch1/threads` stay literal (run: all three unchanged); a UUID becomes `{?}`. This is the
shared, pinned screen, so the capture must not change it. In **this** capture it does not bite the in-scope class: the 17 app-class
keys are `/shop/api/*` constants, `/v1/chat/completions`, `/health`, and `/channels`, `/channels/{?}/threads`, `/threads/{?}/messages`
(UUIDs). It does bite the 690 testclient rows (`/documents/d1/history`, `/matches/m1/confirm`, `/workflow-runs/r1/...`), which are
outside the denominator, and any future capture with numeric ids. A static consumer key for `` `/workspaces/${id}/threads` `` is a
placeholder form (`{?}`), so exact-string comparison fails on a runtime `123`. Owner: **P1-36** (`oracle.py`), not the capture: compare
per segment, where a static `{?}`/`{}`/`{#}` matches any runtime segment, reading raw paths from `all.jsonl` (`edges.jsonl` keeps only
`keyNorm`). Pin it with a planted `/workspaces/123/threads` against `/workspaces/{?}/threads` and a literal mismatch that must stay
unmatched.

**m1 (minor, verified) The `app` class mixes storefront and non-storefront edges.** The 348 app rows / 18 distinct edges are: 11
`spa` edges (host/port null; routes are the eleven `/shop/api` routes, all reached), 5 `load_append.py` + 1 `load_test.sh` edges
to the throwaway server (loopback, port 16500), and `transport.py::_transport` `POST /v1/chat/completions` to `localhost:1234`
(LM Studio, 21 rows). They are separable by `src` and `(host, port)` in `edges.jsonl`, but nothing marks scope and the summary
counts them together. P1-36 fixes the in-scope rule (which of the 7 non-SPA edges are "the system's HTTP API") before it sees
Link data, and reports 18 as "observed", not as the denominator.

**m2 (minor, verified) `testclient` is decided by `host == "testserver"` only.** `sitecustomize.py:51`. A suite test that uses a real
socket would be `app` with a `/tests/` frame. None occurs in `run2` (no app row has a frame under `tests/`; checked), but the guard is
by host, not by frame. Cheap hardening: also classify a frame under `/tests/` as `testclient`; one planted test row.

**m3 (minor, verified) The run is not recorded as a run.** `oracle/logs/` is empty; the runner's stdout (phase rcs, the throwaway
name, the `removed` line) was not saved. `suites` returns 4 when either suite fails, and `test_queries.out` has `460/461` (`§14.8
SUPERSEDES.matchId ... relationship index`, unrelated to HTTP), so `post` must have been run in a second invocation (`edges.jsonl`
21:38:17 vs `test_queries.out` 21:35:40). Evidence the files were the recorded ones therefore rests on mtimes and a clean tree, not a
manifest. Suggest the runner writes `runs/<id>/RUN.json` (commit, tool sha256s, phases with rc, throwaway name, start/end), and the
`suites` rc ignores a failed assertion count that is not about the capture or reports it separately.

**m4 (minor, verified) The session server binds `0.0.0.0`** (`oracle_runner.py` uvicorn `--host 0.0.0.0`) with a throwaway DB and a
random presenter key for about two minutes. Loopback suffices (Chrome is on the same mirrored network, and the run proved
`127.0.0.1:16500` is what it uses). Change to `127.0.0.1` in the next record version.

**n1 (nit)** `oracle_session.mjs` header cites "S14" and "S10 of v2" (the script has S01-S13); the stale lines do not change behaviour but
they are inside the pinned file, so fix them with the next version.

### Coverage against the plan (question 4)

| Promise (`-ml.md` §6.5 / P1-47) | What ran | Gap and ruling |
|---|---|---|
| Every `salesperson` screen and action | S01-S12 real SPA over CDP; all 11 `/shop/api` routes observed; one API client (`client.ts` is the only `fetch(`) so no route is reachable another way | Within the plan |
| `salesperson/tests/e2e/` | Not run (mocked via `page.route`); mapped to S01/S10 and S11/S12 | Stated limit, accepted: mocked specs issue the same SPA calls, and the route set is already complete; the record states it |
| Second `/session` join (post-reset language step posts `displayName ''`, 422) | Not driven (product defect) | Stated limit, accepted: same key and same file-grain element as the first join, so the distinct-edge set is unchanged; the defect should be a backlog item for `falkor-chat`/`salesperson` |
| `/order/advance` | One key, 3 calls (cancel, fulfil, deliver) | Accepted: the key is the body-independent route; the report must say the oracle cannot distinguish the three transitions |
| LLM-dependent turns | 21 LM Studio rows; orders existed (3 advances) | Accepted; v3 recorded the fallback wording |
| Non-test entry points | 4 run (`seed_salesperson.sh`, `verify_salesperson.sh`, `load_test.sh`, `load_append.py`); `listedNotRun`: 12 named scripts | The listed ones are accepted. **M3**: the unlisted HTTP clients are a gap the plan does not allow to stay silent |
| MCP-calling entry points | none observed (`POST /mcp` only as TestClient); `mcp-monitor`, `kiro`, `tank` are `listedNotRun` | Stated limit; no `mcp_tool` oracle edge exists at P1, say so in P1-36 |
| Suites under the shim | pytest 3645 passed; `test_queries.sh` 460/461 | TestClient class outside the denominator, as planned |

None of the gaps blocks P1-36. P1-36's preconditions: M2(a) and M4 pinned in `oracle.py`, the m1 scope rule fixed before scoring, M3
listed in its report.

### What's solid

- The record chain is honest: v1 committed before any capture, v2/v3 name why they supersede, `run1` is kept and called diagnostic.
- The capture is deterministic downstream of the raw logs; I reproduced all three derived files byte for byte.
- `unattributed = 0` is not a defaulted zero: frames come from `sys._getframe`/`/proc` or the source map, and a missing frame yields
  `null`, which `summary` counts.
- The runner's own-throwaway design (local image, `--pull never`, loopback bind verified with `docker inspect`, removal in `finally`,
  refusal to remove a foreign or `16390` name) is sound; `run2` left nothing behind.
- No held-out contact path exists in the tooling.

### Proposed REVIEW.md lines

Target: root `REVIEW.md`, `## Always check`, after the live-spec absolute-word bullet. Source: M1 (this review).

- A guard that fences a shared resource (a port list, a name prefix, a label) is checked against every untouchable resource its
  plan or record names in prose: each one is planted live or in a test as a case that must be refused, and a resource the guard only
  excludes by a label is checked against a foreign instance that carries the same labels.

### Open questions

1. M2: does the stakeholder want function grain for the SPA (new record version, more work) or file grain stated and handled in
   P1-36? I recommend file grain.
2. M3: run `load_demo.py stub` and `web/app.js` now (one more phase), or list them as unrun in the P1-36 report? I recommend listing;
   neither changes the denominator's honesty if stated.
