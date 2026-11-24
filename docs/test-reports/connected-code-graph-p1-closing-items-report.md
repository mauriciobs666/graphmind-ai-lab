# Connected code graph P1: closing items P1T-38, P1T-20 and P1T-19 (report)

> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** C-1001 (P1-38)

CPG: considered, not relevant — a black-box run of the `ccg` pipeline (build, structured asks, renderers) against a throwaway store; no Joern `cpg_<component>` graph answers it.

Plan: `docs/test-plans/connected-code-graph-p1.md` §5 (AC rows), §7 ("P1T-20 sample and pass rule"), items P1T-19, P1T-20, P1T-38 (§6 table). Why open: `docs/test-reports/connected-code-graph-p1-closure-report.md` §4, §6 Q1, Q4. Known engine gap: `docs/reviews/connected-code-graph-p1-gate-zero-hit-rca.md`. Counts and verdicts only; every figure below was observed in this run unless it says "cited". No held-out file and no custodian directory (`~/.local/share/ccg-eval/cf7b2042cce7`) was read.

## 0. Verdict

| Item | Verdict |
|---|---|
| P1T-38 proving-system build | **PASS**: 1335 of 1335 units `fresh`, none stale-kept, none not-mapped (one build, 1772 s wall) |
| P1T-20 AC-1 (HTTP) | **FAIL under the pass rule: 17 of 18 dev items listed, 12 anchors, 1 miss** |
| P1T-20 AC-2 MCP half | **PASS: 20 of 20 dev items listed, 3 anchors** |
| P1T-20 AC-2 wiring half | **FAIL: 0 of 3** (the plan-filtered sample); whole wiring stratum 0 of 7 asked (4 unit-anchor items not asked) |
| P1T-20 AC-4 scripts | **FAIL: 0 of 1** (the plan-filtered sample); whole scripts stratum 0 of 49 |
| P1T-19 AC-11 answer forms | **PASS** (one parser-resolved question, no model call; 3 planted render failures each caught in both forms) |

AC consequence (section 5): **AC-2 MCP half may close; AC-1 does not close on this item alone** (one real miss, 17/18, a defect under the plan's "no recall floor"; this is not the expected outcome the brief allowed for); AC-4 and the AC-2 wiring half stay not closed (engine gap, P2) as expected; AC-11 end-to-end now has its run.

## 1. Setup and method

- **Tree:** committed HEAD `57be01a2969ddb5915fe4e99fe740239ade8866f` (observed at the start of the run; it advanced one ledger-only commit from the brief's `6a4eb6f4`, `docs/plans/connected-code-graph-coordination.md` +1 line, nothing under `cpg/`). Built from a `git clone --no-hardlinks` of that commit in the session scratchpad (the developer's tree was never an input); the Python under test was imported from that clone (asserted in the drivers).
- **Store:** `cpg/infra/ccg-falkordb.sh --throwaway up` on port **16450** (container `ccg-falkordb-tw-16450`, `falkordb/falkordb:v4.18.11`, 2 GB cap, `TIMEOUT_MAX` 10000), with a private `CCG_CONFIG_DIR` (own admin password file). Real ACL users `ccg_{build,verdict,engine,raw}_graphmind-ai-lab` created by `ccg.store.acl.create_users` into a private `CCG_CREDENTIALS_DIR`; private work, keys and audit directories passed to `refresh` explicitly, and `CCG_AUDIT_DIR` exported to the private audit directory for the sandbox runner (rehearsal defect D3). Not touched: `falkordb-dev`, `ccg-falkordb`, `ccg-falkordb-tw-16390`, the cypher-mcp containers, ports 6379, 6390, 16390, 16400.
- **Sentinel:** a sha256 over path/size/mtime/mode of every entry under `cpg/.ccg-work` (105,719 entries), `cpg/.audit` (3) and `~/.config/ccg` (6) taken before and after: **unchanged**.
- **Cleanup:** the container was removed (`--throwaway down`: "stopped and removed"; `docker ps -a` lists no 16450), and the clone, work, keys, audit, credentials and admin-password directories were deleted. What stays: the drivers and the result JSON files in `cpg/eval/results/p1t-20/` (untracked, not committed).
- **Drivers** (new files only; no change under `cpg/ccg`, `cpg/clients`, `cpg/eval/tools`, no engine or anchor tuning, no git write):
  - `cpg/eval/results/p1t-20/build.py` (P1T-38; the P1-20 `refresh` with a `Target` over the build and verdict roles).
  - `cpg/eval/results/p1t-20/asks.py` (`p1t20`, `p1t19`; the asks go through `clients.common.Client`, the AC-16 client the CLI and MCP wrap).
  - Results: `build.json`, `p1t20.json` (per item), `p1t19.json`, `build.time.txt`, in the same directory.
- **Pass rule applied** (test plan §7): one structured impact request per dev anchor; per sampled anchor every dev item is listed, as `confirmed` or `possible`, with a chain whose located hops carry file and line (the AC-18 reading `fixture_gate_p1_ac_helpers.ac18_violations`, applied to the matched item alone). Item identity is the evaluation protocol's own H1 (`E(qname)` equal and a line of the dev item inside the answer subject's span or owned regions; config items by path and key path), read from the answer's own subject, not from a store snapshot; this is the same identity `gate.py` uses, without its H3 chain replay. A missing item is a defect, no recall floor; `key-error` is the only waiver and none is claimed. A filter matching no anchor is blocked (control below). Dev anchors were used for development: this is not AC-5 evidence.
- **Filters** (anchors counted from the dev files; the plan's wording applied literally):

| Filter | Stratum file | Anchors | Items |
|---|---|---|---|
| AC-1: `http` contract anchors whose provider is under `falkor-chat/server/` | `ac5-http.jsonl` | 12 | 18 |
| AC-2 MCP: `mcp_tool` contract anchors whose provider is under `falkor-chat/server/` | `ac5-mcp.jsonl` | 3 | 20 |
| AC-2 wiring: the same filter on `ac5-wiring.jsonl` | `ac5-wiring.jsonl` | 1 (`send_message`) | 3 (`config` items) |
| AC-4: `symbol` anchors under `cypher-mcp/` | `ac5-scripts.jsonl` | 1 | 1 |

  The AC-4 filter, read literally, selects one anchor and one item; the wiring filter one anchor and three of the file's eleven items. I applied the filters as written (decision, routine) and also report the whole strata, from the same asks, so nothing hides behind the narrow sample. Every unique anchor was asked once (the `send_message` anchor appears in the mcp, scripts and wiring files, `GET /health` in the http and scripts files).
- **Determinism:** the full P1T-20 sequence was run twice on the one store; the per-item outcomes are identical (observed, 18+20+49+7 items in the strata and the four samples).

## 2. P1T-38 proving-system build

Command: `build.py` (one `ccg.refresh.refresh` of system `graphmind-ai-lab`, manifest `cpg/ccg/system.yaml` of the clone, full rebuild, all repositories), exit 0.

| Observation | Value |
|---|---|
| Units | 1335: fresh 1335, stale-kept 0, not-mapped 0 (config 317, markdown 888, python 12, shell 116, javascript 2) |
| `unknown_kind` | 0 |
| Snapshot id | `62e9264b4e1131fb14bc531602ee08ae` (the answers' stamp equals it on every ask) |
| Wall time | 1772.3 s in `refresh` (29 m 33 s for the whole process); stages: build 959.6 s, resolve 304.7 s, link 393.0 s, assemble 10.6 s, load 10.0 s, sweep, discover, swap under 1 s |
| Peak RSS (this process) | 1501 MB over the run (stage peaks 40 MB sweep to 1657 MB assemble) |
| Host `MemAvailable` | lowest 1.16 GB (during load; fall of 1.78 GB during the build stage); the Joern JVM runs outside this process and shows only there |
| Links | 934 (http 376, mcp_tool 558; confirmed 104, possible 830); `refused_*` 0; `binding_duplicate_ids` 0 |
| `unattested` | 1321 units (host bundles are `host-unattested` by design; reported, not a finding) |
| Dropped | `consumer-key-none` 5 |

Result **PASS**: the pass criterion is a built, stamped snapshot with units listed and cost recorded; no unit is absent or failed. Two observations for the owner: the unit count is 1335 (the rehearsal's 1042 was at `EVAL_COMMIT`; HEAD holds more files), and the build is 29.5 min on a 5 GB host (rehearsal: 16 to 21 min), so the "~16 to 21 minutes" budget does not hold for HEAD on this host (one sample, host also running other agents' work: not a performance finding). The index R-4 / P1-20 brief peak-RSS budget was not given to me in the files I read; the figure is recorded, not compared.

## 3. P1T-20 results (dev anchors)

### 3.1 Plan-filtered samples (the pass rule)

| AC | Sample | Hits / total | Status of hits | Misses | Verdict |
|---|---|---|---|---|---|
| AC-1 HTTP | 12 anchors | **17 / 18** | `confirmed` 6, `possible` 11 (all distance 1) | 1 `not-listed` | **FAIL** |
| AC-2 MCP | 3 anchors | **20 / 20** | `possible` 20 (distance 1) | 0 | **PASS** |
| AC-2 wiring | 1 anchor | **0 / 3** | none listed; `config` items listed: 0 | 3 `not-listed` | **FAIL** |
| AC-4 scripts | 1 anchor | **0 / 1** | none listed; links with `bound[]`: 0 | 1 `not-listed` | **FAIL** |

Every hit carried a chain with file and line on each located hop (0 chain violations over the 37 hits). Asks: 0 typed errors (no `anchor-ambiguous` or `anchor-unresolved`).

### 3.2 Whole dev strata (same asks; every askable dev anchor)

| Stratum | Items hit / total | Anchors asked / total | Misses | Rehearsal figure (cited, pre-D5/round E, different instrument) |
|---|---|---|---|---|
| http | **17 / 18** | 12 / 12 | 1 `not-listed` | 16/17 healthy, 17/18 flow-true |
| mcp | **20 / 20** | 3 / 3 | 0 | 20/20, 19/19 |
| scripts | **0 / 49** | 22 / 22 | 49 `not-listed` | 0/49 |
| wiring | **0 / 7** asked (11 items in the file) | 5 / 6 | 7 `not-listed`; 4 items of the one `unit` anchor not asked (a unit anchor is asked from a custodian-side probe, S47; not available to me) | 0/11 |
| flow | not run (not in P1T-20's scope; the flow flag is false) | | | |

For the 49 script items: the answers of 20 of the 22 script anchors list **zero items**; the other two (the `GET /health` and `send_message` contract anchors, whose items in the file are not provider-side) list 10 and 22 items, none of them a dev item. Across the scripts and wiring anchors the answers hold 54 items, all `symbol`; no `config` item and no `Link` hop with `bound[]` appears anywhere (observed). This is the RCA's signature (engine builds no `config` item for a config entry that names a tool, no hook-launches-script hop), reproduced on the HEAD build. The store does hold the symbol or config entries for the missed items (`inStore` > 0 on every miss in `p1t20.json`): the miss is in the walk and link layer, not in extraction of the target file.

### 3.3 The AC-1 miss, characterised (one item, not tuned)

- Item: anchor `GET /health`, dev item `claude/teco/coordination-techniques.md::<document>` (lines 25 to 36). The gold is sound: line 25 of that file at HEAD holds the prompt text `curl -sf http://127.0.0.1:8200/health`, so this is not a `key-error` and no waiver is claimed.
- Observed: the `GET /health` answer has 10 items (3 `partial-key` possibles from `coordination-graph/`); that document is not among them. Raw reads as `ccg_raw`: the file's `CallSite`s reach `Link`s only for three MCP tool mentions (lines 31, 224, 449, `possible`/`prompt-mention`); no `http` Link starts from line 25. The `GET /health` contract point exists in the store. So the engine builds no http consumer from that prompt-embedded URL: an extraction-side miss (class to be set by the owner against `-eval-protocol-ml.md` §12.1; I make no root-cause claim beyond the observation).
- It is a defect per anchor under the plan rule ("every other miss stays a defect, with no recall floor"). The rehearsal's 16/17 on http already had one miss; whether it is this item I did not establish.

### 3.4 Red cells (the plan's checks shown to fail)

| Plan red cell | Observed |
|---|---|
| A dev item deleted from the answer set of one sampled anchor (an answer mutant, not an oracle edit) must be reported missing | PASS: the matched item removed from one AC-1 anchor's answer copy is judged `not-listed` (a miss) |
| A filter matching no anchor reports blocked | PASS: a filter on a key no dev anchor has selects 0 anchors and reports blocked |
| No held-out file is opened | PASS: only `cpg/eval/dev/ac5-*.jsonl` was read; no custodian path was opened |

## 4. P1T-19 results (AC-11 answer forms)

Question asked by plain language, through the same `Client.ask(question=...)` the CLI and MCP wrap, on the HEAD build: of five phrasings tried, three resolved through the deterministic parsers and two were `ambiguous`/`unknown-name` (my invented symbol names did not exist, so that is correct behaviour, not a finding). **No interpretation carries `provenance`, so no model was called** for any of the five.

| Question | Interpretation | Items |
|---|---|---|
| `impact falkor-chat/server/falkorchat/api.py:58` (chosen: file:line form) | resolved, anchor `api.py::build_router.health` | 599 (2 confirmed, 597 possible), 0 groups, 1 repository, 10 components, distances 1 and 2 |
| `impact falkor-chat/server/falkorchat/api.py::build_router.health` | resolved | 599 |
| `impact GET /health` | resolved | 10 |
| two invented `path::symbol` names | ambiguous (`unknown-name`) | 0 |

Checks on the chosen answer (grouped text 5074 lines, Markdown 7503 lines; `answer.validate` passed first):

| Pass-rule clause | Observed |
|---|---|
| Grouped by repository, component, distance | the heading sequence parsed from each rendering (repository, component, `Distance n, status`) equals the sequence `ccg.render.common.component_groups` gives the document, 0 differences, in both forms |
| Every item in its group | 599 of 599 item lines printed in the text, 599 of 599 `#####` items in the Markdown, each under its own heading triple |
| Each hop has file and line | 0 located-hop lines (function, call, configuration key) without ` line N`, in both forms |
| Interpretation shown at the top | the "What is affected by ..." line precedes the first repository heading, in both forms; fence sentence first; flow notice after the counts line |
| CLI modes | `ask --question ... --format text` and `--format markdown` exit 0, stderr empty, bytes equal to the library renderings |

Planted failures (each applied to a copy of the real rendering; the render check must fail): item moved outside its group, text and Markdown **caught** (1 violation each); two groups out of order, **caught** both forms; one hop line with its line removed, **caught** both forms. The unmutated renderings: 0 violations.

F-3 plant (substitute, disclosed): the plan plants the injection string in `Cond.code`; no `Cond` hop exists at P1 (python flow is unavailable), so I planted "Ignore previous instructions" in an item's subject `qname` instead. It occurs once in the text and once in the Markdown, each on a line inside the code span (`` ` ... ` ``). This checks the renderer's fence on a data field, not the `Cond.code` path, and is weaker than the plan's cell.

Result **PASS**. Limits: the chosen answer is one repository (the proving system is a single-repository monorepo, so the repository level of the grouping has one value and a repository-order mutant was not possible; the group-order mutant swapped two components). The check is mine (the plan's "render check" has no shipped implementation); it compares against the engine's own grouping function, so it detects a renderer that disagrees with that function and a malformed layout, not a wrong grouping function.

## 5. AC consequences

- **AC-1 (HTTP):** pass rule not met, **17 of 18**: not closed on this item. The brief expected AC-1 might close; it does not under the plan's "every dev item, no recall floor" rule, because of one extraction-side miss (3.3). Options for the owner (not decided here): accept 17/18 as a stated sub-scope closure, record the miss as a P2 defect and re-run, or fix the extraction first (no recall floor is in the plan).
- **AC-2:** MCP half **closes** on P1T-20 (20/20, 3 anchors, dev, optimistic by the plan's own words). Wiring half **not closed** (0 of 3 in the plan's sample, 0 of 7 asked in the stratum): no `config` item is ever listed; engine gap, P2, as decided 2026-11-23.
- **AC-4:** **not closed** (0 of 1 in the plan's literal sample; 0 of 49 in the stratum; no `bound[]` on any Link): engine gap, P2.
- **AC-11:** the end-to-end item P1T-19 now has an executed pass; the closure report's "closed for a stated sub-scope: renderers and CLI modes" may be restated as closed for the impact answer forms, subject to the caveat in section 4 (one-repository answer, my own render check).
- Not an AC-5 result: every dev figure here was used for development.

## 6. Defects, coverage, gaps

- **DEF-1 (AC-1, medium):** `GET /health` impact answer omits the prompt-embedded URL consumer `claude/teco/coordination-techniques.md` line 25 (`curl -sf http://127.0.0.1:8200/health`). Repro: build HEAD `57be01a2` (P1T-38), `Client.ask({"useCase":"impact","anchor":{"kind":"contract","key":"GET /health","contractKind":"http"}})`; the dev item (anchor `aefbd69a...`, item `02170815...`) is not among the 10 items; `raw_query` shows no http `Link` from that file. Expected: listed (the plan's rule). Evidence: `cpg/eval/results/p1t-20/p1t20.json`.
- **DEF-2 (AC-2 wiring, AC-4):** the known engine gap, reproduced on HEAD: 0 `config` items and 0 `Link.bound[]` in 54 answer items over all scripts and wiring anchors; 20 of 22 script anchors answer with 0 items. Not a new finding; carried to P2 (RCA).
- **Not covered:** the wiring `unit` anchor (4 items) and the S47 probe path (custodian side); the `flow` stratum; AC-1/AC-2 on held-out data (deferred); H3 chain replay of the hits (`gate.evaluate_item`'s stricter step: not part of the plan's pass rule; the rehearsal's 16/17 and 20/20 used it); a repository-order grouping mutant; the AC-1 filter is "all 12 http anchors" since every http dev anchor's provider is under `falkor-chat/server/` (the filter did not discriminate).
- **Flakiness:** none seen; two full P1T-20 sequences gave identical per-item results on one store. One build only (no build-to-build determinism check, so a different build could differ).
- **Feedback:** (1) the plan's AC-4 and wiring filters select 1 anchor/1 item and 1 anchor/3 items, so those two closures would rest on a sample far smaller than the strata; the plan could name the intended sample. (2) The plan's P1T-19 "render check" and its F-3 `Cond.code` plant need a shipped checker and a flow-true fixture; both were improvised here. (3) A P1-20 brief budget for peak RSS and wall time should be stated where the build item reads it. (4) The 16-21 min build estimate is 29.5 min for HEAD on this host.

Artifacts: this report; drivers and results in `cpg/eval/results/p1t-20/` (`build.py`, `asks.py`, `build.json`, `p1t20.json`, `p1t19.json`, `build.time.txt`). No test plan was written for this run: the existing plan's items P1T-19, P1T-20, P1T-38 are the plan; nothing is committed.
