# Connected code graph — P1 plan, part 4: egress, gates and phase exit (W0 spec, W4)

> **Status:** active · **Owner:** `architect` · **Tracks:** C-1001 (P1)

Part of `docs/plans/connected-code-graph-p1.md` (decisions D1–D11, flow gate §5, egress decision §6, obligation walk
§4, step index §7, risks §10). Contracts: egress `docs/plans/connected-code-graph-egress-spec3.md`, evaluation `-eval-protocol-ml.md` + `-eval-protocol2-ml.md`, fixtures
`docs/test-plans/connected-code-graph-fixtures.md`. Briefing line for every step: builders see dev items and
fixtures only; no writes to any shared database; held-out items are touched only by P1-37 and P1-43 and only through the
harness. Test artifacts follow the family: `docs/test-plans/connected-code-graph-p1.md` and
`docs/test-reports/connected-code-graph-p1-report.md` (owner `qa-engineer`; one report, one section per gate step).

## P1-28 · egress spec successor `-egress-spec3.md` — `architect`

`Supersedes:` both egress specs (each flips to `superseded`); a consolidated, current-design document, not a patch
list, written for **option C** (index §6): no in-system git fetch.
- **Process split for the structured-call path:** one host process owns policy, run tokens and the **single audit
  writer** and serves structured calls (it must reach `local` endpoints such as LM Studio through mirrored-network
  `localhost`). There is no proxy container, no `bind_proxy`, no reaper and no per-invocation network in P1; the
  document records them, with the old `R2` mitigations (fixed-argv runner, read-only credential, no hooks, bare
  mirror, byte bound) and the TLS-interception option, as **design inputs of the deferred git-fetch path** (trigger:
  a manifest first naming an https `remote`), not as P1 text.
- The content of `cpg/ccg/schemas/egress-v1.schema.json` (the allow-list schema; the file is created by P1-03) **without** a
  `remote` field, the run-kind × component table (§2.4a), and `R2`, `R8`, `R9` stated as **not implemented in P1**
  (a deny with a reason each, in the audit's closed vocabulary).
- The `sandbox` ↔ bundle loader check (`sandboxAuditIds`, `-egress-spec3.md` §5.4) as P1-18/P1-19 implement it.
- AC-8's P1 scenario: the network has no route out; the logging proxy is a **detector** whose every `conn` record
  fails the run (index §6), and which rows are vacuous.
**Gate:** `analyst`; `security-expert` advisory on the structured-call trust boundary and the audit writer.

## P1-29 · broker core — `devops`

- **Files:** `cpg/ccg/broker/__init__.py`, `policy.py` (mode resolution and the rule table `R0`, `R1`, `R3`–`R7`, `R99`,
  first match wins, scope set per run kind; `R2`, `R8`, `R9` are deny rows with a reason), `runs.py` (`open_run`, `close_run`, tokens), `audit.py` (`policy`,
  `decision`, `close` records; append-only; a broker that cannot write an audit record denies the request; the
  `sandbox` record's check is P1-18's `ccg/sandbox_audit.py`), `client.py` (`BrokerClient`),
  `cpg/ccg/schemas/egress-audit-v1.schema.json`,
  `structured.py` (`request(EgressRequest, body) -> Decision + response`, closed body shape for
  `repo-names-and-identity`), `server.py`, `cpg/tests/test_broker_policy.py`, `cpg/tests/test_broker_audit.py`, and `cpg/pyproject.toml`: the `broker` extra
  becomes `jsonschema==4.26.0` and `pyyaml==6.0.3` (the `dev` pins verbatim, no new pin; `-egress-spec3.md` §10.2),
  with the file's header comment (lines 9–10, "No other step edits this file") and the `broker = []` comment (line 53,
  "stdlib only unless P1-28 decides otherwise") reworded in the same edit so the file does not contradict itself;
  index D8 permits this edit. Two dispatches (policy+runs first).
- **Behaviours (each row of `-egress-spec3.md` §8.2 that needs no proxy is a unit test over audit records; the
  spec's §10.1 tests (10)–(20) are added to this list; `teco` re-checks the dispatch split):**
  (1) `open_run(runKind, systemId, codebases | "*", override)`: a `vulndb` run has no codebase, `refresh`/`answer`
  at least one; the token is 256-bit random, held only in broker memory and the run's environment, expires at
  `close_run` or 24 h; (2) mode resolution: local-only default stays local-only, a hosted override on a local-only
  codebase is refused and logged (`P2-OVERRIDE-REFUSED`, row 8.10), a hosted-default codebase with a local override
  stays local (`P3-OVERRIDE-LOCAL`, row 8.11); (3) every model call in a system containing any local-only repository
  is local, interpretation included (`R5`, rows 8.7 and 8.9); an answer touching a local-only repository is entirely
  local by run scope (row 8.8); (4) `R4`: a body with any field beyond `{repoNames, identity.subject,
  identity.issuer}` is denied (row 8.3); (5) `R3` only for a `vulndb` run (row 8.4); (6) a request with no token,
  an enum value outside §2.4, a component outside its run kind or `codebases = []` outside `vulndb` is `R0`
  (row 8.15); (7) canonical endpoint form (§4.3); (8) an allow-list that fails the schema refuses to load;
  (9) the audit record fields are exactly those §5 lists (a schema-validated writer).
- **Done / Gate:** unit tests carry every row they can; `analyst` plus mutation check on `R5`/`R6` precedence and the
  scope set; `security-expert` advisory. **Blocks:** P1-20 (override runs), P1-26 (model calls), P1-31.

## P1-31 · AC-8 harness — `devops`

- **Files:** `cpg/sandbox/ac8/run.sh` (the scenario's network with **no route out** and the logging detector),
  `cpg/sandbox/ac8/window.py` (the detector's start and flush markers and the `conn` verdict; no `conn` ↔ `decision`
  join in P1, `-egress-spec3.md` §6 input 9), `cpg/sandbox/ac8/assertions.py` (rows 8.1–8.16 over the audit, per
  `-egress-spec3.md` §8.2), `cpg/tests/test_ac8_window.py`.
- **Behaviours:** (1) with no allow possible for a proxied stream (option C), the main run passes only with zero
  `conn` records; a separate detector control scenario shows a `conn` (and one outside the window) fails the verdict
  (`-egress-spec3.md` §8.1); (2) preconditions P-a–P-g and the in-run controls of `-egress-spec3.md` §8.1, and each
  row computed, vacuous or unrun as its §8.2 table says (row 8.16 runs only if `devops` provisions `strace`, which this
  host lacks, and its trace parser carries §8.2's positive control); (3) UDP and DNS, which `conn` cannot record, are covered by
  P-a and P-b from each component container; (4) the sandbox-side escape probes (`-egress-spec3.md` §7.3) run inside
  the scenario.
- **Gate:** `analyst`. **Needs:** 29, 18.

## P1-40 · AC-8 part 1 run — `devops`

The scenario of proposal §5.3 on a mixed fixture system, **full refresh plus queries**: zero connections outside the
allow-list for a local-only codebase; a local-only build makes no dependency download and every fetch-capable
frontend is invoked with its fetch flags off (from `toolArgv`, row 8.6); a question touching only hosted
repositories is still interpreted by the local model (8.7); a hosted override on a local-only codebase is refused
(8.10); a hosted-default codebase with a local override stays local (8.11). Live on a throwaway instance. The
result is recorded for `qa-engineer`'s report section. **Gate:** `analyst`. **Needs:** 31, 20, 25, 26.

## P1-32 · evaluation tooling and the AC-5 gate harness — `qa-engineer`

Two dispatches; all code under `cpg/eval/tools/`, tests beside `test_eval_tools*.py`. Every function the harness
needs from `match_keys.py` is **imported**, never reimplemented.
- **32a (verbs):** `regate-list` (the next list: kept clusters plus fresh items, routing by anchor and S(u),
  EP §12.1 step 4 — an item already in dev/held-out/reserve is discarded, an item whose cluster has an anchor in dev
  or in C goes to dev, a kept cluster is joined and not counted as fresh), `exposure` (the exposure record a run
  appends), the AC-24 half of `split_manifest.py` (30 held-out and 14 dev AC-24 items, as S47 produced), and the
  scripted-step verbs that replace `s47_scripted_steps.py` (the §7.4 correction, the §10.3 join, the §10.2 AC-24
  split); each verb's test reproduces the replay record's output **byte for byte** on the stored inputs; the `capture`
  `trigger` flag is cleared after a KA2 run (the P0 nit); the **freeze-check verb** as `-p1-ml.md` §6.2 states it (`split_manifest.py check-report` once per file over
  `git diff --name-only <S47> <record>` minus the recorded exclusions; exit 2 or 4 is a failure, never a skip; hits are
  12-hex probe hashes the custodian maps back privately; a planted held-out probe gives exactly one hit), which the
  custodian runs before the record (index §7, "Before the record"); the oracle candidates are written in the key-item schema and
  passed to `check-report` by `--heldout`, since `--heldout-dir` globs `ac5-*.jsonl` only (`-p1-ml.md` §6.4).
- **32b (harness):** `cpg/eval/tools/gate.py`: verifies the held-out manifest (`eval/heldout-manifest.json`), builds at
  `EVAL_COMMIT` (a build the harness runs through P1-20), recomputes the four-field pre-registration comparison
  (`packSetHash`, `coreTreeHash`, `algorithm`, `defaultDepthBound`) and refuses to call a run "first contact"
  unless all four equal the committed `cpg/eval/preregistration.jsonl` record and `cpg/` has no uncommitted change;
  reads the **stored normalized gold and ids** and runs `match_keys.py renormalize-check` **before scoring, stopping
  the gate on any difference**; asserts C5 (the answer's `stamp.snapshotId` equals the pinned gate snapshot and
  `verdictLogOffset == 0`; `interpretation.request` equals the submitted request; each hit's chain is replayed
  against the gate snapshot — every hop's `(kind, id)` exists and an edge of `edgeType`/`dir` joins consecutive
  hops, with a fabricated-chain item as the positive control; subject span and `qname` re-read by `symbolId`; masked
  H1 comparison; every AC-24 repeat has `provenance.cacheHit == false`; the cache is empty for the gate snapshot);
  applies the hit rule H1–H4 with `E`/`h1_symbol_credit` imported; flags a contract anchor whose `cpId` is absent
  from the gate graph as an `anchor-not-in-graph` miss; implements §12.1's fail procedure with E12's floor (≥ 35
  counting fresh-cluster items) and the "all hits, floor not met" outcome; reads the pin strings recorded in
  `eval-manifest.yaml` and reports a difference from the committed documents (R-6); **pins itself:** compares the
  sha256 of the harness files with `git show <recordCommit>:<path>`, so an edit after the record makes the run "after
  adaptation" unless re-scoring with both versions shows no item's score changes (`-p1-ml.md` §6.1); every output it
  writes goes under the custodian root `~/.local/share/ccg-eval/<EVAL12>/` (`<EVAL12>` is the first 12 hex of `EVAL_COMMIT`; §6.4), and committed reports carry counts,
  hashes and classes only. Intervals use the anchor or the
  cluster as the unit (lumpy held-out, P0 disclosure). **Scope:** the gate's frame is the census scope (index §2), read from `cpg/eval/eval-manifest.yaml`'s `exclusions` at
  `EVAL_COMMIT`, not from the system manifest (a disagreement between the two is reported as a defect): the harness
  flags any listed item whose file lies in the excluded scope as `out-of-scope`, reports the count, and never scores it
  as a hit, a miss or a wrong-possible item; a **held-out** item that is out of scope is triaged `key-error` (EP §12.1,
  confirmed by the stakeholder), never dropped; P1-36's noise frame applies the same filter (`-p1-ml.md` §6.3).
- **Gate:** `analyst` runs the verbs and the harness against planted inputs; `data-scientist` for §12.1 and the
  four-field comparison; mutation check on the routing rules and the floor.

## P1-33 · P1 test plan — `qa-engineer`

`docs/test-plans/connected-code-graph-p1.md`: risk-based, from the fixtures plan's rows (P1 91 of 132; no `fx-lib`
row; TP-014) and the ACs of the P1 row, listing for each row the build it needs and the step that supplies it.
**Gate:** `analyst`. **Needs:** 00.

## P1-41 · fixture gate, part 1: the P1 rows — `qa-engineer`

Execute, on the materialized fixture system (`cpg/fixtures/materialize.py`) built through P1-20: the **91 rows
whose earliest gate is P1** (`docs/test-plans/connected-code-graph-fixtures.md` §6) — every link kind and status P1
builds, including LNK-19/LNK-20 (P1-46's pack), the cross-repo Compose context → unit case (`LNK-15`, `OV-2`,
`-p0-compose-binding.md`), the 8000/8200 URL binding with the example config `possible`, mount composition, the
AC-1 lifting chain (field path, two levels), `ownedRegions` and multi-candidate links (first real cases),
stale-kept and not-mapped (AB-*); **B1's four assertions**, those the fixtures plan gates at P1 (writer W's change
lists reader R; W does not list another writer W2; the RCA assertion cannot run before P5). **Prerequisites checked
first:** P1-02 and P1-06 (F-1, F-9) done; the build ran with `header.history` available. Live on its own throwaway
instance. Rows asserted but red are defects, reported by id. **Output:** the report section. **Gate:** `analyst`.

## P1-42 · fixture gate, part 2: AC-level assertions — `qa-engineer`

**Proving-repo asks on dev anchors** (`cpg/eval/dev/ac5-*.jsonl`, never held-out): AC-1 (changing a `falkor-chat` server endpoint lists its storefront callers with chains), AC-2 (changing a `falkor-chat` MCP tool lists its callers, prompts and wiring, the wiring entries as `config` items per `-answer-spec2.md` §3.3), AC-4 (changing a `cypher-mcp` tool lists the agent and skill scripts and the launch chain, the chain as the `Link`'s `bound[]`, spec2 §3.2), each with its chain; the held-out strata of P1-37 are the AC-5 evidence. AC-6's P1 kinds (runtime-built URL, dynamic import, example config, prompt mention; ≥ 3 planted instances per kind
in different syntactic shapes, none `confirmed`, the planted consumer's own item carrying the kind's edge-level reason, not `flow-unavailable` (`-answer-spec2.md` §5), TP-009); AC-25's planted fixtures **in the form index §5 states** (the data-flow effect as a caller chain, `possible`; for the control-dependency effect, the callers of the function holding the changed condition and, if P1-39's shipped walk lists it, the guarded point `possible`, otherwise the guarded-code item reported not closed); the reverse question; AC-9 stamp; AC-16 pin-and-replay across MCP, CLI,
structured echo and raw recipe; AC-17 for code location and contract point; AC-18 for impact answers (every item
with its chain, file and line); AC-21 (a write through `raw_query`, and a direct `GRAPH.QUERY`/`GRAPH.PROFILE` with
the `ccg_raw_<s>` credential: refused, stamp unchanged). **Gate:** `analyst`. **Needs:** 41, 26.

## P1-34 · flow re-measurement — `data-scientist` + `qa-engineer`

Index §5 is the design; `-p1-ml.md` question 2 fixes the numbers. Python only unless the note rules otherwise.
1. **Tooling:** `cpg/spikes/frontend-fitness/s28_run.py` hard-wires `EXTRACT = CPG/"spikes"/"vertical-slice"/
   "extract.sc"` (line 24, used at line 73), the old projection. This step adds an `--extract` argument (default
   unchanged) and records the sha256 of the script it ran in the run output; a test asserts that a run with
   `--extract cpg/extractors/joern/extract.sc` used P1-09's script and a run without it used the old one.
2. **Entry check** (cost guard): P1-09's report line, the dev recall of the fixed projection, against the entry value
   **465/481** (`-p1-ml.md` §3.1; the dev ceiling is 467). If it is not reached the step ends: `flowAvailable` stays
   false, reported not closed with the figure, and no fresh draw is spent (the expected outcome, index §5).
3. **If entered, pre-register** the draw before any label: the seed
   `ccg-p1a-v1-flow|python|<corpusCommit>` (spurious sample `ccg-p1a-v1-spur|`), 30 methods, a frame of P0-ML §6.4's population at `0caf85e3` minus the 30 P0 methods and minus
   every held-out AC-5 flow anchor (the custodian applies that exclusion and reports the count only), the
   `ccg-p1-flow-python-v1` record (and `-decision` after the run; `-p1-ml.md` §3.3, §8), tier-2 rule, the labeller brief
   (`cpg/spikes/frontend-fitness/flow-brief.template.md` with the patch notes) and the adjudication procedure (one fresh blind
   `qa-engineer` labeller, a Part F-style audit and `data-scientist` rulings; `-p1-ml.md` §3.3), committed before labelling; run
   `s28_run.py`/`s28_metrics.py` in both configurations offline.
4. **Decide** by `-p0-ml.md` §8.4: the record names the figure, the cluster interval, the omission rate and the
   spurious rate; if true, `cpg/ccg/registry/flow-availability.yaml` flips that row (a `graph-dba` edit, gated), and
   P1-42's AC-25 fixtures and P1-37's `flow` stratum are re-run in the true form.
- **Gate:** `analyst`; `data-scientist` owns the method. **Needs:** 09, `-ml`.

## P1-35 · AC-24 smoke harness and run — `qa-engineer`

- **Files:** `cpg/eval/tools/ac24_smoke.py`, `cpg/tests/test_ac24_smoke.py`; the `ccg-prereg-ac24-v1` record appended
  to `cpg/eval/preregistration.jsonl` before the run (by `teco`).
- **Protocol (EP §10.5, provisional):** the held-out AC-24 impact cells (30 items per the S47 split, 14 on dev;
  both counts are from the ledger) against the **local** model the note names. EP §10.5's thresholds are
  written for 24 unambiguous held-out items (the P0 report counts 36 unambiguous and 8 ambiguity items in total and `cpg/eval/dev/ac24.jsonl` holds 12 and 2, which leaves 24 and 6 held-out): pooled ≥ 0.92 understood (≥ 23 of 24; Wilson [0.798, 0.993] at
  23/24); per language ≥ 0.85 (≥ 11 of 12); every ambiguous held-out item understood. **A dev entry check
  runs first** (14 dev items, repeat 1: ≥ 11/12 unambiguous and 2/2 ambiguous; a failure means no held-out run).
  **Repeat 1 is the scored draw** (each held-out item runs 5 times with `cacheBypass`; repeats 2–5 give the flip rate and per-draw rate, reported only);
  `cacheHit == true` on repeat 1 scores the item "not understood" and **stops the run** (a C5 failure; on repeats 2–5 it
  voids only those repeats); outputs go under the custodian root (`-p1-ml.md` §4.3, §6.4). **The harness still re-counts
  the unambiguous and ambiguous items in the held-out 30 from the split output** and, if the count differs, the
  method owner recomputes the thresholds before the run.
  A pass shows the interpreter is not broken on the impact cells; it does **not** certify 0.92. The paired EN/PT
  probe is deferred to the full gate (P5). A failing run is reported as run; the fix goes on dev items and the
  re-run uses the dev set, not the held-out items again.
- **Gate:** `data-scientist`.

## P1-47 · runtime-oracle capture — `qa-engineer`

The runtime oracle (`-ml.md` §4.1; EP §15.1 R1) needs edges **observed at runtime**, and nothing else in P1 produces
them (`RAN` importers are P3). Following `-p1-ml.md` question 5's protocol: run `falkor-chat`'s suites and a
scripted storefront session under access-log or HTTP-client instrumentation and record the observed
`(consumer element, contract key)` edges, the key `ccg.keynorm`-normalized, to the custodian root
`~/.local/share/ccg-eval/<EVAL12>/` (`-p1-ml.md` §6.4: there is no read guard, so the location and the brief rule are the
control). The code is exported with `git archive` at
`EVAL_COMMIT` into that root. Attribution (§6.5): the exported SPA's API client records the first frame in
`src/api/endpoints.ts` (or the first non-wrapper frame, through source maps); Python callers through an HTTP-client
shim at function grain; **shell callers at file grain** (a PATH-shimmed `curl` records the parent script from
`/proc/$PPID/cmdline`, and the item is the `<script>` symbol); `falkor-chat`'s suites' `TestClient` edges are
reported as a class outside the denominator; a frame that cannot be attributed is counted `unattributed`. The
`ccg-oracle-v1` record is committed before the capture. **Fence:** every FalkorDB the run
needs is a **throwaway instance** (`infra/ccg-falkordb.sh --throwaway`) on a free port; the run never connects to
the shared `falkordb-dev` (6379) or to `ccg-falkordb` (6390). The suites and scripts under test default to port 6379
(`${FALKORDB_PORT:-6379}` in `falkor-chat/scripts/`), so the capture wrapper **exports `FALKORDB_HOST` and
`FALKORDB_PORT` explicitly** and then asserts that the *effective* port equals the throwaway's own published port and
is neither 6379 nor 6390, and that the target container carries the `ccg-falkordb=1` label. Test: with
`FALKORDB_PORT` unset the wrapper refuses (the named mutant). **Gate:** `analyst`. **Needs:** 48, 02, `-ml`.

## P1-36 · noise and runtime-oracle tooling — `data-scientist`

Per EP §11 and `-p1-ml.md` question 4. **Files:** `cpg/eval/tools/noise.py` (frame: every listed item and
expanded group member over the held-out anchors' answers; confirmed items censused; possible items stratified by
(reason, distance band 1 / 2 / ≥ 3), sample ≥ 100, `min(N_h, 5)` per stratum then largest-remainder allocation in
exact integer arithmetic capped at `N_h`, draw by hash sort with `seed = "ccg-noise-v1|" + gateId + "|" +
snapshotId`; the frame, sample and labels are written under the custodian root, §6.4), `cpg/eval/tools/oracle.py` (runtime oracle per `-p1-ml.md` §6.5, over the edges P1-47 captured: observed in-scope edges present in the gate
snapshot as a `Link` ÷ observed in-scope edges, Wilson over edges and over distinct keys, reported beside seeded recall,
never pooled, an unattributable frame counted `unattributed`; it reads the custodian root and reports **counts only**,
each oracle miss feeding the next held-out set through the custodian),
`cpg/tests/test_noise.py`. Additions to EP §11 (`-p1-ml.md` §5.1): out-of-scope items leave the frame before allocation (counted per answer and
reason, never labelled); burden is reported **twice**, in-scope and all-listed; a deterministic **naive over-lister**
baseline (every symbol within bound 2 reachable by every name-match candidate of every call hop, plus every provider
and consumer of the same contract kind) gives reduction as Σ baseline burden ÷ Σ system burden, flagged if its recall
is lower; noise is also **split by each possible hop's own edge status** (a `flow-unavailable` chain reported with
the status its edge would have in a flag-true unit). The labeller is a fresh `qa-engineer`, not a pack author, and
applies the rubric of EP §11.4; raw agreement and κ are reported, and below 0.8 raw agreement the possible-precision
figure is marked "labels unstable". The measurement
reports the ratified thresholds (4:1 cap, possible precision ≥ 0.20 pooled, p90 ≤ 100, reduction ≥ 5×; `-p1-ml.md`
§5.3), which apply from the next gate on. **Gate:** `analyst` on arithmetic.

## P1-37 · AC-5 first gate run — `qa-engineer`

Prerequisite: the stakeholder's redo of the 40-item EP §7.3 sample is recorded, with the held-out manifest re-recorded and committed if a held-out file changed (`-p1-ml.md` §9 Q-D); the `ccg-prereg-v1` record `ac5-gate-1` is committed **before** the build (EP §12.3; `teco`), P1-42 is
green, P1-34's decision is recorded, no builder has touched held-out items.
1. **Run** the harness (P1-32b) over the held-out items for the HTTP, MCP and scripts strata, the wiring stratum
   (**reported "screened", not "certified"**: held-out exactly at its floor of 8), and the `flow` stratum **only for a
   language whose flag is true after P1-34** (otherwise "flow-unavailable, not closed", held-out flow stays
   unexposed, flow diagnostics are on dev items only). Counting `H` is 159 beside `H_all` 258 (flow contributes 0
   counting items): the floor of 75 is read against 159 (P0 report, S47).
2. **Report, as run:** every miss, triaged by the gate runner and the method owner into exactly one of `key-error`,
   `extraction`, `recognizer`, `linker-engine`, `depth-bound`, `dynamic` (EP §12.1); recall as run and excluding
   confirmed key errors, side by side; confirmed precision ≥ 0.95; wrong-possible ≤ 4 per real impact, pooled with a
   bootstrap interval, by distance band and at p90 per answer; noise (P1-36), each also **split by each possible
   hop's own edge status** (spec2 §5: own status is `confirmed` iff the hop is or its reason is `flow-unavailable:*`;
   `-p1-ml.md` §6.5); **the runtime oracle beside seeded
   recall**; recall over items, distinct anchors and clusters (`-p1-ml.md` §6.1); the chain length (inter-function hop
   counts over hit items, median and p90 by stratum; the deferred measurement for ML §3.6's 0.98 revisit); the oracle's
   misses as counts by class; the Q-D redo's counts by stratum and class (`-p1-ml.md` §9); intervals with the anchor or cluster as the unit; the P0 disclosures restated (AI labels, completeness 0.845 for `mcp`
   and 0.765 for `flow`, rules revised after results).
3. **On any miss:** the run is reported as run and P1-43 starts.
4. **Possible-precision thresholds:** gate 1 gates at Q ≤ 4 only; the other three are reported with their intervals
   and gate from the next gate on (P1-43, else P2's first gate; `-p1-ml.md` §5.3).
- **Gate:** `analyst` on the report; `data-scientist` on method. **Output:** `docs/test-reports/
  connected-code-graph-p1-report.md`, the gate section.

## P1-43 · AC-5 re-gate (contingent on a miss in P1-37) — `qa-engineer`

Its own brief, written from the first run's triage.
Root-cause and fix on dev; every item of a missed item's cluster moves to dev; re-gate on a **new**
   list — kept clusters plus ≥ 35 counting fresh-cluster items (E12) — written by two isolated key-writing agents
   (`cpg/eval/headless/`, S52 launcher, accepted-plugins and accepted-agents lists, the breach2 guard fix
   036e2cb9). Kept-cluster candidates are read from the custodian root (`-p1-ml.md` §6.4). Before any fresh key-writing the
   `-p1-ml.md` §7 checks are in the briefs: the scope-overlap procedure (§7.1; the two writers get identical
   per-stratum briefs) and EP Appendix B's catch-all patterns re-run at the new `EVAL_COMMIT`, their count recorded in the
   brief and the `{**}` clause added if it is above 0 (§7.2). "All hits, floor not met" is **not** a pass.
- A new `ccg-prereg-v1` record precedes the new build. **Gate:** `analyst`; `data-scientist`. **Needs:** 37.

## P1-38 · phase exit — `teco`

- **AC-closure table** (index §8): each AC of the proposal's P1 row — AC-1, AC-2, AC-4 closed; AC-5 per stratum;
  AC-6 for the P1 kinds; AC-8 part 1; AC-9 (code-snapshot stamp); AC-9b closed; AC-10 (impact stages, via P1-27);
  AC-11 (impact); AC-16 closed (impact, four clients); AC-17 (code location, contract point); AC-18 (impact
  answers); AC-21 closed; AC-24 smoke; AC-25 closed — with the closing step, evidence file, and "reported not
  closed, because" for any that did not come out (the `flow` stratum, AC-25's confirmed form, the vacuous AC-8 rows).
- **`analyst` implementation review** of the whole phase diff: `docs/reviews/connected-code-graph-p1-impl.md`,
  including the growth-stops check; no close on "needs changes".
- **Spike dispositions (index D7):** `cpg/spikes/vertical-slice/` is not deleted. `git grep -n vertical-slice` (tracked, excluding the directory and `docs/`) is run; each hit is re-pointed (`s28_run.py`, in P1-34), recorded as historical (`contact.json`, `store/results/*`, `.sc` header comments) or listed in the BACKLOG item with its disposition (`spikes/store/{bfs,synth,lock_replay}.py`, `store/tests/test_store_spikes.py`). `cpg/README.md` marks the directory frozen.
- **Inputs to the close docs** (index §11): the `HISTORY.md` entries, the `BACKLOG.md` rewrite, the AGENTS.md and
  README edits, the archive flips, the closeout list of what to delete.
- **Gate:** `analyst`.
