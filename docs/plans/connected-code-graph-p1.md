# Connected code graph — P1 plan (impact on this repo, every client)

> **Status:** active · **Owner:** `architect` · **Tracks:** C-1001 (P1)

This is the **index** of the P1 plan. Step detail lives in four ordered sub-plans (§3.1):
`docs/plans/connected-code-graph-p1-extraction.md`, `docs/plans/connected-code-graph-p1-linking.md`,
`docs/plans/connected-code-graph-p1-answers.md`, `docs/plans/connected-code-graph-p1-gates.md`.
Each cites this document for decisions and never restates them.

## 1. Goal & scope

**Goal.** Deliver the proposal's P1 row (`docs/plans/connected-code-graph.md` §4, "P1: Impact on this
repo, every client"): a built, linked, stamped snapshot of this monorepo and an answer engine that
answers an impact question and its reverse (depends-on) through all four clients (agent MCP, CLI/CI,
plain language, raw query), with the egress broker, offline builds and the `teco` v1 seam. Python,
TypeScript, shell, structured config, Compose, `.mcp.json`, Kiro agents and prompt Markdown only.

**Out of scope** (later phases, proposal §4): Java/Kotlin/C#/Go/C++ extractors, gRPC/broker/library
packs and groupings (P2); data, test, change, overlay and review (P3); verdicts, access (P4); RCA,
security (P5); the Arduino slot. The `verdict` MCP tool is not built. Refresh triggers other than
on-demand are P3 (AC-9a). In-system git fetch (broker rule `R2`) and TLS interception are out: the stakeholder took option C for P1 (§6).

## 2. Context & findings

CPG: considered, not relevant — no graph is loaded for `cpg/` (the loaded graphs are `cpg_falkorchat` and unrelated ones); the plan rests on the P0 specs and on the spike code read directly.

- **State at P0 close (2026-11-06; `docs/HISTORY.md`).** `cpg/ccg/` holds contract code only: `ids.py`,
  `bundle.py` (validator and the BS2 gates), `keynorm.py` (`mask_http_segments`), `rulepack.py`,
  `answer.py` (canonical-answer validator), `schemas/`, `registry/reasons.yaml`, `store/ddl_v1.cypher`.
  **Nothing builds, links, loads or answers yet.** The proven mechanics are in `cpg/spikes/vertical-slice/`
  (`extract.sc` 403 lines, `normalize.py` 1,244, `assemble.py` 657, `load.py` 675, `impact.py` 422), to be ported
  under production tests, not edited in place. `cpg/spikes/store/` and `cpg/spikes/frontend-fitness/` are
  measurement tooling that P1's re-measurements reuse.
- **Contracts P1 implements against** (frozen v1): bundle `docs/plans/connected-code-graph-bundle-spec3.md` +
  `-bundle-spec4.md`; graph `-graph.md` + `-p0-graph.md`; packs `-rule-pack-spec.md`; answers `-answer-spec.md` + `-answer-spec2.md`; bundle also `-bundle-spec5.md`;
  egress/sandbox `-egress-spec3.md` (supersedes `-egress-spec.md` + `-egress-spec2.md`); evaluation `-eval-protocol-ml.md` + `-eval-protocol2-ml.md`
  (v1.8 / v1.5 per the ledger).
- **Builder isolation is binding.** Held-out ground truth is outside the repo (`~/.local/share/ccg-eval/`);
  `cpg/eval/heldout-manifest.json` is the only committed trace. Builders work from `cpg/eval/dev/` and the
  fixtures only. A held-out item that any builder debugs against moves to dev (`-eval-protocol-ml.md` §8.4, §12.1).
  Every brief in §7 that touches the proving repo says so.
- **Fixture builds are "non-proving-repo" builds.** The proving repo is this monorepo. `cpg/fixtures/system1`
  (`fx-svc`, `fx-web`, `fx-job`) is not, so F-1 (bundle-at-rest) and F-9 (`Cond.code` masking) must land before
  the first fixture build (`-bundle-spec3.md` §13), and the extractor is developed on `falkor-chat/server` and
  `salesperson/` until then (steps P1-02, P1-06; sequencing in the sub-plans).
- **The compute host is small** (6 GB; `-p0-ml.md` §7). Python extract + normalize cost 4.1–4.2× the Joern
  run in the slice (`docs/test-reports/connected-code-graph-p0-slice-report.md` §7 Q2). P1-20 measures a full refresh.
- **Fixture rows gated at P1:** 91 of 132 (`docs/test-plans/connected-code-graph-fixtures.md` §6, "By earliest
  gate"), read from that document; P1-41 asserts them.
- **System scope equals census scope.** The proving system's manifest (P1-03) excludes `cpg/`, `deprecated/`,
  `docs/**/connected-code-graph*`, `docs/BACKLOG.md` and the content-check files that
  `cpg/eval/eval-manifest.yaml` lists, because the held-out keys were written under that scope
  (`exclusions.paths`; "census scope = evaluated scope"). The fixture system's units are built as separate systems.
  Consequence: impact questions about `cpg/` code are not answerable in P1; the noise and recall frames of P1-36 and
  P1-37 are over the same scope.
- **Step ids.** `P1-NN` (and `P1-48b`) collides with no plan step: the ledger uses `S01–S59`, `P0-NN` and `U33`, and `git log --all`
  carries no `P1-NN` plan step (one-digit review-finding ids `P1-1..4` in `falkor-chat/docs/reviews/` do not match
  the two-digit form). Ids `P1-39` onward are the splits and additions; `P1-30` is unused.

## 3. Design & rationale

### 3.1 Shape of the plan and the dependency waves

P1 is 49 steps (ids `P1-00`..`P1-48` and `P1-48b`; `P1-30` is unused) across extraction, linking, answers and egress/gates. One document would not stay
reviewable, so this index carries decisions, the obligation walk, the step index, the notes P1 needs and the
risks; each sub-plan carries step detail (files, signatures, done-conditions, tests, gates).

| Wave | Content | Sub-plan | Can start |
|---|---|---|---|
| W0 | Specs and intake that unblock builders: answer-spec successor, `bindParts` spec, egress spec successor, dependency intake, bundle-at-rest, the `-ml` and `-graph` notes (§9) | all | now |
| W1 | Extraction: manifest and units, Joern port, normalizers, writer obligations, shell/config/prompt extractors, indexers, projection fix | extraction | after P1-01 |
| W2 | Linking: pack runtime, linker, assembler, store loader and ACL, refresh | linking | after W1's bundle-writing steps |
| W3 | Answers: reader/anchors, BFS rows, flow fallback, renderers, clients, plain-language front, `teco` v1 | answers | engine after the loader; front after the broker core |
| W4 | Egress and gates: broker, AC-8 harness, gate harness, fixture gate, flow re-measure, smoke, noise, AC-5 runs, exit | gates | broker core runs beside W1 |

The egress track (P1-28, P1-29) and the specs of W0 are independent of W1–W3 and run in parallel. Nothing waits
on the stakeholder: Q1 is decided (§6).

### 3.2 Decisions this plan takes (one home; sub-plans cite them as "index D<n>")

| # | Decision | Why |
|---|---|---|
| D1 | **`header.history` `unavailable` stays illegal in P1** (a shallow clone or missing history is a build error), as in `-bundle-spec3.md` §2.2. | No P1 consumer needs it (RCA window and `lastChanged*` for FR-12/FR-14 are P3/P5); widening later is additive. |
| D2 | **`flowAvailable` is data, not code:** `cpg/ccg/registry/flow-availability.yaml`, one row per language (`python`, `javascript`, …) with the value and the measurement it comes from. The assembler writes `Unit.flowAvailable` from it. A row flips only by an accepted re-measurement record (§5). | The flag is "decided per language from P0's measurements" (`-bundle-spec3.md` §2.2); a registry file keeps the decision reviewable and diffable, and gives `coreTreeHash` something to cover. |
| D3 | **The shipped default depth bound is 2** (`DEFAULT_DEPTH_BOUND = 2`, one constant in `cpg/ccg/engine/config.py`), recorded in every pre-registration (`defaultDepthBound`) and in `depthBound.unexplored` counting. | EP Q1: a smaller bound fails `flow`'s distance-2 items by construction (`-eval-protocol-ml.md` §3.6). |
| D4 | **`bindParts` (F-7) is built in P1** as an additive optional field via `-bundle-spec5.md` (P1-13, P1-14), not deferred. | It removes the capability-URL residual the plan accepted "for P1" (`-bundle-spec3.md` §1.7.6) and lifts multi-segment base URLs (`/api/v1`) that cannot bind with clear parts. The cost is one schema delta; the HMAC machinery already lands in P1-07. |
| D5 | **Interpreter prompts and grammars are never `.md` under `cpg/`.** They live in `cpg/ccg/front/prompts/*.txt` and `grammar/*.json`; a test asserts no `.md` file there. | `coreTreeHash` excludes `**/*.md`; a prompt change must change it (`-eval-protocol-ml.md` §12.3, the M5 condition). Satisfied by construction. |
| D6 | **MCP server = the `mcp` Python SDK, stdio, FastMCP style**, as `cypher-mcp/` and `mcp-monitor/` already do; tools `ask`, `resolve`, `refresh`, `raw_query`. | Repo convention; the launch surface stays one command. |
| D7 | **`cpg/spikes/vertical-slice/` is ported, then left frozen (not deleted) through P1.** Each ported module's equivalence test is green **at the step that ports it**, with the stored-bundle sha256 and `0 skipped` recorded (D9's rule); later steps that change output pin their own deltas. Deletion is a backlog item with a per-hit disposition (§11). | Tracked code depends on the directory: `cpg/spikes/store/{bfs,synth,lock_replay}.py` import it, `store/tests/test_store_spikes.py` asserts `cpg/spikes/vertical-slice/impact.py`, and `cpg/spikes/frontend-fitness/s28_run.py` runs its `extract.sc` (P1-34 re-points it). Not deleting costs nothing. |
| D8 | **`pyproject.toml` is edited by P1-01, by P1-29 for the `broker` extra only, and once by P1-03 for the `package-data` entry `ccg/system.yaml`** (the owner is `devops`, except P1-03's one entry; the `dev` pins `jsonschema==4.26.0` and `pyyaml==6.0.3`, no new pin, `-egress-spec3.md` §10.2). P1-01 adds extras `extract`, `engine` and an empty `broker`, registering the `slice` marker, creating `cpg/extractors/__init__.py` and `cpg/clients/__init__.py`, and widening `packages.find` to `["ccg*", "extractors*", "clients*"]`. Any other step that misses a dependency stops and asks `teco`, as in P0 (`cpg/README.md`, Setup). | One owner for pins; the layout of proposal §3.13 puts extractors and clients outside `ccg*`. |
| D9 | **A test that skips because its oracle artifact is absent (the gitignored `cpg/.ccg-work/`) is reported unrun, not green.** Every equivalence gate states the sha256 of the artifact it used and `0 skipped`; the reviewing `analyst` executes it. | A skipped-green equivalence test proves nothing on a fresh checkout. |
| D10 | **Compose mappings and the `.mcp.json` launch chain are linker code, not pack entries.** Packs carry only what their `launch` block holds (framework default ports, port flags, target argument, env ports, mount paths; rule-pack spec §1.6). The **shell `mcp-call` consumer is a new pack** (P1-46). | Compose is core behaviour (`-p0-compose-binding.md`; proposal §3.4 bindings), and a pack entry would re-open the pack format for no gain. |
| D11 | **The build path dispatches by unit kind in one place** (P1-44); no extractor is reachable only from its own tests. | Otherwise shell, config and Markdown extractors never run in a refresh. |

Rejected, and why: **(a)** building the engine as `algo.SSpaths` (P0: item sets differ on 15 of 50 anchors,
no chain; `-p0-graph.md` §6 row 1); **(b)** a third-party proxy (mitmproxy) for the broker — P1 has no proxy listener (option C, §6);
P1-28 decides the process split, and the proxy technology is an input of the deferred path (`-egress-spec3.md` §6); **(c)** keeping the slice code and
patching it — the slice is a spike with no production tests of its own for the new rows.

## 4. Obligation walk (P0 §15 handoffs and P0 disclosures)

Every item has an owning step or "reported not closed, because". Sources: `-p0-2.md` §15 ("Named handoffs to
P1", "Owed to P1 by the P0 results", "Measurements deferred"), `-bundle-spec3.md` §13 and `-bundle-spec4.md` §4,
`-answer-spec.md` carry-over, `-p0-graph.md` §1.1, §6, `-rule-pack-spec.md` §12, `-egress-spec.md`/`-egress-spec2.md`
§3, and `docs/test-reports/connected-code-graph-p0-ground-truth-report.md` `## P0 disclosures`.

### 4.1 Named handoffs (`-p0-2.md` §15)

| Obligation | Owner |
|---|---|
| Production extractors, ported from the slice under P1's own tests, with D15's content check | P1-04, P1-05 (Python/TS/Joern), P1-10 (shell), P1-11 (config), P1-12 (Markdown); P1-44 runs them in a build |
| Pack runtime | P1-16 |
| Linker | P1-17 |
| Store loader: S22's ACL users, `ddl_v1.cypher`, S23's batch ceiling, load-under-`TIMEOUT_MAX` | P1-19 |
| Id-hash fuzz test (throwaway instance only) | P1-19 (instance script from P1-48) |
| Broker process and AC-8's logging proxy, after Q1 | P1-28, P1-29, P1-31, P1-40; Q1 is decided (option C), so the proxy is a detector and the git-fetch path is deferred with a named trigger (§6) |
| Answer engine per S20/S33 with default depth bound ≥ 2 | P1-21–P1-23; bound is index D3 |
| Prompt extractor classifying by `eval-manifest.yaml` globs | P1-12 |
| AC-5 gate harness (held-out manifest check, build at `EVAL_COMMIT`, §12.1 fail procedure with E12's floor, routing by S(u) and exposure, §12.3's four-field comparison incl. `coreTreeHash`, imports H1 span test and §5.4 from `match_keys.py`, `renormalize-check` before scoring and stop on a difference, `anchor-not-in-graph` miss, imports rather than reimplements) | P1-32 (second dispatch), run in P1-37 |
| `regate-list` and `exposure` verbs, if P0 did not build them | P1-32 (first dispatch); P0 did not (ledger 2026-11-06, S49 "Owed") |
| Possible-precision thresholds | P1-36 builds the measurement, P1-37 reports it; the thresholds are ratified (§5.3 of `-p1-ml.md`): the 4:1 cap, possible precision ≥ 0.20, p90 ≤ 100 and ≥ 5×, applied from the next gate on while gate 1 stays at Q ≤ 4 |
| Runtime oracle reported beside seeded recall, never pooled; each oracle miss feeds the next held-out set | P1-47 captures the runtime edges on an isolated instance (never `falkordb-dev`), P1-36 tooling, P1-37 report |
| P1 assembler imports `mask_http_segments`, never reimplements step 8a | P1-18 (a test asserts the import and forbids a second definition) |
| `coreTreeHash` `*.md` condition | index D5, satisfied by construction; asserted in P1-26 |
| AC-24 smoke harness (EP §10.5) | P1-35 |
| `teco` v1 integration (a `cobb` unit) | P1-27 |
| Full root `AGENTS.md` `cpg/` rewrite | §11 (phase-close item, not a build unit) |
| Port or delete `cpg/spikes/` | index D7: ported, kept frozen; deletion is a backlog item with per-hit dispositions (§11) |
| Re-read `flowAvailable` if call-level reach is not a superset of flow reach | P1-39 measures, `-p1-ml.md` question 2 rules (§5) |
| `-spec2` F-10 as one writer done-condition (W2 with `default`, W3, W4, SA-1 writer side, §1.6.1 argv rule, per-entry HMAC) | P1-07; G-4 (CDG self-edge dropped, `DATA` self-edge kept) is applied in the normalizer, P1-05 |
| F-6 (SA-3 role allow-list, per-format `keyPath`) before P1 relies on `bindValue` | P1-11, which precedes P1-17 |
| F-7 (`bindParts`) | index D4; P1-13, P1-14 |
| F-9 (`Cond.code` masking) before the first non-proving build | P1-06, a hard prerequisite of P1-41 |
| F-12 cycle-safe traversal in the BFS | P1-22 |
| F-13 (`http` `keyNorm` step order; no consumer may encode the old order) | P1-17 (`matchers.py`), P1-21 (anchors), P1-18 and P1-32 use `ccg.keynorm` and the RFC-order normalizer only; a test asserts `GET /a/%2E%2E/b` normalizes to `GET /b` through every P1 consumer |
| F-1 bundle-at-rest, before the first non-proving build | P1-02, a hard prerequisite of P1-41 and P1-20's fixture run |

### 4.2 Answer-spec carry-over C1–C9, rule-pack, graph and egress items

| Obligation | Owner or disposition |
|---|---|
| C1 `CONTROLLED_BY` cycle test and mutant | P1-22 |
| C2 renderer tests of answer spec §7.1–§7.2 | P1-24 |
| C3 the three reads for the raw leg's item-set derivation | P1-21 (with `graph-dba` review) |
| C4 count `depthBound.unexplored`, record bound ≥ 2 | P1-22; bound is D3 |
| C5 gate-harness assertions (pinned snapshot, replayed chains, masked H1 comparison, cache empty) | P1-32 |
| C6 `algorithm.version` and bump rule; `function-level-continuation` retirement | P1-00 (rule), P1-22 (bump, retirement) |
| C7 pre-register the did-you-mean cap | P1-35 (method owner; EP §10.4) |
| C8 verdict-log properties on `:Snapshot` | Reported not closed: P4 (no verdict log exists in P1); P1-19 leaves `verdictLogOffset` 0 and the empty-log hash |
| C9 `rcaMarks`, integer ranker | Reported not closed: P5 |
| `cacheBypass` input and EP §10.4 citation; four class-X edge types not carried by any P1 hop; hop kinds for `DEFINED_BY`/`LAUNCHES`/`DEPENDS_ON`/`FROM_DEP`; answer-spec stale text | P1-00 |
| O-3 router in a module variable (`possible` in v1) | Reported not closed: needs a `-spec5`-class record or a P1 mechanism; P1-16's conformance records the `possible` outcome; revisit at P3 |
| O-5 FastMCP inner path override: MCP path matching is prefix-only | Reported not closed: no record carries the value (`mcp.py:64`); behaviour pinned in P1-17 |
| O-9 reason on an `mcp_tool` Link when a binding selects the unit; framework-default port evidence | `-graph` note question 2; implemented in P1-17, P1-18 |
| O-10 where the pack set's identity is recorded in a snapshot | `-graph` note question 3; P1-18 |
| O-12 whether consumers are anchors in contact set C | P1-32 checks the contact set against the ruling the ground-truth custodian made; if none, escalate in P1-37's brief |
| O-13 conformance differences from the slice recognizers; the 24 leaves exercised only at P1 | P1-16, P1-17, P1-46 (done when the leaf-harness set in `cpg/tests/test_rule_pack_leaf_mutation.py` is empty and `test_each_leaf_behaves_as_classified` passes) |
| O-15 pattern keys and keyless `CONSUMES` for topics | Reported not closed: topic packs are P2/P3; P1 builds no topic pack |
| `MOUNTED_BY` (assembler emits mount ids), `SkippedFile`, `PROVIDES` props, multi-candidate links, multi-batch at P1 scale, `ownedRegions` fixture, `arg(i)`/`Ret` rows | P1-18, P1-19 (graph side), P1-22 (rows), P1-41 (`ownedRegions` and multi-candidate fixtures) |
| Contract walk from a provider-less CP lists its consumers (`CONSUMES <-`/`ON`) | P1-22, asserted in P1-41 |
| `TIMEOUT_MAX` 10,000 ms and lock ttl 2,000 ms: P1 re-derives | P1-19 |
| `ConfigEntry.entryId` index before the load; the four per-system ACL users `ccg_<role>_<s>` | P1-19; their lines are `-p1-graph.md` §7 (the graph note `-graph.md` is `Extended by:` `-p0-graph.md` and executed against, so it is not edited); the name module `ccg.sysname` is P1-03's |
| ACL test on a throwaway instance planting `shop`/`shop2`, `shop`/`shop__x` and `shop`/`shop_`, pinning the lock selector and the system-name whitelist | P1-19 |
| Egress: loader check that every `sandboxAuditIds` id has a `sandbox` record | P1-18 (bundle read) and P1-19 (load) |
| Egress: `gitFetchMaxBytesOut` calibration; `remote` as a list; proxy technology | Reported not closed: they belong to the deferred git-fetch path (§6); `egress-v1.schema.json` is created by P1-03, its content specified by P1-28 (`-egress-spec3.md` §4) |
| Linker: cross-repo Compose build-context to unit resolution (`-p0-compose-binding.md`) | P1-17; fixture assertion in P1-41; fallback is a bundle-spec image-name rule, escalated, not patched |
| `jssrc2cpg` ignore override ("P1-verified requirement"), `skippedFiles`, coverage probe | P1-08; if the frontend allows no override, the files are listed `frontend-ignore` and the AC-1 `salesperson/` i18n sites stay `possible` (slice fallback) |
| Slice S12 finding 3 (`isTest` flag on provider and mount records) | Reported not closed: the slice measured 0 providers mounted only from test files (slice report §6.1); a field is a schema change; revisit at P3 with the test packs |
| Deferred measurements: ~18 GB peak re-estimated at P1 | P1-19 reports it (measured at P2) |
| Deferred measurements: typical evidence-chain length (for ML §3.6's 0.98 revisit) | P1-37, from the first gate's answers |
| Deferred measurements: `Link` counts under runtime-built URLs at scale | P1-19 reports from the full-repo load |
| The `security-expert` HMAC and canary review | Reported not closed: scheduled before P5 (`-p0-2.md` §15), after P1 |
| Deferred measurements: `LogTemplate` volume → P5; `RAN` volume → P3 | Reported not closed: those packs and importers are not in P1 |
| Independent provoked-log set per runtime for the 34 infra signatures | Reported not closed: the infrastructure-signature pack is P5 (proposal §4); no P1 step consumes it |
| `tico`: record the four stakeholder decisions (Q5, E11, E12, FR-15 scope and internal hostnames) in the requirements document | Reported not closed here: `tico`'s edit, no P1 step depends on it; it is listed so `teco` can dispatch it (§9) |
| The human/`cobb` question whether root `AGENTS.md` rule 5 gains an "`Extends:` but replaces named clauses" form | Reported not closed: a repository-rule question; P1's successors use `Extends:` and `Supersedes:` as the rule stands |

### 4.3 P0 disclosures that bind P1 (`## P0 disclosures`)

| Disclosure | Disposition |
|---|---|
| Labels are AI labels; only the 8 PT gold items had a human check; the 40 union items were AI-verified (deviation from decision S36) | The stakeholder redoes the 40-item EP §7.3 sample before the `ac5-gate-1` record (`-p1-ml.md` §9 Q-D); a held-out file that changes re-records the held-out manifest first. Every P1 gate report restates the disclosure (P1-37) |
| Rules revised after results (EP v1.5–v1.8, ML v1.11–v1.17) | History, not P1 work. P1's new rules (the fresh flow draw, §5) are recorded before their results (P1-34) |
| KA-flow 15 invalid lines; flow completeness 0.765 is a lower bound (~0.82 if credited) | If the flow stratum is gated, recall is reported with and without the 15 (§5; P1-37) |
| KA2 passes: `mcp` 0.845 and `flow` 0.765 stay below 0.95 | Gate reports carry the completeness; a re-gate's fresh items follow EP §6.5 (P1-37) |
| Scope-overlap open (stratum-tag mismatches) | `-ml` note question 6, before any fresh key-writing |
| Lumpy and thin held-out; wiring at its floor of 8 ("screened", not "certified") | P1-37 reports intervals by anchor or cluster and calls the wiring result "screened" |
| S54 guard gap: O1, O2 (plan wording in the archived `-p0-2.md`) | Reported not closed: an archived plan is not edited; a P1 headless writer run uses `cpg/eval/headless/` as fixed in commit 036e2cb9 |
| S54 O3 (`audit.py` treats an absent-tool call as unrefused) | P1-01 (`devops`), before any new headless writer run |
| Firewall slips (S57, two custodians) | Reported not closed: waived and recorded in P0; no P1 work depends on it |
| EP Appendix B: a catch-all clause is needed before key-writing on a codebase with catch-all routes; the ledger carries no route count | `-ml` note question 6; checked before P1-37 writes any fresh key |
| Part F mid-audit rulings | Reported not closed: P0 history |
| Kotlin unmeasured | Reported not closed: P2 |
| Scripted S47 steps (§7.4 correction, §10.3 join, §10.2 AC-24 split) are not tool verbs | P1-32 (first dispatch); `cpg/eval/tools/s47_scripted_steps.py` is the replay record the verbs must reproduce byte for byte |
| `capture`'s `trigger` flag stays true after a KA2 run (nit) | P1-32 fixes it with the verbs |
| S50 live write to the shared `falkordb-dev` (6379) | P1-19's ACL test runs on a throwaway instance (`infra/ccg-falkordb.sh --throwaway`). Process rule for every P1 brief: "no writes to any shared database; use `ccg-falkordb` or a throwaway instance" (§10) |
| Frontend fitness: `flowAvailable` false for six languages | §5 |

## 5. Flow gate

**Premise.** `flowAvailable` is false for python, javascript, go, java, csharp and cpp in both
configurations; Kotlin is unmeasured (`docs/test-reports/connected-code-graph-p0-frontend-fitness-report.md`
§a2.2.6; rule `-p0-ml.md` §8.4: pooled pair recall ≥ 0.95 observed over ≥ 120 pairs in both configurations,
projection success ≥ 0.95 as precondition). Adjudicated pooled recall: python 308/481 = 0.640, javascript
61/152 = 0.401. A fixed projection must be measured on a **fresh flow draw with a new seed prefix**: the 30
methods per language are development contact for any `extract.sc` fix (§a2.2.1; `-p0-2.md` §15, "Owed to P1").

**What P1 can reach (derived here from the report's §a2.2.2 counts; not a measurement).**

| | connected now | projection-side misses | join-side | not fixable in the projection | ceiling |
|---|---|---|---|---|---|
| python | 308 of 481 | 111 = 67 (R1) + 17 (R3 only) + 27 (comprehension scopes 12, call as CDG source 10, dict/array/spread 4, alias 1) | 40 = 36 (`for` header) + 4 (the `not (...)` contains-join) | frontend 9, definition 9, undetermined 4 | (308 + 151) / 481 = 459 / 481 = **0.954**; with synthetic-cond transparency (option 5) 467 / 481 = **0.971** |
| javascript | 61 of 152 | 18 | 4 | frontend 62 (52 `switch` case labels), definition 4, undetermined 3 | (61 + 22) / 152 = 83 / 152 = **0.546** |

R3 (frontend-synthetic calls such as iterator `next` are transparent intermediates) is a scoring-time relaxation in
the report (§a2.2.1: "R3 is not a projection"), so adopting it is a projection design line of its own. Passing
needs 457 / 481 = 0.950 for python: **149 fixable misses fixed (of 151 without synthetic-cond transparency, of 159 with it), on the very 30 methods that are
development contact.** A faithful P1-09 that fixes R1, the 27 unverified causes and the `for` header reaches
308 + 67 + 27 + 36 = 438 / 481 = 0.911; adding R3 gives 455 / 481 = 0.946, and the contains-join fix 459 / 481 =
0.954 (`-p1-ml.md` §1 F1). The decision at 0.95 is close to a coin flip even for a true-0.95 projection (python's
clustered draw of 30 methods, `-p1-ml.md` §1 F2: P(pass) 0.54–0.56 at 0.95, 0.17–0.25 at 0.93, 0.01–0.04 at 0.90). **Plan expectation: python's flag stays false; javascript's cannot turn true by any projection fix.**

**Options.**
1. **Fix the projection; make the fresh draw conditional and off the critical path (chosen).** P1-09 ports R1, R2,
   R3 (as a projection rule, with a golden), synthetic-cond transparency (option 5), the unverified projection-side
   causes and both join fixes (the `for` header and the contains-join), each fix or a recorded "not fixed, because",
   and reports the dev recall it reaches. P1-34 spends a fresh draw **only if** that dev figure reaches **465/481**
   (`-p1-ml.md` §3.1, from the §8.4 operating curve; dev recall is optimistic, since the fix is fitted to those 30
   methods, so "not run" is the expected outcome). JavaScript/TS
   is not re-measured: reported not closed (ceiling 0.546); the remedy is a frontend-side synthesis of `switch`-case
   conds from the AST, logged as a backlog item at close.
2. Re-measure both languages regardless. Rejected: spends a blind draw on a language whose derived ceiling is 0.546.
3. Do not touch the projection in P1. Rejected: R1 is verified at scale to connect 287 of the 1,000 misses and to lose
   no connected pair, and the proposal's P1 row promises the fix.
4. Lower the 0.95 floor, or decide availability per method. Rejected: the floor is not lowered for any language
   (`-p0-ml.md` §8.4), and a per-method flag would make an answer depend on which method a chain crosses.
5. Project the **transitive** CDG so python's 9 `definition`-side misses connect. Decided in the narrower form
   `-p1-ml.md` §3.4 states: a frontend-synthetic cond is transparent to the projection (`point → C` in place of
   `point → S`), a P1-09 rule that raises the dev ceiling to 467/481. A general transitive CDG is not projected:
   §7.3 requires a direct `CONTROLLED_BY` edge.

**What the engine does.** The assembler writes `Unit.flowAvailable` from index D2's registry. Through a unit
whose flag is false the engine follows call-level reach only, and only the flow-dependent call-continuation hops
are `possible`, `reason: flow-unavailable:<lang>` (`-bundle-spec3.md` §7.2); items reached by resolved edges with
no such hop stay `confirmed` (stakeholder decision 2026-11-09; the closed row list is `-answer-spec2.md` §5).
Items inside a flag-true unit use `DATA` and `CONTROLLED_BY` as `confirmed`. `function-level-continuation` is
retired by the `arg(i)`/`Ret` rows (P1-22; carry-over C6).

**The "safe default" premise is checked, not assumed.** `-p0-ml.md` §8.4 says `false` is safe only if call-level
reach is a recall-superset of flow reach within a unit, and that the rule is revisited if P1's design makes it
not one. P1-39 measures, on the **development** flow items (`cpg/eval/dev/ac5-flow.jsonl`, never the held-out),
the share of flow items that call-level reach alone lists; `-p1-ml.md` §3.5 rules on it. `-p1-ml.md` §3.5 fixes the rule: P1-39 measures four walks (as-built, V-flow, V-both, V-union), and the walk
with the lowest dev burden among those with the maximum share ships; "safe" is claimed only at n/n. That is a design
decision (FR-8 requires an impact that cannot be proven to be listed `possible`), not a stakeholder question. In
every walk only the flow-dependent hops are `possible` (`-answer-spec2.md` §5).

**AC consequences.**
- **AC-5 `flow` stratum.** Gated at P1 only for a language whose flag is true after P1-34 (expected: none).
  Otherwise **reported not closed, flow-unavailable**; the held-out flow items stay unexposed and P1-37 runs flow
  diagnostics on dev items only. If gated, flow recall is reported with and without the 15 invalid KA-flow lines.
- **AC-25 (closes at P1, proposal §4).** The planted fixtures assert what a symbol-level, call-level walk can list.
  The **data-flow effect** (a changed return value used two calls away) is the callers' chain: listed `possible`,
  `flow-unavailable`. The **control-dependency effect** (a changed condition guarding other code) is a
  `CONTROLLED_BY` reach from the condition to the guarded points inside the same function; a symbol-level,
  call-level walk lists no such item (a callee is a depends-on item, not an impact item, in the caller direction).
  With a false flag the fixture therefore asserts only that the callers of the function holding the changed
  condition are listed `possible`, and the **control half of AC-25 is reported not closed until a flag is true**
  (the guarded-code item is asserted in the true form by P1-34's re-run, if ever entered). AC-25 with FR-8 decides
  the reading (`-p1-ml.md` §3.5): if the walk P1-39 ships lists the guarded point `possible` (V-flow or V-union), the
  control half closes on P1-42's fixture; otherwise it is reported not closed.
- **Data and control as separate flags** are not introduced: java's control pairs read 164/169 = 0.970 against
  0.507 for data, but the flag is per language by rule and python's control reads 0.661.

## 6. Egress decision (Q1): decided, option C

**Decision (stakeholder).** For P1 there is **no in-system git fetch**. Rule `R2` (a repository's own git remote)
is not implemented; the broker has no proxy listener, no `git-fetch` binding and no `gitFetchMaxBytesOut`. The
staleness check reads each repository's local head; the operator runs `git fetch` outside the system. This closes
the host-wide tunnel property (`-egress-spec.md` §6.1) by removing the path. (This repository's `origin` is an
SSH-form URL, which a `CONNECT https://host:443` broker could not carry in any case.)

**Deferred, with a trigger.** The git-fetch path (proxy listener, `bind_proxy`, the fixed-argv
`git-fetch` job, read-only credential file, byte-bound calibration, and TLS interception if then wanted) is built in
the first phase in which a manifest names an **https** `remote` for a repository. It is a
backlog item at close (§11), and the manifest loader enforces the deferral: a `remote` key is rejected in P1
(P1-03).

**What still needs building, re-checked.**
- **P1-28** (`-egress-spec3.md`): consolidates the sandbox, allow-list schema, rule table, audit and `sandbox`
  records; states `R2`, `R8` and `R9` as **not implemented in P1** (a deny with a reason, each); decides the
  process split **for the structured-call listener only** (one host process owning policy, tokens and the audit
  writer). The proxy-container split, `bind_proxy`, the reaper and `remote` as a list are out and recorded as the
  deferred item's design inputs. There is no `R2` text to amend later.
- **P1-29** (broker core): rules `R0`, `R1`, `R3`–`R7`, `R99` first-match, with `R2`/`R8`/`R9` as deny rows; modes
  P1–P3; run tokens; structured calls; audit.
- **P1-31/P1-40** (AC-8 part 1): the scenario's network has **no route out**; the logging proxy is a **detector**
  (every `conn` it sees fails the run, since no allow can produce one), not a forwarder. Which rows run and which are
  vacuous (under option C: 8.2, 8.4, 8.12–8.13 and the proxied halves of 8.3 and 8.5) is `-egress-spec3.md` §8.2's table.
- Everything else (model-call path, sandboxed offline builds, bundle-at-rest) is unaffected.

## 7. Step index

Detail (files, signatures, done-conditions, tests) is in the named sub-plan, under the step id. "Needs" is a
hard prerequisite, recomputed from each step's behaviours; a step with no entry is schedulable now. Gate = the
review that must pass before the step counts as done (`analyst` unless stated; every code step also carries the
mutation check of §8, and an equivalence gate follows D9). `ext` = `-p1-extraction.md`, `lnk` = `-p1-linking.md`,
`ans` = `-p1-answers.md`, `gat` = `-p1-gates.md`. Each step is one dispatch of at most about five files and three
sequential sub-steps; a step the sub-plan splits into two dispatches says so. `-ml`/`-graph` mean the notes of §9.

| Step | What | Owner | Part | Needs | Gate |
|---|---|---|---|---|---|
| **W0: specs and intake** | | | | | |
| P1-00 | Answer-spec successor `-answer-spec2.md` (`Extends:`): stale text, `cacheBypass`, hop kinds for `DEFINED_BY`/`LAUNCHES`/`DEPENDS_ON`/`FROM_DEP`, version-bump rule (C6), depth bound D3, §5's fallback wording | `architect` | ans | — | `analyst` |
| P1-01 | Layout and Python dependency intake (D8): extras, package stubs, `slice` marker, `audit.py` O3 | `devops` | ext | — | `analyst` |
| P1-48 | Instance and image: `indexer` image target, `ccg-falkordb` flags and `status` hygiene, `--throwaway` mode | `devops` | ext | — | `analyst` |
| P1-48b | The runner's indexer entries: `offline-run.sh` frontends, argv grammar, image-label check, `PYTHONSAFEPATH`, sandbox tests (`-egress-spec3.md` §7.4, §10.1 (21)–(26)) | `devops` | ext | 28, 48 | `analyst`; `security-expert` advisory |
| P1-02 | Bundle-at-rest, HMAC-key and ACL-credential custody (F-1) | `devops` | ext | — | `analyst`; `security-expert` advisory |
| P1-13 | Bundle spec successor `-bundle-spec5.md` (`bindParts`, F-7) | `architect` | ext | — | `analyst`; `security-expert` |
| P1-28 | Egress spec successor `-egress-spec3.md` (`Supersedes:` `-egress-spec.md`, `-egress-spec2.md`), option C | `architect` | gat | — | `analyst`; `security-expert` advisory |
| P1-45 | Edit `ccg/answer.py`, `answer-v1.schema.json`, `test_answer.py` per `-answer-spec2.md` | `tdd-engineer` | ans | 00 | `analyst` |
| **W1: extraction** | | | | | |
| P1-03 | System manifest loader and unit discovery (scope = census scope; `remote` rejected) | `tdd-engineer` | ext | 01, 48 | `analyst` |
| P1-04 | Joern extractor port (`extract.sc`, runner through `offline-run.sh`), bundle-equivalent to the slice | `graph-dba` | ext | 01, 03 | `analyst` |
| P1-05 | Python and TS/JS normalizer port and bundle writer, equivalent to the slice's stored bundles (two dispatches) | `tdd-engineer` | ext | 04, 02 | `analyst` |
| P1-06 | `Cond.code` literal masking (F-9) — **before the first fixture build** | `graph-dba` | ext | 05 | `analyst` |
| P1-07 | Writer done-condition (F-10): W2, W3, W4, SA-1 writer side, `gate_argv`, per-entry HMAC | `tdd-engineer` | ext | 05, 02, 13 | `analyst`; `security-expert` advisory |
| P1-08 | `jssrc2cpg` ignore override, `skippedFiles`, coverage probe | `graph-dba` | ext | 04, 05 | `analyst` |
| P1-09 | Projection fix (R1, R2, R3, projection-side causes, both join fixes); golden `DATA`/`CONTROLLED_BY` sets | `graph-dba` | ext | 04, 05 | `analyst`; `data-scientist` |
| P1-10 | Shell extractor | `coder` | ext | 01, 07 | `analyst` |
| P1-11 | Config, Compose, `.mcp.json`, Kiro, Dockerfile launch extractors; SA-3 role allow-list and `keyPath` (F-6) | `coder` | ext | 07 | `analyst`; `security-expert` advisory |
| P1-12 | Prompt/Markdown extractor classifying by `eval-manifest.yaml` globs | `coder` | ext | 07 | `analyst` |
| P1-14 | Encode `bindParts` (schema, `bundle.py`, config writer) | `tdd-engineer` | ext | 13, 11, 45 | `analyst` |
| P1-15 | Indexer adoption: `scip-typescript` (recall) and `scip-python` (precision), offline | `coder` + `graph-dba` | ext | 04, 05, 48b, `-ml` | `analyst`; `data-scientist` |
| **W2: linking** | | | | | |
| P1-16 | Pack runtime, conformance over the stored slice bundles | `tdd-engineer` | lnk | 01 | `analyst` |
| P1-46 | Shell `mcp-call` consumer pack (+ `negatives.yaml`), leaf-harness update | `tdd-engineer` | lnk | 16, 10 | `analyst` |
| P1-17 | Linker: matchers, statuses, bindings (Compose and `.mcp.json` are linker code, D10), example configs, prompt mentions | `tdd-engineer` | lnk | 16, 46, 07, 11, 14, `-graph` | `analyst` |
| P1-18 | Assembler: bundles to graph rows, `mask_http_segments` import, `MOUNTED_BY`, `SkippedFile`, `Unit.flowAvailable`, pack-set identity | `graph-dba` | lnk | 16, 17, `-graph` | `analyst` |
| P1-19 | Store loader, DDL, ACL users and test, swap, stamp, fuzz test, re-derived `TIMEOUT_MAX` and lock ttl, size re-estimate | `graph-dba` | lnk | 18, 48, 02, `-graph` | `analyst` |
| P1-44 | Build dispatch by unit kind (Joern py/ts, shell, config, Markdown, SCIP merge if P1-15 is adopted), failure isolated per unit | `coder` | lnk | 03, 04, 05, 10, 11, 12 (15 if adopted) | `analyst` |
| P1-20 | Refresh (on demand, local heads only), stale-kept, not-mapped | `coder` | lnk | 19, 44, 02, 03, 29 | `analyst` |
| **W3: answers** | | | | | |
| P1-21 | Reader layer and anchors, carry-over C3 | `tdd-engineer` | ans | 19, 45 | `analyst`; `graph-dba` on queries |
| P1-22 | BFS rows, F-12, depth bound and `unexplored`, version bump | `tdd-engineer` | ans | 21 | `analyst` |
| P1-23 | Flow fallback, distance and grouping, reverse question, answer builder | `tdd-engineer` | ans | 22, 18 | `analyst` |
| P1-39 | Staleness, stamp and the superset measurement on dev flow items | `tdd-engineer` | ans | 23, 20 | `analyst`; `data-scientist` on the measurement |
| P1-24 | Renderers: grouped text and Markdown report, F-3 fencing, F-4 token screen | `coder` | ans | 45 | `analyst` |
| P1-25 | `ccg` MCP server, CLI, `raw_query`, pin-and-replay | `coder` | ans | 23, 39, 24, 19, 20, 02 | `analyst` |
| P1-26 | Plain-language front: parsers, interpreter, resolver, cache, ambiguity, summary | `coder` | ans | 29, 39, 24, 25, 45, `-ml` | `analyst`; `data-scientist` |
| P1-27 | `teco` v1 seam and `cpg-analysis` skill update | `cobb` | ans | 25 | `analyst` |
| **W4: egress and gates** | | | | | |
| P1-29 | Broker core: policy `R0`, `R1`, `R3`–`R7`, `R99`; runs; structured calls; audit | `devops` | gat | 28, 03 | `analyst`; `security-expert` advisory |
| P1-31 | AC-8 harness: no-route network, logging detector, window, rows 8.1–8.16 | `devops` | gat | 29, 18 | `analyst` |
| P1-40 | AC-8 part 1 run: full refresh plus queries of a mixed fixture system | `devops` | gat | 31, 20, 25, 26 | `analyst` |
| P1-32 | Eval tooling verbs, then the AC-5 gate harness (two dispatches) | `qa-engineer` | gat | 25, 20 (harness half) | `analyst`; `data-scientist` |
| P1-33 | P1 test plan `docs/test-plans/connected-code-graph-p1.md` | `qa-engineer` | gat | 00 | `analyst` |
| P1-41 | Fixture gate, part 1: the 91 P1 rows and scenarios | `qa-engineer` | gat | 33, 20, 25, 06, 02, 08, 10, 12, 14 | `analyst` |
| P1-42 | Fixture gate, part 2: AC-level assertions (AC-6, 9, 16, 17, 18, 21, 25) | `qa-engineer` | gat | 41, 26 | `analyst` |
| P1-34 | Flow re-measurement (§5): re-point `s28_run.py`, entry check, fresh draw only if entered, decision record | `data-scientist` + `qa-engineer` | gat | 09, `-ml` | `analyst` |
| P1-35 | AC-24 smoke harness and run | `qa-engineer` | gat | 26, 29, `-ml` | `data-scientist` |
| P1-47 | Runtime-oracle capture on an isolated instance | `qa-engineer` | gat | 48, 02, `-ml` | `analyst` |
| P1-36 | Noise sampling and runtime-oracle tooling, the measurement of the ratified thresholds | `data-scientist` | gat | 32, 47, `-ml` | `analyst` |
| P1-37 | AC-5 first gate run (HTTP, MCP, scripts, wiring screened, flow per §5) | `qa-engineer` | gat | 32, 42, 34, 36, 47, pre-registration record (rule below) | `analyst`; `data-scientist` |
| P1-43 | AC-5 re-gate (contingent on a miss): its own brief, two isolated key writers | `qa-engineer` | gat | 37 | `analyst`; `data-scientist` |
| P1-38 | Phase exit: AC-closure table, `analyst` impl review, spike dispositions, inputs for the close docs | `teco` | gat | all built steps | `analyst` |

**Critical path** (longest chain): 01 → 03 → 04 → 05 → 07 → 11 → 17 → 18 → 19 → 20 → 25 → 41 → 42 → 37 → 38
(P1-17 also needs 46 and 14; 19 needs 02 and 48; 20 also needs 44, which runs beside 19). Schedulable first: 00, 01,
02, 13, 28, 48 and the two notes of §9; the rule-pack runtime (16) needs only those, and the broker core (29) those plus 03.

**Core freeze before the first gate.** `coreTreeHash` covers every tracked file under `cpg/` except `cpg/eval/**`,
`cpg/packs/**`, `cpg/tests/**` and `**/*.md` (`-eval-protocol-ml.md` §12.3), and a first-contact run needs the record's
commit to precede the build with no uncommitted change under `cpg/`. The `ac5-gate-1` pre-registration record
therefore Needs **every step that edits a tracked file outside those exclusions**: 01, 02, 03–12, 14, 15, 16–26,
28, 29, 31, 34 (`cpg/spikes/**` is hashed), 39, 44, 45, 48, 48b. It also Needs **32 (both dispatches), 35, 36 and 47**,
which are not hashed (`cpg/eval/**` is excluded) but are written before the record and pinned at its commit
(`-p1-ml.md` §6.1, §6.2: the gate harness is the one at the record's commit, and an edit after it makes the run "after
adaptation"). Its commit is the freeze. P1-15 either lands before it or
is declared out of P1's first gate (the gate then runs with the Joern call graph only). A step that edits any of those
files after the record makes the run "after adaptation" (the AC-5 result is lost); P1-43's re-gate writes a new record.

**Before the record** (actors named; `-p1-ml.md` §6.1–§6.2, §8, §9). (1) `teco` has the custodian run the freeze-time
`check-report` (the P1-32a verb) over the path set; a hit is triaged by the custodian and a contacted cluster moves to
dev. (2) `tico` takes the 40-item redo (`-p1-ml.md` §9 Q-D) to the stakeholder; the custodian re-records and commits the
held-out manifest if a held-out file changed. (3) `teco` commits each decision or pre-registration record before the act
it governs: `ccg-p1-indexer-v1` (P1-15), `ccg-p1-flow-fallback-v1` (P1-39), `ccg-p1-flow-python-v1` (P1-34, if entered),
`ccg-oracle-v1` (P1-47), `ccg-prereg-ac24-v1` (P1-35). `ccg-noise-ratify-v1` is `tico`'s requirements-log entry, which
the next gate's record cites.

## 8. Test strategy

The method is the proposal's §5 and the ML notes'; this section records only what binds the build.

- **Altitudes** (proposal §5.1). Unit: each extractor ships fixture sources plus the expected bundle (golden per
  frontend, where a Joern upgrade is caught); each pack is table-driven; the projection has golden `DATA` and
  `CONTROLLED_BY` sets for small methods. Contract: bundle, graph and canonical-answer schemas validate in CI; the
  engine's output passes `ccg.answer` validation (V1–V16). Integration: the fixture system's 91 P1 rows (P1-41) and
  the proving repo's own anchors on `cpg/eval/dev/`.
- **Equivalence before improvement.** A ported module (P1-04, P1-05, P1-16–P1-18, P1-21, P1-22) first reproduces
  the slice's output on the slice corpus, then changes behaviour in a separate, later dispatch. The stored slice
  bundles are in the gitignored work area (`cpg/.ccg-work/vertical-slice/s14a`), so these tests carry a `slice`
  marker, skip when the artifact is absent, and record its sha256; P1-04's first dispatch regenerates it at the
  slice's pinned commit if absent.
- **TDD ordering.** The sub-plans list each step's behaviours as an ordered red-to-green list; `tdd-engineer`
  steps follow it literally, `coder` steps use it as the acceptance list. Every code step ends with a mutation
  check on the contract boundary it implements (surviving mutants are listed in the step's report), as P0's did.
- **Determinism.** Two builds of the same commits give byte-identical bundles (header `builtAt` and
  `sandboxAuditIds` exempt, `-bundle-spec3.md` §8.5) and identical snapshots minus timestamps and `Snapshot.auditSetHash`; the same
  structured request through MCP, CLI and the raw recipe gives identical canonical facts (AC-16, pin-and-replay,
  `-eval-protocol-ml.md` §10.6). Tested in P1-25 and P1-42.
- **Policy and trust.** AC-8 part 1 is one sandboxed scenario (full refresh plus queries; no route out, the logging
  proxy a detector; `-egress-spec3.md` §8) built in P1-31 and run in P1-40 (option C: the `R2` clauses and the proxied rows are vacuous and reported). AC-21 is a write through
  `raw_query` and a direct `GRAPH.QUERY`/`GRAPH.PROFILE` with the `ccg_raw_<system>` credential, both refused by
  the database with the stamp unchanged (P1-25, P1-42). AC-9b is a fixture unit made unbuildable and one never
  built (P1-20, P1-41).
- **Gates that use blind data** are P1-34 (if its entry check passes), P1-35, P1-37 and P1-43. Builders never see held-out items; a held-out item a
  builder debugged against moves to dev and the run is reported as run. P1-37's first run is the result.
- **Live tests** run on `ccg-falkordb` (port 6390) or a throwaway instance, never the shared `falkordb-dev` (6379). Steps that run live in parallel (P1-19, P1-21, P1-25, P1-39, P1-40, P1-41, P1-42, P1-47) each use their own throwaway instance (`infra/ccg-falkordb.sh --throwaway`, P1-48) or the P0 §5.1 reservation rule, because they share fixed key names and `ACL`/`GRAPH.CONFIG` writes.
- **A skipped test is not a pass** (D9): an equivalence or conformance gate records the artifact sha256 and `0 skipped`.
- **AC map at phase exit.** P1-38 produces one table: each AC of the proposal's P1 row, its closing step, and
  the evidence file. An AC that depends on a measurement that did not come out (the `flow` stratum; AC-25's
  confirmed-ness; AC-8's `R2` rows under option C) is listed "reported not closed" with the reason.

## 9. Co-owned notes and other-agent units

For `teco` to dispatch. None is written here. Each note is a precondition of the steps named; steps not named
are not blocked. Slugs follow the closed role set (`-ml`, `-graph`).

| Unit | Owner | Document | Blocks | Question it must answer |
|---|---|---|---|---|
| N-ML | `data-scientist` | `docs/plans/connected-code-graph-p1-ml.md` | P1-15, P1-26, P1-34, P1-35, P1-36, P1-37, P1-47 | **1. Indexers:** how to measure `scip-typescript` and `scip-python` *offline* (P0's efficacy figures ran with dependencies and a biased reference; `-p0-ml.md` §3.2 caveat), the merge rule into resolved/`possible` calls, fold-or-list per language at the 0.95 precision threshold, and the pass/adopt criterion given that no language passed the recall screen. **2. Flow re-measurement:** fresh-draw size and seed prefix for python (≥ 120 pairs, both configurations), labellers and adjudication, the entry-check value in §5 derived from the §8.4 operating curve (it is expected to be above what a 30-method dev set can show, and "not run" is a legitimate outcome), whether javascript is measured at all, a ruling on the superset measurement of P1-39 (when is the flag-false fallback "safe"), and **whether projecting the transitive CDG** (python's 9 `definition`-side misses; ruled in `-p1-ml.md` §3.4 as synthetic-cond transparency, ceiling 467/481) is a legitimate flow pair under §7.3's direct-edge rule. **3. Interpreter:** the local model, prompt, grammar and sampling for the plain-language front; the P1 smoke protocol (EP §10.5); the `ccg-prereg-ac24-v1` record; the did-you-mean `CAP` (C7); what "the local model that actually interprets" means under mixed systems. **4. Noise:** sampling and labelling procedure at P1 (EP §11), the ratification procedure for possible-precision ≥ 0.20, p90 burden ≤ 100 and reduction ≥ 5× (all estimates from `-ml.md` §4.2, not measurements). **5. First-gate pre-registration and scope:** the `ccg-prereg-v1` record for `ac5-gate-1`; what counts as a builder "contact" with held-out items during P1; that the system scope equals the census scope (§2) and how the harness treats a listed item outside it (flag, never score); and the runtime-oracle definition and capture protocol P1-47 follows (ML §4.1: access logs or client instrumentation during the suites and a scripted storefront session, on an isolated instance). **6. Open P0 items:** the scope-overlap question and the catch-all-route clause (§4.3), answered before any fresh key-writing. |
| N-GRAPH | `graph-dba` | `docs/plans/connected-code-graph-p1-graph.md` | P1-18, P1-19, P1-21 | **1.** Graph deltas the P1 loaders need beyond `-p0-graph.md` v1: `ConfigEntry` and `Dependency` (nodes only; no `DEPENDS_ON`/`FROM_DEP` hop before P3, P1-00), the `launchFact` fields carried (G-1/F-11: `port`, `mountPath`, `service`, `argv[0]`, role-relevant pairs, never whole `args`), `Unit.flowAvailable` and `Unit` staleness properties, pack-set identity, `Snapshot` fields. **2.** O-9: the reason on an `mcp_tool` Link a binding selects, and the representation of a framework-default port (no `ConfigEntry` to hang `BOUND_BY` on); the representation of Compose and `.mcp.json` linker bindings (D10). **3.** O-10: where pack-set ids, versions and hashes live in a snapshot. **4.** The reads C3 needs (owner join, `LIFTED_VIA` unit/repo/siteId, edge `reason`) as indexed queries. **5.** `Link` answer size (answer spec §10 item 5) and the memory re-estimate method for P1-19. **6.** The per-system ACL users `ccg_<role>_<s>` and their lines, written in `-p1-graph.md` (not an edit of `-graph.md`), and the system-name whitelist's home. |
| N-DEVOPS-1 | `devops` | P1-01, P1-02, P1-48, P1-48b | P1-03.., P1-20, P1-41 | P1-01: pins and layout. P1-48: image, instance flags, throwaway mode. P1-48b: the runner's indexer entries, after `-egress-spec3.md` (P1-28). P1-02: **file mode, location, retention of bundles, the HMAC key and the four ACL credentials; whether any LLM-backed tool may read a bundle directly** (F-1; stakeholder decision 2026-10-29: FR-15 covers the graph and answers, not local bundle files). Default until decided: bundles only in the gitignored work area. |
| N-DEVOPS-2 | `devops` | P1-29, P1-31, P1-40 | W4 | The broker core, the AC-8 harness and its run, after `-egress-spec3.md` (P1-28). |
| N-COBB | `cobb` | P1-27 | — | The `teco` v1 seam: architect and implementer briefs carry an impact answer and its stamp in place of the `CPG: <graph>, built …` line **for systems `ccg` covers**; `cpg-analysis`'s usage section for the `ccg` tools, its freshness recipe kept for `cpg_*` graphs (proposal §3.13). Which prompt files change and the check that a brief without a stamp is refused. |
| N-SPECS | `architect` | P1-00, P1-13, P1-28 | per the table of §7 | Three successor specs; none waits on the stakeholder. |
| N-ADV | `security-expert` (advisory, optional) | P1-02, P1-07, P1-11, P1-13, P1-28/29 | none | A written view on bundle-at-rest and credential custody, the HMAC and `bindValue` writer rules, `bindParts`, and the broker core; `devops`, `architect` keep final say. |
| N-TICO | `tico` | `docs/requirements/connected-code-graph.md` | none | Record the egress Q1 answer (option C for P1, git-fetch deferred with its trigger) and the four P0 stakeholder decisions still missing from the requirements (§4.2), and the `cpg/`-excluded scope (§2), carry open question Q-E (how AC-8 closes without the fetch path), and record the Q-C and Q-D answers (`-p1-ml.md` §5.3, §9). |

## 10. Risks & open questions

**Process rule for every brief** (from the S50 slip): "no writes to any shared database; use `ccg-falkordb`
(6390) or a throwaway instance" and "builders see dev and fixtures only".

**Risks.**
- **R-1 The flow flag stays false for both languages** (expected, §5): impact through Python/TS is call-level, its
  call-continuation items `possible` (`-answer-spec2.md` §5), with more noise than a flow-aware answer. Mitigation: the noise gate reports by distance band; the
  stratum is not claimed closed. Impact: AC-25's control half (§5).
- **R-2 Indexer adoption offline is unmeasured** (P0's efficacy ran with dependencies; the scip tools need
  `npm ci` and a Python environment, which only the operator provisions, outside the system, with no egress path added in P1).
  Mitigation: P1-15 reports the offline figure first and adopts only on it; the fallback is the Joern call graph
  with `possible` name-match items (the P0 baseline).
- **R-3 The `jssrc2cpg` ignore override may not exist** (spec4 Pass 5 open question 1). Fallback specified:
  `frontend-ignore` listing; the 5 `salesperson/src/i18n/` sources stay unmapped and 10 call sites keep empty targets.
- **R-4 Host cost.** Python extract + normalize measured 4.1–4.2× the Joern run on the slice, on a 6 GB host.
  P1-20 reports a full-refresh wall time and peak memory; if it exceeds the budget agreed in P1-20's brief, bundle
  caching by tree id (a facts-neutral optimization, proposal §3.11) is the lever, not a design change.
- **R-5 Gate budget.** The AC-5 re-gate needs ≥ 35 fresh-cluster items written by two isolated agents (E12).
  A first-run miss therefore costs a key-writing round (about 18–25 headless sessions in P0, `-p0-2.md` §6.6).
  Mitigation: P1-41/P1-42's fixture gate and the dev items run first, so a first-run miss is least likely. The re-gate is its own step (P1-43) with its own brief.
- **R-6 Stale pins.** The ledger (2026-11-06) records the plan and proposal pinned to EP v1.7 while the current
  versions are EP v1.8 and EP2 v1.5, and `cpg/eval/eval-manifest.yaml` pins EP and EP2 at 1.4 (a build-time
  record, never bumped). P1-32 asserts that the pin strings its harness reads are the ones the manifest records
  and reports any difference from the committed documents rather than editing the record.
- **R-7 Layout churn.** Widening `packages.find` (D8) and moving extractors to `cpg/extractors/` may break the
  editable install of existing steps' tests. P1-01 runs the full `cpg` suite before and after.

**Open question for the stakeholder** (the plan proceeds on the stated default):
- **Q-E (AC-8 without the fetch path).** The proposal's P1 row lists AC-8 part 1 as "git-remote fetch only" and the
  requirement closes AC-8 only when all of it is built. With git fetch deferred to an unscheduled trigger (§6), how
  does AC-8 close at P2 exit? Carried to the stakeholder by `tico`; the proposal and requirements are not edited here.

## 11. Documentation impact (feature/phase close)

Items for `teco`'s close, not for build units (root `AGENTS.md`, live-spec and closeout bullets). `cpg/` has
no live spec; the feature's close (after P2) owes a whole-component spec, checked by `analyst`, before it.

- **Root `AGENTS.md`**, the `cpg/` bullet: rewritten in place (not appended) to the shipped state: the `ccg`
  package, the engine and clients, the broker and sandbox, the dedicated `ccg-falkordb` instance, the `cypher-mcp`
  boundary kept. The "Structure" and "Component docs" rows follow. Check the size bar (`awk` line in root
  `AGENTS.md`).
- **`cpg/README.md`**: Layout table (extractors, clients, `ccg/` modules, `registry/flow-availability.yaml`),
  Setup extras (D8), Tests (new markers `slice`, `live`), a "Running a build and an answer" section, the
  vertical-slice directory marked frozen (D7).
- **`skills/joern-cpg/SKILL.md`**: states that the build stage is reused by the `ccg` Joern extractor and that the
  `neo4jcsv` path remains for optional `cpg_*` graphs. **`skills/cpg-analysis/SKILL.md`**: updated by P1-27 into
  the usage skill for the `ccg` tools, freshness recipe kept; `skills/README.md` row.
- **`claude/teco`** prompt, `claude/AGENTS.md` catalog row if wording changes (P1-27, `cobb`).
- **`docs/HISTORY.md`**: one entry per delivered wave and a P1 close entry citing the AC-closure table
  (P1-38), the Q1 decision, the flow-gate outcome, the spike dispositions, and the reports.
  **`docs/BACKLOG.md`** C-1001: rewritten in place to P3 onward, plus the new items this phase creates: the
  TS/JS flow remedy (`switch`-case cond synthesis), AC-25's control half if P1-39's walk leaves it open, the deferred git-fetch path (trigger: a manifest first naming an https `remote`; design inputs in `-egress-spec3.md`), and deletion of `cpg/spikes/vertical-slice/` with its dispositions (re-point or vendor `spikes/store/{bfs,synth,lock_replay}.py` and `store/tests/test_store_spikes.py`, re-point `frontend-fitness/s28_run.py`).
- **Archive flips at P1 close** (by owner): this index and its four sub-plans (`architect`), the `-p1-ml.md` and
  `-p1-graph.md` notes (`data-scientist`, `graph-dba`), the P1 test plan and report (`qa-engineer`). Specs the
  successors supersede flip to `superseded` (`-egress-spec.md`, `-egress-spec2.md`) or gain `Extended by:`
  (`-answer-spec.md`, `-bundle-spec4.md`).
- **Proposal** `docs/plans/connected-code-graph.md`: the P1 row and §3.12 gain an `Extended by:`-style pointer
  or a v3.5 edit only if a decision here (D1–D11, §5, §6) contradicts them; the one contradiction found is AC-8's
  closure under option C (open question Q-E), which is the stakeholder's to resolve before any edit. Requirements: `tico` records the Q1 answer and the four missing P0 decisions.
