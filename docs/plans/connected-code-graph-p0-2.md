# Connected code graph — phase P0 implementation plan (successor)

> **Status:** active · **Owner:** `architect` · **Tracks:** C-1001 (M10) · **Version:** 1.3 · **Supersedes:** `docs/plans/connected-code-graph-p0.md` · **Extends:** `docs/plans/connected-code-graph.md` · **Reviews:** `docs/reviews/connected-code-graph-p0.md`

*2026-10-29 (v1.0): successor to plan v1.3 (the "plan v1.4" that `docs/plans/connected-code-graph-p0-ml.md` §11 addresses). Absorbs the S05 ML note's step outputs, the S03/S04 specs' downstream items, the coordination ledger's carry-forwards, S04 Pass 3 errata, and S06's hand-offs (eval-protocol v1.0, §14 here; its gate is in flight). Disposition table: §16.*
*2026-10-30 (v1.1): folded eval-protocol v1.2 §14/§15, `docs/plans/connected-code-graph-bundle-spec2.md` v1.3 §3–§4 and `docs/plans/connected-code-graph-egress-spec2.md` v1.2 §3 (which replaces errata E1–E2), plus the ledger's routings to 2026-10-30; traceability in §16.1.*
*2026-10-30 (v1.2): review Pass 3 (P3-1–P3-11, n-a–n-d, the recount) and `docs/plans/connected-code-graph-eval-protocol2-ml.md` §3 (F-8, step 8a) folded, plus S10 follow-up notes for `-spec3`; traceability rows P3-*, EP2-*, R-* in §16.1.*
*2026-11-01 (v1.3): D5, the S15 and S16 rows aligned with `-spec3` (consolidation before S15; an SL-7-forced change is a `-spec4`).*

## 0. Goal & scope

**Goal.** Unchanged from v1.3: execute phase P0 of `docs/plans/connected-code-graph.md` §4 —
deliver (a)–(h), settle every §6.2 "for P0 to settle" decision with a recorded measurement, and
either freeze bundle schema v1 and the graph model or record that they stay provisional until P1
exit. P0 gates no AC; its exit condition is the ledger's P0 definition of done
(`docs/plans/connected-code-graph-coordination.md`, "Phase P0").

**In scope:** the steps in §3; the dedicated FalkorDB instance and the toolchain image that P0's
measurements need; the durable code those deliverables require (schemas, validators, sandbox,
fixtures, ground-truth tooling); the results notes that record every measurement.

**Out of scope (named handoffs to P1 or later, never built here):** unit discovery; production
extractors and normalizers; the rule-pack runtime; the linker; the store loader with swap, stamp
and ACL users; the answer engine; the `ccg` MCP server, CLI and `raw_query`; the egress broker
implementation and AC-8's logging proxy; plain-language interpretation; the id-hash fuzz test
(proposal §5.1, needs the P1 loader); the Arduino slot; the four would-be fallback analysers of
`docs/plans/connected-code-graph-p0-ml.md` §5.2 (decided in §2 D13). The handoff list is §15.

### 0.1 Why a successor, and how to read it

- **Collision rule 5.** v1.3 has been gated (review Pass 2 → approve with suggestions; `teco`'s
  stopping-item check, ledger U19) and is being executed against (S01–S05, S07 and S09 accepted;
  S06, S08 and S10 in flight at v1.0; all accepted by v1.1). So the selector answers **yes**, and the change is a successor, not an
  in-place v1.4.
- **Supersedes, not Extends.** This document **replaces** v1.3 as the operative plan, so an
  implementer reads one plan. Extends would be wrong: S24, S25, S27 and S28 are *amended*, which is
  replacement, not addition, and an `Extends:` pair would leave two documents with competing rows
  for the same step. v1.3 now carries `Superseded by:` and `Status: superseded`; its §15 (review
  dispositions, Passes 1–2) stays there as history and is not repeated here.
- **Name: `connected-code-graph-p0-2.md`.** Rule 5 puts the ordinal on the slug. The bare form
  `connected-code-graph-p02` reads as "phase 02", in a document family whose whole subject is
  phases (a `connected-code-graph-p1.md` will sit beside it). The hyphen keeps `p0` an intact token,
  so the file sorts and globs with the other `connected-code-graph-p0*` documents and reads as
  "the P0 plan, second document". `-2` is not in the closed role set, so it is part of the topic
  slug, as the grammar requires. The residual misreading ("phases 0–2") is answered by this header's
  `Supersedes:` pointer. Downstream documents already named by v1.3 keep their names (§10); the gate
  of this document is the next `## Pass N` of `docs/reviews/connected-code-graph-p0.md`, because a
  re-review's value is reading its passes together and this is the same P0 plan topic.
- **Step ids and meanings are stable.** Every v1.3 id keeps its meaning; ledger units, commits and
  reviews cite them. New ids are **S56, S57, S58** (§3). They collide with nothing: not with the ids
  in `git log --oneline`, not in the coordination ledger, not in any spec's numbering, and not with
  the requirements' decision ids S1–S38 (S19 and S29 were rejected for that reason). The ML note's
  `N0`–`N7` labels are never used as step ids; §16 maps each to the step it became.
- **Accepted steps are settled.** For S01–S10 and S58 (all accepted by v1.1) this plan does not
  re-decide the contract; the S08 and S10 follow-up rounds are the specs' own obligations (ES2 §3,
  `-spec2` §4.1), recorded in their rows. Where a spec now controls a v1.3 plan section,
  this plan says so and points to the spec section, and does not restate it. **Spec controls
  plan** wherever the two differ for those steps.

## 1. Context & findings

CPG: considered, not relevant — this successor is docs work against plans, specs and the ledger; `cpg_falkorchat` is stale (built 2026-10-15) and P0's only code-level contact with that component is the slice, which builds its own bundles from source; the ground-truth agents are forbidden to use it (§6.3).

### 1.1 State of the steps (2026-10-30, from the ledger)

| State | Steps | Controlling document for the contract |
|---|---|---|
| **Accepted** | S01 (U20), S02 (U21, `ccg-toolchain:d3184effb566`), S03 (U22, bundle spec v1.3, with `-spec2` v1.3, U32), S04 (U23, egress spec v1.3, with ES2 v1.2, U32), S05 (U24, `docs/plans/connected-code-graph-p0-ml.md` v1.3), S06 (U26, eval-protocol v1.2), S07 (U25), S08 (U28, Pass 2), S09 (U27, commit `6f4e34f9`; `pyproject.toml` extras frozen), S10 (U30), S58 (U31, dispositions in `-spec2`) | the delivered artifact and its gate record in `docs/reviews/connected-code-graph-p0-impl.md` |
| **Accepted, follow-up round dispatched** (ledger 2026-10-30) | S10 (`-spec2` §4.1, before S15), S08 (ES2 §3 plus its Pass 2 carry-overs) | the §3 row; the round's own gate is `analyst` |
| **Not started** | everything else | this document |

S10's R11 brief ("enforce a minimum-content rule only if the spec states one") found none, so the
guard is S11's (§2 D15).

### 1.2 What P0 builds on

| Asset | Fact that shapes P0 |
|---|---|
| `skills/joern-cpg/scripts/` (`build-cpg.sh`, `git-provenance.sh`, `joern-env.sh`, `pipeline.sh`) | The build stage and provenance capture are reused by the slice's extraction (S11). Its neo4jcsv export and `--load` are **not** used by P0: no P0 step loads a `cpg_*` graph. |
| `skills/joern-cpg/references/cpg-model.md` | The verified `pysrc2cpg` topology (34% resolved call sites, intraprocedural `REACHING_DEF`, `FILENAME` only on `METHOD`/`TYPE_DECL`) is the starting point for the slice's projection script. It is Python-only, so the TS half of the slice is new ground. |
| `cpg/` | Now holds S01's `cpg/infra/` (`ccg-falkordb.sh`, `README.md`, `tests/`), S02's `cpg/infra/toolchain/`, and S09's `pyproject.toml` (extras frozen), `README.md`, `.gitignore`, `ccg/ids.py`, `tests/`. |
| Joern v4.0.579 | Host copy at `~/joern/joern-cli`; the image pins the same release by checksum (S02). |
| `falkordb-dev` (`falkordb/falkordb:v4.18.11`, port 6379) | Shared by other sessions and components. **No P0 step writes to it** (§5.1). |
| `ccg-falkordb` (S01) | Up on `127.0.0.1:6390`, same image tag as `falkordb-dev`, so every quirk in `claude/graph-dba/falkordb-quirks.md` (S07-verified) still applies. |
| `ccg-toolchain:d3184effb566` (S02) | Stages `apt-base` → `fetch` → `toolchain`; apt only via `apt-snapshot-install`; `ccg-versions` with a drop-in directory `/usr/local/lib/ccg-versions.d/`. The tag is the content hash of the build inputs. |

### 1.3 Repo and host facts that drive the design (unchanged from v1.3 unless marked)

- Only Python, TS/JS and shell exist in the proving repo. Java, Kotlin, C#, Go and C/C++ have
  **no** in-repo code to measure, so P0(a) needs external corpora (§2 D4).
- The OpenTelemetry demo is AC-7's codebase and must stay unseen until its per-stratum first
  contact (AC-7; ML §4.5). The held-back third codebase must stay unseen by anyone building the
  system. **Neither may be used as a P0(a) corpus**; the full exclusion list is ML note §3.3.
- The host is a WSL2 VM capped at **6 GB RAM / 8 vCPUs** by `C:\Users\mauri\.wslconfig`, which
  protects LM Studio on the 16 GB Windows host. P0's measurements fit only if instance-heavy and
  JVM-heavy steps are serialized (§5.1). Verifying the ~18 GB workplace peak (proposal §3.5) is
  **not** a P0 goal.
- **Changed since v1.3:** on this host the WSL interop binfmt entry is `WSLInterop-late`, not the
  `WSLInterop` path v1.3 named, and `/dev/vsock` exists. Interop detection is egress spec §7.1.1,
  which controls (§2 D6). Only the container backend runs here.
- **Changed since v1.3:** `joern-parse` exits 0 on an empty CPG in some configurations (egress spec
  §9 H12, §11 R11). Exit status alone never certifies a parse (§2 D15).
- **Changed since v1.3:** `ccg-falkordb` stopped answering `PING` for minutes during one
  100k-edge single-query write (review §S03 graph-dba check, "Incident"); cause not established.
  P0 loads stay at ≤ 20k rows per batch until S23 probes it (§2 D16).
- **Changed since v1.3:** a nonzero `TIMEOUT_MAX` also kills writes, with nothing committed
  (S07-verified quirks entry, TIMEOUT). Loads under a nonzero cap pass an explicit `timeout` (§2 D16).

## 2. Design & rationale

D1–D12 keep v1.3's numbers. Where a decision is now owned by an accepted spec or note, the entry is
a pointer. D13–D17 are new.

**D1 — Three kinds of P0 artifact, and only three.**
- **Contract**: a spec under `docs/plans/` written by `architect`, plus its machine-readable form
  (JSON Schema) and a validator under `cpg/ccg/`, which are durable code.
- **Measurement spike**: throwaway code under `cpg/spikes/<name>/`, plus a durable results note
  under `docs/test-reports/` or `docs/plans/…-graph.md`, plus raw results committed as JSON beside
  the spike. Spike code is never imported by `cpg/ccg/`. Whether it is deleted or ported is P1's
  plan's call. **A spike without its results note is not done.**
- **Durable code**: `cpg/ccg/` (schemas, ids, validators), `cpg/sandbox/`, `cpg/infra/`,
  `cpg/fixtures/`, `cpg/eval/`. It is tested (§9) and is the base P1 builds on.

Rejected: writing the slice as P1-quality durable code; the slice exists to fail cheaply.

**D2 — Code layout under `cpg/`** (extends proposal §3.13; P1 may move anything below `cpg/ccg/`).

```
cpg/
  pyproject.toml            package `ccg`, Python 3.12; extras dev/live/spike, frozen (S09)
  README.md                 setup, tests, layout (S09)
  .gitignore                `.ccg-work/` (S09); `.audit/` (S08, egress spec §10; see §5.2)
  ccg/
    ids.py                  canonical tuple → 32-hex id (S09, accepted)
    bundle.py               bundle JSONL read/validate/write, deterministic order (S10)
    answer.py               canonical answer serialization (S34)
    schemas/                bundle-v1, rule-pack-v1, answer-v1, fixture-manifest-v1 (*.schema.json)
    store/ddl_v1.cypher     graph model v1 DDL (S17)
  sandbox/                  offline runner + flag lists + tests + README (S08)
  infra/                    ccg-falkordb.sh + tests (S01); toolchain/ Dockerfile + build.sh (S02, S24)
  packs/<name>/pack.yaml    paper packs, `status: paper` (S31, S32)
  fixtures/system1/         fx-svc (Python), fx-web (TS), fx-lib (Java), expected/*.yaml, materialize.py
  eval/                     EVAL_COMMIT, eval-manifest.yaml, kit-manifest.json, heldout-manifest.json, dev/, tools/, headless/
  spikes/                   frontend-fitness/, vertical-slice/, store/   (throwaway, D1)
    frontend-fitness/       corpora.yaml, frame/, truth/, sample/, labels/, adjudication/, results/, matcher-table.md (S25–S28, S56, S57)
  tests/                    durable-code tests; `live` ones need Docker/FalkorDB/Joern
```

Tests run with `cpg/` as the working directory (`cd cpg && .venv/bin/python -m pytest`), per root
`AGENTS.md`. `live` tests are deselected by default (`-m "not live"`).

**D3 — P0(a) is split into a1 and a2** (unchanged). a1 (S25–S27, with S56/S57 feeding it) is
Joern-only and independent of the slice; a2 (S28) runs the slice's projection script on every
corpus and decides `flowAvailable`. A sequencing fact, not a scope change.

**D4 — Corpora for P0(a).** Python and TS/JS: the proving repo's `falkor-chat/server` and
`salesperson/`. The other five languages: one public OSS project each, pinned by SHA. **The
criteria C1–C8, the exclusion rules 1–8, the reverse rule, the selection rule and the per-candidate
order are ML note §3.2–§3.3, which control.** `graph-dba` proposes; `data-scientist` confirms at
S25's gate. Fetching corpora and running truth tools is hosted download of public code, outside the
sandbox. Rejected: the OpenTelemetry demo "because it has every language".

**D5 — The slice freezes by the proposal's binary rule.** Unchanged: freeze iff all of S14's
SL-1–SL-7 pass; otherwise v1 stays provisional until P1 exit. The contract S15 freezes is
**`docs/plans/connected-code-graph-bundle-spec3.md`** ("`-spec3`"): the consolidation of the bundle
spec v1.3 and `-spec2` v1.3, delivered by S16 **before S15** and required, gated and checked at P0
exit (`bundleSchemaVersion` stays `"1.0"`). Its §1 states the freeze condition (no partial freeze,
SQ-P0-2). **An SL-7-forced change is not folded into `-spec3`:** `-spec3` has been executed against,
so collision rule 5 makes the change a successor, `docs/plans/connected-code-graph-bundle-spec4.md`,
with `Supersedes:` `-spec3`. A freeze covers the **[P-later]** rules untested, so S15's gate record
names every [P-later] item of `-spec3` §10 **and §1.6 and §1.7** as **"frozen, untested"**, with
the first phase able to test it. The graph model is recorded the same way, in a note that
`Extends:` the graph note (S17).

**D6 — Offline sandbox: egress spec §7 controls.** The backends, the escape axes A1–A7, the
refusal conditions (C1, C2, N1, N2), the flag policy, the argv audit and the `/tmp` exec decision
are `docs/plans/connected-code-graph-egress-spec.md` §7 and §5.3. **§7.1.1 controls over v1.3's
D6 and its §4.1 test 1 `WSLInterop` path** (egress §11 R1). S08's test list is egress §10, with
`docs/plans/connected-code-graph-egress-spec2.md` (D18), which amends §7.1, §7.2, §7.3.1, §5.3 and
§10 by named clause. On this host only the container backend runs. What v1.3 D6 said about
AC-8 still holds: the audit format belongs to the egress contract, not to the sandbox code.

**D7 — One dedicated FalkorDB instance, reserved per step** (unchanged; delivered by S01):
`ccg-falkordb`, `falkordb/falkordb:v4.18.11`, `127.0.0.1:6390`, `maxmemory 1536mb`, `noeviction`,
`save ""`, `appendonly no`, admin password in `~/.config/ccg/falkordb-admin.pass`. Every step that
touches it holds a reservation (§5.1), serialized by `teco`. A second instance was rejected (6 GB).

**D8 — Ground truth is pinned to one evaluation commit** (unchanged). S40 pins `EVAL_COMMIT`
(`cpg/eval/EVAL_COMMIT`); every P0 ground-truth artifact refers to it; P1's gates build the
proving-repo snapshot from a worktree at `EVAL_COMMIT` with `cpg/eval/eval-manifest.yaml`'s
exclusions. Gold anchors and key locators use **bundle spec §3.2, §3.4, §3.5 and §6** (the
`repoId` `graphmind-ai-lab`, the source-derivable `qname` grammar of §4, the `ContractPoint`
tuple and `keyNorm`). Whether anonymous and duplicate scopes are eligible anchors is S06's (bundle
spec §11; pending, §14).

**D9 — Contract documents get durable topic slugs** (unchanged):
`connected-code-graph-{bundle-spec, egress-spec, rule-pack-spec, answer-spec}.md`, all `Owner:
architect`, `Extends:` the proposal. Method notes: `connected-code-graph-p0-ml.md` (P0(a)),
`connected-code-graph-eval-protocol-ml.md` (P0(f)), `connected-code-graph-p0-graph.md` (store
spikes, graph model v1). Results notes: `docs/test-reports/connected-code-graph-p0-{frontend-fitness,slice,ground-truth}-report.md`.

**D10 — Ids.** Bundle spec §3.1–§3.2 make v1.3's D10 exact (type-tagged tuples, BF1) and control;
S09 implements them (accepted). `-spec2` changes no golden vector (its §0 and ledger 2026-10-29,
recomputed by the analyst); its `import`/`launchFact` `ordinal` and `nameNorm` changes land in S10's
follow-up round.

**D11 — Held-out ground truth is plaintext outside the repo, hidden by convention only
(stakeholder decision, 2026-10-29).** Unchanged:

| Material | Location | Committed? |
|---|---|---|
| Held-out items | `~/.local/share/ccg-eval/<E>/heldout/*.jsonl`, plaintext | no |
| Dev items | `cpg/eval/dev/*.jsonl`, plaintext | yes |
| `cpg/eval/heldout-manifest.json`: per file the SHA-256 and item counts per stratum; no locators and no item content | the repo | yes |

- Any later change to a held-out file is **detectable**: GC re-hashes against the committed manifest
  at every gate, and a mismatch stops the gate.
- **What hides the keys is convention only.** No builder brief mentions the held-out path. Every P1
  builder runs under the same uid; reading by a builder is neither prevented nor detected. The plan
  adds no other mechanism.
- **Costs, accepted with the decision:** no git backup of the held-out set (the manifest proves what
  it was), and a builder's accidental exposure leaves no trace.
- Replaced options: gpg ciphertext with a passphrase file (v1–v1.2); time-boxed custody (review
  P2-5). The stakeholder judged both too much.

**D12 — Owners** (unchanged, plus D13's custodian). `architect` contracts; `tdd-engineer` contract
code; `graph-dba` Joern and the store; `coder` the slice's assembly and load; `devops` instance,
image, sandbox, launcher; `qa-engineer` fixtures, labels and the custodians; `data-scientist`
protocols, adjudication decisions and ground-truth statistics; `cobb` the headless agent
definitions; `security-expert` the one advisory (S58); `analyst` gates.

**D13 — Frontend-fitness truth: the ML note's pipeline, with a custodian for every data write.**
- **What is adopted.** The pipeline of ML note §11 (N0–N7) and its ordering: S05 → S24 → S25 →
  {S26 ∥ S56} → S57 → S27 → S28 (§3). Truth = blind hand label checked against a typed reference,
  adjudicated (ML §5.1); golden flow sets labelled blind (ML §6.4).
- **One departure, on ownership (ML §11 says owners are suggestions).** `data-scientist`'s write
  guard admits only `docs/plans/`, `docs/reviews/` and its scratchpad
  (`claude/data-scientist/data-scientist.md`, guard `guard-ds-doc-writes.sh`). N4, N5 and N7 have it
  writing under `cpg/spikes/frontend-fitness/`. That is v1.1's review M5 again, and it is fixed the
  same way as in §6: **`data-scientist` decides; someone else writes.**
  - **S57 (pre-Joern adjudication):** a fresh `qa-engineer` **fitness custodian (FC)** that has
    seen no Joern output runs `make_adjudication_input.py`, writes `adjudication/<lang>.input.jsonl`,
    and after the decisions arrive writes `adjudication/<lang>.jsonl` and
    `truth/<lang>.final.jsonl`. `data-scientist` reads the input file in place and returns its
    decisions keyed by site `ordinal`. Neither opens `results/` or any Joern artifact before S57
    closes; `analyst` checks both transcripts. This keeps N4's checkable ordering, which was the
    reason `graph-dba` (who has seen Joern results) does not run the script.
  - **S27 and S28 (miss and flow adjudication):** Joern's answer is necessarily in view there, so
    `graph-dba` writes the adjudication logs and the overturn files from `data-scientist`'s returned
    decisions.
- **The fallback analysers are not built in P0.** The four programs of ML §5.2 (a Pyright
  go-to-definition driver, a TS language-service driver, a Roslyn driver, a clangd-index reader)
  are new code, not tools. Writing four analysers to rescue a reference tool that might fail is
  out of proportion to a measurement spike, and the honest fallback exists: *single-source*, recall
  marked noise-unestimated (ML §5.2). If S24's smoke test fails a primary tool, the language goes
  single-source; whether one analyser is worth building is then a decision for that language, raised
  by `teco` with the evidence.
- **Rejected: running the reference truth in S24.** That is the S24↔S25 circle (review §S05 M3):
  corpora are chosen in S25. S24 proves tools on smoke projects; C5 on real candidates runs inside
  S25, before any Joern run on the candidate.

**D14 — Decision-rule authority for P0(a) (review §S05 M2).** v1.3 §4.3 fixed the precision
adoption rule as a point estimate, and S05 was delegated the sample, truth and `flowAvailable`
criterion but not that rule. This plan **ratifies ML note §8 as the single statement of P0(a)'s
decision rules**, superseding v1.3 §4.3's table:
- **Ratified, within or consistent with the plan's rule:** §8.1's n ≥ 100 floor and the
  counterfactual remedy (indexer adopted only if `Res ∪ Nm ∪ Ref` restores ≥ 0.98, otherwise
  recognizer work by cause class); §8.2 unchanged; §8.3's point-estimate rule (`P_c` < 0.95 →
  adopted; fold iff ≥ 0.95 in both configurations) **with** its n_P ≥ 60 floor; the
  worse-of-two-configurations rule (§8 preamble); §8.4's `flowAvailable` (pooled pair recall
  ≥ 0.95 over ≥ 120 golden pairs, projection success ≥ 0.95 as a precondition). §8.4 replaces v1.3's
  suggested "≥ 20 methods, recall ≥ 0.9", which was explicitly S05's to set.
- **Rejected: §8.3's "proposal for `architect`"** (adopt on precision only when the Wilson upper
  bound is below 0.95). A language at a point estimate of 0.90–0.94 would enter P1 with no indexer,
  carrying up to 9 false *confirmed* edges per 100 resolved sites into AC-5's gated
  confirmed-precision bar (ML §4.2 (i)). An unneeded indexer is a bounded cost; a failed P1 gate is
  not. This agrees with the note's own recommendation.

**D15 — A parse is certified by content, never by exit status (egress §11 R11).** `joern-parse`
can exit 0 with an empty CPG (egress §9 H12). Every step that consumes a CPG checks its content:
S11 (the bundle pipeline), S25 and S28 (fitness runs), S39 (already in v1.3). The check is **one
durable script, `cpg/sandbox/scripts/cpg-content-check.sc`, written by S08's follow-up round** by
promoting test 3's inline script (P3-7), run in the runner's `joern --script` form (egress §7.3.1)
and printing the non-external `METHOD` count per CPG and per source file. It is durable code, so S39
may use it (D1). The unit-level rule is the floor; S11 measures whether the per-file rule holds per
frontend and the check adopts it where it does. **Where it lives:** in the extraction pipeline, not in the bundle schema. A bundle with
`file` records and no `symbol` records can be legitimate (a unit of empty `__init__.py` files), so
a schema rule would reject valid bundles; the defect is "the parser produced nothing from a unit
that has code", which only the pipeline can see. Bundle spec v1.3 states no minimum-content rule,
so S10 enforces none (as its brief says) and S11 owns the guard.

**D16 — P0 loads: batch ceiling and explicit timeout.** Until S23 has probed the `PING` stall (§1.3)
and recorded a ceiling, every P0 write to `ccg-falkordb` (S13, S17, S20, S21, S22) uses batches of
**≤ 20k rows**. After S23, its recorded ceiling governs P0 and is a P1 loader input. Whenever
`TIMEOUT_MAX` is nonzero on the instance, every load query passes an explicit `timeout` (S07's
quirks entry); S22 measures the interaction and S17 encodes it.

**D17 — The `bindValue` advisory is a P0 step (S58), now delivered.** Bundle spec §5.17's
`bindValue` rule is [P-later] and freezes untested with v1 (D5); the spec recommended a
`security-expert` advisory before P1 relies on it (BF8; review §S03 F6). S58 is accepted
(`docs/reviews/connected-code-graph-bindvalue.md`, Passes 1–3) and landed **before S15**. Its
dispositions are recorded **in `-spec2`** (§2 and §5 there), not in S16's successor or a §8.6 note,
so the freeze already carries them. What remains is `-spec2` §3's follow-ups with their owners (§15).

**D18 — Egress errata live in `-spec2` of the egress spec.** `docs/plans/connected-code-graph-egress-spec2.md`
("ES2") **replaces v1.0's errata E1–E2** (ES2 §0): E1's whitespace set is superseded by ES2 §1.1
(strip every leading character ≤ U+0020 and U+007F; split lines at `\n`, `\r`, `\r\n`), and E2 is
restated as ES2 §1.2. Wherever v1.0 said "with errata E1–E2 (§4.1)", read "with ES2". The egress
spec's next revision for P1's broker is a `Supersedes:` successor of both (ES2 §0).

**D19 — Step 8a lives in one durable module, `cpg/ccg/keynorm.py`, written first (EP2 m4, Q1).**
- **Decision.** `mask_http_segments(path, placeholder)` in a new `cpg/ccg/keynorm.py`, importing
  `screen_fails` from `ccg.bundle`. S12 writes it as its own first unit, **P0-11a**, which depends
  only on S10's follow-up round; S12's `assemble.py` (P0-11b) and S49's `match_keys.py` both import
  it. **S49 depends on P0-11a, not on the whole of S12.**
- **Why a new module, not `bundle.py`.** `bundle.py` is the validator, S10's file, and S10's
  follow-up round is editing it now; a second writer would be a shared-file collision (§5.2). The
  8a loop is a `keyNorm` rule, not validation, and only needs `screen_fails`.
- **Consumers.** S12's `assemble.py`, S49's `match_keys.py`, and **the P1 assembler**, which imports
  the same function and never reimplements it (EP2 §3, P1 assembler row; §15).
- **What p0-2 does not fix.** The order of the `keyNorm` steps before 8a (e.g. dot resolution
  against percent-decoding) is EP2's and BS2 §1.8's; every row here cites them rather than restating
  an order.
- **Why P0-11a is split off.** S12 proper waits on S11, deep in the slice chain. Making the ground-
  truth chain (S49 → S45 → S47) wait on the slice would add a dependency on timing, which is what
  EP2 §1.1 put the function in `cpg/ccg/` to avoid. P0-11a is about one file and its tests.
- **Rejected:** EP2 §1.1's fallback (each consumer owns its loop, with a conformance run). It keeps
  two implementations that can drift; the shared import removes that in kind (EP2 §4 R1).

**D20 — The headless CLI pin stays 2.1.312, taken now (review Pass 3, P3-5).**
- **Fact.** `~/.local/share/claude/versions/` now holds 2.1.309, 2.1.310, 2.1.312 and **2.1.314**,
  and `~/.local/bin/claude` points at **2.1.314** (review Pass 3, P3-5; re-listed for v1.2). §6.3's
  flag facts were read from 2.1.312's `--help`.
- **Decision.** Pin **2.1.312**: S53's `prepare-bin` is dispatched **first and alone**, now, while
  2.1.312 is on disk, so the updater's pruning cannot take it. That keeps §6.3's verified flags
  valid with no re-read. If 2.1.312 is already gone when `prepare-bin` runs, it takes the current
  build instead, and §6.3's flag facts are re-read against **that** build's `--help` before S55
  (K-P0-12's deliberate step). Moving to 2.1.314 now was rejected: it would re-open §6.3's argv
  facts for no gain, and the census only needs one stable build.
- **K-P0-12's v1.3 rule is restored** (§13): moving to another build is a deliberate step — a new
  `prepare-bin`, then S55 and S54 re-run.

## 3. Step table

**Kind codes:** **C** contract document; **CC** contract code (schema or validator, durable);
**M** measurement spike (throwaway code + durable results note); **D** durable code; **GT**
ground-truth data; **Doc** documentation only.

**Reservations:** **R** = holds a reservation on the dedicated instance; **X** = exclusive (§5.1).

**Dispatch:** steps that share a unit id are one dispatch, run in order. Each unit is at most 3
steps and about 5 files; fixture source trees (S36–S38) are the stated exception, and S25 may be
dispatched as two units (P0-18a pre-Joern, P0-18b Joern runs, §4.6) because of its size.

**Ids:** `S01`–`S58` plus `S43a` are steps; **S18, S19 and S29 were never assigned**. S56–S58 are
new in this document (§0.1). A stakeholder decision is always written *decision S<n>*, never a bare
`S<n>`. Ledger units cite the `P0-nn` unit id; new units are P0-47…P0-49.

**Gates:** `an` `analyst` review; `ds` `data-scientist` method check; `gd` `graph-dba` check; `tc`
`teco` documentation check; `sh` the stakeholder. **State** column: `acc` accepted, `fl` in flight,
blank = not started. For `acc`/`fl` rows the contract is the cited document; this table does not
re-decide it.

| Id | Unit | State | Owner | Kind | Deliverable (files) | Inputs | Done when | Gate | Deps |
|---|---|---|---|---|---|---|---|---|---|
| S01 | P0-01 | acc | `devops` | D | `cpg/infra/ccg-falkordb.sh`, `cpg/infra/README.md`, `cpg/infra/tests/` | §2 D7 | As accepted (ledger U20; review §S01). | an | — |
| S02 | P0-02 | acc | `devops` | D | `cpg/infra/toolchain/{Dockerfile, build.sh, README.md}` | §7 | As accepted: `ccg-toolchain:d3184effb566` (ledger U21; review §S02 Passes 1–2). Its p2-m1 is S24's. | an | — |
| S03 | P0-03a | acc | `architect` | C | `docs/plans/connected-code-graph-bundle-spec.md` v1.3 **plus** `-spec2` v1.3 (accepted, ledger 2026-10-30) | — | As accepted (U22, U32). The contract is the two read together until `-spec3` (§2 D5). Downstream items: §4.2 and §16. | an + gd | — |
| S04 | P0-03b | acc | `architect` | C | `docs/plans/connected-code-graph-egress-spec.md` v1.3 | — | As accepted (U23), with `docs/plans/connected-code-graph-egress-spec2.md` v1.2 (accepted, U32), which replaces v1.0's errata E1–E2 (§2 D18). | an | — |
| S05 | P0-04 | acc | `data-scientist` | C | `docs/plans/connected-code-graph-p0-ml.md` v1.3 | — | As accepted (U24). Its step outputs are S24–S28, S56, S57 here (§16). | an | — |
| S06 | P0-05 | acc | `data-scientist` | C | `docs/plans/connected-code-graph-eval-protocol-ml.md` v1.2 **plus** `docs/plans/connected-code-graph-eval-protocol2-ml.md` (EP2, `Extends:`, step 8a; in re-gate) | S03 | As accepted (U26, no Pass 3): P0(f) complete, incl. the development-contact rule and GC's procedures. **The contract is EP v1.2 plus EP2 once EP2 is accepted**; EP2 controls in the clauses its §1 names. Hand-offs absorbed (§14, §16.1). | an | S03 |
| S07 | P0-06 | acc | `graph-dba` | Doc | `claude/graph-dba/falkordb-quirks.md` | — | As accepted (U25). | tc | — |
| S08 | P0-07 | acc + follow-up | `devops` | D | `cpg/sandbox/{offline-run.sh, forbidden-flags.txt, allowed-flags.txt, tests/test_sandbox.sh, README.md}`; `.audit/` in `cpg/.gitignore` | S04 + ES2; image from S02 | Accepted (U28, Pass 2). **Egress spec §10 tests 1–10 pass as amended by `docs/plans/connected-code-graph-egress-spec2.md`**; §7.1.1 controls N1. (Supersedes v1.3's "the seven tests of §4.1".) **Follow-up round (ES2 §3, then the S08 Pass 2 carry-overs):** ES2 §1.1's replaced test 10(b) (leading-byte coverage probe over 0x00–0x20 and 0x7f; the `x\r//>` fixture; existing shapes kept); §1.3's one-`--output`-per-live-invocation refusal with the lock outside the overlay `upperdir`, and its two-concurrent-invocations test (exactly one refusal); §1.4's namespace `/usr` cell; §1.5's `usage` path, `netnsDistinct`/`egressMode` on refusals and the `toolArgv` tests; **§1.7's closed runner environment** (`env -i` re-exec with a fixed `PATH`, a set `DOCKER_HOST`, a runner-owned `DOCKER_CONFIG` directory verified empty/owner/`0700` or exit 69, passwd-derived `HOME`, the two `CCG_*` names; the `-p` shebang kept) and its **polluted-environment test** run through `#!` (fake `docker` on `PATH`, `DOCKER_*` redirected, `SHELLOPTS`, an exported function, `BASH_ENV`/`ENV` marker, a foreign `config.json`, a pre-populated config directory → 69; a no-`-p` mutant fails); §1.8's A6(b′) in test 1 and the launch-equivalence check; the README's backend table and concurrency claim; and Pass 2's p2-m1 (scoped hygiene catches unlabelled and killed-runner containers), p2-n1 (lock survives a killed `docker run` client), p2-n2 (no leaked `/tmp/ccg-sbx.*`). **Also (P3-7, §2 D15):** promote test 3's inline Scala script (the one printing `METHODS <cpg>=…` from `cpg.method.isExternal(false)`) to a durable, runner-staged **`cpg/sandbox/scripts/cpg-content-check.sc`** that prints the non-external `METHOD` count per CPG and per source file; test 3 calls it; it is the one content check S11, S25, S28 and S39 use. | an | S02, S04 |
| S09 | P0-08 | acc | `tdd-engineer` | CC | `cpg/pyproject.toml` (extras frozen), `cpg/README.md`, `cpg/.gitignore`, `cpg/ccg/{__init__,ids}.py`, `cpg/tests/test_ids.py` | S03 §3 | As accepted (commit `6f4e34f9`): the 22 golden vectors reproduce; `dependency_record_id` included; `spike` extra carries ML §11 N6. | an | S03 |
| S10 | P0-09 | acc + follow-up | `tdd-engineer` | CC | `cpg/ccg/schemas/bundle-v1.schema.json`, `cpg/ccg/bundle.py`, `cpg/tests/test_bundle.py`; `cpg/README.md` (one touch) | S03; `-spec2` §1, §4.1 | Accepted as briefed (U30): bundle spec §2, §5, §8 and its §11 row; no minimum-content rule (§2 D15). **Follow-up round, before S15 (`-spec2` §4.1 items 1–8):** `ordinal` on `import`/`launchFact` with ids recomputed; `launchFact`'s argv gate and `defaultHash`; `-spec2` §1.7's `bindValue` grammar, token screen (with T6) and name rule, replacing today's classes incl. the bare-IPv6 branch; ASCII `IDENT` and the `kw` head; the `nameNorm` recompute; `controlledBy` rejects `point == cond`; §1.12's README wording; spec-derived tests also read `-spec2`'s replacement sections. **Done when `-spec2` §1.6.3's normative argv coverage probe passes as written** (17,280 grid cells and 6,480 separator-axis cells, zero values in clear; the 675-cell transplant), and AV-1–AV-68 and BV-1–BV-28 **parsed from `-spec2`'s text** pass, plus the SA-6 name probe, and **`-spec2` §1.7.4's W1** ([pre-freeze], validator-checked) is encoded explicitly, with its own test beside BV-10 (n-c). | an | S09 |
| S11 | P0-10 | | `graph-dba` | M | `cpg/spikes/vertical-slice/{extract.sc, normalize.py, run_extract.sh}` | S03, S10; `build-cpg.sh`; S08's `cpg/sandbox/scripts/cpg-content-check.sc` | Bundles for `falkor-chat/server` and `salesperson/` at a pinned commit, built in the sandbox, validate against `bundle-v1`; every slice symbol's `qname` re-derived from source (bundle spec §4.0; a mismatch is SL-7). **Each unit's CPG passes the content check of §2 D15** (zero non-external `METHOD`s from a unit with ≥ 1 non-empty source file fails the build, with a reason), **run through `cpg-content-check.sc`** (S08's, the single shared check S25, S28 and S39 also use; P3-7). S11 also measures, per frontend on the slice units, whether the stricter **per-file** rule (every non-empty source file has ≥ 1 non-external `METHOD`) holds without false failures; the slice report records the result, and the check adopts per-file only where it holds. `lastChanged*` computed outside the sandbox (bundle spec BF12). **`-spec2` §4.1's emitter items:** `nameNorm` by §1.2; `import.ordinal` with repeated names de-duplicated (§1.5); cond→cond `controlledBy` from the CDG (§1.4); ASCII-`IDENT` quoting of field keys (§1.1); **G-4: CDG self-edges dropped and (`point`, `cond`) pairs de-duplicated before writing, verified against the slice CPG**. Tool output names use ES2 §1.5's `@AUDIT_ID@` token (normative). Stage timings recorded. Two runs byte-identical apart from §8.5's exempt fields. | (in S15) | S10's follow-up round; **S08's follow-up round** (the runner-consumer rule, §5.2) |
| S12 | P0-11a, P0-11b | | `coder` | D (P0-11a); M (P0-11b) | **P0-11a:** `cpg/ccg/keynorm.py`, `cpg/tests/test_keynorm.py`. **P0-11b:** `cpg/spikes/vertical-slice/{assemble.py, tests/test_assemble.py}` | `-spec2` §1.7.1 (`screen_fails`), §1.8; `docs/plans/connected-code-graph-eval-protocol2-ml.md` ("EP2") §1.1, §2; P0-11b also S11 bundles, proposal §3.3–§3.4 | **P0-11a (durable, dispatchable as soon as S10's follow-up round is accepted):** `cpg/ccg/keynorm.py` holds the **one shared step-8a function**, `mask_http_segments(path, placeholder)` — a pure function over a path already normalized by the steps EP2's `http` rule and BS2 §1.8 place before 8a (their order is theirs, not restated here) that replaces every literal segment `s` with `screen_fails(s)` (imported from `ccg.bundle`, never copied) by the given placeholder (`{#}` or `{?}`), leaving the verb, empty segments and the four placeholders untouched (EP2 §1.1). `test_keynorm.py` passes **EP2 §2's K1–K22, K30–K33 and K42–K48 in provider form**. Neither `assemble.py` nor `match_keys.py` reimplements it (§2 D19). **P0-11b:** **the `http` `keyNorm` step 8a** in `assemble.py` **imports `mask_http_segments`**, with T6 and the provider `{#}` placeholder (matches only `{?}`, always `possible`). SL-1 and SL-2 pass on the real bundles, unchanged by step 8a. Unit tests on hand-made mini-bundles pass. | an (P0-11a, code review + mutation spot-check); (in S15) for P0-11b | P0-11a: S10's follow-up round. P0-11b: S11, P0-11a |
| S13 | P0-12 | | `coder` | M, **R** | `cpg/spikes/vertical-slice/{load.py, impact.py, tests/test_live_slice.py}` | S12 output; graph note §3.4, §3.5, §6.2–§6.4 | SL-3–SL-6 pass against `ccg_p0slice` on the dedicated instance; loads obey §2 D16. | (in S15) | S12, S01 |
| S14 | P0-12 | | `coder` | Doc | `docs/test-reports/connected-code-graph-p0-slice-report.md` | S11–S13 | Every SL criterion pass/fail with evidence; densities, timings, the determinism diff; every SL-7 delta. The `Link` count by status, BFS peak frontier and field-path state count (M4); the hop count of every `Link` chain. **The proving-repo anchors the slice touched, as locators in bundle spec §3.5 form, by eval-protocol §8.2's definition of "touched" — wrapper and chain hops excluded.** The bundle spec §10 [P-slice] items each marked exercised or not. | — | S13 |
| S15 | gate | | `analyst` + `graph-dba` | gate | a `## Pass N` in `docs/reviews/connected-code-graph-p0.md`; `teco` records freeze or provisional in the ledger | S14, S16 | Verdict given. Freeze iff SL-1–SL-7 all pass (§2 D5). **The record names `-spec3` as the contract, checks that S10 encodes every [pre-freeze] item of `-spec3` §1, and lists every [P-later] item of `-spec3` §10 and its §1.6/§1.7 as "frozen, untested", with its first testable phase** (or "provisional, untested" if v1 does not freeze). | — | S14, S16, S10's follow-up |
| S16 | P0-13 | | `architect`; `tdd-engineer` for the test re-point | C, CC | **`docs/plans/connected-code-graph-bundle-spec3.md`**, `Supersedes:` the bundle spec **and** `-spec2`; `cpg/tests/test_bundle.py` (re-point); `cpg/tests/test_ids.py` if its vectors differ | S14; `-spec2`; S10's follow-up notes | **Not optional; always runs, before S15** (`-spec2` §0, F-5): `-spec3` is the consolidation of the two documents, gated, and S15 is not dispatched until it is accepted. A change forced by S15 (an SL-7 delta) is a `-spec4` that `Supersedes:` `-spec3`, written after S15. `-spec3` includes F-13 (RFC 3986 order for `keyNorm` steps 5–6) and its consumer list (§13 there). **Code half (P3-2), `tdd-engineer`:** re-point `SPEC_TEXT` and `SPEC2_TEXT` in `cpg/tests/test_bundle.py` at `-spec3` (one source; the `-spec2` constant retired); check that `test_ids.py`'s hard-coded golden vectors (`SPEC_CONSTRUCTORS`, `SPEC_SHAPES`) equal `-spec3` §3.6's; renumber `cpg/tests/test_keynorm.py`'s step labels (decode = step 5); the default suite passes. A transcription slip in the 22 golden vectors, AV-1–AV-68 or BV-1–BV-28 is then a red test. **Parsers to change (gate m5):** `_probe_counts` read the removed "Observed on" table, so the three probe tests take their counts from `-spec3` §1.6.3's pass-condition formulas (17,280; 6,480; 675; 85,536), parsed and never retyped; `_spec_secret_tokens` reads the 15 substring words from §1.7.3, not the old §5.17 row; `ALLOW_LIST_ENTRIES` may be asserted against §1.6.3's list. | an | S15 |
| S17 | P0-14 | | `graph-dba` | CC, **R** | `docs/plans/connected-code-graph-p0-graph.md` §1 "Graph model v1"; `cpg/ccg/store/ddl_v1.cypher`; `cpg/tests/test_ddl_live.py` | graph note §3.2, §3.4, §6; S14; bundle spec §9, §12 | Model v1 with deltas is written, and it **states a disposition for bundle spec §9 and each of BF2–BF15** (graph-side rulings in review §S03 graph-dba check; BF16 is a proposal edit, S50). It **includes the `ownedRegions` anchor rule** (narrowest containing interval, span or owned region; a span match beats an owned-region match, then the deeper `parentId` chain; bundle spec §9.1, §13.3 P3-1). It **states the load rule under `TIMEOUT_MAX`** (an explicit `timeout` on every load query when the cap is nonzero; §2 D16) and the batch ceiling from S23. **`-spec2` F-11 and §1.11:** `Cond` as a `CONTROLLED_BY` source (one Cond→Cond edge group); `defaultHash` never carried; F-2 recorded as closed (no cross-entry `valueHash` equality; any future cross-entry hash is bundle-only); **G-1 resolved by carrying only the role-relevant gated `launchFact` fields — `port`, `mountPath`, `service`, argv[0] and the role-relevant pairs — and never the whole of `args`** (security P3-1), or else rendering file and line only, with S33 told which; P2-8's screened `keyPath` property; S-1's `pointId` indexes on `Param`/`Ret`/`Cond` in `ddl_v1.cypher` (S-2 routing). The DDL applies to an empty key, `db.indexes()` matches it, the post-load constraints reach `OPERATIONAL`. | an | S15, S23 |
| S20 | P0-15 | | `graph-dba` | M, **RX** | `cpg/spikes/store/{synth.py, bfs.py, bench_reach.py}`; p0-graph note §2 | graph note Q7, V8, §6.2, §6.4 | BFS vs `algo.SSpaths` benchmarked on a 1,000-file synthetic v2 graph, loaded per §2 D16; the 20-repeat FR-19 diff for both; the decision by §4.5's rule. The BFS keeps a visited set seeded with the anchor, so cond→cond `CONTROLLED_BY` cycles terminate (`-spec2` F-12, G-3), and the synthetic graph includes such a cycle. | an | S01, S09 |
| S21 | P0-15 | | `graph-dba` | M, **RX** | `cpg/spikes/store/swap_probe.py`; p0-graph note §3–§4 | graph note Q4, V2, V10, G13 | `MULTI`/`RENAME` rollback and the telemetry cap/disable probed and recorded with commands and outputs, **including when `telemetry{<key>}` disappears after `GRAPH.DELETE`** (one sighting of asynchronous removal, review §S03 graph-dba check). | an | S20, S09 |
| S22 | P0-16 | | `graph-dba` | M, **RX** | `cpg/spikes/store/{acl_probe.py, lock_replay.py, tests/test_lock_replay.py}`; p0-graph note §5 | proposal §3.5, §3.8; graph note §6.4, V5, V16–V18; quirks TIMEOUT entry | The ACL matrix and lock grants verified; lock/replay fault tests pass, incl. mid-swap expiry; `TIMEOUT_MAX`'s effect on load writes (with and without an explicit `timeout`), `GRAPH.CONSTRAINT` and full-text backfill probed; `TIMEOUT_MAX` and the lock ttl sized by §4.5's rule and labelled **P0-scale defaults that P1 re-derives**. | an | S21, S09 |
| S23 | P0-16 | | `graph-dba` | Doc, M, **RX** | p0-graph note §6 decisions table; `claude/graph-dba/falkordb-quirks.md` (new quirks); `cpg/spikes/store/stall_probe.py` | S20–S22; review §S03 graph-dba "Incident" | Every C-spike decision has a row with its measurement. **The `PING` stall is probed** (single writes of 20k, 50k and 100k edges, `PING` latency sampled from a second connection, CPU and `INFO` recorded) and a **batch ceiling** recorded for §2 D16; a reproducible stall becomes a dated quirks entry with its command. New quirks folded in. | an | S22, S07 |
| S24 | P0-17 | | `devops` | D | `cpg/infra/toolchain/{Dockerfile, build.sh, README.md}`; `smoke/<lang>/` projects | S05 §5.2, §9; S02 | **§4.6 S24** (ML §11 N0, plus the S02 carry-forwards and the image-tag consequence). Items 1–4 and 6 touch only `cpg/infra/toolchain/` and may start now; **item 5 is held until S08's follow-up round is accepted** (P3-4). | an | S02, S05; **S08's follow-up round (item 5 only)** |
| S25 | P0-18 (a, b) | | `graph-dba` | M | `cpg/spikes/frontend-fitness/{corpora.yaml, run_fitness.py, measure.sc, make_adjudication_input.py, matcher-table.md, README.md, frame/, truth/, sample/}` | S05; S03 §4.1/§4.2/§5.1; S08; S24 | **§4.6 S25** (ML §11 N1, N2). | ds + an | S24, S08, S09, **S08's follow-up round** (P3-10) |
| S26 | P0-19a…g | | `qa-engineer`, one fresh instance per language (seven, parallel) | GT | `labels/<lang>.jsonl`, one per dispatch | `sample/<lang>.jsonl` and `sample/<lang>.brief.md` (pasted as-is); that corpus's sources only | Every sampled site labelled by ML §6.5's brief, blind to Joern and the reference (ML §6.2). Appended as it goes; a killed labeller is resumed in place. `teco`'s transcript check clean (ML §6.2). | ds | S25 |
| S56 | P0-47a…g | | `qa-engineer`, one fresh instance per language (seven, parallel; never an S26 labeller) | GT | `labels/<lang>-flow.jsonl` | `sample/<lang>-flow.jsonl`; that corpus only; bundle spec point names (§5.4–§5.6) | **§4.6 S56** (ML §11 N3). Its methods are development contact for flow anchors (eval-protocol §8.2), so its output is an input to S47's contact set. | ds | S25, S03 |
| S57 | P0-48a…g | | `data-scientist` decides; fresh `qa-engineer` fitness custodian (FC) runs and writes, one per language | GT | `adjudication/<lang>.input.jsonl`, `adjudication/<lang>.jsonl`, `truth/<lang>.final.jsonl` | `labels/<lang>.jsonl`, `truth/<lang>.ref.jsonl`; never `results/` or any Joern output | **§4.6 S57** (ML §11 N4; owner per §2 D13). | an | S26 (that language), S25 |
| S27 | P0-20 | | `graph-dba` runs; `data-scientist` adjudicates misses | M | `results/*.json`; `adjudication/<lang>-misses.jsonl`; `truth/<lang>.final2.jsonl`; fitness report §a1 | S25 results; S57 truth | **§4.6 S27** (ML §11 N5). | an + ds | S57 (all languages) |
| S28 | P0-21 | | `graph-dba` runs; `data-scientist` adjudicates | M | fitness report §a2; `results/projection-*.json`; `adjudication/<lang>-flow-{misses,spurious}.jsonl` | S11 `extract.sc`; S56 labels; ML §7.3, §8.4 | **§4.6 S28** (ML §11 N7). | an + ds | S56, S27, S15, **S08's follow-up round** (P3-10) |
| S30 | P0-22 | | `architect` | C | `docs/plans/connected-code-graph-rule-pack-spec.md` | S03 (+S16), S14 bundles, S27 decorator/annotation shapes; proposal §3.3–§3.4 | Format v1, five paper packs walked through, infrastructure-signature pack v0, the severity rule table (§4.2). **Every proving-repo anchor the walkthroughs touch is listed as a locator, by eval-protocol §8.2, hops excluded.** **The server-binding walkthrough runs on `falkor-chat/server`, or on S35's fixture — never on `cypher-mcp` or `mcp-monitor`** (eval-protocol §14, §15 Q10, R9): walking `.mcp.json` → `cypher` would cost `wiring` 3 of its at most 10 eligible items and cap it at 7, short of 8. | an (+ ds consult) | S15, S27, S16 if it runs |
| S31 | P0-23 | | `tdd-engineer` | CC | `cpg/ccg/schemas/rule-pack-v1.schema.json`, `cpg/tests/test_rule_pack_schema.py`, `cpg/packs/{fastapi,fetch,fastmcp}/pack.yaml`, `cpg/ccg/registry/reasons.yaml` (registry cells incl. `PROVIDES topic` on runtime-built/config-derived, O-16) | S30 | Schema = S30. The three packs validate. Each negative example in S30 is rejected. | an | S30 |
| S32 | P0-24 | | `tdd-engineer` | CC | `cpg/packs/{kafka,paper-inhouse-bus,infra-signatures,severity-rules}/pack.yaml`; extend `test_rule_pack_schema.py` | S30 | All seven packs validate. | an | S31 |
| S33 | P0-25 | | `architect` | C | `docs/plans/connected-code-graph-answer-spec.md` | S03 (+S16), S17, S20 decision, eval-protocol §5 (hit rule, E9), §10.4; proposal §3.6 | Answer schema v1 specified (§4.2), and it satisfies eval-protocol §5.2/§14 (groups enumerate members with their own chains; each item carries a span-bearing symbol id and its chain; `stamp.verdictLogOffset`); **the interpretation block carries each resolved symbol anchor's `qname`**, so §10.4 can apply `E` (eval-protocol §15 Q1). **From `-spec2` §3–§4.1:** G-1's rendering consequence (a `launchFact` hop shows file and line only, unless S17 carries the gated fields); **F-3** free text from source (`keyPath`, `Cond.code`, `calleeName`, `qname`, emitted `launchFact` args) is fenced as data wherever an LLM reads it; **F-4** free-text identifiers shown in an answer (`ConfigEntry.keyPath`, a `config_key`/`http` key, `launchFact.mountPath`/`service`/`envVar`) are rendered through `-spec2` §1.7.1's token screen; **F-12** the recipe's traversal over `CONTROLLED_BY` is cycle-safe (a visited set seeded with the anchor). | an (+ ds consult) | S17, S20, S06, S16 if it runs |
| S34 | P0-26 | | `tdd-engineer` | CC | `cpg/ccg/schemas/answer-v1.schema.json`, `cpg/ccg/answer.py`, `cpg/tests/test_answer.py`, `cpg/tests/golden/answer-ac1.json` | S33; S13 output | Canonical bytes deterministic. The golden AC-1 answer, hand-built from the slice's BFS result, validates and round-trips. | an | S33, S13, S16 if it runs |
| S35 | P0-27 | | `qa-engineer` | C | `docs/test-plans/connected-code-graph-fixtures.md` | S03 reason codes; eval-protocol §12.2 minimums; proposal §5.1 | Every planted item listed with kind, expected status, reason, gateable phase and the assertion it serves (§4.7), at or above eval-protocol §12.2's minimums, **with an assertion column for AC-25's four presentation shape properties** (eval-protocol §12.2, m6), which S39 tests. | an + ds | S03, S06 |
| S36 | P0-28 | | `qa-engineer` | GT, D | `cpg/fixtures/system1/fx-svc/**`, `expected/fx-svc.yaml` | S35 | The repo matches S35's fx-svc rows. | (in S39) | S35 |
| S37 | P0-29 | | `qa-engineer` | GT, D | `cpg/fixtures/system1/fx-web/**`, `expected/fx-web.yaml` | S35 | The repo matches S35's fx-web rows. | (in S39) | S35 |
| S38 | P0-30 | | `qa-engineer` | GT, D | `cpg/fixtures/system1/fx-lib/**`, `expected/fx-lib.yaml` | S35 | The repo matches S35's fx-lib rows. | (in S39) | S35 |
| S39 | P0-31 | | `qa-engineer` | D | `cpg/fixtures/system1/expected/cross.yaml`, `cpg/fixtures/materialize.py`, `cpg/ccg/schemas/fixture-manifest-v1.schema.json`, `cpg/tests/test_fixtures.py` | S36–S38, S08, S10 | Materializer deterministic (same commit SHAs every run). Every locator resolves. Manifests validate. The `live` test parses each repo in the sandbox and passes §2 D15's content check. S35's AC-25 assertion column is tested. | an + ds | S36, S37, S38, S08, S10, **S08's follow-up round** (P3-10) |
| S40 | P0-32 | | GC (`qa-engineer`) runs; `data-scientist` checks | GT | `cpg/eval/{EVAL_COMMIT, eval-manifest.yaml, kit-manifest.json}`; master kit and per-role `BRIEF.md`s outside the repo (§6.2) | eval-protocol §2.4, §3.3, §13.1, Appendix B; tooling from S48 | Commit pinned. Kit exported, content-checked (§6.2) and hashed. A brief per role and stratum — **five strata** (`http`, `mcp`, `scripts`, `wiring`, `flow`) — into which **eval-protocol Appendix B is pasted verbatim, and nothing else from that note** (M3; not §3/§4, which carry JSON examples). The flow-anchor list drawn by `export_kit.py --draw-flow-anchors` with a recorded seed (eval-protocol §3.6). `eval-manifest.yaml` carries eval-protocol §3.3's fields, incl. the prompt-bearing Markdown glob list, **plus `units` (each with `ports`, filled from source, §2.4), `cypherMcpTools` and the §2.4 grain table (incl. the Kiro `toolsSettings`/`includeMcpJson`/fallback rows)**, listed by `export_kit.py --list-units` and `--list-cypher-tools`; the method owner checks `ports`. **Per EP2 §1.5, `eval-manifest.yaml` records both protocol documents' paths and versions, and `screen: {specVersion, bundlePySha256}`** (diagnostic, not a stop condition). | an + ds | S03, S06, S48 |
| S41 | P0-33a…e | | `teco` runs the launcher; headless KA session, one per stratum (§6.1) | GT | `roles/KA-<stratum>/out/KA.jsonl`, appended as it goes | role root (§6.2); eval-protocol Appendix B (in the brief) | The provider-first key covers its stratum. The session's audit is clean (§6.5). Five sessions. | (in S45) | S54 |
| S42 | P0-34a…e | | `teco` runs the launcher; headless KB session, one per stratum | GT | `roles/KB-<stratum>/out/KB.jsonl`, appended as it goes | role root; eval-protocol Appendix B (in the brief) | The consumer-first key covers its stratum. The audit is clean. Five sessions. | (in S45) | S54 |
| S43 | P0-35 | | `teco` runs the launcher; headless QW session | GT | `roles/QW/out/QW.jsonl` | role root; eval-protocol §10.1; Appendix B's QW part (in the brief) | The bilingual impact questions written natively, with gold, in eval-protocol §10.1's counts (48 questions). The audit is clean. | (in S47) | S54 |
| S43a | P0-35a | | GC (`qa-engineer`) | GT | `QW-nogold.jsonl`, placed in the AN1 and AN2 role roots | S43; eval-protocol §13.2 | `strip_gold.py --verify`: every question present, no gold field. | ds | S43 |
| S44 | P0-36 | | `teco` runs the launcher; headless AN1 ∥ AN2 sessions | GT | `roles/AN{1,2}/out/AN{1,2}.jsonl` | S43a; eval-protocol Appendix B's AN part (in the brief) | Each annotator has independently labelled every question ambiguous or unambiguous. The audit is clean. | (in S47) | S43a |
| S45 | P0-37 | | GC runs; `data-scientist` adjudicates near-misses by pair id and checks | GT | `~/.local/share/ccg-eval/<E>/work/{audit,match,nearmiss,capture}.json` | S41, S42 outputs; `runs/`; transcripts; eval-protocol §6, §7.1–§7.3, §13.4–§13.5; tooling from S48–S49 | Every session in the run index audited and clean, or voided and redone. Locators validated (eval-protocol §7.1) and scope checked (`match_keys.py scope`, V4 per item after matching). Items matched, incl. N3 pairs (§6.3). **`normalize` applies EP2's `http` `keyNorm` rule as written there** (its step list and order are EP2's, not restated here; step 8a via `mask_http_segments`); each 8a rewrite is a V3 **replace-and-log** entry (item `seq`, rule `8a`, masked-segment count, no item content on stdout), and the method owner reviews that log. N̂ and completeness per stratum by eval-protocol §6.4–§6.5 (at most one extra provider-first pass per stratum). The union and spot-check sample drawn (§6.4). | ds | S41, S42, S49 (and through it EP2 and P0-11a) |
| S46 | gate | | stakeholder (via `teco`/`tico`) | gate | marks on the spot-check sample | S45 sample (eval-protocol §7.3: 40 union items + 8 PT gold) | Each sampled item marked correct or incorrect, with a note on each incorrect one (decision S36; ML §8.3). | sh | S45, S44 |
| S47 | P0-38 | | GC runs; `data-scientist` checks | GT | `cpg/eval/dev/*.jsonl`, `cpg/eval/heldout-manifest.json`, `docs/test-reports/connected-code-graph-p0-ground-truth-report.md`; `~/.local/share/ccg-eval/<E>/heldout/*.jsonl` (D11) | S45, S46, S44; eval-protocol §7.4, §8, §9, §10.2–§10.3, §13.6 | Corrections applied. **Every contact-set source goes through EP2 §1.2's `normalize_contract` before `anchorId`**; **AC-24 gold is stored normalized, each locator with its stored form — provider `{#}` or consumer `{?}`** (EP2 §1.2) — each rewrite logged and counted in the report, and the unserved-gold check run (EP2 §1.2). The development-contact rule (eval-protocol §8, E6) applied to the contact set from S14, S30, the P0 golden tests (incl. S34's `answer-ac1.json`) and S56 — **all of which are closed before S47 starts (Deps, P3-1)** — by **probe cluster** (§5.1: the probe is the served point with the fewest items), then the seeded split (§9, E7) with step 3b's **served-anchor rule** and step 4's eligibility guard. The **reserve** file (up to 35 counting items, never exposed) and an **empty `exposure.jsonl`** are written. Held-out written outside the repo; manifest committed (D11). The report carries counts, N̂, completeness, manifest hashes and held-out anchors per stratum, **the counting `H` beside `H_all`** (E11), and the `served-anchor` moves **as a count**. The manifest's `protocol` string names both documents per EP2 §1.5. The leak check (`--check-report`, EP2 §1.4's two probes: normalized and pre-8a forms) finds **no held-out content** (§6.4). | an + ds + sh (counts only) | S46, **S14, S30, S34, S56** (P3-1), S49 (EP2) |
| S48 | P0-41 | | `qa-engineer` (tooling author; non-builder, never sees keys) | D | `cpg/eval/tools/{export_kit.py, strip_gold.py, audit.py}`, `cpg/tests/test_eval_tools.py`, `cpg/tests/fixtures/headless/` (built from S55's real capture) | eval-protocol §13.1–§13.3; §6.5; S55 capture | Each tool passes its tests, incl. **`export_kit.py --draw-flow-anchors`** (eval-protocol §13.1: Python `ast` and py-tree-sitter **from the `spike` extra**, not S02's image) reproducing a seeded draw, and **`--list-units`** and **`--list-cypher-tools`** (§13.1, §2.4). `audit.py` passes on fixtures shaped like S55's real capture: flags an absolute repo read, a relative `../` read, a sibling-root read, an MCP call, a disallowed tool, a write outside `out/`, an init tool set unequal to `--tools`; does not flag a legitimate in-root read; flags a stream/transcript mismatch and a missing transcript; on a two-attempt session (start, kill with a truncated last line, resume) consumes both attempt logs with no false violation. | an | S06, S09, S55 |
| S49 | P0-42a (`match_keys.py`), P0-42b (`capture.py`, `split_manifest.py`) | | same author as S48 | D | `cpg/eval/tools/{match_keys.py, capture.py, split_manifest.py}`; extend `cpg/tests/test_eval_tools.py` | eval-protocol §13.4–§13.6, Appendix A; **EP2 §1, §2** | **Step 8a (F-8, EP2; P3-3):** `match_keys.py`'s `normalize` applies EP2 §1.1, **importing `mask_http_segments` from `cpg/ccg/keynorm.py`** (§2 D19), and exports **`normalize_contract`** (EP2 §1.2); it adds the **`renormalize-check`** verb (EP2 §1.5), **which takes each AC-24 gold locator's stored form (`{#}` or `{?}`) as an input** (EP2 §1.2, §1.5); `--check-report` probes the normalized and pre-8a forms (EP2 §1.4); **EP2 §2's oracle passes in full, K1–K48** (K40–K42 cover the stored form). **Fallback:** if EP2 is not yet accepted when S49 is dispatched, S49 may start but does not close. On synthetic keys the tools reproduce hand-computed matches, estimates (Chapman and Chao, eval-protocol §6.4, with Appendix A as `capture.py`'s oracle) and a seeded split. **`match_keys.py` has `validate` (eval-protocol §7.1, V1/V2; V2's `config` check uses `pyyaml` from the `dev` extra), `scope` (V4) and `sample` (§7.3) verbs**, each tested, and **exports `E`, `h1_symbol_credit` and `source_check_needed`** (§13.4) for the P1 gate harness to import. `split_manifest.py split` implements §9.1 step 3b (served-anchor rule) and the step 4 guard; `--verify` re-checks the manifest and fails on a one-byte change; `--check-report` uses §9.4's v1.2 pattern and catches a planted held-out locator. **Oracle cases pass:** B1a, B1b, M2, M6, reserve, m4, P2-1a–e, P2-2a–d, P2-3 (§13.4, §13.6). `regate-list` (re-applying step 3b and routing by S(u), §12.1) and `exposure` verbs are specified; they are needed from P1 and may be built in P1. Dispatched as two units (n-d): P0-42a, then P0-42b. | an + ds | S48, **S10's follow-up round, S12's P0-11a, EP2 accepted** (P3-3) |
| S52 | P0-43 | | `cobb` | D | `cpg/eval/headless/agents.json` (four definitions: `ccg-key-writer-provider`, `ccg-key-writer-consumer`, `ccg-question-writer`, `ccg-annotator`), `cpg/eval/headless/settings.json` (the `PreToolUse` path guard and nothing else), `cpg/eval/headless/guard-paths.sh` | §6.1–§6.3; eval-protocol Appendix B; bundle spec §4, §6 | Each definition has its role's method and output contract, quoting **eval-protocol Appendix B verbatim** — not "§4.1/§4.2/§4.3 verbatim", which would carry JSON examples (M3) — and **no** ancestor-plus-line rule (E1), and **no** learning-capture, KB, CPG or repo instructions. Kit text the writers read is fenced as data, never as instructions (`-spec2` F-3, `cobb`'s half). The two key-writer prompts differ in strategy and wording, not only in a flag. The guard denies any file-tool path outside the cwd root, and any write outside `out/`, and has its own tests. The definitions are **not** deployed to `~/.claude/agents/`; they load only through `--agents`. Whether `claude/AGENTS.md`'s catalog rules apply to these harness-only definitions is `cobb`'s call, recorded in the step. | an | S06 |
| S53 | P0-44 | | `devops` | D | `cpg/eval/headless/run-writer.sh` (verbs `prepare-bin`, `prepare`, `start`, `resume`, `status`), `cpg/eval/headless/README.md` | §6.2–§6.3; S52 (except `prepare-bin`) | (0) **`prepare-bin` is dispatched first and alone, now, ahead of S52 (§2 D20, P3-5).** (1) `prepare-bin` copies `~/.local/share/claude/versions/2.1.312` to `$HOME/.local/share/ccg-eval/bin/claude-2.1.312` and records its SHA-256 (if 2.1.312 is gone, it takes the current build, and §6.3's flags are re-read against that build's `--help` before S55, per K-P0-12), and **`env -i HOME="$HOME" PATH=/usr/bin:/bin LANG=C.UTF-8 timeout 10 "$HOME/.local/share/ccg-eval/bin/claude-2.1.312" --version` exits 0**, run by `devops` as part of done. (2) `start` and `resume` build argv from **one** `build_argv` function; dry-run outputs differ **only** in `--session-id <u>` vs `--resume <u>`, the prompt, and the stream-log name. (3) Each attempt writes a new `runs/<uuid>.<n>.stream.jsonl`, never an existing one. (4) The argv carries `--tools Read,Grep,Glob,Write,Edit` and `--disable-slash-commands`. (5) The run index records the CLI version, the binary's SHA-256 and the SHA-256 of `settings.json` and `agents.json` per attempt, and `resume` refuses if any changed. `prepare` is idempotent; the script passes `bash -n` and S53's own tests (§9.1), with `shellcheck` run too only if installed (it is not on this host, P3-11). | an | S52 (`prepare-bin`: none) |
| S54 | P0-45 | | `teco` runs the launcher; GC runs `audit.py`; `devops`/`cobb` fix on failure | gate | rehearsal result in the ground-truth report §Rehearsal (written by GC) | S52, S53, S48, S40 | §6.3's planted-breach rehearsal passes in full, incl. the init-event checks and the **kill-and-resume case**. **No real writer starts before it does.** | an (stakeholder informed) | S52, S53, S48, S40 |
| S55 | P0-46 | | `teco` runs the launcher (`run-writer.sh start capture`) | GT (fixture) | `cpg/tests/fixtures/headless/capture.{1.stream.jsonl,transcript.jsonl}`, scrubbed of the account id; a note in `cpg/eval/headless/README.md` | S53 | One trivial real session, role `capture`, in a **dummy** role root built from a one-file dummy kit (so no S40 kit, no cycle). A one-line brief ("read `kit-src/README.md`, write `out/done.txt`") through the real argv. Settles (a) the real stream-json and transcript shapes, (b) whether `--verbose` is required, (c) that OAuth works under `--restricted`. If (c) fails, the chain stops and goes back to the stakeholder. | an | S53 |
| S58 | P0-49 | acc | `security-expert` (advisory); `architect` dispositions | Doc | `docs/reviews/connected-code-graph-bindvalue.md` (Passes 1–3); dispositions recorded **in `-spec2`** (§2, §5 there) | bundle spec §5.17, BF8; review §S03 F6 | Accepted (U31; dispositions accepted with `-spec2` v1.3, ledger 2026-10-30), before S15. Remaining follow-ups are `-spec2` §3's, owned there (§15). Not a gate (§2 D17). | — | S03 |
| S50 | P0-39 | | `architect` | Doc | `docs/plans/connected-code-graph.md` v3.4 (in place, one dated line) | every results note; bundle spec BF16 | §3.2's table restated as measured; `flowAvailable` and the indexer decisions recorded; §3.5's sizes at P0 scale; §6.2's P0 items marked settled with a pointer each. **§3.8 gains the fourth orphan case, "re-binding on sibling insertion" (BF16)**, with its mitigation (both end fingerprints in the verdict log). **§4's P1 row records eval-protocol Q1** (shipped default depth bound ≥ 2) **and Q3** (prompt extractor classifies by `eval-manifest.yaml`'s glob list). The proposal's `Extended by:` gains this document. | an | all gates |
| S51 | P0-40 | | `teco` | Doc | `docs/HISTORY.md`, `docs/BACKLOG.md` (C-1001), root `AGENTS.md` `cpg/` bullet (one clause) | §10 | §10's list is applied. | tc | S50 |

**Parallel and sequential structure.** S01–S10 and S58 are accepted; S08's and S10's follow-up
rounds are in flight.
- **Now dispatchable:** **S53's `prepare-bin`, first and alone** (§2 D20); P0-15 (S20 → S21);
  **S24 items 1–4 and 6** (item 5 waits on S08's follow-up round, P3-4); S35; S52 (`cobb`) → S53
  (`devops`) → S55 (the capture); S48 after S55.
- **Waiting on the in-flight rounds:** **every runner consumer — S11, S25, S28, S39 and S24 item 5 —
  runs on the post-round runner, after S08's follow-up round is accepted** (P3-10; the reason is ES2
  §1.7's environment change, not `@AUDIT_ID@`, which the runner already carries). S11 and S12's
  P0-11a also wait on S10's round. S49 waits on S48, P0-11a and EP2's acceptance (P3-3).
- **Slice chain:** S11 → S12 (P0-11b) → P0-12 (S13 → S14) → S15.
- **Store chain:** P0-15 (S20 → S21) → P0-16 (S22 → S23), all exclusive.
- **Fitness chain:** S24 → S25 (P0-18a, then P0-18b) → {S26 (seven) ∥ S56 (seven)} → S57 (per
  language, as that language's S26 lands) → S27 → S28 (also after S15).
- **Fixtures:** S36 ∥ S37 ∥ S38 → S39.
- **Ground truth (§6):** S40 waits on S48; **S54** waits on S40, S52 and S53, and no writer starts
  before it passes; then S41 (five KA), S42 (five KB) and S43 (QW), **at most two at a time**
  (§6.6); S43 → S43a → S44; S45 waits on S41, S42 and S49; then S46 → S47. **S47 also waits on
  every contact source — S14, S30, S34 and S56 — so the contact set is complete before S47 closes
  it** (P3-1; eval-protocol §8.2, §8.4). This ties the ground-truth chain's end to the slice and
  contract chains.
- **Before S15:** **S16 always runs** (§3 S16) and must be accepted before S15. **After S15:** an
  SL-7-forced change makes a `-spec4`, which S30, S33 and S34 wait on; otherwise they do not. S17 also waits on S23;
  S30 also waits on S27; S33 waits on S17 and S20; then S31 → S32 and S34.
- **Close:** S50 → S51.

**Critical path:** S08 → S11 → S12 → S13 → S14 → S15 → S17 → S33 → S34 → S50. Beside it, and also
before S50:
- **Fitness:** S24 → S25 → S26 → S57 → S27 → S28. S56 runs beside S26 and must finish before S28.
- **Ground truth:** S06 → S52 → S53 → S55 → S48 → S40 → S54 → S41/S42 → S45 → S46 → S47. With ten
  KA/KB sessions at two at a time, that is about five session-lengths of wall-clock. S47 also joins
  the critical path's S34 and the fitness chain's S56 (P3-1), so S47 is the later of the two chains'
  ends. S46 is the one human wait.

## 4. Step details

### 4.0 What each P0 deliverable is

| P0 item (proposal §4) | Nature | Steps | Durable artifacts |
|---|---|---|---|
| (a) frontend fitness | **Measurement spike** (a1 + a2, §2 D3), protocol first | S05, S24–S28, S56, S57 | `docs/plans/connected-code-graph-p0-ml.md`; `docs/test-reports/connected-code-graph-p0-frontend-fitness-report.md`; `cpg/spikes/frontend-fitness/{corpora.yaml, results/, labels/, truth/, adjudication/}` |
| (b) vertical slice + schema/model freeze | **Contract** + **measurement spike** | S03, S09–S17 | bundle spec + `-spec2`; **`-spec3` required before P0 exit**; `cpg/ccg/{ids.py, bundle.py, schemas/bundle-v1.schema.json, store/ddl_v1.cypher}`; slice report; p0-graph note §1 |
| (c) store spikes | **Measurement spike** on the dedicated instance (durable bring-up script) | S01, S20–S23 | `cpg/infra/ccg-falkordb.sh`; p0-graph note §2–§6 |
| (d) rule-pack format v1 | **Contract** (spec + schema + paper packs) | S30–S32 | rule-pack spec; `rule-pack-v1.schema.json`; `cpg/packs/*/pack.yaml` |
| (e) egress broker contract + offline sandbox | **Contract** + **durable code** | S02, S04, S08 | egress spec; `cpg/sandbox/*`; `cpg/infra/toolchain/*` |
| (f) evaluation protocol | **Contract** (method document) | S06 | `docs/plans/connected-code-graph-eval-protocol-ml.md` |
| (g) fixture system + held-out strata + bilingual cells | **Durable code + ground-truth data** | S35–S49, S52–S55 | fixtures test plan; `cpg/fixtures/**`; `cpg/eval/**`; ground-truth report |
| (h) canonical answer schema | **Contract** (spec + schema + serializer) | S33, S34 | answer spec; `cpg/ccg/{answer.py, schemas/answer-v1.schema.json}` |
| — (bundle spec follow-up) | advisory (accepted) | S58 | `docs/reviews/connected-code-graph-bindvalue.md`; dispositions in `-spec2` |

### 4.1 Environment and sandbox (S01, S02, S08, S24)

- **S01, S02:** accepted. Their contracts are the delivered scripts and READMEs and their gates
  (review §S01, §S02). The S02 image is the base S24 builds on.
- **S08:** accepted, with a follow-up round (§3 S08 row). **The test list is egress spec §10
  (tests 1–10) as amended by ES2**, which supersedes v1.3's seven tests; §7.1.1 controls interop
  detection. **The errata table v1.0 carried here is replaced by ES2** (ES2 §0, §3):
  - **E1 → superseded by ES2 §1.1.** Split lines at `\n`, `\r`, `\r\n`; refuse if any line, after
    removing every leading character ≤ U+0020 and every leading U+007F, starts with `//>`, any
    directive, any case. v1.0's `[ \t\r\n\f\v]` set was too narrow: control-byte-led directives fire
    (replpp uses Java's `String.trim()`). H15's "indented … inert" clauses are false.
  - **E2 → restated unchanged as ES2 §1.2.** `/scripts` is a runner-owned directory volume,
    `ro,noexec`; test 5 lists `/scripts`.

  Also from egress §10: S08 adds `.audit/` to `cpg/.gitignore` (§5.2, finding F15; done).
- **S24 (amended; ML note §11 N0, §5.2, §9).** `devops`. Done when all of:
  1. **Truth tools.** Every §5.2 primary reference tool and every §9 item is pinned in the image
     (Node.js LTS + npm, `scip-python`, `scip-typescript`, `scip-java` with the SemanticDB javac and
     Kotlin plugins, `scip-dotnet`, `golang.org/x/tools/cmd/callgraph` with `scip-go` as fallback,
     `scip-clang` with a matching clang/LLVM and libstdc++ headers, the `scip` CLI, `uv` or `pip`),
     with versions in the content hash.
  2. **Smoke projects, not corpora.** Each primary tool indexes a per-language **smoke project**
     (`cpg/infra/toolchain/smoke/<lang>/`, a few files with one interface or override; `devops`
     picks them) inside the image, read back with `scip print --json` (`callgraph` output for Go).
     A tool that fails is replaced only by an **installed-tool** fallback of ML §5.2; otherwise the
     language is recorded **single-source** in the README (§2 D13). C5 on real corpora is S25's.
  3. **Placement.** The truth tools go in a separate final target `truth` (`FROM toolchain`,
     image `ccg-toolchain-truth:<hash>`), so the image the sandbox runs (`toolchain`) does not
     carry Node, the indexers or clang. The off configuration, S11 and S39 keep a tool set S08's
     flag sweep and frontend-identity test already describe. `scip-clang` runs with parallelism 2.
  4. **S02 carry-forwards (review §S02 Pass 2; ledger 2026-10-29).** Every new stage is
     **`FROM apt-base`** (or `FROM` a stage that is), and installs apt packages **only through
     `apt-snapshot-install`**; no bare `apt-get install`, no `FROM ${BASE}`. `ccg-versions` runs
     **each drop-in in a subshell** that reports its own failure count (for example
     `( fail=0; . "$d"; exit "$fail" ); fail=$((fail + $?))`), so a drop-in's `exit` cannot override
     the count. **Acceptance:** a failing drop-in followed by one containing `exit 0` (p2-m1's
     `20-bad` + `40-exit` pair) makes `ccg-versions` exit non-zero. Each truth tool registers a
     drop-in.
  5. **The rebuilt sandbox image (held until S08's follow-up round is accepted, P3-4).** The tag is
     a content hash of the build inputs, so S24's edit changes the `toolchain` tag too. S08's
     image-dependent tests — 3, 6, 9 and 10(c) — are first re-run against the new tag through the
     runner's one override, **`CCG_TOOLCHAIN_IMAGE`** (kept by ES2 §1.7's closed environment), with
     no edit to the runner. Only when they pass is the runner's default changed, in **one serialized
     edit to `sbx_default_image` in `cpg/sandbox/offline-run.sh`**, after S08's follow-up round is
     accepted (§5.2). `ccg-versions` passes in both images.
  6. `cpg/infra/toolchain/README.md` records each tool's version, its smoke result, any
     single-source language, and both image sizes.

### 4.2 Contracts (S03, S04, S30, S33)

**S03 (accepted): the bundle spec v1.3 controls.** v1.3 plan §4.2's list of what S03 "must contain"
is superseded wherever the spec differs. The differences an implementer will meet:
- **BF3:** `data` records carry **`use`/`def`**, not v1.3's `{from, to}` (spec §5.10); the edge
  property names `fromPath`/`toPath` are unchanged.
- **BF1:** every tuple's first element is the record-type tag (spec §3.1–§3.2), not v1.3's untagged
  tuple table.
- **BF5, BF11, BF12, BF14, BF15:** new records `constant`, `fieldAccess`, `schemaEntry`; a
  symbol's `lang` is its file's; `lastChanged*` computed outside the sandbox; `nameKey` hashed;
  `header.sandboxAuditIds` is an array.
- **Freeze:** spec §1 (§2 D5 here); [P-later] items freeze untested (S15's record).
- **Downstream encoders:** spec §11 is the encoder table; the rows here are written from it.
- **`-spec2` amends about a dozen named clauses** (access paths, `nameNorm`, value-set depth,
  `controlledBy`, `import`, `launchFact` and its argv gate, `configEntry`'s `bindValue` gate and
  `valueHash`, `http` step 8a, locator types, determinism, the graph mapping, validity, provisional
  status). Each §1 heading there names the clause it replaces; there, `-spec2` controls.
- **Follow-ups:** BF16 (proposal §3.8, via S50). The eval-protocol's Q2 (spec §3.5's locator types)
  is **resolved** by `-spec2` §1.9. Consolidation is `-spec3` (§2 D5, S16).

**S04 (accepted): the egress spec v1.3 controls**, with `docs/plans/connected-code-graph-egress-spec2.md`
(D18), which controls in each clause it names (ES2 §1.1–§1.8: the directive refusal, `/scripts`,
container mounts, the namespace `/usr` cell, the `sandbox` record incl. `@AUDIT_ID@`, refusal
precedence, the closed runner environment, escape-axis precisions). Its open stakeholder
question Q1 is §12 here.

**S30, the rule-pack spec**, must contain (unchanged from v1.3 except the anchor list):
- **Format v1:** the pack header (id, version, kind, `vocabVersion`); rules of
  `match`/`extract`/`emit`, with the matchers `exact`, `path-template`, `glob`/`prefix` and
  `regex`; declarations for mount composition (conditional mounts included), role-lifting seeds,
  embedded languages, URL bindings, launch facts and framework defaults; the read-only escape
  hatch, whose query form S30 picks under three constraints: read-only, over one unit's bundle,
  deterministic.
- **Five paper packs:** FastAPI with mount composition; fetch with a wrapper, lifting seeded on
  `fetch` `a0` with the verb at `a1.method`; FastMCP plus URL binding, with the uvicorn default
  8000, the env default `AGENT_TEAM_PORT:-8200` and the Dockerfile `--port`; Kafka, including a
  topic read from config; an invented in-house bus.
- **Walkthroughs:** each pack walked through rule by rule. FastAPI and fetch against S14's **real**
  bundles, and the emitted facts must equal the slice's hard-coded recognizer output. The other
  three against snippets with the records bundle spec §5 would produce. **Every proving-repo
  anchor a walkthrough touches is listed as a locator (bundle spec §3.5), by eval-protocol §8.2's
  definition, with wrapper and chain hops excluded.**
- **Language expressiveness:** the format can express the decorator, annotation and attribute
  shapes S27 measured for all seven languages, or S30 records the gap.
- **Infrastructure-signature pack v0 (a §6.2 item):** ML §8.3's signature classes (connection
  refused, DNS, timeout, OOM kill, TLS, third-party exception classes), with patterns per runtime:
  Python, Node, JVM, .NET and Go.
- **Severity rule table for non-dependency findings (a §6.2 item):** four levels with a stated
  rule each, for injection-type, missing authorization, risky configuration and secrets; secrets
  distinguish present-at-head from removed-within-window. Aligns with bundle spec §7.4.
  `security-expert` may advise; not a gate.
- **Negative examples:** malformed rules the schema must reject, which S31 tests.

**S33, the answer spec**, must contain (unchanged from v1.3; eval-protocol §14 confirms it):
- **Top-level fields:** `answerSchemaVersion`, `interpretation` (structured request, ambiguity
  options, RCA marks, plain-language provenance: model fingerprint, prompt hash, cache key),
  `recipe` (query set plus algorithm id and version), `stamp` (`snapshotId`, `verdictLogOffset`,
  `verdictLogHash`), `staleness` (per unit: built commit and time, `fresh`/`stale-kept`/
  `not-mapped`, reason), `items`, `groups`, and `depthBound` (the bound, plus "stopped at d with k
  unexplored").
- **Recipe:** the algorithm id is S20's choice.
- **Items:** id (a span-bearing symbol id, eval-protocol §5.2), status, reason, direct/indirect,
  distance, repo/component grouping keys, and an evidence chain of hops. Each hop carries
  `{kind, path, line, unit, repo, edgeType, status, reason}`, plus `Link` expansion into
  `BOUND_BY`/`LIFTED_VIA` (`LIFTED_VIA` per site, `{ord, siteId}`, bundle spec BF6).
- **Groups:** folded groups **always enumerate every member**, each with its own chain (ML M2, P3-1,
  eval-protocol §5.2).
- **Ordering:** the deterministic ordering rule and the canonical-bytes rule (bundle spec §8.1).
- **Versioning:** the versioning policy.

### 4.3 Method notes (S05, S06)

**S05 (accepted): ML note v1.3 controls** the corpora (§3), the frame and sample (§4: a census
frame, simple random sample by inverse sampling with a cap of 500, which supersedes v1.3's "at least
150"), the truth sources (§5), the labelling protocol and brief (§6), the metrics (§7) and, as
ratified by §2 D14, the decision rules (§8). Its step outputs became S24–S28, S56 and S57 (§16).
v1.3 §4.3's S05 bullet list is superseded.

**S06 (accepted at v1.2): the eval-protocol note controls** the strata,
counts, item schemas, hit rule, matching, estimators, completeness trigger, development-contact
rule, split, held-out manifest, leak check, AC-24 counts, noise bound, AC-5 fail procedure, planted
minimums, pre-registration format and GC's tooling procedures. What each step takes from it is its
§14, folded into the rows of §3 and listed in §14 here. v1.3 §4.3's S06 bullets (strata, ≥ 75 / ≥ 8,
estimator choice, contact rule, procedures) are the brief S06 answered; where the note decides, it
controls.

### 4.4 The vertical slice (S11–S15)

- **S11, extraction (`graph-dba`):** `run_extract.sh` stages each unit with `git archive` at the
  pinned commit, runs `build-cpg.sh` and the `extract.sc` Joern script through
  `cpg/sandbox/offline-run.sh` (`extract.sc` in egress §7.3.1's `joern --script` form: runner-staged,
  hashed into `sandbox.script.sha256`, no `//>` directive), and times parse, extract and normalize
  separately. **After each parse, the §2 D15 content check runs** before extraction (`joern-parse`
  exit 0 can mean an empty CPG, ES §11 R11). Output stems use ES2 §1.5's `@AUDIT_ID@` token.
  `extract.sc` emits records by the bundle spec as amended by `-spec2` (S11's row lists `-spec2`
  §4.1's emitter items, incl. G-4's self-edge drop and de-duplication):
  - symbols, call sites, decorators and argument value sets;
  - points: `METHOD_PARAMETER_IN` → `param`, `RETURN` → `ret`, a control-structure condition →
    `cond`, and each non-operator `CALL` as a point;
  - **the per-method projection:** for each point's uses, walk `REACHING_DEF` backward inside the
    method through identifiers and operators to the nearest point definitions, and emit one `data`
    record per (use point, def point), with the fields of bundle spec §5.10 (`use`, `def`,
    `toPath`, `fromPath`). Access paths per bundle spec §5.1, capped at depth 3; deeper paths are
    truncated and marked *possible*, `field-truncated`. Then walk `CDG` from each point's statement
    to its condition nodes and emit `controlledBy`;
  - a per-method projection-time histogram.

  `normalize.py` maps the TS and Python frontends' differences away (repo-relative paths,
  object-literal fields) and writes `bundle-v1` JSONL through `cpg/ccg/bundle.py`. **S11's gate
  reviews `extract.sc` as code** (egress §11 R10: the script can do anything a JVM can inside the
  sandbox; `sandbox.script.sha256` ties each run to the reviewed bytes).
- **S12, assembly (`coder`):** recognizers hard-coded; the slice is not a pack runtime.
  - **Value sets** resolve across files: module constants such as `API_PREFIX` (bundle spec §5.9
    `constant`), and `{ref}` entries.
  - **The FastAPI provider** from the decorator path plus the `include_router(…, prefix=…)` prefix
    of the mounting call (`kw.prefix`, bundle spec §5.1), wherever that call is, with any enclosing
    condition recorded.
  - **The fetch consumer** seeded on `a0`, with the verb at `a1.method`.
  - **Role lifting** iterates to a fixpoint over `data` records.
  - **Linking** is exact plus path-template matching, and produces one `Link` per consumer symbol
    with per-site `LIFTED_VIA {ord, siteId}` (bundle spec BF6). `http` keys pass `-spec2` §1.8's
    step 8a (T6, `{#}`), with SL-1 and SL-2 unchanged.
- **S13, load and impact (`coder`, reservation R-B):** `load.py` follows graph note §3.5:
  1. indexes on the empty `ccg_p0slice__build_<id>`;
  2. nodes, then edges, as `UNWIND` batches of **≤ 20k rows** (§2 D16) over one connection, with
     retries off and a socket timeout of minutes, and an explicit query `timeout` if `TIMEOUT_MAX`
     is nonzero;
  3. constraints after the load, polled to `OPERATIONAL`;
  4. `:Snapshot` written last;
  5. `RENAME` to `ccg_p0slice`.

  `impact.py` is a minimal level-synchronous BFS implementing only the §6.2 table rows the slice
  needs: `CALLS ←`, `HAS_* ←`, `DATA ←` path-matched (`arg(head, path)`, BF4), `VIA ←`/`TO ←` and
  `PROVIDES →`. It keeps a best-predecessor map.
- **S14, the slice report, scored against:**

  | # | Criterion | Pass when |
  |---|---|---|
  | SL-1 | AC-1 link and chain | Exactly one **confirmed** `Link {kind: 'http'}` runs from the `salesperson/src/api/endpoints.ts` health call site to the `@router.get("/health")` handler inside `build_storefront_router()` in `falkor-chat/server/falkorchat/storefront_api.py`, keyed `GET /shop/api/health`. Its `LIFTED_VIA` chain is, in order: the endpoints.ts call → `apiFetch` (`p0.path`, verb `p0.method`) → `buildUrl` (`p0`) → the `fetch` call (`a0`, verb `a1.method`). Every hop's file and line matches the source at the pinned commit. |
  | SL-2 | Mount composition | All three `/health` providers are present: the storefront one composed to `/shop/api/health` from the prefix in `app.py`; the dev router's with its `dev_surface` condition recorded; and the bare re-registration. Only the storefront one links to the SL-1 consumer. **Counterfactual:** the slice is re-run on a staged copy whose `API_PREFIX` is `"/x/api"`; the composed key must become `GET /x/api/health`, with the SL-1 link now unmatched. |
  | SL-3 | Load integrity | Every constraint `OPERATIONAL`, none `FAILED`. `:Snapshot` present. `RENAME` done. |
  | SL-4 | Impact answers | Contract mode from the provider's `ContractPoint` lists the endpoints.ts caller: confirmed, distance 1, with the rendered chain expanding `LIFTED_VIA`. Value mode from a change in `buildUrl` lists `apiFetch` and its endpoints.ts callers through path-matched `DATA`. The AC-1 rows of bundle spec §5.10 are present. |
  | SL-5 | Cost and density | For both units, the projected graph is at most 0.7 MB per source file (2× graph note §6.4's upper estimate), by `GRAPH.MEMORY USAGE` ÷ files. The projection stage takes at most 2× the unit's Joern parse time (K7). Nodes and edges per file recorded. |
  | SL-6 | Determinism | Two full runs from the same commit give byte-identical bundles (bundle spec §8.5's exempt fields excepted), identical `Link` sets, and identical BFS answer facts. |
  | SL-7 | Schema accounting | Every field or record added, removed or retyped relative to bundle spec v1.3 is listed, incl. any `qname` re-derivation mismatch, and the final run validates against the schema as amended. |

  The two SL-5 bounds are this plan's engineering choice, so that the freeze has a number to hold
  to; the reviewer may challenge either at S15.
- **S15, the gate.** `analyst` reviews S11–S14; `graph-dba` checks SL-3 and SL-5.
  - **All pass:** v1 (`-spec3`) freezes. If SL-7 lists deltas, a `-spec4` records the amended
    schema and the freeze applies to it.
  - **Any fail:** v1 stays provisional until P1 exit; the P1 plan carries the freeze.
  - **Either way:** the record names `-spec3` (or `-spec4`) as the contract, confirms S10 encodes
    every [pre-freeze] item of §1, lists the [P-later] items and §1.6/§1.7
    "frozen, untested" (or "provisional, untested");
    S17, S28, S30 and S33 proceed — marked "against provisional v1" if v1 did not
    freeze, and re-checked at P1 exit.

### 4.5 Store spikes (S20–S23), all on the dedicated instance

Every load in this section obeys §2 D16.
- **S20:** `synth.py` generates a 1,000-file graph at graph note §6.4's v2 densities: about 455
  nodes and 2,400 edges per file, so roughly 455k nodes, 2.4M edges and about 0.3–0.35 GB, which fits
  `maxmemory 1536mb` with room for a build copy. Hubs from the V13 fan-out distribution.
  - **Anchors:** 50 seeded anchors spanning hub and non-hub symbols.
  - **Comparison:** the stateful BFS (S13's operator, extended to the full §6.2 impact column)
    against `algo.SSpaths`, which takes a single `relDirection` and so stands in only for
    single-direction, single-state segments such as call-only reverse reach. S20 benchmarks those
    segments and says so.
  - **Repeat-diff:** each configuration run 20 times; the `(item, status, distance, chain)` sets
    diffed.
  - **Decision rule:** adopt SSpaths for a segment **only if** (1) 20/20 repeats are identical,
    chains included; (2) its item set equals the BFS's; (3) p95 answer latency is at least 30% lower
    at depth ≥ 3. Otherwise the BFS. Per-level read latencies (p50, p95, p99) recorded for S22.
- **S21:**
  - **Rollback form:** probe `MULTI; RENAME live prev; RENAME build live; EXEC`, (i) cleanly,
    (ii) with `build` absent, (iii) with a reader in flight. Whether Redis rolls back a transaction
    on a command error is not verified here; S21 verifies it. If the form is unsafe, the decision
    stays "zero kept snapshots, rollback = re-assemble last good" (graph note Q4).
  - **Telemetry:** probe the `telemetry{<key>}` stream: can it be disabled or capped by a module or
    graph config? Does an admin `XTRIM` hold? Does `RENAME` leave it behind (V10)? **How long does
    it survive `GRAPH.DELETE`** (the asynchronous-removal sighting)? Record "can telemetry be
    capped", the removal timing, and the operating rule that follows (a retention or redaction check
    run straight after `GRAPH.DELETE` may still see query text).
- **S22:**
  - **ACL matrix:** `ccg_build`, `ccg_verdict`, `ccg_engine`, `ccg_raw_p0` × `GRAPH.QUERY`,
    `RO_QUERY`, `EXPLAIN`, `PROFILE`, `LIST`, `RENAME`, `EXPIRE`, `DEL`, `GRAPH.CONSTRAINT`, `INFO`,
    plus `SET`/`DEL`/`EVAL` on the lock key, a read of `telemetry{…}`, and access to a foreign key.
    Each cell allowed or `NOPERM`, matching proposal §3.5. The `falkordb-py` constructor succeeds
    for each user (V17).
  - **`lock_replay.py`** implements proposal §3.8's lock (`SET … NX PX`, compare-and-delete release
    via `EVAL`), the fsynced JSONL log, patch-and-advance in one query, swap replay and post-`RENAME`
    remainder replay.
  - **`test_lock_replay.py` asserts:** (1) a verdict issued during a build is present after the swap
    (B6); (2) with `PX` forced short and a sleep before `RENAME`, a verdict landing on the old key is
    present after the swap (P2-3); (3) after a kill between append and patch, the next verdict or
    swap re-converges.
  - **Sizing rules:** `TIMEOUT_MAX` default = the smallest round value ≥ 10× the worst p99
    per-level read from S20, with the engine's batch size set so each per-level read is ≤ a tenth of
    it; lock ttl = 3× (worst replay of 1,000 log entries + `RENAME` + remainder check). Both recorded
    as FR-21's and §3.8's defaults, labelled **P0-scale defaults** that P1 re-derives.
  - **Instance-wide side effects.** With the chosen value set, S22 runs a `UNWIND` batch load as
    `ccg_build` **with and without an explicit `timeout`** (the S07-verified quirk says a nonzero
    `TIMEOUT_MAX` kills writes), `GRAPH.CONSTRAINT CREATE` with backfill on a populated key, and a
    full-text index backfill. Any kill or rejection is recorded with the loader rule P1 must follow.
- **S23** writes the p0-graph note's decisions table, **runs the stall probe** (§3 row; reservation
  R-C4) and folds the new quirks into `claude/graph-dba/falkordb-quirks.md` (dated, with commands).
  Expected candidates: the `MULTI` behaviour, the telemetry controls and removal timing, `EVAL`
  under ACL, the write stall.
- **S17**, after S15 and S23, writes p0-graph note §1, "Graph model v1": the graph note §3.2 as
  amended by §6, with the slice's deltas, the S20 algorithm choice, the corrections in §11 F1/F2,
  and the bundle-spec items of its §3 row. It writes `ddl_v1.cypher` (graph note §3.4, §6.4, plus
  any `UNIQUE` gate S17 adopts from the graph-dba check, e.g. `Dependency.depId`) and
  `test_ddl_live.py`.

### 4.6 Frontend fitness (S24–S28, S56, S57)

S24 is in §4.1. Paths below are under `cpg/spikes/frontend-fitness/`. The ML note sections cited
are the specification; this section fixes owners, order and done-conditions.

- **S25 (amended; ML §11 N1, N2), `graph-dba`.** Order inside the step, per candidate and then per
  chosen corpus: **frame → C5 → draws and briefs → first Joern run** (ML §3.2, §11 N2). Dispatch as
  P0-18a (everything before the first Joern run) and P0-18b (the Joern runs), so a kill loses less
  and the pre-Joern artifacts can be gated for S26/S56 to start. Done when:
  1. **Corpora:** `corpora.yaml` lists language, project, URL, commit SHA, build command, licence
     note and the C1–C8 evidence (ML §3.2); exclusions 1–8 and the `opentelemetry`/
     `microservices-demo` grep are checked (ML §3.3). Fetch once, with network, into
     `cpg/.ccg-work/corpora/`.
  2. **Frame:** `frame/<lang>.jsonl`, the tree-sitter census frame (ML §4.1), reproducible from
     `corpora.yaml` alone.
  3. **Reference truth run (N1):** per candidate, **before its first Joern run**,
     `truth/<lang>.c5.json` (tool, version, exit status, documents indexed, 50-site pilot coverage,
     seed prefix `ccg-p0a-v1-c5|`); a C5 failure rejects the candidate and moves to the next one, and
     the tool is never swapped to rescue a corpus. Then, for the chosen corpus,
     `truth/<lang>.ref.jsonl` covering every sampled ordinal (fields per ML §11 N1). Runs in the
     `truth` image with network and the corpus's dependencies, **outside the sandbox and never in
     the off run's workspace** (ML §9).
  4. **Draws and briefs (N2):** `sample/<lang>.jsonl` and `sample/<lang>.private.jsonl` (ML §4.4,
     cap 500); `sample/<lang>.brief.md` (ML §6.5's template, slots filled, bundle spec §4.1 plus the
     language's §4.2 subsection appended verbatim with the spec version; no unfilled `{…}`);
     `sample/<lang>-flow.jsonl` (30 flow methods, ML §6.4, seed prefix `ccg-p0a-v1-flow|`, with each
     method's source text); `matcher-table.md` (ML §7.1).
  5. **`make_adjudication_input.py`**, written here, opening only `labels/` and `truth/` (`analyst`
     checks this statically at S25's gate). `graph-dba` does **not** run it.
  6. **Joern runs (P0-18b):** each frontend runs twice. **Off** through `offline-run.sh` from a
     **clean** `git archive` checkout with no dependency or build-output directories (no
     `node_modules`, venv, restored `vendor/`, `~/.m2`, `~/.gradle`, `~/.nuget`, `GOMODCACHE`). **On**
     in the `toolchain` image with network and the language's fetch flag (`--fetch-dependencies` for
     Java and Go, `--download-dependencies` for Kotlin and C#); `--jar4import-url` never. Every CPG
     passes §2 D15's content check; a CPG with no methods counts as a parse failure of every file in
     it, never as success. `measure.sc` runs through the runner's `joern --script` form and its gate
     reviews it as code (egress §11 R10). Each Joern run's timestamp is later than its corpus's C5
     record.
  7. Gate: `data-scientist` confirms corpora and draws (ML §3.3: "S25's gate checks 1–8 by name");
     `analyst` checks the adjudication-input script statically and `measure.sc`.
- **S26 (blind call-site labels), seven fresh `qa-engineer` dispatches.** Given exactly ML §6.1's
  inputs, with `sample/<lang>.brief.md` pasted as-is; forbidden ML §6.2's list; `teco` greps each
  dispatch's own subagent transcript (ML §6.2) and a hit voids that language's labels. Append and
  resume (ML R5).
- **S56 (new; golden flow labels; ML §11 N3, §6.4), seven fresh `qa-engineer` dispatches**, never
  the same instance as that language's S26 labeller. Input: `sample/<lang>-flow.jsonl` and that
  corpus only; never S11's projection, `extract.sc`, `results/` or `truth/`; ML §6.2's list applies.
  Output: `labels/<lang>-flow.jsonl`, per method its `ordinal` and per pair `{kind: data|control,
  source, sink, evidence}`, with points named as bundle spec §5.4–§5.6 name them. Done when all 30
  methods are labelled, the transcript grep is clean (`teco`), and `data-scientist` has checked the
  label schema, ≥ 1 pair per method and cited evidence. Runs alongside S26.
- **S57 (new; pre-Joern adjudication; ML §11 N4, §5.1 step 3), one unit per language**, as that
  language's S26 lands. **Owners per §2 D13:** a fresh `qa-engineer` fitness custodian (FC) runs
  `make_adjudication_input.py` and writes `adjudication/<lang>.input.jsonl`; `data-scientist`
  reads it in place and returns, per disagreeing or `unsure` site, keyed by `ordinal`: final `T(s)`
  with roles, `N(s)`, evidence, and which source was wrong; FC writes them to
  `adjudication/<lang>.jsonl` and writes `truth/<lang>.final.jsonl` (agreed-or-adjudicated truth for
  every labelled site). Done when every disagreement and `unsure` site has a decision with evidence,
  ML §5.1 step 4's label-noise figures are computed, and `analyst` confirms from **both**
  transcripts that no Joern artifact (`results/`, any `cpg.bin`, `measure.sc` output) was opened.
- **S27 (amended; ML §11 N5), `graph-dba` runs; `data-scientist` adjudicates misses.** In order:
  (1) provisional metrics on `truth/<lang>.final.jsonl`, to list the misses; (2)
  `adjudication/<lang>-misses.jsonl`, every Joern miss in either configuration with its decision,
  cited evidence and cause class (ML §7.2), the decisions being `data-scientist`'s and the file
  `graph-dba`'s (§2 D13); (3) overturns applied as `truth/<lang>.final2.jsonl`; (4) **reported
  metrics computed only on `final2`**, with the provisional figures beside them; (5) ML §8.5's table
  per language, the decisions by ML §8 as ratified (§2 D14), the decorator/annotation/attribute shapes
  S30 needs, "ineligible as held-back" listing the five external corpora (ML §3.3 reverse rule), and
  any evidence from the off runs bearing on egress §11 R7's three `--help`-only flag
  classifications. Done when `analyst` has checked the miss log independently and that every §8
  number traces to `final2`.
- **S28 (amended; ML §11 N7), `graph-dba` runs; `data-scientist` adjudicates.** Runs S11's
  `extract.sc` projection on every corpus, offline (each CPG passing §2 D15), and computes ML §7.3's
  metrics on S56's labels. Writes `adjudication/<lang>-flow-misses.jsonl` (every missed golden pair)
  and `adjudication/<lang>-flow-spurious.jsonl` (the ≤ 50 seeded spurious pairs, prefix
  `ccg-p0a-v1-spur|`), decisions by `data-scientist`, files by `graph-dba`. Overturned misses and
  confirmed omissions are applied to the golden set **before** the reported pair recall (provisional
  beside it). Records MB per file, nodes and edges per file, build time, **symbol and call-site
  density per file** (graph note Q3), the omission rate and the *truth-incomplete* flag (ML §8.4),
  and decides `flowAvailable` per language by ML §8.4. For Python and TS the slice's figures (S14)
  are cross-checked; a projection-success disagreement > 0.05 is investigated before the rule is
  applied. Checker: `analyst` on both adjudication logs, `data-scientist` on the rule.

### 4.7 Fixture system (S35–S39)

S35 fixes, as a numbered table, every planted item. The categories that **must** appear are below;
**eval-protocol §12.2 sets the minimums** and S35 the counts.

| Category | Gateable in |
|---|---|
| Link kinds and statuses: http, mcp_tool, script → tool, launch chains (`.mcp.json`, Compose), URL binding with two launch configs of one unit on 8000 and 8200, plus an example config | P1 |
| Link kinds and statuses: rpc (gRPC), topic (Kafka and the in-house bus), `library_api` with version skew | P2 |
| Link kinds and statuses: data_entity (SQL table/column, graph label), config_key/env_var | P3 |
| **AC-6 unprovable kinds, ≥ 3 shapes each:** runtime-built URL, dynamic import, example config, prompt mention | P1 |
| **AC-6 unprovable kinds, ≥ 3 shapes each:** topic from config | P2 |
| **B1's four assertions** (the proposal review's B1; graph note §6.1), on a class field | P1 |
| **B1's four assertions**, on a SQL table | P3 |
| **AC-25:** 3 data-flow and 3 control-dependency effects, at distances 1–3 | P1 |
| **P3-1:** a planted mis-resolved call, in both Python and TS, so whichever language S27 puts at ≥ 0.95 carries it | P1 |
| **AC-9b:** an unbuildable unit and a never-built unit | P1 |
| **AC-19:** a multi-repo overlay | P3 |
| **SQ-C:** a signature change in Python (the note is kept) and in Java (the note is dropped) | P4 |
| **P2-7:** a taint finding edited at a middle hop and at an end | P5 |

- **Repos:** `fx-svc` (Python: FastAPI, FastMCP, SQL, a Kafka producer, shell scripts, Compose,
  `.mcp.json`, a Markdown prompt); `fx-web` (TS: a fetch wrapper, an MCP client config, a Kafka
  consumer); `fx-lib` (Java: a shared library with overloads, the in-house bus, a gRPC `.proto` with
  its service).
- **Manifests:** each row carries the planted id, kind, both locators (bundle spec §3.5), the
  expected status and reason (bundle spec §7), the phase and the assertion ids. Fixture items are
  **dev** data, never AC-5 held-out (F7).
- **S39's materializer** creates the three git repos under `cpg/.ccg-work/fixtures/` with a scripted
  history (fixed author, committer and dates, so SHAs are stable), including in-window and
  out-of-window commits for the later RCA and secrets windows.

## 5. Shared state and sequencing

### 5.1 FalkorDB instances and graph keys

- **`falkordb-dev` (6379) is shared, and no P0 step writes to it.** Every P0 brief says so. Every
  P0 script takes host and port from an explicit argument, with no default of 6379. The `joern-cpg`
  pipeline's `--load`/`--reset` is not used in P0. Reading `cpg_falkorchat` is permitted for
  builders and **forbidden** for the ground-truth agents (§6.3).
- **`ccg-falkordb` (6390, S01).** Each step that uses it holds one reservation, granted by `teco`
  one at a time, except that two non-exclusive ones may overlap.

  | Reservation | Step | Keys | Exclusive? | Why |
  |---|---|---|---|---|
  | R-B | S13 | `ccg_p0slice`, `ccg_p0slice__build_*` | no | ordinary load and reads |
  | R-C1 | S20 | `ccg_spike_synth*` | **yes** | latency benchmark |
  | R-C2 | S21 | `ccg_spike_swap*` | **yes** | `MULTI`/`RENAME` probes; telemetry config may be instance-wide |
  | R-C3 | S22 | `ccg_spike_lock*`, `ccg_<…>:verdict_lock` | **yes** | `ACL SETUSER` and `GRAPH.CONFIG SET TIMEOUT_MAX` are instance-wide |
  | **R-C4** (new) | S23 | `ccg_spike_stall*` | **yes** | the stall probe can make the whole instance unresponsive for minutes |
  | R-D | S17 | `ccg_ddl_test` | no | DDL applied to an empty key |

  - **Exit hygiene:** each exclusive step restores the instance config it changed (`TIMEOUT_MAX`
    back to 0, probe ACL users deleted) and says so in its note.
  - **P0 exit:** `reset-p0` empties the instance.
- **Host contention (6 GB VM).** S20 and S23 must not overlap any JVM-heavy or build-heavy step:
  S11, S25 (both the truth runs and the Joern runs), S27, S28, S39's live test, S08's JVM tests,
  and the S02/S24 image builds. Joern steps should not overlap each other (2–3 GB heap each), and
  S25's truth runs (`scip-clang` at parallelism 2, `scip-java`, `scip-dotnet`) count as JVM-heavy for
  this rule.

### 5.2 Shared files

| File | Writers, in order | Rule |
|---|---|---|
| `cpg/infra/toolchain/{Dockerfile, build.sh, README.md}` | S02 (done) → S24 | strictly sequential |
| `cpg/sandbox/offline-run.sh` (incl. its image-tag source, `sbx_default_image`) | S08 (incl. its follow-up round) → S24 (one edit, to `sbx_default_image`) | **S24 edits it only after S08's follow-up round is accepted** (P3-4). Before that, S24 re-runs S08's tests through `CCG_TOOLCHAIN_IMAGE`, with no edit (§4.1 S24 item 5) |
| `cpg/sandbox/scripts/cpg-content-check.sc` | S08's follow-up round only | S11, S25, S28, S39 run it through the runner; none edits it (§2 D15, P3-7) |
| **Runner-consumer rule** (P3-10) | — | Every step that runs `offline-run.sh` (S11, S24 item 5, S25, S28, S39) depends on S08's follow-up round, so all of P0's measured runs use one runner |
| `cpg/ccg/keynorm.py`, `cpg/tests/test_keynorm.py` | S12's P0-11a only | S12's P0-11b, S49's `match_keys.py` and the P1 assembler import `mask_http_segments`; none edits or copies it (§2 D19) |
| `cpg/tests/test_bundle.py` | S10 (incl. its follow-up round) → S16 | S16 re-points `SPEC_TEXT`/`SPEC2_TEXT` at `-spec3` (P3-2) |
| `cpg/infra/README.md` | S01 only | — |
| `claude/graph-dba/falkordb-quirks.md` | S07 (done) → S23 | strictly sequential; nobody else writes it in P0 |
| `docs/plans/connected-code-graph-p0-graph.md` | S20 (§2) → S21 (§3–§4) → S22 (§5) → S23 (§6) → S17 (§1) | one owner, sequential; S17 waits for S23 |
| `docs/test-reports/connected-code-graph-p0-frontend-fitness-report.md` | S27 (§a1) → S28 (§a2) | sequential |
| `cpg/spikes/frontend-fitness/{truth,adjudication}/<lang>*` | S25 (`c5`, `ref`) → S57 FC (`input`, adjudication, `final`) → S27 (`misses`, `final2`) → S28 (`flow-*`) | per language, sequential; each writer creates new files and never rewrites an earlier step's |
| `cpg/tests/test_rule_pack_schema.py` | S31 → S32 | sequential |
| `cpg/tests/test_eval_tools.py` | S48 → S49 | sequential |
| `cpg/eval/headless/{agents.json, settings.json, guard-paths.sh}` | S52 only; S53 reads them | on an S54 failure, the fix goes to whichever owner's file is at fault, never to both at once |
| `cpg/pyproject.toml` | S09 (done, **frozen**) | No other step edits it. A missing dependency stops the step and goes to `teco`. (`pyyaml`, which eval-protocol §7.1's V2 check needs, is already in `dev`.) |
| `cpg/.gitignore` | S09 (`.ccg-work/`, done) → S08 (`.audit/`, egress §10) | **Changed from v1.3**, which reserved it to S09: egress spec §10 makes S08 a second writer. S09 is committed, so S08 is the only remaining writer and nothing needs serializing (ledger, 2026-10-29) (F15) |
| `cpg/ccg/schemas/` | S10, S31, S34, S39 write **different** files | no collision |
| `cpg/ccg/bundle.py`, `bundle-v1.schema.json` | S10, then S16's follow-up | only if S15 shows a schema change |
| `docs/plans/connected-code-graph.md` | S50 only | — |
| `docs/reviews/connected-code-graph-p0.md` | `analyst`, one `## Pass N` per gate | `teco` serializes analyst dispatches that touch it |

### 5.3 Facts one step decides that another step must encode

| Decided in | Fact | Must be encoded by |
|---|---|---|
| S03 | the encoder table of bundle spec §11 as amended by `-spec2` §1.12 and §4.1 (tuples, hash, golden vectors, records, order, `qname` grammar, `repoId`, locators incl. `config`/`unit`, `keyNorm` incl. step 8a, reason codes, graph mapping, argv and `bindValue` gates) | S09, S10 (+ follow-up), S11, S12, S25 (brief excerpt), S56 (point names), S35/S39, S06/S40/S52, S33, S17 — per bundle spec §11 and `-spec2` §4.1 |
| S04 | the argv audit format, escape axes, flag lists, `joern --script` form — with `docs/plans/connected-code-graph-egress-spec2.md` (directive rule, `@AUDIT_ID@`, closed runner environment) | S08 (+ follow-up); S11, S25, S28 (scripts through the runner; output stems via `@AUDIT_ID@`) |
| **S15** | **whether bundle schema v1 and the graph model freeze** (and any SL-7 deltas) | S17, S28 (projection script version), S30, S33, S34 |
| S20 | BFS vs `algo.SSpaths`, and per-level latencies | S22, S33, S17 |
| S22 | the `TIMEOUT_MAX` default and the lock ttl; the load-under-cap rule | S17, S50 |
| S23 | the batch ceiling (§2 D16) | S17; P1 loader |
| S05 | corpora criteria, frame, sample, truth sources, brief, metrics, decision rules | S24 (tools), S25 (frame, C5, draws, briefs), S26/S56 (labels), S57 (adjudication), S27/S28 (metrics, rules) |
| S25 | `sample/<lang>.brief.md`, `sample/<lang>-flow.jsonl`, `truth/<lang>.ref.jsonl`, the adjudication-input script | S26, S56, S57 |
| S57 | `truth/<lang>.final.jsonl` | S27 |
| S27 | the indexer and fold/list decisions per language; decorator and annotation shapes | S30, S50 |
| S28 | `flowAvailable` per language | S50; P1 extractors |
| S06 | eval-protocol §14's table (strata, schemas, counts, hit rule, matching, estimators, trigger, contact rule, split, procedures, minimums) | S14/S30 (touched-anchor lists), S33, S35, S40–S49, S52 — per §14 here |
| S40 | `EVAL_COMMIT` and the exclusions (incl. the prompt-bearing glob list) | S41–S47; P1's gate builds and prompt extractor |
| S52, S53 | the allowed tool set, the role-root layout, the stream-log location, the agent names | S48's `audit.py` and S54 |
| S58 | `bindValue` advisory dispositions — **recorded in `-spec2`** (§2, §5 there), before S15 | S10's follow-up (encodes them), S15 (freezes them), `-spec3` (consolidates them); P1 writers (`-spec2` F-10) |

## 6. Ground-truth isolation protocol (P0(g), decision S36, ML §8.2)

Carried from v1.3 §6 unchanged, except where S06 now sets a number (five strata, session counts,
the extra-pass rule, the spot-check size); those places cite the eval-protocol note.

**Decision (stakeholder, 2026-10-29, review B1).** The ground-truth writers run as **separate
headless `claude -p` sessions** that `teco` starts through Bash, with a working directory outside
the repo. They are **not** `Agent` subagents of the build session. `teco` only *runs* the launcher.
`devops` builds the launcher (S53), `cobb` builds the writers' agent definitions (S52), and the
planted-breach rehearsal (S54) must pass before any real writer starts.

**Why the writers exist.** Decision S36: "an isolated read-only agent, spot-checked by the
stakeholder" writes the keys. ML §8.2 adds a second, independent writer so that **completeness**
can be estimated by capture–recapture. Two isolated writers satisfy decision S36 as long as each is
isolated and read-only on the repo, and the stakeholder spot-checks (F4). The protocol covers P1's
AC-5 strata — **five: `http`, `mcp`, `scripts`, `wiring`, `flow` (eval-protocol §3, E2)** — and
AC-24's impact cells. Later phases reuse it.

CLI facts below come from `claude --help` (Claude Code 2.1.312, read 2026-10-29 without starting a
session). Anything marked *verify (S54)* was not observed running, and the rehearsal must confirm it.

### 6.1 Roles

| Role | Runs as | Model | Strategy |
|---|---|---|---|
| KA | headless session, agent `ccg-key-writer-provider` (S52), one session per stratum (five) | `opus` | **provider-first**: list every route, MCP tool, script entry point and anchor first, then hunt each one's consumers |
| KB | headless session, agent `ccg-key-writer-consumer` (S52), one session per stratum (five) | `sonnet` | **consumer-first**: list every outbound call, client config, URL, tool mention and script invocation first, then resolve each one's provider |
| QW | headless session, agent `ccg-question-writer` (S52) | `opus` | writes the AC-24 impact-cell questions (eval-protocol §10.1), natively in EN and PT, with gold |
| AN1, AN2 | headless sessions, agent `ccg-annotator` (S52) | `sonnet`, `opus` | each labels QW's questions ambiguous or not, independently |
| GC (ground-truth custodian) | **in-session** `qa-engineer`, one fresh instance for S40, S45, S47 | default | runs the tooling, writes `cpg/eval/**` and the ground-truth report. It never writes keys; its brief forbids `ingest_document`, and §6.5 checks that by tool name |
| Method owner | in-session `data-scientist` (S40, S45, S47) | default | specifies each procedure (S06); adjudicates near-misses by pair id; checks GC's outputs. Writes nothing outside `docs/plans/`/`docs/reviews/` |
| Launcher operator | `teco` | — | runs `run-writer.sh` (S53); never reads keys |
| Spot-checker | the stakeholder (S46) | — | decision S36's spot-check plus the PT gold spot-check (ML §8.3), sized by eval-protocol §7.3 (E10) |

**Independence.** KA and KB differ in agent definition (two distinct prompts from `cobb`), in model,
and in strategy (ML §8.2). No writer session shares a conversation, a working directory or a
scratch space with another (§6.3).

**Kills and resumption.** A writer is never reused for another role. Each writer appends every
finished item to its output file as it goes. If a session is killed (usage limit or timeout):
1. **Audit the partial session first.** GC runs `audit.py` over every stream log and the
   transcript so far. If any voiding event is present, the session is **not** resumed: its output
   is discarded and a fresh session of the same role and stratum starts from an empty output.
2. **Resume through the same argv builder.** `teco` runs `run-writer.sh resume <role> <stratum>`,
   which calls the same `build_argv` with `--resume <uuid>`, the fixed continuation prompt
   ("continue; your output file lists what is already done") and the next attempt number `<n>`.
3. **The output survives truncation.** The reader drops an unparseable final line left by
   `timeout`; the writer's next append starts on a fresh line.
4. **Each attempt is audited.** Each attempt is recorded in the run index; the final audit covers
   all attempts together.

A resume is the same writer in its own role, so independence is unaffected. A session that cannot
be resumed (a corrupt transcript) is discarded and redone fresh.

### 6.2 What each writer is given (S40 builds the kit; S53's launcher lays out each role root)

Everything lives **outside the repo**, under `~/.local/share/ccg-eval/<E>/`, where `<E>` is the
first 12 characters of `EVAL_COMMIT`:

```
~/.local/share/ccg-eval/<E>/
  kit-src/                      the single exported kit (S40), read-only master copy
  roles/<role>-<stratum>/       one per session = that session's working directory and its ONLY reachable tree
    kit-src/                    a private read-only copy of the master kit
    BRIEF.md                    role brief (S40), read-only
    out/                        the only writable directory; KA.jsonl / KB.jsonl / QW.jsonl / AN?.jsonl
  runs/                         launcher logs — OUTSIDE every role root
    index.jsonl                 one line per attempt: role, stratum, session id, n, model, CLI version, binary/settings/agents SHA-256, start, end, exit, cost
    <session-id>.<n>.stream.jsonl   one per attempt n (start = 1, each resume +1); the audit's primary source
  work/                         GC's working files (audit, match, nearmiss, capture)
  heldout/                      the held-out set, plaintext (S47; D11: hidden by convention only)
```

- **`kit-src/`** is `git archive $EVAL_COMMIT`, extracted, minus `cpg/`, `deprecated/`, every path
  matching `docs/**/connected-code-graph*`, and `docs/BACKLOG.md` (eval-protocol §3.3 and §13.1
  give the full scope and procedure). It is then **content-checked**: GC greps it,
  case-insensitively, for `connected-code-graph|connected code graph|C-1001|\bccg[_ -]`. Every file
  with a hit is excluded both here and in `cpg/eval/eval-manifest.yaml`, is listed in
  `kit-manifest.json`, and is re-grepped to zero. Excluded files that could have held a P1 item are
  counted as excluded scope in the ground-truth report. `kit-src/` has no `.git` and is
  `chmod -R a-w`.
- **`BRIEF.md`** (one per role and stratum) contains: the role's strategy; **eval-protocol
  Appendix B pasted verbatim** (stratum definitions, item schemas and E1's naming rule in
  writer-facing form; nothing else from that note); the rubric "a real dependency
  exists in at least one deployable configuration"; for `flow`, the **shared anchor list** drawn by
  S40 (`export_kit.py --draw-flow-anchors`, recorded seed, eval-protocol §3.6), identical for KA and
  KB; the output path, and a plain statement of what the session cannot reach. QW's brief adds the
  counts per language and cell (eval-protocol §10.1) and the gold form (anchors in bundle spec
  §3.5's forms at `EVAL_COMMIT`, D8, named by E1's rule). AN1 and AN2 get `QW-nogold.jsonl` (S43a)
  inside their own role root.
- **`cpg/eval/kit-manifest.json`** (committed) records the SHA-256 of every kit file and brief, plus
  the exclusion list. The P1 gate's system manifest uses **the same exclusions**
  (`cpg/eval/eval-manifest.yaml`), so the census scope equals the evaluated scope — including the
  prompt-bearing Markdown glob list, which P1's prompt extractor must also use (eval-protocol Q3; §14).

### 6.3 Isolation: what holds by construction, and what is verified

**The launch command.** S53 builds it; §6.3 is the single source of the argv.

```
# one function, build_argv(attempt), used by BOTH start and resume
BIN="$HOME/.local/share/ccg-eval/bin/claude-2.1.312"      # pinned launcher-owned copy
cd ~/.local/share/ccg-eval/<E>/roles/<role>-<stratum> &&
env -i HOME="$HOME" PATH=/usr/bin:/bin LANG=C.UTF-8 \
timeout --kill-after=60 <role timeout> \
"$BIN" -p --restricted --strict-mcp-config --disable-slash-commands \
  --tools Read,Grep,Glob,Write,Edit \
  --allowedTools Read Grep Glob Write Edit \
  --disallowedTools Bash WebFetch WebSearch Agent \
  --settings  <repo>/cpg/eval/headless/settings.json \
  --agents    <repo>/cpg/eval/headless/agents.json --agent <ccg-…> \
  --model <model> --permission-mode dontAsk --permission-prompts none \
  --output-format stream-json --verbose \
  <SESSION>  "<PROMPT>"  > ~/.local/share/ccg-eval/<E>/runs/<uuid>.<n>.stream.jsonl
#  start (attempt n=1):  <SESSION> = --session-id <uuid>   <PROMPT> = contents of BRIEF.md
#  resume (n=2,3,…):     <SESSION> = --resume <uuid>       <PROMPT> = the fixed continuation prompt
```

**Why each part is there:**
- **The binary is invoked by absolute path.** Under `env -i … PATH=/usr/bin:/bin`, the bare name
  `claude` does not resolve (exit 127). `~/.local/bin/claude` is a symlink that the updater moves:
  it pointed at 2.1.312 when §6.3's flags were read, and **now points at 2.1.314** (review Pass 3,
  P3-5). The pinned build is **2.1.312**, which is still in `~/.local/share/claude/versions/` (§2
  D20). S53's `prepare-bin`, dispatched first, copies that version file (a single ELF, about 290 MB)
  to `~/.local/share/ccg-eval/bin/claude-2.1.312` and records its SHA-256. The native updater has
  written four versions in seven days despite `"autoUpdates": false` and prunes old ones, so the
  plan never references the versions directory after `prepare-bin`; an update never touches the
  copy (K-P0-12).
- **`--tools Read,Grep,Glob,Write,Edit`** limits which built-in tools *exist*. `--allowedTools` only
  pre-approves them, which `--permission-mode dontAsk` needs, so both are passed;
  `--disallowedTools` is a redundant second line.
- **`--disable-slash-commands`** disables all skills and commands. `~/.claude/skills/` holds seven
  entries: six symlinks into the repo (`agent-kb-retrieval`, `agent-maintenance`,
  `agent-standards`, `cpg-analysis`, `joern-cpg`, `python-web-quirks`) and a `synced/` directory. A
  `Skill` invocation would read repo content around the file-tool confinement.
- **One argv builder for `start` and `resume`.** Every isolation flag is passed on every attempt;
  the attempts differ **only** in `<SESSION>`, `<PROMPT>` and the stream-log name `<n>`.
- **Per-attempt stream logs.** `<uuid>.<n>.stream.jsonl` is a new file per attempt; an earlier log
  is never truncated, and the audit reads all attempts.
- **`--settings` and `--agents`** are read at launch; their SHA-256s are recorded per attempt,
  because they live in the repo, which builders edit. A changed hash between attempts stops the
  resume.
- **The role timeout** is set in S53: proposed 4 h for KA and KB, 2 h for QW, 1 h for AN.
- **`--verbose`** with `stream-json` under `-p` is settled by S55's capture.
- **`--safe-mode` is rejected:** it disables custom agents and hooks (S52's path guard).

**What the session cannot reach, by how it is built:**

| Channel | Why it is closed | Source |
|---|---|---|
| The repo working directory, and the repo's `CLAUDE.md`/`AGENTS.md` | The cwd is the role root, outside the repo. No ancestor of `~/.local/share/ccg-eval/` holds a `CLAUDE.md` or `AGENTS.md` (probed: `/`, `~`, `~/.local`, `~/.local/share`; `~/.claude/CLAUDE.md` is **0 bytes**) | probe, 2026-10-29 |
| `cypher` and `ws:agent-team` MCP | Both are defined only in the repo's project-level `.mcp.json`. `~/.claude.json` has **no user-scope `mcpServers`**, and none on any project entry. `--strict-mcp-config` with no `--mcp-config` loads no MCP server | probe + help |
| Learning capture into the KB | No KB server is loaded. The `ccg-*` definitions (S52) contain no learning-capture, KB or CPG instructions | S52 |
| Files outside the role root: the repo, sibling roles, `~/.claude/`, `/tmp`, `runs/`, `work/` | `--restricted` "confines the file tools to the working directories (`--add-dir` included)"; no `--add-dir` is passed. S52's `PreToolUse` path guard denies any file-tool path outside the role root, or a write outside `out/` | help + S52 |
| Command execution and the network | `--restricted` removes Bash and the other code-running tools, plus WebFetch; also listed in `--disallowedTools`, as is WebSearch | help |
| Skills and slash commands, incl. the six repo-symlinked user skills | `--disable-slash-commands` | help |
| Any built-in tool beyond the five | `--tools Read,Grep,Glob,Write,Edit`; the init check requires equality | help |
| Spawning sub-agents | `Agent` is disallowed | flags |
| User, project and local settings | `--restricted` "ignores user, project and local settings files"; only `--settings` applies | help |
| A shared scratch space | Each headless run is its own session; no Bash, and file tools cannot reach `/tmp` | construction |
| Auto-memory of the repo project | Memory is keyed by project directory; a role root is a new project, so the repo's `MEMORY.md` is not loaded | inferred; *verify (S54)* |
| Repo-environment variables | `env -i` with a minimal pass-list | construction |

`--bare` is not usable: under it Anthropic auth is "strictly `ANTHROPIC_API_KEY` or
`apiKeyHelper`", and this account uses OAuth plan usage (`~/.claude/.credentials.json`).

**What still has to be verified,** in S54 and then on every run from the stream's `system/init`
event:
1. **User skills:** the init event lists **no** skills and no slash commands.
2. **User agents:** `~/.claude/agents/` holds symlinks into the repo's `claude/` folders, plus
   `prontuarios`. `Agent` is not in `--tools`, so at most a catalog description could leak, and with
   `Agent` absent even that goes. Check the init event's agent list, as a context-leak check.
3. **Plugins:** the init event's plugin list is empty.
4. **Auto-memory:** no memory content appears, and the session cannot write a memory file outside
   its role root.
5. **OAuth works under `--restricted`**, and plan usage is charged; settled by S55 before S48 and
   S54 rely on it, and re-confirmed by the init event and S54.
6. **The confinement holds** against the planted reads below.

**The planted-breach rehearsal (S54, before S41).** `teco` launches one throwaway writer, role
`rehearsal`, whose brief instructs it to attempt each of the following and report what happened:
read `/home/mauricio/prg/graphmind-ai-lab/AGENTS.md` by absolute path; `Grep` with a relative
`../../..` path; read a sibling role's `out/` (a dummy role root with a dummy output); read
`~/.claude/history.jsonl` and `~/.claude/projects/`; read a file under `/tmp`; write outside `out/`;
call `mcp__cypher__query`, `mcp__falkor-chat-agent-team__search_documents` and `ingest_document`;
run Bash, `WebFetch` and `Agent`; read `kit-src/README.md`, a legitimate read, as the control.

**Kill-and-resume case.** A second rehearsal session does legitimate reads, appending to its
output, and `timeout` kills it mid-run. Then: GC audits the partial session; `teco` runs
`run-writer.sh resume`; the resumed attempt makes one more legitimate read and one planted
out-of-root read. **Pass when** both attempts' logs exist under distinct names, the resumed
attempt's init event shows the full isolation flag set, the audit reports no false violation
across the two attempts, and it flags the planted read in attempt 2.

**Pass when:** every illegitimate attempt is refused; the audit (S48) flags every attempt, refused
or not; the control read succeeds and is **not** flagged; items 1–5 are confirmed from the init
event; the kill-and-resume case passes. **Any failure stops the chain**; `devops` and `cobb` fix it
and S54 re-runs. No proceeding on a partial pass.

### 6.4 Where outputs land, and who computes what

1. Each writer's output: `roles/<role>-<stratum>/out/<role>.jsonl`. After the audit passes, GC
   copies it to `work/keys/` (KA, KB) or `work/questions/` (QW, AN).
2. **S43a:** GC produces `QW-nogold.jsonl` with `strip_gold.py` (eval-protocol §13.2) and places it
   in the AN1 and AN2 role roots.
3. **S45:** GC runs `audit.py`, `match_keys.py` and `capture.py` to eval-protocol §6, §7.1–§7.3 and
   §13.3–§13.5.
   - **Audit first** (§6.5). A violation voids that session's output; a fresh session redoes it.
   - **Locator validation** (`match_keys.py validate`, eval-protocol §7.1) before matching; scope
     (`match_keys.py scope`, V4) per item after it.
   - **Mechanical matching** on normalized locators (eval-protocol §6.2).
   - **Near-misses.** GC writes unresolved pairs to `work/nearmiss.jsonl`; `data-scientist` reads
     them in place and returns verdicts by **opaque pair id only**, so no item content passes
     through `teco`'s context; GC applies and logs them (eval-protocol §6.3).
   - **Estimation,** per stratum and pooled: n₁, n₂, m, N̂ by eval-protocol §6.4 (Chapman; with
     three or more lists, strategy-collapsed Chapman alongside Chao's lower bound, the larger
     reported), and completeness = |union| ÷ N̂, reported as **a lower bound on incompleteness**
     (the writers share LLM blind spots). `data-scientist` checks.
   - **Trigger** (eval-protocol §6.5, E5): completeness below 0.95 in a stratum → `teco` launches
     **one** more provider-first session for that stratum (at most one), and S45 re-runs.
   - **Spot-check sample** (`match_keys.py sample`, eval-protocol §7.3): 40 union items plus 8
     Portuguese gold items, for S46.
4. **S46 (stakeholder):** marks each sampled item correct or incorrect; GC applies corrections in
   S47 and logs them (eval-protocol §7.4).
5. **S47:** GC runs `split_manifest.py` (eval-protocol §13.6).
   - **Contact rule first** (eval-protocol §8, E6): every touched anchor (S14, S30, P0 golden tests,
     S56's flow methods) goes to dev with all its items, cluster-wide (a unit and its probe are one
     cluster); it overrides every held-out minimum.
   - **Split** by anchor cluster, seeded hash order, 0.70 held-out share per stratum, then step 3b's
     served-anchor rule (a unit is held out only while every anchor it serves is), the step 4
     top-up with its eligibility guard, and the SQ-P0-1 trigger of eval-protocol §9.1 (E7). Counting
     uses E11 (flow counts toward 75 only across a unit or repo boundary): `H` and `H_all` are both
     reported. Up to 35 counting items go to the **reserve** for re-gates (E7, E12).
   - **Ambiguity** per eval-protocol §10.3 (both annotators must agree).
   - **Write and manifest.** Held-out to `~/.local/share/ccg-eval/<E>/heldout/` (D11); its SHA-256
     and counts to `cpg/eval/heldout-manifest.json` (committed, eval-protocol §9.3); dev to
     `cpg/eval/dev/*.jsonl`.
   - **Report.** `docs/test-reports/connected-code-graph-p0-ground-truth-report.md`: counts,
     n₁/n₂/m, N̂, completeness, adjudication and spot-check tallies, excluded scope, split sizes,
     hashes, the writers' stated hop-count distribution (a prior only), **the audit summary**, and
     **the run count and usage** (§6.6).
   - **Leak check** (`split_manifest.py --check-report`, eval-protocol §9.4): no held-out item in the
     report.
   - **Method check** by `data-scientist`.

   Afterwards the raw keys, `work/`, the role roots, the `runs/` logs and the transcripts stay as
   plaintext under `~/.local/share/ccg-eval/<E>/`, beside `heldout/`, so the audit can be re-run;
   hidden by the same convention (D11), their hashes added to the manifest.

### 6.5 The audit, and lifetime

**The audit** (`cpg/eval/tools/audit.py`, S48; eval-protocol §13.3 points here as the
specification) runs on every writer session, each resume included, and on the rehearsal.

**Sources.**
- **Primary:** the launcher's per-attempt logs `runs/<session-id>.<n>.stream.jsonl`, for every
  attempt in the run index, written outside the role root.
- **Secondary:** the session's own transcript, found by **session id**
  (`~/.claude/projects/*/<session-id>.jsonl`), with no assumption about the directory name.
- **Cross-check over the union of attempts.** Every `tool_use` id in the transcript must appear in
  some attempt's stream log and vice versa; a mismatch, a missing transcript or a missing attempt
  log is a violation. **One exception:** if a call's only stream-side record is the line `timeout`
  cut off at the end of an attempt's log, the call is checked from the transcript alone and
  reported as "stream-truncated". S48's two-attempt fixture proves both rules.

**Checks.**
1. **Init event, in every attempt:** MCP servers empty; the tool set **equal** to `--tools`; the
   plugin list empty; no skills and no slash commands; the agent list as S54 accepted it; the model
   and CLI version equal the run index's; the hashes of `settings.json` and `agents.json` unchanged
   across attempts.
2. **Every `tool_use`:** the tool name is in the allowed set; every path-bearing input (`file_path`,
   `path`, a `Glob` pattern's base, a `Grep` `path`) is **resolved** against the session's cwd, with
   `..` and symlinks normalized, and must lie inside the role root; writes inside `out/`.
3. **Every refused `tool_result`** is reported as an attempted breach. A refused attempt revealed
   nothing, so it is counted but does **not** void the output. Any **successful** call that fails
   check 1 or 2 voids it.
4. **The run index:** every session id in it was audited, and every audited session is in it.

The audit result, as counts and violation types only, goes in the ground-truth report.

**GC and `data-scientist`** run in-session; their subagent transcripts are checked by tool name
only, for `ingest_document`, `WebFetch` and `WebSearch`.

**Who reads held-out:** a P1+ gate run (`qa-engineer`), GC, the `data-scientist` method owner, and
the stakeholder — a convention, not an enforced control (D11). Builder briefs do not mention the
path at all. At each gate, GC re-hashes the held-out files against the manifest before use; a
mismatch stops the gate. That detects modification, not reading.

**After a gate miss** (AC-5's fail procedure): **eval-protocol §12.1, decided by the stakeholder on
2026-10-29** — the held-out anchors that passed stay, the missed anchors move to dev, and fresh blind
items, written under this protocol at the same or a newly pinned `EVAL_COMMIT`, top the list up.
A re-gate pass counts only with the **fresh-item floor** (E12, stakeholder 2026-10-29: ≥ 35
counting items on never-exposed anchors, all hits), drawn first from S47's reserve; every re-gate
reports fresh-only recall and the cumulative first-exposure record.

### 6.6 Usage cost (stakeholder-accepted)

These runs spend plan usage outside the coordinating session. Session counts follow eval-protocol
§14 (five strata, E2) and its Q6:

| Runs | Count |
|---|---|
| Capture (S55) | 1 (trivial) |
| Rehearsal (S54): breach session + kill-and-resume session (2 attempts) | 2, plus 1–2 more if a fix is needed |
| KA, 5 strata | 5 |
| KB, 5 strata | 5 |
| QW | 1 |
| AN1, AN2 | 2 |
| **Base total** | **16–18** |
| Resumes after kills (KA/KB are the long ones) | budget +2 |
| Extra census passes if completeness is short (at most one per stratum, E5) | 0–5 |
| Redo after an audit violation | 0, or more if one occurs |
| **Expected range** | **about 18–25 sessions** (v1.3: 16–22; the rise is S06's E2 and E5) |

KA and KB are census-length sessions; QW, AN and the rehearsal are short. S53's run index records
the reported cost per run, and the ground-truth report totals it. `teco` must not run more than two
writer sessions at once. The +2–3 sessions are within the stakeholder's acceptance in kind
(eval-protocol Q6); `teco` informs the stakeholder of the new range when S41/S42 start.

## 7. Environment

Probed for v1.3 on 2026-10-29, updated with what S01, S02 and S04 established. *Not checked* is
not claimed.

| Need | State now | Gap → provisioning (`devops` unless noted) | Needed by |
|---|---|---|---|
| Joern v4.0.579 | ✅ host `~/joern/joern-cli`; ✅ in `ccg-toolchain:d3184effb566` by checksum (S02) | — | S11, S25–S28, S39 |
| JDK 21 | ✅ host and image | — | same |
| Frontends for the 7 FR-1 languages | ✅ all present; launcher aliases and refusals per egress spec §7.3.1 | — | S25–S28 |
| Toolchains for fetch-on runs and corpus builds (Maven/Gradle, Go, .NET SDK, Kotlin, CMake, `bear`) | ✅ in the image (S02); none on the host | — | S25, S27 |
| Truth tools (ML §5.2, §9) | ❌ not yet in any image | **S24**, `truth` target (§4.1) | S25, S27 |
| py-tree-sitter + ten grammars | ✅ declared in `cpg/pyproject.toml` `spike` (S09) | install `.[spike]` in the venv; in the `truth` image if the harness runs there | S25, S48 |
| No-network sandbox | ✅ Docker 29.7.2, Compose v5.4.0. The namespace backend refuses on this host (egress §7.1.1: `WSLInterop-late` enabled; `/dev/vsock` present) | S08 | S08, S11, S25, S28, S39 |
| Dedicated FalkorDB | ✅ `ccg-falkordb` up on `127.0.0.1:6390` (S01) | — | S13, S17, S20–S23 |
| Python 3.12 + venv for `cpg/` | ✅ 3.12.3; `cpg/.venv` (S09); no `uv` on the host | — | every `CC`/`D` step |
| Redis CLI | ✅ `redis-cli` 7.0.15 | — | S20–S23 |
| Claude Code CLI for the headless writers | ⚠️ `~/.local/bin/claude` → **2.1.314** now; **2.1.312**, whose `--help` §6.3 was verified against, is still in `~/.local/share/claude/versions/` (with 2.1.309, 2.1.310). OAuth at `~/.claude/.credentials.json`; no user-scope MCP servers; `~/.claude/CLAUDE.md` empty | none to install; **S53's `prepare-bin` copies 2.1.312 first, now** (§2 D20, K-P0-12) | S41–S44, S54, S55 |
| `shellcheck` | ❌ not installed (`which shellcheck` exits 1) | not provisioned: S53 is checked by `bash -n` plus its own tests (P3-11) | S53 |
| RAM | ⚠️ WSL VM capped at **6 GB + 4 GB swap, 8 vCPUs** | **Human decision, not `devops`:** leave the cap; §5.1 serialization (K-P0-1) | all |
| Network for corpus fetch, fetch-on and truth runs | assumed available, *not checked* | — | S24, S25 |

**Toolchain per step:** S08 Docker and the image; S09, S10, S31, S32, S34 Python only; S11 the
image and the sandbox; S12 Python; S13, S17 Python, `falkordb-py`, the instance; S20–S23 Python,
`falkordb-py`, `redis-cli`, the instance; S24 Docker plus network; S25 the `truth` and `toolchain`
images, the sandbox, network (fetch, truth and on runs only), the `spike` venv; S26 and S56 only
file reading and grep inside the corpus checkout; S57 Python and the venv (FC), reading only
(`data-scientist`); S27, S28 the sandbox and Python; S39 Python, `git`, the sandbox; S40–S49
`git` and Python; S41–S44, S54, S55 the `claude` CLI, launched by `teco` through S53.

## 8. §6.2 "for P0 to settle" → where each is settled

| §6.2 item | Settled in | Decision rule or method | Recorded in |
|---|---|---|---|
| Resolved-call precision and indexer adoption per language | **S27** | ML §8.1–§8.3 as ratified (§2 D14) | fitness report §a1; proposal §3.2 via S50 |
| Fold vs list of a resolved call's other candidates | **S27** | ML §8.3 (≥ 0.95 in both configurations, n_P ≥ 60) | same |
| `flowAvailable` per language | **S28** (+ S14 for Python and TS) | ML §8.4 | fitness report §a2; proposal §3.2 via S50 |
| BFS vs `algo.SSpaths` | **S20** | §4.5 rule | p0-graph note §2, §6 |
| `MULTI`/`RENAME` rollback | **S21** | probe outcome; default "zero kept, re-assemble" | p0-graph note §3, §6 |
| Whether telemetry can be capped | **S21** | probe outcome + operating rule | p0-graph note §4, §6 |
| `TIMEOUT_MAX` sizing (FR-21's raw-query limit default) | **S22** | §4.5 sizing rule | p0-graph note §5, §6; proposal §3.5 via S50 |
| Verdict-lock ttl | **S22** | §4.5 sizing rule | same; proposal §3.8 via S50 |
| Infrastructure-signature pack initial content | **S30** (+ S32 YAML) | ML §8.3 classes × runtimes | rule-pack spec appendix; `cpg/packs/infra-signatures/` |
| Severity rule table for non-dependency findings | **S30** (+ S32 YAML) | four levels, stated rules | rule-pack spec appendix; `cpg/packs/severity-rules/` |
| `graph-dba`: fold the quirks + V16–V18 into `claude/graph-dba/falkordb-quirks.md` | **S07** (done), new ones in **S23** | dated entries with commands | the quirks file |
| `security-expert` (optional, before P5): HMAC fingerprint and canary review | **not in P0**, a handoff (§15) | — | — |

Other contracts P0 settles: bundle schema v1 and graph model freeze (S15, with S16 and S17);
egress contract (S04, done); evaluation protocol (S06); rule-pack format v1 (S30); answer schema v1
(S33); the P0 load batch ceiling (S23); the `bindValue` advisory (S58).

## 9. Test strategy and gates

### 9.1 Durable code: what is tested at which altitude

- **Default suite:** `cd cpg && .venv/bin/python -m pytest -m "not live"`. No Docker, FalkorDB or
  Joern; green after every CC or D step.
- **Live suite:** `cd cpg && CCG_FALKORDB_PORT=6390 .venv/bin/python -m pytest -m live`.
- **Sandbox suite:** `cpg/sandbox/tests/test_sandbox.sh` (egress spec §10).

Contract code (`tdd-engineer`) is written test-first, in this order.

| Module | Unit tests (red → green order) | Contract or integration |
|---|---|---|
| `ccg/ids.py` (S09, accepted) | bundle spec §3.6's 22 golden vectors and the S09 gate's cases | — |
| `ccg/bundle.py` + `bundle-v1` (S10, accepted; follow-up round) | Accepted suite as delivered. **Follow-up:** `-spec2` §1.6.3's argv coverage probe verbatim (17,280 + 6,480 cells, zero in clear; the 675-cell transplant); AV-1–AV-68 and BV-1–BV-28 parsed from `-spec2`'s text; the SA-6 name probe over all 36 words with the suffix axis; `controlledBy` with `point == cond` rejected; `import`/`launchFact` ids from `ordinal`. | Spec-derived tests read the spec **and** `-spec2`'s replacement sections (`-spec2` §4.1 item 8). |
| `ccg/answer.py` + `answer-v1` (S34) | 1. Canonical bytes identical for shuffled `items`/`groups`. 2. A folded group with a member lacking a chain is rejected. 3. `recipe.algorithm.{id, version}` and `stamp` required. 4. The golden AC-1 answer validates and round-trips. | The golden answer is built from S13's real output. |
| `rule-pack-v1` (S31–S32) | 1. Each paper pack validates. 2. Each negative example in S30 is rejected. | — |
| `fixture-manifest-v1` + `materialize.py` (S39) | 1. Two materializations give identical commit SHAs. 2. Every locator resolves (file exists, line in range, qname found by bundle spec §3.5's rule). 3. Planted ids unique; every S35 row present. | `live`: each fixture repo parses in the sandbox and passes §2 D15's content check. |
| `cpg/sandbox/` (S08, accepted; follow-up round) | egress spec §10 tests 1–10, with `docs/plans/connected-code-graph-egress-spec2.md`: ES2 §1.1's test 10(b) leading-byte probe, §1.3's concurrency test, §1.7's polluted-environment test (with its no-`-p` mutant), A6(b′) in test 1 | per backend; the namespace refusal is itself a test |
| `cpg/infra/toolchain/` (S24) | `ccg-versions`: a failing drop-in plus an `exit 0` drop-in exits non-zero; a passing drop-in exits 0; every truth tool indexes its smoke project | S08 tests 3, 6, 9, 10(c) on the rebuilt `toolchain` image |
| `cpg/eval/tools/audit.py` (S48) | 1. Absolute repo read flagged. 2. Relative `../` read flagged. 3. Symlink escape flagged. 4. Sibling-root read flagged. 5. MCP call and disallowed tool flagged. 6. Write outside `out/` flagged. 7. Legitimate in-root read not flagged. 8. Stream/transcript mismatch and missing transcript flagged. 9. Init event with a non-empty MCP list fails. 10. Two-attempt session: no false violation; a planted attempt-2 read flagged. | S54 runs it on a real session |
| `cpg/eval/tools/export_kit.py` (S48) | the content check excludes a planted `C-1001` file; `--draw-flow-anchors` reproduces a seeded draw on a toy tree | S40 |
| `cpg/eval/tools/{match_keys,capture,split_manifest}.py` (S49) | eval-protocol §13.4–§13.6 cases; `validate` V1/V2 incl. a YAML `config` locator; `sample` reproduces a seeded 40 + 8; `capture.py` matches Appendix A; `--verify` fails on a one-byte change; `--check-report` catches a planted locator; **EP2 §2's K1–K48**; `renormalize-check` stops on a changed id; `normalize` uses the imported `mask_http_segments` (a mutant that copies the loop instead is caught by an import test) | S45, S47 |
| `cpg/ccg/keynorm.py` (S12, P0-11a) | **EP2 §2's K1–K22, K30–K33 and K42–K48 in provider form**; placeholders and empty segments untouched; the screen is `ccg.bundle.screen_fails` itself (mutation spot-check: flip its result, a K-case fails) | imported by S12's `assemble.py` and S49 |
| `cpg/eval/headless/guard-paths.sh` (S52) | 1. Denies paths outside the root, given absolute, relative and symlinked inputs. 2. Denies writes outside `out/`. 3. Allows in-root reads. | S54 |
| `cpg/eval/headless/run-writer.sh` (S53) | The dry run prints §6.3's exact command. `prepare` is idempotent. `resume` refuses an unknown session id. `start` refuses a CLI binary whose hash differs from the recorded one. The script passes `bash -n` (no `shellcheck` on this host, P3-11). | S54 |
| `ccg/store/ddl_v1.cypher` (S17) | — | `live`: applies to an empty key; `db.indexes()` equals the DDL set; constraints `OPERATIONAL` after a one-node load |
| `cpg/infra/ccg-falkordb.sh` (S01, accepted) | — | its own stub-harness test |

**Spike tests** (under `cpg/spikes/*/tests`, outside `testpaths`) are evidence for their results
notes: SL-1–SL-7 for the slice, the lock and replay fault tests for S22. Run and cited, not
maintained after P0.

**Fitness evidence** (S25–S28, S56, S57) is checked at gates rather than by tests: the
reproducibility of the hash sorts from `corpora.yaml`, the brief validity check, the transcript
greps (S26, S56, S57), and `analyst`'s independent checks of the adjudication logs.

### 9.2 Gates

| Kind | Gate | Evidence the gate reads |
|---|---|---|
| C (contract document) | `analyst` review as a `## Pass N`, plus the named domain consult (`graph-dba` for S03, `data-scientist` for S30 and S33) | the document and the sections it implements |
| CC / D (durable code) | `analyst` code review; default suite green; **a mutation spot-check** (the reviewer flips one condition and confirms a test fails) | diff, test run, mutation note |
| M (spike) | `analyst` review of the results note; `data-scientist` for every interval and decision rule (S25–S28); `graph-dba` for density and integrity (S15); `analyst` on S11's `extract.sc` and S25's `measure.sc` and adjudication-input script as code | results note + raw JSON + spike test output |
| GT, fitness (S26, S56, S57) | `data-scientist` checks labels; `teco` greps transcripts; `analyst` confirms S57's no-Joern-artifact rule from both transcripts | labels, adjudication files, transcripts |
| GT, ground truth (S40–S47) | GC produces; `data-scientist` checks method, statistics and adjudications; the stakeholder spot-checks (S46). `analyst` reviews **protocol adherence and counts only** (S47): no reviewer outside §6.5's list sees held-out content | ground-truth report, audit result |
| Doc | `teco` check (S07, S51), or `analyst` (S50) | the edit |
| Advisory (S58) | none; `architect` dispositions each finding | the security-expert report |

**P0 exit** (`teco`) means all of: (1) every row of §3 is gated — **S16 included** (it always runs, P3-8);
(2) every §8 row points to a recorded decision; (3) S15's freeze outcome, with its "frozen,
untested" list, is in the ledger; (4) S58's dispositions are recorded (done, in `-spec2`); (5)
**`docs/plans/connected-code-graph-bundle-spec3.md` exists, is gated, and `Supersedes:` the bundle
spec and `-spec2`** (`-spec2` §0, F-5); (6) §10's documentation list is applied; (7) `reset-p0` has
been run on the dedicated instance.

## 10. Documentation impact

| Document | Change | When | By |
|---|---|---|---|
| `docs/plans/connected-code-graph-p0.md` (v1.3) | `Status: superseded` and `Superseded by:` this document (header metadata only; done in this unit) | now | `architect` |
| `docs/plans/connected-code-graph-coordination.md` | U29 row; future units cite this document's step table. The S15 outcome; agent ids of every ground-truth and fitness instance (§6.1, S26, S56, S57) | throughout | `teco` |
| `docs/BACKLOG.md` C-1001 | **Rewritten in place**, forward-looking only: P0 in progress, pointing to this plan and the ledger; at P0 exit, what remains (P1–P5, the Arduino slot) and that the next step is the P1 plan | P0 start; P0 exit | `teco` |
| `docs/HISTORY.md` | One dated entry per delivered group: (1) environment and sandbox (S01, S02, S08, S24); (2) contracts (S03, S04, S05, S06); (3) slice and freeze outcome (S11–S17); (4) store spikes (S20–S23); (5) frontend fitness (S25–S28, S56, S57); (6) rule-pack and answer contracts (S30–S34); (7) fixtures and ground truth (S35–S55), with counts and no held-out content; (8) the `bindValue` advisory (S58); (9) P0 close (S50) | as each group gates | `teco` |
| `claude/graph-dba/falkordb-quirks.md` | new quirks, incl. the stall if reproducible | S23 | `graph-dba` |
| `docs/plans/connected-code-graph.md` | v3.4 in place, one dated line (S50), incl. BF16 and eval-protocol Q1/Q3; its `Extended by:` gains this document (S50; this unit does not edit the proposal) | S50 | `architect` |
| Root `AGENTS.md` `cpg/` bullet and component-docs row | **One clause at P0 exit** (`cpg/` now holds the `ccg` contracts, schemas, sandbox, fixtures and eval data; `cpg/README.md` in the entry-docs row). Full rewrite at P1 (F10) | S51 | `teco` |
| `cpg/README.md`, `cpg/infra/README.md`, `cpg/infra/toolchain/README.md`, `cpg/sandbox/README.md`, `cpg/eval/headless/README.md`, `cpg/spikes/frontend-fitness/README.md` | new or extended | S09, S01, S02/S24, S08, S53, S25 | implementers |
| New documents (all with the root `AGENTS.md` header block) | `docs/plans/connected-code-graph-{rule-pack-spec,answer-spec,p0-graph}.md`; `docs/test-plans/connected-code-graph-fixtures.md`; `docs/test-reports/connected-code-graph-p0-{frontend-fitness,slice,ground-truth}-report.md`; `docs/reviews/connected-code-graph-bindvalue.md` (S58, done); `docs/plans/connected-code-graph-bundle-spec2.md` and `docs/plans/connected-code-graph-egress-spec2.md` (done, U32, both `Extends:` their base specs, which carry `Extended by:`); **`docs/plans/connected-code-graph-bundle-spec3.md`** (S16 or before P0 exit, `Supersedes:` the bundle spec and `-spec2`, whose headers then gain `Superseded by:`). Names are as v1.3 fixed them; the successor does not rename them | per step | per step |
| `Extended by:` pointers | Each creating step adds the pointer to its parent: proposal ← rule-pack and answer specs (S30, S33); graph note ← p0-graph note (S20); ML note ← eval-protocol note (S06). The bundle, egress and p0-ml pointers are already in place | creating step | creating step's owner |
| `docs/reviews/connected-code-graph-p0.md` | this document's gate is its next `## Pass N` | gate | `analyst` |
| Archive flips | At P0 exit the P0-only documents (this plan, v1.3 already `superseded`, the p0-ml note, the three P0 reports) freeze; `teco` flips them at the milestone close it chooses. The specs and the eval protocol stay `active` | M10 close | `teco` |

## 11. Findings (proposal ↔ notes ↔ specs)

F1–F13 are carried from v1.3 §11, with their dispositions updated where an accepted spec or note
has since acted on them. F14–F18 are new. None stops P0.

| # | Finding (short; v1.3 §11 has the full text of F1–F13) | Disposition now |
|---|---|---|
| F1 | Graph note §6.1 item 6 reads as hiding other candidates above a precision threshold; the later decision (P3-1) keeps them, folded | Unchanged: S35 plants a P3-1 case, S33 enumerates folded members, S17 records the correction in the p0-graph note. Bundle spec §12 follows F1. |
| F2 | Graph note §3.4's `symbolId` includes `unitId`; §6.4 removes it | Settled in bundle spec §3.2; S17 records it. |
| F3 | Proposal calls the graph note "v2"; it is v2.1 | S50 corrects it. |
| F4 | Decision S36 says one isolated writer; ML and the proposal use two | Read as consistent (§6). Not a requirement change. |
| F5 | AC-24 gold is defined against a snapshot P0 does not have | §2 D8: bundle spec ids are pure hashes; S06 confirms the locator forms (eval-protocol §2.2). |
| F6 | Capture–recapture on the flow stratum needs a shared anchor population | eval-protocol §3.6: 16 shared seeded anchors, drawn in S40. |
| F7 | ML lets fixture items fill AC-5's held-out; AC-5 says "in the proving repo" | Unchanged: fixture items are dev-only. |
| F8 | Nothing pinned the keys' commit or census scope | §2 D8 and `eval-manifest.yaml`. |
| F9 | (a) needs the projection script, a product of (b) | §2 D3. |
| F10 | Root `AGENTS.md` `cpg/` bullet goes stale at P0 exit | One clause at P0 exit (S51). |
| F11 | `gosrc2cpg --include-indirect-dependencies` was not forbidden | Forbidden; the full list is egress spec §7.3. |
| F12 | ML §4.4/§4.7 still place the interpretation gate at P4 | P0 follows ML §8.3; no action on the ML note. |
| F13 | Fitness truth assumed toolchains and corpora that did not exist | Settled by ML note §3, §5 and S24 (§4.1). |
| **F14** | ML note §11 has `data-scientist` writing adjudication files under `cpg/spikes/`, outside its write guard (`docs/plans/`, `docs/reviews/`) | §2 D13: `data-scientist` decides, a custodian (S57) or `graph-dba` (S27, S28) writes. |
| **F15** | v1.3 §5.2 reserved `cpg/.gitignore` to S09; egress spec §10 has S08 add `.audit/` | §5.2: S09 → S08, one line. S09 committed first, so no serialization was needed. No contract change to either step. |
| **F16** | The toolchain image tag is a content hash, so S24's Dockerfile edit re-tags the image the sandbox runs | §4.1 S24 items 3 and 5: truth tools in a separate `truth` target; the sandbox runner re-pointed after S08, and S08's image-dependent tests re-run. |
| **F17** | `joern-parse` exit 0 does not mean a non-empty CPG (egress §11 R11); bundle spec v1.3 states no minimum-content rule | §2 D15: S11, S25, S28, S39 check content; S10 enforces nothing new. |
| **F18** | Eval-protocol Q2: bundle spec §3.5 lists two locator types; the eval protocol uses four (`config`, `unit` added, both converted by §3.2) | **Resolved** by `-spec2` §1.9 (two rows added to spec §3.5's table). |

## 12. Stakeholder questions and decision points

**Open stakeholder questions:**
- **SQ-P0-1: AC-5 held-out size at P1** (conditional; raised by `teco` at S47 only if the counts
  call for it). This is the text eval-protocol E7, §9.1 step 5 and §12.1's source 4 point to
  (eval-protocol Q8).
  - **Trigger:** after eval-protocol §8's contact rule, §9.1's served-anchor rule and step 4's
    top-up, the **counting** held-out set `H` (E11: flow counts only across a unit or repo
    boundary) is below 75, or a stratum is below 8. If `H` < 75 while `H_all` ≥ 75, it fires as
    usual (eval-protocol Q9). AC-5 needs at least 75 held-out items "in the proving repo". Which
    then?
  - **Already applied, needing no decision:** (d) the `wiring` stratum (E2, adopted up front).
  - **Inside AC-5's text, conditional:** (e) runtime-observed edges (distinct cross-boundary calls
    seen in access logs or HTTP-client instrumentation while the suites run). This **conflicts with
    ML §4.1**, which says oracle edges are reported next to seeded recall and never pooled. So (e)
    is available only if `data-scientist` explicitly revises that rule, with its reason, **and**
    names a collection step. Eval-protocol v1.2 does neither, so today (e) is off the table.
  - **Needing the stakeholder:** (a) top up with planted items in a fixture repo kept outside the
    repo like the held-out set (D11), written by an isolated agent and reported as its own stratum
    (ML §4.1) — **this amends AC-5**, because a fixture repo is not "in the proving repo" (F7);
    (b) gate on fewer than 75 and report the weaker bound, which amends AC-5's size; (c) wait for
    P3's data stratum before AC-5 is gated at all, which moves an AC-5 gate the proposal places at P1.
  - *Recommendation:* (e) only under the revision above; otherwise (a), reported separately and
    never pooled silently with census items, as an explicit AC-5 amendment. (b) weakens the
    never-miss guarantee the stakeholder ranked first; (c) leaves P1 with no AC-5 evidence.
  - **Known pressure:** `wiring`'s eligible supply is at most 10 under the served-anchor rule, or
    7 if S30 contacts `cypher-mcp` (eval-protocol R9), which is why S30's walkthrough is placed on
    `falkor-chat/server` (§3 S30).
  - **No estimate of whether it will be needed** (restored from v1.3 §12; eval-protocol G5 cites
    it). v1's "55 files" figure was stale: 129 tracked files now name `mcp__cypher__query`, 120 of
    them Markdown. Documentation snippets are not callers (proposal §3.4), so the file count says
    little about the item count until the census exists.
- **Re-gate fresh-item shortfall** (eval-protocol Q7; conditional, `teco` → stakeholder only if it
  occurs): if the reserve, a same-commit top-up and a newly pinned commit together cannot supply
  E12's 35 fresh-cluster items for a re-gate, which option of eval-protocol §12.1 source 4 applies?
- **Egress Q1 (P1, via `tico`): the host-granular git tunnel** (egress spec §6.1, §11 Q1; review
  §S04 M5). Accept the residual risk of an `R2` tunnel that can reach any repository on the
  remote's host, given §6.1's mitigations, or require TLS interception for `R2` with a broker-held
  CA trusted only by the fetch container? **Open; decided at P1.** P0 builds no broker, so P0 is
  not affected.

**Decided since v1.3** (all by the stakeholder, 2026-10-29; `tico` records them in the
requirements, ledger follow-up):
- **AC-5's "re-run on a new blind list"** — eval-protocol §12.1 step 4's reading (passed anchors
  stay, missed anchors move to dev, fresh items top the list up) (§6.5).
- **The fresh-item floor** for a re-gate pass (E12: ≥ 35, all hits).
- **Flow counts toward AC-5's 75 only across a unit or repo boundary** (E11).
- **FR-15's "never stores or shows a secret's value"** covers the graph and answers, not the local
  bundle files; **internal hostnames in clear are acceptable** (`-spec2` §0 sources).

**Decision points for `teco`** (design level, routed to `architect` + `analyst`):
- **SQ-P0-2: a partial slice pass at S15.** Unchanged: the plan executes the proposal's binary rule;
  a partial freeze would be a proposal §4 amendment decided at that point. *Recommendation: don't.*
- **A single-source language after S24** (§2 D13): whether one of ML §5.2's fallback analysers is
  worth building for it. Raised by `teco` with S24's evidence, only if it occurs.

## 13. Risks

| # | Risk | Effect | Mitigation |
|---|---|---|---|
| K-P0-1 | 6 GB WSL cap | OOM or skewed timings | §5.1 serialization; instance at 1,536 MB; Joern heap explicit; 1,000-file synthetic size; the workplace peak is not a P0 claim |
| K-P0-2 | A JVM stalls inside a no-network namespace | Offline Joern runs hang | Settled for the container backend by S02/S04 (`--hostname localhost`, egress §7.1.2); egress §10 test 3 guards it |
| K-P0-3 | The slice fails an SL criterion | v1 provisional; P1 carries the freeze | S30, S33, S34 proceed against provisional v1 and are re-checked at P1 exit |
| K-P0-4 | Low KA/KB overlap → low completeness | Extra passes; slip | eval-protocol §6.5: at most one extra pass per stratum; reported as a lower bound |
| K-P0-5 | Census short of 75 | AC-5 at P1 underpowered | SQ-P0-1 at S47 |
| K-P0-6 | A reference tool fails S24's smoke test, or is too heavy for the host | That language single-source, wider intervals | ML §5.2 installed-tool fallbacks; §2 D13; `scip-clang` at parallelism 2 |
| K-P0-7 | Held-out hidden by convention only (D11) | A builder read goes undetected | **Accepted with the decision.** No added mechanism |
| K-P0-8 | The headless construction leaks through an unlisted channel, or the stream/transcript format changes | Blindness lost, or audit can't run | S54's rehearsal, per-run init checks; the audit fails closed (§6.5) |
| K-P0-9 | Instance image drift | Quirks stop applying | S01 pins `v4.18.11` |
| K-P0-10 | P0's breadth: about 50 steps, about 60 dispatches incl. the seven-way fitness fan-outs | Long wall-clock | Parallel fan-outs; S25 split into two dispatches; S46 announced when S41/S42 start |
| K-P0-11 | Corpus licences, or a corpus that is secretly a held-back candidate | Contamination | ML §3.3 exclusions and reverse rule, checked at S25's gate |
| K-P0-12 | A Claude Code update lands mid-chain (it has: 2.1.314 arrived 2026-10-29, after the plan's flag read), or a different build changes `--restricted`, `--strict-mcp-config` or the stream-json shape | Census stalls, or the construction or the audit silently weakens | The launcher runs a **launcher-owned copy** of 2.1.312, taken by `prepare-bin` before anything else (§2 D20), so updates never touch the census. **Moving to another build is a deliberate step: a new `prepare-bin`, then S55 and S54 re-run** (restored from v1.3). Config hashes are checked per attempt (§6.3) |
| **K-P0-13** | The `ccg-falkordb` write stall recurs during a P0 load | A spike's timings are meaningless, or the instance is unavailable to other reservations | ≤ 20k-row batches until S23's ceiling (§2 D16); S23 probes it in its own exclusive reservation |
| **K-P0-14** | A fitness labeller or custodian sees Joern output | The truth is anchored to the system under test | ML §6.2's forbidden list; transcript greps (S26, S56); `analyst`'s two-transcript check (S57); separate fresh instances per role |
| **K-P0-15** | S24's rebuild changes the sandbox image under S08's accepted tests | A silent sandbox regression | S08's image-dependent tests re-run in S24's done-condition; `truth` target keeps the sandbox image's tool set unchanged |
| **K-P0-16** | S58's advisory finds a `bindValue` gap after S15 freezes v1 | A post-freeze spec change | **Mitigated:** S58 landed before S15, and its dispositions are in `-spec2` (§2, §5 there), which S15 freezes with the spec |
| **K-P0-17** | Fresh items are scarce for a re-gate (eval-protocol R7) | A re-gate passes on every item yet cannot count as an AC-5 pass | The reserve (S47); a top-up census; a new commit; then the Q7 question in §12 |
| **K-P0-18** | The served-anchor rule leaves `wiring` short (eval-protocol R9) | SQ-P0-1 fires at S47 | S30's walkthrough on `falkor-chat/server` or the fixture (§3 S30); then SQ-P0-1's path |

## 14. S06 hand-offs (eval-protocol v1.2, accepted)

S06 is **accepted at v1.2** (`docs/plans/connected-code-graph-eval-protocol-ml.md`; ledger U26,
no Pass 3). Its §14 table and §15 questions are folded into the rows of §3, §6, §12, §13 and §15.
The item-by-item record, including what changed from v1.0 to v1.2, is the traceability table
§16.1 (source **EP**); this section does not repeat it.

**EP2** (`docs/plans/connected-code-graph-eval-protocol2-ml.md`, `Extends:` EP; F-8, step 8a; in
re-gate) is a consumer source too: its §3 obligations are folded into S06, S12, S40, S45, S47, S49,
§15's P1 gate harness and P1 assembler, and §16.1 (rows EP2-*). This plan cites EP2 by section and
encodes no `keyNorm` step order of its own (§2 D19).

What the fold changed, in one line each:
- **Writer text:** S40 and S52 paste **Appendix B verbatim** and nothing else from the note
  (replacing v1.0's "§3/§4 verbatim").
- **Five strata, ten KA/KB sessions,** about 18–25 headless sessions in all (§6.6).
- **Split and counting:** probe clusters, the served-anchor rule (§9.1 step 3b) and step 4's guard,
  E11's counting `H` beside `H_all`, the reserve and an empty `exposure.jsonl` (S47).
- **Tooling:** S48 gains `--list-units`/`--list-cypher-tools` (tree-sitter from the `spike` extra);
  S49 gains `validate`/`scope`/`sample`, the exported `E`/`h1_symbol_credit`/`source_check_needed`,
  the v1.1–v1.2 oracle cases, and P1-deferrable `regate-list`/`exposure`.
- **S30's server-binding walkthrough** on `falkor-chat/server` or the fixture (Q10).
- **S33** carries resolved anchors' `qname`; **S35** gains AC-25's assertion column; **S56**'s
  methods feed S47's contact set.
- **Stakeholder decisions:** Q5's reading, the fresh-item floor E12, the flow-boundary rule E11 (§12).
- **For P1:** depth bound ≥ 2 (Q1), the prompt glob list (Q3), the gate harness's floor, exposure
  record and four-field pre-registration comparison, and the `coreTreeHash` `*.md` condition (§15).
- **Also for P1/S50:** the answer spec's carry-over items C1-C9 (`docs/plans/connected-code-graph-answer-spec.md`, "Carry-over to P1/S50"); the EP §10.4/§5.2 amendments they name (`data-scientist`, before S47/S49); the S22/S23 ACL findings for proposal §3.5 (`docs/plans/connected-code-graph-p0-graph.md` §6); the gate harness and AC-16 pin-and-replay code (EP v1.3 §5.2.1, §10.6, `docs/plans/connected-code-graph-eval-protocol-ml.md`), and S47 depends on EP v1.3 accepted (`CAP` is pre-registered with the held-out ambiguity set).

## 15. Ready to implement

**Plan:** `docs/plans/connected-code-graph-p0-2.md` (this document; supersedes v1.3). Dispatch
from §3; §5 is the sequencing contract, §6 the isolation contract, §4.6 the fitness contract.

**Digest:**
- **Accepted:** S01–S10, S58; `-spec2` v1.3 and ES2 v1.2. **In flight:** S08's and S10's
  follow-up rounds. **P0 exit additionally needs `-spec3`** (§9.2).
- **Dispatchable now:** S24 (`devops`: truth tools in a `truth` target, smoke projects, the
  `ccg-versions` subshell fix, S08 tests re-run on the new tag); S20 → S21 (`graph-dba`,
  exclusive, ≤ 20k-row batches); S35; S52 → S53 → S55.
- **Next:** S11 after this plan's gate and the S08/S10 follow-up rounds (content check per §2 D15,
  `-spec2`'s emitter items incl. G-4, `@AUDIT_ID@`).
- **Four chains:** the slice (S11 → S15 freeze gate, with "frozen, untested" listed); the store
  spikes (S20–S23, incl. the stall probe and batch ceiling); frontend fitness (S24 → S25 →
  {S26 ∥ S56} → S57 → S27 → S28, decisions by ML §8 as ratified in §2 D14); fixtures and ground
  truth (S35–S55, five strata, 18–25 headless sessions, at most two at a time).
- **After S15:** graph model v1 and DDL (S17, incl. the `ownedRegions` anchor rule and the
  load-under-`TIMEOUT_MAX` rule), the rule-pack and answer contracts and their schema code.
- **To close:** S50 folds the measured values, BF16, and eval-protocol Q1/Q3 into the proposal
  (v3.4); S51 updates the docs.

**Named handoffs to P1 (not built in P0):** the production extractors (ported from the slice under
P1's own tests, with §2 D15's content check); the pack runtime; the linker; the store loader
(creating S22's ACL users, applying `ddl_v1.cypher`, obeying S23's batch ceiling and S22's
load-under-cap rule); the id-hash fuzz test; the broker process and AC-8's logging proxy per S04,
**after the stakeholder decides egress Q1**; the answer engine per S20 and S33, **with a shipped
default depth bound ≥ 2** (eval-protocol Q1); the **prompt extractor classifying by
`eval-manifest.yaml`'s glob list** (eval-protocol Q3); the AC-5 gate harness (verifies the held-out
manifest, builds at `EVAL_COMMIT`, applies eval-protocol §11's noise sampling and §12.1's fail
procedure with E12's fresh-item floor, routing by S(u) and the exposure record; computes §12.3's
four-field pre-registration comparison incl. `coreTreeHash`; imports H1's span test and §5.4-under-`E`
from `match_keys.py` rather than re-implementing them; builds S49's `regate-list` and `exposure`
verbs if P0 did not; **reads the stored normalized gold and ids, runs `match_keys.py
renormalize-check` before scoring and stops the gate on any difference, flags a contract anchor
whose `cpId` is absent from the gate graph as an `anchor-not-in-graph` miss, and imports rather than
reimplements** (EP2 §1.5, §3 P1 gate harness row); **ratifies the possible-precision thresholds after
the first P1 measurement** (ML §4.2, the "Thresholds" paragraph: the floor cannot be fixed before P1
measures; restored from v1.3, P3-6); and reports the **runtime oracle** beside seeded recall, never
pooled, each oracle miss feeding the next held-out set (ML §4.1; eval-protocol §15.1 R1)). **The P1
assembler imports `mask_http_segments` from `cpg/ccg/keynorm.py`**, the function S12's P0-11a
writes, and never reimplements step 8a (EP2 §3, P1 assembler row; §2 D19). **P1 condition (eval-protocol §16, M5 nit):** if P1 keeps interpreter prompts
as `.md` under `cpg/`, narrow `coreTreeHash`'s `*.md` exclusion before the first pre-registration; the AC-24 smoke harness (eval-protocol §10.5); `teco` v1 integration (a `cobb` unit);
the full root `AGENTS.md` `cpg/` rewrite; porting or deleting `cpg/spikes/`; re-reading `flowAvailable`
if P1's design makes call-level reach not a superset of flow reach (ML §8.4). **From `-spec2` §3
(owners there):** F-10's P1 writer obligations as one writer done-condition (W2 with `default`, W3,
W4, the writer side of SA-1, §1.6.1's argv rule, G-4's self-edge drop, the per-entry HMAC of
§1.7.5); F-6 (SA-3 role allow-list and per-format `keyPath` derivation, before P1 relies on
`bindValue`); F-7 (`bindParts`, an additive minor field); F-9 (`Cond.code` literal masking in the
P1 normalizer, **before P1's first build of a non-proving repo**); F-12's cycle-safe traversal in the
P1 BFS; F-1 (`devops`: bundle-at-rest file mode, location and retention, and whether any LLM-backed
tool reads a bundle directly, **before P1's first non-proving-repo build**; until then P0 bundles live
only in the gitignored work area).

**Measurements deferred (each with its reason, unchanged from v1.3):** `LogTemplate` volume → P5;
`RAN` edge volume → P3; the ~18 GB peak → re-estimated at P1, measured at P2; typical
evidence-chain length (for ML §3.6's 0.98 revisit) → P1's first AC-5 gate, S47's writer-stated
distribution being a prior only; `Link` counts under runtime-built URLs at scale → P1. **Later than
P1:** the `security-expert` HMAC and canary review before P5; the Arduino slot after P2.

**Follow-ups owned by `architect`:** BF16 into proposal §3.8 (S50); `-spec3` (S16, before S15;
required); the proposal's `Extended by:` pointer to this document (S50); `-spec2` F-3/F-4/F-12
and G-1's rendering in S33. (Eval-protocol Q2 and S58's dispositions are done, in `-spec2`.)

**Follow-ups owned by others:** `data-scientist` — `-spec2` F-8 is **delivered by
`docs/plans/connected-code-graph-eval-protocol2-ml.md`** (EP2, in re-gate) and **encoded at S49**
(with S40, S45, S47 and P0-11a as their rows state); EP2's Q2 (EP §2.2 citing BS2 §1.8 once `-spec3`
consolidates) is the note's own; and eval-protocol G5's "plan §12" citation now resolves to §12's
restored paragraph (n-b). `cobb` — `-spec2` F-3's prompt fencing in S52. The human/`cobb` — whether
root `AGENTS.md` rule 5 should gain an "`Extends:` but replaces named clauses" form (`-spec2` §0;
ledger 2026-10-29). `tico` — record in `docs/requirements/connected-code-graph.md` all four
stakeholder decisions of §12: Q5, E11, E12, **and FR-15's scope plus internal hostnames in clear**
(`-spec2` §0; P3-9).

## 16. Disposition table (absorbed items)

Every item this successor absorbed, its source, and where it landed. "ML" = `docs/plans/connected-code-graph-p0-ml.md` v1.3; "BS" = `docs/plans/connected-code-graph-bundle-spec.md` v1.3; "ES" = `docs/plans/connected-code-graph-egress-spec.md` v1.3; "EP" = `docs/plans/connected-code-graph-eval-protocol-ml.md` v1.0; "Rev" = `docs/reviews/connected-code-graph-p0-impl.md`; "Ledger" = `docs/plans/connected-code-graph-coordination.md`, Notes of 2026-10-29.

| # | Item | Source | Landed in |
|---|---|---|---|
| A1 | S24 amended (N0): primary tools pinned, smoke projects, installed-tool-only fallback, single-source | ML §11 N0, §5.2, §9 | §3 S24; §4.1 S24 items 1–2, 6 |
| A2 | S25 amended: reference truth run, C5 before any Joern run (N1) | ML §11 N1, §3.2, §5.1 | §3 S25; §4.6 S25 item 3 |
| A3 | S25 amended: frame, draws, briefs, flow draw, matcher table (N2) | ML §11 N2, §4, §6.4, §6.5, §7.1 | §4.6 S25 items 2, 4 |
| A4 | Adjudication-input script written in S25, statically checked by `analyst`, not run by `graph-dba` | ML §11 N4, §12 Pass 2 P2-4 | §4.6 S25 items 5, 7 |
| A5 | New step: golden flow labelling (N3) | ML §11 N3, §6.4 | **S56**; §4.6 S56 |
| A6 | New step: pre-Joern adjudication (N4) | ML §11 N4, §5.1 step 3 | **S57**; §4.6 S57; owner per §2 D13 |
| A7 | S27 amended: provisional → miss log → `final2` → reported metrics (N5) | ML §11 N5, §5.1, §7.2 | §3 S27; §4.6 S27 |
| A8 | S28 amended: flow misses and spurious adjudication, overturns before reported recall (N7) | ML §11 N7, §8.4 | §3 S28; §4.6 S28 |
| A9 | S09 brief amended: tree-sitter in `spike` (N6) | ML §11 N6 | already in S09 (accepted); §1.1, §7 |
| A10 | The reference run (named separately in the brief) | ML §11 N1 | = A2 |
| A11 | The flow draw (named separately in the brief) | ML §11 N2, §6.4 | = A3 (`sample/<lang>-flow.jsonl`) |
| A12 | Golden flow sets (named separately in the brief) | ML §6.4, §11 N3 | = A5 |
| A13 | S24↔S25 ordering resolved | Rev §S05 M3; ML §11 "Ordering" | §2 D13 (rejected alternative); §3 deps and structure (S24 → S25 → {S26 ∥ S56} → S57 → S27 → S28) |
| A14 | Rule authority (precision rule; ratify or reject §8.3 proposal) | Rev §S05 M2; ML §8, §8.3, §10 | §2 D14 (ML §8 ratified; upper-bound proposal rejected); §8 |
| A15 | Missing steps | Rev §S05 M5 | A2, A3, A5, A6 |
| A16 | Fallback analysers that would be new code | ML §5.2, §10 | §2 D13 (not built; single-source); §12 decision point |
| A17 | Off run must not see dependency caches | ML §9, R8 | §4.6 S25 item 6 (clean checkout); S08 test 5's mount set covers the mount side |
| A18 | ML §6.4 / §12 (who writes golden flow sets; adjudication placement) | ML §6.4, §10, §12 | A5, A6, §2 D13 |
| A19 | `data-scientist` write guard vs N4/N5/N7 owners | `claude/data-scientist/data-scientist.md` | §2 D13; F14 |
| B1 | BF3: `use`/`def` supersede v1.3 §4.2's `{from, to}` | BS §12 BF3, §5.10; Ledger | §4.2 S03; §4.4 S11 |
| B2 | BF16: re-binding on earlier-sibling insertion → proposal §3.8 fourth orphan case | BS §12 BF16; Ledger | §3 S50; §15 follow-ups (proposal not edited in this unit) |
| B3 | S15's record names [P-later] items "frozen, untested" | BS §1, §10; Ledger | §2 D5; §3 S15; §4.4 S15; §9.2 exit item 3 |
| B4 | Other BS downstream encoders (BF1, BF4–BF15, §11 table, §9) | BS §10–§12; Rev §S03, §S03 graph-dba check | §4.2 S03; §3 S10, S11, S12, S17; §5.3 |
| B5 | [P-slice] items exercised and reported | BS §10 | §3 S14 |
| B6 | Go interface specs and vendored code (S05 Pass 3 Q1/Q2) | Rev §S05 Pass 3; BS §13.2 | settled by BS v1.2 (accepted); nothing to place |
| C1 | ES §7.1.1 controls over plan D6 and §4.1 test 1 `WSLInterop` path | ES §7.1.1, §11 R1; Ledger | §2 D6; §4.1 S08; §1.3; §7 |
| C2 | S08's test list is ES §10 | ES §10 | §3 S08; §4.1; §9.1 |
| C3 | Erratum P3-m1: strip `[ \t\r\n\f\v]` before `//>`; H15 wrong; fixtures in test 10(b) | Rev §S04 Pass 3 P3-m1; Ledger | v1.0: §4.1 erratum E1; **v1.1: superseded by ES2 §1.1** (§2 D18, §4.1) |
| C4 | Erratum P3-n1: bind `/scripts` as a directory; test 5 lists `/scripts` | Rev §S04 Pass 3 P3-n1; Ledger | v1.0: §4.1 erratum E2; **v1.1: restated as ES2 §1.2** (§2 D18, §4.1) |
| C5 | Open Q1 / M5: host-granular git tunnel vs TLS interception, stakeholder at P1 | ES §11 Q1, §6.1; Rev §S04 M5 | §12 (open); §15 P1 handoffs |
| C6 | R10: `joern --script` closed form; script reviewed as code | ES §7.3.1, §11 R10 | §4.4 S11; §4.6 S25 item 6; §9.2 |
| C7 | R7: three `--help`-only classifications, evidence from fetch-off runs | ES §11 R7 | §4.6 S27 |
| C8 | `.audit/` in `cpg/.gitignore` by S08 | ES §10 | §5.2; F15 |
| C9 | R2, R4, R5, R8 (S08-internal or P2) | ES §11 | not placed: S08's own gate or P2; no plan change (v1.1: R2 resolved by ES2 §1.3) |
| D1 | S24: stages `FROM apt-base`, install only via `apt-snapshot-install` | Ledger (U21 accepted); Rev §S02 Pass 2 | §4.1 S24 item 4 |
| D2 | S24: drop-ins run in a subshell so `exit` cannot override the count | Rev §S02 Pass 2 p2-m1; Ledger | §4.1 S24 item 4; §9.1 |
| D3 | S24: failing drop-in + `exit 0` → whole check exits non-zero | Rev §S02 Pass 2 p2-m1 acceptance | §4.1 S24 item 4; §9.1 |
| D4 | S10/S11 (R11): CPG non-empty content check | ES §11 R11, §9 H12; Ledger; teco relay (S10 brief) | §2 D15; §3 S10 (no rule), S11, S39; §4.4; §4.6 S25, S28 |
| D5 | S17: explicit `timeout` under `TIMEOUT_MAX` | Rev §S03 graph-dba "Cross-check from S07"; Ledger | §2 D16; §3 S17, S13, S22; §4.5 |
| D6 | S17: `ownedRegions` anchor rule | BS §13.3 P3-1, §9.1; Rev §S03 Pass 3 P3-1 | §3 S17 |
| D7 | S23: `PING` stall on a 100k-edge write; ≤ 20k rows per batch until probed | Rev §S03 graph-dba "Incident"; Ledger | §2 D16; §3 S23 (stall probe, R-C4); §5.1; K-P0-13 |
| D8 | Before P1: `security-expert` advisory on `bindValue` | BS §12 BF8, §5.17; Rev §S03 F6; Ledger | **S58**; §2 D17; §9.2 exit item 4 |
| D9 | S22: V18, nonzero `TIMEOUT_MAX` kills writes unless explicit `timeout` | Ledger (U25 accepted) | §3 S22; §4.5 S22 |
| D10 | Telemetry stream outlives `GRAPH.DELETE` (one sighting) | Rev §S03 graph-dba "Incident" | §3 S21; §4.5 S21 |
| E1 | EP §14 table (S33, S35, S40–S49, S52, S14/S30) | EP §14 | §14 rows; §3 rows; §6 |
| E2 | Q1: P1 default depth bound ≥ 2 | EP §15.2 Q1, §0 | S50; §15 |
| E3 | Q2: BS §3.5 should list `config` and `unit` locators | EP §15.2 Q2 | F18; §4.2 S03 follow-ups |
| E4 | Q3: P1 prompt extractor uses §3.3's glob list | EP §15.2 Q3 | S40; §6.2; S50; §15 |
| E5 | Q4: YAML parser present | EP §15.2 Q4; teco relay (`pyyaml==6.0.3` in `dev`) | §5.2; S49 |
| E6 | Q5: AC-5 "new blind list" reading | EP §12.1; teco relay (stakeholder decision 2026-10-29) | §6.5; §12 (decided) |
| E7 | Q6 / session count 18–25 | EP §14, §15.2 Q6; teco relay | §6.6 |
| E8 | Anonymous-scope anchors (E1) | EP §2.3; BS §11 | §2 D8; S52 |
| F1 | S09 accepted (`6f4e34f9`); extras frozen | teco relay | §1.1; §3 S09; §5.2 |
| F2 | S10 in flight; R11 handled by S11 | teco relay | §1.1; §3 S10; §2 D15 |

**Items that could not be placed:** none. C9 is deliberately left to S08's own gate and to P2.

### 16.1 v1.1 traceability (one row per handed-onward item)

Sources: **EP** = `docs/plans/connected-code-graph-eval-protocol-ml.md` v1.2 (§14 table, §15.2,
§16); **BS2** = `docs/plans/connected-code-graph-bundle-spec2.md` v1.3 (§0, §3, §4.1, §4.2);
**ES2** = `docs/plans/connected-code-graph-egress-spec2.md` v1.2 (§0, §3); **L** = the ledger,
notes 2026-10-29 to 2026-10-30. "Declined" means not placed in a P0 step, with the reason.

| # | Item | Source | Landed in / reason declined |
|---|---|---|---|
| EP-1 | S33: interpreted symbol anchors carry `qname` (for §10.4's `E`) | EP §14 S33, §15.2 Q1 (v1.1) | §3 S33 |
| EP-2 | S35: assertion column for AC-25's four shape properties; S39 tests it | EP §14 S35, §16 m6 | §3 S35, S39 |
| EP-3 | S40: paste Appendix B verbatim and nothing else (not §3/§4) | EP §14 S40 (M3) | §3 S40; §6.2 |
| EP-4 | S40: `eval-manifest.yaml` gains `units`, `cypherMcpTools`, grain table | EP §14 S40 (v1.1) | §3 S40 |
| EP-5 | S40: units carry `ports` from source; grain table Kiro rows | EP §14 S40 (v1.2, P2-4) | §3 S40 |
| EP-6 | S41/S42: Appendix B; 10 sessions | EP §14 S41/S42 | §3 S41, S42; §6.1; §6.6 |
| EP-7 | S43/S43a/S44: Appendix B's QW and AN parts | EP §14 S43/S43a/S44 | §3 S43, S44 |
| EP-8 | S45: `scope` verb; N3 pairs | EP §14 S45 (v1.1) | §3 S45, S49; §6.4 |
| EP-9 | S47: probe clusters; `H` beside `H_all`; reserve file; empty `exposure.jsonl` | EP §14 S47 (v1.1) | §3 S47; §6.4 |
| EP-10 | S47: probe = served point with fewest items; step 3b served-anchor rule; step 4 guard; `served-anchor` moves as a count | EP §14 S47 (v1.2, P2-1) | §3 S47; §6.4 |
| EP-11 | S48: `--list-units`, `--list-cypher-tools` | EP §14 S48 (v1.1) | §3 S48, S40 |
| EP-12 | S48: tree-sitter from the `spike` extra, correct "or S02's image" | EP §14 S48 (m5) | §3 S48 |
| EP-13 | S49: oracle cases B1a, B1b, M2, M6, reserve, m4 | EP §14 S49 (v1.1) | §3 S49 |
| EP-14 | S49: `regate-list` and `exposure` verbs (needed from P1) | EP §14 S49 (v1.1) | §3 S49 (specified; buildable in P1); §15 gate harness |
| EP-15 | S49: oracle cases P2-1a–e, P2-2a–d, P2-3 | EP §14 S49 (v1.2) | §3 S49 |
| EP-16 | S49: `match_keys.py` exports `E`, `h1_symbol_credit`, `source_check_needed` | EP §14 S49 (v1.2) | §3 S49; §15 gate harness |
| EP-17 | S49: `split` implements step 3b and step 4 guard; `regate-list` re-applies 3b and routes by S(u); `--check-report` uses §9.4's v1.2 pattern | EP §14 S49 (v1.2) | §3 S49 |
| EP-18 | S52: quote Appendix B, not §4.1–§4.3 verbatim | EP §14 S52 (v1.1, M3) | §3 S52 (v1.0 text corrected) |
| EP-19 | S14/S30: touched anchors as locators, hops excluded | EP §14 S14/S30 | §3 S14, S30 (unchanged from v1.0) |
| EP-20 | S30: server-binding walkthrough on `falkor-chat/server` or the fixture, not `cypher-mcp`/`mcp-monitor` (Q10) | EP §14 S30 (v1.2), §15.2 Q10, R9 | §3 S30; §12 SQ-P0-1; K-P0-18 |
| EP-21 | S56: output is an input to S47's contact set | EP §14 S56 (v1.1) | §3 S56, S47; §6.4 |
| EP-22 | P1 gate harness: floor, exposure record, four-field pre-registration comparison (`coreTreeHash`) | EP §14 P1 gate harness (v1.1) | §15 handoffs |
| EP-23 | P1 gate harness: H1 always needs the span test; §5.4 under `E`, imported from `match_keys.py` | EP §14 P1 gate harness (v1.2) | §15 handoffs; §3 S49 (exports) |
| EP-24 | Q1: P1 default depth bound ≥ 2 | EP §15.2 Q1 | §3 S50; §15 |
| EP-25 | Q2: bundle spec §3.5 locator types | EP §15.2 Q2 | resolved by BS2 §1.9 (BS2-22) |
| EP-26 | Q3: P1 prompt extractor uses §3.3's glob list | EP §15.2 Q3 | §3 S40, S50; §6.2; §15 |
| EP-27 | Q4: YAML parser | EP §15.2 Q4 (closed) | §3 S49 (`pyyaml` in `dev`); nothing to change |
| EP-28 | Q5: AC-5 "new blind list" reading, with the fresh-item floor | EP §15.2 Q5, §12.1, E12 | §6.5; §12 (decided) |
| EP-29 | Q6: session count +2–3 | EP §15.2 Q6 | §6.6 |
| EP-30 | Q7: re-gate fresh-item shortfall (conditional stakeholder question) | EP §15.2 Q7 | §12 |
| EP-31 | Q8: restore SQ-P0-1's text in the successor | EP §15.2 Q8 | §12 SQ-P0-1 (full text restored) |
| EP-32 | Q9: `H` < 75 while `H_all` ≥ 75 fires SQ-P0-1 as usual | EP §15.2 Q9 | §12 SQ-P0-1 trigger |
| EP-33 | E11 (flow counts across a boundary) and E12 (floor) as stakeholder decisions | EP §0, L 2026-10-29 | §12 decided; §6.4; §6.5 |
| EP-34 | R7, R9 | EP §15.1 | K-P0-17, K-P0-18 |
| EP-35 | M5 nit: narrow `coreTreeHash`'s `*.md` exclusion if P1 keeps prompts as `.md` | EP §16 Pass 2 | §15 P1 condition |
| BS2-1 | §2 D5: `-spec2` via S16 → `-spec3`; frozen contract = spec + `-spec2` | BS2 §4.2 row 1, §0 | §2 D5; §3 S16 |
| BS2-2 | §2 D17, S58 row, §5.3's S58 row, K-P0-16: dispositions recorded in `-spec2`, before S15; K-P0-16 mitigated | BS2 §4.2 row 2 | §2 D17; §3 S58; §5.3 S58 row; K-P0-16 |
| BS2-3 | §4.0 deliverable (b) row: "bundle spec + `-spec2`; `-spec3` required before P0 exit" | BS2 §4.2 row 3 | §4.0 (b) row |
| BS2-4 | §9.2 P0 exit: `-spec3` exists, is gated, `Supersedes:` both (F-5) | BS2 §4.2 row 4, §0, §3 F-5; teco brief | §9.2 exit item 5; §2 D5; §15 |
| BS2-5 | §3 S03 row: contract = spec v1.3 plus `-spec2` | BS2 §4.2 row 5 | §3 S03 (cites `-spec2` v1.3, the accepted version, not v1.1) |
| BS2-6 | §3 S10 row: §4.1 follow-up round before S15, gate `analyst`; done-condition = §1.6.3's normative probe | BS2 §4.2 row 6, §4.1 S10 items 1–8, §1.6.3; teco brief | §3 S10; §9.1 |
| BS2-7 | §3 S11 row: §4.1's emitter items (`nameNorm`, `import.ordinal` + dedup, cond→cond, ASCII quoting) | BS2 §4.2 row 7, §4.1 S11 | §3 S11; §4.4 |
| BS2-8 | §3 S11 row: G-4 (self-edges dropped, pairs de-duplicated, verified on the slice CPG) | BS2 §4.2 row 11, §4.1 S11; teco brief | §3 S11; §4.4 |
| BS2-9 | §3 S12 row: step 8a (T6, `{#}`), SL-1/SL-2 unchanged | BS2 §4.2 row 8, §4.1 S12 | §3 S12; §4.4 |
| BS2-10 | §3 S15 row: record names spec + `-spec2`; §1.6/§1.7 "frozen, untested"; gate checks S10 encodes every [pre-freeze] item | BS2 §4.2 row 9, §4.1 S15 | §3 S15; §2 D5; §4.4 |
| BS2-11 | §3 S16 row: output `-spec3`, `Supersedes:` both; not optional | BS2 §4.2 row 10 | §3 S16 |
| BS2-12 | §3 S17 row: F-11 (§1.11 graph items, F-2 closure, G-1, P2-8, S-1/S-2), `Cond` source, `defaultHash` not carried | BS2 §4.2 row 12, §3 F-11, §4.1 S17 | §3 S17 |
| BS2-13 | G-1 / security P3-1: carry only role-relevant `launchFact` fields, never all of `args` | BS2 §3 F-11; teco brief | §3 S17 |
| BS2-14 | §3 S33 row: F-3, F-4, F-12, G-1's rendering | BS2 §4.2 row 13, §3, §4.1 S33 | §3 S33 |
| BS2-15 | F-12 also for S20 (`architect` S20/S33 note) | BS2 §3 F-12 | §3 S20 |
| BS2-16 | F-3, `cobb`'s half for S52 | BS2 §3 F-3 | §3 S52; §15 |
| BS2-17 | F-10 P1 writer obligations | BS2 §3 F-10, §4.2 P1 row | §15 P1 handoffs |
| BS2-18 | F-6, F-7, F-9 (deadline) | BS2 §3, §4.2 P1 row | §15 P1 handoffs |
| BS2-19 | F-1 bundle-at-rest (`devops`) | BS2 §3 F-1 | §15 P1 handoffs |
| BS2-20 | F-2 closed (S17 records) | BS2 §3 F-2 | §3 S17 |
| BS2-21 | F-8: step 8a in EP's key normalizer (`data-scientist`) | BS2 §3 F-8 | v1.1: declined (no owner yet). **v1.2: placed** — owned and delivered by EP2 (`data-scientist`), encoded at **S49**, with S40, S45, S47 and S12's P0-11a as their rows state (EP2 §3 row "§16.1 row BS2-21"; P3-3) |
| BS2-22 | §15 F18 resolved by §1.9 | BS2 §4.2 last row | §11 F18; §4.2 |
| BS2-23 | S08: nothing from BS2 | BS2 §4.1 S08 | nothing to place |
| BS2-24 | No golden-vector change | BS2 Ready-to-implement; L 2026-10-29 | §2 D10 |
| BS2-25 | Rule 5 "Extends but replaces named clauses" form question | BS2 §0; L 2026-10-29 | §15 follow-ups (human/`cobb`); **declined from this plan's steps:** a convention question, not P0 work |
| ES2-1 | §4.1: the errata table points to ES2; E1 superseded by §1.1, E2 restated as §1.2 | ES2 §0, §3 row 1 | §4.1; §2 D18 |
| ES2-2 | §2 D6: "with errata E1–E2" → "with ES2" | ES2 §3 row 2 | §2 D6 |
| ES2-3 | §3 S04 row | ES2 §3 row 2 | §3 S04 |
| ES2-4 | §3 S08 row (wording) | ES2 §3 row 2 | §3 S08 |
| ES2-5 | §4.2's S04 paragraph | ES2 §3 row 2 (ES-m5) | §4.2 S04 |
| ES2-6 | §5.3's S04 row | ES2 §3 row 2 | §5.3 S04 row |
| ES2-7 | §9.1's `cpg/sandbox/` row | ES2 §3 row 2 (ES-m5) | §9.1 |
| ES2-8 | S08 done-condition: tests as amended — §1.1's test 10(b), §1.3's concurrency test, §1.7's polluted-environment test, A6(b′) in test 1 | ES2 §3 row 3 | §3 S08; §9.1 |
| ES2-9 | S08 follow-up: §1.1 `\r` split, leading-byte probe, `x\r//>` fixture | ES2 §3 S08, §1.1 | §3 S08 |
| ES2-10 | S08 follow-up: §1.3 one `--output` per live invocation, lock outside the `upperdir` | ES2 §3 S08, §1.3 | §3 S08 |
| ES2-11 | S08 follow-up: §1.4 n2 and m6 (namespace `/usr`) | ES2 §3 S08, §1.4 | §3 S08 |
| ES2-12 | S08 follow-up: §1.5 `usage`, `netnsDistinct`/`egressMode` on refusals, `toolArgv` tests | ES2 §3 S08, §1.5 | §3 S08 |
| ES2-13 | S08 follow-up: §1.7 closed environment (`env -i` re-exec, fixed `PATH`, `DOCKER_HOST`, `DOCKER_CONFIG` dir checks → 69, passwd `HOME`, `-p` shebang) and the polluted test (`BASH_ENV`/`ENV`, pre-populated dir, no-`-p` mutant) | ES2 §3 S08, §1.7 (v1.1–v1.2) | §3 S08 |
| ES2-14 | S08 follow-up: §1.8 A6(b′) and the launch-equivalence check; README backend table and concurrency claim | ES2 §3 S08, §1.8 | §3 S08 |
| ES2-15 | S11: §1.5's `@AUDIT_ID@` contract normative | ES2 §3 S11, §1.5 | §3 S11; §4.4 |
| ES2-16 | Consolidation: the next egress revision (P1 broker) `Supersedes:` both | ES2 §0 | §2 D18 |
| ES2-17 | Q1 git tunnel still open for the stakeholder at P1 | **egress spec v1.3 §11 Q1** (not ES2; listed here as the brief grouped it, n-a); teco brief; L 2026-10-29 | §12 (open); §15 |
| L-1 | S08 Pass 2 carry-overs p2-m1, p2-n1, p2-n2 into S08's follow-up | L 2026-10-29 (U28 Pass 2) | §3 S08 |
| L-2 | S11's brief: R11 content check and G-4 | L 2026-10-29 checkpoint step 4; teco brief | §3 S11; §2 D15; §4.4 |
| L-3 | `.gitignore`: S09 committed, S08 the sole remaining writer, nothing to serialize | L 2026-10-29 (U29 note) | §5.2; F15 |
| L-4 | States: S06 v1.2, S08, S10, S58 accepted; BS2 v1.3 and ES2 v1.2 accepted | L ledger rows U26, U28, U30–U32; notes 2026-10-30 | §1.1; §3 rows; §15 digest |
| L-5 | `tico` records Q5/E11/E12 in the requirements | L 2026-10-29 | §12; §15 follow-ups |

#### v1.2 rows (review Pass 3, EP2 §3, S10 follow-up notes)

Sources: **Rev3** = `docs/reviews/connected-code-graph-p0.md` `## Pass 3`; **EP2** =
`docs/plans/connected-code-graph-eval-protocol2-ml.md` §3 (and its Pass 2 additions relayed by
`teco`), cited by section; **S10n** = the S10 follow-up notes relayed by `teco`. Each row names the
step (or section) whose done-condition encodes the fix, so `teco` can grep it there.

| # | Item | Source | Encoded at (greppable) |
|---|---|---|---|
| P3-1 | S47 depends on S14, S30, S34, S56; structure line | Rev3 P3-1 | §3 S47 Deps "**S14, S30, S34, S56** (P3-1)"; §3 structure "S47 also waits on every contact source" |
| P3-2 | S16 re-points `SPEC_TEXT`/`SPEC2_TEXT` at `-spec3` in both modes; `test_ids.py` vectors checked; §5.2 writer | Rev3 P3-2 | §3 S16 "In both modes (P3-2)"; §5.2 `cpg/tests/test_bundle.py` row |
| P3-3 | F-8 owned; S49 encodes 8a with oracle; S49 depends on EP2; S45/S47 inherit; start-not-close fallback | Rev3 P3-3 | §3 S49 "Step 8a (F-8, EP2; P3-3)", Deps "EP2 accepted (P3-3)"; S45 and S47 Deps; BS2-21 |
| P3-4 | S24 item 5 held until S08's round accepted; `CCG_TOOLCHAIN_IMAGE` for the re-run; one serialized `sbx_default_image` edit | Rev3 P3-4 | §3 S24 Deps "S08's follow-up round (item 5 only)"; §4.1 S24 item 5; §5.2 `offline-run.sh` row |
| P3-5 | CLI pin: pin 2.1.312 via `prepare-bin` first; K-P0-12 rule restored; §6.3 and §7 corrected | Rev3 P3-5 | §2 D20; §3 S53 item (0); §6.3 "now points at 2.1.314"; §7 CLI row; §13 K-P0-12 "(restored from v1.3)" |
| P3-6 | Restore the possible-precision ratification in the P1 harness | Rev3 P3-6 | §15 "ratifies the possible-precision thresholds" |
| P3-7 | One durable content-check script with an owner; per-file variant considered | Rev3 P3-7 | §3 S08 "`cpg/sandbox/scripts/cpg-content-check.sc`"; §2 D15; §3 S11 (per-file measurement); §5.2 row |
| P3-8 | S16 always runs: §9.2 item 1 and §3 structure | Rev3 P3-8 | §9.2 item (1) "S16 included, in whichever mode"; §3 structure "S16 always runs" |
| P3-9 | FR-15/hostnames decision gets a recording owner | Rev3 P3-9 | §15 `tico` follow-up "FR-15's scope plus internal hostnames" |
| P3-10 | Every runner consumer runs on the post-round runner; S11 not "now dispatchable" | Rev3 P3-10 | §3 S11, S25, S28, S39 Deps "S08's follow-up round"; §3 structure "every runner consumer"; §5.2 runner-consumer rule |
| P3-11 | `shellcheck` unavailable | Rev3 P3-11 | §3 S53 "passes `bash -n`"; §7 `shellcheck` row; §9.1 S53 row |
| n-a | Counts wording; ES2-17's source | Rev3 n-a | §16.1 counts below; ES2-17 row |
| n-b | §12 paragraph eval-protocol G5 cites | Rev3 n-b | §12 SQ-P0-1 "No estimate of whether it will be needed"; §15 `data-scientist` follow-up |
| n-c | S10 names `-spec2` §1.7.4 W1 | Rev3 n-c | §3 S10 "W1" |
| n-d | Split S49 into two units | Rev3 n-d | §3 S49 units P0-42a, P0-42b |
| R-1 | EP R1's P1 runtime oracle (unplaced in v1.1) | Rev3 recount; EP §15.1 R1 | §15 "reports the runtime oracle" |
| R-2 | S46's 40 + 8 sample (placed, no row) | Rev3 recount; EP §14 S46 | §3 S46; §6.4 |
| R-3 | S33's base hit rule (placed, no row) | Rev3 recount; EP §14 S33 | §3 S33; §4.2 S33 |
| R-4 | S35's §12.2 minimums (placed, no row) | Rev3 recount; EP §14 S35 | §3 S35; §4.7 |
| R-5 | EP R2 (census small after contact) | Rev3 recount; EP §15.1 R2 | §13 K-P0-5; §12 SQ-P0-1 |
| R-6 | EP R6 (held-out read by a builder) | Rev3 recount; EP §15.1 R6 | §13 K-P0-7; §2 D11 |
| R-7 | W1 implicit | Rev3 recount | = n-c, §3 S10 |
| R-8 | FR-15/hostnames decision, no row | Rev3 recount | = P3-9; §12 decided list |
| R-9 | Dropped v1.3 texts: K-P0-12, possible-precision, §12 G5 paragraph | Rev3 recount | = P3-5, P3-6, n-b |
| EP2-1 | §3 S06 / §14: contract is EP v1.2 plus EP2; §14 consumers gain EP2 | EP2 §3 | §3 S06; §14 "EP2 … is a consumer source" |
| EP2-2 | §3 S10: nothing new | EP2 §3 | nothing to place |
| EP2-3 | §3 S12: writes the shared 8a function in `cpg/ccg/`; K1–K22, K30–K33 and K42–K48 provider form; SL-1/SL-2 unchanged | EP2 §3, §1.1 | §3 S12 P0-11a "K1–K22, K30–K33 and K42–K48"; §9.1 `keynorm.py` row |
| EP2-4 | m4 / Q1: file and module for the shared function; S12 → S49 ordering | EP2 §1.1, §4 Q1, §5 m4 | §2 D19 (`cpg/ccg/keynorm.py`, `mask_http_segments`; S49 after P0-11a); §5.2 `keynorm.py` row |
| EP2-5 | §3 S14, S30, S34: none | EP2 §3 | nothing to place |
| EP2-6 | §3 S40: manifest records both documents' versions and `screen: {specVersion, bundlePySha256}` | EP2 §3, §1.5 | §3 S40 "Per EP2 §1.5" |
| EP2-7 | §3 S41–S44, S52: none | EP2 §3 | nothing to place |
| EP2-8 | §3 S45: `normalize` applies EP2's `http` rule; 8a rewrites are V3 replace-and-log; method owner reviews | EP2 §3, §1.1 | §3 S45 "EP2's `http` `keyNorm` rule"; Deps |
| EP2-9 | §3 S47: C through `normalize_contract` before `anchorId`; gold stored normalized, rewrites logged and counted; unserved-gold check; `protocol` string; `--check-report` two probes | EP2 §3, §1.2, §1.4, §1.5 | §3 S47 "`normalize_contract`", "`protocol` string", "two probes" |
| EP2-10 | §3 S49: imports the shared function; exports `normalize_contract`; `renormalize-check`; two-form probes; oracle K1–K48; Deps S10 and S12 | EP2 §3 | §3 S49 "Step 8a (F-8, EP2; P3-3)", Deps "S10's follow-up round, S12's P0-11a" |
| EP2-11 | P1 gate harness: reads normalized gold and ids; `renormalize-check` before scoring; `anchor-not-in-graph`; imports | EP2 §3, §1.5 | §15 gate harness "renormalize-check before scoring" |
| EP2-12 | §15 F-8 line: delivered by EP2, encoded at S49 | EP2 §3 | §15 `data-scientist` follow-up |
| EP2-13 | §16.1 BS2-21: placed at S49 with S40, S45, S47 | EP2 §3 | BS2-21 row |
| EP2-14 | (EP2 Pass 2) P1 assembler imports the shared 8a function | EP2 §3 (v1.2, relayed) | §15 "The P1 assembler imports `mask_http_segments`"; §2 D19 consumers |
| EP2-15 | (EP2 Pass 2) AC-24 gold locator carries a stored form (`{#}`/`{?}`); `renormalize-check` takes it; oracle K40–K48 | EP2 §1.2, §1.5, §2 (v1.2, relayed) | §3 S47 "stored form"; §3 S49 "K1–K48", "stored form … as an input"; §9.1 S49 row |
| EP2-16 | (EP2 Pass 2) step-5/step-6 order may move: do not encode it | `teco` relay | §2 D19 "What p0-2 does not fix"; §3 S12, S45 cite EP2's rule without an order |
| S10n-1 | `-spec2` §1.1's depth example is non-canonical under §1.1's rule | S10 follow-up | §3 S16 "`-spec3` notes" |
| S10n-2 | Lookaround stays in the schema's `(?!\n)` guards and the path pattern | S10 follow-up | §3 S16 "`-spec3` notes" |
| S10n-3 | T4 is nearly redundant | S10 follow-up | §3 S16 "`-spec3` notes" |
| S10n-4 | Allow-list grid 85,536 cells (18 representatives) vs `-spec2` §1.6.3's 71,280; reconcile | S10 follow-up | §3 S16 "85,536 cells" |

**Counts (n-a: "nothing to place" now counted separately).**
- **v1.1 rows:** EP 35 (33 placed; EP-25 resolved via BS2-22 and EP-27 nothing to place); BS2 25
  (22 placed, 1 declined — BS2-25, the rule-5 form question — and BS2-23, BS2-24 nothing to place /
  no change; BS2-21 moved from declined to placed in v1.2); ES2 17 (all placed; ES2-17's source is
  the egress spec); L 5 (all placed). **82 rows: 77 placed, 4 nothing to place, 1 declined.**
- **v1.2 rows:** Rev3 24 (P3-1–P3-11, n-a–n-d, R-1–R-9; R-7–R-9 cross-refer to findings already
  rowed); EP2 16 (13 placed, 3 nothing to place: EP2-2, -5, -7); S10n 4 (all placed). **44 rows: 41
  placed, 3 nothing to place, 0 declined.**
- **Total 126 rows: 118 placed, 7 nothing to place, 1 declined (BS2-25); none unplaced.**
