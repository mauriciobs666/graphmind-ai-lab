# Connected code graph — phase P0 implementation plan

> **Status:** active · **Owner:** `architect` · **Tracks:** C-1001 (M10) · **Extends:** `docs/plans/connected-code-graph.md` · **Version:** 1.3 · **Reviews:** `docs/reviews/connected-code-graph-p0.md`

*Revision 2026-10-29 (v1.1): review Pass 1 addressed except B1 (pending a stakeholder decision, §6). Dispositions, including the ones I disagree with, are in §15.*
*Revision 2026-10-29 (v1.2): B1 resolved per the stakeholder's decision: the ground-truth writers run as separate headless sessions (§6 rewritten; S40–S48 revised; new S52–S54).*
*Revision 2026-10-29 (v1.3): review Pass 2 (P2-1–P2-4, P2-6, the S14 nit) addressed. Held-out encryption removed by stakeholder decision, which supersedes P2-5. See §15.*

## 0. Goal & scope

**Goal.** Execute phase P0 of `docs/plans/connected-code-graph.md` §4: deliver (a)–(h), settle
every §6.2 "for P0 to settle" decision with a recorded measurement, and either freeze bundle
schema v1 and the graph model or record that they stay provisional until P1 exit. P0 gates no AC;
its exit condition is the ledger's P0 definition of done (`docs/plans/connected-code-graph-coordination.md`,
"Phase P0").

**In scope:** the steps in §3; the dedicated FalkorDB instance and the toolchain image that P0's
measurements need; the durable code those deliverables require (schemas, validators, sandbox,
fixtures, ground-truth tooling); the results notes that record every measurement.

**Out of scope (named handoffs to P1 or later, never built here):** unit discovery; production
extractors and normalizers; the rule-pack runtime; the linker; the store loader with swap, stamp
and ACL users; the answer engine; the `ccg` MCP server, CLI and `raw_query`; the egress broker
implementation and AC-8's logging proxy; plain-language interpretation; the id-hash fuzz test
(proposal §5.1, needs the P1 loader); the Arduino slot. The list of handoffs is §14.

**Naming.** The slug `connected-code-graph-p0` is kept. Root `AGENTS.md`'s prohibition covers
basenames that *begin* with `m<digit>`, `k<digit>` or a date, so it does not apply. The exception
clause ("the milestone token goes inside the slug, never as a prefix; the test is *the topic has no
name without it*") fits a per-phase plan, and the repo already does this
(`docs/plans/small-model-benchmarking-s3-spec.md` … `-s7-spec.md`). The alternative I rejected is
collision rule 5's ordinal successor (`connected-code-graph2.md` with `Extends:`). It is formally
available, because the proposal has been gated, but it would number P0…P5 as 2…7 and say nothing
about the phase. This plan's downstream documents reuse its slug (`docs/reviews/connected-code-graph-p0.md`,
already named in the ledger). P0's contract specs are durable across phases, so they get their own
topic slugs (§2 D9), not the phase token.

## 1. Context & findings

CPG: considered, not relevant — `cpg_falkorchat` is stale (built 2026-10-15, 21 `falkor-chat/server` commits behind) and P0's only code-level contact with that component is P0(b)'s slice, which builds its own bundles from source; nothing in this plan was derived from the graph, and the ground-truth agents are forbidden to use it (§6.3).

**What P0 builds on (read for this plan).**

| Asset | Fact that shapes P0 |
|---|---|
| `skills/joern-cpg/scripts/` (`build-cpg.sh`, `git-provenance.sh`, `joern-env.sh`, `pipeline.sh`) | The build stage and provenance capture are reused by the slice's extraction (S11). Its neo4jcsv export and `--load` are **not** used by P0: no P0 step loads a `cpg_*` graph. |
| `skills/joern-cpg/references/cpg-model.md` | The verified `pysrc2cpg` topology (34% resolved call sites, intraprocedural `REACHING_DEF`, `FILENAME` only on `METHOD`/`TYPE_DECL`) is the starting point for the slice's projection script. It is Python-only, so the TS half of the slice is new ground. |
| `cpg/` | Holds only `.cpg-artifacts/` (gitignored) and `.gitignore`. P0 creates the first code here (§2 D2). |
| Joern v4.0.579 at `~/joern/joern-cli` (`lib/io.joern.joern-cli-4.0.579.jar`) | All seven FR-1 frontends are present, plus bundled astgen binaries for JS/TS, C# and Go. Their fetch flags were re-checked with `--help` for this plan (§7). |
| `falkordb-dev` (`falkordb/falkordb:v4.18.11`, port 6379) | Shared by other sessions and components. **No P0 step writes to it** (§5.1). The same image tag is used for the dedicated instance so that every quirk in `claude/graph-dba/falkordb-quirks.md` and every V-probe in the graph note still applies. |
| `cypher-mcp/build.sh`, `docker-run.sh` | Precedent for a content-hash image tag, reused for the toolchain image (S02). |

**Repo facts that drive the design.**
- Only Python, TS/JS and shell exist in the proving repo. Java, Kotlin, C#, Go and C/C++ have
  **no** in-repo code to measure, so P0(a) needs external corpora (§2 D4).
- The OpenTelemetry demo is AC-7's codebase and must stay unseen until its per-stratum first
  contact (AC-7; ML §4.5). The held-back third codebase (ML §4.5 suggests Online Boutique) must
  stay unseen by anyone building the system. **Neither may be used as a P0(a) corpus.**
- The host is a WSL2 VM capped at **6 GB RAM / 8 vCPUs** by `C:\Users\mauri\.wslconfig`, with a
  comment saying the cap protects LM Studio on the 16 GB Windows host. It showed 5 GB total in
  `free -g`. That is enough for P0's measurements at the sizes in §4, but only if the
  instance-heavy and JVM-heavy steps are serialized (§5.2). Verifying the ~18 GB workplace peak
  (proposal §3.5) is **not** a P0 goal, and this host cannot do it.
- `unshare -rn` (unprivileged user+net namespace) and `bwrap --unshare-net` both give a loopback-only
  namespace with no DNS on this host (probed). **That is not the same as offline on WSL**: Windows
  binaries still launch through interop and use Windows' network stack (review M1). So the
  namespace fallback refuses to start while interop is enabled (§2 D6), and on this host only the
  container backend (Docker 29.7, Compose v5.4) is usable.

## 2. Design & rationale

**D1 — Three kinds of P0 artifact, and only three.**
- **Contract**: a spec under `docs/plans/` written by `architect`, plus its machine-readable form
  (JSON Schema) and a validator under `cpg/ccg/`, which are durable code.
- **Measurement spike**: throwaway code under `cpg/spikes/<name>/`, plus a durable results note
  under `docs/test-reports/` or `docs/plans/…-graph.md`, plus raw results committed as JSON beside
  the spike. Spike code is never imported by `cpg/ccg/`. Whether it is deleted or ported is P1's
  plan's call. The results note is what P1 and the §6.2 decisions cite. **A spike without its
  results note is not done.**
- **Durable code**: `cpg/ccg/` (schemas, ids, validators), `cpg/sandbox/`, `cpg/infra/`,
  `cpg/fixtures/`, `cpg/eval/`. It is tested (§9) and is the base P1 builds on.

Rejected: writing the slice as P1-quality durable code. The slice exists to put the provisional
schema under pressure and to fail cheaply. Holding it to the durable bar would slow exactly the step
whose result decides the freeze. P1 ports what survives, under its own tests.

**D2 — Code layout under `cpg/`.** This extends proposal §3.13, which leaves the details to P1.
P1 may move anything below `cpg/ccg/`.

```
cpg/
  pyproject.toml            package `ccg`, Python 3.12; pytest testpaths=["tests"]; marker `live`
  README.md                 how to set up, run tests, layout (new, S09)
  .gitignore                + `.ccg-work/` (scratch: corpora, bundles, exports)
  ccg/
    ids.py                  canonical tuple → 32-hex id (S09)
    bundle.py               bundle JSONL read/validate/write, deterministic order (S10)
    answer.py               canonical answer serialization (S34)
    schemas/                bundle-v1, rule-pack-v1, answer-v1, fixture-manifest-v1 (*.schema.json)
    store/ddl_v1.cypher     graph model v1 DDL (S17)
  sandbox/                  offline build runner + forbidden-flag list + tests (S08)
  infra/                    ccg-falkordb.sh (S01); toolchain/ Dockerfile + build.sh (S02, S24)
  packs/<name>/pack.yaml    paper packs, `status: paper` (S31, S32)
  fixtures/system1/         fx-svc (Python), fx-web (TS), fx-lib (Java), expected/*.yaml, materialize.py
  eval/                     EVAL_COMMIT, eval-manifest.yaml, kit-manifest.json, heldout-manifest.json, dev/ (held-out itself lives outside the repo, D11)
  spikes/                   frontend-fitness/, vertical-slice/, store/   (throwaway, D1)
  tests/                    durable-code tests; `live` ones need Docker/FalkorDB/Joern
```

Tests run with `cpg/` as the working directory (`cd cpg && .venv/bin/python -m pytest`), per root
`AGENTS.md`. `live` tests are deselected by default (`-m "not live"`) and run explicitly.

**D3 — P0(a) is split into a1 and a2.** The proposal's (a) includes "per-file density and build
time of the flow projection" for every language. But the projection script exists only once the
slice (b) has written it (S11). So:
- **a1** (S25–S27) is Joern-only and independent of (b): parse success, line fidelity, fallback
  call recall, name-match fan-out, resolved-call precision, decorator/annotation shape, value-set
  recovery.
- **a2** (S28) runs the slice's projection on every corpus and decides `flowAvailable`.

This is a sequencing fact, not a scope change.

**D4 — Corpora for P0(a).**
- **Python and TS/JS:** the proving repo's own `falkor-chat/server` and `salesperson/`.
- **The other five languages:** one public, small-to-medium OSS project each, pinned by commit
  SHA. Each must build with that language's standard tool, and must contain the idioms the §3.2
  table names as weak spots (DI or annotations for Java and Kotlin, attribute routing for C#,
  interface dispatch for Go, macros for C/C++).
- **Exclusions:** the OpenTelemetry demo, every candidate for the held-back third codebase
  (including Google's Online Boutique), and anything vendored by either.
- **Selection:** `graph-dba` picks the projects against S05's criteria, and `data-scientist`
  confirms them in S05's gate.
- **Fetching** the corpora is a one-time hosted download of public code. It is not a local-only
  codebase's egress, so it runs with network, outside the sandbox.

Rejected: using the OpenTelemetry demo "because it has every language". That would destroy AC-7's
first contact for every stratum at once.

**D5 — The slice freezes by the proposal's binary rule.** Proposal §4 P0(b): schema v1 and the
graph model freeze **only when the slice passes**; otherwise they stay provisional until P1 exit.
This plan defines "passes" as all of S14's criteria SL-1–SL-7 (§4.4). The rule this plan executes is
the proposal's: any failed criterion leaves v1 provisional. Freezing only the parts that passed
would depart from the proposal's text. So this plan does not offer it. If S15 shows a partial pass
and `teco` thinks a partial freeze is worth having, that is a stakeholder question to raise at that
point (§12, SQ-P0-2).
- The bundle spec is written **once**, as "v1 (candidate)", with its freeze condition stated
  inside it (S03). When the slice passes unchanged, the freeze is recorded by S15's gate (slice
  report, ledger, `docs/HISTORY.md`). Nothing gets edited.
- If the slice forces a schema change, the spec has already been executed against, so collision
  rule 5 requires a successor, `docs/plans/connected-code-graph-bundle-spec2.md`
  (`Supersedes:`), written in S16.
- The graph model is handled the same way. `graph-dba` records "graph model v1 = graph note §3.2
  as amended by §6, plus these slice deltas" in a new note that `Extends:` the graph note (S17).

**D6 — Offline sandbox: container first, namespace fallback.**
- The primary backend is `docker run --network none` on the `ccg-toolchain` image (S02). This
  pins the toolchain, and it works the same way on a workplace build host.
- The fallback is a namespace on the host: `bwrap --unshare-net` with a **minimal bind set**. It
  is for hosts without Docker, and it uses the host's own Joern and JDK.
- **A network namespace alone is not "offline" on WSL** (review M1, verified by `analyst`). WSL
  interop is kernel-wide binfmt: inside `unshare -rn`, `/mnt/c/Windows/System32/cmd.exe` still
  launches, Windows binaries use Windows' network stack, and `/var/run/docker.sock` is visible.
  So the property "no route out" is stated **per backend and per escape axis**:
  - **Container:** no `/mnt/*`, no `docker.sock`, no `/run/WSL`. The source is mounted
    `ro,noexec` and the output `rw,noexec`, so no Windows PE file planted in either can be
    executed through interop.
  - **Namespace fallback:** it binds no `/mnt/*`, no `/var/run/docker.sock` and no `/run/WSL`.
    It **refuses to start while `/proc/sys/fs/binfmt_misc/WSLInterop` is enabled**. On this host
    that means only the container backend is usable. That is accepted, because the container is
    the primary backend anyway.

  S04 contracts these properties, and S08's escape-axis probe tests every axis on both backends
  (§4.1).
- Both backends enforce the same flag policy **before** launch.
  - The forbidden list starts as `--fetch-dependencies`, `--download-dependencies`,
    `--jar4import-url` and `--include-indirect-dependencies`.
  - Every other network- or dependency-shaped flag that the frontends' `--help` shows must be
    classified as forbidden or as reviewed-allowed. Examples are `kotlin2cpg`'s
    `--enable-dependency-resolver-v2` (and its `ENABLE_DEPENDENCY_RESOLVER_V2` environment
    variable), `--generate-nodes-for-dependencies`, and the Gradle download options.
  - A sweep test fails on any flag that is unclassified (m4).
  - The sandbox starts from an emptied environment, so the environment-variable route is closed
    as well.

  Both backends write one audit line per invocation with the full argv.
- AC-8 later asserts against exactly that audit line ("asserted from the command lines", proposal
  §5.3). So the audit format belongs to the egress contract (S04), not to the sandbox code.

**D7 — One dedicated FalkorDB instance, reserved per step.**
- Container `ccg-falkordb`, image `falkordb/falkordb:v4.18.11`, bound to `127.0.0.1:6390` (free
  on this host; 6391–6393 are `graph-dba`'s throwaway-probe precedent).
- `maxmemory 1536mb`, `maxmemory-policy noeviction`, `save ""`, `appendonly no`.
- The admin password lives in `~/.config/ccg/falkordb-admin.pass` (mode 600, outside the repo).
- Every step that touches it holds a **reservation** (§5.1), and `teco` serializes reservations.
  A second instance for parallel spikes was rejected: the 6 GB host cannot hold two instances plus
  a JVM.

**D8 — Ground truth is pinned to one evaluation commit.**
- AC-5's held-out items carry file:line at both ends, and those drift as the repo moves. S40 pins
  `EVAL_COMMIT`, the head of `main` when S40 runs, in `cpg/eval/EVAL_COMMIT`.
- Every P0 ground-truth artifact refers to that commit. **P1's gates build the proving-repo
  snapshot from a worktree at `EVAL_COMMIT`**, using the exclusions in `cpg/eval/eval-manifest.yaml`
  (`deprecated/`, `cpg/`, and the connected-code-graph design docs). The system is
  codebase-agnostic, so evaluating it on a pinned commit loses nothing.
- AC-24's gold anchors, and every key locator, are recorded in S03's id forms at `EVAL_COMMIT`:
  - a symbol is `(lang, repoId, qname[, signature])`, with S03's fixed `repoId` for the proving
    repo and S03's source-derivable `qname` grammar;
  - a contract-point anchor (a route, an MCP tool) is S03's `ContractPoint` tuple
    `(kind, keyNorm)`.

  Both are pure hashes with no `unitId` (graph note §6.4), so the gold resolves to ids without a
  P0 snapshot. That holds only because S03 fixes those three things, which is why S06 and S40
  depend on S03 (finding F5; review M2).

**D9 — Contract documents get durable topic slugs.**
- `docs/plans/connected-code-graph-bundle-spec.md`
- `docs/plans/connected-code-graph-egress-spec.md`
- `docs/plans/connected-code-graph-rule-pack-spec.md`
- `docs/plans/connected-code-graph-answer-spec.md`

All four are `Owner: architect` and `Extends: docs/plans/connected-code-graph.md`. The repo
precedent is `small-model-benchmarking-runner-spec.md`.

The P0 method notes are:
- `docs/plans/connected-code-graph-p0-ml.md`, the measurement protocol for P0(a);
- `docs/plans/connected-code-graph-eval-protocol-ml.md`, the frozen evaluation protocol, P0(f),
  which P1–P5 all cite;
- `docs/plans/connected-code-graph-p0-graph.md`, the store-spike results and graph model v1.

The results notes are `docs/test-reports/connected-code-graph-p0-{frontend-fitness,slice,ground-truth}-report.md`.

**D10 — Ids are SHA-256 truncated to 128 bits** (32 hex), computed over the canonical JSON of the
tuple array. Canonical JSON means UTF-8, sorted keys, no insignificant whitespace, every string
NFC-normalized and stripped. This makes the graph note §3.4's "32-hex-character hash" concrete,
and it is the one place hashing is defined (S03 states it; S09 implements it).

**D11 — Held-out ground truth is plaintext outside the repo, hidden by convention only
(stakeholder decision, 2026-10-29).** There is no encryption, no passphrase and no passphrase file.
- **Where things live:**

  | Material | Location | Committed? |
  |---|---|---|
  | Held-out items | `~/.local/share/ccg-eval/<E>/heldout/*.jsonl`, plaintext | no |
  | Dev items | `cpg/eval/dev/*.jsonl`, plaintext | yes |
  | `cpg/eval/heldout-manifest.json`: per file the SHA-256 and item counts per stratum; no locators and no item content | the repo | yes |

- **What the manifest gives.** Any later change to a held-out file is **detectable**: GC re-hashes
  the files at every gate against the committed manifest, and a mismatch stops the gate. The
  manifest also freezes the set to a commit (ML §4.1: "content-hashed, committed").
- **What hides the keys: convention only.** No builder brief mentions the held-out path or this
  directory. Nothing else stands between a builder and the keys.
  - The path appears in this plan and in the eval-protocol documents, which builders can read.
  - Every P1 builder (`coder`, `tdd-engineer`, …) runs with Bash under the same uid. Any agent
    that went looking, for example while debugging an AC-5 miss, could read the held-out items.
  - **The plan does not claim otherwise, and adds no other mechanism.** Reading by a builder is
    neither prevented nor detected.
- **What this costs.**
  - The held-out set is **not** backed up by git. Losing the directory loses the held-out set,
    though the manifest proves what it was.
  - A builder's accidental exposure leaves no trace.

  Both are accepted with the decision.
- **The v1 options this replaces:**
  - gpg ciphertext in the repo, with a passphrase file (v1–v1.2);
  - time-boxed passphrase custody (review P2-5's recommendation).

  The stakeholder judged both too much.

**D12 — Owners.**
- `architect` writes contracts.
- `tdd-engineer` writes contract code, because its behaviour is fixed up front.
- `graph-dba` drives Joern and the store.
- `coder` writes the slice's assembly and load (spike).
- `devops` does the instance, the image and the sandbox.
- `qa-engineer` writes fixtures and call-site labels.
- `data-scientist` writes the protocols and runs the ground-truth statistics.
- `analyst` gates.

The ground-truth writers are fresh, isolated instances (§6).

## 3. Step table

**Kind codes:**
- **C** is a contract document;
- **CC** is contract code (a schema or validator, durable);
- **M** is a measurement spike (throwaway code plus a durable results note);
- **D** is durable code;
- **GT** is ground-truth data;
- **Doc** is documentation only.

**Reservations:** **R** means the step holds a reservation on the dedicated instance; **X** means
that reservation is exclusive (§5.1).

**Dispatch:** steps that share a unit id are one dispatch, run in order. Each unit is at most 3
steps and about 5 files. Fixture source trees (S36–S38) are the one stated exception: there is one
repo per dispatch, and a repo has a dozen tiny files.

**Ids:** `S01`–`S55` (including `S43a`) are steps. A stakeholder decision is always written *decision S<n>* (for example "decision S36"), never as a bare `S<n>`. Ledger units cite the `P0-nn` unit id.

**Gates:**
- `an` is an `analyst` review;
- `ds` is a `data-scientist` method check;
- `gd` is a `graph-dba` check;
- `tc` is a `teco` documentation check;
- `sh` is the stakeholder.

The step details are in §4.

| Id | Unit | Owner | Kind | Deliverable (files) | Inputs | Done when | Gate | Deps |
|---|---|---|---|---|---|---|---|---|
| S01 | P0-01 | `devops` | D | `cpg/infra/ccg-falkordb.sh`, `cpg/infra/README.md` | §2 D7 | `up`/`status`/`down` are idempotent. `docker port ccg-falkordb` lists only `127.0.0.1:6390` (not `0.0.0.0`; `falkordb-dev` itself publishes `0.0.0.0:6379`, so this must be checked, not assumed). `CONFIG GET` shows `noeviction`, `maxmemory 1536mb`, `save ""` and `appendonly no`. A `GRAPH.QUERY` round trip on `ccg_smoke` works, then that key is deleted. `docker ps` shows `falkordb-dev` unchanged. | an | — |
| S02 | P0-02 | `devops` | D | `cpg/infra/toolchain/Dockerfile`, `cpg/infra/toolchain/build.sh`, `cpg/infra/toolchain/README.md` | §7 | Image `ccg-toolchain:<content-hash>` builds. Inside it, every tool in §7's "image" column prints its version. `joern-parse` of a one-file Python sample succeeds under `--network none`. | an | — |
| S03 | P0-03a | `architect` | C | `docs/plans/connected-code-graph-bundle-spec.md` (v1 candidate) | proposal §3.3; graph note §3.2, §6; §2 D10 | Every record type, field, canonical tuple, id rule, reason code and determinism rule is specified, **including** a source-derivable `qname` grammar per FR-1 language, the fixed `repoId` of the proving repo, and the `ContractPoint` tuple with its `keyNorm` rule (§4.2, M2). It states its freeze condition (§2 D5). It maps every record to graph-note labels and edges. | an + gd | — |
| S04 | P0-03b | `architect` | C | `docs/plans/connected-code-graph-egress-spec.md` | proposal §3.12, §5.3; FR-9; AC-8 | Covers the broker interface, the policy evaluation (sticky local-only), the allow-list format, the audit-record format, the controlled-download path, and the sandbox contract: backends, the **escape axes each backend closes** (§4.2, M1), the forbidden and reviewed-allowed flag lists, and the argv audit. Each AC-8 assertion maps to an audit field. | an | — |
| S05 | P0-04 | `data-scientist` | C | `docs/plans/connected-code-graph-p0-ml.md` | ML §3.6, §8.1 (m1); proposal §4 P0(a); §2 D3–D4 | Fixes the per-language corpus criteria and exclusions, the call-site sample size and seeded draw, the truth source per language, the metrics with intervals, and the decision rules (indexer adoption, fold vs list, `flowAvailable`). | an | — |
| S06 | P0-05 | `data-scientist` | C | `docs/plans/connected-code-graph-eval-protocol-ml.md` | ML §4, §8; AC-5, AC-6, AC-7, AC-24, AC-25; S03's locator and id form; §2 D8, §6 | P0(f) is complete, as listed in §4.3. That includes the **development-contact rule** for anchors the slice, the S30 walkthroughs or fixture tests touch, which `data-scientist` decides (M3), and the procedures GC's tooling implements (M5). | an | S03 |
| S07 | P0-06 | `graph-dba` | Doc | `claude/graph-dba/falkordb-quirks.md` | graph note §1 (V2, V5, V9, V10), §6 (V16–V18) | Seven dated entries, each with its verifying command, folded into the right sections. No existing entry contradicted, or each contradiction resolved in place. | tc | — |
| S08 | P0-07 | `devops` | D | `cpg/sandbox/offline-run.sh`, `cpg/sandbox/forbidden-flags.txt`, `cpg/sandbox/allowed-flags.txt`, `cpg/sandbox/tests/test_sandbox.sh` | S04; image from S02 | The seven sandbox tests of §4.1 pass. The escape-axis probe and the flag sweep run on both backends. | an | S02, S04 |
| S09 | P0-08 | `tdd-engineer` | CC | `cpg/pyproject.toml` (**every** P0 dependency declared here, in extras: `dev`, `live`, `spike`), `cpg/README.md`, `cpg/.gitignore`, `cpg/ccg/__init__.py`, `cpg/ccg/ids.py`, `cpg/tests/test_ids.py` | S03 §ids; §2 D2, D10 | `pytest -m "not live"` is green. The id golden vectors in S03 reproduce. | an | S03 |
| S10 | P0-09 | `tdd-engineer` | CC | `cpg/ccg/schemas/bundle-v1.schema.json`, `cpg/ccg/bundle.py`, `cpg/tests/test_bundle.py` | S03 | The schema encodes S03 exactly. Writes are deterministic (byte-identical on re-write). Unknown fields and missing required fields are rejected. | an | S09 |
| S11 | P0-10 | `graph-dba` | M | `cpg/spikes/vertical-slice/{extract.sc, normalize.py, run_extract.sh}` | S03, S10; `skills/joern-cpg/scripts/build-cpg.sh` | Bundles for `falkor-chat/server` and `salesperson/` at a pinned commit, built in the sandbox, validate against `bundle-v1`. Stage timings are recorded. Two runs are byte-identical apart from header timestamps. | (in S15) | S08, S10 |
| S12 | P0-11 | `coder` | M | `cpg/spikes/vertical-slice/{assemble.py, tests/test_assemble.py}` | S11 bundles; proposal §3.3–§3.4 | SL-1 and SL-2 pass on the real bundles. Unit tests on hand-made mini-bundles pass. | (in S15) | S11 |
| S13 | P0-12 | `coder` | M, **R** | `cpg/spikes/vertical-slice/{load.py, impact.py, tests/test_live_slice.py}` | S12 output; graph note §3.4, §3.5, §6.2–§6.4 | SL-3–SL-6 pass against `ccg_p0slice` on the dedicated instance. | (in S15) | S12, S01 |
| S14 | P0-12 | `coder` | Doc | `docs/test-reports/connected-code-graph-p0-slice-report.md` | S11–S13 | Every SL criterion is recorded pass/fail with evidence. Densities, timings and the determinism diff are included. Every schema field changed or added during the slice is listed (SL-7). The report also records the `Link` count by status, and the BFS's peak frontier size and state count under field paths (M4). It also records the **hop count of every `Link` chain** the slice finds (`LIFTED_VIA` plus `BOUND_BY` hops), which is the system's own chain length on real data at P0 (Pass 2 nit, M4). It lists every proving-repo anchor the slice touched, for S06's development-contact rule (M3). | — | S13 |
| S15 | gate | `analyst` + `graph-dba` | gate | verdict in `docs/reviews/connected-code-graph-p0.md` (a `## Pass N` section); `teco` records freeze or provisional in the ledger | S14 | Verdict given. Freeze iff SL-1–SL-7 all pass (§2 D5). | — | S14 |
| S16 | P0-13 | `architect` | C | *(only if S15 shows a schema change)* `docs/plans/connected-code-graph-bundle-spec2.md` (`Supersedes:` the bundle spec). `tdd-engineer` then updates S10's files | S14 | The successor encodes every SL-7 delta. S10's tests are updated and green. | an | S15 |
| S17 | P0-14 | `graph-dba` | CC, **R** | `docs/plans/connected-code-graph-p0-graph.md` §1 "Graph model v1"; `cpg/ccg/store/ddl_v1.cypher`; `cpg/tests/test_ddl_live.py` | graph note §3.2, §3.4, §6; S14 | The model v1 statement, with deltas, is written. The DDL applies to an empty key, `db.indexes()` matches it, and the post-load constraints reach `OPERATIONAL`. | an | S15, S23 |
| S20 | P0-15 | `graph-dba` | M, **RX** | `cpg/spikes/store/{synth.py, bfs.py, bench_reach.py}`; p0-graph note §2 | graph note Q7, V8, §6.2, §6.4 | BFS vs `algo.SSpaths` is benchmarked on a 1,000-file synthetic v2 graph. The 20-repeat FR-19 diff is run for both. The decision is recorded by the rule in §4.5. | an | S01, S09 |
| S21 | P0-15 | `graph-dba` | M, **RX** | `cpg/spikes/store/swap_probe.py`; p0-graph note §3–§4 | graph note Q4, V2, V10, G13 | `MULTI`/`RENAME` rollback behaviour and the telemetry cap/disable are probed and recorded, each with commands and outputs. | an | S20, S09 |
| S22 | P0-16 | `graph-dba` | M, **RX** | `cpg/spikes/store/{acl_probe.py, lock_replay.py, tests/test_lock_replay.py}`; p0-graph note §5 | proposal §3.5, §3.8; graph note §6.4, V5, V16–V18 | The ACL allow/deny matrix for the four users plus the lock grants is verified. The lock/replay fault tests pass, including mid-swap expiry. `TIMEOUT_MAX`'s effect on load writes and on `GRAPH.CONSTRAINT` is probed. `TIMEOUT_MAX` and the lock ttl are sized by the rule in §4.5, and labelled **P0-scale defaults that P1 re-derives** (m8). | an | S21, S09 |
| S23 | P0-16 | `graph-dba` | Doc | p0-graph note §6 decisions table; `claude/graph-dba/falkordb-quirks.md` (new quirks) | S20–S22 | Every C-spike decision has one row with its measurement. New quirks are folded in. | an | S22, S07 |
| S24 | P0-17 | `devops` | D | `cpg/infra/toolchain/Dockerfile` (ground-truth layer) | S05 §truth sources | Every truth tool S05 names runs in the image. | an | S02, S05 |
| S25 | P0-18 | `graph-dba` | M | `cpg/spikes/frontend-fitness/{corpora.yaml, run_fitness.py, measure.sc, README.md}` | S05, S08, S24 | Corpora pinned by SHA (D4). The harness runs both configurations per language. The seeded call-site sample is drawn and written out **as one file per language**. | ds | S05, S08, S24, S09 |
| S26 | P0-19a…g | `qa-engineer`, **one fresh instance per language** (seven dispatches, parallel) | GT | `cpg/spikes/frontend-fitness/labels/<lang>.jsonl`, one per dispatch | S25 sample for that language; that corpus's sources only | Every sampled call site is labelled with its true callee(s) by a labeller that never sees Joern's or the indexer's answer (§4.6). Labels are appended as they go, so a killed labeller is resumed in place (M6). | ds | S25 |
| S27 | P0-20 | `graph-dba` | M | `cpg/spikes/frontend-fitness/results/*.json`; `docs/test-reports/connected-code-graph-p0-frontend-fitness-report.md` §a1 | S25, S26 | The a1 metrics for 7 languages × {off, on} are recorded with intervals. The indexer and fold/list decisions per language follow S05's rules. | an + ds | S26 |
| S28 | P0-21 | `graph-dba` | M | fitness report §a2; `results/projection-*.json` | S11 `extract.sc`; S05 `flowAvailable` rule | Per language, the report records: projection density (nodes, edges, MB per file); build time; **symbol and call-site density per file** (graph note Q3: the Java/C# call-site density caveat; M4). `flowAvailable` per language is decided. | an + ds | S27, S15 |
| S30 | P0-22 | `architect` | C | `docs/plans/connected-code-graph-rule-pack-spec.md` | S03 (+S16), S14 bundles, S27 decorator/annotation shapes; proposal §3.3–§3.4 | Format v1, five paper packs walked through, infrastructure-signature pack v0, and the severity rule table (§4.2). Every proving-repo anchor the walkthroughs touch is listed (M3). | an (+ ds consult) | S15, S27, S16 if it runs |
| S31 | P0-23 | `tdd-engineer` | CC | `cpg/ccg/schemas/rule-pack-v1.schema.json`, `cpg/tests/test_rule_pack_schema.py`, `cpg/packs/{fastapi,fetch,fastmcp}/pack.yaml` | S30 | Schema = S30. The three packs validate. Each negative example in S30 is rejected. | an | S30 |
| S32 | P0-24 | `tdd-engineer` | CC | `cpg/packs/{kafka,paper-inhouse-bus,infra-signatures,severity-rules}/pack.yaml`; extend `test_rule_pack_schema.py` | S30 | All seven packs validate. | an | S31 |
| S33 | P0-25 | `architect` | C | `docs/plans/connected-code-graph-answer-spec.md` | S03 (+S16), S17, S20 decision, S06 hit rule; proposal §3.6 | Answer schema v1 is specified: interpretation, recipe (algorithm id and version per S20), facts, folded groups, evidence hops, staleness, stamp, provenance (§4.2). | an (+ ds consult) | S17, S20, S06, S16 if it runs |
| S34 | P0-26 | `tdd-engineer` | CC | `cpg/ccg/schemas/answer-v1.schema.json`, `cpg/ccg/answer.py`, `cpg/tests/test_answer.py`, `cpg/tests/golden/answer-ac1.json` | S33; S13 output | The canonical bytes are deterministic. The golden AC-1 answer, hand-built from the slice's BFS result, validates and round-trips. | an | S33, S13, S16 if it runs |
| S35 | P0-27 | `qa-engineer` | C | `docs/test-plans/connected-code-graph-fixtures.md` | S03 reason codes; S06 planted-shape counts; proposal §5.1 | Every planted item is listed with its kind, expected status, reason, the phase it becomes gateable in, and the assertion it serves (§4.7). | an + ds | S03, S06 |
| S36 | P0-28 | `qa-engineer` | GT, D | `cpg/fixtures/system1/fx-svc/**`, `expected/fx-svc.yaml` | S35 | The repo matches S35's fx-svc rows. | (in S39) | S35 |
| S37 | P0-29 | `qa-engineer` | GT, D | `cpg/fixtures/system1/fx-web/**`, `expected/fx-web.yaml` | S35 | The repo matches S35's fx-web rows. | (in S39) | S35 |
| S38 | P0-30 | `qa-engineer` | GT, D | `cpg/fixtures/system1/fx-lib/**`, `expected/fx-lib.yaml` | S35 | The repo matches S35's fx-lib rows. | (in S39) | S35 |
| S39 | P0-31 | `qa-engineer` | D | `cpg/fixtures/system1/expected/cross.yaml`, `cpg/fixtures/materialize.py`, `cpg/ccg/schemas/fixture-manifest-v1.schema.json`, `cpg/tests/test_fixtures.py` | S36–S38, S08, S10 | The materializer is deterministic (same commit SHAs on every run). Every locator resolves. The manifests validate. The `live` test parses each repo in the sandbox. | an + ds | S36, S37, S38, S08, S10 |
| S40 | P0-32 | GC (`qa-engineer`) runs; `data-scientist` checks | GT | `cpg/eval/{EVAL_COMMIT, eval-manifest.yaml, kit-manifest.json}`; master kit and per-role `BRIEF.md`s outside the repo (§6.2) | S06; tooling from S48 | Commit pinned. The kit is exported, content-checked (§6.2) and hashed. A brief exists per role and stratum. The flow-anchor sample is drawn with a recorded seed. | an + ds | S03, S06, S48 |
| S41 | P0-33a…d | `teco` runs the launcher; headless KA session, one per stratum (§6.1) | GT | `roles/KA-<stratum>/out/KA.jsonl`, appended as it goes | role root (§6.2) | The provider-first key covers its stratum. The session's audit is clean (§6.5). | (in S45) | S54 |
| S42 | P0-34a…d | `teco` runs the launcher; headless KB session, one per stratum | GT | `roles/KB-<stratum>/out/KB.jsonl`, appended as it goes | role root | The consumer-first key covers its stratum. The audit is clean. | (in S45) | S54 |
| S43 | P0-35 | `teco` runs the launcher; headless QW session | GT | `roles/QW/out/QW.jsonl` | role root | The bilingual impact questions are written natively, with gold, in the counts S06 sets. The audit is clean. | (in S47) | S54 |
| S43a | P0-35a | GC (`qa-engineer`) | GT | `QW-nogold.jsonl`, placed in the AN1 and AN2 role roots | S43 | `strip_gold.py --verify`: every question is present and no gold field is. | ds | S43 |
| S44 | P0-36 | `teco` runs the launcher; headless AN1 ∥ AN2 sessions | GT | `roles/AN{1,2}/out/AN{1,2}.jsonl` | S43a | Each annotator has independently labelled every question ambiguous or unambiguous. The audit is clean. | (in S47) | S43a |
| S45 | P0-37 | GC (`qa-engineer`) runs; `data-scientist` adjudicates near-misses by pair id and checks | GT | `~/.local/share/ccg-eval/<E>/work/{audit,match,nearmiss,capture}.json` | S41, S42 outputs; `runs/`; transcripts; tooling from S48–S49 | Every session in the run index is audited and clean, or its output is voided and redone. Items are matched. N̂ and completeness are computed per stratum. The union and the spot-check sample are drawn (§6.4). | ds | S41, S42, S49 |
| S46 | gate | stakeholder (via `teco`/`tico`) | gate | marks on the spot-check sample | S45 sample; PT gold sample from S43 | Each sampled item is marked correct or incorrect, with a note on each incorrect one (decision S36; ML §8.3). | sh | S45, S44 |
| S47 | P0-38 | GC (`qa-engineer`) runs; `data-scientist` checks | GT | `cpg/eval/dev/*.jsonl`, `cpg/eval/heldout-manifest.json`, `docs/test-reports/connected-code-graph-p0-ground-truth-report.md` (repo); `~/.local/share/ccg-eval/<E>/heldout/*.jsonl` (outside the repo, plaintext, D11) | S45, S46, S44; S06's development-contact rule | Corrections applied. Split by anchor (seeded) after the development-contact rule. Held-out written outside the repo; its hash manifest committed (D11). The report carries counts, N̂, completeness, manifest hashes and held-out anchors per stratum. The mechanical leak check finds **no held-out content** in the report (§6.4). | an + ds + sh (counts only) | S46 |
| S48 | P0-41 | `qa-engineer` (tooling author; non-builder, never sees keys) | D | `cpg/eval/tools/{export_kit.py, strip_gold.py, audit.py}`, `cpg/tests/test_eval_tools.py`, `cpg/tests/fixtures/headless/` (fixtures built **from S55's real capture**) | S06 procedures; §6.5; S55 capture | Each tool passes its tests. `audit.py` passes on stream-json and transcript fixtures shaped like S55's **real** capture: it flags an absolute repo read, a relative `../` read, a sibling-root read, an MCP call, a disallowed tool, a write outside `out/` and an init tool set unequal to `--tools`; it does not flag a legitimate in-root read; it flags a stream/transcript mismatch and a missing transcript; and on a **two-attempt session** (start, kill with a truncated last line, resume) it consumes both attempt logs with **no false violation** (stopping item 3). | an | S06, S09, S55 |
| S49 | P0-42 | same author as S48 | D | `cpg/eval/tools/{match_keys.py, capture.py, split_manifest.py}`; extend `cpg/tests/test_eval_tools.py` | S06 procedures | On synthetic keys, the tools reproduce hand-computed matches, estimates (two-list and S06's multi-list) and a seeded split. `split_manifest.py` writes the held-out files and a manifest whose hashes `--verify` re-checks, and `--verify` fails on a one-byte change. `split_manifest.py --check-report` catches a planted held-out locator. | an + ds | S48 |
| S52 | P0-43 | `cobb` | D | `cpg/eval/headless/agents.json` (four definitions: `ccg-key-writer-provider`, `ccg-key-writer-consumer`, `ccg-question-writer`, `ccg-annotator`), `cpg/eval/headless/settings.json` (the `PreToolUse` path guard and nothing else), `cpg/eval/headless/guard-paths.sh` | §6.1–§6.3; S06's item schema | Each definition has its role's method and output contract, and **no** learning-capture, KB, CPG or repo instructions. The two key-writer prompts differ in strategy and wording, not only in a flag. The guard denies any file-tool path outside the cwd root, and any write outside `out/`, and has its own tests. The definitions are **not** deployed to `~/.claude/agents/`; they load only through `--agents`. Whether `claude/AGENTS.md`'s agent-catalog rules apply to these harness-only definitions is `cobb`'s call, recorded in the step. | an | S06 |
| S53 | P0-44 | `devops` | D | `cpg/eval/headless/run-writer.sh` (verbs `prepare-bin`, `prepare`, `start`, `resume`, `status`), `cpg/eval/headless/README.md` | §6.2–§6.3; S52 | (1) `prepare-bin` copies `~/.local/share/claude/versions/2.1.312` to `$HOME/.local/share/ccg-eval/bin/claude-2.1.312` and records its SHA-256, and **`env -i HOME="$HOME" PATH=/usr/bin:/bin LANG=C.UTF-8 timeout 10 "$HOME/.local/share/ccg-eval/bin/claude-2.1.312" --version` exits 0**, run by `devops` as part of done (stopping item 1). (2) `start` and `resume` build argv from **one** `build_argv` function, and their dry-run outputs differ **only** in `--session-id <u>` versus `--resume <u>`, the prompt, and the stream-log name (stopping item 2). (3) Each attempt writes a new `runs/<uuid>.<n>.stream.jsonl`, never an existing one (stopping item 3). (4) The argv carries `--tools Read,Grep,Glob,Write,Edit` and `--disable-slash-commands` (stopping item 4). (5) The run index records the CLI version, the binary's SHA-256 and the SHA-256 of `settings.json` and `agents.json` per attempt, and `resume` refuses if any of them changed. `prepare` is idempotent, and the script is shellcheck-clean. | an | S52 |
| S54 | P0-45 | `teco` runs the launcher; GC runs `audit.py`; `devops`/`cobb` fix on failure | gate | rehearsal result in `docs/test-reports/connected-code-graph-p0-ground-truth-report.md` §Rehearsal (written by GC) | S52, S53, S48, S40 | §6.3's planted-breach rehearsal passes in full, including the init-event checks and the **kill-and-resume case** (stopping item 5). **No real writer starts before it does.** | an (stakeholder informed of the result) | S52, S53, S48, S40 |
| S55 | P0-46 | `teco` runs the launcher (`run-writer.sh start capture`) | GT (fixture) | `cpg/tests/fixtures/headless/capture.{1.stream.jsonl,transcript.jsonl}`, both scrubbed of the account id; a short note in `cpg/eval/headless/README.md` | S53 | One trivial real session, role `capture`. It runs in a **dummy** role root that `prepare` builds from a one-file dummy kit with no repo content, so S55 needs no S40 kit and no cycle forms. A one-line brief ("read `kit-src/README.md`, write `out/done.txt`") is launched through the real argv. It settles three things before S48 relies on them (P2-4): (a) the real stream-json and transcript shapes; (b) whether `--verbose` is required; (c) that OAuth works under `--restricted`. If (c) fails, it stops the chain and goes back to the stakeholder, because the cost acceptance assumed plan usage. | an | S53 |
| S50 | P0-39 | `architect` | Doc | `docs/plans/connected-code-graph.md` v3.4 (revised in place, one dated line) | every results note | §3.2's table restated as measured. `flowAvailable` and the indexer decisions recorded. §3.5's sizes restated as measured at P0 scale. §6.2's P0 items marked settled, each with a pointer to its settling step. | an | all gates |
| S51 | P0-40 | `teco` | Doc | `docs/HISTORY.md`, `docs/BACKLOG.md` (C-1001), root `AGENTS.md` `cpg/` bullet (one clause) | §10 | §10's list is applied. | tc | S50 |

**Parallel and sequential structure.**
- **Wave 0, start together, no dependencies:** S01, S02, S03, S04, S05 and S07. These are six
  dispatches. S03 and S04 are separate `architect` dispatches (m2).
- **Wave 1:**
  - S08 waits on S02 and S04. S09 → S10 wait on S03. S06 waits on S03 (M2).
  - P0-15 (S20 → S21) waits on S01 and S09.
  - S24 waits on S02 and S05.
- **Wave 2:**
  - S11 waits on S08 and S10. P0-16 (S22 → S23) follows P0-15.
  - S25 waits on S24, S08 and S09.
  - S35 waits on S06. S52 (`cobb`) waits on S06. S53 (`devops`) follows S52, and S55 (the capture) follows S53. S48 waits on S06, S09 and S55; S49 follows S48.
- **Wave 3:**
  - S12 → P0-12 (S13 → S14) → S15 is the slice chain.
  - The fitness chain is S26 (seven per-language dispatches in parallel) → S27.
  - S36 ∥ S37 ∥ S38 → S39.
  - **Ground truth (headless writers, §6):**
    - S40 waits on S48.
    - **S54, the planted-breach rehearsal,** waits on S40, S52 and S53. No writer starts before it
      passes.
    - Then the headless sessions S41 (four KA), S42 (four KB) and S43 (QW) run, **at most two at
      a time** (§6.6).
    - S43 → S43a → S44. S45 waits on S41, S42 and S49. Then S46 → S47.
- **Wave 4, after S15:**
  - S16 runs only if the schema changed. If it runs, S30, S33 and S34 also wait on it.
  - S17 also waits on S23. S28 also waits on S27. S30 also waits on S27.
  - S33 waits on S17 and S20.
- **Wave 5:** S31 → S32 and S34.
- **Wave 6:** S50 → S51.

**Critical path:** S02 → S08 → S11 → S12 → S13 → S14 → S15 → S17 → S33 → S34 → S50.

Two chains run beside it and must also finish before S50:
- **Fitness:** S05 → S24 → S25 → S26 → S27 → S28.
- **Ground truth:** S03 → S06 → S52 → S53 → S55 → S48 → S40 → S54 → S41/S42 → S45 → S46 → S47. The two-at-a-time limit makes the eight KA/KB sessions about four session-lengths
  of wall-clock, so this chain may well be the longest in P0.

S46 is a human step and is the one wait no agent can shorten.

## 4. Step details

### 4.0 What each P0 deliverable is

| P0 item (proposal §4) | Nature | Steps | Durable artifacts |
|---|---|---|---|
| (a) frontend fitness | **Measurement spike** (a1 + a2, §2 D3), protocol first | S05, S24–S28 | `docs/plans/connected-code-graph-p0-ml.md`; `docs/test-reports/connected-code-graph-p0-frontend-fitness-report.md`; `cpg/spikes/frontend-fitness/{corpora.yaml, results/*.json, labels/*.jsonl}` |
| (b) vertical slice + schema/model freeze | **Contract** (bundle spec + schema code) **+ measurement spike** (slice) | S03, S09–S17 | `docs/plans/connected-code-graph-bundle-spec.md` (+ `-spec2` if changed); `cpg/ccg/{ids.py, bundle.py, schemas/bundle-v1.schema.json, store/ddl_v1.cypher}`; `docs/test-reports/connected-code-graph-p0-slice-report.md`; `docs/plans/connected-code-graph-p0-graph.md` §1 |
| (c) store spikes | **Measurement spike** on the dedicated instance (**durable** bring-up script) | S01, S20–S23 | `cpg/infra/ccg-falkordb.sh`; `docs/plans/connected-code-graph-p0-graph.md` §2–§6 |
| (d) rule-pack format v1 | **Contract** (spec + schema + paper packs) | S30–S32 | `docs/plans/connected-code-graph-rule-pack-spec.md`; `cpg/ccg/schemas/rule-pack-v1.schema.json`; `cpg/packs/*/pack.yaml` |
| (e) egress broker contract + offline sandbox | **Contract** (broker) **+ durable code** (sandbox, image) | S02, S04, S08 | `docs/plans/connected-code-graph-egress-spec.md`; `cpg/sandbox/*`; `cpg/infra/toolchain/*` |
| (f) evaluation protocol | **Contract** (method document) | S06 | `docs/plans/connected-code-graph-eval-protocol-ml.md` |
| (g) fixture system + held-out strata + bilingual cells | **Durable code + ground-truth data** | S35–S47 | `docs/test-plans/connected-code-graph-fixtures.md`; `cpg/fixtures/**`; `cpg/eval/**`; `docs/test-reports/connected-code-graph-p0-ground-truth-report.md` |
| (h) canonical answer schema | **Contract** (spec + schema + serializer) | S33, S34 | `docs/plans/connected-code-graph-answer-spec.md`; `cpg/ccg/{answer.py, schemas/answer-v1.schema.json}` |

### 4.1 Environment and sandbox (S01, S02, S08, S24)

- **S01:** the script takes `up | status | down | reset-p0`.
  - `reset-p0` is the only destructive verb. It refuses unless the target container is named
    `ccg-falkordb` and the port is 6390. It is approval-gated as `devops` destructive ops are.
  - The admin password is read from the file in §2 D7. The script creates that file on first `up`
    if it is absent, and never prints it.
  - No ACL users are created here. S22 probes them; the P1 loader creates them.
- **S02:** pin every version in the Dockerfile. Joern v4.0.579 comes from its GitHub release with
  a SHA-256 check; it is not copied from `~/joern`. The tag is a content hash of the build inputs,
  as in `cypher-mcp/build.sh`. The image is large, so `build.sh` prints its size, and S02's
  results go in `cpg/infra/toolchain/README.md`. That is a separate file from S01's
  `cpg/infra/README.md`, so S01 and S02 can stay parallel (m3).
- **S08 tests**, which must pass for both backends unless a test says otherwise:
  1. **The escape-axis probe.** Every axis must be closed, or the backend refuses to start. No
     request is ever sent to a real external host: the IP check is a connect attempt that must
     fail locally.
     - **IP:** a TCP connect to `1.1.1.1:443` fails, and so does one to the host's LM Studio port.
       WSL's mirrored networking makes `localhost` reach Windows.
     - **DNS:** `getent hosts example.com` fails.
     - **Host Unix sockets:** `/var/run/docker.sock` and any other `*.sock` under `/run` and
       `/var/run` are absent.
     - **WSL interop exec:** a harmless PE binary (`whoami.exe`, staged by the test into the
       output directory) fails to execute.
     - **Mounted Windows paths:** `/mnt/*` and `/run/WSL` are absent.

     For the namespace backend, the test asserts the refusal while `WSLInterop` is enabled. Where
     interop is disabled, it asserts the full probe instead.
  2. Each forbidden flag is refused **before** launch, in each of three forms: `--flag`,
     `--flag=value`, and the flag passed through after `--`. A refusal is exit code 64 plus an
     audit line with `refused: true`.
  3. `joern-parse` of a one-file Python project and a one-file Java project succeeds in under 120 s.
     This guards against JVM local-hostname resolution stalling with no network.
  4. Every invocation appends one audit record in S04's format: argv, backend, image or host
     toolchain, `network: none`, exit code.
  5. The container backend mounts the source `ro,noexec` and the output directory `rw,noexec`,
     and nothing else from the host.
  6. **The flag sweep (m4).** For every installed frontend, the test greps `--help` for
     `fetch|download|url|dependenc|resolver|gradle`. It fails on any matching flag that is in
     neither `forbidden-flags.txt` nor `allowed-flags.txt`. Each entry in `allowed-flags.txt`
     carries a one-line reason. This runs before S25 uses the list, and again on any Joern
     upgrade.
  7. The sandbox environment is empty except for an explicit pass-list, and
     `ENABLE_DEPENDENCY_RESOLVER_V2` is never on that list.
- **S24** adds a layer with whatever truth tools S05 names, for example a typed indexer per
  language or the language build tool that runtime tracing needs. It touches the same file as S02,
  so it runs strictly after S02.

### 4.2 Contracts (S03, S04, S30, S33)

**S03, the bundle spec (v1 candidate)**, must contain:
- **Layout:** one JSONL file per unit. Line 1 is a header record: `bundleSchemaVersion`,
  `unit {repoId, rootPath, lang}`, `builtCommit`, `builtTree`, `builtAt`, tool versions (Joern,
  frontend, normalizer), the sandbox audit id, and `egressMode ∈ {offline, controlled}`. The other
  records follow in a fixed type order, sorted by id within each type.
- **Record types:** `file`, `symbol` (with `complexity`, `fingerprint` and `lastChanged*`),
  `param`/`ret`/`cond` points, `callsite`, `valueSet` (per call-site argument: constants,
  concatenation parts, template parts, cross-file references `{ref: qname}`, or `unknown` with a
  reason), `data {from, to, fromPath, toPath, status, reason}`, `controlledBy`, `decorator`/
  `annotation` (name, plus argument value sets), `import`, `dependency`, `testMarker`,
  `configEntry`, and `launchFact`.
- **Reserved records:** `secretSighting` is reserved for P5, so that later phases add records
  without breaking v1.
- **Canonical tuples and ids** follow §2 D10, plus at least five golden vectors (including
  whitespace-edged, non-ASCII and over-4,096-byte inputs), which S09 tests:

  | Record | Canonical tuple |
  |---|---|
  | unit | `(repoId, rootPath, lang)` |
  | file | `(repoId, path)` |
  | symbol | `(lang, repoId, qname[, signature for Java, Kotlin, C#, C++])` |
  | callsite | `(symbolId, calleeName, ordinal)` |
  | point | `(symbolId, kind, index or ordinal)` |
  | contractPoint | `(kind, keyNorm)` |

- **Source-derivable `qname` grammar (M2).** For each FR-1 language, S03 gives a grammar that a
  reader of the source alone can produce, with no Joern. It covers modules or packages, classes,
  nested and local functions (for example a handler defined inside `build_storefront_router()` in
  `storefront_api.py`), anonymous functions and lambdas, overload signatures, and TS/JS module
  paths. Each frontend's normalizer must produce the same string. S11 checks this on the slice
  units, and a mismatch is an SL-7 delta.
- **`repoId`.** S03 fixes the `repoId` of the proving repo, a manifest name such as
  `graphmind-ai-lab`, and the rule for any other repo: the manifest's repository name, normalized.
- **`keyNorm` rule.** Per contract kind, S03 states how a key is normalized: HTTP is
  `<VERB> <path>` with the path template normalized; MCP is the tool name; topic, table and label
  each have their own form. Keys and gold use exactly this form.

- **Paths and lines:** paths are repo-relative POSIX, and lines are 1-based. The normalizer
  rewrites the parse-root-relative `FILENAME` that `cpg-model.md` describes into a repo-relative
  path.
- **Reason-code registry:** the closed, extensible list from proposal §3–§3.8, each with the phase
  that first emits it.
- **Freeze condition:** cites §2 D5 and S15.
- **Mapping:** every record is mapped to graph note §3.2/§6 labels and edges. `graph-dba` checks
  the mapping in the gate.

**S04, the egress spec**, must contain:
- **Broker interface.** Every outbound attempt declares its purpose, endpoint, payload class and
  the codebase set it serves, and gets back allow or deny with the rule that decided.
- **Policy resolution:** codebase default, then run override, then sticky local-only, then
  system-wide local model calls (FR-9).
- **Allow-list format** in the system manifest.
- **Broker audit-record schema:** `{ts, component, purpose, endpoint, decision, rule, codebases,
  payloadClass, bytes}`.
- **Sandbox contract:** the backends of §2 D6.
  - **Escape axes:** for each backend, which escape axes it closes and how: IP, DNS, host Unix
    sockets, WSL interop exec, mounted host and Windows paths, and the environment.
  - **Refusal:** each backend's refusal condition (the namespace backend refuses while interop is
    enabled).
  - **Flags:** the forbidden and reviewed-allowed flag lists, with the classification rule.
  - **Audit:** the argv audit record.
- **Controlled-download path for hosted-default codebases:** configured registries only, and no
  import-name service unless explicitly allow-listed.
- **AC-8 traceability:** a table from each AC-8 assertion (proposal §5.3) to the audit fields that
  prove it.
- **P1 handoff:** what P1 implements (the broker process and the logging proxy).

**S30, the rule-pack spec**, must contain:
- **Format v1:**
  - the pack header (id, version, kind, `vocabVersion`);
  - rules of `match`/`extract`/`emit`, with the matchers `exact`, `path-template`, `glob`/`prefix`
    and `regex`;
  - declarations for mount composition (conditional mounts included), role-lifting seeds, embedded
    languages, URL bindings, launch facts and framework defaults;
  - the read-only escape hatch. S30 picks its query form under three constraints: read-only, over
    one unit's bundle, deterministic.
- **Five paper packs:**
  - FastAPI with mount composition;
  - fetch with a wrapper, lifting seeded on `fetch` `a0` with the verb at `a1.method`;
  - FastMCP plus URL binding, with the uvicorn default 8000, the env default
    `AGENT_TEAM_PORT:-8200` and the Dockerfile `--port`;
  - Kafka, including a topic read from config;
  - an invented in-house bus.
- **Walkthroughs:** each pack is walked through rule by rule. FastAPI and fetch are walked against
  S14's **real** bundles, and the emitted facts must equal the slice's hard-coded recognizer
  output. The other three are walked against snippets with the bundle records S03 would produce.
- **Language expressiveness:** the format must be able to express the decorator, annotation and
  attribute shapes S27 measured for all seven languages, or S30 records the gap.
- **Infrastructure-signature pack v0 (a §6.2 item):** signature classes from ML §8.3 (connection
  refused, DNS, timeout, OOM kill, TLS, third-party exception classes), with concrete patterns per
  runtime: Python, Node, JVM, .NET and Go.
- **Severity rule table for non-dependency findings (a §6.2 item):** four levels with a stated
  rule each, for injection-type, missing authorization, risky configuration and secrets. Secrets
  distinguish present-at-head from removed-within-window. `security-expert` may advise on request
  (proposal §6.2), but it is not a gate.
- **Negative examples:** malformed rules the schema must reject, which S31 tests.

**S33, the answer spec**, must contain:
- **Top-level fields:** `answerSchemaVersion`, `interpretation` (structured request, ambiguity
  options, RCA marks, and plain-language provenance: model fingerprint, prompt hash, cache key),
  `recipe` (query set plus algorithm id and version), `stamp` (`snapshotId`, `verdictLogOffset`,
  `verdictLogHash`), `staleness` (per unit: built commit and time, `fresh`/`stale-kept`/
  `not-mapped`, reason), `items`, `groups`, and `depthBound` (the bound, plus "stopped at d with k
  unexplored").
- **Recipe:** the algorithm id is S20's choice.
- **Items:** id, status, reason, direct/indirect, distance, repo/component grouping keys, and an
  evidence chain of hops. Each hop carries `{kind, path, line, unit, repo, edgeType, status,
  reason}`, plus `Link` expansion into `BOUND_BY`/`LIFTED_VIA`.
- **Groups:** folded groups, for "other possible callees" and runtime-built collapse, **always
  enumerate every member**, each with its own chain (ML M2, P3-1, S06's hit rule).
- **Ordering:** the deterministic ordering rule, plus the canonical-bytes rule (the same canonical
  JSON as §2 D10).
- **Versioning:** the versioning policy.

### 4.3 Method notes (S05, S06)

**S05, the P0(a) protocol**, sets:
- **Corpora:** the criteria and exclusions of §2 D4.
- **Call-site sample** per language: seeded, and stratified by resolved vs unresolved and by
  call-site kind. This plan suggests at least 150 sites per language. Zero misses on 150 gives a
  Wilson lower bound of about 0.975, and S05 decides.
- **Truth source** per language: a typed indexer, runtime traces of the corpus's own tests, or
  blind hand labels (S26). Where an indexer is the truth, a hand-labelled subsample validates it.
- **Metrics:**
  - fallback recall, with its Wilson lower bound;
  - resolved-call precision;
  - fan-out median and p90;
  - line fidelity: the fraction of sampled symbols whose `lineStart`/`lineEnd` match the source;
  - parse success: files parsed ÷ files;
  - value-set recovery: the fraction of string-constant arguments recovered as constants.
- **Decision rules,** stated once:

  | Decision | Rule |
  |---|---|
  | Indexer adopted (correctness) | Fallback recall below ~0.98 observed (ML §3.6) or resolved-call precision below 0.95 (ML §8.1) |
  | Indexer adopted (cost) | p90 fan-out above about 20 |
  | Fold vs list | Precision ≥ 0.95 folds, below lists (proposal §3.7) |
  | `flowAvailable` | A criterion S05 defines over golden flow sets of small methods per language. This plan suggests ≥ 20 methods per language and DATA/CONTROLLED_BY recall ≥ 0.9, plus projection success on ≥ 95% of methods. |

- **Both configurations:** every measurement runs fetch-off (sandbox) and fetch-on. The
  difference is reported as the K11 gap.

**S06, the evaluation protocol (P0(f))**, sets everything in the S06 row of §3. In particular:
- **AC-5 at P1:** the held-out set over P1's strata (HTTP, MCP, scripts, flow-shaped) is at least
  75 items pooled and at least 8 per kind (ML §4.7). It is split from the union of the two keys
  by anchor, stratified, with a recorded seed. The dev remainder goes to pack authors.
- **Capture–recapture:**
  - the matching rule, which is mechanical first, on normalized (stratum, anchor locator,
    impacted locator) where a locator is (repo path, enclosing qname), then adjudicated;
  - the two-list estimator (Lincoln–Petersen or Chapman, to be chosen and justified), and the
    **multi-list estimator** used once a third census pass exists, since Lincoln–Petersen no
    longer applies then (m7);
  - the completeness threshold below which a further census pass is commissioned (§6.4).
- **Flow-shaped stratum:** a shared anchor sample, because capture–recapture needs both agents
  enumerating the same population (finding F6).
- **AC-24 at P1:** the counts per language for the impact cells (ML §8.3: about 25 held-out,
  12–13 per language), the dev share, the ambiguous subset, and the two-annotator rule.
- **Item schema** for keys and questions, in S03's id forms (M2). Each key item also carries the
  writer's stated chain from anchor to impacted element, as an ordered list of hops with a hop
  count. It is reported as a distribution only (M4).
- **Development-contact rule (M3), decided by `data-scientist`.** Some anchors are touched during
  P0 development: by the slice (SL-1, SL-2, SL-4), by S30's walkthroughs, or by fixture tests.
  S06 decides how those anchors are placed in the split: forced to dev, kept eligible, or
  something else, with the reason. S47 applies the rule before its random split, and reports the
  held-out anchors and items left per stratum. Those counts are SQ-P0-1's trigger. The plan
  supplies S06 with the list of touched anchors: S14 and S30 each list theirs, and S35 lists the
  fixture's, which are not proving-repo anchors.
- **Ground-truth tooling procedures (M5).** S06 specifies each procedure precisely enough for GC
  to implement it without method choices of its own, and for `data-scientist` to check its output:
  - kit export and the content check (§6.2);
  - gold stripping;
  - the audit (§6.5);
  - matching, including the near-miss list and pair-id adjudication;
  - estimation;
  - split, the held-out hash manifest, and the report leak check.
- **AC-7 pre-registration record format:** a pack-set hash, the stratum, and the date of first
  contact.
- **Indirect-item rubric:** ML §8.1.
- **Noise sampling procedure,** used from P1 on.

### 4.4 The vertical slice (S11–S15)

- **S11, extraction (`graph-dba`):** `run_extract.sh` stages each unit with `git archive` at the
  pinned commit, as the 2026-10-15 rebuild did. It then runs `build-cpg.sh` and the
  `extract.sc` Joern script through `cpg/sandbox/offline-run.sh`, and times parse, extract and
  normalize separately. `extract.sc` emits:
  - symbols, call sites, decorators and argument value sets;
  - points: `METHOD_PARAMETER_IN` → `param`, `RETURN` → `ret`, a control-structure condition →
    `cond`, and each non-operator `CALL` as a point;
  - **the per-method projection:** for each point's uses, walk `REACHING_DEF` backward inside
    the method through identifiers and operators to the nearest point definitions, and emit one
    `data` edge per (use point, def point). The access path comes from `<operator>.fieldAccess`
    chains, capped at depth 3; deeper paths are truncated and marked *possible*,
    `field-truncated`. Then walk `CDG` from each point's statement to its condition nodes and emit
    `controlledBy`;
  - a per-method projection-time histogram.

  `normalize.py` maps the TS and Python frontends' differences away (repo-relative paths,
  object-literal fields as `a<i>.<field>`) and writes `bundle-v1` JSONL through `cpg/ccg/bundle.py`.
- **S12, assembly (`coder`):** recognizers are hard-coded in this spike. The slice is not a pack
  runtime.
  - **Value sets** resolve across files: module constants such as `API_PREFIX`, and `{ref}`
    entries.
  - **The FastAPI provider** comes from the decorator path plus the `include_router(…, prefix=…)`
    prefix of the mounting call, wherever that call is, with any enclosing condition recorded.
  - **The fetch consumer** is seeded on `a0`, with the verb at `a1.method`.
  - **Role lifting** iterates to a fixpoint over `data` edges.
  - **Linking** is exact plus path-template matching, and produces a `Link` with ordered
    `LIFTED_VIA`.
- **S13, load and impact (`coder`, reservation R-B):** `load.py` follows graph note §3.5 exactly:
  1. indexes on the empty `ccg_p0slice__build_<id>`;
  2. nodes, then edges, as `UNWIND` batches over one connection, with retries off and a socket
     timeout of minutes;
  3. constraints after the load, polled to `OPERATIONAL`;
  4. `:Snapshot` written last;
  5. `RENAME` to `ccg_p0slice`.

  `impact.py` is a minimal level-synchronous BFS. It implements only the §6.2 table rows the
  slice needs: `CALLS ←`, `HAS_* ←`, `DATA ←` path-matched, `VIA ←`/`TO ←` and `PROVIDES →`. It
  keeps a best-predecessor map.
- **S14, the slice report, scored against these criteria:**

  | # | Criterion | Pass when |
  |---|---|---|
  | SL-1 | AC-1 link and chain | Exactly one **confirmed** `Link {kind: 'http'}` runs from the `salesperson/src/api/endpoints.ts` health call site to the `@router.get("/health")` handler inside `build_storefront_router()` in `falkor-chat/server/falkorchat/storefront_api.py`, keyed `GET /shop/api/health`. Its `LIFTED_VIA` chain is, in order: the endpoints.ts call → `apiFetch` (`p0.path`, verb `p0.method`) → `buildUrl` (`p0`) → the `fetch` call (`a0`, verb `a1.method`). Every hop's file and line matches the source at the pinned commit. |
  | SL-2 | Mount composition | All three `/health` providers are present: the storefront one composed to `/shop/api/health` from the prefix in `app.py`; the dev router's with its `dev_surface` condition recorded; and the bare re-registration. Only the storefront one links to the SL-1 consumer. **Counterfactual (m5):** the slice is re-run on a staged copy whose `API_PREFIX` is `"/x/api"`, and the composed key must become `GET /x/api/health`, with the SL-1 link now unmatched. This replaces the v1 text check (grepping `assemble.py` for `/shop/api`), which `'/shop' + '/api'` would defeat. |
  | SL-3 | Load integrity | Every constraint is `OPERATIONAL` and none is `FAILED`. `:Snapshot` is present. `RENAME` is done. |
  | SL-4 | Impact answers | Contract mode from the provider's `ContractPoint` lists the endpoints.ts caller: confirmed, distance 1, with the rendered chain expanding `LIFTED_VIA`. Value mode from a change in `buildUrl` lists `apiFetch` and its endpoints.ts callers through path-matched `DATA`. |
  | SL-5 | Cost and density | For both units, the projected graph is at most 0.7 MB per source file, which is 2× graph note §6.4's upper estimate, by `GRAPH.MEMORY USAGE` ÷ files. The projection stage takes at most 2× the unit's Joern parse time (K7). Nodes and edges per file are recorded. |
  | SL-6 | Determinism | Two full runs from the same commit give byte-identical bundles (header timestamps excepted), identical `Link` sets, and identical BFS answer facts. |
  | SL-7 | Schema accounting | Every field or record added, removed or retyped relative to S03 is listed, and the final run validates against the schema as amended. |

  The two SL-5 bounds are this plan's engineering choice, made so that the freeze has a number to
  hold to. Exceeding either one means the projection premise, "a projection, not a copy" (§3.14),
  needs rework before freezing. The reviewer may challenge either number at S15.
- **S15, the gate.** `analyst` reviews S11–S14, and `graph-dba` checks SL-3 and SL-5.
  - **All pass:** v1 freezes. If SL-7 lists deltas, the freeze applies to the amended schema, which
    S16's successor document records.
  - **Any fail:** v1 stays provisional until P1 exit, and the P1 plan carries the freeze.
  - **Either way:** S17, S28, S30 and S33 proceed. If v1 is provisional, their outputs are marked
    "against provisional v1" and get re-checked at P1 exit.

### 4.5 Store spikes (S20–S23), all on the dedicated instance

- **S20:** `synth.py` generates a 1,000-file graph at graph note §6.4's v2 densities: about 455
  nodes and 2,400 edges per file, so roughly 455k nodes, 2.4M edges and about 0.3–0.35 GB. That
  fits `maxmemory 1536mb` with room for a build copy. It generates hubs from the V13 fan-out
  distribution.
  - **Anchors:** 50 seeded anchors spanning hub and non-hub symbols.
  - **Comparison:** the stateful BFS (S13's operator, extended to the full §6.2 impact column)
    against `algo.SSpaths`. SSpaths takes a single `relDirection`, so it can stand in only for
    single-direction, single-state segments such as call-only reverse reach. S20 benchmarks those
    segments and says so.
  - **Repeat-diff:** each configuration is run 20 times and the `(item, status, distance, chain)`
    sets are diffed.
  - **Decision rule:** adopt SSpaths for a segment **only if** all three hold:
    1. 20/20 repeats are identical, chains included;
    2. its item set equals the BFS's;
    3. p95 answer latency is at least 30% lower at depth ≥ 3.

    Otherwise use the BFS. Per-level read latencies (p50, p95, p99) are recorded for S22.
- **S21:**
  - **Rollback form:** probe `MULTI; RENAME live prev; RENAME build live; EXEC`, (i) cleanly,
    (ii) with `build` absent, and (iii) with a reader in flight. Redis does not roll back a
    transaction on a command error. That is my understanding of Redis semantics and not verified
    here; S21 verifies it. Record whether the form is safe. If it is not, the decision stays "zero
    kept snapshots, rollback = re-assemble last good" (graph note Q4).
  - **Telemetry:** probe the `telemetry{<key>}` stream. Can it be disabled or capped by a module or
    graph config? Does an admin `XTRIM` hold? Does `RENAME` leave it behind (V10)? Record the
    answer to "can telemetry be capped" and the operating rule that follows.
- **S22:**
  - **ACL matrix:** a probe matrix over the four users `ccg_build`, `ccg_verdict`, `ccg_engine` and
    `ccg_raw_p0` × the commands `GRAPH.QUERY`, `RO_QUERY`, `EXPLAIN`, `PROFILE`, `LIST`, `RENAME`,
    `EXPIRE`, `DEL`, `GRAPH.CONSTRAINT`, `INFO`, plus `SET`/`DEL`/`EVAL` on the lock key, a read of
    `telemetry{…}`, and access to a foreign key. Every cell records allowed or `NOPERM`, and must
    match proposal §3.5. The `falkordb-py` constructor must succeed for each user (V17).
  - **`lock_replay.py`** implements proposal §3.8's lock (`SET … NX PX`, a compare-and-delete
    release via `EVAL`), the fsynced JSONL log, the patch-and-advance in one query, and the swap
    replay plus the post-`RENAME` remainder replay.
  - **`test_lock_replay.py` asserts:**
    1. a verdict issued during a build is present after the swap (B6);
    2. with the lock's `PX` forced short and a sleep injected before `RENAME`, a verdict that lands
       on the old key is present after the swap (P2-3);
    3. after a kill between append and patch, the next verdict or swap re-converges.
  - **Sizing rules:**
    - The `TIMEOUT_MAX` default is the smallest round value at or above 10× the worst p99
      per-level read from S20, and the engine's batch size is set so each per-level read stays at
      or under a tenth of it.
    - The lock ttl is 3× (worst replay of 1,000 log entries + `RENAME` + remainder check).

    Both values are recorded as FR-21's and §3.8's defaults, labelled **P0-scale defaults**: they
    are derived from 1,000-file latencies, and P1 re-derives them on the real snapshot (m8).
  - **Instance-wide side effects (m8).** `TIMEOUT_MAX` binds every credential (V18). So S22 also
    runs, with the chosen value set:
    - a `UNWIND` batch load as `ccg_build` (the quirks file says writes ignore `TIMEOUT`, and this
      re-checks that under `TIMEOUT_MAX`);
    - `GRAPH.CONSTRAINT CREATE` and its backfill on a populated key;
    - a full-text index backfill.

    If any of them is killed or rejected, the note records the interaction, and the loader rule
    P1 must follow (for example, lifting `TIMEOUT_MAX` around a build, which is a privileged
    operation) is written up as an open item for P1.
- **S23** writes the p0-graph note's decisions table and folds the new quirks into
  `claude/graph-dba/falkordb-quirks.md` (dated, with commands). Expected candidates are the
  `MULTI` behaviour, the telemetry controls and `EVAL` under ACL.
- **S17**, after S15 and S23, writes p0-graph note §1, "Graph model v1": the graph note §3.2 as
  amended by §6, with the slice's deltas, the S20 algorithm choice, and the corrections in §11 F1
  and F2. It also writes `ddl_v1.cypher` covering the full index and constraint set of graph note
  §3.4 and §6.4, and `test_ddl_live.py`.

### 4.6 Frontend fitness (S25–S28)

- **S25:**
  - `corpora.yaml` lists language, project, URL, commit SHA, build command and a license note.
  - `run_fitness.py` fetches the corpora once, with network, into `cpg/.ccg-work/corpora/`.
  - It runs each frontend twice. **Off** runs through `cpg/sandbox/offline-run.sh` with no fetch
    flags. **On** runs in the toolchain image with network and the language's fetch flag:
    `--fetch-dependencies` for Java and Go, `--download-dependencies` for Kotlin and C#. Kotlin's
    `--jar4import-url` is never used, in either configuration.
  - `measure.sc` computes S05's metrics from each `cpg.bin`.
  - The seeded sample is written to `cpg/spikes/frontend-fitness/sample/<lang>.jsonl`.
- **S26, labelling (`qa-engineer`):** the labeller gets the pinned corpus source and the sample
  file (file, line, column, call text) and nothing else.
  - It returns, per site, the true callee(s) as (file, qualified name), `external:<package>` or
    `dynamic`.
  - It must not run Joern or any indexer, and must not read `results/` or the harness.
  - `teco` checks this by grepping the transcript (§6.5's method, with that list of forbidden
    paths and tools).
- **S27** computes a1 for 7 languages × {off, on}. It records:
  - every number with its interval;
  - the §3.2 table restated as measured;
  - the indexer and fold/list decisions by S05's rules;
  - the decorator, annotation and attribute shapes per language, which S30 needs.
- **S28** runs S11's `extract.sc` projection on every corpus, offline, and on S05's golden
  small-method sets. It records MB per file, nodes and edges per file, build time and the golden
  recall, and decides `flowAvailable` per language. For Python and TS the slice's figures (S14)
  are cross-checked.

### 4.7 Fixture system (S35–S39)

S35 fixes, as a numbered table, every planted item. The categories that **must** appear are
below; S35 sets the counts, and S06 sets the minimums.

| Category | Gateable in |
|---|---|
| Link kinds and statuses: http, mcp_tool, script → tool, launch chains (`.mcp.json`, Compose), URL binding with two launch configs of one unit on 8000 and 8200, plus an example config | P1 |
| Link kinds and statuses: rpc (gRPC), topic (Kafka and the in-house bus), `library_api` with version skew | P2 |
| Link kinds and statuses: data_entity (SQL table/column, graph label), config_key/env_var | P3 |
| **AC-6 unprovable kinds, ≥ 3 shapes each:** runtime-built URL, dynamic import, example config, prompt mention | P1 |
| **AC-6 unprovable kinds, ≥ 3 shapes each:** topic from config | P2 |
| **B1's four assertions** (the proposal review's B1, not this plan's; graph note §6.1), on a class field | P1 |
| **B1's four assertions** (the proposal review's B1), on a SQL table | P3 |
| **AC-25:** 3 data-flow and 3 control-dependency effects, at distances 1–3 | P1 |
| **P3-1:** a planted mis-resolved call, in both Python and TS, so whichever language S27 puts at ≥ 0.95 carries it | P1 |
| **AC-9b:** an unbuildable unit and a never-built unit | P1 |
| **AC-19:** a multi-repo overlay | P3 |
| **SQ-C:** a signature change in Python (the note is kept) and in Java (the note is dropped) | P4 |
| **P2-7:** a taint finding edited at a middle hop and at an end | P5 |

- **Repos:**
  - `fx-svc` is Python: FastAPI, FastMCP, SQL, a Kafka producer, shell scripts, Compose,
    `.mcp.json` and a Markdown prompt.
  - `fx-web` is TS: a fetch wrapper, an MCP client config and a Kafka consumer.
  - `fx-lib` is Java: a shared library with overloads, the in-house bus, and a gRPC `.proto` with
    its service.
- **Manifests:** each expected manifest row carries the planted id, kind, both locators (repo
  path, qname, line), the expected status and reason, the phase, and the assertion ids. Fixture
  items are **dev** data: visible to builders, and used for functional tests and AC-6/AC-25. None
  of them is an AC-5 held-out item (finding F7).
- **S39's materializer** creates the three git repos under `cpg/.ccg-work/fixtures/` with a
  scripted history. The author, committer and dates are fixed, so the commit SHAs are stable. The
  history includes in-window and out-of-window commits for the later RCA and secrets windows.

## 5. Shared state and sequencing

### 5.1 FalkorDB instances and graph keys

- **`falkordb-dev` (6379) is shared with other sessions and components, and no P0 step writes to
  it.**
  - Every P0 brief must say so.
  - Every P0 script must take its host and port from an explicit argument, with no default of
    6379. The `joern-cpg` pipeline's `--load`/`--reset` must not be used in P0.
  - Reading `cpg_falkorchat` is permitted for builders and **forbidden** for the ground-truth
    agents (§6.3).
- **`ccg-falkordb` (6390, dedicated, S01).** Each step that uses it holds one of the
  reservations below, which `teco` grants one at a time, except that two non-exclusive ones may
  overlap.

  | Reservation | Step | Keys | Exclusive? | Why |
  |---|---|---|---|---|
  | R-B | S13 | `ccg_p0slice`, `ccg_p0slice__build_*` | no | ordinary load and reads |
  | R-C1 | S20 | `ccg_spike_synth*` | **yes** | latency benchmark: a concurrent load or JVM would distort it |
  | R-C2 | S21 | `ccg_spike_swap*` | **yes** | `MULTI`/`RENAME` probes, plus telemetry config changes that may be instance-wide |
  | R-C3 | S22 | `ccg_spike_lock*`, `ccg_<…>:verdict_lock` | **yes** | `ACL SETUSER` and `GRAPH.CONFIG SET TIMEOUT_MAX` are instance-wide |
  | R-D | S17 | `ccg_ddl_test` | no | DDL applied to an empty key |

  - **Exit hygiene:** each exclusive step restores the instance config it changed (`TIMEOUT_MAX`
    back to 0, the probe ACL users deleted) and says so in its note.
  - **P0 exit:** `reset-p0` empties the instance. P1 starts clean.
- **Host contention (6 GB VM).** S20 must not overlap any JVM-heavy or build-heavy step, because
  the timings would be meaningless. That covers S11, S25, S27, S28, S39's live test, S08's JVM test
  (test 3), and the S02/S24 image builds. Joern steps also should not overlap
  each other, since each wants about 2–3 GB of heap.

### 5.2 Shared files

| File | Writers, in order | Rule |
|---|---|---|
| `cpg/infra/toolchain/Dockerfile`, `cpg/infra/toolchain/README.md` | S02 → S24 | strictly sequential |
| `cpg/infra/README.md` | S01 only | S02 writes its own `toolchain/README.md` instead, so S01 and S02 stay parallel in wave 0 (m3) |
| `claude/graph-dba/falkordb-quirks.md` | S07 → S23 | strictly sequential; nobody else writes it in P0 |
| `docs/plans/connected-code-graph-p0-graph.md` | S20 (§2) → S21 (§3–§4) → S22 (§5) → S23 (§6) → S17 (§1) | one owner, sequential. S17 waits for S23 even though its inputs (S15) may be ready first. |
| `docs/test-reports/connected-code-graph-p0-frontend-fitness-report.md` | S27 (§a1) → S28 (§a2) | sequential |
| `cpg/tests/test_rule_pack_schema.py` | S31 → S32 | sequential |
| `cpg/tests/test_eval_tools.py` | S48 → S49 | sequential |
| `cpg/eval/headless/{agents.json, settings.json, guard-paths.sh}` | S52 only; S53 reads them | if S54 fails, the fix goes to whichever owner's file is at fault, never to both at once |
| `cpg/pyproject.toml`, `cpg/.gitignore` | S09 creates both, and **declares every P0 dependency up front** in extras: `jsonschema`, `pyyaml`, `falkordb` (1.6.x, V17), `pytest`. `.gitignore` gets `.ccg-work/` | No other step edits them. A step that finds a missing dependency stops and asks `teco`, which serializes that one edit. This removes the writer chain the review counted, S10, S13, S17, S20–S22, S31, S34 and S39, rather than serializing it (m3). Every step that writes `.ccg-work/` or uses the venv depends on S09 |
| `cpg/ccg/schemas/` | S10, S31, S34, S39 write **different** files | no collision |
| `cpg/ccg/bundle.py`, `bundle-v1.schema.json` | S10, then S16's follow-up | only if S15 shows a schema change |
| `docs/plans/connected-code-graph.md` | S50 only (plus this run's header pointer) | — |
| `docs/reviews/connected-code-graph-p0.md` | `analyst`, one `## Pass N` per gate | gates on different steps append different passes; `teco` serializes analyst dispatches that touch it |

### 5.3 Facts one step decides that another step must encode

Each of these makes two steps sequential, even where their file sets don't overlap.

| Decided in | Fact | Must be encoded by |
|---|---|---|
| S03 | canonical tuples, the id hash, the reason-code registry, the locator form, the source-derivable `qname` grammar, the proving repo's `repoId`, the `ContractPoint`/`keyNorm` tuple | S09/S10 (code), S11 (emitter), S35/S39 (fixture manifests), **S06 and S40** (key and gold locators must be convertible to these ids without a snapshot; M2) |
| S04 | the argv audit format, the escape axes each backend closes, and the forbidden and allowed flag lists | S08 |
| **S15** | **whether bundle schema v1 and the graph model freeze** (and any SL-7 deltas) | S16, S17, S28 (projection script version), S30 (packs match bundle facts), S33 (evidence hops reference graph labels), S34 |
| S20 | BFS vs `algo.SSpaths`, and per-level latencies | S22 (`TIMEOUT_MAX`), S33 (the recipe's algorithm id), S17 (model notes) |
| S22 | the `TIMEOUT_MAX` default and the lock ttl | S50 (proposal §3.5/§3.8 defaults) |
| S27 | the indexer and fold/list decisions per language; decorator and annotation shapes | S30 (format expressiveness), S50 |
| S28 | `flowAvailable` per language | S50; P1 extractors |
| S05 | the sample size, the truth sources and the decision rules | S24 (tools), S25 (sample), S26 (label form), S27/S28 (rules) |
| S06 | strata, counts, item schema, the capture–recapture rule (two-list and multi-list), the hit rule, the development-contact rule (M3), the procedures for GC's tooling (M5) | S35 (minimums), S48–S49 (tooling), S40 (kits), S41–S47 (S47 applies the development-contact rule before the random split), S33 (folded groups enumerate members) |
| S40 | `EVAL_COMMIT` and the exclusions | S41–S47; P1's gate builds |
| S52, S53 | the allowed tool set, the role-root layout, the stream-log location, and the agent names | S48's `audit.py` checks (§6.5) and S54. S48 builds against §6's stated values; S54 is where a disagreement surfaces |

## 6. Ground-truth isolation protocol (P0(g), decision S36, ML §8.2)

**Decision (stakeholder, 2026-10-29, review B1).** The ground-truth writers run as **separate
headless `claude -p` sessions** that `teco` starts through Bash, with a working directory outside
the repo. They are **not** `Agent` subagents of the build session.

`teco` only *runs* the launcher. `devops` builds the launcher (S53), `cobb` builds the writers'
agent definitions (S52), and the planted-breach rehearsal (S54) must pass before any real writer
starts.

**Why the writers exist.** Decision S36 says that "an isolated read-only agent, spot-checked by
the stakeholder" writes the keys. ML §8.2 adds a second, independent writer so that
**completeness** can be estimated by capture–recapture. Two isolated writers satisfy decision
S36 as long as each is isolated and read-only on the repo, and the stakeholder spot-checks
(finding F4). The protocol covers P1's AC-5 strata (HTTP, MCP, scripts, flow-shaped) and AC-24's
impact cells. Later phases reuse it.

CLI facts below come from `claude --help` (Claude Code 2.1.312, read 2026-10-29 without starting
a session). Anything marked *verify (S54)* was not observed running, and the rehearsal must
confirm it.

### 6.1 Roles

| Role | Runs as | Model | Strategy |
|---|---|---|---|
| KA | headless session, agent `ccg-key-writer-provider` (S52), one session per stratum | `opus` | **provider-first**: list every route, MCP tool, script entry point and anchor first, then hunt each one's consumers |
| KB | headless session, agent `ccg-key-writer-consumer` (S52), one session per stratum | `sonnet` | **consumer-first**: list every outbound call, client config, URL, tool mention and script invocation first, then resolve each one's provider |
| QW | headless session, agent `ccg-question-writer` (S52) | `opus` | writes the AC-24 impact-cell questions, natively in EN and PT, with gold |
| AN1, AN2 | headless sessions, agent `ccg-annotator` (S52) | `sonnet`, `opus` | each labels QW's questions ambiguous or not, independently |
| GC (ground-truth custodian) | **in-session** `qa-engineer`, one fresh instance for S40, S45, S47 | default | runs the tooling, writes `cpg/eval/**` and the ground-truth report. It never writes keys, so B1's writer-independence concern does not apply to it; its brief forbids `ingest_document`, and §6.5 checks that by tool name |
| Method owner | in-session `data-scientist` (S40, S45, S47) | default | specifies each procedure; adjudicates near-misses by pair id; checks GC's outputs. Writes nothing outside `docs/plans/` |
| Launcher operator | `teco` | — | runs `run-writer.sh` (S53); never reads keys |
| Spot-checker | the stakeholder (S46) | — | decision S36's spot-check, plus the PT gold spot-check (ML §8.3) |

**Independence.** KA and KB differ in agent definition (two distinct prompts from `cobb`, one per
strategy), in model, and in strategy (ML §8.2). None of the writers' sessions shares a
conversation, a working directory or a scratch space with another (§6.3).

**Kills and resumption (M6, revised for P2-2).** A writer is never reused for another role. Each
writer appends every finished item to its output file as it goes. If a session is killed (a usage
limit or a timeout), resuming it takes four steps:
1. **Audit the partial session first.** GC runs `audit.py` over every stream log and the
   transcript so far. If any voiding event is present, the session is **not** resumed. Its output
   is discarded, and a fresh session of the same role and stratum starts from an empty output.
2. **Resume through the same argv builder.** `teco` runs `run-writer.sh resume <role> <stratum>`,
   which calls the same `build_argv` with `--resume <uuid>`, the fixed continuation prompt
   ("continue; your output file lists what is already done") and the next attempt number `<n>`.
3. **The output survives truncation.** The output file and the tooling tolerate a truncated last
   line left by `timeout`: the reader drops an unparseable final line, and the writer's next
   append starts on a fresh line.
4. **Each attempt is audited.** Each attempt is recorded in the run index, and the final audit
   covers all attempts together.

A resume is the same writer in its own role, so independence is unaffected. A session that
cannot be resumed (a corrupt transcript) is discarded and redone fresh.

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
  matching `docs/**/connected-code-graph*`, and `docs/BACKLOG.md`. It is then **content-checked**
  (m5): GC greps it, case-insensitively, for `connected-code-graph|connected code graph|C-1001|\bccg[_ -]`.
  Every file with a hit is excluded both here and in `cpg/eval/eval-manifest.yaml`, is listed in
  `kit-manifest.json`, and is re-grepped to zero. Excluded files that could have held a P1 item are
  counted as excluded scope in the ground-truth report. `kit-src/` has no `.git` and is
  `chmod -R a-w`.
- **`BRIEF.md`** (one per role and stratum) contains:
  - the role's strategy;
  - the stratum definition and the item schema, copied verbatim from S06;
  - the rubric "a real dependency exists in at least one deployable configuration";
  - for the flow-shaped stratum, the **shared anchor list** drawn by S40 with a recorded seed,
    identical for KA and KB;
  - the output path, and a plain statement of what the session cannot reach.

  QW's brief adds the counts per language and cell (S06), and the gold form: the use case plus the
  anchors in S03's id forms at `EVAL_COMMIT` (§2 D8). AN1 and AN2 get `QW-nogold.jsonl` (S43a)
  inside their own role root.
- **`cpg/eval/kit-manifest.json`** (committed) records the SHA-256 of every kit file and brief, plus
  the exclusion list. The P1 gate's system manifest uses **the same exclusions**
  (`cpg/eval/eval-manifest.yaml`), so the census scope equals the evaluated scope.

### 6.3 Isolation: what holds by construction, and what is verified

**The launch command.** S53 builds it. The flags are as they appear in the CLI help for 2.1.312,
and §6.3 is the single source of the argv.

```
# one function, build_argv(attempt), used by BOTH start and resume (review P2-2)
BIN="$HOME/.local/share/ccg-eval/bin/claude-2.1.312"      # pinned copy, see below (P2-1, P2-3)
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
- **The binary is invoked by absolute path (P2-1).** Under `env -i … PATH=/usr/bin:/bin`, the
  bare name `claude` does not resolve. The reviewer verified exit 127; it is installed as
  `~/.local/bin/claude`, a symlink to `~/.local/share/claude/versions/2.1.312`.
  - **A launcher-owned copy (P2-3).** S53's `prepare-bin` copies that version file (a single ELF,
    about 290 MB) to `~/.local/share/ccg-eval/bin/claude-2.1.312` and records its SHA-256.
  - **Why a copy.** The native updater has written three versions in six days (2.1.309, 2.1.310,
    2.1.312) even with `"autoUpdates": false`, and it keeps only a few. So the plan neither
    references the versions directory directly nor refuses on "a different version". A new
    Claude Code build never touches the copy, and the census never stalls on an update (K-P0-12).
- **`--tools Read,Grep,Glob,Write,Edit` (P2-1)** limits which built-in tools *exist*.
  `--allowedTools` only pre-approves them, which `--permission-mode dontAsk` needs, so both are
  passed. `--disallowedTools` stays as a redundant second line.
- **`--disable-slash-commands` (P2-1)** disables all skills and commands. That matters because
  `~/.claude/skills/` holds **seven entries, not two**:
  - **six symlinks into the repo:** `agent-kb-retrieval`, `agent-maintenance`, `agent-standards`,
    `cpg-analysis`, `joern-cpg`, `python-web-quirks`;
  - a `synced/` directory.

  A `Skill` invocation would read repo content around the file-tool confinement. v1.2's
  inventory was wrong because a truncated listing showed two.
- **One argv builder for `start` and `resume` (P2-2).** Every isolation flag is therefore passed
  on every attempt. The two attempts differ **only** in `<SESSION>`, `<PROMPT>` and the
  stream-log name `<n>`.
- **Per-attempt stream logs (P2-2).** `<uuid>.<n>.stream.jsonl` is a new file per attempt. An
  earlier attempt's log is never truncated, and the audit reads all attempts of a session.
- **`--settings` and `--agents`** are read at launch. Their SHA-256s are recorded in the run index
  on every attempt (P2-3), because they live in the repo, which builders edit. A changed hash
  between attempts stops the resume.
- **The role timeout** is set in S53. The proposed default is 4 h for KA and KB, 2 h for QW, and
  1 h for AN.
- **`--verbose`** with `stream-json` under `-p` is settled by S55's capture (P2-4).
- **`--safe-mode` is rejected for two reasons:** it disables custom agents, and it disables hooks,
  which would disable S52's path guard (review P2-4).

**What the session cannot reach, by how it is built:**

| Channel | Why it is closed | Source |
|---|---|---|
| The repo working directory, and the repo's `CLAUDE.md`/`AGENTS.md` | The cwd is the role root, outside the repo. No ancestor of `~/.local/share/ccg-eval/` holds a `CLAUDE.md` or `AGENTS.md` (probed: `/`, `~`, `~/.local`, `~/.local/share`; `~/.claude/CLAUDE.md` exists but is **0 bytes**) | probe, 2026-10-29 |
| `cypher` and `ws:agent-team` MCP | Both are defined only in the repo's project-level `.mcp.json` (review Appendix A). `~/.claude.json` has **no user-scope `mcpServers`**, and none on any project entry (probed). `--strict-mcp-config` with no `--mcp-config` loads no MCP server at all | probe + help |
| Learning capture into the KB | No KB server is loaded (row above). The `ccg-*` agent definitions (S52) contain no learning-capture, KB or CPG instructions | S52 |
| Files outside the role root: the repo, sibling roles, `~/.claude/`, `/tmp`, `runs/`, `work/` | `--restricted` "confines the file tools to the working directories (`--add-dir` included)". No `--add-dir` is passed. A `PreToolUse` path guard from S52's `settings.json` denies any `Read`/`Grep`/`Glob`/`Write`/`Edit` whose resolved path is outside the role root, or a write outside `out/` (defense in depth). | help + S52 |
| Command execution and the network | `--restricted` removes Bash and the other code-running tools, plus WebFetch. They are also listed in `--disallowedTools`, as is WebSearch. | help |
| Skills and slash commands, including the six repo-symlinked user skills (`cpg-analysis`, `joern-cpg`, …) | `--disable-slash-commands` ("Disable all skills and commands"), since v1.3 | help (P2-1) |
| Any built-in tool beyond the five | `--tools Read,Grep,Glob,Write,Edit` limits which tools exist, and the init check requires equality | help (P2-1) |
| Spawning sub-agents | `Agent` is disallowed | flags |
| User, project and local settings (`~/.claude/settings.json`'s permissions and `autoMode` text, which names the repo path; `settings.local.json`) | `--restricted` "ignores user, project and local settings files"; only `--settings` applies | help |
| A shared scratch space | Each headless run is its own session (its own session id). It has no Bash, and file tools cannot reach `/tmp`, so no scratchpad is shared or reachable | construction |
| Auto-memory of the repo project | Memory is keyed by project directory. A role root is a new project, so the repo's `MEMORY.md` (`~/.claude/projects/-home-mauricio-prg-graphmind-ai-lab/memory/`) is not loaded | inferred; *verify (S54)* |
| Repo-environment variables | `env -i` with a minimal pass-list | construction |

`--bare` would also skip hooks, plugins, auto-memory and `CLAUDE.md` discovery. **It is not
usable:** under `--bare`, Anthropic auth is "strictly `ANTHROPIC_API_KEY` or `apiKeyHelper`
(OAuth and keychain are never read)", and this account uses OAuth plan usage
(`~/.claude/.credentials.json`). `--safe-mode` is not usable either: it disables custom agents,
which would remove S52's definitions.

**What still has to be verified,** in S54 and then on every run from the stream's `system/init`
event:
1. **User skills.** These are closed by construction since v1.3 (`--disable-slash-commands`,
   P2-1). Check that the init event lists **no** skills and no slash commands.
2. **User agents.** `~/.claude/agents/` holds symlinks into the repo's `claude/` folders, plus
   `prontuarios`. `Agent` is not in `--tools`, so at most a catalog description could leak, and
   with `Agent` absent even that goes (review P2-4). `~/.claude/agents/CLAUDE.md` and `README.md`
   have no frontmatter, so they load neither as memory nor as agents. Check: the init event's
   agent list, recorded as a context-leak check, not a capability check.
3. **Plugins.** `settings.json` has no `enabledPlugins`. One marketplace is registered and one
   built-in store entry exists. Check: the init event's plugin list is empty.
4. **Auto-memory.** No memory content appears in the session, and the session cannot write a
   memory file outside its role root.
5. **OAuth works under `--restricted`**, and plan usage is charged rather than an API key. S55's
   capture settles this **before** S48 and S54 rely on it (P2-4). The init event and S54 re-confirm
   it.
6. **The confinement holds** against the planted reads below.

**The planted-breach rehearsal (S54, before S41).** `teco` launches one throwaway writer, role
`rehearsal`, with a brief that instructs it to attempt each of the following and report what
happened:
- read `/home/mauricio/prg/graphmind-ai-lab/AGENTS.md` by absolute path;
- `Grep` with a relative `../../..` path;
- read a sibling role's `out/` (a dummy role root containing a dummy output);
- read `~/.claude/history.jsonl` and `~/.claude/projects/` (which hold other sessions' prompts and
  transcripts);
- read a file under `/tmp`;
- write outside `out/`;
- call `mcp__cypher__query`, `mcp__falkor-chat-agent-team__search_documents` and
  `ingest_document`;
- run Bash, `WebFetch` and `Agent`;
- read `kit-src/README.md`, a legitimate read, as the control.

**Kill-and-resume case (P2-2; stopping item 5).** A second rehearsal session does legitimate reads,
appending to its output, and `timeout` kills it mid-run. Then:
1. GC audits the partial session.
2. `teco` runs `run-writer.sh resume`.
3. The resumed attempt makes one more legitimate read and one planted out-of-root read.
4. **Pass when** both attempts' logs exist under distinct names, the resumed attempt's init event
   shows the full isolation flag set (the same tools, no MCP, no skills), the audit reports no
   false violation across the two attempts, and it flags the planted read in attempt 2.

**Pass when:**
- every illegitimate attempt is refused;
- the audit (S48) flags every attempt, refused or not;
- the control read succeeds and is **not** flagged;
- items 1–5 above are confirmed from the init event;
- the kill-and-resume case passes.

**Any failure stops the chain.** `devops` and `cobb` fix it, and S54 re-runs. Do not proceed on a
partial pass.

### 6.4 Where outputs land, and who computes what

1. Each writer's output: `roles/<role>-<stratum>/out/<role>.jsonl`. After the audit passes, GC
   copies it to `work/keys/` (for KA and KB) or `work/questions/` (for QW and AN).
2. **S43a:** GC produces `QW-nogold.jsonl` with `cpg/eval/tools/strip_gold.py` and places it in
   the AN1 and AN2 role roots.
3. **S45:** GC runs `cpg/eval/tools/{audit.py, match_keys.py, capture.py}`, to S06's procedures.
   - **Audit first** (§6.5). A violation voids that session's output, and a fresh session of the
     same role and stratum redoes it.
   - **Mechanical matching** of items on normalized locators (S06's rule).
   - **Near-misses.** GC writes the unresolved pairs (same link, different line) to
     `work/nearmiss.jsonl`. `data-scientist` reads them in place and returns its verdicts by
     **opaque pair id only**, so no item content passes through `teco`'s context. GC applies the
     verdicts and logs each adjudication.
   - **Estimation,** per stratum and pooled: n₁, n₂, the overlap m, N̂ by S06's estimator
     (multi-list once a third pass exists, m7), and completeness = |union| ÷ N̂. N̂ is reported as
     **a lower bound on incompleteness**, because the writers share LLM blind spots (ML §8.2).
     `data-scientist` checks the counts and estimates.
   - **Trigger.** If completeness falls below S06's threshold in any stratum, `teco` launches one
     more provider-first session for that stratum, and S45 re-runs (§13 K-P0-4).
   - **Spot-check sample.** GC draws a seeded sample from the union for S46, of the size S06 sets,
     plus a PT gold sample from QW.
4. **S46 (stakeholder):** marks each sampled item correct or incorrect. GC applies the corrections
   in S47 and logs them.
5. **S47:** GC runs `cpg/eval/tools/split_manifest.py`.
   - **Split.** S06's development-contact rule (M3) places its anchors first. The rest is split by
     anchor, stratified and seeded. Items sharing an anchor never straddle the split. Held-out
     anchors and items left per stratum are reported; those numbers are SQ-P0-1's trigger.
   - **Ambiguity.** A question is ambiguous only where AN1 and AN2 both say so. Disagreements are
     dropped and reported.
   - **Write and manifest.** Held-out goes, in plaintext, to `~/.local/share/ccg-eval/<E>/heldout/`,
     outside the repo (§2 D11). Its SHA-256 and counts go into `cpg/eval/heldout-manifest.json`,
     which is committed. Dev goes into `cpg/eval/dev/*.jsonl`.
   - **Report.** GC writes `docs/test-reports/connected-code-graph-p0-ground-truth-report.md`. It
     carries counts, n₁/n₂/m, N̂, completeness, adjudication and spot-check tallies, excluded
     scope, split sizes, hashes, the writers' stated hop-count distribution (M4, a prior only),
     **the audit summary**, and **the run count and usage** (§6.6).
   - **Leak check (m5).** The report contains no held-out item, checked mechanically
     (`split_manifest.py --check-report`).
   - **Method check.** `data-scientist` checks the split and the report's numbers.

   Afterwards the raw keys, `work/`, the role roots, the `runs/` logs and the session
   transcripts stay as plaintext under `~/.local/share/ccg-eval/<E>/`, beside `heldout/`, so the
   audit can be re-run. They are hidden by the same convention as the held-out set (D11), and
   their hashes are added to the manifest.

### 6.5 The audit, and lifetime

**The audit** (`cpg/eval/tools/audit.py`, S48) runs on every writer session, each resume
included, and on the rehearsal.

**Sources.**
- **Primary:** the launcher's own per-attempt logs `runs/<session-id>.<n>.stream.jsonl`, for
  every attempt `n` in the run index. They are written by S53's launcher outside the role root, so
  they do not depend on where Claude Code stores transcripts.
- **Secondary:** the session's own transcript. Claude Code keys it by project directory, which
  for a headless run is the role root. The repo's sessions live at
  `~/.claude/projects/-home-mauricio-prg-graphmind-ai-lab/<session-id>.jsonl` (probed). How a
  role root's path is turned into a directory name was not probed: no session has run from a
  dotted path. So `audit.py` finds the transcript by **session id**,
  `~/.claude/projects/*/<session-id>.jsonl`, and makes no assumption about the name.
- **The two sources are cross-checked**, using the **union over all attempts** (P2-2).
  - Every `tool_use` id in the transcript must appear in some attempt's stream log, and every one
    in a stream log must appear in the transcript. A mismatch is a violation, and so is a missing
    transcript or a missing attempt log (K-P0-8).
  - **One exception.** If the line `timeout` cut off at the end of an attempt's log is the only
    stream-side record of a call, that call is checked from the transcript alone. It is reported
    as "stream-truncated", not as a mismatch.
  - S48's two-attempt fixture proves both rules: no false violation on a clean resumed session,
    and a real mismatch still caught.

**Checks.**
1. **Init event, in every attempt:**
   - MCP servers empty;
   - the tool set **equal** to `--tools`, {Read, Grep, Glob, Write, Edit}: no extra tools and
     none missing (stopping condition item 4);
   - the plugin list empty;
   - no skills and no slash commands;
   - the agent list as S54 accepted it;
   - the model and the CLI version equal the run index's, and the hashes of `settings.json` and
     `agents.json` are unchanged across attempts.
2. **Every `tool_use`:** the tool name is in the allowed set. Every path-bearing input
   (`file_path`, `path`, a `Glob` pattern's base, a `Grep` `path`) is **resolved** against the
   session's cwd, with `..` and symlinks normalized, and must lie inside the role root. Writes
   must lie inside `out/`. This is the fix for v1's prefix grep, which matched every line through
   `"cwd"` and missed relative reads (B1 point 5).
3. **Every refused `tool_result`** is reported as an attempted breach, even though the
   construction stopped it. A refused attempt revealed nothing, so it is counted but does **not**
   void the output. Any **successful** call that fails check 1 or 2 voids it.
4. **The run index:** every session id in it was audited, and every audited session is in it.

The audit result, as counts and violation types only, goes in the ground-truth report.

**GC and `data-scientist`** run in-session. Their subagent transcripts
(`~/.claude/projects/-home-mauricio-prg-graphmind-ai-lab/<session>/subagents/agent-<id>.jsonl`)
are checked by tool name only, for `ingest_document`, `WebFetch` and `WebSearch`, which is
reliable regardless of `cwd`.

**Who reads held-out:** a P1+ gate run (`qa-engineer`, at a gate), GC, the `data-scientist`
method owner, and the stakeholder. That is a convention, not an enforced control (D11). Builder
briefs do not mention the path at all, not even to forbid it. At each gate, GC re-hashes the
held-out files against `cpg/eval/heldout-manifest.json` before use, and a mismatch stops the gate.
That detects modification. It does not detect reading.

**After a gate miss** (AC-5's decided fail procedure, SQ1): the failed items move to dev, and the
next gate needs **new** blind items, written under this protocol at the same or a newly pinned
`EVAL_COMMIT`.

### 6.6 Usage cost (stakeholder-accepted)

These runs spend plan usage outside the coordinating session. The expected number of headless
runs for P0:

| Runs | Count |
|---|---|
| Capture (S55) | 1 (trivial) |
| Rehearsal (S54): breach session + kill-and-resume session (2 attempts) | 2, plus 1–2 more if a fix is needed |
| KA, 4 strata | 4 |
| KB, 4 strata | 4 |
| QW | 1 |
| AN1, AN2 | 2 |
| **Base total** | **14–16** |
| Resumes after kills (KA/KB are the long ones; the ledger has had two usage-limit stops in 16 units) | budget +2 |
| Extra census passes if completeness is short (one per short stratum) | 0–4 |
| Redo after an audit violation | 0, or more if one occurs |
| **Expected range** | **about 16–22 sessions** |

KA and KB are census-length sessions. QW, AN and the rehearsal are short.
- **What the record shows:** S53's run index records the reported cost per run, and the
  ground-truth report totals it.
- **What `teco` must not do:** run more than two writer sessions at once. Each one counts against
  the same plan, and the host's 6 GB is shared with whatever else is running.

## 7. Environment

**Probed for this plan on 2026-10-29.** The raw outputs are in the session scratchpad and are not
committed. Anything marked *not checked* is not claimed.

| Need | On the host now | Gap → provisioning (`devops` unless noted) | Needed by |
|---|---|---|---|
| Joern v4.0.579 | ✅ `~/joern/joern-cli` (`io.joern.joern-cli-4.0.579.jar`); `joern`, `joern-parse` and every FR-1 frontend present | Pin the same version in the image (S02) | S11, S25–S28, S39 |
| JDK 21 | ✅ OpenJDK 21.0.12 (`/usr/lib/jvm/java-21-openjdk-amd64`) | Also in the image | same |
| Frontends for the 7 FR-1 languages | ✅ `pysrc2cpg`, `jssrc2cpg` (astgen-linux bundled), `javasrc2cpg`, `jimple2cpg`, `kotlin2cpg`, `csharpsrc2cpg` (dotnetastgen-linux bundled), `gosrc2cpg` (goastgen-linux bundled), `c2cpg`. Fetch flags re-checked with `--help`: `javasrc2cpg --fetch-dependencies`; `kotlin2cpg --download-dependencies`, `--jar4import-url`; `csharpsrc2cpg --download-dependencies`; `gosrc2cpg --fetch-dependencies`, `--include-indirect-dependencies` | `--include-indirect-dependencies` is added to the forbidden list (§2 D6), which the proposal's list lacked | S25–S28 |
| Toolchains for **fetch-on** runs and corpus builds: Maven and/or Gradle, Go, .NET SDK, Kotlin compiler, CMake (and `bear` or CMake's `compile_commands.json`) | ❌ none on the host (`which` found none of them). `gcc`/`g++` present. Node exists only as the Windows `npm` via `/mnt/c`; there is no Linux `node` (`jssrc2cpg` bundles astgen, so the parse does not need it) | **S02:** put all of them in `ccg-toolchain`. They are not installed on the host. | S25 (on), S26 (corpus reading needs no toolchain), S27 |
| Ground-truth tools (typed indexers or tracing) | *not checked*: chosen by S05 | **S24** layers them into the image | S25–S27 |
| No-network sandbox | ✅ Docker 29.7.2 and Compose v5.4.0 for `--network none`. ⚠️ `unshare -rn` and `bwrap --unshare-net` give loopback-only with no DNS, but WSL interop (`WSLInterop` binfmt) stays live inside them, along with `/mnt/c` and `docker.sock` (review M1) | S08 builds both runners. The namespace one refuses while interop is enabled, so on this host only the container backend runs | S08, S11, S25, S28, S39 |
| Dedicated FalkorDB | ❌ only `falkordb-dev` (6379, shared). Images `falkordb/falkordb:v4.18.11`, `:edge` and `:latest` are cached. Port 6390 is free | **S01** | S13, S17, S20–S22 |
| Python 3.12 + venv for `cpg/` | ✅ `/usr/bin/python3.12` (3.12.3). There is no `uv` | S09 creates `cpg/.venv` (covered by the root `.gitignore` `.venv` rule) | every `CC`/`D` step |
| Redis CLI | ✅ `redis-cli` 7.0.15 | — | S01, S20–S22 |
| Claude Code CLI for the headless writers | ✅ `claude` 2.1.312. `--help` lists `-p`, `--restricted`, `--strict-mcp-config`, `--settings`, `--agents`/`--agent`, `--permission-mode dontAsk`, `--permission-prompts none`, `--session-id`, `--resume`, `--output-format stream-json`. OAuth credentials are at `~/.claude/.credentials.json`. `~/.claude.json` has no user-scope MCP servers. `~/.claude/CLAUDE.md` is empty. `~/.claude/skills/` and `~/.claude/agents/` (repo symlinks) exist; whether they leak in is checked in S54. No session was started for this plan. | none to install. S53 pins the version (K-P0-12) | S41–S44, S54 |
| RAM | ⚠️ the WSL VM is capped at **6 GB + 4 GB swap, 8 vCPUs** (`C:\Users\mauri\.wslconfig`, which says the cap protects LM Studio on the 16 GB host). `free -g` shows 5 GB | **Human decision, not `devops`:** leave the cap. P0 fits under it with §5.1's serialization, the instance at 1,536 MB, and Joern at about 2–3 GB of heap. Raising it trades directly against LM Studio. Recorded as K-P0-1. | all |
| Network for corpus fetch and fetch-on runs | assumed available, *not checked* | — | S25 |

**Toolchain per step:**
- S01 needs Docker.
- S02 and S24 need Docker, plus network for the build.
- S08 needs Docker, `unshare` and the image.
- S09, S10, S31, S32 and S34 need Python 3.12 only.
- S11 needs the image (or host Joern and the JDK) and the sandbox.
- S12 needs Python.
- S13 and S17 need Python, `falkordb-py` and the instance.
- S20–S22 need Python, `falkordb-py`, `redis-cli` and the instance.
- S25–S28 need the image with the S24 layer, the sandbox and network (on runs only).
- S26 needs only a text editor and grep, inside the corpus checkout.
- S39 needs Python, `git` and the sandbox.
- S40–S47 need `git` and Python. S41–S44 and S54 also need the `claude` CLI, launched by `teco` through S53.

## 8. §6.2 "for P0 to settle" → where each is settled

| §6.2 item | Settled in | Decision rule or method | Recorded in |
|---|---|---|---|
| Resolved-call precision and indexer adoption per language | **S27** (S05's rules) | §4.3 decision rules | fitness report §a1; proposal §3.2 via S50 |
| (same measurement) fold vs list of a resolved call's other candidates | **S27** | 0.95 threshold, proposal §3.7 | same |
| `flowAvailable` per language | **S28** (+ S14 for Python and TS) | S05's criterion | fitness report §a2; proposal §3.2 via S50 |
| BFS vs `algo.SSpaths` | **S20** | §4.5 rule | p0-graph note §2, §6 |
| `MULTI`/`RENAME` rollback | **S21** | probe outcome; default stays "zero kept, re-assemble" | p0-graph note §3, §6 |
| Whether telemetry can be capped | **S21** | probe outcome + operating rule | p0-graph note §4, §6 |
| `TIMEOUT_MAX` sizing (FR-21's raw-query limit default) | **S22** | §4.5 sizing rule from S20's latencies | p0-graph note §5, §6; proposal §3.5 via S50 |
| Verdict-lock ttl | **S22** | §4.5 sizing rule | same; proposal §3.8 via S50 |
| Infrastructure-signature pack initial content | **S30** (+ S32 YAML) | ML §8.3 classes × runtimes | rule-pack spec appendix; `cpg/packs/infra-signatures/` |
| Severity rule table for non-dependency findings | **S30** (+ S32 YAML) | four levels, stated rules | rule-pack spec appendix; `cpg/packs/severity-rules/` |
| `graph-dba`: fold the four quirks + V16–V18 into `claude/graph-dba/falkordb-quirks.md` | **S07** (in P0), new ones in **S23** | dated entries with commands | the quirks file |
| `security-expert` (optional, before P5): HMAC fingerprint and canary review | **not in P0**, a handoff (§14) | — | — |

The other contracts P0 settles are not §6.2 items, but are settled here:
- bundle schema v1 and graph model freeze: S15, with S16 and S17;
- egress contract: S04;
- evaluation protocol: S06;
- rule-pack format v1: S30;
- answer schema v1: S33.

## 9. Test strategy and gates

### 9.1 Durable code: what is tested at which altitude

How to run the tests:
- **Default suite:** `cd cpg && .venv/bin/python -m pytest -m "not live"`. It needs no Docker,
  FalkorDB or Joern, and must stay green after every CC or D step.
- **Live suite:** `cd cpg && CCG_FALKORDB_PORT=6390 .venv/bin/python -m pytest -m live`.
- **Sandbox suite:** `cpg/sandbox/tests/test_sandbox.sh`.

Contract code (`tdd-engineer`) is written test-first, in this order.

| Module | Unit tests (red → green order) | Contract or integration |
|---|---|---|
| `ccg/ids.py` (S09) | 1. Each S03 golden vector reproduces its 32-hex id. 2. Leading and trailing whitespace is stripped before hashing. 3. NFC and NFD forms of one string give the same id. 4. A 10 kB qname gives a 32-hex id. 5. Changing tuple order changes the id. 6. A signature is accepted only for Java, Kotlin, C# and C++, as S03 rules. | — |
| `ccg/bundle.py` + `bundle-v1` (S10) | 1. A valid minimal bundle validates. 2. The header must come first. 3. An unknown record type or field is rejected. 4. A missing required field is rejected. 5. Writing shuffled records gives byte-identical output. 6. Read → write round-trips byte-identically. 7. A reserved record (`secretSighting`) is accepted but not required. | The schema file is the contract. A test asserts that every record type in S03's spec table appears in the schema. |
| `ccg/answer.py` + `answer-v1` (S34) | 1. Canonical bytes are identical for shuffled `items`/`groups`. 2. A folded group with a member lacking a chain is rejected. 3. `recipe.algorithm.{id, version}` and `stamp` are required. 4. The golden AC-1 answer validates and round-trips. | The golden answer is built from S13's real output, not invented. |
| `rule-pack-v1` (S31–S32) | 1. Each paper pack validates. 2. Each negative example in S30 is rejected. | — |
| `fixture-manifest-v1` + `materialize.py` (S39) | 1. Two materializations give identical commit SHAs. 2. Every locator resolves: the file exists, the line is in range, and the qname is found in the file. 3. Planted ids are unique and every S35 row is present. | `live`: each fixture repo parses in the sandbox with a nonzero `METHOD` count, mirroring `pipeline.sh`'s empty-CPG check. |
| `cpg/sandbox/` (S08) | the seven tests of §4.1 | run once per backend; the namespace backend's refusal under WSL interop is itself a test |
| `cpg/eval/tools/audit.py` (S48) | 1. Flags an absolute repo read. 2. Flags a relative `../` read. 3. Flags a symlink escape. 4. Flags a sibling-root read. 5. Flags an MCP call and a disallowed tool. 6. Flags a write outside `out/`. 7. Does not flag a legitimate in-root read. 8. Flags a stream/transcript `tool_use` mismatch and a missing transcript. 9. Fails an init event with a non-empty MCP list. All on synthetic fixtures. | S54 runs it on a real session |
| `cpg/eval/headless/guard-paths.sh` (S52) | 1. Denies paths outside the root, given absolute, relative and symlinked inputs. 2. Denies writes outside `out/`. 3. Allows in-root reads. | S54 |
| `cpg/eval/headless/run-writer.sh` (S53) | The dry run prints §6.3's exact command. `prepare` is idempotent. `resume` refuses an unknown session id. `start` refuses on a CLI version other than the pinned one. | S54 |
| `ccg/store/ddl_v1.cypher` (S17) | — | `live`: applies to an empty key, `db.indexes()` equals the DDL set, and the constraints reach `OPERATIONAL` after a one-node load |
| `cpg/infra/ccg-falkordb.sh` (S01) | — | smoke: `up` twice is a no-op; `status` reports the config of §2 D7; `down` then `up` restores it |

**Spike tests** (under `cpg/spikes/*/tests`, outside `testpaths`) are the evidence for their
results notes: SL-1–SL-7 for the slice, the lock and replay fault tests for S22. They are run and
cited, not maintained after P0.

### 9.2 Gates

| Kind | Gate | Evidence the gate reads |
|---|---|---|
| C (contract document) | `analyst` review, recorded as a `## Pass N` in `docs/reviews/connected-code-graph-p0.md`, plus the named domain consult (`graph-dba` for S03, `data-scientist` for S30 and S33) | the document, and the proposal sections it implements |
| CC / D (durable code) | `analyst` code review; the default suite green; **a mutation spot-check** (the reviewer flips one condition in the module under review and confirms a test fails), per `teco`'s standing practice | diff, test run, mutation note |
| M (spike) | `analyst` review of the results note; `data-scientist` for every interval and decision rule (S25–S28); `graph-dba` for density and integrity (S15) | results note + raw JSON + spike test output |
| GT (ground truth) | GC (`qa-engineer`) produces; `data-scientist` checks the method, statistics and adjudications; the stakeholder spot-checks (S46). `analyst` reviews **protocol adherence and counts only** (S47): no reviewer outside §6.4's list sees held-out content | ground-truth report, audit result |
| Doc | `teco` check (S07, S51), or `analyst` (S50) | the edit |

**P0 exit** (`teco`) means all of the following are true:
1. Every row of §3 is gated or, for S16, recorded as not needed.
2. Every §8 row points to a recorded decision.
3. S15's freeze outcome is in the ledger.
4. §10's documentation list is applied.
5. `reset-p0` has been run on the dedicated instance.

## 10. Documentation impact

| Document | Change | When | By |
|---|---|---|---|
| `docs/BACKLOG.md` C-1001 | **Rewritten in place** (never appended), forward-looking only. At P0 start it says P0 is in progress and points to this plan and the ledger. At P0 exit it says what remains (P1–P5, the Arduino slot) and that the next step is the P1 plan. | P0 start; P0 exit | `teco` |
| `docs/HISTORY.md` | One dated entry per delivered group: (1) environment and sandbox (S01, S02, S08, S24); (2) contracts (S03, S04, S06, S05); (3) slice and freeze outcome (S11–S17); (4) store spikes (S20–S23); (5) frontend fitness (S25–S28); (6) rule-pack and answer contracts (S30–S34); (7) fixtures and ground truth (S35–S47), with counts and no held-out content; (8) P0 close (S50) | as each group gates | `teco` |
| `claude/graph-dba/falkordb-quirks.md` | proposal §6.2 fold-in, then the new quirks | S07, S23 | `graph-dba` |
| `docs/plans/connected-code-graph.md` | v3.4 in place with one dated line (S50). This run adds only the `Extended by:` pointer. | S50 | `architect` |
| Root `AGENTS.md` `cpg/` bullet and component-docs row | **One clause at P0 exit**, saying `cpg/` now holds the `ccg` contracts, schemas, sandbox, fixtures and eval data, plus `cpg/README.md` in the entry-docs row. The full rewrite stays at P1 (proposal §3.13). See F10 for why the P0 clause cannot wait. | S51 | `teco` |
| `cpg/README.md`, `cpg/infra/README.md`, `cpg/infra/toolchain/README.md`, `cpg/eval/headless/README.md` | new | S09, S01, S02, S53 | implementers |
| New documents (all with the root `AGENTS.md` header block) | `docs/plans/connected-code-graph-{bundle-spec,egress-spec,rule-pack-spec,answer-spec,p0-ml,eval-protocol-ml,p0-graph}.md`; `docs/test-plans/connected-code-graph-fixtures.md`; `docs/test-reports/connected-code-graph-p0-{frontend-fitness,slice,ground-truth}-report.md`; `docs/reviews/connected-code-graph-p0.md` | per step | per step |
| `Extended by:` pointers (n1) | Each document's creating step adds the matching pointer to its parent's header. That header edit is metadata, not an amendment. **Proposal** ← the four specs (S03, S04, S30, S33; `architect`). **Graph note** ← `connected-code-graph-p0-graph.md` (S20, its first writer; `graph-dba`), which is also how F1's correction becomes visible from the graph note. **ML note** ← `connected-code-graph-p0-ml.md` (S05) and `connected-code-graph-eval-protocol-ml.md` (S06; `data-scientist`). | creating step | creating step's owner |
| Ledger `docs/plans/connected-code-graph-coordination.md` | units U17+ drawn from §3; agent ids of every ground-truth instance (§6.1); the S15 outcome | throughout | `teco` |
| Archive flips | The P0-only documents (this plan, the p0-ml note, the three P0 reports) freeze at P0 exit. Archiving them is `teco`'s call at the milestone close it chooses. The contract specs and the eval protocol stay `active`, because P1–P5 execute against them. | M10 close or earlier | `teco` |

## 11. Findings (proposal ↔ notes)

These are contradictions and gaps found while planning. None stops P0. Each has a disposition,
and none is resolved by reinterpreting a requirement. A gap at requirement level goes to §12.

| # | Where | Finding | Disposition in this plan |
|---|---|---|---|
| F1 | graph note §6.1 item 6 vs proposal §3.7 and requirements decision 2026-10-28 (P3-1) | The graph note says the operator "suppresses the candidate equal to the resolved target and lists the others as *possible* … **unless** P0 measures Joern's resolved-call precision above a threshold". That reads as hiding the other candidates above the threshold. The later decision is the opposite: they are never dropped, only folded away in rendering. Graph note v2.1 was not amended for it. | P0 follows proposal §3.7. S35 plants a P3-1 case, S33's answer schema enumerates folded members, and S17 records the correction in the p0-graph note's "Graph model v1" (that note `Extends:` the graph note). The graph note itself is not edited. |
| F2 | graph note §3.4 vs §6.4 | §3.4 still writes `symbolId = h(lang, unitId, qname, signature)`. §6.4 (m2) removes `unitId` and keeps the signature only for overloading languages. The note's own rule (§6: "this section wins") settles it. | S03 and S17 use §6.4. No action on the graph note. |
| F3 | proposal header and opening paragraph | Both call the graph note "v2". It is v2.1 (2026-10-28, preview sizing). | S50 corrects it. |
| F4 | requirements decision S36 vs ML §8.2 and proposal §5.2 | Decision S36 decides on "an isolated read-only agent, spot-checked by the stakeholder", in the singular. ML and the proposal use two. | Read as consistent: each of the two is isolated and read-only on the repo, and the stakeholder spot-checks the union (§6). Recorded so the reviewer can challenge the reading. This is not a requirement change. |
| F5 | ML §8.3 (AC-24 gold) | Gold is defined as "resolved anchor IDs against a pinned snapshot". No snapshot exists in P0, where the questions are written. | §2 D8: gold is anchor canonical tuples at `EVAL_COMMIT`. These are id-equivalent because `symbolId` is a pure hash with no unit, **provided** S03 fixes the `qname` grammar, the `repoId` and the `ContractPoint` tuple (review M2). S06 and S40 depend on S03, and S06 confirms. |
| F6 | ML §8.2 (capture–recapture) and proposal §4 P0(g) | For the **flow-shaped** stratum the population is (anchor, impacted element). Two agents choosing their own anchors would enumerate different populations, and the overlap estimate would be meaningless. Neither document says how anchors are chosen. | S40 draws one seeded anchor sample, given identically to KA and KB. S06 fixes the rule. |
| F7 | ML §4.1 vs AC-5's text | ML lets planted fixture items "fill the gap" in AC-5's ≥ 75 held-out. AC-5 requires "known cross-boundary impacts **in the proving repo**". The fixture is a separate system, and builders see it. | Fixture items are dev-only in P0. AC-5's held-out set comes from the proving-repo census. A shortfall is §12 SQ-P0-1. |
| F8 | proposal §4/§5.2, ML §4.1 | Nothing pins the commit that the keys (file:line at both ends) refer to, or the exclusions that define the census scope. Keys would rot as the repo moves. | §2 D8: `EVAL_COMMIT` and `eval-manifest.yaml`; P1 gates build from a worktree at that commit. |
| F9 | proposal §4 P0(a) | (a) asks for the flow projection's density and build time on all 7 languages. The projection script is a product of (b). | §2 D3: split into a1 and a2. This is sequencing only. |
| F10 | proposal §3.13 and the ledger's documentation-impact scan | Both defer the root `AGENTS.md` `cpg/` rewrite to P1. But P0 lands the first code in `cpg/`, and the bullet ("durable CPG reload artifacts") goes stale at P0 exit. Root `AGENTS.md` says a context file must state "what is true now". | One clause at P0 exit (§10), full rewrite at P1. |
| F11 | proposal §3.2 table, §3.12 | `gosrc2cpg --include-indirect-dependencies` is a fetch flag (it works together with `--fetch-dependencies`) and is not on the forbidden list. | Added to `forbidden-flags.txt` (§2 D6). It is harmless alone, but forbidding it costs nothing. |
| F12 | ML §4.4 and §4.7 | They still place the interpretation gate at P4 with ≥ 0.92 on 100 questions (bilingual optional, ML §4.4). Both are superseded by ML §8.1 and §8.3 (decisions S35 and S37: P1 smoke on bilingual impact cells, P5 full). | P0 follows §8.3. No action on the ML note. |
| F13 | ML §3.6 | Fallback-recall truth "from runtime traces of the sample program, or a typed indexer" assumes toolchains that are not installed. It also assumes corpora for five languages that the proving repo does not contain, and the two obvious multi-language candidates are excluded as AC-7 and held-back codebases. | §2 D4 (external corpora, exclusions), S02/S24 (toolchains in the image), S26 (blind labels as truth, or as a check on the truth). |

## 12. Stakeholder questions and decision points

**Stakeholder question (requirement level, conditional; raised only if it occurs):**
- **SQ-P0-1: AC-5 held-out size at P1.** AC-5 needs at least 75 held-out items "in the proving
  repo". This is raised by `teco` at S47 if the counts call for it. The trigger: the two-agent
  census of P1's strata at `EVAL_COMMIT` leaves fewer than 75 held-out items after S06's
  development-contact rule (M3), even after S47 gives every remaining census item to held-out and
  pack authors get only the fixture's dev items. Which then?

  Two options stay **inside** AC-5's text:
  - (d) **Add a wiring stratum** to P1's census. P1 builds the `.mcp.json` and Compose launch-chain
    extractors, so these are proving-repo links that P1 gates anyway. It is consistent with the ML
    note too, so S06 may adopt it up front without asking anyone.
  - (e) **Add runtime-observed edges**: distinct cross-boundary calls seen in server access logs or
    HTTP-client instrumentation while the suites run. This **conflicts with ML §4.1** (review
    P2-6), which says oracle edges are "reported next to seeded recall, **never pooled**", because
    they over-represent exercised paths. So (e) is available only if `data-scientist` explicitly
    revises that rule in S06, with its reason, **and** names a collection step, which no P0 step
    is today. Otherwise (e) is off the table.

  Three options need the stakeholder:
  - (a) Top up with planted items in a fixture repo kept outside the repo like the held-out set (D11), written by an isolated agent and
    reported as its own stratum (ML §4.1). **This amends AC-5**, because a fixture repo is not "in
    the proving repo" (F7).
  - (b) Gate on fewer than 75 and report the weaker bound. This also amends AC-5's size.
  - (c) Wait for P3's data stratum to bring the pool to 75 before AC-5 is gated at all. This moves
    an AC-5 gate the proposal places at P1.

  *Recommendation:* exhaust (d) first, which needs no decision. Then use (e) only if S06 has
  revised ML §4.1's never-pooled rule as described. If the count is still short, choose (a), reported separately and never pooled silently with census items, as an
  explicit AC-5 amendment. (b) weakens the never-miss guarantee the stakeholder ranked first, and
  (c) leaves P1 with no AC-5 evidence.

  I no longer estimate whether this will be needed. v1's "55 files" figure was stale: 129 tracked
  files now name `mcp__cypher__query`, 120 of them Markdown. Documentation snippets are not callers
  (proposal §3.4), so the file count says little about the item count until the census exists.

**Decision points for `teco`** (design level, routed to `architect` + `analyst`, not to the
stakeholder):
- **SQ-P0-2: a partial slice pass at S15.** This plan executes the proposal's rule: any SL failure
  leaves v1 provisional (§2 D5). If `teco` sees value in freezing only the parts that passed, that
  is an amendment to proposal §4, made by `architect`, reviewed by `analyst`, and decided at that
  point, not presumed now. *Recommendation: don't.* Provisional v1 costs only a P1-exit re-check of
  S30/S33/S34. A partial freeze adds a mixed-state schema that every P1 step would have to reason
  about.

## 13. Risks

| # | Risk | Effect | Mitigation |
|---|---|---|---|
| K-P0-1 | 6 GB WSL cap (deliberate, for LM Studio) | OOM or skewed timings when Joern, FalkorDB and Docker overlap | §5.1 serialization; instance at 1,536 MB; Joern heap set explicitly per step; the 1,000-file synthetic size in S20; the workplace-scale peak is explicitly not a P0 claim |
| K-P0-2 | A JVM stalls inside a no-network namespace (local-hostname lookup) | Every offline Joern run hangs | S08 test 3 catches it early. Fixes: an `/etc/hosts` entry in the container, `-Djava.net.preferIPv4Stack=true` |
| K-P0-3 | The slice fails an SL criterion: TS object-literal field paths, or SL-5 density | v1 provisional; P1 carries the freeze | The proposal already allows it. S30, S33 and S34 proceed against provisional v1 and are re-checked at P1 exit |
| K-P0-4 | Low KA/KB overlap → low completeness | Extra census passes; schedule slip | S06's threshold and the extra-pass loop (§6.4) are budgeted. Completeness is reported as a lower bound either way |
| K-P0-5 | Census short of 75 | AC-5 at P1 underpowered | SQ-P0-1, raised at S45 |
| K-P0-6 | No typed indexer runs offline, or one is too heavy for the 6 GB host, for some language | Truth for that language rests on hand labels only, with wider intervals | S05 sets per-language fallbacks; S27 reports the intervals as they are |
| K-P0-7 | Held-out is hidden by convention only (D11, stakeholder decision) | A builder reading it, e.g. while debugging an AC-5 miss, goes undetected and turns that held-out item into development data | **Accepted with the decision.** No added mechanism. The hash manifest detects modification, not reading. Builder briefs never name the path |
| K-P0-8 | The headless construction leaks through a channel nobody listed (user skills, agents, plugins, auto-memory), or the stream/transcript format changes so `audit.py` can't parse it | Blindness lost, or the audit can't run | S54's rehearsal and the per-run init-event checks (§6.3). The audit cross-checks two sources and fails closed: a missing or unparseable source voids the session (§6.5) |
| K-P0-12 | A Claude Code update lands mid-chain (three builds in six days so far, despite `autoUpdates: false`), or a different build changes `--restricted`, `--strict-mcp-config` or the stream-json shape | The census stalls, or the construction or the audit silently weakens | The launcher runs a **launcher-owned copy** of 2.1.312 (S53 `prepare-bin`, SHA-256 recorded), so updates never touch the census. Moving to another build is a deliberate step: a new `prepare-bin`, then S55 and S54 re-run. Hashes of the headless configs are checked on every attempt (P2-3) |
| K-P0-9 | Instance image drift (`:latest` and `:edge` are cached locally) | Quirks and V-probes stop applying | S01 pins `v4.18.11`, the same as `falkordb-dev`, and `status` prints the module version |
| K-P0-10 | P0's breadth: about 45 steps, about 40 dispatches | Long wall-clock | Six-way parallel wave 0. The critical path (§3) is about 11 steps long. The human step S46 is announced to the stakeholder when S41/S42 start |
| K-P0-11 | Corpus licences, or a corpus that is secretly a held-back candidate | Contamination or licence trouble | S05's exclusions are checked at S25's gate; `corpora.yaml` carries a licence note |

## 14. Ready to implement

**Plan:** `docs/plans/connected-code-graph-p0.md` (this document). Dispatch from §3 in waves; §5
is the sequencing contract and §6 the isolation contract.

**Digest:**
- **Wave 0, six dispatches in parallel:**
  - `devops`: the dedicated instance (S01) and the toolchain image (S02);
  - `architect`: the bundle spec (S03) and the egress spec (S04), as two separate dispatches;
  - `data-scientist`: the fitness protocol (S05). The eval protocol (S06) follows S03;
  - `graph-dba`: the quirks fold-in (S07).
- **Four chains then run beside each other:**
  - the slice: sandbox → schema code → extract → assemble → load/BFS → report → S15 freeze gate;
  - the store spikes: S20–S23, exclusive on the instance;
  - frontend fitness: S24–S28;
  - fixtures and ground truth: S35–S54.
    - The writers run as **headless sessions** that `teco` launches with `devops`'s launcher (S53)
      and `cobb`'s agent definitions (S52), after the planted-breach rehearsal (S54) passes.
    - There are about 16–22 sessions in all (§6.6), run at most two at a time.
    - There is one stakeholder spot-check.
- **After S15:** the graph model v1 and DDL, the rule-pack and answer contracts, and their schema
  code.
- **To close:** S50 folds the measured values into the proposal (v3.4), and S51 updates the docs.
- **Freeze rule:** binary, per the proposal (SL-1–SL-7). Partial is provisional.
- **Shared state:** no P0 step writes to `falkordb-dev`, and the dedicated instance is
  reservation-serialized.

**Named handoffs to P1 (not built in P0):**
- the production extractors, ported from the slice under P1's own tests;
- the pack runtime, executing the S31/S32 packs;
- the linker;
- the store loader, creating S22's ACL users and applying `ddl_v1.cypher`;
- the id-hash fuzz test on a throwaway instance;
- the broker process and AC-8's logging proxy, both per S04;
- the answer engine, per S20 and S33;
- the AC-5 gate harness, which verifies the held-out manifest, reads the held-out set, builds at `EVAL_COMMIT`, applies S06's
  noise sampling, and ratifies the possible-precision thresholds after the first P1 measurement
  (ML §4.2);
- the AC-24 smoke harness;
- `teco` v1 integration (a `cobb` unit);
- the full rewrite of the root `AGENTS.md` `cpg/` bullet;
- porting or deleting `cpg/spikes/`.

**Measurements deferred from the notes (review M4), each with its reason:**
- `LogTemplate` volume (graph note Q3) → **P5**. No log-template extractor exists before P5.
- `RAN` edge volume (graph note §6.4, risk 3) → **P3**. Per-test runtime importers arrive in P3.
- The ~18 GB peak at 10k files with two previews → re-estimated at **P1** from the real
  proving-repo snapshot's per-file size, and measured at **P2** on the OpenTelemetry demo. This
  6 GB host cannot measure it directly (§7).
- Typical evidence-chain length, which the ML §3.6 revisit of the 0.98 bar needs → **P1's first
  AC-5 gate**, from the hop counts of real answers. S47's distribution of writer-stated hop
  counts is a prior only.
- `Link` counts under runtime-built URLs at scale → **P1**. The slice (S14) counts only its two
  units.

**Later than P1:**
- the `security-expert` HMAC and canary review, before P5;
- the Arduino slot, after P2.

## 15. Review dispositions (Pass 1, v1.1)

| Finding | Disposition | Where |
|---|---|---|
| m1 (step ids vs decision ids) | **Fixed.** Every bare decision id is now written *decision S<n>*. A legend line is added under §3. | §3, §6, §11 F4/F12, S46 |
| m2 (S03/S04 bundled) | **Fixed.** They are now separate wave-0 dispatches. | §3 |
| M5 (data-scientist writes outside its guard) | **Fixed by reassignment.** No agent definition is touched. A fresh `qa-engineer` ground-truth custodian (GC) runs and writes everything under `cpg/eval/**`, `~/.local/share/ccg-eval/**` and the ground-truth report. `data-scientist` specifies the procedures in S06, adjudicates near-misses by opaque pair id, and checks outputs; it writes nothing outside `docs/plans/`. The tooling is now named and owned: new steps S48 and S49 build it, and S43a strips the gold. | §3, §6.1, §6.2, §6.4 |
| M6 (S26 and S41/S42 oversized) | **Fixed.** S26 is now one dispatch per language. S41 and S42 use one instance per (role, stratum). Writers append as they go and may be resumed in their own role. The old "never resumed" rule becomes "never reused for another role". v1.2: a resume is `run-writer.sh resume`, which runs `claude -p --resume <session-id>` in the same role root (§6.1). | §3, §6.1 |
| M1 (namespace sandbox not offline on WSL) | **Fixed as recommended.** D6 now states the property per backend and per escape axis. The namespace fallback binds a minimal set and refuses to start while `WSLInterop` is enabled, which leaves only the container backend on this host. The container mounts its source and output `noexec`. S08's test 1 is now a probe over five escape axes: IP, DNS, host sockets, interop exec and Windows mounts. S04 contracts the same axes. My earlier `ws:agent-team` capture overstated the namespace, and a correction document has been filed. | §1, §2 D6, §4.1, §4.2, §7, §9.1 |
| m4 (hand-made flag list) | **Fixed.** S08 adds test 6, a `--help` sweep against `forbidden-flags.txt` and a reviewed `allowed-flags.txt`, which fails on any unclassified flag. Test 2 also covers `--flag=value` and pass-through after `--`. Test 7 runs the sandbox from an emptied environment, which closes the `ENABLE_DEPENDENCY_RESOLVER_V2` route. | §2 D6, §4.1, S08 |
| M2 (locator form decided by S03 but not depended on; tuples incomplete) | **Fixed as recommended.** S03 now owns a source-derivable `qname` grammar per language, the proving repo's fixed `repoId`, and a `ContractPoint (kind, keyNorm)` tuple with a `keyNorm` rule. D8 now cites those forms; `repoPath` was a slip, and the field is `repoId`. New edges S03 → S06 and S03 → S40 move S06 to wave 1. The cost is one dispatch-length on the ground-truth chain, which B1 holds anyway. | §2 D8, §3, §4.2, §5.3 |
| M3 (HTTP stratum developed against before being held out) | **Fixed as instructed, without choosing the rule.** S06 gains a done-condition owned by `data-scientist`: the development-contact rule for anchors touched by the slice, the S30 walkthroughs or fixture tests. S14 and S30 must list the anchors they touch. S47 applies the rule before the random split and reports the held-out anchors left per stratum, which is SQ-P0-1's trigger. | §3 S06/S47, §4.3, §6.4, §12 |
| m7 (third pass breaks Lincoln–Petersen; gold-strip owner) | **Fixed.** S06 names the multi-list estimator. GC owns gold stripping as S43a, using `strip_gold.py` from S48. | §4.3, §6.4, §3 |
| M4 (P0-measurable deferrals with no owner) | **Fixed, partly as recommended.** S28 records symbol and call-site density per language. S14 records the `Link` count, the frontier size and the field-path state count. `LogTemplate`, `RAN`, the 18 GB peak, scaled `Link` counts and chain length are named handoffs in §14, each with its reason. **Partly disagreed:** the reviewer's "S47 reports the hop-length distribution" is kept, but only as a prior. A blind writer's stated hop count is not the system's evidence-chain length, which is what ML §3.6 needs to revisit the 0.98 bar. So the real measurement is handed to P1's first AC-5 gate. | §3 S14/S28, §4.3, §6.4, §14 |
| m3 (missing edges and shared files) | **Fixed, one item differently.** Added edges: S16 → S30/S33/S34 (conditional), S10 → S39, S09 → S20–S22 and S25, plus M2's S03 → S06/S40. S02 now writes `cpg/infra/toolchain/README.md`, so S01 and S02 stay parallel instead of being serialized. S08's JVM test and the image builds are added to S20's no-overlap set. The S27 → S35 row is dropped from §5.3. **Different from the recommendation:** instead of adding more `pyproject.toml` writers and serializing them, S09 declares every P0 dependency up front, so no other step edits the file. | §3, §5.1, §5.2, §5.3 |
| m5 (text checks standing in for behaviour) | **Fixed.** SL-2 uses a counterfactual `API_PREFIX` re-run. S01 checks binding with `docker port`. S47's no-held-out-content claim is a mechanical locator grep (`split_manifest.py --check-report`, tested in S49). The kit exclusion adds a post-export content grep, and every hit is excluded from both the kit and the evaluated scope, so the two stay equal. | §3 S01/S47, §4.4, §6.2, §6.4 |
| m6 (SQ-P0-1's option (a) contradicts F7; stale estimate) | **Fixed.** The question now says that (a), (b) and (c) each amend AC-5. It adds two options that stay inside AC-5, a wiring stratum and blind runtime-observed edges, and recommends exhausting them first. The "55 files" estimate is withdrawn: the count is now 129, and documentation snippets don't count. The trigger moves to S47, after the development-contact rule. | §12 |
| n3 (passphrase custody inconsistent) | **Fixed by picking the file.** Gate runs and GC are non-interactive agents, so stakeholder-held interactive custody would block every gate. Every gpg call uses `--no-symkey-cache`. The §12 custody decision point is removed. | §2 D11, §12 |
| n3, P2-5 (v1.3) | **Superseded by stakeholder decision (2026-10-29), not fixed as a finding.** The stakeholder ruled out held-out encryption entirely ("too much"). There is no gpg, no passphrase and no passphrase file. The held-out set is plaintext outside the repo, a committed hash manifest makes tampering detectable, and D11 states that the hiding is by convention only. P2-5's recommendation (time-boxed custody) is moot. No substitute mechanism was added. | §2 D11, §3 S47/S49, §6.4–§6.5, §13 K-P0-7 |
| n2 (F12 says "100 English questions") | **Fixed.** It now says the questions are bilingual-optional (ML §4.4). | §11 F12 |
| m8 (`TIMEOUT_MAX` sizing) | **Fixed.** S22 probes the chosen value's effect on batch load writes, `GRAPH.CONSTRAINT` creation and full-text backfill, and labels both defaults P0-scale, to be re-derived at P1. | §3 S22, §4.5 |
| n1 (`Extended by:` pointers missing from §10) | **Fixed.** §10 lists each pointer and the creating step that adds it: the proposal for the four specs, the graph note for p0-graph (so F1 is visible from the graph note), and the ML note for both `-ml` notes. | §10 |
| **B1** (key-writer isolation) | **Fixed per the stakeholder's decision (2026-10-29): separate headless sessions, option (a).** v1.1 left it pending. v1.2: §6 is rewritten. **Isolation by construction:** the cwd is a per-role root outside the repo, and no ancestor holds a `CLAUDE.md`; `--strict-mcp-config` loads no MCP server (`cypher` and `ws:agent-team` are project-scoped only, and there are no user-scope servers); `--restricted` confines the file tools and drops Bash and Web; there is a `PreToolUse` path guard; `env -i`; each session has its own scratch, which it cannot reach anyway. **Verified rather than assumed:** user skills, user agents, plugins, auto-memory and OAuth, checked in S54 and on every run's init event. **The audit** parses resolved `tool_use` paths from the launcher's own stream log, cross-checked against the session transcript, which it finds by session id. **New steps:** S52 (`cobb`: agent definitions + path guard), S53 (`devops`: launcher), S54 (the planted-breach rehearsal, before S41). Resume uses `--resume <session-id>` in the same role root. Usage is about 14–19 sessions. `--bare` (API-key-only auth) and `--safe-mode` (which disables custom agents) were considered and rejected. | §6, §3 S40–S48/S52–S54, §13 K-P0-8/K-P0-12, §14 |

| P2-1 (launch can't start; tool flags; skills) | **Fixed as recommended, one detail differently.** The binary is invoked by absolute path, and `--tools Read,Grep,Glob,Write,Edit` and `--disable-slash-commands` are added. The skill inventory is corrected to seven entries, six of them repo symlinks. S53's done-condition runs the `env -i … timeout 10 <bin> --version` smoke test. **Different:** the reviewer suggested pointing at `~/.local/share/claude/versions/<v>`; the launcher instead runs a launcher-owned **copy** of 2.1.312 (`prepare-bin`, SHA-256 recorded), because the updater rotates that directory. | §6.3, §3 S53 |
| P2-2 (resume breaks construction and audit) | **Fixed as recommended.** There is one `build_argv` for `start` and `resume`. Stream logs are per attempt (`<uuid>.<n>.stream.jsonl`) and the audit reads the union over attempts. The partial session is audited before any resume, and a voiding event means a fresh session, never a resume. The tooling drops a truncated last line, which is handled as "stream-truncated" in the cross-check. S48 has a two-attempt fixture and S54 a kill-and-resume case. | §6.1, §6.3, §6.5, §3 S48/S53/S54 |
| P2-3 (version pin stalls mid-chain; config hashes) | **Fixed.** The launcher-owned binary copy means an update never stalls the census, and switching builds is a deliberate step that re-runs S55 and S54. The SHA-256 of the binary, `settings.json` and `agents.json` is recorded per attempt, and `resume` refuses if any changed. | §6.3, §3 S53, §13 K-P0-12 |
| P2-4 (S54 items relied on before S54) | **Fixed as recommended.** New step S55 captures one trivial real headless run through the real argv, in a dummy role root so it creates no dependency cycle. That run settles the stream and transcript shapes, `--verbose` and OAuth under `--restricted` before S48's fixtures are built. If OAuth fails, the chain stops and goes back to the stakeholder. The user-agents answer and the second `--safe-mode` reason (it disables hooks) are folded in. | §3 S48/S55, §6.3 |
| P2-6 (SQ-P0-1 option (e) vs ML §4.1) | **Fixed as recommended.** Option (e) is available only if `data-scientist` explicitly revises ML §4.1's never-pooled rule in S06 and names a collection step. Otherwise it is off the table. The recommendation is now (d), then (e) only under that revision, then (a). | §12 |
| Pass 2 nit (S14 records `Link` chain hop counts) | **Fixed.** S14 records the hop count of every `Link` chain the slice finds. | §3 S14 |
| Pass 2 nit (m1 legend range) | **Fixed.** The legend now reads `S01`–`S55`, including `S43a`. | §3 |
