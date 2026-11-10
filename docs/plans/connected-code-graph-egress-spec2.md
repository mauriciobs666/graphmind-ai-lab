# Connected code graph — egress and sandbox spec, S08-gate amendments

> **Status:** superseded · **Owner:** `architect` · **Tracks:** C-1001 (M10) · **Extends:** `docs/plans/connected-code-graph-egress-spec.md` · **Version:** 1.2 · **Superseded by:** `docs/plans/connected-code-graph-egress-spec3.md` · **Reviews:** `docs/reviews/connected-code-graph-p0-impl.md`

*2026-10-29 (v1.0): S08 gate errata (M1, M2, M3, m2, m4, m6, n1, n2, deviations 1–10) and S04 Pass 3 (P3-m1, P3-n1).*
*2026-10-30 (v1.1): analyst re-gate ES-m1–m5 and ES-n1. The `\r` split is confirmed by probe; the H15 row is named as replaced; the runner-environment rule is closed again; the lock moves outside the `upperdir`; the plan fold-in is completed. Nothing is declined. Dispositions are in §4.*
*2026-10-30 (v1.2): analyst Pass 2 ES-P2-m1 and security Pass 3 E3-1/E3-2, all in §1.7: `-p` is required in the `#!` line; `HOME` (from passwd) joins the closed list; the Docker-config directory has a stated invariant with a 69 refusal; the polluted test gains `BASH_ENV`/`ENV` and a pre-populated config directory. Nothing is declined. Dispositions are in §4.1.*

## 0. What this is, and how to read it

**Read with `docs/plans/connected-code-graph-egress-spec.md` v1.3 ("the spec").** The spec controls
everywhere except where an amendment in §1 names the clause it replaces; there, §1 controls. Section
numbers are the spec's.

**This document replaces the successor plan's errata E1 and E2**
(`docs/plans/connected-code-graph-p0-2.md` §4.1). E2 is restated unchanged as §1.2. E1's
whitespace set `[ \t\r\n\f\v]` is superseded by §1.1.

**Why a successor, and why `Extends:`.**
- **Collision rule 5 requires a successor.** The spec has been gated (S04 Passes 1–3) and executed
  against (S08's runner and tests).
- **`Extends:`.** Every change is a clause-level erratum to a spec that otherwise stands. The
  reasons are those of `docs/plans/connected-code-graph-bundle-spec2.md` §0: re-emitting about
  1,150 gated lines would add review cost and no content. The egress spec has no versioning
  mechanism of its own (it has no §8.6 counterpart), so there was no errata route other than a
  successor.
- **Consolidation:** whenever the egress spec is next revised for P1's broker (§2–§6), that revision
  is a `Supersedes:` successor of both documents.

**Tags.** **[S08]** means the runner or its tests must change: S08's current fix round already lands
some of it (§3). **[doc]** means wording only; the runner already does it.

CPG: considered, not relevant — the inputs are the S08 gate's run evidence and the spec text; no
`cpg_*` graph covers `cpg/sandbox/` (Bash and Perl).

## 1. Amendments (normative)

### 1.1 The `//>` directive refusal (replaces the spec §7.3.1 "In-file directives are refused" bullet's rule and its H15 sentence, and the spec §9 H15 row's "indented by two spaces: … inert" clause; supersedes erratum E1) [S08]

- **The rule.** Split the staged script into lines at every `\n`, `\r` and `\r\n`. Refuse the script
  (`forbidden-directive`, exit 64, no container created) if any line, **after every leading character
  ≤ U+0020 and every leading U+007F is removed**, starts with `//>`. **Any directive counts, in any
  case** (`//> USING DEP`), not only `using dep` (ES-n1). The runner may also split at `\f` and `\v`;
  that is a stricter, fail-closed choice, and it is permitted.
- **Why this character set.** replpp's `UsingDirectives.scanFor` tests
  `line.trim().startsWith("//> using …")`. Java's `String.trim()` removes every char ≤ U+0020, so a
  `\x00`-, `\x01`-, `\x1c`- or `\x1f`-led directive is honoured: a download was attempted (S08
  gate M1, Appendix B). U+007F is not trimmed by Java. It is added to fail closed. The oracle is
  "≤ U+0020", not a list of shapes.
- **Why `\r` splits lines: confirmed by probe (v1.1, ES-m1).** The analyst's re-gate container
  probe (review § `## Spec amendments (bundle-spec2, egress-spec2)`, Appendix B, case c2) ran
  `println("CRSPLIT")\r//> using dep …` and **attempted the `github.com` download**. CRLF and LF
  files with the directive on line 2 did the same (c3, c4), so any line counts, not only the head.
  So `\n`, `\r` and `\r\n` are exactly replpp's line terminators. v1.0's "inferred, not probed" is
  withdrawn.
- **Observations, recorded with that probe, of what does not trigger replpp:**
  - `\f` mid-line does not split (c8);
  - NEL, U+2028 and U+2029 do not split (c5, c6, c11);
  - a DEL-led directive is not honoured (c7), so stripping U+007F is fail-closed only;
  - `//>` followed by two spaces or a tab is inert (c9, c10).
- **H15 is wrong in two places.** §7.3.1's sentence "An indented directive is inert today" and §9
  H15's "the same `using dep` indented by two spaces: exit 0 with no download attempt (inert)" are
  both **false**. Two-space-, `\r`-, `\f`-, `\v`- and control-byte-led directives all attempt the
  download (S04 Pass 3 P3-m1; S08 gate M1). The rest of the H15 row stands (ES-m2). Inert, as
  observed: a BOM-led line (a compile error), an NBSP-led line, `///>`, a mid-line `//>`, and
  Ammonite `import $ivy`. Those are negative controls. The runner may refuse them or admit them.
- **Test 10(b), replaced.**
  - A **coverage probe over the leading byte**: for each byte 0x00–0x20, and for 0x7f, a script
    whose first line is that byte followed by `//> using dep …` must be refused.
  - A script with `x\r//> using dep …` on its first line must be refused.
  - The existing shapes stay (`using resolver`, `using file`, `//> USING DEP …`).

### 1.2 `/scripts` is a directory (restates erratum E2, P3-n1; amends spec §7.1, §7.2's launcher clause and §10 test 5) [doc]

The runner binds its owned temp **directory** at `/scripts`, holding `<name>.sc`. On the container
backend that is a `local`-driver volume, `ro,noexec`. On the namespace backend it is `--ro-bind`. The
script path stays `/scripts/<name>.sc`. In §7.2's launcher clause, "the one read-only script bind at
`/scripts/<name>.sc`" becomes "the one read-only, `noexec` script volume at `/scripts`". Test 5's
allowed set lists `/scripts`.

### 1.3 Container mounts (amends spec §7.1's `/src` and `/out` rows and §7.1.2's last bullet; resolves §11 R2) [doc]

| | Container |
|---|---|
| Source `/src` | a `local`-driver volume of type `none`, `o=bind,ro,noexec` (Docker binds have no `noexec`) |
| Output `/out` | a `local`-driver **overlay** volume, with options `noexec,nosuid,nodev`. Its `upperdir` is the `--output` directory; `lowerdir` and `workdir` are runner-owned scratch beside it, on the same filesystem, removed after the run. A `rw` bind never gets `noexec`, which is why `/out` is an overlay. |

**R2 is resolved.** mountinfo shows `/src ro,…,noexec`, `/out rw,…,noexec overlay` and
`/scripts ro,…,noexec`, and mutants M08 and M09 are killed (S08 gate, deviation 2).

**One `--output` per live invocation** (S08 gate m2). With a shared `upperdir`, overlayfs results are
undefined, and concurrent runs do succeed, so the README's "expected to fail" was false. The
runner refuses an invocation whose `--output` is in use by a live invocation (`usage`, exit 64, one
record; §1.5). The mechanism is S08's choice, but **the lock must live outside the overlay
`upperdir`** (v1.1, ES-m4). v1.0's example, `<output>/.ccg-lock`, sat inside it. That put the lock
in the tool's output and at `/out/.ccg-lock` inside the sandbox, where a tool can unlink it and
free the path for a second runner while the first still runs (*inferred*). Examples that work:
- `flock -n` on a file descriptor opened on the `--output` directory itself;
- a runner state file outside the output, keyed by the directory's device and inode.

A test runs two concurrent invocations and expects exactly one refusal. [S08]

### 1.4 Namespace backend: `/usr` without host submounts (replaces spec §7.1's namespace "Toolchain" cell, and the "with `--tmpfs` masking `/usr/lib/wsl`" clause of §7.2 A5) [doc]

- **The new rule.** `/usr` is bound **entry by entry**, around every host submount below `/usr`, so
  `/usr/lib/wsl` and anything mounted beneath `/usr` is **absent**, not masked. The rest of the
  toolchain cell is unchanged: the Joern bind, the `/lib*` and `/bin` symlinks, `--proc`, `--dev`.
  The minimal `/etc` gains a generated `group` and an `nsswitch.conf` (`files` for passwd, group and
  hosts). The root is `--remount-ro /` after the binds.
- **Why the mask design fails.** bwrap's `--ro-bind` is recursive. Under the spec's `--tmpfs
  /usr/lib/wsl` mask, mountinfo still lists `9p /usr/lib/wsl/drivers` and `overlay
  /usr/lib/wsl/lib`, so A5(b) fails. The mask held for path access only (S08 gate, deviation 1;
  mutant M13 killed).
- **A5(b) has no exemption.** No mount point of any fstype, `tmpfs` included, may lie under `/mnt`,
  `/usr/lib/wsl` or `/run/WSL`. The probe's `tmpfs` exemption goes (S08 gate n2). [S08]
- **`mounts` lists every bind** (spec §5.3: "every host path visible inside"). That means each
  `/usr` entry separately, the `/lib*` and `/bin` binds, and each `/etc` file (S08 gate m6). [S08]
- **Test 5's namespace allowed set** is this bind set.

### 1.5 The `sandbox` record (amends spec §5.3) [S08 for `usage` and `netnsDistinct`; doc otherwise]

- **`refusalReason` adds `usage`** (exit 64). It covers a missing, unknown or malformed runner
  option (`--backend`, `--mode` and the rest), a missing `--`, a bare invocation, and an `--output`
  in use (§1.3). Each of these writes **exactly one** record: S08 gate M2's crash path wrote none.
- **`-h`/`--help`** exits 0 and writes no record. It is the one invocation outside "exactly one per
  invocation".
- **`egressMode`** is recorded only from a validated `--mode`. A `usage` refusal for an invalid or
  missing `--mode` records `offline` (m4).
- **`network`** is `broker` iff a `controlled` invocation was **launched** (P2). Otherwise it is
  `none`, refused records included, so a `--mode controlled` refusal is `egressMode: controlled`,
  `network: none` (deviation 4). `networkName` follows `network`. This replaces "`none` iff
  `offline`".
- **`preconditions.netnsDistinct`** is **by construction from `launcherArgv`**. It is `true` iff the
  invocation launched and its `launcherArgv` creates a new network namespace: `--network none` or
  `--network ccg-egress-<auditId>` (container), `--unshare-all` or `--unshare-net` (namespace). It is
  `false` when nothing launched. The runner never sees inside, and A1(c) is the probe that verifies
  it (deviations 5, m4). In §7.2 A1(c), "and records the result as `preconditions.netnsDistinct`"
  becomes "and passes iff the inodes differ; `netnsDistinct` is §5.3's construction".
- **`toolArgv`, precisely.**
  - The runner replaces the literal token `@AUDIT_ID@` in every tool argument with the `auditId`,
    which is how §5.3's output-stem convention reaches the tool
    (`--param out=/out/@AUDIT_ID@.json`, S11).
  - A launched record's `toolArgv` is the argv as executed, after that substitution and the runner's
    own additions.
  - A refused record's `toolArgv` is the argv the caller passed after `--`.

  S08 gate m3 has the tests. [S08]

### 1.6 Refusal precedence (adds to spec §7) [doc]

Backend resolution, which is read-only (`docker info`, `docker image inspect`, no create), may run
before the argv and directive audit. When an invocation would be refused both with 64 (`usage`, an
argv or directive reason) and with 69 (a backend reason), **it is recorded as 64**. Test 2's
`docker events` window already shows that no container is created either way (deviation 8).

### 1.7 The runner's environment (restates spec §7's "no input from its own environment" as a closed, testable rule; v1.1, ES-m3) [S08]

**The rule is closed, not a list.** Apart from `CCG_AUDIT_DIR` and `CCG_TOOLCHAIN_IMAGE`, nothing in
the caller's environment may influence the runner's behaviour, its `sandbox` record or its
`launcherArgv`. v1.0 restated the rule as two lists of names to "set or clear". That left `PATH`,
`TMPDIR`, `SHELLOPTS`/`BASHOPTS` and exported functions open. And clearing `DOCKER_HOST` and
`DOCKER_CONTEXT` still lets the client take `currentContext` from `$HOME/.docker/config.json`
(*inferred* from Docker's documented precedence, not run).

**The required shape.** Before anything else, the runner re-executes itself under `env -i` with
only these variables:
- a **fixed** `PATH`;
- an **explicitly set** `DOCKER_HOST`, the runner's constant `unix:///var/run/docker.sock`. Setting
  it pins the daemon, and a different socket is a runner edit, not an input;
- `DOCKER_CONFIG`, set to the runner's own Docker-config directory (rule below), so no user
  `config.json` (contexts, proxies, credential helpers) is read;
- **`HOME`, taken from the passwd entry of the runner's uid**, never from the caller's environment.
  The runner already passes it (`sbx_clean_env`), and v1.2 names it on the closed list (E3-1).
  `DOCKER_CONFIG` is pinned, so `HOME` no longer selects a daemon;
- the two `CCG_*` names, passed through.

S08 picks the re-exec mechanism.

**The Docker-config directory (v1.2, E3-2).**
- It is a path **fixed in the runner** under the passwd `HOME`, for example
  `<home>/.local/state/ccg/docker-config`. It is never under `/tmp` or any other shared or
  world-writable tree.
- At each run, the runner creates it if absent, mode `0700`, owned by the runner's uid. It then
  verifies that the directory is **empty**, has that owner and that mode.
- If any check fails, the runner refuses with exit 69, `backend-precondition`, and one record.
  Otherwise a planted `config.json` would bring back exactly what this section removes:
  `credsStore`, `proxies` and `currentContext`.

**Scope (v1.2, ES-P2-m1).**
- **The runner's `#!` line must carry `-p`** (`#!/bin/bash -p`), or an equivalent that makes bash
  ignore `BASH_ENV`, `ENV` and imported functions at startup. Without it, a direct `#!` exec still
  runs a hostile `BASH_ENV` before the first line (analyst Pass 2, run). The current runner has
  the `-p`.
- **The contract covers the runner launched through that `#!` line**, which is how every caller
  (S11, the tests) invokes it. A caller that starts bash explicitly with a hostile `BASH_ENV`
  (`bash offline-run.sh`) runs that file before the runner's first line. That caller already
  executes arbitrary code, so the case is outside the contract, and the README says so.

**Test (polluted environment).** Run the runner **through its `#!` line** with:
- the caller's full environment;
- a fake `docker` planted first on `PATH`;
- `DOCKER_HOST`, `DOCKER_CONTEXT` and `DOCKER_CONFIG` pointing elsewhere;
- `SHELLOPTS=xtrace` and an exported function;
- **`BASH_ENV` and `ENV` naming a file that writes a marker** (v1.2);
- a `$HOME/.docker/config.json` naming another context;
- **a pre-populated Docker-config directory** (v1.2). This case must refuse with 69, as above.

In every other case:
- the record and `launcherArgv` equal a clean run's, modulo `auditId`, timings and paths;
- the marker file is never written.

A mutant that drops the `#!` line's `-p` must fail the test.

### 1.8 Escape-axis precisions (amend spec §7.2)

- **Where the probe launches from, replacing §7.2's "launched through the runner, exactly as a tool
  is launched" [doc].** The probe is launched through the runner's **launch function**, which builds
  `launcherArgv` exactly as for a tool. §7.3.1's closed frontend set and §11 R5's ban on a bypass
  option make this the only route (S08 gate, deviation 9). The production-only paths (the argv
  parser, backend resolution, the record writer) are covered through the real runner by tests 2, 4,
  5 and 10. One check pins the equivalence: a production record's `launcherArgv` equals the probe
  launch's, modulo `auditId`, paths, the `/scripts` mount and the tool tail (S08 gate m5). [S08]
- **A1(b), "no default route" [doc].** A default route is a `0.0.0.0/0` entry in
  `/proc/net/route`, or a `::/0` entry in `/proc/net/ipv6_route`, **without** the `RTF_REJECT` flag
  (`0x0200`). The kernel's `::/0` reject entries on `lo` (flags `00200200`, metric `ffffffff`) are
  unreachable routes, not routes out (deviation 6).
- **A6, values as well as keys [S08].** A new **A6(b′)**: every pass-list value the probe reads from
  its own environment (A6(a)) equals §7.3.4's.
  - `HOME`, `TMPDIR`, `LANG`, `LC_ALL`, `GOTOOLCHAIN`, `DOTNET_CLI_TELEMETRY_OPTOUT`,
    `DOTNET_NOLOGO` and `DOTNET_SKIP_FIRST_TIME_EXPERIENCE` are checked **literally**.
  - `PATH`, `JAVA_HOME`, `GOROOT` and `DOTNET_ROOT` must equal the image's `Config.Env` on the
    container backend, and §7.3.4's stated values on the namespace backend.
  - `JAVA_OPTS` stays under A6(c).

  §7.3.4's values were already normative; A6 only checked keys (S08 gate M3: the
  `GOTOOLCHAIN=auto` and telemetry-on mutants survived). Test 1 carries A6(b′).
- **A6(e), the reading child [doc].** The child that reads may be a program the shell starts:
  `sh -c 'cat /proc/self/environ'` reads `cat`'s environment. That is the descendant environment
  A6(e) bounds. A6(a)'s "its own" rule is for the `env -i` target only (deviation 10).

## 2. Dispositions

"Gate" is `docs/reviews/connected-code-graph-p0-impl.md` § `## S08 (P0-07)`. "P3" is the same
file's § `## S04 (P0-03b)` → `### Pass 3`.

| Item | Source | Decision | Landed | Implementer follow-up |
|---|---|---|---|---|
| E1 / P3-m1 / M1: stripped set | Gate M1; P3 P3-m1; deviation 7 | **Adopted, widened:** ≤ U+0020 plus U+007F, and lines also split at `\r`. The `\r` split is new and inferred. | §1.1 | **S08** (`devops`): the stricter set is landing now. Add the `\r` split, the coverage probe over 0x00–0x20 and 0x7f, and the `x\r//>` fixture. |
| H15 "indented is inert" | P3 P3-m1 | **Corrected.** | §1.1 | — |
| E2 / P3-n1: `/scripts` | P3 P3-n1; gate's E2 note | **Adopted** (restated). | §1.2 | none (already so) |
| Deviation 1: `/usr` entry by entry | Gate, deviation 1 | **Adopted** as the contract. | §1.4 | **S08:** n2 (drop the probe's exemption), m6 (`mounts`). |
| Deviation 2: noexec volumes, `/out` overlay | Gate, deviation 2 | **Adopted.** R2 is resolved. | §1.3 | **S08:** m2 (one `--output` per live invocation, plus a test). |
| Deviation 3: `refusalReason: usage` | Gate, deviation 3; M2 | **Adopted**, with its scope and the one-record rule. | §1.5 | **S08:** M2 (initialise, add the `usage` cases). |
| Deviation 4: `network` always `none` | Gate, deviation 4 | **Adopted:** `broker` iff a launched controlled run. | §1.5 | none |
| Deviation 5 / m4: `netnsDistinct` | Gate, deviation 5; m4 | **Adopted:** by construction, `false` when nothing launched. | §1.5 | **S08:** m4 (`false` on refusals; `egressMode` after validation). |
| Deviation 6: A1(b) `RTF_REJECT` | Gate, deviation 6 | **Adopted.** | §1.8 | none |
| Deviation 8: precedence | Gate, deviation 8 | **Adopted:** 64 over 69. | §1.6 | none |
| Deviation 9: library-mode probe | Gate, deviation 9; m5 | **Adopted:** "through the runner's launch function", with the equivalence pinned. | §1.8 | **S08:** m5 (assert or delete the dead block; the equivalence check). |
| Deviation 10: A6(e) under `sh -c` | Gate, deviation 10 | **Adopted** (precision). | §1.8 | none |
| Deviation 11: `JAVA_OPTS` fixed | Gate, deviation 11 | **No change.** It already full-matches §7.3.4. | — | README only (n5) |
| M3 / gate open question 2: A6 checks values | Gate M3 | **Adopted as a spec erratum** (A6(b′)), not as a test-only addition. The values were already normative, and A6 is where the spec states what the probe proves. | §1.8 | **S08:** env-value tests (already landing). |
| m3: `@AUDIT_ID@` and `toolArgv` | Gate m3 | **Adopted** (precision), so S11 has a contract for the token. | §1.5 | **S08:** m3 tests. |
| n1: runner environment | Gate n1 | **Adopted** (precision). | §1.7 | **S08:** n1. |
| Gate open question 1: land M1 before the erratum? | Gate | **Yes**, it lands now. §1.1 is the erratum it follows. | §1.1 | — |
| M4, m1, n3, n4, n5 | Gate | **No spec change.** These are test and README work. | — | **S08**, as the gate routes them |

## 3. Downstream obligations

- **S08, `devops`** (its current fix round, then one follow-up):
  - §1.1's `\r` line split, the leading-byte coverage probe and the `x\r//>` fixture;
  - §1.3's one-`--output` refusal, with the lock outside the `upperdir`;
  - §1.4's n2 and m6;
  - §1.5's `usage` path, `netnsDistinct`/`egressMode` on refusals and the `toolArgv` tests;
  - §1.7: the `env -i` re-exec with a fixed `PATH`, `DOCKER_HOST`, `DOCKER_CONFIG` and a
    passwd-derived `HOME`; the Docker-config directory checks (exit 69); the `-p` shebang kept; and
    the polluted-environment test, with `BASH_ENV`/`ENV`, a pre-populated config directory and a
    no-`-p` mutant;
  - §1.8's A6(b′) and the launch-equivalence check.

  The README's backend table and its concurrency claim follow.
- **S11, `graph-dba`.** §1.5's `@AUDIT_ID@` contract is now normative. Nothing else changes.
- **Successor-plan changes**, for `teco`'s §14 fold-in (this document does not edit the plan):

  | Plan location | Change |
  |---|---|
  | §4.1 | The S08 errata table points here. E1 is **superseded** by §1.1, and E2 is restated as §1.2. |
  | §2 D6, §3 S04 and S08 rows, §4.2's S04 paragraph, §5.3's S04 row, §9.1's `cpg/sandbox/` row | "with errata E1–E2 (§4.1)" becomes "with `docs/plans/connected-code-graph-egress-spec2.md`". v1.0 said "§8's S04 row", which does not exist; it is §5.3's. §4.2 and §9.1 were missing (ES-m5). |
  | §3 S08 row | Done-condition: egress §10 tests 1–10 **as amended by `-spec2`**: §1.1's test 10(b), §1.3's concurrency test, §1.7's polluted-environment test, and A6(b′) in test 1. |

## 4. v1.1 dispositions (re-gate checklist)

The findings are from `docs/reviews/connected-code-graph-p0-impl.md` § `## Spec amendments
(bundle-spec2, egress-spec2)`, egress-spec2 findings.

| Finding | Disposition |
|---|---|
| ES-m1 `\r` split still "inferred" | **Fixed**, §1.1: confirmed by the review's probe c2, with the negative observations recorded and the runner's extra `\f`/`\v` split permitted. |
| ES-m2 §9 H15 row still says "two spaces … inert" | **Fixed**, §1.1 heading and "H15 is wrong in two places". |
| ES-m3 §1.7 is a list; "clears" leaves inputs open | **Fixed**, §1.7: the closed rule, the `env -i` re-exec with a fixed `PATH`, a set `DOCKER_HOST` and `DOCKER_CONFIG`, and the polluted-environment test. **One addition** beyond the recommendation: `DOCKER_CONFIG` is pinned too, because a cleared context still leaves `config.json`. **One scope statement:** a hostile `BASH_ENV` under `bash offline-run.sh` runs before any re-exec, so it is outside the contract. |
| ES-m4 the lock inside the `upperdir` | **Fixed**, §1.3. |
| ES-m5 plan fold-in incomplete or mislabelled | **Fixed**, §3 table. |
| ES-n1 "case-insensitively" says nothing for `//>` | **Fixed**, §1.1: "any directive counts, in any case". |

### 4.1 v1.2 dispositions

The sources are the analyst's Pass 2 (review § `## Spec amendments (bundle-spec2, egress-spec2)` → `### Pass 2`) and security's Pass 3 (advisory § `## Pass 3 — v1.1 check`, "ES2 §1.7").

| Finding | Disposition |
|---|---|
| ES-P2-m1: the scope rests on the shebang's `-p`; the polluted test lacks `BASH_ENV`/`ENV`; `HOME` drift | **Fixed**, §1.7: `-p` (or an equivalent) is required; the test runs through `#!` with `BASH_ENV`/`ENV`; a no-`-p` mutant must fail; `HOME` is named (as E3-1). |
| E3-1 `HOME` kept by the code, unlisted | **Fixed**, §1.7: `HOME` from passwd is on the closed list. Listing it was preferred to dropping it, because the runner already passes it, and with `DOCKER_CONFIG` pinned it is not a daemon selector. |
| E3-2 the `DOCKER_CONFIG` directory invariant | **Fixed**, §1.7: a fixed, runner-owned path, not under `/tmp`, mode `0700`, verified empty each run, or a 69 `backend-precondition` refusal; the polluted test includes a pre-populated directory. |
| Security Pass 3 observation: the runner does not yet set `DOCKER_HOST`/`DOCKER_CONFIG` | **Routed:** `devops`, S08's §1.7 follow-up (§3). |

## Ready to implement (summary)

Amendments: `docs/plans/connected-code-graph-egress-spec2.md`, read with the egress spec v1.3. It
replaces plan errata E1 and E2.
- **S08:**
  - the `//>` rule: strip ≤ U+0020 and U+007F, split lines at `\n`/`\r`, a coverage probe;
  - one `--output` per live invocation;
  - `usage` refusals with a record;
  - `netnsDistinct` and `egressMode` on refusals;
  - A6(b′) value checks;
  - the runner-environment closure;
  - the launch-equivalence check.
- Everything else here records deviations S08 already made.
