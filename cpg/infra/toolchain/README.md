# ccg-toolchain image

> **Status:** active · **Owner:** `devops` · **Tracks:** C-1001 (P0, S02, S24; P1-48)

Three images from one `Dockerfile`, one content hash:

- **`ccg-toolchain:<hash>`** (target `toolchain`): Joern v4.0.579 plus every FR-1 frontend's
  toolchain, for P0's sandboxed `--network none` builds (S08, S11). Specified by
  `docs/plans/connected-code-graph-p0.md` §4.1 (S02) and §7. It carries **no** Node, indexer
  or clang.
- **`ccg-toolchain-truth:<hash>`** (target `truth`, `FROM toolchain`): the ground-truth tools of
  `docs/plans/connected-code-graph-p0-ml.md` §5.2/§9 and the smoke projects (S24). For S25's
  reference-truth runs, **with network and outside the sandbox**, never in a fetch-off
  run's workspace.
- **`ccg-toolchain-indexer:<hash>`** (target `indexer`, `FROM toolchain`; P1-48): `toolchain` plus
  Node and the two npm SCIP indexers, for P1's offline indexer runs (P1-15). See "The `indexer`
  image" below.

## Build

```bash
cpg/infra/toolchain/build.sh                       # builds ccg-toolchain:<hash>; no-op if it exists
cpg/infra/toolchain/build.sh --truth               # builds ccg-toolchain-truth:<hash> (and the toolchain it is FROM)
cpg/infra/toolchain/build.sh --indexer             # builds ccg-toolchain-indexer:<hash> (and the toolchain it is FROM)
cpg/infra/toolchain/build.sh [--truth|--indexer] --print-tag # just the tag, no build
cpg/infra/toolchain/build.sh [--truth|--indexer] --no-cache  # cold rebuild of the same tag
```

The tag is `sha256(build context)[:12]`. The context is a deterministic tar (sorted names, mtime 0,
uid 0, normalized modes) of exactly `Dockerfile`, `smoke/` and `truth-npm/`, piped to
`docker build` on stdin with no host directory behind it. A change to any pin, smoke file or
lockfile moves the tag, and an existing tag is never rebuilt. All three images carry the same hash, so
an edit to a truth tool also moves the `toolchain` tag even though that image's layers are
unchanged and come from cache. Same convention as `cypher-mcp/build.sh`. A build needs network.
A fetch-off run (S08, S11, S25 off) needs none. A fetch-on run (S25 on) needs the controlled
egress S04/S08 provide; the images grant no network.

**The sandbox runner follows the tag.** `cpg/sandbox/offline-run.sh` takes its default image
from `build.sh --print-tag` (`sbx_default_image`), so the runner's default moves with every change
here. See S24 item 5 below.

## What's in it

Pinned in the `Dockerfile`: the base by digest, every download by its version and its upstream
checksum, and apt packages by Ubuntu snapshot `20261020T000000Z`. The apt pin is enforced in
stage `apt-base`, which every apt-using stage builds from. Its `URIs:` lines are rewritten to
`https://snapshot.ubuntu.com/ubuntu/20261020T000000Z/`, and `apt-snapshot-install` fails the
build if any apt list comes from another host. `apt-get update --snapshot` alone is **not** a
pin: it adds the snapshot alongside the live archive, and apt then takes the newest version from
both. The first S02 image picked up 4 live-archive packages that way. The image keeps the
snapshot sources, so an `apt-get` at run time would also resolve against the snapshot.

| Tool | Version | Source |
|---|---|---|
| Base / JDK | `eclipse-temurin:21.0.12_8-jdk-noble` (Ubuntu 24.04), OpenJDK 21.0.12 | Docker Hub, by digest |
| Joern + frontends | v4.0.579 (`pysrc2cpg`, `jssrc2cpg`, `javasrc2cpg`, `jimple2cpg`, `kotlin2cpg`, `csharpsrc2cpg`, `gosrc2cpg`, `c2cpg`, plus the rest of the release) | GitHub release, sha256 (the same bytes as the host's `~/joern/joern-cli.zip`) |
| Maven | 3.9.16 | Apache, sha512 |
| Gradle | 9.8.2 | services.gradle.org, sha256 |
| Go | 1.27.3 (`GOTOOLCHAIN=local`) | go.dev, sha256 |
| .NET SDK | 10.0.403 (10.0 LTS) | Microsoft, sha512 |
| Kotlin compiler | 2.4.21 | GitHub release, sha256 |
| CMake / bear | 3.28.3 / 3.1.3 | apt snapshot |
| gcc / g++ / make / git | 13.3.0 / 13.3.0 / 4.3 / 2.43.0 | apt snapshot |

The `toolchain` image has **no Node.js**: `jssrc2cpg` bundles astgen. Node lives only in the
`truth` and `indexer` images. Nothing the `truth` stage installs overlaps with S02's list.

`toolchain` is **4.39 GiB** (4,718,150,444 bytes; 4.72 GB), and about 2.6 GiB of that is the
Joern distribution. The full Joern release is kept to match the host. The frontends outside FR-1
(ghidra, abap, swift, rust, php, ruby) make up about 1.2 GiB and could be pruned if size starts
to matter. `truth` is **5.40 GiB** (5,800,705,433 bytes; 5.80 GB), so the truth tools add about
1.01 GiB.

### The `truth` image (S24)

Pinned in the same `Dockerfile` (stage `truth-fetch`, then target `truth`), under `/opt/truth`,
with the tools' versions in the content hash. The npm tools are pinned by committed lockfiles
(`truth-npm/<tool>/package-lock.json`, `npm ci --ignore-scripts`, so every transitive package is
fixed by its sha512 integrity); Go modules by the module proxy and the Go checksum database;
the rest by sha256 (published upstream where it exists; for `scip-clang`, the scip-code Maven
jars and the `scip-dotnet` nupkg the digest was taken on first download, and for the Maven jars
cross-checked against Maven Central's `.sha1`).

| Tool | Version | Role (ML §5.2) | Source |
|---|---|---|---|
| Node.js | 24.22.2 (24 LTS "Krypton") | runtime for the two npm indexers | nodejs.org, sha256 |
| `scip-python` | 0.6.8 | Python reference | npm, lockfile |
| `scip-typescript` | 0.6.0 | TS/JS reference | npm, lockfile |
| `scip-java` | 0.13.2 (`--version` prints `0.0.0-SNAPSHOT`; the jar's sha256 prefix is in `ccg-versions`) | Java/Kotlin: `aggregate` of per-file SCIP shards | GitHub release jar, sha256 |
| `scip-javac` (javac plugin) | 0.13.2 `-all` | Java: `javac -Xplugin:scip` | Maven Central, sha256 |
| `scip-kotlinc` (kotlinc plugin) | 0.13.2 `-all` | Kotlin | Maven Central, sha256 |
| `kotlin-compiler-embeddable` and 6 support jars | 2.4.21 (same as S02's `kotlinc`) | host compiler for `scip-kotlinc`: `kotlinc-scip` | Maven Central, sha256 |
| `scip-dotnet` | 0.2.16 | C# reference | nuget.org nupkg, sha512 |
| `callgraph` (`golang.org/x/tools/cmd/callgraph`) | v0.52.0 | Go reference, `-algo=vta` | Go module proxy + sumdb |
| `scip-go` | v0.2.9 | Go fallback (static targets) | GitHub release, sha256 |
| `scip-clang` | v0.4.1 | C/C++ reference (use `--jobs 2`) | GitHub release, sha256 |
| `scip` CLI | v0.10.0 | `scip print --json` | GitHub release, sha256 |
| `uv` | 0.13.6 | corpus venv | GitHub release, sha256 |
| python3 + venv + pip | 3.12.3 | `scip-python` shells out to `pip list`; corpus venv | apt snapshot |

**Offline type-check of TS fixtures.** The `tsc` that `scip-typescript` depends on (6.0.3, observed
2026-11-07) is at `/opt/truth/npm/scip-typescript/node_modules/.bin/tsc`; it is not on the image's `PATH`, though `node`
(`/opt/truth/node/bin`) is. Run it in this image with `--network none`. The WSL host has `javac`
(JDK 21) but no `node` or `tsc`.

Not in the image: py-tree-sitter and its ten grammars (a `cpg/pyproject.toml` `spike`-extra
dependency; ML note §11 N6, S09's), and any corpus dependency (fetched by S25 at truth time).

**Kotlin runs on the embeddable compiler.** `scip-kotlinc` is built against
`kotlin-compiler-embeddable` (relocated `org.jetbrains.kotlin.com.intellij.*`). On S02's CLI
`kotlinc` it dies with `NoClassDefFoundError .../FlyweightCapableTreeStructure` (observed). The
`truth` image therefore also ships the embeddable compiler (same 2.4.21) and the wrapper
`kotlinc-scip`, which is what Gradle-driven `scip-java index` does as well.

### The `indexer` image (P1-48)

Target `indexer`, `FROM toolchain`. It adds, under `/opt/indexer`:
- Node.js 24.22.2;
- `scip-python` 0.6.8 and `scip-typescript` 0.6.0. These are the same bytes as `truth`'s: both copy
  them from stage `npm-fetch`, which installs `truth-npm/<tool>/package-lock.json` with
  `npm ci --ignore-scripts`;
- `python3` with `venv` and `pip` (apt snapshot). `scip-python` needs an interpreter for its
  environment discovery, and a pre-provisioned venv's `bin/python` resolves to it.

It carries no other truth tool, no smoke project and no `/opt/truth`. `ccg-versions` runs the
`50-indexer.sh` drop-in (node, npm, both indexers, python3). The image is 5,067,839,258 bytes
(4833 MiB), 349,688,814 bytes more than `toolchain`'s 4,718,150,444.

- **Splitting `npm-fetch` out of `truth-fetch` changed no layer.** Across the split
  (`a82faf5219b3` before, `73b9fb76bd2a` after), `toolchain` keeps image ID
  `sha256:a24714d06174…` and `truth` keeps `sha256:ed6be6cf59f9…`. Section (5) of
  `tests/test_toolchain.sh` checks the indexer image.
- **The image still ships `npm`, `npx`, `pip` and `pip3`.** Under `--network none` they cannot
  fetch, but a run that is allowed to call them could. Keeping them out of an indexer run's argv is
  the runner's job, not the image's. `offline-run.sh` has no indexer frontend yet (the
  `-egress-spec3.md` §7.4 revision), so an indexer run is not yet a supported runner path; see
  `cpg/infra/README.md` "Indexer image".
- **The `toolchain` image the sandbox runs carries neither Node nor `/opt/indexer`.** Test
  section (3) checks this.

## Run

The user is `ccg` (uid/gid 1000 with no supplementary groups, the same uid as the lab's host
user), and the working directory is `/work`. `ENTRYPOINT` is empty: the base's
`/__cacert_entrypoint.sh` is dropped, so a plain `docker run` behaves like S08's launch. `CMD` is
`ccg-versions`.

```bash
IMAGE="$(cpg/infra/toolchain/build.sh)"
docker run --rm --network none "$IMAGE" ccg-versions        # every tool's version
docker run --rm --network none --hostname localhost \
  -v "$SRC":/src:ro -v "$OUT":/out "$IMAGE" \
  joern-parse /src --language pythonsrc -o /out/cpg.bin
```

**Pass `--hostname localhost` under `--network none`.** Docker does not put the container's
generated hostname in `/etc/hosts` when there is no network. That makes the JVM's
`InetAddress.getLocalHost()` fail: log4j prints an `UnknownHostException` stack trace. The
failure is immediate, not a stall, and the parse still succeeds. `--hostname localhost`
removes it. S08's runner should set it.

**Run-time fetching is off by default.** No frontend is wrapped with a fetch flag, and
`ENABLE_DEPENDENCY_RESOLVER_V2` is never set. Two egress-relevant `ENV` keys are set:
`GOTOOLCHAIN=local` and `DOTNET_CLI_TELEMETRY_OPTOUT=1`. The first is also written into
`/opt/go/go.env`, so it still holds under `env -i`, where image `ENV` is dropped. The
forbidden-flag gate itself belongs to S08 (`cpg/sandbox/`).

**`/tmp` must be executable.** zstd-jni and jssrc's astgen wrapper execute from the temp
directory. Which mount shape S04/S08 use is the S04 spec's call; see the S02 review's M1 in
`docs/reviews/connected-code-graph-p0-impl.md`.

## Caches

The image ships with **no dependency caches**. `/home/ccg` holds only the skeleton dotfiles,
`/work` is empty, and there is no `.m2`, `.gradle`, `.nuget` or Go module cache anywhere
(checked with `find /`; `/opt/go/pkg` is Go's own toolchain tree, not a module cache). At run
time the tools create their default caches under `$HOME` (`~/.m2`, `~/.gradle`,
`~/.nuget/packages`, `~/go/pkg/mod`). Each `docker run --rm` starts with a fresh, empty home,
so a fetch-off run can only see a cache that is mounted into it on purpose. The truth tools are
installed under `/opt/truth`; the `truth` image was checked the same way (no `.nuget`, `.npm`,
`.m2`, `.gradle` or GOPATH anywhere; `/home/ccg` holds only the skeleton dotfiles, `/work` is
empty). A truth run that fills `~/.m2`, `~/.gradle`, `~/.nuget`, `GOMODCACHE`, `node_modules` or a
venv must do it in its own container or workspace (ML §9, S25).

## Verify: `ccg-versions`

`ccg-versions` prints every tool's version. It **exits 1** if any tool is missing, exits
non-zero, prints nothing, or (for Joern and its frontends) has no versioned jar. It also runs
every `/usr/local/lib/ccg-versions.d/*.sh` in name order, **each in its own subshell** that
reports its own failure count through an `EXIT` trap, so a drop-in's `exit 0` or `set -e` abort
cannot hide a failure (S02 review Pass 2, p2-m1). Each drop-in calls
`v <label> <cmd...>` or `jarv <label> <libdir> <artifact>`. A drop-in that fails `bash -n`
counts as a failure. The `truth` image registers `50-truth.sh`; the `indexer` image registers
`50-indexer.sh`.

## Extending

A new truth tool goes in `truth-fetch` (download, verify, install under `/opt/truth`) and the
`truth` target; never in `toolchain`. Every stage is `FROM apt-base` (or a stage that is), apt
packages only through `apt-snapshot-install`, never `FROM ${BASE}`. Register a version check as a
`ccg-versions.d` drop-in and a smoke project under `smoke/<lang>/`.

## Smoke test: `ccg-truth-smoke`

`docker run --rm --network none "$(cpg/infra/toolchain/build.sh --truth)" ccg-truth-smoke [lang...]`
copies each `smoke/<lang>/` (a few files: an interface or base class, one override, one call
through it) to a scratch dir, indexes it with the language's primary tool and reads the result
back. SCIP tools are read with `scip print --json` and must show the smoke documents, an
`is_implementation` relationship override -> interface method, and a call-site reference to the
interface method. Go is read from `callgraph -algo=vta` and must hold `smoke.total ->
(smoke.Circle).Area` with a `file:line:col`. `go-scip-go` checks the Go fallback. Exit status is
the number of failed languages. It needs no network and writes only under `$TMPDIR`.

This proves the tools run in the image, not their quality on a corpus: C5 on real candidates is S25.

## S24 results (2026-11-01)

All seven languages have a primary reference tool that indexes its smoke project in the image
with `--network none`; **no language is recorded single-source**, and no fallback was needed.
Output of `ccg-truth-smoke`:

| Language | Tool | Result |
|---|---|---|
| Python | `scip-python` 0.6.8 | PASS: 1 document, 32 occurrences, implementation relationship, 1 call-site reference |
| TS/JS | `scip-typescript` 0.6.0 | PASS: 1 document, 18 occurrences, implementation relationship, 1 call-site reference |
| Java | `scip-javac` + `scip-java aggregate` 0.13.2 | PASS: 3 documents, 28 occurrences, implementation relationship, 1 call-site reference |
| Kotlin | `scip-kotlinc` via `kotlinc-scip` + `aggregate` 0.13.2 | PASS: 1 document, 29 occurrences, implementation relationship, 1 call-site reference |
| C# | `scip-dotnet` 0.2.16 | PASS: 3 documents, 57 occurrences, implementation relationship, 1 call-site reference |
| Go | `callgraph -algo=vta` v0.52.0 | PASS: edge `smoke.total -> (smoke.Circle).Area @ main.go:14:44` (21,713 edges in all, stdlib included) |
| C/C++ | `scip-clang` v0.4.1 (`--jobs 2`) | PASS: 3 documents, 81 occurrences, implementation relationship, 1 call-site reference; the fixture includes `<vector>`, `<atomic>` and `<immintrin.h>` and the index must hold `_mm_add_ps` and `std::atomic` |
| Go fallback | `scip-go` v0.2.9 | PASS (not needed) |

The regression checks are `tests/test_toolchain.sh` (33 checks, all passed in the final run):
tag determinism, the `20-bad` + `40-exit` drop-in pair (non-zero in both images, zero in the
pre-S24 control `ccg-toolchain:d3184effb566`), no truth tool in `toolchain`, the smoke baseline,
and a smoke mutant per assertion and language (override edge removed, call through the
interface removed; every one FAILS). It also hides the compiler's own headers
(`--tmpfs /usr/lib/gcc/x86_64-linux-gnu/13/include`, `--tmpfs /usr/include/c++`): the C++ smoke
then FAILS, so it is header-sensitive. `ccg-versions` exits 0 in both images under
`--network none`. It does not run in the build; run it after `build.sh --truth`.

**No system clang in the image.** The ML note asked for "a matching clang/LLVM". Measured:
`scip-clang` is self-contained (its `--version` prints "Based on Clang/LLVM 2078da43e25a…", a
bundled LLVM) and takes the system headers from the compiler named in the compilation database,
which for CMake on this image is `/usr/bin/c++` (S02's gcc 13.3.0 and libstdc++). The header-
sensitive smoke passed with apt `clang` 18.1.3 present, and passed with `/usr/lib/llvm-18` and
`/usr/lib/clang` hidden, so `clang` was dropped from the `truth` stage (1.01 GiB over `toolchain`
instead of 1.24). **Limit:** a corpus whose compilation database names `clang`/`clang++` needs
that compiler on `PATH`, because `scip-clang` runs it to find its resource directory; with the
apt clang 18.1.3 present such a compdb indexed the smoke with `num errored TUs: 0`, with the
compiler hidden CMake could not even configure. Such a corpus is an S25 decision (add a pinned
clang to the `truth` stage then). Also: `num errored TUs: 0` is **not** evidence that headers
resolved (it stayed 0 with the gcc headers hidden); the smoke checks for system-header symbols
instead.

Findings that bind S25:

- `scip-clang` indexes only files **under its working directory**: run it from the project root
  (`--compdb-path build/compile_commands.json`). Run from `build/` it exits 0 with an empty index
  (`scip print --json` then has no `documents` key).
- `scip-java` here is the `org.scip-code` rewrite (`index`, `aggregate`, `snapshot`; no
  `index-semanticdb`). The smoke uses the manual-configuration path (plugin on `javac`, then
  `aggregate`). The automatic `scip-java index` over Maven/Gradle resolves plugins and
  dependencies from the network at index time and was **not** exercised.
- `scip-python` needs `pip` on `PATH` (installed); `scip-dotnet` restores through NuGet, so the
  smoke project sets `NuGetAudit=false` to avoid an `api.nuget.org` call. A corpus with package
  references needs network, as ML §9 expects.
- `scip-java --version` and the `scip-dotnet` SCIP `tool_info` report `0.0.0-SNAPSHOT` /
  `0.1.0-SNAPSHOT`; use the pinned jar/nupkg digest, not those strings, as identity.
- `callgraph` output includes the standard library's edges; filter by package.
- Sizes: `toolchain` 4,718,150,444 bytes (4.39 GiB); `truth` 5,800,705,433 bytes (5.40 GiB).
- Network used by the truth stages: `snapshot.ubuntu.com`, `nodejs.org`, `registry.npmjs.org`,
  `github.com` release assets, `repo1.maven.org`, `proxy.golang.org`, `sum.golang.org`,
  `api.nuget.org`. Smoke runs use none.

**Item 5 (sandbox re-run), final tag `a82faf5219b3`.** `CCG_TOOLCHAIN_IMAGE=ccg-toolchain:a82faf5219b3
cpg/sandbox/tests/test_sandbox.sh --only 3,6,9,10` -> 132 passed, 0 failed, no container or
volume left (tests 3, 6, 9 and 10 including 10(c); test 4's record checks run with them). The
runner's default image needs no edit: `sbx_default_image` runs `build.sh --print-tag` (under
`env -i`), which prints `ccg-toolchain:a82faf5219b3`. A content change here moves the sandbox
default at once, so re-run those tests when the `toolchain` tag moves.

## S02 results (2026-10-29)

- **Cold build: 5 m 31 s for the image at the current tag.** The first image's cold build took
  14 m 24 s, because the 2.1 GB Joern download reset and resumed. The build client's peak RSS was
  55 MB, and the VM kept about 4 GB available.
- **Apt pin.** All 116 `Get:` lines in the build log are from `snapshot.ubuntu.com`. The review's
  snapshot-membership probe (`docs/reviews/connected-code-graph-p0-impl.md`, S02 Appendix A)
  reports `checked 70 not-in-snapshot 0`. The first image reported 4.
- **`ccg-versions`.** Under `--network none` it exits 0 and ends `all checks passed`. With
  `PATH=/usr/bin:/bin` it exits 1 (6 checks failed). With `JOERN_HOME=/nonexistent` it exits 1
  (9 checks failed). With a failing drop-in and a drop-in that has a syntax error, it exits 1.
- **`joern-parse`.** An 8-line, one-file Python project under
  `--network none --hostname localhost --memory 3g` exits 0, logs 0 `UnknownHostException`, and
  writes a 99,398-byte `cpg.bin` in about 2–5 s. The exact size depends on the sample; the
  reviewer's sample produced 90,247 bytes.
