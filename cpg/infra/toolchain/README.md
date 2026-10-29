# ccg-toolchain image

> **Status:** active · **Owner:** `devops` · **Tracks:** C-1001 (P0, S02)

Joern v4.0.579 plus every FR-1 frontend's toolchain in one image, for P0's sandboxed
`--network none` builds (S08, S11) and the frontend-fitness runs (S25–S28). Specified by
`docs/plans/connected-code-graph-p0.md` §4.1 (S02) and §7. S24 appends the ground-truth layer
to the same `Dockerfile`, strictly after S02.

## Build

```bash
cpg/infra/toolchain/build.sh              # builds ccg-toolchain:<hash>; no-op if it exists
cpg/infra/toolchain/build.sh --print-tag  # just the tag, no build
cpg/infra/toolchain/build.sh --no-cache   # cold rebuild of the same tag
```

The tag is `sha256(Dockerfile)[:12]`. The `Dockerfile` is the only build input — it is piped to
`docker build` with no build context — so a change to any pin moves the tag, and an existing tag
is never rebuilt. Same convention as `cypher-mcp/build.sh`. A build needs network. A fetch-off
run (S08, S11, S25 off) needs none. A fetch-on run (S25 on) needs the controlled egress S04/S08
provide; the image itself grants no network.

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

The image has **no Node.js**: `jssrc2cpg` bundles astgen. S24 adds Linux Node.js LTS with its
ground-truth layer. Nothing else S02 installs overlaps with S24's list
(`docs/plans/connected-code-graph-p0-ml.md`).

The image is **4.39 GiB** (4,718,150,014 bytes; 4.72 GB), and about 2.6 GiB of that is the
Joern distribution. The full Joern release is kept to match the host. The frontends outside FR-1
(ghidra, abap, swift, rust, php, ruby) make up about 1.2 GiB and could be pruned if size starts
to matter.

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
so a fetch-off run can only see a cache that is mounted into it on purpose. S24 must install
its truth tools under `/opt` and must not pre-populate any cache under `/home/ccg` or `/work`.

## Verify: `ccg-versions`

`ccg-versions` prints every tool's version. It **exits 1** if any tool is missing, exits
non-zero, prints nothing, or (for Joern and its frontends) has no versioned jar. It also sources
every `/usr/local/lib/ccg-versions.d/*.sh` in name order. Each drop-in calls
`v <label> <cmd...>` or `jarv <label> <libdir> <artifact>` and shares the failure count. A
drop-in that fails `bash -n` counts as a failure.

## Extending (S24)

S24 adds to the `Dockerfile` without editing any S02 layer:

- `[S24-STAGE]` takes a new download stage. Build it `FROM apt-base`, and reuse `fetch-verify`.
- `[S24-LAYER]` sits after the S02 runtime layers and before the non-root tail. Install there
  with `apt-snapshot-install` (that keeps the apt pin), put the tools under `/opt`, and register
  their checks as a drop-in such as `/usr/local/lib/ccg-versions.d/50-truth.sh`.

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
