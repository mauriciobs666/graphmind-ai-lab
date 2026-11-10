# cpg/infra/toolchain/Dockerfile — the `ccg-toolchain` image (C-1001 P0, step S02).
#
# Joern v4.0.579 plus every frontend's toolchain, for P0's sandboxed `--network none`
# builds and the frontend-fitness runs (docs/plans/connected-code-graph-p0.md §4.1, §7).
# Built by ./build.sh, which tags it ccg-toolchain:<sha256(build context)[:12]>. Results
# and usage: ./README.md.
#
# PINNING. Every input is pinned so the content-hash tag is a real identity:
#   - base image by DIGEST (the tag is only a label for humans);
#   - apt packages through an Ubuntu snapshot date: stage `apt-base` rewrites every
#     `URIs:` line to https://snapshot.ubuntu.com/ubuntu/<SNAPSHOT>/ and
#     `apt-snapshot-install` refuses to install if any apt list came from another host.
#     (`apt-get update --snapshot` alone is NOT a pin here: it ADDS the snapshot to the
#     live archive.ubuntu.com sources and apt takes the highest version of the union.)
#   - every downloaded archive by an exact version AND a checksum published upstream;
#     a mismatch fails the build.
# Network is needed at BUILD time only. Nothing here enables dependency fetching at
# RUN time: no frontend is wrapped with a fetch flag, ENABLE_DEPENDENCY_RESOLVER_V2 is
# never set, and GOTOOLCHAIN=local stops `go` from downloading a newer toolchain.
#
# The build context is NOT the host tree: build.sh pipes a deterministic tar of exactly
# this file plus smoke/ and truth-npm/ on stdin, so those three are the only build inputs
# and the tag is the hash of that tar (build.sh). Only `npm-fetch` (truth-npm/) and the
# `truth` target (smoke/) COPY from it.
#
# LAYOUT
#   stage `apt-base`     base + snapshot-only apt sources + apt-snapshot-install
#   stage `fetch`        S02 downloads (verified, unpacked)
#   stage `npm-fetch`    S24 downloads, part 1: Node 24 + the lockfile-pinned npm indexers
#   stage `truth-fetch`  S24 downloads, part 2 (FROM npm-fetch): the other truth tools
#   stage `toolchain`    S02 runtime (the image the sandbox runs): apt, COPY --from=fetch,
#                        ENV, ccg-versions, non-root user + workdir. NOT touched by S24's
#                        tools: no Node, no indexers, no clang in it.
#   stage `truth`        FROM toolchain: the truth tools + smoke projects, as root
#                        and back to ccg. Image ccg-toolchain-truth:<same hash>.
#   stage `indexer`      FROM toolchain (P1-48): Node + scip-python/scip-typescript from
#                        `npm-fetch`, plus python3. Image ccg-toolchain-indexer:<same hash>.
# Any package/dependency cache a truth tool populates must NOT land under /home/ccg or
# /work (the paths a sandboxed fetch-off run inherits) — see README "Caches".

# eclipse-temurin:21.0.12_8-jdk-noble (Ubuntu 24.04, OpenJDK 21.0.12) — index digest.
ARG BASE=eclipse-temurin@sha256:4d271cd5e0624598cf563342f47281b09cb364bc13acbbd7251f49f83470018d
ARG SNAPSHOT=20261020T000000Z

# =================================================================================
# stage apt-base (S02): every later stage (S24's too) builds FROM this, so all apt
# traffic goes to the snapshot only. Use `apt-snapshot-install <pkg>...` for apt work.
# =================================================================================
FROM ${BASE} AS apt-base
ARG SNAPSHOT
RUN sed -Ei "s#^URIs: http://(archive|security)\.ubuntu\.com/ubuntu/?\$#URIs: https://snapshot.ubuntu.com/ubuntu/${SNAPSHOT}/#" \
      /etc/apt/sources.list.d/ubuntu.sources \
 && if grep -hE '^URIs:' /etc/apt/sources.list.d/*.sources | grep -vqF "https://snapshot.ubuntu.com/ubuntu/${SNAPSHOT}/"; then \
      echo "apt-base: a non-snapshot apt source remains:" >&2; grep -hE '^URIs:' /etc/apt/sources.list.d/*.sources >&2; exit 1; fi \
 && if [ -s /etc/apt/sources.list ] && grep -qE '^deb ' /etc/apt/sources.list; then \
      echo "apt-base: /etc/apt/sources.list has live entries" >&2; exit 1; fi
# apt-snapshot-install <pkg>...: update, fail closed unless every downloaded list is
# from snapshot.ubuntu.com, install, drop the lists.
COPY <<'EOF' /usr/local/bin/apt-snapshot-install
#!/bin/sh
set -eu
apt-get update
bad="$(ls /var/lib/apt/lists | grep -vE '^(snapshot\.ubuntu\.com_.*|lock|partial|auxfiles)$' || true)"
if [ -n "$bad" ]; then
  echo "apt-snapshot-install: apt lists from a non-snapshot host:" >&2; echo "$bad" >&2; exit 1
fi
apt-get install -y --no-install-recommends "$@"
rm -rf /var/lib/apt/lists/*
EOF
RUN chmod 0755 /usr/local/bin/apt-snapshot-install

# =================================================================================
# stage fetch (S02): download + verify + unpack. Kept out of the final image so the
# 2.1 GB Joern zip never becomes a layer; only unpacked trees are copied forward.
# Joern first and alone, so its download stays cached when a later pin changes.
# =================================================================================
FROM apt-base AS fetch
RUN apt-snapshot-install curl ca-certificates unzip

# fetch-verify <url> <out> <sha256|sha512> <hex>: resumable download with retries
# (a 2.1 GB transfer reset mid-stream on the first build — curl exit 56), then a
# checksum check that fails the build on mismatch. S24 may reuse it.
COPY <<'EOF' /usr/local/bin/fetch-verify
#!/bin/sh
set -eu
url="$1"; out="$2"; algo="$3"; sum="$4"
i=0
until curl -fsSL --connect-timeout 30 -C - -o "$out" "$url"; do
  i=$((i + 1)); [ "$i" -lt 8 ] || { echo "fetch-verify: giving up on $url" >&2; exit 1; }
  echo "fetch-verify: retry $i for $url" >&2; sleep 5
done
echo "$sum  $out" | "${algo}sum" -c -
EOF
RUN chmod 0755 /usr/local/bin/fetch-verify
WORKDIR /dl

# Joern v4.0.579 — GitHub release asset digest (same bytes as the host's ~/joern zip).
ARG JOERN_VERSION=v4.0.579
ARG JOERN_SHA256=42ef562ebf6dbd49c1dd17bc49c3657782171fb04543534d34c6599b3d19b7c1
RUN fetch-verify "https://github.com/joernio/joern/releases/download/${JOERN_VERSION}/joern-cli.zip" \
      joern-cli.zip sha256 "${JOERN_SHA256}" \
 && mkdir -p /opt/joern && unzip -q joern-cli.zip -d /opt/joern && rm joern-cli.zip \
 && test -f "/opt/joern/joern-cli/lib/io.joern.joern-cli-${JOERN_VERSION#v}.jar"

# Go — go.dev/dl published sha256.
ARG GO_VERSION=1.27.3
ARG GO_SHA256=65efbc80fdc1fa2cd2631cad5b481777a68e8725f34e89f50fc6ac714e379b40
RUN fetch-verify "https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz" go.tgz sha256 "${GO_SHA256}" \
 && tar -C /opt -xzf go.tgz && rm go.tgz \
 && sed -i 's/^GOTOOLCHAIN=auto$/GOTOOLCHAIN=local/' /opt/go/go.env \
 && grep -qx 'GOTOOLCHAIN=local' /opt/go/go.env
# ^ backstop for the ENV below: `env -i` drops image ENV, and go.env ships GOTOOLCHAIN=auto.

# Maven — Apache published sha512.
ARG MAVEN_VERSION=3.9.16
ARG MAVEN_SHA512=831a8591fe20c8243b1dbe7d71e3244f31d1665b0804b2e825e38cbbe5ce0cafb8338851f90780735568773e0a6cd07bbec107cda0b896b008b861075358b6f6
RUN fetch-verify "https://repo.maven.apache.org/maven2/org/apache/maven/apache-maven/${MAVEN_VERSION}/apache-maven-${MAVEN_VERSION}-bin.tar.gz" \
      maven.tgz sha512 "${MAVEN_SHA512}" \
 && tar -C /opt -xzf maven.tgz && mv "/opt/apache-maven-${MAVEN_VERSION}" /opt/maven && rm maven.tgz

# Gradle — services.gradle.org published sha256.
ARG GRADLE_VERSION=9.8.2
ARG GRADLE_SHA256=21bd5da2de49b642904077f38c0efdc7778e92b3ed0889392dca61b97e389607
RUN fetch-verify "https://services.gradle.org/distributions/gradle-${GRADLE_VERSION}-bin.zip" \
      gradle.zip sha256 "${GRADLE_SHA256}" \
 && unzip -q gradle.zip -d /opt && mv "/opt/gradle-${GRADLE_VERSION}" /opt/gradle && rm gradle.zip

# Kotlin compiler — GitHub release asset digest.
ARG KOTLIN_VERSION=2.4.21
ARG KOTLIN_SHA256=7cc140e76daf416a0424a5557f99afd7ca89ebd463ea1101261e3e01c33e75cd
RUN fetch-verify "https://github.com/JetBrains/kotlin/releases/download/v${KOTLIN_VERSION}/kotlin-compiler-${KOTLIN_VERSION}.zip" \
      kotlinc.zip sha256 "${KOTLIN_SHA256}" \
 && unzip -q kotlinc.zip -d /opt && rm kotlinc.zip

# .NET SDK (10.0 LTS) — Microsoft releases.json published sha512.
ARG DOTNET_SDK_VERSION=10.0.403
ARG DOTNET_SHA512=b7b3d8461cf98b07d2552ce9199702fa3ad010ac2671a5f8cd5136f7534e50f6855ba7bbb02b11494563f1d04156276a31cda1103b900e5eecd0d9473dc7b9c5
RUN fetch-verify "https://builds.dotnet.microsoft.com/dotnet/Sdk/${DOTNET_SDK_VERSION}/dotnet-sdk-${DOTNET_SDK_VERSION}-linux-x64.tar.gz" \
      dotnet.tgz sha512 "${DOTNET_SHA512}" \
 && mkdir -p /opt/dotnet && tar -C /opt/dotnet -xzf dotnet.tgz && rm dotnet.tgz

# =================================================================================
# stage truth-fetch (S24; ML note §5.2, §9): the ground-truth tools, downloaded and
# verified here and copied into the `truth` target only. Built FROM fetch (an apt-base
# descendant) for fetch-verify, curl and the Go toolchain. Every input is pinned:
#   - release assets by upstream sha256 (TOFU-pinned for assets upstream publishes no
#     digest for: scip-clang, the scip-code Maven jars, the scip-dotnet nupkg — where
#     Maven Central's own .sha1 was cross-checked);
#   - npm tools by a committed package-lock.json (truth-npm/<tool>/), installed with
#     `npm ci`, so every transitive package is pinned by its sha512 integrity;
#   - Go modules by the module proxy + sumdb (go.sum is verified against sum.golang.org).
# Everything lands under /opt/truth; no cache is copied forward.
# The Node + npm-indexer part is its own stage, `npm-fetch` (P1-48), so the `indexer` target can
# copy just that; `truth-fetch` continues FROM it with the remaining tools. The instructions and
# their order are unchanged by the split, so an existing layer cache still hits.
# =================================================================================
FROM fetch AS npm-fetch
RUN apt-snapshot-install xz-utils
WORKDIR /dl

# Node.js 24 LTS ("Krypton", maintenance LTS line: the oldest LTS the npm indexers' engines
# cover; v26 became LTS only on 2026-10-28) — nodejs.org SHASUMS256.txt.
ARG NODE_VERSION=24.22.2
ARG NODE_SHA256=48cc0e21e048790e54c0cf5749c2725005313d652651c40b7a21feb9697aaf75
RUN fetch-verify "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-x64.tar.xz" node.txz sha256 "${NODE_SHA256}" \
 && mkdir -p /opt/truth/node && tar -C /opt/truth/node --strip-components=1 -xJf node.txz && rm node.txz

# scip-python 0.6.8 and scip-typescript 0.6.0: the committed lockfiles (truth-npm/) fix
# every transitive package. `--ignore-scripts`: no package may run code at install time.
ARG SCIP_PYTHON_VERSION=0.6.8
ARG SCIP_TYPESCRIPT_VERSION=0.6.0
COPY truth-npm /dl/truth-npm
RUN export PATH=/opt/truth/node/bin:$PATH \
 && for t in scip-python scip-typescript; do \
      mkdir -p /opt/truth/npm/$t && cp /dl/truth-npm/$t/package.json /dl/truth-npm/$t/package-lock.json /opt/truth/npm/$t/ \
      && (cd /opt/truth/npm/$t && npm ci --ignore-scripts --no-audit --no-fund) || exit 1; \
    done \
 && /opt/truth/npm/scip-python/node_modules/.bin/scip-python --version | grep -qF "${SCIP_PYTHON_VERSION}" \
 && /opt/truth/npm/scip-typescript/node_modules/.bin/scip-typescript --version | grep -qF "${SCIP_TYPESCRIPT_VERSION}" \
 && rm -rf /root/.npm

FROM npm-fetch AS truth-fetch
# scip CLI v0.10.0 — GitHub release asset digest.
ARG SCIP_VERSION=v0.10.0
ARG SCIP_SHA256=eeb28ebbff443c01609fb1691809958f3f0c7a8651af689a5bc8560061cd8ab0
RUN fetch-verify "https://github.com/sourcegraph/scip/releases/download/${SCIP_VERSION}/scip-linux-amd64.tar.gz" scip.tgz sha256 "${SCIP_SHA256}" \
 && mkdir -p /opt/truth/bin && tar -C /opt/truth/bin -xzf scip.tgz scip && rm scip.tgz

# scip-java v0.13.2 (the org.scip-code rewrite: `index`/`aggregate`/`snapshot`) — the
# standalone jar from the GitHub release, sha256 published; plus its javac and Kotlin
# compiler plugins from Maven Central (the `-all` jars are the self-contained ones; the thin
# scip-javac jar fails with ClassNotFoundException: ScipOptions).
ARG SCIP_JAVA_VERSION=0.13.2
ARG SCIP_JAVA_SHA256=5248136592d19ea2c468f64ebd18587d32de7c961536ff19346f8a025af1d13c
ARG SCIP_JAVAC_SHA256=5bb0d844f4b96becc25f01d64537832dd13c090c1e231472cfd209c70d988ade
ARG SCIP_KOTLINC_SHA256=464553a424af9145ec5996b8a32b6e2c5f4ee5c307264a7244568e8f48095ce0
RUN mkdir -p /opt/truth/scip-java \
 && fetch-verify "https://github.com/sourcegraph/scip-java/releases/download/v${SCIP_JAVA_VERSION}/scip-java-v${SCIP_JAVA_VERSION}" \
      /opt/truth/scip-java/scip-java.jar sha256 "${SCIP_JAVA_SHA256}" \
 && fetch-verify "https://repo1.maven.org/maven2/org/scip-code/scip-javac/${SCIP_JAVA_VERSION}/scip-javac-${SCIP_JAVA_VERSION}-all.jar" \
      /opt/truth/scip-java/scip-javac.jar sha256 "${SCIP_JAVAC_SHA256}" \
 && fetch-verify "https://repo1.maven.org/maven2/org/scip-code/scip-kotlinc/${SCIP_JAVA_VERSION}/scip-kotlinc-${SCIP_JAVA_VERSION}-all.jar" \
      /opt/truth/scip-java/scip-kotlinc.jar sha256 "${SCIP_KOTLINC_SHA256}"

# The Kotlin compiler in its EMBEDDABLE packaging (same version as S02's kotlinc, 2.4.21).
# scip-kotlinc is built against kotlin-compiler-embeddable (relocated org.jetbrains.kotlin.
# com.intellij.*); the CLI kotlinc's compiler jar is not relocated, so the plugin dies there
# with NoClassDefFoundError .../FlyweightCapableTreeStructure. Gradle runs the embeddable
# one, so this is the same pairing `scip-java index` uses; here it is installed offline.
# Jars from Maven Central, sha256 pinned (cross-checked against Central's .sha1).
# Must equal S02's kotlinc (build.txt); a bump of one without the other fails the build here.
ARG KOTLIN_EMBEDDABLE_VERSION=2.4.21
RUN grep -q "^${KOTLIN_EMBEDDABLE_VERSION}" /opt/kotlinc/build.txt \
 && mkdir -p /opt/truth/kotlinc-embeddable && cd /opt/truth/kotlinc-embeddable \
 && M=https://repo1.maven.org/maven2 \
 && fetch-verify $M/org/jetbrains/kotlin/kotlin-compiler-embeddable/${KOTLIN_EMBEDDABLE_VERSION}/kotlin-compiler-embeddable-${KOTLIN_EMBEDDABLE_VERSION}.jar kotlin-compiler-embeddable-${KOTLIN_EMBEDDABLE_VERSION}.jar sha256 ef19419c765e7ac8404465fa026ca8fc4dbb8a822de0fc79aa53c8dce29f1d02 \
 && fetch-verify $M/org/jetbrains/kotlin/kotlin-stdlib/${KOTLIN_EMBEDDABLE_VERSION}/kotlin-stdlib-${KOTLIN_EMBEDDABLE_VERSION}.jar kotlin-stdlib-${KOTLIN_EMBEDDABLE_VERSION}.jar sha256 bd8250210584cb659847dce9cda660f8e9906e5def7fbebcea93dc6dcb6a88a3 \
 && fetch-verify $M/org/jetbrains/kotlin/kotlin-script-runtime/${KOTLIN_EMBEDDABLE_VERSION}/kotlin-script-runtime-${KOTLIN_EMBEDDABLE_VERSION}.jar kotlin-script-runtime-${KOTLIN_EMBEDDABLE_VERSION}.jar sha256 a658cf5902a1525a06c264a5e42b437e97a6679ddeef8ca641246f270cf6a013 \
 && fetch-verify $M/org/jetbrains/kotlin/kotlin-reflect/1.6.10/kotlin-reflect-1.6.10.jar kotlin-reflect-1.6.10.jar sha256 3277ac102ae17aad10a55abec75ff5696c8d109790396434b496e75087854203 \
 && fetch-verify $M/org/jetbrains/kotlin/kotlin-daemon-embeddable/${KOTLIN_EMBEDDABLE_VERSION}/kotlin-daemon-embeddable-${KOTLIN_EMBEDDABLE_VERSION}.jar kotlin-daemon-embeddable-${KOTLIN_EMBEDDABLE_VERSION}.jar sha256 7b3ef24674e5bcffcadfc4e43202ea314194939ecbcdd90a4770b2f1b04e0d07 \
 && fetch-verify $M/org/jetbrains/kotlinx/kotlinx-coroutines-core-jvm/1.8.0/kotlinx-coroutines-core-jvm-1.8.0.jar kotlinx-coroutines-core-jvm-1.8.0.jar sha256 9860906a1937490bf5f3b06d2f0e10ef451e65b95b269f22daf68a3d1f5065c5 \
 && fetch-verify $M/org/jetbrains/annotations/13.0/annotations-13.0.jar annotations-13.0.jar sha256 ace2a10dc8e2d5fd34925ecac03e4988b2c0f851650c94b8cef49ba1bd111478

# scip-go v0.2.9 (the Go fallback, static targets only) — GitHub release asset digest.
ARG SCIP_GO_VERSION=v0.2.9
ARG SCIP_GO_SHA256=4ec4e45ecdbe6f43055ed555254e049f46f28389ade438db3d59679651fa1292
RUN fetch-verify "https://github.com/sourcegraph/scip-go/releases/download/${SCIP_GO_VERSION}/scip-go-linux-amd64.tar.gz" scip-go.tgz sha256 "${SCIP_GO_SHA256}" \
 && tar -C /opt/truth/bin -xzf scip-go.tgz scip-go && rm scip-go.tgz

# callgraph (golang.org/x/tools/cmd/callgraph) v0.52.0 — built from the module proxy,
# verified against the Go checksum database. Statically linked; GOPATH/GOMODCACHE in /tmp.
ARG CALLGRAPH_VERSION=v0.52.0
RUN GOBIN=/opt/truth/bin GOPATH=/tmp/gopath CGO_ENABLED=0 /opt/go/bin/go install "golang.org/x/tools/cmd/callgraph@${CALLGRAPH_VERSION}" \
 && rm -rf /tmp/gopath

# scip-clang v0.4.1 — GitHub release binary (no digest published; sha256 pinned here).
ARG SCIP_CLANG_VERSION=v0.4.1
ARG SCIP_CLANG_SHA256=a434d26a079fd80e1996ba79276cb1ef86b2b2ad28b13e24386285db32c59046
RUN fetch-verify "https://github.com/sourcegraph/scip-clang/releases/download/${SCIP_CLANG_VERSION}/scip-clang-x86_64-linux" \
      /opt/truth/bin/scip-clang sha256 "${SCIP_CLANG_SHA256}" && chmod 0755 /opt/truth/bin/scip-clang

# uv 0.13.6 — GitHub release asset digest.
ARG UV_VERSION=0.13.6
ARG UV_SHA256=f29c3e94ffabd7e6f3ba521dba169a9b7ec12100bf3257087f0f24f6123f3645
RUN fetch-verify "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-x86_64-unknown-linux-gnu.tar.gz" uv.tgz sha256 "${UV_SHA256}" \
 && tar -C /opt/truth/bin --strip-components=1 -xzf uv.tgz uv-x86_64-unknown-linux-gnu/uv uv-x86_64-unknown-linux-gnu/uvx && rm uv.tgz

# scip-dotnet 0.2.16 — the .nupkg from nuget.org (installed in `truth`, which has the SDK).
ARG SCIP_DOTNET_VERSION=0.2.16
ARG SCIP_DOTNET_SHA512=51468f4029ea21615819768ab608e27813f4591d2e1fc2f9e69c2a86d3891ffdc497ecc2179fdc5d75f7d28617adf2986626c38ee9f692753caf5db5da2f9e5b
RUN mkdir -p /opt/truth/nupkg \
 && fetch-verify "https://api.nuget.org/v3-flatcontainer/scip-dotnet/${SCIP_DOTNET_VERSION}/scip-dotnet.${SCIP_DOTNET_VERSION}.nupkg" \
      "/opt/truth/nupkg/scip-dotnet.${SCIP_DOTNET_VERSION}.nupkg" sha512 "${SCIP_DOTNET_SHA512}"

# =================================================================================
# stage toolchain (S02)
# =================================================================================
FROM apt-base AS toolchain
ARG SNAPSHOT

LABEL ccg-toolchain=1 \
      org.opencontainers.image.title="ccg-toolchain" \
      org.opencontainers.image.description="Joern v4.0.579 + frontend toolchains for C-1001 P0 sandboxed builds" \
      org.opencontainers.image.version="joern-4.0.579" \
      ccg.joern.version="4.0.579" \
      ccg.apt.snapshot="${SNAPSHOT}"

# C/C++ build toolchain + CMake + bear (compile_commands.json), git for corpus checkouts,
# libicu for the .NET SDK. Versions are fixed by the snapshot date; ccg-versions reports
# what resolved.
RUN apt-snapshot-install build-essential cmake bear git unzip libicu74

COPY --from=fetch /opt/joern   /opt/joern
COPY --from=fetch /opt/go      /opt/go
COPY --from=fetch /opt/maven   /opt/maven
COPY --from=fetch /opt/gradle  /opt/gradle
COPY --from=fetch /opt/kotlinc /opt/kotlinc
COPY --from=fetch /opt/dotnet  /opt/dotnet

ENV JOERN_HOME=/opt/joern/joern-cli \
    GOROOT=/opt/go \
    GOTOOLCHAIN=local \
    MAVEN_HOME=/opt/maven \
    GRADLE_HOME=/opt/gradle \
    DOTNET_ROOT=/opt/dotnet \
    DOTNET_CLI_TELEMETRY_OPTOUT=1 \
    DOTNET_NOLOGO=1 \
    DOTNET_SKIP_FIRST_TIME_EXPERIENCE=1 \
    PATH=/opt/joern/joern-cli:/opt/go/bin:/opt/maven/bin:/opt/gradle/bin:/opt/kotlinc/bin:/opt/dotnet:/opt/java/openjdk/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# `ccg-versions` prints every pinned tool's version (README "Verify"; S08's audit can
# record its output as the toolchain identity) and EXITS NON-ZERO if any tool is missing,
# fails, or reports an empty version — so it is a usable gate. Later layers extend it
# with drop-ins: every /usr/local/lib/ccg-versions.d/*.sh is sourced in name order and
# can call `v <label> <cmd...>` / `jarv <label> <dir> <artifact>`, sharing the fail count.
COPY <<'EOF' /usr/local/bin/ccg-versions
#!/usr/bin/env bash
set -uo pipefail
fail=0
v() {
  local label="$1" out rc; shift
  printf '== %s\n' "$label"
  out="$("$@" 2>&1)"; rc=$?
  printf '%s\n' "$out" | sed '/^\s*$/d' | head -n 4
  if [ "$rc" -ne 0 ] || [ -z "${out//[[:space:]]/}" ]; then
    echo "!! $label: FAILED (exit $rc)"; fail=$((fail + 1)); return 1
  fi
}
jarv() {   # jarv <label> <lib dir> <artifact>: version from io.joern.<artifact>-<ver>.jar
  local label="$1" ver
  printf '== %s\n' "$label"
  ver="$(ls "$2" 2>/dev/null | sed -n "s/^io\.joern\.$3-\(.*\)\.jar$/\1/p" | grep -v launcher)"
  if [ -z "$ver" ]; then echo "!! $label: FAILED (no io.joern.$3 jar in $2)"; fail=$((fail + 1)); return 1; fi
  echo "$ver"
}
jarv joern "${JOERN_HOME:-/opt/joern/joern-cli}/lib" joern-cli
for f in pysrc2cpg jssrc2cpg javasrc2cpg jimple2cpg kotlin2cpg csharpsrc2cpg gosrc2cpg c2cpg; do
  jarv "$f" "${JOERN_HOME:-/opt/joern/joern-cli}/frontends/$f/lib" "$f"
done
v java        java -version
v mvn         mvn -v
v gradle      bash -c 'set -o pipefail; gradle --version | grep -E "^(Gradle|Kotlin|Groovy|Launcher JVM)"'
v go          go version
v dotnet      dotnet --version
v kotlinc     kotlinc -version
v cmake       cmake --version
v bear        bear --version
v gcc         gcc --version
v g++         g++ --version
v make        make --version
v git         git --version
for d in /usr/local/lib/ccg-versions.d/*.sh; do
  [ -e "$d" ] || continue
  if ! bash -n "$d"; then echo "!! $d: FAILED (syntax)"; fail=$((fail + 1)); continue; fi
  # Each drop-in runs in its own subshell and reports its OWN failure count through an EXIT
  # trap, so a drop-in's `exit 0` (or `set -e` abort) cannot hide the count it accumulated.
  # shellcheck source=/dev/null
  ( fail=0; trap 'rc=$?; if [ "$fail" -gt 0 ]; then exit "$fail"; fi; exit "$rc"' EXIT; . "$d" )
  fail=$((fail + $?))
done
if [ "$fail" -ne 0 ]; then echo "ccg-versions: $fail check(s) FAILED"; exit 1; fi
echo "ccg-versions: all checks passed"
EOF
RUN mkdir -p /usr/local/lib/ccg-versions.d \
 && chmod 0755 /usr/local/bin/ccg-versions

# =================================================================================
# tail (S02) — last in the `toolchain` stage. Non-root: the base's `ubuntu` user is renamed rather than
# a new one added, so the container user is uid/gid 1000 — the same as this lab's host
# user, which lets the sandbox's rw output mount be written without a chown. `-G ''`
# drops the base user's supplementary groups (adm, sudo, …): ccg is in group ccg only.
# =================================================================================
RUN usermod -l ccg -d /home/ccg -m -G '' ubuntu && groupmod -n ccg ubuntu \
 && mkdir -p /work && chown ccg:ccg /work
USER ccg
WORKDIR /work
# Drop the base's /__cacert_entrypoint.sh (it rewrites the JVM truststore and exports
# JAVA_TOOL_OPTIONS when USE_SYSTEM_CA_CERTS is set), so default runs match S08's.
ENTRYPOINT []
CMD ["ccg-versions"]

# =================================================================================
# stage truth (S24; ML note §5.2, §9) — image ccg-toolchain-truth:<hash>. FROM toolchain
# so the sandbox image (`toolchain`) carries none of this: no Node, no indexers, no clang.
# apt work goes through `apt-snapshot-install` (toolchain is FROM apt-base). Tools live
# under /opt/truth; build-time caches are removed and never land in /home/ccg or /work.
# Truth runs happen with network, OUTSIDE the sandbox (ML §9); this image grants none.
# =================================================================================
FROM toolchain AS truth
ARG SCIP_DOTNET_VERSION=0.2.16
USER root

LABEL ccg-toolchain-truth=1 \
      org.opencontainers.image.title="ccg-toolchain-truth" \
      org.opencontainers.image.description="ccg-toolchain + SCIP / callgraph ground-truth indexers for C-1001 P0 frontend fitness"

# python3 + venv + pip (scip-python shells out to `pip list`; the corpus venv).
RUN apt-snapshot-install python3 python3-venv python3-pip

COPY --from=truth-fetch /opt/truth /opt/truth

# scip-dotnet as a dotnet tool under /opt/truth (HOME=/root for the build, then wiped: the
# NuGet cache must not be left anywhere a run inherits).
RUN HOME=/root DOTNET_CLI_HOME=/root dotnet tool install scip-dotnet --version "${SCIP_DOTNET_VERSION}" \
      --tool-path /opt/truth/scip-dotnet --add-source /opt/truth/nupkg \
 && rm -rf /root/.nuget /root/.dotnet /root/.local /opt/truth/nupkg \
 && find /tmp -mindepth 1 -maxdepth 1 -exec rm -rf {} + \
 && ln -s /opt/truth/npm/scip-python/node_modules/.bin/scip-python /opt/truth/bin/scip-python \
 && ln -s /opt/truth/npm/scip-typescript/node_modules/.bin/scip-typescript /opt/truth/bin/scip-typescript \
 && printf '#!/bin/sh\nexec java -jar /opt/truth/scip-java/scip-java.jar "$@"\n' > /opt/truth/bin/scip-java \
 && chmod 0755 /opt/truth/bin/scip-java

# kotlinc-scip <kotlinc args>: the embeddable Kotlin compiler with the scip-kotlinc plugin.
# Required env: SCIP_SOURCEROOT, SCIP_TARGETROOT. Shards land in $SCIP_TARGETROOT; aggregate
# them with `scip-java aggregate --targetroot=... --output=index.scip`.
COPY <<'EOT' /opt/truth/bin/kotlinc-scip
#!/usr/bin/env bash
set -euo pipefail
: "${SCIP_SOURCEROOT:?kotlinc-scip: set SCIP_SOURCEROOT}" "${SCIP_TARGETROOT:?kotlinc-scip: set SCIP_TARGETROOT}"
d=/opt/truth/kotlinc-embeddable
cp="$(printf '%s:' "$d"/*.jar)"; cp="${cp%:}"; stdlib="$(ls "$d"/kotlin-stdlib-*.jar)"
exec java -cp "$cp" org.jetbrains.kotlin.cli.jvm.K2JVMCompiler -no-stdlib -no-reflect -cp "$stdlib" \
  -Xplugin="$SCIP_KOTLINC_JAR" -P "plugin:scip-kotlinc:sourceroot=$SCIP_SOURCEROOT" -P "plugin:scip-kotlinc:targetroot=$SCIP_TARGETROOT" "$@"
EOT
RUN chmod 0755 /opt/truth/bin/kotlinc-scip

ENV PATH=/opt/truth/bin:/opt/truth/node/bin:/opt/truth/scip-dotnet:${PATH} \
    SCIP_JAVAC_JAR=/opt/truth/scip-java/scip-javac.jar \
    SCIP_KOTLINC_JAR=/opt/truth/scip-java/scip-kotlinc.jar \
    CCG_TRUTH_SMOKE_DIR=/opt/truth/smoke

# Smoke projects (smoke/<lang>/) and their runner; `ccg-truth-smoke` indexes each one and
# reads the index back with `scip print --json` (`callgraph` output for Go).
COPY smoke /opt/truth/smoke
RUN install -m 0755 /opt/truth/smoke/ccg-truth-smoke /usr/local/bin/ccg-truth-smoke \
 && rm /opt/truth/smoke/ccg-truth-smoke

# Version checks, run by ccg-versions after the S02 ones (each drop-in in its own subshell).
COPY <<'EOT' /usr/local/lib/ccg-versions.d/50-truth.sh
v node          node --version
v npm           npm --version
v scip-python   scip-python --version
v scip-typescript scip-typescript --version
v scip-java     bash -c 'scip-java --version; sha256sum /opt/truth/scip-java/scip-java.jar | cut -c1-16'
v scip-javac    bash -c 'sha256sum "$SCIP_JAVAC_JAR" | cut -c1-16'
v scip-kotlinc  bash -c 'sha256sum "$SCIP_KOTLINC_JAR" | cut -c1-16; ls /opt/truth/kotlinc-embeddable | grep compiler'
v scip-dotnet   scip-dotnet --version
v callgraph     bash -c 'go version -m "$(command -v callgraph)" | grep -E "^\s*(path|mod)\s"'
v scip-go       scip-go --version
v scip-clang    scip-clang --version
v scip          scip --version
v uv            uv --version
v python3       python3 --version
EOT
RUN chmod 0644 /usr/local/lib/ccg-versions.d/50-truth.sh

USER ccg
WORKDIR /work
ENTRYPOINT []
CMD ["ccg-versions"]

# =================================================================================
# stage indexer (P1-48) — image ccg-toolchain-indexer:<hash>. FROM toolchain, plus Node 24
# and the lockfile-pinned scip-python / scip-typescript (copied from `npm-fetch`, the same
# bytes the `truth` image carries), plus python3 (scip-python needs an interpreter for its
# environment discovery and `pip list`; a pre-provisioned venv's bin/python resolves to it).
# Nothing else: no truth tools, no smoke projects, no clang. The `toolchain` image the sandbox
# runs stays free of Node and indexers, so AC-8's "no dependency download" surface widens
# only for builds that run an indexer (P1-15 sets CCG_TOOLCHAIN_IMAGE to this image per call).
# Tools live under /opt/indexer; no npm/pip cache is left under /home/ccg or /work.
# =================================================================================
FROM toolchain AS indexer
USER root

LABEL ccg-toolchain-indexer=1 \
      org.opencontainers.image.title="ccg-toolchain-indexer" \
      org.opencontainers.image.description="ccg-toolchain + Node 24 + scip-python / scip-typescript for C-1001 P1 offline indexer runs"

RUN apt-snapshot-install python3 python3-venv python3-pip

COPY --from=npm-fetch /opt/truth/node /opt/indexer/node
COPY --from=npm-fetch /opt/truth/npm  /opt/indexer/npm
RUN mkdir -p /opt/indexer/bin \
 && ln -s /opt/indexer/npm/scip-python/node_modules/.bin/scip-python /opt/indexer/bin/scip-python \
 && ln -s /opt/indexer/npm/scip-typescript/node_modules/.bin/scip-typescript /opt/indexer/bin/scip-typescript \
 && rm -rf /root/.npm /root/.cache \
 && find /tmp -mindepth 1 -maxdepth 1 -exec rm -rf {} +

ENV PATH=/opt/indexer/bin:/opt/indexer/node/bin:${PATH}

COPY <<'EOT' /usr/local/lib/ccg-versions.d/50-indexer.sh
v node            node --version
v npm             npm --version
v scip-python     scip-python --version
v scip-typescript scip-typescript --version
v python3         python3 --version
EOT
RUN chmod 0644 /usr/local/lib/ccg-versions.d/50-indexer.sh

USER ccg
WORKDIR /work
ENTRYPOINT []
CMD ["ccg-versions"]
