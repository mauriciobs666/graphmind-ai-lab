# cpg/infra/toolchain/Dockerfile — the `ccg-toolchain` image (C-1001 P0, step S02).
#
# Joern v4.0.579 plus every frontend's toolchain, for P0's sandboxed `--network none`
# builds and the frontend-fitness runs (docs/plans/connected-code-graph-p0.md §4.1, §7).
# Built by ./build.sh, which tags it ccg-toolchain:<sha256(Dockerfile)[:12]>. Results
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
# The build context is EMPTY on purpose (build.sh pipes this file on stdin): nothing is
# COPYed from the host, so this file is the only build input and its hash is the tag.
#
# LAYOUT (S24 extends it without touching the S02 layers):
#   stage `apt-base`     base + snapshot-only apt sources + apt-snapshot-install
#   stage `fetch`        S02 downloads (verified, unpacked)
#   [S24-STAGE]          S24 adds its own download stage(s) here, e.g. `truth-fetch`
#   stage `toolchain`    S02 runtime: apt, COPY --from=fetch, ENV, ccg-versions
#   [S24-LAYER]          S24 adds its ground-truth layer here (COPY --from=truth-fetch, RUN,
#                        plus a /usr/local/lib/ccg-versions.d/<name>.sh drop-in)
#   tail                 non-root user + workdir — always last, S24 does not edit it
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
# [S24-STAGE] — S24 inserts its ground-truth download stage(s) here
# (e.g. `FROM apt-base AS truth-fetch`), reusing the fetch-verify pattern above.
# =================================================================================

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
  # shellcheck source=/dev/null
  . "$d" || true      # a drop-in's own v/jarv calls count failures; don't double-count
done
if [ "$fail" -ne 0 ]; then echo "ccg-versions: $fail check(s) FAILED"; exit 1; fi
echo "ccg-versions: all checks passed"
EOF
RUN mkdir -p /usr/local/lib/ccg-versions.d \
 && chmod 0755 /usr/local/bin/ccg-versions

# =================================================================================
# [S24-LAYER] — S24 inserts its ground-truth layer here (COPY --from=truth-fetch …,
# RUN …). Install truth tools under /opt; keep their caches out of /home/ccg and /work.
# apt work goes through `apt-snapshot-install` (this stage is FROM apt-base). Register
# the tools' version checks as /usr/local/lib/ccg-versions.d/<name>.sh (see above).
# =================================================================================

# =================================================================================
# tail (S02) — always last. Non-root: the base's `ubuntu` user is renamed rather than
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
