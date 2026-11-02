#!/usr/bin/env bash
set -euo pipefail
# build.sh — build the `ccg-toolchain` / `ccg-toolchain-truth` images at their content-hash
# tag (C-1001 P0, S02 + S24).
#
#   cpg/infra/toolchain/build.sh                 # build ccg-toolchain:<hash> (no-op if it exists)
#   cpg/infra/toolchain/build.sh --truth         # build ccg-toolchain-truth:<hash> (and the toolchain it is FROM)
#   cpg/infra/toolchain/build.sh --print-tag     # print ccg-toolchain:<hash> on stdout, build nothing
#   cpg/infra/toolchain/build.sh --truth --print-tag   # print ccg-toolchain-truth:<hash>
#   cpg/infra/toolchain/build.sh --no-cache      # rebuild an existing tag from a cold layer cache
#   cpg/infra/toolchain/build.sh --help
#
# The tag is the first 12 hex chars of sha256 of the BUILD CONTEXT: a deterministic tar (sorted
# names, mtime 0, uid/gid 0, normalized modes) of exactly `Dockerfile`, `smoke/` and
# `truth-npm/` — the only files `docker build` can see, piped on stdin with no host directory
# behind it. So nothing from the host reaches an image without changing the hash, and every
# upstream input is pinned inside (base digest, apt snapshot date, versioned + checksummed
# downloads, npm lockfiles). Both images share one hash; an edit to a truth tool or a smoke
# project therefore also moves the `toolchain` tag (its layers are unchanged and cached).
# Same convention as cypher-mcp/build.sh.
#
# Idempotent: an existing hash tag IS a built image for these exact inputs, so the script
# exits without touching the network. It never removes any image. It tags only this hash's
# `toolchain` and `truth` images; `--truth --no-cache` rebuilds and re-tags BOTH of them.
# Diagnostics go to stderr; stdout carries only the tag (with --print-tag, or on success),
# so callers can do  IMAGE="$(cpg/infra/toolchain/build.sh)".
#
# Env: CCG_TOOLCHAIN_REPO (default ccg-toolchain); the truth repo is "${REPO}-truth".
# Network is required for a build (§7: "S02 and S24 need Docker, plus network").

usage() {
  cat >&2 <<'EOF'
Usage: build.sh [--truth] [--print-tag] [--no-cache] [-h|--help]

  (no args)     build ccg-toolchain:<hash>; exit early if that tag already exists.
                Prints the tag on stdout, the image size on stderr.
  --truth       target ccg-toolchain-truth:<hash> (FROM the toolchain stage) instead.
  --print-tag   print the tag and exit (no docker build, no network).
  --no-cache    rebuild even if the tag exists, with a cold layer cache.
  -h, --help    this help.

Env: CCG_TOOLCHAIN_REPO (default ccg-toolchain)
EOF
}

PRINT_ONLY=0
TRUTH=0
NO_CACHE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --print-tag) PRINT_ONLY=1 ;;
    --truth)     TRUTH=1 ;;
    --no-cache)  NO_CACHE=1 ;;
    -h|--help)   usage; exit 0 ;;
    *)           echo "build.sh: unknown option '$1'" >&2; usage; exit 2 ;;
  esac
  shift
done

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$HERE/Dockerfile" ] || { echo "build.sh: missing $HERE/Dockerfile" >&2; exit 1; }
[ -d "$HERE/smoke" ] && [ -d "$HERE/truth-npm" ] || { echo "build.sh: missing smoke/ or truth-npm/" >&2; exit 1; }

# The build context, as a deterministic tar on stdout (GNU tar). Reproducible across
# checkouts: file contents and the executable bit are the only things that vary.
context() {
  tar -C "$HERE" --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner \
      --mode='u=rwX,go=rX' --format=gnu \
      -cf - Dockerfile smoke truth-npm
}

REPO="${CCG_TOOLCHAIN_REPO:-ccg-toolchain}"
HASH="$(context | sha256sum)"
H12="${HASH:0:12}"
TOOLCHAIN_TAG="$REPO:$H12"
if [ "$TRUTH" -eq 1 ]; then TAG="$REPO-truth:$H12"; TARGET=truth; else TAG="$TOOLCHAIN_TAG"; TARGET=toolchain; fi

if [ "$PRINT_ONLY" -eq 1 ]; then
  echo "$TAG"
  exit 0
fi

command -v docker >/dev/null 2>&1 || { echo "build.sh: docker not on PATH." >&2; exit 1; }
docker info >/dev/null 2>&1 || { echo "build.sh: Docker daemon not reachable." >&2; exit 1; }

report_size() {
  local bytes
  bytes="$(docker image inspect -f '{{.Size}}' "$TAG")"
  echo "build.sh: $TAG — $(( bytes / 1024 / 1024 )) MiB ($bytes bytes)" >&2
}

if [ "$NO_CACHE" -eq 0 ] && docker image inspect "$TAG" >/dev/null 2>&1; then
  echo "build.sh: $TAG already built — nothing to do (--no-cache forces a rebuild)." >&2
  report_size
  echo "$TAG"
  exit 0
fi

CACHE_FLAG=()
[ "$NO_CACHE" -eq 1 ] && CACHE_FLAG=(--no-cache)

echo "build.sh: building $TAG, target $TARGET (downloads ~3.5 GB for the toolchain, more for truth, on a cold cache)…" >&2
# `-` = the tar context on stdin. A truth build also tags the toolchain stage it is FROM
# (a cache hit once built), so the pair always carries one hash. --progress=plain keeps a
# readable log when stderr is redirected to a file.
if [ "$TARGET" = truth ]; then
  context | docker build "${CACHE_FLAG[@]+"${CACHE_FLAG[@]}"}" --progress=plain --target toolchain -t "$TOOLCHAIN_TAG" - >&2
fi
context | docker build "${CACHE_FLAG[@]+"${CACHE_FLAG[@]}"}" --progress=plain --target "$TARGET" -t "$TAG" - >&2

report_size
echo "build.sh: verify with  docker run --rm --network none $TAG ccg-versions" >&2
[ "$TARGET" = truth ] && echo "build.sh: smoke tests:  docker run --rm --network none $TAG ccg-truth-smoke" >&2
echo "$TAG"
