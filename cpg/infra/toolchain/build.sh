#!/usr/bin/env bash
set -euo pipefail
# build.sh — build the `ccg-toolchain` image at its content-hash tag (C-1001 P0, S02).
#
#   cpg/infra/toolchain/build.sh              # build ccg-toolchain:<hash> (no-op if it exists)
#   cpg/infra/toolchain/build.sh --print-tag  # print ccg-toolchain:<hash> on stdout, build nothing
#   cpg/infra/toolchain/build.sh --no-cache   # rebuild an existing tag from a cold layer cache
#   cpg/infra/toolchain/build.sh --help
#
# The tag is the first 12 hex chars of sha256(Dockerfile). The Dockerfile is the ONLY
# build input: it is piped to `docker build` on stdin with no build context, so nothing
# from the host can reach the image without changing this hash, and every upstream
# input inside it is pinned (base digest, apt snapshot date, versioned + checksummed
# downloads). Same convention as cypher-mcp/build.sh, narrowed to one input file.
#
# Idempotent: an existing hash tag IS a built image for these exact inputs, so the
# script exits without touching the network. It never removes or retags any other image.
# Diagnostics go to stderr; stdout carries only the tag (with --print-tag, or on success),
# so callers can do  IMAGE="$(cpg/infra/toolchain/build.sh)".
#
# Env: CCG_TOOLCHAIN_REPO (default ccg-toolchain).
# Network is required for a build (§7: "S02 and S24 need Docker, plus network").

usage() {
  cat >&2 <<'EOF'
Usage: build.sh [--print-tag] [--no-cache] [-h|--help]

  (no args)     build ccg-toolchain:<hash>; exit early if that tag already exists.
                Prints the tag on stdout, the image size on stderr.
  --print-tag   print ccg-toolchain:<hash> and exit (no docker build, no network).
  --no-cache    rebuild even if the tag exists, with a cold layer cache.
  -h, --help    this help.

Env: CCG_TOOLCHAIN_REPO (default ccg-toolchain)
EOF
}

PRINT_ONLY=0
NO_CACHE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --print-tag) PRINT_ONLY=1 ;;
    --no-cache)  NO_CACHE=1 ;;
    -h|--help)   usage; exit 0 ;;
    *)           echo "build.sh: unknown option '$1'" >&2; usage; exit 2 ;;
  esac
  shift
done

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKERFILE="$HERE/Dockerfile"
[ -f "$DOCKERFILE" ] || { echo "build.sh: missing $DOCKERFILE" >&2; exit 1; }

REPO="${CCG_TOOLCHAIN_REPO:-ccg-toolchain}"
HASH="$(sha256sum < "$DOCKERFILE")"
TAG="$REPO:${HASH:0:12}"

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

echo "build.sh: building $TAG (downloads ~3.5 GB on a cold cache)…" >&2
# `-` = Dockerfile from stdin, no build context. --progress=plain keeps a readable log
# when stderr is redirected to a file.
docker build "${CACHE_FLAG[@]+"${CACHE_FLAG[@]}"}" --progress=plain -t "$TAG" - < "$DOCKERFILE" >&2

report_size
echo "build.sh: verify with  docker run --rm --network none $TAG ccg-versions" >&2
echo "$TAG"
