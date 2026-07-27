#!/usr/bin/env bash
set -euo pipefail
exec </dev/null                 # This script reads NOTHING from stdin, ever — when
                                # docker-run.sh calls it, stdin is the MCP pipe.
# build.sh — build the `cpg` MCP server images at the current content-hash tag.
#
# Idempotent, and specifically: if the target tags ALREADY EXIST it does nothing at
# all — no docker build, and no docker pull either. --no-cache is the way to rebuild
# an existing tag.
#
#   cpg/mcp/build.sh                  # runtime + test at <hash>, plus :dev/:test aliases
#   cpg/mcp/build.sh --runtime-only   # just cpg-mcp:<hash>  — what docker-run.sh calls
#   cpg/mcp/build.sh --no-cache       # force a clean rebuild of an existing tag
#   cpg/mcp/build.sh --verify-inputs  # check image-tag.sh covers every Dockerfile COPY
#   cpg/mcp/build.sh --help
#
# ALL output goes to stderr — unconditionally, including --help — and stdin is closed
# above: docker-run.sh calls this on the MCP stdio path, where a stray byte on stdout
# corrupts the protocol and a byte READ from stdin is a byte the server never sees.
# The rule is absolute rather than conditional on purpose: an absolute one is auditable
# by reading the file, a conditional one is not.
#
# Env overrides: CPG_MCP_IMAGE_REPO (default cpg-mcp), CPG_MCP_NO_PULL=1
#
# Design: docs/plans/cpg-mcp-containerization.md §3.3, §3.6, §4.4.

BASE_IMAGE="python:3.12-slim"

usage() {
  cat >&2 <<'EOF'
Usage: build.sh [--runtime-only] [--no-cache] [--verify-inputs] [-h|--help]

  (no args)        build cpg-mcp:<hash> and cpg-mcp:test-<hash>, plus the moving
                   :dev / :test aliases. Exits early, doing nothing, if both hash
                   tags already exist.
  --runtime-only   build only cpg-mcp:<hash> (+ :dev). What docker-run.sh calls on a
                   hash miss.
  --no-cache       rebuild even if the tag exists, with a cold Docker layer cache.
                   Also the way to pick up a refreshed base image:
                     docker pull python:3.12-slim && cpg/mcp/build.sh --no-cache
  --verify-inputs  check image-tag.sh covers every Dockerfile COPY operand, then exit.
                   Run implicitly before every build.
  -h, --help       show this help

All output goes to stderr. Env: CPG_MCP_IMAGE_REPO (default cpg-mcp), CPG_MCP_NO_PULL=1
EOF
}

RUNTIME_ONLY=0
NO_CACHE=0
VERIFY_ONLY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --runtime-only)  RUNTIME_ONLY=1 ;;
    --no-cache)      NO_CACHE=1 ;;
    --verify-inputs) VERIFY_ONLY=1 ;;
    -h|--help)       usage; exit 0 ;;
    *)               echo "build.sh: unknown option '$1'" >&2; usage; exit 2 ;;
  esac
  shift
done

# Resolve everything from this script's own location, so the working directory never
# matters (same idiom as run.sh / setup.sh).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=image-tag.sh
source "$HERE/image-tag.sh"

REPO="${CPG_MCP_IMAGE_REPO:-cpg-mcp}"

if ! cpg_mcp_image_tag; then
  echo "build.sh: cannot determine the content-hash tag (see above). Nothing built." >&2
  exit 1
fi
TAG="$CPG_MCP_TAG"

# --- is the operand covered by image-tag.sh? -------------------------------------
# A FILE operand must appear in cpg_mcp_input_files. A DIRECTORY operand must appear
# in cpg_mcp_input_dirs — i.e. the enumeration WALKS it, with .dockerignore's
# exclusions — which is what makes "add a fixture under tests/" change the hash
# without anyone editing a list.
#
# Called with the set on stdin via PROCESS SUBSTITUTION, never a pipeline: this
# function returns as soon as it matches, which under `set -o pipefail` would make the
# producer die of EPIPE and turn a successful lookup into a failed pipeline. (Found the
# hard way — the first run reported every covered file as a gap.)
_in_nul_set() {                  # $1 = needle; NUL-separated set on stdin
  local needle="$1" item
  while IFS= read -r -d '' item; do
    [ "$item" = "$needle" ] && return 0
  done
  return 1
}

verify_inputs() {
  local gaps=0 src

  # The Dockerfile and .dockerignore are not COPYed but both change what is built,
  # so the invariant covers them explicitly.
  local meta
  for meta in Dockerfile .dockerignore; do
    if ! _in_nul_set "$meta" < <(cpg_mcp_input_files); then
      echo "build.sh --verify-inputs: '$meta' is not in image-tag.sh's file list, but it changes what gets built." >&2
      gaps=$((gaps + 1))
    fi
  done

  while IFS= read -r src; do
    [ -n "$src" ] || continue
    if [ -d "$HERE/$src" ]; then
      if ! _in_nul_set "$src" < <(cpg_mcp_input_dirs); then
        echo "build.sh --verify-inputs: Dockerfile COPYs the DIRECTORY '$src', which image-tag.sh does not walk." >&2
        echo "  Fix: add '$src' to cpg_mcp_input_dirs in cpg/mcp/image-tag.sh (do NOT list its files individually —" >&2
        echo "  a walk is what makes a new file of any extension under it change the hash)." >&2
        gaps=$((gaps + 1))
      fi
    elif [ -f "$HERE/$src" ]; then
      if ! _in_nul_set "$src" < <(cpg_mcp_input_files); then
        echo "build.sh --verify-inputs: Dockerfile COPYs the FILE '$src', which is not in image-tag.sh's input list." >&2
        echo "  Fix: add '$src' to the fixed file list in cpg_mcp_input_files (cpg/mcp/image-tag.sh)." >&2
        gaps=$((gaps + 1))
      fi
    else
      echo "build.sh --verify-inputs: Dockerfile COPYs '$src', which does not exist in the build context." >&2
      gaps=$((gaps + 1))
    fi
  # The Dockerfile parse. Continuations are JOINED before any rule looks at a COPY:
  # a line-oriented parse is blind to `COPY a \` <newline> `    b /app/` and answers
  # "--verify-inputs OK" for it — a FALSE PASS on the only check that enforces
  # "the hash covers every COPY", which is the one failure direction that costs
  # correctness (the file lands in the image, the tag does not move, the launch path
  # serves an image without the change). Regression: tests/test_build_inputs.py.
  done < <(awk '
      # A comment line is never a COPY, and Docker also allows one INSIDE a
      # continuation — dropping it before accumulation handles both cases.
      /^[[:space:]]*#/                { next }
      { line = line $0 }
      /\\[[:space:]]*$/               { sub(/\\[[:space:]]*$/, " ", line); next }
      { $0 = line; line = "" }        # re-splits fields: the rules below see the join
      toupper($1) == "COPY" {
        # `COPY --from=<stage>` reads from another BUILD STAGE, not from the build
        # context, so its sources are not hash inputs at all. Without this, a perfectly
        # legitimate multi-stage COPY is rejected with "does not exist in the build
        # context", which sends the reader hunting for a file that was never missing.
        for (i = 2; i <= NF; i++) if ($i ~ /^--from=/) next
        n = 0; delete f
        for (i = 2; i <= NF; i++) if ($i !~ /^--/) { n++; f[n] = $i }
        for (i = 1; i < n; i++) print f[i]      # every operand but the last (the dest)
      }' "$HERE/Dockerfile")

  if [ "$gaps" -ne 0 ]; then
    echo "build.sh --verify-inputs: $gaps gap(s). The content-hash gate would not notice a change to" >&2
    echo "  the path(s) above, silently serving a stale image — the exact failure the hash exists to" >&2
    echo "  prevent. Refusing to build." >&2
    return 1
  fi
  echo "build.sh: --verify-inputs OK — every Dockerfile COPY operand is covered by image-tag.sh." >&2
}

if [ "$VERIFY_ONLY" -eq 1 ]; then
  verify_inputs
  exit 0
fi

# --- preflight -------------------------------------------------------------------
if ! command -v docker >/dev/null 2>&1; then
  echo "build.sh: docker not on PATH. The host-venv path still works: cpg/mcp/setup.sh then cpg/mcp/run.sh." >&2
  exit 1
fi
if ! docker info >/dev/null 2>&1; then
  echo "build.sh: Docker daemon not reachable (check 'docker context ls')." >&2
  echo "  The host-venv path still works: cpg/mcp/setup.sh then cpg/mcp/run.sh." >&2
  exit 1
fi

RUNTIME_TAG="$REPO:$TAG"
TEST_TAG="$REPO:test-$TAG"

# --- the early exit: an existing hash tag IS a built image for these exact inputs --
# This is what keeps a per-invocation registry dependency out of build.sh too, not
# only out of the launch path: a warm build.sh is a purely local image inspect.
have_image() { docker image inspect "$1" >/dev/null 2>&1; }

if [ "$NO_CACHE" -eq 0 ]; then
  if have_image "$RUNTIME_TAG" && { [ "$RUNTIME_ONLY" -eq 1 ] || have_image "$TEST_TAG"; }; then
    echo "build.sh: $RUNTIME_TAG already built$( [ "$RUNTIME_ONLY" -eq 1 ] || printf ' (and %s)' "$TEST_TAG" ) — nothing to build." >&2
    # Still re-point the moving aliases. Skipping the build must not leave :dev/:test
    # pointing at an OLDER hash — which is exactly what happens after moving between
    # two input states, and would make an ad-hoc `docker run cpg-mcp:dev` silently run
    # stale code. `docker tag` is an instant local metadata op: no network, no build,
    # so it does not reintroduce what the early exit exists to avoid. The launch path
    # never names an alias (§3.6), so this is about humans, not about the gate.
    docker tag "$RUNTIME_TAG" "$REPO:dev" >&2
    [ "$RUNTIME_ONLY" -eq 1 ] || docker tag "$TEST_TAG" "$REPO:test" >&2
    echo "  Aliases re-pointed: $REPO:dev -> $TAG$( [ "$RUNTIME_ONLY" -eq 1 ] || printf ', %s:test -> test-%s' "$REPO" "$TAG" )" >&2
    echo "  Force a rebuild with: cpg/mcp/build.sh --no-cache" >&2
    exit 0
  fi
fi

verify_inputs

# --- pull the base into the IMAGE STORE (not just the build cache) ----------------
# Reached only when something will actually be built. A BuildKit build populates the
# build cache, NOT the image store, and resolves `FROM` metadata over the network
# unless the base is in the store — measured at 0.5 s per build, essentially the whole
# cost of a warm build. Pulling it here is what makes a later rebuild offline-safe.
if [ "${CPG_MCP_NO_PULL:-0}" != "1" ]; then
  if ! docker pull -q "$BASE_IMAGE" >&2; then
    echo "build.sh: WARNING — could not pull $BASE_IMAGE (offline?). Continuing:" >&2
    echo "  the build may still succeed from the local image store or build cache." >&2
  fi
fi

CACHE_FLAG=()
[ "$NO_CACHE" -eq 1 ] && CACHE_FLAG=(--no-cache)

echo "build.sh: building $RUNTIME_TAG (target runtime)…" >&2
docker build "${CACHE_FLAG[@]+"${CACHE_FLAG[@]}"}" --target runtime \
  -t "$RUNTIME_TAG" -t "$REPO:dev" "$HERE" >&2

if [ "$RUNTIME_ONLY" -eq 0 ]; then
  echo "build.sh: building $TEST_TAG (target test)…" >&2
  docker build "${CACHE_FLAG[@]+"${CACHE_FLAG[@]}"}" --target test \
    -t "$TEST_TAG" -t "$REPO:test" "$HERE" >&2
fi

# --- report ----------------------------------------------------------------------
{
  echo
  echo "Built:"
  echo "  $RUNTIME_TAG   (alias $REPO:dev)      — the launch image; docker-run.sh names the hash tag only"
  if [ "$RUNTIME_ONLY" -eq 0 ]; then
    echo "  $TEST_TAG  (alias $REPO:test)     — the in-container test gate"
  else
    echo "  (test image not built — --runtime-only. Run cpg/mcp/build.sh for the test gate.)"
  fi
  echo
  echo "Next:"
  if [ "$RUNTIME_ONLY" -eq 0 ]; then
    echo "  docker run --rm $TEST_TAG python -m pytest tests -q                       # offline gate"
    echo "  docker run --rm --add-host=host.docker.internal:host-gateway $TEST_TAG \\"
    echo "    python -m pytest tests -q -m live                                          # live gate (needs FalkorDB up)"
  fi
  echo "  docker image ls --filter label=cpg-mcp=1                                     # what has accumulated"
} >&2
