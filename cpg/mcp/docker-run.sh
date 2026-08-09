#!/usr/bin/env bash
set -euo pipefail
# docker-run.sh — launch the `cpg` MCP server in a container, speaking MCP over stdio.
#
# This is the path the repo-root .mcp.json names. run.sh is the Docker-less variant
# (host venv) and stays the fallback plus the fast test loop. Like run.sh it resolves
# everything from its own location, so the harness's working directory does not matter.
#
# TWO INVARIANTS, of equal weight:
#  1. NOTHING here may WRITE to stdout — the stdio transport owns it. Every diagnostic
#     goes to stderr explicitly. (docker itself is stdout-clean: its progress and status
#     go to stderr; verified 2026-07-25.)
#  2. NOTHING before the final `exec` may READ from stdin — from the moment this process
#     is spawned, stdin IS the MCP pipe and Claude Code has already written `initialize`
#     into it. A byte consumed here is a byte the server never sees, and the failure is
#     a hung or malformed handshake with no useful diagnostic. `docker build` with a path
#     context does not read stdin today (measured), so this is a latent hazard, not a
#     live bug — which is exactly why it is guarded structurally rather than trusted.
#     Every helper that FORKS carries `</dev/null`, and build.sh closes stdin itself too.
#     The one call without the redirect is `cpg_mcp_image_tag`, a sourced shell function
#     that forks nothing and reads no stdin — named here so the invariant reads as
#     absolute rather than as an unexplained exception.
#
# Env (all optional): FALKORDB_HOST/_PORT, CPG_MCP_MAX_ROWS/_MAX_CELL/_MAX_CHARS/
# _TIMEOUT_MS — forwarded into the container if set, else the image defaults apply.
#   CPG_MCP_IMAGE=<ref>       run this exact image, bypassing the hash gate entirely
#   CPG_MCP_NO_AUTOBUILD=1    never build here; fail with "run cpg/mcp/build.sh"
# See also MCP_TIMEOUT (Claude Code, default 30000 ms) if a first cold build needs more
# startup headroom than the default budget allows. That is the STARTUP budget and is
# distinct from .mcp.json's "timeout": 60000, which is the per-tool-call wall.
#
# Design: docs/plans/cpg-mcp-containerization.md §3.1, §3.3, §3.10, §4.5.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- preflight, HERE and not only in build.sh: it must also cover the no-autobuild
# --- path, where build.sh never runs at all.
if ! command -v docker >/dev/null 2>&1; then
  echo "cpg/mcp/docker-run.sh: docker not on PATH. Fall back to cpg/mcp/run.sh (host venv)." >&2
  exit 1
fi
if ! docker info >/dev/null 2>&1 </dev/null; then
  echo "cpg/mcp/docker-run.sh: Docker daemon not reachable (check 'docker context ls'). Fall back to cpg/mcp/run.sh." >&2
  exit 1
fi

# --- resolve the image: content hash unless the caller pinned one ------------------
PINNED=0
if [ -n "${CPG_MCP_IMAGE:-}" ]; then
  IMAGE="$CPG_MCP_IMAGE"
  PINNED=1                              # "the caller knows what they are running"
else
  # shellcheck source=image-tag.sh
  source "$HERE/image-tag.sh"
  if ! cpg_mcp_image_tag; then          # sets CPG_MCP_TAG; writes nothing to stdout
    echo "cpg/mcp/docker-run.sh: cannot compute the image tag (see above). Fall back to cpg/mcp/run.sh." >&2
    exit 1
  fi
  IMAGE="${CPG_MCP_IMAGE_REPO:-cpg-mcp}:$CPG_MCP_TAG"
fi

# --- the staleness gate. The tag IS the content of the build inputs, so an existing
# --- image is a proof, not a heuristic; a miss means something changed (or nothing was
# --- ever built). ~0.05 s, purely local — no registry contact in any circumstance.
if ! docker image inspect "$IMAGE" >/dev/null 2>&1 </dev/null; then
  if [ "$PINNED" -eq 1 ]; then
    # CPG_MCP_IMAGE bypasses the gate, so autobuilding here would build the CONTENT-HASH
    # tag — never the pinned one — and then still fail on `docker run`, with docker's
    # bare "No such image" as the only diagnostic. Say so instead.
    echo "cpg/mcp/docker-run.sh: CPG_MCP_IMAGE=$IMAGE is not present locally." >&2
    echo "  CPG_MCP_IMAGE bypasses the content-hash gate, so nothing will be built for it." >&2
    echo "  Either build/load that image yourself, or unset CPG_MCP_IMAGE to use the gate" >&2
    echo "  (cpg/mcp/build.sh), or fall back to the host venv: cpg/mcp/run.sh." >&2
    exit 1
  fi
  if [ "${CPG_MCP_NO_AUTOBUILD:-0}" = "1" ]; then
    echo "cpg/mcp/docker-run.sh: image $IMAGE not built and CPG_MCP_NO_AUTOBUILD=1. Run: cpg/mcp/build.sh" >&2
    exit 1
  fi
  echo "cpg/mcp/docker-run.sh: $IMAGE not present — building (first run after a change may take ~15 s)." >&2
  # A failed build is the one launch failure that would otherwise exit on raw BuildKit
  # output with no hint that a fallback exists — and on a fresh clone or an offline
  # machine it is the MOST LIKELY failure. Every other path here has a curated line.
  #
  # CPG_MCP_NO_PULL=1: this autobuild must stay local-only. Without it, build.sh
  # unconditionally `docker pull`s the base image with no timeout on EVERY miss —
  # including a miss triggered by a test-only edit that cannot change the runtime image
  # (the hash covers tests/, which the runtime stage never COPYs) — inside Claude
  # Code's 30 s MCP startup budget. In steady state the base is already in the local
  # image store (an interactive `cpg/mcp/build.sh` put it there), so this changes
  # nothing when it matters and removes the unbounded pull when it doesn't. The
  # explicit `cpg/mcp/build.sh`, run by hand without this override, stays the thing
  # that refreshes the base image. See docs/reviews/cpg-mcp-containerization.md M-8.
  if ! CPG_MCP_NO_PULL=1 "$HERE/build.sh" --runtime-only >&2 </dev/null; then
    echo "cpg/mcp/docker-run.sh: build of $IMAGE FAILED (BuildKit output above)." >&2
    echo "  Offline? A build needs the network unless python:3.12-slim is already in the local image store." >&2
    echo "  Fall back to the host venv: cpg/mcp/run.sh (see cpg/mcp/README.md), or retry with more" >&2
    echo "  startup budget: MCP_TIMEOUT=60000 claude" >&2
    echo "  ...or run cpg/mcp/build.sh once with a network connection to refresh the local image store." >&2
    exit 1
  fi
fi

# --- env forwarding: pass a variable ONLY if it is actually set here ---------------
# Do NOT use the bare `-e VAR` form unconditionally. Measured on Docker 29.6.1: with
# VAR unset in the caller's environment, `-e VAR` does not "fall through to the image
# default" — it UNSETS the variable in the container, clobbering the Dockerfile's
# `ENV FALKORDB_HOST=host.docker.internal` and leaving server.py to fall back to its own
# 127.0.0.1 default, i.e. the container talking to itself. That is precisely the failure
# the image default exists to prevent, and it bites anyone who runs this script by hand
# without exporting FALKORDB_HOST.
#   no -e flag              -> host.docker.internal   (image ENV)
#   -e FALKORDB_HOST, unset -> None                   (image ENV DELETED)
#   -e FALKORDB_HOST, set   -> the caller's value
ENV_ARGS=()
for _v in FALKORDB_HOST FALKORDB_PORT \
          CPG_MCP_MAX_ROWS CPG_MCP_MAX_CELL CPG_MCP_MAX_CHARS CPG_MCP_TIMEOUT_MS; do
  if [ -n "${!_v+set}" ]; then           # set, even if empty — an explicit empty value
    ENV_ARGS+=(-e "$_v=${!_v}")          # is still the caller's choice to make
  fi
done

# --add-host is what makes host.docker.internal resolve on a Linux engine.
# --init: PID-1 python IGNORES SIGTERM; tini forwards it. Measured, not defensive.
# --label: the only handle for finding a leaked container.
# --pull=never: this image is local-only, so a missing tag must say "No such image"
#   rather than docker's misleading "pull access denied".
# No --name: a fixed name would collide the moment two sessions run concurrently,
#   turning benign duplication into a hard failure at session start.
# --read-only --tmpfs /tmp: adopted after probing, not assumed. The server never writes
#   to disk (PYTHONDONTWRITEBYTECODE=1), and every tool-body path was exercised under
#   these flags before adopting them — initialize, tools/list, a real query, EXPLAIN,
#   unknown-graph, PROFILE-refusal and invalid-Cypher — with no filesystem error. They
#   are the first thing to drop if the server ever fails at session start with a
#   read-only or permission error: remove these two flags and re-run the V-6/V-7 probes
#   in cpg/mcp/README.md.
exec docker run -i --rm --init \
  --label cpg-mcp=1 \
  --pull=never \
  --read-only --tmpfs /tmp \
  --add-host=host.docker.internal:host-gateway \
  "${ENV_ARGS[@]+"${ENV_ARGS[@]}"}" \
  "$IMAGE"
