# image-tag.sh — sourced by build.sh and docker-run.sh. Defines the content-hash image
# tag. NOT executable, prints nothing on stdout, reads nothing from stdin.
#
# cpg_mcp_input_dirs   -> directory operands the enumeration WALKS (NUL-separated)
# cpg_mcp_input_files  -> the exact set of build inputs, NUL-separated, relative paths
# cpg_mcp_image_tag    -> sets CPG_MCP_TAG=<hash12>  (assigns a variable; no output)
#
# The hash covers file CONTENTS and RELATIVE paths only — never an absolute path, so
# the value is identical on every machine and no home path can reach a tracked file
# (see the plan's "Path convention"). `sha256sum < "$f"` is deliberate: it digests the
# bytes without the filename, and the relative path is contributed explicitly.
#
# INVARIANT: this must cover every path the Dockerfile COPYs, plus the Dockerfile and
# .dockerignore themselves. A FILE operand is covered by cpg_mcp_input_files; a
# DIRECTORY operand is covered by cpg_mcp_input_dirs, which is walked with the same
# exclusions .dockerignore applies. `build.sh --verify-inputs` checks both rules.
#
# What the hash does NOT cover, deliberately: the base image (python:3.12-slim is a
# moving tag) and pip's dependency resolution (requirements.txt pins ranges). So the
# tag is immutable with respect to the TRACKED build inputs, not a full image identity,
# and nothing here ever refreshes an existing image's base — that is the documented
# manual step `docker pull python:3.12-slim && cpg/mcp/build.sh --no-cache`.
# See docs/plans/cpg-mcp-containerization.md §3.6, §4.3.

CPG_MCP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Directory operands. Walked, not globbed: that is what makes a NEW FILE OF ANY
# EXTENSION under tests/ change the hash automatically, instead of only *.py doing so.
cpg_mcp_input_dirs() {
  printf '%s\0' tests
}

# The fixed file operands, then every file under each walked directory that
# .dockerignore would not exclude. `LC_ALL=C sort` matters — a locale-dependent sort
# would make the hash machine-dependent.
cpg_mcp_input_files() {
  local f d
  for f in Dockerfile .dockerignore requirements.txt requirements-dev.txt server.py pytest.ini; do
    printf '%s\0' "$f"
  done
  while IFS= read -r -d '' d; do
    [ -d "$CPG_MCP_DIR/$d" ] || continue
    ( cd "$CPG_MCP_DIR" \
        && find "$d" -type f ! -path '*/__pycache__/*' ! -name '*.pyc' -print0 \
        | LC_ALL=C sort -z )
  done < <(cpg_mcp_input_dirs)
}

# Sets CPG_MCP_TAG. Returns non-zero (and explains on stderr) rather than hashing an
# absent input: silently skipping one would let two different trees collide on a tag.
cpg_mcp_image_tag() {
  local f h

  while IFS= read -r -d '' f; do
    if [ ! -f "$CPG_MCP_DIR/$f" ]; then
      echo "cpg/mcp/image-tag.sh: build input missing: cpg/mcp/$f" >&2
      echo "  Cannot compute a content hash. Restore the file, or drop it from image-tag.sh" >&2
      echo "  AND from the Dockerfile's COPY list (build.sh --verify-inputs checks they agree)." >&2
      return 3
    fi
  done < <(cpg_mcp_input_files)

  h="$(
    while IFS= read -r -d '' f; do
      printf '%s\0' "$f"
      sha256sum < "$CPG_MCP_DIR/$f"
    done < <(cpg_mcp_input_files) | sha256sum | cut -c1-12
  )"

  if [ "${#h}" -ne 12 ]; then
    echo "cpg/mcp/image-tag.sh: content hash came back malformed ('$h')." >&2
    return 4
  fi

  CPG_MCP_TAG="$h"
}
