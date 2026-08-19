# image-tag.sh — sourced by build.sh and docker-run.sh. Defines the content-hash image
# tag. NOT executable, prints nothing on stdout, reads nothing from stdin.
#
# cypher_mcp_input_dirs   -> directory operands the enumeration WALKS (NUL-separated)
# cypher_mcp_input_files  -> the exact set of build inputs, NUL-separated, relative paths
# cypher_mcp_image_tag    -> sets CYPHER_MCP_TAG=<hash12>  (assigns a variable; no output)
#
# The hash covers file CONTENTS and RELATIVE paths only — never an absolute path, so
# the value is identical on every machine and no home path can reach a tracked file
# (see the plan's "Path convention"). `sha256sum < "$f"` is deliberate: it digests the
# bytes without the filename, and the relative path is contributed explicitly.
#
# INVARIANT: this must cover every path the Dockerfile COPYs, plus the Dockerfile and
# .dockerignore themselves. A FILE operand is covered by cypher_mcp_input_files; a
# DIRECTORY operand is covered by cypher_mcp_input_dirs, which is walked with the same
# exclusions .dockerignore applies. `build.sh --verify-inputs` checks both rules.
#
# What the hash does NOT cover, deliberately: the base image (python:3.12-slim is a
# moving tag) and pip's dependency resolution (requirements.txt pins ranges). So the
# tag is immutable with respect to the TRACKED build inputs, not a full image identity,
# and nothing here ever refreshes an existing image's base — that is the documented
# manual step `docker pull python:3.12-slim && cypher-mcp/build.sh --no-cache`.
# See docs/plans/cpg-mcp-containerization.md §3.6, §4.3.

# Both current callers (build.sh, docker-run.sh) already `set -o pipefail` before
# sourcing this file, which is what lets `if ! ( find ... | sort -z ); then` below
# actually see a `find` failure (pipefail makes the pipeline's exit status the
# rightmost non-zero one, not just sort's). Set it here too, defensively: this file
# affects the CALLER's shell (it is sourced, never executed), so a future caller that
# forgets pipefail would otherwise silently reintroduce the exact "failed find is
# unobserved" gap (m-22) this file exists to close. A no-op for both current callers.
set -o pipefail

CYPHER_MCP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Directory operands. Walked, not globbed: that is what makes a NEW FILE OF ANY
# EXTENSION under tests/ change the hash automatically, instead of only *.py doing so.
cypher_mcp_input_dirs() {
  printf '%s\0' tests
}

# The fixed file operands, then every file under each walked directory that
# .dockerignore would not exclude. `LC_ALL=C sort` matters — a locale-dependent sort
# would make the hash machine-dependent.
#
# Hard-errors (rather than silently degrading) on three cases, each the same failure
# shape a missing FILE operand already gets below — two different trees must never
# collide on one tag:
#   - a walked directory that does not exist (was: silently skipped, colliding with
#     "directory present but empty" — m-20);
#   - a symlink anywhere under a walked directory (the walk is `-type f`, which never
#     matches one, but Docker's build tar carries a symlink AS a symlink — so a tracked
#     symlink change ships into the image without ever moving the tag — m-19);
#   - a `find` that writes an error to stderr partway through, e.g. a permission-denied
#     subtree (previously unobserved: `find`'s error went to stderr but the truncated
#     listing still hashed as a valid, smaller tree — m-22).
#
# Every caller of this function (this file's own cypher_mcp_image_tag, and build.sh's
# verify_inputs via _in_nul_set) may consume its NUL-separated output PARTIALLY — that
# is the whole point of _in_nul_set's early-return-on-match, called out in build.sh's
# own comment above it. Under `pipefail` that early close reliably SIGPIPEs whichever
# of `find`/`sort` is still writing, which is a normal, harmless consequence of a
# partial read — NOT an enumeration failure. So failure here is judged by find's own
# STDERR (captured to a file — the one channel a downstream SIGPIPE never writes to),
# never by an exit code, which a downstream SIGPIPE would otherwise contaminate
# indistinguishably from a real error. And every expression that COULD carry a
# SIGPIPE-tainted non-zero status stays inside an `if`/`!` so it can never trip the
# caller's `set -e` — a bare `var="$(pipeline)"` assignment does not get that
# exemption and would silently kill the whole script (confirmed: `x="$(seq 1 100000 |
# head -n1)"` under `set -euo pipefail` exits 141 before the next line runs).
cypher_mcp_input_files() {
  local f d link errfile

  for f in Dockerfile .dockerignore requirements.txt requirements-dev.txt server.py pytest.ini; do
    printf '%s\0' "$f"
  done

  while IFS= read -r -d '' d; do
    if [ ! -d "$CYPHER_MCP_DIR/$d" ]; then
      echo "cypher-mcp/image-tag.sh: build input directory missing: cypher-mcp/$d" >&2
      echo "  Cannot compute a content hash. Restore the directory, or drop it from" >&2
      echo "  cypher_mcp_input_dirs AND from the Dockerfile's COPY list (build.sh" >&2
      echo "  --verify-inputs checks they agree)." >&2
      return 3
    fi

    errfile="$(mktemp "${TMPDIR:-/tmp}/cypher-mcp-find-err.XXXXXX")" || {
      echo "cypher-mcp/image-tag.sh: mktemp failed; cannot walk cypher-mcp/$d." >&2
      return 3
    }

    link=""
    if ! link="$(cd "$CYPHER_MCP_DIR" \
        && find "$d" -type l ! -path '*/__pycache__/*' ! -path '*/.pytest_cache/*' \
             2>"$errfile" | head -n1)"; then
      : # see the file-header note: a non-zero status here can be a harmless SIGPIPE
    fi
    if [ -s "$errfile" ]; then
      cat "$errfile" >&2
      rm -f "$errfile"
      echo "cypher-mcp/image-tag.sh: 'find' errored scanning cypher-mcp/$d for symlinks" >&2
      echo "  (above) — refusing to hash a tree it could not fully see." >&2
      return 3
    fi
    if [ -n "$link" ]; then
      rm -f "$errfile"
      echo "cypher-mcp/image-tag.sh: symlink under a walked directory: cypher-mcp/$link" >&2
      echo "  The walk only matches '-type f', so it never covers a symlink's target —" >&2
      echo "  yet Docker's build tar ships it. Remove it, or replace it with a real file." >&2
      return 3
    fi

    if ! ( cd "$CYPHER_MCP_DIR" \
        && find "$d" -type f \
             ! -path '*/__pycache__/*' ! -name '*.pyc' ! -path '*/.pytest_cache/*' \
             -print0 2>"$errfile" \
        | LC_ALL=C sort -z ); then
      : # ditto — judged by errfile below, not this exit status
    fi
    if [ -s "$errfile" ]; then
      cat "$errfile" >&2
      rm -f "$errfile"
      echo "cypher-mcp/image-tag.sh: 'find' errored while walking cypher-mcp/$d (above) — a" >&2
      echo "  partially-enumerated tree must not silently hash as a smaller one." >&2
      return 3
    fi
    rm -f "$errfile"
  done < <(cypher_mcp_input_dirs)
}

# Sets CYPHER_MCP_TAG. Returns non-zero (and explains on stderr) rather than hashing an
# absent input: silently skipping one would let two different trees collide on a tag.
#
# Reads cypher_mcp_input_files' NUL-separated output from a real temp FILE, not a process
# substitution: bash cannot hold a NUL byte in a variable (`x="$(printf 'a\0b\0')"`
# silently drops it — confirmed on this box), and a process substitution's exit status
# is never observed by the `while read` that consumes it. Both of those are exactly
# what let a `find` failure inside the walk go unnoticed (m-22) — this is the one place
# that actually matters, since CYPHER_MCP_TAG is the value the launch-time staleness gate
# trusts.
cypher_mcp_image_tag() {
  local f h files_file

  files_file="$(mktemp "${TMPDIR:-/tmp}/cypher-mcp-input-files.XXXXXX")" || {
    echo "cypher-mcp/image-tag.sh: mktemp failed; cannot compute a content hash." >&2
    return 3
  }
  trap 'rm -f "$files_file"' RETURN

  if ! cypher_mcp_input_files > "$files_file"; then
    echo "cypher-mcp/image-tag.sh: failed to enumerate build inputs (see above)." >&2
    return 3
  fi

  while IFS= read -r -d '' f; do
    if [ ! -f "$CYPHER_MCP_DIR/$f" ]; then
      echo "cypher-mcp/image-tag.sh: build input missing: cypher-mcp/$f" >&2
      echo "  Cannot compute a content hash. Restore the file, or drop it from image-tag.sh" >&2
      echo "  AND from the Dockerfile's COPY list (build.sh --verify-inputs checks they agree)." >&2
      return 3
    fi
  done < "$files_file"

  h="$(
    while IFS= read -r -d '' f; do
      printf '%s\0' "$f"
      # Deliberately NOT contributing the file's mode bit (m-18, accepted trade-off):
      # every path COPYed today is invoked via the Dockerfile's CMD (`python server.py`
      # / `python -m pytest tests`), never executed directly, so the executable bit is
      # inert for the current image. Hashing it would cost a stat() per file for a
      # property nothing reads. Revisit if a CMD/ENTRYPOINT ever execs one of these
      # files directly (`./server.py`) — at that point the bit stops being inert.
      sha256sum < "$CYPHER_MCP_DIR/$f"
    done < "$files_file" | sha256sum | cut -c1-12
  )"

  if [ "${#h}" -ne 12 ]; then
    echo "cypher-mcp/image-tag.sh: content hash came back malformed ('$h')." >&2
    return 4
  fi

  CYPHER_MCP_TAG="$h"
}
