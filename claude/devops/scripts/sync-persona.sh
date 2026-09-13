#!/usr/bin/env bash
# Regenerates the SHARED-PERSONA span in devops.md from devops-persona.md.
#
# devops-persona.md is the single source of truth for the shared devops/tank persona text
# (see claude/devops/devops.md's own comment, and the plan at
# opencode/docs/plans/devops-opencode-headless.md §3.1). Whenever devops-persona.md changes,
# run this script to keep devops.md's copy in lockstep — it is not live-included the way
# OpenCode's tank agent includes it, so regeneration is the only sync mechanism on this side.
#
# Idempotent: running it twice in a row with an unchanged devops-persona.md produces a no-op
# diff on devops.md.
#
# Usage: claude/devops/scripts/sync-persona.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEVOPS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PERSONA_FILE="$DEVOPS_DIR/devops-persona.md"
TARGET_FILE="$DEVOPS_DIR/devops.md"

BEGIN_MARKER='<!-- SHARED-PERSONA:BEGIN -->'
END_MARKER='<!-- SHARED-PERSONA:END -->'

if [[ ! -f "$PERSONA_FILE" ]]; then
  echo "sync-persona.sh: missing persona source file: $PERSONA_FILE" >&2
  exit 1
fi

if [[ ! -f "$TARGET_FILE" ]]; then
  echo "sync-persona.sh: missing target file: $TARGET_FILE" >&2
  exit 1
fi

# grep -c exits 1 when the count is 0; guard with `|| true` so `set -e` doesn't abort before
# we get to report a clear error below.
begin_count=$(grep -cF -- "$BEGIN_MARKER" "$TARGET_FILE" || true)
end_count=$(grep -cF -- "$END_MARKER" "$TARGET_FILE" || true)

if [[ "$begin_count" -ne 1 ]]; then
  echo "sync-persona.sh: expected exactly one '$BEGIN_MARKER' marker in $TARGET_FILE, found $begin_count" >&2
  exit 1
fi

if [[ "$end_count" -ne 1 ]]; then
  echo "sync-persona.sh: expected exactly one '$END_MARKER' marker in $TARGET_FILE, found $end_count" >&2
  exit 1
fi

begin_line=$(grep -nF -- "$BEGIN_MARKER" "$TARGET_FILE" | cut -d: -f1)
end_line=$(grep -nF -- "$END_MARKER" "$TARGET_FILE" | cut -d: -f1)

if [[ "$begin_line" -ge "$end_line" ]]; then
  echo "sync-persona.sh: '$BEGIN_MARKER' (line $begin_line) must come before '$END_MARKER' (line $end_line) in $TARGET_FILE" >&2
  exit 1
fi

tmp_file="$(mktemp "${TARGET_FILE}.XXXXXX")"
trap 'rm -f "$tmp_file"' EXIT

{
  head -n "$begin_line" "$TARGET_FILE"
  cat "$PERSONA_FILE"
  tail -n "+$end_line" "$TARGET_FILE"
} > "$tmp_file"

mv "$tmp_file" "$TARGET_FILE"
trap - EXIT

echo "sync-persona.sh: synced $(wc -l < "$PERSONA_FILE") persona lines into $TARGET_FILE"
