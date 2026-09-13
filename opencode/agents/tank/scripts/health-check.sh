#!/usr/bin/env bash
# health-check.sh — run tank's default read-only health/hygiene check.
#
# Outer script: this is the reviewed, scriptable entry point a human or
# scheduler invokes (opencode/docs/plans/devops-opencode-headless.md §4 step
# 9) — distinct from the *inner* scripts (compose-status.sh/compose-up.sh/
# compose-down.sh, same directory) that tank itself invokes via its bash
# tool. This script never touches docker directly; it only starts tank and
# lets tank's own permission.bash allow-list gate what happens next.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TANK_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Project-scoped opencode.json is only loaded from its own directory
# (verified live, plan §2.3) — cd there rather than relying on --dir.
cd "$TANK_DIR"
# --title supplies the session title up front, which skips OpenCode's own
# background auto-title LLM call entirely (verified live, 2026-09-13: with
# no --title, every run also fires a second `agent=title small=true` call
# against the same model, which 500s on ministral-3-3b's chat template —
# "conversation roles must alternate..." — swallowed, harmless to tank's own
# result, but a wasted failing call + log noise on every invocation; passing
# --title suppresses that second call outright, confirmed via the OpenCode
# and LM Studio logs). No config-schema equivalent exists (checked
# opencode.json's agent/top-level fields) — this CLI flag is the only lever.
exec opencode run --agent tank --title "tank health-check" "Run the default health/hygiene check: container/service status, disk usage, dangling images/volumes, and Docker build-cache usage. Report what you find."
