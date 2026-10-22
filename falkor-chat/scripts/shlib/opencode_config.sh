# shellcheck shell=bash
# scripts/shlib/opencode_config.sh — sourced, never executed. The one definition of how
# falkor-chat's bring-up scripts pick FALKORCHAT_OPENCODE_CONFIG when the caller left it
# unset (K-083). Sourced by start_server.sh, start_demo.sh, ensure_demo_models.sh and
# start_agent_team.sh. The one-off maintenance scripts (seed_eval_corpus.sh,
# migrate_embeddings.sh, pin_workspace_embedding_model.sh) deliberately do NOT use it:
# they name the shared file.
#
# resolve_opencode_config — sets AND exports FALKORCHAT_OPENCODE_CONFIG, so every child
# the caller runs (e.g. create_workspace.sh -> pin_workspace_embedding_model.sh) reads the
# same file as the ensure step and uvicorn, not its own default:
#   1. the env var, if set and non-empty — used as-is, silently (the caller's call,
#      e.g. a QA pass's own scratch copy);
#   2. else $HOME/.config/opencode/opencode.local.json, if it exists — a local,
#      untracked, hand-maintained override beside the shared file; one INFO line on
#      stderr so the run's own output shows which file is in play;
#   3. else the shared $HOME/.config/opencode/opencode.json — a WARNING on stderr,
#      because this is the fallback that goes stale silently (e.g. a LAN IP that is
#      unreachable under WSL2); the warning names the local-override path.
# Never writes either file: the shared opencode.json is read by other tools/sessions.

resolve_opencode_config() {
  local local_override shared_default
  if [ -n "${FALKORCHAT_OPENCODE_CONFIG:-}" ]; then
    # explicit override — used as-is, no message; HOME is not even read (set -u safe)
    export FALKORCHAT_OPENCODE_CONFIG
    return 0
  fi
  local_override="$HOME/.config/opencode/opencode.local.json"
  shared_default="$HOME/.config/opencode/opencode.json"
  if [ -f "$local_override" ]; then
    FALKORCHAT_OPENCODE_CONFIG="$local_override"
    echo "INFO: FALKORCHAT_OPENCODE_CONFIG not set — using local override $local_override" >&2
  else
    FALKORCHAT_OPENCODE_CONFIG="$shared_default"
    echo "WARNING: FALKORCHAT_OPENCODE_CONFIG not set and no $local_override found —" >&2
    echo "         falling back to the shared $shared_default. Its provider baseURL(s)" >&2
    echo "         may not be reachable from this box (e.g. a stale LAN IP under WSL2 — see" >&2
    echo "         claude/qa-engineer/qa-testing-techniques.md). To fix without touching that" >&2
    echo "         shared file, create $local_override with a corrected baseURL." >&2
  fi
  export FALKORCHAT_OPENCODE_CONFIG
}
