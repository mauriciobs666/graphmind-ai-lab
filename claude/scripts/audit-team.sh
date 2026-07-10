#!/usr/bin/env bash
# audit-team.sh — deterministic half of the team-coherence certification
# (agent-maintenance skill §4). Read-only: greps the agent collection for the
# mechanical invariants that drift silently; the judgment half (roster
# accuracy, handoff symmetry, subagent-awareness, enforcement parity) stays
# with the maintainer (cobb).
#
# Checks per agent:
#   1. <name>/<name>.md has its kaizen/{plan,history}.md pair
#   2. the agent is symlinked into ~/.claude/agents/ (deployed)
#   3. every frontmatter hook command exists and is executable
#   4. the agent is named in the orchestrator's (teco) prompt — roster drift
#   5. the agent is cataloged in claude/AGENTS.md, claude/README.md, root AGENTS.md
#
# Collection-wide:
#   6. boundary-pair symmetry — adjacent specialists whose scopes border each
#      other must each name the other in their frontmatter `description` (the
#      routing contract every router sees). Pairs declared in BOUNDARY_PAIRS.
#   7. personal-info leak — no tracked file anywhere in the repo may
#      contain the maintainer's personal identifiers: home path, username,
#      git user.name, git user.email, or hostname. Patterns are derived at
#      runtime (never hardcoded here — that would itself be the leak), so the
#      check protects whoever runs it. Committed artifacts must be machine-
#      and identity-portable ($HOME/.claude/agents/<name>/… resolves via the
#      deployment symlink on any machine). Origin: 2026-07-10, six agents'
#      hook commands were committed with the absolute /home/<user>/… path.
#
# Exit 0 = all PASS; exit 1 = at least one FAIL.
# Origin: 2026-07-09 teco interface review — teco's roster had silently missed
# qa-engineer + devops for days; catalogs can't see inter-agent drift.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CL="$ROOT/claude"
AGENTS_HOME="${CLAUDE_AGENTS_DIR:-$HOME/.claude/agents}"
ORCHESTRATOR="teco"

fail=0
pass()    { printf 'PASS  %s\n' "$1"; }
failmsg() { printf 'FAIL  %s\n' "$1"; fail=1; }

agents=()
for d in "$CL"/*/; do
  name="$(basename "$d")"
  [ -f "$d$name.md" ] && agents+=("$name")
done
[ "${#agents[@]}" -gt 0 ] || { failmsg "no agents found under $CL"; exit 1; }
printf 'Auditing %d agents: %s\n\n' "${#agents[@]}" "${agents[*]}"

for a in "${agents[@]}"; do
  src="$CL/$a/$a.md"

  # 1. kaizen pair
  if [ -f "$CL/$a/kaizen/plan.md" ] && [ -f "$CL/$a/kaizen/history.md" ]; then
    pass "$a: kaizen plan + history present"
  else
    failmsg "$a: missing kaizen/plan.md or kaizen/history.md"
  fi

  # 2. deployment symlink
  if [ -e "$AGENTS_HOME/$a" ] && [ "$(readlink -f "$AGENTS_HOME/$a")" = "$(readlink -f "$CL/$a")" ]; then
    pass "$a: deployed ($AGENTS_HOME/$a → claude/$a)"
  else
    failmsg "$a: not symlinked into $AGENTS_HOME (or points elsewhere)"
  fi

  # 3. frontmatter hook commands exist + are executable
  while IFS= read -r hook; do
    [ -n "$hook" ] || continue
    # frontmatter hooks run shell-form (sh -c), so mirror its $HOME/~ expansion
    hook="${hook//\$HOME/$HOME}"
    hook="${hook/#\~/$HOME}"
    if [ -x "$hook" ]; then
      pass "$a: hook exists + executable ($hook)"
    else
      failmsg "$a: hook missing or not executable ($hook)"
    fi
  done < <(awk '/^---$/{f++} f==1 && /command:/{sub(/.*command:[ \t]*/,""); print}' "$src")

  # 4. orchestrator roster completeness
  if [ "$a" != "$ORCHESTRATOR" ]; then
    if grep -qE "\b$a\b" "$CL/$ORCHESTRATOR/$ORCHESTRATOR.md"; then
      pass "$a: present in $ORCHESTRATOR's roster"
    else
      failmsg "$a: NOT mentioned in $ORCHESTRATOR's prompt — roster drift"
    fi
  fi

  # 5. catalogs (agent-context file, human catalog, repo-root context)
  for doc in "$CL/AGENTS.md" "$CL/README.md" "$ROOT/AGENTS.md"; do
    if grep -qE "\b$a\b" "$doc"; then
      pass "$a: cataloged in ${doc#"$ROOT"/}"
    else
      failmsg "$a: missing from ${doc#"$ROOT"/}"
    fi
  done
done

# 6. boundary-pair symmetry in frontmatter descriptions
BOUNDARY_PAIRS=("coder:tdd-engineer" "coder:frontend-engineer" "analyst:qa-engineer" "graph-dba:devops" "architect:data-scientist" "analyst:data-scientist" "graph-dba:data-scientist")
desc_of() { awk '/^---$/{f++} f==1 && /^description:/{sub(/^description:[ \t]*/,""); print; exit}' "$CL/$1/$1.md"; }
echo
for p in "${BOUNDARY_PAIRS[@]}"; do
  for x in "${p%%:*}:${p##*:}" "${p##*:}:${p%%:*}"; do
    s="${x%%:*}"; t="${x##*:}"
    if desc_of "$s" | grep -qE "\b$t\b"; then
      pass "$s: description routes its boundary to $t"
    else
      failmsg "$s: description never names $t — boundary asymmetry (route-away clause missing)"
    fi
  done
done

# 7. personal-info leak — committed artifacts must be machine- and identity-portable
echo
declare -A pii=()                                  # label → pattern (runtime-derived, never hardcoded)
[ -n "${HOME:-}" ]  && pii["home path"]="$HOME"
u="$(id -un 2>/dev/null || true)"
[ -n "$u" ]         && pii["username"]="$u"
gn="$(git -C "$ROOT" config user.name 2>/dev/null || true)"
[ -n "$gn" ]        && pii["git user.name"]="$gn"
ge="$(git -C "$ROOT" config user.email 2>/dev/null || true)"
[ -n "$ge" ]        && pii["git user.email"]="$ge"
hn="$(hostname 2>/dev/null || true)"
[ -n "$hn" ]        && pii["hostname"]="$hn"
leaked=0
for label in "${!pii[@]}"; do
  wordflag=()                                      # short bare tokens get word bounds to avoid substring noise
  case "$label" in username|hostname) wordflag=(-w) ;; esac
  hits="$(git -C "$ROOT" grep -I -n -i "${wordflag[@]}" -F "${pii[$label]}" 2>/dev/null)" || continue
  printf '%s\n' "$hits" | sed 's/^/      /'
  failmsg "repo: $label leaked into tracked files — genericize it (paths: \$HOME/.claude/agents/<name>/…, prose: /home/<user>/…)"
  leaked=1
done
[ "$leaked" -eq 0 ] && pass "repo: no personal identifiers (home path, username, git name/email, hostname) in any tracked file"

echo
if [ "$fail" -eq 0 ]; then
  echo "RESULT: PASS — deterministic checks clean. The judgment checklist (agent-maintenance skill §4) still applies."
else
  echo "RESULT: FAIL — fix the items above, then re-run."
fi
exit "$fail"
