#!/usr/bin/env bash
# audit-team.sh — deterministic half of the team-coherence certification
# (agent-maintenance skill §4). Read-only: greps the agent collection for the
# mechanical invariants that drift silently; the judgment half (roster
# accuracy, handoff symmetry, subagent-awareness, enforcement parity) stays
# with the maintainer (cobb).
#
# Checks per agent:
#   1. <name>/<name>.md has its kaizen/{plan,history,inbox}.md triple
#   2. the agent is symlinked into ~/.claude/agents/ (deployed)
#   3. every frontmatter hook command exists and is executable
#   4. the agent is named in the orchestrator's (teco) prompt — roster drift
#   5. the agent is cataloged in claude/AGENTS.md and claude/README.md — the
#      catalog owners (per-agent; root AGENTS.md deliberately does NOT
#      duplicate the roster since 2026-07-28, see check 5b)
#
# Collection-wide:
#   5b. root AGENTS.md still points at claude/AGENTS.md + claude/README.md
#       (delegates to the catalog rather than re-duplicating it — the
#       2026-07-28 trim removed the inline 12-agent roster on purpose; this
#       checks the pointer survives, not that every name is repeated there).
#   6. boundary-pair symmetry — adjacent specialists whose scopes border each
#      other must each name the other in their frontmatter `description` (the
#      routing contract every router sees). Pairs declared in BOUNDARY_PAIRS.
#   7. personal-info leak — no tracked *or untracked-but-not-gitignored* file
#      anywhere in the repo may contain the maintainer's personal identifiers:
#      home path, username, git user.name, git user.email, or hostname.
#      Patterns are derived at runtime (never hardcoded here — that would
#      itself be the leak), so the check protects whoever runs it. Committed
#      artifacts must be machine- and identity-portable
#      ($HOME/.claude/agents/<name>/… resolves via the deployment symlink on
#      any machine). Origin: 2026-07-10, six agents' hook commands were
#      committed with the absolute /home/<user>/… path. Widened 2026-08-08
#      (C-309b) — a brand-new untracked file used to be invisible to `git
#      grep`; the scan now unions tracked files with untracked-but-not-
#      ignored ones so the gate catches a leak before the first commit, not
#      only after.
#   8. git-commit-authority containment — only tico and teco may document
#      `git add`/`git commit` authority in their own prompt (COMMIT_AUTHORS
#      below); every other agent's <name>.md must stay free of those verbs
#      entirely. Stakeholder decision, 2026-07-30: "I dont want the subagents
#      to proliferate commits, tico and teco are special and have coordination
#      rights" — this check is the deterministic backstop so a future prompt
#      edit can't silently re-open commit authority for a specialist without
#      a human noticing (no PreToolUse hook can gate this: it's a prose
#      capability, not a Write/Edit path or a Bash command pattern).
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

  # 1. kaizen triple (plan/history curated by the maintainer; inbox is the
  #    agent's own append-only learnings capture — agent-maintenance skill §5)
  if [ -f "$CL/$a/kaizen/plan.md" ] && [ -f "$CL/$a/kaizen/history.md" ] && [ -f "$CL/$a/kaizen/inbox.md" ]; then
    pass "$a: kaizen plan + history + inbox present"
  else
    failmsg "$a: missing kaizen/plan.md, kaizen/history.md, or kaizen/inbox.md"
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

  # 5. catalogs (agent-context file, human catalog — the two catalog owners)
  for doc in "$CL/AGENTS.md" "$CL/README.md"; do
    if grep -qE "\b$a\b" "$doc"; then
      pass "$a: cataloged in ${doc#"$ROOT"/}"
    else
      failmsg "$a: missing from ${doc#"$ROOT"/}"
    fi
  done
done

# 5b. root AGENTS.md delegates to the claude/ catalog rather than duplicating
#     it (2026-07-28 trim removed the inline roster on purpose; DRY per the
#     agent-maintenance skill §2). Check the pointer, not every name.
if grep -q 'claude/AGENTS.md' "$ROOT/AGENTS.md" && grep -q 'claude/README.md' "$ROOT/AGENTS.md"; then
  pass "root AGENTS.md: still points to claude/AGENTS.md + claude/README.md (roster delegated, not duplicated)"
else
  failmsg "root AGENTS.md: no longer points to claude/AGENTS.md and/or claude/README.md — the claude/ subagent catalog pointer is missing"
fi

# 6. boundary-pair symmetry in frontmatter descriptions
BOUNDARY_PAIRS=("coder:tdd-engineer" "coder:frontend-engineer" "analyst:qa-engineer" "graph-dba:devops" "architect:data-scientist" "analyst:data-scientist" "graph-dba:data-scientist" "tdd-engineer:qa-engineer")
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
  # Tracked + untracked-but-not-gitignored, so a brand-new file leaking an
  # identifier can't pass this gate silently just because it hasn't been
  # `git add`ed yet (C-309b, 2026-08-08 — plain `git grep` only sees tracked
  # content). Check output emptiness rather than exit code so an empty file
  # list (xargs -r skips the run) and a clean grep (exit 1) both fall through
  # to "no hits" instead of one of them misfiring as a false FAIL.
  hits="$(cd "$ROOT" && git ls-files -z --cached --others --exclude-standard \
            | xargs -0 -r grep -I -n -i "${wordflag[@]}" -F -e "${pii[$label]}" -- 2>/dev/null)"
  [ -n "$hits" ] || continue
  printf '%s\n' "$hits" | sed 's/^/      /'
  failmsg "repo: $label leaked into a tracked or untracked (non-ignored) file — genericize it (paths: \$HOME/.claude/agents/<name>/…, prose: /home/<user>/…)"
  leaked=1
done
[ "$leaked" -eq 0 ] && pass "repo: no personal identifiers (home path, username, git name/email, hostname) in any tracked or untracked (non-ignored) file"

# 8. git-commit-authority containment — only tico/teco may claim git add/commit
echo
COMMIT_AUTHORS=("tico" "teco")
is_commit_author() { local n; for n in "${COMMIT_AUTHORS[@]}"; do [ "$n" = "$1" ] && return 0; done; return 1; }
for a in "${agents[@]}"; do
  if is_commit_author "$a"; then
    if grep -qE '`?git (add|commit)`?' "$CL/$a/$a.md"; then
      pass "$a: documents its git commit authority (stakeholder-approved coordination right)"
    else
      failmsg "$a: is a designated commit author (COMMIT_AUTHORS) but its prompt documents no git add/commit authority — grant missing or worded unrecognizably"
    fi
  else
    if grep -qE '`?git (add|commit)`?' "$CL/$a/$a.md"; then
      failmsg "$a: prompt claims git add/commit authority — only tico/teco may (stakeholder decision 2026-07-30, no proliferation of commit rights)"
    else
      pass "$a: no git commit authority claimed (correct — not tico/teco)"
    fi
  fi
done

echo
if [ "$fail" -eq 0 ]; then
  echo "RESULT: PASS — deterministic checks clean. The judgment checklist (agent-maintenance skill §4) still applies."
else
  echo "RESULT: FAIL — fix the items above, then re-run."
fi
exit "$fail"
