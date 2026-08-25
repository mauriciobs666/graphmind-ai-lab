#!/usr/bin/env bash
# audit-team.sh — deterministic half of the team-coherence certification
# (agent-maintenance skill §4). Read-only: greps the agent collection for the
# mechanical invariants that drift silently; the judgment half (roster
# accuracy, handoff symmetry, subagent-awareness, enforcement parity) stays
# with the maintainer (cobb).
#
# Checks per agent:
#   1. <name>/<name>.md has its kaizen/{plan,history}.md pair (inbox.md is no
#      longer part of the pass/fail condition, in either direction — no
#      agent has one any more: the 12 that existed at the 2026-08-20
#      migration each carried a frozen one, deleted outright 2026-08-21 once
#      fully distilled, and an agent created since never got one; see
#      FR-12/AC-9, docs/plans/generic-cypher-mcp2.md)
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
#   8. git-commit-authority scoping — every one of the 13 agents must document
#      `git add`/`git commit` authority AND state the delegated-subagent
#      carve-out (the grant applies only when the agent runs interactively,
#      not when spawned as an isolated delegate). Stakeholder decision,
#      2026-07-30: "I dont want the subagents to proliferate commits, tico
#      and teco are special and have coordination rights" — restricted broad,
#      mode-unconditioned commit authority to tico/teco only (still true;
#      documented in claude/AGENTS.md's "Git-commit authority" section).
#   9. prompt-weight advisory (NOTE, never FAIL) — the drift tripwire for the
#      prompt-waste doctrine (claude/docs/plans/prompt-waste-reduction.md
#      Stage F). Prints a NOTE per agent whose prompt body exceeds
#      AUDIT_WORD_LIMIT (default 2500) and an INFO corpus total. Deliberately
#      cannot fail: a rule-dense prompt above the line is a pass, and a
#      tripwire that could fail would pressure someone to cut a rule to hit a
#      number — the one outcome that effort exists to prevent.
#      Superseded in part, 2026-08-21: every agent may now additionally
#      commit its own verified work specifically when running interactively
#      — this check is the deterministic backstop so a future prompt edit
#      can't silently drop either half (claiming the verb with no carve-out
#      reads as the old unconditioned grant, which stays tico/teco-only; no
#      claim at all misses the 2026-08-21 policy). No PreToolUse hook can
#      gate this either way: it's prose capability, not a Write/Edit path or
#      a Bash command pattern.
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
notes=0
pass()    { printf 'PASS  %s\n' "$1"; }
failmsg() { printf 'FAIL  %s\n' "$1"; fail=1; }
note()    { printf 'NOTE  %s\n' "$1"; notes=$((notes+1)); }

agents=()
for d in "$CL"/*/; do
  name="$(basename "$d")"
  [ -f "$d$name.md" ] && agents+=("$name")
done
[ "${#agents[@]}" -gt 0 ] || { failmsg "no agents found under $CL"; exit 1; }
printf 'Auditing %d agents: %s\n\n' "${#agents[@]}" "${agents[*]}"

for a in "${agents[@]}"; do
  src="$CL/$a/$a.md"

  # 1. kaizen pair (plan/history curated by the maintainer; inbox.md no
  #    longer exists for any agent — deleted 2026-08-21 once fully
  #    distilled — and was never required by this check, agent-maintenance
  #    skill §5)
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
BOUNDARY_PAIRS=("coder:tdd-engineer" "coder:frontend-engineer" "analyst:qa-engineer" "graph-dba:devops" "architect:data-scientist" "analyst:data-scientist" "graph-dba:data-scientist" "tdd-engineer:qa-engineer" "security-expert:analyst" "security-expert:cobb" "security-expert:devops")
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

# 8. git-commit-authority scoping — every agent claims it, and correctly
#    carves out delegated-subagent mode (2026-08-21 universal grant); tico/teco
#    additionally carry a broader, mode-unconditioned grant documented in
#    claude/AGENTS.md, which this check doesn't distinguish — it only verifies
#    every agent has *some* correctly-scoped commit language.
echo
for a in "${agents[@]}"; do
  if grep -qE '`?git (add|commit)`?' "$CL/$a/$a.md"; then
    if grep -qi "delegated subagent" "$CL/$a/$a.md"; then
      pass "$a: documents git add/commit authority, scoped to interactive mode (delegated-subagent carve-out present)"
    else
      failmsg "$a: claims git add/commit authority but doesn't state the delegated-subagent carve-out — reads as an unconditioned grant, which only tico/teco may claim"
    fi
  else
    failmsg "$a: documents no git add/commit authority — expected under the 2026-08-21 universal interactive-mode grant (claude/AGENTS.md, 'Git-commit authority')"
  fi
done

# 9. prompt-weight advisory — ADVISORY ONLY, never fails, by design.
#    The drift tripwire for the prompt-waste doctrine: a prompt that has
#    regrown past the threshold gets a human read, not a broken build.
#    It must never gate, because a rule-dense prompt legitimately above the
#    line is a pass (the plan's §7: "a file above target with every rule
#    intact passes — the band moves, not the file"), and a tripwire that can
#    fail would pressure someone to cut a rule to reach a number, which is
#    the one outcome the whole effort is built to prevent.
#    Counts the prompt BODY (frontmatter stripped), matching how every
#    figure in claude/docs/plans/prompt-waste-reduction.md was measured.
#    Threshold override: AUDIT_WORD_LIMIT=<n>.
echo
limit="${AUDIT_WORD_LIMIT:-2500}"
total=0
for a in "${agents[@]}"; do
  w=$(awk 'NR>1 && /^---$/{p=1;next} p' "$CL/$a/$a.md" | wc -w)
  total=$((total+w))
  if [ "$w" -gt "$limit" ]; then
    note "$a: prompt body is ${w}w, above the ${limit}w advisory threshold — re-read it against the promotion rule (agent-maintenance §5: a promoted kaizen entry lands as rule + <=1-clause why, nothing else). Advisory only; a rule-dense prompt above the line is a pass."
  fi
done
[ "$notes" -eq 0 ] && pass "prompt weight: all ${#agents[@]} prompt bodies at or under ${limit}w"
printf 'INFO  prompt corpus: %sw across %s agents (mean %sw)\n' \
  "$total" "${#agents[@]}" "$((total / ${#agents[@]}))"

echo
if [ "$fail" -eq 0 ]; then
  echo "RESULT: PASS — deterministic checks clean. The judgment checklist (agent-maintenance skill §4) still applies."
  [ "$notes" -gt 0 ] && echo "        ($notes advisory NOTE(s) above — informational, not failures.)"
else
  echo "RESULT: FAIL — fix the items above, then re-run."
fi
exit "$fail"
