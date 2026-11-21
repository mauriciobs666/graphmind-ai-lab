# Review: coordination-graph S1c (cobb guard widening)

> **Status:** archived · **Owner:** `analyst` · **Tracks:** coordination-graph (S1c, M7)

**Scope:** commit `c85cd417` (`claude/cobb/hooks/guard-cobb-topic-writes.sh`, `claude/cobb/kaizen/history.md`) against `claude/docs/plans/coordination-graph.md` §4 row S1c and `claude/docs/reviews/coordination-graph.md` M7. The core's `case` semantics were read in `claude/scripts/guard-doc-writes.sh`. The hook was executed with synthetic PreToolUse JSON.

**Verdict: approve with suggestions.**

CPG: considered, not relevant — a bash glob allowlist; no code-graph question.

## Findings

**m1 (minor) — the new dir globs also allow `..` traversal out of the intended dirs.** The core does no path normalisation, and `case` lets `*` cross `/`. Observed decisions:

| path | decision |
|---|---|
| `claude/cobb/scripts/../../scripts/guard-doc-writes.sh` | **allow** (the core the plan says stays out) |
| `claude/cobb/scripts/../hooks/guard-cobb-topic-writes.sh` | **allow** (the guard itself) |
| `skills/coordination-graph/../../.claude/settings.json` | **allow** |
| `/other/repo/skills/agent-kb-retrieval/x.sh` | allow (foreign repo, intended by the `*/` form) |
| absolute `.../claude/scripts/guard-doc-writes.sh` | ask |
| `docs/BACKLOG.md` (not re-run; history entry's claim, consistent with the globs) | ask |
| `claude/scripts/audit-team.sh.bak` | ask |
| `coordination-graph/other.py` | ask |

All five new paths allow in both relative and absolute form. Nothing else matches by plain path; the audited deviation is only the `..` form.

This is not new in kind. `skills/agent-maintenance/*` and `claude/*/*.md` already admit the same traversal. But `claude/cobb/scripts/*` and `skills/coordination-graph/*` are the first entries that make non-`.md` targets such as `.sh` and `.json` reachable via `..`. The header's claim "`claude/scripts/*` (incl. this guard's core) stays out" holds for literal paths only. The harness's own protection of `.claude/` is an independent backstop and was not tested here.

Suggested improvement:
- (Preferred, but a core change outside S1c.) Have `guard-doc-writes.sh` return `ask` for any path containing a `/../` segment (`case "$path" in */../*|../*) ...`). That fixes every wrapper at once.
- Alternatively, accept and note it in the header. The risk is accidental drift, not an adversary: Write/Edit paths come from the agent itself.
- Add a probe case for `..` to whichever test covers the guard.

**n1 (nit) — over-broad `*/` forms.** `*/claude/cobb/scripts/*` and the like match any prefix, including foreign repos. This is the documented, repo-wide convention (the header's own explanation), so there is no change to make. Mentioned only because the check asked.

## Answers

1. **Allowlist:** exactly the five stated paths are added, each doubled bare + `*/`-prefixed. They appear identically in the header and in the `exec` glob string, with no extras. Nothing broader than the plan's S1c row. The only widening beyond literal reading is the traversal in m1.
2. **Header comment:** accurate. It lists the five entries, cites plan §4 and M7, names the helper (`hook-probe.sh`), and states that `claude/scripts/*` stays out. That last statement is true for literal paths (checked: ask). It is silent on the `..` caveat (m1).
3. **history.md entry:** in house format — dated H2 at top, `What / Why / Plan items`, most recent first. Its factual claims are consistent with my observations: the five paths allow and `claude/scripts/guard-doc-writes.sh` asks. The `Plan items: —` line matches the house form.

## What's solid

- The allowlist is minimal and explicit, and follows the file's stated "extend the list, don't broaden the globs" rule.
- Individual files (`audit-team.sh`, `kb-files.txt`) are pinned by exact name. Near-miss names (`.bak`, `other.py`) ask.
- The change is confined to two files; no core script was touched.

## Open questions

None blocking. Should the `..`-segment rejection (m1) be scheduled as a separate small unit against `claude/scripts/guard-doc-writes.sh`?
