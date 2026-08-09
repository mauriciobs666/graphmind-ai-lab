# CPG follow-ups (Wave 1) — skill/documentation review

> **Status:** archived · **Owner:** `cobb` · **Tracks:** C-308, C-312, C-319

> **Split note:** the coordination brief's default deliverable path,
> `docs/reviews/cpg-followups-impl.md`, was written to concurrently by `analyst`'s parallel review
> of U3–U5 (`coder`/`tdd-engineer`/`devops`) — the two Wave-2 review units ran at the same time and
> the later write overwrote the earlier one at that shared path. This file is the collision
> fallback named in this unit's brief: `cobb`'s review of the skill/documentation units (U1, U2,
> U6) only. See `docs/reviews/cpg-followups-impl.md` for `analyst`'s U3–U5 review.

Independent review of U1, U2, U6 from `docs/plans/cpg-followups-coordination.md` — the three
skill/documentation units of this round's Wave 1. Reviewed against this repo's skill-authoring
conventions (schema-cited-not-duplicated, live-verification discipline) and the
`agent-maintenance` skill's distillation procedure. All three units' diffs were uncommitted in the
working tree at review time (`git diff` from repo root).

## U1 — C-308: Q4 transitive upward call closure (`skills/cpg-analysis/references/impact-analysis.md`)

**Verdict: approve, no changes needed.**

- **Query correctness.** Traced the three-level climb by hand against the query text: L1 finds
  method names that directly call the target (`CONTAINS`→`CALL{NAME:target}`, same reliable
  direction as Q1); L2/L3 each find method names that contain a call site named after anyone in
  the previous level, net-new only (`WHERE NOT x IN …`), exactly the "who calls a caller"
  semantics the prose claims; the final `caller.NAME IN closure` resolves the accumulated name set
  back to concrete `METHOD` nodes. This is a correct generalization of the existing Q1 pattern, not
  a different algorithm dressed up.
- **`WITH`-splitting idiom.** Cross-checked against `test-gap.md` (its own `WITH L1, collect(...)`
  / `WITH L12, [x IN … WHERE NOT x IN …]` two-step pattern, `_AR_EXP_UpdateEntityIdx` note at line
  66) — Q4 reuses the identical idiom, walked upward instead of downward, as claimed.
- **The self-recursion caveat.** The "don't re-add `caller.NAME <> '<target>'`" warning is
  well-placed (immediately after the query, before a reader would be tempted to "fix" what looks
  like an obvious omission) and the reasoning is sound: Q1 doesn't filter self-calls either, so Q4
  filtering them would be an inconsistency, not a fix.
- **24 vs. 21 rows, caveat accuracy.** The three-way breakdown (1 genuine L2 hit + 2 name-collision
  artifacts: the target's own definition and a Joern synthetic stub) is internally consistent and
  independently checkable — the doc gives the exact confirming query
  (`MATCH (m:METHOD) WHERE m.NAME = 'post_message' RETURN m.FULL_NAME, m.IS_EXTERNAL`) rather than
  just asserting the count. The suggested cleanup (`AND caller.IS_EXTERNAL = false` +
  `FILENAME`+`LINE_NUMBER` exclusion of the target itself) is the right fix and explicitly contrasts
  with the rejected `caller.NAME <>` approach so a reader can't accidentally reintroduce the bug.
- **Schema citation, not duplication.** Confirmed the document's only schema-fact claims
  (`CONTAINS`/`CALL` semantics, `IS_EXTERNAL` as a real boolean, "inbound call resolution is
  sparse") are already stated in `skills/joern-cpg/references/cpg-model.md` (lines 81–144) and this
  doc points at rather than restates them — consistent with the component's own stated rule and
  with how Q1/Q3 already behave in the unmodified parts of the file.
- **Limits section.** The new bullet is additive and correctly scoped (Q4-specific caveat, doesn't
  restate the existing downstream/dynamic-dispatch bullets).

No findings at any severity.

## U2 — C-312: `--verify-prefix` on `pipeline.sh` + `SKILL.md` Gotchas (`skills/joern-cpg/`)

**Verdict: approve, no changes needed; one optional minor.**

- **Repeatable-flag parsing.** `VERIFY_PREFIXES=()` initialized before the arg loop;
  `--verify-prefix) VERIFY_PREFIXES+=("$2"); shift 2 ;;` appends correctly on each occurrence —
  multiple `--verify-prefix a --verify-prefix b` calls populate a proper bash array, not a
  clobbered scalar.
- **No short-circuit, full reporting, correct aggregate exit.** The verification loop
  (`pipeline.sh` lines 106–124) iterates every entry in `VERIFY_PREFIXES` unconditionally, echoing
  an OK or FAILED line for *each* prefix and only setting a `VERIFY_FAILED` flag on failure — it
  does not `exit`/`break` on the first failure. The aggregate `exit 1` (with the fix-it message)
  fires only after the loop completes, so a caller passing three prefixes where the second fails
  still sees all three results before the pipeline dies. This is exactly the behavior the brief
  asked to verify, and it's correct.
- **Gotchas framing.** The SKILL.md edit's "Scripted check" (via `--verify-prefix`, exits non-zero)
  vs. "Manual check" (the pre-existing `MATCH (m:METHOD) RETURN DISTINCT m.FILENAME LIMIT 10` /
  explicit count query, for stage-by-stage runs or an already-loaded graph) split accurately
  reflects what the code does — `--verify-prefix` is only reachable inside the `if [ -n "$LOAD" ]`
  block, so it genuinely doesn't apply when stages are run individually without `--load`.
- **The "red herring" claim.** SKILL.md's new line — that the parse-root/`FILENAME` issue, not
  missing test sources, was the actual root cause of an earlier useless `cpg_falkorchat` build — is
  corroborated verbatim by pre-existing `docs/HISTORY.md` ("That, not the missing test sources, is
  why the pre-rebuild graph was useless") and the existing `docs/BACKLOG.md` C-312 entry. Not a
  fabricated backstory.
- **Live-verification numbers.** 1067 (happy path, `tests/`) / 0 (failure path, `nonexistent/`) are
  corroborated by `claude/graph-dba/kaizen/inbox.md`'s 2026-08-09 correction entry, which records
  the same retry and the same two counts after a container restart — consistent, not just asserted.
- **Minor (optional, not blocking):** `PREFIX` is interpolated unescaped into the Cypher string
  literal (`"MATCH (m:METHOD) WHERE m.FILENAME STARTS WITH \"$PREFIX\" ..."`). A prefix containing
  a `"` would break the query rather than fail cleanly. Low-severity — this is a local
  developer-supplied CLI argument, not attacker-controlled input, and the rest of the script has
  the same trust level for `$GRAPH`/`$SRC`. Worth a one-line comment if anyone hardens this file
  later; not worth blocking C-312 on.

## U6 — C-319: `.mcp.json` approval-scoping bullet (`skills/agent-standards/claude-code.md`) — self-review

Per the brief: this is my own earlier work in this session, reviewed with the same scrutiny I'd
apply to another producer's.

**Distillation bookkeeping — correct.** Checked against `agent-maintenance` skill §5
(verify → route → log → clear):
- **Verify:** `claude/cobb/kaizen/history.md`'s entry states the `~/.claude.json` `projects` map
  was re-checked live before promotion, and is honest that the original `claude mcp list`
  root-vs-subdirectory contrast was *not* re-derived (infra was unreachable this run) — cited from
  the inbox's original evidence instead of fabricating a fresh repro. That's the right call:
  partial re-verification, disclosed as partial, beats a silent full skip or a faked full repro.
- **Route:** into `skills/agent-standards/claude-code.md` §MCP → "Scopes, precedence, and the
  approval gate" — the right destination (on-demand standards doc other agents consult before
  authoring `.mcp.json`/approval-related work), matching the section's existing style (the
  "no per-server tool filter" bullet is likewise sourced from live testing, not the official docs).
- **Log:** dated entries added to both `claude/cobb/kaizen/history.md` and
  `claude/devops/kaizen/history.md` (producer and destination-owner sides both recorded) — correct
  per §5 point 4.
- **Clear:** the 2026-07-25 entry is removed from `claude/devops/kaizen/inbox.md` in the same diff.

**Finding — major: the new bullet asserts an unverified causal mechanism.**

The promoted text reads:

> "...this is approval *scoping*, distinct from the discovery mechanism, which stays uniform via
> `$CLAUDE_PROJECT_DIR` (see below)."

I fetched `code.claude.com/docs/en/mcp` directly to check this. The docs describe
`CLAUDE_PROJECT_DIR` exclusively as an environment variable Claude Code sets **in a spawned MCP
server's process environment** (and available for `${...}` expansion inside `command`/`args`
strings) — "Claude Code sets `CLAUDE_PROJECT_DIR` in the spawned server's environment to the
project root, so your server can resolve project-relative paths without depending on the working
directory." Nothing in the docs, and nothing in the original 2026-07-25 inbox evidence (which only
observed that the server was still *discovered* from a subdirectory — it never invoked or
inspected `CLAUDE_PROJECT_DIR`), establishes that `.mcp.json` *file discovery* is implemented via
that same env-var mechanism. These are two independently true, both cwd-independent facts —
discovery walks up to project root; `${CLAUDE_PROJECT_DIR}` expansion doesn't depend on cwd
either — but "stays uniform **via** `$CLAUDE_PROJECT_DIR`" asserts they're the *same* mechanism,
which is unverified and, per the docs excerpt above, not how `CLAUDE_PROJECT_DIR` is documented to
work (it's about server-env/path-expansion, not about how Claude Code locates the `.mcp.json`
file on disk).

Tracing it back: `docs/BACKLOG.md`'s C-319 entry (the filing that predates this promotion) already
carries the same two facts as **parallel, not causal**, statements — "`.mcp.json` discovery walks
up to the git root... the `$CLAUDE_PROJECT_DIR` form is otherwise cwd-independent" (two clauses,
two subjects, joined by "and... otherwise", not "via"). The promoted bullet compressed that
parallel structure into a single causal claim during promotion, which is where the inaccuracy was
introduced — not in the original C-319 filing or the original inbox entry, both of which keep the
two facts separate.

**Suggested fix** (small, contained edit to `skills/agent-standards/claude-code.md` §MCP): replace

```
this is approval *scoping*, distinct from the discovery mechanism, which stays uniform via
$CLAUDE_PROJECT_DIR (see below).
```

with something that doesn't claim the shared mechanism, e.g.:

```
this is approval *scoping*, distinct from the discovery mechanism, which is cwd-independent for a
different reason (the `${CLAUDE_PROJECT_DIR}` path-expansion form used inside `.mcp.json` entries
is also cwd-independent, but via the server-launch env var described below — not evidence the two
share an implementation).
```

or, more simply, just drop the "(see below)" claim entirely and let the two facts stand
unconnected, matching how the backlog entry that sourced this originally kept them.

**Update 2026-08-09, post-review:** `teco` judged this a genuinely trivial, docs-only,
factual-accuracy fix with no design stakes and asked me to apply my own suggested rewrite directly
rather than route it through a separate review loop. Done — the clause now states the two
cwd-independent facts as parallel/separately-caused, matching `docs/BACKLOG.md`'s original C-319
phrasing. See `claude/cobb/kaizen/history.md`, 2026-08-09 follow-up entry.

**Everything else in the bullet checks out**: the `claude mcp list` root-vs-`falkor-chat/`
contrast, the `~/.claude.json` `projects`-map-is-per-launch-cwd explanation, and the "one extra
approval per subdirectory" consequence are all faithful restatements of the original inbox
evidence with no drift.

## Overall verdict: **approve with suggestions**

U1 (C-308) and U2 (C-312) are clean — no findings, either severity. U6 (C-319) had the bookkeeping
right (verify/route/log/clear all correctly performed) but the promoted text itself introduced one
unverified causal claim not present in either the original inbox entry or the backlog filing that
scoped the work. Severity **major** because it stated a specific, checkable "why" as fact in a
reference doc other agents will cite verbatim — but narrow in blast radius (one clause, didn't
affect the bullet's actionable claims about approval scoping, which were all correct) and cheap to
fix. **Fixed 2026-08-09** (see update above) on `teco`'s trivial-fix exception call. Nothing here
blocked C-308 or C-312.
