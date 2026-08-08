# Safety-net script fixes (audit-team.sh, guard-destructive-ops.sh) — Review

> **Status:** active · **Owner:** `analyst` · **Tracks:** C-309, C-311

## Scope & verdict

Independent review of `cobb`'s uncommitted working-tree diff closing **C-309** (`audit-team.sh`
check 7 blind to untracked files, plus a bookkeeping claim that 5 pre-existing PII leaks are
already gone) and **C-311** (`guard-destructive-ops.sh` blind to `pipeline.sh --reset`, a
destructive op wrapped inside a script). Files reviewed: `claude/scripts/audit-team.sh`,
`claude/scripts/guard-destructive-ops.sh`, `claude/AGENTS.md` (Hook machinery paragraph),
`docs/BACKLOG.md` (C-309/C-311 entries), `docs/HISTORY.md` (2026-08-08 entry). Not reviewed:
`claude/cobb/kaizen/history.md` (also in the diff but outside the brief) — skimmed only for
consistency, not evaluated against the kaizen-entry convention.

Verification method: read every changed line, plus enough surrounding script to judge fit; ran
`claude/scripts/audit-team.sh` against the live tree; independently grepped the 5 paths C-309(a)
claims are clean; planted and removed a throwaway untracked leak file to exercise the new
untracked-file scan path (both the catch and the gitignore-exclusion); drove
`guard-destructive-ops.sh` directly with ~25 synthetic PreToolUse JSON payloads covering the new
branch, all pre-existing branches, and the documented stdin/stdout contract; independently
grepped `skills/*/scripts/` to check the "only wrapper" claim; checked `cbf26c4` (the commit
C-309(a) cites) actually did what's claimed. No test artifacts were left behind.

**Verdict: approve.**

## Findings

No blockers, no majors. Two minor observations, both about scoping precision the change's own
comments already flag as a known trade-off — neither is a regression and neither creates a
silent-bypass risk (the guard's failure mode here is over-asking, not under-asking).

**Minor — the C-311 regex is unanchored on the left, so it matches on `pipeline.sh` as a bare
substring, not on the specific `skills/joern-cpg/scripts/pipeline.sh` path or on an actual
invocation.** Confirmed live:

```
"mypipeline.sh --reset"                                          -> ASK (matches)
"echo 'see pipeline.sh --reset in docs'"                          -> ASK (matches)
"grep -n -- --reset skills/joern-cpg/scripts/pipeline.sh"         -> ASK (matches)
```

A future script merely *named* `*pipeline.sh` with an unrelated `--reset` flag, or an agent
quoting/grepping the known invocation in prose or another command's arguments, will also trip
the guard. This is consistent with the pre-existing branches' design — I confirmed
`"echo 'do not run FLUSHALL in prod'"` and `"grep -rn GRAPH.DELETE skills/"` already exhibit the
identical substring-match behavior against the code *before* this diff, so it is not a
regression and not something this change should have fixed on its own initiative. It's worth
flagging because the new branch's own comment describes the scoping as "this repo has exactly
one such wrapper today" — true of the *target script*, but the regex is looser than that
sentence implies. Suggested improvement (non-blocking, can ride with the "if a second wrapper
appears" trigger the comment already names): anchor on the actual path
(`skills/joern-cpg/scripts/pipeline\.sh`) rather than the bare basename, which would also close
the `mypipeline.sh` false-positive without touching the prose/quoting false-positive (which is
inherent to command-text pattern matching and shared by every branch, old and new).

**Minor — `docs/BACKLOG.md`'s C-312 entry (`Owner: joern`) is now doubly stale** (the `joern`
agent was retired into `graph-dba` in commit `cbf26c4`, per C-309(a)'s own citation in this same
diff) — pre-existing, untouched by this change, out of scope for this review, noted only because
this diff's own C-309(a) writeup surfaces the fact that would fix it. Not a finding against
`cobb`'s diff.

## What's solid

- **C-309(a) bookkeeping claim verified independently and confirmed accurate.** Grepped all five
  cited paths (`.claude/settings.json`, `claude/devops/kaizen/inbox.md`,
  `docs/plans/m2-cpg-analysis-skill.md`, `falkor-chat/docs/requirements/workflow-dependence-overlay.md`)
  directly for `$HOME`, username, hostname, git user.name, git user.email — all clean.
  `claude/joern/kaizen/inbox.md` confirmed absent (`git log -1 cbf26c4` shows the commit really
  is "retire joern agent, fold CPG generation into graph-dba," and its message explicitly says it
  "fixes the personal-info-leak findings audit-team.sh was flagging repo-wide" — corroborating,
  not just self-reported). A full `audit-team.sh` run against the live tree returns
  `RESULT: PASS` on all 8 checks, no code change needed for (a) as claimed.
- **C-309(b) fix works exactly as documented, both directions.** Planted an untracked file under
  `claude/` containing `$HOME`; the gate FAILed on both the "home path" and "username" labels
  with `exit 1` and the offending `file:line:content` printed; removed the file; the gate
  returned to a clean `RESULT: PASS` with `exit 0`. Also confirmed a leak placed inside a
  gitignored path (`cpg/mcp/.venv/...`, matched by the root `.gitignore`) is correctly *not*
  caught — the union of `--cached` + `--others --exclude-standard` behaves as intended, not as an
  unbounded "every file on disk" scan. Verified `xargs -0 -r grep` degrades safely on an empty
  file list (exit 0, no output) and `git ls-files -z --cached --others --exclude-standard`
  produces no duplicates for a file that could theoretically appear in both sets. The switch from
  grep-exit-code to output-emptiness (`[ -n "$hits" ]`) is the right call given the pipeline now
  has an `xargs` stage in the middle — it removes a genuine ambiguity (whose exit code would
  `hits=$(... | xargs ... | ...)` even report?) rather than papering over one.
- **C-311 fix verified against ~15 synthetic invocations.** Both `--reset` token orderings ask
  (with and without a path prefix, with extra flags interposed); `pipeline.sh` without `--reset`
  passes clean; all four pre-existing branches (`GRAPH.DELETE`, `FLUSHALL`/`FLUSHDB`,
  `docker rm -f`, `docker volume prune`/`system prune`, `compose down -v/--volumes`) still fire
  correctly — no regression from the new `elif` insertion point. The documented stdin/stdout
  contract (JSON on stdin, `permissionDecision: ask` JSON on a hit, silent `exit 0` otherwise, and
  `exit 0` even on malformed/non-JSON stdin per the fail-open design) is unchanged — verified with
  a raw-non-JSON stdin payload and a clean-command payload, both still `exit 0`.
- **The "only wrapper" scoping claim holds up under an independent grep.** `grep -rln -- '--reset\|GRAPH.DELETE\|...' skills/*/scripts/`
  turns up `pipeline.sh` (the real wrapper) and `cpg-to-falkordb.py` — but the latter's hits are
  a docstring and an error message *telling the operator* to run `redis-cli GRAPH.DELETE`
  manually (and its own comment notes that manual invocation "is caught by joern's
  destructive-ops guard"), not an internal invocation. `pipeline.sh` itself does call
  `redis-cli ... GRAPH.DELETE "$GRAPH"` under its `--reset` branch (confirmed by reading it), so
  the claim that it's the one genuine wrapper today is accurate, not merely asserted.
- **Docs are internally consistent.** `claude/AGENTS.md`'s "Hook machinery" addition accurately
  describes what the new branch does (the "literal string never reaches the guard" framing
  matches the actual mechanism). `docs/BACKLOG.md`'s C-309/C-311 entries follow this file's
  established resolved-item convention (✅ marker, bolded "Resolved \<date\> by \<owner\>" lead-in,
  `Owner:` trailer) — matching the shape of neighboring resolved entries (C-200, C-320, C-322),
  not just C-313's differently-structured narrative resolution. `docs/HISTORY.md` got a dated,
  most-recent-first entry per the module-documentation convention, with accurate before/after
  framing and no overclaiming (it correctly scopes itself to "no other script or hook contract
  changed").

## Open questions

None — the diff is small, self-contained, and every claim in cobb's report checked out under
direct verification. The two minor findings above are optional hardening, not blockers; whether
to fold the path-anchoring suggestion into this change now or defer it to whenever "a second
wrapper appears" (the trigger the code comment itself already names) is a judgment call for
`cobb`/`teco`, not something this review needs resolved before approving.
