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
  gitignored path (`cypher-mcp/.venv/...`, matched by the root `.gitignore`) is correctly *not*
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

## Pass 2 — 2026-08-08, re-review of the left-anchor follow-up

**Scope.** Re-review of `cobb`'s small follow-up on top of commit `6ab4ffe` (the Pass 1 diff,
already approved above), acting on Pass 1's minor finding #1: `claude/scripts/guard-destructive-ops.sh`
gained a new `LB='(^|[^[:alnum:]])'` left-boundary variable applied to the `pipeline.sh` basename
in the C-311 branch. Also re-checked the accompanying `docs/BACKLOG.md` C-311 follow-up note and
C-312 owner correction (`joern` → `graph-dba`, the other Pass 1 minor finding), and the new
`docs/HISTORY.md` entry. Delta-sized review as requested — I did not re-walk C-309 or the
pre-existing destructive-ops branches from scratch, only re-verified anything the diff could have
touched or that this pass's own findings put in question.

**Verdict: needs changes.**

### Findings

**Major — the left-anchor tightening introduces a genuine regression: `--reset pipeline.sh`
(bare basename, `--reset` first, single separator) no longer matches, where it matched under the
Pass-1 code (commit `6ab4ffe`) that this review already approved.** Root cause, confirmed by
reading the compiled pattern and reproducing character-by-character: the second alternation arm
is `--reset${B}.*${LB}pipeline\.sh${B}`. `B` and `LB` each require *consuming* one real
character (or, only at true start/end of string, matching zero-width). When `--reset` and
`pipeline.sh` are separated by exactly one space and nothing else, that single space is the only
character available to satisfy *both* `B` (the right boundary after `--reset`) and `LB` (the left
boundary before `pipeline.sh`) — one consumes it, leaving nothing for the other, and the
alternative fails to match. Any additional separator character (a path segment, a `./`, two+
spaces before `tr -s ' '` collapses them back to one — doesn't help) supplies the second boundary
character and the match succeeds again, which is why the path-prefixed and `bash`/`sh`-prefixed
cases in `cobb`'s own test matrix all pass while the bare-basename case silently doesn't.

Reproduced directly against the real script (not a paraphrase — `bash "$GUARD"` end to end, JSON
in, JSON/exit-code out; see note on tooling below):

```
$ printf '{"tool_input":{"command":"--reset pipeline.sh"}}' | bash claude/scripts/guard-destructive-ops.sh test-agent
(no output)
$ echo $?
0
```

vs. the same payload against Pass 1's code (`git show 6ab4ffe:claude/scripts/guard-destructive-ops.sh`):

```
$ printf '{"tool_input":{"command":"--reset pipeline.sh"}}' | bash /tmp/.../guard-pass1.sh test-agent
{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"ask",...}}
```

This directly contradicts a specific, written verification claim in this diff's own
`docs/BACKLOG.md` C-311 follow-up and `docs/HISTORY.md` entry: both assert "every realistic
invocation shape (full repo-root path, `bash`/`sh`-prefixed, SKILL.md's documented cwd-relative
form, bare basename, absolute path, `--reset` before or after the path) still asks." The
bare-basename × reset-before-path cell of that matrix does not still ask — I tested it, twice,
independently of `cobb`'s own claim.

**Mitigating context (why this is major, not blocker):** `pipeline.sh` must be the executable
token (or immediately follow an interpreter/path prefix) for the shell to actually run it, and
`--reset` is only ever passed as *its* argument — meaning in any single, genuine Bash invocation
that actually executes `pipeline.sh --reset`, the text `pipeline.sh` necessarily precedes
`--reset`, never the reverse. I could not construct a realistic single-command scenario where a
real, effective `--reset` invocation produces `--reset` immediately before a bare `pipeline.sh` in
the command text — the "reversed order" alternative appears to have been defensive/redundant for
genuine invocations even in Pass 1 (it exists to catch pathological or contrived command
constructions, not the documented usage shape from `skills/joern-cpg/SKILL.md:104`, which is
always `pipeline.sh --reset`, script-first). So this does not appear to reopen the actual C-311
exploit path for a real agent-issued command. What it *does* do is falsify a specific claim the
diff makes about its own verification coverage — a documentation-accuracy defect in a document
whose entire value is being a trustworthy, checked record, on top of a real (if apparently
unreachable-in-practice) regex defect. Given the fix is small and low-risk, there's no reason to
carry a known-inaccurate claim forward.

**Suggested fix:** decouple the two boundary checks instead of making them share one separator.
Two independent, properly-bounded `grep` conditions combined with a shell `&&` sidesteps the
character-sharing bug entirely and is simpler than the single intertwined alternation:

```bash
elif { printf '%s' "$norm" | grep -qiE -- "${LB}pipeline\.sh${B}"; } && \
     { printf '%s' "$norm" | grep -qiE -- "--reset${B}"; }; then
```

Verified this candidate against the full matrix (bug case, documented order, path-prefixed reset-
before, the `mypipeline.sh` false-positive Pass 1 flagged, no-reset, and a `--resetting-option`
superstring guard): matches exactly the cases that should match, rejects exactly the cases that
shouldn't, including the one Pass 1's code got right and Pass 2 broke. Whichever fix `cobb` picks,
`docs/BACKLOG.md`'s and `docs/HISTORY.md`'s "still asks" claim needs to either become true again
or be corrected to state the actual, narrower coverage.

**On the false-positive side (Pass 1's original ask) — confirmed fixed, no new issues.**
Re-ran the concrete false positive Pass 1 found:

```
"mypipeline.sh --reset"   -> PASS (no longer matches — correct, this was the point of the change)
```

and re-confirmed the pre-existing branches (`GRAPH.DELETE`, `FLUSHALL`/`FLUSHDB`, `docker rm -f`,
volume/compose wipes), the documented stdin/stdout contract (malformed stdin, benign command —
both still silent `exit 0`), and the `pipeline.sh`-without-`--reset` and prose-mention cases all
behave exactly as before — no regression on anything *other than* the one case above.

**Minor — the C-312 owner correction is accurate.** `Owner: graph-dba (corrected 2026-08-08 —
the joern agent was retired into graph-dba, commit cbf26c4, ...)` matches `cbf26c4`'s actual
commit message ("retire joern agent, fold CPG generation into graph-dba") — same fact Pass 1
already verified independently. No new check needed beyond confirming the wording lands correctly
in the diff, which it does.

**A note on tooling, for whoever re-runs these tests:** my first attempt at reproducing the bug
by piping a raw string straight into `grep -qiE` at my shell's top level gave a false "still
matches" result — this environment's interactive shell shadows `grep` with a wrapper function
that execs `ugrep` (a Claude-Code-provided convenience layer), which does not reproduce plain GNU
`grep`'s ERE semantics for this pattern. The guard script itself is unaffected (it runs as a
freshly spawned `bash <script>` subprocess, which does not inherit that shell function), and
every result reported above and in Pass 1 went through the actual script via `bash "$GUARD"` /
`printf ... | bash "$GUARD"`, not a bare `grep` call in my shell — but this is exactly the kind of
silent false-confirmation trap the "run it, don't just read it" discipline exists to catch, so
it's worth flagging for the next person testing this file's regex changes in this environment:
verify through the actual script invocation, not a standalone `grep` typed at this shell's prompt.

### Revised verdict

**Needs changes**, narrowly: fix (or otherwise account for) the `--reset`-before-bare-`pipeline.sh`
regression, and correct the "still asks" claim in `docs/BACKLOG.md`/`docs/HISTORY.md` to match
whatever the code actually does once fixed. Everything else in this follow-up — the false-positive
fix itself, the C-312 owner correction, the doc mechanics — is solid and needs no further changes.

## Pass 3 — 2026-08-08, confirmation of the structural fix

**Scope.** Targeted confirmation, not a re-review, of `cobb`'s fix for Pass 2's regression.
`claude/scripts/guard-destructive-ops.sh`'s C-311 branch changed from one intertwined alternation
to two independent `grep -qiE` checks ANDed together (`${LB}pipeline\.sh${B}` and
`${LB}--reset${B}`, each run against the full `$norm` string on its own, no shared `.*` between
them). Re-checked the accompanying `docs/BACKLOG.md`/`docs/HISTORY.md` follow-up prose for
accuracy against the new code, and glanced at the new `claude/cobb/TESTING.md` row + "Gotcha"
subsection (requested as a nice-to-check, not blocking).

**Verdict: approve.**

### What I verified

All of the following ran through the actual script (`bash claude/scripts/guard-destructive-ops.sh`,
piping synthetic `{"tool_input":{"command":"..."}}` payloads on stdin and reading
`permissionDecision`/exit code) — never a standalone `grep -qiE` typed at this shell's own prompt,
per the gotcha both Pass 2 and `cobb`'s new `TESTING.md` entry now flag.

1. **Pass 2's regression case, fixed.** `--reset pipeline.sh` (bare basename, flag first, single
   separator) now correctly asks — the exact case that silently passed clean under the Pass 2 code.
2. **Full matrix holds, order no longer matters at all** (checked both directions for every shape,
   not just the ones in the brief): `pipeline.sh --reset`, `--reset pipeline.sh`,
   `scripts/pipeline.sh --reset`, `--reset scripts/pipeline.sh`, `./pipeline.sh --reset`,
   `--reset ./pipeline.sh`, `bash skills/joern-cpg/scripts/pipeline.sh --reset`,
   `sh skills/joern-cpg/scripts/pipeline.sh --reset`, an absolute path, `--reset` before the full
   repo-root path, and extra flags interposed on either side — every one of these asks.
3. **Pass 1's original false positive stays fixed.** `mypipeline.sh --reset` and a second
   unanchored variant (`xpipeline.sh --reset`) both pass through clean. Also checked the new
   `${LB}--reset${B}` half doesn't introduce its own analogous false positive on the flag side:
   `pipeline.sh --resetting-option` and `pipeline.sh myreset=1` both correctly pass clean (the
   left/right boundary around `--reset` itself is unchanged from the pre-existing branches'
   pattern, just reused).
4. **No regression on any pre-existing branch or the contract.** `GRAPH.DELETE`, `FLUSHALL`,
   `FLUSHDB`, `docker rm -f`, `docker volume prune`, `docker system prune`,
   `docker-compose down -v`, `docker compose down --volumes` all still ask; a benign command and
   `pipeline.sh` without `--reset` still pass clean; malformed non-JSON stdin and a clean command
   both still exit 0 with no output, and a real hit still emits the documented
   `permissionDecision: ask` JSON shape with the reason string intact.
5. **Docs no longer carry the falsified claim, and now describe the real mechanism.**
   `docs/BACKLOG.md`'s C-311 entry gained a "Pass-2 correction, same day" paragraph that states
   the actual bug (shared-separator boundary-group collision), rates it correctly (major, not
   blocker, with the same "no realistic single command reverses the tokens" reasoning this review
   gave), and describes the real fix (two independent ANDed greps) — no residual "before or after
   the path — still asks" language stated as an unqualified fact; where it repeats the phrase, it's
   inside the historical account of what Pass 1 originally (incorrectly) claimed, clearly marked
   as superseded by the Pass-2 correction that follows it. `docs/HISTORY.md` carries the matching
   account. Both read as accurate against the diff, not just internally consistent with each other.

**One genuinely new (and correct, not a defect) behavior worth naming, not flagging:** because the
two checks are now fully independent existence tests rather than one ordered `.*`-joined
alternation, the guard will also ask on a command where `pipeline.sh` and `--reset` both appear
anywhere in the (post-newline-flattening) text but belong to unrelated parts of a compound
command — e.g. `echo pipeline.sh; echo --reset` asks. This is a strict superset of Pass 1's
already-permissive "either order, arbitrary distance in one direction" design, it fails in the
guard's documented safe direction (more asking, not less), and it's consistent with the
pre-existing `GRAPH.DELETE`/`FLUSHALL` branches' own substring-anywhere behavior — not a new
finding, just confirming the widened surface is intentional and benign, not overlooked.

**`claude/cobb/TESTING.md` addition:** read the new row and "Gotcha" subsection; it states the
`ugrep`-shadowing fact accurately (function-not-exported, so a spawned `bash script.sh` subprocess
is unaffected; the two can disagree silently) and gives the correct reliable-verification recipe
(pipe synthetic JSON through the actual script, never a bare `grep` at the prompt) — matches what
I independently found and reported in Pass 2, including the "verify it still holds via `type grep`
vs. `bash -c 'type grep'`" caveat, which is exactly the check I used to diagnose it. No corrections
needed.

### Final verdict

**Approve.** The Pass 2 regression is structurally closed (independent existence checks can't
recreate the shared-separator bug the ordered alternation had), the original Pass 1 false positive
stays fixed, no pre-existing branch or the hook's stdin/stdout contract regressed, and the
`docs/BACKLOG.md`/`docs/HISTORY.md` accounts are accurate and no longer assert the disproven claim.
Clear to commit.
