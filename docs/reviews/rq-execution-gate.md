# `rq()` execution gate — review of the K-009 fix

> **Status:** active · **Owner:** `analyst` · **Tracks:** K-009 (U28)

## Scope & verdict

Diff-scoped review of an **uncommitted** change by `graph-dba`, working tree against `HEAD`
(`c0f7f6c`), across seven files: `skills/joern-cpg/scripts/pipeline.sh` (substantive),
`skills/joern-cpg/scripts/test-stamp-wiring.sh` (regression cover),
`skills/joern-cpg/SKILL.md`, `skills/cpg-analysis/references/freshness.md`,
`claude/graph-dba/falkordb-quirks.md`, `claude/graph-dba/kaizen/{plan,history}.md`.
`claude/docs/plans/kaizen-distillation2-coordination.md` and everything under `model-bench/`
were excluded by the brief and not read as part of this change.

**Verdict: approve with suggestions.** No blockers. The one claim the caller could not check —
that `GRAPH.QUERY` **write** replies carry the statistics trailer, which is the entire
justification for making the gate unconditional — I **confirmed by execution**, and over a wider
set of shapes than were originally measured. The two findings below are about what the change's
own prose *claims*, and about a new tri-state return that no call site reads; neither makes the
shipped code wrong.

**CPG: considered, not relevant — the change is Bash and Markdown; `cpg_falkorchat` (the nearest
loaded graph) contains 0 `FILE` nodes whose `NAME` matches `.sh` (queried this pass), so no
call-graph or impact question here is answerable from it.**

Everything labelled **[executed]** below I ran myself this pass. Everything labelled **[static]**
is reasoning over the code I read without running it.

Graph hygiene: I created two throwaway keys (`analyst_u28_trailer`, `analyst_u28_b`) plus one
plain string key (`analyst_u28_str`) and deleted all three. `GRAPH.LIST` is back to its
starting **25** keys; no `analyst_*` or `u28` key remains. `kaizen_team` was read-only
(`GRAPH.RO_QUERY`-equivalent MCP reads); no `cpg_*`, `ws:*`, `reference` or `test` key was written.

---

## Findings

### Major 1 — the comment claims the gate excludes a part-way reply; a server-truncated reply passes it. [executed]

`skills/joern-cpg/scripts/pipeline.sh:275-280` states the gate is *"anchored on the last line, so
a reply cut off part-way cannot satisfy it"* and lists **mid-stream abort** among what it covers.
That is true for a mid-stream *runtime error* — and false for the other truncation this instance
actually performs. `GRAPH.CONFIG GET RESULTSET_SIZE` is **10000** here: a query returning more
rows is silently capped and still emits a normal trailer. Measured against 200,000 nodes, the
reply is 20,004 lines — header, 10,000 rows, `Cached execution`, trailer — rc 0, gate **passes**
(Appendix A2). The rows were cut off part-way; nothing in the reply says so.

Zero impact on the three current call sites (1 row, ≤8 rows), but the comment is the thing a
future call-site author reads before adding a query that could return more. The same
generalisation stands in two other places this change touched or relies on:
`claude/graph-dba/falkordb-quirks.md:791` (*"there is no partial reply to worry about"* — scoped
by its own lead sentence to mid-stream errors, so defensible, but the bolded clause reads
absolute) and kaizen entry `4f9c21ae…`, whose fact ends *"a partial-reply worry is not a real
failure mode"* — unscoped, and falsified in the general form.

**Suggested improvement** (`graph-dba`): in `pipeline.sh`, replace *"a reply cut off part-way
cannot satisfy it"* with the narrower true statement — *a reply aborted mid-stream by a runtime
error carries no trailer at all, so it fails closed* — and add one clause naming what the trailer
**cannot** see: `RESULTSET_SIZE` (10000 on this instance) truncates a large result set silently
and still trails. Mirror the scoping in `falkordb-quirks.md`. This matters more than a wording nit
because "a credential covering a narrower level than the claim it licenses" is the recurring
defect this whole arc has been closing.

### Major 2 — rc 2 is a new tri-state no caller reads, it is untested, and at the stamp call site it produces exactly the misdiagnosis this change just fixed elsewhere. [executed]

`pipeline.sh:324-330` adds `return 2` for a non-`GRAPH.QUERY`/`GRAPH.RO_QUERY` command. All three
call sites (`:388`, `:413`, `:476`) test `if ! VAR="$(rq …)"`, which cannot tell 2 from 1, and the
refusal writes only to stderr — so `$VAR` is empty and the caller renders `<no reply>`.

Routing the stamp write through `GRAPH.DELETE` in a byte-copy mutant produces:

```
pipeline: FAILED — internal: rq() was called with 'GRAPH.DELETE'. …
pipeline: FAILED — FalkorDB rejected the freshness stamp for 'cpg_fake':
pipeline:   <no reply>
```

FalkorDB never saw the stamp. That is a claim about the server from a call that never reached it —
structurally identical to the read-back defect the second half of this change exists to fix, two
lines below rq's own correct "internal" message. (The read-back and stray sites degrade
gracefully: their wording — "could not read the stamp back", "could not verify the marker's
property list" — happens to stay true under rc 2.)

Separately, the guard has **no test at all**: deleting the entire `case "$cmd"` block from a copy
and running the suite gives 14 PASS / 0 FAIL / exit 0, unchanged (Appendix A3). And
`skills/joern-cpg/SKILL.md:124` now says *"six failure branches … split three ways"* — correct for
the branches it enumerates, but rc 2 is a seventh exit path it does not mention.

**Suggested improvement** (`graph-dba`): capture `rc=$?` at the three call sites and route `rc -eq
2` to a single internal-misuse branch that exits without asserting anything about the graph — note
that `exit 2` *inside* `rq` will not work, because `rq` runs in the `$(…)` subshell and only the
substitution would exit. Add a direct-call test for the guard using the pattern already in
`test-stamp-wiring.sh` for the stray-query guard ("P6-5(a)", `:270-290`), which exists for the same
reason: an unreachable-through-the-block guard is otherwise free to be deleted.

### Minor 3 — one of the new cases' three assertions does not discriminate the fix. [executed]

`test-stamp-wiring.sh:269-270`, case `stamp rejected, no error prefix`, asserts three strings. The
third, `"does NOT need repeating — only the stamp does"`, is printed by `replay_stamp`, which the
**pre-fix** code also reaches via the read-back branch — so the mutation run's `output lacks:` list
names only the other two (Appendix A1). The assertion is true and harmless; it just carries none of
the case's discriminating power, and reading the case one might think it does.

**Suggested improvement:** either drop it, or add a comment saying the first two are the pinning
ones. Low stakes.

### Minor 4 — `git-provenance.sh:105` citation is off by two lines and quotes a paraphrase. [executed]

`claude/graph-dba/kaizen/history.md` (K-008, Fact 2) writes:
`` `git-provenance.sh:105` is `git -C "$dir" status --porcelain -- ":(literal)$name"` ``. Line 105
is the `CPG_SOURCE_TREE` assignment; the status call is **line 107** and reads
`git -C "$dir" status --porcelain -- "$spec"`, with `spec=":(literal)$name"` set at line 76. The
substance is correct — the pathspec is scoped, and I confirmed the file — but the quoted text is a
substitution presented as verbatim.

**Suggested improvement:** cite `git-provenance.sh:76,107` and quote the two lines as they read.

### Minor 5 — both docs' enumeration of test cases omits one case (pre-existing). [executed]

`skills/joern-cpg/SKILL.md:149-158` and `skills/cpg-analysis/references/freshness.md:277-284` each
list the suite's cases and each names **nine**; there are **ten** `run_case` invocations. The
missing one is `regression: merge, pipeline-clean marker` — a control that passes. This omission
predates the change (it is in the diff's context lines, not its additions), but both paragraphs are
open in this diff, so it is nearly free to close.

### Nit 6 — one over-long line introduced in `freshness.md`. [executed]

`skills/cpg-analysis/references/freshness.md:185` is 105 characters; the file's prose otherwise
wraps near 78 (its only other >100 lines are table rows and a pre-existing 108). Introduced by the
`) **A hand-authored marker is subject` join in this diff. Re-wrap.

---

## Question 5 — the two held-open `kaizen_team` entries

I read both from the graph read-only.

**`b7f3c2a1-9d4e-4c11-8a52-6e0f1d3b7c94` (prefix taxonomy) — clear it.** Fully addressed. Its
content is promoted verbatim in substance into `claude/graph-dba/falkordb-quirks.md:753-765` (the
paired-control paragraph: `errMsg:` parse errors and `ERR`/`WRONGTYPE` caught, `Unknown function`
and `Type mismatch` bare and missed, redis-cli exit 0 throughout), and the code defect it was filed
against is fixed and regression-covered. I re-executed both missed shapes this pass and both now
fail the gate.

**`4f9c21ae-7b30-4d62-9c18-6ea5d0b73c41` (mid-stream abort) — hold until Major 1's wording is
narrowed, then clear.** The load-bearing half is fully addressed and I re-verified it: a
mid-stream `Division by zero`, `Query timed out`, and the read-only refusal all come back as one
bare line, rc 0, no trailer. But the entry's fact ends *"a partial-reply worry is not a real
failure mode"* — a general claim I falsified by execution (Major 1). Clearing is irreversible, and
the promoted text inherits that generalisation, so I would land the scoping edit first. If the
curator prefers not to block on it, clearing is still defensible provided Major 1 is filed as its
own follow-up — the entry's evidence is sound, only its concluding generalisation is over-broad.

---

## What's solid

- **The unconditional gate is justified. [executed]** Writes carry the trailer on every shape I
  could construct: the creating `MERGE … SET b = {…} RETURN 1`, the re-run where the `MERGE`
  matched, a plain `CREATE`, index creation, a write with an explicit `TIMEOUT`, a write returning
  zero rows, and — the sharpest case — a `SET` whose `MATCH` matched nothing, whose entire reply is
  `Cached execution: 0` + the trailer and nothing else. Also `CALL db.labels()`,
  `CALL db.indexes()`, `CALL dbms.procedures()`, `LIMIT 0`, and `EXPLAIN`. No false negative found.
  `graph-dba`'s claim holds and then some.
- **Both reasons given for last-line anchoring over a whole-reply substring are real. [executed]**
  `RETURN "Query internal execution time: fake"` puts the literal in a data row and the gate still
  discriminates correctly; `Query timed out` shares the first word with the trailer.
- **The non-query refusal is correctly motivated. [executed]** `GRAPH.DELETE` answers a bare `OK` —
  I observed it deleting my own throwaway keys.
- **Question 3's subtle claim is exactly right. [executed]** New suite against a byte-copy of
  `HEAD`'s `pipeline.sh`: 12 PASS, 2 FAIL, exit 1. The failure detail contains **only**
  `output lacks:` lines — no `rc:` line, no `outcome:` line, no missing-`--- begin stamp ---` line.
  Both cases really are judged on branch wording alone, because both already exited 1 via a later
  assertion, and both really did misdiagnose: the pre-fix run reports *"the freshness stamp did not
  land in 'cpg_fake' … read back: Unknown function 'nosuchfunc'"* (Appendix A1).
- **Suite and arithmetic. [executed]** Working tree: exit 0, 14 `PASS`, 0 `FAIL`. The reconciliation
  in `history.md` checks out: 8 `run_case` at `d43ca40` + 2 new = 10, plus 2 from the stray-query
  guard's two-shape loop, plus P5-1 and P6-6 = 14.
- **Branch accounting is correct. [static]** Six failure branches after the change (stamp rejected;
  read-back could-not-run; read-back absent; stray query unbuildable; stray read failed; stray key
  found), splitting 2 did-not-land / 3 did-land / 1 cannot-say, as `SKILL.md` says.
- **The cannot-say branch is the right shape.** Separating "checked, and it is absent" from "could
  not check" is the more valuable of the two defects fixed here, and the new message says so in
  words rather than leaving the operator to infer it.
- **No live document still carries a falsified claim. [executed]** A repo-wide grep for the
  blacklist strings and the two corrected claims returns hits only in dated history/review records
  (`docs/reviews/cpg-provenance-stamp.md`, `claude/*/kaizen/history.md`,
  `docs/plans/salesperson-ui-coordination.md`) — documents that correctly freeze — plus the
  explanatory comment in `pipeline.sh` and the historical clauses in `falkordb-quirks.md`, both of
  which now read as history and are dated as such.
- **The gate fails closed on the degenerate inputs. [executed]** Empty reply and single-line
  no-newline reply both leave `${out##*$'\n'}` un-matching the trailer, so `rq` returns 1.

## Open questions

1. **Is `RESULTSET_SIZE=10000` intentional on this instance, or inherited default?** It bounds
   every consumer of these graphs, not just `pipeline.sh`, and it is not mentioned in
   `falkordb-quirks.md`. Worth a durable note there regardless of Major 1's disposition — routes to
   `graph-dba`.
2. **Does the curator want Major 1 landed before clearing `4f9c21ae…`?** My recommendation is yes;
   the call is the curator's since the clear is irreversible.

---

## Appendix

### A1 — new suite against a byte-copy of `HEAD`'s `pipeline.sh`

```
EXIT=1   PASS=12   FAIL=2

  FAIL  stamp rejected, no error prefix
          output lacks: FalkorDB rejected the freshness stamp
          output lacks: only the provenance marker is missing
  FAIL  read-back query itself errors
          output lacks: could not read the freshness stamp back
          output lacks: says NOTHING about
          output lacks: whether the stamp landed is UNKNOWN
```

No `rc:`, `outcome:`, or `--- begin stamp ---` problem line in either case. Both pre-fix runs
printed, identically:

```
pipeline: FAILED — the freshness stamp did not land in 'cpg_fake'.
pipeline: read back: b.PARSED_AT                       (stamp_bare_error)
pipeline: read back: Unknown function 'nosuchfunc'     (readback_error)
```

### A2 — `RESULTSET_SIZE` truncation with a normal trailer

```
GRAPH.CONFIG GET RESULTSET_SIZE  -> 10000
MATCH (n:Big) RETURN count(n)    -> 200000
MATCH (n:Big) RETURN n.i, n.s    -> 388974 bytes, 20004 lines
   lines 1-2      n.i / n.s                       (header)
   lines 3-20002  10000 rows × 2 values
   line  20003    Cached execution: 1
   line  20004    Query internal execution time: 1.624008 milliseconds   <- gate PASSES
```

Killing `redis-cli` mid-reply is *not* a counterexample in the other direction: it is all-or-nothing
— `timeout -s KILL` at 0.005s/0.01s gives rc 137 with 0 bytes (gate fails closed via the `||`
branch), at 0.02s/0.04s gives rc 0 with the complete 388974-byte reply. `redis-cli` reads the whole
RESP reply before printing, so a truncated-stdout-with-rc-0 was not reachable.

### A3 — rc-2 guard is untested

Removing the entire `case "$cmd" in GRAPH.QUERY|GRAPH.RO_QUERY) ;; *) … return 2 ;; esac` block
from a copy of `pipeline.sh` and running the current suite against it: **exit 0, 14 PASS, 0 FAIL** —
byte-identical outcome to the unmutated tree.

---

## Pass 2 — 2026-09-09

Re-gate of the delta committed as `48882d8`, against Pass 1 (committed `7edf98c`). Same scope;
same evidence discipline. **[executed]** means I ran it this pass.

**Verdict: approve with suggestions.** Both Pass 1 majors are closed, and Major 2's closure is
better than what I proposed. One new major, and it is the same shape as Pass 1 Major 1 moved one
level up: the replacement guard's stated reach exceeds its mechanism. The code is safe either way —
every miss costs a false *failure*, never a false pass.

Suite reproduced independently: **exit 0, 15 PASS, 0 FAIL.** `GRAPH.LIST` back to 25 keys; I
created and deleted `analyst_u28_p2`, no residue.

### Corrections to Pass 1 — both refutations are right, both were my error

**Pass 1 Minor 4 (`git-provenance.sh` line numbers) — RETRACTED. I was wrong.** [executed]
`grep -n` puts the `status --porcelain` call at **:105** and the `CPG_SOURCE_TREE` assignment at
**:103**, on a clean file. The `history.md` citation was correct. My error was mechanical and worth
naming: I read a `sed -n 103,107p` block of five printed lines and mapped them to line numbers
while silently dropping the **blank line 104**, which shifted everything after it by two. Printed
ranges are not a line-number oracle; `grep -n` or `cat -n` is. The residual half of the finding
(the history quotes `":(literal)$name"` where the source reads `"$spec"`, with `spec` set at :76) I
also withdraw — `spec` *is* that string, so quoting the effective pathspec is a fair rendering, and
carrying a nit forward under a retracted heading would misrepresent the record.

**Pass 1 Open question 1 (`RESULTSET_SIZE` novelty) — RETRACTED. I was wrong.** [executed]
`claude/graph-dba/falkordb-quirks.md:710` carried the cap, verified 2026-07-30, *before* this
delta — including the sharper property that it defeats an explicit larger `LIMIT`. I wrote "it is
not mentioned in `falkordb-quirks.md`" having grepped that file for the blacklist strings and for
`rq` claims, and never for `RESULTSET_SIZE` itself: an assertion of absence without running the
search that would establish it. The implementer's handling was the right one — fold the genuinely
new half (*"a capped reply is structurally indistinguishable from a complete one, which bounds the
statistics-trailer discriminator below"*) onto the existing bullet rather than add a second.

Neither error changes a Pass 1 major, but both are exactly the failure shapes this coordination
keeps being bitten by, and they belong in the record as mine.

### New finding

#### Major 7 — the static call-site check states a reach its mechanism does not have; two *literal* bad commands pass it. [executed]

`test-stamp-wiring.sh:135-139` says: *"WHAT IT COVERS: a LITERAL non-query command written at any
rq call site. WHAT IT DOES NOT: a command reaching rq through a variable."* The mechanism is
`grep -n '\$(rq '` plus `grep -o 'GRAPH\.[A-Z_.]*'`. Two literal-command shapes defeat it:

- **A call site not written as `$(rq …)`.** Adding `rq 'MATCH (b) RETURN b' GRAPH.DELETE || true`
  as a bare statement: the check reports `PASS all 3 rq call sites`, suite exit 0. The new site
  doesn't match the anchor, and the three `$(rq ` sites keep the count at 3 so the `< 3` anchor
  does not fire either.
- **A lowercase command at an existing call site.** `graph.delete` in place of `GRAPH.RO_QUERY`:
  `PASS all 3`. The token grep is case-sensitive, so it finds no `GRAPH.` token on that line and
  the loop never runs. `redis-cli` accepts the lowercase form.

This is the finding the coordinator asked me to test, and the answer is that the problem moved up
one level rather than being solved: the replacement guard *can* redden (verified — see the
disposition of Major 2), but its written bound is inaccurate, in a comment whose own last sentence
is *"a guard whose stated reach exceeds its mechanism is the defect this whole block exists to
close."* Consequence is bounded — a missed misuse produces a false runtime failure, never a false
pass — which is why this is major and not a blocker.

**Suggested improvement, verified by execution rather than proposed** (`graph-dba`): widen the
anchor to `grep -nE '(^|[^_[:alnum:]])rq '` and make the token match case-insensitive
(`grep -oiE 'GRAPH\.[A-Za-z_.]*'`, upcasing before the `case`). Run against five trees this pass
(Appendix B1): clean → 3 sites, none bad; the bare-statement site → **4 sites, `GRAPH.DELETE`
caught**; lowercase → **caught**; the Pass-1-style literal substitution → caught; the
anchor-moved mutant → 2 sites, still reddens on the count. No false positive on the clean tree —
the `rq() {` definition has no space after `rq`, and the prose `$(rq …)` mentions are inside
comment lines the existing `grep -v '^[0-9]*: *#'` filter already removes. If the widening is
declined, the alternative is equally acceptable: narrow the stated bound to *"a literal
`GRAPH.*` command at a `$(rq …)` call site"*, which is what it actually checks.

### Pass 1 dispositions

- **Major 1 (trailer claimed to exclude a part-way reply) — FIXED, and the new positive claim
  holds.** [executed] *"cannot satisfy it part-way"* is gone from `pipeline.sh`; the does-not-cover
  list is now three items with `RESULTSET_SIZE` named and the call-site bound stated in place (1, 1
  and ≤8 rows). I re-measured the figures quoted: `UNWIND range(1,200000) AS x RETURN x` → 10003
  lines, last data row `10000`, trailer intact, rc 0. The mid-stream-abort claim — asserted
  *positively*, which is what the coordinator flagged — is **true and I established it on a harder
  case than the comment cites**: 5,000 matching rows with the division by zero at row 5,000
  (`MATCH (n:R) RETURN CASE WHEN n.x = 5000 THEN 1/0 ELSE n.x END`) returns the single line
  `Division by zero` — 4,999 producible rows genuinely discarded, no header, no trailer.
  `falkordb-quirks.md:797-800` scopes the old wording to *aborts* and marks the generalisation
  explicitly **false** with a cross-reference; `SKILL.md:120-121` widened correctly.
- **Major 2 (rc 2 unreadable and untested) — FIXED, by a better closure than I proposed, and my
  own Pass 1 suggestion was wrong.** [executed] Both bash claims hold: inside `if ! V="$(f)"`,
  `$?` is **0** for a function returning 1 *and* for one returning 2 — the `if` consumed it — so my
  "capture `rc=$?` at the three call sites" does not work as written and would have shipped a
  branch that never fires. And `exit 2` inside `$(…)` kills only the subshell; the script runs to
  its end (the substitution's own rc *is* 2, but every call site's `if !` collapses it anyway).
  Deletion is right. Fail-closed confirmed live: `rq` with `GRAPH.DELETE` returns **1** — in fact
  it cannot even reach the bare `OK` the comment cites, because `rq` always appends the cypher
  argument, so the reply is `ERR wrong number of arguments for 'graph.DELETE' command`. Safer than
  claimed, not less safe. The replacement check **does** redden where the deleted guard could not:
  a literal `GRAPH.DELETE` at an existing call site → `FAIL … GRAPH.DELETE at pipeline.sh:444`,
  exit 1; a call site rewritten out of the anchor → `FAIL found only 2 rq call sites`. The
  self-anchor is real. See Major 7 for where it stops.
- **Minor 3 (non-discriminating assertion) — FIXED as suggested**, by the comment option:
  `test-stamp-wiring.sh:270-274` now states that only the first two `must-contain` strings pin the
  fix and that the third is a branch-completion check, *"do not read it as one."*
- **Minor 4 (line-number citation) — RETRACTED, my error.** See above.
- **Minor 5 (doc enumerations named 9 of 10 cases) — FIXED.** [executed] Both `SKILL.md:155` and
  `freshness.md:281` now name the pipeline-clean-marker control.
- **Nit 6 (over-long line) — FIXED.** [executed] No introduced line over 100 chars remains in the
  edited region of `freshness.md`; the sole survivor is the pre-existing 108-char line.
- **Open question 1 (`RESULTSET_SIZE` novelty) — RETRACTED, my error.** See above.
- **Open question 2 (clear `4f9c21ae…` before or after narrowing) — RESOLVED.** The narrowing has
  landed in both `pipeline.sh` and `falkordb-quirks.md`, so the entry's over-broad closing clause
  no longer has an inheriting home. **Both entries are now clear-able**, on the same reading as
  Pass 1.

### What's solid in the delta

- The three-item does-not-cover list is the right structure: it separates *execution* from
  *correctness* from *completeness*, which is the distinction the whole arc has been converging on,
  and it puts the numeric bound at the point of use rather than in a doc a call-site author won't
  open.
- Moving the precondition from runtime to a static check is the correct direction, and the
  `< 3 call sites` self-anchor is the part I'd have most expected to be missing. A check that can
  silently check nothing is the failure mode; this one refuses to.
- The commit message records the disputed line numbers resolving *against* the reviewer. That is
  the right record to keep.

### Appendix B

#### B1 — the widened anchor, run against five trees

Anchor `grep -nE '(^|[^_[:alnum:]])rq '` (comment lines filtered as today) + case-insensitive
`GRAPH\.[A-Za-z_.]*`, upcased before comparison:

```
CLEAN : sites=3 bad=[none]
M1 (literal GRAPH.DELETE at an existing $(rq ) site)  : sites=3 bad=[GRAPH.DELETE@444]
M2 (a call site rewritten out of the $(rq ) form)     : sites=2 bad=[none]        <- count anchor fires
M3 (new bare-statement call site, GRAPH.DELETE)       : sites=4 bad=[GRAPH.DELETE@420]
M4 (lowercase graph.delete at an existing site)       : sites=3 bad=[graph.delete@444]
```

Against the shipped check, M3 and M4 both report `PASS all 3 rq call sites` with suite exit 0;
M1 and M2 redden.

#### B2 — mid-stream abort with 4,999 producible rows ahead of the error

```
GRAPH.QUERY  <throwaway> 'UNWIND range(1,5000) AS x CREATE (:R {x:x})'
GRAPH.RO_QUERY <throwaway> 'MATCH (n:R) RETURN CASE WHEN n.x = 5000 THEN 1/0 ELSE n.x END AS v'
  -> 1 line, exit 0: "Division by zero"      (no header, no rows, no trailer)
```
