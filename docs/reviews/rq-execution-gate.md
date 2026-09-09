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
