# Kaizen — Change History: graph-dba

> Dated log of actual changes to the `graph-dba` agent. Most recent first.


## 2026-09-09 — K-009 fixed and K-008 closed: `rq()` now recognises success positively (U28)

- **What:** both remaining `pipeline.sh` items closed in one unit, because they touch one file.
  **K-009** — `rq()` returned 0 on a bare FalkorDB runtime-error reply — is **fixed**;
  **K-008** — two CPG-freshness facts overtaken by `6012ddb` — is **closed with no doc edit**,
  both dispositions re-derived by execution here.

### K-009 — the fix

- **What was wrong.** `rq()` classified failure with a prefix `case`
  (`errMsg:*|ERR\ *|WRONGTYPE*|*"read only"*|*"read-only"*`). That is a blacklist of error
  shapes; FalkorDB emits several with no prefix at all and `redis-cli` exits 0 on every error
  reply, so those returned **0** — success — at any call site passing no `[must-contain]`.
- **What replaced it: a positive test, not a longer blacklist.** `rq()` now requires the reply's
  **last line** to begin `Query internal execution time:` — the statistics trailer that only a
  query the server ran to completion emits. Any error shape, including ones nobody has met yet,
  fails **closed**. The prefix `case` is gone entirely rather than kept as a second gate: with the
  trailer required it can only ever agree, and leaving it would suggest the classification is
  still by error shape.
- **Also added: a command guard.** The trailer is a `GRAPH.QUERY`/`GRAPH.RO_QUERY` property.
  `GRAPH.DELETE` answers the bare status `OK` (observed), so `rq()` now **refuses** any other
  command with rc 2 and an internal-error message rather than mis-judging a reply the test cannot
  apply to.
- **Probe table — the helper extracted verbatim from `pipeline.sh` and pointed at a live throwaway
  graph (`gdba_u28_rqprobe`, deleted afterwards), before and after:**

  | probe | reply | before | after |
  |---|---|---|---|
  | `RETURN (((` | `errMsg: Invalid input at end of input: …` | 1 | 1 |
  | `CREATE (:P2)` via `GRAPH.RO_QUERY` | `graph.RO_QUERY is to be executed only on read-only queries` | 1 | 1 |
  | `RETURN nosuchfunc(1)` | `Unknown function 'nosuchfunc'` | **0** | 1 |
  | `MATCH (n:Probe) RETURN keys(n.k)` | `Type mismatch: expected Map, Node, Edge, or Null but was Integer` | **0** | 1 |
  | `UNWIND [1,0] AS x RETURN 1/x AS stray` | `Division by zero` | **0** | 1 |
  | long scan with `TIMEOUT 1` | `Query timed out` | **0** | 1 |
  | **CONTROL** `MATCH (n) RETURN count(n)` | header + row + trailer | 0 | 0 |
  | **CONTROL** stray query returning **zero rows** | header + blank + trailer | 0 | 0 |
  | **CONTROL** `MERGE (b:CpgBuildInfo) SET b = {…} RETURN 1` (a **write**) | counters + trailer | 0 | 0 |

  Three passing controls, not just failing probes: without them "fixed" is indistinguishable from
  "always returns 1". The timeout row was not in the item as opened — it was found here, and it is
  the reason the test is anchored on the *whole* trailer prefix rather than on `Query`. Observed in
  passing: with the graph key absent, every `GRAPH.RO_QUERY` probe answers
  `ERR Invalid graph operation on empty key` and `rq()` returns 1, while the `GRAPH.QUERY` control
  materialises the key and succeeds — the asymmetry `pipeline.sh`'s own comment gives as the reason
  its reads use `GRAPH.RO_QUERY`.
- **Writes carry the trailer**, which the code's own comment had declined to assume and therefore
  left the stamp write ungated. Measured on the exact stamp shape, on the run that created the
  node and on the re-run where the `MERGE` matched. That is why the gate is unconditional instead
  of per call site.
- **Mutation runs — each restored by copy immediately after, never batched:**
  - *pre-fix `pipeline.sh` restored*: the two new wiring cases go **FAIL** (`output lacks:` the
    branch wording), the 12 pre-existing assertions stay PASS — counted from that run's own
    output: 12 `PASS` + 2 `FAIL`, suite exit 1.
  - *gate pattern changed to one that can never match*: the three controls go MISMATCH, the six
    error probes stay caught — the discriminator against "always returns 1".
  - *gate weakened to `Query*`*: only the timeout probe reddens.
  - *gate deleted outright*: all six error probes redden, controls stay green.
- **Second defect found and fixed in the same block.** The `PARSED_AT` read-back call site read
  `rq … GRAPH.RO_QUERY || true` and judged the reply's **text** alone, so a read-back that never
  ran fell into the "the freshness stamp did not land in '<graph>'" branch — a claim about the
  graph from a check that never reached it, followed by advice to re-send a stamp that may be
  sitting there correctly. It is now a distinct branch that says whether the stamp landed is
  **UNKNOWN**, prints the rendered Cypher as comparison material, and tells the operator to check
  by hand. Six failure branches now, not five: two "did not land", three "did land", one "cannot
  say".
- **Regression cover.** `test-stamp-wiring.sh` gains two cases — `stamp_bare_error` (the stamp
  write rejected with no prefix) and `readback_error` (the read-back itself errors). Neither is
  asserted on the exit code: **both exited 1 before the fix too**, by falling through to a later
  assertion and reporting that one's finding, so each asserts its branch's own wording. Suite:
  **14 assertions, of which 10 `run_case`**, all green (`grep -c '^  PASS'` on the run's output =
  14, `FAIL` = 0, exit 0). The two counts reconcile: 10 `run_case` invocations (8 at `d43ca40`,
  plus these 2) and 4 `PASS` lines emitted outside `run_case` — the stray-query guard, which reads
  as one block in the source but loops over two shapes and so prints two, plus the P5-1 and P6-6
  mutation checks. 12 pre-existing, 2 new.
- **What the guard covers, exactly.** Any `GRAPH.QUERY`/`GRAPH.RO_QUERY` reply the server did not
  run to completion — parse error, runtime error, read-only refusal, timeout, mid-stream abort,
  empty reply — plus, via the exit status, an unreachable server. **What it does not cover:** a
  query that ran to completion and answered *wrong*, or answered about the wrong thing. The
  trailer proves **execution, never correctness**; asserting anything about a reply's contents
  stays the caller's job (`[must-contain]`, the `PARSED_AT` read-back). That distinction is
  written into the code comment, not left implicit.

### K-008 — closed, no doc edit warranted

- **Fact 2 (scoped dirty check) — CONFIRMED delivered, by execution.** `git-provenance.sh:76`
  sets `spec=":(literal)$name"` and `:105` reads
  `if [ -n "$(git -C "$dir" status --porcelain -- "$spec" 2>/dev/null || true)" ]; then` — quoted
  as they read, rather than as one substituted line. (`analyst`'s review reports the status call at
  `:107` and `:105` as the `CPG_SOURCE_TREE` assignment; re-checked with `grep -n` against an
  unmodified `git-provenance.sh`, the status call is `:105` and the tree assignment `:103`.) Run
  against this repo while it was
  dirty at `skills/joern-cpg/scripts/`: target `skills/cpg-analysis` → rc 0,
  `commit=d43ca40a69a3`, `dirty=false`; target `skills/joern-cpg/scripts` (the dirt is inside it)
  → `dirty=true`; target the repo root → `dirty=true`. The detector discriminates rather than
  merely returning `false`. `skills/cpg-analysis/references/freshness.md:33` already documents
  `sourceDirty` as scoped — "it says nothing about the rest of the repo".
- **Fact 1 (staging a pruned copy inside the repo) — CONFIRMED superseded, by execution.** A copy
  staged at `cpg/.cpg-artifacts/src/u28probe` (confirmed ignored via `git check-ignore`:
  `cpg/.gitignore:3`) → `cpg_provenance_capture` returns **rc 1** with all four `CPG_SOURCE_*`
  empty; the tracked directory it was copied from returns rc 0 with a real commit. Driving
  `pipeline.sh`'s provenance branch on that path yields `PROVENANCE=none`. `--source-origin` is
  the supported route and `skills/joern-cpg/SKILL.md`'s "No `--exclude`" bullet already says so
  in those words ("Where you stage it doesn't recover the source's git identity —
  `--source-origin` does"). The probe copy was removed.
- **Disposition: closed.** Both facts are already published at their point of use, and neither
  survives as an edit to apply. Nothing re-scoped.

### Review round — `analyst`'s gate (`docs/reviews/rq-execution-gate.md`), closed 2026-09-09

- **Major 1 — the truncation claim was false, narrowed.** `pipeline.sh`'s comment said the
  last-line anchor meant "a reply cut off part-way cannot satisfy it". Re-measured here:
  `GRAPH.CONFIG GET RESULTSET_SIZE` → **10000**, and
  `GRAPH.RO_QUERY <g> "UNWIND range(1,200000) AS x RETURN x"` through the shipped `rq()` returns
  **rc 0**, 10003 lines, last data row `10000`, last line
  `Query internal execution time: 5.220555 milliseconds`. The gate passes a reply that lost 190,000
  rows. The mid-stream-abort half is genuinely closed and is now stated as the scoped fact it is
  (FalkorDB discards the produced rows and answers one bare line), while the comment now names
  three things the trailer cannot see, the third being a complete-looking reply missing rows. The
  three call sites return 1, 1 and ≤8 rows, so none is near the cap — said in the comment, so a
  future call-site author reads the bound rather than inheriting the generalisation.
  `falkordb-quirks.md` had the same generalisation ("no partial reply to worry about") and is now
  scoped to *aborts*, cross-referenced to its **pre-existing** `RESULTSET_SIZE` bullet. That bullet
  answers `analyst`'s open question 1 in the negative: `RESULTSET_SIZE` **was** already documented
  (since 2026-07-30), so the new fact — that a capped reply is structurally indistinguishable from
  a complete one, which is what bounds the trailer discriminator — was folded into it rather than
  written as a second bullet.
- **Major 2 — closure chosen: NARROW the runtime claim, and move the mechanism where it can be
  tested.** The `return 2` guard is deleted. Three findings decided it, each verified here by
  execution rather than taken on report:
  1. *It reached nobody.* All three call sites are `if ! VAR="$(rq …)"`; a stub returning 1 and one
     returning 2 take the same branch, and `$?` read inside the branch is **0** (the `if` consumed
     it), so `$VAR` is empty and the caller renders `<no reply>`.
  2. *`exit 2` could not have escaped either.* `f() { echo …; exit 2; }; out="$(f)"` leaves the
     script running with `rc=2` captured — the exit killed only the `$(…)` subshell. That is the
     same command-substitution trap that ate `cpg_provenance_stamp`'s allow-list two commits ago.
  3. *Deleting it is safe.* Without the guard the untrapped behaviour already fails **closed**: a
     `GRAPH.DELETE` success answers a bare `OK`, which does not match the trailer, so `rq` returns
     1. The cost of a misuse is a false failure, never a false pass — a materially different risk
     from the one K-009 was about.
  Keeping it was not an option: at the stamp site it printed "FalkorDB rejected the freshness
  stamp … `<no reply>`" about a call that never left the shell — structurally the same overclaim
  the read-back branch was fixed for, two lines below `rq`'s own correct message.
- **The precondition is now enforced statically, and it reddens.** `test-stamp-wiring.sh` gains a
  check over `pipeline.sh`'s own `rq` call sites: every one must pass `GRAPH.QUERY` or
  `GRAPH.RO_QUERY`. Mutation-tested both ways, each restored by copy immediately:
  a fourth call site written as `rq "$STAMP" GRAPH.DELETE` → `FAIL … GRAPH.DELETE at
  pipeline.sh:419`; the helper renamed so the grep anchor misses → `FAIL  found only 0 rq call
  sites`. This is what the runtime guard could not do — `analyst` established, and the point is
  the whole reason for the swap, that deleting the runtime guard left the suite byte-identical.
  The check states its own bound in the file: it catches a **literal** wrong command, not one
  reaching `rq` through a variable, which no call site does today.
- **Minors.** Case 1 of the two new ones now carries a comment saying only its first two
  `must-contain` strings pin the fix — the third is printed by the pre-fix code too, so it checks
  the branch reached its end and is not a discriminator. Both docs' case enumerations gained the
  tenth `run_case` (`regression: merge, pipeline-clean marker`) and the new static check. The
  105-char line at `freshness.md:185` is re-wrapped. The `git-provenance.sh` citation is corrected
  above — including `analyst`'s own line numbers, which I re-checked rather than adopted.
- **Suite after the round: 15 `PASS`, 0 `FAIL`, exit 0.** `GRAPH.LIST` back to **25** keys; the
  throwaway keys `gdba_u28_rqprobe`, `gdba_u28_delprobe` and `gdba_u28b` were all deleted.
- **`4f9c21ae-7b30-4d62-9c18-6ea5d0b73c41` is unblocked.** It was held from clearing because its
  closing clause carried Major 1's generalisation; that wording is now scoped in both places the
  promoted text lives.

### Docs updated in the same change

- `skills/joern-cpg/SKILL.md` — the "rejected stamp" bullet now states the trailer mechanism, the
  execution-not-correctness limit, and **six** branches split three ways; the
  `test-stamp-wiring.sh` bullet lists the two new cases.
- `skills/cpg-analysis/references/freshness.md` — the read-back guarantee no longer rests on a
  check that might not have run.
- `claude/graph-dba/falkordb-quirks.md` — the "`redis-cli` exits 0" bullet had two claims this run
  falsified or closed: *"measured on `GRAPH.RO_QUERY` only, so don't assume it of a `GRAPH.QUERY`
  write reply"* (writes **do** carry the trailer — measured) and *"`rq()` still returns 0 on a
  `Type mismatch:` reply"* (fixed). Added: anchor the test on the **last line** and require the
  whole trailer prefix, because `Query timed out` shares the first word and returned data could
  carry the literal.

- **Kept-open graph entries.** `b7f3c2a1-9d4e-4c11-8a52-6e0f1d3b7c94` (the prefix taxonomy) and
  `4f9c21ae-7b30-4d62-9c18-6ea5d0b73c41` (the mid-stream abort that makes the trailer test sound)
  are **fully addressed** by this change — both the code fix and the quirks-file correction. They
  are the curator's to clear.
- **Plan items:** K-008 and K-009 both closed and removed from `plan.md`.


## 2026-09-09 — the CPG stamp-race entry discarded: fixed 40 minutes after it was captured (U26)

- **What:** `cobb`, distilling `graph-dba`'s own single produced `kaizen_team` entry (unit U26),
  **discarded** `f3c1a27e-9b64-4d18-a5e2-7c0b91d4e8a3` (2026-09-07, `suggestedHome: knowledge
  base`). Nothing promoted, no backlog item opened. The entry held that `pipeline.sh` computed the
  `CpgBuildInfo` `SOURCE_COMMIT`/`SOURCE_DIRTY` stamp with `git -C "$SRC"` *after* the load, so on
  a multi-hour build both values were repo-wide and raced any concurrent session.
- **It was accurate when written and dead the same evening.** The entry's `createdAt` is
  2026-09-07T22:40:00Z; `6012ddb` ("capture CPG provenance before the parse, scoped by pathspec")
  is dated 2026-09-07 20:20:21 -0300 — 40 minutes later.
- **Re-derived by execution, not by re-reading the citation.**
  - *Pre-fix* (`6012ddb^`): the stamp block sat at `pipeline.sh:136-140`, after build (`:66`),
    export (`:69`) and transform+load (`:89`), and ran `git -C "$SRC" rev-parse --short HEAD`
    plus an unscoped `git -C "$SRC" status --porcelain`. `git-provenance.sh` did not exist. The
    entry's description of the mechanism is exact.
  - *Now*: provenance is captured at `pipeline.sh:97-110`, before even `mkdir -p "$WORKDIR"`;
    `cpg_provenance_stamp` at `:235` passes the captured `CPG_SOURCE_*` values verbatim, with no
    `git` invocation anywhere between capture and stamp. Dirtiness is
    `git status --porcelain -- ":(literal)<name>"` (`git-provenance.sh:105`), and object ids are
    full 40-char OIDs rather than `--short`.
  - *The race, reproduced in a throwaway repo*: captured `sub/` at `c1`, then moved `HEAD` twice
    and modified a file **outside** `sub/`, then stamped. The rendered Cypher carried
    `SOURCE_COMMIT: "13cbf9a2591fd5d3b447c9ec6d0122f111831a0c"` — the parsed commit — and
    `SOURCE_DIRTY: false`. The pre-fix expressions, evaluated at that same instant, returned
    `4e577c8` and `true`. Both defect directions reproduce under the old code; neither survives
    the new one.
  - *Paired probes against this repo* (dirty under `claude/` and `model-bench/`):
    `cpg_provenance_capture model-bench` → `dirty=true` (the control, proving the detector fires
    at all); `cpg_provenance_capture skills/joern-cpg` → `dirty=false`; an untracked staged copy
    inside the work tree → return code 1, i.e. `PROVENANCE=none` rather than an inherited `HEAD`.
- **Why discarded rather than promoted:** the lesson is already published at the point of use, in
  three places, and stated more completely than the entry states it — the "Provenance" section of
  `skills/joern-cpg/SKILL.md`, the header of `skills/joern-cpg/scripts/git-provenance.sh` (both
  failure directions, with this build's four-commit `HEAD` sequence), and
  `skills/cpg-analysis/references/freshness.md` for a consumer meeting a pre-fix marker. It is not
  a FalkorDB fact, so `falkordb-quirks.md` is the wrong home; it binds only while building a CPG,
  so an always-loaded prompt is the wrong price.
- **No live residue.** `cpg_falkorchat`'s marker reads `PROVENANCE = hand-backfilled`,
  `SOURCE_COMMIT = b795f4c23e066278ba8582d8cdd213d03b73e9df`, `SOURCE_DIRTY = false` over 10
  keys — the correction the entry's own evidence describes, and the shape `freshness.md`
  documents.
- **Graph:** `producedEdges=1`, `mentionEdges=0`, so `otherRemaining == 0` and the whole node was
  `DETACH DELETE`d. This agent's two `MENTIONS`-only nodes (U24's kept-opens against K-009) were
  not touched.
- **Plan items:** none opened. See the K-008 note below, recorded by the same pass.


## 2026-09-08 — K-009 opened, and `falkordb-quirks.md` gains the mid-stream abort (U24)

- **What:** `cobb`, distilling `analyst`'s `kaizen_team` chunk F (unit U24), routed two
  `analyst`-produced entries here.
  - **`claude/graph-dba/falkordb-quirks.md`**, "Ops, config & tooling": one paragraph onto the
    `redis-cli`-exits-0 bullet U23 wrote, from `4f9c21ae-7b30-4d62-9c18-6ea5d0b73c41`. U23's
    affirmative discriminator (a real reply carries a column header plus the
    `Query internal execution time:` trailer) was hedged as *observed across five probes*; this
    closes it as **sound**, because a mid-stream runtime error aborts the whole reply — the rows
    already produced, the column header and the trailer are all discarded. Re-measured on module
    `41811`: `UNWIND [1,0] AS x RETURN 1/x AS stray` → the single line `Division by zero`;
    `Query timed out` identical; a zero-row success still prints header + `Cached execution: 0` +
    trailer. Consequence: requiring the trailer makes a negative "no stray rows" assertion
    fail-**closed**.
  - **`kaizen/plan.md` K-009** (new, high): `skills/joern-cpg/scripts/pipeline.sh`'s `rq()` returns
    **0** on a bare runtime-error reply — the prefix `case` at line 304 misses every unprefixed
    FalkorDB error, so a call site with no expected-substring argument reads a failed query as
    success. From `b7f3c2a1-9d4e-4c11-8a52-6e0f1d3b7c94`. Confirmed by extracting `rq()` verbatim
    and running it against a live graph, with a passing control and a caught parse error to prove
    the instrument.
- **Why:** both facts are `graph-dba`'s — the reply-shape behaviour belongs in its knowledge base,
  and the defect is in the `joern-cpg` pipeline it owns. The **fix** is component code, outside
  `cobb`'s write remit, so it is filed rather than applied. `9124a1f` closed the discarded-output
  half of this same trap; the guard that replaced it reopened it in a new shape.
- **Graph:** both raw entries had their `analyst` `PRODUCED` edge resolved and carry a new
  `MENTIONS`→`graph-dba` edge, so they stay alive for this agent's own distillation pass rather
  than being cleared ahead of the fix.
- **Plan items:** K-009 (opened).

## 2026-09-08 — `falkordb-quirks.md`: `redis-cli` exits 0 on an error reply (U23)


- **What:** `cobb`, distilling `analyst`'s `kaizen_team` chunk E (unit U23, entry
  `6eeaa03e-98f9-42ac-8c6f-d9a0d9a06791`), added one bullet under the tooling run of bullets, after
  the `GRAPH.QUERY`-materializes-an-empty-key entry and before the `GRAPH.EXPLAIN` one: `redis-cli`
  exits 0 on a server *error reply* and prints the error text to STDOUT, so `$?`, `||` and `set -e`
  cannot see a rejected or malformed query. No existing bullet was rewritten. The concurrent CPG
  session's uncommitted `Properties removed` bullet (lines 190-199) was not touched, reflowed or
  reindented; the insertion is ~550 lines below it.
- **Why it is a new bullet rather than a fold.** The file already carried the fact as a *clause
  inside another bullet* — `falkordb-quirks.md:141`, "because `redis-cli` exits 0 on Redis-level
  errors, a `set -e` script sails past it", where the subject is `CREATE VECTOR INDEX` being
  rejected. That aside is true and undated, and it says nothing about which stream carries the
  text, nothing about the one case that *does* exit non-zero, and nothing about what a caller
  should do instead. Those three are the whole value.
- **Paired control, 2026-09-08, `redis-cli 7.0.15` against `localhost:6379`, module `41811`** — a
  good command and a bad one, compared on exit status *and* stream:
  `PING` → stdout `PONG`, exit 0. `NOTACOMMAND` → stdout `ERR unknown command 'NOTACOMMAND'`,
  **stderr empty**, exit 0. `GRAPH.RO_QUERY kaizen_team "MATCH (n:Agent RETURN count(n)"` → stdout
  `errMsg: Invalid input 'R': …`, stderr empty, **exit 0**. `( set -e; redis-cli GRAPH.RO_QUERY …
  >/dev/null; echo … )` prints its trailing echo and exits 0. **The control that turns this into a
  finding:** `redis-cli -p 6399 PING`, nothing listening → empty stdout, `Could not connect to
  Redis at 127.0.0.1:6399: Connection refused` on **stderr**, **exit 1**. `$?` is therefore not
  uniformly useless — it is reliable for connection failure and blind to error replies, which is
  exactly why a guard tested the easy way looks like it works.
- **The named instance is real and already fixed, verified by `git show`, not by narration.**
  `skills/joern-cpg/scripts/pipeline.sh:199` at `6012ddb` reads
  `redis-cli -h "$HOST" -p "$PORT" GRAPH.QUERY "$GRAPH" "$STAMP" >/dev/null` — verbatim the shape
  the entry names, at the line it names, so a failed `CpgBuildInfo` stamp was invisible after a
  multi-hour build. `9124a1f fix(joern-cpg): the stamp can now fail loudly` replaced it with the
  `rq()` helper that captures stdout, keeps the `||` for the connect case and `case`-matches
  `errMsg:*|ERR\ *|WRONGTYPE*|*"read only"*` on the reply text. That structure is precisely the
  split the paired control measures. **No `MENTIONS`→`graph-dba` edge was tagged** and no `K-`
  item opened: nothing is outstanding for `graph-dba` to act on, so an edge would only have left
  a node alive that the next pass could not close.
- **The bullet was corrected before this unit closed, and the correction is the more useful half.**
  As first written it recommended `case "$out" in errMsg:*|ERR\ *|WRONGTYPE*) return 1 ;; esac` —
  copied from `pipeline.sh`'s `rq()`. Verifying the surviving population after the clears surfaced a
  live `analyst` entry dated 2026-09-08, `b7f3c2a1-9d4e-4c11-8a52-6e0f1d3b7c94`, saying FalkorDB
  error replies are **not uniformly prefixed**. Re-derived independently with a paired control
  before acting on it (`GRAPH.RO_QUERY` against `kaizen_team`, each reply tested against that exact
  prefix set): a parse error (`THIS IS NOT CYPHER`) → `errMsg: …` **caught**; an absent graph →
  `ERR Invalid graph operation on empty key` **caught**; `RETURN nosuchfunc(1)` → bare
  `Unknown function 'nosuchfunc'` **missed**; `MATCH (n:KaizenEntry) RETURN keys(n.fact)` → bare
  `Type mismatch: expected Map, Node, Edge, or Null but was String` **missed**; all exit 0, with a
  valid query as the control. So the recommendation I had just shipped was unsound, and
  `skills/joern-cpg/scripts/pipeline.sh`'s `rq()` **still returns 0 on a runtime-error reply** —
  `9124a1f` closed the discarded-output half of the trap and left the bare-error half open. The
  bullet now says: assert the intended effect positively (read the write back), or use a client
  that raises; a prefix `case` is a courtesy message, never the check. `pipeline.sh` is outside
  `cobb`'s write remit, so the live defect is reported to the coordinator, not fixed here.
- **Note for a later distillation pass:** `b7f3c2a1-…` was **not cleared** — it is dated 2026-09-08
  and falls outside U23's pinned 2026-09-07 scope. Its content is now published in this file, so a
  later pass will correctly find it already documented; this line is the record of why.
- **No probe graph created.** Every read used `GRAPH.RO_QUERY` against the existing `kaizen_team`
  key; the deliberately-failing queries were `RO_QUERY` too, which per this file's own bullet
  materializes nothing. Confirmed for the one probe that named a fresh key —
  `GRAPH.RO_QUERY definitely_absent_graph_u23 …` → `ERR Invalid graph operation on empty key`, then
  `EXISTS definitely_absent_graph_u23` → **0**. No `ws:probe-*` or scratch key was created by this
  unit.

## 2026-09-08 — `falkordb-quirks.md`: the projection half of the `UNIQUE`-vs-existence gap (U22), resolving a dangling cross-reference U21 left

- **What:** `cobb`, distilling `analyst`'s `kaizen_team` chunk D (unit U22, entry
  `df03e2c1-86b7-4d9a-8b24-7710785226d4`), added one bullet under *Cypher dialect & query
  behavior*: a property a node does not carry projects as `null`, never an error, so a `UNIQUE`
  constraint is no guarantee a projection's key is present. No existing bullet was rewritten, and
  the concurrent CPG session's uncommitted `Properties removed` bullet (lines 190-199) was left
  byte-identical and unmoved — the new bullet was inserted after it.
- **Why it was not the expected discard.** The entry overlaps U21's already-promoted `28d78725`, and
  the brief expected a discard-a-fortiori. Reading the file as it stands instead found that U21's
  promotion ends with *"see the `RETURN n.prop` → `null` entry under Cypher dialect for the
  projection side of the same gap"* (`falkordb-quirks.md:59`) and **no such entry existed** — a
  pointer written to content that was never added. This entry is that content, so promoting it
  repairs my own prior unit's defect rather than duplicating it.
- **Re-derived read-only, no probe graph created** (module `41811`, 2026-09-08):
  `MATCH (a:Agent {agentId:'cobb'}) RETURN a.noSuchProperty` on `kaizen_team` returns one row
  holding `null`, `IS NULL` is `true`, `keys(a)` is `['agentId']`, nothing raises; and
  `CALL db.constraints()` on the `reference` graph returns four rows, all `UNIQUE`
  (`Product[productId]`, `Step[stepUid]`, `Entity[entityId]`, `WorkflowDef[key,version]`), with **no
  `MANDATORY` row at all** — which is the entry's constraint-side claim, confirmed. The entry's
  downstream observation (`filter_products` → `list_catalog` carrying `productId: null` to the
  caller) is carried attributed and dated to `analyst`, 2026-09-03: it is no longer reproducible,
  since all 15 current `Product` nodes carry the key.
- **Plan items:** none.

## 2026-09-08 — `falkordb-quirks.md`: `UNIQUE`-vs-null, the `EXISTS` subquery gap, and observable queue depth (U21)

- **What:** `cobb`, distilling `analyst`'s `kaizen_team` chunk C (unit U21, entries
  `28d78725-cff4-456b-8ad9-c637ac0b12de` and `94917906-3ae1-40c6-930d-14df73cfaa04`), added three
  bullets to `claude/graph-dba/falkordb-quirks.md` — two under *Indexing, constraints & DDL*, one
  under *Ops, config & tooling*. No existing bullet was rewritten.
- **`UNIQUE` node constraints and null.** A `UNIQUE` constraint does not constrain nodes lacking
  the property, and **"absent" and "explicitly null" are the same state**: `CREATE (:Chan
  {name:'c', pid:null})` reports `Properties set: 1` and `keys(n)` → `[name]`, so the null is
  discarded at write and never stored. Two same-string nodes are correctly rejected (`unique
  constraint violation on node of type Chan`); `DETACH DELETE` → re-`CREATE` of the same value is
  clean, so a delete-then-recreate cycle is not a re-join hazard. Practical verdict: a `UNIQUE` on
  a **nullable marker** property is safe, but it is emphatically not an existence constraint — the
  bullet cross-references the existing `RETURN n.prop` → `null` entry for the projection side.
  This **corrects** the raw entry, which framed absent and explicitly-null as two distinct exempted
  states.
- **`EXISTS { MATCH … }` / `exists((pattern))`.** Both unusable on this build, and they fail at
  **different stages** — the first is a genuine parse error (*"Invalid input '(': expected ':', ','
  or '}'"*), the second parses and dies at plan time (*"Unable to resolve filtered alias
  '(c)-[]->()'"*). The raw entry called both parse failures. The working anti-join is
  `OPTIONAL MATCH … WITH x, t WHERE t IS NULL`, re-run clean.
- **`GRAPH.INFO` exposes live queue depth.** It returns `# Running queries` / `# Waiting queries` /
  `Object Pool` sections and takes **no graph key** (instance-wide, a detail the raw entry omits);
  `GRAPH.CONFIG GET MAX_QUEUED_QUERIES` → `25`. The point is about **done-conditions**: a capacity
  or load-test assertion can be written against observed queue depth rather than degrading to "no
  query was rejected", which only reddens after the cap has already been hit.
- **Evidence:** all three re-derived live 2026-09-08 on module `41811` — the constraint and
  subquery probes against a disposable graph `cobb_u21_probe` (`GRAPH.DELETE`d in the same run,
  confirmed absent from `GRAPH.LIST`), the `GRAPH.INFO`/`GRAPH.CONFIG` reads against the shared dev
  instance. Full disposition record for the chunk: `claude/analyst/kaizen/history.md`, 2026-09-08
  (chunk C / U21).
- **Plan items:** none.

## 2026-09-08 — `falkordb-quirks.md`'s `TIMEOUT` bullet gained the write-side consequences (U20)

- **What:** `cobb`, distilling `analyst`'s `kaizen_team` chunk B (unit U20, entry
  `d8039ade-c9ae-4be0-83eb-681dc2f0b5d5`), extended the existing "Default `TIMEOUT` is 1000ms —
  and writes ignore it entirely" bullet rather than adding a new one: the headline was already
  there, the write-side consequences were not.
- **Added, all measured 2026-09-08 on module `41811` against a disposable graph
  (`cobb_u20_scratch`, deleted after):** the re-measured pair that pins the asymmetry (a
  `CREATE`-ing `UNWIND range(1,20000000)` ran **1.75 s** untouched; a 4-way cartesian `MATCH` over
  400 nodes was killed at **exactly 1.00 s**); that the only bound on a write is the client's
  `socket_timeout` and **it does not roll back** (a `retry`-disabled `redis.Redis(socket_timeout=
  0.5)` raised `TimeoutError` at 0.50 s and all 4 nodes were committed); and — not in the source
  entry, surfaced by the test — that **a retrying client re-applies the write**: the identical
  call through a stock `redis.Redis(...)` left **36 nodes** where the query creates 4, because
  redis-py 8.0.1's default connection carries `Retry(ExponentialWithJitterBackoff(), retries=10)`
  with `TimeoutError` in its supported set. Closed with why this lab is not exposed by default —
  `falkordb-py` disables retry — pooled `Connection.retry._retries == 0`, while the
  `FalkorDB(...).connection` client has no `.retry` and `get_retry()` → `None` (same on 1.6.1 and
  1.6.2: an object difference, not a version one) — so `falkor-chat`
  (`db.py:44`) and `cypher-mcp` (`server.py:903`) get one attempt; a helper script reaching for
  bare redis-py does not.
- **Why:** the entry's own evidence proved only the first half (that a write outruns the server
  timeout) and never demonstrated the no-rollback claim it asserted, so it was tested rather than
  believed — and the test found the sharper hazard.
- **Files:** `claude/graph-dba/falkordb-quirks.md`. Source disposition:
  `claude/analyst/kaizen/history.md` (2026-09-08, U20).

## 2026-09-08 — `falkordb-quirks.md` gained three dialect facts from `analyst`'s raw capture (U19)

- **What:** `cobb`, distilling `analyst`'s `kaizen_team` entries (unit U19,
  `claude/docs/plans/kaizen-distillation2-coordination.md`, chunk A — 2026-08-25..08-30), promoted
  three FalkorDB facts. All three were re-derived from scratch against a disposable graph
  (`cobb_u19_scratch`, deleted after), module `41811` re-confirmed via `INFO modules` — none was
  taken from the entry's own `falkor-chat`-mediated evidence.
  1. **Chained `FOREACH`** (*Cypher dialect*, folded into the existing `FOREACH` bullet, which
     covered nesting and multi-`CREATE` bodies but not chaining): two `FOREACH` clauses follow one
     `WITH` with no `WITH` between them, and each sees property values written earlier in the same
     query — by a preceding `SET` or by a preceding `FOREACH`. Both halves measured: a
     decrement-then-branch statement returned `1/running` on `2 -> 1` and `0/ready` on `1 -> 0`,
     and a `SET d.flag = 7` in one `FOREACH` correctly guarded the next. Makes a counter update
     plus its terminal-state flip one atomic statement.
  2. **`=~` is unsupported outright** (*Cypher dialect*, new bullet — the file had no regex entry
     at all): a hard error, *"FalkorDB does not currently support =~"*, reproduced verbatim.
     Carries the `toLower(...) CONTAINS` replacement and the `any(k IN keys(n) …)` whole-node form.
  3. **Duplicate result columns are rejected** (*Ops, config & tooling*, folded into the
     `result.header` bullet — same subject, un-aliased column naming). `RETURN d.id, d.id` and
     `RETURN count(d), count(d)` both fail with *"Error: Multiple result columns with the same
     name are not supported."* **The raw entry's mechanism was corrected before promotion:** it
     claimed the error comes "at query time — not at parse/compile time", but `GRAPH.EXPLAIN` on
     the same text errors identically, so it is raised in server-side validation and no rows are
     produced. The consequence that survives is the one that matters to a caller — there is no
     client-side parse step, so it still arrives as a `ResponseError` from `.query()`/`.ro_query()`.
- **Why:** engine behaviour, not `analyst` behaviour — it belongs in this on-demand knowledge base
  where every agent that writes Cypher can reach it. Same routing as U1, U10 and U11.
- **Files:** `claude/graph-dba/falkordb-quirks.md`. Source dispositions:
  `claude/analyst/kaizen/history.md` (2026-09-08, U19).

## 2026-09-07 — `falkordb-quirks.md` gained two more verified dialect facts from `coder`'s raw capture (U11)

- **What:** `cobb`, distilling `coder`'s `kaizen_team` entries (unit U11,
  `claude/docs/plans/kaizen-distillation2-coordination.md`, chunk B — 2026-08-31..09-02), promoted
  two further FalkorDB facts into the *Cypher dialect & query behavior* section. Both re-derived
  read-only by `cobb` against the live `reference` graph, module `41811` (v4.18.11) re-confirmed
  via `MODULE LIST` first.
  1. **A node extracted from a collected list by index degrades only in relationship patterns.**
     The raw entry claimed `head(collect(DISTINCT n))` yields something usable as a value but never
     as a pattern node. Re-derivation **narrowed and corrected both halves**: `MATCH (x)` on the
     bare alias re-binds correctly (one row, the same node — not a re-scan), and only a
     *relationship* pattern (`MATCH (x)-[r]-(y)`, `OPTIONAL MATCH (x)-[:REL]->(c)`) raises
     `encountered unexpected type in Record; expected Node`. And the trigger is **index extraction,
     not `head()`** — `collect(DISTINCT p)[0]` fails identically, which the raw entry had explicitly
     attributed to `head()`. `UNWIND` is the working route; both were measured.
  2. **Dynamic property access by variable key (`n[k]` for `k IN keys(n)`) works**, verified with a
     positive and a negative control so the result is not vacuously true — making a whole-graph
     "this value is stored under no property on any label" assertion a single read-only query.
- **Why:** engine behaviour, not `coder` behaviour — it belongs in this on-demand knowledge base
  where every agent that writes Cypher can reach it. Same routing as the four promoted at U10.
- **Files:** `claude/graph-dba/falkordb-quirks.md`. Source dispositions:
  `claude/coder/kaizen/history.md` (2026-09-07, U11).

## 2026-09-07 — `falkordb-quirks.md` gained four engine/client facts promoted out of `coder`'s raw capture (U10)

- **What:** `cobb`, distilling `coder`'s `kaizen_team` entries (unit U10 of
  `claude/docs/plans/kaizen-distillation2-coordination.md`, chunk A — 2026-08-27..29), promoted
  four FalkorDB facts into `graph-dba`'s on-demand knowledge base. `graph-dba` did not produce
  them; they are engine/client behaviour, so they belong here rather than in a `coder` file.
  Every one was **re-derived by `cobb` read-only** against the live instance
  (`redis-cli GRAPH.EXPLAIN` / `GRAPH.RO_QUERY` on the existing `reference` graph, plus one
  `falkordb-py` probe from `falkor-chat/server/.venv`) — no probe graph created, no write issued.
  Build confirmed first: `MODULE LIST` → graph module `ver 41811`, still **v4.18.11**.
  1. **`b3f2a1c4…` — folded INTO the existing `$param IS NULL OR` tuning entry, correcting it.**
     That entry (2026-08-22) closed with "no phrasing avoids that" full scan for the unfiltered
     call. False for a **range** predicate: `p.price >= coalesce($minPrice,-1.0) AND p.price <=
     coalesce($maxPrice,1e9)` plans `Node By Index Scan` with every parameter `NULL`, while the
     `IS NULL OR` form plans `Node By Label Scan` even with a real value bound. It does **not**
     work for equality via a self-referential coalesce (`p.categoryNormalized =
     coalesce($category, p.categoryNormalized)` → label scan; plain `= $category` → index scan).
     Two caveats `cobb` added that the raw entry did not carry: the whole-range index scan buys no
     selectivity (same lesson as the `IS NOT NULL` entry), and the sentinel bounds are a silent
     NULL/type filter that drops rows the `IS NULL OR` form would return.
  2. **`a3f1c2d4…` — new *Query tuning* entry, narrowed to the engine fact.** Re-derived:
     `toLower(p.categoryNormalized) = 'audio'` plans `Node By Label Scan` + `Filter`; the plain
     equality plans `Node By Index Scan`. There are no expression/functional indexes on this
     build, so case-insensitive matching means a precomputed normalized property plus
     client-side parameter normalization. The raw entry's `falkor-chat`-specific half (which
     property was the real scan anchor) was left out — already published in that component's
     `docs/QUERIES.md` §15.2 and in `Repository.filter_products`'s docstring.
  3. **`c1a9e6f0…` — new *Cypher dialect* entry, mechanism sharpened.** `collect()` of a **map
     literal** over a zero-row `OPTIONAL MATCH` returns `[{q: null}]`, size 1 — not `[]`.
     `cobb` measured the contrast the raw entry lacked: `collect(l)` → `[]` and `collect(l.qty)`
     → `[]` on the same row, so the rule is "a map literal is a non-null value whose fields are
     null", not "OPTIONAL MATCH pads its aggregates". Placed beside the `sum(CASE …)` / global
     `collect()` zero-row entries it belongs with.
  4. **`b7e1f0a2…` — new *Ops, config & tooling* entry.** `falkordb-py` 1.6.1 `result.header` is
     `[[type_code, name], …]`, and an un-aliased `RETURN` column is named by its literal
     expression text — probed live: `RETURN p.name AS name, p.price AS price` →
     `[[1,'name'],[1,'price']]`; without `AS` → `[[1,'p.name'],[1,'count(p)']]`. Matters to any
     generic row-to-dict mapper keyed off `res.header`.
- **Why:** `agent-maintenance` §5 step 3 — engine facts route to the on-demand knowledge base that
  owns them, never to an always-loaded prompt and never hoarded in the producing agent's files.
- **Docs touched:** `claude/graph-dba/falkordb-quirks.md` (+43 lines). No change to
  `graph-dba.md`, no plan item opened — nothing here is a `graph-dba` behaviour change.

## 2026-09-07 — `kaizen_team` distillation, U6 (pass 2): 9 current-shape entries — 6 promoted to `falkordb-quirks.md` (one a merged/corrected refinement pair), 1 promoted into `qa-engineer`'s knowledge base, 2 kept open as K-008 — all 9 cleared

- **What:** `cobb` ran `agent-maintenance` §5 for `graph-dba`, unit U6 of the second team-wide
  distillation pass (`claude/docs/plans/kaizen-distillation2-coordination.md`). All 9 entries were
  current-shape, `PRODUCED` by `:Agent {agentId:'graph-dba'}`, all dated 2026-09-02. Zero legacy
  `author`-property entries remained anywhere, so the legacy read was skipped.
- **Running build confirmed FIRST, before any disposition:** `redis-cli MODULE LIST` → graph
  module `ver 41811`, Redis 8.6.3. Still **v4.18.11**, so the eight version-stamped dialect/engine
  claims were judged against the build they were written against.
- **Every entry re-derived, not confirmed from its citation.** Two entries were corrected in the
  process (see 5/6 below) and one entry's own headline was found to misdescribe its own evidence.

  1. **`a79dd064…` — PROMOTED verbatim (Ops, `falkordb-quirks.md`).** Renaming a graph is a plain
     Redis `RENAME`/`RENAMENX` on its key; `GRAPH.COPY` + `GRAPH.DELETE` is unnecessary and
     destructive. Re-derived end to end on a scratch graph: key type `graphdata`; after `RENAME`
     the old key `EXISTS 0` and errors *"Invalid graph operation on empty key"*; counts identical
     (155); the RANGE index survives `OPERATIONAL` and still plans `Node By Index Scan`; reads and
     writes both work; the plan cache follows the key (`Cached execution: 1` on the new name for a
     plan compiled under the old one). `RENAMENX` returned `0` and refused against a taken name,
     `1` against a free one. `docs/plans/salesperson-ui-coordination.md` follow-up 10 had
     explicitly parked this promotion for `cobb`; it is now done.
  2. **`7f3c1a92…` — PROMOTED verbatim (Cypher dialect).** A bare `MATCH` straight after an
     `OPTIONAL MATCH` is rejected with *"A WITH clause is required to introduce a MATCH clause
     after an OPTIONAL MATCH."* Re-derived live against `kaizen_team` (read-only; a parse error
     fires before execution) — exact message match. Placed immediately above the 2026-09-06
     update-clause chaining entry and cross-referenced to it: same `WITH`-as-clause-boundary rule,
     different clause pair. Not merged — that entry is security-flavored and would bury a general
     dialect rule.
  3. **`b2e94c07…` — PROMOTED, MERGED into the existing `FOREACH` bullet (Cypher dialect).** A
     `FOREACH` body may hold several `CREATE` clauses binding across each other, and an
     outer-bound node variable is a legal `CREATE` relationship endpoint inside it. Re-derived: one
     guarded `FOREACH` with 3 `CREATE (…)` node clauses + 3 relationship clauses, `agent` bound by
     an outer `OPTIONAL MATCH`, wrote 3 nodes + 3 relationships; the false path wrote nothing; the
     list-subscript endpoint (`CREATE (ms[k])-[:NEXT]->(ms[k+1])`) errored `Invalid input '['`.
     Merged rather than appended because the file already carried the `FOREACH`-guard idiom and
     the map-projection-endpoint restriction this is the positive counterpart to.
  4. **`e0d7b264…` — PROMOTED (Cypher dialect, beside the sequential-`UNWIND` entry).** An
     `OPTIONAL MATCH` whose stream is still open when a later `UNWIND` expands rows gets
     multiplied by it — same results, no error, only cost. Re-derived by `GRAPH.PROFILE` on a
     51-user/100-cursor scratch graph: open-stream shape gives `Unwind | Records produced: 5100`
     (100 × 51) and `Aggregate` 0.966 ms; collapsing the `OPTIONAL MATCH` with its own
     `WITH … collect(DISTINCT rc)` before the `UNWIND` gives `Unwind | Records produced: 51` and
     `Aggregate` 0.241 ms, same answer. The entry's field-measured 3× wall-clock (≈690 → ≈240 ms
     at 50 participants / 2000 messages) is retained as the at-scale figure.
  5. **`d41f8b60…` + `c8a5e310…` — PROMOTED as ONE MERGED, CORRECTED entry (Query tuning), plus
     two spun-out bullets.** These are a **refinement pair**: `d41f8b60` recommends
     `WHERE prop > ''` as an always-true conjunct to upgrade a label scan to an index scan;
     `c8a5e310` (written later the same day) overturns it. Re-derivation confirmed the overturn on
     a single graph: bare `WHERE u.tokenHash IS NOT NULL` → `Node By Label Scan`, 51 records;
     adding `u.userId > ''` → `Node By Index Scan`, **50** records — the missing row is a
     `userId: 42` integer, because `42 > ''` is `NULL`. `d41f8b60`'s own evidence line claims
     "identical results", which is true only on an all-string population; promoting it verbatim
     would have shipped an idiom that silently under-deletes. The promoted entry therefore states
     the index-anchor fact **and** rejects the workaround on two independent grounds (unsound on
     mixed types; and it buys no selectivity — both plans visit the whole label and the index form
     measured slower, 0.128 vs 0.082 ms median over 20 runs at 52 nodes, reproduced here at 0.455
     vs 0.277 ms). Two further facts were spun out as their own dialect bullets: the cross-type
     comparison semantics (`42 > ''`, `42.5 > ''`, `true > ''` all `NULL`; `'abc' > '' → true`;
     `'' > '' → false` — all re-derived) and `d41f8b60`'s independent second claim that a global
     `collect()` over a zero-row `MATCH` still returns exactly one row carrying empty lists
     (re-derived: `size(users)=0`, `users=[]`, one row).
  6. **`c8a5e310…` trap (1) — PROMOTED, CORRECTED (Cypher dialect).** SQL-style `--` is not a
     Cypher comment. Re-derived: `MATCH (u:User) -- G1` + newline + `RETURN count(u)` errors
     `Invalid input 'G': expected '>' or '('`; the `//` form returns 51. **The entry's own headline
     — "Two traps that both fail SILENTLY in a WHERE clause" — is wrong about this half:** it fails
     *loudly*, with a parse error. The promoted bullet says so explicitly (the real cost is a
     misleading error message, not silence); only trap (2), the `> ''` conjunct, is silent.
  7. **`f6b21d84…` — PROMOTED into `claude/qa-engineer/qa-testing-techniques.md`** (a different
     agent's knowledge base — fully dispositioned there rather than `MENTIONS`-tagged, per the U2
     precedent). A closing code fence glued to the last code line is not a valid CommonMark
     closing fence; the block stays open to the next opening fence, swallowing prose, tables and
     whole sections, silently. Re-derived with `markdown-it-py` 3.0.0 (table rule on) on a minimal
     document: 2 fences / 1 table correct → **1** fence of 12 lines / **0** tables when one closing
     fence is glued, second code block unextractable. Routed to `qa-engineer` because the durable
     rule is a verification-design one — an extract-and-execute loop measures the code, not the
     document, and must assert extraction *shape* (block count, max block length, table count)
     — and because it is not a FalkorDB fact and had no business in `falkordb-quirks.md`.
  8. **`b701038b…` + `4f1976fc…` — KEPT OPEN as K-008.** Both verified true (the staged parse root
     `cpg/.cpg-artifacts/src/falkor-chat-server` is gitignored and `git rev-parse` resolves inside
     it; `pipeline.sh:139` runs `git -C "$SRC" status --porcelain` with no pathspec, and
     `git -C claude/graph-dba status --porcelain` was confirmed to report repo-wide while the
     pathspec-scoped form was empty). Not promoted because their correct homes —
     `skills/joern-cpg/SKILL.md` and `skills/cpg-analysis/references/freshness.md` — are outside
     `cobb`'s unprompted-write remit, and they are also genuinely the right homes: parking them in
     a `claude/graph-dba/` file to stay in-remit would be the hoarding anti-pattern §5 forbids.
     K-008 carries both `entryId`s, the verified statements, and the exact target bullets.
- **Why:** scheduled unit of the second team-wide `kaizen_team` distillation pass
  (`teco`-coordinated), this agent's turn.
- **`MENTIONS` tags added:** none. The one cross-agent fact (#7) was fully dispositioned into
  `qa-engineer`'s own knowledge base instead, so no entry was left for another agent's pass.
- **Order of operations honored:** this entry (and `qa-engineer`'s, and K-008) was written and
  confirmed before any graph mutation.
- **Docs touched:** `claude/graph-dba/falkordb-quirks.md` (7 additions/merges — 1 Ops, 5 Cypher
  dialect, 1 Query tuning), `claude/qa-engineer/qa-testing-techniques.md` (1 new section),
  `claude/graph-dba/kaizen/plan.md` (K-008), `claude/qa-engineer/kaizen/history.md`, this file.
- **Scratch graphs left behind (not deleted — `GRAPH.DELETE` is out of `cobb`'s remit):**
  `scratch_cobb_u6` (probe data for entries 3–6, ~155 nodes) and `scratch_cobb_u6_other` (1 node,
  created solely to prove `RENAMENX` refuses a taken name). Cleanup is the stakeholder's to route.
- **Plan items:** K-008 opened.

## 2026-09-06 — `falkordb-quirks.md` gains the update-clause chaining fact (inbound promotion from `security-expert`'s distillation)

- **What:** one new entry at the end of *Cypher dialect & query behavior* — a single Cypher
  statement can chain `MATCH` → `DETACH DELETE` → `WITH` → a further read clause, with no
  semicolon; the parser's demand for a `WITH` bridge between an update clause and the next read
  clause is trivially satisfiable, and `GRAPH.RO_QUERY` refuses the result at the engine's
  read-only check *on the parsed plan*, not as a syntax error or a text scan.
- **Why here:** it is a general fact about this build, not a `security-expert` rule — the entry's
  own `suggestedHome` named this file and this section, and the file already carries the companion
  parse-then-reject fact (the `GRAPH.EXPLAIN` entry under *Ops*, which relies on the same ordering
  to syntax-check a write). The security-relevant consequence is kept in the promoted text: a
  query-builder DSL's splice-point allowlist is load-bearing on its own, because Cypher grammar
  stops nothing and the read-only command is the only engine-level backstop behind it.
- **Origin:** raw `:KaizenEntry` `a3f1c2e4-9b7d-4e21-8c6a-1f2d3e4b5a6c` (produced by
  `security-expert` 2026-08-26 while reviewing falkor-chat's
  `docs/plans/workflow-nl-query-generation.md`), promoted by `cobb` in the `kaizen_team`
  distillation pass U2. Full disposition reasoning and the re-derivation evidence:
  `claude/security-expert/kaizen/history.md`, 2026-09-06 entry.
- **Verified 2026-09-06 by re-derivation, not by re-reading the citation** — two fresh probes on
  the live instance (module `41811`): the unbridged form returns the engine's own
  `Invalid input 'H': expected WITH`; the `WITH`-bridged form provably parses and is refused only
  by the read-only check. Stamped with that date in the promoted entry.
- **Plan items:** none opened.

## 2026-08-25 — `kaizen_team` distillation, U10 (team-wide pass): 1 legacy entry discarded (duplicate), 4 current-shape `MENTIONS`-only entries — 2 promoted verbatim, 2 promoted merged-and-corrected — all cleared

- **What:** `cobb` ran `agent-maintenance` §5 for `graph-dba`, unit U10 (last unit) of a team-wide,
  agent-per-agent distillation pass coordinated by `teco`
  (`claude/docs/plans/kaizen-distillation-coordination.md`). Read both shapes: the legacy
  `author`-property query (1 hit) and the current-shape `PRODUCED`/`MENTIONS` query (4 hits, **all
  reached only via `MENTIONS`** — each entry's own `PRODUCED` edge had already been resolved/deleted
  by its producing agent's own earlier unit in this same pass, leaving only the
  `(:KaizenEntry)-[:MENTIONS]->(:Agent{agentId:'graph-dba'})` edge for this unit to pick up, per
  this unit's brief).
  1. **`c3e5f8a2…` (legacy, `author:'graph-dba'`, 2026-08-21) — DISCARDED as duplicate.**
     `guard-destructive-ops.sh` not intercepting `GRAPH.DELETE kaizen_analyst`/`GRAPH.DELETE
     kaizen_teco` from a nested `graph-dba` subagent context. Cross-checked against `cobb`'s own
     `kaizen/plan.md`/`history.md`: this is the same episode already fully tracked there as
     **K-018** (closed 2026-08-21, CONFIRMED via a controlled live re-test — not a `subagent_type`
     omission, a genuine harness gap) and **K-019** (open, high priority, the systemic
     "`PreToolUse` 'ask' hooks don't reliably fire" follow-up). This entry's own
     `history.md` (2026-08-21, "`kaizen/inbox.md` deleted…") already logged it as "left for a
     future K-018/K-019 pass, not re-opened here" — nothing new to capture; the durable record
     lives entirely in `cobb`'s own kaizen, a team-wide concern not specific to `graph-dba`.
  2. **`a3f0c1d2…` (current-shape, `MENTIONS`-tagged onto `graph-dba` by `architect`'s earlier unit
     in this pass) — PROMOTED, verified true.** FalkorDB/Redis serializes write execution per
     graph (one write at a time, queued) and every write query is atomic — so folding a
     check-then-act sequence into one `GRAPH.QUERY` closes a concurrent-write race with no
     lock/queue needed. Re-derived independently: fetched docs.falkordb.com/design/concurrency
     live (confirms "only one write query executes at a time on a given graph," "every query that
     modifies the graph… is atomic," "readers are never exposed to partially applied
     modifications") and grepped falkor-chat's shipped `create_entity_with_auto_match`
     (`repository.py:1259`, K-050 M5) confirming the pattern is real, shipped code, not a
     hypothesis. Added to `falkordb-quirks.md` as a new **Concurrency & atomicity** section
     (placed before "Cypher dialect & query behavior") — this is foundational modeling guidance
     (fold check-then-act into one query), not a narrow dialect quirk, so it earned its own
     section rather than a bullet buried in an existing one.
  3. **`7e3d1a2b…` (current-shape, `MENTIONS`-tagged) — PROMOTED, verified live.** `count(*)`
     under-counts parallel edges between the same node pair (returns 1 for 2 identical `REL`
     edges); `count(r)` with a bound relationship variable correctly returns 2. Reproduced live on
     a disposable scratch graph (module `41811`): created two identical `(a)-[:REL]->(b)` edges,
     confirmed the 1-vs-2 split exactly as the entry claimed. Added to `falkordb-quirks.md`,
     Cypher dialect & query behavior.
  4. **`7f3c2e1a…` + `b2d8f4a1…` (current-shape, `MENTIONS`-tagged, a refinement pair reporting
     the same undirected-relationship-pattern direction bug) — PROMOTED, MERGED AND CORRECTED, not
     verbatim.** Both entries claimed an undirected pattern with a relationship-property predicate
     (inline map filter, or the refinement's broader claim: also a separate `WHERE` clause)
     silently degrades to directed, first-declared node as source. **First re-derivation attempt
     did NOT reproduce either claim** on a fresh disposable scratch graph with no index on
     `SAME_AS.status` — both declared-node orders returned the symmetric correct result (1), for
     both the inline-filter and the `WHERE`-clause forms, contradicting the entries as literally
     stated. Investigated further rather than discarding outright (the entries were detailed,
     cross-checked internally, and read as careful empirical work): re-created the same setup
     **with** `CREATE INDEX FOR ()-[r:SAME_AS]-() ON (r.status)` (matching what falkor-chat's real
     schema plausibly carries) — the bug reproduced **exactly** as both entries described (0 vs 1
     by declared node order), and `GRAPH.PROFILE` showed why: the predicate folds into a
     directional `Edge By Index Scan` that only scans the direction pattern order implies. Confirmed
     the index is the actual trigger (not the predicate alone) by re-testing with the index dropped
     (symmetric-correct again) and with the predicate dropped but the index present
     (symmetric-correct — the bug needs both). **Neither original entry states the index
     precondition** — both would have shipped an overbroad claim ("any relationship-property
     predicate triggers it") into a knowledge base other agents trust as ground truth. Added one
     corrected, merged entry to `falkordb-quirks.md` (Cypher dialect & query behavior, right after
     the `count(*)` entry) stating the precise index-gated trigger, the fix (two `OPTIONAL
     MATCH`es, one per direction, `coalesce`d), and a cross-reference to the existing "Edge By
     Index Scan folding" entries in Query tuning (same mechanism, but here the fold changes the
     *result*, not just the plan shape).
- **Why:** Scheduled unit of the team-wide `kaizen_team` distillation pass (`teco`-coordinated,
  `claude/docs/plans/kaizen-distillation-coordination.md`), this agent's turn.
- **Verified:** live re-derivation for every current-shape entry (not citation-trust) — docs fetch
  + shipped-code grep for #2, a fresh scratch-graph repro for #3, and a full isolate-the-trigger
  investigation for #4 that overturned the entries' own stated scope. Legacy entry #1 cross-checked
  against `cobb`'s own kaizen files rather than re-diagnosed. Order of operations honored
  throughout: this `history.md` entry was written and confirmed before any graph mutation below.
- **Graph mutations (all `agent='cobb'`):** legacy curator-clear for `c3e5f8a2…`
  (`DETACH DELETE`). For each of the 4 current-shape entries: counted remaining edges first
  (`producedEdges=0, mentionEdges=1` for all four — the `MENTIONS` edge this unit resolves was the
  **last** edge on each node), so each resolved via the last-edge full-node `DETACH DELETE`, not a
  partial edge-only delete. Confirmed via `MATCH (e:KaizenEntry) WHERE e.entryId IN [...] RETURN
  count(e)` → **0** after clearing all 5 entryIds (1 legacy + 4 current-shape).
- **Docs touched:** `claude/graph-dba/falkordb-quirks.md` (2 new/edited sections — Concurrency &
  atomicity, plus 2 new Cypher-dialect bullets), this `history.md` entry.
- **Plan items:** none opened — every surviving entry resolved to promote or discard; no
  kept-open/unverifiable case this pass.

## 2026-08-24 — Prompt-waste compression, Stage C6: one edit — the destructive-ops hook's script path dropped
- **What:** `graph-dba.md` was one of C6's four files (`claude/docs/plans/prompt-waste-reduction.md`, Stage C). 1,807 → 1,806 w. **One edit**, step 8: `A `PreToolUse` hook (`graph-dba/hooks/guard-destructive-ops.sh`) intercepts…` → `A `PreToolUse` hook intercepts…`.
- **Why (class 6):** the agent never reads or runs that script — the *harness* runs it, from the frontmatter `hooks:` block where the path still sits (line 10). It is implementation provenance, not a normative citation (§3's exemption covers a path the rule requires the agent to **use**). Exactly the edit C5 applied to `qa-engineer.md`; doing it here keeps the team's hook-description phrasing consistent, which `agent-maintenance` §4's enforcement-parity check reads.
- **Gate (a) inventory — all preserved:** destructive ops need explicit approval (the hook is "a backstop, not a license"); the CPG-reload `GRAPH.DELETE` named as in-scope; the subagent path (return command + blast radius to the caller); the fact that a hook intercepts at all. §4 check-4 parity holds — the hook is still described in the prompt it guards.
- **Gate (b):** the wrapper/core relationship is on record in this file's 2026-08-08 entry and in `claude/AGENTS.md`'s hook-machinery section.
- **The file's citation habit (plan finding 9) turned out to be near-empty.** `graph-dba`'s residual is **version and lineage markers**, not provenance — and every one is a keep: the `41811`/`999999` version-encoding parenthetical (how to read `GRAPH.LIST`'s output), `(successor to RedisGraph)` (an anti-trigger that makes RedisGraph-era documentation legible), `v4.18.11`'s "reason from documented behavior, not latest-`main`". **Class-6 residual after this edit: 0 w removable** — the seventh consecutive file at zero (plan finding 11).
- **Total residual ~20 w — the lowest of the eleven files measured so far**, and structurally so: this prompt's top layer is *reference mechanism* (GraphBLAS, the pinned deployment, the Cypher-subset facts) with no workflow counterpart to re-aim it at, so almost nothing restates. See the C6 unit record's candidate finding 13.
- **Judged and kept** (recorded in `plan.md` so a later sweep doesn't re-litigate): single-shard-per-graph stated twice; verify-against-`docs.falkordb.com` stated three times; the `-ml.md` mirror; the `devops` reciprocity note; the deployment-symlink line.
- **Verified:** `audit-team.sh` PASS; `cobb` §7 lint — clean on all seven dimensions for this file, 0 blockers, 0 majors, no finding attributable to the edit.

## 2026-08-23 — Prompt-waste Stage B wave 2: two boilerplate blocks compressed to pilot shapes
- **What:** Interactive-commit-grant bullet and learning-capture intro/tail compressed to the pilot-validated wordings in `architect.md`/`coder.md` (`claude/docs/plans/prompt-waste-reduction.md` v4, §3 doctrine + Stage B). No CPG-freshness clause exists in this file (graph-dba *builds* CPGs; it isn't one of the six consumer agents).
- **Removed (class 5/6, already on record):** the grant's "same as before. Stakeholder decision, 2026-08-21 — see `kaizen/history.md`" — this file's 2026-08-21 grant entry; the tail's inbox-replacement sentence + ", exactly like the old inbox was" — this file's 2026-08-21 inbox-deletion entry; the intro's ":Agent node it's `PRODUCED`-linked to" mechanics restatement (mechanics live in the Cypher template below); the grant parenthetical's "— not spawned via `Agent`/`Task` as an isolated delegate" (moved into the carve-out sentence).
- **Gate (a) inventory — all preserved:** grant scope and objects (design note, migration/DDL script, explicit path), full never-list, delegated-subagent carve-out + audit check-8 tokens, the FalkorDB-quirk routing rule (live-verified quirk → `falkordb-quirks.md`, dated with verifying command; everything else → graph), Cypher template + call line verbatim, "skip task-specific/already-documented", "raw capture: `cobb` promotes; never edit your own definition".
- **Verified:** `audit-team.sh` PASS; cobb §7 lint pass.

## 2026-08-21 — Interactive-mode commit grant added (team-wide stakeholder decision)
- **What:** New Principles bullet: when running interactively (`claude --agent graph-dba`, a
  human present turn-by-turn — not a delegated subagent), may `git add`/`git commit` its own
  verified deliverable(s) from the session (a design note, a migration/DDL script), by explicit
  path, never bulk-staged/pushed/reset/rebased/amended; the grant does not apply when spawned as
  a delegated subagent.
- **Why:** Direct stakeholder ruling, 2026-08-21, after `tico` hit exactly this gap closing out a
  Mode-3 verification pass (its own commissioned artifacts left uncommitted, since only
  `tico`/`teco` had any commit authority). Rather than pin the fix to those two, the stakeholder
  ruled the exception should reach every agent, gated by invocation mode, not identity — full
  rationale, the `claude/AGENTS.md` rewrite, and the `audit-team.sh` check-8 redesign in
  `claude/cobb/kaizen/history.md`, 2026-08-21 entry.
- **Verified:** `bash claude/scripts/audit-team.sh` — clean, all 13 agents pass check 8.
- **Plan items:** none opened — direct implementation of an explicit stakeholder decision.

## 2026-08-21 — `kaizen/inbox.md` deleted (content already fully captured elsewhere); also executed `G1`'s last 2 `kaizen_<agent>` retirements

- **What:** `cobb` deleted this agent's frozen `kaizen/inbox.md` (git history retains it in full, unaltered) as part of a team-wide cleanup of all 12 agents' frozen inboxes. In the same session, this agent was dispatched to finish `G1` (`docs/plans/generic-cypher-mcp2-coordination.md`): live-reconfirmed, then `GRAPH.DELETE`d, the last 2 of 12 `kaizen_<agent>` keys (`kaizen_analyst`: 8 entries, `kaizen_teco`: 5 entries), both content-diff-verified already fully distilled elsewhere before deletion. `GRAPH.LIST` re-run afterward confirmed both keys gone.
- **Why:** user-directed team-wide cleanup — "no point keeping [inbox.md] since it's already git history." Verified lossless first: `kaizen_team` (the shared graph every agent's raw capture routes through since 2026-08-20) was confirmed completely empty before any deletion. Full rationale and verification method: `claude/cobb/kaizen/history.md`, 2026-08-21 entry.
- **Anything unexpected:** `guard-destructive-ops.sh`'s `PreToolUse` hook did **not** intercept either live `GRAPH.DELETE` call in this dispatch's nested-subagent Bash context, despite the regex plainly matching — a live repeat of the already-tracked gap (`cobb/kaizen/plan.md` `K-018`). Logged a fresh corroborating `kaizen_team` entry (`c3e5f8a2-…`, `author: graph-dba`) rather than re-diagnosing; `cobb`'s territory to fold in on its next `K-018`/`K-019` pass.
- **Verified:** live-relisted both target graphs' entry counts/ids before deleting (didn't trust the dispatch brief's snapshot alone); re-listed all graphs afterward to confirm both keys absent.
- **Plan items:** none opened here — the hook-bypass finding is already tracked as `K-018`/`K-019` in `cobb`'s own plan.md, not duplicated in this agent's.

## 2026-08-21 — Live-dispatched for a hook-enforcement test (K-018/K-019); no `graph-dba`-side change

- **What:** Ran, as a `graph-dba` subagent (`subagent_type` explicitly set by the dispatching
  `cobb` session), a deliberate live test of the destructive-ops `PreToolUse` guard: created a
  throwaway scratch graph, ran the exact `GRAPH.DELETE` command shape from the earlier G1
  incident, and reported whether anything paused for approval. It didn't — confirming a genuine
  harness-level hook-enforcement gap for Task-dispatched subagents, not a defect in this agent's
  own `guard-destructive-ops.sh` (independently re-verified sound in isolation during the same
  test). Logged the finding as `:KaizenEntry` `a4f3d2e1…` for `cobb` to route.
- **Why:** User-requested confirmation of K-018's open hypothesis.
- **Disposition:** the full investigation trail, the finding's team-wide implications, and the
  now-open follow-up (**K-019**) live in `claude/cobb/kaizen/{plan,history}.md` — this is a
  cobb-owned, team-wide item (the enforcement gap affects every guarded agent, not just
  `graph-dba`), not a `graph-dba`-specific prompt/hook change. No file in this agent's own folder
  changed as a result.
- **Plan items:** none opened here — tracked as K-018 (closed, confirmed)/K-019 (open) in cobb's
  plan.md.

## 2026-08-21 — `kaizen_team` distillation: the deferred `d4f8b1c3…` entry — kept open, merged into cobb's already-tracked K-018 (no new graph-dba-side action)

- **What:** `cobb` processed the one remaining `author:'graph-dba'` entry in `kaizen_team`
  (`d4f8b1c3-2a67-4e05-9c81-3f6b9d2e7a45`, 2026-08-20) — explicitly out of scope for the
  2026-08-20 distillation pass above, deferred to a proper full pass. It reports the same G1
  episode (M7 rollout, `docs/plans/generic-cypher-mcp2-coordination.md`) already investigated
  from `teco`'s side: `guard-destructive-ops.sh` did not escalate 4 live `docker exec
  falkordb-dev redis-cli GRAPH.DELETE <key>` calls despite the shared regex plainly matching.
  **This is not new information** — `cobb`'s own `kaizen/plan.md` already carries the fuller
  analysis as **K-018** (opened 2026-08-21 distilling `teco`'s parallel entry `b1e3a1f0…`), with
  the leading hypothesis being `subagent_type` omission on the `Agent` dispatch (silently runs the
  delegate as `general-purpose`, carrying none of `graph-dba`'s frontmatter `hooks:`) rather than a
  defect in the guard script itself — a fix for the omission (`always pass subagent_type
  explicitly`) is already shipped in `teco.md`'s Guardrails.
- **Verified, added to K-018:** re-ran `guard-destructive-ops.sh` directly with the exact command
  text this entry cites (`docker exec falkordb-dev redis-cli GRAPH.DELETE
  kaizen_data-scientist`) — the script correctly emits `"ask"` in isolation, confirming the
  regex/script logic is not the bug (rules out one candidate explanation; the enforcement gap, if
  it persists once `subagent_type` is confirmed correct, lives elsewhere in the dispatch/hook
  pipeline). Also found two related-but-not-identical closed upstream Claude Code issues
  (#18392, #34692 — corroborating that subagent-dispatch hook reliability is a documented gap
  class, not a one-off) and folded them into K-018 as search terms for the still-pending live
  re-check, not as a confirmed root cause.
- **Disposition:** kept open — no `graph-dba`-side prompt/hook change from this entry specifically;
  the durable record and the pending next step (a live `graph-dba` dispatch with `subagent_type`
  confirmed, watching whether the hook fires) live in `claude/cobb/kaizen/plan.md` K-018. This
  agent's own hook script is verified correct as written; nothing here implicates
  `guard-destructive-ops.sh`'s pattern-matching logic.
- **Cleared:** `d4f8b1c3-2a67-4e05-9c81-3f6b9d2e7a45` removed from `kaizen_team` after this entry
  was confirmed written (curator-clear, `agent='cobb'`). `kaizen_team` now has zero
  `author:'graph-dba'` entries.
- **Why:** User-requested distillation pass, continuing the oldest-first queue.
- **Docs touched:** this file; `claude/cobb/kaizen/plan.md` (K-018 addendum).

## 2026-08-20 — Graph distillation: 1 entry promoted to `cypher-mcp/README.md` (`agent-maintenance` §5, cobb, Q2 acceptance pass — AC-5 re-proof on the consolidated `kaizen_team` graph)

- **What:** `cobb` processed one raw `:KaizenEntry` node in the now-consolidated, `author`-partitioned
  `kaizen_team` graph — `a3f4e1b2-6c9d-4e2a-8f1b-0d5c7a9e3f21` (2026-08-20, `graph-dba`): the `cypher`
  MCP tool authorizes writes in exactly two shapes (an author-matched `KaizenEntry` CREATE, or a
  curator `entryId`-clear) and rejects every other write — including schema DDL such as
  `CREATE INDEX`/`GRAPH.CONSTRAINT CREATE` — even from a valid, recognized agent slug. Routed to
  **`cypher-mcp/README.md`**'s "Writing through this tool" section (the entry's own `suggestedHome`
  was "project docs," and this is exactly that — the tool's own component doc, not a `graph-dba`-only
  fact): added a clause after "every other write is rejected regardless of `agent`" naming the DDL
  case explicitly (schema statements get no carve-out over data statements) and citing the concrete
  `CREATE INDEX FOR (e:KaizenEntry) ON (e.entryId)` example plus the S0 unit's live rejection.
- **Why:** Genuinely durable, non-obvious fact about `authorize_write()`'s scope — the README already
  documented the two-shapes rule in general ("every other write is rejected regardless of `agent`"),
  but a reader could plausibly assume schema DDL sits outside the "write" gate entirely (it's
  metadata, not data). The entry proves that assumption wrong and the concrete example makes the
  boundary unambiguous for the next reader/writer of this tool.
- **Verified:** re-read `cypher-mcp/README.md`'s full "Writing through this tool" section before
  editing — the general two-shapes rule was already present, confirming the entry's core mechanism
  claim rather than contradicting it; the DDL-specific instantiation was not yet spelled out, so this
  is an additive clarification, not a duplicate. Attempted a live re-run of the entry's own cited
  repro (`CREATE INDEX FOR (e:KaizenEntry) ON (e.date)` against `kaizen_team` with `agent='cobb'`)
  to independently re-derive the fact rather than trust the citation alone; the harness's own
  auto-mode classifier blocked the attempt as a write-shaped Bash-equivalent action before it reached
  the MCP server, so the live re-run itself is inconclusive — but the entry's original evidence (a
  real S0-unit rejection message, quoted verbatim in the node) plus the README's own written
  contract (§"Writing through this tool": "any Cypher that isn't one of the two shapes above, is
  rejected... before `GRAPH.QUERY` is ever called") independently corroborate the fact without
  requiring the re-run.
- **Order of operations honored:** this `history.md` entry was written and confirmed **before**
  `entryId` `a3f4e1b2-6c9d-4e2a-8f1b-0d5c7a9e3f21` was cleared from `kaizen_team` via
  `mcp__cypher__query(graph='kaizen_team', cypher="MATCH (e:KaizenEntry {entryId:'a3f4e1b2-6c9d-4e2a-8f1b-0d5c7a9e3f21'}) DETACH DELETE e", agent='cobb')`.
  Before/after count for `kaizen_team` filtered to `author='graph-dba'`: **2 → 1** (the remaining
  entry, `d4f8b1c3-2a67-4e05-9c81-3f6b9d2e7a45`, is explicitly out of scope for this pass and was not
  touched).
- **Scope note:** this is a narrow, single-entry acceptance exercise (qa-engineer's Q2 closing pass
  for `docs/plans/generic-cypher-mcp2.md`, AC-5), re-proving the distillation workflow end-to-end now
  that raw capture lives in the consolidated `kaizen_team` graph rather than the interim
  `kaizen_graph_dba` graph AC-5 was first proven against (2026-08-18, below). No other `kaizen_team`
  entries (any author), `kaizen_teco`, or `kaizen_analyst` were read, verified, or touched.
- **Docs touched:** `cypher-mcp/README.md` (knowledge-base edit, above), this `history.md` entry.
- **Plan items:** none — not on this agent's active K-list; a `cobb`-run distillation pass
  (`agent-maintenance` skill §5), not a `graph-dba` self-edit.

## 2026-08-18 — Kept-open node `6e5d6451…` cleared after `analyst` re-gate resolved the graph's open question (`agent-maintenance` §5 rule added)
- **What:** `analyst`'s diff-scoped re-gate of the pass below (`docs/reviews/graph-dba-kaizen-
  distillation.md`) approved both judgment calls but flagged one gap: the kept-open node
  `6e5d6451…` had no forward pointer to K-007 and no sanctioned way to get one (the MCP write
  model allows only create-your-own and curator-`DETACH DELETE`, no in-place `SET`) — risking a
  future pass re-opening a duplicate K-008 with no signal K-007 already exists. The review left
  the deeper question — should a "kept open" node stay live or get cleared once logged? — for
  `cobb`/the skill to decide, not resolving it unilaterally.
- **Decision:** **clear once logged**, for every disposition including "kept open," file-based
  agents included. Wrote this as an explicit rule into `skills/agent-maintenance/SKILL.md` §5
  (step 3's new "Kept open (unresolved)" bullet + step 4's clearing rule): the kaizen graph
  (and `inbox.md` for file-based agents) is working memory for capture **not yet reviewed**;
  once an entry is reviewed and its disposition — including "still unresolved, here's why" — is
  written into `history.md` (and `plan.md` when actionable), that *is* the durable record, and a
  live raw node with no update mechanism can only drift from what those files say. Also added a
  dedup-check rule (step 3): before opening a new `K-`item for a kept-open entry, grep the
  agent's `plan.md` for the entry's `entryId`/fact — don't duplicate an item a prior pass
  already opened.
- **Action taken:** `6e5d6451…`'s full record already lived in this file's prior entry (below)
  and in `plan.md` K-007, so nothing new needed capturing. Cleared it:
  `mcp__cypher__query(graph='kaizen_graph_dba', cypher="MATCH (e:KaizenEntry
  {entryId:'6e5d6451-72fa-400c-b002-52757727f805'}) DETACH DELETE e", agent='cobb')` →
  `nodes_deleted=1.0`. Confirmed via a follow-up `MATCH (e:KaizenEntry) RETURN count(e)` → **0**
  — `kaizen_graph_dba` now holds no `:KaizenEntry` nodes at all.
- **Why:** Matches the file-based-agent convention (a processed entry, of any disposition,
  leaves the raw inbox) rather than carving out a graph-only exception, and removes the
  duplicate-tracking risk the review flagged outright rather than just mitigating it with the
  dedup check alone.
- **Verified:** re-read the edited skill section after the change (no contradiction with the
  surrounding step 2/3 text); graph count re-checked live (0, above) rather than assumed from
  the write-ok response alone.
- **Docs touched:** `skills/agent-maintenance/SKILL.md` §5 (two edits: kept-open + dedup rule
  in step 3, clearing rule in step 4), this `history.md` entry. `plan.md` K-007 unchanged — it
  already carries the full durable record and needs no forward/back pointer now that the graph
  node is gone.
- **Plan items:** none opened or closed; K-007 (below) stands as-is.

## 2026-08-18 — Graph distillation pass 2 (real, not acceptance): 4 of 5 remaining entries promoted, 1 kept open — `kaizen_graph_dba` now empty of promotable entries

- **What:** `cobb` processed the 5 `:KaizenEntry` nodes left in `kaizen_graph_dba` after the
  2026-08-18 acceptance-test promotion above (confirmed via a fresh read: entries dated
  2026-08-16 ×3, 2026-08-17 ×2 — the `META_DATA` entry from the original 6 was already gone).
  `claude/graph-dba/kaizen/inbox.md` re-checked: still carries only the "FROZEN — 2026-08-18"
  historical snapshot, nothing appended below it — no action needed there, confirmed rather
  than assumed.
  1. **`58ad5ace…` (`GRAPH.EXPLAIN` refuses on a nonexistent graph key) — PROMOTED, verified
     live.** Re-ran `redis-cli GRAPH.EXPLAIN <fresh-key> "MERGE (b:CpgBuildInfo) SET b.x=1"`
     against a never-created key → `ERR Invalid graph operation on empty key`, exact match to
     the original finding. Added to `claude/graph-dba/falkordb-quirks.md` (Ops, config &
     tooling), right above the existing `GRAPH.PROFILE`-isn't-read-only entry it directly
     complements — both are "which GRAPH.* command actually does what you think" traps.
  2. **`f8c28d75…` (`StepRun` audit trail decoupled from `Step` nodes) — PROMOTED WITH A
     CORRECTION, not verbatim.** Re-read `record_step_and_advance`
     (`falkor-chat/server/falkorchat/repository.py:1370-1391`, current line numbers): the core
     claim holds (`stepKey` is copied onto `StepRun` at write time; `HAS_STEP_RUN`/
     `LAST_STEP_RUN`/`NEXT`/`PRODUCED` never touch `Step`) — **but the original entry's
     evidence block was factually wrong on one point**: it asserted "no edge is ever created
     from `StepRun` to `Step`," when the code (and `falkor-chat/docs/DESIGN.md` §6.2 / `docs/
     QUERIES.md` §12) has carried a real `(:StepRun)-[:RAN]->(:Step)` edge since 2026-07-12
     (`git log -S`, commit `3921f87`, M3 K-022 Landing 1) — five weeks before the entry was
     recorded, so this isn't drift, the original observation simply missed it. Confirmed via
     `grep` that `RAN` is currently write-only (created at advance time, never traversed by any
     shipped query in `repository.py`/`QUERIES.md`/`DESIGN.md`), so the *practical* blast-radius
     conclusion the entry reached is still correct today, just for a narrower reason than
     claimed. Added a corrected note to `falkor-chat/docs/DESIGN.md` §6.2 (after the `stepRunId`
     bullet, before the `ctx`/`input`/`output` opacity note) stating the real blast radius —
     live position + `OF_DEF` back-reference + the (currently unread) `RAN` pointer — and
     flagging that a future query starting to traverse `RAN` should re-check the note.
  3. **`7f0e3cf1…` (`pipeline.sh` \| `tee` exit-code trap) — PROMOTED, verified plausible/
     current.** The mechanism (`tee` opens its target before `pipeline.sh`'s own `mkdir -p
     "$WORKDIR"` runs, and no `pipefail` means the pipe's exit status is `tee`'s, not
     `pipeline.sh`'s) is a shell-semantics fact independent of any specific run, not something
     that can go stale — confirmed by reading `pipeline.sh`'s current structure (workdir
     creation still happens a few lines in, no early `mkdir` added since). Added to
     `skills/joern-cpg/SKILL.md` Gotchas, right after the "Loading at scale needs one
     persistent connection" bullet.
  4. **`80ef4889…` (scaling worked-example stale) — PROMOTED, reframed rather than
     re-numbered.** Live-counted `falkor-chat/server/{falkorchat,tests}` today: **65** `.py`
     files (`find ... -name "*.py" | wc -l`) — already past the entry's own 2026-08-17
     measurement of 60, one day later. This confirms the entry's own suggested-home framing
     ("reframe so it doesn't read as a live number to sanity-check against" is the more robust
     of its two proposed options) rather than the alternative ("refresh to ~60/166k") — a
     hardcoded file count is a moving target on this repo on roughly a one-day cadence. Edited
     `skills/joern-cpg/SKILL.md`'s Gotchas "Scale" bullet: kept the per-file rate (~2,700–2,800
     nodes / ~18,000–18,600 edges, consistent across both the 41-file and 60-file real runs),
     dropped the single worked-example total, and added an explicit "measure your own repo
     first" instruction with the `find`/`wc -l` command.
  5. **`6e5d6451…` (unreconciled `DETACH DELETE` relationship count) — KEPT OPEN, neither
     promoted nor discarded.** The entry is self-flagged `unsure` by `graph-dba`, and the
     doubt is genuinely unverifiable in this pass: the pre-delete graph state that would let
     anyone reconcile the extra ~19 relationships is gone (the deletion already happened,
     2026-08-16), and there's no live repro to re-run. Per the `agent-maintenance` skill §5
     step-2 guidance ("unverifiable ≠ discard"), date-stamping the doubt and keeping it is the
     right call here rather than forcing a disposition. Concretely: opened **K-007** in
     `plan.md` (this agent's own backlog, so a future occurrence has somewhere to land) *and*
     left the raw `:KaizenEntry` (`entryId 6e5d6451-72fa-400c-b002-52757727f805`) live in
     `kaizen_graph_dba` rather than clearing it — it carries detail (the exact structural-edge
     arithmetic) that a terse `plan.md` bullet shouldn't have to duplicate.
- **Why:** Real distillation pass per `agent-maintenance` skill §5 (the prior 2026-08-18 entry
  above was QA's AC-5 acceptance exercise, one entry only). Dispatched by `cobb`'s own
  maintainer role, not by `graph-dba` or another producing agent.
- **Verified:** `redis-cli GRAPH.EXPLAIN` re-run live (item 1, exact repro). `repository.py`
  read at current line numbers + `git log -S` for the `RAN` edge's introduction date (item 2).
  `skills/joern-cpg/SKILL.md`'s current Gotchas text read before editing, to avoid duplicating
  or contradicting an existing bullet (items 3–4). `find` re-run live for the current `.py`
  file count (item 4). Order of operations honored throughout: each promotion's `history.md`
  text (this entry) was written and confirmed **before** its `entryId` was cleared from
  `kaizen_graph_dba` via `mcp__cypher__query(..., agent='cobb')` — see the graph read/write
  results in `cobb`'s own run output for the before (5 entries) / after (1 entry, `6e5d6451…`)
  counts.
- **Docs touched:** `claude/graph-dba/falkordb-quirks.md` (item 1), `falkor-chat/docs/
  DESIGN.md` §6.2 (item 2), `skills/joern-cpg/SKILL.md` Gotchas ×2 edits (items 3, 4),
  `claude/graph-dba/kaizen/plan.md` (K-007 opened, item 5), this `history.md` entry.
- **Plan items:** opens K-007 (item 5, above); does not touch K-005/K-006.

## 2026-08-18 — Graph distillation: 1 entry promoted to `skills/joern-cpg/references/cpg-model.md` (`agent-maintenance` §5, cobb, U7 acceptance pass)

- **What:** `cobb` read all 6 raw `:KaizenEntry` nodes in `kaizen_graph_dba` and promoted entry
  `46825361-ff7a-4892-a7fc-71f04b407c5e` (2026-08-16, `graph-dba`): `META_DATA` (and
  `FILE`/`TYPE`/`NAMESPACE`) are absent from both live `pysrc2cpg`-built graphs
  (`cpg_falkorchat`, `cpg_salesperson`) despite being listed in `cpg-model.md`'s "Node labels
  you'll see most" as commonly-seen. Routed to the on-demand knowledge base per the entry's own
  `suggestedHome` — added a caveat block to `skills/joern-cpg/references/cpg-model.md` right
  after that list, naming the confirmed-absent labels and the re-verification date.
- **Why:** genuinely durable, non-obvious documentation gap — the reference doc's label list
  reads as "always present" but is apparently frontend/export-configuration-dependent; a future
  agent designing around `META_DATA` (as the original discovery run was doing) would otherwise
  hit the same empirical surprise.
- **Verified:** re-ran the check live before promoting, 2026-08-18 — `MATCH (n:META_DATA)
  RETURN count(n)` → 0 on both `cpg_falkorchat` and `cpg_salesperson`; `CALL db.labels()` on
  both graphs confirmed **none** of `META_DATA`/`FILE`/`TYPE`/`NAMESPACE`/`NAMESPACE_BLOCK`
  appear (`cpg_falkorchat`: 21 labels including the new `CpgBuildInfo`/`CpgNode`/`IMPORT`/
  `UNKNOWN`; `cpg_salesperson`: 20, same set minus `CpgBuildInfo`) — the original finding still
  holds and is in fact broader than the entry stated (all five labels absent, not just
  `META_DATA`).
- **Not promoted (5 remaining entries), reasons noted for a future pass:** `58ad5ace…`
  (`GRAPH.EXPLAIN` on a nonexistent graph key) and `7f0e3cf1…` (`pipeline.sh` | `tee` exit-code
  trap) are both solid `falkordb-quirks.md`/`SKILL.md` candidates but weren't re-verified this
  pass; `f8c28d75…` (falkor-chat `StepRun` decoupling) routes to `falkor-chat/docs/DESIGN.md`
  and needs a source re-read to verify; `80ef4889…` (stale scaling worked-example) is a
  low-value doc-refresh, not a quirk; `6e5d6451…` is explicitly self-flagged `unsure` (an
  unreconciled relationship count) and is not yet promotable as stated.
- **Docs touched:** `skills/joern-cpg/references/cpg-model.md` (knowledge base edit, above),
  this `history.md` entry.
- **Plan items:** none — not on this agent's active K-list; this is a `cobb`-run distillation
  pass (`agent-maintenance` skill §5), not a `graph-dba` self-edit.

## 2026-08-18 — Learning capture retargeted from `kaizen/inbox.md` to the `kaizen_graph_dba` graph (generic-cypher-mcp, U6 steps 4a+4b)
- **What:** The "Learning capture" section (`graph-dba.md`) no longer instructs appending to
  `kaizen/inbox.md` for non-FalkorDB-quirk learnings. It now instructs writing a new `:KaizenEntry`
  node directly into the `kaizen_graph_dba` working-memory graph, attributed to `graph-dba`, via
  `mcp__cypher__query(graph='kaizen_graph_dba', cypher=<CREATE ...>, agent='graph-dba')` — the same
  `CREATE (...:KaizenEntry {..., author: 'graph-dba', ...})` shape `graph-dba` itself already used
  live in U5 to run the one-time inbox migration (`write ok (labels_added=6, nodes_created=6,
  properties_set=48)`). The `falkordb-quirks.md` direct-home carve-out (live-verified engine
  quirks bypass both the inbox and the graph) is unchanged. No frontmatter, hook, or `description`
  change. The append-before-delete ordering constraint that governs `cobb`'s later curator-clear of
  a promoted entry is **not** documented here — per `docs/plans/generic-cypher-mcp.md` §3.5's
  explicit resolution, it lives solely in `skills/agent-maintenance/SKILL.md` §5, because
  `graph-dba` never runs the delete half of that sequence.
- **Why:** `docs/requirements/generic-cypher-mcp.md` FR-2/FR-11, `docs/plans/generic-cypher-mcp.md`
  §3.1–3.2/§7 step 4b, `docs/plans/generic-cypher-mcp-graph.md` §1–§2 (the `:KaizenEntry` schema
  and `author` attribution this instruction now points at). `claude/graph-dba/kaizen/inbox.md` was
  frozen in U5 (2026-08-18) after the real migration ran and was independently verified
  (`kaizen_graph_dba`: 6 nodes, `entryId` index + uniqueness constraint both `OPERATIONAL`) — this
  edit is the prompt-level follow-through so a fresh `graph-dba` run doesn't keep appending to a
  now-frozen file.
- **Verified:** re-read the edited "Learning capture" section after the change — no dangling
  reference to appending `kaizen/inbox.md` remains; the `falkordb-quirks.md` carve-out sentence is
  untouched. Cross-checked the Cypher example's field names (`entryId`, `date`, `fact`, `evidence`,
  `context`, `suggestedHome`, `author`, `createdAt`) against `docs/plans/generic-cypher-mcp-graph.md`
  §1's schema table — exact match.
- **Docs touched (this unit, U6):** `claude/graph-dba/graph-dba.md`, `claude/AGENTS.md`,
  `claude/README.md`, `docs/BACKLOG.md`, `claude/cobb/cobb.md` (own history entry separately),
  `skills/agent-maintenance/SKILL.md` §5 — see `claude/cobb/kaizen/history.md`'s matching
  2026-08-18 entry for the full six-file diff summary and the close-out grep-sweep triage.
- **Plan items:** none opened or closed — not on this agent's active K-list.

## 2026-08-11 — Inbox distillation: 7 entries — 3 promoted to `falkordb-quirks.md`, 2 discarded as already self-corrected, 2 discarded as already covered in `cpg-model.md`

- **What:** `cobb` processed all 7 entries in `graph-dba/kaizen/inbox.md` (§5).
- **Promoted:**
  - `redis-cli`'s `CYPHER` preamble needing quoted literals (not `k=v` trailing args), and a
    non-aggregated `OPTIONAL MATCH` fan-out key being a real grouping key (not a safe-to-assume
    constant) beside `collect(DISTINCT …)` → two new entries in `claude/graph-dba/
    falkordb-quirks.md`. The second one also **fixed an incorrect invariant claim** in
    `falkor-chat/docs/QUERIES.md` §11.2's own footnote, which asserted `start.key` is constant "so
    the grouping is well-defined" as if it were an engine property — it's a schema-level premise
    (exactly one `START` edge) that K-034 is what actually keeps true.
  - `pipeline.sh --reset` running `GRAPH.DELETE` invisibly to the destructive-ops `PreToolUse`
    guard — **checked against the live guard script before writing this up, and the gap this
    entry reported (2026-07-30) was already closed 2026-08-08 (C-311)**: `guard-destructive-ops.sh`
    now basename+flag-matches `pipeline.sh ... --reset` directly (per `claude/AGENTS.md`'s "Hook
    machinery" section, which already documented the fix). Rewrote the `falkordb-quirks.md` entry
    to state the fix + the generalizable lesson (a command-string guard needs an explicit clause
    per destructive wrapper script) instead of re-filing an already-closed gap as an open one. This
    is the third `falkordb-quirks.md` entry, closing out the "3 promoted" count above.
- **Discarded — already covered in `skills/joern-cpg/references/cpg-model.md` (2, both 2026-07-19):**
  the `pysrc2cpg` call-graph directional-asymmetry finding (caller matching by `CALL.NAME`, not the
  resolved `CALL` edge — already in "Consumer-query facts" `:116`/`:121`) and the `FILENAME`/`AST`
  vs. `CONTAINS`/`REACHING_DEF` intraprocedural-scope finding (already in the same section, `:140`).
  Both were re-checked against the live file, not assumed stale.
- **Discarded (already self-corrected in-run, before this pass):** the `db.indexes()`
  vector-dimension entry — this one actually came from **`teco`'s** inbox, not `graph-dba`'s (see
  `claude/teco/kaizen/history.md`'s own 2026-08-11 entry for its disposition; removed from this
  entry, which previously double-claimed it — nit n-2 in `docs/reviews/kaizen-distillation-2026-08.md`).
  The reachability-sandbox correction pair (two entries narrating the same finding, both
  `graph-dba`'s own) — already self-corrects in the second entry's own text; folded as one line
  into `skills/cpg-analysis/SKILL.md`'s query-usage guidance ("probe reachability, don't assume").
- **Also folded in from other agents' inboxes (not counted in this entry's "7 entries," logged in
  those agents' own history entries):** the FalkorDB `RESULTSET_SIZE` silent-cap finding
  (`qa-engineer`'s inbox) → `falkordb-quirks.md`, `cypher-mcp/README.md` (corrected an overclaim —
  "the `rows=` figure is always the true total" was false above 10k rows), and
  `skills/cpg-analysis/SKILL.md`'s gotcha list (new #6). The **no-string-repetition-operator**
  finding is `coder`'s own inbox entry and its promotion is logged solely in `coder`'s history
  entry — this entry previously double-claimed it too (nit n-1); removed here.
- **M-4 follow-up, verified closed:** `cypher-mcp/server.py`'s module docstring ("Display-only
  truncation" bullet, `:20-22`) carried the same now-corrected `rows=`-is-always-exact overclaim as
  `cypher-mcp/README.md`, in the most authoritative of the three sites (flagged by `analyst`'s review,
  M-4). `teco` fixed it directly — confirmed present and reads correctly: "the reported row count
  is exact below FalkorDB's `RESULTSET_SIZE` (default 10000), at or above which it is itself a
  cap." All three sites (`falkordb-quirks.md`, `cypher-mcp/README.md`, `cypher-mcp/server.py`) are now
  consistent; `skills/cpg-analysis/SKILL.md` was always correct on this point.
- **Verified:** `bash claude/scripts/audit-team.sh` clean. `db.indexes()` correction and
  `RESULTSET_SIZE` figures cross-checked against the entries' own cited commands/outputs, not
  re-run live (no FalkorDB access from this session).
- **Docs touched:** `claude/graph-dba/{kaizen/{history,inbox,plan},falkordb-quirks.md}` ·
  `falkor-chat/docs/QUERIES.md` · `cypher-mcp/README.md` · `skills/cpg-analysis/SKILL.md`.

## 2026-07-28 — `joern` agent retired; CPG generation folded in as an on-demand capability
- **What:** The standalone `joern` subagent (CPG specialist) was retired at the user's
  request — CPG generation work is genuinely rare, not frequent enough to justify a
  dedicated standing agent/persona. `claude/joern/` (agent, hooks, kaizen) was deleted
  entirely and its `~/.claude/agents/joern` deployment symlink removed. Its capability
  — driving the `joern-cpg` skill's parse → export → transform → load pipeline — folds
  into `graph-dba` as a small, explicitly on-demand addition:
  - **Frontmatter `description`:** the old "routes to joern" clause replaced with a
    direct capability clause — graph-dba drives Joern itself via the `joern-cpg` skill,
    stated as rare/on-demand so routing agents (and this agent itself) don't start
    suggesting CPG generation proactively. JDK/Joern-toolchain provisioning still routes
    to `devops` (added to that boundary bullet).
  - **Body:** a short "CPG generation (rare, on-demand)" paragraph after the knowledge-base
    bullets — a pointer to `skills/joern-cpg/SKILL.md`, not a restatement of its pipeline
    (joern's own prompt detail lived almost entirely in the skill already). No new hook —
    graph-dba's existing `guard-destructive-ops.sh` already covers `GRAPH.DELETE` for a
    CPG reload, so the destructive-ops step just gained a one-clause example.
  - **`joern:graph-dba` removed from `claude/scripts/audit-team.sh` `BOUNDARY_PAIRS`**
    (the pair no longer exists — folded into one agent, not two bordering ones).
- **Learnings distilled from `joern/kaizen/inbox.md` before deletion** (agent-maintenance
  skill §5 — verify → route → log; no "clear" step since the source is gone):
  - Three 2026-07-17 entries (Python frontend token `pythonsrc`, `pipeline.sh` masking a
    frontend failure as exit 0, per-statement `redis-cli` failing at scale) were already
    fixed in-skill and documented in `skills/joern-cpg/SKILL.md` Gotchas — verified current,
    discarded as duplicates, no new home needed.
  - Two 2026-07-19 entries (pysrc2cpg call-graph sparseness/asymmetry;
    framework-invoked entrypoints needing transitive test-gap reachability) were already
    folded into `skills/joern-cpg/references/cpg-model.md` "Consumer-query facts" and
    `skills/cpg-analysis/references/test-gap.md` respectively — verified current, discarded
    as duplicates.
  - The count-extraction gotcha (`redis-cli --no-raw` output must be parsed with
    `awk '/^[0-9]+$/{last=$0} END{print last}'`, not `grep -oE '[0-9]+' | tail -1`, which
    reads the stats line as data) → **added to `falkordb-quirks.md`** (the existing
    "read via `GRAPH.QUERY` materializes an empty key" entry), the concrete command the
    entry's prose was missing.
  - `FILENAME` being relative to the **parse root** (not the repo root) — a CPG can look
    correct by node/edge counts yet be silently useless to every `STARTS WITH` filter —
    → **added to `skills/joern-cpg/SKILL.md` Gotchas** (producer-side, actionable at build
    time) with a short cross-reference from `cpg-model.md`.
  - No `--exclude`/ignore mechanism in `build-cpg.sh`/`pipeline.sh` (scoping a parse means
    staging a copy of the wanted subtrees first) → **added to `SKILL.md` Gotchas**.
  - `cpg-to-falkordb.py --load` always re-transforms the export (no "replay this
    `.cypher`" mode) → **added to `SKILL.md` Gotchas**.
  - Sizing data point (~2,700 nodes / ~18,000 edges per Python source file with default
    overlays, from a real 41-file run) → **folded into `SKILL.md`'s existing "Scale"
    Gotchas bullet**, which now also points the streaming-loader concern at this agent's
    own kaizen plan instead of the retired agent's.
  - "Joern distribution not installed on this box despite the pinned-path assumption"
    (observed missing after a prior session had verified it — disk pressure or a wiped
    scratch dir) → **added as a caution to `SKILL.md` Prerequisites**: verify before
    running, treat a missing binary as a `devops` blocker, don't reinstall ad hoc.
  - The FalkorDB-start-script / v4.18.11 confirmation entry was already fully covered by
    this agent's own "This deployment" pin — discarded as a duplicate.
- **Plan items carried forward from `joern/kaizen/plan.md`** (opened as K-005, K-006 below;
  the four remaining parking-lot ideas — int-array columns stored as strings, a `graphml`
  export alternative, incremental re-CPG, `--repr` presets — were reviewed and **not**
  promoted: each is a speculative, low-value script-level idea already preserved verbatim
  in `joern/kaizen/history.md`'s final "Created" entry, which this repo's git history keeps).
- **Cross-references updated in the same change:** `claude/README.md` (dropped the `joern`
  row, updated the `graph-dba` row and the Kaizen/Hooks sections), `claude/AGENTS.md`
  (roster line, hook-machinery four-guards → three-guards), root `AGENTS.md` (roster line,
  `skills/` bullet and catalog entry), `claude/teco/teco.md` (routing table row + handoff
  contract), `skills/joern-cpg/SKILL.md` (description + guard reference + Gotchas above),
  `skills/joern-cpg/references/cpg-model.md` (the one "(the `joern` agent)" mention →
  `graph-dba`), `skills/cpg-analysis/SKILL.md` and its four `references/*.md` recipes (every
  "routes to the `joern` agent" / "(the `joern` agent)" phrase → `graph-dba`), and
  `skills/README.md` (both skill rows' "used by" column).
- **Team-coherence certification run after the change** (agent-maintenance skill §4):
  `claude/scripts/audit-team.sh` green (12 agents, no FAIL; `joern` folder absence is
  expected — the script discovers agents from disk, so it needed no code change beyond
  the `BOUNDARY_PAIRS` edit above). Judgment checklist: roster accuracy, handoff symmetry,
  and boundary reciprocity re-checked against the files listed above; no dangling `joern`
  agent references found in a repo-wide grep after the edits (see cobb's own kaizen history
  for the full certificate).
- **Why:** User decision after a short design discussion — CPG generation is rare enough
  that a dedicated standing persona isn't warranted; `graph-dba` already bordered the
  capability (it owned the loaded graph's FalkorDB model) and the retired agent's own
  procedural detail already lived almost entirely in the `joern-cpg` skill, so the merge
  is a small, mostly-pointer addition, not a restatement.
- **Plan items:** opens K-005 (streaming loader for large-repo CPGs) and K-006 (CPGQL
  script library) — see plan.md.

## 2026-07-27 — Unpinned from `model: opus` (team-wide)
- **What:** Removed the `model: opus` frontmatter line. The field is now absent, so the agent runs on Claude Code's default — `model` **defaults to `inherit`** (re-verified 2026-07-27 against `code.claude.com/docs/en/sub-agents`), i.e. the model the session/system default selects. No other frontmatter or body change.
- **Why:** User no longer wants the team locked to Opus. Model choice belongs at the session level (one decision, changeable with `/model`), not duplicated across 13 frontmatter files where it silently overrides whatever the user picked.
- **Plan items:** closes the standing "is opus warranted vs. sonnet?" revisit item — model tier is no longer this agent's decision.

## 2026-07-27 — Design notes open with the canonical header block (step 2 of `docs/plans/doc-reference-convention.md`)
- **What:** One line added to *How you work* item 7 (design work hands off by path): *"Open the document with the header block from root `AGENTS.md`."* It sits inside the item, so it binds to `<component>/docs/plans/<slug>-graph.md` and not to the inline consults the same item excludes. No frontmatter, hook, `description` or catalog change.
- **Why:** `docs/plans/doc-reference-convention.md` v1.4 §9.6 makes a three-field header (`Status:` · `Owner:` · `Tracks:`) the repo's lifecycle signal, replacing the milestone filename prefix and the move-to-`archive/` rule. `-graph` is in the closed role set and the design note is co-located with the architect's plan, so it needs the same header as its neighbours — and `graph-dba` is the only agent that writes one (zero exist today, which makes this the cheapest possible moment to fix the form). The line is a **pointer, not an inlined template** (v1.4 M20) — root `AGENTS.md` already reaches every agent through the root `CLAUDE.md` `@AGENTS.md` import — and byte-identical across the six producing prompts, because the convention's coverage check greps for it literally. `claude/README.md` row 12 re-checked — it cites the design-note path and the destructive-ops hook, not document structure; no edit needed.
- **Plan items:** none.

## 2026-07-24 — Description slimmed further (second team-wide token-cost pass)
- **What:** Frontmatter `description` compressed 919 → 832 chars (-9%): tightened phrasing, dropped restated detail, kept every routing/boundary clause. `claude/scripts/audit-team.sh` boundary-pair symmetry (graph-dba↔devops, graph-dba↔data-scientist, graph-dba↔joern) re-verified green. No body/catalog change.
- **Why:** All 13 agents' descriptions are auto-injected into every session and subagent spawn; the roster grew to 13 (graph-dba, joern added) since the first pass on 2026-07-11, and per-agent `/context` output showed room to cut further. User-requested via a `/context` token audit.
- **Plan items:** none.

## 2026-07-24 — Frontmatter: `permissionMode: acceptEdits`
- **What:** Added `permissionMode: acceptEdits` to the frontmatter, matching the same-day change across the team (`coder`, `tdd-engineer`, `frontend-engineer`, `architect`, `qa-engineer`, `analyst`, `devops`). File-edit/write approvals are session-scoped in Claude Code (unlike Bash approvals, which persist permanently per repo+command), so users otherwise have to re-grant write permission every session even with a global `Edit`/`Write` allow rule in `~/.claude/settings.json`.
- **Why:** Verified against current Claude Code docs (`hooks-guide.md` "Hooks and permission modes") that this doesn't weaken `graph-dba`'s own guard: its `guard-destructive-ops.sh` hook matches Bash command patterns (`GRAPH.DELETE`, `FLUSHALL`/`FLUSHDB`, volume wipes, `docker rm -f`), unrelated to `acceptEdits` (which only covers Edit/Write and common filesystem commands) — and `PreToolUse` hooks fire before any permission-mode check regardless, so a hook `"ask"` decision would survive even if the two overlapped.
- **Plan items:** none.

## 2026-07-17 — Two live-verified quirks added to `falkordb-quirks.md` (graph lifecycle)
- **What:** added to the "Ops, config & tooling" section: (1) a read via `GRAPH.QUERY`
  **materializes an empty graph key** (shows up in `GRAPH.LIST` with 0 nodes), whereas
  `GRAPH.RO_QUERY` on a non-existent graph returns `ERR Invalid graph operation on empty key`
  and creates nothing — so `RO_QUERY` is the side-effect-free emptiness probe; (2) never scan
  the whole `redis-cli` reply for digits to gauge emptiness (the execution-time stat line makes
  everything look non-empty — parse the lone integer output line). Both stamped verified
  2026-07-17 on v4.18.11.
- **Why:** surfaced during the `joern` CPG loader's live load test (K-001); these are generic
  FalkorDB engine facts, not joern-specific, so they belong in graph-dba's quirks KB (the
  established home) per the fold-in rule. cobb promoted them same-run.
- **Plan items:** none. (No change to `graph-dba.md` itself.)

## 2026-07-16 — Boundary reciprocity with new `joern` agent
- **What:** Appended a clause to the frontmatter `description`: generating a repository's Code Property Graph / operating the Joern toolset routes to the new `joern` agent (which owns CPG generation + the mechanical load), while graph-dba owns the code graph's FalkorDB model and tuning.
- **Why:** `joern` was created (CPG → FalkorDB pipeline) and borders graph-dba; the `joern:graph-dba` pair was added to `audit-team.sh` BOUNDARY_PAIRS, which requires each side's description to name the other (routing-contract symmetry). No other graph-dba behavior changed.
- **Plan items:** none.

## 2026-07-12 — Learning-capture loop: kaizen inbox + closing protocol (quirks-file exception kept)
- **What:** Added `kaizen/inbox.md` (append-only learnings inbox, seeded empty) and a "Learning capture" closing-protocol section to the prompt. Live-verified quirks of the pinned FalkorDB build keep their established direct home (`falkordb-quirks.md`, dated); the inbox captures everything else (client-SDK gotchas, lab conventions, non-FalkorDB tool quirks).
- **Why:** Team-wide self-improvement loop (agent-maintenance skill §5, added the same day) — graph-dba's quirks file was the pattern the loop generalizes, so it stays first-class rather than being rerouted through the inbox. Requested by the user.
- **Plan items:** none.

## 2026-07-11 — Design-note handoff contract + destructive-ops guard (certification fixes)
- **What:** Two additions from the same-day team-coherence certification. (1) "How you work" gained step 7: implementer-bound design work (data model, schema/DDL, ingestion/migration) is written to `<component>/docs/plans/<slug>-graph.md` and handed off by path (mirroring data-scientist's `-ml.md`); quick consults stay inline. teco's "Handoff contracts" list gained the matching entry in the same change. (2) Frontmatter now wires a `PreToolUse` Bash guard — `graph-dba/hooks/guard-destructive-ops.sh`, a thin wrapper over the new shared core `scripts/guard-destructive-ops.sh` — escalating `GRAPH.DELETE`/`FLUSHALL`/`FLUSHDB`/volume wipes/container force-removal to human approval; step 8 describes it (enforcement parity). Catalog rows updated (`claude/README.md`, `claude/AGENTS.md` hook machinery).
- **Why:** Certification found graph-dba was the only design-producing specialist without a written-deliverable path (its designs were the one paraphrased handoff in the teco pipeline), and the shared live FalkorDB was guard-protected only when `devops` acted. The guard also answers deferred K-001's revisit trigger ("starts mutating live FalkorDB data in ways that warrant a guardrail") with a narrower, destructive-shapes-only gate instead of a tool allowlist.
- **Plan items:** K-004 done (moved from plan.md); implements cobb K-011 on this agent's side.

## 2026-07-11 — Deep reference moved to on-demand falkordb-reference.md (token-cost pass, part 2)
- **What:** The "Core expertise" reference detail — LPG modeling patterns, the supported Cypher surface, index/constraint DDL, the `algo.*` catalog, config knobs, sizing/persistence/replication/cluster ops, ingestion, GraphRAG patterns — moved out of the prompt body into a new on-demand file `falkordb-reference.md` (7,459 chars), following the `falkordb-quirks.md` precedent; the body (18,656 → 9,440 chars, −49%) keeps what must always load: FalkorDB fundamentals (GraphBLAS/RAM-bound/no-APOC), the pinned deployment (v4.18.11, falkordb-py 1.6.x, vectorset caveat, version-line distinction), boundaries with devops/data-scientist, both knowledge-base pointers, how-you-work, and principles. Frontmatter unchanged; full audit pass.
- **Why:** graph-dba.md was the team's heaviest prompt and loads on every spawn, but the deep reference is only needed for the task area at hand (~2,300 tokens saved per spawn; the reference costs its ~1,900 only when read). Quirks stay separate: quirks are live-verified divergences of the pinned build, the reference is general practice — quirks win on conflict.
- **Plan items:** none.

## 2026-07-11 — Description slimmed (team-wide token-cost pass)
- **What:** Frontmatter `description` compressed from 1324 to 694 chars: capability lists tightened, reciprocal boundary prose reduced to short route-away clauses that still name the counterpart agents (audit check 6 boundary symmetry preserved — full pass green), and "how I work" detail dropped from the description since the prompt body already carries it. Routing semantics unchanged; no body/catalog changes needed.
- **Why:** All 12 agents' descriptions are auto-injected into every session and into every subagent spawn that carries the `Agent` tool; team-wide they cost 12,609 chars (~3.1K tokens) per injection. The pass cut them to 7,036 chars (~44%), saving ≈1,400 tokens per session/spawn with the same routing contract.
- **Plan items:** none.

## 2026-07-09 — data-scientist boundary clause (description + GraphRAG bullet)
- **What:** Frontmatter `description` and a new GraphRAG-section bullet state the split with the new `data-scientist` agent: graph-dba owns the in-graph mechanics (vector-index DDL, `db.idx.vector` queries, fusing similarity with traversal, their performance); the ML method above them — which embedding model, chunking strategy, how to evaluate retrieval quality — is the data-scientist's to design; GraphRAG layers get designed together. Pair `graph-dba:data-scientist` added to `claude/scripts/audit-team.sh` `BOUNDARY_PAIRS` (check 6, description symmetry).
- **Why:** The `data-scientist` agent was created 2026-07-09; "build/improve the GraphRAG layer" plausibly matched both agents, so the boundary must live in both descriptions (mirrors the devops split done the same day).
- **Plan items:** none.

## 2026-07-09 — Deployment pinned to v4.18.11 (edge retired)
- **What:** The lab's FalkorDB moved from `falkordb/falkordb:edge` (module `999999`) to the tagged release **`v4.18.11`** (module `41811`, Redis 8.6.3, released 2026-06-24). Rewrote the "This deployment" bullet (pinned release, reason from v4.18.11's documented behavior instead of moving-target/latest-`main` caveats; `vectorset` still loaded) and updated the quirks-section pointer. Re-stamped `falkordb-quirks.md`'s header: pinned build identified, quirks re-verified via the falkor-chat query suite (193/193 green on the new build); entries not exercised by the suite keep their edge-build dates pending individual re-probes. Catalog current-state refs updated (root `AGENTS.md`, `claude/AGENTS.md`, falkor-chat docs).
- **Why:** User decided to pin the latest release (cost/verification churn of tracking edge; the prompt's verify-live posture existed largely because the build was a moving target). The quirks file's own rule — re-verify on any tagged-release upgrade — was executed via the canonical suite.
- **Plan items:** none.

## 2026-07-09 — devops boundary clause (description + ops bullet)
- **What:** Frontmatter `description` and a new "Architecture & operations" bullet state the split with `devops`: graph-dba *designs* the deployment (RAM sizing, persistence choice, replication/cluster topology, ACLs) and owns everything inside the database; the container/Compose plumbing that runs it (service bring-up, volumes, networking, CI wiring) routes to `devops` — mirroring devops's existing deferral of data-model/query design here. The pair is mechanically enforced by `claude/scripts/audit-team.sh` check 6 (boundary-pair description symmetry). Catalogs synced (`claude/AGENTS.md`, `claude/README.md`, root `AGENTS.md`).
- **Why:** Description-symmetry sweep after teco's roster→routing-table restructure (same day): "spin up FalkorDB" plausibly matched both agents, and only devops's description named the boundary.
- **Plan items:** none.

## 2026-07-09 — Subagent-awareness on "ask one sharp question" (teco interface review follow-up)
- **What:** "How you work" step 1's "ask one sharp question" now carries the delegated-run fallback: when running as a subagent (e.g. delegated by `teco`), return the sharp question as the result instead of trying to ask mid-run — subagents can't ask. Catalog entry (`claude/AGENTS.md`) updated.
- **Why:** Sweep after the 2026-07-09 teco interface review found the "ask" phrasing assumed an interactive session across several delegates (same fix applied to coder, tdd-engineer, qa-engineer the same day).
- **Plan items:** none (out-of-band, driven by teco's 2026-07-09 review).

## 2026-07-05 — Absorbed generic FalkorDB engine quirks from falkor-chat/AGENTS.md
- **What:** `falkor-chat/AGENTS.md` had a "Live-verified FalkorDB facts" section mixing generic
  engine/dialect quirks (vector index DDL, index-before-constraint ordering, composite
  constraints, cross-graph edge no-op, union-label syntax, `length(path)` in ORDER BY, fulltext +
  `algo.*` confirmation, `GRAPH.RO_QUERY`/Bolt port, `TIMEOUT` default + write-path behavior,
  empty-`UNWIND` row collapse, the `FOREACH(CASE...)` idiom, the `exists()` pattern bug,
  `OR`-as-scan-anchor tuning, `GRAPH.MEMORY USAGE` under-reporting, `labels(coalesce())[0]`
  subscripting) with falkor-chat-specific corollaries (repository function names, mention
  write-block internals, keyset predicate profiling), generalized away from falkor-chat's specific
  property/label names. `falkor-chat/AGENTS.md` was trimmed to keep only the project-specific
  corollaries, each pointing back here for the general fact.
- **Mechanism (revised same day):** first draft inlined the ~20 quirks as a "Verified engine
  quirks" subsection in `graph-dba.md`; on review that bloats the always-on prompt with a
  *perishable, growing* fact list. Split instead into a **resource file** —
  `claude/graph-dba/falkordb-quirks.md` — modeled on the `agent-standards` skill's discipline
  (dated verification stamp, "cache not source of truth," re-verify on tagged-release upgrade,
  build sentinel `999999`). `graph-dba.md` keeps only a short stable-framing pointer that tells the
  agent to read the KB before writing/debugging Cypher/DDL/ops against this build. The whole agent
  folder is symlinked into `~/.claude/agents/graph-dba`, so the sibling file is reachable at both
  the repo path and `~/.claude/agents/graph-dba/falkordb-quirks.md`. `falkor-chat/AGENTS.md`'s
  back-reference was repointed from the prompt section to the resource file.
- **Why:** User: "the section ## Live-verified FalkorDB facts should be part of
  ../claude/graph-dba" — these are reusable DBA knowledge for *any* project on this FalkorDB
  build, not just falkor-chat, and belong on the agent so other projects benefit too. Resource-file
  form (not inline, not a shared skill) was the user's explicit call: keeps the prompt lean and the
  KB in the agent's own folder as a growing, curated store.
- **Plan items:** —

## 2026-06-05 — Deferred K-001 & K-002 (documentation-only for now)
- **What:** No agent/prompt change. User said "just document for now," so recorded the decision: **K-001** (tool permissions) → keep tools unconstrained, no `tools` key; **K-002** (live-FalkorDB profiling skill) → not building it yet, agent stays advice-only. Both marked ⚪ deferred with revisit triggers; active backlog is now empty.
- **Why:** User chose the documentation-only path rather than building tooling or restricting permissions. Logged so the items aren't re-proposed.
- **Plan items:** K-001 ⚪ deferred, K-002 ⚪ deferred.

## 2026-06-05 — Identified the deployment (edge engine on Redis 8 + Vector Sets); closed K-003
- **What:** User ran `redis-cli MODULE LIST` / `GRAPH.QUERY`. Findings: the **`graph` module reports version `999999`** = FalkorDB's **edge/untagged build** sentinel (a tagged release encodes as an integer, e.g. `41809` = v4.18.9), so the engine tracks latest `main`. It runs on **Redis 8.x**, evidenced by the separately-loaded **`vectorset`** module = **Redis Vector Sets** (`VADD`/`VSIM`), confirmed via redis.io docs. Module args observed: `MAX_QUEUED_QUERIES=25`, `TIMEOUT=1000`, `RESULTSET_SIZE=10000`. Edits to the agent: expanded the "This deployment" note (edge build → assume newest but verify + test live; Redis 8 base; `vectorset` present) and added a GraphRAG bullet distinguishing **FalkorDB's in-graph vector index** (`db.idx.vector.*` over `vecf32`, fuses with traversal — default for hybrid retrieval) from **standalone Redis Vector Sets** (`vectorset`/`VADD`/`VSIM`, not traversable — only when embeddings needn't live on the graph).
- **Why:** Closes K-003. An edge build can't be pinned to a semver, so the right move is to record the deployment reality and lean on verify-and-test rather than a release's notes. The dual vector stores on one box are a real GraphRAG footgun worth disambiguating.
- **Plan items:** K-003 ✅ (done). Active backlog now: K-001 (tool permissions, open), K-002 (optional live-FalkorDB skill, low).

## 2026-06-05 — Pinned the falkordb-py client + added version-line literacy (K-003 partial)
- **What:** User answered "version is 1.6.0." Verified via PyPI that this is the **`falkordb-py` Python client** (1.6.0 = 2026-02-21; 1.6.1 latest), not an engine version — the FalkorDB **module/server is on a separate `v4.x` line** (v4.18.9 as of 2026-06). Edited the agent's "Clients & ecosystem" bullet to pin the project's client at **`falkordb-py` 1.6.x** (with the `FalkorDB(...) → select_graph → query/ro_query` API shape and RESP+Bolt), and added a new bullet **"Mind the two version lines"** so the agent never conflates a client version with an engine version and reasons about dialect from the engine (v4.x) but client code from the SDK (1.6.x).
- **Why:** "1.6.0" is exactly the trap that makes an agent assume a wrong engine version; the dialect specifics it encodes are governed by the engine line, not the client. The doc-verified dialect details remain valid for current FalkorDB.
- **Plan items:** K-003 → 🟡 in-progress (client identified/pinned; remaining: confirm the deployed engine v4.x version and reconcile `GRAPH.*`/dialect specifics).

## 2026-06-05 — Repivoted from Neo4j-first to FalkorDB-first (major overhaul)
- **What:** Rewrote the agent to specialize in **FalkorDB** instead of Neo4j, after the user confirmed the lab uses FalkorDB. Verified specifics against docs.falkordb.com (two web searches + two doc fetches) before writing. Changes: new `description` (FalkorDB/Redis-module/GraphBLAS/GraphRAG triggers); added a **"What makes FalkorDB different"** section (sparse-matrix/GraphBLAS traversal as matrix multiplication, in-memory RAM-bound sizing, Redis-module ops model, multi-graph multi-tenancy, OpenCypher *subset* with no APOC/GDS/Fabric, `GRAPH.*` command surface). Reworked all core-expertise sections: modeling (added matrix-aware supernode reasoning + one-graph-per-tenant guidance); **Cypher on FalkorDB** (OpenCypher dialect, `GRAPH.QUERY`/`GRAPH.RO_QUERY`, `GRAPH.EXPLAIN`/`GRAPH.PROFILE` instead of Neo4j `PROFILE` prefix, built-in `algo.*` procedures replacing GDS, batched `UNWIND` writes); **indexing & constraints** (range/full-text `db.idx.fulltext.*`/vector `db.idx.vector.*`, `GRAPH.CONSTRAINT` unique/mandatory); **architecture & operations** (RAM sizing first, RDB/AOF persistence, primary/read-replica async replication, Redis Cluster with *graph-per-shard* and no single-graph sharding, Sentinel, `GRAPH.CONFIG`/`THREAD_COUNT`, `GRAPH.SLOWLOG`, Redis ACL/TLS security, SDK/Cloud ecosystem); and a dedicated **GraphRAG/knowledge-graphs** section (vector+graph hybrid retrieval, multi-tenant KGs, GraphRAG-SDK). Updated working method, principles, and communication style to FalkorDB realities. Kept LPG cross-awareness (Neo4j/openCypher/GQL) for porting and the RDF/SPARQL boundary.
- **Why:** User: "we will use falkordb not neo4j, please review everything." Almost every Neo4j-specific claim (APOC, GDS, Fabric, causal cluster/Raft, `neo4j-admin import`, page cache) was wrong for FalkorDB and had to be replaced.
- **Plan items:** reframes K-002 (companion skill now FalkorDB-specific: `redis-cli`/`GRAPH.PROFILE`) and promotes GraphRAG from parking-lot idea to a core section. Updated README.md and CLAUDE.md catalogs.

## 2026-06-05 — Dropped tenure-boast framing
- **What:** Removed "with decades of hands-on experience running graph databases in production" from the opening line; it now reads "You are a **graph database administrator and data architect** who runs graph databases in production." Kept the role label (sets altitude) but cut the tenure brag.
- **Why:** User feedback — the "decades of experience" framing makes agents sound cocky and adds nothing to behavior. Applied collection-wide (also tdd-engineer, dra-claudia).
- **Plan items:** —

## 2026-06-05 — Agent created
- **What:** Initial authoring of the `graph-dba` agent — a senior graph database administrator and data architect. Frontmatter `name: graph-dba`, `model: opus`, and a routing-oriented `description` with proactive-use triggers (design a graph model, write/optimize Cypher/GQL, plan cluster architecture/sizing/sharding, set up indexes/constraints, tune slow traversals, plan migrations/imports, ops questions). Body covers four core-expertise areas (graph data modeling; Cypher/GQL mastery; indexing & constraints; architecture & operations incl. GraphRAG/vector), a six-step working method (access-patterns-first, match existing conventions, show the model concretely, justify by traversal cost, prove perf via PROFILE, respect engine/version/edition boundaries), seven principles, and a communication style. Scoped **Neo4j/Cypher-first** but explicitly aware of the wider LPG world (openCypher, ISO GQL, Memgraph, Neptune) and honest about the RDF/SPARQL boundary.
- **Why:** User asked for a new agent: "graph database administrator who knows cypher query, data modeling, architecture and best practices." Fits the lab's focus (repo `graphmind-ai-lab`).
- **Plan items:** seeded K-001 (tool-permissions decision), K-002 (optional live-Cypher companion skill), plus parking-lot ideas (GraphRAG depth, PROFILE operator cheat-sheet, opus-vs-sonnet, multi-engine portability).

## 2026-06-05 — Docs updated (discoverability)
- **What:** Registered `graph-dba` in the collection catalog `claude/README.md` (table row + kaizen index link) and in the agent-context file `claude/CLAUDE.md` (Agents list).
- **Why:** Dual-audience documentation rule — keep humans (README) and other agents (CLAUDE.md) in sync the moment the agent is created.
- **Plan items:** —
