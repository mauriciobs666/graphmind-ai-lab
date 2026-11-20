# Review — Workflow input CAS (`ctxVersion`), implementation diff gate

> **Status:** active · **Owner:** `analyst` · **Tracks:** K-097

## Scope & verdict

This gate reviews the uncommitted K-097 diff against the gated design note
`falkor-chat/docs/plans/workflow-input-cas-graph.md` (its current revision, 2,520 words). Files
in scope:

- `server/falkorchat/{repository,executor,services}.py`
- `server/tests/test_process_input.py`
- `scripts/test_queries.sh`
- the new file `server/tests/test_workflow_input_cas.py`

The brief names `9fc52662` as HEAD. The actual HEAD is `aac2590b`. `9fc52662` is its ancestor,
and the three commits between them touch nothing under `falkor-chat/`, so the baseline is the
same. I applied root `REVIEW.md` and `falkor-chat/REVIEW.md`. I did not edit any code.

**Checks run** (evidence in `falkor-chat/docs/reviews/workflow-input-cas-evidence/impl/`):

- **Targeted pytest on the live tree.** Ran `test_workflow_input_cas`, `test_process_input`,
  `test_workflow_timers`, `test_executor`, `test_repository` and `test_api`: **608 passed**. All of
  them use `ws:test` and `reference-test`. The live `reference` and every `ws:<id>` were left
  untouched. The full 3,709-test run is the coordinator's, not mine.
- **`./scripts/test_queries.sh`, run three times.** Results: 486/486, 486/486, **485/486**. The red
  assertion is §9.1 `since=4001 includes mn3`, and its own output contains `mn3`. That is a false
  red, explained in F2.
- **Mutation table.** Mutated copies of the package in the scratchpad, never the tree. The runner
  is `mutrun.py`; output is `mutrun_output.txt`. It covers the note's six mutants plus four more,
  and the shell compare-without-`coalesce` mutant ran twice. Every copy was verified byte-equal to
  the tree after each run.
- **A probe test** (`test_k097_review_probe.py`, kept outside the tree): green on the tree, red on
  the sweep argument mutant.

**Verdict: approve with suggestions.** The code matches the note's §2 query and its writer table
exactly. Every mutant in the note's gate goes red as predicted. No `ctxVersion` leaks into a
reply, and a CAS miss is still a clean 409 or `raced`. Two things remain:

- F1: the sweep's version binding is correct but no test pins it. The fix is a one-test addition;
  the probe test above is the verified shape.
- F2: a harness defect that predates K-097. It explains the implementer's unexplained red and also
  makes `assert_not_contains` able to pass falsely. It is not K-097's to fix, but it should be
  filed.

CPG: considered, not relevant — the change is confined to functions whose call sites were already enumerated (Pass 1 CPG cross-check); a diff review needed no new call-graph query.

## Findings

### Major

**F1 — The sweep's `ctx_version` binding is correct, but no test pins it.** Verified by mutation.
`services.py:2920-2922` binds `v=run["ctxVersion"]` as a lambda default. That value comes from
the step 5.1 `get_run`, the same read whose `ctx` the marker is merged onto and whose `atStepKey`
was just re-checked. That is the right read (see the sweep ruling below).

Consider the mutant that re-reads the version when the CAS is built:
`ctx_version=self._repo.get_run(ctx.ws, run_id=rid)["ctxVersion"]`. It leaves **all 486 targeted
tests green** (`mutrun_output.txt`, "sweep: version re-read at CAS time"). Test 4 cannot catch it,
because it injects at the executor-repo CAS, after any re-read. The window is real: a human
re-park between 5.1 and the CAS would let the sweep write its stale `timerFired` merge over the
human's input. Test 3b pins the same binding on the `/input` path; the sweep has no counterpart.

**Fix:** add the sweep counterpart of 3b. A proxy on `services._repo` whose first `get_run`
returns the real row, then runs a human `submit_workflow_input` to a same-step re-park, then
returns the now-stale row. Assert `raced` and `ctx == {"note": "from-human"}`.
`impl/test_k097_review_probe.py` has this exact shape. It passes on the tree and goes red on
the mutant (`mutrun_probe_output.txt`). Add the row to the note's mutant table.

**F2 (predates K-097, not this change's to fix) — `test_queries.sh`'s assert helpers can report
a false red or a false green.** Verified by probe. The script uses `set -euo pipefail`, and both
helpers test `echo "$actual" | grep -qF "$expected"`. When `grep -q` matches, it exits before
`echo` finishes writing. `echo` then dies of SIGPIPE (exit 141), and `pipefail` makes the whole
pipeline false. As a result:

- `assert_contains` sometimes reports a **miss on a present string**. In the probe this happened
  1 time in 2,000 on a small output (`pipefail_probe_output.txt`; 0 on a second run). In a full
  run it happened for real: §9.1 failed with `mn3` visibly in its output
  (`test_queries_run3_false_red.txt`). With 486 assertions per run, one false red in a few runs
  is what this predicts.
- `assert_not_contains` sometimes reports **a pass on a present string**. A check that cannot
  fail is a finding under root `REVIEW.md`. On a large output whose match comes early, the probe
  shows the miss 2,000 times out of 2,000.

This is the most likely explanation for the implementer's unexplained `§EMITTED provenance
written exactly once` red. That assertion is at `test_queries.sh:760`, ahead of the edited §12
block (lines 1020 onward). Its data is written only by the §EMITTED section. The shell mutant
only edits `RESUME_CTX`, which first runs at line 1074. So the mutant cannot reach it, and the
two share no state. Two runs of the shell mutant reproduced exactly the six expected §12.13 reds
and nothing else.

**Fix (route to a backlog item, not to K-097):** use `grep -qF -- "$expected" <<<"$actual"` in
both helpers. `grep -F … >/dev/null` without `-q` also works. The probe was run: both forms
matched present needles 2000/2000 and absent needles 0/2000, on small and large inputs
(`pipefail_fix_probe_output.txt`).

### Minor

**F3 — Two docstrings state facts that are no longer true.** Verified by grep.

- `executor.py:623-624`: "Omitted ⇒ the chat/trigger path, byte-identical in behaviour to
  before". The chat path now bumps `ctxVersion` (`repository.py:3015-3016`). Suggested text:
  "unchanged apart from §12.4's version bump".
- `repository.py:105-110` (`WorkflowRunNotWaitingError`): "the §12.13 … CAS returns zero rows
  for **two** reasons — … does not exist, or … not `waiting`". There is now a third reason: a
  stale `ctxVersion`. The 409 mapping it describes is unchanged, so this is only the count and
  list.

### Nits

- **N1 —** The sweep comment at `services.py:2887-2889` was edited by splicing into an existing
  sentence. It now reads "… refuses it — the exact same merge shape `submit_workflow_input`
  already uses …". Split it into two sentences.
- **N2 —** One run of the executor-guard mutant showed four errors in `test_repository.py`'s cart
  tests (`test_add_to_cart_…`, `test_adjust_cart_item_…`). Neither a re-run of the same mutant nor
  the cart tests on their own reproduced them, so I don't attribute them to K-097. I'm recording
  them in case they recur.

## Rulings on the implementer's deviations

| Deviation | Ruling |
|---|---|
| Test 3 starts with an ordinary input | **Accept.** Starting from version 1 lets test 3 tell "bump" apart from "set to a constant". The `resume_run` bump = constant `1` mutant goes red on tests 3 and 6 (`mutrun_output.txt`); from version 0 test 3 alone would miss it. It changes nothing else the test asserts. |
| New test 3b (injection during validation) | **Accept. It is the most valuable addition.** It is the only test that kills the `/input` argument mutant (version re-read at CAS time): only 3b goes red. Tests 1–4 inject after any such re-read. F1 asks for its sweep counterpart. |
| New test G (`ValueError` guard) | **Accept.** It is the only test that kills the "guard removed" mutant, and it asserts that nothing was written (status, `ctx`, raw version absent). |
| `_IntrudeAfterDriveRepo` forwards `ctx_version` | **Accept, as the note required** (§3: any proxy that overrides the CAS must forward the version). The intruder is a separate actor and reads its own current version, so its CAS wins. That is the premise of K-096's "waiting envelope never read back" test, which stays green. Under the compare-without-`coalesce` mutant it goes red with the rest, as expected. |

## The questions in the brief

- **Does the code match the note?** Yes, character for character: the compare and bump at
  `repository.py:3048-3050`, the bumps at `:3015-3016` and `:3091-3092`, the projection at
  `:3150`/`:3161`, the start queries unchanged, and the return columns still `{runId, status}`.
  The `test_queries.sh` copies match the repo strings.
- **The sweep's read.** Step 5.1's read is the right one to compare against. It is the read
  whose `ctx` the marker is merged onto and whose `atStepKey` was just checked, so its version
  names the exact wait instance both decisions were made on. The `atStepKey` check stays useful:
  it buckets `raced` without a CAS round trip. The version compare is what makes the outcome
  correct. A same-step re-park (the case `atStepKey` cannot see) is refused by the version. The
  binding is lambda-default, so it is evaluated once per candidate inside the loop. It cannot
  capture a later candidate's `run`. Only the test pin is missing (F1).
- **K-096 interaction.** No leak:
  - `/input`'s reply is built field by field (`services.py:2542-2545`): `runId`, `status`,
    `ctx`, plus `error`.
  - `ctx` comes from `merged` or, on `failed`, from `_load_json_dict(run["ctx"])`
    (`:2680`, `:2751`). Neither path can carry the `get_run` dict's `ctxVersion` key.
  - The start reply (`:2448`) and the chat resume reply (`:2765`) are built the same way.
  - A CAS miss becomes `executor.resume → None`, then `_drive_or_fault` returns `(None, None,
    None)`, and `_drive_and_report` passes it through unchanged (it only reads back on
    `"failed"`). The result is 409 with nothing written, or `raced` in the sweep (`:2935`).
  - Test 6's refused stale submit checks the "nothing written" half.
- **Mutant table** (all from `mutrun_output.txt`; "3b" is the new test):

  | Mutant | Red | Matches the note |
  |---|---|---|
  | compare dropped | 1, 2, 3, 3b, 4, 6 | yes |
  | §12.13 bump dropped | 1, 2, 3b, 4, 5, 6 | yes (superset) |
  | `resume_run` bump dropped / = constant 1 | 3, 6 | yes |
  | `fail_run` bump dropped | 6 | yes |
  | compare without `coalesce` (Python) | 38 tests incl. 1–6 | yes |
  | compare without `coalesce` (shell, twice) | the six §12.13 cases, nothing else | — |
  | `get_run` without `coalesce` | 39 tests incl. 1–6 | yes |
  | `/input` version re-read | 3b | — (deviation) |
  | sweep version re-read | **none** | F1 |
  | executor guard removed | G | — (deviation) |

- **Absolute words.** An unfiltered grep of `falkorchat/*.py` for the claims the diff relies on
  found two stale ones (F3). The claim "the only writers of an existing run's ctx are
  `resume_run_with_ctx` and `fail_run`" (`services.py:2662-2663`) holds: every `ctx` write is in
  `repository.py`, and `start_run`/`_untriggered` create runs rather than write to existing ones.
  "Every exit from `waiting` bumps" (the new test module's docstring, and the note's §4 wording
  for DESIGN) holds for every production path. At the query level, though, `complete_run`
  (§12.5) has no status guard and does not bump. It could move a `waiting` run to `done`, but
  its only caller is `_drive_loop` after a CAS it won. That is harmless (terminal runs fail every
  compare), but the spec sentence should say "every resume" instead (next section).

## Spec sentences for the close

The note's §4 list is complete against the shipped diff: DESIGN §6.2's five items; QUERIES §12
model block, status-move contract, §12.13, and one term each in §12.4, §12.5 `fail_run` and
§12.7. I checked every item against the code above. Two refinements:

- **The DESIGN invariant bullet (note §4).** Write "every resume (§12.4, §12.13) and every ctx
  write to an existing run bumps it" rather than "every exit from `waiting`". The second wording
  is an absolute that `complete_run` breaks at the query level (see the absolute-words bullet
  above).
- **SERVER.md needs no edit. Confirmed.** Its route table cites §12.7 for
  `GET /workflow-runs/{runId}` (`SERVER.md:312`) and nowhere lists that route's keys. The
  "Response bodies" paragraph (`:438-440`) lists only the two POST bodies, and those gain no key.
  The error-map row "lost the resume CAS (nothing written) → 409" (`:427`) still covers a version
  miss. The public addition is therefore documented only through QUERIES §12.7. That makes the
  `HISTORY.md` entry the note requires ("names it as a public contract addition") the one place
  a reader learns about it, so the close must not drop it.

## What's solid

- The diff is minimal and lands in exactly the places the note names, with no drift. The start
  queries are untouched.
- The tests are written red-first against the harness cases and use real `Repository`/`Services`
  objects. Test 6 asserts the raw property, so `coalesce` cannot hide a mutant. Every mutant the
  note lists is killed.
- The implementer went past the note in the right direction: 3b and G each kill a mutant nothing
  else kills.
- The `test_queries.sh` additions include both a positive control (the fresh-version CAS wins)
  and the ABA replay.

## Proposed REVIEW.md lines

Target: `falkor-chat/REVIEW.md`.

- Under **Always check** (source: this review's F1, and the `/input` precedent where only the new
  test 3b kills the re-read mutant). *Addition:* "A value a compare-and-set checks (a version, a
  status, a key) is pinned by a test that injects a concurrent writer between the read it came
  from and the CAS, before any later read the code could substitute. A test that injects at the
  CAS itself cannot tell 'carried from the read' from 're-read just before the CAS'."
- Under **Evidence bar** (source: F2). *Addition, to prune when the helper is fixed:* "A single
  unexplained red in `./scripts/test_queries.sh` is first checked against the assert helpers'
  `echo | grep -q` under `pipefail` (SIGPIPE: a false miss in `assert_contains`, a false pass in
  `assert_not_contains`) before it is attributed to the change or dismissed. Until the helper is
  fixed, an `assert_not_contains` on a large output is not evidence."

## U11 gate

This gates the U11 fix to F2: commit `8d629cad`, which changes `falkor-chat/scripts/test_queries.sh`
(13 hunks, +22/−22). I checked the working tree against `HEAD` and the copy matches. The
delegate's evidence is in `falkor-chat/docs/reviews/test-queries-sigpipe-evidence/`. Mine is in
`falkor-chat/docs/reviews/workflow-input-cas-evidence/u11/`. I held `ws:test` for this run.

**Verdict: approve.** Every claim the delegate made reproduces. My own planted helper mutants go
red. My own sweep finds no early-exit pipe left in the script.

**What I re-checked:**

- **Probe, re-run** (`probe_pre_1000.txt`, `probe_head_1000.txt`). I ran the delegate's
  `probe.sh` at N=1000 on two copies. The first is the pre-fix script (`8d629cad~1`), which is
  byte-equal to the delegate's `test_queries.orig.sh`. The second is `HEAD`.
  - Pre-fix: **6,035** wrong verdicts. The delegate's 12,012 was at N=2000, so this is the same
    rate. On the small redis-cli-shaped output, 33/1000 were false reds and 2/1000 false greens.
  - HEAD: **0** wrong verdicts.
  - The fix also closes a second defect the old helpers had. A needle starting with `-`, such as
    `-1`, was parsed as a grep option and gave a wrong verdict every time. `--` now stops that.
- **Planted helper mutants** (scratchpad copies only; `probe_mutA_nodash.txt`,
  `probe_mutB_pipe.txt`):
  - **A** removes the `--` from `assert_not_contains`. The probe goes red with 400 wrong verdicts,
    all on the leading-dash controls.
  - **B** reverts `assert_not_contains` to `echo | grep -q`. The probe goes red with 201 wrong
    verdicts: the large-output false pass, 200 out of 200.

  So the probe can both fail and pass.
- **The suite itself:**
  - Two runs at HEAD: **486/486** each (`suite_head_run{1,2}.txt`). A third run, which restored
    `ws:test` after the bootstrap plant below, was also 486/486.
  - **My own in-suite plant** (`tq_plantA_fresh_version_in_aba.sh`). It gives the ABA replay the
    fresh `ctxVersion=1` instead of the stale 0. Result: **481/486** (`suite_plantA.txt`).
  - The five reds include three `assert_not_contains`: "zero rows", "sets no property" and
    "wrote no ctx". That is the helper kind that used to pass falsely, now failing as it should
    through the fixed helper.
  - The delegate's P1, P2 and P3 outputs show 480, 480 and 485 of 486, each red where its plant
    lands. I read them; I did not re-run them. My own plant covers the same property.
- **Early-exit sweep, with my own grep.** I matched every `|` into an early-exiting consumer: any
  `grep|head|awk|sed|read|tail|cut|wc|sort|uniq|tr|xargs|python|jq|tee` after a pipe, plus
  `grep -q`/`-m`, `head`, `awk … exit` and `sed … q` anywhere.
  - Every `grep -q`, `grep -m1` and `awk … exit` in the script now reads from a here-string.
  - The pipes that remain are `grep -n -m1 … <<< | cut` at 16 sites, `echo | grep -A1 | tail -1 |
    tr` at `:377`, and the bootstrap line at `:114`. Their consumers (`cut`, `grep -A1`
    without `-m`, `tail`) read all of their input, so none of these pipes can SIGPIPE its writer.
  - `:399` and `:417` match only Cypher `|` inside query strings.
  - The delegate's 22 sites match this set. Nothing is missing from it.

**Observations (not blocking U11):**

- **The `|| true` at `:114` is a real defect, but a minor one.** Under `pipefail`, a failing
  `bootstrap_schema.sh` makes the pipeline fail, and `|| true` swallows that, so the suite runs on
  with no schema. I planted a stub that prints `ERROR` and exits 3 (`suite_plantB_bootstrap_fails.txt`).
  - The run reported **13 ✗** across unrelated sections: constraints, channel listing, vector
    queries. One of them, "vector query: m3 not in top-2", passed against the error text.
  - It then aborted at `:377` under `set -e`, with no `Results` line.
  - It does fail loudly (exit 1). It is never a false green over the whole run. The cost is
    misdiagnosis: the real cause shows only as one `ERROR` line above many unrelated reds.
  - **Fix:** capture the output (`boot=$(… 2>&1) || { echo "  ✗ bootstrap failed"; echo
    "$boot"; exit 1; }`), then filter it with the here-string grep.
  - **It does not need its own item.** Fold it into one low-priority shell-harness backlog item
    with the next bullet.
- **Other `falkor-chat/scripts/*.sh` files under `pipefail` still use `| grep -q`.** Most are
  `PING | grep -q PONG`, where the 5-byte producer has finished writing before grep exits, so the
  risk is negligible. Two are multi-line: `verify_catalog.sh:77,82` and `start_demo.sh:289`. Their
  failure mode is a false "missing" (a fail-loud false red, never a false green). This is out of
  U11's scope (`test_queries.sh` only). It belongs in the same backlog item.
- **Pre-existing and kept deliberately:** an order check whose `grep -n -m1` finds no row now
  ends the suite with a non-zero exit under `set -e`, rather than reporting a ✗. The old pipeline
  did the same; the delegate's `probe_other_idioms_out.txt` confirms the "absent needle" rows,
  and the fix preserves that.

## Close check

This is step 3 of the live-spec close. It checks commit `89306a90` (DESIGN §6.2, QUERIES §12)
against two things: the shipped diff `7cff155b`, and the spec close list above. The query
strings in `7cff155b` are identical to the ones this review's first section gated. The only code
differences after that gate are docstrings (F3) and the F1 test. Evidence is in
`falkor-chat/docs/reviews/workflow-input-cas-evidence/close/`.

**Verdict: approve with suggestions.** Every Cypher block matches the shipped code. Every item on
the close list is done. Every absolute statement I checked holds. The one finding is
duplication, and all of its copies are consistent (C1).

- **Every §12 Cypher block matches the shipped code. This was checked mechanically**
  (`cypher_match.py`).
  - The script takes all 17 ```cypher blocks in QUERIES §12 at `89306a90`, strips `//` comments
    and normalises whitespace. It compares each one with every `.query`/`.ro_query` string
    literal in `repository.py` at `7cff155b`.
  - Result: **17/17 match**, each to its same-named method (`cypher_match_close.txt`).
  - **Positive control:** the same script on the pre-close docs (`89306a90~1`) reports exactly
    the four K-097 blocks as mismatches: §12.4, §12.5 `fail_run`, §12.7 and §12.13
    (`cypher_match_control_preclose.txt`). So the check can fail.
- **Close list: every item is done.**
  - **DESIGN:**
    - the `WorkflowRun` property list;
    - the run-model bullet, worded as this review asked ("every resume … and every ctx write
      to an existing run"; `complete_run` named as leaving it unchanged);
    - "Who writes the run `ctx`";
    - the R-1 blockquote, replaced by the closed statement;
    - "Timer release…", which names the step 5.1 read F1 pinned;
    - the design-rule blockquote.
  - **QUERIES:**
    - the §12 model block and the status-move contract;
    - §12.4: Cypher, plus "a refused resume does not bump";
    - §12.5 `fail_run`: Cypher, plus prose;
    - §12.7: Cypher, plus prose;
    - §12.13: heading, opening, comment, Cypher, zero-row contract, PROFILE block, indexes/RAM.
  - SERVER.md is unchanged, which is correct (see "Spec sentences for the close").
- **Absolute statements, checked against the code with an unfiltered `git grep` at `7cff155b`:**
  - "on an existing run, by exactly two queries" (DESIGN): every `ctx` write is
    `start_run`/`start_run_untriggered` (create), `resume_run_with_ctx` or `fail_run`.
  - the bump set and the leave-unchanged set: the only `ctxVersion` writes are at
    `repository.py:3018,3052,3094`.
  - "no version bump" on a refused CAS: the `SET` is gated by the `WHERE`.

  The new "live-verified" claim in §12.13 (a run resumed and re-parked after the read gives zero
  rows and no property set) is backed by the `test_queries.sh` ABA case. My in-suite plant
  (U11 gate) shows that case can go red.
- **Text the update left unchanged** (root `REVIEW.md`):
  - DESIGN §6.2's "`status` as a property" note ("guarded single-query CAS flips … can't
    double-resume") is still true.
  - QUERIES' build-quirks bullet ("Guarded CAS via `WHERE` on the current status value
    (§12.3/§12.4)") is still true.
  - The §12.16 RAM paragraph still names §12.13 as the sweep's write.
  - No sentence still says "R-1", "plus one `SET` term", "remains in use unchanged", "can never
    disagree" or "unbuilt" about this path (grep over DESIGN/QUERIES/SERVER).
- **No change history** in the updated text. The "a `status` compare alone cannot see that
  re-park" sentence states a design reason; it does not narrate a past change.

**C1 (minor): the bump and refusal semantics are stated in several places.**
- The set of queries that bump `ctxVersion` and the set that leave it unchanged appear in:
  - DESIGN §6.2's run-model bullet;
  - QUERIES §12.4's note;
  - QUERIES §12.5's note;
  - QUERIES §12.13's zero-row contract ("no resume (§12.4, §12.13) and no ctx write (§12.5)").
- "Refused, nothing written (no flip, no ctx, no version bump)" appears in DESIGN "Who writes"
  and in QUERIES §12.13.

Root `REVIEW.md` reports a fact stated twice in the spec set. I checked every copy against the
code above, and all of them agree today. **Suggested fix**, at the next ordinary revision rather
than as a blocker: keep the per-query facts in QUERIES (§12.4, §12.5, §12.13), and have DESIGN's
bullet state the invariant and cite them, dropping its enumerated list.

**Still owed at step 5:** the `HISTORY.md` entry has to name `GET /workflow-runs/{runId}`'s new
`ctxVersion` key as a public contract addition. That entry is the only place the addition is
named, because SERVER.md defers the route's shape to §12.7. No K-097 `HISTORY.md` entry exists yet,
which is expected for step 5.

## REVIEW.md proposals (final)

These replace this review's earlier "Proposed REVIEW.md lines" section. Neither earlier line has
been applied to `falkor-chat/REVIEW.md` (checked at HEAD).

- **Drop** the earlier *Evidence bar* SIGPIPE line. U11 fixed the defect it guarded against. It
  was never added, so no prune line is needed. U11 also left **no automated guard**: the
  delegate's `probe.sh` is evidence, and no test, CI job or hook runs it. A prune line would need
  such an enforcer to cite, so none is justified.
- **Replace it** with a narrower line, because the suite cannot police its own helpers. Under
  helper mutant B, the false pass shows up only on large outputs, which the probe exercises and
  the suite does not. Target: `falkor-chat/REVIEW.md`, **Always check**. Source: F2 and the U11
  gate. Verbatim:

  > - A change to an assertion helper in `scripts/test_queries.sh` re-runs
  >   `docs/reviews/test-queries-sigpipe-evidence/probe.sh` against the changed script, and the
  >   review reports its `TOTAL wrong verdicts` (0 required) together with one planted helper
  >   mutant that makes it non-zero.

- **Keep** the F1 line. Target: `falkor-chat/REVIEW.md`, **Always check**. Source: F1 and the
  `/input` precedent, where only test 3b kills the re-read mutant. Verbatim:

  > - A value a compare-and-set checks (a version, a status, a key) is pinned by a test that
  >   injects a concurrent writer between the read the value came from and the CAS, ahead of any
  >   later read the code could substitute. A test that injects at the CAS itself cannot tell
  >   "carried from the read" from "re-read just before the CAS".

- **No prunes.** Neither addition makes an existing line in `falkor-chat/REVIEW.md` redundant.
