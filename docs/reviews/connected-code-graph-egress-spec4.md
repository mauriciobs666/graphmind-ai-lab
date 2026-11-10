# Review: connected code graph egress spec 4 (delta after P1-29 and P1-48b)

> **Status:** active · **Owner:** `analyst` · **Tracks:** C-1001 (P1)

## Pass 1 (2026-11-10)

### Scope and verdict

**Reviewed:** `docs/plans/connected-code-graph-egress-spec4.md`, which is uncommitted, 1,157 words, and carries
`Extends:` spec3 (spec3 carries the matching `Extended by:`).

**Checked against:**
- the four source reviews: `docs/reviews/connected-code-graph-p1-29.md`, `-p1-29-security.md`, `-p1-48b.md` and
  `-p1-48b-security.md`;
- the uncommitted code under `cpg/ccg/broker/` and `cpg/sandbox/offline-run.sh` (mid fix round);
- the whole of spec3, grepped for each contract spec4 narrows;
- root `REVIEW.md` (there is no `cpg/REVIEW.md`).

**Executed:**
- `sbx_idx_contained` was extracted verbatim from the runner and run on planted symlink trees in the scratchpad
  (A1).
- `scip-python` was run with `--project-version` in `ccg-toolchain-indexer:73b9fb76bd2a` under `--network none
  --read-only --cap-drop ALL` (A2).
- The frame-cap arithmetic was computed.
- No FalkorDB was involved and no git write was made.

**Verdict: needs changes.** There is one major (S4-M1): the symlink obligation as written accepts an absolute
symlink that points into the tree by its host path. The spec's own example reason says otherwise. The rest is
minor, and every claim traced to its review and to the code, with the exceptions listed below.

CPG: considered, not relevant — no `cpg_*` graph covers `cpg/`; claims were checked against the reviews and the uncommitted code, read and run directly.

### Findings

**S4-M1 (major, verified). The symlink rule accepts an absolute link into the tree, and gives the wrong reason for
refusing `lnk -> /src/sub`.**

§1 has the runner resolve each value on the host with `realpath -e` and states: "an absolute target resolves on
the host, so it is refused". The runner implements the rule faithfully (`offline-run.sh:269-280`), and two of its
cases contradict that sentence (A1):
- `lnk -> /src/sub` is refused only because `realpath -e` *fails*, since the host has no `/src/sub`. It is not
  refused because the target resolves outside the tree. The refusal therefore depends on an unstated "no result
  means refused".
- `lnk -> <host path of the staged tree>/sub` is **accepted** (hits=0). Inside the container the same link
  resolves against the image's filesystem, not the tree. Today it dangles, because the staged path sits under the
  container's empty `/tmp`, so the tool fails closed. A staging path that also exists in the image would let
  `--cwd` reach image content. The check is meant to prove containment, but it proves it in the wrong namespace.

*Fix:*
- State that a value `realpath -e` cannot resolve (dangling link, missing file) is refused `forbidden-flag`. This
  also makes a missing `env.json` a `forbidden-flag`, not `usage`; say that too, since the runner already reports
  "(or does not exist)".
- Refuse any symlink with an absolute target on the walked path. One way is to `readlink` each component under
  the tree, refusing a target that starts with `/`. The other is to resolve relative to the tree as root.
- Add the absolute-into-tree case to test (22)'s refused list. A runner without the new rule accepts it, so the
  case can fail.

**S4-m1 (minor, verified). "(§7.8 states the same)" is false.** Spec3 §7.8 (output naming, one live invocation
per `--output`, signals) says nothing about a non-zero exit or a partial artifact. §7 of spec4 admits the citation
was taken "on the reviews' word". *Fix:* make "a non-zero indexer exit means no artifact; P1-15 must not read a
partial `<auditId>.scip`" an **addition to §7.8**, drop the citation, and delete §7's second bullet.

**S4-m2 (minor, verified). Precedence between containment and a missing mandatory element is not stated.** §1 says
a forbidden element outranks a missing mandatory one. The symlink check is a `forbidden-flag` refusal, but it runs
"after the grammar passes". The runner reaches it only after the `usage` refusal (`offline-run.sh:912-917`). So an
escaping `--cwd` together with a missing `--project-version` is refused as `usage`, with the escape never
recorded. *Fix:* either say that containment runs only on a complete argv, so `usage` wins there, or move it
before the `usage` refusal. Either way, add the combined case to test (22).

**S4-m3 (minor, verified). The reply-cap formula is off from the code.** §3 writes `ceil(MAX_RESPONSE_BYTES * 4 /
3) + 64 KiB`. The code uses `(MAX + 2) // 3 * 4` (`framing.py:37`), which is base64's exact padded length,
`4 * ceil(MAX / 3)`. At 16 MiB the two are 22,369,624 and 22,369,622: the spec's formula is 2 bytes short of a
full encoded response. The 64 KiB slack hides the gap, but the spec is the normative one. *Fix:* write
`4 * ceil(MAX_RESPONSE_BYTES / 3) + 64 KiB`.

**S4-m4 (minor, verified). Launch-by-ID is scoped as indexer-only, but it applies to every container launch.**
The runner puts `$SBX_IMAGE_ID` in the generic container launcher (`offline-run.sh:505`). Spec3 §7.1's Launch cell
still reads "on `ccg-toolchain:<hash>`", and §7.3's launcher clause and test (23) compare `launcherArgv`s. *Fix:*
say the rule covers every container launch (narrowing §7.1), and that `launcherArgv` carries the `sha256:` ID.
Test (22)'s "a retag between inspect and launch does not change the image" has no named way to interpose a
retag, so as written it may never be able to fail. Replace it with: the image element of `launcherArgv` equals
`toolchain.imageId`, with a mutant that launches by name.

**S4-m5 (minor, verified). Spec3 R-E9's list of children is stale under a mandatory `--project-version`.** With
`--project-version 1`, `scip-python` spawned only `python3 -c …`, not `git`, wrote the index and exited 0 (A2).
Spec3 R-E9 still lists `git rev-parse HEAD` as a child. *Fix:* one sentence in spec4: "with `--project-version`,
`scip-python` 0.6.8 runs no `git`". The same run also wrote an index with `env.json = []`, which bears on R-E9's
"format unmeasured" line (that is P1-15's measurement).

**Nit.** §1 decides that `--project-version` is **mandatory**. The `-p1-48b-security.md` ruling recommended
optional, "so a tree that does carry a revision still works"; `-p1-48b.md` preferred mandatory. Mandatory is
consistent, because §1 also rules that no `.git` is staged, so the optional case never arises. Say that in the
*Why*, so the security reviewer's reason is answered rather than silently overridden.

### Checks requested

- **Claims against the reviews and the code: hold, apart from the findings above.** Each one traces:
  - the grammar and the `[A-Za-z0-9._@+-]` component class (`offline-run.sh:252-312`);
  - the `(missing index)` hit;
  - 17 forbidden and 3 allowed entries: I counted the files' non-comment lines;
  - one request per connection (P1-29 ruling 7);
  - the strict §2.0 ancestor rule (security Pass 2, m1);
  - lone surrogates: the frame is malformed (`framing.py:74-114`);
  - an over-long `payloadFields` key gives `null` plus the digest of the list (`server.py:182-185`);
  - `ensure_ascii` (`audit.py:173`);
  - the `RunGrant.auditId` (`client.py:100`);
  - `close` with `upstream-error` (`server.py:216`);
  - the `codebases-in-vulndb` flag `_cb_declared` (`policy.py:238`);
  - the `requestMode` fallback (`policy.py:212-222`);
  - the IP-literal bypass (P1-29 ruling 4);
  - `payloadFields` for R4 (ruling 10);
  - the `repoId` cap (`policy.py:64`).
- **`--project-version` grammar:** consistent with `--project-name` (same pattern, same two-element and
  at-most-once rules). It is a hit on `scip-typescript`, and it is `usage` when missing, under §1's precedence.
  The pattern admits a 40-hex sha and a 64-hex digest. It excludes `+`, which `-p1-48b.md` had proposed for
  semver build metadata, and nothing needs `+` with a stager-supplied sha.
- **Test (22) can fail:** yes for every case except the retag one (S4-m4), and with the additions in S4-M1 and
  S4-m2.
- **Silences against spec3:** I grepped the whole of spec3 for each narrowed contract: `MAX_RESPONSE_BYTES + 64 KiB`
  (§2.2), the `RunGrant` signature (§2.1), the `payloadFields` wording (§5.2), "13 forbidden, 3 allowed" (§7.4),
  `ccg-toolchain:<hash>` (§7.1) and `git` in R-E9. Each is narrowed by spec4, except the two in S4-m4 and S4-m5.
- **Delegate's open items:**
  - §7.8 citation: false (S4-m1).
  - `target.path`: the code uses the manifest `paths` pattern verbatim with `fullmatch` and a 256 cap
    (`policy.py:67, 188`), the same pattern and `maxLength` as spec3 §4.2. `fullmatch` also avoids the
    trailing-newline `$` trap that spec3's S5 exists for.
  - Lone surrogates: handled at the parser and stated correctly.
- **Open questions: genuinely open.** Spec4 §1–§5 add neither the extra ranges nor a runner timeout (`fec0`
  appears only in §6). Q1 matches the security advisory's deferral (Pass 2, "tied to the §3.4 prefix-list
  question"). Q2 re-asks what spec3 R-E10 settled ("taken on a measured need"); posing it as a question is fine.
- **Growth-stops (new `Extends:` successor):** §1–§5 are new normative sentences, each tied to a ruling, and they
  state only the change. §7, "Not stated from the evidence", is process narration, and its first bullet is
  already stale (the runner now has the §1 changes). Delete §7 once S4-m1 is resolved. Nothing restates spec3.

### What's solid

The delta is small and change-only, and it cites its sources. Each ruling it encodes is the one the reviews made.
The lone-surrogate decision (reject at the parser rather than `surrogatepass`) is the safer design and is stated
precisely. The two open questions are honestly left open.

## Appendix

**A1. Containment probe.** I extracted `sbx_idx_contained` from `offline-run.sh` and ran it on a planted tree
(`<scratchpad>/spec4b/t`). Results, with `src` the resolved tree:

```
/src/root (-> /)            cwd:REFUSED env:REFUSED  realpath=/
/src/up (-> ../..)          cwd:REFUSED env:REFUSED  realpath=<scratchpad>
/src/abs (-> /src/sub)      cwd:REFUSED env:REFUSED  realpath: No such file or directory
/src/sub, /src/, /src       cwd:ok      env:ok
/src/rel_in (-> sub)        cwd:ok      env:ok
/src/dotdot_in (-> deep/x/../../sub) cwd:ok env:ok
/src/dangling, /src/nonexist cwd:REFUSED env:REFUSED (realpath fails)
/src/env.json               cwd:REFUSED env:ok
/src/abs_host (-> <host path of tree>/sub)  hits=0, ACCEPTED
```

**A2. `--project-version`.** I logged children with a `PATH` shim, then ran `scip-python index --cwd /tmp/src
--environment /tmp/src/env.json --project-name p --project-version 1 --output /tmp/o.scip` with `env.json = []`.
It gave exit 0 and "Sucessfully wrote SCIP index". The only child was `python3 -c …`; no `git` ran.

## Pass 2 (2026-11-10)

**Scope:** the revised `docs/plans/connected-code-graph-egress-spec4.md` (1,263 words; Pass 1 saw 1,157), re-checked
against spec3, `cpg/ccg/broker/framing.py` and the uncommitted `cpg/sandbox/offline-run.sh` and tests, plus root
`REVIEW.md` (no `cpg/REVIEW.md`). **The runner is still being revised and may lag or lead the spec; where it
differs I say so and the spec is not failed for it.** No FalkorDB write, no tree-mutating git command.

**Verdict: approve with suggestions.** All Pass 1 findings are fixed. One new minor (S4-N1): "walked path" in the
symlink rule is ambiguous, and no listed case distinguishes the two readings.

CPG: considered, not relevant — no `cpg_*` graph covers `cpg/`; claims were checked by running probes and reading the code directly.

### Pass 1 dispositions

- **S4-M1: fixed, with one residual (S4-N1).** §1 now refuses an unresolvable value (dangling link, missing
  `env.json`) as `forbidden-flag`, and any absolute-target symlink on the walked path; the host-path-into-tree
  case is named in the refused list. I re-ran the Pass 1 planted-tree probe (`<scratchpad>/spec4b/t`): the old
  `realpath -e` predicate accepts `abs_host` (so the new case can fail against a runner without the rule) and
  refuses `root`, `up`, `abs`, `dangling`. The current runner's `sbx_idx_walk` (`offline-run.sh:271-295`, extracted
  and run) refuses `abs_host`, `abs`, `root`, `up`, `dangling` and accepts `sub` and `rel_in`.
- **S4-m1: fixed.** §1 "§7.8 gains: failure means no artifact" is worded as an addition; grep for "states the same",
  "reviews' word" and "Not stated" finds nothing, and the old §7 is gone.
- **S4-m2: fixed.** §1 Precedence puts containment before `usage`; §2 adds the combined case.
- **S4-m3: fixed.** Formula is now `4 * ceil(MAX_RESPONSE_BYTES / 3) + 64 KiB`. Computed at 16 MiB: 22,369,624, and
  `framing.py:37` `(MAX + 2) // 3 * 4 + FRAME_SLACK` gives the same (checked, `FRAME_SLACK = 64 * 1024`).
- **S4-m4: fixed.** §1 "Image by ID, every container launch" says it narrows §7.1's "on `ccg-toolchain:<hash>`"
  (spec3 line 782, the only such text). The replacement test can fail: the runner launches `"$SBX_IMAGE_ID"`
  (`offline-run.sh:532`) and `mutants.sh:158` (J12) swaps in `"$SBX_IMAGE"`, whose `ccg-toolchain:<hash>` cannot
  equal a `sha256:` ID. I read the mutant; I did not run `mutants.sh`.
- **S4-m5: fixed.** §1 *Why* states that with `--project-version`, `scip-python` 0.6.8 runs no `git`, so R-E9's
  `git rev-parse HEAD` child is gone; Pass 1's run (A2) is the evidence. Spec3 R-E9 (line 1265) still lists it, and
  spec4 narrows it, as for every other narrowed contract.
- **Nit: fixed.** The *Why* now answers the optional-versus-mandatory ruling ("no `.git` is ever staged").

### Checks

- **Same-revision consistency: holds.** Spec3 has no explicit forbidden-versus-missing precedence (§7.4 line 866
  and §7.5 line 931 are unordered), so spec4's rule is new and contradicts nothing. Order is: grammar
  forbidden, containment (forbidden-class), then `usage`. The empty argv has no `--cwd`/`--environment`, so
  containment has nothing to check and `(missing index)` is reached first (`offline-run.sh:314`); spec3's "bare
  invocation" is a runner invocation with no arguments, not an empty tool argv. A missing `env.json` as
  `forbidden-flag` does not collide with §7.5's "missing mandatory element", since the element is present.
  Wording nit: §1 says containment runs "after the grammar passes" and also "before the `usage` refusal"; if
  `usage` counts as a grammar failure the two read as contradicting. Say "after the forbidden-element check".
- **Test (22) cases can fail:** each launch, `usage` and `forbidden-flag` case maps to a distinct refusal class or
  an exit, so a runner that accepts, or misclassifies, turns it red. The combined case (escaping `--cwd`, missing
  `--project-version`) fails any runner that refuses `usage` first; that ordering was the Pass 1 behavior
  (`offline-run.sh:912-917` then). Not executed against the current runner, which is mid-revision.
- **Growth-stops:** +106 words, all new scope or rework answering Pass 1 (precedence, image-by-ID, R-E9 and §7.8
  sentences, the extra test cases), and §7 was deleted. No finding table or disposition list, and no restated
  spec3 content. Nit: the refused/accepted symlink list and the escaping-`--cwd` example appear in both §1 and §2;
  §2 is the test enumeration, so I let it stand, but §2 could cite §1.

### New finding

**S4-N1 (minor, verified). "Any symlink on the walked path" does not say whether links met while resolving count.**
§1 prescribes `realpath -e`, which reports no links, then adds the absolute-target rule. Read literally (readlink on
each component of the value), it misses a relative link whose target passes through an absolute one. Planted
`rel_via_abs -> deep/abs_host`, `deep/abs_host -> <host path of tree>/sub`: a per-component `readlink` plus
`realpath -e` **accepts** `/src/rel_via_abs` (resolves to `t/sub`), while inside the container the second hop
resolves against the image. No listed case tells the two readings apart, so a literal runner passes (22) and keeps
the S4-M1 hole. The current `sbx_idx_walk` follows spliced targets and **refuses** it (run), so the runner is ahead
of the spec here.
*Fix:* define the walked path as every link traversed, spliced targets included (the walker's behavior), and drop
`realpath -e` as the mechanism. Add `lnk2 -> deep/abs` (relative to an absolute link) to the refused list.

## Pass 3 (2026-11-10)

**Scope:** `docs/plans/connected-code-graph-egress-spec4.md` as revised after `317baccc` (1,600 words; Pass 2 saw
1,263), against spec3, the uncommitted `cpg/sandbox/offline-run.sh` and `tests/`, the plan rows
(`docs/plans/connected-code-graph-p1.md:316`, `-p1-extraction.md:64,353`) and root `REVIEW.md` (no `cpg/REVIEW.md`).
**The P1-48b fix round lags on the watchdog:** the tree has no `SBX_INDEXER_WALL_S` and no exit-124 path (grep), so
every watchdog claim below is design-level. The regular-file rule is already in the runner. No FalkorDB write, no
tree-mutating git command.

**Verdict: needs changes** (one major, S4-M2: a listed mutant cannot fail). The watchdog design is implementable and
the new text contradicts nothing in spec3 or earlier spec4 rules; the remainder is minor.

CPG: considered, not relevant — no `cpg_*` graph covers `cpg/`; claims were checked by running probes and reading the code directly.

### Pass 2 dispositions

- **S4-N1: fixed.** §1 now defines the walk as "following every link it traversed, including links reached through
  the target of a relative link", drops `realpath -e`, and lists `lnk2 -> deep/abs`. The runner's `sbx_idx_walk`
  (`offline-run.sh:271-295`) does exactly this.
- **Wording nit (containment "after the grammar passes"): fixed;** it now reads "after the grammar check and before
  the `usage` refusal".

### Checks

- **Watchdog implementable: yes, with two spec-wording gaps.** Run: a background timer `( sleep 2; kill -USR1 $$ ) &`
  plus a trap on `USR1` interrupts `wait` on a child that ignores TERM; the handler KILLed `jobs -p` and exited 124
  (scratch script). Read: `sbx_on_signal` (`:752-766`) does the kill/`sbx_remove_own`/`sbx_cleanup` steps but
  `exit`s `128+n` and writes no record, so the watchdog is a variant, not a reuse; `sbx_write_record` (`:646`) reads
  only variables set before the launch (`SBX_LAUNCHER`, `SBX_MOUNTS`, `SBX_IMAGE_ID`) and is idempotent through
  `SBX_RECORD_WRITTEN`, so it can run after the kill and after cleanup. Main guard: `:987`. `lib_launch`
  (`test_sandbox.sh:163-177`) sources the runner and calls `sbx_launch` directly with `SBX_INDEXER=1` from
  `LIB_INDEXER`, so a reassignment after `. "$RUNNER"` reaches it, but `lib_launch` has no such hook today.
- **Production cannot take another cap: confirmed, but by `env -i`, not only by the constant.** I copied the runner,
  printed `${SBX_INDEXER_WALL_S:-unset}` after `sbx_clean_env`, ran with `SBX_INDEXER_WALL_S=3`: it printed
  `unset`. The closed allow-list (`sbx_clean_env`, `:~835-850`) drops it before any code that could read it.
- **Regular-file rule is meetable and its cases can fail.** `sbx_idx_contained` (`:300-307`) uses `[ -f ]` after a
  stat-only walk (no open). Run on a planted tree: regular file hits=0; directory, FIFO, symlink to a FIFO and
  relative link to a directory hits=2. A runner that tests `-e` or `! -d` accepts the FIFO, one that reads the file
  hangs, and the spec's per-test timeout catches it; the regular-file positive control exists. Tests at
  `test_sandbox.sh:1143-1147` already assert it.
- **124 fails `check_sandbox_ids`:** spec3 §5.3 condition 3 requires `exitCode: 0` (line 709), so yes by the text.
  `cpg/ccg/sandbox_audit.py` is not built yet (P1-18), so not executed.
- **Narrowing:** the §7.4 last sentence and R-E10 (spec3 lines 883, 1272-1275) are narrowed precisely: R-E10's
  "TERM to the runner, whose handler removes the container" stays true as the outer path. Gaps are S4-m6 and S4-m7.
- **Same-revision interaction: holds.** Containment, precedence and the regular-file rule are all grammar-time and
  precede any launch; the watchdog exists only after launch. A 124 record carries the same `launcherArgv` image ID as
  any launch. `sbx_remove_own` matches on the auditId label, independent of image naming.
- **§6:** carries only the IPv6/metadata range question, which is still unanswered; the wall-clock question is gone.
- **Growth-stops:** +337 words. New scope: the regular-file rule and its test (~90), the watchdog bullet and test
  (~210), §7 (~70). No finding table, disposition list or restated history. Restated fact: §7's owner, files and
  gates also sit in the two plan rows and `-p1-extraction.md:353` (S4-m8). The "Refused/Accepted" list is one long
  unwrapped line (nit).

### Findings

**S4-M2 (major, verified). The "constant read from the environment" mutant cannot fail.** §2 asserts that a test sets
the variable in the environment of a normal invocation and the run is not shortened. A normal invocation re-execs
under `env -i` and drops any key outside the closed set (run above: `unset`), so a runner that reads
`${SBX_INDEXER_WALL_S:-1800}` is also not shortened and survives, and the check cannot go red (root `REVIEW.md`,
every-check rule). *Fix:* kill it in library mode, where nothing scrubs the environment: export
`SBX_INDEXER_WALL_S=3`, source the runner, assert the constant still equals `1800` (the literal from the spec, not
read from the runner). Keep the normal-invocation run only as a note that `env -i` is a second guard.
*Where it goes:* the assertion in the test round, with the mutant row in `mutants.sh`; not in the watchdog step.

**S4-m6 (minor, verified). The watchdog's record contradicts two spec3 sentences it does not name.** Spec3 §5.3
(line 663) and §7.8 Signals (line 987) say a signalled invocation writes no record. The watchdog uses the signal
path and writes one. §5.3's `exitCode` row (line 686) also reads "the tool's exit code", and 124 is the runner's.
*Fix:* add one clause: the watchdog is not a caller signal and records; INT/TERM/HUP still write none; `exitCode`
may be 124. Also say "the same steps as `sbx_on_signal`", since the handler itself exits `128+n` without a record.

**S4-m7 (minor, inferred). Outer bound has no value or ordering.** "P1-15's own timeout stays as the outer bound",
but neither plan gives it a value (`-p1-extraction.md:353` says only "a wall-clock timeout"). At or below 1800 s
plus the up-to-~10 s `sbx_remove_own` loop the inner watchdog never fires and exit 124 and its record are dead
code; and a TERM-ing outer path yields exit 143 with no record. *Fix:* state that the outer timeout exceeds
`SBX_INDEXER_WALL_S` plus 60 s, and that P1-15 treats runner exit 124 as a timeout failure; a P1-15 test asserts the
ordering. Also require the timer to be cancelled on normal exit (a stale timer signals a runner past its trap).

**S4-m8 (minor, verified). Ownership has several homes and they differ.** §7 restates owner, files and gate, which
the plan rows already carry (`-p1.md:316`, `-p1-extraction.md:64`). §7 says `analyst` and `security-expert`
re-gate; `-p1.md:316` says "`analyst`; `security-expert` advisory". *Fix:* delete §7 and keep the plan rows as the
single home, or keep §7 and cut the rows to a pointer; align the security role either way.

**Nits.** (a) Spec4 says "Section numbers are spec3's" yet cites its own §1 ("§1's no-artifact rule", "watchdog
(§1)") beside spec3's §5.3/§5.4; write "spec4 §1". (b) §1 names a "device" refusal but §2 tests only directory and
FIFO; a device node needs root to plant, so say a device is covered by the same `-f` predicate or add a
fixture-injected case. (c) The TERM-instead-of-KILL mutant is killed only by a hang under the suite's timeout (the
client proxies TERM and `wait` blocks); assert elapsed time under the cap plus a margin so it fails by assertion.

### What's solid

The mechanism (internal timer, trap, KILL client, label-based removal, record, 124) is sound and was exercised in
miniature. The regular-file rule is stat-only, already implemented, and its tests have a real positive control and
a timeout. The no-environment-override claim holds for production. Precedence and image-ID rules compose with the
new items without conflict.

## Pass 4 (2026-11-10)

**Scope:** spec4 as revised for Pass 3 (1,790 words; Pass 3 saw 1,600) and the P1-15 line at
`docs/plans/connected-code-graph-p1-extraction.md:353`. The runner still has no watchdog (grep), so the checks
below run against scratch copies, with the constant added by `sed`. No FalkorDB write, no tree-mutating git command.

**Verdict: approve with suggestions.** All four Pass 3 findings are fixed; no major remains. Two new minors (S4-m9,
S4-m10), each a one-line fold.

CPG: considered, not relevant — no `cpg_*` graph covers `cpg/`; claims were checked by running probes and reading the code directly.

### Pass 3 dispositions

- **S4-M2: fixed, and the test can fail.** §2 now has a library-mode test that exports `SBX_INDEXER_WALL_S=3`,
  sources the runner and asserts the literal `1800`, with its own mutant row. Run on copies: constant assigned
  literally, `rc=0`; mutant `SBX_INDEXER_WALL_S=${SBX_INDEXER_WALL_S:-1800}`, `rc=1`. The spec also says the
  normal-invocation run cannot kill it (`env -i`), which matches my Pass 3 probe.
- **S4-m6: fixed.** The "Exception to spec3 §5.3 and §7.8" sub-bullet names both sentences, admits 124 in the
  `exitCode` row, and says the watchdog is a variant of `sbx_on_signal`. It leaves a race, S4-m9.
- **S4-m7: fixed.** The outer bound is stated (cap plus 60 s; runner exit 124 is a timeout failure; timer
  cancelled on normal exit), and the P1-15 line at `-p1-extraction.md:353` carries it. The margin is sound for any
  responsive daemon, see below.
- **S4-m8: fixed.** Old §7 is gone (headings end at §6); the ownership bullet points at the plan rows, and
  `-p1.md:316` and the P1-48b follow-up say "`analyst`; `security-expert` advisory", matching. One stale citation
  remains, S4-m10.
- **Nits (a)-(c):** the device case is now declared untested (needs root); the TERM-mutant is stated as killed by
  the hang under the suite's timeout, which I accept as stated (not by assertion, as I suggested). Nit (a), "spec4
  §1" wording, is not changed; harmless.

### Checks

- **60 s margin.** `sbx_remove_own` (`offline-run.sh:770-790`) runs at most 50 passes; each pass is two `docker`
  calls (measured 0.28 s together here), a 0.2 s sleep, and up to two `docker rm` calls when something is found. The
  worst case, a daemon that keeps re-creating a labelled container for all 50 passes, is about 25 s without the
  removals and perhaps 40 s with them, so the spec's "about 10 s of sleeping ... several times that" understates
  it. Realistic cost is about 1.5 s (three quiet passes). 60 s is sound for any responsive daemon. None of the
  `docker` calls has a timeout, so a hung daemon blocks past any margin; that is outside the codebase's control.
- **m6 exception against "INT, TERM, HUP write none".** Run: a bash handler for INT (`sleep 2`) was preempted by a
  `USR1` handler arriving mid-run, which then exited 124 (the INT handler never finished). So a caller's signal
  followed by the timer firing inside the cleanup window yields exit 124 *with a record* on a signalled run,
  contradicting the sentence the exception was written to keep.

### New findings

**S4-m9 (minor, verified). First-wins is unstated between the watchdog and a caller signal.** Per the run above, the
handlers nest. *Fix:* one sentence: each handler first disarms the other (the INT/TERM/HUP handler cancels the
timer and ignores the watchdog signal; the watchdog `trap -`s INT/TERM/HUP), so a signalled invocation never
records and an expired one never exits `128+n`. A test would need a timed signal; the runner's gate can carry it.

**S4-m10 (minor, verified). Dangling citation.** `docs/plans/connected-code-graph-p1-extraction.md:64` still reads
"egress-spec4 §1, §7"; §7 no longer exists. *Fix:* cite "egress-spec4 §1".

### Growth-stops

+190 words. New scope: the §5.3/§7.8 exception (~50), the margin and timer sentence (~110), the S4-M2 test (~50),
the ownership pointer (~35); minus the deleted §7 (~70). The margin reasoning is design rationale for a number, not
history. No finding table, disposition list or restated spec3 content; the ownership bullet is a pointer.

**Stopping condition (falsifiable).** Spec4 needs no Pass 5 once a pass yields no major and no finding that
changes what the runner must do (a contract change, as opposed to wording or a citation). Pass 3 failed it (S4-M2,
and m6/m7 were contract text); Pass 4 yields only S4-m9, one sentence of contract, and a citation. Folding S4-m9 and
S4-m10 meets it. **Expected value of a Pass 5 on the spec is low:** the spec's remaining risk is in how the runner
implements the timer, handler ordering and cleanup, which the P1-48b follow-up's own `analyst` and `security-expert`
gate covers against the code. Reopen the spec only if that gate finds the spec contradicting the shipped runner.
