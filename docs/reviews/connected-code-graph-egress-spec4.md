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
