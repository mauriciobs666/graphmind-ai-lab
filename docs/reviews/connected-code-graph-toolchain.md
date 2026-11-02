# Review: connected-code-graph S24 items 1-4 and 6 (`cpg/infra/toolchain/`)

> **Status:** active · **Owner:** `analyst` · **Tracks:** C-1001 (P0, S24)

## Scope & verdict

Reviewed the uncommitted diff in `cpg/infra/toolchain/` (`Dockerfile`, `build.sh`, `README.md`, new `smoke/`, new `truth-npm/`) against `docs/plans/connected-code-graph-p0-2.md` §4.1 S24 items 1-4 and 6 and the S02 baseline (`git show 30bb4c5b:cpg/infra/toolchain/Dockerfile`). Item 5 (sandbox re-run, `sbx_default_image`) is held per the plan and was not reviewed. Root `REVIEW.md` applied; no component `REVIEW.md` exists for `cpg/`.

Ran: `ccg-truth-smoke` (8/8) and `ccg-versions` in both images under `--network none`; mutant copies of the smoke projects; the `20-bad` + `40-exit` drop-in pair against the new image and against the S02 image as control; `build.sh --print-tag` determinism probes; lockfile structure checks. Did not rebuild, and did not touch `falkordb-dev`, `ccg-falkordb` or other containers.

**Verdict: approve with suggestions.** No blocker. One major (M1: the clang/LLVM "matching" requirement is unevidenced, and the README states it wrongly).

CPG: considered, not relevant — Dockerfile/shell/infra files; no loaded CPG covers them.

## Findings

### M1 (major, verified): clang is neither "matching" `scip-clang` nor shown to be used

Spec item 1: "`scip-clang` with a matching clang/LLVM". Evidence:
- `scip-clang --version` in the truth image prints "Based on Clang/LLVM 2078da43…", and `strings` on the binary gives `clang version 21.1.8`. The apt package installed is clang 18.1.3 (`ccg-versions` output; Dockerfile line 369).
- README's tool table lists "clang / LLVM 18.1.3 | `scip-clang`'s resource headers" and never mentions the 18 vs 21 gap.
- The cpp smoke still PASSES with both `/usr/lib/llvm-18` and `/usr/lib/clang` hidden by tmpfs mounts (`docker run --tmpfs /usr/lib/llvm-18:exec --tmpfs /usr/lib/clang:exec … ccg-truth-smoke cpp` -> PASS). So the apt `clang` is not exercised by the smoke, and the smoke cannot tell whether it is needed.

Why it matters: S25 runs `scip-clang` on real corpora, where a resource-header version mismatch is the classic failure; today nothing says whether the installed clang plays any role. A wrong README claim also misleads the next reader.

Fix: either (a) drop `clang` from the image if `scip-clang` is self-contained (and say so in README), or (b) keep it, state the 18.1.3 vs 21.1.8 gap in README, and add a smoke fixture that includes a header whose builtin content differs across versions (for example `<immintrin.h>` or `<atomic>`). Also add the mutant above (hide the system clang) as the check that proves the claim.

### m1 (minor, verified by reading): runner default is already re-pointed

`cpg/sandbox/offline-run.sh:279-282` derives `sbx_default_image` from `build.sh --print-tag`, and that tag is now `a18f019e43ed` (verified: `--print-tag`, also under `env -i PATH=/usr/bin:/bin`). The README notes this. The consequence is that every runner invocation in this tree, including S08's in-flight follow-up round, now defaults to an image S08's tests 3/6/9/10(c) have not been run against (item 5 is held). The layers are cached and identical to `d3184effb566` (same size, 4,718,150,444 bytes), so risk is low. For `teco`: pin S08's follow-up runs with `CCG_TOOLCHAIN_IMAGE=ccg-toolchain:d3184effb566` until item 5 re-runs them, or accept the new tag explicitly.

### m2 (minor, verified): the acceptance checks are narrated, not checked in

The README's "Checks of the checker" (override broken -> 0/8) and S02's p2-m1 drop-in acceptance have no durable test in the tree; `smoke` is not wired into the build either (a build succeeds with a broken smoke). I re-ran them and they hold (Appendix A), so this is about regression, not today's correctness. Suggest a small `cpg/infra/toolchain/tests/` script with: the `20-bad` + `40-exit` drop-in pair via a bind mount (expect non-zero; the S02 image `d3184effb566` is the positive control, it exits 0), two smoke mutants (one per assertion: implementation edge removed, call site removed), and the `--print-tag` determinism cases of Appendix B.

### m3 (minor, verified): pins that are trust-on-first-use

Disclosed in the Dockerfile and README, so not a deviation: `scip-clang` binary, the two scip-code jars, the `scip-dotnet` nupkg (the Maven jars were cross-checked against Central `.sha1`; `scip-clang` and the nupkg were not). `callgraph` is pinned by module version plus the Go checksum database only: the repo records no `go.sum`/`h1:` for it (`ccg-versions` prints it at run time). Suggest committing the `h1:` hash and failing the build on a mismatch (`go mod download -json` or a one-line `go version -m` check after install). npm is solid: both lockfiles are `lockfileVersion 3`, all 88 packages from `registry.npmjs.org` with `sha512` integrity, none with install scripts, `npm ci --ignore-scripts`.

### Nits

- n1: `kotlinc-scip` hard-codes `2.4.21` and the embeddable-jar URLs repeat it, with no tie to `KOTLIN_VERSION`; a bump of one will silently pair mismatched compilers. One `ARG` or a `ccg-versions` assertion that the two match would close it.
- n2: stage `truth` declares `ARG SCIP_JAVA_VERSION` and never uses it.
- n3: the truth image inherits `LABEL ccg-toolchain=1` from `toolchain` (verified by `docker inspect`), so a label filter for the sandbox image would also match the truth image. Nothing in `cpg/sandbox` filters on it today (`git grep`).
- n4: `/tmp` in the truth image holds four empty root-owned 0700 dirs from `dotnet tool install`; `toolchain` is clean. Harmless; `rm -rf /tmp/*` in that RUN would tidy it.
- n5: `smoke/` is tarred wholesale, so a stray file (an editor swap, a local `obj/`) moves the tag (verified with a planted `smoke/.stray`: `a18f019e43ed` -> `0e68a7dc64ae`). That is the intended "any input moves the tag" behaviour; just note that a local smoke run must not be done inside the tree.
- n6: `build.sh --truth --no-cache` rebuilds and re-tags the `toolchain` tag too, so "never removes or retags any other image" (header comment) is slightly overstated for that flag combination.

## Spec conformance (checked, no finding)

- **Pinning (item 1).** Every download has a version plus sha256/sha512 (Node, scip CLI, scip-java jars, scip-go, scip-clang, uv, scip-dotnet, kotlin-embeddable jars); npm by lockfile; Go by sumdb; no floating tags (`grep` of the Dockerfile for `latest`/branch refs: none). Node 24 LTS is in maintenance; the stated rationale is odd but the plan only says "LTS".
- **Snapshot-only apt (item 4).** `FROM` lines: `${BASE}` appears once (`apt-base`); `fetch`, `truth-fetch`, `toolchain` descend from `apt-base`, `truth` from `toolchain`. Every apt use in the new stages goes through `apt-snapshot-install` (`xz-utils`; `clang python3 python3-venv python3-pip`); no bare `apt-get install`. In the built image both `URIs:` lines are the snapshot.
- **Placement (item 3).** `docker run ccg-toolchain:a18f019e43ed`: `node npm scip clang callgraph scip-python python3` are all absent; `/opt` has only `dotnet go gradle java joern kotlinc maven`. `truth` is `FROM toolchain`.
- **`ccg-versions` carry-forward (item 4).** Passes (Appendix A): `20-bad` + `40-exit` -> exit 1 (`1 check(s) FAILED`) on the new image, exit 0 on `d3184effb566`, so the check can both fail and pass. `exit 3`, a same-file `v bad false; exit 0` and `trap - EXIT; exit 0` also exit non-zero. A lone `exit 0` drop-in exits 0. The `truth` image registers `50-truth.sh` and passes.
- **Smoke is not vacuous (item 2).** Appendix A. Both assertions (implementation relationship, call-site reference) bite in every language.
- **Build context.** Appendix B. The tag is deterministic and matches the built images.
- **Hygiene.** No `ARG`/`ENV` secrets; `docker history --no-trunc` has no token/secret/auth strings; no build secrets; network used only by build `RUN`s; the `truth` image has no cache directories (`find` for `.npm .nuget .m2 .gradle _cacache gopath`: none; `/home/ccg` skeleton only, `/work` empty); user `ccg` uid 1000, `ENTRYPOINT []`, `CMD ccg-versions`. The nupkg copies under `/opt/truth/scip-dotnet/.store` are the tool's install, not a cache.
- **README (item 6).** Sizes verified (`docker image inspect`: 4,718,150,444 and 6,043,560,489). The `0.0.0-SNAPSHOT`/`0.1.0-SNAPSHOT` caveats are true (smoke output). Per-tool versions match `ccg-versions`. Only inaccuracy: M1's clang row.
- **Stale "tag is sha256(Dockerfile)" claim.** `git grep` over `cpg`, `docs/plans`, `docs/requirements`: no live document still says it. `cpg/sandbox/README.md:46` says only "`build.sh --print-tag`" (correct as written). The one remaining hit is the historical ledger row U21 in `docs/plans/connected-code-graph-coordination.md:58`, a record of the S02 check; leave it.

## What's solid

Stage layout and the `truth`/`toolchain` split are clean and the sandbox image's layers are unchanged. The lockfile discipline is strong. The smoke runner is honest: it fails loudly, uses scratch dirs under `$TMPDIR`, and its assertions are specific (relationship direction, role bit for references, an exact callgraph edge). The README discloses the real caveats (embeddable Kotlin compiler, `scip-java` rewrite, TOFU pins, version strings).

## Open questions

- M1: does `scip-clang` 0.4.1 need the system clang at all? If devops can show it does not, dropping it also saves image size.

## Appendix A: mutation and acceptance runs (all under `--network none`, truth image `a18f019e43ed`, mutated copies mounted via `CCG_TRUTH_SMOKE_DIR`)

| Mutant | Result |
|---|---|
| Baseline | 8/8 PASS |
| Implementation edge removed (python `class Circle:`, ts without `implements`, kotlin, csharp; java and cpp with a lambda/`Sq` stand-in) | python, typescript, kotlin, csharp, java, cpp FAIL with "no is_implementation relationship …" |
| Call through the interface replaced by a constant (all 7 languages) | all FAIL ("only its definition", empty callgraph edge, `0 reference(s)` for scip-go) |
| Go fallback with `Circle` left in place | `go-scip-go` PASS, correct: Go interfaces are structural, so `Circle` still implements `Shape`; not a hole |
| Drop-ins `20-bad.sh` (`v bad false`) + `40-exit.sh` (`exit 0`) | new image: exit 1; S02 image `d3184effb566`: exit 0 (the bug the item fixes) |
| `ccg-versions` on both images | exit 0 |

## Appendix B: build-context determinism (`build.sh --print-tag`)

- Normal, `--truth`, and `env -i PATH=/usr/bin:/bin`: `a18f019e43ed` for all (the runner's call shape).
- A copy of the tree in another directory with mtimes changed to 2001: same tag. `chmod 600` on the Dockerfile: same tag (modes normalized). `chmod +x` on `smoke/go/main.go`: `69ad1343d100` (the exec bit is an input, as the header says). A stray `smoke/.stray`: `0e68a7dc64ae`.
- The tag equals both built images' tag, so the images match the working tree reviewed.

## Pass 2 (tag `a82faf5219b3`)

Re-checked by execution against the working tree (`build.sh --print-tag` / `--truth --print-tag` = `a82faf5219b3`, both images present). Did not rebuild; did not re-run S08's `test_sandbox.sh --only 3,6,9,10` (the reported 132 passed is the coordinator's, not re-verified here).

| ID | Disposition | Evidence rechecked |
|---|---|---|
| M1 | **Fixed (by removal)** | Truth image: `dpkg -l` clang/llvm/libclang count 0; no `/usr/lib/llvm*`; `which clang` empty. README now records the measurement and the limit (a compdb naming `clang` needs it on `PATH`). cpp smoke is header-sensitive: `<vector>`, `<atomic>`, `<immintrin.h>`, 81 occurrences, PASS. Mutants: call site removed -> FAIL ("only its definition"); override edge removed -> FAIL ("no is_implementation relationship"); `--tmpfs /usr/lib/gcc/x86_64-linux-gnu/13/include` (the test's hide-gcc-header mutant) -> FAIL. Not vacuous. |
| m1 | **Superseded** | Item 5 reported done; the runner default follows `--print-tag` (`a82faf5219b3`). Not re-run by me. |
| m2 | **Fixed** | `tests/test_toolchain.sh`: ran it, `33 passed, 0 failed`, exit 0, 1m19s; it covers tag determinism, the p2-m1 pair with the S02 image as control, the toolchain image carrying no truth tools, and per-language smoke mutants. |
| m3 | **Not fixed (accepted)** | Dockerfile still has no committed `h1:` for `callgraph`; scip-clang/nupkg remain TOFU, disclosed. Unchanged minor. |
| n1 | **Fixed** | `ARG KOTLIN_EMBEDDABLE_VERSION` with a `grep` of `/opt/kotlinc/build.txt` tying it to S02's `KOTLIN_VERSION` (Dockerfile:204-205). |
| n2 | **Fixed** | the `SCIP_JAVA_VERSION` ARG now sits only in `truth-fetch`, where it is used (Dockerfile:185-194). |
| n3 | **Not fixed (nit)** | `docker inspect`: truth image still carries `ccg-toolchain=1` as well as `ccg-toolchain-truth=1`. |
| n4 | **Fixed** | `ls -A /tmp` in the truth image: empty. |
| n5 | **Not changed (by design)** | tag is a hash of the whole `smoke/` tree. |
| n6 | **Fixed** | `build.sh:22-24` now says `--truth --no-cache` re-tags both images. |

### New findings

None. One residual note, not a finding: the hide-gcc-header mutant fails the cpp smoke through the call-site assertion rather than a "header not found" message, which is fine because the index is what is asserted; hiding the whole `/usr/lib/gcc/x86_64-linux-gnu` instead breaks CMake's compiler probe, so the narrower `.../13/include` mount is the right mutant.

**Pass 2 verdict: approve.** M1 is closed; remaining items are the accepted minor m3 and nit n3/n5.
