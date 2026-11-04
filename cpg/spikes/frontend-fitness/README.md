# Frontend-fitness spike: S25 setup (P0-18a)

> **Status:** active · **Owner:** `graph-dba` · **Tracks:** C-1001 (P0, S25 / P0-18a; measure.sc and the `joern` stage are P0-18b's tools)

The setup for the seven-language Joern frontend-fitness measurement
(`docs/plans/connected-code-graph-p0-ml.md` "ML", `docs/plans/connected-code-graph-p0-2.md` §4.6). This directory holds
the pinned corpora, the Joern-independent sampling frames, the labeller inputs, the reference truth and the Joern-side
measurement script. **No Joern run on any corpus has been made.** Order inside S25 is frame → C5 → draws and briefs →
first Joern run (ML §3.2); the C5 records below are all earlier than any Joern output, and the only Joern runs so far are
on seven tiny synthetic probe projects (`matcher-table.md`).

**State (2026-11-04): 5 of 7 languages are drawn; 2 need a decision before S26/S56 may start there (§5):**
Python, TS/JS, Go and Java passed C5; **C#** and **C++** failed C5 for structural reasons (draws exist, provisional);
**Kotlin** has no corpus (§5).

## 1. Layout

| Path | What | Who may read it |
|---|---|---|
| `corpora.yaml` | the pinned corpora (SHA, `srcDigest` of the fetched tree), C1-C8 evidence, C5 verdicts, rejected candidates, the C++ build decision | harness, `data-scientist` |
| `run_fitness.py` | the harness: `fetch census frame draw flowdraw c5pilot brief refmap verify joern` (`--help`) | harness |
| `truth_run.sh`, `truth-scripts/{java,cpp}.sh` | reference-truth runs in the `ccg-toolchain-truth` image (network on, outside the sandbox) | harness |
| `make_adjudication_input.py` | N4's input script. Opens only `labels/` and `truth/`; run by `data-scientist`, never by `graph-dba` | S57 |
| `measure.sc`, `matcher-probe.sc`, `matcher-table.md` | Joern-side per-site answers (P0-18b); the probe that produced the matcher table | harness |
| `frame/<lang>.jsonl` | the tree-sitter census frame (ML §4.1): every syntactic call site | **not** labellers |
| `sample/<lang>.jsonl`, `sample/<lang>.brief.md` | what the labeller gets (500 sites; the filled brief, pasted as-is) | S26 labeller |
| `sample/<lang>.private.jsonl` | harness-only (kind, node type, frame size, `siteKey`) | **not** labellers |
| `sample/<lang>-flow.jsonl` | the golden-flow draw (ML §6.4): up to 30 methods with source text | S56 labeller |
| `truth/<lang>.c5-pilot.jsonl`, `truth/<lang>.c5.json`, `truth/<lang>.ref.jsonl` | the 50-site pilot, the C5 record, the reference answer for every sampled ordinal | **not** labellers; S57 |
| `labels/ adjudication/ results/` | empty (`.gitkeep`): later steps write them | later steps |

Scratch (gitignored, `cpg/.ccg-work/`): `corpora/<lang>/src` (the clean `git archive` tree; the labeller's and Joern-off's
tree), `frontend-fitness/{truth-work,truth-out,rejected,probe*}`. Rejected-candidate artifacts live under `rejected/`.

Reproduce: `run_fitness.py fetch <lang>` (needs only `corpora.yaml`), then `frame`, `draw`, `flowdraw`, `c5pilot`,
`brief`; `truth_run.sh <lang>` then `refmap <lang> --mode c5|ref ...`. `run_fitness.py verify` re-derives every frame,
sample, private, flow and brief file and compares: it passes for all six pinned corpora (and `srcDigest` equality).
Tests: `cd cpg && .venv/bin/python -m pytest -q tests/test_frontend_fitness.py` (24 tests).

## 2. Definitions this harness fixes (where ML leaves a choice)

* **Position** `line:column`: 1-based; columns count **code points**, not bytes (bundle spec §5.7). `siteKey` =
  `path:line:column:callText`, the ML §4.4 hash-sort key; repeated keys get `#2`.
* **Callee expression** per node type: Python `call.function`; TS/JS `call_expression.function`, `new X` (from `new`),
  JSX tag; Java `method_invocation` from the object/name to the method name, `new T`, `super`/`this`; Kotlin the first
  child of `call_expression` (a trailing-lambda pair `f(x) { }` is **one** site), the type of a delegation specifier;
  C# `invocation_expression.function`, `new T`; Go `call_expression.function` plus `type_conversion_expression`
  (tree-sitter-go parses `F[T](x)` that way; the labeller marks real conversions `notACall`); C/C++
  `call_expression.function`, `new T`. `calleeName`/`nameLine`/`nameColumn` = the last identifier outside type arguments.
* **Not in the frame (blind spots, add to ML R2):** Kotlin `this(…)`/`super(…)` secondary-constructor delegation, C#
  `new(...)` target-typed creation, implicit calls, and every call inside a tree-sitter ERROR region the grammar cannot
  recover (googletest: 10.5 % of frame sites sit in ERROR nodes; recorded per corpus as `framefilesWithParseErrors`).
* **Files:** spec3 §3.3 extensions minus declaration files (`.d.ts`, `.pyi`); **Kotlin `.kts` build scripts are not
  frame files** (their calls are Gradle-DSL calls into the Gradle API; `kotlin2cpg` takes `.kt`); C/C++ `.h` are parsed as
  C++, `.c` as C. A file is generated if its first five lines match `generated|do not edit` (case-insensitive) or its path
  is under `generatedPaths`. Directory segments in the effective `vendoredDirs` (spec3 §5.0 default) and tool caches are
  excluded. All six pinned corpora use the **default** vendored list.
* **Sample** = ML §4.4 verbatim (`SHA-256(seed|siteKey)`, first 500). The flow population = ML §6.4: bodied, **named**
  (never `<anon>`), 5-25 lines, ≥ 1 parameter other than `self`/`cls`, ≥ 1 branch node (`if`/ternary/`switch`/`match`/
  `when`/loops) and ≥ 1 frame call. Flow qnames follow bundle spec §4.1/§4.2 for named scopes; overload signatures are used
  only for the `#N` suffix; **qname fidelity beyond named scopes is unverified** (Kotlin property accessors, C# primary-
  constructor `<init>`, TS `<init>` field initialisers are not modelled).
* **`ref.jsonl`** (ML §11 N1, plus a `site` object so `make_adjudication_input.py` needs no `sample/`): `static` is
  `{kind: internal|declared, path, qname, defLine}` for an in-corpus target (`declared` = bodiless), `{kind: external,
  package}`, `{kind: local}` (a local variable holding a callable), `{kind: dispatch}` (Go: several VTA edges, targets in
  `impls`) or `{kind: silent}`. `impls` = the SCIP `is_implementation` closure (Go: every in-corpus edge target). A
  definition that is not a file of the clean corpus tree (generated or fetched) is **external**, as in the labeller's view.
  scip-python reports a builtin function as an undefined `local N` symbol; the mapper turns it into `external`.

## 3. Per language (observed)

| Language | Corpus @ pinned SHA | Frame sites / files | Reference | C5 pilot (raw) | Reference at the 500 sampled sites |
|---|---|---|---|---|---|
| Python | proving repo `falkor-chat/server` @ `0caf85e3` | 17,358 / 104 | scip-python 0.6.8 | **0.90** (45/50) PASS | 254 internal, 178 external, 66 silent, 2 local |
| TS/JS | proving repo `salesperson` @ `0caf85e3` | 4,097 / 68 | scip-typescript 0.6.0 (+ a work-copy tsconfig over `tests/`) | **0.98** PASS | 109 internal, 370 external, 21 local |
| Go | `labstack/echo` @ `7af8fac5` (MIT, module `echo/v5`) | 13,362 / 112 | callgraph VTA v0.52.0 `-test` | **0.90** (0.918 without builtins) PASS | 150 internal, 11 dispatch, 268 external, 71 silent |
| Java | `spring-petclinic/spring-petclinic-rest` @ `5180cc9b` (Apache-2.0) | 3,693 / 64 | scip-java 0.13.2 (manual: `javac -Xplugin:scip` + `aggregate`) | **0.92** PASS | 62 internal + 38 declared, 317 external, 83 silent |
| C# | `ThreeMammals/Ocelot` @ `8b8fde19` (MIT) | 24,414 / 547 | scip-dotnet 0.2.16 | **0.86** FAIL (§5.1) | 218 internal + 15 declared, 189 external, 78 silent |
| C++ | `google/googletest` @ `a8380b01` (BSD-3) | 28,490 / 132 | scip-clang 0.4.1, gcc compdb | **0.56** FAIL (§5.2) | 192 internal, 109 external, 196 silent |
| Kotlin | none (§5.3) | — | scip-kotlinc 0.13.2 (S24-proved) | — | — |

C1/C2/C3/C6-C8 and the exclusion greps are in `corpora.yaml` (all six: no OpenTelemetry/`microservices-demo`/broker
hit; Ocelot 55.6k and googletest 54.4k code lines against the 60k ceiling; echo's C6 is the name-based approximation
below). Java's frame holds 3,693 sites (≥ 2,000 ✓) but the reference finds an in-corpus target at only 100 of the 500
sampled sites (TS/JS: 109), so those labellers may end near the 100-site floor of ML §4.3 (the hand labeller may find
more where the reference is silent: 83 Java sites). The flow draw
is 30 methods everywhere except **TS/JS: 14** (the whole population: only 18 bodied named methods have a branch and a
call even at 2-60 lines; ML §8.4's 120-pair floor then needs ≈ 9 pairs per method).

**C6 for Go is approximate**: 14 interface types with ≥ 2 in-corpus implementers by a tree-sitter *name-based* method-set
check (3 when the interface has ≥ 2 methods); `callgraph` emits no implementation relation, so the ML §3.2 "checked by
the reference's implementation relation" cannot be run. Pilot kinds show 2 of 50 sites as dispatch.

## 4. C/C++: gcc compilation database, not a clang build

**Decision.** `compile_commands.json` comes from `cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON` plus a full `cmake --build -j2`
with the image's **gcc 13.3.0**; `scip-clang` (its own bundled Clang/LLVM 21.1.8) indexes it from the project root with
`--jobs 2`. Joern parses the corpus as-is: `c2cpg` reads the source tree, so the Joern runs do not depend on the build.
* **Why.** S24's review M1 settled that `scip-clang` is self-contained and takes system headers from the compiler named in
  the database; the header-sensitive smoke (`<vector> <atomic> <immintrin.h>`) passes on gcc headers with the apt clang
  absent or hidden. A clang build would need a pinned clang added to the `truth` stage, a Dockerfile edit that moves the
  image tag and re-runs S08's tests 3/6/9/10 (S24 item 5), +1.2 GiB, for no claim it changes.
* **Cost.** googletest configure + full build with tests on + `scip-clang`: 213 s wall, 85 TUs, `num errored TUs: 0`
  (5 GB container cap, one `docker stats` sample of 0.68 GB mid-build; peak not measured); leveldb 40 s.
* **What a different choice would change.** "Parse as-is" (no database) makes C/C++ single-source (no reference), which
  ML §5.2 allows only for a tool that fails S24; that would drop the label-noise estimate for C/C++. A clang-built database
  would only change which `#if` branch and which compiler built-ins `scip-clang` sees. **The 0.56 pilot is not evidently
  caused by this choice**: of googletest's 22 silent pilot sites, 4 are in tree-sitter ERROR regions and 6 under
  non-guard `#if` groups (9 distinct); the other 13 were not classified one by one. On leveldb's silent sites I inspected,
  calls inside test-macro arguments (`ASSERT_TRUE(db->Put(..))`) carry only occurrences at the macro invocation's columns
  (`testing/internal/EqHelper#Compare` at columns 2-19), and files outside the compilation database (Windows port,
  benchmarks) are silent by construction. No AC-8-relevant claim and no language scope changes.

## 5. Open decisions (the three that gate S26/S56 for their language)

1. **C# C5 (0.86 on Ocelot, 0.80 on eShopOnWeb).** `scip-dotnet` 0.2.16 emits **no occurrence for a generic-named callee**
   (`F<T>(…)`, `new C<T>(…)`, including in-corpus generics) and none for `nameof`. All 7 silent pilot sites on Ocelot are
   such sites (6 generic, 1 `nameof`); all 10 on eShopOnWeb. Excluding them the pilot is 1.00, but the rule is the raw
   90 % and "the tool is never swapped". Options: (a) keep Ocelot and amend C5 for C# to count generic/`nameof` sites as
   reference-silent by construction (silent + hand-internal sites are adjudicated anyway, ML §5.1 step 3); (b) mark C#
   single-source. Draws, brief and `ref` exist for Ocelot; **do not start S26 for C# until ruled.**
2. **C++ C5 (0.56 on googletest, 0.66 on leveldb).** See §4; same remedy question for `data-scientist`. googletest meets
   every other criterion (leveldb also fails C7: C++11). Its frame is 30 % macro-shaped sites (8,556 of 28,490), the R7 risk
   in ML §10. **Do not start S26 for C++ until ruled.**
3. **Kotlin has no corpus.** The reference needs a Gradle build of an external Kotlin repository; the harness permission
   layer refused that action ("Code from External"), after `scip-java index --build-tool=gradle` had already failed on
   `mokksy/mokksy` in `buildSrc`. Eleven candidates are rejected on stated criteria in `corpora.yaml`; the best static fit
   is `mokksy/mokksy` (Kotlin Multiplatform with wasm/native targets, so `webMain`/`nativeMain` files would be
   reference-silent). Needs the human's permission for the Gradle build and a choice of corpus.

## 6. Firewall and what is not here

* `sample/` and `frame/` carry no Joern output, no reference truth and no adjudication key; `verify` scans their JSON keys,
  and the brief is ML §6.5 verbatim plus the spec3 §4.1 and the language's §4.2 excerpt (it names Joern only to forbid it).
  `sample/*.private.jsonl` and `frame/` hold harness-only fields and must stay out of every labeller's reach (ML §6.2).
* `make_adjudication_input.py` imports only `argparse json sys pathlib`, opens two files (a reader limited to `labels/` and
  `truth/`, and one writer limited to `adjudication/`), and is asserted so by `test_adjudication_script_static_firewall`.
  `declared` labels are passed through unmapped (rule 8 needs source it must not open).
* **Not produced by S25:** the census truths of ML §5.3 (definitions, value sets, decorators; S27 builds them from the same
  tree-sitter), the metrics of ML §7 (Wilson intervals, McNemar), `labels/`, `adjudication/`, `results/`.
* Joern's `CALL` positions are not uniform across frontends (JS/Kotlin columns and C# lines+columns are 0-based);
  `matcher-table.md` records this and `measure.sc` applies it. `measure.sc` ran only on the seven probes (every probe site
  matched except JSX, Kotlin delegation and Go generic instantiation, the by-construction classes); it has not seen a corpus.
* Deviations to rule on: C5's Go pilot counts builtin calls (`len`, `append`) as sites that can never carry a callgraph edge
  (raw 0.90, 0.918 without); `frame/` is 27 MB (reproducible from `corpora.yaml`; consider not committing it).
