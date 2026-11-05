# Connected code graph — P0(a) frontend-fitness measurement protocol

> **Status:** active · **Owner:** `data-scientist` · **Tracks:** C-1001 (M10) · **Extends:** `docs/plans/connected-code-graph-ml.md` · **Version:** 1.14 · **Reviews:** `docs/reviews/connected-code-graph-p0-impl.md` (§S05)

*Revision 2026-11-05 (v1.14): §2 "Site found" now follows §7.1 (a site absent in Joern is not found); before this, §2 (name match credited without a `CALL`) and §7.1 contradicted each other. A post-result clarification, disclosed in the P0 report with the lenient counterpart as a secondary figure (§7.2). Earlier revisions: §12.*

## 0. Question and the decision it serves

**Question.** How should P0(a) measure Joern's fitness per FR-1 language so that S27/S28 can decide,
mechanically and per language: (1) whether a precise indexer or an idiom recognizer is adopted,
(2) whether a resolved call's other candidates are folded or listed, and (3) `flowAvailable`?
Constraints: blind hand labels in **seven per-language `qa-engineer` dispatches** (S26), one
dispatch each; a 6 GB host; truth tools that must run in the S02/S24 `ccg-toolchain` image; no
OpenTelemetry demo and no held-back candidate as a corpus.

**Verdict, in brief.**
- **Sample:** per language, one seeded **simple random sample** of syntactic call sites drawn from a
  **Joern-independent frame**, labelled in order until **150 sites with an in-corpus callee** have
  been labelled or 500 sites total, whichever comes first (§4). Census, not sampling, for
  everything that has a mechanical truth (parse success, line fidelity, fan-out, value sets,
  decorator shape).
- **Truth:** blind hand label **plus** an independent typed reference (the SCIP indexer, or Go's
  `callgraph`), with every disagreement and every Joern miss adjudicated (§5). A single LLM
  labeller alone cannot support a 0.98 gate, because its own error rate is of the same order as
  the 0.02 the gate tolerates.
- **The 0.98 recall gate is a screen, not a certificate.** At n = 150 it passes a language whose
  true recall is 0.99 about 94% of the time and a 0.95 language about 5% of the time, but it
  cannot show recall ≥ 0.98: that needs ≥ 189 sites with zero misses, and a language truly at
  0.99 would almost never produce that. AC-5's end-to-end gate is what certifies recall (§8.1).
- **Precision 0.95 is a point-estimate rule**, for both folding and indexer adoption, as the plan,
  the proposal and ML §8.1 fix it (§8.3). On the ~60–120 resolved sites a sample yields, a 0.95
  observation has a Wilson interval of [0.863, 0.983] at 60 and [0.888, 0.978] at 100. That is
  weak discrimination. It is tolerable for folding only because folding is display-only (proposal
  §3.7). For adoption, the interval is reported beside the decision.
- **`flowAvailable`** has one primary criterion: pooled golden-pair recall ≥ 0.95 observed, plus
  projection success as a precondition (§8.4, with its operating curve).

## 1. Findings from the plan, the proposal and the ML note

Upstream facts this note takes as fixed: the a1/a2 split (plan §2 D3); D4's corpus rule and
exclusions; S26's seven blind dispatches; the fold/list threshold of 0.95, display-only (proposal
§3.7); fallback recall as the gate and fan-out as the cost driver (ML §3.6, §8.1 m1); both
configurations (fetch-off, fetch-on) measured on every corpus.

What reading the real plan and system changed:

| # | Finding | Consequence in this note |
|---|---|---|
| F1 | A sample frame built from Joern's own call nodes cannot see call sites Joern dropped (fuzzy C parse, failed files, macro-expanded code), so recall would be over-estimated exactly where the frontend is weakest. | The frame comes from tree-sitter, independent of Joern. A frame site with no matching Joern call counts as a miss, after a matcher audit (§4.1, §7.1). |
| F2 | Plan §4.3 suggests stratifying by resolved vs unresolved. But "resolved" differs between the off and on configurations, and one label set must serve both. A design stratified on one configuration's output is biased for the other. | One simple random sample per language, shared by both configurations. Call-site kinds are reported as domains, not strata (§4.2). Off-vs-on is then a paired comparison (§7.4). |
| F3 | Most call sites in service code call libraries and builtins. Under a fixed-size sample, the number of sites with an in-corpus callee (the recall denominator) would be random and possibly small. | Inverse sampling: label in seeded order until 150 in-corpus sites. The stop depends only on the label's *internal* flag, never on Joern's answer, so the 150 are a uniform random sample of internal sites (§4.3). |
| F4 | An LLM labeller's error rate is plausibly a few percent. Undetected false *hits* (an incomplete truth set, e.g. a forgotten implementation) bias recall upward by about the same amount the gate tolerates. | Two independent truth sources per site, with adjudication of every disagreement and every miss (§5). |
| F5 | ML §3.6 says a recall failure means "the indexer, or a recognizer". A typed indexer fixes only *resolution* misses (an alias, a re-export, a call under another name that is statically bound). It does not fix callbacks, reflection, DI by string, or framework dispatch. | Every miss is classified by cause, and the cause routes the remedy (§8.1). |
| F6 | Plan §4.3/§4.6 refer to "S05's golden small-method sets" for `flowAvailable`, but no step produces them, and DATA/CONTROLLED_BY *edge* recall is the wrong unit: impact uses reachability between points, and one missing edge breaks a path. | Golden sets are pair-reachability sets, produced by a new step (§6.4, §11 N3). |
| F7 | Plan §7 lists no Linux Node.js. `scip-typescript` and `scip-python` both need it. The truth tools also need each corpus's dependencies installed (types), and the SCIP CLI to read indexes. | Listed in §9 as image needs that §7 does not have. |
| F8 | JSX component use (`<Cart/>`) is a call in impact terms, but it is not a tree-sitter `call_expression`. `salesperson/` is a React SPA. | JSX elements with a capitalized tag are in the TS/JS frame (§4.1). |
| F9 | ML §4.5 describes the held-back candidate by genre (a polyglot multi-service demo with messaging). Excluding only the named Online Boutique leaves the rest of that pool open to contamination. | The exclusion is by genre: no polyglot multi-service demo application is a P0(a) corpus (§3.2). |

## 2. Measurement units and definitions

- **Frame site.** A syntactic call in a corpus source file of the language, enumerated by
  tree-sitter (node types in §4.1). Its key is `(path, line, column, callText)`. `line` and
  `column` (1-based) are the position of the **callee expression's first character**, and
  `callText` is the callee expression's source text, whitespace-collapsed. So in `a.b().c()` the two
  sites share `line:column` and are told apart by `callText` (`a.b` vs `a.b().c`). The frame record
  also carries `calleeName` and `nameLine`, the callee's last identifier and its line, in the bundle
  spec's §5.6 sense (`docs/plans/connected-code-graph-bundle-spec.md`), which §7.1's matcher uses.
- **Symbol-ness follows the bundle spec in force** (v1.2 at this revision): its §4.1 rule 8 is an
  **exhaustive per-language table** of which bodyless declarations are symbols, and the harness
  applies that table as written. It is not restated here. In summary: every function-like definition
  with a body is a symbol, and so is every bodyless declaration the table names, which includes:
  - Python `Protocol`/`@abstractmethod` stubs;
  - TS abstract methods and method signatures in an `interface` or a `type`-alias object type;
  - Java abstract, interface and `native` methods, and a record header as `<init>`;
  - Kotlin abstract/interface and `external` functions, and the primary constructor;
  - C# abstract, interface and `extern` methods, and primary constructors;
  - Go interface method specs of a **named** interface type (`IfaceType.NAME`), and bodyless
    (assembly) functions;
  - C++ pure virtuals and `= default` declarations.

  Not symbols include `= delete`, Kotlin `expect`, a C# `partial` defining declaration, C/C++
  prototypes and other in-class member declarations, `extern` declarations, bodiless `friend`s,
  function-like macros, specs in an **unnamed** interface literal, and `.pyi`/`.d.ts` content.
  **Vendored code is never a symbol** (spec §5.0).
- **Truth set `T(s)`** of a site: the in-corpus **symbols** that are a true target of the call, as
  the §6.5 brief defines them (the brief is the operational definition, and this bullet must stay
  consistent with it). Each member carries a **role**:
  - `impl` (a definition with a body that can run): for a statically bound call, the one
    definition; for a virtual or interface call, the method the receiver's declared type has or
    inherits **plus every in-corpus override or implementation**; for a call through a function
    value or callback, the in-corpus definitions that can flow to the callee expression, traced up
    to two caller levels;
  - `decl` (the static target is a bodiless declaration **that is a symbol** under rule 8): that
    declaration. It is in `T(s)` because a confirmed edge to it is a true fact, and a change to the
    declaration impacts the caller.
  - A bodiless static target that is **not** a symbol goes in the **non-symbol set `N(s)`**,
    outside `T(s)`, **unless the spec substitutes it**. Spec rule 8 maps a C/C++ prototype or
    in-class declaration to its definition, and a C# partial defining declaration to its implementing
    one, when the unit holds that entity. The harness applies the same substitution to both labels
    and Joern targets, so these land on the definition (role `impl`). What remains in `N(s)` is
    what the spec drops: for example a spec in an unnamed Go interface literal, or a `.d.ts` member.
    A resolution into `N(s)` creates no confirmed edge (spec §5.6).
  - **No separate declared set.** v1.1's `D(s)` (bodiless declarations kept outside `T(s)` but
    counted correct for precision) is gone; §12 Pass 2 gives the reasoning.
  - External callees are recorded as `external:<package>` and are not in `T(s)`. Calls into
    committed vendored code are external (§3.1, §7.1).
  - **`dynamic`:** some target cannot be determined from source (a computed name, `eval`,
    reflection on a computed string). The determinable targets are still listed. A dynamic site with
    a non-empty `T(s)` enters recall on that listed set, which is a lower bound on its true set. A
    dynamic site with an empty `T(s)` is excluded from recall and counted. The report gives both
    counts.
  - **`notACall`** sites are excluded from every metric, counted toward the cap (§4.3), and their
    share is reported.
- **Internal site:** `T(s)` is non-empty after adjudication (§5.1). **Only internal sites enter
  recall.** The labeller's stopping count (§4.3) uses its own labels (an `internal` or `declared`
  entry), before adjudication.
- **Joern's answer** for a site in configuration `c` ∈ {off, on}:
  - `Res_c(s)`: the matched Joern `CALL` node's resolved targets that map to in-corpus symbols
    (rule 8). Targets in vendored paths are external, not in-corpus (§7.1).
  - `Nm(s)`: the name-match candidates, every in-corpus symbol whose short name equals the call's
    `calleeName`. Name matching is name-only, within the corpus, as the P1 fallback will do
    (proposal §3.4). Resolved sites also carry their name-match candidates (proposal m1). An
    arity-filtered variant is a reported sensitivity only.
- **Site found** (recall unit): the site matches a Joern `CALL` (§7.1) **and** `T(s) ⊆ Res_c(s) ∪ Nm(s)`; an unmatched site (absent in Joern) is not found. Strict: a site with two true callees, one
  missed, is a miss. Edge-level recall is reported as a secondary figure. The analysis unit is the
  **site**, because edges of one site are not independent (repeated-measures rule). A `decl` member
  shares its name with the call, so `Nm(s)` always covers it. In practice recall is decided by the
  `impl` members.
- **Resolved site** in `c`: `Res_c(s)` is non-empty, so the site creates ≥ 1 confirmed edge.
  **Correctly resolved:** `Res_c(s) ⊆ T(s)`.
  - A resolution to an in-corpus symbol when the truth is external is an error (a false confirmed
    edge).
  - **Not resolved sites for precision**, because they create no confirmed in-graph edge, each
    counted and reported per configuration:
    - *resolved away*: resolved only to external full names (vendored targets included);
    - *resolved to non-symbol*: resolved only into `N(s)`, or to any non-symbol Joern `METHOD`.

    A mixed resolution is judged on its symbol targets only. When the truth is internal, these
    sites are still covered by name match or are misses.
  - **Reported beside `P_c`:** the share of resolved sites whose `Res_c(s)` holds only `decl`
    targets (*resolved to declaration*). In interface-heavy languages these confirmed edges are
    correct but name no implementation; implementations stay *possible* through name match. A high
    share is a usability fact for P1, not a precision error.
- **Fallback recall** `R_c` = found internal sites ÷ internal sites. **Resolved-call precision**
  `P_c` = correctly resolved ÷ resolved, over all labelled sites (internal and external).

## 3. Corpus criteria and exclusions, per language

### 3.1 Fixed corpora (D4)

- **Python:** `falkor-chat/server/` at the commit S25 pins (about 75k lines including tests).
- **TS/JS:** `salesperson/` (`src/`, `tests/`, config `.ts` files), same commit (about 8k lines;
  roughly 4k call-like tokens by a crude grep, enough for a frame of ≥ 2,000 sites, which S25
  confirms).
- `node_modules/`, `dist/`, `test-results/`, `*.egg-info/` and generated files are outside the
  frame. These are proving-repo code; §10 R6 covers the contact this creates.
- **TS/JS a1 frame is non-test files only, and is a census (stakeholder decision).** The frame drops
  the files bundle spec4 §1 lists as `test` (`*.test.<ext>`, `*.spec.<ext>`, `e2e/`, `__tests__/`),
  since the bundle has no symbol for them by design; the 4,097-site frame was 87% such files (3,574
  sites; 444 of the old 500-site sample). What is left is **523 sites in 40 files**. Rules for this
  case only:
  - **Census, not a draw.** All 523 sites are labelled, in §4.4 hash order (so ordinals, resume and
    reproducibility are unchanged), with no stop at 150 internal sites and no 500 cap. A sample of
    500 of 523 would add sampling error for no saving, and the extra 23 sites are 4.6% more work. A
    census has no sampling error: the observed recall is the corpus's value, and §8.1 and §8.3 apply
    unchanged to the observed figures and their floors (n ≥ 100 internal, n_P ≥ 60). Their Wilson
    intervals are then a conservative statement about this corpus, not sampling error. Expected
    internal sites: about 190 (20 of the 56 prod sites in the old sample; Wilson 95% on the share
    gives 128 to 255), so the 100 floor is likely met; unmeasured.
  - **C3's 2,000-site floor is waived for this case.** That floor bounds sampling error of a
    sample, and a census has none; what C3 also guards (a frame large enough to hold the weak-spot
    idioms) is carried by §6.4 and by this being the P1 target code. The 5k–60k line bound is not
    waived and still holds.
  - **Labelling.** The 500-site `labels/javascript.jsonl` is superseded (444 of its sites are test
    sites and it was drawn to the old stop rule) and is kept out of the truth. A fresh `qa-engineer`
    labels all 523 in one dispatch under the regenerated brief (N = 523, one dispatch; append-and-resume,
    R5, covers the length). There is no second pass for "the 22 over 500": the old cap does not apply.
    The 56 prod sites the old labeller also labelled are used only as a test-retest check of label
    agreement between two blind labellers, reported in §5.1 step 4, never as a second vote.
  - **Reference.** `scip-typescript` indexed the whole corpus and is unchanged; only the frame,
    sample, private file, brief and `ref.jsonl` mapping are regenerated (a re-map of the existing index,
    no tool re-run). C5's 50-site pilot was drawn from the old frame; it is re-drawn from the
    523-site frame and re-run before any Joern run on the corpus.
  - **Claims.** AC-8 (egress) is untouched. For AC-5, TS/JS fitness now describes the prod code the
    bundle contains; recall on test code is unmeasured and irrelevant by design. Python keeps its
    tests.
  - **Circularity (TS/JS and Python alike).** The 40 TS/JS files and the Python corpus
    (`falkor-chat/server/`) are the proving repo's code, which is also P1's target and has
    development contact (R6), so neither figure is held-out. *Stopping condition, both languages:*
    the contact list (R6) is applied to the corpus; sites in listed files are reported as a separate
    stratum, and the language passes §8.1 only if the non-contact stratum alone has observed recall
    ≥ 0.98 with ≥ 50 internal sites (fewer: *insufficient*, flagged, not passed). Python's labelled
    sample has about 250 reference-internal sites, so a non-contact stratum of 50 is likely; TS/JS's
    census has about 190 in all. AC-5's re-run on fresh items at P5 is the held-out check.
  - **Stratum floor and flow files.** The ≥ 50 floor counts internal sites of the **final truth**
    (`final2`) outside contact files, inside the labelled prefix: recall exists only for labelled
    sites, so Python's 240 labelled rows (stopped by §4.3) are the base, not the 500-row reference.
    Counts from the reference are forecasts (non-contact internal sites under R6's sources: Python 123 of the
    labelled prefix (ordinals ≤ 240) and 241 of the 500 rows, TS/JS 165; 104 and 185 for Python if
    eval-protocol Q1 widens the slice contact to the scale-probe files). If the final count is below 50 and no Joern output has been opened, the
    labeller may continue in ordinal order to the 500 cap (a stop that depends on the stratum count
    and not on Joern keeps the prefix a uniform sample); otherwise *insufficient*. The S56
    flow-method files are **not** contact for these call-site strata: being hand-labelled blind is
    not tuning, as with the S25/S26 labels, and P1 debugs flow against them only after P0(a). "Flow
    anchor" contact (eval-protocol §8.2) belongs to the later evaluation strata, which S47 adds; it
    is recorded in `contact.json` apart (below) and not applied here. Applying it would take Python
    to 52 and TS/JS to 69 (counts measured before v1.9) and make a floor of 50 the deciding quantity for a contact that has not
    happened.

### 3.2 External corpora: one per language for Java, Kotlin, C#, Go, C/C++

`graph-dba` proposes one project per language in S25's `corpora.yaml`, with the evidence for each
criterion. `data-scientist` confirms at S25's gate. **Every criterion is a must**:

| # | Criterion | Evidence S25 records |
|---|---|---|
| C1 | Public, with a licence that allows a local copy and derived measurement (Apache-2.0, MIT, BSD, EPL, MPL). | SPDX id and licence file path |
| C2 | Single-language project in the target language (≥ 80% of source lines). **Not** a polyglot multi-service demo application of any kind. | `cloc`-style line counts |
| C3 | 5k–60k non-generated lines in the language, so that Joern, the reference indexer and the sandbox fit the 6 GB host (§7 of the plan), and the frame holds ≥ 2,000 call sites. | line count; frame size from the tree-sitter pass |
| C4 | Builds from a clean checkout at the pinned SHA with the language's standard tool, in the image, with network: Maven or Gradle, Gradle for Kotlin, `dotnet build`, `go build ./...`, CMake. C/C++ must emit `compile_commands.json`. | build log, exit 0 |
| C5 | The pinned reference tool (§5.2) indexes it without a fatal error, and gives a reference occurrence, under §5.2's mapping, for ≥ 90% of a 50-site seeded pilot of its frame **after the by-construction exclusions below**. **Run in S25 before any Joern run on the candidate.** A C5 failure rejects the candidate and moves to the next one; the tool is never swapped to rescue a corpus. | indexer log, document count, raw and adjusted pilot coverage, excluded sites by class |
| C6 | Contains the weak-spot idiom of proposal §3.2, at the count shown below, by a tree-sitter or grep census. | counts |
| C7 | A modern dialect: Java ≥ 17; Kotlin **within the version range that the pinned scip-java Kotlin plugin documents or S24's smoke test proves** (scip-kotlin is archived and merged into scip-java, and Kotlin 2.x/K2 support is not stated, so in practice this may mean Kotlin 1.9.x, recorded as a representativeness limitation); C# / .NET ≥ 6; Go ≥ 1.21 with modules; C++ ≥ 17. Last commit within 3 years of the pin. | build files |
| C8 | Has tests in the repository, so that tests-for-a-change shapes are in the frame. | test dir |

**C5: sites reference-silent by construction (closed list; priors, confirmed or overturned at S57).**
A pilot site leaves C5's denominator only if it is still reference-silent after §5.2's mapping and the
harness tags it from the source alone (no Joern, no label) into one of these classes. Raw coverage is always reported beside the adjusted figure.
- **Go:** a builtin call (`len`, `append`, ...). Moot for the pinned echo corpus (0.90 raw).
- **C#:** a callee with a type-argument list (`F<T>(…)`, `new C<T>(…)`) and `nameof`. `scip-dotnet` 0.2.16
  emits no occurrence for them, in-corpus generics included; verified, with the occurrences of each
  silent Ocelot sample site's line read back: all 78 silent sites of the 500 (74 in the class) have
  no occurrence named like their callee.
- **C/C++:** a site under a non-guard `#if`/`#ifdef`/`#else` group (the compilation database builds
  one configuration) and a site whose callee is itself a macro.
- **Limits:** the excluded sites are ≤ 10 of the 50 (a judgment, not derived: more means the
  reference is blind to too much of the frame for the noise estimate). A class is added by a
  revision of this note made before the pilot it would apply to. The C# and C++ classes arrived
  after the Ocelot, eShopOnWeb and googletest pilots, on a structural diagnosis of those pilots'
  silent sites, so a pilot cannot test them; they stand as a ruled exception because they are
  re-evidenced on data they were not derived from: three fresh-seed pilots by the S25 gate (C#
  adjusted 1.00, 1.00, 1.00; C++ 0.894, 0.957, 0.915) and the 500-site samples (C# 422/426 =
  0.991; C++ 417/461 = 0.905 without the type-and-constructor clause, 427/462 = 0.924 with it). A
  later class needs the same: a fresh-seed pilot or the sample, both Joern-independent, passing
  0.90. One fresh pilot below 0.90 is not a failure: at 46 sites and a true 0.92, P(adjusted < 0.90)
  is about 0.3, so one in three is expected, and the sample-level figure decides. *Stopping
  condition:* the exception is withdrawn, and the corpus returns to C5 FAIL for the coordinator, if
  the sample-level adjusted coverage of C# or C++ recomputed from the final `ref.jsonl` is below
  0.90. eShopOnWeb and leveldb stay `FAIL` as recorded under the rule in force when they were
  piloted; eShopOnWeb would pass the amended rule (40/40) and is not reinstated, because Ocelot is
  the pick, is labelled, and reinstating after the fact would choose a corpus by its result;
  leveldb also fails C7. An excluded site stays in the
  sample, is labelled, and is adjudicated like any silent site when the label says internal or
  `unsure` (§5.1 step 3).
- **Effect on label noise.** Noise is estimated only where both sources speak, so each class is a
  stratum with no second opinion. The report gives the comparable share (labelled sites whose
  reference is not silent) and the agreement rate per stratum (§5.1 step 4).

**C# (Ocelot kept; priors).** The generic/`nameof` class applies. Pilot: 7 of 50 silent, all in the
class, so adjusted 43/43. Over the 500 sample sites 74 generic/`nameof` sites are silent, of which
12 are `nameof` and 14 name something defined in the corpus (a name-existence proxy, an upper
bound on internal sites, since `Get` or `CreateLogger` may be external): at most about 6% of the
~230 internal sites expected (§4.3), so the blind stratum is bounded. Selection bias: the reference
is blind to generic callees, the shape of result wrappers (`OkResponse<T>`), DI and typed
configuration reads, which is also where a hand labeller is most likely to err. Mitigation: those
sites are checked from source, before Joern, at S57: every one labelled internal (adjudicated
anyway) and a seeded 40 of those labelled external (hash sort, prefix `ccg-p0a-v1-gen|`).
*Stopping conditions.* (a) The generic-stratum share of the labelled internal sites exceeds 10%:
the "bounded stratum" premise fails and C# is reported stratified (non-generic noise-estimated,
generic single-source), as a flag, not a language drop. (b) The checked sample shows an error rate
above 5% (observed, §5.1 step 4's convention; for the 40 external-labelled sites, ≥ 3): the C#
recall is flagged noise-limited on that stratum. Rejected: marking C# single-source, which would
discard the estimate for the ≥ 84% of sites where the reference speaks (422 of 500) to protect a stratum that
the audit above checks directly. Raw pilot coverage 0.86 (Wilson 95% [0.74, 0.93]) does not separate
the corpus from 0.90 on its own; the sample-level figure (0.844, 422/500, Wilson [0.81, 0.87]) is
the one that sets the comparable share.

**C++ (googletest kept; priors).** Of the 22 silent pilot sites (all classified from the pilot files):
18 are calls inside a test-macro argument list (`EXPECT_*`, `ASSERT_*`, `SCOPED_TRACE`,
`EXPECT_CALL`), including all 13 the harness left unclassified; 6 sit under non-guard `#if`
(other-OS branches; one is a macro invocation itself, one a macro-argument site); the tree-sitter ERROR tag is
not the cause (all 4 tagged sites are macro-argument sites with an occurrence present). `scip-clang`
does emit the callee's symbol at these macro-argument sites, but at the macro invocation's columns
(for `EXPECT_EQ(x, Print(..))`, `Print` at the columns of `EXPECT_EQ`), so the strict
position mapping reports them silent. §5.2's macro-argument rule recovers 15 of the 22 (13 untagged
sites: 11). The remaining 7: the 5 uncompiled-branch sites, an `EXPECT_CALL(..).WillOnce` chain and
a `std::numeric_limits<float>::quiet_NaN` call. Pilot: 28 + 15 = 43 of 50 raw (0.86), 43 of 45
adjusted (0.956, Wilson [0.85, 0.99]); 3 of the 15 rest on the type-and-constructor clause of §5.2,
and without it the pilot is 40/50 raw and 40/45 = 0.889 adjusted, below 0.90. The pilot (n = 45) is
too small to carry the verdict alone; the 500-site sample, which the committed `ref.jsonl` gives
and which does not depend on the pilot, sets it. Over the 500 sample sites: 196 silent strictly;
the rule recovers 123 of them (a name-and-window match, whose false-match rate is unmeasured; 10 rest
on the type-and-constructor clause) and leaves 73: 29 under non-guard `#if`, 9 macro-callee sites
outside `#if`, 3 `ambiguous` and 32 in no class. Comparable share 427/500 (0.854); adjusted
coverage 427/462 = 0.924 (Wilson [0.90, 0.94]), and 417/461 = 0.905 (Wilson [0.87, 0.93]) without
the clause, where one more formerly covered site becomes a by-construction exclusion. The verdict
passes with or without the clause, narrowly without it, and the pilot's marginal pass is not relied on.
Decision: googletest stays, the mapping rule and the class list above apply.
Rejected: changing criteria to find another corpus. Every other candidate in `corpora.yaml` fails a
criterion that guards something real (spdlog: a Kafka client, §3.3 rule 4; yaml-cpp 82.9k and quill
69.2k lines: C3's memory ceiling, unmeasured for Joern; pistache: needs a libcurl the truth image
lacks, and building without tests leaves the test files silent; leveldb: C5 0.66 and C7), and any
C++ corpus with C8's tests and C6's ≥ 50 function-like macros has the same macro-argument
structure. A search would not escape it, so I do not ask for one. Rejected: single-source, which
would drop the estimate where hand labels are most error-prone.
*Stopping conditions.* (a) At S57, the hand-vs-reference disagreement at `relaxed` sites that adjudication
finds to be the reference's error exceeds 10%: the macro-argument rule is withdrawn, those sites
become by-construction silent, and the comparable share is recomputed. (b) The comparable share of
the labelled sites is below 0.80 after (a): return to the coordinator, as the C++ noise estimate no
longer rests on enough sites to continue as designed.

**C6 thresholds (idiom present enough to be sampled at its natural rate, not hand-picked):**
- **Java:** ≥ 20 DI or framework annotations on types or constructors (`@Component`, `@Service`,
  `@Inject`, `@Autowired`, `@Bean`, or JAX-RS/CDI equivalents) and ≥ 10 interfaces with ≥ 1
  in-corpus implementation.
- **Kotlin:** ≥ 20 `suspend fun` and ≥ 5 type-safe builder or DSL lambdas with receiver (Ktor
  routing, Gradle-style builders, `apply`/`with`-heavy configuration), plus the Java DI condition
  if the project is Spring-based.
- **C#:** ≥ 10 attribute-routed actions (`[Route]`, `[HttpGet]`…) and ≥ 2 projects in one solution
  (the cross-project type weak spot).
- **Go:** ≥ 10 interface types with ≥ 2 in-corpus implementations each (checked by the reference's
  implementation relation), and at least one generated-code directory if the project has one
  (it is excluded from the frame but measured in the census metrics).
- **C/C++:** C++, not C only. ≥ 50 function-like macro definitions used in the source, and ≥ 10
  virtual methods with an override.

**Selection rule among qualifying projects:** prefer the one whose layout looks like ordinary
product code (services, libraries, CLIs) over tutorials and samples. If two qualify, pick the
smaller. **Order in S25, per candidate:** C1–C4, C6–C8 by inspection and build, then C5 (the reference run), then and only then the first Joern run. The pick is never revisited after its first Joern run, so that the corpus is not
chosen for its result.

### 3.3 Exclusions (any one excludes)

1. The OpenTelemetry demo, any fork, and the `open-telemetry/*` SDK and instrumentation
   repositories themselves as corpora (the demo's services are built on them).
2. Google's Online Boutique (`GoogleCloudPlatform/microservices-demo`) and its forks.
3. **Demo-genre applications:** any polyglot, multi-service demo or reference application, e.g.
   `dotnet/eShop`, Sock Shop, DeathStarBench, `spring-petclinic-microservices`. A
   single-service, single-language app of the same family (the monolithic `spring-petclinic`) is
   not excluded by this rule.
4. **The held-back profile itself** (ML §4.5 condition 3: "a different shop's style, with a
   messaging stack and an in-house-looking wrapper"). That describes real product code, which
   rules 2–3 do not catch, so: **no candidate that uses a message-broker client** (Kafka,
   RabbitMQ/AMQP, NATS, Pulsar, SQS/SNS, Google Pub/Sub, Azure Service Bus, or an equivalent), by
   dependency or by import. P0(a) needs no messaging, so this costs nothing and keeps the
   held-back profile's defining trait out of every P0(a) corpus.
5. **The held-back candidate register.** Every project named anywhere as a candidate for the
   held-back codebase is excluded by name. Today that is Online Boutique (ML §4.5). There is no
   other list yet. If one is started, it must live in the ML note's §4.5 or a successor, and S25
   checks the register as it stands on the day of the check.
6. **Whole repository roots only.** A corpus is the root of its own repository. It is never a
   subdirectory of a larger repository, so a subdirectory of an excluded repository cannot get in.
7. Anything vendored by, or vendoring, an excluded project.
8. Anything already used to build or test this system (the fixture repos of S35–S39, the slice's
   inputs other than the two fixed corpora), so that fitness is not measured on tuned code.

**The reverse rule.** The five external P0(a) corpora, and their forks, are **ineligible as the
held-back codebase**. Its whole value is that nobody building the system has seen it, and P0(a)
corpora are seen by `graph-dba`, the labellers and `data-scientist`. S27's report lists them under
"ineligible as held-back". Whoever chooses the held-back codebase must check that list.
(`data-scientist`, as the ML note's owner, carries this rule into that choice.)

S25's gate checks 1–8 by name and dependency list. It also greps the corpus for `opentelemetry`
and `microservices-demo` (module lists, `go.mod`, `pom.xml`, `*.csproj`, and source). An
`opentelemetry` hit is not an exclusion by itself: depending on an OpenTelemetry SDK as a library
is allowed. It is a flag that the confirming `data-scientist` resolves in writing (K-P0-11).

## 4. Sample size and the seeded draw

### 4.1 The frame (Joern-independent, F1)

S25 enumerates frame sites with tree-sitter, from the pinned corpus source, excluding test-output,
build-output, vendored, dependency and generated directories. A file is generated if its first
5 lines carry a `generated`/`DO NOT EDIT` marker or it sits under a configured generated path.
Node types:

| Language | Frame node types |
|---|---|
| Python | `call`, **except** a `call` that is the expression of a `decorator` node |
| TS/JS (`.ts`, `.tsx`, `.js`, `.jsx`, `.mjs`) | `call_expression` (tagged templates included, as bundle spec §5.6 counts them), `new_expression`, and `jsx_opening_element` / `jsx_self_closing_element` whose tag starts with an uppercase letter (F8); **except** a call inside a `decorator` node |
| Java | `method_invocation`, `object_creation_expression`, `explicit_constructor_invocation` |
| Kotlin | `call_expression`, `delegation_specifier` constructor calls |
| C# | `invocation_expression`, `object_creation_expression` |
| Go | `call_expression` (conversions such as `T(x)` are labelled `not_a_call`) |
| C/C++ | `call_expression`, `new_expression`; a function-like macro invocation parses as `call_expression` and stays in the frame |

**Decorator and annotation applications are out of the frame** (review P2-4). Bundle spec v1.1
§5.6 makes them `decorator` records, not call sites, so a recall figure over them would measure an
edge the bundle never emits as a call. They are measured instead in the decorator/annotation
shape census (§5.3, §7.2), which is what S30 needs. A call *inside* a decorator's arguments
(`@app.get(make_path())`) is also out, for simplicity; this is reported as part of the blind spot
below.

**Out of the frame, stated as a blind spot:** implicit calls (property accessors in C# and
Kotlin, operator overloads, implicit constructors, Python dunder protocols, destructors). The
recall figure does not cover them; §10 R2.

Each frame site carries a harness-only **kind** (plain function, method on receiver,
qualified/static, constructor, function value / callback, macro, JSX component, tagged template), from
the tree-sitter shape. The kind is a reporting domain (§7). It is never shown to the labeller.

### 4.2 Simple random sample, not stratified (F2)

One sample per language, shared by both configurations, with every site equally likely. Rejected:
stratifying by resolved vs unresolved (plan §4.3's suggestion). Stratum membership would come
from one configuration's output and be wrong for the other. Weighted estimators would replace
plain Wilson intervals. And the resolved/unresolved split is already observed post hoc on an SRS,
with each part a uniform sample of its own subpopulation. Rejected also: a top-up sample of
resolved sites for precision. Those sites would have to be either marked to the labeller (a
blindness leak: "Joern found something here") or hidden in the queue, which breaks §4.3's
stopping rule.

### 4.3 Size and stopping rule (inverse sampling, F3)

The labeller labels sites **in ordinal order, never skipping**, and stops at the first of:
- the **150th internal site** (a site whose label lists ≥ 1 in-corpus callee); or
- the **500th site** labelled (the cap, the same for every language).

Because the stop depends only on the internal/external flag of the labels, not on anything Joern
produced, the internal sites labelled are a uniform random sample of the language's internal
sites. Plain Wilson intervals apply to recall. The external sites labelled are likewise a
uniform sample of the external sites. (The internal/external mix in the labelled prefix is a
slightly biased estimate of the population mix, because of inverse sampling. `P_c` pools the
labelled internal and external resolved sites, so it inherits that bias. Its size is about one site
in 150, far below the precision interval's width, and it is ignored.)

**Why 150 internal sites.** Wilson 95%, recall gate "observed ≥ 0.98", which at n = 150 means
≤ 3 misses:

| Misses of 150 | Observed | Wilson 95% |
|---|---|---|
| 0 | 1.000 | [0.975, 1.000] |
| 1 | 0.993 | [0.963, 0.999] |
| 2 | 0.987 | [0.953, 0.996] |
| 3 | 0.980 | [0.943, 0.993] |
| 4 | 0.973 | [0.933, 0.990] |

Probability the gate passes (binomial, "≤ 3 misses of 150") at true recall r: r = 0.995: 0.99;
0.99: 0.94; 0.985: 0.81; **0.98: 0.65**; 0.97: 0.34; 0.96: 0.15; 0.95: 0.06; 0.93: 0.01.

- So it separates a ≥ 0.99 language from a ≤ 0.95 language with ≥ 94% reliability each way.
  That is the decision P0 needs: which languages need work before P1/P2.
- **It cannot certify 0.98.** A Wilson lower bound ≥ 0.98 needs ≥ 189 internal sites with zero
  misses. At true recall 0.99, 189 sites come out miss-free only about 15% of the time, so a
  "lower bound ≥ 0.98" rule would fail almost every good language. A true-0.98 language is a coin
  flip under any feasible n.
- **Rejected: 200 internal sites, "≤ 4 misses".** Its operating curve is almost the same
  (0.95 pass at 0.99, 0.03 at 0.95) for a third more labelling. The extra lower-bound headroom
  (0.950 vs 0.943 at the threshold) buys nothing the decision uses.
- **Floor:** if the 500-site cap stops a language below **100 internal sites**, recall is reported
  as *insufficient sample* and the language is treated as failing (§8.1). At 100, zero misses
  gives a lower bound of 0.963. That is still a usable screen, with "≤ 2 misses" as the rule.

**Cap: 500 sites, everywhere (m1).** An earlier draft used 400, with a "same-name definition
share" predictor to raise it. That is dropped. The predictor counts every name collision (`get`,
`append`) as internal, so it overstates the share and fires too rarely. And at 400 the trigger
point (0.375) is the coin flip: a 400 cap reaches 150 internal sites only 52% of the time there.
With 500 for every language (binomial, reviewer's Appendix A3, reproduced):
- 150 internal sites are reached with probability 0.52 at an internal share of 0.30, 0.90 at
  0.326, and ≥ 0.99 at 0.35;
- the 100-site floor fails with probability 0.48 at a share of 0.20, 0.004 at 0.25, and ≈ 0 at
  0.30.

The cap, like the stop, depends only on the labels' internal flag, so the draw stays independent
of Joern.

**Capacity, the cost of 500.** Worst case per dispatch: 150 internal sites (each needing the
definition located and, for a virtual call, the implementations grepped) plus up to 350 external
or `notACall` sites (each needing only an import read). This is one long, resumable session (M6's
append-and-resume), longer than the 400 draft by up to 100 cheap sites. A language that ends below
150 internal sites at 500 is reported with its actual n under §8.1's rule.

### 4.4 The seeded draw

The draw is a **hash sort**, not a PRNG shuffle, so that any tool can reproduce it and it does not
depend on frame order or on the Python version:

```
seed     = "ccg-p0a-v1|" + lang + "|" + corpusCommitSha
siteKey  = path + ":" + line + ":" + column + ":" + callText      (§2; path relative to corpus root, '/' separators, NFC)
order    = sort frame by  SHA-256(seed + "|" + siteKey)  as hex, ascending; ties (none expected) by siteKey
sample   = first 500 of order, with ordinal 1..500
```

- `sample/<lang>.jsonl` holds, per site, only `ordinal`, `siteId` (first 16 hex of the hash), `path`,
  `line`, `column`, `callText` and a 7-line source excerpt (3 before, 3 after).
- `sample/<lang>.private.jsonl` (harness-only, §6.2) holds the kind, the tree-sitter node type and
  the frame size.
- Duplicate site keys (two calls with the same callee text at the same column, impossible in
  practice) get an ordinal suffix `#2`.
- The symbol, golden-method and census populations (§7) use the same hash-sort rule with the seed
  prefixes `ccg-p0a-v1-sym|` and `ccg-p0a-v1-flow|`. The C5 pilot (§3.2) uses `ccg-p0a-v1-c5|`
  over the candidate's frame (first 50), and the spurious-pair sample (§8.4) uses
  `ccg-p0a-v1-spur|`.

## 5. Truth source, per language

### 5.1 Design: two independent sources, adjudicated (F4)

For every labelled site, truth = **the blind hand label (S26), checked against a typed reference
answer**, and adjudicated wherever they differ:

1. The **reference** runs in S25 (§11 N1), on each candidate before any Joern run (§3.2 C5), in the image, with network and with the corpus's dependencies installed
   (it is a truth tool, not the system under test, so egress rules for local-only builds do not
   apply, as with D4's corpus fetch). It gives each site's **static target**. Where the tool emits
   an implementation/override relation, it also gives the **dispatch expansion**.
2. The harness compares, per site, the hand label with the reference. **Agree** (same
   in-corpus target set, or both external): that is the truth.
3. **Adjudication** (procedure: §5.4) covers every site where they disagree, every site labelled `unsure`, and
   **every site Joern misses** in either configuration, even where label and reference agree.
   - **Disagreements and `unsure` sites** are adjudicated by `data-scientist` in a **separate
     pre-Joern step** (§11 N4, checked by `analyst`), from a file that has no Joern columns, before
     any Joern result is opened. The step boundary makes that ordering checkable.
   - **Joern misses** are adjudicated in S27, necessarily with Joern's answer in view, by
     `data-scientist`, into a miss-adjudication log that `analyst` checks independently (§11 N5).
   - A miss is overturned only on **cited source evidence** (file:line of a definition, or the
     dispatch path) that the truth set was wrong. This keeps the anchoring toward Joern's answer
     from erasing real misses.
   - Every adjudication records the decision, the evidence and which source was wrong.
4. **Reported:** the hand-label vs reference agreement rate, with a Wilson interval, and the
   adjudicated error rate of each source, and both per stratum (the `relaxed` and by-construction
   classes of §3.2/§5.2, and the rest). This is the label-noise figure. Whether a language's recall is flagged
   noise-limited is decided only by §5.4 step 7 (residual error in the final labels, e_res > 0.05); the
   labeller's own error rate is reported there beside it and gates nothing.

Residual error: a site where both sources are wrong *the same way*, for example a forgotten
implementation that the reference also does not relate. This is most likely for callback and
function-value sites, where SCIP-style indexers give no dispatch set. It is reported as a
limitation by kind, not corrected.

Rejected:
- **Hand labels alone** (plan §4.6 as written): no estimate of label noise, at a gate whose
  tolerance is 2 points.
- **The reference alone, with a hand-labelled subsample** (plan §4.3's alternative): every
  reference shares the blind spot that matters most, callbacks and dispatch sets, so the
  hand labels must cover the whole sample, not a subsample.
- **Runtime traces of the corpora's own tests** (ML §3.6's other option): the traced sites are the
  ones the tests exercise, which is a biased subset; tracing differs for all seven runtimes; and a
  trace shows observed targets only, not the possible-target set `T(s)`. Not worth seven harnesses.

### 5.2 The reference, per language

These tools' current maintenance status was **not verified** for this note (Sourcegraph's indexer
table lists all of them; [the SCIP project](https://github.com/sourcegraph/scip) names most).
**S24 pins each tool to a version and proves it on a smoke project** (a few files per language,
exercising one interface or override; `devops` picks them), not on a corpus: corpora are chosen
in S25, after S24. **C5 is checked in S25** (§3.2). A tool that fails S24's smoke test is replaced by
the listed fallback **only if that fallback is an installed tool**. Otherwise the language falls back
to hand labels alone, flagged *single-source*, with recall marked noise-unestimated (§5.1 step 4
cannot run).

**Fallbacks that would be new code.** These are programs someone would have to write, not tools to
install, so they are listed for the plan owner and are **not** assumed in this protocol: (1) a
Pyright go-to-definition driver (an LSP client) for Python; (2) a TypeScript language-service
driver for TS/JS; (3) a Roslyn (`Microsoft.CodeAnalysis`) driver for C#; (4) a clangd-index reader
for C/C++. Whether any of them gets a step is `architect`'s call. Without one, the fallback for those
four languages is *single-source*.

| Language | Reference (static target) | Dispatch expansion | Needs in the image | Fallback |
|---|---|---|---|---|
| Python | `scip-python` (Pyright-based) | SCIP `is_implementation` where emitted; else hand label only | Node.js LTS + npm; the corpus's own deps installed in a venv so that Pyright sees types | single-source (fallback program 1 would be new code) |
| TS/JS | `scip-typescript` | SCIP `is_implementation` | Node.js LTS + npm; `npm ci` in `salesperson/` | single-source (fallback program 2 would be new code) |
| Java | `scip-java` (SemanticDB) | SCIP `is_implementation` (overrides) | JDK 21 (present), Maven/Gradle (in §7); dependency download at index time | `javac -Xplugin:semanticdb` (installed tool), then scip-java's SemanticDB-to-SCIP conversion |
| Kotlin | `scip-java` with the SemanticDB Kotlin compiler plugin | SCIP `is_implementation` | as Java, plus scip-java's Kotlin compiler plugin (formerly scip-kotlin, now `scip-kotlinc` inside scip-java) at a version matching the corpus's Kotlin version (C7) | single-source |
| C# | `scip-dotnet` (Roslyn) | SCIP `is_implementation` | .NET SDK (in §7), matching the corpus's target framework; `dotnet restore` with network | single-source (fallback program 3 would be new code) |
| Go | `callgraph -algo=vta` from `golang.org/x/tools/cmd/callgraph`: it is a call graph, so it also gives interface and function-value dispatch | built in (VTA) | Go toolchain (in §7) matching `go.mod`; `go mod download` with network; `go install golang.org/x/tools/cmd/callgraph@<pinned>` | `scip-go` for static targets, with hand labels for dispatch |
| C/C++ | `scip-clang` over `compile_commands.json` | SCIP `is_implementation` (overrides) | clang/LLVM matching `scip-clang`'s needs, libstdc++ headers, CMake (in §7); `compile_commands.json` from `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`. **Parallelism 2** (`--jobs`/`-j`, per the pinned version's `--help`): its README asks for about 2 GB RAM per core, and the 6 GB host needs headroom; it runs with no Joern or FalkorDB step alongside (plan §5.1) | single-source (fallback program 4 would be new code) |

Common: the **`scip` CLI** (`scip print --json` or `scip snapshot`) to read every SCIP index, and
**py-tree-sitter with these ten grammars** (`python`, `typescript`, `tsx`,
`javascript`, `java`, `kotlin`, `c_sharp`, `go`, `c`, `cpp`; e.g. via `tree-sitter-language-pack`)
for the frame and for census truths. tree-sitter is a Python dependency: it belongs in
`cpg/pyproject.toml`'s `spike` extra (S09 declares every P0 dependency), and in the image if the
harness runs there.

**Mapping a reference answer to a site.** The SCIP occurrence whose range covers the callee
identifier at the site's line (the last identifier of the callee expression; for a constructor, the
type name), with role reference. Its symbol's definition occurrence gives (file, line) and the
symbol gives the qualified name. Go's `callgraph` edges carry the call position (file:line:col),
matched to the site by file, line and callee name. A site with no reference occurrence is
*reference-silent* and is adjudicated if the hand label says internal.

**C/C++ macro-argument sites.** A site inside the argument list of a macro-shaped call maps to a
non-local, non-definition occurrence in the window from the first line of its outermost enclosing
macro invocation to the site's name line, whose last descriptor name equals the site's callee name.
One distinct symbol in the window: that is the site's `static`, with `relaxed: true`. Two or more
distinct symbols: `silent`, tagged `ambiguous`, and not counted as covered. **Type and constructor:**
the pair `P/X#` and `P/X#X(h).` that `scip-clang` emits at one token counts as one symbol (flagged
`typeCtorMerged`): the brief's target for a constructor call is the type, so both name the same
in-corpus entity. False-match risk: a window mention of the type that is not the site's call, for
example `X` as a type argument beside a call named `X`; the labelled target would still be the same
class, so the cost is a falsely agreeing site, not a wrong target. The clause applies to 10 of the 500
sample sites and 3 of the pilot's. Stopping condition: at S57, if ≥ 3 of those 10 sites are
adjudicated as reference errors, the clause is withdrawn and they become `ambiguous`. The strict position
mapping applies everywhere else. The agreement rate is reported separately for `relaxed` sites.

### 5.3 Census truths (no labeller)

| Metric | Truth |
|---|---|
| Parse success | the file list of the frame's language, after exclusions |
| Line fidelity, method coverage | tree-sitter definition nodes: name, start line (of the name's line, decorators and annotations excluded), end line |
| Value-set recovery | tree-sitter argument nodes that are a string literal, a literal-only concatenation, a template literal without substitutions, or a same-file constant (`final`/`const`/`val`/module-level `UPPER_CASE` in Python) initialized by one of those |
| Decorator/annotation shape | tree-sitter decorator/annotation/attribute nodes attached to definitions, with their literal arguments |
| Fan-out | Joern itself: a count over Joern's own name-match candidates, so no external truth. Methods Joern failed to emit lower fan-out; method coverage is reported beside it so that this is visible |

### 5.4 Adjudication procedure (S57), one fitness custodian per language, in parallel

**Roles.** A fresh `qa-engineer` fitness custodian (FC) per language does the mechanical work and
reads source; `data-scientist` (DS) decides every judged item and returns decisions keyed by
`ordinal`; the FC writes them down. Neither opens `results/`, a CPG or any Joern output
(`analyst` checks both transcripts). **Inputs:** `labels/<lang>.jsonl`, `truth/<lang>.ref.jsonl`,
`labeller-scripts/<lang>/` (provenance only, as evidence of how labels were made) and the corpus tree
`cpg/.ccg-work/corpora/<lang>/src` (the evidence for every decision is a file:line there). Part F
(flow labels, below) adds `sample/<lang>-flow.jsonl` and `labels/<lang>-flow.jsonl`, an extension of
the S57 row's input list that Part F's order of reads makes safe. **Outputs:**
`adjudication/<lang>.input.jsonl` (the script), `adjudication/<lang>.audit.jsonl` (the draws and
their outcomes), `adjudication/<lang>.jsonl` (decisions), `adjudication/<lang>.noise.json`,
`adjudication/<lang>-flow-audit.jsonl`, `truth/<lang>.final.jsonl`. Already fixed by the S25 gate
(`docs/reviews/connected-code-graph-frontend-fitness.md`) and not re-checked here: the input script's
firewall and its `label-notacall-reference-call` reason (F11), `flowcheck`'s endpoint and evidence
checks (F1), the C5 figures (F3, F9) and the contact list (G1, G2).

**Step 1: compare.** Run `make_adjudication_input.py <lang>`; it must exit 0 and report the labelled
count (python 240, javascript 523, go 373, java 500, csharp 313, cpp 187, kotlin 430). Every site
with `status: needs-adjudication` is **adjudicated, with no sampling**: today 328 sites (python 73,
javascript 43, go 35, java 16, csharp 44, cpp 89, kotlin 28). `unsure` and `dynamic` are reasons
whatever the reference says.

**Step 2: strata of the agreed sites.** Every `agreed` site is in exactly one stratum, by the
reference's kind and the label:
- **AI** agreed internal: the label lists an in-corpus target (internal or declared) and the sets match;
- **AE** agreed non-internal: no in-corpus target on either side, the reference being `external`,
  `local` or a dispatch without an in-corpus target, the label external or `notACall`;
- **AS** agreed against a silent reference: no in-corpus target on the label, the reference
  `silent`. This is **single-source**: the script calls it agreed, but the reference gave no
  second opinion, so it is excluded from the comparable share and from the agreement rate.
The **comparable share** is the labelled sites whose reference is not silent; the **agreement
rate** is agreed ÷ comparable, Wilson 95% (S25 gate §7: Java and Go will read lower than their
pilots).

**Step 3: audit the agreed strata (the shared-error and scripted-label check).** Agreement shows only
that two sources agree, so a sample of each agreed stratum is re-decided from source by DS as if it
were a disagreement. Draw within a stratum by ascending SHA-256 of `ccg-p0a-v1-aud|<lang>|<stratum>|`
plus `siteId`, the first n (draw U). A second draw C takes, per **cell** of the stratum not hit by U
(cell = label kind × package or `defPath`, or `notACall`), its first site; C is read and corrected
but **reported apart and never pooled into the rate**. Sizes (all of a stratum if it is smaller):

| Tier | Languages | AI | AE | AS |
|---|---|---|---|---|
| standard | python, go, csharp, kotlin, cpp | 60 | 60 | 30 |
| scripted externals (JS) | javascript | 100 | 100, of which ≥ 40 from the reference-`local` cell | all (1) |
| helper-resolved targets (Java) | java | 100 (a census, N = 100) | 100 | 40 |

JS gets the larger tier because its labels came from a Python rule table written after a full read,
not site by site (the second labeller of the old and new runs both used one, so their 55/56 test-retest
agreement is not independent evidence), all 523 are `sure`, and the table was not kept
(`labeller-scripts/javascript/` is absent). Java gets it because it too has 0 `unsure` and 0 `dynamic`
over 500 sites, against 71 of 187 for C++ (S25 gate §5.1); `lab.py` only indexes definitions from source text
and the call decisions are hand-written, so the audit looks for helper mistakes (overload fan-out, a
wrong `defLine`), not for correlation with the reference. Zero `unsure` is treated as uninformative
about quality, never as evidence of it.

*Decision per audited site:* DS reads the source and records `labelCorrect` true/false with
file:line evidence. An error is a **label** error, whatever the reference says; it is corrected in `final.jsonl`, and what it adds to the flag is set in step 7. *Threshold:* a
stratum is **flagged** iff residual errors (step 7: not a closed class)/n > 0.05 in draw U, which is ≥ 2 of 30, ≥ 3 of 40, ≥ 4 of 60,
≥ 6 of 100 (flag probability at a true 2%: 0.12, 0.05, 0.03, 0.02; at 8%: 0.70, 0.63, 0.72,
0.82; with no error the Wilson upper bound is 0.11, 0.09, 0.06, 0.04). *Sibling sweep:* on any
error of kind `target` or `kind` (a `role`, `coordinate` or `shape` error needs no sweep: step 4's census and the schema check already cover every sibling), the FC lists every site of the language whose label shares the erroneous site's cell or
callee name, and DS re-decides up to 10 of them (all if ≤ 10, else the first 10 by the same
hash); overturns found are applied but do not enter the rate.
*Stop and escalate:* a flagged stratum is extended once (census if N ≤ 150, else a second draw of n);
if the pooled rate is still above 0.05, the stratum is **noise-limited**: the language's recall carries
that flag (§5.1 step 4), S57 for the language stays open, and the FC returns to the coordinator with
the per-cell error table and the cost of relabelling the stratum with a fresh labeller.

**Step 4: the declared and role census (all languages, no sampling).** Rule 8's role of a target
(§2, §6.5's harness mapping) needs the declaring source, which the script may not open, so the FC
does it: take every distinct `(path, defLine)` that appears as a label callee of kind `declared` or
`internal`, or as the reference's `static`/`impls` (62 to 143 per language, plus 0 to 31 declared:
Java 62 + 31, Kotlin 143 + 13, C# 115 + 10, Python 126 + 2, Go 126 + 4, C++ 124 + 3, JS 102 + 0).
For each, the FC reads the lines at `defLine` and records body or bodyless and the declaration form;
DS applies the language's row of bundle spec3 §4.1 rule 8 only where the form is not plainly a body:
symbol → role `decl`; substituted (C/C++ prototype to definition, C# partial) → that definition,
role `impl`; otherwise `N(s)`. A label `internal` at a bodyless form, or `declared` at a body,
is corrected and **logged as a label error with kind `role`** (§6.5). The census also reads the line at
every `defLine`: the right entity with a wrong `defLine` or `qname` is a `coordinate` error, corrected
the same way. Both are **closed classes** (step 7). The 76 label-`declared` and
81 reference-`declared` sites of Java, Kotlin and C# are in step 1's adjudication when they
disagree and in this census always.

**Step 5: language-specific strata (all adjudicated or checked in the same pass).**
- **C++ macro invocations.** 53 labelled sites (28%) are function-like macro invocations
  (`macro-shaped`), 51 of them `unsure` with a partial list from the labeller's first expansion
  level; they are 24 distinct macros (`EXPECT_EQ` 12, `EXPECT_THAT` 10, `EXPECT_TRUE` 6, ...). A
  macro is not a symbol (rule 8), so the truth of such a site is the in-corpus symbols its
  **first-level expansion** calls. DS decides **per macro name**, once, from the `#define`, and
  applies it to every site of that macro; the site's final set is that set, with an overload
  chosen only when the argument types fix it from source. Where they do not (the labeller's
  `Compare` overloads), the site is `scored: false`, `notScoredReason: macro-underdetermined`, counted and
  reported apart as `dynamic`-with-empty-set is (§2), and the recall is reported with and without them
  (R7). *Escalate* if the underdetermined share exceeds 0.30 of the labelled C++ sites: C++ recall
  would describe a remainder, and the scope of the C++ measure is the coordinator's. The other 20
  `unsure` non-macro C++ sites are adjudicated one by one; the three C++ labels with empty
  `callees` and no `notACall` are label errors of kind `shape` unless DS finds the site is not a
  call (then `notACall`).
- **`relaxed`, `ambiguous`, `byConstruction` reference rows** (§3.2, §5.2; C++ 46, 2, 14; Go
  13; C# 50). The agreement rate and the adjudicated reference-error rate are reported per
  stratum. §3.2's withdrawal conditions are evaluated here: the share of `relaxed` C++ sites
  where DS finds the **reference** wrong must not exceed 0.10, and ≥ 3 of the sites that carry
  `typeCtorMerged` (read from `ref.jsonl`; the script does not pass it on) being reference errors
  withdraws that clause. Either result is returned to the coordinator, who reapplies §3.2. The test is on the point estimate; the report states k/n with its Wilson 95% interval and the margin to the cap (C++ at 4/46 = 0.087, interval 0.034 to 0.203, one site below the cap), and never presents the clause as validated by a pass, since the interval admits a true rate above 0.10. The `typeCtorMerged` count is reported whether or not it reaches 3.
- **C# generic and `nameof` stratum.** The FC runs `run_fitness.py genaudit csharp --select` (the S57
  step of §3.2): every such site the label calls internal, plus the seeded 40 labelled external
  (prefix `ccg-p0a-v1-gen|`), re-decided by DS from source. Flag if the labelled-external 40 show
  ≥ 3 errors or any internal-labelled error rate exceeds 0.05; the stratum's share of the labelled
  internal sites over 0.10 makes C# "stratified, generic single-source" (§3.2). This audit stands
  beside, and is not part of, step 3's AE/AI draws.
- **Go dispatch.** A reference of kind `dispatch` (VTA edges) is compared with the label as a set.
  DS decides by §2's virtual-call rule (the declared type's method plus every in-corpus
  implementation) and records which source was incomplete.
- **Python sample length.** The labeller stopped at ordinal 240 on its own 150th internal-or-declared
  site (148 internal, 2 both); the reference holds 129 internal sites in those 240 rows because 25 labelled
  internal are reference-silent. After adjudication the **final internal count** n_int is computed.
  n_int ≥ 150: nothing. 100 ≤ n_int < 150: proceed and report the actual n (§4.3). n_int < 100: the
  labeller resumes in ordinal order, in batches of 50, each batch run through this procedure, until n_int
  ≥ 150 or ordinal 500. That stop depends on labels and adjudication, never on Joern, so the prefix
  stays a uniform sample. The contact strata floor of §3.1 is checked on the same final truth and
  belongs to S27.
- **JS reference-`local`.** 144 of 523 reference rows are `local` (callables held in local variables,
  which the labeller called external or resolved to a defining arrow); 23 disagree and are adjudicated;
  the agreed ones are audited by step 3's ≥ 40 rule.

**Step 6: decisions and the final label.**
- **A decision** (`adjudication/<lang>.jsonl`, one line per adjudicated, overturned or swept site) holds
  `ordinal`, `siteId`, the reasons, the decided set (`callees` with `role` impl/decl, `nonSymbol` for
  `N(s)`, `notACall`, `dynamic`), `wrong` (`label`, `reference`, `both`, or `neither` when the
  reference was silent and the label right), `errorKind` (`target`, `kind`, `role`, `coordinate`, `shape`, `none`; `shape` (the one name for it in steps 5-7, `noise.json` and decisions) is a label line breaking §6.5's schema, such as empty `callees` with neither `notACall` nor `dynamic`),
  and `evidence`: at least one `path:line` of the corpus, ≤ 300 characters. A decision without a
  citation is not a decision (§5.1 step 3).
- **A final label** (`truth/<lang>.final.jsonl`) exists for **every** labelled site, once: `T` (path,
  qname, defLine, role), `N`, `external`, `notACall`, `dynamic`, `scored`, `notScoredReason`, `basis` and
  `strata`. `basis` records the **procedure that produced the line**, by this precedence: `sweep-overturned`, `audit-overturned` or `adjudicated` (the decision changed or settled the line); else `audit-confirmed` (DS re-read the site from source in draw U, draw C, the C# generic audit, step 3's audit or a sibling sweep and found the label right, **in any stratum, AS included**); else `label-only` (an AS site nobody re-read: the label stands alone); else `agreed` (an AI/AE site nobody re-read, roles from step 4). The stratum is a separate fact, carried by an `AS`, `AI` or `AE` entry in `strata` on every agreed-class site. `scored` is false for `notACall`,
  for `dynamic` with empty `T`, for `macro-underdetermined`, and for **`underivable`**: DS read the source
  and neither source nor the code settles the target. Unscored sites are counted and reported by reason;
  an `underivable` share above 0.05 of a language's labelled sites is escalated to the coordinator.
- **What S27 receives** is this file as `final`; S27 applies miss overturns on top as `final2` (§11 N5).

**Step 7: the label-noise figure (`adjudication/<lang>.noise.json`).** Computed by the FC from the
files, with N the labelled sites:
- comparable share and agreement rate (agreed ÷ comparable, Wilson), overall and per stratum
  (AI, AE, `relaxed`, `byConstruction`, `declared`, macro);
- adjudicated counts by `wrong` and `errorKind`; W = sites with the label wrong among the adjudicated;
- audit table per stratum: n, errors k, Wilson, population N_s, and draw C's cell errors;
- **Two figures, always both reported**, each with every stratum's Wilson upper bound for k_s/n_s as the
  conservative bound; the **reference error rate** over the comparable sites likewise:
  - **e_all** = (W + Σ_s N_s·k_s/n_s) ÷ N over every error found, W counted as label-wrong when `wrong` is
    `label` or `both`. It is a property of the **labeller** and drives no gate; the report shows it beside e_res
    (and the label-only W variant as a spread), so a reader sees how much the adjudication had to repair.
  - **e_res** = Σ_s (N_s − n_s)·r_s/n_s ÷ N, with r_s the audited errors of stratum s that no complete
    detector corrects (kinds `target` and `kind`; a `role`, `coordinate` or `shape` error is excluded, below), and N_s − n_s the
    agreed sites of the stratum nobody re-read. It is the **expected number of label errors that remain in
    `final.jsonl`, over N**: adjudicated sites are decided from source and every audited site is corrected, so
    neither leaves a residual; what remains is the unread remainder of the agreed strata, whose rate the audit
    estimates (shared errors included, because DS re-reads from source). Census strata (n = N_s) contribute 0.
  - **The flag is a property of the final labels, and it uses e_res**: e_res > 0.05 makes the language
    **noise-limited** (§5.1 step 4) and a step 3 stratum is flagged on its residual errors. Basis of 0.05: it
    is the §5.1 threshold, set before any result, and at 0.05 the truth is uncertain by up to 5 points of N,
    2.5 times the gate's 2-point tolerance; it is a convention, not a derived optimum. e_all above 0.05 with
    e_res below it is reported as "labeller noisy, final labels repaired", never as noise-limited. The
    report also states that the corrected sites rest on DS's own source reading, whose error is not
    estimated by this procedure; `analyst`'s independent re-read of decisions (Pass 3: 0 wrong in 14 for C++) is the only evidence of it.
  Test-retest for JS is already reported (56 sites, 55 agree; ledger).
  - **`flagByVersion`, required in every `noise.json` and in the report, per language:** an object
    `{"v1.10": {"figure": f, "flagged": b}, "v1.11": {...}, "v1.12": {...}}`, exactly these three keys, each
    `figure` a number in [0,1] and `flagged` a boolean equal to `figure > 0.05`. `v1.10.figure` = e_all (every
    error found, W counted, closed classes included); `v1.11.figure` = the same with closed-class errors
    removed and W still counted; `v1.12.figure` = e_res, equal to the file's top-level e_res. A language whose
    `flagged` differs between versions is named in the report with the rule change that decided it (python,
    kotlin and C++ at the time of writing).
  - **C++ final labels rest on a single DS reading**: 90 sites adjudicated and 86 audited, each decided once
    by DS. `analyst`'s independent reads (18, 0 wrong, Wilson upper about 0.18) are the only check, and the report says so.
- **Closed class** (excluded from e_res because a complete scan found and corrected every instance, so
  nothing is extrapolated and nothing remains). An `errorKind` is closed iff all of: (i) it is `role`,
  `coordinate` or `shape`, a list this note extends only by revision; (ii) it is found by a **complete**
  detector, not by agreement or a sample: step 4's census read every distinct label and reference target and
  recorded a verdict (`noise.json` `census.targetsRead` equals the distinct `(path, defLine)` count of the
  labels, the reference and the final `T`), or the script's `shape` check ran on every label line; (iii) every instance is
  corrected in `final.jsonl` and listed in `noise.json` by ordinal; and (iv) the instances are at most
  **0.10** of the labelled sites. The 0.10 is a tripwire, not a statistic: twice the flag threshold, above
  which a labelling convention (not chance) is suspect and the language is returned; the highest share seen
  is 0.051 (Kotlin primary constructors, bodyless `decl` by bundle spec3 §4.1 rule 8, a spec-given
  convention). `target` and `kind` errors are never closed: a wrong entity, or a wrong internal/external
  call, that the reference shares is invisible to any scan (§5.1's residual error); they enter e_res only through the audit.
- **`noise.json` also carries `openPointsS27`**, one entry per language and never omitted: class-line
  targets (implicit constructors) with their sites and definitions, or the statement that the scan found none; anonymous-lambda
  indices in any `qname` (JS `<anon>#n`, Go, Python, Kotlin); and labels `external` for a package whose source is
  absent from the tree (Java OpenAPI-generated DTOs). S57 compares `(path, defLine)`, so these cost little; S27 needs the list.

**Stop and escalate (one list; "return" means to the coordinator, with the file named).**
1. The script rejects a file, or a count differs from step 1's: stop; return.
2. Any read of `results/`, a CPG or Joern output, or of another language's files: the unit is void; a
   fresh FC restarts it. **One waiver:** a read of a labeller-side row of another language (a queue or input row, not a decision or final file) by the FC is waived, not void, iff the coordinator records it and `analyst` re-derives every draw of the unit from `input.jsonl` alone and finds them identical; DS never having seen the row is also required. Anything else stays void.
3. Flagged stratum after extension, or e_res > 0.05 (step 7): `noise-limited`; unit stays open; return. A closed class above 0.10 of the labelled sites: return. A high e_all alone is a disclosure, not a return.
4. `underivable` > 0.05 of N; C++ macro-underdetermined > 0.30; C# generic stratum > 0.10 of the labelled
   internal sites; `relaxed` reference-error rate > 0.10 or `typeCtorMerged` ≥ 3 reference errors: return.
5. Python n_int < 100: resume the labeller (step 5), no return needed.
6. FC and DS read a declaration form differently: DS decides, the difference is logged.
7. Done when: every adjudicated item has a decision, every audit item an outcome, `final.jsonl` has N
   lines and validates, `noise.json` is written, Part F is run, and `analyst` has checked the transcripts and
   `final.jsonl` against the decisions.

**Part F: sample audit of the flow labels (S56), same unit, pre-Joern.** The seven golden sets are
12 to 30 methods, 151 to 483 pairs (js 12/151, java 30/461, go 30/433, csharp 30/394, cpp 30/344,
kotlin 30/476, python 30/483), written in 5 to 10 tool calls and 90 to 220 seconds each, so their
quality is unmeasured; §8.4's spurious-pair sample only bounds omissions after Joern. This audit
measures both kinds of error before any projection exists.
- *Precondition.* `run_fitness.py flowcheck <lang>` reports `nErrors` 0 and `complete`. At the S25 gate
  it did not for Java (16 evidence errors), Python and Go (one method each with no pair): the S56
  labeller repairs those first (gate Pass 2, F1 disposition).
- *F1: pair validity.* Draw 60 pairs (JS 40) by ascending SHA-256 of `ccg-p0a-v1-flowaud|<lang>|pair|`
  plus method ordinal and pair index, all `unsure` pairs first up to 20. DS checks each against the
  source and the brief's rules (direct flow; a control pair directly decided) and records valid or
  invalid with file:line. **Flag** iff invalid/n > 0.05 (≥ 4 of 60, ≥ 3 of 40); the `unsure`
  pairs are reported apart.
- *F2: omissions.* Draw 6 methods (JS 4) by the same hash with `method`. The FC relabels them from
  `sample/<lang>-flow.jsonl` and the flow brief only, in the same JSON schema and **before opening
  `labels/<lang>-flow.jsonl`** (the transcript shows the order; `teco` greps it). It then diffs the two
  sets; DS judges every pair on one side only from source. Omission rate = valid pairs the golden
  set lacks ÷ valid pairs in the union. **Flag** iff it exceeds 0.05 with ≥ 20 valid pairs (the
  rule of §8.4); fewer than 20 extends the draw by 3 methods once, then reports *insufficient*.
- *Corrections.* `labels/<lang>-flow.jsonl` is never edited. `truth/<lang>.flow-final.jsonl` is the
  golden set with invalid pairs removed and audited-method omissions added; unaudited methods
  stand unchanged. S28 reads it.
- *Escalate.* A flagged F1 or F2: a second fresh S56 labeller relabels the language's methods blind,
  DS adjudicates the differences, and the adjudicated union is the golden set, before S28. Flags
  in three or more languages: return to the coordinator, as the S56 labelling process, not a language,
  is suspect.

## 6. Labelling protocol (S26 blind labels)

### 6.1 Given to the labeller (one fresh `qa-engineer` per language)

- The pinned corpus checkout for **its language only**, read-only, without `.git` history beyond
  the pinned commit. Dependency and build-output directories are absent (no `node_modules/`, venv,
  restored `vendor/`). A repository's own committed vendored code stays, and §6.5 says how to
  label calls into it.
- `sample/<lang>.jsonl` (§4.4).
- **`sample/<lang>.brief.md`, the filled §6.5 brief, pasted into the dispatch as it is.** It is
  the only text of this note the labeller gets. S25 writes it from the §6.5 template, filling the
  per-corpus slots from `corpora.yaml` and appending the `qname` grammar excerpt §6.5 names.
- Tools: file read, `grep`/`rg`, `find`, and append to its own label file.

### 6.2 The labeller must not see or use

- Joern in any form: `joern`, `joern-parse`, any `*2cpg`, any `cpg.bin`, any CPG graph in FalkorDB
  (`mcp__cypher__query`).
- Any symbol-resolving program: SCIP indexers, language servers (Pyright, `tsc`, `gopls`, clangd,
  Roslyn, OmniSharp), IDE back-ends, `go doc`/`go vet`, compilers and build tools (a compiler's
  error output reveals resolution). Reading build files as text is fine.
- `cpg/spikes/frontend-fitness/` except its own `sample/<lang>.jsonl`, its brief and
  `labels/<lang>.jsonl`. In particular `results/`, `truth/`, `adjudication/`,
  `sample/*.private.jsonl`, `corpora.yaml`, the harness scripts, and other languages' files.
- This note, the plan, the fitness report, and any pass threshold. The labeller must not know which
  outcome passes.
- The network (web search, web fetch), the falkor-chat KB, and other agents' outputs.

**Check.** The S26 dispatches are `qa-engineer` subagent runs, not the headless sessions of plan
§6. So the plan §6.5 audit tooling does not apply as is. `teco` greps **that dispatch's own
subagent transcript** (the JSONL the Claude Code harness records for the agent run) for the tool
names and path fragments above, and for any `WebFetch`/`WebSearch`/MCP tool use. A hit voids that
language's labels.

### 6.3 Where the label rules live

The label schema, the truth-set definition and every per-language convention are in **§6.5
only**, so that the brief and the protocol cannot drift apart. The harness validates each label
line against §6.5's schema on read and rejects the file on the first invalid line.

### 6.4 Golden flow sets for `flowAvailable` (S28), and who writes them (F6, Q1)

- **Population:** per language, the corpus methods that are 5–25 lines long, take ≥ 1 parameter,
  contain ≥ 1 branch node (the frame's node list) and ≥ 1 call, and are not generated. Draw **30**
  by the §4.4 hash sort with seed prefix `ccg-p0a-v1-flow|`; a population smaller than 30 is drawn
  whole (TS/JS: 12 methods). **TS/JS excludes the files bundle spec4 §1 lists as `test`**
  (`*.test.<ext>`, `*.spec.<ext>`, `e2e/` or `__tests__/` segments): the bundle has no symbol for them by
  design, so every pair there would be a miss by construction. That removes 2 of the first draw's 14
  methods; the 3 methods of 26–60 lines are all outside tests. The draw is regenerated before any label
  exists. Python tests stay in. The population is not widened to methods whose only guard is a
  short-circuit operator: on TS/JS, requiring the guarded right operand to contain a call adds 0 of
  161 bodied definitions, and the looser test adds 13, 10 of them one cloned 8-line test helper.
- **Conditions in the labels.** The labeller counts a short-circuit operand that guards a point
  (`&&`, `||`, `??`, `and`, `or`) as a condition, as bundle spec §5.5 defines `cond`; the pairs it
  guards are control pairs. This widens the pairs per method, not the methods.
- **Label per method:** every **direct** intraprocedural *pair*, over the four point classes of
  bundle spec §5.4–§5.6 (`param`, `call`, `ret`, `cond`), nothing else being a source or a sink:
  - **data pairs:** the source is a `param` or a call's **result**; the sink is a call (argument or
    receiver), a `ret`, or a `cond` that reads the value. *Direct* means through local variables,
    aliases, loop variables, destructuring, fields and elements of locals, and language operators,
    and **not** through a call: `f(g(x))` is `x`→`g` and `g`→`f`. The sink is the innermost
    condition containing the read;
  - **control pairs:** the source is a `cond`; the sink is a call, a `ret` or another `cond` that it
    directly decides (nearest governing condition, early exits and short-circuit operands included).
  - **Not asked for:** field-write sinks and reads of an implicit `this` (no bundle point exists for
    them: spec §5.4 makes implicit `this` not a `param` for JVM, C# and JS, and the `data` edge
    endpoints are `Param`/`CallSite` as definition and `CallSite`/`Ret`/`Cond` as use). Flow through
    fields is therefore unmeasured: `flowAvailable: true` means flow among those four point classes,
    and the report says so. For C++, spec §5.4's receiver list is closed and names no C++ construct,
    and its implicit-`this` sentence names JVM, C# and JS only, so the brief's treatment of C++'s
    implicit `this` as the JVM's is an extension of the spec, not its text.

  *Stopping conditions (priors, confirmed or overturned at S28).* (a) Direct pairs: if more than 10%
  of the connected data pairs are connected only by a chain of two or more edges, the chain rule is
  carrying the recall and the single-edge figure is the one reported first. (b) Endpoints: if more
  than 10% of a language's golden pairs are `endpoint-unresolved`, S28's adjudicator splits them into
  labeller naming errors and projection gaps before any §8.4 decision; for C++, if the unresolved
  `param` endpoints show a one-position offset (the projection counting `this`), bundle spec §5.4
  gets the C++ sentence and the pairs are re-keyed, not scored as misses. (c) Fields: if P1's AC-5
  strata show a missed impact link that runs through a field or implicit `this` in a language with
  `flowAvailable: true`, that language's flow claim is narrowed to the four point classes in its
  report.

  Each pair is given as (method, source point, sink point), with points named the way S03's
  bundle names `Param{index}`, `Ret`, call-site position and `Cond` code. The brief is
  `cpg/spikes/frontend-fitness/flow-brief.template.md`.
- The labeller must not see S11's projection output, or `extract.sc`. The same §6.2 list applies.
- **Producer:** a new step of seven fresh per-language `qa-engineer` dispatches (§11 N3). It needs
  S25's flow draw and S03's point names, not S26, so it can run alongside S26 as separate
  sessions. Its output is `labels/<lang>-flow.jsonl`. S28 depends on it. The flow truth's second
  check is S28's miss and spurious-pair adjudication (§8.4).

### 6.5 Labeller brief (template; S25 fills the `{…}` slots and pastes nothing else)

Everything between the two rules below is the brief. S25 writes it to `sample/<lang>.brief.md`
with the slots filled from `corpora.yaml`, and appends **verbatim** the bundle spec's `qname`
grammar (`docs/plans/connected-code-graph-bundle-spec.md` §4.1, plus the §4.2 subsection for the
language), stamped with the spec's version at fill time. That excerpt includes rule 8's table.
It is copied at fill time, not into this note, because the spec is still in review.

**Harness mapping of `declared` (unchanged in kind, re-checked for spec v1.2).** The labeller never
applies rule 8. For each `declared` entry, the harness reads the declaration form at `defLine` and
applies spec v1.2's table: a symbol → `T(s)` with role `decl`; a form the spec substitutes →
the substituted definition, role `impl`; otherwise → `N(s)`. A labeller who wrote `internal` for a
bodyless declaration, or `declared` for a definition with a body, is corrected by the same check,
and the correction is logged as a label error for §5.1's noise figure.

**Departure from the bundle spec, stated:** for the overloading languages (Java, Kotlin, C#,
C++) the labeller does **not** write the §4.3 `signature`. `path` + `qname` + `defLine` already
identify one definition, and S27's harness derives the signature from the definition at
`defLine`. Hand-writing signatures would add an error source that no a1 metric needs.

---

**You are labelling call sites in a {LANG} codebase, blind.** For each call site in
`{SAMPLE_FILE}` you write down which functions or methods it can really call, by reading the
source. Your answers are compared with tools later; you will not see their output, and you must
not use any tool that resolves code for you.

**What you have.** The codebase at `{CORPUS_ROOT}` (read-only), the sample file, and your output
file `{LABEL_FILE}`. Use only file reading, `grep`/`rg` and `find`.

**Which call a sample line means.** Each line gives `path`, `line`, `column`, `callText` and a short
excerpt. The site is the call whose **callee expression** (the part before the argument list, such
as `self.repo.get` in `self.repo.get(x)`) starts at `line:column` and reads exactly `callText`.
In `a.b().c()` two sites start at the same place: `callText` `a.b` is the inner call, and `a.b().c`
the outer one.

**What you must not do.** Do not run any program that understands code: no Joern or `*2cpg`, no
indexer, no language server (Pyright, `tsc`, `gopls`, clangd, Roslyn, OmniSharp), no compiler or
build tool, no `go doc`/`go vet`. Do not use the web, MCP tools or any knowledge base. Do not open
anything outside `{CORPUS_ROOT}` except the sample file and your output file. Using any of these
voids your work.

**The corpus boundary.**
- **In the corpus** (a call into these is *internal*): every file under `{CORPUS_ROOT}`, tests
  included, generated files included ({GENERATED_DIRS}).
- **Vendored code** is someone else's library kept in the repository: any file with a directory
  named one of {VENDORED_DIRS} anywhere in its path. A call into it is *external* (the package
  form below).
- Everything else, the language's standard library and installed packages, is *external*.

**What to write for each site: the true callees.**
- A **direct call** to a function or method with a body: that one definition.
- A **method call on an object**: work out the object's declared type by reading (its parameter
  or field type, or the assignment that created it). List the method that type has or inherits
  from its nearest in-corpus ancestor, **plus every in-corpus override or implementation** in a
  subtype of that type (grep for subclasses and implementors). If the type's method has no body
  (an interface, abstract or protocol method, a pure virtual, a TS `type`-alias method signature),
  list it as `declared` and list the
  implementations as `internal`. If the method is **inherited from a base class outside the
  corpus** and the type does not override it, list it as `external` (that base's package), plus any
  in-corpus overrides in subtypes as `internal`. If you cannot work out the type, mark the site
  `unsure` and list what you found.
- **Special declarations.**
  - A primary constructor (Kotlin, C#), or a Java record header when the record has no explicit
    canonical constructor: `declared` `Type.<init>`, `defLine` = the header line.
  - A C++ `= default` member: `declared`.
  - A Kotlin `expect` declaration: label its `actual` as `internal` instead.
  - A C# `partial` method: label the implementing declaration.
  - A C/C++ prototype, or an in-class declaration of a member defined out of line: label the
    definition.
  - A Go call through an interface value: `declared` at the interface that declares the method
    (`IfaceType.NAME`; for a method that comes from an embedded interface, that interface's own
    spec), plus every in-corpus implementation as `internal`. For an unnamed interface literal
    (`interface{ Close() error }` in a signature), still list the spec as `declared`.
- A **call through a variable holding a function** (a callback, a parameter, a stored handler):
  trace where the value comes from, up to two levels of in-corpus callers, and list every
  in-corpus definition that can arrive there. If it goes further, mark `unsure` with the partial
  list.
- **Constructors, macros and JSX:** see the table. Decorator and annotation applications (`@x`,
  `@x(…)`, `[Attr]`) are never meant to be in your sample. If a line points at one, mark it
  `notACall`.
- If **no target at all** can be determined from source (a name computed at run time, `eval`,
  reflection on a computed string), set `dynamic: true`. If some targets can be determined, list
  them too.
- If the site is **not a call** (a type conversion, a macro that expands to no call, a
  lower-case JSX tag), set `notACall: true`.

**Writing a target.**
- `internal`: `path` (relative to `{CORPUS_ROOT}`, `/` separators), `qname` (the grammar in the
  appendix: `path::Scope.name`), and `defLine`. `defLine` is the 1-based line of the
  definition's **name** (the line with `def name`, `function name`, the method name, the class name
  for a type); decorators and annotations above it do not count. For an anonymous function it is
  the line where the function starts. For a Kotlin primary constructor it is the class header
  line. A function whose name exists only after macro expansion gets `"qname": null`, with the
  path and line where the macro is used to define it.
- `declared`: the same three fields, for a bodiless static target.
- `external`: `package`, in the form the table gives.

| Language | `qname` form (appendix) | External `package` | Constructor target | Other kinds | `notACall` |
|---|---|---|---|---|---|
| Python | `path::Class.method`, `path::func`, nested `path::outer.inner` | the top-level module as imported (`fastapi`); standard library `stdlib:<module>` (`stdlib:json`); builtins and methods of built-in types (`len`, `dict.get`, `str.join`) `builtin` | `Foo(…)` → `Foo.__init__` in `Foo` or its nearest in-corpus base; if none, the class `Foo` itself | — | none expected |
| TS/JS | `path::func`, `path::Class.method`, `path::Class.<init>`, binding-named arrows `path::outer.name` | the npm specifier, scope kept (`react`, `@tanstack/react-query`); Node core `stdlib:<module>`; JS and DOM globals and built-in type methods (`fetch`, `JSON.parse`, `Array.prototype.map`) `builtin` | `new Foo(…)` → `Foo.<init>` in `Foo` or its nearest in-corpus base; if none, the class `Foo` | `<Foo …/>`: a function component's function; a class component's `Foo.render`; `<A.B/>`: member `B`; `memo(X)`/HOC-wrapped: trace to `X` | lower-case (intrinsic) JSX tags |
| Java | `path::Class.method`, `path::Outer.Inner.method`, `path::Class.<init>` | the package of the imported type (`org.springframework.web.bind.annotation`); `java.*`/`javax.*`/`jdk.*` `stdlib:<package>` | `new Foo(…)` → the `Foo.<init>` overload that matches (by `defLine`); with no explicit constructor, the class `Foo`. `super(…)`/`this(…)`: the matching `<init>` | annotations are not call sites | none expected |
| Kotlin | `path::func`, `path::Class.method`, `path::Class.Companion.f`, `path::Class.<init>`, `path::Class.prop.<get>` | as Java; `kotlin.*` `stdlib:<package>`; scope functions (`let`, `apply`, `also`) `stdlib:kotlin` | `Foo(…)` → the matching `Foo.<init>` (primary: class header line); none explicit: the class | trailing-lambda DSL calls: the function called, not the lambda | none expected |
| C# | `path::Ns.Sub.Class.Method`, generic types ``Box`1``, `path::Ns.Class.<init>` | the type's namespace (`Microsoft.AspNetCore.Mvc`); `System.*` `stdlib:<namespace>` | `new Foo(…)` → the matching `Foo.<init>`; none explicit: the class | attributes are not call sites | none expected |
| Go | `path::Func`, `path::Type.Method` (receiver base type, no `*`) | the import path (`github.com/spf13/cobra`); standard library `stdlib:<path>` (`stdlib:net/http`); `len`, `append`, `make`… `builtin`. Vendored `vendor/<module>/…` → that module path | Go has none: `NewFoo()` is a plain function; composite literals are not call sites | call through an interface value: every in-corpus type's method that implements it, by method name and signature | type conversions `T(x)` |
| C/C++ | `path::Ns.Class.method`, out-of-line `path::Ns.Class.method`, C `path::func`, `path::Ns.Class.<init>` | the header the declaration comes from: standard `stdlib:<header>` (`stdlib:vector`, `stdlib:stdio.h`); other out-of-corpus headers `header:<include text>` | `new Foo(…)` or `Foo(…)` → the matching `Foo.<init>`; none explicit: the class | function-like macro `M(…)`: the in-corpus functions its expansion calls directly, expanding nested macros, across every `#if` branch; only external calls → those externals | a macro whose expansion calls nothing |

**Output.** One JSON object per line in `{LABEL_FILE}`, in **ordinal order, never skipping a
site**, appended as you go:

```json
{"ordinal": 17, "siteId": "<as in the sample>",
 "callees": [{"kind": "internal", "path": "src/a/b.py", "qname": "src/a/b.py::Repo.get", "defLine": 42},
             {"kind": "external", "package": "falkordb"}],
 "notACall": false, "dynamic": false, "confidence": "sure",
 "evidence": "receiver self.repo: Repo (src/a/svc.py:12); Repo.get b.py:42; no subclass of Repo (grep)"}
```

- `kind` is `internal`, `declared` or `external`. `confidence` is `sure` or `unsure`. **`unsure` is
  always better than a guess or a skip.**
- `evidence`: at most 300 characters, citing file:line for what you relied on.
- **Stop** after the line that brings the number of lines with at least one `internal` or
  `declared` callee to
  **150**, or after ordinal **{CAP}**, whichever comes first.
- **Resume:** if you are restarted, read the last ordinal in `{LABEL_FILE}` and continue with the
  next one. Never rewrite earlier lines.

**Appendix: `qname` grammar** (bundle spec §4.1 and §4.2 {LANG}, version {SPEC_VERSION}, inserted
verbatim by S25).

---


## 7. Metrics and intervals

Every proportion is reported as `k/n`, the point estimate, and its **Wilson 95% score interval**
(the lab's convention, z = 1.959964), per language × configuration. Quantile ranks use exact
integer arithmetic (nearest-rank: rank = ⌈q·n⌉ computed as `-(-q_num*n // q_den)`).

### 7.1 Site matching (before any metric)

A frame site matches a Joern `CALL` in configuration `c` when the file is the same, Joern's line is
the frame's `nameLine` (the callee's last identifier, §2), and Joern's call `name` equals
`calleeName` (for a constructor, the type name
or `<init>`/`.ctor` per frontend; the S25 author writes that table per frontend). If several
match, the one nearest by column wins, then the earlier in Joern's order.
- **Unmatched sampled sites are inspected one by one** by `graph-dba` in S27 and classified as a
  *matcher error* (fix the matcher and rerun; no metric is computed until zero remain) or *absent
  in Joern* (counts as not found).
- The census match rate (frame sites matched ÷ frame sites) is reported per configuration.

**Vendored code (review P2-4).** Joern parses the corpus as the production build would, so
committed vendored directories stay in its input. Their types help resolution, which matters most
for Go's vendored-modules-only offline policy (proposal §3.2). But every Joern `METHOD` whose file
is vendored by bundle spec v1.2 §5.0's test (a directory segment of its path is in the corpus's
effective `vendoredDirs`) is treated as **external**: it is excluded from `Nm(s)`, and a
resolution into it counts as *resolved away* (§2). This matches the labels, which call vendored
targets external. The count of such resolutions is reported. Rejected: dropping vendored
directories from Joern's input. That would lower off-configuration resolution for a reason the
production system does not share. This is the spec's own rule (§5.0: vendored files are parsed
but yield no record, and calls into them are `external`), so the P0 figures describe what the
bundle will hold. **`{VENDORED_DIRS}` in the brief is that same effective list**: the spec's
default (`vendor`, `vendored`, `_vendor`, `third_party`, `third-party`, `node_modules`) unless
`corpora.yaml` records a replacement `vendoredDirs` for the corpus (as a system manifest would). S25
writes the effective list into `corpora.yaml`, and §4.1's frame exclusion uses the same segment test.

### 7.2 a1 metrics (S27), per language × {off, on}

| Metric | Definition | Population | Interval |
|---|---|---|---|
| **Fallback recall** `R_c` | found ÷ internal (§2) | the ≤ 150 internal labelled sites | Wilson |
| Fallback recall, lenient (secondary) | the pre-v1.14 §2 reading: name match credited when no Joern `CALL` matched | same | Wilson; reported beside `R_c`, with the §8.1 remedy it would give |
| Edge-level recall | found true edges ÷ true edges | same | none (secondary, clustered) |
| **Resolved-call precision** `P_c` | correctly resolved ÷ resolved (§2) | labelled sites resolved in `c` | Wilson |
| "Resolved away" | sites resolved only to external names while the truth is internal | labelled internal sites | Wilson |
| **Name-match fan-out** | `|Nm(s)|` per unresolved site: median, p90, p99, max, share > 20 | **census**: every matched unresolved frame site | none (census of this corpus) |
| Fan-out, arity-filtered | same, keeping candidates with compatible arity | census | sensitivity only |
| **Parse success** | files with a Joern `FILE` node and no frontend error for that file ÷ frame files | census | Wilson |
| Method coverage | tree-sitter definitions with a matching Joern `METHOD` (file, name, start line ± 0) ÷ definitions | census | Wilson |
| **Line fidelity** | matched methods whose `lineNumber` and `lineNumberEnd` both equal tree-sitter's ÷ matched methods. Start uses §5.3's convention. A decorator-inclusive variant is reported too, because frontends differ on it | census | Wilson |
| **Value-set recovery** | §5.3 constant arguments whose Joern argument is a `Literal` with the same value, or whose single reaching definition is one ÷ such arguments | census | Wilson |
| Decorator/annotation shape | per kind: share of §5.3 decorated definitions where Joern exposes the decorator name on the method, and share with its literal arguments; plus one worked example per language for S30 | census | Wilson |
| Recall and precision **by kind** | §7.2's first three rows per §4.1 kind | domains of the sample | Wilson, labelled *descriptive* (small n) |
| Label noise | hand vs reference agreement; adjudicated error rate per source (§5.1) | labelled sites | Wilson |

Every recall miss is listed with its **cause class**: *different name* (alias, re-export,
wrapper), *callback/function value*, *dispatch* (implementation not name-matched), *reflection/DI
by string*, *framework-invoked* (decorator/route/annotation dispatch), *macro*, *absent in Joern*
(parse loss), or *other*.

### 7.3 a2 metrics (S28), per language × {off, on}

| Metric | Definition | Interval |
|---|---|---|
| Projection success | methods the S11 projection emits without error or timeout ÷ corpus methods (census) | Wilson |
| **Flow pair recall** | golden pairs (§6.4) the projection connects ÷ golden pairs. A **data pair** is connected when a directed chain of one or more `DATA` edges within the method leads from the source point to the sink point; access-path labels are ignored, and an edge marked `field-truncated` (the bundle's depth-3 cap) still counts as connecting. The golden pairs are direct (§6.4), so the chain allowance only credits a pair the projection routes through intermediate points; the share connected by a single edge is reported beside the pooled figure, so that leniency is visible. A pair whose endpoint matches no projection point, or several, is a miss, listed apart as `endpoint-unresolved`. A **control pair** is connected by a direct `CONTROLLED_BY` edge from the sink point to the condition's `Cond`. Data and control are reported separately and pooled | ratio estimator over the method clusters, formula below |
| Methods fully recovered | methods with every golden pair connected ÷ 30 | Wilson |
| Spurious pairs | projection pairs not in the golden set ÷ projection pairs, on the 30 methods | none (secondary) |
| Density and build time | nodes, edges and MB per file; projection build time per kLOC; symbol and call-site density per file (graph note Q3) | none (descriptive) |

**The flow-pair interval.** Pairs cluster within methods, so the interval is not Wilson. For the
n = 30 methods, with `m_i` golden pairs and `y_i` connected pairs in method i:

```
R̂     = Σ y_i / Σ m_i
m̄     = Σ m_i / n
V̂(R̂) = Σ (y_i − R̂·m_i)²  /  ( n·(n−1)·m̄² )          (linearized ratio variance, no finite-population correction)
CI    = R̂ ± t(0.975, n−1) · √V̂(R̂)                    (t(0.975, 29) = 2.045), clipped to [0, 1]
```

A method with `m_i = 0` (no golden pair, which the §6.4 population rule makes unlikely) is dropped
and n reduced. I rejected the percentile bootstrap for the reason in the lab's technique notes
(row-order instability on a discrete outcome at a fixed seed). The formula above is closed-form.

### 7.4 Off vs on (the K11 gap)

Both configurations are measured on the same sites, so the gap is **paired**: report the
discordant counts (found-on/missed-off and the reverse), **McNemar's exact test**, and a Newcombe
paired-difference interval over the Wilson bounds. A gap is called real only at exact two-sided
p < 0.05. **The exact test decides; there is no net-count shortcut.** The minimum is 6 discordant
sites, all in one direction (6–0, p = 0.031). With 1 the other way it takes 8 (8–1), with 2 it
takes 10, with 3 it takes 12. Non-overlapping marginal intervals are not used as a test.

## 8. Decision rules

Rules are applied per language × configuration first. A language's decision is then the **worse
of its two configurations**, because the system has to serve local-only and hosted codebases
alike. The report records which configuration drove it. Each rule is mechanical given §7's
numbers; S27/S28 apply them and `data-scientist` checks.

### 8.1 Fallback recall → indexer or recognizer (correctness)

**Pass** iff internal sites n ≥ 100 **and** observed `R_c` ≥ 0.98. At n = 150 that means ≤ 3
misses; at 100–149, ≤ ⌊0.02·n⌋ misses. n < 100 is *insufficient* and counts as a fail.

**On a fail, the remedy follows the misses' causes (F5).** The reference ran on every site, so the
counterfactual is observable:
- **Indexer adopted** if adding the reference's answers to Joern's (`Res ∪ Nm ∪ Ref`) brings
  observed recall to ≥ 0.98. Caveats, both recorded: this counterfactual is biased upward, because
  the reference is half of the truth; and the reference ran **with** dependencies, so for an
  off-configuration fail the indexer's offline efficacy is unmeasured and P2's extractor plan must
  measure it.
- **Otherwise, recognizer work**: the language gets a backlog item per cause class holding ≥ 1/3
  of its misses (callback/function value → flow projection or a recognizer; framework-invoked →
  rule pack; reflection/DI by string → rule pack; macro → `compile_commands.json`-aware parse;
  absent in Joern → frontend configuration, then re-measure). The indexer is also adopted if it
  closes at least half the misses.
- In both cases, every miss is listed in the report with its site and cause.

**Why 0.98, and what it means.** It is kept as ML §3.6's prior: a 3-hop chain at 0.98 per hop
keeps about 0.94. P0 neither confirms nor refutes it. S14's measured `Link` chain hop counts (M4)
are what should revisit it. At this sample size it is a **screen** (§4.3), not a certificate.
The report must say "observed ≥ 0.98, Wilson lower bound x", never "recall ≥ 0.98".

### 8.2 Name-match fan-out → indexer (cost)

**Indexer adopted on cost** iff census p90 `|Nm(s)|` over unresolved sites is **> 20** in either
configuration. This is ML §3.6's prior, kept, with one challenge recorded. The provisional noise
bound is 4:1 (ML §8.1, RQ10). A site whose fan-out is above 5 cannot meet 4:1 on its own when only
one candidate is real. So the report also gives the share of unresolved sites with fan-out > 5,
for S06's noise-bound work. Whether 20 is too loose is decided by AC-5's noise measurement at P1,
not here.

### 8.3 Resolved-call precision → folded vs listed, and indexer adoption

**Operative rules** (the plan's §4.3 table, proposal §3.2 and ML §8.1 m1, which fix the adoption
rule as a point estimate; review M2):
- **Folded** iff, in **both** configurations, resolved sites n_P ≥ 60 **and** observed `P_c` ≥ 0.95.
  Otherwise **listed**. n_P < 60 is *insufficient*, and the default is the conservative display.
  The n_P floor is this note's addition, within S05's remit.
- **Indexer adopted (correctness)** iff observed `P_c` < 0.95 in **either** configuration with
  n_P ≥ 60. With n_P < 60 in a configuration, precision is *unmeasured* there. The report says so,
  and that configuration triggers no adoption on precision. The Wilson interval is reported beside
  every decision.

**Why the AC-5 link matters.** A wrong resolution is a false *confirmed* edge, and confirmed
precision ≥ 0.95 is a **gated** AC-5 bar at P1 (ML §4.2 (i)). So low P0 precision is a direct risk
to a P1 gate, not only a display question.

**What the sample can tell apart.** Every figure here is binomial, reproducing the reviewer's
Appendix A4.
- Wilson at the threshold: 57/60 → [0.863, 0.983] (width 0.12); 95/100 → [0.888, 0.978]; 114/120 →
  [0.895, 0.977].
- **P(fold), which needs both configurations to pass.** The two configurations use the same sites,
  so the joint probability lies between the per-configuration figure and the product of two
  independent ones. Over n_P = 60–120:
  - at true precision 0.97: per configuration 0.89–0.93; both configurations **0.80–0.93**;
  - at true precision 0.93: per configuration 0.26–0.39; both configurations **0.07–0.39**.
- **P(adopt on precision)** in a configuration is 1 − P(pass) there. A language truly at 0.97 is
  adopted with about 0.07–0.11 probability per configuration, and at most about twice that over
  both (an unneeded indexer, a cost). A language truly at 0.93 is adopted with 0.61–0.74 probability
  per configuration.
- **Folding is display-only.** Folded candidates stay in the canonical answer and count for AC-5
  (proposal §3.7). If folding ever became a facts filter, this sample could not support it.

**Proposal for `architect` (not in force): adopt on precision only when the Wilson upper bound
is below 0.95.** The rule would be: observed `P_c` < 0.95 makes the language an indexer
*candidate* (proposal §3.7's word), and adoption needs the upper bound below 0.95. That is about
≥ 7 errors at n_P = 60, ≥ 10 at 100 and ≥ 11 at 120 (point 0.88–0.91).
- *For it:* since P3-1 a wrong resolution never drops a same-name true callee, so low precision is
  not a silent miss. It avoids adopting an indexer on a point estimate the interval cannot
  separate from 0.95.
- *Against it, and I now weigh this more:* a language whose point estimate is 0.90–0.94 would
  enter P1 with no indexer, carrying up to 9 errors per 100 resolved sites on edges that feed AC-5's
  gated confirmed-precision bar.
- **My recommendation is to keep the operative point-estimate rule.** An unneeded indexer is a
  bounded cost, paid with probability ≤ ~0.2 at true 0.97. A failed AC-5 confirmed-precision gate
  at P1 is the costlier error. If `architect` wants the trust category anyway, it must ratify it
  together with that AC-5 consequence.

### 8.4 `flowAvailable`

**One primary criterion (review M4).** True iff, in **both** configurations:
- **primary:** pooled flow pair recall (§7.3) is **≥ 0.95 observed**, over **≥ 120 golden
  pairs**, counted after S56's labels and S28's adjudication. With fewer than 120 pairs it is
  *insufficient*, and the answer is false. The floor is not lowered for any language. Two-stage
  rule for a population drawn whole (TS/JS; prior, confirmed at S56): if the labelled pairs are
  below 120, the same labeller labels, in the same dispatch, the language's methods of 26–60 lines
  under the same conditions (TS/JS: 3 methods), once; still below 120 means *insufficient*, reported
  as `flow-unavailable:<lang>:insufficient-sample`, distinct from a measured miss. Whether 12 methods
  yield the 10 pairs each that 120 needs (15 methods with tier 2: 8) is unmeasured. A crude proxy,
  the sink points (calls, returns, conditions) in the methods, is 97 for the 12 and 138 for the 15, so
  120 needs about 1.2 and 0.9 pairs per sink; I have no measured pairs-per-sink. The table below is for 30 methods; for
  n = 120 independent misses give P(pass) 0.99, 0.93, 0.61, 0.26, 0.04 at true recall
  0.98, 0.97, 0.95, 0.93, 0.90, and fewer than 20 methods widens it further (the cluster interval
  is reported beside the decision);
- **precondition** (a different construct, kept): projection success is ≥ 0.95 of corpus methods
  (census). A projection that fails on many methods cannot be trusted on the ones it emits.

Otherwise **false**: impact through the language's units is call-level, and every item is
*possible*, `flow-unavailable:<lang>`.

**Reported, not gating:** the cluster interval of §7.3, the methods fully recovered (k/30,
Wilson), and the spurious-pair rate. The v1.0 rule's "lower bound ≥ 0.85" and "≥ 24 of 30 fully
recovered" are **dropped as criteria**. The reviewer showed (Appendix A6) that the second sat well
above 0.95 when misses are spread out and the first seldom bound when they cluster. The effective
bar therefore depended on an unknown miss structure.

**Operating curve** of the primary criterion: P(pass) for 30 methods with m pairs each (150–240
pairs). Exact, by convolving per-method distributions. Misses are either independent or clustered
by method, with a beta-binomial intra-method correlation ρ of 0.1 or 0.3. The computation is in
the appendix at the end of this note.

| True pair recall | 0.99 | 0.98 | 0.97 | 0.96 | 0.95 | 0.93 | 0.90 |
|---|---|---|---|---|---|---|---|
| m = 5, independent | 1.00 | 0.99 | 0.92 | 0.75 | 0.52 | 0.17 | 0.01 |
| m = 6, independent | 1.00 | 1.00 | 0.95 | 0.81 | 0.59 | 0.18 | 0.01 |
| m = 8, independent | 1.00 | 1.00 | 0.97 | 0.83 | 0.58 | 0.14 | 0.00 |
| m = 6, ρ = 0.1 | 1.00 | 0.98 | 0.92 | 0.78 | 0.59 | 0.24 | 0.03 |
| m = 6, ρ = 0.3 | 0.99 | 0.96 | 0.87 | 0.74 | 0.59 | 0.31 | 0.08 |
| m = 5 / 8, ρ = 0.3 | 0.99 | 0.94 / 0.96 | 0.84 / 0.87 | 0.70 / 0.73 | 0.54 / 0.58 | 0.28 / 0.29 | 0.07 |

- **Reading it:** a language at ≥ 0.98 pair recall gets `true` ≥ 94% of the time, and one at ≤ 0.90
  gets `false` ≥ 92% of the time. At exactly 0.95 the decision is a coin flip, as any threshold
  rule is at its own threshold. Clustering widens the grey zone (0.93 passes up to 31%), which is
  why the cluster interval is reported beside the decision.
- Because both configurations must pass, the joint pass probability is at most the figure shown.
  Flow is intraprocedural, so the two configurations' projections are expected to agree closely.

**Why 0.95 and not the plan's suggested 0.9.** The false branch is the safe one only if call-level
reach is a recall-superset of flow reach within a unit, so `true` should be granted only where flow
loses little. At 0.9 per pair, a 3-pair path keeps 0.729; at 0.95 it keeps 0.857. **If P1's design
makes call-level reach not a superset**, this rule must be revisited, because `false` would then no
longer be the safe default.

**The second check on flow truth (review M5).** The golden pairs come from one labeller, so S28
adjudicates, with cited source evidence as in §5.1:
- **every golden pair the projection misses** (a miss is overturned only by evidence that the pair
  is not real), so false misses cannot fail a language;
- **a seeded sample of up to 50 spurious pairs** (projection pairs not in the golden set, drawn by
  the §4.4 hash sort with prefix `ccg-p0a-v1-spur|`). Each is judged either a real pair the labeller
  omitted or a true spurious edge. The omission rate is reported. It bounds how much an
  incomplete golden set flatters recall, since an omitted pair the projection would also miss is
  invisible. **Flag rule (review P2-3):** with n ≥ 20 sampled spurious pairs, the language's flow
  recall is flagged *truth-incomplete* iff the observed omission rate is **> 0.05**, reported as k/n
  with its Wilson interval. With n < 20 the flag is *insufficient* and not raised. (v1.1's
  "Wilson upper bound > 0.10" fired on every sample with n < 35 even at zero omissions, and on a
  single omission at n = 50.) At n = 50 the flag needs ≥ 3 omissions. It is raised with
  probability 0.08 at a true omission rate of 0.02, 0.46 at 0.05 and 0.89 at 0.10. At n = 20, ≥ 2
  omissions: 0.06, 0.26 and 0.61. The flag is a report qualifier, not a decision input, so this
  sensitivity is enough.

For Python and TS, S28's figures are cross-checked against S14's. A disagreement of more than 0.05
in projection success is investigated before the rule is applied.

### 8.5 One table per language in the report

| Language | R off / on (k/n, Wilson) | Miss causes | Fan-out p90 off / on | P off / on (k/n, Wilson) | Indexer: adopted (recall / cost / precision), or no | Fold or list | `flowAvailable` | Driving configuration |
|---|---|---|---|---|---|---|---|---|

## 9. Image requirements (S02) and gaps against plan §7

S24's done-condition is "every truth tool S05 names runs in the image". The list:

| Item | In plan §7? | Why |
|---|---|---|
| **Node.js LTS (Linux) + npm** | **No.** §7 notes that there is no Linux `node` and that `jssrc2cpg` does not need one | `scip-typescript` and `scip-python` are npm packages; `npm ci` for `salesperson/` types |
| `scip-python`, `scip-typescript` (npm, pinned) | no | §5.2 references |
| `scip-java` (pinned, with the SemanticDB javac and Kotlin compiler plugins) | no | Java and Kotlin reference |
| `scip-dotnet` (dotnet tool, pinned) | no | C# reference |
| `golang.org/x/tools/cmd/callgraph` (pinned module version); `scip-go` as fallback | no | Go reference with VTA dispatch |
| `scip-clang` + a matching clang/LLVM and libstdc++ headers | **No.** §7 has only `gcc`/`g++` on the host | C/C++ reference |
| **`scip` CLI** (pinned) | no | reading every SCIP index as JSON |
| Python 3.12 + **py-tree-sitter and the ten grammars** of §5.2 | Python is listed; tree-sitter is not | frame and census truths. Also declared in `cpg/pyproject.toml`'s `spike` extra, which reaches S09 only if `teco` puts it in S09's brief (§11 N6) |
| Each corpus's **dependency install with network at truth time** (pip/uv venv for `falkor-chat/server`, `npm ci`, Maven/Gradle resolve, `dotnet restore`, `go mod download`) | partly: §7 lists the build tools, not that the truth run needs the dependencies fetched | type-aware references. These run in the image **with network, outside the sandbox**, and never in the fetch-off run's workspace: the off run gets a clean checkout, so fetched jars or modules cannot leak into it |
| Maven and/or Gradle, Go, .NET SDK, Kotlin compiler, CMake (+ `bear` optional), JDK 21 | yes | builds (C4) and references |
| `uv` or `pip` in the image | not stated | the Python corpus venv |

The image must keep the tools' versions in its content hash, like the rest of the toolchain, so
that the truth is reproducible. The **off-configuration run must not see** the dependency caches
the truth run fills (`~/.m2`, `~/.gradle`, `~/.nuget`, `GOMODCACHE`, `node_modules`, the venv). S25
runs them in separate containers or workspaces, and S08's escape-axis probe should include a check
that those caches are not mounted into the sandbox.

## 10. Risks and open questions

**Open questions.** The v1.0 questions Q1 (who writes the golden flow sets) and Q2 (where
adjudication sits) are answered: §11 N3 and N4–N5, as the review recommended. What remains for
`architect` is whether to add those steps in plan v1.4, and the §8.3 precision proposal.

**Risks:**
- **R1. The 0.98 gate cannot be certified at P0** (§4.3). Mitigation: it is worded as a screen,
  and AC-5's end-to-end gate is the certificate.
- **R2. Implicit calls are outside the frame** (C#/Kotlin properties, operators, Python dunder
  protocols). Recall for them is unmeasured. Mitigation: listed in the report as a blind spot. If
  S30's packs or the slice find property-dispatched links that matter, a follow-up sample of
  implicit sites is the cheapest measurement.
- **R3. One corpus per language** is one codebase, so every figure is a property of that corpus.
  Codebase-to-codebase variance is unmeasured. The report must not generalize beyond "on this
  corpus".
- **R4. Reference tools' maintenance is unverified** (§5.2), and the Kotlin SemanticDB plugin is the
  most fragile: scip-kotlin is archived and merged into scip-java, and K2 support is not stated.
  Mitigation: S24's smoke test, C7's version range, and the installed-tool-only fallbacks of §5.2.
  The last resort for any language is hand labels alone, flagged single-source, with recall marked
  noise-unestimated.
- **R5. Labeller session length** (§4.3). The 500 cap makes a worst-case session up to 500 sites.
  Mitigation: append-and-resume. A language that still ends below 100 internal sites (an
  internal share near 0.20) is *insufficient*, which counts as a fail under §8.1.
- **R6. Contact with the proving repo** (the TS/JS census of §3.1 sharpens it). S26 labels call edges in `falkor-chat/server/` and
  `salesperson/`, in a file builders can read. These are call edges, not impact keys, so the
  contamination of AC-5 is low. Even so, the contact list must exist before S27 reads
  the strata of §3.1. *Producer (proposed; `architect` assigns the step id and row):* a new step
  after S11 and S14, before S27, owned by `graph-dba`, who built the extractor and knows what the
  slice touched; checked by `analyst`; not the S57 custodian, who must stay clear of Joern artifacts.
  Output: `cpg/spikes/frontend-fitness/contact.json`, with these keys.
  - `files`: the files of the two corpora that a **development source** names by path or by a
    defined symbol's qualified name, plus S06's own list. Development sources are the artifacts
    that shaped or encode what was built: code, tests of the system under test, fixtures, golden
    bundles and rule packs (eval-protocol §8.2's `cpg/tests/golden/*` and fixture rows), plus exactly
    two documents, each for a stated part: **the slice report's §4** ("Proving-repo anchors the slice
    touched", S14's contact list, a named exception to the measuring-documents rule below), whose
    files are those of its four strict locators: `salesperson/src/api/client.ts` (`buildUrl`) and the
    modules of the three `/health` providers (`storefront_api.py`, `api.py`, `app.py`); and the
    **rule-pack spec** (S30's walkthroughs). The slice report's other sections are not read: its
    scale-probe and link-consumer files (`executor.py`, `llm.py`, `modelconfig.py`, `test_app.py`,
    `test_ensure_demo_models_script.py`, `test_services.py`, `test_storefront_api.py`,
    `test_tools.py`) are eval-protocol Q1's open items, were walked and not tuned against, and
    enter only if Q1 widens the set (the floor of §3.1 holds either way). **Worked examples are not
    contact** (eval-protocol §8.2): the bundle specs, the graph note, the main plan and the other
    plans name `config.py`, `embedding.py`, `guards.py`, `services.py`, `querygen.py`,
    `conftest.py`, `test_fusion.py` and `test_querygen.py` to illustrate grammars, so a file named
    only there is not contact; `config.py` (0 reference-internal sites) goes with them. **Not
    sources:** `cpg/spikes/frontend-fitness/` (the measurement), every `-ml.md` note
    and `docs/reviews/`, `docs/test-reports/` (documents that measure or review rather than build),
    the coordination ledger, `HISTORY.md` and `BACKLOG.md` (logs, derivative of the sources), and
    `cpg/tests/test_frontend_fitness.py` (synthetic snippets). Deriving with and without the excluded
    ones gave the same list on 2026-11-04 (S59 G2); excluding them keeps the check from drifting when
    those documents are edited.
  - `derivedFrom`: the git SHA of the tree the derivation read, and `--check` reads that tree, so a
    later commit cannot change the list.
  - `flowAnchors`: the distinct paths of `sample/<lang>-flow.jsonl` (the draw defines the methods;
    labels may be unfinished), per corpus. Recorded for S47 (eval-protocol §8.2), **not** part of
    `files` and not applied to §3.1's strata.
  The same file is the input to S47's contact set.
- **R7. Inverse-sampling cap interacts with the C/C++ macro kind.** If macro-invocation sites are
  mostly `notACall`, they consume ordinals without adding internal sites. `notACall` sites count
  toward the 500 cap, and the report gives their share. On googletest 17 of the 500 sample sites are
  macro-callee sites with a silent reference (§3.2).
- **R8. The fetch-on and truth runs share one network-enabled image.** A dependency cache that
  leaks into an off run would inflate off-configuration resolution. Mitigation: §9's separation,
  checked in S25.

## 11. What the plan's new and amended steps must produce (for `architect`, plan v1.4)

The S05 note cannot add steps (review M5, m5). These are the outputs each missing or amended step
must produce, so that the step rows can be written from this section. Paths are under
`cpg/spikes/frontend-fitness/` unless stated. Owners are suggestions; `architect` decides.

| Id | Step (new or amended) | Must produce | Inputs; must not see | Done when | Depends on |
|---|---|---|---|---|---|
| **N0** | **S24 amended** (`devops`) | Each §5.2 primary tool, pinned, in the image. A per-language **smoke project** (a few files with one interface or override) and its index, read back with `scip print --json` (`callgraph` output for Go) | §5.2, §9 | every primary tool indexes its smoke project in the image; a tool that cannot is replaced only by an installed-tool fallback, or the language is recorded *single-source* | S02, S05 |
| **N1** | **S25 amended: the reference truth run** (`graph-dba`) | Per candidate, **before its first Joern run**: `truth/<lang>.c5.json` (tool, version, exit status, documents indexed, 50-site pilot coverage, raw and adjusted, excluded count by §3.2 class), then for the chosen corpus `truth/<lang>.ref.jsonl`, one line per sampled site: `ordinal`, `siteId`, `static` (`{path, qname, defLine}`, `declared`, `external`, or `silent`; `relaxed: true` where §5.2's macro-argument rule matched; `ambiguous` as the tag of a `silent` row), `impls` (list), `tool`, `toolVersion` | the image with network and the corpus's dependencies; the fetch-off workspace never sees the dependency caches (§9) | C5 recorded per candidate; `ref.jsonl` covers every sampled ordinal; the Joern run's timestamp is later than the C5 record's | N0, S08, S09; N2's frame (same step) |
| **N2** | **S25 amended: frame, draws, brief, matcher** (`graph-dba`) | `frame/<lang>.jsonl` (the census frame, §4.1); `sample/<lang>.jsonl` and `sample/<lang>.private.jsonl` (§4.4, 500 sites); **`sample/<lang>.brief.md`** (the §6.5 template, slots filled, the bundle spec §4.1 + §4.2-language excerpt appended verbatim with its version); **`sample/<lang>-flow.jsonl`** (the 30 flow methods, §6.4: `ordinal`, `path`, `qname`, `lineStart`, `lineEnd`, and the method's source text); `matcher-table.md` (Joern call `name` per constructor and call kind, per frontend, §7.1) | corpora, tree-sitter | the hash sorts reproduce from `corpora.yaml` alone; every brief validates as §6.5 with no unfilled `{…}` slot | S09. Inside S25 the order is: frame → C5 (N1) → draws and briefs → first Joern run |
| **N3** | **New: golden flow labelling** (seven fresh `qa-engineer` dispatches, one per language) | `labels/<lang>-flow.jsonl`: per method `ordinal`, and per pair `{kind: data or control, source, sink, evidence}`, with points named as in S03's bundle (`Param{index}`, `Ret`, call-site position, `Cond` code) | `sample/<lang>-flow.jsonl` and that corpus only; never S11's projection, `extract.sc`, `results/`, `truth/`; the same §6.2 list | 30 methods labelled; transcript grep clean (§6.2, by `teco`). **Checker: `data-scientist`**, as for S26: label schema valid, every method has ≥ 1 pair, evidence cited | N2, S03 (point names); **not** S26, so it runs alongside S26 |
| **N4** | **New: pre-Joern adjudication** (`data-scientist` decides, a fitness custodian per language runs; `analyst` checks) | The files and procedure of §5.4: `adjudication/<lang>.{input,audit,noise}.json*`, `adjudication/<lang>.jsonl`, `adjudication/<lang>-flow-audit.jsonl`, `truth/<lang>.final.jsonl`, `truth/<lang>.flow-final.jsonl` | `labels/<lang>.jsonl`, `truth/<lang>.ref.jsonl`, the corpus tree, and for Part F the flow sample and flow labels; must not see `results/` or any Joern output | §5.4's "done when"; `analyst` confirms no Joern artifact was read (transcripts) | S26 and S56 for that language, N1 |
| **N5** | **S27 amended** (`graph-dba` runs; `data-scientist` adjudicates misses) | In this order: (1) provisional metrics on N4's `truth/<lang>.final.jsonl`, to list the misses; (2) **`adjudication/<lang>-misses.jsonl`**: every Joern miss in either configuration with its decision, cited evidence and cause class (§7.2); (3) overturns applied, written as `truth/<lang>.final2.jsonl`; (4) **the reported metrics, computed only on `final2`**, with the provisional figures shown beside them for transparency; (5) the §8.5 table | N4's outputs, Joern results | `analyst` has checked the miss log independently, and that every §8 number traces to `final2`; decisions follow §8 mechanically | N4 |
| **N6** | **S09 brief amended** (`teco`) | `py-tree-sitter` and the ten grammars of §5.2 declared in `cpg/pyproject.toml`'s `spike` extra | §5.2 | `pip install -e .[spike]` imports every grammar | S03 |
| **N7** | **S28 amended** (`graph-dba` runs; `data-scientist` adjudicates) | §7.3 metrics on N3's labels. **`adjudication/<lang>-flow-misses.jsonl`** (every missed golden pair) and **`adjudication/<lang>-flow-spurious.jsonl`** (the ≤ 50 seeded spurious pairs, §8.4) | N3's labels, the projection output | overturned misses and confirmed omissions applied to the golden set before the reported pair recall is computed (provisional figure shown beside); `flowAvailable` decided by §8.4; omission rate reported; **checker: `analyst`** on both adjudication logs | N3, S27, S15 |

**Ordering this implies:** S05 → N0 (S24) → N1/N2 (S25) → {S26, N3} in parallel → N4 → N5 (S27) →
N7 (S28). The S24↔S25 circle is broken because S24 proves tools on smoke projects, and C5 on real
candidates moves into S25 (§3.2, §5.2).

## 12. Review dispositions (`docs/reviews/connected-code-graph-p0-impl.md` §S05)

### Pass 1

All findings accepted. On M2 I now agree with the review beyond what it asked.

| Finding | Disposition | Where |
|---|---|---|
| M1 brief not self-contained | **Fixed.** §6.5 is the pasted brief (truth-set definition, `defLine`, corpus boundary, per-kind targets, 7-row table, stop and resume). §6.1–§6.3 point to it. §2 is aligned (`declared` set `D(s)`, superseded in v1.2 by Pass 2 P2-1; partial `dynamic`; `notACall`). The bundle spec §4 grammar is appended at fill time, not copied here. One stated departure: no hand-written `signature` | §2, §6 |
| M2 precision rule exceeds authority | **Fixed.** The plan's point-estimate adoption rule is operative. The upper-bound rule is a labelled proposal with the AC-5 confirmed-precision consequence, and my recommendation is now to keep the point rule | §8.3 |
| M3 S24/S25 circle, C5 contradiction, unowned fallbacks, Kotlin, scip-clang memory | **Fixed.** S24 proves tools on smoke projects. C5 runs in S25 before any Joern run, and a failure moves to the next candidate. Fallbacks are installed tools only; the four would-be programs are listed. C7 limits Kotlin to the plugin's range. scip-clang runs with parallelism 2 | §3.2, §5.2, §11 N0–N1 |
| M4 `flowAvailable` criteria inconsistent | **Fixed.** One primary criterion (pooled pair recall ≥ 0.95, ≥ 120 pairs), with projection success as a precondition; the operating curve is given. The lower-bound and fully-recovered criteria are now reported only | §8.4, appendix |
| M5 four pieces of work with no step | **Specified for `architect`** (not mine to add): N1–N5, N7, plus flow-truth adjudication | §11 |
| m1 0.375 trigger at the coin flip | **Fixed.** The cap is 500 everywhere and the predictor is dropped. Figures and session cost are stated | §4.3 |
| m2 fold probabilities per configuration | **Fixed.** Ranges for both configurations and the true intervals | §0, §8.3 |
| m3 McNemar "net 6" | **Fixed.** The exact test decides; the minimum is 6–0 | §7.4 |
| m4 held-back exclusion | **Fixed.** Message-broker clients excluded, a named register, whole repository roots only, and the reverse rule | §3.3 |
| m5 tree-sitter to S09 | **Specified** as N6, routed to `teco` and `architect` | §9, §11 |
| n1 wording | **Fixed.** Ordinal order; the inverse-sampling-bias sentence corrected; the variance formula and the `DATA`-path definition written out; the transcript check names the subagent transcript | §4.3, §7.3, §6.2 |

### Pass 2 (review S05 Pass 2: approve with suggestions)

| Finding | Disposition | Where |
|---|---|---|
| **P2-1** `D(s)` counts edgeless resolutions as correct | **Fixed, against bundle spec v1.1 rather than the v1.0 reading.** `D(s)` is removed. Its premise ("a bodiless declaration is not a symbol") is false under v1.1 rule 8: every member declaration a call can resolve to (Protocol/abstract, interface, pure virtual, Kotlin primary constructor) is now a symbol. A resolution to it yields a real confirmed edge to a real target. Such declarations are now in `T(s)` with role `decl`, and **count as correct for precision**: the edge is a true fact (this site calls that declared member), and a change to the declaration impacts the caller. **The true-callee definition still separates the two roles** (`impl` and `decl`), because they answer different questions. Recall is decided by `impl` (where code that runs can change; `decl` is always name-matched), and a `decl`-only confirmed edge names no implementation. So the *resolved-to-declaration* share is reported beside `P_c`, as a usability fact, not an error. The reviewer's concern survives only where rule 8 still yields no symbol (Go interface method specs, prototypes, `.d.ts`, bodiless accessors): those go in `N(s)`, and a resolution into them is *resolved to non-symbol*, excluded from n_P exactly as the reviewer proposed. The brief keeps the labeller's `declared` kind as a role tag; the harness maps it to `decl` or `N(s)` by language and declaration kind, so the labeller needs no rule-8 knowledge | §2, §6.5, §11 N4 |
| **P2-2** brief gaps | **Fixed.** The brief now says which call a sample line means (callee-expression anchor, with `callText` disambiguating nested calls), and that a method inherited from an out-of-corpus base, not overridden, is `external` with in-corpus overrides `internal`. Found while fixing: the frame anchored `line` at the callee's first character, but bundle spec §5.6's `line` is the callee's last identifier. The frame now also records `nameLine`/`calleeName`, and §7.1's matcher uses them, so multi-line chained calls cannot fail to match | §2, §6.5, §7.1 |
| **P2-3** truth-incomplete flag over-fires | **Fixed.** Flag on observed omission rate > 0.05 with n ≥ 20, else *insufficient*; operating figures given | §8.4 |
| **P2-4** step gaps, vendored, decorators | **Fixed in the note** (the steps themselves are still `architect`'s, plan v1.4). N4's input script: written by `graph-dba` in S25, run by `data-scientist`, statically checked by `analyst`. N5 and N7 apply overturns before the reported metrics (provisional figures shown beside them). N3's checker is `data-scientist`; N7's is `analyst`. The C5 pilot's seed prefix is `ccg-p0a-v1-c5|`. **Vendored:** Joern keeps parsing vendored code (it helps resolution, as in production), but its `METHOD`s there are external, out of `Nm(s)`, and a resolution into them is *resolved away*. I rejected dropping vendored code from Joern's input, because that would depress off-configuration resolution for a reason production does not share. **Decorators:** removed from the frame and the brief, since bundle spec v1.1 §5.6 makes them `decorator` records, not call sites. They are measured in the decorator-shape census. Calls inside decorator arguments go too, listed as a blind spot | §4.1, §4.4, §7.1, §8.4, §11 |

**v1.3 (2026-10-29):** re-aligned with bundle spec v1.2. The symbol list now follows rule 8's exhaustive table. Go named-interface specs (`IfaceType.NAME`) moved from `N(s)` to `T(s)` as `decl`; unnamed-literal specs stay in `N(s)`. The spec's substitution of prototypes and partial declarations is applied to labels and Joern targets alike. `{VENDORED_DIRS}` equals §5.0's effective list, with a segment test. The brief's special-declaration rules were added and the `declared` mapping re-checked. No statistic changed.

## Appendix. Operating-curve script

The computation behind §8.4's table and §4.3's 500-cap figures. Python 3.12 standard library
only; run with `python3 -I`. It is kept here because this note's author may write only under
`docs/plans/` and `docs/reviews/`. If `teco` wants it as a file, `cpg/eval/scripts/oc.py` is the
suggested home, with this appendix as its source.

```python
"""P0(a) S05 operating curves: flowAvailable pair-recall rule (§8.4) and the 500-site cap (§4.3).
Exact: per-method miss distributions (binomial, or beta-binomial with intra-method correlation rho)
convolved over 30 methods. Python 3.12 stdlib only. Run: python3 -I oc.py"""
from math import comb, lgamma, exp

def betabin(m, a, b):
    def lbeta(x, y): return lgamma(x) + lgamma(y) - lgamma(x + y)
    return [exp(lgamma(m + 1) - lgamma(k + 1) - lgamma(m - k + 1)
                + lbeta(k + a, m - k + b) - lbeta(a, b)) for k in range(m + 1)]

def conv(p, q):
    r = [0.0] * (len(p) + len(q) - 1)
    for i, x in enumerate(p):
        for j, y in enumerate(q):
            r[i + j] += x * y
    return r

def p_pass(recall, methods=30, m=6, rho=0.0):
    """P(observed pooled pair recall >= 0.95); per-method misses ~ Bin(m, 1-recall) or beta-binomial."""
    miss = 1 - recall
    if rho == 0:
        per = [comb(m, k) * miss**k * recall**(m - k) for k in range(m + 1)]
    else:
        s = (1 - rho) / rho
        per = betabin(m, miss * s, recall * s)
    tot = [1.0]
    for _ in range(methods):
        tot = conv(tot, per)
    n = methods * m
    max_miss = n - (-(-95 * n // 100))          # exact: misses <= n - ceil(0.95 n)
    return sum(tot[:max_miss + 1])

def p_below(n, share, k):                      # P(Binomial(n, share) < k)
    return sum(comb(n, i) * share**i * (1 - share)**(n - i) for i in range(k))

if __name__ == "__main__":
    for m in (5, 6, 8):
        for rho in (0.0, 0.1, 0.3):
            print(m, rho, [(r, round(p_pass(r, m=m, rho=rho), 3))
                           for r in (0.99, 0.98, 0.97, 0.96, 0.95, 0.93, 0.90)])
    for share in (0.20, 0.25, 0.30, 0.326, 0.35):
        print("share", share, "P(<100 @500)", round(p_below(500, share, 100), 3),
              "P(reach 150 @500)", round(1 - p_below(500, share, 150), 3))
```
