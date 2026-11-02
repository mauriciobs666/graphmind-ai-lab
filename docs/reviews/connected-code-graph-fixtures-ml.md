# Methodology review — connected code graph fixture system test plan (S35)

> **Status:** active · **Owner:** `data-scientist` · **Tracks:** C-1001 (M10) · **Reviews:** `docs/test-plans/connected-code-graph-fixtures.md` (uncommitted, S35)

**Verdict: approve with suggestions.** The plan is sound as a deterministic dev-stratum conformance suite. I
found no blocker. Its arithmetic is right (recomputed below). Three majors should be folded into the next S35
revision before S36–S38 build rows from it, because each changes fixture content or an assertion count that is
cheap to fix now and expensive afterwards. Evidence level per finding is marked **V** (verified: recomputed or
traced in the text) or **I** (inferred).

Question and decision: do the planted rows support the protocol's statistical claims, and is the design free of
circularity and leakage? The caller either releases S36–S38 on these rows or sends S35 back. My answer is
release, with M1–M3 folded in.

## 1. What was checked

- **Row census (V).** I parsed the tables. 105 rows (LNK 35, AC6 18, SK 12, B1 8, DF 4, CD 4, KN 4, AB 4,
  OV 4, RV 3, TF 3, MR 2, SQC 2, NC 2). The gate split P1 69 · P2 14 · P3 18 · P4 1 · P5 3 matches §6.
- **Difficulty shapes (V).** Counted from the `shape` column: exact literal 3, constant across files 3, template
  literal 3, client wrapper 3, config-bound base URL 3, runtime-built key 4, dynamic import 3, DI/reflection 3,
  prompt mention 3, version skew 3. All ten ≥ 3, as §6 claims.
- **Kind × status matrix (V).** All 20 cells have a row (§7).
- **AC-25 (V).** DF distances 1,2,3 (+1 possible), CD ≥ 3 with CD-2/CD-3 guarding calls in other functions,
  RV ×3: the §12.2 floors are met.
- **KN rows (V).** I derived KN-1..4 by hand (decode, then resolve dots): the expected keys are right.

## 2. The protocol's statistical claims: what the fixture can and cannot carry

The floors, F = 35, the 75 and held-out separation are **proving-repo AC-5 claims** (Wilson bound of 35/35 =
0.901; 75 counting items; the 95% confirmed-precision and ≤ 4 wrong-possible per real impact). The fixture
supports **none** of them, and the plan correctly does not try to. Fixture items are `fixture-dev`, counted in
no `H`, and contribute nothing to the 75 or to F. Verified: §1, §11, TP-013. Consequences I would state in the
plan so no later report over-reads it:

- **Fixture results are existence checks, not rates (S1).** Three instances per shape gives a Wilson lower
  bound of 0.44 for 3/3, and the rows are hand-written, regular and small, so a pass is an upper bound on
  real-repo recall. The floors of 3 are a coverage design, not a power calculation. Add one sentence to §1:
  "fixture pass rates are not recall or precision estimates and are not extrapolated".
- **Held-out separation holds by construction** (different repositories, `stratum: fixture-dev`). TP-013
  verifies the label, which is tautological: it cannot fail unless someone mislabels. The real risk is the other
  direction, contact (S2 below).

## 3. Findings

### Major

**M1. The "44 assertion cells" figure is mis-derived, and the A25-3 assertions are mostly vacuous or
non-discriminating (V).**
- §6 counts "11 items × 4". The 11 is 8 forward rows + 3 `RV` rows, but for an `RV` row A25-1..3 apply to each
  *listed dependency*: 4 + 4 + 3 = 11 dependency entries, not 3. And A25-4 on the 8 forward items is the same
  two assertions (RV-1, RV-2) repeated, not 8 independent ones. The distinct assertions are roughly 24 (forward
  A25-1..3) + 33 (dependency A25-1..3) + 3 (A25-4 per `RV`). The true number is neither 44 nor independent;
  state the real set, because TP-008's "44 assertions pass" will be quoted as evidence.
- A25-3 ("no confirmed after a possible within a group") is non-vacuous for only four groups: DF-2/DF-4,
  CD-3/CD-4, RV-1 `core/1` (`is_member` c, `lookup` p) and RV-2 `core/1` (`tier_of` c, `lookup` p). RV-3 and every
  single-item group make it trivially true.
- Even those four mostly pass under accidental orderings. Alphabetical by name puts confirmed first in
  DF-2/DF-4 (`create_order` < `quote_order`), CD-3/CD-4 (`publish_created` < `publish_hook`) and RV-1
  (`is_member` < `lookup`). Only RV-2 (`lookup` < `tier_of`; also `hooks.py` < `rules.py`) puts the possible item
  first under name or path order. Mutant that must turn TP-008 red: an answer layer that sorts by
  (repo, component, distance, name) with no status key. As written it passes three of four pairs.
- **Fix:** require that in every non-vacuous A25-3 group the possible item sorts *before* the confirmed one
  under name order, path order and (where same file) line order; rename or move the `possible` companions
  accordingly (e.g. `a_`-prefix is unnecessary: swap which function is confirmed). Restate the assertion count
  as the enumerated set, and list the A25-3 pairs explicitly.

**M2. Precision and over-linking are essentially unmeasured: the manifests are not declared closed-world, and the
negative controls are two rows of one type (V).**
- TP-010 compares "every LNK/B1/KN status and reason with its row". Nothing says an *unplanted* emitted link
  fails the run. Only the `RV` rows ("no other dependency may be listed as confirmed"), OV-3 and NC-1/2 forbid
  extras. With ~25 planted confirmed links, the fixture could measure confirmed-precision, but only if the
  manifest is the complete list of cross-boundary links in the generated repositories. The fixtures are authored,
  so completeness is achievable. Declare it, and make "any emitted link or CONSUMES edge not in a manifest row"
  a TP-010 failure, or an adjudicated finding.
- NC-1 (fenced snippet) and NC-2 (comment) are the same family (text that is not code). Two controls cannot
  support an over-linking claim (0/2 has a Wilson upper bound of 0.66), and the plan correctly counts them toward
  no minimum, but the *coverage* of over-link modes is also thin. Missing hard negatives, all cheap in this
  fixture: same path with a different verb (`POST /api/items/{}` against a `GET` provider); a topic whose name
  extends a real one (`orders.created.dlq`); the same tool name on two servers (LNK-02/18 binds by URL, so
  one collision tests it); a consumer in dead or test-only code; a literal that *looks* dynamic but is constant
  and must be **confirmed** (see M3b). Suggest 6 more NC rows; they are not counted toward any minimum.
- **Fix:** add the closed-world sentence and the hard negatives.

**M3. The "≥ 3 per shape" floor is met only after P2, and P2 is scheduled last (V).**
- Requirements decision of 2026-11-01 orders the phases P0 → P1 → P3 → P4 → P5 → P2. Counting rows whose gate is
  ≤ P3: constant across files 2 (LNK-05 is P2), template literal 2 (LNK-09 is P2), DI/reflection **1**
  (LNK-31, 32 are P2), version skew **0** (LNK-33..35 are all P2). TP-004 counts from manifests and passes
  statically, so it cannot warn that the shape strata are thin at the P1, P3, P4 and P5 gates.
- **Fix, either:** (a) add Python/TS instances gateable at P1 for the three short shapes (a Python
  constant-across-files HTTP pair, a TS or Python template-literal pair, a second Python DI/dispatch site; the
  Python skew LNK-34 needs only Python, so gate it at the earlier of P1 and the library_api kind if that kind
  can ship for Python earlier); or (b) have TP-004 print the reachable per-shape count at each gate and record
  that shape-level readings before P2 rest on n = 1..2. (a) is better and costs about five rows.

### Minor

**m1. Hard shapes have no look-alike confirmed twin (V).** Shapes 6–9 (runtime-built, dynamic import, DI,
prompt mention) are possible-only, so an engine that downgrades everything dynamic to `possible` passes AC-6 and
every row of those shapes. The matrix keeps the engine honest elsewhere, but add one constant look-alike per
shape that must be **confirmed**: `importlib.import_module("plugins.x")` with a literal, an f-string whose parts
are all module constants, `getattr(obj, "fixed_name")`. This is the over-conservatism test the AC-6 "second
clause" has no mirror for.

**m2. Several AC-6 rows do not say why they are unprovable (V).** AC6-TOP-2 reads a topic from
`config.bus.topic` JSON, while LNK-28 confirms a config key read from YAML; AC6-TOP-1 reads an env var, while
LNK-13 confirms a base URL bound by Compose. The expected `possible` is only right if the value is absent, or
overridden, or multi-valued in the fixture. Each AC6 row needs a "why unprovable" clause (value not in the repo,
differs per environment, or computed). Otherwise the test enforces a policy and not a ground truth, and the
fixture author can make it pass or fail by accident.

**m3. Truth and expected emission are not separated (V).** For `possible` rows (LNK-16/17/20/29/30/31/32/33/34,
AC6-*) the "planted" fact is the engine's expected *output* (status, reason code), not an independent truth, and
many rows have no `P` locator (LNK-29..32, most AC6). Add a "true target(s)" column, which may be empty for
truly unknowable rows, and assert that the true provider appears in the candidate list for `name-match` and
`port-shared` rows (LNK-16, 17, 30). Otherwise a `possible` with an empty or wrong candidate set satisfies the
row. This matters because the protocol's "never miss" philosophy treats a missing `possible` as a miss.

**m4. Contact accounting is stale and the real contact route is shapes (V).** Eval-protocol §8.2 (S35 row) lists
`repoId ∈ {fx-svc, fx-web, fx-lib}`: `fx-job` is missing, so the GC assertion "no proving-repo locator appears"
would not scan it (`docs/plans/connected-code-graph-eval-protocol-ml.md` §8.2; owner `data-scientist`: I will
amend on the next revision). Because the `repoId` differs, that assertion passes trivially anyway. The leak that
matters is *shape contact*: SK-5 is named after the `salesperson/src/i18n/` case (S11 finding 3) and the
`.mcp.json` launch chain mirrors the proving repo's `cypher` chain. Reusing a proving-repo shape in a dev
fixture is legitimate (protocol §8 contacts anchors, not shapes), but add a provenance cell
"modelled on proving-repo shape: y/n" to the affected rows, so a later disagreement is attributable.

**m5. Scenario preconditions are unverified; two rows can be vacuous (I).**
- AB-1/AB-3: one syntax-error file (`legacy/broken.py`, `fx-job/job/broken.py`) need not fail a unit or a whole
  repository under a tolerant frontend, which skips unparsable files. The scenario then never reaches the state it
  tests. Define "build failed" for these scenarios (a forced failure in the build step is more reliable), and add
  a TP-006 precondition that the scenario state is produced.
- MR-PY/MR-TS: the rows assume the frontends resolve the cast call to `Y`. If a frontend resolves correctly,
  there is no mis-resolution to flag. Add a precondition check that the frontend output is the stated wrong
  callee, as a separate pass item.
- SK-5 (OI-7) is already flagged by the plan; same remedy.

**m6. DF-3 / RV-3 feasibility and the OI-4 circularity (I).** DF-3 flows a Python return value through an HTTP
response into a TS rendering and gates at P1, which requires cross-language data-flow stitched through a link,
not just a link. Confirm that this exists in the P1 design (S20/S33). If it does not, DF-3 and RV-3 are
link-level relations and should not be tagged as `DATA` hops. Separately, distances are *defined* by the
engine's own hop model (one hop per `DATA`/`CONTROLLED_BY` step, an HTTP link as one hop, OI-4): the oracle
and the system share one definition, so A25-1 tests conformance to the model, not that the model is right.
Say so.

**m7. TF-1 and the SQC rows detect only some mutants (I).** TF-1 should also change the line count of the
function above the sink, so a line-number-keyed implementation fails it (a body-only edit that keeps line counts
passes both correct and line-keyed implementations). SQC-PY / SQC-JAVA are single instances asserting a design
consequence (Python `symbolId` has no signature; Java's does), which is a regression pin, not evidence; fine,
but label it so.

### Nits

- **n1.** AC6-EX-3 gates P1 on a P3 kind, while LNK-03 (same kind) gates P3. TP-014 should state which governs.
- **n2.** LNK-35 (skew, same signature, **confirmed**) is the skew shape's control, not a hard instance, so the
  shape has two hard instances plus one easy one. Fine if stated; the "3 each" reads better as "2 hard + 1
  control" there. Some rows combine shapes (LNK-02 URL binding, LNK-04 mount composition + constant, LNK-08
  f-string + prefix + constant), so shape-level attribution of a failure is not clean.
- **n3.** TP-005 checks `KN` against `cpg/ccg/keynorm.py`, which is also the implementation under test. Here the
  expected values come from hand derivation and the F-13 ruling, which is independent of the code, but the
  ruling and the fixture share one author lineage. The check only certifies conformance. Keep the exit
  clause "fix whichever is wrong" tied to an RFC 3986 derivation, never to the implementation's output.
- **n4.** AC6-PM-1 and PM-2 are in the same file and nearly the same syntax class (prose and list), so "3
  distinct syntactic shapes" is closer to 2.5.

## 4. Questions answered

- **Do planted rows and counts support the protocol's claims?** Floors: yes (V). F = 35, the 75, held-out
  separation, confirmed-precision and wrong-possible ratios: not carried by the fixture, correctly (§2).
- **Leakage or circularity?** Low leakage risk. Circularity is definitional (m3, m6, n3): expected status and
  distances come from the same spec as the extractor. Plus a visibility risk: the answer key sits in the repo
  next to the rule packs and the exact AC-6 syntactic forms are in the plan, so the packs can be written to the
  list. Passing AC-6 then proves those forms only. AC-6 closes at the P2 exit, so there is time for a **sealed
  AC-6 variant set**: one or two extra instances per kind in a shape the plan does not list, written by someone
  not on the packs after they freeze, graded once. The OpenTelemetry first contact (AC-7, reported as-is) can
  double as an audit of `possible`/`confirmed` on its dynamic sites.
- **Are the 10 shapes ≥ 3 adequate?** Adequate as coverage at P2 completion, not before (M3). Not adequate for
  any rate statement (S1). Over-conservatism is untested (m1).
- **AC-25 44-cell design?** The coverage is good but the count and the ordering assertion are not (M1).
- **DF-4 / CD-4 companions justified?** Yes: without a group holding both a confirmed and a possible item,
  A25-3 is vacuous, and the companions are the cheapest way to get one. Keep them, but fix their sort position
  (M1). The RV lists already carry mixed groups, so the companions matter only on the forward side.

## 5. Suggested evaluation additions (summary, none requires protocol changes)

1. Closed-world assertion in TP-010; six hard-negative rows (M2).
2. Per-gate reachable shape counts in TP-004, or five more P1-gateable rows (M3).
3. A25 assertion set enumerated; possible-first-under-naive-order pairs (M1).
4. "Why unprovable" and "true target" columns (m2, m3).
5. Scenario and mis-resolution precondition items (m5).
6. Sealed AC-6 variant set at the P2 exit (§4).

## 6. Checks against `REVIEW.md`

- **Growth stops:** the document is new, so no rework growth to report.
- **A check that cannot fail:** TP-013 (label check) and TP-004 (manifest-only census) cannot fail against the
  generated repositories; TP-002 is the one that binds rows to code. TP-008's A25-3 lacks a mutant (M1). Named
  mutants that work: mask-before-decode fails KN-1/2 (TP-005); an engine confirming everything fails TP-009.
- **Live-spec:** no component spec is touched.

### Proposed REVIEW.md lines

None proposed. This is the first review of this document, so there is no recurring class yet.

## Pass 2 (S35 v1.1)

Rechecked by running: I re-parsed the tables (129 rows, 111 unique ids in the first-pass regex plus the hyphenated
AC6 ids; gate split P1 89 · P2 16 · P3 20 · P4 1 · P5 3 equals §6); re-derived the assertion count (24 + 33 + 3 =
**60**); read the four A25-3 pairs; independently re-derived KN-1..5 and the two mutants with a separate
decode/resolve script, and ran the plan's `kn_mutants.py` (same outcomes).

**Dispositions**
- **M1 fixed.** 60 distinct assertions confirmed. All four A25-3 pairs now sort the possible item first under name
  (`accept_quote` < `create_order`, `publish_alert` < `publish_created`, `find_hook` < `is_member`, `find_hook` <
  `tier_of`), under path (`core/hooks.py` < `core/rules.py`) and, for the two same-file pairs, under line. The
  expected answer order is written out, and TP-008 names the status-less-sort mutant.
- **M2 fixed.** Closed-world rule stated in §1 and TP-010 (with a planted unplanted-link stub); NC-3..7 add verb
  mismatch, name extension, tool collision, log message and prefix mismatch. Scope and completeness: N1 below.
- **M3 fixed, residual disclosed.** The per-gate shape table and TP-004 are present. At P1 every shape except skew
  holds 3 (exact 3, const 3, template 3, wrapper 3, config 3, runtime 3, dyn 3, DI 3, prompt 3). Skew reads 0 before
  P2 (library_api is a P2 kind), stated in §6. Accepted: skew readings do not exist before P2.
- **m1 fixed, partly.** TW-1..3 are confirmed look-alikes for dynamic import, runtime-built URL and reflection. No
  twin exists for prompt mention (e.g. a tool name in a real `callTool`, already LNK-02 in effect) or DI via
  `Depends`. Acceptable.
- **m2 fixed.** Every AC6 row has a "why unprovable" clause (TOP-1/2 say the value is set in no file).
- **m3 fixed in the rows, not in a test.** True targets are named on LNK-16/17/20/30/31/32/39/40 and AC6 rows. No TP
  asserts candidate containment: N3.
- **m4 partly.** Provenance line added (LNK-18, 21, SK-5). The `fx-job` entry in eval-protocol §8.2 is still owed by
  me (OI-10 records it); I have not amended it.
- **m5 fixed.** Forced `buildFailure` marker and TP-015 preconditions for AB and MR. SK-5 stays live-only (OI-7).
- **m6 disclosed.** LNK-36 now carries DF-3's link; DF-3/RV-3 are stated as link-level unless a cross-language
  `DATA` hop exists (OI-4), and §1 says A25-1 certifies conformance to the engine's own model. Accepted.
- **m7 fixed.** TF-1 adds a line; SQC rows are labelled regression pins and the seeded edge now targets the edited
  symbol.
- **Nits.** n1 fixed (AC6-EX-3 gate P3). n2 fixed for the control (LNK-41 split from the skew shape; three hard skew
  rows); the compound-shape rows remain (LNK-02, 04, 08), accepted. n3 fixed (execution note names the RFC 3986 and
  spec as arbiter). n4 fixed (PM-2 is a table cell in its own file).

**New findings**
- **N1 (minor). The closed-world rule has no planter-independent check and an undefined scope.** The rule holds only
  if S36–S38/S35b plant no incidental cross-boundary code, and TP-010 tests only its own stub. Unlinked
  consumers (NC-3/4/7 each emit a contract point with no provider) are "edges" that match no link row, so say that
  the rule covers resolved consumer-provider links, and that unmatched contract points are allowed. Add an
  item that greps the generated trees for route, tool, topic and table literals and checks each against a row
  (planted failing input: an extra `fetch("/api/health")` in a fixture file). State whether `fx-svc:tests/test_api.py`
  (SK-12) contains a client call; if so it is an unplanted link.
- **N2 (minor). NC-5 makes LNK-19 ambiguous.** A second `reindex` tool in `fx-job` means `scripts/refresh.sh` running
  `mcp-call reindex` (LNK-19, confirmed `key-match`, no URL binding stated) has two same-named providers. Either bind
  LNK-19 to `svc` through `.mcp.json` in the row, or downgrade it. As written the engine is right to answer either way.
- **N3 (minor). Candidate containment is claimed but untested.** §5.1/§5.3 say the answer's candidate set must contain
  the true target; TP-009 and TP-010 assert status and reason only. Add the assertion, with a stub that emits
  `possible` and an empty or wrong candidate set as its planted failing input.
- **N4 (minor). The KN evidence cites a session scratchpad path** (`.../scratchpad/kn_mutants.py`), which a reader
  cannot reproduce after the session. Commit the harness (e.g. `cpg/tests/`) and cite that path.
- **N5 (nit). TP-008 lacks a distance-first-sort mutant.** The expected orders (api/2 before core/1; bus/2 before
  core/1) discriminate it, so adding the mutant to TP-008's planted inputs makes A25-2 explicitly falsifiable.

**Verdict: approve with suggestions.** All three majors are fixed and the arithmetic and orderings hold. N1–N4 are
small edits for the next revision; none blocks S36–S38. Still owed by `data-scientist`: the `fx-job` amendment to
eval-protocol §8.2.
