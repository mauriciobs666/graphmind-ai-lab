# Review — connected code graph fixture system test plan (S35)

> **Status:** active · **Owner:** `analyst` · **Tracks:** C-1001 (M10)

**Scope.** `docs/test-plans/connected-code-graph-fixtures.md` (uncommitted, qa-engineer, 480 lines), against
`docs/plans/connected-code-graph-p0-2.md` (S35 row, §4.7), `docs/plans/connected-code-graph-bundle-spec3.md`
(§3.4, §3.5, §4.3, §6, §7, §1.7.1, §1.8) and `-spec4.md` §1, `docs/plans/connected-code-graph-eval-protocol-ml.md`
§12.2, `docs/plans/connected-code-graph-eval-protocol2-ml.md` (K42..K48), `docs/requirements/connected-code-graph.md`
(AC-6, 9b, 19, 21a, 25), the proposal §5.1, and root `REVIEW.md` (no `cpg/REVIEW.md` exists). The data-scientist
methodology review `docs/reviews/connected-code-graph-fixtures-ml.md` (M1..M3, m1..m7) is **not repeated**; where a
finding here overlaps it, it is cited. Nothing in the plan was edited.

**Verdict: needs changes** (one blocker, five majors).

CPG: considered, not relevant — a test-plan document over fixtures that do not exist yet; no loaded `cpg_*` graph indexes them.

## 1. Findings

### Blocker

**B-1. Consumer keys written `{}` contradict `keyNorm` step 8 and the plan's own LNK-17.** *(verified: read spec3 §6, rule-pack spec §`onUnknown`, LNK-04/07/08/13/17)*
LNK-04, LNK-07, LNK-08 and LNK-13 expect a **confirmed** `key-match` / `binding-confirmed` link whose consumer key is
`GET .../{}` (`/api/v1/orders/{}`, `/api/files/{}`, `/api/items/{}`). The consumers interpolate a function parameter
(`` `${API_PREFIX}/files/${name}` ``, `f"{BASE}/items/{item_id}"`, `getItem(id)`). Spec3 §6 step 7 gives `{}` to a
*provider* framework syntax (`{id}`, `:id`); step 8 turns "a consumer segment whose value is unresolved" into `{?}`, and the
rule-pack's `partial` mode makes that fact `possible`/`partial-key`. LNK-17 plants exactly that shape (`${id}`, unresolved)
against the **same provider** (`read_item`) as `possible`/`partial-key`. So LNK-08 and LNK-13 versus LNK-17 are one input shape
with opposite expectations, and a conforming engine fails TP-010 on LNK-04/07/08/13. It also hollows the "template literal"
shape: LNK-07 and LNK-08 are two of its three rows, so it is really 1 until fixed.
*Fix.* Either resolve the interpolated part to a constant in source (`` `${API_PREFIX}/files/readme` `` gives literal
`GET /api/files/readme`, matching a `{}` provider as `key-match`), or re-state these rows as `possible`/`partial-key` and plant
new confirmed template-literal rows whose every part is a visible constant. State per row whether each interpolation resolves.

### Major

**M-1. Locators are not in the §3.5 form, so TP-002 cannot pass as written.** *(verified: spec3 §3.4, §3.5, §4.3, Java example at §4.2; counted rows)*
- *repoId and path.* Spec3 §3.5/§4.1: `path` is **repo-relative** and `qname` starts `path + "::"`; the repo is `fx-svc`. The plan's
  shorthand `svc:` = `fx-svc/` and `web:` = `fx-web/app/` puts the repo directory into the path (`fx-svc/api/health.py`); the
  spec3 Java example shows the right form (`repoId fx-lib`, `path src/main/...`). SK-8 (`fx-web/scripts/build.test.ts`) and §4's
  `rootPath` `fx-web/app` repeat it. Define `svc:` as `repoId fx-svc, path` (no prefix) and `web:` as `fx-web`, `app/...`.
- *Java signature.* §4.3 requires a `signature` for every Java method; LNK-23, 24, 25, 31, 33, AC6-TOP-3 and SQC-JAVA give none, and
  LNK-25 is the overload case where it matters. LNK-33/SQC-JAVA use `...` ellipses.
- *Non-locators.* LNK-03/13 ("service `job` env"), LNK-20 ("all mcp_tool providers"), `svc:db/schema.sql#orders`,
  `svc:config/app.yaml#db.pool_size`, "entry `fx-svc`" are prose or an invented `#` form; config anchors are `{repoId, path, keyPath}`
  (spec3 §1.9). `.proto` and `.sql` have no `lang` token in §3.3, so LNK-23's provider and B1S-D cannot convert to a `symbolId`.
- *Units.* §4 titles its column "Components (unit `rootPath`)" yet LNK-18/21/22 say "unit `fx-svc`" (one unit serving `api` and `mcp`),
  and AB-1/2 say unit `legacy`/`tools`. The units, their `rootPath`, `lang` and `vendoredDirs` are never tabulated; if `fx-svc` is a
  root unit, `legacy/` and `tools/` are nested inside it. Without that table the §1.9 unit locators, SK paths and AB scenarios are not resolvable.
*Fix.* Add a unit table (repoId, rootPath, lang, vendoredDirs) and a locator-shorthand sentence; give Java signatures; replace prose
with `{repoId, path, keyPath}`; name the `.proto`/`.sql` anchor as the Java impl / `schemaEntry` `objectKey`.

**M-2. TP-005's named mutant is false, and no KN row exercises the masking order the ruling is about.** *(verified: ran `cpg/ccg/keynorm.py::mask_http_segments` on KN-1..4)*
TP-005 says "a mask-before-decode implementation fails KN-1 and KN-2". None of KN-1..4 contains a segment the token screen (§1.7.1)
masks, in either order: masking the raw strings returns them unchanged, and masking the decoded ones also changes nothing. The mutant
that KN-1/KN-2 do catch is **resolve-before-decode** (K42's). The expected keys themselves are correct (re-derived by hand:
KN-1/2/3 give `GET /api/health`, KN-4 gives `GET /api/files/a%2Fb`; `a%2Fb` passes the screen). To pin decode-then-mask add a segment whose
encoded form passes and whose decoded form fails T6: `/api/hooks/x7k2%2Dq9w4%2Dm1p8%2Dz3v6` must give `/api/hooks/{?}` (consumer) /
`{#}` (provider). Run on the current code, mask-first leaves the secret-shaped text in the key (a leak, SA-9), decode-first masks it.

**M-3. SQC-PY is vacuous as written.** *(verified: read §5.5 DF-4, §5.9)*
The seeded verdict is on "the name-match edge from `quote_order` (DF-4)", but DF-4's edge is `discount.apply(...)`, whereas H3 edits
`core/flow.py::price`. Python `symbolId` has no signature, so the edge's target `apply` is untouched by either rule: "note kept" holds
if the identity rule is implemented, ignored or inverted. Name the callee: `quote_order` must call `price` through an untyped receiver
so the seeded edge targets the edited symbol; then "kept" can fail (an identity that included the parameter list would drop it).
The Java half is sound in shape (the old `quote(int,String)` id vanishes), but LNK-33 says the provider has "only" the 3-arg form
while LNK-25 plants a 4-arg `Locale` overload: reconcile.

**M-4. Several test items cannot fail (root `REVIEW.md`, "every check can both fail and pass").**
Only TP-009 names a failing input. TP-002 (locators resolve), TP-003 (manifests validate), TP-004 (census), TP-012, TP-013 and TP-014 name
no planted bad locator/manifest/row, so S39 has nothing red to demonstrate them with. TP-004 additionally takes its expected counts from
this plan's §6 and the floors from the plan's author reading: add one mutated manifest (a row deleted, a shape tag changed, a gate moved below
its dependency) per item. TP-005's mutant is wrong (M-2). TP-008's weakness is ML M1.

**M-5. The "version skew" shape reaches the floor of 3 only by tag, not by meaning.** *(verified: read LNK-33/34/35)*
§12.2's shape is "cross-repository call with version skew". LNK-33 is inside `fx-lib` (both `quote-cli` and `Pricing`), so it is not
cross-repository; LNK-35 is the same-signature control and has no skew. Only LNK-34 is a cross-repository call whose signature actually
differs. Effective count: **1** (strict), **2** (signature differs), **3** (tag count). Plant two more cross-repository skewed calls
(e.g. fx-web→fx-svc is not a library, so use `fx-job`→`fx-svc` `common` twice more, or split `quote-cli` into its own repo), and keep
LNK-35 as the control. Phase of these rows: ML M3.

### Minor

- **m-1. AC6-EX-3 gate.** `env_var` is a P3 kind (§3 phases), yet AC6-EX-3 gates P1. At P1 either the extractor emits no env_var link, so
  "possible with reason" fails, or the row is satisfied by nothing emitted. Gate it P3 or state that P1 emits `env_var` for example config only.
- **m-2. A25-2 has no expected sequence.** The key is (repository, component, distance). For `CF-DF` that puts `fx-svc/api/2` before
  `fx-svc/core/1` if components sort by name, and the reverse if "closest first". The plan never writes the expected group order, so S39 binds
  an arbitrary one. Write the ordered group list per question and name the component sort. (Counts and A25-3 discriminators: ML M1.)
- **m-3. OV list is incomplete.** `CS-JOB` edits `trigger_server.py::status`; the planted `possible` consumer `web:.../JobStatus.tsx` (LNK-16)
  is a consumer of the same `GET /job/status`, yet is neither in OV-1..4 nor in OV-3's negatives. Likewise LNK-15's caller (`POST /job/trigger`,
  not edited) belongs in OV-3. Decide and list both.
- **m-4. Open items OI-3, OI-5, OI-6.** OI-3 routes to "S19", which is not a P0 step (the table runs S16, S17, S20); route to S30/S16. OI-5's
  premise is stale: the field-depth cap is `fieldDepthCap` = 3 (spec3 §5.1), so a `field-truncated` row (`a.b.c.d`) is plantable now and is a P1 code; only
  `flow-unavailable:<lang>` waits on S27's `flowAvailable`. OI-6 ("the later secrets step") names no owner. OI-1, OI-2, OI-4, OI-7 are real
  and routed (OI-2 verified: p0-2 §4.7 and S39 still say three repos).
- **m-5. `fx-lib` module cycle.** LNK-24: `Pricing.quote` (module `pricing`) calls `FxBus.publish` (module `bus`); LNK-25: `PriceListener.reprice` (module `bus`)
  calls `Pricing.quote`. If `pricing` and `bus` are separate units (§4), that is a cyclic build dependency; move `PriceListener` to `pricing` or `quote-cli`. *(inferred from §4's unit list.)*
- **m-6. Omitted-file rows.** (i) Spec4's coverage probe is a table over axes; SK-1..12 is a hand list. Uncovered cells: `.mts/.cts/.js/.jsx/.mjs/.cjs`
  extensions, `frontend-ignore` under `vendoredDirs` and outside `rootPath`, `__tests__`/`e2e` nested deeper, case. Generate the cells from the axes and let a missing cell fail.
  (ii) TP-007 expects outcomes with override **on and off**, but spec4 says override is unverified and the fallback is the likelier case; OI-7 only
  covers the OFF column. Make ON conditional on the P1 plan. (iii) SK-10 is one row for two files (count 105 vs 106). (iv) SK paths are rootPath-relative while `skippedFiles` paths are repo-relative. The test pattern, `.d.ts`, vendored and precedence rows (SK-1..4, 6, 7, 9, 11, 12) match spec4 §1 exactly.
- **m-7. `target-not-mapped` needs a Link.** AB-1/AB-2 are Python calls into another unit; `target-not-mapped` is a `Link (TO_UNIT)` reason (spec3 §7.2),
  and §6's matrix has no link kind for a plain cross-unit call. Say which kind (library_api or http) carries it.
- **m-8. History windows.** p0-2 §4.7 requires in-window and out-of-window commits; §8 mentions them in prose, with no commit ids, dates, rows or assertion, and
  they would change SHAs of the H0..H3 chain. Add them as named commits or defer explicitly to a successor.
- **m-9. SC-STEADY.** "all rows not named below" excludes LNK-15, LNK-16, LNK-26, B1S-R and the AB-1/2 callers from steady state, because AB-3/AB-4 name them.
  Say "all rows, plus the scenario rows under their scenario".

### Nit
- D-8 covers LNK-31..35 and the §5.1 note LNK-32..35: one scope. Proposal §5.1's B6 (verdict during a swap) is an integration test "on the fixture system" that no row covers; confirm it needs no planted item.

## 2. Re-count (check a)

| Definition | Count |
|---|---|
| Table rows in §5 (script-parsed, IDs unique) | **105** (author's figure holds) |
| Excluding NC-1/2 (declared "not counted toward any minimum") | 103 |
| Also excluding OV-3 (a "not listed" negative) | 102 |
| Files rather than rows (SK-10 plants two) | 106 |
| Status-bearing link/edge rows (LNK, KN, AC6, B1, DF, CD, MR, OV-1/2/4) | 78 (+11 RV dependency entries) |

Spread 102 to 106; the 105 is correct for "one table row". Per category and the gate split P1 69 · P2 14 · P3 18 · P4 1 · P5 3 reproduce
(KN and SK carry their gate in prose, so a column-based count under-reads P1 by 16). Floors: AC-6 six kinds x 3 = 18 met; DF 4 (distances 1,2,3) and CD 4
(CD-2/3 cross-function) met; RV 3 met; shapes 10 x 3 met by tag (runtime-built 4), **except version skew (M-5)**; template literal is 1 until B-1 is fixed;
link kind x status: all 20 cells filled, §7 agrees with §5 row by row. Assertion cells: 44 as stated; distinct assertions about 60 (24 forward A25-1..3 + 33 dependency + 3 reverse), 36 if repeats of RV-1/RV-2 A25-4 are dropped and dependencies are not counted (ML M1).

## 3. Other checks
- **(c) Gates.** Verified by scanning P3 rows: none needs Java; Java rows are the 7 stated and all P2. No P1 row depends on a later kind except AC6-EX-3 (m-1).
- **(d) AC-6 / AC-25.** AC-6 half "none confirmed" is non-vacuous only because TP-009 also requires `possible`+reason; reasons for the unprovable
  rows are mostly right, but several rows do not say why nothing could resolve them (ML m2). AC-25: both effect kinds, distances 1-3, grouping, reverse all planted; see m-2, ML M1.
- **(f) skippedFiles.** See m-6; reasons are limited to `test|frontend-ignore` as spec4 requires.
- **(h) Growth-stops / size.** First version, so no revision growth to report. For the next revision: the derived tables (§6 gate census, §7 matrix, §6 totals)
  restate facts of §5 and will drift; generate them (TP-004) or keep one. "Fixture items are dev data" appears in §1, §6, TP-013, §11; the skew-reason rationale
  appears three times (§5.1 note, D-8, OI-3). Rework must replace text in place with no finding-to-resolution table.

## 4. What is solid
Row counts and gates are consistent and re-derivable; `fx-job` makes AC-19 and cross-repository AC-9b P3 without Java; the K42-shaped KN keys and the
test/decoy omitted-file rows are correct against spec3/spec4; negatives (NC, OV-3, RV "no other confirmed") exist; D-1..D-9 are visible and reversible.

## 5. Open questions
1. B-1: resolve interpolations (preferred) or re-state the rows? Needs the P1 pack owner's confirmation of how a consumer path parameter is expressed.
2. M-5: add skew rows in `fx-job` (cheap) or split `quote-cli` into a fifth repo (changes AC-19 and "four repositories")?

### Proposed REVIEW.md lines
Target: root `REVIEW.md`, "Always check". *Source: M-2.* Add: "A test or plan row that names the mutant it catches is checked by running that mutant against the oracle; a mutant the row does not actually turn red is a finding."

## Pass 2 (plan v1.1, 129 rows claimed)

**Scope.** Re-check in place of `docs/test-plans/connected-code-graph-fixtures.md` v1.1 (634 lines, 10,081 words; was 480 / 7,463). Verified by
execution: a script parsed every §5 table (rows, gates, Java rows, dangling ids); the KN literals were run through `cpg/ccg/keynorm.py` composed
with my own decode/resolve harness (not the plan's); fixes were then checked against each other.

**Verdict: needs changes** (0 blockers, 2 majors, 4 minors). Stopping condition at the end.

### Dispositions of Pass 1

| Finding | Disposition | Evidence rechecked |
|---|---|---|
| B-1 consumer `{}` | **Fixed** | LNK-04/07/08/13/37/38 now interpolate named constants (keys `.../1001`, `readme.txt`, `42`, `7`, `1`); LNK-17 still `p`/`partial-key`; no confirmed row has a parameter-derived segment. Template shape reads 3 at P1 (LNK-07, 08, 38). |
| M-1 locators | **Fixed, one residue** | Paths repo-relative; unit table (§4) with rootPath/lang/vendoredDirs; Java signatures on every Java locator except SQC-JAVA (N-3); `@keyPath` config anchors match spec3 §5.17; `.proto`/`.sql` anchored on code or `schemaEntry`; module cycle gone (`bus` now inside unit `pricing`). Nesting is declared an assumption (OI-8, routed to S16). |
| M-2 KN mutants | **Fixed** | My harness: correct order passes KN-1..5; resolve-before-decode fails exactly KN-1, KN-2; mask-before-decode fails exactly KN-5. KN-5 expected `GET /api/hooks/{?}` is right (decoded text fails T6). |
| M-3 SQC-PY | **Fixed** | Seeded edge is now `estimate` to `price`, the edited symbol. |
| M-4 checks that cannot fail | **Fixed, one weak** | All 17 TP rows name a planted failing input (TP-006's is weak, N-4). |
| M-5 skew | **Fixed** | LNK-33..35 are `fx-job` to `svc-common` (cross-repository, signature differs); LNK-41 is the control. Strict count 3. |
| m-1 AC6-EX-3 | Fixed (gate P3) | row text |
| m-2 A25-2 order | Fixed | §5.5 writes the order; byte order checked (api<core, bus<core, core<db<state) |
| m-3 OV | Fixed | OV-5 (JobStatus, `p`/`port-shared`) added; OV-3 gains `trigger_job` |
| m-4 OIs | Fixed | OI-3 to S30; OI-5 now only `flow-unavailable`; `field-truncated` planted as FT (but see N-1); OI-6 owner S30 |
| m-5 module cycle | Fixed | above |
| m-6 SK | **Fixed** | 18 rows incl. `.mjs/.cts/.jsx`, nested `__tests__`, vendored `i18n`; generated probe table in TP-007; ON column conditional; paths repo-relative; SK-10 split (rows = files) |
| m-7 `target-not-mapped` carrier | Fixed | `http` links (`/legacy/fee`, `/tools/run`), but see N-2 |
| m-8 windows | Fixed | H0..H4 dated; TP-016 with a failing boundary |
| m-9 SC-STEADY | Fixed | scenario text now says what AB-3/4 override |

### Recount (several definitions)

| Definition | Count |
|---|---|
| Table rows in §5 (parsed, 129 unique ids) | **129** (claim holds) |
| Excluding NC and TW ("not counted toward any minimum") | 119 |
| The plan's "positive items" (also minus OV-3, FT-2) | **117** (holds) |
| Also minus LNK-41 and KN-3/KN-4, which the plan itself calls controls | 114 |

Spread 114 to 129; the "12 negatives, twins and controls" list omits three rows the text calls controls, so say which. Per gate P1 89 · P2 16 · P3 20 · P4 1 · P5 3
reproduce exactly. Java rows: LNK-23/24/25/31, AC6-TOP-3, SQC-JAVA, all P2; **no P3 row references Java**. Per-category totals (41 LNK, 5 KN, 18 AC6, 3 TW, 8 B1, 8 DF/CD, 3 RV, 7 NC, 2 MR, 4 AB, 5 OV, 2 SQC, 3 TF, 18 SK, 2 FT) all match.

### New findings

**N-1 (major). FT-2, the boundary control, is itself over the cap.** *(verified: spec3 §1.1 depth-cap bullet, §5.1)*
Heads are `p<i>`/`recv`/`r`; the cap counts fields **after** the head. `self.cfg.a.b.c` is `recv` + 4 fields (cfg, a, b, c), so it is truncated
(`possible`, `field-truncated`), yet FT-2 expects `c`/`intra-flow` and labels it "head + 3 fields". FT-1 (`self.cfg.a.b.c.d`, 5 fields) is truncated either way.
A conforming engine fails TP-010 on FT-2. Fix: FT-2 `return self.a.b.c` (3 fields), FT-1 `return self.a.b.c.d` (4); say "head" means `recv`.

**N-2 (major). The closed-world rule (§1) is contradicted by rows the plan itself plants.** *(verified: grep for `/legacy/fee`, `/tools/run`, NC-3/4/5/7)*
§1 says every cross-boundary link or `CONSUMES`/`PROVIDES` edge emitted must match a row. (a) AB-1/AB-2 plant `legacy_fee` to `GET /legacy/fee` and `run_tool` to
`GET /tools/run`; those links exist in every state and only the scenario expectations are written. §4 says `svc-tools` is "never built" and SC-STEADY says "all four repositories built",
so the steady-state of `svc-tools`, and of those two links, is undefined: a conforming engine emits an unrowed link. (b) NC-5 adds a second `mcp_tool` provider
`reindex` in `fx-job`, and LNK-20 (`shell-dynamic`, no binding) expects "every provider of `svc`" while spec3 §7.2 `unknown-key` is "one link per provider of the kind": the job provider is
an unrowed or extra link. (c) NC-3/NC-4/NC-7 are real `fetch`/subscribe calls, so they emit `CONSUMES` edges that NC rows say nothing about ("no link" only).
Fix: add steady-state rows for the two AB links and state `svc-tools`' steady build state; restrict LNK-20's targets to what binding allows or list the job provider; reword NC-3/4/7 to "a `CONSUMES` edge, no `Link`".

**N-3 (minor).** SQC-JAVA still carries `.../` ellipses (TP-002 cannot convert it); its "untyped `handle`" is not expressible in Java: say the receiver's type is unresolved because it comes from a jar absent from the classpath.
**N-4 (minor).** (i) TP-006's planted input is a parse error, but D15 fails a unit only on zero non-external `METHOD`s (per-file rule not adopted for JS): plant a unit with only comments. (ii) §5.2 cites a session scratchpad path (`/tmp/claude-1000/.../kn_mutants.py`) as the harness; it exists today, will not tomorrow: commit it as TP-005's test. (iii) "library_api is a P2 kind" (skew table note) contradicts AC6-DYN/TW-1, library_api rows gated P1: say "version-skew linking is P2".
**N-5 (minor, growth-stops).** Growth 480 to 634 lines is mostly new scope (24 rows, TP-015..017, unit table). Rework growth to prune at the next revision: the constants rule is stated in §5.1's header and D-10; the skew rationale in §5.1's note, D-8, D-11 and §6; the A25-3 discriminator in §3, §5.5 and TP-008; the 89/16/20/1/3 census and §7 matrix are derived copies; D-10/D-11 are finding dispositions ("ruling, B-1/M-5") restating §5.1; §11's sealed-variant paragraph is a review recommendation, not scope. One dated revision line is correct.
**N-6 (nit).** OI-9 (TW-3 literal `getattr` expected `resolved-call`) is honest but unresolved; a `p`/`name-match` expectation is safer until the P1 extractor is known.

### Stopping condition (falsifiable)
`needs changes` clears when all hold: (1) FT-2's expression has exactly 3 fields after `recv` and FT-1 has 4 (read from the row text); (2) every link the plan plants appears in a
non-scenario row, i.e. `grep -n '/legacy/fee\|/tools/run' docs/test-plans/connected-code-graph-fixtures.md` returns a row outside AB-1/AB-2, and §4/§8 state whether `svc-tools` is built in SC-STEADY;
(3) `grep -n '\.\.\.' ` finds no `...` inside a locator cell. N-3 (ii)-N-5 may ride the same revision; none gates.
