# Connected code graph — fixture trees gate (S36/S37/S38/S35b)

> **Status:** active · **Owner:** `analyst` · **Tracks:** C-1001 (P0) · **Reviews:** `docs/test-plans/connected-code-graph-fixtures.md` v1.2 against `cpg/fixtures/system1/**`

## 1. Scope and verdict

Reviewed: the four untracked trees `cpg/fixtures/system1/{fx-svc,fx-web,fx-lib,fx-job}/**` (132 files) and `expected/{fx-svc,fx-web,fx-lib,fx-job}.yaml`, against test plan v1.2 §3-§8. Manifest key format is each builder's own (fx-svc structured; fx-web/lib/job compact); judged for consistency, not against the S39 schema (does not exist). Nothing in trees or plans edited; no git mutation (a `compileall` I ran created `__pycache__` dirs, removed; file count back to 132).

**Verdict: needs changes** (one blocker, several majors). The trees realize the plan's rows faithfully and almost completely. What fails is mostly the plan's own text: one confirmed cross-repo link no row plans, two literals that the closed-world rule would flag, and one contradiction between rows. Each fix is small; none requires rebuilding a tree.

CPG: considered, not relevant — the fixture trees are not a loaded `cpg_*` graph; this gate is by execution against the plan.

**Checks run (all executed):**
- Builder checkers re-run: `check_job.py` GREEN (29 locators, 26 rows); `check_fxweb.py` GREEN (43/50; run on `fx-web`, RED only when pointed at `fx-web/app`, a path-argument slip); `fxlib/check.py` GREEN (6 rows, 2 units); `s36/check.py` GREEN (89 rows, 156 locators, 56 files).
- Census (own script, plan text parsed afresh): plan §5 yields **131** row IDs; manifests' union is the same 131, none missing, none extra. Per-prefix counts equal §6 (LNK 43, KN 5, AC6 18, TW 3, B1 8, DF/CD 8, RV 3, NC 7, MR 2, AB 4, OV 5, SQC 2, TF 3, SK 18, FT 2). Gate counts from manifests: P1 91 · P2 16 · P3 20 · P4 1 · P5 3, equal §6. Difficulty-shape counts per gate equal the §6 table. All 10 kinds x 2 statuses present.
- Status/reason/gate of 133 manifest rows compared with the plan tables: **0 real differences** (3 printed hits are my parser on `a / b` reasons and a pipe in AC6-PM-2). The 37 rows held by two or three manifests agree on status, reason and gate; locator sets differ only by role fields (see F-4 for keys).
- Mutants (copies, builders' checkers): see F-8.
- Compile: not blocking, and no longer unchecked. `javac` (host JDK 21) with the builder's grpc/Handle stubs compiles fx-lib at HEAD **and** at the H0 variant (the `pre_h3_edits` applied, each `old` occurring once). `tsc --noEmit` strict from the `ccg-toolchain-truth` image (`/opt/truth/npm/scip-typescript/node_modules/.bin/tsc`, `--network none`) passes fx-web with module shims; a planted type error turns it red (positive control). `python -m compileall` passes for fx-svc and fx-job. Limit: shims, so SDK API usage is not checked against real types.

## 2. Findings

### F-1 [blocker, plan + two manifests] `order_report` is an emitted confirmed link with no row
Verified: `fx-job/job/run_report.py:53` writes `INSERT INTO order_report`; `fx-svc/api/reports.py:12` reads `SELECT * FROM order_report`; no `LNK`/`B1`/DDL row exists (`grep order_report` in plan §5: only OV-1). OV-1 asserts it as `confirmed`/`key-match`, but plan §1 says every emitted `Link` must match a row (TP-010, zero unplanted), so the system as planned cannot pass TP-010 at P3.
**Ruling:** add a row, not an exemption (OV-1 already asserts the link). `LNK-44`: `data_entity`, key `sql:table:order_report`, P `job:job/run_report.py::write_report`, C `svc:api/reports.py::report_view`, `c`, `key-match`, gate P3, serves LNK, OV-1. Keep `SELECT *` (no column keys, so no column link); no DDL needed.
**Owner:** `qa-engineer` for plan v1.3 and the census (LNK 44, rows 132, positives 120, P3 21, matrix data_entity cell, OV-1 serves); the fx-job and fx-svc builders add the row to their manifests. Trees unchanged. Verified by reading and grep; the engine behavior is the plan's own statement.

### F-2 [major, plan row + fx-job tree] second reader of column `orders.total`
`run_report.py:43` (`load_orders`, B1S-R) and `:47` (`render_row`, LNK-26) both `SELECT total FROM orders`. The plan's B1S-R text prescribes the first. At column granularity (LNK-26 is `sql:column:orders.total`) `insert_order` -> `load_orders` is an unplanted link, and OV-4 would list `load_orders` too (OV-4 names only `render_row`). Inferred from the key model, not run.
**Fix:** B1S-R reader becomes `SELECT count(*) FROM orders` (table key only; no column key). Plan edit by `qa-engineer`, one-line tree edit by the fx-job builder, then re-run `check_job.py`.

### F-3 [major, plan row + fx-job tree] LNK-14 and LNK-16 contradict each other on `localhost:8000`
LNK-16 is `possible`/`port-shared` because fx-svc Compose (`api` 8000) and fx-job `scripts/dev.sh` (`--port 8000`) both claim 8000; spec3 §7.2 defines `port-shared` as "two different units on one port". LNK-14 is `confirmed`/`binding-confirmed` on the very same origin: `fx-job/config/job.yaml:1` `base_url: http://localhost:8000`. By the rule LNK-14 is `port-shared` too. (The path match cannot rescue it: `/job/status` also matches only one provider and the plan still says `possible`.)
**Fix:** change `base_url` to `http://api:8000`. Host `api` is the Compose service with a build context, which is exactly LNK-22's chain, so the binding resolves to unit `svc` alone. One-line tree edit by the fx-job builder, plus one line in `check_job.py`'s allowed-URL set (its allowlist currently turns that line RED: mutant J5). Plan: record the origin as `http://api:8000` in the LNK-14 row.

### F-4 [major, manifests] two cross-manifest `key` disagreements, plus drifting key forms
LNK-08: fx-svc `GET /api/items/42`, fx-job `GET /api/items/{}`. LNK-13: fx-svc `GET /api/items/1`, fx-web `GET /api/items/{}`. The plan intro says a confirmed row's key is the literal after constants resolve; fx-web already does this for LNK-38 (`/api/items/7`), LNK-07 and LNK-04. Skew rows differ in form (fx-svc full `python:...(amount,*,mode)`, fx-job short `round_money`; LNK-34, 35, 41). Which manifest S39 reads would decide TP-010.
**Fix:** fx-web and fx-job builders use the literal key (`GET /api/items/1`, `/42`); the plan's LNK-08/13 key column states the literal. Skew-key form is for the S39 schema to fix; the fx-svc form is the complete one.

### F-5 [major, plan decision] Compose `web`/`job` services reach no unit
`fx-svc/deploy/compose.yaml`: `api` has `build.context: ..` (LNK-22 resolves to unit `svc`); `web` and `job` have only `image: fx-web:latest` / `fx-job:latest`. LNK-15 (`binding-confirmed` on `localhost:8300`) and OV-2 (`VITE_JOB_BASE=http://job:8300`) need "port 8300 / host `job` = unit `job`". Nothing in either tree establishes it: `fx-job` has no Dockerfile or compose, `dev.sh` says 8000, and the compose file lives in fx-svc, so by file location 8300 is a launch config of unit `svc`. Conversely, if the engine maps `image:` names to repos, `web` and `job` form launch links that no row plans. Inferred from the tree; the engine rule is the S30 owner's.
**Ruling on (f):** placement in `fx-svc/deploy/` is plan-correct (LNK-03's P and OV-2 pin it). The open point is the service-to-unit mapping. **Owner:** `architect`/S30 owner to decide: (i) `image:` name = repo name is a stated binding rule (then add planned launch rows for services `web`, `job`), or (ii) `job` gets `build.context: ../../fx-job` and a minimal `fx-job/Dockerfile` (cross-repo path, siblings in the materialized layout) plus a launch row like LNK-22. Until decided, LNK-15/OV-2 cannot be shown derivable.

### F-6 [major, S39 schema] the checked-in tree means a different commit per repo
fx-svc: tree = H0, history adds forward ops. fx-job: tree = HEAD (already holds the H4 line, `run_report.py:2`), H0 = "minus the H4 line". fx-lib: tree = HEAD (H3), `pre_h3_edits` reverse to H0/H1. fx-web: tree = H0 = H1 = HEAD. Mechanically sound (I applied every `replace_text` and `pre_h3_edits`; each `old` occurs exactly once), but S39 must implement forward and reverse ops and TP-001 determinism depends on it. **Fix:** S39 schema fixes one convention (recommend: tree = H0 everywhere, forward ops only); fx-job and fx-lib manifests are re-expressed by their builders then.

### F-7 [major, plan] RV rows need a stated model and depth cap (your item c)
The tree is consistent with a **pure data-dependency model, functions only, depth <= 3**: `eligible` discards `audit(o)`'s result (`rules.py:34`), `create_order` does not use `repo.save`'s result. Under a call-based reverse, RV-2 also lists `audit` (core/1, confirmed) and RV-3 lists the `repo.save` candidates, so "the list is complete" fails. Independently, RV-3 stops at `base_price` (dist 3) while RV-1 lists dependencies of `base_price` at 1-2, so only an explicit cap reconciles them. Reasons in RV-1/RV-2 are otherwise consistent (`find_hook` `p`/`name-match` via untyped `hooks`; `connect` at db/2 via `is_member`; `load_rate` db/2 via `tier_of`).
**Fix:** one sentence in §5.5: reverse = data dependencies, distance <= 3, functions only; call-based is not the model. Tree stays. Owner `qa-engineer`; S33 confirms.

### F-8 [minor, assurance bound] builder checkers are allowlist-style and gate-blind
Run on copies, all by me:
- the plan's own TP-018 failing input (`fetch("/api/health")`/`requests.get("/api/health")` added to a file) is **undetected** by fx-job (J1 GREEN), fx-web (W1 GREEN) and fx-svc (S1 GREEN), since the literal is globally allowed;
- a second `callTool("get_item")` in a new file is undetected (W5); an unquoted route in a comment is undetected (J3, by design);
- a wrong gate (LNK-26 P3->P1, LNK-05 P2->P1, LNK-24 P2->P1) is undetected in all (J2, W2, L2): no checker compares status/reason/gate to the plan;
- caught correctly: renamed symbol (J4, W3), new route in a new file (W4), unplanned URL (J5), new topic/table/env (S3-S5), changed or extra Java literals (L1, L3, L4).
My own plan comparison above replaces the gate check. **Fix:** S39's TP-018 must key on (literal, enclosing locator), not on a global literal set, or its planted input will pass. Tool-side only; no tree change.

### F-9 [minor, plan row] LNK-24 and AC6-TOP-3 locators (your item d)
LNK-24's P is `FxBus.publish(String,Object)`, but the row says the call is in `Pricing.quote`; every other topic row (LNK-05/09) points at the publishing function. **Ruling:** P = `Pricing.quote(int,String,Currency)` (HEAD/steady state), `FxBus.publish` as the wrapper locator, as in LNK-10..12. The fx-lib manifest already carries `publish_call_site`. AC6-TOP-3: "no true target" is correct; say so in the row (like AC6-URL-3) because TP-019 covers only named targets. Same-unit calls (`PricingService.quote` -> 4-arg `Pricing.quote`, `Pricing.quote` -> `FxBus.publish`): add one sentence to §1, "same-unit calls are `CALLS`, never a `Link`"; otherwise they are an unplanted-link risk. javac and the lib checker pass; no tree change.

### F-10 [minor, fx-svc tree] every `/api` provider key depends on one `for` loop
`api/app.py:11` mounts all seven routers in `for module in (...)`: `include_router(module.router, prefix=API_PREFIX)`. Plan LNK-04 wants "mount composition", but a loop makes one extractor limitation zero out every `/api` row at once, and blurs LNK-04's shape. **Suggest:** unroll into seven explicit `include_router` lines. Inferred.

### F-11 [minor, ground truth and containment]
- B1: `OrderCache.__init__` also writes `entries`, and `render_row` also reads `orders`, so B1-1/B1-3 answers contain extra members beyond the planted W/W2/R. Say B1-1..4 are containment checks plus B1-4's exclusion; otherwise "exact" turns red on correct output.
- LNK-39: nothing in the tree registers `AuditPlugin` into `REGISTRY`, so "true target `AuditPlugin.run`" is not derivable from the repository (AC-6 rows must state why nothing resolves them). Add a registration line or drop the claim.
- `fx-job/job/stubs/fx_svc_common_1_4.pyi` defines `fmt_money` etc. inside unit `job`; an engine indexing `.pyi` may find a same-unit provider for the skew keys. S30 owner (OI-3) to state the stub is a typing source only.
- fx-web `callTool` (wrapper, `client.ts:17`) and the SDK method `c.callTool(...)` share a name; `loadItem`/`loadOrders` (LNK-02/06) call the SDK method directly. Without installed types the by-name fallback may bind them to the wrapper and turn LNK-02 into a lifted-role row. Live check at P1 (TP-006 area); no edit.

### F-12 [nit]
`fx-lib/README.md` says to build with `mvn ...`: pricing has no protobuf plugin and quote-cli imports an absent `com.vendor.handles` jar, so Maven cannot build either (CPG build is tolerant; harmless but misleading). `fx-web/package.json` pins `@modelcontextprotocol/sdk` `1.0.0`, which I believe predates `client/streamableHttp.js` (inferred; pin a later version).

## 3. Rulings on the nine asks

| Ask | Ruling |
|---|---|
| (a) `order_report` | New row LNK-44 (F-1); qa-engineer plan edit, builders add manifest rows. |
| (b) DF-4 `discount.apply_discount`, CD-4 `getattr(_HANDLERS, evt)` | **Accept both.** The plan's `discount.apply(rate)` cannot `name-match` (no `apply` exists); `apply_discount` gives a candidate set containing the true target, and spec3 §7.2 `name-match` is `possible` whatever the candidate count. `publish_alert` is a module function (`self` does not exist); its `possible` comes from `bus.publish_alert` in `notify` (untyped `bus`), not from the `getattr`. Distances verified: DF-2/DF-4 api/2, CD-3/CD-4 bus/2; A25-3 pairs sort possible-first by name, path and line in all four. Plan text edit (qa-engineer). |
| (c) RV | F-7. |
| (d) fx-lib | F-9. |
| (e) LNK-02/12 | Accept: direct SDK call for LNK-02, wrapper for LNK-12, URL binding at `connect` (`localhost:8200/mcp`, matches `server.py` `--port` 8200 and `path="/mcp"`). Hazard in F-11. |
| (f) Compose placement | F-5. |
| (g) AB-1, LNK-42/43 | **Realized.** `legacy/` and `tools/` are units with their own `pyproject.toml`; `/legacy/fee`, `/tools/run` providers exist; `legacy_fee`/`run_tool` fetch them; `legacy/broken.py` is added at H2 and never reverted (svc manifest `buildFailure` H2 -> HEAD); `svc-tools` "never built"; H1 is an allow-empty build point (the "previous refresh"). |
| (h) bases | **Consistent.** Every base is origin-only (`VITE_API_BASE`, `VITE_JOB_BASE`, `VITE_REPORTS_BASE`, `FX_SVC_URL`, `FX_JOB_URL`, `base_url`) and every consumer appends the full path (`${BASE}/api/items/..`, `self.base + "/api/v1/orders"`, `${JOB_BASE}/job/status`). `BASE="/api"` in `run_report.py` is relative like the web `fetch("/api/..")` calls and matches the plan ("`BASE` ends `/api`"); not runnable, irrelevant to a static fixture. State the origin-only convention in the plan (S30 binding rule). |

Cross-repo constants (item 4): routes, MCP 8200 / `POST /mcp` (server default, `.mcp.json`, `refresh.sh`, `client.ts`), Compose `job` 8300 (`FX_JOB_URL` default, `VITE_JOB_BASE`), and `fx_svc_common` 2.1 signatures (`fmt_money(amount,*,locale)`, `round_money(amount,*,mode)`, `split_money(amount,parts,*,remainder)`, `fmt_date(ts)`) against the 1.4 stub and call sites at `render_row`, `requeue`, `fetch_item` all agree with LNK-33..35, 41. Pin: `requirements.txt` `fx-svc-common==1.4`.

## 4. What is solid
Census, gates, statuses and reasons match the plan row for row. History ops apply cleanly (all `replace_text` anchors unique). KN literals, SK files (including decoys and the `scripts/build.test.ts` outside the unit), AC6 shapes, TW twins, NC controls, FT boundary, MR casts, TF chain (TF-1 adds a line above the sink) and OV branches are present and located. Closed-world sweep of routes, tools, topics, tables, labels and env names across all four trees found every literal mapped to a row except `order_report` (F-1) and the `load_orders` column read (F-2). The A25-3 "possible sorts first" plants are real.

## 5. Open questions
1. F-5: which mapping rule does the engine use for Compose services without a build context (S30 owner)?
2. Does S33 resolve reverse questions by data dependency (F-7)?

No new `REVIEW.md` lines proposed.

## Pass 2 (2026-11-04)

**Verdict: approve with suggestions.** The trees may be accepted for S39. Condition: align the plan's LNK-24 P-locator with the fx-lib manifest (one line, below) before TP-002/TP-010 run. F-5 must be decided before P3 asserts LNK-15/OV-2.

CPG: considered, not relevant — fixtures are not a loaded graph; re-checked by execution.

**Executed.** Checkers re-run: job GREEN (27 rows), web GREEN (50), lib GREEN, svc GREEN (90 rows, 157 locators; scratchpad checker is now `s36b/check.py`). Whole-plan recount (own script, v1.3 text): 132 row IDs, equal to the manifest union (none missing or extra); LNK 44; gates P1 91 · P2 16 · P3 21 · P4 1 · P5 3, equal to the plan's §6 and its 120 positives (132 - 12). Status/reason/gate agree across all multi-manifest rows. LNK-44 agrees in job and svc manifests (key `sql:table:order_report`, P `write_report`, C `report_view`, confirmed, `key-match`, P3) and in plan rows LNK-44, OV-1, §7. Python parses; `__pycache__` count 0. Java/TS sources are unchanged apart from `package.json`/README, so the Pass 1 `javac`/`tsc` results stand (I could not re-run javac: my stub dir was cleaned from the scratchpad). Same-revision fixes against each other: the F-2 query (`SELECT count(*)`), the F-3 origin (`http://api:8000`), the unrolled `app.py` and LNK-44 do not collide; no manifest carries a line number for `app.py`, `loader.py` or `run_report.py`.

| Finding | Disposition |
|---|---|
| F-1 | Fixed. LNK-44 in plan, both manifests; census 132 re-verified. |
| F-2 | Fixed. `run_report.py:43` is `SELECT count(*) FROM orders`; plan B1S-R and manifest say so. |
| F-3 | Fixed. `config/job.yaml` is `http://api:8000`; plan LNK-14 and manifest state it; checker GREEN. |
| F-4 | Fixed for LNK-08/13 (keys now agree); skew-key form (LNK-34, 35, 41) still differs in form: carried to the S39 schema. |
| F-5 | **Carried** (architect/S30: Compose `web`/`job` service-to-unit rule). |
| F-6 | **Carried** (S39 schema: one tree-state convention). |
| F-7 | Fixed. Plan §5.5 states data dependency, functions, distance <= 3. |
| F-8 | **Carried** (S39 TP-018 must key on (literal, locator)). |
| F-9 | Partly. Lib manifest LNK-24 now has P = `Pricing.quote(int,String,Currency)` (line 27) with `FxBus.publish` as wrapper, but **plan row LNK-24 (line 163) still names `FxBus.publish` as P**: manifest and plan disagree, TP-002 would flag it. Edit the plan P-locator (`qa-engineer`). AC6-TOP-3 "none knowable" and the same-unit-calls sentence (plan §1) are in. |
| F-10 | Fixed. Seven explicit `include_router` lines in `api/app.py`, same prefix and routers as the loop; the change matches the finding and breaks no locator (no line references). |
| F-11 | Partly. B1 containment wording, the `.pyi`-is-typing-only note and the `callTool` live check are not in the plan (grep finds none); minor, fold into the next plan revision. **LNK-39 ruling:** add one registration line to `fx-svc/plugins/loader.py` (`REGISTRY["audit"] = AuditPlugin`, `MailPlugin` left unregistered) so the true target is derivable from the source; do not drop the claim (TP-019 would lose a row). Risk to check live at P1: a dict-value flow analysis could narrow `REGISTRY.get(kind)` to one target and make the row `confirmed`; if so, register behind a parameterized function instead. Owner: fx-svc builder. |
| F-12 | Fixed. README now says parse-only. SDK pin `1.12.0`: verified, Streamable HTTP first shipped in 1.10.0 ([release notes](https://github.com/modelcontextprotocol/typescript-sdk/releases/tag/1.10.0)), so 1.12.0 includes `client/streamableHttp.js` (import path from a third-party source). |

Growth-stops: plan v1.3 revision note is one dated line; additions are new scope (LNK-44) or in-place rework; no restated history found.
