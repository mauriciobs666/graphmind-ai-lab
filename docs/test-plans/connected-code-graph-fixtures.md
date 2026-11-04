# Connected code graph — fixture system test plan (S35)

> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** C-1001 (M10) · **Extends:** `docs/plans/connected-code-graph-p0-2.md` · **Version:** 1.3 · **Reviews:** `docs/reviews/connected-code-graph-fixtures.md`, `docs/reviews/connected-code-graph-fixtures-ml.md`

*2026-11-04 (v1.3): fixtures-gate fixes F-1 (LNK-44 `order_report`, census 132 rows), F-2 (B1S-R query), F-3 (LNK-14 origin), F-7 (reverse model), F-9/F-11 (AC6-TOP-3 target, same-unit calls, origin-only bases).*

## 1. Scope and objective
S35 fixes, as numbered tables, every item planted in the four-repository fixture system
(`fx-svc`, `fx-web`, `fx-lib`, `fx-job`) at or above `docs/plans/connected-code-graph-eval-protocol-ml.md`
§12.2's minimums (derived in §6). The fixture steps (S36 `fx-svc`, S37 `fx-web`, S38 `fx-lib`, and
S35b `fx-job`, parallel to them) build repositories that match these rows; S39 materializes them as git
repositories and tests this plan's §9. A planted row is the contract: a repository that lacks the row, or a
system that disagrees with the row's expected status or reason, is a defect in one or the other.

Fixture items are **dev** data, never AC-5 held-out, and are reported as their own stratum, never pooled with
proving-repo items (eval-protocol §12.2, F7). AC-6 passes per kind when every instance is `possible` with the
expected reason and **none is confirmed**.

Two limits, stated once. Fixture pass rates are existence checks on hand-written, regular rows: they are **not**
recall or precision estimates and are never extrapolated (three instances give a Wilson lower bound of 0.44 for
3/3). And each manifest is **closed-world** for cross-boundary links: every `Link` (and every provider/consumer pair it
joins) the engine emits for a fixture repository must match a row here, so an unplanted emitted link fails TP-010.
Calls inside one unit (for example `PricingService.quote` to the four-argument `Pricing.quote`, or `Pricing.quote` to `FxBus.publish`) are `CALLS` edges and never produce a `Link`. A consumer with no provider (NC-3, NC-4, NC-7) emits a `CONSUMES` edge to a contract point and **no `Link`**; that is
allowed and is not an unplanted link. TP-018 checks the planter side, by grep. Distances and statuses come from the engine's own hop model
(OI-4), so A25-1 certifies conformance to that model, not that the model is right.

## 2. References
- `docs/plans/connected-code-graph-p0-2.md`: S35 row; §4.7 (fixture system); `docs/plans/connected-code-graph-coordination.md`
  (2026-11-01 entry: `fx-job`, SQ-C split, K4 hedge, new AC-6 row; "keyNorm step-order ruling").
- `docs/plans/connected-code-graph-eval-protocol-ml.md` §12.2 (minimums, AC-25 presentation properties, m6) and
  §4.1-shape list; `docs/plans/connected-code-graph-eval-protocol2-ml.md` (keyNorm oracle K42-K48).
- `docs/plans/connected-code-graph-bundle-spec3.md` §3.5 (locators), §6 (`keyNorm` per kind), §7 (reason
  codes); `docs/plans/connected-code-graph-bundle-spec4.md` §1 (`header.skippedFiles`, closed test pattern).
- `docs/requirements/connected-code-graph.md`: AC-6, AC-9b, AC-19, AC-21a, AC-25. `docs/plans/connected-code-graph.md`
  §5.1 (the proposal's integration list), §3.4 (binding), P3-1, SQ-C, P2-7.
- Not yet written: S33's answer schema (`docs/plans/connected-code-graph-answer-spec.md`). The AC-25 assertions
  below name the properties; S39 binds each to S33's field names (open item OI-1).

## 3. Table conventions
**Row columns** (every table in §5): `ID` · `kind` (contract kind of bundle spec3 §6, or the item's own type) ·
`planted item` (what is written, in which syntactic shape) · `locators` (`P` provider/source, `C` consumer/sink) ·
`st` (status) · `reason` (bundle spec3 §7 code; `a / b` = edge reason / link reason) · `gate` (earliest phase at
which the row may be asserted) · `shape` (eval-protocol §12.2 difficulty tag, where the row carries one) · `serves`.

**Locators (bundle spec3 §3.5, §1.9).** A code element is `repoId:path::qname`, with `path` **repo-relative**
(no repository prefix, no `./`) and `qname` starting `path + "::"`; Java adds the §4.3 signature in parentheses
after the `qname`: `fx-lib:pricing/src/main/java/com/acme/fxlib/Pricing.java::Pricing.quote(int,String,Currency)`.
A configuration entry is `repoId:path @keyPath` (§1.9); a unit is `unit repoId:rootPath lang` (§4's table); a
contract point is `{kind, keyNorm}`. Repository shorthands are `svc` = `fx-svc`, `web` = `fx-web`, `lib` =
`fx-lib`, `job` = `fx-job`; a bare path in a row is under the named repository, so `svc:api/health.py::health` is
`{repoId fx-svc, path api/health.py, qname api/health.py::health}`, and `web:app/src/ui/Health.tsx::Health` carries
its `app/` directory in the path. A `.proto` or `.sql` file has no `lang` token (§3.3), so its row is anchored on
the code that implements or reads it, and the definition itself is a `schemaEntry` (`objectKey` = the `keyNorm`).

**Assertion ids** (what a row serves):
- `AC-6`: possible, with the stated reason, **and not confirmed**. `AC-9b`, `AC-19`, `SQ-C`, `P2-7`, `P3-1`: as
  in §4.7. `LNK`: the row's status and reason are emitted exactly (§4.7 "every link kind and status").
- `B1-1..4` (B1's four assertions): 1 change to writer W lists reader R; 2 RCA from R lists W; 3 renaming the
  entity lists R and W; 4 W does not list another writer W2.
- `A25-1..4` (AC-25's four presentation properties, eval-protocol §12.2 m6), asserted against S33's answer
  fields (OI-1). Forward item (a planted effect of a change): **A25-1** `direct` iff distance 1, else `indirect`,
  with the expected distance; **A25-2** the item sits in its named group (repository, component, distance), and
  groups are ordered **by repository, then component (byte order of the name), then distance ascending**;
  **A25-3** inside a group, every `confirmed` item precedes every `possible` one; **A25-4** the reverse question
  asked of the changed function lists its planted dependencies (its `RV-` row), counted once per `RV-` row, not once
  per forward item. For a **reverse** row, A25-1..3 apply to each listed dependency. Distinct assertions: 8 forward
  items x 3 = 24, plus 11 dependency entries x 3 = 33, plus A25-4 x 3 (RV-1..3) = **60**. Where A25-3 is
  non-vacuous the group holds both statuses, and the `possible` item is planted to sort **before** the
  `confirmed` one under name order, path order and (same file) line order (§5.5), so an answer sorted by
  (repository, component, distance, name) with no status key turns the assertion red.
- `KN`: the key normalizes as the row states (RFC 3986 order, F-13).
- `SK`: the `header.skippedFiles` outcome of bundle spec4 §1. `FT`: the field-depth cap of spec3 §5.1.

**Phases.** `P1` Python/TS/shell/config extractors, http, mcp_tool, launch, AC-6 P1 kinds. `P2` Java, rpc, topic,
version-skew `library_api` linking, topic-from-config. `P3` data_entity, config_key, env_var, query-built-at-runtime, the multi-repo
overlay and cross-repository AC-9b. `P4` verdicts (SQ-C Python half). `P5` taint (P2-7). `fx-lib` rows gate **only
at P2**; every row that needs Java is `P2` and none is promoted earlier.

**Base URLs are origin-only.** Every base-URL value (`VITE_API_BASE`, `VITE_JOB_BASE`, `VITE_REPORTS_BASE`, `FX_SVC_URL`, `FX_JOB_URL`, `base_url`) holds scheme, host and port and no path; each consumer appends the full path.

**Keys follow RFC 3986 order (F-13).** A fixture key is written as the **original** literal in source and as the
`keyNorm` after one pass: decode (step 5), then resolve dots (step 6), then mask (step 8a). Keys with escapes
are in §5.2; every other key in this plan contains no `%`.

**Provenance.** Rows modelled on a proving-repo shape (so a later disagreement is attributable; protocol §8
contacts anchors, not shapes): LNK-18 and LNK-21 (the `.mcp.json` / 8000-8200 chain), SK-5 (the
`salesperson/src/i18n/` case).

## 4. Repositories and units

| Repo | Language | Components |
|---|---|---|
| `fx-svc` | Python (FastAPI, FastMCP, SQL/Cypher, Kafka producer), shell, Compose, `.mcp.json`, Markdown prompts | `api`, `mcp`, `core`, `db`, `bus`, `state`, `clients`, `plugins`, `scripts`, `deploy`, `prompts`, `common` |
| `fx-web` | TS (fetch wrapper, MCP client config, Kafka consumer, React UI) | `api`, `mcp`, `bus`, `ui`, `plugins` |
| `fx-lib` | Java (shared library with overloads, the in-house bus, a gRPC `.proto` and service, a consumer) | `pricing`, `bus` (both in unit `pricing`), `cli` (unit `quote-cli`) |
| `fx-job` | Python and config | `job` |

**Units** (bundle spec3 §1.9; `lang` is the §3.3 token). A file belongs to the **innermost** unit whose `rootPath`
contains it; config, shell and Markdown files are records of that unit (their source list is unused in v1, spec4
§1). Nesting (`fx-svc`'s root unit holds `common`, `legacy`, `tools`) is an assumption: OI-8.

| unitId (name) | repoId | rootPath | lang | vendoredDirs | gates |
|---|---|---|---|---|---|
| `svc` | `fx-svc` | `.` | `python` | none | P1 |
| `svc-common` | `fx-svc` | `common/` | `python` | none | P2 (library unit) |
| `svc-legacy` | `fx-svc` | `legacy/` | `python` | none | P1 (unbuildable, AB-1) |
| `svc-tools` | `fx-svc` | `tools/` | `python` | none | P1 (never built, AB-2) |
| `web` | `fx-web` | `app/` | `javascript` | `vendor` | P1 |
| `lib-pricing` | `fx-lib` | `pricing/` | `java` | none | **P2 only** |
| `lib-cli` | `fx-lib` | `quote-cli/` | `java` | none | **P2 only** |
| `job` | `fx-job` | `.` | `python` | none | P1 (P3 for AC-19, AC-9b) |

**Build states in `SC-STEADY`** (a fresh build at H4): `svc-tools` is **never built**; `svc-legacy` is
**unbuildable** (its forced `buildFailure` marker stays from H2 to HEAD) and has no earlier refresh, so it is also
not mapped; every other unit builds (`lib-*` from P2). The links into the two units are LNK-42 and LNK-43.

`fx-job` is the 4th repository (ledger 2026-11-01), built by step **S35b**, parallel to S36-S38: an HTTP call into an
`fx-svc` endpoint, a read of an `fx-svc` table and env var, HTTP providers (`/job/*`) that `fx-web` consumes, a table
it writes that `fx-svc` reads, the cross-repository version-skew calls into `svc-common`, and the AC-19 un-merged
change. Its manifest is `cpg/fixtures/system1/expected/fx-job.yaml`; the tree is `cpg/fixtures/system1/fx-job/**`.
S39 materializes **four** repositories under `cpg/.ccg-work/fixtures/`. The AC-19 overlay uses `fx-svc`, `fx-web`
and `fx-job`; there is no fifth repository.

## 5. Planted items
### 5.1 Link rows (`LNK`)

`P` is the provider/source and `C` the consumer. A `shape` tag counts toward eval-protocol §12.2's per-shape
minimum (§6). `st` is `confirmed` (c) or `possible` (p). **Every interpolated part in a `confirmed` template or
constant row is itself a constant visible in source** (a module constant, a literal), so the consumer's key is a
literal path such as `GET /api/files/readme.txt`, which matches the provider's `{}` segment as `key-match`; a
value that is a parameter or unresolved makes the key `{?}` and the link `possible` / `partial-key` (spec3 §6
step 8), and LNK-17 is that case. For every `possible` row the "true target" named is a candidate the engine's
answer must contain.

| ID | kind | planted item | locators | st | reason | gate | shape | serves |
|---|---|---|---|---|---|---|---|---|
| LNK-01 | http | `GET /api/health`, literal in the consumer | P `svc:api/health.py::health` · C `web:app/src/ui/Health.tsx::Health` | c | key-match | P1 | exact literal | LNK |
| LNK-02 | mcp_tool | `get_item`, literal `callTool("get_item")`; server picked by URL binding | P `svc:mcp/server.py::get_item` · C `web:app/src/mcp/client.ts::loadItem` | c | key-match | P1 | exact literal | LNK |
| LNK-03 | env_var | `FX_SVC_URL`, literal `os.environ["FX_SVC_URL"]`; Compose sets it for service `job` | P `svc:deploy/compose.yaml @services.job.environment.FX_SVC_URL` · C `job:job/api_client.py::ApiClient.__init__` | c | key-match | P3 | exact literal | LNK |
| LNK-36 | http | `POST /api/v1/orders`, literal `fetch("/api/v1/orders", {method: "POST"})` (also the AC-25 link of DF-3) | P `svc:api/orders.py::create_order` · C `web:app/src/ui/OrderTotal.tsx::OrderTotal` | c | key-match | P1 | exact literal | LNK, DF-3 |
| LNK-04 | http | `GET /api/v1/orders/{}`: path from constants `ORDER_PATH` and `SAMPLE_ORDER` (key `GET /api/v1/orders/1001`); provider is a router `prefix="/v1/orders"` mounted at `/api` (mount composition) | P `svc:api/orders.py::read_order` · C `web:app/src/api/orders.ts::getOrder` (constants in `web:app/src/api/endpoints.ts`) | c | key-match | P1 | constant across files | LNK |
| LNK-05 | topic | `kafka:orders.created`, name in `TOPIC_ORDERS` imported from `bus/constants.py` | P `svc:bus/producer.py::publish_created` · C `web:app/src/bus/consumer.ts::onOrderCreated` | c | key-match | P2 | constant across files | LNK |
| LNK-06 | mcp_tool | `list_orders`, name held in a constant module on each side | P `svc:mcp/server.py::list_orders` (const `svc:mcp/names.py`) · C `web:app/src/mcp/client.ts::loadOrders` (const `web:app/src/mcp/names.ts`) | c | key-match | P1 | constant across files | LNK |
| LNK-37 | http | `GET /api/search`, Python path in `job/paths.py::SEARCH_PATH` | P `svc:api/search.py::search` · C `job:job/run_report.py::search_reports` | c | key-match | P1 | constant across files | LNK |
| LNK-07 | http | `GET /api/files/{}`, TS template `` `${API_PREFIX}/files/${README}` ``, `README = "readme.txt"` constant (key `GET /api/files/readme.txt`) | P `svc:api/files.py::read_file` · C `web:app/src/api/files.ts::fetchFile` | c | key-match | P1 | template literal | LNK |
| LNK-08 | http | `GET /api/items/{}`, Python f-string `f"{BASE}/items/{ITEM_ID}"`; `BASE` (ends `/api`) and `ITEM_ID = 42` are module constants | P `svc:api/items.py::read_item` (router `prefix="/items"` at `/api`) · C `job:job/run_report.py::fetch_item` | c | key-match | P1 | template literal | LNK |
| LNK-38 | http | `GET /api/items/{}`, TS template `` `${API_PREFIX}/items/${RELATED_ID}` `` with `RELATED_ID = 7` constant | P `svc:api/items.py::read_item` · C `web:app/src/ui/Related.tsx::Related` | c | key-match | P1 | template literal | LNK |
| LNK-09 | topic | `kafka:orders.shipped`, name `f"{PREFIX}.shipped"` with a constant prefix | P `svc:bus/producer.py::publish_shipped` · C `web:app/src/bus/consumer.ts::onOrderShipped` | c | key-match | P2 | template literal | LNK |
| LNK-10 | http | `GET /api/search`, TS wrapper `request()` reached by a two-level field path `api.search.query` (AC-1 lifting chain) | P `svc:api/search.py::search` · C `web:app/src/api/index.ts::api.search.query` (wrapper `web:app/src/api/client.ts::request`) | c | lifted-role | P1 | client wrapper | LNK |
| LNK-11 | http | `POST /api/v1/orders`, Python wrapper `ApiClient.post` | P `svc:api/orders.py::create_order` · C `job:job/run_report.py::requeue` (wrapper `job:job/api_client.py::ApiClient.post`) | c | lifted-role | P1 | client wrapper | LNK |
| LNK-12 | mcp_tool | `reindex`, TS `callTool(name)` wrapper | P `svc:mcp/server.py::reindex` · C `web:app/src/ui/Admin.tsx::onReindex` (wrapper `web:app/src/mcp/client.ts::callTool`) | c | lifted-role | P1 | client wrapper | LNK |
| LNK-13 | http | `GET /api/items/{}` (constant `DEFAULT_ITEM = 1`); base URL is `VITE_API_BASE`, set by Compose service `web` to service `api` (Env/Compose binding) | P `svc:api/items.py::read_item` · C `web:app/src/api/items.ts::getDefaultItem` | c | binding-confirmed | P1 | config-bound base URL | LNK |
| LNK-14 | http | `GET /api/health`; base URL `base_url` in `config/job.yaml` (origin `http://api:8000`, the Compose service `api` of LNK-22, so it binds to unit `svc` alone), read by `job/settings.py` | P `svc:api/health.py::health` · C `job:job/run_report.py::preflight` | c | binding-confirmed | P1 | config-bound base URL | LNK |
| LNK-15 | http | `POST /job/trigger`; base URL `os.environ.get("FX_JOB_URL", "http://localhost:8300")`; fx-job's Compose port is 8300 (`job` Compose service: build context `../../fx-job`, port 8300, unit `job`) | P `job:job/trigger_server.py::trigger` · C `svc:clients/jobs.py::trigger_job` | c | binding-confirmed | P1 | config-bound base URL | LNK |
| LNK-16 | http | `GET /job/status` fetched from `http://localhost:8000`; `fx-svc` (Compose) **and** `fx-job` (`scripts/dev.sh`) both claim 8000. True target: `job:job/trigger_server.py::status` | P `job:job/trigger_server.py::status` · C `web:app/src/ui/JobStatus.tsx::JobStatus` | p | port-shared | P1 | | LNK |
| LNK-17 | http | `` fetch(`/api/items/${id}`) `` where `id` is a component prop: key `GET /api/items/{?}`. True target: `read_item` | P `svc:api/items.py::read_item` · C `web:app/src/ui/Item.tsx::ItemView` | p | partial-key | P1 | | LNK |
| LNK-18 | http | URL binding: `http://localhost:8200/mcp` (consumer); ports 8000 and 8200 are launch configs of **one** unit (`svc`: `.mcp.json` and Compose) | P `svc:mcp/server.py::main` (`POST /mcp`) · C `web:app/src/mcp/client.ts::connect` | c | binding-confirmed | P1 | | LNK |
| LNK-19 | script_tool | `scripts/refresh.sh` sets `MCP_URL=http://localhost:8200/mcp` (bound to unit `svc`) and runs `mcp-call reindex` (literal tool name) | P `svc:mcp/server.py::reindex` · C `svc:scripts/refresh.sh::<script>` | c | key-match | P1 | | LNK |
| LNK-20 | script_tool | `scripts/dyn.sh` runs `eval "mcp-call $TOOL"`, no binding. By the `unknown-key` rule (one link per provider of the kind, spec3 §7.2) the candidates are **every** `mcp_tool` provider in the system, including `fx-job`'s second `reindex` (NC-5). True targets: any | P `svc:mcp/server.py::get_item`, `svc:mcp/server.py::reindex`, `svc:mcp/server.py::list_orders`, `job:job/mcp_admin.py::reindex` · C `svc:scripts/dyn.sh::<script>` | p | shell-dynamic | P1 | | LNK |
| LNK-21 | launch | `.mcp.json` server `fx-svc` to `scripts/run-mcp.sh` to `python -m fx_svc.mcp.server` to unit `svc` | P `unit fx-svc:. python` · C `svc:.mcp.json @mcpServers.fx-svc` | c | binding-confirmed | P1 | | LNK |
| LNK-22 | launch | Compose service `api`: build context to unit `svc`, port 8000 | P `unit fx-svc:. python` · C `svc:deploy/compose.yaml @services.api` | c | binding-confirmed | P1 | | LNK |
| LNK-23 | rpc | `/acme.fxlib.Pricing/Quote`: `.proto` service, Java impl, Python stub call with a literal method | P `fx-lib:pricing/src/main/java/com/acme/fxlib/PricingService.java::PricingService.quote(QuoteRequest,StreamObserver<QuoteReply>)` (declared in `pricing/src/main/proto/pricing.proto`) · C `svc:clients/pricing.py::quote` | c | key-match | P2 | | LNK |
| LNK-24 | topic | `inhouse-fxbus:price.updated`: `FxBus.publish("price.updated", e)` in `Pricing.quote` and a listener | P `fx-lib:pricing/src/main/java/com/acme/fxlib/Pricing.java::Pricing.quote(int,String,Currency)` (publish call `fx-lib:pricing/src/main/java/com/acme/fxlib/bus/FxBus.java::FxBus.publish(String,Object)` as the wrapper) · C `fx-lib:pricing/src/main/java/com/acme/fxlib/bus/AuditListener.java::AuditListener.onPrice(PriceEvent)` | c | key-match | P2 | | LNK |
| LNK-25 | library_api | `java:com.acme.fxlib.Pricing.quote(int,String,Currency)`, consumer `quote-cli` on the same version; `Pricing` also has the distinct overload `quote(int,String,Currency,Locale)` | P `fx-lib:pricing/src/main/java/com/acme/fxlib/Pricing.java::Pricing.quote(int,String,Currency)` · C `fx-lib:quote-cli/src/main/java/com/acme/fxlib/cli/QuoteCli.java::QuoteCli.quoteDefault()` | c | key-match | P2 | | LNK |
| LNK-26 | data_entity | `sql:column:orders.total`: written by `INSERT`, read by `SELECT total` in another repository | P `svc:db/orders.py::insert_order` · C `job:job/run_report.py::render_row` | c | key-match | P3 | | LNK |
| LNK-27 | data_entity | `cypher:label:Order`: `CREATE (:Order {id: 1})` and `MATCH (o:Order)` | P `svc:db/graph.py::link_order` · C `svc:db/graph.py::neighbors` | c | key-match | P3 | | LNK |
| LNK-28 | config_key | `pydantic-settings:db.pool_size`: `db.pool_size` in `config/app.yaml`, read through `Settings.db_pool_size` | P `svc:config/app.yaml @db.pool_size` · C `svc:db/orders.py::connect` | c | key-match | P3 | | LNK |
| LNK-29 | config_key | `settings.get(section + "." + key)` with computed parts; no provable provider | C `job:job/settings.py::lookup` | p | runtime-built | P3 | runtime-built key | LNK |
| LNK-30 | call | `repo.save(o)` where `repo: Repo = Depends(get_repo)`; `SqlRepo.save` and `MemRepo.save` both exist. True target: `SqlRepo.save`, both in the candidate set | C `svc:api/orders.py::create_order` | p | name-match | P1 | DI or reflection | LNK |
| LNK-39 | call | service locator: `registry.get(kind).run(evt)`; `AuditPlugin.run` and `MailPlugin.run` exist. True target: `AuditPlugin.run` | C `svc:plugins/loader.py::dispatch` | p | name-match | P1 | DI or reflection | LNK |
| LNK-40 | call | `store.load(item_id)` with `store = Depends(get_store)`; `DbStore.load` and `CacheStore.load` exist. True target: `DbStore.load` | C `svc:api/items.py::read_item` | p | name-match | P1 | DI or reflection | LNK |
| LNK-31 | call | Java reflection `Class.forName(cfg).getMethod("quote")`. True target: `Pricing.quote(int,String,Currency)` | C `fx-lib:quote-cli/src/main/java/com/acme/fxlib/cli/QuoteCli.java::QuoteCli.main(String[])` | p | name-match | P2 | DI or reflection | LNK |
| LNK-32 | rpc | `getattr(stub, method_name)(req)`: method name is a parameter. True target: `/acme.fxlib.Pricing/Quote` | C `svc:clients/pricing.py::quote_dynamic` | p | runtime-built / unknown-key | P2 | DI or reflection | LNK |
| LNK-33 | library_api | **skew:** `fx-job` pins `fx-svc-common==1.4` and is typed against `job/stubs/fx_svc_common_1_4.pyi`; it calls `fmt_money(amount,locale)`; provider 2.1 is `fmt_money(amount,*,locale)`. Key `python:fx_svc_common.fmt.fmt_money(amount,*,locale)` | P `svc:common/fx_svc_common/fmt.py::fmt_money` · C `job:job/run_report.py::render_row` | p | name-only-provider (`versionSkew`) | P2 | version skew | LNK |
| LNK-34 | library_api | **skew:** `round_money(amount)` in 1.4, `round_money(amount,*,mode)` in 2.1 | P `svc:common/fx_svc_common/fmt.py::round_money` · C `job:job/run_report.py::requeue` | p | name-only-provider (`versionSkew`) | P2 | version skew | LNK |
| LNK-35 | library_api | **skew:** `split_money(amount,parts)` in 1.4, `split_money(amount,parts,*,remainder)` in 2.1 | P `svc:common/fx_svc_common/split.py::split_money` · C `job:job/run_report.py::fetch_item` | p | name-only-provider (`versionSkew`) | P2 | version skew | LNK |
| LNK-41 | library_api | **skew control:** `fmt_date(ts)` has the same signature in 1.4 and 2.1; the `versionSkew` note is present on the link | P `svc:common/fx_svc_common/fmt.py::fmt_date` · C `job:job/run_report.py::render_row` | c | key-match (`versionSkew` note) | P2 | | LNK |
| LNK-42 | http | `GET /legacy/fee` fetched by `legacy_fee`; `svc-legacy` is unbuildable and never built in the steady state | P `svc:legacy/app.py::fee` · C `svc:api/orders.py::legacy_fee` | p | target-not-mapped | P1 | | LNK |
| LNK-43 | http | `GET /tools/run` fetched by `run_tool`; `svc-tools` is never built | P `svc:tools/app.py::run` · C `svc:api/orders.py::run_tool` | p | target-not-mapped | P1 | | LNK |
| LNK-44 | data_entity | `sql:table:order_report`: written by `INSERT INTO order_report` in `fx-job`, read by `SELECT * FROM order_report` (no column key, so no column link); no DDL in any repository | P `job:job/run_report.py::write_report` · C `svc:api/reports.py::report_view` | c | key-match | P3 | | LNK, OV-1 |

A Compose build context that leaves its own repository and resolves to a sibling repository's unit (`../../fx-job`, LNK-15, OV-2) is a P1 check owed to the S30 owner.

Reasons for the skew rows (LNK-33..35, 41), LNK-32 and LNK-31 are this plan's decisions (OI-3): bundle spec3 §7.2
has no skew code, and `versionSkew` is a property on the link (graph note `Link.versionSkew`). The skew shape is
**cross-repository by construction**: consumer in `fx-job`, library in `fx-svc`.

### 5.2 Key normalization rows (`KN`, F-13)

Each is a consumer literal in `web` whose key is right only if decoding precedes dot resolution (KN-1, KN-2) and
precedes masking (KN-5). Gate P1; they serve `KN` and `LNK`. Expected keys were derived by hand from spec3 §6
(RFC 3986 order) and the K42/K43 oracle rows (`docs/plans/connected-code-graph-eval-protocol2-ml.md`), then run
through `cpg/ccg/keynorm.py::mask_http_segments` composed with a short decode/resolve harness
(`cpg/tests/test_fixture_keynorm_mutants.py`).

| ID | consumer literal (as written) | expected `keyNorm` | outcome | provider | locator C |
|---|---|---|---|---|---|
| KN-1 | `/api/v1/%2E%2E/health` | `GET /api/health` (decode `%2E%2E`, then resolve; the K42 shape) | c, key-match | `svc:api/health.py::health` | `web:app/src/ui/Probe.tsx::probeEncoded` |
| KN-2 | `/api/items/%2E%2E/%2E%2E/api/health` | `GET /api/health` (two encoded dot segments) | c, key-match | `svc:api/health.py::health` | `web:app/src/ui/Probe.tsx::probeDeep` |
| KN-3 | `/api/%68ealth` | `GET /api/health` (`%68` is unreserved, decoded) | c, key-match | `svc:api/health.py::health` | `web:app/src/ui/Probe.tsx::probeUnreserved` |
| KN-4 | `/api/files/a%2fb` | `GET /api/files/a%2Fb` (reserved escape kept, hex uppercased; one segment, matches `{}`) | c, key-match | `svc:api/files.py::read_file` | `web:app/src/ui/Probe.tsx::probeReserved` |
| KN-5 | `/api/hooks/x7k2%2Dq9w4%2Dm1p8%2Dz3v6` | `GET /api/hooks/{?}` (decoded `x7k2-q9w4-m1p8-z3v6` fails the token screen and is masked; the encoded form passes it) | p, partial-key | `svc:api/hooks.py::receive_hook` (`GET /api/hooks/{}`) | `web:app/src/ui/Probe.tsx::probeToken` |

**Mutants actually run** (same harness, same five literals): *resolve-before-decode* leaves `GET /api/v1/../health`
and `GET /api/items/../../api/health`, so it fails **KN-1 and KN-2**; *mask-before-decode* leaves the secret-shaped
`x7k2-q9w4-m1p8-z3v6` in the key (a leak), so it fails **KN-5** only; the correct order passes all five. KN-3 and
KN-4 are order-insensitive controls. K48 (a non-fixed point) is a unit case, not planted.

### 5.3 AC-6 unprovable kinds (`AC6-*`)

Every row is `possible`, with the stated reason, and **must not be confirmed** (the second clause is the real
test). Three rows per kind, in three distinct syntactic shapes (the `shape` column). Each row says **why** nothing
in the repository can resolve it, so the expectation is a ground truth and not a policy, and names the true
target where one exists (the answer's candidate set must contain it). Value-set `unknown` reasons (spec3 §7.3)
are given so the extractor's side is checkable too.

| ID | kind | syntactic shape | why unprovable; true target | locators | st | reason (edge / link) | gate | tag | serves |
|---|---|---|---|---|---|---|---|---|---|
| AC6-URL-1 | http | Python f-string with a call result: `f"{get_base()}/{ROUTES[name]}"`; value set `runtime-built` | `get_base()` reads a runtime registry; `name` is a parameter. True: `GET /api/search` | C `svc:clients/jobs.py::call_named` | p | runtime-built / unknown-key | P1 | runtime-built key | AC-6 |
| AC6-URL-2 | http | TS parameter-derived: `go(path)` does `fetch(base + path)`; value set `point-derived`; `base` bound by Compose | the path is a parameter of every caller. True: `GET /api/items/{}` | C `web:app/src/api/dyn.ts::go` | p | runtime-built / binding-narrowed | P1 | runtime-built key | AC-6 |
| AC6-URL-3 | http | TS spread: a rest-parameter spread passed to `fetch`; value set `spread-argument` | the arguments come from a rest parameter. True: none knowable | C `web:app/src/api/dyn.ts::forward` | p | runtime-built / unknown-key | P1 | runtime-built key | AC-6 |
| AC6-DYN-1 | library_api | Python `importlib.import_module(name)` | `name` comes from a config value absent from the repository. True: `svc:plugins/audit.py` | C `svc:plugins/loader.py::load_plugin` | p | dynamic-import | P1 | dynamic import | AC-6 |
| AC6-DYN-2 | library_api | Python `__import__("plugins." + name)` | the suffix is a parameter | C `svc:plugins/loader.py::load_legacy` | p | dynamic-import | P1 | dynamic import | AC-6 |
| AC6-DYN-3 | library_api | TS ``await import(`./plugins/${name}`)`` | `name` is a prop | C `web:app/src/plugins/load.ts::loadPlugin` | p | dynamic-import | P1 | dynamic import | AC-6 |
| AC6-EX-1 | launch | `.mcp.example.json` server entry for `fx-svc` (`File.role = example`) | the file is an example, not the deployed config. True: unit `svc` | C `svc:.mcp.example.json @mcpServers.fx-svc` | p | example-config | P1 | | AC-6 |
| AC6-EX-2 | http | `VITE_REPORTS_BASE` set **only** in `deploy/compose.example.yaml`; consumer of `GET /api/reports/{}` | the only binding is an example config. True: `svc:api/reports.py::report_view` | P `svc:api/reports.py::report_view` · C `web:app/src/ui/Reports.tsx::Reports` | p | example-config | P1 | | AC-6 |
| AC6-EX-3 | env_var | `FX_DB_URL` set **only** in `svc:.env.example`, read in `db/orders.py::connect` | the only writer is an example file. **Gate P3**: `env_var` is a P3 kind | C `svc:db/orders.py::connect` | p | example-config | P3 | | AC-6 |
| AC6-PM-1 | mcp_tool | Markdown **prose**: "call the `get_item` tool" in `prompts/agent.md` | a prompt names a tool; it does not call it. True: `get_item` | C `svc:prompts/agent.md::<document>` | p | prompt-mention | P1 | prompt mention | AC-6 |
| AC6-PM-2 | mcp_tool | Markdown **table cell**: `| reindex | rebuilds the index |` in `prompts/tools.md` | same | C `svc:prompts/tools.md::<document>` | p | prompt-mention | P1 | prompt mention | AC-6 |
| AC6-PM-3 | mcp_tool | Python triple-quoted `SYSTEM_PROMPT` naming `list_orders` | same | C `svc:prompts/system.py::<module>` | p | prompt-mention | P1 | prompt mention | AC-6 |
| AC6-TOP-1 | topic | name from an env var `os.environ["ORDERS_TOPIC"]`, set in no file of the repository | the value is absent. True: `kafka:orders.created` | C `svc:bus/producer.py::publish_dynamic` | p | config-derived | P2 | | AC-6 |
| AC6-TOP-2 | topic | name from a config-file key `config.bus.topic` (JSON); the key is in no file of the repository | the key is not set. True: `kafka:orders.created` | C `web:app/src/bus/consumer.ts::onConfigured` | p | config-derived | P2 | | AC-6 |
| AC6-TOP-3 | topic | name from a Java system property `System.getProperty("bus.topic")` | set at launch outside the repository. True: none knowable | C `fx-lib:pricing/src/main/java/com/acme/fxlib/bus/FxBus.java::FxBus.subscribeConfigured(String)` | p | config-derived | P2 | | AC-6 |
| AC6-QL-1 | data_entity | SQL table from an f-string: `f"SELECT count(*) FROM {table}"` | `table` is a parameter. True: `sql:table:orders` | C `svc:db/dynamic.py::count_rows` | p | runtime-built | P3 | | AC-6 |
| AC6-QL-2 | data_entity | Cypher label by `%`-format: `"MATCH (n:%s)" % label` | `label` is a parameter. True: `cypher:label:Order` | C `svc:db/dynamic.py::match_label` | p | runtime-built | P3 | | AC-6 |
| AC6-QL-3 | data_entity | SQL by concatenation: `"SELECT " + cols + " FROM " + tbl` | both are parameters | C `job:job/dynq.py::dump_table` | p | runtime-built | P3 | | AC-6 |

`AC6-QL-*` is the new "query or label built at runtime" kind (gateable P3; decision 2). It reuses the registered
`runtime-built` code (spec3 §7.2, "On: `READS`/`WRITES`"), so it needs no registry change.

**Look-alike twins (`TW-*`), not counted toward any minimum.** Each mirrors an AC-6 shape with every part visible,
so it must be **confirmed**: an engine that downgrades everything dynamic to `possible` passes AC-6 and fails these.

| ID | kind | twin of | planted item | locators | st | reason | gate |
|---|---|---|---|---|---|---|---|
| TW-1 | library_api | AC6-DYN | `importlib.import_module("plugins.audit")`, a literal | C `svc:plugins/loader.py::load_audit` | c | key-match | P1 |
| TW-2 | http | AC6-URL | `f"{BASE}/health"` where `BASE` is a module constant | C `job:job/run_report.py::health_probe` | c | key-match | P1 |
| TW-3 | call | LNK-30/39/40 | `AuditPlugin().run(evt)`: the receiver's type is known from its constructor | C `svc:plugins/loader.py::fixed_dispatch` | c | resolved-call | P1 |

### 5.4 B1's four assertions (`B1*`)

Each entity has two writers (W, W2) and readers (R); `B1-1..4` are asserted on W, W2 and R of each.

| ID | entity | role | locator | st | reason | gate | serves |
|---|---|---|---|---|---|---|---|
| B1F-W | field `svc:state/cache.py::OrderCache.entries` | writer W (`self.entries[k] = v`) | `svc:state/cache.py::OrderCache.put` | c | resolved-receiver | P1 | B1-1, B1-3, B1-4 |
| B1F-R | same | reader R | `svc:state/cache.py::OrderCache.get` | c | resolved-receiver | P1 | B1-1, B1-2, B1-3 |
| B1F-W2 | same | other writer W2 (`del self.entries[k]`) | `svc:state/cache.py::OrderCache.evict` | c | resolved-receiver | P1 | B1-2, B1-3, B1-4 |
| B1F-RU | same, key `?.entries` | reader through an untyped parameter | `svc:api/orders.py::peek` | p | field-name-match | P1 | B1-1 (listed after the confirmed reader), B1-3 |
| B1S-D | `sql:table:orders` | definition (DDL) | `svc:db/schema.sql` (`schemaEntry`, `objectKey` `sql:table:orders`) | c | key-match | P3 | B1-3 (rename) |
| B1S-W | same | writer W (`INSERT INTO orders`) | `svc:db/orders.py::insert_order` | c | key-match | P3 | B1-1, B1-3, B1-4 |
| B1S-R | same | reader R, **in another repo** (`SELECT count(*) FROM orders`: the table key only, so no column key is read; LNK-26's `render_row` is the column reader) | `job:job/run_report.py::load_orders` | c | key-match | P3 | B1-1, B1-2, B1-3 |
| B1S-W2 | same | other writer W2 (`DELETE FROM orders`) | `svc:db/orders.py::archive_orders` | c | key-match | P3 | B1-2, B1-3, B1-4 |

### 5.5 AC-25 impact shape (`DF-*`, `CD-*`, `RV-*`)

The changed function for the data-flow set is `CF-DF` = `svc:core/flow.py::base_price` (its return value
changes); for the control-dependency set, `CF-CD` = `svc:core/rules.py::eligible` (its condition changes). A
`group` is (repository, component, distance). Chains are one `DATA`/`CONTROLLED_BY` hop per distance step. An
HTTP link counts as one hop, and DF-3 and RV-3 are link-level relations unless S20/S33 provide a cross-language
`DATA` hop at P1 (OI-4). The A25-3 pairs (possible item planted to sort first under name, path and line order):

| pair | possible (name, file, line) | confirmed (name, file, line) | group |
|---|---|---|---|
| DF-4 / DF-2 | `accept_quote` (`api/orders.py`, defined first) | `create_order` (same file, later) | fx-svc / api / 2 |
| CD-4 / CD-3 | `publish_alert` (`bus/producer.py`, defined first) | `publish_created` (same file, later) | fx-svc / bus / 2 |
| `find_hook` / `is_member` (RV-1) | `find_hook` (`core/hooks.py`) | `is_member` (`core/rules.py`) | fx-svc / core / 1 |
| `find_hook` / `tier_of` (RV-2) | `find_hook` (`core/hooks.py`) | `tier_of` (`core/rules.py`) | fx-svc / core / 1 |

| ID | effect | locator | dist (direct?) | group | st | reason | gate | serves |
|---|---|---|---|---|---|---|---|---|
| DF-1 | `apply_discount` uses `base_price`'s return | `svc:core/flow.py::apply_discount` | 1 (direct) | fx-svc / core / 1 | c | intra-flow | P1 | A25-1..3; A25-4 via RV-1 |
| DF-2 | `create_order` uses `apply_discount`'s result: "a changed return value used two calls away" | `svc:api/orders.py::create_order` | 2 (indirect) | fx-svc / api / 2 | c | intra-flow | P1 | A25-1..3 |
| DF-3 | the response `total` of `POST /api/v1/orders` shown by the UI (crosses LNK-36) | `web:app/src/ui/OrderTotal.tsx::OrderTotal` | 3 (indirect) | fx-web / ui / 3 | c | key-match | P1 | A25-1..3 |
| DF-4 | `accept_quote` passes the discount through an untyped parameter: `discount.apply(rate)` | `svc:api/orders.py::accept_quote` | 2 (indirect) | fx-svc / api / 2 | p | name-match | P1 | A25-1..3 (A25-3 pair above) |
| CD-1 | a call guarded **inside** `eligible`: `if o.total > LIMIT: audit(o)` | `svc:core/rules.py::audit` | 1 (direct) | fx-svc / core / 1 | c | intra-flow | P1 | A25-1..3; A25-4 via RV-2 |
| CD-2 | a call in **another function** guarded by the condition: `finalize` runs `if eligible(o): notify(o)` | `svc:core/flow.py::notify` | 1 (direct) | fx-svc / core / 1 | c | intra-flow | P1 | A25-1..3 |
| CD-3 | `notify` reaches the bus only under that guard | `svc:bus/producer.py::publish_created` | 2 (indirect) | fx-svc / bus / 2 | c | intra-flow | P1 | A25-1..3 |
| CD-4 | the same guard covers `getattr(self, evt)()` in `publish_alert` | `svc:bus/producer.py::publish_alert` | 2 (indirect) | fx-svc / bus / 2 | p | name-match | P1 | A25-1..3 (A25-3 pair above) |

**Expected answer order** (A25-2; repository, component by byte order, distance ascending; inside a group
`confirmed` first, then by name):
- forward `CF-DF`: (fx-svc, api, 2) [DF-2, DF-4] · (fx-svc, core, 1) [DF-1] · (fx-web, ui, 3) [DF-3];
- forward `CF-CD`: (fx-svc, bus, 2) [CD-3, CD-4] · (fx-svc, core, 1) [CD-1, CD-2].

The reverse question ("what does this depend on?") for the **same changed function**, and a cross-repository one.
The list is known and complete; no other dependency may be listed as `confirmed`. The reverse relation is **data dependency** (what the subject's value is computed from), over functions, to distance <= 3; a call whose result is unused (`audit(o)` in `eligible`) and a call-based reading are not the model, and RV-3 stops at distance 3 for that reason (S33 confirms, OI-1).

| ID | subject | planted dependencies (dist, group, st); listed in this group order | gate | serves |
|---|---|---|---|---|
| RV-1 | `svc:core/flow.py::base_price` | fx-svc/core/1: `svc:core/rules.py::is_member` (c), `svc:core/hooks.py::find_hook` (p, name-match) · fx-svc/db/1: `svc:db/orders.py::load_rate` (c) · fx-svc/db/2: `svc:db/orders.py::connect` (c) | P1 | A25-4; A25-1..3 on each of the 4 |
| RV-2 | `svc:core/rules.py::eligible` | fx-svc/core/1: `svc:core/rules.py::tier_of` (c), `svc:core/hooks.py::find_hook` (p, name-match) · fx-svc/db/2: `svc:db/orders.py::load_rate` (c) · fx-svc/state/1: `svc:state/cache.py::OrderCache.get` (c) | P1 | A25-4; A25-1..3 on each of the 4 |
| RV-3 | `web:app/src/ui/OrderTotal.tsx::OrderTotal` (cross-repository) | fx-svc/api/1: `svc:api/orders.py::create_order` (c) · fx-svc/core/2: `svc:core/flow.py::apply_discount` (c) · fx-svc/core/3: `svc:core/flow.py::base_price` (c) | P1 | A25-4; A25-1..3 on each of the 3 |

Inside a group of `confirmed` items the order is by name. The two `find_hook` rows are the same function listed in
two answers (one planted function, two dependency entries).

### 5.6 Mis-resolved call (`MR-*`, P3-1)

One per language, so whichever language S27 puts at or above 0.95 carries it. The frontend resolves the call to
Y; the true callee is X. Impact from a change to X must list the call site as `possible`, `resolution-unverified`,
inside the folded group, and the canonical answer must contain it. **Precondition (TP-015):** the frontend's own
output resolves the call to Y; if it resolves to X the row is vacuous and is reported `blocked`, not passed.

| ID | planted item | locators | st | reason | gate | serves |
|---|---|---|---|---|---|---|
| MR-PY | `rules = cast(Rules, PremiumRules())`; `rules.fee(x)` resolves to `Rules.fee` | X `svc:core/rules.py::PremiumRules.fee` · Y `svc:core/rules.py::Rules.fee` · C `svc:core/flow.py::price_with` (call site `fee`, ordinal 0) | p | resolution-unverified | P1 | P3-1 |
| MR-TS | `(pick() as Formatter).render(x)` resolves to `Formatter.render` | X `web:app/src/ui/format.ts::FancyFormatter.render` · Y `web:app/src/ui/format.ts::Formatter.render` · C `web:app/src/ui/format.ts::formatTotal` (call site `render`, ordinal 0) | p | resolution-unverified | P1 | P3-1 |

### 5.7 Partial failure (`AB-*`, AC-9b)

Scenarios in §8. "Build failed" is produced by the materializer **forcing** the unit's or repository's build step to
fail (a manifest `buildFailure` marker, plus the syntax-error file as the human-readable cause), because a tolerant
frontend skips unparsable files and would not fail a build on them. **Precondition (TP-015):** the scenario state
(`failed`, `never built`) is observed before the answer is asserted. The single-repo pair gates at P1 (as §4.7);
the cross-repository pair gates at **P3** on `fx-job`, so it needs no Java. The carrier of `target-not-mapped` is
an `http` link (the callers fetch a service endpoint): `legacy` serves `GET /legacy/fee`, `tools` serves `GET /tools/run`.

| ID | scenario | planted item | expected | gate | serves |
|---|---|---|---|---|---|
| AB-1 | `SC-FAIL-UNIT` | `svc-legacy` builds at H1 and fails at every later commit (H2 onward, never reverted); the scenario builds H1, then H4. Differs from LNK-42 by having a previous refresh | the other units update; the answer crossing `svc-legacy` states "from the previous refresh" (H1) with the failure reason (stale-kept) | P1 | AC-9b |
| AB-2 | `SC-NEVER-UNIT` | `svc-tools` is never built (excluded from the build set) | `svc-tools` is reported "not mapped" in the unit report; its link is LNK-43 | P1 | AC-9b |
| AB-3 | `SC-FAIL-REPO` | `fx-job` builds at H1 and fails at H2 | an impact answer reaching `job:job/run_report.py::load_orders` (B1S-R) and `render_row` (LNK-26) states they are from the previous refresh, and why; `fx-svc` and `fx-web` are current | P3 | AC-9b |
| AB-4 | `SC-NEVER-REPO` | `fx-job` is never built | LNK-15 (`POST /job/trigger`) and LNK-16 (`GET /job/status`) are `possible`, `target-not-mapped`; a query for a path in `fx-job` answers "not mapped", not "not found" | P3 | AC-9b |

### 5.8 Multi-repository overlay (`OV-*`, AC-19)

The overlay is an un-merged change in one repository; the system is `fx-svc`, `fx-web` and `fx-job` (`fx-lib`
takes no part). Change set `CS-JOB` (branch `ov-job`, off HEAD): edits `job:job/run_report.py::write_report` (the
columns it writes to `order_report`) and `job:job/trigger_server.py::status` (the response shape). Change set
`CS-SVC` (branch `ov-svc`): renames the column `orders.total` in `svc:db/schema.sql` (its `schemaEntry`).
`web:app/src/ui/JobPanel.tsx::JobPanel` is planted by S37 for OV-2 and has no LNK row.

| ID | change set | affected code (other repositories) | st | reason | gate | serves |
|---|---|---|---|---|---|---|
| OV-1 | `CS-JOB` | listed: `svc:api/reports.py::report_view`, reader of `sql:table:order_report` (written by fx-job; LNK-44) | c | key-match | P3 | AC-19, LNK-44 |
| OV-2 | `CS-JOB` | listed: `JobPanel`, consumer of `GET /job/status` through `VITE_JOB_BASE` = `http://job:8300` (Compose; `job` Compose service: build context `../../fx-job`, port 8300, unit `job`) | c | binding-confirmed | P3 | AC-19 |
| OV-3 | `CS-JOB` | **not listed:** `svc:api/health.py::health`, `web:app/src/ui/Health.tsx::Health`, and `svc:clients/jobs.py::trigger_job` (LNK-15's caller of `POST /job/trigger`, which `CS-JOB` does not edit) | none | none | P3 | AC-19 |
| OV-4 | `CS-SVC` | listed: `job:job/run_report.py::render_row` (LNK-26's reader of `sql:column:orders.total`): the column is removed and a new one added | c | key-match | P3 | AC-19 |
| OV-5 | `CS-JOB` | listed **after** OV-2: `web:app/src/ui/JobStatus.tsx::JobStatus`, the other consumer of `GET /job/status` (LNK-16) | p | port-shared | P3 | AC-19 |

### 5.9 Signature change and the earlier-verdict note (`SQC-*`)

A verdict is seeded on a `possible` edge **whose target is the edited symbol**, before the change (cross-manifest
`verdicts:`, S39). Python identity has no signature in its `symbolId`, so the note is **kept**; Java's includes it,
so the old symbol disappears and the note is **dropped** (bundle spec3 §3.2). These are regression pins of a
design consequence, not independent evidence. Change point H3, state before = H1 (H2 is the breakage scenario).

| ID | change | seeded verdict on | expected | gate | serves |
|---|---|---|---|---|---|
| SQC-PY | H3 edits `svc:core/flow.py::price(item, qty)` to `price(item, qty, *, region=None)` | the `name-match` edge from `svc:api/orders.py::estimate` (`pricer.price(item, qty)`, untyped receiver) to `price` | the earlier-verdict note is kept (a symbol id that included the parameter list would drop it) | **P4** | SQ-C |
| SQC-JAVA | H3 edits `fx-lib:pricing/src/main/java/com/acme/fxlib/Pricing.java::Pricing.quote(int,String)` to `quote(int,String,Currency)` | the `name-match` edge from `fx-lib:quote-cli/src/main/java/com/acme/fxlib/cli/QuoteCli.java::QuoteCli.renderLegacy()` (`handle.quote(qty, sku)`; `handle` is a `com.vendor.handles.Handle` from a jar absent from the classpath, so its type is unresolved) to `Pricing.quote(int,String)` | the old id (signature `(int,String)`) is gone; the note is dropped | **P2** | SQ-C |

### 5.10 Taint finding edited at a hop (`TF-*`, P2-7)

Chain in `svc:api/ops.py` (P5): source `run_cmd` (request parameter `cmd`) to `normalize` (middle hop) to
`exec_cmd` (`subprocess.run(cmd, shell=True)`, the sink). A verdict is seeded on the finding; each row is its own
branch off HEAD with one edit. TF-1 also **adds a line above the sink**, so a line-number-keyed implementation
fails it (a body-only edit that keeps line counts passes both correct and line-keyed code).

| ID | branch | edit | expected | gate | serves |
|---|---|---|---|---|---|
| TF-1 | `tf-mid` | body of `svc:api/ops.py::normalize` (middle hop), plus one added line | the verdict survives | P5 | P2-7 |
| TF-2 | `tf-sink` | `svc:api/ops.py::exec_cmd` (end: the sink) | the verdict expires | P5 | P2-7 |
| TF-3 | `tf-src` | `svc:api/ops.py::run_cmd` (end: the source) | the verdict expires | P5 | P2-7 |

### 5.11 Omitted files (`SK-*`, bundle spec4 §1)

Unit `web` (repoId `fx-web`, `rootPath` `app/`, lang `javascript`, `vendoredDirs` `vendor`); paths are
**repo-relative**, as `skippedFiles` entries are. JS/TS tests are excluded **by design** and listed with reason
`test`; a production file the frontend's default ignore rules drop (the `salesperson/src/i18n/` case, S11 finding 3)
is **included if the P1 extractor can override the ignores**, else listed `frontend-ignore`. The ON column holds
only if the P1 plan verifies the override (spec4 says it is unverified and the fallback is likelier); otherwise
OFF is the expectation. Gate P1. The coverage probe (spec4) is a **generated table** over language x reason x
overlap x location x override: S39 enumerates its cells from the axes and a cell with no row fails TP-007; the
rows below are the named instances.

| ID | path | why it is planted | expected, override ON | expected, override OFF |
|---|---|---|---|---|
| SK-1 | `app/src/api/orders.test.ts` | basename `*.test.<ext>` | listed, `test`; no `file` record | same |
| SK-2 | `app/src/ui/Form.spec.tsx` | basename `*.spec.<ext>` | listed, `test` | same |
| SK-3 | `app/src/__tests__/helpers.ts` | directory segment `__tests__` | listed, `test` | same |
| SK-4 | `app/e2e/flow.ts` | directory segment `e2e` | listed, `test` | same |
| SK-5 | `app/src/i18n/format.ts` | **production** file under the frontend ignore rules | `file` record; not listed | listed, `frontend-ignore`; no `file` record |
| SK-6 | `app/src/i18n/messages.test.ts` | both `test` and frontend-ignored | listed **once**, `test` | listed once, `test` |
| SK-7 | `app/vendor/pkg/x.test.ts` | a test under a vendored directory | not listed; no `file` record (explained by `vendoredDirs`) | same |
| SK-8 | `scripts/build.test.ts` (in `fx-web`, outside `rootPath`) | location outside the unit | not in this unit's list | same |
| SK-9 | `app/src/types/api.d.ts` | declaration file is not a source file | not listed; no `file` record | same |
| SK-10 | `app/src/ui/latest.ts` | decoy: "test" inside a name | `file` record; not listed | same |
| SK-11 | `app/src/ui/e2e-helper.ts` | decoy: "e2e" in a name, not a directory segment | `file` record; not listed | same |
| SK-12 | `fx-svc:tests/test_api.py` | Python tests stay in the graph; the pattern is JS/TS only; the file has no HTTP client call | `file` record; not listed | same |
| SK-13 | `fx-svc:scripts/run.test.sh` | other language: no pattern in v1 | `file` record; not listed | same |
| SK-14 | `app/src/util/a.test.mjs` | extension variant `.mjs` | listed, `test` | same |
| SK-15 | `app/src/util/b.spec.cts` | extension variant `.cts` | listed, `test` | same |
| SK-16 | `app/src/util/c.test.jsx` | extension variant `.jsx` | listed, `test` | same |
| SK-17 | `app/src/__tests__/deep/nested.ts` | `__tests__` nested deeper | listed, `test` | same |
| SK-18 | `app/vendor/i18n/y.ts` | `frontend-ignore` shape under a vendored directory | not listed; no `file` record | same |

### 5.12 Negative controls (`NC-*`)

Planted so that over-linking is caught; they are not counted toward any minimum. Each must produce **no link** to
the provider named (a consumer with no provider keeps its `CONSUMES` edge and gets no `Link`).

| ID | planted item | expected | gate | serves |
|---|---|---|---|---|
| NC-1 | `svc:docs/README.md` has a fenced `curl localhost:8000/api/items/1` snippet ("documentation snippets are not callers", proposal §3.4) | no `CONSUMES`, no link | P1 | LNK |
| NC-2 | `job:job/run_report.py` has the comment `# TODO call POST /api/v1/orders` | no `CONSUMES`, no link | P1 | LNK |
| NC-3 | **verb mismatch:** `web:app/src/ui/Probe.tsx::probeVerb` does `fetch("/api/items/7", {method: "DELETE"})`; `read_item` is `GET` only | a `CONSUMES` edge, no `Link` to `read_item` | P1 | LNK |
| NC-4 | **name extension:** `web:app/src/bus/consumer.ts::onDlq` subscribes `orders.created.dlq`; no producer | a `CONSUMES` edge, no `Link` (not to `kafka:orders.created`) | P2 | LNK |
| NC-5 | **tool collision:** `job:job/mcp_admin.py::reindex` registers a second tool `reindex` | LNK-12 (URL binding) and LNK-19 (`MCP_URL`) each have exactly one provider, `svc:mcp/server.py::reindex`; LNK-20, which has no binding, lists both | P1 | LNK |
| NC-6 | **log message:** `svc:api/items.py::read_item` logs `"GET /api/items/1 ok"` | no `CONSUMES` from that literal | P1 | LNK |
| NC-7 | **prefix mismatch:** `web:app/src/ui/Probe.tsx::probePrefix` fetches `/items/7` (no `/api`) | a `CONSUMES` edge, no confirmed `Link` to `read_item` | P1 | LNK |

### 5.13 Field-depth cap (`FT-*`)

`fieldDepthCap` is 3 fields after the head (bundle spec3 §5.1, §1.1; `self` is the head `recv`, so `self.a.b.c` has 3), so a deeper access is truncated and its
`DATA` fact is `possible`, `field-truncated`; the boundary is a control.

| ID | planted item | locator | st | reason | gate | serves |
|---|---|---|---|---|---|---|
| FT-1 | `return self.a.b.c.d` (`recv` + 4 fields, over the cap) | C `svc:state/cache.py::OrderCache.limit_deep` | p | field-truncated | P1 | FT |
| FT-2 | `return self.a.b.c` (`recv` + 3 fields, exactly the cap) | C `svc:state/cache.py::OrderCache.limit_edge` | c | intra-flow | P1 | FT |

## 6. Minimums: derivation and census

Each floor is eval-protocol §12.2's; the **count** is S35's decision, taken as the floor unless a reason is given.
TP-004 recomputes every figure from the manifests.

| Minimum (§12.2) | Floor | Planted | Derivation | Rows |
|---|---|---|---|---|
| AC-6, runtime-built URL (P1) | 3 inst., 3 shapes | 3 | floor | AC6-URL-1..3 |
| AC-6, dynamic import (P1) | 3 / 3 | 3 | floor | AC6-DYN-1..3 |
| AC-6, example config (P1) | 3 / 3 | 3 | floor | AC6-EX-1..3 |
| AC-6, prompt mention (P1) | 3 / 3 | 3 | floor | AC6-PM-1..3 (prose, table cell, Python string) |
| AC-6, topic from config (P2) | 3 / 3 | 3 | floor | AC6-TOP-1..3 |
| AC-6, query/label built at runtime (P3; decision 2) | >= 3 shapes | 3 | floor | AC6-QL-1..3 |
| AC-25 data-flow effects | >= 3, one at d1, one at d2 | 4 | 3 for the floor, at distances 1, 2, 3 (§4.7); +1 `possible` companion (DF-4) so that A25-3 has a group holding both statuses | DF-1..4 |
| AC-25 control-dependency effects | >= 3, one guarding a call in another function | 4 | 3 for the floor (CD-2, CD-3 guard calls in other functions); +1 companion (CD-4), same reason | CD-1..4 |
| AC-25 reverse | >= 3 planted functions with a known list | 3 | floor | RV-1..3 |
| Difficulty shapes (10) | >= 3 each | see below | floor | LNK, AC6 |
| Link kind x status | >= 1 each | 20 of 20 cells | 10 kinds x 2 statuses (§7) | LNK, AC6, B1 |

**Difficulty shapes, instances by earliest gate** (cumulative; a shape reads as n = 3 only from the gate that holds
3 of its rows). Runtime-built key, dynamic import and prompt mention are the AC-6 rows themselves (no double
planting).

| Shape | P1 | by P2 | by P3 | Rows |
|---|---|---|---|---|
| exact literal | 3 | 3 | 4 | LNK-01, 02, 36 (+03 at P3) |
| constant across files | 3 | 4 | 4 | LNK-04, 06, 37 (+05 at P2) |
| template literal | 3 | 4 | 4 | LNK-07, 08, 38 (+09 at P2) |
| client wrapper | 3 | 3 | 3 | LNK-10..12 |
| config-bound base URL | 3 | 3 | 3 | LNK-13..15 |
| runtime-built key | 3 | 3 | 4 | AC6-URL-1..3 (+LNK-29 at P3) |
| dynamic import | 3 | 3 | 3 | AC6-DYN-1..3 |
| DI or reflection | 3 | 5 | 5 | LNK-30, 39, 40 (+31, 32 at P2) |
| prompt mention | 3 | 3 | 3 | AC6-PM-1..3 |
| version skew (cross-repository) | **0** | 3 | 3 | LNK-33..35 (+ control LNK-41) |

Skew is below 3 before P2 because version-skew `library_api` linking is P2 (§4.7); readings of that shape before P2 do not exist.
Every other shape holds 3 at P1.

**Totals.** Two definitions, stated once: **rows** = every row of every table in §5; **positive items** = rows minus
the 12 negatives, twins and controls (NC-1..7, TW-1..3, OV-3, FT-2).

| Category | Rows | Category | Rows |
|---|---|---|---|
| Link rows (`LNK`) | 44 | Mis-resolved call (`MR`) | 2 |
| Key normalization (`KN`) | 5 | Partial failure (`AB`) | 4 |
| AC-6 unprovable (`AC6`) | 18 | Overlay (`OV`) | 5 |
| Look-alike twins (`TW`) | 3 | Signature change (`SQC`) | 2 |
| B1 four assertions (`B1F`, `B1S`) | 8 | Taint edits (`TF`) | 3 |
| AC-25 forward (`DF`, `CD`) | 8 | Omitted files (`SK`) | 18 |
| AC-25 reverse (`RV`) | 3 | Field depth (`FT`) | 2 |
| Negative controls (`NC`) | 7 | **Rows** | **132** |

Positive items: **120** (LNK-41, KN-3 and KN-4 are controls by purpose but are asserted links, so they count). By earliest gate: P1 91 · P2 16 · P3 21 · P4 1 · P5 3 (= 132). `fx-lib` rows are all P2
(LNK-23, 24, 25, 31, AC6-TOP-3, SQC-JAVA); no P3 row needs Java.

The AC-25 set is **60 distinct assertions** (§3). DF-3 (via LNK-36) and OV-2 plant a UI consumer
with no `LNK` row of their own beyond those: `JobPanel` is planted by S37 and counts toward the link matrix but no
shape. Counts without a §12.2 floor follow §4.7 and the proposal: P3-1 (2, one per language), AC-9b (2 + 2), AC-19
(two change sets, 5 rows), P2-7 (middle, sink end, source end = 3; the proposal says "either end").

## 7. Link-kind x status matrix and reason-code coverage

Every kind x status cell has at least one row; every `possible` row carries its expected reason.

| Kind | confirmed | possible |
|---|---|---|
| http | LNK-01, 04, 07, 08, 10, 11, 13, 14, 15, 18, 36, 37, 38; KN-1..4; TW-2 | LNK-16 (port-shared), 17 (partial-key), 42, 43 (target-not-mapped); KN-5; AC6-URL-1..3; AC6-EX-2 |
| mcp_tool | LNK-02, 06, 12 | AC6-PM-1..3 (prompt-mention) |
| script -> tool | LNK-19 | LNK-20 (shell-dynamic) |
| launch chain | LNK-21, 22 | AC6-EX-1 (example-config) |
| rpc | LNK-23 | LNK-32 (runtime-built / unknown-key) |
| topic (Kafka, in-house bus) | LNK-05, 09 (Kafka), 24 (in-house) | AC6-TOP-1..3 (config-derived) |
| library_api | LNK-25, 41; TW-1 | LNK-33..35 (name-only-provider, skew) |
| data_entity (SQL, graph label) | LNK-26, 27, 44; B1S-* | AC6-QL-1..3 (runtime-built) |
| config_key | LNK-28 | LNK-29 (runtime-built) |
| env_var | LNK-03 | AC6-EX-3 (example-config) |

**Reason codes planted** (spec3 §7.2): `resolved-call` (TW-3), `key-match`, `lifted-role`, `binding-confirmed`,
`resolved-receiver`, `intra-flow`, `name-match`, `resolution-unverified`, `field-name-match`, `field-truncated`
(FT-1), `shell-dynamic`, `runtime-built`, `dynamic-import`, `example-config`, `prompt-mention`, `partial-key`,
`name-only-provider`, `unknown-key`, `binding-narrowed`, `port-shared`, `target-not-mapped`, `config-derived`, and
the `skippedFiles` codes `test` and `frontend-ignore`. **Not planted:** `verdict-confirmed` (the SQC/TF observable
is the note, not a status), `flow-unavailable:<lang>` (needs a unit with `flowAvailable = false`: OI-5),
`headers-unavailable` (C/C++, the Arduino slot), `possibly-reachable` and `advisory-unscored` (P5 findings outside
these categories).

## 8. History and scenarios

S39's materializer builds each repository with a scripted, linear history (fixed author and committer; fixed
dates, so SHAs are stable; §4.7). Commits are named so that rows and scenarios cite them. The **RCA/secrets
window** is `[2026-02-01, 2026-03-01)`; H1-H3 are in it, H0 and H4 outside.

| Commit | Date | Repos | Content | Used by |
|---|---|---|---|---|
| H0 | 2026-01-05 | all | initial tree: every row of §5 except the H1+ edits (out of window) | steady-state rows |
| H1 | 2026-02-10 | all | a successful build point: `legacy` and `fx-job` build; `Pricing.quote(int,String)` and `price(item, qty)` still exist | AB-1, AB-3 ("previous refresh"); SQC state-before |
| H2 | 2026-02-15 | `fx-svc`, `fx-job` | forced build-failure marker on `legacy` (kept to HEAD) and on `fx-job` (reverted at H3), with `legacy/broken.py` and `job/broken.py` (syntax errors) | AB-1, AB-3 (the failing refresh) |
| H3 | 2026-02-20 | `fx-svc`, `fx-job`, `fx-lib` | reverts H2 for `fx-job` only; `price` gains `*, region=None`; `Pricing.quote(int,String)` becomes `(int,String,Currency)` | SQC-PY, SQC-JAVA (before = H1, after = H3) |
| H4 = HEAD | 2026-03-10 | `fx-svc`, `fx-job` | comment-only edits to `svc:core/flow.py` and `job:job/run_report.py` (out of window); the manifests describe this tree | steady state |
| branches `ov-job`, `ov-svc` | off H4 | `fx-job`, `fx-svc` | the un-merged change sets of §5.8 | OV-1..5 |
| branches `tf-mid`, `tf-sink`, `tf-src` | off H4 | `fx-svc` | the single edits of §5.10 | TF-1..3 |

History is linear (H0..H4); H2 is a **scenario** commit: `SC-FAIL-*` build H1, then H2. S35 plants no secret value
(OI-6). Proposal §5.1's "verdict issued during a running build" (B6) needs no planted item: any seeded `possible`
edge (the SQC-PY edge) serves.

| Scenario | Build state | Rows |
|---|---|---|
| `SC-STEADY` | a fresh build at H4: `svc-tools` never built, `svc-legacy` unbuildable (no earlier refresh), all else built (`lib-*` from P2) | every row asserted in this state, except the rows a scenario below overrides |
| `SC-FAIL-UNIT` | H1 built, then H4: `svc-legacy` fails with a previous refresh | AB-1 |
| `SC-NEVER-UNIT` | `svc-tools` excluded from the build set | AB-2 |
| `SC-FAIL-REPO` | H1 built, then H2 for `fx-job`: the repository fails | AB-3 (LNK-15, 16, 26, B1S-R, OV-* are asserted only in `SC-STEADY`) |
| `SC-NEVER-REPO` | `fx-job` never built | AB-4 (same exclusion) |

## 9. Test items (S39 and the fixture steps)

Each item names a **planted failing input** (a deliberately wrong manifest, engine stub or mutant that S39 keeps in
its tests), because a check that cannot fail is not a check. Type: C contract, F functional, I integration, L live.

| ID | title | steps | expected | planted failing input | prio | type | gate |
|---|---|---|---|---|---|---|---|
| TP-001 | Materializer determinism | run `cpg/fixtures/materialize.py` twice into clean dirs | same SHAs for four repositories, branches and tags, both runs | a materializer mutant with the committer date unset (SHAs differ) | high | F | S39 |
| TP-002 | Locators resolve | convert every §5 locator by spec3 §3.5, §1.9 | `qname` starts `path + "::"`; the symbol is in the file at the pinned commit; Java locators carry a §4.3 signature; `line` in span; every `{kind, keyNorm}` hashes | a manifest copy with `svc:api/health.py::health` rewritten to `svc:api/healthz.py::health`, and one Java locator without a signature | high | C | S39 |
| TP-003 | Manifests validate | validate `expected/*.yaml` and `cross.yaml` against `fixture-manifest-v1.schema.json`; codes against the registry | all valid; every `possible` has a reason | a row with `possible` and no reason; a row with reason `foo-bar` | high | C | S39 |
| TP-004 | Minimums census | a script recomputes §6 from the manifests: floors, 20 cells, per-gate shape counts | every floor of eval-protocol §12.2 met; counts equal §6's tables; skew reads 0 before P2 | a manifest copy with one `AC6-PM` row deleted and one shape tag changed (both must turn it red) | high | C | S39 |
| TP-005 | keyNorm rows (F-13) | run `cpg/tests/test_fixture_keynorm_mutants.py` (in the repo; `.venv/bin/python -m pytest` from `cpg/`: 7 passed on 2026-11-01), which normalizes each `KN-*` literal with `cpg/ccg/keynorm.py` composed with decode and dot resolution | the keys of §5.2 | the two mutants in that file: resolve-before-decode fails KN-1, KN-2; mask-before-decode fails KN-5 | high | F | S39 (after S12) |
| TP-006 | Content check, live | parse each repository in the sandbox; `cpg-content-check.sc` (D15) | passes per repository; `fx-lib` only from P2 | a unit whose only file holds comments (zero non-external `METHOD`s: D15's unit-level floor must fail it); if the frontend still emits a module-level `METHOD` for it, a unit with no source files | high | L | S39 |
| TP-007 | Omitted-file probe | generate the probe table (language x reason x overlap x location x override); run each `SK-*` | outcomes of §5.11 per mode; sorted, unique, POSIX; no `file` record for a listed path | an extractor mutant that emits `file` records for test files; a table cell with no row | high | I | P1 |
| TP-008 | AC-25 presentation | ask the forward and reverse questions for `CF-DF`, `CF-CD`, `RV-1..3`; assert the 60 assertions of §3, in the §5.5 order | all pass, bound to S33's fields (OI-1) | answer builders that sort (a) by (repository, component, distance, name) with no status key (the four A25-3 pairs), (b) by distance first (api/2 before core/1 fails A25-2), (c) with distance off by one (A25-1), (d) with groups flattened (A25-2) | high | I | P1 |
| TP-009 | AC-6 gate per kind | per kind, every row `possible` with its reason, none `confirmed` | pass per kind | an engine stub that confirms one instance; one that emits `possible` with no reason | high | F | P1/P2/P3 by kind |
| TP-010 | Link matrix, closed world | emit and compare every `LNK`, `B1`, `KN`, `FT` status and reason; check B1-1..4 on the field (P1) and the table (P3); no emitted link outside the manifests | exact match; zero unplanted links | a stub that emits one unplanted link; one with a wrong reason | high | I | by row |
| TP-011 | Scenarios | run `SC-*` of §8 and the SQC/TF branches | AB, OV, SQC, TF as in §5.7-§5.10 at their gates | a stub that reports a failed unit as current (AB-1); one that keeps the note on the Java change (SQC-JAVA); a line-keyed verdict that fails TF-1 | medium | I | P1-P5 |
| TP-012 | Negative controls | NC-1..7, OV-3 | no link, nothing unaffected listed | a stub linking NC-3 (wrong verb) and NC-7 (missing prefix) to `read_item` | medium | F | P1/P3 |
| TP-013 | Stratum labelling | check manifests and any split file | every row `stratum: fixture-dev`; no row in a held-out split (F7) | a manifest row labelled held-out or unlabelled | medium | C | S39 |
| TP-014 | Gate hygiene | list rows by gate | no row asserted before its dependency's gate; `fx-lib` rows only from P2; P3 rows need no Java | a row `LNK-26` moved to gate P1; a Java row at P1 | medium | C | S39 |
| TP-015 | Scenario and mis-resolution preconditions | observe `failed` / `never built` for AB; observe the frontend resolving MR-PY, MR-TS to Y | preconditions hold, else the row is `blocked` | a repository whose forced-failure marker is removed; a frontend resolving `rules.fee` to X | high | I | P1/P3 |
| TP-016 | History windows | list commits in `[2026-02-01, 2026-03-01)` | exactly H1, H2, H3; H0 and H4 outside | the boundary moved to 2026-02-12 (list changes) | medium | F | S39 |
| TP-017 | Look-alike twins | TW-1..3 | all `confirmed` | an engine that downgrades every dynamic-looking site to `possible` | medium | F | P1 |
| TP-018 | Planter completeness | grep the generated trees for route, tool, topic and table literals and check each against a row (`fx-svc:tests/test_api.py`, SK-12, contains no HTTP client call) | every literal that can form a `Link` is in a row | an extra `fetch("/api/health")` added to a fixture file | medium | C | S39 |
| TP-019 | Candidate containment | for every `possible` row with a named true target (§5.1, §5.3), the answer's candidate set contains it | contained | a stub emitting `possible` with an empty, or a wrong, candidate set | medium | I | by row |

Execution notes for S39: automated in `cpg/tests/test_fixtures.py`, the existing pytest layout of `cpg/tests/`; run
it with `cpg/` as the working directory. Baseline first (`cpg/` suite green) before the fixtures run. Where the
exit clause says "fix whichever is wrong", the arbiter is the RFC 3986 derivation and the spec text, never the
implementation's output.

## 10. Environment, entry and exit criteria

- **Environment:** a Linux host with the sandbox runner (`cpg/sandbox/offline-run.sh`, S08) for the `live` item;
  Python 3.12 for the rest. No write to `falkordb-dev` (6379); the engine items use `ccg-falkordb` (6390) only
  through a reservation `teco` grants. Repositories are generated under `cpg/.ccg-work/fixtures/` (gitignored); the
  sources under `cpg/fixtures/system1/**` are the committed artifact.
- **Data:** named, deterministic; no randomness; fixed dates; the window boundary dates are a manifest field.
- **Entry:** S03 reason codes frozen (spec3 §7, spec4 §1); S06 minimums final; S36-S38 and S35b delivered against
  these rows; S08 and S10 available for TP-006 and TP-003.
- **Exit:** TP-001..004, 013, 014, 016 green with their planted failing inputs red; every row of §5 resolves
  (TP-002); every engine-facing item passes at the gate its row names or is reported `blocked` with the missing
  subject. A disagreement between a row and a repository is fixed in whichever is wrong; a change to a row is an
  S35 revision.

## 11. Out of scope

- The held-out, proving-repo AC-5 items and strata (eval-protocol §9); fixture items are never pooled with them.
- Performance, scale and latency (S20, S21); security of the fixture contents (no secret value is planted, OI-6).
- AC-26 (Arduino; the optional slot's fixture), AC-7's second codebase, AC-5's OpenTelemetry strata.
- C/C++, C#, Go and Kotlin extractors; P5 findings other than the P2-7 verdict rows.
- S33's answer schema beyond the four AC-25 properties.

## 12. Decisions and open items

**Decisions (routine ambiguity; each reversible by an S35 revision)**

- **D-1 `fx-job`**, step **S35b**: the 4th repository, Python/config, gating AC-19 and cross-repository AC-9b at P3
  without Java and carrying the skew rows; `fx-lib` gates only at P2; S39 materializes four repositories; the
  AC-19 overlay uses `fx-svc`, `fx-web`, `fx-job`. I added HTTP providers (`/job/trigger`, `/job/status`) so that a
  never-built repository (AB-4) has something to be "not mapped" for.
- **D-2** AC-6 gains "query or label built at runtime" (AC6-QL, three shapes, P3).
- **D-3** SQ-C is split: SQC-PY gates P4, SQC-JAVA P2; rpc, topic, library_api stay P2.
- **D-4** The K4 hedge at P3 exit uses the Kafka pair across `fx-svc` and `fx-web`: LNK-05 and LNK-09, which gate P2
  and are re-run at P3 exit.
- **D-5** Fixture keys follow F-13 (decode, then mask): KN-1..5; no other row contains a `%`.
- **D-6** Omitted files: SK-1..18 exercise both reasons with the closed test pattern, override on and off.
- **D-7** "script -> tool" is a shell script invoking a named MCP tool through a CLI (`mcp-call`).
- **D-8** Skew, reflection and dynamic-RPC rows (LNK-31..35, 41) use the existing registry (it has no skew code).
  which has no skew code: `name-only-provider` with `versionSkew`, `name-match`, `runtime-built / unknown-key`.
- **D-9** AC-9b's P1 pair is unit-level in `fx-svc`, carried by `http` links; the repository-level pair is `fx-job` at P3.

**Open items**

- **OI-1** S33's answer schema is unwritten, so A25-1..4 are properties; S39 binds each to S33's fields once
  `docs/plans/connected-code-graph-answer-spec.md` exists. A property with no field is an S33 defect.
- **OI-2** `teco` reconciles `docs/plans/connected-code-graph-p0-2.md` §4.7 and the S39 row (they still say three
  repositories) and adds S35b; `S35b` is unused in the plan and ledger (checked).
- **OI-3** Owner: the S30 rule-pack owner. Confirm the skew reasons, and how a consumer's `library_api` key is built
  from a pinned version (this plan assumes a typed stub, `job/stubs/fx_svc_common_1_4.pyi`).
- **OI-4** Owner: S20/S33. Distance across an HTTP link is one hop, and DF-3/RV-3 are link-level unless a
  cross-language `DATA` hop exists at P1; distances are the engine's own model (§1).
- **OI-5** `flow-unavailable:<lang>` is not planted: it needs a unit with `flowAvailable = false` (S27). Owner: S27.
- **OI-6** No secret-sighting rows (§4.7 has no secrets category). Owner: the S30 rule-pack owner, who plants the
  secrets rows with the keyed-hash rule of the graph note.
- **OI-7** Whether `jssrc2cpg` drops `fx-web/app/src/i18n/` by default must be verified live (TP-006/TP-007); if
  not, SK-5's OFF column is untestable and the directory must change. The ON column needs the P1 override plan.
- **OI-8** Owner: S16 (spec3 consolidation owner). Unit nesting ("innermost unit wins"; `svc` holds `svc-common`,
  `svc-legacy`, `svc-tools`) is an assumption.
- **OI-9** TW-3 was a literal `getattr`; spec3 has no resolution rule for it, so it is now a statically typed constructor call (`resolved-call`). The twin for DI by `getattr` is dropped.
  that the P1 extractor resolves a literal `getattr`.
- **OI-10** Owner: `data-scientist`. Eval-protocol §8.2's S35 row lists `repoId` in `{fx-svc, fx-web, fx-lib}`; add `fx-job`.
