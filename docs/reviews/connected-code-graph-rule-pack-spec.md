# Review: connected code graph — rule-pack spec (S30)

> **Status:** active · **Owner:** `analyst` · **Tracks:** C-1001 (M10)

## Scope & verdict

**Reviewed:** `docs/plans/connected-code-graph-rule-pack-spec.md` v1.0 (architect, uncommitted, 1,160 lines) against `docs/plans/connected-code-graph-p0-2.md` (S30/S31/S32 rows, §4.2), `-bundle-spec3.md` + `-bundle-spec4.md`, `-p0-graph.md`, the proposal, `docs/requirements/connected-code-graph.md`, `-eval-protocol-ml.md` §8.2/§9.1/§15, the slice report, the `s14a` bundles (`cpg/.ccg-work/vertical-slice/s14a/`), `cpg/spikes/vertical-slice/assemble.py`, and the source at `defca60f`. Root `REVIEW.md` applied; no component `REVIEW.md` is touched.

**Verdict: needs changes** (2 blockers, 8 majors). The FastAPI/fetch equality claims and the infra-signature pack hold up under re-run. What fails is closure: the format as written cannot yield the schema S31 must produce for the packs the spec itself ships, and the FastMCP walkthrough rests on records the frozen argv gate forbids.

CPG: considered, not relevant — a design-document review checked directly against the stored bundles and source; no loaded graph indexes `cpg/`.

Checks run: `assemble.py` over both `s14a` bundles (61 providers, 11 consumers, 11 links) and an independent re-derivation of the pack rules over the raw records; PyYAML parse of all 12 YAML blocks; compile + sample + negative-control script over the 45 signatures; a recompute of §9 and its 10 vectors; `git show defca60f` for every cited source line; FastMCP and FastAPI behaviour run in `falkor-chat/server/.venv`. The suite was not run (no code under review).

## Blockers

### B-1. The walkthrough's paper launch records contradict the frozen argv gate (§4.3 L2/L4, §4.4 steps 1, 3, 4, 7; O-4)
*Verified (read spec3 §1.6 and §1.6.1, git show of the scripts).* L2 and L4 are `exec "$VENV_DIR/bin/uvicorn" falkorchat.app:app $UVICORN_ARGS …`. Spec3 §1.6 says `command` is "argv[0] after the argv gate, `"<redacted>"` when it fails", and §1.6.1 says argv[0] "is judged by (v1) and (v2) alone"; (v1) admits only `[A-Za-z0-9._/:@+=,-]` and "fails `$NAME`/`${…}`". `$VENV_DIR/bin/uvicorn` contains `$`, so `command` is `<redacted>`, not `uvicorn`. O-4 calls the shape "unspecified"; the frozen text specifies it.
Consequence: `targetArg`/`frameworkDefaults`/`portFlags` all key on `command: {basename: uvicorn}`, so none fires for L2/L4. Steps 1 (unit resolved only via L1 then), 3 (`pack-default` 8000) and 4 (`AGENT_TEAM_PORT` adoption, which needs "a command launch in the same file") are not derivable. The plan-required "env default 8200" demonstration (p0-2 §4.2) is then supported by L3 alone, which adopts nothing.
**Fix (architect):** either (a) re-walk steps 3-4 with the records the gate permits (`command "<redacted>"`, `args [falkorchat.app:app, <redacted>, …]`) and let `targetArg` match on the `module:attr` argument with `command` unconstrained, state that `frameworkDefaults` then cannot fire for these scripts and say so in §4.4; or (b) route a `-spec5` change to (v1) for a `$VAR`-prefixed argv[0] to the freeze owner. (b) reopens a frozen gate; prefer (a). Re-label O-4 as a contradiction, not a gap.

### B-2. The format is not closed: the spec's own packs violate its own definitions, and one declaration has no shape (§1.3, §1.6, §4, §9; N-18)
*Verified by reading each pack block against §1.* S31's done-condition is "the three packs validate" with "schema = S30"; a faithful schema rejects them:
1. **`basename` matcher** (§4 `launch.*.command: {basename: uvicorn}`; §1.2 subject table says "a `basename` matcher") is not in the §1.3 matcher table (`exact|template|glob|prefix|regex`).
2. **`urlBindings.scope`**: §1.6 puts `scope: file|unit` on the `urlBindings` entry; §4 puts `scope: file` inside one `urlSources` item.
3. **`severity` packs' `rows`** (§1.1, §9, N-18, §12.1 "the encoding is S32's") have no shape. S31 must write the schema for kind `severity` and test N-18 (`level: severe`, `rank: 4`) before S32 exists.
4. Smaller undefined shapes the packs use: the value matcher inside `args` (§1.2 says only "an optional value matcher"; Kafka uses presence only); list-valued `resolution: [external, unresolved]` and `scope.imports` lists (OR semantics unstated); `adoptedBy: same-file-launch` and `form: module-attr` (enum members unlisted); the reserved `handler` result column that anchors a `query` row (stated only in §6, not §1.7); the `anyOf` YAML form; `finding.rule` names a §9 row, but §9 computes the row from class/variant/example features (a pack should emit the class, not the row).
**Fix:** give each a one-line definition in §1 (or correct the pack), add `rows` as a table of `{id, level, rank, when}` with the feature vocabulary of §9, and add the missing matcher. Then require S31 to validate the packs from the spec's text mechanically (extract the YAML blocks, as S10 does with spec3 vectors) rather than hand-copy.

## Majors

### M-1. The `glob` in the FastMCP `mountPaths` rule does not match the record it is claimed to match (§4 YAML, §4.2)
*Verified (ran §1.3's glob definition against the real record).* `externalTarget` of `app.py:413` is `mcp/server/fastmcp.py:<module>.FastMCP.streamable_http_app`. §1.3 defines `*` as "any run of characters except `/`", so `*FastMCP.streamable_http_app` does not match (it contains `/`); `**FastMCP.streamable_http_app` does. §4.2 says "which the glob matches". As shipped verbatim, the only MCP mount-path rule never fires, and step 7 has no mount path. **Fix:** use `**…` (or a `suffix`/`regex` matcher), and make S31's positive control run each pack matcher against the real S14 record it is walked on.

### M-2. `tool` rule emits a wrong `confirmed` key for `@mcp.tool("name")` (§4)
*Verified (read `mcp/server/fastmcp/server.py` in the venv, mcp 1.28.1).* `FastMCP.tool(self, name=None, …)`: `name` is the first positional parameter, so `@mcp.tool("lookup")` registers `lookup`. The rule reads only `kw.name`, falls to the function name, and emits `confirmed/key-match` with the wrong key (`a0` is ignored). The S14 corpus has only `args []`, so the walkthrough hides it. **Fix:** extract `name` from `kw.name`, then `a0`, then the target; the format needs a first-non-empty expression (`firstOf`), or `anyOf` with two rules.

### M-3. The FastAPI pack emits `confirmed` keys it cannot know, and §2.3's "S14 has no case" is false (§1.6 `roots`, §2.3, O-3)
*Verified.* (a) `falkor-chat/server/tests/test_app.py:835-845` builds `inner`/`outer` `APIRouter()`s inside a function that also calls `FastAPI()` and mounts `outer` under `/shop`, `inner` under `/api`. The `roots` rule (parent contains `FastAPI(...)`) yields `how: app`, no prefix: S14 holds provider `GET /join` `confirmed/key-match` (`facts.json`: `inner.get`, `test_app.py`), while the real route is `/shop/api/join` (the test asserts it). Two more variable-router cases exist (`test_storefront_api.py:964-970`, `test_input_bound_parity.py:818-834`). (b) `APIRouter(prefix="/v1")` inside a mounted factory: ran FastAPI; `include_router(build(), prefix="/api")` serves `/api/v1/x`; the rule emits `GET /api/x`, `confirmed`. No slot is "unresolved", so N-17's guard does not catch either. **Fix:** replace the §2.3 sentence with the three real cases; make `roots: how: app` apply only when the parent holds no `include_router` call, and give a `constructorPrefix` declaration (or mark any provider whose router constructor has a `prefix` argument `possible`); add the cases as S31 negatives against the paper pack (the pack must not emit `confirmed` there).

### M-4. O-6 (topic direction) is a real contradiction, and the spec overstates its authority (§5, O-6)
*Verified by reading the sources.* Proposal §3.3 "Shared state: one pattern (B1)" and graph note §6.2: publishers `WRITES`, subscribers `READS` a topic, with no `Link`. Spec3 §7.2 says only that `config-derived` is "on `CONSUMES topic`"; it nowhere says a publisher `PROVIDES`. The spec's "as bundle spec §7.2 does" for the provider side is inference. AC-23 ("publisher↔consumer *links* appear") favours the `PROVIDES`/`CONSUMES`+`Link` shape. The two models differ in edges, whether a `Link` exists, and traversal (graph note §6.2 table), and §5.2 step 4 and both packs' `link:` entries depend on the choice.
**Ruling:** real contradiction, not a drafting slip, so O-6 is correctly flagged; it is not a blocker for the S31 schema (all four verbs are in the format) but S32 must not freeze `kafka` and `paper-inhouse-bus` emission verbs until the architect rules (a ruling, then a spec3 §7.2 cell change in a `-spec5`, is the cheap route). Route to `architect` via `teco` before S32 dispatch.

### M-5. There is no machine-readable reason registry, so N-9/N-17 and O-2 are mis-owned (§1.5, §10, §12.1, O-2)
*Verified (`grep` of `cpg/ccg/*.py` and the schema).* §12.1 names "S10's `cpg/ccg/bundle.py` registry". `bundle.py` has none; the bundle schema enumerates only the "bundle: yes" codes. `key-match`, `lifted-role`, `partial-key`, `unknown-key`, `config-derived`, … exist only as spec3 §7.2 text. N-9 ("reason not in the registry") and N-17 ("status agrees with the code's registry status") cannot be implemented without one, and no step creates it.
**O-2 ruling:** `severity-rule:<id>` is *not* a contradiction with the freeze: spec3 §7.4 says "S30's severity rule table adds the others, under the same registry rules". It is an authorized extension; only the form (§7.1: parameterized codes "only the forms listed here") needs recording. **Fix:** S31 owns `cpg/ccg/` registry data (seeded from spec3 §7.2/§7.3 plus `severity-rule:<id>`, 16 ids, with each code's status and `On` set), parsed-from-spec in a test the way S10 does; reword O-2 and the §12.1 row; add the file to the S31 row in p0-2 §3.
Related, minor: spec3 §7.2's `On` column lists `key-match` for `CONSUMES`/`Link` and `partial-key` for `Link` only, yet the packs emit `PROVIDES … key-match`/`partial-key` (§2, `unmounted`) and `CONSUMES … partial-key` (§1.4). The slice did the same (D4: "the recognizer's code"). The new registry should carry the real `On` set.

### M-6. §9 has internal inconsistencies, a glob outside the grammar, and vectors that cannot detect the caps (§9.3, §9.4, §9.6)
*Verified (recomputed all 10 vectors: they hold; then probed the rows).*
1. AUTHZ base: "network × write/delete/privilege change = 3" (any write), but AUTHZ-1 requires "a write on a `sensitive` entity". A network write to a non-sensitive entity has base 3 by the formula and falls to AUTHZ-4 by the row text.
2. RCFG-3 lists "an *exposed-app* entry" without the dev exclusion; the formula gives exposed-app in a dev file 1-1 = 0 (RCFG-4).
3. The dev variant glob `**/*{dev,local,test,override}*` uses brace alternation, which §1.3's glob grammar lacks (matches nothing as written). Expanded, `**/` needs a directory, so root-level `docker-compose.dev.yml` (the file in vector 6) does not match; and the substring pattern matches `compose.latest.yml`, `developers.yml`, `localization.yml`, `contest.yml` (ran it).
4. No vector exercises the example-file cap, the SEC +1 flow adjustment or its cap of 3, or a removed-secret-under-tests floor, so a mutant that ignores any of them survives all 10 vectors. Add one vector per adjustment.
**Fix:** align the AUTHZ-1 text with the formula (or restrict the base to `sensitive`), add the dev exclusion to RCFG-3 or drop it from the formula, replace the glob with a regex on basename tokens, add the four vectors.

### M-7. Bundle query v1: the spec's own example violates its join rule, and the cost bound is not enforced (§1.7, §6, N-14)
*Verified for the first point (read §1.7 against the example); inferred for the second.* Joins are "on id-typed fields only (`id`, `symbolId`, `targetId`, `fileId`, `point`, `cond`, `use`, `def`)", but the example joins on `$h.parentId`. Implemented from the list, N-14 rejects the pack S32 copies verbatim. Separately, nothing requires every binding after the first to be joined to an earlier alias, and the 20,000 cap is on output rows: two unjoined bindings over `callsite` (18,129 in the larger bundle) form 3×10⁸ intermediate rows before the cap trips, which undercuts the stated "cost bounded" reason for rejecting jq. Determinism and non-recursion hold (≤4 bindings, no aggregation, rows sorted by canonical JSON). **Fix:** add `parentId` (and `callSiteId`, `unitId`) to the list; require each later binding to carry at least one join equality, and apply the cap to intermediate rows; add both as N-14 cases.

### M-8. The `data-scientist` consult the S30 row requires was not held, and the author recommends skipping it (O-1)
The S30 row's gate is `an (+ ds consult)` and p0-2 §9 (gate table) names the consult as part of gate type C. The spec cannot waive it ("recommend §8 only, at P5"). §8 is a signature set never run against a failure, so a methods look at the classes, runtime narrowing and the negative-control set is cheap now. Not a blocker for starting S31; **it is a blocker for closing S30.** Owner: `teco` (schedule or record the waiver with the stakeholder).

## Minor

- **m-1. `fetch` lifting emits more than §1.6 says.** §1.6: a lifted slot emits "if it resolves and has no lifted child". `assemble.py:385-409` emits every resolved slot and skips only unresolved ones that have a child. S14 has zero resolved-with-child slots (ran it), so equality holds; a partially resolved intermediate slot would differ. Add to O-13 and decide which is intended.
- **m-2. A positive control that cannot pass (§10 tail).** "A variation of each [of §2-§4] with one rule removed is valid" while "a pack with no `rules` is valid only for `severity` and `infra-signatures`". `fetch` has one rule; its variation has none.
- **m-3. Negatives enumerate cases, not axes (§10).** N-21 tests an unknown key at the top level only. Require a generated probe: for every object in the schema, one planted unknown key; for every enum and every typed leaf, one out-of-range value (`Always check`, schema-walking guards).
- **m-4. `keyForm: glob|regex` and keyless `CONSUMES` have no home in graph model v1.** D4 gives `ContractPoint {cpId, kind, key, keyNorm}`, and a pattern key is indistinguishable from a literal one; a `config-derived` consumer with "no key" (§5.2) has no `ContractPoint` to hang on. Route to `graph-dba`/S17 as an open item; not blocking S31/S32.
- **m-5. `config-derived`/`partial` semantics are not kind-limited.** Registry: `config-derived` is on `CONSUMES topic` only; `onUnknown: config-derived` is accepted for any kind in §1.4. Add a validator check.
- **m-6. Path comparison ignores F-13 (§1.6 `mountPaths`/`urlBindings`).** The pack never normalizes (the core does, once) so F-13 holds for emitted keys. The binding step compares a consumer URL path to a mount path by `prefix`, with no statement that both pass spec3 §6 steps 4-6 first; `http://h/mcp/%2E%2E/admin` would prefix-match `/mcp` on raw text. State that both are normalized before comparison. *Inferred, not run.*
- **m-7. `envPorts` heuristic.** `AGENT_TEAM_FALKORDB_PORT` is adopted (ran the two regexes); `excludeVars` is anchored at the start. Acceptable as documented; note it in "Name heuristic".
- **m-8. `status: paper` vs closed keys.** p0-2 §2 layout says `packs/<name>/pack.yaml` carry `status: paper`; §1.1 has no such key and N-21 rejects unknown keys. Add an optional `status` to §1.1 or drop it from the layout.
- **m-9. Contact list wording (§11).** The `.mcp.json` entry `falkor-chat-agent-team` is listed as "informative, not anchors" while §8.2 says a walked unit contacts its server-level wiring items. Consistent with eval-protocol §9.1 (it is one of falkor-chat/server's four dev entries), but say so once in row 4 so S47 does not read the exclusion as a ruling.
- **m-10. §8 nits.** The §8.1 `signatures:` block contains a placeholder entry (`"<runtime>-<class>…"`) that S32 must not copy; ids `*-third-party-*` break the stated `<runtime>-<class>[-<library>]` grammar; `python-timeout`'s `timed out` also fires on the JVM and .NET samples when no runtime is detected (ran the cross-match), as the "all are emitted" rule allows.
- **m-11. Kafka `produce`** matches any call named `send` in a file that imports `kafka`; add to "Known false matches". `@router.get(path="/x")` (FastAPI accepts `path=`) yields no provider; add `kw.path`.
- **m-12. Housekeeping.** O-11 is answered: FR-1 (`docs/requirements/connected-code-graph.md:95`) lists the seven languages as §7 assumes; close it. O-10 (where the pack-set identity is recorded) has no owner. The query block is duplicated verbatim in §1.7 and §6 (one copy should cite the other). The proposal needs its `Extended by:` pointer when this lands (p0-2 §10, `Extended by:` row), a `teco` task. No growth-stops issue: this is v1.0, not a revision.

## What's solid (verified, so it need not be churned)

- FastAPI/fetch equality: `assemble.py` on `s14a` gives 61 providers, 11 consumers, 11 links. The spec-rule re-derivation (regex + `fastapi` import gate) selects the same 62 decorators as the slice (61 keys after one `drop`), and all 62 and all 7 `include_router` sites sit in `fastapi`-importing files, so the gate is output-neutral. The walked rows (§2.1, §2.2, §3.1 incl. the four `liftedVia` hops and `a0.method` verb path, the three 5-hop chains) match the records.
- §4.1: 25 `mcp.tool` in `mcp.py` plus 3 in 2 test files; §4.2: `app.py:543` mount `/mcp` under `mount_mcp` (542), `:528` and `:551` do not feed from FastMCP; every cited source line at `defca60f` is accurate; the `bindValue` name-rule claim holds.
- All 12 YAML blocks parse; ids and directory names agree. Section 8: 45 signatures, all compile, all samples match, no shared negative control matches, no length >256 or RE2-incompatible construct, five runtimes by six classes plus `any`; runtime detection picks the right runtime on one line each.
- §9 INJ/AUTHZ/RCFG/SEC bases, penalties and rows recompute (16 ids; 10 vectors reproduce) apart from M-6.
- Contact list rows 1-2 equal the slice report's #1-#2; avoiding `cypher-mcp` follows eval-protocol §15 Q10/R9; the wiring-cost claim is consistent with §9.1. The pack never normalizes keys (F-13 clean for emitted keys).
- The relational-select query form is a sound choice over Cypher/jq/code hook; the rejection reasons are right.

## Open questions

1. (`teco` → stakeholder) Waive the `data-scientist` consult on S30, or hold it before close (M-8)?
2. (`architect`) Topic direction ruling (M-4): which model, and does `config-derived`'s `On` cell change?
3. (`architect`/freeze owner) B-1 option (a) or (b)?
4. (`teco`) Add the registry file to S31's row (M-5), and a route for m-4 to `graph-dba`.

### Proposed REVIEW.md lines

Target: root `REVIEW.md`, heading *Always check*. Source: B-1, M-1, M-2 (this review).
- "A walkthrough over **paper records** re-derives each record through the frozen contract that produces it (here the argv gate), and runs every matcher a pack declares (glob, regex, basename) against the real record text it is claimed to match. A pack that cannot recognize its own walkthrough input is a finding."

## Appendix — evidence for B-1, M-1, M-3

**B-1.** `git show defca60f:falkor-chat/scripts/start_server.sh | sed -n 251p` → `exec "$VENV_DIR/bin/uvicorn" falkorchat.app:app $UVICORN_ARGS …`. Spec3 §1.6 `command` row; §1.6.1 "`command` (argv[0]) supplies none either: it is judged by (v1) and (v2) alone"; (v1) "1-128 characters from `[A-Za-z0-9._/:@+=,-]`, which fails `$NAME`/`${…}`".

**M-1.** Real record: callsite `b5d47feb`, `app.py:413`, `externalTarget` `mcp/server/fastmcp.py:<module>.FastMCP.streamable_http_app`. Glob per §1.3 (`*` = `[^/]*`, `**` = `.*`, whole string): `*FastMCP.streamable_http_app` → no match; `**FastMCP.streamable_http_app` → match.

**M-3.** `facts.json` (assemble over `s14a`): `inner.get GET /join` `test_app.py` `confirmed/key-match` `['app']`; `extra.get GET /twelfth` `possible/partial-key` `['unmounted-router']`; `router.post POST /probe/{}` `test_input_bound_parity.py` `confirmed` `['app']`. FastAPI run: `app.include_router(build(), prefix="/api")` with `APIRouter(prefix="/v1")` → `openapi()['paths'] == ['/api/v1/x']`.

## Pass 2 (spec v1.1, `cpg/tests/test_rule_pack_spec.py`)

Scope: v1.1 (1,387 lines, 104 KB) plus the test file (52 tests) and `cpg/tests/data/infra_captures.tsv`. Baseline suite: `tests/test_rule_pack_spec.py` 52 passed; full `-m "not live"` suite 1,452 passed. Scratch mutants are in the session scratchpad (`.../scratchpad/mut/`; spec copies, test copies patched to read them; nothing in the repo was edited).

### Dispositions of Pass 1

- **B-1 fixed (re-derived).** `gate_argv` on the L1 argv returns it unchanged; on L2's argv returns `['<redacted>', 'falkorchat.app:app', '<redacted>', '<redacted>']` (4-word shell tokenization: `$UVICORN_ARGS` and the `${LOG_CONFIG_ARGS…}` word are two redactions); `bind_value_allowed(C1)` is true. §4.3/§4.4 now say so, step 3 "adds nothing", step 4 rests on L3's name plus L4's unit. The text of §4.3 itself is not tied to the test (mutant M2 below).
- **B-2 fixed in text.** `basename` is in §1.3, `urlBindings.scope` moved to the `callsite` source, `formula`/`rows` have a shape (§9.1), `args.value`, any-of, `anyOf` YAML, enum members, the `handler` column and `finding.class` are defined. The claimed *proof of closure* is not (N-1).
- **M-1 fixed.** Pack glob is `**FastMCP…`; mutant M4b (pack reverted to `*`) turns the suite red.
- **M-2 fixed in text** (`firstOf: [kw.name, a0, target]`; `kw.path` for routes); **no test exercises it** (M5 survives).
- **M-3 fixed in text; claims verified by running.** FastAPI 0.139.0: `_probe_rest_app` serves `/probe/{pid}`; nested `inner`/`outer` serves `/shop/api/join`; `APIRouter(prefix="/v1")` in a factory served at `/api/v1/x`. I recomputed `roots.notWith` over `s14a`: exactly two providers change, `inner.get GET /join` and `_probe_rest_app.probe`, and no S14 `APIRouter` call has a `prefix` argument, so `constructorPrefix` changes nothing. Equal to the spec's claim. Side effect (conservative, fine): a correct `@app.get("/real")` in a function that also builds a local `APIRouter()` is demoted to `possible` (ran it).
- **M-4 re-ruled by the coordinator (Option A)**; judged under N-2.
- **M-5 fixed in text** (§1.9 registry file and rows; N-27). The p0-2 S31 row still lacks the file (a `teco` edit the spec names). Residual: N-3.
- **M-6 fixed.** AUTHZ base is any write; the dev penalty is a basename-token regex (V20/V22/V25/V44 cover root-level, `latest`, `contest`); 44 vectors; the test mutates every number, `when` and the operation order, and my mutant M6 (a base 3→4) turns it red.
- **M-7 fixed in text** (`parentId`, `callSiteId`, `unitId` listed; joined bindings; intermediate cap; N-14). The id list is prose only in the spec (M7 survives; minor).
- **M-8 closed** by `docs/reviews/connected-code-graph-rule-pack-spec-ml.md` (its Pass 2 verdict: approve with suggestions); not re-judged here.
- **Minors.** m-1 stated in §1.6/O-13 (fixed); m-2 fixed (`rules` optional for every kind); m-3 prose fixed, probe not delivered (N-1); m-4 is O-15 (open, matters under A, N-2); m-5 fixed (N-27); m-6 fixed (verified: `normalize_path("/mcp/%2E%2E/admin")` = `/admin`; prose only, no test); m-7, m-8 fixed; m-9 fixed; m-10 fixed (§8 reworked, ML review); m-11 fixed (`kw.path`); m-12 fixed (O-11 dropped, O-10 owned, one query copy).

### My mutants on the test file's claims (10 spec edits; 5 survive)

| # | Edit (to a spec copy, test unchanged) | Result |
|---|---|---|
| M1 | `roots.notWith: [{callee: APIRouter}]` → `[{callee: Zzz}]` | **survives**: the test hard-codes `"APIRouter" not in names` and never reads the pack |
| M2 | §4.3 L2 `command <redacted>` → `command uvicorn` | **survives**: the test uses its own argv, not the table |
| M3 | `constructorPrefix … arg: kw.prefix` → `kw.nothing` | **survives** (no fixture; the spec defers it to P1) |
| M5 | `firstOf` loses `{arg: a0}` (re-opens M-2) | **survives**: S14 has only `args []` |
| M7 | drop `parentId` from the §1.7 join-id list | survives (prose; the test's list is its own) |
| M4b | pack glob back to `*FastMCP…` | caught |
| M6 | severity base 3→4 | caught |
| M9 | drop `provisional` on a Kafka emission | caught |
| M10 | registry: `key-match` loses `PROVIDES` | caught |
| M11 | `python-oom-kill` regex `\bMemoryError\b` → `Error` | caught (negatives and cross-runtime) |

Also run: M12 (`go-timeout` loses its `i/o timeout` branch) survives, since only breadth is guarded; M13 (`any-oom-kill` loses the 137 branch) is caught by the recall cases.

### New findings

**N-1 (major). The test file's "proof that the format is closed" is open, and three Pass 1 fixes are prose-only.** Verified: I planted one `zz_planted` key at every dict of the 7 pack blocks (268 positions) and ran `check_pack`; **73 accepted** (53 inside `formula`, 9 under `mounts`, 5 `queries`, 3 `urlBindings`, 2 `launch`, 1 `liftings`). Also accepted by `check_pack`: `status: live` (§1.1 allows only `paper`), a `framework` pack carrying `formula`/`signatures`, a severity pack declared `kind: framework`, a severity pack with `rules`, `base value: 9`, `when: {source: nowhere}`, a missing `secrets` formula, `notWith` with two matchers, `constructorPrefix.arg: p0`. §10's N-26 ("every object, every enum and typed leaf") is realized as one planted key on one object. Survivors M1-M3 and M5 above are the other half: the `notWith`, redaction, `constructorPrefix` and `firstOf` fixes exist only as spec prose and in a test that re-implements them. Because this file is what S31/S32 are told to mirror, an open checker propagates. **Fix:** (a) generate the N-26 probe (planted key at every dict of every pack, out-of-range value at every enum and typed leaf) and assert all rejected; add the §1.1 kind/body-key rule and the `when` vocabulary to the checker and to §10; (b) read `notWith`, `constructorPrefix` and the glob from the pack in the roots test; (c) drive the §4.3 table from the spec text; (d) add a planted-record test for `@mcp.tool("x")` and `@router.get(path="/x")` (the `eval_query` helper shows the pattern) and a `constructorPrefix` fixture, or move all of those to a named P1 test in §12.1. This is root `REVIEW.md`'s schema-walking line, unmet.

**N-2 (major). Option A holds only with three additions; the markers cannot simply be dropped.** Verified by reading §1.4, §1.9, §5, §6, spec3 §7.2, graph D4, and grepping the two B1 copies. What holds: registry codes (`key-match` on `PROVIDES`/`CONSUMES`/`Link`, `config-derived` on `CONSUMES topic`, `unknown-key` on `Link`) match A without a `-spec5`; AC-23 reads as written; publisher = provider and subscriber = consumer map onto the contract-path traversal. What does not: (1) **a publisher whose topic comes from config or a call** has no outcome: `provides` drops it (`onUnknown: drop`, §1.4) and no code covers `PROVIDES` (`runtime-built` is on `CONSUMES`/`READS`/`WRITES`, `config-derived` on `CONSUMES topic`), so the consumers' links to it vanish silently, against AC-6's "each shown as possible". Under B the same case is a registry-backed possible `WRITES`. Either add `PROVIDES topic` to those two `On` sets (§1.9 table) or state the asymmetry in §5. (2) **O-15 becomes a precondition**: §5.2 step 2 emits a keyless `CONSUMES`, step 3 and §6 emit pattern keys (`regex`, `glob`), and D4 gives `CONSUMES` only `→ ContractPoint {cpId, kind, key, keyNorm}`. Under A, S32 asserting those facts needs `graph-dba`'s answer first. (3) **The edit list** to apply with the ruling: delete `provisional` from §1.5 and its checker enum; delete the five `provisional: topic-direction` tokens (§5 ×3, §6 ×2), the §5 "Topic direction is an open ruling" paragraph, the S32 exception in §12.1, and O-6; edit proposal §3.3 line 550 and graph note line 634-635 (the only two copies of "publisher WRITES"; grep `docs/`); say in the proposal that B1's "one pattern" now excludes topics.

**N-3 (minor).** `example-config` is `Link`/`BOUND_BY` in spec3 §7.2 and §1.9 does not add `Finding`, yet §9.1 gives a finding in an example file "`possible` with reason `example-config` beside its `severity-rule`": two reasons, one slot, and an `On` set that excludes `Finding`. Add `Finding` to §1.9's table or drop the sentence.

**N-4 (minor).** One fact, five homes: "the gate redacts a `$`-bearing `command`" appears in §1.3 (`basename` row), §1.6 (`frameworkDefaults`), §1.8, §4.3 and §4.4. Keep §4.3's and cite it (root `AGENTS.md`: a fact has one home). Side effect of the B-1 fix: `targetArg` with no `command` constraint accepts `args[0]` of any command launch shaped `x:y` whose `x.py` exists in-repo (`docker run app:latest` has `args[0]` = `run`, so unlikely, but `tool app:latest` would resolve a unit); bound it with `command` where the pack knows the launcher, or say so.

**N-5 (minor).** `p0-2` §3's S31 row has no `cpg/ccg/registry/reasons.yaml`; the spec marks it a `teco` edit and S31 must not be dispatched before it. The test file also holds a second closure checker (`check_pack`) beside the JSON Schema S31 writes; say which wins when they disagree, or have S31 replace it.

**N-6 (minor, test strength).** `matches()` has no `template` branch and does no path normalization, so §1.3's F-13 sentence is untested; `go-timeout`'s alternation branches have no per-branch positive (M12 survives).

### Growth-stops check (root `AGENTS.md`; v1.1 is a revision of a not-yet-approved document)

83 → 104 KB. v1.0 was never committed, so I cannot diff; by section the growth is: new scope (reviews asked for it) §9.1 formula, 44 vectors and mutation checks (about 12 KB now), §8 captured samples/near-miss negatives/mutation list (about 19 KB now), §1.9 registry, N-24..N-28, the `notWith`/`constructorPrefix`/`firstOf`/`basename` declarations. Rework growth: the body states the current design only, with no finding table, no dispositions and no history except the one-line revision note (allowed); the two reviews are cited by path in the header, not by ID. Restated facts: N-4 (five homes); the duplicate query block was removed (§1.7 cites §6). No rework growth to report beyond N-4.

### Verdict (Pass 2): **needs changes**, one narrow round

Both Pass 1 blockers are fixed and re-verified. Two majors remain, and neither is a design reversal.

**Stopping condition (falsifiable): approve when all four hold:**
1. A re-run of my probe (a planted key at every dict of the 7 pack blocks) accepts **0** of the 268 positions, and the `status: live`, `base value: 9`, `kind: framework` + `formula` and `when: {source: nowhere}` packs above are each rejected by `check_pack`.
2. Mutants M1, M2, M3 and M5 (edits in the table above) each turn at least one test red.
3. The spec states Option A with the N-2 (1) disposition for publisher-side unresolved topics, O-15 named as the precondition for S32's topic assertions, and the markers removed per the edit list (`grep -n provisional docs/plans/connected-code-graph-rule-pack-spec.md cpg/tests/test_rule_pack_spec.py` returns nothing; proposal line 550 and graph note line 634 no longer say "publisher writes").
4. `example-config`/`Finding` (N-3) is consistent between §1.9 and §9.1.

N-4 to N-6 are suggestions and do not gate. No `REVIEW.md` line is proposed: N-1 is covered by the existing schema-walking line.

## Pass 3 (spec v1.2, 106,478 B / 1,415 lines; `cpg/tests/test_rule_pack_spec.py` 1,769 lines)

Suite: `cd cpg && .venv/bin/python -m pytest -q tests/test_rule_pack_spec.py` = 101 passed (re-run, 1.0 s). Mutants ran on copies under the session scratchpad (`.../scratchpad/w/`: spec copy with one edit, test file unchanged; scripts `m1.py`, `m2.py`, `probe.py`, `probe2.py` beside it). Nothing in the repo was edited.

### Dispositions of Pass 2

- **N-1 fixed (stopping conditions 1 and 2 met, re-run by me).** My own probe plants `zz_planted` at every dict of every pack, 313 positions (the 268 pack-block dicts plus the 45 signature entries), each with six value shapes (1, "x", `{exact: x}`, `[]`, `true`, `null`): 1,878 mutants. With value `1`, 0 of 313 accepted. With a string or matcher value, **4 accepted**, all `queries[].from[].where` (see P3-2). `status: live`, `base value: 9`, `kind: framework` + `formula`, `when: {source: nowhere}`, severity-with-rules, `notWith` with two matchers, `constructorPrefix.arg: p0` are each rejected (the 28 `RANGE_NEGATIVES` run green). The probe can fail: its meta-test and the vacuity guard (`tried == independent YAML-mapping count`) hold.
- **Survivors M1, M2, M3, M5: all four now red** (table below). M3 and M3b (callee) are caught by the new fixture test, M5 by the `@mcp.tool("x")` test.
- **N-2 fixed in text, with two stale copies left (P3-3).** Option A is in §5 with preconditions O-15 and O-16; `provisional` is gone (`grep -ni provisio[n]al` on the spec and test: nothing; the test asserts it); proposal line 550 and graph note line 634 now say `PROVIDES`/`CONSUMES` (checked against `git diff`).
- **N-3 fixed.** `Finding` is in `example-config`'s `On` (§1.9) and §9.1 says so; M25 (revert it) turns `test_example_config_...` red. Residual in P3-5.
- **N-4 fixed** (one home in §4.3; `targetArg` states the no-`command` consequence). **N-5 open as designed**: `p0-2` S31 row still lacks `cpg/ccg/registry/reasons.yaml` (`grep reasons.yaml docs/plans/connected-code-graph-p0-2.md`: nothing; `p0-2` is unmodified); the spec marks it a `teco` edit. A precondition for S31 dispatch, not a spec defect. **N-6 fixed** (`template` branch and `normalize_path` are tested; each `go-timeout` branch has a positive).

### Mutants run (45 spec edits; the four named first)

| Edit to the spec copy | Result |
|---|---|
| M1 `notWith: [{callee: Zzz}]` | **caught** (roots test reads the pack) |
| M2 §4.3 L2 `command <redacted>` to `command uvicorn` | **caught** (argv-gate test reads the table) |
| M3 `constructorPrefix.arg: kw.nothing`; M3b `callee: Zzz` | **caught** |
| M5 `firstOf` loses `a0`; M5b loses `kw.name`; M5c loses `target` | **caught** |
| M6 route `firstOf` loses `kw.path`; M7 `roots` callee; M17 tool regex; M22 query join; M25; M26; M27; M31; M33; M35; M15/M15b/M16 (`envPorts`) | caught |
| M5d `firstOf` reordered `[a0, kw.name, ...]` | survives; equivalent (FastMCP raises if both are given) |
| M8 `unmounted` to `confirmed/key-match` (the registry agrees with itself) | survives |
| M9 mount `prefix.default`; M28 mount `callee`; M29 `child.arg`; M10 `pathMatch: exact`; M11 `bareDefault: POST` | survive |
| M19 `portFlags` flag; M20 `frameworkDefaults` port 8001; M18 `urlBindings` keyPath regex | survive |
| M12 / M13 / M30 `scope.imports` module of `fastmcp`, `fastapi`, `tool-call`; M14 / M34 / M32 Kafka and route regex alternatives; M24 Kafka `onUnknown: config-derived` to `runtime-built` | survive |

### Findings (new)

**P3-1 (minor). Declaration leaves of the real packs are still guarded by restated constants, not by the pack.** Verified: 16 of the 45 mutants survive (list above); each is a pack value that a walkthrough sentence or §12.1 claim depends on, and the tests either hard-code the same literal (`"include_router"`, `["fastapi"]`, `{"mcp.server.fastmcp", "fastmcp"}`) or never read it. This is M1's class, left open outside the four named. It does not gate: S31 extracts the packs mechanically from the spec, so the packs cannot drift from the blocks; what is unguarded is block against prose. **Improvement:** a generated leaf probe, not a longer list. For every scalar leaf of the three real packs (fastapi, fetch, fastmcp), mutate it in the spec text and require a red test, or require the leaf's path in a named `UNEXERCISED_UNTIL_P1` set in the test with the §12.1 conformance row that will cover it. A new leaf then fails closed. Place the probe in `test_rule_pack_spec.py` as its own test; its obligation (every leaf classified) is the test's, so the spec does not need to change. `unmounted` (M8) wants one assertion: every `unmounted` is `possible`, since §2.3 and §12.1 say so.

**P3-2 (minor). The generated probe's second half, an out-of-range value at every enum and typed leaf, is hand-listed (N-26 says generated).** Verified with `probe2.py` (replace each scalar leaf of the six non-signature packs by `null`, `{zz: 1}`, `[]`, `99999`, `"zz bad!"`; 1,615 mutants): accepted where a value set exists in the spec: `emit.*.kind` (N-10: not core and not declared in `vocab`), `unmounted.reason` and emission `reason` (N-9: not in the registry; only the registry test looks at them), `lifted.lifting` (an id no `liftings` entry declares), `queries[].from[].record` (not a spec §5 record type), `decorator.form`, and `where` field names (4 accepted planted keys: a misspelled field is a silent never-match). Free strings (keys, regex text, examples) accepted are expected. **Improvement:** fold N-9, N-10 and the `lifting`/`record`/`where`-field membership into `check_pack` (they are also the S31 validator's obligations), and generate the out-of-range probe the way `plant_everywhere` does, with the free-string leaves on an explicit allow-list. Duplicate YAML keys are also invisible to `safe_load` (checked: the spec has none); S31's loader should reject them.

**P3-3 (minor). Two stale statements of the old topic model remain in the proposal.** Verified by `grep -in topic docs/plans/connected-code-graph.md`: line 173 (the B1 row) still says "shared state (data, topics, fields, config values) is one pattern", and line 1260 says the linker "treats data, topics, fields and config values as shared state". Both contradict line 550 and the spec's "Topics are not B1 shared state (proposal §3.3 ... say so)". The graph note line 620 ("hot table or topic", writers of one entity) is mild. The `publisher WRITES` sweep missed them because the model lived in a different phrase. **Fix (owner `architect`/`teco`, proposal edit):** drop "topics" from both; check graph note line 620. Spec §5 cites "proposal §3.3" correctly.

**P3-4 (minor). O-16's owner disagrees with the ruling route.** The brief makes the `PROVIDES` cells on `runtime-built`/`config-derived` a `teco` ruling (the S31 registry cells); the spec says "owner: the freeze owner" and recommends a contract-document note. State `teco` as the owner of the ruling and S31 as the implementer of the cells, so S31 is not blocked on an unnamed owner. If ruled yes, `check_rule`'s N-27 and §1.4 ("valid only on a `consumes` of kind `topic`") change together (the `provides` default `onUnknown: drop` becomes `config-derived`/`runtime-built` for topics).

**P3-5 (minor). A `Finding` now has two reasons and nothing records the shape.** §9.1 says "a second reason on the finding". The Finding label is deferred in the graph model (`p0-graph` line 112), and N-17's "status agrees with the code's registry status" has no rule for a finding whose reasons are `severity-rule:<id>` (status `none`) and `example-config` (`possible`). Add one open item: the Finding record carries a reasons list, and the finding's status is the strongest-uncertainty reason's; owner `graph-dba` with O-15.

### The tdd-engineer's four reported ambiguities (rulings)

1. **"Supported" definition: does not block; fix the spec's wording (spec is wrong, test is right).** §8.1 says "every top-level `|` alternative"; by that rule 6 captured entries are wholly supported (I counted with `sup.py`: `python-third-party-redis`, `python-third-party-httpx` and `jvm-connection-refused` each have one top-level alternative and alternation inside a group), while §8.2 and the test claim 3. The test counts `|` at any group depth, which is what a one-line sample proves. **Edit:** "every alternative of its regex, at any group depth".
2. **Third-party dedupe: does not block.** "Dedupe, exactly" plus the `pairs_of` tests pin it (library added to every other-class pair; own pair only when no other class matched; block and line occurrence in the key). Residual wording: `line` in `(block, line, class)` means the line's occurrence, not its text (the test asserts two identical lines are two pairs); say it.
3. **Near-miss wording: does not block.** The behaviour is tested (no entry matches any near-miss, at least 3 per class, class set equal). Two near-miss lines do not "carry the trigger word in application text" (`connection pool reset scheduled`, `Network is unreachable (cached)`, which guards a removed bare form); reword to "lines that a looser signature would match".
4. **`when` vocabulary: does not block approval; add the table before S31 dispatch.** The vocabulary (feature, its kinds, its values) lives only in the test (`FEATURES`, `KIND_FEATURES`), derived from §9.1 to §9.3 prose and vectors. S31 is told to build from the spec, so the spec should hold the six-row table and a test should assert it equals `FEATURES`. A one-table edit, not a design change.

### Growth-stops check (root `AGENTS.md`; v1.2 is a revision of an unapproved document)

104 KB to 106.5 KB (+2.4 KB). By section: new scope (Option A paragraph, O-15, O-16, `notWith`/`constructorPrefix`/`firstOf` text, the §1.9 `Finding` cell); no finding table, no disposition list, no history beyond the single dated revision line; findings are cited by path in the header, not by ID in the body. The tests' prose comments name N-ids of the spec's own negative table, not review findings. Restated facts: none new. Passes.

### Verdict (Pass 3): **approve with suggestions**

All four stopping conditions hold: (1) 0 of 313 planted-key positions accepted, and the named out-of-range packs rejected; (2) the four named survivors are red; (3) Option A with the unresolved-publisher disposition, O-15 and O-16 as preconditions, markers removed (P3-3 and P3-4 are the leftovers); (4) `example-config` on `Finding` consistent between §1.9 and §9.1, test-guarded. No blocker or major. S31 may be dispatched after `teco` adds the registry file to the `p0-2` S31 row (N-5) and the spec takes the one-table `when` vocabulary and the "supported" wording fix (rulings 1 and 4). P3-1 and P3-2 are for the S31/S32 tests to absorb as generated probes.

**Open questions.** (1) `teco`: O-16 ruling (PROVIDES on `runtime-built`/`config-derived`). (2) `graph-dba`: O-15 and P3-5.

### Proposed REVIEW.md lines

Target: root `REVIEW.md`, heading *Always check*. Source: P3-3 (this review).
- "A ruling that reverses a model is propagated by grepping the **concept** (every document naming the subject: `topic` beside `shared state`), not the old verb phrase; a sweep for `publisher WRITES` found two copies and missed two others."

## Pass 4 (S31 delivered, commit `51f7c95b`: schema, `ccg/rulepack.py`, `ccg/registry/reasons.yaml`, three packs, two tests)

Scope: the diff of `51f7c95b` against spec v1.3 (accepted), p0-2 S31 row, root `REVIEW.md`; no component `REVIEW.md` exists. Spec and spec3 are untouched by the commit (`git diff HEAD` on both: empty; the commit's file list names neither). CPG: considered, not relevant — a Python validator/schema/YAML change reviewed against the spec text and by running mutants; no loaded graph indexes `cpg/ccg/`.

**Verdict: approve with suggestions.** No blocker. One major (an installed wheel loses the registry), four minors (two of them are tests that cannot fail, one is a validator reference gap the harness misfiles as P1-only).

Checks run (observed): `cd cpg && .venv/bin/python -m pytest -q tests/test_rule_pack_schema.py tests/test_rule_pack_spec.py` = 611 passed (22 s); `CCG_LEAF_MUTATION=1 pytest tests/test_rule_pack_leaf_mutation.py` = 3 passed (76 s; the 84 killed / 25 survivors classification holds). My own N-1..N-28 script (`scratchpad/neg.py`, `neg4.py`, built from the N table, not from the test's lists): every negative rejected, N-22 only when the negatives are supplied (see P4-6). Each of the three pack files equals its spec block plus the one `status: paper` line (my own `difflib` over the fenced `yaml` blocks). 41 mutants of `rulepack.py`, `reasons.yaml` and the pack files (one more, the `excludeVars`-required schema edit, was skipped), run on copies under `scratchpad/analyst_s31/` (a copy of `cpg/` with `docs/` beside it; the copy's `ccg` was the one imported, checked): 36 killed, 5 survived: `N` and `R` (P4-4), `a10` (`third-party` id regex loosened to `.*`: the schema id pattern subsumes it except for `python-third-partyx`, a nit), `a7` (an identity edit I wrote, discard it) and `a16` (equivalent: `re.compile` already rejects `\k`/`\g`).

### Judgments asked for

- **Validator vs the spec's `check_rule` on topic (sound).** O-16 is a `teco` ruling that adds the `PROVIDES topic` cells and says §1.4's "valid only on a `consumes`", the `provides` default and N-27 "change with them (S31/S32, for `topic` only)". The validator does exactly that (`default_on_unknown`, `reasons.yaml` rows `runtime-built`/`config-derived`); the http and mcp_tool `provides` cases still reject (N-27a/c run). I read the rows against spec3 §7.2 line by line: all registry codes match plus the §1.9/O-16/§7.4/spec4 additions. **The spec test must not change alone.** `check_rule` (`test_rule_pack_spec.py:376`) is the executable twin of §1.4 line 169 and the N-27 row (line 1342); changing only the test makes it contradict accepted text. Change all four together in the v1.4 wording round. Until then the divergence is recorded only in a docstring and no test fails when it is retired (P4-3).
- **`KNOWN_LAXER` (`envPorts.excludeVars` optional): defensible, but a silent ruling.** §1.6 states no `required` rule; the spec checker indexes it unguarded. The validator's exact-equality assertion (`missed == KNOWN_LAXER`) fails if laxness grows or if the key becomes required, so it is a real pin. But §3 L5 and the §4 soundness paragraph show why an `envPorts` with nothing excluded adopts `FALKORDB_PORT`; optional makes the unsafe pack valid. Ask the architect to rule it in v1.4 (the ledger already lists it).
- **Registry re-derivation test: independent source, not circular.** `seeded_registry()` parses spec3 §7.2/§7.3, §1.9's table and four hand-coded cells from the text; it never reads `reasons.yaml`. Mutants `A1` to `A5` (drop `PROVIDES` from `key-match`, `partial-key` to confirmed, drop `PROVIDES topic`, change a non-pack-edge token, move `lifted-role` to `READS`) are all red on that one test.
- **Silent-argument mutants (required): killed.** Registry `On` token (`A1`, `A3`, `A5`), `LEVELS` order (`D`), `SEV_PREFIX` (`I`), `JOIN_FIELDS` input `parentId` (`E`), partial/consumes/writes default modes (`B`, `C`, `Q`), kind-restriction in `allows` (`J`, `a14`): all red.
- **`fastmcp` has no §12.1 row (confirmed).** The P1 conformance row (spec line 1398) names `fastapi` and `fetch` only; 14 of the 25 `UNEXERCISED_UNTIL_P1` leaves are `fastmcp`'s. Spec v1.4 owes the row; until then those leaves have no test owner at all.

### Findings

**P4-1 (major, verified). `pyproject.toml` omits `ccg/registry/*.yaml`.** `package-data` is `ccg = ["schemas/*.json", "store/*.cypher"]`. I built a wheel from a copy (`setuptools` `bdist_wheel`, scratchpad): the listing has `ccg/schemas/rule-pack-v1.schema.json` and `ccg/rulepack.py` but no `reasons.yaml`. `rulepack._registry()` reads it lazily, so an installed `ccg` imports fine and fails with `FileNotFoundError` on the first pack that names a reason; the editable venv hides it. **Fix:** add `"registry/*.yaml"`. **Guard (derived from the tree, so it can fail):** one test asserting every non-`.py` file under `cpg/ccg/` matches a `package-data` glob of `pyproject.toml`; the planted case is today's tree (red before the one-line fix). Authoring the test and the glob in the same change is fine here because the expected set is the filesystem, not the declaration.

**P4-2 (minor, verified). Two schema patterns are pinned by no test; the test that names one cannot turn red on it.** Of the 29 schema-mutant survivors (builder's `mut5.out`: 118 mutants, 29 survived), I re-derived eight groups as equivalent (below) and found these not equivalent, by planting one input on a mutated copy of the schema: (a) `query.from[].as` `pattern` dropped: alias `TY` (joined but not yielded) and `t y` are accepted; `test_a_query_alias_is_a_lowercase_token` passes under the mutant because its alias is also yielded, where the yield-value pattern rejects it. (b) `query.yield` value `pattern` dropped, or its trailing `$` dropped: `$dec.args.a0!! junk` and `xdec.args.a0` are accepted, since the validator's `args.` branch (`rulepack.py` `query()`) checks only that the record has `args`. A malformed `args.` reference is a silent never-match. `pattern-open-start` on `as` is inferred non-equivalent (not run). **Fix:** one planted alias that is joined but not yielded (`TY`), and two yield references (`xdec.args.a0`, `$dec.args.a0 junk`); each must be rejected, and each of the schema mutants above must turn one red. Count: builder claims 29 equivalent; I count at least 3 non-equivalent (a, b twice) plus 1 inferred.
Equivalent, re-derived (read the validator branch that subsumes the schema keyword, then planted the input): `match` `oneOf` to `anyOf` (the two branches are mutually exclusive, `subjectMatch` is closed and lacks `anyOf`); `from.minItems` (empty `yield` also fails `minProperties`, a non-empty one names an unbound alias); regex `maxLength` ×6 (`regex_problem` runs on every dict with a `regex`, probed with 300 characters); `row.id` pattern ×3 and `row.rank` minimum (`rows()` demands the exact derived id); `signature.id` pattern ×4 (the id is `runtime-class`, both from enums); `minLength` ×3 (`""` is neither a column nor a registry code); `unrequire-kind` ×3 (`kind` is required at the top).

**P4-3 (minor, verified). A `lifted` reference is checked against a fixed enum, not against the lifting; the harness files the leaf as P1-only.** Plant: `fetch` `liftings[0].role.slot: urlZ` is accepted, leaving `rules[0].extract.url: {lifted: url}` dangling; so is `liftings[0].role.kind: topic` against a rule emitting `http`. A renamed `lifted` slot is rejected only because the schema enum is `[url, verb]`. `fetch:liftings[0].role.slot` is therefore not "only a runtime can test it" and belongs in `KILLED`. I read the other 24 survivors (callee names, import prefixes, ports, flags, key literals, `recordConditions`, `prefix.default`, `targetArg.index`): each is a value only a pack reader can exercise; none of them is a static reference. **Fix:** in `_Checker.rule`, require each `{lifted: s}` slot to be the lifting's `role.slot` or one of its `companions[].role`, and a lifted rule's emission kind to equal `role.kind`; then the leaf turns red and `UNEXERCISED_UNTIL_P1` shrinks to 24 (the slow harness asserts it).

**P4-4 (minor, verified). Two validator branches have no test that can fail them.** Mutant `R` (`_clauses` reads only the first `anyOf` clause) and `N` (`slot_mode` takes the last nested `onUnknown` instead of the first) survive all 611 tests. The shipped `R` behaviour is right (an undeclared lifting in the second clause is rejected, run), and `N` is a real ambiguity: `firstOf: [{a0, drop}, {kw.path, partial}]` with `status: confirmed` is accepted today (run), because only the first nested mode is read. **Fix:** a pack with an undeclared `lifted.lifting` and an undeclared `query` in a non-first `anyOf` clause (red under `R`); and reject mixed nested modes in one slot (or state in §1.4 that the first one rules); red under `N`.

**P4-5 (minor, verified). The O-16 divergence is invisible to the differential and is retired by nothing.** The differential walks single edits of the seven packs; none adds `onUnknown` to a `provides topic`, so the one place validator and spec checker now disagree is outside it, and the docstring in `test_rule_pack_schema.py` is the only record. **Fix:** one named test: the kafka pack plus `onUnknown: config-derived` on `produce.topic` is rejected by `spec.check_pack` and accepted by `rulepack`. It turns red when v1.4 changes `check_rule`, which forces the cleanup of the docstring and of `KNOWN_LAXER`'s neighbours. Put it in the S31 round, not in the v1.4 one.

**P4-6 (minor, verified). `load_pack` cannot enforce N-22's negatives.** `load_pack` calls `validate_pack(pack, dirname=...)` with no `negatives`, so the shared and near-miss lines reach the check only when a test passes them (as the schema test does). S32 loads `infra-signatures` from its file: add a `negatives` argument to `load_pack` (or have `load_pack` read them from the pack's directory) in the S32 row.

**P4-7 (nit).** `test_rule_pack_leaf_mutation.py` carries a run-dependent count in a comment ("109 leaves: 84 killed, 25 survivors"); the assertion already pins the sets, so drop the numbers. The schema-test module docstring narrates the spec's pending change; move it to the v1.4 round's ledger line.

### Growth-stops

The commit touches no plan, requirements or review document (the spec is unchanged; `docs/reviews/` gains only this Pass). New scope only. Pass passes.

### Stopping condition (falsifiable)

S31 closes at the next pass when all of these hold: (1) `bdist_wheel` of `cpg/` lists `ccg/registry/reasons.yaml`, and the tree-derived package-data test fails on the pre-fix `pyproject.toml`; (2) the three planted inputs of P4-2 are rejected and the three schema mutants (`as` pattern drop, yield-value pattern drop, its trailing `$`) turn a test red; (3) the dangling `lifted` plant of P4-3 is rejected and `CCG_LEAF_MUTATION=1` reports survivors equal to the 24-leaf set; (4) mutants `R` and `N` are red; (5) the P4-5 pin exists and is green. P4-6 and P4-7 are carried by the S32 row and do not gate S31.

### Open questions

(1) `architect`/`teco`: `excludeVars` required or optional (v1.4). (2) `teco`: v1.4 must change §1.4 lines 160/169, the N-27 row and `check_rule` together, and add the `fastmcp` §12.1 row. (3) Process: another agent shares this scratchpad; while I worked, `scratchpad/mut.py` was overwritten by my own script and `scratchpad/m/` was recreated underneath me, so those two names may have lost another run's files. My artefacts are only under `scratchpad/analyst_s31/` and `neg*.py`, `eq.py`, `p5.py`, `p6.py`.

### Proposed REVIEW.md lines

Target: root `REVIEW.md`, heading *Always check*. Source: P4-2 (the alias test and the yield pattern); Pass 3 P3-2 is the same class.
- "A test that names a schema keyword (`pattern`, `minItems`, `required`) is checked by dropping that keyword from a copy of the schema; if the suite stays green, the input the test plants is also rejected by a second rule, and the test does not guard the keyword. Plant the input where only that keyword can reject it."
