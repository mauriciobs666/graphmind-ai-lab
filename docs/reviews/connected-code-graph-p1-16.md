# Review: C-1001 P1-16, the pack runtime (diff-scoped)

> **Status:** active · **Owner:** `analyst` · **Tracks:** C-1001 (P1)

## Scope and verdict

Reviewed, uncommitted under `cpg/`: `ccg/packrt/*` (8 files), `tests/packrt_helpers.py`, `tests/test_packrt_{match,extract,run,launch,generic}.py`,
`tests/slice/test_packrt_conformance.py`, the `tests/test_rule_pack_leaf_mutation.py` edits, the packrt row of `README.md`. Baseline: the
P1-16 row of `docs/plans/connected-code-graph-p1.md`, its section in `docs/plans/connected-code-graph-p1-linking.md` (and the P1-17/P1-18/P1-46
sections that consume it), `docs/plans/connected-code-graph-rule-pack-spec.md` §1.2–§1.6, §2.3, §4.2–§4.4, §12.1–§12.2. Root `REVIEW.md` applied
(no `cpg/REVIEW.md`). Not reviewed: `extractors/joern/*`, `tests/test_joern_run.py`, `tests/slice/test_extract_equivalence.py`, `sandbox/*`,
`infra/*`, `ccg/broker/*`. No FalkorDB write, no tree-mutating git command; all mutation work on copies in `$SCRATCH/*-r1/` (session scratchpad).

**Verdict: approve with suggestions.** No blocker: the runtime does what the plan's behaviours (1)-(7), (9) ask, equals the slice over the pinned
bundles, and its tests resist 31 data-plumbing mutants of my own (one survivor). Two majors (M1 guard coverage, M2 an unsound `confirmed` port) should
be fixed before P1-17 builds on `launch_facts`; the rest are minor or hand-off notes.

CPG: considered, not relevant — the code under review is new, uncommitted and absent from any loaded `cpg_*` graph; every claim below was checked by running it.

## Checks run (by execution)

| Check | Result |
|---|---|
| `pytest tests/test_packrt_*.py tests/slice/test_packrt_conformance.py tests/test_rule_pack_leaf_mutation.py` (cwd `cpg/`) | 138 passed, 1 skipped (the slow leaf test, env-gated); conformance **ran, 0 skipped**; bundles' sha256 `a0cae513…`/`f2ef6f5a…` match the pinned values |
| `pytest -k "packrt or leaf or rule_pack_schema or keynorm"` | 953 passed, 1 skipped, 4864 deselected. The whole suite (delegate: 5802) was **not** re-run |
| Slow leaf test `CCG_LEAF_MUTATION=1 pytest tests/test_rule_pack_leaf_mutation.py` | 3 passed in 400 s (§4) |

## 1. Conformance counts, re-derived independently

Method: plain `json.loads` of the two s14a bundles and a script that reads **records only** (no `ccg.packrt`, no `assemble.py`): decorator/callsite
regexes and `import` records. Then compared with the runtime and the live oracle.

| Claim | Independent derivation | Runtime | Spread (definitions) |
|---|---|---|---|
| 61 `http` providers | 62 route-shaped decorator records (`<x>.get|post|…|route|api_route`), all in `fastapi`-importing files, 62 distinct handlers; one of them has an unresolved prefix alternative (`line 2239`, the single diagnostic) | 61 facts + 1 diagnostic; oracle 61 + `droppedProviderAlternatives` 1 | 62 candidates → 61 emitted → 57 confirmed (slice: 59; difference = the two §2.3 handlers) + 4 possible |
| 28 `mcp_tool` providers | 28 `<x>.tool` decorators in files importing `mcp.server.fastmcp`/`fastmcp`: `mcp.py` 25, `test_mcp_client.py` 2, `test_input_bound_parity.py` 1 | 28, all `confirmed/key-match` | equal; the 3 look-alikes are the O-8 ones |
| 96 `CONSUMES mcp_tool` | 98 `call_tool` callsites; 96 in files importing `mcp`; the 2 excluded are `tools.py:1410/1506`, `resolution: resolved` internal calls (the import gate correctly drops them); 96 = 93 with a const `a0` + 3 non-const | 93 `confirmed/key-match` + 3 `possible/runtime-built` (key `None`) | equal; the slice has no MCP oracle, so this count rests on my derivation, not on `assemble.py` |
| 11 http consumers | only 1 `fetch` callsite exists (external); the runtime's `lifted` list has 14 slots = 11 resolved leaves + 3 unresolved non-leaves | 11 `confirmed/lifted-role`; oracle 11, `unresolvedConsumers` 0 | the 11 rests on the oracle (independent code) plus the lifting trace; 3 non-resolved slots never become consumers |
| `/mcp` mount | 3 `mount` callsites (`app.py:528,543,551`); only `:543` has an `a1` fed by `streamable_http_app` | 1 `mountPath` fact `/mcp`, `prefix`, line 543, condition `mount_mcp` | equal |
| 7 `include_router` sites vs 3 mount facts | 7 callsites; 3 have a `data` record feeding `a0` from a factory call | 3 mount facts (485, 538, 521) | equal to the slice's `_include_sites` (the other 4 feed a variable/non-factory) |

The conformance test compares providers key-for-key **and** the full `mounts` evidence list against the oracle for 59 of 61 (a strong check), consumers'
`liftedVia` hops entry-for-entry, the diagnostics list, and the status split. Ports: over §4.3's paper records the runtime gives `(8000, explicit)` and
`(8200, env-default)` and no 6379, `launchUnit` for L1, L2, L4 (execution of `test_over_the_paper_records…`).

## 2. Findings

### M1 (major) — the no-pack-specific-code guard misses most pack-derived words, and `mount` is blanket-exempt

Evidence (run): `tests/test_packrt_generic.py:21-35` derives forbidden words only from dict keys `callee`, `basename`, `prefix` (and `id`), and
`FORMAT_WORDS = {"mount"}` removes `mount` from the set although `mount` is `fastmcp.launch.mountPaths[0].call.callee`. I imported `forbidden()`/`hits()`
and planted (words printed: `APIRouter, FastAPI, call_tool, fastapi, fastmcp, fetch, include_router, mcp, mcp.server.fastmcp, streamablehttp_client, uvicorn`):

| Planted source | Caught |
|---|---|
| `if site["calleeName"] == "mount": …` | **no** (exempt word) |
| `if name.endswith(".tool")`, `"--port"`, `"mcpServers"`, `"FastMCP.streamable_http_app"`, `"kw.prefix"` | **no** (regex/glob/flag/arg values are not walked) |
| `b"fastapi"`, `except E as fastapi`, `global fastapi` | no (bytes constants, `ExceptHandler.name`, `Global.names` are not read by `code_words`) |
| `if x == "fastapi"` (comparison), `match … case "fastapi"`, `@fastapi`, attribute, keyword, import, f-string piece | yes |

So the guard enforces "no `fastapi` string", the plan's literal example, not "no pack-specific Python". This is the root-`REVIEW.md` walker class
("recurses into only some axes is a bound, not proof"). The docstring/comment exemption is **not** a hole: a docstring executes nothing (only
`f.__doc__` reads it; I planted that and it passes, which needs deliberate evasion — nit at most).
Suggested fix (I ran its first half): derive the words from **every string leaf** of the three packs, tokenized with `[A-Za-z_]\w*`, minus the
vocabulary of `ccg/schemas/rule-pack-v1.schema.json` (property names, enum and const values). Against the current runtime that leaves exactly five
residual tokens to justify one by one: `get` (the `dict.get` method), `http` (a core contract kind), `unresolved` (registry resolution), `router` (the
`unmounted-router` evidence token) and `mount`. Allow `mount` **by file** (`run.py`, where the `{mount}` slot is read), not globally, so a `mount`
callee in `launch.py` stays red. Then plant one token per leaf-kind (a regex, a glob, a flag, an `arg`, a callee) in a copy of `run.py` and require
red, the plan-mandated coverage probe, built from the packs' own leaves rather than a typed list.

### M2 (major) — `frameworkDefaults` emits a `confirmed` `pack-default` port when an argument is `<redacted>`

Evidence (run): `ccg/packrt/launch.py:78-82` takes the default whenever no `portFlags` flag is *seen*. Planted one launch
`command uvicorn, args ["srv.app:app", "<redacted>"]` (a `$EXTRA_ARGS` word gated out by `ccg.bundle.gate_argv`): output `port 8000, origin pack-default,
status confirmed`. Rule-pack spec §1.6 says the default "reads **emitted** (gated) text, so it is sound only where `command` and `args` are visible", and
§4.4 step 3 relies on that. The hidden argument can be `--port 9000`; the linker (P1-17 behaviour (4)) will then bind a consumer URL on 8000 to this
unit as `binding-confirmed`. `test_a_redacted_command_gets_no_default_because_the_basename_cannot_match` covers only a redacted *command*.
Suggested fix: no `pack-default` when any element of `args` is the gate's `<redacted>` token (take the token from `ccg.bundle`, do not retype it),
or emit it `possible`; test with the planted record above and a positive control (visible args, no flag: still 8000).

### m1 (minor) — a second `mounts` declaration, `class-decorator`, and `embedded` are silently mishandled, against the file's own "refuse, never read as empty" rule

Evidence (run, `ccg/packrt/run.py:79,93`: `indexes[0]` only): a pack with two `mounts` entries emits facts for both but composes provider keys from the
first only, so the output depends on declaration order (`GET /api/x` vs `GET /other/x` when swapped; the other mount's variant is lost with no
diagnostic). `mounts: [{kind: class-decorator, …}]` raises `KeyError: 'child'`, and a non-empty `embedded` is ignored with no error. No shipped or
plan-commissioned pack needs any of them (`fastapi` has one declaration; P1-46's `shell-mcp-call` declares none), hence minor. Fix: raise
`UnsupportedSubject` for `len(mounts) > 1`, a `kind` the runtime does not read, and a non-empty `embedded`, with one planted pack each (decision (f)
applied consistently).

### m2 (minor) — the `keyForm` plumbing is untested

Evidence (mutant): replacing `"keyForm": body.get("keyForm", "literal")` with `"literal"` (`run.py` `_one`) survives all 6 packrt test files plus the
conformance. The field is a spec §1.5 emission field the linker reads (non-literal forms match by form, §1.3). Fix: one planted rule with
`keyForm: glob` asserting the fact's `keyForm`.

### m3 (minor, plan) — nobody is commissioned to empty the last two leaves

Evidence: `docs/plans/connected-code-graph-p1-linking.md:43-45` says the 24 leaves "leave its unexercised set" and the set is "emptied by P1-16, P1-17
and P1-46 together"; `docs/plans/connected-code-graph-p1.md:157` (O-13) makes P1-17 an owner. The P1-17 section (`-p1-linking.md` lines 69-104) lists no
leaf-harness edit; only P1-46 does (line 56). The two `fastmcp:urlBindings[0].urlSources[0|1]` leaves are read only by the linker, so the delegate's
22/24 is the right outcome for P1-16, but with P1-17's Files list as written the set stays non-empty and `test_each_leaf_behaves_as_classified` never
reaches "empty". Suggested fix (architect/teco): add `cpg/tests/test_rule_pack_leaf_mutation.py` and "the linker tests read the `urlBindings` URL
sources through the pack, `GATE` gains `tests/test_linker_*.py`" to P1-17, and reconcile P1-16's "the 24 leave the set" to "22".
Note `GATE` runs only `test_packrt_run.py` and `test_packrt_launch.py` from the runtime tests: a leaf that only the conformance or `test_packrt_match`
reads would stay a survivor, so P1-17 must extend `GATE` for its tests.

### m4 (minor, hand-off) — facts P1-17/P1-18 need that `PackFacts` carries only indirectly

All verified by reading the P1-17/P1-18 sections against the runtime's output:
- **Unknown-key consumers.** P1-17 (3) links a fully unknown key to every provider (`unknown-key`). Decision (c) emits no consumer for an unresolved
  lifted leaf (spec §1.6 says so, and the slice equals it: `unresolvedConsumers`), so the only record of such a site is `PackFacts.lifted` items with
  `resolved: False, leaf: True` (none on S14: its 3 unresolved slots are non-leaves, so no real-data test exercises this). P1-17 must read `lifted`, not `consumers`; say so in its row.
- **`MOUNTED_BY {ord, mountPath}` (graph G6).** The assembler gets `mountPath` from `mounts[].prefixes` joined through a provider's
  `evidence.mounts[].sites[].siteId`, and for a unit-level mount from `launch_facts[fact=mountPath]`; the latter carries no "kinds the rule serves" (G6 says
  every provider of that unit whose kind the rule serves), which is in `urlBindings[].kinds`, not in the fact. Derivable but P1-18 should be told.
- **Pack-set selection.** `run_packs` over the full shipped set raises `UnsupportedSubject` (`paper-inhouse-bus` uses a `query` subject; run: `kafka`,
  `infra-signatures`, `severity-rules` return empty, `paper-inhouse-bus` raises). Loud is right, but P1-18's pack-set identity (O-10) and P1-44's
  dispatch must choose runnable packs and no function says which are; either `run_packs` skips packs with no runtime-evaluable rule (recorded in
  `diagnostics`) or P1-18 names the list.
- **Key steps.** `ccg/packrt/httpkey.py` is the only place steps 4-8 live (`ccg.keynorm` owns 8a only); P1-17 (12) requires a `matchers.py` test of
  `GET /a/%2E%2E/b → GET /b`. `match.path_hit` (the spec's `normalize_path` for `exact|prefix|template`) has **no production caller** (grep: only
  `test_packrt_match.py`), and `scope`/`clause_hit` matchers apply no path normalization. P1-17's `matchers.py` should import
  `ccg.packrt.httpkey.normalize_path_rfc`, not write a second normalizer; name that in the row.

### n1 (nit) — smaller deviations from the plan text

- Plan: "`cpg/tests/test_packrt_*.py` (one file per module)"; there is no `test_packrt_lift.py`, `test_packrt_mount.py` or `test_packrt_httpkey.py` (the
  behaviour is covered in `test_packrt_run.py`/`test_packrt_extract.py`, mutation-checked below), and `run.py`, `httpkey.py`, `launch.py` are not in the plan's
  Files list. Not wrong; the report should state it.
- `PackFacts` has a seventh field, `diagnostics`; the plan lists six. Needed (spec §1.4 "listed, not emitted"); record it in P1-17's input.
- The packrt README row is ~1,100 characters in one table cell; trim it to the interface and the pointers.
- The conformance test reads bundles with `json.loads`, not `ccg.bundle.read_bundle`/`loads`: sound because the sha256 pins the bytes and the runtime
  never reads the header, but a later canonical-form difference in `loads` would not be seen here.

## 3. Routine decisions (item 4)

| Decision | Ruling |
|---|---|
| (a) extra modules `httpkey.py`, `launch.py` (and `run.py`) | Needed: §12.1 requires the launch/config derivation and the `http` key steps; no existing home. Accept; the Files list should be amended after the fact. |
| (b) fact granularity: one fact per `(edge, anchor, kind, key, site)`, mount variants merged into `evidence.mounts` | Matches spec §2.2 and §1.8; 61 facts, no duplicates (the conformance dict-compare could hide duplicates, but the Counter over `out.providers` gives 57+4 = 61 http and 28 `mcp_tool`). Accept. |
| (c) fully unresolved lifted slot not emitted | Equals spec §1.6 and the slice; see m4 for the P1-17 read. Accept. |
| (d) non-`confirmed` `data` on a lifting chain → `possible/partial-key` + `weakenedBy` | An addition the plan does not state but P1-46's `shell-dynamic` data (`possible`) needs; `partial-key` is the only `possible` code valid on `CONSUMES` (registry: `shell-dynamic` is `["shell DATA", callsite]`, `field-truncated` `[DATA, fieldAccess]`). Equal to the slice on S14 (no weak data on the 11 chains). Accept; one-sentence addition to spec §1.6 at the feature's close. |
| (e) launch/mount facts `confirmed`, `reason: None`; `env-default` ports `confirmed` | Matches §4.4 step 8 (`origin` carries the heuristic). Not edges, so no registry code applies; the plan's "every item carries a registry reason" is narrowed for `launch_facts` and `marks` (`test_every_item_carries_a_registry_status_and_reason` covers the other four lists). Accept except the redacted-args case (M2). |
| (f) unsupported constructs raise `Unsupported…` | Right principle; applied inconsistently (m1). No shipped/commissioned pack needs the unsupported forms. |
| (g) multiple `mounts` | See m1. |
| (h) `prefix` on a file path stops at `/` | Correct per spec §1.3 (`/mcp` vs `/mcpx`); module-name prefix stops at `.` (`test_packrt_match`). Accept. |
| (i) plain `json.loads` | See n1. |

## 4. The leaf harness (item 2)

Sample of 16 leaves run through the harness's own `survives()` on fresh scratch copies (4 workers): 14 KILLED-classified leaves (`fastapi:mounts[0].prefix.default`,
`.recordConditions`, `.scope…prefix`, `.constructorPrefix[0].arg`; `fastmcp:launch.frameworkDefaults[0].port|protocol`, `.targetArg[0].index`,
`.portFlags[0].flags[0]`, `.mountPaths[0].call.callee`, `.envPorts[0].excludeVars[0].regex`, `rules[0].scope.imports[1]…`, `rules[1].emit[0].consumes.key`;
`fetch:…companions[0].arg`, `rules[0].emit[0].consumes.key`) all died; the two `urlBindings` leaves survived, as classified. Classification sets are right
for that sample; the full pass is below.

**`CCG_PACKRT_PACKS_FROM_SPEC`.** `packrt_helpers.pack()` returns `copy.deepcopy(spec.PACKS[name])` when the variable is set, else
`load_pack(PACKS/name/pack.yaml)`. In a normal run the shipped file is used; the file-equals-spec pin (`test_rule_pack_schema`'s `pack_file` tests, deselected in
mutation runs) keeps the two equal, so a runtime test cannot pass against a *different* spec text unless the variable is exported in the shell, and
even then the pin test still fails. `test_packrt_conformance.py` and `test_packrt_generic.py` call `load_pack` directly and ignore the variable.
`load_pack` is `validate_pack` plus `yaml` load and does not normalize, so the spec dict equals the loaded one. No finding.

**Full pass (run):** `CCG_LEAF_MUTATION=1 pytest tests/test_rule_pack_leaf_mutation.py` → `3 passed in 400 s`, so
`test_each_leaf_behaves_as_classified` held over every leaf: the survivors are exactly the two `urlBindings` leaves, and the 22 reclassified leaves all
die. The harness stays non-vacuous (`test_the_mutation_harness_can_tell_a_killed_leaf_from_a_survivor` passed).

## 5. Data-plumbing mutants of my own (item 6)

31 mutants on a copy (`$SCRATCH/mut-r1/work-p116a`, `-p116b`; never in place), each run against `test_packrt_{match,extract,run,launch,generic}.py` and the
conformance, `-x`: **30 killed, 1 survived** (`keyForm`, m2). Killed: `weak_of` ignoring `data` status; `_collapse` keeping the worse copy; mount condition
`code`, fact `line`, `factoryIds`, unmasked `prefixes`, constant status; `launchIds` dropped; port range off by one; `launchUnit.target`; `mountPath`
conditions and `pathMatch`; `decorator`, `line` and `qname` fields; `weakenedBy`; hop `viaLine`, `ord`, `lineStart`; verb `partialAlternatives`; `leaf` flag;
diagnostic reason; mark `class`; mount-evidence sort; chain order; `bareDefault` guards; same-file adoption of `envPorts`; `excludeVars`; `targetArg`
ambiguity; `protocol`.

## 6. What's solid

- The conformance test is a live oracle run (not a golden of the runtime's own output), pins the artifact bytes, skips visibly, and compares the mount
  evidence and hops entry-for-entry; the two expected differences are asserted by name and by `how`.
- Spec §12.1's two named fixtures (a router with its own prefix; `FastAPI()` plus a local `APIRouter()`), O-3 and O-5 are asserted as `possible`/prefix-only.
- The runtime reads only `pack.yaml` data; a rename-every-token test (`test_renaming_every_framework_token…`), an order-independence test and loud
  refusal for unsupported subjects/emissions/expressions are in place.
- Status propagation and the lifting fixpoint are mutation-resistant (above).

## 7. Open questions

- **For teco/architect (m3):** reconcile "the 24 leaves leave the set" (P1-16) with the 22 + 2 split, and give P1-17 the leaf-harness edit.
- **Run-of-`run_packs` contract (m4):** should unrunnable packs be skipped-and-recorded by the runtime or filtered by P1-18? A routine call for the
  architect; I did not treat it as a high-stakes fork.

### Proposed REVIEW.md lines

Target `REVIEW.md`, heading "Always check", source: M1 of this review (recurring class: `docs/reviews/connected-code-graph-p1-16.md` M1; the existing
schema-walker line covers enumerators, not word-derivation guards).
- Add: "A guard that derives forbidden words from a data tree (a pack, a schema) must take every string leaf, minus the format's own vocabulary; an
  exemption list is per word *and* per file, and a planted token per leaf kind (regex, glob, flag, argument, callee) must turn it red."
No prune proposed (no line is now enforced elsewhere).

## Pass 2

Executed against the tree: packrt + conformance + leaf files 166 passed, 1 skipped (slow); slow leaf run `CCG_LEAF_MUTATION=1` **3 passed in 494 s** (survivors still exactly the 2 `urlBindings` leaves; conformance ran, 0 skipped). Verdict: **approve** (no open blocker or major; m3/m4 are plan edits, below).

- **M1: fixed.** Re-planted: `calleeName == "mount"` in `launch.py`, `.endswith(".tool")`, `"--port"`, `"mcpServers"`, `"streamable_http_app"`, `"kw.prefix"`, a bytes constant, `except … as fastapi`, `.mount` attr, `streamablehttp_client`, `"PORT"`, `"uvicorn"`, `call_tool=` keyword, an f-string `include_router`: all caught; the real sources are clean. Not a hole: the three-pack limit (the other four packs are not served, and `infra-signatures`/`severity-rules` prose would only add noise); residual risk is that `SERVED` is a typed tuple, so P1-46's `shell-mcp-call` (and any later served pack) must be added by hand or it is unguarded. Accepted residuals by design, not holes: `"mount"`, `"http"`, `"unresolved"`, `"partial-key"` in `run.py` (and `partial-key` in `lift.py`/`mount.py`) pass, so a branch comparing to those strings in those files is invisible. `partial-key` is sound: a registry code (`reason_row`), not a pack word; the ALLOWED rows are pinned as really spelled by `test_every_allowed_token_is_really_spelled…`. Minor: ALLOWED is keyed by (file, kind) with `"*"`, and `("*","attr","get")` is the one wildcard; fine.
- **M2: fixed.** `launch.py:87` suppresses `pack-default` when any arg contains `REDACTED` (substring, so `--<redacted>` too); `launchUnit` still emitted; an explicit `--port` beside a redacted arg still read. Emitting nothing is right for P1-17: spec §4.4 step 3 says "adds nothing", the unit keeps its `launchUnit` and any explicit/env ports, and an unbound URL degrades to `possible` rather than a false `binding-confirmed`. Mutants (equality instead of substring; first arg only; check removed; explicit flag ignored): all killed.
- **m1: fixed.** `run.py:_refuse_what_is_not_evaluated` raises for >1 `mounts`, a non-`call` kind and non-empty `embedded`; my planted two-declaration pack now raises. Over all 7 shipped packs on s14a: six run, `paper-inhouse-bus` raises on its `query` subject exactly as in Pass 1 (not new, and not a pack P1-17/P1-18 can run). Nothing P1-17/P1-18 need is broken. Mutants (limit `>2`, kind check off, `embedded` check off, refusal not called): killed.
- **m2: fixed.** `keyForm` is pinned (`test_packrt_run.py:417`); my Pass 1 mutant is now killed.
- **m3: not fixed (plan edit, outside the tree).** `docs/plans/connected-code-graph-p1-linking.md` is unchanged: P1-17's Files still omit the leaf-harness edit and `GATE` extension. Owner: architect/teco.
- **m4: not fixed (plan/hand-off notes).** Same plan file untouched; P1-17/P1-18 still need the four notes (read `lifted` for unknown-key sites, `kinds` for unit mounts, runnable pack list, reuse `httpkey.normalize_path_rfc`).
- **Nits:** no new per-module test files, accepted: lift/mount/httpkey are covered in `test_packrt_run.py`/`test_packrt_extract.py` and survived 31 + 8 plumbing mutants (one survivor, since pinned), so the plan's "one file per module" is met in effect; the report should say so. The README row is now 662 characters (was ~1,100): fixed.
- **Same-revision interactions:** the new refusals, the REDACTED guard and the reworked guard do not conflict: guard ALLOWED rows and `launch.py`'s `REDACTED` import (`from ccg.bundle import REDACTED`, no literal) pass the guard together; `keyForm` and the refusals leave conformance numbers unchanged (61/28/96/11, `/mcp`).
