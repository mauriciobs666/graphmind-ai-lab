# Connected code graph — P1 plan, part 2: linking, assembly, store, refresh (W2)

> **Status:** active · **Owner:** `architect` · **Tracks:** C-1001 (P1)

*2026-11-20: P1-51, P1-52 and P1-53 added (post-P1-50 report D1, D2, AC-5 wrapper items; D3 RCA R1-R6); P1-49's typed-verb, wrapper-leaf, f-string-name and Done-when (2) text revised in place; revised after the analyst plan gate (`docs/reviews/connected-code-graph-p1-51-53.md`).*

Part of `docs/plans/connected-code-graph-p1.md` (decisions D1–D11, obligation walk §4, step index §7). Contracts:
packs `docs/plans/connected-code-graph-rule-pack-spec.md`, graph `-graph.md` + `-p0-graph.md` (model v1, DDL,
load protocol §4, spike decisions §6), bundle `-bundle-spec3.md` + `-bundle-spec4.md`. The slice's `assemble.py`
hand-coded the FastAPI/fetch recognizers; the pack runtime replaces that, with the slice as the conformance oracle.
Briefing line for every step: dev items and fixtures only; no writes to any shared database (live tests on
`ccg-falkordb`, port 6390, or `infra/ccg-falkordb.sh --throwaway`).

## P1-16 · pack runtime — `tdd-engineer`

- **Files:** `cpg/ccg/packrt/__init__.py`, `match.py` (rule `match`: callee qualified name or regex, decorator,
  receiver type hint, config path, file glob or role), `extract.py` (argument, field path, keyword or config value
  from the resolved value sets), `lift.py` (role lifting to a fixpoint with ordered `LIFTED_VIA` evidence, B2),
  `mount.py` (mount composition and conditional mounts, B3), `cpg/tests/test_packrt_*.py` (one file per module),
  `cpg/tests/slice/test_packrt_conformance.py`. Over five files: two dispatches (`match`+`extract` first).
- **Interface:** `run_packs(bundles: list[list[dict]], packs: list[dict]) -> PackFacts` where
  `PackFacts = (providers, consumers, mounts, lifted, launch_facts, marks)`, every item carrying `status`, `reason`
  (a registry code, `ccg.rulepack.reason_row`) and its hop evidence. Packs are loaded only through
  `ccg.rulepack.load_pack` (which already requires `negatives.yaml` beside signature packs). **No pack-specific
  Python**: the runtime reads `cpg/packs/<name>/pack.yaml`; a `fastapi`-named branch is a defect.
- **Behaviours (red→green):** (1) a literal-key rule emits `confirmed/key-match`; (2) a key built from a
  `{ref}` resolves through value sets, unresolved parts give `possible/partial-key`; (3) role lifting: a
  parameter path flowing by `DATA` to a recognized client call's key argument makes the function a client on that
  path, to a fixpoint, with the chain kept in order (AC-1's `buildUrl.p0` and `apiFetch.p0.path` with verb
  `p0.method`); (4) shell functions lift the same way; (5) mount composition: decorator path + the prefix of
  whatever `include_router(prefix=…)` mounts it, a conditional mount keeps both variants with the condition;
  (6) **conformance:** the `fastapi`, `fetch` and `fastmcp` packs over the stored slice bundles give the slice's
  61 providers and 11 consumers with the same hops; the only differences are the two handlers of rule-pack spec
  §2.3 (`inner.get`, `_probe_rest_app`'s `router.post`) `possible/partial-key`; the two fixtures of §12.1 (a factory
  with `APIRouter(prefix=…)` mounted with another prefix; a function holding `FastAPI()` plus a local `APIRouter()`)
  give `possible`; (7) the `fastmcp` `tool` rule yields `PROVIDES mcp_tool` `confirmed/key-match` for the 25
  `mcp.py` decorators and lists the look-alikes O-8 names (never silently including them); the `tool-call` rule
  yields `CONSUMES mcp_tool`; (8) 22 of the 24 leaves the P0 leaf harness could only exercise at P1 are exercised
  (`cpg/tests/test_rule_pack_leaf_mutation.py` reports them); the other 2 are P1-17's; (9) router-in-a-variable (O-3) and the MCP inner
  path override (O-5) stay `possible`/prefix-only, asserted.
- **Also:** the three packs the runtime runs (`fastapi`, `fetch`, `fastmcp`) carry `status: paper`, the only value
  `cpg/ccg/schemas/rule-pack-v1.schema.json` allows (format v1.5 is frozen), and are pinned to the spec's YAML blocks
  by `cpg/tests/test_rule_pack_schema.py`. **`status: paper` stays**: it is a v1 format token, not a claim that no
  runtime exists; no schema, `ccg/rulepack.py` or pin edit is made, and the P1-46 pack uses the same value. The step's
  report notes that the token reads misleadingly once a runtime exists and leaves any change to a format successor. Of the 24 leaves of `cpg/tests/test_rule_pack_leaf_mutation.py` that only P1 can exercise,
  22 leave its "unexercised" set here; the 2 `fastmcp:urlBindings` leaves stay, because only the linker reads them
  (emptied by P1-16, P1-17 and P1-46 together; `test_each_leaf_behaves_as_classified` passes with the set empty).
- **Done / Gate:** `analyst` (execute the conformance test, do not read it; the report records the artifact sha256
  and `0 skipped`, index D9); mutation check on the lifting fixpoint and on status propagation.

## P1-46 · shell `mcp-call` consumer pack — `tdd-engineer`

No existing pack recognizes a shell consumer of an MCP tool: the `fastmcp` `tool-call` rule matches `call_tool`
under `scope.imports: mcp`, which a script never satisfies, yet fixture rows LNK-19 (`mcp-call reindex`, `confirmed`
`key-match`) and LNK-20 (`eval "mcp-call $TOOL"`, `possible`/`shell-dynamic`, linked to **every** `mcp_tool`
provider including `fx-job`'s second `reindex`, NC-5) gate at P1.
- **Files:** `cpg/packs/shell-mcp-call/pack.yaml`, `cpg/packs/shell-mcp-call/negatives.yaml`,
  `cpg/tests/test_pack_shell_mcp_call.py`, the leaf-harness registration in
  `cpg/tests/test_rule_pack_leaf_mutation.py`, and `SERVED` in `cpg/tests/test_packrt_generic.py`, which must
  gain `shell-mcp-call` by hand or the pack goes unguarded.
- **Behaviours:** a rule matching a call site of the command `mcp-call` in a shell `<script>` extracts the tool
  from argument `a0` and emits `CONSUMES mcp_tool` with the key; a literal gives `confirmed`/`key-match`; a value
  that is a shell variable or a call site under `eval` gives `possible`/`runtime-built` on the `CONSUMES` edge (`shell-dynamic` is
  registered only on `shell DATA` and `callsite`, so `validate_pack` rejects it on the `CONSUMES` edge, which stays so; the `Link` row gains it in P1-17); the pack
  matches the inner call site that `-p1-extraction.md` P1-10 10b adds for an `eval` string, so **P1-46 Needs 10b**; the
  negatives file (required by `ccg.rulepack.load_pack` for signature-style packs; add it regardless) covers a
  comment, a string containing the word, and `mcp-call` as an argument to another command; the pack passes
  `ccg.rulepack.validate_pack`. If the pack format cannot express a shell-file scope (the `fastmcp` rules scope by
  imports), that gap is **returned to the architect as a finding before any engine change**, not worked around in
  the runtime.
- **Known limits, pinned, not fixed in P1.** The tool is `a0` as written, so `mcp-call --json reindex` yields a `confirmed`
  consumer of the key `--json` (the real `reindex` edge is lost), and a prefix wrapper (`sudo`, `exec`, `nohup`, `bash -c`) is an
  under-match, never an over-match. Option skipping is not P1 work: no real `mcp-call` exists in the repository (the only
  scripts that use it are the fixtures, which take no flags), so there is no flag grammar to model, and a generic "skip `-x`
  words" rule is wrong whenever an option takes a value (`--timeout 5 reindex` would give the key `5`). Neither the pack format nor the shell
  extractor changes. The P1-46 tests pin both outcomes by name as known limits, and fixture rows LNK-19 and LNK-20 and the shell
  rows are unchanged (flag-free). **P2 backlog item**, owner `architect`: a per-command flag grammar in the pack format (which words
  are options, which take a value) once a real wrapper is in scope, with wrapper handling in the same item.
- **Not a pack (D10):** Compose mappings and the `.mcp.json` launch chain are linker code (P1-17).
- **Gate:** `analyst`; mutation check on the literal/dynamic split.

## P1-49 · `python-http-client` consumer pack and the unknown-origin key rule — `tdd-engineer`

Two dispatches: **49a (tests only, `qa-engineer`)** authors and commits the red coverage table below; **49b
(`tdd-engineer`)** implements to it. Needs 16, 17, 46 and P1-05 **5d** (a call on an untyped receiver whose name matches an
in-repo method is Joern-linked, hence `resolved`, which a `resolution: [external, unresolved]` seed never sees until 5d
demotes it). Decision D13 (stakeholder, Q-F): P1 builds this pack. No pack seeds a Python HTTP call.

**Done when:** (1) the fixture rows `LNK-08`, `LNK-11`, `LNK-13`, `LNK-14`, `LNK-37`, `TW-2`, `AC6-EX-4` and `DF-3`'s `requeue`
item are asserted at P1 in the forms the key rule below gives; (2) on the dev gold `cpg/eval/dev/ac5-http.jsonl`, each of the 8 items whose impacted element is a Python script has a `CONSUMES http`
linking to the gold provider, at any status, by the path below; the count is reported, a miss is reported with its cause and
never tuned past; dev items only, no held-out file read (the exposure path-set rule, `-p1-ml.md`). Paths (the wrapper calls
are bare same-module calls, kept by 5d's `refsTo` rule, so `callers_of` carries them):
`falkor-chat/scripts/load_append.py::{main,_worker}` call `httpx.Client(...)` methods directly: seeded at the call, which is its
own consumer (`ownSite`, below: `_worker` is reached by `pool.submit(_worker, ...)`, which no call edge carries, and its
`thread_id` parameter feeds the path), key is the path (`/threads/{?}/messages` for a parameter segment), the parameter
`base_url` only feeds `urlBindings`;
`falkor-chat/scripts/seed_nlq_eval_corpus.py::{main,_check_server_reachable,_poll_until_terminal}` call `_http_json(method, path)`,
whose `Request(url, ...)` is seeded and lifted through `path` (and `method`) to each caller's argument; the wrapper composes
`url = f"{BASE_URL}{path}"` in a local before the call, so the seed slot is the bare local and rule 4's limit applies (the
lifted fact is `confirmed`/`lifted-role`);
`salesperson/scripts/load_demo.py::run_participant` and `language_salience_probe.py::{run_participant,_assistant_rows_since}` call
`http_json(method, url)`, whose seed slot is the bare parameter `url`; lifted to the caller's `f"{base_url}/shop/api/..."`
(`\x01/shop/api/...`, `base_url` a parameter: rule 2 drops the origin, `possible`). A caller that passes a parameter into
an informative template of a wrapper is not the consumer, the template's site is (unit P1-51b): the gold wrappers
`_poll_until_terminal` and `_assistant_rows_since` hit at their own sites. The item keys above need the constant text of Python f-strings, which the stored bundles do not carry
(precondition S-1, below): the count is measured on bundles that have it, else reported with that cause; (3) the analyst plan-gate stopping conditions: the coverage table below
is asserted cell by cell with its cause-axis rows committed in 49a before 49b starts; 5d's oracle mechanism is the one stated
there; 5d's rule is the one stated there over both receiver shapes with its probe table.

**Pack.** Files: `cpg/packs/python-http-client/{pack.yaml,negatives.yaml}`, `cpg/tests/test_pack_python_http_client.py`, the
leaf-harness registration in `cpg/tests/test_rule_pack_leaf_mutation.py`, `SERVED` in `cpg/tests/test_packrt_generic.py` (by
hand, or the pack is unguarded; each new `ALLOWED` row there is justified in a comment), and the format additions below
(`ccg/schemas/rule-pack-v1.schema.json`, `ccg/packrt/{lift,run}.py`; spec §1.6 is amended at the feature's close). Every
shape is a lifting; the direct call is the lifting's own seed site. Recorded shapes (probed in the stored bundles, 2,276
bundle files at `builtCommit 8377c326`; the probe is re-run in 49b and reported): `requests.<verb>(u)`, `httpx.<verb>(u)` and
`<client>.<verb>(p)` on an untyped client all have `calleeName` = the verb, `resolution` `unresolved`, no `externalTarget`, no
receiver; `httpx.Client(...)` is `external` `httpx.py:<module>.Client.__init__`; `Request(...)` and `urlopen(...)` are `external`
`urllib.py:<module>.<member>(request).Request.__init__` and `...urlopen`. A module-level `requests.<verb>(u)` call is `external`
`requests.py:<module>.<verb>` (fixture build; none in the proving repository, counts in unit P1-51a), so each verb lifting has a
typed seed beside the untyped one.
- *Verbs* `get|post|put|patch|delete`: one lifting each (`py-get`...), two seeds with `arg: a0` and the same `scope`
  `{imports: [{module: {prefix: requests}}, {module: {prefix: httpx}}]}`: the untyped `{call: {callee: <verb>, resolution:
  [unresolved]}}` and the typed `{call: {callee: <verb>, externalTarget: [{exact: "requests.py:<module>.<verb>"}, {exact:
  "httpx.py:<module>.<verb>"}]}}` (exact, never a glob: `__builtin.dict.get` and `typing.Mapping.get` are `external` too),
  `ownSite: true`, `urlLike: true`; the
  rule's `extract.verb` is `{literal: GET}` (the lifted `Hit` is the *leaf's* callee, so a callee group cannot carry the verb).
- *`urllib.request.Request(url, method=...)`*: seed `externalTarget` glob `urllib.py:*Request.__init__`, `scope` the `urllib`
  prefix, `ownSite: true`; companion `kw.method`, `bareDefault: GET`, `unknown: "*"`. *`urlopen(x)`*: seed `externalTarget` glob
  `urllib.py:*urlopen`, `ownSite` and `urlLike` (so `urlopen(req)`, whose argument is a `Request` object, yields nothing;
  `urlopen("http://h/x")` and a wrapper `def f(url): urlopen(url)` do), verb `{literal: GET}`.
- *`httpx.Client(base_url=...)`* is no seed: `base_url` is a `urlBindings` `callsite` source, `arg: kw.base_url`, `scope: file`;
  the key is the path alone, a base path prefix is not joined. Not covered: `requests.request`, `Session`, `aiohttp`, `http.client`.
- A wrapper such as `http_json(method, url)` is carried by the lifting fixpoint (spec §1.6) when the call reaching it is kept by
  5d (a type-backed member call, or a bare call of a `LOCAL`).

**Format additions (all optional, default off, so the `fetch` pack and the S12 conformance oracle are unchanged).**
`liftings[].seeds[].scope`: the file gate of rule `scope`, evaluated by `run_lifting` on the *seed call's own file* before the
seed is used. A rule's `scope` on a `lifted` match keeps its meaning (the leaf site's file, which for a lifted fact is the
caller's) and this pack leaves it off: the caller of a wrapper need not import the HTTP library; the wrapper's file does.
`liftings[].ownSite` (boolean): a slot is a leaf when no *slot* lies beyond it on the fixpoint (today a parameter node below it also
counts, so a seed fed by a parameter is never a consumer unless a call edge reaches its function). With it, a wrapper
that no call edge reaches (`pool.submit(f, ...)`, an untyped receiver, an entry point) is a consumer at its own site, from its
own template; a wrapper with at least one lifted caller is not a leaf, and is the consumer only when it is a `wrapperSite` (unit
P1-51, part B). `liftings[].urlLike` (boolean): a leaf keeps only
the alternatives that are URL-like (`httpkey.is_url_like(alt)`: it starts with `/` or `//`, matches `scheme://`, or is a run of
unresolved markers followed by `/`) and the wholly unresolved ones; a leaf with no URL-like alternative is dropped from
`PackFacts.lifted` (no keyless `possible` fact either), and `partialAlternatives` is computed before the filter so `[/a, \x01]`
stays partial. Intermediate slots are untouched. A format word the runtime spells must be in the schema vocabulary, as
`test_packrt_generic.py` requires.

**Untyped verbs (decided).** In the stored bundles `requests.get(u)` and `cfg.get("k")` are the same record, so no seed can
separate them; the gate is the file's import of `requests`/`httpx` plus URL-likeness at the leaf, not at the seed (a bare
parameter `path` and `self.base + path` must be seeded: `LNK-11`, every wrapper). Measured on the stored bundles: 725
untyped, `unresolved` Python call sites named `get|post|put|patch|delete` (549 with a non-URL constant argument, 17 with an
unknown argument fed by a parameter, ...) lie in files that import neither library and are never seeded; 5 lie in importing
files (`load_append.py`, the `httpx.Client` calls), all HTTP. **Consequences.** Precision: the residual false consumer is an
untyped `.get/.post(...)` in a file importing `requests`/`httpx` whose argument resolves to a URL-like string
(`routes.get("/health")`); it is `confirmed` when the key is fully known (the key rule is unchanged), 0 in the stored data,
and it enters P1-36's precision measure. Recall: not found are a relative URL with no leading `/` (`client.get("items")` against
a `base_url`), an aliased import (`from requests import get as g`), a verb call in a file that imports neither library (a
session passed in), and a fully unknown URL (no keyless fact). A receiver tag in the bundle (P2, owner `architect`) turns
these into typed seeds. No acceptance number changes: done-condition (1) and (2)'s 8 items reach their provider through the
`Request` lifting (6) or the `httpx.Client` verb liftings (2).

**Precondition: constant text of Python f-strings (unit P1-50, below; decided by the stakeholder 2026-11-20).** The stored
bundles carry no `const` piece for 1,834 of 1,835 Python f-strings, so the `/shop/api/...`, `/threads/...` and `/documents/{id}`
keys of done-condition (2) cannot form: of the 8 items only those whose URL text is a plain constant can link (an estimate from
the stored value sets, not measured). 49b's code and tests are independent of P1-50 (synthetic records); the (2) measurement and
the fixture rows with f-string URLs are taken on bundles rebuilt after it.

**Wrapper reached through a name-linked caller (decided).** `packrt/extract.py:Corpus.callers_of` is built from `targets`
only, so a wrapper called through an untyped receiver (`self._get(path)`, an untyped parameter) has no caller edge after 5d:
its URL role is not lifted from that call. With `ownSite` the wrapper's seed is then a leaf and the consumer is the wrapper's
own site, from its own template: `possible`/`partial-key` when a path piece is unresolved (`/items/{?}`), nothing when the
template holds no URL-like text (`self.base + path` is `\x01\x01`: a `POST /{?}` key would only match any one-segment
provider; a `Request` wrapper with a bare parameter gives a keyless `possible` fact). A wrapper with at least one lifted
caller whose template is not informative (P1-51, part B) is not a leaf, so its name-linked callers contribute nothing: a
pinned limit, not fixed in P1; a P2 item (owner
`architect`) gives lifting the `nameLinked` candidates as weakened callers. Tests: a wrapper called through `self.` and through
an untyped parameter (the call is not lifted; the wrapper's own site as above, for both a `/items/{?}` template and
`self.base + path`); the same wrapper through a constructor-typed receiver (lifted, as the fixtures' `ApiClient`); a wrapper
whose only reference is `pool.submit(w, ...)` (own site); the 8-script count of (2) detects any recall cost.

**Unknown-origin key rule (the contract is unchanged: a `confirmed` fact needs every slot its key uses resolved, N-17).**
`Corpus.alternatives` (`packrt/extract.py`) marks an unresolved piece by element kind. `\x00` (`UNKNOWN`): an `importRef`
(an imported name the resolver could not map to an in-corpus constant), a `ref` cycle or depth cap: a name that should
resolve. `\x01` (`UNKNOWN_VALUE`, new, `httpkey.py`): an `unknown` element (parameter, environment read, field read, call
result), a `ref` whose constant has no `constant` record, the `MAX_ALTERNATIVES` cap: a value not statically known. A `ref` is in-corpus by
construction (the resolver emits one only for a name it found), so a `ref` without a record is a constant the normalizer
does not record: a call-valued one (correctly `\x01`), or an imported in-corpus name whose target is not a recorded constant
(multiply assigned, an export form 5d's scan does not cover).

**Cost of `\x01` for an unrecorded constant: precision and recall, not recall only.** If that constant was a leading path
prefix (`API_PREFIX`), rule 2 drops its segment: the key `GET /health` (i) can match an unrelated `/health` provider as a
false `possible` link, and (ii) no longer matches the real `/api/health`, which the `{?}` segment form matched. No P1
mechanism separates the case (the bundle cannot say why a constant has no record); the false links enter P1-36's
possible-precision measurement like any `possible` link.

**Two predicates, both in `ccg/packrt/httpkey.py`.** `is_wholly_unresolved(piece: str) -> bool`: the piece is exactly one
marker (`\x00` or `\x01`): an *alternative* of a value set that is unknown as a whole. `has_unresolved(s: str) -> bool`: either
marker occurs anywhere in `s`: a *string under construction* that is partly unknown. `\x01\x01` (two adjacent unknown pieces)
is `has_unresolved` and not wholly unresolved, as `\x00\x00` is today. For a consumer `http` URL, `httpkey.split_url(url) ->
(path, origin_dropped)` and `httpkey.normalize_key_parts(raw, form) -> NormalizedKey(key, origin_dropped, unresolved_in_path)`
(a `NamedTuple`; `normalize_key` stays a thin wrapper returning `.key`, `url_path_of(url)` returns `split_url(url)[0]`):
1. *Authority position* (after `scheme://`, or after a `//` that begins the URL, a scheme-relative URL; a `//` anywhere else is
   a path's empty segment, collapsed by step 6; the authority runs to the first `/`, `?`, `#` or the end): a piece with either
   marker there is the host; it is dropped from the key, as a known host is; `origin_dropped` is true (a known host leaves it
   false). A known-host `//h/a` therefore keys `/a`, not as a path segment `h`.
2. *No scheme*: a leading run of `\x01` (one piece or several adjacent) immediately followed by `/` is the origin and is
   dropped, `origin_dropped` true. Every other position keeps today's behaviour, the segment holding the marker becomes `{?}`: a leading `\x00` (a
   path-prefix constant that failed to resolve must not lose its segment, or the real provider stops matching), a partial
   segment (`\x01api/x`, `a\x01/b`, `\x01:8000/x`), a sole piece (`\x01`), `\x01?q`, mid-path, trailing.
3. *Status of a direct or caller-side key* (`packrt/run.py:_one`; the old `"{?}" in key` test is replaced): `confirmed` only if
   `unresolved_in_path` and `origin_dropped` are both false; otherwise `possible`/`partial-key`. `unresolved_in_path` is
   `has_unresolved` of the path *after* the origin drop and *after* cutting the query and fragment: an unresolved piece
   there is not a key slot (the key is verb and path), so it stays `confirmed`, as today; an unresolved origin is not in the
   key either but decides which service answers, so it forfeits `confirmed`. `lift.py`'s `partialAlternatives` tests
   whole alternatives only, so a query-only unknown is decided in `_one` for a lifted fact exactly as for a direct one.
4. *A lifted fact* (`packrt/lift.py:run_lifting`): the caller's key comes from the leaf slot alone, so it is the URL only if every
   slot below the leaf on the lifting chain, the seed's own slot included, is **transparent**: it has no value set, or every
   alternative is wholly unresolved (the bare parameter). The first non-transparent slot (consumer side first) is recorded as
   the item's `opaqueSlot` (`{siteId, arg}`, also `evidence.opaqueSlot`) and the item is `possible`/`partial-key`: a seed
   slot `self.base + path` or `f"{BASE}{path}"` holds a piece the caller's slot cannot see, and a known sibling (`"/api" + path`)
   would put a wrong path in the key. **Limit, pinned by a named test:** a wrapper that composes the URL in a local before
   the seed call (`url = f"{BASE_URL}{path}"; Request(url)`) leaves the seed slot a bare `point-derived` unknown, which the
   bundle cannot tell from the parameter (`data` records and the `unknown` reasons carry no whole-versus-part
   information), so that lifted fact stays `confirmed`/`lifted-role`; telling them apart needs a bundle value-set element
   for a derived local plus the extractor read of local initializers, a P2 item (owner `architect`).

Base-unknown is therefore never `confirmed` wherever the bundle shows the base: a direct site (rules 1-3), a caller-side
template (rules 1-3), a seed or intermediate slot with a sibling (rule 4). `binding-confirmed` (O-9: confirmed and
bound) needs the base known: the fixture rows whose base is a parameter, `self.<field>` or environment read (`LNK-13`,
`LNK-11`, `LNK-14`, `AC6-EX-4`) are asserted `possible`/`partial-key` with their path key at P1 (`LNK-11` by rule 4: its seed slot is
`self.base + path`; `confirmed`/`binding-confirmed` is the P2 environment binding, `-p0-compose-binding.md`); `LNK-37`'s whole URL is
an imported constant and stays `confirmed`; `LNK-08` and `TW-2` interpolate a module constant (`f"{BASE}/health"`), which the bundle
cannot name: `extract.sc` (`ex`, the `formatString` case) drops the child of each `<operator>.formattedValue`, so the piece is a
bare `unknown` and the P1 forms are the observed `possible`/`partial-key` (`GET /items/{?}`; `GET /health`, the leading unknown
dropped as an origin by rule 2), neither linking to its `/api/...` provider. Capturing the name is a P2 item (owner `architect`):
it edits the hashed `extract.sc`, re-extracts every unit, and can change provider and mount keys built from f-strings, so it needs
its own measure and audit. `qa-engineer` verifies each against the rebuilt graph.
Cite `-rule-pack-spec.md` §1.4 and this step in the `httpkey.py` docstring.

**Marker sweep (49b; the grep is run verbatim from `cpg/` and its output attached to the report).**
`grep -rn 'UNKNOWN\|\\x00' ccg extractors --include=*.py | grep -v '^ccg/tests'`. Every hit that compares with the literal
`UNKNOWN` moves to one of the two predicates; each site has the tests named here, each with a `\x01`-only and a mixed alternative:

| Site | Predicate | Test |
|---|---|---|
| `extract.py:Corpus.alternatives` (emits) | n/a | element-kind matrix: `unknown` and `ref` without a record and the alternatives cap give `\x01`; `importRef`, a `ref` cycle and the depth cap give `\x00` |
| `lift.py:_verbs_at` (verb alternative) | whole | verb slot `[\x01]` is partial and falls to `unknown`; `[GET, \x01]` gives `GET`, partial |
| `lift.py:run_lifting`, `resolved` | whole | caller slot `\x01/shop/api/session` is resolved (`load_demo` and `language_salience_probe` keep `GET /shop/api/session`, `possible`); `[\x01]` is not; `\x01\x01` is |
| `lift.py:run_lifting`, both `partialAlternatives` | whole (list membership) | `[/a, \x01]` partial; `["/a?q=\x01"]` is not (decided in `_one`) |
| `lift.py` opaque-slot test (new, rule 4) | whole | the rule 4 tests below |
| `run.py:_apply`, the `url` list | whole | `[\x01/shop/api/session, \x01]` leaves `[\x01/shop/api/session]` |
| `run.py:_one`, `unresolved` | has | a mount prefix holding `\x01` or `\x00` drops the fact with the `unresolved-part` diagnostic; a query-only unknown keeps `unresolved` non-empty and the fact `confirmed` |
| `launch.py:launch_facts`, the `mountPaths` loop | has | a path holding either marker emits no `mountPath` fact; a clean one does |
| `mount.py:MountIndex.facts` (`resolved`, the stored prefix) | has | prefixes `[\x01/api]` and `[\x00/api]` give `possible`/`partial-key` and a stored `{?}/api`, never a raw marker |
| `httpkey.py:normalize_path` | has | a segment `\x01`, `a\x01b` or `\x00` becomes `{?}` |
| `httpkey.py:split_url` (was `url_path_of`) | has, by position | rules 1-2, the spot rows below |

`ccg/linker/bind.py` (the `"\x00" not in text` tests) acts on *binding text*, a different channel that never holds a
value-set marker, and `ccg/linker/matchers.py` reads the post-normalization `{?}` segment: **no change** to either, pinned by a test that
a bind value with `\x01` is not treated specially. The other `\x00` hits (`valuehash`, `atrest`, `anchors`,
shell/markdown/config/scip extractors) are NUL screening of unrelated text: no change.

**Coverage tests (49a, red first, committed before 49b).** (1) A reference model written in the test from rules 1-3 (never
importing the code under test) generates the expected key and status over the live cells of position x scheme x remainder x
count x cause (cells that cannot occur are not generated; the cause axis is meaningful only for a leading whole segment
without a scheme, and the test asserts that), checked against the code; (2) a hand-written spot set of about 20 rows,
including `\x01`, `\x01\x01/x`, `\x00/x`, `\x01api/x`, `a\x01/b`, `\x01:8000/x`, `\x01?q=1`, `http://\x01/a`, `//\x00/a`,
`/x/\x01/y`, the sole `\x01`, with the cause-axis rows (parameter, environment read, `importRef`, ref miss) among them; (3) an
end-to-end test: one path in two units, a consumer with an unknown base, the link is `possible`/`partial-key` to both;
(4) **rule 4, the seed-sibling experiment as the test:** over `tests/test_packrt_run.py:fetch_chain()` add a value set on the seed
slot (`fe`, `a0`) of `concat[unknown, unknown]` (`f"{BASE_URL}{path}"`, `self.base + path`), of `template[unknown, unknown]`
and of `concat[const "http://h", unknown]`: each gives `GET /api/x` `possible`/`partial-key` with `evidence.opaqueSlot` =
`fe`/`a0` (today all three are `confirmed`/`lifted-role`); the unchanged `fetch_chain()` and a seed slot of the bare
`[unknown]` stay `confirmed`; a two-hop chain whose middle slot is `concat[const, unknown]` is `possible`; the caller-side
template `concat[unknown, const "/api/x"]` gives `GET /api/x` `possible` by rule 2; the named limit test (a seed slot bare,
the composition in a local) stays `confirmed`; (5) query-only unknown, direct (`requests.get(f"/items?q={q}")`) and lifted
through a transparent wrapper: both `confirmed`; an origin drop, direct and lifted: both `possible`; (6) the per-site tests of the
table; (7) the existing `tests/test_packrt_*`, `test_linker_link.py`, `slice/test_packrt_conformance.py`,
`slice/test_linker_conformance.py` stay green (the 11 `fetch` consumers of `s14a`: every slot on each lifting chain is
transparent, measured, so rule 4 moves none). Negatives (49b): `dict.get("x")`, `os.environ.get`, a string or comment naming
`requests.get`, `urlopen` of a non-URL object; `validate_pack`; the mutation check on the verb, the import gate, the marker
split, the two predicates, the opaque-slot test and the status downgrade.

**Tests that 49b changes or adds to the 49a set.** In `test_p1_49a_pack_python_http_client.py`: (a) `test_every_rule_is_gated_by_scope_imports` becomes
"every lifting seed carries `scope.imports` and no rule needs one" (the gate lives on the seed, see Format additions); add a
wrapper in a file without an HTTP import called from a file with one (dropped: the seed's file decides) and the reverse (a
wrapper in an importing file called from a non-importing file: kept). (b) `test_pinned_limit_data_without_method_stays_get` and
`test_another_keyword_on_request_does_not_defeat_the_bare_get_default` expect `* /documents`: `lift._verbs_at` withholds
`bareDefault` when the companion's container argument (`kw`, `a1`) holds any value set or flow, deliberately, since `data=`
makes a urllib `Request` a POST and `headers=` cannot be told apart from it by a `kw.*` head; the bare `Request(url)` stays
`GET`, `Request(url, method="POST")` stays `POST`. (c)
`test_a_wrapper_with_no_lifted_caller_is_itself_the_possible_partial_key_consumer_PLAN_STATEMENT` splits in two: a seed slot
`template[const "/items/", unknown]` gives the own-site `POST /items/{?}` `possible`/`partial-key`; a seed slot
`concat[unknown, unknown]` gives no consumer. (d) `test_an_untyped_dot_get_with_a_non_url_string_key...` stays and gains the
parameter-fed `d.get(name)` called with `"timeout"` (dropped at the leaf) and with `"/x"` (kept: the residual limit, pinned by
name). In `test_p1_49a_httpkey.py`: add the known-host `//h/a` row (path `/a`, `origin_dropped` false) to the grid and spot set.
New in 49b: `ownSite` and `urlLike` each with the on/off pair over `fetch_chain()`-style records (the `fetch` pack's output is
identical with both off, run over `s14a`), `is_url_like` rows, `partialAlternatives` surviving the filter, seed `scope` on the
seed's file. Existing tests 49b updates: `tests/test_packrt_extract.py` (the `alternatives` rows at the top: `unknown`, a `ref`
without a record and the cap give `\x01`; `importRef` and a cycle stay `\x00`), `tests/test_packrt_generic.py` (`SERVED`
gains the pack; the token scan then flags the pack's words in the runtime, each resolved by renaming in the runtime or one
justified `ALLOWED` row), `tests/test_rule_pack_leaf_mutation.py` (the pack's leaves registered `KILLED`, `GATE` extended with
the pack's test file).

**Gate:** `analyst` (plan and code); no `data-scientist` note (no method choice; recall is P1-37's measure, already gated).
**Sequencing:** after 5d and 46; before P1-41/P1-42 and the `ac5-gate-1` record (`httpkey.py` and `packrt/` are hashed).

## P1-50 · 5d fix round and Python f-string constants — `tdd-engineer`

**One serialized unit, one normalizer bump.** The 5d gates (`docs/reviews/connected-code-graph-p1-5d-49.md` "P1-05 5d diff gate" F1-F7;
`docs/reviews/connected-code-graph-p1-5d-49-ml.md` B1, M1, M2, m1-m3) and the Python f-string defect both change what the normalizer
writes, so they land together: `NORMALIZER_VERSION` `"0.7"` and one delta `0.7` in `tests/slice/test_normalize_equivalence.py`.
Before the qa rebuild (coordination checklist item 4), after 5d (`69789943`), before 49b's (2) measurement and P1-41/P1-42.
Gate: `analyst` re-gate, then the `data-scientist` re-audit (bar below). **`extract.sc` is unchanged only if the Part B check
(step (c)) shows Joern's f-string pieces are always raw text.** If it forces a raw flag on the piece, the order becomes: script
change, re-pin `extract.sc.sha256`, regenerate the `fixed-*.facts` and the `rt` table, then Part A's figures (the live `rt` pins
and the draw file's `extractScriptSha256` move with it).

### Part A: the 5d fix round (`cpg/extractors/joern/normalize/assemble_bundle.py:_self_rule`, tests, `cpg/extractors/joern/README.md`)

**The `self` rule counts definers in the whole unit, not among Joern's candidates.** Condition (ii) is: the unit's symbol table
`S` holds exactly one class method named `name` (a method whose parent is a type, any class), it is the enclosing method's own
class's, and it is among the site's Joern candidates; (i) and (iii) are unchanged. Counting over the candidates held vacuously
wherever Joern had already narrowed the call to the base method (`WorkflowExecutor._drive_loop` calling `self._execute_step`,
overridden by two in-unit test doubles). Chosen over base-class data in `extract.sc` because it needs no extract change, no re-pinned
script hash and no base-name resolution (a Python base is often a short imported name), and it never confirms what another
definer in the unit could override. A test-double override counts as an override.
**Figures (their one home):** of the 511 sites the first build confirmed, 379 stay `confirmed` (74%) and 132 become `possible`/`name-match`
(26%: 117 Joern-narrowed in `fcs`, 15 elsewhere), status only, candidates listed, 0.47% of the 27,804 formerly `confirmed`
Python call edges; 131 of the 132 are correct confirmations lost to an unrelated same-name class. Per unit the 379 are: claude 9,
falkor-chat/server 147, mcp-monitor 7, salesperson 17, model-bench 22, skills 11, decision-models 166. Limits, reported not
fixed: a subclass in another unit, an instance attribute shadowing the method, same-name classes unrelated by inheritance
(class-aware retention needs base-class data: the P2 item with the SCIP upgrade, owner `architect`; a fail-closed subclass test by
name over the staged source is a possible cheaper P2 route).

**Acceptance, derived independently of `assemble`.** The expectation of `test_the_self_rule_confirms_the_ast_references...` and of the
overlay is written by Python `ast` over the oracle's stage tree (pinned at the oracle commit, as the pinned-stage test requires),
never from `assemble` output: the `self.m()` calls in a method whose class defines `m` by `def`/`async def` in its body, where no
other class anywhere in the unit's stage files (tests included) does, restricted to the sites the type rule demotes and whose facts
list that method among `in`. A disagreement in either direction fails and lists the ids. The hand-listed snapshot
`selfByJoernNarrowing` and its readers (`test_overlay_5d.py`, `test_normalize_equivalence.py`) are deleted. Rows: the other definer
{subclass, unrelated class, superclass} x {present in `in`, absent from `in` (Joern narrowed to the base method)}, plus one live
golden with a real Joern-narrowed `self.m()` and an in-unit override (recorded with the Joern version).

**Draw and re-audit.** `tests/data/p1_05_5d_self_audit_draw.json` is regenerated from the bundles rebuilt at normalizer `0.7`, at the
pinned oracle commit: population ids sorted, `random.Random(20261119).sample(population, 30)`, the same seed over the *new*
population (the previous `drawn` list stays in the file under `previous`, reported beside the new one). The `data-scientist`
re-audit passes on: **30 of 30 correct** by hand against the source at the pinned commit, **and** the `ast` census over every site of
the new population finds no other definer of the name in the unit (0 hits; `a118e830a3fd791d5a6b2c9e6f2cf705` absent); the
population size and the number of drawn sites that were Joern-narrowed are reported (a size more than 10% off 379 is explained).
Any miss fails the rule (narrowed or removed, never a lower bar). The Wilson bound (0.886 at 30/30) is a floor over the in-unit
population only; the census is reported beside the hand audit; the test report says the population list comes from the draw file
and which normalizer built the bundles.

**Typed receivers with a short type name stay demoted: a counted limit.** Joern gives `rt` as the bare annotation name (`Campaign`,
`Fx`) when the annotated class is imported but not resolved, so `target == rt + sep + name` fails: 339 sites, 334 in
`decision-models`, about 9.5% of the demotions and 1.2% of the 27,804. Accepting a short `rt` equal to the target's class name
would be a second `confirmed`-producing rule without an audit, and it wrongly confirms when the annotation names a same-named class
of another unit; the cost it avoids is status only, and `merge_scip` `name_linked` re-confirms a site whose SCIP target agrees
(P1-15). Stated in `cpg/extractors/joern/README.md`; a planted row (an annotated parameter of a sibling-module class) pins it as a
limit. **Reversal (a stakeholder item, owner `teco`, read from the P1-15 evidence run, which reports it):** if more than 10% of
these short-`rt` sites (34 of 339) are still `possible` after `merge_scip`, the rescue rule is reopened with its own audit; the
10% is the architect's proposal and the stakeholder may change it.

**Tests and pins.** JS/TS type rule: rows over `type_backed("javascript", ...)` and a JS `assemble` fixture, one per cell: member
typed to the base (confirmed, separator `:`), `ANY` base (demoted), module-typed `this` (demoted by the type rule, `self` rule
not applied), bare parameter (demoted), bare `LOCAL` (kept); expected values written from the `jssrc2cpg` probe, not from the
function; the mutant `MEMBER_SEP = {"python": ".", "javascript": "."}` turns a row red. Receiver tag: the live test
`test_the_ported_extractor_reproduces_the_slices_raw_facts[fcs]` asserts (a) for every id in `overlay_5d.rt_table(tag)` the live
`rt` equals the table, and (b) the live run's `rt`-bearing facts equal a pinned count and a pinned sha256 of the sorted `(id, rt)`
list for the unit (the live run carries `rt` on every dynamic call with targets, 8,721 in `falkor-chat/server`, far beyond the 1,479
listed, so (b) is what makes a wrong receiver base on a currently backed site visible); the `recvTag` mutant `arg(r, 1)` ->
`arg(r, 2)`, run through a copy of `extract.sc` and never in place, turns it red. Pinned literals that go red and are re-derived,
not edited to pass, each with the independent count that supports it: `1479` and `117` in `test_overlay_5d.py`; `(264, 147, 117)`
(the `self`-rule test) and `(1216, 3)` (`fcs_touched`, `sp_touched`) in `test_normalize_equivalence.py`.
**`merge_scip` disagreement count:** `MergeResult.counts` gains `disagree_partial_kept` (a site with a `name_linked` entry that keeps
a type-backed target and whose SCIP target is outside `J` stays `confirmed`, where the same disagreement with no entry is
demoted); owner the P1-15 `merge_scip` step (`tdd-engineer`), pinned by a case in `tests/test_scip_merge_name_linked.py`, reported
by the evidence run.

### Part B: constant text of Python f-strings

**Defect.** `normalize.core.Resolver.lit` runs `ast.literal_eval` on the Joern piece `c`; for a Python f-string the pieces of the
`formatString` operator carry the *unquoted* text (`tests/golden/projection/facts.python.jsonl`: `{"c":"guard path ","o":"lit"}`), so
every constant piece becomes `{"unknown": "runtime-built"}`: 1,834 of 1,835 Python `template` value sets in the stored bundles have no
`const` (`load_append.py`'s `f"/threads/{thread_id}/messages"` is three unknowns).
**Check first (reported; each is a golden row whose expected value is written from the probe, not from `Resolver`):** run the shipped
`extract.sc` through live Joern over one scratch Python file whose f-strings cover `f"/a/{x}"`, `f"it's {x}"`, `f"{a}'b'{c}"`,
`f"{a}1{b}"`, `f"{{lit}}{x}"`, `f"a\n{x}"`, an implicitly concatenated `"p" f"{x}"`, a nested format spec and an empty piece; record the
raw `c` of every piece (in particular whether `{{` arrives as `{{` or `{`, and where pieces split). **Change:** in `Resolver.element`,
a `lit` child of a `fmt` in a Python file is `{"const": <c>}`, `c` taken as raw text and not parsed, if the check shows Joern never
quotes f-string pieces; if it quotes some, parse-first is kept for the pieces shown quoted and the rule is stated from the data, or
`extract.sc` gains a raw flag on the piece. The normalizer does not merge adjacent `const` pieces; escape sequences stay source text
(irrelevant to URL keys). The empty-piece filter, the depth cap, JavaScript/TypeScript templates (backtick regex), `concat`, every
non-f-string literal and every record kind except `valueSet` and `constant` values holding a Python template **must not change**.

**Independent reference (in the tests, never importing the normalizer).** For each Python `JoinedStr` in the stage trees: its pieces
with each run of adjacent constant text merged into one `const` whose text is the raw source segment (`ast.get_source_segment` on
the `Constant` nodes, exact on Python 3.12 where `ast` merges adjacent text and Joern splits it at implicit concatenation and at
`{{`/`}}`), and each `FormattedValue` as one non-const piece. The compared side is the stored/rebuilt template element with its
adjacent `const` pieces merged by the test. The `{{`/`}}` convention follows the probe. Template elements are matched to a `JoinedStr`
through the stored records: `(file, call line, argument path)` for a `valueSet`, the qname for a `constant`. A disagreement in either
direction fails and lists the ids; a class of difference is admitted only by name in the test, with its probe row.

**Measured effect.** (1) Census over the rebuilt bundles: Python `template` value sets with at least one `const` piece (1 of 1,835
today) against the reference's count, disagreements listed. (2) The delta `0.7` edits the stored Python template elements in
`fcs`/`sp` to the reference's pieces, listed by id, with every other record byte-identical
(`test_whole_bundle_records_are_the_same_multiset_per_type` stays green for the other types). (3) Done-condition (2) of P1-49: the
8-script count before (at most about 3, an estimate) and after (expected 8 of 8), a miss reported with its cause and never tuned
past; the fixture rows with f-string URLs (`LNK-13`, `LNK-14`, `LNK-11`, `AC6-EX-4`) are re-derived by `qa-engineer` on the rebuild.
(4) Provider and mount keys: the number of provider facts and mount prefixes (`run_packs` over the rebuilt bundles) whose key or
prefix changes because an f-string now carries constants, by kind; none may become `confirmed` that was not (a provider or mount
holding an f-string stays unresolved, the mount `unresolved-part` rule and the provider `drop` mode).

### Interactions between the two parts (each pinned by a test)
1. Disjoint output: the `self` rule edits `callsite` `targets`/`resolution`/`status`/`reason`; Part B edits only `valueSet` and
   `constant` values. Callsite ids are `(symbol, name, ordinal)` and hash no value, so the population ids are the same with or
   without Part B: a test builds `fcs`/`sp` with Part A only and with both and asserts the callsite records and id sets equal,
   which is what lets the draw be made once.
2. The newly `possible` sites have no `targets`, so they leave `callers_of`, and a wrapper reached only through them is not lifted
   (the P1-49 name-linked-caller limit). None of P1-49's 8 dev wrappers is called through `self.` or an overridden method (bare
   same-module calls), so done-condition (2) is unaffected; the count is re-run after the bump.
3. Part B turns an all-unknown template into one with constants, never the reverse; rule 4's transparency test already treated
   `\x01\x01...` as non-transparent, so no lifted fact changes class because of it. P1-49's figures are taken after both parts.
4. One version: the delta's `apply` composes the two edit sets over disjoint fields;
   `test_each_delta_pins_its_version_and_the_last_one_is_the_normalizers_current_version` holds with `"0.7"`.

### Done when (on the ten Python units rebuilt at the pinned oracle commit; the figures are the scratch rebuild of `docs/reviews/connected-code-graph-p1-5d-49.md`, the implementer re-derives them independently, a mismatch is investigated and never tuned)
1. `selfResolvedSites` = **379**, per unit as stated above, equal to the `ast` single-definer set over the oracle stage trees computed
   without importing `assemble`, every disagreement listed by id; `a118e830a3fd791d5a6b2c9e6f2cf705` absent.
2. `nameLinkCounts` = (3,694 demoted, 379 self); against the rt-stripped baseline 3,686 `confirmed`->`possible` and 8 narrowed.
3. A scratch mutant restoring (ii) to count over the candidates turns the Joern-narrowed row, the live probe and the census red;
   the JS `MEMBER_SEP` and `recvTag` `arg(r, 2)` mutants (through a copy) turn their rows red.
4. Part B: every Python template's constant pieces equal the reference over `fcs`/`sp` and the census; the callsite records and id
   sets are equal with Part A only and with both.
5. The 8-script count is reported (expected 8 of 8; a miss with its cause).
6. The whole `cpg` suite and `-m "slice and live"` run in one invocation, the known failures named; the draw file regenerated and
   handed to the `data-scientist` (bar above).

**Order inside the unit.** (a) the independent references and the JS/TS, receiver-tag and literal tests, red; (b) the `_self_rule`
change; (c) the Part B live check, then the `Resolver.element` change, goldens regenerated; (d) the delta `0.7`, the pinned
literals, the regenerated draw file; (e) the full run of Done when 6.

## P1-51 · typed `requests`/`httpx` verb seeds and the wrapper's own site — `tdd-engineer`

Two serial dispatches (both edit `packs/python-http-client/pack.yaml` and its test file): **51a**, then **51b**; both are
dispatchable now (neither waits on a stakeholder answer). Cause: `docs/test-reports/connected-code-graph-p1-5d-50-report.md` (D1,
section C, the `GET /0` finding); gate: `docs/reviews/connected-code-graph-p1-51-53.md` (B1, M1, M5). Needs P1-49 and P1-50.
CPG: considered, not relevant — the subject is the `ccg` pack runtime and its stored bundles, not a Joern `cpg_<component>` graph.

### 51a: typed verb seeds
**Basis.** The seed form is P1-49 "Pack" (exact `externalTarget` pair; exact, never a glob or `resolution: [unresolved, external]`,
which would seed `__builtin.dict.get` 358 times and `typing.Mapping.get` 176 in the proving repository). Observed on the proving
repository's bundles (`01cb0dba`): 2,178 callsites are named `get|post|put|patch|delete`, 816 are `unresolved`, none has a verb
target in `requests.py` or `httpx.py` (3 `httpx.py` targets exist, all `Client.__init__`). Observed on the fixture `svc` and `job`
bundles: 5 `requests.py:<module>.<verb>` callsites each (4 `get`, 1 `post`). `httpx.py:<module>.<verb>` is **assumed** by symmetry,
observed nowhere, and pinned only by its synthetic row.
**Stop condition:** if the typed seeds match any callsite on the ten Python units' stored bundles (expected 0), stop and return to
the architect.
- **Files:** `packs/python-http-client/pack.yaml`, `tests/test_pack_python_http_client.py`, the leaf-harness registration
  (`tests/test_rule_pack_leaf_mutation.py`, new leaves `KILLED`), `tests/test_fixtures.py` (the live guard).
- **Done when:** (1) synthetic records of the two shapes, in a file importing the library, give each verb's own-site consumer;
  (2) the real proving-repository `externalTarget` strings give nothing with the import gate satisfied (`__builtin.dict.get`,
  `typing.Mapping.get`, `os.py:<module>.<member>(environ).get`, the `fastapi/testclient.py` targets, `pathlib.py:<module>.Path.get`,
  `FalkorDB.select_graph.delete`), which kills the glob and `resolution` widening mutants; (3) a typed call in a file importing neither
  library is not seeded; (4) live guard (`-m live`): every Python unit of `fixture-system1` that calls `requests.<verb>` yields at
  least one http consumer (gate review `docs/reviews/connected-code-graph-p1-51-53.md` M1: 10 consumers against 0 without the seeds), and the report records the observed form of each consumer of
  `LNK-08`, `LNK-11`, `LNK-13`, `LNK-14`, `LNK-15`, `LNK-37`, `TW-2`, `AC6-EX-4` and `DF-3`'s `requeue`; the forms are the key rule's
  (P1-49), with `LNK-08` `GET /items/{?}` and `TW-2` `GET /health`, both `possible`/`partial-key` (see P1-49); a divergence returns to
  the architect; (5) acceptance report, not a test: over the ten units' stored bundles the `providers`, `mounts`, `consumers`, `lifted`
  multisets are equal before and after (baseline: providers 91, mounts 3, consumers 128, lifted 73).

### 51b: a wrapper whose template is informative is the consumer, not its callers
**Basis.** AC-5's gold items name the wrapper symbols (`_poll_until_terminal`, `_assistant_rows_since`: the gold lines are 289 and 367,
the stored facts 289 and 366; the hit is by symbol, `cpg/eval/dev/ac5-http.jsonl`); AC-5's text and the gold are unchanged. The engine
attributes a wrapper's *callers* as consumers with the key of the leaf slot alone: a bare `\x01` (keyless, linked to every `http`
provider; QA observed 60) or the constant `0` (`GET /0`). A caller that passes an id, cursor or origin into a template has not
requested that value; the template's site is the request.
**Rule** (`liftings[].wrapperSite`, boolean, default off, set on the seven `python-http-client` liftings; `fetch` and the S12 oracle
are byte-equal with it off). For each leaf `L` (as `ownSite` defines it), read **before** the `urlLike` filter and the `resolved`
test: scan `chain_of(L)[1:]` (the leaf excluded; the first-discovered predecessor chain `lift_role` already records, deterministic
because seeds and flows are sorted; a parameter that feeds two slots belongs to the first one reached: a pinned limit) for the first
**informative** slot `X`. A slot is informative when it has an alternative `a` with `is_url_like(a)` that names a resource: after
the authority (a known host or an unknown-origin run) and with the query and fragment cut, the remainder is the root `/` or has a
segment other than `{?}`. A bare origin names none and does not count: it is a base the caller completes, so the caller stays the
consumer. Decide on the remainder's text, not on the normalized path (normalization maps an empty path to `/`). Counts:
`/documents/\x01`, `\x01/shop/api/messages?...`, `/`, `\x01/`, `http://h/`. Does not count: `\x01\x01`, `/api\x01`, `\x01?q`,
`\x01/\x01`, `http://localhost:1234`, `https://\x01`. The reference model carries the shapes `root` (`/`, `\x01/`), `rootorigin`
(`http://h/`) and `origin_only` (`http://localhost:1234`). If `X` exists: `L` is dropped from `PackFacts.lifted` (else `linker/link.py:_unknown_lifted`
re-creates its keyless fact) and `X` is the consumer, once however many leaves map to it: `leaf: true`, its own alternatives under
the `urlLike` filter, hops from `X` to the seed, status by P1-49 rules 1-4 (a query-only unknown stays `confirmed`, a path or origin
unknown is `possible`/`partial-key`; a further non-transparent slot below `X` records its `opaqueSlot`). Verbs come from `X`'s own
chain: a wrapper that passes its `method` parameter into `X` yields `*` (pinned limit). With no informative slot nothing changes
(LNK-11: the caller stays the consumer). **`urlLike` is not touched:** flipping it on `py-request` drops the bare-`\x01` leaf before
the scan (gate review B1: 8 of 8 with `wrapperSite` alone, 6 of 8 with the flip, and 9 real keyless callers lost); dropping non-URL `Request(...)` leaves outside wrapper chains is a separate decision with its own figure
(P1-36). The existing pins `tests/test_pack_python_http_client.py:172` and `:244` (no `urlLike` on `Request`) stay green.
Origins passed by callers are not joined into `X`'s key (P2, owner `architect`, with the env binding). Demotion of keyless leaves
is only this rule's removal of leaves beyond an informative slot; every other keyless leaf stays a keyless `possible` fact (AC-6,
"never miss"); a fan-out cap is P1-36's trade-off.
- **Files:** `ccg/packrt/lift.py` (`run_lifting`, the informative test beside `_opaque_slot`), `ccg/schemas/rule-pack-v1.schema.json`,
  `packs/python-http-client/pack.yaml` (`wrapperSite: true` x7), `tests/test_packrt_wrapper_site.py`, `tests/test_packrt_generic.py`,
  the leaf-harness registration.
- **Tests, red before the engine edit (stopping condition c).** A reference model written in the test, independent of `lift.py`,
  generates the expected facts over the axes the rule varies on: slot kind (leaf, intermediate, seed), alternative shape (no origin,
  `\x01` origin, known origin, `\x00` prefix, one partial segment, query-only, scheme), alternative count (one; mixed with a bare
  `\x01`), informative slots on the chain (0; 1; 2 with a non-informative one nearer), leaves per `X` (1, 2), verb source at `X`
  (literal, parameter), `urlLike` on and off, `wrapperSite` on and off; checked cell by cell against the code. Plus: the linker emits
  no keyless fact for a dropped leaf (through `link()`); `fetch`/`s14a` conformance unchanged. The fixture live check (`LNK-11` stays
  `possible`/`partial-key`) is a regression check only, since the fixtures hold no informative wrapper; the synthetic axes are the
  positive control. Mutation check: the informative test, the leaf drop, the scan start (leaf excluded), the pre-filter read, the
  `wrapperSite` gate.
- **Done when:** (a) the AC-5 count by the P1-49 definition is **8 of 8** on both instruments (store method, bundle method), equal
  item by item; if either is below 8, stop and report the item and its cause, never tune; (b) the ten-unit diff (acceptance report)
  equals: removed consumers `{language_salience_probe.py:397 GET /0}`, added `{seed_nlq_eval_corpus.py:289,
  language_salience_probe.py:366 and :424, load_demo.py:284}`, 7 keyless lifted leaf items removed (5 sites), providers, mounts and
  every `confirmed` consumer equal; any other row stops the unit; (c) the reference-model test is committed red first; (d) the
  `join` link of gold item 5 no longer carries the keyless sites 397 and 438; (e) the whole `cpg` suite and `-m "slice and live"`,
  the 8 known reds named, no new failure.
- **Must not regress:** the 5d precision work (only `packrt/` and a pack change; the bundles are byte-identical), the `fetch`
  output, providers and mounts, every `confirmed` consumer. P1-36's possible-precision measure takes the new `possible` consumers.
- **Gate:** `analyst` (diff; execute the before/after comparison). `packrt/` is hashed: lands before the `ac5-gate-1` record.

## P1-52 · TypeScript MCP-client consumer pack — `tdd-engineer`

Basis: LNK-02/06/12 are P1 rows of `docs/test-plans/connected-code-graph-fixtures.md` §5.1 and count in its §6 per-shape floors; no
requirement names a TS MCP client and the proving repository has none (`callTool` and `@modelcontextprotocol` occur only in
`fx-web`); no plan step built the pack.  Decided by the stakeholder (Q-TSMCP A): P1 builds it.
**Design (no engine or schema edit).** `cpg/packs/ts-mcp-client/{pack.yaml,negatives.yaml}`. Lifting `ts-call-tool`, `role: {kind:
mcp_tool, slot: url}` (`url` is the only lifted-role name the runtime and the schema's `lifted` enum carry; it holds the tool name,
said in a comment), seed `{call: {callee: callTool, resolution: [unresolved]}, arg: a0.name, scope: {imports: [{module: {glob:
"@modelcontextprotocol/sdk/**"}}]}}` (a `prefix` matcher would not match: its separator is `.`), `ownSite: true`; rule `match:
{lifted: {lifting: ts-call-tool}}`, `extract: {tool: {lifted: url}}`, `emit: [{consumes: {kind: mcp_tool, key: "{tool}"}}]`.
`urlBindings`: `{callsite: {callee: URL}, arg: a0, scope: unit}` for `kinds: [mcp_tool]`. **`scope: unit`, not `file`:** LNK-12's
consumer is `Admin.tsx::onReindex` while `new URL("http://localhost:8200/mcp")` is in `mcp/client.ts`; `scope: file` leaves it unbound
(gate review B2: it links to both `reindex` providers), `scope: unit` binds it to `svc` alone (port 8200 matches only
`svc` once P1-53's `scripts/Dockerfile` exists). Limit, pinned by a test: any other literal `new URL(...)` in the unit adds an
alternative to every `mcp_tool` consumer of the unit's URL set. Gate review B2 verified that the `a0.name` object-literal values
resolve (`get_item` literal, `list_orders` through `TOOL_LIST_ORDERS`, `reindex` through the wrapper and `p0`) and that the SDK calls
are `unresolved` while `Admin.tsx:4` resolves to the wrapper.
**Consequence for the rows.** The binding applies to every `mcp_tool` consumer in unit `web`, so LNK-02, LNK-06 and LNK-12 are all
`confirmed`/`binding-confirmed` and each links to exactly the `svc` provider (not `job:job/mcp_admin.py::reindex`). The manifest and
test-plan forms of LNK-02/06 (`key-match`) change in 53a (owner `qa-engineer`).
- **Files:** the two pack files; `tests/test_pack_ts_mcp_client.py`; `SERVED` in `tests/test_packrt_generic.py`; the leaf-harness
  registration; `DEFAULT_PACKS` in `ccg/refresh.py` (8 to 9; grep the tests that pin the pack count and the store's `Pack` count).
- **Done when (reachable at this step, before P1-53b):** (1) over the fixture build the consumer facts exist with the keys `get_item`,
  `list_orders` (through `TOOL_LIST_ORDERS`) and `reindex` (through the wrapper); (2) an offline `Binder` test with a synthetic port
  fact (loopback 8200 owned by unit `svc`) makes an `mcp_tool` consumer in another file of unit `web` bind to `svc` under `scope: unit`;
  the mutant `scope: file` turns it red; (3) negatives (a `callTool` in a file without the SDK import, a non-string name);
  (4) the proving repository's pack facts are equal before and after (0 TS importers). The Link-level `binding-confirmed` rows
  (LNK-02, 06, 12 each linking to exactly the `svc` provider) need the 8200 fact that only P1-53b's `scripts/Dockerfile` creates: they
  are in 53a's live tier and go green at 53b.

## P1-53 · fixture conformance to the P1 binding contract (defect D3) — 53a `qa-engineer`, 53b `tdd-engineer`

Cause and evidence: `docs/reviews/connected-code-graph-p1-fixtures-binding-rca.md` (R1-R6), verified against P1-17 and
`ccg/linker/bind.py`; plan gate `docs/reviews/connected-code-graph-p1-51-53.md` (M2). Order: 51a, 51b, 52, 53a, 53b; 53a authors the pin red, 53b makes it green.
**The contract the fixtures conform to (design).**
- *Launch chain (R2).* A server binds through its `command` entry and the `Dockerfile` beside the script; the script hop is lexical and
  stored nowhere; shell `exec` launch facts and `-m` targets are P2. `launch` rows (LNK-21, LNK-22) are not stored `Link`s at P1: they
  are asserted as Binder outcomes (unit set, `BOUND_BY` entries), as LNK-15's context check already is. The fixture gains
  `fx-svc/scripts/Dockerfile` with `CMD ["uvicorn", "mcp.server:app", "--host", "127.0.0.1", "--port", "8200"]`: it gives LNK-21/45
  their chain, `svc` a `launchUnit`, and `svc` the port 8200 that LNK-12's binding needs. The CMD is **inert** (the real
  `cypher-mcp/Dockerfile` ends `python server.py`; `mcp/server.py` has no `app`): resolution is by path suffix only, and no later unit
  "fixes" it. The LNK-21/45 narrative (`python -m fx_svc.mcp.server`, fixtures plan lines 166 and 191) becomes `run-mcp.sh` to
  `scripts/Dockerfile`.
- *Prompt mention (R1).* LNK-45 is `possible`/`prompt-mention` with `BOUND_BY` ord 0 `.mcp.json @mcpServers.fx-svc-exec` and ord 1 the
  `scripts/Dockerfile` command entry (P1-17 (7), (9); AC-6: a prompt that names a tool is never confirmed). Fixtures plan §7 moves it
  to the `mcp_tool` possible cell.
- *Module path (R5).* `fx-svc/deploy/Dockerfile` CMD becomes `api.app:create_app` (unit-relative, as the fixture imports are):
  `_launched_unit` matches a path suffix; `package-dir` resolution is a P2 item (owner `architect`). It adds a second `launchUnit` for
  `svc` and no link change.
- *Shared port (R3).* `fx-job/deploy/compose.dev.yaml` (service `job-dev`, `build.context: ..`, `8000:8000`) makes loopback :8000
  claimed by units `svc` and `job`: LNK-16 is `possible`/`port-shared`. LNK-13/14/15/22 are unchanged (service hosts and port 8300).
  If the Compose extractor does not take the file, LNK-16 moves to P2 (a census change, back to the stakeholder).
- *Shell consumer (R4).* No path binds a shell consumer by URL; LNK-19's P1 form is unbound `confirmed`/`key-match` to both `reindex`
  providers (stakeholder, Q-NC5 A: P1-17's rule stands; NC-5 narrows to LNK-12, bound, and LNK-20; the P1-36 report carries the count of
  unbound `mcp_tool`/`http` links whose key has providers in more than one unit, a line `teco` adds to `-p1-gates.md` P1-36).
- *Rows that need no fixture edit.* LNK-14's base is a config value no Binder input reads: `possible`/`partial-key` (P1-49 rule 3).
  LNK-08 and TW-2: the observed forms of P1-49. LNK-18 and AC6-EX-2 move to P2 (stakeholder, Q-CENSUS A): the P1 census is **87**, the AC-6 example-config floor at P1 holds 2 instances (EX-1, EX-4); both are filed as P2 backlog items (owner `architect`) in the close list of `docs/plans/connected-code-graph-p1.md`.
- **53a (`qa-engineer`):** (1) the pin, authored red against the unmodified fixtures, two tiers. *Offline tier (no Joern, no store):*
  the real config and shell extractors over `fx-svc`/`fx-job`, plus synthetic provider/consumer facts in the style of
  `tests/linker_bind_helpers.py` (a mention `mcp__fx-svc-exec__list_orders`, a loopback `:8000` consumer, a `:8200/mcp` mcp_tool
  consumer, `reindex` providers in `svc` and `job`), through `run_packs` + `Binder` + `link`. Expected values are independent of the
  Binder: the `launchUnit` multiset **per Dockerfile** from an independent parse of each CMD and the path-suffix rule; `BOUND_BY`
  entries resolved through the bundles' records at the manifest `aux` locators (path + keyPath), never by id. *Live tier* (the
  fixture build, `-m live`): the Link-level rows LNK-45, 16, 19, 12, 02, 06. Run against the unmodified fixtures it fails on exactly
  `launchUnit` for `svc` absent, no `BOUND_BY` on LNK-45, no `port-shared` link, and LNK-02, 06 and 12 unbound (`key-match`, no 8200 fact; LNK-12 links to both
  `reindex` providers), and nothing else. Mutants, each named with its file:
  delete `fx-svc/scripts/Dockerfile`; change the CMD module of `fx-svc/deploy/Dockerfile` to `fx_svc.api.app` (red only through the
  per-Dockerfile `launchUnit` multiset); flip LNK-45 to `confirmed` in the manifest; delete `fx-job/deploy/compose.dev.yaml`.
  (2) Text, in the manifests, `docs/test-plans/connected-code-graph-fixtures.md` and `docs/test-plans/connected-code-graph-p1.md`:
  the owed `p1Form` for LNK-11/13/14/AC6-EX-4 and LNK-15 re-derived, LNK-08/TW-2 observed forms, LNK-45, LNK-19, LNK-02/06/12
  (`binding-confirmed`), P1T-06 and P1T-09 (which still say "the other five `binding-confirmed`" and assert LNK-21/22 as Links), the
  LNK-21/45 narrative; no fixture tree edit. (3) The census sweep (87 P1 rows; the families recomputed without LNK-18 and AC6-EX-2): an unfiltered grep for `89` (and `P1 93`/`P1 92`
  history lines left alone as dated notes) in `docs/test-plans/connected-code-graph-fixtures.md` (§6 totals, §7 matrix), `docs/test-plans/connected-code-graph-p1.md`
  (the family table, P1T-05, §6, the totals lines), `cpg/tests/test_fixtures.py` and the manifests' `gate` fields, which `qa-engineer`
  fixes; the `docs/plans/connected-code-graph-p1-gates.md` hits are already fixed (commit `313c59e9`), as are those of
  `docs/plans/connected-code-graph-p1.md` (87). Every hit is mapped to an owner and fixed or listed. Done: the pin red as stated, `tests/test_fixtures.py`
  consistency green, the sweep list attached.
- **53b (`tdd-engineer`):** `fx-svc/scripts/Dockerfile`, `fx-svc/deploy/Dockerfile`, `fx-job/deploy/compose.dev.yaml`, the pinned
  materialized SHAs and family counts. Done: the pin green with each mutant recorded red with its failure message; the link multiset
  of the fixture build (with the P1-52 pack present) differs from the pre-edit one by exactly five rows: LNK-16 `port-shared` appears; LNK-45
  gains `BOUND_BY` ord 0 and 1; LNK-02 and LNK-06 go `key-match` to `binding-confirmed`; LNK-12 loses its `job` provider link and its `svc`
  link becomes `binding-confirmed`. Any other changed row stops the unit. Must not regress: LNK-13/14/15/22, AC6-EX-1/4, `UNIT_FLOORS` (68, 67), the walk rows.
- **53c (`tdd-engineer`, serial after 53b; it regenerates the same pinned SHAs and family counts):** the AC-6 example-config floor at P1
  (2: AC6-EX-1, AC6-EX-4) counts instances whose P1 form carries the `example-config` reason, not manifest rows gated P1. EX-4 does not
  today: `fx-svc/clients/peer.py` reads `os.environ["FX_PEER_BASE"]`, and `bind.py` reads only `environ.get`/`getenv` (`_ENV_READS`), so
  its link is `possible`/`partial-key`. Edit `peer.py` to `os.environ.get("FX_PEER_BASE", "")` **and** `deploy/compose.example.yaml`'s `FX_PEER_BASE` to `http://api:8000`:
  `example-config` is the reason of a link whose only binding chain runs through a file with `role: example` (the chain is weakened,
  `bind.py` `weaken`; requirements decision S21: example configs are *possible callers*), so the example must point at something the
  binder resolves (service `api` is declared in the primary `compose.yaml`, unit `svc`, the row's true target); the placeholder host
  `peer.example.internal` resolves to nothing and yields an unbound `partial-key`, a different AC-6 shape. Drop EX-4's `p1Form` from
  the manifest (its `reason: example-config` is the P1 form) and adjust its `item` text. Done: the pin's Link tier asserts EX-4
  `possible`/`example-config` through the example Compose chain, and no test pins the old host string (none found by grep).

## P1-17 · linker — `tdd-engineer`

- **Files:** `cpg/ccg/linker/__init__.py`, `matchers.py` (`exact`, `path-template`, `glob`/`prefix`, `regex`),
  `bind.py` (URL, env/Compose and launch bindings), `link.py` (`link(facts, bundles) -> Links`),
  `cpg/ccg/registry/reasons.yaml` (the `on` lists gain: `binding-confirmed` +`VIA`; `shell-dynamic` and `prompt-mention` +`Link`, since the linker derives them from the site (behaviours (3), (7)); `lifted-role` is **not** widened, it stays a `CONSUMES` reason and never a `Link` reason (see Interface); `pack validation` is unaffected because it reads only the four pack-emitted edges; `test_the_registry_file_is_the_seeded_registry`'s seed is updated in the same edit, and bundle spec3 §7.2's table is amended at the feature's close), `cpg/tests/test_linker_*.py`,
  `cpg/tests/slice/test_linker_conformance.py`, and the leaf harness `cpg/tests/test_rule_pack_leaf_mutation.py`: empty
  `UNEXERCISED_UNTIL_P1` of the last 2 `fastmcp:urlBindings` leaves and extend `GATE` with P1-17's test files (`GATE`
  runs only the runtime tests it names, so a leaf read only elsewhere stays a survivor). That edit is P1-17's
  done-condition. Two dispatches (matchers+link first).
- **Hand-off from the P1-16 gate (`ccg.packrt`):** (a) a fully unknown-key consumer exists only in `PackFacts.lifted`
  (`resolved: False, leaf: True`); behaviour (3) reads it from there, not from `consumers`. (b) `run_packs` raises
  `UnsupportedSubject` on `paper-inhouse-bus` (a `query` subject); six of the seven packs run over s14a, and the caller
  names the pack set explicitly. (c) `matchers.py` reuses `ccg.packrt.httpkey.normalize_path_rfc`, no second normalizer.
- **Trust rules for launch and binding facts** (they come from untrusted repo files): (a) the linker never dials, opens or reads a
  path named by a `mount`, `port` or `command` fact; it resolves lexically only. (b) A `mountPath` is joined to its compose file's
  directory and normalised without touching the file system; the result must lie under a repository path of the system manifest
  (a sibling repo is legal, `../../fx-job`), else the link is `possible`/`target-not-mapped`; an absolute path or one that leaves
  every manifest repo is never followed. (c) A service's identity is `(origin, repoId, path, service)`: a `.mcp.json` server and a Compose
  service of one name are different services. (d) A `bindValue` or `bindParts` host in `169.254.0.0/16`, `fe80::/10` or
  `fd00:ec2::254` is never bindable, whatever writer produced the bundle (P1-11 already withholds it, in `writer_rules.bind_fields`). (e) `port` facts carry no published/container distinction in P1 (`-p1-extraction.md` P1-11): a loopback consumer's match through any Compose `port` fact (lone `N`, `expose`, long syntax, `A:B`) is **counted** once per site in `counts["loopback_port_residual"]` and stays as matched (a count, not a weakening) until `portKind` exists. Tests: `../../../../etc`
  and `/var/run` bind nothing; same-name `.mcp.json` and Compose services stay apart; a metadata-address `bindValue` planted by
  hand binds nothing.
- **Interface:** `Link = (linkId, kind, keyNorm, provider anchor, consumer anchor, status, reason, matcher, paramMap,
  evidence: ordered BOUND_BY and LIFTED_VIA entries)`; ids from `ccg.ids.link_id` (one `Link` per (provider, consumer
  **symbol**), `-bundle-spec3.md` BF6, `-p0-graph.md` D6, as the slice oracle `assemble.py` builds it).
  - **`keyNorm` and the `linkId` inputs:** `Link.keyNorm` is the **provider's** `keyNorm` (the `ContractPoint` the `Link` is `ON`);
    `linkId = link_id(kind, provider keyNorm, provider anchor, consumer symbol)`. Sites of one consumer symbol with different consumer
    keys (`GET /a/1`, `GET /a/2` against `GET /a/{}`) are `VIA` entries of **one** `Link`; the consumer's key lives on each `Via`
    (`consumerKey`, D2) and is what `candidateCount` (D6) groups by. A keyless or name-only provider has `keyNorm` `""`. A link into a
    never-built unit has no provider, so `keyNorm` is the consumer's key (`""` if keyless) and the anchor is below. A keyless and a keyed
    consumer site reaching one provider share one `Link`. `-bundle-spec3.md` BF6 is silent on which key and on the unit anchor: its
    one-line amendment (provider key; the `"unit:"`/`"unmapped:"` anchors below) is owed at the feature's close.
  - **Status:** a *site's* status is the weakest hop on its chain; `Link.status` is the **best over its sites**
    (`-p1-graph.md` §2, v1 D2, which governs). `Link.reason` follows O-9 (behaviour (9)) and is always a registry row valid on
    `Link`: never `lifted-role`, which is the `CONSUMES` edge's reason, so an unbound confirmed `http`/`mcp_tool` link reads
    `key-match` and a bound one `binding-confirmed`. Fixture rows LNK-10/11 therefore expect `key-match`, and LNK-12 (which has the
    `localhost:8200/mcp` binding) `binding-confirmed`; the `reason: lifted-role` on those rows in `fx-web.yaml`, `fx-job.yaml`,
    `fx-svc.yaml` and `cross.yaml` is the consumer edge's and is corrected before P1-41.
  - **`matcher`** (a stored `Link` prop, `-p1-graph.md` G12) is the strongest matcher a key comparison applied (of the best site, lowest `siteId` on a tie, when a link has several): one of `exact`,
    `path-template`, `glob`, `prefix`, `regex`; `name` for a `name-only-provider` link; `none` where no key comparison ran
    (`unknown-key`, `binding-narrowed`, `target-not-mapped`). The linker's tests assert membership in this closed set.
  - **`paramMap`** is always `{}` in P1: no pack declares one and the graph does not store it (G12 counts it in
    `dropped["link-param-map"]`). The first pack that declares a mapping owns adding it (a P2 item, owner `architect`).
  - **`linkId` anchor of a link with no provider:** `"unit:" + unitId` for a never-built unit, `"unmapped:" + ref` for a target outside every
    manifest repo (trust rule (b)); neither can equal a hex symbol or entry id.
  - **Dispatch-1 fix-ups (before it is committed):** the `reasons.yaml` rows above, the `linkId` anchor, and the regex guard of
    behaviour (14). **Dispatch 2 changes the seam (briefing):**
    - *Anchors.* A `configEntry` record has `id, repoId, path, line, keyPath, example` and **no `unitId`**; a `launchFact` has `id, path`
      and no `repoId` either. `_Index` therefore walks the **bundles** (not the flattened `Corpus`): the header's `unitId` and `repoId`
      apply to every `configEntry`/`launchFact` record of that bundle, so `unit(anchor)` is the bundle's unit; `role(anchor)` is `example`
      iff a `configEntry`'s `example` is true or its `file` record (looked up by `(repoId, path)`) has `role = example`, else that file's
      role (a `launchFact`'s file by the bundle's `repoId` and its `path`). Providers and consumers anchored on a record id therefore
      survive a unit-restricted binding and weaken under `example-config`.
    - *`Binding`* gains `weaken: tuple[str, str] | None` (`("possible", <reason>)`, the reason a registry code valid on `Link`, checked in
      `__post_init__`) for a possible hop inside the chain (the inferred Compose rule, trust rule (e)'s loopback residual); `_causes` places
      it with the binding causes, after `port-shared`. `unmapped_unit: str | None` is replaced by
      `unmapped: Unmapped | None`, `Unmapped(unit_id: str | None, ref: str)`: `unit_id` is the never-built unit, or `None` for a path outside
      every manifest repo; `ref` is a stable value-free hex id (`tuple_id(["unmapped", normalised path])` in the second case, the unit id
      in the first). `Link.to_unit = unit_id`; the assembler writes `TO_UNIT` iff it is not `None`; the `Link` is `possible`/`target-not-mapped`
      either way.
    - Tests: a config-entry provider and a config-entry consumer under a one-unit binding; an `example` config entry gives `example-config`;
      a `weaken` chain gives a `possible` link with its reason; an out-of-manifest `mountPath` gives `target-not-mapped` with `to_unit None`.
- **Behaviours:** (1) both keys known and matching → `confirmed`/`key-match`; (2) partial or name-only → `possible`
  (`partial-key`, `name-only-provider`); (3) a fully unknown key with no binding → one `possible` link per
  provider of the kind (`unknown-key`), none collapsed; **the link's reason is the consumer call site's own when that is `shell-dynamic`**: a `possible`/`shell-dynamic` site (an `eval` string, P1-10 10b) gives the `Link` reason `shell-dynamic`, which wins over `unknown-key` and `runtime-built` (the linker derives it from the site, since the `CONSUMES` edge cannot carry it), as fixture row LNK-20 expects, which stays `gate: P1` as written once 10b, P1-46 and this rule land; (4) **URL binding:** a consumer URL's port is matched to
  launch facts (explicit flags, env-defaulted values, framework defaults from the pack, Compose mappings), the
  path to the unit's mount paths; on `localhost`, several deployments of one unit (8000 and 8200 serving
  `falkor-chat/server`) are one provider and the link is `binding-confirmed`; two different units on one port are
  `possible`/`port-shared`; with `bindParts` (P1-14) the comparison is on `hostHmac`/`pathHmacs`; (5) env/Compose
  binding: base URL from env → Compose sets it per service → service `build.context` → unit; **the build context
  may lie outside its own repo (`../../fx-job`): the unit index is system-wide** and the case is the first real
  test of `-p0-compose-binding.md`'s inferred rule; if it fails the fallback is a bundle-spec image-name rule,
  which is escalated to `teco`, not patched in a fixture; (6) launch binding: `.mcp.json` server → command → script
  → container → unit (the AC-4 `cypher` chain; **P1 chain shape:** the script hop has no `BOUND_BY` entry because no extractor emits a shell
  script's `exec`/`docker run` launch fact (P1-10 writes no `launchFact`), so the chain is the server's `command` launch entry, then the
  `Dockerfile` `CMD`/`ENTRYPOINT` entry beside the script, the script hop is resolved lexically by the binder and **not stored** in P1; the
  graph note's §2 row naming the script's launch entry and G8's non-`launchUnit` `LAUNCHES` rows are amended at close). **Hop 1 input** is P1-11b's (`-p1-extraction.md`): the real repo-root
  `.mcp.json` names its script only inside `bash -c 'exec "$CLAUDE_PROJECT_DIR/cypher-mcp/docker-run.sh"'`, which the argv gate redacts, so the
  chain binds only once P1-11b emits the script path; the conformance chain test runs over the real extractor output on the repo's own
  `.mcp.json` and `cypher-mcp/`, never hand-patched records. Written, like every binding, as `BOUND_BY` edges in chain order
  with `ord` 0 the entry nearest the consumer; `ord` counts per (link, site) and every `BOUND_BY` edge carries its
  `siteId` (`-p1-graph.md` §1–§2; `-answer-spec2.md` §3.2); (7) `File.role = 'example'` makes every link through it
  `possible`/`example-config`; a prompt mention is `possible`/`prompt-mention`; documentation snippets are not
  callers; **mention call sites have this consumer:** the linker binds only tool mentions (an `mcp__<server>__<tool>` name, or a
  bare name the extractor matched against `known_tools`, P1-44) to providers of that tool, **stripping a leading `./` from a callee
  name before matching** (the extractor keeps names as written, spec3 §5.6); script and command mentions (about 94% of the
  corpus's 1,868) are inert in P1: no link, they stay `CallSite` nodes. Test: `./x.sh` and `x.sh` match one provider; a script mention makes no link; (8) a link into a unit never built is `possible`/`target-not-mapped`; (9) O-9, the `-p1-graph.md` §2 rule for every link kind: a
  `VIA`'s reason is `binding-confirmed` iff the `VIA` is `confirmed` and the link has a `BOUND_BY` edge with that site's
  `siteId` (a confirmed unbound site keeps `key-match`; a `possible` site keeps its own reason), and `Link.reason` is
  `binding-confirmed` iff some confirmed site is; mutants: dropping the `siteId` condition, deriving the `Link` reason
  from "any `BOUND_BY`"; a framework-default port is evidence from the pack, represented as the ruling says; (10) MCP path
  matching is prefix-only (O-5); (11) a `Link` count at proving-repo scale is reported (the deferred "Link counts
  under runtime-built URLs at scale"); (12) **F-13:** every `http` key goes through `ccg.keynorm` and the RFC 3986 order (percent-decode before dot-segment resolution): a test asserts `GET /a/%2E%2E/b` becomes `GET /b` in `matchers.py`, never the old order; (13) **`bindParts` key
  epoch (`-bundle-spec5.md` §5.4, BR-1–BR-3):** `bind_parts_usable` gates every bundle's parts, an unusable bundle's
  URL entries bind by parts derived from `bindValue`, and `link()` returns the `bind-key-mismatch <unitId>` lines
  beside its links for P1-20's refresh report; (14) **regex keys are repo-derived text under Python `re`, which is not linear time:** a `keyForm: regex` key (the `kafka` pack's `pattern`) and a `link` matcher `regex` are matched only when `matchers.py`'s guard admits the pattern: `ccg.rulepack.regex_problem` clean, no quantified group that holds a quantifier or an alternation (`(a+)+`, `(a|b)*`), at most 4 quantifiers in total and at most 2 unbounded (fuzz evidence `cpg/.ccg-work/p1-17b/`; a pair cap too), the quantifier grammar read as Python `re` accepts it (`{,N}` included), and the literal it is matched against at most 256 characters. A refused pattern matches nothing and is **counted** in `Links.counts["refused_patterns"]` (a literal refused as too long in `refused_texts`), never silently dropped. The linker's other count keys: `links`, `by_kind`, `by_status`, `by_reason`, `unknown_key_consumers`, `max_fan_out` (`link.py:count_links`), and the binder's `loopback_port_residual` (`Binder.counts`). No timeout (an in-thread one is unreliable). Tests: `(a+)+$` against `aaaa…!` returns at once and is counted; a plain pattern still matches. `ccg.rulepack` is unchanged (pack regexes are pack-authored). The `matchers.py` docstring's "RE2 subset" claim is reworded to "the RE2 subset plus this guard" (it is true only with the guard). A linear-time engine (an NFA over the RE2 subset, or an RE2 binding, a dependency decision) is a P2 item, owner `architect`.
- **Test owed (mutant F5):** `{**}` is accepted only in the last provider segment: `GET /a/{**}/b` against `GET /a/x/b` does not match. **Registry test (M3):** every reason a `Link` or `Via` emits is asserted to be a registry code whose `on` list holds `Link` (a `Via` reason: `Link` or `VIA`), not only its status. The `packs=` error says "no rule of the given packs consumes lifting X" when packs were passed.
- **Binding decisions:** (a) a prompt mention `mcp__<server>__<tool>` binds only through that server's own `.mcp.json` entry (URL or launch chain) **in the mention's own repository** (the fixtures' mentions and the real monorepo's `.claude/` mentions sit in the same repo as their `.mcp.json`; another repo's config never attributes a mention, and the system manifest has no root-repo notion to extend this), never through a client URL elsewhere in its file; unbound it keeps the unbound rules. (b) A script launch reaches its container through the `Dockerfile` beside the script, and the unit through `targetArg`, else the unit rooted at that directory (none when several units share the root). (c) A primary chain beats `example` chains; a built unit beats an unmapped one. (d) An `ip`-literal host (private ranges included) binds nothing. (e) A Compose service declared only in `example` files never claims a loopback port. (f) The consumer's environment-variable **name** is read only where a bundle record holds it (a Python `os.environ.get`/`getenv` flowing into the URL slot); a JavaScript `import.meta.env.X` or a Python module constant `X = os.environ.get(...)` leaves no record (`normalize/core.py` drops a call-valued constant), so LNK-13 (and LNK-15's default) stay unbound at P1: **decided (stakeholder, 2026-11-11): deferred to P2.** LNK-13 and LNK-15 are unbound at P1 and, their base being an unknown (a call-valued constant, an environment read), are `possible`/`partial-key` with their path key (P1-49 rules 2-3: an unknown leading origin forfeits `confirmed`); the **P2 backlog item** (owner `architect`) is a bundle minor 1.2 value element for environment reads (JS `import.meta.env.X`/`process.env.X`, Python module constants from `os.environ`), with the normalizer, `packrt` and linker reads that bind LNK-13/LNK-15 (and `AC6-EX-2`'s Vite variable) by name. Fixture rows to change (`qa-engineer`): `LNK-13` in `cpg/fixtures/system1/expected/{cross,fx-svc,fx-web}.yaml`, `LNK-15` in `{cross,fx-svc,fx-job}.yaml` (`reason: binding-confirmed` becomes `possible`/`partial-key`, the `refs` Compose lines move to a P2 note); `docs/test-plans/connected-code-graph-fixtures.md` §LNK-13/LNK-15 rows (lines ~154, 156), the cross-repo build-context sentence (~185) and the `config-bound base URL` count row (~489); and check `AB-4`'s `LNK-15` clause (~360): a never-built `fx-job` gives no `target-not-mapped` link for an unbound consumer, so that clause also waits for P2. `OV-2` is already P3.
- **`LAUNCHES` beyond `launchUnit` is P2 (ruling):** no P1 reader or answer hop reads `LAUNCHES` (`-answer-spec2.md` §3.4 and V19 prohibit it as a hop; the launch chain travels as `BOUND_BY` evidence), so no linker launch-target export is built; P2 backlog item (owner `architect`): `Links.launches` built by the binder from the chains it resolves (Compose `build.context` → Unit, entry → script `File`, Dockerfile hops) with the assembler wiring.
- **Owed at the feature's close (spec amendments this step creates):** `-bundle-spec3.md` BF6 (the `Link` key is the provider's `keyNorm`; the `"unit:"`/`"unmapped:"` provider anchors); spec3 §7.2's table (`shell-dynamic` and `prompt-mention` valid on `Link`); `-p1-graph.md` §2's launch-chain row (the script's launch entry: in P1 the script hop has neither a `BOUND_BY` entry nor a `LAUNCHES` edge; only the pack runtime's `launchUnit` edges, command → Unit, exist).
- **Done / Gate:** conformance with the slice's 11 links and chains; the fixture rows LNK-* of the test plan that
  gate at P1 pass in P1-41. `analyst`; mutation check on status weakening and port-sharing.

## P1-18 · assembler — `graph-dba`

- **Files:** `cpg/ccg/assemble/__init__.py`, `rows.py` (bundles + links → `GraphRows`, ported from
  `spikes/vertical-slice/load.py:build_rows`), `snapshot.py` (`:Snapshot` content), `cpg/ccg/registry/
  flow-availability.yaml` (index D2), `cpg/ccg/sandbox_audit.py` and `cpg/ccg/schemas/sandbox-record-v1.schema.json`
  (`-egress-spec3.md` §5.3–§5.4), `cpg/ccg/staleness.py` (created here with the closed constant `FAIL_REASONS`, the
  eight codes of `-p1-graph.md` §1 G4; P1-20 adds `unit_staleness` to the same file), `cpg/tests/test_assemble_rows.py`, `cpg/tests/slice/
  test_assemble_equivalence.py`.
- **Behaviours:** (1) equivalence first: on the slice bundles the node and edge sets equal the slice's
  (38,121 nodes, 90,867 edges recorded in `-p0-graph.md` §1.1) plus only the deltas below; (2) D8's rule: `DATA`
  and `CALLS` store `status`/`reason` only when not the default; (3) `MOUNTED_BY {ord}` with `ord` by
  (mount path, line) (the kinds a unit-level mount serves are not in `launch_facts`: P1-18 derives them from the pack's
  `urlBindings[].kinds`, and `ccg.packrt` is unchanged), provider `symbolId` and mount `callSiteId` as ids (the assembler requirement of
  `-p0-graph.md` D5); (4) `SkippedFile` nodes under `Unit -[:SKIPS]->`; (5) `Unit.flowAvailable` from the registry,
  one row per language (python false, javascript false, others false, kotlin absent = unmeasured), each row citing
  the fitness report section; **no other code reads a flag from anywhere else**; (6) the **assembler imports
  `ccg.keynorm.mask_http_segments` and a test fails if step 8a is defined anywhere else**
  (`-eval-protocol2-ml.md` §3); (7) a bundle is refused at assembly unless
  `ccg.sandbox_audit.check_sandbox_ids` passes every `sandboxAuditIds` entry (`-egress-spec3.md` §5.4: exactly one
  schema-valid, non-refused, exit-0, `offline` record each, read from the runner's `sandbox.jsonl`; P1-19 calls the
  same function; no broker dependency); (8) the pack set's identity (ids,
  versions, content hash) is written where the `-p1-graph.md` ruling (O-10) says; (9) `candidateCount` is **not**
  stored (`-p0-graph.md` D6); (10) the system name is checked again here by `ccg.sysname.check_system_name`, and every key is formed by its
  constructors; (11) a `Unit.failReason` outside `FAIL_REASONS` is refused, and a failing bundle that carries a planted
  `bindValue` and an exception message yields a `failReason` containing neither (`-p1-graph.md` §8 P1-18 (i)).
- **Done / Gate:** `analyst`; `graph-dba`'s own note is the contract. Two builds of the same bundles give identical
  rows (a determinism test).

## P1-18b · unindexable values fail their unit, not the system — `graph-dba`

- **Problem:** engine v4.18.11 stores an indexed string over 4,096 UTF-8 bytes unindexed, trims ASCII whitespace at the ends of stored values (not of lookups), and a UNIQUE constraint over a value above 5,000 bytes crashes the server, so the loader's preflight refuses such a row; one odd file name would then block every refresh of the whole system. The properties that realistically carry them: `File.path` (a hostile or odd file name), then `Symbol.qname` and `ContractPoint.keyNorm`.
- **Ruling (no spec change):** the **assembler** checks, per bundle, every indexed property the loader indexes (one shared predicate and property list in one module, imported by both; the loader's refusal stays as the backstop) and refuses **that unit's bundle** as `bundle:unindexable`, a new code of `FAIL_REASONS` (`ccg/staleness.py`; the `-p1-graph.md` G4 row gains it: "`bundle:unindexable` (a bundle holds an indexed string over 4,096 UTF-8 bytes, edged by ASCII whitespace, or not valid Unicode; the one unit is stale-kept or not-mapped, the system still loads)"). Value-free: no path or key text in the reason. The blast radius is one unit, with its last good bundle kept. Rejected: skipping the file as a `SkippedFile` (`skippedFiles` admits only `test` and `frontend-ignore`; a new reason is a bundle minor 1.2 and a registry row, a **P2 backlog item**, owner `architect`, which would let the unit build without the file) and keeping the whole-build refusal.
- **Files:** the shared predicate module (`cpg/ccg/indexed.py`), `cpg/ccg/assemble/rows.py` (`classify_bundle` calls it), `cpg/ccg/store/loader.py` (imports it), `cpg/ccg/staleness.py`, tests: a trailing-space path, a 4,100-byte path, a lone surrogate each fail only their unit and a second unit still loads; the loader still refuses the same row when it reaches it. **Gate:** `analyst`. **Blocks:** P1-20.

## P1-19 · store loader, ACL, swap, stamp — `graph-dba`

- **Files:** `cpg/ccg/store/loader.py` (`load(store, rows, system, report)`; the slice's `LiveStore`, `zero_retry`,
  `batches`, `poll_constraints` ported), `cpg/ccg/store/acl.py` (`create_users(admin, system)`; the four `ccg_<role>_<s>` users of `-p1-graph.md` §7, named
  through `ccg.sysname.acl_user`), `cpg/ccg/store/swap.py` (`swap(build_conn, verdict_conn, system, build_id)`: the
  lock, the pre-`RENAME` replay onto the build key and the `RENAME build live` run as `ccg_build_<s>`, the post-`RENAME` remainder check and replay as
  `ccg_verdict_<s>`, never as admin; the verdict-lock acquire/release with an **empty verdict log** and a no-op
  replay hook `replay_verdicts(...) -> 0` that P4 fills), `cpg/ccg/store/ddl_v1.cypher`
  (adds the `ConfigEntry.entryId` index of EP §5.2.1 HA-1 and any index `-p1-graph.md` adds, index-before-constraint
  order unchanged), `cpg/tests/test_loader_*.py`, `cpg/tests/test_acl_live.py`, `cpg/tests/test_fuzz_idhash_live.py`.
  Two dispatches (loader+swap; ACL+fuzz+measurements).
- **Behaviours (live where marked):** (1) indexes on the empty `ccg_<system>__build_<id>`, then nodes, then edges in
  `UNWIND` batches of at most `BATCH_MAX = 20,000` (never near the measured 100,000 ceiling), one connection,
  retries zero, a per-query `timeout` ≤ `TIMEOUT_MAX` whenever it is nonzero, a created-count check per edge batch;
  constraints after the load, polled to `OPERATIONAL`, a `FAILED` fails the build; `:Snapshot` last; then the lock,
  plain `RENAME build live` only (`-p1-graph.md` §7); **zero** old snapshots
  kept; (2) **ACL test on a throwaway instance** (`infra/ccg-falkordb.sh --throwaway`, P1-48): it creates the four
  `ccg_<role>_<s>` users of `-p1-graph.md` §7 through `ccg.sysname.acl_user`, plants `shop` beside `shop2`, `shop`
  beside `shop__x` and `shop` beside `shop_`, and asserts the grants of `-p1-graph.md` §8 P1-19 (c), with the old flat
  `~ccg_shop*` pattern as the negative control (it grants `ccg_shop2`); the names `shop__x` and `shop_` are
  **refused by `check_system_name`** before any key exists, and the test shows the pattern alone would grant their
  keys, which is why the whitelist is required and not optional;
  (3) system names are validated by `ccg.sysname.check_system_name` before any key is formed; (4) **id-hash fuzz** (long,
  whitespace-edged and odd-character keys; a `nameKey` over 4,096 bytes must not reach a constraint unhashed)
  against a **throwaway instance only**; (5) a mid-load kill leaves the live key intact and the build key
  removable; (6) the loader refuses to run with a `shop`-prefixed key it does not own.
- **Measurements this step reports** (P1-scale replaces P0's): the worst per-edge-type p99 read latency on the
  loaded proving repo, from which `TIMEOUT_MAX` is **re-derived** with the P0 rule (10 × worst p99, rounded up
  in the 1-2-5 series; P0's 10,000 ms is a default, not a design constant); the lock ttl re-derived as 3× the
  worst (replay + `RENAME` + remainder check); the memory per file and the **~18 GB peak re-estimate**
  (`GRAPH.MEMORY USAGE` and `used_memory` delta, as the slice report did); `Link` counts at scale.
- **ACL grants:** the four users' lines are `-p1-graph.md` §7, which supersedes `-graph.md`'s `~ccg_<system>*` line
  (the graph note is not edited: it is `Extended by:` `-p0-graph.md` and executed against). `create_users` writes each
  system's passwords to `~/.config/ccg/keys/<system>.acl` through `ccg.atrest.write_acl` (P1-02), never to stdout or
  a committed file. **Gate:** `analyst` (the gate runs the planted ACL test itself, on a throwaway instance); mutation
  check on the lock release and the ACL pattern.

## P1-44 · build dispatch by unit kind — `coder`

- **Files:** `cpg/ccg/build.py` (`build_unit(unit, repo, commit, system) -> BundleResult`), `cpg/tests/test_build_dispatch.py`,
  `cpg/tests/test_build_live.py`, `cpg/extractors/joern/run.py` (the one-class addition `RunnerTimeout(RunnerError)`, raised where
  `_run` times out) and `cpg/tests/test_joern_run.py` (a timeout raises it).
- **Behaviours:** (1) the unit's kind selects the extractor: python and javascript units → P1-04 run + P1-05
  normalizer (+ P1-15 SCIP merge when adopted); shell files → P1-10; JSON/YAML/TOML/Compose/`.mcp.json`/Kiro/
  Dockerfile → P1-11; Markdown → P1-12; each non-Joern extractor writes **one bundle per unit** through
  `ccg.atrest.write_bundle`, so a per-file extractor is run over the unit's files and merged into the unit's bundle; **the unit's header is built from the merged record set** (see (8)); (2) an
  unknown kind is a typed error, never a silent skip; (3) a failure in one extractor fails **that unit's bundle**
  and nothing else, and sets one code of `FAIL_REASONS` (`-p1-graph.md` §1 G4), value-free: **the mapping is this step's:**
  `blame.HistoryError` from `require_full_history`, called before extraction → `build:shallow-clone`; `RunnerTimeout` →
  `extract:timeout`; every other `extractors.joern.run.ExtractError` (the base, so a new subclass is covered) and
  `SourceRefused` (the one class of `extractors/common/errors.py`, whichever extractor raises it), `normalize.core.RuntimeRefused`, `assemble_bundle.BuildError` and any other `HistoryError` → `extract:failed`;
  `ccg.bundle.BundleError` from `write_bundle` → `bundle:invalid`. **Any other exception** (`subprocess.CalledProcessError`,
  `TimeoutExpired`, `atrest.AtRestError`, an unforeseen one) **fails that unit alone under `extract:failed`** and never aborts the
  refresh. The reason is the code only: no exception text, no source bytes. Test: a unit holding a Latin-1 `.py` is driven through
  `build_unit`, fails alone (a second unit still builds), `failReason == "extract:failed"` and no source byte is in it; one planted exception per row of the mapping and one unlisted (`CalledProcessError`), which fails its unit alone; (4) bundles and keys are created through `ccg.atrest` only; (5) a test per kind asserts the
  extractor is actually invoked (a stub extractor that records calls), so no extractor is reachable only from its
  own tests; (6) **Markdown hand-offs (production globs):** the prompt-bearing-Markdown globs come from the **system manifest's** `promptBearingMarkdown` block (below) and are passed explicitly (`ctx.globs`); `build.py` never defaults to `cpg/eval/eval-manifest.yaml`, whose role is an eval/fixture input only, and `load_globs()` takes the manifest path as a required argument (the default is removed). A refresh whose Markdown units have no globs fails closed, each such unit with `extract:failed` (value-free), never reading another manifest; test: a refresh with no explicit globs fails closed. The block is a **P1-03a** addition (below). (6a) **Vendored directories:** P1 keeps the global `units.SKIP_DIRS`; the header's `vendoredDirs` is `sorted(SKIP_DIRS)`, exactly what `_unit_files` skips, so it stays honest (a test asserts the header list equals the skip set and equals the eval manifest's `vendoredDirNames`, the census scope); a per-repo list in the manifest is P2.
  (6b) **Markdown hand-offs, continued:** a `MarkdownError` (an oversize file) **skips that file** and counts it in the build report
  (`skippedDocs`); it does not fail the unit, and no `skippedFiles` reason is used (the field is not written for Markdown units in
  v1); P1-44 passes `extract_markdown` (a) `history`, the file's `(lastChangedAt, lastChangedCommit)` from the source repo at the built
  commit through `normalize.blame` (`last_changed` over the whole file, `path_last_commit` for an empty one), (b) `bundle_path` when the
  path in the bundle differs from the repo path the manifest's globs match, and (c) `known_tools`, the system's MCP tool names,
  computed once per build by running the P1-16 mcp pack (`ccg.packrt.run_packs`) over the system's code bundles built before the
  Markdown units (code units first); tests: the stub extractor records all three arguments, an oversize file is skipped while its unit
  builds, and the `fx-svc` fixture's two prompt files yield the AC6-PM call sites end to end; (7) **Config call convention:** `extract_config` is called **once per repository** with all that repository's Compose files (W2's declared-service set is per repository and primary-role only), or with `extra_declared_services` for a split call; for a config blob that produced no records (above the extractor's 512 KiB cap up to 8 MiB, or unparseable) P1-44 writes a `File` (contentHash, lineCount) with its role from `file_role`, since `ConfigResult.roles` is empty for it; a blob over 8 MiB is skipped and counted with **no `File` record** (a `File` needs `contentHash`/`lineCount`, and `skippedFiles` admits only `test` and `frontend-ignore`); (8) **Header merge (`bindParts` obligation, `-p1-extraction.md` P1-14):** before `write_bundle`, P1-44 takes `fields = writer_rules.bind_header_fields(ccg.atrest.hmac_key(system), merged_records)` over the unit's **merged** records (never a union of per-extractor `ConfigResult.header_fields`) and sets `header.bundleSchemaVersion` (`"1.1"` iff any entry has `bindParts`, else `"1.0"`) and `header.bindKeyCheck` (present iff `"1.1"`) from it, so V1a-V1d hold; the key is read through `ccg.atrest` only (behaviour (4)). Tests in `test_build_dispatch.py`: a config unit with a URL entry builds and validates as `"1.1"` with `bindKeyCheck == ccg.atrest.bind_key_check(key)`; a code-only unit and a config unit with no URL entry build as `"1.0"` with no `bindKeyCheck`; a multi-file config unit whose `bindParts` entries come from two files gets one header.
- **`sandboxAuditIds` for host-extractor bundles (ruling):** an empty list is valid exactly for a host-extractor unit (`shell`, `config`, `markdown`); the rule, the rejected alternatives (a minted host `sandbox` record would assert a run that never happened; routing host extractors through the sandbox is a P2 question) and the deltas are `docs/plans/connected-code-graph-bundle-spec6.md`. **P1-44 passes `[]` only from its host-extractor branch, chosen from the manifest unit's kind**, a code unit never gets `[]` (test in spec6 §3), and P1-44 does not ship before P1-44a: until then it keeps raising `MissingAuditIds` for a host bundle.
- **Done / Gate:** `analyst`. **Needs:** 03, 04, 05, 10, 11, 12, 16 (the pack runtime that names the tools).

## P1-03a · `promptBearingMarkdown` in the system manifest — `coder`

- **Files:** `cpg/ccg/schemas/system-manifest-v1.schema.json` (an optional top-level `promptBearingMarkdown: {include: [glob, 1..64], except?: [glob]}`, same alphabet and caps as `extractors/markdown/globs.py`), `cpg/ccg/manifest.py` (loads and validates it through the one glob validator, `globs.py`'s), `cpg/ccg/system.yaml` (the proving system's block, a copy of the eval manifest's; a test asserts the two equal, as it does for `exclusions`), `cpg/fixtures/system1/system.yaml` (the fixture system's block covering its two prompt files), `cpg/extractors/markdown/globs.py` (`load_globs(path)` with no default), `cpg/tests/test_manifest.py`, `cpg/tests/test_build_dispatch.py` (the fail-closed refresh test). **Gate:** `analyst`. **Blocks:** P1-44 close.

## P1-44a · host `sandboxAuditIds` (spec6) — `coder`

- **Files:** `cpg/ccg/schemas/bundle-v1.schema.json` (the `if`/`else` conditional), `cpg/ccg/bundle.py` (`HOST_LANGS`, defined once; rule S6a), `cpg/ccg/sandbox_audit.py` (imports `HOST_LANGS`, the host-empty branch of `check_sandbox_ids`, `attestation`; P1-18's file, delivered), `cpg/ccg/assemble/rows.py` (`unattested_units` in the assembly report; the `minItems` pop and its comment removed), and the tests of `-bundle-spec6.md` §3 (1)-(3), (5). **Gate:** `analyst` + `security-expert`. **Needs:** 18 (delivered). **Blocks:** P1-44 close and P1-20. Also in this step: `classify_bundle` applies S6a itself, `maxItems: 64`, `attestation` by language, the `HOST_KINDS`/`HOST_LANGS` drift test (spec6 §1-§3). **P2 backlog item (owner `architect`, stakeholder-level):** stored bundles carry no integrity tag, so anyone who can write the work directory can author a bundle; an HMAC tag on bundles, like the entry keys, is the candidate, not decided.

## P1-20 · refresh — `coder`

- **Files:** `cpg/ccg/atrest.py` (`sweep_stale_tmp(system, older_than_s=3600)`: removes `.<name>.<8 or 16 hex>.tmp` files left in the system's bundle directory by a crashed build write (`atrest.write_private` draws 8 hex digits, `bundle.write_bundle` 16; `write_private` is unified to 16 in the same step, and the sweep keeps accepting 8 for leftovers); touches nothing else) with its test in `cpg/tests/test_atrest.py`, `cpg/ccg/refresh.py` (`refresh(system, scope=None) -> RefreshReport`; it calls the sweep first, and the report counts what was removed), `cpg/ccg/staleness.py`
  (adds `unit_staleness(unit, head)` to the file P1-18 created with `FAIL_REASONS`; the **one** staleness module,
  which `engine/` imports in P1-39); `refresh` also takes
  `override=None`, and only when one is given opens a broker `refresh` run, fails if the broker is absent, and names
  the `policy` record in its report (`-egress-spec3.md` §2.1, §2.5); `cpg/tests/test_refresh.py`,
  `cpg/tests/test_refresh_live.py`.
- **Behaviours:** (00) **P1-20 passes records only through `read_bundle`/`classify_bundle`**, and `unattested_units` from the assembly report (spec6 §2) appears in the refresh report (no consumer yet); (0) **`Links.bind_key_mismatch`** lines (value-free `bind-key-mismatch <unitId>`) are mapped by `refresh` to that unit's `bundle:bind-key-check` `failReason` and a line in the refresh report (the code is in `FAIL_REASONS`; `classify_bundle` never produces it, only the linker does); (1) per unit: `build_unit` (P1-44) inside its own try; a failure keeps the unit's last good
  bundle marked `stale-kept {built commit, failed at, reason}`, the reason the one code P1-44 (3) assigned (value-free:
  never exception text; P1-20 sets none itself); a never-built unit is `not-mapped`, and every link
  into it is `possible`/`target-not-mapped` (AC-9b); (2a) **`scope`** is a collection of `repoId` and selects which units are **rebuilt**; it never selects which units the graph holds: the assembled graph always contains **every manifest unit**, the out-of-scope ones taken from their last good bundle unchanged with their own `builtCommit` and **with their previous status and failure marker carried** (a unit that was `stale-kept` stays `stale-kept` with its `failedAt` and `failReason` until it is rebuilt, since the marker says its last refresh failed, which a commit comparison cannot show; the carried state is persisted per system in the work directory through `ccg.atrest`, because the build credential cannot read the live key) or `not-mapped`/`manifest:never-built` when none exists, so a scoped refresh can never replace the system graph with a partial one (empty or `str` scope is `ValueError`, an unknown id `RefreshError`; the broker run lists the scoped repos). **A first build that fails with no prior bundle** is `not-mapped` carrying that build's code (`build:shallow-clone`, …); `manifest:never-built` is for a unit no build was attempted for. **A `bind-key-mismatch` unit** is `stale-kept` with `bundle:bind-key-check`, its bundle still loaded, `failedAt` the refresh time and `builtCommit` the bundle's own (it must be rebuilt under the current key). **The default pack set** is the seven packs `fastapi, fastmcp, fetch, infra-signatures, kafka, severity-rules, shell-mcp-call` (`paper-inhouse-bus` raises `UnsupportedSubject` and is excluded; a drift test compares the set with `cpg/packs/` minus that one). **A stale-kept unit's old bundle loads with its own header**, so its `Unit` carries the old `builtCommit`, `failedAt` and `failReason`, `manifestHash` covers `buildStatus` and `failReason` (the snapshot differs from a fresh one), and every answer item through it is marked by P1-39's staleness reader. (2) units are built **one at a time** (6 GB host); (3) after
  all units: assemble (P1-18) → load and swap (P1-19) → a report (units fresh / stale-kept / not-mapped, wall time
  and peak memory per stage); (4) the staleness check compares a unit's built commit with the **local** head by
  commits or by files changed under the unit root; **there is no remote fetch in P1** (index §6): the manifest
  loader rejects a `remote` key, and the report says "head is the local clone's"; (5) a full rebuild is the default
  (bundle caching by tree id is not built; the lever if the budget fails); (6) `refresh` is the one entry point;
  only the on-demand trigger exists in P1 (schedule, threshold and CI are P3, AC-9a); (7) a shallow clone is a build
  error (D1); (8) units are discovered from the **commit** being built, as P1-04 stages that commit's own tree (`git ls-tree` plus `git cat-file`): `discover_units`
  gains an optional `commit` argument that lists `git ls-tree -r -z --name-only <commit>` instead of the index (`git ls-files`, the
  default for the golden), so a staged-uncommitted or work-tree-deleted path never makes a unit.
- **Done:** a full refresh of the proving system on the dev host with **wall time and peak RSS recorded** (R-4); an
  unbuildable-unit fixture and a never-built unit behave as in (1). **Gate:** `analyst`.
