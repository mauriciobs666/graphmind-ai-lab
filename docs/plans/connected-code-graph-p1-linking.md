# Connected code graph — P1 plan, part 2: linking, assembly, store, refresh (W2)

> **Status:** active · **Owner:** `architect` · **Tracks:** C-1001 (P1)

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
  `fd00:ec2::254` is never bindable, whatever writer produced the bundle (P1-11 already withholds it, in `writer_rules.bind_fields`). (e) `port` facts carry no published/container distinction in P1 (`-p1-extraction.md` P1-11): a loopback consumer's match through a Compose `ports: "A:B"` pair is **counted** in `counts["loopback_port_residual"]` and stays as matched (a count, not a weakening) until `portKind` exists. Tests: `../../../../etc`
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
  `Dockerfile` `CMD`/`ENTRYPOINT` entry beside the script, with the `LAUNCHES` edge `command → script File → Unit` carrying the script hop; the
  graph note's §2 row naming the script's launch entry is amended at close). **Hop 1 input** is P1-11b's (`-p1-extraction.md`): the real repo-root
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
  beside its links for P1-20's refresh report; (14) **regex keys are repo-derived text under Python `re`, which is not linear time:** a `keyForm: regex` key (the `kafka` pack's `pattern`) and a `link` matcher `regex` are matched only when `matchers.py`'s guard admits the pattern: `ccg.rulepack.regex_problem` clean, no quantified group that holds a quantifier or an alternation (`(a+)+`, `(a|b)*`), at most 4 quantifiers in total (fuzz evidence `cpg/.ccg-work/p1-17b/`), and the literal it is matched against at most 256 characters. A refused pattern matches nothing and is **counted** in `Links.counts["refused_patterns"]`, never silently dropped. No timeout (an in-thread one is unreliable). Tests: `(a+)+$` against `aaaa…!` returns at once and is counted; a plain pattern still matches. `ccg.rulepack` is unchanged (pack regexes are pack-authored). The `matchers.py` docstring's "RE2 subset" claim is reworded to "the RE2 subset plus this guard" (it is true only with the guard). A linear-time engine (an NFA over the RE2 subset, or an RE2 binding, a dependency decision) is a P2 item, owner `architect`.
- **Test owed (mutant F5):** `{**}` is accepted only in the last provider segment: `GET /a/{**}/b` against `GET /a/x/b` does not match. **Registry test (M3):** every reason a `Link` or `Via` emits is asserted to be a registry code whose `on` list holds `Link` (a `Via` reason: `Link` or `VIA`), not only its status. The `packs=` error says "no rule of the given packs consumes lifting X" when packs were passed.
- **Binding decisions:** (a) a prompt mention `mcp__<server>__<tool>` binds only through that server's own `.mcp.json` entry (URL or launch chain), never through a client URL elsewhere in its file; unbound it keeps the unbound rules. (b) A script launch reaches its container through the `Dockerfile` beside the script, and the unit through `targetArg`, else the unit rooted at that directory (none when several units share the root). (c) A primary chain beats `example` chains; a built unit beats an unmapped one. (d) An `ip`-literal host (private ranges included) binds nothing. (e) A Compose service declared only in `example` files never claims a loopback port. (f) The consumer's environment-variable **name** is read only where a bundle record holds it (a Python `os.environ.get`/`getenv` flowing into the URL slot); a JavaScript `import.meta.env.X` or a Python module constant `X = os.environ.get(...)` leaves no record (`normalize/core.py` drops a call-valued constant), so LNK-13 (and LNK-15's default) stay unbound at P1: **open question Q-ENV** for the stakeholder, in the reply that carries this plan edit (options: a new bundle value element for environment reads, bundle minor 1.2; the default literal as a Python-only alternative; or defer to P2 with LNK-13/LNK-15 expecting `key-match` at P1).
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
  own tests; (6) **Markdown hand-offs:** a `MarkdownError` (an oversize file) **skips that file** and counts it in the build report
  (`skippedDocs`); it does not fail the unit, and no `skippedFiles` reason is used (the field is not written for Markdown units in
  v1); P1-44 passes `extract_markdown` (a) `history`, the file's `(lastChangedAt, lastChangedCommit)` from the source repo at the built
  commit through `normalize.blame` (`last_changed` over the whole file, `path_last_commit` for an empty one), (b) `bundle_path` when the
  path in the bundle differs from the repo path the manifest's globs match, and (c) `known_tools`, the system's MCP tool names,
  computed once per build by running the P1-16 mcp pack (`ccg.packrt.run_packs`) over the system's code bundles built before the
  Markdown units (code units first); tests: the stub extractor records all three arguments, an oversize file is skipped while its unit
  builds, and the `fx-svc` fixture's two prompt files yield the AC6-PM call sites end to end; (7) **Config call convention:** `extract_config` is called **once per repository** with all that repository's Compose files (W2's declared-service set is per repository and primary-role only), or with `extra_declared_services` for a split call; for a file that produced no records (oversize, unparseable) P1-44 takes the `File` role from `file_role`, since `ConfigResult.roles` is empty for it; (8) **Header merge (`bindParts` obligation, `-p1-extraction.md` P1-14):** before `write_bundle`, P1-44 takes `fields = writer_rules.bind_header_fields(ccg.atrest.hmac_key(system), merged_records)` over the unit's **merged** records (never a union of per-extractor `ConfigResult.header_fields`) and sets `header.bundleSchemaVersion` (`"1.1"` iff any entry has `bindParts`, else `"1.0"`) and `header.bindKeyCheck` (present iff `"1.1"`) from it, so V1a-V1d hold; the key is read through `ccg.atrest` only (behaviour (4)). Tests in `test_build_dispatch.py`: a config unit with a URL entry builds and validates as `"1.1"` with `bindKeyCheck == ccg.atrest.bind_key_check(key)`; a code-only unit and a config unit with no URL entry build as `"1.0"` with no `bindKeyCheck`; a multi-file config unit whose `bindParts` entries come from two files gets one header.
- **Done / Gate:** `analyst`. **Needs:** 03, 04, 05, 10, 11, 12, 16 (the pack runtime that names the tools).

## P1-20 · refresh — `coder`

- **Files:** `cpg/ccg/atrest.py` (`sweep_stale_tmp(system, older_than_s=3600)`: removes `.<name>.<16 hex>.tmp` files left in the system's bundle directory by a crashed `write_bundle`; touches nothing else) with its test in `cpg/tests/test_atrest.py`, `cpg/ccg/refresh.py` (`refresh(system, scope=None) -> RefreshReport`; it calls the sweep first, and the report counts what was removed), `cpg/ccg/staleness.py`
  (adds `unit_staleness(unit, head)` to the file P1-18 created with `FAIL_REASONS`; the **one** staleness module,
  which `engine/` imports in P1-39); `refresh` also takes
  `override=None`, and only when one is given opens a broker `refresh` run, fails if the broker is absent, and names
  the `policy` record in its report (`-egress-spec3.md` §2.1, §2.5); `cpg/tests/test_refresh.py`,
  `cpg/tests/test_refresh_live.py`.
- **Behaviours:** (1) per unit: `build_unit` (P1-44) inside its own try; a failure keeps the unit's last good
  bundle marked `stale-kept {built commit, failed at, reason}`, the reason the one code P1-44 (3) assigned (value-free:
  never exception text; P1-20 sets none itself); a never-built unit is `not-mapped`, and every link
  into it is `possible`/`target-not-mapped` (AC-9b); (2) units are built **one at a time** (6 GB host); (3) after
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
