# Connected code graph — rule-pack spec (format v1, five paper packs, infrastructure signatures, severity rules)

> **Status:** active · **Owner:** `architect` · **Tracks:** C-1001 (M10) · **Extends:** `docs/plans/connected-code-graph.md` · **Version:** 1.5 · **Reviews:** `docs/reviews/connected-code-graph-rule-pack-spec.md`, `docs/reviews/connected-code-graph-rule-pack-spec-ml.md`

*Revision 2026-11-04 (v1.5): a slot has one `onUnknown` mode, mixed nested modes rejected (§1.4, N-29); ruling on gate Pass 4 P4-4.*

This is step S30 (unit P0-22) of `docs/plans/connected-code-graph-p0-2.md` (§4.2, "the rule-pack
spec"). It fixes the pack file format S31 turns into `cpg/ccg/schemas/rule-pack-v1.schema.json`, and
the content S31 and S32 turn into the seven `cpg/packs/*/pack.yaml` files. Its claims are checked by
`cpg/tests/test_rule_pack_spec.py`, which extracts this document's YAML blocks and tables, runs every
matcher against the `s14a` records, and runs the signature samples and the severity vectors.

CPG: considered, not relevant — spec work; no loaded graph indexes `cpg/`; every fact below was read from the slice report, the `s14a` bundles and `git show defca60f`.

## 0. Scope, inputs, and what is real versus paper

**Citation convention.** A bare `§n` is a section of **this document**; `spec §n` is a section of
the bundle spec (`-spec3` as amended by `-spec4`); every other document is named.

**Contracts read.** The bundle contract is `docs/plans/connected-code-graph-bundle-spec3.md` plus
`docs/plans/connected-code-graph-bundle-spec4.md` (frozen at `88eb134d`). The graph model is `docs/plans/connected-code-graph-p0-graph.md` (D1–D13). The
design source is proposal §3.3–§3.4 (`docs/plans/connected-code-graph.md`). The contact rule is
`docs/plans/connected-code-graph-eval-protocol-ml.md` §8.2, §14 (S14/S30 row) and §15 (Q10, R9). The
slice evidence is `docs/test-reports/connected-code-graph-p0-slice-report.md` and its bundles
(`cpg/.ccg-work/vertical-slice/s14a/*.bundle.jsonl`, `defca60f`, run dir gitignored).

**In scope:** pack format v1 (§1); five paper packs, each walked rule by rule (§2–§6); the
language-expressiveness table (§7); the infrastructure-signature pack v0 (§8); the severity rule
table (§9); the negative examples S31 tests (§10); the contact list (§11). **Out of scope:** the
pack runtime and linker (P1), the schema file and pack YAMLs (S31/S32), RCA weights (ML §8.3 owns
the ranker), dependency severity (CVSS bands, proposal §3.7).

**Real versus paper.** A walkthrough is only as strong as its inputs, so each pack says which:

| Pack | Provider side | Consumer side | Launch/config side |
|---|---|---|---|
| FastAPI (§2) | real S14 records (`falkor-chat/server`) | — | — |
| fetch (§3) | — | real S14 records (`salesperson`) | — |
| FastMCP (§4) | real S14 records (`falkorchat/mcp.py`, `app.py`) | invented snippet | **paper**: records written from source lines by spec §5.17/§1.6 and passed through the argv gate (`ccg.bundle.gate_argv`); **no extractor emits them yet** (S14 emitted 0 `configEntry`, 0 `launchFact`) |
| Kafka (§5) | invented snippet | invented snippet | — |
| in-house bus (§6) | invented | invented | — |

Where a paper record has a field no source line fixes (an HMAC, an id), it is written `<hmac>` or
omitted; no walkthrough depends on one.

**S27 input.** S27 (frontend fitness, `graph-dba` and `data-scientist`) has not produced its
decorator, annotation and attribute shapes. Every place that needs them is marked **S27 input** and
filled only with what the S14 bundles show (§7). No per-language shape beyond that is stated.

**Decisions made here, with the reason in the section named:** the escape-hatch query form (§1.7);
packs carry no ranking weights (§8); the severity reasons are one parameterized registry code,
and the reason registry is a machine-readable file S31 owns (§1.9, §9.1); env-default port adoption
has an exclusion list (§4.4); the severity formula is the single source and the rows derive from it (§9).

## 1. Format v1 (normative)

A pack is one YAML file, `cpg/packs/<id>/pack.yaml` (AC-22: a pack is one directory; adding one
touches nothing else). Packs are **declarative data**: no code, no I/O. The core owns `keyNorm`
(spec §6), the linker, the reason registry (spec §7), ids, and the argv gate; a pack supplies recognizers and
facts the core cannot know.

### 1.1 Header

| Key | Type | Req | Rule |
|---|---|---|---|
| `packFormat` | int | yes | `1` |
| `id` | string | yes | `[a-z][a-z0-9-]{0,39}`; equals the directory name |
| `version` | string | yes | semver; any change to a body key bumps it |
| `kind` | enum | yes | `framework` \| `protocol` \| `data-store` \| `test-framework` \| `security` \| `infra-signatures` \| `severity` (proposal §3.3's list; `security` holds sources, sinks, sanitizers, auth checks, sensitive marks and risky-config rules) |
| `vocabVersion` | string | yes | the bundle schema version whose spec §6 vocabulary and spec §7 registry the pack is written against: `"1.0"` |
| `status` | enum | no | `paper` (a pack with no extractor behind it yet; p0-2 §2 layout) |

Body keys, all optional: `rules`, `mounts`, `liftings`, `embedded`, `launch`, `urlBindings`, `vocab`,
`link`, `queries` (§1.2–§1.7); `infra-signatures` packs carry `runtimes` and `signatures` (§8) and
`severity` packs carry `formula` and `rows` (§9.1) instead of `rules`. **Unknown keys are rejected at
every object** (N-21, N-26). A pack of declarations only (no `rules`) is valid for any `kind`.

### 1.2 Rules: `match` / `extract` / `emit`

```yaml
rules:
  - id: route                       # unique in the pack; [a-z][a-z0-9-]*
    match: {<subject>: {...}}       # exactly one subject, or anyOf: [<clause>, ...]
    scope: {imports: [...], path: <matcher>, notPath: <matcher>, role: [primary]}   # optional file gate
    extract: {<slot>: <expr>, ...}  # named slots
    emit:    [ {<emission>}, ... ]  # at least one
```

**Subjects** (a clause names exactly one; each reads the bundle record named, with these fields):

| Subject | Record | Matchable fields |
|---|---|---|
| `decorator` | spec §5.12 | `form`, `name`, `args`, `target` (`kind`, `name`, `parentKind`) |
| `callsite` | spec §5.6 | `callee` (= `calleeName`), `callKind`, `resolution` (`resolved`, `unresolved`, `external`), `externalTarget`, `target` (a `targets[].qname`), `args` |
| `config` | spec §5.17 | `format`, `keyPath`, `bindValue` (only when present) |
| `launch` | spec §1.6 | `kind`, `command` (a matcher over the emitted string, usually `basename`), `args`, `port`, `envVar`, `mountPath` |
| `symbol` | spec §5.3 | `kind`, `name`, `qname`, `parentKind` |
| `marker` | spec §5.16 | `framework`, `kindHint`, `evidence` |
| `lifted` | a role-lifting node (§1.6) | `lifting` (an id) |
| `query` | a §1.7 query result row | `query` (an id) |
| `evidence` | free text | `infra-signatures` packs only (§8) |

- **`args`** (decorator or callsite) is a list of `{arg: <argPath>, value?: <matcher>}`; every entry must
  hold. Without `value` it tests that the argument is present; with it, that some alternative of the
  argument is a `const` whose text the matcher accepts.
- **Any-of.** A list in a position that takes one matcher or enum value (`resolution: [external,
  unresolved]`) is OR. `scope.imports` is a list of gates satisfied when **any** entry is. `anyOf:
  [<clause>, ...]` replaces the single subject clause, matches when any clause does, and the clauses
  share one `extract`/`emit`; its YAML form is `match: {anyOf: [{callsite: {...}}, {decorator: {...}}]}`.
- **`scope`** (optional file gate): `imports: [{module: <matcher>, names?: [..]}]` is satisfied when the
  matched record's file has an `import` record (spec §1.5) whose `module` matches (and, if `names` is
  given, imports one of them); `path`/`notPath` are matchers over the file path; `role` is a list of
  `File.role` values (spec §5.2).

**Expressions** (`extract` values): `{arg: <argPath>}` (the value-set alternatives of the matched
record's argument: `a<i>`, `kw.<name>` or `recv`, with spec §5.1 field suffixes; the flow heads `p<i>`, `r`
and `ret` are not arguments); `{literal: <string>}`; `{group: <n>, upper?: true}` (a regex group
of the clause's `name` or `callee` matcher); `{target: name|qname}` (the decorated or matched
symbol); `{config: keyPath|bindValue}`; `{field: <name>}` (a spec §5.13 read of the matched point);
`{yield: <name>}` (a column of a `query` row); `{lifted: url|verb}` (the resolved role value of a
`lifted` node, §1.6); `{firstOf: [<expr>, ...]}` (the first expression whose alternatives are
non-empty, left to right). A slot may carry `onUnknown` (§1.4) and `default` (a literal string).

### 1.3 Matchers

A string-valued field takes **one** matcher; a bare scalar means `exact`.

| Matcher | Meaning |
|---|---|
| `exact: s` | string equality |
| `template: t` | path-template: `{}` is one non-empty segment, `{**}` the one or more remaining; same syntax as spec §6 `http` step 7 |
| `glob: g` | whole-string; `*` any run of characters except `/`, `**` any run (so a value that contains `/` needs `**`), `?` one character; no brace alternation, no classes |
| `basename: b` | the string's last `/`-separated component equals `b` |
| `prefix: p` | string prefix; on a path at a `/` boundary (`/mcp` matches `/mcp` and `/mcp/x`, not `/mcpx`), on a module name at a `.` boundary (`fastapi` matches `fastapi.responses`) |
| `regex: r` | **RE2 subset**: whole-string match (implicit anchors), at most 256 characters, no backreference, lookaround, atomic or possessive group; case-sensitive unless `ignoreCase: true`; the `evidence` subject (§8) searches each line instead |

**Path normalization (testable).** A path-like operand of `exact`, `prefix` or `template` (a mount
path, a consumer URL path, a route) is first passed through `normalize_path(p)` = spec §6 `http` steps
4–6 only: ensure a leading `/`; percent-decode unreserved characters and uppercase the hex of other
escapes; collapse `/` runs and resolve `.`/`..` (RFC 3986 §5.2.4). **Both** operands are normalized,
once, from the original text (spec §6 step 6: a second pass can change a key). Vectors:
`normalize_path("/mcp/%2E%2E/admin")` is `/admin`, so `{prefix: /mcp}` does not match it;
`/mcp/./x` is `/mcp/x`, which it matches; `/a//b` is `/a/b` (F-13).

Why `regex` is restricted: pack evaluation must be linear-time and the same on every host. `template`
applies only to path-like fields (a key, a mount path, a URL path). The **linker's** matcher is a
different choice, per contract kind (`link`, §1.6): `exact`, `path-template`, `glob`, `prefix` or
`regex` (proposal §3.4); a `Link`'s `matcher` records the strongest one applied (`exact` when the keys are equal strings). A key
emitted with a non-literal `keyForm` (`glob`, `regex`, `template`) is matched by that form against the
other side's literal keys of the same kind, whatever `link` says; `link` is the default for literal against literal.

### 1.4 Resolution of a value, and what unknown does

A slot's alternatives are computed as the S12 recognizer computed them (`cpg/spikes/vertical-slice/assemble.py`
`Corpus.alternatives`, the reference for these rules): `const` is itself; `concat`/`template` is the
product of its parts (at most 64 alternatives); `ref` is the referenced `constant`'s alternatives
(depth at most 8, a cycle stops); `importRef` and `unknown` are an unresolved piece. S14's value sets
all have exactly one alternative (slice report §5), so the product and cap paths are unexercised.

`onUnknown`, per slot:
- `drop` (default for `provides` of every kind but `topic`, and for `writes`): an alternative with an unresolved piece is omitted and
  listed in the rule's **diagnostics** (S14 §6.2: bundle spec §6 has no provider-side unresolved
  form; a diagnostic is not a registry code and reaches no graph element).
- `partial` (default for `consumes`, `reads`): the unresolved segment becomes `{?}` (spec §6 step 8),
  the fact is `possible`, reason `partial-key`. Only `http` has a placeholder in spec §6; for any other
  kind `partial` means `runtime-built`.
- `runtime-built`: no key; the fact is `possible`, reason `runtime-built` (only where the registry (§1.9)
  lists the code for the emitted edge: spec3 §7.2's `CONSUMES`, `READS`/`WRITES`, plus `PROVIDES topic`).
- `config-derived` (default for `provides` of kind `topic`, and valid on a `consumes` or a `provides` of
  kind `topic` only: spec3 §7.2 lists the code on `CONSUMES topic`, and §1.9's `PROVIDES topic` cell
  (O-16) adds the publisher side; N-27): as `runtime-built`, with reason `config-derived`, when the unresolved argument's
  point has a spec §5.13 `fieldAccess` read; otherwise `runtime-built`. The name itself is never read:
  a pack cannot extract a configuration value (N-20), and `bindValue`'s grammar admits only ports, hosts and URLs.

**A slot has one mode.** Every `onUnknown` nested in one slot (through `firstOf`) must name the same mode; an expression with none takes it (N-29). Status and reason come from the slot's
resolution (N-17); with `[drop, partial]` an unresolved alternative would be omitted or `possible` by which `firstOf` branch produced it, so the
validator could not say whether `confirmed` is sound, and "the first one rules" would make the second annotation dead text.

A rule may emit `confirmed` only when every slot its `key` uses is resolved in that alternative
(N-17). When an argument's value set is `unknown point-derived`, the value arrives through `data`
records; that case is a **role lifting** (§1.6), never a hand-written flow walk in a rule.

### 1.5 Emissions

| Emission | Fields | Becomes |
|---|---|---|
| `provides` / `consumes` | `kind` (core or pack `vocab`), `key` (a template over slots, `{slot}`; `{mount}` is the mount prefix of §1.6), `keyForm?` (`literal` default, `glob`, `regex`, `template`), `status?`, `reason?` | `PROVIDES`/`CONSUMES` on a `ContractPoint` (graph D4) |
| `reads` / `writes` | `kind`, `key`, as above | `READS`/`WRITES` |
| `test` | `kind: unit|integration|e2e` | `Test` label, `testKind` |
| `mark` | `as: source|sink|sanitizer|auth-check|sensitive`, `class` (a pack token) | a `Mark` |
| `finding` | `finding: injection|authorization|configuration|secrets`, `class` (a §9 feature value, e.g. `host-compromise`), `target: config|symbol` | a `Finding` whose rank §9 computes from the class and the chain |
| `launches` | `target` | `LAUNCHES` |

`status`/`reason` default from the key's resolution (§1.4) and may only be a code of the reason
registry (§1.9; a pack may not invent one, N-9) whose status agrees with `status` (N-17) and whose
`On` set contains the emission's edge (and kind, where the row names one). The **anchor** of
a decorator emission is `targetId`; of a callsite emission, the enclosing `symbolId` with the call
site as the site. A recognizer that matches by `callee` name plus an import gate is enough for
`confirmed` once the key is resolved: the S14 FastAPI providers are `confirmed/key-match` though the
`include_router` call sites are `possible/name-match` (slice report §6). `externalTarget`, where the
frontend resolved it, only narrows.

### 1.6 Declarations

**`mounts`** (composition of a provider key from a prefix; proposal §3.3 "B3"):

```yaml
mounts:
  - id: include-router
    call: {callee: include_router}
    scope: {imports: [{module: {prefix: fastapi}}]}
    child: {arg: a0, resolves: factory-call}
    prefix: {arg: kw.prefix, default: ""}
    roots: [{call: {callee: FastAPI}, how: app, notWith: [{callee: APIRouter}]}]
    constructorPrefix: [{call: {callee: APIRouter}, arg: kw.prefix}]
    unmounted: {status: possible, reason: partial-key}
    recordConditions: true
```

Semantics (the S12 recognizer's, `assemble.py` `_include_sites`, `_mount_chains`, `providers`):
- `child.resolves: factory-call` means: the mount call's `child.arg` is fed by a `data` record
  (`toPath` = that arg, `fromPath` `r`) whose `def` is a call site; that call's `targets` are the
  **factories**; a provider whose decorated handler's `parentId` is a factory is mounted by this call.
- Mounts compose outward: a mount call inside a factory that is itself mounted yields a longer chain,
  outermost prefix first, to any depth (a cycle stops). One variant per chain.
- `prefix`: each chain's prefix alternatives are multiplied with the route's; any unresolved piece
  drops that alternative (§1.4). The key is `verb`, the joined prefixes, then the route, normalized
  by the core.
- `roots`: a handler whose parent contains the named call (`FastAPI(...)`), contains none of the
  `notWith` calls (`APIRouter(...)`), and has no chain is `how: app` (no prefix). A handler with no
  chain that is not a root is `unmounted-router`; its provider is emitted with the `unmounted` status.
  Otherwise the provider is `confirmed/key-match`. Why `notWith`: a decorator record carries the
  receiver text (`inner.get`) but no assigned-variable name (§1.8), so a parent holding both `FastAPI()`
  and a local `APIRouter()` cannot say which one the handler is registered on (§2.3).
- `constructorPrefix`: a provider whose handler's parent (the factory, or the root's parent) contains
  a call matching `call` **with the argument present** gets the `unmounted` status, whatever its chain:
  the router's own prefix is a second prefix source this form does not compose (§2.3). The same
  provider is emitted without that prefix, so the key is a lower bound; `partial-key` says so.
- `recordConditions`: every `controlledBy` condition of each mount call (`{line, code}`) is recorded
  on the mount fact. Graph D5 (`MOUNTED_BY`) is where it lands.
- A second form, `kind: class-decorator` (`decorator: <clause>`, `prefix: {arg}`), composes a prefix
  from a decorator on the handler's parent type (the Spring/ASP.NET shape of proposal §3.3). It is
  **expressible, not walked, and its name shapes are S27 input** (§7).

**`liftings`** (role lifting, proposal §3.3 "B2"):

```yaml
liftings:
  - id: fetch-url
    role: {kind: http, slot: url}
    seeds: [{call: {callee: fetch, resolution: [external, unresolved]}, arg: a0}]
    companions: [{role: verb, arg: a1.method, bareDefault: GET, unknown: "*"}]
```
`role.kind` is a contract kind and `role.slot` names the slot (`url`); `companions[].role` is `verb`;
`bareDefault` and `unknown` are literal verbs.

The core runs the fixpoint (`assemble.py` `lift_role`): a seed slot's value that `data` takes from a
parameter makes that parameter a lifted slot; a parameter makes each caller's matching argument a
lifted slot; a value that `data` takes from a call result continues through the callee's `ret` to
the parameter it came from. Each step is kept as an ordered hop for `LIFTED_VIA` (graph D1:
`{ord, siteId, line, urlPath, verbPath?, viaSiteId?, viaLine?}`). `companions` run the same
fixpoint for a second path; the verb is read from the first slot on the chain, consumer side first,
whose companion path resolves; `bareDefault` applies when the seed call has no second argument.
A rule with `match: {lifted: {lifting: fetch-url}}` emits for each lifted slot that resolves and has
no lifted child (an intermediate hop is not a consumer).

**`embedded`**: `[{call: <clause>, arg: <argPath>, language: sql|cypher, scope?}]`. The core parses
the argument into `READS`/`WRITES` on `data_entity` points; an unparseable or runtime-built text
is `possible/runtime-built` (proposal §3.3). Declared in the format; not exercised by the five packs.

**`launch`** (facts the bundle does not carry; spec §1.6 says framework defaults come from packs):

```yaml
launch:
  frameworkDefaults: [{command: {basename: uvicorn}, port: 8000, protocol: http}]
  portFlags:         [{command: {basename: uvicorn}, flags: ["--port"]}]
  targetArg:         [{index: 0, form: module-attr}]
  envPorts:          [{envVar: {regex: '(?:[A-Z][A-Z0-9]*_)*PORT'}, excludeVars: [{regex: 'DB_PORT'}], adoptedBy: same-file-launch}]
  mountPaths:        [{call: {callee: mount}, path: {arg: a0}, app: {arg: a1, fromCall: {externalTarget: {glob: '**FastMCP.streamable_http_app'}}}, pathMatch: prefix}]
```

Shapes: `command` is any §1.3 matcher over the emitted `command` string; `port` is an int,
`protocol` a string; `flags` a list of strings; `form` is `module-attr` (the only member in v1);
`adoptedBy` is `same-file-launch` (the only member); `excludeVars` is an optional list of matchers (absent is none excluded); `pathMatch`
is `exact` or `prefix`; `fromCall` is a `callsite` clause.

- `frameworkDefaults`: a `command` launch whose emitted `command` matches and whose emitted `args` hold
  no `portFlags` flag yields `(port, unit)` with origin `pack-default`. It reads **emitted** (gated)
  text, so it is sound only where `command` and `args` are visible (§4.3).
- `portFlags`: `--port 8000` and `--port=8000` read from emitted `args`.
- `targetArg`: on any `command` launch (or those whose `command` matches, when the entry gives one),
  positional argument `index` of form `module:attr` resolves to the in-repo file whose
  path ends in `module-with-slashes.py`; that file's unit is the launched unit. Zero or several
  matches resolve nothing. Without a `command` constraint any `command` launch qualifies; a false unit
  needs an in-repo `x.py` for an `x:y` argument.
- `envPorts`: an `env-default` launch fact whose `envVar` matches and is not in `excludeVars`
  contributes `(port, unit)` for the unit launched by a `command` launch **in the same file**. `excludeVars` is
  optional: an exclusion list is pack content, not format, and a pack whose adopter has nothing to exclude has
  nothing to list. A pack that adopts `*PORT` with no exclusion adopts `FALKORDB_PORT` (§4.4 step 5), so the
  shipped `fastmcp` pack keeps its list, which `test_env_port_heuristic_claims` pins.
- `mountPaths`: a mount call whose `app` argument is fed by a call matching `fromCall` declares a
  unit mount path (`path` alternatives), conditions recorded as for `mounts`. `pathMatch` is how a
  consumer URL's path is compared: `exact` or `prefix` (§1.3).

**`urlBindings`**: `[{id, kinds: [<contract kind>], urlSources: [<source>, ...]}]`. A URL source is
`{config: <clause>}` (a `config` clause whose `bindValue` is a URL, spec §1.7.2 `url`) or
`{callsite: <clause>, arg: <argPath>, scope?: file|unit}`. `scope` belongs to the `callsite` form only
and says which consumers of `kinds` that URL binds: `file` (default) those in the call's file, `unit`
those in its unit. The core then does
proposal §3.4: split host/port/path; match the port against launch facts (explicit flags,
env-default, `pack-default`, Compose mappings); match the path against mount paths; on `localhost`
several deployments of one unit are one provider; two different units on one port are
`possible/port-shared`. An example-role file makes every link through it `possible/example-config`.

**`vocab`** (a kind or transport the core does not have): `[{kind, keyNorm, example, transport?}]`.
A new `kind` states its `keyNorm` rule in prose and an example, in spec §6's table form (spec §6's last line;
N-11). A `topic` transport is a lowercase token, or `inhouse-<packId>` with the pack's own id (N-19).

**`link`**: `[{kind, matcher}]`, the linker's matcher for that kind (§1.3). A kind absent here links `exact`.

### 1.7 The escape hatch: bundle query v1

Where no `match` clause can say it (a fact spread over several records), a pack declares a
**bundle query** and a rule consumes its rows with `match: {query: <id>}`. Decided form, a
relational select over the bundle's own records; `queries: [{id, from: [<binding>, ...], yield:
{<column>: "$alias.<field>"}}]` with 1 to 4 bindings `{as, record, where}`, in order. The worked
example is `handler-ns` in §6 (the one copy).

The three constraints, and how this form meets each:
- **Read-only:** a query names bundle record types (spec §5, not `secretSighting`) and field tests; it has
  no write verb, no function call, no I/O; there is nothing to forbid.
- **Over one unit's bundle:** `from` ranges over the records of the unit being evaluated; a join is an
  equality between an earlier alias's field and a later `where` field, on id-typed fields only
  (`id`, `symbolId`, `parentId`, `callSiteId`, `unitId`, `targetId`, `fileId`, `point`, `cond`, `use`,
  `def`), so every join is an index lookup. **Every binding after the first carries at least one join
  equality** (no cross products), and the row cap applies to the intermediate result after each binding,
  not only to the output (N-14).
- **The anchor column.** A rule on a `query` subject anchors each emission at the row's `handler`
  column, which the query must yield (a symbol id); `scope` applies to that symbol's file.
- **Deterministic:** no recursion, no aggregation, no negation. `$alias.args.<argPath>` is the
  alternatives of that argument (one output row per alternative). Rows are de-duplicated and sorted
  by their canonical JSON (spec §3.1); more than 20,000 rows at any stage is a rule error (a diagnostic, no facts).

Rejected: **Cypher over a loaded bundle graph** (needs a live FalkorDB at evaluation, though the
bundles are the assembler's only input; write-prevention means parsing the query; row order is
deterministic only with `ORDER BY`); **jq/JSONPath** (joins need slurping the whole file, record
types are unchecked, cost is unbounded); **a code hook** (not declarative, not statically checkable,
and it would break AC-22's "one directory" test). The cost accepted: P1 implements a small
evaluator, about the relational select subset above.

### 1.8 Determinism, composition, limits

- A pack set's output is a pure function of the unit bundles and the packs (ids, versions, bytes):
  no clock, randomness, network or file read beyond the bundles. Facts are emitted in spec §8.3's record
  order, then by canonical JSON.
- Several packs may emit the same fact; duplicates collapse on `(anchor, kind, key, site)`, and a
  `confirmed` copy wins (as the S12 `best` rule). A `launch` or `vocab` entry repeated across packs
  is the union.
- **No pack can:** change `keyNorm`, add a reason code, read a configuration value that did not pass
  the `bindValue` gate (N-20), emit a secret value, or name a repository (N-21): packs are generic,
  which is why the walkthroughs, not the packs, carry anchors (§11).
- **What the frozen bundle cannot give a rule** (found while walking the packs; each is a P1 or
  `-spec5` question, §12): a call site has no assigned-variable name or receiver text; a decorator
  has no resolution; `fieldAccess` carries no value; a shell `exec` line's variable-bearing argv reaches
  the bundle redacted (§4.3); there is no `isTest` flag on a record.

### 1.9 The reason registry file (S31 owns it)

N-9 and N-17 need a machine-readable registry, and spec §7 is prose. **S31 creates
`cpg/ccg/registry/reasons.yaml`** (row shape `{code, status, on: [<token>...]}`; a token is an edge or record name (`CONSUMES`) or `<EDGE> <kind>` (`PROVIDES topic`), because one code can differ by kind across edges and a row-level `kinds?` cannot say so,
`status` ∈ `confirmed`, `possible`, `none`), seeded by a test that parses spec §7.2 and §7.3 (as S10
parses spec3's vectors), plus the rows spec §7 leaves to this document, with the `On` sets the packs
need and spec §7.2's column does not list (spec §7.4: "S30's severity rule table adds the others"):

| Code | Status | On (added to spec §7.2's) | Used by |
|---|---|---|---|
| `key-match` | confirmed | `PROVIDES` | every provider emission (§2, §4.1, §5, §6) |
| `partial-key` | possible | `PROVIDES`, `CONSUMES` | `unmounted` (§1.6); a consumer with `{?}` |
| `example-config` | possible | `Finding` | a finding in an `example`-role file (§9.1), beside its `severity-rule` |
| `severity-rule:<id>` | none | `Finding` | the 16 ids of §9.1; the only parameterized form added |

`runtime-built` and `config-derived` keep spec §7.2's `On` (`CONSUMES`, `READS`/`WRITES`; `CONSUMES
topic`) plus `PROVIDES topic` (O-16, decided). `advisory-unscored` stays as spec §7.4 has it. A code absent here is N-9.

## 2. Paper pack 1: FastAPI with mount composition (real bundle)

```yaml
packFormat: 1
id: fastapi
version: 0.1.0
kind: framework
vocabVersion: "1.0"
link: [{kind: http, matcher: path-template}]
mounts:
  - {id: include-router, call: {callee: include_router}, scope: {imports: [{module: {prefix: fastapi}}]},
     child: {arg: a0, resolves: factory-call}, prefix: {arg: kw.prefix, default: ""},
     roots: [{call: {callee: FastAPI}, how: app, notWith: [{callee: APIRouter}]}],
     constructorPrefix: [{call: {callee: APIRouter}, arg: kw.prefix}],
     unmounted: {status: possible, reason: partial-key}, recordConditions: true}
rules:
  - id: route
    match: {decorator: {form: decorator, name: {regex: '[A-Za-z_][A-Za-z0-9_]*\.(get|post|put|patch|delete|head|options|trace)'}}}
    scope: {imports: [{module: {prefix: fastapi}}]}
    extract: {verb: {group: 1, upper: true}, route: {firstOf: [{arg: a0}, {arg: kw.path}]}}
    emit: [{provides: {kind: http, key: "{verb} {mount}{route}"}}]
  - id: route-any-method
    match: {decorator: {form: decorator, name: {regex: '[A-Za-z_][A-Za-z0-9_]*\.(api_route|route)'}}}
    scope: {imports: [{module: {prefix: fastapi}}]}
    extract: {verb: {literal: "*"}, route: {firstOf: [{arg: a0}, {arg: kw.path}]}}
    emit: [{provides: {kind: http, key: "{verb} {mount}{route}"}}]
```

The recognizer is the slice's `assemble.py` `providers` (bare `<recv>.<verb>("<path>")` decorators;
`api_route`/`route` give `*`), with one addition the slice did not have: the `scope.imports` gate.
**The gate is output-neutral on S14's bundle:** every decorator that produces one of the 61 providers
and every `include_router` call site (7) sits in a file that imports `fastapi` (checked against
`s14a`'s `import` records). It costs the corpus nothing and keeps `dict.get` decorators in non-web
code from matching on another corpus.

### 2.1 Walkthrough: `GET /shop/api/health` (SL-1's provider)

Records are from `s14a`'s `falkor-chat/server` bundle; ids shortened to eight hex.

| Step | Rule | Input record | Result |
|---|---|---|---|
| 1 | `route`, `match` | `decorator a037d780`: `name` `router.get`, `form` `decorator`, `line` 954, `targetId a6c26487`, `args [{a0: [{const "/health"}]}]` | group 1 = `get`, `verb` = `GET` |
| 2 | `scope` | the file of `a6c26487` (`falkor-chat/server/falkorchat/storefront_api.py`) imports `fastapi` | gate passes |
| 3 | `extract.route` | `args.a0` | alternatives `["/health"]` |
| 4 | `mounts.include-router`, `child` | handler `a6c26487` (`build_storefront_router.health`) has `parentId 54c71547` (`build_storefront_router`). Call site `9bd97f9d` (`include_router`, `app.py:521`, in `create_app 72985088`) has `data 12bdba42`: `toPath a0`, `fromPath r`, `def 6b8beae9` = the call `build_storefront_router(shop)` (`app.py:522`), whose `targets` hold `54c71547` | factory `54c71547` is mounted by `9bd97f9d` |
| 5 | `mounts`, outward | `create_app` is the `targets` of no mount call's `a0` feed | chain = `[9bd97f9d]` |
| 6 | `mounts.prefix` | `valueSet 33afc99e` (`kw.prefix` of `9bd97f9d`) = `[{ref ".../storefront_api.py::API_PREFIX"}]`; `constant f5aef621` (`storefront_api.py:135`) = `[{const "/shop/api"}]` | prefix `/shop/api` |
| 7 | `mounts.recordConditions` | `controlledBy 2e9c605b`: point `9bd97f9d`, cond `77da111b` (`app.py:516`, code `shop is not None`) | condition `{line 516, code "shop is not None"}` |
| 8 | `emit.provides` | key template `"{verb} {mount}{route}"` = `GET /shop/api/health` | core `keyNorm` (spec §6 steps 1–8a) leaves it unchanged |

Emitted: `PROVIDES http GET /shop/api/health` from `storefront_api.py::build_storefront_router.health`,
`confirmed`/`key-match`, one mount `{how: mounted, sites: [{app.py:521}], conditions: [{516, "shop is not
None"}]}`. **This equals** `assemble.py`'s provider record for the same handler (its `key`, `status`,
`reason`, and its single mount with that site and condition), read from a fresh `assemble.py` run on
the `s14a` bundles.

### 2.2 Walkthrough: the two `GET /health` providers (SL-2)

Same two rules, no prefix (`kw.prefix` absent → `default ""`).
- **`api.py::build_router.health`** (`decorator 31dadb63`, `router.get`, line 58). Its factory
  `build_router` is mounted by two `include_router` calls: `app.py:485` (`34f6db37`, with cond
  `dev_surface`, `app.py:484`) and `tests/test_input_bound_parity.py:538` (no condition). Two chains, one
  key `GET /health`, **one provider with two mounts**, `confirmed/key-match`. Equal to the slice's record.
- **`app.py::create_app.health`** (`decorator 3d462336`, `app.get`, line 497). Its parent
  `create_app` is mounted by nothing, but contains the call `FastAPI(...)` (`app.py:479`), so
  `roots` applies (no `APIRouter` call in `create_app`): `how: app`, no sites, no conditions,
`confirmed/key-match`. Equal to the slice's record.

The two providers of `api.py` and `app.py` share one `ContractPoint` (`GET /health`), so the three
providers give two points (slice report G6). Nothing links to `GET /health`: no consumer.

### 2.3 What this walkthrough does not show

- **Equality over all 61 providers** is not walked: it is a conformance test of the pack runtime
  (P1's first pack test: run the `fastapi` pack over `s14a` and compare to `assemble.py`'s
  `providers`), and listing 61 routes here would put each in the contact set (§11).
- **Routers held in a variable** (`router = APIRouter()` then `app.include_router(router)`) are not a
  `factory-call` mount, and the frozen records give no variable-to-decorator edge (§1.8). S14 has three
  such handlers, all in test functions that call `APIRouter()` (checked on `s14a`):
  `test_app.py` `inner.get` `/join` (FastAPI 0.139.0 serves `/shop/api/join`; the slice emits `GET /join`
  `confirmed/key-match`, wrong), `test_input_bound_parity.py` `_probe_rest_app` `router.post` (served at
  `/probe/{pid}`; the slice's `confirmed` is right by luck), and `test_storefront_api.py`
  `extra.get` `/twelfth` (already `possible/partial-key`). With `roots.notWith` the first two become
  `possible/partial-key`; the third is unchanged. **Expected differences from `assemble.py`'s
  `providers`: exactly those two handlers' status** (P1 conformance, §12.1). Module-level routers stay
  unsupported (O-3).
- **A router with its own prefix** (`APIRouter(prefix="/v1")` in a factory mounted by
  `include_router(build(), prefix="/api")`): FastAPI 0.139.0 serves `/api/v1/x`; without
  `constructorPrefix` the pack would emit `GET /api/x` `confirmed`. S14 has no `APIRouter` call with a
  `prefix` argument, so the output on S14 is unchanged; P1 conformance adds both cases as fixtures
  whose provider must come out `possible` (§12.1).
- `drop` is exercised by one real alternative in the corpus (a handler in a test file whose include
  prefix is unresolved; slice report §3, §6.2): the rule omits it and lists it as a diagnostic.

## 3. Paper pack 2: fetch with a wrapper (real bundle)

```yaml
packFormat: 1
id: fetch
version: 0.1.0
kind: framework
vocabVersion: "1.0"
link: [{kind: http, matcher: path-template}]
liftings:
  - id: fetch-url
    role: {kind: http, slot: url}
    seeds: [{call: {callee: fetch, resolution: [external, unresolved]}, arg: a0}]
    companions: [{role: verb, arg: a1.method, bareDefault: GET, unknown: "*"}]
rules:
  - id: http-client
    match: {lifted: {lifting: fetch-url}}
    extract: {url: {lifted: url}, verb: {lifted: verb}}
    emit: [{consumes: {kind: http, key: "{verb} {url}"}}]
```

Seeds are the `fetch` call's URL at `a0` and its verb at `a1.method` (proposal §3.3 "B2"). One
deliberate difference from the slice: the seed requires `resolution` ∈ {`external`, `unresolved`},
so a repo's own function named `fetch` (resolved to an in-repo symbol) is not a seed. The slice
matched the bare name (slice report §6.4(c)); S14 has exactly one `fetch` call site and it is
`external`, so the output is the same.

### 3.1 Walkthrough: `GET /shop/api/health` (SL-1's consumer)

Records are from `s14a`'s `salesperson` bundle. The **wrapper** is `apiFetch`, a function that
forwards its parameter to `fetch`; `buildUrl` is a callee whose return value carries the URL.

| Step | Mechanism | Input record | Result |
|---|---|---|---|
| 1 | seed | `callsite 41d8c78a`: `fetch`, `client.ts:118`, `resolution external` | url slot `(41d8c78a, a0)`; verb slot `(41d8c78a, a1.method)` |
| 2 | call-ret | `valueSet` of `a0` is `unknown point-derived`; `data c214d105` (`use 41d8c78a`, `toPath a0`, `def 5cb61d0b`, `fromPath r`); `5cb61d0b` is the call `buildUrl(...)` (`client.ts:108`), target `beea8ff2`; `buildUrl`'s `ret` has `data` from its `param 23c79216` (`p0`) | lifted slot `(5cb61d0b, a0)`; hop: symbol `buildUrl`, `urlPath p0` |
| 3 | param-def | `data 1f4293ce` (`use 5cb61d0b`, `toPath a0`, `def 3584bd91` = `apiFetch`'s `p0`, `fromPath p0.path`) | lifted parameter `(apiFetch, p0.path)`; hop: symbol `apiFetch`, `urlPath p0.path`, `viaLine 108` |
| 4 | caller | `callsite ac74017f`: `apiFetch`, `endpoints.ts:17`, in `getHealth 8ae57f34`, `targets [apiFetch]` | lifted slot `(ac74017f, a0.path)`; hop: call site, `urlPath a0.path` |
| 5 | resolve | `valueSet 23f6e6c5` (`a0.path`) = `[{template [{ref ".../endpoints.ts::API_PREFIX"}, {const "/health"}]}]`; `constant b2793b7a` (`endpoints.ts:8`) = `"/shop/api"` | no lifted child, resolved: `/shop/api/health` |
| 6 | verb companion | `data e6f94013` (`use 41d8c78a`, `toPath a1.method`, `def 3584bd91`, `fromPath p0.method`) → `(apiFetch, p0.method)` → caller slot `(ac74017f, a0.method)`; `valueSet 0e18bf8b` = `[{const "GET"}]` | first slot on the chain, consumer side first, that resolves: `GET` |
| 7 | `http-client`, `emit.consumes` | `"{verb} {url}"` = `GET /shop/api/health` | core normalizes (consumer form, spec §6 step 8); unchanged |

Emitted: `CONSUMES http GET /shop/api/health` at `endpoints.ts:17` in `getHealth`, `confirmed`,
`lifted-role` (spec §7.2), with the ordered hops

| ord | hop | path:line | `urlPath` | `verbPath` |
|---|---|---|---|---|
| 0 | call site `apiFetch` | `endpoints.ts:17` | `a0.path` | `a0.method` |
| 1 | symbol `apiFetch` | `client.ts:107` (`viaLine` 108) | `p0.path` | `p0.method` |
| 2 | symbol `buildUrl` | `client.ts:73` | `p0` | none |
| 3 | call site `fetch` | `client.ts:118` | `a0` | `a1.method` |

**This equals the slice's output:** `assemble.py`'s consumer record for `getHealth` has `keys
["GET /shop/api/health"]`, `urlPath a0.path`, `partialAlternatives false`, and exactly these four
hops in this order (a fresh run on `s14a`). Against §2.1's provider the key matches exactly:
`Link` `confirmed`, `key-match`, matcher `exact`. (The slice's per-site `reason` is the *match*
result `key-match`; the `CONSUMES` edge's reason is the recognizer's, `lifted-role`. The equality
claimed is on key, status, hops and verb, not on that field.)

### 3.2 Not walked

The three 5-hop chains (a second wrapper, `participantGet`, in front of `apiFetch`) use the same
mechanisms with one more `caller` step. Walking one would put its route in the contact set (§11) for
no new rule exercised, so it is left to the pack runtime's conformance test. A `fetch` whose URL is
wholly unresolved produces no consumer (the slice's `unresolvedConsumers`).

## 4. Paper pack 3: FastMCP plus URL binding (`falkor-chat/server`)

**Placement (eval-protocol §14, §15 Q10, R9).** The server-binding walkthrough runs on
`falkor-chat/server` and nowhere else. `cypher-mcp` and `mcp-monitor` are not walked and no record
of theirs is read: walking `.mcp.json` → `cypher` would cost `wiring` 3 of its at most 10 eligible
items and cap it at 7, short of 8. The one `.mcp.json` entry used is `falkor-chat-agent-team`.

```yaml
packFormat: 1
id: fastmcp
version: 0.1.0
kind: framework
vocabVersion: "1.0"
link: [{kind: mcp_tool, matcher: exact}]
launch:
  frameworkDefaults: [{command: {basename: uvicorn}, port: 8000, protocol: http}]
  portFlags: [{command: {basename: uvicorn}, flags: ["--port"]}]
  targetArg: [{index: 0, form: module-attr}]
  envPorts:
    - envVar: {regex: '(?:[A-Z][A-Z0-9]*_)*PORT'}
      excludeVars: [{regex: '(?:REDIS|POSTGRES|PG|MYSQL|MONGO|KAFKA|FALKORDB|DB|DATABASE|AMQP|RABBITMQ|NATS)_(?:[A-Z0-9]+_)*PORT'}]
      adoptedBy: same-file-launch
  mountPaths:
    - {call: {callee: mount}, path: {arg: a0},
       app: {arg: a1, fromCall: {externalTarget: {glob: '**FastMCP.streamable_http_app'}}}, pathMatch: prefix}
urlBindings:
  - id: mcp-http
    kinds: [mcp_tool]
    urlSources:
      - {config: {keyPath: {regex: 'mcpServers\.[^.]+\.url'}}}
      - {callsite: {callee: streamablehttp_client}, arg: a0, scope: file}
rules:
  - id: tool
    match: {decorator: {form: decorator, name: {regex: '[A-Za-z_][A-Za-z0-9_]*\.tool'}}}
    scope: {imports: [{module: {prefix: mcp.server.fastmcp}}, {module: {prefix: fastmcp}}]}
    extract: {tool: {firstOf: [{arg: kw.name}, {arg: a0}, {target: name}]}}
    emit: [{provides: {kind: mcp_tool, key: "{tool}"}}]
  - id: tool-call
    match: {callsite: {callee: call_tool}}
    scope: {imports: [{module: {prefix: mcp}}]}
    extract: {tool: {arg: a0}}
    emit: [{consumes: {kind: mcp_tool, key: "{tool}"}}]
```

### 4.1 Provider side (real records): the tool `send_message`

| Step | Rule | Input record | Result |
|---|---|---|---|
| 1 | `tool`, `match` | `decorator de368fdf`: `name` `mcp.tool`, `form decorator`, `line 229`, `args []`, `targetId fa4a78d4` (`falkor-chat/server/falkorchat/mcp.py::send_message`) | matches the regex |
| 2 | `scope` | `mcp.py`'s `import` record (`mcp.py:29`): module `mcp.server.fastmcp`, names `[FastMCP]` | gate passes |
| 3 | `extract.tool` | `args` is empty, so `firstOf` falls through `kw.name` and `a0` to `{target: name}` | `send_message` |
| 4 | `emit.provides` | `mcp_tool`, key `send_message` | `keyNorm` `send_message` (spec §6: the function name, case kept) |

Emitted: `PROVIDES mcp_tool send_message` from `mcp.py::send_message`, `confirmed`/`key-match`. The
other 24 `mcp.tool` decorators of `mcp.py` follow the same rule and are not walked (§11). With an
explicit name, an invented one-liner `@mcp.tool(name="lookup")` has `args [{kw.name: [{const "lookup"}]}]`
and gives `lookup`; `@mcp.tool("lookup")` has `a0` and gives `lookup` too, because `FastMCP.tool`'s first
positional parameter is `name` (checked on `mcp` 1.28.1: `signature` is `(self, name=None, title=None,
…)` and `list_tools` reports `lookup`).

**Known false matches.** The recognizer is the receiver-agnostic name `*.tool` plus an import gate,
because a decorator record has no resolution or receiver type (§1.8). It also matches `stub_server.tool`
and `server.tool` in three decorators of two test files that import `FastMCP`. They become
`mcp_tool` providers living under `tests/`: the same unresolved question as the slice's test-file
providers (slice report §6.1, no `isTest` field). The pack does not exclude `tests/` paths, for
the equality reason of §2.

### 4.2 The unit's mount path (real records)

`mountPaths` matches `callsite 6bc8b43c` (`mount`, `app.py:543`): `a0` is `[{const "/mcp"}]`; `a1`
is fed by `data` (`toPath a1`, `fromPath r`) from `callsite` `streamable_http_app` (`app.py:413`,
`externalTarget mcp/server/fastmcp.py:<module>.FastMCP.streamable_http_app`), which the glob
`**FastMCP.streamable_http_app` matches (`*` would not: the string contains `/`; checked by
`cpg/tests/test_rule_pack_spec.py` against the record). Result: the unit `falkor-chat/server` has mount path `/mcp`,
under the condition `mount_mcp` (`controlledBy` → `cond` at `app.py:542`). The two other `mount`
calls of `app.py` (`:528`, `:551`) have `a1` fed by `_SPAStaticFiles(...)` and `StaticFiles(...)`,
which do not match, so they declare no MCP path.

The app's inner endpoint is `/` because `mcp.py:64` assigns `mcp.settings.streamable_http_path =
"/"` (FastMCP's default is `/mcp`). That is an attribute write whose value no bundle record carries
(`fieldAccess` has none, spec §5.13), so the pack cannot read it. `pathMatch: prefix` is the consequence:
a consumer path equal to the mount path or under it matches; the inner path is never compared (O-5).

### 4.3 Launch and config side (paper records, through the argv gate)

Written from the source lines at `defca60f` by spec §1.6/§5.17, then passed through the frozen argv gate
(spec §1.6.1; `cpg/tests/test_rule_pack_spec.py` runs `ccg.bundle.gate_argv` and `bind_value_allowed`
on each). No extractor emits them yet.

| # | Source | Record |
|---|---|---|
| L1 | `falkor-chat/Dockerfile:42` | `kind command`, `command uvicorn`, `args [falkorchat.app:app, --host, 0.0.0.0, --port, 8000, --log-config, /app/config/logging.json]` (the gate leaves it unchanged), `origin explicit`. |
| L2 | `falkor-chat/scripts/start_server.sh:251` | `exec "$VENV_DIR/bin/uvicorn" falkorchat.app:app $UVICORN_ARGS ${LOG_CONFIG_ARGS[@]+…}`: `kind command`, **`command <redacted>`** (the source word contains `$`, which fails the gate's (v1) shape), `args [falkorchat.app:app, <redacted>, <redacted>]` (the two `$`-bearing words; arity is the writer's tokenization), `origin explicit`. |
| L3 | `falkor-chat/scripts/start_agent_team.sh:137` | `kind env-default`, `envVar AGENT_TEAM_PORT`, `default "8200"`, `defaultHash <hmac>`, `origin env-default` (spec §1.6's own example). |
| L4 | `falkor-chat/scripts/start_agent_team.sh:259` | the same shape as L2. |
| L5 | `falkor-chat/scripts/start_agent_team.sh:135` | `kind env-default`, `envVar FALKORDB_PORT`, `default "6379"`. |
| C1 | `.mcp.json`, entry `falkor-chat-agent-team` | `configEntry` `format mcp-json`, `keyPath mcpServers.falkor-chat-agent-team.url`, `bindValue "http://localhost:8200/mcp"` (passes spec §1.7.2 `url` and the §1.7.3 name rule), `example false`. |

### 4.4 Walkthrough: the binding, and the port sources

| Step | Declaration | Input | Result |
|---|---|---|---|
| 1 | `launch.targetArg` | `args[0]` of L1, L2, L4 is `falkorchat.app:app` (the entry constrains no `command`, so the redacted L2/L4 qualify) → `falkorchat/app.py` → exactly one in-repo file ends that way, `falkor-chat/server/falkorchat/app.py` | launched unit = `falkor-chat/server` (python), by three launches |
| 2 | `portFlags` | L1 has `--port 8000` | `(8000, unit)`, origin `explicit` |
| 3 | `frameworkDefaults` | needs `basename uvicorn` and visible args; L1 has `--port` and L2/L4 have `command <redacted>` | **adds nothing**: no `pack-default` port is derivable from the scripts |
| 4 | `envPorts` | L3 `AGENT_TEAM_PORT` matches the regex and no `excludeVars` entry; L4 is a command launch in the same file, whose unit step 1 resolved | `(8200, unit)`, origin `env-default` |
| 5 | `envPorts`, `excludeVars` | L5 `FALKORDB_PORT` matches the regex but also `excludeVars` | adds nothing; without the exclusion the unit would claim `(6379, unit)` |
| 6 | `urlBindings.mcp-http` | C1's `keyPath` matches; `bindValue` parses to host `localhost`, port `8200`, path `/mcp` | candidate ports of the unit: `{8000, 8200}`; 8200 present |
| 7 | path | `/mcp` against mount path `/mcp` (`prefix`, both normalized, §1.3) | match; the condition `mount_mcp` rides on the evidence |
| 8 | result | only `falkor-chat/server` has port 8200 and a `/mcp` mount | bind fact C1 → unit, `confirmed`, `binding-confirmed`; evidence `BOUND_BY`: C1, L3, L4, the mount at `app.py:543` |

**What is not derivable.** The script's real port is `--port ${AGENT_TEAM_PORT}`, the default text of the
`UVICORN_ARGS` assignment (`start_agent_team.sh:157`); that word is not an argv element of any launch
fact, and the gate redacts L4's `$UVICORN_ARGS`. So 8200 rests on one source only: **L3's `envVar`
name read by the `envPorts` heuristic** (a variable-name rule, not data flow), plus the same-file
command launch L4 that names the unit. If the extractor emitted no launch fact for L2/L4, step 4 would
adopt nothing and the binding would rest on 8000 (Dockerfile) and the path, i.e. not select port 8200.

An invented entry `{url: "http://localhost:8000/mcp"}` of the same shape resolves through L1 (explicit):
8000 and 8200 are two deployments of one unit, so one provider and still `confirmed` (proposal §3.4).
The real consumers of 8000 (a Kiro agent and an example config) are not read.

**Tool link (invented consumer).** A client snippet in a file of no repo:

```python
async with streamablehttp_client("http://localhost:8200/mcp") as (r, w, _):
    async with ClientSession(r, w) as s:
        await s.call_tool("send_message", {...})
```

has a `callsite streamablehttp_client` (`a0` `[{const "http://localhost:8200/mcp"}]`) and a `callsite
call_tool` (`a0` `[{const "send_message"}]`). `tool-call` emits `CONSUMES mcp_tool send_message`;
the file-scope URL source binds it to `falkor-chat/server` by steps 6–8; the provider of §4.1 is in
that unit and its key is equal, so the `Link` is `confirmed`, `key-match`, matcher `exact`, with
`BOUND_BY` as in step 8. The tool name has no server in its key (spec §6), so the binding is the only
thing that would discriminate a second unit providing `send_message`.

**Soundness limit, bounded.** `envPorts` is a name heuristic. `excludeVars` is anchored at the start of
the name, so `AGENT_TEAM_FALKORDB_PORT` is adopted although it names a dependency port; a false
`(port, unit)` still needs a consumer URL on that port **and** a matching `/mcp`-style mount path to
produce a link. `frameworkDefaults` is sound only on visible `command` and `args` (§1.6).

## 5. Paper pack 4: Kafka, including a topic read from config

Kafka is a vocabulary example and an invented snippet; it names no proving-repo anchor (eval-protocol
§8.2, "not contact").

```yaml
packFormat: 1
id: kafka
version: 0.1.0
kind: protocol
vocabVersion: "1.0"
vocab:
  - {kind: topic, transport: kafka, keyNorm: "kafka:<name>; name as written, case kept (spec §6, topic row)", example: "kafka:orders.created"}
link: [{kind: topic, matcher: exact}]
rules:
  - id: produce
    match: {callsite: {callee: {regex: 'send|send_and_wait|produce'}}}
    scope: {imports: [{module: {prefix: kafka}}, {module: {prefix: aiokafka}}, {module: {prefix: confluent_kafka}}]}
    extract: {topic: {arg: a0}}
    emit: [{provides: {kind: topic, key: "kafka:{topic}"}}]
  - id: consume-positional
    match: {callsite: {callee: {regex: 'KafkaConsumer|AIOKafkaConsumer'}}}
    scope: {imports: [{module: {prefix: kafka}}, {module: {prefix: aiokafka}}]}
    extract: {topic: {arg: a0, onUnknown: config-derived}}
    emit: [{consumes: {kind: topic, key: "kafka:{topic}"}}]
  - id: consume-pattern
    match: {callsite: {callee: subscribe, args: [{arg: kw.pattern}]}}
    scope: {imports: [{module: {prefix: kafka}}, {module: {prefix: aiokafka}}]}
    extract: {pattern: {arg: kw.pattern}}
    emit: [{consumes: {kind: topic, key: "kafka:{pattern}", keyForm: regex}}]
```

**Topic direction (ruled).** A publisher `PROVIDES` and a subscriber `CONSUMES` a `topic`, with a `Link`
between them: AC-23's "links appear", and spec §7.2's `config-derived` "on `CONSUMES topic`" and `Link`
codes read as written. Topics are not B1 shared state (proposal §3.3, graph note §6.2 say so).

**Preconditions for S32's topic assertions.** (1) *Keyless and pattern keys have no home in
graph model D4* (`ContractPoint {cpId, kind, key, keyNorm}`): §5.2 step 2's keyless `CONSUMES` and the
`regex`/`glob` keys of §5.2 step 3 and §6 cannot be asserted until `graph-dba`/S17 rules (O-15, open). (2) *A
publisher whose topic is built at runtime or read from config* is a possible `PROVIDES topic`
(`onUnknown` defaults to `config-derived` for it, §1.4; spec3 §7.2 lists the codes on `CONSUMES`/`READS`/`WRITES`
and `CONSUMES topic` only, so the `PROVIDES topic` cells are §1.9's, O-16, shipped in `reasons.yaml`). Until
O-15 is ruled, S32 asserts only the literal-key facts (§5.2 step 1, §6 step 1).

### 5.1 Snippet and records (invented; shapes are spec §5.6, §5.8, §5.13)

```python
# orders/producer.py
from kafka import KafkaProducer
producer = KafkaProducer(bootstrap_servers="kafka:9092")
def publish_created(order): producer.send("orders.created", value=order)

# orders/consumer.py
from kafka import KafkaConsumer
from app.config import settings
def start():  return KafkaConsumer(settings.orders_topic, bootstrap_servers="kafka:9092")
def audit(c): c.subscribe(pattern=r"^orders\..*")
```

| Record | Fields |
|---|---|
| `import` (producer.py) | `module kafka`, `names [{KafkaProducer}]` |
| `callsite P` | `calleeName send`, `resolution external`; `valueSet a0 = [{const "orders.created"}]`, `kw.value = [{unknown point-derived}]` |
| `import` (consumer.py) | `module kafka`, `names [{KafkaConsumer}]` |
| `callsite C` | `calleeName KafkaConsumer`; `valueSet a0 = [{unknown point-derived}]` |
| `fieldAccess F` | `pointId C`, `path a0`, `mode read`, `receiverType "?"`, `field orders_topic`, `possible`/`field-name-match` |
| `callsite S` | `calleeName subscribe`; `valueSet kw.pattern = [{const "^orders\\..*"}]` |

### 5.2 Walkthrough, rule by rule

1. **`produce` on `P`.** `send` matches the regex; the file imports `kafka` (gate); `topic` =
   `orders.created`. Emitted: `PROVIDES topic kafka:orders.created`, `confirmed`/`key-match`.
2. **`consume-positional` on `C`.** `KafkaConsumer` matches; `a0` has no resolved alternative, so
   `onUnknown: config-derived` applies, and `F` is the `fieldAccess` read on that point. Emitted:
   `CONSUMES topic` with **no key**, `possible`/`config-derived` (spec §7.2). The core reads `F` into
   `READS field ?.orders_topic` (spec §5.13); mapping that field to a configuration entry is a data-store
   pack's job (e.g. a `pydantic-settings` pack, not in this set). **No topic name is recovered, and none
   can be:** AC-6's topic-from-config outcome is exactly this fact.
3. **`consume-pattern` on `S`.** `subscribe` has `kw.pattern`. Emitted: `CONSUMES topic
   kafka:^orders\..*` with `keyForm regex`, `confirmed`/`key-match`.
4. **Linking (core, proposal §3.4).** Step 3's regex key, matched by `regex` against step 1's literal
   key, gives a `Link` `confirmed`, `key-match`, matcher `regex`. Step 2's consumer has no key and no
   binding: one `Link` to **each** provider of kind `topic`, `possible`, `unknown-key`, each
   enumerated with its own chain, folded only in rendering (ML M2).

**Known false matches.** `produce` matches any call named `send`/`produce` in a file that imports `kafka`
(a socket `send`, a queue `produce`); a decorator or call record has no receiver type (§1.8).

The regex in step 3 is read as §1.3's RE2 subset against the part after `kafka:`; a pattern outside
the subset (a lookaround, which Java's `Pattern` allows) is `runtime-built`, never a guess.

## 6. Paper pack 5: an invented in-house bus

The bus, `acmebus`, and its repository are invented and name no proving-repo anchor. It exercises
what the other four do not: a **pack-declared transport**, a **glob** subscription, a key
composed from a **class-level decorator**, and a **bundle query** (§1.7). The pack's directory is
`cpg/packs/paper-inhouse-bus/` (S32), so its id is `paper-inhouse-bus` and its transport token,
by §1.6's `inhouse-<packId>` rule, is `inhouse-paper-inhouse-bus`.

```yaml
packFormat: 1
id: paper-inhouse-bus
version: 0.1.0
kind: protocol
vocabVersion: "1.0"
vocab:
  - {kind: topic, transport: inhouse-paper-inhouse-bus,
     keyNorm: "inhouse-paper-inhouse-bus:<name>; name as written, case kept (spec §6, topic row)",
     example: "inhouse-paper-inhouse-bus:billing.invoice.paid"}
link: [{kind: topic, matcher: exact}]
queries:
  - id: handler-ns                                  # the worked example of §1.7
    from:
      - {as: dec,  record: decorator, where: {name: {regex: '(?:[A-Za-z_][A-Za-z0-9_]*\.)?on'}}}
      - {as: h,    record: symbol,    where: {id: "$dec.targetId"}}
      - {as: ty,   record: symbol,    where: {id: "$h.parentId"}}
      - {as: ns,   record: decorator, where: {targetId: "$ty.id", name: {regex: '(?:[A-Za-z_][A-Za-z0-9_]*\.)?component'}}}
    yield: {handler: "$h.id", topic: "$dec.args.a0", ns: "$ns.args.a0"}
rules:
  - id: publish
    match: {callsite: {callee: publish}}
    scope: {imports: [{module: {exact: acmebus}}]}
    extract: {topic: {arg: a0}}
    emit: [{provides: {kind: topic, key: "inhouse-paper-inhouse-bus:{topic}"}}]
  - id: subscribe
    match: {query: handler-ns}
    scope: {imports: [{module: {exact: acmebus}}]}
    extract: {ns: {yield: ns}, topic: {yield: topic}}
    emit: [{consumes: {kind: topic, key: "inhouse-paper-inhouse-bus:{ns}.{topic}", keyForm: glob}}]
```

For a `query` subject, `scope` applies to the file of the row's `handler` symbol and the emission's
anchor is that symbol (§1.7).

### 6.1 Snippet and records (invented)

```python
# billing/events.py
from acmebus import Bus, component, on
bus = Bus()
TOPIC_PAID = "billing.invoice.paid"

@component("billing")
class InvoiceHandlers:
    @on("invoice.*")
    def handle(self, evt): ...

def notify(inv): bus.publish(TOPIC_PAID, inv)
```

| Record | Fields |
|---|---|
| `constant` | `billing/events.py::TOPIC_PAID`, `values [{const "billing.invoice.paid"}]` |
| `callsite` | `calleeName publish`, `valueSet a0 = [{ref "billing/events.py::TOPIC_PAID"}]` |
| `symbol` | `InvoiceHandlers` (`kind type`); `handle` (`kind method`, `parentId` = the type) |
| `decorator` on the type | `name component`, `form decorator`, `args [{a0: [{const "billing"}]}]` |
| `decorator` on `handle` | `name on`, `args [{a0: [{const "invoice.*"}]}]` |

### 6.2 Walkthrough

1. **`publish`.** `publish` matches; the file imports `acmebus` (gate). `a0` is a `ref`, resolved
   through the `constant` to `billing.invoice.paid`. Emitted: `PROVIDES topic
   inhouse-paper-inhouse-bus:billing.invoice.paid`, `confirmed`/`key-match`.
2. **`handler-ns`.** `dec` binds the `on` decorator; `h` its target `handle`; `ty` `handle`'s parent
   type; `ns` the `component` decorator whose `targetId` is that type. One row: `{handler: handle,
   topic: "invoice.*", ns: "billing"}`. (Without the class decorator the last join fails and the
   handler is not recognized: the query is an inner join by design. A pack wanting both writes two rules.)
3. **`subscribe`.** Key `inhouse-paper-inhouse-bus:billing.invoice.*`, `keyForm glob`. Emitted:
   `CONSUMES topic …` anchored at `handle`, `confirmed`/`key-match`.
4. **Linking.** The glob key matched against step 1's literal key (`*` spans `paid`) gives a `Link`
   `confirmed`, `key-match`, matcher `glob`.

## 7. Language expressiveness (S27 input)

The plan asks that the format express the decorator, annotation and attribute shapes S27 measured
for all seven languages, or that S30 record the gap. S27 has not run, so this section states (a) what
the format can say, from the frozen record types, (b) what S14 shows per language, and (c) the exact
questions **S27 input** must answer. It states no per-language shape S14 did not show.

**(a) What a rule can say about a registration.** A decorator, annotation or attribute is one
record, `decorator` (spec §5.12): `form` ∈ {`decorator`, `annotation`, `attribute`}; `name` (the
expression without its arguments, whitespace removed; an `Attribute` suffix is not stripped);
`args` over the spec §5.1 paths (`a<i>` positional, `kw.<name>` named: a Python keyword, a C# named
argument, a Java/Kotlin annotation element); the decorated `targetId` (a function, method or type).
A registration made by a **call** instead is a `callsite` subject (`callee`, `args`, `externalTarget`),
with its anchor an enclosing symbol or an `a<i>` that resolves to a symbol. A prefix from a type-level
decorator is `mounts.kind: class-decorator` (§1.6) or a query (§1.7).

**(b) What S14 shows.**

| Language token | S14 evidence | Expressible in v1 |
|---|---|---|
| `python` | `falkor-chat/server` bundle: 213 `decorator` records, `form decorator`; names seen include `router.get`, `router.post`, `app.get`, `mcp.tool`, `pytest.fixture`, `property`; `a0` a `const` route string, or `args []` | yes: §2 and §4 walk it |
| `javascript` (TS/JS) | `salesperson` bundle: **0** `decorator` records; registrations in that bundle are calls (`fetch`, `apiFetch`) | calls: yes (§3); decorator shapes: **S27 input** |
| `java`, `kotlin`, `csharp`, `go`, `cpp` (with `c`) | no S14 unit | **S27 input** for every shape |

(Shell and Markdown have no decorators. spec §5.12's own examples name `GetMapping` and `HttpGet`; they
are spec text, not measurements, and S27 input confirms or replaces them.)

**(c) What S27 input must return, per language** (so a pack author can write the rule): (1) the
registration forms that occur, each as `form` or call; (2) the `name`/`callee` text the bundle
records for the common route and tool registrations of that language's frameworks; (3) the argument
path that carries the route or name (`a0`, `kw.<name>`, or a flattened field); (4) whether a type-level
decorator composes a prefix (`class-decorator`) and the shape; (5) which of these S27's measured
corpora contain.

**Gaps recorded now (each follows from the frozen records, not from a language):**
- *Parameter-level annotations.* `decorator.targetId` is a function, method or type (spec §5.12); an
  annotation on a **parameter** is not a record. A rule needing it (a path-variable or required-header
  annotation) cannot be written (O-7).
- *Receiver identity.* Neither `decorator` nor `callsite` carries receiver text or an assigned-variable
  name, so `mcp.tool` and `other.tool` are indistinguishable except by an import gate (§4.1). Where the
  frontend resolved a call, `externalTarget` narrows it (§4.2); a decorator has no resolution (O-8).
- *Array-valued arguments.* spec §5.8 has no list element, and what the extractor writes for
  `subscribe(["a","b"])` is unspecified, so no rule may rely on it. The Kafka pack uses the positional
  and keyword forms for that reason (§5).
- *Module-variable routers* (§2.3) and *attribute writes* (§4.2) have no record.

## 8. Infrastructure-signature pack v0

**What it is for.** RCA's "outside the mapped code" branch must be scored, not decorative (ML §8.3,
`docs/plans/connected-code-graph-ml.md`; proposal §3.7): an error whose evidence matches an
infrastructure signature is a candidate cause beyond the repository's code. This pack is the
**signature classes and per-runtime patterns only**. Weights and the ranking belong to the ranker
(ML §8.3); a wrong port in a mapped config file is a *wiring* hypothesis traced to its entry, not
an infrastructure signature (ML §8.3), so nothing here matches configuration.

### 8.1 Pack shape (`kind: infra-signatures`)

```yaml
packFormat: 1
id: infra-signatures
version: 0.1.0
kind: infra-signatures
vocabVersion: "1.0"
runtimes:                       # detection narrows, per evidence block; see below
  - {runtime: python, regex: 'Traceback \(most recent call last\):|^\s+File ".*", line \d+, in '}
  - {runtime: node,   regex: '^\s+at .+:\d+:\d+\)?$|node:internal'}
  - {runtime: jvm,    regex: '^\s+at [\w.$]+\([\w$]+\.(?:java|kt|scala):\d+\)|^Caused by: [\w.$]+'}
  - {runtime: dotnet, regex: '^\s+at [^ ]+\(.*\) in .+:line \d+'}
  - {runtime: go,     regex: '^goroutine \d+ \[|^\s+\S+\.go:\d+ \+0x'}
```

A `signatures` entry is `{id, class, runtime, regex, sample, sampleSource?, library?}`:
- `id` is `<runtime>-<class>` or, for `third-party-exception`, `<runtime>-third-party[-<library>]`.
- `class` is the closed set `connection-refused`, `dns-failure`, `timeout`, `oom-kill`, `tls-error`,
  `third-party-exception` (the ML §8.3 list). `runtime` is `python`, `node`, `jvm`, `dotnet`, `go` or
  `any` (container, kernel, or CLI text with no language marker).
- `regex` is §1.3's RE2 subset, but for this subject it is applied by **search, per line** (a
  signature is a fragment of a log or trace line, not the whole line), case-sensitive.
- `sample` is a string its own regex must match. `sampleSource` is `captured` (a verbatim line from a
  failure provoked on this lab's machine, `cpg/tests/data/infra_captures.tsv`), `runtime-source`
  (verbatim from the runtime's or library's source or documentation) or `authored` (the default: written
  by the regex's author, so a typo guard only). **A sample supports only the alternative it matches.**
  An entry is *supported* only when every alternative of its regex, at any group depth (not only top-level `|`), has a `captured` or
  `runtime-source` sample; an entry with any `authored` alternative is at most *partly supported* (say
  which alternatives) and an `authored`-only entry is unvalidated. No report may call a partly
  supported or unvalidated entry supported.
- **Evidence blocks.** Evidence is split into blocks by the Compose service prefix
  (`^[\w.-]+\s+\| `, stripped before matching); unprefixed lines are one block. **Detection narrows per
  block, never gates:** when a `runtimes` regex matches a line of a block, only that runtime's entries
  and the `any` entries apply to that block; when none matches, every entry applies, because the
  patterns are distinctive names.
- **Emission.** A match emits `signature {class, runtime, id}` (and `library`) per block and line.
  The pack's output is deduplicated on `(block, line, class)` (`line` is the line's occurrence in its block, not its text: two identical lines are two pairs), and a `third-party-exception` match
  collapses into another class matched on the same line, keeping its `library`. The ranker scores the
  deduplicated `(class, line)` pairs, so one failure is one pair per class it matches (a Go `i/o timeout`
  inside a DNS lookup is two). The weights are the ranker's (ML §8.3).
- **Dedupe, exactly.** For each block and line, take the matched entries *M* and their classes. If *M*
  holds a class other than `third-party-exception`, the third-party entries add their `library` values to
  those pairs and form no pair of their own; otherwise all `third-party-exception` matches on the line
  form one pair.
  Each pair is `{block, line, class, ids (sorted), libraries (sorted)}`; two entries of one class on one
  line are one pair, whichever libraries they name. Example: the psycopg line of §8.2 is one `connection-refused` pair with
  `libraries [psycopg]`, not two pairs; `sqlalchemy.exc.OperationalError: (psycopg2.OperationalError)
  server closed the connection unexpectedly` is one `third-party-exception` pair with `libraries
  [psycopg, sqlalchemy]`, not one pair per library.

### 8.2 Patterns, per runtime

Patterns come from the runtimes' and libraries' documented exception and error names. Of the 45
entries, **11 carry a `captured` sample** (provoked on one machine: Python 3.12.3 with redis-py and
httpx, and OpenJDK 21), each for **one alternative**; only the 3 single-alternative ones
(`python-oom-kill`, `jvm-dns-failure`, `jvm-oom-kill`) are wholly supported, and **34 are `authored`,
hence unvalidated**: every `node`, `dotnet`, `go` and `any` entry and the rest of `python` and `jvm`.
The captures are **dev data**: taken by the regex author, one machine, and the regexes were written and
revised *after* seeing them, so they are no independent validation. `python-oom-kill`'s `MemoryError` is
provoked in-process; a container OOM kill gives the process a SIGKILL and no such line (the real signal
is `any-oom-kill`'s authored text). The proving repository is Python and TypeScript, so `jvm`,
`dotnet` and `go` entries (21) can never meet a recorded failure there. They are listed for the
format's sake and carry no claim.

```yaml
# python
- {id: python-connection-refused, class: connection-refused, runtime: python, regex: 'ConnectionRefusedError|ConnectionResetError|\[Errno (?:111|61|113|101|104)\]', sample: 'ConnectionRefusedError: [Errno 111] Connection refused', sampleSource: captured}
- {id: python-dns-failure, class: dns-failure, runtime: python, regex: 'socket\.gaierror|\[Errno -(?:2|3|5)\] (?:Name or service not known|Temporary failure in name resolution|No address associated with hostname)|nodename nor servname provided', sample: 'socket.gaierror: [Errno -2] Name or service not known', sampleSource: captured}
- {id: python-timeout, class: timeout, runtime: python, regex: 'socket\.timeout|\bTimeoutError\b|\b(?:Read|Connect|Pool)Timeout\b|\[Errno 110\]', sample: 'TimeoutError: timed out', sampleSource: captured}
- {id: python-oom-kill, class: oom-kill, runtime: python, regex: '\bMemoryError\b', sample: 'MemoryError', sampleSource: captured}
- {id: python-tls-error, class: tls-error, runtime: python, regex: 'ssl\.SSLError|SSLCertVerificationError|CERTIFICATE_VERIFY_FAILED|ssl\.CertificateError', sample: 'ssl.SSLCertVerificationError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: self-signed certificate (_ssl.c:1000)', sampleSource: captured}
- {id: python-third-party-redis, class: third-party-exception, runtime: python, library: redis, regex: 'redis\.exceptions\.(?:ConnectionError|TimeoutError|BusyLoadingError)', sample: 'redis.exceptions.ConnectionError: Error 111 connecting to localhost:1. Connection refused.', sampleSource: captured}
- {id: python-third-party-psycopg, class: third-party-exception, runtime: python, library: psycopg, regex: 'psycopg2?\.OperationalError', sample: 'psycopg2.OperationalError: could not connect to server'}
- {id: python-third-party-sqlalchemy, class: third-party-exception, runtime: python, library: sqlalchemy, regex: 'sqlalchemy\.exc\.OperationalError', sample: 'sqlalchemy.exc.OperationalError: (psycopg2.OperationalError)'}
- {id: python-third-party-pymongo, class: third-party-exception, runtime: python, library: pymongo, regex: 'pymongo\.errors\.(?:ServerSelectionTimeoutError|AutoReconnect)', sample: 'pymongo.errors.ServerSelectionTimeoutError: db:27017'}
- {id: python-third-party-kafka, class: third-party-exception, runtime: python, library: kafka-python, regex: 'kafka\.errors\.(?:NoBrokersAvailable|KafkaTimeoutError)', sample: 'kafka.errors.NoBrokersAvailable: NoBrokersAvailable'}
- {id: python-third-party-botocore, class: third-party-exception, runtime: python, library: botocore, regex: 'botocore\.exceptions\.(?:EndpointConnectionError|ConnectTimeoutError)', sample: 'botocore.exceptions.EndpointConnectionError: Could not connect'}
- {id: python-third-party-requests, class: third-party-exception, runtime: python, library: requests, regex: 'requests\.exceptions\.(?:ConnectionError|ConnectTimeout|ReadTimeout)', sample: 'requests.exceptions.ConnectionError: HTTPConnectionPool'}
- {id: python-third-party-httpx, class: third-party-exception, runtime: python, library: httpx, regex: 'httpx\.(?:ConnectError|ConnectTimeout|ReadTimeout|PoolTimeout)', sample: 'httpx.ConnectError: [Errno 111] Connection refused', sampleSource: captured}
# node
- {id: node-connection-refused, class: connection-refused, runtime: node, regex: '\b(?:ECONNREFUSED|ECONNRESET|EHOSTUNREACH|ENETUNREACH)\b', sample: 'Error: connect ECONNREFUSED 127.0.0.1:5432'}
- {id: node-dns-failure, class: dns-failure, runtime: node, regex: '\b(?:ENOTFOUND|EAI_AGAIN)\b', sample: 'Error: getaddrinfo ENOTFOUND db.internal'}
- {id: node-timeout, class: timeout, runtime: node, regex: '\b(?:ETIMEDOUT|ESOCKETTIMEDOUT|UND_ERR_CONNECT_TIMEOUT|UND_ERR_HEADERS_TIMEOUT)\b', sample: 'Error: connect ETIMEDOUT 10.0.0.5:443'}
- {id: node-oom-kill, class: oom-kill, runtime: node, regex: 'JavaScript heap out of memory|FATAL ERROR: .*Allocation failed', sample: 'FATAL ERROR: Reached heap limit Allocation failed - JavaScript heap out of memory'}
- {id: node-tls-error, class: tls-error, runtime: node, regex: '\b(?:UNABLE_TO_VERIFY_LEAF_SIGNATURE|UNABLE_TO_GET_ISSUER_CERT_LOCALLY|CERT_HAS_EXPIRED|DEPTH_ZERO_SELF_SIGNED_CERT|SELF_SIGNED_CERT_IN_CHAIN|ERR_TLS_CERT_ALTNAME_INVALID)\b', sample: 'code: UNABLE_TO_VERIFY_LEAF_SIGNATURE'}
- {id: node-third-party, class: third-party-exception, runtime: node, regex: '\b(?:MongoServerSelectionError|MongoNetworkError|KafkaJSConnectionError|KafkaJSNumberOfRetriesExceeded|MaxRetriesPerRequestError)\b', sample: 'MongoServerSelectionError: connect ECONNREFUSED'}
# jvm
- {id: jvm-connection-refused, class: connection-refused, runtime: jvm, regex: 'java\.net\.(?:ConnectException: Connection refused|NoRouteToHostException|SocketException: Connection reset)', sample: 'java.net.ConnectException: Connection refused', sampleSource: captured}
- {id: jvm-dns-failure, class: dns-failure, runtime: jvm, regex: 'java\.net\.UnknownHostException', sample: 'java.net.UnknownHostException: nonexistent.invalid', sampleSource: captured}
- {id: jvm-timeout, class: timeout, runtime: jvm, regex: 'java\.net\.SocketTimeoutException|java\.net\.http\.HttpTimeoutException|java\.util\.concurrent\.TimeoutException|ConnectTimeoutException', sample: 'java.net.SocketTimeoutException: Read timed out', sampleSource: captured}
- {id: jvm-oom-kill, class: oom-kill, runtime: jvm, regex: 'java\.lang\.OutOfMemoryError', sample: 'java.lang.OutOfMemoryError: Java heap space', sampleSource: captured}
- {id: jvm-tls-error, class: tls-error, runtime: jvm, regex: 'javax\.net\.ssl\.SSL(?:HandshakeException|PeerUnverifiedException)|sun\.security\.validator\.ValidatorException|PKIX path building failed', sample: 'javax.net.ssl.SSLHandshakeException: PKIX path building failed'}
- {id: jvm-third-party-postgresql, class: third-party-exception, runtime: jvm, library: postgresql, regex: 'PSQLException: (?:Connection to .* refused|The connection attempt failed)', sample: 'org.postgresql.util.PSQLException: Connection to db:5432 refused.'}
- {id: jvm-third-party-hikari, class: third-party-exception, runtime: jvm, library: hikaricp, regex: 'HikariPool\$PoolInitializationException|java\.sql\.SQLTransientConnectionException', sample: 'java.sql.SQLTransientConnectionException: HikariPool-1 - Connection is not available'}
- {id: jvm-third-party-redis, class: third-party-exception, runtime: jvm, library: jedis-lettuce, regex: 'JedisConnectionException|io\.lettuce\.core\.RedisConnectionException', sample: 'redis.clients.jedis.exceptions.JedisConnectionException: Could not get a resource'}
- {id: jvm-third-party-kafka, class: third-party-exception, runtime: jvm, library: kafka-clients, regex: 'org\.apache\.kafka\.common\.errors\.(?:TimeoutException|DisconnectException)', sample: 'org.apache.kafka.common.errors.TimeoutException: Failed to update metadata'}
# dotnet
- {id: dotnet-connection-refused, class: connection-refused, runtime: dotnet, regex: 'SocketException \((?:111|61|10061)\)|actively refused it', sample: 'System.Net.Sockets.SocketException (111): Connection refused'}
- {id: dotnet-dns-failure, class: dns-failure, runtime: dotnet, regex: 'No such host is known|(?:SocketException|HttpRequestException)[^:]*: (?:Name or service not known|Temporary failure in name resolution)', sample: 'System.Net.Sockets.SocketException: No such host is known.'}
- {id: dotnet-timeout, class: timeout, runtime: dotnet, regex: 'System\.TimeoutException|SocketException \((?:110|10060)\)|connected party did not properly respond|HttpClient\.Timeout of', sample: 'System.TimeoutException: The operation has timed out.'}
- {id: dotnet-oom-kill, class: oom-kill, runtime: dotnet, regex: 'System\.OutOfMemoryException', sample: 'System.OutOfMemoryException: Insufficient memory'}
- {id: dotnet-tls-error, class: tls-error, runtime: dotnet, regex: 'System\.Security\.Authentication\.AuthenticationException|The remote certificate is invalid|RemoteCertificateNameMismatch', sample: 'AuthenticationException: The remote certificate is invalid'}
- {id: dotnet-third-party, class: third-party-exception, runtime: dotnet, regex: 'Npgsql\.NpgsqlException|StackExchange\.Redis\.RedisConnectionException|Microsoft\.Data\.SqlClient\.SqlException|Confluent\.Kafka\.KafkaException', sample: 'Npgsql.NpgsqlException (0x80004005): Failed to connect to 10.0.0.1:5432'}
# go
- {id: go-connection-refused, class: connection-refused, runtime: go, regex: 'dial tcp \S+: connect: (?:connection refused|no route to host|network is unreachable)|read tcp .*: connection reset by peer', sample: 'dial tcp 127.0.0.1:5432: connect: connection refused'}
- {id: go-dns-failure, class: dns-failure, runtime: go, regex: 'lookup \S+.*: (?:no such host|server misbehaving)', sample: 'dial tcp: lookup db on 127.0.0.11:53: no such host'}
- {id: go-timeout, class: timeout, runtime: go, regex: 'context deadline exceeded|i/o timeout|Client\.Timeout exceeded', sample: 'Get "http://x": context deadline exceeded'}
- {id: go-oom-kill, class: oom-kill, runtime: go, regex: 'fatal error: (?:runtime: )?out of memory|runtime: out of memory', sample: 'fatal error: runtime: out of memory'}
- {id: go-tls-error, class: tls-error, runtime: go, regex: 'x509: (?:certificate|failed to verify)|tls: (?:failed to verify|handshake failure|bad certificate)', sample: 'x509: certificate signed by unknown authority'}
- {id: go-third-party, class: third-party-exception, runtime: go, regex: 'redis: connection pool timeout|pq: (?:the database system is starting up|sorry, too many clients already)|kafka: client has run out of available brokers|server selection error', sample: 'redis: connection pool timeout'}
# any (container, kernel, CLI)
- {id: any-connection-refused, class: connection-refused, runtime: any, regex: 'curl: \(7\) Failed to connect|[Cc]onnection refused|\bECONNREFUSED\b|[Cc]onnection reset by peer', sample: 'curl: (7) Failed to connect to localhost port 8200'}
- {id: any-dns-failure, class: dns-failure, runtime: any, regex: 'Could not resolve host', sample: 'curl: (6) Could not resolve host: svc'}
- {id: any-timeout, class: timeout, runtime: any, regex: 'Gateway Time-?out|upstream request timeout|curl: \(28\)', sample: '504 Gateway Time-out'}
- {id: any-oom-kill, class: oom-kill, runtime: any, regex: 'OOMKilled|Out of memory: Killed process|exit(?:ed)?(?: with)? code 137\b|Exited \(137\)', sample: 'container exited with code 137 (OOMKilled)'}
- {id: any-tls-error, class: tls-error, runtime: any, regex: 'SSL certificate problem|curl: \((?:35|60)\)', sample: 'curl: (60) SSL certificate problem: self signed certificate'}
```

**Negative controls** (test data, not pack content). Every line must match **no** entry when all
entries apply, and each class has near-miss lines, ones a looser signature (a bare trigger word, or a form since removed) would match:

```yaml
# negatives
shared:
  - 'ValueError: invalid literal for int()'
  - 'KeyError: "id"'
  - 'TypeError: x is not a function'
  - 'NullPointerException at com.acme.Foo(Foo.java:12)'
  - 'System.InvalidOperationException: Sequence contains no elements'
  - 'panic: runtime error: index out of range [3] with length 3'
  - 'AssertionError: expected 3 got 4'
near-miss:
  timeout:
    - 'ValueError: Test timed out waiting for user input'
    - 'RuntimeError: request timed out after retries (app-level)'
    - 'INFO retrying after timeout (attempt 2)'
    - 'tests/test_gateway_timeout_handler.py PASSED'
  oom-kill:
    - 'ERROR: container exit code 1370'
    - 'tests/test_oom_kill_handler.py PASSED'
    - 'redis.exceptions.ResponseError: oom-kill'
    - 'tests/test_memory_error_handler.py PASSED'
  connection-refused:
    - 'Network is unreachable (cached)'
    - 'tests/test_connection_refused_handler.py PASSED'
    - 'INFO connection pool reset scheduled'
    - 'WARN refused to merge the pull request'
  dns-failure:
    - 'ValueError: could not resolve ambiguous name'
    - 'INFO lookup table updated: no such key'
    - 'INFO name resolution cache warmed'
  tls-error:
    - 'ValueError: certificate expiry must be in the future'
    - 'INFO ssl_mode=require parsed'
    - 'ERROR: CertificateStore.load() returned no entries'
  third-party-exception:
    - 'psycopg2.errors.UniqueViolation: duplicate key value'
    - 'redis.exceptions.ResponseError: WRONGTYPE Operation against a key'
    - 'requests.exceptions.InvalidURL: Invalid URL'
known-false-matches:               # app text the bare phrases still match; S32 asserts they DO match (documented imprecision)
  - {line: 'WARNING: Connection refused handled gracefully by retry wrapper', ids: [any-connection-refused]}
  - {line: 'INFO exit code 137 handled', ids: [any-oom-kill]}
dedupe:                           # one pair per (block, line, class); third-party collapses and keeps its library
  - {line: 'psycopg2.OperationalError: connection to server at "db" port 5432 failed: Connection refused', pairs: [{class: connection-refused, libraries: [psycopg]}]}
cross-runtime:                    # a line, and the exact set of ids that may match it, all entries applying
  - {line: 'httpx.ConnectError: [Errno -2] Name or service not known', ids: [python-dns-failure, python-third-party-httpx]}
  - {line: 'psycopg2.OperationalError: connection to server at "db" (172.18.0.2), port 5432 failed: Connection refused', ids: [python-third-party-psycopg, any-connection-refused]}
mixed-blob:                       # per-block narrowing: both services' signatures survive
  evidence: |
    api-1  | Traceback (most recent call last):
    api-1  |   File "/app/x.py", line 3, in main
    api-1  | ConnectionRefusedError: [Errno 111] Connection refused
    web-1  | Error: connect ECONNREFUSED 127.0.0.1:5432
    web-1  |     at TCPConnectWrap.afterConnect [as oncomplete] (node:net:1555:16)
  ids: [python-connection-refused, any-connection-refused, node-connection-refused]
```

S32 asserts every sample, every negative, each `cross-runtime` and `mixed-blob` case, and a mutation
check: a regex of a named entry broadened to `.*` fails at least one negative of its class (N-22).

**Known holes, for P5 to read as holes and not as surprises.** `No route to host`, `Network is
unreachable` and `Connection reset by peer` are folded into `connection-refused` (they are "could not
(keep a) connection" and the closed class set belongs to ML §8.3; whether it needs a `connection-lost`
class is that owner's call, §12.2). A bare `Killed` and a container that exits with no text match nothing.
`python-timeout` still matches an application that raises `TimeoutError` for its own reasons;
`any-oom-kill` matches a `SIGKILL` that was not an OOM kill; `any-connection-refused` matches an
application that prints `Connection refused` or `Connection reset by peer` for its own reasons (the bare
phrase is kept because psycopg and redis print it with no exception name); `Network is unreachable` and
`No route to host` as bare text are no longer matched outside the Python errno, Node, JVM and Go forms; `go-timeout` matches `i/o timeout`
inside DNS errors. A false signature never suppresses a mapped-code hypothesis, but it can **lower a
mapped-code bug's rank** (the outside-branch score is evaluated, ML §8.3), so precision is measured on
the negatives above, run together with ML §8.3's AC-12 "no match" controls (≥ 20 inputs).

**P5 validation (what a recorded failure may and may not do).** ML §8.3's ≥ 5 recorded infrastructure
failures (stop the database, kill the broker, block a route) are recorded as **dev or held-out before
any edit**, by a fixed rule (alternating by record order, dev first). Pack edits use dev failures only;
the held-out failures are scored once, on the final pack, and then the held-out set is spent. P5
reports **ranks and counts, not rates** (5 of 5 has a Wilson 95% lower bound of about 0.57, so no recall
claim is possible), and the `data-scientist` consult at P5 happens on the dev/held-out split **before**
the pack is edited.

## 9. Severity rule table for non-dependency findings

FR-15 asks for a four-level severity with its reason, from stated rules, on every finding;
dependencies take the advisory's score (proposal §3.7) and are out of this table. The four kinds
here are injection-type, missing authorization, risky configuration and secrets. `security-expert`
may advise on these rules; that is not a gate. **Severity here is policy, not a measured quantity**:
there is no labelled severity data and AC-15 requires only a severity with its reason, so the table
is checked for internal consistency (§9.3), not for accuracy, and a rank of one kind is not comparable
with a rank of another (an INJ-1 and an RCFG-1 `critical` come from different bases).

### 9.1 Method: the formula is the single source

For a finding of kind *k* with feature values *f* (facts about the chain, computed by the core), **in this
order**: (1) `base` = the one `base` entry of *k* whose `when` matches *f*; (2) `+ adjust` entries
that match, **capped at 3** (secrets only); (3) `− penalties` that match; (4) floor 0; (5) a matching
`caps` entry limits the result to its `max` (an example-role file gives 0). The **rank** 3, 2, 1, 0 is
`critical`, `high`, `medium`, `low`. The **row** is derived: `INJ`, `AUTHZ`, `RCFG` or `SEC` plus `-(4 −
rank)`, and the finding's **reason** is the registry code `severity-rule:<row id>` (the 16 ids; §1.9).
The finding also shows the feature values behind the rank. A finding in an example file has status
`possible` with reason `example-config` beside its rule (a second reason on the finding; `Finding` is added to that code's `On` set in §1.9). No other prose states a row's
rule; `rows` only names the levels.

`when` maps a feature to a value (equality) or, for `path` and `basename`, to a §1.3 matcher over the
entry's file path or its last component. The **`when` vocabulary** (closed; a feature outside a row's
kind, or a value outside its set, is rejected; `cpg/tests/test_rule_pack_spec.py` `FEATURES`,
`BOOL_FEATURES`, `KIND_FEATURES` and `CLASSES` equal this table):

| Body kind | Features and values |
|---|---|
| `injection` | `source` network \| local; `sink` exec \| data; `sanitizer` none \| partial; `chainConfirmed` true \| false |
| `authorization` | `entry` network \| internal; `operation` write \| read; `authCheck` none \| partial; `chainConfirmed` true \| false |
| `configuration` | `class` host-compromise \| exposed-service \| exposed-app \| hygiene |
| `secrets` | `state` head \| removed; `class` K \| P \| G; `flow` true \| false |
| `caps` (every kind) | `fileRole` primary \| example \| doc |
| file features (every kind, base/adjust/penalties/caps alike) | `fileRole` (as above); `path`, `basename`: a §1.3 matcher |

The pack (S32 encodes it; the checker of §9.3 evaluates this block):

```yaml
packFormat: 1
id: severity-rules
version: 0.1.0
kind: severity
vocabVersion: "1.0"
formula:
  caps: [{when: {fileRole: example}, max: 0}]
  injection:
    base:
      - {when: {source: network, sink: exec}, value: 3}
      - {when: {source: network, sink: data}, value: 2}
      - {when: {source: local, sink: exec}, value: 2}
      - {when: {source: local, sink: data}, value: 1}
    penalties:
      - {when: {chainConfirmed: false}, value: 1}
      - {when: {sanitizer: partial}, value: 1}
  authorization:
    base:
      - {when: {entry: network, operation: write}, value: 3}
      - {when: {entry: network, operation: read}, value: 2}
      - {when: {entry: internal, operation: write}, value: 2}
      - {when: {entry: internal, operation: read}, value: 1}
    penalties:
      - {when: {chainConfirmed: false}, value: 1}
      - {when: {authCheck: partial}, value: 1}
  configuration:
    base:
      - {when: {class: host-compromise}, value: 3}
      - {when: {class: exposed-service}, value: 2}
      - {when: {class: exposed-app}, value: 1}
      - {when: {class: hygiene}, value: 0}
    penalties:
      - {when: {basename: {regex: '(?:.*[._-])?(?:dev|local|test|override)(?:[._-].*)?'}}, value: 1}
  secrets:
    base:
      - {when: {state: head, class: K}, value: 3}
      - {when: {state: head, class: P}, value: 2}
      - {when: {state: head, class: G}, value: 1}
      - {when: {state: removed, class: K}, value: 2}
      - {when: {state: removed, class: P}, value: 1}
      - {when: {state: removed, class: G}, value: 0}
    adjust:
      - {when: {state: head, flow: true}, value: 1}
    penalties:
      - {when: {path: {regex: '(?:.*/)?(?:tests?|__tests__|e2e)/.*'}}, value: 1}
rows:
  - {id: INJ-1, finding: injection, rank: 3, level: critical}
  - {id: INJ-2, finding: injection, rank: 2, level: high}
  - {id: INJ-3, finding: injection, rank: 1, level: medium}
  - {id: INJ-4, finding: injection, rank: 0, level: low}
  - {id: AUTHZ-1, finding: authorization, rank: 3, level: critical}
  - {id: AUTHZ-2, finding: authorization, rank: 2, level: high}
  - {id: AUTHZ-3, finding: authorization, rank: 1, level: medium}
  - {id: AUTHZ-4, finding: authorization, rank: 0, level: low}
  - {id: RCFG-1, finding: configuration, rank: 3, level: critical}
  - {id: RCFG-2, finding: configuration, rank: 2, level: high}
  - {id: RCFG-3, finding: configuration, rank: 1, level: medium}
  - {id: RCFG-4, finding: configuration, rank: 0, level: low}
  - {id: SEC-1, finding: secrets, rank: 3, level: critical}
  - {id: SEC-2, finding: secrets, rank: 2, level: high}
  - {id: SEC-3, finding: secrets, rank: 1, level: medium}
  - {id: SEC-4, finding: secrets, rank: 0, level: low}
```

`formula.<kind>` is `{base: [..], adjust?: [..], penalties: [..]}`, each entry `{when, value}`;
`caps` entries are `{when, max}`; `rows` entries are `{id, finding, rank, level}` and the validator
checks `level` against `rank`, one row per (finding, rank), and the id against the derivation above
(N-18, N-24).

### 9.2 Feature definitions

- *Network-facing* entry or source: an entry point whose contract kind is `http`, `rpc`, `mcp_tool` or a
  `topic` consumer, or a `source` mark of class `network`. *Local* / *internal*: CLI arguments,
  environment, files, configuration; any non-network entry.
- `chainConfirmed`: every hop (`DATA`, `CALLS`, `Link`) on the finding's chain is `confirmed`; any
  `possible` hop makes it `false`. A finding over uncertain links ranks lower and never disappears.
- **Injection** (untrusted input reaches a dangerous operation): `sink` *exec* is shell, `eval`,
  deserialization of untrusted data; *data* is SQL, a file path, an outbound URL or template rendering.
  `sanitizer: partial` is a sanitizer on some paths but not all (a finding exists only when none
  dominates every path).
- **Missing authorization** (a path to a sensitive operation skips an `auth-check`): `operation: write`
  is any write, delete or privilege change (always a sensitive operation); `read` is a read of a
  `sensitive`-marked entity (any other read is no finding). `authCheck: partial` is a check that
  dominates some paths to the same target but not this one.
- **Risky configuration** (rules over `ConfigEntry`; findings `STEP` to the entry): *host-compromise* is
  `privileged: true`, the container runtime socket mounted, `network_mode: host` with a published admin
  port; *exposed-service* is a datastore or admin interface published on all interfaces, TLS
  verification disabled, a remote-code debug console; *exposed-app* is an application port on `0.0.0.0`,
  a debug or verbose flag, permissive CORS (`*`); *hygiene* is an unpinned `latest` image, running as
  root, no healthcheck. A dev/local/test/override variant is a **basename token**, not a substring
  (`docker-compose.latest.yml` and `contest.yml` are not variants).
- **Secrets** (present at HEAD, or committed and removed within the configured window, FR-15): a secret
  at HEAD is usable now; a removed one is usable until rotated, so it starts one lower and its advice is
  "rotate it". The value is never stored or shown (HMAC fingerprint only, proposal §3.7). Class **K** is
  a private-key block or a provider credential with a documented fixed prefix; **P** a password or token
  assigned to a name that fails the spec §1.7.3 name rule; **G** any other pattern-plus-entropy match.
  `flow: true` is a confirmed taint chain taking the value to a log, a response or an outbound call.
- Ranks are *per finding*; a finding that is worse in a later snapshot because its rank rose is the
  review delta's "worsened" (proposal §3.7).

### 9.3 Test vectors for S32 (features → rank → row)

`cpg/tests/test_rule_pack_spec.py` evaluates §9.1's block on each vector, and mutates every number,
every `when` and the order of operations, requiring at least one vector to fail for each mutant. S32
repeats both checks against `cpg/packs/severity-rules/pack.yaml` on these 44 vectors **and** the 11
test-owned vectors X1 to X11 (`EXTRA_VECTORS` in that test file; without them a narrower test-path
regex or a dropped dev token survives the 44). They stay in the test, not this table: they are
mutation-killers on the regexes, not rule claims, and S32 imports the list. Features are `name=value`; `file`
supplies `path` and `basename`; omitted `fileRole` is `primary`, omitted `chainConfirmed` is `true`.

| # | Kind | Features | Rank | Row | Exercises |
|---|---|---|---|---|---|
| V1 | injection | source=network sink=exec sanitizer=none | 3 | INJ-1 | top base |
| V2 | injection | source=network sink=data | 2 | INJ-2 | base 2 |
| V3 | injection | source=local sink=exec | 2 | INJ-2 | base 2 (local) |
| V4 | injection | source=local sink=data | 1 | INJ-3 | base 1 |
| V5 | injection | source=network sink=exec chainConfirmed=false | 2 | INJ-2 | one penalty weight 1 |
| V6 | injection | source=network sink=exec chainConfirmed=false sanitizer=partial | 1 | INJ-3 | two penalties |
| V7 | injection | source=network sink=data chainConfirmed=false sanitizer=partial | 0 | INJ-4 | to the floor |
| V8 | injection | source=local sink=data chainConfirmed=false sanitizer=partial | 0 | INJ-4 | pre-floor −1 |
| V9 | injection | source=local sink=exec sanitizer=partial | 1 | INJ-3 | sanitizer penalty |
| V10 | injection | source=network sink=exec fileRole=example | 0 | INJ-4 | example cap |
| V11 | authorization | entry=network operation=write | 3 | AUTHZ-1 | write to any entity |
| V12 | authorization | entry=network operation=read | 2 | AUTHZ-2 | base 2 |
| V13 | authorization | entry=network operation=write authCheck=partial | 2 | AUTHZ-2 | one penalty weight 1 |
| V14 | authorization | entry=internal operation=write | 2 | AUTHZ-2 | base 2 (internal) |
| V15 | authorization | entry=internal operation=read | 1 | AUTHZ-3 | base 1 |
| V16 | authorization | entry=internal operation=write authCheck=partial | 1 | AUTHZ-3 | penalty |
| V17 | authorization | entry=internal operation=read chainConfirmed=false | 0 | AUTHZ-4 | to the floor |
| V18 | authorization | entry=network operation=write fileRole=example | 0 | AUTHZ-4 | example cap |
| V43 | authorization | entry=network operation=write chainConfirmed=false | 2 | AUTHZ-2 | chain penalty weight 1 |
| V19 | configuration | class=host-compromise file=docker-compose.yml | 3 | RCFG-1 | top base |
| V20 | configuration | class=host-compromise file=docker-compose.dev.yml | 2 | RCFG-2 | dev penalty, root-level file |
| V21 | configuration | class=exposed-service file=deploy/docker-compose.yml | 2 | RCFG-2 | base 2 |
| V22 | configuration | class=exposed-service file=docker-compose.dev.yml | 1 | RCFG-3 | dev penalty |
| V23 | configuration | class=exposed-app file=docker-compose.yml | 1 | RCFG-3 | base 1 |
| V24 | configuration | class=exposed-app file=docker-compose.override.yml | 0 | RCFG-4 | dev penalty to the floor |
| V25 | configuration | class=exposed-service file=docker-compose.latest.yml | 2 | RCFG-2 | `latest` is not a dev token |
| V26 | configuration | class=hygiene file=docker-compose.yml | 0 | RCFG-4 | base 0 |
| V27 | configuration | class=host-compromise file=docker-compose.yml fileRole=example | 0 | RCFG-4 | example cap |
| V28 | secrets | state=head class=K file=src/app.py | 3 | SEC-1 | top base |
| V29 | secrets | state=removed class=K file=src/app.py | 2 | SEC-2 | removed base |
| V30 | secrets | state=removed class=G file=src/app.py | 0 | SEC-4 | removed base 0 |
| V31 | secrets | state=head class=P file=tests/test_x.py | 1 | SEC-3 | test-path penalty |
| V32 | secrets | state=head class=P flow=true file=src/app.py | 3 | SEC-1 | flow +1 |
| V33 | secrets | state=head class=G flow=true file=src/app.py | 2 | SEC-2 | flow +1 on 1 |
| V34 | secrets | state=head class=G file=src/app.py | 1 | SEC-3 | head base 1 |
| V35 | secrets | state=head class=K flow=true file=src/app.py | 3 | SEC-1 | adjustment capped at 3 |
| V36 | secrets | state=removed class=P file=src/app.py | 1 | SEC-3 | removed base 1 |
| V37 | secrets | state=head class=K file=tests/test_x.py | 2 | SEC-2 | K at head under tests |
| V38 | secrets | state=head class=P flow=true file=tests/test_x.py | 2 | SEC-2 | flow and penalty together |
| V39 | secrets | state=removed class=K flow=true file=src/app.py | 2 | SEC-2 | flow only at head |
| V40 | secrets | state=removed class=P file=tests/test_x.py | 0 | SEC-4 | removed secret under tests: floor |
| V41 | secrets | state=head class=K flow=true file=tests/test_x.py | 2 | SEC-2 | cap before penalty |
| V42 | secrets | state=head class=K file=src/app.py fileRole=example | 0 | SEC-4 | example cap |
| V44 | secrets | state=head class=K file=src/contest/app.py | 3 | SEC-1 | `test` as a substring is not a test path |

## 10. Negative examples (S31 tests each)

Each fragment is a malformed pack (or rule) that `rule-pack-v1.schema.json`, or the validator S31
writes beside it, must reject. The five paper packs of §2–§6, the infrastructure-signature pack and
the severity pack must pass (S31/S32 done-condition): **S31 extracts the pack YAML blocks from this
document mechanically** (every fenced block whose first key is `packFormat`, as S10 extracts spec3's
vectors), never by hand copy. A pack file is its block verbatim plus one added `status: paper` line (and a
leading comment); `tests/test_rule_pack_schema.py` pins the equality. Where the check needs the registry (§1.9) or a sample, it is the
validator's, not pure JSON Schema.

| # | Malformed | Clause violated |
|---|---|---|
| N-1 | `packFormat: 2`, or the key absent | §1.1 |
| N-2 | `id: FastAPI` (uppercase), or an `id` not equal to its directory name | §1.1 |
| N-3 | `kind: plugin` | §1.1 `kind` enum |
| N-4 | a rule with `emit: []` | §1.2: at least one emission |
| N-5 | `match: {decorator: {…}, callsite: {…}}` | §1.2: exactly one subject |
| N-6 | `regex: '(?=x)abc'`, or `regex: '(a)\1'` | §1.3: no lookaround, no backreference |
| N-7 | a `regex` of 257 characters | §1.3: at most 256 |
| N-8 | `name: {exact: x, glob: y}` | §1.3: one matcher per field |
| N-9 | `provides: {kind: http, key: "…", reason: looks-fine}` | §1.5: `reason` not in the spec §7 registry (a pack cannot invent a code) |
| N-10 | `provides: {kind: widget, key: "{x}"}` with no `vocab` entry for `widget` | §1.5: kind is neither core nor declared |
| N-11 | `vocab: [{kind: widget, example: "w1"}]` (no `keyNorm` rule) | §1.6: a new kind states its `keyNorm` rule |
| N-12 | `extract: {t: {arg: arg0}}`, `{arg: "a-1"}` or `{arg: p0}` in a decorator/callsite rule | spec §5.1 grammar; `p<i>` is a flow head, not an argument |
| N-13 | two rules with `id: route` | §1.2: rule ids unique in the pack |
| N-14 | a query with a write key, a record type `secretSighting`, five `from` bindings, a join on a non-id field, a binding after the first with no join equality, or no `handler` column when a rule anchors on it | §1.7 |
| N-15 | a `liftings` seed with no `arg` | §1.6 |
| N-16 | a `mounts` entry with `prefix` and no `child` | §1.6 |
| N-17 | `provides` with `status: confirmed` whose key slot has `onUnknown: partial`; or `status: confirmed, reason: partial-key` | §1.4, §1.5: status must agree with resolution and with the code's registry status |
| N-18 | a severity row with `level: severe` or `rank: 4` | §9.1: four levels |
| N-19 | `transport: Kafka`, or `transport: inhouse-other` in the pack `paper-inhouse-bus` | §1.6: lowercase token; `inhouse-<packId>` uses the pack's own id |
| N-20 | `extract: {v: {config: value}}` | §1.2: `config` has only `keyPath` and `bindValue`; a pack cannot read an ungated value |
| N-21 | a top-level `repoId:` (or any key not in §1.1) | §1.1: unknown keys rejected; packs name no repository |
| N-22 | a `signatures` entry with `class: latency`, or whose `sample` its own `regex` does not match, or `sampleSource: guessed`, or whose `regex` matches a shared or near-miss negative, or a regex edit in §8.2's mutation list that no negative rejects | §8.1, §8.2 |
| N-23 | `link: [{kind: http, matcher: fuzzy}]` | §1.3/§1.6: matcher set is `exact`, `path-template`, `glob`, `prefix`, `regex` |
| N-24 | a severity row whose `level` disagrees with its `rank` (`rank: 3, level: low`), a second row for one (finding, rank), or an `id` other than the derived one | §9.1 |
| N-25 | a matcher `{basename: x, exact: y}`; `urlBindings` with a `scope` outside a `callsite` source; `targetArg: {form: dotted}`; `envPorts: {adoptedBy: any}` | §1.3, §1.6: one matcher; closed enums |
| N-26 | **generated probe:** for every object the schema defines, one planted unknown key; for every enum and every typed leaf, one out-of-range value | §1.1: unknown keys rejected everywhere (schema-walking, not an enumerated list) |
| N-27 | `onUnknown: config-derived` on a `provides` of a kind other than `topic`, or on a `consumes` of kind `http`; a `reason` whose registry `On` set lacks the emission's edge and kind (`reason: config-derived` on `PROVIDES http`) | §1.4, §1.9 |
| N-28 | a `firstOf` with no entries, or an empty `anyOf`; a rule on `match: {query: q}` where `q` has no `handler` column | §1.2, §1.7 |
| N-29 | a slot whose nested `firstOf` expressions carry different `onUnknown` modes (`[drop, partial]` or `[partial, drop]`), with or without `status: confirmed` | §1.4: a slot has one mode |

Positive controls beside them: the three packs of §2–§4 are valid, and a variation of each with one
rule removed is valid (`rules` is optional for every kind, §1.1; `fetch`'s variation keeps its
`liftings`). The checker of `cpg/tests/test_rule_pack_spec.py` already validates every pack block of
this document against §1's key and enum tables, as the proof that the format is closed.

## 11. Contact list: proving-repo anchors the walkthroughs touch

By eval-protocol §8.2 (S30 row): every proving-repo **anchor whose link a walkthrough derives**: each
route, each MCP tool, each server unit whose URL or launch binding is walked. Wrapper and chain hops
are excluded. Locators are in bundle spec §3.5/§1.9 form, `repoId` `graphmind-ai-lab`; the
walkthroughs read the repository at `defca60ff16534aa12a74e35856de9be98bfc710` (the S14 pin).

**Contact anchors (S30's list, the input to S47's contact set):**

| # | Walkthrough | Anchor | Locator |
|---|---|---|---|
| 1 | §2.1, §3.1 | route | `{kind: "http", keyNorm: "GET /shop/api/health"}` |
| 2 | §2.2 | route | `{kind: "http", keyNorm: "GET /health"}` |
| 3 | §4.1, §4.4 | MCP tool | `{kind: "mcp_tool", keyNorm: "send_message"}` |
| 4 | §4.3, §4.4 | server unit (URL and launch binding walked) | `{repoId: "graphmind-ai-lab", rootPath: "falkor-chat/server", lang: "python"}` |

Rows 1 and 2 are also in S14's list (slice report §4, #1 and #2); S30 adds **two** anchors, 3 and 4.
Row 4's unit is dev already under the served-anchor rule once S14's routes are contact (eval-protocol
§15 Q10), so walking it costs `wiring` nothing; per eval-protocol §8.2 a walked unit also contacts its server-level
wiring items, which GC derives and this document does not enumerate.

**Named by a walkthrough, not anchors (informative; hops and mechanism excluded):**
- Provider code elements: `falkor-chat/server/falkorchat/storefront_api.py::build_storefront_router.health`,
  `…/api.py::build_router.health`, `…/app.py::create_app.health`, `…/mcp.py::send_message`
  (slice report §4 treats providers as context, not extra anchors).
- Consumer element: `salesperson/src/api/endpoints.ts::getHealth`.
- Wiring item of row 4: `{repoId: "graphmind-ai-lab", path: ".mcp.json", keyPath: "mcpServers.falkor-chat-agent-team"}`
  (eval-protocol §2.4's grain for `.mcp.json`). It is not an *anchor* of this list, but it **is** one of
  the walked unit's server-level wiring items that eval-protocol §8.2/§9.1 contact (it is one of
  `falkor-chat/server`'s four dev entries); listing it here as informative is no ruling for S47.
- **Hops, excluded by eval-protocol §8.2:** `salesperson/src/api/client.ts::apiFetch` and `::buildUrl` as hops, the
  `fetch` call (`client.ts:118`), and the mount calls (`app.py:485`, `:521`, `:543`).
- Launch records of §4.3: a `launchFact` has no locator in v1 (spec §1.9), and Dockerfile items are out
  of census scope (eval-protocol §2.2).

**Not walked, so not contact:** `cypher-mcp` and every `.mcp.json` entry but one (R9, §4); `mcp-monitor`;
the Kiro agent; every other route and tool of `falkor-chat/server` (the 61 providers, the other 24 tools,
the 10 other links); the participant-wrapper chains. The Kafka and in-house-bus packs and the §4.4
client snippet are invented and name no proving-repo anchor.

## 12. Downstream obligations and open items

### 12.1 Obligations

| Owner | Takes from this document |
|---|---|
| S31 (`tdd-engineer`) | `cpg/ccg/schemas/rule-pack-v1.schema.json` and `cpg/tests/test_rule_pack_schema.py` from §1 and §10 (each N-n rejected); **`cpg/ccg/registry/reasons.yaml`** (§1.9) and its parse-from-spec test; `cpg/packs/{fastapi,fetch,fastmcp}/pack.yaml` extracted mechanically from the YAML blocks of §2, §3, §4 (§10 preamble). The schema file carries the RE2-subset and registry checks as validator code beside the JSON Schema. **p0-2 §3's S31 row gains the registry file** (a `teco` edit). |
| S32 (`tdd-engineer`) | `kafka` (§5) and `paper-inhouse-bus` (§6) extracted, asserting the literal-key topic facts only until the preconditions of §5 (O-15, O-16) are ruled; `infra-signatures` from §8.1–§8.2 (the `signatures` list is the file's body; §8's negatives, cross-runtime and mixed-blob cases are its tests, with the mutation check); `severity-rules` from §9.1's block, passing §9.3's 44 vectors plus the test's `EXTRA_VECTORS` (X1 to X11) and the mutation check. |
| P1 pack runtime | a conformance test: run `fastapi`, `fetch` and `fastmcp` over `s14a` and compare to `assemble.py`'s `providers` (61) and consumers (11) and their hops. Expected differences: the two handlers of §2.3 (`inner.get`, `_probe_rest_app`'s `router.post`) are `possible/partial-key`, not `confirmed`. Fixtures: a factory with `APIRouter(prefix=…)` mounted with another prefix, and a function holding `FastAPI()` plus a local `APIRouter()`, must give `possible`. The `fetch` lifting emits only resolved leaves (§1.6); `assemble.py` emits every resolved slot (O-13). **`fastmcp` row:** the `tool` rule over `s14a` yields `PROVIDES mcp_tool`, `confirmed/key-match`, for the 25 `mcp.py` decorators (§4.1) and for the look-alikes O-8 names, each listed rather than silently included; the `tool-call` rule yields `CONSUMES mcp_tool`. S14 did not run the launch/config extractor, so this run is its first exercise: over §4.3's records it must give the `(8200, unit)` port, the `/mcp` mount path and the `confirmed` `Link` of §4.4. The 14 `fastmcp:` leaves of `UNEXERCISED_UNTIL_P1` in `cpg/tests/test_rule_pack_leaf_mutation.py` (import gates, callee names, the tool-call key, launch defaults and flags, URL-source patterns) belong to this row: each must flip a result when mutated. |
| S47 (GC) | §11 as S30's contact list. |
| S27 | §7(c). |

### 12.2 Open items

- **O-3.** A router held in a module variable and mounted by `include_router(router)` has no record
  linking the variable to its decorators (§2.3, §1.8). Unsupported in v1 (`possible`); a `-spec5`
  field or a P1 mechanism would close it.
- **O-5.** FastMCP's inner `streamable_http_path` override (`mcp.py:64`) is an attribute write whose
  value no record carries, so MCP path matching is `prefix`-only (§4.2).
- **O-7.** Parameter-level annotations are not records (§7).
- **O-8.** No receiver text, assigned-variable name or resolution on decorators, and no `isTest`
  flag (slice report §6.1): `mcp.tool` and test-file `stub_server.tool` look alike (§4.1).
- **O-9.** Which reason a `mcp_tool` `Link` carries when a binding selects the unit (`key-match`, as
  written in §4.4, or `binding-confirmed`) is the linker's. A framework-default port has no
  graph element to hang evidence on (`BOUND_BY` targets a `ConfigEntry`; a `pack-default` origin is
  a pack fact): S17/S33 decide the representation.
- **O-10.** Where the identity of the pack set (ids, versions, hashes) is recorded in a snapshot is
  not decided here; determinism (§1.8) needs it recorded for provenance. Owner: S17/S33.
- **O-12.** S14's own Q1 (does "touched" cover consumers and scale-probe anchors?) is open for S15/GC.
  §11 follows eval-protocol §8.2's text (anchors only; the consumer `getHealth` is an impacted element). If GC rules
  consumers are anchors, §11 gains one row for `getHealth`.
- **O-13.** Differences from the slice recognizers, for P1's conformance test to confirm: the
  `scope.imports` gate (§2) and the `fetch` seed's `resolution` requirement (§3) are equal on S14; the
  `fetch` leaf-only emission (§1.6) is equal on S14 (no resolved slot with a lifted child) and differs
  from `assemble.py` on a partially resolved intermediate slot.
- **O-14.** `security-expert` has not advised on §9 (optional, not a gate).
- **O-15.** **Precondition for S32's topic assertions.** Graph model v1 has no home for a pattern key
  (`keyForm: glob|regex`: D4's `ContractPoint` has only `key`, `keyNorm`) or a keyless `CONSUMES`
  (§5.2's `config-derived`); and the `Finding` label is deferred in the graph model, so nothing records
  a Finding carrying two reasons (`severity-rule:<id>` plus `example-config`, §9.1; proposed: a reasons list,
  the finding's status the more uncertain reason's). Owner: `graph-dba` shapes both, with S17.
- **O-16.** **Decided** (`teco` ruling, 2026-11-01): a publisher with an unresolved topic (§5) gets a
  registry code by adding `PROVIDES topic` to the `On` sets of `runtime-built` and `config-derived` as
  §1.9 registry cells; spec3 §7.2 stays frozen. Shipped in `reasons.yaml`; §1.4, N-27 and the spec
  test's `check_rule` agree (v1.4).
  Related, owner ML §8.3: whether `connection-refused` should split into `connection-lost` (§8.2).
