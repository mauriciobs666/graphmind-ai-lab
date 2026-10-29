# Connected code graph — fact-bundle spec (v1 candidate)

> **Status:** active · **Owner:** `architect` · **Tracks:** C-1001 (M10) · **Extends:** `docs/plans/connected-code-graph.md` · **Version:** 1.3 (v1 candidate) · **Reviews:** `docs/reviews/connected-code-graph-p0-impl.md`

*Revision 2026-10-29 (v1.1): S03 gate fixes (analyst F1–F8, graph-dba suggestions and rulings); dispositions in §13.1.*
*Revision 2026-10-29 (v1.2): S03 Pass 2 (N1–N5, P2-n1, P2-n2) and the S05 vendored-code question; dispositions in §13.2.*
*Revision 2026-10-29 (v1.3): S03 Pass 3 (P3-1–P3-4, P3-n1, P3-n2), final static round; dispositions in §13.3.*

This is step S03 (unit P0-03a) of `docs/plans/connected-code-graph-p0.md`.

## 0. Purpose and scope

A **fact bundle** is the output of one build of one unit (one repository, root and language). It
is the only thing the assembler reads about that unit's code (proposal §3.1, §3.3). This spec
fixes:
- the bundle's layout;
- every record type and field;
- the canonical tuples and the id hash;
- the source-derivable `qname` grammar for every FR-1 language;
- the proving repo's `repoId`;
- the `ContractPoint (kind, keyNorm)` tuple and its `keyNorm` rule;
- the reason-code registry;
- the determinism rules;
- the mapping from records to the graph model.

**Readers:**
- S09 implements `cpg/ccg/ids.py` from §3 and must reproduce the golden vectors in §3.6.
- S10 implements `cpg/ccg/schemas/bundle-v1.schema.json` and `cpg/ccg/bundle.py` from §2, §5 and §8.
- S11 emits bundles to this spec.
- S35/S39 write fixture locators in §3.5's form.
- S06/S40 and the headless key writers write key and gold locators in §3.5's form. They work from
  source alone (§4).
- S17 states graph model v1 against §9.

**Out of scope:**
- Assembly outputs: links, lifted roles, contract points, findings and marks. §6 and §3.4 fix only
  the **ids** of the assembly outputs that gold and keys refer to.
- The rule-pack format (S30).
- The answer schema (S33).
- Coverage-importer inputs (proposal §3.3, extension-unit kind "importer"). These are separate
  inputs, not bundle records.

**Authority.** Labels and edges follow the graph note (`docs/plans/connected-code-graph-graph.md`
§3.2, as amended by its §6, which wins where they differ). Where the graph note and the proposal
disagree, this spec follows the P0 plan's §11 findings, F1 and F2 in particular. Each new
disagreement is a numbered finding in §12. None is resolved silently.

**Provisional marks.** **[P-slice]** marks a rule the vertical slice (S11–S14) is expected to
exercise. A mismatch the slice finds is an SL-7 delta. **[P-later]** marks a rule the slice
cannot exercise (Java, Kotlin, C#, Go, C/C++, and P2+ record types). §10 lists both kinds.

## 1. Freeze condition (P0 plan §2 D5)

- **This document is bundle schema v1 (candidate).** It freezes under the proposal's binary rule
  (proposal §4 P0(b)): **v1 freezes only if the vertical slice passes, meaning all of S14's
  SL-1–SL-7.** S15 decides.
- **If S15 passes v1 with no SL-7 delta,** the freeze is recorded by S15's gate (the slice report,
  the ledger and `docs/HISTORY.md`). This document is **not edited**; its `Status:` stays `active`
  until milestone close.
- **If the slice forces any change** (an SL-7 delta, including a `qname` normalizer mismatch,
  §4.0), this document has been executed against. Collision rule 5 then requires a successor,
  `docs/plans/connected-code-graph-bundle-spec2.md` with `Supersedes:` this document, written in
  S16. S10's schema follows the successor.
- **If any SL criterion fails,** v1 stays **provisional until P1 exit**, and the P1 plan carries
  the freeze. Outputs that cite this spec (S17, S28, S30, S33, S34) are then marked "against
  provisional v1" and re-checked at P1 exit.
- **A partial freeze is not available** (D5). Any wish for one goes to the stakeholder as
  SQ-P0-2.
- **A freeze covers [P-later] rules too, untested.** A defect found in them at P2 or later is
  handled by §8.6's versioning policy: additive changes bump the minor version, breaking ones need
  a successor document.

## 2. Bundle envelope and file layout

### 2.1 Files

- **One bundle per unit build.** It is one UTF-8 JSONL file named `<unitId>.bundle.jsonl`. The
  JSONL rules are in §8.
- **Line 1 is the `header` record.** Every later line is one record of a type from §5.
- **Record order** is the fixed type order of §8.3. Within a type, records are sorted by their
  `id` (§3), as byte strings.
- **No other files are part of the bundle.** `cpg.bin`, logs and timings live beside it in the
  build's work area and are not covered by this spec.

### 2.2 The `header` record

| Field | Type | Req | Meaning |
|---|---|---|---|
| `type` | `"header"` | yes | |
| `bundleSchemaVersion` | string `"1.0"` | yes | §8.6 |
| `unit` | object `{repoId, rootPath, lang}` | yes | the unit tuple (§3.2). `rootPath` is repo-relative POSIX with no trailing `/`; the repo root is `"."`. `lang` is a token from §3.3. |
| `unitId` | 32-hex | yes | `h(["unit", repoId, rootPath, lang])` |
| `builtCommit` | 40-hex | yes | commit the source was staged from (`git archive`, P0 §4.4) |
| `builtTree` | 40-hex | yes | `git rev-parse <builtCommit>:<rootPath>`, the root's tree id |
| `builtAt` | string, RFC 3339 UTC, `Z`, second precision | yes | **excluded from determinism** (§8.5) |
| `tools` | object `{joern, frontend, frontendVersion, normalizer, normalizerVersion, extractScript, extractScriptHash}` | yes | All strings. `extractScriptHash` is the SHA-256 of `extract.sc`. Non-Joern extractors put their name and version in `frontend`/`frontendVersion`. |
| `sandboxAuditIds` | array of lowercase UUIDv4, at least 1, in invocation order | yes | Each element is **the `auditId` of one `sandbox` record** of the egress spec (`docs/plans/connected-code-graph-egress-spec.md` §5.1, §5.3; its risk R6). There is one per `offline-run.sh` invocation that produced this bundle's inputs, for example the Joern parse and the `extract.sc` run. Each must have `refused: false`, `exitCode: 0`, and `egressMode` equal to this header's. A bundle naming an id with no such record is invalid (egress §5.3's P1 loader check). An array, not a scalar: finding BF15. **Excluded from determinism** (§8.5). |
| `egressMode` | `"offline"` \| `"controlled"` | yes | proposal §3.12. Same enum as the egress spec §5.3 `sandbox.egressMode`, and equal to it on every referenced record. |
| `flowProjected` | bool | yes | whether the per-method projection ran (`data`/`controlledBy` present). `Unit.flowAvailable` is decided per language from P0's measurements, not here. |
| `fieldDepthCap` | int, `3` | yes | the access-path cap used (§5.6) |
| `vendoredDirs` | array of strings, sorted, possibly empty | yes | the effective vendored-directory list of this unit's repo, applied below its `rootPath` (§5.0 "Vendored paths"): the default list, or the manifest's per-repo replacement. Provenance, so an excluded file is explainable. |
| `history` | `"available"` \| `"unavailable"` | yes | whether `lastChanged*` was computed (§5.3, finding BF12). **In P0 it must be `available`:** a shallow clone or missing history is a build error, not a silent `unavailable` (analyst F8(c)). P1 decides whether `unavailable` becomes legal; if it does, the clone's history depth is a listed input (§8.4). |
| `counts` | object `{<recordType>: int}` | yes | the number of records per type, every §8.3 type present, zero included. A cheap truncation check. |

## 3. Identity: tuples, the hash, `repoId`, locators

### 3.1 The hash (P0 plan §2 D10, made exact)

```
id(tuple) = lowercase-hex( SHA-256( canonical-json( normalize(tuple) ) ) )[0:32]
```

- **The tuple** is a JSON array. **Its first element is always the record-type tag** from §3.2.
  The tag is not decoration: without it a call site `(symbolId, "param", 0)` and a parameter point
  `(symbolId, "param", 0)` hash equal (finding BF1).
- **`normalize`** applies to every string element, recursively: Unicode **NFC** first, then
  strip. Strip removes leading and trailing characters for which Python 3.12's `str.isspace()` is
  true; `str.strip()` with no argument does exactly this. A non-Python implementation must use
  the same character set. Integers are unchanged. Booleans, floats and `null` **never** appear in
  a tuple.
- **`canonical-json`** is UTF-8 with no BOM. Non-ASCII is written literally, not `\u`-escaped.
  The separators are `,` and `:` with no whitespace, and keys are sorted (tuples have no objects;
  the rule is stated for completeness). In Python:
  `json.dumps(t, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")`.
- **Optional tuple elements are omitted, never `null`.** A Python symbol's tuple has 4 elements;
  a Java symbol's has 5.
- **Input length is unbounded.** That is the reason for hashing: a `UNIQUE` property over 4,096
  bytes, or one with edge whitespace, crashes the instance (graph note §3.4).

### 3.2 Canonical tuples

`h(...)` is §3.1's function. **Bundle** ids are carried in bundle records. **Assembly** ids are
computed by the assembler and fixed here because keys, gold or verdicts refer to them.

| Id | Tuple | Where | Notes |
|---|---|---|---|
| `unitId` | `["unit", repoId, rootPath, lang]` | bundle | |
| `fileId` | `["file", repoId, path]` | bundle | `path` is repo-relative POSIX (§8.2) |
| `symbolId` | `["symbol", lang, repoId, qname]`, plus `signature` as a 5th element **only** when `lang` ∈ {`java`, `kotlin`, `csharp`, `cpp`} **and** the symbol is function-like (§4.3) | bundle | **No `unitId`** (graph note §6.4 m2; P0 F2). `lang` is the **file's** language token (§3.3). |
| `callSiteId` | `["callsite", symbolId, calleeName, ordinal]` | bundle | `ordinal` is 0-based, per (enclosing symbol, `calleeName`), in source order (§5.7). Survives line moves (graph note §3.4). |
| `pointId` | `["point", symbolId, kind, n]` | bundle | `kind` ∈ {`param`, `ret`, `cond`}. `n` is the parameter `index` for `param` (the receiver is `-1`, §5.4), and the 0-based source-order ordinal among that symbol's points of that kind for `ret` and `cond`. A call site's point id **is** its `callSiteId`. |
| `constantId` | `["constant", lang, repoId, qname]` | bundle | §5.9 |
| `entryId` (config) | `["configEntry", repoId, path, keyPath]` | bundle | No line, so it survives line moves. |
| `entryId` (launch) | `["launchFact", repoId, path, kind, ordinal]` | bundle | `ordinal` is 0-based per (path, kind), in source order |
| `entryId` (schema) | `["schemaEntry", repoId, path, objectKey]` | bundle | `objectKey` is the `keyNorm` of the defined object (§6) |
| `depId` | `["dependency", ecosystem, package]` | bundle | Repository-neutral (graph note §3.2 `Dependency`). The normalization is in §5.15. |
| `nameKey` | `["name", lang, calleeName]` | assembly | graph-dba's BF14 ruling. The readable `lang`/`name` stay unconstrained on `Name`. |
| record ids with no node | `["<type>", …]` as in each record's §5 entry | bundle | `data`, `controlledBy`, `valueSet`, `decorator`, `fieldAccess`, `import`, `testMarker`. Used only for sorting and deduplication. |
| `cpId` | `["contractPoint", kind, keyNorm]` | assembly | §6. **The tuple gold and keys use for contract anchors** (P0 D8). |
| `linkId` | `["link", kind, keyNorm, providerAnchorId, consumerAnchorId]` | assembly | An anchor id is the enclosing `symbolId`, or the `entryId` for a `ConfigEntry` end (proposal §3.8, generalized; finding BF6). |
| `sightingId` | `["secretSighting", repoId, path, commit, fingerprint]` | bundle, reserved (P5) | §5.19 |

`findingId`, `markId`, `templateId` and `runId` belong to P3/P5 and are not fixed here. The
proposal's §3.8 end-tuple rule for `findingId` stands.

### 3.3 Language tokens

| FR-1 language | `lang` token | Which files |
|---|---|---|
| Python | `python` | `.py`, `.pyi` |
| TypeScript / JavaScript | `javascript` | `.ts` `.tsx` `.mts` `.cts` `.js` `.jsx` `.mjs` `.cjs`. TS and JS are one token: a TS call into JS in the same unit is a real call. |
| Java | `java` | `.java` |
| Kotlin | `kotlin` | `.kt`, `.kts` |
| C#/.NET | `csharp` | `.cs` |
| Go | `go` | `.go` |
| C | `c` | `.c`; and a `.h` the C++ test below does not mark |
| C++ (Arduino `.ino` included) | `cpp` | `.cpp` `.cc` `.cxx` `.c++` `.hpp` `.hh` `.hxx` `.ino`; and a `.h` that, **by its own text**, contains the token `::` or one of the keywords `class`, `namespace`, `template`, `typename`, `operator`, `virtual`, `explicit`, `friend`, `mutable`, `noexcept`, `decltype`, `using`, `public:`, `private:`, `protected:`, or `extern` followed by the string literal `"C"` or `"C++"`. The test ignores comments, string literals other than that one, the contents of `[[…]]` attribute brackets (so C23's `[[gnu::always_inline]]` is not C++), and every arm of any `#if`/`#ifdef`/`#ifndef` group any of whose `#if…`/`#elif` conditions mentions `__cplusplus` (`#if defined(__cplusplus)` and the `#else` of `#ifndef __cplusplus` included). Known limit in §3.3's note. |
| shell | `shell` | `.sh`, `.bash`, or an extensionless file with a `sh`/`bash`/`zsh` shebang |
| structured config | `config` | JSON, YAML, TOML, Compose, `.env`, `Dockerfile`, `.mcp.json`, Kiro agent JSON. These produce `configEntry`/`launchFact` records, never symbols. |
| Markdown prompts | `markdown` | `.md` (the prompt-mention extractor) |

A unit's `lang` is its build manifest's language (`cpp` for a mixed C/C++ unit). A header's language depends only on its own text, so adding a `.cpp` file to a unit re-ids no header symbol (analyst F4). **[P-later]** A symbol's
`lang` is its **file's** token, so a `.c` file in a `cpp` unit yields `c` symbols with no
signature. That matches the proposal: C is not an overloading language (proposal §3.8).

**Known limit of the `.h` test** (analyst N3, v1.2). It is a token test, not a parse, so a C++
header with none of the markers (only free functions, using references or overloads) is read as
`c`. Its functions then carry no signature, and its overloads collapse into rule-7 duplicates
(`clamp`, `clamp#2`), which re-bind on reordering (BF16). The ids stay deterministic. A
parse-based test (reference parameters, same-name functions) was rejected for v1: neither
signal is a single token, and same-name C functions in `#if`/`#else` arms would trip it. The
markers added in v1.2 are C++-reserved words that are not C23 keywords (`constexpr`, `nullptr`
and `bool` are, so they are absent). P2 measures the misclassification rate on its C/C++ corpora.

### 3.4 `repoId`

- **The proving repo's `repoId` is the literal `graphmind-ai-lab`.** It is fixed. It does not
  depend on the clone directory, the remote URL, the host or the user.
- **Any other repo:** the repository name the system manifest gives it, normalized:
  1. NFC;
  2. strip;
  3. lowercase;
  4. replace each run of characters outside `[a-z0-9._-]` with `-`;
  5. trim leading and trailing `-`.

  An empty result is a manifest error. Two repos of one system with the same normalized name are
  also a manifest error. Example: `"Online Boutique"` → `online-boutique`.
- **P0's fixture repos** are `fx-svc`, `fx-web` and `fx-lib`, their directory names under
  `cpg/fixtures/system1/` (S35–S39).
- **`repoId` is repository-neutral identity for keys.** A clone moving on disk changes nothing.
  Renaming the repo in the manifest changes every id in it, as intended.

### 3.5 Locator form (keys, gold, fixture manifests)

**This is the form S06, S35, S39 and S40 use** (P0 D8, M2). Everything in it is written from
source at a commit, with no snapshot.

| Anchor | Locator | Id it converts to |
|---|---|---|
| a code element | `{repoId, path, qname, signature?, line?}` | `symbolId` = `h(["symbol", lang(path), repoId, qname(, signature)])`. `lang` comes from `path` by §3.3: the extension, the shebang for an extensionless shell script, and the file's own content for `.h`. `signature` is required exactly when §4.3 says the symbol carries one. |
| a contract point (route, MCP tool, topic, table, label, …) | `{kind, keyNorm}` | `cpId` = `h(["contractPoint", kind, keyNorm])` |

- **`path` is redundant with `qname`'s prefix on purpose** (§4.1). A locator whose `qname` does not
  start with `path + "::"` is invalid. That makes S39's check "the qname is found in the file"
  mechanical.
- **`line` is informative.** It is not identity: a line inside the symbol's span or one of its
  owned regions (§4.1 rule 4), used for human checking and file+line anchors. It is 1-based (§8.2).
- **Every locator is at the commit the artifact pins:** `EVAL_COMMIT` for ground truth, and the
  materializer's commit for fixtures.
- **The enclosing-symbol rule.** An impact item that is not itself a symbol (a call site, a
  condition) is located by its **enclosing symbol's** locator plus `line`. That is the "(repo path,
  enclosing qname)" locator the P0 plan's §4.3 matching rule names.
- **The ancestor-plus-line form (analyst F3).** An `<anon>#N` scope, or a `#k` duplicate, may
  be located without counting: `{repoId, path, within, line, nth?}`. `within` is the `qname` of
  its nearest **named, non-duplicate** ancestor, **taken by §4.1 rule 2's evaluation scope**, not
  by text, or `path + "::"` for top level. So the decorator lambda at
  `falkor-chat/server/tests/test_fusion.py:241` has `within` = `…/test_fusion.py::`, although its
  line lies inside the decorated test function's span (analyst P2-n1). `line` is the line of the
  scope's **introducing token**: `=>`, `lambda`, `function`, `def`, `func` or `fun`, or, for a
  definition with no such keyword (a Java/C#/C++ method, a class), its name identifier. `nth` is
  the 1-based rank **by that token's column** (in code points) among the matching candidates
  whose token is on that line, and is needed only when there are two or more.
  - **Conversion is mechanical.** The converter parses the file at the pinned commit, takes the
    anonymous or duplicate scopes whose introducing token is on `line` and whose nearest named
    non-duplicate ancestor (by evaluation scope) is `within`, picks the `nth`, and computes its `qname` by §4.1. Then it hashes it as above. Zero
    candidates, or an ambiguous pick with `nth` absent, makes the locator invalid.
  - **This form exists so either S06 decision works.** S06 (`data-scientist`) decides whether
    anonymous scopes are eligible as key and gold anchors at all (coordinator ruling,
    2026-10-29). If they are, writers use this form, never a hand-counted `#N`.

### 3.6 Golden vectors (S09 must reproduce)

Computed by this spec's author with §3.1's Python one-liner (Python 3.12.3). All 22 were recomputed for v1.2 on 2026-10-29 with an independent script; GV-1–GV-21 are unchanged from v1.1, and GV-22 is new. Each row
gives the tuple **before** normalization; the canonical bytes are those of the normalized tuple.

| # | Tuple as given | id | Tests |
|---|---|---|---|
| GV-1 | `["unit","graphmind-ai-lab","falkor-chat/server","python"]` | `0fb899742d4d3687495c30357ca16253` | unit |
| GV-2 | `["file","graphmind-ai-lab","falkor-chat/server/falkorchat/storefront_api.py"]` | `a52ba81a97ade6b58f76e1469662bc6a` | file |
| GV-3 | `["symbol","python","graphmind-ai-lab","falkor-chat/server/falkorchat/storefront_api.py::build_storefront_router.health"]` | `a6c2648709f43a0bad2211a1b8bbcd52` | the SL-1 provider symbol |
| GV-4 | GV-3 with every string whitespace-edged: `["symbol","  python\t","graphmind-ai-lab ","\n<GV-3 qname>  "]` | `a6c2648709f43a0bad2211a1b8bbcd52` | **equals GV-3** (strip) |
| GV-5 | `["symbol","python","graphmind-ai-lab","pkg/café.py::café"]` with `é` as U+00E9 (NFC) | `e600f3985246448b452893a572028ae9` | non-ASCII, written literally |
| GV-6 | GV-5 with `é` as `e` + U+0301 (NFD) | `e600f3985246448b452893a572028ae9` | **equals GV-5** (NFC) |
| GV-7 | `["symbol","java","fx-lib","src/main/java/com/acme/fxlib/Pricing.java::Pricing.quote","(int,String)"]` | `6b2ed6dbf5678679437067c6d3b079a8` | 5-element tuple (overloading language) |
| GV-8 | GV-7 with signature `(int,String,Currency)` | `7543f67cff9b93877688af1475d91d3c` | an overload is a different id |
| GV-9 | `["contractPoint","http","GET /shop/api/health"]` | `a359de518acc3d443ff5efa5597f458f` | the SL-1 contract point |
| GV-10 | `["contractPoint","http","GET /" + "a"×5000]` (5,032 canonical bytes) | `329a43a0d3c0223e5a0deadb23275afa` | input over 4,096 bytes |
| GV-11 | `["symbol","javascript","graphmind-ai-lab","salesperson/src/api/client.ts::apiFetch"]` | `3e0e2e00a2cea8e99fd32ff6c2b1082e` | TS symbol |
| GV-12 | `["callsite","3e0e2e00a2cea8e99fd32ff6c2b1082e","fetch",0]` | `41d8c78a0fa5b76642dc74e73958888a` | integer element; `apiFetch`'s `fetch` call |
| GV-13 | `["point","3e0e2e00a2cea8e99fd32ff6c2b1082e","param",0]` | `3584bd914767bc6ec2cd94a767006185` | `apiFetch`'s `p0` |
| GV-14 | `["callsite","8ae57f349ffed0533e458ee5341f2936","apiFetch",0]`, the first element being `h(["symbol","javascript","graphmind-ai-lab","salesperson/src/api/endpoints.ts::getHealth"])` = `8ae57f349ffed0533e458ee5341f2936` | `ac74017feb936c98cb9fd4bfbe5737e8` | the SL-1 consumer call site |
| GV-15 | `["symbol","c","fw","src/util.c::crc8"]` | `493fb56c61dd5e583f5f5221a0e792fa` | a C symbol in a `cpp` unit: **4 elements, no signature** (BF11, analyst F7) |
| GV-16 | `["symbol","kotlin","fx-svc-kt","src/main/kotlin/acme/Orders.kt::slug","String.()"]` | `2f7967620eecf347b542cf76b466c3f5` | Kotlin extension, receiver in the signature (§4.3) |
| GV-17 | `["point","3177d8d58a31424b59c4c7bd458aa801","param",-1]`, the first element being `h(["symbol","python","graphmind-ai-lab","falkor-chat/server/falkorchat/storefront_api.py::StorefrontHTTPError.__init__"])` | `73397835753cc740b7d9ac82faa84ec2` | the receiver point `-1` (§5.4): a negative integer |
| GV-18 | GV-17 with the last element as the **string** `" -1 "` | `0eb4a24f0b98b6081be51a3fbf070305` | **differs from GV-17**: strings are stripped, integers are not strings |
| GV-19 | `["name","javascript","fetch"]` | `541e72d540d11b7c8647528766e38e77` | `nameKey` (§3.2, BF14) |
| GV-20 | `["fieldAccess","79f8675f500467a9a3d2871ea327ed78","recv","read","falkor-chat/server/falkorchat/tools.py::McpToolClient","_loop"]`. The call site is `h(["callsite",S,"call_soon_threadsafe",1])` with `S` = `h(["symbol","python","graphmind-ai-lab","falkor-chat/server/falkorchat/tools.py::McpToolClient.close"])` = `e308e9b40866f15eb9479730c3d2d5fa`: `tools.py:1480`, the second `call_soon_threadsafe` in `close`. | `468fc6cbf90b5f0160ad35d9474afd52` | analyst F1's case: `self._loop` read at `recv` |
| GV-21 | GV-20 with `"a0"` for `"recv"` | `17e9de61ca6bca69fdd5738fbef102a5` | **differs from GV-20**: the same field read at another path on one point |
| GV-22 | `["symbol","go","fx-svc-go","svc/store.go::Store.Get"]` | `afa8ae88b74f3b469133792e4c28b650` | a Go interface method spec, `IfaceType.NAME`, 4 elements (§4.1 rule 8, §4.2 Go; analyst N5). A call through an embedding interface (`ReadStore`) targets this same id. |

## 4. `qname` grammar, per FR-1 language

### 4.0 Contract

- **A reader with only the source file can write the `qname` of any symbol in it.** The reader
  needs no Joern, no build, no import resolution and no type checker, plus the `signature`
  (§4.3) for an overloading language.
- **Every normalizer must produce exactly this string.** S11 checks that on the slice's units:
  for every `symbol` record, it re-derives the `qname` from source by this section and compares.
  A mismatch is an SL-7 delta (P0 §4.2). **[P-slice]** for Python, TS/JS and shell;
  **[P-later]** for the rest.
- **Joern's `FULL_NAME` is not a `qname`.** For example `falkorchat/services.py:<module>.Services.post_message`
  is parse-root-relative and frontend-specific (`skills/joern-cpg/references/cpg-model.md`). The
  normalizer maps it to this grammar from the method's file, line span and AST nesting, never by
  string surgery on `FULL_NAME`.

### 4.1 Common grammar

```
qname    = path "::" chain
path     = the defining file's repo-relative POSIX path (§8.2)
chain    = segment *( "." segment )
segment  = base [ dup ]
base     = NAME | "<anon>" "#" N | "<init>" | "<module>" | "<script>" | "<document>"
         | NAME "." ( "<get>" | "<set>" | "<del>" )        ; accessors: two segments, the property then the accessor
dup      = "#" N                                           ; N ≥ 2, only on a NAME base
N        = decimal integer, no leading zero
NAME     = the identifier exactly as written in source (case kept, NFC), with any quoting
           removed (Kotlin backticks, C# @verbatim)
```

**Rules that hold in every language:**

1. **The path is part of the `qname`.** Rejected alternative: a language-native dotted name such
   as `falkorchat.storefront_api.build_storefront_router.health`. It cannot be written from
   source alone: it needs the import root, `go.mod`'s module path or tsconfig `paths`. It also
   collides across units of one repo (two `tests/conftest.py` both give `tests.conftest`), and the
   symbol id has no `unitId` to separate them (F2). With the path, a moved build root leaves ids
   unchanged and a moved file changes them, which is the move rule proposal §3.8 records.
2. **Scopes.** A segment is added for each enclosing named scope that **contains** the
   definition: types (class, interface, struct, enum, record, trait, object), functions and
   methods, and the namespace-like scopes listed per language. `if`/`try`/`with`/loop blocks,
   comprehensions and plain blocks never add a segment.
   - **Containment is by evaluation scope, not by text** (analyst F2(b)). Decorator, annotation
     and attribute expressions belong to the scope **enclosing** the decorated definition, even
     though its span includes them (§5.3). So does every type annotation.
   - **Parameter default values** belong to the enclosing scope in Python, where they are
     evaluated at definition time. In JS/TS, Kotlin, C# and C++, where they are evaluated at call
     time, they belong to the function itself.
   - Example: in `@pytest.mark.parametrize(…, ids=lambda c: …)` above `def test_x`
     (`falkor-chat/server/tests/test_fusion.py:241`), the lambda is a module-level `<anon>#N`,
     counted with the module's other anonymous scopes, and not `test_x.<anon>#1`.
3. **Top level.** The chain begins with the first named scope. A top-level function `f` is
   `path::f`, never `path::<module>.f`.
4. **One synthetic file symbol.** Every code file has exactly one file-scope symbol: `<module>`
   (code), `<script>` (shell) or `<document>` (Markdown). Its span is the whole file. Code outside
   every function and type belongs to it. Code in a type body but outside any method (field
   initializers, Python class-body statements, Java static or instance blocks, Kotlin `init {}`)
   belongs to the **type's** symbol, except as below.
   - **Exception: a declared primary constructor owns the initializers** (analyst N4). In a
     Kotlin class whose header declares a primary constructor, and in a C# class, struct or
     record with a primary constructor, the `init {}` blocks (Kotlin) and the **instance** property
     and field initializers (both) belong to the primary constructor's `<init>` symbol, not to the type.
     That is what both languages execute: those regions run as the primary constructor's body,
     and its parameters are in scope there. Its `lineStart`/`lineEnd` stay the header's parameter
     list; the initializer regions are its **owned regions** (§5.3 `ownedRegions`, §3.5 `line`).
     Without the exception, `class Orders(r: Repo) { init { r.load() } }` has the `r` param on
     `Orders.<init>` and its use on `Orders`, and §5.10's same-symbol rule cannot express the
     flow. A Kotlin class with no declared primary constructor keeps rule 4 as stated, and so do
     C# `static` field and property initializers, which belong to the type (v1.3, analyst P3-2). A C#
     primary-constructor parameter used inside a **method** is a capture, a cross-symbol fact
     `data` does not express (a known limit, like any field read).
5. **Anonymous scopes are `<anon>#N`.** These are lambdas, function literals, anonymous classes
   and object expressions. `N` counts from 1, in source order (by start line, then start column),
   over the anonymous scopes with the **same parent chain**. Nesting is ordinary:
   `f.<anon>#1.<anon>#2`.
6. **Constructors are `<init>`**, in every language that has them. Python's `__init__` is
   **not** renamed: it is an ordinary method as written.
7. **Duplicates get `#N`.** When two or more symbols have the same parent chain, the same base and
   the same signature (or none), the first in source order keeps the bare base, and the k-th gets
   `#k` (k ≥ 2). Examples are a redefinition, a conditional definition in both branches, a static
   and an instance member with the same name in JS, and Go's several `init`s.
8. **Which definitions are symbols** (analyst F2(a), F4; S03 Pass 2 N1, N5). Two kinds are
   symbols, and nothing else is:
   - every function-like definition with a body;
   - every **bodyless declaration a call can resolve to** that is named in the "symbols" column
     below. Frontends resolve calls to such declarations (Joern resolves `embedder.embed(…)` to the
     `Protocol` stub); if they were not symbols, `callsite.targets[]` would name ids with no record.

   **The table is exhaustive** (v1.2): a bodyless declaration form it does not name is **not** a
   symbol. Each row is decidable from the declaring file alone (§4.0). The organizing reason for
   the right-hand column is that each entry is either a **redeclaration** of an entity whose
   executable declaration is elsewhere (that declaration is the symbol), or **field-like**, or
   reaches no well-formed call.

   | Language | Bodyless declarations that **are** symbols | Bodyless declarations that are **not** |
   |---|---|---|
   | Python | methods of a `Protocol` class and `@abstractmethod` methods, `...`-bodied stubs included (`falkor-chat/server/falkorchat/embedding.py:37`) | `@overload`/`@typing.overload` stubs; all `.pyi` content |
   | TS/JS | `abstract` methods; method signatures (`m(): T`) in an `interface` body, or in an object type literal that is the whole right-hand side of a `type` alias (`T.m`) | overload signatures; `declare` (ambient) declarations; all `.d.ts` content; property signatures of function type (`f: () => void`, field-like); call, construct and index signatures (no name) |
   | Java | `abstract` methods; bodyless interface methods, annotation-type elements included; `native` methods; a record's header, as its canonical constructor `<init>`, when the record declares no explicit canonical or compact constructor (span: the header's component list); each record component with no explicitly declared accessor, as its implicit accessor method `R.x` with signature `()` (span: that component's header entry; v1.3, analyst P3-3), so an explicit accessor added later keeps the same id | — |
   | Kotlin | abstract and bodyless interface functions; `external` functions; the primary constructor `<init>`, when the class header declares one (`class A(…)`, `class A()` or `class A constructor(…)`; span: the header's parameter list) | `expect` declarations (the `actual` is the symbol); default accessors (field-like) |
   | C# | abstract and bodyless interface methods; `extern` methods; the primary constructor `<init>` of a class, struct or positional record (span: the type's parameter list) | a `partial` method's bodyless defining declaration (the implementing declaration is the symbol); auto-accessors and other bodyless accessors (`{ get; set; }`, abstract or interface properties), field-like |
   | Go | **interface method specs** of a named interface type (a `type` declaration, defined or alias), as `IfaceType.NAME` (§4.2 Go); bodyless function declarations (assembly-implemented) | method specs in an unnamed interface type literal (`func f(c interface{ Close() error })`): no source name to key on; type-set terms of constraint interfaces |
   | C/C++ | pure-virtual in-class declarations (`= 0`), whether or not an out-of-line body also exists; `= default` declarations, in-class or out of line | `= delete` declarations (no well-formed call resolves to one); every other bodyless in-class member declaration (its out-of-line definition, wherever it is, is the symbol); free-function prototypes; `extern` declarations; bodyless `friend` declarations; function-like macros |

   - **C++ is decided from the declaring file alone** (analyst N1). Whether `void set(int) const;`
     in `motor.h` is a symbol no longer depends on whether some `.cpp` defines it: it is not. A
     pure virtual with an out-of-line body gives **two** ids, at two paths
     (`motor.h::hw.Motor.f` and `motor.cpp::hw.Motor.f`).
   - **Resolution into a non-symbol.** `targets[]` (§5.6) carries only ids of symbols under this
     rule. When the frontend resolves a call to a non-symbol declaration, the normalizer
     substitutes the entity's symbol if the unit's CPG holds it (a C/C++ prototype or in-class
     declaration → its definition; a C# partial defining declaration → its implementing one), and
     otherwise drops the target. The same applies to an implicit or compiler-synthesized member
     (an undeclared default constructor). If `targets` ends up empty, §5.6's `possible`/`name-match`
     applies, and `resolution` **stays `resolved`** (v1.3, analyst P3-n2): it records what the
     frontend did, so `resolved` with empty `targets` is exactly "resolved to a non-symbol". **[P-later]** for the non-slice languages; how Joern's frontends model these is not
     verified here.
   - **Vendored code is never a symbol** (§5.0, "Vendored paths").

   Types are symbols even without code, so field contract points have a type `qname` to key on
   (§6).
9. **A segment need not be defined in the same file.** Examples are a Go method on a type declared
   in a sibling file, and a C++ out-of-line member definition. The path is the **definition's**
   file.
10. **`name`** (in the `symbol` record) is the last segment's base, without `#N` or an accessor
    suffix. For `<init>`, `name` is the type's name. `nameNorm` is `name` NFC-casefolded and
    stripped.

### 4.2 Per language, with worked examples

Python, TS/JS and shell examples are **real** (the proving repo at the time of writing; S40
re-checks them at `EVAL_COMMIT`). The others are illustrative snippets: the proving repo has no
code in those languages (P0 §1).

#### Python — **[P-slice]**

- **Scopes:** `class`, `def`, `async def`.
- **Properties:** `@property` on `def x` → `x.<get>`; `@x.setter` → `x.<set>`; `@x.deleter` →
  `x.<del>`. Only these literal decorator forms count; anything else (for example
  `functools.cached_property`) leaves the plain name.
- **Lambdas:** `<anon>#N`. Assignment does not name them: `f = lambda: 0` is still `<anon>#N`,
  following Python's own `__name__`.
- **Overload stubs:** a `def` decorated `@overload` or `@typing.overload` is not a symbol. A `.pyi` stub file yields only its `<module>` symbol.
- **Decorators** never change the name.
- **No signature** in the identity (Python does not overload).

| Source (file:line) | `qname` |
|---|---|
| `falkor-chat/server/falkorchat/storefront_api.py:857` `def build_storefront_router(shop)` | `falkor-chat/server/falkorchat/storefront_api.py::build_storefront_router` |
| same file:946, `def health()` under `@router.get("/health", …)`, **inside** `build_storefront_router` (the SL-1 provider) | `falkor-chat/server/falkorchat/storefront_api.py::build_storefront_router.health` |
| same file:879, `def get_participant(…)` inside it | `falkor-chat/server/falkorchat/storefront_api.py::build_storefront_router.get_participant` |
| same file:321, `def __init__` in `class StorefrontHTTPError` | `falkor-chat/server/falkorchat/storefront_api.py::StorefrontHTTPError.__init__` |
| same file:133, `API_PREFIX = "/shop/api"` (a **constant**, §5.9, not a symbol) | `falkor-chat/server/falkorchat/storefront_api.py::API_PREFIX` |
| `falkor-chat/server/falkorchat/app.py:498`, `def health()` under `@app.get("/health")`, inside an `if` in `create_app` (SL-2's bare re-registration) | `falkor-chat/server/falkorchat/app.py::create_app.health`, because the `if` adds no segment |
| `falkor-chat/server/falkorchat/api.py:57`, `def health(ctx…)` inside `build_router` (SL-2's dev router) | `falkor-chat/server/falkorchat/api.py::build_router.health` |
| `falkor-chat/server/falkorchat/guards.py:147–149`: the `"eq"`, `"ne"` and `"lt"` lambdas in the module-level `_OPS` dict (the third is the argument of `_order(…)`) | `…/guards.py::<anon>#1`, `…::<anon>#2`, `…::<anon>#3` |
| the module itself | `falkor-chat/server/falkorchat/guards.py::<module>` |

#### TypeScript / JavaScript — **[P-slice]**

- **Scopes:** `function`, `class`, methods, `namespace`/`module N {}`, and named function values
  (below).
- **The binding names a function value** (ES2015 function-name inference, restricted to these
  forms). The value must be a function expression or arrow function, optionally wrapped in
  parentheses or `as`/`satisfies`/`!`:
  - the whole initializer of `const|let|var NAME = …`, which gives `NAME`;
  - a class field `NAME = …`, which gives `NAME`;
  - an object-literal property `KEY: …`, or method shorthand `KEY() {}`, which gives `KEY` only
    when `KEY` is a **valid identifier**, written bare or as a string literal (`"cleanup"`). Any
    other key (`"a.b"`, `"x#2"`, a number, a computed `[expr]`) gives `<anon>#N`, so no `.` or
    `#N` can enter a segment (analyst F2(c)).
- **A named function expression takes its own name**, never the binding's (analyst F2(d)):
  `const f = function g() {}` is `g`, and `arr.map(function mapper() {})` is `mapper`. ES2015
  inference names only *anonymous* functions, and the rules above apply only to those.
- **Everything else is `<anon>#N`:** callbacks, IIFEs, `module.exports.x = function…` and other
  member assignments.
- **Special names:** `constructor` → `<init>`. `get x()`/`set x()` → `x.<get>`/`x.<set>`.
  `export default function () {}`, `export default () => …` and `export default class {}` →
  `default`.
- **Private names** keep their `#`: `#name`. A `#N` dup suffix is always `#` plus digits only, so
  the two never collide.
- **Types:** `interface`, `type` aliases and `enum` are type symbols.
- **Overload signatures** without a body are not symbols. Only the implementation is.
- **No signature** in the identity.

| Source (file:line) | `qname` |
|---|---|
| `salesperson/src/api/endpoints.ts:16` `export function getHealth(…)` (encloses the SL-1 consumer call) | `salesperson/src/api/endpoints.ts::getHealth` |
| `salesperson/src/api/client.ts:107` `export async function apiFetch<T>(opts)` | `salesperson/src/api/client.ts::apiFetch` |
| same file:73 `function buildUrl(path, query?)` | `salesperson/src/api/client.ts::buildUrl` |
| same file:91, the arrow passed to `setTimeout(…)` inside `timedSignal` | `salesperson/src/api/client.ts::timedSignal.<anon>#1` |
| same file:95 `const onExternalAbort = () => …` inside `timedSignal` | `salesperson/src/api/client.ts::timedSignal.onExternalAbort` |
| same file:99–100, the `isTimeout: () => …` and `cleanup: () => {…}` properties of the returned object literal | `…/client.ts::timedSignal.isTimeout`, `…/client.ts::timedSignal.cleanup` |
| same file:8 of `endpoints.ts`, `const API_PREFIX = '/shop/api'` (a constant, §5.9) | `salesperson/src/api/endpoints.ts::API_PREFIX` |
| `salesperson/src/api/endpoints.ts:10` `export interface HealthResponse` | `salesperson/src/api/endpoints.ts::HealthResponse` |

#### Shell — **[P-slice]** (the slice parses no shell; the P1 extractor is the first producer)

- **Every function is top level.** Both `name() {…}` and `function name {…}` give `NAME`. Bash
  function names are global, so a function defined inside another is still `path::NAME`.
- **The synthetic file symbol is `<script>`.** Commands outside every function are call sites of
  it.

| Source | `qname` |
|---|---|
| `falkor-chat/scripts/bootstrap_schema.sh:24` `gquery() {…}` | `falkor-chat/scripts/bootstrap_schema.sh::gquery` |
| the script's top-level commands | `falkor-chat/scripts/bootstrap_schema.sh::<script>` |

#### Markdown prompts

The only symbol is `path::<document>`, and each tool mention is a call site of it (graph note §6.3).

#### Java — **[P-later]**

- **Scopes:** classes, interfaces, enums, records and annotation types, nested to any depth;
  methods; local classes.
- **The package is not a segment.** It is one per file and implied by the path.
- **Special names:** constructors are `<init>`. Anonymous classes and lambdas are `<anon>#N`.
  Static and instance initializer blocks belong to the type symbol (§4.1 rule 4).
- **Signature:** required (§4.3).

```java
// fx-lib/src/main/java/com/acme/fxlib/Pricing.java   (repoId fx-lib, path src/main/java/com/acme/fxlib/Pricing.java)
public class Pricing {
  public Pricing(Rates r) {…}                          // Pricing.<init>                       (Rates)
  public Money quote(int qty, String sku) {…}          // Pricing.quote                        (int,String)
  public Money quote(int qty, String sku, Currency c)  // Pricing.quote                        (int,String,Currency)
  static class Cache { void put(String k, Money v) }   // Pricing.Cache.put                    (String,Money)
  void warm() { executor.submit(() -> load());          // Pricing.warm ()   and   Pricing.warm.<anon>#1   (no signature: anonymous, 4.3)
                new Runnable() { public void run() {} }; } // Pricing.warm.<anon>#2.run          ()
}
```

The `qname` is the path plus `::`, for example
`src/main/java/com/acme/fxlib/Pricing.java::Pricing.quote` with signature `(int,String)`. That is
GV-7.

#### Kotlin — **[P-later]**

- **Scopes:** `class`, `interface`, `object`, `enum class`, `companion object` (named
  `Companion` unless it is given a name), functions, and local functions.
- **Top-level functions** sit directly under the path.
- **Special names:** the primary and secondary constructors are `<init>`. A declared primary
  one is a symbol even with no body (§4.1 rule 8), and it **owns** the class's `init {}` blocks
  and property initializers (rule 4's exception, v1.2). Without a declared primary constructor,
  they belong to the type symbol. Property accessors **with a body** are `prop.<get>`/`prop.<set>`. Lambdas and `object :` expressions are `<anon>#N`.
- **Extension functions** use the plain function name; the receiver goes into the signature
  (§4.3). Backticked names are written without the backticks.

```kotlin
// fx-svc-kt/src/main/kotlin/acme/Orders.kt
fun String.slug(): String = …            // Orders.kt::slug                  sig  String.()
class Orders(val repo: Repo) {           // Orders.kt::Orders  (type);  primary ctor  Orders.<init>  (Repo)
  init { repo.warm() }                   // owned by Orders.<init>: a call site of it, not of the type
  companion object { fun of(): Orders }  // Orders.kt::Orders.Companion.of   sig  ()
  val total: Int get() = …               // Orders.kt::Orders.total.<get>    sig  ()
  fun place(o: Order) { list.forEach { … } }   // Orders.place (Order)  and  Orders.place.<anon>#1 (no signature)
}
```

#### C#/.NET — **[P-later]**

- **Scopes:** `namespace` blocks **and** file-scoped namespaces, as segments split on `.`. A file
  can hold several namespaces, so they are not implied by the path. Then types (`class`,
  `struct`, `interface`, `record`, `enum`), methods and local functions.
- **Generic types:** a type with generic parameters is ``NAME`N`` (`` `N `` being the arity),
  because `Foo` and `Foo<T>` can coexist.
- **Special names:** constructors (instance and static) are `<init>`; a static constructor is
  told apart by its signature `static()`. A primary constructor (`class C(int x)`, a positional
  record) is `<init>` with the parameter list's signature; it is a symbol by §4.1 rule 8 and owns
  the type's instance field and property initializers (rule 4's exception; `static` ones stay
  with the type). Property and indexer accessors **with a body** (a
  block or `=>`) are `P.<get>`/`P.<set>`, where an `init` accessor is `<set>` and an indexer's
  property segment is `this[]`. Auto-accessors (`{ get; set; }`) are not symbols (§4.1 rule 8).
- **Explicit interface implementations** (`void IDisposable.Dispose()`) use the member name
  (`Dispose`) as the segment and carry `explicit(IDisposable)` at the head of the signature
  (§4.3). A class can then hold both `Dispose()` and `IDisposable.Dispose()`.
  Operators keep their source text with spaces removed (`operator+`). A finalizer is `~Foo`.
  Lambdas and anonymous methods are `<anon>#N`.
- **Top-level statements** belong to `<module>`.
- **Partial types:** each file's members carry that file's path, and the type symbol is emitted
  once per file that declares a part of it.

```csharp
// src/Acme.Api/OrdersController.cs
namespace Acme.Api {
  public class OrdersController {           // …::Acme.Api.OrdersController
    [HttpGet("{id}")] public Order Get(int id)    // …::Acme.Api.OrdersController.Get      (int)
    public bool TryGet(int id, out Order o)        // …::Acme.Api.OrdersController.TryGet   (int,out Order)
  }
  public class Box<T> { T _v; public T Value { get => _v; set => _v = value; } }  // …::Acme.Api.Box`1.Value.<get> ()  and  .Value.<set> (T)
}
```

#### Go — **[P-later]**

- **No package segment.** A package spans several files, and the path identifies the file.
- **Functions:** `NAME`.
- **Types:** every `type` declaration (struct, interface or other, defined or alias) is a type
  symbol `NAME`; a type declared inside a function is `F.NAME` (§4.1 rule 2). Type parameters are
  not in the name.
- **Methods:** `RecvType.NAME`, with the receiver's base type name; pointer `*` and type
  parameters are dropped.
- **Interface method specs** (v1.2, analyst N5) are symbols (§4.1 rule 8): `IfaceType.NAME`,
  under the interface type's symbol, `kind: method`. **An embedded interface adds no segment**
  and contributes no spec of its own: in `type ReadStore interface { Store; Close() }`, only
  `ReadStore.Close` is a new symbol, and `Get` stays `Store.Get` at `Store`'s own path. A call
  through a `ReadStore` value to `Get` therefore targets `Store.Get`. There is no collision with
  `RecvType.NAME`, because an interface type cannot be a method receiver. When an embedder's
  method set holds several specs of one name (Go 1.14+ overlapping embeds, `interface{ A; B }`
  with both declaring `Close()`), a call through it targets **every** such spec, its own and each
  embedded one's (v1.3, analyst P3-n1; never miss). Specs in an unnamed
  interface literal are not symbols (rule 8).
- **Function literals:** `<anon>#N`.
- **Several `init` functions in one file:** `init`, `init#2`, … (§4.1 rule 7).
- **No signature** in the identity.

```go
// svc/store.go   (repoId fx-svc-go)
type Store interface {                             // svc/store.go::Store            (type)
    Get(k string) string                           // svc/store.go::Store.Get        GV-22
}
type ReadStore interface { Store; Close() error }  // svc/store.go::ReadStore, svc/store.go::ReadStore.Close  (no ReadStore.Get)

// svc/server.go
func NewServer() *Server {…}                       // svc/server.go::NewServer
func (s *Server) Handle(w http.ResponseWriter, …) { // svc/server.go::Server.Handle
    go func() { … }()                              // svc/server.go::Server.Handle.<anon>#1
}
func init() {…}; func init() {…}                   // svc/server.go::init, svc/server.go::init#2
```

#### C and C++ (Arduino `.ino` included) — **[P-later]**

- **Scopes (C++):** `namespace` blocks (an anonymous namespace is `<anonns>`), `class`, `struct`,
  `union`, `enum`, and functions.
- **Out-of-line definitions:** the definition's enclosing namespaces are followed by the
  qualifier's segments, with `::` replaced by `.`.
- **Special names:** constructors are `<init>`. Destructors keep `~Foo`. Operators keep their
  source text with spaces removed: `operator()`, `operator==`, `operator bool` → `operatorbool`.
  Lambdas are `<anon>#N`.
- **Templates:** template parameters are not in the name.
- **Bodyless members** follow §4.1 rule 8's C/C++ row: a pure virtual or `= default` is a
  symbol; any other in-class declaration is not, and its out-of-line definition is.
- **C:** functions directly under the path, with no signature (§3.3).
- **Macro-generated names** (a function whose name exists only after preprocessing) cannot be
  written from source. The normalizer names them by the expanded identifier. They are **not
  eligible as key or gold anchors** (S06 should exclude them).
- **Arduino:** the path is the **original `.ino` file** from the line map (proposal §3.2). The
  generated translation unit and generated prototypes never appear.

```cpp
// fw/src/motor.cpp
namespace hw { namespace {                       // anonymous namespace
  int clamp(int v) {…} } }                       // fw/src/motor.cpp::hw.<anonns>.clamp        (int)
void hw::Motor::set(int rpm) const {…}           // fw/src/motor.cpp::hw.Motor.set             (int)const
hw::Motor::Motor(Pins p) {…}                     // fw/src/motor.cpp::hw.Motor.<init>          (Pins)
// blink/blink.ino
void loop() {…}                                  // blink/blink.ino::loop                       ()
```

### 4.3 `signature` (Java, Kotlin, C#, C++ only)

**Which symbols carry one:** the function-like **named** symbols, meaning functions, methods,
constructors and accessors. Types, `<module>` and `<anon>#N` scopes carry **no** signature, so
their tuple has 4 elements in every language. An anonymous scope is told apart by its ordinal, and
a lambda's parameter types are often not written at all.

```
signature = [ "explicit(" IFACE ")" ] [ receiver "." ] [ "<" N ">" ] "(" [ ptype *( "," ptype ) ] ")" [ quals ] | "static()"
```

- **`ptype`** is each parameter's type **text as written in the definition**, with:
  - parameter names, annotations/attributes, default values and the modifiers `final`, `params`
    and `vararg` removed;
  - all whitespace removed.

  Kept as written: generic arguments, array brackets, qualification (`java.util.List` stays as
  written), `?` nullability, and C++ `const`/`&`/`*`. Java varargs `String...` stays `String...`;
  Kotlin `vararg x: String` becomes `String...`. C# `ref`/`out`/`in` are kept, followed by one
  space (`out Order`), the one space the rule allows. The C# extension-method `this` modifier is
  **removed** (`Slug(this string s)` gives `(string)`), because it does not take part in
  overloading (analyst F4).
- **`receiver`** is the Kotlin extension receiver type, as written.
- **`<N>`** is a C# method's generic arity, when it is above 0.
- **`explicit(IFACE)`** is a C# explicit interface implementation's interface, as written.
- **`quals`** are C++'s trailing `const`, `volatile`, `&` and `&&`, concatenated.
- **A static constructor's signature is `static()`.**
- **Why "as written", not resolved:** a resolved type (`java.lang.String`) needs the classpath,
  which a key writer does not have. A reader of the definition always sees the as-written text,
  and Joern's `METHOD_PARAMETER_IN.code` carries it for the normalizer. The cost is that an edit
  to the text only (`List` → `java.util.List`) changes the id. That is accepted under S24's
  orphaning rule for overloading languages (proposal §3.8). **[P-later]**: S11's slice has no
  overloading language.

## 5. Record types

### 5.0 Common to every record

- **Every record carries `type`** (the record-type name below) and **`id`** (32 hex, from the
  tuple given in its entry). No other key is shared.
- **Optional fields are omitted when absent, never `null`** (§8.1). A required array is always
  present, even empty.
- **Locations:** `path` is repo-relative POSIX. `line`, `lineStart` and `lineEnd` are 1-based
  (§8.2).
- **`status` and `reason`** take values from §7. `status` ∈ {`confirmed`, `possible`}.
- **Reference fields** (`symbolId`, `fileId`, …) name a record **in the same bundle**, unless the
  field says it may refer outside it. The validator checks in-bundle references.
- **Vendored paths** (v1.2; S05 review open question 2). Committed vendored code is **external**,
  never in-repo facts. A path is vendored when one of its directory segments **below the unit's
  `rootPath`** is in the `vendoredDirs` list, so a unit's own root never makes it vendored
  (v1.3, analyst P3-4: a first-party unit rooted at `third_party/ourlib/` is not vendored, while
  its `third_party/ourlib/vendor/…` is). The list is per repo, applied to each of its units: by default `vendor`, `vendored`, `_vendor`, `third_party`, `third-party`
  and `node_modules`. A repo's system-manifest entry may give `vendoredDirs`, which **replaces**
  the default for that repo (for a first-party directory that happens to be named `vendor`, or a
  vendored one that is not). The rule is a path test, decidable without a build.
  - The frontend still **parses** vendored files: their types help resolution, which Go's
    vendored-modules-only offline policy needs (proposal §3.2).
  - A vendored file yields **no v1 record of any type**: no `file`, `symbol`, point, `callsite`,
    `import`, `dependency`, `configEntry` or other record. So vendored code has no `Symbol` node,
    and `CALLS_NAME`'s `Name` never resolves to it.
  - A call **into** vendored code is `resolution: external`; `externalTarget` is the frontend's
    full name, as for any out-of-repo target; it is never in `targets[]`. An `import` that
    resolves into a vendored path carries no `resolvedPath`.
  - `secretSighting` (P5) is not bound by this rule: a secret committed under a vendored path is
    still a sighting. P5 fixes its enclosing end.
  - This agrees with `docs/plans/connected-code-graph-p0-ml.md` §7.1 (vendored `METHOD`s are
    external and out of name-match), so P0's fan-out and precision figures describe the
    production population. The ML note's `{VENDORED_DIRS}` per corpus should be this rule's
    result for that corpus. The proving repo has no path matching the default list at the time
    of writing.

The record types, in their file order (§8.3), are `header` (§2.2), `file`, `symbol`, `param`,
`ret`, `cond`, `callsite`, `valueSet`, `constant`, `data`, `controlledBy`, `decorator`,
`fieldAccess`, `import`, `dependency`, `testMarker`, `configEntry`, `launchFact`, `schemaEntry`
and `secretSighting` (reserved).

### 5.1 Access paths (used by §5.6–§5.8, §5.11)

```
argPath  = argHead *( "." FIELD )        argHead = "a" I | "recv" | "kw." NAME
defPath  = defHead *( "." FIELD )        defHead = "p" I | "recv" | "r"
usePath  = argPath | "ret" *( "." FIELD ) | ""
I        = 0-based decimal index
FIELD    = IDENT | "[" JSONSTRING "]"      ; IDENT: a name that is a valid identifier, as written
```

- **`a<i>`** is positional argument *i*, 0-based, **excluding the receiver**. `fetch(url, init)`
  gives `a0`, `a1`.
- **`recv`** is the call's receiver (`router` in `router.get(…)`) or a method's receiver
  parameter (§5.4).
- **`kw.<name>`** is a keyword or named argument: Python `prefix=…`, C# `name: …`, a Kotlin named
  argument, or a Java/C# annotation element.
- **Object-literal and dict-literal arguments are flattened into fields:** `fetch(url,
  {method: m})` puts `m` at `a1.method` (P0 §4.4).
- **`p<i>`** is the enclosing symbol's parameter *i*. **`r`** is a call's result. **`ret`** is
  the returned value. **One encoding per fact** (analyst F5): a value landing on a `Ret` point is
  always `ret` or `ret.<f…>`, never `""`. `""` is used only as the `toPath` of a `cond` use
  point (a condition reading a value). No `defPath` is `""`.
- **Field segments are quoted unless they are identifiers** (graph-dba suggestion). A key that
  is empty or contains anything outside an identifier, such as `.`, `[` or a space, is written
  `["…"]` with JSON string escaping: `a1["content-type"]`, `kw.headers["a.b"]`. This is
  §5.17's `keyPath` quoting, so no path is ambiguous.
- **Depth cap:** at most `fieldDepthCap` (3) `FIELD`s. A deeper path is truncated to its first
  three fields, and the record becomes `status: possible`, `reason: field-truncated`.
- This extends the graph note §6.3 path grammar with `recv` and `kw.` (finding BF4).
  **[P-slice]**

### 5.2 `file`

Id tuple: `["file", repoId, path]`.

| Field | Type | Req | Meaning |
|---|---|---|---|
| `repoId`, `unitId`, `path` | string | yes | |
| `lang` | §3.3 token | yes | |
| `kind` | `code` \| `config` \| `schema` \| `script` \| `prompt` | yes | graph note §3.2 |
| `role` | `primary` \| `example` \| `doc` | yes | The file's purpose, orthogonal to `kind`, the extractor family (graph-dba's BF7 ruling, narrowed from graph note §6.4's five values so the two fields cannot disagree). `example` applies when the basename matches `*.example*`, `*.sample*`, `example*` or `sample*`, or the file lies under an `examples/` or `samples/` directory. `doc` applies to Markdown that is not a prompt-bearing file. Everything else is `primary`. **[P-slice]** |
| `contentHash` | 64-hex | yes | SHA-256 of the file's bytes at `builtCommit` |
| `lineCount` | int | yes | |

### 5.3 `symbol`

Id tuple: §3.2 `symbolId`.

| Field | Type | Req | Meaning |
|---|---|---|---|
| `lang`, `repoId`, `qname` | string | yes | §3, §4 |
| `signature` | string | when §4.3 requires it | §4.3 |
| `fileId`, `path` | | yes | the defining file |
| `kind` | `module` \| `script` \| `document` \| `type` \| `function` \| `method` \| `constructor` \| `accessor` \| `lambda` | yes | `lambda` covers every `<anon>#N` function scope; an anonymous class is `type` |
| `name`, `nameNorm` | string | yes | §4.1 rule 10 |
| `parentId` | 32-hex | no | the enclosing symbol's id; absent on file symbols |
| `lineStart`, `lineEnd` | int | yes | the definition's span, decorators and annotations included. A primary constructor's span is its parameter list; its initializer regions are **owned regions** (§4.1 rule 4), not part of the span. |
| `ownedRegions` | array of `[lineStart, lineEnd]` int pairs, sorted | exactly when the symbol has owned regions (§4.1 rule 4) | the owned regions' line spans (v1.3, analyst P3-1). Not identity: it feeds no tuple. |
| `fingerprint` | 64-hex | yes | SHA-256 of the span's text, normalized like this: decode as UTF-8; split on `\n` (U+000A) **only**; take lines `lineStart`..`lineEnd`, plus every line of its `ownedRegions` (the sorted, de-duplicated union); strip each by §3.1's `isspace` set (which also removes a `\r`); drop empty lines; join with `\n`; encode as UTF-8 (analyst F8(b)). Comments are **kept** in v1. FR-29's "change" signal (proposal §3.3). **[P-slice]** |
| `complexity` | int ≥ 1 | yes | McCabe: decision points in the symbol's own CFG (owned regions included), excluding nested symbols, plus 1. For a file symbol, computed over its top-level code. |
| `lastChangedAt` | RFC 3339 UTC | when `history` is `available` | the latest committer date among the lines of `git blame --porcelain <builtCommit> -- <path>` over the span plus `ownedRegions` |
| `lastChangedCommit` | 40-hex | with `lastChangedAt` | the commit that date belongs to; ties broken by the commit id that sorts lower |

`lastChanged*` needs git history. S11's sandbox stages a `git archive` tree, which has none, so the
normalizer computes these outside the sandbox from the source repository (finding BF12).

### 5.4 `param`

Id tuple: `["point", symbolId, "param", index]`. Graph label `Param`.

| Field | Type | Req | Meaning |
|---|---|---|---|
| `symbolId` | | yes | |
| `index` | int | yes | 0-based position as written, excluding the receiver. **The receiver parameter is `-1`**, path head `recv`. The receiver parameters are: Python's first parameter of a function directly in a class body, unless the function is decorated `@staticmethod`; a Go method's receiver (`s` in `func (s *Server) …`); a Kotlin extension function's receiver (named `this`); and a C# extension method's `this` parameter (analyst F4). Implicit `this` (JVM, C#, JS instance methods) is not a `param`. **[P-slice]** |
| `name` | string | yes | as written; `*args`/`**kwargs`/`...rest` keep their sigil |
| `line` | int | yes | |

### 5.5 `ret` and `cond`

- **`ret`.** Id tuple: `["point", symbolId, "ret", ordinal]`, where `ordinal` is 0-based over the
  symbol's own `return` statements in source order. Expression-bodied functions (an arrow
  function with an expression body, a Python `lambda`, a Kotlin `= expr`) have one `ret`,
  ordinal 0. Fields: `symbolId`, `ordinal`, `line`.
- **`cond`.** Id tuple: `["point", symbolId, "cond", ordinal]`, where `ordinal` is 0-based over
  the symbol's own control-structure conditions in source order: `if`/`elif`/`while`/`for`
  headers, ternaries, `case` labels, `catch` guards, and short-circuit operands that guard a
  point. Fields: `symbolId`, `ordinal`, `line`, `code`. `code` is the condition's source text,
  whitespace runs collapsed to one space, truncated to 200 characters (graph note §6.3).
  **[P-slice]** (Joern's CDG sources decide what counts. S11 records any class of condition the
  definition above misses.)

### 5.6 `callsite`

Id tuple: §3.2 `callSiteId`. A call site is itself a point (graph note §6.3).

| Field | Type | Req | Meaning |
|---|---|---|---|
| `symbolId` | | yes | the enclosing symbol (§4.1 rule 4) |
| `calleeName` | string | yes | the **last identifier** of the callee expression, as written, without type arguments: `fetch`, `apiFetch`, `include_router`, `get`. A computed callee (`obj[k]()`, `(a‖b)()`) gives `<dynamic>`. `new Foo()` gives `Foo`, `super(…)` gives `super`. A shell command or function name, or a prompt's mentioned tool name, is used as written. |
| `ordinal` | int | yes | §5.7 |
| `callKind` | `call` \| `new` \| `super` \| `tagged-template` \| `command` \| `mention` | yes | |
| `line` | int | yes | the line of the callee's last identifier |
| `resolution` | `resolved` \| `unresolved` \| `external` | yes | the frontend's resolution, as normalized |
| `targets` | array of `{symbolId, qname, signature?}` | yes, possibly empty | in-repo resolved targets. They may be in another file or another bundle, since the id is computable from the `qname` alone. Sorted by `symbolId`. Only symbols under §4.1 rule 8 (non-symbol resolutions are substituted or dropped there); never vendored code (§5.0). |
| `externalTarget` | string | when `resolution` is `external` | the frontend's full name for an out-of-repo target, verbatim. Used only for `FROM_DEP` matching (P5). |
| `status`, `reason` | | yes | `confirmed`/`resolved-call` when `targets` is non-empty; `possible`/`name-match` otherwise; `possible`/`shell-dynamic` or `prompt-mention` for those producers |

- **Which calls are call sites:** every source-level call except language operators. Member
  access, indexing, arithmetic, string concatenation, f-strings and template literals are not call
  sites (Joern's `<operator>.*` calls); tagged templates are.
- **Decorator and annotation applications are not call sites.** They are `decorator` records
  (§5.12), and the one fact is not emitted twice.
- **F1 is followed (P0 §11):** `targets` records Joern's resolution, and the assembler **always**
  also emits `CALLS_NAME` for the site. Other same-name candidates are never dropped.

### 5.7 The call-site ordinal

- **Order** the enclosing symbol's own call sites (not those of nested symbols) by the position of
  their callee's last identifier: line, then column, with columns counted in code points.
- **`ordinal`** is the 0-based rank among the call sites with the same `calleeName`.
- **Using the last identifier** makes chained and nested calls total-ordered. In
  `x.foo().foo()` the inner `foo` comes first; in `f(g())`, `f` comes before `g`.
  **[P-slice]**

### 5.8 `valueSet`

Id tuple: `["valueSet", callSiteId, argPath]`. One record per (call site, argument path) that has
a recoverable or explainable value. These are an assembly input with no graph node (§9).

| Field | Type | Req | Meaning |
|---|---|---|---|
| `callSiteId` | | yes | |
| `arg` | `argPath` (§5.1) | yes | for example `a0`, `a1.method`, `kw.prefix` |
| `values` | array of `element`, at least 1 | yes | **alternatives**: the argument is one of these |

```
element = {"const": STRING | INT | BOOL}
        | {"concat":   [element, …]}          ; a + b + …, in order
        | {"template": [element, …]}          ; template literal / f-string / interpolated string; literal parts are {"const": "…"}
        | {"ref": QNAME}                      ; a constant (§5.9), in this bundle or another, by its qname
        | {"importRef": {"module": STRING, "name": STRING}}   ; an imported name the normalizer could not resolve to a path
        | {"unknown": REASON}                 ; §7.3
```

- **Caps:** at most 16 alternatives, and nesting at most 4 deep. Past either cap the excess is
  replaced by `{"unknown": "value-set-cap"}`.
- **Order:** `values` is sorted by the canonical JSON (§3.1) of each element, which makes it a set.
  The parts of `concat`/`template` keep source order.
- **Resolution is the assembler's job.** A value that flows from a parameter or a call result is
  `{"unknown": "point-derived"}`, and the `data` records say where it comes from. **[P-slice]**

**Examples (the SL-1/SL-2 facts):**
- `endpoints.ts` `getHealth`'s `apiFetch(…)` call. Arg `a0.path`:
  `[{"template":[{"ref":"salesperson/src/api/endpoints.ts::API_PREFIX"},{"const":"/health"}]}]`.
  Arg `a0.method`: `[{"const":"GET"}]`.
- `app.py`'s `app.include_router(storefront_api.build_storefront_router(shop), prefix=storefront_api.API_PREFIX)`.
  Arg `kw.prefix`: `[{"ref":"falkor-chat/server/falkorchat/storefront_api.py::API_PREFIX"}]`. The
  normalizer resolves `from . import storefront_api` to that path.

### 5.9 `constant`

Id tuple: `["constant", lang, repoId, qname]`. A module-level or type-level variable assigned
exactly once, where the right-hand side yields a value set. It is the target of `{"ref": …}`.
Not a symbol: no node (§9). This record type is not in P0 §4.2's list (finding BF5).

| Field | Type | Req | Meaning |
|---|---|---|---|
| `qname` | string | yes | §4 grammar, with the variable name as the last segment: `…/storefront_api.py::API_PREFIX`; a class attribute is `…::Cls.ATTR` |
| `fileId`, `line` | | yes | |
| `values` | array of `element` | yes | as in §5.8 |

### 5.10 `data` (the flow projection, use → def)

Id tuple: `["data", use, def, usePath, defPath]`. Graph edge `DATA`.

| Field | Type | Req | Meaning |
|---|---|---|---|
| `use` | pointId | yes | the point whose value **depends on** the other; the edge's stored source |
| `def` | pointId | yes | the point the value comes from; the edge's stored target |
| `toPath` | `usePath` | yes | where on `use` the value lands: `a0`, `a1.method`, `ret`, or `""` on a `cond` (§5.1) |
| `fromPath` | `defPath` | yes | where on `def` it comes from: `p0.path`, `recv`, `r`; never `""` (§5.1) |
| `status`, `reason` | | yes | `confirmed`/`intra-flow`, or `possible` with `field-truncated` or `shell-dynamic` |

- **Field names.** `fromPath`/`toPath` keep the graph note's names and their value-flow meaning.
  The node fields are `use`/`def`, **not** P0 §4.2's `from`/`to`, because those read as the
  opposite of `fromPath`/`toPath` (finding BF3).
- **Both points belong to the same symbol.** The projection is intraprocedural (proposal §3.3).
- **Endpoint kinds follow from §5.1's heads.** `use` is a `callsite` (an `argPath`), a `ret`
  (`ret…`) or a `cond` (`""`); `def` is a `param` (`p<i>`, `recv`) or a `callsite` (`r`). A `cond`
  is only ever a `use` end, and a `ret` never a `def`.
- **The AC-1 chain** (graph note §6.3):
  - in `apiFetch`, `{use: <fetch call>, def: <buildUrl call>, toPath: "a0", fromPath: "r"}`;
  - `{use: <buildUrl call>, def: <apiFetch p0>, toPath: "a0", fromPath: "p0.path"}`;
  - `{use: <fetch call>, def: <apiFetch p0>, toPath: "a1.method", fromPath: "p0.method"}`;
  - in `buildUrl`, `{use: <ret 0>, def: <buildUrl p0>, toPath: "ret", fromPath: "p0"}`.

  **[P-slice]**

### 5.11 `controlledBy`

Id tuple: `["controlledBy", point, cond]`. Graph edge `CONTROLLED_BY`, confirmed by construction.
Fields: `point` (any pointId: a call site, `ret` or `param`), `cond` (a `cond` pointId). Both are
in the same symbol. The relation is Joern's CDG, from the point's statement to its controlling
condition (P0 §4.4). **[P-slice]**

### 5.12 `decorator` (decorators, annotations, attributes)

Id tuple: `["decorator", targetId, ordinal]`.

| Field | Type | Req | Meaning |
|---|---|---|---|
| `targetId` | symbolId | yes | the decorated function, method or type (a class-level `@RequestMapping` targets the type) |
| `ordinal` | int | yes | 0-based, in source order on that target |
| `form` | `decorator` \| `annotation` \| `attribute` | yes | Python/TS decorators; Java/Kotlin annotations; C# attributes |
| `name` | string | yes | the decorator expression **without its argument list**, as written, whitespace removed: `router.get`, `app.get`, `GetMapping`, `HttpGet`. The `Attribute` suffix is not stripped. |
| `line` | int | yes | |
| `args` | array of `{arg: argPath, values: [element]}` | yes, possibly empty | the argument value sets, inline, sorted by `arg`. The `@router.get("/health", …)` decorator has `{arg: "a0", values: [{"const": "/health"}]}`. |

### 5.13 `fieldAccess`

Id tuple: `["fieldAccess", pointId, path, mode, receiverType, field]`. **`path` is in the tuple** (analyst F1): `self._loop.call_soon_threadsafe(self._loop.stop)` (`falkor-chat/server/falkorchat/tools.py:1480`) reads `_loop` at `recv` and at `a0` of one call site, and without `path` the two records would share an id (GV-20/GV-21). This is the bundle input for
graph note §6.3's `(point)-[:READS|WRITES]->(:ContractPoint {kind: 'field'})`. Not in P0 §4.2's
list (finding BF5). **[P-slice]** if the slice emits fields, **[P-later]** otherwise (AC-1 needs
none).

| Field | Type | Req | Meaning |
|---|---|---|---|
| `pointId` | | yes | for a **read**, the point that uses the value (with `path` its `argPath`/`recv`/`ret`); for a **write**, the point the written value comes from (with `path` its `defPath`) |
| `path` | | yes | |
| `mode` | `read` \| `write` | yes | |
| `receiverType` | type `qname`, or `"?"` | yes | the declaring type, when the receiver's type is resolved from source (Python `self`/`cls` in a method resolves to the enclosing class), else `"?"` |
| `field` | string | yes | |
| `line` | int | yes | |
| `status`, `reason` | | yes | `confirmed`/`resolved-receiver`, or `possible`/`field-name-match` when `receiverType` is `"?"` |

### 5.14 `import`

Id tuple: `["import", fileId, ordinal]`, with `ordinal` 0-based per file in source order. These
are an assembly input: `library_api` consumption and `{importRef}` resolution. No node.

| Field | Type | Req | Meaning |
|---|---|---|---|
| `fileId`, `line` | | yes | |
| `symbolId` | | no | present when the import is inside a function (a local import) |
| `module` | string | yes | the specifier as written: `./client`, `falkorchat.config`, `.` |
| `names` | array of `{name, alias?}` | yes | `[]` for a whole-module import; `*` for a star import. Sorted by (`name`, `alias`), with an absent `alias` first (§8.1). |
| `resolvedPath` | repo-relative path | no | when the normalizer resolves it to an in-repo file |
| `dynamic` | bool | yes | `import(…)`, `importlib.import_module(…)` or `require(<non-literal>)`. When `true` and the specifier is not a constant, the assembler emits `possible`/`dynamic-import` (AC-6). |

### 5.15 `dependency`

Id: the `depId` tuple (§3.2), extended to `["dependency", ecosystem, package, manifestPath, line]`
for the **record** id, so one manifest can declare a package twice (runtime and dev). Graph:
`(:Unit)-[:DEPENDS_ON {version, manifestPath, line}]->(:Dependency {depId, ecosystem, package})`.

| Field | Type | Req | Meaning |
|---|---|---|---|
| `depId` | 32-hex | yes | `h(["dependency", ecosystem, package])` |
| `ecosystem` | OSV ecosystem name: `PyPI`, `npm`, `Maven`, `Go`, `NuGet`, `crates.io`, `ConanCenter` | yes | |
| `package` | string | yes | PyPI: the PEP 503 normalized name. npm: as written, which is lowercase by registry rule. Maven: `groupId:artifactId`. Go: the module path. NuGet: lowercased. |
| `version` | string | yes | the version or range as declared |
| `scope` | `runtime` \| `dev` \| `test` \| `build` \| `optional` | yes | |
| `manifestPath`, `line` | | yes | |

### 5.16 `testMarker`

Id tuple: `["testMarker", symbolId, framework]`. Gives the `Test` secondary label on the symbol.
`testKind` is decided at assembly by pack rules; this record carries only the hint.

Fields:
- `symbolId`;
- `framework` (`pytest`, `unittest`, `vitest`, `jest`, `playwright`, `junit`, `kotest`, `xunit`,
  `nunit`, `go-test`, `gtest`, `bats`, …);
- `kindHint` ∈ {`unit`, `integration`, `e2e`, `unknown`};
- `evidence` ∈ {`decorator`, `name`, `path`, `base-class`, `config`}.

### 5.17 `configEntry`

Id tuple: `["configEntry", repoId, path, keyPath]`. Graph label `ConfigEntry`.

| Field | Type | Req | Meaning |
|---|---|---|---|
| `repoId`, `path`, `line` | | yes | |
| `format` | `json` \| `yaml` \| `toml` \| `compose` \| `env` \| `dockerfile` \| `mcp-json` \| `kiro-agent` \| `ini` | yes | |
| `keyPath` | string | yes | dotted, with segments as written. Array elements are `[i]`; a segment containing `.` or `[` is quoted `["…"]`. Example: `services.falkordb.ports[0]`. |
| `valueHash` | 64-hex | yes | **HMAC-SHA-256** of the value's text, keyed with the system's key kept outside the graph and the repo. A plain hash is brute-forceable for low-entropy secrets (finding BF8). P0's key is `~/.config/ccg/<systemId>.hmac`. |
| `bindValue` | string | no | the value in clear, **only** if all of these hold: (a) it is an integer 1–65535; a host name (`localhost`, an IP literal, or a DNS name of `[a-z0-9.-]` labels); or a URL with no userinfo, no query and no fragment whose path is empty, `/`, or **one** segment of at most 32 characters from `[a-z0-9_-]`, optionally followed by one trailing `/` (`http://localhost:8000/mcp`, `…/mcp/`; analyst P2-n2: the slash carries no information, so admitting it adds no exposure). The **path class is dropped** and multi-segment URLs are excluded until the `security-expert` advisory (analyst F6, interim; coordinator ruling): `https://hooks.slack.com/services/T0/B0/XXXX` does not qualify; (b) no `keyPath` segment matches the secret-name list (case-insensitive **substring** match on `pass`, `pwd`, `secret`, `token`, `key`, `cred`, `auth`, `cookie`, `session`, `private`, `cert`, `dsn`, `conn`, `hook`, `sig`; over-redaction is intended); (c) it is shorter than 512 characters. This is what URL, env and Compose binding read (proposal §3.4). **[P-later]**, needs security review before P1 (BF8). |
| `example` | bool | yes | copied from the file's `role` |

### 5.18 `launchFact` and `schemaEntry`

**`launchFact`.** Id tuple: `["launchFact", repoId, path, kind, ordinal]`. These are explicit
launch facts from Dockerfiles, Compose, `.mcp.json`, Kiro agents and shell `exec` lines.
Framework defaults such as uvicorn's 8000 come from packs at assembly, not from the bundle.

Fields:
- `path`, `line`;
- `kind` ∈ {`command`, `port`, `env-default`, `mount`};
- `command` (argv[0]) and `args` (string array) for `command`;
- `port` (int) and `protocol` for `port`;
- `envVar` and `default` for `env-default` (`AGENT_TEAM_PORT:-8200` gives `envVar:
  "AGENT_TEAM_PORT"`, `default: "8200"`);
- `mountPath` for `mount`;
- `service` (the Compose service, when there is one);
- `origin` ∈ {`explicit`, `env-default`, `compose`}.

Graph: a `ConfigEntry` node (`entryId` = the record id) with `LAUNCHES` to the target `File`/`Unit`
resolved at assembly. It is `BOUND_BY` evidence on links. **[P-later]**

**`schemaEntry`.** Id tuple: `["schemaEntry", repoId, path, objectKey]`, where `objectKey` is the
`keyNorm` of the `data_entity` it defines (§6). From the SQL DDL/migration extractor (P3;
proposal §3.2). Fields: `path`, `line`, `objectKey`, `migration` (string, the migration id when
there is one), `op` ∈ {`create`, `alter`, `drop`, `rename`}. Graph: `SchemaEntry`, `DEFINED_BY`
from the contract point. Not in P0 §4.2's list (finding BF5). **[P-later]**

### 5.19 `secretSighting` (reserved, P5)

Id tuple: `["secretSighting", repoId, path, commit, fingerprint]`.

Fields (graph note §6.4 FR-15 row):
- `fingerprint` (HMAC, as in §5.17);
- `path`, `line`, `commit`, `committedAt`;
- `removedInCommit` (optional);
- `presentAtHead` (bool);
- `symbolId` or `entryId` (the enclosing end, for `findingId`).

**v1 writers must not emit it.** v1 validators accept it, so P5 adds it without a schema break.
The value is never present in any field.

## 6. `ContractPoint (kind, keyNorm)`

- **The tuple and id:** `cpId = h(["contractPoint", kind, keyNorm])` (§3.2). `kind` is a property,
  not a label (graph note §3.2).
- **Who creates them:** recognizers create contract points **at assembly**. No v1 bundle record is
  a contract point. The tuple is fixed here because gold, keys and fixture manifests name contract
  anchors by it (P0 D8).
- **Where `keyNorm` comes from:** packs extract a raw `key` from value sets, and the core
  normalizes it by this section. Keys and gold use exactly this form.
- **Unknown parts:** a key with unresolved parts uses the placeholders below. The linker treats
  them as partial keys (proposal §3.4, `possible`/`partial-key`).
- **Common rule for every kind:** NFC, then strip (§3.1), then the kind's rule. Case is preserved
  unless the kind's rule says otherwise.

| `kind` | `keyNorm` form | Rule | Example |
|---|---|---|---|
| `http` | `<VERB> <path>` | (1) `VERB` is uppercase, from {GET, HEAD, POST, PUT, PATCH, DELETE, OPTIONS, TRACE, CONNECT}, or `*` for an any-method registration or an unknown method. (2) Take the URL's **path only**: scheme, userinfo, host and port belong to URL binding, and query and fragment are dropped. (3) A mounted provider's path is the mount prefix(es), outermost first, concatenated with the route path, **before** steps 4–8. (4) Ensure a leading `/`; an empty path is `/`. (5) Collapse runs of `/`; resolve `.` and `..` segments. (6) Percent-decode the unreserved characters `A–Z a–z 0–9 - . _ ~`, and uppercase the hex of every other escape (RFC 3986 §6.2.2). (7) A path parameter in any framework syntax becomes `{}` for one segment and `{**}` for a catch-all. The syntaxes include `{id}`, `{id:int}`, `:id`, `<id>`, `<int:id>`, `[id]`; the catch-alls include `{p:path}`, `*`, `[...p]`, `{*p}`. (8) A consumer segment whose value is unresolved becomes `{?}`. A trailing `/` is **kept** (frameworks treat `/x` and `/x/` differently); the matcher may relate them as possible. | `GET /shop/api/health`: the storefront `@router.get("/health")` under `prefix=API_PREFIX` (`"/shop/api"`), and the consumer `` `${API_PREFIX}/health` `` with `method: 'GET'`. Both normalize to the same key (SL-1). |
| `mcp_tool` | `<toolName>` | The tool name as registered: the function name, or the explicit `name=`. Case kept. The server is **not** in the key; URL and launch binding select it (proposal §3.4). | `send_message`, `query` |
| `rpc` | `/<package>.<Service>/<Method>` | The gRPC full method path from the `.proto`; `/<Service>/<Method>` when there is no package. Case as declared. | `/acme.fxlib.Pricing/Quote` |
| `topic` | `<transport>:<name>` | `transport` is a pack-declared lowercase token (`kafka`, `amqp`, `nats`, `redis-stream`, `sqs`, or `inhouse-<packId>` for an in-house bus). `name` is as written. | `kafka:orders.created` |
| `data_entity` | `<dialect>:<objectKind>:<name>` | `dialect` ∈ {`sql`, `cypher`, …}. For `sql`, `objectKind` ∈ {`table`, `view`, `column`}, and `name` is `[schema.]table[.column]`: unquoted identifiers lowercased, quoted ones kept with their quotes removed, and the schema present only when written. For `cypher`, `objectKind` ∈ {`label`, `rel`, `prop`}, and `name` is the label, the relationship type, or `<Label>.<prop>`, case kept. **No store scope**, finding BF10. **[P-later]** | `sql:table:orders`, `sql:column:orders.total`, `cypher:label:Message` |
| `env_var` | `<NAME>` | As written; case kept. | `AGENT_TEAM_PORT` |
| `config_key` | `<namespace>:<keyPath>` | `namespace` is pack-declared (`pydantic-settings`, `compose`, `spring`, …). `keyPath` follows §5.17's grammar. **[P-later]** | `spring:server.port` |
| `field` | `<typeQname>.<field>` or `?.<field>` | `typeQname` is the declaring type's `qname` (§4). `?` when the receiver type is unresolved (§5.13). | `falkor-chat/server/falkorchat/services.py::Services.repo` |
| `library_api` | `<lang>:<nativeQname><signature>` | `nativeQname` is the **published import name**, which is what a consumer writes: Python's dotted module path plus the name, a Java/Kotlin/C# fully qualified type plus the member, Go's module path plus the name. `signature` is §4.3's form for **every** language here (a Python signature is its parameter names and markers as written, `(text,*,limit)`), because a library's signature is its contract in any language. Readable, not hashed, finding BF9. **[P-later]** | `java:com.acme.fxlib.Pricing.quote(int,String)` |

New kinds are added by packs. Each new kind must state its `keyNorm` rule in the pack's
vocabulary entry (S30), in the same form as this table.

## 7. Reason codes

### 7.1 Registry rules

- **Closed and extensible.** A code is valid only if it is in this registry, or in a later version
  of it (§8.6). A pack may not invent codes. A new code is a registry change.
- **Parameterized codes** use `code:<param>`, and only the forms listed here.
- **"First emitted"** is the phase that first produces the code: P1–P5 per proposal §4, or "slot"
  for the Arduino slot. "Bundle" marks codes that may appear **in bundle records**. All the others
  are assembly or engine codes, listed so that S35's fixture manifests and S33's answer schema
  draw from one list.

### 7.2 Status reasons

| Code | Status | On | Meaning | Source | First emitted | Bundle |
|---|---|---|---|---|---|---|
| `resolved-call` | confirmed | `CALLS`, callsite | the frontend resolved the target | graph note §3.2 | P1 | yes |
| `verdict-confirmed` | confirmed | `CALLS`, `Link`, `READS`/`WRITES` | a person confirmed it | graph note §3.2–§3.3 | P4 | |
| `intra-flow` | confirmed | `DATA` | a projected intraprocedural flow | this spec | P1 | yes |
| `resolved-receiver` | confirmed | field `READS`/`WRITES` | the receiver's type is resolved | this spec | P1 | yes |
| `key-match` | confirmed | `CONSUMES`, `Link` | both keys known and matching | proposal §3.4 | P1 | |
| `lifted-role` | confirmed | `CONSUMES` | the consumer comes from role lifting with a known key | graph note §6.3 | P1 | |
| `binding-confirmed` | confirmed | `Link` | a URL or launch binding selects exactly one unit (several deployments of one unit included) | proposal §3.4 | P1 | |
| `name-match` | possible | `CALLS_NAME`, callsite | an unresolved call, matched by name | proposal §3.2 | P1 | yes |
| `resolution-unverified` | possible | `CALLS_NAME` candidates of a resolved site | another same-name candidate of a resolved call; never dropped (F1) | proposal §3.7 | P1 | |
| `field-truncated` | possible | `DATA`, `fieldAccess` | access path deeper than the cap | proposal §3.3 | P1 | yes |
| `field-name-match` | possible | field `READS`/`WRITES` | receiver type unresolved | graph note §6.3 | P1 | yes |
| `shell-dynamic` | possible | shell `DATA`, callsite | `eval`, indirect expansion, computed `source` | proposal §3.3 | P1 | yes |
| `runtime-built` | possible | `CONSUMES`, `READS`/`WRITES` | the key is built at runtime and cannot be recovered | proposal §3.3 | P1 | |
| `dynamic-import` | possible | `CONSUMES library_api`, call reach | an import with a non-constant specifier (AC-6) | P0 §4.7 | P1 | |
| `example-config` | possible | `Link`, `BOUND_BY` | wiring through an example config | proposal §3.4 | P1 | |
| `prompt-mention` | possible | callsite (`mention`), `CONSUMES` | a prompt names a tool | proposal §3.4 | P1 | yes |
| `partial-key` | possible | `Link` | a key with `{?}` parts, or a partly known key | proposal §3.4 | P1 | |
| `name-only-provider` | possible | `Link` | the provider was chosen by name only | proposal §3.4 | P1 | |
| `unknown-key` | possible | `Link` | a fully unknown key with no binding: one link per provider of the kind | proposal §3.4 | P1 | |
| `binding-narrowed` | possible | `Link` | an unknown key narrowed by a binding | proposal §3.4 | P1 | |
| `port-shared` | possible | `Link` | two **different** units on one port | proposal §3.4 | P1 | |
| `target-not-mapped` | possible | `Link` (`TO_UNIT`) | the target unit was never built (FR-7a) | proposal §3.11 | P1 | |
| `flow-unavailable:<lang>` | possible | impact items | reach through a unit with `flowAvailable = false`; `<lang>` is a §3.3 token | proposal §3.2 | P1 | |
| `config-derived` | possible | `CONSUMES topic` | a topic name read from config at runtime (AC-6 topic-from-config) | P0 §4.7 | P2 | |
| `headers-unavailable` | possible | `CALLS_NAME` | resolution needed headers absent under local-only | proposal §3.2 | P2 (C/C++), slot | |
| `possibly-reachable` | possible | dependency `Finding` | reachability undecidable; never reported unreachable | proposal §3.7 | P5 | |

### 7.3 Value-set `unknown` reasons (bundle)

| Code | Meaning | First emitted |
|---|---|---|
| `point-derived` | the value flows from a parameter or a call result; see `data` | P1 |
| `runtime-built` | computed by operations the extractor does not fold (calls, formatting with non-constant parts) | P1 |
| `unresolved-ref` | a name whose definition is not a `constant` the normalizer can see | P1 |
| `spread-argument` | `*args`, `**kwargs`, `...spread` or `params` arrays | P1 |
| `value-set-cap` | §5.8's caps reached | P1 |

### 7.4 Severity reasons (findings, P3/P5)

`advisory-unscored` (proposal §3.7, P5). S30's severity rule table adds the others, under the same
registry rules.

## 8. Determinism rules

### 8.1 Serialization

- **One record per line**, terminated by `\n` (LF), with no `\r`. The file ends with exactly one
  `\n`. UTF-8 with no BOM.
- **Each line is the canonical JSON of its record**, as in §3.1: sorted keys, `,`/`:` separators
  with no spaces, non-ASCII written literally.
- **Every string field is NFC-normalized.** Identity fields (ids, `qname`, `path`, `keyPath`,
  `name`, `calleeName`) are also stripped. Free-text fields (`cond.code`, `const` values) are
  **not** stripped: a constant `" x"` keeps its space.
- **Numbers are integers only.** No floats anywhere.
- **`null` never appears.** An absent optional field is omitted.
- **Arrays** are sorted where a field says so (`targets`, `values`, `args`, and `import.names` by (`name`, `alias`), with an absent `alias` first; analyst F8(a)).
  Otherwise they are in source order, as stated per field.

### 8.2 Paths and lines

- **Paths are repo-relative POSIX**, with no leading `./`, no `..`, no trailing `/`, and case as
  in the tree. The normalizer rewrites Joern's parse-root-relative `FILENAME`
  (`skills/joern-cpg/references/cpg-model.md`) by prefixing the unit's `rootPath`.
- **Lines are 1-based.** Arduino locations go through the sketch line map (proposal §3.2), and a
  location that maps to *generated* is dropped.

### 8.3 Record order

The `header` comes first. Then come the types in this order, each block sorted by `id` (byte
order of the lowercase hex):

`file`, `symbol`, `param`, `ret`, `cond`, `callsite`, `valueSet`, `constant`, `data`,
`controlledBy`, `decorator`, `fieldAccess`, `import`, `dependency`, `testMarker`, `configEntry`,
`launchFact`, `schemaEntry`, `secretSighting`.

**Duplicate ids within a type are a bundle error.** Two records with one id mean a tuple rule
failed to separate them, for instance a missed §4.1 rule 7 case. The writer must fail rather than
drop one, the bundle-side mirror of the graph's post-load `UNIQUE` gate (graph note §3.4).

### 8.4 Inputs that must not influence output

- Wall-clock time, except `builtAt`.
- The clone's absolute path, the host and the user. (The system manifest's `vendoredDirs`, by
  contrast, is an input, and `header.vendoredDirs` records it.)
- Locale (string sorting is by code point, never by collation).
- Hash-map iteration order.
- Thread scheduling.
- The order Joern returns nodes in. The normalizer sorts everything by this spec's keys before
  writing.
- The source clone's **history depth**. A shallow clone changes `lastChanged*`, so P0 requires
  full history (§2.2 `history`; analyst F8(c)).

### 8.5 Exempt fields

`header.builtAt` and `header.sandboxAuditIds` are the only fields allowed to differ between two
builds of the same unit at the same commit with the same tools. **SL-6's byte-identity check
compares bundles with these two fields removed.** **[P-slice]**

### 8.6 Versioning

- **`bundleSchemaVersion` is `MAJOR.MINOR`.** v1 is `1.0`.
- **A minor bump is additive only:** a new optional field, a new record type, a new reason code, a
  new contract kind. A validator for `1.n` rejects unknown fields and types (S10's strictness),
  so a reader declares the highest minor it accepts. The assembler refuses a bundle whose minor is
  above its own, and the unit then becomes `stale-kept` (proposal §3.11).
- **A major bump is anything else:** a changed tuple, a changed `qname` rule, a renamed or retyped
  field. Every id-bearing change is major, because it orphans verdicts (proposal §3.8).
- **Who writes a change:** a minor version is written into a revised or successor spec under
  collision rule 5. A major version is always a successor document.
- **The reserved `secretSighting`** is part of `1.0`'s schema, so P5 needs no bump to emit it.

## 9. Mapping to graph-note labels and edges

The labels and edges are those of graph note §3.2 as amended by §6 (§6.3 points, §6.4 deltas).
`graph-dba` checks this table at the S03 gate. S17 restates it as graph model v1.

### 9.1 Record → graph

| Record | Graph label / edge | Properties carried | Not carried (stays in the bundle, or S17 decides) |
|---|---|---|---|
| `header` | `(:Unit {unitId, repoId, rootPath, language, builtCommit, builtTree, builtAt, buildStatus:'fresh'})-[:IN_REPO]->(:Repo {repoId})`; feeds `:Snapshot.bundleSchemaVersion` | `lang` → `Unit.language` (renamed). `flowProjected` is an input to `Unit.flowAvailable`. | `tools`, `egressMode`, `sandboxAuditIds`, `counts`, `history`: build provenance (BF13) |
| `file` | `(:Unit)-[:HAS_FILE]->(:File {fileId, repoId, unitId, path, kind, role})` | `role` ∈ {`primary`, `example`, `doc`} per §5.2 (graph-dba's BF7 ruling, narrowed from graph note §6.4's five values) | `lang`, `contentHash`, `lineCount` (BF13) |
| `symbol` | `(:File)-[:DEFINES]->(:Symbol {symbolId, repoId, unitId, kind, name, nameNorm, qname, path, lineStart, lineEnd, ownedRegions, fingerprint, lastChangedAt, lastChangedCommit})` | `unitId` from the header. `lastChanged*` per graph note §6.4 FR-12. **File+line anchor rule for S17** (v1.3, P3-1; replaces graph note §3.4's span-only test): a line maps to the symbol with the **narrowest** interval, span or owned region, containing it; on a tie a span match beats an owned-region match, then the deeper `parentId` chain wins. `ownedRegions` is carried so the store can apply it (S17 picks the encoding, for example a flattened list). | `lang`, `signature`, `complexity` (proposal §3.3 lists complexity as a risk input), `parentId` (BF13). `isEntry` comes from packs at assembly. `paramToReturn` and `ARG_FLOWS` are **not** produced: they are superseded by points (graph note §6.3). |
| `param` | `(:Symbol)-[:HAS_PARAM]->(:Param {pointId, index, name, line})` | | |
| `ret` | `(:Symbol)-[:HAS_RETURN]->(:Ret {pointId, line})` | | `ordinal` (it is in the id) |
| `cond` | `(:Symbol)-[:HAS_COND]->(:Cond {pointId, line, code})` | | `ordinal` |
| `callsite` | `(:Symbol)-[:HAS_CALL]->(:CallSite {callSiteId, repoId, unitId, path, line, calleeName})`. Each `targets[]` gives `(:CallSite)-[:CALLS {status:'confirmed', reason:'resolved-call'}]->(:Symbol)`. **Every** site except a `<dynamic>` one gives `(:CallSite)-[:CALLS_NAME]->(:Name {nameKey, lang, name})`, and the assembler adds `(:Name)-[:RESOLVES_TO]->(:Symbol)` for every symbol with that `lang` and `name` (graph note §6.1 item 6, F1). A `<dynamic>` site gets no `CALLS_NAME`, because no symbol is named `<dynamic>` and the `Name` node would resolve nothing (graph-dba suggestion). A `targets[]` id whose `Symbol` has no node in the snapshot (a `stale-kept` or `not-mapped` unit) gives no `CALLS` edge: the loader skips it and counts it, and the site's `CALLS_NAME` path stays (graph-dba suggestion; S17 confirms). | `nameKey` = `h(["name", lang, calleeName])` (§3.2, GV-19; graph-dba's BF14 ruling). `lang`/`name` stay readable and unconstrained. `path` comes from the enclosing symbol. | `ordinal`, `callKind`, `resolution`. `externalTarget` is a P5 input for `FROM_DEP`. |
| `valueSet` | **no node.** It is recognizer input. It becomes `ContractPoint.key`/`keyNorm` through `PROVIDES`/`CONSUMES`, and `LIFTED_VIA` evidence. | | everything |
| `constant` | **no node.** It resolves `{ref}` at assembly. | | everything |
| `data` | `(use)-[:DATA {fromPath, toPath, status, reason}]->(def)`: source `CallSite`/`Ret`/`Cond`, target `Param`/`CallSite` (§5.10's endpoint kinds; `Cond` is a use point, §5.1). The edge-load anchors must include `Cond`. | `status`/`reason` per graph note §3.3 ("every dependency edge"). §6.3's edge lists only the paths (BF13). | |
| `controlledBy` | `(point)-[:CONTROLLED_BY]->(:Cond)` | | |
| `decorator` | **no node.** Recognizer input. It yields `(:Symbol)-[:PROVIDES]->(:ContractPoint)` (routes, tools), `(:Symbol)-[:MARKED]->(:Mark)` (auth checks), and the `Test` label or kind. | | everything |
| `fieldAccess` | `(point)-[:READS\|WRITES {path, status, reason, linkId}]->(:ContractPoint {kind:'field', key, keyNorm})`, with `keyNorm` by §6 `field`. Two accesses at different `path`s on one point are two **parallel** edges with the same `linkId` (graph-dba's F1 ruling). | `path` is carried, because the BFS's `arg(head, path)` state needs it to pick the callee `Param`. | `line` (the point carries a line) |
| `import` | **no node.** Assembly input: `CONSUMES library_api`, `{importRef}` resolution, and `dynamic-import`. | | everything |
| `dependency` | `(:Unit)-[:DEPENDS_ON {version, manifestPath, line}]->(:Dependency {depId, ecosystem, package})` | | `scope` (BF13) |
| `testMarker` | the `:Test` secondary label on the `Symbol`, plus `Symbol.testKind` (set at assembly from `kindHint` and pack rules, graph note §6.4 FR-14) | | `framework`, `evidence` |
| `configEntry` | `(:ConfigEntry {entryId, repoId, path, line, keyPath, valueHash})`. Packs add `(:ConfigEntry)-[:WRITES]->(:ContractPoint {kind:'env_var'\|'config_key'})`. It is `BOUND_BY` and `Finding` `STEP` evidence. | | `format`, `bindValue`, `example` (BF8, BF13). There is no `File`→`ConfigEntry` edge in the graph note; the path is denormalized. |
| `launchFact` | a `ConfigEntry` node (`entryId` = the record id), with `(:ConfigEntry)-[:LAUNCHES]->(:File\|:Unit)` resolved at assembly; `BOUND_BY` evidence for URL binding | `path`, `line` | `kind`, `command`/`args`, `port`, `envVar`/`default`, `mountPath`: the binding inputs (BF13) |
| `schemaEntry` | `(:ContractPoint {kind:'data_entity'})-[:DEFINED_BY {ord}]->(:SchemaEntry {entryId, path, line, migration})` | | `op` |
| `secretSighting` (P5) | `(:SecretSighting {sightingId, fingerprint, path, line, commit, committedAt, removedInCommit, presentAtHead})` (graph note §6.4 FR-15) | | |

### 9.2 Graph elements no bundle record produces

These are listed so the mapping is checkably complete. Each is made at assembly, by an importer,
by verdicts, or by the store:
- `:Snapshot` and the `:Repo` name/`headCommit` (from the manifest);
- `Unit` nodes for never-built or failed units (`buildStatus` `stale-kept`/`not-mapped`, `failedAt`, `failReason`), and `Unit.componentId` (FR-11 grouping), all from the manifest and refresh state;
- `Unit.flowAvailable` (per-language configuration from P0's measurements; new against the graph note, for S17 to adopt);
- `CALLS {reason: 'verdict-confirmed'}`, which verdicts create alongside `DISMISSED`;
- external `Symbol` nodes for `FROM_DEP` (P5);
- `Link.paramMap` (graph note §6.1 item 5), and `LIFTED_VIA`'s per-site `siteId` (BF6);
- `ContractPoint` nodes, and `PROVIDES`, `CONSUMES`, `VIA`, `TO`, `ON`, `Link`, `BOUND_BY`,
  `LIFTED_VIA`, `TO_UNIT` and `PART_OF`;
- `READS`/`WRITES` other than fields (embedded SQL/Cypher, config values, packs);
- `MARKED`/`Mark` and `Finding`/`STEP`;
- `RAN`/`CoverageRun` (importers);
- `DISMISSED` (verdicts);
- `EMITS`/`LogTemplate` (P5 packs over call sites and value sets);
- `FROM_DEP` (P5);
- the coverage aggregates `staticCovered`, `runtimeCovered` and `coverageState`, and the risk
  inputs `callerCount` and `securityRelevant`;
- `ContractPoint.overlayState` and `Finding.delta` (overlay build).

## 10. Provisional items

**[P-slice]: the slice (S11–S14) is expected to exercise these. A mismatch is an SL-7 delta.**
- The Python, TS/JS and shell `qname` grammar (§4.0–§4.2), checked by re-deriving every slice
  symbol's `qname` from source. That includes v1.1's evaluation-scope rule for decorator
  lambdas (§4.1 rule 2) and `Protocol`/abstract stubs as symbols (rule 8; five in
  `falkor-chat/server/falkorchat`).
- The ancestor-plus-line locator conversion (§3.5), for whichever anchors S06 makes eligible.
- The vendored-path exclusion (§5.0), vacuously: the proving repo has no vendored path.
- Rule 8's TS row for `type`-alias method signatures (v1.2); the proving repo's TS has none at
  the time of writing, so the slice exercises it only if one appears.
- The access-path grammar, including `recv` and `kw.` (§5.1), and the depth cap.
- The `param` receiver index `-1` (§5.4).
- `ret`/`cond` ordinals and the definition of a condition (§5.5).
- `calleeName` and the call-site ordinal ordering (§5.6–§5.7).
- The value-set element forms and caps (§5.8), and `constant` (§5.9).
- `data` and `controlledBy` (§5.10–§5.11). S14 should also check the AC-1 rows of §5.10.
- `decorator` (§5.12), for the FastAPI decorator path.
- The symbol fingerprint normalization (§5.3).
- File `kind`/`role` (§5.2).
- The `http` `keyNorm` (§6), checked by SL-1 and SL-2's counterfactual `GET /x/api/health`.
- §8.5's exempt fields, checked by SL-6.
- The header fields (§2.2).

**[P-later]: the slice cannot exercise these.** If S15 freezes v1, they freeze with it, untested
(§1). The first phase that can test each:
- the Java, Kotlin, C#, Go and C/C++ grammar, §4.3's signatures, the `.h` language test and
  the non-Python receivers of §5.4 (P2), including v1.2's rule 8 table rows, Go interface
  specs, primary-constructor owned regions (rule 4) and the normalizer's substitution of a
  non-symbol target (rule 8);
- `fieldAccess` (P1 B1 fixtures, unless the slice emits it);
- `configEntry.bindValue` (P1, after the `security-expert` advisory) and `launchFact` (P1);
- `data_entity`, `config_key` and `schemaEntry` (P3);
- `library_api` and `rpc` (P2);
- `secretSighting` (P5).

## 11. Downstream encoders (P0 §5.3)

| Decided here | Encoded by |
|---|---|
| §3.1 hash, §3.2 tuples, §3.6 golden vectors | S09 `cpg/ccg/ids.py`, `cpg/tests/test_ids.py` |
| §2, §5, §7.3 and §8 (records, order, strictness) | S10 `bundle-v1.schema.json`, `bundle.py`. S10 also adds a test that writes two `fieldAccess` records differing only in `path` and gets two ids (analyst F1). |
| §4 `qname`s, §5 records and §6 `http` keys for the slice | S11 `extract.sc`, `normalize.py` (re-derivation check, §4.0) |
| §3.5 locators, §4, §6 and §7 reason codes | S35 fixture rows, S39 `fixture-manifest-v1` ("qname found in file" by §3.5's prefix rule) |
| §3.4 `repoId` `graphmind-ai-lab`, §3.5 locators, §4 grammar (including the macro-name exclusion), §6 `keyNorm` | **S06** item schema and matching rule, **S40** kit and briefs, and the S52 key-writer definitions, which must quote §3.5, §4 and §6 to the writers. **S06 decides whether `<anon>`/`#k` scopes are eligible as key and gold anchors** (coordinator ruling). If they are, writers use §3.5's ancestor-plus-line form; macro-generated names are never eligible. |
| §7 registry | S33's answer `reason` values, S35's expected reasons |
| §9 mapping | S17 graph model v1 |

## 12. Findings

These are disagreements or gaps found while writing this spec, and during its S03 gate. They are recorded, not resolved
silently. The P0 plan's F1 (candidates never dropped) and F2 (no `unitId` in `symbolId`) are
**followed** (§5.6, §3.2).

| # | Where | Finding | Disposition in this spec |
|---|---|---|---|
| BF1 | P0 §4.2 tuple table vs D10 | The table's tuples carry no type tag. Then `callsite (symbolId, "param", 0)` and `point (symbolId, "param", 0)` hash equal, and both are `DATA` endpoints matched by id at edge load. | Every tuple's first element is the record-type tag (§3.1). This is consistent with D10 ("the tuple array"). S09 encodes it. |
| BF2 | graph note §3.4 (`Symbol.qname` index "anchor: qualified name, stack frames") | This spec's `qname` is path-anchored (§4.1 rule 1), not a runtime's native qualified name. An exact `qname` lookup does not match a JVM or .NET frame's `com.acme.Foo.bar`. | Not a v1 blocker: every FR-1 runtime's frames carry file and line, so frame matching can use the `File.path` suffix plus line. If P5 wants native names, a non-identity `nativeQname` symbol field is an additive minor version (§8.6). For P5 and S17 to decide. |
| BF3 | P0 §4.2 `data {from, to, fromPath, toPath}` vs graph note §6.3 | In §6.3, `fromPath` is the path on the edge's **target** (def) and `toPath` the one on its **source** (use). Fields named `from`/`to` would read as the opposite. | The node fields are `use`/`def` (§5.10); the edge property names are unchanged. **Accepted by `graph-dba` (S03 check).** `teco` records in the ledger that P0 §4.2's `{from, to}` is superseded here. |
| BF4 | graph note §6.3 path grammar | It has only `p<i>`, `a<i>`, `r` and `ret`. Receivers (`router.get`) and keyword arguments (`prefix=`, which SL-2 needs) are not expressible. | Adds `recv`, `kw.<name>` and `["…"]`-quoted field segments (§5.1). **Accepted by `graph-dba`.** S17 records the graph side: §6.1's `arg(i, path)` becomes `arg(head, path)`, binding `a<i>` by `index` after the receiver, `recv` to index `-1`, and `kw.<n>` by `Param.name` (including `**kwargs`). |
| BF5 | P0 §4.2 record list | It lacks the record that resolves `{ref: qname}` (no value-set source for `API_PREFIX`), the input for field `READS`/`WRITES` (graph note §6.3), and the input for `DEFINED_BY`/`SchemaEntry` (proposal §3.2 SQL extractor). | Adds `constant`, `fieldAccess` and `schemaEntry` (§5.9, §5.13, §5.18). |
| BF6 | proposal §3.8 `linkId = h(kind, keyNorm, providerSymbolId, consumerSymbolId)` vs graph note §3.4 `(…, providerId, consumerId)` | (a) A provider or consumer may be a `ConfigEntry`, which has no symbol. (b) Two consumer call sites in **one** symbol on one key give one `linkId`, so they share one `Link`, whose `LIFTED_VIA` chains may differ per site. | **Ruled by `graph-dba`.** (a) The anchor is the enclosing `symbolId`, or the `entryId` (§3.2). (b) One `Link` per consumer **symbol**, for verdict stability. Several consumer `CallSite`s `VIA` it, and per-site status stays on each `CONSUMES`. `LIFTED_VIA` carries `{ord, siteId}` (the consumer `callSiteId`), and `Link.status` is the best over its sites. S17 writes both. The tuple is unchanged. |
| BF7 | graph note §3.2 `File.kind` vs §6.4 `File.role` | The two overlap (`config`, `script` in both). §6 does not say whether `role` replaces `kind`. | **Ruled by `graph-dba`:** they are orthogonal. `kind` is the extractor family and `role` the purpose. `role` is narrowed to `{primary, example, doc}` (§5.2), so they cannot disagree. Both go on `File`, unindexed (S17). |
| BF8 | graph note §3.2 `ConfigEntry.valueHash`; proposal §3.4 URL/env binding | (a) A plain hash of a low-entropy config secret is brute-forceable, the graph note's own argument for HMAC fingerprints (§6.4 FR-15). (b) URL, env and Compose binding need some values in clear (`http://localhost:8000/mcp`, `8200`), and the schema has no field for them. | `valueHash` is an HMAC; a closed `bindValue` rule exposes only binding-shaped, non-secret-named values (§5.17). **[P-later].** Recommend a `security-expert` advisory pass on §5.17 before P1 relies on it (proposal §6.2 allows advice; not a gate). |
| BF9 | graph note §6.4 FR-13 (`library_api` key = "qualified name + signature hash") | A hash cannot be written by a key writer with only source and no tooling (M2's premise). | Readable signature in `keyNorm` (§6). It still pairs "old removed + new added" by the qualified name. **[P-later]** |
| BF10 | graph note §3.2 (`data_entity` CPs), §6.3 ("the bound store's label CPs") | `keyNorm` has no store or database scope. Two databases' `users` tables are one contract point, and SQL schema defaulting (`public.`) is dialect-specific. | v1 `keyNorm` is unscoped, with the schema present only when written (§6). P3 decides store scoping; that will be a **major** change if it enters the tuple (§8.6). Flagged now so P3 is not surprised. |
| BF11 | P0 §4.2 symbol tuple `(lang, …)` with `lang` per unit | A mixed C/C++ unit would put signatures on C functions, contrary to proposal §3.8 (C does not overload). | A symbol's `lang` is its **file's** token (§3.3). |
| BF12 | P0 §4.4 (S11 stages with `git archive`) vs proposal §3.3 (`lastChanged*` from git) | The sandboxed tree has no `.git`, so `git blame` cannot run inside it. | The normalizer computes `lastChanged*` outside the sandbox from the source repository; `header.history` records availability (§2.2, §5.3). S11 sequences it. |
| BF13 | graph note §3.2/§6 property lists | Bundle facts with no graph property: `Symbol.complexity` (proposal §3.3 names it), `signature`, `lang`; `File.contentHash`; `Dependency` `scope`; `DATA.status/reason` (§3.3 requires them, §6.3 omits them); and the launch-fact binding inputs. | Carried in the bundle (§9.1 "not carried" column). **S17 decides, and that is correct, because the bundle already carries them** (`graph-dba`). `graph-dba`'s leanings: carry `complexity`, `signature`, `lang`, `File.lang` and `contentHash`; put `scope` on `DEPENDS_ON`. Store `DATA.status`/`reason` only when not `confirmed`/`intra-flow`: measured at +56 bytes per edge, +42 % on a probe graph. **The bundle always carries both fields** either way. |
| BF14 | graph note §3.2 `Name.nameKey` (`py:get`) and §3.4 | (a) The token differs from §3.3 (`python`). (b) `nameKey` is `UNIQUE`-constrained but not hashed, against §3.4's own rule that every constrained id is a hash. A shell `calleeName` is unbounded, and a value over 4,096 bytes crashes the instance at the constraint check. | **Ruled by `graph-dba`, adopted now:** `nameKey = h(["name", lang, calleeName])`, an assembly id in §3.2 with golden vector GV-19. `lang`/`name` stay readable and unconstrained (§9.1). |
| BF15 | egress spec §5.3 ("a bundle whose `auditId` has no record", singular) vs P0 §4.4 (S11 runs `build-cpg.sh` **and** `extract.sc` through `offline-run.sh`) | One bundle can come from two or more sandbox invocations, each with its own `auditId`. Also, egress §5.3 passes `auditId` to the tool "as the output file name stem", but the bundle file is named `<unitId>.bundle.jsonl` (§2.1) and may be written by the normalizer outside the sandbox. | The header carries `sandboxAuditIds`, an array (§2.2); the stem convention applies to the sandboxed tool's own outputs, not to the bundle file. The egress spec's author is rewording its §5.3 to `sandboxAuditIds` in parallel; this spec does not touch it. `teco` reconciles at the gates (egress R6). |
| BF16 | proposal §3.8 (orphan causes: rename, move, signature change) vs §4.1 rules 5 and 7 (ordinals) | Inserting an earlier anonymous or duplicate sibling **re-binds** `<anon>#N`/`#k` to a different element. That element's id survives, so FR-29's fingerprint check reports "possible, earlier verdict noted", but the note is about another function. Proposal §3.8's orphan list does not cover this (analyst F3). | Not fixable in an id that has no source anchor other than order. Recorded for `architect` to carry into proposal §3.8 as a fourth case, "re-binding on sibling insertion", with the mitigation that the verdict log already stores both end fingerprints (proposal §3.8) and P4 can compare them. For ground truth there is no defect, since `EVAL_COMMIT` pins both sides (D8). |

## 13. S03 gate dispositions

### 13.1 Pass 1 (v1.1)

The source is `docs/reviews/connected-code-graph-p0-impl.md`: the `## S03 (P0-03a)` section
(analyst, *needs changes*) and its graph-dba check (*approve with suggestions*). Each recommended
fix was judged on its merits. Where v1.1 departs from a fix as worded, the reason is given.

| Finding | Disposition | Where |
|---|---|---|
| F1 (blocker) `fieldAccess` id omits `path` | **Fixed as recommended.** `path` is in the tuple. GV-20/GV-21 differ only in `path`. S10 tests two records. Graph side: `path` goes on the edge (graph-dba). | §5.13, §3.6, §9.1, §11 |
| F2(a) `Protocol`/abstract stubs | **Fixed, and widened.** The analyst's fix covers Python. Since Joern resolves calls to interface and abstract methods in every typed language too, rule 8 now makes any member declaration a call can resolve to a symbol, in every language. A Python-only fix would leave the same dangling `targets[]` in Java, Kotlin, C#, TS and C++. | §4.1 rule 8 |
| F2(b) decorator and default lambdas | **Fixed.** I chose **evaluation scope**, because that is what the runtime does and what Joern models (the decorator call lives in the enclosing method). Python defaults go to the enclosing scope; call-time defaults (JS/TS, Kotlin, C#, C++) go to the function. | §4.1 rule 2 |
| F2(c) TS string keys | **Fixed as recommended:** only identifier-valid keys name a value. | §4.2 TS |
| F2(d) named function expressions | **Fixed.** A named function expression takes its **own** name, which matches JS's `.name` and stack frames, and makes the spec's appeal to ES2015 inference accurate. | §4.2 TS |
| F3 ordinal locators | **Fixed as recommended:** the ancestor-plus-line locator, converted mechanically. Eligibility is S06's decision. The re-binding is recorded as BF16, its own finding rather than a line in BF2, since BF2 is about stack frames. | §3.5, §11, BF16 |
| F4 P-later errata | **Fixed, every item.** Bodyless accessors are not symbols, and the C# example now has bodied accessors. The Kotlin primary constructor is a symbol by rule 8, and `init {}` goes to the type. The C# extension `this` is removed from signatures. Explicit interface implementations use `explicit(I)`. Go, Kotlin and C# receivers are `-1`. `.h` language comes from the **header's own text**, not the unit's file list. I rejected the alternative "always `cpp`", because it would put signatures on C headers' functions (BF11). Extensionless shell takes its `lang` from the shebang (§3.5). | §3.3, §3.5, §4.1, §4.2, §4.3, §5.4 |
| F5 two `data` encodings | **Fixed.** A `Ret` use point is always `ret[.f…]`, never `""`, which flips the AC-1 row. I chose `ret`, not `""`, because a field-carrying return (`return {path: p0}` → `ret.path`) needs a head that `""` cannot carry, and the graph note's grammar already lists `ret[.f…]`. `""` remains only on `cond`. | §5.1, §5.10 |
| F6 `bindValue` URL secrets | **Interim narrowing applied** (coordinator ruling). The path class is dropped. URLs need no userinfo, query or fragment, and a path of at most one `[a-z0-9_-]{1,32}` segment. The name list gains `pwd`, `dsn`, `conn`, `hook` and `sig`, as a substring match. I kept the URL class rather than dropping it: P1's MCP URL binding (`http://localhost:8000/mcp`, AC-4) needs it, and the one-segment form excludes the Slack-webhook case. **Residual risk for the advisory:** a single 32-character token segment is still admitted. | §5.17 |
| F7 vector gaps | **Fixed:** GV-15 (C in a `cpp` unit, 4 elements), GV-16 (Kotlin extension), GV-17/GV-18 (the receiver `-1` as an integer vs a string), GV-19 (`nameKey`). All 21 were recomputed. GV-1–GV-14 are unchanged. | §3.6 |
| F8(a) `import.names` tie | **Fixed as recommended.** | §5.14, §8.1 |
| F8(b) fingerprint splitting | **Fixed as recommended:** split on `\n` only, strip by the §3.1 set. | §5.3 |
| F8(c) shallow history | **Fixed:** `history` must be `available` in P0, and the history depth is listed as an input. | §2.2, §8.4 |
| Q (open question 1) | `Protocol`/abstract stubs **are** symbols (F2(a) above). | §4.1 |
| graph-dba: `path` on the field edge | **Adopted.** | §9.1 |
| graph-dba: §9.2 gaps | **Adopted:** unbuilt `Unit`s, `componentId`, `flowAvailable`, `verdict-confirmed` `CALLS`, external `FROM_DEP` symbols, `paramMap`, `LIFTED_VIA.siteId`. | §9.2 |
| graph-dba: `nameKey` tuple and vector | **Adopted.** | §3.2, §3.6 |
| graph-dba: narrow `File.role` | **Adopted:** `{primary, example, doc}`. | §5.2 |
| graph-dba: quote field segments | **Adopted:** `["…"]` with JSON escaping. | §5.1 |
| graph-dba: unbuilt `targets[]`, `<dynamic>` | **Adopted** as the stated rule. S17 confirms the loader side. | §9.1 |
| graph-dba rulings BF3, BF4, BF6, BF7, BF13, BF14 | **Applied** to the findings table. | §12 |
| graph-dba: `UNIQUE` on `Dependency.depId`/`entryId`; the `TIMEOUT_MAX` write cross-check; the stall incident | **Not for this spec.** They are graph and load-side matters for S17 and S23, and the ids are already hashes. | — |

### 13.2 Pass 2 (v1.2)

The source is the same review's `### Pass 2` under `## S03 (P0-03a)` (analyst, *approve with
suggestions*), plus open question 2 of the S05 section's `### Pass 3`, which `teco` routed here.
Each suggestion was judged on its merits. Where v1.2 departs from it, the reason is given.

| Finding | Disposition | Where |
|---|---|---|
| N5 Go interface method specs | **Fixed: they are symbols**, `IfaceType.NAME` under the interface type, `kind: method`; an embedded interface adds no segment and no spec, so a call through it targets the embedded type's own `qname`. Rule 8's criterion covered them, and excluding them would have split one fact by language in S26/S27. Specs in an *unnamed* interface literal stay non-symbols: they have no source name to key on. GV-22 added. **Rule 8 is now an exhaustive table** ("these and nothing else"). Writing it closed further gaps the reviewer did not list, all decided the same way: Java `native`, Kotlin `external`, C# `extern` and Go bodyless functions are symbols (each is the entity's only declaration in its language, and calls resolve to it); TS method signatures in a `type`-alias object literal match the `interface` rule; C# primary constructors and Java record headers join the Kotlin primary constructor; Kotlin `expect` and C# partial defining declarations are redeclarations, so not symbols. A resolution into a non-symbol is substituted or dropped by a stated normalizer rule. `docs/plans/connected-code-graph-p0-ml.md` §2's `N(s)` example should move Go interface specs to `T(s)` as `decl`. | §4.1 rule 8, §4.2 Go, §3.6 |
| N2 three stale cells | **Fixed, each confirmed against §5.1 and the rulings.** §5.10 `fromPath` no longer lists `""` (§5.1: no `defPath` is `""`; `recv` added). §9.1 `data` names `Cond` as a use endpoint, and the endpoint kinds are stated once in §5.10. graph-dba's anchor check already counted `Cond` among the point labels. §9.1 `file` cites §5.2's three `role` values (BF7). | §5.10, §9.1 |
| N4 Kotlin `init {}` vs the primary constructor | **Fixed, but not with the suggested span.** `init {}` blocks and property initializers belong to a *declared* primary constructor, in Kotlin and C# (where field and property initializers run in it). The reviewer's "use the class body as the span" was rejected: the span would contain every method, so any method edit would change the constructor's fingerprint (FR-29) and every line in the class would locate to it. Instead the span stays the parameter list, and the initializers are **owned regions**, counted in `fingerprint` and `complexity` and accepted as locator lines. This reverses v1.1's F4 rule-4 erratum for that case. A C# primary-constructor parameter used in a method is a capture; that remains a known limit. Whether Joern's frontends model init code as the constructor's body is not verified. | §4.1 rule 4, §4.2 Kotlin/C#, §5.3, §3.5 |
| N1 C++ out-of-line exception | **Fixed: a one-file rule exists.** Pure virtuals and `= default` are symbols; every other bodyless in-class declaration is not, and its out-of-line definition is. **Departure:** `= delete` is *not* a symbol, against the suggestion, because no well-formed call resolves to it, so making it one adds records no edge can reach. A pure virtual with an out-of-line body gives two ids at two paths, as suggested. | §4.1 rule 8, §4.2 C/C++ |
| N3 `.h` text test | **Partly fixed, with the rest recorded as a known limit.** `[[…]]` attribute contents are excluded, which fixes the C23 case. The excluded region is now every arm of any conditional group whose condition mentions `__cplusplus`. C++-only keywords that are not C23 keywords are added, including `operator`, along with `extern "C"`. **Not adopted:** the `&`-reference-parameter marker. It is not a token test (`&` is also C's address-of and bitwise-and operator), and it would need a declarator parse. **Cost of the limit:** a marker-free C++ header (free functions with references or overloads) is read as `c`, so its functions lose their signatures and its overloads become `#k` duplicates that re-bind on reordering. The ids are still deterministic. P2 measures this. | §3.3 |
| P2-n1 locator ancestor | **Fixed.** `within` is taken by evaluation scope, with the `test_fusion.py:241` example. `line` and `nth` use the introducing token's line and column, and that token is defined for every language. | §3.5 |
| P2-n2 trailing `/` | **Fixed: admitted.** `/mcp/` qualifies, since a trailing slash carries no information. Multi-segment paths stay excluded until the `security-expert` advisory, which still covers the whole URL class. | §5.17 |
| S05 Pass 3 Q2 vendored code | **Decided: external, consistent with the ML note.** Vendored directories are a path-segment test with a default list that the manifest can replace. Vendored files are still parsed but yield no records, and a call into vendored code is `external`. So P0's fan-out and precision figures describe P1's production population, and nothing needs re-reading. The effective list is recorded in `header.vendoredDirs`. `secretSighting` (P5) is exempt. **Rejected:** making vendored code `Symbol`s with a role flag. That would put vendored candidates in name-match, and it would reopen graph-dba's BF7 `role` ruling. | §5.0, §2.2, §8.4 |

### 13.3 Pass 3 (v1.3)

The source is the same review's `### Pass 3` under `## S03 (P0-03a)` (analyst, *approve with
suggestions*). This is the final static round, so each change is the narrowest one that closes
its finding, and no rule was added beyond these six.

| Finding | Disposition | Where |
|---|---|---|
| P3-1 owned regions invisible to the store | **Fixed as recommended.** A `symbol.ownedRegions` field (sorted `[lineStart, lineEnd]` pairs, present exactly when non-empty); `lastChanged*` and `fingerprint` both read span plus `ownedRegions`. S17's anchor rule: the narrowest containing interval, span or owned region. **One addition the suggestion lacked:** a tie-break (a span match beats an owned-region match, then the deeper `parentId` chain). Without it, a one-line lambda inside a one-line `init {}` ties with the constructor's region. The field feeds no tuple, so there is no new vector. | §5.3, §9.1, §4.1 rule 4 |
| P3-2 C# static initializers | **Fixed, more narrowly than suggested.** Only *instance* initializers go to the primary constructor. `static` ones stay with the type under rule 4's default, not with a declared static constructor as the reviewer proposed. A static constructor has no parameters, so there is no N4-style flow to recover, and giving it owned regions would be a new rule. | §4.1 rule 4, §4.2 C# |
| P3-3 Java record implicit accessors | **Fixed, using the second option with a different `qname`.** Each component with no explicit accessor yields an implicit accessor symbol. Its `qname` is `R.x` with signature `()`, which is Java's ordinary method form, **not** `R.x.<get>`: an explicit `x()` declared later then keeps the same id instead of re-iding every caller's target. Its span is the component's header entry. The `fieldAccess` option was rejected: it would turn a syntactic call into a non-call record. No new id form, so no new vector. | §4.1 rule 8 (Java row) |
| P3-4 vendored test path base | **Fixed: the test reads segments below the unit's `rootPath`.** The list stays **per repo**, applied to each unit. Per-unit lists are rejected as unneeded once a unit's own root can never match. `header.vendoredDirs` wording aligned. | §5.0, §2.2 |
| P3-n1 Go overlapping embeds | **Fixed as recommended:** the call targets every spec of that name in the embedder's method set (the never-miss reading). | §4.2 Go |
| P3-n2 `resolution` after a drop | **Fixed as recommended:** `resolution` stays `resolved`, so "resolved with empty `targets`" identifies a resolution to a non-symbol, and S25/S27's denominators don't move. | §4.1 rule 8 |

## Ready to implement (summary)

Spec: `docs/plans/connected-code-graph-bundle-spec.md` (v1 candidate; freezes by §1).
- **S09:** §3.1–§3.2 plus the 22 golden vectors in §3.6.
- **S10:** §2, §5 and §8, with 19 record types plus the header, strict validation, and
  byte-deterministic writes with two exempt header fields; vendored files yield no records (§5.0).
- **S11:** emits §5 and re-derives every `qname` from source by §4. A mismatch is SL-7.
- **S06/S40:** use §3.5's locators, the `repoId` `graphmind-ai-lab`, §4's grammar and §6's
  `keyNorm`. Both need no snapshot.
- **S17:** absorbs §9 and findings BF2–BF16.
