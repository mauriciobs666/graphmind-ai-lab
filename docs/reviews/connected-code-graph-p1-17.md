# Review: P1-17 dispatch 1, linker matchers and link core

> **Status:** active · **Owner:** `analyst` · **Tracks:** C-1001 (P1) · **Reviews:** `docs/plans/connected-code-graph-p1-linking.md` P1-17

## Scope & verdict

Scope: the uncommitted dispatch 1: `cpg/ccg/linker/{__init__,matchers,link}.py`, `cpg/tests/{linker_helpers,test_linker_matchers,test_linker_link}.py`, the `reasons.yaml` row, the one-line `seeded_registry()` amendment in `cpg/tests/test_rule_pack_schema.py`, `cpg/README.md`. Read against the plan's P1-17 step, `-p1-graph.md` §1-§2, `-p0-graph.md` D2/D6, `-bundle-spec3.md` BF6/§5.6, fixture rows LNK-*, root `REVIEW.md` (no `cpg/REVIEW.md` exists). `bind.py`, the conformance test and the leaf-harness edit are dispatch 2 and not faulted.

**Verdict: needs changes.** One blocker (the `Link.key_norm` contract, which feeds `linkId`), three majors. The matching, status, O-9 and mention logic is otherwise sound and well tested.

CPG: considered, not relevant — the linker is new, untracked code, so the loaded graph cannot describe it; read directly.

Run: `tests/test_linker_matchers.py tests/test_linker_link.py tests/test_rule_pack_schema.py` 810 passed; `tests/test_answer.py test_render.py test_fixtures.py` (the other `reasons.yaml` readers) 856 passed. A whole-`tests/` run exceeded the tool limit and was killed, so it is **unrun**. 22 mutants (scratch copy, fresh per mutant, linker tests only): 21 killed, 1 survivor (F5 below). One of mine (a tie-break rewrite) was equivalent on the test ids; a stronger rewrite was killed.

## Findings

### B1 (blocker, verified) · `Link.key_norm` is the consumer's key; the oracle and the graph rulings make it the provider's

`link.py:293-296`: `lid = link_id(kind, <consumer key>, provider anchor, consumer anchor)`, and the test at `test_linker_link.py:33` asserts it ("the link key is the consumer's"). The slice oracle does the opposite: `spikes/vertical-slice/assemble.py:611` `link_id("http", p["key"], p["handlerId"], site["symbolId"])` with `"keyNorm": p["key"]`. `-p0-graph.md` D6 says "a Link is one provider ... **one Link can be reached by sites with different consumer keys**", and D2 keeps `consumerKey` on `VIA` precisely because the Link no longer holds it; `Link` is `ON` a `ContractPoint` (provider keyNorm). Run: one symbol, sites `GET /a/1` and `GET /a/2`, provider `GET /a/{}` gives **two** Links (`('GET /a/1','P',['s1'])`, `('GET /a/2','P',['s2'])`); the oracle gives one. Consequences: the dispatch-2 conformance ("same hops") compares against different `linkId`s; D6's `candidateCount` derivation (equal `VIA.consumerKey` across Links of one symbol) is built for one Link per provider; and decision (c) (`''` for keyless consumers, plus a test "never merges with a keyed site's link") is only needed because of this choice: with the provider key a keyless and a keyed site reaching one provider share one Link, as D6 intends.
Fix: key the Link by the provider's `keyNorm`; for a never-built unit (no provider) the consumer's key is the only one, and `''` for keyless there. BF6's wording ("one key") is ambiguous, so if `architect` rules the consumer key instead, D6 and the slice must be amended; either way get the ruling before `bind.py` and `assemble` build on it. Rewrite `test_linker_link.py:33` and `:391` accordingly.

### M1 (major, verified) · regex keys are not safe from catastrophic backtracking; the docstring says they are

`matchers.py:118-124`, docstring lines 14-15: "applied only inside the RE2 subset". `regex_problem` (`ccg/rulepack.py:182`) rejects lookaround, backreferences, atomic/possessive and length > 256, but **not nested quantifiers**. Run: `match_keys('mcp_tool','regex','(a+)+b','a'*N+'c', provider_form='regex')` takes 0.17 s at N=22, 0.70 s at 24, 2.86 s at 26 (about 4x per two characters), so ~40 characters hangs the build. The pattern and the text are both repo-derived, and `re.fullmatch` has no timeout. Fix: do not run Python `re` on repo-derived patterns without a bound. Options: refuse nested/ambiguous quantifier shapes, cap the text length, or use an RE2 engine if a dependency is acceptable; a test with `(a+)+b` against 40 `a`s that must return in well under a second. The same hole is in `regex_problem` for any other caller; say so to `architect`/`coder`.

### M2 (major, verified) · the `Bindings` seam cannot carry `ConfigEntry` anchors or a weaker chain

The brief's question: "can ConfigEntry-anchored providers/consumers be given units?" No, as built.
- `_Index.unit/role/name` (`link.py:196-213`) look anchors up in `corpus.symbols` only. A provider anchored on an `entryId` has unit `None`: with any `Binding` it is silently dropped (run: provider `ENTRY1`, binding to `svc` gives `[]`), without a binding its role is `None`. LNK-03/21/22/28 and the Compose/`.mcp.json` rows need these anchors; `packrt` only ever anchors on symbols (`run.py:117`), so `bind.py` must synthesize them, and nothing here lets it say which unit they belong to.
- A `configEntry` record carries `example` (schema) and `repoId`/`path`, no `unitId` or file role, so `(7)` example-config (`example-config` on an `.mcp.json.example` entry) is invisible for these anchors.
- `Binding` has no status or reason of its own: a possible hop inside the chain (the inferred Compose rule of plan (5), trust rule (e)'s loopback residual) cannot weaken the site, and trust rule (b)'s "outside every manifest repo gives `possible`/`target-not-mapped`" has no unit to put in `unmapped_unit` (it must be a string for `link_id`).
Fix: let `_Index` resolve a config-entry id (a `config_entries` map by `id` with `example` and a unit/path-to-unit resolver, or a `Bindings.anchor_unit(anchor)` method) and add an optional `Binding.weaken: tuple[status, reason]` plus an unmapped marker that does not need a unit id. Do it in dispatch 2 but pin the seam now with a test using a config-entry provider.

### M3 (major, verified) · the registry-reason test cannot fail on the thing it names; the implemented reasons exceed the registry

`test_every_link_reason_is_a_registry_code_with_the_registry_status` (`:365`) checks code existence and `status` only, never the `on` list. `link.py` emits `shell-dynamic` and `prompt-mention` as `Link`/`VIA` reasons, and the registry lists them only on `["shell DATA", callsite]` and `[callsite (mention), CONSUMES]`; `key-match`, `partial-key`, `example-config` and others are not on `VIA` either (only `binding-confirmed` gained it). Independently found: a confirmed lifted consumer (LNK-10/11/12 expect `lifted-role`) gets `Link.reason` `key-match`, because `_causes` never reads a confirmed consumer's reason (`:251-252`); `lifted-role` is registered on `CONSUMES` only. The test at `:405` passes because it compares status. Fix (after the architect's ruling): assert `("Link" if link else "VIA") in reg[reason]["on"]` for every emitted reason, which then fails for the unregistered ones (planted: any of the above). Whether LNK-10..12 keep `lifted-role` is the architect's call; as shipped the code never emits it on a Link.

### Minor

- **F5 (verified, mutant survives)** `matchers.py:63` `{**}` accepted only in the last provider segment: dropping `i != len(ps) - 1` leaves all 106 tests green. Add `GET /a/{**}/b` vs `GET /a/x/b` (no match) to the rejects table.
- **F6 `packs=` / ValueError (decision b): accepted.** Failing loud beats dropping a fully unknown consumer, and the plan's `link(facts, bundles)` shape is only extended by a keyword. The message `needs packs=` is wrong when packs were passed but name no rule for the lifting; say so ("no rule of the given packs consumes lifting X").
- **F7 never-built unit `linkId` (decision g): acceptable.** The unit id stands in as the anchor (`link.py:291-294`, no collision: ids are domain-tagged tuple hashes), but BF6 names only symbol and `entryId` anchors; add one line to spec3 BF6 or the P1-18 plan so the assembler's `TO_UNIT` writer expects it.
- **F8 reason precedence (d), (e): accepted, with a note.** Only `shell-dynamic` over `unknown-key`/`runtime-built` is specified; `example-config` over `shell-dynamic` and `target-not-mapped` over `port-shared` are the implementer's, documented in the docstring and tested (`:212`). `_own` collapses an unknown consumer reason (for example a weakened `lifted-role`) into `partial-key`, so the lifting cause is lost on the `Link`; acceptable, but note it in the docstring.
- **F9 scale.** `link()` is O(consumers x providers of the kind) with a regex/template match per pair; fine for the slice, check it against the (11) proving-repo count in dispatch 2.

## Judged decisions

- **(a) status.** Not a conflict. `-p1-graph.md` §2, `-p0-graph.md` D2 and BF6 all say `Link.status` is best over its sites; the plan's "weakest hop" describes one site's chain. The code (weakest per site, best per link) matches the governing texts and mutants M3/m3 (flip to worst) are killed. Ask `architect` to reword the plan sentence so a reader does not take it as weakest over sites.
- **`seeded_registry()` amendment:** legitimate. It is the test's pin to the spec-derived registry, the ruling (O-9) and its source are cited in the line, and it is non-vacuous in both directions (the test compares shipped and seeded both ways, so reverting either the YAML row or this line reddens it; I read this, I did not run the revert, the scratch copy lacks the spec docs).
- **(c) `''` key:** see B1.

## What's solid

The O-9 per-site rule is correctly built and well mutated: dropping the `siteId` condition, deriving the `Link` reason from any `BOUND_BY`, flipping the unit filter, dropping `port-shared`, and wrong/global `ord` all go red. Also killed: skipping the doc filter, shell-dynamic, provider example-config, name-only matching, `target-not-mapped`, the `./` strip, F-13 normalization order, regex full-match, the keyless merge, and the key-mismatch passthrough. `matchers.py` reuses `normalize_path_rfc`, never re-normalizes a `keyNorm`, and tests the `%2%45` case. The hand-off (a) (leaf only in `lifted`) is correct and tested against the real fetch and shell packs; LNK-20 runs over the real extractor. Output is deterministic and independent of fact order (tested).

## Open questions

1. (`architect`) B1: the provider or the consumer key in `linkId`/`Link.keyNorm`; the answer decides dispatch 2's conformance.
2. (`architect`, concurrent) the registry gaps of M3: this review found, independently, the unregistered `Link`/`VIA` reasons and that `lifted-role` is never emitted on a `Link`.
