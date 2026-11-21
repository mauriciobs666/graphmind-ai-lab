# Connected code graph, P1-57a and P1-57d: implementation gate

> **Status:** active · **Owner:** `analyst` · **Tracks:** P1-57 (57a, 57d)

## Pass 1

**Scope.** Commits `b7870153` (57a), `42a91574` (two fixtures moved to JS/TS file records), `85bc0e3d` (57d), against
`docs/plans/connected-code-graph-p1-linking.md` P1-57 and `docs/plans/connected-code-graph-coordination.md`; root `REVIEW.md` applied.
Executed in scratch worktrees (`git worktree` of HEAD and of `b7870153^`), never in the user's tree.
`CPG: considered, not relevant — the questions are regex/test-mutation/linker-output facts answered by execution; the cpg_* graphs do not model a pack's regex or a fixture build`

**Verdict: approve with suggestions.** No blocker. Questions 1-4 below answered by execution; one major (no test ties the
regex to the extractor's extension set), the rest minor.

### Answers to the four questions

**Q1, scope regex (verified, run through `run_packs` with the shipped pack).** `.*\.(js|jsx|cjs|mjs|ts|tsx|cts|mts|vue)` is a case-sensitive
`re.fullmatch` (`ccg/packrt/match.py:90`). Accepts: all 8 extensions the extractor emits (`extractors/joern/normalize/javascript.py:18`
`JS_EXTS`, identical to `ccg/units.py:52`), `.d.ts`, `.d.mts`, `x.ts/y.ts`, `x.tsx.ts`, `.ts`, spaces, non-ASCII, backslash paths.
Rejects: `.TS`/`.JS`/`.Tsx`, `x.ts/y`, `a.ets`, `a.mjsx`, `a.ts#`, `a.cjs ` (trailing space), `a.ts\n`, `a_tsx`, `foots`. Regex mutants, each run against
`tests/test_pack_fetch_scope.py` and the schema pin: drop `mts`, `cts` or `vue` -> red in both; add `py` -> 6 red; unescaped `.` -> red
(`foots`, `a_tsx`); drop `.*\.` -> 12 red. All killed; the one survivor (a redundant `d\.ts` alternative) is regex-equivalent.
Two observations, no defect: **(a)** `.vue` is unreachable: neither `JS_EXTS` nor `_SOURCE_EXT` emits it, so the alternative is dead and
`SEEDS`' `app/Page.vue` pins a shape no extractor produces. **(b)** `ccg/units.py:185,285` lowercases the extension when classifying a file while the
normalizer's `endswith(JS_EXTS)` and this regex are case-sensitive; `A.TS` is classified JavaScript yet yields no records, so no fetch is lost, but the two layers disagree on case.

**Q2, 42a91574 fixtures (verified by mutation).**
- *4096-byte drop* (`tests/test_indexed.py:395`): still exercises what it names. Mutants in `ccg/assemble/rows.py`: remove the provider `cp-unindexable`
  drop -> red; replace the `link-cp-unindexable` `continue` with `pass` -> red; fixture symbol returned to a `.py` file -> red (the Link vanishes). Remove the
  consumer-side drop -> still green, because the oversize consumer key is `None` (`consumer-key-none`), not an unindexable one: pre-existing, the same at `b7870153^`, and
  not something 42a91574 changed (see N1).
- *Sort-order pair* (`tests/test_packrt_run.py:482`): same kill profile as at `b7870153^`. Removing only the decorator sort (`run.py:62`) or only the output sort
  (`run.py:220`) survives, removing both is red, in the old and new test alike; the test's docstring says "pins the pair", so the claim holds. The `fts` file-id
  remap keeps the `fetch` chain on a `.ts` file and the test still asserts 2 consumers.

**Q3, 1521 removed Links (verified on `before.json` / `after2.json`).** Four definitions, all agree, plus a per-site count:

| Definition | Removed | Added |
|---|---|---|
| whole canonical-JSON record | 1521 | 0 |
| link id | 1521 (no id changed content: 0 records differ under the same id) | 0 |
| (consumer symbol, provider) pair, fields 3-4 | 1521 | 0 |
| per (link, `VIA` site) | 1872 (every one belongs to a removed link) | 0 |

3263 -> 1742. Removed: `possible`/`unknown-key` 1469 + `possible`/`partial-key` 52; `confirmed` 0. Every removed Link's hop paths are
`claude/cobb/scripts/check_content_loss.py` (3978 hops) or its selftest (1404); every one has a `fetch` hop in a `.py` file; `check_partition(..., fetch)` takes
`fetch` as a callback parameter (`check_content_loss.py:336,341`), so they were the D8 false positive. No TS/JS consumer lost: consumer facts are byte-identical
before and after (171 = 171), the 29 JS/TS consumer facts (`falkor-chat/web/app.js`, `salesperson/src/api/endpoints.ts`) are unchanged, and links with any JS/TS hop are 76 before and 76 after
(0 lost, 0 removed with a JS/TS hop).

**Q4, 57d chain (verified live).** `CCG_FALKORDB_PORT=6390 pytest -m live tests/test_fixture_binding_live.py`: 5 passed (144 s). The pinned chain is
ord 0 = `launchFact`/`port` of service `legacy`|`tools`, ord 1 = `launchFact`/`mount` of the same service, both in `deploy/compose.yaml`
(`tests/test_fixture_binding_live.py`, the new test). That is the contract: `ord` 0 is the entry nearest the consumer
(`ccg/linker/bind.py:135`; `docs/plans/connected-code-graph-p1-graph.md` G7). A URL is code, not a `ConfigEntry`; there is no "URL entry". The plan's
"`BOUND_BY` ord 0 the URL entry" (P1-57 57d files, last bullet) is wrong wording: fix it to "ord 0 the Compose `port` entry the URL host:port matches, ord 1 the `build.context` mount" (F3).

### Findings

**F1 · major · no test ties the fetch regex to the extractor's extension set (verified by probe).** The regex is a hand copy of the plan's list.
`tests/test_pack_fetch_scope.py` pins 10 accepted and 13 rejected written paths; none reads `ccg.units._SOURCE_EXT` or `normalize/javascript.py:JS_EXTS`. If an
extractor gains an extension (or `.vue` is added there and the regex is edited alone), fetch consumers in those files vanish silently, with no red test.
Probe (ran): `all(re.fullmatch(rx, "a"+e) for e in JS_EXTS)` and `set(_SOURCE_EXT javascript keys) == set(JS_EXTS)` both hold today, so the assertion passes now and
can fail on drift. Fix: add that census to `tests/test_pack_fetch_scope.py` (expected values from the extractor tables, the regex from the shipped pack), covering both tables, not just the 10 written shapes.

**F2 · minor · `.vue` is a dead alternative with a pinned test row.** See Q1(a). Either drop it from the regex and `SEEDS`, or keep it and say in the pack comment that no extractor emits it
yet; as written the pin `FETCH_SEED_SCOPE` and the plan list it as if reachable.

**F3 · minor · plan wording.** "`BOUND_BY` ord 0 the URL entry" (P1-57, 57d files) should read as in Q4. `docs/plans/connected-code-graph-p1-linking.md` is the architect's; route there.

**F4 · minor · harness failures present at base, unrelated to 57.** In the cpg suite subset I ran (`test_indexed`, `test_packrt_run`, `test_pack_fetch_scope`,
`test_rule_pack_schema`, `test_rule_pack_leaf_mutation`: 1002 passed, 2 failed, 1 skipped), `test_rule_pack_schema.py::test_every_data_file_under_ccg_is_covered_by_a_package_data_glob` fails on
`front/grammar/interpret.json`, `front/prompts/*.txt` (package-data glob), and `test_rule_pack_leaf_mutation.py::test_the_mutation_harness_can_tell_a_killed_leaf_from_a_survivor`
fails on its unmutated control; the second also fails in a clean worktree of `b7870153^` (the first is visibly about `front/`, not 57). Neither is caused by 57, but a green claim
for the cpg suite is not reproducible by me; route to whoever owns `front/` and the leaf harness. Note the plan asked for a `GATE` runtime test to kill the new `scope` leaf; the scope is not a
spec leaf (the spec block has no scope), and the behavioral test reads the shipped pack directly, so nothing is owed there.

**N1 · nit.** The 4096 test's "consumer" half is never exercised (consumer key `None`); a comment or a consumer-side assertion would make the test say what it covers. Not 57-caused.

### What's solid

- Scope regex: case, dot-escape, anchoring and each extension are pinned by planted inputs; mutants die.
- Schema pin (`FETCH_SEED_LINE_FILE` vs `FETCH_SEED_LINE_SPEC`, line swap plus `expected` rebuild) is an independent literal, red under every regex mutant I tried.
- The Link removal is exactly the D8 population; confirmed and JS/TS Links are untouched; consumers identical.
- 57d: carrier edits are minimal; the live pin asserts literal provider locators, status/reason, `to_unit is None` and both chain entries, and
  offline `tests/test_fixtures.py` + `tests/test_fixture_binding_offline.py` + `tests/test_pack_fetch_scope.py` are green (134 passed).

### Open questions

None blocking. A live mutant of the 57d pin (`8010:8010` -> `8011:8011` in the Compose carrier, run in a scratch worktree) did not complete: it timed out while another agent's live run (`tests/test_fixture_gate_p1.py`) shared the instance, so the pin's red-ability is inferred from its literal expected values, not shown. 57d plan Done-when also asks the live fixture-build link multiset to differ only by the two bound Links; the live pin checks the two Links but not the multiset; I did not
re-derive the multiset (state this as unverified, not green).

### Proposed REVIEW.md lines

None.
