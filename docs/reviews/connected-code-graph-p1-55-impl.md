# Connected code graph P1-55 — implementation gate

> **Status:** active · **Owner:** `analyst` · **Tracks:** P1-55 (ccg P1)

## Scope & verdict

Diff gate on commit `81ef5647` (the one after `4cd528b9`): `cpg/ccg/packrt/httpkey.py` (`names_no_literal_segment`), `cpg/ccg/packrt/run.py` (`_one`), `cpg/packs/fetch/pack.yaml` (`ownSite: true`), `cpg/README.md` row, golden `cpg/tests/data/p1_55_unknown_url.json` + `…_regen.py`, two new test files, five edited existing test files. Baseline: `docs/plans/connected-code-graph-p1-linking.md` P1-55; cause `docs/test-reports/connected-code-graph-p1-report.md` D3/D4/D5; style and lessons from `docs/reviews/connected-code-graph-p1-54-impl.md`. Root `REVIEW.md` applied; `cpg/REVIEW.md` does not exist.

**Verdict: approve with suggestions.** No blocker. The predicate and the `_one` rule match the plan's Decision on every row of my independently written table, the golden is byte-identical to the stored real bundle, every mutant of the rule I planted was killed except one equivalent-today survivor, and the changed-consumer set re-derived with a different instrument is exactly the claimed one site. The findings are unpinned corners and an inaccurate blast-radius wording.

CPG: considered, not relevant — the question is a diff-scoped predicate over bundle records, answered by running the predicate, the pack runtime and mutants over the stored bundles; `cpg_*` graphs describe other components.

**Observed runs** (from `cpg/`, `.venv/bin/python -m pytest`):
- `test_p1_55_keyless_url.py test_p1_55_real_shape.py test_p1_49a_httpkey.py test_p1_49a_sites.py test_packrt_wrapper_site.py test_packrt_lift_options.py test_rule_pack_schema.py`: 1955 passed, 1 failed (`test_every_data_file_under_ccg_is_covered_by_a_package_data_glob`, known).
- `-m "not live and not slice" -k "packrt or linker or p1_49 or p1_5 or rule_pack or httpkey or assemble or keynorm"`: 3306 passed, 2 failed (package-data-glob and leaf-harness survivor-tell, both known), 1 skipped (`CCG_LEAF_MUTATION`, not set: unrun, not green).
- `tests/slice/test_packrt_conformance.py tests/slice/test_linker_conformance.py -m slice`: 24 passed, 1 xfailed (the strict 8200 xfail). The plan asks both unchanged.
- Not run by me: the full suite (the delegate's "9 failed / 16072 passed" is a claim), the AC-5 8-of-8 count, anything needing Joern or the store.

## Findings

### M1 (minor) — a known-parameter segment beside `{?}` is unpinned, and the plan does not decide it

*Verified by mutant.* The plan says "no literal segment"; the code (`httpkey.py:115-121`) stops at `{}` too, so `GET /{}/{?}` stays a key (my table: `/{id}/{?}` → `* /{}/{?}`, predicate False). That reading is defensible (the docstring states it) but the plan's table has no `{}` row. The only `{}` pin is `test_a_known_param_segment_is_not_unknown` on `/{}` alone. Mutant `CONSUMER in segs and all(s in (CONSUMER, '{}') for s in segs)` (so `/{}/{?}` becomes keyless) **survived**: 1287 passed across the 55 files, 49a grid, wrapper-site and lift-options tests.
Suggestion: add `(f"/{{id}}/{U}", False)` to `TABLE` in `tests/test_p1_55_keyless_url.py` (or the opposite row if the architect rules the other way) and one sentence in the plan's Decision naming `{}`. Plan ownership: `architect`.

Related observation, not a finding (the plan lists `/{?}a` explicitly): step 8a masks any segment holding a marker to `{?}` whole, so `"/api" + X` (`/a\x01`) also normalizes to `* /{?}` and is keyless (verified: `/a\x01` → `* /{?}`, predicate True).

### M2 (minor) — the `edge == "CONSUMES"` guard in `_one` is unpinned; equivalent today

*Verified by mutant.* Dropping `edge == "CONSUMES" and` (`run.py:164`, rule applied to providers too) leaves the failing-test set identical to baseline (5 baseline failures, all environment artefacts of my copy, identical before and after). It is equivalent now only because a provider with an unresolved piece reaches `used_modes` `drop` and returns before this line (`run.py:148-152`); no shipped pack sets `partial` on a provider slot (grep of `unknown:` in `packs/*/pack.yaml`: the two `unknown: "*"` are verb companions). A future provider slot with a partial mode would be silently keyless.
Suggestion: one planted provider rule with a `partial` slot whose URL is `\x01\x01`, asserting the provider key stays `GET /{?}`-shaped; or state in the `_one` comment that providers never get here.

### M3 (minor) — the blast-radius wording "19 Links removed, 117 added" is not what changed

*Verified by recomputation* (stored 1306 bundles, `run_packs` → `Binder` → `link`, old code from `git show 81ef5647~1` in a scratch copy, old pack via `ownSite` popped, so a different instrument from the delegate's scratch script: separate package copy, then four identity definitions). Totals reproduce: 3165 → 3263 Links, providers (172) and mounts (5) equal, confirmed consumers equal, one consumer fact differs (`decision-models/decisionmodels/lmstudio.py:366`, site `eb54631a…`, `GET /{?}` `possible`/`partial-key` → key none `possible`/`runtime-built`), consumer count 171 → 171.

| Definition of "changed Link" | removed | added |
|---|---|---|
| full record (JSON of the whole Link tuple) | 19 | 117 |
| Link id, or (kind, key, provider, consumer) | 0 | 98 |
| (provider, consumer-symbol) pair | 0 | 98 |
| distinct consumer symbols touched | 1 | 1 |

The 19 "removed" are existing Links of that one consumer to single-segment providers (`/health`, `/channels`, `/join`, …) that kept their id and were **rewritten**: matcher `path-template` → `none`, Via key `GET /{?}` → null; 16 of them reason `partial-key` → `unknown-key`, 3 stay `partial-key` (the Link keeps its provider-side reason). So before the change the consumer linked to 19 providers; after, to all 117. That is the P1-36 trade-off the plan states, but the plan's Basis says such a consumer "links to nothing" (true on the fixture, false here). `max_fan_out` stays 117, `unknown_key_consumers` 25 → 26, `by_reason.partial-key` 191 → 176, `unknown-key` 2254 → 2367.
`ownSite` inert: after-run with and without `ownSite` is identical in consumers, providers, mounts, Links, counts.
Suggestion: say "19 rewritten, 98 added, one consumer symbol" in the report/history entry that cites this figure.

### M4 (minor) — `test_with_all_three_words_off_or_absent_the_fetch_pack_gives_the_same_facts` no longer tests the fetch pack

`tests/test_packrt_lift_options.py:183-187`: the edit replaces `run(recs, FETCH)` with `run(recs, fetch_with())`, and `fetch_with()` now pops `ownSite`. The stale expectation is correct (the shipped pack differs from all-off by exactly this line), but the test's name and docstring still say "the fetch pack". It now pins runtime defaults of an `ownSite`-less pack, which is a different, weaker claim, and nothing else asserts "shipped pack == absent-words pack except `ownSite`".
Suggestion: rename to "an absent word equals the word off", and let the pin `test_the_fetch_pack_declares_own_site` plus the on/off pair carry the shipped-pack claim (they do).

### M5 (minor) — real-record coverage is the `urllib` site only; `ownSite` on a real `fetch` record and drift protection are not committed

(a) The golden is `urllib.request.Request` (python-http-client pack). The `fetch` + `ownSite` path (D4, `dyn.ts::go`) is tested only with hand-built records (`wrapper_never_called`, `leaf_call`): the P1-17 defect class the plan's "Defect-class lesson" names. I ran the shipped packs over the fixture bundles of the P1-56b scratch build (`scratchpad/p56b/work/.ccg-work/bundles/p56`, not mine, observed once): with `ownSite` `clients/jobs.py::call_named` is `None possible runtime-built` and `app/src/api/dyn.ts::go` is `None possible runtime-built`; without it `go` is absent (34 consumers vs 33). So the claim holds, but no committed test holds it: the plan's Done-when live-fixture line rests on the delegate's report until the 56a pin lands (uncommitted and red on purpose at this writing).
(b) Same residual as P1-54 M1: the census runs over a static golden, so a change in the extractor's naming leaves it green. Golden provenance names the command but not the build commit (`p1_55_unknown_url_regen.py` docstring).
What I did verify: all 16 golden records are byte-identical to the records of the stored bundle `ea820cc0…24cdff79…` (file, symbol, callsite, 9 imports, 2 valueSets, 2 data), so the golden is real, not hand-shaped; the census (`matched == ["\x01\x01"]`) is red under predicate-matches-nothing and rule-never-fires mutants.
Suggestion: have 56a's live pin (or a fixture-bundle golden for `dyn.ts::go`) assert `call_named` and `go` keyless; record the build commit beside the golden.

### Existing-test edits (item 4): judged, none weakens a pin except M4

| File | Edit | Judgement |
|---|---|---|
| `test_p1_49a_httpkey.py` | grid and spot rows compare to `reference_outcome(...)`, a 3-line model written in the test from the plan text | Corrects a stale expectation; independent of `httpkey` (own split and `{?}` literal), so not a mirror. The grid does not contain a `{}`+`{?}` row (M1). |
| `test_p1_49a_sites.py` | `(status, reason)` → `(key, status, reason) == (None, "possible", "runtime-built")` | Strengthens: adds the key. |
| `test_packrt_wrapper_site.py` | model special-cases shape path `/{?}` → `(i, None, "possible")`; `sorted(..., key=repr)`; `g2` key `None`, sorted by `siteId` | The model keys on the test's own shape table, not the implementation; the sort edits are needed for `None` keys and do not loosen (the two-element list is still exact). Acceptable. |
| `test_packrt_lift_options.py` | `fetch_with` pops `ownSite`; the "three words" test uses it | M4. |
| `test_rule_pack_schema.py` | `SPEC_PLUS_LINES`, one extra line allowed | See item 5 below. |

### Item 5: pack pin and README row — narrow, matches

`test_the_pack_file_is_the_spec_block_plus_status_paper` removes exactly the line `    ownSite: true` (asserts `count == 1`) and adds `ownSite: True` to the expected liftings of `fetch` only; the rest of the equality is unchanged. `fetch` has one lifting, so the blanket expectation is exact today; a second lifting without the line, or a second copy of the line, goes red. Mutants of `packs/fetch/pack.yaml` (`ownSite: false`, line removed) were killed by `test_the_fetch_pack_declares_own_site`. The README `ccg/packrt/` row says "its file is the spec block plus that one line" and cites `names_no_literal_segment`, matching code and pin. The pack-file comment says the spec block owes the line at the feature's close; `grep ownSite docs/plans/connected-code-graph-rule-pack-spec.md` returns nothing, so the close obligation is real and owned by `architect` (plan line 516).

### Item 7: same defect class elsewhere — none found

Sweep of `ccg/packrt/*.py` for record-shape reads: `calleeName` (`run.py:73`, `match.py:180`, `lift.py:108`), `externalTarget` (`match.py:187`), `resolution` (`match.py:185`). Every `externalTarget` clause in `packs/*/pack.yaml` against the 1306 stored bundles' real callsite targets: `urllib.py:*Request.__init__` 12 real matches (`urllib.py:<module>.<member>(request).Request.__init__`), `urllib.py:*urlopen` 11, `httpx.py:*Client.__init__` 3, `**FastMCP.streamable_http_app` 2. The exact `requests.py:<module>.<verb>` / `httpx.py:<module>.<verb>` pair rows match 0 real records on this repository: already documented in the README as fixture-only/assumed (`python-http-client` row), so not a new defect. No dotted `callee:` exists to be dead.

## What's solid

- Predicate and rule against my own table (raw consumer strings through `normalize_key`, then the predicate): root `/` stays a key; `\x01/` stays a key; `/{?}`, `/{?}a`, `\x01/\x01`, `/{?}/{?}`, `\x01\x01`, `\x01?q=1`, `/\x00`, `http://h/\x01` are keyless; `/a/{?}`, `/{?}/a`, `?q=1` (root), `http://h/`, `/{id}` stay keys; verb `*` and `GET` and `POST` normalize alike (the predicate reads `key.partition(" ")[2]`). Known or unknown origin gives the same answer because the origin is dropped before the predicate. Only my own expectation for `/a\x01` was wrong, not the code (8a masks it whole).
- Mutants of the rule killed: widen to any `{?}` (`/a/{?}` row), drop root exemption (`/` row), flip or remove `ownSite` (pin and pair), status `confirmed`, reason `partial-key`, key kept, rule never fires, drop the verb `partition`, `{}` counts as unknown on its own. Survivors: M1, M2.
- Live fixture acceptance reproduces on a second set of bundles (M5a); golden is the real stored record (M5).

## Open questions

- M1: does `{}` beside `{?}` mean keyless? (`architect`; the code's current answer is "no, a key".)

### Proposed REVIEW.md lines

Target: `cpg/REVIEW.md` (new) under `## Always check`; source: P1-54 review M1/M2 (`docs/reviews/connected-code-graph-p1-54-impl.md`, still unapplied: no `cpg/REVIEW.md` exists) and this review M5.

- A Binder or pack predicate that reads a bundle record shape has one test whose input is a record extracted from a real build, with the extraction command **and the build commit** recorded beside the data, and a census assertion that the predicate matches at least one real record. A census over a static golden is paired with one live-tier assertion on the committed source file, so extractor drift reddens something. A behaviour that has only hand-built records (an `ownSite` or other lifting option) is named as such in the gate.
- A before/after blast-radius figure ("N Links removed, M added") states the identity it counts by; the gate recomputes it under at least a full-record and an id definition and reports the spread.
