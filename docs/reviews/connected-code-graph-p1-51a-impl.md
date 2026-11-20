# Connected code graph P1-51a (typed `requests`/`httpx` verb seeds) — implementation gate

> **Status:** active · **Owner:** `analyst` · **Tracks:** P1-51a (ccg P1)

## Pass 1

**Scope.** Commit `08d8eeb3` (6 files: `pack.yaml`, `negatives.yaml`, `test_pack_python_http_client.py`, `test_rule_pack_leaf_mutation.py`, `test_fixtures.py`, `cpg/README.md`) against its parent, and against P1-51 part A and P1-49 "Pack" in `docs/plans/connected-code-graph-p1-linking.md`, and the plan gate `docs/reviews/connected-code-graph-p1-51-53.md`. Read root `REVIEW.md` (evidence bar and "a check that cannot fail" apply). Everything ran on scratch extractions of `git archive 08d8eeb3` and `08d8eeb3~1` under the session scratchpad; the live tree (concurrently edited by P1-51b) was not read or run. No FalkorDB. Docker/Joern sandbox used (toolchain image `ccg-toolchain:73b9fb76bd2a`) only for the live guard, writing to tmp.

CPG: considered, not relevant — the subject is the `ccg` pack runtime, its stored bundles and a pack file, not a Joern `cpg_<component>` graph.

**Verdict: approve with suggestions.** No blocker. One major (test-tooling collision, M1) and four minors. Every figure the delegate reported reproduces except one miscount ("15 negatives" is 13).

### Ran vs read

- **Ran:** my own instrument `a51_measure.py` over the 10 stored Python bundles (proving repo, 2,466 files, `builtCommit 01cb0dba`) on the baseline and new scratch trees; the pack file's tests (25), `test_packrt_generic`, `test_p1_49a_*`, the leaf-mutation file (108 passed, 1 failed, 1 skipped overall); a 19-mutant script of my own (`a51_mut.py`); the live guard on the new tree, on a copy with the baseline `pack.yaml`, and together with TP-006 under `CCG_FIXTURES_REPORT`.
- **Read:** the diff, the plan sections, `normalize`/`extract.sc` claims through the plan and the gate (not re-derived here), fixture sources `fx-svc/clients/jobs.py`.
- **Not run:** the slow full leaf-mutation (`CCG_LEAF_MUTATION=1`), the full 15,956-test suite (the delegate's figure is unverified by me; the 4 relevant files are), the httpx typed shape live (nothing produces it), a `requests.Session` typed-shape probe.

### Findings

#### M1 (major, verified): the new live test overwrites TP-006's `CCG_FIXTURES_REPORT`
`tests/test_fixtures.py:2210-2211` (TP-006) and `:2296-2297` (P1-51a) both write the same env path with `write_text`. `-m live` runs them in file order, so the later test destroys the earlier one's output. Run: `CCG_FIXTURES_REPORT=<f> pytest -m live tests/test_fixtures.py` (3 passed) leaves `<f>` holding only the P1-51a report (keys are the two unit ids); TP-006's per-unit verdicts are gone. The documented use of that variable is `docs/reviews/connected-code-graph-fixtures-materializer.md:14` (`-m live` with `CCG_FIXTURES_REPORT`), so an existing acceptance artifact silently changes meaning.
Fix: a distinct variable for the new test (e.g. `CCG_FIXTURES_P1_51A_REPORT`), or write `<path>.p1-51a.json`; pin it by running both tests with one path and asserting both outputs exist. Owner: `tdd-engineer` (could ride in 51b's commit, since both touch the test tree).

#### m1 (minor, verified): LNK-15 observation is right; the plan contradicts itself, so the open item needs an architect ruling, not a code change
Observed (live guard report, new tree): `clients/jobs.py:17` `POST /job/trigger` `possible`/`partial-key`. Trace: `fx-svc/clients/jobs.py:6` `JOB_URL = os.environ.get(...)` (a call-valued module constant, which the normalizer drops, plan line 722 (f)) and line 17 `f"{JOB_URL}/job/trigger"`, whose `{JOB_URL}` is an `unknown` element anyway (`extract.sc` drops the `formattedValue` child, gate M1). Template `[unknown, const "/job/trigger"]`; rule 2 drops the leading unknown as an origin, so key `POST /job/trigger` with `partial-key` and status `possible`. This is exactly P1-49's rule for a base from an environment read, which P1-49's text already asserts `possible`/`partial-key` for LNK-13 (the same shape). The "confirmed/key-match" expectation is plan line 722 (f) (older decision text) and the fixture manifest `fx-job.yaml:93`. So the plan holds two forms for LNK-13/LNK-15 and the commit observed the rule's. Not a 51a defect; its Done-when (4) says "the forms are the key rule's" and records the observed form. Owner: `architect` rules which sentence stands (recommend line 722 (f) is amended: P1-49's rule wins), then 53a (qa-engineer) re-derives the rows (plan line 588 already names LNK-15). Until then `fx-job.yaml`/`cross.yaml`/`fx-svc.yaml` LNK-15 stay `confirmed`/`key-match` (red at P1-41, owned by 53a).

#### m2 (minor, verified): the "15 negatives" count and the README wording overstate
Loaded YAML: negatives 16 → 29, so 13 added (the diff is 15 lines because two are comments). 11 are real proving-repo `externalTarget` strings and every one occurs in the stored bundles with the stated callee (counts: `typing.Mapping.get` 176, `typing.Any.get` 74, `TestClient.get` 79, `TestClient.post` 82, `TestClient.delete` 4, `wf_client.get` 26, `Path.get` 7, `json.loads.<returnValue>.get` 26, `dict.get.<returnValue>.get` 12, `TestClient.__init__.<returnValue>.post` 22, `FalkorDB.select_graph.delete` 6). 2 are synthetic with 0 occurrences (`other.py:<module>.get`, `requests.py:<module>.Session.get`). `cpg/README.md` says `negatives.yaml` "holds the real proving-repository target strings"; the file also holds the synthetic ones. Fix: say "mostly real; two synthetic rows are marked as such" and mark them with a `# synthetic` comment in the YAML. The ledger/commit-message number should read 13.

#### m3 (minor, verified for shape, inferred for the real string): `requests-module-other-member` pins a real HTTP call as a negative on an invented string
`negatives.yaml` `requests-module-other-member: external "requests.py:<module>.Session.get"` expects no consumer. A `requests.Session().get(url)` is an HTTP call, so it is a *known false negative* the pack accepts ("Not covered: `Session`", plan P1-49 and the pack header), not a true negative. The string is also probably not the shape Joern emits: stored bundles show method-on-constructed-object as `<Class>.__init__.<returnValue>.<verb>` (`TestClient.__init__.<returnValue>.post` 22 times, `TestClient.__init__.<returnValue>.get` once), so the real Session form is likely `requests.py:<module>.Session.__init__.<returnValue>.get` (not probed). Effect: the row is useful as the kill for the `requests.py:<module>.*`/`*.get` glob mutants (verified, see below) but mislabelled. Fix: rename it to say it is a pinned limit (`session-get-is-not-covered`) and keep the comment; no pack change. Item 7 below.

#### n1 (nit): the live guard asserts "at least one consumer per unit"
`test_p1_51a_*` (`tests/test_fixtures.py:2294`) fails when a unit yields none, as Done-when (4) specifies. A mutant leaving only the typed `get` seed would still pass it (each unit calls `get`); the synthetic per-verb rows catch that offline, so only the live-shape coverage is thin. The expected multiset (5 per unit, which the report records) is not asserted. Acceptable per the plan; say so in the report rather than tighten.

### Measurements (recomputed)

1. **Typed seeds match 0 stored-bundle callsites.** Over all 73,038 callsites of the 10 Python bundles: 2,178 verb-named, 816 `unresolved` (the figures of the plan gate reproduce). Definition 1 (target exactly `requests.py|httpx.py:<module>.<verb>` equal to the callee): **0**. Definition 2 (verb-named, target starts `requests.py`/`httpx.py`): **0**. Widest (any callsite, any name): 3, all `httpx.py:<module>.Client.__init__`. Stop condition not triggered.
2. **Before/after multisets** (baseline = `git archive 08d8eeb3~1`, `run_packs` with the repo's `DEFAULT_PACKS`, the full tuples of providers, mounts, consumers and lifted written to JSON): providers 91, mounts 3, consumers 128, lifted 73 on both; `cmp` of the two JSON files: **byte-identical**. (My first attempt with `python -I` and no explicit path imported the editable-install `ccg` of the live tree; the measurement script inserts the scratch path first, and I confirmed `ccg.__file__` for every figure reported.)
3. **Fixture Python units 0 → 10.** There are no stored fixture bundles under `cpg/.ccg-work` (it holds fixture source trees), so I ran two forms: (a) a throwaway rebuilt bundle set from earlier in this session: baseline 0, new 10 (5 `fx-job`: `preflight`, `search_reports`, `health_probe`, `fetch_item`, `requeue`; 5 `fx-svc`: `peer_health`, `legacy_fee`, `run_tool`, `call_named`, `trigger_job`); (b) the live guard itself, which builds the units in the sandbox: new pack green, same 10 consumers; baseline pack red (below). So 0 → 10 reproduces on a real build, not on a stored bundle.

### The 11 real negatives and the glob mutants (check 2)

All 11 strings occur (counts under m2). Which negative kills which mutant (my runs of `behaviour_failures` on mutated packs, listing the failing labels):

| Mutant | Killed by |
|---|---|
| `externalTarget` glob `*.get` | 8 negatives (real ones included) |
| `resolution: [unresolved, external]` | 10 negatives + typed-non-http (real strings) |
| glob `*.py:<module>.<verb>` | **only** synthetic `other-library-get-with-the-verb-name-in-the-module` |
| glob `*<module>.get` | **only** the same synthetic row |
| glob `requests.py:<module>.*` / `[requests,httpx].py:*.get` | **only** synthetic `requests-module-other-member` |
| glob `*.py:<module>.*` | `Path.get`, `json.loads...get` (real) and both synthetic |
| glob `requests.py:*get` | the positive httpx cases (offline) + the synthetic member row |
| prefix `requests.py`/`httpx.py` | positive cases (offline) |

The delegate's claim holds, and more than two: four narrow-glob mutants die only to the two synthetic strings. I checked representativeness: no real bare `<lib>.py:<module>.<verb>` target for *any* library exists in the proving repo (regex over all bundles: 0), so no real string can ground those rows; `other.py:<module>.get` is the right shape for the failure (a different library's module-level verb) and `Session.get` is the limit row (m3). Acceptable; the dependency is stated here.

### Valid-alternative mutants, leaf harness, my own mutants (checks 3, 4)

- The 11 new parametrized mutants pass in the 25-test file run (a pass means each mutant is caught). I also ran 19 mutants of my own through `behaviour_failures` and all die: 9 widened-form mutants (above) and 10 silent-argument mutants: httpx `get` target changed to `...post`; `requests` `put` target changed to `...patch`; typed `callee` `get`→`post` on `py-get`; typed import scope only `requests` / only `httpx`; typed scope swapped to `requests` only on `py-post`; import prefix `requests`→`request`; `<module>`→`<modul>`; `requests.py`→`request.py`; `httpx.py`→`httpx.pyi`. Each dies on the typed literal / own-site / cross-import rows (3 failures each), so the cross-library import rows ("`httpx` import alone gates typed `requests.<verb>`") are what pin the scope per verb. Note these rows pin the *loose* gate (either import admits either target) as contract; a future per-library tightening would turn them red, which is deliberate per their comment.
- **Leaf harness 125 → 155:** independent scalar-leaf walk of `pack.yaml` (yaml, not the harness's helper): base 125, new 155; the +30 is 5 verbs x (callee, two exact strings, arg, two import prefixes). `test_the_python_http_client_leaf_mutants_die_behaviourally_and_at_the_validator_in_these_numbers` asserts (74, 81, 0) and passes (+25 behavioural, +5 validator-only, the five `arg` leaves; the `a1` swap is a behavioural mutant in the list). The sibling-swap survivor set is still exactly `{urlBindings[0].id}` (test passes; the swap runs all new leaves, including the typed `exact` strings, against sibling values).
- Suite state: the one failure in the leaf file (`test_the_mutation_harness_can_tell_a_killed_leaf_from_a_survivor`) is red on the parent too (verified on the baseline scratch tree); not caused by this commit.

### Live guard (check 5)

- **Green on arrival is not true of it: red on baseline, mechanism verified.** With the new tree, the test passes (45 s; Docker sandbox). With `08d8eeb3~1`'s `pack.yaml` copied into the new tree, it fails at `tests/test_fixtures.py:2294`, `assert consumers`, for unit `e717...` ("no http consumer from its requests calls"). The earlier assertion (`typed`: the build stores `requests.py:<module>.<verb>` callsites) passes on both, so it is the pack seeds, not the extractor shape, that flips the result, which is what Done-when (4) needs. Its expected values (that the build is typed, that consumers exist) are not read from the pack constants.
- The report is written correctly: keys are unit ids, values the sorted `(path, line, key, status, reason)` tuples: `GET /legacy/fee` and `GET /tools/run` and `GET /api/search` `confirmed`/`lifted-role`; the rest `possible`/`partial-key` (LNK-14 `GET /api/health`, TW-2 `GET /health`, LNK-08 `GET /items/{?}`, `requeue` `POST /api/v1/orders`, `call_named` `GET /{?}`, LNK-15 `POST /job/trigger`). These match the plan's rulings for LNK-08/TW-2 (gate M1). The only problem is M1 above: the file path is shared with TP-006.
- Regex `REQUESTS_VERB_CALL` misses `from requests import get`; fine for the fixtures (they use `requests.<verb>`).

### Open items it raises (check 6)

- **LNK-15:** m1. **httpx typed shape:** assumed, correctly labelled everywhere (pack header, README, synthetic rows), harmless because the pair is exact; nothing live can confirm it (the proving repo and the fixtures hold no typed `httpx` verb; `httpx.<verb>` there is `unresolved` per the plan, so the untyped seed already covers it). The gate's m2 is addressed. A follow-up belongs with the first corpus that types an `httpx` verb; no action now.

### Check 7: `requests.Session().get(url)` / `httpx.Client().get(url)`

Import gate satisfied, so the *untyped* seed (`resolution: unresolved`, `callee: get`) matches such a call when the extractor leaves the receiver untyped (the case for `httpx.Client` in the proving repo: `load_append.py`, plan P1-49), and it is a consumer. If the build types it, the target is `<lib>.py:<module>.<Class>.<...>.get`, which neither seed matches: **not a consumer, correctly by the exact-pair design** and by the plan ("Not covered: `Session`"). The miss is accepted by the plan; it is pinned only by m3's mislabelled row and has no live probe of the real string (inferred from the `TestClient` analogs, not run).

### What's solid

- The seed form is the plan's (exact pair, same `scope`, `arg: a0`, `ownSite`, `urlLike` unchanged), nothing else in `pack.yaml` changed, and the pack comment states the "assumed" status.
- 0 callsites matched, byte-identical multisets, 0 → 10 on a real sandbox build, red on the baseline pack: all reproduced with an independent instrument.
- Negatives cover every real string the plan listed in Done-when (2); mutants for the widened forms die; harness numbers are right.

### Open questions

- None that change scope. m1's architect ruling and M1's variable name are the only decisions.
