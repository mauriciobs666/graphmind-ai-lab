# Connected code graph: plan gate of P1-51, P1-52, P1-53 (and the revised P1-49 text)

> **Status:** active · **Owner:** `analyst` · **Tracks:** C-1001 (P1) · **Reviews:** `docs/plans/connected-code-graph-p1-linking.md` (P1-49 revised text, P1-51, P1-52, P1-53), `docs/plans/connected-code-graph-p1.md` §7 rows P1-49, P1-51..53

## Pass 1

### Scope and verdict

Static plan gate, pre-implementation, with scratch-copy execution on the stored bundles under `cpg/.ccg-work/` (read only; the `cpg/ccg` and `cpg/packs` trees were copied to the session scratchpad and edited there; no FalkorDB service was touched; nothing under the repo was edited). Evidence read: the post-P1-50 report (D1-D3, section C), the D3 RCA, `docs/requirements/connected-code-graph.md` (AC-5, AC-6), the fixtures test plan, `cpg/ccg/packrt/{lift,run,extract,httpkey,match}.py`, `cpg/ccg/linker/{bind,link}.py`, `packs/python-http-client/pack.yaml`, the rule-pack schema, and the existing pack tests. Root `REVIEW.md` read and applied; no component `REVIEW.md` exists for `cpg/`.

CPG: considered, not relevant: the subject is the `ccg` pack runtime and fixture build over stored bundles, not a Joern `cpg_<component>` graph.

**Verdict: needs changes.** Two blockers (each found only by executing the plan's design), five majors. P1-51a (typed seeds) is sound and can be dispatched as written once its Done-when (4) is corrected (M1). P1-51b, P1-52 and P1-53 are not dispatchable as written.

Figures below are observed unless marked read. "Scratch variants" are defined in the Appendix.

### Findings

#### B1 (blocker, verified by execution): 51b's `urlLike: true` on `py-request` makes AC-5 8/8 unreachable and contradicts 51b's own text

Evidence: `packrt/lift.py:_url_like_alternatives` returns `None` when no alternative is URL-like, and `run_lifting` then `continue`s, so a leaf whose only alternative is the bare parameter `\x01` is dropped before any wrapper scan can use it (`httpkey.is_url_like` docstring: "A wholly unresolved piece is not URL-like"). 51b says the opposite in three places: behaviour (h) "bare `\x01` kept", the Demotion bullet ("a `Request` wrapper with a bare parameter and no informative template ... stays a keyless `possible` fact: AC-6 requires it"), and P1-49 lines 179-180 ("a `Request` wrapper with a bare parameter gives a keyless `possible` fact"). The same flag is what P1-49 documents for the verbs ("a leaf with no URL-like alternative is dropped ... no keyless `possible` fact either"), so "kept" cannot be implemented without changing `urlLike` for all seven liftings.

Execution, stored bundles at `01cb0dba` (AC-5 script items by the report's bundle method, three units):

| Variant | AC-5 count | Notes |
|---|---|---|
| baseline | 6 of 8 | reproduces the report (items 4 and 6 miss) |
| A: `wrapperSite` as described + `urlLike: true` on `py-request` (the plan as written) | **6 of 8** | items 4 and 6 still miss: `main:377 _poll_until_terminal(document_id)` is a bare-`\x01` leaf, dropped by `urlLike` before it can promote `seed_nlq_eval_corpus.py:289` |
| B: `wrapperSite` only, `urlLike` left off on `py-request` | **8 of 8** | items 4 and 6 hit via `GET /documents/{?}` (line 289) and `GET /shop/api/messages` (line 366), both `possible`/`partial-key` |

Ten Python units, consumer facts before/after: B removes exactly one (`language_salience_probe.py:397 GET /0`, `possible`) and adds exactly the four sites the plan predicts (`seed_nlq_eval_corpus.py:289`, `language_salience_probe.py:366` and `:424`, `load_demo.py:284`), providers (91) and mounts (3) equal, every `confirmed` consumer equal, 7 keyless lifted leaf items removed (5 distinct sites). A instead removes `GET /0` and `decision-models/decisionmodels/lmstudio.py:366`, adds nothing, and drops **9** distinct keyless leaf entries, including real unknown-URL callers (`claude/cobb/scripts/pwq_kb.py:601`, `falkor-chat/server/falkorchat/transport.py:139`, `model-bench/modelbench/lmstudio.py:664` and `:892`), against the plan's "observed effect: one fact leaves".

Also: the existing tests pin the opposite decision and 51b does not list them: `tests/test_pack_python_http_client.py:172` ("Request relative url is kept (no urlLike on Request)") and the `py-request urlLike` mutant row at `:244`.

Fix (run: variant B): delete the `urlLike` flip from 51b (the Files list, the bullet, behaviour (h), the step-index row text "and `urlLike` on `py-request`"), keep `GET /0` removal via the wrapper rule (it is dropped because its chain holds an informative slot; no `urlLike` needed), and say in the rule that the wrapper scan reads the leaf *before* any `urlLike` filter. If the owner wants `Request(0)` dropped where no informative template exists, that is a separate decision with its own figure over the 9 sites above; it must not ride on `urlLike`.

#### B2 (blocker for P1-52, verified by prototype): the declared `urlBindings` source (`scope: file`) leaves LNK-12 unbound, so NC-5 and R4/R6 fail

Evidence: `ccg/linker/bind.py:_callsite_urls` compares `site_file == my_file` for `scope: file`; LNK-12's consumer is `Admin.tsx::onReindex` (the lifted wrapper caller) while `new URL("http://localhost:8200/mcp")` lives in `mcp/client.ts`. I wrote the pack as P1-52 describes it (seed `callTool`, `resolution: [unresolved]`, `a0.name`, `scope.imports` glob `@modelcontextprotocol/sdk/**`, `ownSite`, rule `{lifted: url}`, `urlBindings` callsite `URL` `a0`) in a scratch copy and ran packs + `Binder` + `link` over the stored fixture bundles with R2/R5/R3 applied:

- `scope: file` (the plan): `Admin.tsx::onReindex` links `confirmed`/`key-match` to **both** `reindex` providers, unbound. The plan's "NC-5 at P1 asserts LNK-12 (bound ... one provider, `svc`)" is false.
- `scope: unit`: one provider, `binding-confirmed`. Port 8200 matches only unit `svc` (`Binder._port_match(8200)`).

Second consequence of the same declaration: LNK-02 and LNK-06 come out `confirmed`/`binding-confirmed`, not `key-match` (the binding applies to every consumer in the unit). P1-52's Done-when says LNK-02 `confirmed`/`key-match`; the manifests say `key-match` (`fx-web.yaml:38,42`). No unit owns updating LNK-02/06's P1 form (53a lists LNK-45, 19, 11, 13, 14, AC6-EX-4 only).

Good news from the same prototype: the "one unverified clause" holds: `a0.name` object-literal values resolve (`get_item` literal, `list_orders` through `TOOL_LIST_ORDERS`, `reindex` through the wrapper and `p0`).

Fix: specify `scope: unit`, state LNK-02/06/12 forms as `binding-confirmed` (or the owner's reasoned alternative), assign the manifest edits to 53a, and add to Done-when: the three consumers link to exactly the `svc` provider (LNK-12's `reindex` has one provider); mutant: `scope: file` must turn that red.

#### M1 (major, verified): 51a Done-when (4) cannot hold for LNK-08 and TW-2; no unit owns the correction

Evidence: with typed seeds applied (scratch variant C) over the fixture bundles, 10 Python consumers appear (baseline 0), so D1's premise is confirmed. But `fx-job/job/run_report.py:24` `f"{BASE}/health"` is stored as `template[unknown runtime-built, const "/health"]` and line 32 `f"{BASE}/items/{ITEM_ID}"` as `template[unknown, const "/items/", unknown]`; `BASE = "/api"` and `ITEM_ID = 42` are recorded `constant`s, but an f-string interpolation is never a `ref`. Observed: TW-2 `GET /health` `possible`/`partial-key` (the leading `\x01` run is dropped as an origin by rule 2, so the key is also wrong), LNK-08 `GET /items/{?}` `possible`. P1-49 line 238 and 51a (4) say both "stay `confirmed`" (`GET /api/health`, `GET /api/items/42`). P1-50 Part B restores constant text only; it does not resolve names inside `FormattedValue`.

Consequence: the typed seeds are necessary and not sufficient; P1-41 rows LNK-08 and TW-2 stay red and neither 53a nor any step owns them. Owner decision (architect, not stakeholder): either (a) extend the normalizer so a `FormattedValue` that is a bare in-corpus name emits `ref` (a 0.8 delta, hashed, so before `ac5-gate-1`), or (b) re-form both rows at P1 as `possible`/`partial-key` with the observed keys and accept that they do not link. Whichever is chosen, 51a (4) must name the observed form per row.

#### M2 (major): the R6 pin test cannot assert what it lists from the inputs it names, and one mutant passes

Evidence (read plus executed): (1) The pin takes "the config and shell bundles of `fx-svc`/`fx-job`". LNK-45's `(status, reason)`, LNK-16's `port-shared`, LNK-19's provider count (2) and LNK-12's (1) are `Link` outcomes that need provider and consumer facts from the Python and TS bundles (Joern, not offline). Offline the test can assert only `launchUnit` facts, `_launch_chains`/`_port_match` results and a `Binding`; the Link-level items need committed synthetic or golden code bundles, which the plan does not provide. (2) "Asserted from the manifest rows": the manifest launch rows carry unit identifiers and prose, no `launchUnit` or `BOUND_BY` ids; the ids are HMAC-derived, so expected ids would have to come from the Binder. Resolve `BOUND_BY` entries through the bundles' records at the manifest `aux` locators (path + keyPath), and compute the expected `launchUnit` set by an independent parse of each Dockerfile `CMD` plus the path-suffix rule. (3) R5 has no link effect (executed: with R2+R3, adding R5 changes zero links; it only adds a second `launchUnit` for `svc` and a duplicate `svc` chain on port 8000), so "change a CMD module path" is red only if the pin asserts the `launchUnit` multiset per Dockerfile; say which Dockerfile each mutant edits. (4) LNK-12's expectation is conditional on P1-52 and on Q-TSMCP; a pin with a branch is not a pin: gate 53a on the Q-TSMCP answer.

Also a seam: R2 says LNK-21/22 "are not stored Links at P1" while "Rows" keeps them at P1 "form unchanged" (`confirmed`/`binding-confirmed`), and `docs/test-plans/connected-code-graph-p1.md` P1T-09 still asserts them as Links ("the other five `binding-confirmed`"). 53a's text list omits P1T-09 and the LNK-21/45 narrative (`python -m fx_svc.mcp.server`, fixtures plan lines 166 and 191) that R2 makes stale.

#### M3 (major, stakeholder decision): R4 turns the NC-5 negative control into a known false `confirmed`

Evidence: executed (all variants): `scripts/refresh.sh` links `confirmed`/`key-match` to **both** `reindex` providers, no `BOUND_BY`; this matches `link.py`'s `Bindings` docstring (an unbound confirmed literal keeps `key-match`), so R4 is faithful to P1-17. But NC-5 (fixtures plan line 466) exists to catch exactly a false `confirmed` to the second same-named tool, and AC-5 requires at least 95% of `confirmed` items to be real. The "known cost" mitigation, "P1-36 reports the count of unbound ... links whose key has providers in more than one unit", has no row in `-p1-gates.md` P1-36 or `-p1-ml.md`.

Options for the stakeholder: (A) as ruled (LNK-19 stays P1 `confirmed`/`key-match` to both; NC-5 narrowed to LNK-12/LNK-20) plus an owned P1-36 line; (B) a P1-17 linker rule that demotes an unbound confirmed link whose key has providers in more than one unit to `possible` with a registry reason (new reason, `reasons.yaml` edit, P1-17 reopened); (C) move LNK-19's one-provider half to P2 and keep it out of P1 floors. Whichever is chosen, an owner and a done-condition are needed.

#### M4 (major): cross-deliverable citations without an owner

Checked by reading `docs/plans/connected-code-graph-p1.md` and grepping `docs/`:
1. **`ac5-gate-1` freeze list** (p1.md lines 405-412: "Needs every step that edits a tracked file outside" `cpg/eval/**`, `cpg/packs/**`, `cpg/tests/**`, `**/*.md`) names 49 and 5d's `extract.sc` but not 50, 51, 52 or 53. 51b edits `ccg/packrt/lift.py` and the schema, 52 edits `ccg/refresh.py` (`DEFAULT_PACKS`), 53b edits fixtures and the manifests under `cpg/fixtures/**`: all hashed. The P1-41, P1-42 rows list "...49" only, and P1-50 has no row in §7 at all (P1-51's "Needs 49, 50" cites a missing row).
2. **Rule-pack spec**: p1.md line 512 lists the spec updates at close (§12.1 rows for 46 and 49; §1.4 rules); `wrapperSite`, the typed seeds and the TS pack (§1.6 vocabulary, §12.1 rows) are not there.
3. **Q-CENSUS fallout**: "89" is stated in at least p1.md lines 48, 391, 431; `-p1-gates.md` lines 125, 131, 144; the P1 test plan lines 24, 47-49, 147-150, 175, 193-195, 286; the fixtures plan §6. 53a "owns the manifest/test-plan text" but its done-condition names no sweep. The sweep should be an unfiltered grep with every hit mapped to an owner, run after the Q-CENSUS and Q-TSMCP answers.

#### M5 (major): the 51b rule and the "informative" test are under-specified in ways that change outcomes

Evidence (read `lift.py` plus the chain dump on the stored bundles):
- Whether the scan includes the leaf is not stated (`_opaque_slot` skips it with `[1:]`; including it would make every informative leaf its own X).
- "the nearest to the leaf wins (the slot `opaqueSlot` already names)" is false when a non-informative non-transparent slot lies nearer: `_opaque_slot` returns the first non-transparent slot, not the first informative one.
- The chain is a tree whose parent is the first slot the BFS reached (`lift_role.add` keeps only the first predecessor). `routing_smoke.py:58 run_participant(...)` maps to the `:424` template only because that slot won the parent race for `p0`; the added `:424` site is correct but order-dependent. State that X is determined by the BFS order, or scan all slots reachable from the parameter.
- X's verbs come from X's own chain, so a wrapper `def f(method, path): http_json(method, f"/items/{path}")` yields `* /items/{?}`; say so as a pinned limit.
- A template with only a query unknown (`f"/items?q={q}"`) is informative and its X is `confirmed`/`lifted-role` (rule 3: a query-only unknown stays confirmed): a new `confirmed` fact replacing a keyless `possible`. Done-when (2) says every other changed consumer fact is `possible`; state the intended status.
- Behaviour list (a)-(j) is a list of shapes. Require instead a reference model written in the test, independent of `lift.py`, over the axes the rule varies on: slot kind (leaf, intermediate, seed), alternative shape (no origin, `\x01` origin, known origin, `\x00` prefix, partial segment, query-only, scheme), alternatives count (one, mixed with a bare `\x01`), number of informative slots on the chain (0, 1, 2 with a non-informative one nearer), number of leaves per X (1, 2), companion verb source (literal at X, parameter at X), and `wrapperSite` on/off. Test (i) (`-m live`, `LNK-11` stays) is green on arrival: the fixtures contain no informative wrapper (executed: variant C with and without `wrapperSite` gives the same ten fixture facts), so it constrains nothing about the new rule; it is a regression check, not a positive control. The positive control is the synthetic set above.

#### Minor

- **m1 (grounding)**: 51b cites `model-bench/.../lmstudio.py:366`; the stored fact is `decision-models/decisionmodels/lmstudio.py:366` and it is a keyed `GET /{?}` consumer, not keyless. 51a cites `TestClient.get` 188: the 188 verb-named `fastapi/testclient.py` targets are `get` 79, `post` 82 and other members; the 358 (`dict.get`), 176 (`Mapping.get`), 816 (`unresolved`), 2,178 (verb-named callsites) and 3 (`httpx.py`) figures reproduce exactly. P1-49 line 107 and 51b both write the gold line 367 as the wrapper site, while the fact's line is 366 (harmless: the hit definition is by symbol).
- **m2**: 51a Done-when (1) calls the `httpx.py:<module>.<verb>` shape "observed"; it is not observed anywhere (0 in the proving repository, no `httpx` verb in the fixtures). State it as assumed and pinned by the synthetic row only.
- **m3 (fixture realism)**: R2's `CMD ["uvicorn", "mcp.server:app", ...]`: the real `cypher-mcp/Dockerfile` ends `python server.py`, and the fixture's `mcp/server.py` has no `app`. Harmless to the Binder (it matches the path suffix only); say the CMD is inert, so no later unit "fixes" it.
- **m4**: Q-TSMCP is referenced (P1-52 heading and the option B paragraph) but its question text lives only in the ledger; Q-CENSUS is spelled out. State both the same way.

### Growth-stops check (root `REVIEW.md`; root `AGENTS.md` growth-stops bullet)

Document: `docs/plans/connected-code-graph-p1-linking.md`, revised 2026-11-20 (diff `9b9b31ba..857f2067`: +221/-9 lines; `-p1.md` +3).
- **New scope (allowed growth):** P1-51, P1-52, P1-53 (about 205 lines).
- **Rework done in place (good):** P1-49's "No `requests`/`httpx` verb call is typed ..." sentence and the "if the fixpoint does not carry a template ... returned to the architect" sentence were replaced, not appended to; the revision note is one dated line.
- **Rework growth to report:** (1) the typed-seed form is stated twice (P1-49 "Pack" bullet and 51a "Decision" plus Done-when (1)): one home needed, the other cites it. (2) 51b "Premise, checked" and P1-49 Done-when (2) both state the wrapper rule and the same four added/seven removed sites. (3) P1-53's intro and each "premise" clause restate the RCA and the QA report (D3 is not a Binder defect; the mismatch list), whose homes are `docs/reviews/connected-code-graph-p1-fixtures-binding-rca.md` and the report section E; the R1-R6 "premise, decision" shape reads as a finding-to-resolution list. Cite the RCA by ID and keep the decisions as design. (4) Counts from the proving repository are restated in P1-49 (lines 121-124) and 51a's premise.
- No finding-to-resolution table, no stacked `Update:` clause, no history narrative inside P1-49.

### Hidden test-harness traps checked

- 51a (2) mutant `*get` glob or `resolution: [unresolved, external]` widening: killed only if the synthetic rows include the listed proving-repository strings with the import gate satisfied (as written); keep that.
- 51a (5) and 51b (2) are before/after figure comparisons with no committed baseline; they are acceptance reports, not tests. Say so, and keep the baseline numbers (providers 91, mounts 3, consumers 128, lifted 73: reproduced exactly) in the report.
- 51b (i) green on arrival (M5). 53a's pin is red by construction; "red for the stated reasons" needs the assertion names listed (M2).
- A suggested fix in this review (drop the `urlLike` flip; `scope: unit`) is a claim: both were run (variants B, and the P1-52 prototype); the assertions that would catch each being wrong are the AC-5 8/8 count with the added/removed sets (B1) and "LNK-12 has exactly one provider" (B2).

### Falsifiable stopping and done conditions proposed (the plan lacks them)

1. **51a:** stop and return to the architect if the typed seeds match any callsite on the ten Python units' stored bundles (expected 0; `exact` over any widened form).
2. **51b:** (a) the AC-5 count is 8 of 8 on both instruments (store method and bundle method), equal item by item; if either is below 8 stop and report the item and cause, do not tune. (b) The ten-unit diff equals: removed consumers `{language_salience_probe.py:397 GET /0}`, added `{seed_nlq_eval_corpus.py:289, language_salience_probe.py:366, :424, load_demo.py:284}`, keyless lifted leaf items removed 7 (5 sites), providers/mounts/`confirmed` equal; any other row stops the unit. (c) The reference-model test of M5 is committed red before the engine edit.
3. **P1-52:** LNK-02, 06, 12 each link to exactly the `svc` provider, and the `scope: file` mutant turns the LNK-12 assertion red.
4. **53a:** the pin is authored after the Q-TSMCP answer and names its assertions; run against the unmodified fixtures it fails on exactly: `launchUnit` for `svc` absent, no `BOUND_BY` on LNK-45, no `port-shared` link, nothing else.
5. **53b:** after the three fixture edits the full link multiset differs from the pre-edit multiset by exactly two rows (LNK-16 `port-shared`, LNK-45 gains `BOUND_BY` ord 0 and 1); observed with the same experiment, so this is expected to pass. Any other changed row stops the unit. Each of the four mutants is recorded red with its failure message, and the R5 mutant asserts the `launchUnit` multiset.

### What is solid (verified)

- **Premise of 51a:** typed seeds with exact `externalTarget` match 0 of the proving repository's 2,178 verb-named callsites. Definition 1 (exact `requests.py:<module>.<verb>` / `httpx.py:<module>.<verb>` equal to the callee): 0. Definition 2 (any verb-named callsite whose target starts `requests.py`/`httpx.py`): 0. Wider (any callsite, any name): 3, all `httpx.py:<module>.Client.__init__`. `__builtin.dict.get` 358, `typing.Mapping.get` 176 and the `TestClient` targets would not be seeded by an exact pair. On the fixture bundles the typed pair is what turns 0 consumers into 10.
- **The `_unknown_lifted` re-creation risk is real and the plan's countermeasure (drop the leaf) is necessary:** promoting X without dropping the leaf left the keyless `unknown-key` sites 397/438 in the `join` link and raised the link count (488 baseline to 492, versus 252 with the drop).
- **R1, R2, R3 are faithful and work offline** (control reproduces QA's 37 links): with `scripts/Dockerfile` the `fx-svc` unit gets a `launchUnit` and port 8200, and the LNK-45 mention link carries `BOUND_BY` ord 0 (`.mcp.json` entry) and ord 1 (the `scripts/Dockerfile` command entry); with `compose.dev.yaml` port 8000 maps to units `svc` and `job` and `GET /job/status` from `JobStatus.tsx` becomes `possible`/`port-shared`. LNK-13/14/15/22 and every other link row are unchanged (link-multiset diff over all three edits: exactly those two rows). R4's unbound outcome is the `Bindings` docstring's rule. Step ids P1-51..53 are free (grep and `git log`); the order 51a, 51b, 52, 53a, 53b has no shared-file clash (51a/51b both edit `pack.yaml` and its test file and are serial; 52 touches `SERVED`/`DEFAULT_PACKS` after 51b).

### Proposed REVIEW.md lines

Target: root `REVIEW.md`, heading "Always check" (source: B1, B2; both found only by prototyping the plan's pack declaration in a scratch copy; reading the flag descriptions found neither). Addition: "A plan step that adds or flips a pack-format word or a pack declaration states its acceptance figures as the output of a prototype run on stored data (scratch copy of the engine and pack). The reviewer runs the combination of every flag the step touches; reasoning about flag interaction from the text is a bound, not proof." No prune applies (no existing line is now enforced elsewhere).

### Open questions (stakeholder or owner)

1. **Q-TSMCP** (stakeholder, pending): A add P1-52 (needs B2 fixed) or B re-gate LNK-02/06/12 to P2 (census 89 to 86). This review's recommendation does not change: A is small and inert on the proving repository, but it must be re-gated on the corrected design.
2. **Q-CENSUS** (stakeholder, pending): unchanged by this review; A (87) remains consistent with R2-R4 as verified.
3. **M3** (stakeholder): options A/B/C above.
4. **M1** (architect): normalizer resolves bare names in f-strings (a) or rows re-formed (b).

## Appendix: runs

All under the session scratchpad `/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/dd493cce-be51-46f4-9ad3-5b22c2dd9c32/scratchpad/` (not committed): `rv` (wrapperSite + `urlLike` on `py-request`, variant A), `rvB` (`wrapperSite` only, variant B), `rvC` (B + typed verb seeds, 51a), `r53/exp.py` (builds the host units of a scratch copy of the fixture repositories at a scratch commit, then `run_packs` + `Binder` + `link`; control = unmodified fixtures), `r53/ts-mcp-client/pack.yaml` (P1-52 prototype), `ac5.py`, `ten.py`, `chains.py`, `fxcons.py`. Python is `cpg/.venv/bin/python`. The three stored units for the AC-5 count are the report's (`falkor-chat/scripts`, `salesperson`, `falkor-chat/server`); the ten-unit runs take every Python bundle with callsites under `cpg/.ccg-work/bundles/graphmind-ai-lab/`. Fixture bundles: the 31 current-head bundles under the scratchpad `work/bundles/fixture-system1/` (the QA throwaway build at `NORMALIZER_VERSION` 0.7); Python and TS bundles were carried, host bundles rebuilt.

Not run: the full `cpg` suite, `-m live` tests, the 51b/53 implementations (none exist); the AC-5 store method (needs the live store; the bundle method reproduced the report's 6/8). My scratch `wrapperSite` is a faithful reading of the plan's text, not the future implementation, so B1's figures bound what a literal reading yields; the 8/8 and the added/removed sets under B are what the plan's rule yields once the `urlLike` flip is removed.
