# Connected code graph: P1 close-out unit (before P3)

> **Status:** active · **Owner:** `architect` · **Tracks:** C-1001 (P1 close-out) · **Extends:** `docs/plans/connected-code-graph-p1.md` · **Reviews:** `docs/reviews/connected-code-graph-p1-closeout.md`, `docs/reviews/connected-code-graph-p1-closeout-ml.md`

CPG: not applicable — a sequencing and scoping plan over closure evidence and plan documents; the code seams it names are read from source, the RCA and the two reviews, no graph query was needed.

## 1. Goal and scope

Carry out of P1 the work that P3 to P5 depend on (stakeholder, 2026-11-24: a "P1 close-out unit before P3"; order P0, P1, P3, P4, P5, P2): scripts linking including DEF-1; the held-out AC-5 gate, run once with a verdict over **http + mcp + scripts (151 counting items)**; then, before P3, the AC-8 real-component run, the AC-24 retry and the AC-10 observation. Status per AC at the start: `docs/test-reports/connected-code-graph-p1-closure-report.md` (§4, §6); backlog home `docs/BACKLOG.md` C-1001.

**Stakeholder decisions folded in (2026-11-24):** wiring is **withheld**: its 8 held-out items stay blind, the AC-2 and AC-4 wiring halves stay open, and `tico` owns the requirements wording change. The scripts resolver's last tier is a unique path-suffix match labelled `possible`, never `confirmed` (§3.1). CO-10 and CO-11 run after the gate and before P3 (§5). S1 is ratified: the at-least-5x noise bar holds in **both** the kind-wide and the key-scoped baselines (§4, CO-6a). CO-12 is included, before P3.

**Out of scope (P2 or later):** the git-remote fetch path and dependency download (egress Q1 = C), non-P1 languages, `library_api` dynamic import (Q-DYN B), MR rows, AC-9b's unbound-crossing rule, the flow stratum (`python.flowAvailable` false; its 99 items stay blind), AC-17's queue-name group, per-user `raw_query` limits (P4), every P3 operator. No builder dispatch reads a held-out list (contact rule, `docs/plans/connected-code-graph-p1-ml.md` §6.2).

## 2. Findings that shape the plan

- **Root cause of scripts 0, wiring 0:** `docs/reviews/connected-code-graph-p1-gate-zero-hit-rca.md` (cited, not restated). Its fix 2, reconciling EP §3.4/§3.5 and `-p1-ml.md` §6.1 with the chosen scope, is owned by CO-6c.
- **Denominators (read from dev gold `cpg/eval/dev/ac5-*.jsonl`; the one set every unit uses):** scripts 49 items in 22 anchors (45 symbol-anchored, 4 contract-anchored: 3 http `GET /health`, 1 mcp); wiring 11 items in 6 anchors (4 unit, 4 symbol, 3 contract), of which 7 are askable without the custodian-side unit probe; http 18 (12 anchors); mcp 20 (3 anchors). The RCA's 46 and 0/11 and the closing-items report's "0/7 asked" are other runs of the same files; the review's "46 symbol-anchored" does not reproduce against the gold file (45) and CO-1 publishes its own count.
- **Walk (settled by reading, `docs/reviews/connected-code-graph-p1-closeout.md` m1; CO-2's first test proves it):** for a `<script>` symbol anchor the impact table has `calls_in` (`CALLS <-` to a `CallSite`) then `owner` (`cpg/ccg/engine/bfs.py:125,140-145`), so a `CALLS` edge into a `<script>` symbol lists the calling site's owner as a one-hop `caller` item, `possible`/`flow-unavailable` in a flag-false unit. No walk row, no `recipe.algorithm.version` bump. `CALLS_NAME` cannot reach (`Name` resolves on `(lang, name)`, `cpg/ccg/assemble/rows.py:482-484`). Two costs: every added edge is a `possible` item that feeds the wrong-`possible` bound, and the V-both rows add callee items per caller hit (`bfs.py:190-196`).
- **What the extractors already give** (analyst probe over dev gold, to be re-derived by CO-1): of 17 Markdown impacted elements, 15 already carry a `possible`/`prompt-mention` call site naming the script (`cpg/extractors/markdown/extract.py:313`); of 26 shell ones, 1 does, and 25 carry only the command word (`source`, `.`, `exec`, `bash`) with the path in the argument's `valueSet` (shell extractor docstring: each argument `a0..` has a `valueSet`), where the path is mostly computed (`$HERE/...`, `"$(dirname "$0")/..."`, `$REPO_DIR/...`).
- **The pass rule is "every item"** (`docs/requirements/connected-code-graph.md` AC-5): any miss means a fix and a re-run on a new blind list. Counting items: http 26 + mcp 46 + scripts 79 = 151, above the 75 floor. A pass supports the cluster-bounded claim only: **0.938 over 58 clusters for the four-stratum draw** (`-p1-ml.md` §6.1, the item bound 0.976 assumes independence); `data-scientist` restates the bound for the three-stratum draw in CO-6c; wiring's "screened, not certified" label is moot while it is withheld. Probability of zero misses over 58 clusters is r^58: 0.56 at true cluster recall 0.99, 0.31 at 0.98, 0.17 at 0.97 (arithmetic, `docs/plans/connected-code-graph-p1-gate-marker-ml.md`, last section): a first-spend miss is the likely outcome.
- Harness state: `STRATA_GATED` lists all four strata and `MIN_COUNTING` is 75 (`cpg/eval/tools/gate.py:53-57`); a withheld stratum is withheld by the gate-marker note's §4 mechanism, with counts and reason in the record. `ccg-prereg-v1`/`ac5-gate-1` does not exist yet (`cpg/eval/preregistration.jsonl`).
- **Freeze hash:** `coreTreeHash` covers every tracked file under `cpg/` except `eval/`, `packs/`, `tests/` and `*.md` (`cpg/eval/tools/ac24_smoke.py:576-596`); `packs/` enter `packSetHash`. So `ccg/` (incl. `ccg/front/`), `extractors/` and `infra/` are core.

## 3. Design and rationale

### 3.1 Scripts: a resolver in the assembler

**Layer: the assembler** (`cpg/ccg/assemble/rows.py`, a new pass after `symbol_ids`/`sym_path` exist, beside the `calls-target-not-in-snapshot` drop at l.546). Extractors run per unit and cannot see other units' files; only the assembler (system-wide snapshot) can match across units. **Carrier: existing records, no bundle-schema change.** The path travels in the call site's argument `valueSet` (bundle-spec3 §5.8, already written by the shell extractor), and a Markdown script mention keeps its token as written; the assembler adds the `CALLS` edge (`CallSite -> Symbol`), taking the call site's own status and reason, which exist today (`shell-dynamic`, `prompt-mention`), so the reason registry does not change. **Unverified, first read of CO-1:** that a `template` `valueSet` keeps the literal tail and marks the dynamic head, and that the token screen does not withhold it as `unknown`. If it does, the fix is an extractor change inside the same record shape; a new field would be a bundle-spec amendment, which is escalated to the stakeholder, not made here.

Tiers, in order, first match wins:
1. **Exact path:** a shell site's literal argument (`./x.sh`, `scripts/x.sh`), resolved relative to the calling script's directory then the unit root, naming a script file of the snapshot: `confirmed`/`resolved-call`.
2. **Unique static-tail suffix match (stakeholder-ruled last tier):** the dynamic head (`$VAR`, `$(dirname "$0")`, `$HERE`) is dropped, `..` and `.` segments normalised, and the remaining tail, **at least two path segments**, is matched as a suffix against the snapshot's script files; **exactly one** match gives a `possible` edge; zero or several give no edge and a counted refresh-report line. Never `confirmed`. A one-segment tail never matches (that is the rejected basename match).
3. A Markdown script mention uses tiers 1 and 2 and is always `possible`/`prompt-mention` (a prose mention proves nothing).

**Rejected:** a new `Link` kind (bundle and answer-spec change for a lexical fact); basename matching (false-positive rate is what the wrong-`possible` bound measures); extractor-side resolution (cannot see across units).

### 3.2 Reach and what is built or withheld

Shapes of the scripts dev gold, counts from the analyst's probe (CO-1 re-derives and publishes the table):

| Shape (dev items) | Disposition |
|---|---|
| Markdown prompt with an inline script token (15) | built: edge only (call site exists) |
| Shell `source`/`.`/`exec`/`bash` with a literal or `$HERE`-headed path (about 25) | built: tiers 1 and 2 |
| `test-stamp-wiring.sh` parse failure (3) | built: **CO-2a** fixes the shell extractor's parse of `bash -c '...'"$HERE"'...'` quoting, red test on that file first |
| 1 Python caller running `bootstrap_schema.sh` | withheld: needs the Python extractor to read a subprocess argument |
| 3 `-m` / `$PY -m pkg` anchors | withheld: `-m` package resolution is a P2 item (`-p1-linking.md` ~l.929) (a `$PY $HERE/server.py` form is built if CO-1 finds tier 2 covers it) |
| 2 front-matter `hooks: command: $HOME/.claude/...` | withheld: new front-matter extraction and an installed, not repo, path |
| 1 contract item whose consumer is Python embedded in a shell script | withheld |

Every withheld shape is reported as such, with its dev count; its held-out analogue count is **unknown** (counting it needs contact with the held-out list). See Q8.

### 3.3 DEF-1: one URL source, two carriers

A `curl`/`wget` URL literal in a shell script becomes an http consumer through a pack beside `shell-mcp-call` (`cpg/packs/shell-http-client/`, new; `possible` with the existing URL reasons when the URL has a dynamic part). The Markdown carrier is **inline spans only**: the dev miss (`claude/teco/coordination-techniques.md:25`) is an inline span, and a fence is not a call (the shipped contract, `cpg/extractors/markdown/extract.py:12`; `curl` is already in `COMMANDS`). CO-1 first checks whether the existing command-mention call site carries the span's URL token; the carrier change is to put the first URL token of an inline `curl`/`wget` span into that call site's argument value set. No fenced carrier unless CO-1 finds a dev item that needs one.

### 3.4 Wiring: withheld

The 11 dev wiring items need a spec change (`docs/plans/connected-code-graph-answer-spec2.md` §3.3: a `config` item exists only as the `BOUND_BY` target of a `Link` reached by `TO <-`; `launchFact` entries are not items; the real build had 0 `bound[]` on 179 `Link` hops). Launches-a-script (3) and names-a-tool (3) are inexpressible; the 4 server-URL entries need a code consumer that the entry binds. **Expectation: CO-1 finds none or one or two expressible entries.** CO-4 (wiring) is therefore removed from the plan; it returns, reduced to shapes needing no answer-spec change, only if CO-1 proves such a shape by a failing-then-passing dev test, and then only as a re-add the stakeholder sees. Consequences: wiring's 8 held-out items stay blind; AC-2 and AC-4 wiring halves stay open; the gate certifies nothing about wiring.

### 3.5 Gate entry criterion

Pre-registered verbatim from the gate-marker note's last section ("Entry criterion", six items), with these plan-level readings: on the dev gold, at the freeze commit F, by a dispatch that has read no held-out material, **every dev item of a gated stratum is a hit, or is removed only as a stakeholder-confirmed `key-error`**; "triaged" under any other category means not met. A shape this plan rules "not built" (§3.2) is reported withheld and also means the criterion is not met for that item: the stakeholder takes that decision with the number in front of them (Q8), it is not absorbed silently. The dev set drove the mechanism design: that caveat is recorded in the preregistration (a dev pass is a fit statistic, not AC-5 evidence). The pre-gate probe tied to this criterion blocks when a gated stratum has **fewer dev hits than dev items** (not only zero).

### 3.6 Tests, not fixtures

New behaviour is tested on synthetic trees under `cpg/tests/`. The fixture builds already contain shell scripts and prompt mentions, so script-to-script `CALLS` and the new pack can move edge counts, `packSetHash` and goldens: CO-2 and CO-3 are done only when the fixture and conformance suites (R1, R7 of the closure report: `test_fixture_gate_p1.py`, `test_fixture_gate_p1_ac.py`, `test_fixtures.py`) are unchanged, or each moved row is owned and re-derived; a silent re-pin is a finding.

## 4. Units

| Unit | Owner | Needs | Done when | Review gate |
|---|---|---|---|---|
| **CO-6a** S1 record | `data-scientist` writes, `teco` commits | none; **first**, before any dispatch prints a noise reduction | the decision record states the 5x bar holds in **both** kind-wide and key-scoped baselines (key-scoped is a subset, so its numeric bar binds; kind-wide carries the equal-recall comparison; wording in the gate-marker note), committed before CO-1 to CO-5 or the CO-9 rehearsal print any variant reduction; those runs print aggregate noise (Q, wrong-`possible` bound) only | `analyst` |
| **CO-1** Dev probe and scope ruling | `qa-engineer` runs, `architect` rules | frozen HEAD build; dev gold only | a table classifying every dev item of scripts (49), http DEF-1 and wiring (11) by the mechanism needed (the shapes of §3.2, plus path literal, `$HERE` head, prompt mention, hook launch, server URL, tool name), with the one denominator of §2; the `valueSet` check of §3.1 and the call-site check of §3.3; the wiring probe of §3.4; the architect's ruling that fixes the built and withheld lists and each unit's expected dev count | `analyst` on the ruling |
| **CO-2** Scripts resolver and **CO-2a** parse defect | `tdd-engineer` | CO-1 | **first red test:** a synthetic three-file tree (caller, `<script>` target, a decoy) through the real assemble and `ask`, asserting the item, its hop kind `caller` and its status, red when the `CALLS` edge is deleted; then tiers 1 and 2 with tests for a `$HERE` head, a missing file (no edge), an ambiguous suffix (no edge), a one-segment tail (no edge), `confirmed` only at tier 1; CO-2a red test on `test-stamp-wiring.sh`; dev scripts: every built-shape item listed; fixture suites per §3.6 | `analyst`, mutation-checked on the tier boundaries |
| **CO-3** URL source (DEF-1) | `tdd-engineer` | CO-1; after CO-2 (shared files) | dev http 18/18, mcp stays 20/20; negatives file for the pack; Markdown carrier inline-only; the contract line in `docs/plans/connected-code-graph-p1-extraction.md` P1-12 and the extractor docstring updated; fixture suites per §3.6 | `analyst` |
| **CO-5** Harness minors | `tdd-engineer` | none; builder-safe, parallel to CO-2/3 (different files, `cpg/eval/tools/gate*.py`) | P5-1..P5-3 (`docs/reviews/connected-code-graph-p1-32b-impl.md`); the dev-gold pre-gate probe of §3.5; the per-kind miss breakdown reported **at stratum level to builders**, the fine breakdown (stratum x anchor type x impacted type) written custodian-only under `work/`; `test_gate_fail.py` and docs agree with `STRATA_GATED`; `gate.HARNESS_FILES` sha pin updated last | `analyst` pass on the diff to `HARNESS_FILES` |
| **CO-6b** The record | `data-scientist` text, `teco` commit | CO-2, CO-3, CO-5, CO-6a, CO-6c, CO-8 done; F chosen | `ccg-prereg-v1`/`ac5-gate-1` appended to `cpg/eval/preregistration.jsonl`: `algorithm.version`, harness pin, F, the gated set (http, mcp, scripts) and the withheld set (flow, wiring) with counts and reasons, floors, the six entry-criterion items of the gate-marker note, the freeze-time `check-report` over the path set, the S1 wording, the withheld-shape statement of §3.2, the cluster bound and r^58 table, the Q4 screen rule if chosen | `analyst` + `data-scientist` |
| **CO-6c** Scope reconciliation | `data-scientist`, `architect` | CO-1 ruling | EP §3.4/§3.5 and `-p1-ml.md` §6.1 (and the gate-marker note's last section) state the gated and withheld strata and the three-stratum cluster bound, so `minimumsMet` and the recall bound stop counting wiring; HA-6 row in EP §5.2.1 | `analyst` |
| **CO-7** Oracle fence test | `tdd-engineer` | only if the unit captures an oracle | the roughly 20-line `oracle_runner.py` test and `ccg-oracle-v5` record (`docs/BACKLOG.md` C-1001) | `analyst` |
| **CO-8** `raw_query` memory hardening | `devops` | none; **must land before F** (`cpg/infra/` is core) | `QUERY_MEM_CAPACITY` 256 MiB in `cpg/infra/ccg-falkordb.sh up` and its drift check, after confirming the engine's and build's heaviest queries stay under it | `security-expert` advisory |
| **CO-9** Gate run | `qa-engineer` (rehearsal), custodian and `teco` (run) | F; CO-6b committed | the no-workaround dev rehearsal at F meets §3.5 (a rehearsal miss means a new F and a new record; the rehearsal count is disclosed); one held-out run, counts-only, one-shot custody under `~/.local/share/ccg-eval/cf7b2042cce7/`; verdict per stratum with the cluster bound; on a miss the P1-43 triage and the Q4 decision | `analyst` + `data-scientist` |
| **CO-10** AC-8 real-component run | `devops`, `security-expert` advisory | CO-9 done; Q1 | spec §8.1 features (a) to (d) in `cpg/sandbox/ac8/run.sh`/`lib.sh` (`docs/plans/connected-code-graph-egress-spec3.md` §8.1; `docs/test-reports/connected-code-graph-p1-ac8-report.md` §4); component images; a local-model path from a no-route network; the 13 unrun rows computed on real components | `analyst`, `security-expert` |
| **CO-11** AC-24 retry | `data-scientist` (model note, out-of-sample design), `qa-engineer` | CO-9 done; Q5 | a candidate model on a **new** dev set (R4b and the "path line N" residual were designed against the spent set; fresh items by a blind custodian); the entry check 11/12 and 2/2, or "not met" against the **a-priori minimum** the stakeholder sets at the start of this unit, without spending the held-out set (`-p1-ml.md` §4.4); only then `ccg-prereg-ac24-v1` | `analyst` + `data-scientist` |
| **CO-12** AC-10 observation | `qa-engineer` | CO-9 done (host free); a build | a live model through the broker runs a `teco`-style architect brief and an implementer brief on a covered system; the three AC-10 behaviours inspected; finding file for P1T-39 | `analyst` |

CO-4 (wiring) is removed (§3.4).

## 5. Sequence and sharing

```
CO-6a first. CO-1 -> CO-2(+2a) -> CO-3 -> freeze commit F -> CO-6b -> CO-9 (rehearsal, then held-out run once)
CO-5, CO-6c, CO-8, (CO-7): parallel, all finished before F (CO-8 and CO-7 touch core)
after CO-9, before P3: CO-10, CO-11, CO-12 (host: one at a time)
```

- **F is the last commit changing anything core** (§2): `ccg/`, `extractors/`, `infra/`, plus the packs and harness pins. CO-8, CO-2, CO-3 and CO-7 all land before F. **CO-10 and CO-11 change `ccg/front/` and `infra/` and so would move F: they run after the gate**, and the AC-5 verdict is the verdict for F only. Later core changes are not re-gated against AC-5; the next full AC-5 assertion is the P5 re-run on fresh held-out items.
- **Shared files:** CO-2 and CO-3 share `cpg/ccg/assemble/rows.py`, `cpg/extractors/shell/extract.py` and `cpg/extractors/markdown/extract.py`: serialized. CO-5 owns `cpg/eval/tools/gate*.py` and the harness pin.
- **Shared host:** a full build is about 30 minutes on this 6 GB host; CO-1, the CO-9 rehearsal, the CO-9 run and CO-12 each build a throwaway store, one at a time. CO-10 and CO-11 both use LM Studio and do not run together.

## 6. Minimum viable close-out

CO-6a, CO-1, CO-2 (+2a), CO-3, CO-5, CO-6b, CO-6c, CO-8, CO-9 (and CO-7 only if an oracle is captured). CO-10, CO-11, CO-12 do not feed the gate; they follow it and precede P3 by the stakeholder's decision. AC-8 closes only at P2 exit; CO-10 closes its real-component part.

## 7. Test strategy

- **Unit and mutation (CO-2, CO-3):** one red test per mechanism on synthetic trees (§4 CO-2 list; for CO-3 a `curl` with a template URL and an inline span that only mentions `curl` in prose); mutants on the tier-1/tier-2 `confirmed`/`possible` split and the suffix-length rule; existing suites stay green (R1 to R8).
- **Dev measurement per unit** proves the done-conditions, dev gold only, no tuning on held-out; a fix that names a dev item is a defect (mechanical scan, §3.5).
- **Noise:** every added edge raises the `possible` count; the rehearsal's Q and wrong-`possible` bound must sit inside the AC-5 bounds (entry item 3).
- **Contract:** answer-schema validation V1 to V16 on every new answer shape; no algorithm bump is expected, a bump re-pins `ALGORITHM_VERSIONS` and goldens in the same commit.

## 8. Documentation impact

`docs/requirements/connected-code-graph.md` decision log (`tico`): the answers of this round, the wiring wording change to AC-2/AC-4/AC-5, and whatever Q1, Q4, Q5, Q8 become. `docs/plans/connected-code-graph-p1.md` header gains `Extended by:` this document (`teco`). `-p1-linking.md` P2 deferral lists (~l.563, 929, 954) and `-p1.md` ~l.518: items built here removed in place at the unit's close (`architect`). `docs/plans/connected-code-graph-rule-pack-spec.md` §12.1 gains the new pack row; `-p1-extraction.md` P1-12 and `cpg/extractors/shell/README.md` the Markdown/shell contract lines. `-p1-gate-marker-ml.md` and `-p1-ml.md` per CO-6c. `docs/HISTORY.md`/`BACKLOG.md` (`teco`/coder) record delivery; `cpg/README.md` sandbox row when CO-10 lands. The live `cpg/` spec stays owed at the feature close after P2.

## 9. Risks

- **A first-spend miss is the likely outcome** (§2, r^58). Withheld script shapes (§3.2) make it likelier: a held-out analogue is a certain miss. The cost is the list and a new blind draw (key-writers, capture-recapture, manifest re-record); wiring's fresh supply is probably nil.
- **Resolver noise:** tier 2 adds `possible` edges; a one-segment-tail rule and uniqueness bound it, the rehearsal measures it.
- **Carrier uncertainty (§3.1):** if the `valueSet` does not carry the tail, an extractor change is needed; a bundle field is a stakeholder-visible amendment.
- **Fixture goldens** may move (§3.6).
- **Host contention** serializes builds; the AC-8 and AC-24 tracks wait on LM Studio and images.

## 10. Open questions

Answered (2026-11-24, folded above): wiring withheld; the resolver's last tier; CO-10 and CO-11 after the gate; S1 both; CO-12 included.

Still open, not decided here:
- **Q1 (Q-E): how AC-8 closes without the git-fetch path.** (A) Close part 1 on the clauses that have a built path; the fetch, download and hosted-override clauses are "vacuous until built", re-asserted at P2 exit. (B) Build a fetch path first (reverses egress Q1 = C). (C) Amend AC-8's wording. (D) Defer CO-10 wholly to P2 exit. Recommendation: A.
- **Q4: a first-spend failure.** (a) Accept: missed clusters move to dev, a new blind draw follows. (b) Narrowed: before the spend, a fresh draw of about 25 items written by a blind key-writer, disjoint by anchor from the held-out and reserve lists (the custodian discards overlaps and reports a count), read once, its decision rule pre-registered, its items dev afterwards; a gross-failure detector only (25 items with no miss gives a lower bound near 0.87); adds a custodian-owned unit. (c) Ask `tico` whether AC-5's every-item rule applies to a first spend or only to a pre-announced final one. Recommendation: (c), then (b) if the key-writers are cheap, else (a).
- **Q5: AC-24.** The a-priori minimum acceptable local target, asked when CO-11 starts; the model choice is a method call (`data-scientist` note first); options for what to do if it is not reached: a lower local target set from the measurement (AC-24 text), a hosted model through the broker, or deferral past P3.

**New decision needed from the stakeholder:**
- **Q8: scripts shapes the plan does not build (§3.2: about 7 dev items, held-out analogues unknown).** (i) Build them before the spend (the Python-subprocess, `-m` and front-matter-hook shapes each need new extraction; `-m` is a listed P2 item). (ii) Spend knowingly: the verdict is likely `fail` if the held-out holds an analogue, and the list is spent. (iii) Withhold the shapes at the gate: needs a shape-level exclusion applied by the custodian from a pre-registered predicate, counts only, with a `data-scientist` ruling that it does not weaken AC-5. Recommendation: (i) for the cheap ones CO-1 identifies, (iii) for the rest if `data-scientist` accepts it, else (i) or an explicit (ii).
