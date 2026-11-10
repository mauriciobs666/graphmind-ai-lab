# python-web-quirks on the knowledge base — the most promising path to a verdict-grade pass (B3)

> **Status:** active · **Owner:** `data-scientist` · **Tracks:** — (U36; **Extends:** `docs/plans/python-web-quirks-kb-embedder-ml.md`; relies on `docs/plans/python-web-quirks-kb-b2-ml.md` §4 and `docs/plans/python-web-quirks-kb-ml.md` §4)

## 1. Question, recommendation, honest odds

**Decision served.** The stakeholder reverses the earlier stop and asks for the path with the best chance of a verdict-grade AC-3 pass after sealed B (73/84) and B2 (110/126, near-domain 9/30) failed at the 0.6b. The caller needs: what to run, under which fixed rule, and who does what.

**Recommendation.**

1. **One configuration, one fresh seal (B3):** `text-embedding-qwen3-embedding-4b` (2560 dims), the retrieval skill's query literal **unchanged**, floor **0.42**, top-K 5, chunking unchanged, in a 4b-embedded **staging** workspace on `:8210`. Rule R1 (§5) is the approved 0.90 / 20% / 0 gate scaled to B3's size. This is the embedder note's configuration; this round tested every other lever offline (§3) and none adds to it on non-sealed data.
2. **My odds that B3 passes R1: about 0.35, plausible range 0.1 to 0.5.** A model-based judgment (§4), not a measurement. The embedder note said 0.15 with a deliberately halved transfer; the difference is a better-grounded calibration of Set A to the sealed sets (§4), not new evidence that the 4b is better. If you want the odds materially higher, the lever is not a retrieval trick but the gate and the agent-facing convention (§2, §6 fork F2).
3. **Hybrid lexical+vector, query rewriting, symptom "search hooks" and a retry are rejected on offline evidence** (§3). The one query-side change with a measurable (small) effect, a content-class instruction literal, is held as a reserve: it needs an FR-7 amendment and its gain does not survive the multiplicity count (§3.5, §7).
4. **If B3 fails the verdict is final** (§5). **Not done, not recommended:** scoring B or B2 with any new configuration.

## 2. Why the gate fails, and whether the gate is the right premise

**Evidence this run** (offline; the cached embeddings of the U35 run reused, every metric recomputed; `ws:pwq-test` 902 chunks, 129 unit chunks; Set A 42 positives in four forms, 7 near and 7 off negatives; sealed files not read). Control: the 0.6b reproduces the embedder note exactly (raw top-5 42/39/29/40, floor-applied 42/36/22/33 at 0.43; 4b at 0.42: 42/41/38/41 and 42/37/26/36).

**Three failure mechanisms, not one:**

| Mechanism | Sealed evidence (report aggregates) | Verdict from the offline evidence |
|---|---|---|
| Floor loss on positives (score level shifts with query register) | B 4 of 77, B2 9 of 119 raw hits lost to 0.43 | Fixable in part. Mean floor-applied hit rate over the three perturbed forms of Set A: 0.6b 0.722, 4b 0.786 at its rule floor 0.42 (0.825 at 0.43). |
| Rank misses (unit outside the top five even without the floor) | B 7/84, B2 7/126 | Mostly competition and register, and not what the gate binds: with the corpus restricted to unit chunks the 4b's raw top-5 on Set A is 42/42/41/42 against 42/41/38/41 all-workspace, yet **floor-applied hits are identical either way** (the floor, not the rank, decides). |
| Near-domain false accepts | B 3/15, B2 9/30 (pooled 12/45 = 0.267) | **Structural for any score floor over a same-register corpus; reducible, not removable.** §2.1. |

### 2.1 A larger near-domain negative proxy (leave-one-unit-out)

Set A has only 7 near negatives, too few to compare levers. Non-sealed construction: for each Set A positive query, **delete that query's whole unit from the candidate pool** and run the query against the other 35 units plus the 773 non-unit chunks. The gotcha the query describes then does not exist in the KB, and the nearest remaining units are real neighbours (the same TestClient, pytest or pool sections): exactly the situation a near-domain negative stands for. 42 queries x 4 forms = 168 negatives clustered in 36 units.

- **Calibration.** 0.6b at 0.43: false-accept rate over the 168 is 51/168 = 0.304 (full form only 17/42 = 0.405); the sealed pooled near-domain rate is 12/45 = 0.267. So the proxy is the right order of magnitude, a little harder than the real negatives (ratio 0.267/0.304 = 0.88, used below as a scale factor with a wide band).
- **Exchange rate.** At matched mean perturbed-form hit rate near 0.78: 0.6b at 0.45 gives 0.778 with false-accept rate 0.435; 4b at 0.42 gives 0.786 with 0.185. Pooled-forms AUC of "best-unit score separates expected unit from LOO negative": 0.6b 0.839, 4b 0.901; difference +0.062, cluster bootstrap by unit (36 units, 400 resamples, seed 11) 95% [+0.026, +0.103]. **The 4b is a real, not marginal, improvement in separation; no floor on the 0.6b buys the same.**
- **What it cannot do.** The 4b's proxy false-accept rate rises from 0.185 (0.42) to 0.298 (0.45) to 0.500 (0.50): there is no floor at which recall is high and false accepts negligible. Separation improves; it does not become clean.

### 2.2 Is a near-domain cap of 6/30 the right premise?

Requirements and ML §4: FR-12/AC-3 defer the numbers to ML §1/§4, and the decision log (2026-11-09) approved them "as proposed". The cost reasoning in the documents cuts both ways:

- **Cost of a false accept**: ML §4.2: "one wrong ~1.4 KB section read". FR-5/the skill: `get_document` returns title, versions, date, so a wrong hit is recognisable on read. The cap was set at the same 20% as the prior art's rate, not derived from a cost comparison.
- **Cost of a miss**: ML §4.1: "A miss is not silent (FR-11 title listing and FR-9 section reads are the fallback), so no 0.95 gate". The skill already tells the agent to run the listing when nothing is under the floor.
- So the approved asymmetry (strict on false accepts, 0.90 on recall) is a judgment the requirements neither contradict nor justify. **The requirements already allow a cheaper design in spirit** (hits show title and date; a fallback exists for misses), **but not in letter**: AC-3 scores "returned under the score floor", FR-7 forbids forking the convention, FR-12 fixes the targets. A two-tier convention for this content class (strong at or under the floor; weak band up to about 0.47 with "read the title; if it is not it, run the listing") would change FR-7, FR-12 and AC-3 and **needs a stakeholder decision** (F2). Offline, for the 4b: recall at 0.47 is 0.897 mean perturbed hit (42/42 full) against 0.786 at 0.42, at the price of showing a (flagged) wrong hit for about 40% of the proxy negatives. Whether an agent dismisses such a hit is an agent-judgment property that no retrieval metric measures; it would have to be tested (§6, W8) before anyone relies on it.
- I do not recommend lowering the bar to obtain a pass: a restated gate is legitimate only as a cost-of-error argument made by the stakeholder **before** the seal (pre-registered, §5 R2), never fitted to B3's result.

## 3. Levers tested offline (non-sealed data only)

All on the 902 chunks and Set A (plus the 168-negative proxy). "Pass probability" is the §4 model's joint P under R1 for the configuration that lever would make; all are estimates.

| Lever | Offline result (observed this run) | Cost / blast radius | Verdict |
|---|---|---|---|
| **3.1 qwen3-embedding-4b alone** | Pooled AUC vs 0.6b +0.062 [+0.026, +0.103]; raw top-5 on second-half queries 38/42 vs 29/42; real Set A near false accepts 0/7 vs 2/7 at their rule floors. | Per-workspace migration; production blast radius in §6. | **Take.** P about 0.35. |
| **3.2 Hybrid lexical+vector** | (a) Normalised-BM25 score separates worse than vector (AUC pos vs proxy, full form: 0.832 vs 0.942 at 0.6b; combining equally: 0.904 / 4b 0.926 vs vector alone 0.986). (b) Rank fusion (RRF, top 5) raises the 0.6b's raw recall on first/second-half forms (39/29 to 41/35) but not the 4b's (41/38 to 41/36), and floor-applied hits fall (4b 141 to 137 summed over four forms). (c) "Vector AND lexical top-K" cuts proxy false accepts (4b: 31 to 24 to 27 summed) at a larger loss of hits (141 to 123 to 131); "vector OR lexical at a looser floor" gains 4 to 13 hits for 11 to 25 more false accepts: both worse than just moving the floor (4b 0.42 to 0.43: +5 hits, +6 false accepts). | Server: no `Chunk.text` full-text index exists (full-text indexes are `Message.text`, `Entity.name`, `Document.title` only, `docs/DESIGN.md` §7); `search_chunks` is vector-only. Needs a per-workspace opt-in index (the RAM objection that kept `Document.text` out applies at scale), RediSearch query escaping for identifiers such as `os.killpg` and `-q -q`, a fusion layer, a tool/API parameter, contract tests and a SERVER.md change: a feature of days, estimate, with `architect`, `coder`, `graph-dba`, `qa-engineer`, touching the MCP schema every consumer sees. | **Reject.** Lexical match adds rank, not discrimination, and the sealed BM25 wins (84/84, 125/126) came from a 129-chunk pool without a floor. |
| **3.3 Query-side: rewrite / retry** | A local model (qwen3-4b-2507, temperature 0) rewrote each query into a one-line "behaviour and identifiers" form. Appending it: AUC 4b full 0.986 to 0.990, second-half 0.823 to 0.823; rewrite only: 0.929 and 0.738 (worse). 0.6b append: 0.942 to 0.956, 0.728 to 0.757. Rewrites were often wrong, so a stronger agent would do better; that is unmeasured. A retry on a miss is a second draw of false accepts and cannot be gated by a fixed query set. | Skill text only. | **Reject as a gated lever.** The existing skill guidance (full one or two sentences, identifiers kept) stays a non-gated improvement (b2 note §3); B3's author reads the shipped skill (§5). |
| **3.4 Unit-side "search hook" line** | A blind (from unit text only; Set A never shown) two-sentence symptom line per unit by the same local model, inserted into chunk 0 and re-embedded: 4b pooled-form AUCs 0.986/0.930/0.823/0.888 to 0.988/0.943/0.830/0.900; hits and false accepts move by 1 to 3 of 168. | Header-format change (the header is outside FR-2's verbatim comparison, so lawful), regenerating 36 documents, **a production re-ingest that needs the stakeholder's deletes** (the tool refuses production `replace`), manifest and gate rework; tension with "verbatim migration". | **Reject.** No measurable effect on symptom-phrased queries; where it could help (design-time "should I" queries) Set A has no signal. The hooks were generated by a 4B model; a human-written hook would be stronger, and untested. |
| **3.5 Content-class instruction literal** (reserve) | One variant fixed in advance (instruction naming "surprising behaviour, error or design decision in Python, web-framework or test code"). 4b pooled-form AUC 0.901 to 0.918, +0.017 [+0.004, +0.035] (0.6b: +0.009 to +0.050 across forms, 8 of 8 comparisons favour it). Its rule floor moves to 0.436 (0.6b 0.415). | Second query literal in the skill's Python/web section; FR-7 amendment ("without changing its guarded literal"); `audit-team.sh` check; the tool must take a template. No re-embedding. | **Reserve, not in B3**: the lower bound survives no multiplicity correction (§7), P moves 0.38 to 0.42 in the model: inside its own noise. |
| **3.6 Two-stage rank / lexical rescoring of unit chunks** | Same data as 3.2 (RRF and AND/OR rules are two-stage forms). | | **Reject** with 3.2. |
| **3.7 Chunking** | Closed by the embedder note §5. | | Not revisited. |

A cross-encoder reranker is not among the 20 chat or four embedding models LM Studio lists here, so it is untested; it would be the one rescoring layer worth a future look.

## 4. Pass probability under the R1 rule (model, with its assumptions)

Anchored only on observed numbers: the sealed aggregates (0.6b pooled floor-applied recall 183/210, near-domain 12/45) and this run's Set A tables.

- **Positive recall transfer.** The 0.6b's real floor-applied recall (0.871) sits between its Set A full-form rate (1.0) and its perturbed-form mean (0.722): weight 0.463 on the perturbed side (sd 0.12 assumed). For the 4b at 0.42: 0.537 x 1.0 + 0.463 x 0.786 = 0.901; at 0.43: 0.919. Extra noise sd 0.025 and a winner's-curse shift of -0.01 applied.
- **False-accept transfer.** Real near rate = 0.88 x proxy all-forms rate (lognormal, sd 0.3). 4b at 0.42: proxy 31/168 = 0.185, so about 0.16.
- Draws beta-binomial (clustering rho 0.05), 20,000 per configuration, seed 7. Results (joint P(recall at least 114 of 126 and near at most 6 of 30)): 4b at 0.42 **0.38**; 4b at 0.43 0.38; with threshold 116 (B2's) 0.32 and 0.33; 0.6b control 0.17 (the control predicts a coin-flip-or-worse and the real result was a fail twice, which is consistent, not a validation).
- **Dominant uncertainties**, in order: the transfer weight 0.463 rests on one model pair and one sealed pool; the proxy is not real near-domain text; the 4b's advantage was picked as the best of many looks (§7). I therefore state **0.35 (range 0.1 to 0.5)**. The configurations differ by less than their own noise, which is why I recommend the simplest one.
- Why the floor stays at 0.42: it is the embedder note's rule value (1.03 x the largest full-form positive's best score), fixed before any seal; at 0.43 hits rise (+5 over four forms) and false accepts rise (+6), P unchanged.

## 5. The B3 protocol (fixed now)

**Spent instruments.** B and B2 stay spent: never reopened, no per-query data read, not re-scored with any model, and no aggregate of them is used as a pass criterion for B3. The pooled B+B2 figures stay descriptive.

**Author.** A third agent: not `data-scientist` (A), not `qa-engineer` (B), not the B2 author (the coordination ledger names it; `teco` assigns, `architect` or `analyst` if neither wrote B2). It reads the 36 unit titles, the pinned source, and the retrieval skill **as shipped** (so the register is what an agent following the convention would type, with no coaching toward Set A); it reads no Set A, no S14/S14b section, no B/B2 material and none of this note's §2 to §4. Where the stakeholder can supply 20 to 30 real recent agent situations, they replace synthetic ones.

**Contents.** Identical shape to B2: 126 positives (3 per section: symptom/error, design-time "should I", free choice; tail probes for units with 4 or more chunks), 30 near-domain and 30 off-domain negatives written to sound like a section and checked against titles only; 4-gram leak check (`claude/cobb/scripts/pwq_leakcheck.py`), the rewrite-only-if-rank-1-under-0.25 probe rule, frozen outside the repo, sha256 in the ledger, scored once, aggregates only. The length and word count of each query is recorded as a covariate (H1 of the b2 note: register is the dominant risk).

**Frozen configuration (recorded before delivery to the scorer).** Embedder `text-embedding-qwen3-embedding-4b` and its LM Studio file (as listed by `GET /v1/models`), 2560 dims, documents raw; query literal exactly as in `skills/agent-kb-retrieval/SKILL.md`; floor 0.42; top-K 5; the 36 current unit documents plus the staging workspace's competing corpus, all 4b-embedded; tool commit, manifest, copy and skill hashes; the staging document list's sha256. **Fidelity gate before the seal** (Set A only, through the staging server): floor-applied hits within 1 of the offline figures (42/37/26/36 over the four forms), real near false accepts at most 1 of 7, off-domain 0 of 7, `gate` clean; a larger gap is investigated before anything is sealed, and the floor is never re-tuned.

**Decision rule R1 (binary, all must hold):**

1. floor-applied positive recall at least **114/126** (0.905, the approved 0.90 gate that B scaled to 76/84, applied to a fresh set) **and** every section hit by at least one of its three queries;
2. near-domain false accepts at most **6/30**; off-domain **0/30**; overall at most 6/60.

The tool also prints the 116/126 clause value (B2's pooled-derived bar) as a reported figure. **The one departure from B2:** B2's 116 existed only to carry B's failed 73 into a pooled 0.90 for the unchanged configuration; a different configuration does not inherit that pool. The sensitivity is 0.05 in pass probability (§4); the stakeholder may veto it, in which case R1 reads 116.

**R2 (optional, only if the stakeholder approves the two-tier convention before the seal; pre-registered, never added afterwards).** Same data, same author, same freeze: recall counted at a weak floor of 0.47 (strong band at or under 0.42) at least 114/126 and every section hit; near-domain false accepts **in the strong band** at most 6/30; off-domain 0/30 in the strong band and weak-band off-domain exposure reported; weak-band near-domain exposure reported. Model P about 0.55. A R2 pass without an R1 pass ships only the two-tier convention and only after W8.

**Multiplicity.** One configuration, one seal, R1 as the primary test; R2, if registered, is a declared second chance on the same data, so a pass on R2 alone is a conditional pass of a different product, not an R1 pass.

**What a stop looks like.** B3 fails R1 (and R2 if registered): (1) AC-3 recorded FAIL on B, B2, B3 with per-set aggregates; (2) the stated deviation from the measurements (B, B2, B3 per-configuration, pooled per configuration only, Wilson intervals, section-clustered); (3) FR-9/FR-11 fallback is the product and `skills/python-web-quirks/` is kept or retired on the stakeholder's cost-of-miss argument; (4) no production floor, chunker or embedder change; (5) the finding "score-only retrieval cannot reach the approved gate on this content with the local embedders"; a re-stated gate, a reranker or a hosted embedder is a **new feature with its own requirements and its own sealed set**, not a B4.

## 6. Work breakdown, owners, dependencies

Stakeholder forks: **F1** veto or accept R1 at 114 (default accept); **F2** register R2 and the two-tier convention, or not (default not); **F3** authorize migrating `ws:pwq-test` to the 4b on `:8210` (nothing on `:8200`). I made the routine calls (4b alone, literal unchanged, 0.42, 114) and ask only these.

| # | Owner | Work | Needs | Gate |
|---|---|---|---|---|
| W1 | `graph-dba` (+ `devops` for the `:8210` process and LM Studio) | Stop the `:8210` process, `scripts/migrate_embeddings.sh pwq-test lmstudio/text-embedding-qwen3-embedding-4b`, overlay `dim: 2560`, restart; verify both indexes at 2560, unit documents fully embedded (K-102 leaves 16 chunks of the workspace unembedded today: they must not be unit chunks), `ws:agent-team` untouched | F3 | index introspection |
| W2 | `coder` or `tdd-engineer` | `pwq_kb.py`: `--set B3` profile (126/30/30, R1 constants, `--set-b3-path`/`PWQ_SET_B3`, its own sentinel), print the 116 clause value, optional R2 weak-band reporting (only if F2); mutation tests; B2 constants untouched | F1, F2 | selftest green, mutants red |
| W3 | `qa-engineer` | Staging fidelity gate (§5) and the frozen-configuration record (hashes, document-list sha256) | W1, W2 | fidelity numbers inside tolerance |
| W4 | `teco` | Assign the B3 author, brief them as in §5, freeze, ledger the sha256 | F1, F2 | leak check, sha recorded |
| W5 | `qa-engineer` | Single sealed scoring on staging (`--exclude` of the S7 ids that still exist there), report S14c, aggregates only | W3, W4 | one invocation, sentinel written |
| W6 | `data-scientist` | After scoring: aggregate-level reading only (no instrument change) | W5 | |
| W7 | on a pass: `architect` plans; `graph-dba` migrates `ws:agent-team`; `qa-engineer` re-runs the Stage 8 45-pair regression and re-derives the **skill-wide** floor; `cobb` edits the skill; `devops` for the outage window | Blast radius flagged in the embedder note §3: the whole `:8200` server stops (including `ws:demo` and `salesperson`), every KB consumer inherits the 4b and its floor | pass, F-decision | separate decision unit |
| W8 | `qa-engineer` + `data-scientist`, only if F2 | Agent-in-the-loop test of the weak band: given the shown wrong hit's title and header, how often does an agent dismiss it (on the near-domain negatives and on proxy negatives) | F2 | before any two-tier shipping |

Order: F1-F3 then (W1 and W2 in parallel) then W3; W4 in parallel with W1 to W3 (the author must not see W3 output); W5 after W3 and W4; W7 only on a pass.

## 7. Multiple-comparisons budget on Set A, and risks

**Looks at Set A so far** (counting configurations or variants evaluated, not metrics): S12 floor sweep (13 steps); b2 note remedies (about 6); embedder note (5 embedder configurations, 6 chunk variants x 2 models); **this round about 41 rule or variant evaluations** (hybrid rules 3 operating points x 9, three score types x 2 models, hooks 2, rewrites 4, instruction literal 2) **plus four floor-sweep curves** (12 floors each) and 7 model-configuration comparisons. None is a significance test against a null. The only inferential statements are three **pre-chosen** cluster-bootstrap contrasts (4b vs 0.6b, class literal vs standard, class literal vs 0.6b), 400 resamples; at three tests a Bonferroni level would be 98.3%, and the class-literal lower bound +0.004 would not clear it. That is why the class literal is a reserve. The 4b is the best of all looks: its Set A advantage is optimistic by construction and the odds in §4 already shade it. B3 is one pre-specified test, so the multiplicity caps the claim (4b clears the rule), not B3's validity.

**Risks and open items.**
- The leave-one-out proxy and the perturbed forms are not B/B2's query distribution; the 0.463 weight, the 0.88 scale and the odds rest on two sealed aggregates.
- Staging is not production: `ws:pwq-test` carries 902 chunks against production's 882 and the S7 captures; documented, expected conservative.
- Floor instability across embedding sessions (the skill's named R6 case) applies to the 4b too; the 4b margin to the closest real near-negative on Set A is thin (0.422 against a rule floor of 0.42).
- Not measured: the 4b's token truncation on the longest units (chunks are at most 1,142 characters), per-query latency in production, other workspaces' behaviour after a migration, whether a human-authored hook or a stronger rewriter would help.
- Scratch analysis scripts are not in the repo; `W3` reproduces the fidelity numbers through the staging server, which is the authoritative check.
