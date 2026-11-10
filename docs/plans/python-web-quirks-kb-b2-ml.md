# python-web-quirks on the knowledge base — why Set B failed, remedies, and the Set B2 protocol

> **Status:** active · **Owner:** `data-scientist` · **Tracks:** — (U33; extends the method note `docs/plans/python-web-quirks-kb-ml.md`, AC-3, plan S14) · **Extended by:** `docs/plans/python-web-quirks-kb-embedder-ml.md`

## 1. Question and recommendation

**Decision served:** the stakeholder chooses the path after the sealed Set B failed the positive arm
(`docs/test-reports/python-web-quirks-kb-report.md`, section "S14"): what to do about the gate, and whether
`skills/python-web-quirks/` may be deleted (plan S16d).

**Recommendation.**

1. **The verdict stands: AC-3 FAIL on B. `skills/python-web-quirks/` stays blocked** until a B2 verdict or an explicit stakeholder deviation.
2. **Change nothing in the retrieval design.** No floor, chunking, title-prefix, embedding-text, limit or hybrid change. Everything I could test without B either does nothing or trades one arm of the gate for the other (§3). Offline evidence cannot validate a remedy anyway: Set A is saturated (§2).
3. **Run one B2 on the unchanged configuration**, authored by a different agent, sealed, with the decision rule of §4 fixed first. Expect a probable fail (joint pass probability roughly 0.1 to 0.3 if the true rates equal B's; §4): B2's value is a second, independent estimate, so the stakeholder's deviation call, if it comes, rests on 210 positive queries instead of 84.
4. **If the stakeholder prefers not to spend B2**, the honest alternative is a deviation stated as a measurement, not a pass: floor-applied recall 73/84 = 0.869 (Wilson 95% 0.781-0.925), raw top-5 77/84 = 0.917 (0.838-0.959), near-domain false accepts 3/15 (0.070-0.452), with FR-9/FR-11 as the fallback for a miss. That is a cost-of-miss decision for the stakeholder, not a method finding.

## 2. Diagnosis (hypotheses ranked)

Non-sealed evidence only, all observed this run on `ws:pwq-test` (`:8210`, read-only searches) and on Set A. B facts are the report's aggregates. I have not read Set B.

The 11 lost positives split into two mechanisms: **7 rank misses** (unit not in the top five even with no floor) and **4 floor losses** (unit in the top five, best score above 0.43).

**H1 (most likely): the 0.43 floor is fragile under any change in query form, and A could not show it.** The floor loss is the mechanism that moves most when queries get less verbatim. Mechanical perturbations of the 42 Set A queries (first half of the words, second half, every third word dropped; no authoring), floor-applied hits of 42, raw top-5 of 42:

| variant | raw top-5 | floor 0.43 | best unit score above 0.43 |
|---|---|---|---|
| full | 42 | 42 | 0 |
| first half | 39 | 36 | 6 |
| second half | 29 | 22 | 19 |
| drop every 3rd | 40 | 33 | 8 |

Cosine distance is not invariant to query length or register; ranks hold much better than absolute scores. Set A's own margin was already thin: the worst found positive scored about 0.417 (ML §6: margin failed by 0.0125 and `adequate=False` at every step), 10 of 42 positives scored above 0.35 and 3 above 0.40. B added 84 differently phrased queries and 4 of 77 found ones crossed 0.43. This was predicted by the S12 finding, not a surprise. Supports: the table; the S12 margin; B's 4 floor losses. Refutes nothing. Not testable without B: the B queries' length and register (I will not measure them).

**H2: competition in the real workspace (non-unit documents and other units' chunks) causes the rank misses.** Set A's displacement was 3 of 42 (rank only, never a miss); B's 7 of 84 is the same rate (0.071 against 0.083), but on B it came with 7 raw misses. Like-for-like on A (semantic restricted to unit chunks, as BM25 sees them, against BM25 over the 129 unit chunks, top five): full 42 against 41; first half 41 against 39; second half 36 against 37; every third dropped 41 against 40. So **BM25's 84/84 on B is mostly the design handicap** (no non-unit competitors, no floor), not a semantic-ranking deficiency; semantic all-workspace raw recall was 39, 29 and 40 on the same variants, so competition costs 1 to 7 per variant and the floor costs 3 to 7 more. Supports: unit-only against all-workspace gap; 26 of 42 Set A positives with a pre-existing hit (ML §4.2). The condition ML §4.3 states (semantic must beat lexical at equal top-5) is not met on B as measured, but the measurement is not equal-footing; the like-for-like check on A shows parity. The 7 displacement queries and the 7 raw misses are equal in count; the tool prints no per-query data, so I do not claim they coincide.

**H3: Set A is easy, so "A passed" carried little information.** Median best unit score 0.28; BM25 41/42 on A; the A author (me) wrote from the section's opening paragraph with a 4-gram leak check, and the sealed-set author used two framings per section and tail probes. A 42/42 against B's 73/84 is not chance if both were one population (hypergeometric P(no miss in A | 11 misses in 126) = 0.009, independence assumed, so it is optimistic given clustering); the raw-rank difference alone is borderline (P = 0.054). Treat A as a necessary check, never as evidence of recall. Wilson 95% on 42/42 is 0.916 to 1.0, which B's 0.781 to 0.925 barely touches.

**H4 (weak): head-chunk shortfall / untitled tail chunks / chunk 0 boilerplate.** Not supported. The returned chunk position is not recorded by the tool (D-1), so B cannot say it. On A, the best-scoring chunk is chunk 0 for 13 of 42 positives (31%), against 28% expected if chunks were exchangeable (129 chunks, 36 units); best chunk seq distribution 0:13, 1:16, 2:5, 3:6, 6:1, 7:1. The report's "head" class is "not tail-flagged" (71 queries, 60 hits = 0.845), the tail class 13 of 13; the 13 tail probes target specific late facts, which are easier to retrieve, so tail against head is not a chunk-0 measure. Offline embedding test with the production embedder (LM Studio `text-embedding-qwen3-embedding-0.6b`, 129 unit chunks, validated against the server: same top score 0.3701 and identical unit-only counts 42/41/36/41 for the four variants): putting the title on every chunk (V1) or stripping the shared chunk-0 header (V2) changes floor-applied hits by +2 (V1) and +1 (V2) of 168 query variants and raw by +2 and +1, with mixed negative effects (V1: NA01 0.424 to 0.441, NA02 0.446 to 0.436). That is noise at this size. Chunk or title changes are not a lever I can show.

**H5: the floor should be per document rather than per chunk.** There is no difference to find: a hit is already "any chunk of the unit within the floor", i.e. the document's best chunk. A rank-relative accept rule (also accept 0.43 to 0.50 if the unit chunk is top-1 overall with a gap of at least 0.05 to the best non-unit chunk) recovers 0, 1, 3 and 0 of the perturbed floor losses on A (full, first half, second half, drop-3rd) and adds no near-domain accept on A; too small to matter.

**The negative arm (3/15 near-domain, at its limit) is structural.** A near-domain negative scores like a positive: on A the seven near negatives' best unit scores are 0.380, 0.424, 0.446, 0.484, 0.493, 0.506, 0.573 against positives up to 0.417; raising the floor trades along a fixed exchange rate (A near false accepts 2/7 at 0.43, 3/7 at 0.45 to 0.47, 5/7 at 0.50; floor-applied hits on the perturbed variants gain 3 to 5 of 42 at 0.47 and 3 to 7 at 0.50). B's 3/15 sits exactly at the 12/15 gate, so any floor rise most likely fails the near-domain clause; ML §6.3's fallback ("score alone cannot separate these") applies.

## 3. Candidate remedies

| Remedy | Evidence | Risk | Verdict |
|---|---|---|---|
| Raise the floor (0.45 to 0.50) | Recovers some floor losses (above); costs near-domain rejection on a metric already at its limit | Moves every KB user (FR-7), needs the Stage 8 45-pair regression; tuning on A after seeing B | Reject |
| Chunking / title-prefix / header strip | Noise-level on A (H4) | Re-ingest of 36 production documents, burns the "no replace/delete" safety posture, retests everything | Reject |
| Top-K 10 instead of 5 | Floor-applied hits on the perturbed variants gain 0 to 1 of 42 at floor 0.43 (1 per variant at 0.45) | Convention literal (FR-7); more tokens per call | Reject |
| BM25 / hybrid | Semantic ranking is at parity with BM25 like-for-like (H2); BM25 has no floor so is not a negative-arm rescue | `search_documents` is vector-only; a lexical layer is server work (`architect`/`graph-dba`), outside this feature | Defer; revisit only if B2 shows semantic misses that lexical would catch |
| Rank-relative accept rule (H5) | Recovers 0 to 3 of 10 to 20 floor losses | Agent-side logic nobody has gated | Reject |
| Query-writing guidance in the retrieval skill (full 1-2 sentence situation, identifiers kept; H1: truncated queries lose floor hits) | Perturbation table | Zero cost; effect on real agents unmeasurable by a fixed-query set, so it earns no gate credit | Do it, as a non-gated improvement, after the verdict (cobb, S15a-style edit); never cited as a remedy for B |
| Weak tier: show hits 0.43 to 0.50 flagged "verify against the title" | Would recover most floor losses | Changes the convention and needs an agent-judgment test the retrieval metrics do not provide; skill-wide | Not now; a stakeholder design decision if B2 also fails |
| Accept with plan deviation | §1 item 4 | A stated deviation after a failed seal is legitimate only as a stated measurement, not as a re-scored pass | Stakeholder's option (§5) |

**Overfitting risk of tuning on A after B.** A (42 + 14) is saturated: 42/42 leaves no headroom to rank remedies, and the perturbed A used above is the same 42 sections re-derived, not independent. A remedy chosen by A-plus-B-aggregates is selected on a signal that already contains B's failure, so a pass on B2 afterwards is partly selection. Hence one configuration goes to B2 (the unchanged one), and any change after B2 fails needs a B3 and a different author again. I declare the candidates I tried here (floor, per-chunk title and header embedding variants, top-10, margin rule, BM25 like-for-like); none was selected.

## 4. Set B2 protocol (falsifiable, fixed before it exists)

**Author:** a different agent from `data-scientist` (A) and `qa-engineer` (B). Recommended `tico` (writes situations as a stakeholder would; reads only requirements and unit titles), fallback `architect`. The author must **not** read Set A (`claude/cobb/scripts/fixtures/pwq-golden-set-a.json`, ML §3), the S14 report section, this note's §2, or any B material. Better still, if the stakeholder can supply 20 to 30 real recent agent situations (from actual sessions), they replace the same number of synthetic ones; real queries are the validity-critical case.

**Contents:** 126 positives (3 per section: one symptom/error framing, one design-time "should I", one free choice; at least one positive per ≥4-chunk unit targets a late-chunk fact, tail flag as before), and 60 negatives (30 near-domain written to sound like a section, checked against titles only; 30 off-domain). Query register: whatever an agent would type, including short ones (H1 says register is the dominant risk, so the author must not be coached toward A's style). Mechanical 4-gram leak check against own section (`pwq_leakcheck.py`), the probe rule of ML §5 (rewrite only when rank-1 < 0.25), frozen outside the repo, sha256 into the ledger, read by no implementer, scored once by `eval --sealed-once --expect-sha256 <ledger>` after the pre-checks of plan S14 (`gate` first, `--exclude` only ids recorded deleted or created after S13b). Aggregate-only reporting, plus the per-query "returned chunk seq" is **not** to be added to the tool for B2 (a post-D-1 fix may be used on A only).

**Frozen configuration:** `--floor 0.43`, `limit=5`, the current production documents, manifest and retrieval skill; their hashes and the manifest are recorded before B2 is delivered to the scorer. Any production change between now and the B2 run voids B2 (it becomes a different configuration).

**Decision rule (all must hold; binary, no indeterminate band):**

1. Floor-applied positive recall **≥ 116/126** (this makes B plus B2 pooled ≥ 189/210 = 0.90: B's 73 are not discarded, because B is a failed measurement of this same system, not a different one) **and** every section hit by at least one of its three queries.
2. Near-domain false accepts **≤ 6/30**; off-domain false accepts **0/30**; overall at most 6/60 (the same 20% and 10% rates as the approved 3/15 and 3/30).
3. Reported, no threshold: raw top-5 recall, floor-applied recall, MRR, distinct documents, displacement count, tail against head, BM25 like-for-like (unit-only semantic against BM25, reported with its handicap stated), and the pooled B+B2 estimate with Wilson intervals (clustered by section, so optimistic).

**Power (exact binomial, this run):** P(positives ≥116/126): true recall 0.95 → 0.948; 0.92 → 0.573; 0.90 → 0.275; 0.87 → 0.053; 0.85 → 0.013. P(near FA ≤6/30): true rate 0.05 → 0.999; 0.10 → 0.974; 0.20 → 0.607; 0.30 → 0.160. With B's point estimates (0.87 and 0.20) the joint pass probability is about 0.03; with a true 0.92 and 0.10 it is about 0.56. If the stakeholder accepts B2-alone at **≥113/126** instead (the same 0.90 on a fresh instrument), the pass probabilities become 0.996 (0.95), 0.868 (0.92), 0.619 (0.90), 0.227 (0.87), at the cost of discounting B's evidence; I do not recommend it. A third set cannot follow: a failed B2 is final and the finding is "score-only retrieval cannot reach the target on this content".

**What the plan's gate becomes.** AC-3 now reads: the stakeholder-approved thresholds are met by B2 under the rule above (pooled recall 0.90 and the negative clauses at the same rates), with B recorded as FAIL and the report's S14 section unchanged. Plan S14 is rerun as S14b (B2, `qa-engineer` scorer, new author); S15c, S16a-d, G3 and S17 stay held until S14b's verdict.

## 5. Recommendation to the stakeholder

Decide between: **(A, recommended) B2 on the unchanged configuration** under §4, then proceed (pass) or take the deviation with 210 queries of evidence (fail); **(B) accept now with a stated deviation** (§1 item 4), proceeding to S15c and S16 without another seal. Either way `skills/python-web-quirks/` stays until the choice is made, because deletion is the step that removes the whole-skill fallback; git retains it at the pin, so (B) is recoverable. I recommend (A) because the target was approved at 0.90 as a judgment and the point estimate (0.869, interval 0.781 to 0.925) is inside the range where the correct reading is still open; a deviation granted on 84 clustered queries would be argued on a number nobody can sharpen cheaply, and B2 costs one authoring unit and one scoring run. Do not change the floor or any production document before B2, and do not approve any change on the basis of Set A.

## 6. Risks and open questions

- **Limitations of this diagnosis.** B's per-query data is sealed; H1 and H2 are inferred from A and mechanical perturbations of A, which share the 42 sections. The perturbations degrade information as well as style, so they bound the floor's sensitivity, they do not reproduce B's queries. I did not run the BM25 over the full workspace (the listing endpoint caps at 200 documents with no paging, so the non-unit corpus could not be enumerated), so the like-for-like parity is for unit chunks only.
- **Pooling and gate-moving.** The pooled rule keeps B's evidence; lowering a threshold after the failure would be goalpost-moving unless the stakeholder restates it from a cost-of-miss argument first.
- **Seal discipline.** The B2 file path is not named in any brief except its author's and the scorer's; the author works from the manifest's unit titles and the pinned source, not from this note's §2.
- **Tool gaps.** D-1 (document-level matching, no returned-chunk position) still limits any tail/head claim; the `--sealed-once` brief must list `--set-b-path`.
- **Production drift.** New captures or edits to `ws:agent-team` between freezing and scoring change the configuration; the S14 pre-check order and `--exclude` rule apply unchanged.
