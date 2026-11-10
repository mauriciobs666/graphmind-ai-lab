# python-web-quirks on the knowledge base — another embedder, chunking, and the B3 design

> **Status:** active · **Owner:** `data-scientist` · **Tracks:** — (U35; **Extends:** `docs/plans/python-web-quirks-kb-b2-ml.md`; relies on `docs/plans/python-web-quirks-kb-ml.md` §6)

## 1. Question and recommendation

**Decision served.** After sealed B (73/84) and B2 (110/126, near-domain 9/30) both failed at `text-embedding-qwen3-embedding-0.6b`, the stakeholder asks whether to test another embedding model, and whether the chunking is at fault. The caller needs: spend a B3 or stop.

**Recommendation.**

1. **Chunking is not a plausible lever (§5): do not put it in B3.** Six chunk variants, two embedders, no variant beats the base on Set A beyond noise.
2. **The only candidate worth a fresh seal is `text-embedding-qwen3-embedding-4b`** (same family, same query prefix). On non-sealed data it is the one model with a positive separation from near-domain negatives (§4) and the one that holds rank under degraded queries (raw top-5 on second-half queries 38 of 42 against 29). Granite-278m and nomic-v1.5 are rejected on the same data.
3. **My honest estimate that 4b passes a B3 under the B2 rule is about 0.1 to 0.3 (point estimate 0.15), a judgment, not a measurement (§6).** A 0.6b B3 would pass with about 0.03 (b2 note §4). B3 costs: one authoring unit, one scoring unit, a staging workspace with 4b embeddings on `:8210`, no production touch. A pass would still owe a production migration with its own blast radius (§3).
4. **If the stakeholder will not spend that, stop (§7):** take the deviation as a measurement. Pooled floor-applied recall 183/210 = 0.871 (Wilson 95% 0.819 to 0.910), raw top-5 196/210 = 0.933 (0.891 to 0.960), near-domain false accepts 12/45 = 0.267 (0.160 to 0.410), off-domain 0/45. The near-domain arm is the binding failure, and the embedder swap is the only remedy with any offline signal against it.
5. **Do not re-score B or B2 with a new model as a gate (§6).** It is not clean; the cheaper and valid instrument is B3.

I recommend **try 4b once, via B3, if the stakeholder values a pass more than one more authoring-and-scoring unit; otherwise stop.** I do not recommend anything between (no bundled variants, no second candidate).

## 2. What is available

**Local LM Studio** (`GET http://localhost:1234/v1/models`, read-only, this run) lists four embedding models: `text-embedding-qwen3-embedding-0.6b` (in use), `text-embedding-qwen3-embedding-4b`, `text-embedding-granite-embedding-278m-multilingual`, `text-embedding-nomic-embed-text-v1.5`; the other 20 entries are chat LLMs. Dimensions observed from embedding calls: 1024 (0.6b), 2560 (4b), 768 (granite, nomic). Embedding with the other three loaded them in LM Studio (JIT); nothing else changed.

**In use now:** `lmstudio/text-embedding-qwen3-embedding-0.6b` (`falkor-chat/config/models.json` `defaults.embedding`, `dim` 1024), documents embedded raw, queries built by the retrieval skill with the Qwen3 `Instruct:`/`Query:` prefix (`skills/agent-kb-retrieval/SKILL.md`).

**model-bench** (`model-bench/results/index.csv`, pack `embedder-graphrag-retrieval` 1.1.0, n = 38 queries): recall@5 / MRR: qwen3-0.6b (Q8_0) 34/38, 0.6278; qwen3-4b (Q4_K_M) 35/38, 0.6610; granite-278m 34/38, 0.6317; nomic-v1.5 32/38, 0.6407; BM25 29/38, 0.5734. p50 latency 24 ms (0.6b, last run 22 ms) against 37 ms (4b). **These differences are one to three queries on 38: they cannot rank the candidates and they do not measure this task** (short GraphRAG corpus, not 129 dense same-register chunks). I use them only to say nothing there contradicts the §4 ordering.

## 3. How a swap works in falkor-chat, and its blast radius

Facts read from `falkor-chat/docs/SERVER.md` §1.8 and `docs/DESIGN.md` §7.1, and observed read-only in FalkorDB this run.

- **Two levels.** The default embedding model is server-wide (`config/models.json`); a per-workspace `WorkspaceConfig.embeddingModelOverride` is a **hard cap** that wins over it. Observed overrides: `ws:agent-team`, `ws:pwq-test`, `ws:demo`, `ws:k069probe` are all pinned to the 0.6b; `ws:acme`, `ws:eval`, `ws:nlq-eval` have **no** override and follow the default.
- **Therefore** editing the server default would change the model for `acme`/`eval`/`nlq-eval` (their vectors stay 0.6b: a model/dimension mismatch, silent ranking garbage or `EmbeddingDimensionError`) and leave the pinned workspaces alone. The supported route is the **per-workspace embedding migration** (`scripts/migrate_embeddings.sh <ws> <ref>`): it needs the serving process **stopped**, re-embeds every Message then Chunk, drops and recreates both vector indexes at the target dimension, sets the override, and the process must be restarted.
- **Dimension.** LM Studio returned 2560 values for 4b even when the request carried `"dimensions": 1024` (observed), so DESIGN's "same 1024-dim MRL, dimension stays 1024" upgrade path does not hold through LM Studio without server-side truncation. A 4b workspace is 2560-dim: the overlay needs `dim: 2560` and the migration rebuilds the indexes (about 10 KB per vector against 4 KB; irrelevant at this size).
- **Re-embedding cost.** `ws:agent-team` holds 882 chunks and 0 messages. I embedded the 902 chunks of `ws:pwq-test` with the 4b through LM Studio in about 114 s (0.6b: 18 s). Compute is trivial; the cost is the outage and the validation.
- **Blast radius of migrating production `ws:agent-team` (`:8200`), flagged, not performed.** (a) The server stops: **every** workspace on that process, including `ws:demo` (12,106 messages) and the `salesperson` agent, is down for the migration and restart; (b) every KB consumer of `ws:agent-team` (all agents using `agent-kb-retrieval`) gets 4b scores, and the **0.43 floor is a skill-wide constant calibrated for the 0.6b**: it must be re-derived and the Stage 8 45-pair regression re-run on the whole corpus, not just this feature (`ML` §6.3); (c) the query prefix is unchanged (same family) so the skill changes only its floor; (d) rollback is another migration. None of this happens before a B3 pass and a stakeholder decision.
- **Chunker is server-wide.** `services.py:1309` calls `chunking.split_into_chunks(text)` with the hard-coded defaults (1000 / 150); there is no per-workspace or per-call knob. A chunking change is a code change to the server for every workspace.

## 4. Offline experiment (non-sealed data only)

**Data.** The 129 unit chunks plus 773 other chunks of `ws:pwq-test` (read-only Cypher), the 42 Set A positives in four forms (full, first half, second half, every third word dropped; the perturbation of the b2 note §2) and the 14 Set A negatives (7 near-domain, 7 off-domain). Embedded through LM Studio with each model's own convention: qwen3 models with the skill's query prefix and raw documents; granite with none; nomic twice (with `search_query:`/`search_document:` task prefixes, and with none, which is what the server would send as it stands). Cosine distance computed offline; metrics as in `claude/cobb/scripts/pwq_kb.py` (document-level hit in the top five, all-workspace). **Validation:** for the 0.6b the offline numbers equal the b2 note and the server (raw top-5 42/39/29/40 all-workspace; best positive 0.417; near negatives 0.424, 0.446, ...), and the offline chunker reproduces the 129 stored chunks exactly (36 of 36 units identical). Not sealed data was used; I did not read `pwq-set-b*.json`.

**Floor without tuning on a sealed set.** A model's distances live on their own scale (granite and nomic compress them), so 0.43 does not transfer. Rule fixed here, derived from Set A only and with one constant inherited from production: **floor = 1.03 x (largest best-unit score among the 42 full-form Set A positives).** The 1.03 is production's own headroom (0.43 over 0.417, ML §6). The rule is computed once per candidate before any seal exists, recorded in the ledger, and never revisited. For the 0.6b it returns 0.430 (the production value).

| candidate | max positive | floor (rule) | min near-neg | margin (near min − pos max) | AUC pos vs near, full / first / second / drop3 | near FA at floor (of 7) / off FA (of 14) | raw top-5, all-workspace, full / first / second / drop3 | floor-applied hits |
|---|---|---|---|---|---|---|---|---|
| qwen3-0.6b (control) | 0.417 | 0.430 | 0.380 | −0.037 | 0.986 / 0.908 / 0.663 / 0.888 | 2 / 0 | 42 / 39 / 29 / 40 | 42 / 36 / 22 / 33 |
| **qwen3-4b** | 0.408 | 0.420 | 0.422 | **+0.014** | **1.000 / 0.963 / 0.891 / 0.956** | **0 / 0** | 42 / 41 / 38 / 41 | 42 / 37 / 26 / 36 |
| qwen3-4b, client-side 1024 truncation (not available through LM Studio) | 0.385 | 0.397 | 0.389 | +0.004 | 1.000 / 0.946 / 0.844 / 0.925 | 1 / 0 | 42 / 39 / 38 / 40 | 42 / 35 / 27 / 33 |
| granite-278m | 0.324 | 0.333 | 0.274 | −0.049 | 0.939 / 0.806 / 0.585 / 0.844 | 5 / 0 | 42 / 40 / 30 / 39 | 42 / 37 / 27 / 38 |
| nomic-v1.5, task prefixes | 0.334 | 0.344 | 0.254 | −0.080 | 0.939 / 0.772 / 0.748 / 0.799 | 7 / 5 | 39 / 35 / 28 / 33 | 39 / 35 / 28 / 33 |
| nomic-v1.5, no prefix | 0.364 | 0.375 | 0.300 | −0.064 | 0.935 / 0.656 / 0.554 / 0.687 | 5 / 1 | 39 / 38 / 25 / 34 | 39 / 37 / 25 / 31 |

**Reading.**

- **Granite and nomic are out:** a negative margin deeper than the 0.6b's, near-domain negatives mostly accepted at their rule floor, nomic also accepts off-domain queries and misses 3 of 42 full-form positives raw. No prefix choice rescues nomic.
- **4b is the only candidate that separates.** AUC of best-unit score between positives and the 7 near negatives is 1.000 on the full form (0.6b 0.986) and stays 0.89 to 0.96 on perturbed forms (0.6b 0.66 to 0.91). Its raw top-5 holds under degraded queries (38 of 42 on second-half against 29; the b2 note's H1 said register is the dominant risk). The perturbed-form floor-applied hits are 99 of 126 against 91; paired, 15 queries only 4b hits, 7 only 0.6b hits (exact McNemar p = 0.134, and the 126 variants share 42 sections, so even that is optimistic). **Not significant, not nothing.**
- **The margin is thin and the evidence small.** +0.014 on a 0.408 positive ceiling is the 0.6b's own S12 situation (0.0125 headroom) in better light. The near-domain arm rests on **7 negatives**: 0 of 7 against 2 of 7 is Fisher p about 0.46. Set A is saturated at full form for every model (42/42 raw), so the signal is in the perturbed forms and the negatives, both weak. The 4b quantisation here is Q4_K_M (model-bench lists the same); the 0.6b is Q8_0.
- **Selection.** I looked at 5 embedder configurations and 6 chunk variants x 2 models on the same 42 sections; the 4b is the maximum of those. Its Set A advantage is optimistic by construction (winner's curse); §6 prices that in.

## 5. Chunking as a cause

**Facts verified.** One document per unit; the chunker is the generic 1000/150, paragraph then sentence boundaries (`falkor-chat/server/falkorchat/chunking.py`); 129 chunks over 36 units; 12 units have 4 or more chunks (u26 has 10, u21 has 9; chunk counts sorted: 10, 9, 7, 7, 6, 6, 5, 5, 4, 4, 4, 4, then 3 x 12, 2 x 10, 1 x 2); document length median 1,724 chars, max 6,554; only chunk 0 carries the document title line, a later chunk carries no title or heading.

**Is it plausible as the cause of the B/B2 failures? Weakly, and the data cannot show it.** Three arguments against: (1) tail probes (late-fact queries) scored higher than head probes in both sealed sets (B 13/13 against 60/71; B2 0.9286 against 0.8661), the opposite of what a missing-late-chunk-context story predicts (document-level matching, D-1, so this says nothing about which chunk returned); (2) the dominant losses are the floor (B: 4, B2: 9 of the lost positives, with raw recall 0.917 and 0.944) and near-domain acceptance, both properties of absolute score level, not of how a document is cut; (3) on Set A the best-scoring chunk is chunk 0 for 13 of 42 positives (b2 note H4). Not testable: whether a near-domain false accept lands on a late chunk (the tool does not record the returned chunk).

**Variants tested offline** (Set A, all-workspace competition from the 773 non-unit chunks of `ws:pwq-test`, floor from the §4 rule recomputed per variant). Raw top-5 / floor-applied hits over (full, first, second, drop3), near false accepts of 7 at the variant's rule floor:

| variant (chunks) | 0.6b raw | 0.6b floor-hits | 0.6b near FA | 4b raw | 4b floor-hits | 4b near FA |
|---|---|---|---|---|---|---|
| base 1000/150 (129) | 42/39/29/40 | 42/36/22/33 | 2 | 42/41/38/41 | 42/37/26/36 | 0 |
| size 2000 (64, 2 units with 4+) | 42/41/30/40 | 42/38/21/35 | 3 | 42/41/35/41 | 42/36/24/34 | 0 |
| size 4000 (40, 0 units with 4+) | 42/41/28/40 | 42/36/18/33 | 2 | 42/41/35/40 | 42/37/22/32 | 1 |
| whole document, one chunk per unit (36) | 42/40/28/39 | 42/34/17/32 | 2 | 42/40/34/39 | 42/37/32/37 | 1 |
| base + title line on every chunk (129) | 42/41/29/39 | 42/39/21/35 | 1 | 42/42/37/40 | 42/36/27/36 | 0 |
| paragraph-packed 1000 + title (115, 8 units with 4+) | 42/41/30/40 | 42/38/20/36 | 2 | 42/42/38/40 | 42/36/28/34 | 0 |

**Reading.** Making units 1 to 2 chunks (size 2000, 4000, whole document) does not raise the 0.6b's recall and lowers its perturbed floor-applied hits (second-half 22 down to between 17 and 21); for the 4b it lowers raw recall on second-half (38 down to between 34 and 35). Whole-document embedding also pushes the 4b's rule floor to 0.500 (positives' best scores rise), which is the wrong direction for negatives. Title-on-every-chunk and paragraph-packing move floor-applied hits by −1 to +3 of 42 per form with no consistent sign across the two models: noise at this size, as the b2 note found (+1 to +2 of 168). Document texts above about 2,000 tokens are truncated by LM Studio at the 2048-token context, a further cost of whole-document embedding on the longest units (max 6,554 characters; I did not measure tokens). **Conclusion: chunking is not a lever I can show, and none of its variants deserves a place next to the embedder in B3.** Putting two changes in one B3 would also make a pass or fail unattributable.

**Cost if someone still wants one.** Offline: nothing more to test on A (it is saturated). Re-ingesting 36 units through the tool on `:8210` only (the b2 note's S-series pattern): chunk text is fixed by the server chunker, so variants that change chunk size are a server code change (server-wide for every workspace, with a re-ingest of every document that should follow it); title-on-every-chunk can only be done by editing the stored text (re-ingest of 36 production documents, against the "no replace/delete" safety posture). Blast radius for a production chunker change: every workspace's future ingests, plus a re-chunk of existing documents.

## 6. Selection-bias rule and the B3 design

**Spent instruments.** Sets B and B2 are spent: their per-query outcomes under the 0.6b determined the failure, and their aggregates have driven every hypothesis since (H1 to H5, §4 here). I have not read them and will not.

**Re-scoring B and B2 with a new model: informative as a paired descriptive comparison, contaminated as a gate.** The model never saw the queries, and a paired comparison on 210 queries would be far more powerful than any fresh set. But: (1) the swap was motivated by the sets' failure under the 0.6b, so a pass is a second look at the same queries after seeing the outcome, the classic reuse that the sealed protocol exists to prevent; (2) the sentinel files and the tool (`--sealed-once`, production target) would have to be bypassed or changed; (3) it would require the 4b-embedded copy of the corpus anyway, which B3 also needs. **Rule: no verdict may rest on re-scoring B or B2; if the stakeholder wants the paired aggregate, it is a one-off by the scorer, aggregates only, labelled "descriptive, not a gate", and it does not replace B3.** This is the one point I leave to the stakeholder (reopening the queries to a tool run is theirs to authorize).

**B3 specification.**

- **Author:** a third agent, different from `data-scientist` (A), `qa-engineer` (B) and the B2 author (named in the coordination ledger, which I did not need to read). It reads no Set A, no S14/S14b section, no part of this note's §4/§5, no B/B2 material; it sees the unit titles and the pinned source only. Real situations from actual recent agent sessions (20 to 30) preferred where the stakeholder can supply them.
- **Contents:** identical shape to B2 (126 positives, 3 per section, tail probes, 4-gram leak check; 30 near-domain and 30 off-domain negatives), frozen outside the repo with sha256 in the ledger, scored once, aggregates only. Near-domain negatives are the arm to invest in: pooled across B and B2 the 0.6b accepted 12/45 = 0.267, and a 30-item arm is what the rule uses.
- **Configuration under test (frozen and recorded before delivery):** `text-embedding-qwen3-embedding-4b` at 2560 dimensions, the Qwen3 query prefix unchanged, **chunking unchanged**, floor **0.420** (the §4 rule's value; recomputed by the same command in the staging workspace before delivery and recorded; it uses no sealed data), top-K 5, the 36 current unit documents plus the competing corpus of `ws:agent-team` re-embedded with the 4b, in a **staging workspace on `:8210`** (not production). Production stays untouched until a pass.
- **Decision rule (binary, fixed now, identical to the b2 note §4):** floor-applied recall at least 116/126 and every section hit; near-domain false accepts at most 6/30; off-domain 0/30; overall at most 6/60. It is the same rule as B2, so the 4b is held to what the 0.6b failed. A fail is final: no B4 for this feature; the finding becomes "score-only retrieval cannot reach the target on this content with the local embedders", and §7's stop applies.
- **Honest probability of passing.** Exact binomial: P(positives at least 116/126) at true recall 0.87 / 0.90 / 0.92 / 0.94 = 0.053 / 0.275 / 0.573 / 0.864; P(near false accepts at most 6/30) at true rate 0.27 / 0.20 / 0.15 / 0.10 = 0.274 / 0.607 / 0.847 / 0.974. The 0.6b pooled is 0.871 and 0.267 (joint pass about 0.015 to 0.03). If the 4b's perturbed-A gain (99 against 91 of 126, +6 points) transferred fully, true recall would be about 0.92 and the near rate about 0.13: joint about 0.5. That is the optimistic bound; the winner's curse and the thin 7-negative evidence justify halving the transfer (recall about 0.89 to 0.90, near about 0.18 to 0.20): joint about 0.1 to 0.2. **I state 0.1 to 0.3, point 0.15. These are estimates, not measurements; the measurement that replaces them is B3.**
- **Multiple-comparisons budget.** The offline phase made 5 embedder configurations and 6 chunk variants x 2 models of looks on A; the choice of one embedder and no chunk change is made from that and recorded here, so it costs the B3 nothing statistically (B3 is one pre-specified test) but it caps the claim: a B3 pass would be evidence that the 4b clears the rule, not that it was the best candidate. Budget going forward: **one configuration, one test, one seal.** No second embedder, no floor adjustment and no chunk variant may be added between this note and the B3 scoring; any such change voids B3 (it becomes a different configuration).

## 7. Recommendation and what a stop looks like

**Recommended:** run B3 on the 4b once, if the stakeholder wants a pass badly enough to fund an authoring unit, a scoring unit and a staging workspace; costs: no production change, compute minutes, the staging corpus built by re-ingest through the tool on `:8210`, and (devops/graph-dba) the staging embedder override and index at 2560. Expect a fail more likely than not.

**If B3 passes:** a separate decision unit, not part of this note: per-workspace migration of `ws:agent-team` (§3 blast radius), the skill-wide floor re-derivation and the 45-pair regression, the skill edit (floor 0.420), `S15c`/`S16`.

**A stop looks like** (the stakeholder's, as stated in the b2 note §1 item 4, updated with B2): (1) AC-3 recorded as FAIL on B and B2, the 4b offline result recorded as a hypothesis with its caveats; (2) a stated deviation from the measurements in §1 item 4 here (pooled 0.871 floor-applied, 0.933 raw, near-domain 12/45, off-domain 0/45), with FR-9/FR-11 as the fallback for a miss; (3) `skills/python-web-quirks/` kept or retired by the stakeholder on a cost-of-miss argument; (4) no floor, chunking or embedder change in production; the query-writing guidance in the retrieval skill (b2 note §3) proceeds as a non-gated improvement; (5) the §4 experiment closed with the 4b noted as the first candidate if the question returns.

## 8. Risks and open questions

- **Set A is saturated and the 7 near negatives are few**: the 4b's offline advantage can be a small-sample artifact. It was selected as the best of several looks.
- **The perturbations degrade information as well as style** (b2 note §6); AUC on the perturbed forms bounds sensitivity, it does not reproduce B/B2 queries.
- **Staging fidelity:** the staging workspace must carry the same competing corpus as production; `ws:agent-team` (882 chunks) and `ws:pwq-test` (902) differ, and the gate and manifest ids are per target. Ownership (`devops`/`graph-dba` for the staging workspace and override; `qa-engineer` as scorer) is the architect's/coordinator's to assign.
- **Skill-wide floor:** a 4b production swap invalidates the 0.43 for every KB consumer; the 45-pair regression is owed then, not now.
- **Stakeholder fork (the only one):** whether a paired, descriptive, aggregate-only re-score of B and B2 with the 4b is authorized (§6); my recommendation is not to use it as evidence for a verdict.
- **Not measured:** 4b token-length truncation at LM Studio's 2048 context for the longest units; per-query latency of 4b in production (model-bench p50 37 ms against 24 ms is the only figure); the other workspaces' (`acme`, `eval`, `nlq-eval`) retrieval behavior.
