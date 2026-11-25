# Connected code graph — P1 close-out: method rulings

> **Status:** active · **Owner:** `data-scientist` · **Tracks:** C-1001 (P1 close-out, CO-9)

Method rulings for `docs/plans/connected-code-graph-p1-closeout.md`; the plan owns design and sequence. Sources read: `docs/plans/connected-code-graph-p1-gate-marker-ml.md`, `docs/plans/connected-code-graph-eval-protocol-ml.md` (EP) §2.1, §5.2, §11, `docs/reviews/connected-code-graph-p1-closeout-co2.md` (M3, M5), `docs/test-reports/connected-code-graph-p1-closeout-co1-report.md`, `docs/requirements/connected-code-graph.md` AC-5, `cpg/eval/results/co2c/summary.json`, `cpg/eval/dev/ac5-scripts.jsonl` (all 49 dev script items have a one-hop writer chain). No held-out data and no custodian directory was read; every held-out figure below is a count.

## Recall of a distance-2 hit through a prose mention

**Ruling in brief.** (1) By the letter, a distance-2 hit **is** a gate hit; the claim it supports is not "the resolver finds the link", so the gate reports it separately and does not credit it as resolver recall. (2) CO-9 reports, beside the scripts hit count, a concordance split, a label-free burden block and a label-free ceiling on the wrong-`possible` ratio, and the EP §11 labelled estimate wherever a labeller may see items. (3) No cap on what is listed or scored; a pre-registered **credit window** and ordering rule is reported next to the hit count, never in place of it. One stakeholder question (end of section).

### 1. What counts as a hit

- AC-5: every item "appears in the answer **individually**, with its evidence, as confirmed or possible".
- EP §5.2: a hit needs H1 (the impacted element), H2 (`confirmed` or `possible`), H3 (a non-empty chain from the asked anchor ending at *x*) and H4 (individual). "What does not matter: the distance the system reports; direct vs indirect; whether the system's chain equals the writer's."
- The gold records the writer's chain, one inter-function hop for all 49 scripts items (`hops = len(chain)`, EP §2.1).

Under that text a gold item found at distance 2, or through a Markdown mention edge, **is a hit**: the gate verdict does not change, and this note does not amend EP §5.2. The reading is the settled one and keeping it keeps the held-out list's rule fixed before contact.

What the text does not settle is the claim. For a distance-2 item the answer's chain is anchor, then the Markdown caller, then the item as a callee of that caller. The gold relationship (item calls anchor) is not what the chain says. The item is found because the same prose file mentions it, so it is a coincidence the resolver did not name (CO-2 M3; the ledger's word). A scripts pass built on such hits would measure fan-out, not link recovery, and would not transfer to a held-out item whose prose caller is absent.

**Definition (mechanical, counts only).** A hit is **concordant** when some qualifying *x* has answer distance equal to the gold's `hops`; for scripts that means distance 1. A hit is **coincident** when every qualifying *x* of the item has distance greater than the gold's `hops`. Distance and `hops` are both in the harness's records; no label is needed. Dev baseline from CO-2c: 34 hits, 30 concordant, 4 coincident (G 1, H 1, I 2), all `possible`.

**Reporting rule.** The scripts line reads `H_scripts = hit (concordant c, coincident k)`. The verdict follows EP §12.1 unchanged. The sentence "the scripts resolver recovers cross-script links" may be written only over concordant hits; coincident ones are reported as "listed through a prose caller".

### 2. What is reported beside the scripts hit count (CO-9)

All three blocks are per stratum, scripts separate from http and mcp, and pooled. Pooled Q is a ratio of sums, so a 124-item answer moves it more than six small answers; the scripts-excluded pool is reported beside the full pool.

**A. Label-free burden block** (counts; the custodian can report it for held-out):
- listed items, total and per answer: median, p90, max; the p90 index is `-(-9 * n // 10)` (EP §11.3);
- listed `possible` and `confirmed` by distance band 1, 2, ≥ 3;
- the same counts restricted to items whose chain contains a `prompt-mention` hop;
- items per credited hit, `J = listed / hits` (dev: 610 / 34 ≈ 18, 610 / 49 ≈ 12 against gold; both computed from the review's figures, to be recomputed by the harness);
- the share of listed items held by the largest answer, and the number of answers over 100 items.

**B. Label-free ceiling on Q.** `Q_ceiling = (listed − hit-matched items) / (hit-matched items)`, treating every listed item that is not a matched gold item as wrong. This is an upper bound, not an estimate: the gold is incomplete, so unmatched items may be real. It is reported under that name and never as noise. On the dev figures it is about 16.9 (`(610 − 34) / 34`), more than four times the AC-5 bound of 4; so counts alone cannot show the bound is met, and the scripts claim "noise within bound" stays **unmade** until C produces it.

**C. EP §11 labelled estimate** (`Ŵ`, `R̂`, `RC`, `Q`, per band), with scripts as its own stratum in the sample frame (strata *h* = reason code × distance band × domain), allocation and seed as EP §11.2. Dev: a labeller who is not a pack author labels the sample on the dev answers at the CO-9 rehearsal; this is the only place a figure for the distance-2 prose rows can be produced before the spend, and the rehearsal must not skip it. Held-out: the labelling rule of EP §11 is unchanged; if labelling of held-out answers is withheld from every human, only A and B are reported and the noise bound is recorded as **not measured**, not as met. The custodian's report carries `n_h`, `f_h`, `t_h`, `N_h` and the per-band Q with its answer-level bootstrap interval, never an item id.

### 3. Cap and ordering rule (pre-registrable)

- **No cap on the answer and no cap on what is scored.** A cap that removes a listed gold item turns a hit into a miss by construction, changes what AC-5 says ("every one appears"), and cannot be tuned after contact.
- **Credit window, reported not gated.** Order each answer by (1) `confirmed` before `possible`; (2) distance ascending; (3) chains without a `prompt-mention` hop before those with one; (4) `itemId` ascending (a fixed total order). `hit@100` counts a hit only if a qualifying *x* lies in the first 100 items; 100 is the ratified p90 burden bound (`ccg-noise-ratify-v1`, 4:1 cap, p90 ≤ 100). The harness reports `hit` and `hit@100` as two numbers; the verdict uses `hit`.
- The ordering is also a ranking ask for the product: it lists the evidence-bearing rows first. The ordering is a harness rule only; the answer's own order is not changed by this note.

### Recommendation and falsifiable condition

Recommend: keep EP §5.2 as is; adopt the concordant/coincident split, blocks A to C and the window as pre-registered CO-9 reporting before the held-out list is spent; do not credit scripts resolver recall from coincident hits.

The recommendation is **refuted** if on the CO-9 dev rehearsal, with labels, either (a) the labelled `Q_scripts` over the 22 dev script answers exceeds 4 (the resolver's added `possible` edges fail the ratified bound regardless of recall), or (b) removing the coincident hits from the dev set leaves concordant recall below the built-family result of 30 of 30, i.e. a built family loses a hit. The scripts claim at the gate is **falsified** on held-out if any scripts item is a coincident hit only, or `Q_scripts` (C) exceeds 4, or is unmeasured while B exceeds 4.

### Stakeholder question (scope / AC meaning)

Should the AC-5 scripts verdict itself credit only concordant hits (a coincident-only item then counts as a miss), instead of the letter of EP §5.2? Recommendation: **no** for this gate: keep the letter, report the split, and state the scripts claim over concordant hits. Reason: a stricter hit definition adopted after the dev run is chosen with its result in view and the held-out list has not been spent, so it can be pre-registered, but it overturns a ratified EP rule and the AC's "appears in the answer"; that is the stakeholder's decision. If the stakeholder prefers strict, the dev set already shows 4 coincident items, so scripts could not meet "every item" on dev without them being removed or the resolver extended.
